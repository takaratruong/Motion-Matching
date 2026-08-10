import ctypes
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from .database import read_holden_database, write_holden_database
from .schema import (
    ArtifactSet,
    FeatureSet,
    G1_SKELETON_NAMES,
    G1_SKELETON_PARENTS,
    G1_SKELETON_SIGNATURE,
)


MAGIC = b"G1TF"
VERSION = 1
DIMS = 4
SUPPORT_MAGIC = b"G1SP"
SUPPORT_DIMS = 3
SUPPORT_COLUMNS = (
    "source_root_height_m",
    "source_left_toe_height_m",
    "source_right_toe_height_m",
)
WALKABILITY_MAGIC = b"G1WM"
_HEADER = struct.Struct("<4sIII")
_UINT32_MAX = (1 << 32) - 1
_FLOAT32_BYTES = np.dtype("<f4").itemsize
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SCENE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_AT_FDCWD = -100
_RENAME_EXCHANGE = 2

MOTION_MANIFEST_KEYS = {
    "schema", "output_fps", "feature_dimensions", "terrain_dimensions",
    "support_dimensions", "terrain_feature_distances_m", "total_clips",
    "grail_clips", "skipped_clips", "database_frames", "diagnostic_mode",
    "sources", "skeleton", "contact", "surface", "database", "sidecars",
    "scene_index", "validation_file", "validation",
}
MANIFEST_BASE_KEYS = MOTION_MANIFEST_KEYS - {
    "database", "sidecars", "scene_index", "validation_file",
}
VALIDATION_KEYS = {
    "schema", "duration_error_s", "fk_max_error_m",
    "quaternion_norm_max_error",
}
INDEX_KEYS = {
    "schema", "default_scene_id", "scene_ids", "scenes",
    "coordinate_signature", "surface_signature",
}
ASSET_DESCRIPTOR_KEYS = {
    "heightfield": {
        "path", "schema", "version", "nx", "nz", "origin_x",
        "origin_z", "cell_size_m", "exterior_height_m", "interpolation",
        "diagonal", "sha256",
    },
    "mesh": {"path", "schema", "sha256"},
    "walkability": {
        "path", "schema", "version", "nx", "nz", "classes", "sha256",
    },
}
COORDINATE_SIGNATURE = "holden-y-up-right-handed-forward-plus-z"
_FLAT_SOURCE_KEYS = {
    "name", "terrain_id", "path", "sha256", "receipt_path",
    "receipt_sha256", "receipt_schema", "receipt_status", "source_fps",
    "source_frames", "output_frames", "left_source_index",
    "right_source_index", "source_alpha",
}
_FLAT_VALIDATION_KEYS = {
    "fk_max_error_m", "duration_error_s", "quaternion_norm_max_error",
}
_FLAT_SCHEMA = "g1-lmm-flat-data/v3"
_FLAT_KINEMATICS_MODEL = {
    "asset": "g1_29dof.xml",
    "sha256": "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376",
    "size_bytes": 26914,
}
_LMM_TERRAIN_SCHEMA = "g1-lmm-terrain-data/v1"
_LMM_TERRAIN_MANIFEST_KEYS = {
    "schema", "output_fps", "database_frames", "source_count",
    "range_count", "trajectory_horizons", "dimensions", "skeleton",
    "feature", "terrain", "support", "contact", "time_filters", "inputs",
    "slope_source_receipt", "sources", "ranges", "continuity",
    "temporal_partition", "database", "features", "sidecars",
    "scene_index", "validation",
}
_LMM_TERRAIN_DYNAMIC_KEYS = {
    "database", "features", "sidecars", "scene_index", "validation",
}
_LMM_TERRAIN_BASE_KEYS = (
    _LMM_TERRAIN_MANIFEST_KEYS - _LMM_TERRAIN_DYNAMIC_KEYS)
_LMM_TERRAIN_VALIDATION_KEYS = {
    "schema", "database", "features", "sidecars", "partition", "inputs",
}
_LMM_TERRAIN_RECEIPT_KEYS = {
    "schema", "status", "clip_id", "hashes", "metadata", "source_fps",
    "target_fps", "source_frames", "provisional_output_frames",
    "admitted_output_frames", "source_span_s", "output_span_s",
    "output_shortfall_s", "rejected_output_ranges", "admitted_output_range",
    "intended_nonflat_provisional_range", "intended_nonflat_admitted_range",
    "interpolation", "basis", "object_rotation_binary32",
    "object_translation_binary32", "inverse_round_trip_max_error_m",
    "exterior_policy", "support_calibration_m",
    "support_calibration_applied_to", "fk_max_error_m",
    "quaternion_norm_max_error",
}
_LMM_TERRAIN_SCENE_KEYS = {
    "schema", "id", "label", "provenance", "coordinate_signature",
    "surface_signature", "terrain_feature_distances_m", "heightfield",
    "mesh", "walkability", "bounds", "spawn", "regions", "routes",
}
_LMM_TERRAIN_INDEX_KEYS = {
    "schema", "default_scene_id", "scene_ids", "scenes",
    "coordinate_signature", "surface_signature",
}


def features_bytes(features: FeatureSet) -> bytes:
    if not isinstance(features, FeatureSet):
        raise TypeError("features must be a FeatureSet")
    features.validate()
    values = np.ascontiguousarray(features.values, dtype="<f4")
    offset = np.ascontiguousarray(features.offset, dtype="<f4")
    scale = np.ascontiguousarray(features.scale, dtype="<f4")
    return b"".join((
        struct.pack("<II", *values.shape), values.tobytes(order="C"),
        struct.pack("<I", len(offset)), offset.tobytes(order="C"),
        struct.pack("<I", len(scale)), scale.tobytes(order="C"),
    ))


def write_features(path: os.PathLike | str, features: FeatureSet) -> None:
    with open(path, "wb") as stream:
        stream.write(features_bytes(features))


def read_features(path: os.PathLike | str) -> FeatureSet:
    with open(path, "rb") as stream:
        def read_exact(size, label):
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError(f"{path}: truncated {label}")
            return payload

        rows, columns = struct.unpack("<II", read_exact(8, "feature header"))
        if rows < 1 or columns != 31:
            raise ValueError(f"{path}: invalid feature dimensions")
        values = np.frombuffer(
            read_exact(rows * columns * 4, "feature payload"),
            dtype="<f4",
        ).reshape(rows, columns).copy()
        offset_count, = struct.unpack("<I", read_exact(4, "offset header"))
        if offset_count != columns:
            raise ValueError(f"{path}: invalid feature offset dimensions")
        offset = np.frombuffer(
            read_exact(columns * 4, "offset payload"), dtype="<f4").copy()
        scale_count, = struct.unpack("<I", read_exact(4, "scale header"))
        if scale_count != columns:
            raise ValueError(f"{path}: invalid feature scale dimensions")
        scale = np.frombuffer(
            read_exact(columns * 4, "scale payload"), dtype="<f4").copy()
        if stream.read(1):
            raise ValueError(f"{path}: trailing feature bytes")
    result = FeatureSet(values, offset, scale)
    result.validate()
    return result


def _authenticate_flat_manifest_inputs(manifest_base) -> None:
    from .sources import load_retarget_npz

    if type(manifest_base) is not dict \
            or manifest_base.get("schema") != _FLAT_SCHEMA:
        raise ValueError("flat manifest schema must be g1-lmm-flat-data/v3")
    model = manifest_base.get("kinematics_model")
    if type(model) is not dict \
            or set(model) != set(_FLAT_KINEMATICS_MODEL) \
            or type(model.get("asset")) is not str \
            or type(model.get("sha256")) is not str \
            or type(model.get("size_bytes")) is not int \
            or model != _FLAT_KINEMATICS_MODEL:
        raise ValueError("flat kinematics model descriptor changed")
    validation = manifest_base.get("validation")
    if type(validation) is not dict \
            or set(validation) != _FLAT_VALIDATION_KEYS:
        raise ValueError("flat validation receipt keys changed")
    for key in _FLAT_VALIDATION_KEYS:
        value = validation[key]
        if type(value) is not float \
                or not np.isfinite(value) or value < 0.0:
            raise ValueError(f"flat validation {key} is invalid")
    if validation["fk_max_error_m"] > 1e-5:
        raise ValueError("flat validation FK error exceeds 1e-5 m")
    if validation["quaternion_norm_max_error"] > 1e-4:
        raise ValueError("flat validation quaternion error exceeds 1e-4")
    output_fps = manifest_base.get("output_fps")
    if type(output_fps) is not float or output_fps != 60.0 \
            or validation["duration_error_s"] > 1.0 / 60.0:
        raise ValueError("flat validation duration/rate receipt is invalid")

    sources = manifest_base.get("sources")
    if type(sources) is not list or len(sources) != 1:
        raise ValueError("flat manifest must authenticate exactly one source")
    source = sources[0]
    if type(source) is not dict or set(source) != _FLAT_SOURCE_KEYS:
        raise ValueError("flat source receipt keys changed")
    if type(source["name"]) is not str or not source["name"] \
            or source["terrain_id"] != "flat" \
            or source["receipt_schema"] \
            != "native-g1-pfnn-sample-retarget/v1" \
            or source["receipt_status"] != "accepted" \
            or type(source["source_fps"]) is not float \
            or source["source_fps"] != 120.0 \
            or type(source["source_frames"]) is not int \
            or type(source["output_frames"]) is not int:
        raise ValueError("flat source identity/type receipt changed")
    for key in ("path", "receipt_path", "sha256", "receipt_sha256"):
        if type(source[key]) is not str or not source[key]:
            raise ValueError(f"flat source {key} is invalid")
    if not os.path.isabs(source["path"]) \
            or not os.path.isabs(source["receipt_path"]):
        raise ValueError("flat source and receipt paths must be absolute")
    try:
        source_digest = sha256_file(source["path"])
        receipt_digest = sha256_file(source["receipt_path"])
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"flat source authentication failed: {error}") from error
    if source_digest != source["sha256"]:
        raise ValueError("flat source SHA-256 mismatch")
    if receipt_digest != source["receipt_sha256"]:
        raise ValueError("flat source receipt SHA-256 mismatch")
    try:
        loaded = load_retarget_npz(source["path"], source["receipt_path"])
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"flat source receipt authentication failed: {error}") \
            from error
    provenance = loaded.provenance
    if source["name"] != loaded.name or source["terrain_id"] != "flat" \
            or source["source_fps"] != loaded.fps \
            or source["source_frames"] != len(loaded.qpos) \
            or provenance["path"] != source["path"] \
            or provenance["sha256"] != source["sha256"] \
            or provenance["receipt_path"] != source["receipt_path"] \
            or provenance["receipt_sha256"] != source["receipt_sha256"] \
            or source["receipt_schema"] != provenance["receipt"]["schema"] \
            or source["receipt_status"] != provenance["receipt"]["status"] \
            or source["receipt_status"] != "accepted":
        raise ValueError("flat source/receipt identity changed")
    output_frames = source["output_frames"]
    left = source["left_source_index"]
    right = source["right_source_index"]
    alpha = source["source_alpha"]
    if type(output_frames) is not int or output_frames < 1 \
            or type(left) is not list or type(right) is not list \
            or type(alpha) is not list \
            or len(left) != output_frames or len(right) != output_frames \
            or len(alpha) != output_frames:
        raise ValueError("flat source interpolation map dimensions changed")
    if any(type(value) is not int for value in left + right) \
            or any(type(value) is not float
                   or not np.isfinite(value) for value in alpha):
        raise ValueError("flat source interpolation map types changed")
    source_first = int(loaded.source_frames[0])
    source_last = int(loaded.source_frames[-1])
    if any(value < source_first or value > source_last
           for value in left + right) \
            or any(value < 0.0 or value > 1.0 for value in alpha):
        raise ValueError("flat source interpolation map values are invalid")


class PublicationCommittedError(RuntimeError):
    """The new output is durable, but old-tree cleanup did not complete."""

    def __init__(self, output_dir, scratch_path, error):
        super().__init__(
            "publication committed and the new output is authoritative, but "
            f"cleanup failed for {scratch_path}: {error}")
        self.committed = True
        self.output_dir = output_dir
        self.scratch_path = scratch_path


class PublicationRollbackError(RuntimeError):
    """A pre-durable commit failed and its rollback could not be certified."""

    def __init__(self, output_dir, scratch_path, commit_error, rollback_error):
        super().__init__(
            "publication durability failed and rollback could not be "
            f"certified; preserved output={output_dir} and "
            f"scratch={scratch_path}: {rollback_error}")
        self.committed = False
        self.output_dir = output_dir
        self.scratch_path = scratch_path
        self.commit_error = commit_error
        self.rollback_error = rollback_error


class ScratchOwnershipError(RuntimeError):
    """A unique scratch pathname no longer names the directory we created."""


@dataclass(frozen=True)
class _SceneSnapshot:
    scene_id: str
    scene_json: bytes
    terrain_bin: bytes
    terrain_obj: bytes
    walkability_bin: bytes
    metadata: dict


@dataclass(frozen=True)
class _ScenePackSnapshot:
    index_json: bytes
    index: dict
    scenes: tuple[_SceneSnapshot, ...]


def _float_matrix_bytes(
    values: np.ndarray, *, magic: bytes, dims: int, label: str
) -> bytes:
    error_message = (
        f"{label} must be a finite real floating matrix shaped (N, {dims}) "
        "with float32 or float64 dtype"
    )
    try:
        source = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(error_message) from error
    if (
        source.ndim != 2
        or source.shape[0] < 1
        or source.shape[1] != dims
        or source.shape[0] > _UINT32_MAX
    ):
        raise ValueError(error_message)
    if source.dtype.kind != "f" or source.dtype.itemsize not in (4, 8):
        raise ValueError(error_message)
    if not np.isfinite(source).all():
        raise ValueError(error_message)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            encoded = np.ascontiguousarray(source, dtype="<f4")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(error_message) from error
    if not np.isfinite(encoded).all():
        raise ValueError(
            f"{label} must remain finite when encoded as float32"
        )
    return _HEADER.pack(magic, VERSION, len(encoded), dims) + encoded.tobytes(
        order="C"
    )


def _read_float_matrix(
    path: os.PathLike | str, *, magic: bytes, dims: int, label: str
) -> np.ndarray:
    with open(path, "rb") as stream:
        header = stream.read(_HEADER.size)
        if len(header) < _HEADER.size:
            raise ValueError(f"{path}: truncated {label} header")
        actual_magic, version, frames, actual_dims = _HEADER.unpack(header)
        if (
            actual_magic != magic
            or version != VERSION
            or actual_dims != dims
        ):
            raise ValueError(f"{path}: unsupported {label} schema")
        if frames < 1:
            raise ValueError(f"{path}: invalid {label} frame count")

        payload_size = frames * dims * _FLOAT32_BYTES
        if payload_size > np.iinfo(np.intp).max:
            raise ValueError(
                f"{path}: {label} payload exceeds platform index limit"
            )
        expected_size = _HEADER.size + payload_size
        actual_size = os.fstat(stream.fileno()).st_size
        if actual_size < expected_size:
            raise ValueError(f"{path}: truncated {label} payload")
        if actual_size > expected_size:
            raise ValueError(f"{path}: trailing {label} bytes")
        payload = stream.read(payload_size)
        if len(payload) != payload_size:
            raise ValueError(f"{path}: truncated {label} payload")

    matrix = np.frombuffer(payload, dtype="<f4").reshape(frames, dims).copy()
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: {label} values must be finite")
    return matrix


def terrain_sidecar_bytes(features: np.ndarray) -> bytes:
    return _float_matrix_bytes(
        features, magic=MAGIC, dims=DIMS, label="terrain sidecar"
    )


def write_terrain_sidecar(
    path: os.PathLike | str, features: np.ndarray
) -> None:
    payload = terrain_sidecar_bytes(features)
    with open(path, "wb") as stream:
        stream.write(payload)


def read_terrain_sidecar(path: os.PathLike | str) -> np.ndarray:
    return _read_float_matrix(
        path, magic=MAGIC, dims=DIMS, label="terrain sidecar"
    )


def support_sidecar_bytes(support: np.ndarray) -> bytes:
    return _float_matrix_bytes(
        support,
        magic=SUPPORT_MAGIC,
        dims=SUPPORT_DIMS,
        label="support sidecar",
    )


def write_support_sidecar(
    path: os.PathLike | str, support: np.ndarray
) -> None:
    payload = support_sidecar_bytes(support)
    with open(path, "wb") as stream:
        stream.write(payload)


def read_support_sidecar(path: os.PathLike | str) -> np.ndarray:
    return _read_float_matrix(
        path,
        magic=SUPPORT_MAGIC,
        dims=SUPPORT_DIMS,
        label="support sidecar",
    )


def walkability_bytes(walkability: np.ndarray) -> bytes:
    error_message = (
        "walkability must be a 2-D integer grid containing classes 0, 1, or 2"
    )
    try:
        source = np.asarray(walkability)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(error_message) from error
    if source.ndim != 2:
        raise ValueError(error_message)
    nz, nx = source.shape
    if nx < 2 or nz < 2:
        raise ValueError("walkability dimensions must be at least 2x2")
    if nx > _UINT32_MAX or nz > _UINT32_MAX:
        raise ValueError("walkability dimensions exceed the G1WM v1 limit")
    payload_size = nx * nz
    if payload_size > np.iinfo(np.intp).max:
        raise ValueError("walkability payload exceeds platform index limit")
    if source.dtype.kind not in "iu":
        raise ValueError(error_message)
    if np.any(source < 0) or np.any(source > 2):
        raise ValueError(error_message)
    encoded = np.ascontiguousarray(source, dtype=np.uint8)
    return _HEADER.pack(
        WALKABILITY_MAGIC, VERSION, nx, nz
    ) + encoded.tobytes(order="C")


def write_walkability(
    path: os.PathLike | str, walkability: np.ndarray
) -> None:
    payload = walkability_bytes(walkability)
    with open(path, "wb") as stream:
        stream.write(payload)


def read_walkability(path: os.PathLike | str) -> np.ndarray:
    label = "walkability sidecar"
    with open(path, "rb") as stream:
        header = stream.read(_HEADER.size)
        if len(header) < _HEADER.size:
            raise ValueError(f"{path}: truncated {label} header")
        magic, version, nx, nz = _HEADER.unpack(header)
        if magic != WALKABILITY_MAGIC or version != VERSION:
            raise ValueError(f"{path}: unsupported {label} schema")
        if nx < 2 or nz < 2:
            raise ValueError(f"{path}: invalid walkability dimensions")

        payload_size = nx * nz
        if payload_size > np.iinfo(np.intp).max:
            raise ValueError(
                f"{path}: walkability payload exceeds platform index limit"
            )
        expected_size = _HEADER.size + payload_size
        actual_size = os.fstat(stream.fileno()).st_size
        if actual_size < expected_size:
            raise ValueError(f"{path}: truncated {label} payload")
        if actual_size > expected_size:
            raise ValueError(f"{path}: trailing {label} bytes")
        payload = stream.read(payload_size)
        if len(payload) != payload_size:
            raise ValueError(f"{path}: truncated {label} payload")

    grid = np.frombuffer(payload, dtype=np.uint8).reshape(nz, nx).copy()
    if np.any(grid > 2):
        raise ValueError(
            f"{path}: walkability values must contain classes 0, 1, or 2"
        )
    return grid


def _deep_normalize_json(value, label="JSON"):
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{label} object keys must be strings")
        return {
            key: _deep_normalize_json(child, f"{label}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _deep_normalize_json(child, f"{label}[{index}]")
            for index, child in enumerate(value)
        ]
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"{label} must be finite")
        return 0.0 if value == 0.0 else value
    if type(value) is int or type(value) in (str, bool) or value is None:
        return value
    raise TypeError(
        f"{label} contains non-JSON scalar {type(value).__name__}")


def canonical_json_bytes(value) -> bytes:
    return (json.dumps(
        _deep_normalize_json(value), indent=2, sort_keys=True,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _parse_cached_json(payload, label):
    if type(payload) is not bytes:
        raise TypeError(f"cached {label} must be exact bytes")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cached {label} is not valid UTF-8 JSON") from error
    if canonical_json_bytes(decoded) != payload:
        raise ValueError(f"cached {label} is not canonical or stable")
    return decoded


def _normalized_manifest_base(manifest):
    if type(manifest) is not dict or set(manifest) != MANIFEST_BASE_KEYS:
        raise ValueError(
            f"manifest base keys must be {sorted(MANIFEST_BASE_KEYS)}")
    normalized = json.loads(canonical_json_bytes(manifest))
    if normalized["schema"] != "g1-terrain-artifacts/v2":
        raise ValueError("manifest schema must be g1-terrain-artifacts/v2")
    validation = normalized["validation"]
    if type(validation) is not dict or set(validation) != VALIDATION_KEYS \
            or validation["schema"] != "g1-terrain-validation/v1":
        raise ValueError(
            "manifest validation object has invalid schema or keys")
    surface = normalized["surface"]
    if type(surface) is not dict or set(surface) != {"semantics", "signature"}:
        raise ValueError("manifest surface object has invalid keys")
    signature = surface["signature"]
    if type(signature) is not str or not _HEX_SHA256.fullmatch(signature):
        raise ValueError("manifest surface signature must be lowercase SHA-256")
    semantics = surface["semantics"]
    if type(semantics) is not dict:
        raise ValueError("manifest surface semantics must be an object")
    expected_signature = hashlib.sha256(json.dumps(
        semantics, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    if signature != expected_signature:
        raise ValueError("manifest surface signature does not match semantics")
    return normalized


def _snapshot_scene_pack(scene_pack, manifest_base):
    from .scenes import REQUIRED_SCENE_IDS

    try:
        index_json = scene_pack.index_json
        source_scenes = tuple(scene_pack.scenes)
    except (AttributeError, TypeError) as error:
        raise TypeError("scene_pack must contain cached index and scenes") \
            from error
    index = _parse_cached_json(index_json, "scene index")
    if type(index) is not dict or set(index) != INDEX_KEYS \
            or index.get("schema") != "g1-terrain-scene-index/v1" \
            or index.get("default_scene_id") != "grail-curb-default" \
            or index.get("coordinate_signature") != COORDINATE_SIGNATURE:
        raise ValueError("scene index keys or schema changed")
    surface_signature = index.get("surface_signature")
    if type(surface_signature) is not str \
            or not _HEX_SHA256.fullmatch(surface_signature):
        raise ValueError("scene index surface signature is invalid")
    if surface_signature != manifest_base["surface"]["signature"]:
        raise ValueError(
            "manifest surface signature does not match scene index")
    if type(index.get("scene_ids")) is not list \
            or tuple(index["scene_ids"]) != REQUIRED_SCENE_IDS:
        raise ValueError("scene index must contain exact required scene IDs")
    if tuple(getattr(scene, "scene_id", None) for scene in source_scenes) \
            != REQUIRED_SCENE_IDS:
        raise ValueError("scene pack must contain exact required scene order")
    descriptors = index.get("scenes")
    if type(descriptors) is not list \
            or len(descriptors) != len(REQUIRED_SCENE_IDS):
        raise ValueError("scene index descriptor count changed")

    snapshots = []
    for descriptor, source, scene_id in zip(
            descriptors, source_scenes, REQUIRED_SCENE_IDS):
        payloads = {}
        for attribute in (
                "scene_json", "terrain_bin", "terrain_obj",
                "walkability_bin"):
            payload = getattr(source, attribute, None)
            if type(payload) is not bytes:
                raise TypeError(
                    f"{scene_id}: cached {attribute} must be exact bytes")
            payloads[attribute] = payload
        metadata = _parse_cached_json(
            payloads["scene_json"], f"{scene_id} scene JSON")
        expected_descriptor = {
            "id": scene_id,
            "path": f"scenes/{scene_id}/scene.json",
            "sha256": hashlib.sha256(
                payloads["scene_json"]).hexdigest(),
        }
        if descriptor != expected_descriptor:
            raise ValueError(
                f"{scene_id}: scene descriptor SHA-256/path mismatch")
        if type(metadata) is not dict \
                or metadata.get("schema") != "g1-terrain-scene/v1" \
                or metadata.get("id") != scene_id:
            raise ValueError(f"{scene_id}: scene JSON ID or schema changed")
        if metadata.get("surface_signature") != surface_signature:
            raise ValueError(
                f"{scene_id}: scene surface signature does not match index")
        asset_specs = (
            ("heightfield", "terrain.bin", "terrain_bin"),
            ("mesh", "terrain.obj", "terrain_obj"),
            ("walkability", "walkability.bin", "walkability_bin"),
        )
        for descriptor_name, expected_path, payload_name in asset_specs:
            asset = metadata.get(descriptor_name)
            if type(asset) is not dict \
                    or set(asset) != ASSET_DESCRIPTOR_KEYS[descriptor_name] \
                    or asset.get("path") != expected_path \
                    or asset.get("sha256") != hashlib.sha256(
                        payloads[payload_name]).hexdigest():
                raise ValueError(
                    f"{scene_id}: {descriptor_name} descriptor "
                    "path/SHA-256 mismatch")
        snapshots.append(_SceneSnapshot(
            scene_id, payloads["scene_json"], payloads["terrain_bin"],
            payloads["terrain_obj"], payloads["walkability_bin"], metadata,
        ))
    return _ScenePackSnapshot(index_json, index, tuple(snapshots))


def _write_bytes_fsync(path, payload):
    if type(payload) is not bytes:
        raise TypeError("artifact payload must be exact bytes")
    with open(path, "xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read_regular_bytes(path):
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ValueError(f"missing staged regular file {path}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"staged node is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open staged regular file {path}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) \
                or (opened.st_dev, opened.st_ino) \
                != (before.st_dev, before.st_ino):
            raise ValueError(f"staged regular file changed during open: {path}")
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"staged regular file changed during read: {path}")
        payload = b"".join(chunks)
        if len(payload) != after.st_size:
            raise ValueError(f"staged regular file changed size: {path}")
        return payload
    finally:
        os.close(descriptor)


def _regular_file_matches_bytes(path, expected):
    if type(expected) is not bytes:
        raise TypeError("expected artifact payload must be exact bytes")
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ValueError(f"missing staged regular file {path}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"staged node is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) \
                or (opened.st_dev, opened.st_ino) \
                != (before.st_dev, before.st_ino):
            raise ValueError(f"staged regular file changed during open: {path}")
        if opened.st_size != len(expected):
            return False
        view = memoryview(expected)
        offset = 0
        while offset < len(view):
            block = os.read(descriptor, min(1024 * 1024, len(view) - offset))
            if not block or block != view[offset:offset + len(block)]:
                return False
            offset += len(block)
        if os.read(descriptor, 1):
            return False
        after = os.fstat(descriptor)
        if (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"staged regular file changed during read: {path}")
        return True
    finally:
        os.close(descriptor)


def sha256_file(path):
    path = os.fspath(path)
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ValueError(f"missing staged regular file {path}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"staged node is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) \
                or (opened.st_dev, opened.st_ino) \
                != (before.st_dev, before.st_ino):
            raise ValueError(f"staged regular file changed during open: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
        after = os.fstat(descriptor)
        if size != after.st_size or (
                opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"staged regular file changed during hash: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _write_database_exclusive(path, artifacts):
    temporary = path + ".building"
    try:
        write_holden_database(temporary, artifacts)
        _fsync_regular_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(os.path.dirname(path))
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _write_scene_pack(staging, scene_pack):
    scenes_root = os.path.join(staging, "scenes")
    os.mkdir(scenes_root)
    _write_bytes_fsync(
        os.path.join(scenes_root, "index.json"), scene_pack.index_json)
    for scene in scene_pack.scenes:
        if not _SCENE_ID_PATTERN.fullmatch(scene.scene_id):
            raise ValueError(f"unsafe scene ID {scene.scene_id!r}")
        scene_dir = os.path.join(scenes_root, scene.scene_id)
        os.mkdir(scene_dir)
        for name, payload in (
                ("scene.json", scene.scene_json),
                ("terrain.bin", scene.terrain_bin),
                ("terrain.obj", scene.terrain_obj),
                ("walkability.bin", scene.walkability_bin)):
            _write_bytes_fsync(os.path.join(scene_dir, name), payload)


def _finalize_manifest(staging, manifest_base):
    manifest = dict(manifest_base)
    manifest["database"] = {
        "path": "database.bin", "schema": "holden-database/v1",
        "sha256": sha256_file(os.path.join(staging, "database.bin")),
    }
    manifest["sidecars"] = {
        "terrain_features": {
            "path": "terrain_features.bin", "schema": "G1TF/v1",
            "version": 1, "dimensions": 4,
            "sha256": sha256_file(
                os.path.join(staging, "terrain_features.bin")),
        },
        "terrain_support": {
            "path": "terrain_support.bin", "schema": "G1SP/v1",
            "version": 1, "dimensions": 3,
            "columns": list(SUPPORT_COLUMNS),
            "sha256": sha256_file(
                os.path.join(staging, "terrain_support.bin")),
        },
    }
    manifest["scene_index"] = {
        "path": "scenes/index.json",
        "schema": "g1-terrain-scene-index/v1",
        "sha256": sha256_file(
            os.path.join(staging, "scenes", "index.json")),
    }
    manifest["validation_file"] = {
        "path": "validation.json",
        "schema": "g1-terrain-validation/v1",
        "sha256": sha256_file(os.path.join(staging, "validation.json")),
    }
    if set(manifest) != MOTION_MANIFEST_KEYS:
        raise ValueError("final motion manifest key set changed")
    return manifest


def _expected_tree(scene_pack):
    files = {
        "database.bin", "terrain_features.bin", "terrain_support.bin",
        "manifest.json", "validation.json", os.path.join("scenes", "index.json"),
    }
    directories = {".", "scenes"}
    for scene in scene_pack.scenes:
        scene_dir = os.path.join("scenes", scene.scene_id)
        directories.add(scene_dir)
        for name in (
                "scene.json", "terrain.bin", "terrain.obj",
                "walkability.bin"):
            files.add(os.path.join(scene_dir, name))
    return files, directories


def _inspect_tree(root):
    try:
        root_stat = os.lstat(root)
    except OSError as error:
        raise ValueError("staged artifact root is missing") from error
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("staged artifact root must be a real directory")
    files = set()
    directories = {"."}
    pending = [(root, ".")]
    while pending:
        directory, relative_directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise ValueError(
                f"cannot inspect staged directory {relative_directory}") \
                from error
        for entry in entries:
            relative = entry.name if relative_directory == "." else \
                os.path.join(relative_directory, entry.name)
            try:
                node = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"cannot inspect staged node {relative}") \
                    from error
            if stat.S_ISREG(node.st_mode):
                files.add(relative)
            elif stat.S_ISDIR(node.st_mode):
                directories.add(relative)
                pending.append((entry.path, relative))
            elif stat.S_ISLNK(node.st_mode):
                raise ValueError(f"staged tree contains symlink {relative}")
            else:
                raise ValueError(
                    f"staged tree contains non-regular node {relative}")
    return files, directories


def _validate_staged_v2(staging, artifacts, manifest, scene_pack):
    actual_files, actual_directories = _inspect_tree(staging)
    expected_files, expected_directories = _expected_tree(scene_pack)
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError(
            "staged artifact tree is incomplete, has extras, or has "
            "unexpected directories")

    expected_bytes = {
        "manifest.json": canonical_json_bytes(manifest),
        "validation.json": canonical_json_bytes(manifest["validation"]),
        os.path.join("scenes", "index.json"): scene_pack.index_json,
    }
    for scene in scene_pack.scenes:
        prefix = os.path.join("scenes", scene.scene_id)
        expected_bytes.update({
            os.path.join(prefix, "scene.json"): scene.scene_json,
            os.path.join(prefix, "terrain.bin"): scene.terrain_bin,
            os.path.join(prefix, "terrain.obj"): scene.terrain_obj,
            os.path.join(prefix, "walkability.bin"): scene.walkability_bin,
        })
    for relative, expected in expected_bytes.items():
        if not _regular_file_matches_bytes(
                os.path.join(staging, relative), expected):
            raise ValueError(f"staged scene/artifact bytes changed: {relative}")

    linked_payloads = (
        (manifest["database"], "database.bin"),
        (manifest["sidecars"]["terrain_features"], "terrain_features.bin"),
        (manifest["sidecars"]["terrain_support"], "terrain_support.bin"),
        (manifest["scene_index"], os.path.join("scenes", "index.json")),
        (manifest["validation_file"], "validation.json"),
    )
    for descriptor, expected_path in linked_payloads:
        if descriptor.get("path") != expected_path:
            raise ValueError(f"manifest path changed for {expected_path}")
        if sha256_file(os.path.join(staging, expected_path)) \
                != descriptor.get("sha256"):
            raise ValueError(
                f"manifest SHA-256 does not trust {expected_path}")

    for descriptor, scene in zip(
            scene_pack.index["scenes"], scene_pack.scenes):
        scene_path = os.path.join(staging, descriptor["path"])
        if sha256_file(scene_path) != descriptor["sha256"]:
            raise ValueError(
                f"{scene.scene_id}: staged scene JSON digest is not trusted")
        for metadata_name, filename in (
                ("heightfield", "terrain.bin"),
                ("mesh", "terrain.obj"),
                ("walkability", "walkability.bin")):
            asset = scene.metadata[metadata_name]
            if asset["path"] != filename:
                raise ValueError(
                    f"{scene.scene_id}: staged asset path is not trusted")
            asset_path = os.path.join(
                staging, "scenes", scene.scene_id, filename)
            if sha256_file(asset_path) != asset["sha256"]:
                raise ValueError(
                    f"{scene.scene_id}: staged asset SHA-256 is not trusted")

    loaded = read_holden_database(os.path.join(staging, "database.bin"))
    loaded.terrain_features = read_terrain_sidecar(
        os.path.join(staging, "terrain_features.bin"))
    loaded.terrain_support = read_support_sidecar(
        os.path.join(staging, "terrain_support.bin"))
    loaded.validate()

    def encoded_equal(actual, expected, dtype):
        actual = np.ascontiguousarray(actual, dtype=dtype)
        expected = np.ascontiguousarray(expected, dtype=dtype)
        if actual.shape != expected.shape:
            return False
        view_dtype = "<u4" if dtype == "<f4" else "u1"
        return np.array_equal(
            actual.view(view_dtype), expected.view(view_dtype))

    for name, dtype in (
            ("positions", "<f4"), ("velocities", "<f4"),
            ("rotations", "<f4"), ("angular_velocities", "<f4"),
            ("parents", "<i4"), ("range_starts", "<i4"),
            ("range_stops", "<i4"), ("contacts", "u1"),
            ("terrain_features", "<f4"), ("terrain_support", "<f4")):
        if not encoded_equal(
                getattr(loaded, name), getattr(artifacts, name), dtype):
            raise ValueError(
                f"staged {name} does not match requested artifacts")


def _fsync_regular_file(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"cannot fsync non-regular staged node {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_directory(path):
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"not a real directory: {path}")
    return descriptor


def _fsync_directory(path):
    descriptor = _open_directory(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root):
    files, directories = _inspect_tree(root)
    for relative in sorted(files):
        _fsync_regular_file(os.path.join(root, relative))
    for relative in sorted(
            directories, key=lambda value: value.count(os.sep), reverse=True):
        path = root if relative == "." else os.path.join(root, relative)
        _fsync_directory(path)


@contextmanager
def _locked_parent(parent):
    descriptor = _open_directory(parent)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _fsync_parent(parent_descriptor):
    os.fsync(parent_descriptor)


def _rename_exchange(source, destination):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise NotImplementedError(
            "Linux renameat2(RENAME_EXCHANGE) is unavailable")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD, os.fsencode(source), _AT_FDCWD,
        os.fsencode(destination), _RENAME_EXCHANGE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            f"renameat2(RENAME_EXCHANGE) failed: {os.strerror(error_number)}",
            destination,
        )


def _output_exists_as_directory(output_dir):
    return _output_directory_identity(output_dir) is not None


def _output_directory_identity(output_dir):
    try:
        node = os.lstat(output_dir)
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(node.st_mode):
        raise ValueError(
            "existing output must be a real directory, not a symlink or "
            "non-directory")
    return node.st_dev, node.st_ino


def _scratch_directory_identity(path):
    try:
        node = os.lstat(path)
    except OSError as error:
        raise ScratchOwnershipError(
            f"owned scratch identity is missing at {path}; refusing deletion") \
            from error
    if not stat.S_ISDIR(node.st_mode):
        raise ScratchOwnershipError(
            f"owned scratch identity is not a real directory at {path}; "
            "refusing deletion")
    return node.st_dev, node.st_ino


def _remove_owned_scratch(path, expected_identity):
    actual_identity = _scratch_directory_identity(path)
    if actual_identity != expected_identity:
        raise ScratchOwnershipError(
            f"owned scratch identity changed at {path}; refusing deletion")
    _remove_path(path)


def _remove_path(path: str) -> None:
    if not os.path.lexists(path):
        return
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.unlink(path)


def publish_artifacts(
    output_dir, artifacts, manifest_base, scene_pack, validate_candidate,
):
    manifest_base = _normalized_manifest_base(manifest_base)
    if not isinstance(artifacts, ArtifactSet):
        raise TypeError("artifacts must be an ArtifactSet")
    artifacts.validate()
    if not callable(validate_candidate):
        raise TypeError("validate_candidate must be callable")
    scene_pack = _snapshot_scene_pack(scene_pack, manifest_base)

    output_dir = os.path.abspath(os.fspath(output_dir))
    parent = os.path.dirname(output_dir)
    basename = os.path.basename(output_dir)
    if not basename:
        raise ValueError("output directory must have a basename")
    os.makedirs(parent, exist_ok=True)
    _output_exists_as_directory(output_dir)

    staging = tempfile.mkdtemp(
        prefix=f".{basename}.staging-", dir=parent)
    staging_identity = _scratch_directory_identity(staging)
    candidate_identity = staging_identity
    preserve_staging = False
    committed = False
    try:
        _write_database_exclusive(
            os.path.join(staging, "database.bin"), artifacts)
        _write_bytes_fsync(
            os.path.join(staging, "terrain_features.bin"),
            terrain_sidecar_bytes(artifacts.terrain_features))
        _write_bytes_fsync(
            os.path.join(staging, "terrain_support.bin"),
            support_sidecar_bytes(artifacts.terrain_support))
        _write_scene_pack(staging, scene_pack)
        _write_bytes_fsync(
            os.path.join(staging, "validation.json"),
            canonical_json_bytes(manifest_base["validation"]))
        manifest = _finalize_manifest(staging, manifest_base)
        _write_bytes_fsync(
            os.path.join(staging, "manifest.json"),
            canonical_json_bytes(manifest))

        _validate_staged_v2(staging, artifacts, manifest, scene_pack)
        validate_candidate(staging)
        _validate_staged_v2(staging, artifacts, manifest, scene_pack)
        _fsync_tree(staging)

        with _locked_parent(parent) as parent_descriptor:
            _validate_staged_v2(staging, artifacts, manifest, scene_pack)
            output_identity = _output_directory_identity(output_dir)
            exchanged = False
            if output_identity is not None:
                _rename_exchange(staging, output_dir)
                exchanged = True
                staging_identity = output_identity
            else:
                os.replace(staging, output_dir)
                staging_identity = None
            try:
                _fsync_parent(parent_descriptor)
            except BaseException as commit_error:
                try:
                    if exchanged:
                        _rename_exchange(staging, output_dir)
                    else:
                        os.replace(output_dir, staging)
                    staging_identity = candidate_identity
                    _fsync_parent(parent_descriptor)
                except BaseException as rollback_error:
                    preserve_staging = True
                    raise PublicationRollbackError(
                        output_dir, staging, commit_error, rollback_error,
                    ) from rollback_error
                raise

            committed = True
            if exchanged:
                try:
                    _remove_owned_scratch(staging, staging_identity)
                except BaseException as cleanup_error:
                    preserve_staging = True
                    raise PublicationCommittedError(
                        output_dir, staging, cleanup_error,
                    ) from cleanup_error
        return manifest
    finally:
        if not committed and not preserve_staging:
            _remove_owned_scratch(staging, staging_identity)


def _validate_staged_flat(staging, artifacts, features, manifest):
    files, directories = _inspect_tree(staging)
    if files != {"database.bin", "features.bin", "manifest.json"} \
            or directories != {"."}:
        raise ValueError("flat artifact tree changed")
    expected_manifest = canonical_json_bytes(manifest)
    if _read_regular_bytes(os.path.join(staging, "manifest.json")) \
            != expected_manifest:
        raise ValueError("flat manifest bytes changed")
    for name in ("database.bin", "features.bin"):
        descriptor = manifest["artifacts"][name]
        path = os.path.join(staging, name)
        node = os.stat(path)
        if descriptor != {
            "path": name,
            "size_bytes": node.st_size,
            "sha256": sha256_file(path),
        }:
            raise ValueError(f"flat {name} descriptor changed")
    loaded_database = read_holden_database(
        os.path.join(staging, "database.bin"))
    loaded_features = read_features(os.path.join(staging, "features.bin"))
    for field in (
        "positions", "velocities", "rotations", "angular_velocities",
        "parents", "range_starts", "range_stops", "contacts",
    ):
        if not np.array_equal(
                getattr(loaded_database, field), getattr(artifacts, field)):
            raise ValueError(f"flat database {field} round trip changed")
    for field in ("values", "offset", "scale"):
        if not np.array_equal(
                getattr(loaded_features, field), getattr(features, field)):
            raise ValueError(f"flat features {field} round trip changed")


def publish_flat_artifacts(
    output_dir, artifacts, features, manifest_base, validate_candidate,
):
    if not isinstance(artifacts, ArtifactSet):
        raise TypeError("artifacts must be an ArtifactSet")
    artifacts.validate()
    if not isinstance(features, FeatureSet):
        raise TypeError("features must be a FeatureSet")
    features.validate()
    if len(features.values) != len(artifacts.positions):
        raise ValueError("database and matching feature rows differ")
    if type(manifest_base) is not dict \
            or manifest_base.get("schema") != _FLAT_SCHEMA \
            or "artifacts" in manifest_base or "status" in manifest_base:
        raise ValueError(
            "flat manifest base must use g1-lmm-flat-data/v3")
    if not callable(validate_candidate):
        raise TypeError("validate_candidate must be callable")
    _authenticate_flat_manifest_inputs(manifest_base)
    normalized_base = json.loads(canonical_json_bytes(manifest_base))

    output_dir = os.path.abspath(os.fspath(output_dir))
    parent = os.path.dirname(output_dir)
    basename = os.path.basename(output_dir)
    if not basename:
        raise ValueError("output directory must have a basename")
    os.makedirs(parent, exist_ok=True)
    _output_exists_as_directory(output_dir)
    staging = tempfile.mkdtemp(prefix=f".{basename}.staging-", dir=parent)
    staging_identity = _scratch_directory_identity(staging)
    candidate_identity = staging_identity
    preserve_staging = False
    committed = False
    try:
        _write_database_exclusive(
            os.path.join(staging, "database.bin"), artifacts)
        _write_bytes_fsync(
            os.path.join(staging, "features.bin"), features_bytes(features))
        artifact_descriptors = {}
        for name in ("database.bin", "features.bin"):
            path = os.path.join(staging, name)
            artifact_descriptors[name] = {
                "path": name,
                "size_bytes": os.stat(path).st_size,
                "sha256": sha256_file(path),
            }
        manifest = dict(normalized_base)
        manifest["status"] = "accepted"
        manifest["artifacts"] = artifact_descriptors
        _write_bytes_fsync(
            os.path.join(staging, "manifest.json"),
            canonical_json_bytes(manifest))

        _validate_staged_flat(staging, artifacts, features, manifest)
        validate_candidate(staging)
        _authenticate_flat_manifest_inputs(manifest)
        _validate_staged_flat(staging, artifacts, features, manifest)
        _fsync_tree(staging)
        with _locked_parent(parent) as parent_descriptor:
            _authenticate_flat_manifest_inputs(manifest)
            _validate_staged_flat(staging, artifacts, features, manifest)
            output_identity = _output_directory_identity(output_dir)
            exchanged = False
            if output_identity is not None:
                _rename_exchange(staging, output_dir)
                exchanged = True
                staging_identity = output_identity
            else:
                os.replace(staging, output_dir)
                staging_identity = None
            try:
                _fsync_parent(parent_descriptor)
            except BaseException as commit_error:
                try:
                    if exchanged:
                        _rename_exchange(staging, output_dir)
                    else:
                        os.replace(output_dir, staging)
                    staging_identity = candidate_identity
                    _fsync_parent(parent_descriptor)
                except BaseException as rollback_error:
                    preserve_staging = True
                    raise PublicationRollbackError(
                        output_dir, staging, commit_error, rollback_error,
                    ) from rollback_error
                raise
            committed = True
            if exchanged:
                try:
                    _remove_owned_scratch(staging, staging_identity)
                except BaseException as cleanup_error:
                    preserve_staging = True
                    raise PublicationCommittedError(
                        output_dir, staging, cleanup_error,
                    ) from cleanup_error
        return manifest
    finally:
        if not committed and not preserve_staging:
            _remove_owned_scratch(staging, staging_identity)


def _require_lmm_descriptor_file(descriptor, *, expected_size, expected_sha):
    if type(descriptor) is not dict \
            or set(descriptor) != {"path", "size_bytes", "sha256"} \
            or type(descriptor.get("path")) is not str \
            or not descriptor["path"] \
            or type(descriptor.get("size_bytes")) is not int \
            or descriptor["size_bytes"] != expected_size \
            or descriptor.get("sha256") != expected_sha:
        raise ValueError("terrain LMM input descriptor changed")
    path = descriptor["path"]
    try:
        node = os.stat(path)
        digest = sha256_file(path)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("terrain LMM input authentication failed") from error
    if node.st_size != expected_size or digest != expected_sha:
        raise ValueError("terrain LMM input SHA-256/size changed")


def _normalized_lmm_terrain_inputs(
    manifest_base, validation, artifacts, features,
):
    from .features import FEATURE_NAMES, FEATURE_WEIGHTS

    if not isinstance(artifacts, ArtifactSet):
        raise TypeError("terrain LMM artifacts must be an ArtifactSet")
    artifacts.validate()
    if not isinstance(features, FeatureSet):
        raise TypeError("terrain LMM features must be a FeatureSet")
    features.validate()
    if type(manifest_base) is not dict \
            or set(manifest_base) != _LMM_TERRAIN_BASE_KEYS:
        raise ValueError("terrain LMM manifest base key set changed")
    if type(validation) is not dict \
            or set(validation) != _LMM_TERRAIN_VALIDATION_KEYS:
        raise ValueError("terrain LMM validation key set changed")
    normalized = json.loads(canonical_json_bytes(manifest_base))
    normalized_validation = json.loads(canonical_json_bytes(validation))
    if normalized["schema"] != _LMM_TERRAIN_SCHEMA \
            or normalized["output_fps"] != 60.0 \
            or normalized["database_frames"] != 851 \
            or normalized["source_count"] != 2 \
            or normalized["range_count"] != 2 \
            or normalized["trajectory_horizons"] != [20, 40, 60] \
            or normalized["dimensions"] != {
                "bones": 31, "features": 31, "latent": 32, "contacts": 2,
            }:
        raise ValueError("terrain LMM fixed manifest dimensions changed")
    if artifacts.positions.shape != (851, 31, 3) \
            or artifacts.contacts.shape != (851, 2) \
            or not np.array_equal(
                artifacts.range_starts, np.array([0, 256], np.int32)) \
            or not np.array_equal(
                artifacts.range_stops, np.array([256, 851], np.int32)) \
            or features.values.shape != (851, 31):
        raise ValueError("terrain LMM database/feature partition changed")
    skeleton = normalized["skeleton"]
    if type(skeleton) is not dict \
            or set(skeleton) != {"names", "parents", "basis", "signature"} \
            or skeleton != {
                "names": list(G1_SKELETON_NAMES),
                "parents": list(G1_SKELETON_PARENTS),
                "basis": "holden-y-up-right-handed-forward-plus-z",
                "signature": G1_SKELETON_SIGNATURE,
            }:
        raise ValueError("terrain LMM canonical skeleton changed")
    feature = normalized["feature"]
    if type(feature) is not dict or set(feature) != {
        "names", "weights", "offset", "scale", "signature",
        "terrain_indices",
    } or feature["names"] != list(FEATURE_NAMES) \
            or feature["weights"] != list(FEATURE_WEIGHTS) \
            or feature["terrain_indices"] != [27, 28, 29, 30] \
            or feature["offset"] != features.offset.tolist() \
            or feature["scale"] != features.scale.tolist() \
            or type(feature["signature"]) is not str \
            or not _HEX_SHA256.fullmatch(feature["signature"]):
        raise ValueError("terrain LMM feature receipt changed")
    terrain_scale = features.scale[27:31]
    if not np.isfinite(terrain_scale).all() \
            or np.any(terrain_scale <= 0.0) \
            or np.any(terrain_scale >= np.finfo(np.float32).max):
        raise ValueError("terrain LMM normalization disabled a terrain input")

    receipt = normalized["slope_source_receipt"]
    if type(receipt) is not dict or set(receipt) != _LMM_TERRAIN_RECEIPT_KEYS \
            or receipt["schema"] != "g1-lmm-authored-slope-source/v1" \
            or receipt["status"] != "provisional" \
            or receipt["clip_id"] != "terrain_slopes__slope_000__000" \
            or receipt["source_fps"] != 25.0 \
            or receipt["target_fps"] != 60.0 \
            or receipt["source_frames"] != 250 \
            or receipt["provisional_output_frames"] != 598 \
            or receipt["admitted_output_frames"] != 595 \
            or receipt["source_span_s"] != 9.96 \
            or receipt["output_span_s"] != 9.95 \
            or receipt["output_shortfall_s"] != 0.010000000000001563 \
            or receipt["rejected_output_ranges"] != [[0, 3]] \
            or receipt["admitted_output_range"] != [3, 598] \
            or receipt["intended_nonflat_provisional_range"] != [168, 542] \
            or receipt["intended_nonflat_admitted_range"] != [165, 539] \
            or receipt["metadata"] != {"scene_scale": 1.0} \
            or receipt["basis"] != "z-up-to-holden-shared-with-robot" \
            or receipt["support_calibration_m"] != 0.012000000104308128 \
            or receipt["support_calibration_applied_to"] != "terrain-only":
        raise ValueError("terrain LMM slope source receipt changed")
    interpolation = receipt.get("interpolation")
    if type(interpolation) is not dict or set(interpolation) != {
        "left_source_index", "right_source_index", "source_alpha",
    } or any(len(interpolation[key]) != 598 for key in interpolation) \
            or any(type(value) is not int for value in
                   interpolation["left_source_index"]
                   + interpolation["right_source_index"]) \
            or any(left > right for left, right in zip(
                interpolation["left_source_index"],
                interpolation["right_source_index"])) \
            or any(type(value) is not float or not np.isfinite(value)
                   or value < 0.0 or value > 1.0
                   for value in interpolation["source_alpha"]):
        raise ValueError("terrain LMM provisional interpolation map changed")
    try:
        object_rotation = np.asarray(
            receipt["object_rotation_binary32"], np.float64)
        object_translation = np.asarray(
            receipt["object_translation_binary32"], np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("terrain LMM reconstruction pose changed") from error
    if object_rotation.shape != (3, 3) \
            or object_translation.shape != (3,) \
            or not np.isfinite(object_rotation).all() \
            or not np.isfinite(object_translation).all() \
            or not np.array_equal(
                object_rotation,
                object_rotation.astype(np.float32).astype(np.float64)) \
            or not np.array_equal(
                object_translation,
                object_translation.astype(np.float32).astype(np.float64)) \
            or type(receipt["inverse_round_trip_max_error_m"]) is not float \
            or not 0.0 <= receipt["inverse_round_trip_max_error_m"] <= 1e-9 \
            or type(receipt["fk_max_error_m"]) is not float \
            or not 0.0 <= receipt["fk_max_error_m"] <= 1e-5 \
            or type(receipt["quaternion_norm_max_error"]) is not float \
            or not 0.0 <= receipt["quaternion_norm_max_error"] <= 1e-4:
        raise ValueError("terrain LMM reconstruction/FK receipt changed")
    receipt_sha = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    terrain = normalized["terrain"]
    if type(terrain) is not dict or set(terrain) != {
        "schema", "feature_distances_m", "source_receipt_sha256", "basis",
        "support_calibration_m", "support_calibration_applied_to",
        "exterior_policy", "query_counts",
    } or terrain["schema"] != "g1-lmm-authored-slope-surface/v1" \
            or terrain["feature_distances_m"] != [0.25, 0.5, 0.75, 1.0] \
            or terrain["source_receipt_sha256"] != receipt_sha \
            or terrain["basis"] != "z-up-to-holden-shared-with-robot" \
            or terrain["support_calibration_m"] != 0.012000000104308128 \
            or terrain["support_calibration_applied_to"] != "terrain-only" \
            or terrain["exterior_policy"] != {
                "source_height_m": 0.0,
                "runtime_height_m": -0.012000000104308128,
                "gradient_xz": [0.0, 0.0], "mesh_wins_inside": True,
            }:
        raise ValueError("terrain LMM surface receipt changed")
    counts = terrain["query_counts"]
    expected_totals = {"features": 2975, "support": 1785, "route": 595}
    if type(counts) is not dict or set(counts) != set(expected_totals):
        raise ValueError("terrain LMM query count groups changed")
    for name, total in expected_totals.items():
        item = counts[name]
        if type(item) is not dict or set(item) != {
            "exact_mesh", "flat_extension",
        } or any(type(item[key]) is not int or item[key] < 0 for key in item) \
                or item["exact_mesh"] + item["flat_extension"] != total:
            raise ValueError("terrain LMM query counts changed")
    if counts != {
        "features": {"exact_mesh": 1922, "flat_extension": 1053},
        "support": {"exact_mesh": 1132, "flat_extension": 653},
        "route": {"exact_mesh": 374, "flat_extension": 221},
    }:
        raise ValueError("terrain LMM authenticated query classification changed")
    if normalized["support"] != {
        "schema": "G1SP/v1", "dimensions": 3,
        "columns": list(SUPPORT_COLUMNS),
    }:
        raise ValueError("terrain LMM support receipt changed")
    contact = normalized["contact"]
    observation_keys = {
        "schema", "left_contact_frames", "right_contact_frames",
        "left_run_count", "right_run_count", "left_max_run_frames",
        "right_max_run_frames", "alternating_run_transition_count",
    }
    if type(contact) is not dict or set(contact) != {
        "semantics", "speed_threshold", "median_filter_frames",
        "median_filter_mode", "observations",
    } or {key: contact[key] for key in (
        "semantics", "speed_threshold", "median_filter_frames",
        "median_filter_mode",
    )} != {
        "semantics": "bundled-orange-duck-global-toe-speed-only",
        "speed_threshold": 0.15, "median_filter_frames": 6,
        "median_filter_mode": "nearest",
    } or type(contact["observations"]) is not dict \
            or set(contact["observations"]) != observation_keys \
            or contact["observations"]["schema"] \
            != "g1-lmm-bilateral-contact/v1":
        raise ValueError("terrain LMM contact receipt changed")
    if normalized["time_filters"] != {
        "root_position_frames": 31, "root_position_order": 3,
        "root_direction_frames": 61, "root_direction_order": 3,
        "contact_median_frames": 6, "forward_terrain_path_rows": 121,
    }:
        raise ValueError("terrain LMM time filters changed")

    inputs = normalized["inputs"]
    if type(inputs) is not dict or set(inputs) != {
        "flat_data", "slope_robot", "slope_usd", "slope_recon",
        "slope_metadata", "g1_xml",
    }:
        raise ValueError("terrain LMM input set changed")
    flat_data = inputs["flat_data"]
    if type(flat_data) is not dict or set(flat_data) != {
        "path", "manifest_path", "schema", "manifest_sha256",
    } or flat_data["manifest_path"] != "manifest.json" \
            or flat_data["schema"] != _FLAT_SCHEMA \
            or flat_data["manifest_sha256"] != (
                "5b5c48ccbb1dbabdbf87842d8b033c15b307199d72a8d90e4e39208ba5382db1"
            ):
        raise ValueError("terrain LMM flat-data descriptor changed")
    flat_manifest_path = os.path.join(
        flat_data["path"], flat_data["manifest_path"])
    if sha256_file(flat_manifest_path) != flat_data["manifest_sha256"]:
        raise ValueError("terrain LMM flat manifest SHA-256 changed")
    with open(flat_manifest_path, "rb") as stream:
        flat_manifest = _parse_cached_json(stream.read(), "flat manifest")
    _authenticate_flat_manifest_inputs(flat_manifest)
    fixed_inputs = {
        "slope_robot": (198603,
            "b77480d5f8f3339a3064276d6f9d443ac3a3456f20eb9195d48add176e561ee1"),
        "slope_usd": (6656,
            "8d1e696fb5bd2aecfa17797549bddd001093b060773cb313185a6a46db7eb5a5"),
        "slope_recon": (411476,
            "d05d6c5a7d6a13eff7e69da0e9e96ab5a79f700bb706611699f0b9c44c9b51ec"),
        "slope_metadata": (85,
            "7d88be608419afd2be127e4ac4c5a8aa1b4863fdf84e86c5ec93356881ee861d"),
        "g1_xml": (26914,
            "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376"),
    }
    for name, (size, digest) in fixed_inputs.items():
        _require_lmm_descriptor_file(
            inputs[name], expected_size=size, expected_sha=digest)
    if receipt["hashes"] != {
        "robot_sha256": fixed_inputs["slope_robot"][1],
        "usd_sha256": fixed_inputs["slope_usd"][1],
        "reconstruction_sha256": fixed_inputs["slope_recon"][1],
        "metadata_sha256": fixed_inputs["slope_metadata"][1],
        "g1_xml_sha256": fixed_inputs["g1_xml"][1],
    }:
        raise ValueError("terrain LMM receipt/input hashes disagree")

    sources = normalized["sources"]
    ranges = normalized["ranges"]
    expected_ranges = [
        {"start": 0, "stop": 256, "source_index": 0},
        {"start": 256, "stop": 851, "source_index": 1},
    ]
    if type(sources) is not list or len(sources) != 2 \
            or ranges != expected_ranges:
        raise ValueError("terrain LMM sources/ranges changed")
    source_keys = {
        "name", "terrain_id", "range", "source_fps", "source_frames",
        "output_frames", "source_map",
    }
    expected_source_values = (
        ("flat", "flat", {"start": 0, "stop": 256}, 120.0, 512, 256),
        ("terrain_slopes__slope_000__000",
         "terrain_slopes__slope_000__000",
         {"start": 256, "stop": 851}, 25.0, 250, 595),
    )
    for source, expected in zip(sources, expected_source_values):
        if type(source) is not dict or set(source) != source_keys \
                or tuple(source[key] for key in (
                    "name", "terrain_id", "range", "source_fps",
                    "source_frames", "output_frames")) != expected:
            raise ValueError("terrain LMM source identity changed")
        source_map = source["source_map"]
        output_frames = source["output_frames"]
        if type(source_map) is not dict or set(source_map) != {
            "left_source_index", "right_source_index", "source_alpha",
        } or any(len(source_map[key]) != output_frames for key in source_map) \
                or any(type(value) is not int for value in
                       source_map["left_source_index"]
                       + source_map["right_source_index"]) \
                or any(left > right for left, right in zip(
                    source_map["left_source_index"],
                    source_map["right_source_index"])) \
                or any(type(value) is not float or not np.isfinite(value)
                       or value < 0.0 or value > 1.0
                       for value in source_map["source_alpha"]):
            raise ValueError("terrain LMM admitted source map changed")
    if sources[1]["source_map"] != {
        "left_source_index": interpolation["left_source_index"][3:],
        "right_source_index": interpolation["right_source_index"][3:],
        "source_alpha": interpolation["source_alpha"][3:],
    }:
        raise ValueError("terrain LMM slope map is not provisional[3:]")
    continuity = normalized["continuity"]
    if type(continuity) is not dict or set(continuity) != {
        "schema", "threshold_rad_per_frame", "minimum_range_frames",
        "source_native_rejected_edge_count",
        "database_local_rejected_edge_count", "union_rejected_edge_count",
        "dropped_fragment_count", "dropped_frame_count",
        "published_range_count", "published_frame_count",
        "maximum_admitted_native_step_rad",
        "maximum_admitted_local_rotation_step_rad", "range_digest_sha256",
        "source_map_digest_sha256",
    } or continuity["schema"] != "g1-lmm-continuity/v1" \
            or continuity["threshold_rad_per_frame"] != 0.25 \
            or continuity["minimum_range_frames"] != 61 \
            or continuity["published_range_count"] != 2 \
            or continuity["published_frame_count"] != 851 \
            or continuity["range_digest_sha256"] \
            != hashlib.sha256(canonical_json_bytes(ranges)).hexdigest() \
            or continuity["source_map_digest_sha256"] != hashlib.sha256(
                canonical_json_bytes([source["source_map"] for source in sources])
            ).hexdigest():
        raise ValueError("terrain LMM continuity receipt changed")
    count_fields = (
        "source_native_rejected_edge_count",
        "database_local_rejected_edge_count", "union_rejected_edge_count",
        "dropped_fragment_count", "dropped_frame_count",
        "published_range_count", "published_frame_count",
    )
    maximum_fields = (
        "maximum_admitted_native_step_rad",
        "maximum_admitted_local_rotation_step_rad",
    )
    if any(type(continuity[key]) is not int or continuity[key] < 0
           for key in count_fields) \
            or any(type(continuity[key]) is not float
                   or not np.isfinite(continuity[key])
                   or continuity[key] < 0.0
                   for key in maximum_fields) \
            or continuity["source_native_rejected_edge_count"] != 0 \
            or continuity["database_local_rejected_edge_count"] != 0 \
            or continuity["union_rejected_edge_count"] != 0 \
            or continuity["dropped_fragment_count"] != 1 \
            or continuity["dropped_frame_count"] != 3 \
            or continuity["maximum_admitted_native_step_rad"] > 0.25 \
            or continuity["maximum_admitted_local_rotation_step_rad"] > 0.25:
        raise ValueError("terrain LMM continuity values changed")
    partition = normalized["temporal_partition"]
    if partition != {
        "boundary_row": 256, "derivatives": "per-source",
        "contacts": "per-source", "feature_future": "clamp-at-range-stop",
        "cross_range_operations": 0,
    }:
        raise ValueError("terrain LMM temporal partition changed")

    if normalized_validation["schema"] \
            != "g1-lmm-terrain-data-validation/v1" \
            or normalized_validation["database"] != {
                "frames": 851, "bones": 31, "contacts": 2,
                "ranges": [[0, 256], [256, 851]],
            } \
            or normalized_validation["features"] != {
                "rows": 851, "dimensions": 31,
                "terrain_indices": [27, 28, 29, 30],
                "terrain_scale": features.scale[27:31].tolist(),
            } \
            or normalized_validation["sidecars"] != {
                "terrain_features": {"rows": 851, "dimensions": 4},
                "terrain_support": {
                    "rows": 851, "dimensions": 3,
                    "columns": list(SUPPORT_COLUMNS),
                },
            } \
            or normalized_validation["partition"] != partition \
            or normalized_validation["inputs"] != {
                "flat_manifest_sha256": flat_data["manifest_sha256"],
                "slope_source_receipt_sha256": receipt_sha,
                "g1_xml_sha256": fixed_inputs["g1_xml"][1],
            }:
        raise ValueError("terrain LMM validation receipt changed")
    return normalized, normalized_validation


def _snapshot_lmm_terrain_scene_pack(scene_pack, terrain):
    try:
        index_json = scene_pack.index_json
        source_scenes = tuple(scene_pack.scenes)
    except (AttributeError, TypeError) as error:
        raise TypeError("terrain LMM scene pack is invalid") from error
    index = _parse_cached_json(index_json, "terrain LMM scene index")
    surface_signature = hashlib.sha256(canonical_json_bytes(terrain)).hexdigest()
    if type(index) is not dict or set(index) != _LMM_TERRAIN_INDEX_KEYS \
            or index["schema"] != "g1-lmm-terrain-scene-index/v1" \
            or index["default_scene_id"] != "authored-slope" \
            or index["scene_ids"] != ["authored-slope"] \
            or index["coordinate_signature"] != COORDINATE_SIGNATURE \
            or index["surface_signature"] != surface_signature \
            or len(source_scenes) != 1 \
            or source_scenes[0].scene_id != "authored-slope":
        raise ValueError("terrain LMM one-scene index changed")
    source = source_scenes[0]
    payloads = {
        "scene_json": source.scene_json,
        "terrain_bin": source.terrain_bin,
        "terrain_obj": source.terrain_obj,
        "walkability_bin": source.walkability_bin,
    }
    if any(type(payload) is not bytes or not payload
           for payload in payloads.values()):
        raise TypeError("terrain LMM scene payloads must be non-empty bytes")
    descriptor = index["scenes"]
    expected_descriptor = {
        "id": "authored-slope",
        "path": "scenes/authored-slope/scene.json",
        "schema": "g1-lmm-terrain-scene/v1",
        "size_bytes": len(payloads["scene_json"]),
        "sha256": hashlib.sha256(payloads["scene_json"]).hexdigest(),
    }
    if type(descriptor) is not list or descriptor != [expected_descriptor]:
        raise ValueError("terrain LMM scene index descriptor changed")
    metadata = _parse_cached_json(
        payloads["scene_json"], "terrain LMM authored scene")
    if type(metadata) is not dict or set(metadata) != _LMM_TERRAIN_SCENE_KEYS \
            or metadata["schema"] != "g1-lmm-terrain-scene/v1" \
            or metadata["id"] != "authored-slope" \
            or metadata["label"] != "Authenticated GRAIL Authored Slope" \
            or metadata["coordinate_signature"] != COORDINATE_SIGNATURE \
            or metadata["surface_signature"] != surface_signature \
            or metadata["terrain_feature_distances_m"] \
            != [0.25, 0.5, 0.75, 1.0]:
        raise ValueError("terrain LMM authored scene identity changed")
    receipt_sha = terrain["source_receipt_sha256"]
    if metadata["provenance"] != {
        "kind": "authenticated-grail",
        "source_ids": ["terrain_slopes__slope_000__000"],
        "parameters": {
            "receipt_schema": "g1-lmm-authored-slope-source/v1",
            "source_receipt_sha256": receipt_sha,
            "route_source": "admitted-holden-simulation-path",
            "surface_source": "authenticated-exact-mesh",
            "exterior_policy": "explicit-flat-zero-before-calibration",
            "support_calibration_m": 0.012000000104308128,
            "support_calibration_applied_to": "terrain-only",
        },
    }:
        raise ValueError("terrain LMM authored scene provenance changed")
    asset_specs = {
        "heightfield": (
            payloads["terrain_bin"], "terrain.bin", "G1HF/v2", {
                "path", "schema", "version", "nx", "nz", "origin_x",
                "origin_z", "cell_size_m", "exterior_height_m",
                "interpolation", "diagonal", "size_bytes", "sha256",
            }),
        "mesh": (
            payloads["terrain_obj"], "terrain.obj", "obj/v1",
            {"path", "schema", "size_bytes", "sha256"}),
        "walkability": (
            payloads["walkability_bin"], "walkability.bin", "G1WM/v1", {
                "path", "schema", "version", "nx", "nz", "classes",
                "size_bytes", "sha256",
            }),
    }
    for name, (payload, path, schema, keys) in asset_specs.items():
        asset = metadata[name]
        if type(asset) is not dict or set(asset) != keys \
                or asset["path"] != path or asset["schema"] != schema \
                or asset["size_bytes"] != len(payload) \
                or asset["sha256"] != hashlib.sha256(payload).hexdigest():
            raise ValueError(f"terrain LMM scene {name} descriptor changed")
    heightfield = metadata["heightfield"]
    if heightfield["version"] != 2 \
            or type(heightfield["nx"]) is not int or heightfield["nx"] < 2 \
            or type(heightfield["nz"]) is not int or heightfield["nz"] < 2 \
            or heightfield["interpolation"] != "fixed-diagonal-triangles" \
            or heightfield["diagonal"] \
            != "min-x-min-z_to_max-x-max-z" \
            or heightfield["exterior_height_m"] \
            != -0.012000000104308128:
        raise ValueError("terrain LMM heightfield contract changed")
    if metadata["walkability"].get("version") != 1 \
            or metadata["walkability"].get("classes") != {
                "blocked": 0, "certified": 1, "stress": 2,
            }:
        raise ValueError("terrain LMM walkability contract changed")

    def require_f32(values, shape, label):
        try:
            array = np.asarray(values, np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"terrain LMM {label} changed") from error
        if array.shape != shape or not np.isfinite(array).all() \
                or not np.array_equal(
                    array, array.astype(np.float32).astype(np.float64)):
            raise ValueError(f"terrain LMM {label} changed")
        return array

    for key in ("origin_x", "origin_z", "cell_size_m", "exterior_height_m"):
        require_f32(heightfield[key], (), f"heightfield {key}")
    bounds = metadata["bounds"]
    expected_bounds = {
        "mesh_min_xyz": 3, "mesh_max_xyz": 3,
        "heightfield_min_xyz": 3, "heightfield_max_xyz": 3,
        "playable_min_xz": 2, "playable_max_xz": 2,
        "lookahead_min_xz": 2, "lookahead_max_xz": 2,
    }
    if type(bounds) is not dict or set(bounds) != set(expected_bounds):
        raise ValueError("terrain LMM authored scene bounds changed")
    for name, length in expected_bounds.items():
        try:
            values = np.asarray(bounds[name], np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("terrain LMM authored scene bounds changed") \
                from error
        if values.shape != (length,) or not np.isfinite(values).all():
            raise ValueError("terrain LMM authored scene bounds changed")
    spawn = metadata["spawn"]
    if type(spawn) is not dict or set(spawn) != {"position", "yaw_radians"}:
        raise ValueError("terrain LMM authored spawn changed")
    spawn_position = require_f32(spawn["position"], (3,), "spawn position")
    require_f32(spawn["yaw_radians"], (), "spawn yaw")
    regions = metadata["regions"]
    if type(regions) is not dict or set(regions) != {
        "certified", "stress", "blocked",
    } or regions["stress"] != [] or regions["blocked"] != [] \
            or type(regions["certified"]) is not list \
            or len(regions["certified"]) != 1 \
            or set(regions["certified"][0]) != {"id", "bounds_xz"} \
            or regions["certified"][0]["id"] != "authored-route":
        raise ValueError("terrain LMM authored regions changed")
    require_f32(
        regions["certified"][0]["bounds_xz"], (4,),
        "certified region bounds")
    routes = metadata["routes"]
    if type(routes) is not list or len(routes) != 1 \
            or set(routes[0]) != {
                "id", "waypoints_xz", "expected_outcome",
                "walkability_class", "landing_hold_seconds",
            } or routes[0]["id"] != "authored-forward" \
            or len(routes[0]["waypoints_xz"]) != 5 \
            or routes[0]["expected_outcome"] != "traverse" \
            or routes[0]["walkability_class"] != 1 \
            or routes[0]["landing_hold_seconds"] != 0.0:
        raise ValueError("terrain LMM authored route changed")
    route_points = require_f32(
        routes[0]["waypoints_xz"], (5, 2), "authored route waypoints")
    if not np.array_equal(
            route_points[0], spawn_position[[0, 2]]):
        raise ValueError("terrain LMM route does not start at spawn")
    snapshot = _SceneSnapshot(
        "authored-slope", payloads["scene_json"], payloads["terrain_bin"],
        payloads["terrain_obj"], payloads["walkability_bin"], metadata)
    return _ScenePackSnapshot(index_json, index, (snapshot,))


def _lmm_root_descriptor(staging, path, schema):
    absolute = os.path.join(staging, path)
    return {
        "path": path,
        "schema": schema,
        "size_bytes": os.stat(absolute).st_size,
        "sha256": sha256_file(absolute),
    }


def _finalize_lmm_terrain_manifest(staging, manifest_base):
    manifest = dict(manifest_base)
    manifest["database"] = _lmm_root_descriptor(
        staging, "database.bin", "holden-database/v1")
    manifest["features"] = _lmm_root_descriptor(
        staging, "features.bin", "g1-lmm-features/v1")
    terrain_sidecar = _lmm_root_descriptor(
        staging, "terrain_features.bin", "G1TF/v1")
    terrain_sidecar.update({"version": 1, "dimensions": 4})
    support_sidecar = _lmm_root_descriptor(
        staging, "terrain_support.bin", "G1SP/v1")
    support_sidecar.update({
        "version": 1, "dimensions": 3, "columns": list(SUPPORT_COLUMNS),
    })
    manifest["sidecars"] = {
        "terrain_features": terrain_sidecar,
        "terrain_support": support_sidecar,
    }
    manifest["scene_index"] = _lmm_root_descriptor(
        staging, os.path.join("scenes", "index.json"),
        "g1-lmm-terrain-scene-index/v1")
    manifest["validation"] = _lmm_root_descriptor(
        staging, "validation.json", "g1-lmm-terrain-data-validation/v1")
    if set(manifest) != _LMM_TERRAIN_MANIFEST_KEYS:
        raise ValueError("terrain LMM final manifest key set changed")
    return manifest


def _validate_staged_lmm_terrain(
    staging, artifacts, features, manifest, validation, scene_pack,
):
    expected_files = {
        "database.bin", "features.bin", "terrain_features.bin",
        "terrain_support.bin", "validation.json", "manifest.json",
        os.path.join("scenes", "index.json"),
        os.path.join("scenes", "authored-slope", "scene.json"),
        os.path.join("scenes", "authored-slope", "terrain.bin"),
        os.path.join("scenes", "authored-slope", "terrain.obj"),
        os.path.join("scenes", "authored-slope", "walkability.bin"),
    }
    files, directories = _inspect_tree(staging)
    if files != expected_files or directories != {
        ".", "scenes", os.path.join("scenes", "authored-slope"),
    }:
        raise ValueError("terrain LMM staged tree changed")
    manifest_base = {
        key: manifest[key] for key in _LMM_TERRAIN_BASE_KEYS}
    normalized_base, normalized_validation = _normalized_lmm_terrain_inputs(
        manifest_base, validation, artifacts, features)
    trusted_scene_pack = _snapshot_lmm_terrain_scene_pack(
        scene_pack, normalized_base["terrain"])
    expected_bytes = {
        "manifest.json": canonical_json_bytes(manifest),
        "validation.json": canonical_json_bytes(normalized_validation),
        os.path.join("scenes", "index.json"): trusted_scene_pack.index_json,
    }
    scene = trusted_scene_pack.scenes[0]
    prefix = os.path.join("scenes", "authored-slope")
    expected_bytes.update({
        os.path.join(prefix, "scene.json"): scene.scene_json,
        os.path.join(prefix, "terrain.bin"): scene.terrain_bin,
        os.path.join(prefix, "terrain.obj"): scene.terrain_obj,
        os.path.join(prefix, "walkability.bin"): scene.walkability_bin,
    })
    for relative, expected in expected_bytes.items():
        if not _regular_file_matches_bytes(
                os.path.join(staging, relative), expected):
            raise ValueError(f"terrain LMM staged bytes changed: {relative}")
    descriptors = (
        (manifest["database"], "database.bin"),
        (manifest["features"], "features.bin"),
        (manifest["sidecars"]["terrain_features"], "terrain_features.bin"),
        (manifest["sidecars"]["terrain_support"], "terrain_support.bin"),
        (manifest["scene_index"], os.path.join("scenes", "index.json")),
        (manifest["validation"], "validation.json"),
    )
    for descriptor, expected_path in descriptors:
        path = os.path.join(staging, expected_path)
        if descriptor["path"] != expected_path \
                or descriptor["size_bytes"] != os.stat(path).st_size \
                or descriptor["sha256"] != sha256_file(path):
            raise ValueError(
                f"terrain LMM descriptor changed for {expected_path}")
    index_descriptor = trusted_scene_pack.index["scenes"][0]
    scene_path = os.path.join(staging, index_descriptor["path"])
    if index_descriptor["size_bytes"] != os.stat(scene_path).st_size \
            or index_descriptor["sha256"] != sha256_file(scene_path):
        raise ValueError("terrain LMM scene descriptor changed")
    for name, filename in (
        ("heightfield", "terrain.bin"), ("mesh", "terrain.obj"),
        ("walkability", "walkability.bin"),
    ):
        descriptor = scene.metadata[name]
        path = os.path.join(staging, prefix, filename)
        if descriptor["size_bytes"] != os.stat(path).st_size \
                or descriptor["sha256"] != sha256_file(path):
            raise ValueError(f"terrain LMM {name} asset changed")

    loaded_database = read_holden_database(os.path.join(staging, "database.bin"))
    loaded_features = read_features(os.path.join(staging, "features.bin"))
    loaded_database.terrain_features = read_terrain_sidecar(
        os.path.join(staging, "terrain_features.bin"))
    loaded_database.terrain_support = read_support_sidecar(
        os.path.join(staging, "terrain_support.bin"))
    loaded_database.validate()

    def bitwise_equal(actual, expected, dtype):
        actual = np.ascontiguousarray(actual, dtype=dtype)
        expected = np.ascontiguousarray(expected, dtype=dtype)
        if actual.shape != expected.shape:
            return False
        if dtype == "<f4":
            return np.array_equal(actual.view("<u4"), expected.view("<u4"))
        return np.array_equal(actual, expected)

    for name, dtype in (
        ("positions", "<f4"), ("velocities", "<f4"),
        ("rotations", "<f4"), ("angular_velocities", "<f4"),
        ("parents", "<i4"), ("range_starts", "<i4"),
        ("range_stops", "<i4"), ("contacts", "u1"),
        ("terrain_features", "<f4"), ("terrain_support", "<f4"),
    ):
        if not bitwise_equal(
                getattr(loaded_database, name), getattr(artifacts, name), dtype):
            raise ValueError(f"terrain LMM staged {name} changed")
    for name in ("values", "offset", "scale"):
        if not bitwise_equal(
                getattr(loaded_features, name), getattr(features, name), "<f4"):
            raise ValueError(f"terrain LMM staged feature {name} changed")


def publish_lmm_terrain_artifacts(
    output_dir, artifacts, features, manifest_base, validation, scene_pack,
    validate_candidate,
):
    if not callable(validate_candidate):
        raise TypeError("validate_candidate must be callable")
    manifest_base, validation = _normalized_lmm_terrain_inputs(
        manifest_base, validation, artifacts, features)
    scene_pack = _snapshot_lmm_terrain_scene_pack(
        scene_pack, manifest_base["terrain"])
    output_dir = os.path.abspath(os.fspath(output_dir))
    parent = os.path.dirname(output_dir)
    basename = os.path.basename(output_dir)
    if not basename:
        raise ValueError("output directory must have a basename")
    os.makedirs(parent, exist_ok=True)
    _output_exists_as_directory(output_dir)
    staging = tempfile.mkdtemp(prefix=f".{basename}.staging-", dir=parent)
    staging_identity = _scratch_directory_identity(staging)
    candidate_identity = staging_identity
    preserve_staging = False
    committed = False
    try:
        _write_database_exclusive(
            os.path.join(staging, "database.bin"), artifacts)
        _write_bytes_fsync(
            os.path.join(staging, "features.bin"), features_bytes(features))
        _write_bytes_fsync(
            os.path.join(staging, "terrain_features.bin"),
            terrain_sidecar_bytes(artifacts.terrain_features))
        _write_bytes_fsync(
            os.path.join(staging, "terrain_support.bin"),
            support_sidecar_bytes(artifacts.terrain_support))
        _write_scene_pack(staging, scene_pack)
        _write_bytes_fsync(
            os.path.join(staging, "validation.json"),
            canonical_json_bytes(validation))
        manifest = _finalize_lmm_terrain_manifest(staging, manifest_base)
        _write_bytes_fsync(
            os.path.join(staging, "manifest.json"),
            canonical_json_bytes(manifest))

        _validate_staged_lmm_terrain(
            staging, artifacts, features, manifest, validation, scene_pack)
        validate_candidate(staging)
        _validate_staged_lmm_terrain(
            staging, artifacts, features, manifest, validation, scene_pack)
        _fsync_tree(staging)
        with _locked_parent(parent) as parent_descriptor:
            _validate_staged_lmm_terrain(
                staging, artifacts, features, manifest, validation, scene_pack)
            output_identity = _output_directory_identity(output_dir)
            exchanged = False
            if output_identity is not None:
                _rename_exchange(staging, output_dir)
                exchanged = True
                staging_identity = output_identity
            else:
                os.replace(staging, output_dir)
                staging_identity = None
            try:
                _fsync_parent(parent_descriptor)
            except BaseException as commit_error:
                try:
                    if exchanged:
                        _rename_exchange(staging, output_dir)
                    else:
                        os.replace(output_dir, staging)
                    staging_identity = candidate_identity
                    _fsync_parent(parent_descriptor)
                except BaseException as rollback_error:
                    preserve_staging = True
                    raise PublicationRollbackError(
                        output_dir, staging, commit_error, rollback_error,
                    ) from rollback_error
                raise
            committed = True
            if exchanged:
                try:
                    _remove_owned_scratch(staging, staging_identity)
                except BaseException as cleanup_error:
                    preserve_staging = True
                    raise PublicationCommittedError(
                        output_dir, staging, cleanup_error,
                    ) from cleanup_error
        return manifest
    finally:
        if not committed and not preserve_staging:
            _remove_owned_scratch(staging, staging_identity)
