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
from .schema import ArtifactSet


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
_LEGACY_DISPATCH = object()

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


def _normalized_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if not isinstance(manifest.get("validation"), dict):
        raise ValueError("manifest must contain a validation object")
    try:
        encoded = json.dumps(
            manifest, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"manifest must contain finite valid JSON: {error}"
        ) from error
    return normalized


def _load_staged_json(path: str, expected: dict) -> None:
    try:
        with open(path, encoding="utf-8") as stream:
            actual = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid staged {os.path.basename(path)}") from error
    if actual != expected:
        raise ValueError(
            f"staged {os.path.basename(path)} does not match requested data"
        )


def _validate_staged_artifacts(
    staging: str, artifacts: ArtifactSet, manifest: dict
) -> None:
    for required in ("terrain.bin", "terrain.obj"):
        if not os.path.isfile(os.path.join(staging, required)):
            raise ValueError(f"terrain writer did not create {required}")

    loaded = read_holden_database(os.path.join(staging, "database.bin"))
    loaded.terrain_features = read_terrain_sidecar(
        os.path.join(staging, "terrain_features.bin")
    )
    loaded.terrain_support = read_support_sidecar(
        os.path.join(staging, "terrain_support.bin")
    )
    loaded.validate()

    expected_arrays = (
        ("positions", "<f4"),
        ("velocities", "<f4"),
        ("rotations", "<f4"),
        ("angular_velocities", "<f4"),
        ("parents", "<i4"),
        ("range_starts", "<i4"),
        ("range_stops", "<i4"),
        ("contacts", "u1"),
        ("terrain_features", "<f4"),
        ("terrain_support", "<f4"),
    )
    for name, dtype in expected_arrays:
        expected = np.ascontiguousarray(getattr(artifacts, name), dtype=dtype)
        if not np.array_equal(getattr(loaded, name), expected):
            raise ValueError(f"staged {name} does not match requested artifacts")

    _load_staged_json(os.path.join(staging, "manifest.json"), manifest)
    _load_staged_json(
        os.path.join(staging, "validation.json"), manifest["validation"]
    )


def _remove_path(path: str) -> None:
    if not os.path.lexists(path):
        return
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.unlink(path)


def _publish_artifacts_v1(
    output_dir: os.PathLike | str,
    artifacts: ArtifactSet,
    manifest: dict,
    terrain_writer,
) -> None:
    manifest = _normalized_manifest(manifest)
    if not callable(terrain_writer):
        raise TypeError("terrain_writer must be callable")

    output_dir = os.path.abspath(os.fspath(output_dir))
    parent = os.path.dirname(output_dir)
    os.makedirs(parent, exist_ok=True)
    backup = output_dir + ".previous"
    if not os.path.lexists(output_dir) and os.path.lexists(backup):
        os.replace(backup, output_dir)
    else:
        _remove_path(backup)
    staging = tempfile.mkdtemp(prefix=".g1_terrain-", dir=parent)
    previous_moved = False
    try:
        write_holden_database(os.path.join(staging, "database.bin"), artifacts)
        write_terrain_sidecar(
            os.path.join(staging, "terrain_features.bin"),
            artifacts.terrain_features,
        )
        write_support_sidecar(
            os.path.join(staging, "terrain_support.bin"),
            artifacts.terrain_support,
        )
        with open(
            os.path.join(staging, "manifest.json"), "w", encoding="utf-8"
        ) as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        with open(
            os.path.join(staging, "validation.json"), "w", encoding="utf-8"
        ) as stream:
            json.dump(
                manifest["validation"],
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )

        terrain_writer(staging)
        _validate_staged_artifacts(staging, artifacts, manifest)

        if os.path.lexists(output_dir):
            os.replace(output_dir, backup)
            previous_moved = True
        try:
            os.replace(staging, output_dir)
        except Exception:
            if previous_moved and not os.path.lexists(output_dir):
                os.replace(backup, output_dir)
                previous_moved = False
            raise
        staging = ""
        if previous_moved:
            _remove_path(backup)
            previous_moved = False
    except Exception:
        if (
            previous_moved
            and os.path.lexists(backup)
            and not os.path.lexists(output_dir)
        ):
            os.replace(backup, output_dir)
        raise
    finally:
        if staging and os.path.lexists(staging):
            _remove_path(staging)


def _publish_artifacts_v2(
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


def publish_artifacts(
    output_dir, artifacts, manifest_base, scene_pack,
    validate_candidate=_LEGACY_DISPATCH,
):
    """Publish v2, with a temporary positional four-value v1 bridge.

    The bridge exists only for the repository's current positional v1 caller.
    Keyword names intentionally follow the five-argument v2 API and do not
    preserve the retired ``manifest=``/``terrain_writer=`` names.
    """
    if validate_candidate is _LEGACY_DISPATCH:
        if not callable(scene_pack):
            raise TypeError(
                "four-argument legacy publication requires a terrain writer")
        return _publish_artifacts_v1(
            output_dir, artifacts, manifest_base, scene_pack)
    return _publish_artifacts_v2(
        output_dir, artifacts, manifest_base, scene_pack,
        validate_candidate)
