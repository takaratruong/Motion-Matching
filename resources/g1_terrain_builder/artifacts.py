import json
import os
import shutil
import struct
import tempfile

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


def publish_artifacts(
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
