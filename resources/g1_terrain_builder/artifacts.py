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
_HEADER = struct.Struct("<4sIII")
_UINT32_MAX = (1 << 32) - 1


def write_terrain_sidecar(
    path: os.PathLike | str, features: np.ndarray
) -> None:
    try:
        values = np.ascontiguousarray(features, dtype="<f4")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "terrain features must be finite (N, 4) float values"
        ) from error
    if (
        values.ndim != 2
        or len(values) < 1
        or values.shape[1] != DIMS
        or len(values) > _UINT32_MAX
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            f"terrain features must be finite (N, 4), got {values.shape}"
        )
    with open(path, "wb") as stream:
        stream.write(_HEADER.pack(MAGIC, VERSION, len(values), DIMS))
        stream.write(values.tobytes())


def read_terrain_sidecar(path: os.PathLike | str) -> np.ndarray:
    with open(path, "rb") as stream:
        data = stream.read()
    if len(data) < _HEADER.size:
        raise ValueError(f"{path}: truncated terrain sidecar header")
    magic, version, frames, dims = _HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION or dims != DIMS:
        raise ValueError(f"{path}: unsupported terrain sidecar schema")
    expected = _HEADER.size + frames * dims * np.dtype("<f4").itemsize
    if len(data) < expected:
        raise ValueError(f"{path}: truncated terrain sidecar payload")
    if len(data) > expected:
        raise ValueError(f"{path}: trailing terrain sidecar bytes")
    if frames < 1:
        raise ValueError(f"{path}: invalid terrain sidecar frame count")
    values = np.frombuffer(
        data, dtype="<f4", count=frames * dims, offset=_HEADER.size
    ).reshape(frames, dims).copy()
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: terrain sidecar values must be finite")
    return values


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
    _remove_path(backup)
    staging = tempfile.mkdtemp(prefix=".g1_terrain-", dir=parent)
    previous_moved = False
    try:
        write_holden_database(os.path.join(staging, "database.bin"), artifacts)
        write_terrain_sidecar(
            os.path.join(staging, "terrain_features.bin"),
            artifacts.terrain_features,
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
