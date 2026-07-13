#!/usr/bin/env python3
"""Independently reload and validate a published G1 terrain artifact set."""

import argparse
import hashlib
import json
import os
import re
import struct
import sys

import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from resources.g1_terrain_builder.artifacts import read_terrain_sidecar
from resources.g1_terrain_builder.database import read_holden_database


SCHEMA = "g1-terrain-artifacts/v1"
OUTPUT_FPS = 25.0
FEATURE_DIMENSIONS = 31
TERRAIN_DIMENSIONS = 4
_HEIGHTFIELD_HEADER = struct.Struct("<4sIII4f")
_OBJ_INDEX = re.compile(r"[1-9][0-9]*\Z")


def _require(condition: bool, contract: str) -> None:
    if not condition:
        raise ValueError(contract)


def _read_json(path: str):
    def object_without_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=object_without_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value {value}")),
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{os.path.basename(path)} is invalid JSON: {error}") \
            from error


def _integer(value, label: str, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    _require(value >= minimum, f"{label} must be at least {minimum}")
    return value


def _finite_number(value, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    value = float(value)
    _require(np.isfinite(value), f"{label} must be finite")
    return value


def _manifest_signature(names: list[str], parents: list[int]) -> str:
    encoded = json.dumps(
        {"names": names, "parents": parents},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_database(path: str):
    try:
        return read_holden_database(path)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"database.bin is invalid: {error}") from error


def _load_terrain_features(path: str) -> np.ndarray:
    try:
        return read_terrain_sidecar(path)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"terrain_features.bin is invalid: {error}") from error


def _validate_manifest_header(manifest: dict) -> None:
    _require(isinstance(manifest, dict), "manifest.json root must be an object")
    _require(manifest.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    _require(
        manifest.get("output_fps") == OUTPUT_FPS,
        f"output_fps must be {OUTPUT_FPS}",
    )
    _require(
        manifest.get("feature_dimensions") == FEATURE_DIMENSIONS,
        f"feature_dimensions must be {FEATURE_DIMENSIONS}",
    )
    _require(
        manifest.get("terrain_dimensions") == TERRAIN_DIMENSIONS,
        f"terrain_dimensions must be {TERRAIN_DIMENSIONS}",
    )


def _validate_skeleton(manifest: dict, database) -> tuple[list[str], list[int]]:
    skeleton = manifest.get("skeleton")
    _require(isinstance(skeleton, dict), "skeleton must be an object")
    names = skeleton.get("names")
    parents = skeleton.get("parents")
    _require(isinstance(names, list), "skeleton names must be a list")
    _require(
        len(names) == FEATURE_DIMENSIONS,
        f"skeleton names must contain {FEATURE_DIMENSIONS} bones",
    )
    _require(
        all(isinstance(name, str) and name for name in names),
        "skeleton names must be non-empty strings",
    )
    _require(len(set(names)) == len(names), "skeleton names must be unique")
    _require(names[0] == "Simulation", "skeleton first bone must be Simulation")
    _require(isinstance(parents, list), "skeleton parents must be a list")
    _require(
        len(parents) == len(names)
        and all(isinstance(value, int) and not isinstance(value, bool)
                for value in parents),
        "skeleton parents must contain one integer per bone",
    )
    database_parents = np.asarray(database.parents, np.int64).tolist()
    _require(
        parents == database_parents,
        "skeleton parents do not match database.bin",
    )
    expected_signature = _manifest_signature(names, parents)
    _require(
        skeleton.get("signature") == expected_signature,
        "skeleton signature does not match names and parents",
    )
    _require(
        database.positions.shape[1] == len(names),
        "database.bin bone count does not match skeleton names",
    )
    return names, parents


def _expected_output_frames(source_frames: int, source_fps: float) -> int:
    duration = (source_frames - 1) / source_fps
    return int(np.floor(duration * OUTPUT_FPS + 1e-9)) + 1


def _validate_sources(manifest: dict, database) -> list[dict]:
    sources = manifest.get("sources")
    _require(isinstance(sources, list) and sources, "sources must be non-empty")
    total_clips = _integer(manifest.get("total_clips"), "total_clips", 1)
    grail_clips = _integer(manifest.get("grail_clips"), "grail_clips", 0)
    skipped_clips = _integer(manifest.get("skipped_clips"), "skipped_clips", 0)
    database_frames = _integer(
        manifest.get("database_frames"), "database_frames", 1)
    _require(skipped_clips == 0, "skipped_clips must be zero")
    _require(total_clips == len(sources), "total_clips does not match sources")
    _require(
        grail_clips == total_clips - 1,
        "grail_clips must count every source after Takara",
    )
    _require(
        database_frames == len(database.positions),
        "database_frames does not match database.bin",
    )
    _require(
        isinstance(manifest.get("diagnostic_mode"), bool),
        "diagnostic_mode must be boolean",
    )
    _require(
        len(database.range_starts) == total_clips,
        "database.bin range count does not match total_clips",
    )

    cursor = 0
    seen_names = set()
    for index, source in enumerate(sources):
        label = f"sources[{index}]"
        _require(isinstance(source, dict), f"{label} must be an object")
        name = source.get("name")
        terrain_id = source.get("terrain_id")
        _require(
            isinstance(name, str) and name,
            f"{label} name must be a non-empty string",
        )
        _require(name not in seen_names, f"duplicate source name {name}")
        seen_names.add(name)
        _require(
            isinstance(terrain_id, str) and terrain_id,
            f"{label} terrain_id must be a non-empty string",
        )
        if index == 0:
            _require(name == "takara_walk_50hz", "first source must be Takara")
            _require(terrain_id == "flat", "Takara terrain_id must be flat")
        else:
            _require(terrain_id == name, f"{label} terrain_id must match name")

        source_fps = _finite_number(source.get("source_fps"), f"{label} source_fps")
        _require(source_fps > 0.0, f"{label} source_fps must be positive")
        source_frames = _integer(
            source.get("source_frames"), f"{label} source_frames", 1)
        output_frames = _integer(
            source.get("output_frames"), f"{label} output_frames", 1)
        range_start = _integer(
            source.get("range_start"), f"{label} range_start", 0)
        range_stop = _integer(
            source.get("range_stop"), f"{label} range_stop", 1)
        _require(
            output_frames == _expected_output_frames(source_frames, source_fps),
            f"{label} output_frames violates the 25 Hz duration contract",
        )
        _require(range_start == cursor, f"{label} range_start is not contiguous")
        _require(
            range_stop == range_start + output_frames,
            f"{label} range_stop does not match output_frames",
        )
        _require(
            int(database.range_starts[index]) == range_start
            and int(database.range_stops[index]) == range_stop,
            f"{label} range does not match database.bin",
        )

        source_map = source.get("source_frame_map")
        _require(
            isinstance(source_map, list)
            and all(isinstance(value, int) and not isinstance(value, bool)
                    for value in source_map),
            f"{label} source frame map must contain integer indices",
        )
        _require(
            len(source_map) == output_frames,
            f"{label} source frame map length does not match output_frames",
        )
        output_t = np.arange(output_frames, dtype=np.float64) / OUTPUT_FPS
        expected_map = np.rint(output_t * source_fps).astype(np.int64)
        expected_map = np.clip(expected_map, 0, source_frames - 1).tolist()
        _require(
            source_map == expected_map,
            f"{label} source frame map content violates 25 Hz provenance",
        )
        cursor = range_stop

    _require(
        cursor == len(database.positions),
        "source ranges do not cover database.bin frames",
    )
    return sources


def _validate_parameters(manifest: dict, clip_count: int) -> dict:
    contact = manifest.get("contact")
    _require(isinstance(contact, dict), "contact must be an object")
    _require(
        contact.get("speed_threshold") == 0.15,
        "contact speed_threshold must be 0.15",
    )
    _require(
        contact.get("height_threshold") == 0.06,
        "contact height_threshold must be 0.06",
    )
    _require(
        contact.get("median_filter_frames") == 3,
        "contact median_filter_frames must be 3",
    )

    terrain = manifest.get("terrain")
    _require(isinstance(terrain, dict), "terrain must be an object")
    _require(
        terrain.get("distances_m") == [0.25, 0.50, 0.75, 1.00],
        "terrain distances_m must be [0.25, 0.5, 0.75, 1.0]",
    )
    _require(
        terrain.get("coordinate_mapping")
        == "mujoco_xyz_to_holden_x_z_neg_y",
        "terrain coordinate_mapping is unsupported",
    )
    _require(
        isinstance(terrain.get("runtime_base"), str)
        and bool(terrain["runtime_base"]),
        "terrain runtime_base must be a non-empty string",
    )
    _require(
        terrain.get("cell_size_m") == 0.02,
        "terrain cell_size_m must be 0.02",
    )
    _require(
        terrain.get("border_m") == 2.0,
        "terrain border_m must be 2.0",
    )

    validation = manifest.get("validation")
    _require(isinstance(validation, dict), "validation must be an object")
    limits = {
        "fk_max_error_m": 0.001,
        "duration_error_s": 1.0 / OUTPUT_FPS + 1e-12,
        "quaternion_norm_max_error": 1e-4,
    }
    for name, limit in limits.items():
        values = validation.get(name)
        _require(
            isinstance(values, list) and len(values) == clip_count,
            f"validation {name} must contain one value per clip",
        )
        numbers = [
            _finite_number(value, f"validation {name}[{index}]")
            for index, value in enumerate(values)
        ]
        _require(
            all(0.0 <= value <= limit for value in numbers),
            f"validation {name} exceeds {limit}",
        )
    return terrain


def _parse_heightfield(
    path: str, metadata: dict, schema_cell_size: float,
) -> tuple[float, float, float, float]:
    try:
        with open(path, "rb") as stream:
            payload = stream.read()
    except OSError as error:
        raise ValueError(f"terrain.bin cannot be read: {error}") from error
    _require(
        len(payload) >= _HEIGHTFIELD_HEADER.size,
        "terrain.bin has a truncated G1HF header",
    )
    magic, version, nx, nz, origin_x, origin_z, cell_size, exterior = (
        _HEIGHTFIELD_HEADER.unpack_from(payload)
    )
    _require(magic == b"G1HF", "terrain.bin magic must be G1HF")
    _require(version == 1, "terrain.bin G1HF version must be 1")
    _require(nx >= 2 and nz >= 2, "terrain.bin G1HF dimensions must be >= 2")
    header_values = [origin_x, origin_z, cell_size, exterior]
    _require(
        np.all(np.isfinite(header_values)),
        "terrain.bin G1HF header values must be finite",
    )
    _require(cell_size > 0.0, "terrain.bin G1HF cell size must be positive")
    expected_size = _HEIGHTFIELD_HEADER.size + nx * nz * 4
    _require(
        len(payload) == expected_size,
        "terrain.bin G1HF payload length does not match nx*nz float32 values",
    )
    heights = np.frombuffer(
        payload,
        dtype="<f4",
        count=nx * nz,
        offset=_HEIGHTFIELD_HEADER.size,
    )
    _require(
        np.all(np.isfinite(heights)),
        "terrain.bin G1HF heights must be finite",
    )

    _require(isinstance(metadata, dict), "terrain heightfield must be an object")
    expected_keys = {
        "nx", "nz", "origin_x", "origin_z", "cell_size", "exterior_height",
    }
    _require(
        set(metadata) == expected_keys,
        "terrain heightfield metadata fields are incomplete",
    )
    _require(metadata.get("nx") == nx, "terrain heightfield nx mismatch")
    _require(metadata.get("nz") == nz, "terrain heightfield nz mismatch")
    comparisons = (
        ("origin_x", origin_x),
        ("origin_z", origin_z),
        ("cell_size", cell_size),
        ("exterior_height", exterior),
    )
    for key, value in comparisons:
        expected = _finite_number(
            metadata.get(key), f"terrain heightfield {key}")
        _require(
            np.isclose(expected, value, rtol=1e-6, atol=1e-6),
            f"terrain heightfield {key} mismatch",
        )
    _require(
        metadata["cell_size"] == schema_cell_size
        and cell_size == float(np.float32(schema_cell_size)),
        "terrain heightfield cell size must match terrain cell_size_m",
    )
    _require(
        metadata["exterior_height"] == 0.0 and exterior == 0.0,
        "terrain heightfield exterior height must be 0.0",
    )
    return (
        float(origin_x),
        float(origin_x + (nx - 1) * cell_size),
        float(origin_z),
        float(origin_z + (nz - 1) * cell_size),
    )


def _parse_obj(path: str) -> tuple[float, float, float, float]:
    try:
        with open(path, encoding="utf-8") as stream:
            lines = stream.read().splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"terrain.obj cannot be read: {error}") from error

    vertices = []
    faces = []
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if fields[0] == "v":
            _require(
                len(fields) == 4,
                f"terrain.obj line {number}: vertex must contain three values",
            )
            try:
                vertex = tuple(float(value) for value in fields[1:])
            except ValueError as error:
                raise ValueError(
                    f"terrain.obj line {number}: invalid vertex") from error
            _require(
                np.all(np.isfinite(vertex)),
                f"terrain.obj line {number}: vertex must be finite",
            )
            vertices.append(vertex)
        elif fields[0] == "f":
            _require(
                len(fields) >= 4,
                f"terrain.obj line {number}: face needs at least three vertices",
            )
            _require(
                all(_OBJ_INDEX.fullmatch(value) for value in fields[1:]),
                f"terrain.obj line {number}: faces require one-based indices",
            )
            face = tuple(int(value) for value in fields[1:])
            _require(
                len(set(face)) >= 3,
                f"terrain.obj line {number}: face is degenerate",
            )
            faces.append((number, face))
        else:
            raise ValueError(
                f"terrain.obj line {number}: malformed record {fields[0]!r}")
    _require(vertices, "terrain.obj must contain vertices")
    _require(faces, "terrain.obj must contain faces")
    for number, face in faces:
        _require(
            max(face) <= len(vertices),
            f"terrain.obj line {number}: face index exceeds vertex count",
        )
    positions = np.asarray(vertices, np.float64)
    return (
        float(positions[:, 0].min()),
        float(positions[:, 0].max()),
        float(positions[:, 2].min()),
        float(positions[:, 2].max()),
    )


def _validate_terrain_coverage(
    heightfield_bounds: tuple[float, float, float, float],
    obj_bounds: tuple[float, float, float, float],
    border: float,
) -> None:
    scale = max(
        1.0,
        abs(border),
        *(abs(value) for value in heightfield_bounds),
        *(abs(value) for value in obj_bounds),
    )
    tolerance = 8.0 * float(np.finfo(np.float32).eps) * scale
    hxmin, hxmax, hzmin, hzmax = heightfield_bounds
    oxmin, oxmax, ozmin, ozmax = obj_bounds
    _require(
        hxmin <= oxmin - border + tolerance
        and hxmax >= oxmax + border - tolerance
        and hzmin <= ozmin - border + tolerance
        and hzmax >= ozmax + border - tolerance,
        "terrain.bin domain does not cover terrain.obj XZ bounds plus "
        "terrain.border_m",
    )


def validate_artifact_directory(artifact_dir: str) -> dict:
    artifact_dir = os.path.abspath(os.fspath(artifact_dir))
    _require(os.path.isdir(artifact_dir), "artifact directory does not exist")
    required = (
        "database.bin",
        "terrain_features.bin",
        "manifest.json",
        "validation.json",
        "terrain.bin",
        "terrain.obj",
    )
    for filename in required:
        _require(
            os.path.isfile(os.path.join(artifact_dir, filename)),
            f"missing {filename}",
        )

    manifest = _read_json(os.path.join(artifact_dir, "manifest.json"))
    validation_file = _read_json(os.path.join(artifact_dir, "validation.json"))
    _validate_manifest_header(manifest)
    _require(
        validation_file == manifest.get("validation"),
        "validation.json does not match manifest validation",
    )

    database = _load_database(os.path.join(artifact_dir, "database.bin"))
    terrain_features = _load_terrain_features(
        os.path.join(artifact_dir, "terrain_features.bin"))
    database.terrain_features = terrain_features
    try:
        database.validate()
    except (TypeError, ValueError) as error:
        raise ValueError(f"ArtifactSet validation failed: {error}") from error
    _require(
        len(database.positions) == len(terrain_features),
        "database.bin and terrain_features.bin frame counts differ",
    )
    _require(
        terrain_features.shape[1] == TERRAIN_DIMENSIONS,
        "terrain_features.bin dimension count must be 4",
    )
    names, _ = _validate_skeleton(manifest, database)
    sources = _validate_sources(manifest, database)
    terrain = _validate_parameters(manifest, len(sources))

    quaternion_error = float(np.max(np.abs(
        np.linalg.norm(database.rotations, axis=-1) - 1.0)))
    _require(
        quaternion_error <= 1e-4,
        f"database.bin quaternion norm error {quaternion_error} exceeds 0.0001",
    )
    heightfield_bounds = _parse_heightfield(
        os.path.join(artifact_dir, "terrain.bin"),
        terrain.get("heightfield"),
        float(terrain["cell_size_m"]),
    )
    obj_bounds = _parse_obj(os.path.join(artifact_dir, "terrain.obj"))
    _validate_terrain_coverage(
        heightfield_bounds, obj_bounds, float(terrain["border_m"]))
    return {
        "frames": len(database.positions),
        "clips": len(sources),
        "bones": len(names),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate published Holden G1 terrain motion artifacts")
    parser.add_argument("artifact_directory")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    path = os.path.abspath(args.artifact_directory)
    try:
        summary = validate_artifact_directory(path)
    except Exception as error:
        print(f"INVALID {path}: {error}", file=sys.stderr)
        return 1
    print(
        f"VALID {SCHEMA} frames={summary['frames']} clips={summary['clips']} "
        f"bones={summary['bones']} terrain_dims={TERRAIN_DIMENSIONS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
