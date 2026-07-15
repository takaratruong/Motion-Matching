"""Deterministically retime the authoritative flat locomotion database.

The playable flat controller historically consumed a 60 Hz, 23-bone Holden
database.  This module preserves that binary as an explicit source artifact and
derives the 25 Hz runtime database range-by-range.  Ranges are never blended
together: endpoints are retained, rotations take the shortest arc, velocities
are recomputed in units per second, and contacts use a documented nearest-source
sample rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional

import numpy as np


SOURCE_FPS = 60.0
TARGET_FPS = 25.0
AUTHORITATIVE_SOURCE_SHA256 = (
    "1212e97aee2c1bdf80caf60d9bf8c151d4c2d4631b285dc88a7ccb756daeb4d8"
)


@dataclass
class FlatDatabase:
    bone_positions: np.ndarray
    bone_velocities: np.ndarray
    bone_rotations: np.ndarray
    bone_angular_velocities: np.ndarray
    bone_parents: np.ndarray
    range_starts: np.ndarray
    range_stops: np.ndarray
    contact_states: np.ndarray


def _read_exact(stream: BinaryIO, byte_count: int, label: str) -> bytes:
    data = stream.read(byte_count)
    if len(data) != byte_count:
        raise ValueError(
            f"truncated flat database while reading {label}: "
            f"expected {byte_count} bytes, got {len(data)}")
    return data


def _read_u32(stream: BinaryIO, label: str) -> int:
    return struct.unpack("<I", _read_exact(stream, 4, label))[0]


def _read_shape(stream: BinaryIO, label: str) -> tuple[int, int]:
    return struct.unpack("<II", _read_exact(stream, 8, label))


def _read_float_array(
        stream: BinaryIO,
        shape: tuple[int, ...],
        label: str) -> np.ndarray:
    count = math.prod(shape)
    data = _read_exact(stream, count * 4, label)
    return np.frombuffer(data, dtype="<f4", count=count).reshape(shape).copy()


def _read_int_array(
        stream: BinaryIO,
        count: int,
        label: str) -> np.ndarray:
    data = _read_exact(stream, count * 4, label)
    return np.frombuffer(data, dtype="<i4", count=count).copy()


def read_database(path: Path | str) -> FlatDatabase:
    """Read the legacy Holden flat-database binary with strict shape checks."""
    path = Path(path)
    with path.open("rb") as stream:
        frame_count, bone_count = _read_shape(stream, "bone_positions shape")
        positions = _read_float_array(
            stream, (frame_count, bone_count, 3), "bone_positions")

        velocity_frames, velocity_bones = _read_shape(
            stream, "bone_velocities shape")
        velocities = _read_float_array(
            stream,
            (velocity_frames, velocity_bones, 3),
            "bone_velocities")

        rotation_frames, rotation_bones = _read_shape(
            stream, "bone_rotations shape")
        rotations = _read_float_array(
            stream,
            (rotation_frames, rotation_bones, 4),
            "bone_rotations")

        angular_frames, angular_bones = _read_shape(
            stream, "bone_angular_velocities shape")
        angular_velocities = _read_float_array(
            stream,
            (angular_frames, angular_bones, 3),
            "bone_angular_velocities")

        parent_count = _read_u32(stream, "bone_parents count")
        parents = _read_int_array(stream, parent_count, "bone_parents")

        start_count = _read_u32(stream, "range_starts count")
        starts = _read_int_array(stream, start_count, "range_starts")
        stop_count = _read_u32(stream, "range_stops count")
        stops = _read_int_array(stream, stop_count, "range_stops")

        contact_frames, contact_count = _read_shape(
            stream, "contact_states shape")
        contacts = np.frombuffer(
            _read_exact(
                stream,
                contact_frames * contact_count,
                "contact_states"),
            dtype=np.uint8,
            count=contact_frames * contact_count).reshape(
                contact_frames, contact_count).copy()

        if stream.read(1):
            raise ValueError("flat database has trailing bytes")

    database = FlatDatabase(
        bone_positions=positions,
        bone_velocities=velocities,
        bone_rotations=rotations,
        bone_angular_velocities=angular_velocities,
        bone_parents=parents,
        range_starts=starts,
        range_stops=stops,
        contact_states=contacts,
    )
    _validate_database(database)
    return database


def _as_little_endian_bytes(values: np.ndarray, dtype: str) -> bytes:
    return np.asarray(values, dtype=dtype).ravel(order="C").tobytes(order="C")


def write_database(path: Path | str, database: FlatDatabase) -> None:
    """Write a deterministic little-endian Holden flat-database binary."""
    _validate_database(database)
    path = Path(path)
    frame_count, bone_count = database.bone_positions.shape[:2]
    contact_count = database.contact_states.shape[1]
    with path.open("wb") as stream:
        for values, components in (
                (database.bone_positions, 3),
                (database.bone_velocities, 3),
                (database.bone_rotations, 4),
                (database.bone_angular_velocities, 3)):
            stream.write(struct.pack("<II", frame_count, bone_count))
            expected_shape = (frame_count, bone_count, components)
            if values.shape != expected_shape:
                raise ValueError(
                    f"array shape changed during write: expected "
                    f"{expected_shape}, got {values.shape}")
            stream.write(_as_little_endian_bytes(values, "<f4"))

        stream.write(struct.pack("<I", bone_count))
        stream.write(_as_little_endian_bytes(database.bone_parents, "<i4"))
        stream.write(struct.pack("<I", len(database.range_starts)))
        stream.write(_as_little_endian_bytes(database.range_starts, "<i4"))
        stream.write(struct.pack("<I", len(database.range_stops)))
        stream.write(_as_little_endian_bytes(database.range_stops, "<i4"))
        stream.write(struct.pack("<II", frame_count, contact_count))
        stream.write(_as_little_endian_bytes(database.contact_states, "u1"))


def _validate_database(database: FlatDatabase) -> None:
    positions = np.asarray(database.bone_positions)
    velocities = np.asarray(database.bone_velocities)
    rotations = np.asarray(database.bone_rotations)
    angular_velocities = np.asarray(database.bone_angular_velocities)
    parents = np.asarray(database.bone_parents)
    starts = np.asarray(database.range_starts)
    stops = np.asarray(database.range_stops)
    contacts = np.asarray(database.contact_states)

    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("bone_positions must have shape [frames, bones, 3]")
    frame_count, bone_count = positions.shape[:2]
    if frame_count < 1 or bone_count < 1:
        raise ValueError("flat database must contain frames and bones")
    expected_vector_shape = (frame_count, bone_count, 3)
    if velocities.shape != expected_vector_shape:
        raise ValueError("bone_velocities shape does not match bone_positions")
    if rotations.shape != (frame_count, bone_count, 4):
        raise ValueError("bone_rotations shape does not match bone_positions")
    if angular_velocities.shape != expected_vector_shape:
        raise ValueError(
            "bone_angular_velocities shape does not match bone_positions")
    if parents.shape != (bone_count,):
        raise ValueError("bone_parents count does not match bone count")
    if contacts.ndim != 2 or contacts.shape[0] != frame_count:
        raise ValueError("contact_states frame count does not match poses")
    if starts.ndim != 1 or stops.shape != starts.shape or len(starts) < 1:
        raise ValueError("range start/stop arrays must be non-empty and equal")
    if starts[0] != 0 or stops[-1] != frame_count:
        raise ValueError("ranges must cover every frame")
    if np.any(starts >= stops):
        raise ValueError("every range must contain at least one frame")
    if len(starts) > 1 and np.any(starts[1:] != stops[:-1]):
        raise ValueError("ranges must be contiguous")
    if not np.all(np.isfinite(positions)):
        raise ValueError("bone_positions must be finite")
    if not np.all(np.isfinite(velocities)):
        raise ValueError("bone_velocities must be finite")
    if not np.all(np.isfinite(rotations)):
        raise ValueError("bone_rotations must be finite")
    if not np.all(np.isfinite(angular_velocities)):
        raise ValueError("bone_angular_velocities must be finite")
    if np.any(np.linalg.norm(rotations, axis=-1) < 1e-8):
        raise ValueError("bone_rotations must be nonzero")
    if np.any((contacts != 0) & (contacts != 1)):
        raise ValueError("contact_states must contain only zero or one")


def _output_frame_count(
        source_frame_count: int,
        source_fps: float,
        target_fps: float) -> int:
    if source_frame_count < 1 or source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("frame count and sample rates must be positive")
    if source_frame_count == 1:
        return 1
    source_duration = (source_frame_count - 1) / source_fps
    # Nearest integer number of target intervals, with an explicit half-up
    # tie rule so output is independent of Python's banker-rounding behavior.
    target_intervals = int(math.floor(
        source_duration * target_fps + 0.5 + 1e-12))
    return max(2, target_intervals + 1)


def _source_coordinates(source_count: int, target_count: int) -> np.ndarray:
    if target_count == 1:
        return np.zeros(1, dtype=np.float64)
    return np.linspace(
        0.0, float(source_count - 1), target_count, dtype=np.float64)


def _interpolate_vectors(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, len(source) - 1)
    alpha = (coordinates - lower).reshape((-1,) + (1,) * (source.ndim - 1))
    output = (1.0 - alpha) * source[lower] + alpha * source[upper]
    return output.astype(np.float32)


def _normalize_and_unroll_quaternions(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    norms = np.linalg.norm(output, axis=-1, keepdims=True)
    if not np.all(np.isfinite(output)) or np.any(norms < 1e-12):
        raise ValueError("quaternion samples must be finite and nonzero")
    output /= norms
    for frame in range(1, len(output)):
        flip = np.sum(output[frame - 1] * output[frame], axis=-1) < 0.0
        output[frame, flip] *= -1.0
    return output


def _interpolate_quaternions(
        values: np.ndarray,
        coordinates: np.ndarray) -> np.ndarray:
    source = _normalize_and_unroll_quaternions(values)
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, len(source) - 1)
    alpha = (coordinates - lower)[:, None, None]
    first = source[lower]
    second = source[upper].copy()
    dots = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dots < 0.0, -second, second)
    dots = np.clip(np.sum(first * second, axis=-1, keepdims=True), -1.0, 1.0)

    output = (1.0 - alpha) * first + alpha * second
    far = dots[..., 0] < 0.9995
    if np.any(far):
        theta = np.arccos(dots[far])
        sine = np.sin(theta)
        far_alpha = np.broadcast_to(alpha, dots.shape)[far]
        output[far] = (
            np.sin((1.0 - far_alpha) * theta) / sine * first[far]
            + np.sin(far_alpha * theta) / sine * second[far])
    output /= np.linalg.norm(output, axis=-1, keepdims=True)
    return output.astype(np.float32)


def _differentiate_vectors(values: np.ndarray, fps: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    output = np.zeros_like(source)
    if len(source) == 1:
        return output.astype(np.float32)
    if len(source) == 2:
        derivative = (source[1] - source[0]) * fps
        output[0] = derivative
        output[1] = derivative
        return output.astype(np.float32)

    output[1:-1] = 0.5 * (source[2:] - source[:-2]) * fps
    output[0] = 0.5 * (-3.0 * source[0] + 4.0 * source[1] - source[2]) * fps
    output[-1] = 0.5 * (
        3.0 * source[-1] - 4.0 * source[-2] + source[-3]) * fps
    return output.astype(np.float32)


def _quaternion_conjugate(values: np.ndarray) -> np.ndarray:
    output = values.copy()
    output[..., 1:] *= -1.0
    return output


def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_w = first[..., 0:1]
    second_w = second[..., 0:1]
    first_vector = first[..., 1:]
    second_vector = second[..., 1:]
    return np.concatenate(
        [first_w * second_w - np.sum(
            first_vector * second_vector, axis=-1, keepdims=True),
         first_w * second_vector + second_w * first_vector
         + np.cross(first_vector, second_vector)],
        axis=-1)


def _scaled_angle_axis(values: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(values, dtype=np.float64).copy()
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
    quaternions = np.where(quaternions[..., 0:1] < 0.0, -quaternions, quaternions)
    vectors = quaternions[..., 1:]
    vector_norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    half_angle = np.arctan2(vector_norm, quaternions[..., 0:1])
    scale = np.where(
        vector_norm < 1e-8,
        2.0,
        2.0 * half_angle / np.maximum(vector_norm, 1e-30))
    return scale * vectors


def _differentiate_quaternions(values: np.ndarray, fps: float) -> np.ndarray:
    source = _normalize_and_unroll_quaternions(values)
    output = np.zeros(source.shape[:-1] + (3,), dtype=np.float64)
    if len(source) == 1:
        return output.astype(np.float32)
    intervals = _scaled_angle_axis(_quaternion_multiply(
        source[1:], _quaternion_conjugate(source[:-1]))) * fps
    output[0] = intervals[0]
    output[-1] = intervals[-1]
    if len(source) > 2:
        output[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])
    return output.astype(np.float32)


def _sample_contacts(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    # Explicit nearest-half-up sampling. Contacts are already temporally
    # filtered in the authoritative source and remain exactly binary here.
    indices = np.floor(coordinates + 0.5).astype(np.int64)
    indices = np.minimum(indices, len(values) - 1)
    return np.asarray(values[indices], dtype=np.uint8).copy()


def retime_database(
        database: FlatDatabase,
        source_fps: float = SOURCE_FPS,
        target_fps: float = TARGET_FPS) -> FlatDatabase:
    """Retime every range independently while preserving range endpoints."""
    _validate_database(database)
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("sample rates must be positive")

    positions: List[np.ndarray] = []
    velocities: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    angular_velocities: List[np.ndarray] = []
    contacts: List[np.ndarray] = []
    starts: List[int] = []
    stops: List[int] = []
    cursor = 0

    for source_start, source_stop in zip(
            database.range_starts, database.range_stops):
        source_start = int(source_start)
        source_stop = int(source_stop)
        source_count = source_stop - source_start
        target_count = _output_frame_count(
            source_count, source_fps, target_fps)
        coordinates = _source_coordinates(source_count, target_count)

        range_positions = _interpolate_vectors(
            database.bone_positions[source_start:source_stop], coordinates)
        range_rotations = _interpolate_quaternions(
            database.bone_rotations[source_start:source_stop], coordinates)
        positions.append(range_positions)
        velocities.append(_differentiate_vectors(range_positions, target_fps))
        rotations.append(range_rotations)
        angular_velocities.append(
            _differentiate_quaternions(range_rotations, target_fps))
        contacts.append(_sample_contacts(
            database.contact_states[source_start:source_stop], coordinates))
        starts.append(cursor)
        cursor += target_count
        stops.append(cursor)

    output = FlatDatabase(
        bone_positions=np.concatenate(positions).astype(np.float32),
        bone_velocities=np.concatenate(velocities).astype(np.float32),
        bone_rotations=np.concatenate(rotations).astype(np.float32),
        bone_angular_velocities=np.concatenate(
            angular_velocities).astype(np.float32),
        bone_parents=np.asarray(database.bone_parents, dtype=np.int32).copy(),
        range_starts=np.asarray(starts, dtype=np.int32),
        range_stops=np.asarray(stops, dtype=np.int32),
        contact_states=np.concatenate(contacts).astype(np.uint8),
    )
    _validate_database(output)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(
        source: FlatDatabase,
        output: FlatDatabase,
        source_hash: str,
        output_hash: str,
        source_fps: float,
        target_fps: float) -> Dict[str, object]:
    ranges: List[Dict[str, object]] = []
    maximum_error = 0.0
    for source_start, source_stop, target_start, target_stop in zip(
            source.range_starts,
            source.range_stops,
            output.range_starts,
            output.range_stops):
        source_frames = int(source_stop - source_start)
        target_frames = int(target_stop - target_start)
        source_duration = (source_frames - 1) / source_fps
        target_duration = (target_frames - 1) / target_fps
        error = abs(target_duration - source_duration)
        maximum_error = max(maximum_error, error)
        ranges.append({
            "source_frames": source_frames,
            "target_frames": target_frames,
            "source_duration_seconds": source_duration,
            "target_duration_seconds": target_duration,
            "duration_error_seconds": error,
        })

    return {
        "schema": "flat-locomotion-retime-v1",
        "source_fps": source_fps,
        "target_fps": target_fps,
        "source_sha256": source_hash,
        "output_sha256": output_hash,
        "source_frame_count": int(source.bone_positions.shape[0]),
        "output_frame_count": int(output.bone_positions.shape[0]),
        "bone_count": int(output.bone_positions.shape[1]),
        "range_count": len(ranges),
        "contact_count": int(output.contact_states.shape[1]),
        "maximum_range_duration_error_seconds": maximum_error,
        "contact_resampling": "nearest-source-sample-half-up",
        "rotation_resampling": "shortest-arc-slerp",
        "velocity_units": "per-second-recomputed-at-target-rate",
        "ranges": ranges,
    }


def retime_file(
        source_path: Path | str,
        output_path: Path | str,
        manifest_path: Path | str,
        expected_source_sha256: Optional[str] = AUTHORITATIVE_SOURCE_SHA256,
        source_fps: float = SOURCE_FPS,
        target_fps: float = TARGET_FPS) -> Dict[str, object]:
    source_path = Path(source_path)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("source and output database paths must differ")
    source_hash = _sha256(source_path)
    if (expected_source_sha256 is not None
            and source_hash != expected_source_sha256.lower()):
        raise ValueError(
            "authoritative source database SHA-256 mismatch: "
            f"expected {expected_source_sha256.lower()}, got {source_hash}")

    source = read_database(source_path)
    output = retime_database(source, source_fps, target_fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_temporary = output_path.with_name(output_path.name + ".tmp")
    manifest_temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        write_database(output_temporary, output)
        output_hash = _sha256(output_temporary)
        record = _manifest(
            source,
            output,
            source_hash,
            output_hash,
            source_fps,
            target_fps)
        manifest_temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        os.replace(output_temporary, output_path)
        os.replace(manifest_temporary, manifest_path)
    finally:
        output_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retime the authoritative 60 Hz flat database to 25 Hz")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("resources/database_60hz.bin"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resources/database.bin"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("resources/database_25hz_manifest.json"))
    parser.add_argument(
        "--expected-source-sha256",
        default=AUTHORITATIVE_SOURCE_SHA256)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    record = retime_file(
        args.source,
        args.output,
        args.manifest,
        expected_source_sha256=args.expected_source_sha256)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
