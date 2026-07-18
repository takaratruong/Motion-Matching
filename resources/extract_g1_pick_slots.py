"""Deterministic offline authoring of static G1 pickup entry slots."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np

from resources.g1_interaction_builder.artifacts import read_artifact_set
from resources.g1_interaction_builder.schema import (
    InteractionArtifact,
    InteractionPhase,
)


_HAND_POSITION_LIMIT_M = np.float32(0.12)
_HAND_ORIENTATION_LIMIT_RADIANS = np.float32(np.deg2rad(25.0))
_ROOT_TABLE_EXPANSION_M = np.float32(0.25) - np.float32(0.01)
_CLEARANCE_RADIUS_M = np.float32(0.04)
_DEDUPE_POSITION_LIMIT_M = np.float32(0.10)
_DEDUPE_YAW_LIMIT_RADIANS = np.float32(np.deg2rad(10.0))
_REJECTED_GATE_ORDER = (
    "hand_mismatch",
    "contact_position",
    "contact_orientation",
    "root_table",
    "hand_table",
    "hand_object",
)

_LEFT_WRIST_INDEX = 23
_RIGHT_WRIST_INDEX = 30


def _as_vector(value: Sequence[float], length: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return vector


def _normalize_quaternion(value: Sequence[float]) -> np.ndarray:
    quaternion = _as_vector(value, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        raise ValueError("quaternion norm must be non-zero")
    return quaternion / norm


def _quaternion_multiply(
    left: Sequence[float], right: Sequence[float]
) -> np.ndarray:
    lw, lx, ly, lz = _normalize_quaternion(left)
    rw, rx, ry, rz = _normalize_quaternion(right)
    return _normalize_quaternion(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _rotate_vector(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(quaternion)
    vx, vy, vz = _as_vector(vector, 3, "vector")
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z)) * vx
            + 2.0 * (x * y - w * z) * vy
            + 2.0 * (x * z + w * y) * vz,
            2.0 * (x * y + w * z) * vx
            + (1.0 - 2.0 * (x * x + z * z)) * vy
            + 2.0 * (y * z - w * x) * vz,
            2.0 * (x * z - w * y) * vx
            + 2.0 * (y * z + w * x) * vy
            + (1.0 - 2.0 * (x * x + y * y)) * vz,
        ),
        dtype=np.float64,
    )


def _quaternion_yaw(value: Sequence[float]) -> float:
    w, x, y, z = _normalize_quaternion(value)
    return math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))


def _yaw_quaternion(yaw: float) -> np.ndarray:
    half_yaw = 0.5 * float(yaw)
    return np.asarray((math.cos(half_yaw), 0.0, math.sin(half_yaw), 0.0))


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def _rotate_xz(value: Sequence[float], yaw: float) -> np.ndarray:
    x, z = _as_vector(value, 2, "planar vector")
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    return np.asarray((cosine * x + sine * z, -sine * x + cosine * z))


def _root_intersects_table(
    root_position: Sequence[float],
    table_position: Sequence[float],
    table_rotation: Sequence[float],
    table_size: Sequence[float],
) -> bool:
    root = _as_vector(root_position, 3, "root position")
    table = _as_vector(table_position, 3, "table position")
    size = _as_vector(table_size, 3, "table size")
    local = _rotate_xz(root[[0, 2]] - table[[0, 2]], -_quaternion_yaw(table_rotation))
    half_x = 0.5 * size[0] + float(_ROOT_TABLE_EXPANSION_M)
    half_z = 0.5 * size[2] + float(_ROOT_TABLE_EXPANSION_M)
    return bool(abs(local[0]) <= half_x and abs(local[1]) <= half_z)


def _segment_intersects_expanded_box(
    start: Sequence[float],
    stop: Sequence[float],
    box_position: Sequence[float],
    box_rotation: Sequence[float],
    box_size: Sequence[float],
    expansion: float,
) -> bool:
    segment_start = _as_vector(start, 3, "segment start")
    segment_stop = _as_vector(stop, 3, "segment stop")
    center = _as_vector(box_position, 3, "box position")
    size = _as_vector(box_size, 3, "box size")
    rotation = _normalize_quaternion(box_rotation)
    inverse_rotation = rotation * np.asarray((1.0, -1.0, -1.0, -1.0))
    local_start = _rotate_vector(inverse_rotation, segment_start - center)
    local_stop = _rotate_vector(inverse_rotation, segment_stop - center)
    half_extents = 0.5 * size + float(expansion)

    lower_parameter = 0.0
    upper_parameter = 1.0
    delta = local_stop - local_start
    for axis in range(3):
        if delta[axis] == 0.0:
            if abs(local_start[axis]) > half_extents[axis]:
                return False
            continue
        first = (-half_extents[axis] - local_start[axis]) / delta[axis]
        second = (half_extents[axis] - local_start[axis]) / delta[axis]
        if first > second:
            first, second = second, first
        lower_parameter = max(lower_parameter, float(first))
        upper_parameter = min(upper_parameter, float(second))
        if lower_parameter > upper_parameter:
            return False
    return True


def _dedupe_candidates(candidates: Sequence[dict]) -> tuple[list[dict], int]:
    retained: list[dict] = []
    deduplicated_count = 0
    for candidate in sorted(candidates, key=lambda row: tuple(row["stable_key"])):
        duplicate = False
        for existing in retained:
            if candidate["active_hand"] != existing["active_hand"]:
                continue
            distance = math.hypot(
                float(candidate["root_x_object_m"])
                - float(existing["root_x_object_m"]),
                float(candidate["root_z_object_m"])
                - float(existing["root_z_object_m"]),
            )
            yaw_distance = abs(
                _wrap_angle(
                    float(candidate["root_yaw_object_radians"])
                    - float(existing["root_yaw_object_radians"])
                )
            )
            if (
                distance <= float(_DEDUPE_POSITION_LIMIT_M)
                and yaw_distance <= float(_DEDUPE_YAW_LIMIT_RADIANS)
            ):
                duplicate = True
                break
        if duplicate:
            deduplicated_count += 1
        else:
            retained.append(candidate)
    return retained, deduplicated_count


def extract_slot_candidates(
    artifact: InteractionArtifact,
    manifest: Mapping[str, object],
    target_sequence_id: str,
) -> dict:
    """Return deterministic target metadata and static-feasible slots."""
    raise NotImplementedError


def extract_slot_candidates_from_pack(
    pack: Path,
    target_sequence_id: str,
) -> dict:
    raise NotImplementedError


def serialize_slot_report(report: dict) -> str:
    def normalize(value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, (np.floating, float)):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("slot report contains a non-finite float")
            rounded = float(format(number, ".9g"))
            return 0.0 if rounded == 0.0 else rounded
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    return json.dumps(
        normalize(report),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--target-sequence-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    report = extract_slot_candidates_from_pack(
        arguments.pack,
        arguments.target_sequence_id,
    )
    _atomic_write_text(arguments.output, serialize_slot_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
