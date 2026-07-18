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
_SEGMENT_PARALLEL_EPSILON = np.float32(1.0e-7)
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
    return math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (x * x + y * y))


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
        if abs(delta[axis]) <= float(_SEGMENT_PARALLEL_EPSILON):
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
    def near_in_representation(
        candidate: dict,
        existing: dict,
        *,
        authored_float32: bool,
    ) -> bool:
        def scalar(row: dict, field: str) -> float:
            value = float(row[field])
            return float(np.float32(value)) if authored_float32 else value

        distance = math.hypot(
            scalar(candidate, "root_x_object_m")
            - scalar(existing, "root_x_object_m"),
            scalar(candidate, "root_z_object_m")
            - scalar(existing, "root_z_object_m"),
        )
        yaw_distance = abs(
            _wrap_angle(
                scalar(candidate, "root_yaw_object_radians")
                - scalar(existing, "root_yaw_object_radians")
            )
        )
        return (
            distance <= float(_DEDUPE_POSITION_LIMIT_M)
            and yaw_distance <= float(_DEDUPE_YAW_LIMIT_RADIANS)
        )

    retained: list[dict] = []
    deduplicated_count = 0
    for candidate in sorted(candidates, key=lambda row: tuple(row["stable_key"])):
        duplicate = False
        for existing in retained:
            if candidate["active_hand"] != existing["active_hand"]:
                continue
            if (
                near_in_representation(
                    candidate, existing, authored_float32=False
                )
                or near_in_representation(
                    candidate, existing, authored_float32=True
                )
            ):
                duplicate = True
                break
        if duplicate:
            deduplicated_count += 1
        else:
            retained.append(candidate)
    return retained, deduplicated_count


def _manifest_records(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    records = manifest.get("clips")
    if not isinstance(records, list):
        raise ValueError("manifest clips must be an array")
    result: list[Mapping[str, object]] = []
    seen_sequences: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"manifest clip {index} must be an object")
        sequence_id = record.get("sequence_id")
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError(
                f"manifest clip {index} sequence_id must be a nonempty string"
            )
        if sequence_id in seen_sequences:
            raise ValueError(f"manifest contains duplicate sequence {sequence_id!r}")
        seen_sequences.add(sequence_id)
        object_id = record.get("object_id")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError(
                f"manifest clip {index} object_id must be a nonempty string"
            )
        result.append(record)
    return result


def _target_manifest_record(
    manifest: Mapping[str, object], target_sequence_id: str
) -> Mapping[str, object]:
    records = _manifest_records(manifest)
    for record in sorted(records, key=lambda item: str(item["sequence_id"])):
        if record["sequence_id"] == target_sequence_id:
            return record
    raise ValueError(f"target sequence {target_sequence_id!r} was not found")


def _joined_manifest_records(
    artifact: InteractionArtifact,
    manifest: Mapping[str, object],
    target_sequence_id: str,
) -> tuple[
    list[tuple[int, Mapping[str, object]]], Mapping[str, object], int
]:
    records = _manifest_records(manifest)
    target_record = next(
        (
            record
            for record in records
            if record["sequence_id"] == target_sequence_id
        ),
        None,
    )
    if target_record is None:
        raise ValueError(f"target sequence {target_sequence_id!r} was not found")

    if "skeleton_parents" not in manifest:
        raise ValueError("manifest skeleton_parents is required")
    manifest_parents = manifest["skeleton_parents"]
    try:
        parents = np.asarray(manifest_parents)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "manifest skeleton_parents must match the artifact parents"
        ) from error
    if (
        parents.shape != artifact.parents.shape
        or not np.issubdtype(parents.dtype, np.integer)
        or not np.array_equal(parents, artifact.parents)
    ):
        raise ValueError(
            "manifest skeleton_parents must match the artifact parents"
        )

    artifact_ranges = {
        (int(start), int(stop)): index
        for index, (start, stop) in enumerate(
            zip(artifact.range_starts, artifact.range_stops)
        )
    }
    joined: list[tuple[int, Mapping[str, object]]] = []
    seen_ranges: set[tuple[int, int]] = set()
    target_index = -1
    for record in sorted(records, key=lambda item: str(item["sequence_id"])):
        start = record.get("range_start")
        stop = record.get("range_stop")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, np.integer))
            or isinstance(stop, bool)
            or not isinstance(stop, (int, np.integer))
        ):
            raise ValueError("manifest clip range bounds must be integers")
        clip_range = (int(start), int(stop))
        if clip_range in seen_ranges:
            raise ValueError(f"manifest contains duplicate clip range {clip_range}")
        seen_ranges.add(clip_range)
        artifact_index = artifact_ranges.get(clip_range)
        if artifact_index is None:
            raise ValueError(
                f"manifest clip range {clip_range} does not match an artifact range"
            )
        active_hand = record.get("active_hand")
        if (
            isinstance(active_hand, bool)
            or not isinstance(active_hand, (int, np.integer))
            or int(active_hand) not in (0, 1)
        ):
            raise ValueError("manifest clip active_hand must be integer 0 or 1")
        if int(active_hand) != int(artifact.active_hands[artifact_index]):
            raise ValueError(
                "manifest active_hand does not match the artifact range"
            )
        joined.append((artifact_index, record))
        if record is target_record:
            target_index = artifact_index

    if seen_ranges != set(artifact_ranges):
        raise ValueError("manifest clip ranges are not a bijection with artifact ranges")
    return joined, target_record, target_index


def _clip_events(
    artifact: InteractionArtifact, clip_index: int
) -> tuple[int, int, int, int, int, int, int]:
    start = int(artifact.range_starts[clip_index])
    stop = int(artifact.range_stops[clip_index])
    phases = artifact.phases[start:stop]
    if np.any(np.diff(phases.astype(np.int16)) < 0):
        raise ValueError(
            f"artifact range {clip_index} phases must be monotonic"
        )
    event_frames: list[int] = []
    for phase in (
        InteractionPhase.REACH,
        InteractionPhase.CONTACT,
        InteractionPhase.LIFT,
        InteractionPhase.HOLD,
    ):
        matches = np.flatnonzero(phases == int(phase))
        if len(matches) == 0:
            raise ValueError(
                f"artifact range {clip_index} is missing the {phase.name} phase"
            )
        event_frames.append(int(matches[0]))
    entry, contact, lift, hold = event_frames
    if not entry < contact < lift < hold:
        raise ValueError(
            f"artifact range {clip_index} phase events are not monotonic"
        )
    return start, stop, entry, contact, lift, hold, contact - 1


def _world_joint_transform(
    artifact: InteractionArtifact, frame: int, joint: int
) -> tuple[np.ndarray, np.ndarray]:
    world_positions = np.empty((len(artifact.parents), 3), dtype=np.float64)
    world_rotations = np.empty((len(artifact.parents), 4), dtype=np.float64)
    for index, parent_value in enumerate(artifact.parents):
        parent = int(parent_value)
        local_position = np.asarray(artifact.positions[frame, index], dtype=np.float64)
        local_rotation = _normalize_quaternion(artifact.rotations[frame, index])
        if parent == -1:
            world_positions[index] = local_position
            world_rotations[index] = local_rotation
        else:
            world_positions[index] = world_positions[parent] + _rotate_vector(
                world_rotations[parent], local_position
            )
            world_rotations[index] = _quaternion_multiply(
                world_rotations[parent], local_rotation
            )
        if index == joint:
            return world_positions[index].copy(), world_rotations[index].copy()
    raise ValueError(f"joint index {joint} is outside the artifact skeleton")


def _map_position_to_target(
    position: Sequence[float],
    source_object_position: Sequence[float],
    source_object_yaw: float,
    target_object_position: Sequence[float],
    target_object_yaw: float,
) -> np.ndarray:
    value = _as_vector(position, 3, "mapped position")
    source_object = _as_vector(
        source_object_position, 3, "source object position"
    )
    target_object = _as_vector(
        target_object_position, 3, "target object position"
    )
    mapped = value.copy()
    mapped[[0, 2]] = target_object[[0, 2]] + _rotate_xz(
        value[[0, 2]] - source_object[[0, 2]],
        _wrap_angle(target_object_yaw - source_object_yaw),
    )
    return mapped


def _float32_position_error(
    measured: Sequence[float], expected: Sequence[float]
) -> float:
    difference = np.asarray(measured, dtype=np.float32) - np.asarray(
        expected, dtype=np.float32
    )
    squared_distance = np.sum(difference * difference, dtype=np.float32)
    return float(np.sqrt(squared_distance).astype(np.float32))


def _float32_orientation_error(
    measured: Sequence[float], expected: Sequence[float]
) -> float:
    measured_quaternion = _normalize_quaternion(measured)
    expected_quaternion = _normalize_quaternion(expected)
    dot = np.float32(np.dot(measured_quaternion, expected_quaternion))
    cosine = np.minimum(np.float32(1.0), np.abs(dot)).astype(np.float32)
    return float(
        np.float32(2.0) * np.arccos(cosine).astype(np.float32)
    )


def extract_slot_candidates(
    artifact: InteractionArtifact,
    manifest: Mapping[str, object],
    target_sequence_id: str,
) -> dict:
    """Return deterministic target metadata and static-feasible slots."""
    artifact.validate()

    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("manifest dataset_id must be a nonempty string")
    schema_version = manifest.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, (int, np.integer))
        or int(schema_version) != 1
    ):
        raise ValueError("manifest schema_version must be exactly 1")
    schema_version = 1

    joined, target_record, target_index = _joined_manifest_records(
        artifact, manifest, target_sequence_id
    )
    events_by_index = [
        _clip_events(artifact, clip_index)
        for clip_index in range(len(artifact.range_starts))
    ]
    (
        target_start,
        target_stop,
        target_entry,
        target_contact,
        target_lift,
        target_hold,
        target_alignment,
    ) = events_by_index[target_index]
    target_alignment_global = target_start + target_alignment
    target_object_position = np.asarray(
        artifact.object_positions[target_alignment_global], dtype=np.float64
    )
    target_object_rotation = _normalize_quaternion(
        artifact.object_rotations[target_alignment_global]
    )
    target_object_yaw = _quaternion_yaw(target_object_rotation)
    target_grasp_position = np.asarray(
        artifact.grasp_positions_object[target_index], dtype=np.float64
    )
    target_affordance_position = target_object_position + _rotate_vector(
        target_object_rotation, target_grasp_position
    )
    target_affordance_rotation = _quaternion_multiply(
        target_object_rotation,
        artifact.grasp_rotations_object[target_index],
    )
    target_active_hand = int(artifact.active_hands[target_index])
    target_table_position = artifact.table_positions[target_index]
    target_table_rotation = artifact.table_rotations[target_index]
    target_table_size = artifact.table_sizes[target_index]
    target_object_dimensions = artifact.object_dimensions[target_index]

    target = {
        "sequence_id": str(target_record["sequence_id"]),
        "object_id": str(target_record["object_id"]),
        "active_hand": target_active_hand,
        "range_start": target_start,
        "range_stop": target_stop,
        "entry_local_frame": target_entry,
        "contact_local_frame": target_contact,
        "lift_local_frame": target_lift,
        "hold_local_frame": target_hold,
        "stop_local_frame": target_stop - target_start,
        "object_alignment_local_frame": target_alignment,
        "object_alignment_position": artifact.object_positions[
            target_alignment_global
        ]
        .astype(float)
        .tolist(),
        "object_alignment_rotation": artifact.object_rotations[
            target_alignment_global
        ]
        .astype(float)
        .tolist(),
        "table_position": artifact.table_positions[target_index]
        .astype(float)
        .tolist(),
        "table_rotation": artifact.table_rotations[target_index]
        .astype(float)
        .tolist(),
        "table_size": artifact.table_sizes[target_index]
        .astype(float)
        .tolist(),
        "object_dimensions": artifact.object_dimensions[target_index]
        .astype(float)
        .tolist(),
        "grasp_position_object": artifact.grasp_positions_object[target_index]
        .astype(float)
        .tolist(),
        "grasp_rotation_object": artifact.grasp_rotations_object[target_index]
        .astype(float)
        .tolist(),
        "approach_direction_object": artifact.approach_directions_object[
            target_index
        ]
        .astype(float)
        .tolist(),
    }

    rejected_counts = {gate: 0 for gate in _REJECTED_GATE_ORDER}
    static_candidates: list[dict] = []
    for clip_index, record in joined:
        (
            start,
            stop,
            entry,
            contact,
            lift,
            hold,
            alignment,
        ) = events_by_index[clip_index]
        active_hand = int(artifact.active_hands[clip_index])
        if active_hand != target_active_hand:
            rejected_counts["hand_mismatch"] += 1
            continue

        alignment_global = start + alignment
        source_object_position = artifact.object_positions[alignment_global]
        source_object_yaw = _quaternion_yaw(
            artifact.object_rotations[alignment_global]
        )
        yaw_delta = _wrap_angle(target_object_yaw - source_object_yaw)
        wrist_index = (
            _LEFT_WRIST_INDEX if active_hand == 0 else _RIGHT_WRIST_INDEX
        )
        contact_global = start + contact
        contact_position, contact_rotation = _world_joint_transform(
            artifact, contact_global, wrist_index
        )
        mapped_contact_position = _map_position_to_target(
            contact_position,
            source_object_position,
            source_object_yaw,
            target_object_position,
            target_object_yaw,
        )
        mapped_contact_rotation = _quaternion_multiply(
            _yaw_quaternion(yaw_delta), contact_rotation
        )
        position_error = _float32_position_error(
            mapped_contact_position, target_affordance_position
        )
        if position_error > float(_HAND_POSITION_LIMIT_M):
            rejected_counts["contact_position"] += 1
            continue
        orientation_error = _float32_orientation_error(
            mapped_contact_rotation, target_affordance_rotation
        )
        if orientation_error > float(_HAND_ORIENTATION_LIMIT_RADIANS):
            rejected_counts["contact_orientation"] += 1
            continue

        root_table_clear = True
        for frame in range(start + entry, stop):
            mapped_root = _map_position_to_target(
                artifact.positions[frame, 0],
                source_object_position,
                source_object_yaw,
                target_object_position,
                target_object_yaw,
            )
            if _root_intersects_table(
                mapped_root,
                target_table_position,
                target_table_rotation,
                target_table_size,
            ):
                root_table_clear = False
                break
        if not root_table_clear:
            rejected_counts["root_table"] += 1
            continue

        mapped_hand_positions: list[np.ndarray] = []
        for frame in range(start + entry, stop):
            wrist_position, _ = _world_joint_transform(
                artifact, frame, wrist_index
            )
            mapped_hand_positions.append(
                _map_position_to_target(
                    wrist_position,
                    source_object_position,
                    source_object_yaw,
                    target_object_position,
                    target_object_yaw,
                )
            )
        hand_table_clear = all(
            not _segment_intersects_expanded_box(
                segment_start,
                segment_stop,
                target_table_position,
                target_table_rotation,
                target_table_size,
                float(_CLEARANCE_RADIUS_M),
            )
            for segment_start, segment_stop in zip(
                mapped_hand_positions, mapped_hand_positions[1:]
            )
        )
        if not hand_table_clear:
            rejected_counts["hand_table"] += 1
            continue

        precontact_segment_count = contact - entry - 1
        hand_object_clear = all(
            not _segment_intersects_expanded_box(
                mapped_hand_positions[index],
                mapped_hand_positions[index + 1],
                target_object_position,
                target_object_rotation,
                target_object_dimensions,
                float(_CLEARANCE_RADIUS_M),
            )
            for index in range(precontact_segment_count)
        )
        if not hand_object_clear:
            rejected_counts["hand_object"] += 1
            continue

        entry_global = start + entry
        source_root_position = artifact.positions[entry_global, 0]
        root_object_xz = _rotate_xz(
            np.asarray(source_root_position, dtype=np.float64)[[0, 2]]
            - np.asarray(source_object_position, dtype=np.float64)[[0, 2]],
            -source_object_yaw,
        )
        root_yaw_object = _wrap_angle(
            _quaternion_yaw(artifact.rotations[entry_global, 0])
            - source_object_yaw
        )
        planar_velocity = np.asarray(
            artifact.velocities[entry_global, 0, [0, 2]], dtype=np.float32
        )
        entry_speed = np.sqrt(
            np.sum(planar_velocity * planar_velocity, dtype=np.float32)
        ).astype(np.float32)
        static_candidates.append(
            {
                "stable_key": [
                    dataset_id,
                    schema_version,
                    str(record["sequence_id"]),
                    entry,
                    active_hand,
                ],
                "sequence_id": str(record["sequence_id"]),
                "object_id": str(record["object_id"]),
                "active_hand": active_hand,
                "source_entry_frame": int(artifact.source_frames[entry_global]),
                "entry_local_frame": entry,
                "contact_local_frame": contact,
                "lift_local_frame": lift,
                "hold_local_frame": hold,
                "stop_local_frame": stop - start,
                "object_alignment_local_frame": alignment,
                "root_x_object_m": float(root_object_xz[0]),
                "root_z_object_m": float(root_object_xz[1]),
                "root_yaw_object_radians": float(root_yaw_object),
                "contact_hand_position_error_m": position_error,
                "contact_hand_orientation_error_radians": orientation_error,
                "entry_root_planar_speed_mps": float(entry_speed),
                "root_table_clear": True,
                "hand_table_clear": True,
                "hand_object_clear": True,
                "static_path_feasible": True,
            }
        )

    retained_candidates, deduplicated_count = _dedupe_candidates(
        static_candidates
    )
    for candidate in retained_candidates:
        for field in (
            "root_x_object_m",
            "root_z_object_m",
            "root_yaw_object_radians",
        ):
            candidate[field] = float(np.float32(candidate[field]))
    return {
        "dataset_id": dataset_id,
        "schema_version": schema_version,
        "target": target,
        "retained_candidates": retained_candidates,
        "rejected_counts_by_gate": rejected_counts,
        "deduplicated_candidate_count": deduplicated_count,
    }


def extract_slot_candidates_from_pack(
    pack: Path,
    target_sequence_id: str,
) -> dict:
    artifact, _, manifest, split, _ = read_artifact_set(Path(pack))
    target_record = _target_manifest_record(manifest, target_sequence_id)
    if not isinstance(split, Mapping):
        raise ValueError("evaluation split must be an object")
    heldout_objects = split.get("heldout_objects")
    if not isinstance(heldout_objects, list):
        raise ValueError("evaluation split heldout_objects must be an array")
    if target_record["object_id"] in heldout_objects:
        raise ValueError(
            f"target object {target_record['object_id']!r} is in the held-out split"
        )
    return extract_slot_candidates(artifact, manifest, target_sequence_id)


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
