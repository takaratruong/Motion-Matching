import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from resources.extract_g1_pick_slots import (
    _dedupe_candidates,
    _root_intersects_table,
    _segment_intersects_expanded_box,
    extract_slot_candidates,
    extract_slot_candidates_from_pack,
    serialize_slot_report,
)
from resources.g1_interaction_builder.metadata import GRAIL_DATASET_ID
from resources.g1_interaction_builder.schema import (
    G1_SKELETON,
    InteractionArtifact,
    InteractionPhase,
)


_SOURCE_SEQUENCE = "pickup_table__source_object__001"
_TARGET_SEQUENCE = "pickup_table__target_object__002"
_SOURCE_OBJECT = "source_object"
_TARGET_OBJECT = "target_object"
_CLIP_FRAMES = 10
_RIGHT_WRIST = 30

_HAND_POSITION_LIMIT_M = np.float32(0.12)
_HAND_ORIENTATION_LIMIT_RADIANS = np.float32(np.deg2rad(25.0))
_CLEARANCE_RADIUS_M = np.float32(0.04)
_ROOT_TABLE_EXPANSION_M = np.float32(0.25) - np.float32(0.01)
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
_SLOT_CANDIDATE_FIELDS = (
    "root_x_object_m",
    "root_z_object_m",
    "root_yaw_object_radians",
    "contact_hand_position_error_m",
    "contact_hand_orientation_error_radians",
    "entry_root_planar_speed_mps",
    "root_table_clear",
    "hand_table_clear",
    "hand_object_clear",
    "static_path_feasible",
)
_SLOT_CANDIDATE_KEYS = {
    "stable_key",
    "sequence_id",
    "object_id",
    "active_hand",
    "source_entry_frame",
    "entry_local_frame",
    "contact_local_frame",
    "lift_local_frame",
    "hold_local_frame",
    "stop_local_frame",
    "object_alignment_local_frame",
    *_SLOT_CANDIDATE_FIELDS,
}
_REPORT_KEYS = {
    "dataset_id",
    "schema_version",
    "target",
    "retained_candidates",
    "rejected_counts_by_gate",
    "deduplicated_candidate_count",
}
_TARGET_KEYS = {
    "sequence_id",
    "object_id",
    "active_hand",
    "range_start",
    "range_stop",
    "entry_local_frame",
    "contact_local_frame",
    "lift_local_frame",
    "hold_local_frame",
    "stop_local_frame",
    "object_alignment_local_frame",
    "object_alignment_position",
    "object_alignment_rotation",
    "table_position",
    "table_rotation",
    "table_size",
    "object_dimensions",
    "grasp_position_object",
    "grasp_rotation_object",
    "approach_direction_object",
}

_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
_CLIP_PHASES = np.array(
    [
        InteractionPhase.APPROACH,
        InteractionPhase.REACH,
        InteractionPhase.REACH,
        InteractionPhase.CONTACT,
        InteractionPhase.LIFT,
        InteractionPhase.HOLD,
        InteractionPhase.HOLD,
        InteractionPhase.HOLD,
        InteractionPhase.HOLD,
        InteractionPhase.HOLD,
    ],
    np.uint8,
)


def _two_clip_artifact_and_manifest():
    """Return two exact-schema clips with manifest records in reverse order."""
    frames = 2 * _CLIP_FRAMES
    positions = np.zeros((frames, 31, 3), np.float32)
    rotations = np.broadcast_to(_IDENTITY, (frames, 31, 4)).copy()

    source = slice(0, _CLIP_FRAMES)
    target = slice(_CLIP_FRAMES, frames)
    positions[source, 0] = np.array([0.0, 1.0, 0.0], np.float32)
    positions[target, 0] = np.array([8.0, 1.0, -1.0], np.float32)

    # All local joints are identity except the Simulation root and right wrist.
    # At Contact the source wrist composes the same object-local grasp as the
    # target affordance after Contact-minus-one planar alignment.
    source_contact = 3
    target_contact = _CLIP_FRAMES + 3
    source_contact_hand = np.array([1.1, 1.2, 2.3], np.float32)
    target_contact_hand = np.array([4.1, 1.2, 3.3], np.float32)
    positions[source_contact:_CLIP_FRAMES, _RIGHT_WRIST] = (
        source_contact_hand - positions[source_contact, 0]
    )
    positions[target_contact:, _RIGHT_WRIST] = (
        target_contact_hand - positions[target_contact, 0]
    )

    object_positions = np.zeros((frames, 3), np.float32)
    object_positions[source] = np.array([1.0, 1.0, 2.0], np.float32)
    object_positions[target] = np.array([4.0, 1.0, 3.0], np.float32)
    # Deliberately make Contact differ from Contact-minus-one. A consumer that
    # uses Contact will report these sentinels instead of the reference poses.
    object_positions[source_contact] = np.array(
        [10.0, 10.0, 20.0], np.float32
    )
    object_positions[target_contact] = np.array(
        [40.0, 10.0, 30.0], np.float32
    )

    phases = np.tile(_CLIP_PHASES, 2)
    hand_contacts = np.zeros((frames, 2), np.uint8)
    hand_contacts[source_contact:_CLIP_FRAMES, 1] = 1
    hand_contacts[target_contact:frames, 1] = 1
    time_to_contact = np.tile(
        np.array([0.12, 0.08, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        2,
    ).astype(np.float32)

    artifact = InteractionArtifact(
        fps=25,
        parents=G1_SKELETON.parents.astype(np.int32, copy=True),
        range_starts=np.array([0, _CLIP_FRAMES], np.int32),
        range_stops=np.array([_CLIP_FRAMES, frames], np.int32),
        positions=positions,
        velocities=np.zeros((frames, 31, 3), np.float32),
        rotations=rotations,
        angular_velocities=np.zeros((frames, 31, 3), np.float32),
        foot_contacts=np.ones((frames, 2), np.uint8),
        hand_contacts=hand_contacts,
        hand_dof=np.zeros((frames, 14), np.float32),
        hand_dof_velocities=np.zeros((frames, 14), np.float32),
        phases=phases,
        active_hands=np.array([1, 1], np.uint8),
        time_to_contact=time_to_contact,
        object_positions=object_positions,
        object_rotations=np.broadcast_to(_IDENTITY, (frames, 4)).copy(),
        object_velocities=np.zeros((frames, 3), np.float32),
        object_angular_velocities=np.zeros((frames, 3), np.float32),
        table_positions=np.array(
            [[0.0, -10.0, 0.0], [0.0, -10.0, 0.0]], np.float32
        ),
        table_rotations=np.broadcast_to(_IDENTITY, (2, 4)).copy(),
        table_sizes=np.array([[1.0, 0.1, 1.0], [1.0, 0.1, 1.0]], np.float32),
        object_dimensions=np.array(
            [[0.1, 0.2, 0.1], [0.1, 0.2, 0.1]], np.float32
        ),
        grasp_positions_object=np.array(
            [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], np.float32
        ),
        grasp_rotations_object=np.broadcast_to(_IDENTITY, (2, 4)).copy(),
        approach_directions_object=np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32
        ),
        source_frames=np.concatenate(
            (
                np.arange(100, 110, dtype=np.int32),
                np.arange(500, 510, dtype=np.int32),
            )
        ),
    )
    artifact.validate()

    source_record = {
        "sequence_id": _SOURCE_SEQUENCE,
        "object_id": _SOURCE_OBJECT,
        "active_hand": 1,
        "range_start": 0,
        "range_stop": _CLIP_FRAMES,
    }
    target_record = {
        "sequence_id": _TARGET_SEQUENCE,
        "object_id": _TARGET_OBJECT,
        "active_hand": 1,
        "range_start": _CLIP_FRAMES,
        "range_stop": frames,
    }
    manifest = {
        "schema_version": 1,
        "dataset_id": GRAIL_DATASET_ID,
        "skeleton_names": list(G1_SKELETON.names),
        "skeleton_parents": G1_SKELETON.parents.astype(int).tolist(),
        # This is intentionally not artifact ordinal order. Each record retains
        # its true global range so stable identity can be joined by range.
        "clips": [target_record, source_record],
    }
    return artifact, manifest


def _candidate(report, sequence_id):
    matches = [
        item
        for item in report["retained_candidates"]
        if item["sequence_id"] == sequence_id
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one candidate for {sequence_id!r}, got {len(matches)}"
        )
    return matches[0]


def _yaw_quaternion(radians):
    half = np.float32(radians) * np.float32(0.5)
    return np.array(
        [np.cos(half), 0.0, np.sin(half), 0.0], np.float32
    )


def _normalized_quaternion(value):
    quaternion = np.asarray(value, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.astype(np.float32)


def _axis_angle_quaternion(axis, radians):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half = float(radians) * 0.5
    return _normalized_quaternion(
        np.concatenate(([np.cos(half)], np.sin(half) * axis))
    )


def _quaternion_multiply(left, right):
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    return _normalized_quaternion(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _quaternion_conjugate(quaternion):
    result = np.asarray(quaternion, dtype=np.float32).copy()
    result[1:] *= np.float32(-1.0)
    return result


def _rotate_by_quaternion(vector, quaternion):
    w, x, y, z = _normalized_quaternion(quaternion).astype(np.float64)
    rotation = np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        np.float64,
    )
    return (rotation @ np.asarray(vector, dtype=np.float64)).astype(
        np.float32
    )


def _orientation_boundary_quaternion(over_limit):
    quaternion = _yaw_quaternion(_HAND_ORIENTATION_LIMIT_RADIANS)
    if over_limit:
        quaternion[0] = np.nextafter(
            quaternion[0], np.float32(-np.inf)
        )
        quaternion[2] = np.sqrt(
            np.float32(1.0) - quaternion[0] * quaternion[0]
        ).astype(np.float32)
    measured = np.float32(2.0) * np.arccos(
        np.abs(quaternion[0])
    ).astype(np.float32)
    return quaternion, measured


def _rotate_y(vector, radians):
    quaternion = _yaw_quaternion(radians)
    w, _, y, _ = quaternion
    x, vertical, z = np.asarray(vector, dtype=np.float32)
    return np.array(
        [
            (w * w - y * y) * x + np.float32(2.0) * w * y * z,
            vertical,
            -np.float32(2.0) * w * y * x + (w * w - y * y) * z,
        ],
        np.float32,
    )


def _box_local_to_world(local, position, yaw_radians):
    return np.asarray(position, dtype=np.float32) + _rotate_y(
        local, yaw_radians
    )


def _set_source_wrist_world(artifact, local_frame, world_position):
    root = artifact.positions[local_frame, 0]
    artifact.positions[local_frame, _RIGHT_WRIST] = (
        np.asarray(world_position, dtype=np.float32) - root
    )


def _assert_gate_counts(
    test_case,
    report,
    expected_gate=None,
    *,
    expected_deduplicated=0,
):
    counts = report["rejected_counts_by_gate"]
    test_case.assertEqual(list(counts), list(_REJECTED_GATE_ORDER))
    test_case.assertEqual(
        counts,
        {
            gate: int(gate == expected_gate)
            for gate in _REJECTED_GATE_ORDER
        },
    )
    test_case.assertEqual(
        report["deduplicated_candidate_count"],
        expected_deduplicated,
    )


def _assert_source_rejected(test_case, report, expected_gate):
    test_case.assertNotIn(
        _SOURCE_SEQUENCE,
        {
            item["sequence_id"]
            for item in report["retained_candidates"]
        },
    )
    _assert_gate_counts(test_case, report, expected_gate)


def _position_error_fixture(error_m):
    artifact, manifest = _two_clip_artifact_and_manifest()
    artifact.object_positions[:_CLIP_FRAMES, 0] = np.float32(0.0)
    artifact.object_positions[_CLIP_FRAMES:, 0] = np.float32(0.0)
    artifact.grasp_positions_object[:, 0] = np.float32(0.0)
    source_hand = np.array([error_m, 1.2, 2.3], np.float32)
    target_hand = np.array([0.0, 1.2, 3.3], np.float32)
    artifact.positions[3:_CLIP_FRAMES, _RIGHT_WRIST] = (
        source_hand - artifact.positions[3, 0]
    )
    artifact.positions[13:, _RIGHT_WRIST] = (
        target_hand - artifact.positions[13, 0]
    )
    return artifact, manifest


def _dedupe_artifact_and_manifest(
    *,
    source_root_x=0.0,
    target_root_x=0.0,
    source_root_yaw=0.0,
    target_root_yaw=0.0,
    source_contact_error_m=0.0,
):
    artifact, manifest = _two_clip_artifact_and_manifest()
    source = slice(0, _CLIP_FRAMES)
    target = slice(_CLIP_FRAMES, 2 * _CLIP_FRAMES)
    source_entry = 1
    target_entry = _CLIP_FRAMES + 1
    source_contact = 3
    target_contact = _CLIP_FRAMES + 3

    object_position = np.array([0.0, 1.0, 0.0], np.float32)
    artifact.object_positions[source] = object_position
    artifact.object_positions[target] = object_position
    artifact.object_rotations[:] = _IDENTITY

    artifact.positions[source, 0] = np.array(
        [source_root_x, 1.0, 0.0], np.float32
    )
    artifact.positions[target, 0] = np.array(
        [target_root_x, 1.0, 0.0], np.float32
    )
    artifact.rotations[:, 0] = _IDENTITY
    artifact.rotations[source_entry, 0] = _yaw_quaternion(
        source_root_yaw
    )
    artifact.rotations[target_entry, 0] = _yaw_quaternion(
        target_root_yaw
    )

    artifact.table_positions[1] = np.array(
        [50.0, -10.0, 50.0], np.float32
    )
    target_grasp = np.array([0.25, 0.20, 0.30], np.float32)
    artifact.grasp_positions_object[1] = target_grasp
    artifact.grasp_rotations_object[1] = _IDENTITY

    precontact_hand = np.array([0.0, 2.0, 2.0], np.float32)
    target_contact_hand = object_position + target_grasp
    source_contact_hand = target_contact_hand + np.array(
        [source_contact_error_m, 0.0, 0.0], np.float32
    )
    artifact.positions[1:source_contact, _RIGHT_WRIST] = (
        precontact_hand - artifact.positions[1:source_contact, 0]
    )
    artifact.positions[
        target_entry:target_contact, _RIGHT_WRIST
    ] = precontact_hand - artifact.positions[
        target_entry:target_contact, 0
    ]
    artifact.positions[source_contact:_CLIP_FRAMES, _RIGHT_WRIST] = (
        source_contact_hand - artifact.positions[source_contact, 0]
    )
    artifact.positions[target_contact:, _RIGHT_WRIST] = (
        target_contact_hand - artifact.positions[target_contact, 0]
    )
    artifact.rotations[:, _RIGHT_WRIST] = _IDENTITY
    artifact.validate()
    return artifact, manifest


class ExtractSlotCandidateIdentityTests(unittest.TestCase):
    def test_resolves_target_events_and_provenance_by_sequence_and_range(self):
        artifact, manifest = _two_clip_artifact_and_manifest()

        report = extract_slot_candidates(
            artifact, manifest, _TARGET_SEQUENCE
        )

        self.assertEqual(report["dataset_id"], GRAIL_DATASET_ID)
        self.assertEqual(report["schema_version"], 1)
        expected_target = {
            "sequence_id": _TARGET_SEQUENCE,
            "object_id": _TARGET_OBJECT,
            "active_hand": 1,
            "range_start": 10,
            "range_stop": 20,
            "entry_local_frame": 1,
            "contact_local_frame": 3,
            "lift_local_frame": 4,
            "hold_local_frame": 5,
            "stop_local_frame": 10,
            "object_alignment_local_frame": 2,
            "object_alignment_position": [4.0, 1.0, 3.0],
            "object_alignment_rotation": [1.0, 0.0, 0.0, 0.0],
            "table_position": artifact.table_positions[1]
            .astype(float)
            .tolist(),
            "table_rotation": artifact.table_rotations[1]
            .astype(float)
            .tolist(),
            "table_size": artifact.table_sizes[1]
            .astype(float)
            .tolist(),
            "object_dimensions": artifact.object_dimensions[1]
            .astype(float)
            .tolist(),
            "grasp_position_object": artifact.grasp_positions_object[1]
            .astype(float)
            .tolist(),
            "grasp_rotation_object": artifact.grasp_rotations_object[1]
            .astype(float)
            .tolist(),
            "approach_direction_object": artifact.approach_directions_object[
                1
            ]
            .astype(float)
            .tolist(),
        }
        for key, expected in expected_target.items():
            with self.subTest(target_key=key):
                self.assertEqual(report["target"][key], expected)

        source = _candidate(report, _SOURCE_SEQUENCE)
        self.assertEqual(
            [
                source["entry_local_frame"],
                source["contact_local_frame"],
                source["lift_local_frame"],
                source["hold_local_frame"],
                source["stop_local_frame"],
                source["object_alignment_local_frame"],
            ],
            [1, 3, 4, 5, 10, 2],
        )
        self.assertEqual(
            source["stable_key"],
            [GRAIL_DATASET_ID, 1, _SOURCE_SEQUENCE, 1, 1],
        )
        self.assertEqual(source["source_entry_frame"], 101)
        self.assertNotIn(source["source_entry_frame"], source["stable_key"])

        target = _candidate(report, _TARGET_SEQUENCE)
        self.assertEqual(
            target["stable_key"],
            [GRAIL_DATASET_ID, 1, _TARGET_SEQUENCE, 1, 1],
        )
        self.assertEqual(target["source_entry_frame"], 501)


class ExtractSlotCandidateTransformAndCompatibilityTests(unittest.TestCase):
    def test_extracts_exact_slot_fields_under_a_quarter_turn_target_yaw(self):
        artifact, manifest = _two_clip_artifact_and_manifest()
        source = slice(0, _CLIP_FRAMES)
        target = slice(_CLIP_FRAMES, 2 * _CLIP_FRAMES)
        artifact.positions[source, 0] = np.array(
            [2.0, 1.0, 4.0], np.float32
        )
        artifact.rotations[1, 0] = _yaw_quaternion(np.deg2rad(30.0))
        artifact.velocities[1, 0] = np.array(
            [3.0, 12.0, 4.0], np.float32
        )

        source_hand = np.array([1.1, 1.2, 2.3], np.float32)
        shoulder_rotation = _axis_angle_quaternion(
            [1.0, 0.0, 0.0], np.deg2rad(35.0)
        )
        shoulder_roll_rotation = _axis_angle_quaternion(
            [0.0, 0.0, 1.0], np.deg2rad(-20.0)
        )
        wrist_parent_world = _quaternion_multiply(
            shoulder_rotation, shoulder_roll_rotation
        )
        wrist_parent_inverse = _quaternion_conjugate(wrist_parent_world)
        source_wrist_local = _rotate_by_quaternion(
            source_hand - artifact.positions[3, 0],
            wrist_parent_inverse,
        )
        artifact.rotations[3:_CLIP_FRAMES, 24] = shoulder_rotation
        artifact.rotations[3:_CLIP_FRAMES, 25] = shoulder_roll_rotation
        artifact.positions[3:_CLIP_FRAMES, _RIGHT_WRIST] = (
            source_wrist_local
        )
        artifact.rotations[3:_CLIP_FRAMES, _RIGHT_WRIST] = (
            wrist_parent_inverse
        )
        np.testing.assert_allclose(
            artifact.positions[3, 0]
            + _rotate_by_quaternion(source_wrist_local, wrist_parent_world),
            source_hand,
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            _quaternion_multiply(wrist_parent_world, wrist_parent_inverse),
            _IDENTITY,
            rtol=0.0,
            atol=1.0e-6,
        )

        # Only target metadata may define the destination collision geometry
        # and affordance. These source values would reject or misalign the
        # candidate if selected by clip ordinal instead.
        artifact.table_positions[0] = np.array(
            [2.0, 1.0, 4.0], np.float32
        )
        artifact.table_rotations[0] = _yaw_quaternion(np.deg2rad(-33.0))
        artifact.table_sizes[0] = np.array([20.0, 20.0, 20.0], np.float32)
        artifact.object_dimensions[0] = np.array(
            [8.0, 9.0, 10.0], np.float32
        )
        artifact.grasp_positions_object[0] = np.array(
            [-6.0, 5.0, -4.0], np.float32
        )
        artifact.grasp_rotations_object[0] = _axis_angle_quaternion(
            [1.0, 1.0, 0.0], np.deg2rad(70.0)
        )

        quarter_turn = np.float32(np.pi / 2.0)
        target_hand = artifact.object_positions[12] + _rotate_y(
            source_hand - artifact.object_positions[2], quarter_turn
        )
        np.testing.assert_allclose(
            target_hand,
            np.array([4.3, 1.2, 2.9], np.float32),
            rtol=0.0,
            atol=1.0e-6,
        )
        artifact.object_rotations[target] = _yaw_quaternion(quarter_turn)
        artifact.positions[13:, _RIGHT_WRIST] = (
            target_hand - artifact.positions[13, 0]
        )
        artifact.rotations[13:, _RIGHT_WRIST] = _yaw_quaternion(quarter_turn)

        report = extract_slot_candidates(
            artifact, manifest, _TARGET_SEQUENCE
        )
        candidate = _candidate(report, _SOURCE_SEQUENCE)

        actual_fields = {
            field: candidate[field] for field in _SLOT_CANDIDATE_FIELDS
        }
        position_error = actual_fields.pop(
            "contact_hand_position_error_m"
        )
        orientation_error = actual_fields.pop(
            "contact_hand_orientation_error_radians"
        )
        self.assertEqual(
            actual_fields,
            {
                "root_x_object_m": 1.0,
                "root_z_object_m": 2.0,
                "root_yaw_object_radians": float(
                    np.float32(np.deg2rad(30.0))
                ),
                "entry_root_planar_speed_mps": 5.0,
                "root_table_clear": True,
                "hand_table_clear": True,
                "hand_object_clear": True,
                "static_path_feasible": True,
            },
        )
        self.assertGreaterEqual(position_error, 0.0)
        self.assertLessEqual(position_error, 1.0e-6)
        self.assertGreaterEqual(orientation_error, 0.0)
        self.assertLessEqual(orientation_error, 1.0e-6)
        _assert_gate_counts(self, report)

    def test_alignment_uses_contact_minus_one_yaws_and_preserves_source_y(self):
        artifact, manifest = _two_clip_artifact_and_manifest()
        source = slice(0, _CLIP_FRAMES)
        target = slice(_CLIP_FRAMES, 2 * _CLIP_FRAMES)
        source_object_position = np.array(
            [-2.0, -1.5, 1.0], np.float32
        )
        target_object_position = np.array(
            [5.0, 2.5, -4.0], np.float32
        )
        source_object_yaw = np.float32(np.deg2rad(35.0))
        target_object_yaw = np.float32(np.deg2rad(-70.0))
        source_object_rotation = _yaw_quaternion(source_object_yaw)
        target_object_rotation = _yaw_quaternion(target_object_yaw)
        target_grasp_position = np.array(
            [0.3, -0.4, 0.2], np.float32
        )
        target_grasp_rotation = _axis_angle_quaternion(
            [1.0, 0.0, 0.0], np.deg2rad(30.0)
        )

        artifact.object_positions[source] = source_object_position
        artifact.object_positions[target] = target_object_position
        artifact.object_rotations[source] = source_object_rotation
        artifact.object_rotations[target] = target_object_rotation
        # Contact is poison: alignment authority is Contact-minus-one.
        artifact.object_positions[3] = np.array(
            [100.0, -50.0, 200.0], np.float32
        )
        artifact.object_positions[13] = np.array(
            [-100.0, 50.0, -200.0], np.float32
        )
        artifact.object_rotations[3] = _yaw_quaternion(
            np.deg2rad(155.0)
        )
        artifact.object_rotations[13] = _yaw_quaternion(
            np.deg2rad(115.0)
        )
        artifact.grasp_positions_object[1] = target_grasp_position
        artifact.grasp_rotations_object[1] = target_grasp_rotation
        artifact.grasp_positions_object[0] = np.array(
            [-6.0, 5.0, -4.0], np.float32
        )
        artifact.grasp_rotations_object[0] = _axis_angle_quaternion(
            [0.0, 1.0, 1.0], np.deg2rad(80.0)
        )

        target_hand = target_object_position + _rotate_y(
            target_grasp_position, target_object_yaw
        )
        source_hand = source_object_position + _rotate_y(
            target_grasp_position, source_object_yaw
        )
        # Candidate grasp metadata is measured, not trusted. Authoring the
        # recorded wrist at the target-required Y makes yaw-only alignment
        # correct; a full object-Y translation would introduce a 4 m error.
        source_hand[1] = target_hand[1]
        source_hand_rotation = _quaternion_multiply(
            source_object_rotation, target_grasp_rotation
        )
        target_hand_rotation = _quaternion_multiply(
            target_object_rotation, target_grasp_rotation
        )

        expected_root_x = np.float32(1.25)
        expected_root_z = np.float32(-0.75)
        expected_root_yaw = np.float32(np.deg2rad(-25.0))
        source_root = source_object_position + _rotate_y(
            [expected_root_x, 0.0, expected_root_z],
            source_object_yaw,
        )
        source_root[1] = np.float32(6.25)
        artifact.positions[source, 0] = source_root
        artifact.rotations[1, 0] = _yaw_quaternion(
            source_object_yaw + expected_root_yaw
        )
        artifact.positions[3:_CLIP_FRAMES, _RIGHT_WRIST] = (
            source_hand - source_root
        )
        artifact.rotations[3:_CLIP_FRAMES, _RIGHT_WRIST] = (
            source_hand_rotation
        )
        artifact.positions[13:, _RIGHT_WRIST] = (
            target_hand - artifact.positions[13, 0]
        )
        artifact.rotations[13:, _RIGHT_WRIST] = target_hand_rotation

        report = extract_slot_candidates(
            artifact, manifest, _TARGET_SEQUENCE
        )
        candidate = _candidate(report, _SOURCE_SEQUENCE)

        self.assertAlmostEqual(
            candidate["root_x_object_m"], float(expected_root_x), places=6
        )
        self.assertAlmostEqual(
            candidate["root_z_object_m"], float(expected_root_z), places=6
        )
        self.assertAlmostEqual(
            candidate["root_yaw_object_radians"],
            float(expected_root_yaw),
            places=6,
        )
        self.assertLessEqual(
            candidate["contact_hand_position_error_m"], 1.0e-6
        )
        self.assertLessEqual(
            candidate["contact_hand_orientation_error_radians"], 1.0e-6
        )

    def test_root_yaw_is_wrapped_across_positive_pi(self):
        artifact, manifest = _two_clip_artifact_and_manifest()
        artifact.rotations[1, 0] = _yaw_quaternion(np.deg2rad(200.0))

        report = extract_slot_candidates(
            artifact, manifest, _TARGET_SEQUENCE
        )

        candidate = _candidate(report, _SOURCE_SEQUENCE)
        self.assertAlmostEqual(
            candidate["root_yaw_object_radians"],
            float(np.float32(np.deg2rad(-160.0))),
            places=6,
        )

    def test_contact_position_gate_accepts_equality_and_rejects_next_float(self):
        cases = (
            (_HAND_POSITION_LIMIT_M, True),
            (
                np.nextafter(
                    _HAND_POSITION_LIMIT_M, np.float32(np.inf)
                ),
                False,
            ),
        )
        for error_m, accepted in cases:
            with self.subTest(error_m=float(error_m)):
                artifact, manifest = _position_error_fixture(error_m)
                report = extract_slot_candidates(
                    artifact, manifest, _TARGET_SEQUENCE
                )
                if accepted:
                    candidate = _candidate(report, _SOURCE_SEQUENCE)
                    self.assertEqual(
                        candidate["contact_hand_position_error_m"],
                        float(error_m),
                    )
                    _assert_gate_counts(self, report)
                else:
                    self.assertNotIn(
                        _SOURCE_SEQUENCE,
                        {
                            item["sequence_id"]
                            for item in report["retained_candidates"]
                        },
                    )
                    _assert_gate_counts(self, report, "contact_position")

    def test_contact_orientation_gate_rejects_next_measurable_quaternion(self):
        cases = ((False, True), (True, False))
        for over_limit, accepted in cases:
            with self.subTest(over_limit=over_limit):
                quaternion, measured_error = (
                    _orientation_boundary_quaternion(over_limit)
                )
                self.assertEqual(
                    np.dot(quaternion, quaternion), np.float32(1.0)
                )
                if over_limit:
                    self.assertGreater(
                        measured_error,
                        _HAND_ORIENTATION_LIMIT_RADIANS,
                    )
                else:
                    self.assertEqual(
                        measured_error,
                        _HAND_ORIENTATION_LIMIT_RADIANS,
                    )
                artifact, manifest = _two_clip_artifact_and_manifest()
                artifact.rotations[3:_CLIP_FRAMES, _RIGHT_WRIST] = (
                    quaternion
                )
                report = extract_slot_candidates(
                    artifact, manifest, _TARGET_SEQUENCE
                )
                if accepted:
                    candidate = _candidate(report, _SOURCE_SEQUENCE)
                    self.assertAlmostEqual(
                        candidate[
                            "contact_hand_orientation_error_radians"
                        ],
                        float(measured_error),
                        places=6,
                    )
                    _assert_gate_counts(self, report)
                else:
                    self.assertNotIn(
                        _SOURCE_SEQUENCE,
                        {
                            item["sequence_id"]
                            for item in report["retained_candidates"]
                        },
                    )
                    _assert_gate_counts(
                        self, report, "contact_orientation"
                    )


class ExtractSlotGeometryBoundaryTests(unittest.TestCase):
    def test_identity_table_root_boundary_is_closed_in_float64(self):
        table_position = np.zeros(3, np.float64)
        table_rotation = _IDENTITY.astype(np.float64)
        table_size = np.array([2.0, 0.1, 4.0], np.float64)
        effective_half_x = (
            np.float64(0.5) * np.float64(table_size[0])
            + np.float64(_ROOT_TABLE_EXPANSION_M)
        )
        outward_half_x = np.nextafter(effective_half_x, np.inf)
        boundary = np.array(
            [effective_half_x, 37.0, 0.0], np.float64
        )
        outward = np.array(
            [outward_half_x, 37.0, 0.0], np.float64
        )

        self.assertTrue(
            _root_intersects_table(
                boundary, table_position, table_rotation, table_size
            )
        )
        self.assertFalse(
            _root_intersects_table(
                outward, table_position, table_rotation, table_size
            )
        )

    def test_yawed_table_maps_interior_and_clear_points_away_from_boundary(self):
        table_position = np.array([3.0, 0.0, -2.0], np.float64)
        table_rotation = _yaw_quaternion(np.pi / 2.0).astype(np.float64)
        table_size = np.array([2.0, 0.1, 4.0], np.float64)
        # A +90-degree table yaw maps local +X to world -Z.
        interior = np.array([3.0, 37.0, -2.5], np.float64)
        clear = np.array([3.0, 37.0, -3.5], np.float64)

        self.assertTrue(
            _root_intersects_table(
                interior, table_position, table_rotation, table_size
            )
        )
        self.assertFalse(
            _root_intersects_table(
                clear, table_position, table_rotation, table_size
            )
        )

    def test_identity_box_tangent_is_closed_in_float64(self):
        box_position = np.zeros(3, np.float64)
        box_rotation = _IDENTITY.astype(np.float64)
        box_size = np.array([2.0, 4.0, 6.0], np.float64)
        expansion = np.float64(_CLEARANCE_RADIUS_M)
        effective_half_x = (
            np.float64(0.5) * np.float64(box_size[0])
            + np.float64(_CLEARANCE_RADIUS_M)
        )
        outward_half_x = np.nextafter(effective_half_x, np.inf)
        boundary_start = np.array(
            [effective_half_x, -3.0, 0.0], np.float64
        )
        boundary_stop = np.array(
            [effective_half_x, 3.0, 0.0], np.float64
        )
        outward_start = np.array(
            [outward_half_x, -3.0, 0.0], np.float64
        )
        outward_stop = np.array(
            [outward_half_x, 3.0, 0.0], np.float64
        )

        self.assertTrue(
            _segment_intersects_expanded_box(
                boundary_start,
                boundary_stop,
                box_position,
                box_rotation,
                box_size,
                expansion,
            )
        )
        self.assertFalse(
            _segment_intersects_expanded_box(
                outward_start,
                outward_stop,
                box_position,
                box_rotation,
                box_size,
                expansion,
            )
        )

    def test_fully_oriented_box_maps_interior_and_clear_segments(self):
        box_position = np.zeros(3, np.float64)
        # Exact permutation: local (x, y, z) maps to world (z, x, y).
        box_rotation = np.array([0.5, 0.5, 0.5, 0.5], np.float64)
        box_size = np.array([2.0, 4.0, 6.0], np.float64)
        expansion = np.float64(_CLEARANCE_RADIUS_M)
        interior_start = np.array([0.0, 0.5, -3.0], np.float64)
        interior_stop = np.array([0.0, 0.5, 3.0], np.float64)
        clear_start = np.array([0.0, 1.5, -3.0], np.float64)
        clear_stop = np.array([0.0, 1.5, 3.0], np.float64)

        self.assertTrue(
            _segment_intersects_expanded_box(
                interior_start,
                interior_stop,
                box_position,
                box_rotation,
                box_size,
                expansion,
            )
        )
        self.assertFalse(
            _segment_intersects_expanded_box(
                clear_start,
                clear_stop,
                box_position,
                box_rotation,
                box_size,
                expansion,
            )
        )


class ExtractSlotCandidatePathGateTests(unittest.TestCase):
    def test_entry_clear_candidate_is_rejected_when_stop_minus_one_root_hits_table(self):
        artifact, manifest = _two_clip_artifact_and_manifest()
        artifact.table_positions[1] = np.array(
            [5.0, 0.0, 1.0], np.float32
        )
        artifact.table_sizes[1] = np.array(
            [0.02, 0.02, 0.02], np.float32
        )
        # Local frame 9 is stop - 1 and remains in Hold.
        artifact.positions[9, 0] = np.array([2.0, 1.0, 0.0], np.float32)

        table_position = artifact.table_positions[1]
        table_rotation = artifact.table_rotations[1]
        table_size = artifact.table_sizes[1]
        self.assertFalse(
            _root_intersects_table(
                np.array([3.0, 1.0, 1.0], np.float32),
                table_position,
                table_rotation,
                table_size,
            )
        )
        self.assertTrue(
            _root_intersects_table(
                np.array([5.0, 1.0, 1.0], np.float32),
                table_position,
                table_rotation,
                table_size,
            )
        )

        report = extract_slot_candidates(
            artifact, manifest, _TARGET_SEQUENCE
        )

        _assert_source_rejected(self, report, "root_table")

    def test_hand_table_sweep_precedes_same_segment_object_collision(self):
        artifact, manifest = _two_clip_artifact_and_manifest()
        mapped_start = np.array([3.9, 1.0, 2.7], np.float32)
        mapped_stop = np.array([4.1, 1.2, 3.3], np.float32)
        scene_translation = np.array([3.0, 0.0, 1.0], np.float32)
        _set_source_wrist_world(
            artifact, 1, mapped_start - scene_translation
        )
        _set_source_wrist_world(
            artifact, 2, mapped_stop - scene_translation
        )

        artifact.table_positions[1] = np.array(
            [4.0, 1.1, 3.0], np.float32
        )
        artifact.table_sizes[1] = np.array(
            [0.02, 0.02, 0.02], np.float32
        )
        self.assertTrue(
            _segment_intersects_expanded_box(
                mapped_start,
                mapped_stop,
                artifact.table_positions[1],
                artifact.table_rotations[1],
                artifact.table_sizes[1],
                _CLEARANCE_RADIUS_M,
            )
        )
        self.assertTrue(
            _segment_intersects_expanded_box(
                mapped_start,
                mapped_stop,
                artifact.object_positions[12],
                artifact.object_rotations[12],
                artifact.object_dimensions[1],
                _CLEARANCE_RADIUS_M,
            )
        )

        report = extract_slot_candidates(
            artifact, manifest, _TARGET_SEQUENCE
        )

        _assert_source_rejected(self, report, "hand_table")

    def test_contact_minus_one_segment_is_checked_but_contact_segment_is_exempt(self):
        mapped_start = np.array([3.9, 1.0, 2.7], np.float32)
        mapped_stop = np.array([4.1, 1.2, 3.3], np.float32)
        scene_translation = np.array([3.0, 0.0, 1.0], np.float32)

        rejected, rejected_manifest = _two_clip_artifact_and_manifest()
        self.assertTrue(
            _segment_intersects_expanded_box(
                mapped_start,
                mapped_stop,
                rejected.object_positions[12],
                rejected.object_rotations[12],
                rejected.object_dimensions[1],
                _CLEARANCE_RADIUS_M,
            )
        )
        _set_source_wrist_world(
            rejected, 1, mapped_start - scene_translation
        )
        _set_source_wrist_world(
            rejected, 2, mapped_stop - scene_translation
        )
        rejected_report = extract_slot_candidates(
            rejected, rejected_manifest, _TARGET_SEQUENCE
        )
        _assert_source_rejected(self, rejected_report, "hand_object")

        accepted, accepted_manifest = _two_clip_artifact_and_manifest()
        _set_source_wrist_world(
            accepted, 2, mapped_start - scene_translation
        )
        accepted_report = extract_slot_candidates(
            accepted, accepted_manifest, _TARGET_SEQUENCE
        )
        candidate = _candidate(accepted_report, _SOURCE_SEQUENCE)
        self.assertTrue(candidate["hand_object_clear"])
        self.assertTrue(candidate["static_path_feasible"])
        _assert_gate_counts(self, accepted_report)

    def test_hand_mismatch_is_counted_before_all_geometry_gates(self):
        artifact, manifest = _two_clip_artifact_and_manifest()
        artifact.active_hands[0] = np.uint8(0)
        artifact.hand_contacts[:_CLIP_FRAMES] = np.uint8(0)
        artifact.hand_contacts[3:_CLIP_FRAMES, 0] = np.uint8(1)
        source_record = next(
            record
            for record in manifest["clips"]
            if record["sequence_id"] == _SOURCE_SEQUENCE
        )
        source_record["active_hand"] = 0

        report = extract_slot_candidates(
            artifact, manifest, _TARGET_SEQUENCE
        )

        _assert_source_rejected(self, report, "hand_mismatch")

    def test_contact_position_precedes_a_later_root_table_collision(self):
        overshoot = np.nextafter(
            _HAND_POSITION_LIMIT_M, np.float32(np.inf)
        )
        artifact, manifest = _position_error_fixture(overshoot)
        artifact.table_positions[1] = np.array(
            [0.0, 0.0, 1.0], np.float32
        )
        artifact.table_sizes[1] = np.array(
            [0.02, 0.02, 0.02], np.float32
        )
        self.assertTrue(
            _root_intersects_table(
                np.array([0.0, 1.0, 1.0], np.float32),
                artifact.table_positions[1],
                artifact.table_rotations[1],
                artifact.table_sizes[1],
            )
        )

        report = extract_slot_candidates(
            artifact, manifest, _TARGET_SEQUENCE
        )

        _assert_source_rejected(self, report, "contact_position")

    def test_contact_orientation_precedes_root_and_hand_table_collisions(self):
        artifact, manifest = _two_clip_artifact_and_manifest()
        over_limit, measured_error = _orientation_boundary_quaternion(True)
        self.assertGreater(
            measured_error, _HAND_ORIENTATION_LIMIT_RADIANS
        )
        artifact.rotations[3:_CLIP_FRAMES, _RIGHT_WRIST] = over_limit
        artifact.table_positions[1] = np.array(
            [3.0, 1.0, 1.0], np.float32
        )
        artifact.table_sizes[1] = np.array(
            [0.02, 0.02, 0.02], np.float32
        )
        mapped_entry = np.array([3.0, 1.0, 1.0], np.float32)
        self.assertTrue(
            _root_intersects_table(
                mapped_entry,
                artifact.table_positions[1],
                artifact.table_rotations[1],
                artifact.table_sizes[1],
            )
        )
        self.assertTrue(
            _segment_intersects_expanded_box(
                mapped_entry,
                mapped_entry,
                artifact.table_positions[1],
                artifact.table_rotations[1],
                artifact.table_sizes[1],
                _CLEARANCE_RADIUS_M,
            )
        )

        report = extract_slot_candidates(
            artifact, manifest, _TARGET_SEQUENCE
        )

        _assert_source_rejected(self, report, "contact_orientation")

    def test_root_table_precedes_hand_table_and_object_collisions(self):
        artifact, manifest = _two_clip_artifact_and_manifest()
        mapped_start = np.array([3.9, 1.0, 2.7], np.float32)
        mapped_stop = np.array([4.1, 1.2, 3.3], np.float32)
        scene_translation = np.array([3.0, 0.0, 1.0], np.float32)
        _set_source_wrist_world(
            artifact, 1, mapped_start - scene_translation
        )
        _set_source_wrist_world(
            artifact, 2, mapped_stop - scene_translation
        )
        artifact.table_positions[1] = np.array(
            [4.0, 1.0, 3.0], np.float32
        )
        artifact.table_sizes[1] = np.array(
            [2.0, 0.02, 4.0], np.float32
        )
        self.assertTrue(
            _root_intersects_table(
                np.array([3.0, 1.0, 1.0], np.float32),
                artifact.table_positions[1],
                artifact.table_rotations[1],
                artifact.table_sizes[1],
            )
        )
        for position, rotation, size in (
            (
                artifact.table_positions[1],
                artifact.table_rotations[1],
                artifact.table_sizes[1],
            ),
            (
                artifact.object_positions[12],
                artifact.object_rotations[12],
                artifact.object_dimensions[1],
            ),
        ):
            self.assertTrue(
                _segment_intersects_expanded_box(
                    mapped_start,
                    mapped_stop,
                    position,
                    rotation,
                    size,
                    _CLEARANCE_RADIUS_M,
                )
            )

        report = extract_slot_candidates(
            artifact, manifest, _TARGET_SEQUENCE
        )

        _assert_source_rejected(self, report, "root_table")


class ExtractSlotCandidateDedupeAndOutputTests(unittest.TestCase):
    def test_opposite_manifest_orders_serialize_identically_in_stable_key_order(self):
        outside_distance = np.nextafter(
            _DEDUPE_POSITION_LIMIT_M, np.float32(np.inf)
        )
        cases = (
            ("outside_distance", outside_distance, 0),
            ("inclusive_duplicate", _DEDUPE_POSITION_LIMIT_M, 1),
        )
        for name, target_root_x, deduplicated in cases:
            with self.subTest(name=name):
                artifact, reverse_manifest = (
                    _dedupe_artifact_and_manifest(
                        target_root_x=target_root_x
                    )
                )
                forward_manifest = copy.deepcopy(reverse_manifest)
                forward_manifest["clips"].reverse()

                reverse_report = extract_slot_candidates(
                    copy.deepcopy(artifact),
                    reverse_manifest,
                    _TARGET_SEQUENCE,
                )
                forward_report = extract_slot_candidates(
                    copy.deepcopy(artifact),
                    forward_manifest,
                    _TARGET_SEQUENCE,
                )
                reverse_text = serialize_slot_report(reverse_report)
                forward_text = serialize_slot_report(forward_report)

                self.assertEqual(
                    reverse_text.encode("utf-8"),
                    forward_text.encode("utf-8"),
                )
                for report in (reverse_report, forward_report):
                    stable_keys = [
                        candidate["stable_key"]
                        for candidate in report["retained_candidates"]
                    ]
                    self.assertEqual(stable_keys, sorted(stable_keys))
                    self.assertEqual(
                        len(stable_keys), 2 - deduplicated
                    )
                    if deduplicated:
                        self.assertEqual(
                            report["retained_candidates"][0][
                                "sequence_id"
                            ],
                            _SOURCE_SEQUENCE,
                        )
                    _assert_gate_counts(
                        self,
                        report,
                        expected_deduplicated=deduplicated,
                    )

    def test_private_dedupe_accepts_exact_promoted_float32_yaw_limit(self):
        first = self._minimal_dedupe_row("a", hand=1)
        boundary = self._minimal_dedupe_row(
            "b",
            hand=1,
            yaw=float(_DEDUPE_YAW_LIMIT_RADIANS),
        )

        retained, deduplicated = _dedupe_candidates(
            [boundary, first]
        )

        self.assertEqual(
            [candidate["stable_key"] for candidate in retained],
            [first["stable_key"]],
        )
        self.assertEqual(deduplicated, 1)

    def test_private_dedupe_is_stable_key_sorted_and_greedy(self):
        first = self._minimal_dedupe_row("a", hand=1, root_x=0.0)
        middle = self._minimal_dedupe_row("b", hand=1, root_x=0.09)
        last = self._minimal_dedupe_row("c", hand=1, root_x=0.18)

        retained, deduplicated = _dedupe_candidates(
            [last, middle, first]
        )

        self.assertEqual(
            [candidate["stable_key"] for candidate in retained],
            [first["stable_key"], last["stable_key"]],
        )
        self.assertEqual(deduplicated, 1)

    def test_private_dedupe_keeps_an_opposite_hand_duplicate(self):
        right = self._minimal_dedupe_row("a", hand=1)
        left = self._minimal_dedupe_row("b", hand=0)

        retained, deduplicated = _dedupe_candidates([left, right])

        self.assertEqual(
            [candidate["stable_key"] for candidate in retained],
            [right["stable_key"], left["stable_key"]],
        )
        self.assertEqual(deduplicated, 0)

    @staticmethod
    def _minimal_dedupe_row(
        suffix,
        *,
        hand,
        root_x=0.0,
        root_z=0.0,
        yaw=0.0,
    ):
        return {
            "stable_key": ["dataset", 1, suffix, 1, hand],
            "active_hand": hand,
            "root_x_object_m": root_x,
            "root_z_object_m": root_z,
            "root_yaw_object_radians": yaw,
        }

    def test_same_hand_dedupe_uses_inclusive_distance_and_wrapped_yaw(self):
        outside_distance = np.nextafter(
            _DEDUPE_POSITION_LIMIT_M, np.float32(np.inf)
        )
        outside_yaw = np.float32(np.deg2rad(10.01))
        self.assertGreater(outside_distance, _DEDUPE_POSITION_LIMIT_M)
        self.assertGreater(outside_yaw, _DEDUPE_YAW_LIMIT_RADIANS)
        cases = (
            (
                "inclusive_limits",
                _DEDUPE_POSITION_LIMIT_M,
                np.float32(0.0),
                _DEDUPE_YAW_LIMIT_RADIANS,
                1,
            ),
            (
                "distance_one_float_outside",
                outside_distance,
                np.float32(0.0),
                _DEDUPE_YAW_LIMIT_RADIANS,
                0,
            ),
            (
                "yaw_just_outside",
                _DEDUPE_POSITION_LIMIT_M,
                np.float32(0.0),
                outside_yaw,
                0,
            ),
            (
                "wrapped_positive_179_to_negative_171",
                _DEDUPE_POSITION_LIMIT_M,
                np.float32(np.deg2rad(179.0)),
                np.float32(np.deg2rad(-171.0)),
                1,
            ),
        )

        for name, distance, source_yaw, target_yaw, deduplicated in cases:
            with self.subTest(name=name):
                artifact, manifest = _dedupe_artifact_and_manifest(
                    target_root_x=distance,
                    source_root_yaw=source_yaw,
                    target_root_yaw=target_yaw,
                )
                if name == "wrapped_positive_179_to_negative_171":
                    # Compose the float32 ten-degree delta onto +179 rather
                    # than independently rounding both endpoint angles. The
                    # resulting quaternion is the -171-degree orientation and
                    # keeps the wrapped separation on the inclusive side.
                    artifact.rotations[_CLIP_FRAMES + 1, 0] = (
                        _quaternion_multiply(
                            _yaw_quaternion(
                                _DEDUPE_YAW_LIMIT_RADIANS
                            ),
                            artifact.rotations[1, 0],
                        )
                    )
                    target_quaternion = artifact.rotations[
                        _CLIP_FRAMES + 1, 0
                    ]
                    target_angle = np.float64(2.0) * np.arctan2(
                        np.float64(target_quaternion[2]),
                        np.float64(target_quaternion[0]),
                    )
                    wrapped_target_angle = np.arctan2(
                        np.sin(target_angle), np.cos(target_angle)
                    )
                    self.assertAlmostEqual(
                        np.rad2deg(wrapped_target_angle),
                        -171.0,
                        places=5,
                    )
                report = extract_slot_candidates(
                    artifact, manifest, _TARGET_SEQUENCE
                )

                expected_retained = 2 - deduplicated
                self.assertEqual(
                    len(report["retained_candidates"]),
                    expected_retained,
                )
                self.assertTrue(
                    all(
                        candidate["active_hand"] == 1
                        for candidate in report["retained_candidates"]
                    )
                )
                stable_keys = [
                    candidate["stable_key"]
                    for candidate in report["retained_candidates"]
                ]
                self.assertEqual(stable_keys, sorted(stable_keys))
                if deduplicated:
                    self.assertEqual(
                        report["retained_candidates"][0]["sequence_id"],
                        _SOURCE_SEQUENCE,
                    )
                _assert_gate_counts(
                    self,
                    report,
                    expected_deduplicated=deduplicated,
                )

    def test_static_rejection_precedes_dedupe(self):
        artifact, manifest = _dedupe_artifact_and_manifest(
            source_contact_error_m=np.float32(0.13)
        )

        report = extract_slot_candidates(
            artifact, manifest, _TARGET_SEQUENCE
        )

        self.assertEqual(
            [
                candidate["sequence_id"]
                for candidate in report["retained_candidates"]
            ],
            [_TARGET_SEQUENCE],
        )
        _assert_gate_counts(self, report, "contact_position")

    def test_candidate_schema_contains_no_runtime_authority_or_global_identity(self):
        outside_distance = np.nextafter(
            _DEDUPE_POSITION_LIMIT_M, np.float32(np.inf)
        )
        artifact, manifest = _dedupe_artifact_and_manifest(
            target_root_x=outside_distance
        )
        report = extract_slot_candidates(
            artifact, manifest, _TARGET_SEQUENCE
        )

        self.assertEqual(set(report), _REPORT_KEYS)
        self.assertEqual(set(report["target"]), _TARGET_KEYS)
        self.assertNotIn("candidate_counts", report)
        self.assertNotIn("near_duplicate", report["rejected_counts_by_gate"])
        self.assertEqual(
            set(report["rejected_counts_by_gate"]),
            set(_REJECTED_GATE_ORDER),
        )
        self.assertEqual(report["deduplicated_candidate_count"], 0)
        for candidate in report["retained_candidates"]:
            self.assertEqual(set(candidate), _SLOT_CANDIDATE_KEYS)
            self.assertIn("source_entry_frame", candidate)
            self.assertTrue(candidate["static_path_feasible"])

        serialized = serialize_slot_report(report)
        normalized_text = serialized.lower()
        for forbidden in (
            "runtime_clip_allowlist",
            "runtime_allowlist",
            "runtime allowlist",
            "match_ready",
            "match-ready",
            "runtime_certified",
            "runtime-certified",
            "near_duplicate",
            "candidate_counts",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, normalized_text)


class ExtractSlotReportSerializationTests(unittest.TestCase):
    def test_serializer_is_canonical_and_normalizes_float32_values(self):
        value = {
            "z_nested": {
                "values": [
                    float(np.float32(0.123456789)),
                    -0.0,
                ]
            },
            "middle_float": float(np.float32(1.23456789)),
            "a_large_float": float(np.float32(123456.789)),
        }

        serialized = serialize_slot_report(value)

        self.assertEqual(
            serialized,
            "{\n"
            '  "a_large_float": 123456.789,\n'
            '  "middle_float": 1.23456788,\n'
            '  "z_nested": {\n'
            '    "values": [\n'
            "      0.123456791,\n"
            "      0.0\n"
            "    ]\n"
            "  }\n"
            "}\n",
        )
        self.assertEqual(
            list(json.loads(serialized)),
            sorted(value),
        )
        self.assertTrue(serialized.endswith("\n"))
        self.assertFalse(serialized.endswith("\n\n"))

    def test_serializer_rejects_every_nonfinite_json_number(self):
        cases = (
            ("nested_dict_nan", {"outer": {"value": float("nan")}}),
            ("nested_list_inf", {"outer": [0.0, float("inf")]}),
            (
                "dict_then_list_negative_inf",
                {"outer": {"values": [float("-inf")]}},
            ),
        )
        for name, value in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError,
                    r"(?i)(non.?finite|out of range|nan|infinity)",
                ):
                    serialize_slot_report({"value": value})

    def test_nonfinite_artifact_geometry_is_rejected_before_candidate_sorting(self):
        messages = []
        error_types = []
        for reverse_clips in (False, True):
            artifact, manifest = _dedupe_artifact_and_manifest(
                target_root_x=np.nextafter(
                    _DEDUPE_POSITION_LIMIT_M, np.float32(np.inf)
                )
            )
            if reverse_clips:
                manifest["clips"].reverse()
            artifact.positions[1, 0, 0] = np.float32(np.nan)

            with self.assertRaisesRegex(
                ValueError, r"(?i)non.?finite"
            ) as raised:
                extract_slot_candidates(
                    artifact, manifest, _TARGET_SEQUENCE
                )
            messages.append(str(raised.exception))
            error_types.append(type(raised.exception))

        self.assertEqual(error_types[0], error_types[1])
        self.assertEqual(messages[0], messages[1])


class ExtractSlotCandidateValidationTests(unittest.TestCase):
    def assert_rejected_deterministically(self, callback, pattern):
        messages = []
        types = []
        for _ in range(2):
            with self.assertRaises(ValueError) as raised:
                callback()
            messages.append(str(raised.exception))
            types.append(type(raised.exception))
        self.assertEqual(types[0], types[1])
        self.assertEqual(messages[0], messages[1])
        self.assertRegex(messages[0], pattern)

    def test_rejects_missing_and_duplicate_sequence_ids(self):
        artifact, manifest = _two_clip_artifact_and_manifest()
        self.assert_rejected_deterministically(
            lambda: extract_slot_candidates(
                artifact, manifest, "pickup_table__missing__999"
            ),
            r"(?i)(missing|not found).*sequence|sequence.*(missing|not found)",
        )

        duplicate = copy.deepcopy(manifest)
        duplicate["clips"][1]["sequence_id"] = _TARGET_SEQUENCE
        self.assert_rejected_deterministically(
            lambda: extract_slot_candidates(
                artifact, duplicate, _TARGET_SEQUENCE
            ),
            r"(?i)duplicate.*sequence",
        )

    def test_rejects_a_manifest_range_that_does_not_match_the_artifact(self):
        artifact, manifest = _two_clip_artifact_and_manifest()
        mismatch = copy.deepcopy(manifest)
        mismatch["clips"][0]["range_start"] = 9

        self.assert_rejected_deterministically(
            lambda: extract_slot_candidates(
                artifact, mismatch, _TARGET_SEQUENCE
            ),
            r"(?i)range",
        )

    def test_rejects_nonmonotonic_phases(self):
        artifact, manifest = _two_clip_artifact_and_manifest()
        malformed = copy.deepcopy(artifact)
        malformed.phases[12:14] = np.array(
            [InteractionPhase.CONTACT, InteractionPhase.REACH], np.uint8
        )

        self.assert_rejected_deterministically(
            lambda: extract_slot_candidates(
                malformed, manifest, _TARGET_SEQUENCE
            ),
            r"(?i)(phase|monotonic)",
        )

    def test_rejects_each_missing_required_event(self):
        replacements = {
            InteractionPhase.REACH: InteractionPhase.APPROACH,
            InteractionPhase.CONTACT: InteractionPhase.LIFT,
            InteractionPhase.LIFT: InteractionPhase.CONTACT,
            InteractionPhase.HOLD: InteractionPhase.LIFT,
        }
        for missing, replacement in replacements.items():
            with self.subTest(missing=missing.name):
                artifact, manifest = _two_clip_artifact_and_manifest()
                malformed = copy.deepcopy(artifact)
                clip = malformed.phases[10:20]
                clip[clip == int(missing)] = int(replacement)

                self.assert_rejected_deterministically(
                    lambda: extract_slot_candidates(
                        malformed, manifest, _TARGET_SEQUENCE
                    ),
                    rf"(?i){missing.name}",
                )

    def test_pack_wrapper_rejects_a_target_object_in_the_heldout_split(self):
        artifact, manifest = _two_clip_artifact_and_manifest()
        split = {
            "seed": 20260714,
            "database_objects": [_SOURCE_OBJECT],
            "heldout_objects": [_TARGET_OBJECT],
        }
        loaded = (artifact, None, manifest, split, None)

        with patch(
            "resources.extract_g1_pick_slots.read_artifact_set",
            return_value=loaded,
        ):
            self.assert_rejected_deterministically(
                lambda: extract_slot_candidates_from_pack(
                    Path("/synthetic/interaction-pack"), _TARGET_SEQUENCE
                ),
                r"(?i)held.?out",
            )


if __name__ == "__main__":
    unittest.main()
