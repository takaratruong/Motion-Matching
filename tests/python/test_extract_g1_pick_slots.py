import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from resources.extract_g1_pick_slots import (
    _root_intersects_table,
    _segment_intersects_expanded_box,
    extract_slot_candidates,
    extract_slot_candidates_from_pack,
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


def _assert_gate_counts(test_case, report, expected_gate=None):
    counts = report["rejected_counts_by_gate"]
    test_case.assertEqual(list(counts), list(_REJECTED_GATE_ORDER))
    test_case.assertEqual(
        counts,
        {
            gate: int(gate == expected_gate)
            for gate in _REJECTED_GATE_ORDER
        },
    )
    test_case.assertEqual(report["deduplicated_candidate_count"], 0)


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
        artifact.positions[3:_CLIP_FRAMES, _RIGHT_WRIST] = (
            source_hand - artifact.positions[3, 0]
        )

        quarter_turn = np.float32(np.pi / 2.0)
        target_hand = np.array([4.3, 1.2, 2.9], np.float32)
        artifact.object_rotations[target] = _yaw_quaternion(quarter_turn)
        artifact.positions[13:, _RIGHT_WRIST] = (
            target_hand - artifact.positions[13, 0]
        )
        artifact.rotations[13:, _RIGHT_WRIST] = _yaw_quaternion(quarter_turn)

        report = extract_slot_candidates(
            artifact, manifest, _TARGET_SEQUENCE
        )
        candidate = _candidate(report, _SOURCE_SEQUENCE)

        self.assertEqual(
            {field: candidate[field] for field in _SLOT_CANDIDATE_FIELDS},
            {
                "root_x_object_m": 1.0,
                "root_z_object_m": 2.0,
                "root_yaw_object_radians": float(
                    np.float32(np.deg2rad(30.0))
                ),
                "contact_hand_position_error_m": 0.0,
                "contact_hand_orientation_error_radians": 0.0,
                "entry_root_planar_speed_mps": 5.0,
                "root_table_clear": True,
                "hand_table_clear": True,
                "hand_object_clear": True,
                "static_path_feasible": True,
            },
        )
        _assert_gate_counts(self, report)

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

    def test_contact_orientation_gate_accepts_equality_and_rejects_next_float(self):
        cases = (
            (_HAND_ORIENTATION_LIMIT_RADIANS, True),
            (
                np.nextafter(
                    _HAND_ORIENTATION_LIMIT_RADIANS,
                    np.float32(np.inf),
                ),
                False,
            ),
        )
        for error_radians, accepted in cases:
            with self.subTest(error_radians=float(error_radians)):
                artifact, manifest = _two_clip_artifact_and_manifest()
                artifact.rotations[3:_CLIP_FRAMES, _RIGHT_WRIST] = (
                    _yaw_quaternion(error_radians)
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
                        float(error_radians),
                        places=7,
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
