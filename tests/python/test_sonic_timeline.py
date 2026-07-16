from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

import numpy as np

from mm_sonic.joints import ContractError, load_joint_contract
from mm_sonic.timeline import (
    CanonicalTargetBuffer,
    PreparedTargetCandidate,
    TargetChunk,
    TargetTimeline,
)
from mm_sonic.transform import (
    holden_to_mujoco_quaternions,
    map_source_joints,
)
from tests.python.test_sonic_resample import (
    JOINT_CONTRACT,
    axis_angle,
    initial_from_chunk,
    make_source_chunk,
    readonly,
)


def accepted_snapshot(timeline: TargetTimeline):
    buffer = timeline.canonical_buffer
    return (
        timeline.last_frame_index,
        timeline.last_accepted_candidate_id,
        buffer.joint_position.tobytes(),
        buffer.joint_velocity.tobytes(),
        buffer.body_quat_w.tobytes(),
        buffer.frame_index.tobytes(),
    )


class InitialTimelineTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_joint_contract(JOINT_CONTRACT)
        self.chunk = make_source_chunk(self.contract)
        self.initial = initial_from_chunk(self.chunk)

    def test_reset_boundary_owns_exactly_one_immutable_frame_zero(self):
        timeline = TargetTimeline(self.initial, self.contract)
        initial = timeline.initial_buffer
        self.assertIsInstance(initial, CanonicalTargetBuffer)
        self.assertEqual(initial.count, 1)
        np.testing.assert_array_equal(initial.frame_index, np.array([0], np.int64))
        expected_position = map_source_joints(
            self.initial.joint_position_source,
            self.initial.source_joint_names,
            self.contract,
        )
        expected_velocity = map_source_joints(
            self.initial.joint_velocity_source,
            self.initial.source_joint_names,
            self.contract,
        )
        expected_quaternion = holden_to_mujoco_quaternions(
            self.initial.physical_pelvis_orientation_holden
        )
        np.testing.assert_array_equal(
            initial.joint_position[0].view(np.uint32),
            expected_position.view(np.uint32),
        )
        np.testing.assert_array_equal(
            initial.joint_velocity[0].view(np.uint32),
            expected_velocity.view(np.uint32),
        )
        np.testing.assert_array_equal(
            initial.body_quat_w[0].view(np.uint32),
            expected_quaternion.view(np.uint32),
        )
        for field in ("joint_position", "joint_velocity", "body_quat_w", "frame_index"):
            value = getattr(initial, field)
            self.assertTrue(value.flags.owndata, field)
            self.assertTrue(value.flags.c_contiguous, field)
            self.assertFalse(value.flags.writeable, field)
        self.assertEqual(timeline.last_frame_index, 0)
        self.assertIsNone(timeline.last_accepted_candidate_id)
        self.assertIsNone(timeline.pending_candidate_id)

    def test_initial_boundary_outside_a_registered_joint_limit_is_rejected(self):
        row = self.contract.rows[0]
        position = self.initial.joint_position_source.copy()
        position[row.source_index] = np.float32(row.upper + 0.01)
        invalid = replace(self.initial, joint_position_source=readonly(position))
        with self.assertRaisesRegex(ContractError, "joint limit"):
            TargetTimeline(invalid, self.contract)


class TransactionalTimelineTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_joint_contract(JOINT_CONTRACT)
        self.first = make_source_chunk(self.contract)
        self.initial = initial_from_chunk(self.first)
        self.timeline = TargetTimeline(self.initial, self.contract)

    def test_prepare_assigns_one_through_twenty_without_mutating_accepted_state(self):
        before = accepted_snapshot(self.timeline)
        prepared = self.timeline.prepare(self.first)
        self.assertIsInstance(prepared, PreparedTargetCandidate)
        self.assertIsInstance(prepared.target, TargetChunk)
        np.testing.assert_array_equal(
            prepared.target.frame_index, np.arange(1, 21, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            prepared.target.timestamps_s,
            prepared.target.frame_index.astype(np.float64) / 50.0,
        )
        self.assertEqual(prepared.target.schema, "target-chunk/v1")
        self.assertEqual(prepared.target.session_id, "session-0")
        self.assertEqual(prepared.target.source_candidate_id, "candidate-1")
        self.assertTrue(prepared.target.accepted_chunk_id)
        self.assertEqual(accepted_snapshot(self.timeline), before)
        self.assertEqual(self.timeline.pending_candidate_id, "candidate-1")
        for field in (
            "frame_index",
            "timestamps_s",
            "joint_position",
            "joint_velocity",
            "body_quat_w",
            "physical_pelvis_position",
            "virtual_root_position",
            "virtual_root_quat_w",
        ):
            value = getattr(prepared.target, field)
            self.assertTrue(value.flags.owndata, field)
            self.assertTrue(value.flags.c_contiguous, field)
            self.assertFalse(value.flags.writeable, field)

    def test_commit_advances_once_and_the_second_chunk_owns_twenty_one_to_forty(self):
        prepared_first = self.timeline.prepare(self.first)
        first_target = self.timeline.commit(prepared_first)
        self.assertIs(first_target, prepared_first.target)
        self.assertEqual(self.timeline.last_frame_index, 20)
        self.assertEqual(self.timeline.last_accepted_candidate_id, "candidate-1")
        self.assertIsNone(self.timeline.pending_candidate_id)
        second = make_source_chunk(
            self.contract,
            candidate_id="candidate-2",
            predecessor_id="candidate-1",
            start_s=0.4,
        )
        prepared_second = self.timeline.prepare(second)
        np.testing.assert_array_equal(
            prepared_second.target.frame_index, np.arange(21, 41, dtype=np.int64)
        )
        second_target = self.timeline.commit(prepared_second)
        self.assertIs(second_target, prepared_second.target)
        canonical = self.timeline.canonical_buffer
        self.assertEqual(canonical.count, 41)
        np.testing.assert_array_equal(
            canonical.frame_index, np.arange(41, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            canonical.joint_position[1:21], first_target.joint_position
        )
        np.testing.assert_array_equal(
            canonical.joint_position[21:41], second_target.joint_position
        )

    def test_prepare_abort_and_regenerate_are_bit_identical(self):
        before = accepted_snapshot(self.timeline)
        first_prepared = self.timeline.prepare(self.first)
        self.timeline.abort("candidate-1")
        self.assertEqual(accepted_snapshot(self.timeline), before)
        self.assertIsNone(self.timeline.pending_candidate_id)
        regenerated = self.timeline.prepare(self.first)
        self.assertEqual(
            regenerated.target.accepted_chunk_id,
            first_prepared.target.accepted_chunk_id,
        )
        self.assertEqual(regenerated.target.hashes, first_prepared.target.hashes)
        for field in (
            "frame_index",
            "timestamps_s",
            "joint_position",
            "joint_velocity",
            "body_quat_w",
            "physical_pelvis_position",
            "virtual_root_position",
            "virtual_root_quat_w",
        ):
            np.testing.assert_array_equal(
                getattr(regenerated.target, field).view(np.uint8),
                getattr(first_prepared.target, field).view(np.uint8),
            )
        with self.assertRaisesRegex(ContractError, "prepared candidate"):
            self.timeline.commit(first_prepared)
        self.assertEqual(self.timeline.pending_candidate_id, "candidate-1")
        self.timeline.commit(regenerated)

    def test_wrong_candidate_commit_and_abort_leave_all_state_unchanged(self):
        prepared = self.timeline.prepare(self.first)
        accepted_before = accepted_snapshot(self.timeline)
        other_chunk = replace(self.first, candidate_id="candidate-other")
        other_timeline = TargetTimeline(self.initial, self.contract)
        other_prepared = other_timeline.prepare(other_chunk)
        with self.assertRaisesRegex(ContractError, "candidate"):
            self.timeline.commit(other_prepared)
        self.assertEqual(accepted_snapshot(self.timeline), accepted_before)
        self.assertEqual(self.timeline.pending_candidate_id, "candidate-1")
        with self.assertRaisesRegex(ContractError, "candidate"):
            self.timeline.abort("candidate-other")
        self.assertEqual(accepted_snapshot(self.timeline), accepted_before)
        self.assertEqual(self.timeline.pending_candidate_id, "candidate-1")
        self.assertIs(self.timeline.commit(prepared), prepared.target)

    def test_wrong_predecessor_session_and_duplicate_prepare_do_not_mutate_state(self):
        cases = (
            replace(self.first, predecessor_id="not-the-reset"),
            replace(self.first, session_id="different-session"),
        )
        for chunk in cases:
            before = accepted_snapshot(self.timeline)
            with self.subTest(chunk=chunk), self.assertRaises(ContractError):
                self.timeline.prepare(chunk)
            self.assertEqual(accepted_snapshot(self.timeline), before)
            self.assertIsNone(self.timeline.pending_candidate_id)

        prepared = self.timeline.prepare(self.first)
        before = accepted_snapshot(self.timeline)
        with self.assertRaisesRegex(ContractError, "pending"):
            self.timeline.prepare(replace(self.first, candidate_id="candidate-2"))
        self.assertEqual(accepted_snapshot(self.timeline), before)
        self.assertEqual(self.timeline.pending_candidate_id, "candidate-1")
        self.timeline.abort(prepared.source_candidate_id)

    def test_scalar_seam_at_tolerance_is_accepted_and_above_is_rejected(self):
        for difference in (np.float32(5.0e-7), np.float32(1.0e-6)):
            position = self.first.joint_position_source.copy()
            position[0, 0] += difference
            candidate = replace(
                self.first,
                candidate_id=f"candidate-below-{float(difference)}",
                joint_position_source=readonly(position),
            )
            prepared = self.timeline.prepare(candidate)
            self.timeline.abort(prepared.source_candidate_id)

        position = self.first.joint_position_source.copy()
        position[0, 0] += np.float32(1.1e-6)
        mismatch = replace(
            self.first,
            candidate_id="candidate-above",
            joint_position_source=readonly(position),
        )
        before = accepted_snapshot(self.timeline)
        with self.assertRaisesRegex(ContractError, "seam"):
            self.timeline.prepare(mismatch)
        self.assertEqual(accepted_snapshot(self.timeline), before)
        self.assertIsNone(self.timeline.pending_candidate_id)

    def test_orientation_seam_uses_one_microradian_geodesic_tolerance(self):
        for angle in (0.5e-6, 1.0e-6):
            orientation = self.first.physical_pelvis_orientation_holden.copy()
            orientation[0] = axis_angle([1.0, 0.0, 0.0], angle)
            candidate = replace(
                self.first,
                candidate_id=f"candidate-angle-{angle}",
                physical_pelvis_orientation_holden=readonly(orientation),
            )
            prepared = self.timeline.prepare(candidate)
            self.timeline.abort(prepared.source_candidate_id)

        orientation = self.first.physical_pelvis_orientation_holden.copy()
        orientation[0] = axis_angle([1.0, 0.0, 0.0], 1.1e-6)
        mismatch = replace(
            self.first,
            candidate_id="candidate-angle-above",
            physical_pelvis_orientation_holden=readonly(orientation),
        )
        before = accepted_snapshot(self.timeline)
        with self.assertRaisesRegex(ContractError, "orientation seam"):
            self.timeline.prepare(mismatch)
        self.assertEqual(accepted_snapshot(self.timeline), before)
        self.assertIsNone(self.timeline.pending_candidate_id)

    def test_antipodal_seam_is_unrolled_against_the_accepted_timeline(self):
        physical = -self.first.physical_pelvis_orientation_holden
        virtual = -self.first.virtual_root_orientation_holden
        antipodal = replace(
            self.first,
            physical_pelvis_orientation_holden=readonly(physical),
            virtual_root_orientation_holden=readonly(virtual),
        )
        target = self.timeline.prepare(antipodal).target
        full_physical = np.concatenate(
            (self.timeline.initial_buffer.body_quat_w, target.body_quat_w), axis=0
        )
        physical_dots = np.sum(
            full_physical[:-1].astype(np.float64)
            * full_physical[1:].astype(np.float64),
            axis=1,
        )
        virtual_dots = np.sum(
            target.virtual_root_quat_w[:-1].astype(np.float64)
            * target.virtual_root_quat_w[1:].astype(np.float64),
            axis=1,
        )
        self.assertTrue(np.all(physical_dots >= 0.0), physical_dots)
        self.assertTrue(np.all(virtual_dots >= 0.0), virtual_dots)

    def test_overshoot_prepare_failure_cannot_publish_or_mutate_timeline(self):
        row = next(row for row in self.contract.rows if row.target_index == 17)
        position = self.first.joint_position_source.copy()
        velocity = self.first.joint_velocity_source.copy()
        position[:2, row.source_index] = 0.0
        velocity[0, row.source_index] = 30.0
        velocity[1, row.source_index] = -30.0
        overshooting = replace(
            self.first,
            joint_position_source=readonly(position),
            joint_velocity_source=readonly(velocity),
        )
        before = accepted_snapshot(self.timeline)
        with self.assertRaisesRegex(ContractError, "joint limit"):
            self.timeline.prepare(overshooting)
        self.assertEqual(accepted_snapshot(self.timeline), before)
        self.assertIsNone(self.timeline.pending_candidate_id)


class TargetHashTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_joint_contract(JOINT_CONTRACT)
        self.chunk = make_source_chunk(self.contract)

    def test_canonical_hash_uses_exact_ordered_little_endian_policy_bytes(self):
        timeline = TargetTimeline(initial_from_chunk(self.chunk), self.contract)
        target = timeline.prepare(self.chunk).target
        payload = b"".join(
            (
                np.ascontiguousarray(target.joint_position, dtype="<f4").tobytes(),
                np.ascontiguousarray(target.joint_velocity, dtype="<f4").tobytes(),
                np.ascontiguousarray(target.body_quat_w, dtype="<f4").tobytes(),
                np.ascontiguousarray(target.frame_index, dtype="<i8").tobytes(),
            )
        )
        self.assertEqual(
            target.hashes["canonical_target_sha256"],
            hashlib.sha256(payload).hexdigest(),
        )

    def test_diagnostic_positions_change_only_the_diagnostic_hash(self):
        baseline_timeline = TargetTimeline(
            initial_from_chunk(self.chunk), self.contract
        )
        baseline = baseline_timeline.prepare(self.chunk).target

        physical = self.chunk.physical_pelvis_position_holden.copy()
        virtual = self.chunk.virtual_root_position_holden.copy()
        physical += np.array([10.0, -4.0, 2.0], np.float32)
        virtual += np.array([-3.0, 5.0, 7.0], np.float32)
        diagnostic_only = replace(
            self.chunk,
            physical_pelvis_position_holden=readonly(physical),
            virtual_root_position_holden=readonly(virtual),
        )
        diagnostic_timeline = TargetTimeline(
            initial_from_chunk(diagnostic_only), self.contract
        )
        changed = diagnostic_timeline.prepare(diagnostic_only).target
        self.assertEqual(
            baseline.hashes["canonical_target_sha256"],
            changed.hashes["canonical_target_sha256"],
        )
        self.assertNotEqual(
            baseline.hashes["diagnostic_sha256"],
            changed.hashes["diagnostic_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
