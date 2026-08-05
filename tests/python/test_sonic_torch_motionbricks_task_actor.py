import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_motionbricks_task_actor import (
    assemble_generated_route,
    contact_phase_support,
    endpoint_support_schedule,
    extract_proxy_keyframes,
    flight_support_schedule,
    infer_generated_support,
    stance_root_clearance_lift,
    stance_root_height_correction,
    stance_anchor_targets,
)


def _route(frame_count: int = 40) -> dict[str, np.ndarray]:
    roots = np.zeros((frame_count, 3), dtype=np.float64)
    roots[:, 0] = np.arange(frame_count, dtype=np.float64) * 0.01
    return {
        "joint_position": np.arange(
            frame_count * 29, dtype=np.float64
        ).reshape(frame_count, 29)
        * 0.001,
        "root_position_world": roots,
        "root_orientation_world_wxyz": np.tile(
            (1.0, 0.0, 0.0, 0.0), (frame_count, 1)
        ),
        "source_support_mask": np.tile(
            np.array((True, False), dtype=np.bool_),
            (frame_count, 1),
        ),
    }


class MotionBricksTaskActorTests(unittest.TestCase):
    def test_contact_phase_switches_support_instead_of_holding_start_foot(self):
        channels = np.array(
            (
                (0.8, 0.7, 0.1, 0.0),
                (0.7, 0.6, 0.2, 0.1),
                (0.3, 0.2, 0.6, 0.7),
                (0.1, 0.0, 0.8, 0.7),
            ),
            dtype=np.float64,
        )

        support = contact_phase_support(
            channels,
            initial_support=np.array((True, False)),
            terminal_support=np.array((False, True)),
            endpoint_window_frames=1,
        )

        np.testing.assert_array_equal(
            support,
            np.array(
                (
                    (True, False),
                    (True, False),
                    (False, True),
                    (False, True),
                )
            ),
        )

    def test_new_stance_anchors_at_generated_touchdown_not_start_position(self):
        feet = np.zeros((6, 2, 3), dtype=np.float64)
        feet[:, 0, 0] = 0.10
        feet[:, 1, 0] = np.array((0.30, 0.40, 0.50, 0.60, 0.60, 0.60))
        support = np.array(
            (
                (True, False),
                (True, False),
                (False, True),
                (False, True),
                (False, True),
                (False, True),
            ),
            dtype=np.bool_,
        )

        targets = stance_anchor_targets(
            foot_position_world=feet,
            support_mask=support,
            surface_height_m=np.zeros((6, 2), dtype=np.float64),
            ankle_origin_sole_m=0.05,
        )

        np.testing.assert_allclose(
            targets[:2, 0], np.tile((0.10, 0.0, 0.05), (2, 1))
        )
        np.testing.assert_allclose(
            targets[2:, 1], np.tile((0.50, 0.0, 0.05), (4, 1))
        )

    def test_stance_height_correction_lands_supported_ankle(self):
        actual = np.array(
            ((0.1, 0.2, 0.09), (0.3, 0.4, 0.20)),
            dtype=np.float64,
        )
        target = np.array(
            ((0.1, 0.2, 0.05), (0.3, 0.4, 0.40)),
            dtype=np.float64,
        )

        correction = stance_root_height_correction(
            actual_foot_position_world=actual,
            target_foot_position_world=target,
            support_mask=np.array((True, False)),
        )

        self.assertAlmostEqual(correction, -0.04)

    def test_stance_clearance_lift_uses_worst_supported_sole(self):
        lift = stance_root_clearance_lift(
            minimum_sole_clearance_m=np.array((-0.044, -0.20)),
            support_mask=np.array((True, False)),
            accepted_clearance_m=-0.02,
        )

        self.assertAlmostEqual(lift, 0.024)

    def test_extracts_ordered_four_frame_proxy_windows(self):
        route = _route()

        sequence = extract_proxy_keyframes(
            route, endpoint_frames=(11, 23, 35)
        )

        self.assertEqual(sequence.qpos.shape, (3, 4, 36))
        self.assertEqual(sequence.support_mask.shape, (3, 4, 2))
        self.assertEqual(sequence.endpoint_frames, (11, 23, 35))
        np.testing.assert_array_equal(
            sequence.qpos[:, -1, :3],
            route["root_position_world"][[11, 23, 35]],
        )
        np.testing.assert_array_equal(
            sequence.support_mask[:, -1],
            route["source_support_mask"][[11, 23, 35]],
        )

    def test_rejects_overlapping_or_airborne_proxy_endpoints(self):
        route = _route()
        with self.assertRaisesRegex(ContractError, "proxy"):
            extract_proxy_keyframes(route, endpoint_frames=(11, 13, 35))

        route["source_support_mask"][23] = False
        with self.assertRaisesRegex(ContractError, "proxy"):
            extract_proxy_keyframes(route, endpoint_frames=(11, 23, 35))

    def test_infers_support_from_clearance_speed_and_sole_points(self):
        support = infer_generated_support(
            foot_speed_mps=np.array(
                ((0.03, 0.30), (0.04, 0.02)), dtype=np.float64
            ),
            minimum_sole_clearance_m=np.array(
                ((0.01, 0.01), (0.08, 0.00)), dtype=np.float64
            ),
            supported_sole_points=np.array(
                ((4, 8), (8, 3)), dtype=np.int64
            ),
        )

        np.testing.assert_array_equal(
            support,
            np.array(((True, False), (False, True))),
        )

    def test_assembles_segments_without_repeating_context(self):
        context = extract_proxy_keyframes(
            _route(12), endpoint_frames=(7,)
        ).qpos[0]
        transition_a = np.concatenate(
            (context, np.repeat(context[-1:], 5, axis=0)), axis=0
        )
        transition_b = np.concatenate(
            (
                transition_a[-4:],
                np.repeat(transition_a[-1:], 3, axis=0),
            ),
            axis=0,
        )
        support_a = np.ones((len(transition_a), 2), dtype=np.bool_)
        support_b = np.ones((len(transition_b), 2), dtype=np.bool_)

        route = assemble_generated_route(
            context_qpos=context,
            context_support=np.ones((4, 2), dtype=np.bool_),
            transition_qpos=(transition_a, transition_b),
            transition_support=(support_a, support_b),
        )

        self.assertEqual(len(route["joint_position"]), 12)
        np.testing.assert_array_equal(
            route["segment_boundaries"], np.array((4, 9, 12))
        )

    def test_support_schedule_locks_start_then_exact_target_window(self):
        schedule = endpoint_support_schedule(
            initial_support=np.array((False, True)),
            target_support=np.array((True, False)),
            frame_count=12,
            target_window_frames=4,
        )

        np.testing.assert_array_equal(
            schedule[:8],
            np.tile((False, True), (8, 1)),
        )
        np.testing.assert_array_equal(
            schedule[8:],
            np.tile((True, False), (4, 1)),
        )

    def test_flight_schedule_has_explicit_takeoff_and_landing(self):
        schedule = flight_support_schedule(
            initial_support=np.array((True, False)),
            target_support=np.array((True, True)),
            frame_count=12,
            context_frames=4,
            target_window_frames=4,
        )

        np.testing.assert_array_equal(
            schedule[:4], np.tile((True, False), (4, 1))
        )
        self.assertFalse(schedule[4:8].any())
        np.testing.assert_array_equal(
            schedule[8:], np.tile((True, True), (4, 1))
        )


if __name__ == "__main__":
    unittest.main()
