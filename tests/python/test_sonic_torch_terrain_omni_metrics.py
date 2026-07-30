import unittest

import numpy as np

from mm_sonic.torch_terrain_omni_metrics import (
    ANKLE_ORIGIN_SOLE_M,
    RescueEvent,
    detect_rescue_cycles,
    evaluate_omni_route,
)


class OmnidirectionalTerrainMetricTests(unittest.TestCase):
    def test_each_foot_is_compared_with_its_own_tread_height(self):
        surface = np.array([[0.1778, 0.3556]], dtype=np.float64)
        feet = np.zeros((1, 2, 3), dtype=np.float64)
        feet[0, :, 2] = surface[0] + ANKLE_ORIGIN_SOLE_M

        metrics = evaluate_omni_route(
            root_xy=np.zeros((1, 2)),
            root_yaw=np.zeros(1),
            foot_position_world=feet,
            foot_surface_height_m=surface,
            command_velocity_world_xy=np.zeros((1, 2)),
            command_heading_world_yaw=np.zeros(1),
        )

        np.testing.assert_array_equal(metrics.stance_mask, [[True, True]])
        self.assertAlmostEqual(metrics.support_height_difference_m.maximum, 0.1778)
        self.assertAlmostEqual(metrics.support_height_error_m.maximum, 0.0)

    def test_sliding_counts_stance_motion_and_excludes_swing_motion(self):
        frame_count = 22
        feet = np.zeros((frame_count, 2, 3), dtype=np.float64)
        surface = np.zeros((frame_count, 2), dtype=np.float64)
        feet[:20, :, 2] = ANKLE_ORIGIN_SOLE_M
        feet[:20, :, 0] = np.linspace(0.0, 0.04, 20)[:, None]
        feet[20:, :, 2] = 0.20
        feet[20:, :, 0] = np.array([0.19, 0.34])[:, None]

        metrics = evaluate_omni_route(
            root_xy=np.zeros((frame_count, 2)),
            root_yaw=np.zeros(frame_count),
            foot_position_world=feet,
            foot_surface_height_m=surface,
            command_velocity_world_xy=np.zeros((frame_count, 2)),
            command_heading_world_yaw=np.zeros(frame_count),
        )

        np.testing.assert_allclose(metrics.stance_slide_m.per_foot, [0.04, 0.04])
        self.assertFalse(metrics.stance_mask[20:].any())

    def test_rescue_cycle_requires_alternation_neighborhood_and_low_progress(self):
        events = (
            RescueEvent("stair/0002", 321, (0.00, 0.00)),
            RescueEvent("stair/0000", 336, (0.01, 0.00)),
            RescueEvent("stair/0002", 323, (0.02, 0.00)),
            RescueEvent("stair/0000", 338, (0.03, 0.00)),
        )
        cycles = detect_rescue_cycles(events)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(cycles[0].event_indices, (0, 1, 2, 3))

        high_progress = events[:-1] + (
            RescueEvent("stair/0000", 338, (0.08, 0.00)),
        )
        self.assertEqual(detect_rescue_cycles(high_progress), ())

        non_rescue = events[:2] + (
            RescueEvent(
                "stair/0002",
                323,
                (0.02, 0.00),
                is_terrain_rescue=False,
            ),
        ) + events[3:]
        self.assertEqual(detect_rescue_cycles(non_rescue), ())


if __name__ == "__main__":
    unittest.main()
