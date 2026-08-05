import math
import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_horizontal_contact_overlay import (
    HorizontalContactPlan,
    horizontal_contact_plan_variants,
)


def representative_stair_height(points_xy):
    points = np.asarray(points_xy, dtype=np.float64)
    x = points[..., 0]
    y = points[..., 1]
    inside = (x >= -0.44) & (x <= 0.44)
    height = np.zeros_like(x)
    height[inside & (y >= 0.21) & (y < 0.49)] = 0.356
    height[inside & (y >= -0.05) & (y < 0.21)] = 0.530
    return height


class HorizontalContactOverlayTests(unittest.TestCase):
    def test_horizontal_route_is_scene_x_and_contains_complete_phases(self):
        plans = horizontal_contact_plan_variants(
            sample_height=representative_stair_height,
            route_start_scene_x=-0.78,
            route_stop_scene_x=0.78,
            foot_lane_scene_y=(0.32, 0.12),
            nominal_step_length_m=0.18,
            nominal_step_frames=18,
            nominal_double_support_frames=4,
        )

        self.assertGreaterEqual(len(plans), 4)
        plan = plans[0]
        self.assertIsInstance(plan, HorizontalContactPlan)
        self.assertEqual(plan.direction_scene_xy, (1.0, 0.0))
        self.assertAlmostEqual(plan.heading_scene_yaw_rad, 0.0)
        self.assertEqual(
            tuple(phase.role for phase in plan.phases),
            ("approach", "entry", "uneven_walk", "exit", "departure"),
        )
        self.assertGreaterEqual(
            sum(
                interval.role == "uneven_walk"
                for interval in plan.intervals
            ),
            2,
        )
        self.assertTrue(
            any(
                abs(
                    left.sole_center_scene_xyz[2]
                    - right.sole_center_scene_xyz[2]
                )
                >= 0.15
                for left, right in plan.overlapping_intervals()
                if left.role == right.role == "uneven_walk"
            )
        )

    def test_every_stance_sole_lands_on_one_surface(self):
        plan = horizontal_contact_plan_variants(
            sample_height=representative_stair_height,
            route_start_scene_x=-0.78,
            route_stop_scene_x=0.78,
            foot_lane_scene_y=(0.32, 0.12),
            nominal_step_length_m=0.18,
            nominal_step_frames=18,
            nominal_double_support_frames=4,
        )[0]

        for interval in plan.intervals:
            corners = interval.sole_corners_scene_xy()
            heights = representative_stair_height(corners)
            self.assertLessEqual(float(np.ptp(heights)), 0.025)
            self.assertAlmostEqual(
                interval.sole_center_scene_xyz[2],
                float(np.median(heights)),
                places=7,
            )

    def test_variants_are_deterministic_and_cover_both_leading_feet(self):
        arguments = dict(
            sample_height=representative_stair_height,
            route_start_scene_x=-0.78,
            route_stop_scene_x=0.78,
            foot_lane_scene_y=(0.32, 0.12),
            nominal_step_length_m=0.18,
            nominal_step_frames=18,
            nominal_double_support_frames=4,
        )

        first = horizontal_contact_plan_variants(**arguments)
        second = horizontal_contact_plan_variants(**arguments)

        self.assertEqual(first, second)
        self.assertEqual({plan.leading_foot for plan in first}, {0, 1})
        self.assertGreater(len({plan.plan_id for plan in first}), 1)

    def test_rejects_a_lane_whose_sole_straddles_a_riser(self):
        with self.assertRaisesRegex(ContractError, "sole straddles"):
            horizontal_contact_plan_variants(
                sample_height=representative_stair_height,
                route_start_scene_x=-0.78,
                route_stop_scene_x=0.78,
                foot_lane_scene_y=(0.205, 0.12),
                nominal_step_length_m=0.18,
                nominal_step_frames=18,
                nominal_double_support_frames=4,
            )

    def test_rejects_a_non_horizontal_heading(self):
        with self.assertRaisesRegex(ContractError, "heading"):
            horizontal_contact_plan_variants(
                sample_height=representative_stair_height,
                route_start_scene_x=-0.78,
                route_stop_scene_x=0.78,
                foot_lane_scene_y=(0.32, 0.12),
                nominal_step_length_m=0.18,
                nominal_step_frames=18,
                nominal_double_support_frames=4,
                heading_scene_yaw_rad=math.pi / 2.0,
            )


if __name__ == "__main__":
    unittest.main()
