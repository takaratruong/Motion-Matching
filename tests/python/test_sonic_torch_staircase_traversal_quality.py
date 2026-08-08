import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_staircase_traversal_quality import (
    admit_staircase_traversal,
    measure_cadence,
    measure_staircase_traversal_quality,
)


class StaircaseTraversalQualityTests(unittest.TestCase):
    @staticmethod
    def synthetic_metrics(**updates):
        metrics = {
            "terrain_events": (
                "ground",
                "elevated",
                "opposite_ground",
            ),
            "maximum_stance_horizontal_step_m": 0.005,
            "maximum_complete_sole_contact_error_m": 0.010,
            "minimum_lateral_foot_separation_m": 0.10,
            "cadence_violation_count": 0,
            "maximum_same_height_double_support_frames": 10,
            "minimum_supported_sole_points": 3,
            "terminal_complete_support": True,
        }
        metrics.update(updates)
        return metrics

    def test_rejects_motion_that_never_reaches_elevated_terrain(self):
        metrics = self.synthetic_metrics(
            terrain_events=("ground", "opposite_ground")
        )

        with self.assertRaisesRegex(
            ContractError, "missing elevated traversal"
        ):
            admit_staircase_traversal(metrics)

    def test_rejects_stance_slide_even_when_vertical_contact_is_valid(self):
        metrics = self.synthetic_metrics(
            maximum_stance_horizontal_step_m=0.018
        )

        with self.assertRaisesRegex(ContractError, "stance foot slides"):
            admit_staircase_traversal(metrics)

    def test_height_change_is_exempt_but_same_height_cadence_is_bounded(self):
        metrics = measure_cadence(
            touchdown_frames=(0, 30, 62, 92, 150),
            touchdown_heights_m=(0.0, 0.0, 0.0, 0.2, 0.0),
        )

        self.assertEqual(
            metrics["same_height_half_step_frames"], [30, 32]
        )
        self.assertEqual(metrics["cadence_violation_count"], 0)

    def test_rejects_crossed_feet_in_heading_local_frame(self):
        metrics = self.synthetic_metrics(
            minimum_lateral_foot_separation_m=-0.03
        )

        with self.assertRaisesRegex(ContractError, "feet cross"):
            admit_staircase_traversal(metrics)

    def test_rejects_excessive_same_height_double_support(self):
        metrics = self.synthetic_metrics(
            maximum_same_height_double_support_frames=26
        )

        with self.assertRaisesRegex(ContractError, "double support"):
            admit_staircase_traversal(metrics)

    def test_measurement_derives_ground_elevated_exit_from_touchdowns(self):
        frames = 10
        support = np.zeros((frames, 2), dtype=np.bool_)
        support[0:3, 0] = True
        support[2:6, 1] = True
        support[5:, 0] = True
        support[7:, 1] = True
        feet = np.zeros((frames, 2, 3), dtype=np.float64)
        feet[:, 0, 1] = 0.10
        feet[:, 1, 1] = -0.10
        feet[2:7, 1, 2] = 0.20
        feet[5:, 0, 2] = 0.20

        # Supply terrain explicitly by scene X so touchdown 1 and 2 are high.
        feet[2:7, 1, 0] = 1.0
        feet[5:, 0, 0] = 1.0
        soles = np.repeat(feet[:, :, None, :], 3, axis=2)

        def stepped_surface(points):
            points = np.asarray(points)
            return np.where(points[..., 0] > 0.5, 0.20, 0.0)

        metrics = measure_staircase_traversal_quality(
            feet_scene_xyz=feet,
            sole_points_scene_xyz=soles,
            support_mask=support,
            heading_scene_xy=(1.0, 0.0),
            sample_surface=stepped_surface,
        )

        self.assertEqual(
            tuple(metrics["terrain_events"]),
            ("ground", "elevated", "elevated", "opposite_ground"),
        )
        self.assertTrue(metrics["terminal_complete_support"])


if __name__ == "__main__":
    unittest.main()
