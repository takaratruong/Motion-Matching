import unittest

import numpy as np

from resources.run_g1_torch_foothold_ablation import (
    arm_rank_key,
    selection_coherence,
)


class FootholdAblationComparisonTests(unittest.TestCase):
    def test_selection_coherence_counts_clip_and_frame_discontinuities(self):
        metrics = selection_coherence(
            selected_clip=np.array(("a", "a", "a", "b", "b", "b")),
            selected_frame=np.array((10, 11, 18, 3, 4, 5)),
            joint_velocity=np.array(
                (
                    (0.0, 0.0),
                    (0.1, 0.0),
                    (0.2, 0.0),
                    (2.0, 0.0),
                    (2.1, 0.0),
                    (2.2, 0.0),
                ),
                dtype=np.float64,
            ),
            dt_s=0.02,
        )

        self.assertEqual(metrics["source_discontinuity_count"], 2)
        self.assertEqual(metrics["cross_clip_transition_count"], 1)
        self.assertEqual(metrics["shortest_sequential_run_frames"], 1)
        self.assertAlmostEqual(
            metrics["transition_joint_velocity_jump_p95_rad_s"], 1.715
        )
        self.assertGreater(metrics["joint_jerk_p95_rad_s3"], 0.0)

    def test_arm_rank_prioritizes_safety_then_route_classes_then_coherence(self):
        unsafe = {
            "safety_violation_count": 1,
            "exception_count": 0,
            "passed_route_class_count": 7,
            "passed_route_count": 21,
            "stalled_moving_fraction": 0.0,
            "stance_slide_m": 0.0,
            "joint_jerk_p95_rad_s3": 0.0,
        }
        safe_rough = {
            "safety_violation_count": 0,
            "exception_count": 0,
            "passed_route_class_count": 5,
            "passed_route_count": 14,
            "stalled_moving_fraction": 0.1,
            "stance_slide_m": 4.0,
            "joint_jerk_p95_rad_s3": 5000.0,
        }
        safe_smooth = dict(safe_rough, joint_jerk_p95_rad_s3=1000.0)

        self.assertLess(arm_rank_key(safe_rough), arm_rank_key(unsafe))
        self.assertLess(arm_rank_key(safe_smooth), arm_rank_key(safe_rough))

    def test_arm_rank_treats_failed_closed_exception_as_coverage_not_safety(self):
        complete = {
            "safety_violation_count": 0,
            "exception_count": 0,
            "passed_route_class_count": 2,
            "passed_route_count": 10,
            "stalled_moving_fraction": 0.1,
            "stance_slide_m": 4.0,
            "joint_jerk_p95_rad_s3": 1000.0,
        }
        exhausted = dict(complete, exception_count=1)

        self.assertLess(arm_rank_key(complete), arm_rank_key(exhausted))


if __name__ == "__main__":
    unittest.main()
