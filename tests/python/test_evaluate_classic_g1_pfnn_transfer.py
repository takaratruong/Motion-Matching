from __future__ import annotations

import unittest

import numpy as np

from mm_sonic.evaluate_classic_g1_pfnn_transfer import (
    joint_reconstruction_metrics,
    released_pfnn_gate_failures,
    source_joint_reconstruction_metrics,
)


class ClassicG1PFNNTransferEvaluationTests(unittest.TestCase):
    def test_joint_metrics_report_exact_error_and_worst_frame_joint(self) -> None:
        target = np.zeros((4, 29), dtype=np.float32)
        predicted = target.copy()
        predicted[2, 3] = np.float32(0.2)

        metrics = joint_reconstruction_metrics(predicted, target)

        self.assertEqual(metrics["sample_count"], 4)
        self.assertAlmostEqual(metrics["joint_mae_rad"], 0.2 / (4 * 29))
        self.assertAlmostEqual(metrics["joint_rmse_rad"], 0.2 / np.sqrt(4 * 29))
        self.assertAlmostEqual(metrics["frame_max_p95_rad"], 0.17)
        self.assertAlmostEqual(metrics["maximum_joint_error_rad"], 0.2)
        self.assertEqual(metrics["worst_frame_index"], 2)
        self.assertEqual(metrics["worst_joint_index"], 3)

    def test_released_gate_cannot_be_hidden_by_good_grail_rows(self) -> None:
        target = np.zeros((21, 29), dtype=np.float32)
        predicted = target.copy()
        predicted[0] = np.float32(0.2)
        clips = np.asarray(
            ("released_pfnn_walk",)
            + tuple(f"terrain_slopes__slope_000__{index:03d}" for index in range(20))
        )

        metrics = source_joint_reconstruction_metrics(predicted, target, clips)
        aggregate = joint_reconstruction_metrics(predicted, target)

        self.assertLess(aggregate["joint_mae_rad"], 0.1)
        self.assertEqual(metrics["released_pfnn"]["sample_count"], 1)
        self.assertAlmostEqual(metrics["released_pfnn"]["joint_mae_rad"], 0.2)
        self.assertEqual(
            released_pfnn_gate_failures(metrics["released_pfnn"]),
            ("joint_mae_rad", "joint_rmse_rad"),
        )
        self.assertEqual(released_pfnn_gate_failures(metrics["grail"]), ())


if __name__ == "__main__":
    unittest.main()
