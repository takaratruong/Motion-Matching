import unittest

import numpy as np

from mm_sonic.torch_terrain_contact_quality import (
    ANKLE_ORIGIN_SOLE_M,
    evaluate_contact_quality,
)


def _inputs(frame_count: int):
    feet = np.zeros((frame_count, 2, 3), dtype=np.float64)
    feet[:, :, 2] = ANKLE_ORIGIN_SOLE_M
    return {
        "foot_position_world": feet,
        "foot_surface_height_m": np.zeros((frame_count, 2), dtype=np.float64),
        "source_support_mask": np.ones((frame_count, 2), dtype=bool),
        "root_position_world": np.zeros((frame_count, 3), dtype=np.float64),
        "command_velocity_world_xy": np.zeros((frame_count, 2), dtype=np.float64),
        "transition_start_mask": np.zeros(frame_count, dtype=bool),
    }


class TerrainContactQualityTests(unittest.TestCase):
    def test_counts_unload_touchdown_and_complete_steps_from_source_support(self):
        values = _inputs(8)
        values["source_support_mask"][:, 0] = [True, True, False, False,
                                                     True, True, True, True]
        values["source_support_mask"][:, 1] = [True, True, True, True,
                                                     True, False, False, True]

        metrics = evaluate_contact_quality(**values)

        self.assertEqual(metrics.unload_count, (1, 1))
        self.assertEqual(metrics.touchdown_count, (1, 1))
        self.assertEqual(metrics.complete_step_count, 2)

    def test_source_supported_floating_is_not_removed_from_denominator(self):
        values = _inputs(6)
        values["foot_position_world"][2:4, 0, 2] = 0.20

        metrics = evaluate_contact_quality(**values)

        np.testing.assert_array_equal(
            metrics.expected_stance_floating_mask[:, 0],
            [False, False, True, True, True, False],
        )
        self.assertAlmostEqual(metrics.expected_stance_floating_fraction[0], 0.5)
        self.assertAlmostEqual(metrics.contact_agreement_fraction, 0.75)

    def test_reports_longest_consecutive_emitted_no_contact_run(self):
        values = _inputs(16)
        values["foot_position_world"][2:13, :, 2] = 0.20

        metrics = evaluate_contact_quality(**values)

        self.assertEqual(metrics.maximum_no_contact_frames, 12)

    def test_moving_without_source_unload_has_zero_step_rate(self):
        values = _inputs(10)
        values["command_velocity_world_xy"][:, 0] = 0.4
        values["root_position_world"][:, 0] = np.linspace(0.0, 0.5, 10)

        metrics = evaluate_contact_quality(**values)

        self.assertEqual(metrics.unload_count, (0, 0))
        self.assertEqual(metrics.complete_step_count, 0)
        self.assertEqual(metrics.complete_steps_per_m, 0.0)
        self.assertEqual(metrics.command_to_unload_frames, ())

    def test_source_stance_drift_is_split_by_transition_neighborhood(self):
        values = _inputs(24)
        values["foot_position_world"][:, 0, 0] = np.arange(24) * 0.01
        values["transition_start_mask"][3] = True

        metrics = evaluate_contact_quality(**values)

        self.assertAlmostEqual(metrics.source_stance_drift_m[0], 0.23)
        self.assertAlmostEqual(metrics.transition_source_stance_drift_m, 0.15)
        self.assertAlmostEqual(metrics.steady_source_stance_drift_m, 0.08)

    def test_command_onset_delays_use_next_source_contact_events(self):
        values = _inputs(8)
        values["command_velocity_world_xy"][1:, 0] = 0.4
        values["source_support_mask"][:, 0] = [True, True, True, False,
                                                     False, True, True, True]

        metrics = evaluate_contact_quality(**values)

        self.assertEqual(metrics.command_to_unload_frames, (2,))
        self.assertEqual(metrics.command_to_touchdown_frames, (4,))

    def test_rejects_misaligned_or_nonfinite_inputs(self):
        values = _inputs(3)
        values["root_position_world"] = np.zeros((2, 3), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "root_position_world"):
            evaluate_contact_quality(**values)

        values = _inputs(3)
        values["foot_position_world"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "foot_position_world"):
            evaluate_contact_quality(**values)


if __name__ == "__main__":
    unittest.main()
