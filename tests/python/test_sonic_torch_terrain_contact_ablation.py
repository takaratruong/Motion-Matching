import unittest

from mm_sonic.torch_terrain_contact_ablation import (
    contact_ablation_passes,
    selected_ranked_action_indices,
)


class TerrainContactAblationTests(unittest.TestCase):
    def test_selects_unique_actions_in_combined_then_landing_order(self):
        payload = {
            "ranking": {
                "best_combined": [
                    {"action_index": 7},
                    {"action_index": 3},
                ],
                "best_landing": [
                    {"action_index": 3},
                    {"action_index": 9},
                ],
                "best_continuity": [{"action_index": 4}],
                "best_two_step": [{"action_index": 5}],
            }
        }

        self.assertEqual(
            selected_ranked_action_indices(payload, maximum_count=4),
            (7, 3, 9, 4),
        )

    def test_contact_acceptance_requires_drift_landing_and_small_residual(self):
        passing = {
            "placed_stance_drift_m": 0.012,
            "projected_stance_drift_m": 0.018,
            "projected_landing_error_m": 0.003,
            "maximum_target_error_m": 0.004,
        }
        self.assertTrue(contact_ablation_passes(passing))
        self.assertFalse(
            contact_ablation_passes(
                {**passing, "projected_stance_drift_m": 0.031}
            )
        )
        self.assertFalse(
            contact_ablation_passes(
                {**passing, "projected_stance_drift_m": 0.023}
            )
        )
        self.assertFalse(
            contact_ablation_passes(
                {**passing, "maximum_target_error_m": 0.006}
            )
        )


if __name__ == "__main__":
    unittest.main()
