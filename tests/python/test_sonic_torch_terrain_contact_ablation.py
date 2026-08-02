import unittest
from dataclasses import replace

import numpy as np

from mm_sonic.torch_terrain_contact_ablation import (
    contact_anchored_prefilter,
    contact_ablation_passes,
    selected_ranked_action_indices,
)
from mm_sonic.torch_contact_oracle_actions import (
    ContactPhaseActionIndex,
    ContactPhaseInventory,
)
from mm_sonic.torch_contact_oracle_search import OracleConstraints
from mm_sonic.torch_terrain_action_quality import build_native_quality_index
from mm_sonic.torch_terrain_quality_oracle import build_quality_action_cache
from tests.python.test_sonic_torch_terrain_contact_quality import _action, _state


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
            "maximum_root_correction_m": 0.05,
            "maximum_root_correction_speed_m_s": 0.5,
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
        self.assertFalse(
            contact_ablation_passes(
                {**passing, "maximum_root_correction_m": 0.101}
            )
        )
        self.assertFalse(
            contact_ablation_passes(
                {**passing, "maximum_root_correction_speed_m_s": 1.01}
            )
        )

    def test_contact_prefilter_ranks_after_support_foot_anchoring(self):
        first = _action()
        farther_feet = first.foot_position_local.clone()
        farther_feet[-1, first.swing_foot, 0] += 0.4
        second = replace(first, clip_index=1, foot_position_local=farther_feet)
        index = ContactPhaseActionIndex(
            actions=(first, second),
            inventory=ContactPhaseInventory(retained_count=2, rejected_by_reason={}),
            exact_successor_indices=(None, None),
        )

        ranked = contact_anchored_prefilter(
            state=_state(),
            cache=build_quality_action_cache(index),
            native_quality=build_native_quality_index(index),
            desired_landing_foot=1,
            desired_landing_world_xyz=np.array((0.9, -0.1, 0.0)),
            command_target_world_xy=np.array((0.2, 0.0)),
            constraints=OracleConstraints(),
            maximum_count=2,
        )

        self.assertEqual(ranked[0], 1)


if __name__ == "__main__":
    unittest.main()
