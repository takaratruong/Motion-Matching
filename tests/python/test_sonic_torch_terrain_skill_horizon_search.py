import math
from dataclasses import replace
import unittest

import torch

from mm_sonic.torch_motion_features import FeatureNormalization, TorchMotionDatabase
from mm_sonic.torch_motion_matcher import MatcherConfig
from mm_sonic.torch_terrain_skill_horizon_search import (
    HorizonSearchConfig,
    HorizonTargets,
    predict_horizon_targets,
    rank_horizon_candidates,
    select_horizon_candidate,
)
from mm_sonic.torch_terrain_skill_horizons import TerrainSkillHorizonInventory


def _database() -> TorchMotionDatabase:
    features = torch.zeros((4, 27), dtype=torch.float32)
    features[2] = 0.2
    features[3] = 0.2
    zeros = torch.zeros(27, dtype=torch.float32)
    return TorchMotionDatabase(
        folder=None,
        device=torch.device("cpu"),
        normalization=FeatureNormalization(zeros, torch.ones_like(zeros)),
        reset_row=0,
        _search_features=features,
        _search_clip_index=torch.arange(4),
        _search_frame_index=torch.zeros(4, dtype=torch.long),
        _source_row_map={},
    )


def _inventory() -> TerrainSkillHorizonInventory:
    return TerrainSkillHorizonInventory(
        entry_row=torch.arange(4, dtype=torch.long),
        skill_index=torch.arange(4, dtype=torch.long),
        clip_index=torch.arange(4, dtype=torch.long),
        entry_frame=torch.zeros(4, dtype=torch.long),
        target_frames=torch.full((4,), 25, dtype=torch.long),
        endpoint_frame_exclusive=torch.full((4,), 30, dtype=torch.long),
        root_displacement_local_xy=torch.tensor(
            [[0.0, 0.0], [0.5, 0.0], [0.4, 0.1], [0.4, 0.1]]
        ),
        yaw_delta_rad=torch.tensor([0.7, -0.7, 0.6, 0.6]),
        root_height_delta_m=torch.tensor([0.1, 0.1, 0.08, 0.08]),
        surface_height_delta_m=torch.tensor(
            [[0.1, 0.1], [0.1, 0.1], [0.08, 0.08], [0.08, 0.08]]
        ),
        maximum_stall_frames=torch.tensor([25, 0, 2, 2]),
        duration_frames=torch.full((4,), 30, dtype=torch.long),
        rejected_by_reason={},
    )


def _target() -> HorizonTargets:
    return HorizonTargets(
        frames=torch.tensor([25], dtype=torch.long),
        displacement_local_xy=torch.tensor([[0.5, 0.0]]),
        yaw_delta_rad=torch.tensor([0.7]),
        root_height_delta_m=torch.tensor([0.1]),
        surface_height_delta_m=torch.tensor([[0.1, 0.1]]),
    )


class HorizonTargetTest(unittest.TestCase):
    def test_targets_sample_bounded_command_at_all_three_horizons(self):
        target = predict_horizon_targets(
            current_velocity_world_xy=torch.zeros(2),
            current_heading_world_yaw=torch.tensor(0.0),
            requested_velocity_world_xy=torch.tensor([1.0, 0.0]),
            requested_heading_world_yaw=torch.tensor(math.pi / 2),
            matcher_config=MatcherConfig(
                acceleration_mps2=1000.0,
                deceleration_mps2=1000.0,
                yaw_rate_rad_s=1000.0,
            ),
        )

        self.assertEqual(target.frames.tolist(), [25, 50, 100])
        self.assertTrue(
            torch.allclose(
                target.displacement_local_xy,
                torch.tensor([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]]),
                atol=2e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                target.yaw_delta_rad,
                torch.full((3,), math.pi / 2),
                atol=2e-6,
            )
        )


class HorizonRankingTest(unittest.TestCase):
    def setUp(self):
        self.database = _database()
        self.inventory = _inventory()
        self.target = _target()

    def test_wrong_turn_and_zero_progress_are_hard_gated_and_ties_are_stable(self):
        ranked = rank_horizon_candidates(
            self.database,
            self.inventory,
            torch.zeros(27),
            self.target,
            HorizonSearchConfig(),
        )

        self.assertEqual(ranked.candidate_indices.tolist(), [2, 3])
        self.assertEqual(ranked.rejected_by_reason["progress"], 1)
        self.assertEqual(ranked.rejected_by_reason["turn"], 1)

    def test_combined_cost_matches_independent_components(self):
        config = HorizonSearchConfig()
        ranked = rank_horizon_candidates(
            self.database,
            self.inventory,
            torch.zeros(27),
            self.target,
            config,
        )
        index = int(torch.nonzero(ranked.candidate_indices == 2)[0])
        entry = config.entry_weight * 27 * 0.2**2
        displacement = config.displacement_weight * (
            ((0.4 - 0.5) / 0.5) ** 2 + ((0.1 - 0.0) / 0.5) ** 2
        )
        yaw = config.yaw_weight * ((0.6 - 0.7) / (math.pi / 2)) ** 2
        height = config.height_weight * (
            ((0.08 - 0.1) / 0.18) ** 2
            + 2 * ((0.08 - 0.1) / 0.18) ** 2
        ) / 3
        duration = config.duration_weight * ((30 - 25) / 100) ** 2
        stall = config.stall_weight * (2 / 25) ** 2
        expected_outcome = displacement + yaw + height + duration + stall

        self.assertAlmostEqual(float(ranked.entry_cost[index]), entry, places=6)
        self.assertAlmostEqual(
            float(ranked.outcome_cost[index]), expected_outcome, places=6
        )
        self.assertAlmostEqual(
            float(ranked.total_cost[index]), entry + expected_outcome, places=6
        )

    def test_selection_validates_in_rank_order_and_reports_terrain_rejections(self):
        visited = []
        result = select_horizon_candidate(
            self.database,
            self.inventory,
            torch.zeros(27),
            self.target,
            terrain_validator=lambda record, row, endpoint: (
                visited.append((record, row, endpoint)) is None and record == 3
            ),
        )

        self.assertEqual(visited, [(2, 2, 30), (3, 3, 30)])
        self.assertEqual(result.record_index, 3)
        self.assertEqual(result.rejected_by_reason["terrain"], 1)

    def test_selection_can_preserve_a_required_horizon_layer(self):
        inventory = replace(
            self.inventory,
            target_frames=torch.tensor([25, 25, 25, 50]),
        )
        target = HorizonTargets(
            frames=torch.tensor([25, 50]),
            displacement_local_xy=torch.tensor([[0.5, 0.0], [0.5, 0.0]]),
            yaw_delta_rad=torch.tensor([0.7, 0.7]),
            root_height_delta_m=torch.tensor([0.1, 0.1]),
            surface_height_delta_m=torch.tensor([[0.1, 0.1], [0.1, 0.1]]),
        )
        visited = []

        result = select_horizon_candidate(
            self.database,
            inventory,
            torch.zeros(27),
            target,
            terrain_validator=lambda record, row, endpoint: (
                visited.append(record) is None
            ),
            required_target_frames=50,
        )

        self.assertEqual(visited, [3])
        self.assertEqual(result.record_index, 3)
        self.assertEqual(result.rejected_by_reason["horizon"], 1)

    def test_current_source_neighborhood_is_excluded(self):
        ranked = rank_horizon_candidates(
            self.database,
            self.inventory,
            torch.zeros(27),
            self.target,
            HorizonSearchConfig(),
            current_clip_index=2,
            current_frame_index=0,
            matcher_config=MatcherConfig(exclusion_frames=20),
        )

        self.assertEqual(ranked.candidate_indices.tolist(), [3])
        self.assertEqual(ranked.rejected_by_reason["local"], 1)


if __name__ == "__main__":
    unittest.main()
