import unittest

import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_motion_features import FeatureNormalization, TorchMotionDatabase
from mm_sonic.torch_terrain_skill_search import (
    select_terrain_skill,
    skill_entry_eligibility,
)
from mm_sonic.torch_terrain_skills import (
    SkillInterval,
    TerrainSkill,
    TerrainSkillInventory,
)


def _database() -> TorchMotionDatabase:
    features = torch.full((10, 27), 10.0, dtype=torch.float32)
    features[2] = 0.3
    features[3] = 0.2
    features[7] = 0.4
    zeros = torch.zeros(27, dtype=torch.float32)
    return TorchMotionDatabase(
        folder=object(),
        device=torch.device("cpu"),
        normalization=FeatureNormalization(zeros, torch.ones_like(zeros)),
        reset_row=0,
        _search_features=features,
        _search_clip_index=torch.arange(10, dtype=torch.long),
        _search_frame_index=torch.zeros(10, dtype=torch.long),
        _source_row_map={},
    )


def _skill(index: int, rows: tuple[int, ...]) -> TerrainSkill:
    return TerrainSkill(
        skill_index=index,
        clip_index=index,
        interval=SkillInterval(0, 5, 10),
        entry_rows=rows,
        support_mask=torch.ones((10, 2), dtype=torch.bool),
        foot_surface_height_m=torch.zeros((10, 2)),
    )


class TerrainSkillSearchTest(unittest.TestCase):
    def setUp(self):
        self.database = _database()
        skills = (_skill(0, (2, 3)), _skill(1, (7,)))
        self.inventory = TerrainSkillInventory(
            skills=skills,
            rejected_by_reason={},
            row_to_skill={2: 0, 3: 0, 7: 1},
        )

    def test_only_php_entry_windows_are_searchable(self):
        eligible = skill_entry_eligibility(self.database, self.inventory)
        self.assertEqual(torch.nonzero(eligible).flatten().tolist(), [2, 3, 7])

    def test_terrain_gate_removes_candidates_before_feature_ranking(self):
        visited = []

        result = select_terrain_skill(
            self.database,
            self.inventory,
            torch.zeros(27),
            current_clip_index=99,
            current_frame_index=0,
            terrain_validator=lambda skill, row: visited.append((skill.skill_index, row))
            is None
            and skill.skill_index == 1,
        )

        self.assertEqual(visited, [(0, 2), (0, 3), (1, 7)])
        self.assertEqual(result.skill.skill_index, 1)
        self.assertEqual(result.selected_row, 7)
        self.assertEqual(result.rejected_by_reason, {"terrain": 2})

    def test_reports_no_compatible_entry(self):
        with self.assertRaisesRegex(ContractError, "no terrain-compatible"):
            select_terrain_skill(
                self.database,
                self.inventory,
                torch.zeros(27),
                current_clip_index=99,
                current_frame_index=0,
                terrain_validator=lambda skill, row: False,
            )


if __name__ == "__main__":
    unittest.main()
