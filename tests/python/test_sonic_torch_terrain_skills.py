import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_motion_features import (
    FeatureNormalization,
    TorchMotionDatabase,
)
from mm_sonic.torch_terrain_features import TerrainDataset
from mm_sonic.torch_terrain_skills import (
    SkillInterval,
    build_terrain_skill_inventory,
    extract_skill_intervals,
    source_contact_height_p95_m,
)


class TerrainSkillIntervalTest(unittest.TestCase):
    def test_source_contact_quality_reads_expanded_manifest(self):
        dataset = type(
            "Dataset",
            (),
            {
                "manifest": {
                    "clips": [
                        {"kind": "flat", "terrain": {"kind": "flat"}},
                        {
                            "kind": "terrain",
                            "admission": {
                                "contact_height_error_m": {"p95": 0.0125}
                            },
                        },
                    ]
                }
            },
        )()

        self.assertIsNone(source_contact_height_p95_m(dataset, 0))
        self.assertEqual(source_contact_height_p95_m(dataset, 1), 0.0125)

    def test_source_contact_quality_rejects_malformed_value(self):
        dataset = type(
            "Dataset",
            (),
            {
                "manifest": {
                    "clips": [
                        {
                            "kind": "terrain",
                            "admission": {
                                "contact_height_error_m": {"p95": "bad"}
                            },
                        }
                    ]
                }
            },
        )()

        with self.assertRaisesRegex(ContractError, "contact quality"):
            source_contact_height_p95_m(dataset, 0)

    def test_extracts_complete_episode_with_pre_skill_entry(self):
        support = torch.ones((30, 2), dtype=torch.bool)
        heights = torch.zeros((30, 2), dtype=torch.float32)
        heights[10:22] = 0.18

        self.assertEqual(
            extract_skill_intervals(
                support,
                heights,
                valid_frame_stop=25,
                entry_window_frames=5,
                minimum_surface_change_m=0.08,
            ),
            (SkillInterval(entry_start=4, playback_start=9, playback_stop=23),),
        )

    def test_never_uses_flight_as_a_skill_boundary(self):
        support = torch.ones((30, 2), dtype=torch.bool)
        support[8:12, 1] = False
        heights = torch.zeros((30, 2), dtype=torch.float32)
        heights[12:22] = 0.18

        interval = extract_skill_intervals(
            support,
            heights,
            valid_frame_stop=25,
            entry_window_frames=5,
            minimum_surface_change_m=0.08,
        )[0]

        self.assertTrue(bool(support[interval.playback_start].all()))
        self.assertTrue(bool(support[interval.playback_stop - 1].all()))
        self.assertEqual(interval.playback_start, 7)

    def test_merges_height_events_without_a_stable_gap(self):
        support = torch.ones((40, 2), dtype=torch.bool)
        heights = torch.zeros((40, 2), dtype=torch.float32)
        heights[10:14] = 0.12
        heights[18:22] = 0.16

        intervals = extract_skill_intervals(
            support,
            heights,
            valid_frame_stop=35,
            entry_window_frames=4,
            minimum_surface_change_m=0.08,
            stable_gap_frames=6,
        )

        self.assertEqual(intervals, (SkillInterval(5, 9, 23),))

    def test_rejects_mismatched_profiles(self):
        with self.assertRaisesRegex(ContractError, "matching shape"):
            extract_skill_intervals(
                torch.ones((10, 2), dtype=torch.bool),
                torch.zeros((9, 2), dtype=torch.float32),
                valid_frame_stop=8,
            )

    def test_flat_clip_is_available_as_level_surface_skill(self):
        frames = 80
        clip = SimpleNamespace(
            valid_frame_stop=frames,
            body_position_world=np.zeros((frames, 3, 3), np.float32),
        )
        folder = SimpleNamespace(
            clips=(clip,),
            layout=SimpleNamespace(
                left_foot_body_index=1,
                right_foot_body_index=2,
            ),
        )
        dataset = TerrainDataset(
            root=Path("."),
            folder=folder,
            clip_grids=(None,),
            clip_alignments=(None,),
            manifest_sha256="dataset",
            _manifest={"clips": [{"kind": "flat"}]},
            device=torch.device("cpu"),
        )
        zeros = torch.zeros(27)
        database = TorchMotionDatabase(
            folder=folder,
            device=torch.device("cpu"),
            normalization=FeatureNormalization(zeros, torch.ones_like(zeros)),
            reset_row=0,
            _search_features=torch.zeros((frames, 27)),
            _search_clip_index=torch.zeros(frames, dtype=torch.long),
            _search_frame_index=torch.arange(frames),
            _source_row_map={(0, frame): frame for frame in range(frames)},
        )

        with mock.patch(
            "mm_sonic.torch_terrain_skills.source_support_mask",
            return_value=torch.ones((frames, 2), dtype=torch.bool),
        ):
            inventory = build_terrain_skill_inventory(dataset, database)

        self.assertEqual(len(inventory.skills), 1)
        skill = inventory.skills[0]
        self.assertEqual(skill.interval, SkillInterval(0, 0, frames))
        self.assertEqual(skill.entry_rows, tuple(range(31)))
        self.assertTrue(
            torch.equal(
                skill.foot_surface_height_m, torch.zeros(frames, 2)
            )
        )


if __name__ == "__main__":
    unittest.main()
