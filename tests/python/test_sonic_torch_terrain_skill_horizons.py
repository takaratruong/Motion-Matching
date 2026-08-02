import math
from pathlib import Path
import unittest

import numpy as np
import torch

from mm_sonic.torch_motion_data import G1_TAKARA_LAYOUT, MotionClip, MotionFolder
from mm_sonic.torch_motion_features import FeatureNormalization, TorchMotionDatabase
from mm_sonic.torch_terrain_features import TerrainDataset
from mm_sonic.torch_terrain_skill_horizons import (
    build_horizon_inventory,
    describe_horizon,
    next_sequential_horizon_endpoint,
    remaining_stall_profile,
    stable_horizon_endpoints,
)
from mm_sonic.torch_terrain_skills import (
    SkillInterval,
    TerrainSkill,
    TerrainSkillInventory,
)


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(yaw)
    return torch.stack(
        (torch.cos(yaw / 2), zeros, zeros, torch.sin(yaw / 2)), dim=-1
    )


class TerrainSkillHorizonPrimitiveTest(unittest.TestCase):
    def test_endpoint_is_first_double_support_within_lateness_cap(self):
        support = torch.zeros((140, 2), dtype=torch.bool)
        support[29] = True
        support[52] = True
        support[90] = True

        self.assertEqual(
            stable_horizon_endpoints(
                support, entry_frame=2, playback_stop=120
            ),
            ((25, 30), (50, 53)),
        )

    def test_remaining_stall_profile_is_time_local(self):
        root_xy = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]
        )
        self.assertEqual(
            remaining_stall_profile(root_xy).tolist(), [2, 1, 0, 0, 1]
        )

    def test_descriptor_uses_entry_heading_and_unwrapped_yaw(self):
        root = torch.zeros((6, 3), dtype=torch.float32)
        root[1] = torch.tensor([0.0, 0.0, 0.8])
        root[4] = torch.tensor([-0.5, 1.0, 0.98])
        yaw = torch.tensor(
            [0.0, math.pi / 2, 2.2, 2.8, math.pi / 2 + 1.75, 0.0]
        )
        surface = torch.zeros((6, 2), dtype=torch.float32)
        surface[4] = torch.tensor([0.18, 0.18])

        record = describe_horizon(
            root_position_world=root,
            root_orientation_world_wxyz=_yaw_quaternion(yaw),
            foot_surface_height_m=surface,
            entry_frame=1,
            endpoint_frame_exclusive=5,
        )

        self.assertTrue(
            torch.allclose(
                record.root_displacement_local_xy,
                torch.tensor([1.0, 0.5]),
                atol=1e-6,
            )
        )
        self.assertAlmostEqual(record.yaw_delta_rad, 1.75, places=5)
        self.assertAlmostEqual(record.root_height_delta_m, 0.18, places=6)
        self.assertTrue(
            torch.allclose(
                record.surface_height_delta_m, torch.tensor([0.18, 0.18])
            )
        )

    def test_builder_flattens_owned_entry_horizons_deterministically(self):
        frames = 140
        body_position = np.zeros((frames, 30, 3), dtype=np.float32)
        body_position[:, 0, 0] = np.arange(frames, dtype=np.float32) * 0.01
        body_position[:, 0, 2] = 0.8
        body_quaternion = np.zeros((frames, 30, 4), dtype=np.float32)
        body_quaternion[..., 0] = 1.0
        clip = MotionClip(
            relative_path="terrain/motion.npz",
            fps=50,
            joint_position=np.zeros((frames, 29), dtype=np.float32),
            joint_velocity=np.zeros((frames, 29), dtype=np.float32),
            body_position_world=body_position,
            body_quaternion_world_wxyz=body_quaternion,
            body_linear_velocity_world=np.zeros((frames, 30, 3), dtype=np.float32),
            body_angular_velocity_world=np.zeros((frames, 30, 3), dtype=np.float32),
        )
        folder = MotionFolder(Path("."), G1_TAKARA_LAYOUT, (clip,), "identity")
        dataset = TerrainDataset(
            root=Path("."), folder=folder, clip_grids=(None,),
            clip_alignments=(None,), manifest_sha256="manifest", _manifest={},
            device=torch.device("cpu"),
        )
        zeros = torch.zeros(27)
        database = TorchMotionDatabase(
            folder=folder,
            device=torch.device("cpu"),
            normalization=FeatureNormalization(zeros, torch.ones_like(zeros)),
            reset_row=0,
            _search_features=torch.zeros((2, 27)),
            _search_clip_index=torch.zeros(2, dtype=torch.long),
            _search_frame_index=torch.tensor([0, 2]),
            _source_row_map={(0, 0): 0, (0, 2): 1},
        )
        support = torch.zeros((frames, 2), dtype=torch.bool)
        support[29] = True
        support[52] = True
        support[90] = True
        skill = TerrainSkill(
            skill_index=0,
            clip_index=0,
            interval=SkillInterval(0, 1, 120),
            entry_rows=(0, 1),
            support_mask=support,
            foot_surface_height_m=torch.zeros((frames, 2)),
        )
        skill_inventory = TerrainSkillInventory(
            skills=(skill,), rejected_by_reason={}, row_to_skill={0: 0, 1: 0}
        )

        inventory = build_horizon_inventory(dataset, database, skill_inventory)

        self.assertEqual(inventory.entry_row.tolist(), [0, 0, 1, 1])
        self.assertEqual(inventory.target_frames.tolist(), [25, 50, 25, 50])
        self.assertEqual(
            inventory.endpoint_frame_exclusive.tolist(), [30, 53, 30, 53]
        )
        self.assertEqual(inventory.rejected_by_reason, {"no_endpoint": 2})


class SequentialEndpointTest(unittest.TestCase):
    def test_next_endpoint_is_after_current_exclusive_endpoint(self):
        support = torch.zeros((100, 2), dtype=torch.bool)
        support[24] = True
        support[49] = True
        support[74] = True

        endpoint = next_sequential_horizon_endpoint(
            support,
            current_endpoint_frame_exclusive=25,
            playback_stop=75,
        )

        self.assertEqual(endpoint, 50)

    def test_no_later_stable_endpoint_returns_none(self):
        support = torch.zeros((75, 2), dtype=torch.bool)
        support[24] = True

        self.assertIsNone(
            next_sequential_horizon_endpoint(
                support,
                current_endpoint_frame_exclusive=25,
                playback_stop=75,
            )
        )


if __name__ == "__main__":
    unittest.main()
