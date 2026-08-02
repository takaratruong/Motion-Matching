import unittest
from types import SimpleNamespace

import numpy as np
import torch

from mm_sonic.torch_motion_features import FeatureNormalization, TorchMotionDatabase
from mm_sonic.torch_motion_matcher import MatcherConfig
from mm_sonic.torch_terrain_skill_horizon_rollout import (
    TerrainSkillHorizonMatcher,
    terrain_height_targets,
)
from mm_sonic.torch_terrain_skill_horizon_search import HorizonTargets
from mm_sonic.torch_terrain_skill_horizons import TerrainSkillHorizonInventory
from mm_sonic.torch_terrain_skills import (
    SkillInterval,
    TerrainSkill,
    TerrainSkillInventory,
)


class _HeightGrid:
    def sample_xy(self, points):
        return points[:, 0]


class _XYProductGrid:
    def sample_xy(self, points):
        return points[:, 0] * points[:, 1]


class _Alignment:
    def matcher_to_scene_xy(self, points):
        return points


class _ConstantGrid:
    def __init__(self):
        self.calls = 0

    def sample_xy(self, points):
        self.calls += 1
        return torch.zeros(points.shape[:-1], device=points.device)


class _FootKinematics:
    def foot_positions(self, joint, root, quaternion):
        return np.repeat(root[:, None, :], 2, axis=1)


def _transactional_fixture(
    *,
    result_filter=None,
    contact_phase_gate=False,
    footprint_terrain_targets=False,
    footprint_preserve_horizon=False,
):
    frames = 80
    body_position = np.zeros((frames, 3, 3), dtype=np.float32)
    body_position[:, :, 0] = np.arange(frames, dtype=np.float32)[:, None] * 0.01
    body_position[:, :, 2] = 0.8
    body_quaternion = np.zeros((frames, 3, 4), dtype=np.float32)
    body_quaternion[..., 0] = 1.0
    clip = SimpleNamespace(
        relative_path="terrain/motion.npz",
        joint_position=np.zeros((frames, 2), dtype=np.float32),
        joint_velocity=np.zeros((frames, 2), dtype=np.float32),
        body_position_world=body_position,
        body_quaternion_world_wxyz=body_quaternion,
        body_linear_velocity_world=np.zeros((frames, 3, 3), dtype=np.float32),
        body_angular_velocity_world=np.zeros((frames, 3, 3), dtype=np.float32),
    )
    folder = SimpleNamespace(
        clips=(clip,),
        layout=SimpleNamespace(
            root_body_index=0,
            left_foot_body_index=1,
            right_foot_body_index=2,
        ),
    )
    support = torch.ones((frames, 2), dtype=torch.bool)
    skill = TerrainSkill(
        skill_index=0,
        clip_index=0,
        interval=SkillInterval(0, 1, 60),
        entry_rows=(0,),
        support_mask=support,
        foot_surface_height_m=torch.zeros((frames, 2)),
    )
    skills = TerrainSkillInventory(
        skills=(skill,), rejected_by_reason={}, row_to_skill={0: 0}
    )
    horizons = TerrainSkillHorizonInventory(
        entry_row=torch.tensor([0]),
        skill_index=torch.tensor([0]),
        clip_index=torch.tensor([0]),
        entry_frame=torch.tensor([0]),
        target_frames=torch.tensor([25]),
        endpoint_frame_exclusive=torch.tensor([26]),
        root_displacement_local_xy=torch.tensor([[0.25, 0.0]]),
        yaw_delta_rad=torch.zeros(1),
        root_height_delta_m=torch.zeros(1),
        surface_height_delta_m=torch.zeros((1, 2)),
        maximum_stall_frames=torch.zeros(1, dtype=torch.long),
        duration_frames=torch.tensor([26]),
        rejected_by_reason={},
    )
    zeros = torch.zeros(27)
    database = TorchMotionDatabase(
        folder=folder,
        device=torch.device("cpu"),
        normalization=FeatureNormalization(zeros, torch.ones_like(zeros)),
        reset_row=0,
        _search_features=torch.zeros((1, 27)),
        _search_clip_index=torch.tensor([0]),
        _search_frame_index=torch.tensor([0]),
        _source_row_map={(0, 0): 0},
    )
    reset_result = SimpleNamespace(
        joint_position=torch.zeros(2),
        joint_velocity=torch.zeros(2),
        root_position_world=torch.tensor([0.0, 0.0, 0.8]),
        root_orientation_world_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        root_linear_velocity_world=torch.zeros(3),
        diagnostics=SimpleNamespace(selected_clip_path="reset", selected_frame=0),
    )

    class _Base:
        def __init__(self):
            self.database = database
            self._state = None

        def reset(self, *, root_position_world_xy=(0.0, 0.0)):
            return reset_result

    grid = _ConstantGrid()
    matcher = TerrainSkillHorizonMatcher(
        base_matcher=_Base(),
        skill_inventory=skills,
        horizon_inventory=horizons,
        dataset=SimpleNamespace(folder=folder),
        query_terrain=SimpleNamespace(query_grid=grid, alignment=_Alignment()),
        foot_kinematics=_FootKinematics(),
        config=MatcherConfig(),
        result_filter=result_filter,
        contact_phase_gate=contact_phase_gate,
        footprint_terrain_targets=footprint_terrain_targets,
        footprint_preserve_horizon=footprint_preserve_horizon,
    )
    return matcher, grid


class TerrainSkillHorizonRolloutTest(unittest.TestCase):
    def test_footprint_horizon_layer_is_explicit(self):
        matcher, _grid = _transactional_fixture(
            footprint_terrain_targets=True,
            footprint_preserve_horizon=True,
        )

        self.assertTrue(matcher.footprint_preserve_horizon)

    def test_contact_phase_gate_is_explicit(self):
        matcher, _grid = _transactional_fixture(contact_phase_gate=True)

        self.assertTrue(matcher.contact_phase_gate)

    def test_optional_result_filter_owns_reset_and_committed_state(self):
        class _Filter:
            def __init__(self):
                self.reset_count = 0
                self.apply_count = 0

            def reset(self):
                self.reset_count += 1

            def apply(self, result):
                self.apply_count += 1
                return SimpleNamespace(
                    **{
                        **vars(result),
                        "root_position_world": result.root_position_world
                        + torch.tensor((0.0, 0.0, 0.01)),
                    }
                )

        result_filter = _Filter()
        matcher, _grid = _transactional_fixture(result_filter=result_filter)

        reset = matcher.reset()
        committed = matcher.commit(matcher.prepare_step((0.0, 0.0), 0.0))

        self.assertEqual(result_filter.reset_count, 1)
        self.assertEqual(result_filter.apply_count, 2)
        self.assertAlmostEqual(float(reset.root_position_world[2]), 0.81)
        self.assertAlmostEqual(float(committed.root_position_world[2]), 0.82)
        self.assertIs(matcher._last_result, committed)

    def test_terrain_height_targets_follow_commanded_world_path(self):
        targets = HorizonTargets(
            frames=torch.tensor([25, 50, 100]),
            displacement_local_xy=torch.tensor(
                [[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]]
            ),
            yaw_delta_rad=torch.zeros(3),
            root_height_delta_m=torch.zeros(3),
            surface_height_delta_m=torch.zeros((3, 2)),
        )
        terrain = SimpleNamespace(
            query_grid=_HeightGrid(), alignment=_Alignment()
        )

        enriched = terrain_height_targets(
            targets,
            current_root_position_world=torch.tensor([1.0, 2.0, 0.8]),
            current_root_yaw=torch.tensor(0.0),
            query_terrain=terrain,
        )

        expected = torch.tensor([0.5, 1.0, 2.0])
        self.assertTrue(torch.allclose(enriched.root_height_delta_m, expected))
        self.assertTrue(
            torch.allclose(
                enriched.surface_height_delta_m,
                expected[:, None].expand(-1, 2),
            )
        )

    def test_terrain_height_targets_sample_left_and_right_footprints(self):
        targets = HorizonTargets(
            frames=torch.tensor([25, 50, 100]),
            displacement_local_xy=torch.tensor(
                [[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]]
            ),
            yaw_delta_rad=torch.zeros(3),
            root_height_delta_m=torch.zeros(3),
            surface_height_delta_m=torch.zeros((3, 2)),
        )
        terrain = SimpleNamespace(
            query_grid=_XYProductGrid(), alignment=_Alignment()
        )

        enriched = terrain_height_targets(
            targets,
            current_root_position_world=torch.tensor([1.0, 2.0, 0.8]),
            current_root_yaw=torch.tensor(0.0),
            current_foot_position_world=torch.tensor(
                [[1.0, 2.3, 0.0], [1.0, 1.8, 0.0]]
            ),
            footprint_split_gain=0.5,
            query_terrain=terrain,
        )

        self.assertTrue(
            torch.allclose(
                enriched.surface_height_delta_m,
                torch.tensor(
                    [[1.0625, 0.9375], [2.125, 1.875], [4.25, 3.75]]
                ),
                atol=1e-6,
            )
        )

    def test_committed_chunk_searches_once_and_emits_consecutive_source_frames(self):
        matcher, grid = _transactional_fixture()
        matcher.reset()
        source_frames = []
        for _ in range(5):
            prepared = matcher.prepare_step((0.0, 0.0), 0.0)
            result = matcher.commit(prepared)
            source_frames.append(result.diagnostics.selected_frame)
            if len(source_frames) == 1:
                search_calls = grid.calls

        self.assertEqual(source_frames, [0, 1, 2, 3, 4])
        self.assertEqual(grid.calls, search_calls)
        self.assertEqual(len(matcher.chunk_events), 1)
        event = matcher.chunk_events[0]
        self.assertEqual(event.entry_row, 0)
        self.assertEqual(event.target_frames, 25)
        self.assertEqual(event.endpoint_frame_exclusive, 26)
        self.assertEqual(event.release_reason, "initial")


if __name__ == "__main__":
    unittest.main()
