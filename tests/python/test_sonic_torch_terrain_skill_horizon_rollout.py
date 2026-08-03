import math
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch

from mm_sonic.torch_motion_features import FeatureNormalization, TorchMotionDatabase
from mm_sonic.torch_motion_matcher import MatcherConfig
from mm_sonic.torch_terrain_skill_horizon_rollout import (
    TerrainSkillHorizonMatcher,
    terrain_height_targets,
)
from mm_sonic.torch_terrain_skill_horizon_search import (
    HorizonTargets,
    predict_horizon_targets,
)
from mm_sonic.torch_terrain_skill_horizons import TerrainSkillHorizonInventory
from mm_sonic.torch_terrain_skills import (
    SkillInterval,
    TerrainSkill,
    TerrainSkillInventory,
)


class _HeightGrid:
    def sample_xy(self, points):
        return points[:, 0]


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
    turning_clip_paths=None,
    maximum_translation_warp_m=0.0,
    maximum_yaw_warp_rad=0.0,
    minimum_endpoint_warp_yaw_rad=0.0,
    maximum_endpoint_warp_yaw_rad=math.pi,
    maximum_endpoint_warp_terrain_delta_m=math.inf,
    minimum_endpoint_warp_velocity_heading_alignment=-1.0,
    continuous_skill_enabled=False,
    contact_feasibility_enabled=False,
    continuation_surface_tolerance_m=0.08,
    two_candidate_runway: bool | None = None,
    runway_suffix_direction: tuple[float, float] = (1.0, 0.0),
    runway_suffix_step_m: float = 0.01,
    single_support: bool = False,
):
    frames = 80
    body_position = np.zeros((frames, 3, 3), dtype=np.float32)
    body_position[:, :, 0] = np.arange(frames, dtype=np.float32)[:, None] * 0.01
    body_position[:, :, 2] = 0.8
    body_position[:, 1:, 2] = 0.035
    if two_candidate_runway is not None:
        suffix = (
            np.arange(frames - 25, dtype=np.float32)[:, None]
            * runway_suffix_step_m
        )
        body_position[25:, :, 0] = body_position[25, 0, 0] + (
            suffix * runway_suffix_direction[0]
        )
        body_position[25:, :, 1] = body_position[25, 0, 1] + (
            suffix * runway_suffix_direction[1]
        )
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
    if single_support:
        support[0, 1] = False
    skill = TerrainSkill(
        skill_index=0,
        clip_index=0,
        interval=SkillInterval(0, 1, 60),
        entry_rows=(0,),
        support_mask=support,
        foot_surface_height_m=torch.zeros((frames, 2)),
    )
    skill_values = (skill,)
    row_to_skill = {0: 0}
    if two_candidate_runway is not None:
        skill = TerrainSkill(
            **{
                **skill.__dict__,
                "interval": SkillInterval(0, 1, 26),
            }
        )
        runway_skill = TerrainSkill(
            skill_index=1,
            clip_index=0,
            interval=SkillInterval(
                0, 1, 60 if two_candidate_runway else 26
            ),
            entry_rows=(1,),
            support_mask=support,
            foot_surface_height_m=torch.zeros((frames, 2)),
        )
        skill_values = (skill, runway_skill)
        row_to_skill = {0: 0, 1: 1}
    skills = TerrainSkillInventory(
        skills=skill_values, rejected_by_reason={}, row_to_skill=row_to_skill
    )
    record_count = len(skill_values)
    horizons = TerrainSkillHorizonInventory(
        entry_row=torch.arange(record_count),
        skill_index=torch.arange(record_count),
        clip_index=torch.zeros(record_count, dtype=torch.long),
        entry_frame=torch.zeros(record_count, dtype=torch.long),
        target_frames=torch.full((record_count,), 25, dtype=torch.long),
        endpoint_frame_exclusive=torch.full(
            (record_count,), 26, dtype=torch.long
        ),
        root_displacement_local_xy=torch.tensor(
            [[0.25, 0.0]] * record_count
        ),
        yaw_delta_rad=torch.zeros(record_count),
        root_height_delta_m=torch.zeros(record_count),
        surface_height_delta_m=torch.zeros((record_count, 2)),
        maximum_stall_frames=torch.zeros(record_count, dtype=torch.long),
        duration_frames=torch.full((record_count,), 26, dtype=torch.long),
        rejected_by_reason={},
    )
    zeros = torch.zeros(27)
    database = TorchMotionDatabase(
        folder=folder,
        device=torch.device("cpu"),
        normalization=FeatureNormalization(zeros, torch.ones_like(zeros)),
        reset_row=0,
        _search_features=torch.stack(
            (torch.zeros(27), torch.full((27,), -0.5))
        )[:record_count],
        _search_clip_index=torch.zeros(record_count, dtype=torch.long),
        _search_frame_index=torch.zeros(record_count, dtype=torch.long),
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
        turning_clip_paths=turning_clip_paths,
        maximum_translation_warp_m=maximum_translation_warp_m,
        maximum_yaw_warp_rad=maximum_yaw_warp_rad,
        minimum_endpoint_warp_yaw_rad=minimum_endpoint_warp_yaw_rad,
        maximum_endpoint_warp_yaw_rad=maximum_endpoint_warp_yaw_rad,
        maximum_endpoint_warp_terrain_delta_m=(
            maximum_endpoint_warp_terrain_delta_m
        ),
        minimum_endpoint_warp_velocity_heading_alignment=(
            minimum_endpoint_warp_velocity_heading_alignment
        ),
        continuous_skill_enabled=continuous_skill_enabled,
        contact_feasibility_enabled=contact_feasibility_enabled,
        continuation_surface_tolerance_m=continuation_surface_tolerance_m,
    )
    return matcher, grid


class TerrainSkillHorizonRolloutTest(unittest.TestCase):
    def test_endpoint_warp_limits_are_explicit(self):
        matcher, _grid = _transactional_fixture(
            maximum_translation_warp_m=0.15,
            maximum_yaw_warp_rad=0.30,
            minimum_endpoint_warp_yaw_rad=0.30,
            maximum_endpoint_warp_yaw_rad=0.90,
            maximum_endpoint_warp_terrain_delta_m=0.02,
            minimum_endpoint_warp_velocity_heading_alignment=0.80,
        )

        self.assertEqual(matcher.maximum_translation_warp_m, 0.15)
        self.assertEqual(matcher.maximum_yaw_warp_rad, 0.30)
        self.assertEqual(matcher.minimum_endpoint_warp_yaw_rad, 0.30)
        self.assertEqual(matcher.maximum_endpoint_warp_yaw_rad, 0.90)
        self.assertEqual(matcher.maximum_endpoint_warp_terrain_delta_m, 0.02)
        self.assertEqual(
            matcher.minimum_endpoint_warp_velocity_heading_alignment, 0.80
        )
        self.assertEqual(matcher.continuation_surface_tolerance_m, 0.08)

    def test_endpoint_warp_regime_latches_at_command_change(self):
        matcher, _grid = _transactional_fixture(
            maximum_translation_warp_m=0.025,
            maximum_yaw_warp_rad=0.10,
            minimum_endpoint_warp_yaw_rad=0.30,
            maximum_endpoint_warp_yaw_rad=0.90,
        )
        matcher.reset()

        matcher._update_endpoint_warp_command(((1.0, 0.0), 0.60))
        self.assertTrue(matcher._endpoint_warp_command_enabled)
        matcher._update_endpoint_warp_command(((1.0, 0.0), 1.20))
        self.assertFalse(matcher._endpoint_warp_command_enabled)
        matcher._update_endpoint_warp_command(((1.0, 0.0), 0.0))
        self.assertFalse(matcher._endpoint_warp_command_enabled)

    def test_turning_clip_layer_is_explicit(self):
        matcher, _grid = _transactional_fixture(
            turning_clip_paths=frozenset({"terrain/motion.npz"})
        )

        self.assertEqual(matcher.turning_clip_paths, {"terrain/motion.npz"})

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


class ContinuationFirstMatcherTest(unittest.TestCase):
    def test_phase_gated_command_change_preempts_single_support_chunk(self):
        matcher, _grid = _transactional_fixture(
            contact_phase_gate=True,
            single_support=True,
            two_candidate_runway=False,
        )
        matcher.config = MatcherConfig(exclusion_frames=0)
        matcher.horizon_inventory.entry_frame[1] = 1
        matcher.reset()
        matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        with mock.patch(
            "mm_sonic.torch_terrain_skill_horizon_rollout.predict_horizon_targets",
            wraps=predict_horizon_targets,
        ) as predict:
            matcher.commit(matcher.prepare_step((0.0, 1.0), 0.0))

        self.assertEqual(len(matcher.chunk_events), 2)
        self.assertEqual(
            matcher.chunk_events[-1].release_reason, "command_change"
        )
        self.assertTrue(
            torch.equal(
                predict.call_args.kwargs["current_velocity_world_xy"],
                torch.tensor([0.0, 1.0]),
            )
        )

    def test_global_search_prefers_candidate_with_coherent_runway(self):
        matcher, _grid = _transactional_fixture(
            continuous_skill_enabled=True,
            two_candidate_runway=True,
        )
        matcher.reset()

        matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        self.assertEqual(matcher.chunk_events[0].skill_index, 1)
        self.assertEqual(matcher.chunk_events[0].selection_mode, "preferred")

    def test_global_search_falls_back_when_no_candidate_has_runway(self):
        matcher, _grid = _transactional_fixture(
            continuous_skill_enabled=True,
            two_candidate_runway=False,
        )
        matcher.reset()

        matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        self.assertEqual(matcher.chunk_events[0].skill_index, 0)
        self.assertEqual(
            matcher.chunk_events[0].selection_mode, "immediate-fallback"
        )

    def test_global_search_rejects_runway_that_diverges_from_command(self):
        matcher, _grid = _transactional_fixture(
            continuous_skill_enabled=True,
            two_candidate_runway=True,
            runway_suffix_direction=(0.0, 1.0),
        )
        matcher.reset()

        matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        self.assertEqual(matcher.chunk_events[0].skill_index, 0)
        self.assertEqual(
            matcher.chunk_events[0].selection_mode, "immediate-fallback"
        )

    def test_global_search_rejects_runway_that_creeps_under_moving_command(self):
        matcher, _grid = _transactional_fixture(
            continuous_skill_enabled=True,
            two_candidate_runway=True,
            runway_suffix_step_m=0.002,
        )
        matcher.reset()

        matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        self.assertEqual(matcher.chunk_events[0].skill_index, 0)
        self.assertEqual(
            matcher.chunk_events[0].selection_mode, "immediate-fallback"
        )

    def test_default_off_keeps_cheapest_immediate_candidate(self):
        matcher, _grid = _transactional_fixture(
            continuous_skill_enabled=False,
            two_candidate_runway=True,
        )
        matcher.reset()

        matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        self.assertEqual(matcher.chunk_events[0].skill_index, 0)
        self.assertEqual(matcher.chunk_events[0].selection_mode, "immediate")

    def test_same_nonzero_command_extends_before_global_search(self):
        matcher, grid = _transactional_fixture(continuous_skill_enabled=True)
        matcher.reset()
        initial_skill = None
        for _ in range(26):
            matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))
            initial_skill = matcher._skill_state.skill
        def fail_global_search(*_args, **_kwargs):
            self.fail("unchanged held command reached global search")

        matcher._try_start_skill = fail_global_search

        result = matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        self.assertEqual(result.diagnostics.selected_frame, 26)
        self.assertIs(matcher._skill_state.skill, initial_skill)
        self.assertEqual(len(matcher.chunk_events), 1)
        self.assertEqual(len(matcher.continuation_events), 1)
        event = matcher.continuation_events[0]
        self.assertEqual(event.start_frame, 26)
        self.assertEqual(event.endpoint_frame_exclusive, 51)
        self.assertEqual(event.command, ((1.0, 0.0), 0.0))

    def test_zero_command_does_not_extend_completed_skill(self):
        matcher, _grid = _transactional_fixture(continuous_skill_enabled=True)
        matcher.reset()
        for _ in range(26):
            matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))
        calls = []

        def no_replacement(*_args, **_kwargs):
            calls.append(True)
            return None

        matcher._try_start_skill = no_replacement

        result = matcher.commit(matcher.prepare_step((0.0, 0.0), 0.0))

        self.assertEqual(calls, [True])
        self.assertEqual(len(matcher.continuation_events), 0)
        self.assertEqual(result.diagnostics.selected_frame, 25)

    def test_changed_command_does_not_extend_completed_skill(self):
        matcher, _grid = _transactional_fixture(continuous_skill_enabled=True)
        matcher.reset()
        for _ in range(26):
            matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))
        calls = []

        def no_replacement(*_args, **_kwargs):
            calls.append(True)
            return None

        matcher._try_start_skill = no_replacement
        result = matcher.commit(matcher.prepare_step((0.0, 1.0), 0.5))

        self.assertEqual(calls, [True])
        self.assertEqual(len(matcher.continuation_events), 0)
        self.assertEqual(result.diagnostics.selected_frame, 25)

    def test_invalid_continuation_falls_back_and_holds_when_none_exists(self):
        matcher, _grid = _transactional_fixture(
            continuous_skill_enabled=True,
            contact_feasibility_enabled=True,
        )
        matcher.reset()
        for _ in range(26):
            matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))
        clip = matcher.dataset.folder.clips[0]
        clip.body_position_world[26:51, 1:, 2] = -0.02
        calls = []

        def no_replacement(*_args, **_kwargs):
            calls.append(True)
            return None

        matcher._try_start_skill = no_replacement
        result = matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        self.assertEqual(calls, [True])
        self.assertEqual(len(matcher.continuation_events), 0)
        self.assertEqual(result.diagnostics.selected_frame, 25)
        self.assertFalse(result.diagnostics.terrain_safety_override)

    def test_continuation_only_uses_relative_surface_not_raw_foot_height(self):
        matcher, _grid = _transactional_fixture(continuous_skill_enabled=True)
        matcher.reset()
        for _ in range(26):
            matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))
        matcher.dataset.folder.clips[0].body_position_world[26:51, 1:, 2] = -0.02

        result = matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        self.assertEqual(result.diagnostics.selected_frame, 26)
        self.assertEqual(len(matcher.continuation_events), 1)


if __name__ == "__main__":
    unittest.main()
