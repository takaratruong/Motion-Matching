import math
import unittest
from unittest import mock
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import torch

from mm_sonic.torch_motion_features import FeatureNormalization, TorchMotionDatabase
from mm_sonic.torch_motion_matcher import MatcherConfig
from mm_sonic.torch_terrain_contact_feasibility import (
    TerrainContactFeasibilityResult,
)
from mm_sonic.torch_terrain_skill_horizon_rollout import (
    TerrainSkillHorizonMatcher,
    terrain_height_targets,
)
from mm_sonic.torch_terrain_skill_horizon_search import (
    HorizonSearchFailure,
    HorizonTargets,
    predict_horizon_targets,
    select_horizon_candidate,
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
        self.origin_xy = torch.tensor([-10.0, -10.0])
        self.cell_size_m = 1.0
        self.height_z = torch.zeros((21, 21))

    def sample_xy(self, points):
        self.calls += 1
        return torch.zeros(points.shape[:-1], device=points.device)


class _FootKinematics:
    def foot_positions(self, joint, root, quaternion):
        return np.repeat(root[:, None, :], 2, axis=1)


class _SoleKinematics:
    def sole_points(self, joint, root, quaternion):
        feet = _FootKinematics().foot_positions(joint, root, quaternion)
        return np.repeat(feet[:, :, None, :], 7, axis=2)


class _JointHeightFootKinematics:
    def foot_positions(self, joint, root, quaternion):
        joint = np.asarray(joint)
        feet = np.zeros((joint.shape[0], 2, 3), dtype=np.float64)
        feet[:, 0, 0] = -0.1
        feet[:, 1, 0] = 0.1
        feet[:, :, 2] = 0.045 + joint[:, :1]
        return feet


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
    emitted_contact_preview_enabled: bool = False,
    maximum_emitted_contact_candidates: int = 64,
    maximum_emitted_contact_rescue_candidates: int = 64,
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
    preview_kwargs = (
        {"emitted_contact_preview_enabled": True}
        if emitted_contact_preview_enabled
        else {}
    )
    matcher = TerrainSkillHorizonMatcher(
        base_matcher=_Base(),
        skill_inventory=skills,
        horizon_inventory=horizons,
        dataset=SimpleNamespace(folder=folder),
        query_terrain=SimpleNamespace(query_grid=grid, alignment=_Alignment()),
        foot_kinematics=_FootKinematics(),
        sole_kinematics=_SoleKinematics(),
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
        maximum_emitted_contact_candidates=(
            maximum_emitted_contact_candidates
        ),
        maximum_emitted_contact_rescue_candidates=(
            maximum_emitted_contact_rescue_candidates
        ),
        **preview_kwargs,
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
            current_foot_position_world=torch.tensor(
                [[0.9, 2.0, 0.0], [1.1, 2.0, 0.0]]
            ),
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

    def test_terrain_height_targets_resolve_unequal_future_foot_surfaces(self):
        targets = HorizonTargets(
            frames=torch.tensor([25]),
            displacement_local_xy=torch.tensor([[0.0, -0.15]]),
            yaw_delta_rad=torch.zeros(1),
            root_height_delta_m=torch.zeros(1),
            surface_height_delta_m=torch.zeros((1, 2)),
        )

        class _StepGrid:
            def sample_xy(self, points):
                return torch.where(
                    points[:, 1] >= -0.20,
                    torch.full_like(points[:, 1], 0.18),
                    torch.zeros_like(points[:, 1]),
                )

        enriched = terrain_height_targets(
            targets,
            current_root_position_world=torch.tensor([0.0, 0.0, 0.8]),
            current_root_yaw=torch.tensor(0.0),
            current_foot_position_world=torch.tensor(
                [[0.0, 0.08, 0.0], [0.0, -0.08, 0.0]]
            ),
            query_terrain=SimpleNamespace(
                query_grid=_StepGrid(), alignment=_Alignment()
            ),
        )

        self.assertTrue(
            torch.allclose(enriched.root_height_delta_m, torch.tensor([0.0]))
        )
        self.assertTrue(
            torch.allclose(
                enriched.surface_height_delta_m,
                torch.tensor([[0.0, -0.18]]),
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
    def test_exhausted_skill_has_no_sequential_continuation(self):
        matcher, _grid = _transactional_fixture(
            continuous_skill_enabled=True,
        )
        matcher.reset()
        matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))
        state = matcher._skill_state
        stop = state.skill.interval.playback_stop
        matcher._skill_state = replace(
            state,
            next_source_frame=stop,
            playback_stop=stop,
        )

        continuation = matcher._try_continue_skill(((1.0, 0.0), 0.0))

        self.assertIsNone(continuation)

    def test_exhausted_global_search_delays_switch_to_safe_source_endpoint(self):
        matcher, _grid = _transactional_fixture(
            continuous_skill_enabled=True,
        )
        matcher.reset()
        for _ in range(26):
            matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        with mock.patch.object(
            matcher, "_source_suffix_matches_command", return_value=False
        ), mock.patch(
            "mm_sonic.torch_terrain_skill_horizon_rollout.select_horizon_candidate",
            side_effect=HorizonSearchFailure({"terrain": 1}),
        ):
            result = matcher.commit(
                matcher.prepare_step((1.0, 0.0), 0.0)
            )

        self.assertEqual(result.diagnostics.selected_frame, 26)
        self.assertEqual(len(matcher.continuation_events), 1)

    def test_batched_terrain_prefilter_checks_full_supported_height_trace(self):
        matcher, _grid = _transactional_fixture(two_candidate_runway=False)
        first = matcher._surface_numpy[0].copy()
        second = matcher._surface_numpy[1].copy()
        second[5:] = 0.2
        matcher._surface_numpy = (first, second)
        result = matcher.reset()

        compatible = matcher._terrain_profile_prefilter(
            torch.tensor([0, 1], dtype=torch.long), result
        )

        self.assertEqual(compatible.tolist(), [True, False])

    def test_batched_terrain_prefilter_rejects_oriented_toe_on_riser(self):
        matcher, grid = _transactional_fixture(single_support=True)
        matcher.dataset.folder.clips[0].body_position_world[..., 0] = 0.0
        grid.sample_xy = lambda points: torch.where(
            points[..., 0] >= 0.10,
            torch.full_like(points[..., 0], 0.18),
            torch.zeros_like(points[..., 0]),
        )
        result = matcher.reset()

        compatible = matcher._terrain_profile_prefilter(
            torch.tensor([0], dtype=torch.long), result
        )

        self.assertEqual(compatible.tolist(), [False])

    def test_batched_terrain_prefilter_rejects_sole_collision_without_landing(
        self,
    ):
        matcher, grid = _transactional_fixture()
        matcher.dataset.folder.clips[0].body_position_world[:, 1:, 0] = 0.0
        grid.sample_xy = lambda points: torch.where(
            points[..., 0] >= 0.10,
            torch.full_like(points[..., 0], 0.18),
            torch.zeros_like(points[..., 0]),
        )
        result = matcher.reset()

        compatible = matcher._terrain_profile_prefilter(
            torch.tensor([0], dtype=torch.long), result
        )

        self.assertEqual(compatible.tolist(), [False])

    def test_emitted_preview_rejects_invalid_cheaper_candidate(self):
        matcher, _grid = _transactional_fixture(
            two_candidate_runway=False,
            continuous_skill_enabled=True,
            emitted_contact_preview_enabled=True,
        )
        folder = matcher.dataset.folder
        bad_clip = folder.clips[0]
        bad_clip.joint_position[:, 0] = 0.2
        good_clip = SimpleNamespace(
            **{
                **vars(bad_clip),
                "relative_path": "terrain/good-motion.npz",
                "joint_position": np.zeros_like(bad_clip.joint_position),
            }
        )
        folder.clips = (bad_clip, good_clip)
        second = TerrainSkill(
            **{
                **matcher.inventory.skills[1].__dict__,
                "clip_index": 1,
            }
        )
        matcher.inventory = TerrainSkillInventory(
            skills=(matcher.inventory.skills[0], second),
            rejected_by_reason={},
            row_to_skill={0: 0, 1: 1},
        )
        matcher.horizon_inventory.clip_index[1] = 1
        matcher.database._search_clip_index[1] = 1
        matcher.database._source_row_map[(1, 0)] = 1
        matcher.foot_kinematics = _JointHeightFootKinematics()
        matcher.reset()

        with mock.patch(
            "mm_sonic.torch_terrain_skill_horizon_rollout.select_horizon_candidate",
            wraps=select_horizon_candidate,
        ) as select:
            matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        self.assertEqual(matcher.chunk_events[0].skill_index, 1)
        self.assertEqual(
            select.call_args.kwargs["maximum_validated_candidates"], 64
        )
        self.assertTrue(callable(select.call_args.kwargs["ranked_prefilter"]))
        self.assertEqual(
            select.call_args.kwargs["maximum_preferred_cost_increase"], 1.0
        )
        self.assertEqual(
            select.call_args.kwargs[
                "maximum_preferred_outcome_cost_increase"
            ],
            0.05,
        )

    def test_emitted_preview_preserves_coherent_runway_preference(self):
        matcher, _grid = _transactional_fixture(
            continuous_skill_enabled=True,
            two_candidate_runway=True,
            emitted_contact_preview_enabled=True,
        )
        matcher.foot_kinematics = _JointHeightFootKinematics()
        matcher.reset()

        matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        self.assertEqual(matcher.chunk_events[0].skill_index, 1)
        self.assertEqual(matcher.chunk_events[0].selection_mode, "preferred")

    def test_emitted_preview_rescue_expands_only_after_primary_failure(self):
        matcher, _grid = _transactional_fixture(
            two_candidate_runway=False,
            continuous_skill_enabled=True,
            emitted_contact_preview_enabled=True,
            maximum_emitted_contact_candidates=64,
            maximum_emitted_contact_rescue_candidates=256,
        )
        matcher.foot_kinematics = _JointHeightFootKinematics()
        matcher.reset()
        calls = []

        def fail_primary_then_select(*args, **kwargs):
            calls.append(
                (
                    kwargs["maximum_validated_candidates"],
                    kwargs["preferred_validator"] is not None,
                )
            )
            if kwargs["maximum_validated_candidates"] == 64:
                raise HorizonSearchFailure({"terrain": 64})
            return select_horizon_candidate(*args, **kwargs)

        with mock.patch(
            "mm_sonic.torch_terrain_skill_horizon_rollout."
            "select_horizon_candidate",
            side_effect=fail_primary_then_select,
        ):
            matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        self.assertEqual(calls, [(64, True), (256, False)])
        self.assertEqual(matcher.chunk_events[0].selection_mode, "immediate")

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

    def test_failed_single_support_preemption_defers_to_safe_boundary(self):
        matcher, _grid = _transactional_fixture(
            contact_phase_gate=True,
            single_support=True,
            two_candidate_runway=False,
        )
        matcher.reset()
        matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))
        original_skill = matcher._skill_state.skill
        calls = []

        def fail_replacement(*_args, **_kwargs):
            calls.append(True)
            raise HorizonSearchFailure(
                {"terrain": 64, "shortlist": 100}
            )

        matcher._try_start_skill = fail_replacement

        prepared = matcher.prepare_step((0.0, 1.0), math.pi / 2.0)
        result = matcher.commit(prepared)

        self.assertEqual(calls, [True])
        self.assertEqual(len(matcher.chunk_events), 1)
        self.assertIs(matcher._skill_state.skill, original_skill)
        self.assertEqual(result.diagnostics.selected_frame, 1)
        self.assertTrue(matcher._replan_pending)
        self.assertEqual(matcher._last_command, ((0.0, 1.0), math.pi / 2.0))

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

    def test_active_chunk_is_not_treated_as_completed_continuation(self):
        matcher, _grid = _transactional_fixture(
            continuous_skill_enabled=True
        )
        matcher.reset()
        matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        continuation = matcher._try_continue_skill(
            ((0.0, 1.0), math.pi / 2.0),
            require_command_match=False,
        )

        self.assertIsNone(continuation)

    def test_failed_stop_search_holds_completed_safe_pose(self):
        matcher, _grid = _transactional_fixture(
            two_candidate_runway=False
        )
        matcher.reset()
        for _ in range(26):
            matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        with mock.patch(
            "mm_sonic.torch_terrain_skill_horizon_rollout.select_horizon_candidate",
            side_effect=HorizonSearchFailure(
                {"terrain": 64, "shortlist": 100}
            ),
        ) as select:
            result = matcher.commit(matcher.prepare_step((0.0, 0.0), 0.0))
            first_search_count = select.call_count
            second = matcher.commit(
                matcher.prepare_step((0.0, 0.0), 0.0)
            )

        self.assertEqual(result.diagnostics.selected_frame, 25)
        self.assertFalse(result.diagnostics.terrain_safety_override)
        self.assertGreater(first_search_count, 0)
        self.assertEqual(select.call_count, first_search_count)
        self.assertEqual(second.diagnostics.selected_frame, 25)

    def test_failed_turn_in_place_search_remains_a_failure(self):
        matcher, _grid = _transactional_fixture(
            two_candidate_runway=False
        )
        matcher.reset()
        for _ in range(26):
            matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))

        with mock.patch(
            "mm_sonic.torch_terrain_skill_horizon_rollout.select_horizon_candidate",
            side_effect=HorizonSearchFailure(
                {"terrain": 64, "shortlist": 100}
            ),
        ):
            with self.assertRaises(HorizonSearchFailure):
                matcher.prepare_step((0.0, 0.0), math.pi / 2.0)

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

    def test_emitted_preview_rejects_unsafe_continuation(self):
        matcher, _grid = _transactional_fixture(
            continuous_skill_enabled=True,
            emitted_contact_preview_enabled=True,
        )
        matcher.foot_kinematics = _JointHeightFootKinematics()
        matcher.reset()
        for _ in range(26):
            matcher.commit(matcher.prepare_step((1.0, 0.0), 0.0))
        matcher._try_start_skill = lambda *_args, **_kwargs: None
        rejection = TerrainContactFeasibilityResult(
            accepted=False,
            reason="sole-penetration",
            maximum_stance_error_m=0.0,
            landing_error_m=0.0,
            minimum_swing_clearance_m=0.0,
            minimum_sole_clearance_m=-0.05,
            maximum_footprint_height_range_m=0.0,
            maximum_height_deformation_m=0.0,
        )

        with mock.patch(
            "mm_sonic.torch_terrain_skill_horizon_rollout."
            "preview_continued_emitted_contact_trace",
            return_value=rejection,
        ) as preview:
            result = matcher.commit(
                matcher.prepare_step((1.0, 0.0), 0.0)
            )

        preview.assert_called_once()
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
