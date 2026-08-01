import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_motion_continuity import TransitionContinuityCosts
from mm_sonic.torch_motion_matcher import (
    EmittedWindowValidator,
    MatcherConfig,
    SearchDecision,
    SegmentCommitment,
    TransitionTerminalEvaluator,
    TorchMotionMatcher,
    _SelectionVisit,
    _loop_revisit_transition_costs,
    decay_spring_offsets,
    predict_command_trajectory,
    rank_exact_transition_candidates,
    select_exact_candidate,
)
from mm_sonic.torch_contact_segments import (
    ContactSegment,
    ContactSegmentIndex,
    SegmentPlacement,
    TerrainContactSegmentPolicy,
)
from mm_sonic.torch_transition_reachability import ReachabilityLimits
from mm_sonic.torch_motion_data import MotionFolder
from tests.python.torch_motion_test_utils import (
    build_varying_takara_arrays,
    write_takara_arrays,
)


class _MatcherExtension:
    name = "matcher_test"
    dimension = 2
    weight = 1.0

    def __init__(self):
        self.query_calls = 0

    def database_rows(self, folder, device):
        output = []
        for clip in folder.clips:
            frame = torch.arange(
                clip.valid_frame_stop, dtype=torch.float32, device=device
            )
            output.append(torch.stack((frame, frame.square() + frame), dim=1))
        return tuple(output)

    def query_row(self, state, trajectory):
        self.query_calls += 1
        value = state.root_position_world[0]
        return torch.stack((value, value.square() + value))


class _ScriptedWindowValidator:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.windows = []

    def __call__(self, window):
        self.windows.append(window.clone())
        return self.decisions.pop(0)


class _FakeFootholdPolicy:
    def __init__(
        self,
        allowed_source,
        added_cost,
        fallback_source=None,
        fallback_candidate_count=256,
        command_velocity_query_blend=1.0,
    ):
        self.allowed_source = tuple(allowed_source)
        self.added_cost = float(added_cost)
        self.fallback_source = (
            None if fallback_source is None else tuple(fallback_source)
        )
        self.fallback_candidate_count = fallback_candidate_count
        self.command_velocity_query_blend = command_velocity_query_blend
        self.calls = 0

    def prepare(self, state, shaped, database):
        self.calls += 1
        eligible = torch.zeros(
            database._search_clip_index.shape,
            dtype=torch.bool,
            device=database.device,
        )
        row = database.row_for_source(*self.allowed_source)
        eligible[row] = True
        fallback = None
        if self.fallback_source is not None:
            fallback = eligible.clone()
            fallback[database.row_for_source(*self.fallback_source)] = True
        return SimpleNamespace(
            row_eligibility=eligible,
            additional_row_cost=torch.full(
                eligible.shape,
                self.added_cost,
                dtype=torch.float32,
                device=database.device,
            ),
            fallback_row_eligibility=fallback,
        )


class _ConstantGrid:
    def __init__(self, height):
        self.height = float(height)

    def sample_xy(self, points):
        return torch.full(
            points.shape[:-1], self.height, dtype=points.dtype,
            device=points.device,
        )


class _IdentityAlignment:
    def __init__(self):
        self.translation_scene_xy = torch.zeros(2)
        self.yaw_scene_from_matcher = torch.zeros(())

    def matcher_to_scene_xy(self, points):
        return points


class _FlatContactFootKinematics:
    def __init__(self, height=0.035, right_height=None):
        self.height = float(height)
        self.right_height = (
            self.height if right_height is None else float(right_height)
        )

    def foot_positions(self, joints, roots, quaternions):
        output = np.zeros((len(joints), 2, 3), np.float64)
        output[:, 0, 2] = self.height
        output[:, 1, 2] = self.right_height
        return output


class _JointDependentContactFootKinematics:
    def foot_positions(self, joints, roots, quaternions):
        output = np.zeros((len(joints), 2, 3), np.float64)
        output[:, 0, :2] = joints[:, :2]
        output[:, 1, :2] = joints[:, 2:4]
        output[:, :, 2] = 0.035
        return output


def _install_contact_policy(
    matcher,
    *,
    start=20,
    end=30,
    foot_kinematics=None,
    terrain_clip_indices=(0,),
    flat_support_transition_cost_weight=0.0,
):
    support = torch.zeros(
        (len(matcher.folder.clips[0].joint_position), 2), dtype=torch.bool
    )
    support[start:end, 0] = True
    support[end : end + 10, 1] = True
    support[end + 10 : end + 20, 0] = True
    index = ContactSegmentIndex.from_support_masks(
        (support,), terrain_clip_indices=terrain_clip_indices,
        minimum_frames=5, maximum_frames=60,
    )
    identity = _IdentityAlignment()
    dataset = SimpleNamespace(
        folder=matcher.folder,
        clip_grids=(_ConstantGrid(0.0),),
        clip_alignments=(identity,),
        device=matcher.device,
    )
    extension = SimpleNamespace(
        dataset=dataset,
        query_grid=_ConstantGrid(0.0),
        alignment=identity,
    )
    policy = TerrainContactSegmentPolicy(
        index=index,
        extension=extension,
        foot_kinematics=(
            _FlatContactFootKinematics()
            if foot_kinematics is None
            else foot_kinematics
        ),
        flat_support_transition_cost_weight=(
            flat_support_transition_cost_weight
        ),
    )
    matcher._contact_segment_policy = policy
    matcher._transition_eligible_rows = policy.entry_eligibility(
        matcher.database
    )
    matcher._terrain_entry_rows = matcher._transition_eligible_rows.clone()
    return policy


class TorchMotionMatcherTests(unittest.TestCase):
    def test_requested_turn_warp_corrects_only_lateral_root_error(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher._foothold_action_policy = SimpleNamespace(
                turn_lateral_root_warp_gain=0.25,
                _requested_turn_active=True,
                _requested_heading_world_yaw=torch.tensor(0.0),
                _requested_turn_delta_rad=math.pi / 4.0,
            )
            matcher.reset()
            state = matcher._state
            shaped = SimpleNamespace(
                velocity_world_xy=torch.tensor((0.4, 0.0))
            )

            shift = matcher._requested_turn_lateral_warp_shift(
                state,
                shaped,
                torch.tensor((0.1, 0.1, 0.8)),
            )

        torch.testing.assert_close(shift, torch.tensor((0.0, -0.025)))

    def test_requested_turn_warp_uses_target_not_shaped_heading(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher._foothold_action_policy = SimpleNamespace(
                turn_lateral_root_warp_gain=0.25,
                _requested_turn_active=True,
                _requested_heading_world_yaw=torch.tensor(0.0),
                _requested_turn_delta_rad=math.pi / 4.0,
            )
            matcher.reset()
            state = matcher._state
            shaped = SimpleNamespace(
                velocity_world_xy=torch.tensor((0.0, 0.4))
            )

            shift = matcher._requested_turn_lateral_warp_shift(
                state,
                shaped,
                torch.tensor((0.1, 0.1, 0.8)),
            )

        torch.testing.assert_close(shift, torch.tensor((0.0, -0.025)))

    def test_large_requested_turn_warp_uses_shaped_direction(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher._foothold_action_policy = SimpleNamespace(
                turn_lateral_root_warp_gain=0.25,
                _requested_turn_active=True,
                _requested_heading_world_yaw=torch.tensor(0.0),
                _requested_turn_delta_rad=math.pi / 2.0,
            )
            matcher.reset()
            state = matcher._state
            shaped = SimpleNamespace(
                velocity_world_xy=torch.tensor((0.0, 0.4))
            )

            shift = matcher._requested_turn_lateral_warp_shift(
                state,
                shaped,
                torch.tensor((0.1, 0.1, 0.8)),
            )

        torch.testing.assert_close(shift, torch.tensor((-0.025, 0.0)))

    def test_reversal_turn_warp_caps_gain(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher._foothold_action_policy = SimpleNamespace(
                turn_lateral_root_warp_gain=0.606,
                reversal_lateral_root_warp_gain=0.25,
                _requested_turn_active=True,
                _requested_heading_world_yaw=torch.tensor(math.pi),
                _requested_turn_delta_rad=math.pi,
            )
            matcher.reset()
            state = matcher._state
            shaped = SimpleNamespace(
                velocity_world_xy=torch.tensor((0.0, 0.4))
            )

            shift = matcher._requested_turn_lateral_warp_shift(
                state,
                shaped,
                torch.tensor((0.1, 0.1, 0.8)),
            )

        torch.testing.assert_close(shift, torch.tensor((-0.025, 0.0)))

    def test_small_turn_warp_caps_gain(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher._foothold_action_policy = SimpleNamespace(
                turn_lateral_root_warp_gain=0.606,
                small_turn_lateral_root_warp_gain=0.25,
                _requested_turn_active=True,
                _requested_heading_world_yaw=torch.tensor(0.0),
                _requested_turn_delta_rad=math.pi / 4.0,
            )
            matcher.reset()
            state = matcher._state
            shaped = SimpleNamespace(
                velocity_world_xy=torch.tensor((0.0, 0.4))
            )

            shift = matcher._requested_turn_lateral_warp_shift(
                state,
                shaped,
                torch.tensor((0.1, 0.1, 0.8)),
            )

        torch.testing.assert_close(shift, torch.tensor((0.0, -0.025)))

    def test_reset_can_place_root_at_requested_world_xy(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")

            result = matcher.reset(root_position_world_xy=(1.25, -0.75))

        torch.testing.assert_close(
            result.root_position_world[:2], torch.tensor((1.25, -0.75))
        )
        torch.testing.assert_close(
            matcher._state.root_position[:2], torch.tensor((1.25, -0.75))
        )

    def test_two_contact_action_extends_validated_segment_window(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
            )
            contact = _install_contact_policy(
                matcher, start=20, end=30
            )
            matcher._foothold_action_policy = SimpleNamespace(
                index=SimpleNamespace(
                    entry=lambda clip, frame: (
                        SimpleNamespace(end_frame=47)
                        if (clip, frame) == (0, 20)
                        else None
                    )
                )
            )
            placement = SegmentPlacement(
                segment=ContactSegment(0, 20, 30, 0),
                vertical_offset_m=0.0,
                source_support_mask=contact.index.support_mask(0)[20:30],
            )

            extended = matcher._extend_foothold_placement(
                placement, maximum_chunk_frames=120
            )
            short = matcher._extend_foothold_placement(
                placement, maximum_chunk_frames=15
            )

        self.assertEqual(extended.segment.end_frame, 47)
        self.assertEqual(extended.source_support_mask.shape, (27, 2))
        self.assertEqual(short.segment.end_frame, 35)

    def test_heading_chunk_cap_can_shorten_a_long_base_segment(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            contact = _install_contact_policy(matcher, start=20, end=45)
            matcher._foothold_action_policy = SimpleNamespace(
                index=SimpleNamespace(
                    entry=lambda clip, frame: (
                        SimpleNamespace(end_frame=60)
                        if (clip, frame) == (0, 20)
                        else None
                    )
                )
            )
            placement = SegmentPlacement(
                segment=ContactSegment(0, 20, 45, 0),
                vertical_offset_m=0.0,
                source_support_mask=contact.index.support_mask(0)[20:45],
            )

            shortened = matcher._extend_foothold_placement(
                placement, maximum_chunk_frames=15
            )

        self.assertEqual(shortened.segment.end_frame, 35)
        self.assertEqual(shortened.source_support_mask.shape, (15, 2))

    def test_rejected_command_interrupt_stops_at_next_replanning_entry(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            contact = _install_contact_policy(matcher, start=20, end=30)
            matcher._foothold_action_policy = SimpleNamespace(
                index=SimpleNamespace(
                    entry=lambda clip, frame: (
                        SimpleNamespace(end_frame=60)
                        if (clip, frame) == (0, 20)
                        else None
                    ),
                    next_entry_frame=lambda clip, frame: (
                        40 if clip == 0 and frame < 40 else None
                    ),
                )
            )
            commitment = SegmentCommitment(
                0,
                20,
                60,
                0,
                0.0,
                command_direction_world_xy=(1.0, 0.0),
                command_heading_world_yaw=0.0,
            )

            shortened = matcher._shorten_interrupted_commitment(
                commitment,
                emitted_frame_index=30,
            )

        self.assertEqual(shortened.end_frame, 40)
        self.assertEqual(shortened.command_direction_world_xy, (1.0, 0.0))

    def test_two_contact_action_does_not_latch_across_command_divergence(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            contact = _install_contact_policy(matcher, start=20, end=30)
            matcher._foothold_action_policy = SimpleNamespace(
                index=SimpleNamespace(
                    entry=lambda clip, frame: (
                        SimpleNamespace(end_frame=47)
                        if (clip, frame) == (0, 20)
                        else None
                    )
                )
            )
            placement = SegmentPlacement(
                segment=ContactSegment(0, 20, 30, 0),
                vertical_offset_m=0.0,
                source_support_mask=contact.index.support_mask(0)[20:30],
            )
            state = matcher._state
            self.assertIsNone(state)
            matcher.reset()
            state = matcher._state
            sideways = predict_command_trajectory(
                state.root_position[:2],
                torch.tensor((0.0, 0.4)),
                state.shaped_heading,
                torch.tensor((0.0, 0.4)),
                torch.tensor(0.0),
                config=matcher.config,
            )

            extended = matcher._extend_foothold_placement(
                placement,
                maximum_chunk_frames=120,
                shaped=sideways,
                yaw_offset=torch.tensor(0.0),
            )

        self.assertEqual(extended.segment.end_frame, 30)

    def test_slow_pivot_keeps_validated_two_contact_action_chunk(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            contact = _install_contact_policy(matcher, start=20, end=30)
            matcher._foothold_action_policy = SimpleNamespace(
                index=SimpleNamespace(
                    entry=lambda clip, frame: (
                        SimpleNamespace(end_frame=47)
                        if (clip, frame) == (0, 20)
                        else None
                    )
                )
            )
            placement = SegmentPlacement(
                segment=ContactSegment(0, 20, 30, 0),
                vertical_offset_m=0.0,
                source_support_mask=contact.index.support_mask(0)[20:30],
            )
            matcher.reset()
            state = matcher._state
            pivot = predict_command_trajectory(
                state.root_position[:2],
                torch.tensor((0.4, 0.0)),
                state.shaped_heading,
                torch.tensor((0.0, 0.05)),
                torch.tensor(math.pi),
                config=matcher.config,
            )

            extended = matcher._extend_foothold_placement(
                placement,
                maximum_chunk_frames=120,
                shaped=pivot,
                yaw_offset=torch.tensor(0.0),
            )

        self.assertEqual(extended.segment.end_frame, 47)

    def test_foothold_query_can_use_shaped_command_velocity(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                foothold_action_policy=_FakeFootholdPolicy(
                    (0, 20), 0.0, command_velocity_query_blend=0.75
                ),
            )
        actual = torch.tensor((0.1, -0.2, 0.3))
        command = torch.tensor((0.5, 0.6))

        conditioned = matcher._query_root_velocity(actual, command)

        torch.testing.assert_close(
            conditioned, torch.tensor((0.4, 0.4, 0.3))
        )

    def test_layered_fallback_can_validate_deeper_than_realtime_window(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(transition_window_candidate_count=2),
                foothold_action_policy=_FakeFootholdPolicy(
                    (0, 20),
                    0.0,
                    fallback_source=(0, 30),
                    fallback_candidate_count=7,
                ),
            )

        self.assertEqual(
            matcher._terrain_rescue_candidate_limit(fallback=False), 2
        )
        self.assertEqual(
            matcher._terrain_rescue_candidate_limit(fallback=True), 7
        )

    def test_foothold_conditioning_preserves_relaxed_fallback_rows(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            policy = _FakeFootholdPolicy(
                (0, 20), 7.0, fallback_source=(0, 30)
            )
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                foothold_action_policy=policy,
            )
            matcher.reset()
            state = matcher._state
            shaped = predict_command_trajectory(
                state.root_position[:2],
                state.shaped_velocity,
                state.shaped_heading,
                torch.tensor((0.4, 0.0)),
                torch.tensor(0.0),
                config=matcher.config,
            )

            _, _, _, fallback = matcher._foothold_conditioning(
                state,
                shaped,
                torch.zeros(
                    matcher.database._search_clip_index.shape,
                    dtype=torch.float32,
                ),
            )

        self.assertEqual(
            torch.nonzero(fallback, as_tuple=False).flatten().tolist(),
            [
                matcher.database.row_for_source(0, 20),
                matcher.database.row_for_source(0, 30),
            ],
        )

    def test_foothold_conditioning_activates_fallback_when_hard_set_is_empty(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                foothold_action_policy=_FakeFootholdPolicy(
                    (0, 20), 7.0, fallback_source=(0, 30)
                ),
            )
            matcher.reset()
            matcher._transition_eligible_rows = torch.zeros(
                matcher.database.feature_shape[0], dtype=torch.bool
            )
            fallback_row = matcher.database.row_for_source(0, 30)
            matcher._transition_eligible_rows[fallback_row] = True
            state = matcher._state
            shaped = predict_command_trajectory(
                state.root_position[:2],
                state.shaped_velocity,
                state.shaped_heading,
                torch.tensor((0.4, 0.0)),
                torch.tensor(0.0),
                config=matcher.config,
            )

            conditioned, _, _, fallback = matcher._foothold_conditioning(
                state,
                shaped,
                torch.zeros(
                    matcher.database._search_clip_index.shape,
                    dtype=torch.float32,
                ),
            )

        self.assertEqual(
            torch.nonzero(conditioned, as_tuple=False).flatten().tolist(),
            [fallback_row],
        )
        torch.testing.assert_close(conditioned, fallback)

    def test_committed_multi_contact_action_ignores_nested_segment_entry(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(
                    search_interval_steps=1,
                    exclusion_frames=0,
                ),
            )
            matcher.reset()
            policy = _install_contact_policy(matcher, start=20, end=30)
            entry_row = matcher.database.row_for_source(0, 20)
            successor = matcher.database.row_for_source(0, 1)
            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=SearchDecision(
                    entry_row, successor, 10.0, 1.0, 1.0, True, True
                ),
            ):
                first = matcher.step((0.4, 0.0), 0.0)

            with mock.patch.object(
                ContactSegmentIndex,
                "entry",
                return_value=ContactSegment(0, 21, 26, 1),
            ) as nested, mock.patch.object(
                TerrainContactSegmentPolicy,
                "validate_emitted",
                side_effect=AssertionError("nested segment was revalidated"),
            ) as validate:
                second = matcher.step((0.4, 0.0), 0.0)

        self.assertTrue(first.diagnostics.segment_committed)
        self.assertTrue(second.diagnostics.segment_committed)
        self.assertEqual(second.diagnostics.selected_frame, 21)
        nested.assert_not_called()
        validate.assert_not_called()

    def test_expired_contact_commitment_is_not_carried_to_successor(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(
                    search_interval_steps=1,
                    exclusion_frames=0,
                ),
            )
            matcher.reset()
            matcher._state = replace(
                matcher._state,
                commitment=SegmentCommitment(
                    clip_index=0,
                    start_frame=0,
                    end_frame=1,
                    entering_foot=0,
                    vertical_offset_m=0.0,
                ),
            )

            result = matcher.step((0.4, 0.0), 0.0)

        self.assertFalse(result.diagnostics.segment_committed)
        self.assertIsNone(matcher._state.commitment)

    def test_foothold_policy_conditions_exact_search_rows_and_costs(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            policy = _FakeFootholdPolicy((0, 20), 7.0)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(
                    search_interval_steps=1,
                    exclusion_frames=0,
                ),
                foothold_action_policy=policy,
            )
            matcher.reset()

            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                wraps=select_exact_candidate,
            ) as search:
                matcher.step((0.4, 0.0), 0.0)

        self.assertEqual(policy.calls, 1)
        kwargs = search.call_args.kwargs
        allowed = matcher.database.row_for_source(0, 20)
        self.assertEqual(
            torch.nonzero(
                kwargs["transition_eligible_rows"], as_tuple=False
            ).flatten().tolist(),
            [allowed],
        )
        self.assertTrue(
            bool((kwargs["additional_transition_costs"] >= 7.0).all().item())
        )

    def test_terrain_candidate_yaw_follows_commanded_facing(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher.reset()
            policy = _install_contact_policy(matcher, start=20, end=30)
            policy.dataset.clip_alignments[0].yaw_scene_from_matcher.fill_(0.8)
            policy.extension.alignment.yaw_scene_from_matcher.fill_(-0.4)
            state = matcher._state
            shaped = predict_command_trajectory(
                state.root_position[:2],
                state.shaped_velocity,
                state.shaped_heading,
                torch.tensor((0.0, 0.4)),
                torch.tensor(1.1),
                has_valid_successor=True,
                config=matcher.config,
            )
            source_yaw = math.atan2(
                2.0
                * float(
                    matcher._clips[0].body_quaternion[20, 0, 0]
                    * matcher._clips[0].body_quaternion[20, 0, 3]
                    + matcher._clips[0].body_quaternion[20, 0, 1]
                    * matcher._clips[0].body_quaternion[20, 0, 2]
                ),
                1.0
                - 2.0
                * float(
                    matcher._clips[0].body_quaternion[20, 0, 2].square()
                    + matcher._clips[0].body_quaternion[20, 0, 3].square()
                ),
            )

            yaw = matcher._candidate_yaw_offset(0, 20, shaped)

        expected = math.atan2(
            math.sin(float(shaped.heading_world_yaw) - source_yaw),
            math.cos(float(shaped.heading_world_yaw) - source_yaw),
        )
        self.assertAlmostEqual(float(yaw), expected, places=6)
        self.assertNotAlmostEqual(
            float(yaw), float(policy.registered_yaw_offset(0)), places=3
        )

    def test_segment_command_compatibility_uses_whole_segment_travel(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher.reset()
            _install_contact_policy(matcher, start=20, end=30)
            root_body = matcher.folder.layout.root_body_index
            matcher._clips[0].body_linear_velocity[
                20, root_body, :2
            ] = torch.tensor((1.0, 0.0))
            matcher._clips[0].body_position[
                20:30, root_body, :2
            ] = torch.stack(
                (
                    torch.zeros(10),
                    torch.linspace(0.0, 1.0, 10),
                ),
                dim=1,
            )
            state = matcher._state
            shaped = predict_command_trajectory(
                state.root_position[:2],
                state.shaped_velocity,
                state.shaped_heading,
                torch.tensor((0.0, 0.4)),
                torch.tensor(0.0),
                has_valid_successor=True,
                config=matcher.config,
            )

            compatible = matcher._segment_command_compatible(
                0, 20, torch.tensor(0.0), shaped
            )

        self.assertTrue(compatible)

    def test_transition_eligibility_rotates_planar_source_travel(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher.reset()
            _install_contact_policy(matcher, start=20, end=30)
            directions = torch.zeros(
                (matcher.database.feature_shape[0], 2), dtype=torch.float32
            )
            entry = matcher.database.row_for_source(0, 20)
            directions[entry] = torch.tensor((0.0, 1.0))
            matcher._terrain_entry_direction_xy = directions
            state = matcher._state
            shaped = predict_command_trajectory(
                state.root_position[:2],
                state.shaped_velocity,
                state.shaped_heading,
                torch.tensor((0.0, 0.4)),
                torch.tensor(0.0),
                has_valid_successor=True,
                config=matcher.config,
            )

            eligible = matcher._command_transition_eligibility(state, shaped)

        self.assertTrue(bool(eligible[entry]))

    def test_terrain_entry_direction_is_enforced_after_command_alignment(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher.reset()
            _install_contact_policy(matcher, start=20, end=30)
            directions = torch.zeros(
                (matcher.database.feature_shape[0], 2), dtype=torch.float32
            )
            entry = matcher.database.row_for_source(0, 20)
            directions[entry] = torch.tensor((0.0, -1.0))
            matcher._terrain_entry_direction_xy = directions
            state = replace(
                matcher._state,
                root_linear_velocity=torch.tensor((0.0, 0.4, 0.0)),
            )
            shaped = predict_command_trajectory(
                state.root_position[:2],
                torch.tensor((0.0, 0.4)),
                state.shaped_heading,
                torch.tensor((0.0, 0.4)),
                torch.tensor(0.0),
                has_valid_successor=True,
                config=matcher.config,
            )

            eligible = matcher._command_transition_eligibility(state, shaped)

        self.assertFalse(bool(eligible[entry]))

    def test_backward_entry_filter_relaxes_after_command_alignment(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher.reset()
            _install_contact_policy(matcher, start=20, end=30)
            directions = torch.zeros(
                (matcher.database.feature_shape[0], 2), dtype=torch.float32
            )
            entry = matcher.database.row_for_source(0, 20)
            directions[entry] = torch.tensor((1.0, 0.0))
            matcher._terrain_entry_direction_xy = directions
            state = replace(
                matcher._state,
                root_linear_velocity=torch.tensor((-0.4, 0.0, 0.0)),
            )
            shaped = predict_command_trajectory(
                state.root_position[:2],
                torch.tensor((-0.4, 0.0)),
                state.shaped_heading,
                torch.tensor((-0.4, 0.0)),
                torch.tensor(0.0),
                has_valid_successor=True,
                config=matcher.config,
            )

            eligible = matcher._command_transition_eligibility(state, shaped)

        self.assertTrue(bool(eligible[entry]))

    def test_slow_pivot_does_not_gate_on_travel_direction(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher.reset()
            _install_contact_policy(matcher, start=20, end=30)
            directions = torch.zeros(
                (matcher.database.feature_shape[0], 2), dtype=torch.float32
            )
            entry = matcher.database.row_for_source(0, 20)
            directions[entry] = torch.tensor((1.0, 0.0))
            matcher._terrain_entry_direction_xy = directions
            state = replace(
                matcher._state,
                root_linear_velocity=torch.tensor((0.0, 0.07, 0.0)),
            )
            shaped = predict_command_trajectory(
                state.root_position[:2],
                torch.tensor((0.0, 0.07)),
                state.shaped_heading,
                torch.tensor((0.0, 0.07)),
                torch.tensor(math.pi),
                has_valid_successor=True,
                config=matcher.config,
            )

            eligible = matcher._command_transition_eligibility(state, shaped)

        self.assertTrue(bool(eligible[entry]))

    def test_flat_support_transition_cost_is_row_aligned(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher.reset()
            _install_contact_policy(
                matcher,
                terrain_clip_indices=(),
                flat_support_transition_cost_weight=400.0,
            )
            state = matcher._state
            shaped = predict_command_trajectory(
                state.root_position[:2],
                state.shaped_velocity,
                state.shaped_heading,
                torch.tensor((0.4, 0.0)),
                torch.tensor(0.0),
                has_valid_successor=True,
                config=matcher.config,
            )
            with mock.patch.object(
                TerrainContactSegmentPolicy,
                "query_support_mask",
                return_value=torch.tensor([False, True]),
            ):
                costs = matcher._flat_support_transition_costs(state, shaped)

        self.assertEqual(costs.shape, matcher.database._search_clip_index.shape)
        self.assertTrue(bool((costs >= 0.0).all().item()))
        self.assertTrue(bool((costs > 0.0).any().item()))

    def test_contact_segment_commitment_is_sequential_until_release(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(
                    search_interval_steps=1,
                    exclusion_frames=0,
                    joint_reference_smoothing_weight=0.2,
                ),
            )
            matcher.reset()
            _install_contact_policy(
                matcher,
                start=20,
                end=30,
                foot_kinematics=_JointDependentContactFootKinematics(),
            )
            entry_row = matcher.database.row_for_source(0, 20)
            successor = matcher.database.row_for_source(0, 1)
            selected = SearchDecision(
                entry_row, successor, 100.0, 1.0, 1.0, True, True
            )
            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=selected,
            ) as search:
                first = matcher.step((0.4, 0.0), 0.0)

            self.assertTrue(first.diagnostics.segment_committed)
            self.assertIsInstance(matcher._state.commitment, SegmentCommitment)
            self.assertEqual(first.diagnostics.segment_start_frame, 20)
            self.assertEqual(first.diagnostics.segment_end_frame, 30)
            fresh_feet = matcher._contact_segment_policy.emitted_foot_positions(
                first.dense_joint_position_window,
                first.dense_root_position_window,
                first.dense_root_orientation_window_wxyz,
            )
            np.testing.assert_allclose(
                first.dense_feature_body_position_window[:, 1:].numpy(),
                fresh_feet,
                rtol=0.0,
                atol=1e-7,
            )
            search.assert_called_once()

            frames = [first.diagnostics.selected_frame]
            searched = [first.diagnostics.searched]
            while frames[-1] + 1 < 30:
                value = matcher.step((0.4, 0.0), 0.0)
                frames.append(value.diagnostics.selected_frame)
                searched.append(value.diagnostics.searched)
                self.assertTrue(value.diagnostics.segment_committed)
                self.assertEqual(
                    value.diagnostics.force_search_reason,
                    "segment_commitment",
                )

            self.assertEqual(frames, list(range(20, 30)))
            self.assertEqual(searched, [True] + [False] * 9)
            self.assertIsNone(matcher._state.commitment)

            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                wraps=select_exact_candidate,
            ) as resumed_search:
                released = matcher.step((-0.4, 0.0), math.pi)
            self.assertFalse(released.diagnostics.segment_committed)
            self.assertIsNone(matcher._state.commitment)
            resumed_search.assert_called_once()
            released_feet = (
                matcher._contact_segment_policy.emitted_foot_positions(
                    released.dense_joint_position_window,
                    released.dense_root_position_window,
                    released.dense_root_orientation_window_wxyz,
                )
            )
            np.testing.assert_allclose(
                released.dense_feature_body_position_window[:, 1:].numpy(),
                released_feet,
                rtol=0.0,
                atol=1e-7,
            )

    def test_contact_commitment_can_play_past_feature_horizon(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=1, exclusion_frames=0),
            )
            matcher.reset()
            _install_contact_policy(matcher, start=50, end=80)
            entry_row = matcher.database.row_for_source(0, 50)
            successor = matcher.database.row_for_source(0, 1)
            self.assertIsNotNone(entry_row)
            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=SearchDecision(
                    entry_row, successor, 100.0, 1.0, 1.0, True, True
                ),
            ):
                first = matcher.step((0.4, 0.0), 0.0)

            frames = [first.diagnostics.selected_frame]
            with mock.patch.object(
                matcher,
                "_contact_commitment_should_interrupt",
                return_value=False,
            ):
                for _ in range(29):
                    value = matcher.step((0.4, 0.0), 0.0)
                    frames.append(value.diagnostics.selected_frame)
                    self.assertEqual(
                        tuple(value.dense_joint_position_window.shape),
                        (46, 29),
                    )

            self.assertEqual(frames, list(range(50, 80)))
            self.assertIsNone(matcher._state.commitment)
            self.assertIsNone(matcher.database.row_for_source(0, 79))

    def test_forced_reversal_can_replace_active_contact_commitment(self):
        arrays = build_varying_takara_arrays(frames=120)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=1, exclusion_frames=0),
            )
            matcher.reset()
            _install_contact_policy(matcher, start=20, end=30)
            root_body = matcher.folder.layout.root_body_index
            matcher._clips[0].body_position[
                30:40, root_body, 0
            ] = torch.linspace(0.0, -1.0, 10)
            second_entry = matcher.database.row_for_source(0, 30)
            matcher._terrain_entry_direction_xy = torch.zeros(
                (matcher.database.feature_shape[0], 2), dtype=torch.float32
            )
            matcher._terrain_entry_direction_xy[second_entry] = torch.tensor(
                (-1.0, 0.0)
            )
            first_entry = matcher.database.row_for_source(0, 20)
            first_successor = matcher.database.row_for_source(0, 1)
            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=SearchDecision(
                    first_entry,
                    first_successor,
                    10.0,
                    1.0,
                    1.0,
                    True,
                    True,
                ),
            ):
                first = matcher.step((0.4, 0.0), 0.0)
            self.assertEqual(first.diagnostics.segment_start_frame, 20)
            active_successor = matcher.database.row_for_source(0, 21)
            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=SearchDecision(
                    second_entry,
                    active_successor,
                    10.0,
                    1.0,
                    1.0,
                    True,
                    True,
                ),
            ):
                reversed_result = matcher.step((-0.4, 0.0), 0.0)

        self.assertTrue(reversed_result.diagnostics.transitioned)
        self.assertEqual(reversed_result.diagnostics.selected_frame, 30)
        self.assertEqual(reversed_result.diagnostics.segment_start_frame, 30)

    def test_active_commitment_rechecks_divergence_after_force_pulse(self):
        arrays = build_varying_takara_arrays(frames=120)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher.reset()
            _install_contact_policy(matcher, start=20, end=30)
            state = replace(
                matcher._state,
                commitment=SegmentCommitment(0, 20, 30, 0, 0.0),
                root_linear_velocity=torch.tensor((0.4, 0.0, 0.0)),
            )
            shaped = predict_command_trajectory(
                state.root_position[:2],
                torch.tensor((0.0, 0.4)),
                state.shaped_heading,
                torch.tensor((0.0, 0.4)),
                torch.tensor(0.0),
                has_valid_successor=True,
                config=matcher.config,
            )
            shaped = replace(shaped, force_search=False)

            interrupt = matcher._contact_commitment_should_interrupt(
                state, shaped
            )

        self.assertTrue(interrupt)

    def test_active_commitment_interrupts_for_heading_error_over_15_degrees(self):
        arrays = build_varying_takara_arrays(frames=120)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher.reset()
            _install_contact_policy(matcher, start=20, end=30)
            state = replace(
                matcher._state,
                commitment=SegmentCommitment(0, 20, 30, 0, 0.0),
                root_linear_velocity=torch.tensor((0.4, 0.0, 0.0)),
            )
            shaped = predict_command_trajectory(
                state.root_position[:2],
                torch.tensor((0.4, 0.0)),
                torch.tensor(0.0),
                torch.tensor((0.4, 0.0)),
                torch.tensor(math.radians(45.0)),
                has_valid_successor=True,
                config=matcher.config,
            )
            shaped = replace(shaped, force_search=False)

            interrupt = matcher._contact_commitment_should_interrupt(
                state, shaped
            )

        self.assertTrue(interrupt)

    def test_latched_action_chunk_interrupts_only_for_new_operator_command(self):
        arrays = build_varying_takara_arrays(frames=120)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher.reset()
            _install_contact_policy(matcher, start=20, end=40)
            state = replace(
                matcher._state,
                commitment=SegmentCommitment(
                    0,
                    20,
                    40,
                    0,
                    0.0,
                    command_direction_world_xy=(-1.0, 0.0),
                    command_heading_world_yaw=math.pi,
                ),
            )
            shaped = predict_command_trajectory(
                state.root_position[:2],
                state.shaped_velocity,
                state.shaped_heading,
                torch.tensor((-0.4, 0.0)),
                torch.tensor(math.pi),
                config=matcher.config,
            )

            same = matcher._contact_commitment_should_interrupt(
                state,
                shaped,
                torch.tensor((-0.4, 0.0)),
                torch.tensor(math.pi),
            )
            changed = matcher._contact_commitment_should_interrupt(
                state,
                shaped,
                torch.tensor((0.0, 0.4)),
                torch.tensor(math.pi / 2.0),
            )

        self.assertFalse(same)
        self.assertTrue(changed)

    def test_validated_contact_commitment_does_not_revalidate_past_its_end(self):
        arrays = build_varying_takara_arrays(frames=120)
        validator = _ScriptedWindowValidator([True])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=1, exclusion_frames=0),
                emitted_window_validator=validator,
            )
            matcher.reset()
            _install_contact_policy(matcher, start=20, end=30)
            entry = matcher.database.row_for_source(0, 20)
            successor = matcher.database.row_for_source(0, 1)
            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=SearchDecision(
                    entry, successor, 10.0, 1.0, 1.0, True, True
                ),
            ):
                first = matcher.step((0.4, 0.0), 0.0)

            second = matcher.step((0.4, 0.0), 0.0)

        self.assertTrue(first.diagnostics.segment_committed)
        self.assertTrue(second.diagnostics.segment_committed)
        self.assertEqual(second.diagnostics.selected_frame, 21)
        self.assertEqual(len(validator.windows), 1)

    def test_failed_full_segment_fk_validation_rejects_every_entry(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=1, exclusion_frames=0),
            )
            matcher.reset()
            _install_contact_policy(
                matcher,
                start=20,
                end=30,
                foot_kinematics=_FlatContactFootKinematics(height=0.20),
            )
            entry_row = matcher.database.row_for_source(0, 20)
            successor = matcher.database.row_for_source(0, 1)
            selected = SearchDecision(
                entry_row, successor, 0.0, 1.0, 1.0, True, True
            )
            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=selected,
            ):
                prepared = matcher.prepare_step((0.4, 0.0), 0.0)

        self.assertFalse(prepared.result.diagnostics.segment_committed)
        self.assertTrue(prepared.result.diagnostics.transition_rejected)
        self.assertEqual(
            prepared.result.diagnostics.segment_rejection_reason,
            "entering_support",
        )
        self.assertEqual(matcher._state.sequence, 0)
        self.assertIsNone(matcher._state.commitment)

    def test_valid_terrain_entry_overrides_flat_incumbent_during_search(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=1, exclusion_frames=0),
            )
            matcher.reset()
            _install_contact_policy(matcher, start=20, end=30)
            successor = matcher.database.row_for_source(0, 1)
            entry = matcher.database.row_for_source(0, 20)
            flat = SearchDecision(
                successor, successor, 0.0, 0.0, 0.0, True, False
            )
            terrain = SearchDecision(
                entry, None, math.inf, 2.0, 2.1, True, True
            )
            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=flat,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher.rank_exact_transition_candidates",
                return_value=(terrain,),
            ):
                result = matcher.step((0.4, 0.0), 0.0)

        self.assertEqual(result.diagnostics.selected_frame, 20)
        self.assertTrue(result.diagnostics.segment_committed)
        self.assertTrue(result.diagnostics.terrain_safety_override)

    def test_valid_terrain_entry_overrides_invalid_terrain_first_choice(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=1, exclusion_frames=0),
            )
            matcher.reset()
            policy = _install_contact_policy(matcher, start=20, end=30)
            successor = matcher.database.row_for_source(0, 1)
            invalid_entry = matcher.database.row_for_source(0, 20)
            valid_entry = matcher.database.row_for_source(0, 30)
            selected = SearchDecision(
                invalid_entry, successor, 0.5, 1.0, 1.0, True, True
            )
            ranked = (
                SearchDecision(
                    invalid_entry, None, math.inf, 1.0, 1.0, True, True
                ),
                SearchDecision(
                    valid_entry, None, math.inf, 2.0, 2.0, True, True
                ),
            )
            original_resolve = policy.resolve_entry

            def resolve_entry(_policy, **kwargs):
                if kwargs["frame_index"] == 20:
                    return None
                return original_resolve(**kwargs)

            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=selected,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher.rank_exact_transition_candidates",
                return_value=ranked,
            ), mock.patch.object(
                TerrainContactSegmentPolicy,
                "resolve_entry",
                new=resolve_entry,
            ):
                result = matcher.step((0.4, 0.0), 0.0)

        self.assertEqual(result.diagnostics.selected_frame, 30)
        self.assertTrue(result.diagnostics.segment_committed)
        self.assertTrue(result.diagnostics.terrain_safety_override)

    def test_rejected_contact_entry_is_transactional(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=1, exclusion_frames=0),
            )
            matcher.reset()
            _install_contact_policy(
                matcher,
                start=20,
                end=30,
                foot_kinematics=_FlatContactFootKinematics(
                    height=0.035, right_height=0.20
                ),
            )
            entry_row = matcher.database.row_for_source(0, 20)
            successor = matcher.database.row_for_source(0, 1)
            selected = SearchDecision(
                entry_row, successor, 0.0, 1.0, 1.0, True, True
            )
            state_before = matcher._state
            history_before = tuple(matcher._selection_history)
            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=selected,
            ), mock.patch.object(
                TerrainContactSegmentPolicy,
                "query_support_mask",
                return_value=torch.tensor([False, True]),
            ):
                prepared = matcher.prepare_step((0.4, 0.0), 0.0)

        self.assertIs(matcher._state, state_before)
        self.assertEqual(tuple(matcher._selection_history), history_before)
        self.assertFalse(prepared.result.diagnostics.segment_committed)
        self.assertTrue(prepared.result.diagnostics.transition_rejected)
        self.assertEqual(
            prepared.result.diagnostics.segment_rejection_reason,
            "support_side_switch",
        )

    def test_transition_eligibility_masks_only_search_transitions(self):
        arrays = build_varying_takara_arrays(frames=80)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            query = matcher.database._search_features[0].clone()
            rows = matcher.database.feature_shape[0]
            eligible = torch.zeros(rows, dtype=torch.bool)
            eligible[2] = True

            initial = select_exact_candidate(
                matcher.database,
                query,
                current_clip_index=0,
                current_frame_index=0,
                incumbent_row=None,
                search=True,
                config=MatcherConfig(exclusion_frames=0),
                transition_eligible_rows=eligible,
            )
            incumbent = select_exact_candidate(
                matcher.database,
                query,
                current_clip_index=0,
                current_frame_index=0,
                incumbent_row=1,
                search=True,
                config=MatcherConfig(exclusion_frames=0),
                transition_eligible_rows=torch.zeros_like(eligible),
            )
            ranked = rank_exact_transition_candidates(
                matcher.database,
                query,
                current_clip_index=0,
                current_frame_index=0,
                config=MatcherConfig(exclusion_frames=0),
                transition_eligible_rows=eligible,
            )

        self.assertEqual(initial.selected_row, 2)
        self.assertEqual(incumbent.selected_row, 1)
        self.assertEqual([value.selected_row for value in ranked], [2])

    def test_explicit_no_contact_policy_is_vanilla_bitwise(self):
        arrays = build_varying_takara_arrays(frames=180)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            default = TorchMotionMatcher.from_folder(root, device="cpu")
            explicit = TorchMotionMatcher.from_folder(
                root, device="cpu", contact_segment_policy=None
            )
            default.reset()
            explicit.reset()
            for step in range(100):
                command = (0.5, 0.0) if step < 50 else (-0.4, 0.2)
                heading = 0.0 if step < 50 else 2.7
                left = default.step(command, heading)
                right = explicit.step(command, heading)
                for field in (
                    "joint_position", "joint_velocity",
                    "root_position_world", "root_orientation_world_wxyz",
                    "dense_joint_position_window",
                    "dense_root_position_window",
                    "dense_feature_body_position_window",
                ):
                    self.assertTrue(
                        torch.equal(getattr(left, field), getattr(right, field)),
                        field,
                    )
                self.assertEqual(
                    replace(
                        left.diagnostics,
                        search_time_ns=None,
                        step_time_ns=0,
                    ),
                    replace(
                        right.diagnostics,
                        search_time_ns=None,
                        step_time_ns=0,
                    ),
                )

    def test_aligned_targets_apply_one_rigid_vertical_translation(self):
        arrays = build_varying_takara_arrays(frames=80)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            zero = matcher._aligned_targets(
                0, 0, torch.tensor(0.0), torch.zeros(2),
                translation_z=0.0, horizon=46,
            )
            raised = matcher._aligned_targets(
                0, 0, torch.tensor(0.0), torch.zeros(2),
                translation_z=0.4, horizon=46,
            )

        expected_root = torch.tensor([0, 0, 0.4]).expand(46, 3)
        expected_body = torch.tensor([0, 0, 0.4]).expand(46, 3, 3)
        torch.testing.assert_close(raised[2] - zero[2], expected_root)
        torch.testing.assert_close(
            raised[6] - zero[6], expected_body
        )

    def test_flat_transition_from_terrain_inherits_current_root_elevation(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "flat", arrays)
            write_takara_arrays(root / "terrain", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                reset_clip_path="terrain/motion.npz",
            )
            matcher.reset()
            state = matcher._state
            self.assertIsNotNone(state)
            raised_root = state.root_position.clone()
            raised_root[2] += 0.4
            state = replace(
                state,
                root_position=raised_root,
                translation_z=0.4,
            )
            matcher._contact_segment_policy = SimpleNamespace(
                index=SimpleNamespace(terrain_clip_indices=frozenset({1}))
            )
            shaped = predict_command_trajectory(
                state.root_position[:2],
                state.shaped_velocity,
                state.shaped_heading,
                torch.tensor((0.4, 0.0)),
                torch.tensor(0.0),
                has_valid_successor=True,
                config=matcher.config,
            )
            flat_row = matcher.database.row_for_source(0, 20)
            candidate = matcher._compose_candidate(
                state,
                shaped,
                flat_row,
                None,
                inherit_terrain_exit_elevation=True,
            )

        source_root_z = float(
            matcher._clips[0].body_position[
                20, matcher.folder.layout.root_body_index, 2
            ].item()
        )
        self.assertAlmostEqual(
            candidate.translation_z,
            float(state.root_position[2].item()) - source_root_z,
            places=6,
        )

    def test_transition_reachability_budget_caps_actual_compositions(self):
        arrays = build_varying_takara_arrays(frames=140)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(exclusion_frames=0),
                emitted_window_validator=lambda _window: True,
            )
            matcher.reset()
            ranked = tuple(
                SearchDecision(
                    matcher.database.row_for_source(0, frame),
                    None,
                    math.inf,
                    float(frame),
                    float(frame),
                    True,
                    True,
                )
                for frame in range(20, 28)
            )
            with mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                return_value=ranked,
            ), mock.patch.object(
                matcher,
                "_compose_candidate",
                wraps=matcher._compose_candidate,
            ) as compose:
                diagnostic = matcher.diagnose_transition_reachability(
                    (0.0, -0.5),
                    -math.pi / 2.0,
                    command_change_frame=120,
                    terminal_evaluator=lambda *_args: False,
                    safe_evaluator=lambda _window: True,
                    limits=ReachabilityLimits(
                        max_depth=2,
                        beam_width=8,
                        max_expanded_states=3,
                        max_source_advance_frames=15,
                    ),
                )

        self.assertFalse(diagnostic.reachable)
        self.assertTrue(diagnostic.budget_exhausted)
        self.assertEqual(diagnostic.expanded_state_count, 3)
        self.assertEqual(compose.call_count, 3)

    def test_transition_reachability_rejects_late_window_unsafety(self):
        arrays = build_varying_takara_arrays(frames=140)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(exclusion_frames=0),
                emitted_window_validator=lambda _window: True,
            )
            matcher.reset()
            row = matcher.database.row_for_source(0, 20)
            ranked = (
                SearchDecision(
                    row,
                    None,
                    math.inf,
                    1.0,
                    1.0,
                    True,
                    True,
                ),
            )
            compose = matcher._compose_candidate

            def late_unsafe_candidate(*args):
                candidate = compose(*args)
                body = candidate.dense_body_position.clone()
                body[15, 1:, 2] = -100.0
                return replace(candidate, dense_body_position=body)

            safety_windows = []

            def full_window_safe(window):
                safety_windows.append(window.clone())
                return bool((window[10:, 1:, 2] > -10.0).all().item())

            with mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                return_value=ranked,
            ), mock.patch.object(
                matcher,
                "_compose_candidate",
                side_effect=late_unsafe_candidate,
            ):
                diagnostic = matcher.diagnose_transition_reachability(
                    (0.0, -0.5),
                    -math.pi / 2.0,
                    command_change_frame=120,
                    terminal_evaluator=lambda *_args: True,
                    safe_evaluator=full_window_safe,
                    limits=ReachabilityLimits(
                        max_depth=1,
                        beam_width=1,
                        max_expanded_states=1,
                        max_source_advance_frames=15,
                    ),
                )

        self.assertFalse(diagnostic.reachable)
        self.assertEqual(diagnostic.expanded_state_count, 1)
        self.assertEqual(len(safety_windows), 1)
        self.assertEqual(tuple(safety_windows[0].shape), (46, 3, 3))

    def test_transition_reachability_diagnostic_is_two_hop_and_read_only(self):
        arrays = build_varying_takara_arrays(frames=140)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(
                    search_interval_steps=1,
                    exclusion_frames=0,
                ),
                emitted_window_validator=lambda _window: True,
            )
            control = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(
                    search_interval_steps=1,
                    exclusion_frames=0,
                ),
                emitted_window_validator=lambda _window: True,
            )
            matcher.reset()
            control.reset()
            bridge_row = matcher.database.row_for_source(0, 20)
            terminal_row = matcher.database.row_for_source(0, 40)
            ranked = (
                (
                    SearchDecision(
                        bridge_row,
                        None,
                        math.inf,
                        1.0,
                        1.1,
                        True,
                        True,
                    ),
                ),
                (
                    SearchDecision(
                        terminal_row,
                        None,
                        math.inf,
                        1.0,
                        1.1,
                        True,
                        True,
                    ),
                ),
            )
            terminal_calls = []
            safe_calls = []

            def terminal(
                _window,
                clip_index,
                frame_index,
                command_velocity_world_xy,
            ):
                terminal_calls.append(command_velocity_world_xy.clone())
                return clip_index == 0 and frame_index == 40

            def full_horizon_safe(window):
                safe_calls.append(window.clone())
                return True

            evaluator: TransitionTerminalEvaluator = terminal
            state_before = matcher._state
            with self.assertRaisesRegex(
                ContractError, "explicit full-window safe evaluator"
            ):
                matcher.diagnose_transition_reachability(
                    (0.0, -0.5),
                    -math.pi / 2.0,
                    command_change_frame=120,
                    terminal_evaluator=evaluator,
                    safe_evaluator=None,
                )
            with mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                side_effect=ranked,
            ) as ranking:
                diagnostic = matcher.diagnose_transition_reachability(
                    (0.0, -0.5),
                    -math.pi / 2.0,
                    command_change_frame=120,
                    terminal_evaluator=evaluator,
                    safe_evaluator=full_horizon_safe,
                    limits=ReachabilityLimits(
                        max_depth=2,
                        beam_width=1,
                        max_expanded_states=8,
                        max_source_advance_frames=1,
                    ),
                )

            self.assertTrue(diagnostic.reachable)
            self.assertEqual(diagnostic.depth_used, 2)
            self.assertEqual(
                (diagnostic.first_clip_index, diagnostic.first_frame_index),
                (0, 20),
            )
            self.assertEqual(
                (
                    diagnostic.terminal_clip_index,
                    diagnostic.terminal_frame_index,
                ),
                (0, 40),
            )
            self.assertIs(matcher._state, state_before)
            self.assertEqual(len(terminal_calls), 2)
            self.assertEqual(len(safe_calls), 2)
            self.assertEqual(
                ranking.call_args_list[1].kwargs["current_frame_index"],
                20,
            )
            for command in terminal_calls:
                torch.testing.assert_close(
                    command,
                    torch.tensor((0.0, -0.5), dtype=torch.float32),
                    rtol=0.0,
                    atol=0.0,
                )

            observed = matcher.step((0.0, -0.5), -math.pi / 2.0)
            expected = control.step((0.0, -0.5), -math.pi / 2.0)
            self.assertEqual(
                observed.diagnostics.selected_clip_path,
                expected.diagnostics.selected_clip_path,
            )
            self.assertEqual(
                observed.diagnostics.selected_frame,
                expected.diagnostics.selected_frame,
            )
            for name in (
                "dense_joint_position_window",
                "dense_joint_velocity_window",
                "dense_root_position_window",
                "dense_root_orientation_window_wxyz",
                "dense_feature_body_position_window",
                "dense_feature_body_velocity_window",
            ):
                torch.testing.assert_close(
                    getattr(observed, name),
                    getattr(expected, name),
                    rtol=0.0,
                    atol=0.0,
                )

    def test_transition_reachability_checks_intermediate_advance_times(self):
        arrays = build_varying_takara_arrays(frames=140)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(exclusion_frames=0),
                emitted_window_validator=lambda _window: True,
            )
            matcher.reset()
            bridge_row = matcher.database.row_for_source(0, 20)
            terminal_row = matcher.database.row_for_source(0, 40)
            dead_row = matcher.database.row_for_source(0, 60)
            ranked_frames = []

            def decision(row):
                return SearchDecision(
                    row,
                    None,
                    math.inf,
                    1.0,
                    1.0,
                    True,
                    True,
                )

            def rank_for_state(*_args, **kwargs):
                current = kwargs["current_frame_index"]
                ranked_frames.append(current)
                if current == 0:
                    return (decision(bridge_row),)
                return (
                    decision(
                        terminal_row if current == 24 else dead_row
                    ),
                )

            with mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                side_effect=rank_for_state,
            ):
                diagnostic = matcher.diagnose_transition_reachability(
                    (0.0, -0.5),
                    -math.pi / 2.0,
                    command_change_frame=120,
                    terminal_evaluator=lambda _window, _clip, frame, _cmd: (
                        frame == 40
                    ),
                    safe_evaluator=lambda _window: True,
                    limits=ReachabilityLimits(
                        max_depth=2,
                        beam_width=1,
                        max_expanded_states=64,
                        max_source_advance_frames=15,
                    ),
                )

        self.assertTrue(diagnostic.reachable)
        self.assertEqual(diagnostic.depth_used, 2)
        self.assertEqual(diagnostic.first_frame_index, 20)
        self.assertEqual(diagnostic.terminal_frame_index, 40)
        self.assertIn(24, ranked_frames)

    def test_transition_reachability_uses_and_propagates_loop_history(self):
        arrays = build_varying_takara_arrays(frames=140)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(exclusion_frames=0),
                emitted_window_validator=lambda _window: True,
            )
            matcher.reset()
            dead_row = matcher.database.row_for_source(0, 10)
            bridge_row = matcher.database.row_for_source(0, 20)
            terminal_row = matcher.database.row_for_source(0, 40)
            history_lengths = []

            def loop_costs(database, history, *_args, **_kwargs):
                history_lengths.append(len(history))
                costs = torch.zeros(
                    database.feature_shape[0], dtype=torch.float32
                )
                costs[dead_row] = 100.0
                return costs

            def decision(row):
                return SearchDecision(
                    row,
                    None,
                    math.inf,
                    1.0,
                    1.0,
                    True,
                    True,
                )

            def rank_with_loop_cost(*_args, **kwargs):
                costs = kwargs["additional_transition_costs"]
                if kwargs["current_frame_index"] == 0:
                    selected = (
                        bridge_row
                        if float(costs[dead_row]) == 100.0
                        else dead_row
                    )
                    return (decision(selected),)
                return (decision(terminal_row),)

            with mock.patch(
                "mm_sonic.torch_motion_matcher."
                "_loop_revisit_transition_costs",
                side_effect=loop_costs,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                side_effect=rank_with_loop_cost,
            ):
                diagnostic = matcher.diagnose_transition_reachability(
                    (0.0, -0.5),
                    -math.pi / 2.0,
                    command_change_frame=120,
                    terminal_evaluator=lambda _window, _clip, frame, _cmd: (
                        frame == 40
                    ),
                    safe_evaluator=lambda _window: True,
                    limits=ReachabilityLimits(
                        max_depth=2,
                        beam_width=1,
                        max_expanded_states=8,
                        max_source_advance_frames=1,
                    ),
                )

        self.assertTrue(diagnostic.reachable)
        self.assertEqual(diagnostic.first_frame_index, 20)
        self.assertEqual(history_lengths, [1, 2])

    def _jerk_rerank_fixture(self, root, *, weight):
        matcher = TorchMotionMatcher.from_folder(
            root,
            device="cpu",
            config=MatcherConfig(
                search_interval_steps=1,
                exclusion_frames=0,
                transition_window_jerk_weight=weight,
                transition_window_candidate_count=2,
            ),
            emitted_window_validator=lambda _window: True,
        )
        matcher.reset()
        state = matcher._state
        successor = matcher.database.row_for_source(
            state.clip_index, state.frame_index + 1
        )
        shaped = predict_command_trajectory(
            state.root_position[:2],
            state.shaped_velocity,
            state.shaped_heading,
            torch.tensor((0.5, 0.0), dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            has_valid_successor=successor is not None,
            config=matcher.config,
        )
        candidates = []
        stop = matcher.folder.clips[0].valid_frame_stop
        for frame in range(20, stop, 8):
            row = matcher.database.row_for_source(0, frame)
            if row is None or row == successor:
                continue
            candidate = matcher._compose_candidate(
                state, shaped, row, successor
            )
            jerk = torch.linalg.vector_norm(
                torch.diff(
                    candidate.dense_joint_position, n=3, dim=0
                )
                / (matcher.config.dt**3),
                dim=1,
            )
            candidates.append(
                (float(torch.quantile(jerk, 0.95).item()), row)
            )
        candidates.sort()
        self.assertGreater(candidates[-1][0], candidates[0][0])
        return matcher, successor, candidates[0], candidates[-1]

    def test_transition_window_reranker_selects_lower_predicted_jerk(self):
        arrays = build_varying_takara_arrays(frames=240)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher, successor, low, high = self._jerk_rerank_fixture(
                root, weight=100.0
            )
            selected = SearchDecision(
                high[1], successor, 1000.0, 1.0, 1.0, True, True
            )
            ranked = (
                SearchDecision(
                    high[1], None, math.inf, 1.0, 1.0, True, True
                ),
                SearchDecision(
                    low[1], None, math.inf, 1.0, 1.0, True, True
                ),
            )
            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=selected,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                return_value=ranked,
            ) as ranking:
                result = matcher.step((0.5, 0.0), 0.0)

        ranking.assert_called_once()
        self.assertEqual(
            result.diagnostics.selected_frame,
            matcher._source_for_row(low[1])[1],
        )
        self.assertTrue(result.diagnostics.transitioned)
        self.assertFalse(result.diagnostics.terrain_safety_override)

    def test_zero_window_jerk_weight_preserves_vanilla_without_ranking(self):
        arrays = build_varying_takara_arrays(frames=240)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher, successor, _low, high = self._jerk_rerank_fixture(
                root, weight=0.0
            )
            selected = SearchDecision(
                high[1], successor, 1000.0, 1.0, 1.0, True, True
            )
            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=selected,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates"
            ) as ranking:
                result = matcher.step((0.5, 0.0), 0.0)

        ranking.assert_not_called()
        self.assertEqual(
            result.diagnostics.selected_frame,
            matcher._source_for_row(high[1])[1],
        )

    def test_direct_matcher_rejects_invalid_window_rerank_config(self):
        arrays = build_varying_takara_arrays(frames=80)
        invalid = (
            (
                MatcherConfig(transition_window_jerk_weight=-1.0),
                "transition_window_jerk_weight",
            ),
            (
                MatcherConfig(transition_window_jerk_weight=math.inf),
                "transition_window_jerk_weight",
            ),
            (
                MatcherConfig(transition_window_jerk_weight=math.nan),
                "transition_window_jerk_weight",
            ),
            (
                MatcherConfig(transition_window_candidate_count=0),
                "transition_window_candidate_count",
            ),
            (
                MatcherConfig(transition_window_candidate_count=True),
                "transition_window_candidate_count",
            ),
            (
                MatcherConfig(transition_window_candidate_count=2.0),
                "transition_window_candidate_count",
            ),
            (
                MatcherConfig(transition_window_jerk_horizon_steps=3),
                "transition_window_jerk_horizon_steps",
            ),
            (
                MatcherConfig(transition_window_jerk_horizon_steps=47),
                "transition_window_jerk_horizon_steps",
            ),
            (
                MatcherConfig(transition_window_jerk_horizon_steps=True),
                "transition_window_jerk_horizon_steps",
            ),
            (
                MatcherConfig(transition_window_jerk_horizon_steps=8.0),
                "transition_window_jerk_horizon_steps",
            ),
            (
                MatcherConfig(joint_reference_smoothing_weight=-0.01),
                "joint_reference_smoothing_weight",
            ),
            (
                MatcherConfig(joint_reference_smoothing_weight=0.51),
                "joint_reference_smoothing_weight",
            ),
            (
                MatcherConfig(joint_reference_smoothing_weight=math.nan),
                "joint_reference_smoothing_weight",
            ),
            (
                MatcherConfig(joint_reference_smoothing_weight=True),
                "joint_reference_smoothing_weight",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            for config, label in invalid:
                with self.subTest(config=config):
                    with self.assertRaisesRegex(ContractError, label):
                        TorchMotionMatcher.from_folder(
                            root, device="cpu", config=config
                        )

    def test_joint_reference_smoothing_uses_previous_and_future_samples(self):
        arrays = build_varying_takara_arrays(frames=120)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            vanilla = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=1),
            )
            smoothed = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(
                    search_interval_steps=1,
                    joint_reference_smoothing_weight=0.25,
                ),
            )
            vanilla_reset = vanilla.reset()
            smoothed_reset = smoothed.reset()
            raw = vanilla.step((0.5, 0.0), 0.0)
            actual = smoothed.step((0.5, 0.0), 0.0)

        torch.testing.assert_close(
            smoothed_reset.joint_position, vanilla_reset.joint_position
        )
        expected_position = raw.dense_joint_position_window.clone()
        expected_position[0] = (
            0.25 * vanilla_reset.joint_position
            + 0.5 * raw.dense_joint_position_window[0]
            + 0.25 * raw.dense_joint_position_window[1]
        )
        expected_position[1:-1] = (
            0.25 * raw.dense_joint_position_window[:-2]
            + 0.5 * raw.dense_joint_position_window[1:-1]
            + 0.25 * raw.dense_joint_position_window[2:]
        )
        expected_velocity = raw.dense_joint_velocity_window.clone()
        expected_velocity[0] = (
            0.25 * vanilla_reset.joint_velocity
            + 0.5 * raw.dense_joint_velocity_window[0]
            + 0.25 * raw.dense_joint_velocity_window[1]
        )
        expected_velocity[1:-1] = (
            0.25 * raw.dense_joint_velocity_window[:-2]
            + 0.5 * raw.dense_joint_velocity_window[1:-1]
            + 0.25 * raw.dense_joint_velocity_window[2:]
        )
        torch.testing.assert_close(
            actual.dense_joint_position_window, expected_position
        )
        torch.testing.assert_close(
            actual.dense_joint_velocity_window, expected_velocity
        )

    def test_spring_matches_independent_equation(self):
        position = torch.tensor([0.4, -0.2], dtype=torch.float32)
        velocity = torch.tensor([-0.1, 0.3], dtype=torch.float32)
        time = torch.tensor(0.17, dtype=torch.float32)
        actual_p, actual_v = decay_spring_offsets(
            position, velocity, halflife_s=0.1, time_s=time
        )
        y = (4.0 * math.log(2.0) / (0.1 + 1.0e-5)) / 2.0
        j1 = velocity.numpy() + position.numpy() * y
        decay = math.exp(-y * 0.17)
        expected_p = decay * (position.numpy() + j1 * 0.17)
        expected_v = decay * (velocity.numpy() - j1 * y * 0.17)
        np.testing.assert_allclose(actual_p.numpy(), expected_p, atol=1e-6)
        np.testing.assert_allclose(actual_v.numpy(), expected_v, atol=1e-6)

    def test_reset_prepare_commit_and_dense_windows(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            reset = matcher.reset()
            self.assertEqual(reset.diagnostics.sequence, 0)
            self.assertEqual(reset.dense_joint_position_window.shape, (46, 29))
            self.assertEqual(
                reset.dense_feature_body_position_window.shape, (46, 3, 3)
            )
            self.assertEqual(
                reset.dense_feature_body_velocity_window.shape, (46, 3, 3)
            )
            np.testing.assert_array_equal(
                reset.joint_position_window.numpy(),
                reset.dense_joint_position_window[::5].numpy(),
            )
            prepared = matcher.prepare_step((0.5, 0.0), 0.0)
            self.assertEqual(reset.diagnostics.sequence, 0)
            result = matcher.commit(prepared)
            self.assertEqual(result.diagnostics.sequence, 1)
            with self.assertRaises(Exception):
                matcher.commit(prepared)
            for value in (
                result.dense_joint_position_window,
                result.dense_root_position_window,
                result.dense_root_orientation_window_wxyz,
                result.dense_feature_body_position_window,
                result.dense_feature_body_velocity_window,
            ):
                self.assertTrue(torch.isfinite(value).all())
            norms = torch.linalg.vector_norm(
                result.dense_root_orientation_window_wxyz, dim=-1
            )
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

            isolated = result.dense_feature_body_position_window.clone()
            isolated += 100.0
            next_result = matcher.step((0.5, 0.0), 0.0)
            self.assertFalse(
                torch.allclose(
                    isolated, next_result.dense_feature_body_position_window
                )
            )

    def test_constructor_rejects_mismatched_continuity_database(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            mismatches = (
                (
                    replace(
                        matcher._continuity,
                        device=torch.device("meta"),
                    ),
                    "same device",
                ),
                (
                    replace(
                        matcher._continuity,
                        _joint_position=(
                            matcher._continuity._joint_position[:-1]
                        ),
                        _joint_velocity=(
                            matcher._continuity._joint_velocity[:-1]
                        ),
                    ),
                    "row counts",
                ),
            )
            for continuity, message in mismatches:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ContractError, message):
                        TorchMotionMatcher(
                            matcher.folder,
                            matcher.database,
                            continuity,
                            matcher._clips,
                            matcher.config,
                        )

    def test_extension_query_and_split_costs_are_used_by_matcher(self):
        arrays = build_varying_takara_arrays(frames=100)
        extension = _MatcherExtension()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root, device="cpu", extension=extension
            )
            matcher.reset()
            result = matcher.step((0.5, 0.0), 0.0)

        self.assertEqual(extension.query_calls, 1)
        self.assertGreaterEqual(result.diagnostics.motion_feature_cost, 0.0)
        self.assertGreaterEqual(result.diagnostics.extension_feature_cost, 0.0)
        self.assertAlmostEqual(
            result.diagnostics.motion_feature_cost
            + result.diagnostics.extension_feature_cost,
            result.diagnostics.selected_feature_cost,
            places=4,
        )

    def test_runtime_passes_blend_age_penalty_after_accepted_transition(self):
        arrays = build_varying_takara_arrays(frames=120)
        config = MatcherConfig(
            search_interval_steps=1,
            transition_settle_duration_s=0.20,
            transition_settle_penalty=10.0,
        )
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=config,
            )
            reset = matcher.reset()
            target_frame = (
                reset.diagnostics.selected_frame + 25
            ) % matcher.folder.clips[0].valid_frame_stop
            target = matcher.database.row_for_source(0, target_frame)
            self.assertIsNotNone(target)

            def scripted(database, normalized_query, **kwargs):
                calls.append(dict(kwargs))
                successor = kwargs["incumbent_row"]
                if len(calls) == 1:
                    return SearchDecision(
                        target,
                        successor,
                        10.0,
                        1.0,
                        1.1,
                        True,
                        True,
                    )
                return SearchDecision(
                    successor,
                    successor,
                    1.0,
                    1.0,
                    1.0,
                    True,
                    False,
                )

            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                side_effect=scripted,
            ):
                transitioned = matcher.step((0.5, 0.0), 0.0)
                matcher.step((0.5, 0.0), 0.0)

        self.assertTrue(transitioned.diagnostics.transitioned)
        self.assertEqual(calls[0]["additional_transition_penalty"], 0.0)
        self.assertGreater(calls[1]["additional_transition_penalty"], 0.0)
        self.assertLess(
            calls[1]["additional_transition_penalty"],
            config.transition_settle_penalty,
        )

    def test_unsafe_incumbent_forces_nonlocal_terrain_rescue(self):
        arrays = build_varying_takara_arrays(frames=140)
        validator = _ScriptedWindowValidator([True, True, False, True])
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=5),
                emitted_window_validator=validator,
            )
            reset = matcher.reset()
            target_frame = (
                reset.diagnostics.selected_frame + 30
            ) % matcher.folder.clips[0].valid_frame_stop
            target = matcher.database.row_for_source(0, target_frame)
            self.assertIsNotNone(target)

            def scripted(database, normalized_query, **kwargs):
                calls.append(dict(kwargs))
                successor = kwargs["incumbent_row"]
                return SearchDecision(
                    successor,
                    successor,
                    2.0,
                    2.0,
                    2.0,
                    kwargs["search"],
                    False,
                )

            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                side_effect=scripted,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                return_value=(
                    SearchDecision(
                        target, None, math.inf, 1.0, 1.1, True, True
                    ),
                ),
            ) as ranking:
                first_result = matcher.step((0.5, 0.0), 0.0)
                second_result = matcher.step((0.5, 0.0), 0.0)
                result = matcher.step((0.5, 0.0), 0.0)

        self.assertFalse(first_result.diagnostics.transitioned)
        self.assertFalse(second_result.diagnostics.transitioned)
        self.assertEqual(len(calls), 3)
        self.assertFalse(calls[2]["search"])
        self.assertEqual(calls[2]["additional_transition_penalty"], 0.0)
        ranking.assert_called_once()
        self.assertTrue(result.diagnostics.transitioned)
        self.assertTrue(result.diagnostics.terrain_safety_override)
        self.assertEqual(result.diagnostics.terrain_safety_override_rank, 1)
        self.assertEqual(result.diagnostics.selected_frame, target_frame)
        self.assertEqual(len(validator.windows), 4)

    def test_real_selector_excludes_unsafe_incumbent_for_rescue(self):
        arrays = build_varying_takara_arrays(frames=160)
        validator = _ScriptedWindowValidator([True, True, False, True])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=5),
                emitted_window_validator=validator,
            )
            matcher.reset()
            matcher.step((0.5, 0.0), 0.0)
            before = matcher.step((0.5, 0.0), 0.0)
            rescued = matcher.step((0.5, 0.0), 0.0)

        self.assertEqual(before.diagnostics.sequence, 2)
        self.assertEqual(rescued.diagnostics.sequence, 3)
        self.assertTrue(rescued.diagnostics.searched)
        self.assertTrue(rescued.diagnostics.transitioned)
        self.assertTrue(rescued.diagnostics.terrain_safety_override)
        self.assertEqual(len(validator.windows), 4)

    def test_ranked_rescue_skips_unsafe_candidate_and_records_rank(self):
        arrays = build_varying_takara_arrays(frames=160)
        validator = _ScriptedWindowValidator(
            [True, True, False, False, True]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=5),
                emitted_window_validator=validator,
            )
            reset = matcher.reset()
            stop = matcher.folder.clips[0].valid_frame_stop
            rows = [
                matcher.database.row_for_source(
                    0,
                    (reset.diagnostics.selected_frame + offset) % stop,
                )
                for offset in (30, 60)
            ]
            self.assertTrue(all(row is not None for row in rows))
            ranked = tuple(
                SearchDecision(
                    row,
                    None,
                    math.inf,
                    float(index),
                    float(index) + 0.1,
                    True,
                    True,
                )
                for index, row in enumerate(rows, start=1)
            )

            with mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                return_value=ranked,
            ) as ranking:
                matcher.step((0.5, 0.0), 0.0)
                matcher.step((0.5, 0.0), 0.0)
                result = matcher.step((0.5, 0.0), 0.0)

        ranking.assert_called_once()
        self.assertTrue(result.diagnostics.transitioned)
        self.assertTrue(result.diagnostics.terrain_safety_override)
        self.assertEqual(result.diagnostics.terrain_safety_override_rank, 2)
        self.assertEqual(result.diagnostics.selected_frame, (
            reset.diagnostics.selected_frame + 60
        ) % stop)
        self.assertEqual(len(validator.windows), 5)

    def test_all_ranked_rescue_candidates_unsafe_fails_transactionally(self):
        arrays = build_varying_takara_arrays(frames=160)
        validator = _ScriptedWindowValidator(
            [True, True, False, False, False]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=5),
                emitted_window_validator=validator,
            )
            reset = matcher.reset()
            stop = matcher.folder.clips[0].valid_frame_stop
            rows = [
                matcher.database.row_for_source(
                    0,
                    (reset.diagnostics.selected_frame + offset) % stop,
                )
                for offset in (30, 60)
            ]
            ranked = tuple(
                SearchDecision(
                    row,
                    None,
                    math.inf,
                    float(index),
                    float(index) + 0.1,
                    True,
                    True,
                )
                for index, row in enumerate(rows, start=1)
            )

            with mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                return_value=ranked,
            ):
                matcher.step((0.5, 0.0), 0.0)
                matcher.step((0.5, 0.0), 0.0)
                with self.assertRaisesRegex(
                    ContractError, "no safe terrain rescue candidate"
                ):
                    matcher.prepare_step((0.5, 0.0), 0.0)

        self.assertEqual(matcher._state.sequence, 2)
        self.assertEqual(len(validator.windows), 5)

    def test_ranked_rescue_composes_only_configured_candidate_count(self):
        arrays = build_varying_takara_arrays(frames=160)
        validator = _ScriptedWindowValidator([True, True, False, False])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(
                    search_interval_steps=5,
                    transition_window_candidate_count=1,
                ),
                emitted_window_validator=validator,
            )
            reset = matcher.reset()
            stop = matcher.folder.clips[0].valid_frame_stop
            rows = [
                matcher.database.row_for_source(
                    0, (reset.diagnostics.selected_frame + offset) % stop
                )
                for offset in (30, 60, 90)
            ]
            ranked = tuple(
                SearchDecision(
                    row,
                    None,
                    math.inf,
                    float(index),
                    float(index) + 0.1,
                    True,
                    True,
                )
                for index, row in enumerate(rows, start=1)
            )

            with mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                return_value=ranked,
            ):
                matcher.step((0.5, 0.0), 0.0)
                matcher.step((0.5, 0.0), 0.0)
                with self.assertRaisesRegex(
                    ContractError, "no safe terrain rescue candidate"
                ):
                    matcher.prepare_step((0.5, 0.0), 0.0)

        self.assertEqual(matcher._state.sequence, 2)
        self.assertEqual(len(validator.windows), 4)

    def test_unsafe_transition_and_unsafe_incumbent_recover_through_ranked_rescue(
        self,
    ):
        arrays = build_varying_takara_arrays(frames=160)
        validator = _ScriptedWindowValidator(
            [True, True, False, False, False, True]
        )
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=5),
                emitted_window_validator=validator,
            )
            reset = matcher.reset()
            stop = matcher.folder.clips[0].valid_frame_stop
            chosen_frame = (reset.diagnostics.selected_frame + 20) % stop
            chosen = matcher.database.row_for_source(0, chosen_frame)
            rescue_rows = [
                matcher.database.row_for_source(
                    0, (reset.diagnostics.selected_frame + offset) % stop
                )
                for offset in (40, 80)
            ]
            self.assertIsNotNone(chosen)
            self.assertTrue(all(row is not None for row in rescue_rows))
            ranked = tuple(
                SearchDecision(
                    row,
                    None,
                    math.inf,
                    float(index),
                    float(index) + 0.1,
                    True,
                    True,
                )
                for index, row in enumerate(rescue_rows, start=1)
            )

            def scripted(database, normalized_query, **kwargs):
                calls.append(dict(kwargs))
                successor = kwargs["incumbent_row"]
                if len(calls) == 3:
                    return SearchDecision(
                        chosen, successor, 10.0, 1.0, 1.1, True, True
                    )
                return SearchDecision(
                    successor,
                    successor,
                    2.0,
                    2.0,
                    2.0,
                    kwargs["search"],
                    False,
                )

            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                side_effect=scripted,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                return_value=ranked,
            ) as ranking:
                first = matcher.step((0.5, 0.0), 0.0)
                second = matcher.step((0.5, 0.0), 0.0)
                result = matcher.step((0.5, 0.0), 0.0)
                sequence = matcher._state.sequence

        self.assertFalse(first.diagnostics.transitioned)
        self.assertFalse(second.diagnostics.transitioned)
        self.assertEqual(len(calls), 3)
        ranking.assert_called_once()
        self.assertTrue(result.diagnostics.searched)
        self.assertTrue(result.diagnostics.transitioned)
        self.assertTrue(result.diagnostics.terrain_safety_override)
        self.assertEqual(result.diagnostics.terrain_safety_override_rank, 2)
        self.assertEqual(
            result.diagnostics.selected_frame,
            (reset.diagnostics.selected_frame + 80) % stop,
        )
        self.assertEqual(sequence, 3)
        self.assertEqual(len(validator.windows), 6)

    def test_unsafe_transition_and_unsafe_incumbent_all_ranked_unsafe_fails_transactionally(
        self,
    ):
        arrays = build_varying_takara_arrays(frames=160)
        validator = _ScriptedWindowValidator(
            [True, True, False, False, False, False]
        )
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=5),
                emitted_window_validator=validator,
            )
            reset = matcher.reset()
            stop = matcher.folder.clips[0].valid_frame_stop
            chosen_frame = (reset.diagnostics.selected_frame + 20) % stop
            chosen = matcher.database.row_for_source(0, chosen_frame)
            rescue_rows = [
                matcher.database.row_for_source(
                    0, (reset.diagnostics.selected_frame + offset) % stop
                )
                for offset in (40, 80)
            ]
            self.assertIsNotNone(chosen)
            self.assertTrue(all(row is not None for row in rescue_rows))
            ranked = tuple(
                SearchDecision(
                    row,
                    None,
                    math.inf,
                    float(index),
                    float(index) + 0.1,
                    True,
                    True,
                )
                for index, row in enumerate(rescue_rows, start=1)
            )

            def scripted(database, normalized_query, **kwargs):
                calls.append(dict(kwargs))
                successor = kwargs["incumbent_row"]
                if len(calls) == 3:
                    return SearchDecision(
                        chosen, successor, 10.0, 1.0, 1.1, True, True
                    )
                return SearchDecision(
                    successor,
                    successor,
                    2.0,
                    2.0,
                    2.0,
                    kwargs["search"],
                    False,
                )

            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                side_effect=scripted,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                return_value=ranked,
            ):
                matcher.step((0.5, 0.0), 0.0)
                matcher.step((0.5, 0.0), 0.0)
                with self.assertRaisesRegex(
                    ContractError, "no safe terrain rescue candidate"
                ):
                    matcher.prepare_step((0.5, 0.0), 0.0)

        self.assertEqual(matcher._state.sequence, 2)
        self.assertEqual(len(validator.windows), 6)

    def test_ranked_rescue_uses_continuity_order_transactionally(self):
        arrays = build_varying_takara_arrays(frames=48)
        continuity = TransitionContinuityCosts(
            position=torch.tensor([0.0, 5.0, 0.1]),
            velocity=torch.tensor([0.0, 5.0, 0.1]),
            total=torch.tensor([0.0, 10.0, 0.2]),
        )

        def make_matcher(root, validator):
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(
                    search_interval_steps=1,
                    exclusion_frames=0,
                    transition_penalty=0.0,
                    transition_joint_position_weight=1.0,
                    transition_joint_velocity_weight=1.0,
                ),
                emitted_window_validator=validator,
            )
            matcher.reset()
            matcher.database._search_features.zero_()
            matcher.database._search_features[1, 0] = 1.0
            matcher.database._search_features[2, 0] = math.sqrt(2.0)
            return matcher

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            validator = _ScriptedWindowValidator([False, False, True])
            matcher = make_matcher(root, validator)
            mean, _ = matcher.database.normalization.parameters_copy()
            self.assertLess(
                torch.sum(torch.square(matcher.database._search_features[1])),
                torch.sum(torch.square(matcher.database._search_features[2])),
            )
            self.assertLess(
                torch.sum(torch.square(matcher.database._search_features[2]))
                + continuity.total[2],
                torch.sum(torch.square(matcher.database._search_features[1]))
                + continuity.total[1],
            )

            with mock.patch(
                "mm_sonic.torch_motion_matcher.extract_query_features",
                return_value=mean,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "TorchMotionDatabase.row_for_source",
                return_value=0,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "TransitionContinuityDatabase.costs",
                return_value=continuity,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                wraps=rank_exact_transition_candidates,
            ) as ranking, mock.patch.object(
                matcher,
                "_compose_candidate",
                wraps=matcher._compose_candidate,
            ) as compose:
                result = matcher.step((0.5, 0.0), 0.0)

            ranking.assert_called_once()
            self.assertIs(
                ranking.call_args.kwargs["additional_transition_costs"],
                continuity.total,
            )
            self.assertEqual(
                [call.args[2] for call in compose.call_args_list],
                [0, 2, 1],
            )
            self.assertEqual(result.diagnostics.selected_frame, 1)
            self.assertEqual(result.diagnostics.terrain_safety_override_rank, 2)
            self.assertEqual(
                result.diagnostics.selected_transition_position_cost, 5.0
            )
            self.assertEqual(
                result.diagnostics.selected_transition_velocity_cost, 5.0
            )
            self.assertEqual(
                result.diagnostics.selected_transition_continuity_cost, 10.0
            )

            all_unsafe = _ScriptedWindowValidator([False, False, False])
            failed = make_matcher(root, all_unsafe)
            failed_mean, _ = failed.database.normalization.parameters_copy()
            sequence = failed._state.sequence
            with mock.patch(
                "mm_sonic.torch_motion_matcher.extract_query_features",
                return_value=failed_mean,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "TorchMotionDatabase.row_for_source",
                return_value=0,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "TransitionContinuityDatabase.costs",
                return_value=continuity,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                wraps=rank_exact_transition_candidates,
            ) as failed_ranking:
                with self.assertRaisesRegex(
                    ContractError, "no safe terrain rescue candidate"
                ):
                    failed.prepare_step((0.5, 0.0), 0.0)

            self.assertIs(
                failed_ranking.call_args.kwargs[
                    "additional_transition_costs"
                ],
                continuity.total,
            )
            self.assertEqual(failed._state.sequence, sequence)

    def test_unsafe_transition_falls_back_to_safe_incumbent_transactionally(
        self,
    ):
        self.assertTrue(issubclass(EmittedWindowValidator, object))
        arrays = build_varying_takara_arrays(frames=120)
        validator = _ScriptedWindowValidator([False, True])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=1),
                emitted_window_validator=validator,
            )
            reset = matcher.reset()
            successor = matcher.database.row_for_source(
                0, reset.diagnostics.selected_frame + 1
            )
            self.assertIsNotNone(successor)
            target_frame = (
                reset.diagnostics.selected_frame + 25
            ) % matcher.folder.clips[0].valid_frame_stop
            if target_frame == reset.diagnostics.selected_frame + 1:
                target_frame = (
                    target_frame + 25
                ) % matcher.folder.clips[0].valid_frame_stop
            target = matcher.database.row_for_source(0, target_frame)
            self.assertIsNotNone(target)
            decision = SearchDecision(
                target,
                successor,
                10.0,
                1.0,
                1.1,
                True,
                True,
            )
            too_expensive = SearchDecision(
                target,
                None,
                math.inf,
                10.0,
                10.1,
                True,
                True,
            )
            state = matcher._state
            shaped = predict_command_trajectory(
                state.root_position[:2],
                state.shaped_velocity,
                state.shaped_heading,
                torch.tensor((0.5, 0.0)),
                torch.tensor(0.0),
                has_valid_successor=True,
                config=matcher.config,
            )
            periodic_shaped = replace(shaped, force_search=False)
            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=decision,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher.predict_command_trajectory",
                return_value=periodic_shaped,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                return_value=(too_expensive,),
            ) as ranking:
                result = matcher.step((0.5, 0.0), 0.0)

        ranking.assert_called_once()
        self.assertEqual(
            result.diagnostics.selected_frame,
            reset.diagnostics.selected_frame + 1,
        )
        self.assertFalse(result.diagnostics.transitioned)
        self.assertTrue(result.diagnostics.transition_rejected)
        self.assertTrue(result.diagnostics.searched)
        self.assertEqual(len(validator.windows), 2)
        self.assertEqual(validator.windows[0].shape, (46, 3, 3))
        validator.windows[0].add_(100.0)
        self.assertFalse(
            torch.allclose(
                validator.windows[0],
                result.dense_feature_body_position_window,
            )
        )

    def test_unsafe_transition_uses_next_ranked_safe_candidate_better_than_incumbent(
        self,
    ):
        arrays = build_varying_takara_arrays(frames=120)
        validator = _ScriptedWindowValidator([False, True, False, True])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=1),
                emitted_window_validator=validator,
            )
            reset = matcher.reset()
            successor = matcher.database.row_for_source(
                0, reset.diagnostics.selected_frame + 1
            )
            unsafe_row = matcher.database.row_for_source(0, 25)
            safe_row = matcher.database.row_for_source(0, 50)
            self.assertIsNotNone(successor)
            self.assertIsNotNone(unsafe_row)
            self.assertIsNotNone(safe_row)
            selected = SearchDecision(
                unsafe_row,
                successor,
                10.0,
                1.0,
                1.1,
                True,
                True,
            )
            ranked = (
                SearchDecision(
                    unsafe_row,
                    None,
                    math.inf,
                    1.0,
                    1.1,
                    True,
                    True,
                ),
                SearchDecision(
                    safe_row,
                    None,
                    math.inf,
                    2.0,
                    2.1,
                    True,
                    True,
                ),
            )
            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=selected,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                return_value=ranked,
            ) as ranking:
                result = matcher.step((0.5, 0.0), 0.0)

        ranking.assert_called_once()
        self.assertEqual(result.diagnostics.selected_frame, 50)
        self.assertTrue(result.diagnostics.transitioned)
        self.assertFalse(result.diagnostics.transition_rejected)
        self.assertTrue(result.diagnostics.terrain_safety_override)
        self.assertEqual(result.diagnostics.terrain_safety_override_rank, 2)
        self.assertEqual(result.diagnostics.selected_total_cost, 2.1)
        self.assertEqual(len(validator.windows), 4)

    def test_ranked_safe_candidate_must_beat_incumbent_after_settle_penalty(
        self,
    ):
        arrays = build_varying_takara_arrays(frames=120)
        validator = _ScriptedWindowValidator([False, True])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(search_interval_steps=1),
                emitted_window_validator=validator,
            )
            reset = matcher.reset()
            successor = matcher.database.row_for_source(
                0, reset.diagnostics.selected_frame + 1
            )
            target = matcher.database.row_for_source(0, 25)
            self.assertIsNotNone(successor)
            self.assertIsNotNone(target)
            selected = SearchDecision(
                target,
                successor,
                10.0,
                1.0,
                1.1,
                True,
                True,
            )
            ranked = SearchDecision(
                target,
                None,
                math.inf,
                9.0,
                9.5,
                True,
                True,
            )
            with mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=selected,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                return_value=(ranked,),
            ), mock.patch(
                "mm_sonic.torch_motion_matcher.active_transition_penalty",
                return_value=0.5,
            ):
                result = matcher.step((0.5, 0.0), 0.0)

        self.assertEqual(
            result.diagnostics.selected_frame,
            reset.diagnostics.selected_frame + 1,
        )
        self.assertFalse(result.diagnostics.transitioned)
        self.assertTrue(result.diagnostics.transition_rejected)
        self.assertEqual(len(validator.windows), 2)

    def test_emitted_window_validator_must_return_exact_bool(self):
        arrays = build_varying_takara_arrays(frames=120)

        def invalid_validator(_window):
            return torch.tensor(True)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                emitted_window_validator=invalid_validator,
            )
            matcher.reset()
            with self.assertRaisesRegex(ContractError, "exact bool"):
                matcher.prepare_step((0.5, 0.0), 0.0)

    def test_unsafe_transition_without_incumbent_uses_ranked_rescue(
        self,
    ):
        arrays = build_varying_takara_arrays(frames=120)
        validator = _ScriptedWindowValidator([False, False, True])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                emitted_window_validator=validator,
            )
            reset = matcher.reset()
            target = matcher.database.row_for_source(0, 20)
            rescue_rows = (
                matcher.database.row_for_source(0, 30),
                matcher.database.row_for_source(0, 40),
            )
            self.assertIsNotNone(target)
            self.assertTrue(all(row is not None for row in rescue_rows))
            decision = SearchDecision(
                target,
                None,
                math.inf,
                1.0,
                1.1,
                True,
                True,
            )
            ranked = tuple(
                SearchDecision(
                    row,
                    None,
                    math.inf,
                    float(index),
                    float(index) + 0.1,
                    True,
                    True,
                )
                for index, row in enumerate(rescue_rows, start=1)
            )
            with mock.patch(
                "mm_sonic.torch_motion_matcher."
                "TorchMotionDatabase.row_for_source",
                return_value=None,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher.select_exact_candidate",
                return_value=decision,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "rank_exact_transition_candidates",
                return_value=ranked,
            ):
                result = matcher.step((0.5, 0.0), 0.0)

        self.assertEqual(result.diagnostics.sequence, 1)
        self.assertEqual(result.diagnostics.selected_frame, 40)
        self.assertTrue(result.diagnostics.terrain_safety_override)
        self.assertEqual(result.diagnostics.terrain_safety_override_rank, 2)

    def test_low_progress_source_revisit_is_penalized_beyond_local_exclusion(
        self,
    ):
        arrays = build_varying_takara_arrays(frames=120)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(exclusion_frames=20),
            )
            matcher.reset()
            current_root = torch.tensor((1.0, -0.5))
            old = _SelectionVisit(0, 0, 40, current_root.clone())
            costs = _loop_revisit_transition_costs(
                matcher.database,
                (old,),
                current_root + torch.tensor((0.03, 0.0)),
                current_clip_index=0,
                current_sequence=22,
                config=matcher.config,
            )
            for frame in (32, 40, 48):
                row = matcher.database.row_for_source(0, frame)
                self.assertEqual(float(costs[row]), 1000.0)
            outside = matcher.database.row_for_source(0, 49)
            self.assertEqual(float(costs[outside]), 0.0)

            recent = _SelectionVisit(10, 0, 40, current_root.clone())
            recent_costs = _loop_revisit_transition_costs(
                matcher.database,
                (recent,),
                current_root,
                current_clip_index=0,
                current_sequence=22,
                config=matcher.config,
            )
            self.assertEqual(float(recent_costs.sum()), 0.0)

            far_costs = _loop_revisit_transition_costs(
                matcher.database,
                (old,),
                current_root + torch.tensor((0.08, 0.0)),
                current_clip_index=0,
                current_sequence=22,
                config=matcher.config,
            )
            self.assertEqual(float(far_costs.sum()), 0.0)

    def test_recent_low_progress_cross_clip_revisit_is_penalized(self):
        arrays = build_varying_takara_arrays(frames=120)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk-a", arrays)
            write_takara_arrays(root / "walk-b", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher.reset()
            current_root = torch.tensor((1.0, -0.5))
            recent_other_clip = _SelectionVisit(
                10, 1, 40, current_root.clone()
            )

            costs = _loop_revisit_transition_costs(
                matcher.database,
                (recent_other_clip,),
                current_root + torch.tensor((0.03, 0.0)),
                current_clip_index=0,
                current_sequence=22,
                config=matcher.config,
            )

            for frame in (32, 40, 48):
                row = matcher.database.row_for_source(1, frame)
                self.assertEqual(float(costs[row]), 25.0)
            same_source_row = matcher.database.row_for_source(0, 40)
            self.assertEqual(float(costs[same_source_row]), 0.0)

    def test_zero_continuity_weights_preserve_matcher_for_100_commands(self):
        arrays = build_varying_takara_arrays(frames=160)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            omitted = TorchMotionMatcher.from_folder(root, device="cpu")
            explicit = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(
                    transition_joint_position_weight=0.0,
                    transition_joint_velocity_weight=0.0,
                ),
            )
            left_reset = omitted.reset()
            right_reset = explicit.reset()
            for reset in (left_reset, right_reset):
                self.assertEqual(
                    reset.diagnostics.selected_transition_position_cost, 0.0
                )
                self.assertEqual(
                    reset.diagnostics.selected_transition_velocity_cost, 0.0
                )
                self.assertEqual(
                    reset.diagnostics.selected_transition_continuity_cost, 0.0
                )
            for step in range(100):
                velocity = (
                    0.4 * math.cos(0.03 * step),
                    0.3 * math.sin(0.03 * step),
                )
                heading = 0.01 * step
                left = omitted.step(velocity, heading)
                right = explicit.step(velocity, heading)
                self.assertEqual(
                    left.diagnostics.selected_clip_path,
                    right.diagnostics.selected_clip_path,
                )
                self.assertEqual(
                    left.diagnostics.selected_frame,
                    right.diagnostics.selected_frame,
                )
                self.assertEqual(
                    replace(
                        left.diagnostics, search_time_ns=None, step_time_ns=0
                    ),
                    replace(
                        right.diagnostics, search_time_ns=None, step_time_ns=0
                    ),
                )
                for diagnostics in (left.diagnostics, right.diagnostics):
                    self.assertEqual(
                        diagnostics.selected_transition_position_cost, 0.0
                    )
                    self.assertEqual(
                        diagnostics.selected_transition_velocity_cost, 0.0
                    )
                    self.assertEqual(
                        diagnostics.selected_transition_continuity_cost, 0.0
                    )
                for name in (
                    "dense_joint_position_window",
                    "dense_joint_velocity_window",
                    "dense_root_position_window",
                    "dense_root_orientation_window_wxyz",
                    "dense_feature_body_position_window",
                    "dense_feature_body_velocity_window",
                ):
                    torch.testing.assert_close(
                        getattr(left, name),
                        getattr(right, name),
                        rtol=0.0,
                        atol=0.0,
                    )

    def test_runtime_continuity_selects_smoother_candidate_and_reports_costs(
        self,
    ):
        arrays = build_varying_takara_arrays(frames=48)
        continuity = TransitionContinuityCosts(
            position=torch.tensor([0.0, 5.0, 0.1]),
            velocity=torch.tensor([0.0, 5.0, 0.1]),
            total=torch.tensor([0.0, 10.0, 0.2]),
        )
        zeros = TransitionContinuityCosts(
            position=torch.zeros(3),
            velocity=torch.zeros(3),
            total=torch.zeros(3),
        )
        config = MatcherConfig(
            search_interval_steps=1,
            exclusion_frames=0,
            transition_penalty=0.0,
            transition_joint_position_weight=1.0,
            transition_joint_velocity_weight=1.0,
        )

        def set_feature_costs(matcher, incumbent_cost=100.0):
            matcher.database._search_features.zero_()
            matcher.database._search_features[0, 0] = math.sqrt(
                incumbent_cost
            )
            matcher.database._search_features[1, 0] = 1.0
            matcher.database._search_features[2, 0] = math.sqrt(1.1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            ordinary = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=replace(
                    config,
                    transition_joint_position_weight=0.0,
                    transition_joint_velocity_weight=0.0,
                ),
            )
            ordinary.reset()
            set_feature_costs(ordinary)
            ordinary_mean, _ = (
                ordinary.database.normalization.parameters_copy()
            )
            with mock.patch(
                "mm_sonic.torch_motion_matcher.extract_query_features",
                return_value=ordinary_mean,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "TorchMotionDatabase.row_for_source",
                return_value=0,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "TransitionContinuityDatabase.costs",
                return_value=zeros,
            ):
                ordinary_result = ordinary.step((0.5, 0.0), 0.0)

            enabled = TorchMotionMatcher.from_folder(
                root, device="cpu", config=config
            )
            enabled.reset()
            set_feature_costs(enabled)
            enabled_mean, _ = enabled.database.normalization.parameters_copy()
            with mock.patch(
                "mm_sonic.torch_motion_matcher.extract_query_features",
                return_value=enabled_mean,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "TorchMotionDatabase.row_for_source",
                return_value=0,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "TransitionContinuityDatabase.costs",
                return_value=continuity,
            ) as costs:
                enabled_result = enabled.step((0.5, 0.0), 0.0)

            self.assertEqual(ordinary_result.diagnostics.selected_frame, 1)
            self.assertEqual(enabled_result.diagnostics.selected_frame, 2)
            costs.assert_called_once()
            self.assertEqual(
                costs.call_args.kwargs["position_weight"], 1.0
            )
            self.assertEqual(
                costs.call_args.kwargs["velocity_weight"], 1.0
            )
            self.assertAlmostEqual(
                enabled_result.diagnostics.selected_transition_position_cost,
                0.1,
                places=6,
            )
            self.assertAlmostEqual(
                enabled_result.diagnostics.selected_transition_velocity_cost,
                0.1,
                places=6,
            )
            self.assertAlmostEqual(
                enabled_result.diagnostics.selected_transition_continuity_cost,
                0.2,
                places=6,
            )
            self.assertAlmostEqual(
                enabled_result.diagnostics.selected_total_cost,
                enabled_result.diagnostics.selected_feature_cost
                + enabled_result.diagnostics.selected_transition_continuity_cost,
                places=6,
            )

            incumbent = TorchMotionMatcher.from_folder(
                root, device="cpu", config=config
            )
            incumbent.reset()
            incumbent.database._search_features.zero_()
            incumbent.database._search_features[2, 0] = 1.0
            incumbent_mean, _ = (
                incumbent.database.normalization.parameters_copy()
            )
            with mock.patch(
                "mm_sonic.torch_motion_matcher.extract_query_features",
                return_value=incumbent_mean,
            ), mock.patch(
                "mm_sonic.torch_motion_matcher."
                "TransitionContinuityDatabase.costs",
                return_value=continuity,
            ):
                incumbent_result = incumbent.step((0.5, 0.0), 0.0)

            self.assertFalse(incumbent_result.diagnostics.transitioned)
            self.assertEqual(incumbent_result.diagnostics.selected_frame, 1)
            self.assertEqual(
                incumbent_result.diagnostics.selected_transition_position_cost,
                0.0,
            )
            self.assertEqual(
                incumbent_result.diagnostics.selected_transition_velocity_cost,
                0.0,
            )
            self.assertEqual(
                incumbent_result.diagnostics.selected_transition_continuity_cost,
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
