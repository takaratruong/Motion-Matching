from dataclasses import dataclass, replace
from enum import Enum
import math
from types import SimpleNamespace
import unittest

import torch

from mm_sonic.joints import ContractError

try:
    from mm_sonic.torch_foothold_actions import (
        FootholdAction,
        FootholdActionIndex,
        action_from_profiles,
    )
except ImportError:
    @dataclass(frozen=True)
    class FootholdAction:
        clip_index: int
        start_frame: int
        end_frame: int
        start_support: tuple[bool, bool]
        landing_feet: tuple[int, int]
        landing_frame_offsets: tuple[int, int]
        landing_xy_start_frame_m: torch.Tensor
        landing_height_delta_m: torch.Tensor
        root_displacement_m: torch.Tensor
        root_yaw_delta_rad: torch.Tensor
        minimum_swing_clearance_m: float
        maximum_unsupported_frames: int

        def __post_init__(self):
            raise AssertionError("foothold action descriptor is missing")

    def action_from_profiles(*_args, **_kwargs):
        raise AssertionError("foothold action extraction is missing")

    class FootholdActionIndex:
        @classmethod
        def from_dataset(cls, *_args, **_kwargs):
            raise AssertionError("foothold action index is missing")

try:
    from mm_sonic.torch_foothold_actions import (
        FootholdPlan,
        first_contact_eligibility,
        plan_footholds,
    )
except ImportError:
    FootholdPlan = None

    def first_contact_eligibility(*_args, **_kwargs):
        raise AssertionError("first-contact filter is missing")

    def plan_footholds(*_args, **_kwargs):
        raise AssertionError("foothold planner is missing")

try:
    from mm_sonic.torch_foothold_actions import (
        FootholdActionPolicy,
        FootholdRanking,
        FootholdSelectionArm,
        FootholdTransitionGraph,
        transition_graph_required,
        rank_foothold_actions,
        two_action_lookahead_cost,
        two_action_terrain_eligibility,
    )
except ImportError:
    FootholdActionPolicy = None
    FootholdTransitionGraph = None
    transition_graph_required = None

    class FootholdSelectionArm(Enum):
        TWO_CONTACT = "two-contact"
        HYBRID = "hybrid"
        LAYERED = "layered"
        CONTINUOUS = "continuous-control"

    def rank_foothold_actions(*_args, **_kwargs):
        raise AssertionError("foothold ranking is missing")


def _two_contact_profiles():
    support = torch.tensor(
        [
            [True, True],
            [True, False],
            [True, False],
            [True, True],
            [False, True],
            [True, True],
        ],
        dtype=torch.bool,
    )
    foot_xy = torch.tensor(
        [
            [[0.00, 0.00], [0.00, 0.20]],
            [[0.00, 0.00], [0.00, 0.20]],
            [[0.00, 0.00], [0.20, 0.20]],
            [[0.00, 0.00], [0.30, 0.20]],
            [[0.00, 0.00], [0.30, 0.20]],
            [[0.30, 0.00], [0.30, 0.20]],
        ],
        dtype=torch.float32,
    )
    surface = torch.tensor(
        [
            [0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.18],
            [0.00, 0.18],
            [0.18, 0.18],
        ],
        dtype=torch.float32,
    )
    root_xy = torch.tensor(
        [
            [1.00, 2.00],
            [1.00, 2.00],
            [1.05, 2.00],
            [1.15, 2.00],
            [1.20, 2.00],
            [1.30, 2.00],
        ],
        dtype=torch.float32,
    )
    root_yaw = torch.full((6,), torch.pi / 2, dtype=torch.float32)
    return support, foot_xy, surface, root_xy, root_yaw


def _action(
    *,
    height=(0.0, 0.0),
    feet=(1, 0),
    xy=None,
    start_frame=0,
    yaw=(0.0, 0.0),
    root=((0.25, 0.0), (0.50, 0.0)),
):
    landing_xy = (
        torch.tensor(((0.30, 0.10), (0.60, -0.10)))
        if xy is None
        else torch.tensor(xy, dtype=torch.float32)
    )
    return FootholdAction(
        clip_index=0,
        start_frame=start_frame,
        end_frame=start_frame + 21,
        start_support=(True, True),
        landing_feet=feet,
        landing_frame_offsets=(10, 20),
        landing_xy_start_frame_m=landing_xy,
        landing_height_delta_m=torch.tensor(height),
        root_displacement_m=torch.tensor(root, dtype=torch.float32),
        root_yaw_delta_rad=torch.tensor(yaw),
        minimum_swing_clearance_m=0.04,
        maximum_unsupported_frames=0,
    )


def _plan(*, height=(0.0, 0.0), feet=(1, 0), xy=None):
    landing_xy = (
        torch.tensor((((0.30, 0.10), (0.60, -0.10)),))
        if xy is None
        else torch.tensor((xy,), dtype=torch.float32)
    )
    return FootholdPlan(
        landing_feet=torch.tensor((feet,), dtype=torch.int64),
        landing_xy_world_m=landing_xy.clone(),
        landing_xy_command_frame_m=landing_xy,
        landing_height_delta_m=torch.tensor((height,)),
        landing_frame_offsets=torch.tensor(((10, 20),), dtype=torch.int64),
        score=torch.zeros(1),
    )


class FootholdActionTests(unittest.TestCase):
    def test_action_index_reports_next_replanning_entry(self):
        first = _action(start_frame=10)
        second = _action(start_frame=30)
        index = FootholdActionIndex(
            actions=(first, second),
            _entries={(0, 10): first, (0, 30): second},
        )

        self.assertEqual(index.next_entry_frame(0, 10), 30)
        self.assertIsNone(index.next_entry_frame(0, 30))

    def test_turn_episode_does_not_bias_approach_but_tracks_residual(self):
        class FlatGrid:
            @staticmethod
            def sample_xy(points):
                return torch.zeros(points.shape[:-1], dtype=points.dtype)

        policy = FootholdActionPolicy(
            index=FootholdActionIndex(actions=(), _entries={}),
            extension=SimpleNamespace(
                query_grid=FlatGrid(),
                alignment=SimpleNamespace(
                    matcher_to_scene_xy=lambda points: points
                ),
            ),
            arm=FootholdSelectionArm.LAYERED_HYBRID,
        )
        identity = torch.tensor((1.0, 0.0, 0.0, 0.0))

        self.assertEqual(
            policy._turn_episode_query(
                sequence=10,
                root_quaternion=identity,
                future_facing=torch.tensor((1.0, 0.0)),
            ),
            (None, False),
        )
        self.assertAlmostEqual(
            policy._heading_maintenance_query(
                root_quaternion=identity,
                future_facing=torch.tensor((1.0, 0.0)),
            ),
            0.0,
        )
        self.assertTrue(
            policy._elevated_heading_maintenance_required(
                surface=torch.tensor((0.36, 0.53)),
                base_height=torch.tensor(0.0),
            )
        )
        self.assertFalse(
            policy._elevated_heading_maintenance_required(
                surface=torch.tensor((0.0, 0.19)),
                base_height=torch.tensor(0.0),
            )
        )
        self.assertFalse(
            policy._heading_maintenance_activated(
                sequence=0,
                surface=torch.tensor((0.0, 0.0)),
                base_height=torch.tensor(0.0),
                future_facing=torch.tensor((1.0, 0.0)),
            )
        )
        self.assertFalse(
            policy._requested_turn_sequence_query(
                sequence=0,
                requested_heading_world_yaw=torch.tensor(0.0),
            )
        )
        self.assertFalse(
            policy._requested_turn_sequence_query(
                sequence=10,
                requested_heading_world_yaw=torch.tensor(math.pi / 2.0),
            )
        )
        self.assertTrue(policy._requested_turn_active)
        policy._requested_heading_world_yaw = torch.tensor(0.0)
        self.assertTrue(
            policy._requested_turn_sequence_query(
                sequence=20,
                requested_heading_world_yaw=torch.tensor(math.pi),
            )
        )
        activated = False
        for sequence in range(224, 234):
            activated = policy._heading_maintenance_activated(
                sequence=sequence,
                surface=torch.tensor((0.36, 0.53)),
                base_height=torch.tensor(0.0),
                future_facing=torch.tensor((1.0, 0.0)),
            )
        self.assertTrue(activated)
        self.assertFalse(
            policy._heading_maintenance_activated(
                sequence=234,
                surface=torch.tensor((0.36, 0.53)),
                base_height=torch.tensor(0.0),
                future_facing=torch.tensor((0.0, 1.0)),
            )
        )
        self.assertAlmostEqual(
            policy._action_turn_target(math.pi), math.pi / 2.0
        )
        self.assertAlmostEqual(
            policy._action_turn_target(-math.pi), -math.pi / 2.0
        )
        desired, hard = policy._turn_episode_query(
            sequence=20,
            root_quaternion=identity,
            future_facing=torch.tensor((0.0, 1.0)),
        )
        self.assertAlmostEqual(desired, math.pi / 2.0)
        self.assertTrue(hard)
        root_yaw = 1.40
        residual, hard = policy._turn_episode_query(
            sequence=30,
            root_quaternion=torch.tensor(
                (math.cos(root_yaw / 2.0), 0.0, 0.0,
                 math.sin(root_yaw / 2.0))
            ),
            future_facing=torch.tensor((0.0, 1.0)),
        )
        self.assertAlmostEqual(residual, math.pi / 2.0 - root_yaw, places=5)
        self.assertFalse(hard)
        self.assertEqual(
            policy._turn_episode_query(
                sequence=0,
                root_quaternion=identity,
                future_facing=torch.tensor((1.0, 0.0)),
            ),
            (None, False),
        )

    def test_large_turn_hard_gates_terminal_facing(self):
        straight_low_cost = _action(height=(0.18, 0.18))
        left_turn_high_cost = _action(
            height=(0.18, 0.18),
            yaw=(math.pi / 4.0, math.pi / 2.0),
        )

        ranking = rank_foothold_actions(
            arm=FootholdSelectionArm.LAYERED_HYBRID,
            plan=_plan(height=(0.18, 0.18)),
            actions=(straight_low_cost, left_turn_high_cost),
            motion_cost=torch.tensor((0.0, 100.0)),
            height_tolerance_m=0.04,
            xy_tolerance_m=0.20,
            timing_tolerance_frames=8,
            desired_yaw_delta_rad=math.pi / 2.0,
            yaw_tolerance_rad=math.pi / 6.0,
        )

        self.assertEqual(ranking.eligible.tolist(), [False, True])
        self.assertEqual(ranking.selected_action, 1)

    def test_heading_maintenance_is_soft_and_cannot_exhaust_coverage(self):
        large_yaw_low_motion_cost = _action(
            height=(0.18, 0.18),
            yaw=(0.8, math.pi / 2.0),
        )
        straight_high_motion_cost = _action(height=(0.18, 0.18))

        ranking = rank_foothold_actions(
            arm=FootholdSelectionArm.LAYERED_HYBRID,
            plan=_plan(height=(0.18, 0.18)),
            actions=(large_yaw_low_motion_cost, straight_high_motion_cost),
            motion_cost=torch.tensor((0.0, 1.0)),
            height_tolerance_m=0.04,
            xy_tolerance_m=0.20,
            timing_tolerance_frames=8,
            desired_yaw_delta_rad=0.0,
            hard_yaw_gate=False,
            yaw_cost_weight=100.0,
            turn_xy_cost_weight=0.0,
        )

        self.assertEqual(ranking.eligible.tolist(), [True, True])
        self.assertEqual(ranking.selected_action, 1)

    def test_turn_cost_penalizes_terminal_root_lateral_drift(self):
        drifting_low_motion_cost = _action(
            height=(0.18, 0.18),
            yaw=(math.pi / 4.0, math.pi / 2.0),
            root=((0.15, 0.0), (0.30, 0.0)),
        )
        aligned_high_motion_cost = _action(
            height=(0.18, 0.18),
            yaw=(math.pi / 4.0, math.pi / 2.0),
            root=((0.0, 0.15), (0.0, 0.30)),
        )

        ranking = rank_foothold_actions(
            arm=FootholdSelectionArm.LAYERED_HYBRID,
            plan=_plan(height=(0.18, 0.18)),
            actions=(drifting_low_motion_cost, aligned_high_motion_cost),
            motion_cost=torch.tensor((0.0, 1.0)),
            height_tolerance_m=0.04,
            xy_tolerance_m=0.20,
            timing_tolerance_frames=8,
            desired_yaw_delta_rad=math.pi / 2.0,
            yaw_tolerance_rad=math.pi / 6.0,
            yaw_cost_weight=0.0,
            turn_xy_cost_weight=0.0,
            turn_root_lateral_cost_weight=100.0,
        )

        self.assertEqual(ranking.eligible.tolist(), [True, True])
        self.assertEqual(ranking.selected_action, 1)

    def test_turn_cost_rewards_commanded_root_progress_by_landing_time(self):
        spinning_low_motion_cost = _action(
            height=(0.18, 0.18),
            yaw=(math.pi / 2.0, math.pi),
            root=((0.0, 0.0), (0.0, 0.0)),
        )
        progressing_high_motion_cost = _action(
            height=(0.18, 0.18),
            yaw=(math.pi / 2.0, math.pi),
            root=((-0.08, 0.0), (-0.15, 0.0)),
        )

        ranking = rank_foothold_actions(
            arm=FootholdSelectionArm.LAYERED_HYBRID,
            plan=_plan(height=(0.18, 0.18)),
            actions=(spinning_low_motion_cost, progressing_high_motion_cost),
            motion_cost=torch.tensor((0.0, 1.0)),
            height_tolerance_m=0.04,
            xy_tolerance_m=0.20,
            timing_tolerance_frames=8,
            desired_yaw_delta_rad=math.pi,
            command_speed_mps=0.38,
            yaw_tolerance_rad=math.pi / 6.0,
            yaw_cost_weight=0.0,
            turn_xy_cost_weight=0.0,
            turn_root_lateral_cost_weight=0.0,
            turn_root_progress_cost_weight=100.0,
        )

        self.assertEqual(ranking.eligible.tolist(), [True, True])
        self.assertEqual(ranking.selected_action, 1)

    def test_layered_policy_filters_true_strafe_actions_before_motion_cost(self):
        forward_low_motion_cost = _action(
            height=(0.18, 0.18),
            root=((0.08, 0.0), (0.15, 0.0)),
        )
        lateral_high_motion_cost = replace(
            _action(
                height=(0.18, 0.18),
                root=((0.0, 0.08), (0.0, 0.15)),
            ),
            start_frame=1,
            end_frame=22,
        )
        policy = FootholdActionPolicy(
            index=FootholdActionIndex(
                actions=(forward_low_motion_cost, lateral_high_motion_cost),
                _entries={
                    (0, 0): forward_low_motion_cost,
                    (0, 1): lateral_high_motion_cost,
                },
            ),
            extension=SimpleNamespace(
                query_grid=SimpleNamespace(
                    sample_xy=lambda points: torch.zeros(points.shape[:-1])
                ),
                alignment=SimpleNamespace(
                    matcher_to_scene_xy=lambda points: points
                ),
            ),
            arm=FootholdSelectionArm.LAYERED_HYBRID,
        )

        ranking = policy._rank_all_actions(
            _plan(height=(0.18, 0.18)),
            velocity_facing_yaw_delta_rad=math.pi / 2.0,
        )

        self.assertEqual(ranking.eligible.tolist(), [False, True])
        self.assertEqual(ranking.selected_action, 1)

    def test_two_action_lookahead_composes_root_motion_and_yaw(self):
        bad_first = _action(
            yaw=(-math.pi / 4.0, -math.pi / 2.0),
            root=((0.20, 0.20), (0.40, 0.40)),
        )
        translating_first = _action(
            yaw=(-math.pi / 4.0, -math.pi / 2.0),
            root=((-0.08, 0.0), (-0.15, 0.0)),
        )
        translating_second = _action(
            yaw=(-math.pi / 4.0, -math.pi / 2.0),
            root=((0.0, -0.08), (0.0, -0.15)),
        )

        cost = two_action_lookahead_cost(
            actions=(bad_first, translating_first, translating_second),
            desired_yaw_delta_rad=-math.pi,
            command_speed_mps=0.38,
            yaw_tolerance_rad=math.pi / 6.0,
            xy_tolerance_m=0.20,
            yaw_cost_weight=100.0,
            root_lateral_cost_weight=100.0,
            root_progress_cost_weight=10.0,
        )

        self.assertTrue(torch.isfinite(cost[1]))
        self.assertLess(float(cost[1]), float(cost[0]))

        # Give the synthetic actions distinct entry keys while retaining their
        # terminal profiles for policy-level sequence ranking.
        translating_first = replace(
            translating_first, start_frame=1, end_frame=22
        )
        translating_second = replace(
            translating_second, start_frame=2, end_frame=23
        )
        index = FootholdActionIndex(
            actions=(bad_first, translating_first, translating_second),
            _entries={
                (0, 0): bad_first,
                (0, 1): translating_first,
                (0, 2): translating_second,
            },
        )
        policy = FootholdActionPolicy(
            index=index,
            extension=SimpleNamespace(
                query_grid=SimpleNamespace(
                    sample_xy=lambda points: torch.zeros(points.shape[:-1])
                ),
                alignment=SimpleNamespace(
                    matcher_to_scene_xy=lambda points: points
                ),
            ),
            arm=FootholdSelectionArm.LAYERED_HYBRID,
            turn_sequence_candidate_count=1,
            turn_sequence_lookahead=True,
        )

        ranking = policy._rank_all_actions(
            _plan(),
            -math.pi / 2.0,
            hard_yaw_gate=True,
            command_speed_mps=0.38,
            command_frame_yaw_delta_rad=-math.pi,
        )

        self.assertEqual(ranking.selected_action, 1)
        self.assertEqual(ranking.selected_successor_action, 2)
        self.assertEqual(
            policy._sequence_candidate_eligibility(ranking).tolist(),
            [False, True, False],
        )

        policy._remember_turn_pair(
            ranking,
            sequence=10,
            future_facing=torch.tensor((-1.0, 0.0)),
        )
        successor = policy._pending_turn_pair_eligibility(
            state=SimpleNamespace(sequence=20, clip_index=0, frame_index=20),
            future_facing=torch.tensor((-1.0, 0.0)),
            ranking=FootholdRanking(
                torch.tensor((True, True, True)),
                torch.zeros(3),
                0,
            ),
        )

        self.assertEqual(successor.tolist(), [False, False, True])

    def test_planned_turn_successor_is_cancelled_when_command_changes(self):
        actions = (
            _action(start_frame=0),
            _action(start_frame=1),
        )
        policy = FootholdActionPolicy(
            index=FootholdActionIndex(
                actions=actions,
                _entries={(0, 0): actions[0], (0, 1): actions[1]},
            ),
            extension=SimpleNamespace(
                query_grid=SimpleNamespace(
                    sample_xy=lambda points: torch.zeros(points.shape[:-1])
                ),
                alignment=SimpleNamespace(
                    matcher_to_scene_xy=lambda points: points
                ),
            ),
            arm=FootholdSelectionArm.LAYERED_HYBRID,
        )
        policy._remember_turn_pair(
            FootholdRanking(
                torch.tensor((True, True)),
                torch.zeros(2),
                0,
                selected_successor_action=1,
            ),
            sequence=10,
            future_facing=torch.tensor((-1.0, 0.0)),
        )

        successor = policy._pending_turn_pair_eligibility(
            state=SimpleNamespace(sequence=20, clip_index=0, frame_index=20),
            future_facing=torch.tensor((0.0, 1.0)),
            ranking=FootholdRanking(
                torch.tensor((True, True)),
                torch.zeros(2),
                0,
            ),
        )

        self.assertIsNone(successor)

    def test_large_sequence_turn_uses_soft_entry_continuity(self):
        policy = FootholdActionPolicy(
            index=FootholdActionIndex(actions=(), _entries={}),
            extension=SimpleNamespace(
                query_grid=SimpleNamespace(
                    sample_xy=lambda points: torch.zeros(points.shape[:-1])
                ),
                alignment=SimpleNamespace(
                    matcher_to_scene_xy=lambda points: points
                ),
            ),
            arm=FootholdSelectionArm.LAYERED_HYBRID,
            turn_sequence_lookahead=True,
        )

        self.assertTrue(
            policy._hard_entry_graph_gate_required(
                graph_required=True,
                hard_yaw_gate=True,
            )
        )
        self.assertFalse(
            policy._hard_entry_graph_gate_required(
                graph_required=True,
                hard_yaw_gate=True,
                sequence_active=True,
            )
        )
        self.assertTrue(
            policy._hard_entry_graph_gate_required(
                graph_required=True,
                hard_yaw_gate=False,
            )
        )
        self.assertTrue(
            policy._hard_entry_graph_gate_required(
                graph_required=True,
                hard_yaw_gate=False,
                heading_maintenance=True,
            )
        )

    def test_two_action_terrain_gate_covers_four_contacts(self):
        climb = _action(height=(0.18, 0.18), start_frame=0)
        descend = _action(height=(-0.18, -0.18), start_frame=1)
        flat = _action(height=(0.0, 0.0), start_frame=2)

        eligible = two_action_terrain_eligibility(
            first_plan=_plan(height=(0.18, 0.18)),
            second_plans=(_plan(height=(-0.18, -0.18)),),
            actions=(climb, descend, flat),
            height_tolerance_m=0.04,
        )

        self.assertEqual(
            eligible.tolist(),
            [
                [False, True, False],
                [False, False, False],
                [False, False, False],
            ],
        )

    def test_sequence_beam_allows_asymmetric_turn_chunks(self):
        shallow_first = _action(
            start_frame=0,
            yaw=(-0.15, -0.35),
            root=((-0.05, 0.0), (-0.10, 0.0)),
        )
        deep_second = _action(
            start_frame=1,
            yaw=(-1.40, -math.pi + 0.35),
            root=((0.0, -0.05), (0.0, -0.10)),
        )
        conventional_but_drifting = _action(
            start_frame=2,
            yaw=(-math.pi / 4.0, -math.pi / 2.0),
            root=((0.10, 0.0), (0.20, 0.0)),
        )
        policy = FootholdActionPolicy(
            index=FootholdActionIndex(
                actions=(
                    shallow_first,
                    deep_second,
                    conventional_but_drifting,
                ),
                _entries={
                    (0, 0): shallow_first,
                    (0, 1): deep_second,
                    (0, 2): conventional_but_drifting,
                },
            ),
            extension=SimpleNamespace(
                query_grid=SimpleNamespace(
                    sample_xy=lambda points: torch.zeros(points.shape[:-1])
                ),
                alignment=SimpleNamespace(
                    matcher_to_scene_xy=lambda points: points
                ),
            ),
            arm=FootholdSelectionArm.LAYERED_HYBRID,
            turn_sequence_lookahead=True,
        )

        ranking = policy._rank_all_actions(
            _plan(),
            -math.pi / 2.0,
            hard_yaw_gate=True,
            command_speed_mps=0.38,
            command_frame_yaw_delta_rad=-math.pi,
        )

        self.assertTrue(ranking.eligible[0].item())
        self.assertEqual(ranking.selected_action, 0)
        self.assertEqual(ranking.selected_successor_action, 1)
        self.assertFalse(
            policy._fallback_hard_yaw_gate_required(
                hard_yaw_gate=True,
                sequence_action_eligibility=torch.tensor((True, False)),
            )
        )

    def test_turn_feasible_set_ranks_closest_terminal_facing(self):
        overshoot_low_motion_cost = _action(
            height=(0.18, 0.18),
            yaw=(1.0, 1.90),
        )
        exact_high_motion_cost = _action(
            height=(0.18, 0.18),
            yaw=(0.8, math.pi / 2.0),
        )

        ranking = rank_foothold_actions(
            arm=FootholdSelectionArm.LAYERED_HYBRID,
            plan=_plan(height=(0.18, 0.18)),
            actions=(overshoot_low_motion_cost, exact_high_motion_cost),
            motion_cost=torch.tensor((0.0, 1.0)),
            height_tolerance_m=0.04,
            xy_tolerance_m=0.20,
            timing_tolerance_frames=8,
            desired_yaw_delta_rad=math.pi / 2.0,
            yaw_tolerance_rad=math.pi / 3.0,
            yaw_cost_weight=100.0,
        )

        self.assertEqual(ranking.eligible.tolist(), [True, True])
        self.assertEqual(ranking.selected_action, 1)

    def test_turn_matching_rotates_landings_into_terminal_command_frame(self):
        wrong_straight_path = _action(
            height=(0.18, 0.18),
            yaw=(math.pi / 4.0, math.pi / 2.0),
        )
        correct_curved_path = _action(
            height=(0.18, 0.18),
            xy=((-0.10, 0.30), (0.10, 0.60)),
            yaw=(math.pi / 4.0, math.pi / 2.0),
        )

        ranking = rank_foothold_actions(
            arm=FootholdSelectionArm.LAYERED_HYBRID,
            plan=_plan(height=(0.18, 0.18)),
            actions=(wrong_straight_path, correct_curved_path),
            motion_cost=torch.tensor((0.0, 1.0)),
            height_tolerance_m=0.04,
            xy_tolerance_m=0.20,
            timing_tolerance_frames=8,
            desired_yaw_delta_rad=math.pi / 2.0,
            yaw_tolerance_rad=math.pi / 6.0,
            yaw_cost_weight=100.0,
            turn_xy_cost_weight=10.0,
        )

        self.assertEqual(ranking.eligible.tolist(), [True, True])
        self.assertEqual(ranking.selected_action, 1)

    def test_large_future_heading_change_bypasses_transition_graph(self):
        root_quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0))

        self.assertTrue(
            transition_graph_required(
                root_quaternion_world_wxyz=root_quaternion,
                future_facing_world_xy=torch.tensor((1.0, 0.0)),
                maximum_heading_error_rad=torch.pi / 4,
            )
        )
        self.assertFalse(
            transition_graph_required(
                root_quaternion_world_wxyz=root_quaternion,
                future_facing_world_xy=torch.tensor((0.0, 1.0)),
                maximum_heading_error_rad=torch.pi / 4,
            )
        )

    def test_transition_graph_rejects_incompatible_action_entry(self):
        graph = FootholdTransitionGraph(
            action_keys=((0, 10), (1, 20)),
            entry_joint_position=torch.stack(
                (torch.full((29,), 2.0), torch.full((29,), 0.05))
            ),
            entry_joint_velocity=torch.stack(
                (torch.full((29,), 3.0), torch.full((29,), 0.20))
            ),
            terminal_joint_position=torch.stack(
                (torch.full((29,), 0.05), torch.full((29,), 2.0))
            ),
            terminal_joint_velocity=torch.stack(
                (torch.full((29,), 0.20), torch.full((29,), 3.0))
            ),
        )

        eligible = graph.entry_eligibility(
            current_joint_position=torch.zeros(29),
            current_joint_velocity=torch.zeros(29),
            maximum_position_error_rad=0.50,
            maximum_velocity_error_rad_s=2.0,
        )

        self.assertEqual(eligible.tolist(), [False, True])
        pair = graph.pair_eligibility(
            maximum_position_error_rad=0.50,
            maximum_velocity_error_rad_s=2.0,
        )
        self.assertEqual(pair.tolist(), [[False, True], [True, False]])

    def test_graph_arm_keeps_only_contact_and_boundary_compatible_rows(self):
        incompatible = _action(height=(0.18, 0.18), start_frame=0)
        compatible = _action(height=(0.18, 0.18), start_frame=1)
        index = FootholdActionIndex(
            actions=(incompatible, compatible),
            _entries={(0, 0): incompatible, (0, 1): compatible},
        )
        graph = FootholdTransitionGraph(
            action_keys=((0, 0), (0, 1)),
            entry_joint_position=torch.stack(
                (torch.full((29,), 0.50), torch.full((29,), 0.02))
            ),
            entry_joint_velocity=torch.stack(
                (torch.full((29,), 1.50), torch.full((29,), 0.10))
            ),
        )

        class StepGrid:
            @staticmethod
            def sample_xy(points):
                return torch.where(
                    points[..., 0] >= 0.24,
                    torch.full_like(points[..., 0], 0.18),
                    torch.zeros_like(points[..., 0]),
                )

        policy = FootholdActionPolicy(
            index=index,
            extension=SimpleNamespace(
                query_grid=StepGrid(),
                alignment=SimpleNamespace(
                    matcher_to_scene_xy=lambda points: points
                ),
            ),
            arm=FootholdSelectionArm.LAYERED_GRAPH_HYBRID,
            transition_graph=graph,
            graph_maximum_position_error_rad=0.90,
            graph_maximum_velocity_error_rad_s=5.0,
            graph_fallback_maximum_position_error_rad=1.50,
            graph_fallback_maximum_velocity_error_rad_s=8.0,
        )
        database = SimpleNamespace(
            _search_clip_index=torch.tensor((0, 0, 1)),
            _search_frame_index=torch.tensor((0, 1, 0)),
            device=torch.device("cpu"),
        )
        state = SimpleNamespace(
            sequence=0,
            feature_body_position=torch.tensor(
                (
                    (0.0, 0.0, 0.8),
                    (0.0, -0.1, 0.035),
                    (0.0, 0.1, 0.035),
                )
            ),
            joint_position=torch.zeros(29),
            joint_velocity=torch.zeros(29),
            root_quaternion=torch.tensor((1.0, 0.0, 0.0, 0.0)),
        )
        shaped = SimpleNamespace(
            velocity_world_xy=torch.tensor((1.0, 0.0)),
            trajectory=SimpleNamespace(
                facing_world_xy=torch.tensor(((1.0, 0.0),))
            ),
        )

        result = policy.prepare(state, shaped, database)

        self.assertTrue(result.terrain_action_required)
        self.assertEqual(result.row_eligibility.tolist(), [False, True, False])
        self.assertEqual(
            result.fallback_row_eligibility.tolist(), [False, True, False]
        )

    def test_two_contact_action_records_alternating_landings(self):
        support, foot_xy, surface, root_xy, root_yaw = (
            _two_contact_profiles()
        )

        action = action_from_profiles(
            clip_index=2,
            start_frame=1,
            support_mask=support,
            foot_xy_m=foot_xy,
            foot_surface_height_m=surface,
            root_xy_m=root_xy,
            root_yaw_rad=root_yaw,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.start_support, (True, False))
        self.assertEqual(action.landing_feet, (1, 0))
        self.assertEqual(action.landing_frame_offsets, (2, 4))
        self.assertEqual(action.end_frame, 6)
        torch.testing.assert_close(
            action.landing_height_delta_m,
            torch.tensor((0.18, 0.18)),
        )
        torch.testing.assert_close(
            action.root_displacement_m,
            torch.tensor(((0.00, -0.15), (0.00, -0.30))),
            atol=1e-6,
            rtol=0.0,
        )

    def test_action_rejects_repeated_same_foot_landing(self):
        with self.assertRaisesRegex(ContractError, "alternate"):
            FootholdAction(
                clip_index=0,
                start_frame=0,
                end_frame=5,
                start_support=(True, False),
                landing_feet=(0, 0),
                landing_frame_offsets=(2, 4),
                landing_xy_start_frame_m=torch.zeros((2, 2)),
                landing_height_delta_m=torch.zeros(2),
                root_displacement_m=torch.zeros((2, 2)),
                root_yaw_delta_rad=torch.zeros(2),
                minimum_swing_clearance_m=0.04,
                maximum_unsupported_frames=0,
            )

    def test_action_returns_none_without_two_alternating_landings(self):
        support, foot_xy, surface, root_xy, root_yaw = (
            _two_contact_profiles()
        )
        support[5, 0] = False

        action = action_from_profiles(
            clip_index=2,
            start_frame=1,
            support_mask=support,
            foot_xy_m=foot_xy,
            foot_surface_height_m=surface,
            root_xy_m=root_xy,
            root_yaw_rad=root_yaw,
        )

        self.assertIsNone(action)

    def test_simultaneous_double_support_onset_is_not_two_landings(self):
        support, foot_xy, surface, root_xy, root_yaw = (
            _two_contact_profiles()
        )
        support[:] = torch.tensor(
            [
                [True, True],
                [False, False],
                [False, False],
                [True, True],
                [False, True],
                [True, True],
            ],
            dtype=torch.bool,
        )

        action = action_from_profiles(
            clip_index=2,
            start_frame=0,
            support_mask=support,
            foot_xy_m=foot_xy,
            foot_surface_height_m=surface,
            root_xy_m=root_xy,
            root_yaw_rad=root_yaw,
        )

        self.assertEqual(action.landing_frame_offsets, (3, 5))
        self.assertEqual(action.landing_feet, (1, 0))

    def test_dataset_index_maps_only_exact_action_entry_rows(self):
        support, foot_xy, surface, root_xy, root_yaw = (
            _two_contact_profiles()
        )
        body = torch.zeros((6, 3, 3), dtype=torch.float32)
        body[:, 0, :2] = root_xy
        body[:, 0, 2] = 0.8
        body[:, 1:, :2] = foot_xy
        body[:, 1:, 2] = surface + 0.035
        half_yaw = root_yaw / 2.0
        root_quaternion = torch.stack(
            (
                torch.cos(half_yaw),
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
            ),
            dim=-1,
        )
        quaternion = torch.zeros((6, 3, 4), dtype=torch.float32)
        quaternion[..., 0] = 1.0
        quaternion[:, 0] = root_quaternion
        clip = SimpleNamespace(
            body_position_world=body.numpy(),
            body_quaternion_world_wxyz=quaternion.numpy(),
        )

        class StepGrid:
            @staticmethod
            def sample_xy(points):
                return torch.where(
                    points[..., 0] >= 0.25,
                    torch.full_like(points[..., 0], 0.18),
                    torch.zeros_like(points[..., 0]),
                )

        class IdentityAlignment:
            @staticmethod
            def matcher_to_scene_xy(points):
                return points

        dataset = SimpleNamespace(
            folder=SimpleNamespace(
                clips=(clip,),
                layout=SimpleNamespace(
                    root_body_index=0,
                    left_foot_body_index=1,
                    right_foot_body_index=2,
                ),
            ),
            clip_grids=(StepGrid(),),
            clip_alignments=(IdentityAlignment(),),
            device=torch.device("cpu"),
        )
        segment_index = SimpleNamespace(
            segments=(SimpleNamespace(clip_index=0, start_frame=1),),
            support_mask=lambda clip_index: support.clone(),
        )

        index = FootholdActionIndex.from_dataset(dataset, segment_index)
        database = SimpleNamespace(
            _search_clip_index=torch.tensor((0, 0, 0)),
            _search_frame_index=torch.tensor((0, 1, 2)),
            device=torch.device("cpu"),
        )

        self.assertEqual(len(index.actions), 1)
        rows = index.rows_for_database(database)
        self.assertIsNone(rows[0])
        self.assertIs(rows[1], index.actions[0])
        self.assertIsNone(rows[2])
        self.assertIs(index.entry(0, 1), index.actions[0])
        self.assertIsNone(index.entry(0, 0))
        torch.testing.assert_close(
            index.actions[0].landing_height_delta_m,
            torch.tensor((0.18, 0.18)),
        )

    def test_flat_approach_cannot_plan_raised_first_landing(self):
        def step_surface(points):
            return torch.where(
                points[..., 0] >= 0.50,
                torch.full_like(points[..., 0], 0.18),
                torch.zeros_like(points[..., 0]),
            )

        plan = plan_footholds(
            foot_xy_m=torch.tensor(((0.00, -0.10), (0.00, 0.10))),
            support_mask=torch.tensor((True, True)),
            command_xy=torch.tensor((1.00, 0.00)),
            sample_surface=step_surface,
            reachable_forward_m=(0.20, 0.40),
            lateral_samples_m=(-0.10, 0.00, 0.10),
            edge_margin_m=0.04,
            beam_width=8,
        )

        self.assertGreater(plan.landing_height_delta_m.shape[0], 0)
        self.assertTrue(
            bool((plan.landing_height_delta_m[:, 0] == 0.0).all().item())
        )

    def test_first_contact_filter_is_hard_not_weighted(self):
        def flat_surface(points):
            return torch.zeros(points.shape[:-1], dtype=points.dtype)

        plan = plan_footholds(
            foot_xy_m=torch.tensor(((0.00, -0.10), (0.00, 0.10))),
            support_mask=torch.tensor((True, True)),
            command_xy=torch.tensor((1.00, 0.00)),
            sample_surface=flat_surface,
            reachable_forward_m=(0.30,),
            lateral_samples_m=(0.00,),
            edge_margin_m=0.04,
            beam_width=8,
        )

        eligible = first_contact_eligibility(
            plan=plan,
            actions=(_action(height=(0.18, 0.18)), _action()),
            height_tolerance_m=0.04,
            xy_tolerance_m=0.20,
            timing_tolerance_frames=8,
        )

        self.assertEqual(eligible.tolist(), [False, True])

    def test_foothold_plan_rejects_an_edge_under_the_sole(self):
        def narrow_surface(points):
            return torch.where(
                points[..., 0] < 0.30,
                torch.zeros_like(points[..., 0]),
                torch.full_like(points[..., 0], -0.20),
            )

        plan = plan_footholds(
            foot_xy_m=torch.tensor(((0.00, -0.10), (0.00, 0.10))),
            support_mask=torch.tensor((True, True)),
            command_xy=torch.tensor((1.00, 0.00)),
            sample_surface=narrow_surface,
            reachable_forward_m=(0.30,),
            lateral_samples_m=(0.10,),
            edge_margin_m=0.04,
            beam_width=8,
        )

        self.assertEqual(plan.landing_xy_world_m.shape, (0, 2, 2))

    def test_foothold_plan_batches_all_candidate_surface_queries(self):
        calls = 0

        def flat_surface(points):
            nonlocal calls
            calls += 1
            return torch.zeros(points.shape[:-1], dtype=points.dtype)

        plan = plan_footholds(
            foot_xy_m=torch.tensor(((0.00, -0.10), (0.00, 0.10))),
            support_mask=torch.tensor((True, True)),
            command_xy=torch.tensor((1.00, 0.00)),
            sample_surface=flat_surface,
            reachable_forward_m=(0.20, 0.30, 0.40),
            lateral_samples_m=(-0.10, 0.00, 0.10),
            edge_margin_m=0.04,
            beam_width=18,
        )

        self.assertEqual(plan.score.shape[0], 18)
        self.assertEqual(calls, 2)

    def test_two_contact_arm_rejects_wrong_second_tread(self):
        ranking = rank_foothold_actions(
            arm=FootholdSelectionArm.TWO_CONTACT,
            plan=_plan(height=(0.18, 0.18)),
            actions=(
                _action(height=(0.18, 0.00)),
                _action(height=(0.18, 0.18)),
            ),
            motion_cost=torch.tensor((0.0, 100.0)),
            height_tolerance_m=0.04,
            xy_tolerance_m=0.20,
            timing_tolerance_frames=8,
        )

        self.assertEqual(ranking.eligible.tolist(), [False, True])
        self.assertEqual(ranking.selected_action, 1)

    def test_hybrid_breaks_feasible_tie_with_motion_cost(self):
        ranking = rank_foothold_actions(
            arm=FootholdSelectionArm.HYBRID,
            plan=_plan(),
            actions=(_action(), _action()),
            motion_cost=torch.tensor((8.0, 1.0)),
            height_tolerance_m=0.04,
            xy_tolerance_m=0.20,
            timing_tolerance_frames=8,
        )

        self.assertEqual(ranking.eligible.tolist(), [True, True])
        self.assertEqual(ranking.selected_action, 1)

    def test_hybrid_uses_motion_cost_after_two_contact_gate(self):
        exact_footholds_rough_pose = _action()
        nearby_footholds_smooth_pose = _action(
            xy=((0.45, 0.10), (0.75, -0.10)),
        )

        ranking = rank_foothold_actions(
            arm=FootholdSelectionArm.HYBRID,
            plan=_plan(),
            actions=(exact_footholds_rough_pose, nearby_footholds_smooth_pose),
            motion_cost=torch.tensor((0.8, 0.0)),
            height_tolerance_m=0.04,
            xy_tolerance_m=0.20,
            timing_tolerance_frames=8,
        )

        self.assertEqual(ranking.eligible.tolist(), [True, True])
        torch.testing.assert_close(
            ranking.additional_cost, torch.zeros(2)
        )
        self.assertEqual(ranking.selected_action, 1)

    def test_first_contact_uses_motion_cost_inside_hard_gate(self):
        exact_footholds_rough_pose = _action()
        nearby_footholds_smooth_pose = _action(
            xy=((0.45, 0.10), (0.75, -0.10)),
        )

        ranking = rank_foothold_actions(
            arm=FootholdSelectionArm.FIRST_CONTACT,
            plan=_plan(),
            actions=(exact_footholds_rough_pose, nearby_footholds_smooth_pose),
            motion_cost=torch.tensor((0.8, 0.0)),
            height_tolerance_m=0.04,
            xy_tolerance_m=0.20,
            timing_tolerance_frames=8,
        )

        self.assertEqual(ranking.eligible.tolist(), [True, True])
        torch.testing.assert_close(
            ranking.additional_cost, torch.zeros(2)
        )
        self.assertEqual(ranking.selected_action, 1)

    def test_layered_arm_keeps_contact_heights_hard_and_xy_soft(self):
        far_but_correct_contacts = _action(
            height=(0.18, 0.18),
            xy=((0.80, 0.50), (1.20, -0.50)),
        )
        wrong_second_height = _action(
            height=(0.18, 0.00),
        )

        ranking = rank_foothold_actions(
            arm=FootholdSelectionArm.LAYERED,
            plan=_plan(height=(0.18, 0.18)),
            actions=(far_but_correct_contacts, wrong_second_height),
            motion_cost=torch.tensor((10.0, 0.0)),
            height_tolerance_m=0.04,
            xy_tolerance_m=0.20,
            timing_tolerance_frames=8,
        )

        self.assertEqual(ranking.eligible.tolist(), [True, False])
        self.assertEqual(ranking.selected_action, 0)

    def test_layered_hybrid_keeps_contact_gate_but_uses_motion_cost(self):
        exact_footholds_rough_pose = _action(height=(0.18, 0.18))
        nearby_footholds_smooth_pose = _action(
            height=(0.18, 0.18),
            xy=((0.45, 0.10), (0.75, -0.10)),
        )
        wrong_second_height = _action(height=(0.18, 0.00))

        ranking = rank_foothold_actions(
            arm=FootholdSelectionArm.LAYERED_HYBRID,
            plan=_plan(height=(0.18, 0.18)),
            actions=(
                exact_footholds_rough_pose,
                nearby_footholds_smooth_pose,
                wrong_second_height,
            ),
            motion_cost=torch.tensor((0.8, 0.0, 0.0)),
            height_tolerance_m=0.04,
            xy_tolerance_m=0.20,
            timing_tolerance_frames=8,
        )

        self.assertEqual(ranking.eligible.tolist(), [True, True, False])
        torch.testing.assert_close(
            ranking.additional_cost, torch.zeros(3)
        )
        self.assertEqual(ranking.selected_action, 1)

    def test_continuous_control_can_prefer_wrong_contact_sequence(self):
        ranking = rank_foothold_actions(
            arm=FootholdSelectionArm.CONTINUOUS,
            plan=_plan(height=(0.18, 0.18)),
            actions=(
                _action(height=(0.18, 0.00)),
                _action(height=(0.18, 0.18)),
            ),
            motion_cost=torch.tensor((0.0, 100.0)),
            height_tolerance_m=0.04,
            xy_tolerance_m=0.20,
            timing_tolerance_frames=8,
        )

        self.assertEqual(ranking.eligible.tolist(), [True, True])
        self.assertEqual(ranking.selected_action, 0)

    def test_policy_maps_hard_contact_filter_back_to_database_rows(self):
        raised = _action(height=(0.18, 0.18), start_frame=0)
        flat = _action(start_frame=1)
        index = FootholdActionIndex(
            actions=(raised, flat),
            _entries={(0, 0): raised, (0, 1): flat},
        )

        class FlatGrid:
            @staticmethod
            def sample_xy(points):
                return torch.zeros(
                    points.shape[:-1],
                    dtype=points.dtype,
                    device=points.device,
                )

        extension = SimpleNamespace(
            query_grid=FlatGrid(),
            alignment=SimpleNamespace(
                matcher_to_scene_xy=lambda points: points
            ),
        )
        policy = FootholdActionPolicy(
            index=index,
            extension=extension,
            arm=FootholdSelectionArm.LAYERED,
        )
        database = SimpleNamespace(
            _search_clip_index=torch.tensor((0, 0, 1)),
            _search_frame_index=torch.tensor((0, 1, 0)),
            device=torch.device("cpu"),
        )
        state = SimpleNamespace(
            feature_body_position=torch.tensor(
                (
                    (0.0, 0.0, 0.8),
                    (0.0, -0.1, 0.035),
                    (0.0, 0.1, 0.035),
                )
            )
        )
        shaped = SimpleNamespace(
            velocity_world_xy=torch.tensor((1.0, 0.0))
        )

        result = policy.prepare(state, shaped, database)

        self.assertEqual(
            result.row_eligibility.tolist(), [False, False, True]
        )
        self.assertEqual(
            result.fallback_row_eligibility.tolist(), [False, False, True]
        )
        self.assertFalse(result.terrain_action_required)
        self.assertEqual(policy.fallback_candidate_count, 1024)
        self.assertEqual(policy.beam_width, 50)
        self.assertEqual(result.candidate_count, 0)

    def test_policy_activates_terrain_actions_only_when_plan_changes_height(self):
        raised = _action(height=(0.18, 0.18), start_frame=0)
        index = FootholdActionIndex(
            actions=(raised,),
            _entries={(0, 0): raised},
        )

        class StepGrid:
            @staticmethod
            def sample_xy(points):
                return torch.where(
                    points[..., 0] >= 0.24,
                    torch.full(
                        points.shape[:-1],
                        0.18,
                        dtype=points.dtype,
                        device=points.device,
                    ),
                    torch.zeros(
                        points.shape[:-1],
                        dtype=points.dtype,
                        device=points.device,
                    ),
                )

        policy = FootholdActionPolicy(
            index=index,
            extension=SimpleNamespace(
                query_grid=StepGrid(),
                alignment=SimpleNamespace(
                    matcher_to_scene_xy=lambda points: points
                ),
            ),
            arm=FootholdSelectionArm.LAYERED,
        )
        database = SimpleNamespace(
            _search_clip_index=torch.tensor((0, 1)),
            _search_frame_index=torch.tensor((0, 0)),
            device=torch.device("cpu"),
        )
        state = SimpleNamespace(
            feature_body_position=torch.tensor(
                (
                    (0.0, 0.0, 0.8),
                    (0.0, -0.1, 0.035),
                    (0.0, 0.1, 0.035),
                )
            )
        )
        shaped = SimpleNamespace(
            velocity_world_xy=torch.tensor((1.0, 0.0))
        )

        result = policy.prepare(state, shaped, database)

        self.assertTrue(result.terrain_action_required)
        self.assertTrue(result.row_eligibility[0].item())
        self.assertFalse(result.row_eligibility[1].item())
        self.assertEqual(
            result.fallback_row_eligibility.tolist(), [True, False]
        )


if __name__ == "__main__":
    unittest.main()
