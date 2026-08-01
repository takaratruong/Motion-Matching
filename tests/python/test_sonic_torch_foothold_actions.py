from dataclasses import dataclass
from enum import Enum
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
        FootholdSelectionArm,
        rank_foothold_actions,
    )
except ImportError:
    FootholdActionPolicy = None

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
    *, height=(0.0, 0.0), feet=(1, 0), xy=None, start_frame=0
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
        root_displacement_m=torch.tensor(
            ((0.25, 0.0), (0.50, 0.0))
        ),
        root_yaw_delta_rad=torch.zeros(2),
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
