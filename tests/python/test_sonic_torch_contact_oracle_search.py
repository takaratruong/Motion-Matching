import math
import unittest
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from unittest import mock

import torch

import mm_sonic.torch_contact_oracle_search as oracle_search

from mm_sonic.torch_contact_oracle_actions import ContactPhaseAction
from mm_sonic.torch_contact_oracle_search import (
    OracleConstraints,
    OracleSearchConfig,
    OracleSearchFailure,
    OracleState,
    constant_command_schedule,
    load_contact_oracle_config,
    place_action,
    search_contact_plan,
    validate_placement,
)


def _one_meter_forward_action() -> ContactPhaseAction:
    return ContactPhaseAction(
        clip_index=0,
        start_frame=10,
        end_frame=12,
        swing_foot=1,
        entry_support=torch.tensor((True, False)),
        exit_support=torch.tensor((True, True)),
        support_mask=torch.tensor(((True, False), (True, True))),
        joint_position=torch.zeros((2, 29)),
        joint_velocity=torch.zeros((2, 29)),
        root_position_local=torch.tensor(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        root_yaw_local=torch.zeros(2),
        foot_position_local=torch.tensor(
            (
                ((0.0, 0.1, 0.035), (0.0, -0.1, 0.135)),
                ((0.0, 0.1, 0.035), (1.0, -0.1, 0.035)),
            )
        ),
        foot_surface_delta_m=torch.zeros((2, 2)),
        minimum_swing_clearance_m=0.0,
    )


def _state(*, yaw: float = 0.0) -> OracleState:
    root = torch.tensor((2.0, 3.0, 0.0))
    local_feet = torch.tensor(((0.0, 0.1), (0.0, -0.1)))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    feet_xy = torch.stack(
        (
            cosine * local_feet[:, 0] - sine * local_feet[:, 1],
            sine * local_feet[:, 0] + cosine * local_feet[:, 1],
        ),
        dim=1,
    ) + root[:2]
    feet = torch.cat((feet_xy, torch.full((2, 1), 0.035)), dim=1)
    return OracleState(
        root_position_world=root,
        root_yaw_world=torch.tensor(yaw),
        foot_position_world=feet,
        support_mask=torch.tensor((True, False)),
        joint_position=torch.zeros(29),
        joint_velocity=torch.zeros(29),
        route_frame=0,
    )


def _graph_action(index: int, entry: float, terminal: float, dx: float):
    action = _one_meter_forward_action()
    joints = action.joint_position.clone()
    joints[0] = entry
    joints[-1] = terminal
    root = action.root_position_local.clone()
    root[-1, 0] = dx
    feet = action.foot_position_local.clone()
    feet[-1, 1, 0] = dx
    return replace(
        action,
        clip_index=index,
        start_frame=10 * index,
        end_frame=10 * index + 2,
        entry_support=torch.tensor((True, True)),
        support_mask=torch.tensor(((True, True), (True, True))),
        root_position_local=root,
        foot_position_local=feet,
        joint_position=joints,
    )


class ContactOracleSearchTests(unittest.TestCase):
    def test_command_schedule_batches_variable_horizon_queries(self):
        schedule = oracle_search.CommandSchedule(
            velocity_world_xy=torch.tensor(
                ((1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0))
            ),
            heading_world_yaw=torch.tensor((0.0, 0.1, 0.2, 0.3)),
        )
        steps = torch.tensor((0, 1, 3), dtype=torch.int64)

        torch.testing.assert_close(
            schedule.displacements(1, steps),
            torch.tensor(((0.0, 0.0), (0.04, 0.0), (0.18, 0.0))),
        )
        torch.testing.assert_close(
            schedule.headings(1, steps), torch.tensor((0.1, 0.2, 0.3))
        )

    def test_loads_frozen_oracle_config_and_rejects_extra_keys(self):
        path = Path("sonic/configs/experiments/torch_grail_contact_oracle.json")
        loaded = load_contact_oracle_config(path)
        self.assertEqual(loaded.search.horizon_landings, 4)
        self.assertEqual(loaded.search.beam_width, 32)
        self.assertEqual(loaded.search.transition_candidate_count, 32)
        self.assertEqual(
            loaded.search.constraints.maximum_joint_position_error_rad, 2.5
        )
        self.assertEqual(loaded.terrain_config.name, "torch_grail_layered_graph_hybrid.json")

        malformed = json.loads(path.read_text())
        malformed["unexpected"] = True
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "bad.json"
            target.write_text(json.dumps(malformed))
            with self.assertRaisesRegex(Exception, "keys"):
                load_contact_oracle_config(target)

    def test_rigid_placement_rotates_source_trajectory_about_entry_root(self):
        placed = place_action(_one_meter_forward_action(), _state(yaw=math.pi / 2.0))

        torch.testing.assert_close(
            placed.root_position_world[-1, :2], torch.tensor((2.0, 4.0))
        )
        self.assertAlmostEqual(
            float(placed.root_yaw_world[-1]), math.pi / 2.0, places=6
        )
        torch.testing.assert_close(
            placed.foot_position_world[0, 0],
            _state(yaw=math.pi / 2.0).foot_position_world[0],
        )

    def test_rejects_landing_on_height_discontinuity_inside_edge_margin(self):
        action = _one_meter_forward_action()
        feet = action.foot_position_local.clone()
        feet[-1, 1, 2] = 0.235
        surface_delta = action.foot_surface_delta_m.clone()
        surface_delta[-1, 1] = 0.2
        action = replace(
            action,
            foot_position_local=feet,
            foot_surface_delta_m=surface_delta,
        )

        def step_surface(points):
            return torch.where(
                points[..., 0] >= 3.0,
                torch.full(points.shape[:-1], 0.2, device=points.device),
                torch.zeros(points.shape[:-1], device=points.device),
            )

        state = _state()
        result = validate_placement(
            placed=place_action(action, state),
            state=state,
            sample_surface=step_surface,
            constraints=OracleConstraints(),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "landing-edge-margin")

    def test_reports_new_swing_foot_height_as_landing_not_stance_failure(self):
        action = _one_meter_forward_action()
        feet = action.foot_position_local.clone()
        feet[-1, action.swing_foot, 2] = 0.20
        action = replace(action, foot_position_local=feet)

        def flat_surface(points):
            return torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)

        state = _state()
        result = validate_placement(
            placed=place_action(action, state),
            state=state,
            sample_surface=flat_surface,
            constraints=OracleConstraints(),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "landing-height")

    def test_rejects_swing_penetration_before_height_deformation(self):
        action = _one_meter_forward_action()
        feet = action.foot_position_local.clone()
        feet[0, action.swing_foot, 2] = 0.0
        surface_delta = action.foot_surface_delta_m.clone()
        surface_delta[-1, action.swing_foot] = 0.2
        action = replace(
            action,
            foot_position_local=feet,
            foot_surface_delta_m=surface_delta,
        )

        def flat_surface(points):
            return torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)

        state = _state()
        result = validate_placement(
            placed=place_action(action, state),
            state=state,
            sample_surface=flat_surface,
            constraints=OracleConstraints(),
        )

        self.assertEqual(result.reason, "swing-penetration")

    def test_rejects_source_to_query_landing_height_deformation(self):
        action = _one_meter_forward_action()
        surface_delta = action.foot_surface_delta_m.clone()
        surface_delta[-1, action.swing_foot] = 0.2
        action = replace(action, foot_surface_delta_m=surface_delta)

        def flat_surface(points):
            return torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)

        state = _state()
        result = validate_placement(
            placed=place_action(action, state),
            state=state,
            sample_surface=flat_surface,
            constraints=OracleConstraints(),
        )

        self.assertEqual(result.reason, "height-deformation")

    def test_accepts_flat_action_and_orders_entry_constraints(self):
        def flat_surface(points):
            return torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)

        action = _one_meter_forward_action()
        state = _state()
        accepted = validate_placement(
            placed=place_action(action, state),
            state=state,
            sample_surface=flat_surface,
            constraints=OracleConstraints(),
        )
        self.assertTrue(accepted.accepted)

        wrong_support = replace(
            state, support_mask=torch.tensor((False, True))
        )
        self.assertEqual(
            validate_placement(
                placed=place_action(action, wrong_support),
                state=wrong_support,
                sample_surface=flat_surface,
                constraints=OracleConstraints(),
            ).reason,
            "support-order",
        )

        displaced_feet = state.foot_position_world.clone()
        displaced_feet[0, 0] += 0.2
        displaced = replace(state, foot_position_world=displaced_feet)
        self.assertEqual(
            validate_placement(
                placed=place_action(action, displaced),
                state=displaced,
                sample_surface=flat_surface,
                constraints=OracleConstraints(),
            ).reason,
            "entry-foot-error",
        )

        wrong_position = replace(state, joint_position=torch.ones(29))
        self.assertEqual(
            validate_placement(
                placed=place_action(action, wrong_position),
                state=wrong_position,
                sample_surface=flat_surface,
                constraints=OracleConstraints(),
            ).reason,
            "joint-position",
        )

        wrong_velocity = replace(
            state, joint_velocity=torch.full((29,), 2.0)
        )
        self.assertEqual(
            validate_placement(
                placed=place_action(action, wrong_velocity),
                state=wrong_velocity,
                sample_surface=flat_surface,
                constraints=OracleConstraints(),
            ).reason,
            "joint-velocity",
        )

    def test_entry_foot_gate_runs_before_full_trajectory_placement(self):
        action = _one_meter_forward_action()
        feet = action.foot_position_local.clone()
        feet[0, 0, 0] += 0.2
        action = replace(action, foot_position_local=feet)

        def flat_surface(points):
            return torch.zeros(
                points.shape[:-1], dtype=points.dtype, device=points.device
            )

        with mock.patch.object(
            oracle_search,
            "place_action",
            side_effect=AssertionError("full placement must not run"),
        ):
            with self.assertRaises(OracleSearchFailure) as caught:
                search_contact_plan(
                    initial_state=_state(),
                    actions=(action,),
                    command_schedule=constant_command_schedule(
                        velocity_world_xy=(0.3, 0.0),
                        heading_world_yaw=0.0,
                        frames=4,
                        device="cpu",
                    ),
                    sample_surface=flat_surface,
                    config=OracleSearchConfig(horizon_landings=1),
                )
        self.assertEqual(
            dict(caught.exception.rejected_by_reason),
            {"entry-foot-error": 1},
        )

    def test_four_contact_search_rejects_cheapest_greedy_dead_end(self):
        actions = (
            _graph_action(0, entry=0.0, terminal=10.0, dx=0.006),
            _graph_action(1, entry=0.0, terminal=1.0, dx=0.010),
            _graph_action(2, entry=1.0, terminal=2.0, dx=0.010),
            _graph_action(3, entry=2.0, terminal=3.0, dx=0.010),
        )

        def flat_surface(points):
            return torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)

        constraints = OracleConstraints(
            maximum_entry_foot_error_m=10.0,
            maximum_joint_position_error_rad=0.2,
            maximum_joint_velocity_error_rad_s=0.2,
        )
        plan = search_contact_plan(
            initial_state=_state(),
            actions=actions,
            command_schedule=constant_command_schedule(
                velocity_world_xy=(0.3, 0.0),
                heading_world_yaw=0.0,
                frames=100,
                device=torch.device("cpu"),
            ),
            sample_surface=flat_surface,
            config=OracleSearchConfig(
                horizon_landings=3,
                beam_width=8,
                constraints=constraints,
            ),
        )

        self.assertEqual(plan.action_indices, (1, 2, 3))
        self.assertGreater(
            plan.expansion.rejected_by_reason["no-successor"], 0
        )

    def test_layered_shortlist_uses_command_cost_not_source_order(self):
        actions = (
            _graph_action(0, entry=0.0, terminal=0.0, dx=-0.20),
            _graph_action(1, entry=0.0, terminal=0.0, dx=0.006),
        )

        def flat_surface(points):
            return torch.zeros(
                points.shape[:-1], dtype=points.dtype, device=points.device
            )

        plan = search_contact_plan(
            initial_state=_state(),
            actions=actions,
            command_schedule=constant_command_schedule(
                velocity_world_xy=(0.3, 0.0),
                heading_world_yaw=0.0,
                frames=4,
                device="cpu",
            ),
            sample_surface=flat_surface,
            config=OracleSearchConfig(
                horizon_landings=1,
                beam_width=1,
                transition_candidate_count=1,
                constraints=OracleConstraints(
                    maximum_entry_foot_error_m=10.0
                ),
            ),
        )

        self.assertEqual(plan.action_indices, (1,))
        self.assertEqual(
            dict(plan.expansion.rejected_by_reason)["shortlist-pruned"], 1
        )

    def test_shortlisted_terrain_validation_is_batched(self):
        action = _one_meter_forward_action()

        def flat_surface(points):
            return torch.zeros(
                points.shape[:-1], dtype=points.dtype, device=points.device
            )

        with mock.patch.object(
            oracle_search,
            "validate_placement",
            side_effect=AssertionError("scalar validation must not run"),
        ):
            plan = search_contact_plan(
                initial_state=_state(),
                actions=(action,),
                command_schedule=constant_command_schedule(
                    velocity_world_xy=(0.3, 0.0),
                    heading_world_yaw=0.0,
                    frames=4,
                    device="cpu",
                ),
                sample_surface=flat_surface,
                config=OracleSearchConfig(horizon_landings=1),
            )

        self.assertEqual(plan.action_indices, (0,))

    def test_successor_feasibility_is_reused_at_next_depth(self):
        action = _graph_action(0, entry=0.0, terminal=0.0, dx=0.006)

        def flat_surface(points):
            return torch.zeros(
                points.shape[:-1], dtype=points.dtype, device=points.device
            )

        with mock.patch.object(
            oracle_search,
            "_feasible_actions",
            wraps=oracle_search._feasible_actions,
        ) as feasible:
            plan = search_contact_plan(
                initial_state=_state(),
                actions=(action,),
                command_schedule=constant_command_schedule(
                    velocity_world_xy=(0.3, 0.0),
                    heading_world_yaw=0.0,
                    frames=10,
                    device="cpu",
                ),
                sample_surface=flat_surface,
                config=OracleSearchConfig(
                    horizon_landings=2,
                    constraints=OracleConstraints(
                        maximum_entry_foot_error_m=10.0
                    ),
                ),
            )

        self.assertEqual(plan.action_indices, (0, 0))
        self.assertEqual(feasible.call_count, 2)
    def test_search_shortens_horizon_but_never_invents_a_fallback(self):
        action = _graph_action(0, entry=0.0, terminal=10.0, dx=0.006)

        def flat_surface(points):
            return torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)

        config = OracleSearchConfig(
            horizon_landings=3,
            beam_width=4,
            constraints=OracleConstraints(
                maximum_entry_foot_error_m=10.0,
                maximum_joint_position_error_rad=0.2,
                maximum_joint_velocity_error_rad_s=0.2,
            ),
        )
        schedule = constant_command_schedule(
            velocity_world_xy=(0.3, 0.0),
            heading_world_yaw=0.0,
            frames=20,
            device="cpu",
        )

        plan = search_contact_plan(
            initial_state=_state(),
            actions=(action,),
            command_schedule=schedule,
            sample_surface=flat_surface,
            config=config,
        )
        self.assertEqual(plan.horizon_landings, 1)
        self.assertEqual(plan.action_indices, (0,))

        incompatible = replace(
            action,
            entry_support=torch.tensor((False, True)),
            support_mask=torch.tensor(((False, True), (True, True))),
        )
        with self.assertRaises(OracleSearchFailure) as caught:
            search_contact_plan(
                initial_state=_state(),
                actions=(incompatible,),
                command_schedule=schedule,
                sample_surface=flat_surface,
                config=config,
            )
        self.assertGreater(caught.exception.rejected_by_reason["support-order"], 0)

    def test_facing_command_is_independent_of_travel_velocity(self):
        straight = _graph_action(0, entry=0.0, terminal=0.0, dx=0.006)
        turning_yaw = straight.root_yaw_local.clone()
        turning_yaw[-1] = math.pi / 2.0
        turning = replace(
            straight,
            clip_index=1,
            start_frame=20,
            end_frame=22,
            root_yaw_local=turning_yaw,
        )

        def flat_surface(points):
            return torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)

        plan = search_contact_plan(
            initial_state=_state(),
            actions=(straight, turning),
            command_schedule=constant_command_schedule(
                velocity_world_xy=(0.3, 0.0),
                heading_world_yaw=math.pi / 2.0,
                frames=20,
                device="cpu",
            ),
            sample_surface=flat_surface,
            config=OracleSearchConfig(
                horizon_landings=1,
                constraints=OracleConstraints(maximum_entry_foot_error_m=10.0),
            ),
        )
        self.assertEqual(plan.action_indices, (1,))

    def test_planned_foothold_xy_breaks_equal_root_motion_tie(self):
        bad = _graph_action(0, entry=0.0, terminal=0.0, dx=0.006)
        bad_feet = bad.foot_position_local.clone()
        bad_feet[-1, bad.swing_foot, 0] = 0.60
        bad = replace(bad, foot_position_local=bad_feet)
        good = _graph_action(1, entry=0.0, terminal=0.0, dx=0.006)
        good_feet = good.foot_position_local.clone()
        good_feet[-1, good.swing_foot, 0] = 0.20
        good = replace(good, foot_position_local=good_feet)

        def flat_surface(points):
            return torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)

        plan = search_contact_plan(
            initial_state=_state(),
            actions=(bad, good),
            command_schedule=constant_command_schedule(
                velocity_world_xy=(0.3, 0.0),
                heading_world_yaw=0.0,
                frames=20,
                device="cpu",
            ),
            sample_surface=flat_surface,
            config=OracleSearchConfig(
                horizon_landings=1,
                constraints=OracleConstraints(maximum_entry_foot_error_m=10.0),
            ),
        )

        self.assertEqual(plan.action_indices, (1,))
        self.assertAlmostEqual(plan.step_costs[0].foothold, 0.0)
        self.assertGreater(plan.step_costs[0].timing, 0.0)


if __name__ == "__main__":
    unittest.main()
