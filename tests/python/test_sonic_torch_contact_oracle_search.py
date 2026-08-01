import math
import unittest
from dataclasses import replace

import torch

from mm_sonic.torch_contact_oracle_actions import ContactPhaseAction
from mm_sonic.torch_contact_oracle_search import (
    OracleConstraints,
    OracleState,
    place_action,
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


class ContactOracleSearchTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
