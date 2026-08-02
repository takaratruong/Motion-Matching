import math
import unittest

import torch

from mm_sonic.torch_contact_oracle_actions import ContactPhaseAction
from mm_sonic.torch_contact_oracle_search import OracleState, place_action
from mm_sonic.torch_terrain_contact_composition import (
    ContactTargetTrajectory,
    build_contact_target_trajectory,
    place_action_contact_anchored,
    project_contact_trajectory,
)


def _action(entry_support=(True, False)):
    orientation = torch.zeros((3, 4), dtype=torch.float32)
    orientation[:, 0] = 1.0
    return ContactPhaseAction(
        clip_index=0,
        start_frame=0,
        end_frame=3,
        swing_foot=1,
        entry_support=torch.tensor(entry_support),
        exit_support=torch.tensor((True, True)),
        support_mask=torch.tensor(
            (entry_support, (True, False), (True, True))
        ),
        joint_position=torch.zeros((3, 29)),
        joint_velocity=torch.zeros((3, 29)),
        root_position_local=torch.tensor(
            ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0))
        ),
        root_yaw_local=torch.zeros(3),
        root_orientation_local_wxyz=orientation,
        foot_position_local=torch.tensor(
            (
                ((0.0, 0.1, -0.8), (0.0, -0.1, -0.72)),
                ((0.0, 0.1, -0.8), (0.1, -0.1, -0.65)),
                ((0.0, 0.1, -0.8), (0.2, -0.1, -0.8)),
            )
        ),
        foot_surface_delta_m=torch.zeros((3, 2)),
        minimum_swing_clearance_m=0.05,
    )


def _state(feet, support=(True, False), yaw=0.0):
    return OracleState(
        root_position_world=torch.tensor((2.0, 3.0, 0.8)),
        root_yaw_world=torch.tensor(yaw),
        root_orientation_world_wxyz=torch.tensor(
            (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        ),
        foot_position_world=torch.tensor(feet, dtype=torch.float32),
        support_mask=torch.tensor(support),
        joint_position=torch.zeros(29),
        joint_velocity=torch.zeros(29),
        route_frame=0,
    )


class _LinearFootKinematics:
    def foot_positions(self, joints, roots, quaternions):
        del quaternions
        feet = torch.as_tensor(joints)[:, :6].reshape(-1, 2, 3)
        return (feet + torch.as_tensor(roots)[:, None, :]).numpy()

    def solve_leg_positions(self, joints, root, quaternion, mask, targets):
        del quaternion
        output = torch.as_tensor(joints).clone()
        for foot in range(2):
            if mask[foot]:
                output[foot * 3 : foot * 3 + 3] = torch.as_tensor(
                    targets[foot] - root
                )
        return output.numpy()


class _FailingFootKinematics(_LinearFootKinematics):
    def solve_leg_positions(self, joints, root, quaternion, mask, targets):
        del joints, root, quaternion, mask, targets
        raise RuntimeError("unreachable")


class TerrainContactCompositionTests(unittest.TestCase):
    def test_single_support_anchor_eliminates_entry_error(self):
        state = _state(((2.25, 3.15, 0.05), (2.0, 2.9, 0.08)))
        action = _action()
        root_placed = place_action(action, state)

        anchored = place_action_contact_anchored(action, state)

        root_error = torch.linalg.vector_norm(
            root_placed.foot_position_world[0, 0] - state.foot_position_world[0]
        )
        self.assertGreater(float(root_error), 0.20)
        self.assertAlmostEqual(anchored.maximum_entry_support_error_m, 0.0, places=6)
        torch.testing.assert_close(
            anchored.placed.foot_position_world[0, 0], state.foot_position_world[0]
        )
        self.assertAlmostEqual(float(anchored.yaw_world), 0.0)

    def test_double_support_fit_recovers_planar_rotation_and_translation(self):
        action = _action(entry_support=(True, True))
        target = ((1.9, 3.0, 0.02), (2.1, 3.0, 0.10))
        state = _state(target, support=(True, True), yaw=0.4)

        anchored = place_action_contact_anchored(action, state)

        self.assertAlmostEqual(float(anchored.yaw_world), math.pi / 2, places=6)
        torch.testing.assert_close(
            anchored.placed.foot_position_world[0], state.foot_position_world
        )
        self.assertAlmostEqual(anchored.maximum_entry_support_error_m, 0.0, places=6)

    def test_double_support_reports_nonrigid_height_residual(self):
        action = _action(entry_support=(True, True))
        state = _state(
            ((1.9, 3.0, 0.02), (2.1, 3.0, 0.02)),
            support=(True, True),
        )

        anchored = place_action_contact_anchored(action, state)

        self.assertAlmostEqual(
            anchored.maximum_entry_support_error_m,
            0.04,
            places=6,
        )

    def test_rigid_fit_preserves_all_pairwise_foot_distances(self):
        action = _action()
        state = _state(((2.25, 3.15, 0.05), (2.0, 2.9, 0.08)))
        anchored = place_action_contact_anchored(action, state)

        source_delta = action.foot_position_local[:, 1] - action.foot_position_local[:, 0]
        placed_delta = (
            anchored.placed.foot_position_world[:, 1]
            - anchored.placed.foot_position_world[:, 0]
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(placed_delta, dim=1),
            torch.linalg.vector_norm(source_delta, dim=1),
        )

    def test_rejects_entry_support_mismatch(self):
        with self.assertRaisesRegex(ValueError, "entry support"):
            place_action_contact_anchored(
                _action(),
                _state(
                    ((2.0, 3.1, 0.0), (2.0, 2.9, 0.0)),
                    support=(False, True),
                ),
            )

    def test_contact_targets_lock_stance_and_warp_swing_to_landing(self):
        raw = torch.tensor(
            (
                ((0.00, 0.10, 0.00), (0.00, -0.10, 0.20)),
                ((0.01, 0.10, 0.00), (0.10, -0.10, 0.25)),
                ((0.02, 0.10, 0.00), (0.20, -0.10, 0.30)),
                ((0.03, 0.10, 0.00), (0.30, -0.10, 0.20)),
                ((0.04, 0.10, 0.00), (0.40, -0.10, 0.00)),
            )
        )
        support = torch.tensor(
            ((True, False),) * 4 + ((True, True),)
        )
        entry = torch.tensor(((1.0, 2.0, 0.0), (0.0, 0.0, 0.0)))
        landing = torch.tensor((0.48, -0.06, 0.12))

        targets = build_contact_target_trajectory(
            raw,
            support,
            swing_foot=1,
            entry_foot_position_world=entry,
            landing_target_world=landing,
        )

        torch.testing.assert_close(
            targets.position_world[:, 0], entry[0].expand(5, 3)
        )
        torch.testing.assert_close(targets.position_world[0, 1], raw[0, 1])
        torch.testing.assert_close(targets.position_world[-1, 1], landing)
        torch.testing.assert_close(
            targets.swing_warp_weight,
            torch.tensor((0.0, 0.15625, 0.5, 0.84375, 1.0)),
        )
        self.assertTrue(bool(targets.solve_mask.all()))

    def test_zero_landing_warp_preserves_raw_swing_trajectory(self):
        raw = torch.tensor(
            (
                ((0.0, 0.1, 0.0), (0.0, -0.1, 0.2)),
                ((0.0, 0.1, 0.0), (0.1, -0.1, 0.3)),
                ((0.0, 0.1, 0.0), (0.2, -0.1, 0.0)),
            )
        )
        support = torch.tensor(
            ((True, False), (True, False), (True, True))
        )

        targets = build_contact_target_trajectory(
            raw,
            support,
            swing_foot=1,
            entry_foot_position_world=raw[0],
            landing_target_world=raw[-1, 1],
        )

        torch.testing.assert_close(targets.position_world[:, 1], raw[:, 1])

    def test_landed_swing_remains_locked_after_touchdown(self):
        raw = torch.tensor(
            (
                ((0.0, 0.1, 0.0), (0.0, -0.1, 0.2)),
                ((0.0, 0.1, 0.0), (0.1, -0.1, 0.3)),
                ((0.0, 0.1, 0.0), (0.2, -0.1, 0.0)),
                ((0.0, 0.1, 0.0), (0.3, -0.1, 0.0)),
            )
        )
        support = torch.tensor(
            ((True, False), (True, False), (True, True), (True, True))
        )
        landing = torch.tensor((0.25, -0.08, 0.04))

        targets = build_contact_target_trajectory(
            raw,
            support,
            swing_foot=1,
            entry_foot_position_world=raw[0],
            landing_target_world=landing,
        )

        torch.testing.assert_close(
            targets.position_world[2:, 1], landing.expand(2, 3)
        )

    def test_contact_projection_hits_every_selected_target(self):
        joints = torch.zeros((3, 29))
        roots = torch.tensor(((1.0, 2.0, 0.5),) * 3)
        quaternions = torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 3)
        positions = torch.tensor(
            (
                ((1.1, 2.1, 0.0), (0.9, 1.9, 0.1)),
                ((1.2, 2.1, 0.0), (0.8, 1.9, 0.2)),
                ((1.3, 2.1, 0.0), (0.7, 1.9, 0.0)),
            )
        )
        targets = ContactTargetTrajectory(
            position_world=positions,
            solve_mask=torch.ones((3, 2), dtype=torch.bool),
            swing_warp_weight=torch.tensor((0.0, 0.5, 1.0)),
        )

        projected = project_contact_trajectory(
            joint_position=joints,
            root_position_world=roots,
            root_orientation_world_wxyz=quaternions,
            targets=targets,
            foot_kinematics=_LinearFootKinematics(),
        )

        torch.testing.assert_close(projected.foot_position_world, positions)
        self.assertAlmostEqual(projected.maximum_target_error_m, 0.0, places=6)
        self.assertGreater(projected.maximum_joint_deformation_rad, 0.0)

    def test_contact_projection_preserves_unselected_leg(self):
        joints = torch.zeros((2, 29))
        joints[:, 3:6] = torch.tensor((0.2, -0.1, 0.3))
        roots = torch.zeros((2, 3))
        quaternions = torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 2)
        targets = ContactTargetTrajectory(
            position_world=torch.tensor(
                (
                    ((0.1, 0.2, 0.3), (9.0, 9.0, 9.0)),
                    ((0.2, 0.3, 0.4), (9.0, 9.0, 9.0)),
                )
            ),
            solve_mask=torch.tensor(((True, False), (True, False))),
            swing_warp_weight=torch.zeros(2),
        )

        projected = project_contact_trajectory(
            joint_position=joints,
            root_position_world=roots,
            root_orientation_world_wxyz=quaternions,
            targets=targets,
            foot_kinematics=_LinearFootKinematics(),
        )

        torch.testing.assert_close(projected.joint_position[:, 3:6], joints[:, 3:6])

    def test_contact_projection_fails_closed_on_unreachable_target(self):
        targets = ContactTargetTrajectory(
            position_world=torch.zeros((2, 2, 3)),
            solve_mask=torch.ones((2, 2), dtype=torch.bool),
            swing_warp_weight=torch.zeros(2),
        )
        with self.assertRaisesRegex(ValueError, "frame 0"):
            project_contact_trajectory(
                joint_position=torch.zeros((2, 29)),
                root_position_world=torch.zeros((2, 3)),
                root_orientation_world_wxyz=torch.tensor(
                    ((1.0, 0.0, 0.0, 0.0),) * 2
                ),
                targets=targets,
                foot_kinematics=_FailingFootKinematics(),
            )


if __name__ == "__main__":
    unittest.main()
