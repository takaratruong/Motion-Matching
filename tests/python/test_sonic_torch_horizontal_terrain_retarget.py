import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_horizontal_terrain_retarget import (
    WideBoundG1TerrainRetargeter,
    nearest_valid_sole_translation,
    retarget_foot_targets,
)


G1_XML = (
    "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
)


class HorizontalTerrainRetargetTest(unittest.TestCase):
    def test_finds_nearest_same_height_sole_footprint(self):
        sole_xy = np.array(
            ((-0.10, -0.04), (0.10, -0.04), (-0.10, 0.04), (0.10, 0.04))
        )

        def stair_height(points):
            points = np.asarray(points)
            return np.where(points[..., 1] < 0.0, 0.18, 0.0)

        shift = nearest_valid_sole_translation(
            sole_point_scene_xy=sole_xy,
            support_surface_height_m=0.0,
            sample_height=stair_height,
            maximum_shift_m=0.08,
            search_resolution_m=0.01,
        )

        np.testing.assert_allclose(shift, (0.0, 0.04), atol=1e-12)

    def test_rejects_a_support_surface_too_narrow_for_the_sole(self):
        sole_xy = np.array(
            ((-0.10, -0.04), (0.10, -0.04), (-0.10, 0.04), (0.10, 0.04))
        )

        with self.assertRaisesRegex(ContractError, "no valid sole footprint"):
            nearest_valid_sole_translation(
                sole_point_scene_xy=sole_xy,
                support_surface_height_m=0.0,
                sample_height=lambda points: np.full(
                    np.asarray(points).shape[:-1], 0.18
                ),
                maximum_shift_m=0.08,
                search_resolution_m=0.01,
            )

    def test_allows_lower_tread_overhang_with_partial_support(self):
        sole_xy = np.array(
            ((-0.10, -0.04), (0.10, -0.04), (-0.10, 0.04), (0.10, 0.04))
        )

        shift = nearest_valid_sole_translation(
            sole_point_scene_xy=sole_xy,
            support_surface_height_m=0.0,
            sample_height=lambda points: np.where(
                np.asarray(points)[..., 1] > 0.0, -0.18, 0.0
            ),
            maximum_shift_m=0.08,
            search_resolution_m=0.01,
        )

        np.testing.assert_allclose(shift, (0.0, 0.0), atol=1e-12)

    def test_fixed_root_solver_reaches_small_swing_lift(self):
        retargeter = WideBoundG1TerrainRetargeter(
            G1_XML, maximum_root_height_deviation_m=1.0e-6
        )
        joints = np.zeros(29)
        root = np.array((0.0, 0.0, 0.8))
        quaternion = np.array((1.0, 0.0, 0.0, 0.0))
        feet = retargeter.kinematics.foot_positions(
            joints[None], root[None], quaternion[None]
        )[0]
        targets = feet.copy()
        targets[1, 2] += 0.05

        solved_joints, solved_root = retargeter.solve_frame(
            joint_position=joints,
            root_position_world=root,
            root_orientation_world_wxyz=quaternion,
            solve_feet=np.array((False, True)),
            target_foot_position_world=targets,
        )

        actual = retargeter.kinematics.foot_positions(
            solved_joints[None],
            solved_root[None],
            quaternion[None],
        )[0]
        self.assertLessEqual(
            float(np.linalg.norm(actual[1] - targets[1])), 0.005
        )
        self.assertLessEqual(abs(float(solved_root[2] - root[2])), 1.0e-6)

    def test_large_support_shift_keeps_the_sole_level(self):
        retargeter = WideBoundG1TerrainRetargeter(
            G1_XML, maximum_root_height_deviation_m=1.0e-6
        )
        joints = np.zeros(29)
        root = np.array((0.0, 0.0, 0.8))
        quaternion = np.array((1.0, 0.0, 0.0, 0.0))
        feet = retargeter.kinematics.foot_positions(
            joints[None], root[None], quaternion[None]
        )[0]
        targets = feet.copy()
        targets[1, 1] += 0.08

        solved_joints, solved_root = retargeter.solve_frame(
            joint_position=joints,
            root_position_world=root,
            root_orientation_world_wxyz=quaternion,
            solve_feet=np.array((False, True)),
            target_foot_position_world=targets,
        )
        sole = MujocoG1SoleKinematics(G1_XML).sole_points(
            solved_joints[None],
            solved_root[None],
            quaternion[None],
        )[0, 1]

        self.assertLess(float(np.ptp(sole[:, 2])), 0.005)

    def test_support_shift_can_move_the_root_horizontally(self):
        retargeter = WideBoundG1TerrainRetargeter(
            G1_XML,
            maximum_root_height_deviation_m=1.0e-6,
            maximum_root_horizontal_deviation_m=0.10,
        )
        joints = np.zeros(29)
        root = np.array((0.0, 0.0, 0.8))
        quaternion = np.array((1.0, 0.0, 0.0, 0.0))
        feet = retargeter.kinematics.foot_positions(
            joints[None], root[None], quaternion[None]
        )[0]
        targets = feet.copy()
        targets[1, 1] += 0.08

        solved_joints, solved_root = retargeter.solve_frame(
            joint_position=joints,
            root_position_world=root,
            root_orientation_world_wxyz=quaternion,
            solve_feet=np.array((False, True)),
            target_foot_position_world=targets,
        )
        actual = retargeter.kinematics.foot_positions(
            solved_joints[None],
            solved_root[None],
            quaternion[None],
        )[0]

        self.assertLessEqual(
            float(np.linalg.norm(actual[1] - targets[1])), 0.005
        )
        self.assertGreater(abs(float(solved_root[1] - root[1])), 0.005)

    def test_solver_accepts_the_previous_solution_as_a_warm_start(self):
        retargeter = WideBoundG1TerrainRetargeter(G1_XML)
        joints = np.zeros(29)
        root = np.array((0.0, 0.0, 0.8))
        quaternion = np.array((1.0, 0.0, 0.0, 0.0))
        feet = retargeter.kinematics.foot_positions(
            joints[None], root[None], quaternion[None]
        )[0]
        first_targets = feet.copy()
        first_targets[1, 1] += 0.06
        first_joints, first_root = retargeter.solve_frame(
            joint_position=joints,
            root_position_world=root,
            root_orientation_world_wxyz=quaternion,
            solve_feet=np.array((False, True)),
            target_foot_position_world=first_targets,
        )
        second_targets = first_targets.copy()
        second_targets[1, 1] += 0.002

        second_joints, _ = retargeter.solve_frame(
            joint_position=joints,
            root_position_world=root,
            root_orientation_world_wxyz=quaternion,
            solve_feet=np.array((False, True)),
            target_foot_position_world=second_targets,
            initial_joint_position=first_joints,
            initial_root_position_world=first_root,
        )

        self.assertLess(
            float(np.max(np.abs(second_joints - first_joints))), 0.10
        )

    def test_support_is_pinned_and_colliding_swing_is_lifted(self):
        feet = np.array(
            [[0.0, 0.0, 0.20], [0.2, 0.0, 0.30]], dtype=np.float64
        )

        mask, targets = retarget_foot_targets(
            foot_position_world=feet,
            support_mask=np.array([True, False]),
            target_surface_at_ankle_m=np.array([0.10, 0.25]),
            minimum_sole_clearance_by_foot_m=np.array([-0.01, -0.06]),
            ankle_origin_sole_m=0.035,
            swing_clearance_margin_m=0.02,
        )

        np.testing.assert_array_equal(mask, [True, True])
        np.testing.assert_allclose(targets[0], [0.0, 0.0, 0.135])
        np.testing.assert_allclose(targets[1], [0.2, 0.0, 0.38])

    def test_clear_swing_foot_is_not_constrained(self):
        mask, _ = retarget_foot_targets(
            foot_position_world=np.array(
                [[0.0, 0.0, 0.20], [0.2, 0.0, 0.30]]
            ),
            support_mask=np.array([True, False]),
            target_surface_at_ankle_m=np.array([0.10, 0.25]),
            minimum_sole_clearance_by_foot_m=np.array([-0.01, 0.03]),
            ankle_origin_sole_m=0.035,
            swing_clearance_margin_m=0.02,
        )

        np.testing.assert_array_equal(mask, [True, False])


if __name__ == "__main__":
    unittest.main()
