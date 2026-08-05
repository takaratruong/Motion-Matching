import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_horizontal_terrain_retarget import (
    WideBoundG1TerrainRetargeter,
    contact_repair_targets,
    nearest_valid_sole_translation,
    raise_penetrating_stance_anchors,
    retarget_foot_targets,
    stance_segment_anchors,
)


G1_XML = (
    "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
)


class HorizontalTerrainRetargetTest(unittest.TestCase):
    def test_only_required_feet_trigger_unreachable_target_error(self):
        retargeter = WideBoundG1TerrainRetargeter(
            G1_XML,
            maximum_joint_deviation_rad=0.35,
            maximum_root_height_deviation_m=0.08,
        )
        joints = np.zeros(29)
        root = np.array((0.0, 0.0, 0.8))
        quaternion = np.array((1.0, 0.0, 0.0, 0.0))
        feet = retargeter.kinematics.foot_positions(
            joints[None], root[None], quaternion[None]
        )[0]
        targets = feet.copy()
        targets[1, 2] += 1.0

        solved_joints, solved_root = retargeter.solve_frame(
            joint_position=joints,
            root_position_world=root,
            root_orientation_world_wxyz=quaternion,
            solve_feet=np.array((True, True)),
            enforce_target_error_feet=np.array((True, False)),
            target_foot_position_weights=np.array((1.0, 0.01)),
            target_foot_position_world=targets,
        )

        actual = retargeter.kinematics.foot_positions(
            solved_joints[None],
            solved_root[None],
            quaternion[None],
        )[0]
        self.assertLess(
            float(np.linalg.norm(actual[0] - targets[0])), 0.005
        )

    def test_optional_com_target_selects_horizontal_balance_solution(self):
        retargeter = WideBoundG1TerrainRetargeter(
            G1_XML,
            maximum_root_horizontal_deviation_m=0.05,
            maximum_root_height_deviation_m=0.08,
            center_of_mass_scale=100.0,
        )
        joints = np.zeros(29)
        root = np.array((0.0, 0.0, 0.8))
        quaternion = np.array((1.0, 0.0, 0.0, 0.0))
        feet = retargeter.kinematics.foot_positions(
            joints[None], root[None], quaternion[None]
        )[0]
        source_com = retargeter.kinematics.center_of_mass_positions(
            joints[None], root[None], quaternion[None]
        )[0]
        target_com_xy = source_com[:2] + (0.0, 0.02)

        solved_joints, solved_root = retargeter.solve_frame(
            joint_position=joints,
            root_position_world=root,
            root_orientation_world_wxyz=quaternion,
            solve_feet=np.array((True, True)),
            target_foot_position_world=feet,
            target_center_of_mass_world_xy=target_com_xy,
        )

        actual_com = retargeter.kinematics.center_of_mass_positions(
            solved_joints[None],
            solved_root[None],
            quaternion[None],
        )[0]
        self.assertLess(
            float(np.linalg.norm(actual_com[:2] - target_com_xy)),
            0.01,
        )

    def test_rejects_nonpositive_hip_yaw_posture_scale(self):
        with self.assertRaisesRegex(ContractError, "bounds"):
            WideBoundG1TerrainRetargeter(
                G1_XML, hip_yaw_posture_scale=0.0
            )

    def test_rejects_nonpositive_foot_position_scale(self):
        with self.assertRaisesRegex(ContractError, "bounds"):
            WideBoundG1TerrainRetargeter(
                G1_XML, foot_position_scale=0.0
            )

    def test_full_foot_frame_target_preserves_heading(self):
        retargeter = WideBoundG1TerrainRetargeter(
            G1_XML,
            maximum_root_height_deviation_m=0.08,
            foot_orientation_scale=10.0,
        )
        joints = np.zeros(29)
        root = np.array((0.0, 0.0, 0.8))
        quaternion = np.array((1.0, 0.0, 0.0, 0.0))
        feet = retargeter.kinematics.foot_positions(
            joints[None], root[None], quaternion[None]
        )[0]
        rotations = retargeter.kinematics.foot_rotations(
            joints[None], root[None], quaternion[None]
        )[0]
        targets = feet.copy()
        targets[1, 2] += 0.04

        solved_joints, solved_root = retargeter.solve_frame(
            joint_position=joints,
            root_position_world=root,
            root_orientation_world_wxyz=quaternion,
            solve_feet=np.array((False, True)),
            target_foot_position_world=targets,
            target_foot_rotation_world=rotations,
        )

        actual = retargeter.kinematics.foot_rotations(
            solved_joints[None],
            solved_root[None],
            quaternion[None],
        )[0]
        self.assertLess(
            float(np.linalg.norm(actual[1, :, 0] - rotations[1, :, 0])),
            0.03,
        )

    def test_level_foot_projects_target_heading_onto_horizontal_plane(self):
        retargeter = WideBoundG1TerrainRetargeter(
            G1_XML,
            maximum_root_height_deviation_m=0.08,
            foot_orientation_scale=10.0,
        )
        joints = np.zeros(29)
        root = np.array((0.0, 0.0, 0.8))
        quaternion = np.array((1.0, 0.0, 0.0, 0.0))
        feet = retargeter.kinematics.foot_positions(
            joints[None], root[None], quaternion[None]
        )[0]
        rotations = retargeter.kinematics.foot_rotations(
            joints[None], root[None], quaternion[None]
        )[0]
        pitch = np.deg2rad(25.0)
        pitch_rotation = np.array(
            (
                (np.cos(pitch), 0.0, np.sin(pitch)),
                (0.0, 1.0, 0.0),
                (-np.sin(pitch), 0.0, np.cos(pitch)),
            )
        )
        rotations[1] = rotations[1] @ pitch_rotation

        solved_joints, solved_root = retargeter.solve_frame(
            joint_position=joints,
            root_position_world=root,
            root_orientation_world_wxyz=quaternion,
            solve_feet=np.array((False, True)),
            target_foot_position_world=feet,
            target_foot_rotation_world=rotations,
            level_feet=np.array((False, True)),
        )

        actual = retargeter.kinematics.foot_rotations(
            solved_joints[None],
            solved_root[None],
            quaternion[None],
        )[0, 1]
        self.assertLess(
            float(np.linalg.norm(actual[:, 2] - (0.0, 0.0, 1.0))),
            0.03,
        )
        expected_heading = rotations[1, :, 0].copy()
        expected_heading[2] = 0.0
        expected_heading /= np.linalg.norm(expected_heading)
        self.assertLess(
            float(np.linalg.norm(actual[:, 0] - expected_heading)),
            0.03,
        )

    def test_stance_segment_anchors_hold_each_support_interval(self):
        feet = np.zeros((6, 2, 3), dtype=np.float64)
        feet[:, 0, 0] = np.arange(6)
        feet[:, 1, 1] = 10.0 + np.arange(6)
        support = np.array(
            [
                [True, False],
                [True, False],
                [False, True],
                [False, True],
                [True, False],
                [True, False],
            ],
            dtype=np.bool_,
        )

        anchors = stance_segment_anchors(
            foot_position_world=feet,
            support_mask=support,
        )

        np.testing.assert_allclose(
            anchors[:2, 0], np.repeat(feet[0:1, 0], 2, axis=0)
        )
        np.testing.assert_allclose(
            anchors[2:4, 1], np.repeat(feet[2:3, 1], 2, axis=0)
        )
        np.testing.assert_allclose(
            anchors[4:, 0], np.repeat(feet[4:5, 0], 2, axis=0)
        )
        self.assertTrue(np.isnan(anchors[~support]).all())

    def test_stance_segment_anchors_can_preserve_segment_endpoints(self):
        feet = np.zeros((5, 2, 3), dtype=np.float64)
        feet[:, 0, 0] = [0.0, 0.2, 0.6, 0.8, 1.0]
        support = np.zeros((5, 2), dtype=np.bool_)
        support[:, 0] = True

        anchors = stance_segment_anchors(
            foot_position_world=feet,
            support_mask=support,
            preserve_segment_endpoints=True,
        )

        np.testing.assert_allclose(
            anchors[:, 0, 0], [0.0, 0.15625, 0.5, 0.84375, 1.0]
        )

    def test_contact_repair_locks_stance_and_caps_swing_lift(self):
        feet = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.1]], dtype=np.float64
        )
        anchors = np.array(
            [[0.0, 0.0, 0.25], [np.nan, np.nan, np.nan]],
            dtype=np.float64,
        )

        mask, targets = contact_repair_targets(
            foot_position_world=feet,
            support_mask=np.array([True, False]),
            stance_anchor_world=anchors,
            minimum_sole_clearance_by_foot_m=np.array([-0.01, -0.30]),
            swing_clearance_margin_m=0.005,
            maximum_swing_lift_per_solve_m=0.08,
        )

        np.testing.assert_array_equal(mask, [True, True])
        np.testing.assert_allclose(targets[0], anchors[0])
        np.testing.assert_allclose(targets[1], [0.4, 0.5, 0.18])

    def test_penetrating_touchdown_raises_whole_stance_interval(self):
        anchors = np.full((4, 2, 3), np.nan, dtype=np.float64)
        anchors[:2, 0] = [0.0, 0.0, 0.10]
        anchors[2:, 1] = [0.2, 0.0, 0.20]
        support = np.array(
            [[True, False], [True, False], [False, True], [False, True]],
            dtype=np.bool_,
        )
        clearance = np.array(
            [[-0.010, 0.0], [-0.010, 0.0], [0.0, -0.026], [0.0, -0.020]]
        )

        repaired = raise_penetrating_stance_anchors(
            stance_anchor_world=anchors,
            support_mask=support,
            minimum_sole_clearance_by_foot_m=clearance,
            penetration_threshold_m=-0.025,
            clearance_target_m=-0.020,
        )

        np.testing.assert_allclose(repaired[:2, 0], anchors[:2, 0])
        np.testing.assert_allclose(
            repaired[2:, 1, 2], anchors[2:, 1, 2] + 0.006
        )

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

    def test_strong_orientation_constraint_levels_uneven_gait_support(self):
        joints = np.array(
            [
                -0.6594299103142957,
                -1.3099796253485574,
                -0.12620101987841628,
                -0.11230822118852846,
                0.15528483176038432,
                0.041794970822852676,
                0.6952146699353007,
                -0.02129298046410546,
                0.2560571524187521,
                1.4050792762757751,
                1.5484058605758848,
                0.14581482196825096,
                0.05277133031620762,
                -0.8726699999999998,
                -0.3287941442800435,
                0.20432931203400037,
                -0.345186150260876,
                0.2617999999999999,
                -0.15686580731430239,
                -0.32139429492117144,
                0.31868645176674604,
                0.6085445590086163,
                0.7667682279758147,
                -0.012783667962171362,
                0.011376160226479647,
                -0.03631921356551514,
                0.08005953085681759,
                0.05169664246369492,
                -0.04414656590087286,
            ]
        )
        root = np.array(
            [1.2633623305008501, 0.7254382649930645, 1.0903051147353917]
        )
        quaternion = np.array(
            [
                -0.7093220966628914,
                0.029441473962307264,
                -0.05055766603895421,
                -0.7024523366046839,
            ]
        )
        retargeter = WideBoundG1TerrainRetargeter(
            G1_XML,
            maximum_root_height_deviation_m=0.08,
            maximum_root_horizontal_deviation_m=0.04,
            foot_orientation_scale=10.0,
        )
        feet = retargeter.kinematics.foot_positions(
            joints[None], root[None], quaternion[None]
        )[0]

        solved_joints, solved_root = retargeter.solve_frame(
            joint_position=joints,
            root_position_world=root,
            root_orientation_world_wxyz=quaternion,
            solve_feet=np.array((True, False)),
            target_foot_position_world=feet,
            level_feet=np.array((True, False)),
        )
        sole = MujocoG1SoleKinematics(G1_XML).sole_points(
            solved_joints[None],
            solved_root[None],
            quaternion[None],
        )[0, 0]

        self.assertLess(float(np.ptp(sole[:, 2])), 0.010)

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

    def test_disjoint_warm_start_does_not_move_a_fixed_root(self):
        retargeter = WideBoundG1TerrainRetargeter(
            G1_XML,
            maximum_root_height_deviation_m=1.0e-6,
            maximum_root_horizontal_deviation_m=1.0e-6,
        )
        joints = np.zeros(29)
        root = np.array((0.0, 0.0, 0.8))
        quaternion = np.array((1.0, 0.0, 0.0, 0.0))
        feet = retargeter.kinematics.foot_positions(
            joints[None], root[None], quaternion[None]
        )[0]

        _, solved_root = retargeter.solve_frame(
            joint_position=joints,
            root_position_world=root,
            root_orientation_world_wxyz=quaternion,
            solve_feet=np.array((False, True)),
            target_foot_position_world=feet,
            initial_joint_position=joints,
            initial_root_position_world=np.array((0.10, 0.0, 0.8)),
        )

        np.testing.assert_allclose(solved_root, root, atol=1.0e-6)

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
