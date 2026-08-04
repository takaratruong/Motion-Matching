import math
import unittest

import numpy as np

from mm_sonic.torch_stair_connector import (
    BoundedFootKinematics,
    StairConnectorBoundary,
    cubic_hermite_trajectory,
    pivot_stance_target,
    planar_foot_alignment,
    single_step_foot_trajectory,
    single_step_sole_trajectory,
    synthesize_double_support_connector,
    synthesize_single_step_connector,
)


class StairConnectorTest(unittest.TestCase):
    def test_pivot_target_rotates_only_the_swing_foot(self):
        feet = np.asarray(
            ((0.0, 0.10, 0.20), (0.0, -0.10, 0.20)),
            dtype=np.float64,
        )

        target = pivot_stance_target(
            feet,
            pivot_foot=0,
            yaw_delta_rad=math.pi / 2,
        )

        np.testing.assert_allclose(target[0], feet[0], atol=1e-12)
        np.testing.assert_allclose(
            target[1],
            (0.20, 0.10, 0.20),
            atol=1e-12,
        )

    def test_single_step_path_lifts_swing_and_holds_pivot(self):
        feet = np.asarray(
            ((0.0, 0.10, 0.20), (0.0, -0.10, 0.20)),
            dtype=np.float64,
        )
        target = np.asarray((0.20, 0.10, 0.38), dtype=np.float64)

        path = single_step_foot_trajectory(
            feet,
            target,
            pivot_foot=0,
            frame_count=9,
            swing_clearance_m=0.10,
        )

        np.testing.assert_allclose(
            path[:, 0],
            np.repeat(feet[None, 0], len(path), axis=0),
        )
        np.testing.assert_allclose(path[0, 1], feet[1])
        np.testing.assert_allclose(path[-1, 1], target)
        self.assertGreater(
            path[4, 1, 2],
            max(feet[1, 2], target[2]) + 0.09,
        )

    def test_single_step_swing_lift_has_near_zero_endpoint_velocity(self):
        feet = np.asarray(
            ((0.0, 0.10, 0.20), (0.0, -0.10, 0.20)),
            dtype=np.float64,
        )
        path = single_step_foot_trajectory(
            feet,
            np.asarray((0.20, 0.10, 0.20)),
            pivot_foot=0,
            frame_count=61,
            swing_clearance_m=0.10,
        )

        vertical_step = np.diff(path[:, 1, 2])
        peak_step = float(np.abs(vertical_step).max())
        self.assertLess(abs(float(vertical_step[0])), peak_step * 0.1)
        self.assertLess(abs(float(vertical_step[-1])), peak_step * 0.1)
        vertical_acceleration = np.diff(path[:, 1, 2], n=2)
        peak_acceleration = float(np.abs(vertical_acceleration).max())
        self.assertLess(
            abs(float(vertical_acceleration[0])),
            peak_acceleration * 0.3,
        )
        self.assertLess(
            abs(float(vertical_acceleration[-1])),
            peak_acceleration * 0.3,
        )

    def test_single_step_sole_path_fixes_pivot_and_yaws_swing_rigidly(self):
        feet = np.asarray(
            ((0.0, 0.10, 0.25), (0.0, -0.10, 0.25)),
            dtype=np.float64,
        )
        local = np.asarray(
            (
                (0.10, 0.04, -0.05),
                (-0.10, -0.04, -0.05),
                (0.0, 0.0, -0.05),
            ),
            dtype=np.float64,
        )
        soles = feet[:, None, :] + local[None, :, :]
        foot_path = single_step_foot_trajectory(
            feet,
            np.asarray((0.20, 0.10, 0.25)),
            pivot_foot=0,
            frame_count=21,
            swing_clearance_m=0.10,
        )

        path = single_step_sole_trajectory(
            soles,
            feet,
            foot_path,
            pivot_foot=0,
            yaw_delta_rad=math.pi / 2.0,
        )

        np.testing.assert_allclose(
            path[:, 0],
            np.repeat(soles[None, 0], len(path), axis=0),
        )
        initial_offsets = soles[1, :, :2] - feet[1, :2]
        terminal_offsets = (
            path[-1, 1, :, :2] - foot_path[-1, 1, :2]
        )
        np.testing.assert_allclose(
            terminal_offsets,
            initial_offsets @ np.asarray(((0.0, 1.0), (-1.0, 0.0))),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.linalg.norm(
                path[:, 1, 0] - path[:, 1, 1],
                axis=1,
            ),
            np.linalg.norm(soles[1, 0] - soles[1, 1]),
        )

    def test_single_step_connector_tracks_pivot_and_swing_with_speed_bound(self):
        class EncodedFootKinematics:
            def solve_leg_positions_bounded(
                self,
                joint_position,
                root_position_world,
                root_orientation_world_wxyz,
                solve_feet,
                target_foot_position_world,
                joint_position_lower,
                joint_position_upper,
            ):
                output = np.asarray(joint_position, dtype=np.float64).copy()
                output[:6] = np.asarray(target_foot_position_world).reshape(6)
                if (
                    np.any(output < joint_position_lower)
                    or np.any(output > joint_position_upper)
                ):
                    raise RuntimeError("test IK exceeded bound")
                return output

            def foot_positions(
                self,
                joint_positions,
                root_positions_world,
                root_orientations_world_wxyz,
            ):
                joints = np.asarray(joint_positions, dtype=np.float64)
                return joints[:, :6].reshape(-1, 2, 3)

        feet = np.asarray(
            ((0.0, 0.10, 0.20), (0.0, -0.10, 0.20)),
            dtype=np.float64,
        )
        joints = np.zeros(29, dtype=np.float64)
        joints[:6] = feet.reshape(6)
        incoming = StairConnectorBoundary(
            joints,
            np.zeros(29),
            np.asarray((0.0, 0.0, 1.0)),
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            feet,
        )

        result = synthesize_single_step_connector(
            incoming,
            swing_target_world=np.asarray((0.20, 0.10, 0.20)),
            pivot_foot=0,
            root_yaw_delta_rad=math.pi / 2,
            kinematics=EncodedFootKinematics(),
            frame_count=21,
            dt_s=0.02,
            swing_clearance_m=0.10,
            root_height_offset_m=0.10,
            maximum_joint_speed_rad_s=13.0,
            maximum_joint_acceleration_rad_s2=100.0,
            joint_position_lower=np.full(29, -10.0),
            joint_position_upper=np.full(29, 10.0),
        )

        np.testing.assert_allclose(
            result.foot_position_world[:, 0],
            np.repeat(feet[None, 0], 21, axis=0),
        )
        np.testing.assert_allclose(
            result.foot_position_world[-1, 1],
            (0.20, 0.10, 0.20),
        )
        self.assertLessEqual(result.maximum_joint_speed_rad_s, 13.0)
        self.assertLessEqual(
            result.maximum_joint_acceleration_rad_s2,
            100.0,
        )
        self.assertAlmostEqual(result.root_position_world[-1, 2], 1.10)

    def test_planar_alignment_places_both_feet_on_target_stance(self):
        source = np.asarray(
            ((0.0, 0.10), (0.0, -0.10)),
            dtype=np.float64,
        )
        yaw = math.radians(30.0)
        rotation = np.asarray(
            (
                (math.cos(yaw), math.sin(yaw)),
                (-math.sin(yaw), math.cos(yaw)),
            )
        )
        target = source @ rotation + (1.2, -0.4)

        result = planar_foot_alignment(source, target)

        np.testing.assert_allclose(result.aligned_source_xy, target, atol=1e-10)
        self.assertAlmostEqual(result.yaw_rad, yaw)
        self.assertLess(result.maximum_residual_m, 1e-10)

    def test_hermite_trajectory_preserves_boundary_position_and_velocity(self):
        start = np.asarray((0.0, 1.0), dtype=np.float64)
        end = np.asarray((2.0, -1.0), dtype=np.float64)
        start_velocity = np.asarray((0.25, -0.50), dtype=np.float64)
        end_velocity = np.asarray((-0.75, 0.10), dtype=np.float64)
        dt_s = 0.02

        trajectory, derivative = cubic_hermite_trajectory(
            start,
            end,
            start_velocity,
            end_velocity,
            frame_count=26,
            dt_s=dt_s,
        )

        np.testing.assert_allclose(trajectory[0], start)
        np.testing.assert_allclose(trajectory[-1], end)
        np.testing.assert_allclose(derivative[0], start_velocity)
        np.testing.assert_allclose(derivative[-1], end_velocity)
        self.assertEqual(trajectory.shape, (26, 2))

    def test_double_support_connector_preserves_stance_and_boundaries(self):
        class FixedFootKinematics:
            def solve_leg_positions(
                self,
                joint_position,
                root_position_world,
                root_orientation_world_wxyz,
                solve_feet,
                target_foot_position_world,
            ):
                return np.asarray(joint_position, dtype=np.float64)

            def foot_positions(
                self,
                joint_positions,
                root_positions_world,
                root_orientations_world_wxyz,
            ):
                return np.repeat(target_feet[None, :, :], len(joint_positions), axis=0)

        target_feet = np.asarray(
            ((0.0, 0.10, 0.20), (0.0, -0.10, 0.20)),
            dtype=np.float64,
        )
        incoming = StairConnectorBoundary(
            joint_position=np.zeros(29),
            joint_velocity=np.zeros(29),
            root_position_world=np.asarray((0.0, 0.0, 1.0)),
            root_velocity_world=np.zeros(3),
            root_orientation_world_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0)),
            foot_position_world=target_feet,
        )
        outgoing = StairConnectorBoundary(
            joint_position=np.full(29, 0.10),
            joint_velocity=np.zeros(29),
            root_position_world=np.asarray((0.05, 0.0, 1.0)),
            root_velocity_world=np.zeros(3),
            root_orientation_world_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0)),
            foot_position_world=target_feet,
        )

        result = synthesize_double_support_connector(
            incoming,
            outgoing,
            kinematics=FixedFootKinematics(),
            frame_count=26,
            dt_s=0.02,
            maximum_joint_speed_rad_s=13.0,
        )

        np.testing.assert_allclose(result.joint_position[0], incoming.joint_position)
        np.testing.assert_allclose(result.joint_position[-1], outgoing.joint_position)
        self.assertLess(result.maximum_foot_error_m, 1e-12)
        self.assertLessEqual(result.maximum_joint_speed_rad_s, 13.0)

    def test_double_support_ik_seed_continues_previous_solution_branch(self):
        class RecordingKinematics:
            def __init__(self):
                self.seeds = []

            def solve_leg_positions(
                self,
                joint_position,
                root_position_world,
                root_orientation_world_wxyz,
                solve_feet,
                target_foot_position_world,
            ):
                seed = np.asarray(joint_position, dtype=np.float64)
                self.seeds.append(seed.copy())
                return seed + 0.01

            def foot_positions(
                self,
                joint_positions,
                root_positions_world,
                root_orientations_world_wxyz,
            ):
                return np.repeat(feet[None, :, :], len(joint_positions), axis=0)

        feet = np.asarray(
            ((0.0, 0.10, 0.0), (0.0, -0.10, 0.0)),
            dtype=np.float64,
        )
        incoming = StairConnectorBoundary(
            np.zeros(29),
            np.zeros(29),
            np.asarray((0.0, 0.0, 1.0)),
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            feet,
        )
        outgoing = StairConnectorBoundary(
            np.full(29, 0.10),
            np.zeros(29),
            np.asarray((0.0, 0.0, 1.0)),
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            feet,
        )
        kinematics = RecordingKinematics()

        result = synthesize_double_support_connector(
            incoming,
            outgoing,
            kinematics=kinematics,
            frame_count=6,
            dt_s=0.02,
            maximum_joint_speed_rad_s=100.0,
        )

        reference, _ = cubic_hermite_trajectory(
            incoming.joint_position,
            outgoing.joint_position,
            incoming.joint_velocity,
            outgoing.joint_velocity,
            frame_count=6,
            dt_s=0.02,
        )
        np.testing.assert_allclose(
            kinematics.seeds[2],
            result.joint_position[1] + reference[2] - reference[1],
        )

    def test_double_support_projects_continuation_seed_to_joint_limits(self):
        class RecordingKinematics:
            def __init__(self):
                self.seeds = []

            def solve_leg_positions(
                self,
                joint_position,
                root_position_world,
                root_orientation_world_wxyz,
                solve_feet,
                target_foot_position_world,
            ):
                seed = np.asarray(joint_position, dtype=np.float64)
                self.seeds.append(seed.copy())
                return seed

            def foot_positions(
                self,
                joint_positions,
                root_positions_world,
                root_orientations_world_wxyz,
            ):
                return np.repeat(feet[None, :, :], len(joint_positions), axis=0)

        feet = np.asarray(
            ((0.0, 0.10, 0.0), (0.0, -0.10, 0.0)),
            dtype=np.float64,
        )
        incoming = StairConnectorBoundary(
            np.full(29, 0.04),
            np.full(29, 2.0),
            np.asarray((0.0, 0.0, 1.0)),
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            feet,
        )
        outgoing = StairConnectorBoundary(
            np.zeros(29),
            np.zeros(29),
            np.asarray((0.0, 0.0, 1.0)),
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            feet,
        )
        kinematics = RecordingKinematics()

        synthesize_double_support_connector(
            incoming,
            outgoing,
            kinematics=kinematics,
            frame_count=6,
            dt_s=0.02,
            maximum_joint_speed_rad_s=100.0,
            joint_position_lower=np.full(29, -0.05),
            joint_position_upper=np.full(29, 0.05),
        )

        self.assertLessEqual(
            float(np.max(np.asarray(kinematics.seeds))),
            0.05,
        )

    def test_double_support_passes_per_frame_speed_bounds_to_bounded_ik(self):
        class BoundedKinematics:
            def __init__(self):
                self.bounds = []

            def solve_leg_positions_bounded(
                self,
                joint_position,
                root_position_world,
                root_orientation_world_wxyz,
                solve_feet,
                target_foot_position_world,
                joint_position_lower,
                joint_position_upper,
            ):
                self.bounds.append(
                    (
                        np.asarray(joint_position_lower).copy(),
                        np.asarray(joint_position_upper).copy(),
                    )
                )
                return np.clip(
                    joint_position,
                    joint_position_lower,
                    joint_position_upper,
                )

            def foot_positions(
                self,
                joint_positions,
                root_positions_world,
                root_orientations_world_wxyz,
            ):
                return np.repeat(feet[None, :, :], len(joint_positions), axis=0)

        feet = np.asarray(
            ((0.0, 0.10, 0.0), (0.0, -0.10, 0.0)),
            dtype=np.float64,
        )
        incoming = StairConnectorBoundary(
            np.zeros(29),
            np.zeros(29),
            np.asarray((0.0, 0.0, 1.0)),
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            feet,
        )
        outgoing = StairConnectorBoundary(
            np.full(29, 0.10),
            np.zeros(29),
            np.asarray((0.0, 0.0, 1.0)),
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            feet,
        )
        kinematics = BoundedKinematics()

        result = synthesize_double_support_connector(
            incoming,
            outgoing,
            kinematics=kinematics,
            frame_count=6,
            dt_s=0.02,
            maximum_joint_speed_rad_s=13.0,
            joint_position_lower=np.full(29, -1.0),
            joint_position_upper=np.full(29, 1.0),
        )

        lower, upper = kinematics.bounds[1]
        np.testing.assert_allclose(
            lower,
            np.maximum(-1.0, result.joint_position[0] - 0.26),
        )
        np.testing.assert_allclose(
            upper,
            np.minimum(1.0, result.joint_position[0] + 0.26),
        )

    def test_bounded_foot_ik_respects_explicit_limits(self):
        class AnalyticalFeet:
            def foot_positions(
                self,
                joint_positions,
                root_positions_world,
                root_orientations_world_wxyz,
            ):
                joints = np.asarray(joint_positions)
                output = np.zeros((len(joints), 2, 3), dtype=np.float64)
                output[:, 0, 0] = joints[:, 0]
                output[:, 1, 0] = joints[:, 1]
                return output

        kinematics = BoundedFootKinematics(AnalyticalFeet())
        seed = np.zeros(29, dtype=np.float64)
        seed[2] = 0.1
        target = np.asarray(
            ((0.20, 0.0, 0.0), (-0.20, 0.0, 0.0)),
            dtype=np.float64,
        )

        solved = kinematics.solve_leg_positions_bounded(
            seed,
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.ones(2, dtype=np.bool_),
            target,
            np.full(29, -0.26),
            np.full(29, 0.26),
        )

        self.assertAlmostEqual(solved[0], 0.20, places=4)
        self.assertAlmostEqual(solved[1], -0.20, places=4)
        self.assertEqual(solved[2], 0.1)
        self.assertLessEqual(float(np.max(np.abs(solved[[0, 1]]))), 0.26)


if __name__ == "__main__":
    unittest.main()
