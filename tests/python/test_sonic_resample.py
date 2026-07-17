from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from types import MappingProxyType
import unittest

import numpy as np

from mm_sonic.joints import ContractError, JointContract, load_joint_contract
from mm_sonic.resample import (
    ResampledSourceChunk,
    hermite_pair,
    resample_source_chunk,
    shortest_path_slerp,
)
from mm_sonic.schema import InitialBoundary, SourceChunk
from mm_sonic.transform import (
    holden_to_mujoco_quaternions,
    holden_to_mujoco_vectors,
    map_source_joints,
)


ROOT = Path(__file__).resolve().parents[2]
JOINT_CONTRACT = ROOT / "sonic" / "configs" / "g1_joint_contract.json"


def readonly(value, dtype=np.float32):
    output = np.ascontiguousarray(value, dtype=dtype).copy(order="C")
    output.flags.writeable = False
    return output


def axis_angle(axis, angle):
    axis = np.asarray(axis, np.float64)
    axis /= np.linalg.norm(axis)
    return np.concatenate(
        ([math.cos(angle / 2.0)], axis * math.sin(angle / 2.0))
    )


def quaternion_angle(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    dot = abs(float(np.dot(left, right)))
    cross_norm = np.linalg.norm(
        left[0] * right[1:]
        - right[0] * left[1:]
        - np.cross(left[1:], right[1:])
    )
    relative_w = abs(
        left[0] * right[0] + float(np.dot(left[1:], right[1:]))
    )
    return 2.0 * math.atan2(float(cross_norm), relative_w) if dot < 1.0 else 0.0


def target_names(contract: JointContract):
    names = [None] * 29
    for row in contract.rows:
        names[row.target_index] = row.target_name
    return tuple(names)


def make_source_chunk(
    contract: JointContract,
    *,
    candidate_id="candidate-1",
    predecessor_id=None,
    start_s=0.0,
):
    local_time = np.arange(11, dtype=np.float64) / 25.0
    global_time = start_s + local_time
    target_velocity = np.linspace(-0.028, 0.028, 29, dtype=np.float64)
    target_position = global_time[:, np.newaxis] * target_velocity[np.newaxis, :]
    target_velocity_rows = np.broadcast_to(target_velocity, (11, 29)).copy()
    source_position = np.empty((11, 29), np.float64)
    source_velocity = np.empty((11, 29), np.float64)
    for row in contract.rows:
        source_position[:, row.source_index] = target_position[:, row.target_index]
        source_velocity[:, row.source_index] = target_velocity_rows[:, row.target_index]

    physical_position = np.stack(
        (global_time, 1.0 + 2.0 * global_time, -0.5 * global_time), axis=1
    )
    virtual_position = np.stack(
        (2.0 * global_time, 0.25 * global_time, 3.0 - 0.1 * global_time),
        axis=1,
    )
    physical_orientation = np.stack(
        [axis_angle([0.0, 1.0, 0.0], 0.1 * time) for time in global_time]
    )
    virtual_orientation = np.stack(
        [axis_angle([0.0, 0.0, 1.0], -0.15 * time) for time in global_time]
    )
    scene = MappingProxyType(
        {
            "scene_id": "sonic-flat-baseline",
            "route_id": "direct",
            "terrain_weight": np.float32(0.0),
            "coordinate_signature": "holden-y-up-right-handed-forward-plus-z",
            "heightfield_sha256": "1" * 64,
            "mesh_sha256": "2" * 64,
            "walkability_sha256": "3" * 64,
        }
    )
    command = MappingProxyType(
        {
            "requested_velocity_holden": readonly([0.0, 0.0, 0.5]),
            "desired_heading_holden_wxyz": readonly([1.0, 0.0, 0.0, 0.0]),
            "applied_velocity_holden": readonly(np.zeros((10, 3))),
        }
    )
    artifacts = MappingProxyType(
        {
            "build_commit": "4ac89a2",
            "joint_contract_sha256": "4" * 64,
            "motion_manifest_sha256": "5" * 64,
            "database_sha256": "6" * 64,
            "terrain_features_sha256": "7" * 64,
            "terrain_support_sha256": "8" * 64,
            "scene_index_sha256": "9" * 64,
            "skeleton_signature": "a" * 64,
            "coordinate_signature": "holden-y-up-right-handed-forward-plus-z",
        }
    )
    return SourceChunk(
        session_id="session-0",
        candidate_id=candidate_id,
        predecessor_id=predecessor_id,
        source_rate_hz=25,
        source_intervals=10,
        timestamps_s=readonly(local_time),
        source_joint_names=tuple(row.source_joint for row in contract.rows),
        target_joint_names=target_names(contract),
        joint_position_source=readonly(source_position),
        joint_velocity_source=readonly(source_velocity),
        physical_pelvis_position_holden=readonly(physical_position),
        physical_pelvis_orientation_holden=readonly(physical_orientation),
        virtual_root_position_holden=readonly(virtual_position),
        virtual_root_orientation_holden=readonly(virtual_orientation),
        selected_database_frame=readonly(np.arange(10), np.int64),
        candidate_preview_count=readonly(np.zeros(10), np.int64),
        candidate_limit_rejection_count=readonly(np.zeros(10), np.int64),
        first_rejected_database_frame=readonly(-np.ones(10), np.int64),
        first_rejected_joint_index=readonly(-np.ones(10), np.int64),
        first_rejected_joint_position=readonly(np.zeros(10)),
        searched=readonly(np.zeros(10), np.bool_),
        transitioned=readonly(np.zeros(10), np.bool_),
        terrain_cost=readonly(np.zeros(10)),
        terrain_values=readonly(np.zeros((10, 4))),
        terrain_points_holden=readonly(np.zeros((10, 4, 3))),
        support_height=readonly(np.zeros(10)),
        support_target=readonly(np.zeros(10)),
        scene=scene,
        command=command,
        artifacts=artifacts,
    )


def initial_from_chunk(chunk: SourceChunk):
    return InitialBoundary(
        session_id=chunk.session_id,
        source_joint_names=chunk.source_joint_names,
        joint_position_source=readonly(chunk.joint_position_source[0]),
        joint_velocity_source=readonly(chunk.joint_velocity_source[0]),
        physical_pelvis_position_holden=readonly(
            chunk.physical_pelvis_position_holden[0]
        ),
        physical_pelvis_orientation_holden=readonly(
            chunk.physical_pelvis_orientation_holden[0]
        ),
        virtual_root_position_holden=readonly(chunk.virtual_root_position_holden[0]),
        virtual_root_orientation_holden=readonly(
            chunk.virtual_root_orientation_holden[0]
        ),
    )


class HermiteAnalyticTests(unittest.TestCase):
    def test_constant_and_linear_trajectories_are_exact(self):
        q0 = np.array([2.0, -3.0], np.float32)
        constant_q, constant_v = hermite_pair(
            q0, np.zeros(2), q0, np.zeros(2), 0.04, 0.5
        )
        np.testing.assert_array_equal(constant_q, q0.astype(np.float64))
        np.testing.assert_array_equal(constant_v, np.zeros(2, np.float64))

        velocity = np.array([1.25, -2.5], np.float64)
        right = q0.astype(np.float64) + 0.04 * velocity
        for u in (0.0, 0.25, 0.5, 1.0):
            position, derivative = hermite_pair(
                q0, velocity, right, velocity, 0.04, u
            )
            np.testing.assert_allclose(
                position,
                q0.astype(np.float64) + u * 0.04 * velocity,
                rtol=0.0,
                atol=5.0e-16,
            )
            np.testing.assert_allclose(
                derivative, velocity, rtol=0.0, atol=2.0e-14
            )
            self.assertEqual(position.dtype, np.dtype(np.float64))
            self.assertEqual(derivative.dtype, np.dtype(np.float64))

    def test_cubic_positions_and_analytic_derivatives_match(self):
        coefficients = np.array(
            [[0.2, -0.3], [1.0, 0.5], [-2.0, 3.0], [4.0, -5.0]],
            np.float64,
        )

        def evaluate(time):
            a, b, c, d = coefficients
            return a + b * time + c * time**2 + d * time**3

        def derivative(time):
            _, b, c, d = coefficients
            return b + 2.0 * c * time + 3.0 * d * time**2

        dt = 0.04
        for u in (0.0, 0.125, 0.5, 0.875, 1.0):
            position, velocity = hermite_pair(
                evaluate(0.0),
                derivative(0.0),
                evaluate(dt),
                derivative(dt),
                dt,
                u,
            )
            np.testing.assert_allclose(
                position, evaluate(u * dt), rtol=0.0, atol=2.0e-16
            )
            np.testing.assert_allclose(
                velocity, derivative(u * dt), rtol=0.0, atol=3.0e-14
            )

    def test_endpoint_derivatives_and_secondary_finite_difference_agree(self):
        q0 = np.array([-0.3, 0.7], np.float64)
        q1 = np.array([0.9, -0.2], np.float64)
        v0 = np.array([1.1, -2.3], np.float64)
        v1 = np.array([-0.4, 3.2], np.float64)
        dt = 0.04
        left_q, left_v = hermite_pair(q0, v0, q1, v1, dt, 0.0)
        right_q, right_v = hermite_pair(q0, v0, q1, v1, dt, 1.0)
        np.testing.assert_array_equal(left_q, q0)
        np.testing.assert_array_equal(left_v, v0)
        np.testing.assert_array_equal(right_q, q1)
        np.testing.assert_array_equal(right_v, v1)

        epsilon = 1.0e-6
        before, _ = hermite_pair(q0, v0, q1, v1, dt, 0.5 - epsilon)
        after, analytic = hermite_pair(q0, v0, q1, v1, dt, 0.5 + epsilon)
        _, midpoint_velocity = hermite_pair(q0, v0, q1, v1, dt, 0.5)
        finite_difference = (after - before) / (2.0 * epsilon * dt)
        np.testing.assert_allclose(
            finite_difference, midpoint_velocity, rtol=2.0e-10, atol=2.0e-9
        )
        self.assertTrue(np.all(np.isfinite(analytic)))

    def test_invalid_time_parameters_are_rejected(self):
        values = np.zeros(1)
        for dt, u in ((0.0, 0.5), (-0.04, 0.5), (0.04, -0.1), (0.04, 1.1)):
            with self.subTest(dt=dt, u=u), self.assertRaises(ContractError):
                hermite_pair(values, values, values, values, dt, u)


class SlerpTests(unittest.TestCase):
    def test_antipodal_quaternions_follow_the_zero_length_shortest_path(self):
        quaternion = axis_angle([1.0, 2.0, 3.0], 1.2)
        for u in (0.0, 0.25, 0.5, 1.0):
            actual = shortest_path_slerp(quaternion, -quaternion, u)
            self.assertLessEqual(quaternion_angle(actual, quaternion), 2.0e-15)
            self.assertAlmostEqual(float(np.linalg.norm(actual)), 1.0, places=15)

    def test_near_identical_quaternions_are_finite_normalized_and_accurate(self):
        left = axis_angle([0.0, 0.0, 1.0], 0.25)
        right = axis_angle([0.0, 0.0, 1.0], 0.25 + 1.0e-9)
        midpoint = shortest_path_slerp(left, right, 0.5)
        expected = axis_angle([0.0, 0.0, 1.0], 0.25 + 0.5e-9)
        self.assertTrue(np.all(np.isfinite(midpoint)))
        self.assertAlmostEqual(float(np.linalg.norm(midpoint)), 1.0, places=15)
        self.assertLessEqual(quaternion_angle(midpoint, expected), 2.0e-15)

    def test_slerp_endpoints_and_half_rotation_are_analytic(self):
        left = axis_angle([1.0, 0.0, 0.0], -0.2)
        right = axis_angle([1.0, 0.0, 0.0], 0.6)
        self.assertLessEqual(
            quaternion_angle(shortest_path_slerp(left, right, 0.0), left),
            2.0e-15,
        )
        self.assertLessEqual(
            quaternion_angle(shortest_path_slerp(left, right, 1.0), right),
            2.0e-15,
        )
        self.assertLessEqual(
            quaternion_angle(
                shortest_path_slerp(left, right, 0.5),
                axis_angle([1.0, 0.0, 0.0], 0.2),
            ),
            2.0e-15,
        )


class SourceChunkResamplingTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_joint_contract(JOINT_CONTRACT)
        self.chunk = make_source_chunk(self.contract)

    def test_eleven_boundaries_produce_midpoint_and_right_endpoint_only(self):
        actual = resample_source_chunk(self.chunk, self.contract)
        self.assertIsInstance(actual, ResampledSourceChunk)
        self.assertEqual(actual.joint_position.shape, (20, 29))
        self.assertEqual(actual.joint_velocity.shape, (20, 29))
        self.assertEqual(actual.body_quat_w.shape, (20, 4))
        self.assertEqual(actual.physical_pelvis_position.shape, (20, 3))
        self.assertEqual(actual.virtual_root_position.shape, (20, 3))
        self.assertEqual(actual.virtual_root_quat_w.shape, (20, 4))
        for field in (
            "joint_position",
            "joint_velocity",
            "body_quat_w",
            "physical_pelvis_position",
            "virtual_root_position",
            "virtual_root_quat_w",
        ):
            value = getattr(actual, field)
            self.assertEqual(value.dtype, np.dtype(np.float32), field)
            self.assertTrue(value.flags.owndata, field)
            self.assertTrue(value.flags.c_contiguous, field)
            self.assertFalse(value.flags.writeable, field)
        expected_left = map_source_joints(
            self.chunk.joint_position_source[:1],
            self.chunk.source_joint_names,
            self.contract,
        )[0]
        self.assertFalse(np.array_equal(actual.joint_position[0], expected_left))

    def test_all_source_right_endpoints_recover_exact_binary32_bits(self):
        actual = resample_source_chunk(self.chunk, self.contract)
        expected_position = map_source_joints(
            self.chunk.joint_position_source,
            self.chunk.source_joint_names,
            self.contract,
        )[1:]
        expected_velocity = map_source_joints(
            self.chunk.joint_velocity_source,
            self.chunk.source_joint_names,
            self.contract,
        )[1:]
        np.testing.assert_array_equal(
            actual.joint_position[1::2].view(np.uint32),
            expected_position.view(np.uint32),
        )
        np.testing.assert_array_equal(
            actual.joint_velocity[1::2].view(np.uint32),
            expected_velocity.view(np.uint32),
        )

    def test_midpoint_uses_binary32_source_interval_promoted_to_float64(self):
        row = next(row for row in self.contract.rows if row.target_index == 0)
        # Fixed binary32 endpoints/derivatives and fixed uint32 oracles.  The
        # expected midpoint uses dt bits 0x3d23d70a promoted to binary64; a
        # binary64 0.04 literal instead produces q=0xbc70d1a4, v=0x417da97c.
        fixture = np.array(
            [0xBEA78EB6, 0x41147FFC, 0x3D7E2899, 0xC164CD2F],
            dtype=np.uint32,
        ).view(np.float32)
        position = self.chunk.joint_position_source.copy()
        velocity = self.chunk.joint_velocity_source.copy()
        position[0, row.source_index] = fixture[0]
        velocity[0, row.source_index] = fixture[1]
        position[1, row.source_index] = fixture[2]
        velocity[1, row.source_index] = fixture[3]
        candidate = replace(
            self.chunk,
            joint_position_source=readonly(position),
            joint_velocity_source=readonly(velocity),
        )

        actual = resample_source_chunk(candidate, self.contract)

        self.assertEqual(
            int(actual.joint_position[0, row.target_index].view(np.uint32)),
            0xBC70D1A7,
        )
        self.assertEqual(
            int(actual.joint_velocity[0, row.target_index].view(np.uint32)),
            0x417DA97D,
        )

    def test_right_endpoints_copy_signed_zero_position_and_velocity_bits(self):
        row = next(row for row in self.contract.rows if row.target_index == 0)
        negative_zero = np.array([0x80000000], dtype=np.uint32).view(np.float32)[0]
        position = self.chunk.joint_position_source.copy()
        velocity = self.chunk.joint_velocity_source.copy()
        position[0, row.source_index] = np.float32(0.25)
        velocity[0, row.source_index] = np.float32(0.5)
        position[1, row.source_index] = negative_zero
        velocity[1, row.source_index] = negative_zero
        candidate = replace(
            self.chunk,
            joint_position_source=readonly(position),
            joint_velocity_source=readonly(velocity),
        )
        mapped_position = map_source_joints(
            candidate.joint_position_source,
            candidate.source_joint_names,
            self.contract,
        )
        mapped_velocity = map_source_joints(
            candidate.joint_velocity_source,
            candidate.source_joint_names,
            self.contract,
        )
        self.assertEqual(
            int(mapped_position[1, row.target_index].view(np.uint32)),
            0x80000000,
        )
        self.assertEqual(
            int(mapped_velocity[1, row.target_index].view(np.uint32)),
            0x80000000,
        )

        actual = resample_source_chunk(candidate, self.contract)

        self.assertEqual(
            int(actual.joint_position[1, row.target_index].view(np.uint32)),
            0x80000000,
        )
        self.assertEqual(
            int(actual.joint_velocity[1, row.target_index].view(np.uint32)),
            0x80000000,
        )

    def test_positions_are_linear_and_orientations_use_shortest_path_slerp(self):
        actual = resample_source_chunk(self.chunk, self.contract)
        physical = holden_to_mujoco_vectors(
            self.chunk.physical_pelvis_position_holden
        )
        virtual = holden_to_mujoco_vectors(self.chunk.virtual_root_position_holden)
        expected_physical = np.empty((20, 3), np.float32)
        expected_virtual = np.empty((20, 3), np.float32)
        for interval in range(10):
            expected_physical[2 * interval] = (
                physical[interval].astype(np.float64)
                + physical[interval + 1].astype(np.float64)
            ) / 2.0
            expected_physical[2 * interval + 1] = physical[interval + 1]
            expected_virtual[2 * interval] = (
                virtual[interval].astype(np.float64)
                + virtual[interval + 1].astype(np.float64)
            ) / 2.0
            expected_virtual[2 * interval + 1] = virtual[interval + 1]
        np.testing.assert_array_equal(
            actual.physical_pelvis_position, expected_physical
        )
        np.testing.assert_array_equal(actual.virtual_root_position, expected_virtual)

        physical_quat = holden_to_mujoco_quaternions(
            self.chunk.physical_pelvis_orientation_holden
        )
        virtual_quat = holden_to_mujoco_quaternions(
            self.chunk.virtual_root_orientation_holden
        )
        for interval in range(10):
            self.assertLessEqual(
                quaternion_angle(
                    actual.body_quat_w[2 * interval],
                    shortest_path_slerp(
                        physical_quat[interval], physical_quat[interval + 1], 0.5
                    ),
                ),
                5.0e-7,
            )
            self.assertLessEqual(
                quaternion_angle(
                    actual.virtual_root_quat_w[2 * interval],
                    shortest_path_slerp(
                        virtual_quat[interval], virtual_quat[interval + 1], 0.5
                    ),
                ),
                5.0e-7,
            )
        np.testing.assert_allclose(
            np.linalg.norm(actual.body_quat_w.astype(np.float64), axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_joint_limit_overshoot_is_rejected_without_clipping(self):
        row = next(row for row in self.contract.rows if row.target_index == 17)
        position = self.chunk.joint_position_source.copy()
        velocity = self.chunk.joint_velocity_source.copy()
        position[:2, row.source_index] = 0.0
        velocity[0, row.source_index] = 30.0
        velocity[1, row.source_index] = -30.0
        overshooting = replace(
            self.chunk,
            joint_position_source=readonly(position),
            joint_velocity_source=readonly(velocity),
        )
        with self.assertRaisesRegex(ContractError, "joint limit"):
            resample_source_chunk(overshooting, self.contract)

    def test_malformed_direct_models_fail_closed(self):
        cases = (
            replace(self.chunk, source_rate_hz=50),
            replace(self.chunk, source_intervals=9),
            replace(self.chunk, joint_position_source=readonly(np.zeros((10, 29)))),
            replace(
                self.chunk,
                physical_pelvis_position_holden=readonly(np.zeros((11, 2))),
            ),
        )
        for chunk in cases:
            with self.subTest(rate=chunk.source_rate_hz), self.assertRaises(
                ContractError
            ):
                resample_source_chunk(chunk, self.contract)


if __name__ == "__main__":
    unittest.main()
