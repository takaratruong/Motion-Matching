import math
from pathlib import Path
import unittest

import numpy as np

from mm_sonic.joints import load_joint_contract
from mm_sonic.schema import ContractError
from mm_sonic.transform import (
    holden_to_mujoco_quaternions,
    holden_to_mujoco_vectors,
    map_source_joints,
    mujoco_to_holden_quaternions,
    mujoco_to_holden_vectors,
)


ROOT = Path(__file__).resolve().parents[2]
JOINT_CONTRACT = ROOT / "sonic" / "configs" / "g1_joint_contract.json"
REGISTERED_TERRAIN_OBJ = (
    ROOT.parent.parent
    / "resources"
    / "g1_terrain"
    / "scenes"
    / "grail-curb-low"
    / "terrain.obj"
)


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
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


class BasisVectorTests(unittest.TestCase):
    def test_all_three_holden_unit_axes_map_to_the_mujoco_basis(self):
        source = np.eye(3, dtype=np.float64)
        expected = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            np.float32,
        )
        actual = holden_to_mujoco_vectors(source)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.dtype(np.float32))
        self.assertTrue(actual.flags.owndata)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertFalse(actual.flags.writeable)

    def test_vector_conversion_is_owned_and_round_trips(self):
        source = np.array([[1.25, 2.5, -3.75], [-8.0, 13.0, 21.0]], np.float64)
        converted = holden_to_mujoco_vectors(source)
        np.testing.assert_array_equal(
            converted,
            np.array([[1.25, 3.75, 2.5], [-8.0, -21.0, 13.0]], np.float32),
        )
        source[:] = 999.0
        np.testing.assert_array_equal(
            converted,
            np.array([[1.25, 3.75, 2.5], [-8.0, -21.0, 13.0]], np.float32),
        )
        restored = mujoco_to_holden_vectors(converted)
        np.testing.assert_array_equal(
            restored,
            np.array([[1.25, 2.5, -3.75], [-8.0, 13.0, 21.0]], np.float32),
        )

    def test_vector_conversion_rejects_wrong_shape_and_nonfinite_values(self):
        for value in (
            np.zeros((2, 2), np.float64),
            np.array([0.0, float("nan"), 0.0]),
            np.array([0.0, 0.0, float("inf")]),
        ):
            with self.subTest(shape=value.shape), self.assertRaises(ContractError):
                holden_to_mujoco_vectors(value)

    def test_registered_terrain_obj_bounds_transform_independently(self):
        self.assertTrue(REGISTERED_TERRAIN_OBJ.is_file(), REGISTERED_TERRAIN_OBJ)
        vertices = []
        with REGISTERED_TERRAIN_OBJ.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("v "):
                    vertices.append([float(value) for value in line.split()[1:4]])
        holden = np.asarray(vertices, np.float64)
        mujoco = holden_to_mujoco_vectors(holden)
        holden_min = holden.min(axis=0)
        holden_max = holden.max(axis=0)
        expected_min = np.array(
            [holden_min[0], -holden_max[2], holden_min[1]], np.float32
        )
        expected_max = np.array(
            [holden_max[0], -holden_min[2], holden_max[1]], np.float32
        )
        np.testing.assert_array_equal(mujoco.min(axis=0), expected_min)
        np.testing.assert_array_equal(mujoco.max(axis=0), expected_max)


class QuaternionBasisTests(unittest.TestCase):
    def test_identity_is_preserved(self):
        actual = holden_to_mujoco_quaternions([[1.0, 0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(
            actual, np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
        )
        self.assertTrue(actual.flags.owndata)
        self.assertFalse(actual.flags.writeable)

    def test_plus_and_minus_ninety_degrees_about_every_axis(self):
        basis_axes = (
            (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])),
        )
        for source_axis, expected_axis in basis_axes:
            for angle in (math.pi / 2.0, -math.pi / 2.0):
                source = axis_angle(source_axis, angle)
                expected = axis_angle(expected_axis, angle).astype(np.float32)
                actual = holden_to_mujoco_quaternions(source)
                with self.subTest(axis=source_axis.tolist(), angle=angle):
                    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-7)

    def test_arbitrary_known_pelvis_pose(self):
        position_holden = np.array([1.25, 2.5, -3.75], np.float64)
        orientation_holden = axis_angle([1.0, 2.0, 3.0], 1.234)
        expected_position = np.array([1.25, 3.75, 2.5], np.float32)
        expected_orientation = np.array(
            [
                orientation_holden[0],
                orientation_holden[1],
                -orientation_holden[3],
                orientation_holden[2],
            ],
            np.float32,
        )
        np.testing.assert_array_equal(
            holden_to_mujoco_vectors(position_holden), expected_position
        )
        np.testing.assert_allclose(
            holden_to_mujoco_quaternions(orientation_holden),
            expected_orientation,
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_antipodes_are_unrolled_to_one_hemisphere(self):
        quaternion = axis_angle([1.0, 2.0, 3.0], 0.75)
        source = np.stack((quaternion, -quaternion, quaternion))
        converted = holden_to_mujoco_quaternions(source)
        self.assertGreaterEqual(float(np.dot(converted[0], converted[1])), 0.0)
        self.assertGreaterEqual(float(np.dot(converted[1], converted[2])), 0.0)
        np.testing.assert_array_equal(converted[0], converted[1])
        np.testing.assert_array_equal(converted[1], converted[2])

    def test_zero_and_near_zero_norms_are_rejected(self):
        for value in (
            np.zeros(4, np.float64),
            np.array([1.0e-13, 0.0, 0.0, 0.0]),
        ):
            with self.subTest(value=value), self.assertRaises(ContractError):
                holden_to_mujoco_quaternions(value)

    def test_quaternion_conversion_rejects_wrong_shape_and_nonfinite_values(self):
        for value in (
            np.zeros((2, 3), np.float64),
            np.array([1.0, float("nan"), 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, float("inf")]),
        ):
            with self.subTest(shape=value.shape), self.assertRaises(ContractError):
                holden_to_mujoco_quaternions(value)

    def test_holden_mujoco_holden_round_trip_has_small_angular_error(self):
        fixtures = np.stack(
            (
                [1.0, 0.0, 0.0, 0.0],
                axis_angle([1.0, 0.0, 0.0], math.pi / 2.0),
                axis_angle([0.0, 1.0, 0.0], -math.pi / 2.0),
                axis_angle([0.0, 0.0, 1.0], math.pi / 2.0),
                axis_angle([1.0, 2.0, 3.0], 1.234),
            )
        )
        converted = holden_to_mujoco_quaternions(fixtures)
        restored = mujoco_to_holden_quaternions(converted)
        errors = [
            quaternion_angle(source, target)
            for source, target in zip(fixtures, restored)
        ]
        self.assertLessEqual(max(errors), 5.0e-4, errors)


class JointMappingTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_joint_contract(JOINT_CONTRACT)
        self.source_names = tuple(row.source_joint for row in self.contract.rows)
        self.target_names = tuple(
            next(
                row.target_name
                for row in self.contract.rows
                if row.target_index == target_index
            )
            for target_index in range(29)
        )

    def test_mapping_uses_registered_names_and_preserves_binary32_bits(self):
        canonical = np.array(
            [
                np.nextafter(np.float32(index + 1), np.float32(np.inf))
                for index in range(29)
            ],
            np.float32,
        )
        permutation = np.array(
            [7, 0, 28, 3, 14, 1, 22, 6, 9, 5, 4, 2, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27],
            np.int64,
        )
        permuted_names = tuple(self.source_names[index] for index in permutation)
        permuted_values = canonical[permutation].astype(np.float64)
        actual = map_source_joints(permuted_values, permuted_names, self.contract)
        expected = np.empty(29, np.float32)
        by_name = dict(zip(permuted_names, np.asarray(permuted_values, np.float32)))
        for target_index, target_name in enumerate(self.target_names):
            expected[target_index] = by_name[target_name]
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        self.assertEqual(actual.dtype, np.dtype(np.float32))
        self.assertTrue(actual.flags.owndata)
        self.assertFalse(actual.flags.writeable)
        permuted_values[:] = 0.0
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_mapping_supports_boundary_matrices(self):
        values = np.arange(2 * 29, dtype=np.float64).reshape(2, 29)
        actual = map_source_joints(values, self.source_names, self.contract)
        self.assertEqual(actual.shape, (2, 29))
        for row in self.contract.rows:
            np.testing.assert_array_equal(
                actual[:, row.target_index], values[:, row.source_index].astype(np.float32)
            )

    def test_mapping_never_falls_back_to_position(self):
        cases = (
            self.source_names[:-1],
            self.source_names[:-1] + (self.source_names[0],),
            self.source_names[:-1] + ("unregistered_joint",),
        )
        for names in cases:
            with self.subTest(names=names[-2:]), self.assertRaises(ContractError):
                map_source_joints(np.zeros((1, len(names))), names, self.contract)


if __name__ == "__main__":
    unittest.main()
