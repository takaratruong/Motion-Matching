from contextlib import contextmanager
from dataclasses import fields
import os
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from resources import quat as holden_quat
from resources.g1_interaction_builder.conversion import (
    convert_interaction,
    finite_difference_quaternions,
    finite_difference_vectors,
    resample_discrete_by_source_frame,
    size_zup_to_yup,
)
from resources.g1_interaction_builder.schema import (
    ConversionValidationError,
    G1_SKELETON,
)
from resources.g1_interaction_builder.sources import load_raw_interaction
from resources.g1_terrain_builder.kinematics import (
    G1Kinematics,
    forward_local_hierarchy,
    quaternions_zup_to_yup,
    vectors_zup_to_yup,
)
from resources.g1_terrain_builder.resample import (
    resample_quaternions_wxyz,
    resample_vectors,
)
from tests.python.interaction_fixture import write_source_fixture


G1_XML = os.environ.get(
    "G1_XML",
    "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml",
)


@contextmanager
def synthetic_raw_interaction():
    with tempfile.TemporaryDirectory() as tmp:
        paths = write_source_fixture(Path(tmp))
        yield load_raw_interaction(
            paths, np.array([0.08, 0.12, 0.20], np.float32)
        )


class InteractionConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics = G1Kinematics(G1_XML)

    def test_derivatives_do_not_wrap_or_cross_a_clip(self):
        x = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], np.float32)
        v = finite_difference_vectors(x, 2.0)
        np.testing.assert_allclose(v[:, 0], [2.0, 3.0, 4.0])

    def test_quaternion_derivatives_use_one_sided_clip_endpoints(self):
        angles = np.array([0.0, 0.1, 0.3], np.float64)
        rotations = np.column_stack(
            (
                np.cos(0.5 * angles),
                np.zeros((3, 2), np.float64),
                np.sin(0.5 * angles),
            )
        )
        rotations[1] *= -1.0
        angular = finite_difference_quaternions(rotations, 2.0)
        np.testing.assert_allclose(
            angular,
            [[0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4]],
            atol=1e-6,
        )

    def test_stationary_quaternion_derivative_emits_no_runtime_warning(self):
        rotations = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0]), (3, 1)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            angular = finite_difference_quaternions(rotations, 25.0)
        np.testing.assert_array_equal(angular, 0.0)
        self.assertEqual(caught, [])

    def test_contact_resampling_uses_nearest_source_frame(self):
        source_frames = np.array([0, 0, 1, 1, 2], np.int32)
        contacts = np.array([[0, 0], [1, 0], [1, 0]], np.uint8)
        np.testing.assert_array_equal(
            resample_discrete_by_source_frame(contacts, source_frames),
            [[0, 0], [0, 0], [1, 0], [1, 0], [1, 0]],
        )

    def test_contact_resampling_rejects_out_of_range_source_frame(self):
        with self.assertRaises(ConversionValidationError) as caught:
            resample_discrete_by_source_frame(
                np.zeros((2, 2), np.uint8), np.array([0, 2], np.int32)
            )
        self.assertEqual(caught.exception.code, "frame_count_mismatch")

    def test_zup_size_axis_order_becomes_yup(self):
        np.testing.assert_allclose(
            size_zup_to_yup(np.array([1.0, 2.0, 3.0])),
            [1.0, 3.0, 2.0],
        )

    def test_native_g1_scene_converts_to_canonical_25_hz(self):
        with synthetic_raw_interaction() as raw:
            c = 2**-0.5
            raw.object_positions[:] = [1.0, 2.0, 3.0]
            raw.object_rotations[:] = [c, 0.0, 0.0, c]
            raw.table_position[:] = [1.0, 2.0, 3.0]
            raw.table_rotation[:] = [c, 0.0, 0.0, c]

            clip, skeleton, report = convert_interaction(
                raw, self.kinematics, target_fps=25.0
            )

        self.assertEqual(clip.positions.shape, (25, 31, 3))
        self.assertEqual(clip.fps, 25.0)
        self.assertEqual(skeleton.signature(), G1_SKELETON.signature())
        self.assertEqual(
            skeleton.signature(),
            "6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7",
        )
        np.testing.assert_array_equal(clip.source_frames, np.arange(25))
        np.testing.assert_allclose(
            np.linalg.norm(clip.rotations, axis=-1), 1.0, atol=1e-4
        )
        np.testing.assert_allclose(
            np.linalg.norm(clip.hand_rotations, axis=-1), 1.0, atol=1e-4
        )
        np.testing.assert_allclose(
            np.linalg.norm(clip.object_rotations, axis=-1), 1.0, atol=1e-4
        )
        for field in fields(clip):
            value = getattr(clip, field.name)
            if isinstance(value, np.ndarray):
                self.assertTrue(
                    np.all(np.isfinite(value)), msg=f"non-finite {field.name}"
                )

        self.assertEqual(report["duration_error_s"], 0.0)
        self.assertLessEqual(report["fk_max_error_m"], 0.001)
        self.assertLessEqual(report["fk_rotation_max_error_degrees"], 0.1)
        self.assertEqual(report["target_fps"], 25.0)
        self.assertEqual(report["left_hand_contact_frames"], 15)
        self.assertEqual(report["right_hand_contact_frames"], 0)

        world_positions, world_rotations = forward_local_hierarchy(
            clip.positions.astype(np.float64),
            clip.rotations.astype(np.float64),
            skeleton.parents,
        )
        np.testing.assert_allclose(
            clip.hand_positions, world_positions[:, [23, 30]], atol=1e-6
        )
        np.testing.assert_allclose(
            clip.hand_rotations, world_rotations[:, [23, 30]], atol=1e-6
        )
        np.testing.assert_array_equal(clip.hand_contacts[:10], 0)
        np.testing.assert_array_equal(clip.hand_contacts[10:, 0], 1)
        np.testing.assert_array_equal(clip.hand_contacts[:, 1], 0)
        np.testing.assert_array_equal(clip.foot_contacts, 1)

        np.testing.assert_allclose(
            clip.object_positions,
            np.tile([1.0, 3.0, -2.0], (25, 1)),
        )
        np.testing.assert_allclose(
            clip.object_rotations,
            np.tile([c, 0.0, c, 0.0], (25, 1)),
            atol=1e-6,
        )
        np.testing.assert_allclose(clip.table_position, [1.0, 3.0, -2.0])
        np.testing.assert_allclose(
            clip.table_rotation, [c, 0.0, c, 0.0], atol=1e-6
        )
        np.testing.assert_allclose(clip.table_size, [1.2, 0.05, 0.8])
        np.testing.assert_allclose(
            clip.object_dimensions, [0.08, 0.12, 0.20]
        )

    def test_all_derivative_fields_are_computed_within_the_clip(self):
        with synthetic_raw_interaction() as raw:
            frame = np.arange(25, dtype=np.float64)
            raw.qpos[:, 0] = 0.001 * frame**2
            root_angles = 0.001 * frame**2
            raw.qpos[:, 3] = np.cos(0.5 * root_angles)
            raw.qpos[:, 4:6] = 0.0
            raw.qpos[:, 6] = np.sin(0.5 * root_angles)
            raw.hand_dof[:, 0] = 0.002 * frame**2
            raw.object_positions[:, 0] = 0.003 * frame**2
            object_angles = 0.002 * frame**2
            raw.object_rotations[:, 0] = np.cos(0.5 * object_angles)
            raw.object_rotations[:, 1:3] = 0.0
            raw.object_rotations[:, 3] = np.sin(0.5 * object_angles)

            expected_hand_dof = resample_vectors(
                raw.hand_dof, raw.fps, 25.0
            )
            expected_object_positions = resample_vectors(
                vectors_zup_to_yup(raw.object_positions), raw.fps, 25.0
            )
            expected_object_rotations = resample_quaternions_wxyz(
                quaternions_zup_to_yup(raw.object_rotations), raw.fps, 25.0
            )
            clip, _, _ = convert_interaction(raw, self.kinematics)

        np.testing.assert_allclose(
            clip.velocities,
            finite_difference_vectors(clip.positions, clip.fps),
        )
        np.testing.assert_allclose(
            clip.angular_velocities,
            finite_difference_quaternions(clip.rotations, clip.fps),
        )
        np.testing.assert_allclose(
            clip.hand_dof_velocities,
            finite_difference_vectors(expected_hand_dof, clip.fps),
        )
        np.testing.assert_allclose(
            clip.object_velocities,
            finite_difference_vectors(expected_object_positions, clip.fps),
        )
        object_angular_local = finite_difference_quaternions(
            expected_object_rotations, clip.fps
        )
        np.testing.assert_allclose(
            clip.object_angular_velocities,
            holden_quat.mul_vec(
                expected_object_rotations, object_angular_local
            ).astype(np.float32),
        )
        self.assertGreater(np.max(np.abs(clip.velocities)), 0.0)
        self.assertGreater(np.max(np.abs(clip.angular_velocities)), 0.0)
        self.assertGreater(np.max(np.abs(clip.hand_dof_velocities)), 0.0)
        self.assertGreater(np.max(np.abs(clip.object_velocities)), 0.0)
        self.assertGreater(
            np.max(np.abs(clip.object_angular_velocities)), 0.0
        )

    def test_rejects_non_25_hz_canonical_conversion(self):
        with synthetic_raw_interaction() as raw:
            with self.assertRaises(ConversionValidationError) as caught:
                convert_interaction(raw, self.kinematics, target_fps=50.0)
        self.assertEqual(caught.exception.code, "fps_mismatch")

    def test_rejects_native_g1_joint_limit_violation(self):
        model = self.kinematics.model
        joint_ids = np.flatnonzero(np.asarray(model.jnt_qposadr) >= 7)
        limited_joint = next(
            int(joint_id)
            for joint_id in joint_ids
            if bool(model.jnt_limited[joint_id])
        )
        qpos_index = int(model.jnt_qposadr[limited_joint])
        upper = float(model.jnt_range[limited_joint, 1])
        with synthetic_raw_interaction() as raw:
            raw.qpos[12, qpos_index] = upper + 0.01
            with self.assertRaises(ConversionValidationError) as caught:
                convert_interaction(raw, self.kinematics)
        self.assertEqual(caught.exception.code, "joint_limit_violation")


if __name__ == "__main__":
    unittest.main()
