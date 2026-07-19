import os
import struct
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from resources.g1_terrain_builder.database import (
    ContactConfig,
    combine_clips,
    derive_contacts,
    derive_velocities,
    forward_kinematics_arrays,
    read_holden_database,
    sample_terrain_support,
    write_holden_database,
)
from resources.g1_terrain_builder.schema import (
    ArtifactSet,
    HoldenClip,
    SkeletonSpec,
)
from resources.g1_terrain_builder.terrain import FlatTerrain, StepTerrain


def _wxyz(rotation: Rotation) -> np.ndarray:
    xyzw = rotation.as_quat()
    return xyzw[..., [3, 0, 1, 2]]


def _expected_database_bytes(artifacts: ArtifactSet) -> bytes:
    arrays2 = (
        (artifacts.positions, "<f4"),
        (artifacts.velocities, "<f4"),
        (artifacts.rotations, "<f4"),
        (artifacts.angular_velocities, "<f4"),
    )
    payload = bytearray()
    for array, dtype in arrays2:
        payload.extend(struct.pack("<II", array.shape[0], array.shape[1]))
        payload.extend(np.ascontiguousarray(array, dtype=dtype).tobytes())
    for array in (
        artifacts.parents, artifacts.range_starts, artifacts.range_stops,
    ):
        payload.extend(struct.pack("<I", len(array)))
        payload.extend(np.ascontiguousarray(array, dtype="<i4").tobytes())
    payload.extend(struct.pack(
        "<II", artifacts.contacts.shape[0], artifacts.contacts.shape[1]))
    payload.extend(np.ascontiguousarray(artifacts.contacts, dtype="u1").tobytes())
    return bytes(payload)


class DatabaseBuilderTests(unittest.TestCase):
    def test_clips_become_nonoverlapping_ranges(self):
        skeleton = SkeletonSpec(
            ("Simulation", "Hips"), np.array([-1, 0], np.int32))
        artifacts = combine_clips(
            [HoldenClip.empty(3, 2), HoldenClip.empty(5, 2)], skeleton,
        )
        np.testing.assert_array_equal(artifacts.range_starts, [0, 3])
        np.testing.assert_array_equal(artifacts.range_stops, [3, 8])

    def test_source_support_samples_root_and_named_toes_in_column_order(self):
        positions = np.zeros((2, 4, 3), np.float64)
        positions[0, :, 0] = [0.25, 0.75, 0.25, 0.75]
        positions[1, :, 0] = [0.75, 0.25, 0.75, 0.25]
        support = sample_terrain_support(
            positions, StepTerrain(0.5, 0.29), 0, 2, 3)
        self.assertEqual(support.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(support, np.array([
            [0.0, 0.0, 0.29],
            [0.29, 0.29, 0.0],
        ], np.float32))

    def test_flat_source_support_is_exact_zero(self):
        positions = np.arange(45, dtype=np.float64).reshape(5, 3, 3)
        support = sample_terrain_support(
            positions, FlatTerrain(), 0, 1, 2)
        self.assertEqual(support.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(
            support, np.zeros((5, 3), np.float32))

    def test_combination_preserves_support_rows_at_clip_boundaries(self):
        first = HoldenClip.empty(3, 1)
        second = HoldenClip.empty(2, 1)
        first.terrain_features[:] = np.arange(10.0, 22.0)
        second.terrain_features[:] = np.arange(20.0, 32.0)
        first.terrain_support[:] = [1.0, 2.0, 3.0]
        second.terrain_support[:] = [4.0, 5.0, 6.0]
        artifacts = combine_clips(
            [first, second],
            SkeletonSpec(("Simulation",), np.array([-1], np.int32)),
        )
        np.testing.assert_array_equal(artifacts.terrain_features, [
            list(range(10, 22)), list(range(10, 22)), list(range(10, 22)),
            list(range(20, 32)), list(range(20, 32)),
        ])
        np.testing.assert_array_equal(artifacts.terrain_support, [
            [1, 2, 3], [1, 2, 3], [1, 2, 3],
            [4, 5, 6], [4, 5, 6],
        ])

    def test_support_rejects_bad_indices_and_nonfinite_heights(self):
        positions = np.zeros((2, 3, 3), np.float64)
        invalid_indices = (
            (0, 1, 3),
            (0, 1, 1),
            (0, 1.0, 2),
            (0, True, 2),
            (0, np.bool_(True), 2),
        )
        for indices in invalid_indices:
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(ValueError, "support bone indices"):
                    sample_terrain_support(
                        positions, FlatTerrain(), *indices)

        class BadTerrain:
            def height(self, x, z):
                return np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            sample_terrain_support(positions, BadTerrain(), 0, 1, 2)

        class Float32OverflowTerrain:
            def height(self, x, z):
                return 1e300

        with self.assertRaisesRegex(ValueError, "finite"):
            sample_terrain_support(
                positions, Float32OverflowTerrain(), 0, 1, 2)

    def test_derivatives_are_physical_and_isolated_per_clip(self):
        fps = 25.0
        positions = np.zeros((5, 1, 3), np.float64)
        positions[:, 0, 0] = np.arange(5) / fps
        angle = np.arange(5) * 0.1
        rotations = np.zeros((5, 1, 4), np.float64)
        rotations[:, 0, 0] = np.cos(angle / 2)
        rotations[:, 0, 2] = np.sin(angle / 2)
        velocity, angular = derive_velocities(positions, rotations, fps)
        np.testing.assert_allclose(velocity[:, 0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(angular[:, 0, 1], 2.5, atol=1e-5)

        shifted = positions.copy()
        shifted[:, 0, 0] += 100.0
        shifted_velocity, shifted_angular = derive_velocities(
            shifted, rotations, fps)
        clip_a, clip_b = HoldenClip.empty(5, 1), HoldenClip.empty(5, 1)
        clip_a.positions, clip_a.velocities = positions, velocity
        clip_a.rotations, clip_a.angular_velocities = rotations, angular
        clip_b.positions, clip_b.velocities = shifted, shifted_velocity
        clip_b.rotations = rotations
        clip_b.angular_velocities = shifted_angular
        artifacts = combine_clips(
            [clip_a, clip_b],
            SkeletonSpec(("Simulation",), np.array([-1], np.int32)),
        )
        self.assertAlmostEqual(artifacts.velocities[4, 0, 0], 1.0)
        self.assertAlmostEqual(artifacts.velocities[5, 0, 0], 1.0)

    def test_angular_velocity_uses_world_space_increment_order(self):
        fps = 25.0
        increment_angle = 0.04
        increment = Rotation.from_rotvec([0.0, increment_angle, 0.0])
        orientations = [Rotation.from_euler("x", 70.0, degrees=True)]
        for _ in range(4):
            orientations.append(increment * orientations[-1])
        rotations = np.stack([_wxyz(q) for q in orientations])[:, None]

        _, angular = derive_velocities(
            np.zeros((5, 1, 3), np.float64), rotations, fps)

        expected = np.tile(
            [0.0, increment_angle * fps, 0.0], (5, 1))
        np.testing.assert_allclose(angular[:, 0], expected, atol=1e-6)

    def test_derivatives_reject_bad_shapes_fps_and_quaternions(self):
        rotations = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1, 1))
        cases = (
            (np.zeros((2, 1, 3)), rotations[:2], 25.0, "three"),
            (np.zeros((3, 1, 2)), rotations, 25.0, "shape"),
            (np.zeros((3, 1, 3)), rotations[:, :, :3], 25.0, "shape"),
            (np.zeros((3, 1, 3)), rotations, 0.0, "fps"),
            (np.full((3, 1, 3), np.nan), rotations, 25.0, "finite"),
            (np.zeros((3, 1, 3)), np.zeros((3, 1, 4)), 25.0,
             "quaternion"),
        )
        for positions, quaternions, fps, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    derive_velocities(positions, quaternions, fps)

        with self.assertRaisesRegex(ValueError, "fps"):
            derive_velocities(
                np.zeros((3, 1, 3)), rotations, "twenty-five")

    def test_contacts_use_terrain_relative_height_and_speed(self):
        feet = np.zeros((5, 2, 3), np.float64)
        feet[:, 0] = [0.75, 0.31, 0.0]
        feet[:, 1, 0] = np.arange(5) * 0.02
        feet[:, 1, 1] = 0.02
        contacts = derive_contacts(
            feet, StepTerrain(0.5, 0.29), 0, 1, 25.0, ContactConfig())
        np.testing.assert_array_equal(
            contacts[:, 0], np.ones(5, np.uint8))
        np.testing.assert_array_equal(
            contacts[:, 1], np.zeros(5, np.uint8))

        feet[:, 0, 1] = 0.0
        penetrated = derive_contacts(
            feet, StepTerrain(0.5, 0.29), 0, 1, 25.0, ContactConfig())
        np.testing.assert_array_equal(
            penetrated[:, 0], np.zeros(5, np.uint8))

    def test_contact_config_and_inputs_are_validated(self):
        feet = np.zeros((5, 2, 3))
        invalid_configs = (
            (dict(speed_threshold=0.0), "speed"),
            (dict(height_threshold=np.nan), "height"),
            (dict(median_filter_frames=2), "filter"),
        )
        for kwargs, message in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    ContactConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, "foot indices"):
            derive_contacts(feet, FlatTerrain(), 0, 2, 25.0)
        with self.assertRaisesRegex(ValueError, "distinct"):
            derive_contacts(feet, FlatTerrain(), 0, 0, 25.0)
        with self.assertRaisesRegex(ValueError, "fps"):
            derive_contacts(feet, FlatTerrain(), 0, 1, np.inf)
        with self.assertRaisesRegex(ValueError, "fps"):
            derive_contacts(feet, FlatTerrain(), 0, 1, "twenty-five")
        with self.assertRaisesRegex(TypeError, "terrain"):
            derive_contacts(feet, object(), 0, 1, 25.0)

    def test_contacts_reject_nonfinite_terrain_height(self):
        class BadTerrain:
            @staticmethod
            def height(x, z):
                return np.nan

        with self.assertRaisesRegex(ValueError, "terrain.*finite"):
            derive_contacts(
                np.zeros((5, 2, 3)), BadTerrain(), 0, 1, 25.0)

    def test_contact_median_filter_fills_a_one_frame_gap(self):
        positions = np.zeros((5, 2, 3), np.float64)
        positions[:, 0, 1] = [0.0, 0.0, 0.2, 0.0, 0.0]
        positions[:, 1, 1] = 0.2
        config = ContactConfig(
            speed_threshold=1e6,
            height_threshold=0.06,
            median_filter_frames=3,
        )

        contacts = derive_contacts(
            positions, FlatTerrain(), 0, 1, 25.0, config)

        np.testing.assert_array_equal(contacts[:, 0], np.ones(5, np.uint8))
        np.testing.assert_array_equal(contacts[:, 1], np.zeros(5, np.uint8))

    def test_contact_filtering_is_isolated_at_clip_boundaries(self):
        clip_a = np.zeros((3, 2, 3), np.float64)
        clip_b = np.zeros((3, 2, 3), np.float64)
        clip_a[:, 0, 1] = [0.0, 0.0, 0.2]
        clip_b[:, 0, 1] = [0.0, 0.2, 0.2]
        clip_a[:, 1, 1] = 0.2
        clip_b[:, 1, 1] = 0.2
        config = ContactConfig(
            speed_threshold=1e6,
            height_threshold=0.06,
            median_filter_frames=3,
        )

        separate = np.concatenate([
            derive_contacts(clip_a, FlatTerrain(), 0, 1, 25.0, config),
            derive_contacts(clip_b, FlatTerrain(), 0, 1, 25.0, config),
        ])
        incorrectly_joined = derive_contacts(
            np.concatenate([clip_a, clip_b]),
            FlatTerrain(), 0, 1, 25.0, config,
        )

        np.testing.assert_array_equal(separate[:, 0], [1, 1, 0, 1, 0, 0])
        np.testing.assert_array_equal(
            incorrectly_joined[:, 0], [1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(separate[2:4, 0], [0, 1])

    def test_forward_kinematics_uses_parent_rotation(self):
        positions = np.zeros((1, 2, 3), np.float64)
        positions[:, 1, 0] = 1.0
        rotations = np.zeros((1, 2, 4), np.float64)
        rotations[:, :, 0] = 1.0
        c = 2**-0.5
        rotations[:, 0] = [c, 0, c, 0]
        gp, gq = forward_kinematics_arrays(
            positions, rotations, np.array([-1, 0], np.int32))
        np.testing.assert_allclose(gp[0, 1], [0, 0, -1], atol=1e-7)
        np.testing.assert_allclose(gq[0, 1], rotations[0, 0], atol=1e-7)

    def test_forward_kinematics_composes_a_noncommuting_three_bone_chain(self):
        local_positions = np.array([[
            [0.5, -0.25, 0.75],
            [1.0, 0.0, 0.0],
            [0.0, 1.5, 0.25],
        ]], np.float64)
        local_rotation_objects = (
            Rotation.from_euler("z", 50.0, degrees=True),
            Rotation.from_euler("x", 65.0, degrees=True),
            Rotation.from_euler("y", -40.0, degrees=True),
        )
        local_rotations = np.stack([
            _wxyz(rotation) for rotation in local_rotation_objects
        ])[None]
        parents = np.array([-1, 0, 1], np.int32)

        global_positions, global_rotations = forward_kinematics_arrays(
            local_positions, local_rotations, parents)

        expected_rotation_objects = (
            local_rotation_objects[0],
            local_rotation_objects[0] * local_rotation_objects[1],
            local_rotation_objects[0] * local_rotation_objects[1]
            * local_rotation_objects[2],
        )
        expected_rotations = np.stack([
            _wxyz(rotation) for rotation in expected_rotation_objects
        ])
        expected_positions = np.empty((3, 3), np.float64)
        expected_positions[0] = local_positions[0, 0]
        expected_positions[1] = (
            expected_positions[0]
            + expected_rotation_objects[0].apply(local_positions[0, 1])
        )
        expected_positions[2] = (
            expected_positions[1]
            + expected_rotation_objects[1].apply(local_positions[0, 2])
        )

        np.testing.assert_allclose(
            global_positions[0], expected_positions, atol=1e-7)
        orientation_alignment = np.abs(np.sum(
            global_rotations[0] * expected_rotations, axis=-1))
        np.testing.assert_allclose(orientation_alignment, 1.0, atol=1e-7)

    def test_forward_kinematics_rejects_malformed_hierarchies(self):
        positions = np.zeros((1, 2, 3), np.float64)
        rotations = np.tile([1.0, 0.0, 0.0, 0.0], (1, 2, 1))
        for parents in (
            np.array([-1]), np.array([-1, 2]), np.array([-1, -1]),
        ):
            with self.subTest(parents=parents):
                with self.assertRaisesRegex(ValueError, "parents"):
                    forward_kinematics_arrays(positions, rotations, parents)

        rotations[:, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "quaternion"):
            forward_kinematics_arrays(
                positions, rotations, np.array([-1, 0], np.int32))

    def test_combine_clips_rejects_invalid_public_inputs(self):
        skeleton = SkeletonSpec(
            ("Simulation", "Hips"), np.array([-1, 0], np.int32))
        with self.assertRaisesRegex(ValueError, "at least one clip"):
            combine_clips([], skeleton)
        with self.assertRaisesRegex(ValueError, "bone count"):
            combine_clips([HoldenClip.empty(3, 1)], skeleton)
        with self.assertRaisesRegex(ValueError, "names.*parents"):
            combine_clips(
                [HoldenClip.empty(3, 2)],
                SkeletonSpec(("Simulation",), np.array([-1, 0], np.int32)),
            )
        with self.assertRaisesRegex(ValueError, "parents"):
            combine_clips(
                [HoldenClip.empty(3, 2)],
                SkeletonSpec(
                    ("Simulation", "Hips"), np.array([-1, -1], np.int32)),
            )

    def test_holden_binary_is_exact_little_endian_and_round_trips(self):
        skeleton = SkeletonSpec(
            ("Simulation", "Hips"), np.array([-1, 0], np.int32))
        artifacts = combine_clips([HoldenClip.empty(4, 2)], skeleton)
        artifacts.positions[:] = np.arange(
            artifacts.positions.size, dtype=np.float32).reshape(
                artifacts.positions.shape) / 10.0
        artifacts.velocities[:] = -artifacts.positions
        artifacts.rotations[:] = np.arange(
            artifacts.rotations.size, dtype=np.float32).reshape(
                artifacts.rotations.shape) / 7.0
        artifacts.angular_velocities[:] = np.arange(
            artifacts.angular_velocities.size, dtype=np.float32).reshape(
                artifacts.angular_velocities.shape) / -11.0
        artifacts.contacts[:, 0] = [0, 1, 0, 1]
        artifacts.terrain_support[:] = np.arange(
            artifacts.terrain_support.size, dtype=np.float32).reshape(
                artifacts.terrain_support.shape)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "database.bin")
            write_holden_database(path, artifacts)
            with open(path, "rb") as stream:
                payload = stream.read()
            loaded = read_holden_database(path)

        self.assertEqual(payload, _expected_database_bytes(artifacts))
        self.assertEqual(struct.unpack("<II", payload[:8]), (4, 2))
        for name in (
            "positions", "velocities", "rotations", "angular_velocities",
            "parents", "range_starts", "range_stops", "contacts",
        ):
            np.testing.assert_array_equal(
                getattr(loaded, name), getattr(artifacts, name))
        self.assertEqual(loaded.positions.dtype, np.dtype("<f4"))
        self.assertEqual(loaded.parents.dtype, np.dtype("<i4"))
        np.testing.assert_array_equal(
            loaded.terrain_features, np.zeros((4, 12), np.float32))
        np.testing.assert_array_equal(
            loaded.terrain_support, np.zeros((4, 3), np.float32))

    def test_holden_writer_rejects_legacy_four_column_terrain_features(self):
        artifacts = ArtifactSet.empty(4, 2)
        artifacts.terrain_features = np.zeros((4, 4), np.float32)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "terrain_features.*12"):
                write_holden_database(
                    os.path.join(td, "database.bin"), artifacts)

    def test_holden_writer_rejects_misaligned_arrays(self):
        artifacts = ArtifactSet.empty(4, 2)
        artifacts.velocities = np.zeros((3, 2, 3), np.float32)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "velocities"):
                write_holden_database(
                    os.path.join(td, "database.bin"), artifacts)

    def test_holden_writer_rejects_invalid_ranges_and_contacts(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "database.bin")
            artifacts = ArtifactSet.empty(4, 2)
            artifacts.range_starts = np.array([0, 3], np.int32)
            artifacts.range_stops = np.array([2, 4], np.int32)
            with self.assertRaisesRegex(ValueError, "contiguous"):
                write_holden_database(path, artifacts)

            artifacts = ArtifactSet.empty(4, 2)
            artifacts.contacts[0, 0] = 2
            with self.assertRaisesRegex(ValueError, "0 or 1"):
                write_holden_database(path, artifacts)

    def test_holden_reader_rejects_truncation_trailing_and_bad_shapes(self):
        skeleton = SkeletonSpec(("Simulation",), np.array([-1], np.int32))
        artifacts = combine_clips([HoldenClip.empty(4, 1)], skeleton)
        with tempfile.TemporaryDirectory() as td:
            valid = os.path.join(td, "valid.bin")
            write_holden_database(valid, artifacts)
            with open(valid, "rb") as stream:
                payload = stream.read()
            position_bytes = artifacts.positions.nbytes
            short_positions = (
                struct.pack("<II", 3, 1)
                + np.ascontiguousarray(
                    artifacts.positions[:3], dtype="<f4").tobytes()
                + payload[8 + position_bytes:]
            )
            corruptions = (
                ("truncated.bin", payload[:-1], "truncated"),
                ("trailing.bin", payload + b"x", "trailing"),
                ("bad-shape.bin", short_positions,
                 "positions|frames|shape"),
            )
            for name, corrupt, message in corruptions:
                path = os.path.join(td, name)
                with open(path, "wb") as stream:
                    stream.write(corrupt)
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, message):
                        read_holden_database(path)


if __name__ == "__main__":
    unittest.main()
