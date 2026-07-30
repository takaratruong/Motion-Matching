import tempfile
import unittest
from pathlib import Path

import numpy as np

from mm_sonic.schema import ContractError
from mm_sonic.torch_motion_data import (
    G1_TAKARA_LAYOUT,
    MotionClip,
    MotionFolder,
    discover_motion_paths,
)

from tests.python.torch_motion_test_utils import (
    build_takara_arrays,
    write_takara_arrays,
    write_takara_clip,
)


class DiscoverMotionPathsTests(unittest.TestCase):
    def test_discovers_nested_clips_in_relative_path_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_clip(root / "b")
            write_takara_clip(root / "a" / "deep")
            write_takara_clip(root / "a" / "shallow")

            paths = discover_motion_paths(root)

            relative = [p.relative_to(root).as_posix() for p in paths]
            self.assertEqual(
                relative,
                [
                    "a/deep/motion.npz",
                    "a/shallow/motion.npz",
                    "b/motion.npz",
                ],
            )


class MotionFolderLoadTests(unittest.TestCase):
    def test_loads_exact_layout_and_makes_owned_arrays_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = build_takara_arrays(frames=60)
            write_takara_arrays(root / "walk", source)

            folder = MotionFolder.load(root)

            self.assertEqual(folder.layout, G1_TAKARA_LAYOUT)
            self.assertEqual(len(folder.clips), 1)
            clip = folder.clips[0]
            self.assertIsInstance(clip, MotionClip)
            self.assertEqual(clip.relative_path, "walk/motion.npz")
            self.assertEqual(clip.fps, 50)
            self.assertEqual(clip.frame_count, 60)
            self.assertEqual(clip.valid_frame_stop, 15)
            self.assertEqual(clip.joint_position.shape, (60, 29))
            self.assertEqual(clip.joint_velocity.shape, (60, 29))
            self.assertEqual(clip.body_position_world.shape, (60, 30, 3))
            self.assertEqual(clip.body_quaternion_world_wxyz.shape, (60, 30, 4))
            self.assertEqual(clip.body_linear_velocity_world.shape, (60, 30, 3))
            self.assertEqual(clip.body_angular_velocity_world.shape, (60, 30, 3))
            np.testing.assert_array_equal(clip.joint_position, source["joint_pos"])
            np.testing.assert_array_equal(clip.joint_velocity, source["joint_vel"])
            np.testing.assert_array_equal(
                clip.body_position_world, source["body_pos_w"]
            )
            np.testing.assert_allclose(
                clip.body_quaternion_world_wxyz,
                source["body_quat_w"],
                rtol=0.0,
                atol=1e-7,
            )
            np.testing.assert_array_equal(
                clip.body_linear_velocity_world, source["body_lin_vel_w"]
            )
            np.testing.assert_array_equal(
                clip.body_angular_velocity_world, source["body_ang_vel_w"]
            )
            for array in (
                clip.joint_position,
                clip.joint_velocity,
                clip.body_position_world,
                clip.body_quaternion_world_wxyz,
                clip.body_linear_velocity_world,
                clip.body_angular_velocity_world,
            ):
                self.assertEqual(array.dtype, np.float32)
                self.assertTrue(array.flags["C_CONTIGUOUS"])
                self.assertFalse(array.flags["WRITEABLE"])
                self.assertTrue(array.flags["OWNDATA"])
                with self.assertRaises(ValueError):
                    array[0] = 0.0

    def test_rejects_each_missing_field_with_relative_path(self):
        fields = (
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        for field in fields:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    arrays = build_takara_arrays(frames=60)
                    del arrays[field]
                    write_takara_arrays(root / "clip", arrays)

                    with self.assertRaisesRegex(
                        ContractError, r"clip/motion\.npz.*" + field
                    ):
                        MotionFolder.load(root)

    def test_rejects_wrong_fps_shapes_lengths_short_clip_and_nonfinite(self):
        def bad(mutate, pattern):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                arrays = build_takara_arrays(frames=60)
                mutate(arrays)
                write_takara_arrays(root / "clip", arrays)
                with self.assertRaisesRegex(ContractError, pattern):
                    MotionFolder.load(root)

        # Wrong FPS value.
        bad(
            lambda a: a.__setitem__("fps", np.array([49], dtype=np.float32)),
            r"clip/motion\.npz.*fps",
        )
        # FPS not scalar.
        bad(
            lambda a: a.__setitem__("fps", np.array([50, 50], dtype=np.float32)),
            r"clip/motion\.npz.*fps",
        )
        # Wrong trailing joint dimension.
        bad(
            lambda a: a.__setitem__(
                "joint_pos", a["joint_pos"][:, :28].copy()
            ),
            r"clip/motion\.npz.*joint_pos",
        )
        # Wrong trailing body dimension.
        bad(
            lambda a: a.__setitem__(
                "body_pos_w", a["body_pos_w"][:, :29, :].copy()
            ),
            r"clip/motion\.npz.*body_pos_w",
        )
        # Wrong quaternion trailing dimension.
        bad(
            lambda a: a.__setitem__(
                "body_quat_w", a["body_quat_w"][:, :, :3].copy()
            ),
            r"clip/motion\.npz.*body_quat_w",
        )
        # Wrong rank.
        bad(
            lambda a: a.__setitem__(
                "joint_pos", a["joint_pos"][:, :, None].copy()
            ),
            r"clip/motion\.npz.*joint_pos",
        )
        # Mismatched frame counts.
        bad(
            lambda a: a.__setitem__(
                "joint_vel", a["joint_vel"][:50].copy()
            ),
            r"clip/motion\.npz.*joint_vel",
        )
        # Too few frames.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays = build_takara_arrays(frames=45)
            write_takara_arrays(root / "clip", arrays)
            with self.assertRaisesRegex(
                ContractError, r"clip/motion\.npz.*frame"
            ):
                MotionFolder.load(root)
        # Non-finite value.
        bad(
            lambda a: a["body_pos_w"].__setitem__((0, 0, 0), np.inf),
            r"clip/motion\.npz.*body_pos_w",
        )
        bad(
            lambda a: a["joint_vel"].__setitem__((3, 5), np.nan),
            r"clip/motion\.npz.*joint_vel",
        )

    def test_malformed_archives_and_non_real_fields_fail_closed(self):
        def rejected(arrays, pattern):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_takara_arrays(root / "clip", arrays)
                with self.assertRaisesRegex(ContractError, pattern):
                    MotionFolder.load(root)

        string_values = build_takara_arrays(frames=60)
        string_values["joint_pos"] = string_values["joint_pos"].astype(str)
        rejected(string_values, r"clip/motion\.npz.*joint_pos")

        object_values = build_takara_arrays(frames=60)
        object_values["joint_pos"] = object_values["joint_pos"].astype(object)
        rejected(object_values, r"clip/motion\.npz.*joint_pos")

        complex_values = build_takara_arrays(frames=60)
        complex_joint_pos = complex_values["joint_pos"].astype(np.complex64)
        complex_joint_pos[0, 0] += np.complex64(1j)
        complex_values["joint_pos"] = complex_joint_pos
        rejected(complex_values, r"clip/motion\.npz.*joint_pos")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed = root / "broken" / "motion.npz"
            malformed.parent.mkdir(parents=True)
            malformed.write_bytes(b"not an npz archive")
            with self.assertRaisesRegex(
                ContractError, r"broken/motion\.npz.*archive"
            ):
                MotionFolder.load(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "directory" / "motion.npz"
            directory.mkdir(parents=True)
            with self.assertRaisesRegex(
                ContractError, r"directory/motion\.npz.*archive"
            ):
                MotionFolder.load(root)

    def test_rejects_bad_quaternion_norm_and_canonicalizes_sign(self):
        # Norm far from one is rejected with the field name.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays = build_takara_arrays(frames=60)
            arrays["body_quat_w"][0, 0, :] = np.array(
                [2.0, 0.0, 0.0, 0.0], dtype=np.float32
            )
            write_takara_arrays(root / "clip", arrays)
            with self.assertRaisesRegex(
                ContractError, r"clip/motion\.npz.*body_quat_w"
            ):
                MotionFolder.load(root)

        # Temporal sign flips are canonicalized: consecutive dots stay >= 0.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays = build_takara_arrays(frames=60, yaw_rate=0.5)
            # Force a sign flip on a subset of frames for one body.
            arrays["body_quat_w"][1, 5, :] *= -1.0
            arrays["body_quat_w"][2, 5, :] *= -1.0
            write_takara_arrays(root / "clip", arrays)

            folder = MotionFolder.load(root)
            quat = folder.clips[0].body_quaternion_world_wxyz
            dots = np.sum(quat[1:] * quat[:-1], axis=-1)
            self.assertTrue(np.all(dots >= -1e-6))
            norms = np.linalg.norm(quat, axis=-1)
            self.assertTrue(np.allclose(norms, 1.0, atol=1e-4))

    def test_valid_source_frames_end_at_t_minus_46_inclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_clip(root / "clip", frames=60)
            folder = MotionFolder.load(root)
            clip = folder.clips[0]
            # Final searchable source frame index is T - 46; stop is exclusive.
            self.assertEqual(clip.valid_frame_stop, clip.frame_count - 45)
            self.assertEqual(clip.valid_frame_stop, 15)
            last_valid = clip.valid_frame_stop - 1
            self.assertEqual(last_valid, clip.frame_count - 46)

    def test_rejects_missing_directory_and_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ContractError, r"motion.*npz|at least one"):
                MotionFolder.load(root)
        missing = Path(tempfile.gettempdir()) / "definitely-not-here-xyz"
        with self.assertRaisesRegex(ContractError, r"director"):
            MotionFolder.load(missing / "nope")

    def test_inventory_digest_changes_with_relative_path_or_file_bytes(self):
        def digest_for(builder):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                builder(root)
                return MotionFolder.load(root).inventory_sha256

        base = digest_for(lambda root: write_takara_clip(root / "walk"))
        self.assertRegex(base, r"^[0-9a-f]{64}$")

        renamed = digest_for(lambda root: write_takara_clip(root / "run"))
        self.assertNotEqual(base, renamed)

        def different_bytes(root):
            arrays = build_takara_arrays(frames=60)
            arrays["joint_pos"][0, 0] += np.float32(1.0)
            write_takara_arrays(root / "walk", arrays)

        mutated = digest_for(different_bytes)
        self.assertNotEqual(base, mutated)

        # Deterministic: identical inputs give identical digest.
        repeat = digest_for(lambda root: write_takara_clip(root / "walk"))
        self.assertEqual(base, repeat)


if __name__ == "__main__":
    unittest.main()
