import glob
import unittest
from unittest import mock
import numpy as np

from resources.g1_terrain_builder import sources as sources_module
from resources.g1_terrain_builder.sources import load_grail, load_takara

TAKARA = "/home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz"
REMAP = "/home/ubuntu/projects/g1_mm/isaac_to_mj.npy"
GRAIL_GLOB = "/home/ubuntu/datasets/GRAIL/data/curb/robot/*.pkl"


class SourceTests(unittest.TestCase):
    @staticmethod
    def _static_object_record(frames=3):
        root_pos = np.tile(
            np.array([[[1.0, 2.0, 3.0]]], np.float32), (frames, 1, 1))
        root_quat = np.tile(
            np.array([[[0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]]],
                     np.float32),
            (frames, 1, 1),
        )
        scale = np.array([[1.0], [2.0], [3.0]], np.float32)
        return {
            "terrain": {
                "root_pos": root_pos,
                "root_quat": root_quat,
                "fps": 25.0,
                "scale": scale,
                "contact_points_left_hand": {},
            }
        }

    def test_takara_is_native_g1_qpos(self):
        clip = load_takara(TAKARA, REMAP)
        self.assertEqual(clip.qpos.shape[1], 36)
        self.assertEqual(clip.fps, 50.0)
        np.testing.assert_allclose(
            np.linalg.norm(clip.qpos[:, 3:7], axis=1), 1.0, atol=1e-4,
        )

    def test_grail_is_native_g1_qpos(self):
        path = sorted(glob.glob(GRAIL_GLOB))[0]
        clip = load_grail(path)
        self.assertEqual(clip.qpos.shape, (250, 36))
        self.assertEqual(clip.fps, 25.0)
        self.assertEqual(clip.source_frames[-1], 249)
        np.testing.assert_allclose(
            np.linalg.norm(clip.qpos[:, 3:7], axis=1), 1.0, atol=1e-4,
        )

    def test_grail_object_pose_loads_static_xyzw_as_frozen_wxyz(self):
        records = self._static_object_record()
        record = records["terrain"]
        with mock.patch.object(
                sources_module.joblib, "load", return_value=records):
            pose = sources_module.load_grail_object_pose("terrain.pkl")

        np.testing.assert_array_equal(pose.root_pos, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            pose.root_quat,
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_array_equal(pose.scale, [1.0, 2.0, 3.0])
        self.assertEqual(pose.fps, 25.0)

        before = (pose.root_pos.copy(), pose.root_quat.copy(), pose.scale.copy())
        record["root_pos"][:] = -10.0
        record["root_quat"][:] = 0.0
        record["scale"][:] = 99.0
        np.testing.assert_array_equal(pose.root_pos, before[0])
        np.testing.assert_array_equal(pose.root_quat, before[1])
        np.testing.assert_array_equal(pose.scale, before[2])
        for array in (pose.root_pos, pose.root_quat, pose.scale):
            self.assertTrue(array.flags.owndata)
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = 0.0

    def test_grail_object_pose_rejects_malformed_or_moving_records(self):
        def load(records):
            with mock.patch.object(
                    sources_module.joblib, "load", return_value=records):
                return sources_module.load_grail_object_pose("terrain.pkl")

        valid = self._static_object_record()
        cases = []
        cases.append(({}, "one object record"))
        cases.append(({
            "first": valid["terrain"],
            "second": self._static_object_record()["terrain"],
        }, "one object record"))
        for missing in ("root_pos", "root_quat", "fps", "scale"):
            record = dict(valid["terrain"])
            del record[missing]
            cases.append(({"terrain": record}, missing))

        bad_shapes = (
            ("root_pos", np.zeros((3, 3), np.float32)),
            ("root_pos", np.zeros((0, 1, 3), np.float32)),
            ("root_quat", np.zeros((3, 4), np.float32)),
            ("root_quat", np.zeros((2, 1, 4), np.float32)),
            ("scale", np.ones(3, np.float32)),
            ("fps", np.array([25.0], np.float32)),
        )
        for field, value in bad_shapes:
            record = dict(valid["terrain"])
            record[field] = value
            cases.append(({"terrain": record}, field))

        nonfinite_position = np.array(
            valid["terrain"]["root_pos"], copy=True)
        nonfinite_position[0, 0, 0] = np.nan
        record = dict(valid["terrain"], root_pos=nonfinite_position)
        cases.append(({"terrain": record}, "finite"))

        unnormalized = np.array(valid["terrain"]["root_quat"], copy=True)
        unnormalized[:] *= 2.0
        record = dict(valid["terrain"], root_quat=unnormalized)
        cases.append(({"terrain": record}, "normalized"))

        moving_position = np.array(
            valid["terrain"]["root_pos"], copy=True)
        moving_position[-1, 0, 0] += 2e-4
        record = dict(valid["terrain"], root_pos=moving_position)
        cases.append(({"terrain": record}, "static"))

        moving_quaternion = np.array(
            valid["terrain"]["root_quat"], copy=True)
        moving_quaternion[-1, 0] = [0.0, 0.0, 0.001, 0.9999995]
        record = dict(valid["terrain"], root_quat=moving_quaternion)
        cases.append(({"terrain": record}, "static"))

        for records, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                    (TypeError, ValueError), message):
                load(records)


if __name__ == "__main__":
    unittest.main()
