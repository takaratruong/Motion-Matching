import glob
import unittest
import numpy as np

from resources.g1_terrain_builder.sources import load_grail, load_takara

TAKARA = "/home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz"
REMAP = "/home/ubuntu/projects/g1_mm/isaac_to_mj.npy"
GRAIL_GLOB = "/home/ubuntu/datasets/GRAIL/data/curb/robot/*.pkl"


class SourceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
