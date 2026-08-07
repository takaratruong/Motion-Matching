import unittest

import numpy as np

from mm_sonic.grail_terrain_source import resample_grail_motion


class GrailTerrainSourceResamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        timeline = np.arange(4, dtype=np.float32)
        self.root_position = np.stack(
            (timeline, np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)),
            axis=1,
        )
        self.root_quaternion_xyzw = np.zeros((4, 4), dtype=np.float32)
        self.root_quaternion_xyzw[:, 3] = 1.0
        self.root_quaternion_xyzw[1, 3] = -1.0
        self.dof_mujoco = timeline[:, None] + np.arange(29, dtype=np.float32)[None, :]

    def test_existing_25_to_50_path_keeps_legacy_samples_and_count(self) -> None:
        result = resample_grail_motion(
            self.root_position,
            self.root_quaternion_xyzw,
            self.dof_mujoco,
            source_fps=25.0,
            target_fps=50.0,
        )
        self.assertEqual(result.fps, 50.0)
        self.assertEqual(len(result.root_position), 7)
        np.testing.assert_allclose(result.root_position[:, 0], np.arange(7) * 0.5)
        np.testing.assert_allclose(result.root_position[-1], self.root_position[-1])

    def test_25_to_30_uses_target_timestamps_and_holds_terminal_pose(self) -> None:
        result = resample_grail_motion(
            self.root_position,
            self.root_quaternion_xyzw,
            self.dof_mujoco,
            source_fps=25.0,
            target_fps=30.0,
        )
        self.assertEqual(result.fps, 30.0)
        self.assertEqual(len(result.root_position), round(4 * 30 / 25))
        np.testing.assert_allclose(
            result.root_position[:, 0],
            (0.0, 5.0 / 6.0, 5.0 / 3.0, 2.5, 3.0),
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(result.root_position[-1], self.root_position[-1])
        np.testing.assert_allclose(
            np.linalg.norm(result.root_quaternion_xyzw, axis=1), 1.0, atol=1.0e-6
        )


if __name__ == "__main__":
    unittest.main()
