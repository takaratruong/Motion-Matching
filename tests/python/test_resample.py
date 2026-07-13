import unittest
import numpy as np
from resources.g1_terrain_builder.resample import (
    output_frame_count, resample_quaternions_wxyz, resample_vectors,
)


class ResampleTests(unittest.TestCase):
    def test_duration_is_preserved(self):
        self.assertEqual(output_frame_count(250, 25.0, 60.0), 599)

    def test_linear_translation(self):
        src = np.array([[0, 0, 0], [1, 0, 0]], np.float64)
        out = resample_vectors(src, 1.0, 2.0)
        np.testing.assert_allclose(out[:, 0], [0.0, 0.5, 1.0], atol=1e-7)

    def test_slerp_takes_short_arc_and_normalizes(self):
        src = np.array([[1, 0, 0, 0], [-0.70710678, 0, -0.70710678, 0]])
        out = resample_quaternions_wxyz(src, 1.0, 2.0)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-7)
        self.assertGreater(out[1, 0], 0.9)
