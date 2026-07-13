import unittest
import numpy as np
from resources.g1_terrain_builder.resample import (
    output_frame_count, resample_quaternions_wxyz, resample_vectors,
)


class ResampleTests(unittest.TestCase):
    def test_duration_is_preserved(self):
        self.assertEqual(output_frame_count(250, 25.0, 25.0), 250)

    def test_linear_translation(self):
        src = np.array([[0, 0, 0], [1, 0, 0]], np.float64)
        out = resample_vectors(src, 1.0, 2.0)
        np.testing.assert_allclose(out[:, 0], [0.0, 0.5, 1.0], atol=1e-7)

    def test_slerp_takes_short_arc_and_normalizes(self):
        src = np.array([[1, 0, 0, 0], [-0.70710678, 0, -0.70710678, 0]])
        out = resample_quaternions_wxyz(src, 1.0, 2.0)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-7)
        self.assertGreater(out[1, 0], 0.9)

    def test_native_25hz_samples_are_not_time_warped(self):
        src = np.arange(250, dtype=np.float64)[:, None]
        out = resample_vectors(src, 25.0, 25.0)
        np.testing.assert_array_equal(out, src)

    def test_50hz_samples_use_exact_25hz_grid(self):
        src = np.arange(6, dtype=np.float64)[:, None]
        out = resample_vectors(src, 50.0, 25.0)
        np.testing.assert_array_equal(out[:, 0], [0.0, 2.0, 4.0])

    def test_slerp_unrolls_each_bone_independently(self):
        src = np.array([
            [[1, 0, 0, 0], [1, 0, 0, 0]],
            [[-1, 0, 0, 0], [-.70710678, 0, -.70710678, 0]],
        ], np.float64)
        out = resample_quaternions_wxyz(src, 1.0, 2.0)
        self.assertEqual(out.shape, (3, 2, 4))
        np.testing.assert_allclose(
            np.linalg.norm(out, axis=-1), 1.0, atol=1e-7)
        self.assertGreater(out[1, 0, 0], 0.999)
        self.assertGreater(out[1, 1, 0], 0.9)

    def test_invalid_resampling_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            resample_vectors(np.empty((0, 3)), 25.0, 25.0)
        with self.assertRaises(ValueError):
            resample_vectors(np.zeros((2, 3)), 0.0, 25.0)
        with self.assertRaises(ValueError):
            resample_vectors(np.zeros((2, 3)), 25.0, 0.0)
        with self.assertRaises(ValueError):
            resample_vectors(
                np.array([[0.0, np.nan, 0.0]]), 25.0, 25.0)
        with self.assertRaises(ValueError):
            resample_quaternions_wxyz(
                np.zeros((2, 4)), 25.0, 25.0)
        with self.assertRaises(ValueError):
            resample_quaternions_wxyz(
                np.array([[np.inf, 0.0, 0.0, 0.0]]), 25.0, 25.0)
