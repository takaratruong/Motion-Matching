import unittest
import numpy as np
from resources.g1_terrain_builder.resample import (
    output_frame_count, resample_map, resample_quaternions_wxyz,
    resample_vectors,
)


class ResampleTests(unittest.TestCase):
    def test_resample_map_50_to_60_has_exact_brackets_and_alpha(self):
        left, right, alpha = resample_map(6, 50.0, 60.0)
        np.testing.assert_array_equal(left[:4], [0, 0, 1, 2])
        np.testing.assert_array_equal(right[:4], [0, 1, 2, 3])
        np.testing.assert_allclose(alpha[:4], [0.0, 5/6, 2/3, 0.5])
        self.assertEqual(left.dtype, np.dtype(np.int32))
        self.assertEqual(right.dtype, np.dtype(np.int32))
        self.assertEqual(alpha.dtype, np.dtype(np.float32))

    def test_resample_map_120_to_60_is_exact_even_source_indices(self):
        left, right, alpha = resample_map(8171, 120.0, 60.0)
        expected = np.arange(0, 8171, 2, dtype=np.int32)
        self.assertEqual(len(left), 4086)
        np.testing.assert_array_equal(left, expected)
        np.testing.assert_array_equal(right, expected)
        np.testing.assert_array_equal(alpha, np.zeros(4086, np.float32))

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
            [[-1, 0, 0, 0], [.70710678, 0, .70710678, 0]],
        ], np.float64)
        self.assertLess(np.dot(src[0, 0], src[1, 0]), 0.0)
        self.assertGreater(np.dot(src[0, 1], src[1, 1]), 0.0)
        out = resample_quaternions_wxyz(src, 1.0, 2.0)
        self.assertEqual(out.shape, (3, 2, 4))
        c22, s22 = np.cos(np.pi / 8.0), np.sin(np.pi / 8.0)
        c45 = 2**-0.5
        expected = np.array([
            [[1, 0, 0, 0], [1, 0, 0, 0]],
            [[1, 0, 0, 0], [c22, 0, s22, 0]],
            [[1, 0, 0, 0], [c45, 0, c45, 0]],
        ], np.float64)
        np.testing.assert_allclose(out, expected, atol=1e-7)

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
