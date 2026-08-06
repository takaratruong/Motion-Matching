from __future__ import annotations

import math
import unittest

import numpy as np

from mm_sonic.motionbricks_hill import GentleHillProfile


class GentleHillProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hill = GentleHillProfile()

    def test_flat_aprons_and_cosine_crest(self) -> None:
        self.assertEqual(self.hill.height((-2.0, 0.0)), 0.0)
        self.assertEqual(self.hill.height((11.0, 0.0)), 0.0)
        self.assertAlmostEqual(
            self.hill.height(
                (self.hill.hill_start_x + 0.5 * self.hill.hill_length, 0.0)
            ),
            self.hill.height_m,
        )

    def test_maximum_grade_is_between_eight_and_ten_degrees(self) -> None:
        self.assertGreaterEqual(self.hill.max_slope_degrees, 8.0)
        self.assertLessEqual(self.hill.max_slope_degrees, 10.0)

    def test_height_rejects_queries_outside_certified_domain(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            self.hill.height((self.hill.domain_x[1] + 0.01, 0.0))
        with self.assertRaisesRegex(ValueError, "outside"):
            self.hill.height((0.0, self.hill.half_width + 0.01))

    def test_profile_is_continuous_at_hill_boundaries(self) -> None:
        epsilon = 1.0e-7
        start = self.hill.hill_start_x
        end = start + self.hill.hill_length
        self.assertLess(abs(self.hill.height((start + epsilon, 0.0))), 1.0e-12)
        self.assertLess(abs(self.hill.height((end - epsilon, 0.0))), 1.0e-12)

    def test_mesh_uses_the_same_height_contract(self) -> None:
        vertices, faces = self.hill.mesh(sample_count=41)
        self.assertEqual(vertices.shape, (82, 3))
        self.assertEqual(faces.shape, (80, 3))
        self.assertTrue(np.isfinite(vertices).all())
        self.assertTrue(np.isfinite(faces).all())
        for vertex in vertices:
            self.assertAlmostEqual(
                float(vertex[2]),
                self.hill.height(vertex[:2]),
                places=12,
            )
        self.assertTrue(
            np.allclose(np.abs(vertices[:, 1]), self.hill.half_width)
        )

    def test_mesh_rejects_too_few_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_count"):
            self.hill.mesh(sample_count=1)


if __name__ == "__main__":
    unittest.main()
