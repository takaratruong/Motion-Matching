from __future__ import annotations

import math
import unittest

import numpy as np


class TerrainPFNNHillMapTests(unittest.TestCase):
    def test_builds_three_c1_hills_with_requested_peak_grades(self) -> None:
        from mm_sonic.terrain_pfnn.hill_map import TerrainPFNNHillMap

        terrain = TerrainPFNNHillMap()

        self.assertEqual(terrain.requested_grades_deg, (10.0, 15.0, 18.9))
        self.assertEqual(len(terrain.hills), 3)
        self.assertGreaterEqual(terrain.hills[0].start_x, 1.5)
        for x in np.linspace(-0.5, 0.5, 21):
            self.assertEqual(terrain.height_at((float(x), 0.0)), 0.0)
        for hill, requested in zip(terrain.hills, terrain.requested_grades_deg):
            samples = np.linspace(hill.start_x, hill.end_x, 2001)
            slopes = np.asarray([terrain.analytic_slope_at(float(x)) for x in samples])
            measured = math.degrees(math.atan(float(np.max(np.abs(slopes)))))
            self.assertAlmostEqual(measured, requested, delta=0.05)
            self.assertAlmostEqual(terrain.analytic_slope_at(hill.start_x), 0.0, places=10)
            self.assertAlmostEqual(terrain.analytic_slope_at(hill.end_x), 0.0, places=10)

        self.assertAlmostEqual(terrain.height_at((terrain.x_min, 0.0)), 0.0, places=10)
        self.assertAlmostEqual(terrain.height_at((terrain.x_max, 0.0)), 0.0, places=10)

    def test_render_mesh_and_runtime_query_share_the_same_triangles(self) -> None:
        from mm_sonic.terrain_pfnn.hill_map import TerrainPFNNHillMap

        terrain = TerrainPFNNHillMap(grid_spacing_m=0.05)
        self.assertEqual(terrain.vertices.ndim, 2)
        self.assertEqual(terrain.vertices.shape[1], 3)
        self.assertEqual(terrain.faces.ndim, 2)
        self.assertEqual(terrain.faces.shape[1], 3)
        self.assertTrue(np.isfinite(terrain.vertices).all())
        self.assertTrue(np.isfinite(terrain.faces).all())

        rng = np.random.default_rng(1701)
        for _ in range(128):
            face = terrain.faces[int(rng.integers(0, len(terrain.faces)))]
            triangle = terrain.vertices[face]
            weights = rng.random(3)
            weights /= weights.sum()
            point = weights @ triangle
            queried = terrain.height_at(point[:2])
            self.assertAlmostEqual(queried, float(point[2]), delta=1.0e-10)

    def test_outside_map_is_unsupported(self) -> None:
        from mm_sonic.terrain_pfnn.hill_map import TerrainPFNNHillMap

        terrain = TerrainPFNNHillMap()
        self.assertIsNone(terrain.height_at((terrain.x_min - 0.01, 0.0)))
        self.assertIsNone(terrain.height_at((terrain.x_max + 0.01, 0.0)))
        self.assertIsNone(terrain.height_at((0.0, terrain.y_max + 0.01)))


if __name__ == "__main__":
    unittest.main()
