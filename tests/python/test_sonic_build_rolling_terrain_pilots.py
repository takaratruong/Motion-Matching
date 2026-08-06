from __future__ import annotations

import unittest

import numpy as np

from mm_sonic.build_rolling_terrain_pilots import (
    PROFILES,
    build_profile,
    intended_stance_support_metrics,
)
from mm_sonic.canonical_terrain_matcher import RegularGridHeightField


class RollingTerrainProfileTests(unittest.TestCase):
    def test_profiles_are_flat_at_entry_and_exit_and_nontrivial_inside(self) -> None:
        for name in PROFILES:
            with self.subTest(profile=name):
                field, metadata = build_profile(name)
                x0 = field.origin_xy[0]
                dx = field.spacing_m[0]
                x = x0 + dx * np.arange(field.height.shape[1])
                entry = x <= 0.25
                exit_region = x >= 3.50
                np.testing.assert_allclose(field.height[:, entry], 0.0, atol=1.0e-8)
                np.testing.assert_allclose(
                    field.height[:, exit_region], 0.0, atol=1.0e-8
                )
                self.assertGreater(float(np.ptp(field.height)), 0.02)
                self.assertTrue(metadata["flat_entry_exit"])

    def test_bilinear_queries_cover_motion_corridor(self) -> None:
        field, _metadata = build_profile("cross_slope_bumps")
        points = np.asarray(((0.0, 0.0), (1.5, -0.2), (3.8, 0.1)))
        height, normal, hit = field.sample(points)

        self.assertTrue(np.all(hit))
        self.assertTrue(np.all(np.isfinite(height)))
        np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.0, atol=1.0e-6)

    def test_stance_support_audit_rejects_collision_free_hover(self) -> None:
        points = np.zeros((5, 2, 2, 3), dtype=np.float64)
        points[:, :, 0, 0] = -0.05
        points[:, :, 1, 0] = 0.05
        field = RegularGridHeightField(
            np.zeros((5, 5), dtype=np.float64),
            spacing_m=0.10,
            origin_xy=(-0.20, -0.20),
        )
        metrics = intended_stance_support_metrics(
            points,
            applied_root_shift_m=np.asarray((0.002, 0.002, 0.012, 0.002, 0.002)),
            field=field,
            fps=50.0,
        )

        self.assertEqual(metrics["intended_stance_frame_foot_count"], 10)
        self.assertEqual(metrics["intended_stance_hover_over_10mm_count"], 2)
        self.assertAlmostEqual(
            metrics["maximum_intended_stance_minimum_clearance_m"], 0.012
        )


if __name__ == "__main__":
    unittest.main()
