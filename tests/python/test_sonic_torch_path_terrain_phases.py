import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_path_terrain_phases import (
    classify_phase_coverage,
    segment_path_surface,
)


class PathTerrainPhaseTests(unittest.TestCase):
    @staticmethod
    def platform_surface(points):
        xy = np.asarray(points, dtype=np.float64)
        inside = (xy[..., 0] >= -0.6) & (xy[..., 0] <= 0.6)
        split = np.where(xy[..., 1] < 0.0, 0.20, 0.35)
        return np.where(inside, split, 0.0)

    def test_segments_ground_platform_ground_into_three_phases(self):
        phases = segment_path_surface(
            path_start_scene_xy=(-1.2, 0.0),
            path_stop_scene_xy=(1.2, 0.0),
            step_width_m=0.20,
            sample_surface=self.platform_surface,
        )

        self.assertEqual([phase.kind for phase in phases], [
            "mount",
            "interior",
            "dismount",
        ])
        self.assertLessEqual(phases[0].start_m, 1.0e-9)
        self.assertLessEqual(abs(phases[1].start_m - 0.60), 0.011)
        self.assertLessEqual(abs(phases[1].stop_m - 1.80), 0.011)
        self.assertGreaterEqual(phases[2].stop_m, 2.4 - 1.0e-9)
        np.testing.assert_allclose(
            phases[1].normalized_support_height_m,
            (0.075, -0.075),
            atol=1.0e-6,
        )

    def test_common_vertical_offset_does_not_change_interior_signature(self):
        first = segment_path_surface(
            path_start_scene_xy=(-1.2, 0.0),
            path_stop_scene_xy=(1.2, 0.0),
            step_width_m=0.20,
            sample_surface=self.platform_surface,
        )[1]

        def raised(points):
            values = self.platform_surface(points)
            xy = np.asarray(points, dtype=np.float64)
            inside = (xy[..., 0] >= -0.6) & (xy[..., 0] <= 0.6)
            return values + np.where(inside, 0.4, 0.0)

        second = segment_path_surface(
            path_start_scene_xy=(-1.2, 0.0),
            path_stop_scene_xy=(1.2, 0.0),
            step_width_m=0.20,
            sample_surface=raised,
        )[1]

        np.testing.assert_allclose(
            first.normalized_support_height_m,
            second.normalized_support_height_m,
            atol=1.0e-6,
        )
        self.assertAlmostEqual(
            second.mean_support_height_m - first.mean_support_height_m,
            0.4,
        )

    def test_boundary_failure_preserves_partial_interior_coverage(self):
        self.assertEqual(
            classify_phase_coverage(
                required_kinds=("mount", "interior", "dismount"),
                certified_kinds=("interior",),
            ),
            "partial",
        )
        self.assertEqual(
            classify_phase_coverage(
                required_kinds=("mount", "interior", "dismount"),
                certified_kinds=("mount", "interior", "dismount"),
            ),
            "full",
        )

    def test_rejects_path_with_different_endpoint_ground_heights(self):
        def ramp(points):
            return np.asarray(points)[..., 0]

        with self.assertRaises(ContractError):
            segment_path_surface(
                path_start_scene_xy=(-1.0, 0.0),
                path_stop_scene_xy=(1.0, 0.0),
                step_width_m=0.20,
                sample_surface=ramp,
            )


if __name__ == "__main__":
    unittest.main()
