import math
import unittest

import numpy as np

from mm_sonic.torch_stair_pivot_search import pivot_target_lattice


class StairPivotSearchTest(unittest.TestCase):
    def test_lattice_preserves_pivot_and_places_swing_on_sampled_surface(self):
        feet = np.asarray(
            ((0.0, 0.10, 0.20), (0.0, -0.10, 0.20)),
            dtype=np.float64,
        )

        def surface(points_xy):
            points = np.asarray(points_xy)
            return np.where(points[:, 0] >= 0.15, 0.18, 0.0)

        candidates = pivot_target_lattice(
            feet,
            yaw_delta_rad=math.pi / 2,
            planar_offsets_m=((0.0, 0.0),),
            sample_surface_height_m=surface,
        )

        self.assertEqual(len(candidates), 2)
        left_pivot = next(item for item in candidates if item.pivot_foot == 0)
        np.testing.assert_allclose(
            left_pivot.target_foot_position_world[0],
            feet[0],
        )
        np.testing.assert_allclose(
            left_pivot.target_foot_position_world[1],
            (0.20, 0.10, 0.38),
        )

    def test_lattice_offsets_only_swing_target(self):
        feet = np.asarray(
            ((0.0, 0.10, 0.20), (0.0, -0.10, 0.20)),
            dtype=np.float64,
        )

        candidates = pivot_target_lattice(
            feet,
            yaw_delta_rad=-math.pi / 2,
            planar_offsets_m=((0.05, -0.02), (-0.05, 0.02)),
            sample_surface_height_m=lambda points: np.zeros(len(points)),
        )

        self.assertEqual(len(candidates), 4)
        for candidate in candidates:
            np.testing.assert_allclose(
                candidate.target_foot_position_world[candidate.pivot_foot],
                feet[candidate.pivot_foot],
            )


if __name__ == "__main__":
    unittest.main()
