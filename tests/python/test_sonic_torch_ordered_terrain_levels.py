import unittest

import numpy as np

from mm_sonic.torch_ordered_terrain_levels import (
    OrderedLevelRejected,
    derive_ordered_terrain_levels,
    match_ordered_touchdown_levels,
)


def synthetic_stair_surface(points):
    points = np.asarray(points, dtype=np.float64)
    y = points[..., 1]
    return np.select(
        (y > 0.90, y > 0.65, y > 0.40),
        (0.0, 0.187, 0.349),
        default=0.0,
    )


class OrderedTerrainLevelTests(unittest.TestCase):
    def contract(self):
        return derive_ordered_terrain_levels(
            path_start_scene_xy=(-0.5, 1.4),
            path_stop_scene_xy=(0.7, 0.2),
            sample_surface=synthetic_stair_surface,
            sample_spacing_m=0.01,
            height_tolerance_m=0.03,
        )

    def test_derives_ordered_levels_and_preserves_ground_return(self):
        contract = self.contract()

        self.assertEqual(
            tuple(round(level.height_m, 3) for level in contract.levels),
            (0.000, 0.187, 0.349, 0.000),
        )
        self.assertFalse(contract.levels[0].requires_double_support)
        self.assertTrue(contract.levels[-1].requires_double_support)

    def test_matches_every_required_level_and_alternating_ground_exit(self):
        contract = self.contract()

        matched = match_ordered_touchdown_levels(
            contract,
            touchdown_progress_m=(0.10, 0.72, 1.08, 1.50, 1.62),
            touchdown_height_m=(0.0, 0.187, 0.349, 0.0, 0.0),
            touchdown_foot=(0, 1, 0, 1, 0),
        )

        self.assertEqual(matched, (0, 1, 2, 3, 3))

    def test_rejects_a_touchdown_that_skips_a_required_level(self):
        with self.assertRaisesRegex(
            OrderedLevelRejected, "skipped-required-level"
        ):
            match_ordered_touchdown_levels(
                self.contract(),
                touchdown_progress_m=(0.10, 1.08, 1.50, 1.62),
                touchdown_height_m=(0.0, 0.349, 0.0, 0.0),
                touchdown_foot=(0, 1, 0, 1),
            )

    def test_rejects_same_foot_ground_exit(self):
        with self.assertRaisesRegex(
            OrderedLevelRejected, "missing-ground-exit"
        ):
            match_ordered_touchdown_levels(
                self.contract(),
                touchdown_progress_m=(0.10, 0.72, 1.08, 1.50, 1.62),
                touchdown_height_m=(0.0, 0.187, 0.349, 0.0, 0.0),
                touchdown_foot=(0, 1, 0, 1, 1),
            )


if __name__ == "__main__":
    unittest.main()
