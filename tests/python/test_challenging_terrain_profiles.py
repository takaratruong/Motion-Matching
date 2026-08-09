from __future__ import annotations

import unittest

import numpy as np

from mm_sonic.challenging_terrain_profiles import PROFILES, build_profile


class ChallengingTerrainProfilesTest(unittest.TestCase):
    def test_profiles_are_deterministic_and_bounded(self) -> None:
        for name in PROFILES:
            first, metadata = build_profile(name)
            second, _ = build_profile(name)
            np.testing.assert_array_equal(first.height, second.height)
            self.assertEqual(metadata["family"], "humanoid_challenging_terrain")
            self.assertLessEqual(float(np.max(first.height)), 0.191)
            self.assertGreaterEqual(float(np.min(first.height)), -0.051)

    def test_profiles_have_flat_entry_and_exit(self) -> None:
        for name in PROFILES:
            field, _ = build_profile(name)
            self.assertTrue(np.allclose(field.height[:, :20], 0.0))
            self.assertTrue(np.allclose(field.height[:, -10:], 0.0))

    def test_discrete_profile_has_exact_plateaus(self) -> None:
        field, _ = build_profile("hct_discrete_blocks")
        rounded = set(np.round(np.unique(field.height), 3))
        self.assertTrue({-0.04, 0.0, 0.055, 0.085, 0.1}.issubset(rounded))


if __name__ == "__main__":
    unittest.main()
