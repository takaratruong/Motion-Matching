import unittest

import numpy as np

from mm_sonic.build_stairs500_omnidirectional_pilots import (
    _temporally_reverse_motion,
    build_path_profile,
)
from mm_sonic.terrain_oracle.stitch import FrameProvenance, StitchedMotion


class StairDirectionalProfileTest(unittest.TestCase):
    def test_diagonal_has_zero_endpoint_turn_rate_and_opposite_offsets(self):
        profile = build_path_profile(
            "diagonal_hard_left",
            np.linspace(0.0, 1.0, 101),
            active_length_m=2.0,
            maximum_amplitude_m=0.4,
        )
        self.assertAlmostEqual(float(profile.path_yaw_offset_rad[0]), 0.0)
        self.assertAlmostEqual(float(profile.path_yaw_offset_rad[-1]), 0.0)
        self.assertLess(float(profile.lateral_offset_m[0]), 0.0)
        self.assertGreater(float(profile.lateral_offset_m[-1]), 0.0)
        self.assertLessEqual(profile.maximum_path_angle_deg, 32.0 + 1.0e-6)

    def test_zigzag_changes_lateral_direction_without_endpoint_pop(self):
        profile = build_path_profile(
            "zigzag_left_right",
            np.linspace(0.0, 1.0, 301),
            active_length_m=2.5,
            maximum_amplitude_m=0.42,
        )
        self.assertGreater(float(np.max(profile.lateral_offset_m)), 0.05)
        self.assertLess(float(np.min(profile.lateral_offset_m)), -0.05)
        self.assertAlmostEqual(float(profile.path_yaw_offset_rad[0]), 0.0)
        self.assertAlmostEqual(float(profile.path_yaw_offset_rad[-1]), 0.0)
        self.assertLessEqual(profile.maximum_path_angle_deg, 28.0 + 1.0e-6)

    def test_crab_profile_separates_travel_tangent_and_facing(self):
        profile = build_path_profile(
            "crab_left",
            np.linspace(0.0, 1.0, 101),
            active_length_m=2.5,
            maximum_amplitude_m=0.42,
        )
        middle = len(profile.facing_yaw_offset_rad) // 2
        self.assertAlmostEqual(
            float(np.degrees(profile.facing_yaw_offset_rad[middle])),
            -24.0,
            places=5,
        )
        self.assertGreater(profile.maximum_path_angle_deg, 10.0)

    def test_straight_profile_is_identity(self):
        profile = build_path_profile(
            "straight",
            np.linspace(0.0, 1.0, 11),
            active_length_m=2.0,
            maximum_amplitude_m=0.4,
        )
        np.testing.assert_array_equal(profile.lateral_offset_m, 0.0)
        np.testing.assert_array_equal(profile.path_yaw_offset_rad, 0.0)
        np.testing.assert_array_equal(profile.facing_yaw_offset_rad, 0.0)

    def test_temporal_reverse_preserves_poses_and_maps_seams(self):
        motion = StitchedMotion(
            fps=50.0,
            root_position_world=np.arange(15, dtype=np.float32).reshape(5, 3),
            root_quaternion_world_wxyz=np.arange(
                20, dtype=np.float32
            ).reshape(5, 4),
            joint_position=np.arange(25, dtype=np.float32).reshape(5, 5),
            provenance=tuple(
                FrameProvenance(3, frame, "clip") for frame in range(5)
            ),
            seam_indices=(2,),
        )
        reversed_motion = _temporally_reverse_motion(motion)
        np.testing.assert_array_equal(
            reversed_motion.root_position_world,
            motion.root_position_world[::-1],
        )
        np.testing.assert_array_equal(
            reversed_motion.joint_position,
            motion.joint_position[::-1],
        )
        self.assertEqual(
            tuple(value.source_frame for value in reversed_motion.provenance),
            (4, 3, 2, 1, 0),
        )
        self.assertEqual(reversed_motion.seam_indices, (3,))


if __name__ == "__main__":
    unittest.main()
