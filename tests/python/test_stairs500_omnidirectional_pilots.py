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

    def test_facing_weave_changes_body_direction_without_bending_path(self):
        profile = build_path_profile(
            "facing_weave_left_right",
            np.linspace(0.0, 1.0, 301),
            active_length_m=2.5,
            maximum_amplitude_m=0.42,
        )
        np.testing.assert_array_equal(profile.lateral_offset_m, 0.0)
        np.testing.assert_array_equal(profile.path_yaw_offset_rad, 0.0)
        self.assertGreater(
            float(np.degrees(np.max(profile.facing_yaw_offset_rad))),
            9.9,
        )
        self.assertLess(
            float(np.degrees(np.min(profile.facing_yaw_offset_rad))),
            -9.9,
        )
        self.assertAlmostEqual(float(profile.facing_yaw_offset_rad[0]), 0.0)
        self.assertAlmostEqual(float(profile.facing_yaw_offset_rad[-1]), 0.0)

    def test_turning_slalom_changes_path_and_pelvis_yaw_repeatedly(self):
        profile = build_path_profile(
            "turning_slalom_left_right",
            np.linspace(0.0, 1.0, 401),
            active_length_m=3.0,
            maximum_amplitude_m=0.42,
        )
        self.assertGreater(float(np.max(profile.lateral_offset_m)), 0.02)
        self.assertLess(float(np.min(profile.lateral_offset_m)), -0.02)
        self.assertGreater(float(np.max(profile.pose_yaw_offset_rad)), 0.01)
        self.assertLess(float(np.min(profile.pose_yaw_offset_rad)), -0.01)
        self.assertLessEqual(profile.maximum_path_angle_deg, 12.0 + 1.0e-6)

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

    def test_cowarp_zigzag_changes_direction_at_bounded_angle(self):
        profile = build_path_profile(
            "cowarp_zigzag_gentle_left_right",
            np.linspace(0.0, 1.0, 401),
            active_length_m=3.0,
            maximum_amplitude_m=0.25,
        )
        self.assertGreater(float(np.max(profile.lateral_offset_m)), 0.02)
        self.assertLess(float(np.min(profile.lateral_offset_m)), -0.02)
        self.assertLessEqual(profile.maximum_path_angle_deg, 12.0 + 1.0e-6)

    def test_cowarp_facing_weave_keeps_path_straight(self):
        profile = build_path_profile(
            "cowarp_facing_weave_left_right",
            np.linspace(0.0, 1.0, 401),
            active_length_m=3.0,
            maximum_amplitude_m=0.25,
        )
        np.testing.assert_array_equal(profile.lateral_offset_m, 0.0)
        np.testing.assert_array_equal(profile.path_yaw_offset_rad, 0.0)
        self.assertGreater(float(np.degrees(np.max(profile.facing_yaw_offset_rad))), 9.9)
        self.assertLess(float(np.degrees(np.min(profile.facing_yaw_offset_rad))), -9.9)

    def test_cowarp_hard_diagonal_reaches_requested_angle(self):
        profile = build_path_profile(
            "cowarp_diagonal_hard_left",
            np.linspace(0.0, 1.0, 1001),
            active_length_m=3.0,
            maximum_amplitude_m=1.0,
        )
        self.assertAlmostEqual(profile.maximum_path_angle_deg, 38.0, places=3)
        self.assertGreater(float(np.ptp(profile.lateral_offset_m)), 0.5)

    def test_fixed_terrain_extreme_travel_reaches_fifty_degrees(self):
        profile = build_path_profile(
            "travel_extreme_left",
            np.linspace(0.0, 1.0, 2001),
            active_length_m=4.0,
            maximum_amplitude_m=2.0,
        )
        self.assertAlmostEqual(profile.maximum_path_angle_deg, 50.0, places=3)
        np.testing.assert_array_equal(profile.pose_yaw_offset_rad, 0.0)
        np.testing.assert_array_equal(profile.facing_yaw_offset_rad, 0.0)

    def test_fixed_terrain_side_on_profile_reaches_seventy_five_degrees(self):
        profile = build_path_profile(
            "face_side_on_right",
            np.linspace(0.0, 1.0, 2001),
            active_length_m=4.0,
            maximum_amplitude_m=2.0,
        )
        np.testing.assert_array_equal(profile.lateral_offset_m, 0.0)
        self.assertAlmostEqual(
            float(np.degrees(np.min(profile.facing_yaw_offset_rad))),
            -75.0,
            places=3,
        )

    def test_extreme_approach_turns_then_holds_its_new_lane(self):
        progress = np.linspace(0.0, 1.0, 2001)
        profile = build_path_profile(
            "approach_extreme_left",
            progress,
            active_length_m=4.0,
            maximum_amplitude_m=2.0,
        )
        self.assertAlmostEqual(profile.maximum_path_angle_deg, 50.0, places=3)
        second_half = profile.lateral_offset_m[progress >= 0.5]
        self.assertLess(float(np.ptp(second_half)), 1.0e-9)
        self.assertGreater(float(second_half[0]), 0.0)

    def test_cowarp_held_lane_dwells_before_returning(self):
        progress = np.linspace(0.0, 1.0, 1201)
        profile = build_path_profile(
            "cowarp_lane_hold_left",
            progress,
            active_length_m=3.0,
            maximum_amplitude_m=0.5,
        )
        middle = profile.lateral_offset_m[
            (progress >= 1.0 / 3.0) & (progress <= 2.0 / 3.0)
        ]
        self.assertLess(float(np.ptp(middle)), 1.0e-9)
        self.assertGreater(float(middle[0]), 0.1)
        self.assertAlmostEqual(float(profile.lateral_offset_m[0]), 0.0, places=9)
        self.assertAlmostEqual(float(profile.lateral_offset_m[-1]), 0.0, places=9)

    def test_cowarp_counterface_slalom_changes_both_sticks(self):
        profile = build_path_profile(
            "cowarp_slalom_counterface_left_right",
            np.linspace(0.0, 1.0, 1001),
            active_length_m=4.0,
            maximum_amplitude_m=0.5,
        )
        self.assertGreater(float(np.max(profile.path_yaw_offset_rad)), 0.0)
        self.assertLess(float(np.min(profile.path_yaw_offset_rad)), 0.0)
        self.assertGreater(float(np.max(profile.facing_yaw_offset_rad)), 0.0)
        self.assertLess(float(np.min(profile.facing_yaw_offset_rad)), 0.0)

    def test_cowarp_hard_crab_opposes_path_with_facing_stick(self):
        profile = build_path_profile(
            "cowarp_crab_hard_left",
            np.linspace(0.0, 1.0, 1001),
            active_length_m=3.0,
            maximum_amplitude_m=0.8,
        )
        peak = int(np.argmax(profile.path_yaw_offset_rad))
        self.assertAlmostEqual(
            float(np.degrees(profile.path_yaw_offset_rad[peak])),
            24.0,
            places=3,
        )
        self.assertAlmostEqual(
            float(
                np.degrees(
                    profile.path_yaw_offset_rad[peak]
                    + profile.facing_yaw_offset_rad[peak]
                )
            ),
            0.0,
            places=3,
        )

    def test_cowarp_hard_facing_weave_reaches_both_twenty_degree_offsets(self):
        profile = build_path_profile(
            "cowarp_facing_weave_hard_left_right",
            np.linspace(0.0, 1.0, 1001),
            active_length_m=3.0,
            maximum_amplitude_m=0.8,
        )
        self.assertGreaterEqual(
            float(np.degrees(np.max(profile.facing_yaw_offset_rad))),
            19.99,
        )
        self.assertLessEqual(
            float(np.degrees(np.min(profile.facing_yaw_offset_rad))),
            -19.99,
        )

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
