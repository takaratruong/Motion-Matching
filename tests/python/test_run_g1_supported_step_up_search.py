import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_g1_supported_step_up_search",
    ROOT / "resources" / "run_g1_supported_step_up_search.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _stair_height(points):
    points = np.asarray(points, dtype=np.float64)
    return np.where(points[..., 1] < 0.25, 0.53, 0.356)


class ContactPatternFilterTests(unittest.TestCase):
    def test_cheap_heading_search_batches_height_queries(self):
        body = np.zeros((3, 20, 3), dtype=np.float64)
        body[1:, 0, 0] = 1.0
        arrays = {"body_pos_w": body}
        sequence = MODULE.CompleteStepUpSequence(
            start_frame=0,
            first_contact_frame=1,
            trailing_contact_frame=2,
            end_exclusive=3,
            landing_foot=0,
            source_first_rise_m=0.18,
            source_final_height_delta_m=0.18,
        )
        calls = []

        def sample_height(points):
            value = np.asarray(points)
            calls.append(value.shape)
            return np.zeros(value.shape[:-1], dtype=np.float64)

        MODULE._cheap_placements(
            arrays=arrays,
            surface=np.zeros((3, 2), dtype=np.float64),
            sequence=sequence,
            alignment_yaw=0.0,
            traversal_angle_deg=45.0,
            sample_height=sample_height,
        )

        self.assertEqual(calls, [(2829, 2, 2)] * 4)

    def test_traversal_heading_rotates_from_cross_tread_toward_ascent(self):
        self.assertAlmostEqual(
            MODULE._scene_direction_angle(
                traversal_angle_deg=45.0, direction_sign=1
            ),
            -np.pi / 4.0,
        )
        self.assertAlmostEqual(
            MODULE._scene_direction_angle(
                traversal_angle_deg=45.0, direction_sign=-1
            ),
            3.0 * np.pi / 4.0,
        )

    def test_preferred_head_on_landing_is_centered_on_the_first_tread(self):
        np.testing.assert_allclose(
            MODULE._preferred_landing_point(90.0),
            (0.0, 0.62),
            atol=1e-12,
        )

    def test_nonhorizontal_search_samples_the_stair_footprint(self):
        points = np.stack(
            MODULE._target_landing_points(
                direction_sign=1, traversal_angle_deg=45.0
            )
        )

        self.assertLessEqual(float(points[:, 0].min()), -0.38)
        self.assertGreaterEqual(float(points[:, 0].max()), 0.38)
        self.assertLessEqual(float(points[:, 1].min()), -0.66)
        self.assertGreaterEqual(float(points[:, 1].max()), 0.66)

    def test_ranking_prefers_fewer_retargeted_frames(self):
        crouched = {
            "contact_height_pattern_error_m": 0.002,
            "maximum_stance_contact_error_m": 0.004,
            "minimum_sole_clearance_m": -0.024,
            "retargeted_swing_frame_count": 31,
            "frame_count": 116,
        }
        upright = {
            "contact_height_pattern_error_m": 0.007,
            "maximum_stance_contact_error_m": 0.004,
            "minimum_sole_clearance_m": -0.024,
            "retargeted_swing_frame_count": 14,
            "frame_count": 103,
        }

        self.assertLess(
            MODULE._candidate_score(upright),
            MODULE._candidate_score(crouched),
        )

    def test_ranking_prefers_a_smaller_support_footprint_shift(self):
        base = {
            "contact_height_pattern_error_m": 0.006,
            "maximum_stance_contact_error_m": 0.004,
            "minimum_sole_clearance_m": -0.024,
            "retargeted_swing_frame_count": 100,
            "frame_count": 100,
        }

        self.assertLess(
            MODULE._candidate_score(
                {**base, "maximum_support_footprint_shift_m": 0.02}
            ),
            MODULE._candidate_score(
                {**base, "maximum_support_footprint_shift_m": 0.08}
            ),
        )

    def test_paired_contact_delta_splits_error_between_two_feet(self):
        self.assertTrue(
            MODULE._paired_contact_delta_matches(
                source_delta_m=-0.386,
                target_delta_m=-0.356,
            )
        )
        self.assertFalse(
            MODULE._paired_contact_delta_matches(
                source_delta_m=-0.410,
                target_delta_m=-0.356,
            )
        )

    def test_root_shift_normalizes_each_authenticated_stance_surface(self):
        support = np.array(((True, False), (True, True)))
        ankle_z = np.array(((0.735, 0.40), (0.735, 0.905)))
        target_surface = np.array(((0.0, 0.0), (0.0, 0.17)))

        shift = MODULE._support_aligned_root_shift(
            support_mask=support,
            ankle_height_m=ankle_z,
            target_surface_height_m=target_surface,
        )

        np.testing.assert_allclose(shift, (-0.70, -0.70), atol=1e-12)

    def test_root_shift_does_not_follow_a_spurious_riser_during_stance(self):
        support = np.ones((4, 2), dtype=np.bool_)
        ankle_z = np.array(
            (
                (0.735, 0.735),
                (0.735, 0.735),
                (0.735, 0.735),
                (0.735, 0.735),
            )
        )
        target_surface = np.array(
            (
                (0.0, 0.0),
                (0.18, 0.0),
                (0.18, 0.0),
                (0.18, 0.0),
            )
        )

        shift = MODULE._support_aligned_root_shift(
            support_mask=support,
            ankle_height_m=ankle_z,
            target_surface_height_m=target_surface,
        )

        np.testing.assert_allclose(shift, -0.70, atol=1e-12)

    def test_retains_native_uneven_contacts_matching_target_stairs(self):
        matches, target_delta = MODULE._contact_pattern_matches(
            source_final_height_delta_m=0.18,
            landing_scene_xy=np.array((0.0, 0.36)),
            trailing_offset_scene_xy=np.array((0.39, -0.20)),
            sample_height=_stair_height,
        )

        self.assertTrue(matches)
        self.assertAlmostEqual(target_delta, 0.174)

    def test_rejects_level_curb_continuation_on_split_height_target(self):
        matches, target_delta = MODULE._contact_pattern_matches(
            source_final_height_delta_m=0.0,
            landing_scene_xy=np.array((0.0, 0.36)),
            trailing_offset_scene_xy=np.array((0.39, -0.20)),
            sample_height=_stair_height,
        )

        self.assertFalse(matches)
        self.assertAlmostEqual(target_delta, 0.174)

    def test_rejects_level_curb_continuation_on_level_target(self):
        matches, target_delta = MODULE._contact_pattern_matches(
            source_final_height_delta_m=0.0,
            landing_scene_xy=np.array((0.0, 0.36)),
            trailing_offset_scene_xy=np.array((0.10, 0.05)),
            sample_height=_stair_height,
        )

        self.assertFalse(matches)
        self.assertAlmostEqual(target_delta, 0.0)

    def test_parser_requires_authenticated_inputs(self):
        arguments = MODULE._parser().parse_args(
            (
                "--source-dataset",
                "source",
                "--target-dataset",
                "target",
                "--config",
                "config.json",
                "--g1-xml",
                "g1.xml",
                "--output",
                "output",
            )
        )

        self.assertEqual(arguments.source_dataset, Path("source"))
        self.assertEqual(arguments.target_dataset, Path("target"))
        self.assertEqual(arguments.traversal_angle_deg, 0.0)
        self.assertEqual(arguments.maximum_placements_per_event, 0)


if __name__ == "__main__":
    unittest.main()
