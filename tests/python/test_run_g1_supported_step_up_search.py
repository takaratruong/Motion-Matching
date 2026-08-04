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


if __name__ == "__main__":
    unittest.main()
