import importlib.util
import unittest
from pathlib import Path

import numpy as np

_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "run_g1_motion_field_edge_placement_scan.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_g1_motion_field_edge_placement_scan", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class MotionFieldEdgePlacementScanTests(unittest.TestCase):
    def test_publication_contact_gate_rejects_visible_stance_float(self):
        self.assertEqual(
            _MODULE.publication_contact_rejections(
                metrics={"maximum_stance_contact_error_m": 0.019},
                maximum_contact_error_m=0.020,
            ),
            (),
        )
        self.assertEqual(
            _MODULE.publication_contact_rejections(
                metrics={"maximum_stance_contact_error_m": 0.021},
                maximum_contact_error_m=0.020,
            ),
            ("publication_contact_error",),
        )

    def test_rigid_yaw_rotation_preserves_pose_and_rotates_root(self):
        route = {
            "joint_position": np.zeros((2, 29)),
            "root_position_world": np.array(((2.0, 1.0, 0.8), (1.0, 2.0, 0.9))),
            "root_orientation_world_wxyz": np.tile((1.0, 0.0, 0.0, 0.0), (2, 1)),
            "source_support_mask": np.array(((True, False), (False, True))),
        }

        rotated = _MODULE.rigidly_rotate_route_yaw(
            route=route,
            pivot_matcher_xy=(1.0, 1.0),
            yaw_offset_rad=np.pi / 2.0,
        )

        np.testing.assert_allclose(
            rotated["root_position_world"],
            ((1.0, 2.0, 0.8), (0.0, 1.0, 0.9)),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            rotated["root_orientation_world_wxyz"],
            np.tile((np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)), (2, 1)),
            atol=1.0e-12,
        )
        np.testing.assert_array_equal(rotated["joint_position"], route["joint_position"])

    def test_target_heading_pair_must_preserve_turn_amount(self):
        self.assertEqual(
            _MODULE.target_heading_pair(
                source_from_heading_degrees=0.0,
                source_to_heading_degrees=45.0,
                requested_from_heading_degrees=45.0,
                requested_to_heading_degrees=90.0,
            ),
            (45.0, 90.0),
        )
        with self.assertRaisesRegex(Exception, "turn amount"):
            _MODULE.target_heading_pair(
                source_from_heading_degrees=0.0,
                source_to_heading_degrees=45.0,
                requested_from_heading_degrees=45.0,
                requested_to_heading_degrees=0.0,
            )

    def test_placement_rejections_include_lane_identity_and_error(self):
        self.assertEqual(
            _MODULE.placement_rejections(
                contact_rejections=(),
                placed_start_line_id="source",
                placed_end_line_id="target",
                expected_start_line_id="source",
                expected_end_line_id="target",
                start_lane_error_m=0.01,
                end_lane_error_m=0.02,
                maximum_lane_error_m=0.03,
            ),
            (),
        )
        self.assertEqual(
            _MODULE.placement_rejections(
                contact_rejections=("penetration",),
                placed_start_line_id="wrong",
                placed_end_line_id="target",
                expected_start_line_id="source",
                expected_end_line_id="target",
                start_lane_error_m=0.01,
                end_lane_error_m=0.04,
                maximum_lane_error_m=0.03,
            ),
            ("lane_assignment_mismatch", "lane_error", "penetration"),
        )

    def test_local_xy_search_starts_at_zero_and_stays_inside_radius(self):
        offsets = _MODULE.local_xy_search_offsets(
            search_radius_m=0.02,
            search_step_m=0.01,
        )

        np.testing.assert_array_equal(offsets[0], (0.0, 0.0))
        self.assertTrue((np.linalg.norm(offsets, axis=1) <= 0.0200001).all())
        self.assertTrue(
            (np.diff(np.linalg.norm(offsets, axis=1)) >= -1.0e-12).all()
        )

    def test_support_clearance_alignment_recovers_vertical_translation(self):
        source = np.full((3, 2, 4), 0.01)
        candidate = np.full((3, 2, 4), -0.19)
        support = np.array(
            ((True, False), (True, True), (False, True)), dtype=bool
        )

        actual = _MODULE.support_aligned_z_translation(
            source_sole_clearance_m=source,
            candidate_sole_clearance_m=candidate,
            support_mask=support,
        )

        self.assertAlmostEqual(actual, 0.20)

    def test_scene_translation_is_rotated_back_to_matcher_frame(self):
        actual = _MODULE.scene_translation_to_matcher(
            scene_translation_xy=(1.0, 0.0),
            yaw_scene_from_matcher_rad=np.pi / 2.0,
        )
        np.testing.assert_allclose(actual, (0.0, -1.0), atol=1.0e-12)

    def test_rigid_placement_preserves_pose_and_adds_xyz_translation(self):
        route = {
            "joint_position": np.arange(87, dtype=np.float64).reshape(3, 29),
            "root_position_world": np.array(
                ((0.0, 1.0, 0.8), (0.1, 1.1, 0.9), (0.2, 1.2, 1.0))
            ),
            "root_orientation_world_wxyz": np.tile(
                (1.0, 0.0, 0.0, 0.0), (3, 1)
            ),
            "source_support_mask": np.array(
                ((True, False), (True, False), (False, True))
            ),
        }

        placed = _MODULE.rigidly_place_route(
            route=route,
            translation_matcher_xyz=(0.4, -0.2, 0.3),
        )

        np.testing.assert_array_equal(placed["joint_position"], route["joint_position"])
        np.testing.assert_array_equal(
            placed["root_orientation_world_wxyz"],
            route["root_orientation_world_wxyz"],
        )
        np.testing.assert_array_equal(
            placed["source_support_mask"], route["source_support_mask"]
        )
        np.testing.assert_allclose(
            placed["root_position_world"],
            route["root_position_world"] + (0.4, -0.2, 0.3),
        )


if __name__ == "__main__":
    unittest.main()
