import unittest

import numpy as np
from mm_sonic.torch_object_motion_field import (
    MotionFieldRouteTrace,
    assign_endpoint_to_heading_line,
    motion_field_intersections,
    rasterized_motion_field,
    transition_opportunities,
)


def _square(center=(0.0, 0.0)):
    cx, cy = center
    return np.array(
        [
            (cx + x, cy + y)
            for x in np.linspace(-0.5, 0.5, 11)
            for y in np.linspace(-0.5, 0.5, 11)
        ],
        dtype=np.float64,
    )


class ObjectMotionFieldTests(unittest.TestCase):
    def test_finite_radius_endpoint_assigns_to_nearest_outgoing_lane(self):
        lines = rasterized_motion_field(
            elevated_scene_xy=_square(),
            heading_degrees=(0.0, -45.0),
            spacing_m=0.20,
            approach_margin_m=0.50,
            exit_margin_m=0.50,
        )
        outgoing = [line for line in lines if line.heading_degrees == -45.0]
        target = outgoing[len(outgoing) // 2 + 1]
        start = np.asarray(target.start_scene_xy)
        stop = np.asarray(target.stop_scene_xy)
        point = start + 0.60 * (stop - start)
        point += 0.03 * np.array((np.sqrt(0.5), np.sqrt(0.5)))

        assignment = assign_endpoint_to_heading_line(
            lines=lines,
            heading_degrees=-45.0,
            endpoint_scene_xy=point,
        )

        self.assertEqual(assignment.line_id, target.line_id)
        self.assertAlmostEqual(assignment.lateral_distance_m, 0.03)
        self.assertGreater(assignment.progress_m, 0.0)

    def test_endpoint_lane_assignment_is_translation_and_yaw_equivariant(self):
        points = _square()
        lines = rasterized_motion_field(
            elevated_scene_xy=points,
            heading_degrees=(0.0, -45.0),
            spacing_m=0.20,
            approach_margin_m=0.50,
            exit_margin_m=0.50,
        )
        endpoint = np.array((0.31, -0.12))
        first = assign_endpoint_to_heading_line(
            lines=lines,
            heading_degrees=-45.0,
            endpoint_scene_xy=endpoint,
        )
        rotation = np.array(((0.0, -1.0), (1.0, 0.0)))
        translation = np.array((1.2, -0.4))
        moved_lines = rasterized_motion_field(
            elevated_scene_xy=points @ rotation.T + translation,
            heading_degrees=(90.0, 45.0),
            spacing_m=0.20,
            approach_margin_m=0.50,
            exit_margin_m=0.50,
        )
        moved = assign_endpoint_to_heading_line(
            lines=moved_lines,
            heading_degrees=45.0,
            endpoint_scene_xy=endpoint @ rotation.T + translation,
        )

        self.assertEqual(moved.lane_index, first.lane_index)
        self.assertAlmostEqual(moved.lateral_distance_m, first.lateral_distance_m)
        self.assertAlmostEqual(moved.progress_m, first.progress_m)

    def test_rasterizes_parallel_lines_for_every_requested_heading(self):
        lines = rasterized_motion_field(
            elevated_scene_xy=_square(),
            heading_degrees=(0.0, 45.0, -45.0, 90.0),
            spacing_m=0.20,
            approach_margin_m=0.50,
            exit_margin_m=0.50,
        )

        self.assertEqual(
            {line.heading_degrees for line in lines},
            {0.0, 45.0, -45.0, 90.0},
        )
        for heading in (0.0, 45.0, -45.0, 90.0):
            family = [line for line in lines if line.heading_degrees == heading]
            offsets = np.array([line.lateral_offset_m for line in family])
            self.assertGreaterEqual(len(offsets), 5)
            np.testing.assert_allclose(np.diff(offsets), 0.20, atol=1.0e-12)

    def test_raster_includes_both_boundary_adjacent_lines(self):
        lines = rasterized_motion_field(
            elevated_scene_xy=_square(),
            heading_degrees=(0.0,),
            spacing_m=0.20,
            approach_margin_m=0.50,
            exit_margin_m=0.50,
        )

        self.assertEqual(len(lines), 6)
        np.testing.assert_allclose(
            [line.lateral_offset_m for line in lines],
            np.linspace(-0.5, 0.5, 6),
            atol=1.0e-12,
        )

    def test_finds_only_cross_heading_segment_intersections(self):
        lines = rasterized_motion_field(
            elevated_scene_xy=_square(),
            heading_degrees=(0.0, 90.0),
            spacing_m=0.20,
            approach_margin_m=0.25,
            exit_margin_m=0.25,
        )

        intersections = motion_field_intersections(lines)

        self.assertTrue(intersections)
        self.assertTrue(
            all(item.first_heading_degrees != item.second_heading_degrees
                for item in intersections)
        )
        center = min(
            intersections,
            key=lambda item: np.linalg.norm(item.scene_xy),
        )
        self.assertLessEqual(
            np.abs(np.asarray(center.scene_xy)).max(),
            0.10 + 1.0e-12,
        )

    def test_geometry_is_translation_and_yaw_equivariant(self):
        points = _square()
        first = rasterized_motion_field(
            elevated_scene_xy=points,
            heading_degrees=(0.0, 45.0),
            spacing_m=0.20,
            approach_margin_m=0.25,
            exit_margin_m=0.25,
        )
        rotation = np.array(((0.0, -1.0), (1.0, 0.0)))
        translation = np.array((1.3, -0.7))
        transformed = points @ rotation.T + translation
        second = rasterized_motion_field(
            elevated_scene_xy=transformed,
            heading_degrees=(90.0, 135.0),
            spacing_m=0.20,
            approach_margin_m=0.25,
            exit_margin_m=0.25,
        )

        self.assertEqual(len(first), len(second))
        for original, moved in zip(first, second):
            np.testing.assert_allclose(
                moved.start_scene_xy,
                np.asarray(original.start_scene_xy) @ rotation.T + translation,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                moved.stop_scene_xy,
                np.asarray(original.stop_scene_xy) @ rotation.T + translation,
                atol=1.0e-12,
            )

    def test_transition_uses_nearby_phase_compatible_frames(self):
        lines = rasterized_motion_field(
            elevated_scene_xy=_square(),
            heading_degrees=(0.0, 90.0),
            spacing_m=0.20,
            approach_margin_m=0.25,
            exit_margin_m=0.25,
        )
        intersections = motion_field_intersections(lines)
        center = min(
            intersections,
            key=lambda item: np.linalg.norm(item.scene_xy),
        )
        crossing = np.asarray(center.scene_xy)
        first = MotionFieldRouteTrace(
            line_id=center.first_line_id,
            root_scene_xy=crossing + np.array(((-0.10, 0.0), (0.0, 0.0), (0.10, 0.0))),
            support_mask=np.array(((True, False), (False, True), (True, False))),
            surface_height_m=np.zeros(3),
        )
        second = MotionFieldRouteTrace(
            line_id=center.second_line_id,
            root_scene_xy=crossing + np.array(((0.0, -0.08), (0.0, 0.01), (0.0, 0.08))),
            support_mask=np.array(((True, False), (True, False), (False, True))),
            surface_height_m=np.zeros(3),
        )

        opportunities = transition_opportunities(
            intersections=(center,),
            route_traces=(first, second),
            search_radius_m=0.12,
            maximum_height_difference_m=0.03,
        )

        self.assertEqual(len(opportunities), 1)
        opportunity = opportunities[0]
        self.assertEqual(opportunity.status, "admitted")
        self.assertEqual(opportunity.reason, "phase_and_height_compatible")
        self.assertEqual((opportunity.first_frame, opportunity.second_frame), (1, 2))
        self.assertEqual(opportunity.first_support, (False, True))
        self.assertEqual(opportunity.second_support, (False, True))

    def test_transition_rejections_remain_explicit(self):
        lines = rasterized_motion_field(
            elevated_scene_xy=_square(),
            heading_degrees=(0.0, 90.0),
            spacing_m=0.20,
            approach_margin_m=0.25,
            exit_margin_m=0.25,
        )
        center = min(
            motion_field_intersections(lines),
            key=lambda item: np.linalg.norm(item.scene_xy),
        )
        crossing = np.asarray(center.scene_xy)

        def trace(line_id, support, height):
            return MotionFieldRouteTrace(
                line_id=line_id,
                root_scene_xy=crossing + np.array(((0.0, 0.0), (0.04, 0.0))),
                support_mask=np.asarray(support, dtype=np.bool_),
                surface_height_m=np.full(2, height),
            )

        mismatch = transition_opportunities(
            intersections=(center,),
            route_traces=(
                trace(center.first_line_id, ((True, False), (True, False)), 0.0),
                trace(center.second_line_id, ((False, True), (False, True)), 0.0),
            ),
            search_radius_m=0.10,
            maximum_height_difference_m=0.03,
        )[0]
        self.assertEqual(mismatch.status, "rejected")
        self.assertEqual(mismatch.reason, "support_phase_mismatch")

        height = transition_opportunities(
            intersections=(center,),
            route_traces=(
                trace(center.first_line_id, ((True, False), (True, False)), 0.0),
                trace(center.second_line_id, ((True, False), (True, False)), 0.10),
            ),
            search_radius_m=0.10,
            maximum_height_difference_m=0.03,
        )[0]
        self.assertEqual(height.status, "rejected")
        self.assertEqual(height.reason, "terrain_height_mismatch")

    def test_transition_selection_is_translation_and_yaw_invariant(self):
        lines = rasterized_motion_field(
            elevated_scene_xy=_square(),
            heading_degrees=(0.0, 90.0),
            spacing_m=0.20,
            approach_margin_m=0.25,
            exit_margin_m=0.25,
        )
        center = min(
            motion_field_intersections(lines),
            key=lambda item: np.linalg.norm(item.scene_xy),
        )
        paths = (
            np.array(((-0.1, 0.0), (0.0, 0.0), (0.1, 0.0))),
            np.array(((0.0, -0.1), (0.0, 0.0), (0.0, 0.1))),
        )
        support = np.array(((True, False), (False, True), (True, False)))
        original = transition_opportunities(
            intersections=(center,),
            route_traces=tuple(
                MotionFieldRouteTrace(
                    line_id=line_id,
                    root_scene_xy=path,
                    support_mask=support,
                    surface_height_m=np.zeros(3),
                )
                for line_id, path in zip(
                    (center.first_line_id, center.second_line_id), paths
                )
            ),
            search_radius_m=0.12,
            maximum_height_difference_m=0.03,
        )[0]

        rotation = np.array(((0.0, -1.0), (1.0, 0.0)))
        translation = np.array((1.2, -0.4))
        moved_intersection = type(center)(
            first_line_id=center.first_line_id,
            second_line_id=center.second_line_id,
            first_heading_degrees=center.first_heading_degrees + 90.0,
            second_heading_degrees=center.second_heading_degrees + 90.0,
            scene_xy=tuple(np.asarray(center.scene_xy) @ rotation.T + translation),
            first_progress_m=center.first_progress_m,
            second_progress_m=center.second_progress_m,
        )
        moved = transition_opportunities(
            intersections=(moved_intersection,),
            route_traces=tuple(
                MotionFieldRouteTrace(
                    line_id=line_id,
                    root_scene_xy=path @ rotation.T + translation,
                    support_mask=support,
                    surface_height_m=np.zeros(3),
                )
                for line_id, path in zip(
                    (center.first_line_id, center.second_line_id), paths
                )
            ),
            search_radius_m=0.12,
            maximum_height_difference_m=0.03,
        )[0]

        self.assertEqual(
            (moved.status, moved.first_frame, moved.second_frame),
            (original.status, original.first_frame, original.second_frame),
        )


if __name__ == "__main__":
    unittest.main()
