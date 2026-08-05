import unittest

import numpy as np

from mm_sonic.torch_path_motion_placement import (
    PathContactSignature,
    RawContactEvent,
    RawMotionWindow,
)
from resources.run_g1_horizontal_grid_coverage import (
    _grid_lanes_from_args,
    _path_polyline_matcher,
    _parser,
    _score_windows_for_lanes,
    _trim_lane_pool,
)


class RunG1HorizontalGridCoverageTests(unittest.TestCase):
    def test_cli_defaults_to_approved_grid_without_source_override(self):
        parser = _parser()
        destinations = {action.dest for action in parser._actions}
        args = parser.parse_args(
            (
                "--source-dataset",
                "dataset",
                "--target-scene",
                "stair",
                "--g1-xml",
                "g1.xml",
                "--output",
                "output",
            )
        )

        self.assertNotIn("source_clip", destinations)
        self.assertNotIn("source_frame", destinations)
        self.assertEqual(args.spacing_m, 0.2)
        self.assertEqual(len(_grid_lanes_from_args(args)), 11)

    @staticmethod
    def signature(heights):
        return PathContactSignature(
            foot_order=(0, 1, 0, 1),
            forward_m=(0.4, 0.8, 1.2, 1.6),
            lateral_m=(0.1, -0.1, 0.1, -0.1),
            contact_frame=(20, 40, 60, 80),
            height_pattern_m=heights,
        )

    def test_windows_are_scored_independently_for_each_lane(self):
        window = RawMotionWindow(
            source_clip="raw",
            start_frame=0,
            stop_frame=90,
            events=tuple(
                RawContactEvent(
                    frame=frame,
                    foot=foot,
                    position_world_xy=(forward, lateral),
                    surface_height_m=height,
                )
                for frame, foot, forward, lateral, height in (
                    (20, 0, 0.4, 0.1, 0.0),
                    (40, 1, 0.8, -0.1, 0.2),
                    (60, 0, 1.2, 0.1, 0.0),
                    (80, 1, 1.6, -0.1, 0.2),
                )
            ),
            root_start_world_xy=(0.0, 0.0),
            forward_progress_m=1.8,
            heading_error_rad=0.0,
        )

        rows = _score_windows_for_lanes(
            (window,),
            (
                (self.signature((0.0, 0.2, 0.0, 0.2)),),
                (self.signature((0.0, 0.0, 0.0, 0.0)),),
            ),
            path_length_m=1.8,
        )

        self.assertEqual(len(rows), 2)
        self.assertLess(rows[0][0]["cost"], rows[1][0]["cost"])
        self.assertEqual(rows[0][0]["source_clip"], "raw")

    def test_lane_pool_is_bounded_without_losing_coverage_candidates(self):
        rows = [
            {
                "cost": float(index),
                "source_clip": f"clip-{index}",
                "start_frame": index,
                "stop_frame": index + 10,
                "forward_progress_m": progress,
            }
            for index, progress in enumerate(
                (0.5, 0.7, 0.9, 1.1, 2.49, 2.51, 3.0)
            )
        ]

        retained = _trim_lane_pool(
            rows, path_length_m=2.5, maximum_results=4
        )

        self.assertEqual(len(retained), 4)
        self.assertIn(
            "clip-4", {row["source_clip"] for row in retained}
        )
        self.assertIn(
            "clip-5", {row["source_clip"] for row in retained}
        )

    def test_path_polyline_follows_terrain_in_matcher_frame(self):
        lane = _grid_lanes_from_args(
            _parser().parse_args(
                (
                    "--source-dataset",
                    "dataset",
                    "--target-scene",
                    "stair",
                    "--g1-xml",
                    "g1.xml",
                    "--minimum-y",
                    "0",
                    "--maximum-y",
                    "0",
                    "--output",
                    "output",
                )
            )
        )[0]

        points = _path_polyline_matcher(
            lane=lane,
            sample_surface=lambda xy: 0.5 * xy[:, 0],
            target_root_scene_xy=np.zeros(2),
            alignment_yaw=0.0,
            sample_count=3,
        )

        self.assertEqual(points.shape, (3, 3))
        np.testing.assert_allclose(points[:, 0], np.linspace(
            lane.start_scene_xy[0], lane.stop_scene_xy[0], 3
        ))
        np.testing.assert_allclose(points[:, 2], 0.5 * points[:, 0] + 0.03)


if __name__ == "__main__":
    unittest.main()
