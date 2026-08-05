import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_path_motion_grid import (
    build_grid_playlist,
    classify_lane,
    horizontal_grid_lanes,
)


class HorizontalPathGridTests(unittest.TestCase):
    def test_grid_has_exact_twenty_centimeter_lanes(self):
        lanes = horizontal_grid_lanes(
            start_x=-1.2795985755,
            stop_x=1.2590216406,
            minimum_y=-1.0,
            maximum_y=1.0,
            spacing_m=0.2,
        )

        self.assertEqual(len(lanes), 11)
        np.testing.assert_allclose(
            [lane.center_y_m for lane in lanes],
            np.linspace(-1.0, 1.0, 11),
            atol=1.0e-12,
        )
        self.assertEqual(lanes[0].lane_id, "lane-neg-1p0")
        self.assertEqual(lanes[5].lane_id, "lane-zero-0p0")
        self.assertEqual(lanes[-1].lane_id, "lane-pos-1p0")
        self.assertEqual(lanes[5].start_scene_xy, (-1.2795985755, 0.0))
        self.assertEqual(lanes[5].stop_scene_xy, (1.2590216406, 0.0))

    def test_grid_rejects_spacing_that_misses_upper_bound(self):
        with self.assertRaises(ContractError):
            horizontal_grid_lanes(
                start_x=-1.0,
                stop_x=1.0,
                minimum_y=-1.0,
                maximum_y=1.0,
                spacing_m=0.3,
            )

    def test_lane_classification_never_hides_a_gap(self):
        self.assertEqual(
            classify_lane(
                path_length_m=2.5,
                covered_intervals=((0.0, 2.5),),
                has_certified_placement=True,
            ),
            "full",
        )
        self.assertEqual(
            classify_lane(
                path_length_m=2.5,
                covered_intervals=((0.0, 2.4),),
                has_certified_placement=True,
            ),
            "partial",
        )
        self.assertEqual(
            classify_lane(
                path_length_m=2.5,
                covered_intervals=(),
                has_certified_placement=False,
            ),
            "infeasible",
        )

    @staticmethod
    def connector(value, frames=3):
        return {
            "joint_position": np.full((frames, 29), value, dtype=np.float64),
            "root_position_world": np.column_stack(
                (
                    np.linspace(value, value + 0.2, frames),
                    np.full(frames, value),
                    np.full(frames, 0.8),
                )
            ),
            "root_orientation_world_wxyz": np.tile(
                (1.0, 0.0, 0.0, 0.0), (frames, 1)
            ),
        }

    def test_playlist_preserves_motion_and_marks_teleports(self):
        lanes = horizontal_grid_lanes(
            start_x=-1.0,
            stop_x=1.0,
            minimum_y=-0.2,
            maximum_y=0.2,
            spacing_m=0.2,
        )
        first = self.connector(1.0)
        second = self.connector(2.0)

        arrays, metadata = build_grid_playlist(
            (
                (lanes[0], "full", first),
                (lanes[2], "partial", second),
            ),
            hold_frames=2,
        )

        self.assertEqual(arrays["joint_position"].shape, (14, 29))
        np.testing.assert_array_equal(
            arrays["joint_position"][2:5], first["joint_position"]
        )
        np.testing.assert_array_equal(
            arrays["joint_position"][9:12], second["joint_position"]
        )
        self.assertEqual(
            [item["lane_id"] for item in metadata["segments"]],
            [lanes[0].lane_id, lanes[2].lane_id],
        )
        self.assertEqual(metadata["segments"][0]["motion_frames"], [2, 5])
        self.assertEqual(metadata["segments"][1]["motion_frames"], [9, 12])
        self.assertEqual(metadata["teleport_boundaries"], [7])

    def test_playlist_rejects_extra_connector_arrays(self):
        lane = horizontal_grid_lanes(
            start_x=-1.0,
            stop_x=1.0,
            minimum_y=0.0,
            maximum_y=0.0,
            spacing_m=0.2,
        )[0]
        connector = self.connector(0.0)
        connector["unexpected"] = np.zeros(3)

        with self.assertRaises(ContractError):
            build_grid_playlist(((lane, "full", connector),), hold_frames=2)


if __name__ == "__main__":
    unittest.main()
