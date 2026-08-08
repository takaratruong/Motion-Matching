import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_path_motion_grid import (
    StaircasePathContract,
    build_grid_playlist,
    build_parallel_grid_playlist,
    classify_staircase_path,
    classify_lane,
    horizontal_grid_lanes,
    parallel_path_grid,
    staircase_parallel_path_grid,
)


class HorizontalPathGridTests(unittest.TestCase):
    def test_staircase_grid_centers_twenty_centimeter_lattice_on_elevated_band(self):
        points = np.array(
            [
                (x, y)
                for x in np.linspace(0.0, 1.0, 6)
                for y in np.linspace(-0.51, 0.49, 6)
            ],
            dtype=np.float64,
        )

        paths = staircase_parallel_path_grid(
            center_start_scene_xy=(-0.5, 0.8),
            heading_scene_xy=(1.0, 0.0),
            path_length_m=2.0,
            elevated_scene_xy=points,
            spacing_m=0.20,
        )

        np.testing.assert_allclose(
            [path.start_scene_xy[1] for path in paths],
            (-0.41, -0.21, -0.01, 0.19, 0.39),
            atol=1.0e-12,
        )
        self.assertTrue(
            all(-0.51 < path.start_scene_xy[1] < 0.49 for path in paths)
        )

    def test_staircase_profile_requires_approach_elevation_and_opposite_exit(self):
        path = parallel_path_grid(
            center_start_scene_xy=(0.0, 0.0),
            heading_scene_xy=(1.0, 0.0),
            path_length_m=2.0,
            minimum_lateral_offset_m=0.0,
            maximum_lateral_offset_m=0.0,
            spacing_m=0.2,
        )[0]

        def staircase(points):
            x = np.asarray(points)[:, 0]
            return np.where((x >= 0.5) & (x <= 1.5), 0.2, 0.0)

        contract = classify_staircase_path(
            path=path, sample_surface=staircase
        )

        self.assertIsInstance(contract, StaircasePathContract)
        self.assertEqual(
            contract.classification, "staircase_intersecting"
        )
        self.assertEqual(
            contract.ordered_surface_heights_m, (0.0, 0.2, 0.0)
        )

    def test_flat_path_is_explicitly_excluded(self):
        path = parallel_path_grid(
            center_start_scene_xy=(0.0, 0.0),
            heading_scene_xy=(1.0, 0.0),
            path_length_m=2.0,
            minimum_lateral_offset_m=0.0,
            maximum_lateral_offset_m=0.0,
            spacing_m=0.2,
        )[0]

        contract = classify_staircase_path(
            path=path,
            sample_surface=lambda points: np.zeros(len(points)),
        )

        self.assertEqual(contract.classification, "flat_only_excluded")

    def test_parallel_grid_uses_runtime_heading_bounds_and_spacing(self):
        paths = parallel_path_grid(
            center_start_scene_xy=(1.2, -0.4),
            heading_scene_xy=(3.0, 4.0),
            path_length_m=2.25,
            minimum_lateral_offset_m=-0.30,
            maximum_lateral_offset_m=0.30,
            spacing_m=0.15,
        )

        self.assertEqual(len(paths), 5)
        np.testing.assert_allclose(
            [path.lateral_offset_m for path in paths],
            (-0.30, -0.15, 0.0, 0.15, 0.30),
            atol=1.0e-12,
        )
        heading = np.array((0.6, 0.8))
        lateral = np.array((-0.8, 0.6))
        for path in paths:
            expected_start = (
                np.array((1.2, -0.4))
                + path.lateral_offset_m * lateral
            )
            np.testing.assert_allclose(
                path.start_scene_xy, expected_start, atol=1.0e-12
            )
            np.testing.assert_allclose(
                np.array(path.stop_scene_xy) - np.array(path.start_scene_xy),
                2.25 * heading,
                atol=1.0e-12,
            )

    def test_parallel_grid_is_rotation_equivariant(self):
        first = parallel_path_grid(
            center_start_scene_xy=(0.2, -0.1),
            heading_scene_xy=(1.0, 0.0),
            path_length_m=1.7,
            minimum_lateral_offset_m=-0.2,
            maximum_lateral_offset_m=0.2,
            spacing_m=0.2,
        )
        second = parallel_path_grid(
            center_start_scene_xy=(0.1, 0.2),
            heading_scene_xy=(0.0, 1.0),
            path_length_m=1.7,
            minimum_lateral_offset_m=-0.2,
            maximum_lateral_offset_m=0.2,
            spacing_m=0.2,
        )
        rotation = np.array(((0.0, -1.0), (1.0, 0.0)))

        for original, rotated in zip(first, second):
            np.testing.assert_allclose(
                rotated.start_scene_xy,
                np.asarray(original.start_scene_xy) @ rotation.T,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                rotated.stop_scene_xy,
                np.asarray(original.stop_scene_xy) @ rotation.T,
                atol=1.0e-12,
            )

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

    def test_parallel_playlist_preserves_runtime_path_geometry(self):
        paths = parallel_path_grid(
            center_start_scene_xy=(0.0, 0.0),
            heading_scene_xy=(0.6, 0.8),
            path_length_m=1.5,
            minimum_lateral_offset_m=-0.15,
            maximum_lateral_offset_m=0.15,
            spacing_m=0.15,
        )

        arrays, metadata = build_parallel_grid_playlist(
            (
                (paths[0], self.connector(1.0)),
                (paths[2], self.connector(2.0)),
            ),
            hold_frames=1,
        )

        self.assertEqual(arrays["joint_position"].shape, (10, 29))
        self.assertEqual(metadata["schema"], "g1-parallel-path-playlist/v1")
        self.assertEqual(
            [item["path_id"] for item in metadata["segments"]],
            [paths[0].path_id, paths[2].path_id],
        )
        self.assertEqual(
            metadata["segments"][0]["start_scene_xy"],
            list(paths[0].start_scene_xy),
        )

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
