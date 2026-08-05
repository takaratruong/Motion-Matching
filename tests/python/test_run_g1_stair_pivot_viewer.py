import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_g1_stair_pivot_viewer",
    ROOT / "resources" / "run_g1_stair_pivot_viewer.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RunG1StairPivotViewerTest(unittest.TestCase):
    def test_parser_exposes_headless_contact_sheet(self):
        args = MODULE._parser().parse_args(
            (
                "--connector",
                "connector.npz",
                "--dataset",
                "dataset",
                "--config",
                "config.json",
                "--g1-xml",
                "g1.xml",
                "--render-contact-sheet",
                "sheet.png",
            )
        )

        self.assertEqual(args.render_contact_sheet, Path("sheet.png"))

    def test_parser_accepts_native_motionbricks_frame_rate(self):
        args = MODULE._parser().parse_args(
            (
                "--connector",
                "connector.npz",
                "--dataset",
                "dataset",
                "--config",
                "config.json",
                "--g1-xml",
                "g1.xml",
                "--frames-per-second",
                "30",
            )
        )

        self.assertEqual(args.frames_per_second, 30.0)

    def test_parser_accepts_grid_summary_and_playlist_metadata(self):
        args = MODULE._parser().parse_args(
            (
                "--connector",
                "connector.npz",
                "--dataset",
                "dataset",
                "--config",
                "config.json",
                "--g1-xml",
                "g1.xml",
                "--grid-summary",
                "grid-summary.json",
                "--playlist-metadata",
                "grid-playlist.json",
            )
        )

        self.assertEqual(args.grid_summary, Path("grid-summary.json"))
        self.assertEqual(
            args.playlist_metadata, Path("grid-playlist.json")
        )

    def test_grid_overlay_loads_all_lanes_and_colors_current_white(self):
        lanes = [
            {
                "lane_id": f"lane-{index}",
                "lane_index": index,
                "center_y_m": -1.0 + 0.2 * index,
                "path_start_scene_xy": [-1.2, -1.0 + 0.2 * index],
                "path_stop_scene_xy": [1.2, -1.0 + 0.2 * index],
                "classification": (
                    "full" if index < 4
                    else "partial" if index < 8
                    else "infeasible"
                ),
                "path_polyline_matcher_xyz": [
                    [-1.2, -1.0 + 0.2 * index, 0.03],
                    [1.2, -1.0 + 0.2 * index, 0.03],
                ],
            }
            for index in range(11)
        ]
        playlist = {
            "schema": "g1-horizontal-grid-playlist/v1",
            "frame_count": 20,
            "segments": [
                {
                    "lane_id": "lane-0",
                    "lane_index": 0,
                    "center_y_m": -1.0,
                    "classification": "full",
                    "segment_frames": [0, 20],
                    "motion_frames": [2, 18],
                }
            ],
            "teleport_boundaries": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "grid-summary.json"
            playlist_path = root / "grid-playlist.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "schema": "g1-horizontal-grid-coverage/v1",
                        "grid": {"lane_count": 11},
                        "lanes": lanes,
                    }
                )
            )
            playlist_path.write_text(json.dumps(playlist))

            overlay = MODULE._load_grid_overlay(
                summary_path, playlist_path, frame_count=20
            )

        self.assertEqual(len(overlay["lanes"]), 11)
        colors = MODULE._grid_line_colors(
            overlay, current_lane_id="lane-0"
        )
        np.testing.assert_allclose(colors[0], (1.0, 1.0, 1.0, 1.0))
        np.testing.assert_allclose(colors[4], (1.0, 0.65, 0.0, 0.9))
        np.testing.assert_allclose(colors[8], (0.9, 0.1, 0.1, 0.9))

    def test_loader_accepts_exact_connector_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "connector.npz"
            np.savez_compressed(
                path,
                joint_position=np.zeros((5, 29)),
                root_position_world=np.zeros((5, 3)),
                root_orientation_world_wxyz=np.tile(
                    (1.0, 0.0, 0.0, 0.0),
                    (5, 1),
                ),
            )

            arrays = MODULE._load_connector(path)

        self.assertEqual(arrays["joint_position"].shape, (5, 29))


if __name__ == "__main__":
    unittest.main()
