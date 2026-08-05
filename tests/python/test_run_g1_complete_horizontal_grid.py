import unittest

import numpy as np

from resources.run_g1_complete_horizontal_grid import _build_playlist


class CompleteHorizontalGridTests(unittest.TestCase):
    @staticmethod
    def connector(offset, frames):
        return {
            "joint_position": np.full((frames, 29), offset),
            "root_position_world": np.column_stack(
                (
                    np.linspace(0.0, 1.0, frames),
                    np.full(frames, offset),
                    np.full(frames, 0.8),
                )
            ),
            "root_orientation_world_wxyz": np.tile(
                (1.0, 0.0, 0.0, 0.0), (frames, 1)
            ),
        }

    def test_playlist_contains_only_complete_connectors_and_all_lane_results(self):
        lanes = [
            {
                "lane_id": f"lane-{index}",
                "lane_index": index,
                "center_y_m": -1.0 + 0.2 * index,
            }
            for index in range(11)
        ]
        connectors = {
            "lane-0": self.connector(0.0, 4),
            "lane-10": self.connector(1.0, 5),
        }

        arrays, metadata, results = _build_playlist(
            lanes=lanes,
            connectors=connectors,
            hold_frames=2,
        )

        self.assertEqual(len(results), 11)
        self.assertEqual(
            [row["classification"] for row in results].count("complete"),
            2,
        )
        self.assertEqual(
            [row["classification"] for row in results].count("unresolved"),
            9,
        )
        self.assertEqual(len(metadata["segments"]), 2)
        self.assertEqual(metadata["segments"][0]["motion_frames"], [2, 6])
        self.assertEqual(metadata["segments"][1]["motion_frames"], [10, 15])
        self.assertEqual(metadata["frame_count"], 17)
        self.assertEqual(len(arrays["joint_position"]), 17)


if __name__ == "__main__":
    unittest.main()
