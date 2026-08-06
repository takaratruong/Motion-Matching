import unittest

import numpy as np

from mm_sonic.joints import ContractError
from resources.run_g1_horizontal_seed_chains import (
    _build_playlist,
    _chain_boundaries,
)


class HorizontalSeedChainTests(unittest.TestCase):
    @staticmethod
    def seeds(intervals=((0.0, 0.7), (0.65, 1.8), (1.5, 2.5))):
        return tuple(
            {"kind": kind, "path_interval_m": list(interval)}
            for kind, interval in zip(
                ("mount", "interior", "dismount"), intervals
            )
        )

    def test_boundaries_report_overlap_without_a_gap(self):
        boundaries = _chain_boundaries(
            self.seeds(), maximum_gap_m=0.40
        )

        self.assertAlmostEqual(boundaries[0]["overlap_m"], 0.05)
        self.assertEqual(boundaries[0]["uncovered_m"], 0.0)
        self.assertAlmostEqual(boundaries[1]["overlap_m"], 0.30)
        self.assertEqual(boundaries[1]["uncovered_m"], 0.0)

    def test_boundaries_report_a_short_uncovered_interval(self):
        boundaries = _chain_boundaries(
            self.seeds(((0.0, 0.5), (0.65, 1.8), (1.5, 2.5))),
            maximum_gap_m=0.40,
        )

        self.assertAlmostEqual(boundaries[0]["uncovered_m"], 0.15)
        self.assertEqual(boundaries[0]["overlap_m"], 0.0)

    def test_boundaries_reject_an_excessive_gap(self):
        with self.assertRaisesRegex(ContractError, "uncovered boundary"):
            _chain_boundaries(
                self.seeds(
                    ((0.0, 0.2), (0.65, 1.8), (1.5, 2.5))
                ),
                maximum_gap_m=0.40,
            )

    def test_boundaries_reject_invalid_seed_order(self):
        seeds = list(self.seeds())
        seeds[1]["kind"] = "dismount"

        with self.assertRaisesRegex(ContractError, "seed order"):
            _chain_boundaries(tuple(seeds), maximum_gap_m=0.40)

    @staticmethod
    def connector(value):
        quaternion = np.zeros((2, 4), dtype=np.float64)
        quaternion[:, 0] = 1.0
        return {
            "joint_position": np.full(
                (2, 29), value, dtype=np.float64
            ),
            "root_position_world": np.full(
                (2, 3), value, dtype=np.float64
            ),
            "root_orientation_world_wxyz": quaternion,
        }

    def test_playlist_preserves_seed_interiors_and_marks_teleports(self):
        originals = [
            self.connector(value) for value in (1.0, 2.0, 3.0)
        ]
        chains = (
            {
                "lane_id": "lane",
                "seeds": tuple(
                    {
                        "kind": kind,
                        "arrays": arrays,
                        "source_clip": f"source-{kind}",
                        "path_interval_m": [index, index + 0.8],
                    }
                    for index, (kind, arrays) in enumerate(
                        zip(
                            ("mount", "interior", "dismount"),
                            originals,
                        )
                    )
                ),
            },
        )

        playlist, metadata = _build_playlist(
            chains, hold_frames=1
        )

        self.assertEqual(len(playlist["joint_position"]), 12)
        for segment, original in zip(metadata["segments"], originals):
            start, stop = segment["motion_frames"]
            np.testing.assert_array_equal(
                playlist["joint_position"][start:stop],
                original["joint_position"],
            )
        self.assertEqual(metadata["teleport_boundaries"], [4, 8])
        self.assertEqual(
            [segment["kind"] for segment in metadata["segments"]],
            ["mount", "interior", "dismount"],
        )


if __name__ == "__main__":
    unittest.main()
