import unittest

import numpy as np

from mm_sonic.torch_path_terrain_phases import TerrainPathPhase
from resources.run_g1_horizontal_grid_phase_coverage import (
    _classify_lane_phases,
    _ordered_phase_records,
    _phase_queries,
)


class RunG1HorizontalGridPhaseCoverageTests(unittest.TestCase):
    @staticmethod
    def phase(height=0.3):
        return TerrainPathPhase(
            kind="interior",
            start_m=0.6,
            stop_m=1.6,
            start_scene_xy=(-0.6, 0.0),
            stop_scene_xy=(0.4, 0.0),
            mean_support_height_m=height,
            normalized_support_height_m=(-0.05, 0.05),
        )

    def test_short_phase_queries_include_two_contact_windows(self):
        queries = _phase_queries(
            phase=self.phase(),
            step_width_m=0.20,
            sample_surface=lambda points: np.where(
                np.asarray(points)[..., 1] < 0.0, 0.25, 0.35
            ),
        )

        self.assertTrue(queries)
        self.assertGreaterEqual(min(len(query.foot_order) for query in queries), 2)
        self.assertLessEqual(min(len(query.foot_order) for query in queries), 3)

    def test_phase_queries_ignore_common_surface_height_offset(self):
        first = _phase_queries(
            phase=self.phase(0.3),
            step_width_m=0.20,
            sample_surface=lambda points: np.where(
                np.asarray(points)[..., 1] < 0.0, 0.25, 0.35
            ),
        )
        second = _phase_queries(
            phase=self.phase(0.7),
            step_width_m=0.20,
            sample_surface=lambda points: np.where(
                np.asarray(points)[..., 1] < 0.0, 0.65, 0.75
            ),
        )

        self.assertEqual(
            [query.height_pattern_m for query in first],
            [query.height_pattern_m for query in second],
        )

    def test_lane_classification_keeps_interior_when_edges_fail(self):
        self.assertEqual(
            _classify_lane_phases(
                (
                    {"kind": "mount", "classification": "infeasible"},
                    {"kind": "interior", "classification": "full"},
                    {"kind": "dismount", "classification": "infeasible"},
                )
            ),
            "partial",
        )
        self.assertEqual(
            _classify_lane_phases(
                tuple(
                    {"kind": kind, "classification": "full"}
                    for kind in ("mount", "interior", "dismount")
                )
            ),
            "full",
        )

    def test_report_order_is_lane_then_mount_interior_dismount(self):
        records = (
            {"lane_index": 1, "kind": "interior"},
            {"lane_index": 0, "kind": "dismount"},
            {"lane_index": 0, "kind": "mount"},
            {"lane_index": 1, "kind": "mount"},
            {"lane_index": 0, "kind": "interior"},
        )

        self.assertEqual(
            [
                (item["lane_index"], item["kind"])
                for item in _ordered_phase_records(records)
            ],
            [
                (0, "mount"),
                (0, "interior"),
                (0, "dismount"),
                (1, "mount"),
                (1, "interior"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
