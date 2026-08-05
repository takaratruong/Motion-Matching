import unittest

from resources.run_g1_path_motion_placement import (
    _parser,
    _placement_shortlist_rows,
    _progress_coverage_error,
    _rank_coarse_rows,
    _retain_event_count_diversity,
    _uncovered_intervals,
)


class RunG1PathMotionPlacementTests(unittest.TestCase):
    def test_progress_coverage_prefers_small_overshoot_to_undershoot(self):
        self.assertLess(
            _progress_coverage_error(2.52, 2.5),
            _progress_coverage_error(2.49, 2.5),
        )

    def test_cli_has_no_selected_source_override(self):
        destinations = {action.dest for action in _parser()._actions}

        self.assertNotIn("source_clip", destinations)
        self.assertNotIn("source_frame", destinations)
        self.assertIn("target_scene", destinations)
        self.assertIn("path_start", destinations)
        self.assertIn("path_stop", destinations)

    def test_coarse_ranking_is_deterministic_without_family_priority(self):
        rows = [
            {"cost": 1.0, "source_clip": "stair-z", "start_frame": 2},
            {"cost": 1.0, "source_clip": "curb-a", "start_frame": 3},
            {"cost": 0.5, "source_clip": "misc", "start_frame": 8},
            {"cost": 1.0, "source_clip": "curb-a", "start_frame": 1},
        ]

        ranked = _rank_coarse_rows(rows, maximum_results=3)

        self.assertEqual(
            [
                (row["source_clip"], row["start_frame"])
                for row in ranked
            ],
            [("misc", 8), ("curb-a", 1), ("curb-a", 3)],
        )

    def test_coarse_retention_preserves_longer_contact_patterns(self):
        rows = [
            {
                "cost": float(cost),
                "source_clip": "one",
                "start_frame": index,
                "stop_frame": index + 10,
                "events": [{}] * count,
            }
            for index, (count, cost) in enumerate(
                ((4, 0.1), (4, 0.2), (4, 0.3), (7, 0.8), (7, 0.9))
            )
        ]

        retained = _retain_event_count_diversity(rows, per_count=2)

        self.assertEqual(
            [len(row["events"]) for row in retained], [4, 4, 7, 7]
        )

    def test_placement_shortlist_preserves_path_length_matches(self):
        rows = [
            {
                "cost": cost,
                "source_clip": f"clip-{index}",
                "start_frame": index,
                "stop_frame": index + 10,
                "forward_progress_m": progress,
            }
            for index, (cost, progress) in enumerate(
                ((0.1, 0.5), (0.6, 2.3), (5.0, 2.5), (10.0, 2.49))
            )
        ]

        retained = _placement_shortlist_rows(
            rows, path_length_m=2.5, maximum_results=3
        )

        self.assertEqual(len(retained), 3)
        self.assertIn("clip-1", {row["source_clip"] for row in retained})

    def test_coverage_reports_prefix_and_suffix_gaps(self):
        self.assertEqual(
            _uncovered_intervals(
                path_length_m=3.0,
                covered_intervals=((0.4, 1.2), (1.1, 2.4)),
            ),
            ((0.0, 0.4), (2.4, 3.0)),
        )


if __name__ == "__main__":
    unittest.main()
