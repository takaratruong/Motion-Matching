import unittest

import numpy as np

from resources.run_g1_horizontal_phase_composite import (
    _accepted_metrics,
    _oriented_sole_contact_metrics,
    _phase_source_slice,
)


class HorizontalPhaseCompositeRunnerTests(unittest.TestCase):
    def test_reads_selected_source_slice_by_phase_kind(self):
        summary = {
            "phases": [
                {
                    "kind": "mount",
                    "selected": {
                        "source_clip": "clip-a",
                        "start_frame": 12,
                        "stop_frame": 34,
                    },
                }
            ]
        }

        self.assertEqual(
            _phase_source_slice(summary, "mount"),
            ("clip-a", 12, 34),
        )

    def test_frozen_acceptance_gates(self):
        good = {
            "minimum_sole_clearance_m": -0.03,
            "maximum_stance_error_m": 0.03,
            "unsupported_frame_count": 0,
        }
        self.assertTrue(_accepted_metrics(good))
        for key, value in (
            ("minimum_sole_clearance_m", -0.031),
            ("maximum_stance_error_m", 0.031),
            ("unsupported_frame_count", 1),
        ):
            bad = dict(good)
            bad[key] = value
            self.assertFalse(_accepted_metrics(bad))

    def test_oriented_sole_contact_counts_edge_contact_as_support(self):
        clearance = np.full((3, 2, 7), 0.2, dtype=np.float64)
        clearance[0, 0, 2] = 0.0
        clearance[1, 1, 4] = -0.01
        clearance[2, 0, 0] = 0.02

        support, metrics = _oriented_sole_contact_metrics(clearance)

        np.testing.assert_array_equal(
            support,
            np.array(((True, False), (False, True), (True, False))),
        )
        self.assertEqual(metrics["unsupported_frame_count"], 0)
        self.assertAlmostEqual(metrics["maximum_stance_error_m"], 0.02)
        self.assertAlmostEqual(metrics["minimum_sole_clearance_m"], -0.01)


if __name__ == "__main__":
    unittest.main()
