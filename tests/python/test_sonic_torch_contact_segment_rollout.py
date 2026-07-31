import unittest

from mm_sonic.torch_contact_segment_rollout import (
    evaluate_contact_segment_acceptance,
)


class ContactSegmentAcceptanceTests(unittest.TestCase):
    def test_every_acceptance_boundary_is_inclusive(self):
        metrics = {
            "unsupported_fraction": 0.15,
            "longest_unsupported_frames": 10,
            "stance_slide_m": 0.35,
            "minimum_foot_clearance_m": -0.03,
            "lost_source_support_fraction": 0.05,
            "committed_sequence_violations": 0,
            "reached_upper_landing": True,
        }

        gates = evaluate_contact_segment_acceptance(metrics)

        self.assertEqual(
            [(gate.name, gate.passed) for gate in gates],
            [
                ("unsupported_fraction", True),
                ("longest_unsupported", True),
                ("stance_slide", True),
                ("minimum_clearance", True),
                ("lost_source_support", True),
                ("committed_sequence", True),
                ("upper_landing", True),
            ],
        )

    def test_each_failed_metric_fails_only_its_gate(self):
        baseline = {
            "unsupported_fraction": 0.0,
            "longest_unsupported_frames": 0,
            "stance_slide_m": 0.0,
            "minimum_foot_clearance_m": 0.0,
            "lost_source_support_fraction": 0.0,
            "committed_sequence_violations": 0,
            "reached_upper_landing": True,
        }
        cases = (
            ("unsupported_fraction", 0.151, "unsupported_fraction"),
            ("longest_unsupported_frames", 11, "longest_unsupported"),
            ("stance_slide_m", 0.351, "stance_slide"),
            ("minimum_foot_clearance_m", -0.031, "minimum_clearance"),
            ("lost_source_support_fraction", 0.051, "lost_source_support"),
            ("committed_sequence_violations", 1, "committed_sequence"),
            ("reached_upper_landing", False, "upper_landing"),
        )
        for field, value, failed_gate in cases:
            with self.subTest(field=field):
                metrics = dict(baseline)
                metrics[field] = value
                failed = [
                    gate.name
                    for gate in evaluate_contact_segment_acceptance(metrics)
                    if not gate.passed
                ]
                self.assertEqual(failed, [failed_gate])


if __name__ == "__main__":
    unittest.main()
