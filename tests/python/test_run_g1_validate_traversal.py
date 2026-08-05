import importlib.util
from pathlib import Path
import unittest


_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "run_g1_validate_traversal.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_g1_validate_traversal", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class ValidateTraversalTests(unittest.TestCase):
    def test_parser_accepts_heading_and_planned_footprints(self):
        args = _MODULE.parser().parse_args(
            [
                "--input",
                "traversal.npz",
                "--target-dataset",
                "dataset",
                "--config",
                "config.json",
                "--g1-xml",
                "g1.xml",
                "--expected-heading-degrees",
                "45",
                "--planned-footprints",
                "footprints.json",
                "--output",
                "metrics.json",
            ]
        )

        self.assertEqual(args.expected_heading_degrees, 45.0)
        self.assertEqual(args.minimum_supported_sole_points, 3)

    def test_accepts_contact_valid_metrics(self):
        _MODULE._enforce_metrics(
            {
                "unsupported_frame_count": 0,
                "maximum_stance_contact_error_m": 0.019,
                "maximum_stance_horizontal_step_m": 0.009,
                "minimum_sole_clearance_m": -0.024,
                "minimum_supported_sole_points": 3,
            }
        )

    def test_rejects_an_unsupported_frame(self):
        with self.assertRaisesRegex(Exception, "unsupported"):
            _MODULE._enforce_metrics(
                {
                    "unsupported_frame_count": 1,
                    "maximum_stance_contact_error_m": 0.0,
                    "maximum_stance_horizontal_step_m": 0.0,
                    "minimum_sole_clearance_m": 0.0,
                    "minimum_supported_sole_points": 8,
                }
            )

    def test_accepts_a_bounded_dynamic_flight_phase(self):
        _MODULE._enforce_metrics(
            {
                "unsupported_frame_count": 5,
                "maximum_stance_contact_error_m": 0.019,
                "maximum_stance_horizontal_step_m": 0.009,
                "minimum_sole_clearance_m": -0.024,
                "minimum_supported_sole_points": 3,
            },
            maximum_unsupported_frames=5,
        )

    def test_rejects_incomplete_sole_support(self):
        with self.assertRaisesRegex(Exception, "sole support"):
            _MODULE._enforce_metrics(
                {
                    "unsupported_frame_count": 0,
                    "maximum_stance_contact_error_m": 0.0,
                    "maximum_stance_horizontal_step_m": 0.0,
                    "minimum_sole_clearance_m": 0.0,
                    "minimum_supported_sole_points": 2,
                }
            )

    def test_rejects_horizontal_stance_sliding(self):
        with self.assertRaisesRegex(Exception, "slides"):
            _MODULE._enforce_metrics(
                {
                    "unsupported_frame_count": 0,
                    "maximum_stance_contact_error_m": 0.0,
                    "maximum_stance_horizontal_step_m": 0.011,
                    "minimum_sole_clearance_m": 0.0,
                    "minimum_supported_sole_points": 8,
                }
            )

    def test_rejects_terminal_mid_swing_and_missing_provenance(self):
        metrics = {
            "unsupported_frame_count": 0,
            "maximum_stance_contact_error_m": 0.0,
            "maximum_stance_horizontal_step_m": 0.0,
            "minimum_sole_clearance_m": 0.0,
            "minimum_supported_sole_points": 8,
            "terminal_complete_support": False,
            "provenance_coverage": 0.5,
        }
        with self.assertRaisesRegex(Exception, "terminal"):
            _MODULE._enforce_metrics(metrics)

        metrics["terminal_complete_support"] = True
        with self.assertRaisesRegex(Exception, "provenance"):
            _MODULE._enforce_metrics(metrics)
