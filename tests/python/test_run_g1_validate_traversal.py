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
    def test_accepts_contact_valid_metrics(self):
        _MODULE._enforce_metrics(
            {
                "unsupported_frame_count": 0,
                "maximum_stance_contact_error_m": 0.019,
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
                    "minimum_sole_clearance_m": 0.0,
                    "minimum_supported_sole_points": 8,
                }
            )

    def test_rejects_incomplete_sole_support(self):
        with self.assertRaisesRegex(Exception, "sole support"):
            _MODULE._enforce_metrics(
                {
                    "unsupported_frame_count": 0,
                    "maximum_stance_contact_error_m": 0.0,
                    "minimum_sole_clearance_m": 0.0,
                    "minimum_supported_sole_points": 2,
                }
            )
