import importlib.util
from pathlib import Path
import unittest

import numpy as np


_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "run_g1_reverse_traversal.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_g1_reverse_traversal", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class ReverseTraversalTests(unittest.TestCase):
    def test_phase_boundaries_reverse_contact_order(self):
        result = _MODULE._reverse_phase_boundaries(
            np.array((0, 35, 88, 103), dtype=np.int64), 103
        )

        np.testing.assert_array_equal(result, (0, 15, 68, 103))

    def test_phase_boundaries_require_full_frame_extent(self):
        with self.assertRaisesRegex(Exception, "phase boundaries"):
            _MODULE._reverse_phase_boundaries(
                np.array((0, 10, 20, 29), dtype=np.int64), 30
            )
