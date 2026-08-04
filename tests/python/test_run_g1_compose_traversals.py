import importlib.util
from pathlib import Path
import unittest

import numpy as np


_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "run_g1_compose_traversals.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_g1_compose_traversals", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _segment(offset: float) -> dict[str, np.ndarray]:
    joints = np.zeros((3, 29), dtype=np.float64)
    joints[:, 0] = offset + np.arange(3)
    roots = np.zeros((3, 3), dtype=np.float64)
    roots[:, 0] = offset + np.arange(3)
    quaternions = np.zeros((3, 4), dtype=np.float64)
    quaternions[:, 0] = 1.0
    support = np.ones((3, 2), dtype=np.bool_)
    return {
        "joint_position": joints,
        "root_position_world": roots,
        "root_orientation_world_wxyz": quaternions,
        "source_support_mask": support,
    }


class ComposeTraversalsTests(unittest.TestCase):
    def test_composition_drops_repeated_junction_frames(self):
        first = _segment(0.0)
        second = _segment(2.0)
        third = _segment(4.0)

        result = _MODULE._compose_segments([first, second, third])

        self.assertEqual(result["joint_position"].shape, (7, 29))
        np.testing.assert_array_equal(
            result["segment_boundaries"], (0, 3, 5, 7)
        )
        np.testing.assert_allclose(
            result["root_position_world"][:, 0],
            (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        )

    def test_composition_rejects_discontinuous_junction(self):
        first = _segment(0.0)
        second = _segment(2.2)

        with self.assertRaisesRegex(Exception, "junction"):
            _MODULE._compose_segments([first, second])
