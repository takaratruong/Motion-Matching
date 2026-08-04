import importlib.util
from pathlib import Path
import unittest

import numpy as np


_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "run_g1_resample_traversal.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_g1_resample_traversal", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class ResampleTraversalTests(unittest.TestCase):
    def test_upsampling_preserves_endpoints_and_support(self):
        joints = np.zeros((3, 29), dtype=np.float64)
        joints[:, 0] = (0.0, 1.0, 2.0)
        roots = np.zeros((3, 3), dtype=np.float64)
        roots[:, 0] = (0.0, 0.1, 0.2)
        quaternions = np.zeros((3, 4), dtype=np.float64)
        quaternions[:, 0] = 1.0
        support = np.array(
            ((True, False), (False, True), (True, False)),
            dtype=np.bool_,
        )

        result = _MODULE._resample(
            joints=joints,
            roots=roots,
            quaternions=quaternions,
            support=support,
            factor=2,
        )

        self.assertEqual(result["joint_position"].shape, (5, 29))
        np.testing.assert_allclose(result["joint_position"][[0, -1]], joints[[0, -1]])
        np.testing.assert_allclose(result["root_position_world"][[0, -1]], roots[[0, -1]])
        np.testing.assert_array_equal(
            result["source_support_mask"][[0, -1]], support[[0, -1]]
        )
        np.testing.assert_array_equal(
            result["source_support_mask"],
            support[[0, 0, 1, 2, 2]],
        )
        np.testing.assert_allclose(
            np.linalg.norm(
                result["root_orientation_world_wxyz"], axis=1
            ),
            1.0,
        )

    def test_rejects_a_non_integer_factor(self):
        arrays = {
            "joints": np.zeros((3, 29)),
            "roots": np.zeros((3, 3)),
            "quaternions": np.tile((1.0, 0.0, 0.0, 0.0), (3, 1)),
            "support": np.ones((3, 2), dtype=np.bool_),
        }

        with self.assertRaisesRegex(Exception, "factor"):
            _MODULE._resample(**arrays, factor=1)
