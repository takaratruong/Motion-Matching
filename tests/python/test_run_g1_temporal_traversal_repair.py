import importlib.util
from pathlib import Path
import unittest

import numpy as np


_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "run_g1_temporal_traversal_repair.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_g1_temporal_traversal_repair", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class TemporalTraversalRepairTests(unittest.TestCase):
    def test_spike_windows_merge_neighboring_joint_and_root_spikes(self):
        joints = np.zeros((30, 29), dtype=np.float64)
        roots = np.zeros((30, 3), dtype=np.float64)
        joints[12:, 4] = 0.8
        roots[14:, 0] = 0.1

        windows = _MODULE._spike_windows(
            joints=joints,
            roots=roots,
            joint_threshold_rad=0.35,
            root_threshold_m=0.04,
            radius_frames=4,
        )

        self.assertEqual(windows, [(7, 18)])

    def test_smooth_window_preserves_endpoints_and_removes_joint_jump(self):
        joints = np.zeros((12, 29), dtype=np.float64)
        roots = np.zeros((12, 3), dtype=np.float64)
        quaternions = np.zeros((12, 4), dtype=np.float64)
        quaternions[:, 0] = 1.0
        joints[6:, 2] = 1.0
        roots[6:, 1] = 0.2

        _MODULE._smooth_window(
            joints=joints,
            roots=roots,
            quaternions=quaternions,
            start=2,
            stop=10,
        )

        self.assertEqual(joints[2, 2], 0.0)
        self.assertEqual(joints[10, 2], 1.0)
        self.assertLess(np.abs(np.diff(joints[:, 2])).max(), 0.2)
        np.testing.assert_allclose(
            np.linalg.norm(quaternions[2:11], axis=1), 1.0
        )
