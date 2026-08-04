import importlib.util
from pathlib import Path
import unittest

import numpy as np


_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "run_g1_ardy_terrain_bridge.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_g1_ardy_terrain_bridge", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class ArdyTerrainBridgeTests(unittest.TestCase):
    def test_stance_height_shift_uses_only_the_supported_foot(self):
        clearance = np.array(
            ((0.02, 0.001), (-0.003, 0.05)), dtype=np.float64
        )
        support = np.array(
            ((False, True), (True, False)), dtype=np.bool_
        )

        result = _MODULE._stance_height_shift(clearance, support)

        np.testing.assert_allclose(result, (0.001, -0.003))

    def test_schedule_always_has_one_support_and_reaches_both_feet(self):
        start = np.array(((0.0, 0.0, 0.1), (0.0, 0.2, 0.2)))
        stop = np.array(((0.1, 0.0, 0.1), (0.1, 0.2, 0.2)))

        targets, support = _MODULE._two_step_schedule(
            start_feet=start,
            stop_feet=stop,
            frame_count=50,
            first_landing_frame=12,
            second_liftoff_frame=14,
            second_landing_frame=34,
            initial_support_foot=1,
            final_support_foot=0,
        )

        self.assertTrue(support.any(axis=1).all())
        np.testing.assert_allclose(targets[0], start)
        np.testing.assert_allclose(targets[-1], stop)
        self.assertTrue(support[0, 1])
        self.assertTrue(support[20, 0])
        self.assertTrue(support[-1, 0])
