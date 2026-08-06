from __future__ import annotations

import unittest

import numpy as np

from mm_sonic.motionbricks_hill_ik import (
    FootPhase,
    _next_foot_phase,
    _project_stance_targets,
    _required_swing_lift,
)


class HillFootIKContractTest(unittest.TestCase):
    def test_stance_hysteresis(self) -> None:
        self.assertEqual(
            _next_foot_phase(FootPhase.SWING, 0.02, 0.20),
            FootPhase.STANCE,
        )
        self.assertEqual(
            _next_foot_phase(FootPhase.STANCE, 0.06, 0.60),
            FootPhase.STANCE,
        )
        self.assertEqual(
            _next_foot_phase(FootPhase.STANCE, 0.08, 0.20),
            FootPhase.RELEASE,
        )
        self.assertEqual(
            _next_foot_phase(FootPhase.RELEASE, 0.08, 0.80),
            FootPhase.SWING,
        )

    def test_stance_targets_follow_per_probe_terrain_height(self) -> None:
        centers = np.asarray(((1.0, 0.0, 0.2), (2.0, 0.0, 0.2)))
        radii = np.asarray((0.02, 0.03))
        targets = _project_stance_targets(
            centers, radii, lambda xy: 0.1 * float(xy[0])
        )
        np.testing.assert_allclose(
            targets,
            ((1.0, 0.0, 0.12), (2.0, 0.0, 0.23)),
        )

    def test_swing_lift_is_upward_only(self) -> None:
        self.assertEqual(
            _required_swing_lift(
                np.asarray(((0.0, 0.0, 0.10),)),
                np.asarray((0.02,)),
                lambda xy: 0.0,
            ),
            0.0,
        )
        self.assertAlmostEqual(
            _required_swing_lift(
                np.asarray(((0.0, 0.0, 0.01),)),
                np.asarray((0.02,)),
                lambda xy: 0.0,
            ),
            0.025,
        )


if __name__ == "__main__":
    unittest.main()
