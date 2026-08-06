from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

from mm_sonic.motionbricks_hill import GentleHillProfile
from mm_sonic.motionbricks_hill_ik import (
    FootPhase,
    MotionBricksHillFootIK,
    _next_foot_phase,
    _project_stance_targets,
    _required_swing_lift,
)

MOTIONBRICKS_G1_SCENE = Path(
    "/home/ubuntu/projects/gear-sonic-pinned-60de0df/"
    "motionbricks/assets/skeletons/g1/scene_29dof.xml"
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


@unittest.skipUnless(MOTIONBRICKS_G1_SCENE.is_file(), "G1 scene unavailable")
class HillFootIKModelTest(unittest.TestCase):
    def setUp(self) -> None:
        import mujoco

        self.model = mujoco.MjModel.from_xml_path(
            str(MOTIONBRICKS_G1_SCENE)
        )

    def test_apply_preserves_root_and_non_leg_qpos(self) -> None:
        raw = self.model.qpos0.copy()
        raw[:3] = (3.25, 0.0, 1.25)
        solver = MotionBricksHillFootIK(
            self.model, GentleHillProfile().height
        )
        solver.apply(raw, 1.0 / 30.0)

        result = solver.apply(raw, 1.0 / 30.0)

        np.testing.assert_array_equal(result.qpos[:7], raw[:7])
        np.testing.assert_array_equal(
            result.qpos[solver.non_leg_qpos_addresses],
            raw[solver.non_leg_qpos_addresses],
        )
        self.assertTrue(np.isfinite(result.qpos).all())
        self.assertLessEqual(
            result.diagnostics.maximum_joint_correction_rad,
            0.35 + 1.0e-9,
        )

    def test_static_flat_pose_enters_stance(self) -> None:
        raw = self.model.qpos0.copy()
        solver = MotionBricksHillFootIK(self.model, lambda xy: 0.0)
        solver.apply(raw, 1.0 / 30.0)

        result = solver.apply(raw, 1.0 / 30.0)

        self.assertTrue(result.diagnostics.accepted)
        self.assertEqual(
            result.diagnostics.phases,
            (FootPhase.STANCE, FootPhase.STANCE),
        )
        self.assertLessEqual(
            result.diagnostics.maximum_target_residual_m,
            0.015,
        )

    def test_bounded_stance_acquisition_reduces_slope_penetration(
        self,
    ) -> None:
        hill = GentleHillProfile()
        raw = self.model.qpos0.copy()
        raw[0] = 3.25
        raw[2] += hill.height(raw[:2])
        solver = MotionBricksHillFootIK(self.model, hill.height)
        warmup = solver.apply(raw, 1.0 / 30.0)

        results = [solver.apply(raw, 1.0 / 30.0) for _ in range(6)]
        final = results[-1]

        np.testing.assert_array_equal(warmup.qpos, raw)
        self.assertTrue(all(item.diagnostics.accepted for item in results))
        self.assertEqual(
            final.diagnostics.phases,
            (FootPhase.STANCE, FootPhase.STANCE),
        )
        self.assertGreater(final.diagnostics.raw_penetration_m, 0.03)
        self.assertLess(final.diagnostics.corrected_penetration_m, 0.005)
        self.assertLessEqual(
            final.diagnostics.maximum_target_residual_m,
            0.015,
        )
        self.assertLessEqual(
            final.diagnostics.maximum_joint_correction_rad,
            0.35 + 1.0e-9,
        )

    def test_failed_height_query_rolls_back_state(self) -> None:
        finite = [True]

        def height_query(xy: object) -> float:
            del xy
            return 0.0 if finite[0] else math.nan

        raw = self.model.qpos0.copy()
        solver = MotionBricksHillFootIK(self.model, height_query)
        solver.apply(raw, 1.0 / 30.0)
        before = repr(solver.snapshot_state())
        finite[0] = False

        result = solver.apply(raw, 1.0 / 30.0)

        np.testing.assert_array_equal(result.qpos, raw)
        self.assertFalse(result.diagnostics.accepted)
        self.assertEqual(repr(solver.snapshot_state()), before)


if __name__ == "__main__":
    unittest.main()
