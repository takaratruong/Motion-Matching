from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

from mm_sonic.motionbricks_hill import GentleHillProfile
from mm_sonic.motionbricks_hill_ik import (
    MotionBricksHillFootIK,
    _bounded_leg_correction,
    _bounded_root_height_correction,
    _project_stance_targets,
    _required_swing_lift,
)

MOTIONBRICKS_G1_SCENE = Path(
    "/home/ubuntu/projects/gear-sonic-pinned-60de0df/"
    "motionbricks/assets/skeletons/g1/scene_29dof.xml"
)


class HillFootIKContractTest(unittest.TestCase):
    def test_supported_root_height_ramps_and_clamps(self) -> None:
        value = 0.0
        values = []
        for _ in range(20):
            value = _bounded_root_height_correction(
                value,
                required_support_shift_m=0.15,
                has_support=True,
            )
            values.append(value)
        self.assertAlmostEqual(values[0], 0.025)
        self.assertTrue(
            all(
                abs(right - left) <= 0.025 + 1.0e-12
                for left, right in zip(values, values[1:])
            )
        )
        self.assertLessEqual(max(values), 0.20)

    def test_airborne_root_height_decays_toward_raw(self) -> None:
        self.assertAlmostEqual(
            _bounded_root_height_correction(
                0.08,
                required_support_shift_m=0.0,
                has_support=False,
            ),
            0.07,
        )
        self.assertAlmostEqual(
            _bounded_root_height_correction(
                -0.08,
                required_support_shift_m=0.0,
                has_support=False,
            ),
            -0.07,
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

    def test_joint_limit_can_force_a_safe_correction_reset(self) -> None:
        correction, limit_forced = _bounded_leg_correction(
            np.asarray((-0.14391381,)),
            np.asarray((-0.49391381,)),
            np.asarray((-0.23123252,)),
            np.asarray((-0.2618,)),
            np.asarray((0.2618,)),
            maximum_correction_rad=0.30,
            maximum_step_rad=0.12,
        )

        np.testing.assert_allclose(correction, (-0.11788519,))
        self.assertTrue(limit_forced)

    def test_released_correction_decays_toward_raw(self) -> None:
        correction, limit_forced = _bounded_leg_correction(
            np.asarray((0.0,)),
            np.asarray((0.0,)),
            np.asarray((0.30,)),
            np.asarray((-1.0,)),
            np.asarray((1.0,)),
            maximum_correction_rad=0.30,
            maximum_step_rad=0.12,
        )

        np.testing.assert_allclose(correction, (0.18,))
        self.assertFalse(limit_forced)


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
        solver.apply(raw, (True, True), 1.0 / 30.0)

        result = solver.apply(raw, (True, True), 1.0 / 30.0)

        np.testing.assert_array_equal(result.qpos[:2], raw[:2])
        np.testing.assert_array_equal(result.qpos[3:7], raw[3:7])
        non_leg_except_root_z = solver.non_leg_qpos_addresses
        non_leg_except_root_z = non_leg_except_root_z[
            non_leg_except_root_z != 2
        ]
        np.testing.assert_array_equal(
            result.qpos[non_leg_except_root_z],
            raw[non_leg_except_root_z],
        )
        self.assertTrue(np.isfinite(result.qpos).all())
        self.assertLessEqual(
            result.diagnostics.maximum_joint_correction_rad,
            0.30 + 1.0e-9,
        )

    def test_authored_toe_off_clears_lock_in_same_frame(self) -> None:
        raw = self.model.qpos0.copy()
        solver = MotionBricksHillFootIK(self.model, lambda xy: 0.0)
        planted = solver.apply(raw, (True, False), 1.0 / 30.0)
        self.assertEqual(planted.diagnostics.locked, (True, False))

        shifted = raw.copy()
        shifted[0] += 0.03
        released = solver.apply(
            shifted, (False, False), 1.0 / 30.0
        )

        self.assertEqual(
            released.diagnostics.authored_stance,
            (False, False),
        )
        self.assertEqual(released.diagnostics.locked, (False, False))
        self.assertIsNone(solver.snapshot_state()[1][0])

    def test_display_root_reduces_eighteen_degree_penetration(self) -> None:
        hill = GentleHillProfile()
        raw = self.model.qpos0.copy()
        raw[0] = 3.25
        raw[2] += hill.height(raw[:2])
        solver = MotionBricksHillFootIK(self.model, hill.height)

        results = [
            solver.apply(raw, (True, True), 1.0 / 30.0)
            for _ in range(8)
        ]
        final = results[-1]

        np.testing.assert_array_equal(final.qpos[:2], raw[:2])
        np.testing.assert_array_equal(final.qpos[3:7], raw[3:7])
        self.assertLessEqual(
            abs(final.diagnostics.root_height_correction_m),
            0.20,
        )
        self.assertLess(
            final.diagnostics.corrected_penetration_m, 0.010
        )
        self.assertLessEqual(
            final.diagnostics.maximum_joint_correction_rad,
            0.30 + 1.0e-9,
        )

    def test_authored_swing_never_creates_a_world_lock(self) -> None:
        raw = self.model.qpos0.copy()
        solver = MotionBricksHillFootIK(self.model, lambda xy: 0.0)

        for _ in range(4):
            result = solver.apply(
                raw, (False, False), 1.0 / 30.0
            )

        self.assertEqual(
            result.diagnostics.locked, (False, False)
        )
        self.assertEqual(solver.snapshot_state()[1], (None, None))

    def test_invalid_terrain_cannot_resurrect_released_lock(self) -> None:
        finite = [True]

        def height_query(xy: object) -> float:
            del xy
            return 0.0 if finite[0] else math.nan

        raw = self.model.qpos0.copy()
        solver = MotionBricksHillFootIK(self.model, height_query)
        solver.apply(raw, (True, False), 1.0 / 30.0)
        finite[0] = False

        result = solver.apply(
            raw, (False, False), 1.0 / 30.0
        )

        np.testing.assert_array_equal(result.qpos, raw)
        self.assertFalse(result.diagnostics.accepted)
        self.assertEqual(result.diagnostics.locked, (False, False))
        self.assertIsNone(solver.snapshot_state()[1][0])


if __name__ == "__main__":
    unittest.main()
