import unittest

import numpy as np

from mm_sonic.joints import PINNED_TARGET_TO_SOURCE_PERMUTATION
from mm_sonic.torch_g1_fk import MujocoG1FootKinematics
from mm_sonic.torch_horizontal_terrain_retarget import (
    WideBoundG1TerrainRetargeter,
)
from mm_sonic.torch_motionbricks_drop_fallback import (
    pad_motionbricks_exact_endpoints,
    terrain_clearance_envelope,
    realize_motionbricks_drop,
    solve_motionbricks_landing_qpos,
)


G1_XML = (
    "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
)
TEMPLATE = (
    "build/g1-traversal-library/horizontal-full/"
    "motionbricks-contact-exit-v4/generated-qpos.npy"
)


class MotionBricksDropFallbackTests(unittest.TestCase):
    def test_endpoint_padding_preserves_exact_poses(self):
        qpos = np.arange(5 * 36, dtype=np.float64).reshape(5, 36)

        padded = pad_motionbricks_exact_endpoints(
            qpos, hold_frames=3
        )

        self.assertEqual(padded.shape, (11, 36))
        np.testing.assert_array_equal(
            padded[:4], np.repeat(qpos[:1], 4, axis=0)
        )
        np.testing.assert_array_equal(
            padded[-4:], np.repeat(qpos[-1:], 4, axis=0)
        )
        np.testing.assert_array_equal(padded[4:-4], qpos[1:-1])

    def test_clearance_envelope_spreads_a_collision_with_bounded_slope(self):
        clearance = np.zeros((14, 2), dtype=np.float64)
        clearance[6, 0] = -0.07

        envelope = terrain_clearance_envelope(
            minimum_sole_clearance_by_frame_foot=clearance,
            accepted_clearance_m=-0.02,
            maximum_correction_step_m=0.01,
        )

        self.assertAlmostEqual(float(envelope[6, 0]), 0.05)
        self.assertTrue(
            (
                np.abs(np.diff(envelope[:, 0]))
                <= 0.01 + 1.0e-12
            ).all()
        )
        np.testing.assert_allclose(envelope[:, 1], 0.0)
        np.testing.assert_allclose(envelope[[0, -1]], 0.0)

    def test_clearance_envelope_can_lift_a_generated_stop_window(self):
        clearance = np.zeros((8, 2), dtype=np.float64)
        clearance[-1, 0] = -0.05

        envelope = terrain_clearance_envelope(
            minimum_sole_clearance_by_frame_foot=clearance,
            accepted_clearance_m=-0.02,
            maximum_correction_step_m=0.01,
            preserve_stop_endpoint=False,
        )

        np.testing.assert_allclose(envelope[0], 0.0)
        self.assertAlmostEqual(float(envelope[-1, 0]), 0.03)
        self.assertTrue(
            (
                np.abs(np.diff(envelope[:, 0]))
                <= 0.01 + 1.0e-12
            ).all()
        )

    def test_clearance_envelope_preserves_full_context_window(self):
        clearance = np.zeros((10, 2), dtype=np.float64)
        clearance[6, 0] = -0.05

        envelope = terrain_clearance_envelope(
            minimum_sole_clearance_by_frame_foot=clearance,
            accepted_clearance_m=-0.02,
            maximum_correction_step_m=0.01,
            preserve_start_frames=4,
            preserve_stop_endpoint=False,
        )

        np.testing.assert_allclose(envelope[:4], 0.0)
        self.assertAlmostEqual(float(envelope[6, 0]), 0.03)

    def test_landing_solver_retargets_released_pose_to_exact_footprints(self):
        template = np.load(TEMPLATE, allow_pickle=False)
        kinematics = MujocoG1FootKinematics(G1_XML)
        targets = np.array(
            ((0.8, -0.08, 0.035), (0.8, -0.32, 0.035)),
            dtype=np.float64,
        )

        qpos = solve_motionbricks_landing_qpos(
            reference_qpos=template[-1],
            target_foot_position_world=targets,
            target_heading_world_xy=np.array((1.0, 0.0)),
            kinematics=kinematics,
            retargeter=WideBoundG1TerrainRetargeter(
                G1_XML,
                maximum_root_horizontal_deviation_m=0.20,
                maximum_root_height_deviation_m=0.30,
                maximum_target_error_m=0.040,
            ),
        )

        self.assertEqual(qpos.shape, (36,))
        joints = qpos[7:][
            np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION)
        ]
        actual = kinematics.foot_positions(
            joints[None], qpos[None, :3], qpos[None, 3:7]
        )[0]
        self.assertLess(
            float(np.linalg.norm(actual - targets, axis=1).max()), 0.015
        )

    def test_drop_preserves_start_and_reaches_aligned_landing_feet(self):
        template = np.load(TEMPLATE, allow_pickle=False)
        kinematics = MujocoG1FootKinematics(G1_XML)
        start_native = template[3]
        target_center = np.array((0.8, -0.2), dtype=np.float64)
        target_feet = np.array(
            (
                (target_center[0], target_center[1] + 0.12, 0.035),
                (target_center[0], target_center[1] - 0.12, 0.035),
            )
        )

        result = realize_motionbricks_drop(
            template_qpos=template,
            start_qpos=start_native,
            target_foot_position_world=target_feet,
            target_heading_world_xy=np.array((1.0, 0.0)),
            kinematics=kinematics,
            retargeter=WideBoundG1TerrainRetargeter(
                G1_XML,
                maximum_root_horizontal_deviation_m=0.20,
                maximum_root_height_deviation_m=0.30,
                maximum_target_error_m=0.040,
            ),
            sample_surface=lambda points: np.zeros(
                np.asarray(points).shape[:-1], dtype=np.float64
            ),
        )

        np.testing.assert_allclose(result.qpos[0], start_native)
        actual = kinematics.foot_positions(
            result.joint_position[-1:],
            result.root_position_world[-1:],
            result.root_orientation_world_wxyz[-1:],
        )[0]
        self.assertLess(
            float(np.linalg.norm(actual - target_feet, axis=1).max()),
            0.015,
        )
        self.assertTrue(result.support_mask[-1].all())
        self.assertTrue((~result.support_mask[1:-1].any(axis=1)).all())


if __name__ == "__main__":
    unittest.main()
