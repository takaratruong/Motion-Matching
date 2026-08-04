import unittest

import numpy as np

from mm_sonic.torch_horizontal_terrain_retarget import (
    WideBoundG1TerrainRetargeter,
    retarget_foot_targets,
)


G1_XML = (
    "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
)


class HorizontalTerrainRetargetTest(unittest.TestCase):
    def test_fixed_root_solver_reaches_small_swing_lift(self):
        retargeter = WideBoundG1TerrainRetargeter(
            G1_XML, maximum_root_height_deviation_m=1.0e-6
        )
        joints = np.zeros(29)
        root = np.array((0.0, 0.0, 0.8))
        quaternion = np.array((1.0, 0.0, 0.0, 0.0))
        feet = retargeter.kinematics.foot_positions(
            joints[None], root[None], quaternion[None]
        )[0]
        targets = feet.copy()
        targets[1, 2] += 0.05

        solved_joints, solved_root = retargeter.solve_frame(
            joint_position=joints,
            root_position_world=root,
            root_orientation_world_wxyz=quaternion,
            solve_feet=np.array((False, True)),
            target_foot_position_world=targets,
        )

        actual = retargeter.kinematics.foot_positions(
            solved_joints[None],
            solved_root[None],
            quaternion[None],
        )[0]
        self.assertLessEqual(
            float(np.linalg.norm(actual[1] - targets[1])), 0.005
        )
        self.assertLessEqual(abs(float(solved_root[2] - root[2])), 1.0e-6)

    def test_support_is_pinned_and_colliding_swing_is_lifted(self):
        feet = np.array(
            [[0.0, 0.0, 0.20], [0.2, 0.0, 0.30]], dtype=np.float64
        )

        mask, targets = retarget_foot_targets(
            foot_position_world=feet,
            support_mask=np.array([True, False]),
            target_surface_at_ankle_m=np.array([0.10, 0.25]),
            minimum_sole_clearance_by_foot_m=np.array([-0.01, -0.06]),
            ankle_origin_sole_m=0.035,
            swing_clearance_margin_m=0.02,
        )

        np.testing.assert_array_equal(mask, [True, True])
        np.testing.assert_allclose(targets[0], [0.0, 0.0, 0.135])
        np.testing.assert_allclose(targets[1], [0.2, 0.0, 0.38])

    def test_clear_swing_foot_is_not_constrained(self):
        mask, _ = retarget_foot_targets(
            foot_position_world=np.array(
                [[0.0, 0.0, 0.20], [0.2, 0.0, 0.30]]
            ),
            support_mask=np.array([True, False]),
            target_surface_at_ankle_m=np.array([0.10, 0.25]),
            minimum_sole_clearance_by_foot_m=np.array([-0.01, 0.03]),
            ankle_origin_sole_m=0.035,
            swing_clearance_margin_m=0.02,
        )

        np.testing.assert_array_equal(mask, [True, False])


if __name__ == "__main__":
    unittest.main()
