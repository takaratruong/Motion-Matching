import unittest

import numpy as np

from mm_sonic.torch_g1_fk import target_state_qpos
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics


G1_XML = "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"


class G1SoleKinematicsTests(unittest.TestCase):
    def test_batched_sole_points_follow_actual_ankle_body_rotation(self):
        import mujoco

        joints = np.zeros((2, 29), np.float64)
        joints[1] = np.linspace(-0.08, 0.08, 29)
        roots = np.array(((0.0, 0.0, 0.8), (0.2, -0.1, 0.9)))
        yaw = 0.35
        quaternions = np.array(
            ((1.0, 0.0, 0.0, 0.0),
             (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)))
        )

        actual = MujocoG1SoleKinematics(G1_XML).sole_points(
            joints, roots, quaternions
        )

        model = mujoco.MjModel.from_xml_path(G1_XML)
        data = mujoco.MjData(model)
        body_ids = tuple(
            int(model.body(name).id)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        )
        local = np.array(
            ((0.035, 0.0, -0.05), (0.12, 0.0, -0.05),
             (-0.05, 0.0, -0.05), (0.12, 0.04, -0.05),
             (0.12, -0.04, -0.05), (-0.05, 0.04, -0.05),
             (-0.05, -0.04, -0.05)),
            np.float64,
        )
        expected = []
        for joint, root, quaternion in zip(joints, roots, quaternions):
            data.qpos[:] = target_state_qpos(joint, root, quaternion)
            mujoco.mj_forward(model, data)
            expected.append(
                np.stack(
                    [
                        data.xpos[body] + local @ data.xmat[body].reshape(3, 3).T
                        for body in body_ids
                    ]
                )
            )

        self.assertEqual(actual.shape, (2, 2, 7, 3))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
