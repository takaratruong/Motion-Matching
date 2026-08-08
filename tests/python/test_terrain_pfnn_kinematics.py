from __future__ import annotations

from pathlib import Path
import unittest

import mujoco
import numpy as np
import torch

from mm_sonic.terrain_oracle.canonical import (
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
)
from mm_sonic.terrain_pfnn.kinematics import (
    TorchG1ForwardKinematics,
    root_tilt_quaternion_wxyz,
)


_ASSETS = Path(
    "/home/ubuntu/projects/gear-sonic-pinned-60de0df/"
    "motionbricks/assets/skeletons/g1"
)
_MODEL = _ASSETS / "g1_29dof.xml"
_SCENE = _ASSETS / "scene_29dof.xml"


class TerrainPFNNKinematicsTests(unittest.TestCase):
    def test_root_tilt_is_the_feature_angle_axis_encoding(self) -> None:
        tilt = np.asarray((0.20, -0.15), np.float64)
        angle = np.linalg.norm(tilt)
        expected = np.asarray(
            (
                np.cos(0.5 * angle),
                tilt[0] * np.sin(0.5 * angle) / angle,
                tilt[1] * np.sin(0.5 * angle) / angle,
                0.0,
            )
        )
        np.testing.assert_allclose(
            root_tilt_quaternion_wxyz(*tilt), expected, rtol=0.0, atol=1.0e-15
        )

    def test_eight_in_limit_poses_match_mujoco_and_have_finite_gradients(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(_MODEL))
        data = mujoco.MjData(model)
        fk = TorchG1ForwardKinematics.from_mjmodel(model).double()
        body_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in ISAACLAB_BODY_NAMES
            ],
            dtype=np.int64,
        )
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ISAACLAB_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        qpos_addresses = model.jnt_qposadr[joint_ids]
        limits = model.jnt_range[joint_ids]
        fractions = np.linspace(0.1, 0.9, 8, dtype=np.float64)[:, None]
        joints = limits[None, :, 0] + fractions * (limits[:, 1] - limits[:, 0])
        root = np.column_stack(
            (
                np.linspace(0.55, 1.05, 8),
                np.linspace(-0.20, 0.20, 8),
                np.linspace(0.15, -0.15, 8),
            )
        )
        joint_tensor = torch.tensor(joints, dtype=torch.float64, requires_grad=True)
        root_tensor = torch.tensor(root, dtype=torch.float64, requires_grad=True)
        actual = fk(root_tensor, joint_tensor)
        expected = []
        for pose_root, pose_joints in zip(root, joints):
            data.qpos[:] = model.qpos0
            data.qpos[:3] = (0.0, 0.0, pose_root[0])
            data.qpos[3:7] = root_tilt_quaternion_wxyz(
                pose_root[1], pose_root[2]
            )
            data.qpos[qpos_addresses] = pose_joints
            mujoco.mj_forward(model, data)
            expected.append(np.asarray(data.xpos[body_ids]).copy())
        expected_tensor = torch.tensor(np.stack(expected), dtype=torch.float64)
        max_error = torch.linalg.vector_norm(actual - expected_tensor, dim=-1).max()
        self.assertLess(float(max_error.detach()), 1.0e-5)
        actual.square().mean().backward()
        self.assertIsNotNone(joint_tensor.grad)
        self.assertTrue(torch.isfinite(joint_tensor.grad).all())
        self.assertGreater(float(joint_tensor.grad.abs().max()), 0.0)

    def test_standalone_and_viewer_scene_have_the_same_signature(self) -> None:
        standalone = TorchG1ForwardKinematics.from_mjcf(_MODEL)
        scene = TorchG1ForwardKinematics.from_mjcf(_SCENE)
        self.assertEqual(standalone.body_names, ISAACLAB_BODY_NAMES)
        self.assertEqual(standalone.joint_names, ISAACLAB_JOINT_NAMES)
        self.assertEqual(
            standalone.kinematic_signature_sha256,
            scene.kinematic_signature_sha256,
        )


if __name__ == "__main__":
    unittest.main()
