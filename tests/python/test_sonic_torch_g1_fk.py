from types import SimpleNamespace
import unittest

import numpy as np

from mm_sonic.joints import (
    ContractError,
    PINNED_TARGET_TO_SOURCE_PERMUTATION,
)

try:
    from mm_sonic.torch_g1_fk import (
        MujocoG1FootKinematics,
        target_state_qpos,
    )
except ImportError:
    def target_state_qpos(*_args, **_kwargs):
        raise AssertionError("shared G1 qpos conversion is missing")

    class MujocoG1FootKinematics:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("shared G1 foot kinematics is missing")


G1_XML = (
    "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
)


class G1FootKinematicsTests(unittest.TestCase):
    def test_target_state_qpos_uses_pinned_joint_permutation(self):
        target = np.arange(29, dtype=np.float64) * 0.01
        qpos = target_state_qpos(
            target,
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        expected_source = np.empty(29, np.float64)
        expected_source[
            np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION)
        ] = target

        self.assertEqual(qpos.shape, (36,))
        np.testing.assert_array_equal(qpos[:7], [1, 2, 3, 1, 0, 0, 0])
        np.testing.assert_array_equal(qpos[7:], expected_source)

    def test_batched_feet_match_direct_mujoco_forward_kinematics(self):
        import mujoco

        joints = np.zeros((2, 29), np.float64)
        joints[1] = np.linspace(-0.1, 0.1, 29)
        roots = np.array([[0.0, 0.0, 0.8], [0.2, -0.1, 0.9]])
        yaw = 0.3
        quaternions = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
            ]
        )
        helper = MujocoG1FootKinematics(G1_XML)
        actual = helper.foot_positions(joints, roots, quaternions)

        model = mujoco.MjModel.from_xml_path(G1_XML)
        data = mujoco.MjData(model)
        foot_ids = np.array(
            [
                model.body("left_ankle_roll_link").id,
                model.body("right_ankle_roll_link").id,
            ]
        )
        expected = []
        for joint, root, quaternion in zip(joints, roots, quaternions):
            data.qpos[:] = target_state_qpos(joint, root, quaternion)
            mujoco.mj_forward(model, data)
            expected.append(data.xpos[foot_ids].copy())

        self.assertEqual(actual.shape, (2, 2, 3))
        np.testing.assert_allclose(
            actual, np.asarray(expected), rtol=0.0, atol=1e-10
        )

    def test_left_foot_ik_reaches_target_and_preserves_other_joints(self):
        helper = MujocoG1FootKinematics(G1_XML)
        joints = np.zeros(29, np.float64)
        root = np.array([0.0, 0.0, 0.8], np.float64)
        quaternion = np.array([1.0, 0.0, 0.0, 0.0], np.float64)
        feet = helper.foot_positions(
            joints[None], root[None], quaternion[None]
        )[0]
        target = feet.copy()
        target[0, 0] += 0.01

        solved = helper.solve_leg_positions(
            joints,
            root,
            quaternion,
            np.array([True, False]),
            target,
        )

        actual = helper.foot_positions(
            solved[None], root[None], quaternion[None]
        )[0]
        self.assertLessEqual(
            float(np.linalg.norm(actual[0] - target[0])), 0.005
        )
        left_leg = {0, 3, 6, 9, 13, 17}
        preserved = [index for index in range(29) if index not in left_leg]
        np.testing.assert_array_equal(solved[preserved], joints[preserved])

    def test_foot_ik_rejects_unreachable_and_malformed_targets(self):
        helper = MujocoG1FootKinematics(G1_XML)
        joints = np.zeros(29, np.float64)
        root = np.array([0.0, 0.0, 0.8], np.float64)
        quaternion = np.array([1.0, 0.0, 0.0, 0.0], np.float64)
        feet = helper.foot_positions(
            joints[None], root[None], quaternion[None]
        )[0]
        unreachable = feet.copy()
        unreachable[0, 2] -= 2.0
        cases = (
            (
                np.array([True, False]),
                unreachable,
                quaternion,
                "unreachable",
            ),
            (
                np.array([False, False]),
                feet,
                quaternion,
                "nonempty",
            ),
            (
                np.array([True, False]),
                np.zeros((1, 3)),
                quaternion,
                "shape",
            ),
            (
                np.array([True, False]),
                feet + np.nan,
                quaternion,
                "finite",
            ),
            (
                np.array([True, False]),
                feet,
                np.zeros(4),
                "unit",
            ),
        )
        for mask, target, orientation, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ContractError, message):
                    helper.solve_leg_positions(
                        joints, root, orientation, mask, target
                    )

    def test_rejects_malformed_or_nonunit_emitted_state(self):
        valid_joint = np.zeros(29)
        valid_root = np.zeros(3)
        valid_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        cases = (
            (np.zeros(28), valid_root, valid_quaternion, "joint"),
            (valid_joint, np.zeros(2), valid_quaternion, "root"),
            (valid_joint, valid_root, np.zeros(4), "unit"),
            (
                valid_joint,
                valid_root,
                np.array([1.0, 0.0, 0.0, np.nan]),
                "finite",
            ),
        )
        for joint, root, quaternion, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ContractError, message):
                    target_state_qpos(joint, root, quaternion)

    def test_batch_requires_matching_finite_shapes(self):
        helper = MujocoG1FootKinematics(G1_XML)
        with self.assertRaisesRegex(ContractError, "batch"):
            helper.foot_positions(
                np.zeros((2, 29)),
                np.zeros((1, 3)),
                np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
            )

    def test_constructor_rejects_non_g1_model(self):
        fake = SimpleNamespace(nq=7)
        with self.assertRaisesRegex(ContractError, "nq"):
            MujocoG1FootKinematics._from_compiled_model_for_test(fake)


if __name__ == "__main__":
    unittest.main()
