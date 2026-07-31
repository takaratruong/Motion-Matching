from types import SimpleNamespace
import unittest

import numpy as np
import torch

from mm_sonic.joints import ContractError

try:
    from mm_sonic.torch_terrain_landing_bridge import (
        generate_terrain_landing_bridge,
        LandingBridgeReference,
        TerrainLandingBridge,
    )
except ImportError:
    generate_terrain_landing_bridge = None
    LandingBridgeReference = TerrainLandingBridge = None


class _IdentityAlignment:
    def matcher_to_scene_xy(self, points):
        return points


class _StepGrid:
    def __init__(self, *, flat=False, fail=False):
        self.flat = flat
        self.fail = fail

    def sample_xy(self, points):
        if self.fail:
            raise ContractError("outside")
        if self.flat:
            return torch.full(points.shape[:-1], 0.15)
        return torch.where(
            points[..., 1] < 0.20,
            torch.full(points.shape[:-1], 0.15),
            torch.zeros(points.shape[:-1]),
        )


class _EncodedFootKinematics:
    def __init__(self, *, fail=False, jump=False):
        self.fail = fail
        self.jump = jump
        self.calls = 0

    def foot_positions(self, joints, _roots, _quaternions):
        values = np.asarray(joints, np.float64)
        return values[:, :6].reshape(-1, 2, 3)

    def solve_leg_positions(
        self, joints, _root, _quaternion, mask, targets
    ):
        if self.fail:
            raise ContractError("scripted IK failure")
        output = np.asarray(joints, np.float64).copy()
        encoded = np.asarray(targets, np.float64).reshape(6)
        selected = np.repeat(np.asarray(mask, bool), 3)
        output[:6][selected] = encoded[selected]
        if self.jump and self.calls == 1:
            output[0] += 1.0
        self.calls += 1
        return output


def _bridge_fixture(*, flat=False, fail_grid=False):
    frames = 6
    body = np.zeros((frames, 3, 3), np.float32)
    body[:, 0, 1] = np.linspace(0.0, 0.25, frames)
    body[:, 0, 2] = np.linspace(0.8, 0.65, frames)
    quaternion = np.zeros((frames, 3, 4), np.float32)
    quaternion[..., 0] = 1.0
    clip = SimpleNamespace(
        joint_position=np.zeros((frames, 29), np.float32),
        body_position_world=body,
        body_quaternion_world_wxyz=quaternion,
    )
    dataset = SimpleNamespace(
        folder=SimpleNamespace(
            clips=(clip,),
            layout=SimpleNamespace(root_body_index=0),
        ),
        device=torch.device("cpu"),
    )
    extension = SimpleNamespace(
        alignment=_IdentityAlignment(),
        query_grid=_StepGrid(flat=flat, fail=fail_grid),
    )
    support = torch.tensor(
        [
            [True, True],
            [True, True],
            [False, True],
            [False, True],
            [True, False],
            [True, True],
        ]
    )
    index = SimpleNamespace(support_mask=lambda _clip_index: support.clone())
    start_joints = np.zeros(29, np.float64)
    start_joints[:6] = [0.0, 0.08, 0.185, 0.0, -0.08, 0.185]
    return dataset, extension, index, start_joints


class TerrainLandingBridgeTests(unittest.TestCase):
    def test_bridge_owns_immutable_exact_shape_arrays(self):
        frames = 6
        bridge = TerrainLandingBridge(
            joint_position=np.zeros((frames, 29)),
            joint_velocity=np.zeros((frames, 29)),
            root_position_world=np.zeros((frames, 3)),
            root_orientation_world_wxyz=np.tile(
                [1.0, 0.0, 0.0, 0.0], (frames, 1)
            ),
            foot_position_world=np.zeros((frames, 2, 3)),
        )

        self.assertEqual(bridge.frame_count, frames)
        for value in (
            bridge.joint_position,
            bridge.joint_velocity,
            bridge.root_position_world,
            bridge.root_orientation_world_wxyz,
            bridge.foot_position_world,
        ):
            self.assertEqual(value.dtype, np.float64)
            self.assertFalse(value.flags.writeable)
        with self.assertRaises(ValueError):
            bridge.joint_position[0, 0] = 1.0

    def test_reference_exposes_exact_authenticated_support_slice(self):
        reference = LandingBridgeReference(14, 361, 421)
        support = torch.zeros((500, 2), dtype=torch.bool)
        support[361:421, 1] = True
        index = SimpleNamespace(support_mask=lambda clip_index: support.clone())

        actual = reference.source_support_mask(index)

        self.assertEqual(reference.frame_count, 60)
        self.assertEqual(tuple(actual.shape), (60, 2))
        self.assertTrue(bool(actual[:, 1].all()))

    def test_bridge_and_reference_reject_malformed_contracts(self):
        with self.assertRaisesRegex(ContractError, "reference"):
            LandingBridgeReference(14, 421, 361)
        with self.assertRaisesRegex(ContractError, "bridge"):
            TerrainLandingBridge(
                joint_position=np.zeros((6, 28)),
                joint_velocity=np.zeros((6, 29)),
                root_position_world=np.zeros((6, 3)),
                root_orientation_world_wxyz=np.tile(
                    [1.0, 0.0, 0.0, 0.0], (6, 1)
                ),
                foot_position_world=np.zeros((6, 2, 3)),
            )

    def test_generator_lands_both_feet_on_lower_surface(self):
        dataset, extension, index, joints = _bridge_fixture()

        bridge = generate_terrain_landing_bridge(
            dataset=dataset,
            extension=extension,
            contact_index=index,
            kinematics=_EncodedFootKinematics(),
            reference=LandingBridgeReference(0, 0, 6),
            start_joint_position=joints,
            start_root_position_world=np.array([0.0, 0.0, 0.8]),
            start_root_orientation_world_wxyz=np.array(
                [1.0, 0.0, 0.0, 0.0]
            ),
            command_direction_world_xy=np.array([0.0, 1.0]),
        )

        self.assertEqual(bridge.frame_count, 6)
        self.assertTrue(np.all(bridge.foot_position_world[-1, :, 1] >= 0.20))
        np.testing.assert_allclose(
            bridge.foot_position_world[-1, :, 2], 0.035, atol=1e-8
        )

    def test_generator_fails_closed_for_terrain_ik_and_discontinuity(self):
        cases = (
            (dict(flat=True), _EncodedFootKinematics(), "lower foothold"),
            (dict(fail_grid=True), _EncodedFootKinematics(), "terrain query"),
            (dict(), _EncodedFootKinematics(fail=True), "bridge IK"),
            (dict(), _EncodedFootKinematics(jump=True), "discontinuity"),
        )
        for fixture_options, kinematics, message in cases:
            with self.subTest(message=message):
                dataset, extension, index, joints = _bridge_fixture(
                    **fixture_options
                )
                with self.assertRaisesRegex(ContractError, message):
                    generate_terrain_landing_bridge(
                        dataset=dataset,
                        extension=extension,
                        contact_index=index,
                        kinematics=kinematics,
                        reference=LandingBridgeReference(0, 0, 6),
                        start_joint_position=joints,
                        start_root_position_world=np.array([0.0, 0.0, 0.8]),
                        start_root_orientation_world_wxyz=np.array(
                            [1.0, 0.0, 0.0, 0.0]
                        ),
                        command_direction_world_xy=np.array([0.0, 1.0]),
                    )


if __name__ == "__main__":
    unittest.main()
