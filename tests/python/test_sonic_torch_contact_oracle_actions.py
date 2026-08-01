import unittest
from types import SimpleNamespace

import numpy as np
import torch

from mm_sonic.torch_contact_oracle_actions import (
    ContactPhaseActionIndex,
    action_from_profiles,
    mirror_contact_phase_action,
    mirror_g1_joint_state,
    with_mirrored_actions,
)
from mm_sonic.torch_contact_segments import ContactSegment


class _IdentityAlignment:
    @staticmethod
    def matcher_to_scene_xy(points):
        return points


class _FlatGrid:
    @staticmethod
    def sample_xy(points):
        return torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)


class _FootKinematics:
    def __init__(self, feet):
        self.feet = feet

    def foot_positions(self, _joints, _roots, _quaternions):
        return self.feet.copy()


class _BrokenFootKinematics:
    @staticmethod
    def foot_positions(_joints, _roots, _quaternions):
        raise RuntimeError("broken FK")


class _SegmentIndex:
    def __init__(self, support, segments):
        self._support = support
        self.segments = segments

    def support_mask(self, clip_index):
        if clip_index != 0:
            raise AssertionError("unexpected clip")
        return self._support.clone()


def _index_fixture():
    frames = 9
    support = torch.tensor(
        [
            [True, False], [True, False], [True, True],
            [False, True], [False, True], [True, True],
            [True, False], [True, False], [True, True],
        ],
        dtype=torch.bool,
    )
    joint_position = np.zeros((frames, 29), np.float32)
    body_position = np.zeros((frames, 3, 3), np.float32)
    body_position[:, 0, 0] = np.arange(frames, dtype=np.float32) * 0.1
    body_position[:, 1, 1] = 0.1
    body_position[:, 2, 1] = -0.1
    body_quaternion = np.zeros((frames, 3, 4), np.float32)
    body_quaternion[..., 0] = 1.0
    clip = SimpleNamespace(
        joint_position=joint_position,
        joint_velocity=np.zeros_like(joint_position),
        body_position_world=body_position,
        body_quaternion_world_wxyz=body_quaternion,
    )
    dataset = SimpleNamespace(
        folder=SimpleNamespace(
            clips=(clip,),
            layout=SimpleNamespace(
                root_body_index=0,
                left_foot_body_index=1,
                right_foot_body_index=2,
            ),
        ),
        clip_grids=(_FlatGrid(),),
        clip_alignments=(_IdentityAlignment(),),
        device=torch.device("cpu"),
    )
    return dataset, support, body_position[:, 1:3]


class ContactOracleActionTests(unittest.TestCase):
    def test_g1_mirror_is_an_involution_with_axial_joint_signs(self):
        joints = torch.arange(29, dtype=torch.float32)
        mirrored = mirror_g1_joint_state(joints)

        self.assertEqual(float(mirrored[0]), float(joints[1]))
        self.assertEqual(float(mirrored[3]), -float(joints[4]))
        self.assertEqual(float(mirrored[2]), -float(joints[2]))
        self.assertEqual(float(mirrored[8]), float(joints[8]))
        torch.testing.assert_close(
            mirror_g1_joint_state(mirrored), joints
        )

    def test_extracts_one_landing_phase_with_inclusive_terminal_contact(self):
        frames = 6
        support = torch.tensor(
            [
                [True, False],
                [True, False],
                [True, True],
                [False, True],
                [False, True],
                [True, True],
            ],
            dtype=torch.bool,
        )
        joint_position = torch.zeros((frames, 29), dtype=torch.float32)
        joint_velocity = torch.zeros_like(joint_position)
        root_position = torch.zeros((frames, 3), dtype=torch.float32)
        root_position[:, 0] = torch.arange(frames, dtype=torch.float32) * 0.1
        root_quaternion = torch.zeros((frames, 4), dtype=torch.float32)
        root_quaternion[:, 0] = 1.0
        foot_position = torch.zeros((frames, 2, 3), dtype=torch.float32)
        foot_position[:, 0, 1] = 0.1
        foot_position[:, 1, 1] = -0.1
        foot_surface = torch.zeros((frames, 2), dtype=torch.float32)

        action = action_from_profiles(
            clip_index=3,
            start_frame=2,
            landing_frame=5,
            support_mask=support,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            root_position_world=root_position,
            root_orientation_world_wxyz=root_quaternion,
            foot_position_world=foot_position,
            foot_surface_height_m=foot_surface,
        )

        self.assertEqual((action.start_frame, action.end_frame), (2, 6))
        self.assertEqual(action.swing_foot, 0)
        self.assertEqual(tuple(action.exit_support.tolist()), (True, True))
        self.assertEqual(action.frame_count, 4)
        torch.testing.assert_close(
            action.root_position_local[-1],
            torch.tensor((0.3, 0.0, 0.0), dtype=torch.float32),
        )
        self.assertAlmostEqual(action.minimum_swing_clearance_m, -0.035)

    def test_dataset_index_reports_inventory_and_exact_source_successor(self):
        dataset, support, feet = _index_fixture()
        segments = (
            ContactSegment(0, 2, 5, 1),
            ContactSegment(0, 5, 8, 0),
            ContactSegment(0, 8, 9, 1),
        )

        index = ContactPhaseActionIndex.from_dataset(
            dataset,
            _SegmentIndex(support, segments),
            _FootKinematics(feet),
        )

        self.assertEqual(index.inventory.retained_count, 2)
        self.assertEqual(
            dict(index.inventory.rejected_by_reason),
            {"no-next-opposite-landing": 1},
        )
        self.assertEqual(index.action(0).source_key, (0, 2, 6, False))
        self.assertEqual(
            index.exact_successor(0).source_key, (0, 5, 9, False)
        )
        self.assertIsNone(index.exact_successor(1))

        augmented = with_mirrored_actions(index)
        self.assertEqual(augmented.inventory.retained_count, 4)
        self.assertEqual(
            [action.mirrored for action in augmented.actions],
            [False, True, False, True],
        )
        self.assertEqual(
            augmented.exact_successor(0).source_key,
            (0, 5, 9, False),
        )
        self.assertEqual(
            augmented.exact_successor(1).source_key,
            (0, 5, 9, True),
        )
        restored = mirror_contact_phase_action(augmented.action(1))
        self.assertFalse(restored.mirrored)
        torch.testing.assert_close(
            restored.joint_position, augmented.action(0).joint_position
        )
        torch.testing.assert_close(
            restored.foot_position_local,
            augmented.action(0).foot_position_local,
        )

    def test_dataset_index_reports_distinct_rejection_reasons(self):
        dataset, support, feet = _index_fixture()
        valid = (ContactSegment(0, 2, 5, 1),)
        invalid_fk = ContactPhaseActionIndex.from_dataset(
            dataset, _SegmentIndex(support, valid), _BrokenFootKinematics()
        )
        self.assertEqual(
            dict(invalid_fk.inventory.rejected_by_reason), {"invalid-fk": 1}
        )

        unstable_support = support.clone()
        unstable_support[5] = unstable_support[4]
        unstable = ContactPhaseActionIndex.from_dataset(
            dataset,
            _SegmentIndex(unstable_support, valid),
            _FootKinematics(feet),
        )
        self.assertEqual(
            dict(unstable.inventory.rejected_by_reason),
            {"unstable-terminal-support": 1},
        )

        inconsistent = ContactPhaseActionIndex.from_dataset(
            dataset,
            _SegmentIndex(support, (ContactSegment(0, 2, 5, 0),)),
            _FootKinematics(feet),
        )
        self.assertEqual(
            dict(inconsistent.inventory.rejected_by_reason),
            {"inconsistent-contact-order": 1},
        )


if __name__ == "__main__":
    unittest.main()
