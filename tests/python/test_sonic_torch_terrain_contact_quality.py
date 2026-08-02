import unittest

import numpy as np
import torch

from mm_sonic.torch_contact_oracle_actions import ContactPhaseAction
from mm_sonic.torch_terrain_contact_quality import build_contact_quality_ablation
from mm_sonic.torch_terrain_quality_states import capture_quality_states


class _LinearFeet:
    @staticmethod
    def foot_positions(joints, roots, _quaternions):
        joints = np.asarray(joints)
        roots = np.asarray(roots)
        return roots[:, None, :] + joints[:, :6].reshape(-1, 2, 3)

    @staticmethod
    def solve_leg_positions(joints, root, _quaternion, mask, targets):
        output = np.array(joints, copy=True)
        for foot in range(2):
            if mask[foot]:
                output[foot * 3 : foot * 3 + 3] = targets[foot] - root
        return output


def _action():
    joints = torch.zeros((4, 29), dtype=torch.float32)
    joints[:, :3] = torch.tensor((0.0, 0.1, 0.0))
    joints[:, 3:6] = torch.tensor(
        (
            (0.0, -0.1, 0.2),
            (0.1, -0.1, 0.3),
            (0.2, -0.1, 0.1),
            (0.3, -0.1, 0.0),
        )
    )
    orientation = torch.zeros((4, 4), dtype=torch.float32)
    orientation[:, 0] = 1.0
    return ContactPhaseAction(
        clip_index=0,
        start_frame=0,
        end_frame=4,
        swing_foot=1,
        entry_support=torch.tensor((True, False)),
        exit_support=torch.tensor((True, True)),
        support_mask=torch.tensor(
            ((True, False), (True, False), (True, False), (True, True))
        ),
        joint_position=joints,
        joint_velocity=torch.zeros((4, 29), dtype=torch.float32),
        root_position_local=torch.zeros((4, 3), dtype=torch.float32),
        root_yaw_local=torch.zeros(4, dtype=torch.float32),
        root_orientation_local_wxyz=orientation,
        foot_position_local=joints[:, :6].reshape(4, 2, 3),
        foot_surface_delta_m=torch.zeros((4, 2), dtype=torch.float32),
        minimum_swing_clearance_m=0.05,
    )


def _state():
    joints = np.array(_action().joint_position[0], dtype=np.float64)
    joints[0] += 0.2
    joint_velocity = np.zeros((1, 29), dtype=np.float64)
    joint_velocity[0, 0] = 1.0
    root = np.zeros((1, 3), dtype=np.float64)
    quaternion = np.array(((1.0, 0.0, 0.0, 0.0),))
    feet = _LinearFeet.foot_positions(joints[None], root, quaternion)
    arrays = {
        "qpos": np.concatenate((root, quaternion, joints[None]), axis=1),
        "joint_position": joints[None],
        "joint_velocity": joint_velocity,
        "root_position_world": root,
        "root_yaw_world": np.zeros(1),
        "foot_position_world": feet,
        "foot_surface_height_m": np.zeros((1, 2)),
        "command_velocity_world_xy": np.zeros((1, 2)),
        "command_heading_world_yaw": np.zeros(1),
        "command_segment_index": np.zeros(1, dtype=np.int64),
        "selected_clip_path": np.array(("clip.npz",)),
        "selected_source_frame": np.zeros(1, dtype=np.int64),
    }
    return capture_quality_states(
        route_name="contact-quality",
        arrays=arrays,
        source_support_mask=np.array(((True, False),)),
        terrain_patch_sampler=lambda points: np.zeros(points.shape[0]),
    )[0]


class TerrainContactQualityTests(unittest.TestCase):
    def test_projection_restores_stance_and_desired_landing(self):
        result = build_contact_quality_ablation(
            state=_state(),
            action=_action(),
            desired_landing_foot=1,
            desired_landing_world_xyz=np.array((0.45, -0.08, 0.04)),
            foot_kinematics=_LinearFeet(),
        )

        self.assertGreater(result.unprojected_stance_drift_m, 0.0)
        self.assertAlmostEqual(result.projected_stance_drift_m, 0.0, places=6)
        self.assertAlmostEqual(result.projected_landing_error_m, 0.0, places=6)
        self.assertAlmostEqual(result.maximum_target_error_m, 0.0, places=6)
        self.assertGreater(result.maximum_joint_deformation_rad, 0.0)
        self.assertEqual(result.projected_qpos.shape, (4, 36))
        self.assertFalse(result.projected_qpos.flags.writeable)

    def test_rejects_action_with_wrong_planned_swing_foot(self):
        with self.assertRaisesRegex(ValueError, "landing foot"):
            build_contact_quality_ablation(
                state=_state(),
                action=_action(),
                desired_landing_foot=0,
                desired_landing_world_xyz=np.zeros(3),
                foot_kinematics=_LinearFeet(),
            )

    def test_stance_root_strategy_moves_support_error_into_root(self):
        kwargs = dict(
            state=_state(),
            action=_action(),
            desired_landing_foot=1,
            desired_landing_world_xyz=np.array((0.45, -0.08, 0.04)),
            foot_kinematics=_LinearFeet(),
        )
        stance_root = build_contact_quality_ablation(
            **kwargs, projection_strategy="stance-root"
        )

        self.assertAlmostEqual(stance_root.projected_stance_drift_m, 0.0, places=6)
        self.assertGreater(stance_root.maximum_root_correction_m, 0.0)
        self.assertAlmostEqual(stance_root.projected_landing_error_m, 0.0, places=6)

    def test_root_only_strategy_does_not_edit_joint_motion(self):
        result = build_contact_quality_ablation(
            state=_state(),
            action=_action(),
            desired_landing_foot=1,
            desired_landing_world_xyz=np.array((0.45, -0.08, 0.04)),
            foot_kinematics=_LinearFeet(),
            projection_strategy="root-only",
        )

        self.assertEqual(result.maximum_joint_deformation_rad, 0.0)
        self.assertEqual(result.maximum_joint_correction_speed_rad_s, 0.0)


if __name__ == "__main__":
    unittest.main()
