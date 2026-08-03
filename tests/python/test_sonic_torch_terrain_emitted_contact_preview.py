import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch

from mm_sonic.torch_contact_segments import ANKLE_ORIGIN_SOLE_M
from mm_sonic.torch_terrain_contact_feasibility import (
    TerrainContactFeasibilityConfig,
)
from mm_sonic.torch_terrain_emitted_contact_preview import (
    preview_continued_emitted_contact_trace,
    preview_emitted_contact_trace,
)
from mm_sonic.torch_terrain_skill_composer import (
    TerrainSkillPose,
    advance_skill,
    extend_skill_state,
    start_skill,
)
from mm_sonic.torch_terrain_skills import SkillInterval, TerrainSkill


class _BatchFeet:
    def foot_positions(self, joint, root, quaternion):
        joint = np.asarray(joint)
        feet = np.zeros((joint.shape[0], 2, 3), dtype=np.float64)
        feet[:, 0, 0] = -0.1
        feet[:, 1, 0] = 0.1
        feet[:, :, 2] = ANKLE_ORIGIN_SOLE_M + joint[:, :1]
        return feet


class _BatchSoles:
    def sole_points(self, joint, root, quaternion):
        feet = _BatchFeet().foot_positions(joint, root, quaternion)
        soles = np.repeat(feet[:, :, None, :], 5, axis=2)
        soles[:, 1, 1, 0] += 0.12
        return soles


def _fixture(*, current_joint: float = 0.0, supported_entry: bool = True):
    frames = 4
    root = np.zeros((frames, 1, 3), dtype=np.float32)
    root[:, 0, 2] = 0.8
    quaternion = np.zeros((frames, 1, 4), dtype=np.float32)
    quaternion[:, 0, 0] = 1.0
    clip = SimpleNamespace(
        joint_position=np.zeros((frames, 1), dtype=np.float32),
        joint_velocity=np.zeros((frames, 1), dtype=np.float32),
        body_position_world=root,
        body_quaternion_world_wxyz=quaternion,
        body_linear_velocity_world=np.zeros((frames, 1, 3), dtype=np.float32),
        body_angular_velocity_world=np.zeros((frames, 1, 3), dtype=np.float32),
    )
    folder = SimpleNamespace(
        clips=(clip,), layout=SimpleNamespace(root_body_index=0)
    )
    support = torch.ones((frames, 2), dtype=torch.bool)
    if not supported_entry:
        support[0] = False
    skill = TerrainSkill(
        skill_index=0,
        clip_index=0,
        interval=SkillInterval(0, 0, frames),
        entry_rows=(0,),
        support_mask=support,
        foot_surface_height_m=torch.zeros((frames, 2)),
    )
    pose = TerrainSkillPose(
        joint_position=torch.tensor([current_joint]),
        joint_velocity=torch.zeros(1),
        root_position_world=torch.tensor([0.0, 0.0, 0.8]),
        root_orientation_world_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        root_linear_velocity_world=torch.zeros(3),
        root_angular_velocity_world=torch.zeros(3),
    )
    return folder, skill, pose


def _preview(
    *, current_joint: float = 0.0, supported_entry: bool = True,
    result_filter=None,
):
    folder, skill, pose = _fixture(
        current_joint=current_joint, supported_entry=supported_entry
    )
    return preview_emitted_contact_trace(
        folder=folder,
        skill=skill,
        selected_entry_frame=0,
        endpoint_frame_exclusive=4,
        current=pose,
        halflife_s=0.1,
        foot_kinematics=_BatchFeet(),
        sample_surface=lambda points: torch.zeros(
            points.shape[:-1], dtype=points.dtype, device=points.device
        ),
        config=TerrainContactFeasibilityConfig(),
        result_filter=result_filter,
    )


class EmittedContactPreviewTest(unittest.TestCase):
    def test_accepts_emitted_supported_feet_on_flat_terrain(self):
        result = _preview()

        self.assertTrue(result.accepted)
        self.assertIsNone(result.reason)

    def test_rejects_inertialized_supported_feet_above_terrain(self):
        result = _preview(current_joint=0.2)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "stance-height")

    def test_rejects_entry_without_support(self):
        result = _preview(supported_entry=False)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "unsupported-entry")

    def test_validates_the_filtered_pose_when_a_preview_filter_is_present(self):
        class _ZeroJointPreview:
            def preview(self, results):
                return tuple(
                    SimpleNamespace(
                        **{
                            **vars(result),
                            "joint_position": torch.zeros_like(
                                result.joint_position
                            ),
                        }
                    )
                    for result in results
                )

        result = _preview(
            current_joint=0.2, result_filter=_ZeroJointPreview()
        )

        self.assertTrue(result.accepted)

    def test_previews_the_same_bounded_endpoint_warp_as_committed_playback(self):
        folder, skill, pose = _fixture()
        with mock.patch(
            "mm_sonic.torch_terrain_emitted_contact_preview.start_skill",
            wraps=start_skill,
        ) as start:
            preview_emitted_contact_trace(
                folder=folder,
                skill=skill,
                selected_entry_frame=0,
                endpoint_frame_exclusive=4,
                current=pose,
                halflife_s=0.1,
                foot_kinematics=_BatchFeet(),
                sample_surface=lambda points: torch.zeros(
                    points.shape[:-1],
                    dtype=points.dtype,
                    device=points.device,
                ),
                config=TerrainContactFeasibilityConfig(),
                target_displacement_local_xy=torch.tensor([0.2, 0.0]),
                target_yaw_delta_rad=torch.tensor(0.4),
                maximum_translation_warp_m=0.1,
                maximum_yaw_warp_rad=0.3,
            )

        self.assertEqual(start.call_args.kwargs["maximum_yaw_warp_rad"], 0.3)
        self.assertTrue(
            torch.equal(
                start.call_args.kwargs["target_yaw_delta_rad"],
                torch.tensor(0.4),
            )
        )

    def test_rejects_a_riser_under_the_actual_oriented_toe(self):
        folder, skill, pose = _fixture()
        skill.support_mask[:, 1] = False
        skill.support_mask[2:, 1] = True

        result = preview_emitted_contact_trace(
            folder=folder,
            skill=skill,
            selected_entry_frame=0,
            endpoint_frame_exclusive=4,
            current=pose,
            halflife_s=0.1,
            foot_kinematics=_BatchFeet(),
            sole_kinematics=_BatchSoles(),
            sample_surface=lambda points: torch.where(
                points[..., 0] >= 0.20,
                torch.full_like(points[..., 0], 0.03),
                torch.zeros_like(points[..., 0]),
            ),
            config=TerrainContactFeasibilityConfig(),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "landing-edge-margin")

    def test_continuation_preview_checks_the_extended_emitted_sole(self):
        folder, skill, pose = _fixture()
        state = start_skill(
            folder,
            skill,
            selected_entry_frame=0,
            current=pose,
            halflife_s=0.1,
            playback_stop=2,
        )
        for _ in range(2):
            state = advance_skill(state).state
        state = extend_skill_state(state, playback_stop=4)

        result = preview_continued_emitted_contact_trace(
            folder=folder,
            state=state,
            current=pose,
            foot_kinematics=_BatchFeet(),
            sole_kinematics=_BatchSoles(),
            sample_surface=lambda points: torch.where(
                points[..., 0] >= 0.20,
                torch.full_like(points[..., 0], 0.08),
                torch.zeros_like(points[..., 0]),
            ),
            config=TerrainContactFeasibilityConfig(),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "sole-penetration")


if __name__ == "__main__":
    unittest.main()
