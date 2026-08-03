import unittest
from types import SimpleNamespace

import numpy as np
import torch

from mm_sonic.torch_terrain_skill_composer import (
    TerrainSkillPose,
    advance_skill,
    can_interrupt_skill,
    extend_skill_state,
    start_skill,
)
from mm_sonic.torch_terrain_skills import SkillInterval, TerrainSkill


def _fixture():
    frames = 16
    joints = 2
    joint_position = np.stack(
        [np.linspace(0.0, 0.3, frames), np.linspace(0.2, -0.1, frames)], axis=1
    ).astype(np.float32)
    joint_velocity = np.gradient(joint_position, 0.02, axis=0).astype(np.float32)
    root_position = np.zeros((frames, 1, 3), dtype=np.float32)
    root_position[:, 0, 0] = np.arange(frames) * 0.01
    root_position[:, 0, 2] = 0.8
    root_quaternion = np.zeros((frames, 1, 4), dtype=np.float32)
    root_quaternion[:, 0, 0] = 1.0
    root_linear_velocity = np.zeros((frames, 1, 3), dtype=np.float32)
    root_linear_velocity[:, 0, 0] = 0.5
    root_angular_velocity = np.zeros((frames, 1, 3), dtype=np.float32)
    clip = SimpleNamespace(
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        body_position_world=root_position,
        body_quaternion_world_wxyz=root_quaternion,
        body_linear_velocity_world=root_linear_velocity,
        body_angular_velocity_world=root_angular_velocity,
    )
    folder = SimpleNamespace(clips=(clip,), layout=SimpleNamespace(root_body_index=0))
    support = torch.ones((frames, 2), dtype=torch.bool)
    support[7, 1] = False
    skill = TerrainSkill(
        skill_index=0,
        clip_index=0,
        interval=SkillInterval(2, 5, 12),
        entry_rows=(2, 3, 4),
        support_mask=support,
        foot_surface_height_m=torch.zeros((frames, 2)),
    )
    pose = TerrainSkillPose(
        joint_position=torch.tensor([0.7, -0.4]),
        joint_velocity=torch.tensor([0.1, -0.2]),
        root_position_world=torch.tensor([2.0, 1.0, 0.9]),
        root_orientation_world_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        root_linear_velocity_world=torch.tensor([0.2, 0.0, 0.0]),
        root_angular_velocity_world=torch.zeros(3),
    )
    return folder, skill, pose


class TerrainSkillComposerTest(unittest.TestCase):
    def test_committed_skill_advances_every_source_frame_once(self):
        folder, skill, pose = _fixture()
        state = start_skill(folder, skill, selected_entry_frame=3, current=pose)
        frames = []
        for _ in range(5):
            step = advance_skill(state)
            state = step.state
            frames.append(step.frame)
        self.assertEqual([frame.source_frame for frame in frames], [3, 4, 5, 6, 7])

    def test_first_pose_is_exact_and_offsets_decay(self):
        folder, skill, pose = _fixture()
        state = start_skill(folder, skill, selected_entry_frame=3, current=pose)
        emitted = []
        for _ in range(4):
            step = advance_skill(state)
            state = step.state
            emitted.append(step.frame)
        self.assertTrue(torch.allclose(emitted[0].joint_position, pose.joint_position))
        self.assertTrue(torch.allclose(emitted[0].root_position_world, pose.root_position_world))
        residuals = [frame.inertialization_residual for frame in emitted]
        self.assertTrue(all(a >= b for a, b in zip(residuals, residuals[1:])))

    def test_interrupt_is_allowed_only_at_stable_double_support(self):
        _, skill, _ = _fixture()
        self.assertFalse(can_interrupt_skill(skill, 7))
        self.assertTrue(can_interrupt_skill(skill, 8))

    def test_can_enter_at_internal_double_support_frame(self):
        folder, skill, pose = _fixture()
        state = start_skill(folder, skill, selected_entry_frame=8, current=pose)
        self.assertEqual(advance_skill(state).frame.source_frame, 8)

    def test_explicit_stable_endpoint_stops_before_whole_skill(self):
        folder, skill, pose = _fixture()
        skill.support_mask[7] = True
        state = start_skill(
            folder,
            skill,
            selected_entry_frame=3,
            current=pose,
            playback_stop=8,
        )
        emitted = []
        while True:
            step = advance_skill(state)
            emitted.append(step.frame.source_frame)
            state = step.state
            if step.completed:
                break

        self.assertEqual(emitted, [3, 4, 5, 6, 7])
        with self.assertRaisesRegex(Exception, "already complete"):
            advance_skill(state)

    def test_single_frame_endpoint_has_finite_zero_warp_progress(self):
        folder, skill, pose = _fixture()
        state = start_skill(
            folder,
            skill,
            selected_entry_frame=3,
            current=pose,
            playback_stop=4,
        )

        step = advance_skill(state)

        self.assertTrue(step.completed)
        self.assertTrue(torch.isfinite(step.frame.qpos).all())

    def test_endpoint_warp_preserves_entry_and_reaches_bounded_target(self):
        folder, skill, pose = _fixture()
        skill.support_mask[7] = True
        state = start_skill(
            folder,
            skill,
            selected_entry_frame=3,
            current=pose,
            playback_stop=8,
            target_displacement_local_xy=torch.tensor([0.20, 0.10]),
            target_yaw_delta_rad=torch.tensor(0.40),
            maximum_translation_warp_m=1.0,
            maximum_yaw_warp_rad=1.0,
        )
        emitted = []
        while True:
            step = advance_skill(state)
            emitted.append(step.frame)
            state = step.state
            if step.completed:
                break

        self.assertTrue(
            torch.allclose(emitted[0].root_position_world, pose.root_position_world)
        )
        self.assertTrue(
            torch.allclose(
                emitted[-1].root_position_world[:2],
                pose.root_position_world[:2] + torch.tensor([0.20, 0.10]),
                atol=1e-6,
            )
        )
        w, x, y, z = emitted[-1].root_orientation_world_wxyz
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self.assertAlmostEqual(float(yaw), 0.40, places=5)

    def test_endpoint_warp_caps_translation_and_yaw_correction(self):
        folder, skill, pose = _fixture()
        skill.support_mask[7] = True
        state = start_skill(
            folder,
            skill,
            selected_entry_frame=3,
            current=pose,
            playback_stop=8,
            target_displacement_local_xy=torch.tensor([1.0, 1.0]),
            target_yaw_delta_rad=torch.tensor(2.0),
            maximum_translation_warp_m=0.05,
            maximum_yaw_warp_rad=0.10,
        )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(state.endpoint_translation_warp_world_xy)),
            0.05 + 1e-7,
        )
        self.assertLessEqual(abs(float(state.endpoint_yaw_warp_rad)), 0.10 + 1e-7)


class TerrainSkillExtensionTest(unittest.TestCase):
    def completed_state(self):
        folder, skill, pose = _fixture()
        skill.support_mask[7] = True
        state = start_skill(
            folder,
            skill,
            selected_entry_frame=3,
            current=pose,
            playback_stop=8,
        )
        while state.next_source_frame < state.playback_stop:
            state = advance_skill(state).state
        return state

    def test_extension_preserves_placement_and_offsets(self):
        completed = self.completed_state()

        extended = extend_skill_state(completed, playback_stop=12)

        self.assertEqual(extended.next_source_frame, 8)
        self.assertEqual(extended.playback_stop, 12)
        self.assertIs(extended.offsets, completed.offsets)
        self.assertIs(extended.translation_world, completed.translation_world)
        self.assertIs(extended.yaw_offset, completed.yaw_offset)

    def test_extended_frames_are_exactly_the_unplayed_suffix(self):
        state = extend_skill_state(self.completed_state(), playback_stop=12)
        emitted = []
        while state.next_source_frame < state.playback_stop:
            step = advance_skill(state)
            emitted.append(step.frame.source_frame)
            state = step.state

        self.assertEqual(emitted, [8, 9, 10, 11])

    def test_extension_folds_completed_warp_into_continuous_placement(self):
        folder, skill, pose = _fixture()
        skill.support_mask[7] = True
        folder.clips[0].body_position_world[8] = (
            folder.clips[0].body_position_world[7]
        )
        state = start_skill(
            folder,
            skill,
            selected_entry_frame=3,
            current=pose,
            playback_stop=8,
            target_displacement_local_xy=torch.tensor([0.20, 0.10]),
            target_yaw_delta_rad=torch.tensor(0.40),
            maximum_translation_warp_m=1.0,
            maximum_yaw_warp_rad=1.0,
        )
        final = None
        while state.next_source_frame < state.playback_stop:
            step = advance_skill(state)
            final = step.frame
            state = step.state

        extended = extend_skill_state(state, playback_stop=12)
        first_suffix = advance_skill(extended).frame

        self.assertTrue(
            torch.allclose(
                first_suffix.root_position_world,
                final.root_position_world,
                atol=1e-3,
            )
        )
        self.assertTrue(
            torch.allclose(
                first_suffix.root_orientation_world_wxyz,
                final.root_orientation_world_wxyz,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.equal(
                extended.endpoint_translation_warp_world_xy,
                torch.zeros(2),
            )
        )
        self.assertEqual(float(extended.endpoint_yaw_warp_rad), 0.0)


if __name__ == "__main__":
    unittest.main()
