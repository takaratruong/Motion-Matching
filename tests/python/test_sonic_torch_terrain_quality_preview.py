import unittest

import numpy as np
import torch

from mm_sonic.torch_contact_oracle_actions import ContactPhaseAction
from mm_sonic.torch_terrain_contact_composition import place_action_contact_anchored
from mm_sonic.torch_terrain_quality_preview import (
    build_quality_preview,
    build_quality_preview_from_placement,
    quality_state_as_oracle,
)
from mm_sonic.torch_terrain_quality_states import capture_quality_states


class _SyntheticFeet:
    @staticmethod
    def foot_positions(joints, roots, _quaternions):
        joints = np.asarray(joints)
        roots = np.asarray(roots)
        feet = np.repeat(roots[:, None, :], 2, axis=1)
        feet[:, 0, 0] += joints[:, 0]
        feet[:, 1, 0] += joints[:, 1]
        feet[:, 0, 1] += 0.1
        feet[:, 1, 1] -= 0.1
        feet[:, :, 2] += 0.035
        return feet


def _action():
    orientation = torch.zeros((3, 4), dtype=torch.float32)
    orientation[:, 0] = 1.0
    return ContactPhaseAction(
        clip_index=0,
        start_frame=0,
        end_frame=3,
        swing_foot=1,
        entry_support=torch.tensor((True, False)),
        exit_support=torch.tensor((True, True)),
        support_mask=torch.tensor(
            ((True, False), (True, False), (True, True))
        ),
        joint_position=torch.zeros((3, 29), dtype=torch.float32),
        joint_velocity=torch.zeros((3, 29), dtype=torch.float32),
        root_position_local=torch.zeros((3, 3), dtype=torch.float32),
        root_yaw_local=torch.zeros(3, dtype=torch.float32),
        root_orientation_local_wxyz=orientation,
        foot_position_local=torch.tensor(
            (
                ((0.0, 0.1, 0.035), (0.0, -0.1, 0.10)),
                ((0.0, 0.1, 0.035), (0.1, -0.1, 0.10)),
                ((0.0, 0.1, 0.035), (0.2, -0.1, 0.035)),
            ),
            dtype=torch.float32,
        ),
        foot_surface_delta_m=torch.zeros((3, 2), dtype=torch.float32),
        minimum_swing_clearance_m=0.05,
    )


def _state():
    joints = np.zeros((1, 29))
    joints[0, 0] = 0.2
    feet = _SyntheticFeet.foot_positions(
        joints, np.zeros((1, 3)), np.array(((1.0, 0.0, 0.0, 0.0),))
    )
    arrays = {
        "qpos": np.concatenate(
            (np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]), joints), axis=1
        ),
        "joint_position": joints,
        "joint_velocity": np.zeros((1, 29)),
        "root_position_world": np.zeros((1, 3)),
        "root_yaw_world": np.zeros(1),
        "foot_position_world": feet,
        "foot_surface_height_m": np.zeros((1, 2)),
        "command_velocity_world_xy": np.zeros((1, 2)),
        "command_heading_world_yaw": np.zeros(1),
        "command_segment_index": np.zeros(1, dtype=np.int64),
        "selected_clip_path": np.array(["clip.npz"]),
        "selected_source_frame": np.zeros(1, dtype=np.int64),
    }
    return capture_quality_states(
        route_name="preview",
        arrays=arrays,
        source_support_mask=np.array(((True, False),)),
        terrain_patch_sampler=lambda points: np.zeros(points.shape[0]),
    )[0]


class TerrainQualityPreviewTests(unittest.TestCase):
    def test_isolates_native_placement_and_inertialization_drift(self):
        preview = build_quality_preview(
            state=_state(),
            action=_action(),
            foot_kinematics=_SyntheticFeet(),
            inertialization_halflife_s=0.10,
        )

        self.assertAlmostEqual(preview.native.source_stance_drift_m, 0.0)
        self.assertAlmostEqual(preview.placed.source_stance_drift_m, 0.0)
        self.assertGreater(preview.composed.source_stance_drift_m, 0.0)
        self.assertGreater(preview.native.joint_boundary_jump_rad, 0.19)
        self.assertAlmostEqual(preview.composed.joint_boundary_jump_rad, 0.0, places=6)
        self.assertEqual(preview.composed.qpos.shape, (3, 36))
        np.testing.assert_array_equal(
            preview.composed.source_support_mask,
            _action().support_mask.numpy(),
        )
        self.assertFalse(preview.composed.qpos.flags.writeable)

    def test_rigid_placement_preserves_pairwise_foot_displacement(self):
        preview = build_quality_preview(
            state=_state(),
            action=_action(),
            foot_kinematics=_SyntheticFeet(),
            inertialization_halflife_s=0.10,
        )

        native_pair = preview.native.foot_position_world[:, 1] - preview.native.foot_position_world[:, 0]
        placed_pair = preview.placed.foot_position_world[:, 1] - preview.placed.foot_position_world[:, 0]
        np.testing.assert_allclose(placed_pair, native_pair, atol=1e-7)

    def test_rejects_nonpositive_halflife_before_fk(self):
        with self.assertRaisesRegex(ValueError, "halflife"):
            build_quality_preview(
                state=_state(),
                action=_action(),
                foot_kinematics=object(),
                inertialization_halflife_s=0.0,
            )

    def test_explicit_contact_placement_recomputes_transition_offsets(self):
        state = _state()
        action = _action()
        current = quality_state_as_oracle(state, action.joint_position)
        contact_placement = place_action_contact_anchored(action, current).placed

        preview = build_quality_preview_from_placement(
            state=state,
            action=action,
            placement=contact_placement,
            foot_kinematics=_SyntheticFeet(),
            inertialization_halflife_s=0.10,
        )

        self.assertGreater(preview.placed.root_boundary_jump_m, 0.19)
        self.assertAlmostEqual(preview.composed.root_boundary_jump_m, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
