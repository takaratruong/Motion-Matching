import unittest

import numpy as np
import torch

from mm_sonic.torch_contact_oracle_actions import (
    ContactPhaseAction,
    ContactPhaseActionIndex,
    ContactPhaseInventory,
)
from mm_sonic.torch_terrain_action_quality import (
    build_native_quality_index,
    describe_native_action_quality,
)


def _action(
    *,
    start_frame: int = 0,
    stance_slide_m: float = 0.0,
    landing_x_m: float = 0.30,
    minimum_clearance_m: float = 0.04,
):
    frames = 4
    support = torch.tensor(
        [[True, False], [True, False], [True, False], [True, True]],
        dtype=torch.bool,
    )
    feet = torch.zeros((frames, 2, 3), dtype=torch.float32)
    feet[:, 0, 0] = torch.linspace(0.0, stance_slide_m, frames)
    feet[:, 0, 1] = 0.10
    feet[:, 1, 0] = torch.linspace(0.0, landing_x_m, frames)
    feet[:, 1, 1] = -0.10
    feet[:, :, 2] = 0.035
    root = torch.zeros((frames, 3), dtype=torch.float32)
    root[:, 0] = torch.linspace(0.0, 0.20, frames)
    yaw = torch.linspace(0.0, 0.10, frames)
    orientation = torch.zeros((frames, 4), dtype=torch.float32)
    orientation[:, 0] = 1.0
    velocity = torch.zeros((frames, 29), dtype=torch.float32)
    velocity[0, 0] = 2.0
    velocity[-1, 1] = 3.0
    return ContactPhaseAction(
        clip_index=0,
        start_frame=start_frame,
        end_frame=start_frame + frames,
        swing_foot=1,
        entry_support=support[0],
        exit_support=support[-1],
        support_mask=support,
        joint_position=torch.zeros((frames, 29), dtype=torch.float32),
        joint_velocity=velocity,
        root_position_local=root,
        root_yaw_local=yaw,
        root_orientation_local_wxyz=orientation,
        foot_position_local=feet,
        foot_surface_delta_m=torch.zeros((frames, 2), dtype=torch.float32),
        minimum_swing_clearance_m=minimum_clearance_m,
    )


def _index(actions, successors):
    return ContactPhaseActionIndex(
        actions=tuple(actions),
        inventory=ContactPhaseInventory(
            retained_count=len(actions), rejected_by_reason={}
        ),
        exact_successor_indices=tuple(successors),
    )


class TerrainActionQualityTests(unittest.TestCase):
    def test_describes_natural_landing_root_motion_and_boundary_speed(self):
        action = _action()
        quality = describe_native_action_quality(4, action, None)

        np.testing.assert_allclose(
            quality.natural_landing_local_xyz,
            [0.30, -0.10, 0.035],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            quality.root_displacement_local_xy, [0.20, 0.0], atol=1e-6
        )
        self.assertEqual(quality.landing_frame_offset, 3)
        self.assertAlmostEqual(quality.root_yaw_delta_rad, 0.10, places=6)
        self.assertAlmostEqual(quality.entry_joint_speed_norm, 2.0)
        self.assertAlmostEqual(quality.terminal_joint_speed_norm, 3.0)

    def test_native_stance_drift_uses_source_support_intervals(self):
        clean = describe_native_action_quality(0, _action(), None)
        sliding = describe_native_action_quality(
            0, _action(stance_slide_m=0.09), None
        )

        self.assertAlmostEqual(clean.source_stance_drift_m, 0.0)
        self.assertAlmostEqual(sliding.source_stance_drift_m, 0.09, places=6)

    def test_build_index_preserves_action_order_and_exact_successor(self):
        first = _action(start_frame=0)
        second = _action(start_frame=3)
        qualities = build_native_quality_index(_index((first, second), (1, None)))

        self.assertEqual([quality.action_index for quality in qualities], [0, 1])
        self.assertEqual(qualities[0].exact_successor_index, 1)
        self.assertIsNone(qualities[1].exact_successor_index)
        self.assertFalse(qualities[0].natural_landing_local_xyz.flags.writeable)

    def test_rejects_inconsistent_action_index(self):
        with self.assertRaisesRegex(ValueError, "action index"):
            describe_native_action_quality(-1, _action(), None)


if __name__ == "__main__":
    unittest.main()
