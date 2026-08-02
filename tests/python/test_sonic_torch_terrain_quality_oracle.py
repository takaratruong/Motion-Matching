import unittest
from dataclasses import replace

import numpy as np
import torch

from mm_sonic.torch_contact_oracle_actions import (
    ContactPhaseAction,
    ContactPhaseActionIndex,
    ContactPhaseInventory,
)
from mm_sonic.torch_contact_oracle_search import OracleConstraints
from mm_sonic.torch_terrain_action_quality import build_native_quality_index
from mm_sonic.torch_terrain_quality_oracle import rank_quality_actions
from mm_sonic.torch_terrain_quality_states import capture_quality_states


def _action(index, *, landing_x, entry_joint=0.0, entry_support=(True, False)):
    orientation = torch.zeros((2, 4), dtype=torch.float32)
    orientation[:, 0] = 1.0
    feet = torch.tensor(
        (
            ((0.0, 0.1, 0.035), (0.0, -0.1, 0.12)),
            ((0.0, 0.1, 0.035), (landing_x, -0.1, 0.035)),
        ),
        dtype=torch.float32,
    )
    joints = torch.zeros((2, 29), dtype=torch.float32)
    joints[0] = entry_joint
    return ContactPhaseAction(
        clip_index=index,
        start_frame=10 * index,
        end_frame=10 * index + 2,
        swing_foot=1,
        entry_support=torch.tensor(entry_support),
        exit_support=torch.tensor((True, True)),
        support_mask=torch.tensor((entry_support, (True, True))),
        joint_position=joints,
        joint_velocity=torch.zeros((2, 29), dtype=torch.float32),
        root_position_local=torch.tensor(
            ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)), dtype=torch.float32
        ),
        root_yaw_local=torch.zeros(2, dtype=torch.float32),
        root_orientation_local_wxyz=orientation,
        foot_position_local=feet,
        foot_surface_delta_m=torch.zeros((2, 2), dtype=torch.float32),
        minimum_swing_clearance_m=0.05,
    )


def _index():
    landing = _action(0, landing_x=0.30, entry_joint=0.18)
    continuity = _action(1, landing_x=0.70)
    combined = _action(2, landing_x=0.40, entry_joint=0.03)
    two_step = _action(3, landing_x=0.55, entry_joint=0.08)
    successor = _action(4, landing_x=0.20, entry_support=(True, True))
    successor_feet = successor.foot_position_local.clone()
    successor_feet[0] = torch.tensor(
        ((-0.2, 0.1, 0.035), (0.35, -0.1, 0.035))
    )
    successor = replace(successor, foot_position_local=successor_feet)
    actions = (landing, continuity, combined, two_step, successor)
    return ContactPhaseActionIndex(
        actions=actions,
        inventory=ContactPhaseInventory(
            retained_count=len(actions), rejected_by_reason={}
        ),
        exact_successor_indices=(None, None, None, 4, None),
    )


def _state(*, support=(True, False)):
    arrays = {
        "qpos": np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 29]),
        "joint_position": np.zeros((1, 29)),
        "joint_velocity": np.zeros((1, 29)),
        "root_position_world": np.zeros((1, 3)),
        "root_yaw_world": np.zeros(1),
        "foot_position_world": np.array(
            [[[0.0, 0.1, 0.035], [0.0, -0.1, 0.035]]]
        ),
        "foot_surface_height_m": np.zeros((1, 2)),
        "command_velocity_world_xy": np.array([[0.2, 0.0]]),
        "command_heading_world_yaw": np.zeros(1),
        "command_segment_index": np.zeros(1, dtype=np.int64),
        "selected_clip_path": np.array(["clip.npz"]),
        "selected_source_frame": np.zeros(1, dtype=np.int64),
    }
    return capture_quality_states(
        route_name="synthetic",
        arrays=arrays,
        source_support_mask=np.array([support], dtype=bool),
        terrain_patch_sampler=lambda points: np.zeros(points.shape[0]),
    )[0]


def _flat(points):
    return torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)


class TerrainQualityOracleTests(unittest.TestCase):
    def test_exhaustive_ranking_keeps_independent_winners(self):
        index = _index()
        result = rank_quality_actions(
            state=_state(),
            index=index,
            native_quality=build_native_quality_index(index),
            desired_landing_world_xyz=np.array((0.30, -0.1, 0.035)),
            command_target_world_xy=np.array((0.20, 0.0)),
            sample_surface=_flat,
            constraints=OracleConstraints(),
            top_k=5,
        )

        self.assertEqual(result.evaluated_action_count, len(index.actions))
        self.assertEqual(result.rejected_by_reason, {"entry-support": 1})
        self.assertEqual(result.best_landing[0].action_index, 0)
        self.assertEqual(result.best_continuity[0].action_index, 1)
        self.assertEqual(result.best_combined[0].action_index, 2)
        self.assertEqual(result.best_two_step[0].action_index, 3)
        self.assertEqual(result.best_two_step[0].exact_successor_index, 4)
        self.assertEqual(len(result.deterministic_sha256), 64)

    def test_ties_use_stable_source_key_order(self):
        action = _action(0, landing_x=0.30)
        later = replace(action, clip_index=1)
        index = ContactPhaseActionIndex(
            actions=(action, later),
            inventory=ContactPhaseInventory(retained_count=2, rejected_by_reason={}),
            exact_successor_indices=(None, None),
        )
        result = rank_quality_actions(
            state=_state(),
            index=index,
            native_quality=build_native_quality_index(index),
            desired_landing_world_xyz=np.array((0.30, -0.1, 0.035)),
            command_target_world_xy=np.array((0.20, 0.0)),
            sample_surface=_flat,
            constraints=OracleConstraints(),
        )

        self.assertEqual([row.action_index for row in result.best_combined], [0, 1])

    def test_zero_feasible_candidates_is_structured_result(self):
        index = _index()
        result = rank_quality_actions(
            state=_state(support=(False, False)),
            index=index,
            native_quality=build_native_quality_index(index),
            desired_landing_world_xyz=np.array((0.30, -0.1, 0.035)),
            command_target_world_xy=np.array((0.20, 0.0)),
            sample_surface=_flat,
            constraints=OracleConstraints(),
        )

        self.assertEqual(result.best_landing, ())
        self.assertEqual(result.best_continuity, ())
        self.assertEqual(result.best_combined, ())
        self.assertEqual(result.best_two_step, ())
        self.assertEqual(result.rejected_by_reason, {"entry-support": 5})

    def test_missing_desired_foothold_fails_before_enumeration(self):
        index = _index()
        with self.assertRaisesRegex(ValueError, "desired landing"):
            rank_quality_actions(
                state=_state(),
                index=index,
                native_quality=build_native_quality_index(index),
                desired_landing_world_xyz=None,
                command_target_world_xy=np.array((0.20, 0.0)),
                sample_surface=_flat,
                constraints=OracleConstraints(),
            )


if __name__ == "__main__":
    unittest.main()
