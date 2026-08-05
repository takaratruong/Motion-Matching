from types import SimpleNamespace
import unittest

import numpy as np
import torch

from mm_sonic.torch_foothold_actions import (
    FootholdAction,
    FootholdActionIndex,
)
from mm_sonic.torch_heading_footprint_search import (
    FootprintActionEdge,
    HeadingFootprintPlan,
)
from mm_sonic.torch_heading_footprint_realizer import (
    RealizationFailure,
    realize_heading_footprint_plan,
)
from mm_sonic.torch_terrain_footprint_candidates import (
    TerrainFootprintCandidate,
)


def _action():
    return FootholdAction(
        clip_index=0,
        start_frame=0,
        end_frame=21,
        start_support=(True, True),
        landing_feet=(0, 1),
        landing_frame_offsets=(10, 20),
        landing_xy_start_frame_m=torch.tensor(
            ((0.25, 0.12), (0.50, -0.12))
        ),
        landing_height_delta_m=torch.zeros(2),
        root_displacement_m=torch.tensor(
            ((0.25, 0.0), (0.50, 0.0))
        ),
        root_yaw_delta_rad=torch.zeros(2),
        minimum_swing_clearance_m=0.04,
        maximum_unsupported_frames=0,
    )


def _plan(heading=(0.0, 1.0)):
    heading_tensor = torch.tensor(heading, dtype=torch.float32)
    heading_tensor /= torch.linalg.vector_norm(heading_tensor)
    lateral = torch.tensor(
        (-heading_tensor[1], heading_tensor[0])
    )
    candidates = []
    for step, (progress, foot) in enumerate(((0.25, 0), (0.50, 1))):
        local = torch.tensor(
            (progress, 0.12 if foot == 0 else -0.12)
        )
        scene = (
            progress * heading_tensor
            + local[1] * lateral
        )
        candidates.append(
            TerrainFootprintCandidate(
                step_index=step,
                foot=foot,
                center_scene_xy=scene,
                center_heading_xy=local,
                offset_heading_xy=torch.zeros(2),
                yaw_scene_rad=float(
                    torch.atan2(heading_tensor[1], heading_tensor[0])
                ),
                surface_height_m=0.0,
                placement_cost=0.0,
            )
        )
    edge = FootprintActionEdge(
        action_key=(0, 0),
        candidate_indices=(0, 0),
        descriptor_cost=0.0,
        transition_cost=0.0,
    )
    return HeadingFootprintPlan(
        heading_scene_xy=heading_tensor,
        footprints=tuple(candidates),
        edges=(edge,),
        action_keys=((0, 0),),
        total_cost=0.0,
    )


def _source(*, terminal_double_support=True):
    frames = 21
    root = np.zeros((frames, 3), dtype=np.float64)
    root[:, 0] = np.linspace(0.0, 0.50, frames)
    root[:, 2] = 0.80
    feet = np.zeros((frames, 2, 3), dtype=np.float64)
    feet[:, 0] = (0.0, 0.12, 0.035)
    feet[:, 1] = (0.0, -0.12, 0.035)
    feet[1:11, 0, 0] = np.linspace(0.0, 0.25, 10)
    feet[10:, 0, 0] = 0.25
    feet[11:, 1, 0] = np.linspace(0.0, 0.50, 10)
    joints = np.zeros((frames, 29), dtype=np.float64)
    joints[:, :6] = feet.reshape(frames, 6)
    body_position = np.zeros((frames, 30, 3), dtype=np.float64)
    body_position[:, 0] = root
    body_position[:, 18:20] = feet
    quaternions = np.zeros((frames, 30, 4), dtype=np.float64)
    quaternions[..., 0] = 1.0
    support = np.ones((frames, 2), dtype=np.bool_)
    support[1:10, 0] = False
    support[11:20, 1] = False
    if not terminal_double_support:
        support[-1, 1] = False
    action = _action()
    return SimpleNamespace(
        clips=(
            SimpleNamespace(
                joint_position=joints,
                body_position_world=body_position,
                body_quaternion_world_wxyz=quaternions,
            ),
        ),
        root_body_index=0,
        foot_body_indices=(18, 19),
        action_index=FootholdActionIndex(
            actions=(action,), _entries={(0, 0): action}
        ),
        support_mask=lambda _clip_index: support,
    )


class _SyntheticKinematics:
    @staticmethod
    def foot_positions(joints, _root, _quaternion):
        joints = np.asarray(joints)
        return joints[:, :6].reshape(len(joints), 2, 3)

    @staticmethod
    def center_of_mass_positions(_joints, root, _quaternion):
        return np.asarray(root)


class _RecordingRetargeter:
    def __init__(self):
        self.calls = []

    def solve_frame(self, **kwargs):
        self.calls.append(kwargs)
        joints = np.asarray(
            kwargs["joint_position"], dtype=np.float64
        ).copy()
        joints[:6] = np.asarray(
            kwargs["target_foot_position_world"]
        ).reshape(6)
        return joints, np.asarray(
            kwargs["root_position_world"], dtype=np.float64
        ).copy()


class _FlatTerrain:
    @staticmethod
    def sample_surface(points_xy):
        points = np.asarray(points_xy)
        return np.zeros(points.shape[:-1], dtype=np.float64)


class HeadingFootprintRealizerTests(unittest.TestCase):
    def test_realizer_rotates_window_and_locks_stance_feet(self):
        retargeter = _RecordingRetargeter()
        result = realize_heading_footprint_plan(
            plan=_plan(heading=(0.0, 1.0)),
            source=_source(),
            terrain=_FlatTerrain(),
            kinematics=_SyntheticKinematics(),
            retargeter=retargeter,
        )

        self.assertLess(result.maximum_stance_error_m, 0.005)
        self.assertGreater(result.root_position_world[-1, 1], 0.49)
        self.assertAlmostEqual(
            result.root_position_world[-1, 0], 0.0, places=6
        )
        self.assertTrue(result.source_support_mask[-1].all())
        self.assertEqual(
            tuple(result.source_frame_provenance[-1]), (0, 20)
        )
        self.assertEqual(len(retargeter.calls), 21)

    def test_realizer_rejects_terminal_mid_swing(self):
        with self.assertRaisesRegex(
            RealizationFailure, "terminal_mid_swing"
        ):
            realize_heading_footprint_plan(
                plan=_plan(),
                source=_source(terminal_double_support=False),
                terrain=_FlatTerrain(),
                kinematics=_SyntheticKinematics(),
                retargeter=_RecordingRetargeter(),
            )


if __name__ == "__main__":
    unittest.main()
