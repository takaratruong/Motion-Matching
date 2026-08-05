import math
import unittest

import torch

from mm_sonic.torch_foothold_actions import (
    FootholdAction,
    FootholdActionIndex,
    FootholdTransitionGraph,
)
from mm_sonic.torch_heading_footprint_path import (
    NominalFootprint,
    heading_local_to_scene,
)
from mm_sonic.torch_heading_footprint_search import (
    search_heading_footprint_plan,
)
from mm_sonic.torch_terrain_footprint_candidates import (
    TerrainFootprintCandidate,
    TerrainFootprintLayer,
)


def _action(
    *,
    start_frame,
    landing_xy,
    root=((0.25, 0.0), (0.50, 0.0)),
):
    return FootholdAction(
        clip_index=0,
        start_frame=start_frame,
        end_frame=start_frame + 21,
        start_support=(True, True),
        landing_feet=(0, 1),
        landing_frame_offsets=(10, 20),
        landing_xy_start_frame_m=torch.tensor(
            landing_xy, dtype=torch.float32
        ),
        landing_height_delta_m=torch.zeros(2),
        root_displacement_m=torch.tensor(root, dtype=torch.float32),
        root_yaw_delta_rad=torch.zeros(2),
        minimum_swing_clearance_m=0.05,
        maximum_unsupported_frames=0,
    )


def _index(*actions):
    return FootholdActionIndex(
        actions=tuple(actions),
        _entries={
            (action.clip_index, action.start_frame): action
            for action in actions
        },
    )


def _graph(*actions):
    count = len(actions)
    entry = torch.zeros((count, 29), dtype=torch.float32)
    terminal = torch.zeros((count, 29), dtype=torch.float32)
    entry[:, 0] = torch.tensor((0.0, 1.0, 2.0))
    terminal[:, 0] = torch.tensor((10.0, 2.0, 10.0))
    return FootholdTransitionGraph(
        action_keys=tuple(
            (action.clip_index, action.start_frame) for action in actions
        ),
        entry_joint_position=entry,
        entry_joint_velocity=torch.zeros_like(entry),
        terminal_joint_position=terminal,
        terminal_joint_velocity=torch.zeros_like(terminal),
    )


def _layers(heading):
    heading = heading / torch.linalg.vector_norm(heading)
    yaw = math.atan2(float(heading[1]), float(heading[0]))
    origin = torch.zeros(2)
    layers = []
    for step_index, (progress, foot) in enumerate(
        ((0.25, 0), (0.50, 1), (0.75, 0), (1.00, 1))
    ):
        local = torch.tensor(
            (progress, 0.12 if foot == 0 else -0.12),
            dtype=torch.float32,
        )
        scene = heading_local_to_scene(local, origin, heading)
        nominal = NominalFootprint(
            step_index=step_index,
            foot=foot,
            center_scene_xy=scene,
            center_heading_xy=local,
            yaw_scene_rad=yaw,
            contact_frame=(step_index + 1) * 10,
        )
        candidate = TerrainFootprintCandidate(
            step_index=step_index,
            foot=foot,
            center_scene_xy=scene,
            center_heading_xy=local,
            offset_heading_xy=torch.zeros(2),
            yaw_scene_rad=yaw,
            surface_height_m=0.0,
            placement_cost=0.0,
        )
        layers.append(
            TerrainFootprintLayer(
                step_index=step_index,
                nominal=nominal,
                candidates=(candidate,),
            )
        )
    return tuple(layers)


class HeadingFootprintSearchTests(unittest.TestCase):
    def setUp(self):
        self.greedy_dead_end = _action(
            start_frame=10,
            landing_xy=((0.25, 0.12), (0.50, -0.12)),
        )
        self.compatible_first = _action(
            start_frame=20,
            landing_xy=((0.23, 0.10), (0.48, -0.10)),
        )
        self.successor = _action(
            start_frame=30,
            landing_xy=((0.25, 0.12), (0.50, -0.12)),
        )
        self.actions = (
            self.greedy_dead_end,
            self.compatible_first,
            self.successor,
        )

    def _search(self, heading):
        return search_heading_footprint_plan(
            layers=_layers(heading),
            heading_scene_xy=heading,
            action_index=_index(*self.actions),
            transition_graph=_graph(*self.actions),
            beam_width=4,
            xy_tolerance_m=0.10,
            height_tolerance_m=0.04,
            timing_tolerance_frames=8,
            maximum_yaw_delta_rad=0.10,
            maximum_transition_position_error_rad=0.10,
            maximum_transition_velocity_error_rad_s=0.10,
        )

    def test_beam_chooses_displaced_contact_with_compatible_successor(self):
        result = self._search(torch.tensor((1.0, 0.0)))

        self.assertEqual(result.action_keys, ((0, 20), (0, 30)))
        self.assertEqual(len(result.footprints), 4)

    def test_global_rotation_preserves_action_chain_and_cost(self):
        forward = self._search(torch.tensor((1.0, 0.0)))
        diagonal = self._search(
            torch.tensor(
                (math.sqrt(0.5), math.sqrt(0.5)), dtype=torch.float32
            )
        )

        self.assertEqual(diagonal.action_keys, forward.action_keys)
        self.assertAlmostEqual(
            diagonal.total_cost, forward.total_cost, places=6
        )


if __name__ == "__main__":
    unittest.main()
