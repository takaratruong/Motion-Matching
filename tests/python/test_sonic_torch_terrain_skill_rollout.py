import unittest

import torch

from mm_sonic.torch_terrain_omni_routes import same_stair_routes
from mm_sonic.torch_terrain_skill_rollout import (
    command_skill_compatible,
    qualification_routes,
)


class TerrainSkillRolloutTest(unittest.TestCase):
    def test_qualification_slice_uses_frozen_route_order(self):
        self.assertEqual(
            [route.name for route in qualification_routes(same_stair_routes())],
            [
                "cross-tread-left-to-right",
                "turn-90-middle-left",
                "diagonal-down-left",
                "side-exit-upper-left",
                "riser-reversal",
                "mixed-adversarial",
            ],
        )

    def test_qualification_slice_rejects_missing_route(self):
        with self.assertRaisesRegex(ValueError, "missing qualification"):
            qualification_routes(same_stair_routes()[:-1])

    def test_whole_skill_gate_requires_commanded_turn_direction(self):
        travel = torch.tensor([0.4, 0.8])
        velocity = torch.tensor([0.0, 0.4])
        self.assertTrue(
            command_skill_compatible(
                skill_travel_local_xy=travel,
                skill_yaw_delta_rad=torch.tensor(1.1),
                requested_velocity_local_xy=velocity,
                requested_heading_delta_rad=torch.tensor(1.57),
            )
        )
        self.assertFalse(
            command_skill_compatible(
                skill_travel_local_xy=travel,
                skill_yaw_delta_rad=torch.tensor(-1.1),
                requested_velocity_local_xy=velocity,
                requested_heading_delta_rad=torch.tensor(1.57),
            )
        )

    def test_whole_skill_gate_rejects_large_turn_for_straight_command(self):
        self.assertFalse(
            command_skill_compatible(
                skill_travel_local_xy=torch.tensor([1.0, 0.0]),
                skill_yaw_delta_rad=torch.tensor(1.3),
                requested_velocity_local_xy=torch.tensor([0.4, 0.0]),
                requested_heading_delta_rad=torch.tensor(0.0),
            )
        )

    def test_whole_skill_gate_rejects_stationary_skill_for_moving_command(self):
        self.assertFalse(
            command_skill_compatible(
                skill_travel_local_xy=torch.zeros(2),
                skill_yaw_delta_rad=torch.tensor(0.0),
                requested_velocity_local_xy=torch.tensor([0.4, 0.0]),
                requested_heading_delta_rad=torch.tensor(0.0),
            )
        )


if __name__ == "__main__":
    unittest.main()
