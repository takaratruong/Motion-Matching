import math
import unittest

import numpy as np

from mm_sonic.torch_terrain_omni_routes import (
    StairFrame,
    same_stair_routes,
    world_commands,
)


EXPECTED_NAMES = (
    "side-mount-left",
    "side-mount-right",
    "cross-tread-left-to-right",
    "cross-tread-right-to-left",
    "turn-45-lower-left",
    "turn-45-lower-right",
    "turn-90-middle-left",
    "turn-90-middle-right",
    "turn-180-upper-left",
    "turn-180-upper-right",
    "diagonal-up-left",
    "diagonal-up-right",
    "diagonal-down-left",
    "diagonal-down-right",
    "side-exit-lower-left",
    "side-exit-lower-right",
    "side-exit-upper-left",
    "side-exit-upper-right",
    "riser-stop-restart",
    "riser-reversal",
    "mixed-adversarial",
)


class SameStairOmnidirectionalRouteTests(unittest.TestCase):
    def test_route_inventory_and_contract_are_frozen(self):
        routes = same_stair_routes()

        self.assertEqual(tuple(route.name for route in routes), EXPECTED_NAMES)
        self.assertEqual(len({route.name for route in routes}), len(routes))
        for route in routes:
            self.assertGreaterEqual(len(route.commands), 2)
            self.assertTrue(route.commands[0].reset_before)
            self.assertFalse(
                any(command.reset_before for command in route.commands[1:])
            )
            self.assertGreater(sum(command.frames for command in route.commands), 0)
            self.assertTrue(all(command.frames > 0 for command in route.commands))
            self.assertTrue(all(command.segment for command in route.commands))
            self.assertIn(
                route.required_outcome,
                {"mount", "traverse", "turn", "exit", "mixed"},
            )

    def test_world_transform_rotates_commands_without_changing_metadata(self):
        frame = StairFrame(
            origin_world_xy=(1.0, -2.0),
            ascent_world_yaw=math.pi / 2.0,
            width_m=0.6223,
            tread_depth_m=0.3302,
            riser_height_m=0.1778,
            tread_count=3,
        )
        route = same_stair_routes()[0]
        transformed = world_commands(frame, route)

        self.assertEqual(len(transformed), len(route.commands))
        for source, target in zip(route.commands, transformed):
            expected_velocity = np.array(
                [-source.velocity_stair_xy[1], source.velocity_stair_xy[0]]
            )
            np.testing.assert_allclose(
                target.velocity_world_xy, expected_velocity, atol=1e-12
            )
            expected_heading = math.atan2(
                math.sin(source.heading_stair_yaw + math.pi / 2.0),
                math.cos(source.heading_stair_yaw + math.pi / 2.0),
            )
            self.assertAlmostEqual(target.heading_world_yaw, expected_heading)
            self.assertEqual(target.frames, source.frames)
            self.assertEqual(target.segment, source.segment)
            self.assertEqual(target.reset_before, source.reset_before)

    def test_mirrored_routes_have_exact_opposite_lateral_commands(self):
        routes = {route.name: route for route in same_stair_routes()}
        pairs = (
            ("side-mount-left", "side-mount-right"),
            ("diagonal-up-left", "diagonal-up-right"),
            ("diagonal-down-left", "diagonal-down-right"),
        )
        for left_name, right_name in pairs:
            left = routes[left_name]
            right = routes[right_name]
            self.assertEqual(len(left.commands), len(right.commands))
            for left_command, right_command in zip(
                left.commands, right.commands
            ):
                self.assertAlmostEqual(
                    left_command.velocity_stair_xy[0],
                    right_command.velocity_stair_xy[0],
                )
                self.assertAlmostEqual(
                    left_command.velocity_stair_xy[1],
                    -right_command.velocity_stair_xy[1],
                )
                self.assertAlmostEqual(
                    left_command.heading_stair_yaw,
                    -right_command.heading_stair_yaw,
                )
                self.assertEqual(left_command.frames, right_command.frames)
                self.assertEqual(left_command.segment, right_command.segment)


if __name__ == "__main__":
    unittest.main()
