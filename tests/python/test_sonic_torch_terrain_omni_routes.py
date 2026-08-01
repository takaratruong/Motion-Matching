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
            command_segments = {
                command.segment for command in route.commands
            }
            self.assertTrue(route.outcome.required_segments)
            self.assertLessEqual(
                set(route.outcome.required_segments), command_segments
            )
            self.assertGreater(route.outcome.min_segment_progress_ratio, 0.0)
            self.assertGreater(route.outcome.min_elevated_foot_samples, 0)

        for route in routes:
            if route.required_outcome == "turn":
                self.assertIsNotNone(
                    route.outcome.final_heading_error_max_rad
                )
            if route.required_outcome == "exit":
                self.assertEqual(route.outcome.final_surface, "flat")
            if route.required_outcome == "mount":
                self.assertEqual(route.outcome.final_surface, "elevated")

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

    def test_routes_reach_the_stair_before_exercising_the_named_outcome(self):
        routes = {route.name: route for route in same_stair_routes()}

        def positions(route):
            velocity = np.asarray(
                [command.velocity_stair_xy for command in route.commands]
            )
            duration = np.asarray(
                [command.frames * 0.02 for command in route.commands]
            )
            return np.vstack(
                (np.zeros(2), np.cumsum(velocity * duration[:, None], axis=0))
            )

        for route in routes.values():
            self.assertGreaterEqual(
                float(positions(route)[:, 0].max()),
                1.30,
                route.name,
            )

        side_mount = positions(routes["side-mount-left"])
        self.assertGreaterEqual(float(side_mount[:, 1].max()), 0.42)
        self.assertLessEqual(float(side_mount[-1, 1]), 0.10)

        for name in ("diagonal-down-left", "diagonal-down-right"):
            route = routes[name]
            first_descent = next(
                index
                for index, command in enumerate(route.commands)
                if command.velocity_stair_xy[0] < -0.05
            )
            self.assertGreaterEqual(
                float(positions(route)[first_descent, 0]), 1.75, name
            )

        for name in ("side-exit-upper-left", "side-exit-upper-right"):
            route = routes[name]
            first_exit = next(
                index
                for index, command in enumerate(route.commands)
                if abs(command.velocity_stair_xy[1]) > 0.20
            )
            self.assertGreaterEqual(
                float(positions(route)[first_exit, 0]), 1.75, name
            )
            self.assertGreaterEqual(
                route.commands[first_exit].frames, 150, name
            )

        expected_turns = {
            "turn-45-lower-left": math.pi / 4,
            "turn-90-middle-left": math.pi / 2,
            "turn-180-upper-left": math.pi,
        }
        for name, expected in expected_turns.items():
            self.assertAlmostEqual(
                routes[name].commands[-1].heading_stair_yaw, expected
            )

    def test_all_turns_replay_full_speed_live_wasd(self):
        routes = {route.name: route for route in same_stair_routes()}
        for name in (
            "turn-45-lower-left",
            "turn-45-lower-right",
            "turn-90-middle-left",
            "turn-90-middle-right",
            "turn-180-upper-left",
            "turn-180-upper-right",
        ):
            turn = routes[name].commands[-1]
            self.assertAlmostEqual(
                math.hypot(*turn.velocity_stair_xy), 0.38
            )
            self.assertGreaterEqual(turn.frames, 100)
            self.assertAlmostEqual(
                turn.heading_stair_yaw,
                math.atan2(
                    turn.velocity_stair_xy[1],
                    turn.velocity_stair_xy[0],
                ),
            )
            self.assertIn(turn.segment, routes[name].outcome.required_segments)
            self.assertLessEqual(
                routes[name].outcome.final_command_lateral_drift_max_m,
                0.10,
            )

    def test_sideways_and_backward_routes_keep_facing_independent_of_velocity(self):
        routes = {route.name: route for route in same_stair_routes()}
        for name in (
            "side-mount-left",
            "side-mount-right",
            "cross-tread-left-to-right",
            "cross-tread-right-to-left",
            "side-exit-upper-left",
            "side-exit-upper-right",
        ):
            lateral = [
                command
                for command in routes[name].commands
                if abs(command.velocity_stair_xy[1]) > 0.20
                and abs(command.velocity_stair_xy[0]) < 0.05
            ]
            self.assertTrue(lateral, name)
            self.assertTrue(
                all(abs(command.heading_stair_yaw) < 1e-12 for command in lateral),
                name,
            )
        reversal = routes["riser-reversal"].commands[-1]
        self.assertLess(reversal.velocity_stair_xy[0], 0.0)
        self.assertAlmostEqual(reversal.heading_stair_yaw, 0.0)

    def test_diagonal_descent_faces_its_sparse_supported_travel_direction(self):
        routes = {route.name: route for route in same_stair_routes()}
        for name in ("diagonal-down-left", "diagonal-down-right"):
            descent = next(
                command
                for command in routes[name].commands
                if command.segment == "diagonal-down"
            )
            self.assertAlmostEqual(
                descent.heading_stair_yaw,
                math.atan2(*reversed(descent.velocity_stair_xy)),
            )


if __name__ == "__main__":
    unittest.main()
