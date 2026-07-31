import unittest

from mm_sonic.torch_transition_reachability import (
    ReachabilityLimits,
    bounded_transition_reachability,
)


class TorchTransitionReachabilityTests(unittest.TestCase):
    def test_two_hop_safe_terminal_is_found_without_future_command(self):
        graph = {
            "start": ("greedy-dead-end", "bridge"),
            "greedy-dead-end": (),
            "bridge": ("lateral-terminal",),
            "lateral-terminal": (),
        }
        identities = {
            "start": (0, 10),
            "greedy-dead-end": (1, 20),
            "bridge": (2, 30),
            "lateral-terminal": (3, 40),
        }
        current_command = object()
        future_command = object()
        commands_seen = []
        source_advance_limits = []

        def expand(state, command, max_source_advance_frames):
            commands_seen.append(command)
            source_advance_limits.append(max_source_advance_frames)
            return graph[state]

        result = bounded_transition_reachability(
            "start",
            command=current_command,
            command_change_frame=120,
            expand=expand,
            is_safe=lambda _state, command: command is current_command,
            is_terminal=lambda state, command: (
                state == "lateral-terminal" and command is current_command
            ),
            identity=lambda state: identities[state],
            limits=ReachabilityLimits(
                max_depth=2,
                beam_width=2,
                max_expanded_states=8,
                max_source_advance_frames=15,
            ),
        )

        self.assertTrue(result.reachable)
        self.assertEqual(result.command_change_frame, 120)
        self.assertEqual(result.depth_used, 2)
        self.assertEqual(
            (result.first_clip_index, result.first_frame_index),
            identities["bridge"],
        )
        self.assertEqual(
            (result.terminal_clip_index, result.terminal_frame_index),
            identities["lateral-terminal"],
        )
        self.assertEqual(result.expanded_state_count, 3)
        self.assertFalse(result.budget_exhausted)
        self.assertGreaterEqual(result.planner_compute_ns, 0)
        self.assertNotIn(future_command, commands_seen)
        self.assertTrue(
            commands_seen
            and all(command is current_command for command in commands_seen)
        )
        self.assertEqual(source_advance_limits, [15, 15, 15])

    def test_budget_exhaustion_is_read_only_and_reports_no_terminal(self):
        incumbent = {"selected": (7, 11)}
        before = dict(incumbent)

        def expand(state, _command, _max_source_advance_frames):
            if state == "start":
                return tuple(f"candidate-{index}" for index in range(10))
            return ()

        result = bounded_transition_reachability(
            "start",
            command=(0.0, -0.5),
            command_change_frame=200,
            expand=expand,
            is_safe=lambda _state, _command: True,
            is_terminal=lambda _state, _command: False,
            identity=lambda state: (
                0,
                0 if state == "start" else int(state.split("-")[-1]) + 1,
            ),
            limits=ReachabilityLimits(
                max_depth=2,
                beam_width=8,
                max_expanded_states=3,
                max_source_advance_frames=15,
            ),
        )

        self.assertFalse(result.reachable)
        self.assertEqual(result.expanded_state_count, 3)
        self.assertTrue(result.budget_exhausted)
        self.assertEqual(result.depth_used, 1)
        self.assertIsNone(result.first_clip_index)
        self.assertIsNone(result.first_frame_index)
        self.assertIsNone(result.terminal_clip_index)
        self.assertIsNone(result.terminal_frame_index)
        self.assertEqual(incumbent, before)

    def test_budget_round_robins_across_beam_parents(self):
        graph = {
            "start": ("first", "second"),
            "first": ("first-dead-1", "first-dead-2"),
            "second": ("second-terminal", "second-dead"),
        }
        identities = {
            name: (0, index)
            for index, name in enumerate(
                (
                    "start",
                    "first",
                    "second",
                    "first-dead-1",
                    "first-dead-2",
                    "second-terminal",
                    "second-dead",
                )
            )
        }
        result = bounded_transition_reachability(
            "start",
            command=object(),
            command_change_frame=10,
            expand=lambda state, _command, _advance: graph.get(state, ()),
            is_safe=lambda _state, _command: True,
            is_terminal=lambda state, _command: state == "second-terminal",
            identity=lambda state: identities[state],
            limits=ReachabilityLimits(
                max_depth=2,
                beam_width=2,
                max_expanded_states=4,
                max_source_advance_frames=1,
            ),
        )
        self.assertTrue(result.reachable)
        self.assertEqual(result.expanded_state_count, 4)
        self.assertEqual(result.first_frame_index, identities["second"][1])
        self.assertEqual(
            result.terminal_frame_index,
            identities["second-terminal"][1],
        )

    def test_limits_cannot_exceed_frozen_diagnostic_budget(self):
        invalid = (
            {"max_depth": 0},
            {"max_depth": 3},
            {"beam_width": 0},
            {"beam_width": 9},
            {"max_expanded_states": 0},
            {"max_expanded_states": 65},
            {"max_source_advance_frames": 0},
            {"max_source_advance_frames": 16},
        )
        for override in invalid:
            with self.subTest(override=override):
                values = {
                    "max_depth": 2,
                    "beam_width": 8,
                    "max_expanded_states": 64,
                    "max_source_advance_frames": 15,
                    **override,
                }
                with self.assertRaises(ValueError):
                    ReachabilityLimits(**values)


if __name__ == "__main__":
    unittest.main()
