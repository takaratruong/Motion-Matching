from types import SimpleNamespace
import unittest

import torch

from mm_sonic.torch_terrain_omni_routes import (
    OmniRoute,
    RouteCommand,
    StairFrame,
)
from mm_sonic.torch_terrain_reachability_rollout import (
    run_route_reachability,
)
from mm_sonic.torch_transition_reachability import (
    TransitionReachabilityDiagnostic,
)


class _FakeMatcher:
    def __init__(self, *, mismatch_field=None):
        self.frame = 0
        self.diagnostic_frames = []
        self.mismatch_field = mismatch_field

    def reset(self):
        self.frame = 0

    def diagnose_transition_reachability(
        self,
        _velocity,
        _heading,
        *,
        command_change_frame,
        terminal_evaluator,
        safe_evaluator,
        limits,
    ):
        self.diagnostic_frames.append(command_change_frame)
        self.asserted_terminal = terminal_evaluator
        self.asserted_safe = safe_evaluator
        self.asserted_limits = limits
        return TransitionReachabilityDiagnostic(
            command_change_frame=command_change_frame,
            planner_compute_ns=10,
            expanded_state_count=3,
            depth_used=2,
            reachable=True,
            first_clip_index=1,
            first_frame_index=20,
            terminal_clip_index=2,
            terminal_frame_index=40,
            budget_exhausted=False,
        )

    def step(self, _velocity, _heading, *, dt):
        self.frame += 1
        value = torch.full((2, 2), float(self.frame))
        tensors = {
            name: value.clone()
            for name in (
                "joint_position",
                "joint_velocity",
                "root_position_world",
                "root_orientation_world_wxyz",
                "dense_joint_position_window",
                "dense_joint_velocity_window",
                "dense_root_position_window",
                "dense_root_orientation_window_wxyz",
                "dense_feature_body_position_window",
                "dense_feature_body_velocity_window",
                "joint_position_window",
                "joint_velocity_window",
                "root_position_window",
                "root_orientation_window_wxyz",
            )
        }
        if self.mismatch_field is not None:
            tensors[self.mismatch_field][0, 0] += 1.0
        return SimpleNamespace(
            diagnostics=SimpleNamespace(
                sequence=self.frame,
                selected_clip_path="clip",
                selected_frame=self.frame,
                incumbent_cost=1.0,
                selected_feature_cost=2.0,
                motion_feature_cost=1.0,
                extension_feature_cost=1.0,
                selected_total_cost=2.0,
                selected_transition_position_cost=0.0,
                selected_transition_velocity_cost=0.0,
                selected_transition_continuity_cost=0.0,
                searched=True,
                transitioned=False,
                transition_rejected=False,
                terrain_safety_override=False,
                terrain_safety_override_rank=-1,
                force_search_reason=None,
            ),
            **tensors,
        )


class TorchTerrainReachabilityRolloutTests(unittest.TestCase):
    def test_route_diagnostic_runs_at_exit_boundary_and_is_read_only(self):
        route = OmniRoute(
            name="fixture-exit",
            required_outcome="exit",
            commands=(
                RouteCommand(
                    (0.4, 0.0), 0.0, 3, "approach", reset_before=True
                ),
                RouteCommand((0.0, 0.4), 0.0, 2, "exit-left"),
                RouteCommand((0.0, 0.4), 0.0, 1, "exit-left"),
            ),
        )
        stair = StairFrame((0.0, 0.0), 0.0, 0.6, 0.3, 0.18, 3)
        matcher = _FakeMatcher()
        control = _FakeMatcher()
        terminal = object()
        safe = object()

        result = run_route_reachability(
            route,
            stair_frame=stair,
            matcher=matcher,
            control_matcher=control,
            terminal_evaluator=terminal,
            safe_evaluator=safe,
        )

        self.assertTrue(result.behavior_unchanged)
        self.assertEqual(result.completed_frames, 6)
        self.assertEqual(matcher.diagnostic_frames, [3])
        self.assertEqual(control.diagnostic_frames, [])
        self.assertEqual(len(result.diagnostics), 1)
        self.assertEqual(result.diagnostics[0].command_change_frame, 3)
        self.assertIs(matcher.asserted_terminal, terminal)
        self.assertIs(matcher.asserted_safe, safe)
        self.assertEqual(
            result.observed_output_sha256,
            result.control_output_sha256,
        )

    def test_route_comparator_covers_all_result_tensors(self):
        route = OmniRoute(
            name="fixture-exit",
            required_outcome="exit",
            commands=(
                RouteCommand((0.0, 0.4), 0.0, 1, "exit-left"),
            ),
        )
        stair = StairFrame((0.0, 0.0), 0.0, 0.6, 0.3, 0.18, 3)
        result = run_route_reachability(
            route,
            stair_frame=stair,
            matcher=_FakeMatcher(mismatch_field="joint_velocity_window"),
            control_matcher=_FakeMatcher(),
            terminal_evaluator=object(),
            safe_evaluator=object(),
        )
        self.assertFalse(result.behavior_unchanged)
        self.assertNotEqual(
            result.observed_output_sha256,
            result.control_output_sha256,
        )


if __name__ == "__main__":
    unittest.main()
