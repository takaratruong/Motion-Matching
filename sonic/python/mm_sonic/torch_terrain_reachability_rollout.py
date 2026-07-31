"""Read-only transition reachability replays on frozen terrain routes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import torch

from .torch_terrain_omni_routes import OmniRoute, StairFrame, world_commands
from .torch_transition_reachability import (
    ReachabilityLimits,
    TransitionReachabilityDiagnostic,
)


_RESULT_TENSORS = (
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
_STABLE_DIAGNOSTICS = (
    "sequence",
    "selected_clip_path",
    "selected_frame",
    "incumbent_cost",
    "selected_feature_cost",
    "motion_feature_cost",
    "extension_feature_cost",
    "selected_total_cost",
    "selected_transition_position_cost",
    "selected_transition_velocity_cost",
    "selected_transition_continuity_cost",
    "searched",
    "transitioned",
    "transition_rejected",
    "terrain_safety_override",
    "terrain_safety_override_rank",
    "force_search_reason",
)


@dataclass(frozen=True)
class RouteReachabilityRun:
    route_name: str
    completed_frames: int
    behavior_unchanged: bool
    observed_output_sha256: str
    control_output_sha256: str
    diagnostics: tuple[TransitionReachabilityDiagnostic, ...]


def _stable_diagnostics(result) -> tuple:
    return tuple(
        getattr(result.diagnostics, name) for name in _STABLE_DIAGNOSTICS
    )


def _same_result(left, right) -> bool:
    if _stable_diagnostics(left) != _stable_diagnostics(right):
        return False
    return all(
        torch.equal(getattr(left, name), getattr(right, name))
        for name in _RESULT_TENSORS
    )


def _update_output_hash(digest, result) -> None:
    digest.update(repr(_stable_diagnostics(result)).encode("utf-8"))
    for name in _RESULT_TENSORS:
        tensor = getattr(result, name)
        digest.update(name.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(
            tensor.detach().cpu().contiguous().numpy().tobytes(order="C")
        )


def run_route_reachability(
    route: OmniRoute,
    *,
    stair_frame: StairFrame,
    matcher,
    control_matcher,
    terminal_evaluator,
    safe_evaluator,
    diagnostic_segments: Sequence[str] = ("exit-left", "side-exit"),
    limits: ReachabilityLimits = ReachabilityLimits(),
) -> RouteReachabilityRun:
    """Replay one route beside a control and diagnose only exit boundaries."""

    if not isinstance(route, OmniRoute):
        raise TypeError("route must be OmniRoute")
    if not isinstance(stair_frame, StairFrame):
        raise TypeError("stair_frame must be StairFrame")
    segments = tuple(diagnostic_segments)
    if (
        not segments
        or any(not isinstance(segment, str) or not segment for segment in segments)
    ):
        raise ValueError("diagnostic_segments must contain non-empty strings")

    matcher.reset()
    control_matcher.reset()
    diagnostics = []
    behavior_unchanged = True
    completed_frames = 0
    previous_issued_command = None
    observed_digest = hashlib.sha256()
    control_digest = hashlib.sha256()
    for command in world_commands(stair_frame, route):
        issued_command = (
            command.velocity_world_xy,
            command.heading_world_yaw,
        )
        command_changed = issued_command != previous_issued_command
        if command.segment in segments and command_changed:
            diagnostics.append(
                matcher.diagnose_transition_reachability(
                    command.velocity_world_xy,
                    command.heading_world_yaw,
                    command_change_frame=completed_frames,
                    terminal_evaluator=terminal_evaluator,
                    safe_evaluator=safe_evaluator,
                    limits=limits,
                )
            )
        previous_issued_command = issued_command
        for _ in range(command.frames):
            observed = matcher.step(
                command.velocity_world_xy,
                command.heading_world_yaw,
                dt=0.02,
            )
            control = control_matcher.step(
                command.velocity_world_xy,
                command.heading_world_yaw,
                dt=0.02,
            )
            behavior_unchanged &= _same_result(observed, control)
            _update_output_hash(observed_digest, observed)
            _update_output_hash(control_digest, control)
            completed_frames += 1
    return RouteReachabilityRun(
        route_name=route.name,
        completed_frames=completed_frames,
        behavior_unchanged=behavior_unchanged,
        observed_output_sha256=observed_digest.hexdigest(),
        control_output_sha256=control_digest.hexdigest(),
        diagnostics=tuple(diagnostics),
    )
