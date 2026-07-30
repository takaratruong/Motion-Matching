"""Pure same-stair omnidirectional command routes.

The stair-local frame uses +x for ascent and +y for the stair's left side.
Routes are immutable and renderer-independent.  They intentionally contain
commands only; a rollout owns placement of the robot at a route's reset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class StairFrame:
    origin_world_xy: tuple[float, float]
    ascent_world_yaw: float
    width_m: float
    tread_depth_m: float
    riser_height_m: float
    tread_count: int


@dataclass(frozen=True)
class RouteCommand:
    velocity_stair_xy: tuple[float, float]
    heading_stair_yaw: float
    frames: int
    segment: str
    reset_before: bool = False


@dataclass(frozen=True)
class WorldCommand:
    velocity_world_xy: tuple[float, float]
    heading_world_yaw: float
    frames: int
    segment: str
    reset_before: bool = False


@dataclass(frozen=True)
class OmniRoute:
    name: str
    commands: tuple[RouteCommand, ...]
    required_outcome: Literal["mount", "traverse", "turn", "exit", "mixed"]


# Compatibility names for the first frozen consumer test.
StairCommand = RouteCommand
Route = OmniRoute


def _route(
    name: str,
    outcome: Literal["mount", "traverse", "turn", "exit", "mixed"],
    specs: tuple[tuple[str, int, float, float, float], ...],
) -> OmniRoute:
    return OmniRoute(
        name=name,
        required_outcome=outcome,
        commands=tuple(
            RouteCommand(
                velocity_stair_xy=(vx, vy),
                heading_stair_yaw=heading,
                frames=frames,
                segment=segment,
                reset_before=index == 0,
            )
            for index, (segment, frames, vx, vy, heading) in enumerate(specs)
        ),
    )


def _mirror(name: str, source: OmniRoute) -> OmniRoute:
    return OmniRoute(
        name=name,
        required_outcome=source.required_outcome,
        commands=tuple(
            replace(
                command,
                velocity_stair_xy=(
                    command.velocity_stair_xy[0],
                    -command.velocity_stair_xy[1],
                ),
                heading_stair_yaw=-command.heading_stair_yaw,
            )
            for command in source.commands
        ),
    )


_SIDE_MOUNT_LEFT = _route(
    "side-mount-left",
    "mount",
    (
        ("approach", 40, 0.00, 0.30, math.pi / 2.0),
        ("mount", 45, 0.35, 0.15, math.pi / 6.0),
    ),
)
_CROSS_TREAD_LEFT = _route(
    "cross-tread-left-to-right",
    "traverse",
    (
        ("enter", 30, 0.10, 0.30, math.pi / 2.0),
        ("cross", 45, 0.15, 0.30, math.pi / 3.0),
        ("settle", 25, 0.05, 0.20, math.pi / 4.0),
    ),
)
_TURN_45_LEFT = _route(
    "turn-45-lower-left",
    "turn",
    (
        ("ascend", 30, 0.40, 0.00, 0.0),
        ("pivot", 25, 0.20, 0.15, math.pi / 4.0),
    ),
)
_TURN_90_LEFT = _route(
    "turn-90-middle-left",
    "turn",
    (
        ("ascend", 35, 0.40, 0.00, 0.0),
        ("pivot", 30, 0.10, 0.20, math.pi / 2.0),
    ),
)
_TURN_180_LEFT = _route(
    "turn-180-upper-left",
    "turn",
    (
        ("ascend", 40, 0.40, 0.00, 0.0),
        ("pivot", 35, -0.10, 0.15, math.pi),
    ),
)
_DIAGONAL_UP_LEFT = _route(
    "diagonal-up-left",
    "traverse",
    (
        ("diagonal", 40, 0.30, 0.25, math.pi / 5.0),
        ("continue", 35, 0.30, 0.20, math.pi / 6.0),
    ),
)
_DIAGONAL_DOWN_LEFT = _route(
    "diagonal-down-left",
    "traverse",
    (
        ("descend", 40, -0.30, 0.25, math.pi - math.pi / 5.0),
        ("continue", 35, -0.30, 0.20, math.pi - math.pi / 6.0),
    ),
)
_SIDE_EXIT_LOWER_LEFT = _route(
    "side-exit-lower-left",
    "exit",
    (
        ("descend", 30, -0.30, 0.00, math.pi),
        ("exit", 35, -0.10, 0.30, math.pi / 2.0),
    ),
)
_SIDE_EXIT_UPPER_LEFT = _route(
    "side-exit-upper-left",
    "exit",
    (
        ("ascend", 30, 0.30, 0.00, 0.0),
        ("exit", 35, 0.05, 0.30, math.pi / 2.0),
    ),
)
_RISER_STOP_RESTART = _route(
    "riser-stop-restart",
    "mixed",
    (
        ("ascend", 30, 0.40, 0.00, 0.0),
        ("stop", 20, 0.00, 0.00, 0.0),
        ("restart", 30, 0.40, 0.00, 0.0),
    ),
)
_RISER_REVERSAL = _route(
    "riser-reversal",
    "mixed",
    (
        ("ascend", 30, 0.40, 0.00, 0.0),
        ("reverse", 30, -0.40, 0.00, math.pi),
    ),
)
_MIXED_ADVERSARIAL = _route(
    "mixed-adversarial",
    "mixed",
    (
        ("mount", 25, 0.20, 0.30, math.pi / 3.0),
        ("turn", 25, 0.10, -0.20, -math.pi / 4.0),
        ("descend", 25, -0.30, 0.10, math.pi),
        ("exit", 25, 0.00, 0.35, math.pi / 2.0),
    ),
)

_ROUTES = (
    _SIDE_MOUNT_LEFT,
    _mirror("side-mount-right", _SIDE_MOUNT_LEFT),
    _CROSS_TREAD_LEFT,
    _mirror("cross-tread-right-to-left", _CROSS_TREAD_LEFT),
    _TURN_45_LEFT,
    _mirror("turn-45-lower-right", _TURN_45_LEFT),
    _TURN_90_LEFT,
    _mirror("turn-90-middle-right", _TURN_90_LEFT),
    _TURN_180_LEFT,
    _mirror("turn-180-upper-right", _TURN_180_LEFT),
    _DIAGONAL_UP_LEFT,
    _mirror("diagonal-up-right", _DIAGONAL_UP_LEFT),
    _DIAGONAL_DOWN_LEFT,
    _mirror("diagonal-down-right", _DIAGONAL_DOWN_LEFT),
    _SIDE_EXIT_LOWER_LEFT,
    _mirror("side-exit-lower-right", _SIDE_EXIT_LOWER_LEFT),
    _SIDE_EXIT_UPPER_LEFT,
    _mirror("side-exit-upper-right", _SIDE_EXIT_UPPER_LEFT),
    _RISER_STOP_RESTART,
    _RISER_REVERSAL,
    _MIXED_ADVERSARIAL,
)


def same_stair_routes() -> tuple[OmniRoute, ...]:
    return _ROUTES


def world_commands(
    frame: StairFrame, route: OmniRoute
) -> tuple[WorldCommand, ...]:
    cos_yaw = math.cos(frame.ascent_world_yaw)
    sin_yaw = math.sin(frame.ascent_world_yaw)
    return tuple(
        WorldCommand(
            velocity_world_xy=(
                command.velocity_stair_xy[0] * cos_yaw
                - command.velocity_stair_xy[1] * sin_yaw,
                command.velocity_stair_xy[0] * sin_yaw
                + command.velocity_stair_xy[1] * cos_yaw,
            ),
            heading_world_yaw=math.atan2(
                math.sin(command.heading_stair_yaw + frame.ascent_world_yaw),
                math.cos(command.heading_stair_yaw + frame.ascent_world_yaw),
            ),
            frames=command.frames,
            segment=command.segment,
            reset_before=command.reset_before,
        )
        for command in route.commands
    )
