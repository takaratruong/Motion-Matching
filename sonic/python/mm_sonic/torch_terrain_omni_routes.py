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
        ("move-outside", 60, 0.00, 0.38, 0.0),
        ("approach-alongside", 180, 0.38, 0.00, 0.0),
        ("mount-from-left", 60, 0.00, -0.38, 0.0),
        ("continue-up", 70, 0.38, 0.00, 0.0),
    ),
)
_CROSS_TREAD_LEFT = _route(
    "cross-tread-left-to-right",
    "traverse",
    (
        ("move-outside", 60, 0.00, 0.38, 0.0),
        ("approach-lower-tread", 200, 0.38, 0.00, 0.0),
        ("cross-left-to-right", 120, 0.00, -0.38, 0.0),
        ("settle", 30, 0.15, 0.00, 0.0),
    ),
)
_TURN_45_LEFT = _route(
    "turn-45-lower-left",
    "turn",
    (
        ("approach-lower", 190, 0.38, 0.00, 0.0),
        ("pivot-45", 50, 0.05, 0.05, math.pi / 4.0),
    ),
)
_TURN_90_LEFT = _route(
    "turn-90-middle-left",
    "turn",
    (
        ("ascend-middle", 225, 0.38, 0.00, 0.0),
        ("pivot-90", 70, 0.00, 0.08, math.pi / 2.0),
    ),
)
_TURN_180_LEFT = _route(
    "turn-180-upper-left",
    "turn",
    (
        ("ascend-upper", 260, 0.38, 0.00, 0.0),
        ("pivot-180", 100, -0.05, 0.05, math.pi),
    ),
)
_DIAGONAL_UP_LEFT = _route(
    "diagonal-up-left",
    "traverse",
    (
        ("approach-riser", 165, 0.38, 0.00, 0.0),
        ("diagonal-up", 120, 0.30, 0.18, math.atan2(0.18, 0.30)),
        ("settle", 30, 0.15, 0.00, 0.0),
    ),
)
_DIAGONAL_DOWN_LEFT = _route(
    "diagonal-down-left",
    "traverse",
    (
        ("ascend-upper", 270, 0.38, 0.00, 0.0),
        ("diagonal-down", 160, -0.30, 0.15, 0.0),
        ("settle", 30, -0.15, 0.00, 0.0),
    ),
)
_SIDE_EXIT_LOWER_LEFT = _route(
    "side-exit-lower-left",
    "exit",
    (
        ("approach-lower", 195, 0.38, 0.00, 0.0),
        ("exit-left", 70, 0.00, 0.38, 0.0),
        ("continue-off", 30, 0.15, 0.00, 0.0),
    ),
)
_SIDE_EXIT_UPPER_LEFT = _route(
    "side-exit-upper-left",
    "exit",
    (
        ("ascend-upper", 265, 0.38, 0.00, 0.0),
        ("exit-left", 75, 0.00, 0.38, 0.0),
        ("continue-off", 30, 0.15, 0.00, 0.0),
    ),
)
_RISER_STOP_RESTART = _route(
    "riser-stop-restart",
    "mixed",
    (
        ("approach-riser", 160, 0.38, 0.00, 0.0),
        ("stop-at-riser", 30, 0.00, 0.00, 0.0),
        ("restart-up", 100, 0.38, 0.00, 0.0),
    ),
)
_RISER_REVERSAL = _route(
    "riser-reversal",
    "mixed",
    (
        ("ascend-middle", 230, 0.38, 0.00, 0.0),
        ("reverse-down", 160, -0.38, 0.00, 0.0),
    ),
)
_MIXED_ADVERSARIAL = _route(
    "mixed-adversarial",
    "mixed",
    (
        ("approach", 180, 0.38, 0.00, 0.0),
        ("diagonal-mount", 80, 0.28, 0.18, math.atan2(0.18, 0.28)),
        ("turn-across", 70, 0.05, -0.20, -math.pi / 2.0),
        ("reverse-down", 100, -0.30, 0.10, 0.0),
        ("side-exit", 70, 0.00, 0.38, 0.0),
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
