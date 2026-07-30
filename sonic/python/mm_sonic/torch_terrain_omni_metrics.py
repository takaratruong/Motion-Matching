"""Pure metrics for same-stair kinematic motion-matching rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


DT_S = 0.02
ANKLE_ORIGIN_SOLE_M = 0.035
STANCE_CLEARANCE_TOLERANCE_M = 0.02
STANCE_VERTICAL_SPEED_MAX_MPS = 0.12
RESCUE_NEIGHBORHOOD_FRAMES = 8
RESCUE_CYCLE_PROGRESS_MIN_M = 0.05
MOVING_COMMAND_SPEED_MIN_MPS = 0.10
STALL_ROOT_SPEED_MAX_MPS = 0.03


@dataclass(frozen=True)
class Distribution:
    samples: tuple[float, ...]
    minimum: float
    maximum: float
    mean: float
    p95: float


@dataclass(frozen=True)
class PerFootDistance:
    per_foot: tuple[float, float]
    total: float


@dataclass(frozen=True)
class RescueEvent:
    clip_id: str
    source_frame: int
    root_xy: tuple[float, float]
    is_terrain_rescue: bool = True


@dataclass(frozen=True)
class RescueCycle:
    event_indices: tuple[int, int, int, int]
    clip_pair: tuple[str, str]
    root_progress_m: float


@dataclass(frozen=True)
class OmniRouteMetrics:
    stance_mask: np.ndarray
    foot_clearance_m: np.ndarray
    support_height_error_m: Distribution
    support_height_difference_m: Distribution
    penetration_depth_m: Distribution
    stance_slide_m: PerFootDistance
    heading_error_rad: Distribution
    root_progress_m: float
    root_jerk_m_s3: Distribution
    root_velocity_error_mps: Distribution
    stalled_moving_mask: np.ndarray
    stalled_moving_fraction: float
    longest_stall_frames: int
    transition_count: int
    rescue_cycles: tuple[RescueCycle, ...]
    required_outcome_completed: bool | None


def _distribution(values: np.ndarray | Sequence[float]) -> Distribution:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return Distribution((), 0.0, 0.0, 0.0, 0.0)
    if not np.isfinite(array).all():
        raise ValueError("metric samples must be finite")
    return Distribution(
        samples=tuple(float(value) for value in array),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        mean=float(np.mean(array)),
        p95=float(np.percentile(array, 95)),
    )


def detect_rescue_cycles(
    events: Sequence[RescueEvent],
) -> tuple[RescueCycle, ...]:
    """Detect low-progress alternating A-B-A-B terrain-rescue loops."""

    cycles: list[RescueCycle] = []
    index = 0
    while index + 3 < len(events):
        window = events[index : index + 4]
        first, second, third, fourth = window
        clip_pair = (first.clip_id, second.clip_id)
        alternating = (
            first.clip_id != second.clip_id
            and third.clip_id == first.clip_id
            and fourth.clip_id == second.clip_id
        )
        neighborhoods = (
            abs(third.source_frame - first.source_frame)
            <= RESCUE_NEIGHBORHOOD_FRAMES
            and abs(fourth.source_frame - second.source_frame)
            <= RESCUE_NEIGHBORHOOD_FRAMES
        )
        progress = math.dist(first.root_xy, fourth.root_xy)
        if (
            all(event.is_terrain_rescue for event in window)
            and alternating
            and neighborhoods
            and progress < RESCUE_CYCLE_PROGRESS_MIN_M
        ):
            cycles.append(
                RescueCycle(
                    event_indices=(index, index + 1, index + 2, index + 3),
                    clip_pair=clip_pair,
                    root_progress_m=progress,
                )
            )
            index += 4
        else:
            index += 1
    return tuple(cycles)


def _array(
    value: np.ndarray | Sequence[float],
    name: str,
    shape_tail: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape (frames, {shape_tail})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def evaluate_omni_route(
    *,
    root_xy: np.ndarray,
    root_yaw: np.ndarray,
    foot_position_world: np.ndarray,
    foot_surface_height_m: np.ndarray,
    command_velocity_world_xy: np.ndarray,
    command_heading_world_yaw: np.ndarray,
    selected_clip_id: Sequence[str] | None = None,
    selected_source_frame: Sequence[int] | None = None,
    rescue_events: Sequence[RescueEvent] = (),
    required_outcome_completed: bool | None = None,
) -> OmniRouteMetrics:
    """Evaluate one 50 Hz route from renderer-visible FK kinematics."""

    root_xy_array = _array(root_xy, "root_xy", (2,))
    feet = _array(foot_position_world, "foot_position_world", (2, 3))
    surface = _array(
        foot_surface_height_m, "foot_surface_height_m", (2,)
    )
    command_velocity = _array(
        command_velocity_world_xy, "command_velocity_world_xy", (2,)
    )
    yaw = np.asarray(root_yaw, dtype=np.float64)
    command_yaw = np.asarray(command_heading_world_yaw, dtype=np.float64)
    frame_count = root_xy_array.shape[0]
    expected_scalar_shape = (frame_count,)
    if yaw.shape != expected_scalar_shape or command_yaw.shape != expected_scalar_shape:
        raise ValueError("yaw arrays must have shape (frames,)")
    if not np.isfinite(yaw).all() or not np.isfinite(command_yaw).all():
        raise ValueError("yaw arrays must be finite")
    if (
        feet.shape[0] != frame_count
        or surface.shape[0] != frame_count
        or command_velocity.shape[0] != frame_count
    ):
        raise ValueError("all metric arrays must share a frame count")

    clearance = feet[:, :, 2] - surface
    vertical_speed = np.zeros_like(clearance)
    if frame_count > 1:
        vertical_speed[1:] = np.diff(feet[:, :, 2], axis=0) / DT_S
    support_error = np.abs(clearance - ANKLE_ORIGIN_SOLE_M)
    stance = (
        (support_error <= STANCE_CLEARANCE_TOLERANCE_M)
        & (np.abs(vertical_speed) <= STANCE_VERTICAL_SPEED_MAX_MPS)
    )

    slide_per_foot = np.zeros(2, dtype=np.float64)
    if frame_count > 1:
        horizontal_step = np.linalg.norm(
            np.diff(feet[:, :, :2], axis=0), axis=2
        )
        stance_intervals = stance[:-1] & stance[1:]
        slide_per_foot = np.sum(
            np.where(stance_intervals, horizontal_step, 0.0), axis=0
        )

    both_supported = np.all(stance, axis=1)
    support_height_difference = np.abs(surface[:, 0] - surface[:, 1])
    supported_errors = support_error[stance]
    penetration = np.maximum(surface - feet[:, :, 2], 0.0)

    heading_error = np.abs(
        np.arctan2(np.sin(yaw - command_yaw), np.cos(yaw - command_yaw))
    )
    root_progress = (
        float(np.linalg.norm(root_xy_array[-1] - root_xy_array[0]))
        if frame_count > 1
        else 0.0
    )
    if frame_count >= 4:
        root_jerk = np.linalg.norm(
            np.diff(root_xy_array, n=3, axis=0) / (DT_S**3), axis=1
        )
    else:
        root_jerk = np.empty(0, dtype=np.float64)

    root_velocity = np.zeros_like(root_xy_array)
    if frame_count > 1:
        root_velocity[1:] = np.diff(root_xy_array, axis=0) / DT_S
    moving_command = (
        np.linalg.norm(command_velocity, axis=1)
        >= MOVING_COMMAND_SPEED_MIN_MPS
    )
    if frame_count:
        moving_command[0] = False
    stalled = moving_command & (
        np.linalg.norm(root_velocity, axis=1) <= STALL_ROOT_SPEED_MAX_MPS
    )
    moving_count = int(np.sum(moving_command))
    longest_stall = 0
    current_stall = 0
    for value in stalled:
        current_stall = current_stall + 1 if value else 0
        longest_stall = max(longest_stall, current_stall)
    velocity_error = np.linalg.norm(
        root_velocity - command_velocity, axis=1
    )

    transition_count = 0
    if selected_clip_id is not None or selected_source_frame is not None:
        if selected_clip_id is None or selected_source_frame is None:
            raise ValueError("selection clip and frame must be supplied together")
        if len(selected_clip_id) != frame_count or len(selected_source_frame) != frame_count:
            raise ValueError("selection arrays must share the frame count")
        transition_count = sum(
            current_clip != previous_clip
            or int(current_frame) != int(previous_frame) + 1
            for previous_clip, current_clip, previous_frame, current_frame in zip(
                selected_clip_id,
                selected_clip_id[1:],
                selected_source_frame,
                selected_source_frame[1:],
            )
        )

    return OmniRouteMetrics(
        stance_mask=stance,
        foot_clearance_m=clearance,
        support_height_error_m=_distribution(supported_errors),
        support_height_difference_m=_distribution(
            support_height_difference[both_supported]
        ),
        penetration_depth_m=_distribution(penetration),
        stance_slide_m=PerFootDistance(
            per_foot=(float(slide_per_foot[0]), float(slide_per_foot[1])),
            total=float(np.sum(slide_per_foot)),
        ),
        heading_error_rad=_distribution(heading_error),
        root_progress_m=root_progress,
        root_jerk_m_s3=_distribution(root_jerk),
        root_velocity_error_mps=_distribution(
            velocity_error[moving_command]
        ),
        stalled_moving_mask=stalled,
        stalled_moving_fraction=(
            float(np.sum(stalled)) / moving_count if moving_count else 0.0
        ),
        longest_stall_frames=longest_stall,
        transition_count=transition_count,
        rescue_cycles=detect_rescue_cycles(rescue_events),
        required_outcome_completed=required_outcome_completed,
    )
