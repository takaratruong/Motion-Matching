"""Contact-phase-preserving terrain stop, restart, and reversal pilots.

The global terrain controller already owns a small bank of complete pose paths
that passed exact G1/mesh collision checks.  This module changes *when* those
paths are traversed without changing where a planted foot is placed.  A
quintic velocity envelope slows an authored path to a measured double-support
pose, optionally holds it, and then either resumes or traverses the same pose
path in reverse.

This is deliberately an offline kinematic composer.  It produces clean motion
and command labels for pilot evaluation; it does not read odometry or robot
state at deployment time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .terrain_oracle.math3d import slerp_wxyz
from .terrain_oracle.stitch import FrameProvenance, StitchedMotion


NATIVE = np.uint8(0)
DECELERATE = np.uint8(1)
HOLD = np.uint8(2)
ACCELERATE_FORWARD = np.uint8(3)
NATIVE_FORWARD = np.uint8(4)
ACCELERATE_REVERSE = np.uint8(5)
NATIVE_REVERSE = np.uint8(6)

PHASE_NAMES = (
    "native_approach",
    "decelerate",
    "hold",
    "accelerate_forward",
    "native_forward",
    "accelerate_reverse",
    "native_reverse",
)


@dataclass(frozen=True)
class ManeuverSchedule:
    """Fractional source-frame coordinates and their semantic phases."""

    source_coordinate: np.ndarray
    phase: np.ndarray
    pivot_source_frame: int
    mode: str
    pivot_source_frames: tuple[int, ...] = ()


@dataclass(frozen=True)
class SupportPivot:
    """A low-motion, measured double-support frame inside terrain."""

    frame_index: int
    left_clearance_m: float
    right_clearance_m: float
    root_speed_mps: float
    joint_speed_rms_rad_s: float
    terrain_progress: float


def _quintic_smootherstep(value: object) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _velocity_ramp(frame_count: int, *, accelerating: bool) -> np.ndarray:
    """Return N source-frame increments with zero endpoint acceleration."""

    count = int(frame_count)
    if count < 4:
        raise ValueError("maneuver ramp must contain at least four frames")
    phase = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)[1:]
    speed = _quintic_smootherstep(phase)
    return speed if accelerating else 1.0 - speed


def build_maneuver_schedule(
    frame_count: int,
    *,
    pivot_source_frame: int,
    mode: str,
    ramp_frames: int = 24,
    hold_frames: int = 30,
) -> ManeuverSchedule:
    """Build a native-rate stop/restart or stop/reverse source-time path.

    Source coordinates advance by one per output frame outside the maneuver.
    Around ``pivot_source_frame`` they use integrated quintic velocity ramps,
    so velocity and acceleration approach zero before the held support pose.
    """

    count = int(frame_count)
    pivot = int(pivot_source_frame)
    ramp = int(ramp_frames)
    hold = int(hold_frames)
    if count < 8:
        raise ValueError("source motion is too short")
    if mode not in ("stop_restart", "reverse"):
        raise ValueError("mode must be 'stop_restart' or 'reverse'")
    if hold < 1:
        raise ValueError("maneuver hold must contain at least one frame")

    deceleration = _velocity_ramp(ramp, accelerating=False)
    acceleration = _velocity_ramp(ramp, accelerating=True)
    slow_span = float(np.sum(deceleration))
    restart_span = float(np.sum(acceleration))
    start_coordinate = float(pivot) - slow_span
    if start_coordinate < 1.0 or pivot + restart_span > count - 2:
        raise ValueError("pivot has insufficient source context for ramp")

    native_prefix = np.arange(
        0.0, math.floor(start_coordinate) + 1.0, dtype=np.float64
    )
    if native_prefix[-1] < start_coordinate - 1.0e-9:
        native_prefix = np.append(native_prefix, start_coordinate)
    else:
        native_prefix[-1] = start_coordinate

    deceleration_coordinates = start_coordinate + np.cumsum(deceleration)
    # Floating-point summation should not make the hold pose slightly mobile.
    deceleration_coordinates[-1] = float(pivot)
    hold_coordinates = np.full(hold, float(pivot), dtype=np.float64)

    if mode == "stop_restart":
        acceleration_coordinates = float(pivot) + np.cumsum(acceleration)
        native_start = float(acceleration_coordinates[-1])
        suffix = np.arange(
            math.floor(native_start) + 1.0,
            float(count),
            dtype=np.float64,
        )
        coordinates = np.concatenate(
            (
                native_prefix,
                deceleration_coordinates,
                hold_coordinates,
                acceleration_coordinates,
                suffix,
            )
        )
        phases = np.concatenate(
            (
                np.full(len(native_prefix), NATIVE, dtype=np.uint8),
                np.full(ramp, DECELERATE, dtype=np.uint8),
                np.full(hold, HOLD, dtype=np.uint8),
                np.full(ramp, ACCELERATE_FORWARD, dtype=np.uint8),
                np.full(len(suffix), NATIVE_FORWARD, dtype=np.uint8),
            )
        )
    else:
        acceleration_coordinates = float(pivot) - np.cumsum(acceleration)
        native_start = float(acceleration_coordinates[-1])
        suffix = np.arange(
            math.ceil(native_start) - 1.0,
            -1.0,
            -1.0,
            dtype=np.float64,
        )
        coordinates = np.concatenate(
            (
                native_prefix,
                deceleration_coordinates,
                hold_coordinates,
                acceleration_coordinates,
                suffix,
            )
        )
        phases = np.concatenate(
            (
                np.full(len(native_prefix), NATIVE, dtype=np.uint8),
                np.full(ramp, DECELERATE, dtype=np.uint8),
                np.full(hold, HOLD, dtype=np.uint8),
                np.full(ramp, ACCELERATE_REVERSE, dtype=np.uint8),
                np.full(len(suffix), NATIVE_REVERSE, dtype=np.uint8),
            )
        )

    coordinates = np.clip(coordinates, 0.0, float(count - 1))
    if len(coordinates) != len(phases) or not np.isfinite(coordinates).all():
        raise RuntimeError("invalid maneuver schedule")
    return ManeuverSchedule(
        source_coordinate=np.asarray(coordinates, dtype=np.float64),
        phase=phases,
        pivot_source_frame=pivot,
        mode=mode,
        pivot_source_frames=(pivot,),
    )


def build_bounce_schedule(
    frame_count: int,
    *,
    lower_pivot_source_frame: int,
    upper_pivot_source_frame: int,
    ramp_frames: int = 16,
    hold_frames: int = 4,
) -> ManeuverSchedule:
    """Traverse forward, retreat on terrain, then resume forward.

    Both direction changes occur at measured double-support source poses.  As
    with :func:`build_maneuver_schedule`, this only changes time along an
    already grounded path: it cannot introduce foot sliding or terrain
    penetration that was absent from the source motion.
    """

    count = int(frame_count)
    lower = int(lower_pivot_source_frame)
    upper = int(upper_pivot_source_frame)
    ramp = int(ramp_frames)
    hold = int(hold_frames)
    if count < 8 or not 0 < lower < upper < count - 1:
        raise ValueError("bounce pivots must be ordered inside the source")
    if hold < 1:
        raise ValueError("maneuver hold must contain at least one frame")

    deceleration = _velocity_ramp(ramp, accelerating=False)
    acceleration = _velocity_ramp(ramp, accelerating=True)
    slow_span = float(np.sum(deceleration))
    restart_span = float(np.sum(acceleration))
    first_deceleration_start = float(upper) - slow_span
    reverse_deceleration_start = float(lower) + slow_span
    reverse_native_start = float(upper) - restart_span
    if first_deceleration_start < 1.0:
        raise ValueError("upper bounce pivot has insufficient approach context")
    if reverse_native_start <= reverse_deceleration_start + 1.0:
        raise ValueError("bounce pivots are too close for smooth reversals")
    if float(lower) + restart_span > count - 2:
        raise ValueError("lower bounce pivot has insufficient restart context")

    native_prefix = np.arange(
        0.0, math.floor(first_deceleration_start) + 1.0, dtype=np.float64
    )
    if native_prefix[-1] < first_deceleration_start - 1.0e-9:
        native_prefix = np.append(native_prefix, first_deceleration_start)
    else:
        native_prefix[-1] = first_deceleration_start

    decelerate_forward = first_deceleration_start + np.cumsum(deceleration)
    decelerate_forward[-1] = float(upper)
    hold_upper = np.full(hold, float(upper), dtype=np.float64)
    accelerate_reverse = float(upper) - np.cumsum(acceleration)

    reverse_native = np.arange(
        math.ceil(float(accelerate_reverse[-1])) - 1.0,
        math.floor(reverse_deceleration_start),
        -1.0,
        dtype=np.float64,
    )
    if (
        len(reverse_native) == 0
        or reverse_native[-1] > reverse_deceleration_start + 1.0e-9
    ):
        reverse_native = np.append(reverse_native, reverse_deceleration_start)
    else:
        reverse_native[-1] = reverse_deceleration_start

    decelerate_reverse = reverse_deceleration_start - np.cumsum(deceleration)
    decelerate_reverse[-1] = float(lower)
    hold_lower = np.full(hold, float(lower), dtype=np.float64)
    accelerate_forward = float(lower) + np.cumsum(acceleration)
    native_forward_start = float(accelerate_forward[-1])
    native_suffix = np.arange(
        math.floor(native_forward_start) + 1.0,
        float(count),
        dtype=np.float64,
    )

    coordinates = np.concatenate(
        (
            native_prefix,
            decelerate_forward,
            hold_upper,
            accelerate_reverse,
            reverse_native,
            decelerate_reverse,
            hold_lower,
            accelerate_forward,
            native_suffix,
        )
    )
    phases = np.concatenate(
        (
            np.full(len(native_prefix), NATIVE, dtype=np.uint8),
            np.full(ramp, DECELERATE, dtype=np.uint8),
            np.full(hold, HOLD, dtype=np.uint8),
            np.full(ramp, ACCELERATE_REVERSE, dtype=np.uint8),
            np.full(len(reverse_native), NATIVE_REVERSE, dtype=np.uint8),
            np.full(ramp, DECELERATE, dtype=np.uint8),
            np.full(hold, HOLD, dtype=np.uint8),
            np.full(ramp, ACCELERATE_FORWARD, dtype=np.uint8),
            np.full(len(native_suffix), NATIVE_FORWARD, dtype=np.uint8),
        )
    )
    coordinates = np.clip(coordinates, 0.0, float(count - 1))
    if len(coordinates) != len(phases) or not np.isfinite(coordinates).all():
        raise RuntimeError("invalid bounce schedule")
    if float(np.max(np.abs(np.diff(coordinates)))) > 1.000001:
        raise RuntimeError("bounce schedule exceeds native playback speed")
    return ManeuverSchedule(
        source_coordinate=np.asarray(coordinates, dtype=np.float64),
        phase=phases,
        pivot_source_frame=upper,
        mode="bounce",
        pivot_source_frames=(upper, lower),
    )


def choose_support_pivot(
    motion: StitchedMotion,
    per_foot_clearance_m: object,
    *,
    per_foot_speed_mps: object | None = None,
    terrain_start_frame: int,
    terrain_stop_frame: int,
    maximum_support_clearance_m: float = 0.008,
    maximum_support_speed_mps: float = 0.12,
    central_fraction: tuple[float, float] = (0.20, 0.80),
    minimum_support_run_frames: int = 1,
) -> SupportPivot:
    """Select a central double-support pose rather than an arbitrary frame."""

    clearance = np.asarray(per_foot_clearance_m, dtype=np.float64)
    count = len(motion.root_position_world)
    if clearance.shape != (count, 2) or not np.isfinite(clearance).all():
        raise ValueError("per-foot clearance must be finite [T,2]")
    speed = (
        None
        if per_foot_speed_mps is None
        else np.asarray(per_foot_speed_mps, dtype=np.float64)
    )
    if speed is not None and (
        speed.shape != (count, 2) or not np.isfinite(speed).all()
    ):
        raise ValueError("per-foot speed must be finite [T,2]")
    if (
        not math.isfinite(float(maximum_support_speed_mps))
        or float(maximum_support_speed_mps) <= 0.0
    ):
        raise ValueError("maximum support speed must be positive and finite")
    start = int(terrain_start_frame)
    stop = int(terrain_stop_frame)
    lower_fraction, upper_fraction = map(float, central_fraction)
    if (
        not 0 <= start < stop <= count
        or not 0.0 <= lower_fraction < upper_fraction <= 1.0
        or int(minimum_support_run_frames) < 1
    ):
        raise ValueError("invalid terrain pivot bounds")

    root = np.asarray(motion.root_position_world, dtype=np.float64)
    joints = np.asarray(motion.joint_position, dtype=np.float64)
    root_speed = np.linalg.norm(np.gradient(root, axis=0) * motion.fps, axis=1)
    joint_speed = np.sqrt(
        np.mean(np.square(np.gradient(joints, axis=0) * motion.fps), axis=1)
    )
    span = stop - start
    central_start = start + int(math.floor(lower_fraction * span))
    central_stop = start + int(math.ceil(upper_fraction * span))
    supported = np.max(clearance, axis=1) <= float(maximum_support_clearance_m)
    if speed is not None:
        supported &= np.max(speed, axis=1) <= float(maximum_support_speed_mps)

    candidates: list[int] = []
    index = central_start
    while index < central_stop:
        if not supported[index]:
            index += 1
            continue
        run_start = index
        while index < central_stop and supported[index]:
            index += 1
        run_stop = index
        if run_stop - run_start >= int(minimum_support_run_frames):
            # Avoid threshold-crossing endpoints; the interior is the most
            # robust static pose under renderer and tracker tolerances.
            if run_stop - run_start >= 3:
                candidates.extend(range(run_start + 1, run_stop - 1))
            else:
                candidates.extend(range(run_start, run_stop))

    if not candidates:
        raise ValueError("terrain span has no central double-support window")
    candidate = np.asarray(candidates, dtype=np.int64)
    progress = (candidate - start) / float(span)
    score = (
        root_speed[candidate]
        + 0.05 * joint_speed[candidate]
        + 2.0 * np.max(clearance[candidate], axis=1)
        + 0.20 * np.abs(progress - 0.5)
    )
    selected = int(candidate[int(np.argmin(score))])
    return SupportPivot(
        frame_index=selected,
        left_clearance_m=float(clearance[selected, 0]),
        right_clearance_m=float(clearance[selected, 1]),
        root_speed_mps=float(root_speed[selected]),
        joint_speed_rms_rad_s=float(joint_speed[selected]),
        terrain_progress=float((selected - start) / float(span)),
    )


def choose_support_pivots(
    motion: StitchedMotion,
    per_foot_clearance_m: object,
    *,
    count: int,
    per_foot_speed_mps: object | None = None,
    terrain_start_frame: int,
    terrain_stop_frame: int,
    maximum_support_clearance_m: float = 0.008,
    maximum_support_speed_mps: float = 0.12,
    central_fraction: tuple[float, float] = (0.15, 0.85),
    minimum_support_run_frames: int = 1,
) -> tuple[SupportPivot, ...]:
    """Choose distinct early/middle/late support poses on one traversal.

    A single globally best pivot biases every generated maneuver toward the
    same stair.  Splitting the admissible terrain interval into equal progress
    bands produces behavior at several heights while preserving the exact
    support requirements of :func:`choose_support_pivot`.  Empty bands are
    skipped; callers can still use the pivots that are genuinely supported.
    """

    requested = int(count)
    if requested < 1:
        raise ValueError("support pivot count must be positive")
    if requested == 1:
        return (
            choose_support_pivot(
                motion,
                per_foot_clearance_m,
                per_foot_speed_mps=per_foot_speed_mps,
                terrain_start_frame=terrain_start_frame,
                terrain_stop_frame=terrain_stop_frame,
                maximum_support_clearance_m=maximum_support_clearance_m,
                maximum_support_speed_mps=maximum_support_speed_mps,
                central_fraction=central_fraction,
                minimum_support_run_frames=minimum_support_run_frames,
            ),
        )

    lower, upper = map(float, central_fraction)
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("invalid terrain pivot fractions")
    edges = np.linspace(lower, upper, requested + 1, dtype=np.float64)
    pivots: list[SupportPivot] = []
    for band_lower, band_upper in zip(edges[:-1], edges[1:], strict=True):
        try:
            pivot = choose_support_pivot(
                motion,
                per_foot_clearance_m,
                per_foot_speed_mps=per_foot_speed_mps,
                terrain_start_frame=terrain_start_frame,
                terrain_stop_frame=terrain_stop_frame,
                maximum_support_clearance_m=maximum_support_clearance_m,
                maximum_support_speed_mps=maximum_support_speed_mps,
                central_fraction=(float(band_lower), float(band_upper)),
                minimum_support_run_frames=minimum_support_run_frames,
            )
        except ValueError:
            continue
        if all(pivot.frame_index != value.frame_index for value in pivots):
            pivots.append(pivot)
    if not pivots:
        raise ValueError("terrain span has no supported pivot in any band")
    return tuple(sorted(pivots, key=lambda value: value.frame_index))


def resample_stitched_motion(
    motion: StitchedMotion,
    schedule: ManeuverSchedule,
) -> StitchedMotion:
    """Sample root, orientation, joints, and provenance at fractional frames."""

    coordinate = np.asarray(schedule.source_coordinate, dtype=np.float64)
    count = len(motion.root_position_world)
    if (
        coordinate.ndim != 1
        or len(coordinate) < 2
        or np.min(coordinate) < 0.0
        or np.max(coordinate) > count - 1
    ):
        raise ValueError("source coordinates are outside the motion")
    lower = np.floor(coordinate).astype(np.int64)
    upper = np.minimum(lower + 1, count - 1)
    fraction = coordinate - lower
    root = (
        (1.0 - fraction[:, None])
        * np.asarray(motion.root_position_world[lower], dtype=np.float64)
        + fraction[:, None]
        * np.asarray(motion.root_position_world[upper], dtype=np.float64)
    )
    joints = (
        (1.0 - fraction[:, None])
        * np.asarray(motion.joint_position[lower], dtype=np.float64)
        + fraction[:, None]
        * np.asarray(motion.joint_position[upper], dtype=np.float64)
    )
    quaternion = slerp_wxyz(
        motion.root_quaternion_world_wxyz[lower],
        motion.root_quaternion_world_wxyz[upper],
        fraction,
    )
    nearest = np.clip(np.rint(coordinate).astype(np.int64), 0, count - 1)
    provenance = tuple(motion.provenance[int(index)] for index in nearest)

    seam_indices: list[int] = []
    for seam in motion.seam_indices:
        distance = np.abs(coordinate - float(seam))
        seam_indices.append(int(np.argmin(distance)))
    return StitchedMotion(
        fps=float(motion.fps),
        root_position_world=np.asarray(root, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(quaternion, dtype=np.float32),
        joint_position=np.asarray(joints, dtype=np.float32),
        provenance=provenance,
        seam_indices=tuple(sorted(set(seam_indices))),
    )


def command_labels(
    motion: StitchedMotion,
    schedule: ManeuverSchedule,
) -> dict[str, np.ndarray]:
    """Derive synchronized two-stick labels from the clean output path."""

    root = np.asarray(motion.root_position_world, dtype=np.float64)
    velocity = np.gradient(root[:, :2], axis=0) * float(motion.fps)
    quaternion = np.asarray(motion.root_quaternion_world_wxyz, dtype=np.float64)
    w, x, y, z = quaternion.T
    facing = np.unwrap(
        np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )
    hold = np.asarray(schedule.phase == HOLD, dtype=np.bool_)
    # ``np.gradient`` intentionally looks across a phase boundary.  The raw
    # joystick label, however, is exactly centred throughout the requested
    # hold, including its first and last held samples.
    velocity[hold] = 0.0
    return {
        "command_velocity_world_xy": np.asarray(velocity, dtype=np.float32),
        "command_facing_yaw_world_rad": np.asarray(facing, dtype=np.float32),
        "command_stop": hold,
        "maneuver_phase": np.asarray(schedule.phase, dtype=np.uint8),
        "source_coordinate": np.asarray(
            schedule.source_coordinate, dtype=np.float32
        ),
    }


def save_maneuver_motion(
    path: str | Path,
    motion: StitchedMotion,
    schedule: ManeuverSchedule,
) -> Path:
    """Write a regular stitched-motion bundle plus synchronized commands."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = command_labels(motion, schedule)
    np.savez_compressed(
        destination,
        fps=np.asarray(motion.fps, dtype=np.float32),
        root_position_world=np.asarray(motion.root_position_world, np.float32),
        root_quaternion_world_wxyz=np.asarray(
            motion.root_quaternion_world_wxyz, np.float32
        ),
        joint_position=np.asarray(motion.joint_position, np.float32),
        seam_indices=np.asarray(motion.seam_indices, dtype=np.int64),
        source_archive_clip_index=np.asarray(
            [value.archive_clip_index for value in motion.provenance],
            dtype=np.int64,
        ),
        source_frame=np.asarray(
            [value.source_frame for value in motion.provenance], dtype=np.int64
        ),
        source_clip_id=np.asarray(
            [value.clip_id for value in motion.provenance], dtype=np.str_
        ),
        maneuver_mode=np.asarray(schedule.mode, dtype=np.str_),
        pivot_source_frame=np.asarray(
            schedule.pivot_source_frame, dtype=np.int64
        ),
        pivot_source_frames=np.asarray(
            schedule.pivot_source_frames
            or (schedule.pivot_source_frame,),
            dtype=np.int64,
        ),
        maneuver_phase_names=np.asarray(PHASE_NAMES, dtype=np.str_),
        **labels,
    )
    return destination


__all__ = (
    "ACCELERATE_FORWARD",
    "ACCELERATE_REVERSE",
    "DECELERATE",
    "HOLD",
    "ManeuverSchedule",
    "NATIVE",
    "NATIVE_FORWARD",
    "NATIVE_REVERSE",
    "PHASE_NAMES",
    "SupportPivot",
    "build_bounce_schedule",
    "build_maneuver_schedule",
    "choose_support_pivot",
    "choose_support_pivots",
    "command_labels",
    "resample_stitched_motion",
    "save_maneuver_motion",
)
