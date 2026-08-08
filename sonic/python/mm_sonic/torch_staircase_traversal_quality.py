"""Independent contact and naturalness metrics for staircase traversals."""

from __future__ import annotations

import math

import numpy as np

from .joints import ContractError


def _true_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    owned = np.asarray(mask, dtype=np.bool_)
    if owned.ndim != 1:
        raise ContractError("quality run mask is invalid")
    padded = np.pad(owned.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    return tuple((int(start), int(stop)) for start, stop in zip(starts, stops))


def measure_cadence(
    *,
    touchdown_frames: object,
    touchdown_heights_m: object,
    height_change_threshold_m: float = 0.05,
    allowed_relative_error: float = 0.20,
) -> dict[str, object]:
    """Measure cadence only between contacts on the same terrain level."""

    frames = np.asarray(touchdown_frames, dtype=np.int64)
    heights = np.asarray(touchdown_heights_m, dtype=np.float64)
    if (
        frames.ndim != 1
        or heights.shape != frames.shape
        or len(frames) < 2
        or np.any(np.diff(frames) <= 0)
        or not np.isfinite(heights).all()
        or not math.isfinite(float(height_change_threshold_m))
        or float(height_change_threshold_m) <= 0.0
        or not math.isfinite(float(allowed_relative_error))
        or not 0.0 <= float(allowed_relative_error) < 1.0
    ):
        raise ContractError("touchdown cadence input is invalid")
    intervals = np.diff(frames)
    same_height = (
        np.abs(np.diff(heights)) <= float(height_change_threshold_m)
    )
    trusted = intervals[same_height]
    target = float(np.median(trusted)) if len(trusted) else 0.0
    violations = (
        np.abs(trusted - target)
        > float(allowed_relative_error) * target
        if target > 0.0
        else np.zeros(len(trusted), dtype=np.bool_)
    )
    return {
        "target_half_step_frames": target,
        "same_height_half_step_frames": trusted.astype(int).tolist(),
        "cadence_violation_count": int(violations.sum()),
    }


def _terrain_event_labels(
    touchdown_heights_m: np.ndarray,
    *,
    elevated_threshold_m: float,
) -> tuple[str, ...]:
    ground = float(touchdown_heights_m[0])
    observed_elevated = False
    labels = []
    for height in touchdown_heights_m:
        is_elevated = float(height) > ground + elevated_threshold_m
        if is_elevated:
            observed_elevated = True
            labels.append("elevated")
        elif observed_elevated:
            labels.append("opposite_ground")
        else:
            labels.append("ground")
    return tuple(labels)


def measure_staircase_traversal_quality(
    *,
    feet_scene_xyz: object,
    sole_points_scene_xyz: object,
    support_mask: object,
    heading_scene_xy: object,
    sample_surface,
    elevated_threshold_m: float = 0.05,
    sole_contact_lower_m: float = -0.025,
    sole_contact_upper_m: float = 0.035,
) -> dict[str, object]:
    """Measure actual support and terrain events without search metadata."""

    feet = np.asarray(feet_scene_xyz, dtype=np.float64)
    soles = np.asarray(sole_points_scene_xyz, dtype=np.float64)
    support = np.asarray(support_mask)
    heading = np.asarray(heading_scene_xy, dtype=np.float64)
    frame_count = len(feet)
    if (
        frame_count < 2
        or feet.shape != (frame_count, 2, 3)
        or soles.ndim != 4
        or soles.shape[:2] != (frame_count, 2)
        or soles.shape[2] < 3
        or soles.shape[3] != 3
        or support.shape != (frame_count, 2)
        or support.dtype != np.bool_
        or heading.shape != (2,)
        or not np.isfinite(feet).all()
        or not np.isfinite(soles).all()
        or not np.isfinite(heading).all()
        or float(np.linalg.norm(heading)) <= 0.0
        or not callable(sample_surface)
    ):
        raise ContractError("staircase traversal quality input is invalid")
    sole_surface = np.asarray(
        sample_surface(soles[..., :2]), dtype=np.float64
    )
    if sole_surface.shape != soles.shape[:-1] or not np.isfinite(
        sole_surface
    ).all():
        raise ContractError("staircase traversal terrain samples are invalid")
    sole_clearance = soles[..., 2] - sole_surface
    supported_points = (
        (sole_clearance >= float(sole_contact_lower_m))
        & (sole_clearance <= float(sole_contact_upper_m))
    ).sum(axis=2)
    stance_sole_error = np.max(np.abs(sole_clearance), axis=2)
    consecutive_support = support[:-1] & support[1:]
    foot_step = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=2)

    rising = support & ~np.vstack(
        (np.zeros((1, 2), dtype=np.bool_), support[:-1])
    )
    touchdown_frame, touchdown_foot = np.nonzero(rising)
    order = np.argsort(touchdown_frame, kind="stable")
    touchdown_frame = touchdown_frame[order]
    touchdown_foot = touchdown_foot[order]
    if len(touchdown_frame) < 2:
        raise ContractError("traversal has insufficient touchdown events")
    touchdown_height = np.asarray(
        [
            float(np.median(sole_surface[frame, foot]))
            for frame, foot in zip(touchdown_frame, touchdown_foot)
        ],
        dtype=np.float64,
    )
    cadence = measure_cadence(
        touchdown_frames=touchdown_frame,
        touchdown_heights_m=touchdown_height,
        height_change_threshold_m=float(elevated_threshold_m),
    )

    forward = heading / np.linalg.norm(heading)
    lateral = np.array((-forward[1], forward[0]), dtype=np.float64)
    support_keyframes = support.all(axis=1) | rising.any(axis=1)
    lateral_separation = (
        feet[:, 0, :2] - feet[:, 1, :2]
    ) @ lateral

    foot_surface = np.median(sole_surface, axis=2)
    same_height_double = support.all(axis=1) & (
        np.abs(foot_surface[:, 0] - foot_surface[:, 1])
        <= float(elevated_threshold_m)
    )
    longest_double = max(
        (stop - start for start, stop in _true_runs(same_height_double)),
        default=0,
    )
    metrics: dict[str, object] = {
        "terrain_events": _terrain_event_labels(
            touchdown_height,
            elevated_threshold_m=float(elevated_threshold_m),
        ),
        "touchdown_frames": touchdown_frame.astype(int).tolist(),
        "touchdown_feet": touchdown_foot.astype(int).tolist(),
        "touchdown_heights_m": touchdown_height.tolist(),
        "maximum_stance_horizontal_step_m": float(
            foot_step[consecutive_support].max()
            if bool(consecutive_support.any())
            else 0.0
        ),
        "maximum_complete_sole_contact_error_m": float(
            stance_sole_error[support].max()
            if bool(support.any())
            else float("inf")
        ),
        "minimum_supported_sole_points": int(
            supported_points[support].min()
            if bool(support.any())
            else 0
        ),
        "minimum_lateral_foot_separation_m": float(
            lateral_separation[support_keyframes].min()
            if bool(support_keyframes.any())
            else float("-inf")
        ),
        "maximum_same_height_double_support_frames": int(longest_double),
        "terminal_complete_support": bool(support[-1].all()),
        "minimum_sole_clearance_m": float(sole_clearance.min()),
        **cadence,
    }
    return metrics


def admit_staircase_traversal(metrics: object) -> None:
    """Raise a stable failure when a route is unfit for grid playback."""

    if not isinstance(metrics, dict):
        raise ContractError("staircase traversal metrics are invalid")
    try:
        events = tuple(str(value) for value in metrics["terrain_events"])
        stance_step = float(metrics["maximum_stance_horizontal_step_m"])
        contact_error = float(
            metrics["maximum_complete_sole_contact_error_m"]
        )
        separation = float(metrics["minimum_lateral_foot_separation_m"])
        cadence_violations = int(metrics["cadence_violation_count"])
        double_support = int(
            metrics["maximum_same_height_double_support_frames"]
        )
        supported_points = int(metrics["minimum_supported_sole_points"])
        terminal_support = bool(metrics["terminal_complete_support"])
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(
            "staircase traversal metrics are invalid"
        ) from error
    if (
        len(events) < 3
        or events[0] != "ground"
        or "elevated" not in events[1:-1]
    ):
        raise ContractError("traversal is missing elevated traversal")
    if events[-1] != "opposite_ground":
        raise ContractError("traversal is missing opposite-ground exit")
    if stance_step > 0.010:
        raise ContractError("traversal stance foot slides")
    if contact_error > 0.025:
        raise ContractError("traversal loses complete-sole contact")
    if separation < 0.0:
        raise ContractError("traversal feet cross")
    if cadence_violations:
        raise ContractError(
            "traversal same-height cadence is inconsistent"
        )
    if double_support > 20:
        raise ContractError(
            "traversal has excessive same-height double support"
        )
    if supported_points < 3:
        raise ContractError("traversal has incomplete sole support")
    if not terminal_support:
        raise ContractError("traversal terminal phase is mid-swing")
