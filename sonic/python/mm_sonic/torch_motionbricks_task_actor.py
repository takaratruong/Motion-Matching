"""Sparse terrain proxy keyframes for MotionBricks task actors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .joints import (
    ContractError,
    PINNED_TARGET_TO_SOURCE_PERMUTATION,
)


@dataclass(frozen=True)
class ProxyKeyframeSequence:
    qpos: np.ndarray
    support_mask: np.ndarray
    endpoint_frames: tuple[int, ...]


def contact_phase_support(
    contact_channels: object,
    *,
    initial_support: object,
    terminal_support: object,
    endpoint_window_frames: int = 4,
) -> np.ndarray:
    """Convert MotionBricks heel/toe channels into a supported gait phase."""

    channels = np.asarray(contact_channels, dtype=np.float64)
    initial = np.asarray(initial_support)
    terminal = np.asarray(terminal_support)
    frame_count = len(channels)
    if (
        channels.shape != (frame_count, 4)
        or frame_count < 2
        or not np.isfinite(channels).all()
        or initial.shape != (2,)
        or terminal.shape != (2,)
        or initial.dtype != np.bool_
        or terminal.dtype != np.bool_
        or not bool(initial.any())
        or not bool(terminal.any())
        or type(endpoint_window_frames) is not int
        or endpoint_window_frames < 1
        or 2 * endpoint_window_frames > frame_count
    ):
        raise ContractError("MotionBricks contact phase input is invalid")
    foot_scores = channels.reshape(frame_count, 2, 2).max(axis=2)
    support = np.zeros((frame_count, 2), dtype=np.bool_)
    selected = np.argmax(foot_scores, axis=1)
    support[np.arange(frame_count), selected] = True
    support[:endpoint_window_frames] = initial
    support[-endpoint_window_frames:] = terminal
    return np.ascontiguousarray(support)


def stance_anchor_targets(
    *,
    foot_position_world: object,
    support_mask: object,
    surface_height_m: object,
    ankle_origin_sole_m: float,
) -> np.ndarray:
    """Hold each stance at its generated touchdown location."""

    feet = np.asarray(foot_position_world, dtype=np.float64)
    support = np.asarray(support_mask)
    surface = np.asarray(surface_height_m, dtype=np.float64)
    frame_count = len(feet)
    if (
        feet.shape != (frame_count, 2, 3)
        or support.shape != (frame_count, 2)
        or support.dtype != np.bool_
        or surface.shape != (frame_count, 2)
        or not np.isfinite(feet).all()
        or not np.isfinite(surface).all()
        or not np.isfinite(ankle_origin_sole_m)
        or ankle_origin_sole_m <= 0.0
    ):
        raise ContractError("MotionBricks stance anchor input is invalid")
    targets = feet.copy()
    for foot in range(2):
        anchor: np.ndarray | None = None
        for frame in range(frame_count):
            if not support[frame, foot]:
                anchor = None
                continue
            if anchor is None:
                anchor = feet[frame, foot].copy()
                anchor[2] = (
                    surface[frame, foot] + float(ankle_origin_sole_m)
                )
            targets[frame, foot] = anchor
    return np.ascontiguousarray(targets)


def stance_root_height_correction(
    *,
    actual_foot_position_world: object,
    target_foot_position_world: object,
    support_mask: object,
    maximum_absolute_correction_m: float = 0.08,
) -> float:
    """Return the bounded pelvis-Z shift that lands supported ankles."""

    actual = np.asarray(actual_foot_position_world, dtype=np.float64)
    target = np.asarray(target_foot_position_world, dtype=np.float64)
    support = np.asarray(support_mask)
    if (
        actual.shape != (2, 3)
        or target.shape != (2, 3)
        or support.shape != (2,)
        or support.dtype != np.bool_
        or not bool(support.any())
        or not np.isfinite(actual).all()
        or not np.isfinite(target).all()
        or not np.isfinite(maximum_absolute_correction_m)
        or maximum_absolute_correction_m <= 0.0
    ):
        raise ContractError("MotionBricks stance height input is invalid")
    correction = float(
        np.mean(target[support, 2] - actual[support, 2])
    )
    if abs(correction) > float(maximum_absolute_correction_m):
        raise ContractError(
            "MotionBricks stance height correction is too large"
        )
    return correction


def stance_root_clearance_lift(
    *,
    minimum_sole_clearance_m: object,
    support_mask: object,
    accepted_clearance_m: float = -0.02,
    maximum_lift_m: float = 0.04,
) -> float:
    """Lift the pelvis enough to clear the worst supported sole point."""

    clearance = np.asarray(
        minimum_sole_clearance_m, dtype=np.float64
    )
    support = np.asarray(support_mask)
    if (
        clearance.shape != (2,)
        or support.shape != (2,)
        or support.dtype != np.bool_
        or not bool(support.any())
        or not np.isfinite(clearance).all()
        or not np.isfinite(accepted_clearance_m)
        or accepted_clearance_m >= 0.0
        or not np.isfinite(maximum_lift_m)
        or maximum_lift_m <= 0.0
    ):
        raise ContractError("MotionBricks stance clearance input is invalid")
    lift = max(
        0.0,
        float(accepted_clearance_m)
        - float(clearance[support].min()),
    )
    if lift > float(maximum_lift_m):
        raise ContractError(
            "MotionBricks stance clearance lift is too large"
        )
    return lift


def _native_qpos(
    joints: np.ndarray,
    roots: np.ndarray,
    quaternions: np.ndarray,
) -> np.ndarray:
    native_joints = np.empty_like(joints)
    native_joints[
        ..., np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION)
    ] = joints
    return np.ascontiguousarray(
        np.concatenate((roots, quaternions, native_joints), axis=-1),
        dtype=np.float64,
    )


def _validated_route(
    route: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        joints = np.asarray(route["joint_position"], dtype=np.float64)
        roots = np.asarray(
            route["root_position_world"], dtype=np.float64
        )
        quaternions = np.asarray(
            route["root_orientation_world_wxyz"], dtype=np.float64
        )
        support = np.asarray(route["source_support_mask"])
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("MotionBricks proxy route is incomplete") from error
    frame_count = len(joints)
    if (
        frame_count < 4
        or joints.shape != (frame_count, 29)
        or roots.shape != (frame_count, 3)
        or quaternions.shape != (frame_count, 4)
        or support.shape != (frame_count, 2)
        or support.dtype != np.bool_
        or not all(
            np.isfinite(value).all()
            for value in (joints, roots, quaternions)
        )
        or np.any(
            np.abs(np.linalg.norm(quaternions, axis=1) - 1.0)
            > 1.0e-4
        )
    ):
        raise ContractError("MotionBricks proxy route arrays are invalid")
    return joints, roots, quaternions, support


def extract_proxy_keyframes(
    route: Mapping[str, object],
    *,
    endpoint_frames: Sequence[int],
) -> ProxyKeyframeSequence:
    """Extract ordered four-frame target windows from a certified route."""

    joints, roots, quaternions, support = _validated_route(route)
    endpoints = tuple(endpoint_frames)
    if (
        not endpoints
        or any(type(frame) is not int for frame in endpoints)
        or endpoints != tuple(sorted(endpoints))
        or any(frame < 3 or frame >= len(joints) for frame in endpoints)
        or any(
            right - left < 4
            for left, right in zip(endpoints, endpoints[1:])
        )
        or any(not bool(support[frame].any()) for frame in endpoints)
    ):
        raise ContractError("MotionBricks proxy endpoints are invalid")
    windows = tuple(
        slice(endpoint - 3, endpoint + 1) for endpoint in endpoints
    )
    qpos = np.stack(
        tuple(
            _native_qpos(
                joints[window],
                roots[window],
                quaternions[window],
            )
            for window in windows
        )
    )
    support_windows = np.stack(
        tuple(support[window] for window in windows)
    ).copy()
    qpos.setflags(write=False)
    support_windows.setflags(write=False)
    return ProxyKeyframeSequence(qpos, support_windows, endpoints)


def infer_generated_support(
    *,
    foot_speed_mps: object,
    minimum_sole_clearance_m: object,
    supported_sole_points: object,
) -> np.ndarray:
    """Infer conservative stance labels from generated terrain evidence."""

    speed = np.asarray(foot_speed_mps, dtype=np.float64)
    clearance = np.asarray(
        minimum_sole_clearance_m, dtype=np.float64
    )
    points = np.asarray(supported_sole_points)
    if (
        speed.ndim != 2
        or speed.shape[1] != 2
        or clearance.shape != speed.shape
        or points.shape != speed.shape
        or not np.issubdtype(points.dtype, np.integer)
        or not np.isfinite(speed).all()
        or not np.isfinite(clearance).all()
        or np.any(speed < 0.0)
        or np.any(points < 0)
    ):
        raise ContractError("MotionBricks support evidence is invalid")
    return np.ascontiguousarray(
        (speed <= 0.12)
        & (clearance >= -0.025)
        & (clearance <= 0.035)
        & (points >= 3)
    )


def endpoint_support_schedule(
    *,
    initial_support: object,
    target_support: object,
    frame_count: int,
    target_window_frames: int = 4,
) -> np.ndarray:
    """Lock initial contacts until MotionBricks' exact target window."""

    initial = np.asarray(initial_support)
    target = np.asarray(target_support)
    if (
        initial.dtype != np.bool_
        or target.dtype != np.bool_
        or initial.shape != (2,)
        or target.shape != (2,)
        or not bool(initial.any())
        or not bool(target.any())
        or type(frame_count) is not int
        or type(target_window_frames) is not int
        or target_window_frames < 1
        or frame_count <= target_window_frames
    ):
        raise ContractError("MotionBricks support schedule is invalid")
    output = np.tile(initial, (frame_count, 1))
    output[-target_window_frames:] = target
    return np.ascontiguousarray(output)


def flight_support_schedule(
    *,
    initial_support: object,
    target_support: object,
    frame_count: int,
    context_frames: int = 4,
    target_window_frames: int = 4,
) -> np.ndarray:
    """Represent an explicit support-to-flight-to-landing transition."""

    initial = np.asarray(initial_support)
    target = np.asarray(target_support)
    if (
        initial.dtype != np.bool_
        or target.dtype != np.bool_
        or initial.shape != (2,)
        or target.shape != (2,)
        or not bool(initial.any())
        or not bool(target.any())
        or type(frame_count) is not int
        or type(context_frames) is not int
        or type(target_window_frames) is not int
        or context_frames < 1
        or target_window_frames < 1
        or frame_count <= context_frames + target_window_frames
    ):
        raise ContractError("MotionBricks flight schedule is invalid")
    output = np.zeros((frame_count, 2), dtype=np.bool_)
    output[:context_frames] = initial
    output[-target_window_frames:] = target
    return output


def _validated_native_qpos(value: object, name: str) -> np.ndarray:
    qpos = np.asarray(value, dtype=np.float64)
    if (
        qpos.ndim != 2
        or qpos.shape[1] != 36
        or len(qpos) < 4
        or not np.isfinite(qpos).all()
        or np.any(
            np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1.0)
            > 1.0e-4
        )
    ):
        raise ContractError(f"MotionBricks {name} qpos is invalid")
    return qpos


def assemble_generated_route(
    *,
    context_qpos: object,
    context_support: object,
    transition_qpos: Sequence[object],
    transition_support: Sequence[object],
) -> dict[str, np.ndarray]:
    """Concatenate native transitions after removing context overlap."""

    context = _validated_native_qpos(context_qpos, "context")
    if len(context) != 4:
        raise ContractError("MotionBricks context must contain four frames")
    support = np.asarray(context_support)
    transitions = tuple(transition_qpos)
    support_transitions = tuple(transition_support)
    if (
        support.shape != (4, 2)
        or support.dtype != np.bool_
        or not transitions
        or len(transitions) != len(support_transitions)
    ):
        raise ContractError("MotionBricks route assembly is invalid")

    chunks = [context.copy()]
    support_chunks = [support.copy()]
    boundaries = [4]
    previous = context
    for qpos_value, support_value in zip(
        transitions, support_transitions
    ):
        qpos = _validated_native_qpos(qpos_value, "transition")
        transition_mask = np.asarray(support_value)
        if (
            transition_mask.shape != (len(qpos), 2)
            or transition_mask.dtype != np.bool_
            or not np.allclose(qpos[:4], previous[-4:], atol=1.0e-6)
        ):
            raise ContractError(
                "MotionBricks transition context is inconsistent"
            )
        chunks.append(qpos[4:].copy())
        support_chunks.append(transition_mask[4:].copy())
        boundaries.append(boundaries[-1] + len(qpos) - 4)
        previous = qpos

    native = np.ascontiguousarray(np.concatenate(chunks, axis=0))
    output_support = np.ascontiguousarray(
        np.concatenate(support_chunks, axis=0)
    )
    permutation = np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION)
    return {
        "joint_position": np.ascontiguousarray(
            native[:, 7 + permutation]
        ),
        "root_position_world": native[:, :3].copy(),
        "root_orientation_world_wxyz": native[:, 3:7].copy(),
        "source_support_mask": output_support,
        "segment_boundaries": np.asarray(
            boundaries, dtype=np.int64
        ),
    }
