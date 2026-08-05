#!/usr/bin/env python3
"""Generate a terrain task-actor route with the MotionBricks backbone."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import torch

from mm_sonic.joints import ContractError
from mm_sonic.joints import PINNED_TARGET_TO_SOURCE_PERMUTATION
from mm_sonic.torch_contact_segments import ANKLE_ORIGIN_SOLE_M
from mm_sonic.torch_motionbricks_task_actor import (
    assemble_generated_route,
    contact_phase_support,
    endpoint_support_schedule,
    extract_proxy_keyframes,
    flight_support_schedule,
    infer_generated_support,
    stance_anchor_targets,
    stance_root_clearance_lift,
    stance_root_height_correction,
)
from mm_sonic.torch_motionbricks_drop_fallback import (
    terrain_clearance_envelope,
)
from mm_sonic.torch_supported_step_up import (
    smooth_swing_clearance_lift,
)
from mm_sonic.torch_horizontal_terrain_retarget import (
    nearest_valid_sole_translation,
)


def _four_native_qpos(value: object) -> np.ndarray:
    qpos = np.asarray(value, dtype=np.float32)
    if (
        qpos.shape != (4, 36)
        or not np.isfinite(qpos).all()
        or np.any(
            np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1.0)
            > 1.0e-4
        )
    ):
        raise ContractError(
            "MotionBricks constraint qpos must have shape (4, 36)"
        )
    return qpos


def motionbricks_constraints(
    agent: object,
    context_qpos: object,
    target_qpos: object,
) -> dict[str, torch.Tensor]:
    """Convert native G1 qpos into direct MotionBricks constraints."""

    context = torch.as_tensor(
        _four_native_qpos(context_qpos),
        dtype=torch.float32,
        device=agent._device,
    )[None]
    target = torch.as_tensor(
        _four_native_qpos(target_qpos),
        dtype=torch.float32,
        device=agent._device,
    )[None]
    context_positions, context_rotations = (
        agent._converter.convert_mujoco_qpos_to_motion_transforms(context)
    )
    target_positions, target_rotations = (
        agent._converter.convert_mujoco_qpos_to_motion_transforms(target)
    )
    projected_root = target_positions[:, :, 0] * torch.tensor(
        (1.0, 0.0, 1.0),
        dtype=target_positions.dtype,
        device=target_positions.device,
    )
    target_headings = torch.atan2(
        target_rotations[:, :, 0, 0, 2],
        target_rotations[:, :, 0, 2, 2],
    )
    return {
        "context_global_joint_positions": context_positions,
        "context_global_joint_rotations": context_rotations,
        "target_global_joint_positions": (
            target_positions - projected_root[:, :, None]
        ),
        "target_global_joint_rotations": target_rotations,
        "target_global_root_positions": projected_root,
        "target_root_headings": target_headings,
    }


def allowed_token_masks(
    minimum_tokens: int, maximum_tokens: int
) -> dict[int, torch.Tensor]:
    """Return one duration mask for every checkpoint-supported length."""

    if (
        type(minimum_tokens) is not int
        or type(maximum_tokens) is not int
        or minimum_tokens < 1
        or maximum_tokens < minimum_tokens
    ):
        raise ContractError("MotionBricks token bounds are invalid")
    count = maximum_tokens - minimum_tokens + 1
    output: dict[int, torch.Tensor] = {}
    for token_count in range(minimum_tokens, maximum_tokens + 1):
        mask = torch.zeros((1, count), dtype=torch.int64)
        mask[0, token_count - minimum_tokens] = 1
        output[token_count] = mask
    return output


def preferred_candidate_frame_count(
    *,
    source_frame_count: int,
    source_frames_per_second: float,
    output_frames_per_second: float,
    available_frame_counts: tuple[int, ...],
) -> int:
    """Map proxy timing to the nearest supported generated duration."""

    if (
        type(source_frame_count) is not int
        or source_frame_count < 1
        or not math.isfinite(source_frames_per_second)
        or source_frames_per_second <= 0.0
        or not math.isfinite(output_frames_per_second)
        or output_frames_per_second <= 0.0
        or not available_frame_counts
        or any(
            type(frame_count) is not int or frame_count < 1
            for frame_count in available_frame_counts
        )
    ):
        raise ContractError("MotionBricks proxy cadence is invalid")
    target = (
        source_frame_count
        * output_frames_per_second
        / source_frames_per_second
    )
    return min(
        available_frame_counts,
        key=lambda frame_count: (abs(frame_count - target), frame_count),
    )


def normalize_generated_qpos(value: object) -> np.ndarray:
    """Normalize bounded quaternion roundoff from MotionBricks decoding."""

    qpos = np.asarray(value, dtype=np.float64)
    if (
        qpos.ndim != 2
        or qpos.shape[1] != 36
        or len(qpos) < 4
        or not np.isfinite(qpos).all()
    ):
        raise ContractError("MotionBricks generated qpos is invalid")
    norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    if np.any(norms <= 0.0) or np.any(np.abs(norms - 1.0) > 1.0e-3):
        raise ContractError(
            "MotionBricks generated quaternion is malformed"
        )
    output = qpos.copy()
    output[:, 3:7] /= norms[:, None]
    return np.ascontiguousarray(output)


def contact_projection_metrics(
    raw_qpos: object,
    projected_qpos: object,
) -> dict[str, object]:
    """Measure how much terrain projection changed a generated motion."""

    raw = normalize_generated_qpos(raw_qpos)
    projected = normalize_generated_qpos(projected_qpos)
    if raw.shape != projected.shape:
        raise ContractError(
            "MotionBricks contact projection arrays do not match"
        )
    joint_delta = np.abs(projected[:, 7:] - raw[:, 7:])
    root_delta = np.linalg.norm(
        projected[:, :3] - raw[:, :3], axis=1
    )
    joint_by_frame = joint_delta.max(axis=1)
    combined = np.maximum(joint_by_frame, root_delta)
    return {
        "mean_contact_projection_joint_delta_rad": float(
            joint_delta.mean()
        ),
        "maximum_contact_projection_joint_delta_rad": float(
            joint_delta.max()
        ),
        "maximum_contact_projection_root_delta_m": float(
            root_delta.max()
        ),
        "maximum_contact_projection_frame": int(combined.argmax()),
    }


def _candidate_rejections(
    metrics: dict[str, object],
) -> tuple[str, ...]:
    required = {
        "candidate_id",
        "minimum_sole_clearance_m",
        "maximum_stance_contact_error_m",
        "maximum_stance_horizontal_step_m",
        "maximum_heading_error_rad",
        "endpoint_root_error_m",
        "unsupported_frame_count",
        "mean_joint_acceleration",
        "maximum_joint_step_rad",
    }
    if set(metrics) < required:
        raise ContractError("MotionBricks candidate metrics are incomplete")
    numeric = tuple(
        metrics[name]
        for name in required
        if name != "candidate_id"
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric
    ):
        raise ContractError("MotionBricks candidate metrics are invalid")
    output: list[str] = []
    if float(metrics["minimum_sole_clearance_m"]) < -0.025:
        output.append("terrain penetration")
    if float(metrics["maximum_stance_contact_error_m"]) > 0.026:
        output.append("stance contact error")
    if float(metrics["maximum_stance_horizontal_step_m"]) > 0.010:
        output.append("stance slide")
    if float(metrics["maximum_heading_error_rad"]) > 0.30:
        output.append("heading error")
    if float(metrics["endpoint_root_error_m"]) > 0.10:
        output.append("endpoint root error")
    if int(metrics["unsupported_frame_count"]) > 0:
        if int(metrics["unsupported_frame_count"]) != int(
            metrics.get("planned_flight_frame_count", 0)
        ):
            output.append("unsupported frame")
    return tuple(output)


def select_candidate(
    candidates: object,
    *,
    preferred_frame_count: int | None = None,
) -> dict[str, object]:
    """Select the best hard-valid MotionBricks candidate."""

    if not isinstance(candidates, (tuple, list)) or not candidates:
        raise ContractError("MotionBricks candidates are invalid")
    if (
        preferred_frame_count is not None
        and (
            type(preferred_frame_count) is not int
            or preferred_frame_count < 1
        )
    ):
        raise ContractError(
            "MotionBricks preferred frame count is invalid"
        )
    valid: list[dict[str, object]] = []
    for value in candidates:
        if not isinstance(value, dict):
            raise ContractError("MotionBricks candidates are invalid")
        if preferred_frame_count is not None and (
            type(value.get("frame_count")) is not int
            or int(value["frame_count"]) < 1
        ):
            raise ContractError(
                "MotionBricks candidate frame count is invalid"
            )
        if not _candidate_rejections(value):
            valid.append(value)
    if not valid:
        raise ContractError("MotionBricks has no valid terrain candidate")
    return min(
        valid,
        key=lambda item: (
            (
                abs(int(item["frame_count"]) - preferred_frame_count)
                if preferred_frame_count is not None
                else 0
            ),
            float(item["endpoint_root_error_m"]),
            float(item["maximum_stance_horizontal_step_m"]),
            -float(item["minimum_sole_clearance_m"]),
            float(item["mean_joint_acceleration"]),
            float(item["maximum_joint_step_rad"]),
            str(item["candidate_id"]),
        ),
    )


def _target_arrays_from_native(
    qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    permutation = np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION)
    return (
        np.ascontiguousarray(qpos[:, 7 + permutation]),
        np.ascontiguousarray(qpos[:, :3]),
        np.ascontiguousarray(qpos[:, 3:7]),
    )


def _heading_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion.T
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrapped_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def candidate_metrics(
    *,
    candidate_id: str,
    native_qpos: np.ndarray,
    target_qpos: np.ndarray,
    desired_heading_rad: float,
    foot_kinematics: object,
    sole_kinematics: object,
    measurement_extension: object,
    frames_per_second: float = 30.0,
    planned_support: np.ndarray | None = None,
    planned_flight: bool = False,
) -> tuple[dict[str, object], np.ndarray]:
    """Measure one generated transition against the target staircase."""

    qpos = np.asarray(native_qpos, dtype=np.float64)
    target = _four_native_qpos(target_qpos).astype(np.float64)
    if (
        not isinstance(candidate_id, str)
        or not candidate_id
        or qpos.ndim != 2
        or qpos.shape[1] != 36
        or len(qpos) < 5
        or not np.isfinite(qpos).all()
        or not math.isfinite(desired_heading_rad)
        or not math.isfinite(frames_per_second)
        or frames_per_second <= 0.0
    ):
        raise ContractError("MotionBricks candidate input is invalid")

    joints, roots, quaternions = _target_arrays_from_native(qpos)
    feet = foot_kinematics.foot_positions(
        joints, roots, quaternions
    )
    soles = sole_kinematics.sole_points(
        joints, roots, quaternions
    )
    import torch as local_torch

    alignment = measurement_extension.alignment
    grid = measurement_extension.query_grid
    foot_surface = (
        grid.sample_xy(
            alignment.matcher_to_scene_xy(
                local_torch.tensor(feet[..., :2], dtype=local_torch.float32)
            )
        )
        .cpu()
        .numpy()
    )
    sole_surface = (
        grid.sample_xy(
            alignment.matcher_to_scene_xy(
                local_torch.tensor(
                    soles[..., :2], dtype=local_torch.float32
                )
            )
        )
        .cpu()
        .numpy()
    )
    foot_clearance = (
        feet[..., 2] - foot_surface - float(ANKLE_ORIGIN_SOLE_M)
    )
    sole_clearance = soles[..., 2] - sole_surface
    minimum_by_foot = sole_clearance.min(axis=2)
    minimum_sole_flat = int(np.argmin(sole_clearance))
    minimum_sole_index = np.unravel_index(
        minimum_sole_flat, sole_clearance.shape
    )
    supported_points = (
        (sole_clearance >= -0.025) & (sole_clearance <= 0.035)
    ).sum(axis=2)
    foot_step = np.linalg.norm(np.diff(feet, axis=0), axis=2)
    foot_speed = np.empty_like(minimum_by_foot)
    foot_speed[0] = foot_step[0] * frames_per_second
    foot_speed[1:] = foot_step * frames_per_second
    inferred_support = infer_generated_support(
        foot_speed_mps=foot_speed,
        minimum_sole_clearance_m=minimum_by_foot,
        supported_sole_points=supported_points,
    )
    if planned_support is None:
        support = inferred_support
    else:
        support = np.asarray(planned_support)
        if (
            support.shape != (len(qpos), 2)
            or support.dtype != np.bool_
        ):
            raise ContractError(
                "MotionBricks planned support is invalid"
            )
    consecutive = support[:-1] & support[1:]
    horizontal_step = np.linalg.norm(
        np.diff(feet[..., :2], axis=0), axis=2
    )
    stance_step = np.where(consecutive, horizontal_step, -np.inf)
    if bool(consecutive.any()):
        stance_step_flat = int(np.argmax(stance_step))
        stance_step_index = np.unravel_index(
            stance_step_flat, stance_step.shape
        )
        maximum_stance_step_frame = int(stance_step_index[0] + 1)
        maximum_stance_step_foot = int(stance_step_index[1])
    else:
        maximum_stance_step_frame = -1
        maximum_stance_step_foot = -1
    headings = _heading_wxyz(quaternions)
    heading_error = np.abs(
        _wrapped_angle(headings - desired_heading_rad)
    )
    joint_delta = np.diff(joints, axis=0)
    joint_acceleration = (
        np.diff(joints, n=2, axis=0)
        if len(joints) > 2
        else np.zeros((1, 29))
    )
    metrics: dict[str, object] = {
        "candidate_id": candidate_id,
        "frame_count": len(qpos),
        "minimum_sole_clearance_m": float(sole_clearance.min()),
        "minimum_sole_clearance_frame": int(minimum_sole_index[0]),
        "minimum_sole_clearance_foot": int(minimum_sole_index[1]),
        "maximum_stance_contact_error_m": float(
            np.abs(foot_clearance[support]).max()
            if bool(support.any())
            else math.inf
        ),
        "maximum_stance_horizontal_step_m": float(
            horizontal_step[consecutive].max()
            if bool(consecutive.any())
            else 0.0
        ),
        "maximum_stance_horizontal_step_frame": (
            maximum_stance_step_frame
        ),
        "maximum_stance_horizontal_step_foot": maximum_stance_step_foot,
        "maximum_heading_error_rad": float(heading_error.max()),
        "endpoint_root_error_m": float(
            np.linalg.norm(qpos[-1, :3] - target[-1, :3])
        ),
        "endpoint_joint_error_rad": float(
            np.abs(qpos[-1, 7:] - target[-1, 7:]).max()
        ),
        "unsupported_frame_count": int((~support.any(axis=1)).sum()),
        "planned_flight_frame_count": int(
            (~support.any(axis=1)).sum() if planned_flight else 0
        ),
        "inferred_unsupported_frame_count": int(
            (~inferred_support.any(axis=1)).sum()
        ),
        "mean_joint_acceleration": float(
            np.abs(joint_acceleration).mean()
        ),
        "maximum_joint_step_rad": float(
            np.abs(joint_delta).max()
        ),
    }
    metrics["rejections"] = list(_candidate_rejections(metrics))
    return metrics, support


def project_candidate_contacts(
    *,
    native_qpos: np.ndarray,
    target_qpos: np.ndarray,
    initial_support: np.ndarray,
    target_support: np.ndarray,
    retargeter: object,
    swing_retargeter: object,
    foot_kinematics: object,
    sole_kinematics: object,
    measurement_extension: object,
    support_schedule: np.ndarray | None = None,
) -> np.ndarray:
    """Project explicit stance locks and colliding swings onto terrain."""

    qpos = normalize_generated_qpos(native_qpos)
    _four_native_qpos(target_qpos)
    joints, roots, quaternions = _target_arrays_from_native(qpos)
    schedule = (
        endpoint_support_schedule(
            initial_support=np.asarray(initial_support),
            target_support=np.asarray(target_support),
            frame_count=len(qpos),
            target_window_frames=4,
        )
        if support_schedule is None
        else np.asarray(support_schedule)
    )
    if schedule.shape != (len(qpos), 2) or schedule.dtype != np.bool_:
        raise ContractError("MotionBricks projection schedule is invalid")
    generated_feet = foot_kinematics.foot_positions(
        joints, roots, quaternions
    )
    generated_soles = sole_kinematics.sole_points(
        joints, roots, quaternions
    )
    start_feet = generated_feet[3].copy()
    import torch as local_torch

    alignment = measurement_extension.alignment
    grid = measurement_extension.query_grid
    generated_surface = (
        grid.sample_xy(
            alignment.matcher_to_scene_xy(
                local_torch.tensor(
                    generated_feet[..., :2], dtype=local_torch.float32
                )
            )
        )
        .cpu()
        .numpy()
    )
    stance_targets = stance_anchor_targets(
        foot_position_world=generated_feet,
        support_mask=schedule,
        surface_height_m=generated_surface,
        ankle_origin_sole_m=float(ANKLE_ORIGIN_SOLE_M),
    )
    yaw = float(
        alignment.yaw_scene_from_matcher.detach().cpu().item()
    )
    cosine = math.cos(yaw)
    sine = math.sin(yaw)

    def sample_scene_height(points_scene_xy: np.ndarray) -> np.ndarray:
        return (
            grid.sample_xy(
                local_torch.tensor(
                    points_scene_xy, dtype=local_torch.float32
                )
            )
            .cpu()
            .numpy()
        )

    for foot in range(2):
        starts = np.flatnonzero(
            schedule[:, foot]
            & np.concatenate(
                (
                    np.ones(1, dtype=np.bool_),
                    ~schedule[:-1, foot],
                )
            )
        )
        for start in starts:
            if start < 4:
                continue
            stop = int(start) + 1
            while stop < len(schedule) and schedule[stop, foot]:
                stop += 1
            sole_scene_xy = (
                alignment.matcher_to_scene_xy(
                    local_torch.tensor(
                        generated_soles[start, foot, :, :2],
                        dtype=local_torch.float32,
                    )
                )
                .cpu()
                .numpy()
            )
            shift_scene = nearest_valid_sole_translation(
                sole_point_scene_xy=sole_scene_xy,
                support_surface_height_m=float(
                    generated_surface[start, foot]
                ),
                sample_height=sample_scene_height,
                maximum_shift_m=0.08,
                search_resolution_m=0.005,
                safety_margin_m=0.005,
            )
            shift_matcher = np.array(
                (
                    cosine * shift_scene[0] + sine * shift_scene[1],
                    -sine * shift_scene[0] + cosine * shift_scene[1],
                ),
                dtype=np.float64,
            )
            stance_targets[start:stop, foot, :2] += shift_matcher
            target_scene_xy = (
                alignment.matcher_to_scene_xy(
                    local_torch.tensor(
                        stance_targets[start, foot, :2],
                        dtype=local_torch.float32,
                    )
                )
                .cpu()
                .numpy()
            )
            target_surface = float(
                sample_scene_height(target_scene_xy[None])[0]
            )
            stance_targets[start:stop, foot, 2] = (
                target_surface + float(ANKLE_ORIGIN_SOLE_M)
            )
    stance_targets[:4, np.asarray(initial_support)] = start_feet[
        np.asarray(initial_support)
    ]
    for frame in range(4, len(qpos)):
        support = schedule[frame]
        if not bool(support.any()):
            continue
        targets = generated_feet[frame].copy()
        targets[support] = stance_targets[frame, support]
        try:
            joints[frame], roots[frame] = retargeter.solve_frame(
                joint_position=joints[frame],
                root_position_world=roots[frame],
                root_orientation_world_wxyz=quaternions[frame],
                solve_feet=support,
                target_foot_position_world=targets,
                level_feet=support,
                initial_joint_position=joints[frame - 1],
                initial_root_position_world=roots[frame - 1],
            )
        except ContractError as error:
            raise ContractError(
                f"stance projection frame {frame}: {error}"
            ) from error
        solved_feet = foot_kinematics.foot_positions(
            joints[frame : frame + 1],
            roots[frame : frame + 1],
            quaternions[frame : frame + 1],
        )[0]
        roots[frame, 2] += stance_root_height_correction(
            actual_foot_position_world=solved_feet,
            target_foot_position_world=targets,
            support_mask=support,
        )
        solved_soles = sole_kinematics.sole_points(
            joints[frame : frame + 1],
            roots[frame : frame + 1],
            quaternions[frame : frame + 1],
        )[0]
        solved_surface = (
            grid.sample_xy(
                alignment.matcher_to_scene_xy(
                    local_torch.tensor(
                        solved_soles[..., :2],
                        dtype=local_torch.float32,
                    )
                )
            )
            .cpu()
            .numpy()
        )
        roots[frame, 2] += stance_root_clearance_lift(
            minimum_sole_clearance_m=(
                solved_soles[..., 2] - solved_surface
            ).min(axis=1),
            support_mask=support,
        )

    feet = foot_kinematics.foot_positions(joints, roots, quaternions)
    soles = sole_kinematics.sole_points(joints, roots, quaternions)
    sole_surface = (
        grid.sample_xy(
            alignment.matcher_to_scene_xy(
                local_torch.tensor(
                    soles[..., :2], dtype=local_torch.float32
                )
            )
        )
        .cpu()
        .numpy()
    )
    clearance = (soles[..., 2] - sole_surface).min(axis=2)
    planned_lift = smooth_swing_clearance_lift(
        support_mask=schedule,
        minimum_sole_clearance_m=clearance,
        smoothing_radius_frames=8,
    )
    for frame in np.flatnonzero((planned_lift > 0.0).any(axis=1)):
        if frame < 4:
            continue
        solve = planned_lift[frame] > 0.0
        targets = feet[frame].copy()
        targets[:, 2] += planned_lift[frame]
        try:
            joints[frame], roots[frame] = swing_retargeter.solve_frame(
                joint_position=joints[frame],
                root_position_world=roots[frame],
                root_orientation_world_wxyz=quaternions[frame],
                solve_feet=solve,
                target_foot_position_world=targets,
                level_feet=np.zeros(2, dtype=np.bool_),
                initial_joint_position=joints[frame - 1],
                initial_root_position_world=roots[frame - 1],
            )
        except ContractError as error:
            raise ContractError(
                f"swing projection frame {frame}: {error}"
            ) from error
    output = qpos.copy()
    native_joints = np.empty_like(joints)
    native_joints[
        ..., np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION)
    ] = joints
    output[:, :3] = roots
    output[:, 7:] = native_joints
    return normalize_generated_qpos(output)


@contextmanager
def _motionbricks_working_directory(root: Path) -> Iterator[None]:
    original = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(original)


def _namespace(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        EXP="default",
        planner="default",
        controller="wasd",
        clips="G1",
        result_dir=str(root / "out"),
        data_root=str(root / "datasets"),
        explicit_dataset_folder=str(root / "datasets" / "motionbricks-G1"),
        humanoid_scene_xml=str(
            root / "assets" / "skeletons" / "g1" / "scene_29dof.xml"
        ),
        skeleton_xml=str(root / "assets" / "skeletons" / "g1" / "g1.xml"),
        clips_ckpt=str(root / "out" / "G1-clip.ckpt"),
        reprocess_clips=False,
        return_model_configs=True,
        return_dataloader=False,
        lookat_movement_direction=False,
        pre_filter_qpos=False,
        source_root_realignment=False,
        target_root_realignment=True,
        force_canonicalization=False,
        skip_ending_target_cond=False,
        random_speed_scale=False,
        speed_scale=[1.0, 1.0],
        use_qpos=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motionbricks", type=Path, required=True)
    parser.add_argument("--proxy-route", type=Path, required=True)
    parser.add_argument(
        "--proxy-endpoints", default="29,73,132,201,232,257"
    )
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.motionbricks.resolve()
    proxy_route = args.proxy_route.resolve()
    target_dataset = args.target_dataset.resolve()
    config = args.config.resolve()
    g1_xml = args.g1_xml.resolve()
    output = args.output.resolve()
    if (
        not (root / "motionbricks").is_dir()
        or not (root / "out" / "G1-clip.ckpt").is_file()
    ):
        raise ContractError("pinned MotionBricks checkout is invalid")
    try:
        endpoints = tuple(
            int(value.strip())
            for value in args.proxy_endpoints.split(",")
            if value.strip()
        )
    except ValueError as error:
        raise ContractError("MotionBricks proxy endpoints are invalid") from error
    if not endpoints:
        raise ContractError("MotionBricks proxy endpoints are invalid")
    with np.load(proxy_route, allow_pickle=False) as archive:
        route = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
        }
    proxies = extract_proxy_keyframes(
        route, endpoint_frames=(3, *endpoints)
    )

    dependency_path = root / ".deps" / "python"
    if dependency_path.is_dir():
        sys.path.insert(0, str(dependency_path))
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    from mm_sonic.torch_g1_fk import MujocoG1FootKinematics
    from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
    from mm_sonic.torch_horizontal_terrain_retarget import (
        WideBoundG1TerrainRetargeter,
    )
    from mm_sonic.torch_terrain_rollout import (
        load_experiment_config,
        resolve_stair_config,
    )

    resolved = resolve_stair_config(
        target_dataset,
        load_experiment_config(config),
        device="cpu",
    )
    foot_kinematics = MujocoG1FootKinematics(g1_xml)
    sole_kinematics = MujocoG1SoleKinematics(g1_xml)
    retargeter = WideBoundG1TerrainRetargeter(
        g1_xml,
        maximum_joint_deviation_rad=1.4,
        maximum_root_height_deviation_m=0.20,
        maximum_root_horizontal_deviation_m=0.04,
        maximum_target_error_m=0.045,
    )
    landing_retargeter = WideBoundG1TerrainRetargeter(
        g1_xml,
        maximum_joint_deviation_rad=1.8,
        maximum_root_height_deviation_m=0.40,
        maximum_root_horizontal_deviation_m=0.25,
        maximum_target_error_m=0.008,
    )
    swing_retargeter = WideBoundG1TerrainRetargeter(
        g1_xml,
        maximum_joint_deviation_rad=1.4,
        maximum_root_height_deviation_m=1.0e-6,
        maximum_root_horizontal_deviation_m=1.0e-6,
        maximum_target_error_m=0.022,
    )
    desired_heading = float(
        _heading_wxyz(proxies.qpos[0, -1:, 3:7])[0]
    )
    all_metrics: list[dict[str, object]] = []
    selected_metrics: list[dict[str, object]] = []
    transitions: list[np.ndarray] = []
    transition_support: list[np.ndarray] = []
    context = proxies.qpos[0].copy()

    with _motionbricks_working_directory(root):
        from motionbricks.motion_backbone.demo.utils import navigation_demo

        print("Loading the released MotionBricks backbone...", flush=True)
        demo = navigation_demo(_namespace(root))
        agent = demo.full_agent
        backbone = agent._inferencer._root_model.backbone_net
        minimum_tokens = int(backbone._args["min_tokens"])
        maximum_tokens = int(backbone._args["max_tokens"])
        masks = allowed_token_masks(minimum_tokens, maximum_tokens)
        for segment_index, target in enumerate(proxies.qpos[1:]):
            records: list[dict[str, object]] = []
            candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            source_start = proxies.endpoint_frames[segment_index]
            source_stop = proxies.endpoint_frames[segment_index + 1]
            is_planned_flight = bool(
                (
                    ~np.asarray(route["source_support_mask"])[
                        source_start : source_stop + 1
                    ].any(axis=1)
                ).any()
            )
            for token_count, mask in masks.items():
                constraints = motionbricks_constraints(
                    agent, context, target
                )
                constraints["allowed_pred_num_tokens"] = mask.to(
                    agent._device
                )
                with torch.no_grad():
                    model_features, generated, frame_count = (
                        agent._generate_inbetween_frames(constraints)
                    )
                count = int(frame_count.item())
                raw = (
                    generated[0, :count]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                candidate = normalize_generated_qpos(
                    np.concatenate((context, raw[4:]), axis=0)
                )
                candidate_id = (
                    f"segment-{segment_index:02d}-tokens-{token_count:02d}"
                )
                raw_candidate = candidate.copy()
                try:
                    initial_support = (
                        proxies.support_mask[segment_index, -1]
                        if segment_index == 0
                        else transition_support[-1][-1]
                    )
                    planned_support = (
                        flight_support_schedule(
                            initial_support=initial_support,
                            target_support=proxies.support_mask[
                                segment_index + 1, -1
                            ],
                            frame_count=len(candidate),
                            context_frames=4,
                            target_window_frames=4,
                        )
                        if is_planned_flight
                        else contact_phase_support(
                            model_features[
                                0,
                                :count,
                                agent._motion_rep.indices["foot_contacts"],
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            initial_support=initial_support,
                            terminal_support=proxies.support_mask[
                                segment_index + 1, -1
                            ],
                            endpoint_window_frames=4,
                        )
                    )
                    if not is_planned_flight:
                        target_joints, target_roots, target_quaternions = (
                            _target_arrays_from_native(candidate)
                        )
                        raw_soles = sole_kinematics.sole_points(
                            target_joints,
                            target_roots,
                            target_quaternions,
                        )
                        raw_surface = (
                            resolved.measurement_extension.query_grid.sample_xy(
                                resolved.measurement_extension.alignment.matcher_to_scene_xy(
                                    torch.tensor(
                                        raw_soles[..., :2],
                                        dtype=torch.float32,
                                    )
                                )
                            )
                            .cpu()
                            .numpy()
                        )
                        raw_clearance = (
                            raw_soles[..., 2] - raw_surface
                        ).min(axis=2)
                        root_lift = terrain_clearance_envelope(
                            minimum_sole_clearance_by_frame_foot=raw_clearance,
                            accepted_clearance_m=-0.020,
                            maximum_correction_step_m=0.025,
                            preserve_stop_endpoint=False,
                        ).max(axis=1)
                        candidate[:, 2] += root_lift
                    candidate = project_candidate_contacts(
                        native_qpos=candidate,
                        target_qpos=target,
                        initial_support=initial_support,
                        target_support=proxies.support_mask[
                            segment_index + 1, -1
                        ],
                        retargeter=(
                            landing_retargeter
                            if int(
                                proxies.support_mask[
                                    segment_index + 1, -1
                                ].sum()
                            )
                            > int(planned_support[0].sum())
                            else retargeter
                        ),
                        swing_retargeter=swing_retargeter,
                        foot_kinematics=foot_kinematics,
                        sole_kinematics=sole_kinematics,
                        measurement_extension=(
                            resolved.measurement_extension
                        ),
                        support_schedule=planned_support,
                    )
                    candidate[:4] = context
                except ContractError as error:
                    print(
                        json.dumps(
                            {
                                "candidate": candidate_id,
                                "rejections": [
                                    f"contact projection: {error}"
                                ],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                metrics, support = candidate_metrics(
                    candidate_id=candidate_id,
                    native_qpos=candidate,
                    target_qpos=target,
                    desired_heading_rad=desired_heading,
                    foot_kinematics=foot_kinematics,
                    sole_kinematics=sole_kinematics,
                    measurement_extension=resolved.measurement_extension,
                    planned_support=planned_support,
                    planned_flight=is_planned_flight,
                )
                metrics["contact_projection"] = True
                metrics.update(
                    contact_projection_metrics(raw_candidate, candidate)
                )
                records.append(metrics)
                candidates[candidate_id] = (candidate, support)
                print(
                    json.dumps(
                        {
                            "candidate": candidate_id,
                            "rejections": metrics["rejections"],
                            "endpoint_root_error_m": metrics[
                                "endpoint_root_error_m"
                            ],
                            "minimum_sole_clearance_m": metrics[
                                "minimum_sole_clearance_m"
                            ],
                            "minimum_sole_clearance_frame": metrics[
                                "minimum_sole_clearance_frame"
                            ],
                            "minimum_sole_clearance_foot": metrics[
                                "minimum_sole_clearance_foot"
                            ],
                            "maximum_stance_contact_error_m": metrics[
                                "maximum_stance_contact_error_m"
                            ],
                            "maximum_stance_horizontal_step_m": metrics[
                                "maximum_stance_horizontal_step_m"
                            ],
                            "maximum_stance_horizontal_step_frame": metrics[
                                "maximum_stance_horizontal_step_frame"
                            ],
                            "maximum_stance_horizontal_step_foot": metrics[
                                "maximum_stance_horizontal_step_foot"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if not records:
                raise ContractError(
                    "MotionBricks contact projection rejected every candidate"
                )
            preferred_frames = preferred_candidate_frame_count(
                source_frame_count=source_stop - source_start,
                source_frames_per_second=50.0,
                output_frames_per_second=30.0,
                available_frame_counts=tuple(
                    token_count * 4 for token_count in masks
                ),
            )
            selected = select_candidate(
                records, preferred_frame_count=preferred_frames
            )
            selected["preferred_frame_count"] = preferred_frames
            candidate, support = candidates[str(selected["candidate_id"])]
            transitions.append(candidate)
            transition_support.append(support)
            selected_metrics.append(selected)
            all_metrics.extend(records)
            context = candidate[-4:].copy()

    assembled = assemble_generated_route(
        context_qpos=proxies.qpos[0],
        context_support=proxies.support_mask[0],
        transition_qpos=tuple(transitions),
        transition_support=tuple(transition_support),
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "traversal.npz", **assembled)
    plan = {
        "schema": "g1-motionbricks-terrain-task-actor/v1",
        "proxy_route": str(proxy_route),
        "proxy_endpoints": list(endpoints),
        "motionbricks_checkout": str(root),
        "frames_per_second": 30.0,
        "selected_candidates": [
            str(metrics["candidate_id"]) for metrics in selected_metrics
        ],
    }
    (output / "proxy-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "candidate-metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": "g1-motionbricks-terrain-task-actor-metrics/v1",
        "frame_count": len(assembled["joint_position"]),
        "segment_count": len(transitions),
        "minimum_sole_clearance_m": min(
            float(value["minimum_sole_clearance_m"])
            for value in selected_metrics
        ),
        "maximum_stance_contact_error_m": max(
            float(value["maximum_stance_contact_error_m"])
            for value in selected_metrics
        ),
        "maximum_stance_horizontal_step_m": max(
            float(value["maximum_stance_horizontal_step_m"])
            for value in selected_metrics
        ),
        "unsupported_frame_count": int(
            (~assembled["source_support_mask"].any(axis=1)).sum()
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
