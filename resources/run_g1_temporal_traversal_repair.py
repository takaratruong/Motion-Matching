#!/usr/bin/env python3
"""Repair isolated IK branch jumps in a contact-valid traversal artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_contact_segments import ANKLE_ORIGIN_SOLE_M
from mm_sonic.torch_g1_fk import MujocoG1FootKinematics
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_horizontal_terrain_retarget import (
    WideBoundG1TerrainRetargeter,
)
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-radius-frames", type=int, default=10)
    parser.add_argument("--joint-spike-threshold-rad", type=float, default=0.35)
    parser.add_argument("--root-spike-threshold-m", type=float, default=0.04)
    return parser


def _spike_windows(
    *,
    joints: object,
    roots: object,
    joint_threshold_rad: float,
    root_threshold_m: float,
    radius_frames: int,
) -> list[tuple[int, int]]:
    joint = np.asarray(joints, dtype=np.float64)
    root = np.asarray(roots, dtype=np.float64)
    if (
        joint.ndim != 2
        or joint.shape[1:] != (29,)
        or root.shape != (len(joint), 3)
        or len(joint) < 3
        or not np.isfinite(joint).all()
        or not np.isfinite(root).all()
        or joint_threshold_rad <= 0.0
        or root_threshold_m <= 0.0
        or type(radius_frames) is not int
        or radius_frames < 1
    ):
        raise ContractError("temporal traversal repair input is invalid")
    joint_step = np.abs(np.diff(joint, axis=0)).max(axis=1)
    root_step = np.linalg.norm(np.diff(root, axis=0), axis=1)
    boundaries = np.flatnonzero(
        (joint_step > joint_threshold_rad)
        | (root_step > root_threshold_m)
    )
    windows = [
        (
            max(0, int(boundary) - radius_frames),
            min(len(joint) - 1, int(boundary) + 1 + radius_frames),
        )
        for boundary in boundaries
    ]
    merged: list[tuple[int, int]] = []
    for start, stop in windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _smooth_window(
    *,
    joints: np.ndarray,
    roots: np.ndarray,
    quaternions: np.ndarray,
    start: int,
    stop: int,
) -> None:
    if not 0 <= start < stop < len(joints):
        raise ContractError("temporal traversal window is invalid")
    linear = np.linspace(0.0, 1.0, stop - start + 1)
    smooth = linear * linear * (3.0 - 2.0 * linear)
    joints[start : stop + 1] = (
        (1.0 - smooth[:, None]) * joints[start]
        + smooth[:, None] * joints[stop]
    )
    roots[start : stop + 1] = (
        (1.0 - smooth[:, None]) * roots[start]
        + smooth[:, None] * roots[stop]
    )
    first = quaternions[start].copy()
    final = quaternions[stop].copy()
    if float(np.dot(first, final)) < 0.0:
        final *= -1.0
    values = (
        (1.0 - smooth[:, None]) * first + smooth[:, None] * final
    )
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    quaternions[start : stop + 1] = values


def _terrain_clearance(
    *,
    joints: np.ndarray,
    roots: np.ndarray,
    quaternions: np.ndarray,
    alignment: object,
    target_grid: object,
    foot_kinematics: MujocoG1FootKinematics,
    sole_kinematics: MujocoG1SoleKinematics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ankles = foot_kinematics.foot_positions(joints, roots, quaternions)
    soles = sole_kinematics.sole_points(joints, roots, quaternions)
    ankle_scene = alignment.matcher_to_scene_xy(
        torch.tensor(ankles[..., :2], dtype=torch.float32)
    )
    sole_scene = alignment.matcher_to_scene_xy(
        torch.tensor(soles[..., :2], dtype=torch.float32)
    )
    ankle_surface = target_grid.sample_xy(ankle_scene).cpu().numpy()
    sole_surface = target_grid.sample_xy(sole_scene).cpu().numpy()
    return (
        ankles[..., 2] - ankle_surface - float(ANKLE_ORIGIN_SOLE_M),
        soles[..., 2] - sole_surface,
        ankles,
    )


def main() -> int:
    args = _parser().parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    required = (
        "joint_position",
        "root_position_world",
        "root_orientation_world_wxyz",
        "source_support_mask",
    )
    if any(name not in arrays for name in required):
        raise ContractError("temporal traversal artifact is incomplete")
    joints = arrays["joint_position"]
    roots = arrays["root_position_world"]
    quaternions = arrays["root_orientation_world_wxyz"]
    support = arrays["source_support_mask"]
    if (
        quaternions.shape != (len(joints), 4)
        or support.shape != (len(joints), 2)
        or support.dtype != np.bool_
        or not bool(support.any(axis=1).all())
    ):
        raise ContractError("temporal traversal artifact arrays are invalid")
    windows = _spike_windows(
        joints=joints,
        roots=roots,
        joint_threshold_rad=args.joint_spike_threshold_rad,
        root_threshold_m=args.root_spike_threshold_m,
        radius_frames=args.window_radius_frames,
    )
    if not windows:
        raise ContractError("temporal traversal contains no repairable spike")

    resolved = resolve_stair_config(
        args.target_dataset,
        load_experiment_config(args.config),
        device="cpu",
    )
    foot_kinematics = MujocoG1FootKinematics(args.g1_xml)
    sole_kinematics = MujocoG1SoleKinematics(args.g1_xml)
    original_ankles = foot_kinematics.foot_positions(
        joints, roots, quaternions
    )
    original_ankle_clearance, _, _ = _terrain_clearance(
        joints=joints,
        roots=roots,
        quaternions=quaternions,
        alignment=resolved.measurement_extension.alignment,
        target_grid=resolved.measurement_extension.query_grid,
        foot_kinematics=foot_kinematics,
        sole_kinematics=sole_kinematics,
    )
    support_targets = original_ankles.copy()
    support_targets[..., 2] -= np.where(
        support, original_ankle_clearance, 0.0
    )
    for start, stop in windows:
        _smooth_window(
            joints=joints,
            roots=roots,
            quaternions=quaternions,
            start=start,
            stop=stop,
        )

    retargeter = WideBoundG1TerrainRetargeter(
        args.g1_xml,
        maximum_root_height_deviation_m=0.12,
        maximum_root_horizontal_deviation_m=1.0e-6,
        maximum_target_error_m=0.040,
    )
    swing_retargeter = WideBoundG1TerrainRetargeter(
        args.g1_xml,
        maximum_root_height_deviation_m=1.0e-6,
        maximum_root_horizontal_deviation_m=1.0e-6,
        maximum_target_error_m=0.040,
    )
    for start, stop in windows:
        for frame in range(start, stop + 1):
            joints[frame], roots[frame] = retargeter.solve_frame(
                joint_position=joints[frame],
                root_position_world=roots[frame],
                root_orientation_world_wxyz=quaternions[frame],
                solve_feet=support[frame],
                target_foot_position_world=support_targets[frame],
                level_feet=support[frame],
                initial_joint_position=(
                    None if frame == start else joints[frame - 1]
                ),
                initial_root_position_world=(
                    None if frame == start else roots[frame - 1]
                ),
            )
    normalization_order = range(len(joints))
    repaired_ankle_clearance, _, _ = _terrain_clearance(
        joints=joints,
        roots=roots,
        quaternions=quaternions,
        alignment=resolved.measurement_extension.alignment,
        target_grid=resolved.measurement_extension.query_grid,
        foot_kinematics=foot_kinematics,
        sole_kinematics=sole_kinematics,
    )
    for frame in normalization_order:
        roots[frame, 2] -= float(
            repaired_ankle_clearance[frame, support[frame]].mean()
        )

    swing_corrected = 0
    for frame in normalization_order:
        for iteration in range(3):
            _, sole_clearance, ankles = _terrain_clearance(
                joints=joints[frame : frame + 1],
                roots=roots[frame : frame + 1],
                quaternions=quaternions[frame : frame + 1],
                alignment=resolved.measurement_extension.alignment,
                target_grid=resolved.measurement_extension.query_grid,
                foot_kinematics=foot_kinematics,
                sole_kinematics=sole_kinematics,
            )
            minimum = sole_clearance[0].min(axis=1)
            solve = (~support[frame]) & (minimum < -0.025)
            if not bool(solve.any()):
                break
            targets = ankles[0].copy()
            targets[solve, 2] += 0.005 - minimum[solve]
            joints[frame], roots[frame] = swing_retargeter.solve_frame(
                joint_position=joints[frame],
                root_position_world=roots[frame],
                root_orientation_world_wxyz=quaternions[frame],
                solve_feet=solve,
                target_foot_position_world=targets,
                level_feet=np.zeros(2, dtype=np.bool_),
                initial_joint_position=(
                    None if frame == 0 else joints[frame - 1]
                ),
                initial_root_position_world=(
                    None if frame == 0 else roots[frame - 1]
                ),
            )
            swing_corrected += 1

    ankle_clearance, sole_clearance, _ = _terrain_clearance(
        joints=joints,
        roots=roots,
        quaternions=quaternions,
        alignment=resolved.measurement_extension.alignment,
        target_grid=resolved.measurement_extension.query_grid,
        foot_kinematics=foot_kinematics,
        sole_kinematics=sole_kinematics,
    )
    supported_points = (
        (sole_clearance >= -0.025) & (sole_clearance <= 0.035)
    ).sum(axis=2)
    metrics = {
        "schema": "g1-temporal-traversal-repair/v1",
        "frame_count": len(joints),
        "repair_windows": [list(window) for window in windows],
        "swing_corrected_solve_count": swing_corrected,
        "maximum_joint_step_rad": float(
            np.abs(np.diff(joints, axis=0)).max()
        ),
        "maximum_root_step_m": float(
            np.linalg.norm(np.diff(roots, axis=0), axis=1).max()
        ),
        "maximum_stance_contact_error_m": float(
            np.abs(ankle_clearance[support]).max()
        ),
        "minimum_sole_clearance_m": float(sole_clearance.min()),
        "minimum_supported_sole_points": int(
            supported_points[support].min()
        ),
    }
    if metrics["maximum_stance_contact_error_m"] > 0.020:
        raise ContractError(
            "temporal repair loses stance contact: "
            f"{metrics['maximum_stance_contact_error_m']:.6f} m"
        )
    if metrics["minimum_sole_clearance_m"] < -0.025:
        raise ContractError("temporal repair penetrates the staircase")
    if metrics["minimum_supported_sole_points"] < 3:
        raise ContractError("temporal repair has incomplete sole support")
    arrays["minimum_sole_clearance_by_frame"] = sole_clearance.min(
        axis=(1, 2)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "traversal.npz", **arrays)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
