#!/usr/bin/env python3
"""Validate a G1 traversal against the target terrain and contact schedule."""

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
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)


def _enforce_metrics(metrics: dict[str, int | float]) -> None:
    if int(metrics["unsupported_frame_count"]) != 0:
        raise ContractError("traversal contains an unsupported frame")
    if float(metrics["maximum_stance_contact_error_m"]) > 0.020:
        raise ContractError("traversal loses stance contact")
    if float(metrics["maximum_stance_horizontal_step_m"]) > 0.010:
        raise ContractError("traversal stance foot slides horizontally")
    if float(metrics["minimum_sole_clearance_m"]) < -0.025:
        raise ContractError("traversal penetrates the staircase")
    if int(metrics["minimum_supported_sole_points"]) < 3:
        raise ContractError("traversal has incomplete sole support")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        joints = np.asarray(archive["joint_position"]).copy()
        roots = np.asarray(archive["root_position_world"]).copy()
        quaternions = np.asarray(
            archive["root_orientation_world_wxyz"]
        ).copy()
        support = np.asarray(archive["source_support_mask"]).copy()
    frame_count = len(joints)
    if (
        frame_count < 2
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
            np.abs(np.linalg.norm(quaternions, axis=1) - 1.0) > 1.0e-4
        )
    ):
        raise ContractError("traversal artifact arrays are invalid")

    resolved = resolve_stair_config(
        args.target_dataset,
        load_experiment_config(args.config),
        device="cpu",
    )
    feet = MujocoG1FootKinematics(args.g1_xml).foot_positions(
        joints, roots, quaternions
    )
    soles = MujocoG1SoleKinematics(args.g1_xml).sole_points(
        joints, roots, quaternions
    )
    alignment = resolved.measurement_extension.alignment
    grid = resolved.measurement_extension.query_grid
    foot_surface = grid.sample_xy(
        alignment.matcher_to_scene_xy(
            torch.tensor(feet[..., :2], dtype=torch.float32)
        )
    ).cpu().numpy()
    sole_surface = grid.sample_xy(
        alignment.matcher_to_scene_xy(
            torch.tensor(soles[..., :2], dtype=torch.float32)
        )
    ).cpu().numpy()
    foot_clearance = (
        feet[..., 2] - foot_surface - float(ANKLE_ORIGIN_SOLE_M)
    )
    sole_clearance = soles[..., 2] - sole_surface
    supported_points = (
        (sole_clearance >= -0.025) & (sole_clearance <= 0.035)
    ).sum(axis=2)
    consecutive_support = support[:-1] & support[1:]
    horizontal_foot_step = np.linalg.norm(
        np.diff(feet[..., :2], axis=0), axis=2
    )
    metrics: dict[str, int | float | str] = {
        "schema": "g1-terrain-traversal-validation/v1",
        "frame_count": frame_count,
        "unsupported_frame_count": int((~support.any(axis=1)).sum()),
        "maximum_joint_step_rad": float(
            np.abs(np.diff(joints, axis=0)).max()
        ),
        "maximum_root_step_m": float(
            np.linalg.norm(np.diff(roots, axis=0), axis=1).max()
        ),
        "maximum_stance_contact_error_m": float(
            np.abs(foot_clearance[support]).max()
        ),
        "maximum_stance_horizontal_step_m": float(
            horizontal_foot_step[consecutive_support].max()
            if bool(consecutive_support.any())
            else 0.0
        ),
        "minimum_sole_clearance_m": float(sole_clearance.min()),
        "minimum_supported_sole_points": int(
            supported_points[support].min()
        ),
    }
    _enforce_metrics(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
