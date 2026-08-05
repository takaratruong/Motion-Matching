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


def _enforce_metrics(
    metrics: dict[str, int | float],
    *,
    maximum_unsupported_frames: int = 0,
    minimum_supported_sole_points: int = 3,
) -> None:
    if (
        type(maximum_unsupported_frames) is not int
        or maximum_unsupported_frames < 0
        or type(minimum_supported_sole_points) is not int
        or minimum_supported_sole_points < 1
    ):
        raise ContractError("validation allowance is invalid")
    if (
        int(metrics["unsupported_frame_count"])
        > maximum_unsupported_frames
    ):
        raise ContractError("traversal contains an unsupported frame")
    if float(metrics["maximum_stance_contact_error_m"]) > 0.020:
        raise ContractError("traversal loses stance contact")
    if float(metrics["maximum_stance_horizontal_step_m"]) > 0.010:
        raise ContractError("traversal stance foot slides horizontally")
    if float(metrics["minimum_sole_clearance_m"]) < -0.025:
        raise ContractError("traversal penetrates the staircase")
    if (
        int(metrics["minimum_supported_sole_points"])
        < minimum_supported_sole_points
    ):
        raise ContractError("traversal has incomplete sole support")
    if (
        "terminal_complete_support" in metrics
        and not bool(metrics["terminal_complete_support"])
    ):
        raise ContractError("traversal terminal phase is mid-swing")
    if (
        "provenance_coverage" in metrics
        and float(metrics["provenance_coverage"]) < 1.0
    ):
        raise ContractError("traversal provenance is incomplete")
    if (
        "maximum_heading_error_degrees" in metrics
        and float(metrics["maximum_heading_error_degrees"]) > 35.0
    ):
        raise ContractError("traversal heading error is excessive")
    if (
        "progress_error_m" in metrics
        and float(metrics["progress_error_m"]) > 0.20
    ):
        raise ContractError("traversal progress error is excessive")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-unsupported-frames", type=int, default=0)
    parser.add_argument("--minimum-supported-sole-points", type=int, default=3)
    parser.add_argument("--expected-heading-degrees", type=float)
    parser.add_argument("--planned-footprints", type=Path)
    return parser


def _yaw_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def main() -> int:
    args = parser().parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        joints = np.asarray(archive["joint_position"]).copy()
        roots = np.asarray(archive["root_position_world"]).copy()
        quaternions = np.asarray(
            archive["root_orientation_world_wxyz"]
        ).copy()
        support = np.asarray(archive["source_support_mask"]).copy()
        provenance = (
            np.asarray(archive["source_frame_provenance"]).copy()
            if "source_frame_provenance" in archive
            else np.empty((0, 2), dtype=np.int64)
        )
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
    stance_sole_error = np.max(np.abs(sole_clearance), axis=2)
    supported_stance_error = np.where(
        support, stance_sole_error, -np.inf
    )
    worst_flat = int(np.argmax(supported_stance_error))
    worst_frame, worst_foot = np.unravel_index(
        worst_flat, supported_stance_error.shape
    )
    provenance_valid = (
        provenance.shape == (frame_count, 2)
        and provenance.dtype.kind in "iu"
    )
    provenance_coverage = (
        float((provenance >= 0).all(axis=1).mean())
        if provenance_valid
        else 0.0
    )
    metrics: dict[str, int | float | str | bool] = {
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
        "maximum_complete_sole_contact_error_m": float(
            stance_sole_error[support].max()
        ),
        "worst_contact_frame": int(worst_frame),
        "worst_contact_foot": int(worst_foot),
        "terminal_complete_support": bool(support[-1].all()),
        "provenance_coverage": provenance_coverage,
    }
    footprint_records = None
    if args.planned_footprints is not None:
        try:
            footprint_records = json.loads(
                args.planned_footprints.read_text(encoding="utf-8")
            )
        except Exception as error:
            raise ContractError(
                "planned footprint artifact is invalid"
            ) from error
        if (
            not isinstance(footprint_records, list)
            or not footprint_records
        ):
            raise ContractError("planned footprint artifact is invalid")
        feet_scene = alignment.matcher_to_scene_xy(
            torch.tensor(feet[..., :2], dtype=torch.float32)
        ).cpu().numpy()
        footprint_errors = []
        for record in footprint_records:
            try:
                foot = int(record["foot"])
                center = np.asarray(
                    record["center_scene_xy"], dtype=np.float64
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ContractError(
                    "planned footprint artifact is invalid"
                ) from error
            if foot not in (0, 1) or center.shape != (2,):
                raise ContractError(
                    "planned footprint artifact is invalid"
                )
            eligible = support[:, foot]
            error = np.linalg.norm(
                feet_scene[eligible, foot] - center, axis=1
            )
            footprint_errors.append(float(error.min()))
        metrics["worst_footprint"] = int(np.argmax(footprint_errors))
        metrics["maximum_footprint_center_error_m"] = float(
            max(footprint_errors)
        )
    if args.expected_heading_degrees is not None:
        if not np.isfinite(args.expected_heading_degrees):
            raise ContractError("expected heading is invalid")
        yaw_scene = (
            _yaw_from_wxyz(quaternions)
            + float(alignment.yaw_scene_from_matcher.item())
        )
        expected = np.deg2rad(args.expected_heading_degrees)
        error = np.arctan2(
            np.sin(yaw_scene - expected), np.cos(yaw_scene - expected)
        )
        metrics["maximum_heading_error_degrees"] = float(
            np.rad2deg(np.max(np.abs(error)))
        )
        heading = np.array(
            (np.cos(expected), np.sin(expected)), dtype=np.float64
        )
        root_scene = alignment.matcher_to_scene_xy(
            torch.tensor(roots[:, :2], dtype=torch.float32)
        ).cpu().numpy()
        observed_progress = float(
            (root_scene[-1] - root_scene[0]) @ heading
        )
        metrics["observed_progress_m"] = observed_progress
        if footprint_records is not None:
            expected_progress = max(
                float(record["center_heading_xy"][0])
                for record in footprint_records
            )
            metrics["expected_progress_m"] = expected_progress
            metrics["progress_error_m"] = abs(
                observed_progress - expected_progress
            )
    _enforce_metrics(
        metrics,
        maximum_unsupported_frames=args.maximum_unsupported_frames,
        minimum_supported_sole_points=(
            args.minimum_supported_sole_points
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
