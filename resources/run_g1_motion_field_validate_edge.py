#!/usr/bin/env python3
"""Validate one object-field edge with flight-aware terrain contact gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from mm_sonic.torch_g1_fk import MujocoG1FootKinematics
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)


def _module():
    path = Path(__file__).with_name("run_g1_motion_field_transitions.py")
    spec = importlib.util.spec_from_file_location("motion_field_transitions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--heading-change-degrees", type=float, required=True)
    parser.add_argument("--observed-heading-change-degrees", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        route = {
            name: np.asarray(archive[name]).copy()
            for name in (
                "joint_position",
                "root_position_world",
                "root_orientation_world_wxyz",
                "source_support_mask",
            )
        }
    resolved = resolve_stair_config(
        args.target_dataset,
        load_experiment_config(args.config),
        device="cpu",
    )
    feet = MujocoG1FootKinematics(args.g1_xml).foot_positions(
        route["joint_position"],
        route["root_position_world"],
        route["root_orientation_world_wxyz"],
    )
    soles = MujocoG1SoleKinematics(args.g1_xml).sole_points(
        route["joint_position"],
        route["root_position_world"],
        route["root_orientation_world_wxyz"],
    )
    extension = resolved.measurement_extension
    scene_xy = extension.alignment.matcher_to_scene_xy(
        torch.tensor(soles[..., :2], dtype=torch.float32)
    )
    clearance = soles[..., 2] - (
        extension.query_grid.sample_xy(scene_xy).cpu().numpy()
    )
    transition = _module()
    metrics = transition.terrain_transition_contact_metrics(
        sole_clearance_m=clearance,
        support_mask=route["source_support_mask"],
    )
    consecutive = (
        route["source_support_mask"][:-1]
        & route["source_support_mask"][1:]
    )
    foot_step = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=2)
    metrics.update(
        maximum_stance_horizontal_step_m=float(
            foot_step[consecutive].max() if consecutive.any() else 0.0
        ),
        maximum_joint_step_rad=float(
            np.abs(np.diff(route["joint_position"], axis=0)).max()
        ),
        maximum_root_step_m=float(
            np.linalg.norm(
                np.diff(route["root_position_world"], axis=0), axis=1
            ).max()
        ),
        heading_change_error_degrees=abs(
            args.observed_heading_change_degrees - args.heading_change_degrees
        ),
    )
    rejections = transition.transition_candidate_rejections(metrics)
    report = {
        "schema": "g1-motion-field-edge-validation/v1",
        "input": str(args.input.resolve()),
        "artifact_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "target_dataset": str(args.target_dataset.resolve()),
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "query_scene": resolved.resolved_config["query_scene"],
        "frame_count": len(route["joint_position"]),
        "metrics": metrics,
        "rejections": list(rejections),
        "validated": not rejections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0 if not rejections else 1


if __name__ == "__main__":
    raise SystemExit(main())
