#!/usr/bin/env python3
"""Slice and locally repair a route using its authentic support schedule."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_terrain_rollout import load_experiment_config, resolve_stair_config


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
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--search-radius-m", type=float, default=0.02)
    parser.add_argument("--search-step-m", type=float, default=0.002)
    parser.add_argument("--repair-mode", choices=("xy", "z"), default="xy")
    parser.add_argument("--maximum-clearance-lift-m", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        stop = args.stop_frame
        window = slice(args.start_frame, stop)
        route = {
            name: np.asarray(archive[name][window]).copy()
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
    soles = MujocoG1SoleKinematics(args.g1_xml).sole_points(
        route["joint_position"],
        route["root_position_world"],
        route["root_orientation_world_wxyz"],
    )
    extension = resolved.measurement_extension

    def matcher_to_scene(points: np.ndarray) -> np.ndarray:
        return extension.alignment.matcher_to_scene_xy(
            torch.tensor(points, dtype=torch.float32)
        ).numpy()

    def surface(points: np.ndarray) -> np.ndarray:
        return extension.query_grid.sample_xy(
            torch.tensor(points, dtype=torch.float32)
        ).numpy()

    transition = _module()
    if args.repair_mode == "xy":
        correction = transition.minimum_root_xy_contact_correction_schedule(
            sole_position_matcher=soles,
            support_mask=route["source_support_mask"],
            matcher_to_scene_xy=matcher_to_scene,
            sample_scene_surface=surface,
            search_radius_m=args.search_radius_m,
            search_step_m=args.search_step_m,
        )
        repaired = transition.apply_root_xy_correction_schedule(
            route=route, correction_xy_m=correction
        )
    else:
        scene_xy = matcher_to_scene(soles[..., :2])
        clearance = soles[..., 2] - surface(scene_xy)
        correction = transition.minimum_root_z_clearance_correction_schedule(
            sole_clearance_m=clearance,
            support_mask=route["source_support_mask"],
            maximum_lift_m=args.maximum_clearance_lift_m,
            search_step_m=args.search_step_m,
        )
        repaired = transition.apply_root_z_correction_schedule(
            route=route, correction_z_m=correction
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **repaired)
    correction_magnitude = (
        np.linalg.norm(correction, axis=1)
        if correction.ndim == 2
        else np.abs(correction)
    )
    record = {
        "schema": "g1-motion-field-route-contact-repair/v1",
        "input": str(args.input.resolve()),
        "source_frames": [args.start_frame, args.stop_frame],
        "repair_mode": args.repair_mode,
        "correction": correction.tolist(),
        "corrected_frame_count": int(
            (correction_magnitude > 0.0).sum()
        ),
        "maximum_correction_m": float(correction_magnitude.max()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
