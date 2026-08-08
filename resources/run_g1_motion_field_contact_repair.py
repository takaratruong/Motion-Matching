#!/usr/bin/env python3
"""Repair quantization-scale terrain contact misses without pose IK."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from mm_sonic.joints import PINNED_TARGET_TO_SOURCE_PERMUTATION
from mm_sonic.torch_g1_fk import MujocoG1FootKinematics
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)


def _transition_module():
    path = Path(__file__).with_name("run_g1_motion_field_transitions.py")
    spec = importlib.util.spec_from_file_location(
        "run_g1_motion_field_transitions", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--target-dataset", type=Path, required=True)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--g1-xml", type=Path, required=True)
    result.add_argument("--support-foot", choices=("left", "right"), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--incoming", type=Path)
    result.add_argument("--successor", type=Path)
    result.add_argument("--assembled-output", type=Path)
    result.add_argument("--search-radius-m", type=float, default=0.02)
    result.add_argument("--search-step-m", type=float, default=0.002)
    result.add_argument("--prefix-blend-stop-frame", type=int)
    result.add_argument("--terminal-target", type=Path)
    result.add_argument("--terminal-target-frame", type=int, default=0)
    result.add_argument("--terminal-blend-start-frame", type=int)
    return result


def main() -> int:
    args = parser().parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        native = np.asarray(archive["native_qpos"], dtype=np.float64)
    permutation = np.asarray(
        PINNED_TARGET_TO_SOURCE_PERMUTATION, dtype=np.int64
    )
    support = np.zeros((len(native), 2), dtype=np.bool_)
    support[:, 0 if args.support_foot == "left" else 1] = True
    route = {
        "joint_position": np.ascontiguousarray(native[:, 7:][:, permutation]),
        "root_position_world": np.ascontiguousarray(native[:, :3]),
        "root_orientation_world_wxyz": np.ascontiguousarray(native[:, 3:7]),
        "source_support_mask": support,
    }

    transition = _transition_module()
    if args.prefix_blend_stop_frame is not None:
        route = transition.stance_anchored_prefix_blend(
            route=route,
            stance_foot=0 if args.support_foot == "left" else 1,
            blend_stop_frame=args.prefix_blend_stop_frame,
            foot_kinematics=MujocoG1FootKinematics(args.g1_xml),
        )
    terminal_arguments = (
        args.terminal_target,
        args.terminal_blend_start_frame,
    )
    if any(value is not None for value in terminal_arguments):
        if not all(value is not None for value in terminal_arguments):
            raise ValueError(
                "terminal-target and terminal-blend-start-frame must be paired"
            )
        with np.load(args.terminal_target, allow_pickle=False) as archive:
            target_frame = args.terminal_target_frame
            route = transition.stance_anchored_terminal_blend(
                route=route,
                target_joint_position=archive["joint_position"][target_frame],
                target_root_position_world=archive["root_position_world"][
                    target_frame
                ],
                target_root_orientation_world_wxyz=archive[
                    "root_orientation_world_wxyz"
                ][target_frame],
                stance_foot=0 if args.support_foot == "left" else 1,
                blend_start_frame=args.terminal_blend_start_frame,
                foot_kinematics=MujocoG1FootKinematics(args.g1_xml),
            )

    resolved = resolve_stair_config(
        args.target_dataset,
        load_experiment_config(args.config),
        device="cpu",
    )
    extension = resolved.measurement_extension
    soles = MujocoG1SoleKinematics(args.g1_xml).sole_points(
        route["joint_position"],
        route["root_position_world"],
        route["root_orientation_world_wxyz"],
    )

    def matcher_to_scene(points: np.ndarray) -> np.ndarray:
        return (
            extension.alignment.matcher_to_scene_xy(
                torch.tensor(points, dtype=torch.float32)
            )
            .cpu()
            .numpy()
        )

    def sample_surface(points: np.ndarray) -> np.ndarray:
        return (
            extension.query_grid.sample_xy(
                torch.tensor(points, dtype=torch.float32)
            )
            .cpu()
            .numpy()
        )

    correction = transition.minimum_root_xy_contact_correction_schedule(
        sole_position_matcher=soles,
        support_mask=support,
        matcher_to_scene_xy=matcher_to_scene,
        sample_scene_surface=sample_surface,
        search_radius_m=args.search_radius_m,
        search_step_m=args.search_step_m,
    )
    repaired = transition.apply_root_xy_correction_schedule(
        route=route, correction_xy_m=correction
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **repaired)
    assembly_paths = (args.incoming, args.successor, args.assembled_output)
    if any(path is not None for path in assembly_paths):
        if not all(path is not None for path in assembly_paths):
            raise ValueError(
                "incoming, successor, and assembled-output must be used together"
            )

        def load_route(path: Path) -> dict[str, np.ndarray]:
            with np.load(path, allow_pickle=False) as archive:
                return {
                    name: np.asarray(archive[name]).copy()
                    for name in (
                        "joint_position",
                        "root_position_world",
                        "root_orientation_world_wxyz",
                        "source_support_mask",
                    )
                }

        assembled = transition.assemble_exact_successor_edge(
            incoming_route=load_route(args.incoming),
            connector=repaired,
            successor=load_route(args.successor),
        )
        args.assembled_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.assembled_output, **assembled)
    record = {
        "method": "minimum-rigid-root-xy-contact-correction",
        "input": str(args.input.resolve()),
        "support_foot": args.support_foot,
        "search_radius_m": args.search_radius_m,
        "search_step_m": args.search_step_m,
        "prefix_blend_stop_frame": args.prefix_blend_stop_frame,
        "terminal_target": (
            str(args.terminal_target.resolve())
            if args.terminal_target is not None
            else None
        ),
        "terminal_blend_start_frame": args.terminal_blend_start_frame,
        "corrected_frame_count": int(
            (np.linalg.norm(correction, axis=1) > 0.0).sum()
        ),
        "maximum_correction_m": float(
            np.linalg.norm(correction, axis=1).max()
        ),
        "correction_xy_m": correction.tolist(),
        "assembled_output": (
            str(args.assembled_output.resolve())
            if args.assembled_output is not None
            else None
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
