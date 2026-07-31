#!/usr/bin/env python3
"""Run the bounded terrain-transition diagnostic on frozen side exits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from mm_sonic.torch_motion_matcher import TorchMotionMatcher
from mm_sonic.torch_terrain_features import (
    TerrainFootClearanceValidator,
    TerrainTransitionTerminalEvaluator,
)
from mm_sonic.torch_terrain_omni_rollout import fit_resolved_normalization
from mm_sonic.torch_terrain_omni_routes import (
    StairFrame,
    same_stair_routes,
)
from mm_sonic.torch_terrain_reachability_rollout import (
    run_route_reachability,
)
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    matcher_config_from_resolved,
    resolve_stair_config,
    terrain_transition_validator_from_resolved,
)
from mm_sonic.torch_transition_reachability import ReachabilityLimits


DEFAULT_ROUTES = (
    "side-exit-lower-left",
    "side-exit-lower-right",
    "side-exit-upper-left",
    "side-exit-upper-right",
)
EVIDENCE_LIMITS = ReachabilityLimits(
    max_depth=2,
    beam_width=4,
    max_expanded_states=64,
    max_source_advance_frames=15,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--normalization-dataset", required=True)
    parser.add_argument("--normalization-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help="Run only this side-exit route; repeat to select several.",
    )
    return parser


def _normalization_sha256(normalization) -> str:
    mean, scale = normalization.parameters_copy()
    digest = hashlib.sha256()
    digest.update(b"g1-motion-feature-normalization/v1")
    digest.update(mean.detach().cpu().numpy().tobytes())
    digest.update(scale.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _write_json_atomic(path: str | Path, payload: dict) -> None:
    destination = Path(os.path.abspath(os.fspath(path)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved = resolve_stair_config(
        args.dataset,
        load_experiment_config(args.config),
        device=args.device,
    )
    normalization_resolved = resolve_stair_config(
        args.normalization_dataset,
        load_experiment_config(args.normalization_config),
        device=args.device,
    )
    normalization = fit_resolved_normalization(normalization_resolved)
    runtime_safe = terrain_transition_validator_from_resolved(resolved)
    minimum_clearance = float(
        resolved.resolved_config["acceptance"][
            "minimum_foot_clearance_m"
        ]
    )
    full_window_safe = TerrainFootClearanceValidator(
        extension=resolved.measurement_extension,
        preview_steps=46,
        minimum_clearance_m=minimum_clearance,
    )
    terminal = TerrainTransitionTerminalEvaluator(
        extension=resolved.measurement_extension,
        horizon_steps=46,
        minimum_command_progress_m=0.05,
        minimum_surface_drop_m=0.05,
        minimum_clearance_m=minimum_clearance,
    )

    def matcher() -> TorchMotionMatcher:
        return TorchMotionMatcher.from_folder(
            resolved.dataset.root,
            device=args.device,
            config=matcher_config_from_resolved(
                resolved.resolved_config
            ),
            extension=resolved.measurement_extension,
            reset_clip_path=resolved.resolved_config["reset_clip"],
            emitted_window_validator=runtime_safe,
            normalization_override=normalization,
        )

    observed_matcher = matcher()
    control_matcher = matcher()
    direction = resolved.resolved_config[
        "reference_direction_matcher_xy"
    ]
    stair_frame = StairFrame(
        origin_world_xy=(0.0, 0.0),
        ascent_world_yaw=math.atan2(
            float(direction[1]), float(direction[0])
        ),
        width_m=0.6223,
        tread_depth_m=0.3302,
        riser_height_m=0.1778,
        tread_count=3,
    )
    inventory = {route.name: route for route in same_stair_routes()}
    names = tuple(args.route or DEFAULT_ROUTES)
    missing = sorted(set(names) - set(inventory))
    if missing:
        raise ValueError(
            f"unknown omnidirectional routes: {', '.join(missing)}"
        )

    runs = []
    for name in names:
        run = run_route_reachability(
            inventory[name],
            stair_frame=stair_frame,
            matcher=observed_matcher,
            control_matcher=control_matcher,
            terminal_evaluator=terminal,
            safe_evaluator=full_window_safe,
            limits=EVIDENCE_LIMITS,
        )
        runs.append(
            {
                "route": run.route_name,
                "completed_frames": run.completed_frames,
                "behavior_unchanged": run.behavior_unchanged,
                "observed_output_sha256": (
                    run.observed_output_sha256
                ),
                "control_output_sha256": run.control_output_sha256,
                "diagnostics": [
                    asdict(diagnostic)
                    for diagnostic in run.diagnostics
                ],
            }
        )

    payload = {
        "schema": "g1-terrain-transition-reachability-evidence/v1",
        "dataset_identity": resolved.dataset.manifest_sha256,
        "base_config_sha256": resolved.base_config_sha256,
        "normalization_sha256": _normalization_sha256(normalization),
        "runtime_safety_preview_steps": runtime_safe.preview_steps,
        "diagnostic_safety_preview_steps": 46,
        "limits": asdict(EVIDENCE_LIMITS),
        "terminal": {
            "horizon_steps": 46,
            "minimum_command_progress_m": 0.05,
            "minimum_surface_drop_per_foot_m": 0.05,
            "maximum_terminal_sole_error_m": (
                terminal.maximum_terminal_sole_error_m
            ),
            "same_frame_double_support_drop_progress": True,
            "final_single_support": True,
        },
        "runs": runs,
    }
    _write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "status": "PUBLISHED",
                "route_count": len(runs),
                "all_behavior_unchanged": all(
                    run["behavior_unchanged"] for run in runs
                ),
                "reachable_count": sum(
                    diagnostic["reachable"]
                    for run in runs
                    for diagnostic in run["diagnostics"]
                ),
                "output": str(Path(args.output).resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
