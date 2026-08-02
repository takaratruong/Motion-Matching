#!/usr/bin/env python3
"""Run and save fixed-horizon GRAIL terrain-skill qualification routes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from mm_sonic.torch_terrain_omni_rollout import save_omni_matrix
from mm_sonic.torch_terrain_omni_routes import same_stair_routes
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)
from mm_sonic.torch_terrain_skill_horizon_rollout import (
    HorizonChunkEvent,
    run_resolved_horizon_matrix,
)
from mm_sonic.torch_terrain_skill_horizon_search import (
    HorizonSearchConfig,
    horizon_search_config_from_experiment,
)
from mm_sonic.torch_terrain_skill_rollout import qualification_routes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--g1-xml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--route", action="append", default=[])
    parser.add_argument("--qualification-slice", action="store_true")
    parser.add_argument(
        "--foot-lock",
        action="store_true",
        help="Pin source-supported feet to the query terrain during playback.",
    )
    parser.add_argument(
        "--contact-phase-gate",
        action="store_true",
        help="Prefer skill entries with the current source support pattern.",
    )
    parser.add_argument(
        "--swing-clearance-margin-m",
        type=float,
        default=None,
        help="Lift unsupported ankles above query terrain by this margin.",
    )
    parser.add_argument(
        "--foot-correction-halflife-s",
        type=float,
        default=0.04,
        help="Inertialization halflife for terrain foot corrections.",
    )
    parser.add_argument(
        "--swing-plan-sigma-frames",
        type=float,
        default=None,
        help="Spread future source-path clearance backward by this sigma.",
    )
    parser.add_argument(
        "--maximum-source-contact-p95-m",
        type=float,
        default=None,
        help="Exclude source clips with worse authenticated contact fit.",
    )
    parser.add_argument(
        "--normalization-source",
        default=None,
        help="Freeze feature normalization to another motion corpus.",
    )
    parser.add_argument(
        "--turning-source-corpus",
        default=None,
        help="Restrict turning chunks to clips present in this corpus.",
    )
    parser.add_argument(
        "--ablation",
        choices=(
            "combined",
            "entry-only",
            "outcome-only",
            "dimension-normalized",
            "legacy-combined",
        ),
        default="combined",
    )
    return parser


def select_routes(names: list[str], use_qualification_slice: bool):
    inventory = same_stair_routes()
    if names and use_qualification_slice:
        raise ValueError("explicit routes and qualification slice are exclusive")
    if use_qualification_slice:
        return qualification_routes(inventory)
    if not names:
        return inventory
    requested = set(names)
    if len(requested) != len(names):
        raise ValueError("route names must not be repeated")
    known = {route.name for route in inventory}
    missing = sorted(requested - known)
    if missing:
        raise ValueError("unknown route names: " + ", ".join(missing))
    return tuple(route for route in inventory if route.name in requested)


def search_config_from_experiment(config: Mapping) -> HorizonSearchConfig:
    try:
        return horizon_search_config_from_experiment(config)
    except Exception as error:
        raise ValueError(str(error)) from error


def canonical_chunk_events(events: Sequence[HorizonChunkEvent]) -> bytes:
    chunks = []
    for index, event in enumerate(events):
        if not isinstance(event, HorizonChunkEvent):
            raise TypeError("chunk events must contain HorizonChunkEvent values")
        chunks.append(
            {
                "chunk_index": index,
                "entry_row": event.entry_row,
                "skill_index": event.skill_index,
                "target_frames": event.target_frames,
                "endpoint_frame_exclusive": event.endpoint_frame_exclusive,
                "entry_cost": event.cost.entry,
                "displacement_cost": event.cost.displacement,
                "yaw_cost": event.cost.yaw,
                "height_cost": event.cost.height,
                "duration_cost": event.cost.duration,
                "stall_cost": event.cost.stall,
                "outcome_cost": event.cost.outcome,
                "total_cost": event.cost.total,
                "rejected_by_reason": dict(event.rejected_by_reason),
                "release_reason": event.release_reason,
            }
        )
    identity_payload = {"schema": "g1-terrain-horizon-chunks/v1", "chunks": chunks}
    identity_bytes = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload = {
        **identity_payload,
        "deterministic_sha256": hashlib.sha256(identity_bytes).hexdigest(),
    }
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def save_horizon_matrix(matrix, events_by_route, output: str | Path) -> None:
    output = Path(os.path.abspath(os.fspath(output)))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"multi-horizon output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    payload = staging_parent / "payload"
    try:
        save_omni_matrix(matrix, payload)
        run_names = {run.route.name for run in matrix.runs}
        if set(events_by_route) != run_names:
            raise ValueError("chunk-event routes do not match matrix routes")
        for route_name, events in events_by_route.items():
            route_dir = payload / "routes" / route_name
            if not route_dir.is_dir():
                raise ValueError(f"matrix route directory is missing: {route_name}")
            (route_dir / "chunk-events.json").write_bytes(
                canonical_chunk_events(events)
            )
        payload.rename(output)
    finally:
        if staging_parent.exists():
            shutil.rmtree(staging_parent)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    search_config = search_config_from_experiment(config)
    if args.ablation == "entry-only":
        search_config = replace(
            search_config,
            displacement_weight=0.0,
            yaw_weight=0.0,
            height_weight=0.0,
            duration_weight=0.0,
            stall_weight=0.0,
        )
    elif args.ablation == "outcome-only":
        search_config = replace(search_config, entry_weight=0.0)
    elif args.ablation == "dimension-normalized":
        search_config = replace(search_config, entry_weight=1.0 / 27.0)
    elif args.ablation == "legacy-combined":
        search_config = replace(search_config, entry_weight=1.0)
    resolved = resolve_stair_config(args.dataset, config, device=args.device)
    matrix, events = run_resolved_horizon_matrix(
        resolved,
        g1_xml=args.g1_xml,
        routes=select_routes(args.route, args.qualification_slice),
        search_config=search_config,
        foot_lock=args.foot_lock,
        contact_phase_gate=args.contact_phase_gate,
        swing_clearance_margin_m=args.swing_clearance_margin_m,
        foot_correction_halflife_s=args.foot_correction_halflife_s,
        swing_plan_sigma_frames=args.swing_plan_sigma_frames,
        maximum_source_contact_p95_m=args.maximum_source_contact_p95_m,
        normalization_source=args.normalization_source,
        turning_source_corpus=args.turning_source_corpus,
    )
    save_horizon_matrix(matrix, events, args.output)
    execution_completed = all(
        run.completed_without_exception for run in matrix.runs
    )
    print(
        json.dumps(
            {
                "execution_completed": execution_completed,
                "matrix_pass": matrix.matrix_pass,
                "route_count": len(matrix.runs),
                "chunk_count": sum(len(values) for values in events.values()),
                "ablation": args.ablation,
                "foot_lock": args.foot_lock,
                "contact_phase_gate": args.contact_phase_gate,
                "swing_clearance_margin_m": args.swing_clearance_margin_m,
                "foot_correction_halflife_s": args.foot_correction_halflife_s,
                "swing_plan_sigma_frames": args.swing_plan_sigma_frames,
                "maximum_source_contact_p95_m": (
                    args.maximum_source_contact_p95_m
                ),
                "normalization_source": args.normalization_source,
                "turning_source_corpus": args.turning_source_corpus,
                "deterministic_sha256": matrix.deterministic_sha256,
                "output": args.output,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if execution_completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
