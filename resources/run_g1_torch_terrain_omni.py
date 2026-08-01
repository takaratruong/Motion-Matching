#!/usr/bin/env python3
"""Run and save the deterministic same-stair omnidirectional matrix."""

from __future__ import annotations

import argparse
import json

from mm_sonic.torch_terrain_omni_rollout import (
    fit_resolved_normalization,
    run_resolved_omni_matrix,
    save_omni_matrix,
)
from mm_sonic.torch_terrain_omni_routes import same_stair_routes
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--g1-xml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--contact-segments",
        action="store_true",
        help="Load the contact-segment experiment schema and policy.",
    )
    parser.add_argument(
        "--foothold-arm",
        choices=(
            "first-contact",
            "two-contact",
            "hybrid",
            "continuous-control",
        ),
        help="Condition contact-segment entries on this foothold ablation arm.",
    )
    parser.add_argument(
        "--maximum-step-time-ms",
        type=float,
        default=1000.0,
        help="Fail a route after a matcher step exceeds this latency.",
    )
    parser.add_argument(
        "--normalization-dataset",
        help="Optional reference corpus whose feature normalization is frozen.",
    )
    parser.add_argument(
        "--normalization-config",
        help="Experiment config paired with --normalization-dataset.",
    )
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help="Run only this route name; repeat to select multiple routes.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inventory = same_stair_routes()
    if args.route:
        by_name = {route.name: route for route in inventory}
        missing = sorted(set(args.route) - set(by_name))
        if missing:
            raise ValueError(f"unknown route names: {', '.join(missing)}")
        routes = tuple(by_name[name] for name in args.route)
    else:
        routes = inventory
    contact_policy = None
    foothold_policy = None
    if args.contact_segments:
        from mm_sonic.torch_contact_segment_rollout import (
            build_contact_segment_policy,
            load_contact_segment_config,
            resolve_contact_segment_config,
        )

        resolved = resolve_contact_segment_config(
            args.dataset,
            load_contact_segment_config(args.config),
            device=args.device,
        )
        contact_policy = build_contact_segment_policy(resolved, args.g1_xml)
    else:
        resolved = resolve_stair_config(
            args.dataset,
            load_experiment_config(args.config),
            device=args.device,
        )
    if args.foothold_arm and contact_policy is None:
        raise ValueError("--foothold-arm requires --contact-segments")
    if args.foothold_arm:
        from mm_sonic.torch_foothold_actions import (
            FootholdActionIndex,
            FootholdActionPolicy,
            FootholdSelectionArm,
        )

        foothold_policy = FootholdActionPolicy(
            index=FootholdActionIndex.from_dataset(
                resolved.dataset, contact_policy.index
            ),
            extension=resolved.measurement_extension,
            arm=FootholdSelectionArm(args.foothold_arm),
        )
    if args.maximum_step_time_ms <= 0.0:
        raise ValueError("maximum step time must be positive")
    if bool(args.normalization_dataset) != bool(args.normalization_config):
        raise ValueError(
            "normalization dataset and config must be supplied together"
        )
    normalization = None
    if args.normalization_dataset:
        normalization_resolved = resolve_stair_config(
            args.normalization_dataset,
            load_experiment_config(args.normalization_config),
            device=args.device,
        )
        normalization = fit_resolved_normalization(normalization_resolved)
    matrix = run_resolved_omni_matrix(
        resolved,
        g1_xml=args.g1_xml,
        routes=routes,
        normalization_override=normalization,
        contact_segment_policy=contact_policy,
        foothold_action_policy=foothold_policy,
        maximum_step_time_ns=int(args.maximum_step_time_ms * 1_000_000),
    )
    save_omni_matrix(matrix, args.output)
    print(
        json.dumps(
            {
                "matrix_pass": matrix.matrix_pass,
                "route_count": len(matrix.runs),
                "deterministic_sha256": matrix.deterministic_sha256,
                "output": args.output,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if matrix.matrix_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
