#!/usr/bin/env python3
"""Run and save the deterministic same-stair omnidirectional matrix."""

from __future__ import annotations

import argparse
import json
import math

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
            "layered",
            "layered-hybrid",
            "continuous-control",
        ),
        help="Condition contact-segment entries on this foothold ablation arm.",
    )
    parser.add_argument(
        "--foothold-height-tolerance-m",
        type=float,
        default=0.06,
        help="Hard per-contact height tolerance; must remain below half a riser.",
    )
    parser.add_argument(
        "--maximum-step-time-ms",
        type=float,
        default=1000.0,
        help="Fail a route after a matcher step exceeds this latency.",
    )
    parser.add_argument(
        "--transition-joint-position-weight",
        type=float,
        help="Override the non-negative full-pose transition position weight.",
    )
    parser.add_argument(
        "--transition-joint-velocity-weight",
        type=float,
        help="Override the non-negative full-pose transition velocity weight.",
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


def _apply_transition_overrides(config: dict, args) -> None:
    for field, override in (
        (
            "transition_joint_position_weight",
            args.transition_joint_position_weight,
        ),
        (
            "transition_joint_velocity_weight",
            args.transition_joint_velocity_weight,
        ),
    ):
        if override is None:
            continue
        if not math.isfinite(override) or override < 0.0:
            raise ValueError(f"{field} override must be finite and non-negative")
        config["matcher"][field] = override


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

        experiment_config = load_contact_segment_config(args.config)
        _apply_transition_overrides(experiment_config, args)
        resolved = resolve_contact_segment_config(
            args.dataset, experiment_config, device=args.device
        )
        contact_policy = build_contact_segment_policy(resolved, args.g1_xml)
    else:
        experiment_config = load_experiment_config(args.config)
        _apply_transition_overrides(experiment_config, args)
        resolved = resolve_stair_config(
            args.dataset, experiment_config, device=args.device
        )
    if args.foothold_arm and contact_policy is None:
        raise ValueError("--foothold-arm requires --contact-segments")
    if not 0.0 < args.foothold_height_tolerance_m < 0.0889:
        raise ValueError(
            "foothold height tolerance must be positive and below half a riser"
        )
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
            height_tolerance_m=args.foothold_height_tolerance_m,
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
