#!/usr/bin/env python3
"""Run and save the offline G1 contact-space terrain oracle."""

from __future__ import annotations

import argparse
import json

from mm_sonic.torch_contact_oracle_rollout import (
    run_resolved_oracle_matrix,
    save_oracle_matrix,
)
from mm_sonic.torch_contact_oracle_search import load_contact_oracle_config
from mm_sonic.torch_terrain_omni_routes import same_stair_routes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--g1-xml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help="Run only this route name; repeat to select multiple routes.",
    )
    return parser


def select_routes(names: list[str]):
    inventory = same_stair_routes()
    if not names:
        return inventory
    requested = set(names)
    if len(requested) != len(names):
        raise ValueError("route names must not be repeated")
    known = {route.name for route in inventory}
    missing = sorted(requested - known)
    if missing:
        raise ValueError(f"unknown route names: {', '.join(missing)}")
    return tuple(route for route in inventory if route.name in requested)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment = load_contact_oracle_config(args.config)
    matrix = run_resolved_oracle_matrix(
        dataset=args.dataset,
        experiment=experiment,
        g1_xml=args.g1_xml,
        device=args.device,
        routes=select_routes(args.route),
    )
    save_oracle_matrix(matrix, args.output)
    execution_completed = all(
        run.completed_without_exception for run in matrix.runs
    )
    print(
        json.dumps(
            {
                "execution_completed": execution_completed,
                "matrix_pass": matrix.matrix_pass,
                "route_count": len(matrix.runs),
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
