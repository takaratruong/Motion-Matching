#!/usr/bin/env python3
"""Run and save the PHP-style GRAIL terrain-skill qualification slice."""

from __future__ import annotations

import argparse
import json

from mm_sonic.torch_terrain_omni_rollout import save_omni_matrix
from mm_sonic.torch_terrain_omni_routes import same_stair_routes
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)
from mm_sonic.torch_terrain_skill_rollout import (
    qualification_routes,
    run_resolved_skill_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--g1-xml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--route", action="append", default=[])
    parser.add_argument("--qualification-slice", action="store_true")
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    resolved = resolve_stair_config(args.dataset, config, device=args.device)
    matrix = run_resolved_skill_matrix(
        resolved,
        g1_xml=args.g1_xml,
        routes=select_routes(args.route, args.qualification_slice),
    )
    save_omni_matrix(matrix, args.output)
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
