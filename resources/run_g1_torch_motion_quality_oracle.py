#!/usr/bin/env python3
"""Run the exhaustive offline terrain-motion quality oracle."""

from __future__ import annotations

import argparse
import json

from mm_sonic.torch_terrain_quality_report import run_quality_oracle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--terrain-config", required=True)
    parser.add_argument("--oracle-config", required=True)
    parser.add_argument("--g1-xml", required=True)
    parser.add_argument("--baseline-artifacts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_quality_oracle(
        dataset=args.dataset,
        terrain_config=args.terrain_config,
        oracle_config=args.oracle_config,
        g1_xml=args.g1_xml,
        baseline_artifacts=args.baseline_artifacts,
        output=args.output,
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
