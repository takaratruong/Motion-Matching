#!/usr/bin/env python3
"""Run the offline G1 terrain contact-composition ablation."""

from __future__ import annotations

import argparse
import json

from mm_sonic.torch_terrain_contact_ablation import run_contact_ablation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--terrain-config", required=True)
    parser.add_argument("--oracle-config", required=True)
    parser.add_argument("--g1-xml", required=True)
    parser.add_argument("--baseline-artifacts", required=True)
    parser.add_argument("--quality-oracle-artifacts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-actions-per-state", required=True, type=int)
    parser.add_argument(
        "--projection-strategy",
        required=True,
        choices=("joint-only", "stance-root", "root-only"),
    )
    parser.add_argument(
        "--selection-strategy",
        required=True,
        choices=("saved", "contact-anchored"),
    )
    parser.add_argument("--root-correction-scale", required=True, type=float)
    parser.add_argument("--root-smoothing-passes", required=True, type=int)
    parser.add_argument("--joint-smoothing-passes", required=True, type=int)
    parser.add_argument("--reproject-smoothed-joints", action="store_true")
    parser.add_argument("--reprojection-blend", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_contact_ablation(
        dataset=args.dataset,
        terrain_config=args.terrain_config,
        oracle_config=args.oracle_config,
        g1_xml=args.g1_xml,
        baseline_artifacts=args.baseline_artifacts,
        quality_oracle_artifacts=args.quality_oracle_artifacts,
        output=args.output,
        device=args.device,
        maximum_actions_per_state=args.max_actions_per_state,
        projection_strategy=args.projection_strategy,
        selection_strategy=args.selection_strategy,
        root_correction_scale=args.root_correction_scale,
        root_smoothing_passes=args.root_smoothing_passes,
        joint_smoothing_passes=args.joint_smoothing_passes,
        reproject_smoothed_joints=args.reproject_smoothed_joints,
        reprojection_blend=args.reprojection_blend,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
