#!/usr/bin/env python3
"""Train the deterministic 60 Hz G1 learned-motion-matching model bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--stage",
        choices=("all", "overfit", "decompressor", "stepper", "projector"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--overfit-steps", type=int, default=1000)
    parser.add_argument("--decompressor-steps", type=int, default=50_000)
    parser.add_argument("--stepper-steps", type=int, default=100_000)
    parser.add_argument("--projector-steps", type=int, default=50_000)
    parser.add_argument("--stepper-window", type=int, default=20)
    parser.add_argument("--withheld-frames", type=int, default=64)
    parser.add_argument("--withheld-halo", type=int, default=60)
    parser.add_argument(
        "--single-clip-overfit-canary",
        action="store_true",
        help=(
            "fit and evaluate the complete admitted single-clip corpus without "
            "claiming heldout generalization"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from resources.g1_lmm.training import TrainingConfig, train_flat_bundle

    arguments = build_parser().parse_args(argv)
    config = TrainingConfig(
        seed=arguments.seed,
        device=arguments.device,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        overfit_steps=arguments.overfit_steps,
        decompressor_steps=arguments.decompressor_steps,
        stepper_steps=arguments.stepper_steps,
        projector_steps=arguments.projector_steps,
        stepper_window=arguments.stepper_window,
        withheld_frames=arguments.withheld_frames,
        withheld_halo=arguments.withheld_halo,
        single_clip_overfit_canary=arguments.single_clip_overfit_canary,
    )
    receipt = train_flat_bundle(
        arguments.data_directory,
        arguments.output_directory,
        config=config,
        stage=arguments.stage,
    )
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
