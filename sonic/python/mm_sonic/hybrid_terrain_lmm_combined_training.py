"""Explicit training/evaluation CLI for an authenticated combined corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_sonic.hybrid_terrain_lmm_combined import load_combined_cache
from mm_sonic.hybrid_terrain_lmm_training import (
    HybridModelConfig,
    evaluate_hybrid_generator,
    load_hybrid_generator,
    train_hybrid_generator,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train from a combined corpus")
    train.add_argument("--corpus", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument(
        "--variant", choices=("latent32", "latent64", "wider"), required=True
    )
    train.add_argument("--device", required=True)
    train.add_argument(
        "--loss-profile",
        choices=("uniform", "visual-articulation"),
        default="uniform",
    )
    train.add_argument("--post-coverage-steps", type=int, default=20_000)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--seed", type=int, default=1234)
    train.add_argument("--learning-rate", type=float, default=1.0e-3)
    train.add_argument("--weight-decay", type=float, default=1.0e-4)
    train.add_argument("--normalization-chunk-size", type=int, default=2048)
    train.add_argument("--evaluation-chunk-size", type=int, default=4096)
    train.add_argument("--fit-all-rows", action="store_true")
    train.add_argument("--selection-model", type=Path)
    train.add_argument("--selection-model-manifest-sha256")

    evaluate = commands.add_parser(
        "evaluate", help="evaluate a model bound to a combined corpus"
    )
    evaluate.add_argument("--corpus", required=True, type=Path)
    evaluate.add_argument("--model", required=True, type=Path)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--chunk-size", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    corpus = load_combined_cache(arguments.corpus)
    if arguments.command == "train":
        config = HybridModelConfig(
            variant=arguments.variant,
            seed=arguments.seed,
            device=arguments.device,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            post_coverage_steps=arguments.post_coverage_steps,
            normalization_chunk_size=arguments.normalization_chunk_size,
            evaluation_chunk_size=arguments.evaluation_chunk_size,
            fit_all_rows=arguments.fit_all_rows,
            loss_profile=arguments.loss_profile,
        )
        receipt = train_hybrid_generator(
            corpus,
            arguments.output,
            config=config,
            selection_model=arguments.selection_model,
            selection_model_manifest_sha256=(
                arguments.selection_model_manifest_sha256
            ),
        )
    else:
        generator = load_hybrid_generator(
            arguments.model, corpus=corpus, device=arguments.device
        )
        receipt = evaluate_hybrid_generator(
            corpus, generator, chunk_size=arguments.chunk_size
        )
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
