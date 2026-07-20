#!/usr/bin/env python3
"""Train the object/grasp/entry-conditioned interaction funnel model."""

import argparse
from pathlib import Path

import torch

from resources.g1_interaction_builder.diffusion import train_funnel
from resources.g1_interaction_builder.funnel_dataset import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=Path("build/smart-pickup/full-pack"))
    parser.add_argument("--output", type=Path, default=Path("build/g1-funnels/checkpoint.pt"))
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    dataset = load_dataset(args.pack)
    conditions = torch.from_numpy(dataset.conditions)
    funnels = torch.from_numpy(dataset.funnels)
    train_funnel(conditions, funnels, args.output, steps=args.steps, batch_size=args.batch_size)
    print(f"wrote {args.output} from {len(dataset.conditions)} rows")


if __name__ == "__main__":
    main()
