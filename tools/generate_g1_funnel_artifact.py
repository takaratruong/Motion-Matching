#!/usr/bin/env python3
"""Generate and certify one learned object/grasp funnel proposal batch."""

import argparse
from pathlib import Path

import numpy as np
import torch

from resources.g1_interaction_builder.diffusion import sample_checkpoint
from resources.g1_interaction_builder.funnel_dataset import load_dataset
from resources.g1_interaction_builder.proposal_artifact import (
    ProposalArtifact,
    certify_proposals,
    write_proposal_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("build/g1-funnels/checkpoint.pt"))
    parser.add_argument("--pack", type=Path, default=Path("build/smart-pickup/full-pack"))
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026071901)
    parser.add_argument("--output", type=Path, default=Path("build/g1-funnels/beer10-proposals.npz"))
    args = parser.parse_args()
    dataset = load_dataset(args.pack)
    if not 0 <= args.row < len(dataset.conditions):
        raise SystemExit(f"row must be in [0, {len(dataset.conditions)})")
    condition = torch.from_numpy(dataset.conditions[args.row:args.row + 1])
    proposals = sample_checkpoint(args.checkpoint, condition, seed=args.seed)[0].cpu().numpy()
    accepted = certify_proposals(proposals)
    artifact = ProposalArtifact(1, condition[0].numpy(), proposals, np.arange(32, dtype=np.uint64), accepted)
    write_proposal_artifact(args.output, artifact)
    print(f"wrote {args.output}: accepted {int(accepted.sum())}/32 proposals")


if __name__ == "__main__":
    main()
