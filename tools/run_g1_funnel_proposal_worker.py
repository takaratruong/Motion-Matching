#!/usr/bin/env python3
"""Sample one strict entry-conditioned proposal response artifact."""

import argparse
import hashlib
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from resources.g1_interaction_builder.diffusion import sample_checkpoint
from resources.g1_interaction_builder.proposal_artifact import (
    ProposalArtifact,
    certify_proposals,
    write_proposal_artifact,
)
from resources.g1_interaction_builder.proposal_request import read_proposal_request


def run_proposal_worker(
    request_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    device: str | None = None,
) -> ProposalArtifact:
    request = read_proposal_request(request_path)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).digest()
    if checkpoint_sha256 != request.checkpoint_sha256:
        raise ValueError("proposal request checkpoint identity mismatch")
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    outward = sample_checkpoint(
        checkpoint_path,
        torch.from_numpy(request.condition[None]),
        device=target_device,
        seed=request.batch_seed,
    )[0].cpu().numpy()
    execution = np.ascontiguousarray(outward[:, ::-1], dtype=np.float32)
    accepted = certify_proposals(execution)
    seeds = np.asarray(
        [(request.batch_seed + index) % 2**64 for index in range(32)],
        dtype=np.uint64,
    )
    artifact = ProposalArtifact(
        2,
        request.request_id,
        request.batch_seed,
        request.checkpoint_sha256,
        request.condition,
        execution,
        seeds,
        accepted,
    )
    write_proposal_artifact(output_path, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_proposal_worker(args.request, args.checkpoint, args.output)
    print(f"wrote {args.output}: accepted {int(artifact.accepted.sum())}/32 proposals")


if __name__ == "__main__":
    main()
