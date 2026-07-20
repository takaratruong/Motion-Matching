"""Strict, portable proposal artifacts and learned-funnel certification."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ProposalArtifact:
    schema_version: int
    condition: np.ndarray
    proposals: np.ndarray
    seeds: np.ndarray
    accepted: np.ndarray

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported proposal artifact schema")
        if self.condition.shape != (18,):
            raise ValueError("condition must have shape (18,)")
        if self.proposals.shape != (32, 16, 4):
            raise ValueError("proposals must have shape (32, 16, 4)")
        if self.seeds.shape != (32,) or self.accepted.shape != (32,):
            raise ValueError("proposal metadata has the wrong shape")
        if not np.isfinite(self.condition).all() or not np.isfinite(self.proposals).all():
            raise ValueError("proposal artifact contains non-finite values")
        if not np.allclose(np.linalg.norm(self.proposals[..., 2:4], axis=-1), 1.0, atol=2e-5):
            raise ValueError("proposal yaw vectors are not unit length")


def certify_proposals(proposals: np.ndarray) -> np.ndarray:
    proposals = np.asarray(proposals, dtype=np.float32)
    if proposals.shape != (32, 16, 4):
        raise ValueError("proposals must have shape (32, 16, 4)")
    translation_step = np.linalg.norm(np.diff(proposals[..., :2], axis=1), axis=-1)
    yaw_step = np.abs(np.arctan2(
        proposals[:, 1:, 2] * proposals[:, :-1, 3]
        - proposals[:, 1:, 3] * proposals[:, :-1, 2],
        proposals[:, 1:, 2] * proposals[:, :-1, 2]
        + proposals[:, 1:, 3] * proposals[:, :-1, 3],
    ))
    arc = translation_step.sum(axis=1)
    return (
        (translation_step.max(axis=1) <= 0.08)
        & (yaw_step.max(axis=1) <= np.deg2rad(15.0))
        & (arc >= 0.15)
    )


def write_proposal_artifact(path: Path, artifact: ProposalArtifact) -> None:
    artifact.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    np.savez(
        temporary,
        condition=artifact.condition.astype("<f4"),
        proposals=artifact.proposals.astype("<f4"),
        seeds=artifact.seeds.astype("<u8"),
        accepted=artifact.accepted.astype("?"),
    )
    generated = Path(str(temporary) + ".npz")
    generated.replace(temporary)
    temporary.replace(path)
    manifest = path.with_suffix(".json")
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "proposal_count": 32, "sample_count": 16,
             "condition_dim": 18, "accepted_count": int(artifact.accepted.sum())},
            sort_keys=True, separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )


def read_proposal_artifact(path: Path) -> ProposalArtifact:
    path = Path(path)
    with np.load(path, allow_pickle=False) as values:
        artifact = ProposalArtifact(1, values["condition"], values["proposals"], values["seeds"], values["accepted"])
    artifact.validate()
    return artifact
