"""Strict, portable proposal artifacts and learned-funnel certification.

The proposal artifact is a canonical little-endian binary blob shared with the
C++ runtime loader (interaction_funnel_artifact.cpp). The controller never runs
Python or Torch; it loads this complete offline artifact before learned mode
starts, so the byte layout here and the loader there must stay identical.

Layout (all integers and floats little-endian):

    offset  bytes  field
    0       8      magic b"G1FUNNL2"
    8       4      uint32 schema_version (== 2)
    12      4      uint32 condition_dim  (== 24)
    16      4      uint32 proposal_count (== 32)
    20      4      uint32 sample_count   (== 16)
    24      4      uint32 sample_width   (== 4)
    28      8      uint64 request_id
    36      8      uint64 batch_seed
    44      32     checkpoint SHA-256 bytes
    76      96     float32[24] condition
    172     8192   execution float32[32][16][4], each (x,z,sin_yaw,cos_yaw)
    8364    256    uint64[32] seeds
    8620    32     uint8[32] accepted flags (0 or 1)
"""

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np

MAGIC = b"G1FUNNL2"
SCHEMA_VERSION = 2
CONDITION_DIM = 24
PROPOSAL_COUNT = 32
SAMPLE_COUNT = 16
SAMPLE_WIDTH = 4
_HEADER = struct.Struct("<8sIIIII")
_IDENTITY = struct.Struct("<QQ32s")
_YAW_UNIT_ATOL = 2e-5
MAX_TRANSLATION_STEP = 0.08
MAX_YAW_STEP_RADIANS = np.deg2rad(15.0)
MIN_ARC_LENGTH = 0.15


@dataclass(frozen=True)
class ProposalArtifact:
    schema_version: int
    request_id: int
    batch_seed: int
    checkpoint_sha256: bytes
    condition: np.ndarray
    proposals: np.ndarray
    seeds: np.ndarray
    accepted: np.ndarray

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported proposal artifact schema")
        if not isinstance(self.request_id, int) or not 0 < self.request_id < 2**64:
            raise ValueError("request_id must be a nonzero uint64")
        if not isinstance(self.batch_seed, int) or not 0 <= self.batch_seed < 2**64:
            raise ValueError("batch_seed must be a uint64")
        if not isinstance(self.checkpoint_sha256, bytes) or len(self.checkpoint_sha256) != 32:
            raise ValueError("checkpoint_sha256 must contain 32 bytes")
        if self.condition.shape != (CONDITION_DIM,):
            raise ValueError("condition must have shape (24,)")
        if self.proposals.shape != (PROPOSAL_COUNT, SAMPLE_COUNT, SAMPLE_WIDTH):
            raise ValueError("proposals must have shape (32, 16, 4)")
        if self.seeds.shape != (PROPOSAL_COUNT,) or self.accepted.shape != (PROPOSAL_COUNT,):
            raise ValueError("proposal metadata has the wrong shape")
        if not np.isfinite(self.condition).all() or not np.isfinite(self.proposals).all():
            raise ValueError("proposal artifact contains non-finite values")
        if not np.allclose(
            np.linalg.norm(self.proposals[..., 2:4], axis=-1), 1.0, atol=_YAW_UNIT_ATOL
        ):
            raise ValueError("proposal yaw vectors are not unit length")
        expected_entry = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        if not np.array_equal(
            self.proposals[:, 0],
            np.broadcast_to(expected_entry, self.proposals[:, 0].shape),
        ):
            raise ValueError("execution proposals must start at the entry identity")
        if len(np.unique(self.seeds)) != PROPOSAL_COUNT:
            raise ValueError("proposal seeds must be unique")
        accepted = np.asarray(self.accepted, dtype=bool)
        certifiable = certify_proposals(self.proposals)
        if np.any(accepted & ~certifiable):
            raise ValueError("accepted proposals must satisfy continuity constraints")


def certify_proposals(proposals: np.ndarray) -> np.ndarray:
    proposals = np.asarray(proposals, dtype=np.float32)
    if proposals.shape != (PROPOSAL_COUNT, SAMPLE_COUNT, SAMPLE_WIDTH):
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
        (translation_step.max(axis=1) <= MAX_TRANSLATION_STEP)
        & (yaw_step.max(axis=1) <= MAX_YAW_STEP_RADIANS)
        & (arc >= MIN_ARC_LENGTH)
    )


def serialize_proposal_artifact(artifact: ProposalArtifact) -> bytes:
    artifact.validate()
    parts = [
        _HEADER.pack(
            MAGIC, SCHEMA_VERSION, CONDITION_DIM, PROPOSAL_COUNT, SAMPLE_COUNT, SAMPLE_WIDTH
        ),
        _IDENTITY.pack(
            artifact.request_id,
            artifact.batch_seed,
            artifact.checkpoint_sha256,
        ),
        np.ascontiguousarray(artifact.condition, dtype="<f4").tobytes(),
        np.ascontiguousarray(artifact.proposals, dtype="<f4").tobytes(),
        np.ascontiguousarray(artifact.seeds, dtype="<u8").tobytes(),
        np.ascontiguousarray(np.asarray(artifact.accepted, dtype=bool), dtype=bool)
        .astype(np.uint8)
        .tobytes(),
    ]
    return b"".join(parts)


def deserialize_proposal_artifact(blob: bytes) -> ProposalArtifact:
    condition_bytes = CONDITION_DIM * 4
    proposal_bytes = PROPOSAL_COUNT * SAMPLE_COUNT * SAMPLE_WIDTH * 4
    seed_bytes = PROPOSAL_COUNT * 8
    accepted_bytes = PROPOSAL_COUNT
    expected = (_HEADER.size + _IDENTITY.size + condition_bytes + proposal_bytes
                + seed_bytes + accepted_bytes)
    if len(blob) != expected:
        raise ValueError("proposal artifact has the wrong byte length")

    magic, version, condition_dim, proposal_count, sample_count, sample_width = (
        _HEADER.unpack_from(blob, 0)
    )
    if magic != MAGIC:
        raise ValueError("proposal artifact has a bad magic header")
    if version != SCHEMA_VERSION:
        raise ValueError("unsupported proposal artifact schema")
    if (condition_dim, proposal_count, sample_count, sample_width) != (
        CONDITION_DIM, PROPOSAL_COUNT, SAMPLE_COUNT, SAMPLE_WIDTH
    ):
        raise ValueError("proposal artifact has invalid dimensions")

    offset = _HEADER.size
    request_id, batch_seed, checkpoint_sha256 = _IDENTITY.unpack_from(blob, offset)
    offset += _IDENTITY.size
    condition = np.frombuffer(blob, dtype="<f4", count=CONDITION_DIM, offset=offset).copy()
    offset += condition_bytes
    proposals = np.frombuffer(
        blob, dtype="<f4", count=PROPOSAL_COUNT * SAMPLE_COUNT * SAMPLE_WIDTH, offset=offset
    ).reshape(PROPOSAL_COUNT, SAMPLE_COUNT, SAMPLE_WIDTH).copy()
    offset += proposal_bytes
    seeds = np.frombuffer(blob, dtype="<u8", count=PROPOSAL_COUNT, offset=offset).copy()
    offset += seed_bytes
    accepted_raw = np.frombuffer(blob, dtype=np.uint8, count=PROPOSAL_COUNT, offset=offset)
    if np.any(accepted_raw > 1):
        raise ValueError("proposal artifact has invalid accepted flags")
    accepted = accepted_raw.astype(bool).copy()

    artifact = ProposalArtifact(
        SCHEMA_VERSION,
        request_id,
        batch_seed,
        checkpoint_sha256,
        condition,
        proposals,
        seeds,
        accepted,
    )
    artifact.validate()
    return artifact


def write_proposal_artifact(path: Path, artifact: ProposalArtifact) -> None:
    blob = serialize_proposal_artifact(artifact)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(blob)
    temporary.replace(path)


def read_proposal_artifact(path: Path) -> ProposalArtifact:
    return deserialize_proposal_artifact(Path(path).read_bytes())
