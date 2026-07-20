"""Strict request artifact for one asynchronous funnel proposal batch."""

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


MAGIC = b"G1FREQ02"
SCHEMA_VERSION = 2
CONDITION_DIM = 24
_HEADER = struct.Struct("<8sIQQ32s")
_BYTE_LENGTH = _HEADER.size + CONDITION_DIM * 4


@dataclass(frozen=True)
class ProposalRequest:
    request_id: int
    batch_seed: int
    checkpoint_sha256: bytes
    condition: np.ndarray

    def validate(self) -> None:
        if not isinstance(self.request_id, int) or not 0 < self.request_id < 2**64:
            raise ValueError("request_id must be a nonzero uint64")
        if not isinstance(self.batch_seed, int) or not 0 <= self.batch_seed < 2**64:
            raise ValueError("batch_seed must be a uint64")
        if not isinstance(self.checkpoint_sha256, bytes) or len(self.checkpoint_sha256) != 32:
            raise ValueError("checkpoint_sha256 must contain 32 bytes")
        condition = np.asarray(self.condition)
        if condition.shape != (CONDITION_DIM,):
            raise ValueError("condition must have shape (24,)")
        if not np.isfinite(condition).all():
            raise ValueError("condition must be finite")


def serialize_proposal_request(request: ProposalRequest) -> bytes:
    request.validate()
    condition = np.ascontiguousarray(request.condition, dtype="<f4")
    return _HEADER.pack(
        MAGIC,
        SCHEMA_VERSION,
        request.request_id,
        request.batch_seed,
        request.checkpoint_sha256,
    ) + condition.tobytes(order="C")


def deserialize_proposal_request(blob: bytes) -> ProposalRequest:
    if len(blob) != _BYTE_LENGTH:
        raise ValueError("proposal request has the wrong byte length")
    magic, version, request_id, batch_seed, checkpoint_sha256 = _HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError("proposal request has a bad magic header")
    if version != SCHEMA_VERSION:
        raise ValueError("unsupported proposal request schema")
    condition = np.frombuffer(
        blob, dtype="<f4", count=CONDITION_DIM, offset=_HEADER.size
    ).copy()
    request = ProposalRequest(request_id, batch_seed, checkpoint_sha256, condition)
    request.validate()
    return request


def write_proposal_request(path: Path, request: ProposalRequest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(serialize_proposal_request(request))
    temporary.replace(path)


def read_proposal_request(path: Path) -> ProposalRequest:
    return deserialize_proposal_request(Path(path).read_bytes())
