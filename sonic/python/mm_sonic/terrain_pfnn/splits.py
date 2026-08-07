"""Leakage-safe deterministic identities for terrain-PFNN data splits."""

from __future__ import annotations

import hashlib
import re
from typing import Literal


SplitName = Literal["train", "validation", "test"]


# The twelve released flat G1 LAFAN stems sorted once by SHA-256(stem).
# The first eight are training, the next two validation, and the final two test.
LAFAN_BASE_STEMS_BY_DIGEST = (
    "walk1_subject1",
    "walk4_subject1",
    "walk1_subject2",
    "walk3_subject5",
    "walk2_subject4",
    "walk3_subject2",
    "walk3_subject3",
    "walk3_subject1",
    "walk2_subject3",
    "walk3_subject4",
    "walk2_subject1",
    "walk1_subject5",
)

_LAFAN_SPLITS: dict[str, SplitName] = {
    "walk1_subject1": "train",
    "walk4_subject1": "train",
    "walk1_subject2": "train",
    "walk3_subject5": "train",
    "walk2_subject4": "train",
    "walk3_subject2": "train",
    "walk3_subject3": "train",
    "walk3_subject1": "train",
    "walk2_subject3": "validation",
    "walk3_subject4": "validation",
    "walk2_subject1": "test",
    "walk1_subject5": "test",
}


def terrain_identity(clip_id: str) -> str:
    """Collapse all ``__NNN`` variants onto their sealed slope family."""

    if type(clip_id) is not str or not clip_id:
        raise ValueError("clip_id must be a nonempty string")
    match = re.search(r"slope_(\d+)", clip_id)
    if match is None:
        return clip_id.split("__", 1)[0]
    return f"slope_{int(match.group(1)):03d}"


def split_identity(name: str) -> SplitName:
    """Return exactly one stable split without inspecting window contents."""

    identity = terrain_identity(name)
    if identity.endswith(".csv"):
        identity = identity[:-4]
    explicit = _LAFAN_SPLITS.get(identity)
    if explicit is not None:
        return explicit
    bucket = int(
        hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16
    ) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


__all__ = [
    "LAFAN_BASE_STEMS_BY_DIGEST",
    "SplitName",
    "split_identity",
    "terrain_identity",
]
