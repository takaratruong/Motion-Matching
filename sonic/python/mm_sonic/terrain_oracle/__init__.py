"""Canonical terrain-oracle motion contracts."""

from collections.abc import Sequence

from .canonical import (
    CanonicalClip,
    CanonicalTerrainMesh,
    CommandTrack,
    SourceIdentity,
    TerrainBinding,
    derive_clip_kinematics,
)
from .math3d import RigidTransform


def corpus_main(argv: Sequence[str] | None = None) -> int:
    """Run the terrain-corpus CLI without creating an import cycle."""

    from .corpus_cli import main

    return main(argv)


__all__ = (
    "CanonicalClip",
    "CanonicalTerrainMesh",
    "CommandTrack",
    "RigidTransform",
    "SourceIdentity",
    "TerrainBinding",
    "corpus_main",
    "derive_clip_kinematics",
)
