"""Scene-aware motion-matching integration for the pinned GEAR-SONIC stack."""

from pathlib import Path

__version__ = "0.1.0"

from .full_walking_terrain_lmm_contracts import (
    FullWalkingInventory,
    LaneArtifact,
    RangeRecord,
    SourceRecord,
    SplitAssignment,
    build_split_ledger,
    connected_split_groups,
    load_lane,
    publish_lane_exclusive,
)


def build_inventory(
    *, bank: Path, pfnn_root: Path, grail_root: Path, takara: Path
) -> FullWalkingInventory:
    """Lazily import the inventory CLI module so ``python -m`` stays warning-free."""

    from .full_walking_terrain_lmm_inventory import build_inventory as build

    return build(bank=bank, pfnn_root=pfnn_root, grail_root=grail_root, takara=takara)

__all__ = [
    "FullWalkingInventory",
    "LaneArtifact",
    "RangeRecord",
    "SourceRecord",
    "SplitAssignment",
    "build_inventory",
    "build_split_ledger",
    "connected_split_groups",
    "load_lane",
    "publish_lane_exclusive",
]
