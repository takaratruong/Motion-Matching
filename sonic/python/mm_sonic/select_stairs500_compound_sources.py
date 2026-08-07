"""Select source-diverse directional stairs for compound maneuver authoring."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Sequence

import zarr

from .export_stairs500_directional_bank import _quality_tier
from .mirror_stairs500_omnidirectional_pilots import (
    DEFAULT_ARCHIVE,
    _realized_directional_motion,
)


def behavior_family(mode: str) -> str:
    """Collapse angle variants without erasing distinct joystick behaviors."""

    value = str(mode)
    if "turning_slalom" in value:
        return "turning_slalom"
    if "turning_zigzag" in value:
        return "turning_zigzag"
    if value.startswith("travel_"):
        return "fixed_facing_oblique"
    if "counterface" in value:
        return "counterface"
    if "crab" in value:
        return "crab"
    if "facing_weave" in value:
        return "facing_weave"
    if "diagonal" in value:
        return "diagonal"
    if "lane" in value:
        return "lane"
    if "slalom" in value:
        return "path_slalom"
    if "zigzag" in value:
        return "path_zigzag"
    if value.startswith("turning_"):
        return "turning"
    if value.startswith("face_"):
        return "facing"
    return "other"


def select(
    roots: Sequence[Path],
    *,
    archive_path: Path,
    count: int,
    maximum_per_source: int = 2,
) -> list[dict[str, object]]:
    """Round-robin behavior, direction, time direction, and source identity."""

    archive = zarr.open_group(str(archive_path.expanduser().resolve()), mode="r")
    groups: dict[tuple[str, str, str, bool], list[tuple[Path, dict[str, object]]]] = (
        defaultdict(list)
    )
    seen_paths: set[Path] = set()
    for root in roots:
        for report_path in root.expanduser().resolve().rglob("report.json"):
            path = report_path.resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            report = json.loads(path.read_text())
            if report.get("status") != "accepted":
                continue
            if str(report.get("source_kind")) not in {
                "native",
                "temporal_reverse",
            }:
                continue
            if not _realized_directional_motion(path, report, archive):
                continue
            family = behavior_family(str(report["mode"]))
            if family == "other":
                continue
            key = (
                str(report["traversal"]),
                str(report["source_kind"]),
                family,
                bool(report.get("mirror_of")),
            )
            groups[key].append((path, report))

    for candidates in groups.values():
        candidates.sort(
            key=lambda item: (
                _quality_tier(item[1]) != "gold",
                int(item[1]["clip_index"]),
                str(item[1]["mode"]),
            )
        )
    if not groups:
        raise ValueError("no realized directional stair candidates found")

    selected: list[tuple[Path, dict[str, object], str]] = []
    selected_paths: set[Path] = set()
    per_source: dict[int, int] = defaultdict(int)
    ordered_groups = sorted(groups)
    while len(selected) < int(count):
        progress = False
        for key in ordered_groups:
            candidates = groups[key]
            match_index = next(
                (
                    index
                    for index, (path, report) in enumerate(candidates)
                    if path not in selected_paths
                    and per_source[int(report["clip_index"])]
                    < int(maximum_per_source)
                ),
                None,
            )
            if match_index is None:
                continue
            path, report = candidates.pop(match_index)
            selected.append((path, report, key[2]))
            selected_paths.add(path)
            per_source[int(report["clip_index"])] += 1
            progress = True
            if len(selected) == int(count):
                break
        if not progress:
            break

    if len(selected) < int(count):
        raise ValueError(
            f"requested {count} source-diverse candidates, found {len(selected)} "
            f"under maximum_per_source={maximum_per_source}"
        )
    return [
        {
            "label": (
                f"{rank:03d}_{family}_clip{int(report['clip_index']):03d}_"
                f"{report['source_kind']}_{report['mode']}"
            ),
            "report": str(path),
            "clip_index": int(report["clip_index"]),
            "traversal": str(report["traversal"]),
            "source_kind": str(report["source_kind"]),
            "mode": str(report["mode"]),
            "behavior_family": family,
            "mirrored": bool(report.get("mirror_of")),
            "quality_tier": _quality_tier(report),
        }
        for rank, (path, report, family) in enumerate(selected)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--maximum-per-source", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.count < 1 or arguments.maximum_per_source < 1:
        parser.error("counts must be positive")
    rows = select(
        arguments.root,
        archive_path=arguments.archive,
        count=arguments.count,
        maximum_per_source=arguments.maximum_per_source,
    )
    destination = arguments.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(destination),
                "selection_count": len(rows),
                "distinct_source_count": len(
                    {int(row["clip_index"]) for row in rows}
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
