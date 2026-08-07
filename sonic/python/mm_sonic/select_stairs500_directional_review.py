"""Select a compact, behavior-balanced video review set from a stair bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import zarr

from .mirror_stairs500_omnidirectional_pilots import (
    DEFAULT_ARCHIVE,
    _realized_directional_motion,
)


def select(
    roots: Sequence[Path],
    *,
    archive_path: Path,
    output: Path,
    count: int,
) -> list[dict[str, str]]:
    archive = zarr.open_group(str(archive_path.expanduser().resolve()), mode="r")
    candidates: list[tuple[Path, dict[str, object]]] = []
    for root in roots:
        for report_path in root.expanduser().resolve().rglob("report.json"):
            report = json.loads(report_path.read_text())
            if report.get("status") != "accepted":
                continue
            if str(report.get("source_kind")) not in {
                "native",
                "temporal_reverse",
            }:
                continue
            if _realized_directional_motion(report_path, report, archive):
                candidates.append((report_path, report))
    candidates.sort(
        key=lambda item: (
            bool(item[1].get("mirror_of")),
            int(item[1]["clip_index"]),
            str(item[1]["source_kind"]),
            str(item[1]["mode"]),
        )
    )
    if len(candidates) < count:
        raise ValueError(
            f"review needs {count} realized candidates, found {len(candidates)}"
        )

    Predicate = Callable[[dict[str, object]], bool]
    strata: tuple[tuple[str, Predicate], ...] = (
        (
            "up_native_diagonal",
            lambda r: r["traversal"] == "up"
            and r["source_kind"] == "native"
            and "diagonal" in str(r["mode"]),
        ),
        (
            "down_native_diagonal",
            lambda r: r["traversal"] == "down"
            and r["source_kind"] == "native"
            and "diagonal" in str(r["mode"]),
        ),
        (
            "up_backward_straight",
            lambda r: r["traversal"] == "up"
            and r["source_kind"] == "temporal_reverse"
            and r["mode"] == "straight",
        ),
        (
            "down_backward_straight",
            lambda r: r["traversal"] == "down"
            and r["source_kind"] == "temporal_reverse"
            and r["mode"] == "straight",
        ),
        (
            "up_native_weave",
            lambda r: r["traversal"] == "up"
            and r["source_kind"] == "native"
            and any(
                value in str(r["mode"]) for value in ("zigzag", "slalom")
            ),
        ),
        (
            "down_native_weave",
            lambda r: r["traversal"] == "down"
            and r["source_kind"] == "native"
            and any(
                value in str(r["mode"]) for value in ("zigzag", "slalom")
            ),
        ),
        (
            "up_backward_diagonal",
            lambda r: r["traversal"] == "up"
            and r["source_kind"] == "temporal_reverse"
            and "diagonal" in str(r["mode"]),
        ),
        (
            "down_backward_diagonal",
            lambda r: r["traversal"] == "down"
            and r["source_kind"] == "temporal_reverse"
            and "diagonal" in str(r["mode"]),
        ),
        (
            "up_turning",
            lambda r: r["traversal"] == "up"
            and any(
                value in str(r["mode"]) for value in ("turning", "crab")
            ),
        ),
        (
            "down_turning",
            lambda r: r["traversal"] == "down"
            and any(
                value in str(r["mode"]) for value in ("turning", "crab")
            ),
        ),
        ("real_stair_left", lambda r: int(r["clip_index"]) == 473),
        ("real_stair_right", lambda r: int(r["clip_index"]) == 485),
    )
    chosen: list[tuple[str, Path, dict[str, object]]] = []
    used: set[Path] = set()
    for stratum, predicate in strata:
        match = next(
            (
                (path, report)
                for path, report in candidates
                if path not in used and predicate(report)
            ),
            None,
        )
        if match is None:
            continue
        path, report = match
        chosen.append((stratum, path, report))
        used.add(path)
        if len(chosen) == count:
            break
    for path, report in candidates:
        if len(chosen) == count:
            break
        if path in used:
            continue
        chosen.append(("coverage_fill", path, report))
        used.add(path)

    payload = [
        {
            "label": (
                f"{index:02d}_{stratum}_clip{int(report['clip_index']):03d}_"
                f"{report['source_kind']}_{report['mode']}"
            ),
            "report": str(path),
        }
        for index, (stratum, path, report) in enumerate(chosen)
    ]
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    arguments = parser.parse_args()
    print(
        json.dumps(
            select(
                arguments.root,
                archive_path=arguments.archive,
                output=arguments.output,
                count=arguments.count,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
