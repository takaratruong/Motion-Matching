"""Select a compact, behavior-balanced video review set from a stair bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import zarr

from .export_stairs500_directional_bank import _quality_tier
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
            _quality_tier(item[1]) != "gold",
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
    # This manifest is a visual gate for the *new* command coverage, not a
    # random sample of the much larger straight/gentle class.  Quotas keep
    # exotic behaviors visible even after the source sweep is scaled up.
    strata: tuple[tuple[str, int, Predicate], ...] = (
        (
            "up_extreme_travel",
            2,
            lambda r: r["traversal"] == "up"
            and any(
                value in str(r["mode"])
                for value in ("diagonal_hard", "travel_", "lane_")
            ),
        ),
        (
            "down_extreme_travel",
            2,
            lambda r: r["traversal"] == "down"
            and any(
                value in str(r["mode"])
                for value in ("diagonal_hard", "travel_", "lane_")
            ),
        ),
        (
            "up_independent_facing",
            3,
            lambda r: r["traversal"] == "up"
            and any(
                value in str(r["mode"])
                for value in ("counterface", "facing_weave", "crab")
            ),
        ),
        (
            "down_independent_facing",
            3,
            lambda r: r["traversal"] == "down"
            and any(
                value in str(r["mode"])
                for value in ("counterface", "facing_weave", "crab")
            ),
        ),
        (
            "up_turning_weave",
            3,
            lambda r: r["traversal"] == "up"
            and "turning_" in str(r["mode"])
            and any(
                value in str(r["mode"]) for value in ("zigzag", "slalom")
            ),
        ),
        (
            "down_turning_weave",
            3,
            lambda r: r["traversal"] == "down"
            and "turning_" in str(r["mode"])
            and any(
                value in str(r["mode"]) for value in ("zigzag", "slalom")
            ),
        ),
        (
            "up_native_diagonal",
            3,
            lambda r: r["traversal"] == "up"
            and r["source_kind"] == "native"
            and any(
                value in str(r["mode"])
                for value in ("diagonal_medium", "diagonal_soft")
            ),
        ),
        (
            "down_native_diagonal",
            3,
            lambda r: r["traversal"] == "down"
            and r["source_kind"] == "native"
            and any(
                value in str(r["mode"])
                for value in ("diagonal_medium", "diagonal_soft")
            ),
        ),
        (
            "up_backward_diagonal",
            4,
            lambda r: r["traversal"] == "up"
            and r["source_kind"] == "temporal_reverse"
            and "diagonal" in str(r["mode"]),
        ),
        (
            "down_backward_diagonal",
            4,
            lambda r: r["traversal"] == "down"
            and r["source_kind"] == "temporal_reverse"
            and "diagonal" in str(r["mode"]),
        ),
        (
            "real_stair_left",
            1,
            lambda r: int(r["clip_index"]) == 473,
        ),
        (
            "real_stair_right",
            1,
            lambda r: int(r["clip_index"]) == 485,
        ),
        (
            "up_native_weave",
            3,
            lambda r: r["traversal"] == "up"
            and r["source_kind"] == "native"
            and any(
                value in str(r["mode"]) for value in ("zigzag", "slalom")
            ),
        ),
        (
            "down_native_weave",
            3,
            lambda r: r["traversal"] == "down"
            and r["source_kind"] == "native"
            and any(
                value in str(r["mode"]) for value in ("zigzag", "slalom")
            ),
        ),
        (
            "up_turning",
            3,
            lambda r: r["traversal"] == "up"
            and any(
                value in str(r["mode"]) for value in ("turning", "crab")
            ),
        ),
        (
            "down_turning",
            3,
            lambda r: r["traversal"] == "down"
            and any(
                value in str(r["mode"]) for value in ("turning", "crab")
            ),
        ),
    )
    chosen: list[tuple[str, Path, dict[str, object]]] = []
    used: set[Path] = set()
    for stratum, quota, predicate in strata:
        # Exercise both lateral polarities in the visual gate.  Without this
        # alternation, deterministic sorting fills every quota from the
        # unmirrored half before an exact reflected motion is ever shown.
        for slot in range(quota):
            preferred_mirror = bool(slot % 2)
            match = next(
                (
                    (path, report)
                    for path, report in candidates
                    if path not in used
                    and predicate(report)
                    and bool(report.get("mirror_of")) == preferred_mirror
                ),
                None,
            )
            if match is None:
                match = next(
                    (
                        (path, report)
                        for path, report in candidates
                        if path not in used and predicate(report)
                    ),
                    None,
                )
            if match is None:
                break
            path, report = match
            chosen.append((stratum, path, report))
            used.add(path)
            if len(chosen) == count:
                break
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
            "quality_tier": _quality_tier(report),
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
