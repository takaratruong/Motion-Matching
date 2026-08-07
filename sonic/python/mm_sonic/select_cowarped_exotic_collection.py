"""Select a balanced collection set from exact-gated co-warp compounds."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def _bank(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("bank must have the form NAME=ACCEPTED_JSON")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("bank must have the form NAME=ACCEPTED_JSON")
    return name, Path(path).expanduser().resolve()


def _duration_s(row: dict[str, object]) -> float:
    if "duration_s" in row:
        return float(row["duration_s"])
    with np.load(str(row["motion"]), allow_pickle=False) as payload:
        frame_count = len(payload["root_position_world"])
        fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
    if fps <= 0.0:
        raise ValueError(f"invalid fps for {row['motion']}")
    return frame_count / fps


def load_rows(
    banks: tuple[tuple[str, Path], ...],
    *,
    minimum_duration_s: float = 0.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for name, path in banks:
        payload = json.loads(path.read_text())
        values = (
            payload.get("rows", payload.get("selections"))
            if isinstance(payload, dict)
            else payload
        )
        if not isinstance(values, list):
            raise ValueError(f"{path}: expected a row list")
        for value in values:
            row = dict(value)
            # Direct fixed-terrain directional exports use ``mode`` rather
            # than the source/compound pair used by retimed co-warps.  Keep
            # them in the same balanced collection as an explicit direct
            # event so extreme oblique/side-on ascents are not discarded.
            if "source_mode" not in row and "mode" in row:
                row["source_mode"] = str(row["mode"])
            row.setdefault("compound_mode", "direct")
            pilot = str(row["pilot"])
            if pilot in seen:
                continue
            seen.add(pilot)
            row["collection_bank"] = str(name)
            if "motion" not in row:
                manifest_text, label = pilot.rsplit("#", 1)
                manifest_path = Path(manifest_text)
                manifest = json.loads(manifest_path.read_text())
                matches = [
                    candidate
                    for candidate in manifest["pilots"]
                    if str(candidate["label"]) == label
                ]
                if len(matches) != 1:
                    raise ValueError(f"cannot resolve selected pilot {pilot}")
                matched = dict(matches[0])
                row["motion"] = str(matched["motion"])
                row["manifest"] = str(manifest_path)
                if "directional_report" in matched:
                    row["report"] = str(matched["directional_report"])
            row.setdefault(
                "clip_family",
                "stairs500" if "stairs500" in pilot else "unknown",
            )
            row.setdefault("clip_traversal", row.get("traversal", "unknown"))
            row["duration_s"] = _duration_s(row)
            if float(row["duration_s"]) < float(minimum_duration_s):
                continue
            rows.append(row)
    return rows


def _physical_source(row: dict[str, object]) -> tuple[str, int]:
    family = str(row.get("clip_family", "unknown"))
    if family.startswith("stairs500"):
        family = "stairs500"
    return (
        family,
        int(row["clip_index"]),
    )


def select(
    rows: list[dict[str, object]],
    *,
    count: int,
    maximum_per_physical_source: int,
) -> list[dict[str, object]]:
    requested = min(max(0, int(count)), len(rows))
    maximum = int(maximum_per_physical_source)
    if requested and maximum < 1:
        raise ValueError("maximum_per_physical_source must be positive")

    selected: list[dict[str, object]] = []
    used: set[str] = set()
    per_bank: Counter[str] = Counter()
    per_source_mode: Counter[str] = Counter()
    per_compound_mode: Counter[str] = Counter()
    per_traversal: Counter[str] = Counter()
    per_combination: Counter[tuple[str, str, str]] = Counter()
    per_physical: Counter[tuple[str, int]] = Counter()

    while len(selected) < requested:
        best: tuple[tuple[object, ...], dict[str, object]] | None = None
        for row in rows:
            pilot = str(row["pilot"])
            physical = _physical_source(row)
            if pilot in used or per_physical[physical] >= maximum:
                continue
            bank = str(row["collection_bank"])
            source_mode = str(row["source_mode"])
            compound_mode = str(row["compound_mode"])
            traversal = str(row.get("clip_traversal", "unknown"))
            combination = (bank, source_mode, compound_mode)
            # Lexicographic round-robin: cover every bank/behavior/event
            # combination before adding its next example, then prefer new
            # physical sources and cleaner exact-audit margins.
            rank: tuple[object, ...] = (
                per_combination[combination],
                per_bank[bank],
                per_source_mode[source_mode],
                per_compound_mode[compound_mode],
                per_traversal[traversal],
                per_physical[physical],
                float(row.get("maximum_stance_run_drift_m", 0.0)),
                float(row.get("maximum_foot_penetration_m", 0.0)),
                pilot,
            )
            if best is None or rank < best[0]:
                best = (rank, row)
        if best is None:
            break
        row = dict(best[1])
        pilot = str(row["pilot"])
        bank = str(row["collection_bank"])
        source_mode = str(row["source_mode"])
        compound_mode = str(row["compound_mode"])
        traversal = str(row.get("clip_traversal", "unknown"))
        physical = _physical_source(row)
        selected.append(row)
        used.add(pilot)
        per_bank[bank] += 1
        per_source_mode[source_mode] += 1
        per_compound_mode[compound_mode] += 1
        per_traversal[traversal] += 1
        per_combination[(bank, source_mode, compound_mode)] += 1
        per_physical[physical] += 1
    return selected


def plot_routes(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from mm_sonic.summarize_cowarped_directional_bank import _local_route_data

    banks = sorted({str(row["collection_bank"]) for row in rows})
    if not banks:
        return
    colors = {
        "stop_restart": "tab:blue",
        "reverse": "tab:orange",
        "bounce": "tab:green",
    }
    figure, axes = plt.subplots(
        1,
        len(banks),
        figsize=(6.0 * len(banks), 5.5),
        squeeze=False,
    )
    for axis, bank in zip(axes.flat, banks, strict=False):
        subset = [row for row in rows if str(row["collection_bank"]) == bank]
        for row in subset:
            actual, intended, _facing = _local_route_data(row)
            mode = str(row["compound_mode"])
            axis.plot(
                intended[:, 0],
                intended[:, 1],
                color="0.55",
                alpha=0.10,
                linewidth=0.6,
                linestyle="--",
            )
            axis.plot(
                actual[:, 0],
                actual[:, 1],
                color=colors.get(mode, "tab:purple"),
                alpha=0.30,
                linewidth=0.9,
            )
        axis.set_title(f"{bank} (n={len(subset)})")
        axis.set_xlabel("forward along source terrain (m)")
        axis.set_ylabel("lateral (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.axhline(0.0, color="black", alpha=0.15, linewidth=0.6)
        axis.grid(alpha=0.12)
    handles = [
        Line2D((0,), (0,), color=color, label=mode.replace("_", "/"))
        for mode, color in colors.items()
    ]
    handles.append(
        Line2D((0,), (0,), color="0.55", linestyle="--", label="intended")
    )
    figure.legend(handles=handles, loc="lower center", ncol=len(handles))
    figure.suptitle(
        "Balanced exact-gated terrain collection\n"
        "solid: realized G1 root; dashed gray: intended route"
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_selection(
    rows: list[dict[str, object]],
    output: Path,
    *,
    requested_count: int,
    maximum_per_physical_source: int,
    minimum_duration_s: float,
) -> dict[str, object]:
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "cowarped-exotic-collection/v1",
        "requested_count": int(requested_count),
        "selected_count": len(rows),
        "maximum_per_physical_source": int(maximum_per_physical_source),
        "minimum_duration_s": float(minimum_duration_s),
        "by_bank": dict(sorted(Counter(str(row["collection_bank"]) for row in rows).items())),
        "by_source_mode": dict(sorted(Counter(str(row["source_mode"]) for row in rows).items())),
        "by_compound_mode": dict(sorted(Counter(str(row["compound_mode"]) for row in rows).items())),
        "by_traversal": dict(sorted(Counter(str(row.get("clip_traversal", "unknown")) for row in rows).items())),
        "distinct_physical_source_count": len({_physical_source(row) for row in rows}),
        "duration_s": {
            "minimum": min((float(row["duration_s"]) for row in rows), default=0.0),
            "median": float(
                np.median([float(row["duration_s"]) for row in rows])
            )
            if rows
            else 0.0,
            "maximum": max((float(row["duration_s"]) for row in rows), default=0.0),
        },
        "routes_figure": str(destination.with_suffix(".routes.png")),
        "selections": rows,
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    plot_routes(rows, destination.with_suffix(".routes.png"))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", action="append", default=[], type=_bank)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--maximum-per-physical-source", type=int, default=8)
    parser.add_argument("--minimum-duration-s", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if not arguments.bank:
        parser.error("at least one --bank is required")
    if arguments.minimum_duration_s < 0.0:
        parser.error("--minimum-duration-s cannot be negative")
    rows = load_rows(
        tuple(arguments.bank),
        minimum_duration_s=arguments.minimum_duration_s,
    )
    chosen = select(
        rows,
        count=arguments.count,
        maximum_per_physical_source=arguments.maximum_per_physical_source,
    )
    result = write_selection(
        chosen,
        arguments.output,
        requested_count=arguments.count,
        maximum_per_physical_source=arguments.maximum_per_physical_source,
        minimum_duration_s=arguments.minimum_duration_s,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "selections"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
