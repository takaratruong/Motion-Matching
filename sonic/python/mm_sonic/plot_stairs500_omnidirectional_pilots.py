"""Plot every accepted directional stair route with travel and facing cues."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import zarr

from .mirror_stairs500_omnidirectional_pilots import (
    _realized_directional_motion,
)


DEFAULT_ARCHIVE = Path(
    "/move/data/terrain-aware/motion-matching/"
    "grail-stairs500-clean-50hz-v1.zarr"
)


def _wrap(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def plot(
    roots: Sequence[Path],
    *,
    archive_path: Path,
    output: Path,
) -> dict[str, object]:
    resolved_roots = tuple(path.expanduser().resolve() for path in roots)
    report_paths = sorted(
        {
            path.resolve()
            for root in resolved_roots
            for path in root.rglob("report.json")
        }
    )
    reports = [
        (path, json.loads(path.read_text())) for path in report_paths
    ]
    if not reports:
        raise ValueError(
            "no directional pilot reports under "
            + ", ".join(str(value) for value in resolved_roots)
        )
    archive = zarr.open_group(str(archive_path), mode="r")
    nominally_accepted = [
        (path, row) for path, row in reports if row.get("status") == "accepted"
    ]
    accepted = [
        (path, row)
        for path, row in nominally_accepted
        if _realized_directional_motion(path, row, archive)
    ]
    present_kinds = {str(row["source_kind"]) for _path, row in accepted}
    source_kinds = tuple(
        kind
        for kind in ("native", "temporal_reverse", "reverse")
        if kind in present_kinds
    ) + tuple(
        sorted(
            present_kinds
            - {"native", "temporal_reverse", "reverse"}
        )
    )
    panels = tuple(
        (traversal, source_kind)
        for traversal in ("up", "down")
        for source_kind in source_kinds
    )
    modes = sorted({str(row["mode"]) for _path, row in accepted})
    palette = plt.get_cmap("tab20")
    colors = {
        mode: palette(index % 20) for index, mode in enumerate(modes)
    }
    figure, axes = plt.subplots(
        2,
        max(1, len(source_kinds)),
        figsize=(7.4 * max(1, len(source_kinds)), 11.0),
        constrained_layout=True,
        squeeze=False,
    )
    maximum_lateral = 0.5
    for axis, (traversal, source_kind) in zip(
        axes.flat, panels, strict=True
    ):
        rows = [
            (path, row)
            for path, row in accepted
            if row["traversal"] == traversal
            and row["source_kind"] == source_kind
        ]
        for report_path, row in rows:
            motion_path = report_path.parent / "motion.npz"
            with np.load(motion_path, allow_pickle=False) as arrays:
                root_world = np.asarray(
                    arrays["root_position_world"], dtype=np.float64
                )
                intended_root_world = np.asarray(
                    arrays.get("intended_root_position_world", root_world),
                    dtype=np.float64,
                )
                facing_world = np.asarray(
                    arrays["command_facing_yaw_world_rad"], dtype=np.float64
                )
                velocity_world = np.asarray(
                    arrays["command_velocity_world_xy"], dtype=np.float64
                )
            clip_index = int(row["clip_index"])
            route_yaw = float(
                row.get(
                    "route_yaw_rad",
                    float(archive["travel_yaw_rad"][clip_index])
                    + (
                        math.pi
                        if row.get("source_kind") == "temporal_reverse"
                        else 0.0
                    ),
                )
            )
            direction = np.asarray(
                (math.cos(route_yaw), math.sin(route_yaw)), dtype=np.float64
            )
            lateral = np.asarray((-direction[1], direction[0]))
            relative = root_world[:, :2] - root_world[0, :2]
            progress = relative @ direction
            offset = relative @ lateral
            intended_relative = (
                intended_root_world[:, :2] - root_world[0, :2]
            )
            intended_progress = intended_relative @ direction
            intended_offset = intended_relative @ lateral
            maximum_lateral = max(
                maximum_lateral, float(np.max(np.abs(offset)))
            )
            mode = str(row["mode"])
            color = colors[mode]
            axis.plot(
                progress,
                offset,
                color=color,
                linewidth=1.45,
                alpha=0.78,
                label=mode,
            )
            axis.plot(
                intended_progress,
                intended_offset,
                color=color,
                linewidth=0.85,
                linestyle="--",
                alpha=0.45,
            )
            samples = np.linspace(
                0, len(root_world) - 1, min(8, len(root_world)), dtype=int
            )
            speed = np.linalg.norm(velocity_world[samples], axis=1)
            moving = speed > 0.04
            if np.any(moving):
                travel_local_x = velocity_world[samples] @ direction
                travel_local_y = velocity_world[samples] @ lateral
                norm = np.maximum(
                    np.hypot(travel_local_x, travel_local_y), 1.0e-9
                )
                axis.quiver(
                    progress[samples][moving],
                    offset[samples][moving],
                    travel_local_x[moving] / norm[moving],
                    travel_local_y[moving] / norm[moving],
                    color=[color],
                    angles="xy",
                    scale_units="xy",
                    scale=4.2,
                    width=0.003,
                    alpha=0.75,
                )
            relative_facing = _wrap(facing_world[samples] - route_yaw)
            axis.quiver(
                progress[samples],
                offset[samples],
                np.cos(relative_facing),
                np.sin(relative_facing),
                color="#111111",
                angles="xy",
                scale_units="xy",
                scale=6.4,
                width=0.0023,
                alpha=0.55,
            )
        axis.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        axis.set_title(
            f"{traversal.capitalize()} / {source_kind} ({len(rows)} accepted)"
        )
        axis.set_xlabel("stair-axis progress from clip start (m)")
        axis.set_ylabel("stair-local lateral displacement (m)")
        axis.grid(alpha=0.16, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_aspect("auto")

    lateral_limit = 1.10 * maximum_lateral
    for axis in axes.flat:
        axis.set_ylim(-lateral_limit, lateral_limit)
    handles: dict[str, object] = {}
    for axis in axes.flat:
        found_handles, found_labels = axis.get_legend_handles_labels()
        for handle, label in zip(found_handles, found_labels, strict=True):
            handles.setdefault(label, handle)
    if handles:
        figure.legend(
            handles.values(),
            handles.keys(),
            loc="outside lower center",
            ncol=min(7, len(handles)),
            frameon=False,
        )
    figure.suptitle(
        "Exact-audited stair-local directional pilot\n"
        "solid = realized root; dashed = intended path; "
        "colored arrows = travel; black arrows = body facing",
        fontsize=15,
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)

    cell_counts = Counter(
        (str(row["traversal"]), str(row["source_kind"]))
        for _path, row in accepted
    )
    mode_counts = Counter(str(row["mode"]) for _path, row in accepted)
    summary = {
        "schema": "stairs500-omnidirectional-pilot-figure/v1",
        "roots": [str(value) for value in resolved_roots],
        "report_count": len(reports),
        "accepted_count": len(accepted),
        "rejected_count": len(reports) - len(accepted),
        "skipped_noop_count": len(nominally_accepted) - len(accepted),
        "cell_counts": {
            f"{traversal}|{kind}": int(cell_counts[(traversal, kind)])
            for traversal, kind in panels
        },
        "mode_counts": dict(sorted(mode_counts.items())),
        "maximum_lateral_displacement_m": maximum_lateral,
        "figure": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            plot(
                arguments.root,
                archive_path=arguments.archive,
                output=arguments.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
