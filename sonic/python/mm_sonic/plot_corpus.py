"""Plot intended command paths beside generated motion-matching root paths."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def plot(
    corpus: Path,
    output: Path,
    *,
    max_clips: int = 12,
    base_only: bool = False,
    columns: int = 4,
    cell_width: float = 4.2,
    cell_height: float = 4.0,
    dpi: int = 180,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import zarr

    root = zarr.open(str(corpus.expanduser().resolve()), mode="r")
    ends = np.asarray(root["meta/episode_ends"], dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1]))
    names = [str(value) for value in np.asarray(root["meta/episode_clip"])]
    indices = [
        index
        for index, name in enumerate(names)
        if not base_only or not name.endswith("__mirror")
    ][: int(max_clips)]
    count = len(indices)
    columns = max(1, int(columns))
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(float(cell_width) * columns, float(cell_height) * rows),
        squeeze=False,
    )
    for axis, index in zip(
        axes.flat,
        indices,
        strict=False,
    ):
        name = names[index]
        start = starts[index]
        stop = ends[index]
        select = slice(int(start), int(stop))
        intended = np.asarray(root["data/command_path_world_xy"][select])
        intended = intended - intended[0]
        requested = (
            np.asarray(root["data/command_requested_path_world_xy"][select])
            if "command_requested_path_world_xy" in root["data"]
            else None
        )
        if requested is not None:
            requested = requested - requested[0]
        actual = np.asarray(root["data/body_pos_w"][select])[:, 0, :2]
        actual = actual - actual[0]
        facing = np.asarray(
            root["data/command_shaped_heading_world_yaw"][select]
        )
        axis.plot(
            intended[:, 0],
            intended[:, 1],
            color="#d62728",
            linewidth=2.0,
            label="filtered command ball",
        )
        if requested is not None:
            axis.plot(
                requested[:, 0],
                requested[:, 1],
                color="#9467bd",
                linewidth=1.1,
                linestyle="--",
                alpha=0.8,
                label="raw stick integral",
            )
        axis.plot(
            actual[:, 0],
            actual[:, 1],
            color="#1f77b4",
            linewidth=1.8,
            label="motion-matched root",
        )
        stride = max(1, len(facing) // 12)
        indices = np.arange(0, len(facing), stride)
        axis.quiver(
            actual[indices, 0],
            actual[indices, 1],
            np.cos(facing[indices]),
            np.sin(facing[indices]),
            color="#2ca02c",
            angles="xy",
            scale_units="xy",
            scale=4.0,
            width=0.006,
            label="requested facing",
        )
        axis.set_title(name, fontsize=9)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world y (m)")
    for axis in axes.flat[count:]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4)
    figure.suptitle("Takara motion matching: stick, ball, and root paths", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(dpi))
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=12)
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="plot only base clips, excluding their exact mirrored partners",
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--cell-width", type=float, default=4.2)
    parser.add_argument("--cell-height", type=float, default=4.0)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    plot(
        args.corpus,
        args.output,
        max_clips=args.max_clips,
        base_only=args.base_only,
        columns=args.columns,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        dpi=args.dpi,
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
