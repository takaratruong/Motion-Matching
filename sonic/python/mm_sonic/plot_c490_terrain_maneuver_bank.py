"""Plot registered terrain profiles and motion paths for a C490 pilot bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .generate_generic_stair_route import load_target_mesh
from .motionbricks_global_terrain_viewer import _terrain_surface_height


STATUS_COLOUR = {
    "pending_dense_visual_review": "#2ca02c",
    "source_rejected": "#d62728",
    "no_supported_pivot": "#ff9f1c",
    "pilot_gate_rejected": "#d62728",
}


def _arc_length(xy: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    )


def plot_bank(bank_root: Path, output: Path) -> Path:
    root = bank_root.expanduser().resolve()
    summaries = sorted(root.glob("clip_*/clip_summary.json"))
    if not summaries:
        raise ValueError(f"no clip summaries beneath {root}")
    columns = 3
    rows = int(np.ceil(len(summaries) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(15.0, 3.25 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, summary_path in zip(axes.flat, summaries, strict=False):
        summary = json.loads(summary_path.read_text())
        source_motion = summary_path.parent / "source" / "motion.npz"
        with np.load(source_motion, allow_pickle=False) as payload:
            root_xyz = np.asarray(payload["root_position_world"], dtype=np.float64)
        manifest_path = summary_path.parent / "pilots" / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            position = manifest["terrain_position_world"]
            quaternion = manifest["terrain_quaternion_world_from_usd_wxyz"]
            pivot_frame = int(manifest["pilots"][0]["pivot"]["frame_index"])
        else:
            # C490 sources share the documented registration even when the
            # downstream pivot gate rejects the clip.
            position = (0.0, 0.0, 0.0)
            quaternion = (0.7071068286895752, 0.0, 0.0, -0.7071067094802856)
            pivot_frame = None
        terrain = load_target_mesh(
            Path(summary["terrain_usd"]),
            position_world=position,
            quaternion_world_from_usd_wxyz=quaternion,
        )
        ray_origin_z = float(np.max(terrain.vertices_world[:, 2]) + 2.0)
        support = np.asarray(
            [
                _terrain_surface_height(
                    terrain, value[:2], ray_origin_z=ray_origin_z
                )
                for value in root_xyz
            ],
            dtype=np.float64,
        )
        distance = _arc_length(root_xyz[:, :2])
        support -= support[0]
        pelvis = root_xyz[:, 2] - root_xyz[0, 2]
        colour = STATUS_COLOUR.get(summary["status"], "#4c78a8")
        axis.fill_between(distance, 0.0, support, color="#9ad29a", alpha=0.65)
        axis.plot(distance, support, color="#157f3b", linewidth=2.0, label="terrain")
        axis.plot(distance, pelvis, color="#26547c", linewidth=1.5, label="pelvis Δz")
        if pivot_frame is not None:
            axis.axvline(
                distance[pivot_frame], color="#ef476f", linewidth=1.4, linestyle="--"
            )
        axis.set_title(
            f"{summary['clip_index']} · {summary['clip_traversal']} · "
            f"{summary['root_delta_z_m']:+.2f} m",
            color=colour,
            fontweight="bold",
        )
        axis.text(
            0.02,
            0.96,
            str(summary["status"]).replace("pending_dense_visual_review", "admitted"),
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            color=colour,
        )
        axis.grid(alpha=0.2)
        axis.set_xlabel("root path distance (m)")
        axis.set_ylabel("height relative to start (m)")
    for axis in axes.flat[len(summaries) :]:
        axis.axis("off")
    axes.flat[0].legend(loc="lower left", ncols=2, fontsize=8)
    figure.suptitle(
        "C490 terrain-maneuver scale-up · registered source profiles",
        fontsize=15,
        fontweight="bold",
    )
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(plot_bank(arguments.bank_root, arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
