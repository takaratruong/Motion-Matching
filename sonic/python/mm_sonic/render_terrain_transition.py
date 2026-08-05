"""Render an audited clean kinematic transition on the Justin staircase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .offline_corpus import BODY_NAMES
from .render_corpus_clip import SKELETON_EDGES
from .terrain_catalog import (
    JUSTIN_STAIR_ORIGIN_WORLD_XY,
    JUSTIN_STAIR_RISE_M,
    JUSTIN_STAIR_RUN_M,
    JUSTIN_STAIR_WIDTH_M,
    load_database,
)
from .terrain_motion import wrap_angle


def _hard_gate(db, left: int, right: int) -> dict[str, float | bool]:
    contact_equal = bool(np.array_equal(db.contact[left], db.contact[right]))
    tread_equal = bool(np.all(
        (~db.contact[left]) | (db.tread_id[left] == db.tread_id[right])))
    origin_error = float(np.linalg.norm(
        db.stair_origin_root_xy[left] - db.stair_origin_root_xy[right]))
    yaw_error = float(abs(wrap_angle(
        db.stair_ascent_yaw_root[left] - db.stair_ascent_yaw_root[right])))
    planted = 0.0
    for foot in range(2):
        if db.contact[left, foot]:
            planted = max(planted, float(np.linalg.norm(
                db.feet_xyz_stair[left, foot] - db.feet_xyz_stair[right, foot])))
    return {
        "contact_equal": contact_equal, "tread_equal": tread_equal,
        "stair_origin_error_m": origin_error, "stair_yaw_error_rad": yaw_error,
        "planted_foot_error_m": planted,
        "passes": bool(contact_equal and tread_equal and origin_error <= 0.18
                       and yaw_error <= np.deg2rad(15.0) and planted <= 0.035),
        "passes_execution_10mm": bool(
            contact_equal and tread_equal and origin_error <= 0.18
            and yaw_error <= np.deg2rad(15.0) and planted <= 0.010),
    }


def render(catalog: Path, source_zarr: Path, output: Path, *,
           from_clip: str, from_frame: int, to_clip: str, to_frame: int,
           pre_frames: int = 70, post_frames: int = 10) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    import zarr

    db = load_database(catalog)
    report = json.loads(catalog.with_suffix(".json").read_text())
    names = report["clips"]
    clip_index = {name: i for i, name in enumerate(names)}
    lookup = {(int(c), int(f)): i for i, (c, f) in enumerate(
        zip(db.source_clip, db.source_frame, strict=True))}
    left = lookup[(clip_index[from_clip], from_frame)]
    right = lookup[(clip_index[to_clip], to_frame)]
    gate = _hard_gate(db, left, right)
    if not gate["passes"]:
        raise ValueError(f"transition fails hard gates: {gate}")

    source = zarr.open(str(source_zarr), mode="r")
    starts = np.asarray(source["clip_start_idx"], dtype=np.int64)
    ends = np.asarray(source["clip_end_idx"], dtype=np.int64)
    body_all = np.asarray(source["body_pos_w"], dtype=np.float32)
    joints_all = np.asarray(source["joint_pos"], dtype=np.float32)
    before_local = np.arange(max(0, from_frame - pre_frames), from_frame + 1)
    after_local = np.arange(to_frame, min(to_frame + post_frames + 1,
                                          ends[clip_index[to_clip]] - starts[clip_index[to_clip]]))
    before = starts[clip_index[from_clip]] + before_local
    after = starts[clip_index[to_clip]] + after_local
    indices = np.concatenate((before, after))
    body = body_all[indices]
    joints = joints_all[indices]
    switch = len(before)
    root_jump = float(np.linalg.norm(body[switch, 0] - body[switch - 1, 0]))
    foot_jump = float(np.max(np.linalg.norm(
        body[switch, (18, 19)] - body[switch - 1, (18, 19)], axis=1)))
    joint_jump = float(np.max(np.abs(joints[switch] - joints[switch - 1])))
    metrics = {
        "kind": "clean_kinematic_transition_not_physics_rollout",
        "from": {"clip": from_clip, "frame": from_frame},
        "to": {"clip": to_clip, "frame": to_frame},
        "hard_gate": gate,
        "root_position_jump_m": root_jump,
        "maximum_foot_position_jump_m": foot_jump,
        "maximum_joint_position_jump_rad": joint_jump,
        "frame_count": int(len(body)), "fps": 50,
    }

    index = {name: i for i, name in enumerate(BODY_NAMES)}
    edges = [(index[a], index[b]) for a, b in SKELETON_EDGES]
    figure = plt.figure(figsize=(12.8, 6.4))
    robot_ax = figure.add_subplot(1, 2, 1, projection="3d")
    map_ax = figure.add_subplot(1, 2, 2)
    robot_ax.view_init(elev=19, azim=-62)
    robot_ax.set_box_aspect((1.6, 1.0, 1.2))
    segments = [robot_ax.plot([], [], [], color="#222", lw=2.3)[0] for _ in edges]
    joints_plot = robot_ax.scatter([], [], [], s=11, color="#2878b5")
    feet_plot = robot_ax.scatter([np.nan, np.nan], [np.nan, np.nan],
                                 [np.nan, np.nan], s=55,
                                 color=["#377eb8", "#e41a1c"])
    root_history = robot_ax.plot([], [], [], color="#9467bd", lw=1.7)[0]
    # Fixed-world three-tread stair geometry.
    ox, oy = JUSTIN_STAIR_ORIGIN_WORLD_XY
    for tread in range(3):
        x_hi = ox - tread * JUSTIN_STAIR_RUN_M
        x_lo = x_hi - JUSTIN_STAIR_RUN_M
        z = (tread + 1) * JUSTIN_STAIR_RISE_M
        xx, yy = np.meshgrid([x_lo, x_hi],
                             [oy - JUSTIN_STAIR_WIDTH_M / 2, oy + JUSTIN_STAIR_WIDTH_M / 2])
        robot_ax.plot_surface(xx, yy, np.full_like(xx, z), color="#999", alpha=0.38)
    robot_ax.set_xlim(-1.25, 0.45)
    robot_ax.set_ylim(oy - 0.7, oy + 0.7)
    robot_ax.set_zlim(0, 1.65)
    robot_ax.set_title("Clean kinematics (not physics)")
    robot_ax.set_xlabel("world x")
    robot_ax.set_ylabel("world y")
    robot_ax.set_zlabel("z")

    map_ax.plot(body[:, 0, 0], body[:, 0, 1], color="#9467bd", lw=2)
    map_ax.scatter([body[switch - 1, 0, 0]], [body[switch - 1, 0, 1]],
                   marker="x", s=90, color="#d62728", label="clip switch")
    marker = map_ax.scatter([], [], s=60, color="#2878b5")
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.set_xlim(-1.25, 0.45)
    map_ax.set_ylim(oy - 0.7, oy + 0.7)
    map_ax.grid(alpha=0.25)
    map_ax.legend()
    map_ax.set_title("Root path on fixed stair")
    status = map_ax.text(0.02, 0.02, "", transform=map_ax.transAxes,
                         family="monospace", fontsize=9,
                         bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    figure.suptitle(f"Contact-safe switch: {from_clip}:{from_frame} → {to_clip}:{to_frame}")
    figure.tight_layout()

    def update(frame: int):
        points = body[frame]
        for line, (a, b) in zip(segments, edges, strict=True):
            pair = points[[a, b]]
            line.set_data_3d(pair[:, 0], pair[:, 1], pair[:, 2])
        joints_plot._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        feet_plot._offsets3d = (points[(18, 19), 0], points[(18, 19), 1],
                                points[(18, 19), 2])
        root_history.set_data_3d(body[:frame + 1, 0, 0], body[:frame + 1, 0, 1],
                                 body[:frame + 1, 0, 2])
        marker.set_offsets(points[0, :2][None, :])
        side = "PRE" if frame < switch else "POST"
        status.set_text(
            f"t={frame / 50:4.2f}s  {side}\n"
            f"root jump={root_jump*1000:5.1f} mm\n"
            f"foot jump={foot_jump*1000:5.1f} mm\n"
            f"joint jump={np.rad2deg(joint_jump):4.1f} deg\n"
            f"planted gate={float(gate['planted_foot_error_m'])*1000:4.1f} mm")
        return (*segments, joints_plot, feet_plot, root_history, marker, status)

    frames = np.arange(0, len(body), 2)
    animation = FuncAnimation(figure, update, frames=frames, interval=40, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(output), writer=FFMpegWriter(
        fps=25, codec="libx264", bitrate=3500,
        extra_args=("-pix_fmt", "yuv420p")), dpi=135)
    plt.close(figure)
    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-zarr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-clip", required=True)
    parser.add_argument("--from-frame", type=int, required=True)
    parser.add_argument("--to-clip", required=True)
    parser.add_argument("--to-frame", type=int, required=True)
    parser.add_argument("--pre-frames", type=int, default=70)
    parser.add_argument("--post-frames", type=int, default=10)
    args = parser.parse_args()
    metrics = render(args.catalog, args.source_zarr, args.output,
                     from_clip=args.from_clip, from_frame=args.from_frame,
                     to_clip=args.to_clip, to_frame=args.to_frame,
                     pre_frames=args.pre_frames, post_frames=args.post_frames)
    print(metrics)


if __name__ == "__main__":
    main()
