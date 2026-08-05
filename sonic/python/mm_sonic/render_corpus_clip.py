"""Render one offline motion-matching clip with command diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SKELETON_EDGES = (
    ("pelvis", "left_hip_pitch_link"),
    ("left_hip_pitch_link", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_hip_yaw_link"),
    ("left_hip_yaw_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_pitch_link"),
    ("left_ankle_pitch_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_pitch_link"),
    ("right_hip_pitch_link", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_hip_yaw_link"),
    ("right_hip_yaw_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_pitch_link"),
    ("right_ankle_pitch_link", "right_ankle_roll_link"),
    ("pelvis", "waist_yaw_link"),
    ("waist_yaw_link", "waist_roll_link"),
    ("waist_roll_link", "torso_link"),
    ("torso_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_shoulder_yaw_link"),
    ("left_shoulder_yaw_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_roll_link"),
    ("left_wrist_roll_link", "left_wrist_pitch_link"),
    ("left_wrist_pitch_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_shoulder_yaw_link"),
    ("right_shoulder_yaw_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_roll_link"),
    ("right_wrist_roll_link", "right_wrist_pitch_link"),
    ("right_wrist_pitch_link", "right_wrist_yaw_link"),
)


def _clip_slice(root, clip_id: str) -> slice:
    names = [str(value) for value in np.asarray(root["meta/episode_clip"])]
    if clip_id not in names:
        raise ValueError(f"unknown clip {clip_id!r}; choices: {names}")
    ends = np.asarray(root["meta/episode_ends"], dtype=np.int64)
    index = names.index(clip_id)
    start = 0 if index == 0 else int(ends[index - 1])
    return slice(start, int(ends[index]))


def render(
    corpus: Path,
    clip_id: str,
    output: Path,
    *,
    fps: int = 25,
    source_fps: int = 50,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    import zarr

    root = zarr.open(str(corpus.expanduser().resolve()), mode="r")
    select = _clip_slice(root, clip_id)
    body = np.asarray(root["data/body_pos_w"][select], dtype=np.float32)
    command_path = np.asarray(
        root["data/command_path_world_xy"][select], dtype=np.float32
    )
    requested_path = (
        np.asarray(
            root["data/command_requested_path_world_xy"][select],
            dtype=np.float32,
        )
        if "command_requested_path_world_xy" in root["data"]
        else None
    )
    command_heading = np.asarray(
        root["data/command_shaped_heading_world_yaw"][select], dtype=np.float32
    )
    requested_velocity = np.asarray(
        root["data/command_requested_velocity_world_xy"][select],
        dtype=np.float32,
    )
    raw_sticks = np.asarray(
        root["data/command_raw_sticks"][select], dtype=np.float32
    )
    ball_velocity = (
        np.asarray(
            root["data/command_ball_velocity_world_xy"][select],
            dtype=np.float32,
        )
        if "command_ball_velocity_world_xy" in root["data"]
        else None
    )
    clamp_distance = (
        np.asarray(
            root["data/matcher_root_clamp_distance_m"][select],
            dtype=np.float32,
        )
        if "matcher_root_clamp_distance_m" in root["data"]
        else None
    )
    body_names = [str(value) for value in np.asarray(root["meta/body_names"])]
    body_index = {name: index for index, name in enumerate(body_names)}
    edges = tuple((body_index[a], body_index[b]) for a, b in SKELETON_EDGES)

    root_xy = body[:, 0, :2]
    root_relative = root_xy - root_xy[0]
    intended_relative = command_path - command_path[0]
    requested_relative = (
        requested_path - requested_path[0] if requested_path is not None else None
    )
    intended_world = intended_relative + root_xy[0]
    stride = max(1, round(source_fps / fps))
    frames = np.arange(0, len(body), stride, dtype=np.int64)

    figure = plt.figure(figsize=(12.8, 6.4))
    robot_axis = figure.add_subplot(1, 2, 1, projection="3d")
    map_axis = figure.add_subplot(1, 2, 2)
    robot_axis.view_init(elev=18, azim=-65)
    robot_axis.set_box_aspect((1.0, 1.0, 1.25))
    robot_axis.set_zlim(0.0, 1.8)
    robot_axis.set_xlabel("world x")
    robot_axis.set_ylabel("world y")
    robot_axis.set_zlabel("z")
    robot_axis.set_title("Clean motion-matched kinematics")

    segments = [
        robot_axis.plot([], [], [], color="#222222", linewidth=2.5)[0]
        for _edge in edges
    ]
    joints = robot_axis.scatter([], [], [], color="#1f77b4", s=12)
    actual_history = robot_axis.plot(
        [], [], [], color="#1f77b4", linewidth=1.5, label="generated root"
    )[0]
    intended_history = robot_axis.plot(
        [], [], [], color="#d62728", linewidth=1.8, label="command ball"
    )[0]
    ball = robot_axis.scatter([], [], [], color="#d62728", s=75)
    robot_axis.legend(loc="upper left")

    map_axis.plot(
        intended_relative[:, 0],
        intended_relative[:, 1],
        color="#d62728",
        linewidth=2.2,
        label="intended / command ball",
    )
    if requested_relative is not None:
        map_axis.plot(
            requested_relative[:, 0],
            requested_relative[:, 1],
            color="#9467bd",
            linewidth=1.2,
            linestyle="--",
            alpha=0.85,
            label="raw stick integral",
        )
    map_axis.plot(
        root_relative[:, 0],
        root_relative[:, 1],
        color="#1f77b4",
        linewidth=2.0,
        label="generated root",
    )
    actual_marker = map_axis.scatter([], [], color="#1f77b4", s=55, zorder=5)
    intended_marker = map_axis.scatter([], [], color="#d62728", s=55, zorder=5)
    facing_arrow = map_axis.quiver(
        [0.0],
        [0.0],
        [1.0],
        [0.0],
        color="#2ca02c",
        angles="xy",
        scale_units="xy",
        scale=2.0,
        width=0.009,
        label="requested facing",
    )
    all_paths = [intended_relative, root_relative]
    if requested_relative is not None:
        all_paths.append(requested_relative)
    all_xy = np.concatenate(all_paths, axis=0)
    padding = 0.18
    map_axis.set_xlim(
        float(all_xy[:, 0].min() - padding),
        float(all_xy[:, 0].max() + padding),
    )
    map_axis.set_ylim(
        float(all_xy[:, 1].min() - padding),
        float(all_xy[:, 1].max() + padding),
    )
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.grid(alpha=0.25)
    map_axis.set_xlabel("x from clip start (m)")
    map_axis.set_ylabel("y from clip start (m)")
    map_axis.set_title("Intended versus generated path")
    map_axis.legend(loc="best")
    status = map_axis.text(
        0.02,
        0.02,
        "",
        transform=map_axis.transAxes,
        ha="left",
        va="bottom",
        family="monospace",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    figure.suptitle(f"Takara motion matching: {clip_id}")
    figure.tight_layout()

    def update(frame: int):
        points = body[frame]
        center = points[0, :2]
        robot_axis.set_xlim(center[0] - 1.0, center[0] + 1.0)
        robot_axis.set_ylim(center[1] - 1.0, center[1] + 1.0)
        for line, (left, right) in zip(segments, edges, strict=True):
            pair = points[[left, right]]
            line.set_data_3d(pair[:, 0], pair[:, 1], pair[:, 2])
        joints._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        actual_history.set_data_3d(
            root_xy[: frame + 1, 0],
            root_xy[: frame + 1, 1],
            np.full(frame + 1, 0.025),
        )
        intended_history.set_data_3d(
            intended_world[: frame + 1, 0],
            intended_world[: frame + 1, 1],
            np.full(frame + 1, 0.04),
        )
        ball._offsets3d = (
            [intended_world[frame, 0]],
            [intended_world[frame, 1]],
            [0.06],
        )
        actual_marker.set_offsets(root_relative[frame : frame + 1])
        intended_marker.set_offsets(intended_relative[frame : frame + 1])
        heading = command_heading[frame]
        facing_arrow.set_offsets(root_relative[frame : frame + 1])
        facing_arrow.set_UVC([np.cos(heading)], [np.sin(heading)])
        left = raw_sticks[frame, :2]
        right = raw_sticks[frame, 2:]
        velocity = requested_velocity[frame]
        filtered_velocity = (
            ball_velocity[frame] if ball_velocity is not None else velocity
        )
        clamp = float(clamp_distance[frame]) if clamp_distance is not None else 0.0
        separation = np.linalg.norm(
            root_relative[frame] - intended_relative[frame]
        )
        status.set_text(
            f"t={frame / source_fps:5.2f}s  separation={separation:4.2f} m\n"
            f"left=({left[0]:+.2f},{left[1]:+.2f})  "
            f"right=({right[0]:+.2f},{right[1]:+.2f})\n"
            f"requested v=({velocity[0]:+.2f},{velocity[1]:+.2f}) m/s\n"
            f"ball v=({filtered_velocity[0]:+.2f},"
            f"{filtered_velocity[1]:+.2f}) m/s  clamp={clamp:.3f} m"
        )
        return (
            *segments,
            joints,
            actual_history,
            intended_history,
            ball,
            actual_marker,
            intended_marker,
            facing_arrow,
            status,
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=frames,
        interval=1000.0 / fps,
        blit=False,
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        str(output),
        writer=FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=3200,
            extra_args=("-pix_fmt", "yuv420p"),
        ),
        dpi=135,
    )
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    render(args.corpus, args.clip, args.output, fps=args.fps)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
