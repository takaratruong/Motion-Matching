"""Render an audited terrain splice with the articulated G1 MuJoCo mesh."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .gear_action import isaaclab_to_mujoco_joint_vector
from .offline_corpus import JOINT_NAMES
from .render_terrain_transition import _hard_gate
from .terrain_catalog import load_database


_MUJOCO_SOURCE_INDICES = (
    0, 3, 6, 9, 13, 17,
    1, 4, 7, 10, 14, 18,
    2, 5, 8,
    11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28,
)
MUJOCO_JOINT_NAMES = tuple(JOINT_NAMES[index] for index in _MUJOCO_SOURCE_INDICES)
DEFAULT_G1_STAIR_SCENE = Path(
    "/move/u/justingu/Projects/holosoma/src/holosoma_retargeting/"
    "holosoma_retargeting/demo_data/climb/up_pos_30/"
    "g1_29dof_spherehand_w_multi_boxes.xml"
)


def build_ffmpeg_label(
    from_clip: str,
    from_frame: int,
    to_clip: str,
    to_frame: int,
    switch_time_s: float,
    planted_foot_error_m: float,
) -> str:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    return (
        "drawbox=x=0:y=0:w=iw:h=76:color=black@0.76:t=fill,"
        f"drawtext=fontfile={font}:"
        "text='CLEAN KINEMATICS - NOT PHYSICS':"
        "fontcolor=yellow:fontsize=25:x=20:y=10,"
        f"drawtext=fontfile={font}:"
        f"text='{from_clip} frame {from_frame} -> {to_clip} frame {to_frame}"
        f" | switch t={switch_time_s:.2f}s | planted mismatch="
        f"{planted_foot_error_m * 1000:.1f} mm':"
        "fontcolor=white:fontsize=17:x=20:y=45"
    )


def build_qpos(
    root_position_xyz: np.ndarray,
    root_quaternion_xyzw: np.ndarray,
    joint_position_isaaclab: np.ndarray,
    model: object,
) -> np.ndarray:
    """Build MuJoCo qpos from the clean archive's native IsaacLab arrays."""
    import mujoco

    position = np.asarray(root_position_xyz, dtype=np.float64)
    quaternion = np.asarray(root_quaternion_xyzw, dtype=np.float64)
    joints = np.asarray(joint_position_isaaclab)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("root_position_xyz must contain three finite values")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("root_quaternion_xyzw must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("root quaternion must be nonzero")
    quaternion = quaternion / norm
    mapped = isaaclab_to_mujoco_joint_vector(joints)

    free = np.flatnonzero(
        np.asarray(model.jnt_type) == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    if free.size != 1:
        raise ValueError("G1 render model must contain exactly one free joint")
    qpos = np.zeros(int(model.nq), dtype=np.float64)
    root_address = int(model.jnt_qposadr[int(free[0])])
    qpos[root_address : root_address + 3] = position
    qpos[root_address + 3 : root_address + 7] = quaternion[[3, 0, 1, 2]]
    for value, name in zip(mapped, MUJOCO_JOINT_NAMES, strict=True):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"G1 render model is missing joint {name}")
        qpos[int(model.jnt_qposadr[joint_id])] = value
    return qpos


def _transition_indices(
    database: object,
    clip_names: list[str],
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    from_clip: str,
    from_frame: int,
    to_clip: str,
    to_frame: int,
    pre_frames: int,
    post_frames: int,
) -> tuple[np.ndarray, int, dict[str, float | bool]]:
    clip_index = {name: index for index, name in enumerate(clip_names)}
    lookup = {
        (int(clip), int(frame)): row
        for row, (clip, frame) in enumerate(
            zip(database.source_clip, database.source_frame, strict=True)
        )
    }
    left = lookup[(clip_index[from_clip], from_frame)]
    right = lookup[(clip_index[to_clip], to_frame)]
    gate = _hard_gate(database, left, right)
    if not gate["passes_execution_10mm"]:
        raise ValueError(f"transition fails the 10 mm execution gate: {gate}")
    before_local = np.arange(max(0, from_frame - pre_frames), from_frame + 1)
    to_index = clip_index[to_clip]
    to_count = int(ends[to_index] - starts[to_index])
    after_local = np.arange(to_frame, min(to_frame + post_frames + 1, to_count))
    before = starts[clip_index[from_clip]] + before_local
    after = starts[to_index] + after_local
    return np.concatenate((before, after)), len(before), gate


def render_mesh_transition(
    catalog: Path,
    source_zarr: Path,
    output: Path,
    *,
    scene_xml: Path = DEFAULT_G1_STAIR_SCENE,
    from_clip: str = "up_pos_30",
    from_frame: int = 214,
    to_clip: str = "up_pos_90",
    to_frame: int = 243,
    pre_frames: int = 70,
    post_frames: int = 50,
) -> Path:
    """Render the selected reference frames without stepping MuJoCo physics."""
    import mujoco
    import zarr

    database = load_database(catalog)
    catalog_report = json.loads(catalog.with_suffix(".json").read_text())
    clip_names = list(catalog_report["clips"])
    source = zarr.open(str(source_zarr), mode="r")
    if source.attrs.get("quaternion_convention") != "xyzw":
        raise ValueError("source archive must declare xyzw quaternions")
    starts = np.asarray(source["clip_start_idx"], dtype=np.int64)
    ends = np.asarray(source["clip_end_idx"], dtype=np.int64)
    indices, switch, gate = _transition_indices(
        database, clip_names, starts, ends,
        from_clip=from_clip, from_frame=from_frame,
        to_clip=to_clip, to_frame=to_frame,
        pre_frames=pre_frames, post_frames=post_frames,
    )
    root_position = np.asarray(source["body_pos_w"], dtype=np.float32)[
        indices, 0
    ]
    root_quaternion = np.asarray(source["body_quat_w"], dtype=np.float32)[
        indices, 0
    ]
    joint_position = np.asarray(source["joint_pos"], dtype=np.float32)[indices]

    model = mujoco.MjModel.from_xml_path(str(scene_xml.resolve()))
    qpos = np.stack(
        [
            build_qpos(position, quaternion, joints, model)
            for position, quaternion, joints in zip(
                root_position, root_quaternion, joint_position, strict=True
            )
        ]
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}.mp4")
    width, height, fps = 1280, 720, 25
    half_width = width // 2
    switch_time = (switch - 1) / 50.0
    label = build_ffmpeg_label(
        from_clip, from_frame, to_clip, to_frame, switch_time,
        float(gate["planted_foot_error_m"]),
    )
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-vf", label, "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(temporary),
    ]

    data = mujoco.MjData(model)
    fixed = mujoco.MjvCamera()
    tracking = mujoco.MjvCamera()
    midpoint = (root_position[:, :2].min(axis=0)
                + root_position[:, :2].max(axis=0)) / 2.0
    fixed.lookat[:] = (float(midpoint[0]), float(midpoint[1]), 0.75)
    fixed.distance = 3.0
    fixed.azimuth = 55.0
    fixed.elevation = -18.0
    tracking.distance = 2.1
    tracking.azimuth = 100.0
    tracking.elevation = -12.0
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), half_width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    left = right = encoder = None
    try:
        left = mujoco.Renderer(model, height=height, width=half_width)
        right = mujoco.Renderer(model, height=height, width=half_width)
        encoder = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert encoder.stdin is not None
        for frame in range(0, len(qpos), 2):
            data.qpos[:] = qpos[frame]
            data.time = frame / 50.0
            mujoco.mj_forward(model, data)
            tracking.lookat[:] = (
                float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])
            )
            left.update_scene(data, camera=fixed)
            right.update_scene(data, camera=tracking)
            combined = np.concatenate((left.render(), right.render()), axis=1)
            encoder.stdin.write(
                np.ascontiguousarray(combined, dtype=np.uint8).tobytes()
            )
        encoder.stdin.close()
        stderr = encoder.stderr.read() if encoder.stderr is not None else b""
        returncode = encoder.wait()
        encoder = None
        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr[-2000:]!r}")
        os.replace(temporary, output)
    finally:
        if left is not None:
            left.close()
        if right is not None:
            right.close()
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=10)
        temporary.unlink(missing_ok=True)

    metrics = {
        "kind": "actual_g1_mesh_clean_kinematics_not_physics",
        "video": str(output),
        "scene_xml": str(scene_xml.resolve()),
        "source_quaternion_convention": "xyzw",
        "mujoco_root_quaternion_convention": "wxyz",
        "from": {"clip": from_clip, "frame": from_frame},
        "to": {"clip": to_clip, "frame": to_frame},
        "switch_output_frame": int((switch - 1) // 2),
        "hard_gate": gate,
        "frame_count_source_50hz": int(len(qpos)),
        "frame_count_video_25hz": int((len(qpos) + 1) // 2),
    }
    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-zarr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_G1_STAIR_SCENE)
    parser.add_argument("--from-clip", default="up_pos_30")
    parser.add_argument("--from-frame", type=int, default=214)
    parser.add_argument("--to-clip", default="up_pos_90")
    parser.add_argument("--to-frame", type=int, default=243)
    parser.add_argument("--pre-frames", type=int, default=70)
    parser.add_argument("--post-frames", type=int, default=50)
    args = parser.parse_args()
    print(render_mesh_transition(
        args.catalog, args.source_zarr, args.output,
        scene_xml=args.scene_xml,
        from_clip=args.from_clip, from_frame=args.from_frame,
        to_clip=args.to_clip, to_frame=args.to_frame,
        pre_frames=args.pre_frames, post_frames=args.post_frames,
    ))


if __name__ == "__main__":
    main()
