"""Render a captured HCT rollout as an articulated G1 on its terrain.

Isaac Sim's RTX path is unreliable on the MOVE nodes, so this performs pure
kinematic playback in MuJoCo.  New archives contain the exact simulator terrain
mesh; older archives fall back to the moving 17x11 policy height scan.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial import Delaunay

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402


DEFAULT_G1_SCENE = Path("/move/u/justingu/TML-BeyondMimic/assets/g1/scene_29dof.xml")
ALIASES = {
    "torso_joint": "waist_yaw_joint",
    "left_elbow_pitch_joint": "left_elbow_joint",
    "right_elbow_pitch_joint": "right_elbow_joint",
    "left_elbow_roll_joint": "left_wrist_roll_joint",
    "right_elbow_roll_joint": "right_wrist_roll_joint",
}
LEG_JOINTS = {
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
}


def reconstruct_scan_mesh(scan_xyz: np.ndarray, *, bin_size_m: float = 0.055) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(scan_xyz, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 100:
        raise ValueError("too few finite height-scan hits to reconstruct terrain")

    # Merge the heavily overlapping moving scans without averaging across a
    # vertical step: retain the highest sampled surface in each planar bin.
    key = np.rint(points[:, :2] / bin_size_m).astype(np.int64)
    order = np.lexsort((key[:, 1], key[:, 0]))
    points = points[order]
    key = key[order]
    boundary = np.r_[True, np.any(key[1:] != key[:-1], axis=1)]
    start = np.flatnonzero(boundary)
    stop = np.r_[start[1:], len(points)]
    vertices = np.empty((len(start), 3), dtype=np.float64)
    for output, (left, right) in enumerate(zip(start, stop, strict=True)):
        group = points[left:right]
        highest = int(np.argmax(group[:, 2]))
        vertices[output] = group[highest]

    triangulation = Delaunay(vertices[:, :2], qhull_options="QJ")
    faces = np.asarray(triangulation.simplices, dtype=np.int32)
    triangle = vertices[faces]
    edge = np.stack((
        np.linalg.norm(triangle[:, 0, :2] - triangle[:, 1, :2], axis=1),
        np.linalg.norm(triangle[:, 1, :2] - triangle[:, 2, :2], axis=1),
        np.linalg.norm(triangle[:, 2, :2] - triangle[:, 0, :2], axis=1),
    ), axis=1)
    # Do not bridge unobserved gaps between disjoint parts of a route.
    faces = faces[np.max(edge, axis=1) <= 0.18]
    if len(faces) < 50:
        raise ValueError("height scans did not form a connected terrain corridor")
    return vertices.astype(np.float32), faces


def crop_exact_terrain_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    root_xy: np.ndarray,
    *,
    margin_m: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep exact triangles intersecting a padded rollout bounding box."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    lower = np.min(root_xy, axis=0) - margin_m
    upper = np.max(root_xy, axis=0) + margin_m
    triangle_xy = vertices[faces, :2]
    selected = np.all(np.max(triangle_xy, axis=1) >= lower, axis=1)
    selected &= np.all(np.min(triangle_xy, axis=1) <= upper, axis=1)
    selected_faces = faces[selected]
    if len(selected_faces) < 50:
        raise ValueError("exact terrain crop contains too few triangles")
    used, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3).astype(np.int32)


def _choose_env(data: np.lib.npyio.NpzFile, terrain: str | None, program: str | None) -> int:
    selected = np.asarray(data["clean_envs"], dtype=bool)
    if terrain is not None:
        selected &= np.asarray(data["terrain_name"]).astype(str) == terrain
    if program is not None:
        selected &= np.asarray(data["program_name"]).astype(str) == program
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise ValueError(f"no clean rollout matches terrain={terrain!r}, program={program!r}")
    score = np.asarray(data["command_rmse"], dtype=np.float64)[indices]
    return int(indices[int(np.argmin(score))])


def _build_model(vertices: np.ndarray, faces: np.ndarray, scene: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(scene.resolve()))
    for geom in spec.geoms:
        if geom.name == "floor":
            geom.pos = (0.0, 0.0, -10.0)
    spec.add_mesh(
        name="observed_terrain",
        uservert=vertices.ravel().tolist(),
        userface=faces.ravel().tolist(),
        smoothnormal=0,
        inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
    )
    spec.worldbody.add_geom(
        name="observed_terrain",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="observed_terrain",
        rgba=(0.30, 0.39, 0.27, 1.0),
        contype=0,
        conaffinity=0,
    )
    return spec.compile()


def _joint_mapping(model: mujoco.MjModel, names: list[str]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    mapped_names = set()
    for source_index, source_name in enumerate(names):
        target_name = ALIASES.get(source_name, source_name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target_name)
        if joint_id >= 0:
            mapping[source_index] = int(model.jnt_qposadr[joint_id])
            mapped_names.add(target_name)
    missing = sorted(LEG_JOINTS - mapped_names)
    if missing:
        raise ValueError(f"render model is missing gait joints: {missing}")
    return mapping


def render(
    source: Path,
    output: Path,
    *,
    env_index: int | None,
    terrain: str | None,
    program: str | None,
    scene: Path,
    stride: int,
) -> Path:
    archive = np.load(source, allow_pickle=False)
    if env_index is None:
        env_index = _choose_env(archive, terrain, program)
    count = int(archive["root_pos"].shape[1])
    if env_index < 0 or env_index >= count:
        raise IndexError(f"env {env_index} is outside [0,{count})")

    root_position = np.asarray(archive["root_pos"][:, env_index], dtype=np.float64)[::stride]
    root_quaternion = np.asarray(archive["root_quat"][:, env_index], dtype=np.float64)[::stride]
    joint_position = np.asarray(archive["joint_pos"][:, env_index], dtype=np.float64)[::stride]
    scan = np.asarray(archive["scan_xyz"][:, env_index], dtype=np.float64)
    command = np.asarray(archive["commands"][:, env_index], dtype=np.float64)[::stride]
    joint_names = [str(value) for value in archive["joint_names"]]
    terrain_name = str(archive["terrain_name"][env_index])
    program_name = str(archive["program_name"][env_index])
    dt = float(archive["dt"]) * stride

    if "terrain_vertices_world" in archive.files and "terrain_faces" in archive.files:
        vertices, faces = crop_exact_terrain_mesh(
            archive["terrain_vertices_world"], archive["terrain_faces"], root_position[:, :2]
        )
        terrain_source = "exact Isaac simulator terrain mesh"
    else:
        vertices, faces = reconstruct_scan_mesh(scan)
        terrain_source = "terrain reconstructed from policy height scan"
    model = _build_model(vertices, faces, scene)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 640)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 720)
    data = mujoco.MjData(model)
    mapping = _joint_mapping(model, joint_names)
    midpoint = 0.5 * (root_position[:, :2].min(axis=0) + root_position[:, :2].max(axis=0))
    span = float(np.linalg.norm(root_position[:, :2].max(axis=0) - root_position[:, :2].min(axis=0)))
    ground_mid = float(np.median(vertices[:, 2]))
    fixed = mujoco.MjvCamera()
    fixed.lookat[:] = (float(midpoint[0]), float(midpoint[1]), ground_mid + 0.55)
    fixed.distance = max(3.0, 1.15 * span + 1.8)
    fixed.azimuth = 125.0
    fixed.elevation = -28.0
    tracking = mujoco.MjvCamera()
    tracking.distance = 2.3
    tracking.azimuth = 125.0
    tracking.elevation = -14.0

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}.mp4")
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    label = (
        "drawbox=x=0:y=0:w=iw:h=70:color=black@0.72:t=fill,"
        f"drawtext=fontfile={font}:text='{terrain_name} | {program_name} | env {env_index}':"
        "fontcolor=white:fontsize=22:x=18:y=10,"
        f"drawtext=fontfile={font}:text='actual simulated G1 state | {terrain_source}':"
        "fontcolor=yellow:fontsize=16:x=18:y=42"
    )
    fps = max(1, int(round(1.0 / dt)))
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", "1280x720", "-r", str(fps), "-i", "-", "-vf", label,
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    left = mujoco.Renderer(model, height=720, width=640)
    right = mujoco.Renderer(model, height=720, width=640)
    try:
        assert ffmpeg.stdin is not None
        for frame in range(len(root_position)):
            data.qpos[:] = model.qpos0
            data.qpos[:3] = root_position[frame]
            data.qpos[3:7] = root_quaternion[frame]
            for source_index, address in mapping.items():
                data.qpos[address] = joint_position[frame, source_index]
            mujoco.mj_forward(model, data)
            tracking.lookat[:] = root_position[frame]
            left.update_scene(data, camera=fixed)
            right.update_scene(data, camera=tracking)
            image = np.concatenate((left.render(), right.render()), axis=1)
            ffmpeg.stdin.write(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
        ffmpeg.stdin.close()
        error = ffmpeg.stderr.read() if ffmpeg.stderr is not None else b""
        returncode = ffmpeg.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {error[-2000:]!r}")
        os.replace(temporary, output)
    finally:
        left.close()
        right.close()
        if ffmpeg.poll() is None:
            ffmpeg.terminate()
            ffmpeg.wait(timeout=10)
        temporary.unlink(missing_ok=True)

    metadata = {
        "source": str(source.resolve()),
        "video": str(output),
        "env_index": env_index,
        "terrain": terrain_name,
        "program": program_name,
        "frame_count": len(root_position),
        "fps": fps,
        "command_rmse": float(archive["command_rmse"][env_index]),
        "clean": bool(archive["clean_envs"][env_index]),
        "terrain_vertices": len(vertices),
        "terrain_faces": len(faces),
        "terrain_source": terrain_source,
        "sole_clearance_note": (
            "Not inferred across the Isaac/MuJoCo asset boundary; use the recorded Isaac "
            "contacts and exact-terrain visual review."
        ),
        "command_range": {
            "minimum": command.min(axis=0).tolist(),
            "maximum": command.max(axis=0).tolist(),
        },
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=int)
    parser.add_argument("--terrain")
    parser.add_argument("--program")
    parser.add_argument("--scene", type=Path, default=DEFAULT_G1_SCENE)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("stride must be positive")
    render(
        args.source,
        args.output,
        env_index=args.env,
        terrain=args.terrain,
        program=args.program,
        scene=args.scene,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
