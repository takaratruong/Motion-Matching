"""Render a stitched clean kinematic motion on one exact stairs500 mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Sequence

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

from .terrain_oracle.math3d import RigidTransform
from .terrain_oracle.render_media import _build_scene_model
from .terrain_oracle.source_grail import _load_usd_mesh
from .terrain_oracle.stitch import FrameProvenance, StitchedMotion
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


def _populate_qpos(
    model: object,
    data: object,
    *,
    root_position_world: np.ndarray,
    root_quaternion_world_wxyz: np.ndarray,
    joint_position: np.ndarray,
    joint_addresses: tuple[int, ...],
    root_address: int,
) -> None:
    data.qpos[root_address : root_address + 3] = root_position_world
    data.qpos[root_address + 3 : root_address + 7] = (
        root_quaternion_world_wxyz
    )
    for address, value in zip(
        joint_addresses, joint_position, strict=True
    ):
        data.qpos[address] = value


def load_stitched_motion_npz(path: str | Path) -> StitchedMotion:
    """Load the lossless motion bundle written by the terrain composers."""

    with np.load(Path(path).expanduser().resolve(), allow_pickle=False) as data:
        roots = np.asarray(data["root_position_world"], dtype=np.float32)
        clip_indices = np.asarray(
            data["source_archive_clip_index"], dtype=np.int64
        )
        source_frames = np.asarray(data["source_frame"], dtype=np.int64)
        clip_ids = np.asarray(data["source_clip_id"])
        provenance = tuple(
            FrameProvenance(int(clip), int(frame), str(clip_id))
            for clip, frame, clip_id in zip(
                clip_indices, source_frames, clip_ids, strict=True
            )
        )
        return StitchedMotion(
            fps=float(np.asarray(data["fps"])),
            root_position_world=roots,
            root_quaternion_world_wxyz=np.asarray(
                data["root_quaternion_world_wxyz"], dtype=np.float32
            ),
            joint_position=np.asarray(data["joint_position"], dtype=np.float32),
            provenance=provenance,
            seam_indices=tuple(
                int(value) for value in np.asarray(data["seam_indices"])
            ),
        )


def render_stitched_motion(
    motion: StitchedMotion,
    *,
    model_path: str | Path,
    output_path: str | Path,
    archive_path: str | Path | None = None,
    target_clip_index: int | None = None,
    target_mesh: TerrainMeshIndex | None = None,
    joint_names: Sequence[str] | None = None,
    width: int = 960,
    height: int = 540,
    frame_stride: int = 1,
    camera_azimuth_offset_deg: float = 90.0,
    camera_elevation_deg: float = -16.0,
    camera_distance: float = 2.8,
    camera_follow_root: bool = True,
) -> dict[str, object]:
    """Render articulated G1 kinematics without stepping physics."""

    import mujoco
    import zarr

    if len(motion.root_position_world) == 0:
        raise ValueError("stitched motion is empty")
    archive = None
    if target_mesh is None or joint_names is None:
        if archive_path is None:
            raise ValueError(
                "archive_path is required unless target_mesh and joint_names "
                "are supplied"
            )
        archive = zarr.open_group(str(Path(archive_path)), mode="r")
    target_index = (
        None if target_clip_index is None else int(target_clip_index)
    )
    terrain_path = None
    if target_mesh is None:
        if target_index is None:
            raise ValueError("target_mesh or target_clip_index is required")
        terrain_path = Path(str(archive["terrain_usd_path"][target_index]))
        terrain_transform = RigidTransform(
            np.asarray(archive["terrain_position_env"][target_index]),
            np.asarray(archive["terrain_rotation_env_wxyz"][target_index]),
        )
        mesh = _load_usd_mesh(
            terrain_path,
            source_asset_sha256="0" * 64,
        )
        target_mesh = TerrainMeshIndex(mesh, terrain_transform)
    elif not isinstance(target_mesh, TerrainMeshIndex):
        raise TypeError("target_mesh must be a TerrainMeshIndex")
    mesh = target_mesh.mesh
    terrain_transform = target_mesh.world_from_terrain
    spec = mujoco.MjSpec.from_file(str(Path(model_path).resolve()))
    model, visual_mesh_count = _build_scene_model(
        spec, mesh, terrain_transform
    )
    # The robot XML carries its own flat ground plane.  When an explicit
    # terrain mesh is rendered at z=0 the coplanar surfaces z-fight, producing
    # the rapidly changing floor pattern seen in the interactive viewer.
    # Keeping the plane invisible fixed that pattern but made the flat context
    # outside finite ramp meshes look like flight.  Lower the render-only plane
    # by one centimetre instead: it remains visible outside the terrain asset,
    # cannot z-fight with the exact mesh, and stays collision-disabled.
    for ground_name in ("ground", "floor"):
        ground_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, ground_name
        )
        if ground_id >= 0:
            model.geom_pos[ground_id, 2] -= 0.01
            model.geom_contype[ground_id] = 0
            model.geom_conaffinity[ground_id] = 0
    data = mujoco.MjData(model)

    root_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
    )
    if root_id < 0:
        raise ValueError("G1 model has no floating_base_joint")
    root_address = int(model.jnt_qposadr[root_id])
    if joint_names is None:
        assert archive is not None
        joint_names = tuple(str(value) for value in archive["joint_names"][:])
    else:
        joint_names = tuple(str(value) for value in joint_names)
    joint_addresses: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"G1 model is missing joint {name}")
        joint_addresses.append(int(model.jnt_qposadr[joint_id]))

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.stem}.tmp-{os.getpid()}{output.suffix}"
    )
    stride = int(frame_stride)
    if stride <= 0:
        raise ValueError("frame_stride must be positive")
    rendered_frames = list(range(0, len(motion.root_position_world), stride))
    if rendered_frames[-1] != len(motion.root_position_world) - 1:
        rendered_frames.append(len(motion.root_position_world) - 1)
    fps = float(motion.fps) / float(stride)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]

    displacement = (
        np.asarray(motion.root_position_world[-1, :2], dtype=np.float64)
        - np.asarray(motion.root_position_world[0, :2], dtype=np.float64)
    )
    travel_yaw_deg = math.degrees(
        math.atan2(float(displacement[1]), float(displacement[0]))
    )
    camera = mujoco.MjvCamera()
    camera.distance = float(camera_distance)
    camera.azimuth = travel_yaw_deg + float(camera_azimuth_offset_deg)
    camera.elevation = float(camera_elevation_deg)
    static_lookat = np.mean(motion.root_position_world, axis=0)
    static_lookat[2] -= 0.05
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)

    renderer = encoder = None
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        encoder = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert encoder.stdin is not None
        for frame in rendered_frames:
            _populate_qpos(
                model,
                data,
                root_position_world=motion.root_position_world[frame],
                root_quaternion_world_wxyz=(
                    motion.root_quaternion_world_wxyz[frame]
                ),
                joint_position=motion.joint_position[frame],
                joint_addresses=tuple(joint_addresses),
                root_address=root_address,
            )
            data.time = frame / float(motion.fps)
            mujoco.mj_forward(model, data)
            if camera_follow_root:
                camera.lookat[:] = motion.root_position_world[frame]
                camera.lookat[2] -= 0.05
            else:
                camera.lookat[:] = static_lookat
            renderer.update_scene(data, camera=camera)
            encoder.stdin.write(
                np.ascontiguousarray(
                    renderer.render(), dtype=np.uint8
                ).tobytes()
            )
        encoder.stdin.close()
        errors = encoder.stderr.read() if encoder.stderr is not None else b""
        if encoder.stderr is not None:
            encoder.stderr.close()
        returncode = encoder.wait()
        encoder = None
        if returncode != 0:
            raise RuntimeError(
                "ffmpeg failed: "
                + errors.decode("utf-8", errors="replace")[-2000:]
            )
        os.replace(temporary, output)
    finally:
        if renderer is not None:
            renderer.close()
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=10)
        temporary.unlink(missing_ok=True)

    return {
        "video": str(output),
        "frame_count": len(rendered_frames),
        "source_frame_count": len(motion.root_position_world),
        "frame_stride": stride,
        "camera_azimuth_offset_deg": float(camera_azimuth_offset_deg),
        "camera_elevation_deg": float(camera_elevation_deg),
        "camera_distance": float(camera_distance),
        "camera_follow_root": bool(camera_follow_root),
        "fps": fps,
        "duration_s": (len(motion.root_position_world) - 1)
        / float(motion.fps),
        "target_clip_index": target_index,
        "terrain_usd_path": (
            None if terrain_path is None else str(terrain_path)
        ),
        "terrain_face_count": int(np.sum(mesh.valid_faces)),
        "visual_mesh_geom_count": visual_mesh_count,
        "seam_indices": list(motion.seam_indices),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--terrain-usd", type=Path, required=True)
    parser.add_argument("--terrain-position", type=float, nargs=3, default=(0, 0, 0))
    parser.add_argument(
        "--terrain-quaternion-wxyz",
        type=float,
        nargs=4,
        default=(1, 0, 0, 0),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "/move/data/terrain-aware/motion-matching/"
            "grail-stairs500-clean-50hz-v1.zarr"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/move/u/justingu/Projects/TWIST2/assets/g1/"
            "g1_29dof_rev_1_0.xml"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    terrain_path = arguments.terrain_usd.expanduser().resolve()
    mesh = _load_usd_mesh(
        terrain_path,
        source_asset_sha256=hashlib.sha256(terrain_path.read_bytes()).hexdigest(),
    )
    target_mesh = TerrainMeshIndex(
        mesh,
        RigidTransform(
            np.asarray(arguments.terrain_position, dtype=np.float32),
            np.asarray(arguments.terrain_quaternion_wxyz, dtype=np.float32),
        ),
    )
    result = render_stitched_motion(
        load_stitched_motion_npz(arguments.motion),
        archive_path=arguments.archive,
        target_mesh=target_mesh,
        model_path=arguments.model,
        output_path=arguments.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = ("load_stitched_motion_npz", "render_stitched_motion")


if __name__ == "__main__":
    raise SystemExit(main())
