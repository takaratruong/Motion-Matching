"""Convert a synthesized terrain route into the tracker recovery NPZ schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .grail_terrain_source import (
    G1MujocoFK,
    MUJOCO_TO_ISAACLAB_JOINT,
    derive_clip_angular_velocity,
    derive_clip_velocity,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from .render_stitched_motion import load_stitched_motion_npz


def isaaclab_to_mujoco_joints(value: object) -> np.ndarray:
    joints = np.asarray(value)
    if joints.ndim < 1 or joints.shape[-1] != 29:
        raise ValueError("IsaacLab joints must end in width 29")
    inverse = np.argsort(MUJOCO_TO_ISAACLAB_JOINT)
    return np.ascontiguousarray(joints[..., inverse])


def build(motion_path: Path, output: Path, *, model_path: Path) -> Path:
    motion = load_stitched_motion_npz(motion_path)
    fps = float(motion.fps)
    if fps != 50.0:
        raise ValueError("tracker recovery references must be 50 Hz")
    roots = np.asarray(motion.root_position_world, dtype=np.float32)
    root_wxyz = np.asarray(
        motion.root_quaternion_world_wxyz, dtype=np.float32
    )
    joints = np.asarray(motion.joint_position, dtype=np.float32)
    payload = build_reference_payload(
        roots, root_wxyz, joints, fps=fps, model_path=model_path
    )
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return destination


def build_reference_payload(
    roots: np.ndarray,
    root_wxyz: np.ndarray,
    joints: np.ndarray,
    *,
    fps: float,
    model_path: Path,
) -> dict[str, np.ndarray]:
    """Materialize the reference matcher fields from canonical root/joints."""

    fk = G1MujocoFK(model_path)
    body = fk.forward(
        roots,
        wxyz_to_xyzw(root_wxyz),
        isaaclab_to_mujoco_joints(joints),
    )
    body_xyzw = np.asarray(body.body_quaternion_world_xyzw, dtype=np.float32)
    return {
        "fps": np.asarray(fps, dtype=np.float32),
        "joint_pos": joints,
        "joint_vel": derive_clip_velocity(joints, fps=fps),
        "body_pos_w": np.asarray(body.body_position_world, dtype=np.float32),
        "body_quat_w": xyzw_to_wxyz(body_xyzw).astype(np.float32),
        "body_lin_vel_w": derive_clip_velocity(
            body.body_position_world, fps=fps
        ),
        "body_ang_vel_w": derive_clip_angular_velocity(
            body_xyzw, fps=fps
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    arguments = parser.parse_args(argv)
    print(build(arguments.motion, arguments.output, model_path=arguments.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
