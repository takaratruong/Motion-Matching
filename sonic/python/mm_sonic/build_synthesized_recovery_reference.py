"""Convert a synthesized terrain route into the tracker recovery NPZ schema."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaw_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(quaternion), -1, 0)
    return np.arctan2(
        2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
    )


def _reference_task12(payload: dict[str, np.ndarray]) -> np.ndarray:
    """Apply the training corpus' canonical command inference to this motion."""

    root_quat = np.asarray(payload["body_quat_w"][:, 0], dtype=np.float64)
    root_velocity = np.asarray(payload["body_lin_vel_w"][:, 0], dtype=np.float64)
    root_angular = np.asarray(payload["body_ang_vel_w"][:, 0], dtype=np.float64)
    yaw = _yaw_from_wxyz(root_quat)
    cosine = np.cos(-yaw)
    sine = np.sin(-yaw)
    local_x = cosine * root_velocity[:, 0] - sine * root_velocity[:, 1]
    local_y = sine * root_velocity[:, 0] + cosine * root_velocity[:, 1]
    command3 = np.stack((local_x, local_y, root_angular[:, 2]), axis=1)
    offsets = np.asarray((6, 12, 18, 24), dtype=np.int64)
    frames = np.arange(len(command3), dtype=np.int64)[:, None]
    future = np.minimum(frames + offsets[None, :], len(command3) - 1)
    return np.ascontiguousarray(command3[future].reshape(len(command3), 12), dtype=np.float32)


def build(
    motion_path: Path,
    output: Path,
    *,
    model_path: Path,
    scene_contract_path: Path | None = None,
) -> Path:
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
    payload.update(
        {
            "source_archive_clip_index": np.asarray(
                [value.archive_clip_index for value in motion.provenance],
                dtype=np.int64,
            ),
            "source_frame": np.asarray(
                [value.source_frame for value in motion.provenance],
                dtype=np.int64,
            ),
            "source_clip_id": np.asarray(
                [str(value.clip_id) for value in motion.provenance],
                dtype=np.str_,
            ),
        }
    )
    if scene_contract_path is not None:
        contract_source = scene_contract_path.expanduser().resolve()
        contract = json.loads(contract_source.read_text())
        payload.update(
            {
                "task12": _reference_task12(payload),
                "command_schedule_kind": np.asarray(
                    "canonical_kinematic_inference_from_warped_expert_reference"
                ),
                "reference_scene_id": np.asarray(str(contract["scene_id"])),
                "reference_geometry_sha256": np.asarray(
                    str(contract["geometry_sha256"])
                ),
                "scene_contract_sha256": np.asarray(_sha256(contract_source)),
            }
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
    parser.add_argument("--scene-contract", type=Path)
    arguments = parser.parse_args(argv)
    print(
        build(
            arguments.motion,
            arguments.output,
            model_path=arguments.model,
            scene_contract_path=arguments.scene_contract,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
