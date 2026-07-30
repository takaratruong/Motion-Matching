"""Strict native Takara motion-folder loader for the in-process Torch matcher.

This module loads Takara-format G1 motion folders directly, keeping the native
Z-up / wxyz representation described in the design contract. It performs no
Torch import so that baseline test discovery never depends on the optional
runtime, and it never resamples or reorders clips.

The frozen layout identity is ``g1-29dof-isaaclab-v1``: 29 IsaacLab joints, 30
bodies, pelvis body index ``0``, and feet body indices ``18``/``19``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .joints import ContractError

# The minimum searchable feature horizon: the final SONIC source offset is 45,
# so a clip needs at least 46 frames and only frames ``0..T-46`` are searchable.
_FINAL_SOURCE_OFFSET = 45
_MIN_FRAMES = _FINAL_SOURCE_OFFSET + 1
_QUATERNION_NORM_TOLERANCE = 1e-4


@dataclass(frozen=True)
class MotionLayout:
    identity: str
    joint_count: int
    body_count: int
    root_body_index: int
    left_foot_body_index: int
    right_foot_body_index: int


G1_TAKARA_LAYOUT = MotionLayout(
    identity="g1-29dof-isaaclab-v1",
    joint_count=29,
    body_count=30,
    root_body_index=0,
    left_foot_body_index=18,
    right_foot_body_index=19,
)


@dataclass(frozen=True)
class MotionClip:
    relative_path: str
    fps: int
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_position_world: np.ndarray
    body_quaternion_world_wxyz: np.ndarray
    body_linear_velocity_world: np.ndarray
    body_angular_velocity_world: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.joint_position.shape[0])

    @property
    def valid_frame_stop(self) -> int:
        return self.frame_count - _FINAL_SOURCE_OFFSET


@dataclass(frozen=True)
class MotionFolder:
    root: Path
    layout: MotionLayout
    clips: Sequence[MotionClip]
    inventory_sha256: str

    @staticmethod
    def load(
        root: "str | Path",
        *,
        layout: MotionLayout = G1_TAKARA_LAYOUT,
    ) -> "MotionFolder":
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            raise ContractError(f"motion root is not a directory: {root_path}")

        paths = discover_motion_paths(root_path)
        if not paths:
            raise ContractError(
                f"motion root contains at least one motion.npz: {root_path}"
            )

        clips: list[MotionClip] = []
        inventory = hashlib.sha256()
        for path in paths:
            relative = path.relative_to(root_path).as_posix()
            clip = _load_clip(path, relative, layout)
            clips.append(clip)

            raw = path.read_bytes()
            inventory.update(relative.encode("utf-8"))
            inventory.update(b"\x00")
            inventory.update(str(len(raw)).encode("ascii"))
            inventory.update(b"\x00")
            inventory.update(hashlib.sha256(raw).hexdigest().encode("ascii"))

        return MotionFolder(
            root=root_path,
            layout=layout,
            clips=tuple(clips),
            inventory_sha256=inventory.hexdigest(),
        )


def discover_motion_paths(root: "str | Path") -> list[Path]:
    """Return every ``motion.npz`` under ``root`` sorted by POSIX relative path."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ContractError(f"motion root is not a directory: {root_path}")
    found = list(root_path.rglob("motion.npz"))
    return sorted(found, key=lambda p: p.relative_to(root_path).as_posix())


def _fail(relative: str, field: str, detail: str) -> "ContractError":
    return ContractError(f"{relative}: field {field} {detail}")


def _load_clip(
    path: Path, relative: str, layout: MotionLayout
) -> MotionClip:
    with np.load(path, allow_pickle=False) as data:
        present = set(data.files)
        required = (
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        for field in required:
            if field not in present:
                raise _fail(relative, field, "is missing")

        fps_raw = np.asarray(data["fps"])
        if fps_raw.size != 1:
            raise _fail(relative, "fps", "must contain exactly one value")
        fps_value = float(fps_raw.reshape(-1)[0])
        if not np.isfinite(fps_value) or fps_value != 50.0:
            raise _fail(relative, "fps", "must equal 50")

        joint_pos = _as_finite_float32(data["joint_pos"], relative, "joint_pos")
        joint_vel = _as_finite_float32(data["joint_vel"], relative, "joint_vel")
        body_pos = _as_finite_float32(data["body_pos_w"], relative, "body_pos_w")
        body_quat = _as_finite_float32(
            data["body_quat_w"], relative, "body_quat_w"
        )
        body_lin = _as_finite_float32(
            data["body_lin_vel_w"], relative, "body_lin_vel_w"
        )
        body_ang = _as_finite_float32(
            data["body_ang_vel_w"], relative, "body_ang_vel_w"
        )

    j = layout.joint_count
    b = layout.body_count
    _require_shape(joint_pos, 2, (j,), relative, "joint_pos")
    _require_shape(joint_vel, 2, (j,), relative, "joint_vel")
    _require_shape(body_pos, 3, (b, 3), relative, "body_pos_w")
    _require_shape(body_quat, 3, (b, 4), relative, "body_quat_w")
    _require_shape(body_lin, 3, (b, 3), relative, "body_lin_vel_w")
    _require_shape(body_ang, 3, (b, 3), relative, "body_ang_vel_w")

    frames = joint_pos.shape[0]
    for array, field in (
        (joint_vel, "joint_vel"),
        (body_pos, "body_pos_w"),
        (body_quat, "body_quat_w"),
        (body_lin, "body_lin_vel_w"),
        (body_ang, "body_ang_vel_w"),
    ):
        if array.shape[0] != frames:
            raise _fail(
                relative, field, f"frame count {array.shape[0]} != {frames}"
            )

    if frames < _MIN_FRAMES:
        raise _fail(
            relative,
            "joint_pos",
            f"clip has {frames} frames, needs at least {_MIN_FRAMES}",
        )

    quat = _canonicalize_quaternions(body_quat, relative)

    return MotionClip(
        relative_path=relative,
        fps=int(fps_value),
        joint_position=_own_readonly(joint_pos),
        joint_velocity=_own_readonly(joint_vel),
        body_position_world=_own_readonly(body_pos),
        body_quaternion_world_wxyz=_own_readonly(quat),
        body_linear_velocity_world=_own_readonly(body_lin),
        body_angular_velocity_world=_own_readonly(body_ang),
    )


def _as_finite_float32(array: np.ndarray, relative: str, field: str) -> np.ndarray:
    source = np.asarray(array)
    if not np.all(np.isfinite(source)):
        raise _fail(relative, field, "contains non-finite values")
    converted = np.ascontiguousarray(source, dtype=np.float32)
    if not np.all(np.isfinite(converted)):
        raise _fail(relative, field, "is not representable as finite float32")
    return converted


def _require_shape(
    array: np.ndarray,
    rank: int,
    trailing: tuple[int, ...],
    relative: str,
    field: str,
) -> None:
    if array.ndim != rank:
        raise _fail(relative, field, f"must have rank {rank}, got {array.ndim}")
    if tuple(array.shape[1:]) != trailing:
        raise _fail(
            relative,
            field,
            f"trailing shape {tuple(array.shape[1:])} != {trailing}",
        )


def _canonicalize_quaternions(quat: np.ndarray, relative: str) -> np.ndarray:
    norms = np.linalg.norm(quat, axis=-1)
    if np.any(np.abs(norms - 1.0) > _QUATERNION_NORM_TOLERANCE):
        raise _fail(relative, "body_quat_w", "has a quaternion norm error above 1e-4")

    normalized = (quat / norms[..., None]).astype(np.float32)

    # Flip a sample whenever its dot with the previous frame is negative, per body.
    frames = normalized.shape[0]
    for f in range(1, frames):
        dots = np.sum(normalized[f] * normalized[f - 1], axis=-1)
        flip = dots < 0.0
        if np.any(flip):
            normalized[f, flip, :] = -normalized[f, flip, :]
    return normalized


def _own_readonly(array: np.ndarray) -> np.ndarray:
    owned = np.ascontiguousarray(array, dtype=np.float32).copy()
    owned.setflags(write=False)
    return owned
