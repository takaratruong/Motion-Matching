"""Convert pinned GRAIL qpos into native 50 Hz Torch motion arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mm_sonic.joints import PINNED_TARGET_TO_SOURCE_PERMUTATION
from resources.g1_terrain_builder.database import derive_velocities
from resources.g1_terrain_builder.resample import (
    resample_quaternions_wxyz,
    resample_vectors,
)

from .corpus import PinnedStairSource


TARGET_FPS = 50
ISAAC_BODY_HOLDEN_NAMES = (
    "Hips",
    "LeftHipPitch",
    "RightHipPitch",
    "Spine",
    "LeftHipRoll",
    "RightHipRoll",
    "Spine1",
    "LeftHipYaw",
    "RightHipYaw",
    "Spine2",
    "LeftKnee",
    "RightKnee",
    "LeftShoulderPitch",
    "RightShoulderPitch",
    "LeftAnkle",
    "RightAnkle",
    "LeftShoulderRoll",
    "RightShoulderRoll",
    "LeftToe",
    "RightToe",
    "LeftShoulderYaw",
    "RightShoulderYaw",
    "LeftElbow",
    "RightElbow",
    "LeftWristRoll",
    "RightWristRoll",
    "LeftWristPitch",
    "RightWristPitch",
    "LeftWrist",
    "RightWrist",
)


def _float32(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float32)


def _normalize_unroll(quaternions: np.ndarray) -> np.ndarray:
    output = np.asarray(quaternions, np.float64).copy()
    norms = np.linalg.norm(output, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms < 1e-12):
        raise ValueError("body quaternion is non-finite or zero length")
    output /= norms
    for frame in range(1, len(output)):
        flip = np.sum(output[frame - 1] * output[frame], axis=-1) < 0.0
        output[frame, flip] *= -1.0
    return output


@dataclass(frozen=True)
class NativeMotionArrays:
    fps: int
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_position_world: np.ndarray
    body_quaternion_world_wxyz: np.ndarray
    body_linear_velocity_world: np.ndarray
    body_angular_velocity_world: np.ndarray

    def as_npz_fields(self) -> dict[str, np.ndarray]:
        return {
            "fps": np.array([self.fps], np.int64),
            "joint_pos": self.joint_position,
            "joint_vel": self.joint_velocity,
            "body_pos_w": self.body_position_world,
            "body_quat_w": self.body_quaternion_world_wxyz,
            "body_lin_vel_w": self.body_linear_velocity_world,
            "body_ang_vel_w": self.body_angular_velocity_world,
        }

    def validate(self) -> None:
        frames = len(self.joint_position)
        expected = {
            "joint_position": (frames, 29),
            "joint_velocity": (frames, 29),
            "body_position_world": (frames, 30, 3),
            "body_quaternion_world_wxyz": (frames, 30, 4),
            "body_linear_velocity_world": (frames, 30, 3),
            "body_angular_velocity_world": (frames, 30, 3),
        }
        if self.fps != TARGET_FPS or frames < 46:
            raise ValueError("native motion must contain at least 46 frames at 50 Hz")
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != np.float32:
                raise ValueError(f"{name} shape/dtype does not match native contract")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        norms = np.linalg.norm(self.body_quaternion_world_wxyz, axis=-1)
        if np.max(np.abs(norms - 1.0)) > 1e-4:
            raise ValueError("native body quaternion norm error exceeds 1e-4")


def _body_permutation(names: tuple[str, ...] | list[str]) -> np.ndarray:
    if len(names) != len(set(names)):
        raise ValueError("kinematics body names contain duplicates")
    index = {name: i for i, name in enumerate(names)}
    missing = [name for name in ISAAC_BODY_HOLDEN_NAMES if name not in index]
    if missing:
        raise ValueError(f"kinematics is missing bodies: {missing}")
    return np.array([index[name] for name in ISAAC_BODY_HOLDEN_NAMES], np.int64)


def convert_source_to_native_50hz(
    source: PinnedStairSource,
    kinematics,
) -> NativeMotionArrays:
    if not isinstance(source, PinnedStairSource):
        raise TypeError("source must be a PinnedStairSource")
    qpos = np.asarray(source.robot_qpos_mujoco, np.float64)
    root_position = resample_vectors(
        qpos[:, :3], source.source_fps, TARGET_FPS
    )
    root_quaternion = resample_quaternions_wxyz(
        qpos[:, 3:7], source.source_fps, TARGET_FPS
    )
    joints_mujoco = resample_vectors(
        qpos[:, 7:], source.source_fps, TARGET_FPS
    )
    output_qpos = np.concatenate(
        (root_position, root_quaternion, joints_mujoco), axis=1
    )

    body_position_mujoco, body_quaternion_mujoco = (
        kinematics.world_from_qpos(output_qpos)
    )
    body_position_mujoco = np.asarray(body_position_mujoco, np.float64)
    body_quaternion_mujoco = np.asarray(body_quaternion_mujoco, np.float64)
    expected = (len(output_qpos), len(kinematics.names))
    if body_position_mujoco.shape != expected + (3,):
        raise ValueError("kinematics returned an invalid body-position shape")
    if body_quaternion_mujoco.shape != expected + (4,):
        raise ValueError("kinematics returned an invalid body-quaternion shape")

    permutation = _body_permutation(tuple(kinematics.names))
    body_position = body_position_mujoco[:, permutation]
    body_quaternion = _normalize_unroll(
        body_quaternion_mujoco[:, permutation]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        body_linear_velocity, body_angular_velocity = derive_velocities(
            body_position, body_quaternion, TARGET_FPS
        )

    joint_position = joints_mujoco[
        :, np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION, np.int64)
    ]
    joint_velocity = (
        np.gradient(joint_position, axis=0, edge_order=2) * TARGET_FPS
    )
    output = NativeMotionArrays(
        fps=TARGET_FPS,
        joint_position=_float32(joint_position),
        joint_velocity=_float32(joint_velocity),
        body_position_world=_float32(body_position),
        body_quaternion_world_wxyz=_float32(body_quaternion),
        body_linear_velocity_world=_float32(body_linear_velocity),
        body_angular_velocity_world=_float32(body_angular_velocity),
    )
    output.validate()
    return output
