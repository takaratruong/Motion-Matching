"""Differentiable forward kinematics for the canonical native G1 model."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from mm_sonic.terrain_oracle.canonical import (
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
)


KINEMATIC_SIGNATURE_SCHEMA = "mm-sonic-g1-kinematic-signature/v1"


def root_tilt_quaternion_wxyz(
    tilt_x: float, tilt_y: float
) -> np.ndarray:
    """Decode the feature contract's yaw-free x/y angle-axis tilt."""

    x, y = float(tilt_x), float(tilt_y)
    angle = math.hypot(x, y)
    if angle < 1.0e-12:
        return np.asarray((1.0, 0.5 * x, 0.5 * y, 0.0), np.float64)
    scale = math.sin(0.5 * angle) / angle
    return np.asarray((math.cos(0.5 * angle), scale * x, scale * y, 0.0), np.float64)


def _root_tilt_matrix(tilt: torch.Tensor) -> torch.Tensor:
    angle_squared = tilt.square().sum(dim=-1)
    threshold = 1.0e-8
    safe_angle = torch.sqrt(angle_squared.clamp_min(threshold))
    exact_scale = torch.sin(0.5 * safe_angle) / safe_angle
    series_scale = 0.5 - angle_squared / 48.0 + angle_squared.square() / 3840.0
    scale = torch.where(angle_squared >= threshold, exact_scale, series_scale)
    exact_w = torch.cos(0.5 * safe_angle)
    series_w = 1.0 - angle_squared / 8.0 + angle_squared.square() / 384.0
    w = torch.where(angle_squared >= threshold, exact_w, series_w)
    zero = torch.zeros_like(w)
    quaternion = torch.stack(
        (w, scale * tilt[:, 0], scale * tilt[:, 1], zero), dim=-1
    )
    return _quaternion_matrix_wxyz(quaternion)


def _quaternion_matrix_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(quaternion.dtype).eps)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
            2 * (x * z + y * w), 2 * (x * y + z * w),
            1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(
        torch.finfo(axis.dtype).eps
    )
    x, y, z = axis.unbind()
    zero = torch.zeros((), dtype=axis.dtype, device=axis.device)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    outer = axis[:, None] * axis[None, :]
    cosine = torch.cos(angle)[..., None, None]
    sine = torch.sin(angle)[..., None, None]
    return cosine * identity + (1.0 - cosine) * outer + sine * skew


def _hex_rows(value: np.ndarray) -> list[Any]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        return [float(item).hex() for item in array]
    return [_hex_rows(row) for row in array]


class TorchG1ForwardKinematics(nn.Module):
    """Compiled MuJoCo constants with a differentiable Torch forward pass.

    Each non-root canonical body is required to own exactly one hinge.  Its
    transform is ``T_body0 @ T_joint @ R(axis, q) @ T_-joint``.  The pelvis
    free joint is replaced by the supplied local height and x/y angle-axis tilt.
    """

    def __init__(
        self,
        *,
        parent_indices: np.ndarray,
        body_position: np.ndarray,
        body_quaternion_wxyz: np.ndarray,
        body_joint_indices: np.ndarray,
        joint_to_body_indices: np.ndarray,
        joint_types: np.ndarray,
        joint_position: np.ndarray,
        joint_axis: np.ndarray,
        joint_limits: np.ndarray,
    ) -> None:
        super().__init__()
        self.body_names = ISAACLAB_BODY_NAMES
        self.joint_names = ISAACLAB_JOINT_NAMES
        self.register_buffer("parent_indices", torch.as_tensor(parent_indices, dtype=torch.int64))
        self.register_buffer("body_position", torch.as_tensor(body_position, dtype=torch.float64))
        self.register_buffer("body_quaternion_wxyz", torch.as_tensor(body_quaternion_wxyz, dtype=torch.float64))
        self.register_buffer("body_joint_indices", torch.as_tensor(body_joint_indices, dtype=torch.int64))
        self.register_buffer("joint_to_body_indices", torch.as_tensor(joint_to_body_indices, dtype=torch.int64))
        self.register_buffer("joint_types", torch.as_tensor(joint_types, dtype=torch.int64))
        self.register_buffer("joint_position", torch.as_tensor(joint_position, dtype=torch.float64))
        self.register_buffer("joint_axis", torch.as_tensor(joint_axis, dtype=torch.float64))
        self.register_buffer("joint_limits", torch.as_tensor(joint_limits, dtype=torch.float64))
        payload = {
            "schema": KINEMATIC_SIGNATURE_SCHEMA,
            "body_names": list(self.body_names),
            "joint_names": list(self.joint_names),
            "parent_indices": np.asarray(parent_indices, np.int64).tolist(),
            "joint_to_body_indices": np.asarray(
                joint_to_body_indices, np.int64
            ).tolist(),
            "joint_types": np.asarray(joint_types, np.int64).tolist(),
            "body_pos": _hex_rows(body_position),
            "body_quat": _hex_rows(body_quaternion_wxyz),
            "jnt_pos": _hex_rows(joint_position),
            "jnt_axis": _hex_rows(joint_axis),
            "jnt_range": _hex_rows(joint_limits),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        self.kinematic_signature_payload = payload
        self.kinematic_signature_sha256 = hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_mjcf(cls, path: str | Path) -> "TorchG1ForwardKinematics":
        import mujoco

        return cls.from_mjmodel(mujoco.MjModel.from_xml_path(str(Path(path))))

    @classmethod
    def from_mjmodel(cls, model: object) -> "TorchG1ForwardKinematics":
        import mujoco

        def names(kind: Any, count: int) -> tuple[str | None, ...]:
            return tuple(mujoco.mj_id2name(model, kind, index) for index in range(count))

        body_names = names(mujoco.mjtObj.mjOBJ_BODY, int(model.nbody))
        joint_names = names(mujoco.mjtObj.mjOBJ_JOINT, int(model.njnt))
        if any(body_names.count(name) != 1 for name in ISAACLAB_BODY_NAMES):
            raise ValueError("MuJoCo model must contain each canonical G1 body exactly once")
        if any(joint_names.count(name) != 1 for name in ISAACLAB_JOINT_NAMES):
            raise ValueError("MuJoCo model must contain each canonical G1 joint exactly once")
        body_ids = np.asarray([body_names.index(name) for name in ISAACLAB_BODY_NAMES], np.int64)
        joint_ids = np.asarray([joint_names.index(name) for name in ISAACLAB_JOINT_NAMES], np.int64)
        body_lookup = {int(body_id): index for index, body_id in enumerate(body_ids)}
        parents = np.empty(30, np.int64)
        for index, body_id in enumerate(body_ids):
            parent_id = int(model.body_parentid[body_id])
            if index == 0:
                if parent_id != 0:
                    raise ValueError("canonical pelvis must be parented to world")
                parents[index] = -1
            elif parent_id not in body_lookup:
                raise ValueError("canonical G1 parent tree leaves the canonical robot")
            else:
                parents[index] = body_lookup[parent_id]
                if parents[index] >= index:
                    raise ValueError("canonical G1 body order must be topological")

        root_body_id = int(body_ids[0])
        root_joint_start = int(model.body_jntadr[root_body_id])
        if (
            int(model.body_jntnum[root_body_id]) != 1
            or int(model.jnt_type[root_joint_start]) != int(mujoco.mjtJoint.mjJNT_FREE)
        ):
            raise ValueError("canonical pelvis must own exactly one free joint")

        joint_to_body = np.asarray(
            [body_lookup.get(int(model.jnt_bodyid[joint_id]), -1) for joint_id in joint_ids],
            np.int64,
        )
        if sorted(joint_to_body.tolist()) != list(range(1, 30)):
            raise ValueError("canonical non-root bodies must own one canonical joint each")
        joint_types = np.asarray(model.jnt_type[joint_ids], np.int64)
        if np.any(joint_types != int(mujoco.mjtJoint.mjJNT_HINGE)):
            raise ValueError("canonical G1 non-root joints must all be hinges")
        body_joint_indices = np.full(30, -1, np.int64)
        for joint_index, body_index in enumerate(joint_to_body):
            body_id = int(body_ids[body_index])
            if (
                int(model.body_jntnum[body_id]) != 1
                or int(model.body_jntadr[body_id]) != int(joint_ids[joint_index])
            ):
                raise ValueError("canonical hinge topology is not one joint per body")
            body_joint_indices[body_index] = joint_index
        limits = np.asarray(model.jnt_range[joint_ids], np.float64).copy()
        if not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("canonical G1 hinge limits must be finite and ordered")
        return cls(
            parent_indices=parents,
            body_position=np.asarray(model.body_pos[body_ids], np.float64).copy(),
            body_quaternion_wxyz=np.asarray(model.body_quat[body_ids], np.float64).copy(),
            body_joint_indices=body_joint_indices,
            joint_to_body_indices=joint_to_body,
            joint_types=joint_types,
            joint_position=np.asarray(model.jnt_pos[joint_ids], np.float64).copy(),
            joint_axis=np.asarray(model.jnt_axis[joint_ids], np.float64).copy(),
            joint_limits=limits,
        )

    def forward(
        self, root_height_roll_pitch: torch.Tensor, joint_position: torch.Tensor
    ) -> torch.Tensor:
        if (
            root_height_roll_pitch.ndim != 2
            or root_height_roll_pitch.shape[1] != 3
            or joint_position.shape != (len(root_height_roll_pitch), 29)
        ):
            raise ValueError("G1 FK expects root[B,3] and joints[B,29]")
        if root_height_roll_pitch.device != joint_position.device:
            raise ValueError("G1 FK inputs must share a device")
        dtype = joint_position.dtype
        device = joint_position.device
        root = root_height_roll_pitch.to(dtype=dtype)
        body_pos = self.body_position.to(dtype=dtype, device=device)
        body_quat = self.body_quaternion_wxyz.to(dtype=dtype, device=device)
        joint_origin = self.joint_position.to(dtype=dtype, device=device)
        joint_axis = self.joint_axis.to(dtype=dtype, device=device)
        batch = len(root)

        zeros = torch.zeros_like(root[:, 0])
        rotations: list[torch.Tensor] = [_root_tilt_matrix(root[:, 1:3])]
        positions: list[torch.Tensor] = [
            torch.stack((zeros, zeros, root[:, 0]), dim=-1)
        ]
        fixed_rotation = _quaternion_matrix_wxyz(body_quat)
        for body_index in range(1, 30):
            parent_index = int(self.parent_indices[body_index])
            joint_index = int(self.body_joint_indices[body_index])
            parent_rotation = rotations[parent_index]
            initial_rotation = torch.matmul(
                parent_rotation, fixed_rotation[body_index]
            )
            initial_position = positions[parent_index] + torch.matmul(
                parent_rotation, body_pos[body_index].expand(batch, 3).unsqueeze(-1)
            ).squeeze(-1)
            delta_rotation = _axis_angle_matrix(
                joint_axis[joint_index], joint_position[:, joint_index]
            )
            origin = joint_origin[joint_index].expand(batch, 3)
            origin_delta = origin - torch.matmul(
                delta_rotation, origin.unsqueeze(-1)
            ).squeeze(-1)
            positions.append(
                initial_position
                + torch.matmul(initial_rotation, origin_delta.unsqueeze(-1)).squeeze(-1)
            )
            rotations.append(torch.matmul(initial_rotation, delta_rotation))
        return torch.stack(positions, dim=1)


__all__ = [
    "KINEMATIC_SIGNATURE_SCHEMA",
    "TorchG1ForwardKinematics",
    "root_tilt_quaternion_wxyz",
]
