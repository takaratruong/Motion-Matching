"""Deterministic multi-support landing bridges for terrain coverage holes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .joints import ContractError
from .torch_contact_segments import ANKLE_ORIGIN_SOLE_M


_DT_S = 0.02
_SWING_CLEARANCE_M = 0.10
_LOWER_SURFACE_MIN_DROP_M = 0.05
_FOOTHOLD_MIN_DISTANCE_M = 0.10
_FOOTHOLD_MAX_DISTANCE_M = 0.80
_FOOTHOLD_STEP_M = 0.02
_MAX_JOINT_STEP_RAD = 0.35


def _readonly_f64(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ContractError("terrain landing bridge arrays are invalid")
    owned = np.ascontiguousarray(array).copy()
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True)
class LandingBridgeReference:
    clip_index: int
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if (
            type(self.clip_index) is not int
            or self.clip_index < 0
            or type(self.start_frame) is not int
            or self.start_frame < 0
            or type(self.end_frame) is not int
            or self.end_frame <= self.start_frame
        ):
            raise ContractError("terrain landing bridge reference is invalid")

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    def source_support_mask(self, index: object) -> torch.Tensor:
        try:
            support = index.support_mask(self.clip_index)
        except Exception as error:
            raise ContractError(
                "terrain landing bridge reference support is invalid"
            ) from error
        if (
            not isinstance(support, torch.Tensor)
            or support.dtype != torch.bool
            or support.ndim != 2
            or support.shape[1] != 2
            or support.shape[0] < self.end_frame
        ):
            raise ContractError(
                "terrain landing bridge reference support is invalid"
            )
        return support[self.start_frame : self.end_frame].detach().clone()


@dataclass(frozen=True)
class TerrainLandingBridge:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    root_position_world: np.ndarray
    root_orientation_world_wxyz: np.ndarray
    foot_position_world: np.ndarray

    def __post_init__(self) -> None:
        try:
            frames = int(np.asarray(self.joint_position).shape[0])
        except Exception as error:
            raise ContractError("terrain landing bridge arrays are invalid") from error
        if frames < 2:
            raise ContractError("terrain landing bridge arrays are invalid")
        joints = _readonly_f64(self.joint_position, (frames, 29))
        velocities = _readonly_f64(self.joint_velocity, (frames, 29))
        roots = _readonly_f64(self.root_position_world, (frames, 3))
        quaternions = _readonly_f64(
            self.root_orientation_world_wxyz, (frames, 4)
        )
        feet = _readonly_f64(self.foot_position_world, (frames, 2, 3))
        norms = np.linalg.norm(quaternions, axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-4):
            raise ContractError("terrain landing bridge quaternions are invalid")
        object.__setattr__(self, "joint_position", joints)
        object.__setattr__(self, "joint_velocity", velocities)
        object.__setattr__(self, "root_position_world", roots)
        object.__setattr__(self, "root_orientation_world_wxyz", quaternions)
        object.__setattr__(self, "foot_position_world", feet)

    @property
    def frame_count(self) -> int:
        return int(self.joint_position.shape[0])


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array(
        [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
        np.float64,
    )


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        np.float64,
    )


def _sample_surface(extension: object, points_xy: np.ndarray) -> np.ndarray:
    try:
        device = extension.dataset.device
    except AttributeError:
        device = torch.device("cpu")
    points = torch.tensor(points_xy, dtype=torch.float32, device=device)
    try:
        scene = extension.alignment.matcher_to_scene_xy(points)
        surface = extension.query_grid.sample_xy(scene)
    except Exception as error:
        raise ContractError("terrain landing bridge terrain query failed") from error
    values = np.asarray(surface.detach().cpu().numpy(), np.float64)
    if values.shape != points_xy.shape[:-1] or not np.isfinite(values).all():
        raise ContractError("terrain landing bridge terrain query failed")
    return values


def _lower_foothold(
    extension: object,
    foot_position: np.ndarray,
    direction_xy: np.ndarray,
) -> np.ndarray:
    start_surface = float(_sample_surface(extension, foot_position[None, :2])[0])
    count = int(
        round(
            (_FOOTHOLD_MAX_DISTANCE_M - _FOOTHOLD_MIN_DISTANCE_M)
            / _FOOTHOLD_STEP_M
        )
    )
    distances = np.linspace(
        _FOOTHOLD_MIN_DISTANCE_M,
        _FOOTHOLD_MAX_DISTANCE_M,
        count + 1,
    )
    points = foot_position[None, :2] + distances[:, None] * direction_xy
    surfaces = _sample_surface(extension, points)
    lower = np.flatnonzero(
        surfaces <= start_surface - _LOWER_SURFACE_MIN_DROP_M
    )
    if lower.size == 0:
        raise ContractError("terrain landing bridge has no lower foothold")
    selected = int(lower[0])
    return np.array(
        [points[selected, 0], points[selected, 1],
         surfaces[selected] + ANKLE_ORIGIN_SOLE_M],
        np.float64,
    )


def _foot_target_path(
    support: np.ndarray,
    start: np.ndarray,
    landing: np.ndarray,
) -> np.ndarray:
    frames = support.shape[0]
    release = next(
        (
            frame
            for frame in range(1, frames)
            if support[frame - 1] and not support[frame]
        ),
        None,
    )
    if release is None:
        raise ContractError("terrain landing bridge source has no foot release")
    onset = next(
        (frame for frame in range(release + 1, frames) if support[frame]),
        None,
    )
    if onset is None:
        raise ContractError("terrain landing bridge source has no foot landing")
    output = np.repeat(start[None, :], frames, axis=0)
    span = onset - release + 1
    for frame in range(release, onset + 1):
        phase = (frame - release + 1) / span
        smooth = phase * phase * (3.0 - 2.0 * phase)
        output[frame] = (1.0 - smooth) * start + smooth * landing
        output[frame, 2] += (
            4.0 * _SWING_CLEARANCE_M * phase * (1.0 - phase)
        )
    output[onset + 1 :] = landing
    return output


def generate_terrain_landing_bridge(
    *,
    dataset: object,
    extension: object,
    contact_index: object,
    kinematics: object,
    reference: LandingBridgeReference,
    start_joint_position: object,
    start_root_position_world: object,
    start_root_orientation_world_wxyz: object,
    command_direction_world_xy: object,
) -> TerrainLandingBridge:
    """Retarget one authenticated multi-support landing into query terrain."""

    if not isinstance(reference, LandingBridgeReference):
        raise ContractError("terrain landing bridge reference is invalid")
    try:
        clip = dataset.folder.clips[reference.clip_index]
        root_index = int(dataset.folder.layout.root_body_index)
    except Exception as error:
        raise ContractError("terrain landing bridge dataset is invalid") from error
    if reference.end_frame > clip.joint_position.shape[0]:
        raise ContractError("terrain landing bridge reference is invalid")
    joints0 = _readonly_f64(start_joint_position, (29,)).copy()
    root0 = _readonly_f64(start_root_position_world, (3,)).copy()
    quaternion0 = _readonly_f64(
        start_root_orientation_world_wxyz, (4,)
    ).copy()
    if abs(float(np.linalg.norm(quaternion0)) - 1.0) > 1e-4:
        raise ContractError("terrain landing bridge root orientation is invalid")
    direction = _readonly_f64(command_direction_world_xy, (2,)).copy()
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        raise ContractError("terrain landing bridge command is invalid")
    direction /= norm
    support = reference.source_support_mask(contact_index).cpu().numpy()
    frames = reference.frame_count
    source_joints = np.asarray(
        clip.joint_position[reference.start_frame : reference.end_frame],
        np.float64,
    )
    source_root = np.asarray(
        clip.body_position_world[
            reference.start_frame : reference.end_frame, root_index
        ],
        np.float64,
    )
    source_quaternion = np.asarray(
        clip.body_quaternion_world_wxyz[
            reference.start_frame : reference.end_frame, root_index
        ],
        np.float64,
    )
    if (
        source_joints.shape != (frames, 29)
        or source_root.shape != (frames, 3)
        or source_quaternion.shape != (frames, 4)
    ):
        raise ContractError("terrain landing bridge source arrays are invalid")

    yaw_offset = _yaw_from_wxyz(quaternion0) - _yaw_from_wxyz(
        source_quaternion[0]
    )
    cosine = math.cos(yaw_offset)
    sine = math.sin(yaw_offset)
    rotation = np.array(((cosine, -sine), (sine, cosine)), np.float64)
    roots = np.empty((frames, 3), np.float64)
    roots[:, :2] = root0[:2] + (
        source_root[:, :2] - source_root[0, :2]
    ) @ rotation.T
    roots[:, 2] = root0[2] + source_root[:, 2] - source_root[0, 2]
    yaw_rotation = _yaw_quaternion(yaw_offset)
    quaternions = np.stack(
        [_quat_multiply(yaw_rotation, value) for value in source_quaternion]
    )
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)

    try:
        start_feet = np.asarray(
            kinematics.foot_positions(
                joints0[None, :], root0[None, :], quaternion0[None, :]
            )[0],
            np.float64,
        )
    except Exception as error:
        raise ContractError("terrain landing bridge FK failed") from error
    if start_feet.shape != (2, 3) or not np.isfinite(start_feet).all():
        raise ContractError("terrain landing bridge FK failed")
    start_surfaces = _sample_surface(extension, start_feet[:, :2])
    start_targets = start_feet.copy()
    start_targets[:, 2] = start_surfaces + ANKLE_ORIGIN_SOLE_M
    landing_targets = np.stack(
        [
            _lower_foothold(extension, start_targets[foot], direction)
            for foot in range(2)
        ]
    )
    target_paths = np.stack(
        [
            _foot_target_path(
                support[:, foot], start_targets[foot], landing_targets[foot]
            )
            for foot in range(2)
        ],
        axis=1,
    )

    solved = np.empty((frames, 29), np.float64)
    initial_offset = joints0 - source_joints[0]
    previous = joints0.copy()
    for frame in range(frames):
        decay = 0.5 ** (frame * _DT_S / 0.10)
        seed = source_joints[frame] + initial_offset * decay
        if frame > 0:
            leg_indices = (0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18)
            seed[list(leg_indices)] = previous[list(leg_indices)]
        try:
            pose = np.asarray(
                kinematics.solve_leg_positions(
                    seed,
                    roots[frame],
                    quaternions[frame],
                    np.array([True, True]),
                    target_paths[frame],
                ),
                np.float64,
            )
        except Exception as error:
            raise ContractError("terrain landing bridge IK failed") from error
        if pose.shape != (29,) or not np.isfinite(pose).all():
            raise ContractError("terrain landing bridge IK failed")
        if frame > 0 and float(np.max(np.abs(pose - previous))) > _MAX_JOINT_STEP_RAD:
            raise ContractError("terrain landing bridge joint discontinuity")
        solved[frame] = pose
        previous = pose

    try:
        feet = np.asarray(
            kinematics.foot_positions(solved, roots, quaternions), np.float64
        )
    except Exception as error:
        raise ContractError("terrain landing bridge FK failed") from error
    if feet.shape != (frames, 2, 3) or not np.isfinite(feet).all():
        raise ContractError("terrain landing bridge FK failed")
    velocity = np.empty_like(solved)
    velocity[1:] = (solved[1:] - solved[:-1]) / _DT_S
    velocity[0] = velocity[1]
    return TerrainLandingBridge(
        joint_position=solved,
        joint_velocity=velocity,
        root_position_world=roots,
        root_orientation_world_wxyz=quaternions,
        foot_position_world=feet,
    )
