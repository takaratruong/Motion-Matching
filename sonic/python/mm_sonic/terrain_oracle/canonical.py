"""Immutable, validated representation of canonical terrain-motion clips."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re

import numpy as np

from mm_sonic.joints import ContractError

from .math3d import RigidTransform, angular_velocity_world_wxyz, finite_difference


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_COORDINATE_CONVENTION = "z-up-right-handed"
CLEAN_POSE_ORIGIN = "clean-motion-corpus"

# Kept here rather than imported from a source adapter so the canonical boundary
# remains usable without optional simulator/runtime dependencies.
ISAACLAB_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

ISAACLAB_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


def _readonly_float32(value: object, label: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "iuf":
        raise ContractError(f"{label} must be a real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.asarray(source, dtype=np.float64).astype(
            np.float32, order="C", casting="unsafe", copy=True
        )
    if not np.isfinite(output).all():
        raise ContractError(f"{label} must contain finite binary32 values")
    output = np.ascontiguousarray(output).copy(order="C")
    output.flags.writeable = False
    return output


def _readonly_bool(value: object, label: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "biu":
        raise ContractError(f"{label} must be a boolean array")
    output = np.ascontiguousarray(np.asarray(source, dtype=np.bool_)).copy(order="C")
    output.flags.writeable = False
    return output


def _readonly_int32(value: object, label: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "iu":
        raise ContractError(f"{label} must be an integer array")
    output = np.ascontiguousarray(np.asarray(source, dtype=np.int32)).copy(order="C")
    output.flags.writeable = False
    return output


def _hash(value: str, label: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ContractError(f"{label} must be a lowercase SHA-256 hex digest")


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ContractError(f"{label} must be a nonempty string")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ContractError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class SourceIdentity:
    """Provenance that makes coordinate and quaternion conventions explicit."""

    source_format: str
    source_path: str
    source_size_bytes: int
    source_sha256: str
    source_license_id: str
    coordinate_convention: str
    quaternion_convention: str
    pose_origin: str

    def __post_init__(self) -> None:
        for name in (
            "source_format",
            "source_path",
            "source_license_id",
            "coordinate_convention",
            "quaternion_convention",
            "pose_origin",
        ):
            _nonempty(getattr(self, name), name)
        _nonnegative_int(self.source_size_bytes, "source_size_bytes")
        _hash(self.source_sha256, "source_sha256")


@dataclass(frozen=True)
class CommandTrack:
    observed_travel_stick_xy: np.ndarray
    observed_facing_stick_xy: np.ndarray
    observed_mask: np.ndarray
    inferred_velocity_local_xy: np.ndarray
    inferred_facing_local_xy: np.ndarray
    inferred_yaw_rate_rad_s: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "observed_travel_stick_xy",
            "observed_facing_stick_xy",
            "inferred_velocity_local_xy",
            "inferred_facing_local_xy",
            "inferred_yaw_rate_rad_s",
        ):
            object.__setattr__(self, name, _readonly_float32(getattr(self, name), name))
        object.__setattr__(
            self, "observed_mask", _readonly_bool(self.observed_mask, "observed_mask")
        )

    def validate(self, frame_count: int) -> None:
        expected = {
            "observed_travel_stick_xy": (frame_count, 2),
            "observed_facing_stick_xy": (frame_count, 2),
            "observed_mask": (frame_count,),
            "inferred_velocity_local_xy": (frame_count, 2),
            "inferred_facing_local_xy": (frame_count, 2),
            "inferred_yaw_rate_rad_s": (frame_count,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ContractError(f"commands.{name} must have shape {shape}")


@dataclass(frozen=True)
class TerrainBinding:
    asset_path: str
    asset_size_bytes: int
    asset_sha256: str
    asset_license_id: str
    mesh_sha256: str
    world_from_terrain: RigidTransform
    validity_mask_path: str | None

    def __post_init__(self) -> None:
        _nonempty(self.asset_path, "asset_path")
        _positive_int(self.asset_size_bytes, "asset_size_bytes")
        _hash(self.asset_sha256, "asset_sha256")
        _nonempty(self.asset_license_id, "asset_license_id")
        _hash(self.mesh_sha256, "mesh_sha256")
        if not isinstance(self.world_from_terrain, RigidTransform):
            raise ContractError("world_from_terrain must be a RigidTransform")
        if self.validity_mask_path is not None:
            _nonempty(self.validity_mask_path, "validity_mask_path")


@dataclass(frozen=True)
class CanonicalTerrainMesh:
    vertices_local: np.ndarray
    faces: np.ndarray
    valid_faces: np.ndarray
    source_asset_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices_local", _readonly_float32(self.vertices_local, "vertices_local"))
        object.__setattr__(self, "faces", _readonly_int32(self.faces, "faces"))
        object.__setattr__(self, "valid_faces", _readonly_bool(self.valid_faces, "valid_faces"))
        _hash(self.source_asset_sha256, "source_asset_sha256")
        if self.vertices_local.ndim != 2 or self.vertices_local.shape[1:] != (3,):
            raise ContractError("vertices_local must have shape [N,3]")
        if self.faces.ndim != 2 or self.faces.shape[1:] != (3,):
            raise ContractError("faces must have shape [M,3]")
        if self.valid_faces.shape != (self.faces.shape[0],):
            raise ContractError("valid_faces must have shape [M]")
        if np.any(self.faces < 0) or np.any(self.faces >= len(self.vertices_local)):
            raise ContractError("faces reference a vertex outside vertices_local")


@dataclass(frozen=True)
class CanonicalClip:
    clip_id: str
    fps: float
    source: SourceIdentity
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    root_position_world: np.ndarray
    root_quaternion_world_wxyz: np.ndarray
    joint_position: np.ndarray
    root_linear_velocity_world: np.ndarray
    root_angular_velocity_world: np.ndarray
    joint_velocity: np.ndarray
    body_position_world: np.ndarray
    body_quaternion_world_wxyz: np.ndarray
    body_linear_velocity_world: np.ndarray
    body_angular_velocity_world: np.ndarray
    sole_position_world: np.ndarray
    sole_quaternion_world_wxyz: np.ndarray
    heel_position_world: np.ndarray
    toe_position_world: np.ndarray
    contact: np.ndarray
    contact_confidence: np.ndarray
    commands: CommandTrack
    terrain: TerrainBinding | None
    action_tags: tuple[str, ...]
    mirror_of: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.clip_id, "clip_id")
        if not isinstance(self.source, SourceIdentity):
            raise ContractError("source must be a SourceIdentity")
        if type(self.joint_names) not in (tuple, list):
            raise ContractError("joint_names must be a tuple of strings")
        if type(self.body_names) not in (tuple, list):
            raise ContractError("body_names must be a tuple of strings")
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(self, "body_names", tuple(self.body_names))
        for name in (
            "root_position_world",
            "root_quaternion_world_wxyz",
            "joint_position",
            "root_linear_velocity_world",
            "root_angular_velocity_world",
            "joint_velocity",
            "body_position_world",
            "body_quaternion_world_wxyz",
            "body_linear_velocity_world",
            "body_angular_velocity_world",
            "sole_position_world",
            "sole_quaternion_world_wxyz",
            "heel_position_world",
            "toe_position_world",
            "contact",
            "contact_confidence",
        ):
            object.__setattr__(self, name, _readonly_float32(getattr(self, name), name))
        if not isinstance(self.commands, CommandTrack):
            raise ContractError("commands must be a CommandTrack")
        if self.terrain is not None and not isinstance(self.terrain, TerrainBinding):
            raise ContractError("terrain must be a TerrainBinding or None")
        if type(self.action_tags) not in (tuple, list) or not self.action_tags:
            raise ContractError("action_tags must be a nonempty tuple of strings")
        tags = tuple(_nonempty(value, "action tag") for value in self.action_tags)
        object.__setattr__(self, "action_tags", tags)
        if self.mirror_of is not None:
            _nonempty(self.mirror_of, "mirror_of")

    @property
    def frame_count(self) -> int:
        return int(self.joint_position.shape[0])

    def validate(self) -> None:
        if type(self.fps) not in (int, float) or self.fps != 50.0 or self.frame_count < 2:
            raise ContractError("canonical clip must contain at least two 50 Hz frames")
        if self.source.quaternion_convention != "wxyz":
            raise ContractError("root quaternion provenance must explicitly be wxyz")
        if self.source.coordinate_convention != CANONICAL_COORDINATE_CONVENTION:
            raise ContractError(
                "coordinate convention must explicitly be z-up-right-handed"
            )
        if self.source.pose_origin != CLEAN_POSE_ORIGIN:
            raise ContractError("canonical clip must have a clean pose origin")
        if self.joint_names != ISAACLAB_JOINT_NAMES:
            raise ContractError("joint_names must equal the IsaacLab canonical order")
        if self.body_names != ISAACLAB_BODY_NAMES:
            raise ContractError("body_names must equal the IsaacLab canonical order")
        frame_count = self.frame_count
        expected = {
            "root_position_world": (frame_count, 3),
            "root_quaternion_world_wxyz": (frame_count, 4),
            "joint_position": (frame_count, 29),
            "root_linear_velocity_world": (frame_count, 3),
            "root_angular_velocity_world": (frame_count, 3),
            "joint_velocity": (frame_count, 29),
            "body_position_world": (frame_count, 30, 3),
            "body_quaternion_world_wxyz": (frame_count, 30, 4),
            "body_linear_velocity_world": (frame_count, 30, 3),
            "body_angular_velocity_world": (frame_count, 30, 3),
            "sole_position_world": (frame_count, 2, 3),
            "sole_quaternion_world_wxyz": (frame_count, 2, 4),
            "heel_position_world": (frame_count, 2, 3),
            "toe_position_world": (frame_count, 2, 3),
            "contact": (frame_count, 2),
            "contact_confidence": (frame_count, 2),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not np.isfinite(value).all():
                raise ContractError(f"{name} must be finite with shape {shape}")
        for name in (
            "root_quaternion_world_wxyz",
            "body_quaternion_world_wxyz",
            "sole_quaternion_world_wxyz",
        ):
            norms = np.linalg.norm(getattr(self, name), axis=-1)
            if not np.allclose(norms, 1.0, atol=1.0e-5):
                raise ContractError(f"{name} must be normalized wxyz")
        if np.any(self.contact < 0.0) or np.any(self.contact > 1.0):
            raise ContractError("contact must be in [0,1]")
        if np.any(self.contact_confidence < 0.0) or np.any(self.contact_confidence > 1.0):
            raise ContractError("contact_confidence must be in [0,1]")
        self.commands.validate(frame_count)


def derive_clip_kinematics(clip: CanonicalClip) -> CanonicalClip:
    """Return a copy whose velocity traces are derived at the 50 Hz boundary."""

    if not isinstance(clip, CanonicalClip):
        raise ContractError("derive_clip_kinematics requires a CanonicalClip")
    if clip.fps != 50.0 or clip.frame_count < 2:
        raise ContractError("canonical clip must contain at least two 50 Hz frames")
    derived = replace(
        clip,
        root_linear_velocity_world=finite_difference(clip.root_position_world, clip.fps),
        root_angular_velocity_world=angular_velocity_world_wxyz(
            clip.root_quaternion_world_wxyz, clip.fps
        ),
        joint_velocity=finite_difference(clip.joint_position, clip.fps),
        body_linear_velocity_world=finite_difference(clip.body_position_world, clip.fps),
        body_angular_velocity_world=angular_velocity_world_wxyz(
            clip.body_quaternion_world_wxyz, clip.fps
        ),
    )
    derived.validate()
    return derived
