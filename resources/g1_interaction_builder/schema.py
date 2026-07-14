from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

from resources.g1_terrain_builder.schema import SkeletonSpec


class InteractionHand(IntEnum):
    LEFT = 0
    RIGHT = 1


class InteractionPhase(IntEnum):
    APPROACH = 0
    REACH = 1
    CONTACT = 2
    LIFT = 3
    HOLD = 4


@dataclass(frozen=True)
class SourcePaths:
    sequence_id: str
    object_id: str
    robot: Path
    objects: Path
    meta: Path
    object_usd: Path


@dataclass
class RawInteractionClip:
    sequence_id: str
    object_id: str
    fps: float
    qpos: np.ndarray
    hand_dof: np.ndarray
    object_positions: np.ndarray
    object_rotations: np.ndarray
    hand_contacts: np.ndarray
    table_position: np.ndarray
    table_rotation: np.ndarray
    table_size: np.ndarray
    object_dimensions: np.ndarray
    source_frames: np.ndarray

    def validate(self) -> None:
        qpos = np.asarray(self.qpos)
        if qpos.ndim != 2 or qpos.shape[1] != 36 or qpos.shape[0] == 0:
            _validation_error(
                SourceValidationError,
                "invalid_shape",
                self.sequence_id,
                f"qpos shape must be (T, 36), got {qpos.shape}",
            )
        frames = qpos.shape[0]
        _require_positive_fps(
            SourceValidationError, self.sequence_id, self.fps
        )
        arrays = (
            ("qpos", qpos, (frames, 36)),
            ("hand dof", self.hand_dof, (frames, 14)),
            ("object position", self.object_positions, (frames, 3)),
            ("object rotation", self.object_rotations, (frames, 4)),
            ("hand contact", self.hand_contacts, (frames, 2)),
            ("table position", self.table_position, (3,)),
            ("table rotation", self.table_rotation, (4,)),
            ("table dimension", self.table_size, (3,)),
            ("object dimension", self.object_dimensions, (3,)),
            ("source frame", self.source_frames, (frames,)),
        )
        for name, value, shape in arrays:
            _require_shape(
                SourceValidationError,
                self.sequence_id,
                name,
                np.asarray(value),
                shape,
                frames,
            )
        _require_finite(
            SourceValidationError,
            self.sequence_id,
            arrays[:-1],
        )
        _require_unit_quaternions(
            SourceValidationError,
            self.sequence_id,
            (
                ("root quaternion", qpos[:, 3:7]),
                ("object quaternion", self.object_rotations),
                ("table quaternion", self.table_rotation),
            ),
        )
        _require_contacts(
            SourceValidationError,
            self.sequence_id,
            "hand contacts",
            self.hand_contacts,
        )
        _require_source_frames(
            SourceValidationError,
            self.sequence_id,
            self.source_frames,
        )
        _require_dimensions(
            SourceValidationError,
            self.sequence_id,
            self.table_size,
            self.object_dimensions,
        )


@dataclass
class CanonicalInteractionClip:
    sequence_id: str
    object_id: str
    fps: float
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    angular_velocities: np.ndarray
    foot_contacts: np.ndarray
    hand_contacts: np.ndarray
    hand_dof: np.ndarray
    hand_dof_velocities: np.ndarray
    hand_positions: np.ndarray
    hand_rotations: np.ndarray
    object_positions: np.ndarray
    object_rotations: np.ndarray
    object_velocities: np.ndarray
    object_angular_velocities: np.ndarray
    table_position: np.ndarray
    table_rotation: np.ndarray
    table_size: np.ndarray
    object_dimensions: np.ndarray
    source_frames: np.ndarray

    def validate(self) -> None:
        positions = np.asarray(self.positions)
        if positions.ndim != 3 or positions.shape[1:] != (31, 3):
            _validation_error(
                ConversionValidationError,
                "invalid_shape",
                self.sequence_id,
                f"position shape must be (T, 31, 3), got {positions.shape}",
            )
        if positions.shape[0] == 0:
            _validation_error(
                ConversionValidationError,
                "frame_count_mismatch",
                self.sequence_id,
                "position array has no frames",
            )
        frames = positions.shape[0]
        _require_positive_fps(
            ConversionValidationError, self.sequence_id, self.fps
        )
        arrays = (
            ("position", positions, (frames, 31, 3)),
            ("velocity", self.velocities, (frames, 31, 3)),
            ("rotation", self.rotations, (frames, 31, 4)),
            (
                "angular velocity",
                self.angular_velocities,
                (frames, 31, 3),
            ),
            ("foot contact", self.foot_contacts, (frames, 2)),
            ("hand contact", self.hand_contacts, (frames, 2)),
            ("hand dof", self.hand_dof, (frames, 14)),
            (
                "hand dof velocity",
                self.hand_dof_velocities,
                (frames, 14),
            ),
            ("hand position", self.hand_positions, (frames, 2, 3)),
            ("hand rotation", self.hand_rotations, (frames, 2, 4)),
            ("object position", self.object_positions, (frames, 3)),
            ("object rotation", self.object_rotations, (frames, 4)),
            ("object velocity", self.object_velocities, (frames, 3)),
            (
                "object angular velocity",
                self.object_angular_velocities,
                (frames, 3),
            ),
            ("table position", self.table_position, (3,)),
            ("table rotation", self.table_rotation, (4,)),
            ("table dimension", self.table_size, (3,)),
            ("object dimension", self.object_dimensions, (3,)),
            ("source frame", self.source_frames, (frames,)),
        )
        for name, value, shape in arrays:
            _require_shape(
                ConversionValidationError,
                self.sequence_id,
                name,
                np.asarray(value),
                shape,
                frames,
            )
        _require_finite(
            ConversionValidationError,
            self.sequence_id,
            arrays[:-1],
        )
        _require_unit_quaternions(
            ConversionValidationError,
            self.sequence_id,
            (
                ("local quaternion", self.rotations),
                ("hand quaternion", self.hand_rotations),
                ("object quaternion", self.object_rotations),
                ("table quaternion", self.table_rotation),
            ),
        )
        _require_contacts(
            ConversionValidationError,
            self.sequence_id,
            "foot contacts",
            self.foot_contacts,
        )
        _require_contacts(
            ConversionValidationError,
            self.sequence_id,
            "hand contacts",
            self.hand_contacts,
        )
        _require_source_frames(
            ConversionValidationError,
            self.sequence_id,
            self.source_frames,
        )
        _require_dimensions(
            ConversionValidationError,
            self.sequence_id,
            self.table_size,
            self.object_dimensions,
        )


@dataclass
class LabeledInteractionClip:
    motion: CanonicalInteractionClip
    active_hand: InteractionHand
    phases: np.ndarray
    time_to_contact: np.ndarray
    contact_frame: int
    lift_frame: int
    hold_frame: int
    support_height: float
    grasp_position_object: np.ndarray
    grasp_rotation_object: np.ndarray
    approach_direction_object: np.ndarray


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    start: int
    stop: int


@dataclass
class FeatureSet:
    values: np.ndarray
    offsets: np.ndarray
    scales: np.ndarray
    groups: tuple[FeatureGroup, ...]


@dataclass(frozen=True)
class EvaluationSplit:
    seed: int
    database_objects: tuple[str, ...]
    heldout_objects: tuple[str, ...]


@dataclass(frozen=True)
class PhaseConfig:
    stable_source_samples: int = 3
    max_relative_position_m: float = 0.02
    max_relative_angle_radians: float = 0.17453292519943295
    reach_seconds: float = 1.0
    lift_height_m: float = 0.05
    hold_speed_mps: float = 0.1
    hold_seconds: float = 0.2
    grasp_average_seconds: float = 0.2


class InteractionBuildError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class SourceValidationError(InteractionBuildError):
    pass


class ConversionValidationError(InteractionBuildError):
    pass


class InteractionValidationError(InteractionBuildError):
    pass


@dataclass
class InteractionArtifact:
    fps: int
    parents: np.ndarray
    range_starts: np.ndarray
    range_stops: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    angular_velocities: np.ndarray
    foot_contacts: np.ndarray
    hand_contacts: np.ndarray
    hand_dof: np.ndarray
    hand_dof_velocities: np.ndarray
    phases: np.ndarray
    active_hands: np.ndarray
    time_to_contact: np.ndarray
    object_positions: np.ndarray
    object_rotations: np.ndarray
    object_velocities: np.ndarray
    object_angular_velocities: np.ndarray
    table_positions: np.ndarray
    table_rotations: np.ndarray
    table_sizes: np.ndarray
    object_dimensions: np.ndarray
    grasp_positions_object: np.ndarray
    grasp_rotations_object: np.ndarray
    approach_directions_object: np.ndarray
    source_frames: np.ndarray


def _validation_error(
    error_type: type[InteractionBuildError],
    code: str,
    sequence_id: str,
    message: str,
) -> None:
    raise error_type(code, f"{sequence_id}: {message}")


def _require_positive_fps(
    error_type: type[InteractionBuildError], sequence_id: str, fps: float
) -> None:
    try:
        value = float(fps)
    except (TypeError, ValueError):
        _validation_error(
            error_type, "invalid_fps", sequence_id, f"invalid fps {fps!r}"
        )
    if not np.isfinite(value) or value <= 0:
        _validation_error(
            error_type, "invalid_fps", sequence_id, f"invalid fps {fps!r}"
        )


def _require_shape(
    error_type: type[InteractionBuildError],
    sequence_id: str,
    name: str,
    value: np.ndarray,
    expected: tuple[int, ...],
    frames: int,
) -> None:
    if value.shape == expected:
        return
    code = (
        "frame_count_mismatch"
        if value.ndim > 0 and value.shape[0] != frames
        else "invalid_shape"
    )
    _validation_error(
        error_type,
        code,
        sequence_id,
        f"{name} shape must be {expected}, got {value.shape}",
    )


def _require_finite(
    error_type: type[InteractionBuildError],
    sequence_id: str,
    arrays: tuple[tuple[str, object, tuple[int, ...]], ...],
) -> None:
    for name, value, _ in arrays:
        try:
            finite = np.isfinite(np.asarray(value)).all()
        except TypeError:
            finite = False
        if not finite:
            _validation_error(
                error_type,
                "non_finite",
                sequence_id,
                f"{name} contains non-finite values",
            )


def _require_unit_quaternions(
    error_type: type[InteractionBuildError],
    sequence_id: str,
    values: tuple[tuple[str, object], ...],
) -> None:
    for name, value in values:
        quaternions = np.asarray(value, np.float64)
        norms = np.linalg.norm(quaternions, axis=-1)
        if not np.all(np.abs(norms - 1.0) <= 1e-4):
            _validation_error(
                error_type,
                "invalid_quaternion",
                sequence_id,
                f"{name} is not a unit quaternion",
            )


def _require_contacts(
    error_type: type[InteractionBuildError],
    sequence_id: str,
    name: str,
    value: np.ndarray,
) -> None:
    contacts = np.asarray(value)
    if contacts.dtype != np.dtype(np.uint8) or not np.isin(
        contacts, (0, 1)
    ).all():
        _validation_error(
            error_type,
            "invalid_contact",
            sequence_id,
            f"{name} must be uint8 values in {{0, 1}}",
        )


def _require_source_frames(
    error_type: type[InteractionBuildError],
    sequence_id: str,
    value: np.ndarray,
) -> None:
    source_frames = np.asarray(value)
    if source_frames.dtype != np.dtype(np.int32):
        _validation_error(
            error_type,
            "invalid_source_frames",
            sequence_id,
            "source frames must have dtype int32",
        )
    if np.any(source_frames < 0) or np.any(np.diff(source_frames) < 0):
        _validation_error(
            error_type,
            "invalid_source_frames",
            sequence_id,
            "source frames must be nonnegative and nondecreasing",
        )


def _require_dimensions(
    error_type: type[InteractionBuildError],
    sequence_id: str,
    table_size: np.ndarray,
    object_dimensions: np.ndarray,
) -> None:
    for name, value in (
        ("table dimensions", table_size),
        ("object dimensions", object_dimensions),
    ):
        dimensions = np.asarray(value)
        if np.any(dimensions <= 0):
            _validation_error(
                error_type,
                "invalid_dimensions",
                sequence_id,
                f"{name} must be positive",
            )


G1_SKELETON = SkeletonSpec(
    names=(
        "Simulation", "Hips",
        "LeftHipPitch", "LeftHipRoll", "LeftHipYaw", "LeftKnee", "LeftAnkle", "LeftToe",
        "RightHipPitch", "RightHipRoll", "RightHipYaw", "RightKnee", "RightAnkle", "RightToe",
        "Spine", "Spine1", "Spine2",
        "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbow",
        "LeftWristRoll", "LeftWristPitch", "LeftWrist",
        "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbow",
        "RightWristRoll", "RightWristPitch", "RightWrist",
    ),
    parents=np.array(
        [-1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
         15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29],
        np.int32,
    ),
)

assert G1_SKELETON.signature() == (
    "6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7"
)
