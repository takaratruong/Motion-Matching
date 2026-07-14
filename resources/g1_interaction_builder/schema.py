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


_ARTIFACT_FLOAT_TOLERANCE = 1e-4
_ARTIFACT_HOLD_CONTACT_SAMPLES = 5


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

    def validate(self) -> None:
        if isinstance(self.fps, bool) or self.fps != 25:
            raise InteractionValidationError(
                "artifact_fps", f"artifact fps must be exactly 25, got {self.fps!r}"
            )

        positions = np.asarray(self.positions)
        if positions.ndim != 3 or positions.shape[1:] != (31, 3):
            raise InteractionValidationError(
                "artifact_shape",
                "positions shape must be (N, 31, 3), "
                f"got {positions.shape}",
            )
        frame_count = positions.shape[0]
        if frame_count == 0:
            raise InteractionValidationError(
                "artifact_shape", "positions must contain at least one frame"
            )

        starts = np.asarray(self.range_starts)
        if starts.ndim != 1 or len(starts) == 0:
            raise InteractionValidationError(
                "artifact_shape",
                "range_starts shape must be a nonempty (C,) vector, "
                f"got {starts.shape}",
            )
        clip_count = len(starts)

        arrays = (
            ("parents", self.parents, (31,), np.int32),
            ("range_starts", starts, (clip_count,), np.int32),
            (
                "range_stops",
                self.range_stops,
                (clip_count,),
                np.int32,
            ),
            ("positions", positions, (frame_count, 31, 3), np.float32),
            (
                "velocities",
                self.velocities,
                (frame_count, 31, 3),
                np.float32,
            ),
            (
                "rotations",
                self.rotations,
                (frame_count, 31, 4),
                np.float32,
            ),
            (
                "angular_velocities",
                self.angular_velocities,
                (frame_count, 31, 3),
                np.float32,
            ),
            (
                "foot_contacts",
                self.foot_contacts,
                (frame_count, 2),
                np.uint8,
            ),
            (
                "hand_contacts",
                self.hand_contacts,
                (frame_count, 2),
                np.uint8,
            ),
            (
                "hand_dof",
                self.hand_dof,
                (frame_count, 14),
                np.float32,
            ),
            (
                "hand_dof_velocities",
                self.hand_dof_velocities,
                (frame_count, 14),
                np.float32,
            ),
            ("phases", self.phases, (frame_count,), np.uint8),
            (
                "active_hands",
                self.active_hands,
                (clip_count,),
                np.uint8,
            ),
            (
                "time_to_contact",
                self.time_to_contact,
                (frame_count,),
                np.float32,
            ),
            (
                "object_positions",
                self.object_positions,
                (frame_count, 3),
                np.float32,
            ),
            (
                "object_rotations",
                self.object_rotations,
                (frame_count, 4),
                np.float32,
            ),
            (
                "object_velocities",
                self.object_velocities,
                (frame_count, 3),
                np.float32,
            ),
            (
                "object_angular_velocities",
                self.object_angular_velocities,
                (frame_count, 3),
                np.float32,
            ),
            (
                "table_positions",
                self.table_positions,
                (clip_count, 3),
                np.float32,
            ),
            (
                "table_rotations",
                self.table_rotations,
                (clip_count, 4),
                np.float32,
            ),
            (
                "table_sizes",
                self.table_sizes,
                (clip_count, 3),
                np.float32,
            ),
            (
                "object_dimensions",
                self.object_dimensions,
                (clip_count, 3),
                np.float32,
            ),
            (
                "grasp_positions_object",
                self.grasp_positions_object,
                (clip_count, 3),
                np.float32,
            ),
            (
                "grasp_rotations_object",
                self.grasp_rotations_object,
                (clip_count, 4),
                np.float32,
            ),
            (
                "approach_directions_object",
                self.approach_directions_object,
                (clip_count, 3),
                np.float32,
            ),
            (
                "source_frames",
                self.source_frames,
                (frame_count,),
                np.int32,
            ),
        )
        validated = {}
        for name, value, shape, dtype in arrays:
            array = np.asarray(value)
            if array.shape != shape:
                raise InteractionValidationError(
                    "artifact_shape",
                    f"{name} shape must be {shape}, got {array.shape}",
                )
            expected_dtype = np.dtype(dtype)
            if array.dtype != expected_dtype:
                raise InteractionValidationError(
                    "artifact_dtype",
                    f"{name} dtype must be {expected_dtype.name}, "
                    f"got {array.dtype}",
                )
            validated[name] = array

        if not np.array_equal(validated["parents"], G1_SKELETON.parents):
            raise InteractionValidationError(
                "artifact_parents",
                "parents must match the exact G1 parent array",
            )

        stops = validated["range_stops"]
        if starts[0] != 0:
            raise InteractionValidationError(
                "artifact_ranges", "range coverage must start at 0"
            )
        for index, (start, stop) in enumerate(zip(starts, stops)):
            if stop <= start:
                raise InteractionValidationError(
                    "artifact_ranges",
                    f"empty range {index}: [{int(start)}, {int(stop)})",
                )
            if index:
                previous_stop = stops[index - 1]
                if start > previous_stop:
                    raise InteractionValidationError(
                        "artifact_ranges",
                        f"range coverage has gap before range {index}",
                    )
                if start < previous_stop:
                    raise InteractionValidationError(
                        "artifact_ranges",
                        f"range coverage has overlap before range {index}",
                    )
        if stops[-1] != frame_count:
            raise InteractionValidationError(
                "artifact_ranges",
                "range coverage must end at frame count "
                f"{frame_count}, got {int(stops[-1])}",
            )

        for name, array in validated.items():
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                raise InteractionValidationError(
                    "artifact_non_finite",
                    f"{name} contains non-finite values",
                )

        for name in ("foot_contacts", "hand_contacts", "active_hands"):
            if not np.isin(validated[name], (0, 1)).all():
                raise InteractionValidationError(
                    "artifact_enum",
                    f"{name} must contain only 0 or 1",
                )

        phases = validated["phases"]
        if not np.isin(phases, tuple(int(value) for value in InteractionPhase)).all():
            raise InteractionValidationError(
                "artifact_phases", "phases must contain values from 0 through 4"
            )
        required_phases = (
            (InteractionPhase.CONTACT, "contact"),
            (InteractionPhase.LIFT, "lift"),
            (InteractionPhase.HOLD, "hold"),
        )
        source_frames = validated["source_frames"]
        time_to_contact = validated["time_to_contact"]
        hand_contacts = validated["hand_contacts"]
        approach_directions = validated["approach_directions_object"]
        for index, (start, stop) in enumerate(zip(starts, stops)):
            clip_phases = phases[start:stop]
            if np.any(np.diff(clip_phases.astype(np.int16)) < 0):
                raise InteractionValidationError(
                    "artifact_phases",
                    f"phases must be monotonic within range {index}",
                )
            for phase, label in required_phases:
                if not np.any(clip_phases == int(phase)):
                    raise InteractionValidationError(
                        "artifact_phases",
                        f"range {index} is missing the {label} phase",
                    )
            clip_source_frames = source_frames[start:stop]
            if np.any(clip_source_frames < 0) or np.any(
                np.diff(clip_source_frames) < 0
            ):
                raise InteractionValidationError(
                    "artifact_source_frames",
                    "source_frames must be nonnegative and nondecreasing "
                    f"within range {index}",
                )

            clip_time = time_to_contact[start:stop]
            if np.any(clip_time < -_ARTIFACT_FLOAT_TOLERANCE):
                raise InteractionValidationError(
                    "artifact_time_to_contact",
                    "time_to_contact must be nonnegative within tolerance "
                    f"in range {index}",
                )
            if np.any(
                np.diff(clip_time.astype(np.float64))
                > _ARTIFACT_FLOAT_TOLERANCE
            ):
                raise InteractionValidationError(
                    "artifact_time_to_contact",
                    "time_to_contact must be nonincreasing within tolerance "
                    f"in range {index}",
                )
            contact_or_later = clip_phases >= int(
                InteractionPhase.CONTACT
            )
            if np.any(
                np.abs(clip_time[contact_or_later])
                > _ARTIFACT_FLOAT_TOLERANCE
            ):
                raise InteractionValidationError(
                    "artifact_time_to_contact",
                    "time_to_contact must be zero from CONTACT onward "
                    f"within tolerance in range {index}",
                )

            approach = approach_directions[index].astype(np.float64)
            if abs(float(approach[1])) > _ARTIFACT_FLOAT_TOLERANCE:
                raise InteractionValidationError(
                    "artifact_approach_direction",
                    "approach direction must be horizontal in range "
                    f"{index}",
                )
            approach_norm = float(np.linalg.norm(approach))
            if abs(approach_norm - 1.0) > _ARTIFACT_FLOAT_TOLERANCE:
                raise InteractionValidationError(
                    "artifact_approach_direction",
                    "approach direction must be unit length in range "
                    f"{index}",
                )

            active_hand = int(validated["active_hands"][index])
            clip_contacts = hand_contacts[start:stop, active_hand]
            for phase in (InteractionPhase.CONTACT, InteractionPhase.LIFT):
                if not np.all(
                    clip_contacts[clip_phases == int(phase)] == 1
                ):
                    raise InteractionValidationError(
                        "artifact_hand_contacts",
                        f"{phase.name} phase must have active-hand contact "
                        f"in range {index}",
                    )
            hold_indices = np.flatnonzero(
                clip_phases == int(InteractionPhase.HOLD)
            )
            if len(hold_indices) < _ARTIFACT_HOLD_CONTACT_SAMPLES:
                raise InteractionValidationError(
                    "artifact_hand_contacts",
                    "range "
                    f"{index} must contain at least "
                    f"{_ARTIFACT_HOLD_CONTACT_SAMPLES} HOLD samples",
                )
            if not np.all(
                clip_contacts[
                    hold_indices[:_ARTIFACT_HOLD_CONTACT_SAMPLES]
                ]
                == 1
            ):
                raise InteractionValidationError(
                    "artifact_hand_contacts",
                    "first 5 HOLD samples must have active-hand contact "
                    f"in range {index}",
                )

        for name in (
            "rotations",
            "object_rotations",
            "table_rotations",
            "grasp_rotations_object",
        ):
            norms = np.linalg.norm(validated[name].astype(np.float64), axis=-1)
            if not np.all(np.abs(norms - 1.0) <= 1e-4):
                raise InteractionValidationError(
                    "artifact_quaternion",
                    f"{name} must contain unit quaternions",
                )

        for name in ("table_sizes", "object_dimensions"):
            if np.any(validated[name] <= 0):
                raise InteractionValidationError(
                    "artifact_dimensions", f"{name} must be positive"
                )


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
