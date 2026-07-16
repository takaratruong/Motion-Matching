"""Transactional ownership of the canonical 50 Hz SONIC target timeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .joints import ContractError, JointContract
from .resample import (
    MappedSourceBoundary,
    ResampledSourceChunk,
    _freeze_mapping,
    _joint_limits,
    resample_source_chunk,
)
from .schema import InitialBoundary, SourceChunk
from .transform import (
    holden_to_mujoco_quaternions,
    holden_to_mujoco_vectors,
    map_source_joints,
)


_JOINT_COUNT = 29
_TARGET_ROWS = 20
_TARGET_RATE_HZ = 50.0
_SEAM_TOLERANCE = 1.0e-6
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _owned_array(
    value: object,
    dtype: np.dtype | type,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    source = np.asarray(value)
    if source.shape != shape or source.dtype.kind not in "iuf":
        raise ContractError(f"{label} must have shape {shape} and a real dtype")
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.asarray(source, dtype=dtype, order="C").copy(order="C")
    if output.dtype.kind == "f" and not np.all(np.isfinite(output)):
        raise ContractError(f"{label} must contain finite values")
    output.flags.writeable = False
    return output


def _check_quaternions(value: np.ndarray, label: str) -> None:
    norms = np.linalg.norm(value.astype(np.float64), axis=-1)
    if np.any(np.abs(norms - 1.0) > 1.0e-5):
        raise ContractError(f"{label} must contain unit quaternions")


def _hash_arrays(*arrays: tuple[np.ndarray, str]) -> str:
    digest = hashlib.sha256()
    for value, dtype in arrays:
        digest.update(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TargetChunk:
    schema: str
    session_id: str
    accepted_chunk_id: str
    source_candidate_id: str
    frame_index: np.ndarray
    timestamps_s: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_quat_w: np.ndarray
    physical_pelvis_position: np.ndarray
    virtual_root_position: np.ndarray
    virtual_root_quat_w: np.ndarray
    scene: Mapping[str, object]
    command: Mapping[str, object]
    hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema != "target-chunk/v1" or type(self.schema) is not str:
            raise ContractError("target schema must equal target-chunk/v1")
        for label, value in (
            ("session_id", self.session_id),
            ("accepted_chunk_id", self.accepted_chunk_id),
            ("source_candidate_id", self.source_candidate_id),
        ):
            if type(value) is not str or not value:
                raise ContractError(f"target {label} must be a nonempty string")
        frame_index = _owned_array(
            self.frame_index, np.int64, (_TARGET_ROWS,), "target frame_index"
        )
        if np.any(frame_index < 0) or not np.all(np.diff(frame_index) == 1):
            raise ContractError("target frame_index must be nonnegative and contiguous")
        timestamps = _owned_array(
            self.timestamps_s, np.float64, (_TARGET_ROWS,), "target timestamps_s"
        )
        if not np.array_equal(
            timestamps, frame_index.astype(np.float64) / _TARGET_RATE_HZ
        ):
            raise ContractError("target timestamps_s must equal frame_index / 50.0")
        joint_position = _owned_array(
            self.joint_position,
            np.float32,
            (_TARGET_ROWS, _JOINT_COUNT),
            "target joint_position",
        )
        joint_velocity = _owned_array(
            self.joint_velocity,
            np.float32,
            (_TARGET_ROWS, _JOINT_COUNT),
            "target joint_velocity",
        )
        body_quat_w = _owned_array(
            self.body_quat_w,
            np.float32,
            (_TARGET_ROWS, 4),
            "target body_quat_w",
        )
        physical_position = _owned_array(
            self.physical_pelvis_position,
            np.float32,
            (_TARGET_ROWS, 3),
            "target physical_pelvis_position",
        )
        virtual_position = _owned_array(
            self.virtual_root_position,
            np.float32,
            (_TARGET_ROWS, 3),
            "target virtual_root_position",
        )
        virtual_quat = _owned_array(
            self.virtual_root_quat_w,
            np.float32,
            (_TARGET_ROWS, 4),
            "target virtual_root_quat_w",
        )
        _check_quaternions(body_quat_w, "target body_quat_w")
        _check_quaternions(virtual_quat, "target virtual_root_quat_w")
        hashes = dict(self.hashes) if isinstance(self.hashes, Mapping) else None
        if hashes is None or set(hashes) != {
            "canonical_target_sha256",
            "diagnostic_sha256",
        }:
            raise ContractError("target hashes must contain the exact registered keys")
        if any(
            type(value) is not str or _SHA256.fullmatch(value) is None
            for value in hashes.values()
        ):
            raise ContractError("target hashes must be lowercase SHA-256 digests")
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "joint_position", joint_position)
        object.__setattr__(self, "joint_velocity", joint_velocity)
        object.__setattr__(self, "body_quat_w", body_quat_w)
        object.__setattr__(self, "physical_pelvis_position", physical_position)
        object.__setattr__(self, "virtual_root_position", virtual_position)
        object.__setattr__(self, "virtual_root_quat_w", virtual_quat)
        object.__setattr__(self, "scene", _freeze_mapping(self.scene, "target scene"))
        object.__setattr__(
            self, "command", _freeze_mapping(self.command, "target command")
        )
        object.__setattr__(self, "hashes", MappingProxyType(hashes))

    @property
    def buffer(self) -> CanonicalTargetBuffer:
        return CanonicalTargetBuffer(
            joint_position=self.joint_position,
            joint_velocity=self.joint_velocity,
            body_quat_w=self.body_quat_w,
            frame_index=self.frame_index,
        )


@dataclass(frozen=True)
class CanonicalTargetBuffer:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_quat_w: np.ndarray
    frame_index: np.ndarray

    def __post_init__(self) -> None:
        raw_frame = np.asarray(self.frame_index)
        if raw_frame.ndim != 1 or raw_frame.shape[0] <= 0:
            raise ContractError("canonical frame_index must be a nonempty vector")
        count = int(raw_frame.shape[0])
        frame_index = _owned_array(
            raw_frame, np.int64, (count,), "canonical frame_index"
        )
        if np.any(frame_index < 0) or not np.all(np.diff(frame_index) == 1):
            raise ContractError(
                "canonical frame_index must be nonnegative and contiguous"
            )
        joint_position = _owned_array(
            self.joint_position,
            np.float32,
            (count, _JOINT_COUNT),
            "canonical joint_position",
        )
        joint_velocity = _owned_array(
            self.joint_velocity,
            np.float32,
            (count, _JOINT_COUNT),
            "canonical joint_velocity",
        )
        body_quat_w = _owned_array(
            self.body_quat_w,
            np.float32,
            (count, 4),
            "canonical body_quat_w",
        )
        _check_quaternions(body_quat_w, "canonical body_quat_w")
        object.__setattr__(self, "joint_position", joint_position)
        object.__setattr__(self, "joint_velocity", joint_velocity)
        object.__setattr__(self, "body_quat_w", body_quat_w)
        object.__setattr__(self, "frame_index", frame_index)

    @property
    def count(self) -> int:
        return int(self.frame_index.shape[0])


@dataclass(frozen=True)
class PreparedTargetCandidate:
    target: TargetChunk
    source_candidate_id: str
    predecessor_id: str | None
    _end_boundary: MappedSourceBoundary = field(repr=False, compare=False)
    _expected_last_frame_index: int = field(repr=False, compare=False)


def _validate_joint_limits(position: np.ndarray, contract: JointContract) -> None:
    lower, upper = _joint_limits(contract)
    invalid = (position < lower) | (position > upper)
    if np.any(invalid):
        joint = int(np.flatnonzero(invalid)[0])
        raise ContractError(
            f"initial boundary joint limit violation at target joint {joint}"
        )


def _initial_boundary(
    initial: InitialBoundary,
    contract: JointContract,
) -> MappedSourceBoundary:
    if not isinstance(initial, InitialBoundary):
        raise ContractError("initial must be an immutable InitialBoundary")
    if type(initial.session_id) is not str or not initial.session_id:
        raise ContractError("initial session_id must be a nonempty string")
    position = map_source_joints(
        initial.joint_position_source, initial.source_joint_names, contract
    )
    velocity = map_source_joints(
        initial.joint_velocity_source, initial.source_joint_names, contract
    )
    _validate_joint_limits(position.astype(np.float64), contract)
    physical_position = holden_to_mujoco_vectors(
        initial.physical_pelvis_position_holden
    )
    physical_quat = holden_to_mujoco_quaternions(
        initial.physical_pelvis_orientation_holden
    )
    virtual_position = holden_to_mujoco_vectors(initial.virtual_root_position_holden)
    virtual_quat = holden_to_mujoco_quaternions(
        initial.virtual_root_orientation_holden
    )
    return MappedSourceBoundary(
        joint_position=_owned_array(
            position, np.float32, (_JOINT_COUNT,), "initial joint position"
        ),
        joint_velocity=_owned_array(
            velocity, np.float32, (_JOINT_COUNT,), "initial joint velocity"
        ),
        body_quat_w=_owned_array(
            physical_quat, np.float32, (4,), "initial physical orientation"
        ),
        physical_pelvis_position=_owned_array(
            physical_position, np.float32, (3,), "initial physical position"
        ),
        virtual_root_position=_owned_array(
            virtual_position, np.float32, (3,), "initial virtual position"
        ),
        virtual_root_quat_w=_owned_array(
            virtual_quat, np.float32, (4,), "initial virtual orientation"
        ),
    )


def _quaternion_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_f64 = np.asarray(left, dtype=np.float64)
    right_f64 = np.asarray(right, dtype=np.float64)
    left_f64 /= np.linalg.norm(left_f64)
    right_f64 /= np.linalg.norm(right_f64)
    left_w = float(left_f64[0])
    right_w = float(right_f64[0])
    left_xyz = left_f64[1:]
    right_xyz = right_f64[1:]
    relative_w = left_w * right_w + float(np.dot(left_xyz, right_xyz))
    relative_xyz = (
        left_w * right_xyz
        - right_w * left_xyz
        - np.cross(left_xyz, right_xyz)
    )
    return 2.0 * math.atan2(
        float(np.linalg.norm(relative_xyz)), abs(relative_w)
    )


def _check_seam(
    accepted: MappedSourceBoundary,
    candidate: MappedSourceBoundary,
) -> None:
    for label, left, right in (
        ("joint position", accepted.joint_position, candidate.joint_position),
        ("joint velocity", accepted.joint_velocity, candidate.joint_velocity),
        (
            "physical pelvis position",
            accepted.physical_pelvis_position,
            candidate.physical_pelvis_position,
        ),
        (
            "virtual root position",
            accepted.virtual_root_position,
            candidate.virtual_root_position,
        ),
    ):
        error = float(
            np.max(
                np.abs(
                    left.astype(np.float64) - right.astype(np.float64)
                )
            )
        )
        if error > _SEAM_TOLERANCE:
            raise ContractError(
                f"{label} seam mismatch {error:.17g} exceeds 1e-6"
            )
    for label, left, right in (
        (
            "physical orientation",
            accepted.body_quat_w,
            candidate.body_quat_w,
        ),
        (
            "virtual orientation",
            accepted.virtual_root_quat_w,
            candidate.virtual_root_quat_w,
        ),
    ):
        error = _quaternion_distance(left, right)
        if error > _SEAM_TOLERANCE:
            raise ContractError(
                f"{label} seam mismatch {error:.17g} rad exceeds 1e-6"
            )


def _align_quaternion_rows(
    rows: np.ndarray,
    preceding: np.ndarray,
) -> np.ndarray:
    output = np.ascontiguousarray(rows, dtype=np.float32).copy(order="C")
    previous = np.asarray(preceding, dtype=np.float64)
    for index in range(output.shape[0]):
        if float(np.dot(previous, output[index].astype(np.float64))) < 0.0:
            output[index] *= -1.0
        previous = output[index].astype(np.float64)
    output.flags.writeable = False
    return output


def _end_boundary(resampled: ResampledSourceChunk) -> MappedSourceBoundary:
    return MappedSourceBoundary(
        joint_position=_owned_array(
            resampled.joint_position[-1],
            np.float32,
            (_JOINT_COUNT,),
            "candidate end joint position",
        ),
        joint_velocity=_owned_array(
            resampled.joint_velocity[-1],
            np.float32,
            (_JOINT_COUNT,),
            "candidate end joint velocity",
        ),
        body_quat_w=_owned_array(
            resampled.body_quat_w[-1],
            np.float32,
            (4,),
            "candidate end physical orientation",
        ),
        physical_pelvis_position=_owned_array(
            resampled.physical_pelvis_position[-1],
            np.float32,
            (3,),
            "candidate end physical position",
        ),
        virtual_root_position=_owned_array(
            resampled.virtual_root_position[-1],
            np.float32,
            (3,),
            "candidate end virtual position",
        ),
        virtual_root_quat_w=_owned_array(
            resampled.virtual_root_quat_w[-1],
            np.float32,
            (4,),
            "candidate end virtual orientation",
        ),
    )


class TargetTimeline:
    """Own frame zero, one pending candidate, and committed target rows."""

    def __init__(self, initial: InitialBoundary, contract: JointContract):
        boundary = _initial_boundary(initial, contract)
        self._session_id = initial.session_id
        self._contract = contract
        self._initial_buffer = CanonicalTargetBuffer(
            joint_position=boundary.joint_position[np.newaxis, :],
            joint_velocity=boundary.joint_velocity[np.newaxis, :],
            body_quat_w=boundary.body_quat_w[np.newaxis, :],
            frame_index=np.array([0], dtype=np.int64),
        )
        self._canonical_buffer = self._initial_buffer
        self._last_boundary = boundary
        self._last_frame_index = 0
        self._last_accepted_candidate_id: str | None = None
        self._pending: PreparedTargetCandidate | None = None

    @property
    def initial_buffer(self) -> CanonicalTargetBuffer:
        return self._initial_buffer

    @property
    def canonical_buffer(self) -> CanonicalTargetBuffer:
        return self._canonical_buffer

    @property
    def last_frame_index(self) -> int:
        return self._last_frame_index

    @property
    def last_accepted_candidate_id(self) -> str | None:
        return self._last_accepted_candidate_id

    @property
    def pending_candidate_id(self) -> str | None:
        return (
            None if self._pending is None else self._pending.source_candidate_id
        )

    def prepare(self, source: SourceChunk) -> PreparedTargetCandidate:
        if self._pending is not None:
            raise ContractError(
                "target candidate "
                f"{self._pending.source_candidate_id} is already pending"
            )
        if not isinstance(source, SourceChunk):
            raise ContractError("source must be an immutable SourceChunk")
        if source.session_id != self._session_id:
            raise ContractError("source candidate session does not match the timeline")
        if source.predecessor_id != self._last_accepted_candidate_id:
            raise ContractError(
                "source candidate predecessor does not match the accepted timeline"
            )
        if source.candidate_id == self._last_accepted_candidate_id:
            raise ContractError("source candidate repeats the accepted candidate ID")
        resampled = resample_source_chunk(source, self._contract)
        _check_seam(self._last_boundary, resampled.left_boundary)
        resampled = replace(
            resampled,
            body_quat_w=_align_quaternion_rows(
                resampled.body_quat_w, self._last_boundary.body_quat_w
            ),
            virtual_root_quat_w=_align_quaternion_rows(
                resampled.virtual_root_quat_w,
                self._last_boundary.virtual_root_quat_w,
            ),
        )
        first_index = self._last_frame_index + 1
        frame_index = np.arange(
            first_index, first_index + _TARGET_ROWS, dtype=np.int64
        )
        timestamps = frame_index.astype(np.float64) / _TARGET_RATE_HZ
        canonical_hash = _hash_arrays(
            (resampled.joint_position, "<f4"),
            (resampled.joint_velocity, "<f4"),
            (resampled.body_quat_w, "<f4"),
            (frame_index, "<i8"),
        )
        diagnostic_hash = _hash_arrays(
            (resampled.physical_pelvis_position, "<f4"),
            (resampled.virtual_root_position, "<f4"),
            (resampled.virtual_root_quat_w, "<f4"),
        )
        accepted_chunk_id = (
            f"{self._session_id}:target:{first_index:020d}-"
            f"{first_index + _TARGET_ROWS - 1:020d}"
        )
        target = TargetChunk(
            schema="target-chunk/v1",
            session_id=self._session_id,
            accepted_chunk_id=accepted_chunk_id,
            source_candidate_id=source.candidate_id,
            frame_index=frame_index,
            timestamps_s=timestamps,
            joint_position=resampled.joint_position,
            joint_velocity=resampled.joint_velocity,
            body_quat_w=resampled.body_quat_w,
            physical_pelvis_position=resampled.physical_pelvis_position,
            virtual_root_position=resampled.virtual_root_position,
            virtual_root_quat_w=resampled.virtual_root_quat_w,
            scene=resampled.scene,
            command=resampled.command,
            hashes={
                "canonical_target_sha256": canonical_hash,
                "diagnostic_sha256": diagnostic_hash,
            },
        )
        prepared = PreparedTargetCandidate(
            target=target,
            source_candidate_id=source.candidate_id,
            predecessor_id=source.predecessor_id,
            _end_boundary=_end_boundary(resampled),
            _expected_last_frame_index=self._last_frame_index,
        )
        self._pending = prepared
        return prepared

    def commit(self, prepared: PreparedTargetCandidate) -> TargetChunk:
        if not isinstance(prepared, PreparedTargetCandidate):
            raise ContractError("commit requires a prepared target candidate")
        if self._pending is None:
            raise ContractError("no prepared candidate is pending")
        if prepared.source_candidate_id != self._pending.source_candidate_id:
            raise ContractError(
                "commit candidate ID does not match the pending candidate"
            )
        if prepared is not self._pending:
            raise ContractError("commit requires the current prepared candidate object")
        if prepared._expected_last_frame_index != self._last_frame_index:
            raise ContractError("prepared candidate no longer matches timeline state")
        target = prepared.target
        current = self._canonical_buffer
        combined = CanonicalTargetBuffer(
            joint_position=np.concatenate(
                (current.joint_position, target.joint_position), axis=0
            ),
            joint_velocity=np.concatenate(
                (current.joint_velocity, target.joint_velocity), axis=0
            ),
            body_quat_w=np.concatenate(
                (current.body_quat_w, target.body_quat_w), axis=0
            ),
            frame_index=np.concatenate((current.frame_index, target.frame_index)),
        )
        new_last_frame = int(target.frame_index[-1])
        self._canonical_buffer = combined
        self._last_boundary = prepared._end_boundary
        self._last_frame_index = new_last_frame
        self._last_accepted_candidate_id = prepared.source_candidate_id
        self._pending = None
        return target

    def abort(self, candidate_id: str) -> None:
        if type(candidate_id) is not str or not candidate_id:
            raise ContractError("abort candidate ID must be a nonempty string")
        if self._pending is None:
            raise ContractError("no prepared candidate is pending")
        if candidate_id != self._pending.source_candidate_id:
            raise ContractError(
                "abort candidate ID does not match the pending candidate"
            )
        self._pending = None
