"""Exact 25 Hz to 50 Hz interpolation for validated MM source chunks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .joints import ContractError, JointContract
from .schema import SourceChunk
from .transform import (
    holden_to_mujoco_quaternions,
    holden_to_mujoco_vectors,
    map_source_joints,
)


_SOURCE_RATE_HZ = 25
_SOURCE_INTERVALS = 10
_SUPPORTED_SOURCE_INTERVALS = (5, 10)
_SOURCE_BOUNDARIES = 11
_TARGET_ROWS = 20
_TARGET_ROWS_PER_INTERVAL = 2
_JOINT_COUNT = 29
_SOURCE_DT_S = float(np.float32(0.04))
_MINIMUM_QUATERNION_NORM = 1.0e-12
_SLERP_LINEAR_THRESHOLD = 0.9995


@dataclass(frozen=True)
class MappedSourceBoundary:
    """One source boundary after name mapping and basis conversion."""

    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_quat_w: np.ndarray
    physical_pelvis_position: np.ndarray
    virtual_root_position: np.ndarray
    virtual_root_quat_w: np.ndarray


@dataclass(frozen=True)
class ResampledSourceChunk:
    """Twenty new target rows plus the checked-but-unemitted left boundary."""

    session_id: str
    candidate_id: str
    predecessor_id: str | None
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_quat_w: np.ndarray
    physical_pelvis_position: np.ndarray
    virtual_root_position: np.ndarray
    virtual_root_quat_w: np.ndarray
    scene: Mapping[str, object]
    command: Mapping[str, object]
    left_boundary: MappedSourceBoundary


def _float64_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    source = np.asarray(value)
    if source.shape != shape or source.dtype.kind not in "iuf":
        raise ContractError(f"{label} must have shape {shape} and a real dtype")
    output = np.asarray(source, dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise ContractError(f"{label} must contain only finite values")
    return output


def _readonly_float32(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    source = _float64_array(value, shape, label)
    with np.errstate(over="ignore", invalid="ignore"):
        output = source.astype(np.float32, order="C", copy=True)
    if not np.all(np.isfinite(output)):
        raise ContractError(f"{label} exceeds the finite binary32 range")
    output = np.ascontiguousarray(output).copy(order="C")
    output.flags.writeable = False
    return output


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, np.ndarray):
        output = np.ascontiguousarray(value).copy(order="C")
        output.flags.writeable = False
        return output
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ContractError(f"{label} keys must be strings")
    return MappingProxyType(
        {key: _freeze_value(item) for key, item in value.items()}
    )


def hermite_pair(
    q0: np.ndarray,
    v0: np.ndarray,
    q1: np.ndarray,
    v1: np.ndarray,
    dt: float,
    u: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate cubic Hermite position and its analytic time derivative."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64) for value in (q0, v0, q1, v1)
    )
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ContractError("Hermite endpoint arrays must have identical shapes")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ContractError("Hermite endpoint arrays must be finite")
    if type(dt) not in (int, float) or not math.isfinite(float(dt)) or dt <= 0.0:
        raise ContractError("Hermite dt must be finite and positive")
    if (
        type(u) not in (int, float)
        or not math.isfinite(float(u))
        or not 0.0 <= u <= 1.0
    ):
        raise ContractError("Hermite u must be finite and in [0, 1]")
    q0_f64, v0_f64, q1_f64, v1_f64 = arrays
    dt_f64 = float(dt)
    u_f64 = float(u)
    u2 = u_f64 * u_f64
    u3 = u2 * u_f64
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u_f64
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    q = (
        h00 * q0_f64
        + h10 * dt_f64 * v0_f64
        + h01 * q1_f64
        + h11 * dt_f64 * v1_f64
    )
    dh00 = 6.0 * u2 - 6.0 * u_f64
    dh10 = 3.0 * u2 - 4.0 * u_f64 + 1.0
    dh01 = -6.0 * u2 + 6.0 * u_f64
    dh11 = 3.0 * u2 - 2.0 * u_f64
    v = (
        (dh00 * q0_f64 + dh01 * q1_f64) / dt_f64
        + dh10 * v0_f64
        + dh11 * v1_f64
    )
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
        raise ContractError("Hermite interpolation produced a non-finite result")
    return q, v


def shortest_path_slerp(
    q0: np.ndarray,
    q1: np.ndarray,
    u: float,
) -> np.ndarray:
    """Normalize and interpolate wxyz quaternions on the shortest path."""

    left = np.asarray(q0, dtype=np.float64)
    right = np.asarray(q1, dtype=np.float64)
    if (
        left.shape != right.shape
        or left.ndim == 0
        or left.shape[-1] != 4
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ContractError("SLERP endpoints must be finite matching wxyz arrays")
    if (
        type(u) not in (int, float)
        or not math.isfinite(float(u))
        or not 0.0 <= u <= 1.0
    ):
        raise ContractError("SLERP u must be finite and in [0, 1]")
    left_norm = np.linalg.norm(left, axis=-1)
    right_norm = np.linalg.norm(right, axis=-1)
    if np.any(left_norm < _MINIMUM_QUATERNION_NORM) or np.any(
        right_norm < _MINIMUM_QUATERNION_NORM
    ):
        raise ContractError("SLERP endpoint quaternion norm is below 1e-12")
    left = left / left_norm[..., np.newaxis]
    right = right / right_norm[..., np.newaxis]
    dot = np.sum(left * right, axis=-1)
    right = np.where((dot < 0.0)[..., np.newaxis], -right, right)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    parameter = float(u)
    linear = dot >= _SLERP_LINEAR_THRESHOLD
    theta = np.arccos(dot)
    sine = np.sin(theta)
    safe_sine = np.where(linear, 1.0, sine)
    left_weight = np.sin((1.0 - parameter) * theta) / safe_sine
    right_weight = np.sin(parameter * theta) / safe_sine
    spherical = (
        left_weight[..., np.newaxis] * left
        + right_weight[..., np.newaxis] * right
    )
    lerped = (1.0 - parameter) * left + parameter * right
    output = np.where(linear[..., np.newaxis], lerped, spherical)
    norms = np.linalg.norm(output, axis=-1)
    if np.any(norms < _MINIMUM_QUATERNION_NORM) or not np.all(np.isfinite(norms)):
        raise ContractError("SLERP produced an invalid quaternion")
    return output / norms[..., np.newaxis]


def _joint_limits(contract: JointContract) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(_JOINT_COUNT, np.nan, dtype=np.float64)
    upper = np.full(_JOINT_COUNT, np.nan, dtype=np.float64)
    if len(contract.rows) != _JOINT_COUNT:
        raise ContractError("joint contract must contain exactly 29 rows")
    for row in contract.rows:
        if row.target_index < 0 or row.target_index >= _JOINT_COUNT:
            raise ContractError("joint contract target indices are invalid")
        if np.isfinite(lower[row.target_index]):
            raise ContractError("joint contract target indices are duplicated")
        lower[row.target_index] = row.lower
        upper[row.target_index] = row.upper
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ContractError("joint contract target limits are incomplete")
    return lower, upper


def _validate_joint_limits(
    position: np.ndarray,
    contract: JointContract,
    label: str,
) -> None:
    lower, upper = _joint_limits(contract)
    invalid = (position < lower[np.newaxis, :]) | (
        position > upper[np.newaxis, :]
    )
    if np.any(invalid):
        row, joint = np.argwhere(invalid)[0]
        raise ContractError(
            f"{label} joint limit violation at row {int(row)}, target joint "
            f"{int(joint)}: {position[row, joint]:.17g} not in "
            f"[{lower[joint]:.17g}, {upper[joint]:.17g}]"
        )


def _mapped_boundary(
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    body_quat_w: np.ndarray,
    physical_position: np.ndarray,
    virtual_position: np.ndarray,
    virtual_quat_w: np.ndarray,
) -> MappedSourceBoundary:
    return MappedSourceBoundary(
        joint_position=_readonly_float32(
            joint_position, (_JOINT_COUNT,), "mapped boundary joint position"
        ),
        joint_velocity=_readonly_float32(
            joint_velocity, (_JOINT_COUNT,), "mapped boundary joint velocity"
        ),
        body_quat_w=_readonly_float32(
            body_quat_w, (4,), "mapped boundary physical orientation"
        ),
        physical_pelvis_position=_readonly_float32(
            physical_position, (3,), "mapped boundary physical position"
        ),
        virtual_root_position=_readonly_float32(
            virtual_position, (3,), "mapped boundary virtual position"
        ),
        virtual_root_quat_w=_readonly_float32(
            virtual_quat_w, (4,), "mapped boundary virtual orientation"
        ),
    )


def resample_source_chunk(
    source: SourceChunk,
    contract: JointContract,
) -> ResampledSourceChunk:
    """Map and resample 11 source boundaries into exactly 20 new rows."""

    if not isinstance(source, SourceChunk):
        raise ContractError("source must be an immutable SourceChunk")
    if type(source.session_id) is not str or not source.session_id:
        raise ContractError("source session_id must be a nonempty string")
    if type(source.candidate_id) is not str or not source.candidate_id:
        raise ContractError("source candidate_id must be a nonempty string")
    if source.predecessor_id is not None and type(source.predecessor_id) is not str:
        raise ContractError("source predecessor_id must be a string or None")
    if source.source_rate_hz != _SOURCE_RATE_HZ:
        raise ContractError("source rate must equal 25 Hz")
    if source.source_intervals not in _SUPPORTED_SOURCE_INTERVALS:
        raise ContractError(
            "source interval count must be one of "
            + " or ".join(str(count) for count in _SUPPORTED_SOURCE_INTERVALS)
        )
    source_intervals = source.source_intervals
    source_boundaries = source_intervals + 1
    target_rows = source_intervals * _TARGET_ROWS_PER_INTERVAL
    timestamps = _float64_array(
        source.timestamps_s, (source_boundaries,), "source timestamps"
    )
    expected_timestamps = np.array(
        [
            np.float32(index) / np.float32(_SOURCE_RATE_HZ)
            for index in range(source_boundaries)
        ],
        dtype=np.float32,
    )
    source_timestamp_bits = np.asarray(source.timestamps_s, dtype=np.float32)
    if not np.array_equal(
        source_timestamp_bits.view(np.uint32), expected_timestamps.view(np.uint32)
    ) or not np.all(np.diff(timestamps) > 0.0):
        raise ContractError("source timestamps must be the exact increasing 25 Hz grid")

    _float64_array(
        source.joint_position_source,
        (source_boundaries, _JOINT_COUNT),
        "source joint position",
    )
    _float64_array(
        source.joint_velocity_source,
        (source_boundaries, _JOINT_COUNT),
        "source joint velocity",
    )
    _float64_array(
        source.physical_pelvis_position_holden,
        (source_boundaries, 3),
        "source physical pelvis position",
    )
    _float64_array(
        source.physical_pelvis_orientation_holden,
        (source_boundaries, 4),
        "source physical pelvis orientation",
    )
    _float64_array(
        source.virtual_root_position_holden,
        (source_boundaries, 3),
        "source virtual root position",
    )
    _float64_array(
        source.virtual_root_orientation_holden,
        (source_boundaries, 4),
        "source virtual root orientation",
    )

    mapped_position = map_source_joints(
        source.joint_position_source, source.source_joint_names, contract
    )
    mapped_velocity = map_source_joints(
        source.joint_velocity_source, source.source_joint_names, contract
    )
    physical_position = holden_to_mujoco_vectors(
        source.physical_pelvis_position_holden
    )
    virtual_position = holden_to_mujoco_vectors(source.virtual_root_position_holden)
    physical_quaternion = holden_to_mujoco_quaternions(
        source.physical_pelvis_orientation_holden
    )
    virtual_quaternion = holden_to_mujoco_quaternions(
        source.virtual_root_orientation_holden
    )

    mapped_position_f64 = mapped_position.astype(np.float64)
    mapped_velocity_f64 = mapped_velocity.astype(np.float64)
    _validate_joint_limits(mapped_position_f64, contract, "source endpoint")
    target_position = np.empty((target_rows, _JOINT_COUNT), np.float64)
    target_velocity = np.empty((target_rows, _JOINT_COUNT), np.float64)
    target_physical_position = np.empty((target_rows, 3), np.float64)
    target_virtual_position = np.empty((target_rows, 3), np.float64)
    target_physical_quaternion = np.empty((target_rows, 4), np.float64)
    target_virtual_quaternion = np.empty((target_rows, 4), np.float64)
    for interval in range(source_intervals):
        midpoint = 2 * interval
        right = midpoint + 1
        midpoint_q, midpoint_v = hermite_pair(
            mapped_position_f64[interval],
            mapped_velocity_f64[interval],
            mapped_position_f64[interval + 1],
            mapped_velocity_f64[interval + 1],
            _SOURCE_DT_S,
            0.5,
        )
        target_position[midpoint] = midpoint_q
        target_position[right] = mapped_position[interval + 1]
        target_velocity[midpoint] = midpoint_v
        target_velocity[right] = mapped_velocity[interval + 1]
        target_physical_position[midpoint] = (
            physical_position[interval].astype(np.float64)
            + physical_position[interval + 1].astype(np.float64)
        ) * 0.5
        target_physical_position[right] = physical_position[interval + 1]
        target_virtual_position[midpoint] = (
            virtual_position[interval].astype(np.float64)
            + virtual_position[interval + 1].astype(np.float64)
        ) * 0.5
        target_virtual_position[right] = virtual_position[interval + 1]
        target_physical_quaternion[midpoint] = shortest_path_slerp(
            physical_quaternion[interval], physical_quaternion[interval + 1], 0.5
        )
        target_physical_quaternion[right] = physical_quaternion[interval + 1]
        target_virtual_quaternion[midpoint] = shortest_path_slerp(
            virtual_quaternion[interval], virtual_quaternion[interval + 1], 0.5
        )
        target_virtual_quaternion[right] = virtual_quaternion[interval + 1]

    _validate_joint_limits(target_position, contract, "interpolated")
    result_position = _readonly_float32(
        target_position, (target_rows, _JOINT_COUNT), "target joint position"
    )
    result_velocity = _readonly_float32(
        target_velocity, (target_rows, _JOINT_COUNT), "target joint velocity"
    )
    result_physical_position = _readonly_float32(
        target_physical_position,
        (target_rows, 3),
        "target physical pelvis position",
    )
    result_virtual_position = _readonly_float32(
        target_virtual_position,
        (target_rows, 3),
        "target virtual root position",
    )
    result_physical_quaternion = _readonly_float32(
        target_physical_quaternion,
        (target_rows, 4),
        "target physical pelvis orientation",
    )
    result_virtual_quaternion = _readonly_float32(
        target_virtual_quaternion,
        (target_rows, 4),
        "target virtual root orientation",
    )
    return ResampledSourceChunk(
        session_id=source.session_id,
        candidate_id=source.candidate_id,
        predecessor_id=source.predecessor_id,
        joint_position=result_position,
        joint_velocity=result_velocity,
        body_quat_w=result_physical_quaternion,
        physical_pelvis_position=result_physical_position,
        virtual_root_position=result_virtual_position,
        virtual_root_quat_w=result_virtual_quaternion,
        scene=_freeze_mapping(source.scene, "source scene"),
        command=_freeze_mapping(source.command, "source command"),
        left_boundary=_mapped_boundary(
            mapped_position[0],
            mapped_velocity[0],
            physical_quaternion[0],
            physical_position[0],
            virtual_position[0],
            virtual_quaternion[0],
        ),
    )
