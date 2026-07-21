"""Frozen full-body motion schema for simultaneous walk/pickup diffusion."""

from dataclasses import dataclass

import numpy as np

from resources import quat as holden_quat


BONE_COUNT = 31
FRAME_DIM = 3 + BONE_COUNT * 6 + 3
WALK_FRAMES = 50
PICKUP_FRAMES = 50
OVERLAP_FRAMES = 20
TIMELINE_FRAMES = 80
FPS = 25.0
_ROTATION_EPSILON = 1e-8


@dataclass(frozen=True)
class DecodedMotion:
    positions: np.ndarray
    rotations: np.ndarray
    velocities: np.ndarray
    angular_velocities: np.ndarray
    contacts: np.ndarray


def walk_slice() -> slice:
    return slice(0, WALK_FRAMES)


def pickup_slice() -> slice:
    return slice(WALK_FRAMES - OVERLAP_FRAMES, TIMELINE_FRAMES)


def _array(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _anchor(value: np.ndarray) -> np.ndarray:
    return _array(value, (3,), "anchor")


def _normalised_quaternions(rotations: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(rotations, axis=-1, keepdims=True)
    if np.any(lengths < _ROTATION_EPSILON):
        raise ValueError("rotations contain a zero-length quaternion")
    result = rotations / lengths
    result = np.where(result[..., :1] < 0.0, -result, result)
    return result


def _quaternion_to_rotation6d(rotations: np.ndarray) -> np.ndarray:
    matrices = holden_quat.to_xform(rotations)
    return np.concatenate((matrices[..., :, 0], matrices[..., :, 1]), axis=-1)


def _rotation6d_to_quaternion(rotation6d: np.ndarray) -> np.ndarray:
    first = rotation6d[..., :3]
    first_length = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_length < _ROTATION_EPSILON):
        raise ValueError("rotation6d first column norm is below 1e-8")
    first = first / first_length

    second = rotation6d[..., 3:] - np.sum(
        first * rotation6d[..., 3:], axis=-1, keepdims=True
    ) * first
    second_length = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_length < _ROTATION_EPSILON):
        raise ValueError("rotation6d second column norm is below 1e-8")
    second = second / second_length
    third = np.cross(first, second)
    matrices = np.stack((first, second, third), axis=-1)
    rotations = np.asarray(holden_quat.from_xform(matrices), dtype=np.float64)
    rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
    return np.where(rotations[..., :1] < 0.0, -rotations, rotations)


def _finite_difference_vectors(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    if len(values) == 1:
        return result
    result[0] = (values[1] - values[0]) * FPS
    result[-1] = (values[-1] - values[-2]) * FPS
    if len(values) > 2:
        result[1:-1] = (values[2:] - values[:-2]) * (0.5 * FPS)
    return result


def _finite_difference_quaternions(rotations: np.ndarray) -> np.ndarray:
    result = np.zeros(rotations.shape[:-1] + (3,), dtype=np.float64)
    if len(rotations) == 1:
        return result
    unrolled = holden_quat.unroll(rotations.copy())
    result[0] = holden_quat.to_scaled_angle_axis(
        holden_quat.mul(unrolled[1], holden_quat.inv(unrolled[0]))
    ) * FPS
    result[-1] = holden_quat.to_scaled_angle_axis(
        holden_quat.mul(unrolled[-1], holden_quat.inv(unrolled[-2]))
    ) * FPS
    if len(rotations) > 2:
        result[1:-1] = holden_quat.to_scaled_angle_axis(
            holden_quat.mul(unrolled[2:], holden_quat.inv(unrolled[:-2]))
        ) * (0.5 * FPS)
    return result


def encode_motion(
    positions: np.ndarray, rotations: np.ndarray, contacts: np.ndarray,
    anchor: np.ndarray,
) -> np.ndarray:
    """Encode a local Holden pose sequence into the strict 192-channel schema."""
    positions = np.asarray(positions, dtype=np.float64)
    if (
        positions.ndim != 3
        or positions.shape[1:] != (BONE_COUNT, 3)
        or not np.isfinite(positions).all()
    ):
        raise ValueError("positions must be finite with shape (N, 31, 3)")
    frames = len(positions)
    rotations = _array(rotations, (frames, BONE_COUNT, 4), "rotations")
    contacts = _array(contacts, (frames, 3), "contacts")
    anchor = _anchor(anchor)

    rotation6d = _quaternion_to_rotation6d(_normalised_quaternions(rotations))
    return np.concatenate(
        (positions[:, 0] - anchor, rotation6d.reshape(frames, -1), contacts),
        axis=-1,
    ).astype(np.float32)


def decode_motion(
    encoded: np.ndarray, local_positions: np.ndarray, anchor: np.ndarray,
) -> DecodedMotion:
    """Decode root motion and rotations while restoring frozen local offsets."""
    encoded = np.asarray(encoded, dtype=np.float64)
    if encoded.ndim != 2 or encoded.shape[1] != FRAME_DIM or not np.isfinite(encoded).all():
        raise ValueError(f"encoded motion must be finite with shape (N, {FRAME_DIM})")
    frames = len(encoded)
    if frames == 0:
        raise ValueError("encoded motion must contain at least one frame")
    local_positions = _array(local_positions, (BONE_COUNT, 3), "local_positions")
    anchor = _anchor(anchor)

    rotations = _rotation6d_to_quaternion(
        encoded[:, 3:3 + BONE_COUNT * 6].reshape(frames, BONE_COUNT, 6)
    )
    positions = np.broadcast_to(
        local_positions, (frames, BONE_COUNT, 3)
    ).copy()
    positions[:, 0] = encoded[:, :3] + anchor
    return DecodedMotion(
        positions=positions.astype(np.float32),
        rotations=rotations.astype(np.float32),
        velocities=_finite_difference_vectors(positions).astype(np.float32),
        angular_velocities=_finite_difference_quaternions(rotations).astype(
            np.float32
        ),
        contacts=encoded[:, -3:].astype(np.float32),
    )
