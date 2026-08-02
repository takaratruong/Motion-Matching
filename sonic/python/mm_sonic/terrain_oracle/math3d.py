"""Small, dependency-free 3D operations for canonical motion data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mm_sonic.joints import ContractError


_MINIMUM_QUATERNION_NORM = 1.0e-12


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


def _vector(value: object, width: int, label: str) -> np.ndarray:
    output = _readonly_float32(value, label)
    if output.shape != (width,):
        raise ContractError(f"{label} must have shape ({width},)")
    return output


def _quaternion(value: object, label: str) -> np.ndarray:
    output = _vector(value, 4, label)
    if not np.isclose(float(np.linalg.norm(output)), 1.0, atol=1.0e-5):
        raise ContractError(f"{label} must be a normalized wxyz quaternion")
    return output


def quaternion_multiply_wxyz(left: object, right: object) -> np.ndarray:
    """Multiply wxyz quaternions, broadcasting leading dimensions."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape[-1:] != (4,) or right_array.shape[-1:] != (4,):
        raise ContractError("quaternion operands must end in width 4")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ContractError("quaternion operands must be finite")
    lw, lx, ly, lz = np.moveaxis(left_array, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right_array, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quaternion_inverse_wxyz(value: object) -> np.ndarray:
    """Return the inverse of finite wxyz quaternions."""

    output = np.asarray(value, dtype=np.float64).copy()
    if output.shape[-1:] != (4,) or not np.isfinite(output).all():
        raise ContractError("quaternions must be finite with width 4")
    norm_squared = np.sum(output * output, axis=-1, keepdims=True)
    if np.any(norm_squared < _MINIMUM_QUATERNION_NORM**2):
        raise ContractError("quaternion norm is too small")
    output[..., 1:] *= -1.0
    return output / norm_squared


def unroll_quaternions_wxyz(value: object) -> np.ndarray:
    """Normalize a quaternion sequence and choose continuous antipodes."""

    output = _readonly_float32(value, "quaternions")
    if output.ndim < 2 or output.shape[-1] != 4:
        raise ContractError("quaternions must have shape [T,...,4]")
    norms = np.linalg.norm(output, axis=-1)
    if np.any(norms < _MINIMUM_QUATERNION_NORM):
        raise ContractError("quaternions contain a zero norm")
    normalized = (np.asarray(output, dtype=np.float64) / norms[..., None]).copy()
    traces = normalized.reshape((normalized.shape[0], -1, 4))
    for index in range(1, traces.shape[0]):
        flip = np.sum(traces[index - 1] * traces[index], axis=-1) < 0.0
        traces[index, flip] *= -1.0
    return _readonly_float32(normalized, "unrolled quaternions")


def slerp_wxyz(start: object, end: object, fraction: object) -> np.ndarray:
    """Interpolate normalized wxyz quaternions along the shortest path."""

    left = np.asarray(start, dtype=np.float64)
    right = np.asarray(end, dtype=np.float64)
    if left.shape[-1:] != (4,) or right.shape[-1:] != (4,):
        raise ContractError("slerp quaternions must end in width 4")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ContractError("slerp quaternions must be finite")
    left_norm = np.linalg.norm(left, axis=-1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=-1, keepdims=True)
    if np.any(left_norm < _MINIMUM_QUATERNION_NORM) or np.any(right_norm < _MINIMUM_QUATERNION_NORM):
        raise ContractError("slerp quaternion norm is too small")
    left = left / left_norm
    right = right / right_norm
    dot = np.sum(left * right, axis=-1, keepdims=True)
    right = np.where(dot < 0.0, -right, right)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    fraction_array = np.asarray(fraction, dtype=np.float64)[..., None]
    theta = np.arccos(dot)
    sine = np.sin(theta)
    near = sine < 1.0e-7
    with np.errstate(invalid="ignore", divide="ignore"):
        left_weight = np.sin((1.0 - fraction_array) * theta) / sine
        right_weight = np.sin(fraction_array * theta) / sine
    blended = left_weight * left + right_weight * right
    linear = (1.0 - fraction_array) * left + fraction_array * right
    output = np.where(near, linear, blended)
    output /= np.linalg.norm(output, axis=-1, keepdims=True)
    return _readonly_float32(output, "slerp output")


def finite_difference(value: object, fps: float) -> np.ndarray:
    """Differentiate a [T,...] trace using central interior differences."""

    samples = _readonly_float32(value, "finite-difference values")
    if samples.ndim < 1 or samples.shape[0] < 2:
        raise ContractError("finite-difference values need at least two frames")
    if type(fps) not in (int, float) or not np.isfinite(fps) or fps <= 0.0:
        raise ContractError("fps must be a positive finite number")
    result = np.empty_like(samples)
    result[0] = (samples[1] - samples[0]) * np.float32(fps)
    result[-1] = (samples[-1] - samples[-2]) * np.float32(fps)
    if samples.shape[0] > 2:
        result[1:-1] = (samples[2:] - samples[:-2]) * np.float32(fps / 2.0)
    result.flags.writeable = False
    return result


def angular_velocity_world_wxyz(quaternions: object, fps: float) -> np.ndarray:
    """Estimate world-frame angular velocity from a wxyz orientation trace."""

    trace = unroll_quaternions_wxyz(quaternions)
    derivative = finite_difference(trace, fps)
    product = quaternion_multiply_wxyz(derivative, quaternion_inverse_wxyz(trace))
    return _readonly_float32(2.0 * product[..., 1:], "angular velocity")


@dataclass(frozen=True)
class RigidTransform:
    """A local-to-world rigid transform represented by a wxyz quaternion."""

    translation_world: np.ndarray
    quaternion_world_from_local_wxyz: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "translation_world", _vector(self.translation_world, 3, "translation")
        )
        object.__setattr__(
            self,
            "quaternion_world_from_local_wxyz",
            _quaternion(self.quaternion_world_from_local_wxyz, "rotation"),
        )

    def compose(self, other: "RigidTransform") -> "RigidTransform":
        if not isinstance(other, RigidTransform):
            raise ContractError("can only compose a RigidTransform")
        return RigidTransform(
            self.apply_points(other.translation_world),
            quaternion_multiply_wxyz(
                self.quaternion_world_from_local_wxyz,
                other.quaternion_world_from_local_wxyz,
            ),
        )

    def inverse(self) -> "RigidTransform":
        inverse_rotation = quaternion_inverse_wxyz(
            self.quaternion_world_from_local_wxyz
        )
        return RigidTransform(
            self._rotate(-np.asarray(self.translation_world), inverse_rotation),
            inverse_rotation,
        )

    @staticmethod
    def _rotate(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        pure = np.concatenate(
            (np.zeros(np.asarray(points).shape[:-1] + (1,), dtype=np.float64), points),
            axis=-1,
        )
        rotated = quaternion_multiply_wxyz(
            quaternion_multiply_wxyz(rotation, pure), quaternion_inverse_wxyz(rotation)
        )
        return rotated[..., 1:]

    def apply_points(self, points_local: object) -> np.ndarray:
        points = np.asarray(points_local, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ContractError("points must be finite with trailing width 3")
        output = self._rotate(
            points, np.asarray(self.quaternion_world_from_local_wxyz, dtype=np.float64)
        ) + self.translation_world
        return _readonly_float32(output, "transformed points")
