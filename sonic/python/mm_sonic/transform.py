"""Named joint and Holden/MuJoCo coordinate transforms."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .joints import ContractError, JointContract, reorder_source_to_target


_JOINT_COUNT = 29
_MINIMUM_QUATERNION_NORM = 1.0e-12
_HOLDEN_TO_MUJOCO_BASIS = np.array(
    [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
    dtype=np.float64,
)


def _readonly_float32(value: np.ndarray, label: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.asarray(value, dtype=np.float64).astype(
            np.float32, order="C", casting="unsafe", copy=True
        )
    if not np.all(np.isfinite(output)):
        raise ContractError(f"{label} is outside the finite binary32 range")
    output = np.ascontiguousarray(output).copy(order="C")
    output.flags.writeable = False
    return output


def _finite_numeric(value: object, width: int, label: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim == 0 or source.shape[-1] != width:
        raise ContractError(f"{label} must end in width {width}")
    if source.dtype.kind not in "iuf":
        raise ContractError(f"{label} must be a real numeric array")
    source = np.asarray(source, dtype=np.float64)
    if not np.all(np.isfinite(source)):
        raise ContractError(f"{label} must contain only finite values")
    return source


def holden_to_mujoco_vectors(value: object) -> np.ndarray:
    """Map Holden ``[x, y, z]`` vectors to MuJoCo ``[x, -z, y]``."""

    source = _finite_numeric(value, 3, "Holden vectors")
    output = np.empty(source.shape, dtype=np.float64)
    output[..., 0] = source[..., 0]
    output[..., 1] = -source[..., 2]
    output[..., 2] = source[..., 1]
    return _readonly_float32(output, "MuJoCo vectors")


def mujoco_to_holden_vectors(value: object) -> np.ndarray:
    """Map MuJoCo ``[x, y, z]`` vectors to Holden ``[x, z, -y]``."""

    source = _finite_numeric(value, 3, "MuJoCo vectors")
    output = np.empty(source.shape, dtype=np.float64)
    output[..., 0] = source[..., 0]
    output[..., 1] = source[..., 2]
    output[..., 2] = -source[..., 1]
    return _readonly_float32(output, "Holden vectors")


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_inverse(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value, dtype=np.float64).copy()
    output[..., 1:] *= -1.0
    return output


def _normalized_unrolled(value: object, label: str) -> np.ndarray:
    source = _finite_numeric(value, 4, label).copy(order="C")
    norms = np.linalg.norm(source, axis=-1)
    if np.any(norms < _MINIMUM_QUATERNION_NORM):
        raise ContractError(f"{label} contains a quaternion below norm 1e-12")
    source /= norms[..., np.newaxis]
    flat = source.reshape((-1, 4))
    for index in range(1, flat.shape[0]):
        if float(np.dot(flat[index - 1], flat[index])) < 0.0:
            flat[index] *= -1.0
    return source


def _change_quaternion_basis(
    value: object,
    basis: np.ndarray,
    label: str,
) -> np.ndarray:
    source = _normalized_unrolled(value, label)
    broadcast_basis = np.broadcast_to(basis, source.shape)
    inverse = np.broadcast_to(_quat_inverse(basis), source.shape)
    converted = _quat_multiply(
        _quat_multiply(broadcast_basis, source), inverse
    )
    converted /= np.linalg.norm(converted, axis=-1)[..., np.newaxis]
    return _readonly_float32(converted, label)


def holden_to_mujoco_quaternions(value: object) -> np.ndarray:
    """Conjugate normalized wxyz quaternions into the MuJoCo basis."""

    return _change_quaternion_basis(
        value,
        _HOLDEN_TO_MUJOCO_BASIS,
        "Holden quaternions",
    )


def mujoco_to_holden_quaternions(value: object) -> np.ndarray:
    """Conjugate normalized wxyz quaternions into the Holden basis."""

    return _change_quaternion_basis(
        value,
        _quat_inverse(_HOLDEN_TO_MUJOCO_BASIS),
        "MuJoCo quaternions",
    )


def map_source_joints(
    values: object,
    source_joint_names: Sequence[str],
    contract: JointContract,
) -> np.ndarray:
    """Cast once to float32, then reorder only through checked joint names."""

    # The Task 2 public reorder API performs the canonical contract validation.
    reorder_source_to_target(np.zeros(_JOINT_COUNT, dtype=np.float32), contract)
    if type(source_joint_names) not in (list, tuple):
        raise ContractError("source_joint_names must be a list or tuple")
    names = tuple(source_joint_names)
    expected = tuple(row.source_joint for row in contract.rows)
    if (
        len(names) != _JOINT_COUNT
        or any(type(name) is not str or not name for name in names)
        or len(set(names)) != _JOINT_COUNT
        or set(names) != set(expected)
    ):
        raise ContractError(
            "source_joint_names do not exactly cover the joint contract"
        )
    source = _finite_numeric(values, _JOINT_COUNT, "source joint values")
    with np.errstate(over="ignore", invalid="ignore"):
        source_float32 = source.astype(np.float32, order="C", copy=True)
    if not np.all(np.isfinite(source_float32)):
        raise ContractError("source joint values exceed finite binary32")
    indices = {name: index for index, name in enumerate(names)}
    output = np.empty(source_float32.shape, dtype=np.float32, order="C")
    for row in contract.rows:
        output[..., row.target_index] = source_float32[
            ..., indices[row.source_joint]
        ]
    output.flags.writeable = False
    return output
