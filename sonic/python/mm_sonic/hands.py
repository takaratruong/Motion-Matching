"""Mirrored Dex3 hand contract: motor order, limits, neutral profile, identity."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .joints import ContractError


LEFT_HAND_JOINT_ORDER = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
)

RIGHT_HAND_JOINT_ORDER = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

HAND_JOINT_COUNT = 7

# Pinned Dex3 motor limits in mirrored command order. The relaxed-fist targets
# below are the midpoint-derived close posture and lie within these ranges; the
# ranges are used only to validate overrides, never to alter the pinned targets.
LEFT_HAND_JOINT_RANGES = (
    (-1.0472, 1.0472),
    (-0.72426, 1.0472),
    (-1.74533, 1.74533),
    (-1.5708, 0.10472),
    (-1.74533, 0.10472),
    (-1.5708, 0.10472),
    (-1.74533, 0.10472),
)

RIGHT_HAND_JOINT_RANGES = (
    (-1.0472, 1.0472),
    (-1.0472, 0.72426),
    (-1.74533, 1.74533),
    (-0.10472, 1.5708),
    (-0.10472, 1.74533),
    (-0.10472, 1.5708),
    (-0.10472, 1.74533),
)

_LEFT_NEUTRAL = (0.0, 0.163, 0.875, -0.785, -0.875, -0.785, -0.875)
_RIGHT_NEUTRAL = (0.0, -0.154, -0.875, 0.785, 0.875, 0.785, 0.875)

_PROFILE_NAME = "dex3-relaxed-fist-v1"
_HASH_DOMAIN = b"mm-sonic-dex3-hand-targets/v1\0"
_LIMIT_EPSILON = 1.0e-6
_PIN_MARGIN = 0.01


@dataclass(frozen=True, eq=False)
class Dex3HandTargets:
    profile: str
    left: tuple[float, ...]
    right: tuple[float, ...]

    @property
    def left_f32(self) -> np.ndarray:
        return np.asarray(self.left, dtype="<f4")

    @property
    def right_f32(self) -> np.ndarray:
        return np.asarray(self.right, dtype="<f4")

    def __eq__(self, other: object) -> bool:
        """Compare the profile and the exact float32 values sent on the wire."""

        return (
            isinstance(other, Dex3HandTargets)
            and self.profile == other.profile
            and self.left_f32.tobytes(order="C")
            == other.left_f32.tobytes(order="C")
            and self.right_f32.tobytes(order="C")
            == other.right_f32.tobytes(order="C")
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.profile,
                self.left_f32.tobytes(order="C"),
                self.right_f32.tobytes(order="C"),
            )
        )


@dataclass(frozen=True)
class HandTrackingReport:
    median_absolute_error_rad: tuple[float, ...]
    median_position_rad: tuple[float, ...]
    unexpected_limit_joints: tuple[str, ...]
    error_pass: bool
    limit_pass: bool

    @property
    def passed(self) -> bool:
        return self.error_pass and self.limit_pass


def _validate_side(
    value: object,
    ranges: Sequence[tuple[float, float]],
    label: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list, np.ndarray)):
        raise ContractError(f"{label} must be a sequence of {HAND_JOINT_COUNT} numbers")
    items = list(value)
    if len(items) != HAND_JOINT_COUNT:
        raise ContractError(
            f"{label} must contain exactly {HAND_JOINT_COUNT} numbers, got {len(items)}"
        )
    converted: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, (int, float, np.floating, np.integer)):
            raise ContractError(f"{label}[{index}] must be a finite number")
        number = float(item)
        if not math.isfinite(number):
            raise ContractError(f"{label}[{index}] must be a finite number")
        lower, upper = ranges[index]
        if number < lower - _LIMIT_EPSILON or number > upper + _LIMIT_EPSILON:
            raise ContractError(
                f"{label}[{index}] = {number} is outside limit [{lower}, {upper}]"
            )
        converted.append(number)
    return tuple(converted)


NEUTRAL_HAND_TARGETS = Dex3HandTargets(
    profile=_PROFILE_NAME,
    left=_validate_side(_LEFT_NEUTRAL, LEFT_HAND_JOINT_RANGES, "left neutral"),
    right=_validate_side(_RIGHT_NEUTRAL, RIGHT_HAND_JOINT_RANGES, "right neutral"),
)


def validate_hand_targets(
    value: object,
    *,
    label: str = "hand_targets",
) -> Dex3HandTargets:
    """Validate one complete pinned-profile target without changing its values."""

    if not isinstance(value, Dex3HandTargets):
        raise ContractError(f"{label} must be Dex3HandTargets")
    if value.profile != _PROFILE_NAME:
        raise ContractError(f"{label} profile is not the pinned profile")
    return Dex3HandTargets(
        profile=_PROFILE_NAME,
        left=_validate_side(value.left, LEFT_HAND_JOINT_RANGES, f"{label}.left"),
        right=_validate_side(
            value.right,
            RIGHT_HAND_JOINT_RANGES,
            f"{label}.right",
        ),
    )


def resolve_hand_targets(
    *,
    left_hand_joints: object | None = None,
    right_hand_joints: object | None = None,
    default: Dex3HandTargets = NEUTRAL_HAND_TARGETS,
) -> Dex3HandTargets:
    """Resolve each supplied side independently; omission restores its default."""

    validated_default = validate_hand_targets(default, label="default hand targets")
    if left_hand_joints is None:
        left = validated_default.left
    else:
        left = _validate_side(left_hand_joints, LEFT_HAND_JOINT_RANGES, "left_hand_joints")
    if right_hand_joints is None:
        right = validated_default.right
    else:
        right = _validate_side(right_hand_joints, RIGHT_HAND_JOINT_RANGES, "right_hand_joints")
    return Dex3HandTargets(profile=validated_default.profile, left=left, right=right)


def _targets_sha256(targets: Dex3HandTargets) -> str:
    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    for label, order, vector in (
        ("left", LEFT_HAND_JOINT_ORDER, targets.left_f32),
        ("right", RIGHT_HAND_JOINT_ORDER, targets.right_f32),
    ):
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update("\0".join(order).encode("ascii"))
        digest.update(b"\0")
        digest.update(vector.tobytes())
    return digest.hexdigest()


def hand_targets_record(targets: Dex3HandTargets) -> dict[str, object]:
    """Return profile, mirrored joint orders, f32 vectors, and a domain-separated SHA-256."""

    canonical = validate_hand_targets(targets)
    return {
        "profile": canonical.profile,
        "left_joint_order": list(LEFT_HAND_JOINT_ORDER),
        "right_joint_order": list(RIGHT_HAND_JOINT_ORDER),
        "left_targets": [float(value) for value in canonical.left_f32],
        "right_targets": [float(value) for value in canonical.right_f32],
        "sha256": _targets_sha256(canonical),
    }


def parse_hand_targets_record(value: object) -> Dex3HandTargets:
    """Require the exact record keys/orders and recompute its vector SHA-256."""

    expected_keys = {
        "profile",
        "left_joint_order",
        "right_joint_order",
        "left_targets",
        "right_targets",
        "sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ContractError("hand targets record keys are invalid")
    if value["profile"] != _PROFILE_NAME:
        raise ContractError("hand targets record profile is not the pinned profile")
    if tuple(value["left_joint_order"]) != LEFT_HAND_JOINT_ORDER:
        raise ContractError("hand targets record left joint order is invalid")
    if tuple(value["right_joint_order"]) != RIGHT_HAND_JOINT_ORDER:
        raise ContractError("hand targets record right joint order is invalid")
    left = _validate_side(value["left_targets"], LEFT_HAND_JOINT_RANGES, "left_targets")
    right = _validate_side(value["right_targets"], RIGHT_HAND_JOINT_RANGES, "right_targets")
    targets = validate_hand_targets(
        Dex3HandTargets(profile=_PROFILE_NAME, left=left, right=right)
    )
    if not isinstance(value["sha256"], str) or value["sha256"] != _targets_sha256(targets):
        raise ContractError("hand targets record SHA-256 does not match its vectors")
    return targets


def _named_qpos_addresses(model: Any, names: Sequence[str]) -> tuple[int, ...]:
    addresses: list[int] = []
    for name in names:
        joint = model.joint(name)
        addresses.append(int(model.jnt_qposadr[int(joint.id)]))
    return tuple(addresses)


def hand_qpos_addresses(model: Any) -> tuple[int, ...]:
    """Resolve the 14 exact named joints through model.jnt_qposadr."""

    return _named_qpos_addresses(model, LEFT_HAND_JOINT_ORDER + RIGHT_HAND_JOINT_ORDER)


def hand_joint_ranges(model: Any) -> tuple[tuple[float, float], ...]:
    """Resolve the same 14 exact named joints through model.jnt_range."""

    ranges: list[tuple[float, float]] = []
    for name in LEFT_HAND_JOINT_ORDER + RIGHT_HAND_JOINT_ORDER:
        joint = model.joint(name)
        lower, upper = (float(bound) for bound in model.jnt_range[int(joint.id)])
        ranges.append((lower, upper))
    return tuple(ranges)


def measure_hand_tracking(
    states: Sequence[Any],
    *,
    qpos_addresses: Sequence[int],
    joint_ranges: Sequence[tuple[float, float]],
    targets: Dex3HandTargets,
    final_seconds: float = 1.0,
) -> HandTrackingReport:
    """Measure final-window medians, <=0.20 rad errors, and unexpected limit pinning."""

    joint_names = LEFT_HAND_JOINT_ORDER + RIGHT_HAND_JOINT_ORDER
    expected = 2 * HAND_JOINT_COUNT
    addresses = tuple(int(address) for address in qpos_addresses)
    ranges = tuple(joint_ranges)
    if len(addresses) != expected or len(ranges) != expected:
        raise ContractError("hand tracking requires 14 addresses and ranges")
    if not states:
        raise ContractError("hand tracking requires at least one state")
    validated_targets = validate_hand_targets(targets)
    target_vector = np.asarray(
        validated_targets.left + validated_targets.right,
        dtype=np.float64,
    )

    times = np.asarray([float(state.sim_time_s) for state in states], dtype=np.float64)
    final_start = float(times.max()) - float(final_seconds)
    window = [state for state, time in zip(states, times) if time >= final_start]
    if not window:
        raise ContractError("hand tracking final window is empty")

    positions = np.asarray(
        [[float(state.qpos[address]) for address in addresses] for state in window],
        dtype=np.float64,
    )
    median_position = np.median(positions, axis=0)
    median_abs_error = np.median(np.abs(positions - target_vector), axis=0)

    unexpected: list[str] = []
    for index, (lower, upper) in enumerate(ranges):
        position = float(median_position[index])
        target = float(target_vector[index])
        for bound in (lower, upper):
            if abs(position - bound) <= _PIN_MARGIN and abs(target - bound) > _PIN_MARGIN:
                unexpected.append(joint_names[index])
                break

    error_pass = bool(np.all(median_abs_error <= 0.20))
    limit_pass = not unexpected
    return HandTrackingReport(
        median_absolute_error_rad=tuple(float(value) for value in median_abs_error),
        median_position_rad=tuple(float(value) for value in median_position),
        unexpected_limit_joints=tuple(unexpected),
        error_pass=error_pass,
        limit_pass=limit_pass,
    )
