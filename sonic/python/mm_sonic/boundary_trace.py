"""Immutable input-transition-to-prefix boundary trace (Stage R1, Task 1).

One controller-owned record binds a single X11 input transition to the single
committed prefix it produced. Its monotone timestamps and per-stage numeric
vectors make command, generation, transport, and tracking failures
distinguishable. The record is pure: it performs no I/O and copies nothing it
cannot validate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .joints import ContractError


# The eight design timestamps in their mandated non-decreasing order:
# X11 transition -> sampled -> MM start -> MM complete -> committed ->
# published+ack -> physics released -> first simulated frame.
_TIMESTAMP_FIELDS = (
    "input_observed_ns",
    "sampled_ns",
    "mm_started_ns",
    "mm_completed_ns",
    "committed_ns",
    "published_ack_ns",
    "physics_released_ns",
    "first_simulated_frame_ns",
)

# Registered numeric-vector widths in the MuJoCo target basis.
_VECTOR_FIELDS = (
    ("requested_velocity_mujoco", 3),
    ("requested_heading_mujoco_wxyz", 4),
    ("generated_virtual_root_displacement_mujoco", 3),
    ("published_physical_root_displacement_mujoco", 3),
    ("observed_mujoco_root_displacement", 3),
)


def _nonempty_identity(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ContractError(f"trace {label} must be a nonempty string")
    return value


def _timestamp(value: object, label: str) -> int:
    # bool is an int subclass; reject it so timestamps stay explicit integers.
    if type(value) is not int or value < 0:
        raise ContractError(
            f"trace {label} must be a nonnegative integer nanosecond timestamp"
        )
    return value


def _finite_vector(value: object, width: int, label: str) -> tuple[float, ...]:
    if type(value) not in (tuple, list) or len(value) != width:
        raise ContractError(f"trace {label} must have width {width}")
    output: list[float] = []
    for item in value:
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ContractError(f"trace {label} must contain finite values")
        output.append(float(item))
    return tuple(output)


@dataclass(frozen=True)
class BoundaryTrace:
    """One immutable binding of an input transition to a presented prefix."""

    input_transition_id: str
    presented_prefix_id: str
    input_observed_ns: int
    sampled_ns: int
    mm_started_ns: int
    mm_completed_ns: int
    committed_ns: int
    published_ack_ns: int
    physics_released_ns: int
    first_simulated_frame_ns: int
    requested_velocity_mujoco: tuple[float, float, float]
    requested_heading_mujoco_wxyz: tuple[float, float, float, float]
    generated_virtual_root_displacement_mujoco: tuple[float, float, float]
    published_physical_root_displacement_mujoco: tuple[float, float, float]
    observed_mujoco_root_displacement: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_transition_id",
            _nonempty_identity(self.input_transition_id, "input_transition_id"),
        )
        object.__setattr__(
            self,
            "presented_prefix_id",
            _nonempty_identity(self.presented_prefix_id, "presented_prefix_id"),
        )

        for name in _TIMESTAMP_FIELDS:
            object.__setattr__(self, name, _timestamp(getattr(self, name), name))
        for earlier, later in zip(_TIMESTAMP_FIELDS, _TIMESTAMP_FIELDS[1:]):
            if getattr(self, later) < getattr(self, earlier):
                raise ContractError(f"trace {later} must not precede {earlier}")

        for name, width in _VECTOR_FIELDS:
            object.__setattr__(
                self, name, _finite_vector(getattr(self, name), width, name)
            )
