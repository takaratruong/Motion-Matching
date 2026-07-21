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


# The eight design timestamps in their corrected non-decreasing order:
# X11 transition -> sampled -> MM start -> MM complete ->
# publication sent (wait=False socket return) -> committed ->
# physics release requested -> simulation advance completed.
#
# Publication is *sent* strictly before mm.commit/timeline.commit in the
# adapter, so publication_sent_ns precedes committed_ns.  Two prior
# acknowledgement/first-frame timestamp fields are deliberately absent: the
# responsive publish is wait=False (no GEAR acknowledgement after CONTROL
# activation) and no first-simulated-frame timestamp is exposed anywhere.
_TIMESTAMP_FIELDS = (
    "input_observed_ns",
    "sampled_ns",
    "mm_started_ns",
    "mm_completed_ns",
    "publication_sent_ns",
    "committed_ns",
    "physics_release_requested_ns",
    "simulation_advance_completed_ns",
)

# Registered numeric-vector widths in the MuJoCo target basis.  The false
# published-physical-root-displacement field is removed: the SONIC wire
# carries only joint_position/body_quat_w and no root translation, so there is
# no independent published physical-root channel to report.
_VECTOR_FIELDS = (
    ("requested_velocity_mujoco", 3),
    ("requested_heading_mujoco_wxyz", 4),
    ("generated_virtual_root_displacement_mujoco", 3),
    # First and last source-step applied velocities in the MuJoCo target
    # basis, transformed by the existing Holden-to-MuJoCo helper. These are
    # honest producer-side references, never inferred from displacement.
    ("applied_velocity_mujoco_first", 3),
    ("applied_velocity_mujoco_last", 3),
    # First and last source-step applied headings in the MuJoCo target basis.
    # Honest producer-side references for the capped desired heading.
    ("applied_heading_mujoco_wxyz_first", 4),
    ("applied_heading_mujoco_wxyz_last", 4),
)

# Applied-heading fields must additionally be unit quaternions within the
# repository's shared tolerance.
_UNIT_QUATERNION_FIELDS = (
    "applied_heading_mujoco_wxyz_first",
    "applied_heading_mujoco_wxyz_last",
)
_QUATERNION_NORM_TOLERANCE = 1.0e-5


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
    publication_sent_ns: int
    committed_ns: int
    physics_release_requested_ns: int
    simulation_advance_completed_ns: int
    requested_velocity_mujoco: tuple[float, float, float]
    requested_heading_mujoco_wxyz: tuple[float, float, float, float]
    generated_virtual_root_displacement_mujoco: tuple[float, float, float]
    # First/last source-step applied velocities in MuJoCo coordinates.
    applied_velocity_mujoco_first: tuple[float, float, float]
    applied_velocity_mujoco_last: tuple[float, float, float]
    # First/last source-step applied headings in MuJoCo coordinates.
    applied_heading_mujoco_wxyz_first: tuple[float, float, float, float]
    applied_heading_mujoco_wxyz_last: tuple[float, float, float, float]
    # Real state-log measurement, or None when the log does not bound the
    # released chunk.  Never a made-up zero.
    observed_mujoco_root_displacement: tuple[float, float, float] | None

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
        for name in _UNIT_QUATERNION_FIELDS:
            norm = math.sqrt(sum(component * component
                                 for component in getattr(self, name)))
            if abs(norm - 1.0) > _QUATERNION_NORM_TOLERANCE:
                raise ContractError(f"trace {name} must be a unit quaternion")

        # observed_mujoco_root_displacement is real-or-unavailable: a genuine
        # width-3 finite measurement, or None when the state log does not bound
        # the released chunk.  It is never fabricated as a zero vector.
        observed = self.observed_mujoco_root_displacement
        if observed is not None:
            object.__setattr__(
                self,
                "observed_mujoco_root_displacement",
                _finite_vector(observed, 3, "observed_mujoco_root_displacement"),
            )
