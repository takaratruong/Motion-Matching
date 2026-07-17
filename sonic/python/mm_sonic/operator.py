"""Pure boundary-latched commands for the non-scored SONIC operator."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

from .commands import CommandSample
from .joints import ContractError


@dataclass(frozen=True)
class OperatorState:
    """One complete keyboard/gamepad state queued for the next chunk boundary."""

    forward: bool = False
    backward: bool = False
    left: bool = False
    right: bool = False
    heading_left: bool = False
    heading_right: bool = False
    stand: bool = False
    terminate: bool = False

    def __post_init__(self) -> None:
        if any(type(getattr(self, field.name)) is not bool for field in fields(self)):
            raise ContractError("operator state fields must be boolean")


@dataclass(frozen=True)
class OperatorLimits:
    """Positive command limits for manual demonstrations."""

    forward_mps: float = 1.0
    backward_mps: float = 0.5
    lateral_mps: float = 0.5
    heading_step_rad: float = math.radians(5.0)

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ContractError(
                    f"operator limit {field.name} must be positive and finite"
                )
            object.__setattr__(self, field.name, float(value))


class OperatorSampler:
    """Latch only the newest state and convert it at a 0.4-second boundary."""

    def __init__(self, limits: OperatorLimits | None = None) -> None:
        if limits is None:
            limits = OperatorLimits()
        if type(limits) is not OperatorLimits:
            raise ContractError("operator sampler limits must be OperatorLimits")
        self._limits = limits
        self._pending_state = OperatorState()
        self._desired_yaw = 0.0
        self._terminated = False

    @property
    def limits(self) -> OperatorLimits:
        return self._limits

    def update(self, state: OperatorState) -> None:
        if type(state) is not OperatorState:
            raise ContractError("operator update must be an OperatorState")
        self._pending_state = state

    def sample_boundary(self, chunk_index: int) -> CommandSample | None:
        if type(chunk_index) is not int or chunk_index < 0:
            raise ContractError("operator chunk_index must be a nonnegative integer")
        if self._terminated:
            return None

        state = self._pending_state
        self._pending_state = OperatorState()
        if state.terminate:
            self._terminated = True
            return None

        heading_direction = int(state.heading_left) - int(state.heading_right)
        if heading_direction:
            self._desired_yaw = math.remainder(
                self._desired_yaw
                + heading_direction * self._limits.heading_step_rad,
                2.0 * math.pi,
            )

        velocity = self._velocity(state)
        half_yaw = 0.5 * self._desired_yaw
        return CommandSample(
            chunk_index=chunk_index,
            requested_velocity_mujoco=velocity,
            desired_heading_mujoco_wxyz=(
                math.cos(half_yaw),
                0.0,
                0.0,
                math.sin(half_yaw),
            ),
        )

    def _velocity(self, state: OperatorState) -> tuple[float, float, float]:
        if state.stand:
            return (0.0, 0.0, 0.0)
        longitudinal = int(state.forward) - int(state.backward)
        lateral = int(state.left) - int(state.right)
        if longitudinal == 0 and lateral == 0:
            return (0.0, 0.0, 0.0)

        norm = math.hypot(longitudinal, lateral)
        unit_x = longitudinal / norm
        unit_y = lateral / norm
        if longitudinal > 0:
            speed = self._limits.forward_mps
        elif longitudinal < 0:
            speed = self._limits.backward_mps
        else:
            speed = self._limits.lateral_mps
        x = unit_x * speed
        y = unit_y * speed
        if abs(y) > self._limits.lateral_mps:
            scale = self._limits.lateral_mps / abs(y)
            x *= scale
            y *= scale
        return (
            0.0 if x == 0.0 else x,
            0.0 if y == 0.0 else y,
            0.0,
        )
