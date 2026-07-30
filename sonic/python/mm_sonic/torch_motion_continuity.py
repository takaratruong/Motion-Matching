"""Row-aligned full-pose continuity costs for Torch motion matching."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .joints import ContractError
from .torch_motion_data import MotionFolder
from .torch_motion_features import resolve_torch_device


@dataclass(frozen=True)
class TransitionContinuityCosts:
    position: torch.Tensor
    velocity: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class TransitionContinuityDatabase:
    device: torch.device
    _joint_position: torch.Tensor
    _joint_velocity: torch.Tensor
    _position_scale: torch.Tensor
    _velocity_scale: torch.Tensor
    _position_active: torch.Tensor
    _velocity_active: torch.Tensor

    @classmethod
    def from_folder(
        cls, folder: MotionFolder, device: str | torch.device
    ) -> "TransitionContinuityDatabase":
        resolved = resolve_torch_device(device)
        position = torch.cat(
            tuple(
                torch.tensor(
                    clip.joint_position[: clip.valid_frame_stop],
                    dtype=torch.float32,
                    device=resolved,
                )
                for clip in folder.clips
            ),
            dim=0,
        )
        velocity = torch.cat(
            tuple(
                torch.tensor(
                    clip.joint_velocity[: clip.valid_frame_stop],
                    dtype=torch.float32,
                    device=resolved,
                )
                for clip in folder.clips
            ),
            dim=0,
        )
        position_scale = torch.std(position, dim=0, correction=0)
        velocity_scale = torch.std(velocity, dim=0, correction=0)
        position_active = torch.isfinite(position_scale) & (position_scale > 0)
        velocity_active = torch.isfinite(velocity_scale) & (velocity_scale > 0)
        return cls(
            resolved,
            position,
            velocity,
            position_scale,
            velocity_scale,
            position_active,
            velocity_active,
        )

    def source_states_copy(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._joint_position.clone(), self._joint_velocity.clone()

    def scales_copy(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._position_scale.clone(),
            self._velocity_scale.clone(),
            self._position_active.clone(),
            self._velocity_active.clone(),
        )

    def costs(
        self,
        current_joint_position: torch.Tensor,
        current_joint_velocity: torch.Tensor,
        *,
        position_weight: float,
        velocity_weight: float,
    ) -> TransitionContinuityCosts:
        _validate_current_state(
            current_joint_position,
            "current_joint_position",
            self.device,
        )
        _validate_current_state(
            current_joint_velocity,
            "current_joint_velocity",
            self.device,
        )
        _validate_weight(position_weight, "position_weight")
        _validate_weight(velocity_weight, "velocity_weight")

        row_count = self._joint_position.shape[0]
        position = torch.zeros(
            row_count, dtype=torch.float32, device=self.device
        )
        if position_weight > 0:
            if not bool(torch.any(self._position_active).item()):
                raise ContractError(
                    "position continuity has no active components"
                )
            residual = (
                self._joint_position[:, self._position_active]
                - current_joint_position[self._position_active]
            ) / self._position_scale[self._position_active]
            position = position_weight * torch.mean(
                torch.square(residual), dim=1
            )

        velocity = torch.zeros(
            row_count, dtype=torch.float32, device=self.device
        )
        if velocity_weight > 0:
            if not bool(torch.any(self._velocity_active).item()):
                raise ContractError(
                    "velocity continuity has no active components"
                )
            residual = (
                self._joint_velocity[:, self._velocity_active]
                - current_joint_velocity[self._velocity_active]
            ) / self._velocity_scale[self._velocity_active]
            velocity = velocity_weight * torch.mean(
                torch.square(residual), dim=1
            )

        total = position + velocity
        for component in (position, velocity, total):
            valid = torch.isfinite(component) & (component >= 0)
            if not bool(torch.all(valid).item()):
                raise ContractError(
                    "continuity costs must be finite and non-negative"
                )

        return TransitionContinuityCosts(
            position=position,
            velocity=velocity,
            total=total,
        )


def _validate_current_state(
    value: torch.Tensor,
    name: str,
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise ContractError(f"{name} must be a Torch tensor")
    if tuple(value.shape) != (29,):
        raise ContractError(f"{name} must have shape (29,)")
    if value.dtype != torch.float32:
        raise ContractError(f"{name} must use float32")
    if value.device != device:
        raise ContractError(f"{name} must be on the database device")
    if not bool(torch.all(torch.isfinite(value)).item()):
        raise ContractError(f"{name} must be finite")


def _validate_weight(value: float, name: str) -> None:
    try:
        finite = math.isfinite(value)
        nonnegative = value >= 0
    except (TypeError, ValueError):
        raise ContractError(f"{name} must be finite and non-negative") from None
    if not finite or not nonnegative:
        raise ContractError(f"{name} must be finite and non-negative")
