"""Contact-anchored placement and composition for terrain contact actions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .torch_contact_oracle_actions import ContactPhaseAction
from .torch_contact_oracle_search import OracleState, PlacedContactPhase


def _rotate_xy(values: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            cosine * values[..., 0] - sine * values[..., 1],
            sine * values[..., 0] + cosine * values[..., 1],
        ),
        dim=-1,
    )


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(yaw)
    return torch.stack(
        (torch.cos(yaw / 2.0), zero, zero, torch.sin(yaw / 2.0))
    )


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


@dataclass(frozen=True)
class ContactAnchoredPlacement:
    placed: PlacedContactPhase
    yaw_world: torch.Tensor
    translation_world: torch.Tensor
    maximum_entry_support_error_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.placed, PlacedContactPhase):
            raise ValueError("contact-anchored placed action is invalid")
        reference = self.placed.root_position_world
        if (
            not isinstance(self.yaw_world, torch.Tensor)
            or self.yaw_world.shape != torch.Size([])
            or self.yaw_world.dtype != reference.dtype
            or self.yaw_world.device != reference.device
            or not torch.isfinite(self.yaw_world)
            or not isinstance(self.translation_world, torch.Tensor)
            or tuple(self.translation_world.shape) != (3,)
            or self.translation_world.dtype != reference.dtype
            or self.translation_world.device != reference.device
            or not torch.isfinite(self.translation_world).all()
            or isinstance(self.maximum_entry_support_error_m, bool)
            or not isinstance(self.maximum_entry_support_error_m, (int, float))
            or not math.isfinite(float(self.maximum_entry_support_error_m))
            or float(self.maximum_entry_support_error_m) < 0.0
        ):
            raise ValueError("contact-anchored placement metadata is invalid")
        object.__setattr__(self, "yaw_world", self.yaw_world.detach().clone())
        object.__setattr__(
            self, "translation_world", self.translation_world.detach().clone()
        )
        object.__setattr__(
            self,
            "maximum_entry_support_error_m",
            float(self.maximum_entry_support_error_m),
        )


def _fit_entry_support(
    action: ContactPhaseAction, state: OracleState
) -> tuple[torch.Tensor, torch.Tensor]:
    support_indices = torch.nonzero(state.support_mask, as_tuple=False).flatten()
    source = action.foot_position_local[0, support_indices]
    target = state.foot_position_world[support_indices]
    if support_indices.numel() == 1:
        yaw = state.root_yaw_world
    elif support_indices.numel() == 2:
        source_delta = source[1, :2] - source[0, :2]
        target_delta = target[1, :2] - target[0, :2]
        if (
            float(torch.linalg.vector_norm(source_delta).item()) <= 1e-6
            or float(torch.linalg.vector_norm(target_delta).item()) <= 1e-6
        ):
            raise ValueError("contact-anchored double support is degenerate")
        source_angle = torch.atan2(source_delta[1], source_delta[0])
        target_angle = torch.atan2(target_delta[1], target_delta[0])
        yaw = torch.atan2(
            torch.sin(target_angle - source_angle),
            torch.cos(target_angle - source_angle),
        )
    else:
        raise ValueError("contact-anchored placement requires entry support")
    rotated_xy = _rotate_xy(source[:, :2], yaw)
    translation_xy = torch.mean(target[:, :2] - rotated_xy, dim=0)
    translation_z = torch.mean(target[:, 2] - source[:, 2])
    return yaw, torch.cat((translation_xy, translation_z.reshape(1)))


def place_action_contact_anchored(
    action: ContactPhaseAction, state: OracleState
) -> ContactAnchoredPlacement:
    """Rigidly fit the complete action to current entry support contacts."""

    if not isinstance(action, ContactPhaseAction) or not isinstance(
        state, OracleState
    ):
        raise ValueError("contact-anchored placement inputs are invalid")
    if action.root_position_local.device != state.root_position_world.device:
        raise ValueError("contact-anchored placement devices do not match")
    if not bool(torch.equal(action.entry_support, state.support_mask)):
        raise ValueError("contact-anchored entry support must match exactly")
    yaw, translation = _fit_entry_support(action, state)
    root_xy = _rotate_xy(action.root_position_local[:, :2], yaw) + translation[:2]
    root_z = action.root_position_local[:, 2:] + translation[2]
    feet_xy = _rotate_xy(action.foot_position_local[..., :2], yaw) + translation[:2]
    feet_z = action.foot_position_local[..., 2:] + translation[2]
    root_yaw = action.root_yaw_local + yaw
    root_yaw = torch.atan2(torch.sin(root_yaw), torch.cos(root_yaw))
    orientation = _quaternion_multiply(
        _yaw_quaternion(yaw), action.root_orientation_local_wxyz
    )
    placed = PlacedContactPhase(
        action=action,
        root_position_world=torch.cat((root_xy, root_z), dim=1),
        root_yaw_world=root_yaw,
        root_orientation_world_wxyz=orientation,
        foot_position_world=torch.cat((feet_xy, feet_z), dim=2),
    )
    entry_error = torch.linalg.vector_norm(
        placed.foot_position_world[0, state.support_mask]
        - state.foot_position_world[state.support_mask],
        dim=1,
    )
    return ContactAnchoredPlacement(
        placed=placed,
        yaw_world=yaw,
        translation_world=translation,
        maximum_entry_support_error_m=float(entry_error.max().item()),
    )
