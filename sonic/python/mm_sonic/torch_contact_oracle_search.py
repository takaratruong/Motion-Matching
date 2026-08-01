"""Rigid placement and planning for the offline terrain contact oracle."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from .joints import ContractError
from .torch_contact_oracle_actions import ContactPhaseAction
from .torch_contact_segments import ANKLE_ORIGIN_SOLE_M


def _owned(
    value: torch.Tensor,
    shape: tuple[int, ...],
    label: str,
    *,
    boolean: bool = False,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or (value.dtype != torch.bool if boolean else not value.dtype.is_floating_point)
        or (not boolean and not torch.isfinite(value).all())
    ):
        raise ContractError(f"contact oracle {label} is invalid")
    return value.detach().clone()


@dataclass(frozen=True)
class OracleState:
    root_position_world: torch.Tensor
    root_yaw_world: torch.Tensor
    foot_position_world: torch.Tensor
    support_mask: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    route_frame: int
    source_history: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        fields = (
            ("root_position_world", (3,), False),
            ("root_yaw_world", (), False),
            ("foot_position_world", (2, 3), False),
            ("support_mask", (2,), True),
            ("joint_position", (29,), False),
            ("joint_velocity", (29,), False),
        )
        owned = []
        device = None
        for name, shape, boolean in fields:
            value = _owned(
                getattr(self, name), shape, name.replace("_", " "), boolean=boolean
            )
            if device is not None and value.device != device:
                raise ContractError("contact oracle state devices do not match")
            device = value.device
            owned.append((name, value))
        if type(self.route_frame) is not int or self.route_frame < 0:
            raise ContractError("contact oracle route frame is invalid")
        history = self.source_history
        if (
            not isinstance(history, tuple)
            or any(
                not isinstance(key, tuple)
                or len(key) != 3
                or any(type(value) is not int or value < 0 for value in key)
                for key in history
            )
        ):
            raise ContractError("contact oracle source history is invalid")
        for name, value in owned:
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PlacedContactPhase:
    action: ContactPhaseAction
    root_position_world: torch.Tensor
    root_yaw_world: torch.Tensor
    foot_position_world: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.action, ContactPhaseAction):
            raise ContractError("placed contact oracle action is invalid")
        frames = self.action.frame_count
        values = (
            ("root_position_world", (frames, 3)),
            ("root_yaw_world", (frames,)),
            ("foot_position_world", (frames, 2, 3)),
        )
        owned = []
        for name, shape in values:
            value = _owned(getattr(self, name), shape, name.replace("_", " "))
            if value.device != self.action.root_position_local.device:
                raise ContractError("placed contact oracle devices do not match")
            owned.append((name, value))
        for name, value in owned:
            object.__setattr__(self, name, value)


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


def place_action(
    action: ContactPhaseAction, state: OracleState
) -> PlacedContactPhase:
    """Rigidly anchor a source contact phase to the state's root SE(2)."""

    if not isinstance(action, ContactPhaseAction) or not isinstance(
        state, OracleState
    ):
        raise ContractError("contact oracle placement inputs are invalid")
    if action.root_position_local.device != state.root_position_world.device:
        raise ContractError("contact oracle placement devices do not match")
    root_xy = (
        _rotate_xy(action.root_position_local[:, :2], state.root_yaw_world)
        + state.root_position_world[:2]
    )
    root_z = (
        action.root_position_local[:, 2:]
        + state.root_position_world[2]
    )
    feet_xy = (
        _rotate_xy(action.foot_position_local[..., :2], state.root_yaw_world)
        + state.root_position_world[:2]
    )
    feet_z = (
        action.foot_position_local[..., 2:]
        + state.root_position_world[2]
    )
    yaw = action.root_yaw_local + state.root_yaw_world
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    return PlacedContactPhase(
        action=action,
        root_position_world=torch.cat((root_xy, root_z), dim=1),
        root_yaw_world=yaw,
        foot_position_world=torch.cat((feet_xy, feet_z), dim=2),
    )


@dataclass(frozen=True)
class OracleConstraints:
    stance_height_tolerance_m: float = 0.05
    landing_height_tolerance_m: float = 0.03
    minimum_swing_clearance_m: float = -0.03
    maximum_height_deformation_m: float = 0.06
    edge_margin_m: float = 0.04
    maximum_edge_height_range_m: float = 0.025
    maximum_entry_foot_error_m: float = 0.08
    maximum_joint_position_error_rad: float = 1.50
    maximum_joint_velocity_error_rad_s: float = 8.0

    def __post_init__(self) -> None:
        positive = (
            self.stance_height_tolerance_m,
            self.landing_height_tolerance_m,
            self.maximum_height_deformation_m,
            self.edge_margin_m,
            self.maximum_edge_height_range_m,
            self.maximum_entry_foot_error_m,
            self.maximum_joint_position_error_rad,
            self.maximum_joint_velocity_error_rad_s,
        )
        values = (*positive, self.minimum_swing_clearance_m)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ) or any(float(value) <= 0.0 for value in positive):
            raise ContractError("contact oracle constraints are invalid")


@dataclass(frozen=True)
class FeasibilityResult:
    accepted: bool
    reason: str | None
    stance_error_m: float
    landing_error_m: float
    minimum_swing_clearance_m: float

    def __post_init__(self) -> None:
        if (
            type(self.accepted) is not bool
            or (self.reason is not None and (not isinstance(self.reason, str) or not self.reason))
            or self.accepted != (self.reason is None)
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (
                    self.stance_error_m,
                    self.landing_error_m,
                    self.minimum_swing_clearance_m,
                )
            )
        ):
            raise ContractError("contact oracle feasibility result is invalid")


def _sample_surface(
    sample_surface: Callable[[torch.Tensor], torch.Tensor], points: torch.Tensor
) -> torch.Tensor:
    try:
        values = sample_surface(points)
    except ContractError:
        raise
    except Exception as error:
        raise ContractError("contact oracle terrain sampling failed") from error
    if (
        not isinstance(values, torch.Tensor)
        or values.shape != points.shape[:-1]
        or values.device != points.device
        or values.dtype != points.dtype
        or not torch.isfinite(values).all()
    ):
        raise ContractError("contact oracle terrain sampler returned invalid heights")
    return values


def _result(
    reason: str | None,
    *,
    stance: float = 0.0,
    landing: float = 0.0,
    swing: float = 0.0,
) -> FeasibilityResult:
    return FeasibilityResult(
        accepted=reason is None,
        reason=reason,
        stance_error_m=float(stance),
        landing_error_m=float(landing),
        minimum_swing_clearance_m=float(swing),
    )


def validate_placement(
    *,
    placed: PlacedContactPhase,
    state: OracleState,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    constraints: OracleConstraints,
) -> FeasibilityResult:
    """Apply deterministic hard contact constraints to a rigid action."""

    if (
        not isinstance(placed, PlacedContactPhase)
        or not isinstance(state, OracleState)
        or not callable(sample_surface)
        or not isinstance(constraints, OracleConstraints)
    ):
        raise ContractError("contact oracle feasibility inputs are invalid")
    action = placed.action
    if bool((state.support_mask & ~action.entry_support).any().item()):
        return _result("support-order")
    entry_error = torch.linalg.vector_norm(
        placed.foot_position_world[0] - state.foot_position_world, dim=1
    )
    if bool(
        (
            entry_error[state.support_mask]
            > float(constraints.maximum_entry_foot_error_m)
        ).any().item()
    ):
        return _result("entry-foot-error")
    position_error = float(
        torch.linalg.vector_norm(action.joint_position[0] - state.joint_position).item()
    )
    if position_error > float(constraints.maximum_joint_position_error_rad):
        return _result("joint-position")
    velocity_error = float(
        torch.linalg.vector_norm(action.joint_velocity[0] - state.joint_velocity).item()
    )
    if velocity_error > float(constraints.maximum_joint_velocity_error_rad_s):
        return _result("joint-velocity")

    surface = _sample_surface(
        sample_surface, placed.foot_position_world[..., :2]
    )
    sole_clearance = (
        placed.foot_position_world[..., 2]
        - surface
        - float(ANKLE_ORIGIN_SOLE_M)
    )
    swing_foot = action.swing_foot
    stance_support = action.support_mask.clone()
    stance_support[:, swing_foot] = False
    stance_error = float(
        torch.abs(sole_clearance[stance_support]).max().item()
        if bool(stance_support.any().item())
        else 0.0
    )
    if stance_error > float(constraints.stance_height_tolerance_m):
        return _result("stance-height", stance=stance_error)

    landing_error = abs(float(sole_clearance[-1, swing_foot].item()))
    if landing_error > float(constraints.landing_height_tolerance_m):
        return _result(
            "landing-height", stance=stance_error, landing=landing_error
        )

    margin = float(constraints.edge_margin_m)
    offsets = torch.tensor(
        ((0.0, 0.0), (margin, 0.0), (-margin, 0.0), (0.0, margin), (0.0, -margin)),
        dtype=placed.foot_position_world.dtype,
        device=placed.foot_position_world.device,
    )
    landing_xy = placed.foot_position_world[-1, swing_foot, :2]
    edge_heights = _sample_surface(sample_surface, landing_xy[None, :] + offsets)
    if float((edge_heights.max() - edge_heights.min()).item()) > float(
        constraints.maximum_edge_height_range_m
    ):
        return _result(
            "landing-edge-margin", stance=stance_error, landing=landing_error
        )

    swing_samples = sole_clearance[:, swing_foot][
        ~action.support_mask[:, swing_foot]
    ]
    minimum_swing = float(
        swing_samples.min().item() if swing_samples.numel() else landing_error
    )
    if minimum_swing < float(constraints.minimum_swing_clearance_m):
        return _result(
            "swing-penetration",
            stance=stance_error,
            landing=landing_error,
            swing=minimum_swing,
        )

    query_delta = surface[-1, swing_foot] - surface[0, swing_foot]
    source_delta = action.foot_surface_delta_m[-1, swing_foot]
    if abs(float((query_delta - source_delta).item())) > float(
        constraints.maximum_height_deformation_m
    ):
        return _result(
            "height-deformation",
            stance=stance_error,
            landing=landing_error,
            swing=minimum_swing,
        )
    return _result(
        None,
        stance=stance_error,
        landing=landing_error,
        swing=minimum_swing,
    )
