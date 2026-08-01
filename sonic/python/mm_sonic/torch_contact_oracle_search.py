"""Rigid placement and planning for the offline terrain contact oracle."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Callable
from typing import Mapping, Sequence

import torch

from .joints import ContractError
from .torch_contact_oracle_actions import ContactPhaseAction
from .torch_contact_segments import ANKLE_ORIGIN_SOLE_M
from .torch_foothold_actions import FootholdPlan, plan_footholds


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
    root_orientation_world_wxyz: torch.Tensor
    foot_position_world: torch.Tensor
    support_mask: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    route_frame: int
    source_history: tuple[tuple[int, int, int, bool], ...] = ()

    def __post_init__(self) -> None:
        fields = (
            ("root_position_world", (3,), False),
            ("root_yaw_world", (), False),
            ("root_orientation_world_wxyz", (4,), False),
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
                or len(key) != 4
                or any(
                    type(value) is not int or value < 0
                    for value in key[:3]
                )
                or type(key[3]) is not bool
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
    root_orientation_world_wxyz: torch.Tensor
    foot_position_world: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.action, ContactPhaseAction):
            raise ContractError("placed contact oracle action is invalid")
        frames = self.action.frame_count
        values = (
            ("root_position_world", (frames, 3)),
            ("root_yaw_world", (frames,)),
            ("root_orientation_world_wxyz", (frames, 4)),
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


def _quaternion_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(yaw)
    return torch.stack(
        (torch.cos(yaw / 2.0), zero, zero, torch.sin(yaw / 2.0)), dim=-1
    )


def _quaternion_multiply_wxyz(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    w1, x1, y1, z1 = left.unbind(dim=-1)
    w2, x2, y2, z2 = right.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
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
    orientation = _quaternion_multiply_wxyz(
        _quaternion_from_yaw(state.root_yaw_world),
        action.root_orientation_local_wxyz,
    )
    return PlacedContactPhase(
        action=action,
        root_position_world=torch.cat((root_xy, root_z), dim=1),
        root_yaw_world=yaw,
        root_orientation_world_wxyz=orientation,
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
    stance_support[-1, swing_foot] = False
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

    swing_samples = sole_clearance[~action.support_mask]
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


@dataclass(frozen=True)
class CommandSchedule:
    velocity_world_xy: torch.Tensor
    heading_world_yaw: torch.Tensor
    dt: float = 0.02
    origin_world_xy: torch.Tensor | None = None

    def __post_init__(self) -> None:
        velocity = self.velocity_world_xy
        heading = self.heading_world_yaw
        origin = self.origin_world_xy
        if (
            not isinstance(velocity, torch.Tensor)
            or velocity.ndim != 2
            or velocity.shape[1] != 2
            or not velocity.dtype.is_floating_point
            or not torch.isfinite(velocity).all()
            or not isinstance(heading, torch.Tensor)
            or heading.shape != velocity.shape[:1]
            or heading.dtype != velocity.dtype
            or heading.device != velocity.device
            or not torch.isfinite(heading).all()
            or velocity.shape[0] < 1
            or isinstance(self.dt, bool)
            or not isinstance(self.dt, (int, float))
            or not math.isfinite(float(self.dt))
            or float(self.dt) <= 0.0
            or (
                origin is not None
                and (
                    not isinstance(origin, torch.Tensor)
                    or tuple(origin.shape) != (2,)
                    or origin.dtype != velocity.dtype
                    or origin.device != velocity.device
                    or not torch.isfinite(origin).all()
                )
            )
        ):
            raise ContractError("contact oracle command schedule is invalid")
        object.__setattr__(self, "velocity_world_xy", velocity.detach().clone())
        object.__setattr__(self, "heading_world_yaw", heading.detach().clone())
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(
            self,
            "origin_world_xy",
            None if origin is None else origin.detach().clone(),
        )

    @property
    def frame_count(self) -> int:
        return int(self.heading_world_yaw.shape[0])

    def displacement(self, start_frame: int, steps: int) -> torch.Tensor:
        if type(start_frame) is not int or start_frame < 0 or type(steps) is not int or steps < 0:
            raise ContractError("contact oracle command interval is invalid")
        if steps == 0:
            return torch.zeros(
                2,
                dtype=self.velocity_world_xy.dtype,
                device=self.velocity_world_xy.device,
            )
        indices = torch.arange(
            start_frame,
            start_frame + steps,
            dtype=torch.int64,
            device=self.velocity_world_xy.device,
        ).clamp(max=self.frame_count - 1)
        return self.velocity_world_xy[indices].sum(dim=0) * self.dt

    def displacements(
        self, start_frame: int, steps: torch.Tensor
    ) -> torch.Tensor:
        if (
            type(start_frame) is not int
            or start_frame < 0
            or not isinstance(steps, torch.Tensor)
            or steps.ndim != 1
            or steps.dtype not in (torch.int32, torch.int64)
            or steps.device != self.velocity_world_xy.device
            or bool((steps < 0).any().item())
        ):
            raise ContractError("contact oracle command intervals are invalid")
        prefix = torch.cat(
            (
                torch.zeros(
                    (1, 2),
                    dtype=self.velocity_world_xy.dtype,
                    device=self.velocity_world_xy.device,
                ),
                torch.cumsum(self.velocity_world_xy, dim=0),
            ),
            dim=0,
        )
        frame_count = self.frame_count
        start = min(start_frame, frame_count)
        end = steps + start_frame
        clamped_end = end.clamp(max=frame_count)
        base = prefix[clamped_end] - prefix[start]
        prior_tail = max(0, start_frame - frame_count)
        tail = (end - frame_count).clamp(min=0) - prior_tail
        return (
            base + tail[:, None] * self.velocity_world_xy[-1]
        ) * self.dt

    def heading(self, frame: int) -> torch.Tensor:
        if type(frame) is not int or frame < 0:
            raise ContractError("contact oracle command frame is invalid")
        return self.heading_world_yaw[min(frame, self.frame_count - 1)]

    def target_positions(self, frames: torch.Tensor) -> torch.Tensor:
        if self.origin_world_xy is None:
            raise ContractError("contact oracle command path has no origin")
        return self.origin_world_xy[None, :] + self.displacements(0, frames)

    def target_position(self, frame: int) -> torch.Tensor:
        if type(frame) is not int or frame < 0:
            raise ContractError("contact oracle command target frame is invalid")
        target = self.target_positions(
            torch.tensor(
                (frame,),
                dtype=torch.int64,
                device=self.velocity_world_xy.device,
            )
        )
        return target[0]

    def headings(
        self, start_frame: int, steps: torch.Tensor
    ) -> torch.Tensor:
        if (
            type(start_frame) is not int
            or start_frame < 0
            or not isinstance(steps, torch.Tensor)
            or steps.ndim != 1
            or steps.dtype not in (torch.int32, torch.int64)
            or steps.device != self.heading_world_yaw.device
            or bool((steps < 0).any().item())
        ):
            raise ContractError("contact oracle command headings are invalid")
        indices = (steps + start_frame).clamp(max=self.frame_count - 1)
        return self.heading_world_yaw[indices]

    def velocity(self, frame: int) -> torch.Tensor:
        if type(frame) is not int or frame < 0:
            raise ContractError("contact oracle command frame is invalid")
        return self.velocity_world_xy[min(frame, self.frame_count - 1)]


def constant_command_schedule(
    *,
    velocity_world_xy: tuple[float, float],
    heading_world_yaw: float,
    frames: int,
    device: torch.device | str,
) -> CommandSchedule:
    try:
        resolved = torch.device(device)
        velocity = torch.tensor(
            velocity_world_xy, dtype=torch.float32, device=resolved
        )
        heading = float(heading_world_yaw)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ContractError("contact oracle constant command is invalid") from error
    if (
        tuple(velocity.shape) != (2,)
        or not torch.isfinite(velocity).all()
        or not math.isfinite(heading)
        or type(frames) is not int
        or frames < 1
    ):
        raise ContractError("contact oracle constant command is invalid")
    return CommandSchedule(
        velocity_world_xy=velocity[None, :].repeat(frames, 1),
        heading_world_yaw=torch.full(
            (frames,), heading, dtype=torch.float32, device=resolved
        ),
    )


@dataclass(frozen=True)
class OracleCost:
    path: float = 0.0
    facing: float = 0.0
    foothold: float = 0.0
    timing: float = 0.0
    joint_position: float = 0.0
    joint_velocity: float = 0.0
    stance_motion: float = 0.0
    clearance_margin: float = 0.0
    reachability: float = 0.0
    repetition: float = 0.0

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in self.components()
        ):
            raise ContractError("contact oracle cost is invalid")

    def components(self) -> tuple[float, ...]:
        return (
            float(self.path),
            float(self.facing),
            float(self.foothold),
            float(self.timing),
            float(self.joint_position),
            float(self.joint_velocity),
            float(self.stance_motion),
            float(self.clearance_margin),
            float(self.reachability),
            float(self.repetition),
        )

    @property
    def total(self) -> float:
        return sum(self.components())

    def __add__(self, other: object) -> "OracleCost":
        if not isinstance(other, OracleCost):
            return NotImplemented
        return OracleCost(
            *(left + right for left, right in zip(self.components(), other.components()))
        )


@dataclass(frozen=True)
class OracleSearchConfig:
    horizon_landings: int = 4
    beam_width: int = 32
    foothold_beam_width: int = 50
    transition_candidate_count: int = 32
    constraints: OracleConstraints = OracleConstraints()
    path_weight: float = 25.0
    facing_weight: float = 10.0
    foothold_weight: float = 50.0
    timing_weight: float = 0.10
    joint_position_weight: float = 0.50
    joint_velocity_weight: float = 0.05
    stance_motion_weight: float = 100.0
    clearance_margin_weight: float = 1.0
    reachability_weight: float = 10.0
    repetition_weight: float = 2.0

    def __post_init__(self) -> None:
        if (
            type(self.horizon_landings) is not int
            or self.horizon_landings < 1
            or type(self.beam_width) is not int
            or self.beam_width < 1
            or type(self.foothold_beam_width) is not int
            or self.foothold_beam_width < 1
            or type(self.transition_candidate_count) is not int
            or self.transition_candidate_count < 1
            or not isinstance(self.constraints, OracleConstraints)
        ):
            raise ContractError("contact oracle search dimensions are invalid")
        weights = (
            self.path_weight,
            self.facing_weight,
            self.foothold_weight,
            self.timing_weight,
            self.joint_position_weight,
            self.joint_velocity_weight,
            self.stance_motion_weight,
            self.clearance_margin_weight,
            self.reachability_weight,
            self.repetition_weight,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in weights
        ):
            raise ContractError("contact oracle search weights are invalid")


@dataclass(frozen=True)
class ContactOracleExperimentConfig:
    terrain_config: Path
    search: OracleSearchConfig

    def __post_init__(self) -> None:
        path = Path(self.terrain_config).resolve()
        if not path.is_file() or not isinstance(self.search, OracleSearchConfig):
            raise ContractError("contact oracle experiment config is invalid")
        object.__setattr__(self, "terrain_config", path)


def load_contact_oracle_config(
    path: str | Path,
) -> ContactOracleExperimentConfig:
    """Load the strict frozen contact-oracle experiment schema."""

    target = Path(path).resolve()
    try:
        raw = json.loads(target.read_text())
    except Exception as error:
        raise ContractError(f"cannot load contact oracle config: {target}") from error
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "terrain_config",
        "search",
        "constraints",
        "weights",
    }:
        raise ContractError("contact oracle config keys are invalid")
    if raw["schema"] != "g1-contact-space-oracle/v1":
        raise ContractError("contact oracle config schema is invalid")
    search = raw["search"]
    constraints = raw["constraints"]
    weights = raw["weights"]
    search_keys = {
        "horizon_landings",
        "beam_width",
        "foothold_beam_width",
        "transition_candidate_count",
    }
    constraint_keys = set(OracleConstraints.__dataclass_fields__)
    weight_keys = {
        "path_weight",
        "facing_weight",
        "foothold_weight",
        "timing_weight",
        "joint_position_weight",
        "joint_velocity_weight",
        "stance_motion_weight",
        "clearance_margin_weight",
        "reachability_weight",
        "repetition_weight",
    }
    if (
        not isinstance(search, dict)
        or set(search) != search_keys
        or not isinstance(constraints, dict)
        or set(constraints) != constraint_keys
        or not isinstance(weights, dict)
        or set(weights) != weight_keys
    ):
        raise ContractError("contact oracle nested config keys are invalid")
    terrain_value = raw["terrain_config"]
    if not isinstance(terrain_value, str) or not terrain_value:
        raise ContractError("contact oracle terrain config path is invalid")
    terrain_path = Path(terrain_value)
    if not terrain_path.is_absolute():
        try:
            repository_root = target.parents[3]
        except IndexError as error:
            raise ContractError("contact oracle config location is invalid") from error
        terrain_path = repository_root / terrain_path
    try:
        resolved_constraints = OracleConstraints(**constraints)
        resolved_search = OracleSearchConfig(
            constraints=resolved_constraints,
            **search,
            **weights,
        )
    except TypeError as error:
        raise ContractError("contact oracle config values are invalid") from error
    return ContactOracleExperimentConfig(
        terrain_config=terrain_path,
        search=resolved_search,
    )


@dataclass(frozen=True)
class OracleExpansionDiagnostics:
    expanded_candidate_count: int
    rejected_by_reason: Mapping[str, int]
    beam_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        rejected = dict(self.rejected_by_reason)
        if (
            type(self.expanded_candidate_count) is not int
            or self.expanded_candidate_count < 0
            or not isinstance(self.beam_sizes, tuple)
            or any(type(value) is not int or value < 0 for value in self.beam_sizes)
            or any(
                not isinstance(reason, str)
                or not reason
                or type(count) is not int
                or count < 1
                for reason, count in rejected.items()
            )
        ):
            raise ContractError("contact oracle expansion diagnostics are invalid")
        object.__setattr__(self, "rejected_by_reason", MappingProxyType(rejected))


@dataclass(frozen=True)
class OraclePlan:
    action_indices: tuple[int, ...]
    placements: tuple[PlacedContactPhase, ...]
    step_costs: tuple[OracleCost, ...]
    total_cost: OracleCost
    horizon_landings: int
    expansion: OracleExpansionDiagnostics

    def __post_init__(self) -> None:
        count = len(self.action_indices)
        if (
            not isinstance(self.action_indices, tuple)
            or any(type(value) is not int or value < 0 for value in self.action_indices)
            or len(self.placements) != count
            or len(self.step_costs) != count
            or any(not isinstance(value, PlacedContactPhase) for value in self.placements)
            or any(not isinstance(value, OracleCost) for value in self.step_costs)
            or not isinstance(self.total_cost, OracleCost)
            or type(self.horizon_landings) is not int
            or self.horizon_landings != count
            or not isinstance(self.expansion, OracleExpansionDiagnostics)
        ):
            raise ContractError("contact oracle plan is invalid")


class OracleSearchFailure(ContractError):
    def __init__(self, rejected_by_reason: Mapping[str, int]):
        self.rejected_by_reason = MappingProxyType(dict(rejected_by_reason))
        super().__init__("no feasible one-contact oracle action")


@dataclass(frozen=True)
class _SearchNode:
    state: OracleState
    action_indices: tuple[int, ...]
    placements: tuple[PlacedContactPhase, ...]
    step_costs: tuple[OracleCost, ...]
    total_cost: OracleCost
    cached_feasible: tuple[
        tuple[int, PlacedContactPhase, FeasibilityResult], ...
    ] | None = None
    cached_rejected: Mapping[str, int] | None = None

    @property
    def source_key_path(self) -> tuple[tuple[int, int, int, bool], ...]:
        return tuple(placement.action.source_key for placement in self.placements)


@dataclass(frozen=True)
class _ActionEntryCache:
    support: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    foot_position_local: torch.Tensor
    root_displacement_local: torch.Tensor
    terminal_yaw_local: torch.Tensor
    frame_steps: torch.Tensor
    foot_trajectory_local: torch.Tensor
    support_trajectory: torch.Tensor
    valid_frames: torch.Tensor
    surface_delta_m: torch.Tensor
    swing_foot: torch.Tensor

    @classmethod
    def from_actions(
        cls, actions: Sequence[ContactPhaseAction]
    ) -> "_ActionEntryCache":
        device = actions[0].root_position_local.device
        dtype = actions[0].root_position_local.dtype
        count = len(actions)
        frames = max(action.frame_count for action in actions)
        foot_trajectory = torch.zeros(
            (count, frames, 2, 3), dtype=dtype, device=device
        )
        support_trajectory = torch.zeros(
            (count, frames, 2), dtype=torch.bool, device=device
        )
        valid_frames = torch.zeros(
            (count, frames), dtype=torch.bool, device=device
        )
        surface_delta = torch.zeros(
            (count, frames, 2), dtype=dtype, device=device
        )
        for index, action in enumerate(actions):
            length = action.frame_count
            foot_trajectory[index, :length] = action.foot_position_local
            support_trajectory[index, :length] = action.support_mask
            valid_frames[index, :length] = True
            surface_delta[index, :length] = action.foot_surface_delta_m
        return cls(
            support=torch.stack([action.entry_support for action in actions]),
            joint_position=torch.stack(
                [action.joint_position[0] for action in actions]
            ),
            joint_velocity=torch.stack(
                [action.joint_velocity[0] for action in actions]
            ),
            foot_position_local=torch.stack(
                [action.foot_position_local[0] for action in actions]
            ),
            root_displacement_local=torch.stack(
                [action.root_position_local[-1, :2] for action in actions]
            ),
            terminal_yaw_local=torch.stack(
                [action.root_yaw_local[-1] for action in actions]
            ),
            frame_steps=torch.tensor(
                [action.frame_count - 1 for action in actions],
                dtype=torch.int64,
                device=device,
            ),
            foot_trajectory_local=foot_trajectory,
            support_trajectory=support_trajectory,
            valid_frames=valid_frames,
            surface_delta_m=surface_delta,
            swing_foot=torch.tensor(
                [action.swing_foot for action in actions],
                dtype=torch.int64,
                device=device,
            ),
        )

    def candidate_indices(
        self,
        state: OracleState,
        constraints: OracleConstraints,
        rejected: Counter[str] | None,
    ) -> torch.Tensor:
        eligible = torch.ones(
            self.support.shape[0],
            dtype=torch.bool,
            device=self.support.device,
        )
        support = ~(
            state.support_mask[None, :] & ~self.support
        ).any(dim=1)
        if rejected is not None:
            count = int((eligible & ~support).sum().item())
            if count:
                rejected["support-order"] += count
        eligible &= support
        foot_xy = (
            _rotate_xy(
                self.foot_position_local[..., :2], state.root_yaw_world
            )
            + state.root_position_world[None, None, :2]
        )
        foot_z = (
            self.foot_position_local[..., 2]
            + state.root_position_world[2]
        )
        foot_world = torch.cat((foot_xy, foot_z[..., None]), dim=2)
        foot_error = torch.linalg.vector_norm(
            foot_world - state.foot_position_world[None, :, :], dim=2
        )
        entry = ~(
            foot_error[:, state.support_mask]
            > float(constraints.maximum_entry_foot_error_m)
        ).any(dim=1)
        if rejected is not None:
            count = int((eligible & ~entry).sum().item())
            if count:
                rejected["entry-foot-error"] += count
        eligible &= entry
        position = torch.linalg.vector_norm(
            self.joint_position - state.joint_position[None, :], dim=1
        ) <= float(constraints.maximum_joint_position_error_rad)
        if rejected is not None:
            count = int((eligible & ~position).sum().item())
            if count:
                rejected["joint-position"] += count
        eligible &= position
        velocity = torch.linalg.vector_norm(
            self.joint_velocity - state.joint_velocity[None, :], dim=1
        ) <= float(constraints.maximum_joint_velocity_error_rad_s)
        if rejected is not None:
            count = int((eligible & ~velocity).sum().item())
            if count:
                rejected["joint-velocity"] += count
        eligible &= velocity
        return torch.nonzero(eligible, as_tuple=False).flatten()

    def shortlisted_indices(
        self,
        candidates: torch.Tensor,
        state: OracleState,
        schedule: CommandSchedule,
        foothold_plan: FootholdPlan,
        config: OracleSearchConfig,
        rejected: Counter[str] | None,
    ) -> torch.Tensor:
        limit = int(config.transition_candidate_count)
        if candidates.numel() <= limit:
            return candidates
        steps = self.frame_steps[candidates]
        actual = _rotate_xy(
            self.root_displacement_local[candidates], state.root_yaw_world
        )
        if schedule.origin_world_xy is None:
            desired = schedule.displacements(state.route_frame, steps)
        else:
            desired = schedule.target_positions(steps + state.route_frame)
            actual = actual + state.root_position_world[None, :2]
        path = float(config.path_weight) * torch.sum(
            torch.square(actual - desired), dim=1
        )
        desired_heading = schedule.headings(state.route_frame, steps)
        terminal_heading = (
            self.terminal_yaw_local[candidates] + state.root_yaw_world
        )
        heading_error = torch.atan2(
            torch.sin(terminal_heading - desired_heading),
            torch.cos(terminal_heading - desired_heading),
        )
        facing = float(config.facing_weight) * torch.square(heading_error)
        joint_position = float(config.joint_position_weight) * torch.sum(
            torch.square(
                self.joint_position[candidates]
                - state.joint_position[None, :]
            ),
            dim=1,
        )
        joint_velocity = float(config.joint_velocity_weight) * torch.sum(
            torch.square(
                self.joint_velocity[candidates]
                - state.joint_velocity[None, :]
            ),
            dim=1,
        )
        entry_foot_xy = (
            _rotate_xy(
                self.foot_position_local[candidates, :, :2],
                state.root_yaw_world,
            )
            + state.root_position_world[None, None, :2]
        )
        entry_foot_z = (
            self.foot_position_local[candidates, :, 2]
            + state.root_position_world[2]
        )
        entry_feet = torch.cat(
            (entry_foot_xy, entry_foot_z[..., None]), dim=2
        )
        entry_contact = float(config.stance_motion_weight) * torch.mean(
            torch.square(
                entry_feet[:, state.support_mask]
                - state.foot_position_world[None, state.support_mask]
            ),
            dim=(1, 2),
        )
        terminal = self.frame_steps[candidates]
        swing = self.swing_foot[candidates]
        rows = torch.arange(
            candidates.numel(), dtype=torch.int64, device=candidates.device
        )
        landing_local = self.foot_trajectory_local[candidates][
            rows, terminal, swing, :2
        ]
        landing_world = (
            _rotate_xy(landing_local, state.root_yaw_world)
            + state.root_position_world[None, :2]
        )
        target_xy = foothold_plan.landing_xy_world_m[:, 0]
        target_feet = foothold_plan.landing_feet[:, 0]
        target_frames = foothold_plan.landing_frame_offsets[:, 0]
        if target_xy.shape[0]:
            matching = swing[:, None] == target_feet[None, :]
            landing_error = torch.sum(
                torch.square(
                    landing_world[:, None, :] - target_xy[None, :, :]
                ),
                dim=2,
            )
            timing_error = torch.square(
                terminal[:, None].to(torch.float32)
                - target_frames[None, :].to(torch.float32)
            )
            target_cost = (
                float(config.foothold_weight) * landing_error
                + float(config.timing_weight) * timing_error
            )
            target_cost = torch.where(
                matching,
                target_cost,
                torch.full_like(target_cost, torch.inf),
            ).amin(dim=1)
            target_cost = torch.where(
                torch.isfinite(target_cost),
                target_cost,
                torch.full_like(target_cost, 100.0),
            )
        else:
            target_cost = torch.zeros_like(path)
        score = (
            path
            + facing
            + joint_position
            + joint_velocity
            + entry_contact
            + target_cost
        )
        order = torch.argsort(score, stable=True)
        if rejected is not None:
            rejected["shortlist-pruned"] += int(candidates.numel()) - limit
        return candidates[order[:limit]]

    def terrain_feasibility(
        self,
        candidates: torch.Tensor,
        state: OracleState,
        sample_surface: Callable[[torch.Tensor], torch.Tensor],
        constraints: OracleConstraints,
    ) -> tuple[FeasibilityResult, ...]:
        if not candidates.numel():
            return ()
        local = self.foot_trajectory_local[candidates]
        foot_xy = (
            _rotate_xy(local[..., :2], state.root_yaw_world)
            + state.root_position_world[None, None, None, :2]
        )
        foot_z = local[..., 2] + state.root_position_world[2]
        foot_world = torch.cat((foot_xy, foot_z[..., None]), dim=3)
        surface = _sample_surface(sample_surface, foot_world[..., :2])
        clearance = (
            foot_world[..., 2] - surface - float(ANKLE_ORIGIN_SOLE_M)
        )
        support = self.support_trajectory[candidates]
        valid = self.valid_frames[candidates]
        swing = self.swing_foot[candidates]
        rows = torch.arange(
            candidates.numel(), dtype=torch.int64, device=candidates.device
        )
        terminal = self.frame_steps[candidates]
        stance = support & valid[..., None]
        stance[rows, terminal, swing] = False
        stance_error = torch.where(
            stance, torch.abs(clearance), torch.zeros_like(clearance)
        ).amax(dim=(1, 2))
        landing_clearance = clearance[rows, terminal, swing]
        landing_error = torch.abs(landing_clearance)

        margin = float(constraints.edge_margin_m)
        offsets = torch.tensor(
            (
                (0.0, 0.0),
                (margin, 0.0),
                (-margin, 0.0),
                (0.0, margin),
                (0.0, -margin),
            ),
            dtype=foot_world.dtype,
            device=foot_world.device,
        )
        landing_xy = foot_world[rows, terminal, swing, :2]
        edge = _sample_surface(
            sample_surface, landing_xy[:, None, :] + offsets[None, :, :]
        )
        edge_range = edge.amax(dim=1) - edge.amin(dim=1)

        swing_samples = valid[..., None] & ~support
        minimum_swing = torch.where(
            swing_samples,
            clearance,
            torch.full_like(clearance, torch.inf),
        ).amin(dim=(1, 2))
        minimum_swing = torch.where(
            swing_samples.any(dim=(1, 2)), minimum_swing, landing_error
        )
        query_delta = (
            surface[rows, terminal, swing] - surface[rows, 0, swing]
        )
        source_delta = self.surface_delta_m[candidates][
            rows, terminal, swing
        ]
        deformation = torch.abs(query_delta - source_delta)

        results = []
        for index in range(int(candidates.numel())):
            reason = None
            if float(stance_error[index].item()) > float(
                constraints.stance_height_tolerance_m
            ):
                reason = "stance-height"
            elif float(landing_error[index].item()) > float(
                constraints.landing_height_tolerance_m
            ):
                reason = "landing-height"
            elif float(edge_range[index].item()) > float(
                constraints.maximum_edge_height_range_m
            ):
                reason = "landing-edge-margin"
            elif float(minimum_swing[index].item()) < float(
                constraints.minimum_swing_clearance_m
            ):
                reason = "swing-penetration"
            elif float(deformation[index].item()) > float(
                constraints.maximum_height_deformation_m
            ):
                reason = "height-deformation"
            results.append(
                _result(
                    reason,
                    stance=float(stance_error[index].item()),
                    landing=float(landing_error[index].item()),
                    swing=float(minimum_swing[index].item()),
                )
            )
        return tuple(results)


def advance_state(
    state: OracleState, placed: PlacedContactPhase
) -> OracleState:
    action = placed.action
    steps = action.frame_count - 1
    return OracleState(
        root_position_world=placed.root_position_world[-1],
        root_yaw_world=placed.root_yaw_world[-1],
        root_orientation_world_wxyz=(
            placed.root_orientation_world_wxyz[-1]
        ),
        foot_position_world=placed.foot_position_world[-1],
        support_mask=action.exit_support,
        joint_position=action.joint_position[-1],
        joint_velocity=action.joint_velocity[-1],
        route_frame=state.route_frame + steps,
        source_history=(*state.source_history, action.source_key),
    )


def _feasible_actions(
    state: OracleState,
    actions: Sequence[ContactPhaseAction],
    schedule: CommandSchedule,
    foothold_plan: FootholdPlan,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    config: OracleSearchConfig,
    entry_cache: _ActionEntryCache,
    rejected: Counter[str] | None = None,
    *,
    exhaustive_on_empty: bool = False,
) -> list[tuple[int, PlacedContactPhase, FeasibilityResult]]:
    output = []
    candidates = entry_cache.candidate_indices(
        state, config.constraints, rejected
    )
    candidate_indices = entry_cache.shortlisted_indices(
        candidates, state, schedule, foothold_plan, config, None
    )

    def evaluate(indices: torch.Tensor) -> None:
        results = entry_cache.terrain_feasibility(
            indices, state, sample_surface, config.constraints
        )
        for index, result in zip(indices.detach().cpu().tolist(), results):
            action = actions[index]
            if result.accepted:
                placed = place_action(action, state)
                output.append((index, placed, result))
            elif rejected is not None:
                assert result.reason is not None
                rejected[result.reason] += 1

    evaluate(candidate_indices)
    pruned_count = int(candidates.numel() - candidate_indices.numel())
    if output or pruned_count == 0 or not exhaustive_on_empty:
        if rejected is not None and pruned_count:
            rejected["shortlist-pruned"] += pruned_count
        return output

    remaining = candidates[
        ~torch.isin(candidates, candidate_indices)
    ]
    evaluate(remaining)
    return output


def _edge_cost(
    state: OracleState,
    placed: PlacedContactPhase,
    feasibility: FeasibilityResult,
    schedule: CommandSchedule,
    config: OracleSearchConfig,
    successor_count: int,
    foothold_plan: FootholdPlan,
) -> OracleCost:
    action = placed.action
    steps = action.frame_count - 1
    if schedule.origin_world_xy is None:
        desired_displacement = schedule.displacement(state.route_frame, steps)
        actual_displacement = (
            placed.root_position_world[-1, :2]
            - state.root_position_world[:2]
        )
    else:
        desired_displacement = schedule.target_position(
            state.route_frame + steps
        )
        actual_displacement = placed.root_position_world[-1, :2]
    path = float(
        config.path_weight
        * torch.sum(torch.square(actual_displacement - desired_displacement)).item()
    )
    desired_heading = schedule.heading(state.route_frame + steps)
    heading_error = torch.atan2(
        torch.sin(placed.root_yaw_world[-1] - desired_heading),
        torch.cos(placed.root_yaw_world[-1] - desired_heading),
    )
    facing = float(config.facing_weight * torch.square(heading_error).item())
    matching_foot = foothold_plan.landing_feet[:, 0] == action.swing_foot
    if bool(matching_foot.any().item()):
        target_xy = foothold_plan.landing_xy_world_m[matching_foot, 0]
        xy_error_sq = torch.sum(
            torch.square(
                target_xy
                - placed.foot_position_world[-1, action.swing_foot, :2]
            ),
            dim=1,
        )
        target_timing = foothold_plan.landing_frame_offsets[
            matching_foot, 0
        ].to(torch.float32)
        timing_error_sq = torch.square(target_timing - float(steps))
        combined = (
            config.foothold_weight * xy_error_sq
            + config.timing_weight * timing_error_sq
        )
        target = int(torch.argmin(combined).item())
        foothold = float(config.foothold_weight * xy_error_sq[target].item())
        timing = float(config.timing_weight * timing_error_sq[target].item())
    elif foothold_plan.score.numel():
        foothold = float(config.foothold_weight * 100.0)
        timing = float(config.timing_weight * 100.0)
    else:
        foothold = 0.0
        timing = 0.0
    joint_position = float(
        config.joint_position_weight
        * torch.sum(torch.square(action.joint_position[0] - state.joint_position)).item()
    )
    joint_velocity = float(
        config.joint_velocity_weight
        * torch.sum(torch.square(action.joint_velocity[0] - state.joint_velocity)).item()
    )
    entry_contact_error = torch.mean(
        torch.square(
            placed.foot_position_world[0, state.support_mask]
            - state.foot_position_world[state.support_mask]
        )
    )
    stance = action.entry_support.clone()
    stance[action.swing_foot] = False
    if bool(stance.any().item()):
        displacement = (
            placed.foot_position_world[:, stance]
            - placed.foot_position_world[0:1, stance]
        )
        stance_error = torch.mean(torch.square(displacement))
    else:
        stance_error = torch.zeros_like(entry_contact_error)
    stance_motion = float(
        config.stance_motion_weight
        * (entry_contact_error + stance_error).item()
    )
    clearance_deficit = max(0.0, 0.05 - feasibility.minimum_swing_clearance_m)
    repetition_count = state.source_history.count(action.source_key)
    return OracleCost(
        path=path,
        facing=facing,
        foothold=foothold,
        timing=timing,
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        stance_motion=stance_motion,
        clearance_margin=config.clearance_margin_weight * clearance_deficit**2,
        reachability=config.reachability_weight / (1.0 + successor_count),
        repetition=config.repetition_weight * repetition_count,
    )


def _foothold_plan_for_state(
    state: OracleState,
    schedule: CommandSchedule,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    config: OracleSearchConfig,
) -> FootholdPlan:
    return plan_footholds(
        foot_xy_m=state.foot_position_world[:, :2],
        support_mask=state.support_mask,
        command_xy=schedule.velocity(state.route_frame),
        sample_surface=sample_surface,
        reachable_forward_m=(0.20, 0.25, 0.30, 0.35, 0.40),
        lateral_samples_m=(-0.15, -0.075, 0.0, 0.075, 0.15),
        edge_margin_m=config.constraints.edge_margin_m,
        beam_width=config.foothold_beam_width,
    )


def _search_horizon(
    *,
    horizon: int,
    initial_state: OracleState,
    actions: Sequence[ContactPhaseAction],
    command_schedule: CommandSchedule,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    config: OracleSearchConfig,
    rejected: Counter[str],
    entry_cache: _ActionEntryCache,
    exhaustive_failure_audit: bool = False,
) -> tuple[_SearchNode | None, tuple[int, ...], int]:
    beam = (
        _SearchNode(initial_state, (), (), (), OracleCost()),
    )
    beam_sizes = []
    expanded = 0
    for depth in range(horizon):
        candidates: list[_SearchNode] = []
        for node in beam:
            foothold_plan = _foothold_plan_for_state(
                node.state,
                command_schedule,
                sample_surface,
                config,
            )
            if node.cached_feasible is None:
                feasible = _feasible_actions(
                    node.state,
                    actions,
                    command_schedule,
                    foothold_plan,
                    sample_surface,
                    config,
                    entry_cache,
                    rejected,
                    exhaustive_on_empty=exhaustive_failure_audit,
                )
            else:
                feasible = list(node.cached_feasible)
                if node.cached_rejected:
                    rejected.update(node.cached_rejected)
            expanded += len(actions)
            for action_index, placed, result in feasible:
                child_state = advance_state(node.state, placed)
                successor_count = 0
                successors = None
                successor_rejected = None
                if depth + 1 < horizon:
                    successor_rejected = Counter()
                    child_foothold_plan = _foothold_plan_for_state(
                        child_state,
                        command_schedule,
                        sample_surface,
                        config,
                    )
                    successors = tuple(
                        _feasible_actions(
                            child_state,
                            actions,
                            command_schedule,
                            child_foothold_plan,
                            sample_surface,
                            config,
                            entry_cache,
                            successor_rejected,
                        )
                    )
                    successor_count = len(successors)
                    if successor_count == 0:
                        rejected["no-successor"] += 1
                        continue
                edge_cost = _edge_cost(
                    node.state,
                    placed,
                    result,
                    command_schedule,
                    config,
                    successor_count,
                    foothold_plan,
                )
                candidates.append(
                    _SearchNode(
                        state=child_state,
                        action_indices=(*node.action_indices, action_index),
                        placements=(*node.placements, placed),
                        step_costs=(*node.step_costs, edge_cost),
                        total_cost=node.total_cost + edge_cost,
                        cached_feasible=successors,
                        cached_rejected=successor_rejected,
                    )
                )
        candidates.sort(
            key=lambda node: (node.total_cost.total, node.source_key_path)
        )
        beam = tuple(candidates[: config.beam_width])
        beam_sizes.append(len(beam))
        if not beam:
            return None, tuple(beam_sizes), expanded
    return beam[0], tuple(beam_sizes), expanded


def search_contact_plan(
    *,
    initial_state: OracleState,
    actions: Sequence[ContactPhaseAction],
    command_schedule: CommandSchedule,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    config: OracleSearchConfig = OracleSearchConfig(),
) -> OraclePlan:
    """Return the best complete horizon, shortening only after exhaustion."""

    inventory = tuple(actions)
    if (
        not isinstance(initial_state, OracleState)
        or not inventory
        or any(not isinstance(action, ContactPhaseAction) for action in inventory)
        or not isinstance(command_schedule, CommandSchedule)
        or not callable(sample_surface)
        or not isinstance(config, OracleSearchConfig)
    ):
        raise ContractError("contact oracle search inputs are invalid")
    if any(
        action.root_position_local.device != initial_state.root_position_world.device
        for action in inventory
    ) or command_schedule.velocity_world_xy.device != initial_state.root_position_world.device:
        raise ContractError("contact oracle search devices do not match")
    rejected: Counter[str] = Counter()
    entry_cache = _ActionEntryCache.from_actions(inventory)
    total_expanded = 0
    all_beam_sizes: list[int] = []
    for horizon in range(config.horizon_landings, 0, -1):
        node, beam_sizes, expanded = _search_horizon(
            horizon=horizon,
            initial_state=initial_state,
            actions=inventory,
            command_schedule=command_schedule,
            sample_surface=sample_surface,
            config=config,
            rejected=rejected,
            entry_cache=entry_cache,
            exhaustive_failure_audit=horizon == 1,
        )
        total_expanded += expanded
        all_beam_sizes.extend(beam_sizes)
        if node is not None:
            return OraclePlan(
                action_indices=node.action_indices,
                placements=node.placements,
                step_costs=node.step_costs,
                total_cost=node.total_cost,
                horizon_landings=horizon,
                expansion=OracleExpansionDiagnostics(
                    expanded_candidate_count=total_expanded,
                    rejected_by_reason=rejected,
                    beam_sizes=tuple(all_beam_sizes),
                ),
            )
    raise OracleSearchFailure(rejected)
