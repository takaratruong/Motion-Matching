"""Bounded operator commands and exact dense Torch motion search."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .joints import ContractError
from .torch_motion_features import CommandTrajectory, TorchMotionDatabase


@dataclass(frozen=True)
class MatcherConfig:
    dt: float = 0.02
    search_interval_steps: int = 5
    acceleration_mps2: float = 1.5
    deceleration_mps2: float = 2.0
    yaw_rate_rad_s: float = math.radians(120.0)
    stop_speed_mps: float = 0.05
    reversal_speed_mps: float = 0.15
    exclusion_frames: int = 20
    transition_penalty: float = 0.1
    inertialization_halflife_s: float = 0.10


@dataclass(frozen=True)
class ShapedCommand:
    velocity_world_xy: torch.Tensor
    heading_world_yaw: torch.Tensor
    trajectory: CommandTrajectory
    force_search: bool


@dataclass(frozen=True)
class SearchDecision:
    selected_row: int
    incumbent_row: int | None
    incumbent_cost: float
    selected_feature_cost: float
    selected_total_cost: float
    searched: bool
    transitioned: bool


def _require_planar_pair(
    current: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(current, torch.Tensor) or tuple(current.shape) != (2,):
        raise ContractError("current velocity must be a torch.Tensor with shape (2,)")
    if not isinstance(target, torch.Tensor) or tuple(target.shape) != (2,):
        raise ContractError("target velocity must be a torch.Tensor with shape (2,)")
    if current.device != target.device or current.dtype != target.dtype:
        raise ContractError("current and target velocity device/dtype must agree")
    if not current.dtype.is_floating_point:
        raise ContractError("velocity tensors must have floating-point dtype")
    if not torch.isfinite(current).all() or not torch.isfinite(target).all():
        raise ContractError("velocity tensors must be finite")
    return current, target


def bounded_velocity_step(
    current_velocity_world_xy: torch.Tensor,
    target_velocity_world_xy: torch.Tensor,
    *,
    config: MatcherConfig = MatcherConfig(),
    dt: float | None = None,
) -> torch.Tensor:
    """Move one planar velocity vector toward another by a bounded L2 step."""
    current, target = _require_planar_pair(
        current_velocity_world_xy, target_velocity_world_xy
    )
    step_dt = config.dt if dt is None else float(dt)
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ContractError("dt must be finite and positive")

    current_speed = torch.linalg.vector_norm(current)
    target_speed = torch.linalg.vector_norm(target)
    decreasing = bool(
        (target_speed < current_speed).item() or (torch.dot(current, target) < 0).item()
    )
    acceleration = (
        config.deceleration_mps2 if decreasing else config.acceleration_mps2
    )
    delta = target - current
    distance = torch.linalg.vector_norm(delta)
    maximum = acceleration * step_dt
    if bool((distance <= maximum).item()):
        return target.clone()
    return current + delta * (maximum / distance)


def _wrapped_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def bounded_yaw_step(
    current_heading_world_yaw: torch.Tensor,
    target_heading_world_yaw: torch.Tensor,
    *,
    config: MatcherConfig = MatcherConfig(),
    dt: float | None = None,
) -> torch.Tensor:
    """Advance yaw along the shortest wrapped arc under the configured cap."""
    current = current_heading_world_yaw
    target = target_heading_world_yaw
    if (
        not isinstance(current, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or current.numel() != 1
        or target.numel() != 1
    ):
        raise ContractError("yaw values must be scalar torch.Tensor values")
    if current.device != target.device or current.dtype != target.dtype:
        raise ContractError("current and target yaw device/dtype must agree")
    if not current.dtype.is_floating_point:
        raise ContractError("yaw tensors must have floating-point dtype")
    if not torch.isfinite(current).all() or not torch.isfinite(target).all():
        raise ContractError("yaw tensors must be finite")
    step_dt = config.dt if dt is None else float(dt)
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ContractError("dt must be finite and positive")

    delta = _wrapped_angle(target - current)
    cap = config.yaw_rate_rad_s * step_dt
    return _wrapped_angle(current + torch.clamp(delta, min=-cap, max=cap))


def _force_search_for_command(
    current_velocity_world_xy: torch.Tensor,
    requested_velocity_world_xy: torch.Tensor,
    has_valid_successor: bool,
    config: MatcherConfig,
) -> bool:
    current_speed = float(torch.linalg.vector_norm(current_velocity_world_xy))
    requested_speed = float(torch.linalg.vector_norm(requested_velocity_world_xy))
    current_stopped = current_speed <= config.stop_speed_mps
    requested_stopped = requested_speed <= config.stop_speed_mps
    reversal = (
        current_speed >= config.reversal_speed_mps
        and requested_speed >= config.reversal_speed_mps
        and float(torch.dot(current_velocity_world_xy, requested_velocity_world_xy))
        < 0.0
    )
    return (
        current_stopped != requested_stopped
        or reversal
        or not bool(has_valid_successor)
    )


def search_is_due(
    call_index: int,
    force_search: bool,
    config: MatcherConfig = MatcherConfig(),
) -> bool:
    """Return whether this call is on the 10 Hz cadence or is forced."""
    if config.search_interval_steps <= 0:
        raise ContractError("search_interval_steps must be positive")
    if int(call_index) != call_index or call_index < 0:
        raise ContractError("call_index must be a non-negative integer")
    return bool(force_search) or int(call_index) % config.search_interval_steps == 0


def predict_command_trajectory(
    root_position_world_xy: torch.Tensor,
    current_velocity_world_xy: torch.Tensor,
    current_heading_world_yaw: torch.Tensor,
    requested_velocity_world_xy: torch.Tensor,
    requested_heading_world_yaw: torch.Tensor,
    *,
    has_valid_successor: bool = True,
    config: MatcherConfig = MatcherConfig(),
) -> ShapedCommand:
    """Shape one command and simulate it through the 15/30/45-frame horizons."""
    current_velocity, requested_velocity = _require_planar_pair(
        current_velocity_world_xy, requested_velocity_world_xy
    )
    if (
        not isinstance(root_position_world_xy, torch.Tensor)
        or tuple(root_position_world_xy.shape) != (2,)
        or root_position_world_xy.device != current_velocity.device
        or root_position_world_xy.dtype != current_velocity.dtype
        or not torch.isfinite(root_position_world_xy).all()
    ):
        raise ContractError(
            "root_position_world_xy must be a finite shape-(2,) tensor "
            "matching command device/dtype"
        )

    first_velocity = bounded_velocity_step(
        current_velocity, requested_velocity, config=config
    )
    first_heading = bounded_yaw_step(
        current_heading_world_yaw, requested_heading_world_yaw, config=config
    )
    force_search = _force_search_for_command(
        current_velocity, requested_velocity, has_valid_successor, config
    )

    position = root_position_world_xy.clone()
    velocity = current_velocity.clone()
    heading = current_heading_world_yaw.clone()
    sampled_positions: list[torch.Tensor] = []
    sampled_facing: list[torch.Tensor] = []
    horizon_set = {15, 30, 45}
    for step in range(1, 46):
        velocity = bounded_velocity_step(velocity, requested_velocity, config=config)
        heading = bounded_yaw_step(heading, requested_heading_world_yaw, config=config)
        position = position + velocity * config.dt
        if step in horizon_set:
            sampled_positions.append(position.clone())
            sampled_facing.append(
                torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
            )

    return ShapedCommand(
        velocity_world_xy=first_velocity,
        heading_world_yaw=first_heading,
        trajectory=CommandTrajectory(
            position_world_xy=torch.stack(sampled_positions),
            facing_world_xy=torch.stack(sampled_facing),
        ),
        force_search=force_search,
    )


def _validate_search_inputs(
    database: TorchMotionDatabase,
    normalized_query: torch.Tensor,
    incumbent_row: int | None,
) -> tuple[torch.Tensor, int]:
    features = database._search_features
    if (
        not isinstance(normalized_query, torch.Tensor)
        or normalized_query.ndim != 1
        or normalized_query.shape[0] != features.shape[1]
    ):
        raise ContractError(
            f"normalized_query must have shape ({features.shape[1]},)"
        )
    if normalized_query.device != features.device or normalized_query.dtype != torch.float32:
        raise ContractError("normalized_query must be float32 on the database device")
    if not torch.isfinite(normalized_query).all():
        raise ContractError("normalized_query must be finite")
    row_count = int(features.shape[0])
    if incumbent_row is not None and not 0 <= incumbent_row < row_count:
        raise ContractError("incumbent_row is outside the database")
    return features, row_count


def select_exact_candidate(
    database: TorchMotionDatabase,
    normalized_query: torch.Tensor,
    *,
    current_clip_index: int,
    current_frame_index: int,
    incumbent_row: int | None,
    search: bool,
    config: MatcherConfig = MatcherConfig(),
) -> SearchDecision:
    """Select the exact lowest-cost eligible row with continuation hysteresis."""
    features, row_count = _validate_search_inputs(
        database, normalized_query, incumbent_row
    )
    if incumbent_row is None and not search:
        raise ContractError("search=False requires a valid incumbent")

    # One dense float32 squared-L2 reduction over every database row.
    feature_costs = torch.sum(
        torch.square(features - normalized_query.unsqueeze(0)), dim=1
    )
    incumbent_cost_tensor = (
        feature_costs[incumbent_row]
        if incumbent_row is not None
        else torch.tensor(math.inf, device=features.device)
    )
    if not search:
        incumbent_cost = float(incumbent_cost_tensor.item())
        return SearchDecision(
            selected_row=incumbent_row,
            incumbent_row=incumbent_row,
            incumbent_cost=incumbent_cost,
            selected_feature_cost=incumbent_cost,
            selected_total_cost=incumbent_cost,
            searched=False,
            transitioned=False,
        )

    eligible = torch.ones(row_count, dtype=torch.bool, device=features.device)
    local = (
        database._search_clip_index == int(current_clip_index)
    ) & (
        torch.abs(database._search_frame_index - int(current_frame_index))
        <= config.exclusion_frames
    )
    eligible &= ~local
    if incumbent_row is not None:
        eligible[incumbent_row] = True

    total_costs = feature_costs + config.transition_penalty
    if incumbent_row is not None:
        total_costs[incumbent_row] = feature_costs[incumbent_row]
    total_costs = total_costs.masked_fill(~eligible, math.inf)
    candidate_row_tensor = torch.argmin(total_costs)
    candidate_total_tensor = total_costs[candidate_row_tensor]
    if incumbent_row is None:
        selected_row_tensor = candidate_row_tensor
    else:
        selected_row_tensor = torch.where(
            candidate_total_tensor < incumbent_cost_tensor,
            candidate_row_tensor,
            torch.tensor(incumbent_row, device=features.device),
        )
    selected_feature_cost_tensor = feature_costs[selected_row_tensor]
    selected_total_cost_tensor = total_costs[selected_row_tensor]

    # Convert all diagnostics in one device synchronization.
    diagnostics = torch.stack(
        (
            selected_row_tensor.to(torch.float64),
            incumbent_cost_tensor.to(torch.float64),
            selected_feature_cost_tensor.to(torch.float64),
            selected_total_cost_tensor.to(torch.float64),
            candidate_total_tensor.to(torch.float64),
        )
    ).cpu().tolist()
    selected_row = int(diagnostics[0])
    incumbent_cost = float(diagnostics[1])
    selected_feature_cost = float(diagnostics[2])
    selected_total_cost = float(diagnostics[3])
    candidate_total = float(diagnostics[4])
    if not math.isfinite(candidate_total):
        raise ContractError("no valid motion-matching candidate exists")

    return SearchDecision(
        selected_row=selected_row,
        incumbent_row=incumbent_row,
        incumbent_cost=incumbent_cost,
        selected_feature_cost=selected_feature_cost,
        selected_total_cost=selected_total_cost,
        searched=True,
        transitioned=incumbent_row is None or selected_row != incumbent_row,
    )
