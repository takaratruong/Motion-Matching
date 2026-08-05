"""Bounded operator commands and exact dense Torch motion search."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import math
from pathlib import Path
import time

import torch

from .joints import ContractError
from .torch_motion_data import MotionFolder
from .torch_motion_features import (
    CommandTrajectory,
    GeneratedFeatureState,
    TorchMotionDatabase,
    extract_query_features,
)


@dataclass(frozen=True)
class MatcherConfig:
    dt: float = 0.02
    trajectory_model: str = "legacy"
    search_interval_steps: int = 5
    acceleration_mps2: float = 1.5
    deceleration_mps2: float = 2.0
    yaw_rate_rad_s: float = math.radians(120.0)
    stop_speed_mps: float = 0.05
    reversal_speed_mps: float = 0.15
    exclusion_frames: int = 20
    transition_penalty: float = 0.1
    inertialization_halflife_s: float = 0.10
    simulation_velocity_halflife_s: float = 0.27
    simulation_rotation_halflife_s: float = 0.27
    adjustment_position_halflife_s: float = 0.10
    adjustment_rotation_halflife_s: float = 0.20
    adjustment_position_max_ratio: float = 0.50
    adjustment_rotation_max_ratio: float = 0.50
    clamping_max_distance_m: float = 0.15
    clamping_max_angle_rad: float = math.pi / 2.0
    max_source_joint_step_rad: float | None = None
    feature_weight_overrides: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.trajectory_model not in {"legacy", "takara_ball"}:
            raise ContractError(
                "trajectory_model must be 'legacy' or 'takara_ball'"
            )
        positive = {
            "dt": self.dt,
            "simulation_velocity_halflife_s": self.simulation_velocity_halflife_s,
            "simulation_rotation_halflife_s": self.simulation_rotation_halflife_s,
            "adjustment_position_halflife_s": self.adjustment_position_halflife_s,
            "adjustment_rotation_halflife_s": self.adjustment_rotation_halflife_s,
            "clamping_max_distance_m": self.clamping_max_distance_m,
            "clamping_max_angle_rad": self.clamping_max_angle_rad,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ContractError(f"{name} must be finite and positive")
        ratios = {
            "adjustment_position_max_ratio": self.adjustment_position_max_ratio,
            "adjustment_rotation_max_ratio": self.adjustment_rotation_max_ratio,
        }
        for name, value in ratios.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ContractError(f"{name} must be finite and non-negative")
        if self.max_source_joint_step_rad is not None and (
            not math.isfinite(float(self.max_source_joint_step_rad))
            or float(self.max_source_joint_step_rad) <= 0.0
        ):
            raise ContractError(
                "max_source_joint_step_rad must be finite and positive"
            )
        names = [str(name) for name, _value in self.feature_weight_overrides]
        if len(names) != len(set(names)):
            raise ContractError("feature weight overrides contain duplicate groups")
        for name, value in self.feature_weight_overrides:
            if not name or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ContractError(
                    "feature weight overrides must be named, finite, and positive"
                )


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
    decreasing = (target_speed < current_speed) | (torch.dot(current, target) < 0)
    acceleration = torch.where(
        decreasing,
        torch.as_tensor(config.deceleration_mps2, device=current.device),
        torch.as_tensor(config.acceleration_mps2, device=current.device),
    )
    delta = target - current
    distance = torch.linalg.vector_norm(delta)
    maximum = acceleration * step_dt
    fraction = torch.clamp(maximum / distance.clamp_min(1e-12), max=1.0)
    return current + delta * fraction


def _wrapped_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _takara_fast_negexp(value: float) -> float:
    """The rational exp(-x) approximation used by Takara's C++ runtime."""
    return 1.0 / (
        1.0 + value + 0.48 * value * value + 0.235 * value * value * value
    )


def critically_damped_position_step(
    position: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    desired_velocity: torch.Tensor,
    *,
    halflife_s: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Takara's exact persistent simulation-position spring."""
    if not all(
        isinstance(value, torch.Tensor)
        for value in (position, velocity, acceleration, desired_velocity)
    ):
        raise ContractError("simulation position values must be tensors")
    if not (
        position.shape
        == velocity.shape
        == acceleration.shape
        == desired_velocity.shape
    ):
        raise ContractError("simulation position values must have equal shapes")
    if not (
        position.device
        == velocity.device
        == acceleration.device
        == desired_velocity.device
        and position.dtype
        == velocity.dtype
        == acceleration.dtype
        == desired_velocity.dtype
    ):
        raise ContractError("simulation position device/dtype values must agree")
    if not position.dtype.is_floating_point:
        raise ContractError("simulation position values must be floating point")
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (position, velocity, acceleration, desired_velocity)
    ):
        raise ContractError("simulation position values must be finite")
    if (
        not math.isfinite(halflife_s)
        or halflife_s <= 0.0
        or not math.isfinite(dt)
        or dt < 0.0
    ):
        raise ContractError("spring halflife must be positive and dt non-negative")

    y = (4.0 * math.log(2.0) / (halflife_s + 1.0e-5)) / 2.0
    j0 = velocity - desired_velocity
    j1 = acceleration + j0 * y
    decay = _takara_fast_negexp(y * dt)
    next_position = (
        decay * ((-j1 / (y * y)) + ((-j0 - j1 * dt) / y))
        + j1 / (y * y)
        + j0 / y
        + desired_velocity * dt
        + position
    )
    next_velocity = decay * (j0 + j1 * dt) + desired_velocity
    next_acceleration = decay * (acceleration - j1 * y * dt)
    return next_position, next_velocity, next_acceleration


def critically_damped_yaw_step(
    yaw: torch.Tensor,
    angular_velocity: torch.Tensor,
    desired_yaw: torch.Tensor,
    *,
    halflife_s: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Yaw-only specialization of Takara's exact quaternion spring."""
    if not all(
        isinstance(value, torch.Tensor)
        for value in (yaw, angular_velocity, desired_yaw)
    ):
        raise ContractError("simulation yaw values must be tensors")
    if any(value.numel() != 1 for value in (yaw, angular_velocity, desired_yaw)):
        raise ContractError("simulation yaw values must be scalar tensors")
    if not (
        yaw.device == angular_velocity.device == desired_yaw.device
        and yaw.dtype == angular_velocity.dtype == desired_yaw.dtype
    ):
        raise ContractError("simulation yaw device/dtype values must agree")
    if not yaw.dtype.is_floating_point:
        raise ContractError("simulation yaw values must be floating point")
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (yaw, angular_velocity, desired_yaw)
    ):
        raise ContractError("simulation yaw values must be finite")
    if (
        not math.isfinite(halflife_s)
        or halflife_s <= 0.0
        or not math.isfinite(dt)
        or dt < 0.0
    ):
        raise ContractError("spring halflife must be positive and dt non-negative")

    y = (4.0 * math.log(2.0) / (halflife_s + 1.0e-5)) / 2.0
    j0 = _wrapped_angle(yaw.reshape(()) - desired_yaw.reshape(()))
    j1 = angular_velocity.reshape(()) + j0 * y
    decay = _takara_fast_negexp(y * dt)
    next_yaw = _wrapped_angle(
        decay * (j0 + j1 * dt) + desired_yaw.reshape(())
    )
    next_angular_velocity = decay * (
        angular_velocity.reshape(()) - j1 * y * dt
    )
    return next_yaw, next_angular_velocity


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
    current_speed = torch.linalg.vector_norm(current_velocity_world_xy)
    requested_speed = torch.linalg.vector_norm(requested_velocity_world_xy)
    current_stopped = current_speed <= config.stop_speed_mps
    requested_stopped = requested_speed <= config.stop_speed_mps
    reversal = (
        current_speed >= config.reversal_speed_mps
    ) & (
        requested_speed >= config.reversal_speed_mps
    ) & (
        torch.dot(current_velocity_world_xy, requested_velocity_world_xy) < 0.0
    )
    forced = (current_stopped != requested_stopped) | reversal
    return bool(forced.item()) or not bool(has_valid_successor)


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

    force_search = _force_search_for_command(
        current_velocity, requested_velocity, has_valid_successor, config
    )
    steps = torch.arange(
        1, 46, device=current_velocity.device, dtype=current_velocity.dtype
    )
    delta_velocity = requested_velocity - current_velocity
    distance = torch.linalg.vector_norm(delta_velocity)
    decreasing = (
        torch.linalg.vector_norm(requested_velocity)
        < torch.linalg.vector_norm(current_velocity)
    ) | (torch.dot(current_velocity, requested_velocity) < 0)
    acceleration = torch.where(
        decreasing,
        torch.as_tensor(config.deceleration_mps2, device=current_velocity.device),
        torch.as_tensor(config.acceleration_mps2, device=current_velocity.device),
    )
    fractions = torch.clamp(
        steps * acceleration * config.dt / distance.clamp_min(1e-12), max=1.0
    )
    velocities = current_velocity + fractions[:, None] * delta_velocity
    positions = root_position_world_xy + torch.cumsum(
        velocities * config.dt, dim=0
    )

    heading_delta = _wrapped_angle(
        requested_heading_world_yaw - current_heading_world_yaw
    ).reshape(())
    signed_steps = torch.sign(heading_delta) * torch.minimum(
        steps * config.yaw_rate_rad_s * config.dt,
        torch.abs(heading_delta).expand_as(steps),
    )
    headings = _wrapped_angle(
        current_heading_world_yaw.reshape(()) + signed_steps
    )
    horizon_indices = torch.tensor(
        (14, 29, 44), device=current_velocity.device
    )
    sampled_headings = headings[horizon_indices]

    return ShapedCommand(
        velocity_world_xy=velocities[0],
        heading_world_yaw=headings[0],
        trajectory=CommandTrajectory(
            position_world_xy=positions[horizon_indices],
            facing_world_xy=torch.stack(
                (torch.cos(sampled_headings), torch.sin(sampled_headings)), dim=-1
            ),
        ),
        force_search=force_search,
    )


@dataclass(frozen=True)
class TakaraBallPrediction:
    command: ShapedCommand
    next_position_world_xy: torch.Tensor
    next_velocity_world_xy: torch.Tensor
    next_acceleration_world_xy: torch.Tensor
    next_heading_world_yaw: torch.Tensor
    next_heading_velocity_rad_s: torch.Tensor


def predict_takara_ball_trajectory(
    simulation_position_world_xy: torch.Tensor,
    simulation_velocity_world_xy: torch.Tensor,
    simulation_acceleration_world_xy: torch.Tensor,
    simulation_heading_world_yaw: torch.Tensor,
    simulation_heading_velocity_rad_s: torch.Tensor,
    requested_velocity_world_xy: torch.Tensor,
    requested_heading_world_yaw: torch.Tensor,
    *,
    has_valid_successor: bool = True,
    config: MatcherConfig = MatcherConfig(trajectory_model="takara_ball"),
) -> TakaraBallPrediction:
    """Predict and advance Takara's persistent command-space simulation ball."""
    if config.trajectory_model != "takara_ball":
        raise ContractError("Takara ball prediction requires trajectory_model=takara_ball")
    current_velocity, requested_velocity = _require_planar_pair(
        simulation_velocity_world_xy, requested_velocity_world_xy
    )
    if (
        not isinstance(simulation_position_world_xy, torch.Tensor)
        or simulation_position_world_xy.shape != current_velocity.shape
        or simulation_position_world_xy.device != current_velocity.device
        or simulation_position_world_xy.dtype != current_velocity.dtype
    ):
        raise ContractError("simulation position must match planar velocity")
    if (
        not isinstance(simulation_acceleration_world_xy, torch.Tensor)
        or simulation_acceleration_world_xy.shape != current_velocity.shape
        or simulation_acceleration_world_xy.device != current_velocity.device
        or simulation_acceleration_world_xy.dtype != current_velocity.dtype
    ):
        raise ContractError("simulation acceleration must match planar velocity")

    horizon_positions = []
    horizon_headings = []
    predicted_position = simulation_position_world_xy
    predicted_velocity = current_velocity
    predicted_acceleration = simulation_acceleration_world_xy
    trajectory_sample_time = 15 * config.dt
    for sample_index in (1, 2, 3):
        predicted_position, predicted_velocity, predicted_acceleration = (
            critically_damped_position_step(
                predicted_position,
                predicted_velocity,
                predicted_acceleration,
                requested_velocity,
                halflife_s=config.simulation_velocity_halflife_s,
                dt=trajectory_sample_time,
            )
        )
        heading, _heading_velocity = critically_damped_yaw_step(
            simulation_heading_world_yaw,
            simulation_heading_velocity_rad_s,
            requested_heading_world_yaw,
            halflife_s=config.simulation_rotation_halflife_s,
            dt=sample_index * trajectory_sample_time,
        )
        horizon_positions.append(predicted_position)
        horizon_headings.append(heading)

    next_position, next_velocity, next_acceleration = (
        critically_damped_position_step(
            simulation_position_world_xy,
            current_velocity,
            simulation_acceleration_world_xy,
            requested_velocity,
            halflife_s=config.simulation_velocity_halflife_s,
            dt=config.dt,
        )
    )
    next_heading, next_heading_velocity = critically_damped_yaw_step(
        simulation_heading_world_yaw,
        simulation_heading_velocity_rad_s,
        requested_heading_world_yaw,
        halflife_s=config.simulation_rotation_halflife_s,
        dt=config.dt,
    )
    sampled_headings = torch.stack(horizon_headings)
    force_search = _force_search_for_command(
        current_velocity, requested_velocity, has_valid_successor, config
    )
    return TakaraBallPrediction(
        command=ShapedCommand(
            velocity_world_xy=next_velocity,
            heading_world_yaw=next_heading,
            trajectory=CommandTrajectory(
                position_world_xy=torch.stack(horizon_positions),
                facing_world_xy=torch.stack(
                    (torch.cos(sampled_headings), torch.sin(sampled_headings)),
                    dim=-1,
                ),
            ),
            force_search=force_search,
        ),
        next_position_world_xy=next_position,
        next_velocity_world_xy=next_velocity,
        next_acceleration_world_xy=next_acceleration,
        next_heading_world_yaw=next_heading,
        next_heading_velocity_rad_s=next_heading_velocity,
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
    excluded_rows: tuple[int, ...] = (),
) -> SearchDecision:
    """Select the exact lowest-cost eligible row with continuation hysteresis."""
    features, row_count = _validate_search_inputs(
        database, normalized_query, incumbent_row
    )
    if incumbent_row is None and not search:
        raise ContractError("search=False requires a valid incumbent")
    if (
        not isinstance(excluded_rows, tuple)
        or any(type(row) is not int for row in excluded_rows)
        or len(set(excluded_rows)) != len(excluded_rows)
        or any(not 0 <= row < row_count for row in excluded_rows)
    ):
        raise ContractError(
            "excluded_rows must contain unique database row integers"
        )
    if excluded_rows and not search:
        raise ContractError("excluding candidates requires search=True")

    if not search:
        assert incumbent_row is not None
        incumbent_cost = float(
            torch.sum(
                torch.square(features[incumbent_row] - normalized_query)
            ).item()
        )
        return SearchDecision(
            selected_row=incumbent_row,
            incumbent_row=incumbent_row,
            incumbent_cost=incumbent_cost,
            selected_feature_cost=incumbent_cost,
            selected_total_cost=incumbent_cost,
            searched=False,
            transitioned=False,
        )

    # Search boundaries use one dense float32 squared-L2 reduction.
    feature_costs = torch.sum(
        torch.square(features - normalized_query.unsqueeze(0)), dim=1
    )
    incumbent_cost_tensor = (
        feature_costs[incumbent_row]
        if incumbent_row is not None
        else torch.tensor(math.inf, device=features.device)
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
    if excluded_rows:
        eligible[
            torch.tensor(
                excluded_rows,
                dtype=torch.long,
                device=features.device,
            )
        ] = False

    total_costs = feature_costs + config.transition_penalty
    if incumbent_row is not None:
        total_costs[incumbent_row] = feature_costs[incumbent_row]
    total_costs = total_costs.masked_fill(~eligible, math.inf)
    candidate_row_tensor = torch.argmin(total_costs)
    candidate_total_tensor = total_costs[candidate_row_tensor]
    incumbent_is_eligible = (
        incumbent_row is not None
        and bool(eligible[incumbent_row].item())
    )
    if not incumbent_is_eligible:
        selected_row_tensor = candidate_row_tensor
    else:
        assert incumbent_row is not None
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


# ---------------------------------------------------------------------------
# Inertialized native-frame playback.
# ---------------------------------------------------------------------------


def decay_spring_offsets(
    position_offset: torch.Tensor,
    velocity_offset: torch.Tensor,
    *,
    halflife_s: float,
    time_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate Holden's exact critically damped offset spring."""
    if not math.isfinite(halflife_s) or halflife_s <= 0.0:
        raise ContractError("halflife_s must be finite and positive")
    y = (4.0 * math.log(2.0) / (halflife_s + 1.0e-5)) / 2.0
    if time_s.ndim:
        time_s = time_s.reshape(
            tuple(time_s.shape) + (1,) * position_offset.ndim
        )
    j1 = velocity_offset + position_offset * y
    decay = torch.exp(-y * time_s)
    return (
        decay * (position_offset + j1 * time_s),
        decay * (velocity_offset - j1 * y * time_s),
    )


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_inverse(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)), -1)


def _quat_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_to_scaled_axis(q: torch.Tensor) -> torch.Tensor:
    q = _quat_normalize(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    length = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(length, q[..., :1])
    return q[..., 1:] * (angle / length.clamp_min(1e-8))


def _quat_from_scaled_axis(value: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    half = angle / 2.0
    scale = torch.sin(half) / angle.clamp_min(1e-8)
    q = torch.cat((torch.cos(half), value * scale), dim=-1)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1.0
    return _quat_normalize(torch.where(angle < 1e-8, identity, q))


def _rotate_z(values: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    x, y = values[..., 0], values[..., 1]
    while c.ndim < x.ndim:
        c = c.unsqueeze(-1)
        s = s.unsqueeze(-1)
    return torch.stack((c * x - s * y, s * x + c * y, values[..., 2]), -1)


def _damped_adjustment_alpha(halflife_s: float, dt: float) -> float:
    return 1.0 - _takara_fast_negexp(
        (math.log(2.0) * dt) / (halflife_s + 1.0e-5)
    )


@dataclass(frozen=True)
class _RootAdjustment:
    translation_world_xy: torch.Tensor
    yaw_delta: torch.Tensor
    adjustment_distance_m: torch.Tensor
    adjustment_angle_rad: torch.Tensor
    clamp_distance_m: torch.Tensor
    clamp_angle_rad: torch.Tensor


def _bounded_root_adjustment(
    character_position_world_xy: torch.Tensor,
    character_velocity_world_xy: torch.Tensor,
    character_heading_world_yaw: torch.Tensor,
    character_angular_velocity_world: torch.Tensor,
    simulation_position_world_xy: torch.Tensor,
    simulation_heading_world_yaw: torch.Tensor,
    config: MatcherConfig,
) -> _RootAdjustment:
    """Port Takara's velocity-bounded root adjustment followed by clamping."""
    difference = simulation_position_world_xy - character_position_world_xy
    adjustment = difference * _damped_adjustment_alpha(
        config.adjustment_position_halflife_s, config.dt
    )
    adjustment_length = torch.linalg.vector_norm(adjustment)
    maximum = (
        config.adjustment_position_max_ratio
        * torch.linalg.vector_norm(character_velocity_world_xy)
        * config.dt
    )
    adjustment_scale = torch.clamp(
        maximum / adjustment_length.clamp_min(1.0e-8), max=1.0
    )
    adjustment = adjustment * adjustment_scale
    after_adjustment = character_position_world_xy + adjustment

    clamp_difference = after_adjustment - simulation_position_world_xy
    clamp_distance = torch.linalg.vector_norm(clamp_difference)
    clamp_scale = torch.clamp(
        config.clamping_max_distance_m / clamp_distance.clamp_min(1.0e-8),
        max=1.0,
    )
    after_clamp = simulation_position_world_xy + clamp_difference * clamp_scale
    clamp_translation = after_clamp - after_adjustment

    yaw_difference = _wrapped_angle(
        simulation_heading_world_yaw - character_heading_world_yaw
    )
    yaw_adjustment = yaw_difference * _damped_adjustment_alpha(
        config.adjustment_rotation_halflife_s, config.dt
    )
    maximum_yaw = (
        config.adjustment_rotation_max_ratio
        * torch.linalg.vector_norm(character_angular_velocity_world)
        * config.dt
    )
    yaw_adjustment = torch.clamp(
        yaw_adjustment, min=-maximum_yaw, max=maximum_yaw
    )
    after_adjustment_yaw = _wrapped_angle(
        character_heading_world_yaw + yaw_adjustment
    )
    residual_yaw = _wrapped_angle(
        after_adjustment_yaw - simulation_heading_world_yaw
    )
    clamped_residual_yaw = torch.clamp(
        residual_yaw,
        min=-config.clamping_max_angle_rad,
        max=config.clamping_max_angle_rad,
    )
    after_clamp_yaw = _wrapped_angle(
        simulation_heading_world_yaw + clamped_residual_yaw
    )
    yaw_clamp = _wrapped_angle(after_clamp_yaw - after_adjustment_yaw)
    return _RootAdjustment(
        translation_world_xy=adjustment + clamp_translation,
        yaw_delta=_wrapped_angle(after_clamp_yaw - character_heading_world_yaw),
        adjustment_distance_m=torch.linalg.vector_norm(adjustment),
        adjustment_angle_rad=torch.abs(yaw_adjustment),
        clamp_distance_m=torch.linalg.vector_norm(clamp_translation),
        clamp_angle_rad=torch.abs(yaw_clamp),
    )


def _apply_root_adjustment(
    root_position: torch.Tensor,
    root_quaternion: torch.Tensor,
    root_velocity: torch.Tensor,
    root_angular_velocity: torch.Tensor,
    body_position: torch.Tensor,
    body_quaternion: torch.Tensor,
    body_velocity: torch.Tensor,
    body_angular_velocity: torch.Tensor,
    adjustment: _RootAdjustment,
) -> tuple[torch.Tensor, ...]:
    """Apply one rigid horizontal root correction to an entire dense window."""
    pivot = root_position[0].clone()
    yaw = adjustment.yaw_delta
    translation = torch.zeros(3, dtype=root_position.dtype, device=root_position.device)
    translation[:2] = adjustment.translation_world_xy

    adjusted_root_position = _rotate_z(root_position - pivot, yaw) + pivot + translation
    adjusted_body_position = (
        _rotate_z(body_position - pivot.reshape(1, 1, 3), yaw)
        + pivot.reshape(1, 1, 3)
        + translation.reshape(1, 1, 3)
    )
    adjusted_root_velocity = _rotate_z(root_velocity, yaw)
    adjusted_root_angular_velocity = _rotate_z(root_angular_velocity, yaw)
    adjusted_body_velocity = _rotate_z(body_velocity, yaw)
    adjusted_body_angular_velocity = _rotate_z(body_angular_velocity, yaw)
    yaw_quaternion = _quat_from_yaw(yaw)
    adjusted_root_quaternion = _quat_normalize(
        _quat_mul(yaw_quaternion.expand_as(root_quaternion), root_quaternion)
    )
    adjusted_body_quaternion = _quat_normalize(
        _quat_mul(yaw_quaternion.expand_as(body_quaternion), body_quaternion)
    )
    return (
        adjusted_root_position,
        adjusted_root_quaternion,
        adjusted_root_velocity,
        adjusted_root_angular_velocity,
        adjusted_body_position,
        adjusted_body_quaternion,
        adjusted_body_velocity,
        adjusted_body_angular_velocity,
    )


def _rotate_offsets(offsets: "_Offsets", yaw: torch.Tensor) -> "_Offsets":
    return _Offsets(
        joint_position=offsets.joint_position,
        joint_velocity=offsets.joint_velocity,
        root_position=_rotate_z(offsets.root_position, yaw),
        root_linear_velocity=_rotate_z(offsets.root_linear_velocity, yaw),
        root_rotation_axis=_rotate_z(offsets.root_rotation_axis, yaw),
        root_angular_velocity=_rotate_z(offsets.root_angular_velocity, yaw),
        body_position=_rotate_z(offsets.body_position, yaw),
        body_velocity=_rotate_z(offsets.body_velocity, yaw),
        body_rotation_axis=_rotate_z(offsets.body_rotation_axis, yaw),
        body_angular_velocity=_rotate_z(offsets.body_angular_velocity, yaw),
        elapsed_s=offsets.elapsed_s,
    )


@dataclass(frozen=True)
class MotionMatchDiagnostics:
    sequence: int
    selected_clip_path: str
    selected_frame: int
    incumbent_cost: float
    selected_feature_cost: float
    selected_total_cost: float
    searched: bool
    transitioned: bool
    force_search_reason: str | None
    search_time_ns: int | None
    step_time_ns: int


@dataclass(frozen=True)
class MotionMatchResult:
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    root_position_world: torch.Tensor
    root_orientation_world_wxyz: torch.Tensor
    body_position_world: torch.Tensor
    body_orientation_world_wxyz: torch.Tensor
    body_linear_velocity_world: torch.Tensor
    body_angular_velocity_world: torch.Tensor
    dense_joint_position_window: torch.Tensor
    dense_joint_velocity_window: torch.Tensor
    dense_root_position_window: torch.Tensor
    dense_root_orientation_window_wxyz: torch.Tensor
    dense_body_position_window: torch.Tensor
    joint_position_window: torch.Tensor
    joint_velocity_window: torch.Tensor
    root_position_window: torch.Tensor
    root_orientation_window_wxyz: torch.Tensor
    applied_command_velocity_world_xy: torch.Tensor
    applied_command_heading_world_yaw: torch.Tensor
    simulation_ball_position_world_xy: torch.Tensor
    simulation_ball_velocity_world_xy: torch.Tensor
    simulation_ball_heading_world_yaw: torch.Tensor
    simulation_ball_heading_velocity_rad_s: torch.Tensor
    root_adjustment_distance_m: torch.Tensor
    root_adjustment_angle_rad: torch.Tensor
    root_clamp_distance_m: torch.Tensor
    root_clamp_angle_rad: torch.Tensor
    diagnostics: MotionMatchDiagnostics


@dataclass(frozen=True)
class PreparedMotionMatch:
    result: MotionMatchResult
    selected_row: int
    _owner_token: object
    _base_sequence: int
    _next_state: object


@dataclass(frozen=True)
class ForcedMotionAlignment:
    """World placement used for one externally selected source frame.

    Ordinary flat motion matching keeps the historical root-to-root placement.
    A terrain-aware selector instead supplies the rigid transform that maps the
    *recorded terrain* onto the target terrain.  Keeping this transform in the
    matcher state makes every successor frame stay registered to the same
    staircase, including its vertical placement.
    """

    yaw_offset_rad: float
    translation_world_xyz: tuple[float, float, float]
    synchronize_simulation_character: bool = True

    def __post_init__(self) -> None:
        values = (self.yaw_offset_rad, *self.translation_world_xyz)
        if len(self.translation_world_xyz) != 3 or not all(
            math.isfinite(float(value)) for value in values
        ):
            raise ContractError("forced motion alignment must be finite 3D yaw/translation")
        if type(self.synchronize_simulation_character) is not bool:
            raise ContractError("simulation-character synchronization must be boolean")


@dataclass(frozen=True)
class _DeviceClip:
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    body_position: torch.Tensor
    body_quaternion: torch.Tensor
    body_linear_velocity: torch.Tensor
    body_angular_velocity: torch.Tensor


@dataclass(frozen=True)
class _Offsets:
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    root_position: torch.Tensor
    root_linear_velocity: torch.Tensor
    root_rotation_axis: torch.Tensor
    root_angular_velocity: torch.Tensor
    body_position: torch.Tensor
    body_velocity: torch.Tensor
    body_rotation_axis: torch.Tensor
    body_angular_velocity: torch.Tensor
    elapsed_s: float


@dataclass(frozen=True)
class _MatcherState:
    sequence: int
    clip_index: int
    frame_index: int
    yaw_offset: torch.Tensor
    translation_xy: torch.Tensor
    height_offset: torch.Tensor
    shaped_velocity: torch.Tensor
    shaped_heading: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    root_position: torch.Tensor
    root_quaternion: torch.Tensor
    root_linear_velocity: torch.Tensor
    root_angular_velocity: torch.Tensor
    feature_body_position: torch.Tensor
    feature_body_quaternion: torch.Tensor
    feature_body_velocity: torch.Tensor
    feature_body_angular_velocity: torch.Tensor
    offsets: _Offsets
    simulation_position: torch.Tensor
    simulation_velocity: torch.Tensor
    simulation_acceleration: torch.Tensor
    simulation_heading: torch.Tensor
    simulation_heading_velocity: torch.Tensor


def _copy_result(result: MotionMatchResult) -> MotionMatchResult:
    values = {
        name: getattr(result, name).clone()
        for name in (
            "joint_position",
            "joint_velocity",
            "root_position_world",
            "root_orientation_world_wxyz",
            "body_position_world",
            "body_orientation_world_wxyz",
            "body_linear_velocity_world",
            "body_angular_velocity_world",
            "dense_joint_position_window",
            "dense_joint_velocity_window",
            "dense_root_position_window",
            "dense_root_orientation_window_wxyz",
            "dense_body_position_window",
            "joint_position_window",
            "joint_velocity_window",
            "root_position_window",
            "root_orientation_window_wxyz",
            "applied_command_velocity_world_xy",
            "applied_command_heading_world_yaw",
            "simulation_ball_position_world_xy",
            "simulation_ball_velocity_world_xy",
            "simulation_ball_heading_world_yaw",
            "simulation_ball_heading_velocity_rad_s",
            "root_adjustment_distance_m",
            "root_adjustment_angle_rad",
            "root_clamp_distance_m",
            "root_clamp_angle_rad",
        )
    }
    return MotionMatchResult(**values, diagnostics=result.diagnostics)


class TorchMotionMatcher:
    """One in-process native Takara matcher with transactional state updates."""

    def __init__(
        self,
        folder: MotionFolder,
        database: TorchMotionDatabase,
        clips: tuple[_DeviceClip, ...],
        config: MatcherConfig,
    ) -> None:
        self.folder = folder
        self.database = database
        self.device = database.device
        self.config = config
        self._clips = clips
        self._row_sources = tuple(
            zip(
                database._search_clip_index.cpu().tolist(),
                database._search_frame_index.cpu().tolist(),
            )
        )
        self._owner_token = object()
        self._state: _MatcherState | None = None
        self._supported_reset_cache: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ] | None = None

    @classmethod
    def from_folder(
        cls,
        motions_dir: str | Path,
        *,
        device: str = "auto",
        config: MatcherConfig = MatcherConfig(),
    ) -> "TorchMotionMatcher":
        resolved = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ContractError("CUDA motion matcher requested but unavailable")
        folder = MotionFolder.load(motions_dir)
        return cls.from_motion_folder(folder, device=resolved, config=config)

    @classmethod
    def from_motion_folder(
        cls,
        folder: MotionFolder,
        *,
        device: "str | torch.device" = "auto",
        config: MatcherConfig = MatcherConfig(),
    ) -> "TorchMotionMatcher":
        """Build a matcher from an already assembled in-memory motion folder."""

        resolved = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ContractError("CUDA motion matcher requested but unavailable")
        if not isinstance(folder, MotionFolder):
            raise ContractError("folder must be a MotionFolder")
        database = TorchMotionDatabase.from_folder(
            folder,
            device=resolved,
            group_weights=dict(config.feature_weight_overrides),
            max_joint_step_rad=config.max_source_joint_step_rad,
        )
        clips = tuple(
            _DeviceClip(
                joint_position=torch.tensor(
                    clip.joint_position, dtype=torch.float32, device=resolved
                ),
                joint_velocity=torch.tensor(
                    clip.joint_velocity, dtype=torch.float32, device=resolved
                ),
                body_position=torch.tensor(
                    clip.body_position_world, dtype=torch.float32, device=resolved
                ),
                body_quaternion=torch.tensor(
                    clip.body_quaternion_world_wxyz,
                    dtype=torch.float32,
                    device=resolved,
                ),
                body_linear_velocity=torch.tensor(
                    clip.body_linear_velocity_world,
                    dtype=torch.float32,
                    device=resolved,
                ),
                body_angular_velocity=torch.tensor(
                    clip.body_angular_velocity_world,
                    dtype=torch.float32,
                    device=resolved,
                ),
            )
            for clip in folder.clips
        )
        return cls(folder, database, clips, config)

    @property
    def motion_inventory_sha256(self) -> str:
        return self.folder.inventory_sha256

    @property
    def current_root_world_yaw(self) -> float:
        """Current generated root yaw for robot-relative operator commands."""
        if self._state is None:
            raise ContractError("matcher must be reset before reading root yaw")
        return float(_quat_yaw(self._state.root_quaternion).detach().cpu().item())

    def _source_for_row(self, row: int) -> tuple[int, int]:
        pair = self._row_sources[row]
        return int(pair[0]), int(pair[1])

    def _aligned_targets(
        self,
        clip_index: int,
        frame_index: int,
        yaw_offset: torch.Tensor,
        translation_xy: torch.Tensor,
        height_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        clip = self._clips[clip_index]
        sl = slice(frame_index, frame_index + 46)
        root = self.folder.layout.root_body_index
        joint_p = clip.joint_position[sl]
        joint_v = clip.joint_velocity[sl]
        body_p = _rotate_z(clip.body_position[sl], yaw_offset)
        body_p = body_p.clone()
        body_p[..., :2] += translation_xy
        body_p[..., 2] += height_offset
        body_v = _rotate_z(clip.body_linear_velocity[sl], yaw_offset)
        yaw_q = _quat_from_yaw(yaw_offset)
        body_q = _quat_normalize(
            _quat_mul(
                yaw_q.expand(46, clip.body_quaternion.shape[1], 4),
                clip.body_quaternion[sl],
            )
        )
        body_w = _rotate_z(clip.body_angular_velocity[sl], yaw_offset)
        root_p = body_p[:, root]
        root_q = body_q[:, root]
        root_v = body_v[:, root]
        root_w = body_w[:, root]
        return (
            joint_p,
            joint_v,
            root_p,
            root_q,
            root_v,
            root_w,
            body_p,
            body_q,
            body_v,
            body_w,
        )

    def _make_result(
        self,
        *,
        sequence: int,
        clip_index: int,
        frame_index: int,
        decision: SearchDecision,
        force_reason: str | None,
        search_time_ns: int | None,
        step_start_ns: int,
        dense_joint_p: torch.Tensor,
        dense_joint_v: torch.Tensor,
        dense_root_p: torch.Tensor,
        dense_root_q: torch.Tensor,
        dense_body_p: torch.Tensor,
        dense_body_q: torch.Tensor,
        dense_body_v: torch.Tensor,
        dense_body_w: torch.Tensor,
        applied_command_velocity: torch.Tensor,
        applied_command_heading: torch.Tensor,
        simulation_ball_position: torch.Tensor,
        simulation_ball_velocity: torch.Tensor,
        simulation_ball_heading: torch.Tensor,
        simulation_ball_heading_velocity: torch.Tensor,
        root_adjustment_distance: torch.Tensor,
        root_adjustment_angle: torch.Tensor,
        root_clamp_distance: torch.Tensor,
        root_clamp_angle: torch.Tensor,
    ) -> MotionMatchResult:
        sample = torch.arange(0, 46, 5, device=self.device)
        diagnostics = MotionMatchDiagnostics(
            sequence=sequence,
            selected_clip_path=self.folder.clips[clip_index].relative_path,
            selected_frame=frame_index,
            incumbent_cost=decision.incumbent_cost,
            selected_feature_cost=decision.selected_feature_cost,
            selected_total_cost=decision.selected_total_cost,
            searched=decision.searched,
            transitioned=decision.transitioned,
            force_search_reason=force_reason,
            search_time_ns=search_time_ns,
            step_time_ns=time.perf_counter_ns() - step_start_ns,
        )
        result = MotionMatchResult(
            joint_position=dense_joint_p[0].clone(),
            joint_velocity=dense_joint_v[0].clone(),
            root_position_world=dense_root_p[0].clone(),
            root_orientation_world_wxyz=dense_root_q[0].clone(),
            body_position_world=dense_body_p[0].clone(),
            body_orientation_world_wxyz=dense_body_q[0].clone(),
            body_linear_velocity_world=dense_body_v[0].clone(),
            body_angular_velocity_world=dense_body_w[0].clone(),
            dense_joint_position_window=dense_joint_p.clone(),
            dense_joint_velocity_window=dense_joint_v.clone(),
            dense_root_position_window=dense_root_p.clone(),
            dense_root_orientation_window_wxyz=dense_root_q.clone(),
            dense_body_position_window=dense_body_p.clone(),
            joint_position_window=dense_joint_p[sample].clone(),
            joint_velocity_window=dense_joint_v[sample].clone(),
            root_position_window=dense_root_p[sample].clone(),
            root_orientation_window_wxyz=dense_root_q[sample].clone(),
            applied_command_velocity_world_xy=applied_command_velocity.clone(),
            applied_command_heading_world_yaw=applied_command_heading.clone(),
            simulation_ball_position_world_xy=simulation_ball_position.clone(),
            simulation_ball_velocity_world_xy=simulation_ball_velocity.clone(),
            simulation_ball_heading_world_yaw=simulation_ball_heading.clone(),
            simulation_ball_heading_velocity_rad_s=(
                simulation_ball_heading_velocity.clone()
            ),
            root_adjustment_distance_m=root_adjustment_distance.clone(),
            root_adjustment_angle_rad=root_adjustment_angle.clone(),
            root_clamp_distance_m=root_clamp_distance.clone(),
            root_clamp_angle_rad=root_clamp_angle.clone(),
            diagnostics=diagnostics,
        )
        finite = torch.isfinite(dense_joint_p).all()
        finite &= torch.isfinite(dense_joint_v).all()
        finite &= torch.isfinite(dense_root_p).all()
        finite &= torch.isfinite(dense_root_q).all()
        finite &= torch.isfinite(dense_body_p).all()
        finite &= torch.isfinite(dense_body_q).all()
        finite &= torch.isfinite(dense_body_v).all()
        finite &= torch.isfinite(dense_body_w).all()
        finite &= torch.isfinite(applied_command_velocity).all()
        finite &= torch.isfinite(applied_command_heading).all()
        finite &= torch.isfinite(simulation_ball_position).all()
        finite &= torch.isfinite(simulation_ball_velocity).all()
        finite &= torch.isfinite(simulation_ball_heading).all()
        finite &= torch.isfinite(simulation_ball_heading_velocity).all()
        finite &= torch.isfinite(root_adjustment_distance).all()
        finite &= torch.isfinite(root_adjustment_angle).all()
        finite &= torch.isfinite(root_clamp_distance).all()
        finite &= torch.isfinite(root_clamp_angle).all()
        unit = torch.all(
            torch.abs(torch.linalg.vector_norm(dense_root_q, dim=-1) - 1.0)
            <= 1e-5
        )
        unit &= torch.all(
            torch.abs(torch.linalg.vector_norm(dense_body_q, dim=-1) - 1.0)
            <= 1e-5
        )
        if not bool((finite & unit).item()):
            raise ContractError("motion match produced invalid dense output")
        return result

    def _supported_reset_arrays(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        cache = self._supported_reset_cache
        if cache is not None:
            return cache
        rows = self.database.feature_shape[0]
        joint = torch.empty(
            (rows, 29), dtype=torch.float32, device=self.device
        )
        feet = torch.empty(
            (rows, 2, 3), dtype=torch.float32, device=self.device
        )
        foot_speed = torch.empty(
            (rows, 2), dtype=torch.float32, device=self.device
        )
        root_velocity = torch.empty(
            (rows, 2), dtype=torch.float32, device=self.device
        )
        clip_rows = self.database._search_clip_index
        frame_rows = self.database._search_frame_index
        root_body = self.folder.layout.root_body_index
        foot_bodies = (
            self.folder.layout.left_foot_body_index,
            self.folder.layout.right_foot_body_index,
        )
        for clip_index, clip in enumerate(self._clips):
            row = torch.nonzero(
                clip_rows == clip_index, as_tuple=False
            ).flatten()
            if row.numel() == 0:
                continue
            frame = frame_rows[row]
            body_position = clip.body_position[frame]
            body_velocity = clip.body_linear_velocity[frame]
            root_position = body_position[:, root_body]
            root_quaternion = clip.body_quaternion[frame, root_body]
            root_yaw = _quat_yaw(root_quaternion)
            joint[row] = clip.joint_position[frame]
            feet[row] = _rotate_z(
                body_position[:, foot_bodies] - root_position[:, None],
                -root_yaw,
            )
            foot_speed[row] = torch.linalg.vector_norm(
                body_velocity[:, foot_bodies], dim=-1
            )
            root_velocity[row] = _rotate_z(
                body_velocity[:, root_body], -root_yaw
            )[:, :2]
        cache = (joint, feet, foot_speed, root_velocity)
        self._supported_reset_cache = cache
        return cache

    def select_supported_reset_row(
        self,
        *,
        joint_position: object,
        feet_position_root_local: object,
        support_contact: object,
        root_velocity_local_xy: object,
        maximum_support_foot_speed_mps: float = 0.18,
    ) -> int:
        """Choose a flat source phase compatible with a terrain landing.

        This is an initialization query, not an online environment input.  It
        compares root-local support geometry, joint phase, and local root
        velocity against every searchable clean flat frame so the subsequent
        inertialization does not begin from an arbitrary reset pose.
        """

        joint_target = torch.as_tensor(
            joint_position, dtype=torch.float32, device=self.device
        )
        feet_target = torch.as_tensor(
            feet_position_root_local,
            dtype=torch.float32,
            device=self.device,
        )
        contact = torch.as_tensor(
            support_contact, dtype=torch.bool, device=self.device
        )
        velocity_target = torch.as_tensor(
            root_velocity_local_xy,
            dtype=torch.float32,
            device=self.device,
        )
        speed_limit = float(maximum_support_foot_speed_mps)
        if (
            joint_target.shape != (29,)
            or feet_target.shape != (2, 3)
            or contact.shape != (2,)
            or not bool(torch.any(contact).item())
            or velocity_target.shape != (2,)
            or not all(
                bool(torch.isfinite(value).all().item())
                for value in (
                    joint_target,
                    feet_target,
                    velocity_target,
                )
            )
            or not math.isfinite(speed_limit)
            or speed_limit <= 0.0
        ):
            raise ContractError("supported reset query is invalid")
        joint, feet, foot_speed, root_velocity = (
            self._supported_reset_arrays()
        )
        foot_error = torch.linalg.vector_norm(
            feet[:, contact] - feet_target[contact], dim=-1
        ).amax(dim=1)
        joint_error = torch.sqrt(
            torch.mean(
                torch.square(joint - joint_target[None]),
                dim=1,
            )
        )
        support_speed = foot_speed[:, contact].amax(dim=1)
        velocity_error = torch.linalg.vector_norm(
            root_velocity - velocity_target[None], dim=1
        )
        cost = (
            4.0 * foot_error
            + joint_error
            + 2.0 * torch.clamp_min(
                support_speed - speed_limit, 0.0
            )
            + 0.5 * velocity_error
        )
        if not bool(torch.isfinite(cost).all().item()):
            raise ContractError("supported reset costs are non-finite")
        return int(torch.argmin(cost).item())

    def source_lead_row(self, row: int, *, lead_frames: int) -> int:
        """Return the nearest searchable row before one matched phase."""

        return self.source_lead_rows(row, lead_frames=lead_frames)[0]

    def source_lead_rows(
        self, row: int, *, lead_frames: int
    ) -> tuple[int, ...]:
        """Return the contiguous searchable lead-in through a matched phase."""

        if (
            type(row) is not int
            or not 0 <= row < self.database.feature_shape[0]
            or type(lead_frames) is not int
            or lead_frames < 0
        ):
            raise ContractError("source lead query is invalid")
        clip_index, frame_index = self._source_for_row(row)
        target = max(0, frame_index - lead_frames)
        rows: list[int] = []
        for frame in range(target, frame_index + 1):
            candidate = self.database.row_for_source(clip_index, frame)
            if candidate is not None:
                rows.append(int(candidate))
        if not rows:
            return (row,)
        if rows[-1] != row:
            rows.append(row)
        return tuple(rows)

    def reset_to_row(
        self,
        row: int,
        *,
        alignment: ForcedMotionAlignment | None = None,
    ) -> MotionMatchResult:
        if (
            type(row) is not int
            or not 0 <= row < self.database.feature_shape[0]
        ):
            raise ContractError("reset row is outside the motion database")
        step_start = time.perf_counter_ns()
        clip_index, frame_index = self._source_for_row(row)
        clip = self._clips[clip_index]
        root = self.folder.layout.root_body_index
        source_root = clip.body_position[frame_index, root]
        source_yaw = _quat_yaw(clip.body_quaternion[frame_index, root])
        if alignment is None:
            yaw_offset = -source_yaw
            rotated_root = _rotate_z(source_root, yaw_offset)
            translation = -rotated_root[:2]
            height_offset = torch.zeros((), device=self.device)
        else:
            yaw_offset = torch.tensor(
                alignment.yaw_offset_rad,
                dtype=torch.float32,
                device=self.device,
            )
            translation_xyz = torch.tensor(
                alignment.translation_world_xyz,
                dtype=torch.float32,
                device=self.device,
            )
            translation = translation_xyz[:2]
            height_offset = translation_xyz[2]
        targets = self._aligned_targets(
            clip_index,
            frame_index,
            yaw_offset,
            translation,
            height_offset,
        )
        jp, jv, rp, rq, rv, rw, bp, bq, bv, bw = targets
        zeros = _Offsets(
            torch.zeros_like(jp[0]),
            torch.zeros_like(jv[0]),
            torch.zeros_like(rp[0]),
            torch.zeros_like(rv[0]),
            torch.zeros(3, device=self.device),
            torch.zeros_like(rw[0]),
            torch.zeros_like(bp[0]),
            torch.zeros_like(bv[0]),
            torch.zeros_like(bp[0]),
            torch.zeros_like(bw[0]),
            0.0,
        )
        decision = SearchDecision(row, None, math.inf, 0.0, 0.0, False, False)
        result = self._make_result(
            sequence=0,
            clip_index=clip_index,
            frame_index=frame_index,
            decision=decision,
            force_reason=None,
            search_time_ns=None,
            step_start_ns=step_start,
            dense_joint_p=jp,
            dense_joint_v=jv,
            dense_root_p=rp,
            dense_root_q=rq,
            dense_body_p=bp,
            dense_body_q=bq,
            dense_body_v=bv,
            dense_body_w=bw,
            applied_command_velocity=torch.zeros(2, device=self.device),
            applied_command_heading=torch.zeros((), device=self.device),
            simulation_ball_position=rp[0, :2],
            simulation_ball_velocity=torch.zeros(2, device=self.device),
            simulation_ball_heading=_quat_yaw(rq[0]),
            simulation_ball_heading_velocity=torch.zeros((), device=self.device),
            root_adjustment_distance=torch.zeros((), device=self.device),
            root_adjustment_angle=torch.zeros((), device=self.device),
            root_clamp_distance=torch.zeros((), device=self.device),
            root_clamp_angle=torch.zeros((), device=self.device),
        )
        self._state = _MatcherState(
            sequence=0,
            clip_index=clip_index,
            frame_index=frame_index,
            yaw_offset=yaw_offset,
            translation_xy=translation,
            height_offset=height_offset,
            shaped_velocity=torch.zeros(2, device=self.device),
            shaped_heading=torch.zeros((), device=self.device),
            joint_position=jp[0].clone(),
            joint_velocity=jv[0].clone(),
            root_position=rp[0].clone(),
            root_quaternion=rq[0].clone(),
            root_linear_velocity=rv[0].clone(),
            root_angular_velocity=rw[0].clone(),
            feature_body_position=bp[0].clone(),
            feature_body_quaternion=bq[0].clone(),
            feature_body_velocity=bv[0].clone(),
            feature_body_angular_velocity=bw[0].clone(),
            offsets=zeros,
            simulation_position=rp[0, :2].clone(),
            simulation_velocity=torch.zeros(2, device=self.device),
            simulation_acceleration=torch.zeros(2, device=self.device),
            simulation_heading=_quat_yaw(rq[0]).clone(),
            simulation_heading_velocity=torch.zeros((), device=self.device),
        )
        return _copy_result(result)

    def reset(self) -> MotionMatchResult:
        return self.reset_to_row(self.database.reset_row)

    def prepare_step(
        self,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
        *,
        dt: float = 0.02,
        excluded_rows: tuple[int, ...] = (),
        forced_row: int | None = None,
        forced_alignment: ForcedMotionAlignment | None = None,
    ) -> PreparedMotionMatch:
        step_start = time.perf_counter_ns()
        state = self._state
        if state is None:
            raise ContractError("reset must be called before prepare_step")
        if not math.isfinite(dt) or abs(dt - self.config.dt) > 1e-12:
            raise ContractError(f"dt must equal {self.config.dt}")
        if (
            forced_row is not None
            and (
                type(forced_row) is not int
                or not 0 <= forced_row < self.database.feature_shape[0]
                or bool(excluded_rows)
            )
        ):
            raise ContractError(
                "forced_row must be a database row and cannot be combined "
                "with exclusions"
            )
        if forced_alignment is not None and forced_row is None:
            raise ContractError("forced_alignment requires forced_row")
        requested_v = torch.tensor(
            velocity_world_xy, dtype=torch.float32, device=self.device
        )
        requested_h = torch.tensor(
            heading_world_yaw, dtype=torch.float32, device=self.device
        )
        successor = self.database.row_for_source(
            state.clip_index, state.frame_index + 1
        )
        if self.config.trajectory_model == "takara_ball":
            ball_prediction = predict_takara_ball_trajectory(
                state.simulation_position,
                state.simulation_velocity,
                state.simulation_acceleration,
                state.simulation_heading,
                state.simulation_heading_velocity,
                requested_v,
                requested_h,
                has_valid_successor=successor is not None,
                config=self.config,
            )
            shaped = ball_prediction.command
            next_simulation_position = ball_prediction.next_position_world_xy
            next_simulation_velocity = ball_prediction.next_velocity_world_xy
            next_simulation_acceleration = (
                ball_prediction.next_acceleration_world_xy
            )
            next_simulation_heading = ball_prediction.next_heading_world_yaw
            next_simulation_heading_velocity = (
                ball_prediction.next_heading_velocity_rad_s
            )
        else:
            shaped = predict_command_trajectory(
                state.root_position[:2],
                state.shaped_velocity,
                state.shaped_heading,
                requested_v,
                requested_h,
                has_valid_successor=successor is not None,
                config=self.config,
            )
            next_simulation_position = (
                state.simulation_position
                + shaped.velocity_world_xy * self.config.dt
            )
            next_simulation_velocity = shaped.velocity_world_xy
            next_simulation_acceleration = torch.zeros_like(
                state.simulation_acceleration
            )
            next_simulation_heading = shaped.heading_world_yaw
            next_simulation_heading_velocity = _wrapped_angle(
                shaped.heading_world_yaw - state.simulation_heading
            ) / self.config.dt
        feature_state = GeneratedFeatureState(
            root_position_world=state.root_position,
            root_orientation_world_wxyz=state.root_quaternion,
            root_linear_velocity_world=state.root_linear_velocity,
            left_foot_position_world=state.feature_body_position[
                self.folder.layout.left_foot_body_index
            ],
            right_foot_position_world=state.feature_body_position[
                self.folder.layout.right_foot_body_index
            ],
            left_foot_velocity_world=state.feature_body_velocity[
                self.folder.layout.left_foot_body_index
            ],
            right_foot_velocity_world=state.feature_body_velocity[
                self.folder.layout.right_foot_body_index
            ],
        )
        query = self.database.normalization.normalize(
            extract_query_features(feature_state, shaped.trajectory)
        )
        search = (
            search_is_due(
                state.sequence, shaped.force_search, self.config
            )
            or bool(excluded_rows)
            or forced_row is not None
        )
        search_start = time.perf_counter_ns()
        if forced_row is None:
            decision = select_exact_candidate(
                self.database,
                query,
                current_clip_index=state.clip_index,
                current_frame_index=state.frame_index,
                incumbent_row=successor,
                search=search,
                config=self.config,
                excluded_rows=excluded_rows,
            )
        else:
            features = self.database._search_features
            target_cost = float(
                torch.sum(
                    torch.square(features[forced_row] - query)
                ).item()
            )
            incumbent_cost = (
                math.inf
                if successor is None
                else float(
                    torch.sum(
                        torch.square(features[successor] - query)
                    ).item()
                )
            )
            transitioned = successor is None or forced_row != successor
            decision = SearchDecision(
                selected_row=forced_row,
                incumbent_row=successor,
                incumbent_cost=incumbent_cost,
                selected_feature_cost=target_cost,
                selected_total_cost=(
                    target_cost
                    + (
                        self.config.transition_penalty
                        if transitioned
                        else 0.0
                    )
                ),
                searched=True,
                transitioned=transitioned,
            )
        search_time = time.perf_counter_ns() - search_start if search else None
        clip_index, frame_index = self._source_for_row(decision.selected_row)
        transitioned = decision.transitioned or successor is None
        if forced_alignment is not None:
            yaw_offset = torch.tensor(
                forced_alignment.yaw_offset_rad,
                dtype=torch.float32,
                device=self.device,
            )
            translation_xyz = torch.tensor(
                forced_alignment.translation_world_xyz,
                dtype=torch.float32,
                device=self.device,
            )
            translation = translation_xyz[:2]
            height_offset = translation_xyz[2]
        elif transitioned:
            clip = self._clips[clip_index]
            root = self.folder.layout.root_body_index
            source_pos = clip.body_position[frame_index, root]
            source_yaw = _quat_yaw(clip.body_quaternion[frame_index, root])
            if self.config.trajectory_model == "takara_ball":
                yaw_offset = _wrapped_angle(
                    _quat_yaw(state.root_quaternion) - source_yaw
                )
                desired_xy = state.root_position[:2]
            else:
                yaw_offset = _wrapped_angle(shaped.heading_world_yaw - source_yaw)
                desired_xy = (
                    state.root_position[:2]
                    + shaped.velocity_world_xy * self.config.dt
                )
            rotated = _rotate_z(source_pos, yaw_offset)
            translation = desired_xy - rotated[:2]
            height_offset = torch.zeros((), device=self.device)
        else:
            yaw_offset = state.yaw_offset
            translation = state.translation_xy
            height_offset = state.height_offset
        targets = self._aligned_targets(
            clip_index,
            frame_index,
            yaw_offset,
            translation,
            height_offset,
        )
        jp, jv, rp, rq, rv, rw, bp, bq, bv, bw = targets
        if transitioned:
            offsets = _Offsets(
                state.joint_position - jp[0],
                state.joint_velocity - jv[0],
                state.root_position - rp[0],
                state.root_linear_velocity - rv[0],
                _quat_to_scaled_axis(
                    _quat_mul(state.root_quaternion, _quat_inverse(rq[0]))
                ),
                state.root_angular_velocity - rw[0],
                state.feature_body_position - bp[0],
                state.feature_body_velocity - bv[0],
                _quat_to_scaled_axis(
                    _quat_mul(
                        state.feature_body_quaternion,
                        _quat_inverse(bq[0]),
                    )
                ),
                state.feature_body_angular_velocity - bw[0],
                self.config.dt,
            )
        else:
            offsets = _Offsets(
                state.offsets.joint_position,
                state.offsets.joint_velocity,
                state.offsets.root_position,
                state.offsets.root_linear_velocity,
                state.offsets.root_rotation_axis,
                state.offsets.root_angular_velocity,
                state.offsets.body_position,
                state.offsets.body_velocity,
                state.offsets.body_rotation_axis,
                state.offsets.body_angular_velocity,
                state.offsets.elapsed_s + self.config.dt,
            )
        times = (
            torch.arange(46, device=self.device, dtype=torch.float32)
            * self.config.dt
            + offsets.elapsed_s
        )
        jpo, jvo = decay_spring_offsets(
            offsets.joint_position, offsets.joint_velocity,
            halflife_s=self.config.inertialization_halflife_s, time_s=times,
        )
        rpo, rvo = decay_spring_offsets(
            offsets.root_position, offsets.root_linear_velocity,
            halflife_s=self.config.inertialization_halflife_s, time_s=times,
        )
        bao, bvo = decay_spring_offsets(
            offsets.body_position, offsets.body_velocity,
            halflife_s=self.config.inertialization_halflife_s, time_s=times,
        )
        qao, qwo = decay_spring_offsets(
            offsets.root_rotation_axis, offsets.root_angular_velocity,
            halflife_s=self.config.inertialization_halflife_s, time_s=times,
        )
        bqao, bqwo = decay_spring_offsets(
            offsets.body_rotation_axis, offsets.body_angular_velocity,
            halflife_s=self.config.inertialization_halflife_s, time_s=times,
        )
        dense_jp = jp + jpo
        dense_jv = jv + jvo
        dense_rp = rp + rpo
        dense_rv = rv + rvo
        dense_bp = bp + bao
        dense_bv = bv + bvo
        dense_rq = _quat_normalize(
            _quat_mul(_quat_from_scaled_axis(qao), rq)
        )
        dense_rw = rw + qwo
        dense_bq = _quat_normalize(
            _quat_mul(_quat_from_scaled_axis(bqao), bq)
        )
        dense_bw = bw + bqwo
        root_adjustment = _RootAdjustment(
            translation_world_xy=torch.zeros(2, device=self.device),
            yaw_delta=torch.zeros((), device=self.device),
            adjustment_distance_m=torch.zeros((), device=self.device),
            adjustment_angle_rad=torch.zeros((), device=self.device),
            clamp_distance_m=torch.zeros((), device=self.device),
            clamp_angle_rad=torch.zeros((), device=self.device),
        )
        if (
            self.config.trajectory_model == "takara_ball"
            and forced_alignment is None
        ):
            root_adjustment = _bounded_root_adjustment(
                dense_rp[0, :2],
                dense_rv[0, :2],
                _quat_yaw(dense_rq[0]),
                dense_rw[0],
                next_simulation_position,
                next_simulation_heading,
                self.config,
            )
            pivot = dense_rp[0].clone()
            (
                dense_rp,
                dense_rq,
                dense_rv,
                dense_rw,
                dense_bp,
                dense_bq,
                dense_bv,
                dense_bw,
            ) = _apply_root_adjustment(
                dense_rp,
                dense_rq,
                dense_rv,
                dense_rw,
                dense_bp,
                dense_bq,
                dense_bv,
                dense_bw,
                root_adjustment,
            )
            translation_3d = torch.zeros(3, device=self.device)
            translation_3d[:2] = translation
            translation = (
                _rotate_z(
                    translation_3d - pivot,
                    root_adjustment.yaw_delta,
                )
                + pivot
            )[:2] + root_adjustment.translation_world_xy
            yaw_offset = _wrapped_angle(
                yaw_offset + root_adjustment.yaw_delta
            )
            offsets = _rotate_offsets(offsets, root_adjustment.yaw_delta)
        elif (
            self.config.trajectory_model == "takara_ball"
            and forced_alignment is not None
            and forced_alignment.synchronize_simulation_character
        ):
            # Terrain registration has priority over the ordinary horizontal
            # root correction.  Keep the internal simulation character
            # co-located with the terrain-locked generated root so a speed
            # mismatch cannot accumulate and snap forward at the landing.
            next_simulation_position = dense_rp[0, :2].clone()
            next_simulation_velocity = dense_rv[0, :2].clone()
            next_simulation_acceleration = torch.zeros_like(
                next_simulation_acceleration
            )
            next_simulation_heading = _quat_yaw(dense_rq[0]).clone()
            next_simulation_heading_velocity = dense_rw[0, 2].clone()
        elif (
            self.config.trajectory_model == "takara_ball"
            and forced_alignment is not None
        ):
            # Local motion snippets still require an exact rigid placement,
            # but their command board remains joystick-driven.  Recenter the
            # board at the published character while retaining its filtered
            # velocity, acceleration, and requested heading.  This prevents
            # both failure modes: copying a stopped source velocity into the
            # command, and letting a persistent global ball run metres ahead
            # while terrain registration disables ordinary root adjustment.
            next_simulation_position = dense_rp[0, :2].clone()
        force_reason = "terrain_phase_plan" if forced_row is not None else None
        if forced_row is None and successor is None:
            force_reason = "clip_end"
        elif forced_row is None and shaped.force_search:
            force_reason = "command_transition"
        result = self._make_result(
            sequence=state.sequence + 1,
            clip_index=clip_index,
            frame_index=frame_index,
            decision=decision,
            force_reason=force_reason,
            search_time_ns=search_time,
            step_start_ns=step_start,
            dense_joint_p=dense_jp,
            dense_joint_v=dense_jv,
            dense_root_p=dense_rp,
            dense_root_q=dense_rq,
            dense_body_p=dense_bp,
            dense_body_q=dense_bq,
            dense_body_v=dense_bv,
            dense_body_w=dense_bw,
            applied_command_velocity=shaped.velocity_world_xy,
            applied_command_heading=shaped.heading_world_yaw,
            simulation_ball_position=next_simulation_position,
            simulation_ball_velocity=next_simulation_velocity,
            simulation_ball_heading=next_simulation_heading,
            simulation_ball_heading_velocity=next_simulation_heading_velocity,
            root_adjustment_distance=root_adjustment.adjustment_distance_m,
            root_adjustment_angle=root_adjustment.adjustment_angle_rad,
            root_clamp_distance=root_adjustment.clamp_distance_m,
            root_clamp_angle=root_adjustment.clamp_angle_rad,
        )
        next_state = _MatcherState(
            sequence=state.sequence + 1,
            clip_index=clip_index,
            frame_index=frame_index,
            yaw_offset=yaw_offset,
            translation_xy=translation,
            height_offset=height_offset,
            shaped_velocity=shaped.velocity_world_xy.clone(),
            shaped_heading=shaped.heading_world_yaw.clone(),
            joint_position=dense_jp[0].clone(),
            joint_velocity=dense_jv[0].clone(),
            root_position=dense_rp[0].clone(),
            root_quaternion=dense_rq[0].clone(),
            root_linear_velocity=dense_rv[0].clone(),
            root_angular_velocity=dense_rw[0].clone(),
            feature_body_position=dense_bp[0].clone(),
            feature_body_quaternion=dense_bq[0].clone(),
            feature_body_velocity=dense_bv[0].clone(),
            feature_body_angular_velocity=dense_bw[0].clone(),
            offsets=offsets,
            simulation_position=next_simulation_position.clone(),
            simulation_velocity=next_simulation_velocity.clone(),
            simulation_acceleration=next_simulation_acceleration.clone(),
            simulation_heading=next_simulation_heading.clone(),
            simulation_heading_velocity=next_simulation_heading_velocity.clone(),
        )
        return PreparedMotionMatch(
            _copy_result(result),
            int(decision.selected_row),
            self._owner_token,
            state.sequence,
            next_state,
        )

    def prepare_row(
        self,
        row: int,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
        *,
        dt: float = 0.02,
    ) -> PreparedMotionMatch:
        """Stage a phase-planner-selected row without committing state."""

        return self.prepare_step(
            velocity_world_xy,
            heading_world_yaw,
            dt=dt,
            forced_row=row,
        )

    def prepare_steps(
        self,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
        *,
        maximum_candidates: int,
        dt: float = 0.02,
    ) -> Iterator[PreparedMotionMatch]:
        """Yield distinct candidates ranked from the same matcher state."""

        if (
            type(maximum_candidates) is not int
            or maximum_candidates <= 0
        ):
            raise ContractError("maximum_candidates must be positive")
        excluded: list[int] = []
        limit = min(
            maximum_candidates, int(self.database.feature_shape[0])
        )
        for _rank in range(limit):
            try:
                prepared = self.prepare_step(
                    velocity_world_xy,
                    heading_world_yaw,
                    dt=dt,
                    excluded_rows=tuple(excluded),
                )
            except ContractError as error:
                if (
                    excluded
                    and str(error)
                    == "no valid motion-matching candidate exists"
                ):
                    return
                raise
            excluded.append(int(prepared.selected_row))
            yield prepared

    def commit(self, prepared: PreparedMotionMatch) -> MotionMatchResult:
        state = self._state
        if (
            state is None
            or not isinstance(prepared, PreparedMotionMatch)
            or prepared._owner_token is not self._owner_token
            or prepared._base_sequence != state.sequence
            or not isinstance(prepared._next_state, _MatcherState)
        ):
            raise ContractError("prepared motion match is foreign, stale, or used")
        self._state = prepared._next_state
        return _copy_result(prepared.result)

    def step(
        self,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
        *,
        dt: float = 0.02,
    ) -> MotionMatchResult:
        return self.commit(
            self.prepare_step(velocity_world_xy, heading_world_yaw, dt=dt)
        )
