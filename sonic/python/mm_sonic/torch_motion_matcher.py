"""Bounded operator commands and exact dense Torch motion search."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Protocol

import torch

from .joints import ContractError
from .torch_motion_continuity import (
    TransitionContinuityCosts,
    TransitionContinuityDatabase,
)
from .torch_motion_data import MotionFolder
from .torch_motion_features import (
    CommandTrajectory,
    GeneratedFeatureState,
    SearchFeatureExtension,
    TorchMotionDatabase,
    extract_query_features,
    resolve_torch_device,
    validated_extension_query_row,
)


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
    transition_settle_duration_s: float = 0.0
    transition_settle_penalty: float = 0.0
    transition_joint_position_weight: float = 0.0
    transition_joint_velocity_weight: float = 0.0
    transition_window_jerk_weight: float = 0.0
    transition_window_candidate_count: int = 32


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
    selected_transition_cost: float = 0.0


class EmittedWindowValidator(Protocol):
    """Optional state-dependent validation of one composed output window."""

    def __call__(
        self, feature_body_position_window: torch.Tensor
    ) -> bool:
        ...


def active_transition_penalty(
    blend_age_s: float,
    config: MatcherConfig = MatcherConfig(),
) -> float:
    """Return the linearly decayed cost of interrupting an active blend."""

    age = float(blend_age_s)
    duration = float(config.transition_settle_duration_s)
    magnitude = float(config.transition_settle_penalty)
    if not math.isfinite(age) or age < 0.0:
        raise ContractError("blend_age_s must be finite and non-negative")
    if not math.isfinite(duration) or duration < 0.0:
        raise ContractError(
            "transition_settle_duration_s must be finite and non-negative"
        )
    if not math.isfinite(magnitude) or magnitude < 0.0:
        raise ContractError(
            "transition_settle_penalty must be finite and non-negative"
        )
    if (
        duration == 0.0
        or magnitude == 0.0
        or age >= duration
        or math.isclose(age, duration, rel_tol=0.0, abs_tol=1e-12)
    ):
        return 0.0
    return magnitude * (1.0 - age / duration)


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


def _validated_transition_costs(
    features: torch.Tensor,
    additional_transition_costs: torch.Tensor | None,
) -> torch.Tensor:
    if additional_transition_costs is None:
        return torch.zeros(
            features.shape[0], dtype=torch.float32, device=features.device
        )
    costs = additional_transition_costs
    if (
        not isinstance(costs, torch.Tensor)
        or tuple(costs.shape) != (features.shape[0],)
        or costs.dtype != torch.float32
        or costs.device != features.device
        or not bool(torch.isfinite(costs).all().item())
        or bool((costs < 0).any().item())
    ):
        raise ContractError(
            "additional_transition_costs must be finite non-negative float32 "
            "with one row on the database device"
        )
    return costs


def select_exact_candidate(
    database: TorchMotionDatabase,
    normalized_query: torch.Tensor,
    *,
    current_clip_index: int,
    current_frame_index: int,
    incumbent_row: int | None,
    search: bool,
    config: MatcherConfig = MatcherConfig(),
    additional_transition_penalty: float = 0.0,
    additional_transition_costs: torch.Tensor | None = None,
) -> SearchDecision:
    """Select the exact lowest-cost eligible row with continuation hysteresis."""
    additional_penalty = float(additional_transition_penalty)
    if not math.isfinite(additional_penalty) or additional_penalty < 0.0:
        raise ContractError(
            "additional_transition_penalty must be finite and non-negative"
        )
    features, row_count = _validate_search_inputs(
        database, normalized_query, incumbent_row
    )
    transition_costs = _validated_transition_costs(
        features, additional_transition_costs
    )
    if incumbent_row is None and not search:
        raise ContractError("search=False requires a valid incumbent")

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

    total_costs = (
        feature_costs
        + config.transition_penalty
        + additional_penalty
        + transition_costs
    )
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
    selected_transition_cost_tensor = transition_costs[selected_row_tensor]
    if incumbent_row is not None:
        selected_transition_cost_tensor = torch.where(
            selected_row_tensor == incumbent_row,
            torch.zeros((), dtype=torch.float32, device=features.device),
            selected_transition_cost_tensor,
        )

    # Convert all diagnostics in one device synchronization.
    diagnostics = torch.stack(
        (
            selected_row_tensor.to(torch.float64),
            incumbent_cost_tensor.to(torch.float64),
            selected_feature_cost_tensor.to(torch.float64),
            selected_total_cost_tensor.to(torch.float64),
            candidate_total_tensor.to(torch.float64),
            selected_transition_cost_tensor.to(torch.float64),
        )
    ).cpu().tolist()
    selected_row = int(diagnostics[0])
    incumbent_cost = float(diagnostics[1])
    selected_feature_cost = float(diagnostics[2])
    selected_total_cost = float(diagnostics[3])
    candidate_total = float(diagnostics[4])
    selected_transition_cost = float(diagnostics[5])
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
        selected_transition_cost=selected_transition_cost,
    )


def rank_exact_transition_candidates(
    database: TorchMotionDatabase,
    normalized_query: torch.Tensor,
    *,
    current_clip_index: int,
    current_frame_index: int,
    config: MatcherConfig = MatcherConfig(),
    additional_transition_costs: torch.Tensor | None = None,
) -> tuple[SearchDecision, ...]:
    """Rank every finite eligible transition by exact matching cost."""

    features, row_count = _validate_search_inputs(
        database, normalized_query, None
    )
    transition_costs = _validated_transition_costs(
        features, additional_transition_costs
    )
    feature_costs = torch.sum(
        torch.square(features - normalized_query.unsqueeze(0)), dim=1
    )
    eligible = torch.ones(
        row_count, dtype=torch.bool, device=features.device
    )
    local = (
        database._search_clip_index == int(current_clip_index)
    ) & (
        torch.abs(
            database._search_frame_index - int(current_frame_index)
        )
        <= config.exclusion_frames
    )
    eligible &= ~local
    eligible &= torch.isfinite(feature_costs)
    eligible_rows = torch.nonzero(eligible, as_tuple=False).flatten()
    if eligible_rows.numel() == 0:
        return ()

    eligible_feature_costs = feature_costs[eligible_rows]
    eligible_total_costs = (
        eligible_feature_costs
        + config.transition_penalty
        + transition_costs[eligible_rows]
    )
    order = torch.argsort(eligible_total_costs, stable=True)
    ranked_rows = eligible_rows[order]
    ranked_feature_costs = eligible_feature_costs[order]
    ranked_total_costs = eligible_total_costs[order]
    ranked_transition_costs = transition_costs[ranked_rows]
    diagnostics = torch.stack(
        (
            ranked_rows.to(torch.float64),
            ranked_feature_costs.to(torch.float64),
            ranked_total_costs.to(torch.float64),
            ranked_transition_costs.to(torch.float64),
        ),
        dim=1,
    ).cpu().tolist()
    return tuple(
        SearchDecision(
            selected_row=int(row),
            incumbent_row=None,
            incumbent_cost=math.inf,
            selected_feature_cost=float(feature_cost),
            selected_total_cost=float(total_cost),
            searched=True,
            transitioned=True,
            selected_transition_cost=float(transition_cost),
        )
        for row, feature_cost, total_cost, transition_cost in diagnostics
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


@dataclass(frozen=True)
class MotionMatchDiagnostics:
    sequence: int
    selected_clip_path: str
    selected_frame: int
    incumbent_cost: float
    selected_feature_cost: float
    motion_feature_cost: float
    extension_feature_cost: float
    selected_total_cost: float
    selected_transition_position_cost: float
    selected_transition_velocity_cost: float
    selected_transition_continuity_cost: float
    searched: bool
    transitioned: bool
    transition_rejected: bool
    terrain_safety_override: bool
    terrain_safety_override_rank: int
    force_search_reason: str | None
    search_time_ns: int | None
    step_time_ns: int


@dataclass(frozen=True)
class MotionMatchResult:
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    root_position_world: torch.Tensor
    root_orientation_world_wxyz: torch.Tensor
    dense_joint_position_window: torch.Tensor
    dense_joint_velocity_window: torch.Tensor
    dense_root_position_window: torch.Tensor
    dense_root_orientation_window_wxyz: torch.Tensor
    dense_feature_body_position_window: torch.Tensor
    dense_feature_body_velocity_window: torch.Tensor
    joint_position_window: torch.Tensor
    joint_velocity_window: torch.Tensor
    root_position_window: torch.Tensor
    root_orientation_window_wxyz: torch.Tensor
    diagnostics: MotionMatchDiagnostics


@dataclass(frozen=True)
class PreparedMotionMatch:
    result: MotionMatchResult
    _owner_token: object
    _base_sequence: int
    _next_state: object


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
    elapsed_s: float


@dataclass(frozen=True)
class _MatcherState:
    sequence: int
    clip_index: int
    frame_index: int
    yaw_offset: torch.Tensor
    translation_xy: torch.Tensor
    shaped_velocity: torch.Tensor
    shaped_heading: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    root_position: torch.Tensor
    root_quaternion: torch.Tensor
    root_linear_velocity: torch.Tensor
    root_angular_velocity: torch.Tensor
    feature_body_position: torch.Tensor
    feature_body_velocity: torch.Tensor
    offsets: _Offsets


@dataclass(frozen=True)
class _ComposedCandidate:
    clip_index: int
    frame_index: int
    transitioned: bool
    yaw_offset: torch.Tensor
    translation_xy: torch.Tensor
    offsets: _Offsets
    dense_joint_position: torch.Tensor
    dense_joint_velocity: torch.Tensor
    dense_root_position: torch.Tensor
    dense_root_quaternion: torch.Tensor
    dense_root_linear_velocity: torch.Tensor
    dense_root_angular_velocity: torch.Tensor
    dense_body_position: torch.Tensor
    dense_body_velocity: torch.Tensor


def _copy_result(result: MotionMatchResult) -> MotionMatchResult:
    values = {
        name: getattr(result, name).clone()
        for name in (
            "joint_position",
            "joint_velocity",
            "root_position_world",
            "root_orientation_world_wxyz",
            "dense_joint_position_window",
            "dense_joint_velocity_window",
            "dense_root_position_window",
            "dense_root_orientation_window_wxyz",
            "dense_feature_body_position_window",
            "dense_feature_body_velocity_window",
            "joint_position_window",
            "joint_velocity_window",
            "root_position_window",
            "root_orientation_window_wxyz",
        )
    }
    return MotionMatchResult(**values, diagnostics=result.diagnostics)


class TorchMotionMatcher:
    """One in-process native Takara matcher with transactional state updates."""

    def __init__(
        self,
        folder: MotionFolder,
        database: TorchMotionDatabase,
        continuity: TransitionContinuityDatabase,
        clips: tuple[_DeviceClip, ...],
        config: MatcherConfig,
        emitted_window_validator: EmittedWindowValidator | None = None,
    ) -> None:
        jerk_weight = config.transition_window_jerk_weight
        if (
            isinstance(jerk_weight, bool)
            or not isinstance(jerk_weight, (int, float))
            or not math.isfinite(jerk_weight)
            or jerk_weight < 0
        ):
            raise ContractError(
                "transition_window_jerk_weight must be a finite "
                "non-negative number"
            )
        candidate_count = config.transition_window_candidate_count
        if (
            type(candidate_count) is not int
            or candidate_count <= 0
        ):
            raise ContractError(
                "transition_window_candidate_count must be a positive integer"
            )
        if continuity.device != database.device:
            raise ContractError(
                "continuity and motion databases must use the same device"
            )
        continuity_rows = (
            continuity._joint_position.shape[0],
            continuity._joint_velocity.shape[0],
        )
        if any(
            row_count != database._search_features.shape[0]
            for row_count in continuity_rows
        ):
            raise ContractError(
                "continuity and motion database row counts must match"
            )
        self.folder = folder
        self.database = database
        self._continuity = continuity
        self.device = database.device
        self.config = config
        self._emitted_window_validator = emitted_window_validator
        self._clips = clips
        self._row_sources = tuple(
            zip(
                database._search_clip_index.cpu().tolist(),
                database._search_frame_index.cpu().tolist(),
            )
        )
        self._owner_token = object()
        self._state: _MatcherState | None = None

    @classmethod
    def from_folder(
        cls,
        motions_dir: str | Path,
        *,
        device: str = "auto",
        config: MatcherConfig = MatcherConfig(),
        extension: SearchFeatureExtension | None = None,
        reset_clip_path: str | None = None,
        emitted_window_validator: EmittedWindowValidator | None = None,
    ) -> "TorchMotionMatcher":
        resolved = resolve_torch_device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ContractError("CUDA motion matcher requested but unavailable")
        folder = MotionFolder.load(motions_dir)
        database = TorchMotionDatabase.from_folder(
            folder,
            device=resolved,
            extension=extension,
            reset_clip_path=reset_clip_path,
        )
        continuity = TransitionContinuityDatabase.from_folder(folder, resolved)
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
        return cls(
            folder,
            database,
            continuity,
            clips,
            config,
            emitted_window_validator,
        )

    @property
    def motion_inventory_sha256(self) -> str:
        return self.folder.inventory_sha256

    def _source_for_row(self, row: int) -> tuple[int, int]:
        pair = self._row_sources[row]
        return int(pair[0]), int(pair[1])

    def _aligned_targets(
        self,
        clip_index: int,
        frame_index: int,
        yaw_offset: torch.Tensor,
        translation_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        clip = self._clips[clip_index]
        sl = slice(frame_index, frame_index + 46)
        root = self.folder.layout.root_body_index
        bodies = torch.tensor(
            (
                root,
                self.folder.layout.left_foot_body_index,
                self.folder.layout.right_foot_body_index,
            ),
            device=self.device,
        )
        joint_p = clip.joint_position[sl]
        joint_v = clip.joint_velocity[sl]
        body_p = _rotate_z(clip.body_position[sl][:, bodies], yaw_offset)
        body_p = body_p.clone()
        body_p[..., :2] += translation_xy
        body_v = _rotate_z(clip.body_linear_velocity[sl][:, bodies], yaw_offset)
        root_p = body_p[:, 0]
        root_v = body_v[:, 0]
        yaw_q = _quat_from_yaw(yaw_offset)
        root_q = _quat_normalize(
            _quat_mul(yaw_q.expand(46, 4), clip.body_quaternion[sl, root])
        )
        root_w = _rotate_z(clip.body_angular_velocity[sl, root], yaw_offset)
        return joint_p, joint_v, root_p, root_q, root_v, root_w, body_p, body_v

    def _ranked_terrain_rescue(
        self,
        state: _MatcherState,
        shaped: ShapedCommand,
        query: torch.Tensor,
        continuity: TransitionContinuityCosts,
        successor: int,
        incumbent_cost: float,
        validator,
        search_time,
    ):
        rescue_start = time.perf_counter_ns()
        ranked_rescue_decisions = rank_exact_transition_candidates(
            self.database,
            query,
            current_clip_index=state.clip_index,
            current_frame_index=state.frame_index,
            config=self.config,
            additional_transition_costs=continuity.total,
        )
        rescue_time = time.perf_counter_ns() - rescue_start
        search_time = (
            rescue_time if search_time is None else search_time + rescue_time
        )
        safe_rescue = None
        for rank, rescue_decision in enumerate(
            ranked_rescue_decisions, start=1
        ):
            rescue_candidate = self._compose_candidate(
                state,
                shaped,
                rescue_decision.selected_row,
                successor,
            )
            rescue_accepted = validator(
                rescue_candidate.dense_body_position.clone()
            )
            if type(rescue_accepted) is not bool:
                raise ContractError(
                    "emitted-window validator must return exact bool"
                )
            if rescue_accepted:
                safe_rescue = (rank, rescue_decision, rescue_candidate)
                break
        if safe_rescue is None:
            raise ContractError("no safe terrain rescue candidate")
        rank, rescue_decision, candidate = safe_rescue
        decision = SearchDecision(
            selected_row=rescue_decision.selected_row,
            incumbent_row=successor,
            incumbent_cost=incumbent_cost,
            selected_feature_cost=rescue_decision.selected_feature_cost,
            selected_total_cost=rescue_decision.selected_total_cost,
            searched=True,
            transitioned=True,
            selected_transition_cost=(
                rescue_decision.selected_transition_cost
            ),
        )
        return rank, decision, candidate, search_time

    def _rerank_transition_window(
        self,
        state: _MatcherState,
        shaped: ShapedCommand,
        query: torch.Tensor,
        continuity: TransitionContinuityCosts,
        successor: int,
        decision: SearchDecision,
        candidate: _ComposedCandidate,
        validator,
        settle_penalty: float,
        search_time: int | None,
    ) -> tuple[SearchDecision, _ComposedCandidate, int | None]:
        weight = float(self.config.transition_window_jerk_weight)
        if weight <= 0.0:
            return decision, candidate, search_time
        rerank_start = time.perf_counter_ns()
        ranked = rank_exact_transition_candidates(
            self.database,
            query,
            current_clip_index=state.clip_index,
            current_frame_index=state.frame_index,
            config=self.config,
            additional_transition_costs=continuity.total,
        )
        rerank_time = time.perf_counter_ns() - rerank_start
        search_time = (
            rerank_time
            if search_time is None
            else search_time + rerank_time
        )

        def predicted_p95(value: _ComposedCandidate) -> torch.Tensor:
            jerk = torch.linalg.vector_norm(
                torch.diff(
                    value.dense_joint_position, n=3, dim=0
                )
                / (self.config.dt**3),
                dim=1,
            )
            return torch.quantile(jerk, 0.95)

        records = []
        original_in_prefix = False
        limit = self.config.transition_window_candidate_count
        for ranked_decision in ranked[:limit]:
            raw_total = (
                ranked_decision.selected_total_cost + settle_penalty
            )
            if raw_total >= decision.incumbent_cost:
                continue
            if ranked_decision.selected_row == decision.selected_row:
                reranked_candidate = candidate
                original_in_prefix = True
            else:
                reranked_candidate = self._compose_candidate(
                    state,
                    shaped,
                    ranked_decision.selected_row,
                    successor,
                )
            records.append(
                (
                    ranked_decision,
                    reranked_candidate,
                    raw_total,
                    torch.as_tensor(
                        raw_total,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    + weight
                    * predicted_p95(reranked_candidate)
                    / 1000.0,
                )
            )
        if not original_in_prefix:
            records.append(
                (
                    decision,
                    candidate,
                    decision.selected_total_cost,
                    torch.as_tensor(
                        decision.selected_total_cost,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    + weight * predicted_p95(candidate) / 1000.0,
                )
            )
        scores = torch.stack([record[3] for record in records])
        order = torch.argsort(scores, stable=True).cpu().tolist()
        for index in order:
            ranked_decision, reranked_candidate, raw_total, _score = (
                records[index]
            )
            if (
                ranked_decision.selected_row != decision.selected_row
                and validator is not None
            ):
                accepted = validator(
                    reranked_candidate.dense_body_position.clone()
                )
                if type(accepted) is not bool:
                    raise ContractError(
                        "emitted-window validator must return exact bool"
                    )
                if not accepted:
                    continue
            if ranked_decision.selected_row == decision.selected_row:
                return decision, candidate, search_time
            return (
                SearchDecision(
                    selected_row=ranked_decision.selected_row,
                    incumbent_row=successor,
                    incumbent_cost=decision.incumbent_cost,
                    selected_feature_cost=(
                        ranked_decision.selected_feature_cost
                    ),
                    selected_total_cost=raw_total,
                    searched=True,
                    transitioned=True,
                    selected_transition_cost=(
                        ranked_decision.selected_transition_cost
                    ),
                ),
                reranked_candidate,
                search_time,
            )
        return decision, candidate, search_time

    def _compose_candidate(
        self,
        state: _MatcherState,
        shaped: ShapedCommand,
        selected_row: int,
        incumbent_row: int | None,
    ) -> _ComposedCandidate:
        clip_index, frame_index = self._source_for_row(selected_row)
        transitioned = (
            incumbent_row is None or selected_row != incumbent_row
        )
        if transitioned:
            clip = self._clips[clip_index]
            root = self.folder.layout.root_body_index
            source_pos = clip.body_position[frame_index, root]
            source_yaw = _quat_yaw(
                clip.body_quaternion[frame_index, root]
            )
            yaw_offset = _wrapped_angle(
                shaped.heading_world_yaw - source_yaw
            )
            desired_xy = (
                state.root_position[:2]
                + shaped.velocity_world_xy * self.config.dt
            )
            rotated = _rotate_z(source_pos, yaw_offset)
            translation = desired_xy - rotated[:2]
        else:
            yaw_offset = state.yaw_offset
            translation = state.translation_xy
        targets = self._aligned_targets(
            clip_index, frame_index, yaw_offset, translation
        )
        jp, jv, rp, rq, rv, rw, bp, bv = targets
        if transitioned:
            offsets = _Offsets(
                state.joint_position - jp[0],
                state.joint_velocity - jv[0],
                state.root_position - rp[0],
                state.root_linear_velocity - rv[0],
                _quat_to_scaled_axis(
                    _quat_mul(
                        state.root_quaternion, _quat_inverse(rq[0])
                    )
                ),
                state.root_angular_velocity - rw[0],
                state.feature_body_position - bp[0],
                state.feature_body_velocity - bv[0],
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
                state.offsets.elapsed_s + self.config.dt,
            )
        times = (
            torch.arange(46, device=self.device, dtype=torch.float32)
            * self.config.dt
            + offsets.elapsed_s
        )
        jpo, jvo = decay_spring_offsets(
            offsets.joint_position,
            offsets.joint_velocity,
            halflife_s=self.config.inertialization_halflife_s,
            time_s=times,
        )
        rpo, rvo = decay_spring_offsets(
            offsets.root_position,
            offsets.root_linear_velocity,
            halflife_s=self.config.inertialization_halflife_s,
            time_s=times,
        )
        bao, bvo = decay_spring_offsets(
            offsets.body_position,
            offsets.body_velocity,
            halflife_s=self.config.inertialization_halflife_s,
            time_s=times,
        )
        qao, qwo = decay_spring_offsets(
            offsets.root_rotation_axis,
            offsets.root_angular_velocity,
            halflife_s=self.config.inertialization_halflife_s,
            time_s=times,
        )
        return _ComposedCandidate(
            clip_index=clip_index,
            frame_index=frame_index,
            transitioned=transitioned,
            yaw_offset=yaw_offset,
            translation_xy=translation,
            offsets=offsets,
            dense_joint_position=jp + jpo,
            dense_joint_velocity=jv + jvo,
            dense_root_position=rp + rpo,
            dense_root_quaternion=_quat_normalize(
                _quat_mul(_quat_from_scaled_axis(qao), rq)
            ),
            dense_root_linear_velocity=rv + rvo,
            dense_root_angular_velocity=rw + qwo,
            dense_body_position=bp + bao,
            dense_body_velocity=bv + bvo,
        )

    def _make_result(
        self,
        *,
        sequence: int,
        clip_index: int,
        frame_index: int,
        decision: SearchDecision,
        continuity: TransitionContinuityCosts | None,
        transition_rejected: bool,
        terrain_safety_override: bool,
        terrain_safety_override_rank: int,
        force_reason: str | None,
        search_time_ns: int | None,
        step_start_ns: int,
        motion_feature_cost: float,
        extension_feature_cost: float,
        dense_joint_p: torch.Tensor,
        dense_joint_v: torch.Tensor,
        dense_root_p: torch.Tensor,
        dense_root_q: torch.Tensor,
        dense_body_p: torch.Tensor,
        dense_body_v: torch.Tensor,
    ) -> MotionMatchResult:
        sample = torch.arange(0, 46, 5, device=self.device)
        if decision.transitioned:
            if continuity is None:
                raise ContractError(
                    "transition diagnostics require continuity costs"
                )
            (
                position_cost,
                velocity_cost,
                total_cost,
            ) = torch.stack(
                (
                    continuity.position[decision.selected_row],
                    continuity.velocity[decision.selected_row],
                    continuity.total[decision.selected_row],
                )
            ).to(torch.float64).cpu().tolist()
        else:
            position_cost = velocity_cost = total_cost = 0.0
        diagnostics = MotionMatchDiagnostics(
            sequence=sequence,
            selected_clip_path=self.folder.clips[clip_index].relative_path,
            selected_frame=frame_index,
            incumbent_cost=decision.incumbent_cost,
            selected_feature_cost=decision.selected_feature_cost,
            motion_feature_cost=motion_feature_cost,
            extension_feature_cost=extension_feature_cost,
            selected_total_cost=decision.selected_total_cost,
            selected_transition_position_cost=float(position_cost),
            selected_transition_velocity_cost=float(velocity_cost),
            selected_transition_continuity_cost=float(total_cost),
            searched=decision.searched,
            transitioned=decision.transitioned,
            transition_rejected=transition_rejected,
            terrain_safety_override=terrain_safety_override,
            terrain_safety_override_rank=terrain_safety_override_rank,
            force_search_reason=force_reason,
            search_time_ns=search_time_ns,
            step_time_ns=time.perf_counter_ns() - step_start_ns,
        )
        result = MotionMatchResult(
            joint_position=dense_joint_p[0].clone(),
            joint_velocity=dense_joint_v[0].clone(),
            root_position_world=dense_root_p[0].clone(),
            root_orientation_world_wxyz=dense_root_q[0].clone(),
            dense_joint_position_window=dense_joint_p.clone(),
            dense_joint_velocity_window=dense_joint_v.clone(),
            dense_root_position_window=dense_root_p.clone(),
            dense_root_orientation_window_wxyz=dense_root_q.clone(),
            dense_feature_body_position_window=dense_body_p.clone(),
            dense_feature_body_velocity_window=dense_body_v.clone(),
            joint_position_window=dense_joint_p[sample].clone(),
            joint_velocity_window=dense_joint_v[sample].clone(),
            root_position_window=dense_root_p[sample].clone(),
            root_orientation_window_wxyz=dense_root_q[sample].clone(),
            diagnostics=diagnostics,
        )
        finite = torch.isfinite(dense_joint_p).all()
        finite &= torch.isfinite(dense_joint_v).all()
        finite &= torch.isfinite(dense_root_p).all()
        finite &= torch.isfinite(dense_root_q).all()
        finite &= torch.isfinite(dense_body_p).all()
        finite &= torch.isfinite(dense_body_v).all()
        unit = torch.all(
            torch.abs(torch.linalg.vector_norm(dense_root_q, dim=-1) - 1.0)
            <= 1e-5
        )
        if not bool((finite & unit).item()):
            raise ContractError("motion match produced invalid dense output")
        return result

    def reset(self) -> MotionMatchResult:
        step_start = time.perf_counter_ns()
        row = self.database.reset_row
        clip_index, frame_index = self._source_for_row(row)
        clip = self._clips[clip_index]
        root = self.folder.layout.root_body_index
        source_root = clip.body_position[frame_index, root]
        source_yaw = _quat_yaw(clip.body_quaternion[frame_index, root])
        yaw_offset = -source_yaw
        rotated_root = _rotate_z(source_root, yaw_offset)
        translation = -rotated_root[:2]
        targets = self._aligned_targets(
            clip_index, frame_index, yaw_offset, translation
        )
        jp, jv, rp, rq, rv, rw, bp, bv = targets
        zeros = _Offsets(
            torch.zeros_like(jp[0]),
            torch.zeros_like(jv[0]),
            torch.zeros_like(rp[0]),
            torch.zeros_like(rv[0]),
            torch.zeros(3, device=self.device),
            torch.zeros_like(rw[0]),
            torch.zeros_like(bp[0]),
            torch.zeros_like(bv[0]),
            self.config.transition_settle_duration_s,
        )
        decision = SearchDecision(row, None, math.inf, 0.0, 0.0, False, False)
        result = self._make_result(
            sequence=0,
            clip_index=clip_index,
            frame_index=frame_index,
            decision=decision,
            continuity=None,
            transition_rejected=False,
            terrain_safety_override=False,
            terrain_safety_override_rank=0,
            force_reason=None,
            search_time_ns=None,
            step_start_ns=step_start,
            motion_feature_cost=0.0,
            extension_feature_cost=0.0,
            dense_joint_p=jp,
            dense_joint_v=jv,
            dense_root_p=rp,
            dense_root_q=rq,
            dense_body_p=bp,
            dense_body_v=bv,
        )
        self._state = _MatcherState(
            0, clip_index, frame_index, yaw_offset, translation,
            torch.zeros(2, device=self.device),
            torch.zeros((), device=self.device),
            jp[0].clone(), jv[0].clone(), rp[0].clone(), rq[0].clone(),
            rv[0].clone(), rw[0].clone(), bp[0].clone(), bv[0].clone(), zeros,
        )
        return _copy_result(result)

    def prepare_step(
        self,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
        *,
        dt: float = 0.02,
    ) -> PreparedMotionMatch:
        step_start = time.perf_counter_ns()
        state = self._state
        if state is None:
            raise ContractError("reset must be called before prepare_step")
        if not math.isfinite(dt) or abs(dt - self.config.dt) > 1e-12:
            raise ContractError(f"dt must equal {self.config.dt}")
        requested_v = torch.tensor(
            velocity_world_xy, dtype=torch.float32, device=self.device
        )
        requested_h = torch.tensor(
            heading_world_yaw, dtype=torch.float32, device=self.device
        )
        successor = self.database.row_for_source(
            state.clip_index, state.frame_index + 1
        )
        shaped = predict_command_trajectory(
            state.root_position[:2],
            state.shaped_velocity,
            state.shaped_heading,
            requested_v,
            requested_h,
            has_valid_successor=successor is not None,
            config=self.config,
        )
        feature_state = GeneratedFeatureState(
            root_position_world=state.root_position,
            root_orientation_world_wxyz=state.root_quaternion,
            root_linear_velocity_world=state.root_linear_velocity,
            left_foot_position_world=state.feature_body_position[1],
            right_foot_position_world=state.feature_body_position[2],
            left_foot_velocity_world=state.feature_body_velocity[1],
            right_foot_velocity_world=state.feature_body_velocity[2],
        )
        raw_query = extract_query_features(feature_state, shaped.trajectory)
        if self.database._extension is not None:
            raw_query = torch.cat(
                (
                    raw_query,
                    validated_extension_query_row(
                        self.database._extension,
                        feature_state,
                        shaped.trajectory,
                    ),
                )
            )
        query = self.database.normalization.normalize(raw_query)
        search = search_is_due(
            state.sequence, shaped.force_search, self.config
        )
        settle_penalty = active_transition_penalty(
            state.offsets.elapsed_s, self.config
        )
        continuity = self._continuity.costs(
            state.joint_position,
            state.joint_velocity,
            position_weight=self.config.transition_joint_position_weight,
            velocity_weight=self.config.transition_joint_velocity_weight,
        )
        search_start = time.perf_counter_ns()
        decision = select_exact_candidate(
            self.database,
            query,
            current_clip_index=state.clip_index,
            current_frame_index=state.frame_index,
            incumbent_row=successor,
            search=search,
            config=self.config,
            additional_transition_penalty=settle_penalty,
            additional_transition_costs=continuity.total,
        )
        search_time = time.perf_counter_ns() - search_start if search else None
        candidate = self._compose_candidate(
            state, shaped, decision.selected_row, successor
        )
        transition_rejected = False
        terrain_safety_override = False
        terrain_safety_override_rank = 0
        validator = self._emitted_window_validator
        if validator is not None:
            accepted = validator(candidate.dense_body_position.clone())
            if type(accepted) is not bool:
                raise ContractError(
                    "emitted-window validator must return exact bool"
                )
            if not accepted:
                if not candidate.transitioned:
                    if successor is None:
                        raise ContractError(
                            "unsafe incumbent has no source row"
                        )
                    (
                        terrain_safety_override_rank,
                        decision,
                        candidate,
                        search_time,
                    ) = self._ranked_terrain_rescue(
                        state,
                        shaped,
                        query,
                        continuity,
                        successor,
                        decision.incumbent_cost,
                        validator,
                        search_time,
                    )
                    terrain_safety_override = True
                else:
                    if successor is None:
                        raise ContractError(
                            "unsafe transition has no incumbent"
                        )
                    incumbent = self._compose_candidate(
                        state, shaped, successor, successor
                    )
                    incumbent_accepted = validator(
                        incumbent.dense_body_position.clone()
                    )
                    if type(incumbent_accepted) is not bool:
                        raise ContractError(
                            "emitted-window validator must return exact bool"
                        )
                    if not incumbent_accepted:
                        (
                            terrain_safety_override_rank,
                            decision,
                            candidate,
                            search_time,
                        ) = self._ranked_terrain_rescue(
                            state,
                            shaped,
                            query,
                            continuity,
                            successor,
                            decision.incumbent_cost,
                            validator,
                            search_time,
                        )
                        terrain_safety_override = True
                    else:
                        candidate = incumbent
                        decision = SearchDecision(
                            selected_row=successor,
                            incumbent_row=successor,
                            incumbent_cost=decision.incumbent_cost,
                            selected_feature_cost=decision.incumbent_cost,
                            selected_total_cost=decision.incumbent_cost,
                            searched=decision.searched,
                            transitioned=False,
                        )
                        transition_rejected = True
        if (
            decision.transitioned
            and not terrain_safety_override
            and self.config.transition_window_jerk_weight > 0.0
        ):
            (
                decision,
                candidate,
                search_time,
            ) = self._rerank_transition_window(
                state,
                shaped,
                query,
                continuity,
                successor,
                decision,
                candidate,
                validator,
                settle_penalty,
                search_time,
            )
        residual_sq = torch.square(
            self.database._search_features[decision.selected_row] - query
        )
        motion_cost_tensor = residual_sq[
            : self.database.motion_feature_dim
        ].sum()
        extension_cost_tensor = residual_sq[
            self.database.motion_feature_dim :
        ].sum()
        motion_feature_cost, extension_feature_cost = (
            torch.stack((motion_cost_tensor, extension_cost_tensor))
            .to(torch.float64)
            .cpu()
            .tolist()
        )
        force_reason = None
        if successor is None:
            force_reason = "clip_end"
        elif shaped.force_search:
            force_reason = "command_transition"
        result = self._make_result(
            sequence=state.sequence + 1,
            clip_index=candidate.clip_index,
            frame_index=candidate.frame_index,
            decision=decision,
            continuity=continuity,
            transition_rejected=transition_rejected,
            terrain_safety_override=terrain_safety_override,
            terrain_safety_override_rank=terrain_safety_override_rank,
            force_reason=force_reason,
            search_time_ns=search_time,
            step_start_ns=step_start,
            motion_feature_cost=float(motion_feature_cost),
            extension_feature_cost=float(extension_feature_cost),
            dense_joint_p=candidate.dense_joint_position,
            dense_joint_v=candidate.dense_joint_velocity,
            dense_root_p=candidate.dense_root_position,
            dense_root_q=candidate.dense_root_quaternion,
            dense_body_p=candidate.dense_body_position,
            dense_body_v=candidate.dense_body_velocity,
        )
        next_state = _MatcherState(
            state.sequence + 1,
            candidate.clip_index,
            candidate.frame_index,
            candidate.yaw_offset,
            candidate.translation_xy,
            shaped.velocity_world_xy.clone(), shaped.heading_world_yaw.clone(),
            candidate.dense_joint_position[0].clone(),
            candidate.dense_joint_velocity[0].clone(),
            candidate.dense_root_position[0].clone(),
            candidate.dense_root_quaternion[0].clone(),
            candidate.dense_root_linear_velocity[0].clone(),
            candidate.dense_root_angular_velocity[0].clone(),
            candidate.dense_body_position[0].clone(),
            candidate.dense_body_velocity[0].clone(),
            candidate.offsets,
        )
        return PreparedMotionMatch(
            _copy_result(result), self._owner_token, state.sequence, next_state
        )

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
