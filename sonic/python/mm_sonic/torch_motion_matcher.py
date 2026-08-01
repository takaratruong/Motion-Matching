"""Bounded operator commands and exact dense Torch motion search."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from pathlib import Path
import time
from typing import Protocol, Sequence

import torch

from .joints import ContractError
from .torch_contact_segments import (
    SegmentPlacement,
    SegmentValidation,
    TerrainContactSegmentPolicy,
)
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
    FeatureNormalization,
    extract_query_features,
    resolve_torch_device,
    validated_extension_query_row,
)
from .torch_transition_reachability import (
    ReachabilityLimits,
    TransitionReachabilityDiagnostic,
    bounded_transition_reachability,
)


LOOP_HISTORY_FRAMES = 128
LOOP_SOURCE_NEIGHBORHOOD_FRAMES = 8
LOOP_ROOT_PROGRESS_M = 0.05
LOOP_REVISIT_PENALTY = 1000.0
RECENT_CROSS_CLIP_REVISIT_PENALTY = 25.0
CONTIGUOUS_SEGMENT_COMMAND_ALIGNMENT_MIN = 0.5
CONTACT_COMMITMENT_HEADING_ALIGNMENT_MIN = math.cos(math.radians(15.0))


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
    joint_reference_smoothing_weight: float = 0.0
    transition_settle_duration_s: float = 0.0
    transition_settle_penalty: float = 0.0
    transition_joint_position_weight: float = 0.0
    transition_joint_velocity_weight: float = 0.0
    transition_window_jerk_weight: float = 0.0
    transition_window_candidate_count: int = 32
    transition_window_jerk_horizon_steps: int = 46


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


class TransitionTerminalEvaluator(Protocol):
    """Classify a composed safe transition under the current issued command."""

    def __call__(
        self,
        feature_body_position_window: torch.Tensor,
        clip_index: int,
        frame_index: int,
        command_velocity_world_xy: torch.Tensor,
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


def _validated_transition_eligible_rows(
    features: torch.Tensor,
    transition_eligible_rows: torch.Tensor | None,
) -> torch.Tensor:
    if transition_eligible_rows is None:
        return torch.ones(
            features.shape[0], dtype=torch.bool, device=features.device
        )
    eligible = transition_eligible_rows
    if (
        not isinstance(eligible, torch.Tensor)
        or eligible.dtype != torch.bool
        or tuple(eligible.shape) != (features.shape[0],)
        or eligible.device != features.device
    ):
        raise ContractError(
            "transition_eligible_rows must be boolean with one row on the "
            "database device"
        )
    return eligible


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
    transition_eligible_rows: torch.Tensor | None = None,
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
    transition_eligible = _validated_transition_eligible_rows(
        features, transition_eligible_rows
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
    eligible &= transition_eligible
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
    transition_eligible_rows: torch.Tensor | None = None,
) -> tuple[SearchDecision, ...]:
    """Rank every finite eligible transition by exact matching cost."""

    features, row_count = _validate_search_inputs(
        database, normalized_query, None
    )
    transition_costs = _validated_transition_costs(
        features, additional_transition_costs
    )
    transition_eligible = _validated_transition_eligible_rows(
        features, transition_eligible_rows
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
    eligible &= transition_eligible
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


def _rotate_xy(values: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    x, y = values[..., 0], values[..., 1]
    while c.ndim < x.ndim:
        c = c.unsqueeze(-1)
        s = s.unsqueeze(-1)
    return torch.stack((c * x - s * y, s * x + c * y), -1)


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
    segment_committed: bool
    segment_start_frame: int | None
    segment_end_frame: int | None
    segment_entering_foot: int | None
    segment_vertical_offset_m: float | None
    segment_rejection_reason: str | None


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
class SegmentCommitment:
    clip_index: int
    start_frame: int
    end_frame: int
    entering_foot: int
    vertical_offset_m: float

    def __post_init__(self) -> None:
        if (
            type(self.clip_index) is not int
            or self.clip_index < 0
            or type(self.start_frame) is not int
            or self.start_frame < 0
            or type(self.end_frame) is not int
            or self.end_frame <= self.start_frame
            or self.entering_foot not in (0, 1)
            or not math.isfinite(float(self.vertical_offset_m))
        ):
            raise ContractError("segment commitment fields are invalid")
        object.__setattr__(
            self, "vertical_offset_m", float(self.vertical_offset_m)
        )


@dataclass(frozen=True)
class _MatcherState:
    sequence: int
    clip_index: int
    frame_index: int
    yaw_offset: torch.Tensor
    translation_xy: torch.Tensor
    translation_z: float
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
    commitment: SegmentCommitment | None


@dataclass(frozen=True)
class _SelectionVisit:
    sequence: int
    clip_index: int
    frame_index: int
    root_position_xy: torch.Tensor


def _loop_revisit_transition_costs(
    database: TorchMotionDatabase,
    history: Sequence[_SelectionVisit],
    current_root_position_xy: torch.Tensor,
    *,
    current_clip_index: int,
    current_sequence: int,
    config: MatcherConfig,
) -> torch.Tensor:
    """Penalize source neighborhoods revisited without meaningful root progress."""

    root = current_root_position_xy
    if (
        not isinstance(root, torch.Tensor)
        or tuple(root.shape) != (2,)
        or root.device != database.device
        or not torch.isfinite(root).all()
    ):
        raise ContractError(
            "loop revisit root position must be finite shape-(2,) "
            "on the database device"
        )
    if not 0 <= int(current_clip_index) < len(database.folder.clips):
        raise ContractError("loop revisit current clip index is invalid")
    costs = torch.zeros(
        database.feature_shape[0],
        dtype=torch.float32,
        device=database.device,
    )
    eligible_visits = []
    penalties = []
    for visit in history:
        if not isinstance(visit, _SelectionVisit):
            raise ContractError("loop revisit history is invalid")
        age = int(current_sequence) - visit.sequence
        if age <= 0 or age > LOOP_HISTORY_FRAMES:
            continue
        if (
            age <= config.exclusion_frames
            and visit.clip_index == int(current_clip_index)
        ):
            continue
        if (
            visit.root_position_xy.device != database.device
            or tuple(visit.root_position_xy.shape) != (2,)
            or not torch.isfinite(visit.root_position_xy).all()
        ):
            raise ContractError("loop revisit history position is invalid")
        eligible_visits.append(visit)
        penalties.append(
            RECENT_CROSS_CLIP_REVISIT_PENALTY
            if age <= config.exclusion_frames
            else LOOP_REVISIT_PENALTY
        )
    if not eligible_visits:
        return costs

    visit_roots = torch.stack(
        [visit.root_position_xy for visit in eligible_visits]
    )
    low_progress = (
        torch.linalg.vector_norm(visit_roots - root, dim=1)
        <= LOOP_ROOT_PROGRESS_M
    )
    visit_clips = torch.tensor(
        [visit.clip_index for visit in eligible_visits],
        dtype=database._search_clip_index.dtype,
        device=database.device,
    )
    visit_frames = torch.tensor(
        [visit.frame_index for visit in eligible_visits],
        dtype=database._search_frame_index.dtype,
        device=database.device,
    )
    visit_penalties = torch.tensor(
        penalties, dtype=torch.float32, device=database.device
    )
    neighborhoods = (
        database._search_clip_index[:, None] == visit_clips[None, :]
    ) & (
        torch.abs(
            database._search_frame_index[:, None] - visit_frames[None, :]
        )
        <= LOOP_SOURCE_NEIGHBORHOOD_FRAMES
    )
    return torch.where(
        neighborhoods & low_progress[None, :],
        visit_penalties[None, :],
        0.0,
    ).amax(dim=1)


@dataclass(frozen=True)
class _ComposedCandidate:
    clip_index: int
    frame_index: int
    transitioned: bool
    yaw_offset: torch.Tensor
    translation_xy: torch.Tensor
    translation_z: float
    segment_placement: SegmentPlacement | None
    segment_validation: SegmentValidation | None
    contact_entry_valid: bool
    offsets: _Offsets
    dense_joint_position: torch.Tensor
    dense_joint_velocity: torch.Tensor
    dense_root_position: torch.Tensor
    dense_root_quaternion: torch.Tensor
    dense_root_linear_velocity: torch.Tensor
    dense_root_angular_velocity: torch.Tensor
    dense_body_position: torch.Tensor
    dense_body_velocity: torch.Tensor


@dataclass(frozen=True)
class _ReachabilityContinuation:
    state: _MatcherState
    history: tuple[_SelectionVisit, ...]


@dataclass(frozen=True)
class _ReachabilityNode:
    state: _MatcherState
    candidate: _ComposedCandidate
    continuations: tuple[_ReachabilityContinuation, ...]


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
        contact_segment_policy: TerrainContactSegmentPolicy | None = None,
        foothold_action_policy: object | None = None,
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
        jerk_horizon = config.transition_window_jerk_horizon_steps
        if (
            type(jerk_horizon) is not int
            or not 4 <= jerk_horizon <= 46
        ):
            raise ContractError(
                "transition_window_jerk_horizon_steps must be an integer "
                "in [4, 46]"
            )
        smoothing_weight = config.joint_reference_smoothing_weight
        if (
            isinstance(smoothing_weight, bool)
            or not isinstance(smoothing_weight, (int, float))
            or not math.isfinite(smoothing_weight)
            or not 0.0 <= smoothing_weight <= 0.5
        ):
            raise ContractError(
                "joint_reference_smoothing_weight must be finite and in "
                "[0, 0.5]"
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
        if (
            contact_segment_policy is not None
            and not isinstance(
                contact_segment_policy, TerrainContactSegmentPolicy
            )
        ):
            raise ContractError("contact_segment_policy is invalid")
        if contact_segment_policy is not None:
            policy_folder = contact_segment_policy.dataset.folder
            if (
                torch.device(contact_segment_policy.dataset.device)
                != database.device
                or policy_folder.inventory_sha256 != folder.inventory_sha256
            ):
                raise ContractError(
                    "contact segment policy does not match the motion database"
                )
        self._contact_segment_policy = contact_segment_policy
        if foothold_action_policy is not None and not callable(
            getattr(foothold_action_policy, "prepare", None)
        ):
            raise ContractError("foothold_action_policy is invalid")
        self._foothold_action_policy = foothold_action_policy
        self._transition_eligible_rows = (
            None
            if contact_segment_policy is None
            else contact_segment_policy.entry_eligibility(database)
        )
        if contact_segment_policy is None:
            self._terrain_entry_rows = None
            self._terrain_entry_direction_xy = None
        else:
            terrain_clips = torch.tensor(
                sorted(contact_segment_policy.index.terrain_clip_indices),
                dtype=database._search_clip_index.dtype,
                device=database.device,
            )
            self._terrain_entry_rows = self._transition_eligible_rows & torch.isin(
                database._search_clip_index, terrain_clips
            )
            directions = torch.zeros(
                (database._search_clip_index.shape[0], 2),
                dtype=torch.float32,
                device=database.device,
            )
            root = folder.layout.root_body_index
            for clip_index in contact_segment_policy.index.terrain_clip_indices:
                rows = self._terrain_entry_rows & (
                    database._search_clip_index == clip_index
                )
                frames = database._search_frame_index[rows]
                if frames.numel() == 0:
                    continue
                end_frames = torch.tensor(
                    [
                        contact_segment_policy.index.entry(
                            clip_index, int(frame)
                        ).end_frame
                        - 1
                        for frame in frames.detach().cpu().tolist()
                    ],
                    dtype=frames.dtype,
                    device=frames.device,
                )
                displacement = (
                    clips[clip_index].body_position[end_frames, root, :2]
                    - clips[clip_index].body_position[frames, root, :2]
                )
                norm = torch.linalg.vector_norm(
                    displacement, dim=1, keepdim=True
                ).clamp_min(1e-6)
                directions[rows] = displacement / norm
            self._terrain_entry_direction_xy = directions
        self._clips = clips
        root = folder.layout.root_body_index
        feet = torch.tensor(
            (
                folder.layout.left_foot_body_index,
                folder.layout.right_foot_body_index,
            ),
            device=database.device,
        )
        self._transition_source_root_yaw = torch.cat(
            tuple(
                _quat_yaw(
                    clip.body_quaternion[: source.valid_frame_stop, root]
                )
                for clip, source in zip(clips, folder.clips)
            )
        )
        self._transition_source_foot_offset_xy = torch.cat(
            tuple(
                clip.body_position[: source.valid_frame_stop, feet, :2]
                - clip.body_position[
                    : source.valid_frame_stop, root, :2
                ][:, None, :]
                for clip, source in zip(clips, folder.clips)
            )
        )
        self._row_sources = tuple(
            zip(
                database._search_clip_index.cpu().tolist(),
                database._search_frame_index.cpu().tolist(),
            )
        )
        self._owner_token = object()
        self._state: _MatcherState | None = None
        self._selection_history: list[_SelectionVisit] = []

    def _flat_support_transition_costs(
        self,
        state: _MatcherState,
        shaped: ShapedCommand,
    ) -> torch.Tensor:
        rows = self.database._search_clip_index.shape
        costs = torch.zeros(rows, dtype=torch.float32, device=self.device)
        policy = self._contact_segment_policy
        if policy is None or policy.flat_support_transition_cost_weight == 0.0:
            return costs
        if state.clip_index in policy.index.terrain_clip_indices:
            return costs
        support = policy.query_support_mask(
            state.feature_body_position,
            state.feature_body_velocity,
        )
        if not bool(support.any().item()):
            return costs
        yaw = shaped.heading_world_yaw - self._transition_source_root_yaw
        offsets = self._transition_source_foot_offset_xy
        cosine = torch.cos(yaw)[:, None]
        sine = torch.sin(yaw)[:, None]
        rotated = torch.stack(
            (
                cosine * offsets[..., 0] - sine * offsets[..., 1],
                sine * offsets[..., 0] + cosine * offsets[..., 1],
            ),
            dim=-1,
        )
        desired_root = (
            state.root_position[:2]
            + shaped.velocity_world_xy * self.config.dt
        )
        predicted = desired_root[None, None, :] + rotated
        residual = predicted[:, support] - state.feature_body_position[
            1:, :2
        ][support]
        costs = policy.flat_support_transition_cost_weight * torch.mean(
            torch.square(residual), dim=(1, 2)
        )
        terrain = torch.tensor(
            sorted(policy.index.terrain_clip_indices),
            dtype=self.database._search_clip_index.dtype,
            device=self.device,
        )
        if terrain.numel() > 0:
            costs = torch.where(
                torch.isin(self.database._search_clip_index, terrain),
                0.0,
                costs,
            )
        return costs

    def _command_transition_eligibility(
        self, state: _MatcherState, shaped: ShapedCommand
    ) -> torch.Tensor | None:
        eligible = self._transition_eligible_rows
        if eligible is None or self._terrain_entry_direction_xy is None:
            return eligible
        command = self._command_travel_direction(shaped)
        if not self._terrain_command_deviated(state, shaped):
            return eligible
        yaw = shaped.heading_world_yaw - self._transition_source_root_yaw
        directions = _rotate_xy(self._terrain_entry_direction_xy, yaw)
        compatible = (
            directions @ command
            >= CONTIGUOUS_SEGMENT_COMMAND_ALIGNMENT_MIN
        )
        return eligible & (~self._terrain_entry_rows | compatible)

    def _foothold_conditioning(
        self,
        state: _MatcherState,
        shaped: ShapedCommand,
        transition_costs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        eligible = self._command_transition_eligibility(state, shaped)
        terrain_eligible = self._command_terrain_entry_rows(state, shaped)
        policy = self._foothold_action_policy
        if policy is None:
            return eligible, terrain_eligible, transition_costs
        try:
            result = policy.prepare(state, shaped, self.database)
            row_eligibility = result.row_eligibility
            additional_row_cost = result.additional_row_cost
        except ContractError:
            raise
        except Exception as error:
            raise ContractError("foothold action policy failed") from error
        row_shape = self.database._search_clip_index.shape
        if (
            not isinstance(row_eligibility, torch.Tensor)
            or row_eligibility.shape != row_shape
            or row_eligibility.dtype != torch.bool
            or row_eligibility.device != self.device
            or not isinstance(additional_row_cost, torch.Tensor)
            or additional_row_cost.shape != row_shape
            or not additional_row_cost.dtype.is_floating_point
            or additional_row_cost.device != self.device
            or not torch.isfinite(additional_row_cost).all()
            or bool((additional_row_cost < 0.0).any().item())
        ):
            raise ContractError("foothold action policy result is invalid")
        conditioned = (
            row_eligibility
            if eligible is None
            else eligible & row_eligibility
        )
        conditioned_terrain = (
            None
            if terrain_eligible is None
            else terrain_eligible & row_eligibility
        )
        return (
            conditioned,
            conditioned_terrain,
            transition_costs + additional_row_cost,
        )

    def _command_terrain_entry_rows(
        self, state: _MatcherState, shaped: ShapedCommand
    ) -> torch.Tensor | None:
        if self._terrain_entry_rows is None:
            return None
        eligible = self._command_transition_eligibility(state, shaped)
        return self._terrain_entry_rows & eligible

    def _terrain_command_deviated(
        self, state: _MatcherState, shaped: ShapedCommand
    ) -> bool:
        command = self._command_travel_direction(shaped)
        velocity = state.root_linear_velocity[:2]
        speed = torch.linalg.vector_norm(velocity)
        travel_deviated = (
            float(speed.item()) <= 1e-6
            or float(torch.dot(velocity / speed, command).item())
            < CONTIGUOUS_SEGMENT_COMMAND_ALIGNMENT_MIN
        )
        root_yaw = _quat_yaw(state.root_quaternion)
        root_facing = torch.stack((torch.cos(root_yaw), torch.sin(root_yaw)))
        heading_deviated = float(
            torch.dot(root_facing, shaped.trajectory.facing_world_xy[-1]).item()
        ) < CONTACT_COMMITMENT_HEADING_ALIGNMENT_MIN
        return travel_deviated or heading_deviated

    def _contact_commitment_should_interrupt(
        self, state: _MatcherState, shaped: ShapedCommand
    ) -> bool:
        return (
            state.commitment is not None
            and self._terrain_command_deviated(state, shaped)
        )

    @staticmethod
    def _command_travel_direction(shaped: ShapedCommand) -> torch.Tensor:
        displacement = (
            shaped.trajectory.position_world_xy[-1]
            - shaped.trajectory.position_world_xy[0]
        )
        distance = torch.linalg.vector_norm(displacement)
        if float(distance.item()) > 1e-6:
            return displacement / distance
        speed = torch.linalg.vector_norm(shaped.velocity_world_xy)
        if float(speed.item()) > 1e-6:
            return shaped.velocity_world_xy / speed
        return shaped.trajectory.facing_world_xy[-1]

    def _candidate_yaw_offset(
        self,
        clip_index: int,
        frame_index: int,
        shaped: ShapedCommand,
    ) -> torch.Tensor:
        root = self.folder.layout.root_body_index
        source_yaw = _quat_yaw(
            self._clips[clip_index].body_quaternion[frame_index, root]
        )
        return _wrapped_angle(shaped.heading_world_yaw - source_yaw)

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
        normalization_override: FeatureNormalization | None = None,
        contact_segment_policy: TerrainContactSegmentPolicy | None = None,
        foothold_action_policy: object | None = None,
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
            normalization_override=normalization_override,
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
            contact_segment_policy,
            foothold_action_policy,
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
        *,
        translation_z: float,
        horizon: int,
    ) -> tuple[torch.Tensor, ...]:
        if type(horizon) is not int or horizon < 1:
            raise ContractError("aligned target horizon must be positive")
        vertical = float(translation_z)
        if not math.isfinite(vertical):
            raise ContractError("aligned target vertical translation must be finite")
        clip = self._clips[clip_index]
        if not 0 <= frame_index < clip.joint_position.shape[0]:
            raise ContractError("aligned target frame is outside the source clip")
        indices = torch.arange(
            frame_index,
            frame_index + horizon,
            dtype=torch.long,
            device=self.device,
        ).clamp_max(clip.joint_position.shape[0] - 1)
        root = self.folder.layout.root_body_index
        bodies = torch.tensor(
            (
                root,
                self.folder.layout.left_foot_body_index,
                self.folder.layout.right_foot_body_index,
            ),
            device=self.device,
        )
        joint_p = clip.joint_position[indices]
        joint_v = clip.joint_velocity[indices]
        body_p = _rotate_z(clip.body_position[indices][:, bodies], yaw_offset)
        body_p = body_p.clone()
        body_p[..., :2] += translation_xy
        if vertical != 0.0:
            body_p[..., 2] += vertical
        body_v = _rotate_z(
            clip.body_linear_velocity[indices][:, bodies], yaw_offset
        )
        root_p = body_p[:, 0]
        root_v = body_v[:, 0]
        yaw_q = _quat_from_yaw(yaw_offset)
        root_q = _quat_normalize(
            _quat_mul(
                yaw_q.expand(horizon, 4),
                clip.body_quaternion[indices, root],
            )
        )
        root_w = _rotate_z(
            clip.body_angular_velocity[indices, root], yaw_offset
        )
        return joint_p, joint_v, root_p, root_q, root_v, root_w, body_p, body_v

    def _ranked_terrain_rescue(
        self,
        state: _MatcherState,
        shaped: ShapedCommand,
        query: torch.Tensor,
        transition_costs: torch.Tensor,
        transition_eligibility: torch.Tensor | None,
        successor: int | None,
        incumbent_cost: float,
        validator,
        search_time,
        *,
        additional_transition_penalty: float = 0.0,
        maximum_total_cost: float | None = None,
        require_safe_candidate: bool = True,
    ):
        rescue_start = time.perf_counter_ns()
        ranked_rescue_decisions = rank_exact_transition_candidates(
            self.database,
            query,
            current_clip_index=state.clip_index,
            current_frame_index=state.frame_index,
            config=self.config,
            additional_transition_costs=transition_costs,
            transition_eligible_rows=transition_eligibility,
        )
        rescue_time = time.perf_counter_ns() - rescue_start
        search_time = (
            rescue_time if search_time is None else search_time + rescue_time
        )
        safe_rescue = None
        for rank, rescue_decision in enumerate(
            itertools.islice(
                ranked_rescue_decisions,
                self.config.transition_window_candidate_count,
            ),
            start=1,
        ):
            effective_total_cost = (
                rescue_decision.selected_total_cost
                + additional_transition_penalty
            )
            if (
                maximum_total_cost is not None
                and effective_total_cost >= maximum_total_cost
            ):
                break
            rescue_candidate = self._compose_candidate(
                state,
                shaped,
                rescue_decision.selected_row,
                successor,
            )
            rescue_accepted = rescue_candidate.contact_entry_valid
            if rescue_accepted and validator is not None:
                rescue_accepted = validator(
                    rescue_candidate.dense_body_position[:46].clone()
                )
                if type(rescue_accepted) is not bool:
                    raise ContractError(
                        "emitted-window validator must return exact bool"
                    )
            if rescue_accepted:
                safe_rescue = (
                    rank,
                    rescue_decision,
                    rescue_candidate,
                    effective_total_cost,
                )
                break
        if safe_rescue is None:
            if not require_safe_candidate:
                return None
            raise ContractError("no safe terrain rescue candidate")
        rank, rescue_decision, candidate, effective_total_cost = safe_rescue
        decision = SearchDecision(
            selected_row=rescue_decision.selected_row,
            incumbent_row=successor,
            incumbent_cost=incumbent_cost,
            selected_feature_cost=rescue_decision.selected_feature_cost,
            selected_total_cost=effective_total_cost,
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
        transition_costs: torch.Tensor,
        transition_eligibility: torch.Tensor | None,
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
            additional_transition_costs=transition_costs,
            transition_eligible_rows=transition_eligibility,
        )
        rerank_time = time.perf_counter_ns() - rerank_start
        search_time = (
            rerank_time
            if search_time is None
            else search_time + rerank_time
        )

        def predicted_p95(value: _ComposedCandidate) -> torch.Tensor:
            horizon = self.config.transition_window_jerk_horizon_steps
            jerk = torch.linalg.vector_norm(
                torch.diff(
                    value.dense_joint_position[:horizon], n=3, dim=0
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
            if not reranked_candidate.contact_entry_valid:
                continue
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
                    reranked_candidate.dense_body_position[:46].clone()
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
        *,
        inherit_terrain_exit_elevation: bool = False,
        source_override: tuple[int, int] | None = None,
    ) -> _ComposedCandidate:
        if source_override is None:
            clip_index, frame_index = self._source_for_row(selected_row)
            transitioned = (
                incumbent_row is None or selected_row != incumbent_row
            )
        else:
            clip_index, frame_index = source_override
            transitioned = False
        placement = None
        contact_entry_valid = True
        policy = self._contact_segment_policy
        if transitioned:
            clip = self._clips[clip_index]
            root = self.folder.layout.root_body_index
            source_pos = clip.body_position[frame_index, root]
            yaw_offset = self._candidate_yaw_offset(
                clip_index, frame_index, shaped
            )
            desired_xy = (
                state.root_position[:2]
                + shaped.velocity_world_xy * self.config.dt
            )
            rotated = _rotate_z(source_pos, yaw_offset)
            translation = desired_xy - rotated[:2]
            translation_z = 0.0
            if (
                policy is not None
                and state.clip_index in policy.index.terrain_clip_indices
                and clip_index not in policy.index.terrain_clip_indices
                and inherit_terrain_exit_elevation
            ):
                translation_z = float(
                    (state.root_position[2] - source_pos[2]).item()
                )
            if (
                policy is not None
                and clip_index in policy.index.terrain_clip_indices
            ):
                current_support = policy.query_support_mask(
                    state.feature_body_position,
                    state.feature_body_velocity,
                )
                placement = policy.resolve_entry(
                    clip_index=clip_index,
                    frame_index=frame_index,
                    yaw_offset=yaw_offset,
                    translation_xy=translation,
                    current_support_mask=current_support,
                )
                contact_entry_valid = placement is not None
                if (
                    placement is not None
                    and self._terrain_command_deviated(state, shaped)
                    and not self._segment_command_compatible(
                        clip_index,
                        frame_index,
                        yaw_offset,
                        shaped,
                    )
                ):
                    placement = None
                    contact_entry_valid = False
                if placement is not None:
                    translation_z = placement.vertical_offset_m
        else:
            yaw_offset = state.yaw_offset
            translation = state.translation_xy
            translation_z = state.translation_z
            if policy is not None:
                contiguous = policy.index.entry(clip_index, frame_index)
                if (
                    contiguous is not None
                    and (
                        not self._terrain_command_deviated(state, shaped)
                        or self._segment_command_compatible(
                            clip_index,
                            frame_index,
                            yaw_offset,
                            shaped,
                        )
                    )
                ):
                    placement = SegmentPlacement(
                        segment=contiguous,
                        vertical_offset_m=translation_z,
                        source_support_mask=policy.index.support_mask(
                            clip_index
                        )[contiguous.start_frame : contiguous.end_frame],
                    )
        horizon = max(
            46,
            placement.segment.frame_count if placement is not None else 46,
        )
        targets = self._aligned_targets(
            clip_index,
            frame_index,
            yaw_offset,
            translation,
            translation_z=translation_z,
            horizon=horizon,
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
            torch.arange(horizon, device=self.device, dtype=torch.float32)
            * self.config.dt
            + offsets.elapsed_s
        )
        inertialization_halflife = self.config.inertialization_halflife_s
        if (
            transitioned
            and placement is not None
            and self._contact_segment_policy is not None
            and self._contact_segment_policy.entry_inertialization_halflife_s
            is not None
        ):
            inertialization_halflife = (
                self._contact_segment_policy.entry_inertialization_halflife_s
            )
        jpo, jvo = decay_spring_offsets(
            offsets.joint_position,
            offsets.joint_velocity,
            halflife_s=inertialization_halflife,
            time_s=times,
        )
        rpo, rvo = decay_spring_offsets(
            offsets.root_position,
            offsets.root_linear_velocity,
            halflife_s=inertialization_halflife,
            time_s=times,
        )
        bao, bvo = decay_spring_offsets(
            offsets.body_position,
            offsets.body_velocity,
            halflife_s=inertialization_halflife,
            time_s=times,
        )
        qao, qwo = decay_spring_offsets(
            offsets.root_rotation_axis,
            offsets.root_angular_velocity,
            halflife_s=inertialization_halflife,
            time_s=times,
        )
        dense_joint_position = jp + jpo
        dense_joint_velocity = jv + jvo
        dense_joint_position, dense_joint_velocity = (
            self._smooth_joint_reference(
                state,
                dense_joint_position,
                dense_joint_velocity,
                disable=(placement is not None or state.commitment is not None),
            )
        )
        dense_root_position = rp + rpo
        dense_root_quaternion = _quat_normalize(
            _quat_mul(_quat_from_scaled_axis(qao), rq)
        )
        dense_body_position = bp + bao
        dense_body_velocity = bv + bvo
        segment_validation = None
        if placement is not None:
            frame_count = placement.segment.frame_count
            segment_validation = policy.validate_emitted(
                placement,
                joint_position=dense_joint_position[:frame_count],
                root_position=dense_root_position[:frame_count],
                root_orientation_wxyz=dense_root_quaternion[:frame_count],
            )
            contact_entry_valid = segment_validation.accepted
        if (
            policy is not None
            and contact_entry_valid
            and (
                placement is not None
                or state.commitment is not None
                or (
                    not transitioned
                    and clip_index in policy.index.terrain_clip_indices
                )
            )
        ):
            feet = torch.tensor(
                policy.emitted_foot_positions(
                    dense_joint_position,
                    dense_root_position,
                    dense_root_quaternion,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            authoritative_body = torch.cat(
                (dense_root_position[:, None, :], feet), dim=1
            )
            authoritative_velocity = torch.empty_like(authoritative_body)
            authoritative_velocity[0] = dense_body_velocity[0]
            authoritative_velocity[1:] = (
                authoritative_body[1:] - authoritative_body[:-1]
            ) / self.config.dt
            dense_body_position = authoritative_body
            dense_body_velocity = authoritative_velocity

        return _ComposedCandidate(
            clip_index=clip_index,
            frame_index=frame_index,
            transitioned=transitioned,
            yaw_offset=yaw_offset,
            translation_xy=translation,
            translation_z=translation_z,
            segment_placement=placement,
            segment_validation=segment_validation,
            contact_entry_valid=contact_entry_valid,
            offsets=offsets,
            dense_joint_position=dense_joint_position,
            dense_joint_velocity=dense_joint_velocity,
            dense_root_position=dense_root_position,
            dense_root_quaternion=dense_root_quaternion,
            dense_root_linear_velocity=rv + rvo,
            dense_root_angular_velocity=rw + qwo,
            dense_body_position=dense_body_position,
            dense_body_velocity=dense_body_velocity,
        )

    def _segment_command_compatible(
        self,
        clip_index: int,
        frame_index: int,
        yaw_offset: torch.Tensor,
        shaped: ShapedCommand,
    ) -> bool:
        command = self._command_travel_direction(shaped)
        root = self.folder.layout.root_body_index
        policy = self._contact_segment_policy
        segment = (
            None
            if policy is None
            else policy.index.entry(clip_index, frame_index)
        )
        if segment is None:
            return False
        source_travel = _rotate_z(
            self._clips[clip_index].body_position[
                segment.end_frame - 1, root
            ]
            - self._clips[clip_index].body_position[frame_index, root],
            yaw_offset,
        )[:2]
        source_speed = torch.linalg.vector_norm(source_travel)
        if float(source_speed.item()) <= 1e-6:
            return False
        alignment = torch.dot(
            command,
            source_travel / source_speed,
        )
        return float(alignment.item()) >= (
            CONTIGUOUS_SEGMENT_COMMAND_ALIGNMENT_MIN
        )

    def _reachability_query(
        self,
        state: _MatcherState,
        history: Sequence[_SelectionVisit],
        requested_velocity: torch.Tensor,
        requested_heading: torch.Tensor,
    ) -> tuple[ShapedCommand, torch.Tensor, torch.Tensor, int | None]:
        successor = self.database.row_for_source(
            state.clip_index, state.frame_index + 1
        )
        active_commitment = (
            state.commitment is not None
            and state.frame_index + 1 < state.commitment.end_frame
        )
        shaped = predict_command_trajectory(
            state.root_position[:2],
            state.shaped_velocity,
            state.shaped_heading,
            requested_velocity,
            requested_heading,
            has_valid_successor=(successor is not None or active_commitment),
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
        continuity = self._continuity.costs(
            state.joint_position,
            state.joint_velocity,
            position_weight=self.config.transition_joint_position_weight,
            velocity_weight=self.config.transition_joint_velocity_weight,
        )
        loop_costs = _loop_revisit_transition_costs(
            self.database,
            history,
            state.root_position[:2],
            current_clip_index=state.clip_index,
            current_sequence=state.sequence,
            config=self.config,
        )
        transition_costs = (
            continuity.total + loop_costs
            if bool((loop_costs > 0.0).any().item())
            else continuity.total
        )
        if (
            self._contact_segment_policy is not None
            and self._contact_segment_policy.flat_support_transition_cost_weight
            > 0.0
        ):
            transition_costs = transition_costs + (
                self._flat_support_transition_costs(state, shaped)
            )
        return shaped, query, transition_costs, successor

    def _reachability_state_after(
        self,
        state: _MatcherState,
        candidate: _ComposedCandidate,
        shaped: ShapedCommand,
        requested_velocity: torch.Tensor,
        requested_heading: torch.Tensor,
        advance_frames: int,
    ) -> _MatcherState:
        velocity = shaped.velocity_world_xy
        heading = shaped.heading_world_yaw
        for index in range(advance_frames):
            next_frame = candidate.frame_index + index + 1
            next_shaped = predict_command_trajectory(
                candidate.dense_root_position[index + 1, :2],
                velocity,
                heading,
                requested_velocity,
                requested_heading,
                has_valid_successor=(
                    self.database.row_for_source(
                        candidate.clip_index, next_frame + 1
                    )
                    is not None
                ),
                config=self.config,
            )
            velocity = next_shaped.velocity_world_xy
            heading = next_shaped.heading_world_yaw
        index = advance_frames
        offsets = _Offsets(
            candidate.offsets.joint_position,
            candidate.offsets.joint_velocity,
            candidate.offsets.root_position,
            candidate.offsets.root_linear_velocity,
            candidate.offsets.root_rotation_axis,
            candidate.offsets.root_angular_velocity,
            candidate.offsets.body_position,
            candidate.offsets.body_velocity,
            candidate.offsets.elapsed_s + advance_frames * self.config.dt,
        )
        return _MatcherState(
            sequence=state.sequence + advance_frames + 1,
            clip_index=candidate.clip_index,
            frame_index=candidate.frame_index + advance_frames,
            yaw_offset=candidate.yaw_offset,
            translation_xy=candidate.translation_xy,
            translation_z=candidate.translation_z,
            shaped_velocity=velocity.clone(),
            shaped_heading=heading.clone(),
            joint_position=candidate.dense_joint_position[index].clone(),
            joint_velocity=candidate.dense_joint_velocity[index].clone(),
            root_position=candidate.dense_root_position[index].clone(),
            root_quaternion=candidate.dense_root_quaternion[index].clone(),
            root_linear_velocity=(
                candidate.dense_root_linear_velocity[index].clone()
            ),
            root_angular_velocity=(
                candidate.dense_root_angular_velocity[index].clone()
            ),
            feature_body_position=(
                candidate.dense_body_position[index].clone()
            ),
            feature_body_velocity=(
                candidate.dense_body_velocity[index].clone()
            ),
            offsets=offsets,
            commitment=state.commitment,
        )

    def _reachability_continuations(
        self,
        state: _MatcherState,
        history: Sequence[_SelectionVisit],
        candidate: _ComposedCandidate,
        shaped: ShapedCommand,
        requested_velocity: torch.Tensor,
        requested_heading: torch.Tensor,
        maximum_advance_frames: int,
    ) -> tuple[_ReachabilityContinuation, ...]:
        output = []
        for advance in range(1, maximum_advance_frames + 1):
            child_state = self._reachability_state_after(
                state,
                candidate,
                shaped,
                requested_velocity,
                requested_heading,
                advance - 1,
            )
            additions = tuple(
                _SelectionVisit(
                    sequence=state.sequence + offset + 1,
                    clip_index=candidate.clip_index,
                    frame_index=candidate.frame_index + offset,
                    root_position_xy=(
                        candidate.dense_root_position[offset, :2].clone()
                    ),
                )
                for offset in range(advance)
            )
            combined = tuple(history) + additions
            output.append(
                _ReachabilityContinuation(
                    child_state,
                    combined[-LOOP_HISTORY_FRAMES:],
                )
            )
        return tuple(output)

    def diagnose_transition_reachability(
        self,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
        *,
        command_change_frame: int,
        terminal_evaluator: TransitionTerminalEvaluator,
        safe_evaluator: EmittedWindowValidator,
        limits: ReachabilityLimits = ReachabilityLimits(),
    ) -> TransitionReachabilityDiagnostic:
        """Search two hypothetical transitions without committing matcher state."""

        state = self._state
        if state is None:
            raise ContractError(
                "reset must be called before transition reachability"
            )
        runtime_validator = self._emitted_window_validator
        if runtime_validator is None:
            raise ContractError(
                "transition reachability requires an emitted-window validator"
            )
        if safe_evaluator is None:
            raise ContractError(
                "transition reachability requires an explicit full-window "
                "safe evaluator"
            )
        diagnostic_validator = safe_evaluator
        if not callable(diagnostic_validator):
            raise TypeError("safe_evaluator must be callable")
        if not callable(terminal_evaluator):
            raise TypeError("terminal_evaluator must be callable")
        requested_velocity = torch.tensor(
            velocity_world_xy, dtype=torch.float32, device=self.device
        )
        requested_heading = torch.tensor(
            heading_world_yaw, dtype=torch.float32, device=self.device
        )

        def expand(
            parent: _MatcherState | _ReachabilityNode,
            _command,
            max_source_advance_frames: int,
        ):
            hypothetical = isinstance(parent, _ReachabilityNode)
            contexts = (
                parent.continuations
                if hypothetical
                else (
                    _ReachabilityContinuation(
                        parent,
                        tuple(self._selection_history),
                    ),
                )
            )
            ranked_contexts = []
            for context in contexts:
                shaped, query, transition_costs, successor = (
                    self._reachability_query(
                        context.state,
                        context.history,
                        requested_velocity,
                        requested_heading,
                    )
                )
                ranked = rank_exact_transition_candidates(
                    self.database,
                    query,
                    current_clip_index=context.state.clip_index,
                    current_frame_index=context.state.frame_index,
                    config=self.config,
                    additional_transition_costs=transition_costs,
                    transition_eligible_rows=(
                        self._transition_eligible_rows
                    ),
                )
                ranked_contexts.append(
                    (context, shaped, successor, ranked)
                )
            for rank in range(limits.beam_width):
                for context, shaped, successor, ranked in ranked_contexts:
                    if rank >= len(ranked):
                        continue
                    decision = ranked[rank]
                    candidate = self._compose_candidate(
                        context.state,
                        shaped,
                        decision.selected_row,
                        successor,
                    )
                    continuations = (
                        ()
                        if hypothetical
                        else self._reachability_continuations(
                            context.state,
                            context.history,
                            candidate,
                            shaped,
                            requested_velocity,
                            requested_heading,
                            max_source_advance_frames,
                        )
                    )
                    yield _ReachabilityNode(
                        context.state,
                        candidate,
                        continuations,
                    )

        def is_safe(node: _ReachabilityNode, _command) -> bool:
            accepted = diagnostic_validator(
                node.candidate.dense_body_position[:46].clone()
            )
            if type(accepted) is not bool:
                raise ContractError(
                    "emitted-window validator must return exact bool"
                )
            return accepted

        def is_terminal(node: _ReachabilityNode, _command) -> bool:
            terminal = terminal_evaluator(
                node.candidate.dense_body_position[:46].clone(),
                node.candidate.clip_index,
                node.candidate.frame_index,
                requested_velocity.clone(),
            )
            if type(terminal) is not bool:
                raise ContractError(
                    "transition terminal evaluator must return exact bool"
                )
            return terminal

        return bounded_transition_reachability(
            state,
            command=(
                requested_velocity.clone(),
                requested_heading.clone(),
            ),
            command_change_frame=command_change_frame,
            expand=expand,
            is_safe=is_safe,
            is_terminal=is_terminal,
            identity=lambda node: (
                node.candidate.clip_index,
                node.candidate.frame_index,
            ),
            limits=limits,
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
        segment_commitment: SegmentCommitment | None,
        segment_rejection_reason: str | None,
    ) -> MotionMatchResult:
        dense_joint_p = dense_joint_p[:46]
        dense_joint_v = dense_joint_v[:46]
        dense_root_p = dense_root_p[:46]
        dense_root_q = dense_root_q[:46]
        dense_body_p = dense_body_p[:46]
        dense_body_v = dense_body_v[:46]
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
            segment_committed=segment_commitment is not None,
            segment_start_frame=(
                None
                if segment_commitment is None
                else segment_commitment.start_frame
            ),
            segment_end_frame=(
                None
                if segment_commitment is None
                else segment_commitment.end_frame
            ),
            segment_entering_foot=(
                None
                if segment_commitment is None
                else segment_commitment.entering_foot
            ),
            segment_vertical_offset_m=(
                None
                if segment_commitment is None
                else segment_commitment.vertical_offset_m
            ),
            segment_rejection_reason=segment_rejection_reason,
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

    def _smooth_joint_reference(
        self,
        state: _MatcherState,
        dense_joint_position: torch.Tensor,
        dense_joint_velocity: torch.Tensor,
        *,
        disable: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = (
            0.0
            if disable
            else float(self.config.joint_reference_smoothing_weight)
        )
        if weight == 0.0:
            return dense_joint_position, dense_joint_velocity
        center = 1.0 - 2.0 * weight
        position = dense_joint_position.clone()
        velocity = dense_joint_velocity.clone()
        position[0] = (
            weight * state.joint_position
            + center * dense_joint_position[0]
            + weight * dense_joint_position[1]
        )
        velocity[0] = (
            weight * state.joint_velocity
            + center * dense_joint_velocity[0]
            + weight * dense_joint_velocity[1]
        )
        position[1:-1] = (
            weight * dense_joint_position[:-2]
            + center * dense_joint_position[1:-1]
            + weight * dense_joint_position[2:]
        )
        velocity[1:-1] = (
            weight * dense_joint_velocity[:-2]
            + center * dense_joint_velocity[1:-1]
            + weight * dense_joint_velocity[2:]
        )
        return position, velocity

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
            clip_index,
            frame_index,
            yaw_offset,
            translation,
            translation_z=0.0,
            horizon=46,
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
            segment_commitment=None,
            segment_rejection_reason=None,
        )
        self._state = _MatcherState(
            0, clip_index, frame_index, yaw_offset, translation, 0.0,
            torch.zeros(2, device=self.device),
            torch.zeros((), device=self.device),
            jp[0].clone(), jv[0].clone(), rp[0].clone(), rq[0].clone(),
            rv[0].clone(), rw[0].clone(), bp[0].clone(), bv[0].clone(), zeros,
            None,
        )
        self._selection_history = [
            _SelectionVisit(
                sequence=0,
                clip_index=clip_index,
                frame_index=frame_index,
                root_position_xy=rp[0, :2].clone(),
            )
        ]
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
        active_commitment = (
            state.commitment is not None
            and state.frame_index + 1 < state.commitment.end_frame
        )
        shaped = predict_command_trajectory(
            state.root_position[:2],
            state.shaped_velocity,
            state.shaped_heading,
            requested_v,
            requested_h,
            has_valid_successor=(successor is not None or active_commitment),
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
        interrupting_commitment = (
            active_commitment
            and self._contact_commitment_should_interrupt(state, shaped)
        )
        committed_playback = active_commitment and not interrupting_commitment
        settle_penalty = active_transition_penalty(
            state.offsets.elapsed_s, self.config
        )
        continuity = self._continuity.costs(
            state.joint_position,
            state.joint_velocity,
            position_weight=self.config.transition_joint_position_weight,
            velocity_weight=self.config.transition_joint_velocity_weight,
        )
        loop_costs = _loop_revisit_transition_costs(
            self.database,
            self._selection_history,
            state.root_position[:2],
            current_clip_index=state.clip_index,
            current_sequence=state.sequence,
            config=self.config,
        )
        transition_costs = (
            continuity.total + loop_costs
            if bool((loop_costs > 0.0).any().item())
            else continuity.total
        )
        if (
            self._contact_segment_policy is not None
            and self._contact_segment_policy.flat_support_transition_cost_weight
            > 0.0
        ):
            transition_costs = transition_costs + (
                self._flat_support_transition_costs(state, shaped)
            )
        transition_eligibility = self._command_transition_eligibility(
            state, shaped
        )
        terrain_transition_eligibility = self._command_terrain_entry_rows(
            state, shaped
        )
        if not committed_playback:
            (
                transition_eligibility,
                terrain_transition_eligibility,
                transition_costs,
            ) = self._foothold_conditioning(
                state, shaped, transition_costs
            )
        committed_source = None
        if committed_playback:
            commitment = state.commitment
            assert commitment is not None
            next_frame = state.frame_index + 1
            expected_source = (commitment.clip_index, next_frame)
            if (
                commitment.clip_index != state.clip_index
                or next_frame
                >= self._clips[commitment.clip_index].joint_position.shape[0]
            ):
                raise ContractError(
                    "active contact segment has no exact source successor"
                )
            diagnostic_row = successor
            if (
                diagnostic_row is None
                or self._source_for_row(diagnostic_row) != expected_source
            ):
                diagnostic_row = self.database.row_for_source(
                    commitment.clip_index,
                    self.folder.clips[
                        commitment.clip_index
                    ].valid_frame_stop
                    - 1,
                )
            if diagnostic_row is None:
                raise ContractError(
                    "active contact segment has no diagnostic source row"
                )
            feature_cost = float(
                torch.sum(
                    torch.square(
                        self.database._search_features[diagnostic_row] - query
                    )
                ).item()
            )
            decision = SearchDecision(
                selected_row=diagnostic_row,
                incumbent_row=diagnostic_row,
                incumbent_cost=feature_cost,
                selected_feature_cost=feature_cost,
                selected_total_cost=feature_cost,
                searched=False,
                transitioned=False,
            )
            search_time = None
            committed_source = expected_source
        else:
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
                additional_transition_costs=transition_costs,
                transition_eligible_rows=transition_eligibility,
            )
            search_time = (
                time.perf_counter_ns() - search_start if search else None
            )
        candidate = self._compose_candidate(
            state,
            shaped,
            decision.selected_row,
            successor,
            source_override=committed_source,
        )
        transition_rejected = False
        terrain_safety_override = False
        terrain_safety_override_rank = 0
        validator = self._emitted_window_validator
        policy = self._contact_segment_policy
        if (
            policy is not None
            and search
            and not committed_playback
            and (
                not bool(
                    self._terrain_entry_rows[decision.selected_row].item()
                )
                or not candidate.contact_entry_valid
            )
        ):
            terrain_rank_start = time.perf_counter_ns()
            terrain_entries = rank_exact_transition_candidates(
                self.database,
                query,
                current_clip_index=state.clip_index,
                current_frame_index=state.frame_index,
                config=self.config,
                additional_transition_costs=transition_costs,
                transition_eligible_rows=terrain_transition_eligibility,
            )
            terrain_rank_time = time.perf_counter_ns() - terrain_rank_start
            search_time = (
                terrain_rank_time
                if search_time is None
                else search_time + terrain_rank_time
            )
            for rank, terrain_decision in enumerate(
                itertools.islice(
                    terrain_entries,
                    self.config.transition_window_candidate_count,
                ),
                start=1,
            ):
                terrain_candidate = self._compose_candidate(
                    state,
                    shaped,
                    terrain_decision.selected_row,
                    successor,
                )
                if not terrain_candidate.contact_entry_valid:
                    continue
                if validator is not None:
                    terrain_accepted = validator(
                        terrain_candidate.dense_body_position[:46].clone()
                    )
                    if type(terrain_accepted) is not bool:
                        raise ContractError(
                            "emitted-window validator must return exact bool"
                        )
                    if not terrain_accepted:
                        continue
                candidate = terrain_candidate
                decision = SearchDecision(
                    selected_row=terrain_decision.selected_row,
                    incumbent_row=successor,
                    incumbent_cost=decision.incumbent_cost,
                    selected_feature_cost=(
                        terrain_decision.selected_feature_cost
                    ),
                    selected_total_cost=(
                        terrain_decision.selected_total_cost
                        + settle_penalty
                    ),
                    searched=True,
                    transitioned=True,
                    selected_transition_cost=(
                        terrain_decision.selected_transition_cost
                    ),
                )
                terrain_safety_override = True
                terrain_safety_override_rank = rank
                break
        segment_rejection_reason = None
        if not candidate.contact_entry_valid:
            segment_rejection_reason = (
                candidate.segment_validation.reason
                if candidate.segment_validation is not None
                else "incompatible_support_side"
            )
            rescue = self._ranked_terrain_rescue(
                state,
                shaped,
                query,
                transition_costs,
                transition_eligibility,
                successor,
                decision.incumbent_cost,
                validator,
                search_time,
                additional_transition_penalty=settle_penalty,
                maximum_total_cost=(
                    decision.incumbent_cost
                    if successor is not None
                    else None
                ),
                require_safe_candidate=successor is None,
            )
            if rescue is not None:
                (
                    terrain_safety_override_rank,
                    decision,
                    candidate,
                    search_time,
                ) = rescue
                terrain_safety_override = True
            else:
                assert successor is not None
                candidate = self._compose_candidate(
                    state, shaped, successor, successor
                )
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
        if validator is not None and not committed_playback:
            accepted = validator(candidate.dense_body_position[:46].clone())
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
                        transition_costs,
                        transition_eligibility,
                        successor,
                        decision.incumbent_cost,
                        validator,
                        search_time,
                        additional_transition_penalty=settle_penalty,
                    )
                    terrain_safety_override = True
                else:
                    if successor is None:
                        (
                            terrain_safety_override_rank,
                            decision,
                            candidate,
                            search_time,
                        ) = self._ranked_terrain_rescue(
                            state,
                            shaped,
                            query,
                            transition_costs,
                            transition_eligibility,
                            successor,
                            decision.incumbent_cost,
                            validator,
                            search_time,
                            additional_transition_penalty=settle_penalty,
                        )
                        terrain_safety_override = True
                    else:
                        incumbent = self._compose_candidate(
                            state, shaped, successor, successor
                        )
                        incumbent_accepted = validator(
                            incumbent.dense_body_position[:46].clone()
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
                                transition_costs,
                                transition_eligibility,
                                successor,
                                decision.incumbent_cost,
                                validator,
                                search_time,
                                additional_transition_penalty=settle_penalty,
                            )
                            terrain_safety_override = True
                        else:
                            rescue = self._ranked_terrain_rescue(
                                state,
                                shaped,
                                query,
                                transition_costs,
                                transition_eligibility,
                                successor,
                                decision.incumbent_cost,
                                validator,
                                search_time,
                                additional_transition_penalty=settle_penalty,
                                maximum_total_cost=decision.incumbent_cost,
                                require_safe_candidate=False,
                            )
                            if rescue is not None:
                                (
                                    terrain_safety_override_rank,
                                    decision,
                                    candidate,
                                    search_time,
                                ) = rescue
                                terrain_safety_override = True
                            else:
                                candidate = incumbent
                                decision = SearchDecision(
                                    selected_row=successor,
                                    incumbent_row=successor,
                                    incumbent_cost=decision.incumbent_cost,
                                    selected_feature_cost=(
                                        decision.incumbent_cost
                                    ),
                                    selected_total_cost=(
                                        decision.incumbent_cost
                                    ),
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
                transition_costs,
                transition_eligibility,
                successor,
                decision,
                candidate,
                validator,
                settle_penalty,
                search_time,
            )
        if (
            candidate.transitioned
            and policy is not None
            and state.clip_index in policy.index.terrain_clip_indices
            and candidate.clip_index
            not in policy.index.terrain_clip_indices
        ):
            candidate = self._compose_candidate(
                state,
                shaped,
                decision.selected_row,
                successor,
                inherit_terrain_exit_elevation=True,
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
        if committed_playback:
            force_reason = "segment_commitment"
        elif successor is None:
            force_reason = "clip_end"
        elif shaped.force_search:
            force_reason = "command_transition"
        dense_joint_p = candidate.dense_joint_position
        dense_joint_v = candidate.dense_joint_velocity
        output_commitment = (
            None
            if interrupting_commitment and candidate.transitioned
            else state.commitment
        )
        if (
            output_commitment is None
            and candidate.segment_placement is not None
            and candidate.contact_entry_valid
            and candidate.segment_validation is not None
            and candidate.segment_validation.accepted
        ):
            placement = candidate.segment_placement
            output_commitment = SegmentCommitment(
                clip_index=candidate.clip_index,
                start_frame=placement.segment.start_frame,
                end_frame=placement.segment.end_frame,
                entering_foot=placement.segment.entering_foot,
                vertical_offset_m=placement.vertical_offset_m,
            )
        if output_commitment is not None and not (
            output_commitment.clip_index == candidate.clip_index
            and output_commitment.start_frame
            <= candidate.frame_index
            < output_commitment.end_frame
        ):
            raise ContractError(
                "emitted frame is outside the active contact segment"
            )
        next_commitment = output_commitment
        if (
            output_commitment is not None
            and candidate.frame_index == output_commitment.end_frame - 1
        ):
            next_commitment = None
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
            dense_joint_p=dense_joint_p,
            dense_joint_v=dense_joint_v,
            dense_root_p=candidate.dense_root_position,
            dense_root_q=candidate.dense_root_quaternion,
            dense_body_p=candidate.dense_body_position,
            dense_body_v=candidate.dense_body_velocity,
            segment_commitment=output_commitment,
            segment_rejection_reason=segment_rejection_reason,
        )
        next_state = _MatcherState(
            state.sequence + 1,
            candidate.clip_index,
            candidate.frame_index,
            candidate.yaw_offset,
            candidate.translation_xy,
            candidate.translation_z,
            shaped.velocity_world_xy.clone(), shaped.heading_world_yaw.clone(),
            dense_joint_p[0].clone(),
            dense_joint_v[0].clone(),
            candidate.dense_root_position[0].clone(),
            candidate.dense_root_quaternion[0].clone(),
            candidate.dense_root_linear_velocity[0].clone(),
            candidate.dense_root_angular_velocity[0].clone(),
            candidate.dense_body_position[0].clone(),
            candidate.dense_body_velocity[0].clone(),
            candidate.offsets,
            next_commitment,
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
        self._selection_history.append(
            _SelectionVisit(
                sequence=self._state.sequence,
                clip_index=self._state.clip_index,
                frame_index=self._state.frame_index,
                root_position_xy=self._state.root_position[:2].clone(),
            )
        )
        if len(self._selection_history) > LOOP_HISTORY_FRAMES:
            del self._selection_history[
                : len(self._selection_history) - LOOP_HISTORY_FRAMES
            ]
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
