"""Bounded operator commands and exact dense Torch motion search."""

from __future__ import annotations

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
    dense_joint_position_window: torch.Tensor
    dense_joint_velocity_window: torch.Tensor
    dense_root_position_window: torch.Tensor
    dense_root_orientation_window_wxyz: torch.Tensor
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
        database = TorchMotionDatabase.from_folder(folder, device=resolved)
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
            dense_joint_position_window=dense_joint_p.clone(),
            dense_joint_velocity_window=dense_joint_v.clone(),
            dense_root_position_window=dense_root_p.clone(),
            dense_root_orientation_window_wxyz=dense_root_q.clone(),
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
        query = self.database.normalization.normalize(
            extract_query_features(feature_state, shaped.trajectory)
        )
        search = search_is_due(
            state.sequence, shaped.force_search, self.config
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
        )
        search_time = time.perf_counter_ns() - search_start if search else None
        clip_index, frame_index = self._source_for_row(decision.selected_row)
        transitioned = decision.transitioned or successor is None
        if transitioned:
            clip = self._clips[clip_index]
            root = self.folder.layout.root_body_index
            source_pos = clip.body_position[frame_index, root]
            source_yaw = _quat_yaw(clip.body_quaternion[frame_index, root])
            yaw_offset = _wrapped_angle(shaped.heading_world_yaw - source_yaw)
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
                    _quat_mul(state.root_quaternion, _quat_inverse(rq[0]))
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
        force_reason = None
        if successor is None:
            force_reason = "clip_end"
        elif shaped.force_search:
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
        )
        next_state = _MatcherState(
            state.sequence + 1, clip_index, frame_index, yaw_offset, translation,
            shaped.velocity_world_xy.clone(), shaped.heading_world_yaw.clone(),
            dense_jp[0].clone(), dense_jv[0].clone(), dense_rp[0].clone(),
            dense_rq[0].clone(), dense_rv[0].clone(), dense_rw[0].clone(),
            dense_bp[0].clone(), dense_bv[0].clone(), offsets,
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
