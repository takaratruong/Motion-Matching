"""Layered GPU ranking for fixed-horizon terrain-skill outcomes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Callable, Mapping

import torch

from .joints import ContractError
from .torch_motion_features import TorchMotionDatabase
from .torch_motion_matcher import (
    MatcherConfig,
    bounded_velocity_step,
    bounded_yaw_step,
)
from .torch_terrain_skill_horizons import (
    HORIZON_TARGET_FRAMES,
    MAXIMUM_ENDPOINT_LATENESS_FRAMES,
    TerrainSkillHorizonInventory,
)


_SEARCH_CONFIG_FIELDS = {
    "horizons",
    "maximum_endpoint_lateness_frames",
    "entry_weight",
    "displacement_weight",
    "yaw_weight",
    "height_weight",
    "duration_weight",
    "stall_weight",
    "moving_speed_mps",
    "minimum_progress_m",
    "turn_gate_rad",
    "surface_gate_m",
}


@dataclass(frozen=True)
class HorizonSearchConfig:
    # The entry term sums 27 normalized feature dimensions; average it so the
    # entry and low-dimensional outcome groups have comparable natural scale.
    entry_weight: float = 1.0 / 27.0
    displacement_weight: float = 8.0
    yaw_weight: float = 3.0
    height_weight: float = 4.0
    duration_weight: float = 0.25
    stall_weight: float = 0.05
    moving_speed_mps: float = 0.10
    minimum_progress_m: float = 0.05
    turn_gate_rad: float = math.radians(30.0)
    surface_gate_m: float = 0.05


@dataclass(frozen=True)
class HorizonTargets:
    frames: torch.Tensor
    displacement_local_xy: torch.Tensor
    yaw_delta_rad: torch.Tensor
    root_height_delta_m: torch.Tensor
    surface_height_delta_m: torch.Tensor


@dataclass(frozen=True)
class RankedHorizonCandidates:
    candidate_indices: torch.Tensor
    entry_cost: torch.Tensor
    displacement_cost: torch.Tensor
    yaw_cost: torch.Tensor
    height_cost: torch.Tensor
    duration_cost: torch.Tensor
    stall_cost: torch.Tensor
    outcome_cost: torch.Tensor
    total_cost: torch.Tensor
    rejected_by_reason: Mapping[str, int]


@dataclass(frozen=True)
class HorizonCost:
    entry: float
    displacement: float
    yaw: float
    height: float
    duration: float
    stall: float
    outcome: float
    total: float


@dataclass(frozen=True)
class TerrainSkillHorizonResult:
    record_index: int
    entry_row: int
    endpoint_frame_exclusive: int
    target_frames: int
    cost: HorizonCost
    rejected_by_reason: Mapping[str, int]
    selection_mode: str


class HorizonSearchFailure(ContractError):
    def __init__(self, rejected_by_reason: Mapping[str, int]):
        self.rejected_by_reason = MappingProxyType(dict(rejected_by_reason))
        super().__init__(
            "no terrain-compatible horizon candidate exists: "
            + ", ".join(
                f"{key}={value}"
                for key, value in self.rejected_by_reason.items()
            )
        )


TerrainHorizonValidator = Callable[[int, int, int], bool]


def horizon_search_config_from_experiment(
    config: Mapping,
) -> HorizonSearchConfig:
    """Load the exact qualified horizon weights from an experiment mapping."""

    descriptor = config.get("multi_horizon")
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != _SEARCH_CONFIG_FIELDS
    ):
        raise ContractError("multi_horizon fields are invalid")
    if descriptor["horizons"] != list(HORIZON_TARGET_FRAMES):
        raise ContractError(
            f"multi_horizon horizons must equal {list(HORIZON_TARGET_FRAMES)}"
        )
    if (
        descriptor["maximum_endpoint_lateness_frames"]
        != MAXIMUM_ENDPOINT_LATENESS_FRAMES
    ):
        raise ContractError(
            "multi_horizon endpoint lateness must equal "
            f"{MAXIMUM_ENDPOINT_LATENESS_FRAMES}"
        )
    values = {
        key: descriptor[key]
        for key in _SEARCH_CONFIG_FIELDS
        if key not in {"horizons", "maximum_endpoint_lateness_frames"}
    }
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values.values()
    ):
        raise ContractError(
            "multi_horizon weights and gates must be finite non-negative"
        )
    return HorizonSearchConfig(
        **{key: float(value) for key, value in values.items()}
    )


def _rotate_inverse_xy(values: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        (
            c * values[..., 0] + s * values[..., 1],
            -s * values[..., 0] + c * values[..., 1],
        ),
        dim=-1,
    )


def _wrapped_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def predict_horizon_targets(
    *,
    current_velocity_world_xy: torch.Tensor,
    current_heading_world_yaw: torch.Tensor,
    requested_velocity_world_xy: torch.Tensor,
    requested_heading_world_yaw: torch.Tensor,
    desired_root_height_delta_m: torch.Tensor | None = None,
    desired_surface_height_delta_m: torch.Tensor | None = None,
    matcher_config: MatcherConfig = MatcherConfig(),
    target_frames: tuple[int, ...] = HORIZON_TARGET_FRAMES,
) -> HorizonTargets:
    """Simulate the bounded command at requested 50-Hz frame horizons."""

    if (
        not isinstance(current_velocity_world_xy, torch.Tensor)
        or tuple(current_velocity_world_xy.shape) != (2,)
        or not isinstance(requested_velocity_world_xy, torch.Tensor)
        or tuple(requested_velocity_world_xy.shape) != (2,)
        or current_velocity_world_xy.device != requested_velocity_world_xy.device
        or current_velocity_world_xy.dtype != requested_velocity_world_xy.dtype
        or not current_velocity_world_xy.dtype.is_floating_point
        or not torch.isfinite(current_velocity_world_xy).all()
        or not torch.isfinite(requested_velocity_world_xy).all()
    ):
        raise ContractError("horizon command velocities are invalid")
    if (
        not isinstance(current_heading_world_yaw, torch.Tensor)
        or current_heading_world_yaw.numel() != 1
        or not isinstance(requested_heading_world_yaw, torch.Tensor)
        or requested_heading_world_yaw.numel() != 1
        or current_heading_world_yaw.device != current_velocity_world_xy.device
        or requested_heading_world_yaw.device != current_velocity_world_xy.device
        or current_heading_world_yaw.dtype != current_velocity_world_xy.dtype
        or requested_heading_world_yaw.dtype != current_velocity_world_xy.dtype
        or not isinstance(target_frames, tuple)
        or not target_frames
        or any(type(frame) is not int or frame <= 0 for frame in target_frames)
        or tuple(sorted(set(target_frames))) != target_frames
    ):
        raise ContractError("horizon command headings or frames are invalid")

    velocity = current_velocity_world_xy
    heading = current_heading_world_yaw.reshape(())
    initial_heading = heading
    displacement_world = torch.zeros_like(velocity)
    sampled_displacement: list[torch.Tensor] = []
    sampled_yaw: list[torch.Tensor] = []
    for frame in range(1, target_frames[-1] + 1):
        velocity = bounded_velocity_step(
            velocity, requested_velocity_world_xy, config=matcher_config
        )
        heading = bounded_yaw_step(
            heading, requested_heading_world_yaw, config=matcher_config
        )
        displacement_world = displacement_world + velocity * matcher_config.dt
        if frame in target_frames:
            sampled_displacement.append(
                _rotate_inverse_xy(displacement_world, initial_heading)
            )
            sampled_yaw.append(_wrapped_angle(heading - initial_heading))

    device = current_velocity_world_xy.device
    dtype = current_velocity_world_xy.dtype
    frames = torch.tensor(target_frames, dtype=torch.long, device=device)
    target_count = len(target_frames)
    if desired_root_height_delta_m is None:
        desired_root_height_delta_m = torch.zeros(
            target_count, dtype=dtype, device=device
        )
    if desired_surface_height_delta_m is None:
        desired_surface_height_delta_m = torch.zeros(
            (target_count, 2), dtype=dtype, device=device
        )
    if (
        not isinstance(desired_root_height_delta_m, torch.Tensor)
        or tuple(desired_root_height_delta_m.shape) != (target_count,)
        or desired_root_height_delta_m.device != device
        or desired_root_height_delta_m.dtype != dtype
        or not torch.isfinite(desired_root_height_delta_m).all()
        or not isinstance(desired_surface_height_delta_m, torch.Tensor)
        or tuple(desired_surface_height_delta_m.shape) != (target_count, 2)
        or desired_surface_height_delta_m.device != device
        or desired_surface_height_delta_m.dtype != dtype
        or not torch.isfinite(desired_surface_height_delta_m).all()
    ):
        raise ContractError("horizon desired height targets are invalid")
    return HorizonTargets(
        frames=frames,
        displacement_local_xy=torch.stack(sampled_displacement),
        yaw_delta_rad=torch.stack(sampled_yaw),
        root_height_delta_m=desired_root_height_delta_m.clone(),
        surface_height_delta_m=desired_surface_height_delta_m.clone(),
    )


def _validate_ranking_inputs(
    database: TorchMotionDatabase,
    inventory: TerrainSkillHorizonInventory,
    normalized_query: torch.Tensor,
    targets: HorizonTargets,
    config: HorizonSearchConfig,
) -> None:
    if not isinstance(database, TorchMotionDatabase):
        raise ContractError("horizon search requires a TorchMotionDatabase")
    if not isinstance(inventory, TerrainSkillHorizonInventory):
        raise ContractError("horizon search requires a horizon inventory")
    if not isinstance(targets, HorizonTargets):
        raise ContractError("horizon search requires HorizonTargets")
    features = database._search_features
    if (
        not isinstance(normalized_query, torch.Tensor)
        or tuple(normalized_query.shape) != (features.shape[1],)
        or normalized_query.device != database.device
        or normalized_query.dtype != torch.float32
        or not torch.isfinite(normalized_query).all()
    ):
        raise ContractError("horizon normalized query is invalid")
    count = inventory.record_count
    aligned = (
        inventory.entry_row,
        inventory.skill_index,
        inventory.clip_index,
        inventory.entry_frame,
        inventory.target_frames,
        inventory.endpoint_frame_exclusive,
        inventory.yaw_delta_rad,
        inventory.root_height_delta_m,
        inventory.maximum_stall_frames,
        inventory.duration_frames,
    )
    if count < 1 or any(tuple(value.shape) != (count,) for value in aligned):
        raise ContractError("horizon inventory scalar arrays are misaligned")
    if (
        tuple(inventory.root_displacement_local_xy.shape) != (count, 2)
        or tuple(inventory.surface_height_delta_m.shape) != (count, 2)
        or any(value.device != database.device for value in aligned)
        or inventory.root_displacement_local_xy.device != database.device
        or inventory.surface_height_delta_m.device != database.device
    ):
        raise ContractError("horizon inventory vector arrays are misaligned")
    horizon_count = int(targets.frames.shape[0])
    if (
        targets.frames.ndim != 1
        or tuple(targets.displacement_local_xy.shape) != (horizon_count, 2)
        or tuple(targets.yaw_delta_rad.shape) != (horizon_count,)
        or tuple(targets.root_height_delta_m.shape) != (horizon_count,)
        or tuple(targets.surface_height_delta_m.shape) != (horizon_count, 2)
        or any(
            value.device != database.device
            for value in (
                targets.frames,
                targets.displacement_local_xy,
                targets.yaw_delta_rad,
                targets.root_height_delta_m,
                targets.surface_height_delta_m,
            )
        )
    ):
        raise ContractError("horizon target arrays are misaligned")
    numeric = tuple(vars(config).values())
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in numeric):
        raise ContractError("horizon search configuration is invalid")


def rank_horizon_candidates(
    database: TorchMotionDatabase,
    inventory: TerrainSkillHorizonInventory,
    normalized_query: torch.Tensor,
    targets: HorizonTargets,
    config: HorizonSearchConfig = HorizonSearchConfig(),
    *,
    current_clip_index: int | None = None,
    current_frame_index: int | None = None,
    matcher_config: MatcherConfig = MatcherConfig(),
) -> RankedHorizonCandidates:
    """Hard-gate and stably rank all row/horizon records on the device."""

    _validate_ranking_inputs(database, inventory, normalized_query, targets, config)
    matches = inventory.target_frames[:, None] == targets.frames[None, :]
    known_target = matches.any(dim=1)
    target_index = torch.argmax(matches.to(torch.long), dim=1)
    desired_displacement = targets.displacement_local_xy[target_index]
    desired_yaw = targets.yaw_delta_rad[target_index]
    desired_root_height = targets.root_height_delta_m[target_index]
    desired_surface = targets.surface_height_delta_m[target_index]

    entry = config.entry_weight * torch.sum(
        torch.square(
            database._search_features[inventory.entry_row]
            - normalized_query.unsqueeze(0)
        ),
        dim=1,
    )
    displacement = config.displacement_weight * torch.sum(
        torch.square(
            (inventory.root_displacement_local_xy - desired_displacement) / 0.5
        ),
        dim=1,
    )
    yaw_error = _wrapped_angle(inventory.yaw_delta_rad - desired_yaw)
    yaw = config.yaw_weight * torch.square(yaw_error / (math.pi / 2.0))
    height_components = torch.cat(
        (
            ((inventory.root_height_delta_m - desired_root_height) / 0.18)[:, None],
            (inventory.surface_height_delta_m - desired_surface) / 0.18,
        ),
        dim=1,
    )
    height = config.height_weight * torch.mean(
        torch.square(height_components), dim=1
    )
    duration = config.duration_weight * torch.square(
        (inventory.duration_frames.to(torch.float32) - inventory.target_frames) / 100.0
    )
    stall = config.stall_weight * torch.square(
        inventory.maximum_stall_frames.to(torch.float32) / 25.0
    )
    outcome = displacement + yaw + height + duration + stall
    total = entry + outcome

    desired_turning = torch.abs(desired_yaw) >= config.turn_gate_rad
    rejected_turn = desired_turning & (inventory.yaw_delta_rad * desired_yaw <= 0)
    desired_speed = torch.linalg.vector_norm(desired_displacement, dim=1) / (
        inventory.target_frames.to(torch.float32) * 0.02
    )
    rejected_progress = (desired_speed >= config.moving_speed_mps) & (
        torch.linalg.vector_norm(inventory.root_displacement_local_xy, dim=1)
        <= config.minimum_progress_m
    )
    desired_surface_mean = desired_surface.mean(dim=1)
    candidate_surface_mean = inventory.surface_height_delta_m.mean(dim=1)
    rejected_surface = (
        torch.abs(desired_surface_mean) >= config.surface_gate_m
    ) & (desired_surface_mean * candidate_surface_mean <= 0)
    finite = (
        known_target
        & torch.isfinite(entry)
        & torch.isfinite(outcome)
        & torch.isfinite(total)
    )
    if (current_clip_index is None) != (current_frame_index is None):
        raise ContractError("horizon current source must be fully specified")
    if current_clip_index is None:
        rejected_local = torch.zeros_like(finite)
    else:
        if (
            type(current_clip_index) is not int
            or type(current_frame_index) is not int
            or current_clip_index < 0
            or current_frame_index < 0
            or matcher_config.exclusion_frames < 0
        ):
            raise ContractError("horizon current source is invalid")
        rejected_local = (
            inventory.clip_index == current_clip_index
        ) & (
            torch.abs(inventory.entry_frame - current_frame_index)
            <= matcher_config.exclusion_frames
        )
    eligible = (
        finite
        & ~rejected_turn
        & ~rejected_progress
        & ~rejected_surface
        & ~rejected_local
    )
    candidates = torch.nonzero(eligible, as_tuple=False).flatten()
    order = torch.argsort(total[candidates], stable=True)
    candidates = candidates[order]

    def selected(values: torch.Tensor) -> torch.Tensor:
        return values[candidates]

    rejected = MappingProxyType(
        {
            "turn": int(rejected_turn.sum().item()),
            "progress": int(rejected_progress.sum().item()),
            "surface": int(rejected_surface.sum().item()),
            "local": int(rejected_local.sum().item()),
            "nonfinite": int((~finite).sum().item()),
        }
    )
    return RankedHorizonCandidates(
        candidate_indices=candidates,
        entry_cost=selected(entry),
        displacement_cost=selected(displacement),
        yaw_cost=selected(yaw),
        height_cost=selected(height),
        duration_cost=selected(duration),
        stall_cost=selected(stall),
        outcome_cost=selected(outcome),
        total_cost=selected(total),
        rejected_by_reason=rejected,
    )


def select_horizon_candidate(
    database: TorchMotionDatabase,
    inventory: TerrainSkillHorizonInventory,
    normalized_query: torch.Tensor,
    targets: HorizonTargets,
    *,
    terrain_validator: TerrainHorizonValidator,
    preferred_validator: TerrainHorizonValidator | None = None,
    maximum_preferred_cost_increase: float = math.inf,
    maximum_preferred_outcome_cost_increase: float = math.inf,
    config: HorizonSearchConfig = HorizonSearchConfig(),
    current_clip_index: int | None = None,
    current_frame_index: int | None = None,
    matcher_config: MatcherConfig = MatcherConfig(),
) -> TerrainSkillHorizonResult:
    """Validate candidates in ranked order and return the first valid chunk."""

    if not callable(terrain_validator) or (
        preferred_validator is not None and not callable(preferred_validator)
    ):
        raise ContractError("horizon terrain validator must be callable")
    if (
        math.isnan(float(maximum_preferred_cost_increase))
        or float(maximum_preferred_cost_increase) < 0.0
        or math.isnan(float(maximum_preferred_outcome_cost_increase))
        or float(maximum_preferred_outcome_cost_increase) < 0.0
    ):
        raise ContractError("preferred horizon cost budget is invalid")
    ranked = rank_horizon_candidates(
        database,
        inventory,
        normalized_query,
        targets,
        config,
        current_clip_index=current_clip_index,
        current_frame_index=current_frame_index,
        matcher_config=matcher_config,
    )
    terrain_rejected = 0
    preferred_rejected = 0
    diagnostics = torch.stack(
        (
            ranked.candidate_indices.to(torch.float64),
            ranked.entry_cost.to(torch.float64),
            ranked.displacement_cost.to(torch.float64),
            ranked.yaw_cost.to(torch.float64),
            ranked.height_cost.to(torch.float64),
            ranked.duration_cost.to(torch.float64),
            ranked.stall_cost.to(torch.float64),
            ranked.outcome_cost.to(torch.float64),
            ranked.total_cost.to(torch.float64),
        ),
        dim=1,
    ).cpu().tolist()
    fallback: tuple[list[float], int, int, int] | None = None

    def build_result(
        values: list[float],
        record: int,
        row: int,
        endpoint: int,
        *,
        selection_mode: str,
    ) -> TerrainSkillHorizonResult:
        rejected = dict(ranked.rejected_by_reason)
        rejected["terrain"] = terrain_rejected
        if preferred_validator is not None:
            rejected["preferred"] = preferred_rejected
        return TerrainSkillHorizonResult(
            record_index=record,
            entry_row=row,
            endpoint_frame_exclusive=endpoint,
            target_frames=int(inventory.target_frames[record].item()),
            cost=HorizonCost(
                entry=float(values[1]),
                displacement=float(values[2]),
                yaw=float(values[3]),
                height=float(values[4]),
                duration=float(values[5]),
                stall=float(values[6]),
                outcome=float(values[7]),
                total=float(values[8]),
            ),
            rejected_by_reason=MappingProxyType(rejected),
            selection_mode=selection_mode,
        )

    for values in diagnostics:
        if fallback is not None and values[8] > (
            fallback[0][8] + float(maximum_preferred_cost_increase)
        ):
            break
        record = int(values[0])
        row = int(inventory.entry_row[record].item())
        endpoint = int(inventory.endpoint_frame_exclusive[record].item())
        if terrain_validator(record, row, endpoint):
            if preferred_validator is None:
                return build_result(
                    values, record, row, endpoint, selection_mode="immediate"
                )
            if fallback is None:
                fallback = (values, record, row, endpoint)
            if values[7] > (
                fallback[0][7]
                + float(maximum_preferred_outcome_cost_increase)
            ):
                preferred_rejected += 1
                continue
            if preferred_validator(record, row, endpoint):
                return build_result(
                    values, record, row, endpoint, selection_mode="preferred"
                )
            preferred_rejected += 1
        else:
            terrain_rejected += 1
    if fallback is not None:
        values, record, row, endpoint = fallback
        return build_result(
            values,
            record,
            row,
            endpoint,
            selection_mode="immediate-fallback",
        )
    rejected = dict(ranked.rejected_by_reason)
    rejected["terrain"] = terrain_rejected
    raise HorizonSearchFailure(rejected)
