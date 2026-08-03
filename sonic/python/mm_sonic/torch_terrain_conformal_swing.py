"""Deterministic terrain-conformal optimization of one foot swing path."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from .joints import ContractError


def _finite_number(value: object, *, name: str, positive: bool) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
        or (not positive and float(value) < 0.0)
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ContractError(f"terrain-conformal swing {name} must be {qualifier}")
    return float(value)


@dataclass(frozen=True)
class TerrainConformalSwingConfig:
    clearance_margin_m: float = 0.025
    edge_probe_m: float = 0.04
    maximum_edge_height_range_m: float = 0.025
    reference_weight: float = 1.0
    smoothness_weight: float = 4.0
    clearance_weight: float = 200.0
    edge_weight: float = 100.0
    endpoint_weight: float = 1000.0
    sample_count: int = 128
    iteration_count: int = 8
    temperature: float = 0.05
    perturbation_xy_std_m: float = 0.01
    perturbation_z_std_m: float = 0.04
    maximum_xy_deformation_m: float = 0.04
    maximum_z_deformation_m: float = 0.16

    def __post_init__(self) -> None:
        for name in (
            "clearance_margin_m",
            "edge_probe_m",
            "maximum_edge_height_range_m",
        ):
            value = _finite_number(
                getattr(self, name),
                name=name.removesuffix("_m").replace("_", " "),
                positive=True,
            )
            object.__setattr__(self, name, value)
        for name in (
            "sample_count",
            "iteration_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ContractError(
                    "terrain-conformal swing "
                    + name.replace("_", " ")
                    + " must be a positive integer"
                )
        for name in (
            "temperature",
            "perturbation_xy_std_m",
            "perturbation_z_std_m",
            "maximum_xy_deformation_m",
            "maximum_z_deformation_m",
        ):
            value = _finite_number(
                getattr(self, name),
                name=name.removesuffix("_m").replace("_", " "),
                positive=True,
            )
            object.__setattr__(self, name, value)
        for name in (
            "reference_weight",
            "smoothness_weight",
            "clearance_weight",
            "edge_weight",
            "endpoint_weight",
        ):
            value = _finite_number(
                getattr(self, name),
                name=name.replace("_", " "),
                positive=False,
            )
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class TerrainConformalSwingCost:
    reference: torch.Tensor
    smoothness: torch.Tensor
    clearance: torch.Tensor
    edge: torch.Tensor
    endpoint: torch.Tensor
    total: torch.Tensor

    def __post_init__(self) -> None:
        values = (
            self.reference,
            self.smoothness,
            self.clearance,
            self.edge,
            self.endpoint,
            self.total,
        )
        reference = self.total
        if (
            not isinstance(reference, torch.Tensor)
            or reference.ndim != 1
            or not reference.dtype.is_floating_point
            or not torch.isfinite(reference).all()
            or any(
                not isinstance(value, torch.Tensor)
                or value.shape != reference.shape
                or value.dtype != reference.dtype
                or value.device != reference.device
                or not torch.isfinite(value).all()
                or bool((value < 0.0).any().item())
                for value in values
            )
        ):
            raise ContractError("terrain-conformal swing cost is invalid")
        for name in (
            "reference",
            "smoothness",
            "clearance",
            "edge",
            "endpoint",
            "total",
        ):
            object.__setattr__(self, name, getattr(self, name).detach().clone())


@dataclass(frozen=True)
class TerrainConformalSwingResult:
    path: torch.Tensor
    raw_cost: TerrainConformalSwingCost
    optimized_cost: TerrainConformalSwingCost
    improved: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, torch.Tensor)
            or self.path.ndim != 2
            or tuple(self.path.shape[1:]) != (3,)
            or self.path.shape[0] < 3
            or self.path.dtype != torch.float32
            or not torch.isfinite(self.path).all()
            or not isinstance(self.raw_cost, TerrainConformalSwingCost)
            or not isinstance(self.optimized_cost, TerrainConformalSwingCost)
            or self.raw_cost.total.shape != torch.Size((1,))
            or self.optimized_cost.total.shape != torch.Size((1,))
            or self.raw_cost.total.device != self.path.device
            or self.optimized_cost.total.device != self.path.device
            or type(self.improved) is not bool
        ):
            raise ContractError("terrain-conformal swing result is invalid")
        object.__setattr__(self, "path", self.path.detach().clone())


def _validate_cost_inputs(
    paths: torch.Tensor,
    raw_path: torch.Tensor,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    toe_offset_xy: torch.Tensor,
    heel_offset_xy: torch.Tensor,
    config: TerrainConformalSwingConfig,
) -> None:
    if (
        not isinstance(paths, torch.Tensor)
        or paths.ndim != 3
        or paths.shape[0] < 1
        or paths.shape[1] < 3
        or tuple(paths.shape[2:]) != (3,)
        or paths.dtype != torch.float32
        or not torch.isfinite(paths).all()
        or not isinstance(raw_path, torch.Tensor)
        or tuple(raw_path.shape) != (paths.shape[1], 3)
        or raw_path.dtype != paths.dtype
        or raw_path.device != paths.device
        or not torch.isfinite(raw_path).all()
        or not callable(sample_surface)
        or not isinstance(toe_offset_xy, torch.Tensor)
        or tuple(toe_offset_xy.shape) not in ((2,), (paths.shape[1], 2))
        or toe_offset_xy.dtype != paths.dtype
        or toe_offset_xy.device != paths.device
        or not torch.isfinite(toe_offset_xy).all()
        or not isinstance(heel_offset_xy, torch.Tensor)
        or tuple(heel_offset_xy.shape) not in ((2,), (paths.shape[1], 2))
        or heel_offset_xy.dtype != paths.dtype
        or heel_offset_xy.device != paths.device
        or not torch.isfinite(heel_offset_xy).all()
        or not isinstance(config, TerrainConformalSwingConfig)
    ):
        raise ContractError("terrain-conformal swing cost inputs are invalid")


def terrain_conformal_swing_cost(
    paths: torch.Tensor,
    raw_path: torch.Tensor,
    *,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    toe_offset_xy: torch.Tensor,
    heel_offset_xy: torch.Tensor,
    config: TerrainConformalSwingConfig,
) -> TerrainConformalSwingCost:
    """Score batched mid-foot trajectories against one terrain surface."""

    _validate_cost_inputs(
        paths,
        raw_path,
        sample_surface,
        toe_offset_xy,
        heel_offset_xy,
        config,
    )
    toe = (
        toe_offset_xy[None].expand(paths.shape[1], 2)
        if toe_offset_xy.ndim == 1
        else toe_offset_xy
    )
    heel = (
        heel_offset_xy[None].expand(paths.shape[1], 2)
        if heel_offset_xy.ndim == 1
        else heel_offset_xy
    )
    axis = toe - heel
    norm = torch.linalg.vector_norm(axis, dim=1)
    if bool((norm <= 1.0e-6).any().item()):
        raise ContractError("terrain-conformal swing foot offsets are degenerate")
    lateral = torch.stack((-axis[:, 1], axis[:, 0]), dim=1) / norm[:, None]
    offsets = torch.stack(
        (
            torch.zeros_like(toe),
            toe,
            heel,
            toe + lateral * float(config.edge_probe_m),
            heel - lateral * float(config.edge_probe_m),
        ),
        dim=1,
    )
    sample_xy = paths[..., None, :2] + offsets[None]
    try:
        surface = sample_surface(sample_xy)
    except Exception as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(
            "terrain-conformal swing surface sampler failed"
        ) from error
    if (
        not isinstance(surface, torch.Tensor)
        or surface.shape != sample_xy.shape[:-1]
        or surface.dtype != paths.dtype
        or surface.device != paths.device
        or not torch.isfinite(surface).all()
    ):
        raise ContractError(
            "terrain-conformal swing surface sampler returned invalid heights"
        )

    reference = torch.mean((paths - raw_path[None]).square(), dim=(1, 2))
    second_difference = paths[:, 2:] - 2.0 * paths[:, 1:-1] + paths[:, :-2]
    smoothness = torch.mean(second_difference.square(), dim=(1, 2))
    point_height = paths[..., 2, None]
    clearance_deficit = torch.relu(
        surface[..., :3]
        + float(config.clearance_margin_m)
        - point_height
    )
    clearance = torch.mean(clearance_deficit.square(), dim=(1, 2))
    height_range = surface.max(dim=2).values - surface.min(dim=2).values
    edge = torch.mean(
        torch.relu(
            height_range - float(config.maximum_edge_height_range_m)
        ).square(),
        dim=1,
    )
    endpoint = torch.mean(
        torch.stack(
            (
                (paths[:, 0] - raw_path[0]).square(),
                (paths[:, -1] - raw_path[-1]).square(),
            ),
            dim=1,
        ),
        dim=(1, 2),
    )
    total = (
        float(config.reference_weight) * reference
        + float(config.smoothness_weight) * smoothness
        + float(config.clearance_weight) * clearance
        + float(config.edge_weight) * edge
        + float(config.endpoint_weight) * endpoint
    )
    return TerrainConformalSwingCost(
        reference=reference,
        smoothness=smoothness,
        clearance=clearance,
        edge=edge,
        endpoint=endpoint,
        total=total,
    )


def _smooth_perturbations(values: torch.Tensor) -> torch.Tensor:
    weights = (1.0, 2.0, 3.0, 2.0, 1.0)
    padded = torch.cat(
        (
            values[:, :1].expand(-1, 2, -1),
            values,
            values[:, -1:].expand(-1, 2, -1),
        ),
        dim=1,
    )
    return sum(
        weight * padded[:, offset : offset + values.shape[1]]
        for offset, weight in enumerate(weights)
    ) / sum(weights)


def _bounded_candidates(
    center: torch.Tensor,
    raw: torch.Tensor,
    perturbation: torch.Tensor,
    config: TerrainConformalSwingConfig,
) -> torch.Tensor:
    candidates = center[None] + perturbation
    delta = candidates - raw[None]
    planar = torch.clamp(
        delta[..., :2],
        min=-float(config.maximum_xy_deformation_m),
        max=float(config.maximum_xy_deformation_m),
    )
    vertical = torch.clamp(
        delta[..., 2:],
        min=-float(config.maximum_z_deformation_m),
        max=float(config.maximum_z_deformation_m),
    )
    candidates = raw[None] + torch.cat((planar, vertical), dim=2)
    candidates[:, 0] = raw[0]
    candidates[:, -1] = raw[-1]
    return candidates


def optimize_terrain_conformal_swing(
    raw_path: torch.Tensor,
    swing_mask: torch.Tensor,
    *,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    toe_offset_xy: torch.Tensor,
    heel_offset_xy: torch.Tensor,
    config: TerrainConformalSwingConfig,
    seed: int,
) -> TerrainConformalSwingResult:
    """Optimize one contiguous swing and preserve its timing and endpoints."""

    if (
        not isinstance(raw_path, torch.Tensor)
        or raw_path.ndim != 2
        or tuple(raw_path.shape[1:]) != (3,)
        or raw_path.shape[0] < 3
        or raw_path.dtype != torch.float32
        or not torch.isfinite(raw_path).all()
        or not isinstance(swing_mask, torch.Tensor)
        or swing_mask.shape != raw_path.shape[:1]
        or swing_mask.dtype != torch.bool
        or swing_mask.device != raw_path.device
        or not isinstance(config, TerrainConformalSwingConfig)
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise ContractError("terrain-conformal swing optimizer inputs are invalid")
    indices = torch.nonzero(swing_mask, as_tuple=False).flatten()
    if indices.numel() < 3:
        raise ContractError(
            "terrain-conformal swing mask must contain at least three samples"
        )
    start = int(indices[0].item())
    stop = int(indices[-1].item()) + 1
    expected = torch.arange(start, stop, device=raw_path.device)
    if not bool(torch.equal(indices, expected)):
        raise ContractError("terrain-conformal swing mask must be contiguous")

    raw_swing = raw_path[start:stop]
    toe_swing = (
        toe_offset_xy[start:stop]
        if isinstance(toe_offset_xy, torch.Tensor)
        and toe_offset_xy.ndim == 2
        and toe_offset_xy.shape[0] == raw_path.shape[0]
        else toe_offset_xy
    )
    heel_swing = (
        heel_offset_xy[start:stop]
        if isinstance(heel_offset_xy, torch.Tensor)
        and heel_offset_xy.ndim == 2
        and heel_offset_xy.shape[0] == raw_path.shape[0]
        else heel_offset_xy
    )
    raw_cost = terrain_conformal_swing_cost(
        raw_swing[None],
        raw_swing,
        sample_surface=sample_surface,
        toe_offset_xy=toe_swing,
        heel_offset_xy=heel_swing,
        config=config,
    )
    current = raw_swing.clone()
    best = current.clone()
    best_total = raw_cost.total[0].clone()
    generator = torch.Generator(device=raw_path.device)
    generator.manual_seed(seed)
    standard_deviation = torch.tensor(
        (
            float(config.perturbation_xy_std_m),
            float(config.perturbation_xy_std_m),
            float(config.perturbation_z_std_m),
        ),
        dtype=raw_path.dtype,
        device=raw_path.device,
    )
    for _ in range(config.iteration_count):
        perturbation = torch.randn(
            (
                config.sample_count,
                raw_swing.shape[0],
                3,
            ),
            dtype=raw_path.dtype,
            device=raw_path.device,
            generator=generator,
        ) * standard_deviation
        perturbation = _smooth_perturbations(perturbation)
        perturbation[:, 0] = 0.0
        perturbation[:, -1] = 0.0
        candidates = _bounded_candidates(
            current, raw_swing, perturbation, config
        )
        costs = terrain_conformal_swing_cost(
            candidates,
            raw_swing,
            sample_surface=sample_surface,
            toe_offset_xy=toe_swing,
            heel_offset_xy=heel_swing,
            config=config,
        )
        minimum = costs.total.min()
        weights = torch.softmax(
            -(costs.total - minimum) / float(config.temperature), dim=0
        )
        proposal = torch.sum(weights[:, None, None] * candidates, dim=0)
        proposal[0] = raw_swing[0]
        proposal[-1] = raw_swing[-1]
        proposal_cost = terrain_conformal_swing_cost(
            proposal[None],
            raw_swing,
            sample_surface=sample_surface,
            toe_offset_xy=toe_swing,
            heel_offset_xy=heel_swing,
            config=config,
        )
        sample_index = int(torch.argmin(costs.total).item())
        sample_total = costs.total[sample_index]
        if sample_total < proposal_cost.total[0]:
            current = candidates[sample_index].clone()
            current_total = sample_total
        else:
            current = proposal
            current_total = proposal_cost.total[0]
        if current_total < best_total:
            best = current.clone()
            best_total = current_total.clone()

    improved = bool(best_total < raw_cost.total[0])
    selected = best if improved else raw_swing
    optimized_cost = terrain_conformal_swing_cost(
        selected[None],
        raw_swing,
        sample_surface=sample_surface,
        toe_offset_xy=toe_swing,
        heel_offset_xy=heel_swing,
        config=config,
    )
    output = raw_path.clone()
    output[start:stop] = selected
    output[~swing_mask] = raw_path[~swing_mask]
    output[start] = raw_path[start]
    output[stop - 1] = raw_path[stop - 1]
    return TerrainConformalSwingResult(
        path=output,
        raw_cost=raw_cost,
        optimized_cost=optimized_cost,
        improved=improved,
    )
