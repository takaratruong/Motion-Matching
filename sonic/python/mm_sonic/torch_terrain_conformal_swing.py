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
        or tuple(toe_offset_xy.shape) != (2,)
        or toe_offset_xy.dtype != paths.dtype
        or toe_offset_xy.device != paths.device
        or not torch.isfinite(toe_offset_xy).all()
        or not isinstance(heel_offset_xy, torch.Tensor)
        or tuple(heel_offset_xy.shape) != (2,)
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
    axis = toe_offset_xy - heel_offset_xy
    norm = torch.linalg.vector_norm(axis)
    if float(norm.item()) <= 1.0e-6:
        raise ContractError("terrain-conformal swing foot offsets are degenerate")
    lateral = torch.stack((-axis[1], axis[0])) / norm
    offsets = torch.stack(
        (
            torch.zeros_like(toe_offset_xy),
            toe_offset_xy,
            heel_offset_xy,
            toe_offset_xy + lateral * float(config.edge_probe_m),
            heel_offset_xy - lateral * float(config.edge_probe_m),
        )
    )
    sample_xy = paths[..., None, :2] + offsets[None, None, :, :]
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
