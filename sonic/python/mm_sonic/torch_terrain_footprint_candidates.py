"""Full-sole terrain candidates around nominal constant-heading contacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch

from .joints import ContractError
from .torch_heading_footprint_path import (
    NominalFootprint,
    NominalFootprintPath,
    heading_basis,
    heading_local_to_scene,
)


def _owned_vector(
    value: object,
    *,
    label: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (2,)
        or not value.dtype.is_floating_point
        or (device is not None and value.device != device)
        or (dtype is not None and value.dtype != dtype)
        or not torch.isfinite(value).all()
    ):
        raise ContractError(f"terrain footprint {label} is invalid")
    return value.detach().clone()


@dataclass(frozen=True)
class TerrainFootprintCandidate:
    step_index: int
    foot: int
    center_scene_xy: torch.Tensor
    center_heading_xy: torch.Tensor
    offset_heading_xy: torch.Tensor
    yaw_scene_rad: float
    surface_height_m: float
    placement_cost: float

    def __post_init__(self) -> None:
        if (
            type(self.step_index) is not int
            or self.step_index < 0
            or self.foot not in (0, 1)
            or any(
                not math.isfinite(float(value))
                for value in (
                    self.yaw_scene_rad,
                    self.surface_height_m,
                    self.placement_cost,
                )
            )
            or self.placement_cost < 0.0
        ):
            raise ContractError("terrain footprint candidate is invalid")
        scene = _owned_vector(
            self.center_scene_xy, label="scene center"
        )
        local = _owned_vector(
            self.center_heading_xy,
            label="heading center",
            device=scene.device,
            dtype=scene.dtype,
        )
        offset = _owned_vector(
            self.offset_heading_xy,
            label="heading offset",
            device=scene.device,
            dtype=scene.dtype,
        )
        object.__setattr__(self, "center_scene_xy", scene)
        object.__setattr__(self, "center_heading_xy", local)
        object.__setattr__(self, "offset_heading_xy", offset)
        object.__setattr__(
            self, "yaw_scene_rad", float(self.yaw_scene_rad)
        )
        object.__setattr__(
            self, "surface_height_m", float(self.surface_height_m)
        )
        object.__setattr__(
            self, "placement_cost", float(self.placement_cost)
        )


@dataclass(frozen=True)
class TerrainFootprintLayer:
    step_index: int
    nominal: NominalFootprint
    candidates: tuple[TerrainFootprintCandidate, ...]

    def __post_init__(self) -> None:
        if (
            type(self.step_index) is not int
            or self.step_index < 0
            or not isinstance(self.nominal, NominalFootprint)
            or self.nominal.step_index != self.step_index
            or not isinstance(self.candidates, tuple)
            or any(
                not isinstance(item, TerrainFootprintCandidate)
                or item.step_index != self.step_index
                or item.foot != self.nominal.foot
                for item in self.candidates
            )
        ):
            raise ContractError("terrain footprint layer is invalid")


class FootprintCandidateFailure(ContractError):
    def __init__(
        self,
        *,
        code: str,
        step_index: int,
        attempted_offsets: int,
        reasons: tuple[str, ...],
    ) -> None:
        if (
            code != "no_terrain_footprint"
            or type(step_index) is not int
            or step_index < 0
            or type(attempted_offsets) is not int
            or attempted_offsets < 1
            or not isinstance(reasons, tuple)
            or not reasons
            or any(not isinstance(item, str) or not item for item in reasons)
        ):
            raise ContractError("terrain footprint failure is invalid")
        self.code = code
        self.step_index = step_index
        self.attempted_offsets = attempted_offsets
        self.reasons = reasons
        super().__init__(
            f"{code} at step {step_index}: {'; '.join(reasons)}"
        )


def _samples(
    values: Sequence[float], *, positive: bool, label: str
) -> tuple[float, ...]:
    try:
        output = tuple(float(item) for item in values)
    except (TypeError, ValueError) as error:
        raise ContractError(
            f"terrain footprint {label} offsets are invalid"
        ) from error
    if (
        not output
        or any(not math.isfinite(item) for item in output)
        or (positive and any(item <= 0.0 for item in output))
    ):
        raise ContractError(
            f"terrain footprint {label} offsets are invalid"
        )
    return tuple(sorted(set(output)))


def _sole_templates(
    values: object,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(values, tuple) or len(values) != 2:
        raise ContractError("terrain footprint sole templates are invalid")
    output = []
    for value in values:
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.shape[1] != 2
            or len(value) < 3
            or value.device != device
            or value.dtype != dtype
            or not torch.isfinite(value).all()
        ):
            raise ContractError(
                "terrain footprint sole templates are invalid"
            )
        output.append(value.detach().clone())
    return output[0], output[1]


def terrain_footprint_layers(
    *,
    path: NominalFootprintPath,
    sole_offsets_by_foot: object,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    forward_offsets_m: Sequence[float],
    lateral_offsets_m: Sequence[float],
    maximum_surface_variation_m: float,
    edge_safety_margin_m: float,
) -> tuple[TerrainFootprintLayer, ...]:
    """Search bounded XY neighborhoods for complete supported sole prints."""

    if (
        not isinstance(path, NominalFootprintPath)
        or not callable(sample_surface)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in (
                maximum_surface_variation_m,
                edge_safety_margin_m,
            )
        )
    ):
        raise ContractError("terrain footprint search input is invalid")
    forward = _samples(
        forward_offsets_m, positive=False, label="forward"
    )
    lateral = _samples(
        lateral_offsets_m, positive=False, label="lateral"
    )
    device = path.origin_scene_xy.device
    dtype = path.origin_scene_xy.dtype
    soles = _sole_templates(
        sole_offsets_by_foot, device=device, dtype=dtype
    )
    offset_pairs = tuple(
        (forward_value, lateral_value)
        for forward_value in forward
        for lateral_value in lateral
    )
    offset_tensor = torch.tensor(
        offset_pairs, dtype=dtype, device=device
    )
    basis = heading_basis(path.heading_scene_xy)
    margin = float(edge_safety_margin_m)
    margin_offsets = torch.tensor(
        (
            (0.0, 0.0),
            (margin, 0.0),
            (-margin, 0.0),
            (0.0, margin),
            (0.0, -margin),
        ),
        dtype=dtype,
        device=device,
    )
    layers = []
    for nominal in path.footprints:
        centers_heading = (
            nominal.center_heading_xy[None, :] + offset_tensor
        )
        centers_scene = heading_local_to_scene(
            centers_heading,
            path.origin_scene_xy,
            path.heading_scene_xy,
        )
        sole_local = (
            soles[nominal.foot][:, None, :]
            + margin_offsets[None, :, :]
        ).reshape(-1, 2)
        oriented_sole = sole_local @ basis.T
        sample_points = (
            centers_scene[:, None, :] + oriented_sole[None, :, :]
        )
        heights = sample_surface(sample_points)
        if (
            not isinstance(heights, torch.Tensor)
            or heights.shape != sample_points.shape[:-1]
            or heights.device != device
            or heights.dtype != dtype
            or not torch.isfinite(heights).all()
        ):
            raise ContractError(
                "terrain footprint surface sampler returned invalid heights"
            )
        variation = heights.amax(dim=1) - heights.amin(dim=1)
        valid = variation <= float(maximum_surface_variation_m)
        candidates = []
        for candidate_index in torch.nonzero(
            valid, as_tuple=False
        ).flatten().tolist():
            offset = offset_tensor[candidate_index]
            forward_value = float(offset[0].item())
            lateral_value = float(offset[1].item())
            candidates.append(
                TerrainFootprintCandidate(
                    step_index=nominal.step_index,
                    foot=nominal.foot,
                    center_scene_xy=centers_scene[candidate_index],
                    center_heading_xy=centers_heading[candidate_index],
                    offset_heading_xy=offset,
                    yaw_scene_rad=nominal.yaw_scene_rad,
                    surface_height_m=float(
                        heights[candidate_index].median().item()
                    ),
                    placement_cost=(
                        forward_value * forward_value
                        + lateral_value * lateral_value
                    ),
                )
            )
        candidates.sort(
            key=lambda item: (
                item.placement_cost,
                abs(float(item.offset_heading_xy[0].item())),
                abs(float(item.offset_heading_xy[1].item())),
                float(item.offset_heading_xy[0].item()),
                float(item.offset_heading_xy[1].item()),
            )
        )
        if not candidates:
            raise FootprintCandidateFailure(
                code="no_terrain_footprint",
                step_index=nominal.step_index,
                attempted_offsets=len(offset_pairs),
                reasons=(
                    "no complete sole footprint has one supporting surface",
                ),
            )
        layers.append(
            TerrainFootprintLayer(
                step_index=nominal.step_index,
                nominal=nominal,
                candidates=tuple(candidates),
            )
        )
    return tuple(layers)
