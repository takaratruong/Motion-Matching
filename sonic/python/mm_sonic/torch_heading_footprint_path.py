"""Constant-heading nominal footprint paths in scene-local coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .joints import ContractError


def _owned_float_tensor(
    value: object,
    *,
    shape: tuple[int, ...],
    label: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or not value.dtype.is_floating_point
        or (device is not None and value.device != device)
        or not torch.isfinite(value).all()
    ):
        raise ContractError(f"heading footprint {label} is invalid")
    return value.detach().clone()


@dataclass(frozen=True)
class ConstantHeadingRequest:
    start_foot_scene_xy: torch.Tensor
    start_support: torch.Tensor
    heading_scene_xy: torch.Tensor
    distance_m: float
    speed_mps: float
    stride_m: float
    step_width_m: float
    frames_per_second: float

    def __post_init__(self) -> None:
        feet = _owned_float_tensor(
            self.start_foot_scene_xy,
            shape=(2, 2),
            label="start feet",
        )
        support = self.start_support
        if (
            not isinstance(support, torch.Tensor)
            or tuple(support.shape) != (2,)
            or support.dtype != torch.bool
            or support.device != feet.device
            or not bool(support.any().item())
        ):
            raise ContractError("heading footprint start support is invalid")
        heading = _owned_float_tensor(
            self.heading_scene_xy,
            shape=(2,),
            label="heading",
            device=feet.device,
        )
        if float(torch.linalg.vector_norm(heading).item()) <= 1.0e-6:
            raise ContractError("heading footprint heading is zero")
        scalars = (
            self.distance_m,
            self.speed_mps,
            self.stride_m,
            self.step_width_m,
            self.frames_per_second,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in scalars
        ):
            raise ContractError("heading footprint request scalar is invalid")
        object.__setattr__(self, "start_foot_scene_xy", feet)
        object.__setattr__(self, "start_support", support.detach().clone())
        object.__setattr__(
            self,
            "heading_scene_xy",
            heading / torch.linalg.vector_norm(heading),
        )
        for name in (
            "distance_m",
            "speed_mps",
            "stride_m",
            "step_width_m",
            "frames_per_second",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))


@dataclass(frozen=True)
class NominalFootprint:
    step_index: int
    foot: int
    center_scene_xy: torch.Tensor
    center_heading_xy: torch.Tensor
    yaw_scene_rad: float
    contact_frame: int

    def __post_init__(self) -> None:
        if (
            type(self.step_index) is not int
            or self.step_index < 0
            or self.foot not in (0, 1)
            or type(self.contact_frame) is not int
            or self.contact_frame < 1
            or not math.isfinite(float(self.yaw_scene_rad))
        ):
            raise ContractError("nominal footprint metadata is invalid")
        scene = _owned_float_tensor(
            self.center_scene_xy,
            shape=(2,),
            label="scene center",
        )
        local = _owned_float_tensor(
            self.center_heading_xy,
            shape=(2,),
            label="heading center",
            device=scene.device,
        )
        object.__setattr__(self, "center_scene_xy", scene)
        object.__setattr__(self, "center_heading_xy", local)
        object.__setattr__(
            self, "yaw_scene_rad", float(self.yaw_scene_rad)
        )


@dataclass(frozen=True)
class NominalFootprintPath:
    origin_scene_xy: torch.Tensor
    heading_scene_xy: torch.Tensor
    footprints: tuple[NominalFootprint, ...]
    progress_m: float

    def __post_init__(self) -> None:
        origin = _owned_float_tensor(
            self.origin_scene_xy,
            shape=(2,),
            label="path origin",
        )
        heading = _owned_float_tensor(
            self.heading_scene_xy,
            shape=(2,),
            label="path heading",
            device=origin.device,
        )
        if (
            abs(float(torch.linalg.vector_norm(heading).item()) - 1.0)
            > 1.0e-5
            or not isinstance(self.footprints, tuple)
            or any(
                not isinstance(item, NominalFootprint)
                or item.center_scene_xy.device != origin.device
                for item in self.footprints
            )
            or not math.isfinite(float(self.progress_m))
            or self.progress_m < 0.0
        ):
            raise ContractError("nominal footprint path is invalid")
        object.__setattr__(self, "origin_scene_xy", origin)
        object.__setattr__(self, "heading_scene_xy", heading)
        object.__setattr__(self, "progress_m", float(self.progress_m))


def heading_basis(heading_scene_xy: torch.Tensor) -> torch.Tensor:
    """Return columns for forward and left axes in the scene frame."""

    heading = _owned_float_tensor(
        heading_scene_xy,
        shape=(2,),
        label="heading",
    )
    norm = torch.linalg.vector_norm(heading)
    if float(norm.item()) <= 1.0e-6:
        raise ContractError("heading footprint heading is zero")
    forward = heading / norm
    lateral = torch.stack((-forward[1], forward[0]))
    return torch.stack((forward, lateral), dim=1)


def heading_local_to_scene(
    local_xy: torch.Tensor,
    origin_scene_xy: torch.Tensor,
    heading_scene_xy: torch.Tensor,
) -> torch.Tensor:
    """Map heading-local row vectors into the scene XY frame."""

    if (
        not isinstance(local_xy, torch.Tensor)
        or local_xy.ndim < 1
        or local_xy.shape[-1] != 2
        or not local_xy.dtype.is_floating_point
        or not torch.isfinite(local_xy).all()
    ):
        raise ContractError("heading footprint local points are invalid")
    origin = _owned_float_tensor(
        origin_scene_xy,
        shape=(2,),
        label="origin",
        device=local_xy.device,
    )
    heading = _owned_float_tensor(
        heading_scene_xy,
        shape=(2,),
        label="heading",
        device=local_xy.device,
    )
    if origin.dtype != local_xy.dtype or heading.dtype != local_xy.dtype:
        raise ContractError("heading footprint point dtypes differ")
    return origin + local_xy @ heading_basis(heading).T


def scene_to_heading_local(
    scene_xy: torch.Tensor,
    origin_scene_xy: torch.Tensor,
    heading_scene_xy: torch.Tensor,
) -> torch.Tensor:
    """Map scene row vectors into the commanded heading frame."""

    if (
        not isinstance(scene_xy, torch.Tensor)
        or scene_xy.ndim < 1
        or scene_xy.shape[-1] != 2
        or not scene_xy.dtype.is_floating_point
        or not torch.isfinite(scene_xy).all()
    ):
        raise ContractError("heading footprint scene points are invalid")
    origin = _owned_float_tensor(
        origin_scene_xy,
        shape=(2,),
        label="origin",
        device=scene_xy.device,
    )
    heading = _owned_float_tensor(
        heading_scene_xy,
        shape=(2,),
        label="heading",
        device=scene_xy.device,
    )
    if origin.dtype != scene_xy.dtype or heading.dtype != scene_xy.dtype:
        raise ContractError("heading footprint point dtypes differ")
    return (scene_xy - origin) @ heading_basis(heading)


def nominal_footprint_path(
    request: ConstantHeadingRequest,
) -> NominalFootprintPath:
    """Generate alternating contacts along one arbitrary constant heading."""

    if not isinstance(request, ConstantHeadingRequest):
        raise ContractError("heading footprint request is invalid")
    origin = request.start_foot_scene_xy.mean(dim=0)
    start_local = scene_to_heading_local(
        request.start_foot_scene_xy,
        origin,
        request.heading_scene_xy,
    )
    unsupported = torch.nonzero(
        ~request.start_support, as_tuple=False
    ).flatten()
    if len(unsupported) == 1:
        first_foot = int(unsupported[0].item())
    else:
        first_foot = min(
            range(2),
            key=lambda foot: (float(start_local[foot, 0].item()), foot),
        )
    step_count = int(
        math.floor(request.distance_m / request.stride_m + 1.0e-9)
    )
    yaw = math.atan2(
        float(request.heading_scene_xy[1].item()),
        float(request.heading_scene_xy[0].item()),
    )
    footprints = []
    for step_index in range(step_count):
        foot = (first_foot + step_index) % 2
        progress = (step_index + 1) * request.stride_m
        lateral = (
            0.5 * request.step_width_m
            if foot == 0
            else -0.5 * request.step_width_m
        )
        local = torch.tensor(
            (progress, lateral),
            dtype=origin.dtype,
            device=origin.device,
        )
        scene = heading_local_to_scene(
            local, origin, request.heading_scene_xy
        )
        footprints.append(
            NominalFootprint(
                step_index=step_index,
                foot=foot,
                center_scene_xy=scene,
                center_heading_xy=local,
                yaw_scene_rad=yaw,
                contact_frame=max(
                    1,
                    round(
                        progress
                        / request.speed_mps
                        * request.frames_per_second
                    ),
                ),
            )
        )
    progress = (
        footprints[-1].center_heading_xy[0].item()
        if footprints
        else 0.0
    )
    return NominalFootprintPath(
        origin_scene_xy=origin,
        heading_scene_xy=request.heading_scene_xy,
        footprints=tuple(footprints),
        progress_m=float(progress),
    )
