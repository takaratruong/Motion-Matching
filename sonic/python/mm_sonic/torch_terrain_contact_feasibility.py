"""Every-frame contact feasibility for rigidly placed terrain skills."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from .joints import ContractError
from .torch_contact_segments import ANKLE_ORIGIN_SOLE_M
from .torch_motion_features import TorchMotionDatabase
from .torch_terrain_features import TerrainDataset, TerrainFeatureExtension
from .torch_terrain_skills import TerrainSkill


def _positive_finite(value: object, *, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ContractError(f"terrain contact feasibility {name} is invalid")
    return float(value)


@dataclass(frozen=True)
class TerrainContactFeasibilityConfig:
    """Hard bounds used before a terrain-skill candidate may play."""

    sample_stride: int = 1
    stance_height_tolerance_m: float = 0.05
    landing_height_tolerance_m: float = 0.03
    edge_margin_m: float = 0.04
    maximum_edge_height_range_m: float = 0.025
    minimum_swing_clearance_m: float = -0.005
    minimum_sole_clearance_m: float = -0.025
    maximum_height_deformation_m: float = 0.06

    def __post_init__(self) -> None:
        if type(self.sample_stride) is not int or self.sample_stride != 1:
            raise ContractError(
                "terrain contact feasibility sample stride must equal one"
            )
        _positive_finite(
            self.stance_height_tolerance_m, name="stance tolerance"
        )
        _positive_finite(
            self.landing_height_tolerance_m, name="landing tolerance"
        )
        _positive_finite(self.edge_margin_m, name="edge margin")
        _positive_finite(
            self.maximum_edge_height_range_m, name="edge height range"
        )
        for name in ("minimum_swing_clearance_m", "minimum_sole_clearance_m"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ContractError(
                    "terrain contact feasibility clearance is invalid"
                )
        _positive_finite(
            self.maximum_height_deformation_m, name="height deformation"
        )


@dataclass(frozen=True)
class TerrainContactFeasibilityResult:
    """Measured hard-gate outcome for one placed contact trace."""

    accepted: bool
    reason: str | None
    maximum_stance_error_m: float
    landing_error_m: float
    minimum_swing_clearance_m: float
    minimum_sole_clearance_m: float
    maximum_footprint_height_range_m: float
    maximum_height_deformation_m: float

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise ContractError("terrain contact feasibility result is invalid")
        reason_valid = self.reason is None or (
            isinstance(self.reason, str) and bool(self.reason)
        )
        if not reason_valid or self.accepted != (self.reason is None):
            raise ContractError("terrain contact feasibility result is invalid")
        metrics = (
            self.maximum_stance_error_m,
            self.landing_error_m,
            self.minimum_swing_clearance_m,
            self.minimum_sole_clearance_m,
            self.maximum_footprint_height_range_m,
            self.maximum_height_deformation_m,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in metrics
        ):
            raise ContractError("terrain contact feasibility result is invalid")
        if any(
            float(value) < 0.0
            for value in (
                self.maximum_stance_error_m,
                self.landing_error_m,
                self.maximum_footprint_height_range_m,
                self.maximum_height_deformation_m,
            )
        ):
            raise ContractError("terrain contact feasibility result is invalid")


def _result(
    reason: str | None,
    *,
    stance: float = 0.0,
    landing: float = 0.0,
    swing: float = 0.0,
    sole: float = 0.0,
    footprint: float = 0.0,
    deformation: float = 0.0,
) -> TerrainContactFeasibilityResult:
    return TerrainContactFeasibilityResult(
        accepted=reason is None,
        reason=reason,
        maximum_stance_error_m=float(stance),
        landing_error_m=float(landing),
        minimum_swing_clearance_m=float(swing),
        minimum_sole_clearance_m=float(sole),
        maximum_footprint_height_range_m=float(footprint),
        maximum_height_deformation_m=float(deformation),
    )


def _sample_surface(
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    points_xy: torch.Tensor,
) -> torch.Tensor:
    values = sample_surface(points_xy)
    if (
        not isinstance(values, torch.Tensor)
        or values.shape != points_xy.shape[:-1]
        or values.device != points_xy.device
        or values.dtype != points_xy.dtype
        or not torch.isfinite(values).all()
    ):
        raise ContractError(
            "terrain contact feasibility sampler returned invalid heights"
        )
    return values


def validate_placed_contact_trace(
    *,
    foot_position_world: torch.Tensor,
    support_mask: torch.Tensor,
    source_surface_height_m: torch.Tensor,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    config: TerrainContactFeasibilityConfig,
    align_initial_support: bool = False,
    landing_footprint_position_world: torch.Tensor | None = None,
) -> TerrainContactFeasibilityResult:
    """Validate one already-placed two-foot trace at every source frame."""

    if (
        not isinstance(foot_position_world, torch.Tensor)
        or not foot_position_world.dtype.is_floating_point
        or foot_position_world.ndim != 3
        or tuple(foot_position_world.shape[1:]) != (2, 3)
        or foot_position_world.shape[0] < 1
        or not torch.isfinite(foot_position_world).all()
        or not isinstance(support_mask, torch.Tensor)
        or support_mask.dtype != torch.bool
        or tuple(support_mask.shape) != tuple(foot_position_world.shape[:2])
        or support_mask.device != foot_position_world.device
        or not isinstance(source_surface_height_m, torch.Tensor)
        or tuple(source_surface_height_m.shape) != tuple(support_mask.shape)
        or source_surface_height_m.device != foot_position_world.device
        or source_surface_height_m.dtype != foot_position_world.dtype
        or not torch.isfinite(source_surface_height_m).all()
        or not callable(sample_surface)
        or not isinstance(config, TerrainContactFeasibilityConfig)
        or type(align_initial_support) is not bool
        or (
            landing_footprint_position_world is not None
            and (
                not isinstance(landing_footprint_position_world, torch.Tensor)
                or landing_footprint_position_world.ndim != 4
                or tuple(landing_footprint_position_world.shape[:2])
                != tuple(foot_position_world.shape[:2])
                or landing_footprint_position_world.shape[2] < 3
                or landing_footprint_position_world.shape[3] != 3
                or landing_footprint_position_world.dtype
                != foot_position_world.dtype
                or landing_footprint_position_world.device
                != foot_position_world.device
                or not torch.isfinite(
                    landing_footprint_position_world
                ).all()
            )
        )
    ):
        raise ContractError("terrain contact feasibility trace is invalid")

    landing_mask = torch.zeros_like(support_mask)
    if support_mask.shape[0] > 1:
        landing_mask[1:] = ~support_mask[:-1] & support_mask[1:]
    landing_indices = torch.nonzero(landing_mask, as_tuple=False)
    try:
        surface = _sample_surface(
            sample_surface, foot_position_world[..., :2]
        )
        footprint_range = 0.0
        sole_clearance = 0.0
        if landing_footprint_position_world is not None:
            sole_surface = _sample_surface(
                sample_surface,
                landing_footprint_position_world[..., :2],
            )
            sole_clearance = float(
                (
                    landing_footprint_position_world[..., 2]
                    - sole_surface
                ).min().item()
            )
        if landing_indices.numel():
            if landing_footprint_position_world is None:
                margin = float(config.edge_margin_m)
                offsets = torch.tensor(
                    (
                        (0.0, 0.0),
                        (margin, 0.0),
                        (-margin, 0.0),
                        (0.0, margin),
                        (0.0, -margin),
                    ),
                    dtype=foot_position_world.dtype,
                    device=foot_position_world.device,
                )
                landing_xy = foot_position_world[
                    landing_indices[:, 0], landing_indices[:, 1], :2
                ]
                footprint_xy = landing_xy[:, None, :] + offsets[None, :, :]
            else:
                footprint_xy = landing_footprint_position_world[
                    landing_indices[:, 0], landing_indices[:, 1], :, :2
                ]
            landing_footprints = _sample_surface(sample_surface, footprint_xy)
            footprint_range = float(
                (
                    landing_footprints.max(dim=1).values
                    - landing_footprints.min(dim=1).values
                ).max().item()
            )
    except ContractError:
        return _result("terrain-domain")

    vertical_offset = torch.zeros(
        (),
        dtype=foot_position_world.dtype,
        device=foot_position_world.device,
    )
    if align_initial_support:
        anchor_mask = support_mask[0]
        if not bool(anchor_mask.any().item()):
            raise ContractError(
                "terrain contact feasibility initial support is unavailable"
            )
        source_sole_z = (
            foot_position_world[0, :, 2] - float(ANKLE_ORIGIN_SOLE_M)
        )
        vertical_offset = (
            surface[0, anchor_mask] - source_sole_z[anchor_mask]
        ).mean()
    clearance = (
        foot_position_world[..., 2]
        + vertical_offset
        - surface
        - float(ANKLE_ORIGIN_SOLE_M)
    )
    stance_error = float(
        torch.abs(clearance[support_mask]).max().item()
        if bool(support_mask.any().item())
        else 0.0
    )
    landing_error = float(
        torch.abs(clearance[landing_mask]).max().item()
        if bool(landing_mask.any().item())
        else 0.0
    )
    swing_clearance = float(
        clearance[~support_mask].min().item()
        if bool((~support_mask).any().item())
        else 0.0
    )
    deformation = 0.0
    if landing_indices.numel():
        frames = landing_indices[:, 0]
        feet = landing_indices[:, 1]
        query_delta = surface[frames, feet] - surface[0, feet]
        source_delta = (
            source_surface_height_m[frames, feet]
            - source_surface_height_m[0, feet]
        )
        deformation = float(torch.abs(query_delta - source_delta).max().item())

    metrics = dict(
        stance=stance_error,
        landing=landing_error,
        swing=swing_clearance,
        sole=sole_clearance,
        footprint=footprint_range,
        deformation=deformation,
    )
    if stance_error > float(config.stance_height_tolerance_m):
        return _result("stance-height", **metrics)
    if landing_error > float(config.landing_height_tolerance_m):
        return _result("landing-height", **metrics)
    if footprint_range > float(config.maximum_edge_height_range_m):
        return _result("landing-edge-margin", **metrics)
    if sole_clearance < float(config.minimum_sole_clearance_m):
        return _result("sole-penetration", **metrics)
    if swing_clearance < float(config.minimum_swing_clearance_m):
        return _result("swing-penetration", **metrics)
    if deformation > float(config.maximum_height_deformation_m):
        return _result("height-deformation", **metrics)
    return _result(None, **metrics)


def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _rotate_xy(values: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    x, y = values[..., 0], values[..., 1]
    return torch.stack(
        (cosine * x - sine * y, sine * x + cosine * y), dim=-1
    )


def validate_terrain_skill_contact_trace(
    *,
    skill: TerrainSkill,
    canonical_entry_row: int,
    endpoint_frame_exclusive: int,
    dataset: TerrainDataset,
    database: TorchMotionDatabase,
    query_terrain: TerrainFeatureExtension,
    current_root_position_world: torch.Tensor,
    current_root_orientation_world_wxyz: torch.Tensor,
    config: TerrainContactFeasibilityConfig,
) -> TerrainContactFeasibilityResult:
    """Rigidly place a candidate horizon and validate every contact frame."""

    if (
        not isinstance(skill, TerrainSkill)
        or not isinstance(dataset, TerrainDataset)
        or not isinstance(database, TorchMotionDatabase)
        or not isinstance(query_terrain, TerrainFeatureExtension)
        or not isinstance(config, TerrainContactFeasibilityConfig)
    ):
        raise ContractError("terrain skill contact feasibility inputs are invalid")
    if type(canonical_entry_row) is not int or not (
        0 <= canonical_entry_row < database.feature_shape[0]
    ):
        raise ContractError("terrain skill contact feasibility entry row is invalid")
    if int(database._search_clip_index[canonical_entry_row].item()) != skill.clip_index:
        raise ContractError("terrain skill contact feasibility row does not own skill")
    entry_frame = int(database._search_frame_index[canonical_entry_row].item())
    if (
        type(endpoint_frame_exclusive) is not int
        or not entry_frame < endpoint_frame_exclusive <= skill.interval.playback_stop
        or not bool(skill.support_mask[endpoint_frame_exclusive - 1].all().item())
    ):
        raise ContractError("terrain skill contact feasibility endpoint is invalid")
    if (
        not isinstance(current_root_position_world, torch.Tensor)
        or tuple(current_root_position_world.shape) != (3,)
        or not current_root_position_world.dtype.is_floating_point
        or current_root_position_world.device != database.device
        or not torch.isfinite(current_root_position_world).all()
        or not isinstance(current_root_orientation_world_wxyz, torch.Tensor)
        or tuple(current_root_orientation_world_wxyz.shape) != (4,)
        or current_root_orientation_world_wxyz.dtype
        != current_root_position_world.dtype
        or current_root_orientation_world_wxyz.device != database.device
        or not torch.isfinite(current_root_orientation_world_wxyz).all()
    ):
        raise ContractError("terrain skill contact feasibility root is invalid")

    clip = dataset.folder.clips[skill.clip_index]
    layout = dataset.folder.layout
    root_index = layout.root_body_index
    feet_indices = (layout.left_foot_body_index, layout.right_foot_body_index)
    source_root = torch.as_tensor(
        clip.body_position_world[entry_frame, root_index],
        dtype=current_root_position_world.dtype,
        device=database.device,
    )
    source_quaternion = torch.as_tensor(
        clip.body_quaternion_world_wxyz[entry_frame, root_index],
        dtype=current_root_position_world.dtype,
        device=database.device,
    )
    yaw_offset = _yaw_from_wxyz(
        current_root_orientation_world_wxyz
    ) - _yaw_from_wxyz(source_quaternion)
    source_feet = torch.as_tensor(
        clip.body_position_world[
            entry_frame:endpoint_frame_exclusive, feet_indices, :
        ],
        dtype=current_root_position_world.dtype,
        device=database.device,
    )
    placed_feet = torch.empty_like(source_feet)
    placed_feet[..., :2] = current_root_position_world[:2] + _rotate_xy(
        source_feet[..., :2] - source_root[:2], yaw_offset
    )
    placed_feet[..., 2] = current_root_position_world[2] + (
        source_feet[..., 2] - source_root[2]
    )

    def sample_query(points_xy: torch.Tensor) -> torch.Tensor:
        return query_terrain.query_grid.sample_xy(
            query_terrain.alignment.matcher_to_scene_xy(points_xy)
        )

    return validate_placed_contact_trace(
        foot_position_world=placed_feet,
        support_mask=skill.support_mask[
            entry_frame:endpoint_frame_exclusive
        ],
        source_surface_height_m=skill.foot_surface_height_m[
            entry_frame:endpoint_frame_exclusive
        ].to(dtype=placed_feet.dtype),
        sample_surface=sample_query,
        config=config,
        align_initial_support=True,
    )
