"""Transactional fixed-horizon terrain-skill rollout and qualification adapter."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .joints import ContractError
from .torch_motion_data import MotionFolder
from .torch_motion_features import (
    GeneratedFeatureState,
    TorchMotionDatabase,
    extract_query_features,
)
from .torch_motion_matcher import MatcherConfig, TorchMotionMatcher
from .torch_terrain_omni_routes import OmniRoute
from .torch_terrain_contact_feasibility import (
    TerrainContactFeasibilityConfig,
    TerrainContactFeasibilityResult,
    validate_placed_contact_trace,
)
from .torch_terrain_emitted_contact_preview import (
    preview_emitted_contact_trace,
)
from .torch_terrain_skill_composer import extend_skill_state, start_skill
from .torch_terrain_skill_horizon_search import (
    HorizonCost,
    HorizonSearchFailure,
    HorizonSearchConfig,
    HorizonTargets,
    TerrainSkillHorizonResult,
    predict_horizon_targets,
    select_horizon_candidate,
)
from .torch_terrain_skill_horizons import (
    TerrainSkillHorizonInventory,
    build_horizon_inventory,
    next_sequential_horizon_endpoint,
    remaining_stall_profile,
)
from .torch_terrain_skill_rollout import (
    TerrainSkillMatcher,
    _rotate_xy,
    _yaw_from_wxyz,
    terrain_skill_compatible,
)
from .torch_terrain_skills import TerrainSkillInventory, build_terrain_skill_inventory


@dataclass(frozen=True)
class HorizonChunkEvent:
    entry_row: int
    skill_index: int
    target_frames: int
    endpoint_frame_exclusive: int
    cost: HorizonCost
    rejected_by_reason: Mapping[str, int]
    release_reason: str
    selection_mode: str = "immediate"


@dataclass(frozen=True)
class TerrainContinuationEvent:
    skill_index: int
    start_frame: int
    endpoint_frame_exclusive: int
    command: tuple[tuple[float, float], float]
    validation: TerrainContactFeasibilityResult


def terrain_height_targets(
    targets: HorizonTargets,
    *,
    current_root_position_world: torch.Tensor,
    current_root_yaw: torch.Tensor,
    query_terrain: Any,
) -> HorizonTargets:
    """Sample desired vertical change along the commanded root path."""

    if not isinstance(targets, HorizonTargets):
        raise ContractError("terrain height prediction requires HorizonTargets")
    if (
        not isinstance(current_root_position_world, torch.Tensor)
        or tuple(current_root_position_world.shape) != (3,)
        or current_root_position_world.device != targets.displacement_local_xy.device
        or current_root_position_world.dtype != targets.displacement_local_xy.dtype
        or not torch.isfinite(current_root_position_world).all()
        or not isinstance(current_root_yaw, torch.Tensor)
        or current_root_yaw.numel() != 1
        or current_root_yaw.device != current_root_position_world.device
        or current_root_yaw.dtype != current_root_position_world.dtype
    ):
        raise ContractError("terrain height prediction root state is invalid")
    try:
        world_displacement = _rotate_xy(
            targets.displacement_local_xy, current_root_yaw.reshape(())
        )
        world_xy = current_root_position_world[:2].unsqueeze(0) + world_displacement
        sample_xy = torch.cat(
            (current_root_position_world[:2].unsqueeze(0), world_xy), dim=0
        )
        height = query_terrain.query_grid.sample_xy(
            query_terrain.alignment.matcher_to_scene_xy(sample_xy)
        )
    except (AttributeError, TypeError) as error:
        raise ContractError("terrain height prediction query is invalid") from error
    if (
        not isinstance(height, torch.Tensor)
        or tuple(height.shape) != (targets.frames.shape[0] + 1,)
        or height.device != current_root_position_world.device
        or not torch.isfinite(height).all()
    ):
        raise ContractError("terrain height prediction samples are invalid")
    delta = (height[1:] - height[0]).to(targets.displacement_local_xy.dtype)
    return HorizonTargets(
        frames=targets.frames,
        displacement_local_xy=targets.displacement_local_xy,
        yaw_delta_rad=targets.yaw_delta_rad,
        root_height_delta_m=delta,
        surface_height_delta_m=delta[:, None].expand(-1, 2).clone(),
    )


class TerrainSkillHorizonMatcher(TerrainSkillMatcher):
    """Use local outcome ranking while preserving transactional frame playback."""

    def __init__(
        self,
        *,
        base_matcher: TorchMotionMatcher,
        skill_inventory: TerrainSkillInventory,
        horizon_inventory: TerrainSkillHorizonInventory,
        dataset: Any,
        query_terrain: Any,
        foot_kinematics: Any,
        config: MatcherConfig,
        search_config: HorizonSearchConfig = HorizonSearchConfig(),
        terrain_tolerance_m: float = 0.06,
        result_filter: Any | None = None,
        contact_phase_gate: bool = False,
        turning_clip_paths: frozenset[str] | None = None,
        maximum_translation_warp_m: float = 0.0,
        maximum_yaw_warp_rad: float = 0.0,
        minimum_endpoint_warp_yaw_rad: float = 0.0,
        maximum_endpoint_warp_yaw_rad: float = math.pi,
        maximum_endpoint_warp_terrain_delta_m: float = math.inf,
        minimum_endpoint_warp_velocity_heading_alignment: float = -1.0,
        continuous_skill_enabled: bool = False,
        contact_feasibility_enabled: bool = False,
        continuation_surface_tolerance_m: float = 0.08,
        contact_feasibility_config: TerrainContactFeasibilityConfig = (
            TerrainContactFeasibilityConfig()
        ),
        minimum_continuation_progress_m: float = 0.05,
        maximum_continuation_stall_frames: int = 5,
        minimum_continuation_velocity_alignment: float = 0.5,
        minimum_continuation_commanded_progress_ratio: float = 0.35,
        maximum_continuation_heading_regression_rad: float = 0.15,
        maximum_preferred_cost_increase: float = 1.0,
        maximum_preferred_outcome_cost_increase: float = 0.05,
        emitted_contact_preview_enabled: bool = False,
    ) -> None:
        self.base = base_matcher
        self.database = base_matcher.database
        self.inventory = skill_inventory
        self.horizon_inventory = horizon_inventory
        self.dataset = dataset
        self.query_terrain = query_terrain
        self.foot_kinematics = foot_kinematics
        self.config = config
        self.search_config = search_config
        self.terrain_tolerance_m = float(terrain_tolerance_m)
        if result_filter is not None and (
            not callable(getattr(result_filter, "reset", None))
            or not callable(getattr(result_filter, "apply", None))
        ):
            raise ContractError("terrain skill result filter is invalid")
        self.result_filter = result_filter
        if type(contact_phase_gate) is not bool:
            raise ContractError("contact phase gate must be boolean")
        self.contact_phase_gate = contact_phase_gate
        if turning_clip_paths is not None and (
            not isinstance(turning_clip_paths, frozenset)
            or not turning_clip_paths
            or any(
                not isinstance(path, str) or not path
                for path in turning_clip_paths
            )
        ):
            raise ContractError("turning clip paths must be a non-empty frozenset")
        self.turning_clip_paths = turning_clip_paths
        if (
            not math.isfinite(float(maximum_translation_warp_m))
            or float(maximum_translation_warp_m) < 0.0
            or not math.isfinite(float(maximum_yaw_warp_rad))
            or float(maximum_yaw_warp_rad) < 0.0
            or not math.isfinite(float(minimum_endpoint_warp_yaw_rad))
            or not math.isfinite(float(maximum_endpoint_warp_yaw_rad))
            or not 0.0 <= float(minimum_endpoint_warp_yaw_rad)
            <= float(maximum_endpoint_warp_yaw_rad)
            <= math.pi
            or math.isnan(float(maximum_endpoint_warp_terrain_delta_m))
            or float(maximum_endpoint_warp_terrain_delta_m) < 0.0
            or not math.isfinite(
                float(minimum_endpoint_warp_velocity_heading_alignment)
            )
            or not -1.0 <= float(
                minimum_endpoint_warp_velocity_heading_alignment
            ) <= 1.0
        ):
            raise ContractError("endpoint warp limits are invalid")
        self.maximum_translation_warp_m = float(maximum_translation_warp_m)
        self.maximum_yaw_warp_rad = float(maximum_yaw_warp_rad)
        self.minimum_endpoint_warp_yaw_rad = float(
            minimum_endpoint_warp_yaw_rad
        )
        self.maximum_endpoint_warp_yaw_rad = float(
            maximum_endpoint_warp_yaw_rad
        )
        self.maximum_endpoint_warp_terrain_delta_m = float(
            maximum_endpoint_warp_terrain_delta_m
        )
        self.minimum_endpoint_warp_velocity_heading_alignment = float(
            minimum_endpoint_warp_velocity_heading_alignment
        )
        if (
            type(continuous_skill_enabled) is not bool
            or type(contact_feasibility_enabled) is not bool
            or type(emitted_contact_preview_enabled) is not bool
        ):
            raise ContractError("continuous terrain skill flags must be boolean")
        if not isinstance(
            contact_feasibility_config, TerrainContactFeasibilityConfig
        ):
            raise ContractError("continuous terrain contact config is invalid")
        if (
            not math.isfinite(float(minimum_continuation_progress_m))
            or float(minimum_continuation_progress_m) <= 0.0
            or not math.isfinite(float(continuation_surface_tolerance_m))
            or not 0.0 < float(continuation_surface_tolerance_m) < 0.10
            or type(maximum_continuation_stall_frames) is not int
            or maximum_continuation_stall_frames < 0
            or not math.isfinite(
                float(minimum_continuation_velocity_alignment)
            )
            or not -1.0 <= float(
                minimum_continuation_velocity_alignment
            ) <= 1.0
            or not math.isfinite(
                float(minimum_continuation_commanded_progress_ratio)
            )
            or not 0.0 < float(
                minimum_continuation_commanded_progress_ratio
            ) <= 1.0
            or not math.isfinite(
                float(maximum_continuation_heading_regression_rad)
            )
            or float(maximum_continuation_heading_regression_rad) < 0.0
            or not math.isfinite(float(maximum_preferred_cost_increase))
            or float(maximum_preferred_cost_increase) < 0.0
            or not math.isfinite(
                float(maximum_preferred_outcome_cost_increase)
            )
            or float(maximum_preferred_outcome_cost_increase) < 0.0
        ):
            raise ContractError("continuous terrain skill limits are invalid")
        self.continuous_skill_enabled = continuous_skill_enabled
        self.contact_feasibility_enabled = contact_feasibility_enabled
        self.emitted_contact_preview_enabled = emitted_contact_preview_enabled
        self.contact_feasibility_config = contact_feasibility_config
        self.minimum_continuation_progress_m = float(
            minimum_continuation_progress_m
        )
        self.continuation_surface_tolerance_m = float(
            continuation_surface_tolerance_m
        )
        self.maximum_continuation_stall_frames = (
            maximum_continuation_stall_frames
        )
        self.minimum_continuation_velocity_alignment = float(
            minimum_continuation_velocity_alignment
        )
        self.minimum_continuation_commanded_progress_ratio = float(
            minimum_continuation_commanded_progress_ratio
        )
        self.maximum_continuation_heading_regression_rad = float(
            maximum_continuation_heading_regression_rad
        )
        self.maximum_preferred_cost_increase = float(
            maximum_preferred_cost_increase
        )
        self.maximum_preferred_outcome_cost_increase = float(
            maximum_preferred_outcome_cost_increase
        )
        self._clip_path_to_index = {
            clip.relative_path: index
            for index, clip in enumerate(dataset.folder.clips)
        }
        self._support_by_clip = {
            skill.clip_index: skill.support_mask for skill in skill_inventory.skills
        }
        self._pending = None
        self._skill_state = None
        self._last_result = None
        self._last_command = None
        self._replan_pending = False
        self._shaped_velocity = torch.zeros(2, device=self.database.device)
        self._shaped_heading = torch.zeros((), device=self.database.device)
        self._feet = None
        self._foot_velocity = torch.zeros((2, 3), device=self.database.device)
        self._root_velocity = torch.zeros(3, device=self.database.device)
        self._events: list[HorizonChunkEvent] = []
        self._continuation_events: list[TerrainContinuationEvent] = []
        self._prepared_continuation: TerrainContinuationEvent | None = None
        self._prepared_selection: TerrainSkillHorizonResult | None = None
        self._prepared_release_reason: str | None = None
        self._endpoint_warp_command = None
        self._endpoint_warp_command_enabled = False

    def _source_suffix_matches_command(
        self,
        *,
        clip: Any,
        start_frame: int,
        endpoint_frame_exclusive: int,
        yaw_offset: torch.Tensor,
        command: tuple[tuple[float, float], float],
    ) -> bool:
        """Reject source continuations that leave the held command manifold."""

        root_index = int(self.dataset.folder.layout.root_body_index)
        roots = torch.tensor(
            clip.body_position_world[
                start_frame - 1 : endpoint_frame_exclusive, root_index
            ],
            dtype=yaw_offset.dtype,
            device=yaw_offset.device,
        )
        velocity = torch.as_tensor(
            command[0], dtype=yaw_offset.dtype, device=yaw_offset.device
        )
        speed = torch.linalg.vector_norm(velocity)
        displacement_world = _rotate_xy(
            roots[-1, :2] - roots[0, :2], yaw_offset
        )
        progress = torch.linalg.vector_norm(displacement_world)
        if float(speed.item()) > 0.05 and float(progress.item()) > 1e-8:
            alignment = torch.dot(displacement_world, velocity) / (
                progress * speed
            )
            if float(alignment.item()) < (
                self.minimum_continuation_velocity_alignment
            ):
                return False
            commanded_progress = torch.dot(
                displacement_world, velocity / speed
            )
            expected_progress = speed * (
                (endpoint_frame_exclusive - start_frame) * self.config.dt
            )
            if float(commanded_progress.item()) < float(
                expected_progress.item()
            ) * self.minimum_continuation_commanded_progress_ratio:
                return False

        orientations = torch.tensor(
            clip.body_quaternion_world_wxyz[
                [start_frame - 1, endpoint_frame_exclusive - 1], root_index
            ],
            dtype=yaw_offset.dtype,
            device=yaw_offset.device,
        )
        placed_yaw = _yaw_from_wxyz(orientations) + yaw_offset
        desired_heading = torch.as_tensor(
            command[1], dtype=yaw_offset.dtype, device=yaw_offset.device
        )
        heading_error = torch.abs(
            torch.atan2(
                torch.sin(desired_heading - placed_yaw),
                torch.cos(desired_heading - placed_yaw),
            )
        )
        return float(heading_error[1].item()) <= (
            float(heading_error[0].item())
            + self.maximum_continuation_heading_regression_rad
        )

    @property
    def chunk_events(self) -> tuple[HorizonChunkEvent, ...]:
        return tuple(self._events)

    @property
    def continuation_events(self) -> tuple[TerrainContinuationEvent, ...]:
        return tuple(self._continuation_events)

    def reset(
        self, *, root_position_world_xy: tuple[float, float] = (0.0, 0.0)
    ):
        result = super().reset(root_position_world_xy=root_position_world_xy)
        self._events.clear()
        self._continuation_events.clear()
        self._prepared_continuation = None
        self._prepared_selection = None
        self._prepared_release_reason = None
        self._endpoint_warp_command = None
        self._endpoint_warp_command_enabled = False
        return result

    def _try_continue_skill(self, command):
        state = self._skill_state
        if not self.continuous_skill_enabled or state is None:
            return None
        if bool(state.endpoint_translation_warp_world_xy.abs().max().item()) or bool(
            state.endpoint_yaw_warp_rad.abs().item()
        ):
            return None
        endpoint = next_sequential_horizon_endpoint(
            state.skill.support_mask,
            current_endpoint_frame_exclusive=state.playback_stop,
            playback_stop=state.skill.interval.playback_stop,
        )
        if endpoint is None:
            return None

        start = state.playback_stop
        clip = self.dataset.folder.clips[state.skill.clip_index]
        layout = self.dataset.folder.layout
        root_index = int(layout.root_body_index)
        feet_indices = (
            int(layout.left_foot_body_index),
            int(layout.right_foot_body_index),
        )
        source_roots = torch.tensor(
            clip.body_position_world[start - 1 : endpoint, root_index],
            dtype=state.translation_world.dtype,
            device=state.translation_world.device,
        )
        progress = float(
            torch.linalg.vector_norm(
                source_roots[-1, :2] - source_roots[0, :2]
            ).item()
        )
        stalls = remaining_stall_profile(source_roots[:, :2])
        if (
            progress < self.minimum_continuation_progress_m
            or int(stalls.max().item()) > self.maximum_continuation_stall_frames
        ):
            return None
        if not self._source_suffix_matches_command(
            clip=clip,
            start_frame=start,
            endpoint_frame_exclusive=endpoint,
            yaw_offset=state.yaw_offset,
            command=command,
        ):
            return None

        source_feet = torch.tensor(
            clip.body_position_world[start - 1 : endpoint, feet_indices, :],
            dtype=state.translation_world.dtype,
            device=state.translation_world.device,
        )
        placed_feet = torch.empty_like(source_feet)
        placed_feet[..., :2] = _rotate_xy(
            source_feet[..., :2], state.yaw_offset
        ) + state.translation_world[:2]
        placed_feet[..., 2] = source_feet[..., 2] + state.translation_world[2]

        def sample_query(points_xy: torch.Tensor) -> torch.Tensor:
            return self.query_terrain.query_grid.sample_xy(
                self.query_terrain.alignment.matcher_to_scene_xy(points_xy)
            )

        support = state.skill.support_mask[start - 1 : endpoint]
        source_surface = state.skill.foot_surface_height_m[
            start - 1 : endpoint
        ].to(dtype=placed_feet.dtype)
        if self.contact_feasibility_enabled:
            validation = validate_placed_contact_trace(
                foot_position_world=placed_feet,
                support_mask=support,
                source_surface_height_m=source_surface,
                sample_surface=sample_query,
                config=self.contact_feasibility_config,
                align_initial_support=True,
            )
        else:
            try:
                target_surface = sample_query(placed_feet[..., :2])
            except ContractError:
                validation = TerrainContactFeasibilityResult(
                    accepted=False,
                    reason="terrain-domain",
                    maximum_stance_error_m=0.0,
                    landing_error_m=0.0,
                    minimum_swing_clearance_m=0.0,
                    maximum_footprint_height_range_m=0.0,
                    maximum_height_deformation_m=0.0,
                )
            else:
                anchor_mask = support[0]
                if not bool(anchor_mask.any().item()):
                    raise ContractError(
                        "continuous terrain skill has no boundary support"
                    )
                source_anchor = source_surface[0, anchor_mask].mean()
                target_anchor = target_surface[0, anchor_mask].mean()
                surface_error = float(
                    torch.abs(
                        (target_surface - target_anchor)
                        - (source_surface - source_anchor)
                    )[support].max().item()
                )
                accepted = (
                    surface_error <= self.continuation_surface_tolerance_m
                )
                validation = TerrainContactFeasibilityResult(
                    accepted=accepted,
                    reason=None if accepted else "surface-profile",
                    maximum_stance_error_m=surface_error,
                    landing_error_m=0.0,
                    minimum_swing_clearance_m=0.0,
                    maximum_footprint_height_range_m=0.0,
                    maximum_height_deformation_m=surface_error,
                )
        if not validation.accepted:
            return None
        self._prepared_continuation = TerrainContinuationEvent(
            skill_index=state.skill.skill_index,
            start_frame=start,
            endpoint_frame_exclusive=endpoint,
            command=command,
            validation=validation,
        )
        return extend_skill_state(state, playback_stop=endpoint)

    def _update_endpoint_warp_command(self, command) -> None:
        if self._last_result is None:
            raise ContractError("endpoint warp command requires reset")
        try:
            velocity, requested_heading = command
            normalized = (
                (float(velocity[0]), float(velocity[1])),
                float(requested_heading),
            )
        except (IndexError, TypeError, ValueError) as error:
            raise ContractError("endpoint warp command is invalid") from error
        current_yaw = _yaw_from_wxyz(
            self._last_result.root_orientation_world_wxyz
        )
        desired = torch.as_tensor(
            normalized[1], dtype=current_yaw.dtype, device=current_yaw.device
        )
        magnitude = abs(
            float(
                torch.atan2(
                    torch.sin(desired - current_yaw),
                    torch.cos(desired - current_yaw),
                ).item()
            )
        )
        speed = math.hypot(*normalized[0])
        alignment = (
            1.0
            if speed <= 1e-8
            else (
                normalized[0][0] * math.cos(normalized[1])
                + normalized[0][1] * math.sin(normalized[1])
            )
            / speed
        )
        self._endpoint_warp_command = normalized
        self._endpoint_warp_command_enabled = (
            self.minimum_endpoint_warp_yaw_rad
            <= magnitude
            <= self.maximum_endpoint_warp_yaw_rad
            and alignment
            >= self.minimum_endpoint_warp_velocity_heading_alignment
        )

    def _try_start_skill(
        self,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
        shaped: Any,
    ):
        result = self._last_result
        feature_state = GeneratedFeatureState(
            root_position_world=result.root_position_world,
            root_orientation_world_wxyz=result.root_orientation_world_wxyz,
            root_linear_velocity_world=self._root_velocity,
            left_foot_position_world=self._feet[0],
            right_foot_position_world=self._feet[1],
            left_foot_velocity_world=self._foot_velocity[0],
            right_foot_velocity_world=self._foot_velocity[1],
        )
        normalized_query = self.database.normalization.normalize(
            extract_query_features(feature_state, shaped.trajectory)
        )
        current_yaw = _yaw_from_wxyz(result.root_orientation_world_wxyz)
        requested_velocity = torch.tensor(
            velocity_world_xy, dtype=torch.float32, device=self.database.device
        )
        requested_heading = torch.tensor(
            heading_world_yaw, dtype=torch.float32, device=self.database.device
        )
        requested_command = (
            (float(velocity_world_xy[0]), float(velocity_world_xy[1])),
            float(heading_world_yaw),
        )
        command_changed = (
            self._last_command is not None
            and requested_command != self._last_command
        )
        search_current_velocity = (
            requested_velocity
            if self.contact_phase_gate and command_changed
            else self._shaped_velocity
        )
        targets = predict_horizon_targets(
            current_velocity_world_xy=search_current_velocity,
            current_heading_world_yaw=current_yaw,
            requested_velocity_world_xy=requested_velocity,
            requested_heading_world_yaw=requested_heading,
            matcher_config=self.config,
        )
        targets = terrain_height_targets(
            targets,
            current_root_position_world=result.root_position_world,
            current_root_yaw=current_yaw,
            query_terrain=self.query_terrain,
        )
        desired_turning_by_frames = {
            int(frame.item()): abs(float(yaw.item()))
            >= self.search_config.turn_gate_rad
            for frame, yaw in zip(targets.frames, targets.yaw_delta_rad)
        }

        current_clip: int | None = None
        current_frame: int | None = None
        if self._skill_state is not None:
            current_clip = self._skill_state.skill.clip_index
            current_frame = self._skill_state.next_source_frame - 1
        elif self.base._state is not None:
            current_clip = self.base._state.clip_index
            current_frame = self.base._state.frame_index

        current_support = None
        if self.contact_phase_gate:
            current_path = str(result.diagnostics.selected_clip_path)
            source_clip = self._clip_path_to_index.get(current_path)
            support_profile = self._support_by_clip.get(source_clip)
            source_frame = int(result.diagnostics.selected_frame)
            if support_profile is not None and 0 <= source_frame < len(
                support_profile
            ):
                current_support = support_profile[source_frame]

        def compatible(
            record: int,
            row: int,
            endpoint: int,
            *,
            require_phase: bool,
        ) -> bool:
            skill_index = int(self.horizon_inventory.skill_index[record].item())
            skill = self.inventory.skills[skill_index]
            if self.turning_clip_paths is not None:
                target_frames = int(
                    self.horizon_inventory.target_frames[record].item()
                )
                clip_path = self.dataset.folder.clips[
                    skill.clip_index
                ].relative_path
                if (
                    desired_turning_by_frames[target_frames]
                    and clip_path not in self.turning_clip_paths
                ):
                    return False
            if require_phase and current_support is not None:
                entry_frame = int(
                    self.horizon_inventory.entry_frame[record].item()
                )
                if not torch.equal(
                    current_support, skill.support_mask[entry_frame]
                ):
                    return False
            if not terrain_skill_compatible(
                skill=skill,
                canonical_entry_row=row,
                dataset=self.dataset,
                database=self.database,
                query_terrain=self.query_terrain,
                current_root_position_world=result.root_position_world,
                current_root_orientation_world_wxyz=result.root_orientation_world_wxyz,
                tolerance_m=self.terrain_tolerance_m,
                playback_stop=endpoint,
            ):
                return False
            if not self.emitted_contact_preview_enabled:
                return True

            entry_frame = int(
                self.horizon_inventory.entry_frame[record].item()
            )

            def sample_query(points_xy: torch.Tensor) -> torch.Tensor:
                return self.query_terrain.query_grid.sample_xy(
                    self.query_terrain.alignment.matcher_to_scene_xy(
                        points_xy
                    )
                )

            validation = preview_emitted_contact_trace(
                folder=self.dataset.folder,
                skill=skill,
                selected_entry_frame=entry_frame,
                endpoint_frame_exclusive=endpoint,
                current=self._current_pose(),
                halflife_s=self.config.inertialization_halflife_s,
                foot_kinematics=self.foot_kinematics,
                sample_surface=sample_query,
                config=self.contact_feasibility_config,
            )
            return validation.accepted

        def select(*, require_phase: bool):
            prefer_runway = (
                self.continuous_skill_enabled
                and math.hypot(*velocity_world_xy) > 0.05
            )

            def runway_compatible(
                record: int, row: int, endpoint: int
            ) -> bool:
                skill_index = int(
                    self.horizon_inventory.skill_index[record].item()
                )
                skill = self.inventory.skills[skill_index]
                if endpoint >= skill.interval.playback_stop:
                    return False
                later = next_sequential_horizon_endpoint(
                    skill.support_mask,
                    current_endpoint_frame_exclusive=endpoint,
                    playback_stop=skill.interval.playback_stop,
                )
                if later is None:
                    return False
                clip = self.dataset.folder.clips[skill.clip_index]
                root_index = int(self.dataset.folder.layout.root_body_index)
                roots = torch.tensor(
                    clip.body_position_world[endpoint - 1 : later, root_index],
                    dtype=torch.float32,
                    device=self.database.device,
                )
                progress = float(
                    torch.linalg.vector_norm(
                        roots[-1, :2] - roots[0, :2]
                    ).item()
                )
                if progress < self.minimum_continuation_progress_m:
                    return False
                if int(remaining_stall_profile(roots[:, :2]).max().item()) > (
                    self.maximum_continuation_stall_frames
                ):
                    return False
                entry_frame = int(
                    self.horizon_inventory.entry_frame[record].item()
                )
                entry_orientation = torch.tensor(
                    clip.body_quaternion_world_wxyz[
                        entry_frame,
                        int(self.dataset.folder.layout.root_body_index),
                    ],
                    dtype=current_yaw.dtype,
                    device=current_yaw.device,
                )
                yaw_offset = current_yaw - _yaw_from_wxyz(
                    entry_orientation
                )
                if not self._source_suffix_matches_command(
                    clip=clip,
                    start_frame=endpoint,
                    endpoint_frame_exclusive=later,
                    yaw_offset=yaw_offset,
                    command=(velocity_world_xy, heading_world_yaw),
                ):
                    return False
                return terrain_skill_compatible(
                    skill=skill,
                    canonical_entry_row=row,
                    dataset=self.dataset,
                    database=self.database,
                    query_terrain=self.query_terrain,
                    current_root_position_world=result.root_position_world,
                    current_root_orientation_world_wxyz=(
                        result.root_orientation_world_wxyz
                    ),
                    tolerance_m=self.continuation_surface_tolerance_m,
                    playback_stop=later,
                )

            return select_horizon_candidate(
                self.database,
                self.horizon_inventory,
                normalized_query,
                targets,
                terrain_validator=lambda record, row, endpoint: compatible(
                    record,
                    row,
                    endpoint,
                    require_phase=require_phase,
                ),
                preferred_validator=(
                    runway_compatible if prefer_runway else None
                ),
                maximum_preferred_cost_increase=(
                    self.maximum_preferred_cost_increase
                ),
                maximum_preferred_outcome_cost_increase=(
                    self.maximum_preferred_outcome_cost_increase
                ),
                config=self.search_config,
                current_clip_index=current_clip,
                current_frame_index=current_frame,
                matcher_config=self.config,
            )

        try:
            selected = select(require_phase=current_support is not None)
        except HorizonSearchFailure:
            if current_support is None:
                raise
            selected = select(require_phase=False)
        skill_index = int(
            self.horizon_inventory.skill_index[selected.record_index].item()
        )
        if not 0 <= skill_index < len(self.inventory.skills):
            raise ContractError("selected horizon has no terrain skill owner")
        skill = self.inventory.skills[skill_index]
        if self.inventory.row_to_skill.get(selected.entry_row) != skill_index:
            raise ContractError("selected horizon row ownership is inconsistent")
        entry_frame = int(
            self.horizon_inventory.entry_frame[selected.record_index].item()
        )
        if self._skill_state is None:
            release_reason = "initial"
        elif self._replan_pending or (
            self._last_command is not None
            and (
                (float(velocity_world_xy[0]), float(velocity_world_xy[1])),
                float(heading_world_yaw),
            )
            != self._last_command
        ):
            release_reason = "command_change"
        else:
            release_reason = "endpoint"
        self._prepared_selection = selected
        self._prepared_release_reason = release_reason
        target_index = int(
            torch.nonzero(
                targets.frames == selected.target_frames, as_tuple=False
            ).flatten()[0].item()
        )
        warp_displacement = targets.displacement_local_xy[target_index]
        warp_yaw = targets.yaw_delta_rad[target_index]
        terrain_delta = float(
            torch.max(
                torch.abs(
                    torch.cat(
                        (
                            targets.root_height_delta_m[target_index].reshape(1),
                            targets.surface_height_delta_m[target_index],
                        )
                    )
                )
            ).item()
        )
        warp_enabled = (
            self._endpoint_warp_command_enabled
            and terrain_delta <= self.maximum_endpoint_warp_terrain_delta_m
        )
        translation_limit = (
            self.maximum_translation_warp_m if warp_enabled else 0.0
        )
        yaw_limit = self.maximum_yaw_warp_rad if warp_enabled else 0.0
        if (
            translation_limit > 0.0
            or yaw_limit > 0.0
        ):
            exact_duration = selected.endpoint_frame_exclusive - entry_frame
            exact_target = predict_horizon_targets(
                current_velocity_world_xy=self._shaped_velocity,
                current_heading_world_yaw=current_yaw,
                requested_velocity_world_xy=requested_velocity,
                requested_heading_world_yaw=requested_heading,
                matcher_config=self.config,
                target_frames=(exact_duration,),
            )
            warp_displacement = exact_target.displacement_local_xy[0]
            warp_yaw = exact_target.yaw_delta_rad[0]
        return start_skill(
            self.dataset.folder,
            skill,
            selected_entry_frame=entry_frame,
            current=self._current_pose(),
            halflife_s=self.config.inertialization_halflife_s,
            playback_stop=selected.endpoint_frame_exclusive,
            target_displacement_local_xy=warp_displacement,
            target_yaw_delta_rad=warp_yaw,
            maximum_translation_warp_m=translation_limit,
            maximum_yaw_warp_rad=yaw_limit,
        )

    def prepare_step(self, *args, **kwargs):
        self._prepared_selection = None
        self._prepared_release_reason = None
        self._prepared_continuation = None
        try:
            velocity = (
                args[0]
                if len(args) >= 1
                else kwargs["velocity_world_xy"]
            )
            heading = (
                args[1]
                if len(args) >= 2
                else kwargs["heading_world_yaw"]
            )
            command = (
                (float(velocity[0]), float(velocity[1])),
                float(heading),
            )
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ContractError("endpoint warp step command is invalid") from error
        if command != self._endpoint_warp_command:
            self._update_endpoint_warp_command(command)
        state = self._skill_state
        command_changed = (
            self._last_command is not None and command != self._last_command
        )
        if (
            self.contact_phase_gate
            and command_changed
            and state is not None
            and state.next_source_frame < state.playback_stop
        ):
            self._skill_state = replace(
                state, playback_stop=state.next_source_frame
            )
            try:
                return super().prepare_step(*args, **kwargs)
            finally:
                self._skill_state = state
        return super().prepare_step(*args, **kwargs)

    def commit(self, prepared):
        selected = self._prepared_selection
        release_reason = self._prepared_release_reason
        continuation = self._prepared_continuation
        result = super().commit(prepared)
        if selected is not None:
            skill_index = int(
                self.horizon_inventory.skill_index[selected.record_index].item()
            )
            self._events.append(
                HorizonChunkEvent(
                    entry_row=selected.entry_row,
                    skill_index=skill_index,
                    target_frames=selected.target_frames,
                    endpoint_frame_exclusive=selected.endpoint_frame_exclusive,
                    cost=selected.cost,
                    rejected_by_reason=MappingProxyType(
                        dict(selected.rejected_by_reason)
                    ),
                    release_reason=release_reason or "endpoint",
                    selection_mode=selected.selection_mode,
                )
            )
        if continuation is not None:
            self._continuation_events.append(continuation)
        self._prepared_selection = None
        self._prepared_release_reason = None
        self._prepared_continuation = None
        return result


def run_resolved_horizon_matrix(
    resolved,
    *,
    g1_xml: str,
    routes: Sequence[OmniRoute],
    terrain_tolerance_m: float = 0.06,
    search_config: HorizonSearchConfig = HorizonSearchConfig(),
    foot_lock: bool = False,
    contact_phase_gate: bool = False,
    continuous_skill_enabled: bool = False,
    swing_clearance_margin_m: float | None = None,
    foot_correction_halflife_s: float = 0.04,
    swing_plan_sigma_frames: float | None = None,
    maximum_source_contact_p95_m: float | None = None,
    normalization_source: str | None = None,
    turning_source_corpus: str | None = None,
    maximum_translation_warp_m: float = 0.0,
    maximum_yaw_warp_rad: float = 0.0,
    minimum_endpoint_warp_yaw_rad: float = 0.0,
    maximum_endpoint_warp_yaw_rad: float = math.pi,
    maximum_endpoint_warp_terrain_delta_m: float = math.inf,
    minimum_endpoint_warp_velocity_heading_alignment: float = -1.0,
):
    """Run the renderer-independent route harness with horizon skill MM."""

    if swing_clearance_margin_m is not None and not foot_lock:
        raise ContractError("swing clearance requires terrain foot lock")

    import mujoco

    from .torch_g1_fk import MujocoG1FootKinematics
    from .torch_terrain_live_viewer import (
        apply_kinematic_state,
        build_kinematic_scene,
        matcher_result_qpos,
    )
    from .torch_terrain_omni_rollout import (
        KinematicSample,
        resolved_stair_reset_position,
        run_omni_matrix,
    )
    from .torch_terrain_omni_routes import StairFrame
    from .torch_terrain_rollout import matcher_config_from_resolved

    config = matcher_config_from_resolved(resolved.resolved_config)
    normalization_override = None
    normalization_identity = "self"
    if normalization_source is not None:
        source_folder = MotionFolder.load(normalization_source)
        source_database = TorchMotionDatabase.from_folder(
            source_folder,
            device=resolved.device,
            reset_clip_path=resolved.resolved_config["reset_clip"],
        )
        normalization_override = source_database.normalization
        normalization_identity = source_folder.inventory_sha256
        del source_database, source_folder
    turning_clip_paths = None
    turning_identity = "all"
    if turning_source_corpus is not None:
        turning_folder = MotionFolder.load(turning_source_corpus)
        turning_clip_paths = frozenset(
            clip.relative_path for clip in turning_folder.clips
        )
        turning_identity = turning_folder.inventory_sha256
        del turning_folder
    base = TorchMotionMatcher.from_folder(
        resolved.dataset.root,
        device=str(resolved.device),
        config=config,
        reset_clip_path=resolved.resolved_config["reset_clip"],
        normalization_override=normalization_override,
    )
    skills = build_terrain_skill_inventory(
        resolved.dataset,
        base.database,
        maximum_source_contact_p95_m=maximum_source_contact_p95_m,
    )
    horizons = build_horizon_inventory(resolved.dataset, base.database, skills)
    foot_kinematics = MujocoG1FootKinematics(g1_xml)
    result_filter = None
    if foot_lock:
        from .torch_terrain_foot_lock import build_terrain_foot_lock

        result_filter = build_terrain_foot_lock(
            resolved,
            foot_kinematics,
            swing_clearance_margin_m=swing_clearance_margin_m,
            correction_halflife_s=foot_correction_halflife_s,
            swing_plan_sigma_frames=swing_plan_sigma_frames,
        )
    route_matchers: dict[str, TerrainSkillHorizonMatcher] = {}
    model, data = build_kinematic_scene(g1_xml, resolved)
    left_ankle = int(model.body("left_ankle_roll_link").id)
    right_ankle = int(model.body("right_ankle_roll_link").id)

    def kinematics(result) -> KinematicSample:
        qpos = matcher_result_qpos(result)
        apply_kinematic_state(mujoco, model, data, qpos)
        w, x, y, z = qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return KinematicSample(
            qpos=qpos,
            joint_position=result.joint_position.detach().cpu().numpy(),
            joint_velocity=result.joint_velocity.detach().cpu().numpy(),
            root_position_world=result.root_position_world.detach().cpu().numpy(),
            root_yaw_world=yaw,
            foot_position_world=np.stack(
                (data.xpos[left_ankle], data.xpos[right_ankle])
            ).astype(np.float64, copy=True),
        )

    measurement = resolved.measurement_extension

    def terrain_sampler(matcher_xy: np.ndarray) -> np.ndarray:
        points = torch.tensor(
            matcher_xy, dtype=torch.float32, device=resolved.device
        )
        return (
            measurement.query_grid.sample_xy(
                measurement.alignment.matcher_to_scene_xy(points)
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    direction = np.asarray(
        resolved.resolved_config["reference_direction_matcher_xy"],
        dtype=np.float64,
    )
    stair_frame = StairFrame(
        origin_world_xy=(0.0, 0.0),
        ascent_world_yaw=math.atan2(float(direction[1]), float(direction[0])),
        width_m=0.6223,
        tread_depth_m=0.3302,
        riser_height_m=0.1778,
        tread_count=3,
    )
    search_identity = json.dumps(
        asdict(search_config), sort_keys=True, separators=(",", ":")
    )
    identity = hashlib.sha256(
        (
            resolved.base_config_sha256
            + ":horizon-skills-v1:"
            + search_identity
            + (":foot-lock-v1" if foot_lock else ":raw-playback")
            + (
                f":swing-clearance-v1:{float(swing_clearance_margin_m):.9g}"
                if swing_clearance_margin_m is not None
                else ":native-swing"
            )
            + f":foot-halflife:{float(foot_correction_halflife_s):.9g}"
            + (
                f":swing-plan-v1:{float(swing_plan_sigma_frames):.9g}"
                if swing_plan_sigma_frames is not None
                else ":reactive-clearance"
            )
            + (":phase-gate-v1" if contact_phase_gate else ":implicit-phase")
            + (
                ":continuous-skill-v1"
                if continuous_skill_enabled
                else ":boundary-search-v1"
            )
            + (
                ":continuous-surface-tolerance:0.08"
                if continuous_skill_enabled
                else ""
            )
            + (
                f":source-contact-p95:{float(maximum_source_contact_p95_m):.9g}"
                if maximum_source_contact_p95_m is not None
                else ":all-source-contact-quality"
            )
            + f":normalization:{normalization_identity}"
            + f":turning-source:{turning_identity}"
            + f":translation-warp:{float(maximum_translation_warp_m):.9g}"
            + f":yaw-warp:{float(maximum_yaw_warp_rad):.9g}"
            + f":warp-yaw-min:{float(minimum_endpoint_warp_yaw_rad):.9g}"
            + f":warp-yaw-max:{float(maximum_endpoint_warp_yaw_rad):.9g}"
            + (
                ":warp-terrain-max:"
                f"{float(maximum_endpoint_warp_terrain_delta_m):.9g}"
            )
            + (
                ":warp-velocity-heading-min:"
                f"{float(minimum_endpoint_warp_velocity_heading_alignment):.9g}"
            )
        ).encode()
    ).hexdigest()

    def matcher_factory(route: OmniRoute) -> TerrainSkillHorizonMatcher:
        matcher = TerrainSkillHorizonMatcher(
            base_matcher=base,
            skill_inventory=skills,
            horizon_inventory=horizons,
            dataset=resolved.dataset,
            query_terrain=resolved.measurement_extension,
            foot_kinematics=foot_kinematics,
            config=config,
            search_config=search_config,
            terrain_tolerance_m=terrain_tolerance_m,
            result_filter=result_filter,
            contact_phase_gate=contact_phase_gate,
            continuous_skill_enabled=continuous_skill_enabled,
            turning_clip_paths=turning_clip_paths,
            maximum_translation_warp_m=maximum_translation_warp_m,
            maximum_yaw_warp_rad=maximum_yaw_warp_rad,
            minimum_endpoint_warp_yaw_rad=minimum_endpoint_warp_yaw_rad,
            maximum_endpoint_warp_yaw_rad=maximum_endpoint_warp_yaw_rad,
            maximum_endpoint_warp_terrain_delta_m=(
                maximum_endpoint_warp_terrain_delta_m
            ),
            minimum_endpoint_warp_velocity_heading_alignment=(
                minimum_endpoint_warp_velocity_heading_alignment
            ),
        )
        route_matchers[route.name] = matcher
        return matcher

    matrix = run_omni_matrix(
        routes=tuple(routes),
        stair_frame=stair_frame,
        matcher_factory=matcher_factory,
        kinematics=kinematics,
        terrain_sampler=terrain_sampler,
        dataset_identity=resolved.dataset.manifest_sha256,
        config_identity=identity,
        reset_root_position_world_xy=resolved_stair_reset_position(resolved),
    )
    return matrix, {
        name: matcher.chunk_events for name, matcher in route_matchers.items()
    }
