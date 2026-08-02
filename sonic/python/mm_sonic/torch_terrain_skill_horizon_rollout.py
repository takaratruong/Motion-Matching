"""Transactional fixed-horizon terrain-skill rollout and qualification adapter."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .joints import ContractError
from .torch_motion_features import GeneratedFeatureState, extract_query_features
from .torch_motion_matcher import MatcherConfig, TorchMotionMatcher
from .torch_terrain_omni_routes import OmniRoute
from .torch_terrain_skill_composer import start_skill
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


def terrain_height_targets(
    targets: HorizonTargets,
    *,
    current_root_position_world: torch.Tensor,
    current_root_yaw: torch.Tensor,
    current_foot_position_world: torch.Tensor | None = None,
    footprint_split_gain: float = 1.0,
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
    if current_foot_position_world is not None and (
        not isinstance(current_foot_position_world, torch.Tensor)
        or tuple(current_foot_position_world.shape) != (2, 3)
        or current_foot_position_world.device
        != current_root_position_world.device
        or current_foot_position_world.dtype != current_root_position_world.dtype
        or not torch.isfinite(current_foot_position_world).all()
    ):
        raise ContractError("terrain height prediction feet are invalid")
    if (
        isinstance(footprint_split_gain, bool)
        or not isinstance(footprint_split_gain, (int, float))
        or not math.isfinite(float(footprint_split_gain))
        or not 0.0 <= float(footprint_split_gain) <= 1.0
    ):
        raise ContractError("footprint split gain is invalid")
    try:
        world_displacement = _rotate_xy(
            targets.displacement_local_xy, current_root_yaw.reshape(())
        )
        world_xy = current_root_position_world[:2].unsqueeze(0) + world_displacement
        samples = [current_root_position_world[:2].unsqueeze(0), world_xy]
        future_foot_xy = None
        if current_foot_position_world is not None:
            local_foot_offset = _rotate_xy(
                current_foot_position_world[:, :2]
                - current_root_position_world[:2],
                -current_root_yaw.reshape(()),
            )
            future_yaw = (
                current_root_yaw.reshape(()) + targets.yaw_delta_rad
            )
            future_foot_xy = world_xy[:, None, :] + _rotate_xy(
                local_foot_offset.unsqueeze(0).expand(
                    targets.frames.shape[0], -1, -1
                ),
                future_yaw[:, None],
            )
            samples.extend(
                (
                    current_foot_position_world[:, :2],
                    future_foot_xy.reshape(-1, 2),
                )
            )
        sample_xy = torch.cat(samples, dim=0)
        height = query_terrain.query_grid.sample_xy(
            query_terrain.alignment.matcher_to_scene_xy(sample_xy)
        )
    except (AttributeError, TypeError) as error:
        raise ContractError("terrain height prediction query is invalid") from error
    if (
        not isinstance(height, torch.Tensor)
        or tuple(height.shape) != (sample_xy.shape[0],)
        or height.device != current_root_position_world.device
        or not torch.isfinite(height).all()
    ):
        raise ContractError("terrain height prediction samples are invalid")
    horizon_count = targets.frames.shape[0]
    delta = (height[1 : horizon_count + 1] - height[0]).to(
        targets.displacement_local_xy.dtype
    )
    if current_foot_position_world is None:
        surface_delta = delta[:, None].expand(-1, 2).clone()
    else:
        current_surface = height[horizon_count + 1 : horizon_count + 3]
        future_surface = height[horizon_count + 3 :].reshape(
            horizon_count, 2
        )
        raw_surface_delta = (future_surface - current_surface).to(
            targets.displacement_local_xy.dtype
        )
        lateral_split = raw_surface_delta - raw_surface_delta.mean(
            dim=1, keepdim=True
        )
        surface_delta = delta[:, None] + (
            float(footprint_split_gain) * lateral_split
        )
    return HorizonTargets(
        frames=targets.frames,
        displacement_local_xy=targets.displacement_local_xy,
        yaw_delta_rad=targets.yaw_delta_rad,
        root_height_delta_m=delta,
        surface_height_delta_m=surface_delta,
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
        footprint_terrain_targets: bool = False,
        footprint_split_gain: float = 1.0,
        footprint_preserve_horizon: bool = False,
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
        if type(footprint_terrain_targets) is not bool:
            raise ContractError("footprint terrain targets must be boolean")
        self.footprint_terrain_targets = footprint_terrain_targets
        if (
            isinstance(footprint_split_gain, bool)
            or not isinstance(footprint_split_gain, (int, float))
            or not math.isfinite(float(footprint_split_gain))
            or not 0.0 <= float(footprint_split_gain) <= 1.0
        ):
            raise ContractError("footprint split gain is invalid")
        self.footprint_split_gain = float(footprint_split_gain)
        if type(footprint_preserve_horizon) is not bool or (
            footprint_preserve_horizon and not footprint_terrain_targets
        ):
            raise ContractError(
                "footprint horizon layer requires footprint terrain targets"
            )
        self.footprint_preserve_horizon = footprint_preserve_horizon
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
        self._prepared_selection: TerrainSkillHorizonResult | None = None
        self._prepared_release_reason: str | None = None

    @property
    def chunk_events(self) -> tuple[HorizonChunkEvent, ...]:
        return tuple(self._events)

    def reset(
        self, *, root_position_world_xy: tuple[float, float] = (0.0, 0.0)
    ):
        result = super().reset(root_position_world_xy=root_position_world_xy)
        self._events.clear()
        self._prepared_selection = None
        self._prepared_release_reason = None
        return result

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
        command_targets = predict_horizon_targets(
            current_velocity_world_xy=self._shaped_velocity,
            current_heading_world_yaw=current_yaw,
            requested_velocity_world_xy=requested_velocity,
            requested_heading_world_yaw=requested_heading,
            matcher_config=self.config,
        )
        baseline_targets = terrain_height_targets(
            command_targets,
            current_root_position_world=result.root_position_world,
            current_root_yaw=current_yaw,
            query_terrain=self.query_terrain,
        )
        targets = baseline_targets
        if self.footprint_terrain_targets:
            targets = terrain_height_targets(
                command_targets,
                current_root_position_world=result.root_position_world,
                current_root_yaw=current_yaw,
                current_foot_position_world=self._feet,
                footprint_split_gain=self.footprint_split_gain,
                query_terrain=self.query_terrain,
            )

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
            if require_phase and current_support is not None:
                entry_frame = int(
                    self.horizon_inventory.entry_frame[record].item()
                )
                if not torch.equal(
                    current_support, skill.support_mask[entry_frame]
                ):
                    return False
            return terrain_skill_compatible(
                skill=skill,
                canonical_entry_row=row,
                dataset=self.dataset,
                database=self.database,
                query_terrain=self.query_terrain,
                current_root_position_world=result.root_position_world,
                current_root_orientation_world_wxyz=result.root_orientation_world_wxyz,
                tolerance_m=self.terrain_tolerance_m,
                playback_stop=endpoint,
            )

        def select(
            target_values: HorizonTargets,
            *,
            require_phase: bool,
            required_target_frames: int | None = None,
        ):
            return select_horizon_candidate(
                self.database,
                self.horizon_inventory,
                normalized_query,
                target_values,
                terrain_validator=lambda record, row, endpoint: compatible(
                    record,
                    row,
                    endpoint,
                    require_phase=require_phase,
                ),
                config=self.search_config,
                current_clip_index=current_clip,
                current_frame_index=current_frame,
                matcher_config=self.config,
                required_target_frames=required_target_frames,
            )

        def select_with_phase(
            target_values: HorizonTargets,
            *,
            required_target_frames: int | None = None,
        ):
            try:
                return select(
                    target_values,
                    require_phase=current_support is not None,
                    required_target_frames=required_target_frames,
                )
            except HorizonSearchFailure:
                if current_support is None:
                    raise
                return select(
                    target_values,
                    require_phase=False,
                    required_target_frames=required_target_frames,
                )

        if self.footprint_preserve_horizon:
            baseline_selected = select_with_phase(baseline_targets)
            try:
                selected = select_with_phase(
                    targets,
                    required_target_frames=baseline_selected.target_frames,
                )
            except HorizonSearchFailure:
                selected = baseline_selected
        else:
            selected = select_with_phase(targets)
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
        return start_skill(
            self.dataset.folder,
            skill,
            selected_entry_frame=entry_frame,
            current=self._current_pose(),
            halflife_s=self.config.inertialization_halflife_s,
            playback_stop=selected.endpoint_frame_exclusive,
        )

    def prepare_step(self, *args, **kwargs):
        self._prepared_selection = None
        self._prepared_release_reason = None
        return super().prepare_step(*args, **kwargs)

    def commit(self, prepared):
        selected = self._prepared_selection
        release_reason = self._prepared_release_reason
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
                )
            )
        self._prepared_selection = None
        self._prepared_release_reason = None
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
    footprint_terrain_targets: bool = False,
    footprint_split_gain: float = 1.0,
    footprint_preserve_horizon: bool = False,
    swing_clearance_margin_m: float | None = None,
    foot_correction_halflife_s: float = 0.04,
    swing_plan_sigma_frames: float | None = None,
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
    base = TorchMotionMatcher.from_folder(
        resolved.dataset.root,
        device=str(resolved.device),
        config=config,
        reset_clip_path=resolved.resolved_config["reset_clip"],
    )
    skills = build_terrain_skill_inventory(resolved.dataset, base.database)
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
                ":footprint-terrain-targets-v1"
                if footprint_terrain_targets
                else ":root-terrain-targets"
            )
            + f":footprint-split-gain:{float(footprint_split_gain):.9g}"
            + (
                ":preserve-baseline-horizon-v1"
                if footprint_preserve_horizon
                else ":free-footprint-horizon"
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
            footprint_terrain_targets=footprint_terrain_targets,
            footprint_split_gain=footprint_split_gain,
            footprint_preserve_horizon=footprint_preserve_horizon,
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
