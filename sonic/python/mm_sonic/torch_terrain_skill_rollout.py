"""Fast qualification helpers for PHP-style coherent terrain skills."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import SimpleNamespace
import time
from typing import Any, Sequence

import numpy as np
import torch

from .joints import ContractError
from .torch_motion_features import (
    GeneratedFeatureState,
    TorchMotionDatabase,
    extract_query_features,
)
from .torch_motion_matcher import (
    MatcherConfig,
    TorchMotionMatcher,
    predict_command_trajectory,
)
from .torch_terrain_features import TerrainDataset, TerrainFeatureExtension
from .torch_terrain_omni_routes import OmniRoute
from .torch_terrain_skill_composer import (
    TerrainSkillPose,
    TerrainSkillState,
    advance_skill,
    can_interrupt_skill,
    start_skill,
)
from .torch_terrain_skill_search import select_terrain_skill
from .torch_terrain_skills import (
    TerrainSkill,
    TerrainSkillInventory,
    build_terrain_skill_inventory,
)


QUALIFICATION_ROUTE_NAMES = (
    "cross-tread-left-to-right",
    "turn-90-middle-left",
    "diagonal-down-left",
    "side-exit-upper-left",
    "riser-reversal",
    "mixed-adversarial",
)


def qualification_routes(routes: Sequence[OmniRoute]) -> tuple[OmniRoute, ...]:
    by_name = {route.name: route for route in routes}
    missing = [name for name in QUALIFICATION_ROUTE_NAMES if name not in by_name]
    if missing:
        raise ValueError(
            "missing qualification routes: " + ", ".join(missing)
        )
    return tuple(by_name[name] for name in QUALIFICATION_ROUTE_NAMES)


def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _rotate_xy(values: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    x, y = values[..., 0], values[..., 1]
    return torch.stack((c * x - s * y, s * x + c * y), dim=-1)


def command_skill_compatible(
    *,
    skill_travel_local_xy: torch.Tensor,
    skill_yaw_delta_rad: torch.Tensor,
    requested_velocity_local_xy: torch.Tensor,
    requested_heading_delta_rad: torch.Tensor,
) -> bool:
    """Gate whole-skill intent before using MM to choose its entry pose."""

    desired_turn = float(requested_heading_delta_rad.item())
    skill_turn = float(skill_yaw_delta_rad.item())
    turn_gate = math.radians(30.0)
    if abs(desired_turn) >= turn_gate:
        required = min(abs(desired_turn) * 0.5, math.radians(60.0))
        if not (skill_turn * desired_turn > 0.0 and abs(skill_turn) >= required):
            return False
        requested_speed = torch.linalg.vector_norm(requested_velocity_local_xy)
        travel_length = torch.linalg.vector_norm(skill_travel_local_xy)
        if float(requested_speed.item()) <= 0.05:
            return True
        if float(travel_length.item()) <= 0.05:
            return False
        alignment = torch.dot(
            requested_velocity_local_xy / requested_speed,
            skill_travel_local_xy / travel_length,
        )
        return bool((alignment >= 0.25).item())
    if abs(skill_turn) > math.radians(60.0):
        return False
    requested_speed = torch.linalg.vector_norm(requested_velocity_local_xy)
    travel_length = torch.linalg.vector_norm(skill_travel_local_xy)
    if float(requested_speed.item()) <= 0.05:
        return True
    if float(travel_length.item()) <= 0.05:
        return False
    alignment = torch.dot(
        requested_velocity_local_xy / requested_speed,
        skill_travel_local_xy / travel_length,
    )
    return bool((alignment >= 0.70).item())


def terrain_skill_compatible(
    *,
    skill: TerrainSkill,
    canonical_entry_row: int,
    dataset: TerrainDataset,
    database: TorchMotionDatabase,
    query_terrain: TerrainFeatureExtension,
    current_root_position_world: torch.Tensor,
    current_root_orientation_world_wxyz: torch.Tensor,
    tolerance_m: float = 0.06,
    sample_stride: int = 5,
) -> bool:
    """Compare a placed skill's supported surface trace to the query scene."""

    if not isinstance(skill, TerrainSkill):
        raise ContractError("terrain compatibility requires a TerrainSkill")
    if type(canonical_entry_row) is not int or not (
        0 <= canonical_entry_row < database.feature_shape[0]
    ):
        raise ContractError("terrain compatibility entry row is invalid")
    if type(sample_stride) is not int or sample_stride < 1:
        raise ContractError("terrain compatibility sample stride is invalid")
    if not isinstance(tolerance_m, (int, float)) or not 0 < float(tolerance_m):
        raise ContractError("terrain compatibility tolerance is invalid")
    reference = current_root_position_world
    if (
        not isinstance(reference, torch.Tensor)
        or tuple(reference.shape) != (3,)
        or not isinstance(current_root_orientation_world_wxyz, torch.Tensor)
        or tuple(current_root_orientation_world_wxyz.shape) != (4,)
        or reference.device != database.device
        or current_root_orientation_world_wxyz.device != database.device
    ):
        raise ContractError("terrain compatibility current root is invalid")
    if int(database._search_clip_index[canonical_entry_row].item()) != skill.clip_index:
        raise ContractError("terrain compatibility row does not own skill clip")
    entry_frame = int(database._search_frame_index[canonical_entry_row].item())
    clip = dataset.folder.clips[skill.clip_index]
    layout = dataset.folder.layout
    root_index = layout.root_body_index
    feet_indices = (layout.left_foot_body_index, layout.right_foot_body_index)
    source_root = torch.tensor(
        clip.body_position_world[entry_frame, root_index],
        dtype=torch.float32,
        device=database.device,
    )
    source_quaternion = torch.tensor(
        clip.body_quaternion_world_wxyz[entry_frame, root_index],
        dtype=torch.float32,
        device=database.device,
    )
    yaw_offset = _yaw_from_wxyz(current_root_orientation_world_wxyz) - _yaw_from_wxyz(
        source_quaternion
    )
    frames = torch.arange(
        entry_frame,
        skill.interval.playback_stop,
        sample_stride,
        dtype=torch.long,
        device=database.device,
    )
    last = skill.interval.playback_stop - 1
    if frames.numel() == 0 or int(frames[-1].item()) != last:
        frames = torch.cat(
            (frames, torch.tensor([last], dtype=torch.long, device=database.device))
        )
    source_feet = torch.tensor(
        clip.body_position_world[:, feet_indices, :2],
        dtype=torch.float32,
        device=database.device,
    )[frames]
    placed_xy = current_root_position_world[:2] + _rotate_xy(
        source_feet - source_root[:2], yaw_offset
    )
    support = skill.support_mask[frames]
    if not bool(support.any().item()):
        return False
    try:
        target_surface = query_terrain.query_grid.sample_xy(
            query_terrain.alignment.matcher_to_scene_xy(placed_xy.reshape(-1, 2))
        ).reshape_as(support)
    except ContractError:
        return False
    source_surface = skill.foot_surface_height_m[frames]
    first_supported = int(torch.nonzero(support.any(dim=1), as_tuple=False)[0].item())
    anchor_mask = support[first_supported]
    source_anchor = source_surface[first_supported][anchor_mask].mean()
    target_anchor = target_surface[first_supported][anchor_mask].mean()
    errors = torch.abs(
        (target_surface - target_anchor) - (source_surface - source_anchor)
    )[support]
    return bool((errors.max() <= float(tolerance_m)).item())


@dataclass(frozen=True)
class _PreparedSkillStep:
    result: Any
    kind: str
    payload: Any
    shaped_velocity: torch.Tensor
    shaped_heading: torch.Tensor
    command: tuple[tuple[float, float], float]
    replan_pending: bool


class TerrainSkillMatcher:
    """Transactional adapter: flat MM entry, then coherent skill playback."""

    def __init__(
        self,
        *,
        base_matcher: TorchMotionMatcher,
        inventory: TerrainSkillInventory,
        dataset: TerrainDataset,
        query_terrain: TerrainFeatureExtension,
        foot_kinematics: Any,
        config: MatcherConfig,
        terrain_tolerance_m: float = 0.06,
    ) -> None:
        self.base = base_matcher
        self.database = base_matcher.database
        self.inventory = inventory
        self.dataset = dataset
        self.query_terrain = query_terrain
        self.foot_kinematics = foot_kinematics
        self.config = config
        self.terrain_tolerance_m = float(terrain_tolerance_m)
        self._pending = None
        self._skill_state: TerrainSkillState | None = None
        self._last_result = None
        self._last_command = None
        self._replan_pending = False
        self._shaped_velocity = torch.zeros(2, device=self.database.device)
        self._shaped_heading = torch.zeros((), device=self.database.device)
        self._feet = None
        self._foot_velocity = torch.zeros((2, 3), device=self.database.device)
        self._root_velocity = torch.zeros(3, device=self.database.device)
        self._intent_travel_local, self._intent_yaw_delta = (
            self._build_intent_rows()
        )

    def _build_intent_rows(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_count = self.database.feature_shape[0]
        travel = torch.full(
            (row_count, 2), float("nan"), device=self.database.device
        )
        yaw_delta = torch.full(
            (row_count,), float("nan"), device=self.database.device
        )
        root_index = self.dataset.folder.layout.root_body_index
        for skill in self.inventory.skills:
            rows = torch.tensor(
                skill.entry_rows, dtype=torch.long, device=self.database.device
            )
            frames = self.database._search_frame_index[rows]
            clip = self.dataset.folder.clips[skill.clip_index]
            roots = torch.tensor(
                clip.body_position_world[:, root_index, :2],
                dtype=torch.float32,
                device=self.database.device,
            )
            quaternions = torch.tensor(
                clip.body_quaternion_world_wxyz[:, root_index],
                dtype=torch.float32,
                device=self.database.device,
            )
            yaws = _yaw_from_wxyz(quaternions)
            steps = torch.atan2(
                torch.sin(yaws[1:] - yaws[:-1]),
                torch.cos(yaws[1:] - yaws[:-1]),
            )
            cumulative = torch.cat(
                (torch.zeros(1, device=self.database.device), torch.cumsum(steps, 0))
            )
            end = skill.interval.playback_stop - 1
            travel[rows] = _rotate_xy(roots[end] - roots[frames], -yaws[frames])
            yaw_delta[rows] = cumulative[end] - cumulative[frames]
        return travel, yaw_delta

    def _command_eligible_rows(
        self,
        requested_local: torch.Tensor,
        desired_heading_delta: torch.Tensor,
    ) -> torch.Tensor:
        valid = torch.isfinite(self._intent_yaw_delta)
        desired = desired_heading_delta
        turn_gate = math.radians(30.0)
        if abs(float(desired.item())) >= turn_gate:
            required = min(abs(float(desired.item())) * 0.5, math.radians(60.0))
            turn_valid = valid & (self._intent_yaw_delta * desired > 0) & (
                torch.abs(self._intent_yaw_delta) >= required
            )
            speed = torch.linalg.vector_norm(requested_local)
            lengths = torch.linalg.vector_norm(self._intent_travel_local, dim=1)
            if float(speed.item()) <= 0.05:
                return turn_valid
            alignment = torch.sum(
                self._intent_travel_local * (requested_local / speed), dim=1
            ) / lengths.clamp_min(1e-8)
            return turn_valid & (lengths > 0.05) & (alignment >= 0.25)
        valid &= torch.abs(self._intent_yaw_delta) <= math.radians(60.0)
        speed = torch.linalg.vector_norm(requested_local)
        lengths = torch.linalg.vector_norm(self._intent_travel_local, dim=1)
        if float(speed.item()) <= 0.05:
            return valid
        alignment = torch.sum(
            self._intent_travel_local * (requested_local / speed), dim=1
        ) / lengths.clamp_min(1e-8)
        return valid & (lengths > 0.05) & (alignment >= 0.70)

    def _feet_for_result(self, result: Any) -> torch.Tensor:
        feet = self.foot_kinematics.foot_positions(
            result.joint_position.detach().cpu().numpy()[None, :],
            result.root_position_world.detach().cpu().numpy()[None, :],
            result.root_orientation_world_wxyz.detach().cpu().numpy()[None, :],
        )[0]
        return torch.tensor(feet, dtype=torch.float32, device=self.database.device)

    def reset(
        self, *, root_position_world_xy: tuple[float, float] = (0.0, 0.0)
    ):
        result = self.base.reset(root_position_world_xy=root_position_world_xy)
        self._pending = None
        self._skill_state = None
        self._last_result = result
        self._last_command = None
        self._replan_pending = False
        self._shaped_velocity.zero_()
        self._shaped_heading.zero_()
        self._feet = self._feet_for_result(result)
        self._foot_velocity.zero_()
        self._root_velocity.zero_()
        return result

    def _current_pose(self) -> TerrainSkillPose:
        result = self._last_result
        return TerrainSkillPose(
            joint_position=result.joint_position,
            joint_velocity=result.joint_velocity,
            root_position_world=result.root_position_world,
            root_orientation_world_wxyz=result.root_orientation_world_wxyz,
            root_linear_velocity_world=self._root_velocity,
            root_angular_velocity_world=torch.zeros_like(self._root_velocity),
        )

    def _try_start_skill(
        self,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
        shaped,
    ) -> TerrainSkillState | None:
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
        raw_query = extract_query_features(feature_state, shaped.trajectory)
        normalized_query = self.database.normalization.normalize(raw_query)
        current_clip = -1
        current_frame = -1000
        if self._skill_state is not None:
            current_clip = self._skill_state.skill.clip_index
            current_frame = self._skill_state.next_source_frame - 1
        elif self.base._state is not None:
            current_clip = self.base._state.clip_index
            current_frame = self.base._state.frame_index

        current_yaw = _yaw_from_wxyz(result.root_orientation_world_wxyz)
        desired_heading_delta = torch.atan2(
            torch.sin(torch.as_tensor(heading_world_yaw, device=self.database.device) - current_yaw),
            torch.cos(torch.as_tensor(heading_world_yaw, device=self.database.device) - current_yaw),
        )
        requested_world = torch.tensor(
            velocity_world_xy, dtype=torch.float32, device=self.database.device
        )
        requested_local = _rotate_xy(requested_world, -current_yaw)

        def compatible(skill: TerrainSkill, row: int) -> bool:
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
                tolerance_m=self.terrain_tolerance_m,
            )

        try:
            selected = select_terrain_skill(
                self.database,
                self.inventory,
                normalized_query,
                current_clip_index=current_clip,
                current_frame_index=current_frame,
                terrain_validator=compatible,
                config=self.config,
                command_eligible_rows=self._command_eligible_rows(
                    requested_local, desired_heading_delta
                ),
            )
        except ContractError as error:
            if "no terrain-compatible" in str(error):
                return None
            raise
        entry_frame = int(
            self.database._search_frame_index[selected.selected_row].item()
        )
        return start_skill(
            self.dataset.folder,
            selected.skill,
            selected_entry_frame=entry_frame,
            current=self._current_pose(),
            halflife_s=self.config.inertialization_halflife_s,
        )

    def _skill_result(self, step, elapsed_ns: int):
        frame = step.frame
        clip = self.dataset.folder.clips[frame.clip_index]
        diagnostics = SimpleNamespace(
            selected_clip_path=clip.relative_path,
            selected_frame=frame.source_frame,
            terrain_safety_override=True,
            step_time_ns=elapsed_ns,
            search_time_ns=None,
        )
        return SimpleNamespace(
            joint_position=frame.joint_position,
            joint_velocity=frame.joint_velocity,
            root_position_world=frame.root_position_world,
            root_orientation_world_wxyz=frame.root_orientation_world_wxyz,
            root_linear_velocity_world=frame.root_linear_velocity_world,
            diagnostics=diagnostics,
        )

    def _hold_result(self, elapsed_ns: int):
        previous = self._last_result
        diagnostics = SimpleNamespace(
            selected_clip_path=str(previous.diagnostics.selected_clip_path),
            selected_frame=int(previous.diagnostics.selected_frame),
            terrain_safety_override=False,
            step_time_ns=elapsed_ns,
            search_time_ns=None,
        )
        return SimpleNamespace(
            joint_position=previous.joint_position.clone(),
            joint_velocity=torch.zeros_like(previous.joint_velocity),
            root_position_world=previous.root_position_world.clone(),
            root_orientation_world_wxyz=(
                previous.root_orientation_world_wxyz.clone()
            ),
            root_linear_velocity_world=torch.zeros(3, device=self.database.device),
            diagnostics=diagnostics,
        )

    def prepare_step(
        self,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
        *,
        dt: float = 0.02,
    ) -> _PreparedSkillStep:
        if self._pending is not None:
            raise ContractError("terrain skill prepared step is uncommitted")
        if self._last_result is None:
            raise ContractError("reset must be called before prepare_step")
        start_ns = time.perf_counter_ns()
        requested_velocity = torch.tensor(
            velocity_world_xy, dtype=torch.float32, device=self.database.device
        )
        requested_heading = torch.tensor(
            heading_world_yaw, dtype=torch.float32, device=self.database.device
        )
        shaped = predict_command_trajectory(
            self._last_result.root_position_world[:2],
            self._shaped_velocity,
            self._shaped_heading,
            requested_velocity,
            requested_heading,
            config=self.config,
        )
        command = (
            (float(velocity_world_xy[0]), float(velocity_world_xy[1])),
            float(heading_world_yaw),
        )
        skill_state = self._skill_state
        command_changed = self._last_command is not None and command != self._last_command
        replan_requested = self._replan_pending or command_changed
        can_switch = (
            skill_state is None
            or skill_state.next_source_frame >= skill_state.skill.interval.playback_stop
            or (
                replan_requested
                and skill_state.next_source_frame > skill_state.selected_entry_frame
                and can_interrupt_skill(
                    skill_state.skill, skill_state.next_source_frame - 1
                )
            )
        )
        switch_attempted = False
        if can_switch:
            switch_attempted = True
            replacement = self._try_start_skill(
                velocity_world_xy, heading_world_yaw, shaped
            )
            if replacement is not None:
                skill_state = replacement
        if (
            skill_state is not None
            and skill_state.next_source_frame < skill_state.skill.interval.playback_stop
        ):
            step = advance_skill(skill_state)
            result = self._skill_result(step, time.perf_counter_ns() - start_ns)
            prepared = _PreparedSkillStep(
                result, "skill", step.state, shaped.velocity_world_xy,
                shaped.heading_world_yaw, command,
                replan_requested and not switch_attempted,
            )
        elif self._skill_state is None:
            base_prepared = self.base.prepare_step(
                velocity_world_xy, heading_world_yaw, dt=dt
            )
            prepared = _PreparedSkillStep(
                base_prepared.result,
                "base",
                base_prepared,
                shaped.velocity_world_xy,
                shaped.heading_world_yaw,
                command,
                replan_requested and not switch_attempted,
            )
        else:
            prepared = _PreparedSkillStep(
                self._hold_result(time.perf_counter_ns() - start_ns),
                "hold",
                self._skill_state,
                shaped.velocity_world_xy,
                shaped.heading_world_yaw,
                command,
                replan_requested and not switch_attempted,
            )
        self._pending = prepared
        return prepared

    def commit(self, prepared: _PreparedSkillStep):
        if prepared is not self._pending:
            raise ContractError("terrain skill prepared step is foreign or stale")
        old_feet = self._feet
        old_root = self._last_result.root_position_world
        if prepared.kind == "base":
            result = self.base.commit(prepared.payload)
        else:
            result = prepared.result
            if prepared.kind == "skill":
                self._skill_state = prepared.payload
        self._last_result = result
        self._last_command = prepared.command
        self._replan_pending = prepared.replan_pending
        self._shaped_velocity = prepared.shaped_velocity
        self._shaped_heading = prepared.shaped_heading
        self._feet = self._feet_for_result(result)
        self._foot_velocity = (self._feet - old_feet) / self.config.dt
        self._root_velocity = (
            result.root_position_world - old_root
        ) / self.config.dt
        self._pending = None
        return result


def run_resolved_skill_matrix(
    resolved,
    *,
    g1_xml: str,
    routes: Sequence[OmniRoute],
    terrain_tolerance_m: float = 0.06,
):
    """Run the existing renderer-independent route harness with skill MM."""

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
    inventory = build_terrain_skill_inventory(resolved.dataset, base.database)
    foot_kinematics = MujocoG1FootKinematics(g1_xml)
    matcher = TerrainSkillMatcher(
        base_matcher=base,
        inventory=inventory,
        dataset=resolved.dataset,
        query_terrain=resolved.measurement_extension,
        foot_kinematics=foot_kinematics,
        config=config,
        terrain_tolerance_m=terrain_tolerance_m,
    )
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
    identity = hashlib.sha256(
        (resolved.base_config_sha256 + ":php-skills-v1").encode()
    ).hexdigest()
    return run_omni_matrix(
        routes=tuple(routes),
        stair_frame=stair_frame,
        matcher_factory=lambda _route: matcher,
        kinematics=kinematics,
        terrain_sampler=terrain_sampler,
        dataset_identity=resolved.dataset.manifest_sha256,
        config_identity=identity,
        reset_root_position_world_xy=resolved_stair_reset_position(resolved),
    )
