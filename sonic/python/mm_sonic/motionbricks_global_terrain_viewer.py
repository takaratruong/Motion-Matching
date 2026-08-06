"""Browser/Switch viewer for MotionBricks flat motion plus global courses.

This is a deliberately privileged, purely kinematic ceiling.  Far from the
stair, the official MotionBricks G1 generator receives ordinary robot-local
two-stick commands.  At the best compatible prevalidated entry portal it
commits to an exact-mesh-audited flat->stair->flat course, then seeds
MotionBricks from the last four authored poses so joystick control resumes on
the landing.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import socket
import sys
import time

import numpy as np

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .generate_motionbricks_terrain_transitions import _load_demo
from .gear_action import (
    isaaclab_to_mujoco_joint_vector,
    mujoco_to_isaaclab_joint_vector,
)
from .motionbricks_terrain_portal import (
    MotionBricksTerrainCourse,
    PortalCapture,
    TerrainCoursePlayback,
    _yaw_wxyz,
    select_terrain_seam_portal,
    select_terrain_portal,
)
from .privileged_terrain_matcher import DEFAULT_STAIRS_ARCHIVE
from .terrain_interactive_viewer import (
    _append_overlay_to_scene,
    _configure_terrain_browser_ui,
)
from .terrain_oracle.math3d import RigidTransform
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.render_media import _build_scene_model
from .terrain_oracle.source_grail import _load_usd_mesh
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


DEFAULT_MOTIONBRICKS_ROOT = Path(
    "/move/u/bodow/Projects/reference-nvidia-groot-wbc/motionbricks"
)
DEFAULT_COURSE = Path(
    "/move/u/bodow/Projects/Motion-Matching-takara-corpus/"
    "artifacts/global_scene_terrain/"
    "motionbricks_course_target26_pos45_v1/"
    "reused_composition_v10_automatic_safe_phase/motion.npz"
)
PORTAL_ENTRY_LEAD_TIME_S = 0.80
FLAT_TERRAIN_GUARD_LOOKAHEAD_M = 0.32
FLAT_SUPPORT_TOLERANCE_M = 0.018
LANDING_SEAM_BLEND_CLEARANCE_M = 0.005
MAXIMUM_SUPPORT_FOOT_CLEARANCE_M = 0.060
EXIT_FOOT_LOCK_ACTIVATION_CLEARANCE_M = 0.040
EXIT_FOOT_LOCK_RELEASE_CLEARANCE_M = 0.025
EXIT_FOOT_LOCK_RELEASE_CONFIRMATION_FRAMES = 2
EXIT_FOOT_LOCK_RELEASE_FRAMES = 10
EXIT_FOOT_LOCK_MAXIMUM_JOINT_STEP_RAD = 0.22
EXIT_FOOT_LOCK_MAXIMUM_TARGET_ERROR_M = 0.015
EXIT_FOOT_LOCK_MAXIMUM_DURATION_FRAMES = 45
COMMAND_BUFFER_MOVING_SPEED_MPS = 0.06
COMMAND_BUFFER_DIRECTION_COSINE = 0.50
COMMAND_BUFFER_VELOCITY_DELTA_MPS = 0.22
COMMAND_BUFFER_FACING_DELTA_RAD = math.radians(28.0)


@dataclass(frozen=True)
class MotionBricksCommandSubmission:
    generated: bool
    invalidated: bool
    invalidation_reasons: tuple[str, ...]
    discarded_conditioning_frame_count: int


class MotionBricksCommandBufferInvalidator:
    """Invalidate a generated future when the operator changes intent.

    MotionBricks normally replans only after part of its current generated
    buffer has played.  That is desirable for small joystick jitter, but a
    stop/start, reversal, or large steering change must not keep playing the
    old trajectory.  The reference command here is the command that actually
    generated the live buffer, rather than merely the command seen on the
    previous controller tick.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._velocity_world_xy: np.ndarray | None = None
        self._facing_yaw_world: float | None = None
        self._mode_name: str | None = None

    def observe_generated(
        self,
        *,
        velocity_world_xy: object,
        facing_yaw_world: float,
        mode_name: str,
    ) -> None:
        velocity = np.asarray(velocity_world_xy, dtype=np.float64)
        if velocity.shape != (2,) or not np.isfinite(velocity).all():
            raise ValueError("MotionBricks command velocity must be finite xy")
        if not math.isfinite(float(facing_yaw_world)):
            raise ValueError("MotionBricks command facing must be finite")
        self._velocity_world_xy = velocity.copy()
        self._facing_yaw_world = float(facing_yaw_world)
        self._mode_name = str(mode_name)

    def invalidation_reasons(
        self,
        *,
        velocity_world_xy: object,
        facing_yaw_world: float,
        mode_name: str,
    ) -> tuple[str, ...]:
        velocity = np.asarray(velocity_world_xy, dtype=np.float64)
        if velocity.shape != (2,) or not np.isfinite(velocity).all():
            raise ValueError("MotionBricks command velocity must be finite xy")
        facing = float(facing_yaw_world)
        if not math.isfinite(facing):
            raise ValueError("MotionBricks command facing must be finite")
        if self._velocity_world_xy is None:
            return ("uninitialized",)

        previous = self._velocity_world_xy
        previous_speed = float(np.linalg.norm(previous))
        speed = float(np.linalg.norm(velocity))
        previous_moving = previous_speed >= COMMAND_BUFFER_MOVING_SPEED_MPS
        moving = speed >= COMMAND_BUFFER_MOVING_SPEED_MPS
        reasons: list[str] = []
        if str(mode_name) != self._mode_name:
            reasons.append("mode_change")
        if moving != previous_moving:
            reasons.append("start_stop")
        elif moving and previous_moving:
            cosine = float(np.dot(previous, velocity) / (previous_speed * speed))
            if cosine <= COMMAND_BUFFER_DIRECTION_COSINE:
                reasons.append("direction_change")
            elif float(np.linalg.norm(velocity - previous)) >= (
                COMMAND_BUFFER_VELOCITY_DELTA_MPS
            ):
                reasons.append("velocity_change")
        if abs(
            math.remainder(
                facing - float(self._facing_yaw_world), 2.0 * math.pi
            )
        ) >= COMMAND_BUFFER_FACING_DELTA_RAD:
            reasons.append("facing_change")
        return tuple(reasons)


def _catalog_course_paths(catalog_path: Path, target_clip_index: int) -> tuple[Path, ...]:
    """Return every authored and generated direction available for one mesh."""

    catalog = json.loads(catalog_path.expanduser().resolve().read_text())
    paths = []
    for route in catalog.get("routes", ()):
        if int(route["target_clip_index"]) != int(target_clip_index):
            continue
        paths.extend(
            Path(entry["motion_path"]).expanduser().resolve()
            for entry in route["entry_families"]
        )
    unique = tuple(dict.fromkeys(paths))
    if not unique:
        raise ValueError(
            f"terrain catalog has no courses for target {target_clip_index}"
        )
    return unique


def _route_manifest(
    manifest_path: Path,
) -> tuple[
    Path,
    tuple[float, float, float],
    tuple[float, float, float, float],
    tuple[Path, ...],
    tuple[tuple[float, float], ...],
]:
    """Load one compiled multi-event terrain scene for the browser viewer."""

    payload = json.loads(manifest_path.expanduser().resolve().read_text())
    if (
        payload.get("schema") != "generic-terrain-portal-manifest/v1"
        or payload.get("status") != "accepted"
    ):
        raise ValueError("terrain route manifest is not an accepted portal bundle")
    courses = tuple(
        Path(value).expanduser().resolve()
        for value in payload.get("course_motions", ())
    )
    if not courses:
        raise ValueError("terrain route manifest has no portal courses")
    events = payload.get("events", ())
    if not isinstance(events, list) or len(events) != len(courses):
        raise ValueError("terrain route manifest has no per-course event axes")
    lateral_axes = []
    for event in events:
        start = np.asarray(event["start_xy"], dtype=np.float64)
        end = np.asarray(event["end_xy"], dtype=np.float64)
        direction = end - start
        direction /= max(float(np.linalg.norm(direction)), 1.0e-8)
        lateral_axes.append((-float(direction[1]), float(direction[0])))
    position = tuple(float(value) for value in payload["terrain_position_world"])
    quaternion = tuple(
        float(value)
        for value in payload["terrain_quaternion_world_from_usd_wxyz"]
    )
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError("terrain route manifest has an invalid scene transform")
    return (
        Path(str(payload["terrain_usd"])).expanduser().resolve(),
        position,
        quaternion,
        courses,
        tuple(lateral_axes),
    )


def _rotate_xy(value: object, yaw: float) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ),
        dtype=np.float64,
    )


def _configure_ui(browser_control: object) -> None:
    _configure_terrain_browser_ui(browser_control)
    replacements = {
        "G1 terrain motion matching · clean kinematics":
            "G1 MotionBricks · global terrain portal",
        "Waiting for first clean-kinematic G1 frame":
            "Loading MotionBricks and the audited terrain course",
        "the kinematic player stops at the next double support":
            "the committed stair route completes before joystick control resumes",
        "MM command [": "MotionBricks/terrain command [",
    }
    template = str(browser_control._HTML_TEMPLATE)
    for source, target in replacements.items():
        template = template.replace(source, target)
    browser_control._HTML_TEMPLATE = template


def _command_targets(
    command: object,
    *,
    current_yaw: float,
    desired_facing_yaw: float,
    dt_s: float,
) -> tuple[np.ndarray, float, float, str]:
    local = np.asarray(
        (float(command.travel_forward), float(command.travel_left)),
        dtype=np.float64,
    )
    magnitude = float(np.linalg.norm(local))
    direction = local / magnitude if magnitude > 1.0e-6 else np.zeros(2)
    magnitude = float(np.clip(magnitude * float(command.speed_scale), 0.0, 1.0))
    if bool(command.facing_active):
        desired_facing_yaw = math.remainder(
            current_yaw
            + math.atan2(
                float(command.facing_left), float(command.facing_forward)
            ),
            2.0 * math.pi,
        )
    elif abs(float(command.yaw_left)) > 1.0e-6:
        desired_facing_yaw = math.remainder(
            desired_facing_yaw
            + float(command.yaw_left)
            * float(command.turn_scale)
            * 1.20
            * dt_s,
            2.0 * math.pi,
        )
    movement_world = _rotate_xy(direction, current_yaw)
    if magnitude <= 0.06:
        return np.zeros(2), desired_facing_yaw, 0.0, "idle"
    if magnitude < 0.48:
        speed = 0.18 + 0.42 * magnitude
        return movement_world * speed, desired_facing_yaw, speed, "slow_walk"
    speed = 0.22 + 0.68 * magnitude
    return movement_world * speed, desired_facing_yaw, speed, "walk"


def _submit_motionbricks(
    full_agent: object,
    controller: object,
    *,
    context_qpos: object,
    velocity_world_xy: object,
    facing_yaw_world: float,
    mode_name: str,
    force: bool,
    random_seed: int = 0,
    support_height_world: float = 0.0,
) -> bool:
    import torch
    from motionbricks.motion_backbone.demo.clips import clip_holder_G1

    velocity = np.asarray(velocity_world_xy, dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    direction = velocity / speed if speed > 1.0e-6 else np.zeros(2)
    mode_index = list(clip_holder_G1.CLIPS.keys()).index(mode_name)
    if isinstance(context_qpos, torch.Tensor):
        context = context_qpos.detach().clone().to(dtype=torch.float32)
    else:
        context = torch.as_tensor(
            np.asarray(context_qpos, dtype=np.float32), dtype=torch.float32
        )
    if context.ndim == 2:
        context = context[None]
    support_height = float(support_height_world)
    if not math.isfinite(support_height):
        raise ValueError("MotionBricks support height must be finite")
    # MotionBricks is trained with its locomotion floor at z=0.  Canonicalize
    # a raised-platform context to that floor, then restore the same support
    # height to its generated root.  Passing absolute elevated z directly
    # makes the learned inbetween collapse the pelvis by exactly the platform
    # height while it tries to return to its training floor.
    context[..., 2] -= support_height
    signals = {
        "movement_direction": torch.as_tensor(
            (direction[0], direction[1], 0.0), dtype=torch.float32
        ).view(1, 3),
        "facing_direction": torch.as_tensor(
            (
                math.cos(float(facing_yaw_world)),
                math.sin(float(facing_yaw_world)),
                0.0,
            ),
            dtype=torch.float32,
        ).view(1, 3),
        "mode": torch.tensor([[mode_index]], dtype=torch.long),
        "target_vel": torch.tensor([speed], dtype=torch.float32),
        "random_seed": torch.tensor([int(random_seed)], dtype=torch.long),
        "allowed_pred_num_tokens": controller.get_default_allowed_pred_num_tokens(
            mode_index
        ),
        "context_mujoco_qpos": context,
    }
    previous_qpos = full_agent.frames.get("mujoco_qpos")
    with torch.no_grad():
        full_agent.generate_new_frames(
            signals,
            controller.get_controller_dt() * 2.0,
            force_generation=force,
        )
        generated = full_agent.frames["mujoco_qpos"] is not previous_qpos
        if abs(support_height) > 1.0e-8 and generated:
            full_agent.frames["mujoco_qpos"][..., 2] += support_height
    return bool(generated)


def _verified_motionbricks_context(history: object) -> np.ndarray:
    """Return four chronological, already-published poses for a safe replan."""

    values = tuple(np.asarray(value, dtype=np.float64) for value in history)
    if not values:
        raise ValueError("MotionBricks command invalidation has no safe history")
    context = np.stack(values)
    if context.ndim != 2 or context.shape[1] != 36:
        raise ValueError("MotionBricks safe history must contain qpos[36]")
    if len(context) > 4:
        context = context[-4:]
    if len(context) < 4:
        context = np.concatenate(
            (np.repeat(context[:1], 4 - len(context), axis=0), context),
            axis=0,
        )
    return context


def _submit_responsive_motionbricks(
    full_agent: object,
    controller: object,
    *,
    command_buffer: MotionBricksCommandBufferInvalidator,
    verified_history: object,
    velocity_world_xy: object,
    facing_yaw_world: float,
    mode_name: str,
    random_seed: int = 0,
    support_height_world: float = 0.0,
) -> MotionBricksCommandSubmission:
    """Submit a command, replacing a materially stale future when necessary."""

    reasons = command_buffer.invalidation_reasons(
        velocity_world_xy=velocity_world_xy,
        facing_yaw_world=facing_yaw_world,
        mode_name=mode_name,
    )
    discarded = 0
    if reasons:
        context = _verified_motionbricks_context(verified_history)
        generated = _submit_motionbricks(
            full_agent,
            controller,
            context_qpos=context,
            velocity_world_xy=velocity_world_xy,
            facing_yaw_world=facing_yaw_world,
            mode_name=mode_name,
            force=True,
            random_seed=random_seed,
            support_height_world=support_height_world,
        )
        # MotionBricks includes the four supplied conditioning poses at the
        # front of a forced batch.  They have already been published; consume
        # them so the next frame is the first new future pose, not a rewind.
        if generated:
            for _ in range(len(context)):
                full_agent.get_next_frame()
            discarded = len(context)
    else:
        generated = _submit_motionbricks(
            full_agent,
            controller,
            context_qpos=full_agent.get_context_mujoco_qpos(),
            velocity_world_xy=velocity_world_xy,
            facing_yaw_world=facing_yaw_world,
            mode_name=mode_name,
            force=False,
            random_seed=random_seed,
            support_height_world=support_height_world,
        )
    if generated:
        command_buffer.observe_generated(
            velocity_world_xy=velocity_world_xy,
            facing_yaw_world=facing_yaw_world,
            mode_name=mode_name,
        )
    return MotionBricksCommandSubmission(
        generated=bool(generated),
        invalidated=bool(reasons),
        invalidation_reasons=reasons,
        discarded_conditioning_frame_count=discarded,
    )


def _terrain_surface_height(
    terrain: TerrainMeshIndex,
    xy: object,
    *,
    ray_origin_z: float,
) -> float:
    point = np.asarray(xy, dtype=np.float64)
    hit = terrain.raycast(
        np.asarray((point[0], point[1], ray_origin_z)),
        np.asarray((0.0, 0.0, -1.0)),
    )
    return 0.0 if hit is None else float(hit.position_world[2])


def _guard_flat_velocity(
    terrain: TerrainMeshIndex,
    root_xyz: object,
    velocity_world_xy: object,
    *,
    ray_origin_z: float,
    lookahead_distance_m: float = FLAT_TERRAIN_GUARD_LOOKAHEAD_M,
    maximum_support_change_m: float = 0.018,
) -> tuple[np.ndarray, bool]:
    root = np.asarray(root_xyz, dtype=np.float64)
    velocity = np.asarray(velocity_world_xy, dtype=np.float64)
    if float(np.linalg.norm(velocity)) < 1.0e-6:
        return velocity, False
    current_height = _terrain_surface_height(
        terrain, root[:2], ray_origin_z=ray_origin_z
    )
    direction = velocity / float(np.linalg.norm(velocity))
    distances = np.linspace(
        min(0.08, float(lookahead_distance_m)),
        float(lookahead_distance_m),
        6,
    )
    future_heights = np.asarray(
        [
            _terrain_surface_height(
                terrain,
                root[:2] + distance * direction,
                ray_origin_z=ray_origin_z,
            )
            for distance in distances
        ],
        dtype=np.float64,
    )
    unsafe = bool(
        np.max(np.abs(future_heights - current_height))
        > float(maximum_support_change_m)
    )
    return (np.zeros(2, dtype=np.float64) if unsafe else velocity), unsafe


def _flat_pose_support_error(
    qpos: object,
    *,
    sole_adapter: _G1FootfallAdapter,
    terrain: TerrainMeshIndex,
    ray_origin_z: float,
    support_height_world: float,
) -> float:
    """Measure whether a purportedly flat pose has entered non-flat terrain.

    The velocity lookahead is intentionally only an early warning.  A decoded
    MotionBricks frame can move obliquely or farther than its commanded root
    velocity predicts, so the actual root and both complete sole footprints
    are checked before the frame is published.
    """

    value = np.asarray(qpos, dtype=np.float64)
    if value.shape != (36,) or not np.isfinite(value).all():
        raise ValueError("flat support validation expects one finite qpos[36]")
    soles = sole_adapter.sole_support_points_for_pose(
        root_position=value[:3],
        root_quaternion_wxyz=value[3:7],
        joints=mujoco_to_isaaclab_joint_vector(value[7:]),
    )
    sample_xy = [value[:2]]
    sample_xy.extend(
        point[:2]
        for points in soles
        for point in np.asarray(points, dtype=np.float64)
    )
    errors = [
        abs(
            _terrain_surface_height(
                terrain, xy, ray_origin_z=ray_origin_z
            )
            - float(support_height_world)
        )
        for xy in sample_xy
    ]
    return float(max(errors, default=0.0))


def _minimum_sole_clearance_m(
    qpos: object,
    *,
    sole_adapter: _G1FootfallAdapter,
    terrain: TerrainMeshIndex,
    ray_origin_z: float,
) -> float:
    """Return the closest sole-point height above the exact terrain mesh.

    Collision-only audits cannot detect a character hovering above a lower
    surface.  A walking pose should always have at least one sole point near
    support, even while the other foot is in swing.
    """

    value = np.asarray(qpos, dtype=np.float64)
    if value.shape != (36,) or not np.isfinite(value).all():
        raise ValueError("sole-clearance validation expects one finite qpos[36]")
    return float(
        min(
            _per_foot_minimum_sole_clearance_m(
                value,
                sole_adapter=sole_adapter,
                terrain=terrain,
                ray_origin_z=ray_origin_z,
            )
        )
    )


def _per_foot_minimum_sole_clearance_m(
    qpos: object,
    *,
    sole_adapter: _G1FootfallAdapter,
    terrain: TerrainMeshIndex,
    ray_origin_z: float,
) -> tuple[float, float]:
    """Return exact-mesh minimum sole clearance for left and right feet."""

    value = np.asarray(qpos, dtype=np.float64)
    if value.shape != (36,) or not np.isfinite(value).all():
        raise ValueError("per-foot clearance validation expects one finite qpos[36]")
    soles = sole_adapter.sole_support_points_for_pose(
        root_position=value[:3],
        root_quaternion_wxyz=value[3:7],
        joints=mujoco_to_isaaclab_joint_vector(value[7:]),
    )
    clearances: list[float] = []
    for points in soles:
        per_point = [
            float(point[2])
            - _terrain_surface_height(
                terrain, point[:2], ray_origin_z=ray_origin_z
            )
            for point in np.asarray(points, dtype=np.float64)
        ]
        if not per_point:
            raise ValueError("sole-clearance validation found no support points")
        clearances.append(float(min(per_point)))
    if len(clearances) != 2:
        raise ValueError("sole-clearance validation expects exactly two feet")
    return clearances[0], clearances[1]


@dataclass(frozen=True)
class ExitFootLockFrame:
    """One course-to-live-gait handoff result and its mechanical evidence."""

    qpos: np.ndarray
    armed: bool
    active: bool
    triggered: bool
    released: bool
    support_foot: int | None
    raw_minimum_clearance_m: float
    corrected_minimum_clearance_m: float
    joint_correction_rad: float
    foot_target_error_m: float
    maximum_joint_step_rad: float


class MotionBricksExitFootLock:
    """Preserve support while MotionBricks changes phase after a course.

    A portal is an authored trajectory and its last frame may have only one
    planted foot.  MotionBricks can decode a different first live gait phase;
    merely blending joints can therefore put both soles in the air.  This
    handoff latches the planted sole in world space, solves only the two legs
    against explicit sole targets, and fades the latch only after the raw live
    gait has established the opposite support foot.
    """

    def __init__(
        self,
        *,
        sole_adapter: _G1FootfallAdapter,
        ik_adapter: _G1FootfallAdapter,
        terrain: TerrainMeshIndex,
        ray_origin_z: float,
        activation_clearance_m: float = EXIT_FOOT_LOCK_ACTIVATION_CLEARANCE_M,
        release_clearance_m: float = EXIT_FOOT_LOCK_RELEASE_CLEARANCE_M,
        release_confirmation_frames: int = (
            EXIT_FOOT_LOCK_RELEASE_CONFIRMATION_FRAMES
        ),
        release_frames: int = EXIT_FOOT_LOCK_RELEASE_FRAMES,
        maximum_joint_step_rad: float = EXIT_FOOT_LOCK_MAXIMUM_JOINT_STEP_RAD,
        maximum_target_error_m: float = EXIT_FOOT_LOCK_MAXIMUM_TARGET_ERROR_M,
        maximum_duration_frames: int = EXIT_FOOT_LOCK_MAXIMUM_DURATION_FRAMES,
    ) -> None:
        values = (
            activation_clearance_m,
            release_clearance_m,
            maximum_joint_step_rad,
            maximum_target_error_m,
        )
        if (
            not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values)
            or int(release_confirmation_frames) <= 0
            or int(release_frames) <= 0
            or int(maximum_duration_frames) <= 0
        ):
            raise ValueError("exit foot-lock parameters must be positive")
        self.sole_adapter = sole_adapter
        self.ik_adapter = ik_adapter
        self.terrain = terrain
        self.ray_origin_z = float(ray_origin_z)
        self.activation_clearance_m = float(activation_clearance_m)
        self.release_clearance_m = float(release_clearance_m)
        self.release_confirmation_frames = int(release_confirmation_frames)
        self.release_frames = int(release_frames)
        self.maximum_joint_step_rad = float(maximum_joint_step_rad)
        self.maximum_target_error_m = float(maximum_target_error_m)
        self.maximum_duration_frames = int(maximum_duration_frames)
        self.reset()

    def reset(self) -> None:
        self._armed = False
        self._active = False
        self._support_foot: int | None = None
        self._anchor_centres: np.ndarray | None = None
        self._previous_joints: np.ndarray | None = None
        self._grounded_confirmation = 0
        self._flat_confirmation = 0
        self._release_frame = -1
        self._active_frames = 0
        self._support_height_world = 0.0

    @property
    def armed(self) -> bool:
        return bool(self._armed)

    @property
    def active(self) -> bool:
        # External callers need to bypass the flat-only guard for the entire
        # terrain-boundary handoff, including supported pass-through frames.
        # ``_active`` itself denotes only whether IK correction is engaged.
        return bool(self._armed)

    def arm(self, course_exit_qpos: object) -> dict[str, object]:
        """Latch the most grounded sole of a verified course endpoint."""

        value = np.asarray(course_exit_qpos, dtype=np.float64)
        if value.shape != (36,) or not np.isfinite(value).all():
            raise ValueError("exit foot lock expects one finite course qpos[36]")
        clearances = _per_foot_minimum_sole_clearance_m(
            value,
            sole_adapter=self.sole_adapter,
            terrain=self.terrain,
            ray_origin_z=self.ray_origin_z,
        )
        support_foot = int(np.argmin(clearances))
        if clearances[support_foot] > self.activation_clearance_m:
            raise ValueError(
                "cannot arm exit foot lock from an unsupported course endpoint: "
                f"minimum_clearance={clearances[support_foot]:.4f}m"
            )
        centres = self.sole_adapter.sole_positions_for_pose(
            root_position=value[:3],
            root_quaternion_wxyz=value[3:7],
            joints=mujoco_to_isaaclab_joint_vector(value[7:]),
        )
        self.reset()
        self._armed = True
        self._support_foot = support_foot
        self._anchor_centres = np.asarray(
            centres[support_foot], dtype=np.float64
        ).copy()
        self._previous_joints = mujoco_to_isaaclab_joint_vector(value[7:])
        self._support_height_world = _terrain_surface_height(
            self.terrain, value[:2], ray_origin_z=self.ray_origin_z
        )
        return {
            "support_foot": support_foot,
            "left_clearance_m": float(clearances[0]),
            "right_clearance_m": float(clearances[1]),
        }

    @staticmethod
    def _smoothstep(value: float) -> float:
        amount = float(np.clip(value, 0.0, 1.0))
        return amount * amount * (3.0 - 2.0 * amount)

    def apply(self, proposed_qpos: object) -> ExitFootLockFrame:
        """Validate or contact-correct one newly decoded flat frame."""

        value = np.asarray(proposed_qpos, dtype=np.float64).copy()
        if value.shape != (36,) or not np.isfinite(value).all():
            raise ValueError("exit foot lock expects one finite proposed qpos[36]")
        raw_clearances = _per_foot_minimum_sole_clearance_m(
            value,
            sole_adapter=self.sole_adapter,
            terrain=self.terrain,
            ray_origin_z=self.ray_origin_z,
        )
        raw_minimum = float(min(raw_clearances))
        support_foot = self._support_foot
        if not self._armed:
            return ExitFootLockFrame(
                qpos=value,
                armed=False,
                active=False,
                triggered=False,
                released=False,
                support_foot=None,
                raw_minimum_clearance_m=raw_minimum,
                corrected_minimum_clearance_m=raw_minimum,
                joint_correction_rad=0.0,
                foot_target_error_m=0.0,
                maximum_joint_step_rad=0.0,
            )
        self._active_frames += 1
        if self._active_frames > self.maximum_duration_frames:
            raise RuntimeError(
                "MotionBricks failed to establish support before the exit "
                f"foot-lock limit ({self.maximum_duration_frames} frames)"
            )
        flat_plane_error = _flat_pose_support_error(
            value,
            sole_adapter=self.sole_adapter,
            terrain=self.terrain,
            ray_origin_z=self.ray_origin_z,
            support_height_world=self._support_height_world,
        )
        if not self._active and raw_minimum <= self.activation_clearance_m:
            if flat_plane_error <= FLAT_SUPPORT_TOLERANCE_M:
                self._flat_confirmation += 1
            else:
                self._flat_confirmation = 0
            if self._flat_confirmation >= self.release_confirmation_frames:
                self.reset()
                return ExitFootLockFrame(
                    qpos=value,
                    armed=False,
                    active=False,
                    triggered=False,
                    released=True,
                    support_foot=support_foot,
                    raw_minimum_clearance_m=raw_minimum,
                    corrected_minimum_clearance_m=raw_minimum,
                    joint_correction_rad=0.0,
                    foot_target_error_m=0.0,
                    maximum_joint_step_rad=0.0,
                )
            return ExitFootLockFrame(
                qpos=value,
                armed=True,
                active=True,
                triggered=False,
                released=False,
                support_foot=support_foot,
                raw_minimum_clearance_m=raw_minimum,
                corrected_minimum_clearance_m=raw_minimum,
                joint_correction_rad=0.0,
                foot_target_error_m=0.0,
                maximum_joint_step_rad=0.0,
            )

        triggered = not self._active
        self._active = True
        if (
            support_foot is None
            or self._anchor_centres is None
            or self._previous_joints is None
        ):
            raise RuntimeError("exit foot lock is active without an anchor")

        incoming_foot = 1 - support_foot
        if raw_clearances[incoming_foot] <= self.release_clearance_m:
            self._grounded_confirmation += 1
        else:
            self._grounded_confirmation = 0
        if (
            self._release_frame < 0
            and self._grounded_confirmation >= self.release_confirmation_frames
        ):
            self._release_frame = 0

        authored_joints = mujoco_to_isaaclab_joint_vector(value[7:])
        raw_centres = self.sole_adapter.sole_positions_for_pose(
            root_position=value[:3],
            root_quaternion_wxyz=value[3:7],
            joints=authored_joints,
        )
        targets = [raw_centres[0].copy(), raw_centres[1].copy()]
        release_amount = 0.0
        if self._release_frame >= 0:
            release_amount = self._smoothstep(
                (self._release_frame + 1) / self.release_frames
            )
        targets[support_foot] = (
            (1.0 - release_amount) * self._anchor_centres
            + release_amount * raw_centres[support_foot]
        )
        adapted_joints, correction, target_error = self.ik_adapter.adapt_to_targets(
            root_position=value[:3],
            root_quaternion_wxyz=value[3:7],
            authored_joints=authored_joints,
            sole_targets_world=targets,
            initial_joints=self._previous_joints,
            continuity_joints=self._previous_joints,
            continuity_feet=(True, True),
            maximum_continuity_joint_step_rad=(
                self.maximum_joint_step_rad - 1.0e-6
            ),
        )
        maximum_step = float(
            np.max(np.abs(adapted_joints - self._previous_joints))
        )
        corrected = value.copy()
        corrected[7:] = isaaclab_to_mujoco_joint_vector(adapted_joints)
        corrected_minimum = _minimum_sole_clearance_m(
            corrected,
            sole_adapter=self.sole_adapter,
            terrain=self.terrain,
            ray_origin_z=self.ray_origin_z,
        )
        if maximum_step > self.maximum_joint_step_rad + 1.0e-8:
            raise RuntimeError(
                "exit foot lock exceeded temporal joint-step bound: "
                f"frame={self._active_frames} "
                f"{maximum_step:.4f}rad > {self.maximum_joint_step_rad:.4f}rad "
                f"raw_clearance={raw_minimum:.4f}m "
                f"correction={correction:.4f}rad"
            )
        if target_error > self.maximum_target_error_m:
            raise RuntimeError(
                "exit foot lock missed its support target: "
                f"{target_error:.4f}m > {self.maximum_target_error_m:.4f}m"
            )
        if corrected_minimum > self.activation_clearance_m:
            raise RuntimeError(
                "exit foot lock produced an unsupported frame: "
                f"minimum_clearance={corrected_minimum:.4f}m"
            )

        self._previous_joints = adapted_joints.copy()
        released = False
        if self._release_frame >= 0:
            self._release_frame += 1
            if self._release_frame >= self.release_frames:
                # IK has faded fully back to the authored live gait.  Keep the
                # boundary handoff armed until the complete footprint—not
                # merely one supported sole—has cleared onto the flat plane.
                self._active = False
                self._release_frame = -1
                self._grounded_confirmation = 0
                self._flat_confirmation = 0
        return ExitFootLockFrame(
            qpos=corrected,
            armed=self.armed,
            active=self.active,
            triggered=triggered,
            released=released,
            support_foot=support_foot,
            raw_minimum_clearance_m=raw_minimum,
            corrected_minimum_clearance_m=float(corrected_minimum),
            joint_correction_rad=float(correction),
            foot_target_error_m=float(target_error),
            maximum_joint_step_rad=maximum_step,
        )


def _flat_pose_matches_support(
    qpos: object,
    *,
    sole_adapter: _G1FootfallAdapter,
    terrain: TerrainMeshIndex,
    ray_origin_z: float,
    support_height_world: float,
    maximum_support_error_m: float = FLAT_SUPPORT_TOLERANCE_M,
) -> bool:
    """Return whether a flat-controller frame stays on its current plane."""

    return bool(
        _flat_pose_support_error(
            qpos,
            sole_adapter=sole_adapter,
            terrain=terrain,
            ray_origin_z=ray_origin_z,
            support_height_world=support_height_world,
        )
        <= float(maximum_support_error_m)
    )


def _laterally_registered_course(
    course: MotionBricksTerrainCourse,
    current_xy: object,
    *,
    entry_lead_time_s: float,
    maximum_lateral_shift_m: float = 3.0,
    lateral_axis_world_xy: object | None = None,
    registration_frame_index: int | None = None,
) -> tuple[MotionBricksTerrainCourse, float]:
    """Place an authored crossing in the operator's current terrain lane."""

    entry_frame = (
        course.entry_frame_index(entry_lead_time_s)
        if registration_frame_index is None
        else int(registration_frame_index)
    )
    if entry_frame < 0 or entry_frame > course.seam_indices[0]:
        raise ValueError("lane registration frame must not follow terrain entry")
    if lateral_axis_world_xy is None:
        direction = course.travel_direction_world_xy(entry_frame)
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    else:
        normal = np.asarray(lateral_axis_world_xy, dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-8)
    delta = (
        np.asarray(current_xy, dtype=np.float64)
        - course.root_position_world[entry_frame, :2]
    )
    requested_shift = float(np.dot(delta, normal))
    lateral_shift = float(
        np.clip(
            requested_shift,
            -float(maximum_lateral_shift_m),
            float(maximum_lateral_shift_m),
        )
    )
    if abs(lateral_shift) < 1.0e-6:
        return course, 0.0
    root = course.root_position_world.copy()
    root[:, :2] += lateral_shift * normal[None]
    return (
        MotionBricksTerrainCourse(
            path=course.path,
            fps=course.fps,
            root_position_world=root,
            root_quaternion_world_wxyz=(
                course.root_quaternion_world_wxyz
            ),
            joint_position_isaaclab=course.joint_position_isaaclab,
            seam_indices=course.seam_indices,
        ),
        lateral_shift,
    )


def _registration_matches_terrain(
    original: MotionBricksTerrainCourse,
    registered: MotionBricksTerrainCourse,
    *,
    terrain: TerrainMeshIndex,
    ray_origin_z: float,
    entry_frame: int,
    maximum_height_error_m: float = 0.003,
    lateral_axis_world_xy: object | None = None,
) -> bool:
    """Verify that a lane shift preserves the exact support-height profile."""

    frames = np.unique(
        np.linspace(
            int(entry_frame),
            original.frame_count - 1,
            min(64, original.frame_count - int(entry_frame)),
        ).round().astype(np.int64)
    )
    if lateral_axis_world_xy is None:
        direction = original.travel_direction_world_xy(entry_frame)
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    else:
        normal = np.asarray(lateral_axis_world_xy, dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-8)
    for frame in frames:
        for sole_offset in (-0.16, 0.0, 0.16):
            source_xy = (
                original.root_position_world[frame, :2]
                + sole_offset * normal
            )
            target_xy = (
                registered.root_position_world[frame, :2]
                + sole_offset * normal
            )
            source_hit = terrain.raycast(
                np.asarray((source_xy[0], source_xy[1], ray_origin_z)),
                np.asarray((0.0, 0.0, -1.0)),
            )
            target_hit = terrain.raycast(
                np.asarray((target_xy[0], target_xy[1], ray_origin_z)),
                np.asarray((0.0, 0.0, -1.0)),
            )
            if source_hit is None or target_hit is None:
                return False
            if (
                abs(
                    float(source_hit.position_world[2])
                    - float(target_hit.position_world[2])
                )
                > float(maximum_height_error_m)
            ):
                return False
    return True


def _project_live_root_above_support(
    qpos: np.ndarray,
    *,
    sole_adapter: _G1FootfallAdapter,
    terrain: object,
    ray_origin_z: float,
    sole_clearance_m: float = 0.003,
    maximum_lift_m: float = 0.06,
    support_height_world: float | None = None,
) -> tuple[np.ndarray, float]:
    """Apply the globally privileged game-style root-height projection."""

    value = np.asarray(qpos, dtype=np.float64).copy()
    soles = sole_adapter.sole_support_points_for_pose(
        root_position=value[:3],
        root_quaternion_wxyz=value[3:7],
        joints=mujoco_to_isaaclab_joint_vector(value[7:]),
    )
    required_lifts = [
        (
            _terrain_surface_height(
                terrain, point[:2], ray_origin_z=ray_origin_z
            )
            if support_height_world is None
            else float(support_height_world)
        )
        + float(sole_clearance_m)
        - float(point[2])
        for points in soles
        for point in np.asarray(points, dtype=np.float64)
    ]
    lift = float(
        np.clip(
            max(required_lifts, default=0.0),
            0.0,
            float(maximum_lift_m),
        )
    )
    value[2] += lift
    return value, lift


def _submit_phase_matched_course_exit(
    full_agent: object,
    controller: object,
    *,
    context_qpos: np.ndarray,
    velocity_world_xy: np.ndarray,
    facing_yaw_world: float,
    mode_name: str,
    support_height_world: float,
) -> dict[str, object]:
    """Seed the least-disruptive live gait phase after an authored portal."""

    import torch

    context = np.asarray(context_qpos, dtype=np.float64)
    if context.ndim != 2 or context.shape[1] != 36:
        raise ValueError("course exit context must have shape (T,36)")
    candidates: list[dict[str, float | int]] = []
    for seed in range(0, 64, 4):
        _submit_motionbricks(
            full_agent,
            controller,
            context_qpos=context,
            velocity_world_xy=velocity_world_xy,
            facing_yaw_world=facing_yaw_world,
            mode_name=mode_name,
            force=True,
            random_seed=seed,
            support_height_world=support_height_world,
        )
        generated_value = full_agent.frames["mujoco_qpos"]
        generated = (
            generated_value[0].detach().cpu().numpy()
            if isinstance(generated_value, torch.Tensor)
            else np.asarray(generated_value)[0]
        )
        future_start = len(context)
        horizon = min(len(generated) - future_start, 24)
        if horizon < 2:
            raise RuntimeError("MotionBricks exit has no future frames")
        review = np.asarray(
            generated[future_start : future_start + horizon], dtype=np.float64
        )
        root_drop = max(0.0, float(context[-1, 2] - np.min(review[:, 2])))
        double_knee = float(np.max(np.mean(review[:, (10, 16)], axis=1)))
        pose_gap = float(
            np.sqrt(np.mean((review[0, 7:] - context[-1, 7:]) ** 2))
        )
        joint_step = float(
            np.max(
                np.abs(
                    np.diff(
                        np.concatenate((context[-1:, 7:], review[:, 7:]), axis=0),
                        axis=0,
                    )
                )
            )
        )
        score = (
            4.0 * root_drop
            + 0.20 * double_knee
            + 0.50 * pose_gap
            + 0.20 * joint_step
        )
        candidates.append(
            {
                "seed": seed,
                "score": score,
                "root_drop_m": root_drop,
                "double_knee_peak_rad": double_knee,
                "first_pose_rmse_rad": pose_gap,
                "maximum_joint_step_rad": joint_step,
            }
        )
    selected = min(candidates, key=lambda value: float(value["score"]))
    _submit_motionbricks(
        full_agent,
        controller,
        context_qpos=context,
        velocity_world_xy=velocity_world_xy,
        facing_yaw_world=facing_yaw_world,
        mode_name=mode_name,
        force=True,
        random_seed=int(selected["seed"]),
        support_height_world=support_height_world,
    )
    # The decoded prefix is the conditioning history, not new motion.
    for _ in range(len(context)):
        full_agent.get_next_frame()
    return {
        "selected": selected,
        "candidate_count": len(candidates),
        "score_minimum": float(selected["score"]),
        "score_median": float(
            np.median([value["score"] for value in candidates])
        ),
        "score_maximum": float(max(value["score"] for value in candidates)),
        "discarded_conditioning_frame_count": len(context),
    }


def run(
    *,
    target_clip_index: int | None,
    course_paths: tuple[Path, ...],
    course_lateral_axes: tuple[tuple[float, float] | None, ...] | None,
    motionbricks_root: Path,
    stairs_archive: Path,
    terrain_usd: Path | None,
    terrain_position_world: tuple[float, float, float],
    terrain_quaternion_world_from_usd_wxyz: tuple[float, float, float, float],
    model_path: Path,
    host: str,
    port: int,
    token: str | None,
    wait_seconds: float,
) -> dict[str, object]:
    import mujoco
    import torch
    from diffusion_policy.inference import (
        sonic_interactive_controller as browser_control,
    )

    motionbricks_root = motionbricks_root.expanduser().resolve()
    for value in (motionbricks_root, motionbricks_root / "scripts"):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    _configure_ui(browser_control)
    courses = tuple(
        MotionBricksTerrainCourse.load(path) for path in course_paths
    )
    if not courses:
        raise ValueError("viewer needs at least one terrain course")
    if course_lateral_axes is None:
        lateral_axes: tuple[tuple[float, float] | None, ...] = (None,) * len(
            courses
        )
    else:
        lateral_axes = tuple(course_lateral_axes)
        if len(lateral_axes) != len(courses):
            raise ValueError(
                "course_lateral_axes must contain one entry per terrain course"
            )
    primary_course = courses[0]
    primary_course_qpos = primary_course.native_mujoco_qpos()

    print("[GLOBAL TERRAIN] Loading official MotionBricks checkpoints...", flush=True)
    full_agent, controller = _load_demo(motionbricks_root, seed=7)
    if terrain_usd is None:
        if target_clip_index is None:
            raise ValueError(
                "target_clip_index is required when terrain_usd is omitted"
            )
        import zarr

        archive = zarr.open_group(str(stairs_archive), mode="r")
        terrain_path = Path(
            str(archive["terrain_usd_path"][target_clip_index])
        )
        terrain_transform = RigidTransform(
            np.asarray(archive["terrain_position_env"][target_clip_index]),
            np.asarray(
                archive["terrain_rotation_env_wxyz"][target_clip_index]
            ),
        )
    else:
        terrain_path = terrain_usd.expanduser().resolve()
        terrain_transform = RigidTransform(
            np.asarray(terrain_position_world, dtype=np.float32),
            np.asarray(
                terrain_quaternion_world_from_usd_wxyz,
                dtype=np.float32,
            ),
        )
    terrain_mesh = _load_usd_mesh(
        terrain_path, source_asset_sha256="0" * 64
    )
    terrain_index = TerrainMeshIndex(terrain_mesh, terrain_transform)
    import zarr

    support_archive = zarr.open_group(str(stairs_archive), mode="r")
    sole_adapter = _G1FootfallAdapter(
        model_path,
        tuple(str(value) for value in support_archive["joint_names"][:]),
        maximum_joint_correction_rad=0.1,
    )
    spec = mujoco.MjSpec.from_file(str(model_path.expanduser().resolve()))
    model, visual_mesh_count = _build_scene_model(
        spec, terrain_mesh, terrain_transform
    )
    data = mujoco.MjData(model)
    ray_origin_z = float(np.max(terrain_index.vertices_world[:, 2]) + 2.0)
    exit_foot_lock_adapter = _G1FootfallAdapter(
        model_path,
        tuple(str(value) for value in support_archive["joint_names"][:]),
        maximum_joint_correction_rad=1.6,
        target_tolerance_m=5.0e-4,
        maximum_iterations=96,
        damping=0.008,
    )
    exit_foot_lock = MotionBricksExitFootLock(
        sole_adapter=sole_adapter,
        ik_adapter=exit_foot_lock_adapter,
        terrain=terrain_index,
        ray_origin_z=ray_origin_z,
    )
    course_support_clearance_maxima = tuple(
        float(
            max(
                _minimum_sole_clearance_m(
                    qpos,
                    sole_adapter=sole_adapter,
                    terrain=terrain_index,
                    ray_origin_z=ray_origin_z,
                )
                for qpos in course.native_mujoco_qpos()
            )
        )
        for course in courses
    )
    unsupported = [
        index
        for index, value in enumerate(course_support_clearance_maxima)
        if value > MAXIMUM_SUPPORT_FOOT_CLEARANCE_M
    ]
    if unsupported:
        raise ValueError(
            "terrain course bank contains sustained unsupported poses: "
            + ", ".join(
                f"course {index}={course_support_clearance_maxima[index]:.3f}m"
                for index in unsupported
            )
        )

    state = browser_control.InteractiveControllerState(
        token=token,
        deadman_timeout_s=0.40,
        max_stream_fps=20.0,
        task_dim=4,
        command_profile=browser_control.DIRECT_TWO_STICK_TASK4,
        max_speed_mps=0.90,
        max_yaw_rate_rad_s=1.20,
    )
    server = browser_control.InteractiveControllerServer(
        state, host=host, port=port
    )
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 960)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 540)
    renderer = mujoco.Renderer(model, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.distance = 3.1
    camera.azimuth = 135.0
    camera.elevation = -18.0

    current_qpos = primary_course_qpos[0].copy()
    flat_support_height = _terrain_surface_height(
        terrain_index, current_qpos[:2], ray_origin_z=ray_origin_z
    )
    desired_facing_yaw = primary_course.start_yaw_world
    playback: TerrainCoursePlayback | None = None
    active_course: MotionBricksTerrainCourse | None = None
    mode = "flat"
    last_capture = PortalCapture(False, math.inf, math.inf, -1.0, math.inf)
    blocked_by_terrain = False
    tick = 0
    sim_time_s = 0.0
    actual_trail: deque[np.ndarray] = deque(maxlen=200)
    selected_path = primary_course.root_position_world[
        :: max(1, primary_course.frame_count // 80)
    ].copy()
    selected_path[:, 2] += 0.10
    selected_facing = np.tile(np.asarray((1.0, 0.0)), (len(selected_path), 1))
    command_path = np.repeat(current_qpos[None, :3], 4, axis=0)
    command_facing = np.tile(np.asarray((1.0, 0.0)), (4, 1))
    safe_flat_history: deque[np.ndarray] = deque(maxlen=4)
    command_buffer = MotionBricksCommandBufferInvalidator()
    flat_support_recovery_count = 0

    def seed_motionbricks(
        course: MotionBricksTerrainCourse,
        *,
        at_end: bool,
        velocity: np.ndarray,
        facing: float,
        name: str,
    ) -> None:
        nonlocal current_qpos, flat_support_height
        context = course.resampled_context_qpos(
            at_end=at_end, target_fps=30.0, frame_count=4
        )
        support_height = _terrain_surface_height(
            terrain_index, current_qpos[:2], ray_origin_z=ray_origin_z
        )
        flat_support_height = support_height
        if at_end:
            phase_result = _submit_phase_matched_course_exit(
                full_agent,
                controller,
                context_qpos=context,
                velocity_world_xy=velocity,
                facing_yaw_world=facing,
                mode_name=name,
                support_height_world=support_height,
            )
            print(
                "[GLOBAL TERRAIN] selected live exit phase "
                f"seed={phase_result['selected']['seed']} "
                f"score={phase_result['score_minimum']:.4f}",
                flush=True,
            )
            lock_receipt = exit_foot_lock.arm(current_qpos)
            command_buffer.observe_generated(
                velocity_world_xy=velocity,
                facing_yaw_world=facing,
                mode_name=name,
            )
            print(
                "[GLOBAL TERRAIN] armed course-exit support handoff "
                f"foot={lock_receipt['support_foot']} "
                f"left_clearance={lock_receipt['left_clearance_m']:.4f} "
                f"right_clearance={lock_receipt['right_clearance_m']:.4f}",
                flush=True,
            )
            return
        exit_foot_lock.reset()
        _submit_motionbricks(
            full_agent,
            controller,
            context_qpos=context,
            velocity_world_xy=velocity,
            facing_yaw_world=facing,
            mode_name=name,
            force=True,
            support_height_world=support_height,
        )
        for _ in range(len(context)):
            full_agent.get_next_frame()
        command_buffer.observe_generated(
            velocity_world_xy=velocity,
            facing_yaw_world=facing,
            mode_name=name,
        )
        current_qpos, _ = _project_live_root_above_support(
            np.asarray(full_agent.get_next_frame(), dtype=np.float64),
            sole_adapter=sole_adapter,
            terrain=terrain_index,
            ray_origin_z=ray_origin_z,
            support_height_world=flat_support_height,
        )

    def reset_session() -> None:
        nonlocal current_qpos, desired_facing_yaw, playback, active_course, mode
        nonlocal flat_support_height
        nonlocal tick, sim_time_s, blocked_by_terrain, last_capture
        nonlocal flat_support_recovery_count
        full_agent.reset()
        exit_foot_lock.reset()
        command_buffer.reset()
        current_qpos = primary_course_qpos[0].copy()
        desired_facing_yaw = primary_course.start_yaw_world
        seed_motionbricks(
            primary_course,
            at_end=False,
            velocity=np.zeros(2),
            facing=desired_facing_yaw,
            name="idle",
        )
        playback = None
        active_course = None
        mode = "flat"
        tick = 0
        sim_time_s = 0.0
        blocked_by_terrain = False
        flat_support_recovery_count = 0
        last_capture = PortalCapture(False, math.inf, math.inf, -1.0, math.inf)
        actual_trail.clear()
        safe_flat_history.clear()
        safe_flat_history.append(current_qpos.copy())

    def recover_safe_flat_pose() -> None:
        """Re-seed MotionBricks at the last verified flat pose after escape."""

        nonlocal current_qpos
        exit_foot_lock.reset()
        if not safe_flat_history:
            raise RuntimeError("flat support recovery has no verified pose")
        context = np.stack(tuple(safe_flat_history))
        if len(context) < 4:
            context = np.concatenate(
                (
                    np.repeat(context[:1], 4 - len(context), axis=0),
                    context,
                ),
                axis=0,
            )
        current_qpos = context[-1].copy()
        _submit_motionbricks(
            full_agent,
            controller,
            context_qpos=context,
            velocity_world_xy=np.zeros(2, dtype=np.float64),
            facing_yaw_world=_yaw_wxyz(current_qpos[3:7]),
            mode_name="idle",
            force=True,
            support_height_world=flat_support_height,
        )
        # The decoded prefix is conditioning history, not new motion.
        for _ in range(len(context)):
            full_agent.get_next_frame()
        command_buffer.observe_generated(
            velocity_world_xy=np.zeros(2, dtype=np.float64),
            facing_yaw_world=_yaw_wxyz(current_qpos[3:7]),
            mode_name="idle",
        )

    def write_qpos() -> None:
        data.qpos[:] = current_qpos
        data.time = sim_time_s
        mujoco.mj_forward(model, data)

    def publish_frame() -> None:
        camera.lookat[:] = current_qpos[:3]
        renderer.update_scene(data, camera=camera)
        _append_overlay_to_scene(
            renderer.scene,
            command_path_world=command_path,
            command_facing_local=command_facing,
            command_root_yaw=_yaw_wxyz(current_qpos[3:7]),
            selected_path_world=selected_path,
            selected_facing_local=selected_facing,
            selected_root_yaw=0.0,
            actual_trail=actual_trail,
            mujoco=mujoco,
        )
        state.publish_frame(
            frame=renderer.render(), physics_step=tick, sim_time_s=sim_time_s
        )

    reset_session()
    write_qpos()
    server.start()
    hostname = socket.gethostname()
    receipt = {
        "schema": "motionbricks_global_terrain_browser_v1",
        "hostname": hostname,
        "port": server.port,
        "token": state.token,
        "url_after_tunnel": f"http://localhost:{server.port}/?token={state.token}",
        "tunnel_command": (
            f"ssh -N -L {server.port}:{hostname}:{server.port} "
            "bodow@scdt.stanford.edu"
        ),
        "kind": "clean_kinematics_not_physics",
        "flat_controller": "official MotionBricks G1",
        "terrain_controller": "precompiled exact-mesh-audited global portal course",
        "target_clip_index": (
            None if target_clip_index is None else int(target_clip_index)
        ),
        "course_motions": [str(course.path) for course in courses],
        "course_frame_counts": [course.frame_count for course in courses],
        "course_fps": [course.fps for course in courses],
        "course_lateral_axes_world_xy": [
            None if axis is None else list(axis) for axis in lateral_axes
        ],
        "portal_entry_lead_time_s": PORTAL_ENTRY_LEAD_TIME_S,
        "flat_terrain_guard_lookahead_m": (
            FLAT_TERRAIN_GUARD_LOOKAHEAD_M
        ),
        "flat_support_validation": {
            "maximum_support_error_m": FLAT_SUPPORT_TOLERANCE_M,
            "samples": "root plus every left/right sole support point",
            "failure_action": "hold last verified pose and re-seed idle gait",
        },
        "course_support_clearance": {
            "threshold_m": MAXIMUM_SUPPORT_FOOT_CLEARANCE_M,
            "maximum_by_course_m": list(course_support_clearance_maxima),
        },
        "course_exit_support_handoff": {
            "activation_clearance_m": EXIT_FOOT_LOCK_ACTIVATION_CLEARANCE_M,
            "release_clearance_m": EXIT_FOOT_LOCK_RELEASE_CLEARANCE_M,
            "release_confirmation_frames": (
                EXIT_FOOT_LOCK_RELEASE_CONFIRMATION_FRAMES
            ),
            "release_frames": EXIT_FOOT_LOCK_RELEASE_FRAMES,
            "maximum_joint_step_rad": EXIT_FOOT_LOCK_MAXIMUM_JOINT_STEP_RAD,
            "maximum_target_error_m": EXIT_FOOT_LOCK_MAXIMUM_TARGET_ERROR_M,
        },
        "course_seam_indices": [
            list(course.seam_indices) for course in courses
        ],
        "entry_family_count": len(courses),
        "terrain_usd_path": str(terrain_path),
        "visual_mesh_geom_count": int(visual_mesh_count),
        "global_scene_information_used": True,
        "external_robot_odometry_used": False,
        "switch_controller": "browser Gamepad API",
        "command_frame": "left stick robot-local travel; right stick robot-local facing",
        "route_commitment": "joystick resumes after the audited course landing",
    }
    print(
        "MOTIONBRICKS_GLOBAL_TERRAIN_READY "
        + json.dumps(receipt, sort_keys=True),
        flush=True,
    )
    print(
        "[GLOBAL TERRAIN] Connect the controller, press Start, and push along "
        "the orange entry path.",
        flush=True,
    )

    next_tick = time.perf_counter()
    try:
        publish_frame()
        if not state.wait_for_start(wait_seconds):
            raise TimeoutError("browser did not request Start before timeout")
        while not state.should_end():
            if state.consume_reset_request():
                reset_session()
                write_qpos()

            command, _, _ = state.current_command()
            yaw = _yaw_wxyz(current_qpos[3:7])
            flat_dt = 1.0 / 30.0
            requested_velocity, desired_facing_yaw, target_speed, mode_name = (
                _command_targets(
                    command,
                    current_yaw=yaw,
                    desired_facing_yaw=desired_facing_yaw,
                    dt_s=flat_dt,
                )
            )
            blocked_by_terrain = False
            if playback is None:
                seam_registered_rows = [
                    _laterally_registered_course(
                        course,
                        current_qpos[:2],
                        entry_lead_time_s=PORTAL_ENTRY_LEAD_TIME_S,
                        lateral_axis_world_xy=lateral_axis,
                        registration_frame_index=course.seam_indices[0],
                    )
                    for course, lateral_axis in zip(courses, lateral_axes)
                ]
                seam_registered_courses = tuple(
                    row[0] for row in seam_registered_rows
                )
                seam_selection = select_terrain_seam_portal(
                    seam_registered_courses,
                    current_qpos,
                    requested_velocity,
                )
                if seam_selection is not None:
                    registered_rows = seam_registered_rows
                    registered_courses = seam_registered_courses
                    selection = seam_selection
                    entry_kind = "landing_seam"
                    blend_frames = 18
                else:
                    registered_rows = [
                        _laterally_registered_course(
                            course,
                            current_qpos[:2],
                            entry_lead_time_s=PORTAL_ENTRY_LEAD_TIME_S,
                            lateral_axis_world_xy=lateral_axis,
                        )
                        for course, lateral_axis in zip(courses, lateral_axes)
                    ]
                    registered_courses = tuple(
                        row[0] for row in registered_rows
                    )
                    selection = select_terrain_portal(
                        registered_courses,
                        current_qpos,
                        requested_velocity,
                        entry_lead_time_s=PORTAL_ENTRY_LEAD_TIME_S,
                    )
                    entry_kind = "flat_lead"
                    blend_frames = 18
                candidate_course = registered_courses[selection.course_index]
                lateral_shift = registered_rows[selection.course_index][1]
                last_capture = selection.capture
                selected_path = candidate_course.root_position_world[
                    selection.entry_frame_index :: max(
                        1, candidate_course.frame_count // 80
                    )
                ].copy()
                selected_path[:, 2] += 0.10
                selected_facing = np.tile(
                    np.asarray((1.0, 0.0)), (len(selected_path), 1)
                )
                terrain_equivalent = (
                    last_capture.accepted
                    and _registration_matches_terrain(
                        courses[selection.course_index],
                        candidate_course,
                        terrain=terrain_index,
                        ray_origin_z=ray_origin_z,
                        entry_frame=selection.entry_frame_index,
                        lateral_axis_world_xy=(
                            lateral_axes[selection.course_index]
                        ),
                    )
                )
                if last_capture.accepted and terrain_equivalent:
                    exit_foot_lock.reset()
                    playback = TerrainCoursePlayback(
                        candidate_course,
                        current_qpos,
                        blend_frames=blend_frames,
                        start_frame=selection.entry_frame_index,
                        allow_terrain_seam_start=(
                            entry_kind == "landing_seam"
                        ),
                    )
                    active_course = candidate_course
                    mode = "course"
                    current_qpos = playback.next_qpos()
                    current_qpos, _ = _project_live_root_above_support(
                        current_qpos,
                        sole_adapter=sole_adapter,
                        terrain=terrain_index,
                        ray_origin_z=ray_origin_z,
                        sole_clearance_m=(
                            LANDING_SEAM_BLEND_CLEARANCE_M
                            if entry_kind == "landing_seam"
                            else 0.003
                        ),
                    )
                    dt = 1.0 / candidate_course.fps
                    print(
                        f"frame={tick:06d} PORTAL_COMMIT "
                        f"family={selection.course_index} "
                        f"entry_kind={entry_kind} "
                        f"entry_frame={selection.entry_frame_index} "
                        f"lateral_shift={lateral_shift:.3f} "
                        f"position_error={last_capture.position_error_m:.3f} "
                        f"yaw_error_deg={math.degrees(last_capture.yaw_error_rad):.1f} "
                        f"pose_rmse={last_capture.lower_body_rmse_rad:.3f}",
                        flush=True,
                    )
                else:
                    if last_capture.accepted and not terrain_equivalent:
                        last_capture = PortalCapture(
                            False,
                            last_capture.position_error_m,
                            last_capture.yaw_error_rad,
                            last_capture.travel_alignment,
                            last_capture.lower_body_rmse_rad,
                        )
                    guarded_velocity, blocked_by_terrain = _guard_flat_velocity(
                        terrain_index,
                        current_qpos[:3],
                        requested_velocity,
                        ray_origin_z=ray_origin_z,
                    )
                    if blocked_by_terrain:
                        target_speed = 0.0
                        mode_name = "idle"
                    # The phase-matched course exit and its anchored support
                    # transfer form one atomic safety phase.  Replanning the
                    # generated future underneath the locked foot can turn a
                    # smooth release into a joint snap.  The current joystick
                    # target is retained above and takes effect immediately
                    # after release from the verified handoff history.
                    if not exit_foot_lock.armed:
                        submission = _submit_responsive_motionbricks(
                            full_agent,
                            controller,
                            command_buffer=command_buffer,
                            verified_history=safe_flat_history,
                            velocity_world_xy=guarded_velocity,
                            facing_yaw_world=desired_facing_yaw,
                            mode_name=mode_name,
                            support_height_world=flat_support_height,
                        )
                        if submission.invalidated:
                            print(
                                f"frame={tick:06d} COMMAND_BUFFER_INVALIDATED "
                                f"reasons={','.join(submission.invalidation_reasons)}",
                                flush=True,
                            )
                    proposed_qpos, _ = _project_live_root_above_support(
                        np.asarray(
                            full_agent.get_next_frame(), dtype=np.float64
                        ),
                        sole_adapter=sole_adapter,
                        terrain=terrain_index,
                        ray_origin_z=ray_origin_z,
                        support_height_world=flat_support_height,
                    )
                    handoff = exit_foot_lock.apply(proposed_qpos)
                    proposed_qpos = handoff.qpos
                    if handoff.triggered:
                        print(
                            f"frame={tick:06d} EXIT_FOOT_LOCK "
                            f"support_foot={handoff.support_foot} "
                            f"raw_clearance_m={handoff.raw_minimum_clearance_m:.4f}",
                            flush=True,
                        )
                    if handoff.released:
                        print(
                            f"frame={tick:06d} EXIT_FOOT_LOCK_RELEASE "
                            f"corrected_clearance_m={handoff.corrected_minimum_clearance_m:.4f} "
                            f"joint_step_rad={handoff.maximum_joint_step_rad:.4f}",
                            flush=True,
                        )
                    # While the outgoing sole is intentionally latched at a
                    # terrain boundary, its footprint may still straddle the
                    # final riser and the flat landing.  The exit-lock IK has
                    # already checked exact-mesh support and target residual;
                    # applying the older single-plane guard here would reject
                    # that valid contact and disable the safety handoff.
                    support_error = (
                        0.0
                        if handoff.active
                        else _flat_pose_support_error(
                            proposed_qpos,
                            sole_adapter=sole_adapter,
                            terrain=terrain_index,
                            ray_origin_z=ray_origin_z,
                            support_height_world=flat_support_height,
                        )
                    )
                    if support_error > FLAT_SUPPORT_TOLERANCE_M:
                        blocked_by_terrain = True
                        flat_support_recovery_count += 1
                        recover_safe_flat_pose()
                        print(
                            f"frame={tick:06d} FLAT_SUPPORT_RECOVERY "
                            f"support_error_m={support_error:.4f} "
                            f"count={flat_support_recovery_count}",
                            flush=True,
                        )
                    else:
                        current_qpos = proposed_qpos
                        safe_flat_history.append(current_qpos.copy())
                    mode = "flat"
                    dt = flat_dt
            else:
                if active_course is None:
                    raise RuntimeError("course playback lost its active course")
                current_qpos = playback.next_qpos()
                if (
                    playback.index
                    <= playback.start_frame + playback.blend_frames
                ):
                    current_qpos, _ = _project_live_root_above_support(
                        current_qpos,
                        sole_adapter=sole_adapter,
                        terrain=terrain_index,
                        ray_origin_z=ray_origin_z,
                        sole_clearance_m=(
                            LANDING_SEAM_BLEND_CLEARANCE_M
                            if playback.start_frame
                            == playback.course.seam_indices[0]
                            else 0.003
                        ),
                    )
                dt = 1.0 / active_course.fps
                if playback.done:
                    seed_motionbricks(
                        active_course,
                        at_end=True,
                        velocity=requested_velocity,
                        facing=desired_facing_yaw,
                        name=mode_name,
                    )
                    playback = None
                    active_course = None
                    mode = "flat"
                    safe_flat_history.clear()
                    safe_flat_history.append(current_qpos.copy())
                    dt = flat_dt
                    print(
                        f"frame={tick:06d} COURSE_COMPLETE joystick_resumed=1",
                        flush=True,
                    )

            tick += 1
            sim_time_s += dt
            write_qpos()
            actual_trail.append(
                np.asarray(
                    (current_qpos[0], current_qpos[1], current_qpos[2] + 0.10)
                )
            )
            horizons = np.asarray((0.0, 0.3, 0.6, 0.9))
            command_path = np.empty((4, 3), dtype=np.float64)
            command_path[:, :2] = (
                current_qpos[:2] + horizons[:, None] * requested_velocity
            )
            command_path[:, 2] = current_qpos[2] + 0.10
            relative_facing = math.remainder(
                desired_facing_yaw - _yaw_wxyz(current_qpos[3:7]),
                2.0 * math.pi,
            )
            command_facing = np.tile(
                np.asarray(
                    (math.cos(relative_facing), math.sin(relative_facing))
                ),
                (4, 1),
            )
            task = np.asarray(
                (
                    requested_velocity[0],
                    requested_velocity[1],
                    math.cos(relative_facing),
                    math.sin(relative_facing),
                )
            )
            state.observe_policy(
                policy_call=tick,
                task=task,
                virtual_position_xy=command_path[-1, :2],
                actual_position_xy=current_qpos[:2],
                virtual_facing_yaw=desired_facing_yaw,
                actual_yaw=_yaw_wxyz(current_qpos[3:7]),
                target_velocity_xy=requested_velocity,
                target_facing_xy=(
                    math.cos(relative_facing), math.sin(relative_facing)
                ),
            )
            state.observe_actor_call(
                policy_call=tick,
                task=task,
                action=np.asarray(
                    (
                        float(mode == "course"),
                        float(blocked_by_terrain),
                        last_capture.position_error_m
                        if math.isfinite(last_capture.position_error_m)
                        else -1.0,
                    )
                ),
            )
            if tick % 2 == 0:
                publish_frame()
            if tick % 150 == 0:
                print(
                    f"frame={tick:06d} mode={mode} "
                    f"blocked={int(blocked_by_terrain)} "
                    f"portal_distance={last_capture.position_error_m:.3f}",
                    flush=True,
                )

            next_tick += dt
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -0.50:
                next_tick = time.perf_counter()
        return receipt
    finally:
        renderer.close()
        server.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-clip-index", type=int)
    parser.add_argument("--course-motion", type=Path, action="append")
    parser.add_argument("--terrain-catalog", type=Path)
    parser.add_argument(
        "--terrain-route-manifest", type=Path, action="append"
    )
    parser.add_argument("--terrain-usd", type=Path)
    parser.add_argument(
        "--terrain-position",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--terrain-quaternion-wxyz",
        type=float,
        nargs=4,
        default=(1.0, 0.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--motionbricks-root", type=Path, default=DEFAULT_MOTIONBRICKS_ROOT
    )
    parser.add_argument("--stairs-archive", type=Path, default=DEFAULT_STAIRS_ARCHIVE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--browser-host", default="0.0.0.0")
    parser.add_argument("--browser-port", type=int, default=8772)
    parser.add_argument("--browser-token")
    parser.add_argument("--browser-wait-seconds", type=float, default=43200.0)
    arguments = parser.parse_args()
    target_clip_index = arguments.target_clip_index
    manifest_scene = None
    manifest_courses: list[Path] = []
    manifest_lateral_axes: list[tuple[float, float]] = []
    if arguments.terrain_route_manifest is not None:
        if arguments.terrain_usd is not None or arguments.terrain_catalog is not None:
            parser.error(
                "--terrain-route-manifest cannot be combined with "
                "--terrain-usd or --terrain-catalog"
            )
        manifest_scenes = [
            _route_manifest(path)
            for path in arguments.terrain_route_manifest
        ]
        manifest_scene = manifest_scenes[0]
        for scene in manifest_scenes:
            if (
                scene[0] != manifest_scene[0]
                or not np.allclose(scene[1], manifest_scene[1], atol=1.0e-8)
                or not np.allclose(scene[2], manifest_scene[2], atol=1.0e-8)
            ):
                parser.error(
                    "all --terrain-route-manifest values must describe the "
                    "same globally placed terrain"
                )
            manifest_courses.extend(scene[3])
            manifest_lateral_axes.extend(scene[4])
    if (
        arguments.terrain_usd is None
        and manifest_scene is None
        and target_clip_index is None
    ):
        target_clip_index = 26
    course_paths = list(arguments.course_motion or ())
    course_lateral_axes: list[tuple[float, float] | None] = [
        None for _ in course_paths
    ]
    if arguments.terrain_catalog is not None:
        if target_clip_index is None:
            parser.error("--terrain-catalog requires --target-clip-index")
        catalog_paths = _catalog_course_paths(
            arguments.terrain_catalog, target_clip_index
        )
        course_paths.extend(catalog_paths)
        course_lateral_axes.extend(None for _ in catalog_paths)
    if manifest_scene is not None:
        course_paths.extend(manifest_courses)
        course_lateral_axes.extend(manifest_lateral_axes)
    unique_courses: dict[Path, tuple[float, float] | None] = {}
    for path, lateral_axis in zip(course_paths, course_lateral_axes):
        unique_courses.setdefault(path, lateral_axis)
    course_paths = list(unique_courses)
    course_lateral_axes = list(unique_courses.values())
    terrain_usd = arguments.terrain_usd
    terrain_position = tuple(arguments.terrain_position)
    terrain_quaternion = tuple(arguments.terrain_quaternion_wxyz)
    if manifest_scene is not None:
        terrain_usd, terrain_position, terrain_quaternion, _, _ = manifest_scene
    if not course_paths:
        course_paths = [DEFAULT_COURSE]
        course_lateral_axes = [None]
    run(
        target_clip_index=target_clip_index,
        course_paths=tuple(course_paths),
        course_lateral_axes=tuple(course_lateral_axes),
        motionbricks_root=arguments.motionbricks_root,
        stairs_archive=arguments.stairs_archive,
        terrain_usd=terrain_usd,
        terrain_position_world=terrain_position,
        terrain_quaternion_world_from_usd_wxyz=terrain_quaternion,
        model_path=arguments.model_path,
        host=arguments.browser_host,
        port=arguments.browser_port,
        token=arguments.browser_token,
        wait_seconds=arguments.browser_wait_seconds,
    )


if __name__ == "__main__":
    main()
