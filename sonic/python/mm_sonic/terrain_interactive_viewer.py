"""Interactive clean-kinematic G1 viewer for hybrid terrain motion matching.

This entry point intentionally never steps MuJoCo physics. It combines free
Takara/BONES flat matching with fixed-world Justin stair frames, writes the
result directly to ``qpos``, and calls ``mj_forward`` only so the articulated
G1 mesh can be inspected.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np

from .hybrid_terrain_interactive import (
    FlatKinematicSource,
    HybridKinematicSession,
    HybridMode,
    KinematicPose,
    StairHandoffIndex,
    flat_preview_uses_nominal_sole_proxy,
    flat_selected_preview_local,
    pose_from_stair_row,
)
from .operator_x11 import (
    X11KeyStateProvider,
    normalized_state_from_pressed,
)
from .render_terrain_transition_mesh import (
    DEFAULT_G1_STAIR_SCENE,
    build_qpos,
)
from .terrain_catalog import FUTURE_OFFSETS, load_database
from .terrain_interactive import (
    KinematicTerrainSession,
    TerrainCommandBall,
    TerrainKeyboardMapper,
)
from .terrain_scene import (
    FourWayPlatformScene,
    flat_motion_preview_is_safe,
    observe_stair_from_height_map,
    path_has_unsafe_height_change,
    shape_stair_handoff_velocity,
    shape_terrain_approach_facing_yaw,
    shape_terrain_approach_velocity,
)


DEFAULT_TERRAIN_CATALOG = Path(
    "/move/u/bodow/Projects/Motion-Matching-takara-corpus/"
    "artifacts/terrain_catalog/"
    "justin_grail_c490_repaired_v9_entry45.npz"
)
DEFAULT_JUSTIN_SOURCE = Path(
    "/move/u/bodow/Projects/Motion-Matching-takara-corpus/"
    "artifacts/terrain_catalog/"
    "justin_grail_c490_repaired_v9.zarr"
)
DEFAULT_FLAT_MOTIONS = Path(
    "/move/data/terrain-aware/motion-matching/"
    "takara_bones_walk_support_v2_startstop"
)


def interactive_transition_guard_options() -> dict[str, object]:
    """Return the retained browser/evaluator entry-lock recipe.

    Full-route source-contact IK was an experimental variant that introduced
    extra corrections and transactional holds on otherwise valid authored
    stair frames.  The interactive viewer must match the retained evaluator:
    lock only the outgoing support at the splice, then release after the short
    entry window.
    """

    return {
        "track_source_contacts": False,
        "source_contact_delay_frames": 4,
    }


def interactive_handoff_options() -> dict[str, float]:
    """Use the visually validated distributed alignment for both directions."""

    return {
        "maximum_planted_foot_error_m": 0.07,
        "maximum_planar_warp_m": 0.07,
        "maximum_continuation_joint_step_rad": 0.25,
    }


def _yaw_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


@dataclass(frozen=True)
class TerrainKinematicArchive:
    """The clean source arrays needed by direct kinematic playback."""

    clip_names: tuple[str, ...]
    clip_start: np.ndarray
    clip_end: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    joint_pos: np.ndarray
    fps: float
    joint_vel: np.ndarray | None = None
    body_lin_vel_w: np.ndarray | None = None
    body_ang_vel_w: np.ndarray | None = None

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.clip_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("clip_names must be non-empty and unique")
        starts = np.asarray(self.clip_start, dtype=np.int64)
        ends = np.asarray(self.clip_end, dtype=np.int64)
        if starts.shape != (len(names),) or ends.shape != starts.shape:
            raise ValueError("clip boundaries do not match clip_names")
        if np.any(starts < 0) or np.any(ends <= starts):
            raise ValueError("clip boundaries must be ordered non-empty ranges")
        body_pos = np.asarray(self.body_pos_w, dtype=np.float32)
        body_quat = np.asarray(self.body_quat_w, dtype=np.float32)
        joints = np.asarray(self.joint_pos, dtype=np.float32)
        frame_count = len(body_pos)
        if body_pos.shape != (frame_count, 30, 3):
            raise ValueError("body_pos_w must have shape [N,30,3]")
        if body_quat.shape != (frame_count, 30, 4):
            raise ValueError("body_quat_w must have shape [N,30,4]")
        if joints.shape != (frame_count, 29):
            raise ValueError("joint_pos must have shape [N,29]")
        if int(np.max(ends)) > frame_count:
            raise ValueError("clip boundary exceeds the source frame count")
        if not all(
            np.isfinite(value).all()
            for value in (body_pos, body_quat, joints)
        ):
            raise ValueError("kinematic source contains non-finite values")
        optional = {
            "joint_vel": (self.joint_vel, (frame_count, 29)),
            "body_lin_vel_w": (
                self.body_lin_vel_w,
                (frame_count, 30, 3),
            ),
            "body_ang_vel_w": (
                self.body_ang_vel_w,
                (frame_count, 30, 3),
            ),
        }
        for name, (value, shape) in optional.items():
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float32)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(
                    f"{name} must contain finite values with shape {shape}"
                )
            object.__setattr__(
                self, name, np.ascontiguousarray(array)
            )
        rate = float(self.fps)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("fps must be positive and finite")
        object.__setattr__(self, "clip_names", names)
        object.__setattr__(self, "clip_start", np.ascontiguousarray(starts))
        object.__setattr__(self, "clip_end", np.ascontiguousarray(ends))
        object.__setattr__(self, "body_pos_w", np.ascontiguousarray(body_pos))
        object.__setattr__(
            self, "body_quat_w", np.ascontiguousarray(body_quat)
        )
        object.__setattr__(self, "joint_pos", np.ascontiguousarray(joints))
        object.__setattr__(self, "fps", rate)

    @classmethod
    def from_zarr(cls, path: str | Path) -> "TerrainKinematicArchive":
        import zarr

        root = zarr.open(str(Path(path)), mode="r")
        if root.attrs.get("quaternion_convention") != "xyzw":
            raise ValueError("Justin source must declare xyzw quaternions")
        fps_value = np.asarray(root["fps"]).reshape(-1)
        if fps_value.size != 1:
            raise ValueError("Justin source fps must be scalar")
        return cls(
            clip_names=tuple(str(value) for value in root["clip_names"][:]),
            clip_start=np.asarray(root["clip_start_idx"], dtype=np.int64),
            clip_end=np.asarray(root["clip_end_idx"], dtype=np.int64),
            body_pos_w=np.asarray(root["body_pos_w"], dtype=np.float32),
            body_quat_w=np.asarray(root["body_quat_w"], dtype=np.float32),
            joint_pos=np.asarray(root["joint_pos"], dtype=np.float32),
            fps=float(fps_value[0]),
            joint_vel=np.asarray(root["joint_vel"], dtype=np.float32),
            body_lin_vel_w=np.asarray(
                root["body_lin_vel_w"], dtype=np.float32
            ),
            body_ang_vel_w=np.asarray(
                root["body_ang_vel_w"], dtype=np.float32
            ),
        )

    def global_index(self, source_clip: int, source_frame: int) -> int:
        if type(source_clip) is not int or not 0 <= source_clip < len(
            self.clip_names
        ):
            raise ValueError("source_clip is outside the archive")
        if type(source_frame) is not int:
            raise ValueError("source_frame must be an integer")
        frame_count = int(
            self.clip_end[source_clip] - self.clip_start[source_clip]
        )
        if not 0 <= source_frame < frame_count:
            raise ValueError("source_frame is outside clip")
        return int(self.clip_start[source_clip]) + source_frame

    def root_pose(
        self, source_clip: int, source_frame: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        index = self.global_index(source_clip, source_frame)
        position = self.body_pos_w[index, 0]
        quaternion = self.body_quat_w[index, 0]
        return (
            position,
            quaternion,
            self.joint_pos[index],
            _yaw_xyzw(quaternion),
        )


def local_path_to_world(
    local_xy: np.ndarray,
    *,
    root_position_world: np.ndarray,
    root_yaw_world: float,
    height_offset_m: float = 0.08,
) -> np.ndarray:
    """Place a robot-local 2-D command path at the current mesh root."""

    path = np.asarray(local_xy, dtype=np.float64)
    root = np.asarray(root_position_world, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or not np.isfinite(path).all():
        raise ValueError("local_xy must have shape [N,2] and be finite")
    if root.shape != (3,) or not np.isfinite(root).all():
        raise ValueError("root_position_world must have shape [3]")
    yaw = float(root_yaw_world)
    offset = float(height_offset_m)
    if not math.isfinite(yaw) or not math.isfinite(offset):
        raise ValueError("root yaw and height offset must be finite")
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(((c, -s), (s, c)), dtype=np.float64)
    world = np.empty((len(path), 3), dtype=np.float64)
    world[:, :2] = root[:2] + path @ rotation.T
    world[:, 2] = root[2] + offset
    return world


def browser_command_targets(
    command: object,
    *,
    current_yaw_world: float,
    filtered_facing_yaw_world: float,
    max_speed_mps: float,
    max_yaw_rate_rad_s: float,
    dt_s: float,
) -> tuple[np.ndarray, float]:
    """Convert browser/Gamepad axes to robot-local travel and facing targets."""

    speed = float(max_speed_mps)
    yaw_rate = float(max_yaw_rate_rad_s)
    dt = float(dt_s)
    current_yaw = float(current_yaw_world)
    filtered_yaw = float(filtered_facing_yaw_world)
    if (
        speed <= 0.0
        or yaw_rate <= 0.0
        or dt <= 0.0
        or not all(
            math.isfinite(value)
            for value in (speed, yaw_rate, dt, current_yaw, filtered_yaw)
        )
    ):
        raise ValueError("browser command scales and yaw must be finite")
    velocity = (
        np.asarray(
            (
                float(command.travel_forward),
                float(command.travel_left),
            ),
            dtype=np.float64,
        )
        * speed
        * float(command.speed_scale)
    )
    if bool(command.facing_active):
        desired_yaw = current_yaw + math.atan2(
            float(command.facing_left),
            float(command.facing_forward),
        )
    elif abs(float(command.yaw_left)) > 1.0e-8:
        desired_yaw = (
            filtered_yaw
            + float(command.yaw_left)
            * yaw_rate
            * float(command.turn_scale)
            * dt
        )
    else:
        # Centering the right stick retains the last facing request. Snapping
        # it to the lagging character yaw creates a moving target and makes
        # otherwise smooth flat turns wobble.
        desired_yaw = filtered_yaw
    return velocity, math.remainder(desired_yaw, 2.0 * math.pi)


def browser_diagnostic_action(outcome: object) -> np.ndarray:
    """Return finite browser-only match diagnostics for every outcome."""

    cost = float(outcome.cost)
    if not math.isfinite(cost):
        cost = -1.0
    action = np.asarray(
        (
            cost,
            float(outcome.candidate_count),
            float(outcome.maximum_planted_foot_mismatch_m),
        ),
        dtype=np.float64,
    )
    if not np.isfinite(action).all():
        raise ValueError("browser match diagnostic must be finite")
    return action


def _build_four_way_platform_model(
    scene_xml: Path,
    terrain_scene: FourWayPlatformScene,
    mujoco: object,
) -> object:
    """Replace the source scene's fixed stair meshes with finite box geometry."""

    spec = mujoco.MjSpec.from_file(str(scene_xml.resolve()))
    for body in tuple(spec.bodies):
        if str(body.name).startswith("multi_boxes_box"):
            spec.delete(body)
    colors = {
        "platform": (0.24, 0.28, 0.34, 1.0),
        "west": (0.24, 0.52, 0.76, 1.0),
        "east": (0.30, 0.62, 0.72, 1.0),
        "south": (0.36, 0.56, 0.72, 1.0),
        "north": (0.28, 0.48, 0.68, 1.0),
    }
    for box in terrain_scene.box_geometries():
        group = box.name.split("_")[0]
        spec.worldbody.add_geom(
            name=f"terrain_{box.name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=box.center_world_xyz,
            euler=(0.0, 0.0, box.yaw_world),
            size=box.half_size_xyz,
            rgba=colors[group],
            contype=1,
            conaffinity=1,
        )
    return spec.compile()


def _initial_row(database: object, archive: TerrainKinematicArchive,
                 clip_name: str, source_frame: int) -> int:
    try:
        clip = archive.clip_names.index(clip_name)
    except ValueError as error:
        raise ValueError(f"unknown start clip {clip_name!r}") from error
    rows = np.flatnonzero(
        (database.source_clip == clip)
        & (database.source_frame == source_frame)
    )
    if rows.size != 1:
        raise ValueError(
            f"start frame {clip_name}:{source_frame} is not searchable"
        )
    return int(rows[0])


def _catalog_clip_names(catalog: Path) -> tuple[str, ...]:
    report = json.loads(catalog.with_suffix(".json").read_text())
    clips = tuple(str(value) for value in report["clips"])
    if not clips:
        raise ValueError("terrain catalog report has no clips")
    return clips


def _trajectory_world(
    local_xy: np.ndarray,
    root_position: np.ndarray,
    root_yaw: float,
) -> np.ndarray:
    path = np.vstack((np.zeros((1, 2), dtype=np.float32), local_xy))
    return local_path_to_world(
        path,
        root_position_world=root_position,
        root_yaw_world=root_yaw,
        height_offset_m=0.10,
    )


def _append_sphere(scene: object, position: np.ndarray, radius: float,
                   rgba: np.ndarray, mujoco: object) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius], dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _append_line(scene: object, start: np.ndarray, end: np.ndarray,
                 width: float, rgba: np.ndarray, mujoco: object) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def _append_path(scene: object, points: np.ndarray, color: np.ndarray,
                 mujoco: object, *, spheres: bool) -> None:
    for start, end in zip(points[:-1], points[1:], strict=True):
        _append_line(scene, start, end, 4.0, color, mujoco)
    if spheres:
        for point in points[1:]:
            _append_sphere(scene, point, 0.027, color, mujoco)


def _facing_endpoint(
    point_world: np.ndarray,
    facing_local: np.ndarray,
    root_yaw_world: float,
) -> np.ndarray:
    c, s = math.cos(root_yaw_world), math.sin(root_yaw_world)
    direction = np.asarray(
        (
            c * facing_local[0] - s * facing_local[1],
            s * facing_local[0] + c * facing_local[1],
            0.0,
        ),
        dtype=np.float64,
    )
    return point_world + 0.18 * direction


def _update_overlay(
    viewer: object,
    *,
    command_path_world: np.ndarray,
    command_facing_local: np.ndarray,
    command_root_yaw: float,
    selected_path_world: np.ndarray,
    selected_facing_local: np.ndarray,
    selected_root_yaw: float,
    actual_trail: deque[np.ndarray],
    mujoco: object,
) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0
    _append_overlay_to_scene(
        scene,
        command_path_world=command_path_world,
        command_facing_local=command_facing_local,
        command_root_yaw=command_root_yaw,
        selected_path_world=selected_path_world,
        selected_facing_local=selected_facing_local,
        selected_root_yaw=selected_root_yaw,
        actual_trail=actual_trail,
        mujoco=mujoco,
    )


def _append_overlay_to_scene(
    scene: object,
    *,
    command_path_world: np.ndarray,
    command_facing_local: np.ndarray,
    command_root_yaw: float,
    selected_path_world: np.ndarray,
    selected_facing_local: np.ndarray,
    selected_root_yaw: float,
    actual_trail: deque[np.ndarray],
    mujoco: object,
) -> None:
    """Append command/selection diagnostics to an existing rendered scene."""

    cyan = np.asarray((0.05, 0.90, 1.0, 0.95), np.float32)
    orange = np.asarray((1.0, 0.55, 0.05, 0.90), np.float32)
    red = np.asarray((1.0, 0.12, 0.12, 0.80), np.float32)
    _append_path(scene, command_path_world, cyan, mujoco, spheres=True)
    _append_path(scene, selected_path_world, orange, mujoco, spheres=False)
    if len(actual_trail) >= 2:
        _append_path(
            scene, np.asarray(actual_trail), red, mujoco, spheres=False
        )
    _append_line(
        scene,
        command_path_world[-1],
        _facing_endpoint(
            command_path_world[-1],
            command_facing_local[-1],
            command_root_yaw,
        ),
        5.0,
        cyan,
        mujoco,
    )
    _append_line(
        scene,
        selected_path_world[-1],
        _facing_endpoint(
            selected_path_world[-1],
            selected_facing_local[-1],
            selected_root_yaw,
        ),
        5.0,
        orange,
        mujoco,
    )


def _configure_terrain_browser_ui(browser_module: object) -> None:
    """Relabel the existing tested SONIC browser shell for kinematics."""

    replacements = {
        "SONIC clean-direct Task4 · interactive":
            "G1 terrain motion matching · clean kinematics",
        "Waiting for first MuJoCo frame":
            "Waiting for first clean-kinematic G1 frame",
        "this policy may continue walking":
            "the kinematic player stops at the next double support",
        "End this GPU simulation session?":
            "End this clean-kinematic session?",
        "Task4 [": "MM command [",
        "actor Task4 [": "selected command [",
        "odometry used for command:":
            "root translation odometry used for command:",
    }
    template = str(browser_module._HTML_TEMPLATE)
    for source, target in replacements.items():
        template = template.replace(source, target)
    browser_module._HTML_TEMPLATE = template


def _rotate_world_to_initial(
    vector_world_xy: np.ndarray, initial_yaw_world: float
) -> np.ndarray:
    c, s = math.cos(initial_yaw_world), math.sin(initial_yaw_world)
    x, y = np.asarray(vector_world_xy, dtype=np.float64)
    return np.asarray((c * x + s * y, -s * x + c * y))


def guard_runtime_pose_repair(
    candidate_pose: object,
    *,
    previous_accepted_pose: object | None,
    repairer: object,
) -> tuple[object, object, bool]:
    """Never expose a pose rejected by the exact terrain collision oracle."""

    repair = getattr(repairer, "repair", None)
    if not callable(repair):
        raise ValueError("repairer must provide repair(pose)")
    result = repair(candidate_pose)
    if bool(getattr(result, "accepted", False)):
        return result.pose, result, False
    if previous_accepted_pose is None:
        raise RuntimeError(
            "initial kinematic pose violates the terrain collision gate"
        )
    return previous_accepted_pose, result, True


def run_browser_interactive(
    catalog: Path,
    source_zarr: Path,
    scene_xml: Path,
    *,
    flat_motions_dir: Path = DEFAULT_FLAT_MOTIONS,
    flat_device: str = "cpu",
    start_clip: str = "up_zero",
    start_frame: int = 0,
    host: str = "0.0.0.0",
    port: int = 8765,
    token: str | None = None,
    wait_seconds: float = 43200.0,
    max_speed_mps: float = 0.70,
    max_yaw_rate_rad_s: float = 0.70,
    maximum_pose_repair_joint_delta_rad: float = 0.10,
) -> dict[str, object]:
    """Serve hybrid flat↔stair kinematics through the browser/Gamepad UI."""

    try:
        from diffusion_policy.inference import (
            sonic_interactive_controller as browser_control,
        )
    except ImportError as error:
        raise RuntimeError(
            "browser frontend requires the TML-BeyondMimic interactive "
            "controller on PYTHONPATH"
        ) from error
    import mujoco
    from .torch_motion_matcher import MatcherConfig, TorchMotionMatcher

    _configure_terrain_browser_ui(browser_control)
    database = load_database(catalog)
    archive = TerrainKinematicArchive.from_zarr(source_zarr)
    if _catalog_clip_names(catalog) != archive.clip_names:
        raise ValueError("catalog and Justin archive clip order differ")
    initial = _initial_row(database, archive, start_clip, start_frame)
    terrain_scene = FourWayPlatformScene()
    start_stair = terrain_scene.stairs[0]
    initial_position = start_stair.stair_to_world_xyz(
        np.asarray(
            (
                database.root_xy_stair[initial, 0],
                database.root_xy_stair[initial, 1],
                database.root_height_above_stair_base_m[initial],
            ),
            dtype=np.float32,
        )
    )
    initial_yaw = math.remainder(
        start_stair.ascent_yaw_world
        - float(database.stair_ascent_yaw_root[initial]),
        2.0 * math.pi,
    )
    print(
        f"[HYBRID MM] Loading {flat_motions_dir} on {flat_device}",
        flush=True,
    )
    flat_matcher = TorchMotionMatcher.from_folder(
        flat_motions_dir,
        device=flat_device,
        config=MatcherConfig(
            trajectory_model="takara_ball",
            max_source_joint_step_rad=0.35,
        ),
    )
    model = _build_four_way_platform_model(
        scene_xml, terrain_scene, mujoco
    )
    from .terrain_pose_repair import G1TerrainPoseRepair
    from .terrain_foot_lock import (
        G1TerrainFootLock,
        G1TerrainTransitionGuard,
    )

    pose_repairer = G1TerrainPoseRepair(
        model,
        terrain_scene,
        maximum_penetration_m=0.006,
        maximum_joint_delta_rad=maximum_pose_repair_joint_delta_rad,
    )
    transition_guard = G1TerrainTransitionGuard(
        pose_repairer,
        G1TerrainFootLock(
            model,
            terrain_scene,
            maximum_foot_penetration_m=0.010,
            maximum_locked_foot_drift_m=0.025,
        ),
        **interactive_transition_guard_options(),
    )
    initial_support_pose = pose_from_stair_row(
        database,
        archive,
        initial,
        placement=start_stair,
    )
    initial_support_pose, _initial_support_repair, _ = (
        guard_runtime_pose_repair(
            initial_support_pose,
            previous_accepted_pose=None,
            repairer=pose_repairer,
        )
    )
    flat_source = FlatKinematicSource(flat_matcher)
    session = HybridKinematicSession(
        database,
        archive,
        flat_source,
        initial_root_position_world=initial_position,
        initial_root_yaw_world=initial_yaw,
        initial_support_pose_world=initial_support_pose,
        initial_support_contact=np.asarray(
            database.contact[initial], dtype=bool
        ),
        entry_index=StairHandoffIndex(
            database,
            archive,
            **interactive_handoff_options(),
        ),
    )
    initial_pose = session.reset()
    initial_pose, _initial_flat_repair, _ = guard_runtime_pose_repair(
        initial_pose,
        previous_accepted_pose=None,
        repairer=pose_repairer,
    )
    session.current_pose = initial_pose
    ball = TerrainCommandBall()
    ball.reset(current_yaw_world=initial_yaw)

    data = mujoco.MjData(model)
    data.qpos[:] = build_qpos(
        initial_pose.root_position_world,
        initial_pose.root_orientation_world_xyzw,
        initial_pose.joint_position,
        model,
    )
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth = max(
        int(model.vis.global_.offwidth), 960
    )
    model.vis.global_.offheight = max(
        int(model.vis.global_.offheight), 540
    )
    renderer = mujoco.Renderer(model, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = initial_pose.root_position_world
    camera.distance = 2.6
    camera.azimuth = math.degrees(initial_yaw) + 55.0
    camera.elevation = -18.0

    state = browser_control.InteractiveControllerState(
        token=token,
        deadman_timeout_s=0.40,
        max_stream_fps=20.0,
        task_dim=4,
        command_profile=browser_control.DIRECT_TWO_STICK_TASK4,
        max_speed_mps=max_speed_mps,
        max_yaw_rate_rad_s=max_yaw_rate_rad_s,
    )
    server = browser_control.InteractiveControllerServer(
        state, host=host, port=port
    )
    actual_trail: deque[np.ndarray] = deque(maxlen=150)
    future_root = np.zeros((len(FUTURE_OFFSETS), 2), np.float32)
    future_facing = np.tile(
        np.asarray([1.0, 0.0], np.float32),
        (len(FUTURE_OFFSETS), 1),
    )

    def publish_frame() -> None:
        pose = session.current_pose
        if pose is None:
            raise RuntimeError("hybrid session has no current pose")
        position = pose.root_position_world
        yaw = pose.root_yaw_world
        command_world = _trajectory_world(
            future_root, position, yaw
        )
        if (
            session.mode is HybridMode.STAIR
            and session.terrain_session is not None
        ):
            row = int(session.terrain_session.active_row)
            selected_root = database.future_root_xy[row]
            selected_facing = database.future_facing_xy[row]
        else:
            selected_root, selected_facing = flat_selected_preview_local(
                flat_source,
                pose,
                offsets_frames=FUTURE_OFFSETS,
            )
        selected_world = _trajectory_world(
            selected_root,
            position,
            yaw,
        )
        camera.lookat[:] = position
        renderer.update_scene(data, camera=camera)
        _append_overlay_to_scene(
            renderer.scene,
            command_path_world=command_world,
            command_facing_local=future_facing,
            command_root_yaw=yaw,
            selected_path_world=selected_world,
            selected_facing_local=selected_facing,
            selected_root_yaw=yaw,
            actual_trail=actual_trail,
            mujoco=mujoco,
        )
        state.publish_frame(
            frame=renderer.render(),
            physics_step=session.tick,
            sim_time_s=session.tick / archive.fps,
        )

    server.start()
    hostname = socket.gethostname()
    receipt = {
        "schema": "terrain_mm_hybrid_kinematic_browser_ready_v2",
        "hostname": hostname,
        "port": server.port,
        "token": state.token,
        "url_after_tunnel": (
            f"http://localhost:{server.port}/?token={state.token}"
        ),
        "tunnel_command": (
            f"ssh -N -L {server.port}:{hostname}:{server.port} "
            "bodow@scdt.stanford.edu"
        ),
        "kind": "clean_kinematics_not_physics",
        "terrain_perception": {
            "input": "48x32 robot-centred causal height map",
            "global_root_used_for_matching": False,
            "layout": "finite central platform with four stair instances",
        },
        "motion_sources": {
            "flat": str(flat_motions_dir.resolve()),
            "stairs": str(source_zarr.resolve()),
        },
        "switch_controller": "browser Gamepad API",
        "root_translation_odometry_used_for_command": False,
        "terrain_pose_repair": {
            "exact_mujoco_collision_gate": True,
            "maximum_foot_penetration_m": 0.006,
            "maximum_joint_delta_rad": (
                pose_repairer.maximum_joint_delta_rad
            ),
            "rejected_pose_behavior": "hold_previous_accepted_pose",
            "entry_contact_lock_frames": (
                transition_guard.contact_hold_frames
            ),
            "entry_contact_lock_forced_release": (
                not transition_guard.track_source_contacts
            ),
            "source_contact_tracking": (
                transition_guard.track_source_contacts
            ),
            "landing_handoff": (
                "phase-matched flat bridge until both complete soles are "
                "beyond the top tread"
            ),
        },
    }
    print("TERRAIN_INTERACTIVE_READY "
          + json.dumps(receipt, sort_keys=True), flush=True)
    print(
        "[TERRAIN INTERACTIVE] Open the localhost URL, connect/select the "
        "Switch controller, then press Start simulation.",
        flush=True,
    )
    period = 1.0 / archive.fps
    next_tick = time.perf_counter()
    last_telemetry_key: tuple[str, str, int, int] | None = None
    try:
        publish_frame()
        if not state.wait_for_start(wait_seconds):
            raise TimeoutError("browser did not request Start before timeout")
        while not state.should_end():
            if state.consume_reset_request():
                transition_guard.end_stair()
                reset_pose = session.reset()
                reset_pose, _reset_repair, _ = guard_runtime_pose_repair(
                    reset_pose,
                    previous_accepted_pose=None,
                    repairer=pose_repairer,
                )
                session.current_pose = reset_pose
                ball.reset(current_yaw_world=initial_yaw)
                actual_trail.clear()
                future_root[:] = 0.0
                future_facing[:] = (1.0, 0.0)
                data.qpos[:] = build_qpos(
                    reset_pose.root_position_world,
                    reset_pose.root_orientation_world_xyzw,
                    reset_pose.joint_position,
                    model,
                )
                mujoco.mj_forward(model, data)

            query_pose = session.current_pose
            if query_pose is None:
                raise RuntimeError("hybrid session has no current pose")
            query_position = query_pose.root_position_world
            query_yaw = query_pose.root_yaw_world
            command, _, _ = state.current_command()
            target_velocity, desired_yaw = browser_command_targets(
                command,
                current_yaw_world=query_yaw,
                filtered_facing_yaw_world=ball.facing_yaw_world,
                max_speed_mps=max_speed_mps,
                max_yaw_rate_rad_s=max_yaw_rate_rad_s,
                dt_s=period,
            )
            intent_future_root, _intent_future_facing = ball.preview(
                target_velocity,
                desired_yaw,
                query_yaw,
                FUTURE_OFFSETS,
                archive.fps,
            )
            height_map = terrain_scene.height_map(
                root_position_world=query_position,
                root_yaw_world=query_yaw,
            )
            observed_stair = observe_stair_from_height_map(height_map)
            desired_yaw = shape_terrain_approach_facing_yaw(
                current_yaw_world=query_yaw,
                requested_facing_yaw_world=desired_yaw,
                local_velocity_xy=target_velocity,
                stair=observed_stair,
                facing_override_active=bool(command.facing_active)
                or abs(float(command.yaw_left)) > 1.0e-8,
            )
            target_velocity = shape_terrain_approach_velocity(
                target_velocity, observed_stair
            )
            handoff_active = session.mode in (
                HybridMode.STAIR_ARMED,
                HybridMode.SAFE_STOP,
            )
            alignment_velocity_robot = (
                session.approach_velocity_correction_robot_xy.copy()
            )
            target_velocity = shape_stair_handoff_velocity(
                target_velocity,
                alignment_velocity_robot,
                handoff_active=handoff_active,
                phase_staging_active=session.phase_staging_active,
            )
            ball.update(
                target_velocity, desired_yaw, query_yaw, period
            )
            future_root, future_facing = ball.preview(
                target_velocity,
                desired_yaw,
                query_yaw,
                FUTURE_OFFSETS,
                archive.fps,
            )
            stop_requested = bool(
                np.linalg.norm(target_velocity) < 0.01
                and np.linalg.norm(ball.velocity_world) < 0.08
            )
            flat_path_safe = not path_has_unsafe_height_change(
                height_map,
                # Guard the next 120 ms, not the whole intent horizon.  This
                # lets flat MM reach a compatible pre-contact pose while still
                # preventing the next executed segment from crossing an edge.
                future_root[:1],
                # The coarse intent path may approach inside the detected
                # stair corridor.  The selected seven-frame flat motion is
                # still transactionally checked below with sole geometry, so
                # this does not permit the matcher to cross a riser.
                stair=observed_stair,
            )
            c, s = math.cos(query_yaw), math.sin(query_yaw)
            target_velocity_world = np.asarray(
                (
                    c * target_velocity[0] - s * target_velocity[1],
                    s * target_velocity[0] + c * target_velocity[1],
                ),
                dtype=np.float64,
            )
            safe_recovery_velocity_world = np.asarray(
                (
                    c * alignment_velocity_robot[0]
                    - s * alignment_velocity_robot[1],
                    s * alignment_velocity_robot[0]
                    + c * alignment_velocity_robot[1],
                ),
                dtype=np.float64,
            )

            def validate_flat_preview(prepared: object) -> bool:
                selected_roots, selected_bodies = (
                    flat_source.prepared_preview_world(
                        prepared,
                        frame_count=7,
                    )
                )
                if not flat_motion_preview_is_safe(
                    height_map,
                    current_root_position_world=query_position,
                    current_root_yaw_world=query_yaw,
                    dense_root_position_world=selected_roots,
                    dense_body_position_world=selected_bodies,
                    check_nominal_soles=(
                        flat_preview_uses_nominal_sole_proxy(
                            session.mode,
                            exact_pose_repair=True,
                            post_landing_recovery=(
                                session.post_landing_recovery_active
                            ),
                        )
                    ),
                ):
                    return False
                return bool(
                    pose_repairer.repair(prepared.pose).accepted
                )

            outcome = session.step(
                target_velocity_world,
                desired_yaw,
                future_root,
                future_facing,
                stop_requested=stop_requested,
                stair_observation=observed_stair,
                flat_path_safe=flat_path_safe,
                flat_preview_validator=validate_flat_preview,
                safe_recovery_velocity_world_xy=(
                    safe_recovery_velocity_world
                    if handoff_active
                    else np.zeros(2, dtype=np.float64)
                ),
                intent_future_root_xy=intent_future_root,
                terrain_pose_filter=transition_guard,
            )
            if (
                transition_guard.last_pose_repair_result is not None
            ):
                pose = outcome.pose
            else:
                pose, _repair, repair_held = guard_runtime_pose_repair(
                    outcome.pose,
                    previous_accepted_pose=query_pose,
                    repairer=pose_repairer,
                )
                outcome = replace(
                    outcome,
                    pose=pose,
                    reason=(
                        outcome.reason
                        if not repair_held
                        else (
                            f"{outcome.reason}; exact collision repair "
                            "rejected, holding last accepted pose"
                        )
                    ),
                )
            session.current_pose = pose
            position = pose.root_position_world
            quaternion = pose.root_orientation_world_xyzw
            joints = pose.joint_position
            selected_yaw = pose.root_yaw_world
            data.qpos[:] = build_qpos(
                position, quaternion, joints, model
            )
            data.time = session.tick / archive.fps
            mujoco.mj_forward(model, data)
            actual_trail.append(
                np.asarray(
                    (position[0], position[1], position[2] + 0.10),
                    dtype=np.float64,
                )
            )

            actual_local = _rotate_world_to_initial(
                position[:2] - initial_position[:2], initial_yaw
            )
            yaw_from_initial = selected_yaw - initial_yaw
            c, s = math.cos(yaw_from_initial), math.sin(yaw_from_initial)
            endpoint_local = actual_local + np.asarray(
                (
                    c * future_root[-1, 0] - s * future_root[-1, 1],
                    s * future_root[-1, 0] + c * future_root[-1, 1],
                )
            )
            horizon_s = FUTURE_OFFSETS[-1] / archive.fps
            command_task = np.concatenate(
                (
                    future_root[-1] / horizon_s,
                    future_facing[-1],
                )
            )
            target_relative_yaw = desired_yaw - query_yaw
            target_facing = (
                math.cos(target_relative_yaw),
                math.sin(target_relative_yaw),
            )
            state.observe_policy(
                policy_call=session.tick,
                task=command_task,
                virtual_position_xy=endpoint_local,
                actual_position_xy=actual_local,
                virtual_facing_yaw=desired_yaw - initial_yaw,
                actual_yaw=selected_yaw - initial_yaw,
                target_velocity_xy=target_velocity,
                target_facing_xy=target_facing,
            )
            state.observe_actor_call(
                policy_call=session.tick,
                task=command_task,
                action=browser_diagnostic_action(outcome),
            )
            if session.tick % 3 == 0:
                publish_frame()
            telemetry_key = (
                outcome.mode.value,
                str(outcome.reason),
                int(outcome.source_clip),
                int(outcome.source_frame),
            )
            if (
                outcome.entered_stairs
                or outcome.exited_stairs
                or telemetry_key != last_telemetry_key
                or session.tick % 50 == 0
            ):
                print(
                    _hybrid_telemetry(session, archive, outcome),
                    flush=True,
                )
            last_telemetry_key = telemetry_key

            next_tick += period
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -0.25:
                next_tick = time.perf_counter()
        return receipt
    finally:
        renderer.close()
        server.close()


def _hybrid_telemetry(
    session: HybridKinematicSession,
    archive: TerrainKinematicArchive,
    outcome: object,
) -> str:
    if int(outcome.source_clip) >= 0:
        source = (
            f"{archive.clip_names[int(outcome.source_clip)]}:"
            f"{int(outcome.source_frame):04d}"
        )
    else:
        result = session.flat_source.last_result
        diagnostics = getattr(result, "diagnostics", None)
        source = (
            "flat"
            if diagnostics is None
            else (
                f"{diagnostics.selected_clip_path}:"
                f"{diagnostics.selected_frame:04d}"
            )
        )
    return (
        f"frame={session.tick:05d} mode={outcome.mode.value} "
        f"source={source} candidates={outcome.candidate_count} "
        f"cost={outcome.cost:.3f} "
        f"planted={outcome.maximum_planted_foot_mismatch_m * 1000:.1f}mm "
        f"enter={int(outcome.entered_stairs)} "
        f"exit={int(outcome.exited_stairs)} reason={outcome.reason}"
    )


def _telemetry(
    session: KinematicTerrainSession,
    archive: TerrainKinematicArchive,
    outcome: object,
) -> str:
    row = session.active_row
    contact = session.database.contact[row]
    tread = session.database.tread_id[row]
    support = "".join(
        name
        for name, active in zip(("L", "R"), contact, strict=True)
        if active
    ) or "flight"
    return (
        f"frame={session.tick:05d} "
        f"source={archive.clip_names[outcome.source_clip]}:"
        f"{outcome.source_frame:04d} support={support} "
        f"tread={tuple(int(value) for value in tread)} "
        f"search={int(outcome.searched)} candidates={outcome.candidate_count} "
        f"cost={outcome.cost:.3f} "
        f"planted={outcome.maximum_planted_foot_mismatch_m * 1000:.1f}mm "
        f"switch={int(outcome.switched)} hold={int(outcome.held)} "
        f"reason={outcome.reason}"
    )


def run_scripted_canary(
    catalog: Path,
    source_zarr: Path,
    *,
    start_clip: str = "up_zero",
    start_frame: int = 0,
    frame_count: int = 300,
    local_velocity_xy: tuple[float, float] = (0.46, 0.0),
) -> dict[str, object]:
    """Exercise the real catalog without a display or MuJoCo viewer."""

    database = load_database(catalog)
    archive = TerrainKinematicArchive.from_zarr(source_zarr)
    if _catalog_clip_names(catalog) != archive.clip_names:
        raise ValueError("catalog and Justin archive clip order differ")
    initial = _initial_row(
        database, archive, start_clip, start_frame
    )
    session = KinematicTerrainSession(database, initial_row=initial)
    ball = TerrainCommandBall()
    _, _, _, initial_yaw = archive.root_pose(
        session.active_clip, session.active_frame
    )
    ball.reset(current_yaw_world=initial_yaw)
    target_velocity = np.asarray(local_velocity_xy, dtype=np.float64)
    maximum_mismatch = 0.0
    maximum_root_jump = 0.0
    cross_clip_events: list[dict[str, object]] = []
    for _ in range(frame_count):
        prior_position, _, _, prior_yaw = archive.root_pose(
            session.active_clip, session.active_frame
        )
        ball.update(
            target_velocity,
            desired_facing_yaw_world=initial_yaw,
            current_yaw_world=prior_yaw,
            dt_s=1.0 / archive.fps,
        )
        future_root, future_facing = ball.preview(
            target_velocity,
            desired_facing_yaw_world=initial_yaw,
            current_yaw_world=prior_yaw,
            offsets_frames=FUTURE_OFFSETS,
            fps=archive.fps,
        )
        stop_requested = bool(
            np.linalg.norm(target_velocity) < 0.01
            and np.linalg.norm(ball.velocity_world) < 0.08
        )
        outcome = session.step(
            future_root,
            future_facing,
            stop_requested=stop_requested,
        )
        position, _, _, _ = archive.root_pose(
            outcome.source_clip, outcome.source_frame
        )
        root_jump = float(np.linalg.norm(position - prior_position))
        maximum_root_jump = max(maximum_root_jump, root_jump)
        maximum_mismatch = max(
            maximum_mismatch,
            outcome.maximum_planted_foot_mismatch_m,
        )
        if outcome.switched:
            cross_clip_events.append(
                {
                    "tick": session.tick,
                    "source_clip": archive.clip_names[outcome.source_clip],
                    "source_frame": outcome.source_frame,
                    "root_jump_m": root_jump,
                    "planted_mismatch_m": (
                        outcome.maximum_planted_foot_mismatch_m
                    ),
                }
            )
    return {
        "kind": "clean_kinematic_terrain_mm_canary_not_physics",
        "frames": frame_count,
        "start": f"{start_clip}:{start_frame}",
        "finish": (
            f"{archive.clip_names[session.active_clip]}:"
            f"{session.active_frame}"
        ),
        "switch_count": session.switch_count,
        "hold_count": session.hold_count,
        "maximum_planted_mismatch_m": maximum_mismatch,
        "maximum_root_jump_m": maximum_root_jump,
        "cross_clip_events": cross_clip_events,
    }


def run_interactive(
    catalog: Path,
    source_zarr: Path,
    scene_xml: Path,
    *,
    start_clip: str = "up_zero",
    start_frame: int = 0,
) -> None:
    """Run the live 50 Hz clean-kinematic viewer."""

    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "DISPLAY is unset; launch this from the graphical slam session"
        )
    import mujoco
    import mujoco.viewer

    database = load_database(catalog)
    archive = TerrainKinematicArchive.from_zarr(source_zarr)
    if _catalog_clip_names(catalog) != archive.clip_names:
        raise ValueError("catalog and Justin archive clip order differ")
    initial = _initial_row(
        database, archive, start_clip, start_frame
    )
    session = KinematicTerrainSession(database, initial_row=initial)
    initial_position, initial_quaternion, initial_joints, initial_yaw = (
        archive.root_pose(session.active_clip, session.active_frame)
    )
    mapper = TerrainKeyboardMapper(initial_yaw_world=initial_yaw)
    ball = TerrainCommandBall()
    ball.reset(current_yaw_world=initial_yaw)

    model = mujoco.MjModel.from_xml_path(str(scene_xml.resolve()))
    data = mujoco.MjData(model)
    data.qpos[:] = build_qpos(
        initial_position, initial_quaternion, initial_joints, model
    )
    mujoco.mj_forward(model, data)

    print("G1 TERRAIN MOTION MATCHING — CLEAN KINEMATICS, NOT PHYSICS")
    print("W/A/S/D: robot-local travel | Shift: slower walk")
    print("Ctrl + arrows: independent facing | arrows: orbit camera")
    print("Q/E: zoom | Space: stop | Backspace: restart | X: exit")
    print("cyan: intended command ball | orange: selected future | red: actual")
    print("search=10 Hz, playback=50 Hz, planted-foot gate=10 mm")

    provider: X11KeyStateProvider | None = None
    actual_trail: deque[np.ndarray] = deque(maxlen=150)
    previous_pressed = frozenset()
    period = 1.0 / archive.fps
    try:
        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=True, show_right_ui=True
        ) as viewer:
            provider = X11KeyStateProvider()
            viewer.cam.lookat[:] = initial_position
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = math.degrees(initial_yaw)
            viewer.cam.elevation = -18.0
            next_tick = time.perf_counter()
            while viewer.is_running():
                levels = provider.sample()
                pressed = levels.pressed if levels.focused else frozenset()
                rising = pressed - previous_pressed
                previous_pressed = pressed
                if "BACKSPACE" in rising:
                    session.reset(initial)
                    ball.reset(current_yaw_world=initial_yaw)
                    actual_trail.clear()
                    print("RESTART -> initial clean kinematic frame")

                query_position, _, _, query_yaw = archive.root_pose(
                    session.active_clip, session.active_frame
                )
                state = normalized_state_from_pressed(pressed)
                intent = mapper.update(
                    state,
                    current_yaw_world=query_yaw,
                    dt_s=period,
                )
                if intent.terminate:
                    break
                ball.update(
                    intent.local_velocity_xy,
                    intent.desired_facing_yaw_world,
                    query_yaw,
                    period,
                )
                future_root, future_facing = ball.preview(
                    intent.local_velocity_xy,
                    intent.desired_facing_yaw_world,
                    query_yaw,
                    FUTURE_OFFSETS,
                    archive.fps,
                )
                stop_requested = bool(
                    intent.stand
                    or (
                        np.linalg.norm(intent.local_velocity_xy) < 0.01
                        and np.linalg.norm(ball.velocity_world) < 0.08
                    )
                )
                outcome = session.step(
                    future_root,
                    future_facing,
                    stop_requested=stop_requested,
                )
                position, quaternion, joints, selected_yaw = archive.root_pose(
                    outcome.source_clip, outcome.source_frame
                )
                data.qpos[:] = build_qpos(
                    position, quaternion, joints, model
                )
                data.time = session.tick / archive.fps
                mujoco.mj_forward(model, data)
                actual_trail.append(
                    np.asarray(
                        (position[0], position[1], position[2] + 0.10),
                        dtype=np.float64,
                    )
                )

                command_world = _trajectory_world(
                    future_root, query_position, query_yaw
                )
                selected_world = _trajectory_world(
                    database.future_root_xy[outcome.row],
                    position,
                    selected_yaw,
                )
                with viewer.lock():
                    viewer.cam.lookat[:] = position
                    if (
                        (state.right_x != 0.0 or state.right_z != 0.0)
                        and not state.strafe
                    ) or state.zoom != 0.0:
                        viewer.cam.azimuth = math.degrees(
                            intent.camera_azimuth_rad
                        )
                        viewer.cam.elevation = -math.degrees(
                            intent.camera_altitude_rad
                        )
                        viewer.cam.distance = intent.camera_distance_m
                    _update_overlay(
                        viewer,
                        command_path_world=command_world,
                        command_facing_local=future_facing,
                        command_root_yaw=query_yaw,
                        selected_path_world=selected_world,
                        selected_facing_local=(
                            database.future_facing_xy[outcome.row]
                        ),
                        selected_root_yaw=selected_yaw,
                        actual_trail=actual_trail,
                        mujoco=mujoco,
                    )
                viewer.sync()
                if (
                    outcome.switched
                    or outcome.held
                    or outcome.searched
                    or session.tick % 25 == 0
                ):
                    print(_telemetry(session, archive, outcome), flush=True)

                next_tick += period
                remaining = next_tick - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                elif remaining < -0.25:
                    next_tick = time.perf_counter()
    finally:
        if provider is not None:
            provider.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean-kinematic G1 terrain motion-matching viewer"
    )
    parser.add_argument(
        "--catalog", type=Path, default=DEFAULT_TERRAIN_CATALOG
    )
    parser.add_argument(
        "--source-zarr", type=Path, default=DEFAULT_JUSTIN_SOURCE
    )
    parser.add_argument(
        "--flat-motions-dir", type=Path, default=DEFAULT_FLAT_MOTIONS
    )
    parser.add_argument("--flat-device", default="cpu")
    parser.add_argument(
        "--scene-xml", type=Path, default=DEFAULT_G1_STAIR_SCENE
    )
    parser.add_argument("--start-clip", default="up_zero")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--frontend",
        choices=("browser", "x11"),
        default="browser",
    )
    parser.add_argument("--browser-host", default="0.0.0.0")
    parser.add_argument("--browser-port", type=int, default=8765)
    parser.add_argument("--browser-token")
    parser.add_argument(
        "--browser-wait-seconds", type=float, default=43200.0
    )
    parser.add_argument(
        "--maximum-pose-repair-joint-delta-rad",
        type=float,
        default=0.10,
        help=(
            "runtime cap on leg-only collision-repair displacement from "
            "the authored pose"
        ),
    )
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=0,
        help="run a scripted real-catalog canary instead of opening a viewer",
    )
    args = parser.parse_args()
    if args.headless_frames < 0:
        parser.error("--headless-frames must be nonnegative")
    if args.headless_frames:
        print(
            json.dumps(
                run_scripted_canary(
                    args.catalog,
                    args.source_zarr,
                    start_clip=args.start_clip,
                    start_frame=args.start_frame,
                    frame_count=args.headless_frames,
                ),
                indent=2,
            )
        )
        return
    if args.frontend == "browser":
        run_browser_interactive(
            args.catalog,
            args.source_zarr,
            args.scene_xml,
            flat_motions_dir=args.flat_motions_dir,
            flat_device=args.flat_device,
            start_clip=args.start_clip,
            start_frame=args.start_frame,
            host=args.browser_host,
            port=args.browser_port,
            token=args.browser_token,
            wait_seconds=args.browser_wait_seconds,
            maximum_pose_repair_joint_delta_rad=(
                args.maximum_pose_repair_joint_delta_rad
            ),
        )
    else:
        run_interactive(
            args.catalog,
            args.source_zarr,
            args.scene_xml,
            start_clip=args.start_clip,
            start_frame=args.start_frame,
        )


if __name__ == "__main__":
    main()
