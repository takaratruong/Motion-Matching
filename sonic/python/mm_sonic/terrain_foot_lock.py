"""Stateful support-aware foot locking for G1 terrain transitions.

This module sits after pose inertialization.  A source-contact rising edge
latches the actual MuJoCo sole-sphere positions in world space.  While contact
remains asserted, a bounded pelvis-height settlement handles an otherwise
unreachable hovering straight leg, then damped least-squares IK moves only
that foot's six leg joints.  The target fades after contact release so the
source swing motion can take over without a one-frame discontinuity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .gear_action import mujoco_to_isaaclab_joint_vector
from .hybrid_terrain_interactive import KinematicPose
from .offline_corpus import BODY_NAMES
from .render_terrain_transition_mesh import MUJOCO_JOINT_NAMES, build_qpos
from .terrain_catalog import (
    JUSTIN_STAIR_RISE_M,
    JUSTIN_STAIR_RUN_M,
    JUSTIN_STAIR_TREADS,
)
from .terrain_scene import StairPlacement


_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_LEG_JOINT_NAMES = (
    (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ),
    (
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ),
)


@dataclass(frozen=True)
class TerrainFootLockResult:
    pose: KinematicPose
    accepted: bool
    repaired: bool
    locked: tuple[bool, bool]
    releasing: tuple[bool, bool]
    sole_world_positions: tuple[np.ndarray, np.ndarray]
    maximum_locked_foot_drift_m: float
    maximum_foot_penetration_m: float
    maximum_forbidden_penetration_m: float
    maximum_joint_delta_rad: float
    iterations: int
    reason: str


@dataclass(frozen=True)
class TerrainFootLockState:
    """Opaque rollback state returned by :meth:`snapshot_state`."""

    foot_targets: tuple[np.ndarray | None, np.ndarray | None]
    release_weights: tuple[float | None, float | None]
    previous_contact: tuple[bool, bool]
    joint_correction: tuple[np.ndarray, np.ndarray]
    joint_correction_initialized: bool


@dataclass(frozen=True)
class TerrainTransitionGuardState:
    """Rollback state for the bounded entry-contact guard."""

    foot_lock_state: object
    entry_frame: int
    entry_lock_active: bool
    entry_contact: tuple[bool, bool] | None


@dataclass(frozen=True)
class _FootState:
    target: np.ndarray
    release_weight: float | None


@dataclass(frozen=True)
class _CollisionState:
    maximum_foot_penetration_m: float
    maximum_forbidden_penetration_m: float


def _descendants(model: object, root_body: int) -> frozenset[int]:
    values: set[int] = set()
    for body_id in range(1, int(model.nbody)):
        cursor = body_id
        while cursor > 0:
            if cursor == root_body:
                values.add(body_id)
                break
            cursor = int(model.body_parentid[cursor])
    return frozenset(values)


class G1TerrainFootLock:
    """Latch support contacts with bounded pelvis settlement and leg IK."""

    def __init__(
        self,
        model: object,
        scene: object,
        *,
        maximum_foot_penetration_m: float = 0.005,
        maximum_locked_foot_drift_m: float = 0.010,
        maximum_joint_correction_rad: float = 0.25,
        maximum_joint_correction_step_rad: float = 0.08,
        maximum_support_snap_down_m: float = 0.020,
        release_halflife_s: float = 0.08,
        maximum_iterations: int = 32,
        damping: float = 0.012,
        posture_weight: float = 0.00005,
    ) -> None:
        import mujoco

        if not callable(getattr(scene, "height_at_world_xy", None)):
            raise ValueError(
                "scene must provide height_at_world_xy for terrain queries"
            )
        values = (
            maximum_foot_penetration_m,
            maximum_locked_foot_drift_m,
            maximum_joint_correction_rad,
            maximum_joint_correction_step_rad,
            maximum_support_snap_down_m,
            release_halflife_s,
            damping,
        )
        if (
            not all(math.isfinite(float(value)) and float(value) > 0.0
                    for value in values)
            or int(maximum_iterations) <= 0
            or not math.isfinite(float(posture_weight))
            or float(posture_weight) < 0.0
        ):
            raise ValueError("terrain foot-lock parameters are invalid")

        self.model = model
        self.scene = scene
        self.maximum_foot_penetration_m = float(
            maximum_foot_penetration_m
        )
        self.maximum_locked_foot_drift_m = float(
            maximum_locked_foot_drift_m
        )
        self.maximum_joint_correction_rad = float(
            maximum_joint_correction_rad
        )
        self.maximum_joint_correction_step_rad = float(
            maximum_joint_correction_step_rad
        )
        self.maximum_support_snap_down_m = float(
            maximum_support_snap_down_m
        )
        self.release_halflife_s = float(release_halflife_s)
        self.maximum_iterations = int(maximum_iterations)
        self.damping = float(damping)
        self.posture_weight = float(posture_weight)

        self._terrain_geoms = frozenset(
            geom_id
            for geom_id in range(int(model.ngeom))
            if (
                str(model.geom(geom_id).name) in {"ground", "floor"}
                or str(model.geom(geom_id).name).startswith("terrain_")
                or str(model.geom(geom_id).name).startswith("course_")
            )
        )
        if not self._terrain_geoms:
            raise ValueError("foot-lock model contains no terrain geometry")

        sphere_type = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        foot_geoms: list[frozenset[int]] = []
        sphere_geoms: list[tuple[int, ...]] = []
        for name in _FOOT_BODY_NAMES:
            body_id = int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, name
                )
            )
            if body_id < 0:
                raise ValueError(f"foot-lock model is missing body {name}")
            descendants = _descendants(model, body_id)
            geoms = frozenset(
                geom_id
                for geom_id in range(int(model.ngeom))
                if int(model.geom_bodyid[geom_id]) in descendants
            )
            spheres = tuple(
                geom_id
                for geom_id in sorted(geoms)
                if int(model.geom_type[geom_id]) == sphere_type
            )
            if not geoms or not spheres:
                raise ValueError(
                    f"foot-lock model has no sole spheres below {name}"
                )
            foot_geoms.append(geoms)
            sphere_geoms.append(spheres)
        if foot_geoms[0] & foot_geoms[1]:
            raise ValueError("left and right foot geometry overlap")
        self._foot_geoms = (foot_geoms[0], foot_geoms[1])
        self._sphere_geoms = (sphere_geoms[0], sphere_geoms[1])

        leg_qpos: list[np.ndarray] = []
        leg_dofs: list[np.ndarray] = []
        leg_limits: list[np.ndarray] = []
        for names in _LEG_JOINT_NAMES:
            ids = np.asarray(
                [
                    mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                    for name in names
                ],
                dtype=np.int32,
            )
            if np.any(ids < 0):
                raise ValueError("foot-lock model is missing a G1 leg joint")
            leg_qpos.append(
                np.asarray(model.jnt_qposadr[ids], dtype=np.int32)
            )
            leg_dofs.append(
                np.asarray(model.jnt_dofadr[ids], dtype=np.int32)
            )
            leg_limits.append(
                np.asarray(model.jnt_range[ids], dtype=np.float64)
            )
        self._leg_qpos = (leg_qpos[0], leg_qpos[1])
        self._leg_dofs = (leg_dofs[0], leg_dofs[1])
        self._leg_limits = (leg_limits[0], leg_limits[1])

        self._body_ids = tuple(
            int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, name
                )
            )
            for name in BODY_NAMES
        )
        self._joint_ids_mujoco_order = tuple(
            int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
            )
            for name in MUJOCO_JOINT_NAMES
        )
        if (
            any(body < 0 for body in self._body_ids)
            or any(joint < 0 for joint in self._joint_ids_mujoco_order)
        ):
            raise ValueError("foot-lock model lacks the native G1 layout")
        free = np.flatnonzero(
            np.asarray(model.jnt_type)
            == int(mujoco.mjtJoint.mjJNT_FREE)
        )
        if free.size != 1:
            raise ValueError("foot-lock model must contain one free joint")
        self._root_qpos_address = int(
            model.jnt_qposadr[int(free[0])]
        )
        self._data = mujoco.MjData(model)
        self.reset()

    def reset(self) -> None:
        """Forget all contact-edge and release state."""

        self._states: tuple[_FootState | None, _FootState | None] = (
            None,
            None,
        )
        self._previous_contact = (False, False)
        self._joint_correction = (
            np.zeros(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
        )
        self._joint_correction_initialized = False

    @staticmethod
    def _snapshot_target(target: np.ndarray | None) -> np.ndarray | None:
        if target is None:
            return None
        copied = np.array(target, dtype=np.float64, copy=True, order="C")
        copied.setflags(write=False)
        return copied

    def snapshot_state(self) -> TerrainFootLockState:
        """Return a detached immutable snapshot of all temporal state."""

        return TerrainFootLockState(
            foot_targets=(
                self._snapshot_target(
                    None
                    if self._states[0] is None
                    else self._states[0].target
                ),
                self._snapshot_target(
                    None
                    if self._states[1] is None
                    else self._states[1].target
                ),
            ),
            release_weights=(
                None
                if self._states[0] is None
                else self._states[0].release_weight,
                None
                if self._states[1] is None
                else self._states[1].release_weight,
            ),
            previous_contact=self._previous_contact,
            joint_correction=(
                self._snapshot_target(self._joint_correction[0]),
                self._snapshot_target(self._joint_correction[1]),
            ),
            joint_correction_initialized=bool(
                self._joint_correction_initialized
            ),
        )

    def restore_state(self, snapshot: TerrainFootLockState) -> None:
        """Restore a snapshot without retaining references to its arrays."""

        if not isinstance(snapshot, TerrainFootLockState):
            raise ValueError(
                "snapshot must be a TerrainFootLockState"
            )
        if (
            len(snapshot.foot_targets) != 2
            or len(snapshot.release_weights) != 2
            or len(snapshot.previous_contact) != 2
            or len(snapshot.joint_correction) != 2
            or type(snapshot.joint_correction_initialized) is not bool
            or any(type(value) is not bool
                   for value in snapshot.previous_contact)
        ):
            raise ValueError("terrain foot-lock snapshot is malformed")

        restored: list[_FootState | None] = [None, None]
        for foot in range(2):
            target = snapshot.foot_targets[foot]
            weight = snapshot.release_weights[foot]
            if target is None:
                if weight is not None:
                    raise ValueError(
                        "snapshot release weight has no foot target"
                    )
                continue
            value = np.asarray(target)
            expected = (len(self._sphere_geoms[foot]), 3)
            if (
                value.shape != expected
                or value.dtype.kind not in "iuf"
                or not np.all(np.isfinite(value))
            ):
                raise ValueError(
                    "snapshot foot target is incompatible with this model"
                )
            if weight is not None and (
                not math.isfinite(float(weight))
                or not 0.0 <= float(weight) <= 1.0
            ):
                raise ValueError("snapshot release weight is invalid")
            restored[foot] = _FootState(
                np.array(value, dtype=np.float64, copy=True, order="C"),
                None if weight is None else float(weight),
            )

        self._states = (restored[0], restored[1])
        self._previous_contact = (
            snapshot.previous_contact[0],
            snapshot.previous_contact[1],
        )
        corrections: list[np.ndarray] = []
        for value in snapshot.joint_correction:
            correction = np.asarray(value)
            if (
                correction.shape != (6,)
                or correction.dtype.kind not in "iuf"
                or not np.isfinite(correction).all()
                or float(np.max(np.abs(correction)))
                > self.maximum_joint_correction_rad + 1.0e-6
            ):
                raise ValueError(
                    "snapshot joint correction is incompatible"
                )
            corrections.append(
                np.array(
                    correction,
                    dtype=np.float64,
                    copy=True,
                    order="C",
                )
            )
        self._joint_correction = (
            corrections[0],
            corrections[1],
        )
        self._joint_correction_initialized = (
            snapshot.joint_correction_initialized
        )

    def _sole_positions(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(
                [
                    self._data.geom_xpos[geom_id]
                    for geom_id in self._sphere_geoms[0]
                ],
                dtype=np.float64,
            ).copy(),
            np.asarray(
                [
                    self._data.geom_xpos[geom_id]
                    for geom_id in self._sphere_geoms[1]
                ],
                dtype=np.float64,
            ).copy(),
        )

    def _collision_state(self) -> _CollisionState:
        import mujoco

        mujoco.mj_forward(self.model, self._data)
        mujoco.mj_collision(self.model, self._data)
        foot_penetration = 0.0
        forbidden = 0.0
        for index in range(int(self._data.ncon)):
            contact = self._data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in self._terrain_geoms:
                robot_geom = geom2
            elif geom2 in self._terrain_geoms:
                robot_geom = geom1
            else:
                continue
            if robot_geom in self._terrain_geoms:
                continue
            depth = max(0.0, -float(contact.dist))
            if (
                robot_geom in self._foot_geoms[0]
                or robot_geom in self._foot_geoms[1]
            ):
                foot_penetration = max(foot_penetration, depth)
            else:
                forbidden = max(forbidden, depth)
        return _CollisionState(foot_penetration, forbidden)

    def support_contact(
        self,
        pose: KinematicPose,
        *,
        maximum_sole_clearance_m: float = 0.020,
    ) -> np.ndarray:
        """Measure which feet actually support the entry pose.

        At a motion splice, the inertialized pose is still the outgoing flat
        pose even though the incoming clip already carries its own contact
        phase.  Latching the incoming label can therefore lock the wrong foot.
        Use exact contacts plus a small sole-clearance margin on the pose that
        will really be published.
        """

        import mujoco

        if not isinstance(pose, KinematicPose):
            raise ValueError("pose must be a KinematicPose")
        clearance_limit = float(maximum_sole_clearance_m)
        if (
            not math.isfinite(clearance_limit)
            or clearance_limit < 0.0
        ):
            raise ValueError(
                "maximum_sole_clearance_m must be finite and nonnegative"
            )
        self._data.qpos[:] = build_qpos(
            pose.root_position_world,
            pose.root_orientation_world_xyzw,
            pose.joint_position,
            self.model,
        )
        mujoco.mj_forward(self.model, self._data)
        mujoco.mj_collision(self.model, self._data)
        supported = [False, False]
        for index in range(int(self._data.ncon)):
            contact = self._data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in self._terrain_geoms:
                robot_geom = geom2
            elif geom2 in self._terrain_geoms:
                robot_geom = geom1
            else:
                continue
            for foot in range(2):
                if robot_geom in self._foot_geoms[foot]:
                    supported[foot] = True
        # Exact collision support wins.  Promoting a nearby swing foot as a
        # second support changes the contact topology and can select the wrong
        # handoff phase.
        if any(supported):
            return np.asarray(supported, dtype=bool)

        sole = self._sole_positions()
        clearances = np.full(2, math.inf, dtype=np.float64)
        for foot in range(2):
            for geom_id, center in zip(
                self._sphere_geoms[foot], sole[foot], strict=True
            ):
                terrain_height = self.scene.height_at_world_xy(
                    center[:2]
                )
                clearance = (
                    float(center[2])
                    - float(self.model.geom_size[geom_id, 0])
                    - terrain_height
                )
                clearances[foot] = min(clearances[foot], clearance)
        # A collision-free kinematic pose can hover slightly because the
        # source skeleton and collision feet are not identical.  In that case
        # choose only the nearest sole as the support hypothesis; declaring
        # both feet supported is exactly the phase bug this fallback exists to
        # avoid.
        nearest = int(np.argmin(clearances))
        if clearances[nearest] <= clearance_limit:
            supported[nearest] = True
        return np.asarray(supported, dtype=bool)

    def _exact_support_contact_for_current_pose(self) -> np.ndarray:
        """Return terrain contacts for the two executable sole geometries."""

        import mujoco

        mujoco.mj_forward(self.model, self._data)
        mujoco.mj_collision(self.model, self._data)
        supported = [False, False]
        for index in range(int(self._data.ncon)):
            contact = self._data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in self._terrain_geoms:
                robot_geom = geom2
            elif geom2 in self._terrain_geoms:
                robot_geom = geom1
            else:
                continue
            for foot in range(2):
                if robot_geom in self._foot_geoms[foot]:
                    supported[foot] = True
        return np.asarray(supported, dtype=bool)

    def exact_support_contact(self, pose: KinematicPose) -> np.ndarray:
        """Return only real MuJoCo sole/terrain contacts, with no near fallback."""

        if not isinstance(pose, KinematicPose):
            raise ValueError("pose must be a KinematicPose")
        self._data.qpos[:] = build_qpos(
            pose.root_position_world,
            pose.root_orientation_world_xyzw,
            pose.joint_position,
            self.model,
        )
        return self._exact_support_contact_for_current_pose()

    def landing_stance_contact(
        self,
        pose: KinematicPose,
        *,
        maximum_foot_speed_mps: float = 0.20,
        maximum_sole_clearance_m: float = 0.020,
        previous_pose: KinematicPose | None = None,
        dt_s: float | None = None,
    ) -> np.ndarray:
        """Infer genuine landing stance from support proximity and foot speed.

        Collision alone is insufficient: a moving swing foot can scrape along
        the top platform for many frames and would then be mistaken for a
        planted foot.  Conversely, clean kinematics can hover a few
        millimetres at touchdown.  A foot is stance only when it is slow in
        world space and either exactly contacting or within a small support
        clearance.
        """

        if not isinstance(pose, KinematicPose):
            raise ValueError("pose must be a KinematicPose")
        speed_limit = float(maximum_foot_speed_mps)
        clearance_limit = float(maximum_sole_clearance_m)
        if (previous_pose is None) != (dt_s is None):
            raise ValueError(
                "previous_pose and dt_s must be provided together"
            )
        if previous_pose is not None and not isinstance(
            previous_pose, KinematicPose
        ):
            raise ValueError("previous_pose must be a KinematicPose")
        step = None if dt_s is None else float(dt_s)
        if (
            not math.isfinite(speed_limit)
            or speed_limit <= 0.0
            or not math.isfinite(clearance_limit)
            or clearance_limit < 0.0
            or (
                step is not None
                and (not math.isfinite(step) or step <= 0.0)
            )
        ):
            raise ValueError("landing stance thresholds are invalid")
        self._data.qpos[:] = build_qpos(
            pose.root_position_world,
            pose.root_orientation_world_xyzw,
            pose.joint_position,
            self.model,
        )
        exact = self._exact_support_contact_for_current_pose()
        sole = self._sole_positions()
        result = np.zeros(2, dtype=bool)
        speeds = np.linalg.norm(
            np.asarray(
                pose.body_linear_velocity_world[[18, 19]],
                dtype=np.float64,
            ),
            axis=1,
        )
        if previous_pose is not None:
            measured_speed = np.linalg.norm(
                (
                    pose.body_position_world[[18, 19]]
                    - previous_pose.body_position_world[[18, 19]]
                )
                / float(step),
                axis=1,
            )
            # Pose inertialization currently carries source velocity fields
            # but does not include the derivative of its decaying positional
            # offset.  The finite difference is therefore the authoritative
            # guard against classifying a visibly moving foot as planted.
            speeds = np.maximum(speeds, measured_speed)
        for foot in range(2):
            minimum_clearance = math.inf
            for geom_id, center in zip(
                self._sphere_geoms[foot], sole[foot], strict=True
            ):
                clearance = (
                    float(center[2])
                    - float(self.model.geom_size[geom_id, 0])
                    - self.scene.height_at_world_xy(center[:2])
                )
                minimum_clearance = min(minimum_clearance, clearance)
            result[foot] = bool(
                speeds[foot] <= speed_limit
                and (
                    bool(exact[foot])
                    or minimum_clearance <= clearance_limit
                )
            )
        return result

    def landing_support(
        self,
        pose: KinematicPose,
        placement: StairPlacement,
        *,
        side: str = "top",
        maximum_sole_clearance_m: float = 0.015,
        boundary_tolerance_m: float = 0.002,
    ) -> np.ndarray:
        """Return which complete soles are genuinely supported on the landing.

        A double-support label on an endpoint tread is not yet a safe
        stair-to-flat handoff: a flat gait can immediately move its trailing
        foot back over the exposed edge.  For ascent, every sphere footprint
        must lie beyond the top-tread boundary on the raised platform.  For
        descent, every footprint must lie before the first-riser boundary on
        the ground.  The placement is reconstructed from the causal
        robot-local terrain observation; no externally measured root
        translation or global navigation map is part of the interface.
        """

        import mujoco

        if not isinstance(pose, KinematicPose):
            raise ValueError("pose must be a KinematicPose")
        if not isinstance(placement, StairPlacement):
            raise ValueError("placement must be a StairPlacement")
        landing_side = str(side)
        if landing_side not in {"top", "ground"}:
            raise ValueError("landing side must be 'top' or 'ground'")
        clearance_limit = float(maximum_sole_clearance_m)
        tolerance = float(boundary_tolerance_m)
        if (
            not math.isfinite(clearance_limit)
            or clearance_limit < 0.0
            or not math.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("landing thresholds must be finite and nonnegative")

        self._data.qpos[:] = build_qpos(
            pose.root_position_world,
            pose.root_orientation_world_xyzw,
            pose.joint_position,
            self.model,
        )
        mujoco.mj_forward(self.model, self._data)
        exact_support = self._exact_support_contact_for_current_pose()
        sole = self._sole_positions()
        top_boundary = JUSTIN_STAIR_RUN_M * JUSTIN_STAIR_TREADS
        support_height = (
            float(placement.origin_world_xyz[2])
            + (
                JUSTIN_STAIR_RISE_M * JUSTIN_STAIR_TREADS
                if landing_side == "top"
                else 0.0
            )
        )
        supported = np.zeros(2, dtype=bool)
        for foot in range(2):
            if not bool(exact_support[foot]):
                continue
            minimum_clearance = math.inf
            valid = True
            for geom_id, center in zip(
                self._sphere_geoms[foot], sole[foot], strict=True
            ):
                radius = float(self.model.geom_size[geom_id, 0])
                local = placement.world_to_stair_xyz(center)
                complete_footprint = (
                    float(local[0]) - radius
                    >= top_boundary - tolerance
                    if landing_side == "top"
                    else float(local[0]) + radius <= tolerance
                )
                if not complete_footprint:
                    valid = False
                    break
                terrain_height = self.scene.height_at_world_xy(center[:2])
                if abs(terrain_height - support_height) > 0.010:
                    valid = False
                    break
                clearance = float(center[2]) - radius - terrain_height
                if clearance < -self.maximum_foot_penetration_m - 1.0e-6:
                    valid = False
                    break
                minimum_clearance = min(minimum_clearance, clearance)
            # Heel/toe roll leaves some spheres above the plane even in a
            # valid stance.  At least one sole sample under each complete
            # footprint must be close enough to support; requiring every
            # sample to be close delayed handoff for an entire platform.
            supported[foot] = bool(
                valid and minimum_clearance <= clearance_limit
            )
        return supported

    def landing_complete(
        self,
        pose: KinematicPose,
        placement: StairPlacement,
        *,
        side: str = "top",
        maximum_sole_clearance_m: float = 0.015,
        boundary_tolerance_m: float = 0.002,
    ) -> bool:
        """Return whether both complete soles currently support the landing."""

        return bool(
            np.all(
                self.landing_support(
                    pose,
                    placement,
                    side=side,
                    maximum_sole_clearance_m=maximum_sole_clearance_m,
                    boundary_tolerance_m=boundary_tolerance_m,
                )
            )
        )

    def _proposed_states(
        self,
        contact: tuple[bool, bool],
        sole: tuple[np.ndarray, np.ndarray],
        dt_s: float,
    ) -> tuple[_FootState | None, _FootState | None]:
        decay = math.exp(
            -math.log(2.0) * dt_s / self.release_halflife_s
        )
        result: list[_FootState | None] = [None, None]
        for foot in range(2):
            old = self._states[foot]
            if contact[foot]:
                if not self._previous_contact[foot] or old is None:
                    result[foot] = _FootState(
                        self._project_support_target(
                            foot, sole[foot]
                        ),
                        None,
                    )
                else:
                    result[foot] = _FootState(old.target.copy(), None)
                continue
            if self._previous_contact[foot] and old is not None:
                result[foot] = _FootState(old.target.copy(), 1.0)
                continue
            if old is not None and old.release_weight is not None:
                weight = old.release_weight * decay
                if weight >= 0.01:
                    result[foot] = _FootState(old.target.copy(), weight)
        return (result[0], result[1])

    def _project_support_target(
        self, foot: int, sole_centers_world: np.ndarray
    ) -> np.ndarray:
        """Project a newly latched rigid sole target onto terrain support.

        Retargeted clips contain both small penetrations and small hovers.
        The rigid vertical shift that leaves every sole sphere outside the
        terrain while putting the closest one exactly on its support surface
        is the maximum per-sphere required shift.  Downward projection is
        bounded so an erroneous swing-foot label cannot pull a high foot to
        the ground.
        """

        target = np.asarray(
            sole_centers_world, dtype=np.float64
        ).copy()
        required_shifts: list[float] = []
        for geom_id, center in zip(
            self._sphere_geoms[foot], target, strict=True
        ):
            required_center_z = (
                self.scene.height_at_world_xy(center[:2])
                + float(self.model.geom_size[geom_id, 0])
            )
            required_shifts.append(
                required_center_z - float(center[2])
            )
        vertical_shift = max(required_shifts)
        vertical_shift = max(
            vertical_shift, -self.maximum_support_snap_down_m
        )
        target[:, 2] += vertical_shift
        return np.ascontiguousarray(target)

    def _solve_foot(
        self,
        foot: int,
        target: np.ndarray,
        reference_qpos: np.ndarray,
    ) -> int:
        import mujoco

        data = self._data
        dofs = self._leg_dofs[foot]
        addresses = self._leg_qpos[foot]
        limits = self._leg_limits[foot]
        reference = reference_qpos[addresses].copy()
        identity = np.eye(len(dofs), dtype=np.float64)
        jacobian_position = np.zeros(
            (3, int(self.model.nv)), dtype=np.float64
        )
        iterations = 0
        for iteration in range(self.maximum_iterations):
            mujoco.mj_forward(self.model, data)
            current = self._sole_positions()[foot]
            error = (target - current).reshape(-1)
            if float(np.max(np.linalg.norm(
                (target - current), axis=1
            ))) <= 2.5e-4:
                break
            rows: list[np.ndarray] = []
            for geom_id in self._sphere_geoms[foot]:
                point = np.asarray(
                    data.geom_xpos[geom_id], dtype=np.float64
                )
                body_id = int(self.model.geom_bodyid[geom_id])
                jacobian_position.fill(0.0)
                mujoco.mj_jac(
                    self.model,
                    data,
                    jacobian_position,
                    None,
                    point,
                    body_id,
                )
                rows.append(jacobian_position[:, dofs].copy())
            jacobian = np.vstack(rows)
            transpose = jacobian.T
            lhs = (
                transpose @ jacobian
                + (self.damping ** 2 + self.posture_weight) * identity
            )
            rhs = (
                transpose @ error
                + self.posture_weight
                * (reference - data.qpos[addresses])
            )
            delta = np.linalg.solve(lhs, rhs)
            delta = np.clip(delta, -0.04, 0.04)
            correction = np.clip(
                data.qpos[addresses] + delta - reference,
                -self.maximum_joint_correction_rad,
                self.maximum_joint_correction_rad,
            )
            data.qpos[addresses] = np.clip(
                reference + correction,
                limits[:, 0] + 1.0e-6,
                limits[:, 1] - 1.0e-6,
            )
            iterations = iteration + 1
        mujoco.mj_forward(self.model, data)
        return iterations

    def _pose_from_data(self, source: KinematicPose) -> KinematicPose:
        data = self._data
        mapped = np.asarray(
            [
                data.qpos[int(self.model.jnt_qposadr[joint_id])]
                for joint_id in self._joint_ids_mujoco_order
            ],
            dtype=np.float64,
        )
        joints = mujoco_to_isaaclab_joint_vector(mapped).astype(np.float32)
        body_position = np.asarray(
            [data.xpos[body_id] for body_id in self._body_ids],
            dtype=np.float32,
        )
        body_wxyz = np.asarray(
            [data.xquat[body_id] for body_id in self._body_ids],
            dtype=np.float32,
        )
        root = self._root_qpos_address
        root_wxyz = np.asarray(
            data.qpos[root + 3 : root + 7], dtype=np.float32
        )
        return KinematicPose(
            root_position_world=np.asarray(
                data.qpos[root : root + 3], dtype=np.float32
            ),
            root_orientation_world_xyzw=root_wxyz[[1, 2, 3, 0]],
            joint_position=joints,
            joint_velocity=source.joint_velocity.copy(),
            body_position_world=body_position,
            body_orientation_world_xyzw=body_wxyz[:, [1, 2, 3, 0]],
            body_linear_velocity_world=(
                source.body_linear_velocity_world.copy()
            ),
            body_angular_velocity_world=(
                source.body_angular_velocity_world.copy()
            ),
        )

    def apply(
        self,
        pose: KinematicPose,
        source_contact: object,
        *,
        dt_s: float,
        minimum_swing_clearance_m: float = 0.0,
    ) -> TerrainFootLockResult:
        """Apply one transactional contact-lock update."""

        import mujoco

        if not isinstance(pose, KinematicPose):
            raise ValueError("pose must be a KinematicPose")
        contact_array = np.asarray(source_contact)
        if contact_array.shape != (2,) or contact_array.dtype.kind != "b":
            raise ValueError("source_contact must contain two booleans")
        step = float(dt_s)
        swing_clearance = float(minimum_swing_clearance_m)
        if (
            not math.isfinite(step)
            or step <= 0.0
            or not math.isfinite(swing_clearance)
            or swing_clearance < 0.0
        ):
            raise ValueError(
                "dt_s must be positive and swing clearance nonnegative"
            )
        contact = (bool(contact_array[0]), bool(contact_array[1]))
        reference_qpos = build_qpos(
            pose.root_position_world,
            pose.root_orientation_world_xyzw,
            pose.joint_position,
            self.model,
        )
        self._data.qpos[:] = reference_qpos
        mujoco.mj_forward(self.model, self._data)
        initial_sole = self._sole_positions()
        proposed = self._proposed_states(contact, initial_sole, step)

        # A nearly straight authored stance leg cannot extend to remove even
        # a few millimetres of retargeting hover.  Settle root height by the
        # common downward component before asking fixed-root leg IK to handle
        # the remaining position/orientation error.  Root XY and orientation
        # remain untouched, and the bounded shift is recomputed from the
        # measured robot/terrain geometry rather than any global stair pose.
        downward_shifts = [
            float(np.median(
                proposed[foot].target[:, 2] - initial_sole[foot][:, 2]
            ))
            for foot in range(2)
            if (
                contact[foot]
                and proposed[foot] is not None
                and float(np.median(
                    proposed[foot].target[:, 2]
                    - initial_sole[foot][:, 2]
                )) < 0.0
            )
        ]
        if downward_shifts:
            root_shift = float(np.median(downward_shifts))
            self._data.qpos[self._root_qpos_address + 2] += max(
                root_shift, -self.maximum_support_snap_down_m
            )
            mujoco.mj_forward(self.model, self._data)
            initial_sole = self._sole_positions()

        iterations = 0
        solve_targets: list[np.ndarray | None] = [None, None]
        for foot, state in enumerate(proposed):
            if state is None:
                continue
            if state.release_weight is None:
                target = state.target
            else:
                target = (
                    initial_sole[foot]
                    + state.release_weight
                    * (state.target - initial_sole[foot])
                )
            solve_targets[foot] = target
            iterations += self._solve_foot(
                foot, target, reference_qpos
            )

        # A moving swing foot that merely intersects the landing must not be
        # reclassified as stance and allowed to skate.  Raise the complete
        # sole proxy above the local terrain envelope while preserving the
        # authored horizontal path and all non-leg joints.
        if swing_clearance > 0.0:
            mujoco.mj_forward(self.model, self._data)
            current_sole = self._sole_positions()
            for foot in range(2):
                if contact[foot]:
                    continue
                required_lift = 0.0
                for geom_id, center in zip(
                    self._sphere_geoms[foot],
                    current_sole[foot],
                    strict=True,
                ):
                    clearance = (
                        float(center[2])
                        - float(self.model.geom_size[geom_id, 0])
                        - self.scene.height_at_world_xy(center[:2])
                    )
                    required_lift = max(
                        required_lift,
                        swing_clearance - clearance,
                    )
                if required_lift <= 0.0:
                    continue
                target = current_sole[foot].copy()
                target[:, 2] += required_lift
                iterations += self._solve_foot(
                    foot, target, reference_qpos
                )

        # Bound the *change in IK offset* rather than the generated source
        # motion itself.  Independent per-frame IK solutions can otherwise
        # jump between two equally valid leg configurations when a foot
        # changes contact regime, even though both individual corrections are
        # within the absolute 0.25-rad budget.
        proposed_joint_correction: list[np.ndarray] = []
        for foot in range(2):
            addresses = self._leg_qpos[foot]
            desired = (
                self._data.qpos[addresses] - reference_qpos[addresses]
            )
            bounded = (
                desired
                if not self._joint_correction_initialized
                else (
                    self._joint_correction[foot]
                    + np.clip(
                        desired - self._joint_correction[foot],
                        -self.maximum_joint_correction_step_rad,
                        self.maximum_joint_correction_step_rad,
                    )
                )
            )
            bounded = np.clip(
                bounded,
                -self.maximum_joint_correction_rad,
                self.maximum_joint_correction_rad,
            )
            self._data.qpos[addresses] = np.clip(
                reference_qpos[addresses] + bounded,
                self._leg_limits[foot][:, 0] + 1.0e-6,
                self._leg_limits[foot][:, 1] - 1.0e-6,
            )
            proposed_joint_correction.append(
                np.ascontiguousarray(bounded, dtype=np.float64)
            )

        collision = self._collision_state()
        final_sole = self._sole_positions()
        locked_drift = 0.0
        for foot, active in enumerate(contact):
            if active and proposed[foot] is not None:
                locked_drift = max(
                    locked_drift,
                    float(np.max(np.linalg.norm(
                        final_sole[foot] - proposed[foot].target,
                        axis=1,
                    ))),
                )
        candidate = self._pose_from_data(pose)
        joint_delta = float(np.max(np.abs(
            candidate.joint_position.astype(np.float64)
            - pose.joint_position.astype(np.float64)
        )))

        reasons: list[str] = []
        tolerance = 1.0e-6
        if (
            collision.maximum_foot_penetration_m
            > self.maximum_foot_penetration_m + tolerance
        ):
            reasons.append(
                "sole penetration exceeds "
                f"{1000.0 * self.maximum_foot_penetration_m:.1f} mm"
            )
        if collision.maximum_forbidden_penetration_m > 0.0:
            reasons.append("forbidden body collides with terrain")
        if (
            locked_drift
            > self.maximum_locked_foot_drift_m + tolerance
        ):
            reasons.append(
                "locked-foot drift exceeds "
                f"{1000.0 * self.maximum_locked_foot_drift_m:.1f} mm"
            )
        if (
            joint_delta
            > self.maximum_joint_correction_rad + tolerance
        ):
            reasons.append("joint correction exceeds 0.25 rad")
        accepted = not reasons
        if accepted:
            self._states = proposed
            self._previous_contact = contact
            self._joint_correction = (
                proposed_joint_correction[0],
                proposed_joint_correction[1],
            )
            self._joint_correction_initialized = True
        return TerrainFootLockResult(
            pose=candidate if accepted else pose,
            accepted=accepted,
            repaired=accepted and joint_delta > 1.0e-8,
            locked=(
                accepted and contact[0] and proposed[0] is not None,
                accepted and contact[1] and proposed[1] is not None,
            ),
            releasing=(
                accepted
                and proposed[0] is not None
                and proposed[0].release_weight is not None,
                accepted
                and proposed[1] is not None
                and proposed[1].release_weight is not None,
            ),
            sole_world_positions=final_sole,
            maximum_locked_foot_drift_m=locked_drift,
            maximum_foot_penetration_m=(
                collision.maximum_foot_penetration_m
            ),
            maximum_forbidden_penetration_m=(
                collision.maximum_forbidden_penetration_m
            ),
            maximum_joint_delta_rad=joint_delta,
            iterations=iterations,
            reason="accepted" if accepted else "; ".join(reasons),
        )


class G1TerrainTransitionGuard:
    """Combine exact pose repair with a bounded entry foot lock.

    Motion matching supplies the authored gait.  At a flat-to-stair splice,
    the selected clip can place its currently planted foot several centimetres
    from the flat source's planted foot.  The existing pose inertializer
    smooths joints and root, while this guard preserves the actual sole
    contact for a short fixed window and then forces the lock to fade.  It
    deliberately disables itself after release so imperfect contact labels
    cannot pin a foot for an entire staircase.
    """

    def __init__(
        self,
        pose_repairer: object,
        foot_locker: object,
        *,
        contact_hold_frames: int = 4,
        track_source_contacts: bool = False,
        source_contact_delay_frames: int = 30,
        dt_s: float = 1.0 / 50.0,
        landing_maximum_foot_speed_mps: float = 0.20,
        landing_maximum_sole_clearance_m: float = 0.020,
        landing_contact_acquire_frames: int = 2,
        landing_minimum_swing_clearance_m: float = 0.012,
    ) -> None:
        landing_speed = float(landing_maximum_foot_speed_mps)
        landing_clearance = float(landing_maximum_sole_clearance_m)
        swing_clearance = float(landing_minimum_swing_clearance_m)
        if (
            type(contact_hold_frames) is not int
            or contact_hold_frames <= 0
            or type(track_source_contacts) is not bool
            or type(source_contact_delay_frames) is not int
            or source_contact_delay_frames < contact_hold_frames
            or not math.isfinite(float(dt_s))
            or float(dt_s) <= 0.0
            or not math.isfinite(landing_speed)
            or landing_speed <= 0.0
            or not math.isfinite(landing_clearance)
            or landing_clearance < 0.0
            or type(landing_contact_acquire_frames) is not int
            or landing_contact_acquire_frames <= 0
            or not math.isfinite(swing_clearance)
            or swing_clearance < 0.0
            or not callable(getattr(pose_repairer, "repair", None))
            or not callable(getattr(foot_locker, "apply", None))
            or not callable(getattr(foot_locker, "reset", None))
            or not callable(
                getattr(foot_locker, "snapshot_state", None)
            )
            or not callable(
                getattr(foot_locker, "restore_state", None)
            )
        ):
            raise ValueError("terrain transition guard inputs are invalid")
        self.pose_repairer = pose_repairer
        self.foot_locker = foot_locker
        self.contact_hold_frames = contact_hold_frames
        self.track_source_contacts = track_source_contacts
        self.source_contact_delay_frames = source_contact_delay_frames
        self.dt_s = float(dt_s)
        self.landing_maximum_foot_speed_mps = landing_speed
        self.landing_maximum_sole_clearance_m = landing_clearance
        self.landing_contact_acquire_frames = landing_contact_acquire_frames
        self.landing_minimum_swing_clearance_m = swing_clearance
        self.last_pose_repair_result: object | None = None
        self.last_foot_lock_result: object | None = None
        self.end_stair()

    @property
    def entry_frame(self) -> int:
        return self._entry_frame

    @property
    def entry_lock_active(self) -> bool:
        return self._entry_lock_active

    @property
    def measured_stance_contact(self) -> tuple[bool, bool]:
        """Return the last raw measured landing-stance decision."""

        return self._landing_measured_contact

    @property
    def ground_proximity_contact(self) -> tuple[bool, bool]:
        """Return the last speed-independent measured support proximity."""

        return self._landing_ground_proximity_contact

    @property
    def trusted_landing_contact(self) -> tuple[bool, bool]:
        """Return the contact state currently authorized for locking."""

        return self._landing_contact

    @property
    def landing_filter_outcome(self) -> str:
        """Return the last continuous landing-filter outcome."""

        return self._landing_filter_outcome

    @property
    def landing_filter_reason(self) -> str:
        """Return the last continuous landing-filter diagnostic reason."""

        return self._landing_filter_reason

    def support_contact(
        self,
        pose: KinematicPose,
        *,
        maximum_sole_clearance_m: float | None = None,
    ) -> np.ndarray:
        """Expose the measured support phase for entry matching."""

        method = getattr(self.foot_locker, "support_contact", None)
        if not callable(method):
            raise ValueError(
                "terrain transition foot locker cannot measure support"
            )
        contact = np.asarray(
            method(pose)
            if maximum_sole_clearance_m is None
            else method(
                pose,
                maximum_sole_clearance_m=maximum_sole_clearance_m,
            )
        )
        if contact.shape != (2,) or contact.dtype.kind != "b":
            raise ValueError(
                "foot locker support_contact must return two booleans"
            )
        return np.ascontiguousarray(contact, dtype=bool)

    def landing_complete(
        self,
        pose: KinematicPose,
        placement: StairPlacement,
    ) -> bool:
        """Latch a genuine endpoint landing by each complete foot."""

        support_method = getattr(
            self.foot_locker, "landing_support", None
        )
        if callable(support_method):
            support = np.asarray(
                support_method(
                    pose,
                    placement,
                    side=self._landing_side,
                )
            )
            if support.shape != (2,) or support.dtype.kind != "b":
                raise ValueError(
                    "foot locker landing_support must return two booleans"
                )
            self._landing_supported_feet[0] = bool(
                self._landing_supported_feet[0] or support[0]
            )
            self._landing_supported_feet[1] = bool(
                self._landing_supported_feet[1] or support[1]
            )
            return bool(
                all(self._landing_supported_feet)
                and np.any(support)
            )
        method = getattr(self.foot_locker, "landing_complete", None)
        if not callable(method):
            raise ValueError(
                "terrain transition foot locker cannot measure landing"
            )
        value = method(pose, placement, side=self._landing_side)
        if type(value) is not bool:
            raise ValueError("foot locker landing_complete must return bool")
        return value

    def begin_entry(self) -> None:
        """Arm a fresh short contact lock for one handoff attempt."""

        self.foot_locker.reset()
        self._entry_frame = 0
        self._entry_lock_active = True
        self._entry_contact: tuple[bool, bool] | None = None
        self.last_pose_repair_result = None
        self.last_foot_lock_result = None

    def end_stair(self) -> None:
        """Disable and clear temporal IK state after a landing or reset."""

        self.foot_locker.reset()
        self._entry_frame = 0
        self._entry_lock_active = False
        self._entry_contact = None
        self._landing_previous_pose: KinematicPose | None = None
        self._landing_measured_contact = (False, False)
        self._landing_ground_proximity_contact = (False, False)
        self._landing_contact = (False, False)
        self._landing_touchdown_frames = [0, 0]
        self._landing_supported_feet = [False, False]
        self._landing_side = "top"
        self._landing_filter_outcome = "reset"
        self._landing_filter_reason = "reset"
        self.last_pose_repair_result = None
        self.last_foot_lock_result = None

    def begin_landing(
        self,
        pose: KinematicPose,
        *,
        side: str = "top",
        defer_contact_acquisition: bool = False,
    ) -> None:
        """Prime landing stance targets at the final authored stair pose.

        ``defer_contact_acquisition`` is reserved for diagnostic playback that
        must observe the configured number of displayed frames before earning
        a lock.  The normal transition path retains immediate landing prime.
        """

        if not isinstance(pose, KinematicPose):
            raise ValueError("pose must be a KinematicPose")
        landing_side = str(side)
        if landing_side not in {"top", "ground"}:
            raise ValueError("landing side must be 'top' or 'ground'")
        if type(defer_contact_acquisition) is not bool:
            raise ValueError("defer_contact_acquisition must be bool")
        self.end_stair()
        self._landing_side = landing_side
        contact_method = getattr(
            self.foot_locker, "landing_stance_contact", None
        )
        if not callable(contact_method):
            raise ValueError(
                "terrain foot locker cannot classify landing stance"
            )
        contact = np.asarray(
            contact_method(
                pose,
                maximum_foot_speed_mps=(
                    self.landing_maximum_foot_speed_mps
                ),
                maximum_sole_clearance_m=(
                    self.landing_maximum_sole_clearance_m
                ),
            ),
            dtype=bool,
        )
        self._landing_previous_pose = pose
        self._landing_measured_contact = (
            bool(contact[0]),
            bool(contact[1]),
        )
        self._landing_ground_proximity_contact = (
            bool(contact[0]),
            bool(contact[1]),
        )
        seeded_contact = (
            np.zeros(2, dtype=bool)
            if defer_contact_acquisition
            else contact
        )
        self._landing_contact = (
            bool(seeded_contact[0]),
            bool(seeded_contact[1]),
        )
        self._landing_touchdown_frames = [
            self.landing_contact_acquire_frames if bool(value) else 0
            for value in seeded_contact
        ]
        seeded = self.foot_locker.apply(
            pose,
            seeded_contact,
            dt_s=self.dt_s,
            minimum_swing_clearance_m=(
                self.landing_minimum_swing_clearance_m
            ),
        )
        self.last_foot_lock_result = seeded
        if not bool(getattr(seeded, "accepted", False)):
            self._landing_filter_outcome = "prime-rejected"
            self._landing_filter_reason = str(
                getattr(seeded, "reason", "foot lock prime rejected")
            )
            self.foot_locker.reset()
            return
        self._landing_filter_outcome = (
            "primed-active"
            if any(
                bool(value)
                for value in getattr(seeded, "locked", ())
            )
            else "primed-idle"
        )
        self._landing_filter_reason = str(
            getattr(seeded, "reason", "accepted")
        )

    def _recover_trusted_pre_repair_contact(
        self,
        pose: KinematicPose,
        source_contact: np.ndarray,
        rejected_repair: object,
    ) -> KinematicPose | None:
        """Retry a rejected repair through an existing trusted foot target."""

        continued = np.asarray(
            [
                self._landing_contact[foot] and bool(source_contact[foot])
                for foot in range(2)
            ],
            dtype=bool,
        )
        rejected_reason = str(
            getattr(rejected_repair, "reason", "pose repair rejected")
        )
        if not bool(np.any(continued)):
            self._landing_filter_outcome = "repair-rejected"
            self._landing_filter_reason = rejected_reason
            return None

        snapshot = self.foot_locker.snapshot_state()
        locked = self.foot_locker.apply(
            pose,
            continued,
            dt_s=self.dt_s,
            minimum_swing_clearance_m=(
                self.landing_minimum_swing_clearance_m
            ),
        )
        self.last_foot_lock_result = locked
        if not bool(getattr(locked, "accepted", False)):
            self.foot_locker.restore_state(snapshot)
            self._landing_filter_outcome = (
                "pre-repair-recovery-lock-rejected"
            )
            self._landing_filter_reason = (
                "trusted recovery lock rejected after pre-repair rejection: "
                f"{rejected_reason}; "
                f"{getattr(locked, 'reason', 'foot lock rejected')}"
            )
            return None
        filtered = getattr(locked, "pose", None)
        if not isinstance(filtered, KinematicPose):
            self.foot_locker.restore_state(snapshot)
            self._landing_filter_outcome = (
                "pre-repair-recovery-lock-rejected"
            )
            self._landing_filter_reason = (
                "trusted recovery foot locker returned an invalid pose"
            )
            return None

        post = self.pose_repairer.repair(filtered)
        self.last_pose_repair_result = post
        if not bool(getattr(post, "accepted", False)):
            self.foot_locker.restore_state(snapshot)
            self._landing_filter_outcome = (
                "pre-repair-recovery-post-repair-rejected"
            )
            self._landing_filter_reason = (
                "trusted recovery post-lock repair rejected after "
                "pre-repair rejection: "
                f"{rejected_reason}; "
                f"{getattr(post, 'reason', 'pose repair rejected')}"
            )
            return None
        result = getattr(post, "pose", None)
        if not isinstance(result, KinematicPose):
            self.foot_locker.restore_state(snapshot)
            self._landing_filter_outcome = (
                "pre-repair-recovery-post-repair-rejected"
            )
            self._landing_filter_reason = (
                "trusted recovery post-lock repair returned an invalid pose"
            )
            return None

        # Commit guard state only after both existing utilities accept.  No
        # measurement or source label can acquire a new foot in this path.
        self._landing_previous_pose = pose
        self._landing_contact = (
            bool(continued[0]),
            bool(continued[1]),
        )
        self._landing_touchdown_frames = [
            self.landing_contact_acquire_frames if bool(value) else 0
            for value in continued
        ]
        if any(bool(value) for value in getattr(locked, "locked", ())):
            self._landing_filter_outcome = "pre-repair-recovery-active"
        elif any(
            bool(value) for value in getattr(locked, "releasing", ())
        ):
            self._landing_filter_outcome = "pre-repair-recovery-releasing"
        else:
            self._landing_filter_outcome = "pre-repair-recovery-idle"
        self._landing_filter_reason = (
            "recovered trusted contact after pre-repair rejection: "
            f"{rejected_reason}"
        )
        return result

    def filter_landing(
        self,
        pose: KinematicPose,
        *,
        trusted_source_contact: object | None = None,
    ) -> KinematicPose | None:
        """Apply bounded stance locking during a flat landing bridge.

        With a trusted source label, an untrusted foot requires both source
        contact and measured ground proximity for acquisition.  Existing
        trusted contacts use source continuation/release.  Omitting the label
        preserves the measured speed-and-support policy used by the normal
        terrain-transition path.
        """

        if not isinstance(pose, KinematicPose):
            raise ValueError("pose must be a KinematicPose")
        source_contact: np.ndarray | None = None
        if trusted_source_contact is not None:
            source_value = np.asarray(trusted_source_contact)
            if source_value.shape != (2,) or source_value.dtype.kind != "b":
                raise ValueError(
                    "trusted_source_contact must contain two booleans"
                )
            source_contact = np.ascontiguousarray(source_value, dtype=bool)
        repair = self.pose_repairer.repair(pose)
        self.last_pose_repair_result = repair
        self.last_foot_lock_result = None
        if not bool(getattr(repair, "accepted", False)):
            if source_contact is None:
                self._landing_filter_outcome = "repair-rejected"
                self._landing_filter_reason = str(
                    getattr(repair, "reason", "pose repair rejected")
                )
                return None
            return self._recover_trusted_pre_repair_contact(
                pose,
                source_contact,
                repair,
            )
        candidate = getattr(repair, "pose", None)
        if not isinstance(candidate, KinematicPose):
            raise ValueError("pose repairer returned an invalid pose")
        contact_method = getattr(
            self.foot_locker, "landing_stance_contact", None
        )
        if not callable(contact_method):
            raise ValueError(
                "terrain foot locker cannot classify landing stance"
            )
        contact_kwargs = (
            {}
            if self._landing_previous_pose is None
            else {
                "previous_pose": self._landing_previous_pose,
                "dt_s": self.dt_s,
            }
        )
        raw_contact = np.asarray(
            contact_method(
                candidate,
                maximum_foot_speed_mps=(
                    self.landing_maximum_foot_speed_mps
                ),
                maximum_sole_clearance_m=(
                    self.landing_maximum_sole_clearance_m
                ),
                **contact_kwargs,
            ),
            dtype=bool,
        )
        self._landing_previous_pose = candidate
        self._landing_measured_contact = (
            bool(raw_contact[0]),
            bool(raw_contact[1]),
        )
        proximity_contact = (
            raw_contact
            if source_contact is None
            else self.support_contact(
                candidate,
                maximum_sole_clearance_m=(
                    self.landing_maximum_sole_clearance_m
                ),
            )
        )
        self._landing_ground_proximity_contact = (
            bool(proximity_contact[0]),
            bool(proximity_contact[1]),
        )
        contact_values: list[bool] = [False, False]
        for foot in range(2):
            if self._landing_contact[foot]:
                contact_values[foot] = bool(
                    raw_contact[foot]
                    if source_contact is None
                    else source_contact[foot]
                )
                self._landing_touchdown_frames[foot] = (
                    self.landing_contact_acquire_frames
                    if contact_values[foot]
                    else 0
                )
            elif bool(
                raw_contact[foot]
                if source_contact is None
                else source_contact[foot] and proximity_contact[foot]
            ):
                self._landing_touchdown_frames[foot] += 1
                contact_values[foot] = (
                    self._landing_touchdown_frames[foot]
                    >= self.landing_contact_acquire_frames
                )
            else:
                self._landing_touchdown_frames[foot] = 0
        self._landing_contact = (
            contact_values[0],
            contact_values[1],
        )
        contact = np.asarray(contact_values, dtype=bool)
        locked = self.foot_locker.apply(
            candidate,
            contact,
            dt_s=self.dt_s,
            minimum_swing_clearance_m=(
                self.landing_minimum_swing_clearance_m
            ),
        )
        self.last_foot_lock_result = locked
        if not bool(getattr(locked, "accepted", False)):
            rejected_reason = str(
                getattr(locked, "reason", "foot lock rejected")
            )
            # Do not drop a saturated stance correction in one frame.  The
            # lock update is transactional, so retry the same pose with both
            # contacts released.  This starts the existing inertialized
            # release spring from the last committed target and avoids the
            # large joint snap caused by reset-and-publish.
            released = self.foot_locker.apply(
                candidate,
                np.zeros(2, dtype=bool),
                dt_s=self.dt_s,
                minimum_swing_clearance_m=(
                    self.landing_minimum_swing_clearance_m
                ),
            )
            self.last_foot_lock_result = released
            if not bool(getattr(released, "accepted", False)):
                # Neither lock path can publish transactionally.  Signal the
                # caller to discard this once-repaired inner pose, clear all
                # landing state, and publish its advancing inertialized source
                # candidate rather than one side of an IK flip.
                self.foot_locker.reset()
                self._landing_filter_outcome = (
                    "bypass-after-double-reject"
                )
                self._landing_filter_reason = (
                    f"{rejected_reason}; release rejected: "
                    f"{getattr(released, 'reason', 'foot lock rejected')}"
                )
                return None
            locked = released
            self._landing_filter_outcome = "release-after-reject"
            self._landing_filter_reason = rejected_reason
        elif any(
            bool(value) for value in getattr(locked, "locked", ())
        ):
            self._landing_filter_outcome = "active"
            self._landing_filter_reason = str(
                getattr(locked, "reason", "accepted")
            )
        elif any(
            bool(value) for value in getattr(locked, "releasing", ())
        ):
            self._landing_filter_outcome = "releasing"
            self._landing_filter_reason = str(
                getattr(locked, "reason", "accepted")
            )
        else:
            self._landing_filter_outcome = "idle"
            self._landing_filter_reason = str(
                getattr(locked, "reason", "accepted")
            )
        filtered = getattr(locked, "pose", None)
        if not isinstance(filtered, KinematicPose):
            raise ValueError("foot locker returned an invalid pose")
        post = self.pose_repairer.repair(filtered)
        self.last_pose_repair_result = post
        if not bool(getattr(post, "accepted", False)):
            self._landing_filter_outcome = "post-repair-rejected"
            self._landing_filter_reason = str(
                getattr(post, "reason", "post-lock pose repair rejected")
            )
            return None
        result = getattr(post, "pose", None)
        if not isinstance(result, KinematicPose):
            raise ValueError("pose repairer returned an invalid pose")
        return result

    reset = end_stair

    def snapshot_state(self) -> TerrainTransitionGuardState:
        return TerrainTransitionGuardState(
            foot_lock_state=self.foot_locker.snapshot_state(),
            entry_frame=int(self._entry_frame),
            entry_lock_active=bool(self._entry_lock_active),
            entry_contact=self._entry_contact,
        )

    def restore_state(
        self, snapshot: TerrainTransitionGuardState
    ) -> None:
        if (
            not isinstance(snapshot, TerrainTransitionGuardState)
            or type(snapshot.entry_frame) is not int
            or snapshot.entry_frame < 0
            or type(snapshot.entry_lock_active) is not bool
            or (
                snapshot.entry_contact is not None
                and (
                    len(snapshot.entry_contact) != 2
                    or any(
                        type(value) is not bool
                        for value in snapshot.entry_contact
                    )
                )
            )
        ):
            raise ValueError("terrain transition guard snapshot is invalid")
        self.foot_locker.restore_state(snapshot.foot_lock_state)
        self._entry_frame = snapshot.entry_frame
        self._entry_lock_active = snapshot.entry_lock_active
        self._entry_contact = snapshot.entry_contact

    def __call__(
        self,
        pose: KinematicPose,
        source_contact: object,
    ) -> KinematicPose | None:
        """Return a repaired pose, or ``None`` for transactional rollback."""

        if not isinstance(pose, KinematicPose):
            raise ValueError("pose must be a KinematicPose")
        contact = np.asarray(source_contact)
        if contact.shape != (2,) or contact.dtype.kind != "b":
            raise ValueError("source_contact must contain two booleans")

        repair = self.pose_repairer.repair(pose)
        self.last_pose_repair_result = repair
        self.last_foot_lock_result = None
        if not bool(getattr(repair, "accepted", False)):
            return None
        candidate = getattr(repair, "pose", None)
        if not isinstance(candidate, KinematicPose):
            raise ValueError("pose repairer returned an invalid pose")

        if not self._entry_lock_active:
            return candidate
        if self._entry_contact is None:
            support_method = getattr(
                self.foot_locker, "support_contact", None
            )
            measured = (
                support_method(candidate)
                if callable(support_method)
                else contact
            )
            measured_array = np.asarray(measured)
            if (
                measured_array.shape != (2,)
                or measured_array.dtype.kind != "b"
            ):
                raise ValueError(
                    "foot locker support_contact must return two booleans"
                )
            self._entry_contact = (
                bool(measured_array[0]),
                bool(measured_array[1]),
            )
        lock_contact = (
            np.asarray(self._entry_contact, dtype=bool)
            if self._entry_frame < self.contact_hold_frames
            else (
                contact
                if (
                    self.track_source_contacts
                    and self._entry_frame
                    >= self.source_contact_delay_frames
                )
                else np.zeros(2, dtype=bool)
            )
        )
        locked = self.foot_locker.apply(
            candidate, lock_contact, dt_s=self.dt_s
        )
        self.last_foot_lock_result = locked
        if not bool(getattr(locked, "accepted", False)):
            # This lock is an entry-splice visual aid, not a safety oracle.
            # Once one corrected entry pose has been published, refusing every
            # subsequent authored frame can deadlock transactional playback at
            # the stair edge.  The exact collision repair above has already
            # accepted ``candidate``, so bounded mode may safely release the
            # cosmetic lock and continue.  Experimental full-route source
            # contact tracking remains strict and transactional.
            if self._entry_frame > 0:
                self.foot_locker.reset()
                if not self.track_source_contacts:
                    self._entry_lock_active = False
                    self._entry_contact = None
                return candidate
            return None
        candidate = getattr(locked, "pose", None)
        if not isinstance(candidate, KinematicPose):
            raise ValueError("foot locker returned an invalid pose")

        # Leg-only IK can introduce a small new sole penetration even though
        # the authored pose passed the first exact collision repair.  Validate
        # and repair the pose that will actually be published, not only the
        # pre-IK input.
        post_lock_repair = self.pose_repairer.repair(candidate)
        self.last_pose_repair_result = post_lock_repair
        if not bool(getattr(post_lock_repair, "accepted", False)):
            return None
        candidate = getattr(post_lock_repair, "pose", None)
        if not isinstance(candidate, KinematicPose):
            raise ValueError(
                "pose repairer returned an invalid post-lock pose"
            )

        self._entry_frame += 1
        if (
            not self.track_source_contacts
            and
            self._entry_frame > self.contact_hold_frames
            and not any(bool(value) for value in locked.locked)
            and not any(bool(value) for value in locked.releasing)
        ):
            self.foot_locker.reset()
            self._entry_lock_active = False
        return candidate


__all__ = [
    "G1TerrainFootLock",
    "G1TerrainTransitionGuard",
    "TerrainFootLockResult",
    "TerrainFootLockState",
    "TerrainTransitionGuardState",
]
