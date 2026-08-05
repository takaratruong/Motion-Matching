"""Contact-preserving terrain repair for clean G1 kinematics.

The motion database remains the source of style and timing.  This module only
repairs a selected pose when the exact G1 collision geometry intersects the
observed terrain.  It follows the multi-point Jacobian pattern used by TCRS:
root and upper body normally stay fixed, while the six joints of an offending
leg move the actual MuJoCo sole spheres by the smallest correction along the
hard collision oracle's 3-D contact normal.  A bounded pelvis-height fallback
is reserved for genuinely tread-normal residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .gear_action import mujoco_to_isaaclab_joint_vector
from .hybrid_terrain_interactive import KinematicPose
from .joints import ContractError
from .offline_corpus import BODY_NAMES
from .render_terrain_transition_mesh import (
    MUJOCO_JOINT_NAMES,
    build_qpos,
)


_FOOT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
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
class TerrainPoseRepairResult:
    pose: KinematicPose
    accepted: bool
    repaired: bool
    maximum_foot_penetration_before_m: float
    maximum_foot_penetration_after_m: float
    maximum_forbidden_penetration_after_m: float
    maximum_joint_delta_rad: float
    root_height_offset_m: float
    iterations: int


@dataclass(frozen=True)
class _CollisionState:
    foot_penetration_m: tuple[float, float]
    sole_clearance_m: tuple[float, float]
    # Smallest useful world-space separation direction for the deepest
    # terrain contact on each foot.  A vertical-only correction is sufficient
    # for tread penetration, but not for a toe or heel intersecting a riser.
    # Keeping the MuJoCo contact normal here lets the same bounded leg IK move
    # the complete sole away from either kind of surface.
    foot_separation_world_m: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]
    forbidden_penetration_m: float
    enforce_clearance: bool = False

    @property
    def maximum_foot_penetration_m(self) -> float:
        return max(
            self.foot_penetration_with_clearance_m(0),
            self.foot_penetration_with_clearance_m(1),
        )

    def foot_penetration_with_clearance_m(self, foot: int) -> float:
        if not self.enforce_clearance:
            return self.foot_penetration_m[foot]
        return max(
            self.foot_penetration_m[foot],
            -self.sole_clearance_m[foot],
            0.0,
        )


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


class G1TerrainPoseRepair:
    """Repair sole penetration using bounded 12-DoF leg-only Jacobian IK."""

    def __init__(
        self,
        model: object,
        scene: object,
        *,
        maximum_penetration_m: float = 0.005,
        target_penetration_m: float = 0.0025,
        maximum_lift_m: float = 0.08,
        maximum_joint_delta_rad: float = 0.25,
        maximum_root_height_correction_m: float = 0.04,
        prefer_root_lift_above_m: float | None = None,
        root_lift_only: bool = False,
        include_visual_mesh_clearance: bool = False,
        maximum_iterations: int = 24,
        maximum_contact_passes: int = 6,
        damping: float = 0.025,
        posture_weight: float = 0.0002,
    ) -> None:
        import mujoco

        self.model = model
        self.scene = scene
        self.maximum_penetration_m = float(maximum_penetration_m)
        self.target_penetration_m = float(target_penetration_m)
        self.maximum_lift_m = float(maximum_lift_m)
        self.maximum_joint_delta_rad = float(maximum_joint_delta_rad)
        self.maximum_root_height_correction_m = float(
            maximum_root_height_correction_m
        )
        self.prefer_root_lift_above_m = (
            None
            if prefer_root_lift_above_m is None
            else float(prefer_root_lift_above_m)
        )
        self.root_lift_only = bool(root_lift_only)
        self.include_visual_mesh_clearance = bool(
            include_visual_mesh_clearance
        )
        self.maximum_iterations = int(maximum_iterations)
        self.maximum_contact_passes = int(maximum_contact_passes)
        self.damping = float(damping)
        self.posture_weight = float(posture_weight)
        if not callable(getattr(scene, "height_at_world_xy", None)):
            raise ValueError("scene must provide height_at_world_xy")
        if (
            not 0.0 <= self.target_penetration_m
            < self.maximum_penetration_m
            or not math.isfinite(self.maximum_lift_m)
            or self.maximum_lift_m <= 0.0
            or not math.isfinite(self.maximum_joint_delta_rad)
            or self.maximum_joint_delta_rad <= 0.0
            or not math.isfinite(self.maximum_root_height_correction_m)
            or self.maximum_root_height_correction_m <= 0.0
            or (
                self.prefer_root_lift_above_m is not None
                and (
                    not math.isfinite(self.prefer_root_lift_above_m)
                    or self.prefer_root_lift_above_m
                    <= self.maximum_penetration_m
                )
            )
            or self.maximum_iterations <= 0
            or self.maximum_contact_passes <= 0
            or not math.isfinite(self.damping)
            or self.damping <= 0.0
            or not math.isfinite(self.posture_weight)
            or self.posture_weight < 0.0
        ):
            raise ValueError("terrain pose repair parameters are invalid")

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
            raise ValueError("repair model contains no terrain_* geometry")

        foot_bodies: list[int] = []
        foot_geoms: list[frozenset[int]] = []
        sphere_geoms: list[tuple[int, ...]] = []
        visual_meshes: list[tuple[tuple[int, np.ndarray], ...]] = []
        sphere_type = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
        for name in _FOOT_BODY_NAMES:
            body_id = int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, name
                )
            )
            if body_id < 0:
                raise ValueError(f"repair model is missing body {name}")
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
            meshes: list[tuple[int, np.ndarray]] = []
            for geom_id in sorted(geoms):
                if int(model.geom_type[geom_id]) != mesh_type:
                    continue
                mesh_id = int(model.geom_dataid[geom_id])
                start = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                vertices = np.asarray(
                    model.mesh_vert[start : start + count],
                    dtype=np.float64,
                ).copy()
                if count > 0:
                    meshes.append((geom_id, vertices))
            if not geoms or not spheres:
                raise ValueError(
                    f"repair model has no sole spheres below {name}"
                )
            foot_bodies.append(body_id)
            foot_geoms.append(geoms)
            sphere_geoms.append(spheres)
            visual_meshes.append(tuple(meshes))
        if foot_geoms[0] & foot_geoms[1]:
            raise ValueError("left and right repair foot geometry overlap")
        self._foot_bodies = (foot_bodies[0], foot_bodies[1])
        self._foot_geoms = (foot_geoms[0], foot_geoms[1])
        self._sphere_geoms = (sphere_geoms[0], sphere_geoms[1])
        self._visual_meshes = (visual_meshes[0], visual_meshes[1])

        leg_joint_ids: list[np.ndarray] = []
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
                raise ValueError("repair model is missing a G1 leg joint")
            leg_joint_ids.append(ids)
            leg_qpos.append(
                np.asarray(model.jnt_qposadr[ids], dtype=np.int32)
            )
            leg_dofs.append(
                np.asarray(model.jnt_dofadr[ids], dtype=np.int32)
            )
            leg_limits.append(
                np.asarray(model.jnt_range[ids], dtype=np.float64)
            )
        self._leg_joint_ids = (leg_joint_ids[0], leg_joint_ids[1])
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
        if any(body_id < 0 for body_id in self._body_ids):
            raise ValueError("repair model lacks the native 30-body G1 layout")
        self._joint_ids_mujoco_order = tuple(
            int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
            )
            for name in MUJOCO_JOINT_NAMES
        )
        if any(joint_id < 0 for joint_id in self._joint_ids_mujoco_order):
            raise ValueError("repair model lacks the native 29-joint G1 layout")
        free = np.flatnonzero(
            np.asarray(model.jnt_type)
            == int(mujoco.mjtJoint.mjJNT_FREE)
        )
        if free.size != 1:
            raise ValueError("repair model must contain exactly one free joint")
        self._root_qpos_address = int(
            model.jnt_qposadr[int(free[0])]
        )
        self._data = mujoco.MjData(model)

    def _collision_state(self) -> _CollisionState:
        import mujoco

        data = self._data
        mujoco.mj_forward(self.model, data)
        mujoco.mj_collision(self.model, data)
        penetration = [0.0, 0.0]
        separation = [np.zeros(3, dtype=np.float64) for _ in range(2)]
        forbidden = 0.0
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if (
                geom1 in self._terrain_geoms
                and geom2 not in self._terrain_geoms
            ):
                robot_geom = geom2
            elif (
                geom2 in self._terrain_geoms
                and geom1 not in self._terrain_geoms
            ):
                robot_geom = geom1
            else:
                continue
            depth = max(0.0, -float(contact.dist))
            if robot_geom in self._foot_geoms[0]:
                foot = 0
            elif robot_geom in self._foot_geoms[1]:
                foot = 1
            else:
                forbidden = max(forbidden, depth)
                continue
            if depth > penetration[foot]:
                # MuJoCo's contact normal points from geom1 toward geom2.
                # Move a robot geom2 along +normal, or a robot geom1 along
                # -normal, to separate it from the terrain.
                normal = np.asarray(contact.frame[:3], dtype=np.float64)
                direction = normal if robot_geom == geom2 else -normal
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm > 1.0e-12:
                    separation[foot] = depth * direction / direction_norm
                penetration[foot] = depth

        clearance: list[float] = []
        for foot, spheres in enumerate(self._sphere_geoms):
            values = []
            for geom_id in spheres:
                center = np.asarray(
                    data.geom_xpos[geom_id], dtype=np.float64
                )
                bottom = center.copy()
                bottom[2] -= float(self.model.geom_size[geom_id, 0])
                terrain_height = self.scene.height_at_world_xy(bottom[:2])
                values.append(float(bottom[2] - terrain_height))
            # MuJoCo's four tiny sole spheres do not cover the rendered foot
            # mesh when the source goes onto its toe.  That let the visible
            # foot extend several centimetres through the platform even while
            # collision metrics were clean.  Audit the actual rendered mesh
            # vertices in their current world pose as the final clearance
            # oracle; this is cheap for the two ~6.5k-vertex foot meshes.
            if self.include_visual_mesh_clearance:
                for geom_id, vertices_local in self._visual_meshes[foot]:
                    rotation = np.asarray(
                        data.geom_xmat[geom_id], dtype=np.float64
                    ).reshape(3, 3)
                    vertices_world = (
                        vertices_local @ rotation.T
                        + np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
                    )
                    try:
                        terrain_height = np.asarray(
                            self.scene.height_at_world_xy(
                                vertices_world[:, :2]
                            ),
                            dtype=np.float64,
                        )
                        if terrain_height.shape != (len(vertices_world),):
                            raise ValueError("scalar terrain height query")
                    except (TypeError, ValueError, ContractError):
                        terrain_height = np.asarray(
                            [
                                self.scene.height_at_world_xy(point)
                                for point in vertices_world[:, :2]
                            ],
                            dtype=np.float64,
                        )
                    values.append(
                        float(np.min(vertices_world[:, 2] - terrain_height))
                    )
            clearance.append(min(values))
        return _CollisionState(
            foot_penetration_m=(penetration[0], penetration[1]),
            sole_clearance_m=(clearance[0], clearance[1]),
            foot_separation_world_m=(
                tuple(float(value) for value in separation[0]),
                tuple(float(value) for value in separation[1]),
            ),
            forbidden_penetration_m=forbidden,
            enforce_clearance=self.include_visual_mesh_clearance,
        )

    def sole_clearance_m(
        self, pose: KinematicPose
    ) -> tuple[float, float]:
        """Measure the two visible sole clearances for grounding feedback."""

        if not isinstance(pose, KinematicPose):
            raise ValueError("pose must be a KinematicPose")
        self._data.qpos[:] = build_qpos(
            pose.root_position_world,
            pose.root_orientation_world_xyzw,
            pose.joint_position,
            self.model,
        )
        state = self._collision_state()
        return tuple(float(value) for value in state.sole_clearance_m)

    def _required_translation(
        self, state: _CollisionState, foot: int
    ) -> np.ndarray:
        """Return a bounded 3-D sole correction from exact contact normals."""

        contact_deficit = max(
            0.0,
            state.foot_penetration_m[foot]
            - self.target_penetration_m
        )
        correction = np.asarray(
            state.foot_separation_world_m[foot], dtype=np.float64
        ).copy()
        contact_depth = float(np.linalg.norm(correction))
        if contact_deficit > 0.0 and contact_depth > 1.0e-12:
            correction *= (
                contact_deficit + 5.0e-4
            ) / contact_depth
        else:
            correction.fill(0.0)

        # Sole probes and optional rendered-mesh vertices are height queries,
        # so they contribute an additional upward inequality.  Do not add it
        # twice when the exact contact normal already points upward.
        probe_deficit = (
            max(
                0.0,
                -state.sole_clearance_m[foot]
                - self.target_penetration_m,
            )
            if state.enforce_clearance
            else 0.0
        )
        required_up = probe_deficit + (5.0e-4 if probe_deficit > 0.0 else 0.0)
        correction[2] = max(float(correction[2]), required_up)
        return correction

    def _repair_foot(
        self,
        foot: int,
        translation_world_m: object,
        reference_qpos: np.ndarray,
    ) -> int:
        import mujoco

        translation = np.asarray(translation_world_m, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("foot translation must be a finite 3-vector")
        distance = float(np.linalg.norm(translation))
        if not 0.0 < distance <= self.maximum_lift_m:
            return 0
        data = self._data
        sphere_geoms = self._sphere_geoms[foot]
        dofs = self._leg_dofs[foot]
        qpos_addresses = self._leg_qpos[foot]
        limits = self._leg_limits[foot]
        target = np.asarray(
            [data.geom_xpos[geom_id] for geom_id in sphere_geoms],
            dtype=np.float64,
        )
        target += translation
        reference = reference_qpos[qpos_addresses].copy()
        identity = np.eye(len(dofs), dtype=np.float64)
        jacobian_position = np.zeros(
            (3, int(self.model.nv)), dtype=np.float64
        )
        rows: list[np.ndarray] = []
        iterations = 0
        for iteration in range(self.maximum_iterations):
            mujoco.mj_forward(self.model, data)
            current = np.asarray(
                [data.geom_xpos[geom_id] for geom_id in sphere_geoms],
                dtype=np.float64,
            )
            error = (target - current).reshape(-1)
            if float(np.max(np.abs(error))) <= 3.0e-4:
                break
            rows.clear()
            for geom_id in sphere_geoms:
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
            current_leg = data.qpos[qpos_addresses]
            lhs = (
                transpose @ jacobian
                + (
                    self.damping * self.damping
                    + self.posture_weight
                )
                * identity
            )
            rhs = (
                transpose @ error
                + self.posture_weight * (reference - current_leg)
            )
            delta = np.linalg.solve(lhs, rhs)
            delta = np.clip(delta, -0.04, 0.04)
            remaining = np.clip(
                data.qpos[qpos_addresses] + delta
                - reference,
                -self.maximum_joint_delta_rad,
                self.maximum_joint_delta_rad,
            )
            data.qpos[qpos_addresses] = np.clip(
                reference + remaining,
                limits[:, 0] + 1.0e-5,
                limits[:, 1] - 1.0e-5,
            )
            iterations = iteration + 1
        mujoco.mj_forward(self.model, data)
        return iterations

    def _pose_from_data(
        self, source: KinematicPose
    ) -> KinematicPose:
        data = self._data
        mapped = np.asarray(
            [
                data.qpos[int(self.model.jnt_qposadr[joint_id])]
                for joint_id in self._joint_ids_mujoco_order
            ],
            dtype=np.float64,
        )
        joints = mujoco_to_isaaclab_joint_vector(mapped).astype(
            np.float32
        )
        body_position = np.asarray(
            [data.xpos[body_id] for body_id in self._body_ids],
            dtype=np.float32,
        )
        body_wxyz = np.asarray(
            [data.xquat[body_id] for body_id in self._body_ids],
            dtype=np.float32,
        )
        root_address = self._root_qpos_address
        root_position = np.asarray(
            data.qpos[root_address : root_address + 3],
            dtype=np.float32,
        )
        root_wxyz = np.asarray(
            data.qpos[root_address + 3 : root_address + 7],
            dtype=np.float32,
        )
        return KinematicPose(
            root_position_world=root_position,
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

    def repair(self, pose: KinematicPose) -> TerrainPoseRepairResult:
        if not isinstance(pose, KinematicPose):
            raise ValueError("pose must be a KinematicPose")
        reference_qpos = build_qpos(
            pose.root_position_world,
            pose.root_orientation_world_xyzw,
            pose.joint_position,
            self.model,
        )
        self._data.qpos[:] = reference_qpos
        before = self._collision_state()
        needs_repair = tuple(
            before.foot_penetration_with_clearance_m(foot)
            > self.maximum_penetration_m
            for foot in range(2)
        )
        if not any(needs_repair):
            accepted = (
                before.forbidden_penetration_m <= 0.0
                and before.maximum_foot_penetration_m
                <= self.maximum_penetration_m
            )
            return TerrainPoseRepairResult(
                pose=pose,
                accepted=accepted,
                repaired=False,
                maximum_foot_penetration_before_m=(
                    before.maximum_foot_penetration_m
                ),
                maximum_foot_penetration_after_m=(
                    before.maximum_foot_penetration_m
                ),
                maximum_forbidden_penetration_after_m=(
                    before.forbidden_penetration_m
                ),
                maximum_joint_delta_rad=0.0,
                root_height_offset_m=0.0,
                iterations=0,
            )

        root_height_offset = 0.0
        iterations = 0
        initial_corrections = tuple(
            self._required_translation(before, foot) for foot in range(2)
        )
        active_corrections = tuple(
            initial_corrections[foot]
            for foot in range(2)
            if needs_repair[foot]
        )
        # Pelvis lift is a good fallback for penetration into a horizontal
        # tread, but it cannot resolve a toe embedded in a vertical riser.
        # In that case use the contact-normal leg solve and keep root motion
        # untouched.
        corrections_are_vertical = bool(active_corrections) and all(
            float(np.linalg.norm(value[:2]))
            <= max(1.0e-5, 0.35 * abs(float(value[2])))
            for value in active_corrections
        )
        prefer_root = (
            self.prefer_root_lift_above_m is not None
            and before.maximum_foot_penetration_m
            >= self.prefer_root_lift_above_m
            and (self.root_lift_only or corrections_are_vertical)
        )
        if prefer_root:
            root_height_offset = min(
                self.maximum_root_height_correction_m,
                max(
                    0.0,
                    before.maximum_foot_penetration_m
                    - self.target_penetration_m
                    + 5.0e-4,
                ),
            )
            self._data.qpos[
                self._root_qpos_address + 2
            ] += root_height_offset
            raised_reference = self._data.qpos.copy()
            raised_state = self._collision_state()
            if not self.root_lift_only:
                for foot in range(2):
                    if (
                        raised_state.foot_penetration_with_clearance_m(foot)
                        <= self.maximum_penetration_m
                    ):
                        continue
                    iterations += self._repair_foot(
                        foot,
                        self._required_translation(raised_state, foot),
                        raised_reference,
                    )
            after = self._collision_state()
        else:
            # One solve removes the deepest contact.  A foot can touch a tread
            # and its adjacent riser simultaneously, so recompute exact
            # contacts and take several small separation steps rather than
            # solving only the first surface and publishing a residual clip.
            for _contact_pass in range(self.maximum_contact_passes):
                state = self._collision_state()
                repaired_any = False
                for foot in range(2):
                    if (
                        state.foot_penetration_with_clearance_m(foot)
                        <= self.maximum_penetration_m
                    ):
                        continue
                    correction = self._required_translation(state, foot)
                    if float(np.linalg.norm(correction)) <= 1.0e-12:
                        continue
                    iterations += self._repair_foot(
                        foot, correction, reference_qpos
                    )
                    repaired_any = True
                if not repaired_any:
                    break
            after = self._collision_state()
        if (
            after.maximum_foot_penetration_m
            > self.maximum_penetration_m + 1.0e-6
            and not prefer_root
            and not self.root_lift_only
        ):
            # A nearly straight support leg can be unable to shorten enough
            # with fixed-root IK.  TCRS reconstructs support-aware pelvis
            # height before its fixed-root solve; use the same bounded
            # fallback here, keeping root XY/orientation and the entire upper
            # body motion unchanged.  This is deliberately attempted only
            # after the leg-only solve fails.
            residual_corrections = tuple(
                self._required_translation(after, foot) for foot in range(2)
            )
            residual_is_vertical = all(
                (
                    after.foot_penetration_with_clearance_m(foot)
                    <= self.maximum_penetration_m
                    or float(np.linalg.norm(residual_corrections[foot][:2]))
                    <= max(
                        1.0e-5,
                        0.35 * abs(float(residual_corrections[foot][2])),
                    )
                )
                for foot in range(2)
            )
            if residual_is_vertical:
                self._data.qpos[:] = reference_qpos
                root_height_offset = min(
                    self.maximum_root_height_correction_m,
                    max(
                        0.0,
                        before.maximum_foot_penetration_m
                        - self.target_penetration_m
                        + 5.0e-4,
                    ),
                )
                self._data.qpos[
                    self._root_qpos_address + 2
                ] += root_height_offset
                raised_reference = self._data.qpos.copy()
                raised_state = self._collision_state()
                for foot in range(2):
                    if (
                        raised_state.foot_penetration_with_clearance_m(foot)
                        <= self.maximum_penetration_m
                    ):
                        continue
                    iterations += self._repair_foot(
                        foot,
                        self._required_translation(raised_state, foot),
                        raised_reference,
                    )
                after = self._collision_state()
        repaired_pose = self._pose_from_data(pose)
        delta = float(
            np.max(
                np.abs(
                    repaired_pose.joint_position.astype(np.float64)
                    - pose.joint_position.astype(np.float64)
                )
            )
        )
        accepted = (
            after.forbidden_penetration_m <= 0.0
            and after.maximum_foot_penetration_m
            <= self.maximum_penetration_m + 1.0e-6
        )
        return TerrainPoseRepairResult(
            pose=repaired_pose,
            accepted=accepted,
            repaired=(delta > 1.0e-8 or root_height_offset > 0.0),
            maximum_foot_penetration_before_m=(
                before.maximum_foot_penetration_m
            ),
            maximum_foot_penetration_after_m=(
                after.maximum_foot_penetration_m
            ),
            maximum_forbidden_penetration_after_m=(
                after.forbidden_penetration_m
            ),
            maximum_joint_delta_rad=delta,
            root_height_offset_m=root_height_offset,
            iterations=iterations,
        )


__all__ = [
    "G1TerrainPoseRepair",
    "TerrainPoseRepairResult",
]
