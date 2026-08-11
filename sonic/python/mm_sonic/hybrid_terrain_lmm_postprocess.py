"""Existing-utility display filtering for the full-walking terrain LMM."""

from __future__ import annotations

import math

import numpy as np

from .gear_action import mujoco_to_isaaclab_joint_vector
from .hybrid_terrain_interactive import KinematicPose, PoseInertializer
from .offline_corpus import BODY_NAMES
from .render_terrain_transition_mesh import MUJOCO_JOINT_NAMES, build_qpos
from .terrain_foot_lock import G1TerrainFootLock
from .terrain_oracle.math3d import angular_velocity_world_wxyz
from .terrain_pose_repair import G1TerrainPoseRepair


_IDENTITY = "existing-pose-inertializer-repair-foot-lock/v1"


class NativeQposPoseAdapter:
    """Convert native G1 MuJoCo qpos samples to ``KinematicPose``."""

    def __init__(self, model: object) -> None:
        import mujoco

        self.model = model
        free = np.flatnonzero(
            np.asarray(model.jnt_type) == int(mujoco.mjtJoint.mjJNT_FREE)
        )
        if free.size != 1 or int(model.nq) != 36 or int(model.nv) != 35:
            raise ValueError("model must have the native 36-qpos G1 layout")
        self._root_qpos_address = int(model.jnt_qposadr[int(free[0])])
        self._joint_ids = tuple(
            int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
            )
            for joint_name in MUJOCO_JOINT_NAMES
        )
        self._body_ids = tuple(
            int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, body_name
                )
            )
            for body_name in BODY_NAMES
        )
        if any(index < 0 for index in (*self._joint_ids, *self._body_ids)):
            raise ValueError("model lacks the named native G1 joint/body layout")
        self._joint_qpos_addresses = np.asarray(
            model.jnt_qposadr[np.asarray(self._joint_ids)], dtype=np.int32
        )
        self._joint_dof_addresses = np.asarray(
            model.jnt_dofadr[np.asarray(self._joint_ids)], dtype=np.int32
        )
        self._data = mujoco.MjData(model)
        self.reset()

    def reset(self) -> None:
        self._previous_qpos: np.ndarray | None = None
        self._previous_body_position: np.ndarray | None = None
        self._previous_body_quaternion_wxyz: np.ndarray | None = None

    def to_pose(self, qpos: object, *, dt_s: float) -> KinematicPose:
        """Validate qpos, run FK, derive velocities, and update prior samples."""

        import mujoco

        value = np.asarray(qpos)
        step = float(dt_s)
        if (
            value.shape != (int(self.model.nq),)
            or value.dtype.kind not in "iuf"
            or not np.isfinite(value).all()
        ):
            raise ValueError("qpos must contain 36 finite numeric values")
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        current_qpos = np.ascontiguousarray(value, dtype=np.float64)
        self._data.qpos[:] = current_qpos
        mujoco.mj_forward(self.model, self._data)

        body_position = np.asarray(
            [self._data.xpos[index] for index in self._body_ids],
            dtype=np.float64,
        )
        body_wxyz = np.asarray(
            [self._data.xquat[index] for index in self._body_ids],
            dtype=np.float64,
        )
        joint_velocity = np.zeros(29, dtype=np.float64)
        body_linear_velocity = np.zeros((30, 3), dtype=np.float64)
        body_angular_velocity = np.zeros((30, 3), dtype=np.float64)
        if self._previous_qpos is not None:
            qvel = np.zeros(int(self.model.nv), dtype=np.float64)
            mujoco.mj_differentiatePos(
                self.model,
                qvel,
                step,
                self._previous_qpos,
                current_qpos,
            )
            joint_velocity = mujoco_to_isaaclab_joint_vector(
                qvel[self._joint_dof_addresses]
            )
            assert self._previous_body_position is not None
            assert self._previous_body_quaternion_wxyz is not None
            body_linear_velocity = (
                body_position - self._previous_body_position
            ) / step
            body_angular_velocity = np.asarray(
                angular_velocity_world_wxyz(
                    np.stack(
                        (self._previous_body_quaternion_wxyz, body_wxyz), axis=0
                    ),
                    1.0 / step,
                )[-1],
                dtype=np.float64,
            )

        root = self._root_qpos_address
        root_wxyz = current_qpos[root + 3 : root + 7]
        pose = KinematicPose(
            root_position_world=current_qpos[root : root + 3],
            root_orientation_world_xyzw=root_wxyz[[1, 2, 3, 0]],
            joint_position=mujoco_to_isaaclab_joint_vector(
                current_qpos[self._joint_qpos_addresses]
            ),
            joint_velocity=joint_velocity,
            body_position_world=body_position,
            body_orientation_world_xyzw=body_wxyz[:, [1, 2, 3, 0]],
            body_linear_velocity_world=body_linear_velocity,
            body_angular_velocity_world=body_angular_velocity,
        )
        self._previous_qpos = current_qpos.copy()
        self._previous_body_position = body_position.copy()
        self._previous_body_quaternion_wxyz = body_wxyz.copy()
        return pose


class ExistingUtilityPosePostprocessor:
    """Compose the existing inertializer, terrain repair, and foot lock."""

    def __init__(
        self,
        model: object,
        scene: object,
        *,
        fps: float = 60.0,
        inertialization_halflife_s: float = 0.10,
        pose_adapter: object | None = None,
        pose_repairer: object | None = None,
        foot_locker: object | None = None,
    ) -> None:
        rate = float(fps)
        half = float(inertialization_halflife_s)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("fps must be finite and positive")
        if not math.isfinite(half) or half <= 0.0:
            raise ValueError(
                "inertialization_halflife_s must be finite and positive"
            )
        if not callable(getattr(scene, "height_at_world_xy", None)):
            raise ValueError("scene must provide height_at_world_xy")
        self.model = model
        self.scene = scene
        self.fps = rate
        self.inertialization_halflife_s = half
        self.pose_adapter = (
            NativeQposPoseAdapter(model) if pose_adapter is None else pose_adapter
        )
        self.pose_repairer = (
            G1TerrainPoseRepair(model, scene)
            if pose_repairer is None
            else pose_repairer
        )
        self.foot_locker = (
            G1TerrainFootLock(model, scene) if foot_locker is None else foot_locker
        )
        self.reset()

    def reset(self) -> None:
        """Clear adapter, inertializer, displayed pose, and foot-lock state."""

        self.pose_adapter.reset()
        self.foot_locker.reset()
        self._inertializer: PoseInertializer | None = None
        self._inertializer_elapsed_s = 0.0
        self._displayed_pose: KinematicPose | None = None
        self._displayed_qpos: np.ndarray | None = None
        self._previous_row: int | None = None
        self._previous_range_index: int | None = None
        self.pose_repair_count = 0
        self.foot_lock_accept_count = 0
        self.foot_lock_bypass_count = 0
        self.raw_repair_failure_count = 0
        self.last_reason = "reset"

    @staticmethod
    def _index(value: object, name: str) -> int:
        if type(value) not in (int, np.int32, np.int64):
            raise ValueError(f"{name} must be an integer")
        result = int(value)
        if result < 0:
            raise ValueError(f"{name} must be nonnegative")
        return result

    def step(
        self,
        qpos: object,
        *,
        row: object,
        range_index: object,
        source_contact: object,
        dt_s: float,
    ) -> np.ndarray:
        """Return the non-freezing, utility-filtered display qpos."""

        step = float(dt_s)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        row_value = self._index(row, "row")
        range_value = self._index(range_index, "range_index")
        contacts = np.asarray(source_contact)
        if contacts.shape != (2,) or contacts.dtype.kind != "b":
            raise ValueError("source_contact must contain two booleans")

        raw_target = self.pose_adapter.to_pose(qpos, dt_s=step)
        transition = (
            self._previous_row is not None
            and row_value != self._previous_row
            and not (
                range_value == self._previous_range_index
                and row_value == self._previous_row + 1
            )
        )
        if transition and self._displayed_pose is not None:
            self._inertializer = PoseInertializer(
                self._displayed_pose,
                raw_target,
                halflife_s=self.inertialization_halflife_s,
            )
            self._inertializer_elapsed_s = 0.0
        elif self._inertializer is not None:
            self._inertializer_elapsed_s += step

        used_inertializer = self._inertializer is not None
        candidate = (
            raw_target
            if not used_inertializer
            else self._inertializer.apply(
                raw_target, elapsed_s=self._inertializer_elapsed_s
            )
        )
        repaired = self.pose_repairer.repair(candidate)
        if not bool(repaired.accepted) and used_inertializer:
            self.last_reason = str(
                getattr(repaired, "reason", "inertialized pose repair rejected")
            )
            self._inertializer = None
            self._inertializer_elapsed_s = 0.0
            repaired = self.pose_repairer.repair(raw_target)
        if not bool(repaired.accepted):
            self.raw_repair_failure_count += 1
            self.last_reason = str(
                getattr(repaired, "reason", "raw source pose repair rejected")
            )
            self._previous_row = row_value
            self._previous_range_index = range_value
            if self._displayed_qpos is None:
                raise RuntimeError("initial raw source pose repair rejected")
            return self._displayed_qpos.copy()

        repaired_pose = repaired.pose
        if bool(getattr(repaired, "repaired", False)):
            self.pose_repair_count += 1
        snapshot = self.foot_locker.snapshot_state()
        locked = self.foot_locker.apply(
            repaired_pose, contacts, dt_s=step
        )
        if bool(locked.accepted):
            displayed_pose = locked.pose
            self.foot_lock_accept_count += 1
        else:
            self.foot_locker.restore_state(snapshot)
            displayed_pose = repaired_pose
            self.foot_lock_bypass_count += 1
            self.last_reason = str(getattr(locked, "reason", "foot lock rejected"))

        displayed_qpos = build_qpos(
            displayed_pose.root_position_world,
            displayed_pose.root_orientation_world_xyzw,
            displayed_pose.joint_position,
            self.model,
        )
        self._displayed_pose = displayed_pose
        self._displayed_qpos = displayed_qpos.copy()
        self._previous_row = row_value
        self._previous_range_index = range_value
        return displayed_qpos

    def identity(self) -> dict[str, object]:
        """Return immutable policy values and current diagnostic counters."""

        return {
            "diagnostic_display_postprocessor": _IDENTITY,
            "inertialization_halflife_s": self.inertialization_halflife_s,
            "pose_repair_count": self.pose_repair_count,
            "foot_lock_accept_count": self.foot_lock_accept_count,
            "foot_lock_bypass_count": self.foot_lock_bypass_count,
            "raw_repair_failure_count": self.raw_repair_failure_count,
            "last_reason": self.last_reason,
        }


__all__ = ["ExistingUtilityPosePostprocessor", "NativeQposPoseAdapter"]
