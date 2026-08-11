"""Existing-utility display filtering for the full-walking terrain LMM."""

from __future__ import annotations

import math

import numpy as np

from .gear_action import mujoco_to_isaaclab_joint_vector
from .hybrid_terrain_interactive import KinematicPose, PoseInertializer
from .offline_corpus import BODY_NAMES
from .render_terrain_transition_mesh import MUJOCO_JOINT_NAMES, build_qpos
from .terrain_foot_lock import G1TerrainFootLock, G1TerrainTransitionGuard
from .terrain_oracle.math3d import angular_velocity_world_wxyz
from .terrain_pose_repair import G1TerrainPoseRepair


_IDENTITY = (
    "existing-pose-inertializer-repair-"
    "source-proximity-acquire-source-continue-foot-lock/v6"
)


class _ObservedPoseRepairer:
    """Count every repair call made internally by the transition guard."""

    def __init__(self, delegate: object, observer: object) -> None:
        self.delegate = delegate
        self.observer = observer

    def repair(self, pose: KinematicPose) -> object:
        result = self.delegate.repair(pose)
        self.observer(result)
        return result


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
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            for joint_name in MUJOCO_JOINT_NAMES
        )
        self._body_ids = tuple(
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
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
            body_linear_velocity = (body_position - self._previous_body_position) / step
            body_angular_velocity = np.asarray(
                angular_velocity_world_wxyz(
                    np.stack((self._previous_body_quaternion_wxyz, body_wxyz), axis=0),
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
            raise ValueError("inertialization_halflife_s must be finite and positive")
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
        self.transition_guard = G1TerrainTransitionGuard(
            _ObservedPoseRepairer(self.pose_repairer, self._record_pose_repair),
            self.foot_locker,
            contact_hold_frames=4,
            track_source_contacts=False,
            source_contact_delay_frames=4,
            dt_s=1.0 / rate,
        )
        self.reset()

    def reset(self) -> None:
        """Clear adapter, inertializer, displayed pose, and foot-lock state."""

        self.pose_adapter.reset()
        self.transition_guard.reset()
        self._inertializer: PoseInertializer | None = None
        self._inertializer_elapsed_s = 0.0
        self._displayed_pose: KinematicPose | None = None
        self._displayed_qpos: np.ndarray | None = None
        self._previous_row: int | None = None
        self._previous_range_index: int | None = None
        self._continuous_lock_started = False
        self.pose_repair_count = 0
        self.foot_lock_accept_count = 0
        self.foot_lock_bypass_count = 0
        self.pose_repair_rejection_count = 0
        self.continuous_lock_active_frame_count = 0
        self.continuous_lock_releasing_frame_count = 0
        self.continuous_lock_idle_frame_count = 0
        self.continuous_lock_recovery_count = 0
        self.continuous_lock_bypass_count = 0
        self.source_proximity_candidate_frame_count = 0
        self.source_proximity_acquisition_frame_count = 0
        self.source_without_proximity_frame_count = 0
        self.proximity_without_source_frame_count = 0
        self.trusted_source_continuation_frame_count = 0
        self.trusted_source_override_frame_count = 0
        self.trusted_source_release_frame_count = 0
        self.trusted_pre_repair_recovery_attempt_count = 0
        self.trusted_pre_repair_recovery_success_count = 0
        self.trusted_pre_repair_recovery_failure_count = 0
        self.trusted_pre_repair_recovery_lock_reject_count = 0
        self.trusted_pre_repair_recovery_post_reject_count = 0
        self.last_measured_stance_contact = (False, False)
        self.last_ground_proximity_contact = (False, False)
        self.last_trusted_landing_contact = (False, False)
        self.last_published_locked_contact = (False, False)
        self.last_pose_repair_rejection: dict[str, float] | None = None
        self.last_reason = "reset"

    def _record_pose_repair(self, result: object) -> None:
        if bool(getattr(result, "accepted", False)):
            self.pose_repair_count += 1
            return
        self.pose_repair_rejection_count += 1
        metrics: dict[str, float] = {}
        for name in (
            "maximum_forbidden_penetration_after_m",
            "maximum_foot_penetration_after_m",
            "maximum_joint_delta_rad",
        ):
            try:
                value = float(getattr(result, name))
            except (AttributeError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                metrics[name] = value
        self.last_pose_repair_rejection = metrics or None
        explicit_reason = getattr(result, "reason", None)
        if isinstance(explicit_reason, str) and explicit_reason:
            self.last_reason = explicit_reason
        elif metrics:
            self.last_reason = (
                "pose repair rejected: forbidden "
                f"{metrics.get('maximum_forbidden_penetration_after_m', math.nan):.6f} m; "
                "foot "
                f"{metrics.get('maximum_foot_penetration_after_m', math.nan):.6f} m; "
                "joint "
                f"{metrics.get('maximum_joint_delta_rad', math.nan):.6f} rad"
            )
        else:
            self.last_reason = "pose repair rejected"

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
        # An untrusted foot needs both authenticated corpus contact and
        # measured ground proximity for two displayed frames.  Speed is not
        # an acquisition authority because a search splice can itself inflate
        # finite-difference speed.  Once trusted, source continuation/release
        # preserves the existing world-space stance target.
        if not self._continuous_lock_started:
            self.transition_guard.begin_landing(
                candidate,
                defer_contact_acquisition=True,
            )
            self._continuous_lock_started = True
        prior_trusted_contact = self.transition_guard.trusted_landing_contact
        displayed_pose = self.transition_guard.filter_landing(
            candidate,
            trusted_source_contact=contacts,
        )
        lock_result = self.transition_guard.last_foot_lock_result
        outcome = self.transition_guard.landing_filter_outcome
        outcome_reason = self.transition_guard.landing_filter_reason
        self.last_measured_stance_contact = (
            self.transition_guard.measured_stance_contact
        )
        self.last_ground_proximity_contact = (
            self.transition_guard.ground_proximity_contact
        )
        self.last_trusted_landing_contact = (
            self.transition_guard.trusted_landing_contact
        )
        published_filtered_pose = displayed_pose is not None
        self.last_published_locked_contact = (False, False)
        if (
            published_filtered_pose
            and lock_result is not None
            and bool(getattr(lock_result, "accepted", False))
        ):
            published_locked = tuple(getattr(lock_result, "locked", ()))
            if len(published_locked) == 2:
                self.last_published_locked_contact = (
                    bool(published_locked[0]),
                    bool(published_locked[1]),
                )
        if displayed_pose is None:
            self.continuous_lock_bypass_count += 1
            if lock_result is not None:
                self.foot_lock_bypass_count += 1
            # A pre-lock or post-lock repair rejection must not freeze source
            # playback.  A post-lock rejection may already have committed a
            # target for an unpublished pose, so clear every landing temporal
            # field and re-prime from a later advancing candidate.
            self.transition_guard.reset()
            self._continuous_lock_started = False
            self.last_measured_stance_contact = (
                self.transition_guard.measured_stance_contact
            )
            self.last_ground_proximity_contact = (
                self.transition_guard.ground_proximity_contact
            )
            self.last_trusted_landing_contact = (
                self.transition_guard.trusted_landing_contact
            )
            displayed_pose = candidate
        elif lock_result is not None:
            if bool(getattr(lock_result, "accepted", False)):
                self.foot_lock_accept_count += 1
            else:
                self.foot_lock_bypass_count += 1

        policy_applied = bool(
            published_filtered_pose
            and lock_result is not None
            and getattr(lock_result, "accepted", False)
            and outcome
            not in {
                "release-after-reject",
                "bypass-after-double-reject",
            }
        )
        if policy_applied:
            candidate_evidence = tuple(
                not prior_trusted_contact[foot]
                and bool(contacts[foot])
                and self.last_ground_proximity_contact[foot]
                for foot in range(2)
            )
            acquired = tuple(
                not prior_trusted_contact[foot]
                and self.last_trusted_landing_contact[foot]
                for foot in range(2)
            )
            source_without_proximity = tuple(
                not prior_trusted_contact[foot]
                and bool(contacts[foot])
                and not self.last_ground_proximity_contact[foot]
                for foot in range(2)
            )
            proximity_without_source = tuple(
                not prior_trusted_contact[foot]
                and not bool(contacts[foot])
                and self.last_ground_proximity_contact[foot]
                for foot in range(2)
            )
            continued = tuple(
                prior_trusted_contact[foot]
                and bool(contacts[foot])
                and self.last_trusted_landing_contact[foot]
                for foot in range(2)
            )
            overridden = tuple(
                continued[foot] and not self.last_measured_stance_contact[foot]
                for foot in range(2)
            )
            released = tuple(
                prior_trusted_contact[foot]
                and not bool(contacts[foot])
                and not self.last_trusted_landing_contact[foot]
                for foot in range(2)
            )
            self.source_proximity_candidate_frame_count += int(
                any(candidate_evidence)
            )
            self.source_proximity_acquisition_frame_count += int(any(acquired))
            self.source_without_proximity_frame_count += int(
                any(source_without_proximity)
            )
            self.proximity_without_source_frame_count += int(
                any(proximity_without_source)
            )
            self.trusted_source_continuation_frame_count += int(any(continued))
            self.trusted_source_override_frame_count += int(any(overridden))
            self.trusted_source_release_frame_count += int(any(released))

        successful_pre_repair_recovery = outcome in {
            "pre-repair-recovery-active",
            "pre-repair-recovery-releasing",
            "pre-repair-recovery-idle",
        }
        failed_pre_repair_recovery = outcome in {
            "pre-repair-recovery-lock-rejected",
            "pre-repair-recovery-post-repair-rejected",
        }
        if successful_pre_repair_recovery or failed_pre_repair_recovery:
            self.trusted_pre_repair_recovery_attempt_count += 1
        if successful_pre_repair_recovery:
            self.trusted_pre_repair_recovery_success_count += 1
        elif failed_pre_repair_recovery:
            self.trusted_pre_repair_recovery_failure_count += 1
        if outcome == "pre-repair-recovery-lock-rejected":
            self.trusted_pre_repair_recovery_lock_reject_count += 1
        elif outcome == "pre-repair-recovery-post-repair-rejected":
            self.trusted_pre_repair_recovery_post_reject_count += 1

        if outcome in {"active", "pre-repair-recovery-active"}:
            self.continuous_lock_active_frame_count += 1
        elif outcome in {
            "releasing",
            "release-after-reject",
            "pre-repair-recovery-releasing",
        }:
            self.continuous_lock_releasing_frame_count += 1
        elif outcome in {"idle", "pre-repair-recovery-idle"}:
            self.continuous_lock_idle_frame_count += 1
        if outcome == "release-after-reject" or successful_pre_repair_recovery:
            self.continuous_lock_recovery_count += 1
            self.last_reason = outcome_reason
        elif failed_pre_repair_recovery:
            self.last_reason = outcome_reason
        elif outcome == "bypass-after-double-reject":
            # A returned pose needs the explicit outcome count.  A rejected
            # transaction was already counted by the ``None`` bypass above.
            if published_filtered_pose:
                self.continuous_lock_bypass_count += 1
            self.last_reason = outcome_reason

        if not isinstance(displayed_pose, KinematicPose):
            raise ValueError("terrain continuous foot lock returned an invalid pose")

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
            "contact_policy": (
                "authenticated-source-and-measured-ground-proximity-two-frame-"
                "acquire-source-continue-release"
            ),
            "source_contacts_required_for_acquisition": True,
            "measured_ground_proximity_required_for_acquisition": True,
            "measured_speed_used_for_acquisition": False,
            "source_contacts_used_for_trusted_continuation": True,
            "measured_stance_maximum_foot_speed_mps": (
                self.transition_guard.landing_maximum_foot_speed_mps
            ),
            "ground_proximity_maximum_sole_clearance_m": (
                self.transition_guard.landing_maximum_sole_clearance_m
            ),
            "source_proximity_acquire_frames": (
                self.transition_guard.landing_contact_acquire_frames
            ),
            "minimum_swing_clearance_m": (
                self.transition_guard.landing_minimum_swing_clearance_m
            ),
            "foot_lock_release_halflife_s": float(
                getattr(self.foot_locker, "release_halflife_s", 0.08)
            ),
            "continuous_lock_dt_s": self.transition_guard.dt_s,
            "pose_repair_count": self.pose_repair_count,
            "foot_lock_accept_count": self.foot_lock_accept_count,
            "foot_lock_bypass_count": self.foot_lock_bypass_count,
            "pose_repair_rejection_count": self.pose_repair_rejection_count,
            "continuous_lock_active_frame_count": (
                self.continuous_lock_active_frame_count
            ),
            "continuous_lock_releasing_frame_count": (
                self.continuous_lock_releasing_frame_count
            ),
            "continuous_lock_idle_frame_count": (self.continuous_lock_idle_frame_count),
            "continuous_lock_recovery_count": (self.continuous_lock_recovery_count),
            "continuous_lock_bypass_count": (self.continuous_lock_bypass_count),
            "source_proximity_candidate_frame_count": (
                self.source_proximity_candidate_frame_count
            ),
            "source_proximity_acquisition_frame_count": (
                self.source_proximity_acquisition_frame_count
            ),
            "source_without_proximity_frame_count": (
                self.source_without_proximity_frame_count
            ),
            "proximity_without_source_frame_count": (
                self.proximity_without_source_frame_count
            ),
            "trusted_source_continuation_frame_count": (
                self.trusted_source_continuation_frame_count
            ),
            "trusted_source_override_frame_count": (
                self.trusted_source_override_frame_count
            ),
            "trusted_source_release_frame_count": (
                self.trusted_source_release_frame_count
            ),
            "trusted_pre_repair_recovery_attempt_count": (
                self.trusted_pre_repair_recovery_attempt_count
            ),
            "trusted_pre_repair_recovery_success_count": (
                self.trusted_pre_repair_recovery_success_count
            ),
            "trusted_pre_repair_recovery_failure_count": (
                self.trusted_pre_repair_recovery_failure_count
            ),
            "trusted_pre_repair_recovery_lock_reject_count": (
                self.trusted_pre_repair_recovery_lock_reject_count
            ),
            "trusted_pre_repair_recovery_post_reject_count": (
                self.trusted_pre_repair_recovery_post_reject_count
            ),
            "last_measured_stance_contact": (self.last_measured_stance_contact),
            "last_ground_proximity_contact": (
                self.last_ground_proximity_contact
            ),
            "last_trusted_landing_contact": (self.last_trusted_landing_contact),
            "last_published_locked_contact": (
                self.last_published_locked_contact
            ),
            "last_pose_repair_rejection": (
                None
                if self.last_pose_repair_rejection is None
                else dict(self.last_pose_repair_rejection)
            ),
            "last_reason": self.last_reason,
        }


__all__ = ["ExistingUtilityPosePostprocessor", "NativeQposPoseAdapter"]
