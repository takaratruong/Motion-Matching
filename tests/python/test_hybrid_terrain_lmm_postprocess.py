from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from resources.g1_terrain_builder.schema import ArtifactSet, FeatureSet

from mm_sonic.hybrid_terrain_interactive import KinematicPose
from mm_sonic.hybrid_terrain_lmm_postprocess import (
    ExistingUtilityPosePostprocessor,
    NativeQposPoseAdapter,
)
from mm_sonic.hybrid_terrain_lmm_runtime import HybridMatcher
from mm_sonic.hybrid_terrain_lmm_viewer import DEFAULT_G1_XML
from mm_sonic.hybrid_terrain_lmm_viewer import (
    build_diagnostic_collision_model,
    load_scene_terrain,
)
from mm_sonic.render_terrain_transition_mesh import build_qpos
from mm_sonic.terrain_foot_lock import G1TerrainFootLock, G1TerrainTransitionGuard


@pytest.fixture(scope="module")
def model() -> object:
    assert Path(DEFAULT_G1_XML).is_file()
    return mujoco.MjModel.from_xml_path(str(DEFAULT_G1_XML))


def _qpos(model: object, *, x: float, joint_offset: float = 0.0) -> np.ndarray:
    value = np.array(model.qpos0, dtype=np.float64, copy=True)
    value[0] = x
    value[7:] += joint_offset
    return value


def test_native_qpos_adapter_round_trips_native_g1_order_and_quaternion(model) -> None:
    adapter = NativeQposPoseAdapter(model)
    qpos = _qpos(model, x=0.25, joint_offset=0.01)
    qpos[3:7] = np.asarray((0.9238795325, 0.0, 0.0, 0.3826834324))

    pose = adapter.to_pose(qpos, dt_s=1.0 / 60.0)
    roundtrip = build_qpos(
        pose.root_position_world,
        pose.root_orientation_world_xyzw,
        pose.joint_position,
        model,
    )

    np.testing.assert_allclose(roundtrip, qpos, atol=1.0e-6, rtol=0.0)
    np.testing.assert_allclose(
        pose.root_orientation_world_xyzw,
        qpos[[4, 5, 6, 3]],
        atol=1.0e-7,
        rtol=0.0,
    )


def test_native_qpos_adapter_derives_finite_velocities(model) -> None:
    adapter = NativeQposPoseAdapter(model)
    adapter.to_pose(_qpos(model, x=0.0), dt_s=1.0 / 60.0)
    pose = adapter.to_pose(_qpos(model, x=0.01, joint_offset=0.005), dt_s=1.0 / 60.0)

    assert pose.joint_velocity.shape == (29,)
    assert pose.body_linear_velocity_world.shape == (30, 3)
    assert pose.body_angular_velocity_world.shape == (30, 3)
    assert np.isfinite(pose.joint_velocity).all()
    assert np.isfinite(pose.body_linear_velocity_world).all()
    assert np.isfinite(pose.body_angular_velocity_world).all()
    assert np.linalg.norm(pose.joint_velocity) > 0.0


class _RecordingRepairer:
    def __init__(self) -> None:
        self.calls: list[KinematicPose] = []
        self.outcomes: list[tuple[bool, bool, str]] = []

    def queue(
        self, accepted: bool, *, repaired: bool = False, reason: str = "rejected"
    ) -> None:
        self.outcomes.append((accepted, repaired, reason))

    def repair(self, pose: KinematicPose) -> object:
        self.calls.append(pose)
        accepted, repaired, reason = (
            self.outcomes.pop(0) if self.outcomes else (True, False, "accepted")
        )
        return SimpleNamespace(
            pose=pose, accepted=accepted, repaired=repaired, reason=reason
        )


class _KneeOffsetRepairer(_RecordingRepairer):
    def __init__(self) -> None:
        super().__init__()
        self.knee_offsets: list[float] = []

    def repair(self, pose: KinematicPose) -> object:
        self.calls.append(pose)
        offset = self.knee_offsets.pop(0) if self.knee_offsets else 0.0
        joints = pose.joint_position.copy()
        # IsaacLab index 9 maps to MuJoCo qpos index 10 (left knee).
        joints[9] += offset
        repaired = replace(pose, joint_position=joints)
        return SimpleNamespace(
            pose=repaired,
            accepted=True,
            repaired=abs(offset) > 0.0,
            reason="accepted",
        )


class _RecordingFootLocker:
    def __init__(self) -> None:
        self.calls: list[tuple[KinematicPose, np.ndarray, float]] = []
        self.support_calls: list[KinematicPose] = []
        self.measured_support = np.asarray((False, True), dtype=bool)
        self.accepted: list[bool] = []
        self.reset_calls = 0
        self.restore_calls = 0
        self._serial = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def snapshot_state(self) -> object:
        self._serial += 1
        return ("snapshot", self._serial)

    def restore_state(self, snapshot: object) -> None:
        assert snapshot == ("snapshot", self._serial)
        self.restore_calls += 1

    def support_contact(
        self,
        pose: KinematicPose,
        *,
        maximum_sole_clearance_m: float = 0.020,
    ) -> np.ndarray:
        assert maximum_sole_clearance_m == pytest.approx(0.020)
        self.support_calls.append(pose)
        return self.measured_support.copy()

    def landing_stance_contact(
        self,
        pose: KinematicPose,
        *,
        maximum_foot_speed_mps: float = 0.20,
        maximum_sole_clearance_m: float = 0.020,
        previous_pose: KinematicPose | None = None,
        dt_s: float | None = None,
    ) -> np.ndarray:
        del maximum_foot_speed_mps, maximum_sole_clearance_m, previous_pose, dt_s
        self.support_calls.append(pose)
        return self.measured_support.copy()

    def apply(
        self,
        pose: KinematicPose,
        source_contact: object,
        *,
        dt_s: float,
        minimum_swing_clearance_m: float = 0.0,
    ) -> object:
        del minimum_swing_clearance_m
        self.calls.append(
            (pose, np.array(source_contact, dtype=bool, copy=True), float(dt_s))
        )
        accepted = self.accepted.pop(0) if self.accepted else True
        return SimpleNamespace(
            pose=pose,
            accepted=accepted,
            repaired=False,
            locked=(bool(source_contact[0]), bool(source_contact[1])),
            releasing=(False, False),
            reason="accepted" if accepted else "lock rejected",
        )


class _MeasuredContinuousFootLocker(_RecordingFootLocker):
    def __init__(self) -> None:
        super().__init__()
        self.measured_stance = np.asarray((True, False), dtype=bool)
        self.measured_stance_queue: list[np.ndarray] = []
        self.landing_stance_calls: list[
            tuple[KinematicPose, KinematicPose | None, float | None]
        ] = []
        self.swing_clearances: list[float] = []
        self.releasing_queue: list[tuple[bool, bool]] = []

    def landing_stance_contact(
        self,
        pose: KinematicPose,
        *,
        maximum_foot_speed_mps: float = 0.20,
        maximum_sole_clearance_m: float = 0.020,
        previous_pose: KinematicPose | None = None,
        dt_s: float | None = None,
    ) -> np.ndarray:
        assert maximum_foot_speed_mps == pytest.approx(0.20)
        assert maximum_sole_clearance_m == pytest.approx(0.020)
        self.landing_stance_calls.append((pose, previous_pose, dt_s))
        return np.array(
            self.measured_stance_queue.pop(0)
            if self.measured_stance_queue
            else self.measured_stance,
            dtype=bool,
            copy=True,
        )

    def apply(
        self,
        pose: KinematicPose,
        source_contact: object,
        *,
        dt_s: float,
        minimum_swing_clearance_m: float = 0.0,
    ) -> object:
        contact = np.array(source_contact, dtype=bool, copy=True)
        self.calls.append((pose, contact, float(dt_s)))
        self.swing_clearances.append(float(minimum_swing_clearance_m))
        accepted = self.accepted.pop(0) if self.accepted else True
        return SimpleNamespace(
            pose=pose,
            accepted=accepted,
            repaired=False,
            locked=(accepted and bool(contact[0]), accepted and bool(contact[1])),
            releasing=(
                self.releasing_queue.pop(0)
                if self.releasing_queue
                else (False, False)
            ),
            reason="accepted" if accepted else "lock rejected",
        )


class _StepBoundedKneeFootLocker(_MeasuredContinuousFootLocker):
    """Model the existing lock's unbounded first correction and step limit."""

    def __init__(self) -> None:
        super().__init__()
        self._correction = 0.0
        self._correction_initialized = False

    def reset(self) -> None:
        super().reset()
        self._correction = 0.0
        self._correction_initialized = False

    def apply(
        self,
        pose: KinematicPose,
        source_contact: object,
        *,
        dt_s: float,
        minimum_swing_clearance_m: float = 0.0,
    ) -> object:
        contact = np.array(source_contact, dtype=bool, copy=True)
        self.calls.append((pose, contact, float(dt_s)))
        clearance = float(minimum_swing_clearance_m)
        self.swing_clearances.append(clearance)
        desired = 0.25 if clearance > 0.0 else 0.0
        if self._correction_initialized:
            desired = self._correction + float(
                np.clip(desired - self._correction, -0.08, 0.08)
            )
        self._correction = desired
        self._correction_initialized = True
        joints = pose.joint_position.copy()
        joints[9] += self._correction
        filtered = replace(pose, joint_position=joints)
        return SimpleNamespace(
            pose=filtered,
            accepted=True,
            repaired=abs(self._correction) > 0.0,
            locked=(bool(contact[0]), bool(contact[1])),
            releasing=(False, False),
            reason="accepted",
        )


def _processor(model):
    repairer = _RecordingRepairer()
    locker = _RecordingFootLocker()
    processor = ExistingUtilityPosePostprocessor(
        model,
        SimpleNamespace(height_at_world_xy=lambda _xy: 0.0),
        pose_repairer=repairer,
        foot_locker=locker,
    )
    return processor, repairer, locker


def _continuous_processor(model):
    repairer = _RecordingRepairer()
    locker = _MeasuredContinuousFootLocker()
    processor = ExistingUtilityPosePostprocessor(
        model,
        SimpleNamespace(height_at_world_xy=lambda _xy: 0.0),
        pose_repairer=repairer,
        foot_locker=locker,
    )
    return processor, repairer, locker


def test_source_false_scraping_proximity_resets_two_frame_acquisition(
    model,
) -> None:
    processor, _repairer, locker = _continuous_processor(model)
    # The visible foot is close to terrain but exceeds the stance-speed gate,
    # as happens during a motion-match splice.
    locker.measured_stance = np.asarray((False, False), dtype=bool)
    locker.measured_support = np.asarray((True, False), dtype=bool)
    dt = 1.0 / 60.0
    source_contacts = (
        (True, False),
        (False, False),
        (True, False),
        (True, False),
    )

    for index, source_contact in enumerate(source_contacts):
        processor.step(
            _qpos(model, x=0.01 * index),
            row=10 + index,
            range_index=0,
            source_contact=np.asarray(source_contact, dtype=bool),
            dt_s=dt,
        )

    # The source-false scraping frame resets acquisition.  Only the final two
    # consecutive source-plus-proximity observations earn the left-foot lock.
    assert len(locker.landing_stance_calls) == len(source_contacts) + 1
    assert len(locker.calls) == len(source_contacts) + 1
    assert locker.reset_calls == 3
    for _pose, contact, call_dt in locker.calls[:-1]:
        np.testing.assert_array_equal(contact, (False, False))
        assert call_dt == pytest.approx(dt)
    np.testing.assert_array_equal(locker.calls[-1][1], (True, False))
    assert locker.calls[-1][2] == pytest.approx(dt)
    assert locker.swing_clearances == pytest.approx(
        [0.0] + [0.012] * len(source_contacts)
    )
    identity = processor.identity()
    assert identity["source_contacts_required_for_acquisition"] is True
    assert identity["measured_ground_proximity_required_for_acquisition"] is True
    assert identity["measured_speed_used_for_acquisition"] is False
    assert identity["source_contacts_used_for_trusted_continuation"] is True
    assert identity["last_measured_stance_contact"] == (False, False)
    assert identity["last_ground_proximity_contact"] == (True, False)
    assert identity["last_trusted_landing_contact"] == (True, False)
    assert identity["last_published_locked_contact"] == (True, False)
    assert identity["source_proximity_candidate_frame_count"] == 3
    assert identity["source_proximity_acquisition_frame_count"] == 1
    assert identity["proximity_without_source_frame_count"] == 1


def test_source_proximity_acquisition_ignores_transition_speed(model) -> None:
    processor, _repairer, locker = _continuous_processor(model)
    locker.measured_stance = np.asarray((False, False), dtype=bool)
    locker.measured_support = np.asarray((True, False), dtype=bool)
    dt = 1.0 / 60.0

    processor.step(
        _qpos(model, x=0.0),
        row=10,
        range_index=0,
        source_contact=np.asarray((True, False), dtype=bool),
        dt_s=dt,
    )
    processor.step(
        _qpos(model, x=0.01),
        row=11,
        range_index=0,
        source_contact=np.asarray((True, False), dtype=bool),
        dt_s=dt,
    )

    np.testing.assert_array_equal(locker.calls[0][1], (False, False))
    np.testing.assert_array_equal(locker.calls[1][1], (False, False))
    np.testing.assert_array_equal(locker.calls[2][1], (True, False))


def test_source_contact_alone_cannot_acquire_without_ground_proximity(model) -> None:
    processor, _repairer, locker = _continuous_processor(model)
    locker.measured_stance = np.asarray((False, False), dtype=bool)
    locker.measured_support = np.asarray((False, False), dtype=bool)
    dt = 1.0 / 60.0

    for index in range(3):
        processor.step(
            _qpos(model, x=0.01 * index),
            row=10 + index,
            range_index=0,
            source_contact=np.asarray((True, False), dtype=bool),
            dt_s=dt,
        )

    for _pose, contact, _call_dt in locker.calls:
        np.testing.assert_array_equal(contact, (False, False))
    identity = processor.identity()
    assert identity["last_ground_proximity_contact"] == (False, False)
    assert identity["last_published_locked_contact"] == (False, False)
    assert identity["source_without_proximity_frame_count"] == 3
    assert identity["source_proximity_acquisition_frame_count"] == 0


def test_existing_support_contact_rejects_136mm_airborne_feet(model) -> None:
    scene = SimpleNamespace(height_at_world_xy=lambda _xy: 0.0)
    locker = G1TerrainFootLock(model, scene)
    adapter = NativeQposPoseAdapter(model)
    airborne = _qpos(model, x=0.0)
    airborne[2] += 0.136

    pose = adapter.to_pose(airborne, dt_s=1.0 / 60.0)

    np.testing.assert_array_equal(locker.support_contact(pose), (False, False))


def test_diagnostic_owned_locker_uses_strict_active_and_release_residuals(
    model,
) -> None:
    scene = SimpleNamespace(height_at_world_xy=lambda _xy: 0.0)

    processor = ExistingUtilityPosePostprocessor(model, scene)
    diagnostic = processor.foot_locker
    normal = G1TerrainFootLock(model, scene)

    assert diagnostic.maximum_locked_foot_drift_m == pytest.approx(0.0005)
    assert diagnostic.maximum_releasing_foot_drift_m == pytest.approx(0.0005)
    assert diagnostic.defer_swing_clearance_until_release_complete is True
    assert normal.maximum_locked_foot_drift_m == pytest.approx(0.010)
    assert normal.maximum_releasing_foot_drift_m is None
    assert normal.defer_swing_clearance_until_release_complete is False
    identity = processor.identity()
    assert identity["diagnostic_display_postprocessor"] == (
        "existing-pose-inertializer-repair-source-proximity-acquire-"
        "source-continue-strict-final-foot-lock/v7"
    )
    assert identity["diagnostic_owned_foot_locker"] is True
    assert identity["foot_lock_maximum_locked_foot_drift_m"] == pytest.approx(
        0.0005
    )
    assert identity["foot_lock_maximum_releasing_foot_drift_m"] == pytest.approx(
        0.0005
    )
    assert identity["foot_lock_maximum_joint_correction_rad"] == pytest.approx(0.25)
    assert identity["foot_lock_maximum_joint_correction_step_rad"] == pytest.approx(
        0.08
    )
    assert identity["trusted_source_lock_final_authority"] is True
    assert identity["trusted_post_lock_safety_validator_requires_noop"] is True


def test_release_retry_rejects_an_unsolved_existing_release_target(model) -> None:
    scene = SimpleNamespace(height_at_world_xy=lambda _xy: 0.0)
    locker = G1TerrainFootLock(
        model,
        scene,
        maximum_locked_foot_drift_m=0.0005,
    )
    # Set the wished-for diagnostic policy explicitly so this test fails on
    # behavior, rather than on the new constructor API being absent.
    locker.maximum_releasing_foot_drift_m = 0.0005
    locker.defer_swing_clearance_until_release_complete = True
    adapter = NativeQposPoseAdapter(model)
    initial_qpos = _qpos(model, x=0.0)
    initial_qpos[2] += 0.024
    shifted_qpos = initial_qpos.copy()
    shifted_qpos[0] += 0.040
    initial = adapter.to_pose(initial_qpos, dt_s=1.0 / 60.0)
    shifted = adapter.to_pose(shifted_qpos, dt_s=1.0 / 60.0)

    prime = locker.apply(
        initial,
        np.asarray((True, False), dtype=bool),
        dt_s=1.0 / 60.0,
        minimum_swing_clearance_m=0.0,
    )
    active = locker.apply(
        shifted,
        np.asarray((True, False), dtype=bool),
        dt_s=1.0 / 60.0,
        minimum_swing_clearance_m=0.012,
    )
    released = locker.apply(
        shifted,
        np.asarray((False, False), dtype=bool),
        dt_s=1.0 / 60.0,
        minimum_swing_clearance_m=0.012,
    )

    assert prime.accepted is True
    assert active.accepted is False
    assert active.maximum_locked_foot_drift_m > 0.0005
    assert released.accepted is False
    assert released.maximum_releasing_foot_drift_m > 0.0005
    assert "releasing-foot drift exceeds 0.5 mm" in released.reason


def test_swing_clearance_waits_until_existing_release_target_expires(model) -> None:
    scene = SimpleNamespace(height_at_world_xy=lambda _xy: 0.0)
    locker = G1TerrainFootLock(
        model,
        scene,
        maximum_locked_foot_drift_m=0.0005,
    )
    locker.maximum_releasing_foot_drift_m = 0.0005
    locker.defer_swing_clearance_until_release_complete = True
    qpos = _qpos(model, x=0.0)
    qpos[2] += 0.024
    pose = NativeQposPoseAdapter(model).to_pose(qpos, dt_s=1.0 / 60.0)

    prime = locker.apply(
        pose,
        np.asarray((True, True), dtype=bool),
        dt_s=1.0 / 60.0,
        minimum_swing_clearance_m=0.0,
    )
    released = locker.apply(
        pose,
        np.asarray((False, True), dtype=bool),
        dt_s=1.0 / 60.0,
        minimum_swing_clearance_m=0.012,
    )
    planar_step = float(
        np.max(
            np.linalg.norm(
                released.sole_world_positions[0][:, :2]
                - prime.sole_world_positions[0][:, :2],
                axis=1,
            )
        )
    )

    assert prime.accepted is True
    assert released.accepted is True
    assert released.releasing == (True, False)
    assert planar_step <= 0.001


def test_guard_without_source_keeps_speed_qualified_acquisition_policy(model) -> None:
    repairer = _RecordingRepairer()
    locker = _MeasuredContinuousFootLocker()
    locker.measured_stance = np.asarray((False, False), dtype=bool)
    locker.measured_support = np.asarray((True, False), dtype=bool)
    guard = G1TerrainTransitionGuard(
        repairer,
        locker,
        contact_hold_frames=4,
        source_contact_delay_frames=4,
        dt_s=1.0 / 60.0,
    )
    pose = NativeQposPoseAdapter(model).to_pose(
        _qpos(model, x=0.0), dt_s=1.0 / 60.0
    )

    guard.begin_landing(pose, defer_contact_acquisition=True)
    guard.filter_landing(pose)
    guard.filter_landing(pose)

    for _pose, contact, _call_dt in locker.calls:
        np.testing.assert_array_equal(contact, (False, False))
    # The speed-free support utility is diagnostic-source policy only.
    assert locker.support_calls == []


def test_trusted_final_validator_mutation_bypasses_stale_lock_pose(model) -> None:
    repairer = _KneeOffsetRepairer()
    locker = _MeasuredContinuousFootLocker()
    guard = G1TerrainTransitionGuard(
        repairer,
        locker,
        contact_hold_frames=4,
        source_contact_delay_frames=4,
        dt_s=1.0 / 60.0,
    )
    guard.trusted_source_lock_final_authority = True
    pose = NativeQposPoseAdapter(model).to_pose(
        _qpos(model, x=0.0), dt_s=1.0 / 60.0
    )
    guard.begin_landing(pose)
    repairer.knee_offsets[:] = [0.0, 0.10]

    filtered = guard.filter_landing(
        pose,
        trusted_source_contact=np.asarray((True, False), dtype=bool),
    )

    assert filtered is None
    assert guard.landing_filter_outcome == "post-repair-mutated"
    assert "mutated accepted foot-lock pose" in guard.landing_filter_reason


def test_no_source_path_still_publishes_existing_post_repair(model) -> None:
    repairer = _KneeOffsetRepairer()
    locker = _MeasuredContinuousFootLocker()
    guard = G1TerrainTransitionGuard(
        repairer,
        locker,
        contact_hold_frames=4,
        source_contact_delay_frames=4,
        dt_s=1.0 / 60.0,
    )
    guard.trusted_source_lock_final_authority = True
    pose = NativeQposPoseAdapter(model).to_pose(
        _qpos(model, x=0.0), dt_s=1.0 / 60.0
    )
    guard.begin_landing(pose)
    repairer.knee_offsets[:] = [0.0, 0.10]

    filtered = guard.filter_landing(pose)

    assert isinstance(filtered, KinematicPose)
    assert filtered.joint_position[9] - pose.joint_position[9] == pytest.approx(0.10)


def test_published_release_mask_is_zeroed_on_following_bypass(model) -> None:
    processor, repairer, locker = _continuous_processor(model)
    locker.measured_support = np.asarray((True, False), dtype=bool)
    source = np.asarray((True, False), dtype=bool)
    dt = 1.0 / 60.0
    for row in (1, 2):
        processor.step(
            _qpos(model, x=0.01 * (row - 1)),
            row=row,
            range_index=0,
            source_contact=source,
            dt_s=dt,
        )
    locker.releasing_queue[:] = [(True, False)]

    processor.step(
        _qpos(model, x=0.02),
        row=3,
        range_index=0,
        source_contact=np.asarray((False, False), dtype=bool),
        dt_s=dt,
    )

    identity = processor.identity()
    assert identity["last_published_locked_contact"] == (False, False)
    assert identity["last_published_releasing_contact"] == (True, False)

    repairer.queue(False, reason="pre-repair rejected")
    processor.step(
        _qpos(model, x=0.03),
        row=4,
        range_index=0,
        source_contact=np.asarray((False, False), dtype=bool),
        dt_s=dt,
    )

    assert processor.identity()["last_published_releasing_contact"] == (
        False,
        False,
    )


def test_trusted_recovery_validator_mutation_restores_transaction(model) -> None:
    repairer = _RecordingRepairer()
    locker = _MeasuredContinuousFootLocker()
    guard = G1TerrainTransitionGuard(
        repairer,
        locker,
        contact_hold_frames=4,
        source_contact_delay_frames=4,
        dt_s=1.0 / 60.0,
    )
    guard.trusted_source_lock_final_authority = True
    pose = NativeQposPoseAdapter(model).to_pose(
        _qpos(model, x=0.0), dt_s=1.0 / 60.0
    )
    guard.begin_landing(pose)
    repairer.queue(False, reason="pre-repair rejected")
    repairer.queue(True, repaired=True)
    restores_before = locker.restore_calls

    filtered = guard.filter_landing(
        pose,
        trusted_source_contact=np.asarray((True, False), dtype=bool),
    )

    assert filtered is None
    assert locker.restore_calls == restores_before + 1
    assert guard.landing_filter_outcome == (
        "pre-repair-recovery-post-repair-mutated"
    )


def test_double_reject_clears_deferred_neutral_prime_state(model) -> None:
    repairer = _RecordingRepairer()
    locker = _MeasuredContinuousFootLocker()
    guard = G1TerrainTransitionGuard(
        repairer,
        locker,
        contact_hold_frames=4,
        source_contact_delay_frames=4,
        dt_s=1.0 / 60.0,
    )
    pose = NativeQposPoseAdapter(model).to_pose(
        _qpos(model, x=0.0), dt_s=1.0 / 60.0
    )
    guard.begin_landing(pose, defer_contact_acquisition=True)
    locker.accepted[:] = [False, False]

    filtered = guard.filter_landing(
        pose,
        trusted_source_contact=np.asarray((False, False), dtype=bool),
    )

    assert filtered is None
    assert guard._landing_neutral_lock_primed is False


def test_deferred_landing_primes_zero_correction_before_display_filter(model) -> None:
    repairer = _KneeOffsetRepairer()
    locker = _StepBoundedKneeFootLocker()
    processor = ExistingUtilityPosePostprocessor(
        model,
        SimpleNamespace(height_at_world_xy=lambda _xy: 0.0),
        pose_repairer=repairer,
        foot_locker=locker,
    )
    repairer.knee_offsets[:] = [0.25, 0.0]
    source = _qpos(model, x=0.0)

    displayed = processor.step(
        source,
        row=1,
        range_index=0,
        source_contact=np.asarray((True, False), dtype=bool),
        dt_s=1.0 / 60.0,
    )

    assert locker.swing_clearances == pytest.approx((0.0, 0.012))
    assert displayed[10] - source[10] == pytest.approx(0.33, abs=1.0e-6)


def test_deferred_landing_reprimes_after_raw_prime_rejection(model) -> None:
    repairer = _RecordingRepairer()
    locker = _MeasuredContinuousFootLocker()
    locker.accepted[:] = [False, True, True]
    guard = G1TerrainTransitionGuard(
        repairer,
        locker,
        contact_hold_frames=4,
        source_contact_delay_frames=4,
        dt_s=1.0 / 60.0,
    )
    pose = NativeQposPoseAdapter(model).to_pose(
        _qpos(model, x=0.0), dt_s=1.0 / 60.0
    )

    guard.begin_landing(pose, defer_contact_acquisition=True)
    filtered = guard.filter_landing(
        pose,
        trusted_source_contact=np.asarray((False, False), dtype=bool),
    )

    assert isinstance(filtered, KinematicPose)
    assert locker.swing_clearances == pytest.approx((0.0, 0.0, 0.012))


def test_normal_landing_prime_keeps_existing_swing_clearance(model) -> None:
    repairer = _RecordingRepairer()
    locker = _MeasuredContinuousFootLocker()
    guard = G1TerrainTransitionGuard(
        repairer,
        locker,
        contact_hold_frames=4,
        source_contact_delay_frames=4,
        dt_s=1.0 / 60.0,
    )
    pose = NativeQposPoseAdapter(model).to_pose(
        _qpos(model, x=0.0), dt_s=1.0 / 60.0
    )

    guard.begin_landing(pose)

    assert locker.swing_clearances == pytest.approx((0.012,))


def test_measured_acquired_contact_survives_search_transition_until_source_release(
    model,
) -> None:
    processor, _repairer, locker = _continuous_processor(model)
    locker.measured_support = np.asarray((True, False), dtype=bool)
    locker.measured_stance_queue[:] = [
        np.asarray((False, False), dtype=bool),
        np.asarray((True, False), dtype=bool),
        np.asarray((True, False), dtype=bool),
        np.asarray((False, False), dtype=bool),
        np.asarray((False, False), dtype=bool),
        np.asarray((False, False), dtype=bool),
    ]
    dt = 1.0 / 60.0

    for index, (row, range_index, source_contact) in enumerate(
        (
            (10, 0, (True, False)),
            (11, 0, (True, False)),
            (30, 1, (True, False)),
            (31, 1, (False, False)),
            (32, 1, (True, False)),
        )
    ):
        processor.step(
            _qpos(model, x=0.01 * index),
            row=row,
            range_index=range_index,
            source_contact=np.asarray(source_contact, dtype=bool),
            dt_s=dt,
        )

    np.testing.assert_array_equal(locker.calls[0][1], (False, False))
    np.testing.assert_array_equal(locker.calls[1][1], (False, False))
    np.testing.assert_array_equal(locker.calls[2][1], (True, False))
    # The discontinuous row/range jump creates a measured speed rejection,
    # but the authenticated source label may continue the already-trusted
    # left foot.  Source false then releases it, and stale source true cannot
    # reacquire while measured stance remains false.
    np.testing.assert_array_equal(locker.calls[3][1], (True, False))
    np.testing.assert_array_equal(locker.calls[4][1], (False, False))
    np.testing.assert_array_equal(locker.calls[5][1], (False, False))
    identity = processor.identity()
    assert identity["last_measured_stance_contact"] == (False, False)
    assert identity["last_trusted_landing_contact"] == (False, False)
    assert identity["trusted_source_continuation_frame_count"] == 1
    assert identity["trusted_source_override_frame_count"] == 1
    assert identity["trusted_source_release_frame_count"] == 1


def test_pose_repair_rejection_identity_reports_actionable_mechanical_values(
    model,
) -> None:
    class RejectingRepairer:
        @staticmethod
        def repair(pose: KinematicPose) -> object:
            return SimpleNamespace(
                pose=pose,
                accepted=False,
                repaired=False,
                maximum_forbidden_penetration_after_m=0.03125,
                maximum_foot_penetration_after_m=0.0065,
                maximum_joint_delta_rad=0.1875,
            )

    locker = _MeasuredContinuousFootLocker()
    processor = ExistingUtilityPosePostprocessor(
        model,
        SimpleNamespace(height_at_world_xy=lambda _xy: 0.0),
        pose_repairer=RejectingRepairer(),
        foot_locker=locker,
    )

    processor.step(
        _qpos(model, x=0.0),
        row=1,
        range_index=0,
        source_contact=np.asarray((True, True), dtype=bool),
        dt_s=1.0 / 60.0,
    )

    identity = processor.identity()
    assert identity["last_pose_repair_rejection"] == {
        "maximum_forbidden_penetration_after_m": 0.03125,
        "maximum_foot_penetration_after_m": 0.0065,
        "maximum_joint_delta_rad": 0.1875,
    }
    assert "forbidden 0.031250 m" in identity["last_reason"]
    assert "foot 0.006500 m" in identity["last_reason"]
    assert "joint 0.187500 rad" in identity["last_reason"]


def test_switch_is_continuous_and_successor_and_neutral_ticks_decay(model) -> None:
    processor, _repairer, locker = _processor(model)
    dt = 1.0 / 60.0
    target_a = _qpos(model, x=0.0)
    target_b = _qpos(model, x=1.0, joint_offset=0.10)
    target_c = _qpos(model, x=1.1, joint_offset=0.11)
    displayed_a = processor.step(
        target_a,
        row=10,
        range_index=1,
        source_contact=np.asarray((False, True)),
        dt_s=dt,
    )

    switched = processor.step(
        target_b,
        row=20,
        range_index=2,
        source_contact=np.asarray((True, False)),
        dt_s=dt,
    )
    successor = processor.step(
        target_c,
        row=21,
        range_index=2,
        source_contact=np.asarray((True, False)),
        dt_s=dt,
    )
    neutral = processor.step(
        target_c,
        row=21,
        range_index=2,
        source_contact=np.asarray((True, False)),
        dt_s=dt,
    )

    np.testing.assert_allclose(switched, displayed_a, atol=1.0e-6, rtol=0.0)
    assert np.linalg.norm(neutral - target_c) < np.linalg.norm(switched - target_b)
    assert np.linalg.norm(neutral - target_c) < np.linalg.norm(successor - target_c)
    # One landing-stance classification at prime plus stance and explicit
    # ground-proximity classifications on every displayed tick.
    assert len(locker.support_calls) == 9
    assert len(locker.calls) == 5
    expected_contacts = (
        (False, False),
        (False, False),
        (False, False),
        (False, False),
        (False, False),
    )
    for (_pose, contact, _dt_s), expected in zip(
        locker.calls, expected_contacts, strict=True
    ):
        np.testing.assert_array_equal(contact, expected)
    assert locker.reset_calls == 3


def test_active_lock_rejection_uses_existing_release_without_freezing(model) -> None:
    processor, _repairer, locker = _continuous_processor(model)
    locker.measured_support = np.asarray((True, False), dtype=bool)
    dt = 1.0 / 60.0
    processor.step(
        _qpos(model, x=0.0),
        row=1,
        range_index=0,
        source_contact=np.asarray((True, False)),
        dt_s=dt,
    )
    locker.accepted[:] = [False, True]

    released = processor.step(
        _qpos(model, x=0.05),
        row=2,
        range_index=0,
        source_contact=np.asarray((True, False)),
        dt_s=dt,
    )

    assert released[0] == pytest.approx(0.05)
    np.testing.assert_array_equal(locker.calls[-2][1], (True, False))
    np.testing.assert_array_equal(locker.calls[-1][1], (False, False))
    identity = processor.identity()
    assert identity["continuous_lock_recovery_count"] == 1
    assert identity["continuous_lock_releasing_frame_count"] == 1
    assert identity["continuous_lock_bypass_count"] == 0
    assert identity["last_reason"] == "lock rejected"


def test_double_lock_rejection_bypasses_advancing_candidate(model) -> None:
    processor, _repairer, locker = _continuous_processor(model)
    dt = 1.0 / 60.0
    processor.step(
        _qpos(model, x=0.0),
        row=1,
        range_index=0,
        source_contact=np.asarray((True, True)),
        dt_s=dt,
    )
    locker.accepted[:] = [False, False]

    bypassed = processor.step(
        _qpos(model, x=0.05),
        row=2,
        range_index=0,
        source_contact=np.asarray((True, True)),
        dt_s=dt,
    )

    assert bypassed[0] == pytest.approx(0.05)
    identity = processor.identity()
    assert identity["continuous_lock_bypass_count"] == 1
    assert identity["foot_lock_bypass_count"] == 1
    assert "release rejected" in identity["last_reason"]


def test_double_lock_rejection_does_not_publish_one_side_of_knee_flip(model) -> None:
    repairer = _KneeOffsetRepairer()
    locker = _MeasuredContinuousFootLocker()
    locker.measured_stance = np.asarray((False, False), dtype=bool)
    processor = ExistingUtilityPosePostprocessor(
        model,
        SimpleNamespace(height_at_world_xy=lambda _xy: 0.0),
        pose_repairer=repairer,
        foot_locker=locker,
    )
    dt = 1.0 / 60.0
    source_contact = np.asarray((True, False), dtype=bool)
    processor.step(
        _qpos(model, x=0.0),
        row=1,
        range_index=0,
        source_contact=source_contact,
        dt_s=dt,
    )

    # Reproduce the S2 seam: a once-repaired knee pose is followed by two
    # rejected lock attempts, then the next advancing row accepts the
    # opposite repair direction.  Publishing the rejected tick's inner
    # repair creates a 0.50-rad display flip despite smooth source rows.
    repairer.knee_offsets[:] = [-0.25]
    locker.accepted[:] = [False, False]
    resets_before_bypass = locker.reset_calls
    bypass_source = _qpos(model, x=0.05)
    bypassed = processor.step(
        bypass_source,
        row=2,
        range_index=0,
        source_contact=source_contact,
        dt_s=dt,
    )
    bypass_identity = processor.identity()
    resets_after_bypass = locker.reset_calls
    calls_after_bypass = len(locker.calls)

    repairer.knee_offsets[:] = [0.25, 0.0]
    recovery_source = _qpos(model, x=0.10)
    recovered = processor.step(
        recovery_source,
        row=3,
        range_index=0,
        source_contact=source_contact,
        dt_s=dt,
    )

    np.testing.assert_allclose(bypassed, bypass_source, atol=1.0e-6, rtol=0.0)
    assert bypassed[0] == pytest.approx(0.05)
    assert recovered[0] == pytest.approx(0.10)
    assert abs(float(recovered[10] - bypassed[10])) <= 0.25 + 1.0e-6
    assert bypass_identity["continuous_lock_bypass_count"] == 1
    assert bypass_identity["foot_lock_bypass_count"] == 1
    assert bypass_identity["last_reason"] == (
        "lock rejected; release rejected: lock rejected"
    )
    assert resets_after_bypass == resets_before_bypass + 2
    assert locker.reset_calls == resets_after_bypass + 1
    assert len(locker.calls) == calls_after_bypass + 2


def test_trusted_source_recovers_pre_repair_rejection_on_original_pose(model) -> None:
    processor, repairer, locker = _continuous_processor(model)
    locker.measured_support = np.asarray((True, False), dtype=bool)
    dt = 1.0 / 60.0
    source_contact = np.asarray((True, False), dtype=bool)
    for row in (1, 2):
        processor.step(
            _qpos(model, x=0.01 * (row - 1)),
            row=row,
            range_index=0,
            source_contact=source_contact,
            dt_s=dt,
        )
    identity_before = processor.identity()
    calls_before = len(locker.calls)
    resets_before = locker.reset_calls
    restores_before = locker.restore_calls
    repairer.queue(False, reason="pre-repair rejected")
    repairer.queue(True)
    source = _qpos(model, x=0.05)

    recovered = processor.step(
        source,
        row=3,
        range_index=0,
        source_contact=source_contact,
        dt_s=dt,
    )

    np.testing.assert_allclose(recovered, source, atol=1.0e-6, rtol=0.0)
    assert len(locker.calls) == calls_before + 1
    np.testing.assert_array_equal(locker.calls[-1][1], (True, False))
    assert locker.calls[-1][0] is repairer.calls[-2]
    assert locker.reset_calls == resets_before
    assert locker.restore_calls == restores_before
    identity = processor.identity()
    assert identity["trusted_pre_repair_recovery_attempt_count"] == 1
    assert identity["trusted_pre_repair_recovery_success_count"] == 1
    assert identity["trusted_pre_repair_recovery_failure_count"] == 0
    assert identity["last_published_locked_contact"] == (True, False)
    assert identity["continuous_lock_recovery_count"] == (
        identity_before["continuous_lock_recovery_count"] + 1
    )
    assert identity["continuous_lock_active_frame_count"] == (
        identity_before["continuous_lock_active_frame_count"] + 1
    )
    assert identity["continuous_lock_bypass_count"] == (
        identity_before["continuous_lock_bypass_count"]
    )
    assert identity["last_reason"] == (
        "recovered trusted contact after pre-repair rejection: "
        "pre-repair rejected"
    )


def test_pre_repair_recovery_post_reject_restores_and_reprimes(model) -> None:
    processor, repairer, locker = _continuous_processor(model)
    locker.measured_support = np.asarray((True, False), dtype=bool)
    dt = 1.0 / 60.0
    source_contact = np.asarray((True, False), dtype=bool)
    for row in (1, 2):
        processor.step(
            _qpos(model, x=0.01 * (row - 1)),
            row=row,
            range_index=0,
            source_contact=source_contact,
            dt_s=dt,
        )
    calls_before = len(locker.calls)
    resets_before = locker.reset_calls
    restores_before = locker.restore_calls
    repairer.queue(False, reason="pre-repair rejected")
    repairer.queue(False, reason="recovery post-repair rejected")
    rejected_source = _qpos(model, x=0.05)

    bypassed = processor.step(
        rejected_source,
        row=3,
        range_index=0,
        source_contact=source_contact,
        dt_s=dt,
    )
    resets_after_bypass = locker.reset_calls
    recovered_source = _qpos(model, x=0.06)
    advanced = processor.step(
        recovered_source,
        row=4,
        range_index=0,
        source_contact=source_contact,
        dt_s=dt,
    )

    np.testing.assert_allclose(bypassed, rejected_source, atol=1.0e-6, rtol=0.0)
    np.testing.assert_allclose(advanced, recovered_source, atol=1.0e-6, rtol=0.0)
    assert locker.restore_calls == restores_before + 1
    assert resets_after_bypass == resets_before + 1
    assert locker.reset_calls == resets_after_bypass + 1
    assert len(locker.calls) == calls_before + 3
    identity = processor.identity()
    assert identity["trusted_pre_repair_recovery_attempt_count"] == 1
    assert identity["trusted_pre_repair_recovery_success_count"] == 0
    assert identity["trusted_pre_repair_recovery_failure_count"] == 1
    assert identity["trusted_pre_repair_recovery_post_reject_count"] == 1
    assert identity["continuous_lock_bypass_count"] == 1
    assert identity["last_published_locked_contact"] == (False, False)
    assert identity["last_reason"] == (
        "trusted recovery post-lock repair rejected after pre-repair rejection: "
        "pre-repair rejected; recovery post-repair rejected"
    )


def test_pre_repair_recovery_lock_reject_restores_transaction(model) -> None:
    processor, repairer, locker = _continuous_processor(model)
    locker.measured_support = np.asarray((True, False), dtype=bool)
    dt = 1.0 / 60.0
    source_contact = np.asarray((True, False), dtype=bool)
    for row in (1, 2):
        processor.step(
            _qpos(model, x=0.01 * (row - 1)),
            row=row,
            range_index=0,
            source_contact=source_contact,
            dt_s=dt,
        )
    calls_before = len(locker.calls)
    restores_before = locker.restore_calls
    repairer.queue(False, reason="pre-repair rejected")
    locker.accepted[:] = [False]
    source = _qpos(model, x=0.05)

    bypassed = processor.step(
        source,
        row=3,
        range_index=0,
        source_contact=source_contact,
        dt_s=dt,
    )

    np.testing.assert_allclose(bypassed, source, atol=1.0e-6, rtol=0.0)
    assert len(locker.calls) == calls_before + 1
    np.testing.assert_array_equal(locker.calls[-1][1], (True, False))
    assert locker.restore_calls == restores_before + 1
    identity = processor.identity()
    assert identity["trusted_pre_repair_recovery_attempt_count"] == 1
    assert identity["trusted_pre_repair_recovery_success_count"] == 0
    assert identity["trusted_pre_repair_recovery_failure_count"] == 1
    assert identity["trusted_pre_repair_recovery_lock_reject_count"] == 1
    assert identity["continuous_lock_bypass_count"] == 1
    assert identity["last_published_locked_contact"] == (False, False)
    assert identity["last_reason"] == (
        "trusted recovery lock rejected after pre-repair rejection: "
        "pre-repair rejected; lock rejected"
    )


def test_pre_repair_rejection_with_source_false_does_not_recover(model) -> None:
    processor, repairer, locker = _continuous_processor(model)
    locker.measured_support = np.asarray((True, False), dtype=bool)
    dt = 1.0 / 60.0
    for row in (1, 2):
        processor.step(
            _qpos(model, x=0.01 * (row - 1)),
            row=row,
            range_index=0,
            source_contact=np.asarray((True, False), dtype=bool),
            dt_s=dt,
        )
    calls_before = len(locker.calls)
    repairer.queue(False, reason="pre-repair rejected")
    source = _qpos(model, x=0.05)

    bypassed = processor.step(
        source,
        row=3,
        range_index=0,
        source_contact=np.asarray((False, False), dtype=bool),
        dt_s=dt,
    )

    np.testing.assert_allclose(bypassed, source, atol=1.0e-6, rtol=0.0)
    assert len(locker.calls) == calls_before
    identity = processor.identity()
    assert identity["trusted_pre_repair_recovery_attempt_count"] == 0
    assert identity["last_trusted_landing_contact"] == (False, False)
    assert identity["last_published_locked_contact"] == (False, False)


def test_pre_repair_rejection_cannot_recover_without_prior_trust(model) -> None:
    processor, repairer, locker = _continuous_processor(model)
    locker.measured_stance = np.asarray((False, False), dtype=bool)
    locker.measured_support = np.asarray((True, False), dtype=bool)
    repairer.queue(False, reason="pre-repair rejected")
    source = _qpos(model, x=0.05)

    bypassed = processor.step(
        source,
        row=1,
        range_index=0,
        source_contact=np.asarray((True, False), dtype=bool),
        dt_s=1.0 / 60.0,
    )

    np.testing.assert_allclose(bypassed, source, atol=1.0e-6, rtol=0.0)
    # The deferred zero-contact prime is the only foot-lock call.  Source true
    # may not create trust inside the pre-repair recovery path.
    assert len(locker.calls) == 1
    np.testing.assert_array_equal(locker.calls[0][1], (False, False))
    identity = processor.identity()
    assert identity["last_trusted_landing_contact"] == (False, False)
    assert identity["last_published_locked_contact"] == (False, False)
    assert identity["trusted_pre_repair_recovery_attempt_count"] == 0


def test_post_lock_repair_rejection_resets_stale_lock_then_recovers(model) -> None:
    processor, repairer, locker = _continuous_processor(model)
    dt = 1.0 / 60.0
    processor.step(
        _qpos(model, x=0.0),
        row=1,
        range_index=0,
        source_contact=np.asarray((False, False)),
        dt_s=dt,
    )
    repairer.queue(True)
    repairer.queue(False, reason="post-lock repair rejected")

    bypassed = processor.step(
        _qpos(model, x=0.10),
        row=2,
        range_index=0,
        source_contact=np.asarray((False, False)),
        dt_s=dt,
    )
    resets_after_rejection = locker.reset_calls
    recovered = processor.step(
        _qpos(model, x=0.20),
        row=3,
        range_index=0,
        source_contact=np.asarray((False, False)),
        dt_s=dt,
    )

    assert bypassed[0] == pytest.approx(0.10)
    assert recovered[0] == pytest.approx(0.20)
    assert locker.reset_calls == resets_after_rejection + 1
    identity = processor.identity()
    assert identity["continuous_lock_bypass_count"] == 1
    assert identity["pose_repair_rejection_count"] == 1
    assert identity["last_reason"] == "post-lock repair rejected"


def test_pre_lock_repair_rejection_advances_and_reprimes_next_tick(model) -> None:
    processor, repairer, locker = _continuous_processor(model)
    repairer.queue(False, reason="pre-lock repair rejected")
    dt = 1.0 / 60.0

    bypassed = processor.step(
        _qpos(model, x=0.15),
        row=1,
        range_index=0,
        source_contact=np.asarray((True, True)),
        dt_s=dt,
    )
    resets_after_rejection = locker.reset_calls
    recovered = processor.step(
        _qpos(model, x=0.16),
        row=2,
        range_index=0,
        source_contact=np.asarray((True, True)),
        dt_s=dt,
    )

    assert bypassed[0] == pytest.approx(0.15)
    assert recovered[0] == pytest.approx(0.16)
    assert locker.reset_calls == resets_after_rejection + 1
    identity = processor.identity()
    assert identity["continuous_lock_bypass_count"] == 1
    assert identity["pose_repair_rejection_count"] == 1
    assert identity["last_reason"] == "pre-lock repair rejected"


def test_real_existing_utilities_process_native_pose_on_nonflat_scene() -> None:
    terrain = load_scene_terrain("hills")
    collision_model = build_diagnostic_collision_model(DEFAULT_G1_XML, terrain)
    artifacts = ArtifactSet.empty(3, 31)
    artifacts.range_starts = np.asarray((0,), dtype=np.int32)
    artifacts.range_stops = np.asarray((3,), dtype=np.int32)
    artifacts.rotations[..., 0] = 1.0
    artifacts.positions[:, 1, 1] = float(collision_model.qpos0[2])
    feature_rows = np.zeros((3, 31), dtype=np.float32)
    feature_rows[0, 27:31] = -10.0
    feature_rows[1:, 27:31] = 10.0
    corpus = SimpleNamespace(
        artifacts=artifacts,
        features=FeatureSet(
            feature_rows,
            np.zeros(31, dtype=np.float32),
            np.ones(31, dtype=np.float32),
        ),
        family_ids=np.asarray((0,), dtype=np.int32),
        family_names=("hill",),
        fps=60.0,
        horizons=(20, 40, 60),
    )
    generator = SimpleNamespace(latent=np.zeros((3, 1), dtype=np.float32))
    native_source = np.array(collision_model.qpos0, dtype=np.float64, copy=True)
    native_source[:2] = (-3.0, 2.0)

    def place_native_source(
        decoded: object,
        simulation_position: object,
        _simulation_rotation: object,
        _native_model: object,
    ) -> np.ndarray:
        value = native_source.copy()
        value[2] = float(np.asarray(simulation_position)[1]) + float(
            np.asarray(decoded)[1]
        )
        return value

    matcher = HybridMatcher(
        corpus,
        generator,
        terrain.authority,
        native_model=collision_model,
        pose_converter=place_native_source,
        diagnostic_stability=True,
        diagnostic_canonical_source_pose=True,
        initial_root_xy=(-3.0, 2.0),
    )
    processor = ExistingUtilityPosePostprocessor(
        collision_model,
        SimpleNamespace(height_at_world_xy=terrain.authority.height_at),
    )

    displayed = processor.step(
        matcher.state.qpos,
        row=matcher.state.row,
        range_index=matcher.state.range_index,
        source_contact=np.asarray(
            matcher.artifacts.contacts[matcher.state.row], dtype=bool
        ),
        dt_s=matcher.dt,
    )

    assert matcher.state.pose_source == "canonical-diagnostic"
    assert matcher.state.support_status == "SUPPORTED"
    assert displayed.shape == (36,)
    assert np.isfinite(displayed).all()
    assert processor.pose_repair_count == 1
    assert processor.foot_lock_accept_count == 0
    assert processor.pose_repair_rejection_count == 0


def test_reset_clears_temporal_state_and_diagnostics(model) -> None:
    processor, _repairer, locker = _processor(model)
    processor.step(
        _qpos(model, x=0.0),
        row=1,
        range_index=0,
        source_contact=np.asarray((False, False)),
        dt_s=1.0 / 60.0,
    )
    processor.reset()

    identity = processor.identity()
    assert identity == {
        "diagnostic_display_postprocessor": (
            "existing-pose-inertializer-repair-injected-foot-lock/unverified"
        ),
        "inertialization_halflife_s": 0.10,
        "contact_policy": (
            "authenticated-source-and-measured-ground-proximity-two-frame-"
            "acquire-source-continue-release"
        ),
        "source_contacts_required_for_acquisition": True,
        "measured_ground_proximity_required_for_acquisition": True,
        "measured_speed_used_for_acquisition": False,
            "source_contacts_used_for_trusted_continuation": True,
            "diagnostic_owned_foot_locker": False,
            "foot_lock_maximum_locked_foot_drift_m": None,
            "foot_lock_maximum_releasing_foot_drift_m": None,
            "foot_lock_maximum_joint_correction_rad": None,
            "foot_lock_maximum_joint_correction_step_rad": None,
            "foot_lock_defer_swing_clearance_until_release_complete": False,
            "trusted_source_lock_final_authority": False,
            "trusted_post_lock_safety_validator_requires_noop": False,
        "measured_stance_maximum_foot_speed_mps": 0.20,
        "ground_proximity_maximum_sole_clearance_m": 0.020,
        "source_proximity_acquire_frames": 2,
        "minimum_swing_clearance_m": 0.012,
        "foot_lock_release_halflife_s": 0.08,
        "continuous_lock_dt_s": 1.0 / 60.0,
        "pose_repair_count": 0,
        "foot_lock_accept_count": 0,
        "foot_lock_bypass_count": 0,
        "pose_repair_rejection_count": 0,
        "continuous_lock_active_frame_count": 0,
        "continuous_lock_releasing_frame_count": 0,
        "continuous_lock_idle_frame_count": 0,
        "continuous_lock_recovery_count": 0,
        "continuous_lock_bypass_count": 0,
        "source_proximity_candidate_frame_count": 0,
        "source_proximity_acquisition_frame_count": 0,
        "source_without_proximity_frame_count": 0,
        "proximity_without_source_frame_count": 0,
        "trusted_source_continuation_frame_count": 0,
        "trusted_source_override_frame_count": 0,
        "trusted_source_release_frame_count": 0,
        "trusted_pre_repair_recovery_attempt_count": 0,
        "trusted_pre_repair_recovery_success_count": 0,
        "trusted_pre_repair_recovery_failure_count": 0,
        "trusted_pre_repair_recovery_lock_reject_count": 0,
        "trusted_pre_repair_recovery_post_reject_count": 0,
        "last_measured_stance_contact": (False, False),
        "last_ground_proximity_contact": (False, False),
        "last_trusted_landing_contact": (False, False),
            "last_published_locked_contact": (False, False),
            "last_published_releasing_contact": (False, False),
        "last_pose_repair_rejection": None,
        "last_reason": "reset",
    }
    assert locker.reset_calls == 4
