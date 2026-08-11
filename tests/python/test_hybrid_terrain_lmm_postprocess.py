from __future__ import annotations

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

    def support_contact(self, pose: KinematicPose) -> np.ndarray:
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
            releasing=(False, False),
            reason="accepted" if accepted else "lock rejected",
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


def test_continuous_lock_uses_measured_stance_on_every_tick_and_ignores_source(
    model,
) -> None:
    processor, _repairer, locker = _continuous_processor(model)
    dt = 1.0 / 60.0
    rows = (10, 11, 12, 13, 14, 15, 30, 31)

    for index, row in enumerate(rows):
        processor.step(
            _qpos(model, x=0.01 * index),
            row=row,
            range_index=0 if row < 30 else 1,
            source_contact=np.asarray((False, bool(index % 2)), dtype=bool),
            dt_s=dt,
        )

    # One priming classification/apply plus one continuous filtered update
    # per displayed tick.  The discontinuous row switch must not reset the
    # planted world-space target.
    assert len(locker.landing_stance_calls) == len(rows) + 1
    assert len(locker.calls) == len(rows) + 1
    assert locker.reset_calls == 3
    for _pose, contact, call_dt in locker.calls:
        np.testing.assert_array_equal(contact, (True, False))
        assert call_dt == pytest.approx(dt)
    assert locker.swing_clearances == pytest.approx([0.012] * (len(rows) + 1))
    identity = processor.identity()
    assert identity["source_contacts_used_for_locking"] is False
    assert identity["last_measured_stance_contact"] == (True, False)
    assert identity["continuous_lock_active_frame_count"] == len(rows)


def test_continuous_measured_stance_requires_two_frames_to_acquire(model) -> None:
    processor, _repairer, locker = _continuous_processor(model)
    locker.measured_stance_queue[:] = [
        np.asarray((False, False), dtype=bool),
        np.asarray((True, False), dtype=bool),
        np.asarray((True, False), dtype=bool),
    ]
    dt = 1.0 / 60.0

    processor.step(
        _qpos(model, x=0.0),
        row=1,
        range_index=0,
        source_contact=np.asarray((False, True), dtype=bool),
        dt_s=dt,
    )
    processor.step(
        _qpos(model, x=0.01),
        row=2,
        range_index=0,
        source_contact=np.asarray((False, True), dtype=bool),
        dt_s=dt,
    )

    np.testing.assert_array_equal(locker.calls[0][1], (False, False))
    np.testing.assert_array_equal(locker.calls[1][1], (False, False))
    np.testing.assert_array_equal(locker.calls[2][1], (True, False))
    assert processor.identity()["last_measured_stance_contact"] == (True, False)


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
    assert len(locker.support_calls) == 5
    assert len(locker.calls) == 5
    for _pose, contact, _dt_s in locker.calls:
        np.testing.assert_array_equal(contact, (False, True))
    assert locker.reset_calls == 3


def test_active_lock_rejection_uses_existing_release_without_freezing(model) -> None:
    processor, _repairer, locker = _continuous_processor(model)
    dt = 1.0 / 60.0
    processor.step(
        _qpos(model, x=0.0),
        row=1,
        range_index=0,
        source_contact=np.asarray((False, True)),
        dt_s=dt,
    )
    locker.accepted[:] = [False, True]

    released = processor.step(
        _qpos(model, x=0.05),
        row=2,
        range_index=0,
        source_contact=np.asarray((False, True)),
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
            "existing-pose-inertializer-repair-measured-continuous-foot-lock/v4"
        ),
        "inertialization_halflife_s": 0.10,
        "contact_policy": "measured-support-speed-hysteresis",
        "source_contacts_used_for_locking": False,
        "measured_stance_maximum_foot_speed_mps": 0.20,
        "measured_stance_maximum_sole_clearance_m": 0.020,
        "measured_stance_acquire_frames": 2,
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
        "last_measured_stance_contact": (False, False),
        "last_pose_repair_rejection": None,
        "last_reason": "reset",
    }
    assert locker.reset_calls == 4
