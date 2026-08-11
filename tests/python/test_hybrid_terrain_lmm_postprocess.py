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

    def apply(
        self, pose: KinematicPose, source_contact: object, *, dt_s: float
    ) -> object:
        self.calls.append(
            (pose, np.array(source_contact, dtype=bool, copy=True), float(dt_s))
        )
        accepted = self.accepted.pop(0) if self.accepted else True
        return SimpleNamespace(
            pose=pose,
            accepted=accepted,
            repaired=False,
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
    np.testing.assert_array_equal(locker.calls[0][1], (False, True))
    np.testing.assert_array_equal(locker.calls[1][1], (True, False))


def test_lock_rejection_restores_snapshot_and_publishes_repaired_source(model) -> None:
    processor, _repairer, locker = _processor(model)
    locker.accepted[:] = [True, False]
    previous = processor.step(
        _qpos(model, x=0.0),
        row=1,
        range_index=0,
        source_contact=np.asarray((True, False)),
        dt_s=1.0 / 60.0,
    )
    source = _qpos(model, x=0.05, joint_offset=0.01)
    rejected = processor.step(
        source,
        row=2,
        range_index=0,
        source_contact=np.asarray((False, True)),
        dt_s=1.0 / 60.0,
    )

    assert locker.restore_calls == 1
    assert not np.array_equal(rejected, previous)
    np.testing.assert_allclose(rejected, source, atol=1.0e-6, rtol=0.0)
    assert processor.foot_lock_bypass_count == 1
    assert processor.identity()["last_reason"] == "lock rejected"


def test_repair_fallback_bypasses_raw_failure_resets_lock_then_recovers(
    model,
) -> None:
    processor, repairer, locker = _processor(model)
    dt = 1.0 / 60.0
    processor.step(
        _qpos(model, x=0.0),
        row=1,
        range_index=0,
        source_contact=np.asarray((True, False)),
        dt_s=dt,
    )
    target = _qpos(model, x=0.6, joint_offset=0.05)
    repairer.queue(False, reason="blend rejected")
    repairer.queue(True, repaired=True)
    fallback = processor.step(
        target,
        row=20,
        range_index=3,
        source_contact=np.asarray((False, True)),
        dt_s=dt,
    )
    np.testing.assert_allclose(fallback, target, atol=1.0e-6, rtol=0.0)
    assert processor.pose_repair_count == 2
    assert processor.identity()["last_reason"] == "blend rejected"

    failed_target = _qpos(model, x=0.8, joint_offset=0.08)
    repairer.queue(False, reason="blend bad")
    repairer.queue(False, reason="raw bad")
    bypassed = processor.step(
        failed_target,
        row=40,
        range_index=4,
        source_contact=np.asarray((True, False)),
        dt_s=dt,
    )
    assert np.all(np.isfinite(bypassed))
    assert processor.raw_repair_failure_count == 1
    assert processor.foot_lock_bypass_count == 1
    assert processor.identity()["last_reason"] == "raw bad"
    assert len(locker.calls) == 2
    assert locker.reset_calls == 2

    recovered_target = _qpos(model, x=0.85, joint_offset=0.09)
    recovered = processor.step(
        recovered_target,
        row=41,
        range_index=4,
        source_contact=np.asarray((True, False)),
        dt_s=dt,
    )
    assert not np.array_equal(recovered, bypassed)
    assert locker.calls[-1][1].tolist() == [True, False]


def test_successor_raw_repair_failure_publishes_source_without_duplicate_retry(
    model,
) -> None:
    processor, repairer, _locker = _processor(model)
    displayed = processor.step(
        _qpos(model, x=0.0),
        row=1,
        range_index=0,
        source_contact=np.asarray((True, False)),
        dt_s=1.0 / 60.0,
    )
    repairer.queue(False, reason="raw successor rejected")
    source = _qpos(model, x=0.1)
    bypassed = processor.step(
        source,
        row=2,
        range_index=0,
        source_contact=np.asarray((True, False)),
        dt_s=1.0 / 60.0,
    )

    assert not np.array_equal(bypassed, displayed)
    np.testing.assert_allclose(bypassed, source, atol=1.0e-6, rtol=0.0)
    assert len(repairer.calls) == 2
    assert processor.raw_repair_failure_count == 1


def test_initial_raw_repair_failure_bypasses_without_stall_then_recovers(model) -> None:
    processor, repairer, locker = _processor(model)
    initial = _qpos(model, x=0.15, joint_offset=0.01)
    repairer.queue(False, reason="initial raw rejected")

    bypassed = processor.step(
        initial,
        row=7,
        range_index=2,
        source_contact=np.asarray((False, True)),
        dt_s=1.0 / 60.0,
    )

    np.testing.assert_allclose(bypassed, initial, atol=1.0e-7, rtol=0.0)
    assert locker.calls == []
    assert processor.raw_repair_failure_count == 1
    assert processor.identity()["last_reason"] == "initial raw rejected"

    recovered = processor.step(
        _qpos(model, x=0.16, joint_offset=0.012),
        row=8,
        range_index=2,
        source_contact=np.asarray((False, True)),
        dt_s=1.0 / 60.0,
    )

    assert not np.array_equal(recovered, bypassed)
    assert processor.pose_repair_count == 1
    assert processor.foot_lock_accept_count == 1


def test_pose_repair_count_counts_every_accepted_repair_call(model) -> None:
    processor, repairer, _locker = _processor(model)

    processor.step(
        _qpos(model, x=0.0),
        row=1,
        range_index=0,
        source_contact=np.asarray((False, False)),
        dt_s=1.0 / 60.0,
    )
    repairer.queue(False, reason="blend rejected")
    repairer.queue(True, repaired=False)
    processor.step(
        _qpos(model, x=0.2),
        row=10,
        range_index=1,
        source_contact=np.asarray((False, False)),
        dt_s=1.0 / 60.0,
    )

    assert processor.pose_repair_count == 2


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
    assert processor.foot_lock_accept_count == 1
    assert processor.raw_repair_failure_count == 0


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
            "existing-pose-inertializer-repair-foot-lock/v1"
        ),
        "inertialization_halflife_s": 0.10,
        "pose_repair_count": 0,
        "foot_lock_accept_count": 0,
        "foot_lock_bypass_count": 0,
        "raw_repair_failure_count": 0,
        "last_reason": "reset",
    }
    assert locker.reset_calls == 2
