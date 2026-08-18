from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import mm_sonic.justin_recovery as justin_recovery
from mm_sonic.justin_recovery import (
    HISTORY_FRAMES,
    TRACKER_HISTORY_FRAMES,
    RecoveryContractError,
    build_recovery_proposal,
    learner_state_in_reference_frame,
    learner_tracker_history_in_reference_frame,
    learner_xy_to_reference,
    load_recovery_trace,
    load_reference_bank,
    rank_recovery_candidates,
    reference_task12,
    select_rewind_queries,
)


def _write_trace(path: Path, frames: int = 140) -> Path:
    t = np.arange(frames, dtype=np.float32)
    qpos = np.zeros((frames, 36), np.float64)
    qpos[:, 0] = 2.0 + 0.01 * t
    qpos[:, 2] = 0.793
    qpos[:, 3] = 1.0
    qvel = np.zeros((frames, 35), np.float64)
    joint_pos = np.zeros((frames, 29), np.float32)
    joint_pos[:, 0] = np.sin(t * 0.1)
    joint_vel = np.zeros_like(joint_pos)
    joint_vel[1:, 0] = np.diff(joint_pos[:, 0]) * 50.0
    body_pos = np.zeros((frames, 30, 3), np.float32)
    body_pos[:, 0] = qpos[:, :3]
    body_pos[:, 6] = qpos[:, :3] + np.asarray((0.0, 0.12, -0.75))
    body_pos[:, 12] = qpos[:, :3] + np.asarray((0.0, -0.12, -0.75))
    body_quat = np.zeros((frames, 30, 4), np.float32)
    body_quat[..., 0] = 1.0
    body_lin = np.zeros((frames, 30, 3), np.float32)
    body_lin[:, 0, 0] = 0.5
    body_lin[:, 12, 0] = 0.5
    body_ang = np.zeros_like(body_lin)
    task12 = np.tile(np.asarray((0.5, 0.0, 0.0), np.float32), (frames, 4))
    np.savez(
        path,
        physics_step=np.arange(frames, dtype=np.int64) * 4,
        policy_call_index=np.arange(frames, dtype=np.int64),
        qpos_mujoco=qpos,
        qvel_mujoco=qvel,
        joint_pos_isaac=joint_pos,
        joint_vel_isaac=joint_vel,
        action_isaac_normalized=np.tile(
            np.linspace(-1.0, 1.0, 29, dtype=np.float32), (frames, 1)
        ),
        body_pos_mujoco=body_pos,
        body_quat_wxyz_mujoco=body_quat,
        body_lin_vel_world_mujoco=body_lin,
        body_ang_vel_world_mujoco=body_ang,
        left_foot_contact=np.ones(frames, np.bool_),
        right_foot_contact=np.zeros(frames, np.bool_),
        forbidden_nonfoot_contact=np.zeros(frames, np.bool_),
        minimum_contact_distance_m=np.full(frames, -0.001, np.float32),
        task12=task12,
    )
    return path


def _write_reference(path: Path, frames: int = 160, offset: float = 0.0) -> Path:
    t = np.arange(frames, dtype=np.float32)
    root = np.zeros((frames, 3), np.float32)
    root[:, 0] = 0.78935 - 0.01 * t
    root[:, 1] = -4.3529
    root[:, 2] = 0.793
    joint_pos = np.zeros((frames, 29), np.float32)
    joint_pos[:, 0] = np.sin(t * 0.1) + offset
    joint_vel = np.zeros_like(joint_pos)
    joint_vel[1:, 0] = np.diff(joint_pos[:, 0]) * 50.0
    body_pos = np.repeat(root[:, None, :], 30, axis=1)
    body_pos[:, 18] += np.asarray((0.0, 0.12, -0.75), np.float32)
    body_pos[:, 19] += np.asarray((0.0, -0.12, -0.75), np.float32)
    body_quat = np.zeros((frames, 30, 4), np.float32)
    # The reference travels along world -X while facing world -X, so its
    # canonical local command is +0.5 m/s forward like the learner trace.
    body_quat[..., 3] = 1.0
    body_lin = np.zeros((frames, 30, 3), np.float32)
    body_lin[:, 0, 0] = -0.5
    body_lin[:, 18, 0] = 0.0  # inferred left support
    body_lin[:, 19, 0] = 0.5
    body_ang = np.zeros_like(body_lin)
    np.savez(
        path,
        fps=np.asarray([50], np.int64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
    )
    return path


def test_rewind_query_uses_physics_step_and_full_history(tmp_path: Path) -> None:
    trace = load_recovery_trace(_write_trace(tmp_path / "trace.npz"))
    query = select_rewind_queries(
        trace, first_fall_physics_step=500, rewinds_s=(0.5,)
    )[0]
    assert query.physics_step == 400
    assert query.trace_index == 100
    assert query.history_start == 100 - HISTORY_FRAMES + 1


def test_rewind_query_skips_already_collapsed_state(tmp_path: Path) -> None:
    path = _write_trace(tmp_path / "trace.npz")
    with np.load(path) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["qpos_mujoco"][100, 2] = 0.4
    np.savez(path, **arrays)
    trace = load_recovery_trace(path)
    query = select_rewind_queries(
        trace, first_fall_physics_step=500, rewinds_s=(0.5, 0.75)
    )[0]
    assert query.rewind_s == 0.75


def test_rewind_anchors_to_sustained_collapse_onset(tmp_path: Path) -> None:
    path = _write_trace(tmp_path / "trace.npz")
    with np.load(path) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["qpos_mujoco"][105:, 2] = 0.4
    np.savez(path, **arrays)
    trace = load_recovery_trace(path)
    query = select_rewind_queries(
        trace, first_fall_physics_step=500, rewinds_s=(0.25,)
    )[0]
    # Sustained collapse begins at physics step 420; 0.25 s earlier targets
    # step 370 and selects the preceding 50 Hz row at step 368.
    assert query.physics_step == 368


def test_scene_transform_preserves_learner_offset(tmp_path: Path) -> None:
    mapped = learner_xy_to_reference(np.asarray((3.2, 0.1)))
    np.testing.assert_allclose(mapped, (-0.41065, -4.4529), atol=1e-6)

    trace = load_recovery_trace(_write_trace(tmp_path / "trace.npz"))
    query = select_rewind_queries(
        trace, first_fall_physics_step=500, rewinds_s=(0.5,)
    )[0]
    state = learner_state_in_reference_frame(trace, query)
    assert state["root_pose_wxyz"].shape == (7,)
    assert state["root_velocity_world"].shape == (6,)
    assert state["joint_pos_isaac"].shape == (29,)
    # pi world-yaw transform turns +X learner velocity into -X reference velocity.
    assert state["root_velocity_world"][0] == pytest.approx(-0.5)


def test_tracker_history_is_exact_chronological_learner_history(
    tmp_path: Path,
) -> None:
    trace = load_recovery_trace(_write_trace(tmp_path / "trace.npz"))
    query = select_rewind_queries(
        trace, first_fall_physics_step=500, rewinds_s=(0.5,)
    )[0]
    history = learner_tracker_history_in_reference_frame(trace, query)
    assert len(history) == TRACKER_HISTORY_FRAMES
    assert [row["trace_index"] for row in history] == list(
        range(query.trace_index - TRACKER_HISTORY_FRAMES + 1, query.trace_index + 1)
    )
    assert history[-1]["state"] == {
        name: value.tolist()
        for name, value in learner_state_in_reference_frame(trace, query).items()
    }
    np.testing.assert_allclose(
        history[-1]["last_action_isaac_normalized"],
        trace.arrays["action_isaac_normalized"][query.trace_index],
    )


def test_bank_size_is_deliberately_small(tmp_path: Path) -> None:
    paths = [_write_reference(tmp_path / f"clip_{i}.npz") for i in range(3)]
    with pytest.raises(RecoveryContractError, match="4..8"):
        load_reference_bank(paths)


def test_ranker_prefers_matching_pose_history_and_support(tmp_path: Path) -> None:
    trace = load_recovery_trace(_write_trace(tmp_path / "trace.npz"))
    query = select_rewind_queries(
        trace, first_fall_physics_step=500, rewinds_s=(0.5,)
    )[0]
    paths = [
        _write_reference(tmp_path / "matching.npz", offset=0.0),
        _write_reference(tmp_path / "offset.npz", offset=1.5),
        _write_reference(tmp_path / "offset2.npz", offset=2.0),
        _write_reference(tmp_path / "offset3.npz", offset=2.5),
    ]
    bank = load_reference_bank(paths)
    ranked = rank_recovery_candidates(trace, query, bank, top_k=8)
    assert ranked[0].clip_name == "matching"
    assert ranked[0].support == "left"
    assert ranked[0].frames_remaining >= 50
    assert ranked[0].command_knot_rmse == pytest.approx(0.0)
    np.testing.assert_allclose(
        np.asarray(ranked[0].reference_task12).reshape(4, 3)[:, 0], 0.5
    )


def test_reference_task12_uses_canonical_future_offsets(tmp_path: Path) -> None:
    clip = load_reference_bank(
        [_write_reference(tmp_path / f"clip_{index}.npz") for index in range(4)]
    )[0]
    task = reference_task12(clip, 30)
    assert task.shape == (4, 3)
    np.testing.assert_allclose(task, np.tile((0.5, 0.0, 0.0), (4, 1)), atol=1e-6)


def test_ranker_hard_rejects_command_mismatch(tmp_path: Path) -> None:
    trace_path = _write_trace(tmp_path / "trace.npz")
    with np.load(trace_path) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["task12"][:] = np.tile((0.5, 0.5, 0.8), 4)
    np.savez(trace_path, **arrays)
    trace = load_recovery_trace(trace_path)
    query = select_rewind_queries(
        trace, first_fall_physics_step=500, rewinds_s=(0.5,)
    )[0]
    bank = load_reference_bank(
        [_write_reference(tmp_path / f"clip_{index}.npz") for index in range(4)]
    )
    with pytest.raises(RecoveryContractError, match="no Justin reference candidate"):
        rank_recovery_candidates(trace, query, bank)


def test_proposal_preserves_a_rejected_rewind(tmp_path: Path, monkeypatch) -> None:
    trace = load_recovery_trace(_write_trace(tmp_path / "trace.npz"))
    paths = [_write_reference(tmp_path / f"clip_{i}.npz") for i in range(4)]
    bank = load_reference_bank(paths)
    real_ranker = justin_recovery.rank_recovery_candidates

    def reject_second_rewind(trace, query, clips, *, top_k):
        if query.rewind_s == 0.5:
            raise RecoveryContractError("deliberate hard-filter rejection")
        return real_ranker(trace, query, clips, top_k=top_k)

    monkeypatch.setattr(justin_recovery, "rank_recovery_candidates", reject_second_rewind)
    proposal = build_recovery_proposal(
        trace,
        bank,
        first_fall_physics_step=500,
        rewinds_s=(0.25, 0.5),
    )
    assert proposal["queries"][0]["candidates"]
    assert proposal["schema"] == "justin-s13-tracker-recovery-proposal/v3"
    assert len(proposal["queries"][0]["learner_tracker_history"]) == 10
    assert proposal["queries"][1]["candidates"] == []
    assert "deliberate hard-filter rejection" in proposal["queries"][1]["matching_error"]


def test_trace_rejects_cadence_drift(tmp_path: Path) -> None:
    path = _write_trace(tmp_path / "trace.npz")
    with np.load(path) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["physics_step"][10] += 1
    np.savez(path, **arrays)
    with pytest.raises(RecoveryContractError, match="exact 50 Hz"):
        load_recovery_trace(path)
