from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from mm_sonic.justin_tracker_recovery import (
    AdaptiveReferenceClock,
    SecondStageRematchTrigger,
    TrackerRecoveryError,
    _compatible_urdf_extension_name,
    _disable_builtin_failure_terminations,
    _kit_cache_args,
    _slerp_wxyz,
    evaluate_second_stage_recovery,
    evaluate_tracker_attempt,
    select_proposal_candidate,
    select_second_stage_reference_frame,
    validate_exact_learner_history,
    validate_reference_blend,
)


def test_kit_cache_args_isolate_only_ujitso(tmp_path: Path) -> None:
    args = _kit_cache_args(tmp_path / "kit-cache")
    assert "/UJITSO/datastore/localCachePath=" in args
    assert str(tmp_path / "kit-cache" / "DerivedDataCache") in args
    assert "/app/userConfigPath=" not in args


def test_urdf_extension_uses_installed_patch_version() -> None:
    assert (
        _compatible_urdf_extension_name("isaacsim.asset.importer.urdf-2.4.31")
        == "isaacsim.asset.importer.urdf-2.4.30"
    )
    assert _compatible_urdf_extension_name("unrelated-1.0") == "unrelated-1.0"


def test_recovery_screen_disables_only_training_failure_terminations() -> None:
    class Manager:
        active_terms = ("time_out", "anchor_pos", "foot_pos_xyz")

        def __init__(self):
            self.cfgs = {
                "time_out": SimpleNamespace(time_out=True, func=object()),
                "anchor_pos": SimpleNamespace(time_out=False, func=object()),
                "foot_pos_xyz": SimpleNamespace(time_out=False, func=object()),
            }

        def get_term_cfg(self, name):
            return self.cfgs[name]

        def set_term_cfg(self, name, cfg):
            self.cfgs[name] = cfg

    manager = Manager()
    original_timeout = manager.cfgs["time_out"].func
    env = SimpleNamespace(env=SimpleNamespace(termination_manager=manager))
    disabled = _disable_builtin_failure_terminations(env)
    assert disabled == ("anchor_pos", "foot_pos_xyz")
    assert manager.cfgs["time_out"].func is original_timeout


def _proposal() -> dict:
    return {
        "schema": "justin-s13-tracker-recovery-proposal/v1",
        "queries": [{"candidates": [{"frame": 10}]}],
    }


def test_select_proposal_candidate_accepts_command_matched_schema() -> None:
    proposal = _proposal()
    proposal["schema"] = "justin-s13-tracker-recovery-proposal/v2"
    _, candidate = select_proposal_candidate(
        proposal, query_index=0, candidate_index=0
    )
    assert candidate["frame"] == 10


def _exact_history_query() -> dict:
    state = {
        "root_pose_wxyz": [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
        "root_velocity_world": [0.0] * 6,
        "joint_pos_isaac": [0.0] * 29,
        "joint_vel_isaac": [0.0] * 29,
    }
    return {
        "trace_index": 19,
        "learner_state": state,
        "learner_tracker_history": [
            {
                "trace_index": index,
                "physics_step": index * 4,
                "state": state,
                "last_action_isaac_normalized": [index / 100.0] * 29,
            }
            for index in range(10, 20)
        ],
    }


def test_exact_history_maps_learner_chronology_to_reference_clock() -> None:
    history = validate_exact_learner_history(
        _exact_history_query(), {"frame": 100}
    )
    assert [row["trace_index"] for row in history] == list(range(10, 20))
    assert [row["reference_frame"] for row in history] == list(range(91, 101))
    assert history[-1]["last_action_isaac_normalized"].shape == (29,)


def test_exact_history_rejects_noncontiguous_or_state_drift() -> None:
    query = _exact_history_query()
    query["learner_tracker_history"][3]["trace_index"] = 99
    with pytest.raises(TrackerRecoveryError, match="not contiguous"):
        validate_exact_learner_history(query, {"frame": 100})

    query = _exact_history_query()
    query["learner_tracker_history"][-1]["state"] = dict(
        query["learner_tracker_history"][-1]["state"]
    )
    query["learner_tracker_history"][-1]["state"]["joint_pos_isaac"] = [0.1] * 29
    with pytest.raises(TrackerRecoveryError, match="differs from history tail"):
        validate_exact_learner_history(query, {"frame": 100})


def test_select_proposal_candidate_is_strict() -> None:
    query, candidate = select_proposal_candidate(
        _proposal(), query_index=0, candidate_index=0
    )
    assert candidate["frame"] == 10
    with pytest.raises(TrackerRecoveryError, match="out of range"):
        select_proposal_candidate(_proposal(), query_index=1, candidate_index=0)


@pytest.mark.parametrize("value", [0.0, 0.25, 1.0])
def test_reference_blend_accepts_closed_unit_interval(value: float) -> None:
    assert validate_reference_blend(value) == value


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf])
def test_reference_blend_rejects_invalid_values(value: float) -> None:
    with pytest.raises(TrackerRecoveryError, match=r"in \[0, 1\]"):
        validate_reference_blend(value)


def test_quaternion_blend_uses_shortest_arc() -> None:
    import torch

    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    same_rotation_opposite_sign = -identity
    halfway = _slerp_wxyz(identity, same_rotation_opposite_sign, 0.5)
    assert torch.allclose(halfway, identity)

    half_turn = torch.tensor([0.0, 0.0, 0.0, 1.0])
    quarter_turn = _slerp_wxyz(identity, half_turn, 0.5)
    assert torch.allclose(
        quarter_turn,
        torch.tensor(
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=quarter_turn.dtype
        ),
        atol=1.0e-6,
    )


def test_tracker_acceptance_requires_screen_completion_and_stable_top() -> None:
    platform_root = np.tile(np.asarray((-1.0, -4.3, 1.3)), (60, 1))
    accepted = evaluate_tracker_attempt(
        mpjpe_mm=np.full(60, 120.0),
        root_position_m=platform_root,
        terminated_early=False,
        completed_reference=True,
    )
    assert accepted["screen_pass"] is True
    assert accepted["reached_platform"] is True
    assert accepted["stable_platform"] is True
    assert accepted["stable_top"] is True
    assert accepted["accepted_recovery"] is True


@pytest.mark.parametrize(
    ("mpjpe", "root", "terminated", "completed", "failed_gate"),
    [
        (
            np.full(60, 301.0),
            np.tile((-1.0, -4.3, 1.3), (60, 1)),
            False,
            True,
            "screen_pass",
        ),
        (
            np.full(60, 120.0),
            np.tile((-0.7, -4.3, 1.14), (60, 1)),
            False,
            True,
            "reached_platform",
        ),
        (
            np.full(60, 120.0),
            np.tile((-1.0, -4.3, 1.3), (60, 1)),
            True,
            True,
            "screen_pass",
        ),
        (
            np.full(60, 120.0),
            np.tile((-1.0, -4.3, 1.3), (60, 1)),
            False,
            False,
            "stable_top",
        ),
    ],
)
def test_tracker_acceptance_rejects_each_failure_gate(
    mpjpe, root, terminated, completed, failed_gate
) -> None:
    result = evaluate_tracker_attempt(
        mpjpe_mm=mpjpe,
        root_position_m=root,
        terminated_early=terminated,
        completed_reference=completed,
    )
    assert result[failed_gate] is False
    assert result["accepted_recovery"] is False


def test_second_stage_evaluation_ignores_pre_rematch_tracking_spike() -> None:
    root = np.tile(np.asarray((-1.0, -4.3, 1.3)), (60, 1))
    result = evaluate_second_stage_recovery(
        mpjpe_mm=np.concatenate((np.full(10, 405.0), np.full(50, 74.0))),
        root_position_m=root,
        rematch_step=10,
        terminated_early=False,
        completed_reference=True,
    )
    assert result["rematch_step"] == 10
    assert result["suffix_steps"] == 50
    assert result["peak_mpjpe_mm"] == pytest.approx(74.0)
    assert result["accepted_second_stage_recovery"] is True


def test_adaptive_reference_clock_freezes_resumes_and_holds_terminal() -> None:
    clock = AdaptiveReferenceClock(
        freeze_root_error_m=0.20,
        resume_root_error_m=0.12,
        max_consecutive_hold_steps=2,
    )
    assert clock.should_advance(0.10, at_terminal=False) is True
    assert clock.should_advance(0.21, at_terminal=False) is False
    assert clock.should_advance(0.15, at_terminal=False) is False
    assert clock.should_advance(0.14, at_terminal=False) is True
    assert clock.should_advance(0.11, at_terminal=False) is True
    assert clock.should_advance(0.05, at_terminal=True) is False
    receipt = clock.receipt()
    assert receipt["freeze_events"] == 1
    assert receipt["resume_events"] == 1
    assert receipt["hold_steps"] == 3
    assert receipt["terminal_hold_steps"] == 1
    assert receipt["longest_hold_steps"] == 2
    assert receipt["forced_advance_steps"] == 1


def test_adaptive_reference_clock_rejects_unbounded_pose_holds() -> None:
    with pytest.raises(TrackerRecoveryError, match="consecutive hold step"):
        AdaptiveReferenceClock(max_consecutive_hold_steps=0)


@pytest.mark.parametrize(
    ("freeze", "resume"),
    [(0.1, 0.1), (0.1, 0.2), (np.inf, 0.1), (0.2, 0.0)],
)
def test_adaptive_reference_clock_rejects_invalid_thresholds(
    freeze: float, resume: float
) -> None:
    with pytest.raises(TrackerRecoveryError, match="resume error < freeze error"):
        AdaptiveReferenceClock(
            freeze_root_error_m=freeze, resume_root_error_m=resume
        )


def test_second_stage_rematch_requires_consecutive_step_two_observations() -> None:
    trigger = SecondStageRematchTrigger(stable_steps=3)
    on_step = np.asarray((-0.65, -4.30, 1.12))
    off_step = np.asarray((-0.50, -4.30, 1.12))
    assert trigger.observe(on_step) is False
    assert trigger.observe(on_step) is False
    assert trigger.observe(off_step) is False
    assert trigger.observe(on_step) is False
    assert trigger.observe(on_step) is False
    assert trigger.observe(on_step) is True
    assert trigger.observe(on_step) is False


def test_second_stage_reference_match_balances_root_and_joint_pose() -> None:
    match = select_second_stage_reference_frame(
        robot_root_position_m=np.asarray((-0.60, -4.30, 1.10)),
        robot_joint_position_rad=np.zeros(4),
        candidate_frames=np.asarray((230, 240)),
        reference_root_position_m=np.asarray(
            ((-0.60, -4.30, 1.10), (-0.65, -4.30, 1.10))
        ),
        reference_joint_position_rad=np.asarray(((1.0,) * 4, (0.0,) * 4)),
        joint_weight_m_per_rad=0.10,
    )
    assert match["frame"] == 240
    assert match["root_position_error_m"] == pytest.approx(0.05)
    assert match["joint_position_rmse_rad"] == pytest.approx(0.0)
    assert match["score_m"] == pytest.approx(0.05)


def test_second_stage_reference_match_rejects_shape_drift() -> None:
    with pytest.raises(TrackerRecoveryError, match="invalid shapes"):
        select_second_stage_reference_frame(
            robot_root_position_m=np.zeros(3),
            robot_joint_position_rad=np.zeros(4),
            candidate_frames=np.asarray((230, 240)),
            reference_root_position_m=np.zeros((2, 3)),
            reference_joint_position_rad=np.zeros((2, 3)),
            joint_weight_m_per_rad=0.10,
        )
