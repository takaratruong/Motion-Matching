from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from mm_sonic.justin_tracker_recovery import (
    TrackerRecoveryError,
    _compatible_urdf_extension_name,
    _disable_builtin_failure_terminations,
    _kit_cache_args,
    _slerp_wxyz,
    evaluate_tracker_attempt,
    select_proposal_candidate,
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
    accepted = evaluate_tracker_attempt(
        mpjpe_mm=np.full(60, 120.0),
        root_z_m=np.full(60, 1.3),
        terminated_early=False,
        completed_reference=True,
    )
    assert accepted["screen_pass"] is True
    assert accepted["stable_top"] is True
    assert accepted["accepted_recovery"] is True


@pytest.mark.parametrize(
    ("mpjpe", "root_z", "terminated", "completed", "failed_gate"),
    [
        (np.full(60, 301.0), np.full(60, 1.3), False, True, "screen_pass"),
        (np.full(60, 120.0), np.full(60, 1.0), False, True, "stable_top"),
        (np.full(60, 120.0), np.full(60, 1.3), True, True, "screen_pass"),
        (np.full(60, 120.0), np.full(60, 1.3), False, False, "stable_top"),
    ],
)
def test_tracker_acceptance_rejects_each_failure_gate(
    mpjpe, root_z, terminated, completed, failed_gate
) -> None:
    result = evaluate_tracker_attempt(
        mpjpe_mm=mpjpe,
        root_z_m=root_z,
        terminated_early=terminated,
        completed_reference=completed,
    )
    assert result[failed_gate] is False
    assert result["accepted_recovery"] is False
