from __future__ import annotations

import pytest
from mm_sonic.full_walking_terrain_lmm_viewer import (
    FORMAL_SCENE_IDS,
    _full_formal_artifact_authorities,
    _full_runtime_step_hz,
    acceptance_failures,
    build_parser,
)
from mm_sonic.hybrid_terrain_lmm_viewer import _runtime_step_hz


def _green_receipt() -> dict[str, object]:
    return {
        "route_command_authority_sha256": "a" * 64,
        "candidate": {
            "route_command_authority_sha256": "a" * 64,
            "forward_count": 10_000,
            "slip": {"median_mps": 0.01, "p95_mps": 0.04},
            "query_distance_p95": 0.5,
            "speed_mae_mps": 0.08,
            "safety_counts": {
                name: 0
                for name in (
                    "candidate_exhaustion",
                    "fallback",
                    "joint_clamp",
                    "native_limit_violation",
                    "nonfinite",
                    "out_of_range_successor",
                )
            },
            "by_terrain": {
                name: {"canonical_identity_count": 2}
                for name in ("flat", "curb", "slope", "stair")
            },
        },
        "baseline": {
            "route_command_authority_sha256": "a" * 64,
            "forward_count": 4_168,
            "slip": {"median_mps": 0.04, "p95_mps": 0.1},
            "query_distance_p95": 1.0,
        },
        "comparison": {
            "query_distance_p95_reduction_fraction": 0.5,
            "slip_p95_reduction_fraction": 0.6,
        },
        "formal_identity": {
            "corpus_schema": "g1-full-walking-terrain-lmm-corpus/v1",
            "model_schema": "g1-full-walking-terrain-lmm-model/v1",
            "fps": 60.0,
            "horizons": [20, 40, 60],
            "source_identity_count": 15_918,
            "inventory_terminal_count": 15_918,
            "pfnn_bvh_count": 80,
            "split_receipt_current": True,
            "test_receipt_current": True,
            "refit_receipt_current": True,
            "determinism_receipt_current": True,
            "full_search": True,
            "motion_root_ownership": True,
            "supported_speed_envelope_current": True,
            "retry_budget": 32,
            "g1_xml_sha256": "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376",
            "g1_asset_inventory_sha256": "47dcad0d533434233e7769486ae79074751be9eeb583af39f38e219c96c71a28",
            "authenticated_scene_ids": list(FORMAL_SCENE_IDS),
            "baseline_authority_pre": {
                "corpus_path": "sonic/runs/g1-hybrid-terrain-lmm/corpus-primary-pfnn-v2-strict",
                "corpus_manifest_sha256": (
                    "084c168b473e730ec24419a49f4be1526226e5f95a2dd81ae4d367c729889cdb"
                ),
                "model_path": (
                    "sonic/runs/g1-hybrid-terrain-lmm/"
                    "final-combined-v2-strict-latent32-visual-v1-allrows"
                ),
                "model_manifest_sha256": (
                    "f200db342dd514df6deb6203f3be014054efd4f9038dc0486cc3a4b274a6d826"
                ),
                "selection_model_manifest_sha256": (
                    "0f8d36395012d59deac2c54a88ba64252fc959fae00cd9255a4545479b49fa90"
                ),
                "selection_evaluation_sha256": (
                    "959f65141346a0dbd322472470414cb7f29fb1d070206cb16b0fc7ee8c08d494"
                ),
                "ramp_smoke_receipt_path": (
                    "sonic/runs/g1-hybrid-terrain-lmm/evidence/"
                    "final-combined-v2-strict-latent32-visual-v1-allrows-"
                    "ramp10-full-1000-formal-v2.json"
                ),
                "ramp_smoke_receipt_sha256": (
                    "0ad82ae339c6f55d7a4af52b18548a09d4b208624fa359832f99f0d88ab25f66"
                ),
                "stairs_smoke_receipt_path": (
                    "sonic/runs/g1-hybrid-terrain-lmm/evidence/"
                    "final-combined-v2-strict-latent32-visual-v1-allrows-"
                    "stairs-standard-full-1000-formal-v2.json"
                ),
                "stairs_smoke_receipt_sha256": (
                    "0c2f1de3e65af7fcec60cbb1cfbd001b605439a5d28aefe654698e00e8426a44"
                ),
            },
            "baseline_authority_post": {
                "corpus_path": "sonic/runs/g1-hybrid-terrain-lmm/corpus-primary-pfnn-v2-strict",
                "corpus_manifest_sha256": (
                    "084c168b473e730ec24419a49f4be1526226e5f95a2dd81ae4d367c729889cdb"
                ),
                "model_path": (
                    "sonic/runs/g1-hybrid-terrain-lmm/"
                    "final-combined-v2-strict-latent32-visual-v1-allrows"
                ),
                "model_manifest_sha256": (
                    "f200db342dd514df6deb6203f3be014054efd4f9038dc0486cc3a4b274a6d826"
                ),
                "selection_model_manifest_sha256": (
                    "0f8d36395012d59deac2c54a88ba64252fc959fae00cd9255a4545479b49fa90"
                ),
                "selection_evaluation_sha256": (
                    "959f65141346a0dbd322472470414cb7f29fb1d070206cb16b0fc7ee8c08d494"
                ),
                "ramp_smoke_receipt_path": (
                    "sonic/runs/g1-hybrid-terrain-lmm/evidence/"
                    "final-combined-v2-strict-latent32-visual-v1-allrows-"
                    "ramp10-full-1000-formal-v2.json"
                ),
                "ramp_smoke_receipt_sha256": (
                    "0ad82ae339c6f55d7a4af52b18548a09d4b208624fa359832f99f0d88ab25f66"
                ),
                "stairs_smoke_receipt_path": (
                    "sonic/runs/g1-hybrid-terrain-lmm/evidence/"
                    "final-combined-v2-strict-latent32-visual-v1-allrows-"
                    "stairs-standard-full-1000-formal-v2.json"
                ),
                "stairs_smoke_receipt_sha256": (
                    "0c2f1de3e65af7fcec60cbb1cfbd001b605439a5d28aefe654698e00e8426a44"
                ),
            },
            "baseline_authority_current_pre": True,
            "baseline_authority_current_post": True,
        },
    }


def test_formal_acceptance_is_green_only_for_full_safe_evidence() -> None:
    assert acceptance_failures(_green_receipt()) == []


def test_formal_acceptance_rejects_every_provenance_and_runtime_shortcut() -> None:
    mutations = {
        "overnight-cache": ("formal_identity", "corpus_schema", "old"),
        "partial-pfnn": ("formal_identity", "pfnn_bvh_count", 17),
        "capped-search": ("formal_identity", "full_search", False),
        "wrong-assets": ("formal_identity", "g1_xml_sha256", "0" * 64),
        "missing-split": ("formal_identity", "split_receipt_current", False),
        "missing-test": ("formal_identity", "test_receipt_current", False),
        "missing-refit": ("formal_identity", "refit_receipt_current", False),
        "command-root": ("formal_identity", "motion_root_ownership", False),
        "exhaustion": ("candidate", "safety_counts", {"candidate_exhaustion": 1}),
        "too-short": ("candidate", "forward_count", 9_999),
    }
    for label, (section, key, value) in mutations.items():
        receipt = _green_receipt()
        receipt[section][key] = value  # type: ignore[index]
        assert acceptance_failures(receipt), label


def test_each_formal_terrain_requires_two_canonical_identities() -> None:
    receipt = _green_receipt()
    receipt["candidate"]["by_terrain"]["stair"]["canonical_identity_count"] = 1  # type: ignore[index]

    assert "stair-two-canonical-identities-required" in acceptance_failures(receipt)


def test_formal_acceptance_requires_frozen_baseline_provenance_not_only_25hz() -> None:
    missing = _green_receipt()
    missing["formal_identity"].pop("baseline_authority_pre")  # type: ignore[index]
    missing["formal_identity"]["baseline_authority_current_pre"] = False  # type: ignore[index]

    assert "frozen-baseline-provenance-required" in acceptance_failures(missing)
    assert "baseline-authority-current-required" in acceptance_failures(missing)

    changed = _green_receipt()
    changed["formal_identity"]["baseline_authority_post"]["model_manifest_sha256"] = (
        "0" * 64
    )  # type: ignore[index]

    assert "frozen-baseline-provenance-required" in acceptance_failures(changed)
    assert "baseline-authority-unchanged-required" in acceptance_failures(changed)


def test_full_viewer_authority_gate_rejects_legacy_combined_counts() -> None:
    identity = _green_receipt()["formal_identity"]

    assert _full_formal_artifact_authorities(identity)

    legacy = {
        "corpus_row_count": 3_973_057,
        "corpus_range_count": 15_833,
        "corpus_train_row_count": 3_575_813,
        "corpus_evaluation_row_count": 397_244,
        "generator_loader_authority": True,
        "total_safe_row_count": 3_957_224,
        "total_safe_family_counts": {
            "flat": 18_174,
            "curb": 440_481,
            "slope": 462_393,
            "stair": 3_036_176,
        },
    }
    assert not _full_formal_artifact_authorities(legacy)

    mutated = dict(identity)
    mutated["source_identity_count"] = 15_917
    assert not _full_formal_artifact_authorities(mutated)


def test_cli_matches_the_planned_smoke_and_view_commands() -> None:
    parser = build_parser()
    smoke = parser.parse_args(
        [
            "smoke",
            "--corpus",
            "corpus",
            "--model",
            "model",
            "--baseline-corpus",
            "baseline-corpus",
            "--baseline-model",
            "baseline-model",
            "--scene",
            "flat-standard",
            "--scene",
            "stairs-standard",
            "--frames-per-scene",
            "2500",
        ]
    )
    view = parser.parse_args(
        ["view", "--corpus", "corpus", "--model", "model", "--scene", "ramp-10-up-down"]
    )

    assert smoke.frames_per_scene == 2500
    assert smoke.scenes == ["flat-standard", "stairs-standard"]
    assert view.scene == "ramp-10-up-down"


def test_interactive_runtime_uses_authenticated_matcher_rate() -> None:
    assert _runtime_step_hz(type("Matcher", (), {"fps": 60.0})()) == 60.0
    assert _runtime_step_hz(object()) == 25.0


def test_full_interactive_runtime_requires_authenticated_exact_60hz() -> None:
    assert _full_runtime_step_hz(type("Matcher", (), {"fps": 60.0})()) == 60.0
    with pytest.raises(ValueError, match="authenticated 60 Hz"):
        _full_runtime_step_hz(object())
    with pytest.raises(ValueError, match="authenticated 60 Hz"):
        _full_runtime_step_hz(type("Matcher", (), {"fps": 25.0})())
