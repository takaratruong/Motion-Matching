from __future__ import annotations

from mm_sonic.full_walking_terrain_lmm_viewer import (
    FORMAL_SCENE_IDS,
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


def test_cli_matches_the_planned_smoke_and_view_commands() -> None:
    parser = build_parser()
    smoke = parser.parse_args(
        [
            "smoke",
            "--corpus", "corpus",
            "--model", "model",
            "--baseline-corpus", "baseline-corpus",
            "--baseline-model", "baseline-model",
            "--scene", "flat-standard",
            "--scene", "stairs-standard",
            "--frames-per-scene", "2500",
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
