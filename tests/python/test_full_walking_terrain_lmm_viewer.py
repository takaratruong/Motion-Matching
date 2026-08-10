from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import mm_sonic.full_walking_terrain_lmm_viewer as viewer_module
from mm_sonic.full_walking_terrain_lmm_viewer import (
    FORMAL_SCENE_IDS,
    FULL_LABEL,
    _baseline_authority_exact,
    _baseline_authority_snapshot,
    _formal_identity,
    _full_formal_artifact_authorities,
    _full_runtime_identity,
    _full_runtime_step_hz,
    _load_baseline_corpus,
    acceptance_failures,
    build_parser,
    main,
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
            "corpus_row_count": 9_758_524,
            "corpus_range_count": 16_999,
            "corpus_eligible_row_count": 9_758_524,
            "corpus_manifest_authority_current": True,
            "model_manifest_authority_current": True,
            "model_corpus_binding_current": True,
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
                "selection_model_path": (
                    "sonic/runs/g1-hybrid-terrain-lmm/"
                    "selection-combined-v2-strict-latent32-visual-v1"
                ),
                "selection_evaluation_sha256": (
                    "959f65141346a0dbd322472470414cb7f29fb1d070206cb16b0fc7ee8c08d494"
                ),
                "selection_evaluation_path": (
                    "sonic/runs/g1-hybrid-terrain-lmm/"
                    "selection-combined-v2-strict-latent32-visual-v1/"
                    "evaluation.json"
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
                "selection_model_path": (
                    "sonic/runs/g1-hybrid-terrain-lmm/"
                    "selection-combined-v2-strict-latent32-visual-v1"
                ),
                "selection_evaluation_sha256": (
                    "959f65141346a0dbd322472470414cb7f29fb1d070206cb16b0fc7ee8c08d494"
                ),
                "selection_evaluation_path": (
                    "sonic/runs/g1-hybrid-terrain-lmm/"
                    "selection-combined-v2-strict-latent32-visual-v1/"
                    "evaluation.json"
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

    for name, value in (
        ("corpus_row_count", 9_758_523),
        ("corpus_range_count", 16_998),
        ("corpus_eligible_row_count", 9_758_523),
        ("fps", 25.0),
        ("horizons", [8, 16, 24]),
    ):
        wrong = dict(identity)
        wrong[name] = value
        assert not _full_formal_artifact_authorities(wrong), name


def _write_authority(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_baseline_snapshot_hashes_every_frozen_file_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorities = dict(viewer_module._OVERNIGHT_BASELINE_AUTHORITIES)
    corpus_root = tmp_path / authorities["corpus_path"]
    model_root = tmp_path / authorities["model_path"]
    selection_root = tmp_path / authorities["selection_model_path"]
    paths = {
        "corpus_manifest_sha256": corpus_root / "manifest.json",
        "model_manifest_sha256": model_root / "manifest.json",
        "selection_model_manifest_sha256": selection_root / "manifest.json",
        "selection_evaluation_sha256": selection_root / "evaluation.json",
        "ramp_smoke_receipt_sha256": tmp_path / authorities["ramp_smoke_receipt_path"],
        "stairs_smoke_receipt_sha256": tmp_path
        / authorities["stairs_smoke_receipt_path"],
    }
    for index, (name, path) in enumerate(paths.items()):
        authorities[name] = _write_authority(path, f"authority-{index}".encode())
    monkeypatch.setattr(viewer_module, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(viewer_module, "_OVERNIGHT_BASELINE_AUTHORITIES", authorities)
    evaluator = SimpleNamespace(
        corpus=SimpleNamespace(source_root=corpus_root),
        generator=SimpleNamespace(root=model_root),
    )

    first = _baseline_authority_snapshot(evaluator)
    assert _baseline_authority_exact(first)

    paths["ramp_smoke_receipt_sha256"].unlink()
    absent = _baseline_authority_snapshot(evaluator)
    assert absent["ramp_smoke_receipt_sha256"] is None
    assert not _baseline_authority_exact(absent)
    absent_receipt = _green_receipt()
    absent_receipt["formal_identity"]["baseline_authority_pre"] = absent
    assert "frozen-baseline-provenance-required" in acceptance_failures(absent_receipt)

    _write_authority(paths["ramp_smoke_receipt_sha256"], b"tampered-ramp-authority")
    tampered = _baseline_authority_snapshot(evaluator)
    assert (
        tampered["ramp_smoke_receipt_sha256"]
        != authorities["ramp_smoke_receipt_sha256"]
    )
    assert tampered != first
    assert not _baseline_authority_exact(tampered)
    changed_receipt = _green_receipt()
    changed_receipt["formal_identity"]["baseline_authority_pre"] = first
    changed_receipt["formal_identity"]["baseline_authority_post"] = tampered
    assert "baseline-authority-unchanged-required" in acceptance_failures(
        changed_receipt
    )


def test_frozen_combined_baseline_uses_the_combined_authority_loader() -> None:
    loaded = object()
    module = SimpleNamespace(load_combined_cache=mock.Mock(return_value=loaded))
    with mock.patch.object(
        viewer_module.importlib, "import_module", return_value=module
    ) as imported:
        assert _load_baseline_corpus(Path("baseline")) is loaded

    imported.assert_called_once_with("mm_sonic.hybrid_terrain_lmm_combined")
    module.load_combined_cache.assert_called_once_with(Path("baseline"))


def test_formal_identity_reads_authenticated_task4_corpus_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    manifest = {
        "schema": "g1-full-walking-terrain-lmm-corpus/v1",
        "fps": 60.0,
        "horizons": [20, 40, 60],
        "rows": 9_758_524,
        "ranges": 16_999,
        "sources": 15_918,
        "eligible_rows": 9_758_524,
    }
    payload = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    (root / "manifest.json").write_bytes(payload)
    source_names = tuple(
        [f"pfnn:{index}" for index in range(80)]
        + [f"grail:{index}" for index in range(15_918 - 80)]
    )
    corpus = SimpleNamespace(
        root=root,
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        fps=60.0,
        horizons=(20, 40, 60),
        source_names=source_names,
    )
    evaluator = SimpleNamespace(
        corpus=corpus,
        generator=SimpleNamespace(
            manifest={
                "schema": "g1-full-walking-terrain-lmm-model/v1",
                "corpus_manifest_sha256": corpus.manifest_sha256,
            }
        ),
        matchers={},
        adapters={},
    )
    g1_xml = tmp_path / "g1.xml"
    g1_xml.write_text("<mujoco/>")
    monkeypatch.setattr(
        viewer_module,
        "_g1_xml_asset_identity",
        lambda _path: ("a" * 64, {}),
    )

    identity = _formal_identity(evaluator, scenes=(), g1_xml=g1_xml)

    assert identity["corpus_schema"] == manifest["schema"]
    assert identity["inventory_terminal_count"] == 15_918
    assert identity["pfnn_bvh_count"] == 80
    assert identity["corpus_row_count"] == 9_758_524
    assert identity["corpus_range_count"] == 16_999
    assert identity["corpus_eligible_row_count"] == 9_758_524
    assert identity["corpus_manifest_authority_current"] is True


def test_full_view_passes_full_identity_and_visible_label_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = SimpleNamespace()
    generator = SimpleNamespace(
        manifest={"canonical_selection_accepted": True},
        config=SimpleNamespace(fit_all_rows=True),
        canonical_selection_verified=True,
        selection_provenance_verified=True,
    )
    adapter = SimpleNamespace(
        authority=object(),
        spawn_native_xy=np.zeros(2),
        spawn_heading=0.0,
        scene_authenticated=True,
    )
    matcher = SimpleNamespace(
        generator=generator,
        search_acceptance_eligible=True,
        fps=60.0,
    )
    call: dict[str, object] = {}

    def fake_run_interactive(*args: object, **kwargs: object) -> dict[str, object]:
        call.update(kwargs)
        label = kwargs["runtime_label_resolver"](
            matcher,
            adapter,
            scene_authentication_current=True,
            formal_authorities_current=True,
        )
        return {"accepted": True, "label": label}

    monkeypatch.setattr(viewer_module, "_load_full_corpus", lambda _path: corpus)
    monkeypatch.setattr(viewer_module, "_load_generator", lambda *_args: generator)
    monkeypatch.setattr(
        viewer_module, "load_scene_terrain", lambda *_args, **_kwargs: adapter
    )
    monkeypatch.setattr(
        viewer_module, "HybridMatcher", lambda *_args, **_kwargs: matcher
    )
    monkeypatch.setattr(viewer_module, "run_interactive", fake_run_interactive)
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(MjModel=SimpleNamespace(from_xml_path=lambda _path: object())),
    )

    assert main(["view", "--corpus", "corpus", "--model", "model"]) == 0

    assert call["runtime_identity_resolver"] is _full_runtime_identity
    assert call["formal_authority_predicate"] is _full_formal_artifact_authorities
    assert call["runtime_label_resolver"] is viewer_module._full_runtime_label
    assert fake_run_interactive(matcher, adapter, **call)["label"] == FULL_LABEL


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
