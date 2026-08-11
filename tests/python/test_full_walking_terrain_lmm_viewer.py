from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mm_sonic.full_walking_terrain_lmm_viewer as viewer_module
import numpy as np
import pytest
from resources.g1_terrain_builder.artifacts import canonical_json_bytes
from mm_sonic.full_walking_terrain_lmm_viewer import (
    FORMAL_SCENE_IDS,
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
            "determinism_receipt_path": (
                "sonic/runs/g1-full-walking-terrain-lmm/evidence/"
                "corpus-v1-determinism.json"
            ),
            "determinism_receipt_schema": (
                "g1-full-walking-terrain-lmm-determinism/v1"
            ),
            "split_receipt_current": True,
            "test_receipt_current": True,
            "refit_receipt_current": True,
            "determinism_receipt_current": True,
            "full_search": True,
            "search_backend_identities": {
                scene: "cpu-ckdtree-exact" for scene in FORMAL_SCENE_IDS
            },
            "cpu_search_backend_only": True,
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
                "primary_cache_manifest_path": (
                    "sonic/runs/g1-hybrid-terrain-lmm/corpus-v2-strict/manifest.json"
                ),
                "primary_cache_manifest_sha256": (
                    "e6fbe9413d4cbca58697b90a92f12a1832d28b961972624dd373bc72fa45494e"
                ),
                "primary_cache_manifest_schema": (
                    "g1-hybrid-terrain-lmm-corpus/v2-strict"
                ),
                "primary_cache_manifest_size_bytes": 3_528,
                "primary_cache_receipt_current": True,
                "pfnn_supplement_manifest_path": (
                    "sonic/runs/g1-hybrid-terrain-lmm/pfnn-supplement-v1/manifest.json"
                ),
                "pfnn_supplement_manifest_sha256": (
                    "618ff022ed781ffcace6d7cc95ca74420468e9be2882903548b8b103b5cc7db8"
                ),
                "pfnn_supplement_manifest_schema": ("pfnn-terrain-lmm-supplement/v1"),
                "pfnn_supplement_manifest_size_bytes": 73_888,
                "pfnn_supplement_receipt_current": True,
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
                "primary_cache_manifest_path": (
                    "sonic/runs/g1-hybrid-terrain-lmm/corpus-v2-strict/manifest.json"
                ),
                "primary_cache_manifest_sha256": (
                    "e6fbe9413d4cbca58697b90a92f12a1832d28b961972624dd373bc72fa45494e"
                ),
                "primary_cache_manifest_schema": (
                    "g1-hybrid-terrain-lmm-corpus/v2-strict"
                ),
                "primary_cache_manifest_size_bytes": 3_528,
                "primary_cache_receipt_current": True,
                "pfnn_supplement_manifest_path": (
                    "sonic/runs/g1-hybrid-terrain-lmm/pfnn-supplement-v1/manifest.json"
                ),
                "pfnn_supplement_manifest_sha256": (
                    "618ff022ed781ffcace6d7cc95ca74420468e9be2882903548b8b103b5cc7db8"
                ),
                "pfnn_supplement_manifest_schema": ("pfnn-terrain-lmm-supplement/v1"),
                "pfnn_supplement_manifest_size_bytes": 73_888,
                "pfnn_supplement_receipt_current": True,
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


def test_formal_acceptance_rejects_programmatic_gpu_candidate() -> None:
    mutations = (
        (
            {scene: "single-gpu-full-row-fp32:cuda:5" for scene in FORMAL_SCENE_IDS},
            True,
        ),
        (
            {scene: "cpu-ckdtree-exact" for scene in FORMAL_SCENE_IDS},
            False,
        ),
        ({"flat-standard": "cpu-ckdtree-exact"}, True),
    )
    for backend_identities, cpu_only in mutations:
        receipt = _green_receipt()
        receipt["formal_identity"]["search_backend_identities"] = (  # type: ignore[index]
            backend_identities
        )
        receipt["formal_identity"]["cpu_search_backend_only"] = cpu_only  # type: ignore[index]

        assert "cpu-exact-search-backend-required" in acceptance_failures(receipt)


def test_formal_acceptance_rejects_every_provenance_and_runtime_shortcut() -> None:
    mutations = {
        "overnight-cache": ("formal_identity", "corpus_schema", "old"),
        "partial-pfnn": ("formal_identity", "pfnn_bvh_count", 17),
        "capped-search": ("formal_identity", "full_search", False),
        "wrong-assets": ("formal_identity", "g1_xml_sha256", "0" * 64),
        "missing-split": ("formal_identity", "split_receipt_current", False),
        "missing-test": ("formal_identity", "test_receipt_current", False),
        "missing-refit": ("formal_identity", "refit_receipt_current", False),
        "missing-determinism": (
            "formal_identity",
            "determinism_receipt_current",
            False,
        ),
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
    upstream = {
        "primary_cache": (
            "g1-hybrid-terrain-lmm-corpus/v2-strict",
            "sonic/runs/g1-hybrid-terrain-lmm/corpus-v2-strict/manifest.json",
        ),
        "pfnn_supplement": (
            "pfnn-terrain-lmm-supplement/v1",
            "sonic/runs/g1-hybrid-terrain-lmm/pfnn-supplement-v1/manifest.json",
        ),
    }
    upstream_receipts = {}
    for label, (schema, relative) in upstream.items():
        path = tmp_path / relative
        payload = (json.dumps({"schema": schema}, sort_keys=True) + "\n").encode()
        digest = _write_authority(path, payload)
        prefix = "primary_cache" if label == "primary_cache" else "pfnn_supplement"
        authorities[f"{prefix}_manifest_path"] = relative
        authorities[f"{prefix}_manifest_sha256"] = digest
        authorities[f"{prefix}_manifest_schema"] = schema
        authorities[f"{prefix}_manifest_size_bytes"] = len(payload)
        authorities[f"{prefix}_receipt_current"] = True
        upstream_receipts[label] = {
            "path": str(path),
            "schema": schema,
            "sha256": digest,
            "size_bytes": len(payload),
        }
    for index, (name, path) in enumerate(paths.items()):
        authorities[name] = _write_authority(path, f"authority-{index}".encode())
    monkeypatch.setattr(viewer_module, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(viewer_module, "_OVERNIGHT_BASELINE_AUTHORITIES", authorities)
    evaluator = SimpleNamespace(
        corpus=SimpleNamespace(
            source_root=corpus_root,
            manifest_receipt={"authorities": upstream_receipts},
        ),
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

    primary_path = Path(upstream_receipts["primary_cache"]["path"])
    primary_path.unlink()
    missing_upstream = _baseline_authority_snapshot(evaluator)
    assert missing_upstream["primary_cache_manifest_sha256"] is None
    assert missing_upstream["primary_cache_receipt_current"] is False
    assert not _baseline_authority_exact(missing_upstream)

    _write_authority(primary_path, b'{"schema":"tampered"}\n')
    tampered_upstream = _baseline_authority_snapshot(evaluator)
    assert (
        tampered_upstream["primary_cache_manifest_sha256"]
        != authorities["primary_cache_manifest_sha256"]
    )
    assert tampered_upstream["primary_cache_receipt_current"] is False
    assert not _baseline_authority_exact(tampered_upstream)


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
    matcher = SimpleNamespace(
        search_scope="full-range-safe-corpus",
        searchable_rows=np.arange(3),
        total_searchable_row_count=3,
        search_backend_identity="cpu-ckdtree-exact",
        state=SimpleNamespace(root_motion_source="canonical-simulation-se2"),
        walking_speed_p95_mps=1.0,
        candidate_retry_budget=32,
    )
    evaluator = SimpleNamespace(
        corpus=corpus,
        generator=SimpleNamespace(
            manifest={
                "schema": "g1-full-walking-terrain-lmm-model/v1",
                "corpus_manifest_sha256": corpus.manifest_sha256,
            }
        ),
        matchers={"flat-standard": matcher},
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
    assert identity["determinism_receipt_current"] is False
    assert identity["search_backend_identities"] == {
        "flat-standard": "cpu-ckdtree-exact"
    }
    assert identity["cpu_search_backend_only"] is True

    matcher.search_backend_identity = "single-gpu-full-row-fp32:cuda:5"
    gpu_identity = _formal_identity(evaluator, scenes=(), g1_xml=g1_xml)
    assert gpu_identity["search_backend_identities"] == {
        "flat-standard": "single-gpu-full-row-fp32:cuda:5"
    }
    assert gpu_identity["cpu_search_backend_only"] is False


def test_formal_identity_requires_separate_exact_deterministic_rebuild_receipt(
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
        "members": {},
    }
    manifest_payload = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(manifest_payload)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    corpus = SimpleNamespace(
        root=root,
        manifest_sha256=manifest_sha,
        fps=60.0,
        horizons=(20, 40, 60),
        source_names=tuple(
            [f"pfnn:{index}" for index in range(80)]
            + [f"grail:{index}" for index in range(15_918 - 80)]
        ),
    )
    receipt_path = tmp_path / (
        "sonic/runs/g1-full-walking-terrain-lmm/evidence/corpus-v1-determinism.json"
    )
    receipt = {
        "schema": "g1-full-walking-terrain-lmm-determinism/v1",
        "status": "accepted",
        "reference_manifest_sha256": manifest_sha,
        "reproduced_manifest_sha256": manifest_sha,
        "members": {
            "manifest.json": {
                "path": "manifest.json",
                "sha256": manifest_sha,
                "size_bytes": len(manifest_payload),
            }
        },
    }
    _write_authority(receipt_path, canonical_json_bytes(receipt))
    monkeypatch.setattr(viewer_module, "_REPOSITORY_ROOT", tmp_path)

    current = viewer_module._full_corpus_authority(corpus)

    assert current["determinism_receipt_current"] is True
    assert current["determinism_receipt_schema"] == receipt["schema"]
    assert isinstance(current["determinism_receipt_sha256"], str)

    receipt_path.write_bytes(
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    noncanonical = viewer_module._full_corpus_authority(corpus)
    assert noncanonical["determinism_receipt_current"] is False


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
        diagnostic_stability=True,
        diagnostic_canonical_source_pose=True,
        diagnostic_mechanical_clearance_bounds_m=(),
        search_acceptance_eligible=True,
        search_backend_identity="cpu-ckdtree-exact",
        fps=60.0,
    )
    call: dict[str, object] = {}
    matcher_call: dict[str, object] = {}

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

    def fake_matcher(*_args: object, **kwargs: object) -> object:
        matcher_call.update(kwargs)
        return matcher

    monkeypatch.setattr(viewer_module, "HybridMatcher", fake_matcher)
    monkeypatch.setattr(
        viewer_module,
        "configure_single_gpu_visibility",
        lambda _device: pytest.fail("CPU view must not configure GPU visibility"),
        raising=False,
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
    assert call["display_postprocessor_factory"] is not None
    assert matcher_call["search_device"] is None
    assert matcher_call["diagnostic_stability"] is True
    assert matcher_call["diagnostic_canonical_source_pose"] is True
    assert (
        "EXISTING INERTIALIZER + TERRAIN FOOT LOCK"
        in fake_run_interactive(matcher, adapter, **call)["label"]
    )


def test_full_formal_matcher_explicitly_keeps_diagnostic_stability_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = SimpleNamespace(
        authority=object(),
        spawn_native_xy=np.zeros(2),
        spawn_heading=0.0,
    )
    matcher = object()
    matcher_call: dict[str, object] = {}

    def fake_matcher(*_args: object, **kwargs: object) -> object:
        matcher_call.update(kwargs)
        return matcher

    monkeypatch.setattr(
        viewer_module, "load_scene_terrain", lambda *_args, **_kwargs: adapter
    )
    monkeypatch.setattr(viewer_module, "HybridMatcher", fake_matcher)
    monkeypatch.setattr(viewer_module, "build_viewer_model", lambda *_args: object())
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(MjModel=SimpleNamespace(from_xml_path=lambda _path: object())),
    )
    evaluator = viewer_module._MuJoCoRouteEvaluator(
        object(), object(), g1_xml=Path("g1.xml"), fps=60.0
    )

    built, _, _ = evaluator._matcher("flat-standard")

    assert built is matcher
    assert matcher_call["diagnostic_stability"] is False
    assert matcher_call["diagnostic_canonical_source_pose"] is False


def test_full_gpu_view_configures_before_load_and_wires_search_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    corpus = SimpleNamespace()
    generator = SimpleNamespace()
    adapter = SimpleNamespace(
        authority=object(),
        spawn_native_xy=np.zeros(2),
        spawn_heading=0.0,
    )
    matcher = SimpleNamespace()

    def configure(device: str) -> None:
        events.append(("configure", device, "jax" in sys.modules))

    def load_corpus(_path: Path) -> object:
        events.append("corpus")
        return corpus

    def load_generator(*_args: object) -> object:
        events.append("model")
        return generator

    def build_matcher(*_args: object, **kwargs: object) -> object:
        events.append(("matcher", kwargs["search_device"]))
        return matcher

    monkeypatch.setattr(
        viewer_module, "configure_single_gpu_visibility", configure, raising=False
    )
    monkeypatch.setattr(viewer_module, "_load_full_corpus", load_corpus)
    monkeypatch.setattr(viewer_module, "_load_generator", load_generator)
    monkeypatch.setattr(
        viewer_module, "load_scene_terrain", lambda *_args, **_kwargs: adapter
    )
    monkeypatch.setattr(viewer_module, "HybridMatcher", build_matcher)
    monkeypatch.setattr(
        viewer_module, "run_interactive", lambda *_args, **_kwargs: {"accepted": True}
    )
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(MjModel=SimpleNamespace(from_xml_path=lambda _path: object())),
    )

    assert (
        main(
            [
                "view",
                "--corpus",
                "corpus",
                "--model",
                "model",
                "--search-device",
                "cuda:5",
            ]
        )
        == 0
    )

    assert events == [
        ("configure", "cuda:5", False),
        "corpus",
        "model",
        ("matcher", "cuda:5"),
    ]


def test_full_gpu_view_rejects_conflicting_visibility_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mm_sonic.hybrid_terrain_lmm_gpu_search import (
        configure_single_gpu_visibility,
    )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setattr(
        viewer_module,
        "configure_single_gpu_visibility",
        configure_single_gpu_visibility,
        raising=False,
    )
    monkeypatch.setattr(
        viewer_module,
        "_load_full_corpus",
        lambda _path: pytest.fail("visibility conflict must fail before corpus load"),
    )

    with pytest.raises(RuntimeError, match="visibility conflicts"):
        main(
            [
                "view",
                "--corpus",
                "corpus",
                "--model",
                "model",
                "--search-device",
                "cuda:5",
            ]
        )


def test_full_formal_smoke_never_configures_gpu_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = object()
    generator = object()
    evaluators: list[object] = []

    def build_evaluator(*_args: object, **_kwargs: object) -> object:
        evaluator = object()
        evaluators.append(evaluator)
        return evaluator

    monkeypatch.setattr(
        viewer_module,
        "configure_single_gpu_visibility",
        lambda _device: pytest.fail("formal smoke must remain CPU-only"),
    )
    monkeypatch.setattr(
        viewer_module,
        "build_diagnostic_collision_model",
        lambda *_args: pytest.fail("formal smoke must not build a collision oracle"),
    )
    monkeypatch.setattr(viewer_module, "_load_full_corpus", lambda _path: corpus)
    monkeypatch.setattr(viewer_module, "_load_baseline_corpus", lambda _path: corpus)
    monkeypatch.setattr(viewer_module, "_load_generator", lambda *_args: generator)
    monkeypatch.setattr(viewer_module, "_MuJoCoRouteEvaluator", build_evaluator)
    monkeypatch.setattr(
        viewer_module,
        "run_formal_smoke",
        lambda baseline, candidate, **_kwargs: {
            "accepted": baseline is evaluators[0] and candidate is evaluators[1]
        },
    )

    argv = [
        "smoke",
        "--corpus",
        "corpus",
        "--model",
        "model",
        "--baseline-corpus",
        "baseline-corpus",
        "--baseline-model",
        "baseline-model",
    ]
    for scene in FORMAL_SCENE_IDS:
        argv.extend(("--scene", scene))

    assert main(argv) == 0
    assert len(evaluators) == 2


def test_full_gpu_identity_and_label_remain_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matcher = SimpleNamespace(
        corpus=object(),
        generator=SimpleNamespace(
            manifest={"canonical_selection_accepted": True},
            config=SimpleNamespace(fit_all_rows=True),
            canonical_selection_verified=True,
            selection_provenance_verified=True,
        ),
        search_acceptance_eligible=True,
        search_backend_identity="single-gpu-full-row-fp32:cuda:5",
        last_search_elapsed_ms=7.25,
        warm_search_elapsed_ms=11.5,
        search_scope="full-range-safe-corpus",
        searchable_rows=np.arange(3),
        total_searchable_row_count=3,
    )
    terrain = SimpleNamespace(
        scene_id="ramp-10-up-down",
        scene_evidence_status="authenticated-indexed",
        scene_authenticated=True,
    )
    monkeypatch.setattr(viewer_module, "_formal_identity", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        viewer_module, "scene_authentication_is_current", lambda *_args: True
    )

    identity = _full_runtime_identity(matcher, terrain, Path("g1.xml"))
    label = viewer_module._full_runtime_label(
        matcher,
        terrain,
        scene_authentication_current=True,
        formal_authorities_current=True,
    )

    assert identity["search_backend_identity"] == ("single-gpu-full-row-fp32:cuda:5")
    assert identity["last_search_elapsed_ms"] == 7.25
    assert identity["first_runtime_search_elapsed_ms"] == 11.5
    assert "DIAGNOSTIC" in label
    assert "EXACT SEARCH" not in label


@pytest.mark.parametrize("canonical_source", (False, True))
def test_full_stability_identity_reports_canonical_source_policy(
    monkeypatch: pytest.MonkeyPatch, canonical_source: bool
) -> None:
    matcher = SimpleNamespace(
        corpus=object(),
        generator=object(),
        diagnostic_stability=True,
        diagnostic_canonical_source_pose=canonical_source,
        search_scope="mechanically-filtered-range-safe-corpus",
        searchable_rows=np.arange(5),
        total_searchable_row_count=6,
    )
    terrain = SimpleNamespace(scene_id="flat-standard")
    monkeypatch.setattr(viewer_module, "_formal_identity", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        viewer_module, "scene_authentication_is_current", lambda *_args: True
    )

    identity = _full_runtime_identity(matcher, terrain, Path("g1.xml"))

    assert identity["diagnostic_canonical_source_pose"] is canonical_source
    assert "diagnostic_pose_source" not in identity
    assert "diagnostic_leg_search_limit_rad" not in identity
    assert "diagnostic_arm_slew_limit_rad_per_frame" not in identity


@pytest.mark.parametrize("canonical_source", (False, True))
def test_full_stability_label_reports_mechanical_filter_inventory(
    canonical_source: bool,
) -> None:
    matcher = SimpleNamespace(
        diagnostic_stability=True,
        diagnostic_canonical_source_pose=canonical_source,
        diagnostic_mechanical_clearance_bounds_m=(0.4, 1.2),
        diagnostic_mechanical_retained_searchable_row_count=5,
        total_searchable_row_count=6,
        search_backend_identity="single-gpu-mechanically-filtered-fp32:cuda:5",
        search_acceptance_eligible=False,
    )

    label = viewer_module._full_runtime_label(matcher)

    assert "MECHANICALLY FILTERED" in label
    assert "0.400" in label
    assert "1.200" in label
    assert "5/6" in label
    assert ("CANONICAL SOURCE POSES" in label) is canonical_source
    assert "LEG SEARCH" not in label
    assert "ARM SLEW" not in label
    assert "EXISTING INERTIALIZER + TERRAIN FOOT LOCK" in label
    assert "NOT ACCEPTANCE EVIDENCE" in label


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
        [
            "view",
            "--corpus",
            "corpus",
            "--model",
            "model",
            "--scene",
            "ramp-10-up-down",
            "--search-device",
            "cuda:5",
        ]
    )

    assert smoke.frames_per_scene == 2500
    assert smoke.scenes == ["flat-standard", "stairs-standard"]
    assert not hasattr(smoke, "search_device")
    assert view.scene == "ramp-10-up-down"
    assert view.search_device == "cuda:5"

    with pytest.raises(SystemExit):
        parser.parse_args(
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
                "--search-device",
                "cuda:5",
            ]
        )


def test_interactive_runtime_uses_authenticated_matcher_rate() -> None:
    assert _runtime_step_hz(type("Matcher", (), {"fps": 60.0})()) == 60.0
    assert _runtime_step_hz(object()) == 25.0


def test_full_interactive_runtime_requires_authenticated_exact_60hz() -> None:
    assert _full_runtime_step_hz(type("Matcher", (), {"fps": 60.0})()) == 60.0
    with pytest.raises(ValueError, match="authenticated 60 Hz"):
        _full_runtime_step_hz(object())
    with pytest.raises(ValueError, match="authenticated 60 Hz"):
        _full_runtime_step_hz(type("Matcher", (), {"fps": 25.0})())
