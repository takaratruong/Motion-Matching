"""Formal MuJoCo evidence and interactive viewer for the full walking corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from resources.g1_terrain_builder.artifacts import canonical_json_bytes

from .full_walking_terrain_lmm_evaluation import (
    EvaluationSeries,
    FormalRoute,
    compare_baseline_candidate,
)
from .hybrid_terrain_lmm_runtime import CommandState, HybridMatcher
from .hybrid_terrain_lmm_postprocess import ExistingUtilityPosePostprocessor
from .hybrid_terrain_lmm_gpu_search import configure_single_gpu_visibility
from .full_walking_terrain_lmm_inventory import load_inventory
from .hybrid_terrain_lmm_viewer import (
    DEFAULT_G1_XML,
    _g1_xml_asset_identity,
    _generator_acceptance_status,
    _load_generator,
    build_diagnostic_collision_model,
    build_viewer_model,
    load_scene_terrain,
    run_interactive,
    scene_authentication_is_current,
    write_receipt_exclusive,
)

FORMAL_SCENE_IDS = (
    "flat-standard",
    "grail-curb-default",
    "ramp-10-up-down",
    "stairs-standard",
)
_TERRAINS = ("flat", "curb", "slope", "stair")
_G1_XML_SHA256 = "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376"
_G1_ASSET_INVENTORY_SHA256 = (
    "47dcad0d533434233e7769486ae79074751be9eeb583af39f38e219c96c71a28"
)
_SAFETY_KEYS = (
    "candidate_exhaustion",
    "fallback",
    "joint_clamp",
    "native_limit_violation",
    "nonfinite",
    "out_of_range_successor",
)
FULL_LABEL = (
    "FULL WALKING TERRAIN LMM "
    "(EXACT SEARCH + LEARNED GENERATOR; FULL WALKING CORPUS ONLY)"
)
FULL_DIAGNOSTIC_LABEL = (
    "FULL WALKING TERRAIN LMM (DIAGNOSTIC SEARCH/RUNTIME; NOT ACCEPTANCE EVIDENCE)"
)
FULL_DIAGNOSTIC_TERRAIN_LABEL = (
    "FULL WALKING TERRAIN LMM "
    "(DIAGNOSTIC UNAUTHENTICATED TERRAIN; NOT ACCEPTANCE EVIDENCE)"
)
FULL_DIAGNOSTIC_MODEL_LABEL = (
    "FULL WALKING TERRAIN LMM "
    "(DIAGNOSTIC UNVERIFIED FULL-CORPUS MODEL; NOT ACCEPTANCE EVIDENCE)"
)


def _controller_diagnostic_label(matcher: object) -> str:
    try:
        identity = getattr(matcher, "controller_identity")
        heading_policy = getattr(matcher, "controller_heading_policy")
        yaw_rate = float(getattr(matcher, "controller_yaw_rate_rad_s"))
        acceleration = float(getattr(matcher, "controller_acceleration_mps2"))
        deceleration = float(getattr(matcher, "controller_deceleration_mps2"))
        stop_speed = float(getattr(matcher, "controller_stop_speed_mps"))
        search_interval = float(getattr(matcher, "controller_search_interval_s"))
    except (AttributeError, TypeError, ValueError):
        return ""
    if (
        type(identity) is not str
        or type(heading_policy) is not str
        or not all(
            math.isfinite(value)
            for value in (
                yaw_rate,
                acceleration,
                deceleration,
                stop_speed,
                search_interval,
            )
        )
    ):
        return ""
    return (
        f"CONTROL {identity}; heading {heading_policy}; "
        f"yaw-rate {yaw_rate:.3f} rad/s; accel {acceleration:.3f} m/s^2; "
        f"decel {deceleration:.3f} m/s^2; stop {stop_speed:.3f} m/s; "
        f"search {search_interval:.3f} s"
    )


_FULL_FORMAL_ARTIFACT_AUTHORITIES = {
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
        "sonic/runs/g1-full-walking-terrain-lmm/evidence/corpus-v1-determinism.json"
    ),
    "determinism_receipt_schema": "g1-full-walking-terrain-lmm-determinism/v1",
    "determinism_receipt_current": True,
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DETERMINISM_RECEIPT_PATH = (
    "sonic/runs/g1-full-walking-terrain-lmm/evidence/corpus-v1-determinism.json"
)
_DETERMINISM_RECEIPT_SCHEMA = "g1-full-walking-terrain-lmm-determinism/v1"
_OVERNIGHT_BASELINE_AUTHORITIES = {
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
    "primary_cache_manifest_schema": "g1-hybrid-terrain-lmm-corpus/v2-strict",
    "primary_cache_manifest_size_bytes": 3_528,
    "primary_cache_receipt_current": True,
    "pfnn_supplement_manifest_path": (
        "sonic/runs/g1-hybrid-terrain-lmm/pfnn-supplement-v1/manifest.json"
    ),
    "pfnn_supplement_manifest_sha256": (
        "618ff022ed781ffcace6d7cc95ca74420468e9be2882903548b8b103b5cc7db8"
    ),
    "pfnn_supplement_manifest_schema": "pfnn-terrain-lmm-supplement/v1",
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
        "selection-combined-v2-strict-latent32-visual-v1/evaluation.json"
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
}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _full_formal_artifact_authorities(identity: object) -> bool:
    getter = getattr(identity, "get", lambda *_args: None)
    return all(
        getter(name) == expected
        for name, expected in _FULL_FORMAL_ARTIFACT_AUTHORITIES.items()
    )


def _baseline_authority_exact(identity: object) -> bool:
    getter = getattr(identity, "get", lambda *_args: None)
    return all(
        getter(name) == expected
        for name, expected in _OVERNIGHT_BASELINE_AUTHORITIES.items()
    )


def _path_identity(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def _observed_file_sha256(path: Path) -> str | None:
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            return None
        return hashlib.sha256(resolved.read_bytes()).hexdigest()
    except (OSError, RuntimeError, ValueError):
        return None


def _baseline_upstream_authority(
    corpus: object, authority_name: str, field_prefix: str
) -> dict[str, object]:
    receipt = _mapping(getattr(corpus, "manifest_receipt", {}))
    authorities = _mapping(receipt.get("authorities"))
    descriptor = _mapping(authorities.get(authority_name))
    path_value = descriptor.get("path")
    path = Path(path_value) if isinstance(path_value, str) else None
    payload: bytes | None = None
    manifest: Mapping[str, Any] = {}
    observed_sha: str | None = None
    if path is not None:
        try:
            resolved = path.resolve(strict=True)
            if resolved.is_file():
                payload = resolved.read_bytes()
                decoded = json.loads(payload)
                if type(decoded) is dict:
                    manifest = decoded
                observed_sha = hashlib.sha256(payload).hexdigest()
        except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError, ValueError):
            pass
    expected_path = _REPOSITORY_ROOT / str(
        _OVERNIGHT_BASELINE_AUTHORITIES[f"{field_prefix}_manifest_path"]
    )
    expected_sha = _OVERNIGHT_BASELINE_AUTHORITIES[f"{field_prefix}_manifest_sha256"]
    expected_schema = _OVERNIGHT_BASELINE_AUTHORITIES[f"{field_prefix}_manifest_schema"]
    expected_size = _OVERNIGHT_BASELINE_AUTHORITIES[
        f"{field_prefix}_manifest_size_bytes"
    ]
    descriptor_current = (
        path is not None
        and path.resolve() == expected_path.resolve()
        and descriptor.get("sha256") == expected_sha == observed_sha
        and descriptor.get("schema") == expected_schema == manifest.get("schema")
        and descriptor.get("size_bytes") == expected_size
        and payload is not None
        and len(payload) == expected_size
    )
    return {
        f"{field_prefix}_manifest_path": _path_identity(path),
        f"{field_prefix}_manifest_sha256": observed_sha,
        f"{field_prefix}_manifest_schema": manifest.get("schema"),
        f"{field_prefix}_manifest_size_bytes": (
            len(payload) if payload is not None else None
        ),
        f"{field_prefix}_receipt_current": descriptor_current,
    }


def _baseline_authority_snapshot(evaluator: object) -> dict[str, object]:
    corpus = getattr(evaluator, "corpus", None)
    generator = getattr(evaluator, "generator", None)
    corpus_root = Path(getattr(corpus, "source_root", "."))
    model_root = getattr(generator, "root", None)
    selection_root = _REPOSITORY_ROOT / str(
        _OVERNIGHT_BASELINE_AUTHORITIES["selection_model_path"]
    )
    selection_evaluation = _REPOSITORY_ROOT / str(
        _OVERNIGHT_BASELINE_AUTHORITIES["selection_evaluation_path"]
    )
    ramp_receipt = _REPOSITORY_ROOT / str(
        _OVERNIGHT_BASELINE_AUTHORITIES["ramp_smoke_receipt_path"]
    )
    stairs_receipt = _REPOSITORY_ROOT / str(
        _OVERNIGHT_BASELINE_AUTHORITIES["stairs_smoke_receipt_path"]
    )
    return {
        **_baseline_upstream_authority(corpus, "primary_cache", "primary_cache"),
        **_baseline_upstream_authority(corpus, "pfnn_supplement", "pfnn_supplement"),
        "corpus_path": _path_identity(corpus_root),
        "corpus_manifest_sha256": _observed_file_sha256(corpus_root / "manifest.json"),
        "model_path": _path_identity(Path(model_root))
        if model_root is not None
        else None,
        "model_manifest_sha256": (
            _observed_file_sha256(Path(model_root) / "manifest.json")
            if model_root is not None
            else None
        ),
        "selection_model_path": _path_identity(selection_root),
        "selection_model_manifest_sha256": _observed_file_sha256(
            selection_root / "manifest.json"
        ),
        "selection_evaluation_path": _path_identity(selection_evaluation),
        "selection_evaluation_sha256": _observed_file_sha256(selection_evaluation),
        "ramp_smoke_receipt_path": _path_identity(ramp_receipt),
        "ramp_smoke_receipt_sha256": _observed_file_sha256(ramp_receipt),
        "stairs_smoke_receipt_path": _path_identity(stairs_receipt),
        "stairs_smoke_receipt_sha256": _observed_file_sha256(stairs_receipt),
    }


def _json_file_identity(path: Path) -> tuple[Mapping[str, Any], str | None]:
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            return {}, None
        payload = resolved.read_bytes()
        value = json.loads(payload)
        if type(value) is not dict:
            return {}, None
        return value, hashlib.sha256(payload).hexdigest()
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError, ValueError):
        return {}, None


def _determinism_receipt_authority(
    corpus_root: Path, corpus_manifest: Mapping[str, Any], corpus_sha256: object
) -> dict[str, object]:
    path = _REPOSITORY_ROOT / _DETERMINISM_RECEIPT_PATH
    receipt: Mapping[str, Any] = {}
    payload: bytes | None = None
    digest: str | None = None
    try:
        resolved = path.resolve(strict=True)
        if resolved.is_file():
            payload = resolved.read_bytes()
            decoded = json.loads(payload)
            if type(decoded) is dict:
                receipt = decoded
            digest = hashlib.sha256(payload).hexdigest()
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError, ValueError):
        pass
    manifest_path = corpus_root / "manifest.json"
    try:
        manifest_size = manifest_path.resolve(strict=True).stat().st_size
    except OSError:
        manifest_size = None
    expected_members = dict(_mapping(corpus_manifest.get("members")))
    expected_members["manifest.json"] = {
        "path": "manifest.json",
        "sha256": corpus_sha256,
        "size_bytes": manifest_size,
    }
    canonical = False
    if receipt and payload is not None:
        try:
            canonical = payload == canonical_json_bytes(dict(receipt))
        except (TypeError, ValueError):
            canonical = False
    current = (
        canonical
        and set(receipt)
        == {
            "schema",
            "status",
            "reference_manifest_sha256",
            "reproduced_manifest_sha256",
            "members",
        }
        and receipt.get("schema") == _DETERMINISM_RECEIPT_SCHEMA
        and receipt.get("status") == "accepted"
        and receipt.get("reference_manifest_sha256") == corpus_sha256
        and receipt.get("reproduced_manifest_sha256") == corpus_sha256
        and receipt.get("members") == expected_members
    )
    return {
        "determinism_receipt_path": _path_identity(path),
        "determinism_receipt_schema": receipt.get("schema"),
        "determinism_receipt_sha256": digest,
        "determinism_receipt_current": current,
    }


def _full_corpus_authority(corpus: object) -> dict[str, object]:
    root_value = getattr(corpus, "root", getattr(corpus, "source_root", None))
    if root_value is None:
        return {"corpus_manifest_authority_current": False}
    manifest, observed_sha = _json_file_identity(Path(root_value) / "manifest.json")
    published_sha = getattr(corpus, "manifest_sha256", None)
    source_names = tuple(str(value) for value in getattr(corpus, "source_names", ()))
    current = (
        isinstance(observed_sha, str)
        and observed_sha == published_sha
        and manifest.get("fps") == getattr(corpus, "fps", None)
        and tuple(_mapping(manifest).get("horizons", ()))
        == tuple(getattr(corpus, "horizons", ()))
        and manifest.get("sources") == len(source_names)
    )
    return {
        **_determinism_receipt_authority(Path(root_value), manifest, observed_sha),
        "corpus_schema": manifest.get("schema"),
        "fps": manifest.get("fps"),
        "horizons": manifest.get("horizons"),
        "source_identity_count": manifest.get("sources"),
        "inventory_terminal_count": manifest.get("sources"),
        "pfnn_bvh_count": sum("pfnn" in name.lower() for name in source_names),
        "corpus_row_count": manifest.get("rows"),
        "corpus_range_count": manifest.get("ranges"),
        "corpus_eligible_row_count": manifest.get("eligible_rows"),
        "corpus_manifest_sha256": observed_sha,
        "corpus_manifest_path": _path_identity(Path(root_value) / "manifest.json"),
        "corpus_manifest_authority_current": current,
    }


def _full_model_authority(
    generator: object, corpus_sha256: object
) -> dict[str, object]:
    root = getattr(generator, "root", None)
    loaded_manifest = _mapping(getattr(generator, "manifest", {}))
    if root is None:
        manifest, observed_sha = loaded_manifest, None
    else:
        manifest, observed_sha = _json_file_identity(Path(root) / "manifest.json")
    published_sha = getattr(generator, "manifest_sha256", None)
    mapping_current = bool(manifest) and dict(manifest) == dict(loaded_manifest)
    digest_current = isinstance(observed_sha, str) and (
        published_sha is None or published_sha == observed_sha
    )
    artifact_identity = _mapping(manifest.get("artifact_identity"))
    model_schema = artifact_identity.get("schema", manifest.get("schema"))
    corpus_current = (
        manifest.get("corpus_manifest_sha256") == corpus_sha256
        and artifact_identity.get("corpus_manifest_sha256") == corpus_sha256
    )
    return {
        "model_schema": model_schema,
        "model_manifest_sha256": observed_sha,
        "model_manifest_path": _path_identity(Path(root) / "manifest.json")
        if root is not None
        else None,
        "model_manifest_authority_current": mapping_current and digest_current,
        "model_corpus_binding_current": corpus_current,
    }


def _full_runtime_label(
    matcher: HybridMatcher,
    terrain: object | None = None,
    *,
    scene_authentication_current: bool | None = None,
    search_acceptance_current: bool | None = None,
    formal_authorities_current: bool | None = None,
) -> str:
    def with_controller_label(label: str) -> str:
        controller_label = _controller_diagnostic_label(matcher)
        return f"{label} | {controller_label}" if controller_label else label

    if getattr(matcher, "diagnostic_stability", False):
        pose_policy = (
            "CANONICAL SOURCE POSES; "
            if getattr(matcher, "diagnostic_canonical_source_pose", False)
            else ""
        )
        pose_policy += (
            "PoseInertializer + source+proximity-acquire/source-continue "
            "strict-final G1TerrainFootLock; "
            "0.10s half-life; "
        )
        bounds = tuple(getattr(matcher, "diagnostic_mechanical_clearance_bounds_m", ()))
        retained = getattr(
            matcher,
            "diagnostic_mechanical_retained_searchable_row_count",
            None,
        )
        total = getattr(matcher, "total_searchable_row_count", None)
        if len(bounds) == 2 and retained is not None and total is not None:
            return with_controller_label(
                "FULL WALKING TERRAIN LMM "
                f"(DIAGNOSTIC {pose_policy}MECHANICALLY FILTERED "
                f"{float(bounds[0]):.3f}.."
                f"{float(bounds[1]):.3f} M; {int(retained)}/{int(total)} "
                "RANGE-SAFE ROWS; NOT ACCEPTANCE EVIDENCE)"
            )
        return with_controller_label(
            "FULL WALKING TERRAIN LMM "
            f"(DIAGNOSTIC {pose_policy}MECHANICALLY FILTERED SEARCH; "
            "NOT ACCEPTANCE EVIDENCE)"
        )
    if (
        getattr(matcher, "search_backend_identity", "cpu-ckdtree-exact")
        != "cpu-ckdtree-exact"
    ):
        return FULL_DIAGNOSTIC_LABEL
    search_eligible = (
        matcher.search_acceptance_eligible
        if search_acceptance_current is None
        else search_acceptance_current
    )
    if not search_eligible:
        return FULL_DIAGNOSTIC_LABEL
    if not _generator_acceptance_status(getattr(matcher, "generator", None))[
        "accepted"
    ]:
        return FULL_DIAGNOSTIC_MODEL_LABEL
    if formal_authorities_current is False:
        return FULL_DIAGNOSTIC_MODEL_LABEL
    if terrain is not None and (
        not getattr(terrain, "scene_authenticated", False)
        or scene_authentication_current is False
    ):
        return FULL_DIAGNOSTIC_TERRAIN_LABEL
    return FULL_LABEL


def _full_runtime_step_hz(matcher: object) -> float:
    rate = getattr(matcher, "fps", None)
    if rate is None:
        raise ValueError(
            "full walking interactive viewer requires authenticated 60 Hz runtime FPS"
        )
    value = float(rate)
    if not math.isfinite(value) or value != 60.0:
        raise ValueError(
            "full walking interactive viewer requires authenticated 60 Hz runtime FPS"
        )
    return value


def acceptance_failures(receipt: Mapping[str, object]) -> list[str]:
    """Return every failed formal gate; malformed or absent evidence fails closed."""

    root = _mapping(receipt)
    candidate = _mapping(root.get("candidate"))
    baseline = _mapping(root.get("baseline"))
    comparison = _mapping(root.get("comparison"))
    identity = _mapping(root.get("formal_identity"))
    baseline_authority_pre = _mapping(identity.get("baseline_authority_pre"))
    baseline_authority_post = _mapping(identity.get("baseline_authority_post"))
    failures: list[str] = []

    def gate(passed: bool, name: str) -> None:
        if not passed:
            failures.append(name)

    gate(
        identity.get("corpus_schema") == "g1-full-walking-terrain-lmm-corpus/v1",
        "full-corpus-required",
    )
    gate(
        identity.get("model_schema") == "g1-full-walking-terrain-lmm-model/v1",
        "full-all-row-model-required",
    )
    gate(
        _full_formal_artifact_authorities(identity),
        "full-formal-artifact-authorities-required",
    )
    gate(identity.get("fps") == 60.0, "60hz-candidate-required")
    gate(identity.get("horizons") == [20, 40, 60], "60hz-horizons-required")
    gate(
        identity.get("source_identity_count") == 15_918,
        "full-source-inventory-required",
    )
    gate(
        identity.get("inventory_terminal_count") == 15_918,
        "terminal-source-receipts-required",
    )
    gate(identity.get("pfnn_bvh_count") == 80, "full-pfnn-required")
    for field, name in (
        ("split_receipt_current", "split-receipt-required"),
        ("test_receipt_current", "one-time-test-receipt-required"),
        ("refit_receipt_current", "all-row-refit-receipt-required"),
        ("determinism_receipt_current", "deterministic-corpus-receipt-required"),
        ("full_search", "full-search-required"),
        ("motion_root_ownership", "motion-derived-root-required"),
        ("supported_speed_envelope_current", "supported-speed-envelope-required"),
    ):
        gate(identity.get(field) is True, name)
    search_backend_identities = _mapping(identity.get("search_backend_identities"))
    gate(
        identity.get("cpu_search_backend_only") is True
        and set(search_backend_identities) == set(FORMAL_SCENE_IDS)
        and all(
            type(backend) is str and backend == "cpu-ckdtree-exact"
            for backend in search_backend_identities.values()
        ),
        "cpu-exact-search-backend-required",
    )
    retry_budget = identity.get("retry_budget")
    gate(type(retry_budget) is int and retry_budget > 0, "finite-retry-budget-required")
    gate(identity.get("g1_xml_sha256") == _G1_XML_SHA256, "canonical-g1-xml-required")
    gate(
        identity.get("g1_asset_inventory_sha256") == _G1_ASSET_INVENTORY_SHA256,
        "canonical-g1-assets-required",
    )
    authenticated = identity.get("authenticated_scene_ids")
    gate(
        isinstance(authenticated, list) and set(authenticated) == set(FORMAL_SCENE_IDS),
        "four-authenticated-scenes-required",
    )
    gate(
        _baseline_authority_exact(baseline_authority_pre)
        and _baseline_authority_exact(baseline_authority_post),
        "frozen-baseline-provenance-required",
    )
    gate(
        identity.get("baseline_authority_current_pre") is True
        and identity.get("baseline_authority_current_post") is True,
        "baseline-authority-current-required",
    )
    gate(
        baseline_authority_pre == baseline_authority_post,
        "baseline-authority-unchanged-required",
    )
    gate(
        root.get("route_command_authority_sha256")
        == baseline.get("route_command_authority_sha256")
        == candidate.get("route_command_authority_sha256")
        and isinstance(root.get("route_command_authority_sha256"), str),
        "identical-route-command-authority-required",
    )
    gate(
        type(candidate.get("forward_count")) is int
        and int(candidate["forward_count"]) >= 10_000,
        "minimum-10000-candidate-forwards",
    )
    safety = _mapping(candidate.get("safety_counts"))
    for name in _SAFETY_KEYS:
        gate(safety.get(name) == 0, f"zero-{name.replace('_', '-')}-required")
    query_reduction = comparison.get("query_distance_p95_reduction_fraction")
    slip_reduction = comparison.get("slip_p95_reduction_fraction")
    gate(
        isinstance(query_reduction, (int, float)) and query_reduction >= 0.30,
        "query-distance-p95-improvement-30pct",
    )
    gate(
        isinstance(slip_reduction, (int, float)) and slip_reduction >= 0.50,
        "slip-p95-improvement-50pct",
    )
    slip = _mapping(candidate.get("slip"))
    median = slip.get("median_mps")
    p95 = slip.get("p95_mps")
    speed_mae = candidate.get("speed_mae_mps")
    gate(
        isinstance(median, (int, float)) and math.isfinite(median) and median <= 0.02,
        "slip-median-at-most-0.02mps",
    )
    gate(
        isinstance(p95, (int, float)) and math.isfinite(p95) and p95 <= 0.08,
        "slip-p95-at-most-0.08mps",
    )
    gate(
        isinstance(speed_mae, (int, float))
        and math.isfinite(speed_mae)
        and speed_mae <= 0.12,
        "speed-mae-at-most-0.12mps",
    )
    by_terrain = _mapping(candidate.get("by_terrain"))
    for terrain in _TERRAINS:
        terrain_result = _mapping(by_terrain.get(terrain))
        gate(
            type(terrain_result.get("canonical_identity_count")) is int
            and terrain_result["canonical_identity_count"] >= 2,
            f"{terrain}-two-canonical-identities-required",
        )
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--corpus", type=Path, required=True)
    smoke.add_argument("--model", type=Path, required=True)
    smoke.add_argument("--baseline-corpus", type=Path, required=True)
    smoke.add_argument("--baseline-model", type=Path, required=True)
    smoke.add_argument("--scene", dest="scenes", action="append", required=True)
    smoke.add_argument("--frames-per-scene", type=int, default=2_500)
    smoke.add_argument("--g1-xml", type=Path, default=DEFAULT_G1_XML)
    smoke.add_argument("--receipt", type=Path)
    view = commands.add_parser("view")
    view.add_argument("--corpus", type=Path, required=True)
    view.add_argument("--model", type=Path, required=True)
    view.add_argument("--scene", default="ramp-10-up-down")
    view.add_argument("--g1-xml", type=Path, default=DEFAULT_G1_XML)
    view.add_argument("--gamepad")
    view.add_argument("--max-render-frames", type=int, default=0)
    view.add_argument("--search-device")
    view.add_argument("--receipt", type=Path)
    return parser


def _counter(state: object, *names: str) -> int:
    for name in names:
        value = getattr(state, name, None)
        if type(value) is int:
            return value
    return 0


def _canonical_source_identity(corpus: object, state: object) -> str:
    """Use full-corpus source IDs and a stable range ID for the legacy baseline."""

    row = int(state.row)
    source_ids = np.asarray(getattr(corpus, "source_ids", ()))
    source_names = tuple(str(value) for value in getattr(corpus, "source_names", ()))
    if source_ids.shape == (len(corpus.artifacts.positions),):
        source_index = int(source_ids[row])
        if 0 <= source_index < len(source_names):
            return source_names[source_index]
        raise ValueError(
            "full corpus source ID is outside the authenticated source map"
        )
    range_ids = np.asarray(getattr(corpus, "range_ids", ()))
    if range_ids.shape == (len(corpus.artifacts.positions),):
        return f"legacy-range-{int(range_ids[row])}"
    return f"legacy-row-{row}"


_DIAGNOSTIC_SCENE_FAMILIES = {
    "ramp-10-up-down": "slope",
    "grail-curb-default": "curb",
    "stairs-standard": "stair",
    "flat-standard": "flat",
}
_PFNN_TERRAIN_SEMANTICS = frozenset(("flat", "rocky", "jumpy", "beam"))


def _diagnostic_scene_row_mask(corpus: object, adapter: object) -> np.ndarray | None:
    """Return the authenticated scene's diagnostic-only source eligibility mask."""

    if (
        not bool(getattr(adapter, "scene_authenticated", False))
        or getattr(adapter, "scene_evidence_status", None) != "authenticated-indexed"
    ):
        return None
    scene_id = getattr(adapter, "scene_id", None)
    family_name = _DIAGNOSTIC_SCENE_FAMILIES.get(scene_id)
    if family_name is None:
        raise ValueError(f"scene family mapping is missing for {scene_id!r}")
    corpus_family_names = getattr(corpus, "family_names", None)
    if corpus_family_names is not None and tuple(corpus_family_names) != _TERRAINS:
        raise ValueError("scene family mapping does not match corpus families")
    family_ids = np.asarray(getattr(corpus, "family_ids", ()))
    source_ids = np.asarray(getattr(corpus, "source_ids", ()))
    source_names = tuple(str(value) for value in getattr(corpus, "source_names", ()))
    if (
        family_ids.shape != (len(source_ids),)
        or source_ids.ndim != 1
        or not len(source_ids)
        or not np.issubdtype(family_ids.dtype, np.integer)
        or not np.issubdtype(source_ids.dtype, np.integer)
        or np.any(family_ids < 0)
        or np.any(family_ids >= len(_TERRAINS))
        or source_ids.min(initial=0) < 0
        or source_ids.max(initial=-1) >= len(source_names)
    ):
        raise ValueError("scene family mapping corpus row metadata is invalid")
    family_index = _TERRAINS.index(family_name)
    if family_name != "flat":
        mask = family_ids == family_index
    else:
        root = getattr(corpus, "root", None)
        inventory_sha = getattr(corpus, "inventory_manifest_sha256", None)
        if not isinstance(root, Path) or not isinstance(inventory_sha, str):
            raise ValueError("flat scene inventory authority is unavailable")
        inventory = load_inventory(
            root / "inventory.json", expected_manifest_sha256=inventory_sha
        )
        records = tuple(inventory.sources)
        inventory_names = tuple(sorted(str(record.source_id) for record in records))
        if inventory_names != tuple(sorted(source_names)):
            raise ValueError("flat scene inventory source mapping changed")
        allowed_source_ids: set[str] = set()
        for record in records:
            authority = record.authority
            kind = authority.get("kind") if isinstance(authority, Mapping) else None
            if kind == "takara":
                if record.family != "flat":
                    raise ValueError("flat scene Takara family mapping changed")
                allowed_source_ids.add(record.source_id)
            elif kind == "pfnn":
                semantics = (
                    authority.get("terrain_semantics")
                    if isinstance(authority, Mapping)
                    else None
                )
                if semantics not in _PFNN_TERRAIN_SEMANTICS:
                    raise ValueError("flat scene PFNN terrain semantics changed")
                if semantics == "flat":
                    if record.family != "flat":
                        raise ValueError("flat scene PFNN family mapping changed")
                    allowed_source_ids.add(record.source_id)
        if not allowed_source_ids:
            raise ValueError("flat scene inventory has no eligible sources")
        mask = np.isin(
            np.asarray(source_names, dtype=object)[source_ids],
            tuple(sorted(allowed_source_ids)),
        )
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != family_ids.shape or not np.any(mask):
        raise ValueError("scene family mapping authorized no corpus rows")
    return mask


class _MuJoCoRouteEvaluator:
    """Adapter that turns matcher factories into post-forward evaluation series."""

    def __init__(
        self,
        corpus: object,
        generator: object,
        *,
        g1_xml: Path,
        fps: float,
        scene_corpus: object | None = None,
    ) -> None:
        self.corpus = corpus
        self.scene_corpus = corpus if scene_corpus is None else scene_corpus
        self.generator = generator
        self.g1_xml = Path(g1_xml)
        self.fps = float(fps)
        self.matchers: dict[str, HybridMatcher] = {}
        self.adapters: dict[str, object] = {}

    def _matcher(self, scene_id: str) -> tuple[HybridMatcher, object, object]:
        import mujoco

        adapter = load_scene_terrain(scene_id, corpus=self.scene_corpus)
        native_model = mujoco.MjModel.from_xml_path(str(self.g1_xml))
        matcher = HybridMatcher(
            self.corpus,
            self.generator,
            adapter.authority,
            native_model=native_model,
            initial_root_xy=adapter.spawn_native_xy,
            initial_heading=adapter.spawn_heading,
            diagnostic_stability=False,
            diagnostic_canonical_source_pose=False,
        )
        model = build_viewer_model(self.g1_xml, adapter)
        self.matchers[scene_id] = matcher
        self.adapters[scene_id] = adapter
        return matcher, adapter, model

    def evaluate_formal_route(
        self,
        route: FormalRoute,
        sample_times: np.ndarray,
        command_speed: np.ndarray,
        command_steering: np.ndarray,
        *,
        g1_xml: Path,
    ) -> EvaluationSeries:
        if Path(g1_xml).resolve() != self.g1_xml.resolve():
            raise ValueError("route evaluator G1 XML authority changed")
        import mujoco

        from .terrain_oracle.contact import SoleGeometry

        matcher, _adapter, model = self._matcher(route.scene_id)
        data = mujoco.MjData(model)
        geometry = SoleGeometry.from_model(model)
        body_ids = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in geometry.body_names
        )
        frames = len(sample_times)
        probes = np.empty((frames, 8, 3), dtype=np.float64)
        contacts = np.empty((frames, 8), dtype=np.bool_)
        roots = np.empty((frames, 2), dtype=np.float64)
        desired = np.empty(frames, dtype=np.float64)
        distances = np.empty(frames, dtype=np.float64)
        source_ids: list[str] = []
        initial = matcher.state
        prior_counters = {
            "candidate_exhaustion": _counter(
                initial, "candidate_exhaustion_count", "exhaustion_count"
            ),
            "fallback": _counter(initial, "fallback_count"),
            "joint_clamp": _counter(initial, "joint_clamp_count"),
            "out_of_range_successor": _counter(initial, "out_of_range_successor_count"),
        }
        native_violations = 0
        nonfinite = 0
        for frame in range(frames):
            if frame == 0:
                state = matcher.reset()
            else:
                state = matcher.step(
                    CommandState(command_speed[frame], command_steering[frame]),
                    dt=1.0 / self.fps,
                )
            qpos = np.asarray(state.qpos, dtype=np.float64)
            nonfinite += int(
                qpos.shape != (int(model.nq),) or not np.isfinite(qpos).all()
            )
            if nonfinite:
                raise ValueError("formal route produced nonfinite or wrong-sized qpos")
            for joint_id in range(1, int(model.njnt)):
                address = int(model.jnt_qposadr[joint_id])
                lower, upper = model.jnt_range[joint_id]
                native_violations += int(
                    qpos[address] < lower - 1e-5 or qpos[address] > upper + 1e-5
                )
            data.qpos[:] = qpos
            data.time = float(sample_times[frame])
            mujoco.mj_forward(model, data)
            for foot, body_id in enumerate(body_ids):
                rotation = data.xmat[body_id].reshape(3, 3)
                probes[frame, foot * 4 : (foot + 1) * 4] = (
                    geometry.corner_positions_body[foot] @ rotation.T
                    + data.xpos[body_id]
                )
            row_contacts = np.asarray(
                getattr(state, "contacts", matcher.artifacts.contacts[int(state.row)]),
                dtype=np.bool_,
            ).reshape(2)
            contacts[frame] = np.repeat(row_contacts, 4)
            roots[frame] = np.asarray(state.root_position_world, dtype=np.float64)[:2]
            envelope = getattr(
                matcher,
                "walking_speed_p95_mps",
                getattr(matcher, "supported_speed_envelope_mps", (-0.45, 0.45)),
            )
            if np.asarray(envelope).ndim == 0:
                max_speed = abs(float(envelope))
            else:
                max_speed = max(abs(float(value)) for value in envelope)
            desired[frame] = abs(
                float(
                    getattr(
                        state,
                        "desired_speed_mps",
                        float(command_speed[frame]) * max_speed,
                    )
                )
            )
            distances[frame] = float(state.search_distance)
            source_ids.append(_canonical_source_identity(self.corpus, state))
        final = matcher.state
        safety = {
            "candidate_exhaustion": _counter(
                final, "candidate_exhaustion_count", "exhaustion_count"
            )
            - prior_counters["candidate_exhaustion"],
            "fallback": _counter(final, "fallback_count") - prior_counters["fallback"],
            "joint_clamp": _counter(final, "joint_clamp_count")
            - prior_counters["joint_clamp"],
            "native_limit_violation": native_violations,
            "nonfinite": nonfinite,
            "out_of_range_successor": _counter(final, "out_of_range_successor_count")
            - prior_counters["out_of_range_successor"],
        }
        return EvaluationSeries(
            probes=probes,
            contacts=contacts,
            root_xy=roots,
            desired_speed_mps=desired,
            query_distances=distances,
            canonical_source_ids=tuple(source_ids),
            forward_count=frames,
            safety_counts=safety,
        )


def _load_full_corpus(path: Path) -> object:
    module = importlib.import_module("mm_sonic.full_walking_terrain_lmm_corpus")
    return module.load_full_corpus(path)


def _load_baseline_corpus(path: Path) -> object:
    module = importlib.import_module("mm_sonic.hybrid_terrain_lmm_combined")
    return module.load_combined_cache(path)


def _manifest(generator: object) -> Mapping[str, Any]:
    return _mapping(getattr(generator, "manifest", {}))


def _formal_identity(
    evaluator: _MuJoCoRouteEvaluator, *, scenes: Sequence[str], g1_xml: Path
) -> dict[str, object]:
    corpus = evaluator.corpus
    corpus_authority = _full_corpus_authority(corpus)
    model_authority = _full_model_authority(
        evaluator.generator, corpus_authority.get("corpus_manifest_sha256")
    )
    asset_sha, _assets = _g1_xml_asset_identity(g1_xml)
    matchers = tuple(evaluator.matchers.values())
    full_search = bool(matchers) and all(
        matcher.search_scope == "full-range-safe-corpus"
        and len(matcher.searchable_rows) == matcher.total_searchable_row_count
        for matcher in matchers
    )
    search_backend_identities = {
        str(scene_id): getattr(matcher, "search_backend_identity", None)
        for scene_id, matcher in sorted(
            evaluator.matchers.items(), key=lambda item: str(item[0])
        )
    }
    cpu_search_backend_only = bool(search_backend_identities) and all(
        type(backend) is str and backend == "cpu-ckdtree-exact"
        for backend in search_backend_identities.values()
    )
    return {
        **corpus_authority,
        **model_authority,
        "split_receipt_current": corpus_authority.get(
            "corpus_manifest_authority_current"
        )
        is True,
        "test_receipt_current": getattr(
            evaluator.generator, "test_receipt_current", False
        )
        is True,
        "refit_receipt_current": getattr(
            evaluator.generator, "refit_receipt_current", False
        )
        is True,
        "determinism_receipt_current": corpus_authority.get(
            "determinism_receipt_current"
        )
        is True,
        "full_search": full_search,
        "search_backend_identities": search_backend_identities,
        "cpu_search_backend_only": cpu_search_backend_only,
        "motion_root_ownership": bool(matchers)
        and all(
            getattr(matcher.state, "root_motion_source", None)
            == "canonical-simulation-se2"
            for matcher in matchers
        ),
        "supported_speed_envelope_current": bool(matchers)
        and all(
            math.isfinite(float(getattr(matcher, "walking_speed_p95_mps", math.nan)))
            and float(getattr(matcher, "walking_speed_p95_mps", 0.0)) > 0.0
            for matcher in matchers
        ),
        "retry_budget": min(
            (
                int(getattr(matcher, "candidate_retry_budget", 0))
                for matcher in matchers
            ),
            default=0,
        ),
        "g1_xml_sha256": hashlib.sha256(Path(g1_xml).read_bytes()).hexdigest(),
        "g1_asset_inventory_sha256": asset_sha,
        "authenticated_scene_ids": sorted(
            scene
            for scene in scenes
            if getattr(evaluator.adapters.get(scene), "scene_evidence_status", None)
            == "authenticated-indexed"
        ),
    }


def _full_runtime_identity(
    matcher: HybridMatcher, terrain: object, g1_xml: Path
) -> dict[str, object]:
    evaluator = type(
        "_InteractiveAuthority",
        (),
        {
            "corpus": matcher.corpus,
            "generator": matcher.generator,
            "matchers": {str(getattr(terrain, "scene_id", "scene")): matcher},
            "adapters": {str(getattr(terrain, "scene_id", "scene")): terrain},
        },
    )()
    identity = dict(
        _formal_identity(
            evaluator,
            scenes=(str(getattr(terrain, "scene_id", "scene")),),
            g1_xml=Path(g1_xml),
        )
    )
    identity.update(
        {
            "identity_capture_succeeded": True,
            "scene_id": getattr(terrain, "scene_id", None),
            "scene_evidence_status": getattr(terrain, "scene_evidence_status", None),
            "scene_authentication_current": scene_authentication_is_current(
                matcher.corpus, terrain
            ),
            "generator_acceptance_status": _generator_acceptance_status(
                matcher.generator
            ),
            "search_scope": matcher.search_scope,
            "searched_row_count": len(matcher.searchable_rows),
            "total_safe_row_count": matcher.total_searchable_row_count,
            "search_backend_identity": getattr(
                matcher, "search_backend_identity", "cpu-ckdtree-exact"
            ),
            "last_search_elapsed_ms": getattr(matcher, "last_search_elapsed_ms", None),
            "first_runtime_search_elapsed_ms": getattr(
                matcher, "warm_search_elapsed_ms", None
            ),
            "controller_identity": getattr(matcher, "controller_identity", None),
            "controller_heading_policy": getattr(
                matcher, "controller_heading_policy", None
            ),
            "controller_yaw_rate_rad_s": getattr(
                matcher, "controller_yaw_rate_rad_s", None
            ),
            "controller_acceleration_mps2": getattr(
                matcher, "controller_acceleration_mps2", None
            ),
            "controller_deceleration_mps2": getattr(
                matcher, "controller_deceleration_mps2", None
            ),
            "controller_stop_speed_mps": getattr(
                matcher, "controller_stop_speed_mps", None
            ),
            "controller_search_interval_s": getattr(
                matcher, "controller_search_interval_s", None
            ),
            "diagnostic_stability": getattr(matcher, "diagnostic_stability", False),
            "diagnostic_canonical_source_pose": bool(
                getattr(matcher, "diagnostic_canonical_source_pose", False)
            ),
            "diagnostic_authorized_row_count": getattr(
                matcher, "diagnostic_authorized_row_count", None
            ),
            "diagnostic_authorized_searchable_row_count": getattr(
                matcher, "diagnostic_authorized_searchable_row_count", None
            ),
            "diagnostic_retained_row_count": getattr(
                matcher, "diagnostic_retained_row_count", None
            ),
            "diagnostic_retained_searchable_row_count": getattr(
                matcher, "diagnostic_retained_searchable_row_count", None
            ),
            "diagnostic_mechanical_clearance_bounds_m": list(
                getattr(matcher, "diagnostic_mechanical_clearance_bounds_m", ())
            ),
            "diagnostic_mechanical_retained_row_count": getattr(
                matcher, "diagnostic_mechanical_retained_row_count", None
            ),
            "diagnostic_mechanical_rejected_row_count": getattr(
                matcher, "diagnostic_mechanical_rejected_row_count", None
            ),
            "diagnostic_mechanical_retained_searchable_row_count": getattr(
                matcher,
                "diagnostic_mechanical_retained_searchable_row_count",
                None,
            ),
            "diagnostic_mechanical_rejected_searchable_row_count": getattr(
                matcher,
                "diagnostic_mechanical_rejected_searchable_row_count",
                None,
            ),
            "diagnostic_mechanical_successor_clearance_limit_m": getattr(
                matcher,
                "diagnostic_mechanical_successor_clearance_limit_m",
                None,
            ),
            "diagnostic_mechanical_discontinuous_successor_edge_count": getattr(
                matcher,
                "diagnostic_mechanical_discontinuous_successor_edge_count",
                None,
            ),
        }
    )
    return identity


def _default_routes(
    scenes: Sequence[str], frames_per_scene: int
) -> tuple[FormalRoute, ...]:
    if type(frames_per_scene) is not int or frames_per_scene < 2:
        raise ValueError("formal frames per scene must be at least two")
    duration = (frames_per_scene - 1) / 60.0
    times = np.asarray((0.0, 0.12, 0.36, 0.52, 0.68, 0.78, 0.90)) * duration
    return tuple(
        FormalRoute(
            scene_id=scene,
            duration_seconds=duration,
            command_times=times,
            command_speed=np.asarray((0.0, 1.0, 1.0, 1.0, 0.0, -0.4, 0.8)),
            command_steering=np.asarray((0.0, 0.0, 0.65, -0.65, 0.0, 0.0, 0.0)),
        )
        for scene in scenes
    )


def run_formal_smoke(
    baseline: _MuJoCoRouteEvaluator,
    candidate: _MuJoCoRouteEvaluator,
    *,
    routes: Sequence[FormalRoute],
    g1_xml: Path,
) -> dict[str, object]:
    baseline_authority_pre = _baseline_authority_snapshot(baseline)
    receipt = dict(
        compare_baseline_candidate(
            baseline=baseline, candidate=candidate, routes=routes, g1_xml=g1_xml
        )
    )
    baseline_authority_post = _baseline_authority_snapshot(baseline)
    receipt["schema"] = "g1-full-walking-terrain-lmm-formal-evidence/v1"
    receipt["formal_identity"] = {
        **_formal_identity(
            candidate, scenes=[route.scene_id for route in routes], g1_xml=g1_xml
        ),
        "baseline_authority_pre": baseline_authority_pre,
        "baseline_authority_post": baseline_authority_post,
        "baseline_authority_current_pre": _baseline_authority_exact(
            baseline_authority_pre
        ),
        "baseline_authority_current_post": _baseline_authority_exact(
            baseline_authority_post
        ),
    }
    failures = acceptance_failures(receipt)
    receipt["acceptance_failures"] = failures
    receipt["accepted"] = not failures
    return receipt


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "view" and arguments.search_device is not None:
        configure_single_gpu_visibility(arguments.search_device)
    corpus = _load_full_corpus(arguments.corpus)
    generator = _load_generator(arguments.model, corpus)
    if arguments.command == "smoke":
        if tuple(arguments.scenes) != FORMAL_SCENE_IDS:
            raise ValueError("formal smoke requires the four canonical scenes in order")
        baseline_corpus = _load_baseline_corpus(arguments.baseline_corpus)
        baseline_generator = _load_generator(arguments.baseline_model, baseline_corpus)
        baseline = _MuJoCoRouteEvaluator(
            baseline_corpus,
            baseline_generator,
            g1_xml=arguments.g1_xml,
            fps=25.0,
            scene_corpus=corpus,
        )
        candidate = _MuJoCoRouteEvaluator(
            corpus, generator, g1_xml=arguments.g1_xml, fps=60.0
        )
        receipt = run_formal_smoke(
            baseline,
            candidate,
            routes=_default_routes(arguments.scenes, arguments.frames_per_scene),
            g1_xml=arguments.g1_xml,
        )
    else:
        import mujoco

        adapter = load_scene_terrain(arguments.scene, corpus=corpus)
        native_model = mujoco.MjModel.from_xml_path(str(arguments.g1_xml))
        diagnostic_row_mask = _diagnostic_scene_row_mask(corpus, adapter)
        matcher = HybridMatcher(
            corpus,
            generator,
            adapter.authority,
            native_model=native_model,
            search_device=arguments.search_device,
            initial_root_xy=adapter.spawn_native_xy,
            initial_heading=adapter.spawn_heading,
            diagnostic_stability=True,
            diagnostic_canonical_source_pose=True,
            diagnostic_row_mask=diagnostic_row_mask,
        )

        def display_postprocessor_factory(
            _render_model: object, matcher: object, terrain: object
        ) -> ExistingUtilityPosePostprocessor:
            collision_model = build_diagnostic_collision_model(
                arguments.g1_xml, terrain
            )
            scene = SimpleNamespace(height_at_world_xy=terrain.authority.height_at)
            return ExistingUtilityPosePostprocessor(
                collision_model,
                scene,
                fps=_full_runtime_step_hz(matcher),
                inertialization_halflife_s=0.10,
            )

        receipt = run_interactive(
            matcher,
            adapter,
            g1_xml=arguments.g1_xml,
            gamepad=arguments.gamepad,
            max_render_frames=arguments.max_render_frames,
            formal_authority_predicate=_full_formal_artifact_authorities,
            runtime_label_resolver=_full_runtime_label,
            runtime_step_hz_resolver=_full_runtime_step_hz,
            runtime_identity_resolver=_full_runtime_identity,
            display_postprocessor_factory=display_postprocessor_factory,
        )
    print(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False),
        flush=True,
    )
    if arguments.receipt is not None:
        write_receipt_exclusive(arguments.receipt, receipt)
    return 0 if bool(receipt.get("accepted", True)) else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "FORMAL_SCENE_IDS",
    "FULL_LABEL",
    "_full_formal_artifact_authorities",
    "_full_runtime_identity",
    "_full_runtime_step_hz",
    "acceptance_failures",
    "build_parser",
    "main",
    "run_formal_smoke",
)
