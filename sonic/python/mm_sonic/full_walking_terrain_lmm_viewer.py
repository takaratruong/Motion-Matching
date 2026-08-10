"""Formal MuJoCo evidence and interactive viewer for the full walking corpus."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .full_walking_terrain_lmm_evaluation import (
    EvaluationSeries,
    FormalRoute,
    compare_baseline_candidate,
)
from .hybrid_terrain_lmm_runtime import CommandState, HybridMatcher
from .hybrid_terrain_lmm_viewer import (
    DEFAULT_G1_XML,
    _g1_xml_asset_identity,
    _load_generator,
    build_viewer_model,
    load_scene_terrain,
    run_interactive,
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


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def acceptance_failures(receipt: Mapping[str, object]) -> list[str]:
    """Return every failed formal gate; malformed or absent evidence fails closed."""

    root = _mapping(receipt)
    candidate = _mapping(root.get("candidate"))
    baseline = _mapping(root.get("baseline"))
    comparison = _mapping(root.get("comparison"))
    identity = _mapping(root.get("formal_identity"))
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
    gate(identity.get("fps") == 60.0, "60hz-candidate-required")
    gate(identity.get("horizons") == [20, 40, 60], "60hz-horizons-required")
    gate(identity.get("source_identity_count") == 15_918, "full-source-inventory-required")
    gate(identity.get("inventory_terminal_count") == 15_918, "terminal-source-receipts-required")
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
    retry_budget = identity.get("retry_budget")
    gate(type(retry_budget) is int and retry_budget > 0, "finite-retry-budget-required")
    gate(identity.get("g1_xml_sha256") == _G1_XML_SHA256, "canonical-g1-xml-required")
    gate(
        identity.get("g1_asset_inventory_sha256") == _G1_ASSET_INVENTORY_SHA256,
        "canonical-g1-assets-required",
    )
    authenticated = identity.get("authenticated_scene_ids")
    gate(
        isinstance(authenticated, list)
        and set(authenticated) == set(FORMAL_SCENE_IDS),
        "four-authenticated-scenes-required",
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
    gate(isinstance(query_reduction, (int, float)) and query_reduction >= 0.30,
         "query-distance-p95-improvement-30pct")
    gate(isinstance(slip_reduction, (int, float)) and slip_reduction >= 0.50,
         "slip-p95-improvement-50pct")
    slip = _mapping(candidate.get("slip"))
    median = slip.get("median_mps")
    p95 = slip.get("p95_mps")
    speed_mae = candidate.get("speed_mae_mps")
    gate(isinstance(median, (int, float)) and math.isfinite(median) and median <= 0.02,
         "slip-median-at-most-0.02mps")
    gate(isinstance(p95, (int, float)) and math.isfinite(p95) and p95 <= 0.08,
         "slip-p95-at-most-0.08mps")
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
        raise ValueError("full corpus source ID is outside the authenticated source map")
    range_ids = np.asarray(getattr(corpus, "range_ids", ()))
    if range_ids.shape == (len(corpus.artifacts.positions),):
        return f"legacy-range-{int(range_ids[row])}"
    return f"legacy-row-{row}"


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
            "candidate_exhaustion": _counter(initial, "candidate_exhaustion_count", "exhaustion_count"),
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
            nonfinite += int(qpos.shape != (int(model.nq),) or not np.isfinite(qpos).all())
            if nonfinite:
                raise ValueError("formal route produced nonfinite or wrong-sized qpos")
            for joint_id in range(1, int(model.njnt)):
                address = int(model.jnt_qposadr[joint_id])
                lower, upper = model.jnt_range[joint_id]
                native_violations += int(qpos[address] < lower - 1e-5 or qpos[address] > upper + 1e-5)
            data.qpos[:] = qpos
            data.time = float(sample_times[frame])
            mujoco.mj_forward(model, data)
            for foot, body_id in enumerate(body_ids):
                rotation = data.xmat[body_id].reshape(3, 3)
                probes[frame, foot * 4:(foot + 1) * 4] = (
                    geometry.corner_positions_body[foot] @ rotation.T + data.xpos[body_id]
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
            desired[frame] = abs(float(getattr(
                state,
                "desired_speed_mps",
                float(command_speed[frame]) * max_speed,
            )))
            distances[frame] = float(state.search_distance)
            source_ids.append(_canonical_source_identity(self.corpus, state))
        final = matcher.state
        safety = {
            "candidate_exhaustion": _counter(final, "candidate_exhaustion_count", "exhaustion_count")
            - prior_counters["candidate_exhaustion"],
            "fallback": _counter(final, "fallback_count") - prior_counters["fallback"],
            "joint_clamp": _counter(final, "joint_clamp_count") - prior_counters["joint_clamp"],
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
    module = importlib.import_module("mm_sonic.hybrid_terrain_lmm_data")
    return module.load_hybrid_cache(path)


def _manifest(generator: object) -> Mapping[str, Any]:
    return _mapping(getattr(generator, "manifest", {}))


def _formal_identity(
    evaluator: _MuJoCoRouteEvaluator, *, scenes: Sequence[str], g1_xml: Path
) -> dict[str, object]:
    corpus = evaluator.corpus
    generator_manifest = _manifest(evaluator.generator)
    corpus_manifest = _mapping(getattr(corpus, "manifest", {}))
    asset_sha, _assets = _g1_xml_asset_identity(g1_xml)
    source_names = tuple(str(value) for value in getattr(corpus, "source_names", ()))
    pfnn_count = sum("pfnn" in name.lower() for name in source_names)
    matchers = tuple(evaluator.matchers.values())
    full_search = bool(matchers) and all(
        matcher.search_scope == "full-range-safe-corpus"
        and len(matcher.searchable_rows) == matcher.total_searchable_row_count
        for matcher in matchers
    )
    return {
        "corpus_schema": corpus_manifest.get("schema", getattr(corpus, "schema", None)),
        "model_schema": generator_manifest.get("schema"),
        "fps": float(getattr(corpus, "fps", math.nan)),
        "horizons": list(getattr(corpus, "horizons", ())),
        "source_identity_count": len(source_names),
        "inventory_terminal_count": corpus_manifest.get("inventory_terminal_count"),
        "pfnn_bvh_count": corpus_manifest.get("pfnn_bvh_count", pfnn_count),
        "split_receipt_current": getattr(corpus, "split_receipt_current", False) is True,
        "test_receipt_current": getattr(evaluator.generator, "test_receipt_current", False) is True,
        "refit_receipt_current": getattr(evaluator.generator, "refit_receipt_current", False) is True,
        "determinism_receipt_current": getattr(corpus, "determinism_receipt_current", False) is True,
        "full_search": full_search,
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
            (int(getattr(matcher, "candidate_retry_budget", 0)) for matcher in matchers),
            default=0,
        ),
        "g1_xml_sha256": __import__("hashlib").sha256(Path(g1_xml).read_bytes()).hexdigest(),
        "g1_asset_inventory_sha256": asset_sha,
        "authenticated_scene_ids": sorted(
            scene for scene in scenes
            if getattr(evaluator.adapters.get(scene), "scene_evidence_status", None)
            == "authenticated-indexed"
        ),
    }


def _default_routes(scenes: Sequence[str], frames_per_scene: int) -> tuple[FormalRoute, ...]:
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
    receipt = dict(
        compare_baseline_candidate(
            baseline=baseline, candidate=candidate, routes=routes, g1_xml=g1_xml
        )
    )
    receipt["schema"] = "g1-full-walking-terrain-lmm-formal-evidence/v1"
    receipt["formal_identity"] = _formal_identity(
        candidate, scenes=[route.scene_id for route in routes], g1_xml=g1_xml
    )
    failures = acceptance_failures(receipt)
    receipt["acceptance_failures"] = failures
    receipt["accepted"] = not failures
    return receipt


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
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
        matcher = HybridMatcher(
            corpus,
            generator,
            adapter.authority,
            native_model=native_model,
            initial_root_xy=adapter.spawn_native_xy,
            initial_heading=adapter.spawn_heading,
        )
        receipt = run_interactive(
            matcher,
            adapter,
            g1_xml=arguments.g1_xml,
            gamepad=arguments.gamepad,
            max_render_frames=arguments.max_render_frames,
        )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
    if arguments.receipt is not None:
        write_receipt_exclusive(arguments.receipt, receipt)
    return 0 if bool(receipt.get("accepted", True)) else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "FORMAL_SCENE_IDS",
    "acceptance_failures",
    "build_parser",
    "main",
    "run_formal_smoke",
)
