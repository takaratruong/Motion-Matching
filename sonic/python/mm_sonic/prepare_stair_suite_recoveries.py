"""Prepare strict geometry/command-matched recovery proposals for the suite.

This stage is deliberately fail-closed.  It accepts only exact-audited expert
routes, embeds Task12 from a real learner trace, and never substitutes a dense
policy success rollout when an expert route is unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .build_synthesized_recovery_reference import build as build_reference
from .justin_recovery import (
    RecoveryContractError,
    RecoverySceneContract,
    build_recovery_proposal,
    load_recovery_trace,
    load_reference_clip,
    write_recovery_proposal,
)
from .select_command_compatible_stair_expert import select as select_expert


REWINDS_S = tuple(np.arange(0.25, 5.01, 0.25).tolist())


def _scene_contract(path: Path) -> tuple[dict[str, object], RecoverySceneContract]:
    payload = json.loads(path.read_text())
    return payload, RecoverySceneContract(
        scene_id=str(payload["scene_id"]),
        learner_front_xy=tuple(payload["learner_front_xy"]),
        reference_front_xy=tuple(payload["reference_front_xy"]),
        learner_heading_yaw_rad=float(payload["learner_heading_yaw_rad"]),
        reference_heading_yaw_rad=float(payload["reference_heading_yaw_rad"]),
        tread_m=float(payload["tread_m"]),
        num_steps=int(payload["num_steps"]),
        riser_progress_m=tuple(payload["riser_progress_m"]),
        geometry_sha256=str(payload["geometry_sha256"]),
        exact_command_required=bool(payload["exact_command_required"]),
    )


def _cases_for_scene(baseline_root: Path, scene_id: str):
    rows = []
    for validation_path in sorted(baseline_root.glob("*/validation.json")):
        validation = json.loads(validation_path.read_text())
        if validation.get("scene_id") != scene_id:
            continue
        case_root = validation_path.parent
        metrics_path = case_root / "metrics.json"
        trace_path = case_root / "recovery_trace.npz"
        if not metrics_path.is_file() or not trace_path.is_file():
            raise FileNotFoundError(f"incomplete baseline case {case_root}")
        rows.append(
            (case_root.name, validation, json.loads(metrics_path.read_text()), trace_path)
        )
    if not rows:
        raise ValueError(f"{scene_id} has no baseline cases")
    return rows


def prepare(
    *,
    suite_root: Path,
    baseline_root: Path,
    model_path: Path,
    proposal_dirname: str = "proposals",
    summary_name: str = "preparation_summary.json",
) -> dict[str, object]:
    root = suite_root.expanduser().resolve()
    baseline = baseline_root.expanduser().resolve()
    scene_rows = []
    for scene_root in sorted((root / "scenes").glob("stairset_*")):
        contract_path = scene_root / "scene_contract.json"
        contract_payload, scene = _scene_contract(contract_path)
        route_summary_path = scene_root / "route" / "summary.json"
        route_summary = (
            json.loads(route_summary_path.read_text())
            if route_summary_path.is_file() else {"status": "missing"}
        )
        row: dict[str, object] = {
            "scene_id": scene.scene_id,
            "geometry_sha256": scene.geometry_sha256,
            "route_status": route_summary.get("status"),
            "route_method": route_summary.get("method"),
            "reference": None,
            "cases": [],
        }
        cases = _cases_for_scene(baseline, scene.scene_id)
        if route_summary.get("status") != "accepted":
            row["rejection"] = "no exact-audited expert route"
            scene_rows.append(row)
            continue
        selection = select_expert(
            route_dir=scene_root / "route",
            contract_path=contract_path,
            target_forward_speed_mps=0.55,
        )
        motion_path = Path(
            str(selection["selected"]["recovery_motion_path"])
        ).resolve()
        reference_path = scene_root / "expert_reference.npz"
        build_reference(
            motion_path,
            reference_path,
            model_path=model_path,
            scene_contract_path=contract_path,
        )
        reference = load_reference_clip(reference_path)
        row["reference"] = {
            "path": str(reference.path),
            "sha256": reference.sha256,
            "frames": reference.frame_count,
            "command_source": (
                "canonical kinematic inference from exact warped expert reference"
            ),
            "command_selection": str(
                (scene_root / "route" / "command_selection.json").resolve()
            ),
        }
        for case_id, validation, metrics, trace_path in cases:
            case_row: dict[str, object] = {
                "case_id": case_id,
                "baseline_success": bool(validation["success"]),
                "trace": str(trace_path),
            }
            if validation["success"]:
                case_row["status"] = "baseline_pass_not_recovered"
                row["cases"].append(case_row)
                continue
            trace = load_recovery_trace(trace_path)
            formal_fall = metrics.get("first_fall_step")
            failure_step = (
                int(formal_fall)
                if formal_fall is not None
                else int(trace.arrays["physics_step"][-1])
            )
            failure_source = (
                "reported_first_fall"
                if formal_fall is not None else "terminal_degradation"
            )
            try:
                proposal = build_recovery_proposal(
                    trace,
                    (reference,),
                    first_fall_physics_step=failure_step,
                    failure_source=failure_source,
                    rewinds_s=REWINDS_S,
                    top_k=16,
                    scene=scene,
                )
            except RecoveryContractError as error:
                case_row.update(status="no_strict_match", reason=str(error))
            else:
                proposal["evaluation_geometry"] = contract_payload[
                    "evaluation_geometry"
                ]
                proposal["expert_route"] = {
                    "summary": str(route_summary_path.resolve()),
                    "method": route_summary["method"],
                    "motion": str(motion_path),
                    "dense_policy_fallback_used": False,
                }
                proposal_path = scene_root / proposal_dirname / f"{case_id}.json"
                write_recovery_proposal(proposal, proposal_path)
                surviving = [
                    value for value in proposal["queries"] if value["candidates"]
                ]
                best = min(
                    (
                        candidate
                        for query in surviving
                        for candidate in query["candidates"]
                    ),
                    key=lambda value: value["total_cost"],
                )
                case_row.update(
                    status="strict_match_ready",
                    proposal=str(proposal_path.resolve()),
                    matching_rewinds=len(surviving),
                    best_command_max_abs_error=float(
                        max(
                            abs(value)
                            for value in np.asarray(
                                best["reference_task12"], dtype=np.float64
                            )
                            - np.asarray(
                                surviving[0]["learner_task12"], dtype=np.float64
                            )
                        )
                    ),
                )
            row["cases"].append(case_row)
        scene_rows.append(row)
    summary = {
        "schema": "stair-suite10-strict-recovery-preparation/v1",
        "suite_root": str(root),
        "baseline_root": str(baseline),
        "policy_success_references_used": False,
        "scene_count": len(scene_rows),
        "scenes": scene_rows,
    }
    summary["accepted_expert_routes"] = sum(
        value["route_status"] == "accepted" for value in scene_rows
    )
    summary["strict_matches_ready"] = sum(
        case.get("status") == "strict_match_ready"
        for scene in scene_rows for case in scene["cases"]
    )
    summary["failed_baseline_cases"] = sum(
        not case["baseline_success"]
        for scene in scene_rows for case in scene["cases"]
    )
    destination = root / summary_name
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--proposal-dirname", default="proposals")
    parser.add_argument("--summary-name", default="preparation_summary.json")
    arguments = parser.parse_args(argv)
    result = prepare(
        suite_root=arguments.suite_root,
        baseline_root=arguments.baseline_root,
        model_path=arguments.model,
        proposal_dirname=arguments.proposal_dirname,
        summary_name=arguments.summary_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
