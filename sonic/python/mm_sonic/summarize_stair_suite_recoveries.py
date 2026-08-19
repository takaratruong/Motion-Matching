"""Summarize exact-command stair reruns and tracker recovery attempts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODES = ("exact", "blend25", "blend50")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def summarize(root: Path) -> dict[str, object]:
    suite = root.expanduser().resolve()
    manifest = _read(suite / "reference_command_eval_manifest.json")
    case_by_id = {
        str(value["case_id"]): value for value in manifest["cases"]
    }
    rows = []
    for case_id, manifest_case in sorted(case_by_id.items()):
        scene_id = str(manifest_case["scene_id"])
        baseline_root = suite / "reference_command_baseline_traces" / case_id
        validation_path = baseline_root / "validation.json"
        proposal_path = (
            suite / "scenes" / scene_id / "proposals_refcmd" / f"{case_id}.json"
        )
        row: dict[str, object] = {
            "case_id": case_id,
            "scene_id": scene_id,
            "approach_distance_m": manifest_case.get("approach_distance_m"),
            "reference_command_baseline": None,
            "proposal": None,
            "attempts": {},
        }
        if validation_path.is_file():
            validation = _read(validation_path)
            row["reference_command_baseline"] = {
                "success": bool(validation["success"]),
                "failed_gates": validation.get("failed_gates", []),
                "validation": str(validation_path),
                "video": validation.get("video"),
            }
        if proposal_path.is_file():
            proposal = _read(proposal_path)
            candidates = [
                candidate
                for query in proposal["queries"]
                for candidate in query["candidates"]
            ]
            row["proposal"] = {
                "path": str(proposal_path),
                "candidate_count": len(candidates),
                "minimum_rewind_s": min(
                    float(query["rewind_s"])
                    for query in proposal["queries"]
                    if query["candidates"]
                ),
                "maximum_command_knot_rmse": max(
                    float(value["command_knot_rmse"]) for value in candidates
                ),
                "exact_command_required": bool(
                    proposal["command_matching"]["exact_equality_required"]
                ),
                "geometry_sha256": proposal["scene_transform"][
                    "geometry_sha256"
                ],
            }
        for mode in MODES:
            attempt = (
                suite
                / "recovery_attempts"
                / scene_id
                / f"{case_id}_{mode}"
            )
            receipt_path = attempt / "receipt.json"
            if not receipt_path.is_file():
                row["attempts"][mode] = None
                continue
            receipt = _read(receipt_path)
            evaluation = receipt["evaluation"]
            row["attempts"][mode] = {
                "path": str(attempt),
                "accepted_recovery": bool(evaluation["accepted_recovery"]),
                "screen_pass": bool(evaluation["screen_pass"]),
                "completed_reference": bool(
                    evaluation["completed_reference"]
                ),
                "terminated_early": bool(evaluation["terminated_early"]),
                "reached_platform": bool(evaluation["reached_platform"]),
                "stable_platform": bool(evaluation["stable_platform"]),
                "mean_mpjpe_mm": float(evaluation["mean_mpjpe_mm"]),
                "peak_mpjpe_mm": float(evaluation["peak_mpjpe_mm"]),
                "rewind_s": float(receipt["rewind_s"]),
                "reference_blend": float(
                    receipt["state_injection"]["reference_blend"]
                ),
            }
        rows.append(row)

    summary = {
        "schema": "stair-suite10-recovery-results/v1",
        "suite_root": str(suite),
        "case_count": len(rows),
        "reference_command_baseline_completed": sum(
            value["reference_command_baseline"] is not None for value in rows
        ),
        "reference_command_baseline_successes": sum(
            bool(value["reference_command_baseline"]["success"])
            for value in rows
            if value["reference_command_baseline"] is not None
        ),
        "strict_proposals": sum(value["proposal"] is not None for value in rows),
        "maximum_proposal_command_knot_rmse": max(
            (
                float(value["proposal"]["maximum_command_knot_rmse"])
                for value in rows
                if value["proposal"] is not None
            ),
            default=None,
        ),
        "attempts_completed_by_mode": {
            mode: sum(value["attempts"][mode] is not None for value in rows)
            for mode in MODES
        },
        "accepted_recoveries_by_mode": {
            mode: sum(
                bool(value["attempts"][mode]["accepted_recovery"])
                for value in rows
                if value["attempts"][mode] is not None
            )
            for mode in MODES
        },
        "cases": rows,
    }
    destination = suite / "recovery_results_summary.json"
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(json.dumps(summarize(arguments.suite_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
