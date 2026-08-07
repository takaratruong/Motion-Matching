"""Summarize and select a diverse visual review of co-warp compounds."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def _strict(pilot: dict[str, object]) -> bool:
    collision = dict(pilot["collision_audit"])
    mechanics = dict(pilot["mechanics"])
    quality = dict(pilot["quality"])
    source_quality = dict(pilot["source_quality"])
    drift = float(quality["maximum_stance_run_drift_m"])
    return bool(
        pilot.get("automatic_gate_accepted")
        and pilot.get("authored_pivot_constraints_used")
        and float(collision["maximum_foot_penetration_m"]) <= 0.005
        and float(collision["maximum_forbidden_body_penetration_m"]) <= 0.0
        and float(mechanics["maximum_joint_step_rad"]) <= 0.20
        and float(mechanics["maximum_root_acceleration_m_s2"]) <= 30.0
        and float(quality["maximum_stance_foot_step_m"]) <= 0.003
        and drift <= 0.020
        and drift
        <= float(source_quality["maximum_stance_run_drift_m"]) + 0.005
    )


def collect(root: Path) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    attempted = 0
    for summary_path in sorted(root.glob("source_*/clip_summary.json")):
        source_root = summary_path.parent
        summary = json.loads(summary_path.read_text())
        source = json.loads((source_root / "source.json").read_text())
        manifest_path = Path(str(summary["pilot_manifest"]))
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        for pilot in manifest["pilots"]:
            attempted += 1
            if not _strict(pilot):
                continue
            rows.append(
                {
                    "pilot": f"{manifest_path}#{pilot['label']}",
                    "manifest": str(manifest_path),
                    "label": str(pilot["label"]),
                    "motion": str(pilot["motion"]),
                    "terrain_usd": str(summary["terrain_usd"]),
                    "clip_index": int(summary["clip_index"]),
                    "clip_name": str(summary["clip_name"]),
                    "clip_family": str(summary["clip_family"]),
                    "clip_traversal": str(summary["clip_traversal"]),
                    "source_mode": str(summary["source_mode"]),
                    "compound_mode": str(pilot["mode"]),
                    "terrain_progress": float(
                        pilot["pivot"]["terrain_progress"]
                    ),
                    "source_selection_score": float(
                        source.get("selection_score", 0.0)
                    ),
                    "maximum_foot_penetration_m": float(
                        pilot["collision_audit"]["maximum_foot_penetration_m"]
                    ),
                    "maximum_forbidden_body_penetration_m": float(
                        pilot["collision_audit"][
                            "maximum_forbidden_body_penetration_m"
                        ]
                    ),
                    "maximum_stance_run_drift_m": float(
                        pilot["quality"]["maximum_stance_run_drift_m"]
                    ),
                }
            )
    return rows, attempted


def select_review(
    rows: list[dict[str, object]],
    *,
    count: int,
    maximum_per_physical_source: int,
) -> list[dict[str, object]]:
    requested = min(int(count), len(rows))
    maximum = int(maximum_per_physical_source)
    if requested < 1 or maximum < 1:
        return []
    selected: list[dict[str, object]] = []
    used: set[str] = set()
    per_source: dict[tuple[str, int], int] = defaultdict(int)
    seen_compound: set[str] = set()
    seen_source_mode: set[str] = set()
    seen_pair: set[tuple[str, str]] = set()
    seen_traversal: set[str] = set()
    seen_family: set[str] = set()
    seen_progress: set[str] = set()
    while len(selected) < requested:
        best: tuple[tuple[float, str], dict[str, object]] | None = None
        for row in rows:
            identity = str(row["pilot"])
            physical = (str(row["clip_family"]), int(row["clip_index"]))
            if identity in used or per_source[physical] >= maximum:
                continue
            compound = str(row["compound_mode"])
            source_mode = str(row["source_mode"])
            traversal = str(row["clip_traversal"])
            family = str(row["clip_family"])
            progress = float(row["terrain_progress"])
            progress_bin = (
                "early"
                if progress < 1.0 / 3.0
                else "late"
                if progress >= 2.0 / 3.0
                else "middle"
            )
            score = 0.0
            score += 120.0 * (compound not in seen_compound)
            score += 90.0 * (source_mode not in seen_source_mode)
            score += 45.0 * ((source_mode, compound) not in seen_pair)
            score += 25.0 * (traversal not in seen_traversal)
            score += 20.0 * (family not in seen_family)
            score += 8.0 * (progress_bin not in seen_progress)
            score += min(float(row["source_selection_score"]), 80.0)
            score += {"bounce": 4.0, "reverse": 2.0}.get(compound, 0.0)
            score -= 120.0 * float(row["maximum_stance_run_drift_m"])
            score -= 50.0 * float(row["maximum_foot_penetration_m"])
            ranked = (score, identity)
            if best is None or ranked > best[0]:
                best = (ranked, row)
        if best is None:
            break
        row = best[1]
        identity = str(row["pilot"])
        physical = (str(row["clip_family"]), int(row["clip_index"]))
        compound = str(row["compound_mode"])
        source_mode = str(row["source_mode"])
        progress = float(row["terrain_progress"])
        selected.append(row)
        used.add(identity)
        per_source[physical] += 1
        seen_compound.add(compound)
        seen_source_mode.add(source_mode)
        seen_pair.add((source_mode, compound))
        seen_traversal.add(str(row["clip_traversal"]))
        seen_family.add(str(row["clip_family"]))
        seen_progress.add(
            "early"
            if progress < 1.0 / 3.0
            else "late"
            if progress >= 2.0 / 3.0
            else "middle"
        )
    return selected


def summarize(
    root: Path,
    *,
    review_count: int,
    maximum_per_physical_source: int,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    rows, attempted = collect(root)
    review = select_review(
        rows,
        count=review_count,
        maximum_per_physical_source=maximum_per_physical_source,
    )
    payload = {
        "schema": "cowarped-compound-bank-summary/v1",
        "root": str(root),
        "attempted_pilot_count": attempted,
        "strict_accepted_count": len(rows),
        "distinct_physical_source_count": len(
            {(str(row["clip_family"]), int(row["clip_index"])) for row in rows}
        ),
        "strict_by_source_mode": dict(
            sorted(Counter(str(row["source_mode"]) for row in rows).items())
        ),
        "strict_by_compound_mode": dict(
            sorted(Counter(str(row["compound_mode"]) for row in rows).items())
        ),
        "strict_by_traversal": dict(
            sorted(Counter(str(row["clip_traversal"]) for row in rows).items())
        ),
        "review_count": len(review),
        "maximum_foot_penetration_m": max(
            (float(row["maximum_foot_penetration_m"]) for row in rows),
            default=0.0,
        ),
        "maximum_forbidden_body_penetration_m": max(
            (
                float(row["maximum_forbidden_body_penetration_m"])
                for row in rows
            ),
            default=0.0,
        ),
    }
    (root / "accepted.json").write_text(
        json.dumps(
            {"schema": "cowarped-compound-selection/v1", "rows": rows},
            indent=2,
        )
        + "\n"
    )
    (root / "review.json").write_text(
        json.dumps(
            {
                "schema": "cowarped-compound-review/v1",
                "selections": review,
            },
            indent=2,
        )
        + "\n"
    )
    (root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--review-count", type=int, default=32)
    parser.add_argument("--maximum-per-physical-source", type=int, default=2)
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            summarize(
                arguments.root,
                review_count=arguments.review_count,
                maximum_per_physical_source=(
                    arguments.maximum_per_physical_source
                ),
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
