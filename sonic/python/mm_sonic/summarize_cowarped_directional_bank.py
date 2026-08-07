"""Summarize, select, and plot an exact-gated paired co-warp bank."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np


def _quality(row: dict[str, object]) -> float:
    warp = dict(row["warp"])
    collision = dict(row["collision_audit"])
    return float(
        float(warp["maximum_path_angle_deg"])
        + float(warp["maximum_facing_offset_deg"])
        + 20.0 * float(warp["lateral_offset_range_m"])
        - 80.0 * float(warp["maximum_stance_run_drift_m"])
        - 50.0 * float(collision["maximum_foot_penetration_m"])
    )


def collect(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    accepted: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for aggregate_path in sorted(root.glob("clip_*/aggregate.json")):
        aggregate = json.loads(aggregate_path.read_text())
        source = {
            "clip_index": int(aggregate["clip_index"]),
            "clip_name": str(aggregate["clip_name"]),
            "clip_family": str(aggregate["clip_family"]),
            "clip_traversal": str(aggregate["clip_traversal"]),
            "status": str(aggregate["status"]),
            "attempted": int(aggregate["attempted"]),
            "accepted": int(aggregate["accepted"]),
        }
        sources.append(source)
        for report in aggregate.get("reports", []):
            if report.get("status") != "accepted":
                continue
            destination = aggregate_path.parent / str(report["mode"])
            row = {
                **source,
                "mode": str(report["mode"]),
                "motion": str((destination / "motion.npz").resolve()),
                "terrain_usd": str(Path(str(report["terrain_usd"])).resolve()),
                "report": str((destination / "report.json").resolve()),
                "warp": dict(report["warp"]),
                "mesh_warp": dict(report["mesh_warp"]),
                "collision_audit": dict(report["collision_audit"]),
            }
            row["selection_score"] = _quality(row)
            accepted.append(row)
    return accepted, sources


def select_review(
    rows: list[dict[str, object]], count: int
) -> list[dict[str, object]]:
    ordered = sorted(rows, key=_quality, reverse=True)
    selected: list[dict[str, object]] = []
    selected_paths: set[str] = set()
    # First expose every admitted command family.  Then prefer new physical
    # sources before filling remaining slots with the strongest trajectories.
    for mode in sorted({str(row["mode"]) for row in ordered}):
        row = next(value for value in ordered if value["mode"] == mode)
        selected.append(row)
        selected_paths.add(str(row["report"]))
        if len(selected) >= count:
            return selected
    used_sources = {int(row["clip_index"]) for row in selected}
    for row in ordered:
        if str(row["report"]) in selected_paths:
            continue
        if int(row["clip_index"]) in used_sources:
            continue
        selected.append(row)
        selected_paths.add(str(row["report"]))
        used_sources.add(int(row["clip_index"]))
        if len(selected) >= count:
            return selected
    for row in ordered:
        if str(row["report"]) in selected_paths:
            continue
        selected.append(row)
        if len(selected) >= count:
            break
    return selected


def plot_routes(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    traversals = ("up", "down")
    for axis, traversal in zip(axes, traversals):
        subset = [row for row in rows if row["clip_traversal"] == traversal]
        for row in subset:
            with np.load(str(row["motion"]), allow_pickle=False) as payload:
                root = np.asarray(payload["root_position_world"], dtype=np.float64)
            xy = root[:, :2] - root[0, :2]
            _left, _singular, right = np.linalg.svd(
                xy - np.mean(xy, axis=0), full_matrices=False
            )
            forward = np.asarray(right[0], dtype=np.float64)
            probe = xy[min(len(xy) - 1, max(1, len(xy) // 3))]
            if float(np.dot(probe, forward)) < 0.0:
                forward *= -1.0
            yaw = math.atan2(float(forward[1]), float(forward[0]))
            cosine, sine = math.cos(-yaw), math.sin(-yaw)
            local = np.stack(
                (
                    cosine * xy[:, 0] - sine * xy[:, 1],
                    sine * xy[:, 0] + cosine * xy[:, 1],
                ),
                axis=1,
            )
            axis.plot(local[:, 0], local[:, 1], alpha=0.28, linewidth=1.0)
        axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
        axis.set_title(f"{traversal}: {len(subset)} accepted paths")
        axis.set_xlabel("forward progress (m)")
        axis.set_ylabel("lateral displacement (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
    figure.suptitle("Exact-gated paired terrain + motion co-warps")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def summarize(root: Path, review_count: int) -> dict[str, object]:
    accepted, sources = collect(root)
    root.mkdir(parents=True, exist_ok=True)
    review = select_review(accepted, min(int(review_count), len(accepted)))
    by_mode = Counter(str(row["mode"]) for row in accepted)
    by_traversal = Counter(str(row["clip_traversal"]) for row in accepted)
    summary = {
        "schema": "cowarped-directional-bank-summary/v1",
        "root": str(root.resolve()),
        "source_count": len(sources),
        "source_admission_count": sum(int(row["accepted"]) > 0 for row in sources),
        "attempted_motion_count": sum(int(row["attempted"]) for row in sources),
        "accepted_motion_count": len(accepted),
        "accepted_by_mode": dict(sorted(by_mode.items())),
        "accepted_by_traversal": dict(sorted(by_traversal.items())),
        "maximum_path_angle_deg": max(
            (float(dict(row["warp"])["maximum_path_angle_deg"]) for row in accepted),
            default=0.0,
        ),
        "maximum_facing_offset_deg": max(
            (float(dict(row["warp"])["maximum_facing_offset_deg"]) for row in accepted),
            default=0.0,
        ),
        "maximum_lateral_offset_range_m": max(
            (float(dict(row["warp"])["lateral_offset_range_m"]) for row in accepted),
            default=0.0,
        ),
        "maximum_foot_penetration_m": max(
            (
                float(dict(row["collision_audit"])["maximum_foot_penetration_m"])
                for row in accepted
            ),
            default=0.0,
        ),
        "maximum_forbidden_body_penetration_m": max(
            (
                float(
                    dict(row["collision_audit"])[
                        "maximum_forbidden_body_penetration_m"
                    ]
                )
                for row in accepted
            ),
            default=0.0,
        ),
        "review_count": len(review),
        "artifacts": {
            "accepted": "accepted.json",
            "review": "review.json",
            "routes": "all_routes.png",
        },
    }
    (root / "accepted.json").write_text(
        json.dumps({"schema": "cowarped-directional-selection/v1", "rows": accepted}, indent=2)
        + "\n"
    )
    (root / "review.json").write_text(
        json.dumps({"schema": "cowarped-directional-review/v1", "rows": review}, indent=2)
        + "\n"
    )
    plot_routes(accepted, root / "all_routes.png")
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--review-count", type=int, default=24)
    arguments = parser.parse_args()
    print(
        json.dumps(
            summarize(arguments.root.expanduser().resolve(), arguments.review_count),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
