"""Package densely reviewed natural terrain motions and exact derivatives."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _command_statistics(motion_path: Path) -> dict[str, object]:
    with np.load(motion_path, allow_pickle=False) as arrays:
        velocity = np.asarray(
            arrays["command_velocity_robot_local_xy"], dtype=np.float64
        )
        stop = np.asarray(arrays["command_stop"], dtype=bool)
        fps = float(np.asarray(arrays["fps"]).reshape(()))
        frame_count = int(len(velocity))
    speed = np.linalg.norm(velocity, axis=1)
    moving = speed >= 0.15
    angle = np.degrees(np.arctan2(velocity[:, 1], velocity[:, 0]))
    return {
        "frame_count": frame_count,
        "duration_s": float(frame_count / fps),
        "median_moving_speed_mps": (
            float(np.median(speed[moving])) if np.any(moving) else 0.0
        ),
        "median_moving_local_angle_deg": (
            float(np.median(angle[moving])) if np.any(moving) else 0.0
        ),
        "stop_fraction": float(np.mean(stop)),
    }


def _audit_summary(payload: dict[str, object]) -> dict[str, object]:
    collision = payload.get("collision_audit") or {}
    stance = payload.get("stance_contact_audit") or {}
    warp = payload.get("warp") or payload.get("motion_metrics") or {}
    return {
        "maximum_foot_penetration_m": collision.get(
            "maximum_foot_penetration_m"
        ),
        "maximum_forbidden_body_penetration_m": collision.get(
            "maximum_forbidden_body_penetration_m"
        ),
        "maximum_stance_run_drift_m": stance.get(
            "maximum_stance_run_drift_m",
            warp.get("maximum_stance_run_drift_m"),
        ),
        "minimum_stance_support_point_count": stance.get(
            "minimum_stance_support_point_count",
            warp.get("minimum_stance_support_point_count"),
        ),
        "maximum_joint_step_rad": warp.get("maximum_joint_step_rad"),
        "maximum_root_acceleration_m_s2": warp.get(
            "maximum_root_acceleration_m_s2"
        ),
    }


def _derivative_row(
    *,
    root: Path,
    source: dict[str, object],
    clip_index: int,
    construction: str,
    relative_report: Path,
    visual_basis: str,
) -> dict[str, object] | None:
    report_path = root / relative_report
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text())
    allowed = (
        {"accepted"}
        if construction == "exact_scene_mirror"
        else {"pending_dense_visual_review"}
    )
    if report.get("status") not in allowed:
        return None
    motion_path = Path(str(report["motion"])).expanduser().resolve()
    terrain_path = Path(str(report["terrain_usd"])).expanduser().resolve()
    if not motion_path.is_file() or not terrain_path.is_file():
        return None
    return {
        "clip_index": clip_index,
        "clip_family": source.get("clip_family"),
        "clip_traversal": source.get("clip_traversal"),
        "construction": construction,
        "visual_basis": visual_basis,
        "motion": str(motion_path),
        "terrain_usd": str(terrain_path),
        "report": str(report_path.resolve()),
        "commands": _command_statistics(motion_path),
        "audit": _audit_summary(report),
    }


def _source_row(
    source: dict[str, object], visual: dict[str, object]
) -> dict[str, object]:
    motion_path = Path(str(source["motion"])).expanduser().resolve()
    return {
        "clip_index": int(source["clip_index"]),
        "clip_family": source.get("clip_family"),
        "clip_traversal": source.get("clip_traversal"),
        "construction": "densely_reviewed_natural_source",
        "visual_basis": "dense_5fps_review",
        "visual_review_notes": visual.get("notes"),
        "motion": str(motion_path),
        "terrain_usd": str(
            Path(str(source["terrain_usd"])).expanduser().resolve()
        ),
        "report": str(Path(str(source["report"])).expanduser().resolve()),
        "commands": _command_statistics(motion_path),
        "audit": _audit_summary(source),
    }


def build(review_roots: Sequence[Path], output: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for value in review_roots:
        root = value.expanduser().resolve()
        review = json.loads((root / "review.json").read_text())
        visual = json.loads((root / "visual_review.json").read_text())
        by_clip = {int(row["clip_index"]): row for row in review["rows"]}
        for decision in visual["rows"]:
            if decision.get("status") != "accepted":
                continue
            clip_index = int(decision["clip_index"])
            source = by_clip[clip_index]
            rows.append(_source_row(source, decision))
            candidates = (
                (
                    "exact_scene_mirror",
                    Path("mirrors") / f"clip_{clip_index}" / "report.json",
                    "exact_scene_reflection_of_dense_reviewed_source",
                ),
                (
                    "exact_time_reverse",
                    Path("time_reverse") / f"clip_{clip_index}" / "report.json",
                    "exact_pose_reverse_of_dense_reviewed_source",
                ),
                (
                    "exact_mirror_time_reverse",
                    Path("time_reverse_mirrors")
                    / f"clip_{clip_index}"
                    / "report.json",
                    "exact_pose_reverse_of_exact_scene_mirror",
                ),
            )
            for construction, report, visual_basis in candidates:
                row = _derivative_row(
                    root=root,
                    source=source,
                    clip_index=clip_index,
                    construction=construction,
                    relative_report=report,
                    visual_basis=visual_basis,
                )
                if row is not None:
                    rows.append(row)

    construction_counts = Counter(str(row["construction"]) for row in rows)
    family_counts = Counter(str(row["clip_family"]) for row in rows)
    payload: dict[str, object] = {
        "schema": "natural-terrain-gold-manifest/v1",
        "row_count": len(rows),
        "construction_counts": dict(sorted(construction_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "rows": rows,
    }
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    payload = build(arguments.review_root, arguments.output)
    print(json.dumps({"output": str(arguments.output), "rows": payload["row_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
