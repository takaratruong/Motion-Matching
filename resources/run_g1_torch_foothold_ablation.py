#!/usr/bin/env python3
"""Compare completed foothold-selection matrices with one frozen scorecard."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCHEMA = "g1-foothold-arm-comparison/v1"
PENETRATION_LIMIT_M = 0.03


def _finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def selection_coherence(
    *,
    selected_clip: Sequence[str] | np.ndarray,
    selected_frame: Sequence[int] | np.ndarray,
    joint_velocity: np.ndarray,
    dt_s: float,
) -> dict[str, float | int]:
    """Measure whether selected source motion and emitted joints stay coherent."""

    clips = np.asarray(selected_clip)
    frames = np.asarray(selected_frame)
    velocity = np.asarray(joint_velocity, dtype=np.float64)
    dt = _finite(dt_s, "dt_s")
    count = clips.shape[0]
    if (
        dt <= 0.0
        or clips.ndim != 1
        or frames.shape != (count,)
        or velocity.ndim != 2
        or velocity.shape[0] != count
        or not np.issubdtype(frames.dtype, np.integer)
        or not np.isfinite(velocity).all()
    ):
        raise ValueError("selection coherence arrays are invalid")
    if count < 2:
        discontinuity = np.zeros(0, dtype=np.bool_)
        cross_clip = np.zeros(0, dtype=np.bool_)
    else:
        cross_clip = clips[1:] != clips[:-1]
        discontinuity = cross_clip | (frames[1:] != frames[:-1] + 1)
    boundaries = np.flatnonzero(discontinuity) + 1
    run_lengths = np.diff(np.concatenate(([0], boundaries, [count])))
    if count >= 3:
        joint_jerk = np.linalg.norm(
            np.diff(velocity, n=2, axis=0) / (dt * dt), axis=1
        )
    else:
        joint_jerk = np.zeros(0, dtype=np.float64)
    if count >= 2:
        transition_velocity_jump = np.linalg.norm(
            np.diff(velocity, axis=0), axis=1
        )[discontinuity]
    else:
        transition_velocity_jump = np.zeros(0, dtype=np.float64)
    return {
        "source_discontinuity_count": int(np.sum(discontinuity)),
        "cross_clip_transition_count": int(np.sum(cross_clip)),
        "shortest_sequential_run_frames": (
            int(np.min(run_lengths)) if run_lengths.size else 0
        ),
        "transition_joint_velocity_jump_p95_rad_s": (
            float(np.percentile(transition_velocity_jump, 95))
            if transition_velocity_jump.size
            else 0.0
        ),
        "transition_joint_velocity_jump_max_rad_s": (
            float(np.max(transition_velocity_jump))
            if transition_velocity_jump.size
            else 0.0
        ),
        "joint_jerk_p95_rad_s3": (
            float(np.percentile(joint_jerk, 95)) if joint_jerk.size else 0.0
        ),
        "joint_jerk_max_rad_s3": (
            float(np.max(joint_jerk)) if joint_jerk.size else 0.0
        ),
    }


def arm_rank_key(summary: Mapping[str, object]) -> tuple[float, ...]:
    """Lower is better under the approved safety-first selection contract."""

    required = (
        "safety_violation_count",
        "exception_count",
        "passed_route_class_count",
        "passed_route_count",
        "stalled_moving_fraction",
        "stance_slide_m",
        "joint_jerk_p95_rad_s3",
    )
    values = {name: _finite(summary[name], name) for name in required}
    return (
        values["safety_violation_count"],
        -values["passed_route_class_count"],
        values["exception_count"],
        -values["passed_route_count"],
        values["stalled_moving_fraction"],
        values["stance_slide_m"],
        values["joint_jerk_p95_rad_s3"],
    )


def _route_class(name: str) -> str:
    for prefix, route_class in (
        ("side-mount-", "side-mount"),
        ("cross-tread-", "cross-tread"),
        ("turn-", "turn"),
        ("diagonal-", "diagonal"),
        ("side-exit-", "side-exit"),
        ("riser-", "riser-control"),
        ("mixed-", "mixed"),
    ):
        if name.startswith(prefix):
            return route_class
    raise ValueError(f"unknown route class: {name}")


def summarize_arm(name: str, root: str | Path) -> dict[str, object]:
    root = Path(root)
    try:
        matrix = json.loads((root / "matrix.json").read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"cannot load completed matrix for {name}") from error
    route_files = sorted((root / "routes").glob("*/metrics.json"))
    if len(route_files) != len(matrix.get("routes", ())):
        raise ValueError(f"matrix route artifacts are incomplete for {name}")

    routes: dict[str, dict[str, object]] = {}
    class_results: dict[str, list[bool]] = {}
    exception_count = 0
    penetration_route_count = 0
    maximum_penetration = 0.0
    slide = 0.0
    stalled_frames = 0
    moving_frames = 0
    source_discontinuities = 0
    cross_clip_transitions = 0
    shortest_run: int | None = None
    transition_velocity_jump_p95 = 0.0
    transition_velocity_jump_max = 0.0
    joint_jerk: list[np.ndarray] = []
    for metrics_path in route_files:
        value = json.loads(metrics_path.read_text(encoding="utf-8"))
        route_name = str(value["route"])
        passed = bool(value["outcome"]["completed"])
        route_class = _route_class(route_name)
        class_results.setdefault(route_class, []).append(passed)
        failure = value.get("failure")
        exception_count += int(failure is not None)
        penetration = _finite(
            value["metrics"]["penetration_depth_m"]["maximum"],
            "penetration",
        )
        maximum_penetration = max(maximum_penetration, penetration)
        penetration_route_count += int(penetration > PENETRATION_LIMIT_M)
        route_slide = _finite(
            value["metrics"]["stance_slide_m"]["total"], "slide"
        )
        slide += route_slide
        arrays_path = metrics_path.with_name("arrays.npz")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            coherence = selection_coherence(
                selected_clip=arrays["selected_clip_path"],
                selected_frame=arrays["selected_source_frame"],
                joint_velocity=arrays["joint_velocity"],
                dt_s=0.02,
            )
            command_speed = np.linalg.norm(
                arrays["command_velocity_world_xy"], axis=1
            )
            moving_frames += int(np.sum(command_speed >= 0.10))
            stalled_frames += int(
                value["metrics"]["stalled_moving_frame_count"]
            )
            velocity = np.asarray(arrays["joint_velocity"], np.float64)
            if velocity.shape[0] >= 3:
                joint_jerk.append(
                    np.linalg.norm(
                        np.diff(velocity, n=2, axis=0) / (0.02**2), axis=1
                    )
                )
        source_discontinuities += int(
            coherence["source_discontinuity_count"]
        )
        cross_clip_transitions += int(
            coherence["cross_clip_transition_count"]
        )
        run = int(coherence["shortest_sequential_run_frames"])
        shortest_run = run if shortest_run is None else min(shortest_run, run)
        transition_velocity_jump_p95 = max(
            transition_velocity_jump_p95,
            float(coherence["transition_joint_velocity_jump_p95_rad_s"]),
        )
        transition_velocity_jump_max = max(
            transition_velocity_jump_max,
            float(coherence["transition_joint_velocity_jump_max_rad_s"]),
        )
        routes[route_name] = {
            "passed": passed,
            "failure_reasons": value["outcome"]["failure_reasons"],
            "exception": failure,
            "stance_slide_m": route_slide,
            "maximum_penetration_m": penetration,
            "longest_stall_frames": value["metrics"]["longest_stall_frames"],
            **coherence,
        }

    jerk = np.concatenate(joint_jerk) if joint_jerk else np.zeros(0)
    passed_classes = sorted(
        route_class
        for route_class, outcomes in class_results.items()
        if outcomes and all(outcomes)
    )
    summary: dict[str, object] = {
        "name": name,
        "root": str(root),
        "matrix_sha256": matrix["deterministic_sha256"],
        "dataset_identity": matrix["dataset_identity"],
        "config_identity": matrix["config_identity"],
        "route_count": len(routes),
        "passed_route_count": sum(
            int(value["passed"]) for value in routes.values()
        ),
        "passed_route_classes": passed_classes,
        "passed_route_class_count": len(passed_classes),
        "exception_count": exception_count,
        "penetration_violation_route_count": penetration_route_count,
        "safety_violation_count": penetration_route_count,
        "maximum_penetration_m": maximum_penetration,
        "stance_slide_m": slide,
        "stalled_moving_fraction": (
            stalled_frames / moving_frames if moving_frames else 0.0
        ),
        "source_discontinuity_count": source_discontinuities,
        "cross_clip_transition_count": cross_clip_transitions,
        "shortest_sequential_run_frames": shortest_run or 0,
        "transition_joint_velocity_jump_p95_rad_s": (
            transition_velocity_jump_p95
        ),
        "transition_joint_velocity_jump_max_rad_s": (
            transition_velocity_jump_max
        ),
        "joint_jerk_p95_rad_s3": (
            float(np.percentile(jerk, 95)) if jerk.size else 0.0
        ),
        "joint_jerk_max_rad_s3": float(np.max(jerk)) if jerk.size else 0.0,
        "routes": routes,
    }
    return summary


def compare_arms(specifications: Sequence[tuple[str, str | Path]]) -> dict:
    if not specifications:
        raise ValueError("at least one arm is required")
    names = [name for name, _ in specifications]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("arm names must be non-empty and unique")
    summaries = [summarize_arm(name, root) for name, root in specifications]
    identities = {summary["dataset_identity"] for summary in summaries}
    route_counts = {summary["route_count"] for summary in summaries}
    if len(identities) != 1 or len(route_counts) != 1:
        raise ValueError("arm matrices are not comparable")
    ordered = sorted(summaries, key=arm_rank_key)
    return {
        "schema": SCHEMA,
        "winner": ordered[0]["name"],
        "ranking": [summary["name"] for summary in ordered],
        "arms": summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=ARTIFACT_ROOT",
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    specifications = []
    for raw in args.arm:
        name, separator, root = raw.partition("=")
        if not separator or not name or not root:
            raise ValueError("--arm must be NAME=ARTIFACT_ROOT")
        specifications.append((name, root))
    comparison = compare_arms(specifications)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(comparison, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "winner": comparison["winner"],
                "ranking": comparison["ranking"],
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
