"""Build failed-case evals using Task12 copied from the matched expert frame.

This is the controlled command-parity rerun.  It uses the old failed rollout
only to choose a pre-failure terrain stage/progress and an expert pose frame;
the new dense rollout is then run from scratch with that expert frame's exact
Task12 bytes.  Recovery matching can therefore require equality rather than a
soft velocity tolerance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .justin_recovery import (
    HISTORY_FRAMES,
    MIN_REMAINING_FRAMES,
    RecoverySceneContract,
    _scene_progress,
    _terrain_stage,
    load_recovery_trace,
    load_reference_clip,
    reference_task12,
    select_rewind_queries,
)


def _scene(path: Path) -> RecoverySceneContract:
    payload = json.loads(path.read_text())
    return RecoverySceneContract(
        scene_id=str(payload["scene_id"]),
        learner_front_xy=tuple(payload["learner_front_xy"]),
        reference_front_xy=tuple(payload["reference_front_xy"]),
        learner_heading_yaw_rad=float(payload["learner_heading_yaw_rad"]),
        reference_heading_yaw_rad=float(payload["reference_heading_yaw_rad"]),
        tread_m=float(payload["tread_m"]),
        num_steps=int(payload["num_steps"]),
        riser_progress_m=tuple(payload["riser_progress_m"]),
        geometry_sha256=str(payload["geometry_sha256"]),
        exact_command_required=True,
    )


def _matched_reference_frame(
    *, trace_path: Path, failure_step: int, scene: RecoverySceneContract, reference_path: Path
) -> dict[str, object]:
    trace = load_recovery_trace(trace_path)
    reference = load_reference_clip(reference_path)
    queries = select_rewind_queries(
        trace,
        first_fall_physics_step=failure_step,
        rewinds_s=tuple(value / 4.0 for value in range(1, 21)),
    )
    root = np.asarray(reference.arrays["body_pos_w"][:, 0], dtype=np.float64)
    progress = _scene_progress(
        root[:, :2],
        front_xy=scene.reference_front_xy,
        heading_yaw_rad=scene.reference_heading_yaw_rad,
    )
    stages = _terrain_stage(
        progress,
        tread_m=scene.tread_m,
        num_steps=scene.num_steps,
        riser_progress_m=scene.riser_progress_m,
    )
    frames = np.arange(reference.frame_count)
    matches = []
    for query in queries:
        learner_progress = float(
            _scene_progress(
                trace.arrays["qpos_mujoco"][query.trace_index, :2],
                front_xy=scene.learner_front_xy,
                heading_yaw_rad=scene.learner_heading_yaw_rad,
            )
        )
        learner_stage = int(
            _terrain_stage(
                learner_progress,
                tread_m=scene.tread_m,
                num_steps=scene.num_steps,
                riser_progress_m=scene.riser_progress_m,
            )
        )
        valid = (
            (frames >= HISTORY_FRAMES - 1)
            & (frames < reference.frame_count - MIN_REMAINING_FRAMES)
            & (stages == learner_stage)
            & (np.abs(progress - learner_progress) <= 0.25)
        )
        candidates = frames[valid]
        if not len(candidates):
            continue
        learner_joint = np.asarray(
            trace.arrays["joint_pos_isaac"][query.trace_index], dtype=np.float64
        )
        learner_z = float(trace.arrays["qpos_mujoco"][query.trace_index, 2])
        joint_rmse = np.sqrt(
            np.mean(
                (
                    np.asarray(
                        reference.arrays["joint_pos"][candidates], dtype=np.float64
                    )
                    - learner_joint[None]
                ) ** 2,
                axis=1,
            )
        )
        score = (
            4.0 * np.abs(progress[candidates] - learner_progress)
            + 2.0 * np.abs(root[candidates, 2] - learner_z)
            + joint_rmse
        )
        index = int(np.argmin(score))
        matches.append(
            (
                float(score[index]),
                query,
                learner_progress,
                learner_stage,
                int(candidates[index]),
                float(joint_rmse[index]),
            )
        )
    if not matches:
        raise ValueError("no expert frame shares the old pre-failure stage/progress")
    match_score, query, learner_progress, learner_stage, frame, joint_rmse = min(
        matches, key=lambda value: (value[0], value[1].rewind_s)
    )
    return {
        "frame": frame,
        "task12": reference_task12(reference, frame).reshape(-1).astype(float).tolist(),
        "rewind_s": query.rewind_s,
        "old_trace_index": query.trace_index,
        "learner_progress_m": learner_progress,
        "reference_progress_m": float(progress[frame]),
        "terrain_stage": learner_stage,
        "joint_position_rmse_rad": joint_rmse,
        "match_score": match_score,
        "reference": str(reference_path.resolve()),
        "reference_sha256": reference.sha256,
    }


def build(
    *, base_manifest_path: Path, baseline_root: Path, suite_root: Path, output: Path
) -> dict[str, object]:
    base = json.loads(base_manifest_path.expanduser().resolve().read_text())
    baseline = baseline_root.expanduser().resolve()
    suite = suite_root.expanduser().resolve()
    cases = []
    receipts = []
    for source_case in base["cases"]:
        case_id = str(source_case["case_id"])
        case_root = baseline / case_id
        validation = json.loads((case_root / "validation.json").read_text())
        if bool(validation["success"]):
            continue
        metrics = json.loads((case_root / "metrics.json").read_text())
        trace_path = case_root / "recovery_trace.npz"
        with np.load(trace_path, allow_pickle=False) as trace:
            terminal_step = int(np.asarray(trace["physics_step"])[-1])
        failure_step = int(
            metrics["first_fall_step"]
            if metrics.get("first_fall_step") is not None else terminal_step
        )
        scene_id = str(source_case["scene_id"])
        scene_root = suite / "scenes" / scene_id
        match = _matched_reference_frame(
            trace_path=trace_path,
            failure_step=failure_step,
            scene=_scene(scene_root / "scene_contract.json"),
            reference_path=scene_root / "expert_reference.npz",
        )
        case = dict(source_case)
        case["case_id"] = f"{case_id}_refcmd"
        case["task12"] = match["task12"]
        case["training_status"] = (
            "unseen_staircase_geometry_exact_expert_command_parity"
        )
        case["reference_command_match"] = {
            key: value for key, value in match.items() if key != "task12"
        }
        cases.append(case)
        receipts.append({"source_case_id": case_id, **match})
    manifest = dict(base)
    manifest.update(
        description=(
            "Dense State41 rerun of the 25 failed stair-suite cases with exact "
            "Task12 copied from a geometry-matched expert reference frame."
        ),
        cases=cases,
        command_parity={
            "schema": "expert-reference-command-parity/v1",
            "source_manifest": str(base_manifest_path.expanduser().resolve()),
            "failed_cases_only": True,
            "case_count": len(cases),
            "matches": receipts,
        },
    )
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = build(
        base_manifest_path=arguments.base_manifest,
        baseline_root=arguments.baseline_root,
        suite_root=arguments.suite_root,
        output=arguments.output,
    )
    print(json.dumps({"output": str(arguments.output.resolve()), "cases": len(result["cases"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
