"""Roll and mechanically evaluate continuous terrain-conditioned matching.

This is intentionally a kinematic evaluator.  It answers the question that
must be settled before SONIC is involved: can ordinary frame-level motion
matching continuously choose and blend G1 motions whose prospective feet fit
the known target surface, without stair labels or pre-authored portals?
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from .canonical_terrain_matcher import (
    ContinuousTerrainMotionMatcher,
    ContinuousTerrainRuntimeConfig,
    GenericTerrainSearchConfig,
    exact_trajectory_from_clip,
    generated_state_from_clip,
    load_canonical_terrain_library,
    select_terrain_candidate,
)
from .terrain_oracle.storage import load_corpus
from .torch_motion_features import FEATURE_HORIZON_FRAMES
from .torch_motion_matcher import ForcedMotionAlignment, MatcherConfig


DEFAULT_CORPUS = Path(
    "/move/data/terrain-aware/motion-matching/terrain-oracle-task9.wFKkIV/raw"
)
DEFAULT_FAMILIES = (
    "grail/c490_curb/",
    "grail/c490_slope/",
    "grail/c490_stair_p1/",
    "grail/c490_stair_p2/",
)
DEFAULT_FLAT_CLIPS = (
    "flat/walk_forward_start_002__A021",
    "flat/walk_forward_start_002__A021_M",
    "flat/walk_sideway_135_start_002__A021",
    "flat/walk_sideway_135_start_002__A021_M",
)


def _yaw_from_wxyz(value: object) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrapped(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _default_source_ids(
    corpus: Path,
    target_clip_id: str,
    *,
    per_family: int,
    leave_target_out: bool,
) -> list[str]:
    manifest = load_corpus(corpus)
    available = {record.clip_id for record in manifest.clips}
    selected = [clip_id for clip_id in DEFAULT_FLAT_CLIPS if clip_id in available]
    for prefix in DEFAULT_FAMILIES:
        family = sorted(
            record.clip_id
            for record in manifest.clips
            if record.clip_id.startswith(prefix)
            and (not leave_target_out or record.clip_id != target_clip_id)
        )
        selected.extend(family[:per_family])
    if not leave_target_out and target_clip_id not in selected:
        selected.append(target_clip_id)
    return selected


def _alignment_between_roots(
    source_clip: object,
    source_frame: int,
    target_clip: object,
    target_frame: int,
) -> ForcedMotionAlignment:
    source_position = np.asarray(source_clip.root_position_world[source_frame], np.float64)
    target_position = np.asarray(target_clip.root_position_world[target_frame], np.float64)
    source_yaw = float(_yaw_from_wxyz(source_clip.root_quaternion_world_wxyz[source_frame]))
    target_yaw = float(_yaw_from_wxyz(target_clip.root_quaternion_world_wxyz[target_frame]))
    yaw_offset = math.atan2(
        math.sin(target_yaw - source_yaw), math.cos(target_yaw - source_yaw)
    )
    cosine = math.cos(yaw_offset)
    sine = math.sin(yaw_offset)
    rotated = np.asarray(
        (
            cosine * source_position[0] - sine * source_position[1],
            sine * source_position[0] + cosine * source_position[1],
            source_position[2],
        )
    )
    translation = target_position - rotated
    return ForcedMotionAlignment(
        yaw_offset_rad=yaw_offset,
        translation_world_xyz=tuple(float(value) for value in translation),
        synchronize_simulation_character=False,
    )


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    source_ids = list(args.source_clip_id)
    if not source_ids:
        source_ids = _default_source_ids(
            args.corpus,
            args.target_clip_id,
            per_family=args.source_per_family,
            leave_target_out=args.leave_target_out,
        )
    source_library = load_canonical_terrain_library(
        args.corpus, clip_ids=source_ids
    )
    target_library = load_canonical_terrain_library(
        args.corpus, clip_ids=[args.target_clip_id]
    )
    target = target_library.canonical_clips[0]
    target_field = target_library.height_fields[0]
    matcher = ContinuousTerrainMotionMatcher.from_library(
        source_library,
        target_field,
        device=args.device,
        matcher_config=MatcherConfig(
            trajectory_model="takara_ball",
            search_interval_steps=args.search_interval_steps,
            inertialization_halflife_s=args.inertialization_halflife_s,
        ),
        search_config=GenericTerrainSearchConfig(
            preselection_count=args.preselection_count,
            terrain_height_weight=args.terrain_height_weight,
            terrain_normal_weight=args.terrain_normal_weight,
            transition_penalty=args.transition_penalty,
            contact_timing_weight=args.contact_timing_weight,
        ),
        runtime_config=ContinuousTerrainRuntimeConfig(
            vertical_adaptation=not args.disable_vertical_adaptation,
            vertical_halflife_s=args.vertical_halflife_s,
            maximum_vertical_step_m=args.maximum_vertical_step_m,
        ),
    )

    start = int(args.start_frame)
    available_steps = target.frame_count - FEATURE_HORIZON_FRAMES[-1] - start - 1
    steps = min(args.steps, available_steps)
    if steps <= 0:
        raise ValueError("target frame range is too short for rolling evaluation")

    if args.target_clip_id in source_library.clip_ids:
        source_index = source_library.clip_ids.index(args.target_clip_id)
        reset_row = matcher.database.row_for_source(source_index, start)
        if reset_row is None:
            raise ValueError("target start frame is not searchable")
        reset_alignment = ForcedMotionAlignment(
            yaw_offset_rad=0.0,
            translation_world_xyz=(0.0, 0.0, 0.0),
            synchronize_simulation_character=False,
        )
    else:
        initial = select_terrain_candidate(
            matcher.database,
            matcher.terrain_rows,
            generated_state_from_clip(target, start, device=matcher.device),
            exact_trajectory_from_clip(target, start, device=matcher.device),
            target_field,
            config=matcher.search_config,
        )
        reset_row = initial.selected_row
        reset_alignment = _alignment_between_roots(
            source_library.canonical_clips[initial.selected_clip_index],
            initial.selected_frame_index,
            target,
            start,
        )
    reset = matcher.reset_to_row(int(reset_row), alignment=reset_alignment)

    roots = [reset.root_position_world.detach().cpu().numpy()]
    root_yaws = [
        float(_yaw_from_wxyz(reset.root_orientation_world_wxyz.detach().cpu().numpy()))
    ]
    root_quaternions = [
        reset.root_orientation_world_wxyz.detach().cpu().numpy()
    ]
    joints = [reset.joint_position.detach().cpu().numpy()]
    feet = [matcher._generated_sole_positions(reset).detach().cpu().numpy()]
    contacts = []
    selected_clip_ids: list[str] = []
    selected_frames: list[int] = []
    search_rows: list[dict[str, object]] = []

    path_to_index = {
        clip.relative_path: index
        for index, clip in enumerate(source_library.folder.clips)
    }
    for offset in range(steps):
        frame = start + offset
        requested_velocity = tuple(
            float(value) for value in target.root_linear_velocity_world[frame, :2]
        )
        requested_heading = float(
            _yaw_from_wxyz(target.root_quaternion_world_wxyz[frame])
        )
        result = matcher.step(requested_velocity, requested_heading)
        clip_index = path_to_index[result.diagnostics.selected_clip_path]
        selected_clip_ids.append(source_library.clip_ids[clip_index])
        selected_frames.append(result.diagnostics.selected_frame)
        contacts.append(
            np.asarray(
                source_library.canonical_clips[clip_index].contact[
                    result.diagnostics.selected_frame
                ]
                >= 0.5,
                dtype=np.bool_,
            )
        )
        roots.append(result.root_position_world.detach().cpu().numpy())
        root_yaws.append(
            float(
                _yaw_from_wxyz(
                    result.root_orientation_world_wxyz.detach().cpu().numpy()
                )
            )
        )
        root_quaternions.append(
            result.root_orientation_world_wxyz.detach().cpu().numpy()
        )
        joints.append(result.joint_position.detach().cpu().numpy())
        feet.append(
            matcher._generated_sole_positions(result).detach().cpu().numpy()
        )
        event = matcher.last_search_event
        if event is not None and event.sequence == result.diagnostics.sequence:
            search_rows.append(
                {
                    "sequence": event.sequence,
                    "source_clip_id": event.source_clip_id,
                    "selected_frame": event.selection.selected_frame_index,
                    "transitioned": event.transitioned,
                    "feature_cost": event.selection.feature_cost,
                    "terrain_cost": event.selection.terrain_cost,
                    "contact_cost": event.selection.contact_cost,
                    "pose_cost": event.selection.pose_cost,
                    "terrain_rms_m": event.selection.terrain_rms_m,
                    "minimum_prospective_foot_clearance_m": (
                        event.selection.minimum_prospective_foot_clearance_m
                    ),
                    "total_cost": event.selection.total_cost,
                }
            )

    roots_array = np.asarray(roots, dtype=np.float64)
    yaw_array = np.asarray(root_yaws, dtype=np.float64)
    quaternion_array = np.asarray(root_quaternions, dtype=np.float64)
    joints_array = np.asarray(joints, dtype=np.float64)
    feet_array = np.asarray(feet, dtype=np.float64)
    contacts_array = np.asarray(contacts, dtype=np.bool_)
    height, _normal, hit = target_field.sample(feet_array[1:, :, :2])
    clearance = feet_array[1:, :, 2] - height
    contact_clearance = clearance[contacts_array & hit]
    consecutive_contact = contacts_array[1:] & contacts_array[:-1]
    planted_displacement = np.linalg.norm(
        np.diff(feet_array[1:, :, :2], axis=0), axis=-1
    )[consecutive_contact]
    root_step = np.linalg.norm(np.diff(roots_array, axis=0), axis=-1)
    yaw_step = np.abs(_wrapped(np.diff(yaw_array)))
    joint_step = np.max(np.abs(np.diff(joints_array, axis=0)), axis=-1)
    root_acceleration = np.linalg.norm(
        np.diff(roots_array, n=2, axis=0) / (0.02 * 0.02), axis=-1
    )
    target_root = np.asarray(
        target.root_position_world[start + 1 : start + steps + 1], dtype=np.float64
    )
    path_error = np.linalg.norm(roots_array[1:, :2] - target_root[:, :2], axis=-1)

    def maximum(values: np.ndarray, default: float = 0.0) -> float:
        return float(np.max(values)) if values.size else default

    metrics = {
        "steps": steps,
        "search_count": len(search_rows),
        "transition_count": sum(bool(row["transitioned"]) for row in search_rows),
        "selected_clip_count": len(set(selected_clip_ids)),
        "selected_clip_histogram": dict(Counter(selected_clip_ids).most_common()),
        "terrain_query_hit_fraction": float(np.mean(hit)),
        "maximum_contact_penetration_m": max(0.0, maximum(-contact_clearance)),
        "maximum_contact_hover_m": max(0.0, maximum(contact_clearance)),
        "maximum_planted_foot_drift_m_per_frame": maximum(planted_displacement),
        "maximum_root_step_m": maximum(root_step),
        "maximum_root_yaw_step_rad": maximum(yaw_step),
        "maximum_joint_step_rad": maximum(joint_step),
        "maximum_root_acceleration_mps2": maximum(root_acceleration),
        "path_error_mean_m": float(np.mean(path_error)),
        "path_error_p95_m": float(np.quantile(path_error, 0.95)),
        "path_error_final_m": float(path_error[-1]),
        "search_terrain_rms_mean_m": (
            float(np.mean([row["terrain_rms_m"] for row in search_rows]))
            if search_rows
            else math.nan
        ),
        "search_terrain_rms_p95_m": (
            float(np.quantile([row["terrain_rms_m"] for row in search_rows], 0.95))
            if search_rows
            else math.nan
        ),
        "search_minimum_prospective_foot_clearance_m": (
            float(
                np.min(
                    [
                        row["minimum_prospective_foot_clearance_m"]
                        for row in search_rows
                    ]
                )
            )
            if search_rows
            else math.nan
        ),
    }
    gates = {
        "terrain_queries_hit": bool(np.all(hit)),
        "contact_penetration_le_5mm": metrics["maximum_contact_penetration_m"] <= 0.005,
        "planted_drift_le_10mm": metrics["maximum_planted_foot_drift_m_per_frame"] <= 0.010,
        "root_step_le_60mm": metrics["maximum_root_step_m"] <= 0.060,
        "root_yaw_step_le_0p35rad": metrics["maximum_root_yaw_step_rad"] <= 0.35,
        "joint_step_le_0p25rad": metrics["maximum_joint_step_rad"] <= 0.25,
        "root_acceleration_le_40mps2": metrics["maximum_root_acceleration_mps2"] <= 40.0,
    }
    report = {
        "schema": "continuous-terrain-rolling-eval/v1",
        "corpus": str(args.corpus.resolve()),
        "target_clip_id": args.target_clip_id,
        "target_left_out": bool(args.leave_target_out),
        "start_frame": start,
        "source_clip_ids": list(source_library.clip_ids),
        "source_row_count": matcher.database.feature_shape[0],
        "matcher_config": {
            "search_interval_steps": args.search_interval_steps,
            "inertialization_halflife_s": args.inertialization_halflife_s,
        },
        "search_config": {
            "preselection_count": args.preselection_count,
            "terrain_height_weight": args.terrain_height_weight,
            "terrain_normal_weight": args.terrain_normal_weight,
            "transition_penalty": args.transition_penalty,
            "contact_timing_weight": args.contact_timing_weight,
        },
        "metrics": metrics,
        "gates": gates,
        "passed": bool(all(gates.values())),
        "searches": search_rows,
    }
    np.savez_compressed(
        args.output.with_suffix(".trace.npz"),
        root_position_world=roots_array.astype(np.float32),
        root_yaw_world=yaw_array.astype(np.float32),
        root_quaternion_world_wxyz=quaternion_array.astype(np.float32),
        joint_position=joints_array.astype(np.float32),
        foot_position_world=feet_array.astype(np.float32),
        contact=contacts_array,
        selected_frame=np.asarray(selected_frames, dtype=np.int32),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--target-clip-id", required=True)
    parser.add_argument("--source-clip-id", action="append", default=[])
    parser.add_argument("--source-per-family", type=int, default=2)
    parser.add_argument("--leave-target-out", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--search-interval-steps", type=int, default=5)
    parser.add_argument("--inertialization-halflife-s", type=float, default=0.10)
    parser.add_argument("--preselection-count", type=int, default=2048)
    parser.add_argument("--terrain-height-weight", type=float, default=8.0)
    parser.add_argument("--terrain-normal-weight", type=float, default=0.75)
    parser.add_argument("--transition-penalty", type=float, default=1.0)
    parser.add_argument("--contact-timing-weight", type=float, default=4.0)
    parser.add_argument("--disable-vertical-adaptation", action="store_true")
    parser.add_argument("--vertical-halflife-s", type=float, default=0.08)
    parser.add_argument("--maximum-vertical-step-m", type=float, default=0.012)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = evaluate(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "searches"}, indent=2))
    print(f"wrote {args.output}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
