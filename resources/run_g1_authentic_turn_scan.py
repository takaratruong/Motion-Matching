#!/usr/bin/env python3
"""Retrieve terrain-valid authentic GRAIL turns from an exact route boundary."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from mm_sonic.joints import ContractError
from mm_sonic.torch_contact_segments import source_support_mask
from mm_sonic.torch_g1_fk import MujocoG1FootKinematics
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_terrain_features import TerrainDataset
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)


def _transition_module():
    path = Path(__file__).with_name("run_g1_motion_field_transitions.py")
    spec = importlib.util.spec_from_file_location("motion_field_transitions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-dataset", type=Path, required=True)
    result.add_argument("--target-dataset", type=Path, required=True)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--g1-xml", type=Path, required=True)
    result.add_argument("--incoming", type=Path, required=True)
    result.add_argument("--incoming-frame", type=int, required=True)
    result.add_argument("--yaw-delta-degrees", type=float, required=True)
    result.add_argument("--minimum-frames", type=int, default=25)
    result.add_argument("--maximum-frames", type=int, default=45)
    result.add_argument("--heading-tolerance-degrees", type=float, default=12.0)
    result.add_argument("--maximum-entry-pose-rad", type=float, default=0.85)
    result.add_argument("--maximum-entry-translation-m", type=float, default=0.12)
    result.add_argument("--minimum-forward-progress-m", type=float, default=0.02)
    result.add_argument("--maximum-results", type=int, default=20)
    result.add_argument("--maximum-prefiltered-candidates", type=int, default=0)
    result.add_argument("--maximum-candidates-per-clip", type=int, default=0)
    result.add_argument("--output", type=Path, required=True)
    return result


def _yaw(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(quaternion), -1, 0)
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrapped(value: np.ndarray | float) -> np.ndarray:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def _route_from_clip(clip, support: np.ndarray, start: int, stop: int):
    window = slice(start, stop)
    return {
        "joint_position": np.asarray(clip.joint_position[window]),
        "root_position_world": np.asarray(
            clip.body_position_world[window, 0]
        ),
        "root_orientation_world_wxyz": np.asarray(
            clip.body_quaternion_world_wxyz[window, 0]
        ),
        "source_support_mask": np.asarray(support[window], dtype=np.bool_),
    }


def main() -> int:
    args = _parser().parse_args()
    started = time.perf_counter()
    transition = _transition_module()
    source = TerrainDataset.load(args.source_dataset, device="cpu")
    target = resolve_stair_config(
        args.target_dataset,
        load_experiment_config(args.config),
        device="cpu",
    )
    extension = target.measurement_extension
    foot_kinematics = MujocoG1FootKinematics(args.g1_xml)
    sole_kinematics = MujocoG1SoleKinematics(args.g1_xml)
    with np.load(args.incoming, allow_pickle=False) as archive:
        incoming = {
            name: np.asarray(archive[name]).copy()
            for name in (
                "joint_position",
                "root_position_world",
                "root_orientation_world_wxyz",
                "source_support_mask",
            )
        }
    frame = args.incoming_frame
    incoming_support = incoming["source_support_mask"][frame]
    if int(incoming_support.sum()) != 1:
        raise ValueError("authentic turn scan requires one incoming stance foot")
    stance_foot = int(np.flatnonzero(incoming_support)[0])
    incoming_joint = incoming["joint_position"][frame]
    incoming_root = incoming["root_position_world"][frame]
    incoming_orientation = incoming["root_orientation_world_wxyz"][frame]
    incoming_feet = foot_kinematics.foot_positions(
        incoming_joint[None], incoming_root[None], incoming_orientation[None]
    )[0]
    desired_delta = math.radians(args.yaw_delta_degrees)
    heading_tolerance = math.radians(args.heading_tolerance_degrees)
    descriptors = source.manifest["accepted_clips"]
    prefiltered = []
    support_cache: dict[int, np.ndarray] = {}
    for clip_index, clip in enumerate(source.folder.clips):
        support = source_support_mask(source, clip_index).cpu().numpy()
        support_cache[clip_index] = support
        roots = clip.body_position_world[:, 0]
        yaws = _yaw(clip.body_quaternion_world_wxyz[:, 0])
        pose_error = np.abs(clip.joint_position - incoming_joint).max(axis=1)
        for duration in range(args.minimum_frames, args.maximum_frames + 1):
            if duration >= clip.frame_count:
                continue
            starts = np.arange(0, clip.frame_count - duration + 1)
            stops = starts + duration
            delta = _wrapped(yaws[stops - 1] - yaws[starts])
            displacement = np.linalg.norm(
                roots[stops - 1, :2] - roots[starts, :2], axis=1
            )
            displacement_xy = roots[stops - 1, :2] - roots[starts, :2]
            forward_progress = (
                np.cos(yaws[starts]) * displacement_xy[:, 0]
                + np.sin(yaws[starts]) * displacement_xy[:, 1]
            )
            accepted = (
                np.all(support[starts] == incoming_support, axis=1)
                & (pose_error[starts] <= args.maximum_entry_pose_rad)
                & (np.abs(_wrapped(delta - desired_delta)) <= heading_tolerance)
                & (displacement >= 0.04)
                & (displacement <= 0.60)
                & (forward_progress >= args.minimum_forward_progress_m)
            )
            for start, stop, yaw_delta in zip(
                starts[accepted], stops[accepted], delta[accepted]
            ):
                prefiltered.append(
                    (clip_index, int(start), int(stop), float(yaw_delta))
                )

    prefiltered_count = len(prefiltered)
    def prefilter_key(item):
        return (
            abs(float(_wrapped(item[3] - desired_delta))),
            float(
                np.abs(
                    source.folder.clips[item[0]].joint_position[item[1]]
                    - incoming_joint
                ).max()
            ),
            item[0],
            item[1],
            item[2],
        )

    if args.maximum_candidates_per_clip > 0:
        selected_by_clip = []
        for clip_index in sorted({item[0] for item in prefiltered}):
            clip_candidates = [
                item for item in prefiltered if item[0] == clip_index
            ]
            clip_candidates.sort(key=prefilter_key)
            selected_by_clip.extend(
                clip_candidates[: args.maximum_candidates_per_clip]
            )
        prefiltered = selected_by_clip
    if args.maximum_prefiltered_candidates > 0:
        prefiltered.sort(key=prefilter_key)
        prefiltered = prefiltered[: args.maximum_prefiltered_candidates]
    evaluated = 0
    accepted_records = []
    accepted_routes = []
    near_records = []
    near_routes = []
    rejection_counts: dict[str, int] = {}
    for clip_index, start, stop, yaw_delta in prefiltered:
        clip = source.folder.clips[clip_index]
        source_route = _route_from_clip(
            clip, support_cache[clip_index], start, stop
        )
        source_root = source_route["root_position_world"]
        yaw_delta_place = float(_yaw(incoming_orientation)) - float(
            _yaw(source_route["root_orientation_world_wxyz"][0])
        )
        cosine, sine = math.cos(yaw_delta_place), math.sin(yaw_delta_place)
        relative = source_root[-1] - source_root[0]
        natural_endpoint = incoming_root + np.asarray(
            (
                cosine * relative[0] - sine * relative[1],
                sine * relative[0] + cosine * relative[1],
                relative[2],
            )
        )
        placed = transition.place_authentic_transition(
            source_route=source_route,
            incoming_root_position_world=incoming_root,
            incoming_root_orientation_world_wxyz=incoming_orientation,
            outgoing_root_position_world=natural_endpoint,
        )
        placed_feet = foot_kinematics.foot_positions(
            placed["joint_position"],
            placed["root_position_world"],
            placed["root_orientation_world_wxyz"],
        )
        correction = incoming_feet[stance_foot] - placed_feet[0, stance_foot]
        entry_pose = float(
            np.abs(source_route["joint_position"][0] - incoming_joint).max()
        )
        entry_rejections = transition.transition_entry_candidate_rejections(
            {
                "required_contact_root_translation_m": float(
                    np.linalg.norm(correction)
                ),
                "starting_pose_max_joint_delta_rad": entry_pose,
                "support_compatible": True,
            }
        )
        if entry_rejections or np.linalg.norm(correction) > args.maximum_entry_translation_m:
            for reason in entry_rejections or ("entry_root_relocation",):
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        try:
            route = transition.align_authentic_transition_through_support_transfer(
                placed_route=placed,
                placed_start_foot_position_world=placed_feet[0, stance_foot],
                target_start_foot_position_world=incoming_feet[stance_foot],
            )
        except ContractError:
            rejection_counts["missing_support_transfer"] = (
                rejection_counts.get("missing_support_transfer", 0) + 1
            )
            continue
        evaluated += 1
        soles = sole_kinematics.sole_points(
            route["joint_position"],
            route["root_position_world"],
            route["root_orientation_world_wxyz"],
        )
        scene_xy = extension.alignment.matcher_to_scene_xy(
            torch.tensor(soles[..., :2], dtype=torch.float32)
        )
        clearance = soles[..., 2] - (
            extension.query_grid.sample_xy(scene_xy).cpu().numpy()
        )
        metrics = transition.terrain_transition_contact_metrics(
            sole_clearance_m=clearance,
            support_mask=route["source_support_mask"],
        )
        feet = foot_kinematics.foot_positions(
            route["joint_position"],
            route["root_position_world"],
            route["root_orientation_world_wxyz"],
        )
        consecutive = (
            route["source_support_mask"][:-1]
            & route["source_support_mask"][1:]
        )
        horizontal_step = np.linalg.norm(
            np.diff(feet[..., :2], axis=0), axis=2
        )
        metrics.update(
            maximum_stance_horizontal_step_m=float(
                horizontal_step[consecutive].max()
                if consecutive.any()
                else 0.0
            ),
            maximum_joint_step_rad=float(
                np.abs(np.diff(route["joint_position"], axis=0)).max()
            ),
            maximum_root_step_m=float(
                np.linalg.norm(
                    np.diff(route["root_position_world"], axis=0), axis=1
                ).max()
            ),
            heading_change_error_degrees=abs(
                math.degrees(float(_wrapped(yaw_delta - desired_delta)))
            ),
        )
        rejections = transition.transition_candidate_rejections(metrics)
        if rejections:
            for reason in rejections:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            deficit = (
                max(0.0, -0.025 - float(metrics["minimum_sole_clearance_m"]))
                + max(
                    0.0,
                    float(metrics["maximum_stance_contact_error_m"]) - 0.035,
                )
                + 0.01
                * max(
                    0.0,
                    3.0 - float(metrics["minimum_supported_sole_points"]),
                )
                + max(
                    0.0,
                    float(metrics["maximum_stance_horizontal_step_m"]) - 0.010,
                )
                + max(
                    0.0, float(metrics["maximum_root_step_m"]) - 0.040
                )
                + max(
                    0.0, float(metrics["maximum_joint_step_rad"]) - 0.250
                )
            )
            if len(rejections) <= 2:
                near_records.append(
                    {
                        "clip_index": clip_index,
                        "logical_name": descriptors[clip_index]["logical_name"],
                        "source_frames": [start, stop],
                        "yaw_delta_degrees": math.degrees(yaw_delta),
                        "entry_pose_max_rad": entry_pose,
                        "entry_contact_translation_m": float(
                            np.linalg.norm(correction)
                        ),
                        "metrics": metrics,
                        "rejections": list(rejections),
                        "deficit": float(deficit),
                    }
                )
                near_routes.append(route)
            continue
        descriptor = descriptors[clip_index]
        record = {
            "clip_index": clip_index,
            "logical_name": descriptor["logical_name"],
            "source_frames": [start, stop],
            "yaw_delta_degrees": math.degrees(yaw_delta),
            "entry_pose_max_rad": entry_pose,
            "entry_contact_translation_m": float(np.linalg.norm(correction)),
            "natural_endpoint_matcher_xyz": route[
                "root_position_world"
            ][-1].tolist(),
            "metrics": metrics,
        }
        record["score"] = float(
            metrics["maximum_stance_contact_error_m"]
            + 0.25 * metrics["maximum_stance_horizontal_step_m"]
            + 0.01 * metrics["heading_change_error_degrees"]
            + 0.10 * entry_pose
            + 0.10 * np.linalg.norm(correction)
        )
        accepted_records.append(record)
        accepted_routes.append(route)

    order = np.argsort([record["score"] for record in accepted_records])
    args.output.mkdir(parents=True, exist_ok=True)
    published = []
    for rank, index in enumerate(order[: args.maximum_results]):
        record = accepted_records[int(index)]
        artifact = args.output / f"candidate-{rank:03d}.npz"
        np.savez_compressed(artifact, **accepted_routes[int(index)])
        published.append({**record, "artifact": str(artifact)})
    near_order = np.argsort([record["deficit"] for record in near_records])
    published_near = []
    for rank, index in enumerate(near_order[:20]):
        artifact = args.output / f"near-candidate-{rank:03d}.npz"
        np.savez_compressed(artifact, **near_routes[int(index)])
        published_near.append(
            {**near_records[int(index)], "artifact": str(artifact)}
        )
    report = {
        "schema": "g1-authentic-turn-scan/v1",
        "desired_yaw_delta_degrees": args.yaw_delta_degrees,
        "prefiltered_candidate_count": prefiltered_count,
        "selected_prefiltered_candidate_count": len(prefiltered),
        "fully_evaluated_candidate_count": evaluated,
        "accepted_candidate_count": len(accepted_records),
        "published_candidate_count": len(published),
        "rejection_counts": rejection_counts,
        "elapsed_s": time.perf_counter() - started,
        "candidates": published,
        "near_candidates": published_near,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in report if key != "candidates"}))
    return 0 if published else 1


if __name__ == "__main__":
    raise SystemExit(main())
