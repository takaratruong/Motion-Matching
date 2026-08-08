#!/usr/bin/env python3
"""Rigidly place one validated edge template at every matching field crossing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
from mm_sonic.joints import ContractError
from mm_sonic.torch_g1_fk import MujocoG1FootKinematics
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_object_motion_field import assign_endpoint_to_heading_line
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)


def _module(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene_translation_to_matcher(
    *, scene_translation_xy: object, yaw_scene_from_matcher_rad: float
) -> np.ndarray:
    """Map a scene-frame displacement through the inverse planar alignment."""

    value = np.asarray(scene_translation_xy, dtype=np.float64)
    yaw = float(yaw_scene_from_matcher_rad)
    if value.shape != (2,) or not np.isfinite(value).all() or not math.isfinite(yaw):
        raise ContractError("scene translation is invalid")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array(
        (cosine * value[0] + sine * value[1], -sine * value[0] + cosine * value[1]),
        dtype=np.float64,
    )


def local_xy_search_offsets(
    *, search_radius_m: float, search_step_m: float
) -> np.ndarray:
    """Return deterministic smallest-first rigid placement corrections."""

    if (
        not math.isfinite(float(search_radius_m))
        or not math.isfinite(float(search_step_m))
        or search_radius_m < 0.0
        or search_step_m <= 0.0
    ):
        raise ContractError("local placement search is invalid")
    count = math.floor(search_radius_m / search_step_m + 1.0e-12)
    values = np.arange(-count, count + 1, dtype=np.float64) * search_step_m
    candidates = np.array(
        [
            (x, y)
            for x in values
            for y in values
            if math.hypot(float(x), float(y)) <= search_radius_m + 1.0e-12
        ],
        dtype=np.float64,
    )
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            float(candidates[index] @ candidates[index]),
            float(candidates[index, 0]),
            float(candidates[index, 1]),
        ),
    )
    return np.ascontiguousarray(candidates[order])


def placement_rejections(
    *,
    contact_rejections: object,
    placed_start_line_id: str,
    placed_end_line_id: str,
    expected_start_line_id: str,
    expected_end_line_id: str,
    start_lane_error_m: float,
    end_lane_error_m: float,
    maximum_lane_error_m: float,
) -> tuple[str, ...]:
    """Combine terrain and exact raster-cell admission gates."""

    rejections = list(contact_rejections)
    if (
        placed_start_line_id != expected_start_line_id
        or placed_end_line_id != expected_end_line_id
    ):
        rejections.append("lane_assignment_mismatch")
    if max(start_lane_error_m, end_lane_error_m) > maximum_lane_error_m:
        rejections.append("lane_error")
    return tuple(sorted(set(rejections)))


def publication_contact_rejections(
    *, metrics: object, maximum_contact_error_m: float
) -> tuple[str, ...]:
    """Apply the visual-quality stance-contact gate used for graph publication."""

    try:
        error = float(metrics["maximum_stance_contact_error_m"])
    except (KeyError, TypeError, ValueError) as exception:
        raise ContractError("publication contact metrics are invalid") from exception
    maximum = float(maximum_contact_error_m)
    if not math.isfinite(error) or not math.isfinite(maximum) or maximum <= 0.0:
        raise ContractError("publication contact metrics are invalid")
    return ("publication_contact_error",) if error > maximum else ()


def target_heading_pair(
    *,
    source_from_heading_degrees: float,
    source_to_heading_degrees: float,
    requested_from_heading_degrees: float | None,
    requested_to_heading_degrees: float | None,
) -> tuple[float, float]:
    """Resolve a rigidly rotated heading pair without changing the turn."""

    values = (
        source_from_heading_degrees,
        source_to_heading_degrees,
        requested_from_heading_degrees,
        requested_to_heading_degrees,
    )
    if not all(value is None or math.isfinite(float(value)) for value in values):
        raise ContractError("target heading pair is invalid")
    if (requested_from_heading_degrees is None) != (
        requested_to_heading_degrees is None
    ):
        raise ContractError("target heading pair is incomplete")
    if requested_from_heading_degrees is None:
        return float(source_from_heading_degrees), float(source_to_heading_degrees)
    source_change = source_to_heading_degrees - source_from_heading_degrees
    requested_change = requested_to_heading_degrees - requested_from_heading_degrees
    if abs(source_change - requested_change) > 1.0e-6:
        raise ContractError("target heading pair changes the turn amount")
    return float(requested_from_heading_degrees), float(requested_to_heading_degrees)


def rigidly_rotate_route_yaw(
    *, route: dict[str, np.ndarray], pivot_matcher_xy: object, yaw_offset_rad: float
) -> dict[str, np.ndarray]:
    """Apply one global yaw about a matcher-frame pivot without pose edits."""

    pivot = np.asarray(pivot_matcher_xy, dtype=np.float64)
    yaw = float(yaw_offset_rad)
    if pivot.shape != (2,) or not np.isfinite(pivot).all() or not math.isfinite(yaw):
        raise ContractError("rigid route yaw rotation is invalid")
    output = rigidly_place_route(
        route=route,
        translation_matcher_xyz=(0.0, 0.0, 0.0),
    )
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    centered = output["root_position_world"][:, :2] - pivot
    output["root_position_world"][:, :2] = centered @ rotation.T + pivot
    yaw_quaternion = np.asarray(
        (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)),
        dtype=np.float64,
    )
    right = output["root_orientation_world_wxyz"]
    lw, lx, ly, lz = yaw_quaternion
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    output["root_orientation_world_wxyz"] = np.ascontiguousarray(
        np.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            axis=-1,
        )
    )
    return output


def rigidly_place_route(
    *, route: dict[str, np.ndarray], translation_matcher_xyz: object
) -> dict[str, np.ndarray]:
    """Apply a single XYZ translation while preserving the complete pose sequence."""

    translation = np.asarray(translation_matcher_xyz, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ContractError("rigid route placement is invalid")
    required = (
        "joint_position",
        "root_position_world",
        "root_orientation_world_wxyz",
        "source_support_mask",
    )
    try:
        output = {name: np.asarray(route[name]).copy() for name in required}
    except (KeyError, TypeError) as error:
        raise ContractError("rigid route placement is invalid") from error
    if output["root_position_world"].ndim != 2 or output["root_position_world"].shape[1] != 3:
        raise ContractError("rigid route placement is invalid")
    output["root_position_world"] = np.ascontiguousarray(
        output["root_position_world"] + translation
    )
    return output


def support_aligned_z_translation(
    *,
    source_sole_clearance_m: object,
    candidate_sole_clearance_m: object,
    support_mask: object,
) -> float:
    """Align a rigid placement to terrain using its authentic support samples."""

    source = np.asarray(source_sole_clearance_m, dtype=np.float64)
    candidate = np.asarray(candidate_sole_clearance_m, dtype=np.float64)
    support = np.asarray(support_mask)
    if (
        source.ndim != 3
        or source.shape != candidate.shape
        or support.shape != source.shape[:2]
        or support.dtype != np.bool_
        or not support.any()
        or not np.isfinite(source).all()
        or not np.isfinite(candidate).all()
    ):
        raise ContractError("support-aligned vertical placement is invalid")
    selected = np.broadcast_to(support[..., None], source.shape)
    return float(np.median(source[selected]) - np.median(candidate[selected]))


def _load_route(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: np.asarray(archive[name]).copy()
            for name in (
                "joint_position",
                "root_position_world",
                "root_orientation_world_wxyz",
                "source_support_mask",
            )
        }


def _crossing_lines(crossing, from_heading: float, to_heading: float) -> tuple[str, str]:
    if abs(crossing.first_heading_degrees - from_heading) < 1.0e-6:
        source = crossing.first_line_id
    elif abs(crossing.second_heading_degrees - from_heading) < 1.0e-6:
        source = crossing.second_line_id
    else:
        raise ContractError("crossing does not contain source heading")
    if abs(crossing.first_heading_degrees - to_heading) < 1.0e-6:
        target = crossing.first_line_id
    elif abs(crossing.second_heading_degrees - to_heading) < 1.0e-6:
        target = crossing.second_line_id
    else:
        raise ContractError("crossing does not contain target heading")
    return source, target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--edge-specifications", type=Path, required=True)
    parser.add_argument("--edge-id", required=True)
    parser.add_argument("--placement-template-id")
    parser.add_argument("--target-from-heading-degrees", type=float)
    parser.add_argument("--target-to-heading-degrees", type=float)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--maximum-lane-error-m", type=float, default=0.10)
    parser.add_argument(
        "--maximum-publication-contact-error-m", type=float, default=0.020
    )
    parser.add_argument("--local-search-radius-m", type=float, default=0.04)
    parser.add_argument("--local-search-step-m", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph_module = _module("run_g1_motion_field_transition_graph.py")
    transition = _module("run_g1_motion_field_transitions.py")
    lines, intersections = graph_module.load_motion_field_geometry(args.geometry)
    geometry = json.loads(args.geometry.read_text("utf-8"))
    expected_query_scene = (
        "terrain/grail/"
        + geometry["geometry"]["target_scene"]
        + "/motion.npz"
    )
    specifications = json.loads(args.edge_specifications.read_text("utf-8"))["edges"]
    matches = [item for item in specifications if item.get("edge_id") == args.edge_id]
    if len(matches) != 1:
        raise ContractError("placement scan edge id must resolve exactly once")
    specification = matches[0]
    resolved = resolve_stair_config(
        args.target_dataset,
        load_experiment_config(args.config),
        device="cpu",
    )
    extension = resolved.measurement_extension

    def matcher_to_scene(points: object) -> np.ndarray:
        return (
            extension.alignment.matcher_to_scene_xy(
                torch.as_tensor(points, dtype=torch.float32)
            )
            .cpu()
            .numpy()
        )

    record = graph_module.load_edge_record(
        specification=specification,
        matcher_to_scene_xy=matcher_to_scene,
        expected_query_scene=expected_query_scene,
    )
    artifact = Path(record["artifact"])
    route = _load_route(artifact)
    source_from_heading = float(record["from_heading_degrees"])
    source_to_heading = float(record["to_heading_degrees"])
    start_assignment = assign_endpoint_to_heading_line(
        lines=lines,
        heading_degrees=source_from_heading,
        endpoint_scene_xy=record["start_scene_xy"],
    )
    end_assignment = assign_endpoint_to_heading_line(
        lines=lines,
        heading_degrees=source_to_heading,
        endpoint_scene_xy=record["end_scene_xy"],
    )
    source_crossings = [
        item
        for item in intersections
        if {item.first_line_id, item.second_line_id}
        == {start_assignment.line_id, end_assignment.line_id}
    ]
    if len(source_crossings) != 1:
        raise ContractError("template edge crossing is ambiguous")
    source_crossing = source_crossings[0]
    from_heading, to_heading = target_heading_pair(
        source_from_heading_degrees=source_from_heading,
        source_to_heading_degrees=source_to_heading,
        requested_from_heading_degrees=args.target_from_heading_degrees,
        requested_to_heading_degrees=args.target_to_heading_degrees,
    )
    candidates = [
        item
        for item in intersections
        if {item.first_heading_degrees, item.second_heading_degrees}
        == {from_heading, to_heading}
    ]
    foot_kinematics = MujocoG1FootKinematics(args.g1_xml)
    sole_kinematics = MujocoG1SoleKinematics(args.g1_xml)
    yaw = float(extension.alignment.yaw_scene_from_matcher.cpu().item())
    source_crossing_matcher = scene_translation_to_matcher(
        scene_translation_xy=(
            np.asarray(source_crossing.scene_xy, dtype=np.float64)
            - extension.alignment.translation_scene_xy.cpu().numpy()
        ),
        yaw_scene_from_matcher_rad=yaw,
    )
    yaw_offset_degrees = from_heading - source_from_heading
    rotated_route = rigidly_rotate_route_yaw(
        route=route,
        pivot_matcher_xy=source_crossing_matcher,
        yaw_offset_rad=math.radians(yaw_offset_degrees),
    )

    def surface(point: object) -> float:
        value = torch.as_tensor(point, dtype=torch.float32)
        return float(extension.query_grid.sample_xy(value).cpu().item())

    source_surface = surface(source_crossing.scene_xy)
    source_soles = sole_kinematics.sole_points(
        route["joint_position"],
        route["root_position_world"],
        route["root_orientation_world_wxyz"],
    )
    source_sole_scene_xy = matcher_to_scene(source_soles[..., :2])
    source_clearance = source_soles[..., 2] - (
        extension.query_grid.sample_xy(
            torch.as_tensor(source_sole_scene_xy, dtype=torch.float32)
        )
        .cpu()
        .numpy()
    )
    output_records = []
    accepted = 0
    local_offsets = local_xy_search_offsets(
        search_radius_m=args.local_search_radius_m,
        search_step_m=args.local_search_step_m,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for index, crossing in enumerate(candidates):
        from_line_id, to_line_id = _crossing_lines(crossing, from_heading, to_heading)
        scene_delta = np.asarray(crossing.scene_xy) - source_crossing.scene_xy
        matcher_delta = scene_translation_to_matcher(
            scene_translation_xy=scene_delta,
            yaw_scene_from_matcher_rad=yaw,
        )
        target_surface = surface(crossing.scene_xy)
        selected = None
        fallback = None
        for local_offset in local_offsets:
            horizontal_translation = np.array(
                (
                    matcher_delta[0] + local_offset[0],
                    matcher_delta[1] + local_offset[1],
                    0.0,
                ),
                dtype=np.float64,
            )
            horizontally_placed = rigidly_place_route(
                route=rotated_route,
                translation_matcher_xyz=horizontal_translation,
            )
            candidate_soles = sole_kinematics.sole_points(
                horizontally_placed["joint_position"],
                horizontally_placed["root_position_world"],
                horizontally_placed["root_orientation_world_wxyz"],
            )
            candidate_sole_scene_xy = matcher_to_scene(candidate_soles[..., :2])
            candidate_clearance = candidate_soles[..., 2] - (
                extension.query_grid.sample_xy(
                    torch.as_tensor(candidate_sole_scene_xy, dtype=torch.float32)
                )
                .cpu()
                .numpy()
            )
            vertical_translation = support_aligned_z_translation(
                source_sole_clearance_m=source_clearance,
                candidate_sole_clearance_m=candidate_clearance,
                support_mask=route["source_support_mask"],
            )
            trial_clearance = candidate_clearance + vertical_translation
            trial_metrics = transition.terrain_transition_contact_metrics(
                sole_clearance_m=trial_clearance,
                support_mask=route["source_support_mask"],
            )
            trial = (
                horizontal_translation,
                vertical_translation,
                local_offset,
                trial_metrics,
            )
            if fallback is None:
                fallback = trial
            contact_rejections = transition.transition_candidate_rejections(
                {
                    **trial_metrics,
                    "maximum_stance_horizontal_step_m": 0.0,
                    "maximum_joint_step_rad": 0.0,
                    "maximum_root_step_m": 0.0,
                    "heading_change_error_degrees": 0.0,
                }
            )
            contact_rejections = tuple(contact_rejections) + publication_contact_rejections(
                metrics=trial_metrics,
                maximum_contact_error_m=args.maximum_publication_contact_error_m,
            )
            trial_start_scene, trial_end_scene = matcher_to_scene(
                horizontally_placed["root_position_world"][[0, -1], :2]
            )
            trial_start = assign_endpoint_to_heading_line(
                lines=lines,
                heading_degrees=from_heading,
                endpoint_scene_xy=trial_start_scene,
            )
            trial_end = assign_endpoint_to_heading_line(
                lines=lines,
                heading_degrees=to_heading,
                endpoint_scene_xy=trial_end_scene,
            )
            trial_rejections = placement_rejections(
                contact_rejections=contact_rejections,
                placed_start_line_id=trial_start.line_id,
                placed_end_line_id=trial_end.line_id,
                expected_start_line_id=from_line_id,
                expected_end_line_id=to_line_id,
                start_lane_error_m=trial_start.lateral_distance_m,
                end_lane_error_m=trial_end.lateral_distance_m,
                maximum_lane_error_m=args.maximum_lane_error_m,
            )
            if not trial_rejections:
                selected = trial
                break
        if selected is None:
            assert fallback is not None
            selected = fallback
        horizontal_translation, vertical_translation, local_offset, _ = selected
        translation = horizontal_translation.copy()
        translation[2] = vertical_translation
        placed = rigidly_place_route(
            route=rotated_route, translation_matcher_xyz=translation
        )
        feet = foot_kinematics.foot_positions(
            placed["joint_position"],
            placed["root_position_world"],
            placed["root_orientation_world_wxyz"],
        )
        soles = sole_kinematics.sole_points(
            placed["joint_position"],
            placed["root_position_world"],
            placed["root_orientation_world_wxyz"],
        )
        scene_xy = matcher_to_scene(soles[..., :2])
        clearance = soles[..., 2] - (
            extension.query_grid.sample_xy(
                torch.as_tensor(scene_xy, dtype=torch.float32)
            )
            .cpu()
            .numpy()
        )
        metrics = transition.terrain_transition_contact_metrics(
            sole_clearance_m=clearance,
            support_mask=placed["source_support_mask"],
        )
        consecutive = (
            placed["source_support_mask"][:-1]
            & placed["source_support_mask"][1:]
        )
        foot_step = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=2)
        metrics.update(
            maximum_stance_horizontal_step_m=float(
                foot_step[consecutive].max() if consecutive.any() else 0.0
            ),
            maximum_joint_step_rad=float(
                np.abs(np.diff(placed["joint_position"], axis=0)).max()
            ),
            maximum_root_step_m=float(
                np.linalg.norm(
                    np.diff(placed["root_position_world"], axis=0), axis=1
                ).max()
            ),
            heading_change_error_degrees=abs(
                float(record["observed_heading_change_degrees"])
                - (to_heading - from_heading)
            ),
        )
        start_scene, end_scene = matcher_to_scene(
            placed["root_position_world"][[0, -1], :2]
        )
        placed_start = assign_endpoint_to_heading_line(
            lines=lines,
            heading_degrees=from_heading,
            endpoint_scene_xy=start_scene,
        )
        placed_end = assign_endpoint_to_heading_line(
            lines=lines,
            heading_degrees=to_heading,
            endpoint_scene_xy=end_scene,
        )
        rejections = placement_rejections(
            contact_rejections=(
                tuple(transition.transition_candidate_rejections(metrics))
                + publication_contact_rejections(
                    metrics=metrics,
                    maximum_contact_error_m=args.maximum_publication_contact_error_m,
                )
            ),
            placed_start_line_id=placed_start.line_id,
            placed_end_line_id=placed_end.line_id,
            expected_start_line_id=from_line_id,
            expected_end_line_id=to_line_id,
            start_lane_error_m=placed_start.lateral_distance_m,
            end_lane_error_m=placed_end.lateral_distance_m,
            maximum_lane_error_m=args.maximum_lane_error_m,
        )
        item = {
            "index": index,
            "from_line_id": from_line_id,
            "to_line_id": to_line_id,
            "placed_start_line_id": placed_start.line_id,
            "placed_end_line_id": placed_end.line_id,
            "placed_start_lane_error_m": placed_start.lateral_distance_m,
            "placed_end_lane_error_m": placed_end.lateral_distance_m,
            "intersection_scene_xy": crossing.scene_xy,
            "translation_matcher_xyz": translation.tolist(),
            "local_placement_correction_matcher_xy": local_offset.tolist(),
            "source_surface_height_m": source_surface,
            "target_surface_height_m": target_surface,
            "metrics": metrics,
            "rejections": list(rejections),
            "validated": not rejections,
        }
        if not rejections:
            path = args.output / f"placement-{accepted:03d}.npz"
            np.savez_compressed(path, **placed)
            item["artifact"] = str(path.resolve())
            item["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            accepted += 1
        output_records.append(item)
    report = {
        "schema": "g1-motion-field-edge-placement-scan/v1",
        "template_artifact": str(artifact),
        "template_edge_id": args.placement_template_id or args.edge_id,
        "source_template_edge_id": args.edge_id,
        "source_from_heading_degrees": source_from_heading,
        "source_to_heading_degrees": source_to_heading,
        "target_from_heading_degrees": from_heading,
        "target_to_heading_degrees": to_heading,
        "rigid_yaw_offset_degrees": yaw_offset_degrees,
        "maximum_lane_error_m": args.maximum_lane_error_m,
        "maximum_publication_contact_error_m": (
            args.maximum_publication_contact_error_m
        ),
        "local_search_radius_m": args.local_search_radius_m,
        "local_search_step_m": args.local_search_step_m,
        "query_scene": expected_query_scene,
        "candidate_count": len(candidates),
        "accepted_count": accepted,
        "records": output_records,
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps({"report": str(report_path), "candidate_count": len(candidates), "accepted_count": accepted}))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
