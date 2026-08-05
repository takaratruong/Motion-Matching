"""Reconstruct an exact stair route from globally planned one-riser fragments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Callable, Collection, Sequence

import numpy as np

from .terrain_oracle.motion_graph import TransitionConfig
from .terrain_oracle.stair_foothold_anchors import FootholdAnchorConfig
from .terrain_oracle.stair_fragment_reconstruction import (
    ExactStairFragmentBank,
    PlannedFragmentReconstructionRejected,
    enrich_stair_fragment_bank_from_exact_mesh,
    reconstruct_stair_fragment_sequence,
)
from .terrain_oracle.stair_fragment_sequence import (
    StairFragmentSequenceConfig,
    load_stair_fragment_records_jsonl,
    plan_stair_fragment_sequences,
)
from .terrain_oracle.stair_geometry_warp import _archive_terrain_index
from .terrain_oracle.stair_support_route import sample_stair_support_route
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


DEFAULT_ARCHIVE = Path(
    "/move/data/terrain-aware/motion-matching/"
    "grail-stairs500-clean-50hz-v1.zarr"
)
DEFAULT_FRAGMENT_BANK = Path(
    "/move/data/terrain-aware/motion-matching/"
    "stairs500-fragments-all500-exact-v1/fragments.jsonl"
)
DEFAULT_MODEL = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/"
    "g1_29dof_rev_1_0.xml"
)


def _precomputed_exact_bank_metadata(
    bank_path: str | Path,
    record_count: int,
    *,
    archive_path: str | Path | None = None,
) -> dict[str, object] | None:
    """Return a matching geometry-only exact-bank sidecar when available."""

    sidecar = Path(bank_path).resolve().parent / "exact_measurement.json"
    try:
        value = json.loads(sidecar.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(value, dict):
        return None
    try:
        exact_record_count = int(value.get("exact_record_count", -1))
        target_samples = int(value.get("target_motion_samples_used", -1))
    except (TypeError, ValueError):
        return None
    if exact_record_count != int(record_count):
        return None
    if target_samples != 0:
        return None
    expected_digest = str(value.get("fragment_bank_sha256", ""))
    if not expected_digest:
        return None
    actual_digest = hashlib.sha256(Path(bank_path).read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        return None
    if archive_path is not None:
        expected_archive = value.get("archive_path")
        if expected_archive is None or (
            Path(str(expected_archive)).expanduser().resolve()
            != Path(archive_path).expanduser().resolve()
        ):
            return None
    return value


@dataclass(frozen=True)
class RankedReconstructionOutcome:
    """First passing ranked reconstruction plus all earlier honest rejections."""

    selected_rank: int | None
    selected_plan: object | None
    reconstruction: object | None
    rejected_plans: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class AcceptedRankedReconstruction:
    """One downstream-validated reconstruction retained from ranked search."""

    rank: int
    plan: object
    reconstruction: object


def select_entry_diverse_plans(
    plans: Sequence[object],
    *,
    maximum_plans: int,
    preserve_ranked_plans: int,
    diversify_exit: bool = False,
) -> tuple[object, ...]:
    """Keep the best prefix, then spend trials on boundary diversity."""

    if maximum_plans < 1:
        raise ValueError("maximum plans must be positive")
    if preserve_ranked_plans < 0:
        raise ValueError("preserved ranked plans must be nonnegative")
    ordered = tuple(plans)
    if len(ordered) <= maximum_plans:
        return ordered
    prefix_count = min(preserve_ranked_plans, maximum_plans, len(ordered))
    selected = list(ordered[:prefix_count])
    selected_ids = {id(plan) for plan in selected}

    def entry_fragment_id(plan: object) -> str | None:
        selections = tuple(getattr(plan, "selections"))
        if not selections:
            return None
        return str(selections[0].fragment.fragment_id)

    def exit_fragment_id(plan: object) -> str | None:
        selections = tuple(getattr(plan, "selections"))
        if not selections:
            return None
        return str(selections[-1].fragment.fragment_id)

    seen_entries = {entry_fragment_id(plan) for plan in selected}
    seen_exits = {exit_fragment_id(plan) for plan in selected}
    criteria = (
        (
            lambda plan: entry_fragment_id(plan) not in seen_entries
            and exit_fragment_id(plan) not in seen_exits
        ),
        (lambda plan: exit_fragment_id(plan) not in seen_exits),
        (lambda plan: entry_fragment_id(plan) not in seen_entries),
    ) if diversify_exit else (
        lambda plan: entry_fragment_id(plan) not in seen_entries,
    )
    for criterion in criteria:
        for plan in ordered[prefix_count:]:
            if id(plan) in selected_ids or not criterion(plan):
                continue
            selected.append(plan)
            selected_ids.add(id(plan))
            seen_entries.add(entry_fragment_id(plan))
            seen_exits.add(exit_fragment_id(plan))
            if len(selected) == maximum_plans:
                return tuple(selected)
    for plan in ordered[prefix_count:]:
        if id(plan) in selected_ids:
            continue
        selected.append(plan)
        if len(selected) == maximum_plans:
            break
    return tuple(selected)


def _plan_identity(plan: object, rank: int) -> dict[str, object]:
    selections = tuple(getattr(plan, "selections"))
    return {
        "rank": int(rank),
        "total_cost": float(getattr(plan, "total_cost")),
        "clip_indices": [
            int(selection.fragment.archive_clip_index)
            for selection in selections
        ],
        "fragment_ids": [
            str(selection.fragment.fragment_id)
            for selection in selections
        ],
    }


def try_ranked_reconstructions(
    plans: Sequence[object],
    reconstruct: Callable[[object], object],
    *,
    validate_reconstruction: Callable[[object], None] | None = None,
    report_rejection: Callable[[dict[str, object]], None] | None = None,
) -> RankedReconstructionOutcome:
    """Try ranked plans in order, including optional downstream validation."""

    rejected: list[dict[str, object]] = []
    for rank, plan in enumerate(plans):
        try:
            reconstruction = reconstruct(plan)
            if validate_reconstruction is not None:
                validate_reconstruction(reconstruction)
        except PlannedFragmentReconstructionRejected as error:
            record = {
                **_plan_identity(plan, rank),
                "stage": error.stage,
                "message": error.message,
            }
            # Keep the public record ordering compact and predictable.
            record = {
                "rank": record["rank"],
                "total_cost": record["total_cost"],
                "stage": record["stage"],
                "message": record["message"],
                "clip_indices": record["clip_indices"],
                "fragment_ids": record["fragment_ids"],
            }
            rejected.append(record)
            if report_rejection is not None:
                report_rejection(record)
            continue
        return RankedReconstructionOutcome(
            selected_rank=rank,
            selected_plan=plan,
            reconstruction=reconstruction,
            rejected_plans=tuple(rejected),
        )
    return RankedReconstructionOutcome(
        selected_rank=None,
        selected_plan=None,
        reconstruction=None,
        rejected_plans=tuple(rejected),
    )


def collect_ranked_reconstructions(
    plans: Sequence[object],
    reconstruct: Callable[[object], object],
    *,
    maximum_accepted: int,
    validate_reconstruction: Callable[[object], None] | None = None,
    report_rejection: Callable[[dict[str, object]], None] | None = None,
) -> tuple[
    tuple[AcceptedRankedReconstruction, ...],
    tuple[dict[str, object], ...],
]:
    """Continue ranked search until several honest passing variants exist."""

    if maximum_accepted < 1:
        raise ValueError("maximum accepted reconstructions must be positive")
    accepted: list[AcceptedRankedReconstruction] = []
    rejected: list[dict[str, object]] = []
    for rank, plan in enumerate(plans):
        try:
            reconstruction = reconstruct(plan)
            if validate_reconstruction is not None:
                validate_reconstruction(reconstruction)
        except PlannedFragmentReconstructionRejected as error:
            identity = _plan_identity(plan, rank)
            record = {
                "rank": identity["rank"],
                "total_cost": identity["total_cost"],
                "stage": error.stage,
                "message": error.message,
                "clip_indices": identity["clip_indices"],
                "fragment_ids": identity["fragment_ids"],
            }
            rejected.append(record)
            if report_rejection is not None:
                report_rejection(record)
            continue
        accepted.append(
            AcceptedRankedReconstruction(rank, plan, reconstruction)
        )
        if len(accepted) == maximum_accepted:
            break
    return tuple(accepted), tuple(rejected)


def _motion_arrays(reconstruction: object) -> dict[str, np.ndarray]:
    motion = reconstruction.motion
    provenance = tuple(motion.provenance)
    return {
        "fps": np.asarray(float(motion.fps), dtype=np.float32),
        "root_position_world": np.asarray(motion.root_position_world),
        "root_quaternion_world_wxyz": np.asarray(
            motion.root_quaternion_world_wxyz
        ),
        "joint_position": np.asarray(motion.joint_position),
        "seam_indices": np.asarray(motion.seam_indices, dtype=np.int64),
        "source_archive_clip_index": np.asarray(
            [value.archive_clip_index for value in provenance],
            dtype=np.int64,
        ),
        "source_frame": np.asarray(
            [value.source_frame for value in provenance], dtype=np.int64
        ),
        "source_clip_id": np.asarray(
            [str(value.clip_id) for value in provenance], dtype=np.str_
        ),
    }


def _diagnostic_arrays(reconstruction: object) -> dict[str, np.ndarray]:
    return {
        "source_transition_fractions": np.asarray(
            reconstruction.source_transition_fractions, dtype=np.float64
        ),
        "source_switch_indices": np.asarray(
            reconstruction.source_switch_indices, dtype=np.int64
        ),
        "per_frame_joint_correction_rad": np.asarray(
            reconstruction.per_frame_joint_correction_rad
        ),
        "per_frame_foot_target_error_m": np.asarray(
            reconstruction.per_frame_foot_target_error_m
        ),
        "per_frame_foot_target_error_by_foot_m": np.asarray(
            reconstruction.per_frame_foot_target_error_by_foot_m
        ),
        "per_frame_minimum_sole_clearance_m": np.asarray(
            reconstruction.per_frame_minimum_sole_clearance_m
        ),
        "per_frame_triangle_sphere_penetration_m": np.asarray(
            reconstruction.per_frame_triangle_sphere_penetration_m
        ),
        "per_frame_root_clearance_lift_m": np.asarray(
            reconstruction.per_frame_root_clearance_lift_m
        ),
        "per_frame_swing_foot_clearance_lift_m": np.asarray(
            reconstruction.per_frame_swing_foot_clearance_lift_m
        ),
        "per_frame_swing_route_adjustment": np.asarray(
            reconstruction.per_frame_swing_route_adjustment
        ),
        "per_frame_stance_support_point_count": np.asarray(
            reconstruction.per_frame_stance_support_point_count
        ),
        "per_frame_sole_target_world_xyz": np.asarray(
            reconstruction.per_frame_sole_target_world_xyz
        ),
        "per_frame_stance_mask": np.asarray(
            reconstruction.per_frame_stance_mask
        ),
    }


def write_reconstruction_bundle(
    output: str | Path,
    reconstruction: object,
    summary: dict[str, object],
) -> Path:
    """Stage and publish the three reconstruction artifacts as one directory."""

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        np.savez_compressed(stage / "motion.npz", **_motion_arrays(reconstruction))
        np.savez_compressed(
            stage / "diagnostics.npz",
            **_diagnostic_arrays(reconstruction),
        )
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return destination


def write_rejection_bundle(
    output: str | Path,
    summary: dict[str, object],
) -> Path:
    """Publish a failed search summary so expensive diagnostics are retained."""

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return destination


def _route_summary(route: object) -> dict[str, object]:
    return {
        "start_xy": np.asarray(route.start_xy, dtype=float).tolist(),
        "end_xy": np.asarray(route.end_xy, dtype=float).tolist(),
        "levels": [
            {
                "height_m": float(level.height_m),
                "route_start_distance_m": float(
                    level.route_start_distance_m
                ),
                "route_stop_distance_m": float(
                    level.route_stop_distance_m
                ),
                "visible_in_mesh": bool(level.visible_in_mesh),
                "has_left_foothold": (
                    level.left_foothold_center_xy is not None
                ),
                "has_right_foothold": (
                    level.right_foothold_center_xy is not None
                ),
            }
            for level in route.levels
        ],
    }


def _selection_summary(plan: object) -> dict[str, object]:
    result = _plan_identity(plan, rank=0)
    result.pop("rank")
    result.update(
        {
            "total_geometry_cost": float(plan.total_geometry_cost),
            "total_seam_cost": float(plan.total_seam_cost),
            "selections": [
                {
                    "transition_index": int(
                        selection.target.transition_index
                    ),
                    "fragment_id": str(selection.fragment.fragment_id),
                    "archive_clip_index": int(
                        selection.fragment.archive_clip_index
                    ),
                    "source_range": [
                        int(selection.fragment.source_start_frame),
                        int(selection.fragment.source_stop_frame),
                    ],
                    "role": str(selection.fragment.role.value),
                    "geometry_cost": float(selection.geometry_cost),
                    "seam_cost": float(selection.seam_cost),
                }
                for selection in plan.selections
            ],
        }
    )
    return result


def _metric_summary(reconstruction: object) -> dict[str, object]:
    names = (
        "maximum_joint_correction_rad",
        "maximum_foot_target_error_m",
        "maximum_stance_foot_target_error_m",
        "maximum_swing_foot_target_error_m",
        "maximum_sole_penetration_m",
        "maximum_triangle_sphere_penetration_m",
        "minimum_sole_clearance_m",
        "maximum_root_clearance_lift_m",
        "maximum_swing_foot_clearance_lift_m",
        "maximum_root_route_adjustment_m",
        "maximum_root_route_lateral_deviation_m",
        "minimum_stance_support_point_count",
        "maximum_root_translation_step_m",
        "maximum_root_rotation_step_rad",
        "maximum_root_acceleration_m_s2",
        "maximum_joint_step_rad",
        "maximum_fragment_seam_joint_step_rad",
        "maximum_source_switch_joint_step_rad",
    )
    return {
        name: (
            int(getattr(reconstruction, name))
            if name == "minimum_stance_support_point_count"
            else float(getattr(reconstruction, name))
        )
        for name in names
    }


def _jsonable_arguments(arguments: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in vars(arguments).items():
        if isinstance(value, Path):
            result[name] = str(value)
        elif isinstance(value, tuple):
            result[name] = list(value)
        else:
            result[name] = value
    return result


def _print_rejection(record: dict[str, object]) -> None:
    print(
        json.dumps({"event": "plan_rejected", **record}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _effective_foothold_route_length(
    route_length_m: float,
    *,
    absolute_length_m: float | None,
    extension_m: float,
) -> float:
    """Resolve an optional absolute foothold horizon against a route extension."""

    if absolute_length_m is None:
        if extension_m < 0.0:
            raise ValueError("foothold route extension must be nonnegative")
        return float(route_length_m + extension_m)
    if absolute_length_m < route_length_m:
        raise ValueError(
            "foothold route length must cover the requested target route"
        )
    return float(absolute_length_m)


def run(
    arguments: argparse.Namespace,
    *,
    target_mesh: TerrainMeshIndex | None = None,
    excluded_source_clip_indices: Collection[int] = (),
    validate_reconstruction: Callable[[object], None] | None = None,
    accepted_reconstruction_callback: (
        Callable[[AcceptedRankedReconstruction], None] | None
    ) = None,
    maximum_accepted_reconstructions: int = 1,
) -> tuple[dict[str, object], object | None]:
    """Run the exact-bank global planner and ranked mechanical reconstruction."""

    import zarr

    archive_path = arguments.archive.expanduser().resolve()
    bank_path = arguments.fragment_bank.expanduser().resolve()
    model_path = arguments.model.expanduser().resolve()
    archive = zarr.open_group(str(archive_path), mode="r")
    target_clip_index = getattr(arguments, "target_clip_index", None)
    if target_mesh is None:
        if target_clip_index is None:
            raise ValueError("target_mesh or target_clip_index is required")
        target_mesh = _archive_terrain_index(archive, int(target_clip_index))
    elif not isinstance(target_mesh, TerrainMeshIndex):
        raise TypeError("target_mesh must be a TerrainMeshIndex")
    start = np.asarray(arguments.start_xy, dtype=np.float64)
    end = np.asarray(arguments.end_xy, dtype=np.float64)
    direction = end - start
    route_length = float(np.linalg.norm(direction))
    if route_length <= 0.0:
        raise ValueError("target route start and end must differ")
    direction /= route_length
    foothold_route_length = _effective_foothold_route_length(
        route_length,
        absolute_length_m=arguments.foothold_route_length_m,
        extension_m=arguments.foothold_route_extension_m,
    )
    target_route = sample_stair_support_route(
        target_mesh,
        start,
        end,
        arguments.ground_fallback_height_m,
        arguments.sole_half_length_m,
        arguments.sole_half_width_m,
        sample_spacing_m=arguments.route_sample_spacing_m,
    )
    foothold_route = sample_stair_support_route(
        target_mesh,
        start,
        start + foothold_route_length * direction,
        arguments.ground_fallback_height_m,
        arguments.sole_half_length_m,
        arguments.sole_half_width_m,
        sample_spacing_m=arguments.route_sample_spacing_m,
    )

    raw_fragments = load_stair_fragment_records_jsonl(bank_path)
    precomputed_exact = _precomputed_exact_bank_metadata(
        bank_path,
        len(raw_fragments),
        archive_path=archive_path,
    )
    if arguments.enrich_bank_from_source_mesh:
        exact_bank = enrich_stair_fragment_bank_from_exact_mesh(
            archive_path,
            raw_fragments,
            ground_fallback_height_m=arguments.ground_fallback_height_m,
            sole_half_length_m=arguments.sole_half_length_m,
            sole_half_width_m=arguments.sole_half_width_m,
            route_sample_spacing_m=arguments.route_sample_spacing_m,
        )
    elif precomputed_exact is not None:
        exact_bank = ExactStairFragmentBank(
            records=raw_fragments,
            rejected=tuple(
                (str(value[0]), str(value[1]))
                for value in precomputed_exact.get("rejected", ())
            ),
        )
    else:
        raise ValueError(
            "fragment bank has no matching exact-geometry sidecar; "
            "use --enrich-bank-from-source-mesh"
        )
    transition_config = TransitionConfig(
        max_root_height_delta_m=arguments.seam_max_root_height_delta_m,
        max_root_tilt_delta_rad=arguments.seam_max_root_tilt_delta_rad,
        max_joint_position_rmse_rad=(
            arguments.seam_max_joint_position_rmse_rad
        ),
        max_root_linear_velocity_delta_m_s=(
            arguments.seam_max_root_linear_velocity_delta_m_s
        ),
        max_root_angular_velocity_delta_rad_s=(
            arguments.seam_max_root_angular_velocity_delta_rad_s
        ),
        max_joint_velocity_rmse_rad_s=(
            arguments.seam_max_joint_velocity_rmse_rad_s
        ),
        max_support_sole_delta_m=arguments.seam_max_support_sole_delta_m,
        max_same_support_gap_s=arguments.seam_max_same_support_gap_s,
        max_neighbors_per_fragment=arguments.seam_max_neighbors_per_fragment,
        safe_stop_max_planar_speed_m_s=(
            arguments.seam_safe_stop_max_planar_speed_m_s
        ),
        safe_stop_max_yaw_rate_rad_s=(
            arguments.seam_safe_stop_max_yaw_rate_rad_s
        ),
        pose_cost_weight=arguments.seam_pose_cost_weight,
        velocity_cost_weight=arguments.seam_velocity_cost_weight,
        sole_cost_weight=arguments.seam_sole_cost_weight,
    )
    excluded_sources = {
        int(value) for value in excluded_source_clip_indices
    }
    if not arguments.include_target_clip and target_clip_index is not None:
        excluded_sources.add(int(target_clip_index))
    plan_config = StairFragmentSequenceConfig(
        max_rise_error_m=arguments.max_rise_error_m,
        max_run_error_m=arguments.max_run_error_m,
        geometry_cost_weight=arguments.geometry_cost_weight,
        seam_cost_weight=arguments.seam_cost_weight,
        require_boundary_roles=not arguments.allow_nonboundary_roles,
        excluded_archive_clip_indices=frozenset(excluded_sources),
        transition_config=transition_config,
    )
    planner_pool_plans = (
        arguments.maximum_plans
        if arguments.planner_pool_plans is None
        else arguments.planner_pool_plans
    )
    if planner_pool_plans < arguments.maximum_plans:
        raise ValueError("planner pool must cover all reconstruction trials")
    plan_pool = plan_stair_fragment_sequences(
        target_route,
        exact_bank.records,
        plan_config,
        maximum_plans=planner_pool_plans,
        maximum_paths_per_layer=arguments.beam_width,
    )
    plans = select_entry_diverse_plans(
        plan_pool,
        maximum_plans=arguments.maximum_plans,
        preserve_ranked_plans=arguments.preserve_ranked_plans,
        diversify_exit=arguments.diversify_boundaries,
    )
    anchor_config = FootholdAnchorConfig(
        max_longitudinal_adjustment_m=(
            arguments.anchor_max_longitudinal_adjustment_m
        ),
        max_lateral_adjustment_m=(
            arguments.anchor_max_lateral_adjustment_m
        ),
        max_yaw_adjustment_rad=arguments.anchor_max_yaw_adjustment_rad,
        max_vertical_adjustment_m=(
            arguments.anchor_max_vertical_adjustment_m
        ),
        longitudinal_samples=arguments.anchor_longitudinal_samples,
        lateral_samples=arguments.anchor_lateral_samples,
        yaw_samples=arguments.anchor_yaw_samples,
        support_height_tolerance_m=(
            arguments.anchor_support_height_tolerance_m
        ),
        collision_clearance_m=arguments.anchor_collision_clearance_m,
        minimum_support_points=arguments.anchor_minimum_support_points,
        lateral_seed_offsets_m=arguments.anchor_lateral_seed_offsets_m,
        route_lateral_offset_m=arguments.anchor_route_lateral_offset_m,
        route_alignment_weight=arguments.anchor_route_alignment_weight,
        tread_edge_clearance_m=arguments.anchor_tread_edge_clearance_m,
    )

    def reconstruct(plan: object) -> object:
        return reconstruct_stair_fragment_sequence(
            plan,
            target_route,
            archive_path=archive_path,
            model_path=model_path,
            target_clip_index=target_clip_index,
            target_mesh=target_mesh,
            excluded_source_clip_indices=excluded_sources,
            decay_frames=arguments.decay_frames,
            ground_fallback_height_m=arguments.ground_fallback_height_m,
            sole_half_length_m=arguments.sole_half_length_m,
            sole_half_width_m=arguments.sole_half_width_m,
            route_sample_spacing_m=arguments.route_sample_spacing_m,
            maximum_joint_correction_rad=(
                arguments.maximum_joint_correction_rad
            ),
            maximum_foot_target_error_m=(
                arguments.maximum_foot_target_error_m
            ),
            maximum_swing_foot_target_error_m=(
                arguments.maximum_swing_foot_target_error_m
            ),
            maximum_sole_penetration_m=(
                arguments.maximum_sole_penetration_m
            ),
            maximum_root_clearance_lift_m=(
                arguments.maximum_root_clearance_lift_m
            ),
            maximum_swing_foot_clearance_lift_m=(
                arguments.maximum_swing_foot_clearance_lift_m
            ),
            minimum_stance_support_points=(
                arguments.minimum_stance_support_points
            ),
            support_contact_tolerance_m=(
                arguments.support_contact_tolerance_m
            ),
            maximum_source_switch_joint_step_rad=(
                arguments.maximum_source_switch_joint_step_rad
            ),
            maximum_fragment_seam_joint_step_rad=(
                arguments.maximum_fragment_seam_joint_step_rad
            ),
            maximum_root_translation_step_m=(
                arguments.maximum_root_translation_step_m
            ),
            maximum_root_rotation_step_rad=(
                arguments.maximum_root_rotation_step_rad
            ),
            maximum_root_acceleration_m_s2=(
                arguments.maximum_root_acceleration_m_s2
            ),
            maximum_interframe_joint_step_rad=(
                arguments.maximum_interframe_joint_step_rad
            ),
            foothold_route=foothold_route,
            foothold_anchor_config=anchor_config,
            swing_anchor_blend_frames=arguments.swing_anchor_blend_frames,
            swing_anchor_release_frames=arguments.swing_anchor_release_frames,
            minimum_swing_foot_clearance_m=(
                arguments.minimum_swing_foot_clearance_m
            ),
            maximum_swing_planar_deviation_m=(
                arguments.maximum_swing_planar_deviation_m
            ),
            root_anchor_smoothing_frames=(
                arguments.root_anchor_smoothing_frames
            ),
            allow_target_motion_clip=arguments.include_target_clip,
            temporal_ik_warm_start=arguments.temporal_ik_warm_start,
            maximum_swing_leg_joint_step_rad=(
                arguments.maximum_swing_leg_joint_step_rad
            ),
        )

    if accepted_reconstruction_callback is None:
        outcome = try_ranked_reconstructions(
            plans,
            reconstruct,
            validate_reconstruction=validate_reconstruction,
            report_rejection=_print_rejection,
        )
    else:
        accepted, rejected = collect_ranked_reconstructions(
            plans,
            reconstruct,
            maximum_accepted=maximum_accepted_reconstructions,
            validate_reconstruction=validate_reconstruction,
            report_rejection=_print_rejection,
        )
        for candidate in accepted:
            accepted_reconstruction_callback(candidate)
        first = None if not accepted else accepted[0]
        outcome = RankedReconstructionOutcome(
            selected_rank=None if first is None else first.rank,
            selected_plan=None if first is None else first.plan,
            reconstruction=None if first is None else first.reconstruction,
            rejected_plans=rejected,
        )
    common: dict[str, object] = {
        "schema": "stair-fragment-route-reconstruction/v1",
        "status": (
            "passed" if outcome.reconstruction is not None else "rejected"
        ),
        "archive_path": str(archive_path),
        "fragment_bank_path": str(bank_path),
        "model_path": str(model_path),
        "target_clip_index": (
            None if target_clip_index is None else int(target_clip_index)
        ),
        "excluded_source_clip_indices": sorted(excluded_sources),
        "raw_fragment_count": len(raw_fragments),
        "exact_fragment_count": len(exact_bank.records),
        "fragment_bank_mode": (
            "source_mesh_enriched"
            if arguments.enrich_bank_from_source_mesh
            else "loaded_directly"
        ),
        "precomputed_exact_bank_reused": (
            precomputed_exact is not None
            and not arguments.enrich_bank_from_source_mesh
        ),
        "exact_fragment_rejections": [
            {"fragment_id": fragment_id, "message": message}
            for fragment_id, message in exact_bank.rejected
        ],
        "ranked_plan_count": len(plans),
        "rejected_plans": list(outcome.rejected_plans),
        "effective_foothold_route_length_m": foothold_route_length,
        "target_route": _route_summary(target_route),
        "foothold_route": _route_summary(foothold_route),
        "configuration": _jsonable_arguments(arguments),
    }
    if outcome.reconstruction is None:
        return common, None
    assert outcome.selected_rank is not None
    assert outcome.selected_plan is not None
    reconstruction = outcome.reconstruction
    common.update(
        {
            "selected_rank": outcome.selected_rank,
            "selected_plan": _selection_summary(outcome.selected_plan),
            "frame_count": len(reconstruction.motion.root_position_world),
            "fps": float(reconstruction.motion.fps),
            "metrics": _metric_summary(reconstruction),
            "foothold_anchor_diagnostics": [
                asdict(value)
                for value in reconstruction.foothold_anchor_diagnostics
            ],
            "artifacts": {
                "motion": "motion.npz",
                "diagnostics": "diagnostics.npz",
                "summary": "summary.json",
            },
        }
    )
    return common, reconstruction


def _floats(value: str) -> tuple[float, ...]:
    result = tuple(
        float(part) for part in value.split(",") if part.strip()
    )
    if not result:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--fragment-bank", type=Path, default=DEFAULT_FRAGMENT_BANK
    )
    parser.add_argument(
        "--enrich-bank-from-source-mesh",
        action="store_true",
        help=(
            "remeasure a raw fragment bank against each source mesh; "
            "the default all500-exact bank is loaded directly"
        ),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-clip-index", type=int, required=True)
    parser.add_argument(
        "--include-target-clip",
        action="store_true",
        help=(
            "allow exact source motion from the target terrain; the default "
            "retains the stronger leave-one-target-out evaluation"
        ),
    )
    parser.add_argument(
        "--start-xy", type=float, nargs=2, required=True, metavar=("X", "Y")
    )
    parser.add_argument(
        "--end-xy", type=float, nargs=2, required=True, metavar=("X", "Y")
    )
    parser.add_argument("--foothold-route-length-m", type=float, default=None)
    parser.add_argument(
        "--foothold-route-extension-m", type=float, default=0.25
    )
    parser.add_argument("--maximum-plans", type=int, default=128)
    parser.add_argument("--planner-pool-plans", type=int, default=None)
    parser.add_argument("--preserve-ranked-plans", type=int, default=8)
    parser.add_argument("--diversify-boundaries", action="store_true")
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-rise-error-m", type=float, default=0.08)
    parser.add_argument("--max-run-error-m", type=float, default=0.15)
    parser.add_argument("--geometry-cost-weight", type=float, default=1.0)
    parser.add_argument("--seam-cost-weight", type=float, default=1.0)
    parser.add_argument("--allow-nonboundary-roles", action="store_true")

    seam = parser.add_argument_group("motion-graph seam thresholds")
    seam.add_argument("--seam-max-root-height-delta-m", type=float, default=0.18)
    seam.add_argument("--seam-max-root-tilt-delta-rad", type=float, default=0.35)
    seam.add_argument(
        "--seam-max-joint-position-rmse-rad", type=float, default=0.45
    )
    seam.add_argument(
        "--seam-max-root-linear-velocity-delta-m-s",
        type=float,
        default=0.9,
    )
    seam.add_argument(
        "--seam-max-root-angular-velocity-delta-rad-s",
        type=float,
        default=2.0,
    )
    seam.add_argument(
        "--seam-max-joint-velocity-rmse-rad-s", type=float, default=4.0
    )
    seam.add_argument(
        "--seam-max-support-sole-delta-m", type=float, default=0.18
    )
    seam.add_argument(
        "--seam-max-same-support-gap-s", type=float, default=0.8
    )
    seam.add_argument(
        "--seam-max-neighbors-per-fragment", type=int, default=32
    )
    seam.add_argument(
        "--seam-safe-stop-max-planar-speed-m-s", type=float, default=0.25
    )
    seam.add_argument(
        "--seam-safe-stop-max-yaw-rate-rad-s", type=float, default=0.5
    )
    seam.add_argument("--seam-pose-cost-weight", type=float, default=1.0)
    seam.add_argument("--seam-velocity-cost-weight", type=float, default=0.7)
    seam.add_argument("--seam-sole-cost-weight", type=float, default=1.2)

    reconstruction = parser.add_argument_group("reconstruction thresholds")
    reconstruction.add_argument("--decay-frames", type=float, default=8.0)
    reconstruction.add_argument(
        "--ground-fallback-height-m", type=float, default=0.0
    )
    reconstruction.add_argument(
        "--sole-half-length-m", type=float, default=0.10
    )
    reconstruction.add_argument(
        "--sole-half-width-m", type=float, default=0.055
    )
    reconstruction.add_argument(
        "--route-sample-spacing-m", type=float, default=0.01
    )
    reconstruction.add_argument(
        "--maximum-joint-correction-rad", type=float, default=0.55
    )
    reconstruction.add_argument(
        "--maximum-foot-target-error-m", type=float, default=0.003
    )
    reconstruction.add_argument(
        "--maximum-swing-foot-target-error-m", type=float, default=0.05
    )
    reconstruction.add_argument(
        "--maximum-sole-penetration-m", type=float, default=0.005
    )
    reconstruction.add_argument(
        "--maximum-root-clearance-lift-m", type=float, default=0.005
    )
    reconstruction.add_argument(
        "--maximum-swing-foot-clearance-lift-m",
        type=float,
        default=0.25,
    )
    reconstruction.add_argument(
        "--minimum-stance-support-points", type=int, default=1
    )
    reconstruction.add_argument(
        "--support-contact-tolerance-m", type=float, default=0.02
    )
    reconstruction.add_argument(
        "--maximum-source-switch-joint-step-rad",
        type=float,
        default=0.20,
    )
    reconstruction.add_argument(
        "--maximum-fragment-seam-joint-step-rad",
        type=float,
        default=0.20,
    )
    reconstruction.add_argument(
        "--maximum-root-translation-step-m", type=float, default=0.04
    )
    reconstruction.add_argument(
        "--maximum-root-rotation-step-rad",
        type=float,
        default=math.radians(5.0),
    )
    reconstruction.add_argument(
        "--maximum-root-acceleration-m-s2",
        type=float,
        default=25.0,
    )
    reconstruction.add_argument(
        "--maximum-interframe-joint-step-rad",
        type=float,
        default=0.25,
    )
    reconstruction.add_argument(
        "--swing-anchor-blend-frames", type=int, default=48
    )
    reconstruction.add_argument(
        "--swing-anchor-release-frames", type=int, default=8
    )
    reconstruction.add_argument(
        "--minimum-swing-foot-clearance-m", type=float, default=0.08
    )
    reconstruction.add_argument(
        "--maximum-swing-planar-deviation-m", type=float, default=0.12
    )
    reconstruction.add_argument(
        "--root-anchor-smoothing-frames", type=float, default=8.0
    )
    reconstruction.add_argument(
        "--temporal-ik-warm-start",
        action="store_true",
        help=(
            "initialize each frame's leg IK from the preceding adapted pose "
            "while retaining the current authored pose as its regularizer"
        ),
    )
    reconstruction.add_argument(
        "--maximum-swing-leg-joint-step-rad",
        type=float,
        default=None,
        help=(
            "bound each unplanted leg joint relative to the preceding "
            "adapted frame; stance legs remain unconstrained"
        ),
    )
    reconstruction.add_argument(
        "--maximum-full-foot-penetration-m",
        type=float,
        default=None,
        help=(
            "optionally reject ranked plans using the complete rendered foot "
            "mesh instead of only the compact sole probes"
        ),
    )
    reconstruction.add_argument(
        "--maximum-forbidden-body-penetration-m",
        type=float,
        default=0.0,
    )

    anchor = parser.add_argument_group("foothold anchor search")
    anchor.add_argument(
        "--anchor-max-longitudinal-adjustment-m",
        type=float,
        default=0.45,
    )
    anchor.add_argument(
        "--anchor-max-lateral-adjustment-m", type=float, default=0.60
    )
    anchor.add_argument(
        "--anchor-max-yaw-adjustment-rad", type=float, default=0.0
    )
    anchor.add_argument(
        "--anchor-max-vertical-adjustment-m", type=float, default=0.08
    )
    anchor.add_argument(
        "--anchor-longitudinal-samples", type=int, default=13
    )
    anchor.add_argument("--anchor-lateral-samples", type=int, default=7)
    anchor.add_argument("--anchor-yaw-samples", type=int, default=1)
    anchor.add_argument(
        "--anchor-support-height-tolerance-m",
        type=float,
        default=0.015,
    )
    anchor.add_argument(
        "--anchor-collision-clearance-m", type=float, default=0.0025
    )
    anchor.add_argument(
        "--anchor-minimum-support-points", type=int, default=1
    )
    anchor.add_argument(
        "--anchor-lateral-seed-offsets-m",
        type=_floats,
        default=(-0.11, 0.0, 0.11),
    )
    anchor.add_argument(
        "--anchor-route-lateral-offset-m", type=float, default=0.11
    )
    anchor.add_argument(
        "--anchor-route-alignment-weight", type=float, default=10.0
    )
    anchor.add_argument(
        "--anchor-tread-edge-clearance-m", type=float, default=0.01
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    accepted_audits: dict[int, object] = {}
    validate_reconstruction = None
    if arguments.maximum_full_foot_penetration_m is not None:
        from .terrain_oracle.stair_motion_collision_audit import (
            audit_stair_motion_collisions,
        )

        def validate_reconstruction(reconstruction: object) -> None:
            audit = audit_stair_motion_collisions(
                reconstruction.motion,
                archive_path=arguments.archive,
                target_clip_index=arguments.target_clip_index,
                model_path=arguments.model,
                maximum_foot_penetration_m=(
                    arguments.maximum_full_foot_penetration_m
                ),
                maximum_forbidden_body_penetration_m=(
                    arguments.maximum_forbidden_body_penetration_m
                ),
            )
            accepted_audits[id(reconstruction)] = audit
            if not audit.accepted:
                raise PlannedFragmentReconstructionRejected(
                    "full_body_collision",
                    (
                        "complete-foot penetration "
                        f"{audit.maximum_foot_penetration_m:.6f} m; "
                        "forbidden-body penetration "
                        f"{audit.maximum_forbidden_body_penetration_m:.6f} m"
                    ),
                )

    summary, reconstruction = run(
        arguments, validate_reconstruction=validate_reconstruction
    )
    if reconstruction is None:
        destination = write_rejection_bundle(arguments.output, summary)
        print(
            json.dumps(
                {"status": "rejected", "output": str(destination)},
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    audit = accepted_audits.get(id(reconstruction))
    if audit is not None:
        summary["full_body_collision_audit"] = audit.to_dict()
    destination = write_reconstruction_bundle(
        arguments.output,
        reconstruction,
        summary,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(destination),
                "selected_rank": summary["selected_rank"],
                "rejected_plan_count": len(summary["rejected_plans"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "AcceptedRankedReconstruction",
    "collect_ranked_reconstructions",
    "RankedReconstructionOutcome",
    "main",
    "run",
    "try_ranked_reconstructions",
    "write_rejection_bundle",
    "write_reconstruction_bundle",
)
