"""Reconstruct planned stair fragments onto an exact target support route."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Collection, Mapping, Sequence

import numpy as np

from .contact import CanonicalMeshQuery
from .fragments import SupportPhase
from .reference_stitch import _G1FootfallAdapter
from .stair_foothold_anchors import (
    FootholdAnchorConfig,
    FootholdAnchorDiagnostics,
    FootholdAnchorRejected,
    NominalFootSolePose,
    StanceSpan,
    _foot_collision_envelope_points,
    anchor_planted_foot,
)
from .stair_fragment_sequence import StairFragmentSequencePlan
from .stair_geometry_warp import _archive_terrain_index, _level_boundaries, build_stair_geometry_warp
from .stair_support_route import (
    StairSupportRoute,
    VisibleTread,
    sample_stair_support_route,
)
from .stairs500_fragments import StairFragmentRecord
from .stitch import FragmentSelection, StitchedMotion, stitch_archive
from .terrain_mesh import TerrainMeshIndex


class PlannedFragmentReconstructionRejected(ValueError):
    """A reconstruction failure with the pipeline stage that rejected it."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = str(stage)
        self.message = str(message)
        super().__init__(f"{self.stage}: {self.message}")


@dataclass(frozen=True)
class StairFragmentReconstructionResult:
    """One continuous authored reconstruction and its target-mesh audit."""

    motion: StitchedMotion
    source_route: StairSupportRoute
    target_route: StairSupportRoute
    source_transition_fractions: tuple[float, ...]
    source_switch_indices: tuple[int, ...]
    maximum_joint_correction_rad: float
    maximum_foot_target_error_m: float
    maximum_stance_foot_target_error_m: float
    maximum_swing_foot_target_error_m: float
    maximum_sole_penetration_m: float
    maximum_triangle_sphere_penetration_m: float
    minimum_sole_clearance_m: float
    maximum_root_clearance_lift_m: float
    maximum_swing_foot_clearance_lift_m: float
    maximum_root_route_adjustment_m: float
    maximum_root_route_lateral_deviation_m: float
    minimum_stance_support_point_count: int
    maximum_root_translation_step_m: float
    maximum_root_rotation_step_rad: float
    maximum_root_acceleration_m_s2: float
    maximum_joint_step_rad: float
    maximum_fragment_seam_joint_step_rad: float
    maximum_source_switch_joint_step_rad: float
    per_frame_joint_correction_rad: np.ndarray
    per_frame_foot_target_error_m: np.ndarray
    per_frame_foot_target_error_by_foot_m: np.ndarray
    per_frame_minimum_sole_clearance_m: np.ndarray
    per_frame_triangle_sphere_penetration_m: np.ndarray
    per_frame_root_clearance_lift_m: np.ndarray
    per_frame_swing_foot_clearance_lift_m: np.ndarray
    per_frame_swing_route_adjustment: np.ndarray
    per_frame_stance_support_point_count: np.ndarray
    per_frame_sole_target_world_xyz: np.ndarray
    per_frame_stance_mask: np.ndarray
    foothold_anchor_diagnostics: tuple[FootholdAnchorDiagnostics, ...]


@dataclass(frozen=True)
class ExactStairFragmentGeometry:
    """Source geometry measured from the authored fragment's paired mesh."""

    transition_fraction: float
    base_height_m: float
    signed_rise_m: float
    route_length_m: float


@dataclass(frozen=True)
class ExactStairFragmentBank:
    """Planner-ready records plus fragments that lacked one measurable riser."""

    records: tuple[StairFragmentRecord, ...]
    rejected: tuple[tuple[str, str], ...]


def build_synthetic_source_support_route(
    motion: StitchedMotion,
    *,
    signed_rises_m: Sequence[float],
    transition_fractions: Sequence[float],
    base_height_m: float,
) -> StairSupportRoute:
    """Build one source support level per selected fragment transition.

    ``transition_fractions`` locate the one riser inside each stitched source
    span.  They are measured against the corresponding source mesh, while the
    seam indices identify that span after authored fragments have been joined.
    """

    rises = tuple(float(value) for value in signed_rises_m)
    fractions = tuple(float(value) for value in transition_fractions)
    if not rises:
        raise ValueError("synthetic source route needs at least one riser")
    if len(rises) != len(fractions):
        raise ValueError("source rise and transition-fraction counts differ")
    if len(motion.seam_indices) != len(rises) - 1:
        raise ValueError("stitched seams do not partition selected fragments")
    if len(motion.root_position_world) < len(rises) + 1:
        raise ValueError("stitched motion is too short for selected fragments")
    if any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("source transition fractions must lie in [0, 1]")

    root_xy = np.asarray(motion.root_position_world[:, :2], dtype=np.float64)
    start_xy = root_xy[0]
    first_span_stop = (
        int(motion.seam_indices[0])
        if motion.seam_indices
        else len(root_xy)
    )
    direction_vector = root_xy[first_span_stop - 1] - start_xy
    direction_norm = float(np.linalg.norm(direction_vector))
    if direction_norm <= 1.0e-9:
        raise ValueError("stitched source route has zero planar length")
    direction = direction_vector / direction_norm
    coordinates = (root_xy - start_xy) @ direction
    route_length = float(coordinates[-1])
    if route_length <= 1.0e-9:
        raise ValueError("stitched source route makes no forward progress")
    end_xy = start_xy + route_length * direction

    starts = (0,) + tuple(int(value) for value in motion.seam_indices)
    stops = tuple(int(value) for value in motion.seam_indices) + (
        len(root_xy),
    )
    boundaries = [0.0]
    for start, stop, fraction in zip(starts, stops, fractions, strict=True):
        if not 0 <= start < stop <= len(root_xy):
            raise ValueError("stitched seam is outside the motion frame range")
        span_start = float(coordinates[start])
        span_stop = float(coordinates[stop - 1])
        boundary = span_start + fraction * (span_stop - span_start)
        if boundary <= boundaries[-1] + 1.0e-9:
            raise ValueError(
                "synthetic source riser boundaries are not strictly ordered"
            )
        boundaries.append(boundary)
    if route_length <= boundaries[-1] + 1.0e-9:
        raise ValueError("last source riser leaves no final support interval")
    boundaries.append(route_length)

    heights = [float(base_height_m)]
    for rise in rises:
        heights.append(heights[-1] + rise)
    levels = []
    for index, height in enumerate(heights):
        level_start = boundaries[index]
        level_stop = boundaries[index + 1]
        levels.append(
            VisibleTread(
                height_m=height,
                route_start_distance_m=level_start,
                route_stop_distance_m=level_stop,
                route_start_xy=np.asarray(
                    start_xy + level_start * direction, dtype=np.float32
                ),
                route_stop_xy=np.asarray(
                    start_xy + level_stop * direction, dtype=np.float32
                ),
                visible_in_mesh=True,
                left_foothold_center_xy=None,
                right_foothold_center_xy=None,
            )
        )
    return StairSupportRoute(
        start_xy=np.asarray(start_xy, dtype=np.float32),
        end_xy=np.asarray(end_xy, dtype=np.float32),
        levels=tuple(levels),
    )


def _fragment_selection(plan: StairFragmentSequencePlan) -> tuple[FragmentSelection, ...]:
    if not plan.supported:
        detail = "unsupported plan" if plan.reason is None else plan.reason.message
        raise PlannedFragmentReconstructionRejected("plan", detail)
    if not plan.selections:
        raise PlannedFragmentReconstructionRejected(
            "plan", "supported plan contains no riser selections"
        )
    return tuple(
        FragmentSelection(
            selection.fragment.archive_clip_index,
            selection.fragment.source_start_frame,
            selection.fragment.source_stop_frame,
        )
        for selection in plan.selections
    )


def _measure_source_fragment_geometry(
    archive: object,
    record: object,
    *,
    ground_fallback_height_m: float,
    sole_half_length_m: float,
    sole_half_width_m: float,
    route_sample_spacing_m: float,
    source_mesh: TerrainMeshIndex | None = None,
) -> ExactStairFragmentGeometry:
    """Locate this record's single source-mesh riser in its own root span."""

    clip_index = int(record.archive_clip_index)  # type: ignore[attr-defined]
    clip_start = int(archive["clip_start_idx"][clip_index])
    start = clip_start + int(record.source_start_frame)  # type: ignore[attr-defined]
    stop = clip_start + int(record.source_stop_frame)  # type: ignore[attr-defined]
    roots = np.asarray(archive["body_pos_w"][start:stop, 0], dtype=np.float64)
    if len(roots) < 2:
        raise PlannedFragmentReconstructionRejected(
            "source_transition", "selected fragment has fewer than two frames"
        )
    mesh = (
        _archive_terrain_index(archive, clip_index)
        if source_mesh is None
        else source_mesh
    )
    route = sample_stair_support_route(
        mesh,
        roots[0, :2],
        roots[-1, :2],
        ground_fallback_height_m,
        sole_half_length_m,
        sole_half_width_m,
        sample_spacing_m=route_sample_spacing_m,
    )
    route_length = float(
        np.linalg.norm(
            np.asarray(route.end_xy, dtype=np.float64)
            - np.asarray(route.start_xy, dtype=np.float64)
        )
    )
    if route_length <= 1.0e-9:
        raise PlannedFragmentReconstructionRejected(
            "source_transition", "selected source route has zero planar length"
        )
    direction = 1 if record.source_direction == "up" else -1  # type: ignore[attr-defined]
    transitions = [
        index
        for index, (source, target) in enumerate(
            zip(route.levels, route.levels[1:])
        )
        if direction * (float(target.height_m) - float(source.height_m))
        > 1.0e-6
    ]
    if len(transitions) != 1:
        raise PlannedFragmentReconstructionRejected(
            "source_transition",
            "selected one-riser fragment sampled "
            f"{len(transitions)} source-mesh risers",
        )
    boundary = _level_boundaries(route)[transitions[0] + 1]
    fraction = float(boundary / route_length)
    if not 0.0 < fraction < 1.0:
        raise PlannedFragmentReconstructionRejected(
            "source_transition",
            f"source-mesh riser fraction {fraction:.6f} is outside (0, 1)",
        )
    transition = transitions[0]
    signed_rise = float(
        route.levels[transition + 1].height_m
        - route.levels[transition].height_m
    )
    return ExactStairFragmentGeometry(
        transition_fraction=fraction,
        base_height_m=float(route.levels[0].height_m),
        signed_rise_m=signed_rise,
        route_length_m=route_length,
    )


def _source_transition_fraction(
    archive: object,
    record: object,
    *,
    ground_fallback_height_m: float,
    sole_half_length_m: float,
    sole_half_width_m: float,
    route_sample_spacing_m: float,
) -> tuple[float, float, float]:
    measurement = _measure_source_fragment_geometry(
        archive,
        record,
        ground_fallback_height_m=ground_fallback_height_m,
        sole_half_length_m=sole_half_length_m,
        sole_half_width_m=sole_half_width_m,
        route_sample_spacing_m=route_sample_spacing_m,
    )
    return (
        measurement.transition_fraction,
        measurement.base_height_m,
        measurement.signed_rise_m,
    )


def enrich_stair_fragment_bank_from_exact_mesh(
    archive_path: str | Path,
    fragments: Sequence[StairFragmentRecord],
    *,
    ground_fallback_height_m: float = 0.0,
    sole_half_length_m: float = 0.10,
    sole_half_width_m: float = 0.055,
    route_sample_spacing_m: float = 0.01,
) -> ExactStairFragmentBank:
    """Replace retrieval hints with exact per-fragment mesh rise and run."""

    import zarr

    archive = zarr.open_group(str(Path(archive_path)), mode="r")
    meshes: dict[int, TerrainMeshIndex] = {}
    records: list[StairFragmentRecord] = []
    rejected: list[tuple[str, str]] = []
    for fragment in fragments:
        if (
            fragment.segmentation_basis
            == "stable_tread_exact_mesh_height"
            and fragment.riser_transition_count == 1
        ):
            # These records were bounded by consecutive, exact-mesh support
            # heights during extraction.  Re-measuring the terrain under the
            # pelvis is both redundant and wrong: a leading foot commonly
            # reaches its new tread before the pelvis crosses that riser.
            # Keep the exact support-height rise and authored root travel.
            records.append(fragment)
            continue
        clip_index = int(fragment.archive_clip_index)
        if clip_index not in meshes:
            meshes[clip_index] = _archive_terrain_index(archive, clip_index)
        try:
            measurement = _measure_source_fragment_geometry(
                archive,
                fragment,
                ground_fallback_height_m=ground_fallback_height_m,
                sole_half_length_m=sole_half_length_m,
                sole_half_width_m=sole_half_width_m,
                route_sample_spacing_m=route_sample_spacing_m,
                source_mesh=meshes[clip_index],
            )
        except PlannedFragmentReconstructionRejected as error:
            rejected.append((fragment.fragment_id, error.message))
            continue
        records.append(
            replace(
                fragment,
                source_rise_m=abs(measurement.signed_rise_m),
                approximate_run_m=measurement.route_length_m,
            )
        )
    return ExactStairFragmentBank(tuple(records), tuple(rejected))


def _authored_stance_mask(
    motion: StitchedMotion,
    plan: StairFragmentSequencePlan,
) -> np.ndarray:
    """Transfer source contact phases through stitch provenance without target poses."""

    mask = np.zeros((len(motion.provenance), 2), dtype=bool)
    starts = (0,) + tuple(int(value) for value in motion.seam_indices)
    stops = tuple(int(value) for value in motion.seam_indices) + (
        len(motion.provenance),
    )
    for selection, start, stop in zip(plan.selections, starts, stops, strict=True):
        record = selection.fragment
        for output_index in range(start, stop):
            source_offset = (
                int(motion.provenance[output_index].source_frame)
                - int(record.source_start_frame)
            )
            phase = next(
                (
                    span.phase
                    for span in record.descriptor.contact_phases
                    if span.start_offset <= source_offset < span.stop_offset
                ),
                SupportPhase.FLIGHT,
            )
            mask[output_index, 0] = phase in (
                SupportPhase.LEFT,
                SupportPhase.DOUBLE,
            )
            mask[output_index, 1] = phase in (
                SupportPhase.RIGHT,
                SupportPhase.DOUBLE,
            )
    return mask


def _stance_spans(mask: object) -> tuple[StanceSpan, ...]:
    """Return deterministic half-open planted runs for each authored foot."""

    stance = np.asarray(mask, dtype=bool)
    if stance.ndim != 2 or stance.shape[1:] != (2,):
        raise ValueError("stance mask must have shape [frames, 2]")
    spans: list[StanceSpan] = []
    for foot in range(2):
        start: int | None = None
        for frame, active in enumerate(stance[:, foot]):
            if bool(active) and start is None:
                start = frame
            elif not bool(active) and start is not None:
                spans.append(StanceSpan(foot, start, frame))
                start = None
        if start is not None:
            spans.append(StanceSpan(foot, start, len(stance)))
    return tuple(spans)


def _blend_anchored_sole_offsets(
    nominal_targets: object,
    anchored_targets: object,
    assigned: object,
) -> np.ndarray:
    """Interpolate anchor edits through flight while preserving fixed stances."""

    nominal = np.asarray(nominal_targets, dtype=np.float64)
    anchored = np.asarray(anchored_targets, dtype=np.float64)
    fixed = np.asarray(assigned, dtype=bool)
    if (
        nominal.ndim != 4
        or nominal.shape[1] != 2
        or nominal.shape[-1] != 3
        or anchored.shape != nominal.shape
        or fixed.shape != nominal.shape[:2]
    ):
        raise ValueError("sole anchor arrays have inconsistent shapes")
    result = nominal.copy()
    offsets = anchored - nominal
    frame_coordinates = np.arange(len(nominal), dtype=np.float64)
    for foot in range(2):
        known = np.flatnonzero(fixed[:, foot])
        if not len(known):
            continue
        keys = known.tolist()
        values = [offsets[index, foot] for index in known]
        if keys[0] != 0:
            keys.insert(0, 0)
            values.insert(0, np.zeros_like(values[0]))
        if keys[-1] != len(nominal) - 1:
            keys.append(len(nominal) - 1)
            values.append(np.zeros_like(values[-1]))
        key_array = np.asarray(keys, dtype=np.float64)
        value_array = np.asarray(values, dtype=np.float64)
        interpolated = np.empty_like(nominal[:, foot])
        for sphere in range(nominal.shape[2]):
            for coordinate in range(3):
                interpolated[:, sphere, coordinate] = np.interp(
                    frame_coordinates,
                    key_array,
                    value_array[:, sphere, coordinate],
                )
        result[:, foot] += interpolated
        result[fixed[:, foot], foot] = anchored[fixed[:, foot], foot]
    return result


def _replace_bracketed_swing_trajectories(
    nominal_targets: object,
    anchored_targets: object,
    stance: object,
    *,
    minimum_clearance_m: float,
    maximum_planar_deviation_m: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep each complete authored swing in a corridor between its anchors.

    The planted poses define where a swing must begin and end, but replacing
    every in-between pose with a synthetic straight arc can ask the authored
    pelvis and leg pose to reach a foot position it never produced.  Preserve
    the anchor-offset authored trajectory whenever it is already plausible,
    and translate only gross planar deviations back to a bounded corridor
    around a smooth anchor interpolation.
    """

    nominal = np.asarray(nominal_targets, dtype=np.float64)
    result = np.asarray(anchored_targets, dtype=np.float64).copy()
    mask = np.asarray(stance, dtype=bool)
    if (
        nominal.ndim != 4
        or nominal.shape != result.shape
        or mask.shape != nominal.shape[:2]
    ):
        raise ValueError("bracketed swing arrays have inconsistent shapes")
    maximum_deviation = float(maximum_planar_deviation_m)
    if not math.isfinite(maximum_deviation) or maximum_deviation < 0.0:
        raise ValueError(
            "maximum planar swing deviation must be finite and nonnegative"
        )
    bracketed = np.zeros(mask.shape, dtype=bool)
    spans = _stance_spans(mask)
    for foot in range(2):
        foot_spans = tuple(span for span in spans if span.foot_index == foot)
        for source, target in zip(foot_spans, foot_spans[1:]):
            start = source.stop_frame - 1
            stop = target.start_frame
            if stop <= start + 1:
                continue
            for frame in range(start + 1, stop):
                linear = (frame - start) / float(stop - start)
                smooth = linear * linear * (3.0 - 2.0 * linear)
                anchor_interpolation = (
                    (1.0 - smooth) * result[start, foot]
                    + smooth * result[stop, foot]
                )
                nominal_interpolation = (
                    (1.0 - smooth) * nominal[start, foot]
                    + smooth * nominal[stop, foot]
                )
                candidate = result[frame, foot].copy()
                planar_deviation = np.mean(
                    candidate[:, :2] - anchor_interpolation[:, :2],
                    axis=0,
                )
                deviation_norm = float(np.linalg.norm(planar_deviation))
                if deviation_norm > maximum_deviation:
                    candidate[:, :2] += (
                        maximum_deviation / deviation_norm - 1.0
                    ) * planar_deviation
                authored_lift = max(
                    0.0,
                    float(
                        np.mean(
                            nominal[frame, foot, :, 2]
                            - nominal_interpolation[:, 2]
                        )
                    ),
                )
                clearance_arc = (
                    float(minimum_clearance_m)
                    * 4.0
                    * linear
                    * (1.0 - linear)
                )
                required_lift = max(authored_lift, clearance_arc)
                current_lift = float(
                    np.mean(
                        candidate[:, 2] - anchor_interpolation[:, 2]
                    )
                )
                if current_lift < required_lift:
                    candidate[:, 2] += required_lift - current_lift
                result[frame, foot] = candidate
                bracketed[frame, foot] = True
    return result, bracketed


def _interpolate_fixed_vectors(
    values: np.ndarray,
    fixed: np.ndarray,
) -> np.ndarray:
    """Linearly carry mechanically fixed values through unsupported gaps."""

    result = np.asarray(values, dtype=np.float64).copy()
    assigned = np.asarray(fixed, dtype=bool)
    if result.ndim != 2 or assigned.shape != (len(result),):
        raise ValueError("fixed-vector arrays have inconsistent shapes")
    known = np.flatnonzero(assigned)
    if not len(known):
        return result
    keys = known.tolist()
    key_values = [result[index].copy() for index in known]
    if keys[0] != 0:
        keys.insert(0, 0)
        key_values.insert(0, np.zeros(result.shape[1], dtype=np.float64))
    if keys[-1] != len(result) - 1:
        keys.append(len(result) - 1)
        key_values.append(np.zeros(result.shape[1], dtype=np.float64))
    coordinates = np.arange(len(result), dtype=np.float64)
    for axis in range(result.shape[1]):
        result[:, axis] = np.interp(
            coordinates,
            np.asarray(keys, dtype=np.float64),
            np.asarray(key_values, dtype=np.float64)[:, axis],
        )
    return result


def _smooth_root_anchor_shifts(
    shifts: object,
    *,
    sigma_frames: float,
) -> np.ndarray:
    """Remove pelvis teleports when the active planted-foot anchor changes.

    Each stance supplies a useful pelvis-translation target, but switching
    from one anchored foot to the other can make that target discontinuous.
    The legs can absorb the gradual difference through IK; the pelvis cannot
    physically realize a one-frame translation.  A zero-phase Gaussian is
    appropriate here because this is the privileged offline reconstruction,
    not the causal runtime controller.
    """

    values = np.asarray(shifts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,):
        raise ValueError("root anchor shifts must have shape [frames, 3]")
    sigma = float(sigma_frames)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("root anchor smoothing sigma must be finite and nonnegative")
    if sigma == 0.0 or len(values) < 2:
        return values.copy()
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(coordinates / sigma))
    kernel /= np.sum(kernel)
    result = np.empty_like(values)
    for axis in range(values.shape[1]):
        padded = np.pad(values[:, axis], (radius, radius), mode="edge")
        result[:, axis] = np.convolve(padded, kernel, mode="valid")
    return result


def _motion_step_metrics(
    root_position_world: object,
    root_quaternion_world_wxyz: object,
    joint_position: object,
) -> tuple[float, float, float]:
    """Return the largest adjacent-frame root, rotation, and joint steps."""

    roots = np.asarray(root_position_world, dtype=np.float64)
    quaternions = np.asarray(root_quaternion_world_wxyz, dtype=np.float64)
    joints = np.asarray(joint_position, dtype=np.float64)
    if not (len(roots) == len(quaternions) == len(joints)):
        raise ValueError("motion arrays must have the same frame count")
    if len(roots) < 2:
        return 0.0, 0.0, 0.0

    root_step = float(
        np.max(np.linalg.norm(np.diff(roots, axis=0), axis=1))
    )
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("root quaternion has zero norm")
    unit_quaternions = quaternions / norms
    adjacent_dots = np.abs(
        np.sum(unit_quaternions[1:] * unit_quaternions[:-1], axis=1)
    )
    rotation_step = float(
        np.max(2.0 * np.arccos(np.clip(adjacent_dots, 0.0, 1.0)))
    )
    joint_step = float(np.max(np.abs(np.diff(joints, axis=0))))
    return root_step, rotation_step, joint_step


def _maximum_root_acceleration_m_s2(
    root_position_world: object,
    *,
    fps: float,
) -> float:
    """Return the largest finite-difference root acceleration magnitude."""

    roots = np.asarray(root_position_world, dtype=np.float64)
    rate = float(fps)
    if roots.ndim != 2 or roots.shape[1:] != (3,):
        raise ValueError("root positions must have shape [frames,3]")
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("fps must be finite and positive")
    if len(roots) < 3:
        return 0.0
    acceleration = np.diff(roots, n=2, axis=0) * rate * rate
    return float(np.max(np.linalg.norm(acceleration, axis=1)))


def _maximum_joint_step_at_indices(
    joint_position: object,
    indices: Sequence[int],
) -> float:
    """Return the largest joint step entering any listed frame."""

    joints = np.asarray(joint_position, dtype=np.float64)
    values = tuple(int(index) for index in indices)
    if not values:
        return 0.0
    if any(index <= 0 or index >= len(joints) for index in values):
        raise ValueError("joint-step index is outside the motion")
    return max(
        float(np.max(np.abs(joints[index] - joints[index - 1])))
        for index in values
    )


def _stance_guidance_weights(
    stance: np.ndarray,
    *,
    blend_frames: int = 12,
    release_frames: int = 8,
) -> np.ndarray:
    """Use hard stance anchors with smooth touchdown and liftoff ramps."""

    mask = np.asarray(stance, dtype=bool)
    weights = np.zeros(mask.shape, dtype=np.float64)
    width = max(1, int(blend_frames))
    release = max(0, int(release_frames))
    for span in _stance_spans(mask):
        foot = span.foot_index
        start = max(0, span.start_frame - width)
        if start < span.start_frame:
            frames = np.arange(start, span.start_frame, dtype=np.int64)
            coordinate = (
                frames - start + 1
            ) / float(span.start_frame - start + 1)
            smooth = coordinate * coordinate * (3.0 - 2.0 * coordinate)
            weights[frames, foot] = np.maximum(
                weights[frames, foot], smooth
            )
        weights[span.start_frame : span.stop_frame, foot] = 1.0
        stop = min(len(mask), span.stop_frame + release)
        if span.stop_frame < stop:
            frames = np.arange(span.stop_frame, stop, dtype=np.int64)
            coordinate = (
                frames - span.stop_frame + 1
            ) / float(stop - span.stop_frame + 1)
            smooth = coordinate * coordinate * (3.0 - 2.0 * coordinate)
            weights[frames, foot] = np.maximum(
                weights[frames, foot], 1.0 - smooth
            )
    return weights


def _smooth_swing_route_adjustments(
    adjustments: object,
    stance: object,
    *,
    decay_frames: float = 4.0,
) -> np.ndarray:
    """Ramp collision-clearance shifts inside each swing phase.

    A per-frame riser query can make a centimetre-scale route correction appear
    or disappear in one frame.  Preserve every required correction at its
    source frame and spread a decaying envelope only through frames where that
    foot is not planted.
    """

    values = np.asarray(adjustments, dtype=np.float64)
    mask = np.asarray(stance, dtype=bool)
    decay = float(decay_frames)
    if values.ndim != 3 or values.shape[1:] != (2, 3):
        raise ValueError("swing route adjustments must have shape [frames,2,3]")
    if mask.shape != values.shape[:2]:
        raise ValueError("swing route adjustment stance mask has wrong shape")
    if not math.isfinite(decay) or decay <= 0.0:
        raise ValueError("swing route adjustment decay must be positive")

    result = np.zeros_like(values)
    frames = np.arange(len(values), dtype=np.float64)
    for foot in range(2):
        swing = ~mask[:, foot]
        active = np.flatnonzero(
            swing & np.any(np.abs(values[:, foot]) > 0.0, axis=1)
        )
        for source in active:
            weight = (
                np.exp(-np.abs(frames - source) / decay)
                * swing
            )
            candidate = values[source, foot][None] * weight[:, None]
            stronger = np.abs(candidate) > np.abs(result[:, foot])
            result[:, foot] = np.where(
                stronger, candidate, result[:, foot]
            )
    return result


def _clearance_lift_requirements(
    scaffold_clearances: object,
    stance: object,
    *,
    maximum_sole_penetration_m: float,
    maximum_foot_target_error_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate conservative swing lift from final-bound stance rejection.

    Swing targets reserve the full permitted IK residual because they can be
    lifted before solving.  A planted target cannot be lifted without breaking
    its foothold, so its scaffold is screened against the final penetration
    bound; the post-IK penetration checks remain authoritative.
    """

    clearances = np.asarray(scaffold_clearances, dtype=np.float64)
    stance_mask = np.asarray(stance, dtype=bool)
    swing_penetration_budget = max(
        0.0,
        float(maximum_sole_penetration_m)
        - float(maximum_foot_target_error_m),
    )
    required_lift = np.maximum(
        0.0,
        -swing_penetration_budget - clearances,
    )
    planted_collision = (
        np.maximum(
            0.0,
            -float(maximum_sole_penetration_m) - clearances,
        )
        * stance_mask
    )
    return required_lift, planted_collision


def _clear_swing_sole_target(
    sole_centres: object,
    sphere_radii: object,
    *,
    mesh: TerrainMeshIndex,
    route: StairSupportRoute,
    maximum_longitudinal_adjustment_m: float = 0.20,
    maximum_vertical_adjustment_m: float = 0.25,
    clearance_m: float = 0.0025,
    foot_forward_extension_m: float = 0.025,
    foot_backward_extension_m: float = 0.045,
    foot_lateral_extension_m: float = 0.010,
    foot_collision_height_m: float = 0.060,
    mesh_query: CanonicalMeshQuery | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a small route shift/lift that clears the complete G1 foot.

    The four sole spheres describe contact with horizontal support, but the
    collision mesh extends beyond them at both toe and heel.  Raycasting only
    the sphere bottoms can therefore miss a swing foot whose mesh clips a
    vertical riser.  The extra toe/heel probes conservatively cover that
    authored collision envelope.
    """

    centres = np.asarray(sole_centres, dtype=np.float64)
    radii = np.asarray(sphere_radii, dtype=np.float64)
    start = np.asarray(route.start_xy, dtype=np.float64)
    end = np.asarray(route.end_xy, dtype=np.float64)
    direction = end - start
    direction /= np.linalg.norm(direction)
    ray_origin = (
        float(np.max(mesh.vertices_world[:, 2]))
        if len(mesh.vertices_world)
        else 0.0
    ) + 1.0
    exact_query = (
        CanonicalMeshQuery(mesh.mesh, mesh.world_from_terrain)
        if mesh_query is None
        else mesh_query
    )

    def extended_support_points(shifted: np.ndarray) -> np.ndarray:
        support = shifted.copy()
        support[:, 2] -= radii
        if len(support) < 4:
            return support
        envelope = _foot_collision_envelope_points(
            support,
            foot_forward_extension_m=foot_forward_extension_m,
            foot_backward_extension_m=foot_backward_extension_m,
            foot_lateral_extension_m=foot_lateral_extension_m,
            foot_collision_height_m=foot_collision_height_m,
        )
        return np.concatenate((support, envelope), axis=0)

    def candidate(longitudinal: float) -> tuple[np.ndarray, float, float]:
        shifted = centres.copy()
        shifted[:, :2] += float(longitudinal) * direction
        support = extended_support_points(shifted)
        required_vertical = 0.0
        for point in support:
            hit = mesh.raycast(
                np.asarray((point[0], point[1], ray_origin)),
                np.asarray((0.0, 0.0, -1.0)),
            )
            if hit is not None:
                required_vertical = max(
                    required_vertical,
                    float(hit.position_world[2])
                    + float(clearance_m)
                    - float(point[2]),
                )
        required_vertical = max(0.0, required_vertical)
        shifted[:, 2] += required_vertical
        distances = np.asarray(
            exact_query.query(shifted).distance_m, dtype=np.float64
        )
        penetration = float(
            np.max(np.maximum(0.0, radii - distances))
        )
        return shifted, required_vertical, penetration

    unchanged, unchanged_vertical, unchanged_penetration = candidate(0.0)
    if (
        unchanged_vertical <= 0.0
        and unchanged_penetration <= clearance_m
    ):
        return unchanged, np.zeros(3, dtype=np.float64)

    best: tuple[tuple[float, float, float], np.ndarray, np.ndarray] | None = None
    for longitudinal in np.linspace(
        -maximum_longitudinal_adjustment_m,
        maximum_longitudinal_adjustment_m,
        41,
    ):
        shifted, required_vertical, penetration = candidate(
            float(longitudinal)
        )
        if required_vertical > maximum_vertical_adjustment_m:
            continue
        if penetration > clearance_m:
            continue
        normalized_longitudinal = (
            0.0
            if maximum_longitudinal_adjustment_m <= 0.0
            else longitudinal / maximum_longitudinal_adjustment_m
        )
        normalized_vertical = (
            0.0
            if maximum_vertical_adjustment_m <= 0.0
            else required_vertical / maximum_vertical_adjustment_m
        )
        rank = (
            normalized_longitudinal**2
            + 4.0 * normalized_vertical**2,
            required_vertical,
            abs(float(longitudinal)),
        )
        translation = np.asarray(
            (float(longitudinal), 0.0, required_vertical),
            dtype=np.float64,
        )
        if best is None or rank < best[0]:
            best = (rank, shifted, translation)
    if best is None:
        return centres.copy(), np.zeros(3, dtype=np.float64)
    return best[1], best[2]


def _sole_yaw(support_points: np.ndarray) -> float:
    """Estimate the authored foot-forward yaw from ordered G1 sole samples."""

    points = np.asarray(support_points, dtype=np.float64)
    if len(points) >= 4:
        direction = np.mean(points[len(points) // 2 :, :2], axis=0) - np.mean(
            points[: len(points) // 2, :2], axis=0
        )
    elif len(points) >= 2:
        direction = points[-1, :2] - points[0, :2]
    else:
        direction = np.asarray((1.0, 0.0))
    if float(np.linalg.norm(direction)) <= 1.0e-9:
        return 0.0
    return math.atan2(float(direction[1]), float(direction[0]))


def _stance_target_level(
    route: StairSupportRoute,
    support_points_world: object,
) -> int:
    """Choose the spatially nearest route interval, then check sole height."""

    points = np.asarray(support_points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
        raise ValueError("stance support points must have shape [N,3]")
    start = np.asarray(route.start_xy, dtype=np.float64)
    direction = np.asarray(route.end_xy, dtype=np.float64) - start
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-9:
        raise ValueError("foothold route has zero length")
    direction /= length
    coordinate = float((np.mean(points[:, :2], axis=0) - start) @ direction)
    height = float(np.median(points[:, 2]))

    def rank(index: int) -> tuple[float, float, int]:
        level = route.levels[index]
        interval_gap = max(
            float(level.route_start_distance_m) - coordinate,
            0.0,
            coordinate - float(level.route_stop_distance_m),
        )
        height_gap = abs(float(level.height_m) - height)
        return interval_gap, height_gap, index

    return min(range(len(route.levels)), key=rank)


def _anchor_stance_sole_targets(
    nominal_targets: object,
    stance: object,
    sphere_radii: Sequence[np.ndarray],
    collision_envelopes: Sequence[Sequence[np.ndarray]] | None = None,
    *,
    target_mesh: TerrainMeshIndex,
    foothold_route: StairSupportRoute,
    ground_fallback_height_m: float,
    config: FootholdAnchorConfig,
    maximum_monotonic_level_step: int | None = None,
    target_level_by_stance_span: Mapping[tuple[int, int, int], int]
    | None = None,
) -> tuple[np.ndarray, tuple[FootholdAnchorDiagnostics, ...]]:
    """Lock every authored stance run to one terrain-valid world foothold."""

    nominal = np.asarray(nominal_targets, dtype=np.float64)
    stance_mask = np.asarray(stance, dtype=bool)
    anchored = nominal.copy()
    assigned = np.zeros(stance_mask.shape, dtype=bool)
    diagnostics: list[FootholdAnchorDiagnostics] = []
    if not foothold_route.levels:
        raise PlannedFragmentReconstructionRejected(
            "target_foothold", "foothold route contains no support levels"
        )
    spans = _stance_spans(stance_mask)
    target_levels: dict[tuple[int, int, int], int] = {}
    if target_level_by_stance_span is not None:
        valid_keys = {
            (span.foot_index, span.start_frame, span.stop_frame)
            for span in spans
        }
        unknown = set(target_level_by_stance_span) - valid_keys
        if unknown:
            raise ValueError("explicit stance-level schedule contains unknown spans")
        for key, value in target_level_by_stance_span.items():
            level = int(value)
            if level < 0 or level >= len(foothold_route.levels):
                raise ValueError("explicit stance level is outside the support route")
            target_levels[key] = level
    if maximum_monotonic_level_step is not None:
        maximum_step = int(maximum_monotonic_level_step)
        if maximum_step < 1:
            raise ValueError("maximum monotonic level step must be positive")
        chronological = sorted(
            spans,
            key=lambda value: (
                value.start_frame + value.stop_frame,
                value.foot_index,
            ),
        )
        previous: int | None = None
        for span in chronological:
            frame = (span.start_frame + span.stop_frame - 1) // 2
            foot = span.foot_index
            radii = np.asarray(sphere_radii[foot], dtype=np.float64)
            support = np.asarray(nominal[frame, foot], dtype=np.float64).copy()
            support[:, 2] -= radii
            proposed = _stance_target_level(foothold_route, support)
            target_level = (
                proposed
                if previous is None
                else max(previous, min(proposed, previous + maximum_step))
            )
            key = (span.foot_index, span.start_frame, span.stop_frame)
            target_levels.setdefault(key, target_level)
            previous = target_level
    for span in spans:
        frame = (span.start_frame + span.stop_frame - 1) // 2
        foot = span.foot_index
        radii = np.asarray(sphere_radii[foot], dtype=np.float64)
        centres = np.asarray(nominal[frame, foot], dtype=np.float64)
        support = centres.copy()
        support[:, 2] -= radii
        target_level = target_levels.get(
            (span.foot_index, span.start_frame, span.stop_frame),
            _stance_target_level(foothold_route, support),
        )
        pose = NominalFootSolePose(
            sole_center_world=np.mean(centres, axis=0),
            sole_support_points_world=support,
            yaw_rad=_sole_yaw(support),
            collision_envelope_points_world=(
                None
                if collision_envelopes is None
                else np.asarray(
                    collision_envelopes[frame][foot],
                    dtype=np.float64,
                )
            ),
        )
        try:
            anchor = anchor_planted_foot(
                target_mesh,
                foothold_route,
                span,
                pose,
                target_level_index=target_level,
                ground_fallback_height_m=ground_fallback_height_m,
                config=config,
            )
        except FootholdAnchorRejected as error:
            detail = error.diagnostics
            raise PlannedFragmentReconstructionRejected(
                "target_foothold",
                f"frame span {span.start_frame}:{span.stop_frame}, "
                f"foot {span.foot_index}, level {target_level}: "
                f"{detail.rejection_reason}",
            ) from error
        anchor_centres = np.asarray(
            anchor.pose.sole_support_points_world, dtype=np.float64
        ).copy()
        anchor_centres[:, 2] += radii
        anchored[span.start_frame : span.stop_frame, foot] = anchor_centres
        assigned[span.start_frame : span.stop_frame, foot] = True
        diagnostics.append(anchor.diagnostics)
    return (
        _blend_anchored_sole_offsets(nominal, anchored, assigned),
        tuple(diagnostics),
    )


def reconstruct_stair_fragment_sequence(
    plan: StairFragmentSequencePlan,
    target_route: StairSupportRoute,
    *,
    archive_path: str | Path,
    model_path: str | Path,
    target_clip_index: int | None = None,
    target_mesh: TerrainMeshIndex | None = None,
    excluded_source_clip_indices: Collection[int] = (),
    decay_frames: float = 8.0,
    ground_fallback_height_m: float = 0.0,
    sole_half_length_m: float = 0.10,
    sole_half_width_m: float = 0.055,
    route_sample_spacing_m: float = 0.01,
    maximum_joint_correction_rad: float = 0.36,
    maximum_foot_target_error_m: float = 0.001,
    maximum_swing_foot_target_error_m: float = 0.05,
    maximum_sole_penetration_m: float = 0.0025,
    maximum_root_clearance_lift_m: float = 0.005,
    maximum_swing_foot_clearance_lift_m: float = 0.25,
    minimum_stance_support_points: int = 1,
    support_contact_tolerance_m: float = 0.02,
    maximum_source_switch_joint_step_rad: float = 0.20,
    maximum_fragment_seam_joint_step_rad: float = 0.20,
    maximum_root_translation_step_m: float = 0.04,
    maximum_root_rotation_step_rad: float = math.radians(5.0),
    maximum_root_acceleration_m_s2: float = 25.0,
    maximum_interframe_joint_step_rad: float = 0.25,
    foothold_route: StairSupportRoute | None = None,
    foothold_anchor_config: FootholdAnchorConfig = FootholdAnchorConfig(),
    swing_anchor_blend_frames: int = 48,
    swing_anchor_release_frames: int = 8,
    minimum_swing_foot_clearance_m: float = 0.08,
    maximum_swing_planar_deviation_m: float = 0.12,
    root_anchor_smoothing_frames: float = 8.0,
    allow_target_motion_clip: bool = False,
    temporal_ik_warm_start: bool = False,
    maximum_swing_leg_joint_step_rad: float | None = None,
) -> StairFragmentReconstructionResult:
    """Stitch planned source fragments once, then fit that trace to target mesh support.

    When ``temporal_ik_warm_start`` is enabled, each foot-fit solve starts its
    leg joints from the preceding adapted frame.  The authored pose remains
    the IK posture regularizer; this only keeps successive solves on the same
    local branch.
    """

    import zarr

    archive = zarr.open_group(str(Path(archive_path)), mode="r")
    if target_mesh is None:
        if target_clip_index is None:
            raise ValueError(
                "target_mesh or target_clip_index is required for reconstruction"
            )
        target_mesh = _archive_terrain_index(archive, int(target_clip_index))
    elif not isinstance(target_mesh, TerrainMeshIndex):
        raise TypeError("target_mesh must be a TerrainMeshIndex")

    selections = _fragment_selection(plan)
    excluded_sources = {int(value) for value in excluded_source_clip_indices}
    if target_clip_index is not None and not allow_target_motion_clip:
        excluded_sources.add(int(target_clip_index))
    aliased_sources = tuple(
        selection.archive_clip_index
        for selection in selections
        if selection.archive_clip_index in excluded_sources
    )
    if aliased_sources:
        raise PlannedFragmentReconstructionRejected(
            "source_selection",
            "planned source fragments include an excluded source motion clip: "
            + ", ".join(str(value) for value in sorted(set(aliased_sources))),
        )
    try:
        stitched = stitch_archive(
            Path(archive_path), selections, decay_frames=decay_frames
        )
    except ValueError as error:
        raise PlannedFragmentReconstructionRejected(
            "source_stitch", str(error)
        ) from error
    source_geometry = tuple(
        _measure_source_fragment_geometry(
            archive,
            selection.fragment,
            ground_fallback_height_m=ground_fallback_height_m,
            sole_half_length_m=sole_half_length_m,
            sole_half_width_m=sole_half_width_m,
            route_sample_spacing_m=route_sample_spacing_m,
        )
        for selection in plan.selections
    )
    fractions = tuple(item.transition_fraction for item in source_geometry)
    try:
        source_route = build_synthetic_source_support_route(
            stitched,
            # The manifest rise is a retrieval hint.  Contact fitting must use
            # the exact authored mesh rise; otherwise a foot planted on a
            # 25.7 cm source step can be treated as if it came from a 17.7 cm
            # step and hover by the full metadata error on the target.
            signed_rises_m=tuple(item.signed_rise_m for item in source_geometry),
            transition_fractions=fractions,
            base_height_m=source_geometry[0].base_height_m,
        )
        warp = build_stair_geometry_warp(source_route, target_route)
    except ValueError as error:
        raise PlannedFragmentReconstructionRejected(
            "synthetic_source_route", str(error)
        ) from error

    if minimum_stance_support_points < 1:
        raise PlannedFragmentReconstructionRejected(
            "target_support", "minimum stance support points must be positive"
        )
    if support_contact_tolerance_m <= 0.0:
        raise PlannedFragmentReconstructionRejected(
            "target_support", "stance support tolerance must be positive"
        )
    requested_root = warp.warp_points(
        stitched.root_position_world, smooth_height=True
    )
    warped_root = requested_root.copy()
    warped_quaternions = warp.rotate_quaternions_wxyz(
        stitched.root_quaternion_world_wxyz
    )
    joint_names = tuple(str(value) for value in archive["joint_names"][:])
    adapter = _G1FootfallAdapter(
        Path(model_path),
        joint_names,
        maximum_joint_correction_rad=maximum_joint_correction_rad,
    )
    source_joints = np.asarray(stitched.joint_position, dtype=np.float64)
    sphere_radii = adapter.sole_sphere_radii()
    if minimum_stance_support_points > min(len(value) for value in sphere_radii):
        raise PlannedFragmentReconstructionRejected(
            "target_support", "minimum stance support points exceeds sole samples"
        )
    ray_origin_height = float(np.max(target_mesh.vertices_world[:, 2])) + 1.0
    stance = _authored_stance_mask(stitched, plan)
    anchor_route = target_route if foothold_route is None else foothold_route

    def vertical_clearance(support_points: Sequence[np.ndarray]) -> float:
        clearances = []
        for point in np.concatenate(tuple(support_points)):
            hit = target_mesh.raycast(
                np.asarray((point[0], point[1], ray_origin_height)),
                np.asarray((0.0, 0.0, -1.0)),
            )
            surface_height = (
                ground_fallback_height_m
                if hit is None
                else float(hit.position_world[2])
            )
            clearances.append(float(point[2] - surface_height))
        return min(clearances)

    def support_point_count(points: np.ndarray) -> int:
        count = 0
        for point in np.asarray(points, dtype=np.float64):
            hit = target_mesh.raycast(
                np.asarray((point[0], point[1], ray_origin_height)),
                np.asarray((0.0, 0.0, -1.0)),
            )
            surface_height = (
                ground_fallback_height_m
                if hit is None
                else float(hit.position_world[2])
            )
            if abs(float(point[2]) - surface_height) <= support_contact_tolerance_m:
                count += 1
        return count

    nominal_soles_by_frame: list[tuple[np.ndarray, np.ndarray]] = []
    nominal_collision_envelopes_by_frame: list[
        tuple[np.ndarray, np.ndarray]
    ] = []
    for frame in range(len(source_joints)):
        source_soles = adapter.sole_positions_for_pose(
            root_position=stitched.root_position_world[frame],
            root_quaternion_wxyz=stitched.root_quaternion_world_wxyz[frame],
            joints=source_joints[frame],
        )
        source_collision_envelopes = (
            adapter.foot_collision_envelope_points_for_pose(
                root_position=stitched.root_position_world[frame],
                root_quaternion_wxyz=(
                    stitched.root_quaternion_world_wxyz[frame]
                ),
                joints=source_joints[frame],
            )
        )
        # A swing foot follows the smoothed riser transition.  An authored
        # stance foot is a contact anchor: smoothing its height correction
        # makes it hover when source and target riser heights differ.
        target_soles = tuple(
            warp.warp_rigid_point_cloud(
                sole,
                smooth_height=not bool(stance[frame, foot]),
            )
            for foot, sole in enumerate(source_soles)
        )
        nominal_soles_by_frame.append(target_soles)
        nominal_collision_envelopes_by_frame.append(
            tuple(
                warp.warp_rigid_point_cloud(
                    envelope,
                    smooth_height=not bool(stance[frame, foot]),
                )
                for foot, envelope in enumerate(
                    source_collision_envelopes
                )
            )  # type: ignore[arg-type]
        )

    nominal_soles_array = np.asarray(
        nominal_soles_by_frame, dtype=np.float64
    )
    try:
        target_soles_array, anchor_diagnostics = _anchor_stance_sole_targets(
            nominal_soles_array,
            stance,
            sphere_radii,
            nominal_collision_envelopes_by_frame,
            target_mesh=target_mesh,
            foothold_route=anchor_route,
            ground_fallback_height_m=ground_fallback_height_m,
            config=foothold_anchor_config,
        )
    except ValueError as error:
        if isinstance(error, PlannedFragmentReconstructionRejected):
            raise
        raise PlannedFragmentReconstructionRejected(
            "target_foothold", str(error)
        ) from error
    (
        target_soles_array,
        bracketed_swing,
    ) = _replace_bracketed_swing_trajectories(
        nominal_soles_array,
        target_soles_array,
        stance,
        minimum_clearance_m=minimum_swing_foot_clearance_m,
        maximum_planar_deviation_m=maximum_swing_planar_deviation_m,
    )

    target_soles_by_frame: list[tuple[np.ndarray, np.ndarray]] = [
        (target_soles_array[frame, 0], target_soles_array[frame, 1])
        for frame in range(len(target_soles_array))
    ]
    provisional_soles_by_frame: list[tuple[np.ndarray, np.ndarray]] = []
    root_anchor_shifts = np.zeros((len(source_joints), 3), dtype=np.float64)
    root_anchor_fixed = np.zeros(len(source_joints), dtype=bool)
    for frame in range(len(source_joints)):
        provisional_soles = adapter.sole_positions_for_pose(
            root_position=warped_root[frame],
            root_quaternion_wxyz=warped_quaternions[frame],
            joints=source_joints[frame],
        )
        provisional_soles_by_frame.append(provisional_soles)
        active_feet = np.flatnonzero(stance[frame])
        if not len(active_feet):
            continue
        root_anchor_shifts[frame] = np.mean(
            np.stack(
                [
                    np.mean(target_soles_array[frame, foot], axis=0)
                    - np.mean(provisional_soles[foot], axis=0)
                    for foot in active_feet
                ]
            ),
            axis=0,
        )
        root_anchor_fixed[frame] = True
    root_anchor_shifts = _interpolate_fixed_vectors(
        root_anchor_shifts, root_anchor_fixed
    )
    root_anchor_shifts = _smooth_root_anchor_shifts(
        root_anchor_shifts,
        sigma_frames=root_anchor_smoothing_frames,
    )
    warped_root += root_anchor_shifts
    guidance_weights = _stance_guidance_weights(
        stance,
        blend_frames=swing_anchor_blend_frames,
        release_frames=swing_anchor_release_frames,
    )
    guidance_weights[bracketed_swing] = 1.0
    mesh_query = CanonicalMeshQuery(
        target_mesh.mesh, target_mesh.world_from_terrain
    )

    scaffold_clearances = np.empty((len(source_joints), 2), dtype=np.float64)
    swing_route_adjustments = np.zeros(
        (len(source_joints), 2, 3), dtype=np.float64
    )
    for frame in range(len(source_joints)):
        provisional_soles = tuple(
            sole + root_anchor_shifts[frame][None]
            for sole in provisional_soles_by_frame[frame]
        )
        target_soles = tuple(
            (
                target_soles_array[frame, foot]
                if stance[frame, foot]
                else (
                    provisional_soles[foot]
                    + guidance_weights[frame, foot]
                    * (
                        target_soles_array[frame, foot]
                        - provisional_soles[foot]
                    )
                )
            )
            for foot in range(2)
        )
        cleared_targets = list(target_soles)
        for foot in range(2):
            if stance[frame, foot]:
                continue
            (
                cleared_targets[foot],
                swing_route_adjustments[frame, foot],
            ) = _clear_swing_sole_target(
                cleared_targets[foot],
                sphere_radii[foot],
                mesh=target_mesh,
                route=anchor_route,
                mesh_query=mesh_query,
            )
        target_soles = (cleared_targets[0], cleared_targets[1])
        target_soles_by_frame[frame] = target_soles
        target_support = tuple(
            target_soles[foot]
            - np.asarray((0.0, 0.0, 1.0))[None]
            * sphere_radii[foot][:, None]
            for foot in range(2)
        )
        for foot in range(2):
            scaffold_clearances[frame, foot] = vertical_clearance(
                (target_support[foot],)
            )

    smoothed_swing_adjustments = _smooth_swing_route_adjustments(
        swing_route_adjustments,
        stance,
    )
    route_direction_xy = (
        np.asarray(anchor_route.end_xy, dtype=np.float64)
        - np.asarray(anchor_route.start_xy, dtype=np.float64)
    )
    route_direction_xy /= np.linalg.norm(route_direction_xy)
    for frame in range(len(target_soles_by_frame)):
        updated = []
        for foot, sole in enumerate(target_soles_by_frame[frame]):
            difference = (
                smoothed_swing_adjustments[frame, foot]
                - swing_route_adjustments[frame, foot]
            )
            world_difference = np.asarray(
                (
                    difference[0] * route_direction_xy[0],
                    difference[0] * route_direction_xy[1],
                    difference[2],
                ),
                dtype=np.float64,
            )
            updated.append(sole + world_difference[None])
        target_soles_by_frame[frame] = (updated[0], updated[1])
        target_support = tuple(
            target_soles_by_frame[frame][foot]
            - np.asarray((0.0, 0.0, 1.0))[None]
            * sphere_radii[foot][:, None]
            for foot in range(2)
        )
        for foot in range(2):
            scaffold_clearances[frame, foot] = vertical_clearance(
                (target_support[foot],)
            )
    swing_route_adjustments = smoothed_swing_adjustments

    required_lift, planted_collision = _clearance_lift_requirements(
        scaffold_clearances,
        stance,
        maximum_sole_penetration_m=maximum_sole_penetration_m,
        maximum_foot_target_error_m=maximum_foot_target_error_m,
    )
    if np.any(planted_collision > 0.0):
        frame, foot = np.unravel_index(
            int(np.argmax(planted_collision)), planted_collision.shape
        )
        raise PlannedFragmentReconstructionRejected(
            "target_foothold",
            "planted anchor intersects target terrain by "
            f"{float(planted_collision[frame, foot]):.6f} m at frame "
            f"{frame}, foot {foot}",
        )
    swing_lift = np.zeros_like(required_lift)
    frame_indices = np.arange(len(required_lift), dtype=np.float64)
    for foot in range(2):
        for active in np.flatnonzero(
            (required_lift[:, foot] > 0.0) & ~stance[:, foot]
        ):
            same_swing = ~stance[:, foot]
            swing_lift[:, foot] = np.maximum(
                swing_lift[:, foot],
                required_lift[active, foot]
                * np.exp(-np.abs(frame_indices - active) / 4.0)
                * same_swing,
            )
    if (
        float(np.max(swing_lift))
        > maximum_swing_foot_clearance_lift_m
    ):
        frame, foot = np.unravel_index(
            int(np.argmax(swing_lift)), swing_lift.shape
        )
        raise PlannedFragmentReconstructionRejected(
            "mechanical_bounds",
            "geometry warp needs excessive swing-foot clearance lift: "
            f"{float(np.max(swing_lift)):.6f} m at frame {frame}, foot {foot}",
        )
    for frame in range(len(swing_lift)):
        target_soles_by_frame[frame] = tuple(
            sole
            + np.asarray((0.0, 0.0, swing_lift[frame, foot]))[None]
            for foot, sole in enumerate(target_soles_by_frame[frame])
        )
    clearance_lift = np.zeros(len(source_joints), dtype=np.float64)

    adapted_joints = np.empty_like(source_joints)
    corrections = np.empty(len(source_joints), dtype=np.float64)
    errors = np.empty(len(source_joints), dtype=np.float64)
    errors_by_foot = np.empty((len(source_joints), 2), dtype=np.float64)
    minimum_clearances = np.empty(len(source_joints), dtype=np.float64)
    triangle_penetrations = np.empty(len(source_joints), dtype=np.float64)
    stance_support_counts = np.full((len(source_joints), 2), -1, dtype=np.int16)
    unsupported_stance: list[tuple[int, int, int]] = []
    for frame in range(len(source_joints)):
        (
            adapted_joints[frame],
            corrections[frame],
            errors[frame],
        ) = adapter.adapt_to_targets(
            root_position=warped_root[frame],
            root_quaternion_wxyz=warped_quaternions[frame],
            authored_joints=source_joints[frame],
            sole_targets_world=target_soles_by_frame[frame],
            initial_joints=(
                adapted_joints[frame - 1]
                if temporal_ik_warm_start and frame > 0
                else None
            ),
            continuity_joints=(
                adapted_joints[frame - 1]
                if maximum_swing_leg_joint_step_rad is not None and frame > 0
                else None
            ),
            continuity_feet=~stance[frame],
            maximum_continuity_joint_step_rad=(
                maximum_swing_leg_joint_step_rad if frame > 0 else None
            ),
        )
        final_centres = adapter.sole_positions_for_pose(
            root_position=warped_root[frame],
            root_quaternion_wxyz=warped_quaternions[frame],
            joints=adapted_joints[frame],
        )
        for foot in range(2):
            errors_by_foot[frame, foot] = float(
                np.max(
                    np.linalg.norm(
                        target_soles_by_frame[frame][foot]
                        - final_centres[foot],
                        axis=1,
                    )
                )
            )
        errors[frame] = float(np.max(errors_by_foot[frame]))
        support_points = adapter.sole_support_points_for_pose(
            root_position=warped_root[frame],
            root_quaternion_wxyz=warped_quaternions[frame],
            joints=adapted_joints[frame],
        )
        minimum_clearances[frame] = vertical_clearance(support_points)
        for foot in range(2):
            if not stance[frame, foot]:
                continue
            count = support_point_count(support_points[foot])
            stance_support_counts[frame, foot] = count
            if count < minimum_stance_support_points:
                unsupported_stance.append((frame, foot, count))
        centres = np.concatenate(final_centres)
        radii = np.concatenate(sphere_radii)
        distances = np.asarray(mesh_query.query(centres).distance_m, dtype=np.float64)
        triangle_penetrations[frame] = float(
            np.max(np.maximum(0.0, radii - distances))
        )

    maximum_correction = float(np.max(corrections))
    maximum_error = float(np.max(errors))
    maximum_stance_error = float(np.max(errors_by_foot[stance]))
    maximum_swing_error = float(np.max(errors_by_foot[~stance]))
    minimum_clearance = float(np.min(minimum_clearances))
    maximum_penetration = max(0.0, -minimum_clearance)
    maximum_triangle_penetration = float(np.max(triangle_penetrations))
    advertised_stance_counts = stance_support_counts[stance_support_counts >= 0]
    if not len(advertised_stance_counts):
        raise PlannedFragmentReconstructionRejected(
            "target_support", "selected fragments provide no authored stance samples"
        )
    violations: list[tuple[str, str]] = []
    if unsupported_stance:
        frame, foot, count = unsupported_stance[0]
        side = "left" if foot == 0 else "right"
        violations.append(
            (
                "target_support",
                f"frame {frame}, {side}, "
                f"{count}/{minimum_stance_support_points} required sole "
                "support points",
            )
        )
    if maximum_correction > maximum_joint_correction_rad + 1.0e-6:
        violations.append(
            (
                "mechanical_bounds",
                f"joint correction {maximum_correction:.6f} rad exceeds bound",
            )
        )
    if maximum_stance_error > maximum_foot_target_error_m:
        frame, foot = np.unravel_index(
            int(np.argmax(np.where(stance, errors_by_foot, -np.inf))),
            errors_by_foot.shape,
        )
        violations.append(
            (
                "mechanical_bounds",
                "stance-foot residual "
                f"{maximum_stance_error:.6f} m at frame {frame}, foot {foot} "
                "exceeds bound",
            )
        )
    if maximum_swing_error > maximum_swing_foot_target_error_m:
        frame, foot = np.unravel_index(
            int(np.argmax(np.where(~stance, errors_by_foot, -np.inf))),
            errors_by_foot.shape,
        )
        violations.append(
            (
                "mechanical_bounds",
                "swing-foot guidance residual "
                f"{maximum_swing_error:.6f} m at frame {frame}, foot {foot} "
                "exceeds bound",
            )
        )
    if maximum_penetration > maximum_sole_penetration_m + 1.0e-6:
        violations.append(
            (
                "mechanical_bounds",
                f"vertical penetration {maximum_penetration:.6f} m exceeds bound",
            )
        )
    if maximum_triangle_penetration > maximum_sole_penetration_m + 1.0e-6:
        frame = int(np.argmax(triangle_penetrations))
        violations.append(
            (
                "mechanical_bounds",
                "triangle-sphere penetration "
                f"{maximum_triangle_penetration:.6f} m at frame {frame} "
                "exceeds bound",
            )
        )

    (
        maximum_root_step,
        maximum_rotation_step,
        maximum_joint_step,
    ) = _motion_step_metrics(
        warped_root,
        warped_quaternions,
        adapted_joints,
    )
    maximum_root_acceleration = _maximum_root_acceleration_m_s2(
        warped_root,
        fps=stitched.fps,
    )
    if maximum_root_step > maximum_root_translation_step_m:
        violations.append(
            (
                "mechanical_bounds",
                "root translation step "
                f"{maximum_root_step:.6f} m exceeds bound",
            )
        )
    if maximum_rotation_step > maximum_root_rotation_step_rad:
        violations.append(
            (
                "mechanical_bounds",
                "root rotation step "
                f"{maximum_rotation_step:.6f} rad exceeds bound",
            )
        )
    if maximum_root_acceleration > maximum_root_acceleration_m_s2:
        violations.append(
            (
                "mechanical_bounds",
                "root acceleration "
                f"{maximum_root_acceleration:.6f} m/s^2 exceeds bound",
            )
        )
    if maximum_joint_step > maximum_interframe_joint_step_rad:
        joint_steps = np.abs(np.diff(adapted_joints, axis=0))
        step_frame, step_joint = np.unravel_index(
            int(np.argmax(joint_steps)), joint_steps.shape
        )
        violations.append(
            (
                "mechanical_bounds",
                "joint step "
                f"{maximum_joint_step:.6f} rad at frame "
                f"{int(step_frame) + 1}, joint {joint_names[int(step_joint)]} "
                "exceeds bound",
            )
        )

    source_switches = tuple(
        index
        for index in range(1, len(stitched.provenance))
        if (
            stitched.provenance[index - 1].archive_clip_index
            != stitched.provenance[index].archive_clip_index
        )
    )
    fragment_seam_step = _maximum_joint_step_at_indices(
        adapted_joints,
        stitched.seam_indices,
    )
    source_switch_step = _maximum_joint_step_at_indices(
        adapted_joints,
        source_switches,
    )
    if fragment_seam_step > maximum_fragment_seam_joint_step_rad:
        violations.append(
            (
                "mechanical_bounds",
                "fragment-seam joint step "
                f"{fragment_seam_step:.6f} rad exceeds bound",
            )
        )
    if source_switch_step > maximum_source_switch_joint_step_rad:
        violations.append(
            (
                "mechanical_bounds",
                "source-switch joint step "
                f"{source_switch_step:.6f} rad exceeds bound",
            )
        )
    if violations:
        stage = (
            "target_support"
            if any(value[0] == "target_support" for value in violations)
            else "mechanical_bounds"
        )
        message = "; ".join(
            f"[{violation_stage}] {detail}"
            for violation_stage, detail in violations
        )
        raise PlannedFragmentReconstructionRejected(stage, message)
    route_adjustment = np.linalg.norm(warped_root - requested_root, axis=1)
    target_start = np.asarray(target_route.start_xy, dtype=np.float64)
    target_lateral = np.asarray(
        (-warp.target_direction_xy[1], warp.target_direction_xy[0])
    )
    route_lateral_deviation = np.abs(
        (warped_root[:, :2] - target_start) @ target_lateral
    )
    motion = StitchedMotion(
        fps=stitched.fps,
        root_position_world=np.asarray(warped_root, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(warped_quaternions, dtype=np.float32),
        joint_position=np.asarray(adapted_joints, dtype=np.float32),
        provenance=stitched.provenance,
        seam_indices=stitched.seam_indices,
    )
    return StairFragmentReconstructionResult(
        motion=motion,
        source_route=source_route,
        target_route=target_route,
        source_transition_fractions=fractions,
        source_switch_indices=source_switches,
        maximum_joint_correction_rad=maximum_correction,
        maximum_foot_target_error_m=maximum_error,
        maximum_stance_foot_target_error_m=maximum_stance_error,
        maximum_swing_foot_target_error_m=maximum_swing_error,
        maximum_sole_penetration_m=maximum_penetration,
        maximum_triangle_sphere_penetration_m=maximum_triangle_penetration,
        minimum_sole_clearance_m=minimum_clearance,
        maximum_root_clearance_lift_m=float(np.max(clearance_lift)),
        maximum_swing_foot_clearance_lift_m=float(np.max(swing_lift)),
        maximum_root_route_adjustment_m=float(np.max(route_adjustment)),
        maximum_root_route_lateral_deviation_m=float(np.max(route_lateral_deviation)),
        minimum_stance_support_point_count=int(np.min(advertised_stance_counts)),
        maximum_root_translation_step_m=maximum_root_step,
        maximum_root_rotation_step_rad=maximum_rotation_step,
        maximum_root_acceleration_m_s2=maximum_root_acceleration,
        maximum_joint_step_rad=maximum_joint_step,
        maximum_fragment_seam_joint_step_rad=fragment_seam_step,
        maximum_source_switch_joint_step_rad=source_switch_step,
        per_frame_joint_correction_rad=np.asarray(corrections, dtype=np.float32),
        per_frame_foot_target_error_m=np.asarray(errors, dtype=np.float32),
        per_frame_foot_target_error_by_foot_m=np.asarray(
            errors_by_foot, dtype=np.float32
        ),
        per_frame_minimum_sole_clearance_m=np.asarray(minimum_clearances, dtype=np.float32),
        per_frame_triangle_sphere_penetration_m=np.asarray(triangle_penetrations, dtype=np.float32),
        per_frame_root_clearance_lift_m=np.asarray(clearance_lift, dtype=np.float32),
        per_frame_swing_foot_clearance_lift_m=np.asarray(swing_lift, dtype=np.float32),
        per_frame_swing_route_adjustment=np.asarray(
            swing_route_adjustments, dtype=np.float32
        ),
        per_frame_stance_support_point_count=np.asarray(stance_support_counts, dtype=np.int16),
        per_frame_sole_target_world_xyz=np.asarray(
            target_soles_by_frame, dtype=np.float32
        ),
        per_frame_stance_mask=np.asarray(stance, dtype=bool),
        foothold_anchor_diagnostics=anchor_diagnostics,
    )


__all__ = (
    "ExactStairFragmentBank",
    "ExactStairFragmentGeometry",
    "PlannedFragmentReconstructionRejected",
    "StairFragmentReconstructionResult",
    "build_synthetic_source_support_route",
    "enrich_stair_fragment_bank_from_exact_mesh",
    "reconstruct_stair_fragment_sequence",
)
