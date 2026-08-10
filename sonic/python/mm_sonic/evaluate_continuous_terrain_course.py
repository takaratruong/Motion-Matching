"""Evaluate continuous motion matching on a generated mixed heightfield course."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from .joints import ContractError
from .canonical_terrain_matcher import (
    ContinuousTerrainMotionMatcher,
    ContinuousTerrainRuntimeConfig,
    GenericTerrainSearchConfig,
    RegularGridHeightField,
    load_canonical_terrain_library,
)
from .terrain_oracle.storage import load_corpus
from .terrain_oracle.contact import SoleGeometry
from .torch_motion_matcher import ForcedMotionAlignment, MatcherConfig


DEFAULT_CORPUS = Path(
    "/move/data/terrain-aware/motion-matching/terrain-oracle-task9.wFKkIV/raw"
)
FAMILIES = (
    "grail/c490_curb/",
    "grail/c490_slope/",
    "grail/c490_stair_p1/",
    "grail/c490_stair_p2/",
)
MOTIONBRICKS_FAMILY = "motionbricks/"
DEFAULT_MODEL = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/g1_29dof_rev_1_0.xml"
)
FLAT_CLIPS = (
    "flat/takara_walk",
    "flat/walk_forward_start_002__A021",
    "flat/walk_forward_start_002__A021_M",
    "flat/walk_sideway_135_start_002__A021",
    "flat/walk_sideway_135_start_002__A021_M",
)


def _yaw(value: object) -> float:
    w, x, y, z = (float(item) for item in np.asarray(value))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _evenly_spaced(values: list[str], count: int) -> list[str]:
    count = min(max(0, int(count)), len(values))
    if not count:
        return []
    indices = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return [values[int(index)] for index in indices]


def _balanced_ids(
    corpus: Path, per_family: int, motionbricks_count: int = 24
) -> list[str]:
    manifest = load_corpus(corpus)
    available = {record.clip_id for record in manifest.clips}
    selected = [clip_id for clip_id in FLAT_CLIPS if clip_id in available]
    for prefix in FAMILIES:
        family = sorted(
            record.clip_id
            for record in manifest.clips
            if record.clip_id.startswith(prefix)
        )
        selected.extend(_evenly_spaced(family, per_family))
    motionbricks = sorted(
        record.clip_id
        for record in manifest.clips
        if record.clip_id.startswith(MOTIONBRICKS_FAMILY)
        and "human_random" not in record.clip_id
    )
    native = [clip_id for clip_id in motionbricks if not clip_id.endswith("__mirror")]
    mirrored = [clip_id for clip_id in motionbricks if clip_id.endswith("__mirror")]
    native_count = (int(motionbricks_count) + 1) // 2
    mirror_count = int(motionbricks_count) // 2
    native_selection = _evenly_spaced(native, native_count)
    mirror_selection = _evenly_spaced(mirrored, mirror_count)
    for index in range(max(len(native_selection), len(mirror_selection))):
        if index < len(native_selection):
            selected.append(native_selection[index])
        if index < len(mirror_selection):
            selected.append(mirror_selection[index])
    return list(dict.fromkeys(selected))


def build_mixed_course(
    *, spacing_m: float = 0.05
) -> tuple[RegularGridHeightField, dict[str, object]]:
    """One continuous flat/ramp/rough/stair/crest/descent heightmap."""

    # Keep a generous flat exit apron after the rough patch ends at x=9.6 m.
    # Candidate search includes future root/foot horizons, so a 0.4 m apron
    # made an otherwise valid traversal run off the finite query domain before
    # the character itself reached the end of the course.
    x = np.arange(-3.5, 13.5 + spacing_m / 2.0, spacing_m)
    y = np.arange(-2.0, 2.0 + spacing_m / 2.0, spacing_m)
    grid_x, grid_y = np.meshgrid(x, y)
    height = np.zeros_like(grid_x)

    # Gentle approach ramp: 0 -> 0.30 m.
    ramp = np.clip((grid_x + 1.5) / 1.5, 0.0, 1.0)
    height += 0.30 * ramp

    # Broad rolling patch; unlike a labelled obstacle this simply changes the
    # local height/normal signal available to every candidate.
    rolling_window = np.clip((grid_x - 0.1) / 0.35, 0.0, 1.0) * np.clip(
        (1.3 - grid_x) / 0.35, 0.0, 1.0
    )
    height += rolling_window * (
        0.045 * np.sin(2.0 * math.pi * (grid_x - 0.1) / 0.75)
        + 0.025 * np.sin(2.0 * math.pi * grid_y / 1.4)
    )

    # Five irregular-height treads, then a crest.  The representation is a
    # heightmap, so the one-cell boundaries are steep ramps rather than portal
    # events; search remains identical everywhere.
    tread_edges = (1.35, 1.68, 2.04, 2.34, 2.72)
    tread_rises = (0.13, 0.16, 0.12, 0.17, 0.14)
    for edge, rise in zip(tread_edges, tread_rises, strict=True):
        height += rise * np.clip((grid_x - edge) / spacing_m, 0.0, 1.0)

    # Cross-slope on the crest tests terrain normals as well as heights.
    crest_window = np.clip((grid_x - 2.8) / 0.35, 0.0, 1.0) * np.clip(
        (4.4 - grid_x) / 0.35, 0.0, 1.0
    )
    height += crest_window * 0.055 * grid_y

    # Irregular descent, followed by a smooth down-ramp to flat.
    down_edges = (4.45, 4.82, 5.15, 5.53, 5.86)
    for edge, rise in zip(down_edges, tread_rises[::-1], strict=True):
        height -= rise * np.clip((grid_x - edge) / spacing_m, 0.0, 1.0)
    down_ramp = np.clip((grid_x - 6.2) / 1.8, 0.0, 1.0)
    height -= 0.30 * down_ramp

    # Small bounded roughness after the descent.
    rough_window = np.clip((grid_x - 8.0) / 0.35, 0.0, 1.0) * np.clip(
        (9.6 - grid_x) / 0.35, 0.0, 1.0
    )
    height += rough_window * (
        0.035 * np.sin(5.1 * grid_x + 1.7 * grid_y)
        + 0.020 * np.sin(8.3 * grid_x - 3.9 * grid_y)
    )
    field = RegularGridHeightField(
        height,
        spacing_m=spacing_m,
        origin_xy=(float(x[0]), float(y[0])),
    )
    metadata = {
        "spacing_m": spacing_m,
        "origin_xy": [float(x[0]), float(y[0])],
        "shape": list(height.shape),
        "x_range_m": [float(x[0]), float(x[-1])],
        "y_range_m": [float(y[0]), float(y[-1])],
        "height_range_m": [float(height.min()), float(height.max())],
        "segments": [
            "flat approach",
            "ramp",
            "rolling patch",
            "irregular ascent treads",
            "cross-sloped crest",
            "irregular descent treads",
            "down-ramp",
            "rough patch",
        ],
    }
    return field, metadata


def build_flat_course(
    *, spacing_m: float = 0.05
) -> tuple[RegularGridHeightField, dict[str, object]]:
    """Large closed-loop arena for isolated two-stick source coverage."""

    x = np.arange(-3.5, 10.0 + spacing_m / 2.0, spacing_m)
    y = np.arange(-6.0, 6.0 + spacing_m / 2.0, spacing_m)
    height = np.zeros((len(y), len(x)), dtype=np.float64)
    field = RegularGridHeightField(
        height,
        spacing_m=spacing_m,
        origin_xy=(float(x[0]), float(y[0])),
    )
    return field, {
        "spacing_m": spacing_m,
        "origin_xy": [float(x[0]), float(y[0])],
        "shape": list(height.shape),
        "x_range_m": [float(x[0]), float(x[-1])],
        "y_range_m": [float(y[0]), float(y[-1])],
        "height_range_m": [0.0, 0.0],
        "segments": ["flat closed-loop two-stick arena"],
    }


def _reset_alignment(
    source_clip: object,
    source_frame: int,
    source_field: object,
    target_field: object,
    start_xy: tuple[float, float],
    target_yaw_rad: float = 0.0,
) -> ForcedMotionAlignment:
    source_position = np.asarray(source_clip.root_position_world[source_frame], np.float64)
    source_yaw = _yaw(source_clip.root_quaternion_world_wxyz[source_frame])
    source_surface, _normal, source_hit = source_field.sample(
        np.asarray(source_clip.sole_position_world[source_frame, :, :2])
    )
    if not np.all(source_hit):
        raise ValueError("flat reset row has no source support")
    target_surface, _normal, target_hit = target_field.sample(
        np.asarray((start_xy,), dtype=np.float64)
    )
    if not bool(target_hit[0]):
        raise ValueError("course start is outside the target terrain")
    clearance = source_position[2] - float(np.median(source_surface))
    target_position = np.asarray(
        (start_xy[0], start_xy[1], float(target_surface[0]) + clearance)
    )
    yaw_offset = float(target_yaw_rad) - source_yaw
    cosine = math.cos(yaw_offset)
    sine = math.sin(yaw_offset)
    rotated = np.asarray(
        (
            cosine * source_position[0] - sine * source_position[1],
            sine * source_position[0] + cosine * source_position[1],
            source_position[2],
        )
    )
    return ForcedMotionAlignment(
        yaw_offset_rad=yaw_offset,
        translation_world_xyz=tuple(float(value) for value in target_position - rotated),
        synchronize_simulation_character=False,
    )


def _command_for_step(
    profile: str,
    *,
    step: int,
    root_position_world: np.ndarray,
    speed_mps: float,
    start_x: float,
    two_stick_leg_steps: int = 80,
) -> tuple[tuple[float, float], float, float]:
    """Return world velocity, independent facing, and path target Y."""

    if profile == "straight":
        return (speed_mps, 0.0), 0.0, 0.0
    if profile == "side_on_left":
        # Traverse the terrain along +X while the body faces +Y.  This is a
        # genuine 90-degree travel/facing request, not a pelvis-yaw relabel.
        return (speed_mps, 0.0), math.pi / 2.0, 0.0
    if profile == "side_on_right":
        return (speed_mps, 0.0), -math.pi / 2.0, 0.0
    if profile == "backward":
        # The travel stick remains forward through the course while the body
        # faces exactly backward, requiring a real backward terrain gait.
        return (speed_mps, 0.0), math.pi, 0.0
    if profile == "two_stick":
        # Eight 1.6-second legs form a bounded octagonal loop.  Facing stays
        # fixed while travel spans 0, 45, 90, 135, and 180 degrees in both
        # lateral directions.  Abrupt user commands are intentionally passed
        # through the same runtime command shaping as an interactive stick.
        travel_angles = np.deg2rad(
            np.asarray((0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0))
        )
        if two_stick_leg_steps <= 0:
            raise ValueError("two-stick leg duration must be positive")
        angle = float(
            travel_angles[
                (step // int(two_stick_leg_steps)) % len(travel_angles)
            ]
        )
        return (
            (speed_mps * math.cos(angle), speed_mps * math.sin(angle)),
            0.0,
            0.0,
        )
    if profile != "steering":
        raise ValueError(f"unsupported command profile {profile!r}")
    root_x, root_y = (float(value) for value in root_position_world[:2])
    phase = 0.72 * (root_x - start_x)
    target_y = 0.62 * math.sin(phase)
    tangent = 0.62 * 0.72 * math.cos(phase)
    tangent_norm = math.sqrt(1.0 + tangent * tangent)
    feedforward_x = speed_mps / tangent_norm
    feedforward_y = speed_mps * tangent / tangent_norm
    requested_y = float(
        np.clip(
            feedforward_y + 0.85 * (target_y - root_y),
            -0.32,
            0.32,
        )
    )
    requested_x = math.sqrt(max(speed_mps * speed_mps - requested_y**2, 0.04))
    travel_yaw = math.atan2(requested_y, requested_x)
    # Exercise the two-stick interface: facing smoothly diverges from travel
    # direction while the route itself keeps curving across the terrain.
    facing_yaw = travel_yaw + math.radians(35.0) * math.sin(0.55 * step * 0.02)
    return (requested_x, requested_y), facing_yaw, target_y


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "the exact G1 sole-footprint terrain search requires mujoco"
        ) from error
    sole_geometry = SoleGeometry.from_model(
        mujoco.MjModel.from_xml_path(str(args.model))
    )
    source_ids = list(args.source_clip_id) or _balanced_ids(
        args.corpus,
        args.source_per_family,
        args.motionbricks_source_count,
    )
    library = load_canonical_terrain_library(args.corpus, clip_ids=source_ids)
    if args.course_profile == "flat":
        field, course = build_flat_course(spacing_m=args.terrain_spacing_m)
    else:
        field, course = build_mixed_course(spacing_m=args.terrain_spacing_m)
    matcher = ContinuousTerrainMotionMatcher.from_library(
        library,
        field,
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
            prospective_foot_clearance_weight=(
                args.prospective_foot_clearance_weight
            ),
            maximum_prospective_flight_clearance_deficit_m=(
                args.maximum_prospective_flight_clearance_deficit_m
            ),
            minimum_prospective_foot_clearance_m=(
                args.minimum_prospective_foot_clearance_m
            ),
            transition_penalty=args.transition_penalty,
            compatible_transition_penalty=(
                args.compatible_transition_penalty
            ),
            contact_timing_weight=args.contact_timing_weight,
            joint_position_weight=args.joint_position_weight,
            joint_velocity_weight=args.joint_velocity_weight,
        ),
        runtime_config=ContinuousTerrainRuntimeConfig(
            vertical_adaptation=not args.disable_vertical_adaptation,
            output_grounding=args.output_grounding,
            vertical_halflife_s=args.vertical_halflife_s,
            maximum_vertical_step_m=args.maximum_vertical_step_m,
            terrain_command_adaptation=(
                not args.disable_terrain_command_adaptation
            ),
        ),
        sole_geometry=sole_geometry,
    )
    flat_id = args.reset_clip_id
    if flat_id is None:
        flat_id = next(
            clip_id for clip_id in FLAT_CLIPS if clip_id in library.clip_ids
        )
    if flat_id not in library.clip_ids:
        raise ValueError(f"reset clip is not in the source library: {flat_id}")
    flat_index = library.clip_ids.index(flat_id)
    reset_frame = int(args.reset_frame)
    reset_row = matcher.database.row_for_source(flat_index, reset_frame)
    if reset_row is None:
        raise ValueError("flat reset frame is not searchable")
    alignment = _reset_alignment(
        library.canonical_clips[flat_index],
        reset_frame,
        library.height_fields[flat_index],
        field,
        (args.start_x, args.start_y),
        target_yaw_rad=math.radians(args.start_yaw_deg),
    )
    reset = matcher.reset_to_row(int(reset_row), alignment=alignment)

    roots = [reset.root_position_world.detach().cpu().numpy()]
    quaternions = [reset.root_orientation_world_wxyz.detach().cpu().numpy()]
    joints = [reset.joint_position.detach().cpu().numpy()]
    foot_probes = []
    contacts = []
    source_sole_speeds = []
    selected_ids: list[str] = []
    selected_clip_indices: list[int] = []
    selected_source_frames: list[int] = []
    searches: list[dict[str, object]] = []
    requested_velocities: list[tuple[float, float]] = []
    requested_headings: list[float] = []
    target_path_y: list[float] = []
    applied_velocities: list[np.ndarray] = []
    applied_headings: list[float] = []
    path_to_index = {
        clip.relative_path: index for index, clip in enumerate(library.folder.clips)
    }
    foot_indices = [
        library.folder.layout.left_foot_body_index,
        library.folder.layout.right_foot_body_index,
    ]
    source_sole_speed_mps = tuple(
        np.linalg.norm(
            np.gradient(
                np.asarray(clip.sole_position_world, dtype=np.float64), axis=0
            )
            * float(clip.fps),
            axis=2,
        )
        for clip in library.canonical_clips
    )
    for step in range(args.steps):
        requested_velocity, requested_heading, target_y = _command_for_step(
            args.command_profile,
            step=step,
            root_position_world=roots[-1],
            speed_mps=args.speed_mps,
            start_x=args.start_x,
            two_stick_leg_steps=args.two_stick_leg_steps,
        )
        requested_velocities.append(requested_velocity)
        requested_headings.append(requested_heading)
        target_path_y.append(target_y)
        try:
            result = matcher.step(requested_velocity, requested_heading)
        except ContractError as error:
            raise ContractError(
                "continuous course search failed at "
                f"step {step}, root_xy={tuple(float(value) for value in roots[-1][:2])}: "
                f"{error}"
            ) from error
        applied_velocities.append(
            result.applied_command_velocity_world_xy.detach().cpu().numpy()
        )
        applied_headings.append(
            float(result.applied_command_heading_world_yaw.detach().cpu().item())
        )
        clip_index = path_to_index[result.diagnostics.selected_clip_path]
        selected_ids.append(library.clip_ids[clip_index])
        source_frame = result.diagnostics.selected_frame
        selected_clip_indices.append(clip_index)
        selected_source_frames.append(source_frame)
        contacts.append(
            np.asarray(
                library.canonical_clips[clip_index].contact[source_frame] >= 0.5,
                dtype=np.bool_,
            )
        )
        source_sole_speeds.append(
            source_sole_speed_mps[clip_index][source_frame]
        )
        roots.append(result.root_position_world.detach().cpu().numpy())
        quaternions.append(
            result.root_orientation_world_wxyz.detach().cpu().numpy()
        )
        joints.append(result.joint_position.detach().cpu().numpy())
        foot_probes.append(
            matcher._generated_foot_probe_positions(result).detach().cpu().numpy()
        )
        event = matcher.last_search_event
        if event is not None and event.sequence == result.diagnostics.sequence:
            searches.append(
                {
                    "sequence": event.sequence,
                    "source_clip_id": event.source_clip_id,
                    "selected_frame": event.selection.selected_frame_index,
                    "transitioned": event.transitioned,
                    "terrain_rms_m": event.selection.terrain_rms_m,
                    "minimum_prospective_foot_clearance_m": (
                        event.selection.minimum_prospective_foot_clearance_m
                    ),
                    "maximum_prospective_flight_clearance_deficit_m": (
                        event.selection.maximum_prospective_flight_clearance_deficit_m
                    ),
                    "feature_cost": event.selection.feature_cost,
                    "terrain_cost": event.selection.terrain_cost,
                    "contact_cost": event.selection.contact_cost,
                    "pose_cost": event.selection.pose_cost,
                    "total_cost": event.selection.total_cost,
                }
            )

    root = np.asarray(roots, dtype=np.float64)
    quaternion = np.asarray(quaternions, dtype=np.float64)
    joint = np.asarray(joints, dtype=np.float64)
    foot_probe = np.asarray(foot_probes, dtype=np.float64)
    foot = foot_probe[:, :, 0]
    contact = np.asarray(contacts, dtype=np.bool_)
    height, _normal, hit = field.sample(foot_probe[..., :2])
    clearance = foot_probe[..., 2] - height
    contact_probe_mask = contact[:, :, None] & hit
    contact_clearance = clearance[contact_probe_mask]
    planted = np.linalg.norm(np.diff(foot[..., :2], axis=0), axis=-1)
    planted = planted[contact[1:] & contact[:-1]]
    root_step = np.linalg.norm(np.diff(root, axis=0), axis=-1)
    root_yaw = np.asarray([_yaw(value) for value in quaternion])
    yaw_step = np.abs(np.arctan2(np.sin(np.diff(root_yaw)), np.cos(np.diff(root_yaw))))
    joint_step = np.max(np.abs(np.diff(joint, axis=0)), axis=-1)
    acceleration = np.linalg.norm(np.diff(root, n=2, axis=0) / 0.0004, axis=-1)
    velocity = root_step / 0.02
    requested_velocity_array = np.asarray(requested_velocities, dtype=np.float64)
    requested_heading_array = np.asarray(requested_headings, dtype=np.float64)
    target_path_y_array = np.asarray(target_path_y, dtype=np.float64)
    applied_velocity_array = np.asarray(applied_velocities, dtype=np.float64)
    applied_heading_array = np.asarray(applied_headings, dtype=np.float64)
    actual_velocity_xy = np.diff(root[:, :2], axis=0) / 0.02
    velocity_tracking_rmse = float(
        np.sqrt(np.mean(np.square(actual_velocity_xy - requested_velocity_array)))
    )
    heading_error = np.abs(
        np.arctan2(
            np.sin(root_yaw[1:] - requested_heading_array),
            np.cos(root_yaw[1:] - requested_heading_array),
        )
    )
    terrain_under_root, _normal, root_hit = field.sample(root[:, :2])
    pelvis_clearance = root[:, 2] - terrain_under_root

    def max_or_zero(value: np.ndarray) -> float:
        return float(np.max(value)) if value.size else 0.0

    metrics = {
        "steps": args.steps,
        "duration_s": args.steps * 0.02,
        "progress_x_m": float(root[-1, 0] - root[0, 0]),
        "path_length_m": float(
            np.sum(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1))
        ),
        "lateral_displacement_m": float(root[-1, 1] - root[0, 1]),
        "height_gain_m": float(root[-1, 2] - root[0, 2]),
        "maximum_reached_terrain_height_m": float(np.nanmax(terrain_under_root)),
        "minimum_pelvis_clearance_m": float(np.nanmin(pelvis_clearance)),
        "terrain_query_hit_fraction": float(np.mean(hit)),
        "maximum_contact_penetration_m": max(0.0, max_or_zero(-contact_clearance)),
        "maximum_contact_hover_m": max(0.0, max_or_zero(contact_clearance)),
        "maximum_planted_foot_drift_m_per_frame": max_or_zero(planted),
        "maximum_root_step_m": max_or_zero(root_step),
        "maximum_root_yaw_step_rad": max_or_zero(yaw_step),
        "maximum_joint_step_rad": max_or_zero(joint_step),
        "maximum_root_acceleration_mps2": max_or_zero(acceleration),
        "near_stationary_fraction": float(np.mean(velocity < 0.03)),
        "lateral_span_m": float(np.ptp(root[:, 1])),
        "requested_lateral_speed_max_mps": float(
            np.max(np.abs(requested_velocity_array[:, 1]))
        ),
        "velocity_tracking_rmse_mps": velocity_tracking_rmse,
        "heading_tracking_mae_rad": float(np.mean(heading_error)),
        "search_count": len(searches),
        "transition_count": sum(bool(row["transitioned"]) for row in searches),
        "selected_clip_count": len(set(selected_ids)),
        "selected_motionbricks_frame_count": sum(
            value.startswith(MOTIONBRICKS_FAMILY) for value in selected_ids
        ),
        "selected_family_histogram": dict(
            Counter("/".join(value.split("/")[:2]) for value in selected_ids).most_common()
        ),
        "search_terrain_rms_mean_m": float(
            np.mean([row["terrain_rms_m"] for row in searches])
        ),
        "search_terrain_rms_p95_m": float(
            np.quantile([row["terrain_rms_m"] for row in searches], 0.95)
        ),
        "search_minimum_prospective_foot_clearance_m": float(
            np.min(
                [
                    row["minimum_prospective_foot_clearance_m"]
                    for row in searches
                ]
            )
        ),
        "search_maximum_prospective_flight_clearance_deficit_m": float(
            np.max(
                [
                    row["maximum_prospective_flight_clearance_deficit_m"]
                    for row in searches
                ]
            )
        ),
    }
    gates = {
        "terrain_queries_hit": bool(np.all(hit) and np.all(root_hit)),
        "progress_at_least_1m": metrics["progress_x_m"] >= 1.0,
        "lateral_displacement_le_0p75m": abs(metrics["lateral_displacement_m"]) <= 0.75,
        "pelvis_clearance_at_least_0p45m": metrics["minimum_pelvis_clearance_m"] >= 0.45,
        "contact_penetration_le_5mm": metrics["maximum_contact_penetration_m"] <= 0.005,
        "planted_drift_le_10mm": metrics["maximum_planted_foot_drift_m_per_frame"] <= 0.010,
        "root_step_le_60mm": metrics["maximum_root_step_m"] <= 0.060,
        "root_yaw_step_le_0p35rad": metrics["maximum_root_yaw_step_rad"] <= 0.35,
        "joint_step_le_0p25rad": metrics["maximum_joint_step_rad"] <= 0.25,
        "root_acceleration_le_40mps2": metrics["maximum_root_acceleration_mps2"] <= 40.0,
        "near_stationary_fraction_le_0p25": metrics["near_stationary_fraction"] <= 0.25,
    }
    if args.command_profile == "steering":
        gates["lateral_span_at_least_0p35m"] = metrics["lateral_span_m"] >= 0.35
    if args.command_profile == "two_stick":
        gates.pop("progress_at_least_1m")
        gates["closed_loop_path_length_at_least_2m"] = bool(
            metrics["path_length_m"] >= 2.0
        )
        travel_yaw = np.arctan2(
            requested_velocity_array[:, 1], requested_velocity_array[:, 0]
        )
        separation = np.abs(
            np.arctan2(
                np.sin(travel_yaw - requested_heading_array),
                np.cos(travel_yaw - requested_heading_array),
            )
        )
        metrics["maximum_requested_travel_facing_separation_rad"] = float(
            np.max(separation)
        )
        gates["two_stick_separation_at_least_2p8rad"] = bool(
            np.max(separation) >= 2.8
        )
        gates["motionbricks_source_is_exercised"] = bool(
            metrics["selected_motionbricks_frame_count"] > 0
        )
    report = {
        "schema": "continuous-terrain-procedural-course-eval/v1",
        "corpus": str(args.corpus.resolve()),
        "course": course,
        "course_profile": args.course_profile,
        "command_profile": args.command_profile,
        "source_clip_ids": list(library.clip_ids),
        "source_row_count": matcher.database.feature_shape[0],
        "metrics": metrics,
        "gates": gates,
        "passed": bool(all(gates.values())),
        "searches": searches,
    }
    np.savez_compressed(
        args.output.with_suffix(".trace.npz"),
        root_position_world=root.astype(np.float32),
        root_quaternion_world_wxyz=quaternion.astype(np.float32),
        joint_position=joint.astype(np.float32),
        foot_position_world=foot.astype(np.float32),
        foot_probe_position_world=foot_probe.astype(np.float32),
        contact=contact,
        source_sole_speed_mps=np.asarray(
            source_sole_speeds, dtype=np.float32
        ),
        requested_velocity_world_xy=requested_velocity_array.astype(np.float32),
        requested_heading_world_yaw=requested_heading_array.astype(np.float32),
        applied_velocity_world_xy=applied_velocity_array.astype(np.float32),
        applied_heading_world_yaw=applied_heading_array.astype(np.float32),
        target_path_y=target_path_y_array.astype(np.float32),
        selected_clip_index=np.asarray(selected_clip_indices, dtype=np.int32),
        selected_source_frame=np.asarray(selected_source_frames, dtype=np.int32),
    )
    np.savez_compressed(
        args.output.with_suffix(".terrain.npz"),
        height=field.height.astype(np.float32),
        valid=field.valid,
        spacing_m=np.asarray(field.spacing_m, dtype=np.float32),
        origin_xy=np.asarray(field.origin_xy, dtype=np.float32),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--source-clip-id", action="append", default=[])
    parser.add_argument("--reset-clip-id")
    parser.add_argument("--reset-frame", type=int, default=0)
    parser.add_argument("--source-per-family", type=int, default=8)
    parser.add_argument("--motionbricks-source-count", type=int, default=24)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--speed-mps", type=float, default=0.40)
    parser.add_argument("--two-stick-leg-steps", type=int, default=80)
    parser.add_argument(
        "--command-profile",
        choices=(
            "straight",
            "steering",
            "two_stick",
            "side_on_left",
            "side_on_right",
            "backward",
        ),
        default="straight",
    )
    parser.add_argument(
        "--course-profile", choices=("mixed", "flat"), default="mixed"
    )
    parser.add_argument("--start-x", type=float, default=-3.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    parser.add_argument("--terrain-spacing-m", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--search-interval-steps", type=int, default=5)
    parser.add_argument("--inertialization-halflife-s", type=float, default=0.10)
    parser.add_argument("--preselection-count", type=int, default=4096)
    parser.add_argument("--terrain-height-weight", type=float, default=8.0)
    parser.add_argument("--terrain-normal-weight", type=float, default=0.75)
    parser.add_argument(
        "--prospective-foot-clearance-weight", type=float, default=8.0
    )
    parser.add_argument(
        "--maximum-prospective-flight-clearance-deficit-m",
        type=float,
        default=math.inf,
    )
    parser.add_argument(
        "--minimum-prospective-foot-clearance-m", type=float, default=-0.12
    )
    parser.add_argument("--transition-penalty", type=float, default=8.0)
    parser.add_argument(
        "--compatible-transition-penalty", type=float, default=8.0
    )
    parser.add_argument("--contact-timing-weight", type=float, default=4.0)
    parser.add_argument("--joint-position-weight", type=float, default=0.25)
    parser.add_argument("--joint-velocity-weight", type=float, default=0.05)
    parser.add_argument("--disable-vertical-adaptation", action="store_true")
    parser.add_argument(
        "--disable-terrain-command-adaptation",
        action="store_true",
        help=(
            "Preserve the exact two-stick travel/facing request even near "
            "steep terrain instead of rotating it toward the terrain normal."
        ),
    )
    parser.add_argument("--output-grounding", action="store_true")
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
