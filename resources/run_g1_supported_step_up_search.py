#!/usr/bin/env python3
"""Search GRAIL for a native split-height entry onto the target staircase."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_contact_segments import (
    ANKLE_ORIGIN_SOLE_M,
    STANCE_CLEARANCE_TOLERANCE_M,
    STANCE_VERTICAL_SPEED_MAX_MPS,
)
from mm_sonic.torch_g1_fk import MujocoG1FootKinematics
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_horizontal_terrain_retarget import (
    WideBoundG1TerrainRetargeter,
    nearest_valid_sole_translation,
)
from mm_sonic.torch_supported_step_up import (
    CompleteStepUpSequence,
    find_complete_step_up_sequence,
    smooth_swing_clearance_lift,
    swing_clearance_targets,
    validate_placed_step_up,
)
from mm_sonic.torch_terrain_features import _TorchHeightGrid
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)


_FEET = (18, 19)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument(
        "--source-clip",
        help="Optionally restrict the authenticated search to one logical clip.",
    )
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-rise-m", type=float, default=0.28)
    parser.add_argument("--maximum-rise-m", type=float, default=0.42)
    parser.add_argument("--approach-frames", type=int, default=35)
    parser.add_argument("--maximum-facing-error-deg", type=float, default=10.0)
    parser.add_argument(
        "--traversal-angle-deg",
        type=float,
        default=0.0,
        help=(
            "Angle from cross-tread travel toward stair ascent: "
            "0 is horizontal, 45 is diagonal, and 90 is head-on."
        ),
    )
    parser.add_argument(
        "--maximum-placements-per-event",
        type=int,
        default=0,
        help=(
            "Retain only this many geometrically nearest placements per "
            "source event; zero evaluates every matching placement."
        ),
    )
    return parser


def _contact_pattern_matches(
    *,
    source_final_height_delta_m: float,
    landing_scene_xy: object,
    trailing_offset_scene_xy: object,
    sample_height: Callable[[np.ndarray], np.ndarray],
    tolerance_m: float = 0.025,
    minimum_split_height_m: float = 0.080,
) -> tuple[bool, float]:
    landing = np.asarray(landing_scene_xy, dtype=np.float64)
    offset = np.asarray(trailing_offset_scene_xy, dtype=np.float64)
    if (
        landing.shape != (2,)
        or offset.shape != (2,)
        or not np.isfinite(landing).all()
        or not np.isfinite(offset).all()
        or not math.isfinite(float(source_final_height_delta_m))
        or not math.isfinite(float(tolerance_m))
        or not math.isfinite(float(minimum_split_height_m))
        or tolerance_m <= 0.0
        or minimum_split_height_m <= 0.0
    ):
        raise ContractError("step-up contact-pattern input is invalid")
    heights = np.asarray(
        sample_height(np.stack((landing, landing + offset))),
        dtype=np.float64,
    )
    if heights.shape != (2,) or not np.isfinite(heights).all():
        raise ContractError("step-up contact-pattern heights are invalid")
    target_delta = float(heights[1] - heights[0])
    return (
        abs(target_delta) >= minimum_split_height_m
        and abs(float(source_final_height_delta_m))
        >= minimum_split_height_m
        and abs(target_delta - float(source_final_height_delta_m))
        <= tolerance_m,
        target_delta,
    )


def _paired_contact_delta_matches(
    *, source_delta_m: float, target_delta_m: float
) -> bool:
    if not (
        math.isfinite(float(source_delta_m))
        and math.isfinite(float(target_delta_m))
    ):
        raise ContractError("paired step-up contact delta is invalid")
    return abs(float(source_delta_m) - float(target_delta_m)) <= (
        2.0 * float(STANCE_CLEARANCE_TOLERANCE_M)
    )


def _candidate_score(metrics: dict) -> float:
    frames = int(metrics["frame_count"])
    retargeted = int(metrics["retargeted_swing_frame_count"])
    if frames < 1 or not 0 <= retargeted <= frames:
        raise ContractError("step-up candidate fidelity metrics are invalid")
    return (
        float(metrics["contact_height_pattern_error_m"]) * 8.0
        + float(metrics["maximum_stance_contact_error_m"]) * 4.0
        + max(0.0, -float(metrics["minimum_sole_clearance_m"])) * 2.0
        + 1.0 * retargeted / frames
        + 2.0
        * float(metrics.get("maximum_support_footprint_shift_m", 0.0))
    )


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array(
        (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)),
        dtype=np.float64,
    )


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _yaw_from_wxyz(value: np.ndarray) -> float:
    w, x, y, z = (float(item) for item in value)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _angle_error(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


def _sample_numpy(grid: _TorchHeightGrid, points: np.ndarray) -> np.ndarray:
    return (
        grid.sample_xy(torch.tensor(points, dtype=torch.float32))
        .cpu()
        .numpy()
    )


def _support_aligned_root_shift(
    *,
    support_mask: object,
    ankle_height_m: object,
    target_surface_height_m: object,
) -> np.ndarray:
    support = np.asarray(support_mask)
    ankle = np.asarray(ankle_height_m, dtype=np.float64)
    target = np.asarray(target_surface_height_m, dtype=np.float64)
    if (
        support.dtype != np.bool_
        or support.ndim != 2
        or support.shape[1:] != (2,)
        or ankle.shape != support.shape
        or target.shape != support.shape
        or not np.isfinite(ankle).all()
        or not np.isfinite(target).all()
        or not bool(support.any(axis=1).all())
    ):
        raise ContractError("step-up stance normalization input is invalid")
    desired = target + float(ANKLE_ORIGIN_SOLE_M) - ankle
    segment_shift = np.full_like(desired, np.nan)
    for foot in range(2):
        padded = np.pad(support[:, foot], (1, 1))
        transitions = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        for begin, end in zip(starts, stops):
            segment_shift[begin:end, foot] = desired[begin, foot]
    return np.asarray(
        [
            float(segment_shift[frame, support[frame]].mean())
            for frame in range(len(support))
        ],
        dtype=np.float64,
    )


def _source_arrays(
    root: Path, descriptor: dict
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with np.load(
        root / descriptor["relative_motion_path"], allow_pickle=False
    ) as archive:
        arrays = {
            name: np.asarray(archive[name], dtype=np.float64)
            for name in (
                "joint_pos",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
            )
        }
    grid = _TorchHeightGrid.load(
        root / descriptor["terrain"]["path"], torch.device("cpu")
    )
    tx, ty, yaw = (
        float(value)
        for value in descriptor["terrain"]["motion_to_terrain_xy_yaw"]
    )
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.array(
        ((cosine, -sine), (sine, cosine)), dtype=np.float64
    )
    body = arrays["body_pos_w"]
    feet_scene = (
        body[:, _FEET, :2] @ rotation.T + np.array((tx, ty))
    )
    surface = _sample_numpy(grid, feet_scene)
    support = (
        np.abs(
            body[:, _FEET, 2]
            - surface
            - float(ANKLE_ORIGIN_SOLE_M)
        )
        <= float(STANCE_CLEARANCE_TOLERANCE_M)
    ) & (
        np.abs(arrays["body_lin_vel_w"][:, _FEET, 2])
        <= float(STANCE_VERTICAL_SPEED_MAX_MPS)
    )
    return arrays, surface, support


def _first_contact_events(
    *,
    arrays: dict[str, np.ndarray],
    surface: np.ndarray,
    support: np.ndarray,
    minimum_rise_m: float,
    maximum_rise_m: float,
    approach_frames: int,
    maximum_facing_error_rad: float,
) -> list[CompleteStepUpSequence]:
    body = arrays["body_pos_w"]
    quaternion = arrays["body_quat_w"][:, 0]
    frames = len(body)
    output = []
    for landing_foot in (0, 1):
        trailing_foot = 1 - landing_foot
        for first in range(3, frames - 3):
            if (
                not bool(support[first : first + 3, landing_foot].all())
                or bool(support[first - 3 : first, landing_foot].any())
                or not bool(support[first, trailing_foot])
            ):
                continue
            rise = float(
                surface[first, landing_foot]
                - surface[first, trailing_foot]
            )
            if not minimum_rise_m <= rise <= maximum_rise_m:
                continue
            start = max(0, first - approach_frames)
            travel = body[first, 0, :2] - body[start, 0, :2]
            progress = float(np.linalg.norm(travel))
            if progress < 0.08:
                continue
            travel_yaw = math.atan2(float(travel[1]), float(travel[0]))
            facing_error = _angle_error(
                travel_yaw, _yaw_from_wxyz(quaternion[start])
            )
            if facing_error > maximum_facing_error_rad:
                continue
            try:
                sequence = find_complete_step_up_sequence(
                    support_mask=support,
                    surface_height_m=surface,
                    start_frame=start,
                    first_contact_frame=first,
                    landing_foot=landing_foot,
                    stable_contact_frames=3,
                )
            except ContractError:
                continue
            output.append(sequence)
    return output


def _display_stop(
    support: np.ndarray, sequence: CompleteStepUpSequence
) -> int:
    del support
    return sequence.end_exclusive


def _scene_direction_angle(
    *, traversal_angle_deg: float, direction_sign: int
) -> float:
    if (
        not math.isfinite(float(traversal_angle_deg))
        or not 0.0 <= float(traversal_angle_deg) <= 90.0
        or direction_sign not in (-1, 1)
    ):
        raise ContractError("traversal heading is invalid")
    forward = -math.radians(float(traversal_angle_deg))
    return forward if direction_sign > 0 else forward + math.pi


def _preferred_landing_point(traversal_angle_deg: float) -> np.ndarray:
    _scene_direction_angle(
        traversal_angle_deg=traversal_angle_deg, direction_sign=1
    )
    fraction = float(traversal_angle_deg) / 90.0
    return np.array(
        (
            -0.36 * math.cos(math.radians(traversal_angle_deg)),
            0.32 + 0.30 * fraction,
        ),
        dtype=np.float64,
    )


def _target_landing_points(
    direction_sign: int, traversal_angle_deg: float
) -> tuple[np.ndarray, ...]:
    _scene_direction_angle(
        traversal_angle_deg=traversal_angle_deg,
        direction_sign=direction_sign,
    )
    if traversal_angle_deg == 0.0:
        x_values = (
            np.arange(-0.40, -0.279, 0.02)
            if direction_sign > 0
            else np.arange(0.28, 0.401, 0.02)
        )
        y_values = np.arange(0.30, 0.421, 0.02)
    else:
        # The target staircase occupies approximately x=[-.42,.44] and
        # y=[-.73,.73]. Two-centimetre placement resolution is necessary:
        # a G1 sole has only a narrow valid center interval on a 26 cm tread
        # when rotated diagonally.
        x_values = np.arange(-0.40, 0.401, 0.02)
        y_values = np.arange(-0.68, 0.681, 0.02)
    return tuple(
        np.array((x, y), dtype=np.float64)
        for x in x_values
        for y in y_values
    )


def _cheap_placements(
    *,
    arrays: dict[str, np.ndarray],
    surface: np.ndarray,
    sequence: CompleteStepUpSequence,
    alignment_yaw: float,
    traversal_angle_deg: float,
    sample_height: Callable[[np.ndarray], np.ndarray],
) -> list[dict]:
    body = arrays["body_pos_w"]
    start = sequence.start_frame
    first = sequence.first_contact_frame
    trailing = sequence.trailing_contact_frame
    landing_foot = sequence.landing_foot
    other = 1 - landing_foot
    source_travel = body[first, 0, :2] - body[start, 0, :2]
    source_angle = math.atan2(
        float(source_travel[1]), float(source_travel[0])
    )
    output = []
    for direction_sign in (1, -1):
        scene_direction_angle = _scene_direction_angle(
            traversal_angle_deg=traversal_angle_deg,
            direction_sign=direction_sign,
        )
        matcher_target_angle = scene_direction_angle - alignment_yaw
        yaw = matcher_target_angle - source_angle
        cosine = math.cos(yaw + alignment_yaw)
        sine = math.sin(yaw + alignment_yaw)
        matcher_to_scene_rotation = np.array(
            ((cosine, -sine), (sine, cosine)), dtype=np.float64
        )
        first_other_offset = matcher_to_scene_rotation @ (
            body[first, _FEET[other], :2]
            - body[first, _FEET[landing_foot], :2]
        )
        final_other_offset = matcher_to_scene_rotation @ (
            body[trailing, _FEET[other], :2]
            - body[first, _FEET[landing_foot], :2]
        )
        source_first_other_delta = float(
            surface[first, other] - surface[first, landing_foot]
        )
        landings = np.stack(
            _target_landing_points(direction_sign, traversal_angle_deg)
        )
        first_points = np.stack(
            (landings, landings + first_other_offset), axis=1
        )
        final_points = np.stack(
            (landings, landings + final_other_offset), axis=1
        )
        first_heights = np.asarray(
            sample_height(first_points), dtype=np.float64
        )
        final_heights = np.asarray(
            sample_height(final_points), dtype=np.float64
        )
        expected_shape = (len(landings), 2)
        if (
            first_heights.shape != expected_shape
            or final_heights.shape != expected_shape
            or not np.isfinite(first_heights).all()
            or not np.isfinite(final_heights).all()
        ):
            raise ContractError("batched step-up contact heights are invalid")
        first_delta = first_heights[:, 1] - first_heights[:, 0]
        final_delta = final_heights[:, 1] - final_heights[:, 0]
        matches = (
            np.abs(first_delta - source_first_other_delta)
            <= 2.0 * float(STANCE_CLEARANCE_TOLERANCE_M)
        ) & (
            np.abs(final_delta) >= 0.080
        ) & (
            abs(sequence.source_final_height_delta_m) >= 0.080
        ) & (
            np.abs(
                final_delta - sequence.source_final_height_delta_m
            )
            <= 0.025
        )
        for index in np.flatnonzero(matches):
            output.append(
                {
                    "direction_sign": direction_sign,
                    "traversal_angle_deg": traversal_angle_deg,
                    "yaw_matcher": yaw,
                    "landing_scene_xy": landings[index],
                    "target_first_height_m": float(
                        first_heights[index, 0]
                    ),
                    "target_final_height_delta_m": float(
                        final_delta[index]
                    ),
                }
            )
    return output


def _placed_candidate(
    *,
    arrays: dict[str, np.ndarray],
    surface: np.ndarray,
    support: np.ndarray,
    sequence: CompleteStepUpSequence,
    placement: dict,
    alignment: object,
    target_grid: _TorchHeightGrid,
    foot_kinematics: MujocoG1FootKinematics,
    sole_kinematics: MujocoG1SoleKinematics,
    terrain_retargeter: WideBoundG1TerrainRetargeter,
) -> tuple[dict, dict[str, np.ndarray]]:
    display_stop = _display_stop(support, sequence)
    start = sequence.start_frame
    local_first = sequence.first_contact_frame - start
    local_trailing = sequence.trailing_contact_frame - start
    joints = arrays["joint_pos"][start:display_stop].copy()
    roots = arrays["body_pos_w"][start:display_stop, 0].copy()
    quaternions = arrays["body_quat_w"][start:display_stop, 0].copy()
    local_support = support[start:display_stop].copy()
    landing_foot = sequence.landing_foot
    landing_stable_stop = min(
        len(local_support), local_trailing + 3
    )
    if not bool(
        local_support[
            local_trailing:landing_stable_stop, landing_foot
        ].all()
    ):
        # Treat a one-frame overlap as the dynamic transfer it is. Pinning
        # both feet independently at that instant creates an artificial
        # double-support constraint and a discontinuous IK branch.
        local_support[local_trailing:, landing_foot] = False
    yaw = float(placement["yaw_matcher"])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.array(
        ((cosine, -sine), (sine, cosine)), dtype=np.float64
    )
    roots[:, :2] = roots[:, :2] @ rotation.T
    yaw_quaternion = np.broadcast_to(
        _yaw_quaternion(yaw), quaternions.shape
    )
    quaternions = _quat_multiply(yaw_quaternion, quaternions)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)

    ankles = foot_kinematics.foot_positions(
        joints, roots, quaternions
    )
    soles = sole_kinematics.sole_points(joints, roots, quaternions)
    alignment_yaw = float(alignment.yaw_scene_from_matcher)
    cosine = math.cos(alignment_yaw)
    sine = math.sin(alignment_yaw)
    rotation_scene = np.array(
        ((cosine, -sine), (sine, cosine)), dtype=np.float64
    )
    translation = alignment.translation_scene_xy.cpu().numpy()
    ankle_scene = ankles[..., :2] @ rotation_scene.T + translation
    sole_scene = soles[..., :2] @ rotation_scene.T + translation
    scene_shift = (
        np.asarray(placement["landing_scene_xy"])
        - ankle_scene[local_first, landing_foot]
    )
    ankle_scene += scene_shift
    sole_scene += scene_shift
    matcher_shift = rotation_scene.T @ scene_shift
    roots[:, :2] += matcher_shift
    ankles[..., :2] += matcher_shift
    soles[..., :2] += matcher_shift

    ankle_surface = _sample_numpy(target_grid, ankle_scene)
    sole_surface = _sample_numpy(target_grid, sole_scene)
    root_height_shift = _support_aligned_root_shift(
        support_mask=local_support,
        ankle_height_m=ankles[..., 2],
        target_surface_height_m=ankle_surface,
    )
    roots[:, 2] += root_height_shift
    ankles[..., 2] += root_height_shift[:, None]
    soles[..., 2] += root_height_shift[:, None, None]
    ankle_clearance = (
        ankles[..., 2]
        - ankle_surface
        - float(ANKLE_ORIGIN_SOLE_M)
    )
    sole_clearance = soles[..., 2] - sole_surface
    base_ankle_xy = ankles[..., :2].copy()
    base_ankle_height = ankles[..., 2].copy()
    planned_swing_lift = (
        smooth_swing_clearance_lift(
            support_mask=local_support,
            minimum_sole_clearance_m=sole_clearance.min(axis=2),
            smoothing_radius_frames=16,
        )
        if float(placement["traversal_angle_deg"]) == 0.0
        else np.zeros_like(sole_clearance.min(axis=2))
    )
    support_target_xy = np.full(
        (len(joints), 2, 2), np.nan, dtype=np.float64
    )
    support_target_surface = np.full(
        (len(joints), 2), np.nan, dtype=np.float64
    )
    support_footprint_shift = np.zeros(
        (len(joints), 2), dtype=np.float64
    )
    planned_swing_xy = np.full(
        (len(joints), 2, 2), np.nan, dtype=np.float64
    )
    if float(placement["traversal_angle_deg"]) > 0.0:
        for foot in range(2):
            padded = np.pad(local_support[:, foot], (1, 1))
            transitions = np.diff(padded.astype(np.int8))
            starts = np.flatnonzero(transitions == 1)
            stops = np.flatnonzero(transitions == -1)
            for begin, end in zip(starts, stops):
                stable_stop = min(end, begin + 3)
                support_surface = float(
                    np.median(
                        ankle_surface[begin:stable_stop, foot]
                    )
                )
                footprint_shift_scene = nearest_valid_sole_translation(
                    sole_point_scene_xy=sole_scene[
                        begin:stable_stop, foot
                    ].reshape(-1, 2),
                    support_surface_height_m=support_surface,
                    sample_height=lambda points: _sample_numpy(
                        target_grid, points
                    ),
                    maximum_shift_m=0.12,
                    search_resolution_m=0.01,
                    safety_margin_m=0.025,
                )
                target_xy = (
                    ankles[begin, foot, :2]
                    + rotation_scene.T @ footprint_shift_scene
                )
                footprint_shift_matcher = (
                    rotation_scene.T @ footprint_shift_scene
                )
                target_scene_xy = (
                    target_xy @ rotation_scene.T + translation
                )
                target_surface = float(
                    _sample_numpy(
                        target_grid, target_scene_xy[None, :]
                    )[0]
                )
                support_target_xy[begin:end, foot] = target_xy
                support_target_surface[begin:end, foot] = target_surface
                support_footprint_shift[begin:end, foot] = float(
                    np.linalg.norm(footprint_shift_scene)
                )
                transition_frames = 48
                pre_start = max(0, begin - transition_frames)
                if begin > pre_start:
                    fraction = np.arange(
                        1, begin - pre_start + 1, dtype=np.float64
                    ) / float(begin - pre_start + 1)
                    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
                    planned_swing_xy[pre_start:begin, foot] = (
                        ankles[pre_start:begin, foot, :2]
                        + smooth[:, None] * footprint_shift_matcher
                    )
                post_stop = min(len(joints), end + transition_frames)
                if post_stop > end:
                    fraction = np.arange(
                        post_stop - end, 0, -1, dtype=np.float64
                    ) / float(post_stop - end + 1)
                    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
                    planned_swing_xy[end:post_stop, foot] = (
                        ankles[end:post_stop, foot, :2]
                        + smooth[:, None] * footprint_shift_matcher
                    )
    base_root_xy = roots[:, :2].copy()
    planned_swing_xy_offset = (
        planned_swing_xy - ankles[..., :2]
    )
    retargeted_frames = 0
    for frame in range(len(joints)):
        frame_retargeted = False
        for iteration in range(3):
            solve_feet, targets = swing_clearance_targets(
                foot_position_world=ankles[frame],
                support_mask=local_support[frame],
                minimum_sole_clearance_m=sole_clearance[frame].min(
                    axis=1
                ),
            )
            colliding_support = local_support[frame] & (
                sole_clearance[frame].min(axis=1) < -0.025
            )
            if bool(colliding_support.any()):
                solve_feet |= local_support[frame]
            planned_swing = (
                ~local_support[frame]
            ) & (planned_swing_lift[frame] > 0.0)
            if bool(planned_swing.any()):
                solve_feet |= planned_swing
                targets[planned_swing, 2] = np.maximum(
                    targets[planned_swing, 2],
                    base_ankle_height[frame, planned_swing]
                    + planned_swing_lift[frame, planned_swing],
                )
            footprint_support = np.isfinite(
                support_target_xy[frame]
            ).all(axis=1)
            planned_xy = np.isfinite(
                planned_swing_xy[frame]
            ).all(axis=1) & ~local_support[frame]
            if bool(planned_xy.any()):
                # Let the support-foot solve establish the body's translated
                # frame before steering the swing foot relative to it.
                if iteration == 0 and bool(footprint_support.any()):
                    solve_feet[planned_xy] = False
                    targets[planned_xy] = ankles[frame, planned_xy]
                else:
                    solve_feet |= planned_xy
                    root_delta_xy = (
                        roots[frame, :2] - base_root_xy[frame]
                    )
                    targets[planned_xy, :2] = (
                        base_ankle_xy[frame, planned_xy]
                        + planned_swing_xy_offset[frame, planned_xy]
                        + root_delta_xy
                    )
                    first_xy_solve = (
                        iteration == 1
                        if bool(footprint_support.any())
                        else iteration == 0
                    )
                    if first_xy_solve:
                        targets[planned_xy, 2] = ankles[
                            frame, planned_xy, 2
                        ]
            if iteration == 0 and bool(footprint_support.any()):
                solve_feet |= footprint_support
                targets[footprint_support, :2] = support_target_xy[
                    frame, footprint_support
                ]
                targets[footprint_support, 2] = (
                    support_target_surface[frame, footprint_support]
                    + float(ANKLE_ORIGIN_SOLE_M)
                )
            if not bool(solve_feet.any()):
                break
            ordinary_support = (
                local_support[frame] & ~footprint_support
            )
            targets[ordinary_support, 2] = (
                ankle_surface[frame, ordinary_support]
                + float(ANKLE_ORIGIN_SOLE_M)
            )
            try:
                joints[frame], roots[frame] = (
                    terrain_retargeter.solve_frame(
                    joint_position=joints[frame],
                    root_position_world=roots[frame],
                    root_orientation_world_wxyz=quaternions[frame],
                    solve_feet=solve_feet,
                    target_foot_position_world=targets,
                    level_feet=solve_feet & local_support[frame],
                    initial_joint_position=(
                        None
                        if frame == 0
                        else joints[frame]
                        if iteration > 0
                        else joints[frame - 1]
                    ),
                    initial_root_position_world=(
                        None
                        if frame == 0
                        else roots[frame]
                        if iteration > 0
                        else roots[frame - 1]
                    ),
                    )
                )
            except ContractError as error:
                raise ContractError(
                    f"step-up frame {frame} retarget failed: {error}"
                ) from error
            ankles[frame] = foot_kinematics.foot_positions(
                joints[frame : frame + 1],
                roots[frame : frame + 1],
                quaternions[frame : frame + 1],
            )[0]
            soles[frame] = sole_kinematics.sole_points(
                joints[frame : frame + 1],
                roots[frame : frame + 1],
                quaternions[frame : frame + 1],
            )[0]
            ankle_scene[frame] = (
                ankles[frame, :, :2] @ rotation_scene.T
                + translation
            )
            sole_scene[frame] = (
                soles[frame, :, :, :2] @ rotation_scene.T
                + translation
            )
            ankle_surface[frame] = _sample_numpy(
                target_grid, ankle_scene[frame]
            )
            sole_surface[frame] = _sample_numpy(
                target_grid, sole_scene[frame]
            )
            ankle_clearance[frame] = (
                ankles[frame, :, 2]
                - ankle_surface[frame]
                - float(ANKLE_ORIGIN_SOLE_M)
            )
            sole_clearance[frame] = (
                soles[frame, :, :, 2] - sole_surface[frame]
            )
            frame_retargeted = True
        if frame_retargeted:
            retargeted_frames += 1
    local_sequence = replace(
        sequence,
        start_frame=0,
        first_contact_frame=local_first,
        trailing_contact_frame=local_trailing,
        end_exclusive=len(roots),
    )
    validation = validate_placed_step_up(
        sequence=local_sequence,
        source_support_mask=local_support,
        ankle_clearance_m=ankle_clearance,
        sole_clearance_m=sole_clearance,
        target_final_height_delta_m=float(
            placement["target_final_height_delta_m"]
        ),
    )
    metrics = {
        "direction_sign": int(placement["direction_sign"]),
        "traversal_angle_deg": float(
            placement["traversal_angle_deg"]
        ),
        "landing_scene_xy": np.asarray(
            placement["landing_scene_xy"]
        ).tolist(),
        "first_contact_frame": int(local_first),
        "trailing_contact_frame": int(local_trailing),
        "frame_count": int(len(roots)),
        "source_first_rise_m": sequence.source_first_rise_m,
        "source_final_height_delta_m": (
            sequence.source_final_height_delta_m
        ),
        "target_final_height_delta_m": float(
            placement["target_final_height_delta_m"]
        ),
        "unsupported_frame_count": validation.unsupported_frame_count,
        "final_split_height_contact_transfer": (
            validation.final_split_height_contact_transfer
        ),
        "contact_height_pattern_error_m": (
            validation.contact_height_pattern_error_m
        ),
        "maximum_stance_contact_error_m": (
            validation.maximum_stance_contact_error_m
        ),
        "minimum_sole_clearance_m": (
            validation.minimum_sole_clearance_m
        ),
        "retargeted_swing_frame_count": retargeted_frames,
        "maximum_support_footprint_shift_m": float(
            support_footprint_shift.max()
        ),
        "final_target_surface_height_m": ankle_surface[
            local_trailing
        ].tolist(),
        "final_foot_scene_xy": ankle_scene[local_trailing].tolist(),
    }
    artifact = {
        "joint_position": joints,
        "root_position_world": roots,
        "root_orientation_world_wxyz": quaternions,
        "source_support_mask": local_support,
        "phase_boundaries": np.asarray(
            (0, local_first, local_trailing, len(roots)), dtype=np.int64
        ),
        "minimum_sole_clearance_by_frame": sole_clearance.min(
            axis=(1, 2)
        ),
        "planned_swing_lift_m": planned_swing_lift,
        "planned_swing_xy": planned_swing_xy,
    }
    return metrics, artifact


def main() -> int:
    args = _parser().parse_args()
    if (
        args.approach_frames < 3
        or not 0.08 <= args.minimum_rise_m < args.maximum_rise_m
        or not 0.0 < args.maximum_facing_error_deg <= 90.0
        or not 0.0 <= args.traversal_angle_deg <= 90.0
        or args.maximum_placements_per_event < 0
    ):
        raise ContractError("step-up search thresholds are invalid")
    manifest = json.loads(
        (args.source_dataset / "manifest.json").read_text("utf-8")
    )
    resolved = resolve_stair_config(
        args.target_dataset,
        load_experiment_config(args.config),
        device="cpu",
    )
    extension = resolved.measurement_extension
    target_grid = extension.query_grid
    sample_target = lambda points: _sample_numpy(target_grid, points)
    foot_kinematics = MujocoG1FootKinematics(args.g1_xml)
    sole_kinematics = MujocoG1SoleKinematics(args.g1_xml)
    terrain_retargeter = WideBoundG1TerrainRetargeter(
        args.g1_xml,
        maximum_root_height_deviation_m=1.0e-6,
        maximum_root_horizontal_deviation_m=1.0e-6,
        maximum_target_error_m=0.030,
    )
    accepted = []
    accepted_artifacts = []
    source_event_count = 0
    cheap_placement_count = 0
    for descriptor_index, descriptor in enumerate(manifest["clips"]):
        logical_name = str(descriptor.get("logical_name", ""))
        terrain = descriptor.get("terrain")
        if (
            not logical_name.startswith(("grail-stair", "grail-curb"))
            or (
                args.source_clip is not None
                and logical_name != args.source_clip
            )
            or not isinstance(terrain, dict)
            or "path" not in terrain
        ):
            continue
        arrays, surface, support = _source_arrays(
            args.source_dataset, descriptor
        )
        events = _first_contact_events(
            arrays=arrays,
            surface=surface,
            support=support,
            minimum_rise_m=args.minimum_rise_m,
            maximum_rise_m=args.maximum_rise_m,
            approach_frames=args.approach_frames,
            maximum_facing_error_rad=math.radians(
                args.maximum_facing_error_deg
            ),
        )
        source_event_count += len(events)
        for sequence in events:
            placements = _cheap_placements(
                arrays=arrays,
                surface=surface,
                sequence=sequence,
                alignment_yaw=float(
                    extension.alignment.yaw_scene_from_matcher
                ),
                traversal_angle_deg=args.traversal_angle_deg,
                sample_height=sample_target,
            )
            cheap_placement_count += len(placements)
            if args.maximum_placements_per_event:
                preferred = _preferred_landing_point(
                    args.traversal_angle_deg
                )
                placements.sort(
                    key=lambda placement: (
                        placement["direction_sign"] < 0,
                        float(
                            np.linalg.norm(
                                np.asarray(
                                    placement["landing_scene_xy"]
                                )
                                - preferred
                            )
                        ),
                    )
                )
                placements = placements[
                    : args.maximum_placements_per_event
                ]
            for placement in placements:
                try:
                    metrics, artifact = _placed_candidate(
                        arrays=arrays,
                        surface=surface,
                        support=support,
                        sequence=sequence,
                        placement=placement,
                        alignment=extension.alignment,
                        target_grid=target_grid,
                        foot_kinematics=foot_kinematics,
                        sole_kinematics=sole_kinematics,
                        terrain_retargeter=terrain_retargeter,
                    )
                except ContractError as error:
                    if args.maximum_placements_per_event == 1:
                        print(
                            f"rejected {logical_name} "
                            f"frames={sequence.start_frame}:"
                            f"{sequence.end_exclusive}: {error}",
                            flush=True,
                        )
                    continue
                metrics["source_clip"] = logical_name
                metrics["source_descriptor_index"] = descriptor_index
                metrics["score"] = _candidate_score(metrics)
                accepted.append(metrics)
                accepted_artifacts.append(artifact)
        if descriptor_index % 100 == 0:
            print(
                f"scanned={descriptor_index + 1} "
                f"events={source_event_count} "
                f"cheap={cheap_placement_count} "
                f"accepted={len(accepted)}",
                flush=True,
            )
    if not accepted:
        raise ContractError(
            "no native GRAIL sequence matches the target uneven contacts"
        )
    order = sorted(
        range(len(accepted)),
        key=lambda index: (
            accepted[index]["score"],
            accepted[index]["source_clip"],
        ),
    )
    ranked = [accepted[index] for index in order]
    best = ranked[0]
    artifact = accepted_artifacts[order[0]]
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "step-up.npz", **artifact)
    (args.output / "ranked.json").write_text(
        json.dumps(ranked, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": "g1-native-uneven-step-up/v1",
        "source_event_count": source_event_count,
        "cheap_placement_count": cheap_placement_count,
        "accepted_candidate_count": len(accepted),
        **best,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
