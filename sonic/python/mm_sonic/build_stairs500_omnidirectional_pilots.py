"""Build exact-audited directional variants of clean stairs500 motions.

The original stairs500 clips mostly traverse one fixed stair centreline.  This
builder deliberately changes that limitation without moving the terrain or
sliding a planted foot in time.  It maps source stair progress onto a smooth
stair-local path, maps each sole by its own source progress, and re-solves the
two legs against those explicit sole targets.  A complete-G1 collision audit
then rejects variants that intersect the exact source stair mesh.

This is a privileged kinematic data generator.  Root position is used only to
author and audit the clean motion; the saved causal interface remains ordinary
two-stick planar velocity plus independently specified facing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import zarr

from .compose_coherent_block_plan import _repair_exact_mesh_clearance
from .render_stitched_motion import load_stitched_motion_npz
from .terrain_oracle.math3d import quaternion_multiply_wxyz
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.stair_fragment_reconstruction import (
    _anchor_stance_sole_targets,
    _blend_anchored_sole_offsets,
    _clear_swing_sole_target,
    _interpolate_fixed_vectors,
    _replace_bracketed_swing_trajectories,
    _smooth_root_anchor_shifts,
    _smooth_swing_route_adjustments,
    _stance_guidance_weights,
    _stance_spans,
)
from .terrain_oracle.stair_foothold_anchors import FootholdAnchorConfig
from .terrain_oracle.stair_geometry_warp import (
    _archive_terrain_index,
    _level_boundaries,
    motion_conditioned_stair_support_route,
)
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.stitch import StitchedMotion
from .terrain_oracle.contact import CanonicalMeshQuery


DEFAULT_ARCHIVE = Path(
    "/move/data/terrain-aware/motion-matching/"
    "grail-stairs500-clean-50hz-v1.zarr"
)
DEFAULT_SELECTION = Path(
    "/move/data/terrain-aware/sonic-rollouts/"
    "terrain_maneuver_stairs500_v1/selected200_v1.json"
)
DEFAULT_SOURCE_BANK_ROOT = Path(
    "/move/data/terrain-aware/sonic-rollouts/"
    "terrain_maneuver_stairs500_v1/full_bank_v1"
)
DEFAULT_MODEL = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/"
    "g1_29dof_rev_1_0.xml"
)
DEFAULT_OUTPUT = Path(
    "/move/data/terrain-aware/sonic-rollouts/"
    "terrain_maneuver_stairs500_v1/omnidirectional_pilot_v1"
)


@dataclass(frozen=True)
class PathProfile:
    """Per-sample stair-local path and facing offsets."""

    lateral_offset_m: np.ndarray
    path_yaw_offset_rad: np.ndarray
    pose_yaw_offset_rad: np.ndarray
    foot_yaw_offset_rad: np.ndarray
    facing_yaw_offset_rad: np.ndarray
    normalized_progress: np.ndarray
    amplitude_m: float
    maximum_path_angle_deg: float


@dataclass(frozen=True)
class WarpDiagnostics:
    maximum_joint_correction_rad: float
    maximum_sole_target_error_m: float
    maximum_stance_sole_target_error_m: float
    maximum_swing_sole_target_error_m: float
    maximum_stance_run_drift_m: float
    minimum_stance_support_point_count: int
    maximum_root_translation_step_m: float
    maximum_root_rotation_step_rad: float
    maximum_joint_step_rad: float
    maximum_root_acceleration_m_s2: float
    maximum_root_anchor_shift_m: float
    maximum_lateral_offset_m: float
    lateral_offset_range_m: float
    realized_root_progress_range_m: float
    realized_root_lateral_range_m: float
    realized_root_vertical_range_m: float
    maximum_path_angle_deg: float
    maximum_facing_offset_deg: float


def _quintic(value: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return x**3 * (10.0 + x * (-15.0 + 6.0 * x))


def _quintic_derivative(value: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return 30.0 * x**2 * (1.0 - x) ** 2


def _piecewise_quintic(
    progress: np.ndarray,
    knot_values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate equal-width knots with zero derivative at every knot."""

    u = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    values = np.asarray(tuple(knot_values), dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("path profile needs at least two knots")
    segments = len(values) - 1
    coordinate = np.minimum(u * segments, np.nextafter(float(segments), 0.0))
    index = np.floor(coordinate).astype(np.int64)
    local = coordinate - index
    delta = values[index + 1] - values[index]
    output = values[index] + delta * _quintic(local)
    derivative = delta * _quintic_derivative(local) * segments
    output[u >= 1.0] = values[-1]
    derivative[(u <= 0.0) | (u >= 1.0)] = 0.0
    return output, derivative


def _profile_knots(
    mode: str,
) -> tuple[tuple[float, ...], float, float, float, float]:
    """Return knots, path angle, facing, pelvis gain, and foot-yaw gain."""

    profiles = {
        # ``straight`` is useful primarily for the exact temporal reverse of a
        # clean traversal: it gives a full-stair backward locomotion example
        # without altering any authored contact or pose.
        "straight": ((0.0, 0.0), 0.0, 0.0, 0.0, 0.0),
        # The first eight modes keep authored uphill/downhill foot yaw while
        # changing travel.  That is genuine two-stick diagonal/lateral motion
        # and avoids rotating a long G1 foot into the adjacent riser.
        "diagonal_soft_left": ((-1.0, 1.0), 18.0, 0.0, 1.0, 0.0),
        "diagonal_soft_right": ((1.0, -1.0), 18.0, 0.0, 1.0, 0.0),
        "diagonal_gentle_left": ((-1.0, 1.0), 10.0, 0.0, 1.0, 0.0),
        "diagonal_gentle_right": ((1.0, -1.0), 10.0, 0.0, 1.0, 0.0),
        "diagonal_hard_left": ((-1.0, 1.0), 32.0, 0.0, 1.0, 0.0),
        "diagonal_hard_right": ((1.0, -1.0), 32.0, 0.0, 1.0, 0.0),
        "diagonal_medium_left": ((-1.0, 1.0), 24.0, 0.0, 1.0, 0.0),
        "diagonal_medium_right": ((1.0, -1.0), 24.0, 0.0, 1.0, 0.0),
        # Pure two-stick oblique travel: successive footholds and the root move
        # across the staircase while body facing stays on the authored stair
        # axis.  This avoids conflating a velocity command with an equal yaw
        # command and supplies the missing "move diagonally while looking
        # ahead" examples.
        "travel_soft_left": ((-1.0, 1.0), 18.0, 0.0, 0.0, 0.0),
        "travel_soft_right": ((1.0, -1.0), 18.0, 0.0, 0.0, 0.0),
        # Ten-degree fixed-facing travel is still visibly oblique, but fits
        # many narrower stair clips that cannot mechanically realize the
        # 18--32 degree families.  This is the dense coverage tier rather
        # than a replacement for the harder stress cases below.
        "travel_gentle_left": ((-1.0, 1.0), 10.0, 0.0, 0.0, 0.0),
        "travel_gentle_right": ((1.0, -1.0), 10.0, 0.0, 0.0, 0.0),
        "travel_micro_left": ((-1.0, 1.0), 5.0, 0.0, 0.0, 0.0),
        "travel_micro_right": ((1.0, -1.0), 5.0, 0.0, 0.0, 0.0),
        "travel_medium_left": ((-1.0, 1.0), 24.0, 0.0, 0.0, 0.0),
        "travel_medium_right": ((1.0, -1.0), 24.0, 0.0, 0.0, 0.0),
        "travel_hard_left": ((-1.0, 1.0), 32.0, 0.0, 0.0, 0.0),
        "travel_hard_right": ((1.0, -1.0), 32.0, 0.0, 0.0, 0.0),
        # Fixed-terrain stress cases.  Unlike the paired co-warps below, the
        # staircase is not bent under the character: these really traverse
        # the source treads at up to fifty degrees while the pelvis keeps its
        # authored stair-forward heading.
        "travel_extreme_left": ((-1.0, 1.0), 50.0, 0.0, 0.0, 0.0),
        "travel_extreme_right": ((1.0, -1.0), 50.0, 0.0, 0.0, 0.0),
        # The matching body-aligned stress cases turn the pelvis with travel
        # while retaining the authored foot yaw.  Exact IK/contact/collision
        # gates decide whether an individual staircase is wide enough.
        "diagonal_extreme_left": ((-1.0, 1.0), 50.0, 0.0, 1.0, 0.0),
        "diagonal_extreme_right": ((1.0, -1.0), 50.0, 0.0, 1.0, 0.0),
        # A fifty-degree *approach* is less pathological than traversing the
        # complete staircase at fifty degrees.  These enter a neighbouring
        # lane during the first half of the terrain passage, then continue
        # straight for the remaining steps.  The unchanged mesh still makes
        # this a real oblique stair entry rather than a bent-stair co-warp.
        "approach_extreme_left": ((0.0, 1.0, 1.0), 50.0, 0.0, 0.0, 0.0),
        "approach_extreme_right": ((0.0, -1.0, -1.0), 50.0, 0.0, 0.0, 0.0),
        "approach_turn_extreme_left": (
            (0.0, 1.0, 1.0),
            50.0,
            0.0,
            1.0,
            0.0,
        ),
        "approach_turn_extreme_right": (
            (0.0, -1.0, -1.0),
            50.0,
            0.0,
            1.0,
            0.0,
        ),
        "lane_left": ((0.0, 1.0), 20.0, 0.0, 1.0, 0.0),
        "lane_right": ((0.0, -1.0), 20.0, 0.0, 1.0, 0.0),
        "lane_gentle_left": ((0.0, 1.0), 10.0, 0.0, 1.0, 0.0),
        "lane_gentle_right": ((0.0, -1.0), 10.0, 0.0, 1.0, 0.0),
        "zigzag_left_right": ((0.0, 1.0, -1.0, 0.0), 28.0, 0.0, 1.0, 0.0),
        "zigzag_right_left": ((0.0, -1.0, 1.0, 0.0), 28.0, 0.0, 1.0, 0.0),
        "zigzag_gentle_left_right": (
            (0.0, 1.0, -1.0, 0.0),
            14.0,
            0.0,
            1.0,
            0.0,
        ),
        "zigzag_gentle_right_left": (
            (0.0, -1.0, 1.0, 0.0),
            14.0,
            0.0,
            1.0,
            0.0,
        ),
        "zigzag_micro_left_right": (
            (0.0, 1.0, -1.0, 0.0),
            6.0,
            0.0,
            0.0,
            0.0,
        ),
        "zigzag_micro_right_left": (
            (0.0, -1.0, 1.0, 0.0),
            6.0,
            0.0,
            0.0,
            0.0,
        ),
        "crab_left": ((-1.0, 1.0), 20.0, -24.0, 0.0, 0.0),
        "crab_right": ((1.0, -1.0), 20.0, 24.0, 0.0, 0.0),
        "crab_gentle_left": ((-1.0, 1.0), 12.0, -12.0, 0.0, 0.0),
        "crab_gentle_right": ((1.0, -1.0), 12.0, 12.0, 0.0, 0.0),
        "face_left": ((0.0, 0.0), 0.0, 28.0, 0.0, 0.0),
        "face_right": ((0.0, 0.0), 0.0, -28.0, 0.0, 0.0),
        "face_soft_left": ((0.0, 0.0), 0.0, 12.0, 0.0, 0.0),
        "face_soft_right": ((0.0, 0.0), 0.0, -12.0, 0.0, 0.0),
        "face_micro_left": ((0.0, 0.0), 0.0, 8.0, 0.0, 0.0),
        "face_micro_right": ((0.0, 0.0), 0.0, -8.0, 0.0, 0.0),
        "face_hard_left": ((0.0, 0.0), 0.0, 45.0, 0.0, 0.0),
        "face_hard_right": ((0.0, 0.0), 0.0, -45.0, 0.0, 0.0),
        # Near-side-on stair climbing on the unchanged source staircase.  The
        # path remains uphill while the pelvis turns seventy-five degrees in
        # the middle of the stair passage and eases back at the landings.
        "face_side_on_left": ((0.0, 0.0), 0.0, 75.0, 0.0, 0.0),
        "face_side_on_right": ((0.0, 0.0), 0.0, -75.0, 0.0, 0.0),
        # These profiles exercise a genuinely independent second stick.  The
        # path bends one way while the pelvis faces the other; the feet retain
        # their authored stair yaw so variants remain mechanically auditable.
        "counterface_left": ((-1.0, 1.0), 10.0, -10.0, 0.0, 0.0),
        "counterface_right": ((1.0, -1.0), 10.0, 10.0, 0.0, 0.0),
        "counterface_micro_left": ((-1.0, 1.0), 8.0, -8.0, 0.0, 0.0),
        "counterface_micro_right": ((1.0, -1.0), 8.0, 8.0, 0.0, 0.0),
        # A facing weave changes body direction twice while the stair path is
        # unchanged.  Its non-sinusoidal facing curve is supplied below.
        "facing_weave_left_right": ((0.0, 0.0), 0.0, 10.0, 0.0, 0.0),
        "facing_weave_right_left": ((0.0, 0.0), 0.0, 10.0, 0.0, 0.0),
        "facing_weave_micro_left_right": (
            (0.0, 0.0),
            0.0,
            8.0,
            0.0,
            0.0,
        ),
        "facing_weave_micro_right_left": (
            (0.0, 0.0),
            0.0,
            8.0,
            0.0,
            0.0,
        ),
        # A longer weave changes travel direction twice while the feet remain
        # world-locked during every detected stance run.
        "slalom_left_right": (
            (0.0, 1.0, -1.0, 1.0, 0.0),
            20.0,
            0.0,
            1.0,
            0.0,
        ),
        "slalom_right_left": (
            (0.0, -1.0, 1.0, -1.0, 0.0),
            20.0,
            0.0,
            1.0,
            0.0,
        ),
        # Smaller simultaneous path/body turns are retained as a distinct
        # family rather than contaminating every diagonal with riser-facing
        # foot yaw.
        "turning_left": ((-1.0, 1.0), 18.0, 0.0, 0.5, 0.25),
        "turning_right": ((1.0, -1.0), 18.0, 0.0, 0.5, 0.25),
        "turning_gentle_left": ((-1.0, 1.0), 10.0, 0.0, 0.5, 0.25),
        "turning_gentle_right": ((1.0, -1.0), 10.0, 0.0, 0.5, 0.25),
        "turning_micro_left": ((-1.0, 1.0), 8.0, 0.0, 0.5, 0.25),
        "turning_micro_right": ((1.0, -1.0), 8.0, 0.0, 0.5, 0.25),
        "turning_ultramicro_left": (
            (-1.0, 1.0),
            4.0,
            0.0,
            0.35,
            0.10,
        ),
        "turning_ultramicro_right": (
            (1.0, -1.0),
            4.0,
            0.0,
            0.35,
            0.10,
        ),
        # Unlike the path-only slaloms, these rotate the pelvis and a fraction
        # of foot yaw with every direction change.  This is the closest clean
        # kinematic analogue of steering repeatedly while already on stairs.
        "turning_slalom_left_right": (
            (0.0, 1.0, -1.0, 1.0, 0.0),
            12.0,
            0.0,
            0.35,
            0.15,
        ),
        "turning_slalom_right_left": (
            (0.0, -1.0, 1.0, -1.0, 0.0),
            12.0,
            0.0,
            0.35,
            0.15,
        ),
        "turning_zigzag_left_right": (
            (0.0, 1.0, -1.0, 0.0),
            12.0,
            0.0,
            0.40,
            0.15,
        ),
        "turning_zigzag_right_left": (
            (0.0, -1.0, 1.0, 0.0),
            12.0,
            0.0,
            0.40,
            0.15,
        ),
        # Co-warp profiles rotate the pelvis and the complete foot frame with
        # the path tangent.  They are intended for paired motion+terrain
        # deformation: the registered mesh is bent by the identical map, so
        # these can author much stronger rough-terrain steering without
        # asking a planted foot to conform to an unrelated fixed surface.
        "cowarp_diagonal_left": ((-1.0, 1.0), 28.0, 0.0, 1.0, 1.0),
        "cowarp_diagonal_right": ((1.0, -1.0), 28.0, 0.0, 1.0, 1.0),
        "cowarp_lane_left": ((0.0, 1.0), 24.0, 0.0, 1.0, 1.0),
        "cowarp_lane_right": ((0.0, -1.0), 24.0, 0.0, 1.0, 1.0),
        # A visibly wider paired-terrain tier.  Held-lane and step-change
        # profiles dwell at their lateral extrema instead of immediately
        # smoothing back through the centreline, which makes the command
        # change useful to a long-horizon policy rather than a tiny wiggle.
        "cowarp_diagonal_hard_left": ((-1.0, 1.0), 38.0, 0.0, 1.0, 1.0),
        "cowarp_diagonal_hard_right": ((1.0, -1.0), 38.0, 0.0, 1.0, 1.0),
        # Hard crab diagonals use the second stick to oppose the travel
        # tangent.  At peak obliquity the pelvis remains approximately
        # stair-forward while the paired motion and terrain travel at 24 deg.
        "cowarp_crab_hard_left": ((-1.0, 1.0), 24.0, -24.0, 1.0, 1.0),
        "cowarp_crab_hard_right": ((1.0, -1.0), 24.0, 24.0, 1.0, 1.0),
        "cowarp_lane_hold_left": ((0.0, 1.0, 1.0, 0.0), 28.0, 0.0, 1.0, 1.0),
        "cowarp_lane_hold_right": (
            (0.0, -1.0, -1.0, 0.0),
            28.0,
            0.0,
            1.0,
            1.0,
        ),
        "cowarp_step_change_left_right": (
            (0.0, 1.0, 1.0, -1.0, -1.0, 0.0),
            26.0,
            0.0,
            1.0,
            1.0,
        ),
        "cowarp_step_change_right_left": (
            (0.0, -1.0, -1.0, 1.0, 1.0, 0.0),
            26.0,
            0.0,
            1.0,
            1.0,
        ),
        "cowarp_slalom_left_right": (
            (0.0, 1.0, -1.0, 1.0, 0.0),
            18.0,
            0.0,
            1.0,
            1.0,
        ),
        "cowarp_slalom_right_left": (
            (0.0, -1.0, 1.0, -1.0, 0.0),
            18.0,
            0.0,
            1.0,
            1.0,
        ),
        "cowarp_slalom_gentle_left_right": (
            (0.0, 1.0, -1.0, 1.0, 0.0),
            12.0,
            0.0,
            1.0,
            1.0,
        ),
        "cowarp_slalom_gentle_right_left": (
            (0.0, -1.0, 1.0, -1.0, 0.0),
            12.0,
            0.0,
            1.0,
            1.0,
        ),
        "cowarp_zigzag_gentle_left_right": (
            (0.0, 1.0, -1.0, 0.0),
            12.0,
            0.0,
            1.0,
            1.0,
        ),
        "cowarp_zigzag_gentle_right_left": (
            (0.0, -1.0, 1.0, 0.0),
            12.0,
            0.0,
            1.0,
            1.0,
        ),
        # Paired terrain deformation supplies the path contact; these modes
        # independently offset body facing so all four travel/facing sign
        # combinations appear in the clean terrain corpus.
        "cowarp_diagonal_left_face_left": (
            (-1.0, 1.0),
            14.0,
            10.0,
            1.0,
            1.0,
        ),
        "cowarp_diagonal_left_face_right": (
            (-1.0, 1.0),
            14.0,
            -10.0,
            1.0,
            1.0,
        ),
        "cowarp_diagonal_right_face_left": (
            (1.0, -1.0),
            14.0,
            10.0,
            1.0,
            1.0,
        ),
        "cowarp_diagonal_right_face_right": (
            (1.0, -1.0),
            14.0,
            -10.0,
            1.0,
            1.0,
        ),
        "cowarp_facing_weave_left_right": (
            (0.0, 0.0),
            0.0,
            10.0,
            0.0,
            0.0,
        ),
        "cowarp_facing_weave_right_left": (
            (0.0, 0.0),
            0.0,
            10.0,
            0.0,
            0.0,
        ),
        "cowarp_facing_weave_hard_left_right": (
            (0.0, 0.0),
            0.0,
            20.0,
            0.0,
            0.0,
        ),
        "cowarp_facing_weave_hard_right_left": (
            (0.0, 0.0),
            0.0,
            20.0,
            0.0,
            0.0,
        ),
        # Both sticks change while already on the terrain: the route slaloms
        # while body facing traverses the opposite sequence.  Foot frames
        # still follow the paired terrain tangent, preserving stance contact.
        "cowarp_slalom_counterface_left_right": (
            (0.0, 1.0, -1.0, 1.0, 0.0),
            16.0,
            12.0,
            1.0,
            1.0,
        ),
        "cowarp_slalom_counterface_right_left": (
            (0.0, -1.0, 1.0, -1.0, 0.0),
            16.0,
            12.0,
            1.0,
            1.0,
        ),
    }
    try:
        return profiles[mode]
    except KeyError as error:
        raise ValueError(f"unknown directional stair mode: {mode}") from error


def build_path_profile(
    mode: str,
    progress: object,
    *,
    active_length_m: float,
    maximum_amplitude_m: float,
) -> PathProfile:
    """Build a bounded profile whose advertised angle is its actual maximum."""

    u = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    (
        knots,
        requested_angle_deg,
        facing_offset_deg,
        pose_yaw_gain,
        foot_yaw_gain,
    ) = _profile_knots(mode)
    unit_offset, unit_derivative = _piecewise_quintic(u, knots)
    length = float(active_length_m)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("active path length must be positive")
    maximum_unit_derivative = float(np.max(np.abs(unit_derivative)))
    if maximum_unit_derivative <= 1.0e-9 or requested_angle_deg == 0.0:
        amplitude = 0.0
    else:
        angle_limited = (
            math.tan(math.radians(requested_angle_deg))
            * length
            / maximum_unit_derivative
        )
        amplitude = min(float(maximum_amplitude_m), angle_limited)
    lateral = amplitude * unit_offset
    derivative_m_per_m = amplitude * unit_derivative / length
    path_yaw = np.arctan(derivative_m_per_m)
    pose_yaw = float(pose_yaw_gain) * path_yaw
    foot_yaw = float(foot_yaw_gain) * path_yaw
    # Independent facing is introduced and removed with zero endpoint speed.
    # Most profiles use one smooth excursion.  Facing-weave profiles use
    # explicit knots so they contain both left and right body orientations
    # without changing the travel path.
    facing_knots = {
        "facing_weave_left_right": (0.0, 1.0, -1.0, 0.0),
        "facing_weave_right_left": (0.0, -1.0, 1.0, 0.0),
        "facing_weave_micro_left_right": (0.0, 1.0, -1.0, 0.0),
        "facing_weave_micro_right_left": (0.0, -1.0, 1.0, 0.0),
        "cowarp_facing_weave_left_right": (0.0, 1.0, -1.0, 0.0),
        "cowarp_facing_weave_right_left": (0.0, -1.0, 1.0, 0.0),
        "cowarp_facing_weave_hard_left_right": (0.0, 1.0, -1.0, 0.0),
        "cowarp_facing_weave_hard_right_left": (0.0, -1.0, 1.0, 0.0),
        "cowarp_slalom_counterface_left_right": (0.0, -1.0, 1.0, 0.0),
        "cowarp_slalom_counterface_right_left": (0.0, 1.0, -1.0, 0.0),
    }.get(mode)
    if facing_knots is None:
        facing_window = np.sin(np.pi * u) ** 2
    else:
        facing_window, _facing_derivative = _piecewise_quintic(
            u, facing_knots
        )
    facing = math.radians(facing_offset_deg) * facing_window
    return PathProfile(
        lateral_offset_m=lateral,
        path_yaw_offset_rad=path_yaw,
        pose_yaw_offset_rad=pose_yaw,
        foot_yaw_offset_rad=foot_yaw,
        facing_yaw_offset_rad=facing,
        normalized_progress=u,
        amplitude_m=float(amplitude),
        maximum_path_angle_deg=float(np.degrees(np.max(np.abs(path_yaw)))),
    )


def _yaw_quaternion(value: float) -> np.ndarray:
    half = 0.5 * float(value)
    return np.asarray((math.cos(half), 0.0, 0.0, math.sin(half)))


def _rotate_z(points: np.ndarray, yaw: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    result = values.copy()
    result[..., 0] = cosine * values[..., 0] - sine * values[..., 1]
    result[..., 1] = sine * values[..., 0] + cosine * values[..., 1]
    return result


def _path_coordinates(
    points_world: np.ndarray,
    *,
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lateral_axis = np.asarray((-direction_xy[1], direction_xy[0]))
    relative = np.asarray(points_world, dtype=np.float64)[..., :2] - origin_xy
    return relative @ direction_xy, relative @ lateral_axis


def _map_points(
    points_world: np.ndarray,
    *,
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
    active_start_m: float,
    active_stop_m: float,
    mode: str,
    maximum_amplitude_m: float,
) -> tuple[np.ndarray, PathProfile]:
    points = np.asarray(points_world, dtype=np.float64)
    progress_axis, lateral = _path_coordinates(
        points, origin_xy=origin_xy, direction_xy=direction_xy
    )
    active_length = float(active_stop_m - active_start_m)
    normalized = (progress_axis - active_start_m) / active_length
    profile = build_path_profile(
        mode,
        normalized,
        active_length_m=active_length,
        maximum_amplitude_m=maximum_amplitude_m,
    )
    lateral_axis = np.asarray((-direction_xy[1], direction_xy[0]))
    tangent = np.stack(
        (
            np.cos(profile.path_yaw_offset_rad) * direction_xy[0]
            - np.sin(profile.path_yaw_offset_rad) * direction_xy[1],
            np.sin(profile.path_yaw_offset_rad) * direction_xy[0]
            + np.cos(profile.path_yaw_offset_rad) * direction_xy[1],
        ),
        axis=-1,
    )
    normal = np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
    centreline = (
        origin_xy
        + progress_axis[..., None] * direction_xy
        + profile.lateral_offset_m[..., None] * lateral_axis
    )
    output = points.copy()
    output[..., :2] = centreline + lateral[..., None] * normal
    return output, profile


def _map_rigid_sole(
    sole_world: np.ndarray,
    **path_arguments: object,
) -> tuple[np.ndarray, PathProfile]:
    sole = np.asarray(sole_world, dtype=np.float64)
    centre = np.mean(sole, axis=0)
    mapped_centre, profile = _map_points(centre[None], **path_arguments)
    rotated = _rotate_z(sole - centre, float(profile.path_yaw_offset_rad[0]))
    return mapped_centre[0] + rotated, profile


def _terrain_height(target_mesh: object, xy: np.ndarray, ray_z: float) -> float:
    hit = target_mesh.raycast(
        np.asarray((xy[0], xy[1], ray_z), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )
    return -math.inf if hit is None else float(hit.position_world[2])


def _corresponding_warped_surface_point(
    source_mesh: object,
    target_mesh: object,
    point_world: np.ndarray,
    *,
    ray_z: float,
) -> np.ndarray | None:
    """Map a source surface point through paired meshes by face barycentrics."""

    point = np.asarray(point_world, dtype=np.float64)
    hit = source_mesh.raycast(
        np.asarray((point[0], point[1], ray_z), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )
    if hit is None or abs(float(point[2]) - float(hit.position_world[2])) > 0.020:
        return None
    face_index = int(hit.face_index)
    if not bool(target_mesh.mesh.valid_faces[face_index]):
        return None
    vertex_indices = np.asarray(source_mesh.mesh.faces[face_index], dtype=np.int64)
    source_triangle = np.asarray(
        source_mesh.vertices_world[vertex_indices], dtype=np.float64
    )
    target_triangle = np.asarray(
        target_mesh.vertices_world[vertex_indices], dtype=np.float64
    )
    edge0 = source_triangle[1] - source_triangle[0]
    edge1 = source_triangle[2] - source_triangle[0]
    relative = np.asarray(hit.position_world, dtype=np.float64) - source_triangle[0]
    d00 = float(np.dot(edge0, edge0))
    d01 = float(np.dot(edge0, edge1))
    d11 = float(np.dot(edge1, edge1))
    d20 = float(np.dot(relative, edge0))
    d21 = float(np.dot(relative, edge1))
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) <= 1.0e-12:
        return None
    weight1 = (d11 * d20 - d01 * d21) / denominator
    weight2 = (d00 * d21 - d01 * d20) / denominator
    weights = np.asarray((1.0 - weight1 - weight2, weight1, weight2))
    return weights @ target_triangle


def _anchor_stance_soles_to_continuous_mesh(
    nominal_targets: np.ndarray,
    stance: np.ndarray,
    sphere_radii: Sequence[np.ndarray],
    *,
    target_mesh: object,
    ray_z: float,
    source_nominal_targets: np.ndarray | None = None,
    source_mesh: object | None = None,
    source_ray_z: float | None = None,
) -> np.ndarray:
    """Lock mapped stance feet directly to an arbitrary continuous surface.

    Stair routes provide discrete support levels.  Hills, ramps, and rough
    ground do not.  For those surfaces each mapped sole probe is raycast at
    the middle of its authored stance run, then the resulting rigid target is
    held for that complete run.  Swing offsets are interpolated between the
    exact stance anchors, after which the regular G1 IK and complete-mesh gate
    remain authoritative.
    """

    nominal = np.asarray(nominal_targets, dtype=np.float64)
    stance_mask = np.asarray(stance, dtype=bool)
    anchored = nominal.copy()
    assigned = np.zeros(stance_mask.shape, dtype=bool)
    for span in _stance_spans(stance_mask):
        frame = (span.start_frame + span.stop_frame - 1) // 2
        foot = int(span.foot_index)
        target = np.asarray(nominal[frame, foot], dtype=np.float64).copy()
        radii = np.asarray(sphere_radii[foot], dtype=np.float64)
        if source_mesh is not None:
            if source_nominal_targets is None or source_ray_z is None:
                raise ValueError("paired mesh anchoring needs source sole targets")
            source_target = np.asarray(
                source_nominal_targets[frame, foot], dtype=np.float64
            )
            residuals: list[np.ndarray] = []
            for probe, source_point in enumerate(source_target):
                source_surface = source_point.copy()
                source_surface[2] -= radii[probe]
                mapped_surface = _corresponding_warped_surface_point(
                    source_mesh,
                    target_mesh,
                    source_surface,
                    ray_z=float(source_ray_z),
                )
                if mapped_surface is None:
                    continue
                required_centre = mapped_surface.copy()
                required_centre[2] += radii[probe]
                residuals.append(required_centre - target[probe])
            if len(residuals) < 2:
                raise ValueError(
                    "mapped stance sole has fewer than two paired terrain probes"
                )
            residual = np.asarray(residuals, dtype=np.float64)
            # The path map is spatially varying, while a real sole is rigid.
            # Retain its mapped yaw and choose one whole-foot translation:
            # least-squares in XY and the smallest clearance-preserving lift.
            target[:, :2] += np.mean(residual[:, :2], axis=0)
            vertical_shift = float(np.max(residual[:, 2]))
        else:
            heights = np.asarray(
                [
                    _terrain_height(target_mesh, point[:2], ray_z)
                    for point in target
                ],
                dtype=np.float64,
            )
            supported = np.isfinite(heights)
            if int(np.count_nonzero(supported)) < 2:
                raise ValueError(
                    "mapped stance sole has fewer than two terrain support probes"
                )
            # Narrow stair treads and rough crests legitimately leave a toe or
            # heel probe outside the finite terrain patch.  Stance detection
            # and final admission both require at least two supported probes.
            required_centres = heights[supported] + radii[supported]
            vertical_shift = float(
                np.max(required_centres - target[supported, 2])
            )
        # A G1 sole is rigid.  Assigning every probe the terrain height can
        # produce four non-coplanar targets on rough ground and asks the leg
        # IK to realize an impossible foot shape.  Preserve the authored sole
        # orientation and apply the smallest whole-foot vertical shift that
        # clears every probe; exact support and hover checks below decide
        # whether the resulting rigid contact is actually usable.
        target[:, 2] += vertical_shift
        anchored[span.start_frame : span.stop_frame, foot] = target
        assigned[span.start_frame : span.stop_frame, foot] = True
    if not np.any(assigned):
        raise ValueError("source motion has no supported stance run")
    blended = _blend_anchored_sole_offsets(nominal, anchored, assigned)
    guidance = _stance_guidance_weights(
        stance_mask,
        blend_frames=12,
        release_frames=8,
    )
    return nominal + guidance[:, :, None, None] * (blended - nominal)


def _stance_runs_maximum_drift(
    centres: np.ndarray, stance: np.ndarray
) -> float:
    maximum = 0.0
    for foot in range(2):
        start = 0
        while start < len(stance):
            if not stance[start, foot]:
                start += 1
                continue
            stop = start + 1
            while stop < len(stance) and stance[stop, foot]:
                stop += 1
            if stop - start >= 2:
                values = centres[start:stop, foot, :2]
                maximum = max(
                    maximum,
                    float(np.max(np.linalg.norm(values - values[0], axis=1))),
                )
            start = stop
    return maximum


def _remove_short_stance_runs(
    stance: np.ndarray, *, minimum_run_frames: int
) -> np.ndarray:
    """Remove isolated slow/contact samples that are not planted phases."""

    result = np.asarray(stance, dtype=bool).copy()
    minimum = int(minimum_run_frames)
    if minimum < 1:
        raise ValueError("minimum stance run must be positive")
    if minimum == 1:
        return result
    for foot in range(2):
        start = 0
        while start < len(result):
            if not result[start, foot]:
                start += 1
                continue
            stop = start + 1
            while stop < len(result) and result[stop, foot]:
                stop += 1
            if stop - start < minimum:
                result[start:stop, foot] = False
            start = stop
    return result


def _fill_short_stance_gaps(
    stance: np.ndarray, *, maximum_gap_frames: int
) -> np.ndarray:
    """Close brief contact-classification dropouts inside one planted phase."""

    result = np.asarray(stance, dtype=bool).copy()
    maximum = int(maximum_gap_frames)
    if maximum < 0:
        raise ValueError("maximum stance gap must be nonnegative")
    if maximum == 0:
        return result
    for foot in range(2):
        start = 0
        while start < len(result):
            if result[start, foot]:
                start += 1
                continue
            stop = start + 1
            while stop < len(result) and not result[stop, foot]:
                stop += 1
            if (
                0 < start
                and stop < len(result)
                and stop - start <= maximum
            ):
                result[start:stop, foot] = True
            start = stop
    return result


def _terrain_clear_bracketed_swings(
    nominal_targets: np.ndarray,
    anchored_targets: np.ndarray,
    stance: np.ndarray,
    sphere_radii: Sequence[np.ndarray],
    *,
    target_mesh: object,
    target_route: object,
    minimum_clearance_m: float,
    adjustment_taper_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lift a flat gait's complete swing phases over exact stair risers."""

    targets, bracketed = _replace_bracketed_swing_trajectories(
        nominal_targets,
        anchored_targets,
        stance,
        minimum_clearance_m=float(minimum_clearance_m),
    )
    adjustments = np.zeros((len(targets), 2, 3), dtype=np.float64)
    mesh_query = CanonicalMeshQuery(
        target_mesh.mesh, target_mesh.world_from_terrain
    )
    for frame in range(len(targets)):
        for foot in range(2):
            if stance[frame, foot]:
                continue
            targets[frame, foot], adjustments[frame, foot] = (
                _clear_swing_sole_target(
                    targets[frame, foot],
                    sphere_radii[foot],
                    mesh=target_mesh,
                    route=target_route,
                    mesh_query=mesh_query,
                )
            )
    smoothed = _smooth_swing_route_adjustments(adjustments, stance)
    taper = int(adjustment_taper_frames)
    if taper < 0:
        raise ValueError("swing adjustment taper must be nonnegative")
    if taper:
        for foot in range(2):
            start_frame = 0
            while start_frame < len(stance):
                if stance[start_frame, foot]:
                    start_frame += 1
                    continue
                stop_frame = start_frame + 1
                while (
                    stop_frame < len(stance)
                    and not stance[stop_frame, foot]
                ):
                    stop_frame += 1
                for frame in range(start_frame, stop_frame):
                    distance = min(
                        frame - start_frame + 1,
                        stop_frame - frame,
                    )
                    linear = min(1.0, distance / float(taper + 1))
                    smooth = linear * linear * (3.0 - 2.0 * linear)
                    smoothed[frame, foot] *= smooth
                start_frame = stop_frame
    direction = np.asarray(target_route.end_xy, dtype=np.float64) - np.asarray(
        target_route.start_xy, dtype=np.float64
    )
    direction /= np.linalg.norm(direction)
    for frame in range(len(targets)):
        for foot in range(2):
            difference = smoothed[frame, foot] - adjustments[frame, foot]
            targets[frame, foot] += np.asarray(
                (
                    difference[0] * direction[0],
                    difference[0] * direction[1],
                    difference[2],
                ),
                dtype=np.float64,
            )[None]
    return targets, bracketed, smoothed


def warp_motion(
    source: StitchedMotion,
    *,
    adapter: _G1FootfallAdapter,
    target_mesh: object,
    target_route: object | None,
    direction_xy: np.ndarray,
    active_start_m: float,
    active_stop_m: float,
    mode: str,
    maximum_amplitude_m: float,
    source_mesh: object | None = None,
    spatial_sole_mapping: bool = False,
    minimum_stance_run_frames: int = 1,
    maximum_stance_gap_frames: int = 0,
    terrain_clear_swings: bool = False,
    minimum_swing_clearance_m: float = 0.08,
    swing_adjustment_taper_frames: int = 0,
    foothold_anchor_config: FootholdAnchorConfig | None = None,
    root_height_offset_m: np.ndarray | None = None,
    root_progress_offset_m: np.ndarray | None = None,
    root_anchor_smoothing_frames: float | None = None,
    maximum_foothold_level_step: int | None = None,
    ik_multistart: bool = False,
    target_level_by_stance_span: Mapping[tuple[int, int, int], int]
    | None = None,
    maximum_ik_continuity_step_rad: float | None = None,
    root_anchor_vertical: bool = True,
    root_anchor_vertical_limit_m: float | None = None,
    ik_continuity_acceptable_error_m: float = 0.008,
    ground_fallback_height_m: float | None = None,
    post_selection_maximum_joint_step_rad: float | None = None,
) -> tuple[StitchedMotion, WarpDiagnostics, dict[str, np.ndarray]]:
    """Warp one source motion and fit its legs to non-sliding sole targets."""

    roots = np.asarray(source.root_position_world, dtype=np.float64)
    quaternions = np.asarray(
        source.root_quaternion_world_wxyz, dtype=np.float64
    )
    authored_joints = np.asarray(source.joint_position, dtype=np.float64)
    origin_xy = roots[0, :2].copy()
    path_arguments = {
        "origin_xy": origin_xy,
        "direction_xy": direction_xy,
        "active_start_m": float(active_start_m),
        "active_stop_m": float(active_stop_m),
        "mode": mode,
        "maximum_amplitude_m": float(maximum_amplitude_m),
    }
    mapped_roots, root_profile = _map_points(roots, **path_arguments)
    if root_height_offset_m is not None:
        height_offset = np.asarray(root_height_offset_m, dtype=np.float64)
        if height_offset.shape != (len(roots),) or not np.isfinite(
            height_offset
        ).all():
            raise ValueError(
                "root height offset must contain one finite value per frame"
            )
        mapped_roots[:, 2] += height_offset
    else:
        height_offset = np.zeros(len(roots), dtype=np.float64)
    if root_progress_offset_m is not None:
        progress_offset = np.asarray(root_progress_offset_m, dtype=np.float64)
        if progress_offset.shape != (len(roots),) or not np.isfinite(
            progress_offset
        ).all():
            raise ValueError(
                "root progress offset must contain one finite value per frame"
            )
        mapped_roots[:, :2] += progress_offset[:, None] * direction_xy[None]
    else:
        progress_offset = np.zeros(len(roots), dtype=np.float64)
    mapped_quaternions = np.empty_like(quaternions)
    source_soles: list[tuple[np.ndarray, np.ndarray]] = []
    source_envelopes: list[tuple[np.ndarray, np.ndarray]] = []
    source_centres = np.empty((len(roots), 2, 3), dtype=np.float64)
    for frame in range(len(roots)):
        delta_yaw = float(
            root_profile.pose_yaw_offset_rad[frame]
            + root_profile.facing_yaw_offset_rad[frame]
        )
        mapped_quaternions[frame] = quaternion_multiply_wxyz(
            _yaw_quaternion(delta_yaw), quaternions[frame]
        )
        mapped_quaternions[frame] /= np.linalg.norm(mapped_quaternions[frame])
        feet = adapter.sole_positions_for_pose(
            root_position=roots[frame],
            root_quaternion_wxyz=quaternions[frame],
            joints=authored_joints[frame],
        )
        source_soles.append(feet)
        source_envelopes.append(
            adapter.foot_collision_envelope_points_for_pose(
                root_position=roots[frame],
                root_quaternion_wxyz=quaternions[frame],
                joints=authored_joints[frame],
            )
        )
        source_centres[frame] = np.asarray(
            [np.mean(value, axis=0) for value in feet]
        )
    fps = float(source.fps)
    source_speed = np.linalg.norm(
        np.gradient(source_centres, axis=0) * fps, axis=2
    )
    sphere_radii = adapter.sole_sphere_radii()
    support_mesh = target_mesh if source_mesh is None else source_mesh
    source_ray_z = float(np.max(support_mesh.vertices_world[:, 2]) + 1.0)
    ray_z = float(np.max(target_mesh.vertices_world[:, 2]) + 1.0)
    source_support_count = np.zeros((len(roots), 2), dtype=np.int16)
    for frame, feet in enumerate(source_soles):
        for foot in range(2):
            support = feet[foot].copy()
            support[:, 2] -= sphere_radii[foot]
            source_support_count[frame, foot] = sum(
                abs(
                    float(point[2])
                    - _terrain_height(support_mesh, point[:2], source_ray_z)
                )
                <= 0.020
                for point in support
            )
    stance = _remove_short_stance_runs(
        _fill_short_stance_gaps(
            (source_support_count >= 2) & (source_speed <= 0.25),
            maximum_gap_frames=maximum_stance_gap_frames,
        ),
        minimum_run_frames=minimum_stance_run_frames,
    )

    # First move each authored sole with the pelvis' path transform.  This is
    # exactly reachable when facing is unchanged.  Then freeze every authored
    # stance run at its middle-frame pose and interpolate those anchor edits
    # through swing.  Using a foot's spatial progress directly here causes a
    # leading swing foot to see the curve before the pelvis does; on a hard
    # diagonal that requested an impossible 30--40 cm one-frame reach.  The
    # stance-anchor formulation is the usual game-animation construction:
    # turns happen by choosing successive footholds while planted feet remain
    # fixed in world coordinates.
    nominal_targets = np.empty(
        (
            len(roots),
            2,
            len(source_soles[0][0]),
            3,
        ),
        dtype=np.float64,
    )
    nominal_collision_envelopes: list[tuple[np.ndarray, np.ndarray]] = []
    for frame, feet in enumerate(source_soles):
        foot_yaw = float(root_profile.foot_yaw_offset_rad[frame])
        mapped_envelopes: list[np.ndarray] = []
        for foot in range(2):
            if spatial_sole_mapping:
                nominal_targets[frame, foot], _ = _map_rigid_sole(
                    feet[foot], **path_arguments
                )
                mapped_envelope, _ = _map_rigid_sole(
                    source_envelopes[frame][foot], **path_arguments
                )
                mapped_envelopes.append(mapped_envelope)
            else:
                nominal_targets[frame, foot] = (
                    mapped_roots[frame]
                    + _rotate_z(feet[foot] - roots[frame], foot_yaw)
                )
                mapped_envelopes.append(
                    mapped_roots[frame]
                    + _rotate_z(
                        source_envelopes[frame][foot] - roots[frame], foot_yaw
                    )
                )
        nominal_collision_envelopes.append(
            (mapped_envelopes[0], mapped_envelopes[1])
        )
    if target_route is None:
        target_soles_array = _anchor_stance_soles_to_continuous_mesh(
            nominal_targets,
            stance,
            sphere_radii,
            target_mesh=target_mesh,
            ray_z=ray_z,
            source_nominal_targets=np.asarray(source_soles),
            source_mesh=source_mesh,
            source_ray_z=source_ray_z,
        )
    else:
        anchor_config = foothold_anchor_config or FootholdAnchorConfig(
            max_longitudinal_adjustment_m=0.16,
            max_lateral_adjustment_m=0.0,
            max_yaw_adjustment_rad=math.radians(12.0),
            longitudinal_samples=17,
            lateral_samples=1,
            yaw_samples=5,
            lateral_seed_offsets_m=(0.0,),
            route_lateral_offset_m=0.0,
            route_alignment_weight=0.0,
            minimum_support_points=2,
        )
        target_soles_array, _anchor_diagnostics = _anchor_stance_sole_targets(
            nominal_targets,
            stance,
            sphere_radii,
            nominal_collision_envelopes,
            target_mesh=target_mesh,
            foothold_route=target_route,
            ground_fallback_height_m=0.0,
            config=anchor_config,
            maximum_monotonic_level_step=maximum_foothold_level_step,
            target_level_by_stance_span=target_level_by_stance_span,
        )
        if terrain_clear_swings:
            (
                target_soles_array,
                bracketed_swing,
                swing_route_adjustment,
            ) = _terrain_clear_bracketed_swings(
                nominal_targets,
                target_soles_array,
                stance,
                sphere_radii,
                target_mesh=target_mesh,
                target_route=target_route,
                minimum_clearance_m=minimum_swing_clearance_m,
                adjustment_taper_frames=swing_adjustment_taper_frames,
            )
        else:
            bracketed_swing = np.zeros(stance.shape, dtype=bool)
            swing_route_adjustment = np.zeros(
                (len(stance), 2, 3), dtype=np.float64
            )
    target_soles = [
        (target_soles_array[frame, 0], target_soles_array[frame, 1])
        for frame in range(len(target_soles_array))
    ]

    intended_roots = mapped_roots.copy()

    # Keep the pelvis inside the support polygon implied by the newly locked
    # footholds.  A path-only pelvis can otherwise be several centimetres too
    # far around a bend during double support, leaving the IK solver to absorb
    # the entire curve with one leg.  The shift is derived from active support
    # feet, interpolated through flight, and smoothed; it does not alter any
    # planted sole target.
    root_anchor_shift = np.zeros((len(roots), 3), dtype=np.float64)
    root_anchor_fixed = np.zeros(len(roots), dtype=bool)
    for frame in range(len(roots)):
        active = np.flatnonzero(stance[frame])
        if not len(active):
            continue
        provisional = adapter.sole_positions_for_pose(
            root_position=mapped_roots[frame],
            root_quaternion_wxyz=mapped_quaternions[frame],
            joints=authored_joints[frame],
        )
        root_anchor_shift[frame] = np.mean(
            np.stack(
                [
                    np.mean(target_soles[frame][foot], axis=0)
                    - np.mean(provisional[foot], axis=0)
                    for foot in active
                ]
            ),
            axis=0,
        )
        root_anchor_fixed[frame] = True
    root_anchor_shift = _interpolate_fixed_vectors(
        root_anchor_shift, root_anchor_fixed
    )
    if not root_anchor_vertical:
        if root_anchor_vertical_limit_m is None:
            root_anchor_shift[:, 2] = 0.0
        else:
            vertical_limit = float(root_anchor_vertical_limit_m)
            if not math.isfinite(vertical_limit) or vertical_limit < 0.0:
                raise ValueError(
                    "root anchor vertical limit must be finite and nonnegative"
                )
            root_anchor_shift[:, 2] = np.clip(
                root_anchor_shift[:, 2], -vertical_limit, vertical_limit
            )
    root_anchor_shift = _smooth_root_anchor_shifts(
        root_anchor_shift,
        sigma_frames=(
            float(root_anchor_smoothing_frames)
            if root_anchor_smoothing_frames is not None
            else (1.0 if target_route is None else 4.0)
        ),
    )
    mapped_roots += root_anchor_shift

    adapted_joints = np.empty_like(authored_joints)
    corrections = np.empty(len(roots), dtype=np.float64)
    errors = np.empty(len(roots), dtype=np.float64)
    errors_by_foot = np.empty((len(roots), 2), dtype=np.float64)
    adapted_centres = np.empty((len(roots), 2, 3), dtype=np.float64)
    target_support_count = np.zeros((len(roots), 2), dtype=np.int16)
    for frame in range(len(roots)):
        warm_candidate = adapter.adapt_to_targets(
            root_position=mapped_roots[frame],
            root_quaternion_wxyz=mapped_quaternions[frame],
            authored_joints=authored_joints[frame],
            sole_targets_world=target_soles[frame],
            initial_joints=(adapted_joints[frame - 1] if frame else None),
            continuity_joints=(adapted_joints[frame - 1] if frame else None),
            continuity_feet=tuple(
                not bool(stance[frame, foot]) for foot in range(2)
            ),
            maximum_continuity_joint_step_rad=(
                maximum_ik_continuity_step_rad if frame else None
            ),
        )
        candidates = [warm_candidate]
        if ik_multistart and frame:
            candidates.append(
                adapter.adapt_to_targets(
                    root_position=mapped_roots[frame],
                    root_quaternion_wxyz=mapped_quaternions[frame],
                    authored_joints=authored_joints[frame],
                    sole_targets_world=target_soles[frame],
                    initial_joints=None,
                    continuity_joints=adapted_joints[frame - 1],
                    continuity_feet=tuple(
                        not bool(stance[frame, foot]) for foot in range(2)
                    ),
                    maximum_continuity_joint_step_rad=(
                        maximum_ik_continuity_step_rad
                    ),
                )
            )
            continuity_candidate = np.asarray(
                authored_joints[frame], dtype=np.float64
            ).copy()
            leg_addresses = np.concatenate(adapter._leg_qpos)
            leg_mask = np.isin(adapter._joint_addresses, leg_addresses)
            continuity_candidate[leg_mask] = adapted_joints[frame - 1][
                leg_mask
            ]
            continuity_feet = adapter.sole_positions_for_pose(
                root_position=mapped_roots[frame],
                root_quaternion_wxyz=mapped_quaternions[frame],
                joints=continuity_candidate,
            )
            continuity_error = max(
                float(
                    np.max(
                        np.linalg.norm(
                            target_soles[frame][foot]
                            - continuity_feet[foot],
                            axis=1,
                        )
                    )
                )
                for foot in range(2)
            )
            candidates.append(
                (
                    continuity_candidate,
                    float(
                        np.max(
                            np.abs(
                                continuity_candidate
                                - authored_joints[frame]
                            )
                        )
                    ),
                    continuity_error,
                )
            )
        acceptable = [
            value
            for value in candidates
            if value[2] <= float(ik_continuity_acceptable_error_m)
        ]
        if acceptable:
            near_best = acceptable
        else:
            best_error = min(value[2] for value in candidates)
            near_best = [
                value
                for value in candidates
                if value[2] <= best_error + 0.002
            ]
        selected = min(
            near_best,
            key=lambda value: (
                (
                    0.0
                    if not frame
                    else float(
                        np.max(
                            np.abs(value[0] - adapted_joints[frame - 1])
                        )
                    )
                ),
                value[1],
            ),
        )
        adapted_joints[frame], corrections[frame], errors[frame] = selected
        if frame and post_selection_maximum_joint_step_rad is not None:
            maximum_step = float(post_selection_maximum_joint_step_rad)
            if not math.isfinite(maximum_step) or maximum_step <= 0.0:
                raise ValueError("post-selection joint step must be positive")
            adapted_joints[frame] = np.clip(
                adapted_joints[frame],
                adapted_joints[frame - 1] - maximum_step,
                adapted_joints[frame - 1] + maximum_step,
            )
            corrections[frame] = float(
                np.max(
                    np.abs(
                        adapted_joints[frame] - authored_joints[frame]
                    )
                )
            )
        final_feet = adapter.sole_positions_for_pose(
            root_position=mapped_roots[frame],
            root_quaternion_wxyz=mapped_quaternions[frame],
            joints=adapted_joints[frame],
        )
        adapted_centres[frame] = np.asarray(
            [np.mean(value, axis=0) for value in final_feet]
        )
        for foot in range(2):
            errors_by_foot[frame, foot] = float(
                np.max(
                    np.linalg.norm(
                        target_soles[frame][foot] - final_feet[foot], axis=1
                    )
                )
            )
            support = final_feet[foot].copy()
            support[:, 2] -= sphere_radii[foot]
            support_points = 0
            for point in support:
                height = _terrain_height(target_mesh, point[:2], ray_z)
                if not math.isfinite(height) and ground_fallback_height_m is not None:
                    height = float(ground_fallback_height_m)
                if abs(float(point[2]) - height) <= 0.020:
                    support_points += 1
            target_support_count[frame, foot] = support_points
        errors[frame] = float(np.max(errors_by_foot[frame]))

    advertised_support = target_support_count[stance]
    minimum_support = int(np.min(advertised_support)) if len(advertised_support) else 0
    root_step = float(np.max(np.linalg.norm(np.diff(mapped_roots, axis=0), axis=1)))
    quaternion_dot = np.abs(
        np.sum(mapped_quaternions[1:] * mapped_quaternions[:-1], axis=1)
    )
    rotation_step = float(
        np.max(2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0)))
    )
    joint_step = float(np.max(np.abs(np.diff(adapted_joints, axis=0))))
    root_acceleration = np.diff(mapped_roots, n=2, axis=0) * fps * fps
    diagnostics = WarpDiagnostics(
        maximum_joint_correction_rad=float(np.max(corrections)),
        maximum_sole_target_error_m=float(np.max(errors)),
        maximum_stance_sole_target_error_m=float(
            np.max(errors_by_foot[stance]) if np.any(stance) else math.inf
        ),
        maximum_swing_sole_target_error_m=float(
            np.max(errors_by_foot[~stance]) if np.any(~stance) else 0.0
        ),
        maximum_stance_run_drift_m=_stance_runs_maximum_drift(
            adapted_centres, stance
        ),
        minimum_stance_support_point_count=minimum_support,
        maximum_root_translation_step_m=root_step,
        maximum_root_rotation_step_rad=rotation_step,
        maximum_joint_step_rad=joint_step,
        maximum_root_acceleration_m_s2=float(
            np.max(np.linalg.norm(root_acceleration, axis=1))
        ),
        maximum_root_anchor_shift_m=float(
            np.max(np.linalg.norm(root_anchor_shift, axis=1))
        ),
        maximum_lateral_offset_m=float(
            np.max(np.abs(root_profile.lateral_offset_m))
        ),
        lateral_offset_range_m=float(
            np.ptp(root_profile.lateral_offset_m)
        ),
        realized_root_progress_range_m=float(
            np.ptp(
                _path_coordinates(
                    mapped_roots,
                    origin_xy=origin_xy,
                    direction_xy=direction_xy,
                )[0]
            )
        ),
        realized_root_lateral_range_m=float(
            np.ptp(
                _path_coordinates(
                    mapped_roots,
                    origin_xy=origin_xy,
                    direction_xy=direction_xy,
                )[1]
            )
        ),
        realized_root_vertical_range_m=float(np.ptp(mapped_roots[:, 2])),
        maximum_path_angle_deg=root_profile.maximum_path_angle_deg,
        maximum_facing_offset_deg=float(
            np.degrees(np.max(np.abs(root_profile.facing_yaw_offset_rad)))
        ),
    )
    motion = StitchedMotion(
        fps=source.fps,
        root_position_world=np.asarray(mapped_roots, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(
            mapped_quaternions, dtype=np.float32
        ),
        joint_position=np.asarray(adapted_joints, dtype=np.float32),
        provenance=source.provenance,
        seam_indices=source.seam_indices,
    )
    extras = {
        "path_lateral_offset_m": np.asarray(
            root_profile.lateral_offset_m, dtype=np.float32
        ),
        "path_yaw_offset_rad": np.asarray(
            root_profile.path_yaw_offset_rad, dtype=np.float32
        ),
        "pose_yaw_offset_rad": np.asarray(
            root_profile.pose_yaw_offset_rad, dtype=np.float32
        ),
        "foot_yaw_offset_rad": np.asarray(
            root_profile.foot_yaw_offset_rad, dtype=np.float32
        ),
        "facing_yaw_offset_rad": np.asarray(
            root_profile.facing_yaw_offset_rad, dtype=np.float32
        ),
        "path_normalized_progress": np.asarray(
            root_profile.normalized_progress, dtype=np.float32
        ),
        "intended_root_position_world": np.asarray(
            intended_roots, dtype=np.float32
        ),
        "authored_stance_mask": stance,
        "target_stance_support_point_count": target_support_count,
        "per_frame_sole_target_error_m": np.asarray(errors, dtype=np.float32),
        "per_frame_sole_target_error_by_foot_m": np.asarray(
            errors_by_foot, dtype=np.float32
        ),
        "per_frame_joint_correction_rad": np.asarray(
            corrections, dtype=np.float32
        ),
        "target_sole_center_world": np.asarray(
            [
                [np.mean(value, axis=0) for value in frame]
                for frame in target_soles
            ],
            dtype=np.float32,
        ),
        "target_sole_points_world": np.asarray(
            target_soles_array, dtype=np.float32
        ),
        "nominal_sole_center_world": np.asarray(
            np.mean(nominal_targets, axis=2), dtype=np.float32
        ),
        "adapted_sole_center_world": np.asarray(
            adapted_centres, dtype=np.float32
        ),
        "terrain_root_anchor_shift_world": np.asarray(
            root_anchor_shift, dtype=np.float32
        ),
        "bracketed_swing_mask": np.asarray(
            (
                bracketed_swing
                if target_route is not None
                else np.zeros(stance.shape, dtype=bool)
            ),
            dtype=np.bool_,
        ),
        "per_frame_swing_route_adjustment": np.asarray(
            (
                swing_route_adjustment
                if target_route is not None
                else np.zeros((len(stance), 2, 3), dtype=np.float64)
            ),
            dtype=np.float32,
        ),
        "terrain_root_height_offset_m": np.asarray(
            height_offset, dtype=np.float32
        ),
        "terrain_root_progress_offset_m": np.asarray(
            progress_offset, dtype=np.float32
        ),
    }
    return motion, diagnostics, extras


def _wrapped_yaw(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64)
    w, x, y, z = q.T
    return np.unwrap(
        np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )


def _save_motion(
    path: Path,
    motion: StitchedMotion,
    *,
    mode: str,
    extras: dict[str, np.ndarray],
) -> None:
    root = np.asarray(motion.root_position_world, dtype=np.float64)
    velocity_world = np.gradient(root[:, :2], axis=0) * float(motion.fps)
    facing = _wrapped_yaw(motion.root_quaternion_world_wxyz)
    cosine, sine = np.cos(facing), np.sin(facing)
    velocity_local = np.stack(
        (
            cosine * velocity_world[:, 0] + sine * velocity_world[:, 1],
            -sine * velocity_world[:, 0] + cosine * velocity_world[:, 1],
        ),
        axis=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fps=np.asarray(motion.fps, dtype=np.float32),
        root_position_world=np.asarray(motion.root_position_world, np.float32),
        root_quaternion_world_wxyz=np.asarray(
            motion.root_quaternion_world_wxyz, np.float32
        ),
        joint_position=np.asarray(motion.joint_position, np.float32),
        seam_indices=np.asarray(motion.seam_indices, dtype=np.int64),
        source_archive_clip_index=np.asarray(
            [value.archive_clip_index for value in motion.provenance],
            dtype=np.int64,
        ),
        source_frame=np.asarray(
            [value.source_frame for value in motion.provenance], dtype=np.int64
        ),
        source_clip_id=np.asarray(
            [value.clip_id for value in motion.provenance], dtype=np.str_
        ),
        maneuver_mode=np.asarray(mode, dtype=np.str_),
        command_velocity_world_xy=np.asarray(velocity_world, dtype=np.float32),
        command_velocity_robot_local_xy=np.asarray(
            velocity_local, dtype=np.float32
        ),
        command_facing_yaw_world_rad=np.asarray(facing, dtype=np.float32),
        **extras,
    )


def _source_paths(
    selection: dict[str, object],
    *,
    include_reverse: bool,
) -> tuple[tuple[str, Path], ...]:
    direct_source = selection.get("source_motion")
    if direct_source is not None:
        source = Path(str(direct_source)).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"prepared source motion is missing: {source}")
        if include_reverse:
            raise ValueError(
                "direct source selections do not provide a reverse pilot"
            )
        return (("native", source),)
    manifest_path = Path(str(selection["pilot"])).expanduser().resolve()
    if "#" in str(manifest_path):
        # Path strips no fragments itself, but retain support for a plain dict.
        manifest_path = Path(str(selection["pilot"]).split("#", 1)[0])
    manifest = json.loads(manifest_path.read_text())
    source = manifest_path.parent.parent / "source" / "motion.npz"
    values: list[tuple[str, Path]] = [("native", source)]
    if include_reverse:
        reverse = next(
            (
                Path(str(row["motion"]))
                for row in manifest["pilots"]
                if row.get("mode") == "reverse"
                and bool(row.get("automatic_gate_accepted"))
            ),
            None,
        )
        if reverse is not None:
            values.append(("reverse", reverse))
    return tuple(values)


def _temporally_reverse_motion(motion: StitchedMotion) -> StitchedMotion:
    """Reverse a clean traversal without changing any spatial pose.

    Time reversal preserves exact terrain contacts.  A forward ascent becomes
    a backward descent, and a forward descent becomes a backward ascent.  It
    therefore adds the missing negative local-velocity support without
    synthesizing a physically dubious backward stair gait.
    """

    frame_count = len(motion.root_position_world)
    seams = tuple(
        sorted(
            frame_count - int(index)
            for index in motion.seam_indices
            if 0 < int(index) < frame_count
        )
    )
    return StitchedMotion(
        fps=motion.fps,
        root_position_world=np.asarray(motion.root_position_world)[::-1].copy(),
        root_quaternion_world_wxyz=np.asarray(
            motion.root_quaternion_world_wxyz
        )[::-1].copy(),
        joint_position=np.asarray(motion.joint_position)[::-1].copy(),
        provenance=tuple(reversed(motion.provenance)),
        seam_indices=seams,
    )


def _active_route_interval(route: object) -> tuple[float, float]:
    boundaries = _level_boundaries(route)
    if len(boundaries) < 3:
        raise ValueError("stair route has no interior support transition")
    start = max(0.0, float(boundaries[1]) - 0.30)
    stop = min(float(boundaries[-1]), float(boundaries[-2]) + 0.30)
    if stop - start < 0.75:
        start = float(boundaries[0])
        stop = float(boundaries[-1])
    return start, stop


def _selected_rows(
    selection_path: Path,
    clip_indices: Sequence[int],
    *,
    source_bank_root: Path,
) -> list[dict[str, object]]:
    payload = json.loads(selection_path.read_text())
    source_rows = payload.get("selections")
    if source_rows is None:
        source_rows = payload.get("sources")
        if source_rows is None:
            raise ValueError("selection must contain selections or sources")
        expanded_rows: list[dict[str, object]] = []
        bank = source_bank_root.expanduser().resolve()
        for source_row in source_rows:
            row = dict(source_row)
            clip_index = int(row["clip_index"])
            clip_name = str(row["clip_name"])
            source_motion = (
                bank
                / f"clip_{clip_index:03d}_{clip_name}"
                / "source"
                / "motion.npz"
            )
            if not source_motion.is_file():
                raise FileNotFoundError(
                    f"prepared source motion is missing: {source_motion}"
                )
            row["source_motion"] = str(source_motion)
            expanded_rows.append(row)
        source_rows = expanded_rows
    requested = tuple(int(value) for value in clip_indices)
    unique: dict[int, dict[str, object]] = {}
    for row in source_rows:
        index = int(row["clip_index"])
        if requested and index not in requested:
            continue
        unique.setdefault(index, row)
    if requested:
        missing = sorted(set(requested) - set(unique))
        if missing:
            raise ValueError(f"selected clip indices are unavailable: {missing}")
        return [unique[index] for index in requested]
    # A deliberately diverse four-source default pilot: ordinary and steep
    # ascents plus ordinary and shallow descents, all on wide synthetic stairs.
    defaults = (364, 6, 343, 207)
    return [unique[index] for index in defaults if index in unique]


def build(arguments: argparse.Namespace) -> dict[str, object]:
    archive_path = arguments.archive.expanduser().resolve()
    archive = zarr.open_group(str(archive_path), mode="r")
    joint_names = tuple(str(value) for value in archive["joint_names"][:])
    adapter = _G1FootfallAdapter(
        arguments.model.expanduser().resolve(),
        joint_names,
        maximum_joint_correction_rad=arguments.maximum_joint_correction_rad,
    )
    rows = _selected_rows(
        arguments.selection,
        arguments.clip_index,
        source_bank_root=arguments.source_bank_root,
    )
    modes = tuple(arguments.mode) if arguments.mode else tuple(
        name
        for name in (
            "diagonal_soft_left",
            "diagonal_soft_right",
            "diagonal_hard_left",
            "diagonal_hard_right",
            "lane_left",
            "lane_right",
            "zigzag_left_right",
            "zigzag_right_left",
            "crab_left",
            "crab_right",
            "face_left",
            "face_right",
            "turning_left",
            "turning_right",
        )
    )
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    accepted = 0
    attempted = 0
    for row in rows:
        clip_index = int(row["clip_index"])
        target_mesh = _archive_terrain_index(archive, clip_index)
        travel_yaw = float(archive["travel_yaw_rad"][clip_index])
        direction = np.asarray(
            (math.cos(travel_yaw), math.sin(travel_yaw)), dtype=np.float64
        )
        target_route = (
            None
            if arguments.continuous_terrain
            else motion_conditioned_stair_support_route(archive, clip_index)
        )
        if target_route is None:
            active_start = 0.0
            active_stop = 0.0
        else:
            active_start, active_stop = _active_route_interval(target_route)
        source_entries = [
            (kind, path, False)
            for kind, path in _source_paths(
                row, include_reverse=arguments.include_reverse
            )
        ]
        if arguments.include_temporal_reverse:
            native_path = source_entries[0][1]
            source_entries.append(("temporal_reverse", native_path, True))
        for source_kind, source_path, temporal_reverse in source_entries:
            source = load_stitched_motion_npz(source_path)
            source_direction = direction
            source_active_start = active_start
            source_active_stop = active_stop
            if temporal_reverse:
                source = _temporally_reverse_motion(source)
                source_direction = -direction
            source_progress, _source_lateral = _path_coordinates(
                np.asarray(source.root_position_world, dtype=np.float64),
                origin_xy=np.asarray(source.root_position_world[0, :2]),
                direction_xy=source_direction,
            )
            source_progress_range = float(np.ptp(source_progress))
            source_vertical_range = float(
                np.ptp(source.root_position_world[:, 2])
            )
            if (
                source_progress_range < 0.50
                or source_vertical_range < arguments.minimum_vertical_range_m
            ):
                print(
                    "STAIRS500_DIRECTIONAL_SOURCE_SKIPPED "
                    f"clip={clip_index} source={source_kind} "
                    f"progress_range_m={source_progress_range:.3f} "
                    f"vertical_range_m={source_vertical_range:.3f}",
                    flush=True,
                )
                continue
            if target_route is None:
                source_active_start = 0.10 * source_progress_range
                source_active_stop = 0.90 * source_progress_range
            if temporal_reverse:
                if target_route is None:
                    source_active_start = 0.10 * source_progress_range
                    source_active_stop = 0.90 * source_progress_range
                else:
                    source_active_start = max(
                        0.0, source_progress_range - active_stop
                    )
                    source_active_stop = min(
                        source_progress_range,
                        source_progress_range - active_start,
                    )
                if source_active_stop - source_active_start < 0.75:
                    source_active_start = 0.0
                    source_active_stop = source_progress_range
            for mode in modes:
                # Native straight is already present in the clean archive.
                # The straight mode exists to add its missing backward-time
                # counterpart without duplicating the source clip.
                if mode == "straight" and not temporal_reverse:
                    continue
                attempted += 1
                label = f"clip_{clip_index:03d}_{source_kind}_{mode}"
                destination = output / label
                destination.mkdir(parents=True, exist_ok=True)
                report: dict[str, object] = {
                    "label": label,
                    "clip_index": clip_index,
                    "clip_name": str(archive["clip_names"][clip_index]),
                    "source_traversal": str(
                        archive["clip_traversal"][clip_index]
                    ),
                    "traversal": (
                        "down"
                        if temporal_reverse
                        and str(archive["clip_traversal"][clip_index]) == "up"
                        else "up"
                        if temporal_reverse
                        and str(archive["clip_traversal"][clip_index]) == "down"
                        else str(archive["clip_traversal"][clip_index])
                    ),
                    "source_kind": source_kind,
                    "source_motion": str(source_path),
                    "mode": mode,
                    "route_yaw_rad": float(
                        travel_yaw + (math.pi if temporal_reverse else 0.0)
                    ),
                    "active_route_interval_m": [
                        source_active_start,
                        source_active_stop,
                    ],
                    "status": "rejected",
                }
                try:
                    identity_temporal_reverse = bool(
                        temporal_reverse and mode == "straight"
                    )
                    if identity_temporal_reverse:
                        motion = source
                        sample_count = len(source.root_position_world)
                        extras = {
                            "path_lateral_offset_m": np.zeros(
                                sample_count, dtype=np.float32
                            ),
                            "path_yaw_offset_rad": np.zeros(
                                sample_count, dtype=np.float32
                            ),
                            "pose_yaw_offset_rad": np.zeros(
                                sample_count, dtype=np.float32
                            ),
                            "foot_yaw_offset_rad": np.zeros(
                                sample_count, dtype=np.float32
                            ),
                            "facing_yaw_offset_rad": np.zeros(
                                sample_count, dtype=np.float32
                            ),
                            "path_normalized_progress": np.clip(
                                (
                                    source_progress - source_active_start
                                )
                                / max(
                                    source_active_stop - source_active_start,
                                    1.0e-6,
                                ),
                                0.0,
                                1.0,
                            ).astype(np.float32),
                            "intended_root_position_world": np.asarray(
                                source.root_position_world, dtype=np.float32
                            ),
                        }
                        report["warp"] = {
                            "identity_temporal_reverse": True,
                            "realized_root_progress_range_m": (
                                source_progress_range
                            ),
                            "realized_root_vertical_range_m": (
                                source_vertical_range
                            ),
                        }
                        realized = True
                        mechanical = True
                    else:
                        motion, diagnostics, extras = warp_motion(
                            source,
                            adapter=adapter,
                            target_mesh=target_mesh,
                            target_route=target_route,
                            direction_xy=source_direction,
                            active_start_m=source_active_start,
                            active_stop_m=source_active_stop,
                            mode=mode,
                            maximum_amplitude_m=arguments.maximum_amplitude_m,
                        )
                        report["warp"] = asdict(diagnostics)
                        (
                            _knots,
                            requested_path_angle_deg,
                            requested_facing_offset_deg,
                            _pose_gain,
                            _foot_gain,
                        ) = _profile_knots(mode)
                        realized = bool(
                            (
                                requested_path_angle_deg == 0.0
                                or (
                                    diagnostics.lateral_offset_range_m >= 0.02
                                    and diagnostics.maximum_path_angle_deg
                                    >= 0.55 * requested_path_angle_deg
                                )
                            )
                            and (
                                requested_facing_offset_deg == 0.0
                                or diagnostics.maximum_facing_offset_deg
                                >= 0.55 * abs(requested_facing_offset_deg)
                            )
                        )
                        mechanical = bool(
                            diagnostics.maximum_joint_correction_rad
                            <= arguments.maximum_joint_correction_rad + 1.0e-6
                            and diagnostics.maximum_stance_sole_target_error_m
                            <= arguments.maximum_stance_sole_target_error_m
                            and diagnostics.maximum_swing_sole_target_error_m
                            <= arguments.maximum_swing_sole_target_error_m
                            and diagnostics.maximum_stance_run_drift_m
                            <= arguments.maximum_stance_run_drift_m
                            and diagnostics.minimum_stance_support_point_count
                            >= arguments.minimum_stance_support_points
                            and diagnostics.maximum_root_translation_step_m
                            <= 0.035
                            and diagnostics.maximum_root_rotation_step_rad
                            <= 0.080
                            and diagnostics.maximum_joint_step_rad <= 0.20
                            and diagnostics.maximum_root_acceleration_m_s2
                            <= 30.0
                            and realized
                        )
                    report["realized_command_accepted"] = realized
                    report["mechanical_accepted"] = mechanical
                    if not mechanical:
                        # A full MuJoCo mesh audit cannot rescue a motion that
                        # already violates the kinematic/command gate.  Keep
                        # the compact diagnostic report, but avoid auditing
                        # and storing an unusable multi-megabyte candidate.
                        report["collision_audit_skipped_reason"] = (
                            "mechanical_rejection"
                        )
                        (destination / "report.json").write_text(
                            json.dumps(report, indent=2, sort_keys=True) + "\n"
                        )
                        reports.append(report)
                        print(
                            "STAIRS500_DIRECTIONAL "
                            f"{attempted} label={label} "
                            f"status={report['status']}",
                            flush=True,
                        )
                        continue
                    collision = audit_stair_motion_collisions(
                        motion,
                        model_path=arguments.model,
                        target_mesh=target_mesh,
                        joint_names=joint_names,
                        maximum_foot_penetration_m=(
                            arguments.maximum_foot_penetration_m
                        ),
                        maximum_forbidden_body_penetration_m=0.0,
                    )
                    report["pre_clearance_collision_audit"] = collision.to_dict()
                    clearance_lift = 0.0
                    if (
                        not collision.accepted
                        and collision.maximum_forbidden_body_penetration_m == 0.0
                        and collision.maximum_foot_penetration_m <= 0.020
                    ):
                        original_motion = motion
                        motion, _repair_audit, clearance_lift = (
                            _repair_exact_mesh_clearance(
                                motion,
                                collision,
                                archive_path=archive_path,
                                target_clip_index=clip_index,
                                target_mesh=target_mesh,
                                model_path=arguments.model,
                                maximum_total_lift_m=(
                                    arguments.maximum_clearance_lift_m
                                ),
                            )
                        )
                        extras["clearance_root_lift_m"] = np.asarray(
                            motion.root_position_world[:, 2]
                            - original_motion.root_position_world[:, 2],
                            dtype=np.float32,
                        )
                        collision = audit_stair_motion_collisions(
                            motion,
                            model_path=arguments.model,
                            target_mesh=target_mesh,
                            joint_names=joint_names,
                            maximum_foot_penetration_m=(
                                arguments.maximum_foot_penetration_m
                            ),
                            maximum_forbidden_body_penetration_m=0.0,
                        )
                    report["clearance_repair_maximum_m"] = float(
                        clearance_lift
                    )
                    report["collision_audit"] = collision.to_dict()
                    report["status"] = (
                        "accepted" if mechanical and collision.accepted else "rejected"
                    )
                    _save_motion(
                        destination / "motion.npz",
                        motion,
                        mode=mode,
                        extras=extras,
                    )
                    (destination / "collision_audit.json").write_text(
                        json.dumps(collision.to_dict(), indent=2, sort_keys=True)
                        + "\n"
                    )
                    if report["status"] == "accepted":
                        accepted += 1
                except Exception as error:  # preserve every failed pilot reason
                    report["error"] = f"{type(error).__name__}: {error}"
                (destination / "report.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n"
                )
                reports.append(report)
                print(
                    "STAIRS500_DIRECTIONAL "
                    f"{attempted} label={label} status={report['status']}",
                    flush=True,
                )
    summary = {
        "schema": "stairs500-omnidirectional-pilot/v1",
        "archive": str(archive_path),
        "selection": str(arguments.selection.expanduser().resolve()),
        "model": str(arguments.model.expanduser().resolve()),
        "attempted": attempted,
        "accepted": accepted,
        "rejected": attempted - accepted,
        "clip_indices": [int(row["clip_index"]) for row in rows],
        "modes": list(modes),
        "include_reverse": bool(arguments.include_reverse),
        "include_temporal_reverse": bool(
            arguments.include_temporal_reverse
        ),
        "continuous_terrain": bool(arguments.continuous_terrain),
        "reports": reports,
    }
    (output / "aggregate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--source-bank-root", type=Path, default=DEFAULT_SOURCE_BANK_ROOT
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clip-index", type=int, action="append", default=[])
    parser.add_argument("--mode", action="append", default=[])
    parser.add_argument("--include-reverse", action="store_true")
    parser.add_argument("--include-temporal-reverse", action="store_true")
    parser.add_argument(
        "--continuous-terrain",
        action="store_true",
        help=(
            "anchor stance probes directly to a continuous terrain mesh "
            "instead of requiring discrete stair support levels"
        ),
    )
    parser.add_argument("--minimum-vertical-range-m", type=float, default=0.06)
    parser.add_argument("--maximum-amplitude-m", type=float, default=0.42)
    parser.add_argument("--maximum-joint-correction-rad", type=float, default=0.45)
    parser.add_argument(
        "--maximum-stance-sole-target-error-m", type=float, default=0.005
    )
    parser.add_argument(
        "--maximum-swing-sole-target-error-m", type=float, default=0.060
    )
    parser.add_argument("--maximum-stance-run-drift-m", type=float, default=0.015)
    parser.add_argument("--minimum-stance-support-points", type=int, default=2)
    parser.add_argument("--maximum-foot-penetration-m", type=float, default=0.005)
    parser.add_argument("--maximum-clearance-lift-m", type=float, default=0.018)
    return parser


def main(argv: list[str] | None = None) -> int:
    summary = build(_parser().parse_args(argv))
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "reports"},
            indent=2,
            sort_keys=True,
        )
    )
    # Zero admissions from one source is a valid geometric result, not a job
    # failure.  Every attempted variant and its rejection reason is already
    # persisted in the reports; returning nonzero here unnecessarily blocks
    # heterogeneous source sweeps and their post-processing dependencies.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
