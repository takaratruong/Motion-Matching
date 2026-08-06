"""Probe a clean MotionBricks walk on gentle bumps and continuous slopes.

These diagnostic pilots preserve the authored joints, XY path, and cadence,
then apply a conservative vertical root-clearance envelope.  That envelope is
useful for rejecting impossible terrain early, but collision clearance alone
does not prove valid support: lifting the whole body can leave a stance foot
hovering.  The automatic gate therefore also requires the source motion's
intended stance feet to remain close to the exact target surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from .build_c490_curb_slope_archive import DEFAULT_OUTPUT as DEFAULT_ARCHIVE
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .canonical_terrain_matcher import RegularGridHeightField
from .compose_coherent_block_plan import _mechanically_accepted
from .compose_motionbricks_terrain_course import _resample_motionbricks
from .compose_privileged_stair_route import _maximum_steps
from .terrain_maneuver_composer import PHASE_NAMES
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.stitch import StitchedMotion


DEFAULT_SOURCE = Path(
    "/move/u/bodow/Projects/reference-nvidia-groot-wbc/motionbricks/"
    "motionbricks_sonic_smoke_20260803_nativeqpos/recordings/canonical/"
    "straight_start_stop.npz"
)

PROFILES = ("rolling_bumps", "smooth_hill", "cross_slope_bumps")


def _smootherstep(value: object) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def build_profile(
    profile: str,
    *,
    spacing_m: float = 0.04,
) -> tuple[RegularGridHeightField, dict[str, object]]:
    """Return a bounded, flat-entry/exit procedural terrain profile."""

    name = str(profile)
    if name not in PROFILES:
        raise ValueError(f"unknown rolling terrain profile {name!r}")
    spacing = float(spacing_m)
    x = np.arange(-0.8, 4.0 + spacing / 2.0, spacing)
    y = np.arange(-1.2, 1.2 + spacing / 2.0, spacing)
    grid_x, grid_y = np.meshgrid(x, y)
    enter = _smootherstep((grid_x - 0.30) / 0.45)
    leave = _smootherstep((3.45 - grid_x) / 0.45)
    window = enter * leave

    if name == "rolling_bumps":
        height = window * (
            0.018 * np.sin(2.0 * np.pi * (grid_x - 0.30) / 1.20)
            + 0.008 * np.sin(2.0 * np.pi * grid_y / 1.35)
        )
        description = "low rolling bumps with a weak lateral undulation"
    elif name == "smooth_hill":
        phase = np.clip((grid_x - 0.35) / 3.00, 0.0, 1.0)
        height = window * 0.12 * np.sin(np.pi * phase) ** 2
        description = "continuous twelve-centimetre up-and-down hill"
    else:
        height = window * (
            0.040 * grid_y
            + 0.010 * np.sin(2.0 * np.pi * (grid_x - 0.30) / 1.30)
        )
        description = "cross-slope with superimposed longitudinal bumps"

    field = RegularGridHeightField(
        height,
        spacing_m=spacing,
        origin_xy=(float(x[0]), float(y[0])),
    )
    metadata: dict[str, object] = {
        "profile": name,
        "description": description,
        "spacing_m": spacing,
        "origin_xy": [float(x[0]), float(y[0])],
        "shape": list(height.shape),
        "x_range_m": [float(x[0]), float(x[-1])],
        "y_range_m": [float(y[0]), float(y[-1])],
        "height_range_m": [float(np.min(height)), float(np.max(height))],
        "flat_entry_exit": True,
    }
    return field, metadata


def intended_stance_support_metrics(
    source_support_points: object,
    *,
    applied_root_shift_m: object,
    field: RegularGridHeightField,
    fps: float,
    source_contact_clearance_m: float = 0.005,
    source_contact_speed_mps: float = 0.10,
) -> dict[str, float | int]:
    """Measure target-surface hover for stance phases authored on flat ground.

    ``source_support_points`` contains exact sole-sphere bottoms in the clean
    source pose.  Contact is inferred only from a sole close to the source
    floor *and* a slow foot.  The same samples are then vertically displaced
    by the proposed root envelope and queried against the target height field.
    This catches the otherwise invisible failure mode where collision passes
    because the entire robot was lifted above the terrain.
    """

    points = np.asarray(source_support_points, dtype=np.float64)
    shift = np.asarray(applied_root_shift_m, dtype=np.float64)
    if (
        points.ndim != 4
        or points.shape[1] != 2
        or points.shape[-1] != 3
        or shift.shape != (len(points),)
        or not np.isfinite(points).all()
        or not np.isfinite(shift).all()
        or not np.isfinite(float(fps))
        or float(fps) <= 0.0
    ):
        raise ValueError("support samples or root shift are invalid")

    source_minimum_z = np.min(points[..., 2], axis=2)
    source_centres = np.mean(points, axis=2)
    source_speed = np.linalg.norm(
        np.gradient(source_centres, axis=0) * float(fps), axis=2
    )
    # The MotionBricks and audit MJCF sole proxies have a small constant
    # vertical convention offset.  Calibrate that offset from slow-foot
    # samples, not the deepest one-percent sample: the latter is usually a
    # brief toe-down extreme and would classify only a few accidental frames.
    slow = source_speed <= float(source_contact_speed_mps)
    if not np.all(np.any(slow, axis=0)):
        raise ValueError("source motion has no slow samples for one foot")
    source_floor_z = np.asarray(
        [np.median(source_minimum_z[slow[:, foot], foot]) for foot in range(2)],
        dtype=np.float64,
    )
    source_clearance = source_minimum_z - source_floor_z[None, :]
    intended_stance = (
        (source_clearance >= -0.0051)
        & (source_clearance <= float(source_contact_clearance_m))
        & (source_speed <= float(source_contact_speed_mps))
    )
    if not np.any(intended_stance):
        raise ValueError("source motion has no strictly classified stance samples")

    adapted = points.copy()
    adapted[..., 2] += shift[:, None, None]
    target_clearance = np.empty(adapted.shape[:-1], dtype=np.float64)
    for frame in range(len(adapted)):
        for foot in range(2):
            surface, _normal, hit = field.sample(adapted[frame, foot, :, :2])
            if not np.all(hit):
                raise ValueError("adapted foot leaves procedural terrain")
            target_clearance[frame, foot] = adapted[frame, foot, :, 2] - surface

    stance_minimum = np.min(target_clearance, axis=2)[intended_stance]
    stance_span = np.ptp(target_clearance, axis=2)[intended_stance]
    return {
        "intended_stance_frame_foot_count": int(len(stance_minimum)),
        "intended_stance_hover_over_5mm_count": int(
            np.sum(stance_minimum > 0.005)
        ),
        "intended_stance_hover_over_10mm_count": int(
            np.sum(stance_minimum > 0.010)
        ),
        "p95_intended_stance_minimum_clearance_m": float(
            np.quantile(stance_minimum, 0.95)
        ),
        "maximum_intended_stance_minimum_clearance_m": float(
            np.max(stance_minimum)
        ),
        "p95_intended_stance_sole_clearance_span_m": float(
            np.quantile(stance_span, 0.95)
        ),
        "maximum_intended_stance_sole_clearance_span_m": float(
            np.max(stance_span)
        ),
    }


def _adapt_motion(
    source: StitchedMotion,
    *,
    field: RegularGridHeightField,
    adapter: _G1FootfallAdapter,
    minimum_ground_clearance_m: float,
) -> tuple[StitchedMotion, dict[str, float]]:
    roots = np.asarray(source.root_position_world, dtype=np.float64).copy()
    quaternions = np.asarray(
        source.root_quaternion_world_wxyz, dtype=np.float64
    ).copy()
    joints = np.asarray(source.joint_position, dtype=np.float64).copy()
    source_support_points: list[tuple[np.ndarray, np.ndarray]] = []
    for frame in range(len(roots)):
        pose = {
            "root_position": np.asarray(
                source.root_position_world[frame], np.float64
            ),
            "root_quaternion_wxyz": quaternions[frame],
            "joints": np.asarray(source.joint_position[frame], np.float64),
        }
        source_support_points.append(adapter.sole_support_points_for_pose(**pose))
    required_shift = np.empty(len(roots), dtype=np.float64)
    for frame, feet in enumerate(source_support_points):
        source_clearance = np.asarray(
            [float(np.min(points[:, 2])) for points in feet], dtype=np.float64
        )
        selected = source_clearance <= float(np.min(source_clearance) + 0.025)
        per_foot_required = np.empty(2, dtype=np.float64)
        for foot, points in enumerate(feet):
            surface, _normal, hit = field.sample(points[:, :2])
            if not np.all(hit):
                raise ValueError("source foot leaves procedural terrain")
            per_foot_required[foot] = float(
                np.max(surface - points[:, 2])
            )
        required_shift[frame] = float(
            np.max(per_foot_required[selected])
            + float(minimum_ground_clearance_m)
        )

    # A short symmetric look-ahead raises the pelvis before a support edge
    # arrives.  The final max with the exact requirement preserves the hard
    # no-penetration envelope while avoiding framewise height chatter.
    radius = 4
    padded = np.pad(required_shift, (radius, radius), mode="edge")
    envelope = np.asarray(
        [np.max(padded[index : index + 2 * radius + 1]) for index in range(len(roots))]
    )
    kernel = np.asarray((1, 2, 3, 4, 5, 4, 3, 2, 1), dtype=np.float64)
    kernel /= np.sum(kernel)
    smoothed = np.convolve(envelope, kernel, mode="same")
    applied_shift = np.maximum(smoothed, required_shift)
    roots[:, 2] += applied_shift

    support_metrics = intended_stance_support_metrics(
        np.asarray(source_support_points, dtype=np.float64),
        applied_root_shift_m=applied_shift,
        field=field,
        fps=source.fps,
    )

    return (
        StitchedMotion(
            fps=source.fps,
            root_position_world=np.asarray(roots, dtype=np.float32),
            root_quaternion_world_wxyz=np.asarray(quaternions, dtype=np.float32),
            joint_position=np.asarray(joints, dtype=np.float32),
            provenance=source.provenance,
            seam_indices=source.seam_indices,
        ),
        {
            "maximum_joint_correction_rad": 0.0,
            "maximum_sole_target_error_m": 0.0,
            "minimum_ground_clearance_m": float(minimum_ground_clearance_m),
            "minimum_applied_root_shift_m": float(np.min(applied_shift)),
            "maximum_applied_root_shift_m": float(np.max(applied_shift)),
            "maximum_root_shift_step_m": float(
                np.max(np.abs(np.diff(applied_shift)))
            ),
            "maximum_root_shift_acceleration_m_s2": float(
                np.max(np.abs(np.diff(applied_shift, n=2)))
                * source.fps
                * source.fps
            ),
            **support_metrics,
        },
    )


def _save_motion(path: Path, motion: StitchedMotion) -> Path:
    root = np.asarray(motion.root_position_world, dtype=np.float64)
    velocity = np.gradient(root[:, :2], axis=0) * motion.fps
    quaternion = np.asarray(motion.root_quaternion_world_wxyz, dtype=np.float64)
    w, x, y, z = quaternion.T
    facing = np.unwrap(
        np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
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
            [value.archive_clip_index for value in motion.provenance], np.int64
        ),
        source_frame=np.asarray(
            [value.source_frame for value in motion.provenance], np.int64
        ),
        source_clip_id=np.asarray(
            [value.clip_id for value in motion.provenance], dtype=np.str_
        ),
        command_velocity_world_xy=np.asarray(velocity, dtype=np.float32),
        command_facing_yaw_world_rad=np.asarray(facing, dtype=np.float32),
        command_stop=np.asarray(
            np.linalg.norm(velocity, axis=1) < 0.04, dtype=np.bool_
        ),
        maneuver_phase=np.zeros(len(root), dtype=np.uint8),
        maneuver_phase_names=np.asarray(PHASE_NAMES, dtype=np.str_),
    )
    return path.resolve()


def _source_motion(path: Path) -> StitchedMotion:
    segment = _resample_motionbricks(path.expanduser().resolve(), label="motionbricks_flat")
    return StitchedMotion(
        fps=50.0,
        root_position_world=segment.root_position_world,
        root_quaternion_world_wxyz=segment.root_quaternion_world_wxyz,
        joint_position=segment.joint_position,
        provenance=segment.provenance,
        seam_indices=(),
    )


def build(arguments: argparse.Namespace) -> dict[str, object]:
    source = _source_motion(arguments.source)
    archive = zarr.open_group(str(arguments.motion_archive.resolve()), mode="r")
    joint_names = tuple(str(value) for value in archive["joint_names"][:])
    adapter = _G1FootfallAdapter(
        arguments.model_path.resolve(),
        joint_names,
        # This pilot uses the adapter only for exact forward kinematics.
        # No joint-space terrain correction is applied.
        maximum_joint_correction_rad=0.0,
    )
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles = tuple(arguments.profile) or PROFILES
    rows = []
    for name in profiles:
        field, terrain_metadata = build_profile(
            name, spacing_m=arguments.terrain_spacing_m
        )
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        terrain_path = destination / "terrain.npz"
        np.savez_compressed(
            terrain_path,
            height=np.asarray(field.height, dtype=np.float32),
            valid=np.asarray(field.valid, dtype=np.bool_),
            spacing_m=np.asarray(field.spacing_m, dtype=np.float32),
            origin_xy=np.asarray(field.origin_xy, dtype=np.float32),
        )
        motion, adaptation = _adapt_motion(
            source,
            field=field,
            adapter=adapter,
            minimum_ground_clearance_m=arguments.minimum_ground_clearance_m,
        )
        motion_path = _save_motion(destination / "motion.npz", motion)
        mechanics = _maximum_steps(motion)
        collision = audit_stair_motion_collisions(
            motion,
            target_mesh=field.index,
            joint_names=joint_names,
            model_path=arguments.model_path,
            maximum_foot_penetration_m=arguments.maximum_foot_penetration_m,
            maximum_forbidden_body_penetration_m=0.0,
        )
        collision_path = destination / "collision_audit.json"
        collision_path.write_text(
            json.dumps(collision.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        automatic = bool(
            collision.accepted
            and _mechanically_accepted(mechanics)
            and adaptation[
                "maximum_intended_stance_minimum_clearance_m"
            ]
            <= arguments.maximum_intended_stance_hover_m
            and adaptation[
                "p95_intended_stance_minimum_clearance_m"
            ]
            <= arguments.p95_intended_stance_hover_m
        )
        row = {
            "label": name,
            "status": (
                "pending_dense_visual_review"
                if automatic
                else "automatic_gate_rejected"
            ),
            "motion": str(motion_path),
            "terrain_npz": str(terrain_path.resolve()),
            "terrain": terrain_metadata,
            "source_motion": str(arguments.source.resolve()),
            "frame_count": len(motion.root_position_world),
            "duration_s": float(len(motion.root_position_world) / motion.fps),
            "adaptation": adaptation,
            "mechanics": mechanics,
            "collision_audit": {
                "accepted": bool(collision.accepted),
                "maximum_foot_penetration_m": float(
                    collision.maximum_foot_penetration_m
                ),
                "maximum_forbidden_body_penetration_m": float(
                    collision.maximum_forbidden_body_penetration_m
                ),
                "report": str(collision_path.resolve()),
            },
            "automatic_gate_accepted": automatic,
        }
        (destination / "summary.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )
        rows.append(row)
        print(
            f"{name}: {row['status']} correction="
            f"{adaptation['maximum_joint_correction_rad']:.3f}rad error="
            f"{adaptation['maximum_sole_target_error_m']:.4f}m",
            flush=True,
        )
    result = {
        "schema": "rolling-terrain-pilot-manifest/v1",
        "source_motion": str(arguments.source.resolve()),
        "pilot_count": len(rows),
        "automatic_gate_accepted_count": sum(
            bool(row["automatic_gate_accepted"]) for row in rows
        ),
        "pilots": rows,
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {**result, "manifest": str(manifest)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, action="append", default=[])
    parser.add_argument("--motion-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--terrain-spacing-m", type=float, default=0.04)
    parser.add_argument(
        "--minimum-ground-clearance-m", type=float, default=0.002
    )
    parser.add_argument("--maximum-foot-penetration-m", type=float, default=0.005)
    parser.add_argument(
        "--maximum-intended-stance-hover-m", type=float, default=0.010
    )
    parser.add_argument(
        "--p95-intended-stance-hover-m", type=float, default=0.006
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    result = build(_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "pilot_count": result["pilot_count"],
                "automatic_gate_accepted_count": result[
                    "automatic_gate_accepted_count"
                ],
                "manifest": result["manifest"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["automatic_gate_accepted_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
