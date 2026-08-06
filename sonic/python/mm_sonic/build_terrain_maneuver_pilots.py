"""Build exact-audited stop/restart/reversal pilots on known terrain courses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import zarr

from .build_c490_curb_slope_archive import DEFAULT_OUTPUT as DEFAULT_ARCHIVE
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .compose_coherent_block_plan import _mechanically_accepted
from .compose_privileged_stair_route import _maximum_steps
from .generate_generic_stair_route import load_target_mesh
from .motionbricks_global_terrain_viewer import _terrain_surface_height
from .render_stitched_motion import load_stitched_motion_npz
from .terrain_maneuver_composer import (
    HOLD,
    build_maneuver_schedule,
    choose_support_pivot,
    resample_stitched_motion,
    save_maneuver_motion,
)
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.stitch import StitchedMotion


DEFAULT_MANIFEST = Path(
    "/move/u/bodow/Projects/Motion-Matching-takara-corpus/artifacts/"
    "generic_terrain/waypoint_routes/mixed_gauntlet_v1/portals_entry_v2/"
    "portal_manifest.json"
)


def _foot_kinematics(
    motion: StitchedMotion,
    *,
    adapter: _G1FootfallAdapter,
    terrain: object,
    ray_origin_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    clearance = np.empty((len(motion.root_position_world), 2), dtype=np.float64)
    centre = np.empty((len(motion.root_position_world), 2, 3), dtype=np.float64)
    for frame, (root, quaternion, joints) in enumerate(
        zip(
            motion.root_position_world,
            motion.root_quaternion_world_wxyz,
            motion.joint_position,
            strict=True,
        )
    ):
        feet = adapter.sole_support_points_for_pose(
            root_position=np.asarray(root, dtype=np.float64),
            root_quaternion_wxyz=np.asarray(quaternion, dtype=np.float64),
            joints=np.asarray(joints, dtype=np.float64),
        )
        for foot, points in enumerate(feet):
            values = np.asarray(points, dtype=np.float64)
            centre[frame, foot] = np.mean(values, axis=0)
            clearance[frame, foot] = min(
                float(point[2])
                - _terrain_surface_height(
                    terrain, point[:2], ray_origin_z=ray_origin_z
                )
                for point in values
            )
    return clearance, centre


def _run_maximum_drift(
    positions_xy: np.ndarray, support: np.ndarray
) -> tuple[float, float]:
    maximum_step = 0.0
    maximum_run_drift = 0.0
    for foot in range(2):
        index = 0
        while index < len(support):
            if not support[index, foot]:
                index += 1
                continue
            start = index
            while index < len(support) and support[index, foot]:
                index += 1
            stop = index
            if stop - start < 2:
                continue
            values = positions_xy[start:stop, foot]
            maximum_step = max(
                maximum_step,
                float(np.max(np.linalg.norm(np.diff(values, axis=0), axis=1))),
            )
            maximum_run_drift = max(
                maximum_run_drift,
                float(np.max(np.linalg.norm(values - values[0], axis=1))),
            )
    return maximum_step, maximum_run_drift


def _quality_metrics(
    motion: StitchedMotion,
    *,
    clearance_m: np.ndarray,
    foot_centres_world: np.ndarray,
    hold_mask: np.ndarray,
    support_threshold_m: float,
    support_speed_mps: float,
) -> dict[str, object]:
    root = np.asarray(motion.root_position_world, dtype=np.float64)
    joints = np.asarray(motion.joint_position, dtype=np.float64)
    fps = float(motion.fps)
    root_acceleration = np.diff(root, n=2, axis=0) * fps * fps
    root_jerk = np.diff(root, n=3, axis=0) * fps * fps * fps
    joint_acceleration = np.diff(joints, n=2, axis=0) * fps * fps
    foot_speed = np.linalg.norm(
        np.gradient(foot_centres_world, axis=0) * fps, axis=2
    )
    support = (
        (clearance_m >= -0.0051)
        & (clearance_m <= float(support_threshold_m))
        & (foot_speed <= float(support_speed_mps))
    )
    maximum_step, maximum_run_drift = _run_maximum_drift(
        foot_centres_world[:, :, :2], support
    )
    minimum_support = np.min(clearance_m, axis=1)
    hold_clearance = clearance_m[hold_mask]
    hold_left = (
        None
        if len(hold_clearance) == 0
        else float(np.max(hold_clearance[:, 0]))
    )
    hold_right = (
        None
        if len(hold_clearance) == 0
        else float(np.max(hold_clearance[:, 1]))
    )
    return {
        "support_threshold_m": float(support_threshold_m),
        "support_speed_threshold_mps": float(support_speed_mps),
        "maximum_minimum_sole_clearance_m": float(np.max(minimum_support)),
        "maximum_stance_foot_step_m": maximum_step,
        "maximum_stance_run_drift_m": maximum_run_drift,
        "maximum_root_jerk_m_s3": float(
            np.max(np.linalg.norm(root_jerk, axis=1))
        ),
        "root_acceleration_p99_m_s2": float(
            np.percentile(np.linalg.norm(root_acceleration, axis=1), 99.0)
        ),
        "joint_acceleration_p99_rad_s2": float(
            np.percentile(np.abs(joint_acceleration), 99.0)
        ),
        "hold_frame_count": int(np.sum(hold_mask)),
        "hold_maximum_left_clearance_m": hold_left,
        "hold_maximum_right_clearance_m": hold_right,
    }


def _direct_terrain_transform(
    *,
    archive_path: Path,
    terrain_usd: Path,
    explicit_position: object | None,
    explicit_quaternion_wxyz: object | None,
) -> tuple[list[float], list[float], str]:
    """Resolve a direct motion's terrain transform without assuming identity.

    GRAIL USD meshes are stored in their source authoring frame.  The clean
    archive records the rigid transform that places each mesh in the same
    world frame as its retargeted motion.  An identity fallback is therefore
    unsafe: it can render and audit a valid motion against a perpendicular
    copy of the right mesh.  Fill omitted values from the matching archive
    record and require explicit values for terrains outside that archive.
    """

    position = (
        None
        if explicit_position is None
        else np.asarray(explicit_position, dtype=np.float64)
    )
    quaternion = (
        None
        if explicit_quaternion_wxyz is None
        else np.asarray(explicit_quaternion_wxyz, dtype=np.float64)
    )
    if position is not None and position.shape != (3,):
        raise ValueError("terrain position must contain three values")
    if quaternion is not None and quaternion.shape != (4,):
        raise ValueError("terrain quaternion must contain four values")
    if position is not None and quaternion is not None:
        return position.tolist(), quaternion.tolist(), "explicit"

    archive = zarr.open_group(str(archive_path.expanduser().resolve()), mode="r")
    requested = terrain_usd.expanduser().resolve()
    matches = [
        index
        for index, value in enumerate(archive["terrain_usd_path"][:])
        if Path(str(value)).expanduser().resolve() == requested
    ]
    if not matches:
        missing = []
        if position is None:
            missing.append("--terrain-position")
        if quaternion is None:
            missing.append("--terrain-quaternion-wxyz")
        raise ValueError(
            "terrain USD is not present in the motion archive; provide "
            + " and ".join(missing)
        )
    archive_position = np.asarray(
        archive["terrain_position_env"][matches[0]], dtype=np.float64
    )
    archive_quaternion = np.asarray(
        archive["terrain_rotation_env_wxyz"][matches[0]], dtype=np.float64
    )
    for index in matches[1:]:
        candidate_position = np.asarray(
            archive["terrain_position_env"][index], dtype=np.float64
        )
        candidate_quaternion = np.asarray(
            archive["terrain_rotation_env_wxyz"][index], dtype=np.float64
        )
        quaternion_agrees = bool(
            np.allclose(candidate_quaternion, archive_quaternion, atol=1.0e-6)
            or np.allclose(
                candidate_quaternion, -archive_quaternion, atol=1.0e-6
            )
        )
        if (
            not np.allclose(
                candidate_position, archive_position, atol=1.0e-6
            )
            or not quaternion_agrees
        ):
            raise ValueError(
                "matching archive terrain records disagree on transform"
            )
    return (
        (archive_position if position is None else position).tolist(),
        (archive_quaternion if quaternion is None else quaternion).tolist(),
        (
            "archive_lookup"
            if position is None and quaternion is None
            else "explicit_plus_archive"
        ),
    )


def build_pilots(arguments: argparse.Namespace) -> dict[str, object]:
    direct_sources = tuple(
        Path(value).expanduser().resolve() for value in arguments.source_motion
    )
    if direct_sources:
        if arguments.terrain_usd is None:
            raise ValueError("--terrain-usd is required with --source-motion")
        terrain_usd = arguments.terrain_usd.expanduser().resolve()
        terrain_position, terrain_quaternion, transform_source = (
            _direct_terrain_transform(
                archive_path=arguments.motion_archive,
                terrain_usd=terrain_usd,
                explicit_position=arguments.terrain_position,
                explicit_quaternion_wxyz=arguments.terrain_quaternion_wxyz,
            )
        )
        manifest_path: Path | None = None
        manifest = {
            "terrain_usd": str(terrain_usd),
            "terrain_position_world": terrain_position,
            "terrain_quaternion_world_from_usd_wxyz": terrain_quaternion,
            "terrain_transform_source": transform_source,
            "course_motions": [str(value) for value in direct_sources],
            "events": [
                {"kind": "direct_terrain"} for _value in direct_sources
            ],
        }
    else:
        manifest_path = arguments.portal_manifest.expanduser().resolve()
        manifest = json.loads(manifest_path.read_text())
    target_mesh = load_target_mesh(
        Path(str(manifest["terrain_usd"])),
        position_world=manifest["terrain_position_world"],
        quaternion_world_from_usd_wxyz=(
            manifest["terrain_quaternion_world_from_usd_wxyz"]
        ),
    )
    archive = zarr.open_group(
        str(arguments.motion_archive.expanduser().resolve()), mode="r"
    )
    joint_names = tuple(str(value) for value in archive["joint_names"][:])
    adapter = _G1FootfallAdapter(
        arguments.model_path.expanduser().resolve(),
        joint_names,
        maximum_joint_correction_rad=0.35,
    )
    ray_origin_z = float(np.max(target_mesh.vertices_world[:, 2]) + 2.0)
    selected_events = (
        tuple(int(value) for value in arguments.event_index)
        if arguments.event_index
        else tuple(range(len(manifest["course_motions"])))
    )
    modes = tuple(str(value) for value in arguments.mode)
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for event_index in selected_events:
        source_path = Path(
            str(manifest["course_motions"][event_index])
        ).expanduser().resolve()
        source = load_stitched_motion_npz(source_path)
        source_clearance, source_centres = _foot_kinematics(
            source,
            adapter=adapter,
            terrain=target_mesh,
            ray_origin_z=ray_origin_z,
        )
        source_foot_speed = np.linalg.norm(
            np.gradient(source_centres, axis=0) * source.fps, axis=2
        )
        terrain_start = (
            int(source.seam_indices[0]) if source.seam_indices else 0
        )
        terrain_stop = (
            int(source.seam_indices[-1])
            if len(source.seam_indices) > 1
            else len(source.root_position_world)
        )
        pivot = choose_support_pivot(
            source,
            source_clearance,
            per_foot_speed_mps=source_foot_speed,
            terrain_start_frame=terrain_start,
            terrain_stop_frame=terrain_stop,
            maximum_support_clearance_m=arguments.pivot_clearance_m,
            maximum_support_speed_mps=arguments.pivot_speed_mps,
            central_fraction=tuple(arguments.pivot_central_fraction),
        )
        event = manifest["events"][event_index]
        for mode in modes:
            ramp_frames = int(arguments.ramp_frames)
            hold_frames = int(arguments.hold_frames)
            if mode == "reverse":
                reverse_ramp = getattr(arguments, "reverse_ramp_frames", None)
                reverse_hold = getattr(arguments, "reverse_hold_frames", None)
                if reverse_ramp is not None:
                    ramp_frames = int(reverse_ramp)
                if reverse_hold is not None:
                    hold_frames = int(reverse_hold)
            schedule = build_maneuver_schedule(
                len(source.root_position_world),
                pivot_source_frame=pivot.frame_index,
                mode=mode,
                ramp_frames=ramp_frames,
                hold_frames=hold_frames,
            )
            motion = resample_stitched_motion(source, schedule)
            label = f"event_{event_index:02d}_{event['kind']}_{mode}"
            destination = output / label
            destination.mkdir(parents=True, exist_ok=True)
            motion_path = save_maneuver_motion(
                destination / "motion.npz", motion, schedule
            )
            clearance, centres = _foot_kinematics(
                motion,
                adapter=adapter,
                terrain=target_mesh,
                ray_origin_z=ray_origin_z,
            )
            mechanics = _maximum_steps(motion)
            collision = audit_stair_motion_collisions(
                motion,
                target_mesh=target_mesh,
                joint_names=joint_names,
                model_path=arguments.model_path,
                maximum_foot_penetration_m=arguments.maximum_foot_penetration_m,
                maximum_forbidden_body_penetration_m=0.0,
            )
            quality = _quality_metrics(
                motion,
                clearance_m=clearance,
                foot_centres_world=centres,
                hold_mask=np.asarray(schedule.phase == HOLD),
                support_threshold_m=arguments.support_clearance_m,
                support_speed_mps=arguments.support_speed_mps,
            )
            source_quality = _quality_metrics(
                source,
                clearance_m=source_clearance,
                foot_centres_world=source_centres,
                hold_mask=np.zeros(len(source.root_position_world), dtype=bool),
                support_threshold_m=arguments.support_clearance_m,
                support_speed_mps=arguments.support_speed_mps,
            )
            collision_payload = collision.to_dict()
            collision_path = destination / "collision_audit.json"
            collision_path.write_text(
                json.dumps(collision_payload, indent=2, sort_keys=True) + "\n"
            )
            automatic = bool(
                collision.accepted
                and _mechanically_accepted(mechanics)
                and quality["maximum_minimum_sole_clearance_m"]
                <= arguments.maximum_any_support_clearance_m
                and quality["maximum_stance_foot_step_m"]
                <= arguments.maximum_stance_step_m
                and quality["maximum_stance_foot_step_m"]
                <= source_quality["maximum_stance_foot_step_m"] + 0.003
                and quality["maximum_stance_run_drift_m"]
                <= arguments.maximum_stance_run_drift_m
                and quality["maximum_stance_run_drift_m"]
                <= source_quality["maximum_stance_run_drift_m"] + 0.005
                and quality["hold_maximum_left_clearance_m"]
                <= arguments.pivot_clearance_m
                and quality["hold_maximum_right_clearance_m"]
                <= arguments.pivot_clearance_m
            )
            row: dict[str, object] = {
                "label": label,
                "event_index": event_index,
                "kind": str(event["kind"]),
                "mode": mode,
                "ramp_frames": ramp_frames,
                "hold_frames": hold_frames,
                "status": (
                    "pending_dense_visual_review"
                    if automatic
                    else "automatic_gate_rejected"
                ),
                "motion": str(motion_path),
                "source_motion": str(source_path),
                "frame_count": len(motion.root_position_world),
                "duration_s": float(
                    len(motion.root_position_world) / motion.fps
                ),
                "pivot": {
                    "frame_index": pivot.frame_index,
                    "left_clearance_m": pivot.left_clearance_m,
                    "right_clearance_m": pivot.right_clearance_m,
                    "root_speed_mps": pivot.root_speed_mps,
                    "joint_speed_rms_rad_s": pivot.joint_speed_rms_rad_s,
                    "terrain_progress": pivot.terrain_progress,
                },
                "mechanics": mechanics,
                "collision_audit": {
                    "accepted": bool(collision.accepted),
                    "maximum_foot_penetration_m": float(
                        collision.maximum_foot_penetration_m
                    ),
                    "maximum_forbidden_body_penetration_m": float(
                        collision.maximum_forbidden_body_penetration_m
                    ),
                    "report": str(collision_path),
                },
                "quality": quality,
                "source_quality": source_quality,
                "automatic_gate_accepted": automatic,
            }
            (destination / "summary.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n"
            )
            rows.append(row)
            print(
                f"{label}: {row['status']} "
                f"foot={collision.maximum_foot_penetration_m:.4f}m "
                f"slide={quality['maximum_stance_run_drift_m']:.4f}m",
                flush=True,
            )

    result: dict[str, object] = {
        "schema": "terrain-maneuver-pilot-manifest/v1",
        "source_portal_manifest": (
            None if manifest_path is None else str(manifest_path)
        ),
        "direct_source_motion_count": len(direct_sources),
        "terrain_usd": str(Path(str(manifest["terrain_usd"])).resolve()),
        "terrain_position_world": manifest["terrain_position_world"],
        "terrain_quaternion_world_from_usd_wxyz": (
            manifest["terrain_quaternion_world_from_usd_wxyz"]
        ),
        "terrain_transform_source": manifest.get(
            "terrain_transform_source", "portal_manifest"
        ),
        "pilot_count": len(rows),
        "automatic_gate_accepted_count": sum(
            bool(row["automatic_gate_accepted"]) for row in rows
        ),
        "pilots": rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portal-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--source-motion",
        type=Path,
        action="append",
        default=[],
        help="build directly from a motion on one exact terrain instead of a portal manifest",
    )
    parser.add_argument("--terrain-usd", type=Path)
    parser.add_argument("--terrain-position", type=float, nargs=3)
    parser.add_argument(
        "--terrain-quaternion-wxyz",
        type=float,
        nargs=4,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--event-index", type=int, action="append", default=[])
    parser.add_argument(
        "--mode",
        choices=("stop_restart", "reverse"),
        action="append",
        default=[],
    )
    parser.add_argument("--ramp-frames", type=int, default=24)
    parser.add_argument("--hold-frames", type=int, default=30)
    parser.add_argument(
        "--reverse-ramp-frames",
        type=int,
        help="override --ramp-frames for direction reversals",
    )
    parser.add_argument(
        "--reverse-hold-frames",
        type=int,
        help="override --hold-frames for direction reversals",
    )
    parser.add_argument("--pivot-clearance-m", type=float, default=0.008)
    parser.add_argument("--pivot-speed-mps", type=float, default=0.12)
    parser.add_argument(
        "--pivot-central-fraction",
        type=float,
        nargs=2,
        default=(0.20, 0.80),
    )
    parser.add_argument("--support-clearance-m", type=float, default=0.005)
    parser.add_argument("--support-speed-mps", type=float, default=0.10)
    parser.add_argument(
        "--maximum-any-support-clearance-m", type=float, default=0.060
    )
    parser.add_argument("--maximum-foot-penetration-m", type=float, default=0.005)
    parser.add_argument("--maximum-stance-step-m", type=float, default=0.003)
    parser.add_argument(
        "--maximum-stance-run-drift-m", type=float, default=0.012
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.mode:
        arguments.mode = ["stop_restart", "reverse"]
    result = build_pilots(arguments)
    print(
        json.dumps(
            {
                "pilot_count": result["pilot_count"],
                "automatic_gate_accepted_count": result[
                    "automatic_gate_accepted_count"
                ],
                "manifest": str(arguments.output.expanduser().resolve() / "manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0 if result["automatic_gate_accepted_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
