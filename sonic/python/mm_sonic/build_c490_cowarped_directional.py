"""Author directional rough-terrain kinematics by co-warping motion and mesh.

The earlier continuous-terrain experiment bent a registered motion while
leaving its terrain fixed.  That asks the legs to invent new footholds on an
unrelated surface and fails on most hills.  Here the exact registered terrain
and its motion are deformed by the same smooth stair-local path map.  Contact
therefore remains authored; G1 IK only absorbs the small finite-foot error of
the spatially varying rigid transform.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr

from .build_c490_curb_slope_archive import DEFAULT_OUTPUT as DEFAULT_ARCHIVE
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .build_stairs500_omnidirectional_pilots import (
    _map_points,
    _path_coordinates,
    _save_motion,
    warp_motion,
)
from .compose_coherent_block_plan import _repair_exact_mesh_clearance
from .evaluate_archive_identity_traversal import evaluate
from .render_stitched_motion import load_stitched_motion_npz, render_stitched_motion
from .terrain_oracle.canonical import CanonicalTerrainMesh
from .terrain_oracle.math3d import RigidTransform
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.stair_geometry_warp import _archive_terrain_index
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


DEFAULT_OUTPUT = Path(
    "/move/data/terrain-aware/sonic-rollouts/"
    "terrain_maneuver_general_v1/c490_cowarped_directional_v1"
)
DEFAULT_MODES = (
    "cowarp_diagonal_left",
    "cowarp_diagonal_right",
    "cowarp_lane_left",
    "cowarp_lane_right",
    "cowarp_slalom_left_right",
    "cowarp_slalom_right_left",
)


def _identity_transform() -> RigidTransform:
    return RigidTransform(
        np.zeros(3, dtype=np.float32),
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
    )


def cowarp_terrain_mesh(
    source_mesh: TerrainMeshIndex,
    *,
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
    active_start_m: float,
    active_stop_m: float,
    mode: str,
    maximum_amplitude_m: float,
) -> tuple[TerrainMeshIndex, dict[str, float]]:
    """Apply the exact motion path map to every registered terrain vertex."""

    arguments = {
        "origin_xy": np.asarray(origin_xy, dtype=np.float64),
        "direction_xy": np.asarray(direction_xy, dtype=np.float64),
        "active_start_m": float(active_start_m),
        "active_stop_m": float(active_stop_m),
        "mode": str(mode),
        "maximum_amplitude_m": float(maximum_amplitude_m),
    }
    warped_vertices, profile = _map_points(
        np.asarray(source_mesh.vertices_world, dtype=np.float64),
        **arguments,
    )
    source_triangles = np.asarray(source_mesh.vertices_world, dtype=np.float64)[
        source_mesh.mesh.faces
    ]
    target_triangles = warped_vertices[source_mesh.mesh.faces]
    source_normals = np.cross(
        source_triangles[:, 1] - source_triangles[:, 0],
        source_triangles[:, 2] - source_triangles[:, 0],
    )
    target_normals = np.cross(
        target_triangles[:, 1] - target_triangles[:, 0],
        target_triangles[:, 2] - target_triangles[:, 0],
    )
    source_area2 = np.linalg.norm(source_normals, axis=1)
    target_area2 = np.linalg.norm(target_normals, axis=1)
    face_centres = np.mean(source_triangles, axis=1)
    face_progress, face_lateral = _path_coordinates(
        face_centres,
        origin_xy=np.asarray(origin_xy, dtype=np.float64),
        direction_xy=np.asarray(direction_xy, dtype=np.float64),
    )
    # Archive meshes often contain a large remote ground apron.  A strong
    # curve can self-overlap that irrelevant apron even though the complete
    # robot corridor remains a clean, one-to-one deformation.  Export and
    # audit the route corridor plus generous approach/side context instead of
    # retaining folded geometry the motion can never reach.
    valid_faces = (
        np.asarray(source_mesh.mesh.valid_faces, dtype=bool)
        & (face_progress >= float(active_start_m) - 1.25)
        & (face_progress <= float(active_stop_m) + 1.25)
        & (np.abs(face_lateral) <= 1.50)
    )
    usable = (
        valid_faces
        & (source_area2 > 1.0e-10)
        & (target_area2 > 1.0e-10)
    )
    if not np.any(usable):
        raise ValueError("co-warped terrain contains no usable triangle")
    normal_cosine = np.sum(
        source_normals[usable] * target_normals[usable], axis=1
    ) / (source_area2[usable] * target_area2[usable])
    area_ratio = target_area2[usable] / source_area2[usable]
    diagnostics = {
        "maximum_vertex_displacement_m": float(
            np.max(
                np.linalg.norm(
                    warped_vertices - source_mesh.vertices_world, axis=1
                )
            )
        ),
        "minimum_face_normal_cosine": float(np.min(normal_cosine)),
        "minimum_face_area_ratio": float(np.min(area_ratio)),
        "maximum_face_area_ratio": float(np.max(area_ratio)),
        "maximum_path_angle_deg": float(profile.maximum_path_angle_deg),
        "lateral_offset_range_m": float(np.ptp(profile.lateral_offset_m)),
        "retained_face_count": int(np.count_nonzero(valid_faces)),
        "source_valid_face_count": int(
            np.count_nonzero(source_mesh.mesh.valid_faces)
        ),
    }
    if diagnostics["minimum_face_normal_cosine"] <= 0.0:
        raise ValueError("co-warp folds at least one terrain face")
    target = CanonicalTerrainMesh(
        vertices_local=np.asarray(warped_vertices, dtype=np.float32),
        faces=np.asarray(source_mesh.mesh.faces, dtype=np.int32),
        valid_faces=np.asarray(valid_faces, dtype=bool),
        source_asset_sha256="0" * 64,
    )
    return TerrainMeshIndex(target, _identity_transform()), diagnostics


def write_terrain_usd(mesh: TerrainMeshIndex, output: Path) -> Path:
    """Persist a world-coordinate co-warped mesh for rendering/collection."""

    from pxr import Gf, Usd, UsdGeom

    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    faces = np.asarray(mesh.mesh.faces[mesh.mesh.valid_faces], dtype=np.int32)
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    usd_mesh = UsdGeom.Mesh.Define(stage, "/Terrain")
    usd_mesh.CreatePointsAttr(
        [Gf.Vec3f(*map(float, point)) for point in mesh.vertices_world]
    )
    usd_mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    usd_mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.SetDefaultPrim(usd_mesh.GetPrim())
    stage.GetRootLayer().Save()
    return destination


def _mechanical_acceptance(
    diagnostics: object,
    *,
    maximum_joint_correction_rad: float,
    maximum_stance_error_m: float,
    maximum_swing_error_m: float,
    maximum_stance_drift_m: float,
) -> bool:
    return bool(
        diagnostics.maximum_joint_correction_rad
        <= float(maximum_joint_correction_rad) + 1.0e-6
        and diagnostics.maximum_stance_sole_target_error_m
        <= float(maximum_stance_error_m)
        and diagnostics.maximum_swing_sole_target_error_m
        <= float(maximum_swing_error_m)
        and diagnostics.maximum_stance_run_drift_m
        <= float(maximum_stance_drift_m)
        and diagnostics.minimum_stance_support_point_count >= 2
        and diagnostics.maximum_root_translation_step_m <= 0.045
        and diagnostics.maximum_root_rotation_step_rad <= 0.100
        and diagnostics.maximum_joint_step_rad <= 0.20
        and diagnostics.maximum_root_acceleration_m_s2 <= 30.0
    )


def build_clip(
    *,
    archive_path: Path,
    model_path: Path,
    clip_index: int,
    output_root: Path,
    modes: Sequence[str] = DEFAULT_MODES,
    maximum_amplitude_m: float = 0.65,
    maximum_joint_correction_rad: float = 0.45,
    maximum_stance_error_m: float = 0.006,
    maximum_swing_error_m: float = 0.060,
    maximum_stance_drift_m: float = 0.010,
    render: bool = False,
    render_frame_stride: int = 2,
) -> dict[str, object]:
    """Build and exact-audit one paired directional rough-terrain family."""

    archive_path = archive_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    archive = zarr.open_group(str(archive_path), mode="r")
    index = int(clip_index)
    if not 0 <= index < len(archive["clip_names"]):
        raise ValueError(f"clip index {index} is outside the archive")
    clip_name = str(archive["clip_names"][index])
    clip_root = output_root / f"clip_{index:03d}_{clip_name}"
    source_root = clip_root / "source"
    source_report = evaluate(
        archive_path=archive_path,
        model_path=model_path,
        target_clip_index=index,
        output_dir=source_root,
        render=False,
        repair_clearance=True,
    )
    summary: dict[str, object] = {
        "schema": "c490-cowarped-directional-clip/v1",
        "clip_index": index,
        "clip_name": clip_name,
        "clip_family": str(archive["clip_family"][index]),
        "clip_traversal": str(archive["clip_traversal"][index]),
        "source_report": source_report,
        "attempted": 0,
        "accepted": 0,
        "reports": [],
    }
    if source_report["status"] != "accepted":
        summary["status"] = "source_rejected"
        clip_root.mkdir(parents=True, exist_ok=True)
        (clip_root / "aggregate.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return summary

    source = load_stitched_motion_npz(source_root / "motion.npz")
    source_mesh = _archive_terrain_index(archive, index)
    direction = np.asarray(
        (
            math.cos(float(archive["travel_yaw_rad"][index])),
            math.sin(float(archive["travel_yaw_rad"][index])),
        ),
        dtype=np.float64,
    )
    progress, _ = _path_coordinates(
        source.root_position_world,
        origin_xy=np.asarray(source.root_position_world[0, :2]),
        direction_xy=direction,
    )
    progress_min = float(np.min(progress))
    progress_range = float(np.ptp(progress))
    if progress_range < 0.75:
        raise ValueError("source route is too short for co-warp steering")
    active_start = progress_min + 0.10 * progress_range
    active_stop = progress_min + 0.90 * progress_range
    if active_stop - active_start < 0.75:
        active_start = progress_min
        active_stop = progress_min + progress_range

    joint_names = tuple(str(value) for value in archive["joint_names"][:])
    adapter = _G1FootfallAdapter(
        model_path,
        joint_names,
        maximum_joint_correction_rad=float(maximum_joint_correction_rad),
    )
    reports: list[dict[str, object]] = []
    for mode in tuple(modes):
        summary["attempted"] = int(summary["attempted"]) + 1
        destination = clip_root / str(mode)
        destination.mkdir(parents=True, exist_ok=True)
        report: dict[str, object] = {
            "clip_index": index,
            "clip_name": clip_name,
            "mode": str(mode),
            "source_motion": str((source_root / "motion.npz").resolve()),
            "active_route_interval_m": [active_start, active_stop],
            "status": "rejected",
        }
        try:
            target_mesh, mesh_diagnostics = cowarp_terrain_mesh(
                source_mesh,
                origin_xy=np.asarray(source.root_position_world[0, :2]),
                direction_xy=direction,
                active_start_m=active_start,
                active_stop_m=active_stop,
                mode=str(mode),
                maximum_amplitude_m=float(maximum_amplitude_m),
            )
            terrain_usd = write_terrain_usd(
                target_mesh, destination / "terrain.usda"
            )
            motion, diagnostics, extras = warp_motion(
                source,
                adapter=adapter,
                source_mesh=source_mesh,
                target_mesh=target_mesh,
                target_route=None,
                direction_xy=direction,
                active_start_m=active_start,
                active_stop_m=active_stop,
                mode=str(mode),
                maximum_amplitude_m=float(maximum_amplitude_m),
                # Swing feet follow the pelvis-local transform so a leading
                # foot does not enter the curve before the body can reach it.
                # Complete stance runs are then locked through exact paired
                # face correspondence inside ``warp_motion``.
                spatial_sole_mapping=False,
            )
            report["terrain_usd"] = str(terrain_usd)
            report["mesh_warp"] = mesh_diagnostics
            report["warp"] = asdict(diagnostics)
            realized = bool(
                (
                    diagnostics.lateral_offset_range_m >= 0.05
                    and diagnostics.maximum_path_angle_deg >= 7.0
                )
                or diagnostics.maximum_facing_offset_deg >= 7.0
            )
            mechanical = realized and _mechanical_acceptance(
                diagnostics,
                maximum_joint_correction_rad=maximum_joint_correction_rad,
                maximum_stance_error_m=maximum_stance_error_m,
                maximum_swing_error_m=maximum_swing_error_m,
                maximum_stance_drift_m=maximum_stance_drift_m,
            )
            report["realized_command_accepted"] = realized
            report["mechanical_accepted"] = mechanical
            if mechanical:
                collision = audit_stair_motion_collisions(
                    motion,
                    model_path=model_path,
                    target_mesh=target_mesh,
                    joint_names=joint_names,
                    maximum_foot_penetration_m=0.005,
                    maximum_forbidden_body_penetration_m=0.0,
                )
                report["pre_clearance_collision_audit"] = collision.to_dict()
                clearance_lift = 0.0
                if (
                    not collision.accepted
                    and collision.maximum_forbidden_body_penetration_m == 0.0
                    and collision.maximum_foot_penetration_m <= 0.020
                ):
                    original = motion
                    motion, _repair, clearance_lift = _repair_exact_mesh_clearance(
                        motion,
                        collision,
                        archive_path=archive_path,
                        target_clip_index=index,
                        target_mesh=target_mesh,
                        model_path=model_path,
                        maximum_total_lift_m=0.018,
                    )
                    extras["clearance_root_lift_m"] = np.asarray(
                        motion.root_position_world[:, 2]
                        - original.root_position_world[:, 2],
                        dtype=np.float32,
                    )
                    collision = audit_stair_motion_collisions(
                        motion,
                        model_path=model_path,
                        target_mesh=target_mesh,
                        joint_names=joint_names,
                        maximum_foot_penetration_m=0.005,
                        maximum_forbidden_body_penetration_m=0.0,
                    )
                report["clearance_repair_maximum_m"] = float(clearance_lift)
                report["collision_audit"] = collision.to_dict()
                report["status"] = "accepted" if collision.accepted else "rejected"
                _save_motion(
                    destination / "motion.npz",
                    motion,
                    mode=str(mode),
                    extras=extras,
                )
                (destination / "collision_audit.json").write_text(
                    json.dumps(collision.to_dict(), indent=2, sort_keys=True)
                    + "\n"
                )
                if report["status"] == "accepted":
                    summary["accepted"] = int(summary["accepted"]) + 1
                    if render:
                        media = render_stitched_motion(
                            motion,
                            model_path=model_path,
                            target_mesh=target_mesh,
                            joint_names=joint_names,
                            output_path=destination / "g1_kinematic_25fps.mp4",
                            width=640,
                            height=360,
                            frame_stride=max(1, int(render_frame_stride)),
                            camera_follow_root=True,
                        )
                        report["video"] = str(media["video"])
            else:
                report["collision_audit_skipped_reason"] = (
                    "mechanical_rejection"
                )
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        (destination / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        reports.append(report)
        print(
            f"C490_COWARP clip={index} mode={mode} status={report['status']}",
            flush=True,
        )
    summary["reports"] = reports
    summary["status"] = (
        "pending_dense_visual_review"
        if int(summary["accepted"])
        else "no_admission"
    )
    (clip_root / "aggregate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--clip-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", action="append", default=[])
    parser.add_argument("--maximum-amplitude-m", type=float, default=0.65)
    parser.add_argument("--maximum-joint-correction-rad", type=float, default=0.45)
    parser.add_argument("--maximum-stance-error-m", type=float, default=0.006)
    parser.add_argument("--maximum-swing-error-m", type=float, default=0.060)
    parser.add_argument("--maximum-stance-drift-m", type=float, default=0.010)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-frame-stride", type=int, default=2)
    arguments = parser.parse_args(argv)
    summary = build_clip(
        archive_path=arguments.archive,
        model_path=arguments.model,
        clip_index=arguments.clip_index,
        output_root=arguments.output_root,
        modes=tuple(arguments.mode or DEFAULT_MODES),
        maximum_amplitude_m=arguments.maximum_amplitude_m,
        maximum_joint_correction_rad=arguments.maximum_joint_correction_rad,
        maximum_stance_error_m=arguments.maximum_stance_error_m,
        maximum_swing_error_m=arguments.maximum_swing_error_m,
        maximum_stance_drift_m=arguments.maximum_stance_drift_m,
        render=arguments.render,
        render_frame_stride=arguments.render_frame_stride,
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "reports"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
