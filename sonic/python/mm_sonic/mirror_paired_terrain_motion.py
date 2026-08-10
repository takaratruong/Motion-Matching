"""Mirror an accepted motion and its authored terrain as one exact pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import zarr

from .build_c490_cowarped_directional import write_terrain_usd
from .mirror_stairs500_omnidirectional_pilots import (
    _mirror_quaternion,
    _reflect_points,
    _reflect_vectors,
    _swap_left_right,
)
from .offline_corpus import JOINT_MIRROR_PERMUTATION, JOINT_MIRROR_SIGNS
from .render_stitched_motion import (
    load_stitched_motion_npz,
    render_stitched_motion,
)
from .terrain_oracle.canonical import CanonicalTerrainMesh
from .terrain_oracle.math3d import RigidTransform
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.source_grail import _load_usd_mesh
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


def _identity_transform() -> RigidTransform:
    return RigidTransform(
        np.zeros(3, dtype=np.float32),
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
    )


def _stair_axis(
    archive: object,
    clip_index: int,
    terrain: TerrainMeshIndex,
) -> tuple[float, np.ndarray, float]:
    """Recover the stair-normal reflection axis, not the oblique travel axis."""

    travel_yaw = float(archive["travel_yaw_rad"][clip_index])
    approach_delta = math.radians(
        float(archive["clip_approach_heading_delta_deg"][clip_index])
    )
    axis_yaw = travel_yaw - approach_delta
    lateral = np.asarray(
        (-math.sin(axis_yaw), math.cos(axis_yaw)), dtype=np.float64
    )
    faces = np.asarray(terrain.mesh.faces[terrain.mesh.valid_faces], dtype=np.int64)
    used = np.unique(faces.reshape(-1))
    vertices = np.asarray(terrain.vertices_world[used], dtype=np.float64)
    minimum_height = float(np.min(vertices[:, 2]))
    elevated = vertices[:, 2] > minimum_height + 0.03
    centre_vertices = vertices[elevated] if np.any(elevated) else vertices
    lateral_coordinate = centre_vertices[:, :2] @ lateral
    centre = 0.5 * (
        float(np.min(lateral_coordinate)) + float(np.max(lateral_coordinate))
    )
    return axis_yaw, lateral, centre


def _mirror_foot_points(
    values: np.ndarray,
    *,
    lateral: np.ndarray,
    centre: float,
) -> np.ndarray:
    mirrored = _reflect_points(values, lateral=lateral, centre=centre)
    if mirrored.ndim >= 3 and mirrored.shape[1] == 2:
        mirrored = mirrored[:, (1, 0)]
    if mirrored.ndim == 4 and mirrored.shape[2] == 4:
        mirrored = mirrored[:, :, (1, 0, 3, 2)]
    return mirrored


def build(
    *,
    source_report: Path,
    archive_path: Path,
    model_path: Path,
    output: Path,
    render: bool = False,
) -> dict[str, object]:
    report_path = source_report.expanduser().resolve()
    source = json.loads(report_path.read_text())
    if source.get("status") != "accepted":
        raise ValueError("paired mirroring requires an accepted source report")
    motion_path = report_path.parent / "motion.npz"
    terrain_path = Path(str(source["terrain_usd"])).expanduser().resolve()
    if not motion_path.is_file() or not terrain_path.is_file():
        raise FileNotFoundError("accepted paired source assets are incomplete")

    archive_path = archive_path.expanduser().resolve()
    archive = zarr.open_group(str(archive_path), mode="r")
    clip_index = int(source["clip_index"])
    joint_names = tuple(str(value) for value in archive["joint_names"][:])
    mesh = _load_usd_mesh(
        terrain_path,
        source_asset_sha256=hashlib.sha256(terrain_path.read_bytes()).hexdigest(),
    )
    terrain = TerrainMeshIndex(mesh, _identity_transform())
    axis_yaw, lateral, centre = _stair_axis(archive, clip_index, terrain)

    with np.load(motion_path, allow_pickle=False) as bundle:
        arrays = {name: np.asarray(bundle[name]).copy() for name in bundle.files}

    arrays["root_position_world"] = _reflect_points(
        arrays["root_position_world"], lateral=lateral, centre=centre
    ).astype(np.float32)
    arrays["root_quaternion_world_wxyz"] = _mirror_quaternion(
        arrays["root_quaternion_world_wxyz"], axis_yaw
    ).astype(np.float32)
    arrays["joint_position"] = (
        arrays["joint_position"][:, JOINT_MIRROR_PERMUTATION]
        * JOINT_MIRROR_SIGNS
    ).astype(np.float32)

    point_fields = (
        "intended_root_position_world",
        "target_sole_center_world",
        "target_sole_points_world",
        "nominal_sole_center_world",
        "nominal_sole_points_world",
        "pre_conformance_target_sole_points_world",
        "adapted_sole_center_world",
        "source_anchored_sole_points_world",
    )
    for name in point_fields:
        if name in arrays:
            arrays[name] = _mirror_foot_points(
                arrays[name], lateral=lateral, centre=centre
            ).astype(np.float32)

    vector_fields = (
        "terrain_root_anchor_shift_world",
        "per_frame_swing_route_adjustment",
    )
    for name in vector_fields:
        if name not in arrays:
            continue
        value = _reflect_vectors(arrays[name], lateral=lateral)
        if value.ndim >= 3 and value.shape[1] == 2:
            value = value[:, (1, 0)]
        arrays[name] = value.astype(np.float32)

    foot_fields = (
        "authored_stance_mask",
        "target_stance_support_point_count",
        "per_frame_sole_target_error_by_foot_m",
        "bracketed_swing_mask",
        "planned_partial_rigid_support_mask",
        "planned_support_contact_point_count",
    )
    for name in foot_fields:
        if name in arrays:
            arrays[name] = arrays[name][:, (1, 0)]

    arrays["command_velocity_world_xy"] = _reflect_vectors(
        arrays["command_velocity_world_xy"], lateral=lateral
    ).astype(np.float32)
    facing = 2.0 * axis_yaw - np.asarray(
        arrays["command_facing_yaw_world_rad"], dtype=np.float64
    )
    facing = np.unwrap(np.arctan2(np.sin(facing), np.cos(facing)))
    arrays["command_facing_yaw_world_rad"] = facing.astype(np.float32)
    velocity = np.asarray(arrays["command_velocity_world_xy"], dtype=np.float64)
    cosine, sine = np.cos(facing), np.sin(facing)
    arrays["command_velocity_robot_local_xy"] = np.stack(
        (
            cosine * velocity[:, 0] + sine * velocity[:, 1],
            -sine * velocity[:, 0] + cosine * velocity[:, 1],
        ),
        axis=1,
    ).astype(np.float32)
    for name in (
        "path_lateral_offset_m",
        "path_yaw_offset_rad",
        "pose_yaw_offset_rad",
        "foot_yaw_offset_rad",
        "facing_yaw_offset_rad",
    ):
        if name in arrays:
            arrays[name] *= -1.0
    mode = _swap_left_right(str(np.asarray(arrays["maneuver_mode"])))
    arrays["maneuver_mode"] = np.asarray(mode, dtype=np.str_)
    arrays["symmetry_mirrored"] = np.asarray(True, dtype=np.bool_)
    arrays["symmetry_pair_id"] = np.asarray(str(report_path), dtype=np.str_)

    mirrored_vertices = _reflect_points(
        terrain.vertices_world, lateral=lateral, centre=centre
    )
    mirrored_mesh = CanonicalTerrainMesh(
        vertices_local=np.asarray(mirrored_vertices, dtype=np.float32),
        faces=np.asarray(terrain.mesh.faces[:, (0, 2, 1)], dtype=np.int32),
        valid_faces=np.asarray(terrain.mesh.valid_faces, dtype=bool),
        source_asset_sha256="0" * 64,
    )
    mirrored_terrain = TerrainMeshIndex(mirrored_mesh, _identity_transform())

    destination = output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    mirrored_motion_path = destination / "motion.npz"
    np.savez_compressed(mirrored_motion_path, **arrays)
    mirrored_terrain_path = write_terrain_usd(
        mirrored_terrain, destination / "terrain.usda"
    )
    motion = load_stitched_motion_npz(mirrored_motion_path)
    collision = audit_stair_motion_collisions(
        motion,
        model_path=model_path.expanduser().resolve(),
        target_mesh=mirrored_terrain,
        joint_names=joint_names,
        maximum_foot_penetration_m=0.005,
        maximum_forbidden_body_penetration_m=0.0,
    )
    # Reflection should preserve contact analytically, but independently
    # measure it on the serialized mirrored pair.  This catches incomplete
    # left/right field swaps or support-probe ordering mistakes which a body
    # collision audit alone cannot see.
    from .build_bones_side_on_stair_pilots import _audit_stance_contact_support

    adapter = _G1FootfallAdapter(
        model_path.expanduser().resolve(),
        joint_names,
        maximum_joint_correction_rad=0.45,
    )
    contact = _audit_stance_contact_support(
        motion,
        arrays,
        adapter=adapter,
        target_mesh=mirrored_terrain,
        ground_fallback_height_m=float(
            np.min(mirrored_terrain.vertices_world[:, 2])
        ),
    )
    stance = np.asarray(arrays["authored_stance_mask"], dtype=bool)
    support = np.asarray(
        arrays["target_stance_support_point_count"], dtype=np.int64
    )
    spans = [
        int(
            np.count_nonzero(
                stance[:, foot]
                & np.concatenate((np.ones(1, dtype=bool), ~stance[:-1, foot]))
            )
        )
        for foot in range(2)
    ]
    supported_frames = [
        int(np.count_nonzero(stance[:, foot] & (support[:, foot] >= 2)))
        for foot in range(2)
    ]
    balanced = bool(
        min(spans) >= 2
        and min(supported_frames) >= 8
        and np.all(support[stance] >= 2)
    )
    mirrored_report = {
        "schema": "paired-terrain-motion-mirror/v1",
        "status": (
            "accepted"
            if collision.accepted and balanced and bool(contact["accepted"])
            else "rejected"
        ),
        "source_report": str(report_path),
        "source_motion": str(motion_path),
        "source_terrain_usd": str(terrain_path),
        "clip_index": clip_index,
        "mode": mode,
        "reflection_axis_yaw_rad": axis_yaw,
        "reflection_lateral_coordinate_m": centre,
        "motion": str(mirrored_motion_path),
        "terrain_usd": str(mirrored_terrain_path),
        "collision_audit": collision.to_dict(),
        "stance_contact_audit": contact,
        "stance_spans_per_foot": spans,
        "supported_stance_frames_per_foot": supported_frames,
        "balanced_two_foot_support_accepted": balanced,
    }
    if render and mirrored_report["status"] == "accepted":
        media = render_stitched_motion(
            motion,
            model_path=model_path.expanduser().resolve(),
            target_mesh=mirrored_terrain,
            joint_names=joint_names,
            output_path=destination / "g1_kinematic_25fps.mp4",
            frame_stride=2,
            camera_follow_root=True,
        )
        mirrored_report["video"] = str(media["video"])
    (destination / "report.json").write_text(
        json.dumps(mirrored_report, indent=2, sort_keys=True) + "\n"
    )
    return mirrored_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "/move/data/terrain-aware/motion-matching/"
            "grail-stairs500-clean-50hz-v1.zarr"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/move/u/justingu/Projects/TWIST2/assets/g1/"
            "g1_29dof_rev_1_0.xml"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    arguments = parser.parse_args(argv)
    result = build(
        source_report=arguments.source_report,
        archive_path=arguments.archive,
        model_path=arguments.model,
        output=arguments.output,
        render=arguments.render,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
