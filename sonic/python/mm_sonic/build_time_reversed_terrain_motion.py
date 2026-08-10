"""Create an exact-contact time-reversed derivative of an accepted terrain clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr

from .build_bones_side_on_stair_pilots import (
    MAXIMUM_ARM_EXCURSION_RAD,
    MAXIMUM_ARM_P99_ACCELERATION_RAD_S2,
    MAXIMUM_ARM_RMS_VELOCITY_RAD_S,
    MAXIMUM_ARM_VELOCITY_RAD_S,
    MAXIMUM_WRIST_EXCURSION_RAD,
    MAXIMUM_WRIST_P99_ACCELERATION_RAD_S2,
    MAXIMUM_WRIST_RMS_VELOCITY_RAD_S,
    MAXIMUM_WRIST_VELOCITY_RAD_S,
    _audit_stance_contact_support,
    _retimed_motion_metrics,
    _upper_limb_motion_metrics,
)
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .build_stairs500_omnidirectional_pilots import (
    DEFAULT_ARCHIVE,
    _save_motion,
    _temporally_reverse_framewise_extras,
    _temporally_reverse_motion,
)
from .render_stitched_motion import load_stitched_motion_npz
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.stair_geometry_warp import (
    _archive_terrain_index,
    motion_conditioned_stair_support_route,
)
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.source_grail import _load_usd_mesh
from .terrain_oracle.terrain_mesh import TerrainMeshIndex, _identity_transform


_CORE_BUNDLE_KEYS = frozenset(
    {
        "fps",
        "root_position_world",
        "root_quaternion_world_wxyz",
        "joint_position",
        "seam_indices",
        "source_archive_clip_index",
        "source_frame",
        "source_clip_id",
        "maneuver_mode",
        "command_velocity_world_xy",
        "command_velocity_robot_local_xy",
        "command_facing_yaw_world_rad",
        "command_stop",
    }
)


def _load_framewise_extras(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as arrays:
        return {
            name: np.asarray(arrays[name])
            for name in arrays.files
            if name not in _CORE_BUNDLE_KEYS
        }


def _upper_limb_accepted(metrics: dict[str, float]) -> bool:
    return bool(
        metrics["maximum_arm_excursion_rad"] <= MAXIMUM_ARM_EXCURSION_RAD
        and metrics["maximum_wrist_excursion_rad"]
        <= MAXIMUM_WRIST_EXCURSION_RAD
        and metrics["maximum_arm_velocity_rad_s"] <= MAXIMUM_ARM_VELOCITY_RAD_S
        and metrics["maximum_wrist_velocity_rad_s"]
        <= MAXIMUM_WRIST_VELOCITY_RAD_S
        and metrics["maximum_joint_rms_arm_velocity_rad_s"]
        <= MAXIMUM_ARM_RMS_VELOCITY_RAD_S
        and metrics["maximum_joint_rms_wrist_velocity_rad_s"]
        <= MAXIMUM_WRIST_RMS_VELOCITY_RAD_S
        and metrics["p99_arm_acceleration_rad_s2"]
        <= MAXIMUM_ARM_P99_ACCELERATION_RAD_S2
        and metrics["p99_wrist_acceleration_rad_s2"]
        <= MAXIMUM_WRIST_P99_ACCELERATION_RAD_S2
    )


def build(arguments: argparse.Namespace) -> dict[str, object]:
    source_path = arguments.source.expanduser().resolve()
    destination = arguments.output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source_report = None
    if arguments.source_report is not None:
        source_report = json.loads(
            arguments.source_report.expanduser().resolve().read_text()
        )
    registered_source_qualification = bool(
        source_report is not None
        and source_report.get("registered_source_qualification")
    )
    row: dict[str, object] = {
        "schema": "exact-terrain-time-reverse/v1",
        "construction": "exact_pose_time_reverse",
        "source_motion": str(source_path),
        "target_clip_index": int(arguments.clip_index),
        "registered_source_qualification": registered_source_qualification,
        "status": "rejected",
    }
    try:
        archive = zarr.open_group(
            str(arguments.archive.expanduser().resolve()), mode="r"
        )
        joint_names = tuple(str(value) for value in archive["joint_names"][:])
        paired_terrain_path = arguments.terrain_usd
        if (
            paired_terrain_path is None
            and source_report is not None
            and source_report.get("terrain_usd") is not None
        ):
            paired_terrain_path = Path(str(source_report["terrain_usd"]))
        if paired_terrain_path is None:
            target_mesh = _archive_terrain_index(
                archive, int(arguments.clip_index)
            )
            route = motion_conditioned_stair_support_route(
                archive, int(arguments.clip_index)
            )
            ground_height = float(min(level.height_m for level in route.levels))
            terrain_usd = str(archive["terrain_usd_path"][arguments.clip_index])
        else:
            terrain_path = paired_terrain_path.expanduser().resolve()
            terrain_mesh = _load_usd_mesh(
                terrain_path, source_asset_sha256="0" * 64
            )
            target_mesh = TerrainMeshIndex(terrain_mesh, _identity_transform())
            ground_height = float(np.min(target_mesh.vertices_world[:, 2]))
            terrain_usd = str(terrain_path)
        row["terrain_usd"] = terrain_usd
        adapter = _G1FootfallAdapter(
            arguments.model.expanduser().resolve(),
            joint_names,
            maximum_joint_correction_rad=0.0,
        )
        source = load_stitched_motion_npz(source_path)
        source_extras = _load_framewise_extras(source_path)
        motion = _temporally_reverse_motion(source)
        extras = _temporally_reverse_framewise_extras(
            source_extras, frame_count=len(source.root_position_world)
        )
        collision = audit_stair_motion_collisions(
            motion,
            model_path=arguments.model.expanduser().resolve(),
            target_mesh=target_mesh,
            joint_names=joint_names,
            maximum_foot_penetration_m=0.005,
            maximum_forbidden_body_penetration_m=0.0,
        )
        row["collision_audit"] = collision.to_dict()
        if not collision.accepted:
            raise ValueError("time-reversed motion failed exact collision audit")
        mechanics = _retimed_motion_metrics(motion)
        row["motion_metrics"] = mechanics
        # Exact reversal cannot introduce a new step or acceleration.  Use
        # the same mechanics envelope as the natural-source admission path so
        # an already admitted pose sequence is not rejected solely because it
        # is enumerated in the opposite temporal order.
        maximum_root_translation_step_m = 0.045
        maximum_root_rotation_step_rad = 0.100
        maximum_root_acceleration_m_s2 = 32.0
        if (
            mechanics["maximum_joint_step_rad"] > 0.20
            or mechanics["maximum_root_translation_step_m"]
            > maximum_root_translation_step_m
            or mechanics["maximum_root_rotation_step_rad"]
            > maximum_root_rotation_step_rad
            or mechanics["maximum_root_acceleration_m_s2"]
            > maximum_root_acceleration_m_s2
        ):
            raise ValueError("time-reversed motion failed smoothness audit")
        upper_limb = _upper_limb_motion_metrics(
            motion.joint_position, joint_names, fps=motion.fps
        )
        row["upper_limb_metrics"] = upper_limb
        if not _upper_limb_accepted(upper_limb):
            raise ValueError("time-reversed motion failed upper-limb energy audit")
        contact = _audit_stance_contact_support(
            motion,
            extras,
            adapter=adapter,
            target_mesh=target_mesh,
            ground_fallback_height_m=ground_height,
            maximum_core_stance_probe_hover_m=(
                0.10 if registered_source_qualification else 0.012
            ),
            minimum_core_stance_support_point_count=(
                2 if registered_source_qualification else 3
            ),
        )
        row["stance_contact_audit"] = contact
        if not bool(contact["accepted"]):
            raise ValueError("time-reversed motion failed no-hover contact audit")
        extras["construction"] = np.asarray(
            "exact_pose_time_reverse", dtype=np.str_
        )
        output_motion = destination / "motion.npz"
        _save_motion(
            output_motion,
            motion,
            mode="exact_terrain_time_reverse",
            extras=extras,
        )
        row["motion"] = str(output_motion.resolve())
        row["frame_count"] = len(motion.root_position_world)
        row["duration_s"] = float(len(motion.root_position_world) / motion.fps)
        row["status"] = "pending_dense_visual_review"
    except (ValueError, RuntimeError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
    (destination / "report.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )
    return row


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--clip-index", type=int, required=True)
    parser.add_argument("--terrain-usd", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = build(arguments)
    print(
        json.dumps(
            {
                "status": result["status"],
                "motion": result.get("motion"),
                "report": str(
                    (arguments.output.expanduser().resolve() / "report.json")
                ),
                "error": result.get("error"),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "pending_dense_visual_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
