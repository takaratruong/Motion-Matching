"""Mirror accepted stair-direction pilots across the exact stair centreline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import zarr

from .offline_corpus import JOINT_MIRROR_PERMUTATION, JOINT_MIRROR_SIGNS
from .render_stitched_motion import load_stitched_motion_npz
from .terrain_oracle.stair_geometry_warp import _archive_terrain_index
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)


DEFAULT_ARCHIVE = Path(
    "/move/data/terrain-aware/motion-matching/"
    "grail-stairs500-clean-50hz-v1.zarr"
)
DEFAULT_MODEL = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/"
    "g1_29dof_rev_1_0.xml"
)


def _multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(np.asarray(left), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _yaw_quaternion(yaw: float, count: int) -> np.ndarray:
    value = np.asarray(
        (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)),
        dtype=np.float64,
    )
    return np.broadcast_to(value, (int(count), 4))


def _mirror_quaternion(quaternion: np.ndarray, axis_yaw: float) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    local = _multiply_wxyz(_yaw_quaternion(-axis_yaw, len(value)), value)
    local *= np.asarray((1.0, -1.0, 1.0, -1.0), dtype=np.float64)
    result = _multiply_wxyz(_yaw_quaternion(axis_yaw, len(value)), local)
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result


def _swap_left_right(value: str) -> str:
    return (
        str(value)
        .replace("left", "__temporary_left__")
        .replace("right", "left")
        .replace("__temporary_left__", "right")
    )


def _stair_centreline(
    archive: object, clip_index: int
) -> tuple[np.ndarray, np.ndarray, float]:
    yaw = float(archive["travel_yaw_rad"][clip_index])
    direction = np.asarray((math.cos(yaw), math.sin(yaw)), dtype=np.float64)
    lateral = np.asarray((-direction[1], direction[0]))
    mesh = _archive_terrain_index(archive, clip_index)
    vertices = np.asarray(mesh.vertices_world, dtype=np.float64)
    elevated = vertices[:, 2] > float(np.min(vertices[:, 2])) + 0.03
    values = vertices[elevated, :2] @ lateral
    if not len(values):
        raise ValueError("stair mesh has no elevated centreline evidence")
    centre = 0.5 * (float(np.min(values)) + float(np.max(values)))
    return direction, lateral, centre


def _reflect_points(
    values: np.ndarray, *, lateral: np.ndarray, centre: float
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    distance = result[..., :2] @ lateral - float(centre)
    result[..., :2] -= 2.0 * distance[..., None] * lateral
    return result


def _reflect_vectors(values: np.ndarray, *, lateral: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result[..., :2] -= (
        2.0 * (result[..., :2] @ lateral)[..., None] * lateral
    )
    return result


def _realized_directional_motion(
    report_path: Path, report: dict[str, object], archive: object
) -> bool:
    """Reject constant-offset no-ops before they can be symmetry-expanded."""

    with np.load(report_path.parent / "motion.npz", allow_pickle=False) as data:
        root = np.asarray(data["root_position_world"], dtype=np.float64)
        lateral_offset = np.asarray(
            data["path_lateral_offset_m"], dtype=np.float64
        )
        path_yaw = np.asarray(data["path_yaw_offset_rad"], dtype=np.float64)
        facing_offset = np.asarray(
            data["facing_yaw_offset_rad"], dtype=np.float64
        )
    clip_index = int(report["clip_index"])
    yaw = float(archive["travel_yaw_rad"][clip_index])
    direction = np.asarray((math.cos(yaw), math.sin(yaw)), dtype=np.float64)
    progress = (root[:, :2] - root[0, :2]) @ direction
    if float(np.ptp(progress)) < 0.50 or float(np.ptp(root[:, 2])) < 0.06:
        return False
    mode = str(report["mode"])
    if mode == "straight":
        return str(report.get("source_kind")) == "temporal_reverse"
    realized_path = bool(
        float(np.ptp(lateral_offset)) >= 0.02
        and float(np.max(np.abs(path_yaw))) >= math.radians(5.0)
    )
    realized_facing = bool(
        float(np.max(np.abs(facing_offset))) >= math.radians(5.0)
    )
    return realized_path or realized_facing


def mirror_one(
    report_path: Path,
    *,
    archive: object,
    archive_path: Path,
    model_path: Path,
    output: Path,
) -> dict[str, object]:
    report = json.loads(report_path.read_text())
    if report.get("status") != "accepted":
        raise ValueError("only accepted directional pilots may be mirrored")
    source_path = report_path.parent / "motion.npz"
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    clip_index = int(report["clip_index"])
    direction, lateral, centre = _stair_centreline(archive, clip_index)
    axis_yaw = math.atan2(float(direction[1]), float(direction[0]))
    arrays["root_position_world"] = _reflect_points(
        arrays["root_position_world"], lateral=lateral, centre=centre
    ).astype(np.float32)
    if "intended_root_position_world" in arrays:
        arrays["intended_root_position_world"] = _reflect_points(
            arrays["intended_root_position_world"],
            lateral=lateral,
            centre=centre,
        ).astype(np.float32)
    arrays["root_quaternion_world_wxyz"] = _mirror_quaternion(
        arrays["root_quaternion_world_wxyz"], axis_yaw
    ).astype(np.float32)
    arrays["joint_position"] = (
        arrays["joint_position"][:, JOINT_MIRROR_PERMUTATION]
        * JOINT_MIRROR_SIGNS
    ).astype(np.float32)
    arrays["command_velocity_world_xy"] = _reflect_vectors(
        arrays["command_velocity_world_xy"], lateral=lateral
    ).astype(np.float32)
    facing = 2.0 * axis_yaw - np.asarray(
        arrays["command_facing_yaw_world_rad"], dtype=np.float64
    )
    arrays["command_facing_yaw_world_rad"] = np.unwrap(
        np.arctan2(np.sin(facing), np.cos(facing))
    ).astype(np.float32)
    cosine = np.cos(arrays["command_facing_yaw_world_rad"])
    sine = np.sin(arrays["command_facing_yaw_world_rad"])
    velocity = arrays["command_velocity_world_xy"]
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
    for name in (
        "authored_stance_mask",
        "target_stance_support_point_count",
        "per_frame_sole_target_error_by_foot_m",
    ):
        if name in arrays:
            arrays[name] = arrays[name][:, (1, 0)]
    mode = _swap_left_right(str(np.asarray(arrays["maneuver_mode"])))
    arrays["maneuver_mode"] = np.asarray(mode, dtype=np.str_)
    arrays["symmetry_pair_id"] = np.asarray(str(report["label"]), dtype=np.str_)
    arrays["symmetry_mirrored"] = np.asarray(True, dtype=np.bool_)

    label = _swap_left_right(str(report["label"])) + "_mirror"
    destination = output / label
    destination.mkdir(parents=True, exist_ok=True)
    motion_path = destination / "motion.npz"
    np.savez_compressed(motion_path, **arrays)
    motion = load_stitched_motion_npz(motion_path)
    target_mesh = _archive_terrain_index(archive, clip_index)
    collision = audit_stair_motion_collisions(
        motion,
        model_path=model_path,
        target_mesh=target_mesh,
        joint_names=tuple(str(value) for value in archive["joint_names"][:]),
        maximum_foot_penetration_m=0.005,
        maximum_forbidden_body_penetration_m=0.0,
    )
    mirrored_report = dict(report)
    mirrored_report.update(
        {
            "label": label,
            "mode": mode,
            "mirror_of": str(report_path.parent),
            "motion": str(motion_path),
            "collision_audit": collision.to_dict(),
            "status": "accepted" if collision.accepted else "rejected",
        }
    )
    (destination / "collision_audit.json").write_text(
        json.dumps(collision.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (destination / "report.json").write_text(
        json.dumps(mirrored_report, indent=2, sort_keys=True) + "\n"
    )
    return mirrored_report


def build(
    root: Path,
    *,
    output: Path,
    archive_path: Path,
    model_path: Path,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, object]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("mirror shard index must be within shard count")
    archive_path = archive_path.expanduser().resolve()
    archive = zarr.open_group(str(archive_path), mode="r")
    admitted_reports: list[Path] = []
    skipped_noop = 0
    for path in sorted(root.expanduser().resolve().rglob("report.json")):
        report = json.loads(path.read_text())
        if report.get("status") != "accepted":
            continue
        if not _realized_directional_motion(path, report, archive):
            skipped_noop += 1
            continue
        admitted_reports.append(path)
    reports = admitted_reports[shard_index::shard_count]
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, report_path in enumerate(reports):
        result = mirror_one(
            report_path,
            archive=archive,
            archive_path=archive_path,
            model_path=model_path.expanduser().resolve(),
            output=output,
        )
        results.append(result)
        print(
            f"STAIRS500_MIRROR {index + 1}/{len(reports)} "
            f"label={result['label']} status={result['status']}",
            flush=True,
        )
    summary = {
        "schema": "stairs500-omnidirectional-mirrors/v1",
        "source_root": str(root.expanduser().resolve()),
        "source_admitted_count": len(admitted_reports),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "attempted": len(results),
        "accepted": sum(row["status"] == "accepted" for row in results),
        "rejected": sum(row["status"] != "accepted" for row in results),
        "skipped_noop": skipped_noop,
        "reports": results,
    }
    summary_name = (
        "aggregate.json"
        if shard_count == 1
        else f"aggregate_shard_{shard_index:02d}_of_{shard_count:02d}.json"
    )
    (output / summary_name).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    arguments = parser.parse_args()
    summary = build(
        arguments.root,
        output=arguments.output,
        archive_path=arguments.archive,
        model_path=arguments.model,
        shard_index=arguments.shard_index,
        shard_count=arguments.shard_count,
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "reports"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
