"""Publish an indexed bank of exact-audited global terrain traversals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .privileged_terrain_matcher import DEFAULT_STAIRS_ARCHIVE
from .stair_mesh_profile import measure_archive_stair_profile
from .terrain_oracle.reference_stitch import _G1FootfallAdapter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaw_wxyz(value: np.ndarray) -> float:
    w, x, y, z = (float(component) for component in value)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def build_index(
    *,
    registered_root: Path,
    archive_path: Path,
    output_path: Path,
    model_path: Path = DEFAULT_G1_MJCF,
    maximum_terminal_support_height_error_m: float = 0.08,
) -> dict[str, object]:
    import zarr

    archive = zarr.open_group(str(archive_path), mode="r")
    adapter = _G1FootfallAdapter(
        model_path,
        tuple(str(value) for value in archive["joint_names"][:]),
        maximum_joint_correction_rad=0.95,
        target_tolerance_m=2.5e-4,
        maximum_iterations=96,
        damping=0.006,
        posture_weight=2.0e-5,
    )
    rows: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for summary_path in sorted(registered_root.glob("target_*/summary.json")):
        summary = json.loads(summary_path.read_text())
        target = int(summary.get("target_clip_index", -1))
        motion_path = summary_path.parent / "motion.npz"
        if summary.get("status") != "accepted" or not motion_path.is_file():
            rejected.append(
                {
                    "target_clip_index": target,
                    "summary": str(summary_path.resolve()),
                    "reason": "registered traversal is not accepted or motion is missing",
                }
            )
            continue
        with np.load(motion_path, allow_pickle=False) as arrays:
            root = np.asarray(arrays["root_position_world"], dtype=np.float64)
            quaternion = np.asarray(
                arrays["root_quaternion_world_wxyz"], dtype=np.float64
            )
            joints = np.asarray(arrays["joint_position"], dtype=np.float64)
            fps = float(np.asarray(arrays["fps"]).item())
        if (
            target < 0
            or root.ndim != 2
            or root.shape[1:] != (3,)
            or quaternion.shape != (len(root), 4)
            or joints.shape != (len(root), 29)
            or len(root) < 3
        ):
            raise ValueError(f"invalid registered traversal: {motion_path}")
        physical_profile = measure_archive_stair_profile(archive, target)
        origin_z = float(archive["terrain_position_env"][target, 2])
        bottom_z = origin_z
        top_z = origin_z + float(physical_profile.height_m)
        traversal = str(archive["clip_traversal"][target])
        expected_terminal_heights = (
            (bottom_z, top_z)
            if traversal == "up"
            else (top_z, bottom_z)
        )
        observed_terminal_heights: list[float] = []
        for frame in (0, len(root) - 1):
            soles = adapter.sole_positions_for_pose(
                root_position=root[frame],
                root_quaternion_wxyz=quaternion[frame],
                joints=joints[frame],
            )
            observed_terminal_heights.append(
                float(
                    np.min(
                        [
                            np.mean(sole, axis=0)[2]
                            for sole in soles
                        ]
                    )
                )
            )
        terminal_errors = tuple(
            abs(observed - expected)
            for observed, expected in zip(
                observed_terminal_heights,
                expected_terminal_heights,
                strict=True,
            )
        )
        terminal_coverage = {
            "accepted": bool(
                max(terminal_errors)
                <= float(maximum_terminal_support_height_error_m)
            ),
            "maximum_support_height_error_m": float(max(terminal_errors)),
            "threshold_m": float(maximum_terminal_support_height_error_m),
            "observed_entry_support_height_m": observed_terminal_heights[0],
            "observed_exit_support_height_m": observed_terminal_heights[1],
            "expected_entry_surface_height_m": expected_terminal_heights[0],
            "expected_exit_surface_height_m": expected_terminal_heights[1],
        }
        if not terminal_coverage["accepted"]:
            rejected.append(
                {
                    "target_clip_index": target,
                    "summary": str(summary_path.resolve()),
                    "reason": (
                        "motion does not reach both physical terrain terminals"
                    ),
                    "physical_terminal_coverage": terminal_coverage,
                }
            )
            continue
        rows.append(
            {
                "target_clip_index": target,
                "label": str(archive["clip_names"][target]),
                "traversal": traversal,
                "nominal_geometry": {
                    "rise_m": float(archive["stair_rise_m"][target]),
                    "tread_m": float(archive["stair_tread_m"][target]),
                    "step_count": int(archive["stair_n_steps"][target]),
                },
                "physical_geometry": {
                    "rise_m": float(physical_profile.rise_m),
                    "tread_m": float(physical_profile.tread_m),
                    "level_count": int(physical_profile.level_count),
                    "height_m": float(physical_profile.height_m),
                    "run_m": float(physical_profile.run_m),
                    "tread_edges_m": list(physical_profile.tread_edges_m),
                    "tread_heights_m": list(
                        physical_profile.tread_heights_m
                    ),
                },
                "physical_terminal_coverage": terminal_coverage,
                "travel_yaw_rad": float(archive["travel_yaw_rad"][target]),
                "terrain_position_world_xyz": [
                    float(value)
                    for value in archive["terrain_position_env"][target]
                ],
                "terrain_rotation_world_wxyz": [
                    float(value)
                    for value in archive["terrain_rotation_env_wxyz"][target]
                ],
                "terrain_usd_path": str(archive["terrain_usd_path"][target]),
                "motion_path": str(motion_path.resolve()),
                "motion_sha256": _sha256(motion_path),
                "fps": fps,
                "frame_count": int(len(root)),
                "entry": {
                    "root_position_world_xyz": root[0].tolist(),
                    "root_yaw_world_rad": _yaw_wxyz(quaternion[0]),
                },
                "exit": {
                    "root_position_world_xyz": root[-1].tolist(),
                    "root_yaw_world_rad": _yaw_wxyz(quaternion[-1]),
                },
                "quality": {
                    "maximum_foot_penetration_m": float(
                        summary["full_body_audit"][
                            "maximum_foot_penetration_m"
                        ]
                    ),
                    "maximum_forbidden_body_penetration_m": float(
                        summary["full_body_audit"][
                            "maximum_forbidden_body_penetration_m"
                        ]
                    ),
                    "maximum_joint_step_rad": float(
                        summary["mechanics"]["maximum_joint_step_rad"]
                    ),
                    "maximum_root_acceleration_m_s2": float(
                        summary["mechanics"][
                            "maximum_root_acceleration_m_s2"
                        ]
                    ),
                    "stance_slip_p95_mps": float(
                        summary["stance_slip"]["stance_slip_p95_mps"]
                    ),
                },
                "summary_path": str(summary_path.resolve()),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["traversal"]),
            int(row["nominal_geometry"]["step_count"]),
            float(row["nominal_geometry"]["rise_m"]),
            float(row["nominal_geometry"]["tread_m"]),
            int(row["target_clip_index"]),
        )
    )
    result = {
        "schema": "global_terrain_motion_index_v2",
        "globally_privileged": True,
        "purely_kinematic": True,
        "archive_path": str(archive_path.resolve()),
        "registered_root": str(registered_root.resolve()),
        "model_path": str(model_path.resolve()),
        "maximum_terminal_support_height_error_m": float(
            maximum_terminal_support_height_error_m
        ),
        "route_count": len(rows),
        "ascent_count": sum(row["traversal"] == "up" for row in rows),
        "descent_count": sum(row["traversal"] == "down" for row in rows),
        "routes": rows,
        "rejected": rejected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def select_nearest_routes(
    index: dict[str, object],
    *,
    traversal: str,
    rise_m: float,
    tread_m: float,
    step_count: int,
    limit: int = 8,
) -> tuple[dict[str, object], ...]:
    """Rank compatible audited assets for an observed global stair mesh."""

    if traversal not in ("up", "down") or int(step_count) <= 0:
        raise ValueError("terrain route query is invalid")
    ranked: list[tuple[float, dict[str, object]]] = []
    for row in index.get("routes", ()):  # type: ignore[union-attr]
        if str(row["traversal"]) != traversal:
            continue
        geometry = row.get("physical_geometry", row["nominal_geometry"])
        route_step_count = int(
            geometry.get("level_count", geometry.get("step_count", 0))
        )
        cost = float(
            ((float(geometry["rise_m"]) - float(rise_m)) / 0.035) ** 2
            + ((float(geometry["tread_m"]) - float(tread_m)) / 0.055) ** 2
            + ((route_step_count - int(step_count)) / 1.5) ** 2
        )
        ranked.append((cost, row))
    ranked.sort(key=lambda item: (item[0], int(item[1]["target_clip_index"])))
    return tuple(
        {**row, "geometry_query_cost": cost}
        for cost, row in ranked[: int(limit)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_STAIRS_ARCHIVE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument(
        "--maximum-terminal-support-height-error-m", type=float, default=0.08
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = build_index(
        registered_root=arguments.registered_root,
        archive_path=arguments.archive,
        output_path=arguments.output,
        model_path=arguments.model_path,
        maximum_terminal_support_height_error_m=(
            arguments.maximum_terminal_support_height_error_m
        ),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "routes"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
