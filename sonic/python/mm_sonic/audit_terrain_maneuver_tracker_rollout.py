"""Audit saved SONIC terrain rollouts against their exact terrain meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .generate_generic_stair_route import load_target_mesh
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.stitch import FrameProvenance, StitchedMotion


def _rollout_zarr(path: Path) -> Path:
    source = path.expanduser().resolve()
    if source.is_dir() and source.suffix != ".zarr":
        matches = sorted(source.glob("*.zarr"))
        if len(matches) != 1:
            raise ValueError(
                f"expected one rollout zarr in {source}, found {len(matches)}"
            )
        return matches[0]
    return source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--maximum-foot-penetration-m", type=float, default=0.010)
    parser.add_argument(
        "--maximum-forbidden-body-penetration-m", type=float, default=1.0e-6
    )
    arguments = parser.parse_args(argv)

    if arguments.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if not 0 <= arguments.shard_index < arguments.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")

    rollout_path = _rollout_zarr(arguments.rollout)
    rollout = zarr.open_group(str(rollout_path), mode="r")
    data = rollout["data"]
    meta = rollout["meta"]
    if str(np.asarray(meta["quaternion_convention"])) != "wxyz":
        raise ValueError("rollout quaternions must use wxyz convention")

    ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), ends[:-1]))
    clips = np.asarray(meta["episode_clip"][:]).astype(str)
    origins = np.asarray(meta["episode_env_origin"][:], dtype=np.float64)
    failures = np.asarray(meta["episode_failed"][:], dtype=bool)
    mpjpe = np.asarray(meta["episode_mpjpe_mm"][:], dtype=np.float64)
    joint_names = tuple(str(value) for value in meta["joint_names"][:])
    default_joint_position = np.asarray(
        meta["default_joint_pos"][:], dtype=np.float64
    )

    bundle = arguments.bundle.expanduser().resolve()
    records = {
        str(row["stem"]): row
        for row in json.loads((bundle / "clips.json").read_text())
    }
    unique_clips = sorted(set(clips))
    assigned_clips = set(
        unique_clips[arguments.shard_index :: arguments.shard_count]
    )
    episode_indices = sorted(
        (index for index, clip in enumerate(clips) if clip in assigned_clips),
        key=lambda index: (clips[index], index),
    )

    rows: list[dict[str, object]] = []
    for index in episode_indices:
        start, stop = int(starts[index]), int(ends[index])
        clip = str(clips[index])
        terrain_pose = records[clip]["terrain"]
        terrain = load_target_mesh(
            bundle / "object_usd" / f"{clip}.usd",
            position_world=terrain_pose["position_env"],
            quaternion_world_from_usd_wxyz=(
                terrain_pose["rotation_env_wxyz"]
            ),
        )
        frame_count = stop - start
        motion = StitchedMotion(
            fps=50.0,
            root_position_world=(
                np.asarray(data["root_pos"][start:stop], dtype=np.float64)
                - origins[index]
            ),
            root_quaternion_world_wxyz=np.asarray(
                data["root_rot"][start:stop], dtype=np.float64
            ),
            # SONIC rollout joint positions are stored relative to this pose.
            joint_position=(
                np.asarray(data["joint_pos"][start:stop], dtype=np.float64)
                + default_joint_position
            ),
            provenance=tuple(
                FrameProvenance(-1, frame, clip)
                for frame in range(frame_count)
            ),
            seam_indices=(),
        )
        audit = audit_stair_motion_collisions(
            motion,
            model_path=arguments.model,
            target_mesh=terrain,
            joint_names=joint_names,
            maximum_foot_penetration_m=(
                arguments.maximum_foot_penetration_m
            ),
            maximum_forbidden_body_penetration_m=(
                arguments.maximum_forbidden_body_penetration_m
            ),
        )
        row = {
            "episode": int(index),
            "clip": clip,
            "frame_count": frame_count,
            "episode_failed": bool(failures[index]),
            "peak_mpjpe_mm": float(mpjpe[index]),
            "geometry_accepted": bool(audit.accepted),
            "maximum_foot_penetration_m": float(
                audit.maximum_foot_penetration_m
            ),
            "maximum_forbidden_body_penetration_m": float(
                audit.maximum_forbidden_body_penetration_m
            ),
            "foot_threshold_exceedance_frame_count": len(
                audit.foot_threshold_exceedance_frame_indices
            ),
            "forbidden_body_threshold_exceedance_frame_count": len(
                audit.forbidden_body_threshold_exceedance_frame_indices
            ),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "rollout": str(rollout_path),
                "bundle": str(bundle),
                "shard_index": int(arguments.shard_index),
                "shard_count": int(arguments.shard_count),
                "thresholds": {
                    "maximum_foot_penetration_m": float(
                        arguments.maximum_foot_penetration_m
                    ),
                    "maximum_forbidden_body_penetration_m": float(
                        arguments.maximum_forbidden_body_penetration_m
                    ),
                },
                "episode_count": len(rows),
                "geometry_accepted_count": sum(
                    bool(row["geometry_accepted"]) for row in rows
                ),
                "episodes": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
