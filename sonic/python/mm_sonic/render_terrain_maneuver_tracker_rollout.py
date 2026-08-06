"""Render saved SONIC terrain-maneuver episodes without rerunning physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .generate_generic_stair_route import load_target_mesh
from .render_stitched_motion import render_stitched_motion
from .terrain_oracle.stitch import FrameProvenance, StitchedMotion


def _rollout_zarr(path: Path) -> Path:
    source = path.expanduser().resolve()
    if source.is_dir() and source.suffix != ".zarr":
        matches = sorted(source.glob("*.zarr"))
        if len(matches) != 1:
            raise ValueError(f"expected one rollout zarr in {source}, found {len(matches)}")
        return matches[0]
    return source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--episode", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--frame-stride", type=int, default=2)
    arguments = parser.parse_args(argv)

    rollout = zarr.open_group(str(_rollout_zarr(arguments.rollout)), mode="r")
    data = rollout["data"]
    meta = rollout["meta"]
    convention = str(np.asarray(meta["quaternion_convention"]))
    if convention != "wxyz":
        raise ValueError(f"unsupported rollout quaternion convention: {convention}")
    ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), ends[:-1]))
    clips = np.asarray(meta["episode_clip"][:]).astype(str)
    origins = np.asarray(meta["episode_env_origin"][:], dtype=np.float64)
    joint_names = tuple(str(value) for value in meta["joint_names"][:])

    bundle = arguments.bundle.expanduser().resolve()
    records = {
        str(row["stem"]): row
        for row in json.loads((bundle / "clips.json").read_text())
    }
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, object]] = []
    for episode in arguments.episode:
        index = int(episode)
        if index < 0 or index >= len(ends):
            raise IndexError(f"episode {index} is outside [0, {len(ends)})")
        start, stop = int(starts[index]), int(ends[index])
        clip = str(clips[index])
        record = records[clip]
        terrain_pose = record["terrain"]
        terrain = load_target_mesh(
            bundle / "object_usd" / f"{clip}.usd",
            position_world=terrain_pose["position_env"],
            quaternion_world_from_usd_wxyz=terrain_pose["rotation_env_wxyz"],
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
            joint_position=np.asarray(
                data["joint_pos"][start:stop], dtype=np.float64
            ),
            provenance=tuple(
                FrameProvenance(-1, frame, clip) for frame in range(frame_count)
            ),
            seam_indices=(),
        )
        output = output_dir / f"tp_ep{index:02d}_{clip}.mp4"
        receipt = render_stitched_motion(
            motion,
            model_path=arguments.model,
            output_path=output,
            target_mesh=terrain,
            joint_names=joint_names,
            width=arguments.width,
            height=arguments.height,
            frame_stride=arguments.frame_stride,
        )
        receipts.append(
            {
                "episode": index,
                "clip": clip,
                "failed": bool(meta["episode_failed"][index]),
                "peak_mpjpe_mm": float(meta["episode_mpjpe_mm"][index]),
                **receipt,
            }
        )
        print(output, flush=True)
    (output_dir / "render_manifest.json").write_text(
        json.dumps({"videos": receipts}, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
