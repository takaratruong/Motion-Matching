"""Render a continuous terrain-matching trace as articulated G1 kinematics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .canonical_terrain_matcher import (
    RegularGridHeightField,
    load_canonical_terrain_library,
)
from .render_stitched_motion import render_stitched_motion
from .terrain_oracle.canonical import ISAACLAB_JOINT_NAMES
from .terrain_oracle.stitch import FrameProvenance, StitchedMotion


DEFAULT_CORPUS = Path(
    "/move/data/terrain-aware/motion-matching/terrain-oracle-task9.wFKkIV/raw"
)
DEFAULT_MODEL = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/g1_29dof_rev_1_0.xml"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-clip-id")
    target.add_argument("--terrain", type=Path)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    with np.load(args.trace.expanduser().resolve(), allow_pickle=False) as data:
        roots = np.asarray(data["root_position_world"], dtype=np.float32)
        quaternions = np.asarray(
            data["root_quaternion_world_wxyz"], dtype=np.float32
        )
        joints = np.asarray(data["joint_position"], dtype=np.float32)
    if args.stride <= 0:
        raise ValueError("stride must be positive")
    roots = roots[:: args.stride]
    quaternions = quaternions[:: args.stride]
    joints = joints[:: args.stride]
    provenance = tuple(
        FrameProvenance(0, frame, "continuous-terrain-generated")
        for frame in range(len(roots))
    )
    motion = StitchedMotion(
        fps=50.0 / float(args.stride),
        root_position_world=roots,
        root_quaternion_world_wxyz=quaternions,
        joint_position=joints,
        provenance=provenance,
        seam_indices=(),
    )
    if args.terrain is None:
        target_library = load_canonical_terrain_library(
            args.corpus, clip_ids=[args.target_clip_id]
        )
        target_mesh = target_library.height_fields[0].index
        joint_names = target_library.canonical_clips[0].joint_names
    else:
        with np.load(args.terrain.expanduser().resolve(), allow_pickle=False) as data:
            spacing = tuple(float(value) for value in data["spacing_m"])
            origin = tuple(float(value) for value in data["origin_xy"])
            field = RegularGridHeightField(
                np.asarray(data["height"], dtype=np.float64),
                valid=np.asarray(data["valid"], dtype=np.bool_),
                spacing_m=spacing,
                origin_xy=origin,
            )
        target_mesh = field.index
        joint_names = ISAACLAB_JOINT_NAMES
    result = render_stitched_motion(
        motion,
        model_path=args.model,
        output_path=args.output,
        target_mesh=target_mesh,
        joint_names=joint_names,
        width=args.width,
        height=args.height,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
