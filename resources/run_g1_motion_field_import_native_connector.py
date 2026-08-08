#!/usr/bin/env python3
"""Import a MotionBricks native-qpos connector into route-array form."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mm_sonic.joints import PINNED_TARGET_TO_SOURCE_PERMUTATION


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        native = np.asarray(archive["native_qpos"], dtype=np.float64)
        support = np.asarray(archive["source_support_mask"], dtype=np.bool_)
    permutation = np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION, dtype=np.int64)
    route = {
        "joint_position": np.ascontiguousarray(native[:, 7:][:, permutation]),
        "root_position_world": np.ascontiguousarray(native[:, :3]),
        "root_orientation_world_wxyz": np.ascontiguousarray(native[:, 3:7]),
        "source_support_mask": np.ascontiguousarray(support),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **route)
    print(f"wrote {args.output} ({len(native)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
