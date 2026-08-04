#!/usr/bin/env python3
"""Create the time-reversed descent counterpart of a traversal artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mm_sonic.joints import ContractError


def _reverse_phase_boundaries(
    boundaries: object, frame_count: int
) -> np.ndarray:
    values = np.asarray(boundaries)
    if (
        values.shape != (4,)
        or values.dtype.kind not in "iu"
        or list(values) != sorted(values.tolist())
        or int(values[0]) != 0
        or int(values[-1]) != frame_count
    ):
        raise ContractError("traversal phase boundaries are invalid")
    return np.asarray(
        (
            0,
            frame_count - int(values[2]),
            frame_count - int(values[1]),
            frame_count,
        ),
        dtype=np.int64,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    required = (
        "joint_position",
        "root_position_world",
        "root_orientation_world_wxyz",
        "source_support_mask",
    )
    if any(name not in arrays for name in required):
        raise ContractError("traversal artifact is incomplete")
    frame_count = len(arrays["joint_position"])
    if (
        frame_count < 2
        or arrays["joint_position"].shape != (frame_count, 29)
        or arrays["root_position_world"].shape != (frame_count, 3)
        or arrays["root_orientation_world_wxyz"].shape != (frame_count, 4)
        or arrays["source_support_mask"].shape != (frame_count, 2)
        or arrays["source_support_mask"].dtype != np.bool_
        or not bool(arrays["source_support_mask"].any(axis=1).all())
    ):
        raise ContractError("traversal artifact arrays are invalid")
    for name, value in tuple(arrays.items()):
        if value.ndim >= 1 and len(value) == frame_count:
            arrays[name] = np.ascontiguousarray(value[::-1])
    if "phase_boundaries" in arrays:
        arrays["phase_boundaries"] = _reverse_phase_boundaries(
            arrays["phase_boundaries"], frame_count
        )
    metrics = {
        "schema": "g1-reversed-traversal/v1",
        "frame_count": frame_count,
        "unsupported_frame_count": int(
            (~arrays["source_support_mask"].any(axis=1)).sum()
        ),
        "maximum_joint_step_rad": float(
            np.abs(np.diff(arrays["joint_position"], axis=0)).max()
        ),
        "maximum_root_step_m": float(
            np.linalg.norm(
                np.diff(arrays["root_position_world"], axis=0), axis=1
            ).max()
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "traversal.npz", **arrays)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
