#!/usr/bin/env python3
"""Temporally upsample a contact-valid G1 traversal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from mm_sonic.joints import ContractError


def _resample(
    *,
    joints: object,
    roots: object,
    quaternions: object,
    support: object,
    factor: int,
) -> dict[str, np.ndarray]:
    joint = np.asarray(joints, dtype=np.float64)
    root = np.asarray(roots, dtype=np.float64)
    quaternion = np.asarray(quaternions, dtype=np.float64)
    support_mask = np.asarray(support)
    frame_count = len(joint)
    if (
        type(factor) is not int
        or factor < 2
        or frame_count < 2
        or joint.shape != (frame_count, 29)
        or root.shape != (frame_count, 3)
        or quaternion.shape != (frame_count, 4)
        or support_mask.shape != (frame_count, 2)
        or support_mask.dtype != np.bool_
        or not bool(support_mask.any(axis=1).all())
        or not all(
            np.isfinite(value).all()
            for value in (joint, root, quaternion)
        )
        or np.any(
            np.abs(np.linalg.norm(quaternion, axis=1) - 1.0) > 1.0e-4
        )
    ):
        raise ContractError("traversal resampling factor or arrays are invalid")
    source_time = np.arange(frame_count, dtype=np.float64)
    target_time = np.linspace(
        0.0,
        float(frame_count - 1),
        (frame_count - 1) * factor + 1,
    )
    output_joints = np.stack(
        [
            np.interp(target_time, source_time, joint[:, index])
            for index in range(29)
        ],
        axis=1,
    )
    output_roots = np.stack(
        [
            np.interp(target_time, source_time, root[:, index])
            for index in range(3)
        ],
        axis=1,
    )
    output_quaternions = Slerp(
        source_time,
        Rotation.from_quat(quaternion, scalar_first=True),
    )(target_time).as_quat(scalar_first=True)
    nearest = np.rint(target_time).astype(np.int64)
    return {
        "joint_position": np.ascontiguousarray(output_joints),
        "root_position_world": np.ascontiguousarray(output_roots),
        "root_orientation_world_wxyz": np.ascontiguousarray(
            output_quaternions
        ),
        "source_support_mask": np.ascontiguousarray(
            support_mask[nearest]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        arrays = _resample(
            joints=archive["joint_position"],
            roots=archive["root_position_world"],
            quaternions=archive["root_orientation_world_wxyz"],
            support=archive["source_support_mask"],
            factor=args.factor,
        )
        if "phase_boundaries" in archive:
            boundaries = np.asarray(
                archive["phase_boundaries"], dtype=np.int64
            )
            if (
                boundaries.shape != (4,)
                or int(boundaries[0]) != 0
                or int(boundaries[-1])
                != len(archive["joint_position"])
                or list(boundaries) != sorted(boundaries.tolist())
            ):
                raise ContractError(
                    "traversal phase boundaries are invalid"
                )
            arrays["phase_boundaries"] = np.asarray(
                (
                    0,
                    int(boundaries[1]) * args.factor,
                    int(boundaries[2]) * args.factor,
                    len(arrays["joint_position"]),
                ),
                dtype=np.int64,
            )
    joints = arrays["joint_position"]
    roots = arrays["root_position_world"]
    support = arrays["source_support_mask"]
    metrics = {
        "schema": "g1-resampled-traversal/v1",
        "frame_count": len(joints),
        "resampling_factor": args.factor,
        "maximum_joint_step_rad": float(
            np.abs(np.diff(joints, axis=0)).max()
        ),
        "maximum_root_step_m": float(
            np.linalg.norm(np.diff(roots, axis=0), axis=1).max()
        ),
        "unsupported_frame_count": int((~support.any(axis=1)).sum()),
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
