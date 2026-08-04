#!/usr/bin/env python3
"""Compose contact-valid G1 traversal primitives into one route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mm_sonic.joints import ContractError


_FRAME_ARRAYS = (
    "joint_position",
    "root_position_world",
    "root_orientation_world_wxyz",
    "source_support_mask",
)


def _validate_segment(segment: dict[str, np.ndarray]) -> None:
    if any(name not in segment for name in _FRAME_ARRAYS):
        raise ContractError("traversal segment is incomplete")
    frame_count = len(segment["joint_position"])
    if (
        frame_count < 2
        or segment["joint_position"].shape != (frame_count, 29)
        or segment["root_position_world"].shape != (frame_count, 3)
        or segment["root_orientation_world_wxyz"].shape != (frame_count, 4)
        or segment["source_support_mask"].shape != (frame_count, 2)
        or segment["source_support_mask"].dtype != np.bool_
        or not bool(segment["source_support_mask"].any(axis=1).all())
        or not all(
            np.isfinite(segment[name]).all()
            for name in _FRAME_ARRAYS
            if name != "source_support_mask"
        )
        or np.any(
            np.abs(
                np.linalg.norm(
                    segment["root_orientation_world_wxyz"], axis=1
                )
                - 1.0
            )
            > 1.0e-4
        )
    ):
        raise ContractError("traversal segment arrays are invalid")


def _junction_error(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    joint = float(
        np.abs(
            first["joint_position"][-1] - second["joint_position"][0]
        ).max()
    )
    root = float(
        np.linalg.norm(
            first["root_position_world"][-1]
            - second["root_position_world"][0]
        )
    )
    quaternion_dot = float(
        abs(
            np.dot(
                first["root_orientation_world_wxyz"][-1],
                second["root_orientation_world_wxyz"][0],
            )
        )
    )
    orientation = float(2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0)))
    return joint, root, orientation


def _compose_segments(
    segments: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if len(segments) < 2:
        raise ContractError("at least two traversal segments are required")
    for segment in segments:
        _validate_segment(segment)
    pieces: dict[str, list[np.ndarray]] = {
        name: [] for name in _FRAME_ARRAYS
    }
    boundaries = [0]
    for index, segment in enumerate(segments):
        start = 0
        if index:
            joint, root, orientation = _junction_error(
                segments[index - 1], segment
            )
            if joint > 0.35 or root > 0.05 or orientation > np.deg2rad(8.0):
                raise ContractError(
                    "traversal junction is discontinuous: "
                    f"joint={joint:.6f} rad, root={root:.6f} m, "
                    f"orientation={np.rad2deg(orientation):.3f} deg"
                )
            if joint < 1.0e-8 and root < 1.0e-8 and orientation < 1.0e-8:
                start = 1
        for name in _FRAME_ARRAYS:
            pieces[name].append(segment[name][start:])
        boundaries.append(
            boundaries[-1] + len(segment["joint_position"][start:])
        )
    output = {
        name: np.ascontiguousarray(np.concatenate(values, axis=0))
        for name, values in pieces.items()
    }
    output["segment_boundaries"] = np.asarray(
        boundaries, dtype=np.int64
    )
    return output


def _load_segment(specification: str) -> dict[str, np.ndarray]:
    path_text, separator, slice_text = specification.rpartition("@")
    path = Path(path_text if separator else specification)
    start = 0
    stop: int | None = None
    if separator:
        fields = slice_text.split(":")
        if len(fields) != 2:
            raise ContractError("segment slice must use START:STOP")
        try:
            start = int(fields[0]) if fields[0] else 0
            stop = int(fields[1]) if fields[1] else None
        except ValueError as error:
            raise ContractError("segment slice is invalid") from error
    with np.load(path, allow_pickle=False) as archive:
        segment = {
            name: np.asarray(archive[name])[start:stop].copy()
            for name in _FRAME_ARRAYS
            if name in archive
        }
    _validate_segment(segment)
    return segment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segment",
        action="append",
        required=True,
        help="Traversal NPZ, optionally followed by @START:STOP.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _compose_segments(
        [_load_segment(specification) for specification in args.segment]
    )
    joints = output["joint_position"]
    roots = output["root_position_world"]
    support = output["source_support_mask"]
    metrics = {
        "schema": "g1-composed-traversal/v1",
        "frame_count": len(joints),
        "segment_boundaries": output["segment_boundaries"].tolist(),
        "unsupported_frame_count": int((~support.any(axis=1)).sum()),
        "maximum_joint_step_rad": float(
            np.abs(np.diff(joints, axis=0)).max()
        ),
        "maximum_root_step_m": float(
            np.linalg.norm(np.diff(roots, axis=0), axis=1).max()
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "traversal.npz", **output)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
