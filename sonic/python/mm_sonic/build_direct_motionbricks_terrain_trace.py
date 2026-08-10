"""Place one unbroken MotionBricks clip over a known terrain course.

This is a kinematic proposal, not a contact repair.  It preserves the exact
source gait phase and travel/facing relationship, applies one planar rigid
registration, and gives the pelvis only a smooth common terrain-height lift.
The normal rigid-foothold postprocessor remains responsible for full-sole
support, swing clearance, collision checks, and final acceptance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .canonical_terrain_matcher import RegularGridHeightField


def _yaw_wxyz(value: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(value, dtype=np.float64), -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _left_multiply_yaw(quaternion_wxyz: np.ndarray, yaw: float) -> np.ndarray:
    source = np.asarray(quaternion_wxyz, dtype=np.float64)
    cosine = math.cos(0.5 * float(yaw))
    sine = math.sin(0.5 * float(yaw))
    w, x, y, z = np.moveaxis(source, -1, 0)
    return np.stack(
        (
            cosine * w - sine * z,
            cosine * x - sine * y,
            cosine * y + sine * x,
            cosine * z + sine * w,
        ),
        axis=-1,
    )


def _rotate_xy(values: np.ndarray, yaw: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    result = source.copy()
    result[..., 0] = cosine * source[..., 0] - sine * source[..., 1]
    result[..., 1] = sine * source[..., 0] + cosine * source[..., 1]
    return result


def _smooth_bounded_height(
    desired: np.ndarray, *, maximum_step_m: float = 0.012
) -> np.ndarray:
    target = np.asarray(desired, dtype=np.float64)
    result = np.empty_like(target)
    result[0] = target[0]
    for frame in range(1, len(target)):
        result[frame] = result[frame - 1] + float(
            np.clip(
                target[frame] - result[frame - 1],
                -float(maximum_step_m),
                float(maximum_step_m),
            )
        )
    # A zero-phase three-frame pass removes the corners introduced by the hard
    # rate limit without changing the terrain-height endpoints materially.
    kernel = np.asarray((1.0, 2.0, 3.0, 2.0, 1.0), dtype=np.float64)
    kernel /= np.sum(kernel)
    padded = np.pad(result, (2, 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _clip_record(corpus: Path, clip_id: str) -> dict[str, object]:
    manifest = json.loads((corpus / "manifest.json").read_text())
    for record in manifest["clips"]:
        if record["clip_id"] == clip_id:
            return record
    raise ValueError(f"clip is not present in corpus: {clip_id}")


def build(args: argparse.Namespace) -> dict[str, object]:
    record = _clip_record(args.corpus, args.clip_id)
    with np.load(args.corpus / str(record["relative_path"]), allow_pickle=False) as data:
        source = {key: np.asarray(data[key]) for key in data.files}
    total = len(source["root_position_world"])
    start = int(args.start_frame)
    stop = total if args.stop_frame is None else min(total, int(args.stop_frame))
    if start < 0 or stop - start < 3:
        raise ValueError("source frame range is invalid")

    root_all = np.asarray(source["root_position_world"], dtype=np.float64)
    quaternion_all = np.asarray(
        source["root_quaternion_world_wxyz"], dtype=np.float64
    )
    source_yaw = float(_yaw_wxyz(quaternion_all[start]))
    target_yaw = math.radians(float(args.start_yaw_deg))
    yaw_offset = target_yaw - source_yaw
    rotated_root = _rotate_xy(root_all, yaw_offset)
    translation_xy = np.asarray((args.start_x, args.start_y), dtype=np.float64) - (
        rotated_root[start, :2]
    )
    rotated_root[:, :2] += translation_xy
    rotated_quaternion = _left_multiply_yaw(quaternion_all, yaw_offset)

    with np.load(args.terrain, allow_pickle=False) as data:
        terrain_arrays = {key: np.asarray(data[key]) for key in data.files}
    field = RegularGridHeightField(
        terrain_arrays["height"],
        spacing_m=tuple(float(value) for value in terrain_arrays["spacing_m"]),
        origin_xy=tuple(float(value) for value in terrain_arrays["origin_xy"]),
        valid=terrain_arrays.get("valid"),
    )
    root = rotated_root[start:stop].copy()
    terrain_height, _normal, hit = field.sample(root[:, :2])
    if not np.all(hit):
        raise ValueError("registered source path leaves the target height field")
    # The canonical MotionBricks corpus is authored over a world-horizontal
    # source floor (normally z=0).  Register that floor to the *absolute*
    # target surface.  Subtracting the target height at frame zero would only
    # be valid if the source pose had already been placed on that surface; on
    # a raised course it embeds the entire character by that initial height.
    terrain_lift = _smooth_bounded_height(
        terrain_height - float(args.source_ground_height_m),
        maximum_step_m=float(args.maximum_vertical_step_m),
    )
    root[:, 2] += terrain_lift

    sole = _rotate_xy(
        np.asarray(source["sole_position_world"], dtype=np.float64), yaw_offset
    )
    sole[..., :2] += translation_xy
    sole = sole[start:stop].copy()
    sole[..., 2] += terrain_lift[:, None]
    native_speed = np.linalg.norm(
        np.gradient(
            np.asarray(source["sole_position_world"], dtype=np.float64), axis=0
        )
        * float(args.fps),
        axis=2,
    )[start:stop]
    source_sole_height_above_ground = (
        np.asarray(source["sole_position_world"], dtype=np.float64)[
            start:stop, :, 2
        ]
        - float(args.source_ground_height_m)
    )
    velocity = np.diff(root[:, :2], axis=0) * float(args.fps)
    heading = np.full(len(root) - 1, target_yaw, dtype=np.float32)
    source_frames = np.arange(start, stop - 1, dtype=np.int32)

    trace = args.output.with_suffix(".trace.npz")
    terrain = args.output.with_suffix(".terrain.npz")
    np.savez_compressed(
        trace,
        root_position_world=np.asarray(root, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(
            rotated_quaternion[start:stop], dtype=np.float32
        ),
        joint_position=np.asarray(source["joint_position"][start:stop], dtype=np.float32),
        foot_position_world=np.asarray(sole, dtype=np.float32),
        foot_probe_position_world=np.asarray(sole[:, :, None], dtype=np.float32),
        contact=np.asarray(
            source["contact"][start : stop - 1] >= 0.5, dtype=np.bool_
        ),
        source_sole_speed_mps=np.asarray(native_speed, dtype=np.float32),
        source_sole_height_above_ground_m=np.asarray(
            source_sole_height_above_ground, dtype=np.float32
        ),
        requested_velocity_world_xy=np.asarray(velocity, dtype=np.float32),
        requested_heading_world_yaw=heading,
        applied_velocity_world_xy=np.asarray(velocity, dtype=np.float32),
        applied_heading_world_yaw=heading,
        target_path_y=np.full(len(root) - 1, float(args.start_y), dtype=np.float32),
        selected_clip_index=np.zeros(len(root) - 1, dtype=np.int32),
        selected_source_frame=source_frames,
    )
    np.savez_compressed(terrain, **terrain_arrays)
    report = {
        "schema": "direct-motionbricks-terrain-trace/v1",
        "clip_id": args.clip_id,
        "source_frame_range": [start, stop],
        "frame_count": len(root),
        "yaw_offset_rad": yaw_offset,
        "root_progress_xy_m": (root[-1, :2] - root[0, :2]).tolist(),
        "maximum_terrain_lift_step_m": float(
            np.max(np.abs(np.diff(terrain_lift)))
        ),
        "source_ground_height_m": float(args.source_ground_height_m),
        "initial_target_terrain_height_m": float(terrain_height[0]),
        "trace": str(trace.resolve()),
        "terrain": str(terrain.resolve()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--terrain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--start-x", type=float, default=-3.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-yaw-deg", type=float, default=90.0)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--maximum-vertical-step-m", type=float, default=0.012)
    parser.add_argument("--source-ground-height-m", type=float, default=0.0)
    args = parser.parse_args()
    args.corpus = args.corpus.expanduser().resolve()
    args.terrain = args.terrain.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
