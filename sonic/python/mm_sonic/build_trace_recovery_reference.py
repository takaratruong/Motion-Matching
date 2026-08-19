"""Turn a successful passive learner trace into a tracker control reference."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .build_synthesized_recovery_reference import build_reference_payload


def transform_root_to_reference(
    root_pose_wxyz: np.ndarray,
    *,
    learner_front_xy: tuple[float, float],
    reference_front_xy: tuple[float, float],
    yaw_delta_rad: float,
) -> np.ndarray:
    root = np.asarray(root_pose_wxyz, dtype=np.float64).copy()
    if root.ndim != 2 or root.shape[1] != 7 or not np.isfinite(root).all():
        raise ValueError("root poses must be finite [T,7] WXYZ rows")
    c, s = math.cos(yaw_delta_rad), math.sin(yaw_delta_rad)
    delta = root[:, :2] - np.asarray(learner_front_xy)
    root[:, 0] = reference_front_xy[0] + c * delta[:, 0] - s * delta[:, 1]
    root[:, 1] = reference_front_xy[1] + s * delta[:, 0] + c * delta[:, 1]
    half = 0.5 * yaw_delta_rad
    yaw = np.asarray((math.cos(half), 0.0, 0.0, math.sin(half)))
    aw, ax, ay, az = yaw
    bw, bx, by, bz = np.moveaxis(root[:, 3:7], -1, 0)
    root[:, 3:7] = np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )
    return np.ascontiguousarray(root, dtype=np.float32)


def build(
    trace_path: Path,
    output: Path,
    *,
    learner_front_xy: tuple[float, float],
    reference_front_xy: tuple[float, float],
    yaw_delta_rad: float,
    stop_physics_step: int,
    model_path: Path,
) -> Path:
    with np.load(trace_path.expanduser().resolve(), allow_pickle=False) as trace:
        steps = np.asarray(trace["physics_step"], dtype=np.int64)
        stop = int(np.searchsorted(steps, int(stop_physics_step), side="right"))
        if stop < 3:
            raise ValueError("reference crop is too short")
        roots = transform_root_to_reference(
            np.concatenate(
                (
                    np.asarray(trace["qpos_mujoco"][:stop, :3]),
                    np.asarray(trace["qpos_mujoco"][:stop, 3:7]),
                ),
                axis=1,
            ),
            learner_front_xy=learner_front_xy,
            reference_front_xy=reference_front_xy,
            yaw_delta_rad=yaw_delta_rad,
        )
        joints = np.asarray(trace["joint_pos_isaac"][:stop], dtype=np.float32)
        fps = float(np.asarray(trace["control_hz"]).reshape(-1)[0])
    payload = build_reference_payload(
        roots[:, :3], roots[:, 3:7], joints, fps=fps, model_path=model_path
    )
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learner-front-xy", type=float, nargs=2, required=True)
    parser.add_argument("--reference-front-xy", type=float, nargs=2, default=(0, 0))
    parser.add_argument("--yaw-delta-rad", type=float, default=0.0)
    parser.add_argument("--stop-physics-step", type=int, default=3200)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    arguments = parser.parse_args(argv)
    print(
        build(
            arguments.trace,
            arguments.output,
            learner_front_xy=tuple(arguments.learner_front_xy),
            reference_front_xy=tuple(arguments.reference_front_xy),
            yaw_delta_rad=arguments.yaw_delta_rad,
            stop_physics_step=arguments.stop_physics_step,
            model_path=arguments.model,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
