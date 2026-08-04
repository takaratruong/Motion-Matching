#!/usr/bin/env python3
"""Project an ARDY G1 transition prior onto an explicit terrain contact schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mm_sonic.joints import (
    ContractError,
    PINNED_TARGET_TO_SOURCE_PERMUTATION,
)
from mm_sonic.torch_contact_segments import ANKLE_ORIGIN_SOLE_M
from mm_sonic.torch_g1_fk import (
    MujocoG1FootKinematics,
    target_state_qpos,
)
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_horizontal_terrain_retarget import (
    WideBoundG1TerrainRetargeter,
)
from mm_sonic.torch_supported_step_up import smooth_swing_clearance_lift
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)


def _load_frame(path: Path, frame: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        output = {
            "joint": np.asarray(archive["joint_position"])[frame].copy(),
            "root": np.asarray(archive["root_position_world"])[frame].copy(),
            "quaternion": np.asarray(
                archive["root_orientation_world_wxyz"]
            )[frame].copy(),
            "support": np.asarray(
                archive["source_support_mask"], dtype=np.bool_
            )[frame].copy(),
        }
    return output


def _stance_height_shift(
    ankle_clearance: object, support_mask: object
) -> np.ndarray:
    clearance = np.asarray(ankle_clearance, dtype=np.float64)
    support = np.asarray(support_mask)
    if (
        clearance.ndim != 2
        or clearance.shape[1:] != (2,)
        or support.shape != clearance.shape
        or support.dtype != np.bool_
        or not np.isfinite(clearance).all()
        or not bool(support.any(axis=1).all())
    ):
        raise ContractError("ARDY stance-height normalization is invalid")
    return np.asarray(
        [
            float(clearance[frame, support[frame]].mean())
            for frame in range(len(clearance))
        ],
        dtype=np.float64,
    )


def _two_step_schedule(
    *,
    start_feet: object,
    stop_feet: object,
    frame_count: int,
    first_landing_frame: int,
    second_liftoff_frame: int,
    second_landing_frame: int,
    initial_support_foot: int,
    final_support_foot: int,
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start_feet, dtype=np.float64)
    stop = np.asarray(stop_feet, dtype=np.float64)
    if (
        start.shape != (2, 3)
        or stop.shape != (2, 3)
        or not np.isfinite(start).all()
        or not np.isfinite(stop).all()
        or initial_support_foot not in (0, 1)
        or final_support_foot not in (0, 1)
        or not 0
        < first_landing_frame
        < second_liftoff_frame
        < second_landing_frame
        < frame_count
    ):
        raise ContractError("ARDY terrain bridge schedule is invalid")
    targets = np.empty((frame_count, 2, 3), dtype=np.float64)
    support = np.zeros((frame_count, 2), dtype=np.bool_)
    first_moving_foot = 1 - initial_support_foot
    second_moving_foot = initial_support_foot
    for frame in range(frame_count):
        first_alpha = float(
            np.clip(frame / first_landing_frame, 0.0, 1.0)
        )
        second_alpha = float(
            np.clip(
                (frame - second_liftoff_frame)
                / (second_landing_frame - second_liftoff_frame),
                0.0,
                1.0,
            )
        )
        alpha = np.ones(2, dtype=np.float64)
        alpha[first_moving_foot] = first_alpha
        alpha[second_moving_foot] = second_alpha
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        targets[frame] = (
            (1.0 - smooth[:, None]) * start
            + smooth[:, None] * stop
        )
        targets[frame, first_moving_foot, 2] += (
            0.08 * 4.0 * first_alpha * (1.0 - first_alpha)
        )
        targets[frame, second_moving_foot, 2] += (
            0.10 * 4.0 * second_alpha * (1.0 - second_alpha)
        )
        if frame <= first_landing_frame:
            support[frame, initial_support_foot] = True
        elif frame <= second_landing_frame:
            support[frame, first_moving_foot] = True
        else:
            support[frame, final_support_foot] = True
    return targets, support


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-qpos-csv", type=Path, required=True)
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument("--outgoing", type=Path, required=True)
    parser.add_argument("--incoming-frame", type=int, required=True)
    parser.add_argument("--outgoing-frame", type=int, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    incoming = _load_frame(args.incoming, args.incoming_frame)
    outgoing = _load_frame(args.outgoing, args.outgoing_frame)
    qpos = np.loadtxt(args.ardy_qpos_csv, delimiter=",")
    if qpos.ndim != 2 or qpos.shape[1:] != (36,) or len(qpos) < 8:
        raise ContractError("ARDY qpos prior is invalid")
    incoming_qpos = target_state_qpos(
        incoming["joint"], incoming["root"], incoming["quaternion"]
    )
    qpos[:, :2] += incoming_qpos[:2]
    permutation = np.asarray(
        PINNED_TARGET_TO_SOURCE_PERMUTATION, dtype=np.int64
    )
    joints = qpos[:, 7:][:, permutation].copy()
    roots = qpos[:, :3].copy()
    quaternions = qpos[:, 3:7].copy()
    joints[0] = incoming["joint"]
    roots[0] = incoming["root"]
    quaternions[0] = incoming["quaternion"]

    foot_kinematics = MujocoG1FootKinematics(args.g1_xml)
    sole_kinematics = MujocoG1SoleKinematics(args.g1_xml)
    start_feet = foot_kinematics.foot_positions(
        incoming["joint"][None],
        incoming["root"][None],
        incoming["quaternion"][None],
    )[0]
    stop_feet = foot_kinematics.foot_positions(
        outgoing["joint"][None],
        outgoing["root"][None],
        outgoing["quaternion"][None],
    )[0]
    frame_count = len(joints)
    first_landing = int(round(0.24 * frame_count))
    second_liftoff = int(round(0.28 * frame_count))
    second_landing = int(round(0.68 * frame_count))
    if incoming["support"].sum() != 1 or outgoing["support"].sum() != 1:
        raise ContractError(
            "ARDY terrain bridge endpoints require singleton support"
        )
    targets, support = _two_step_schedule(
        start_feet=start_feet,
        stop_feet=stop_feet,
        frame_count=frame_count,
        first_landing_frame=first_landing,
        second_liftoff_frame=second_liftoff,
        second_landing_frame=second_landing,
        initial_support_foot=int(np.flatnonzero(incoming["support"])[0]),
        final_support_foot=int(np.flatnonzero(outgoing["support"])[0]),
    )
    retargeter = WideBoundG1TerrainRetargeter(
        args.g1_xml,
        maximum_joint_deviation_rad=1.4,
        maximum_root_height_deviation_m=0.12,
        maximum_root_horizontal_deviation_m=0.04,
        maximum_target_error_m=0.04,
    )
    for frame in range(1, frame_count - 1):
        joints[frame], roots[frame] = retargeter.solve_frame(
            joint_position=joints[frame],
            root_position_world=roots[frame],
            root_orientation_world_wxyz=quaternions[frame],
            solve_feet=np.ones(2, dtype=np.bool_),
            target_foot_position_world=targets[frame],
            level_feet=support[frame],
            initial_joint_position=joints[frame - 1],
            initial_root_position_world=roots[frame - 1],
        )
    joints[-1] = outgoing["joint"]
    roots[-1] = outgoing["root"]
    quaternions[-1] = outgoing["quaternion"]

    resolved = resolve_stair_config(
        args.target_dataset,
        load_experiment_config(args.config),
        device="cpu",
    )
    ankles = foot_kinematics.foot_positions(joints, roots, quaternions)
    soles = sole_kinematics.sole_points(joints, roots, quaternions)
    alignment = resolved.measurement_extension.alignment
    grid = resolved.measurement_extension.query_grid
    ankle_surface = grid.sample_xy(
        alignment.matcher_to_scene_xy(
            torch.tensor(ankles[..., :2], dtype=torch.float32)
        )
    ).cpu().numpy()
    sole_surface = grid.sample_xy(
        alignment.matcher_to_scene_xy(
            torch.tensor(soles[..., :2], dtype=torch.float32)
        )
    ).cpu().numpy()
    ankle_clearance = (
        ankles[..., 2] - ankle_surface - float(ANKLE_ORIGIN_SOLE_M)
    )
    sole_clearance = soles[..., 2] - sole_surface
    stance_height_shift = _stance_height_shift(
        ankle_clearance, support
    )
    roots[:, 2] -= stance_height_shift
    ankles[..., 2] -= stance_height_shift[:, None]
    soles[..., 2] -= stance_height_shift[:, None, None]
    ankle_clearance -= stance_height_shift[:, None]
    sole_clearance -= stance_height_shift[:, None, None]
    planned_lift = smooth_swing_clearance_lift(
        support_mask=support,
        minimum_sole_clearance_m=sole_clearance.min(axis=2),
        smoothing_radius_frames=4,
    )
    swing_retargeter = WideBoundG1TerrainRetargeter(
        args.g1_xml,
        maximum_root_height_deviation_m=1.0e-6,
        maximum_root_horizontal_deviation_m=1.0e-6,
        maximum_target_error_m=0.04,
    )
    corrected_frames = 0
    for frame in np.flatnonzero((planned_lift > 0.0).any(axis=1)):
        solve = planned_lift[frame] > 0.0
        targets = ankles[frame].copy()
        targets[:, 2] += planned_lift[frame]
        joints[frame], roots[frame] = swing_retargeter.solve_frame(
            joint_position=joints[frame],
            root_position_world=roots[frame],
            root_orientation_world_wxyz=quaternions[frame],
            solve_feet=solve,
            target_foot_position_world=targets,
            level_feet=np.zeros(2, dtype=np.bool_),
            initial_joint_position=(
                None if frame == 0 else joints[frame - 1]
            ),
            initial_root_position_world=(
                None if frame == 0 else roots[frame - 1]
            ),
        )
        corrected_frames += 1
    if corrected_frames:
        ankles = foot_kinematics.foot_positions(
            joints, roots, quaternions
        )
        soles = sole_kinematics.sole_points(joints, roots, quaternions)
        ankle_surface = grid.sample_xy(
            alignment.matcher_to_scene_xy(
                torch.tensor(ankles[..., :2], dtype=torch.float32)
            )
        ).cpu().numpy()
        sole_surface = grid.sample_xy(
            alignment.matcher_to_scene_xy(
                torch.tensor(soles[..., :2], dtype=torch.float32)
            )
        ).cpu().numpy()
        ankle_clearance = (
            ankles[..., 2]
            - ankle_surface
            - float(ANKLE_ORIGIN_SOLE_M)
        )
        sole_clearance = soles[..., 2] - sole_surface
    metrics = {
        "schema": "g1-ardy-terrain-bridge/v1",
        "frame_count": frame_count,
        "maximum_joint_step_rad": float(
            np.abs(np.diff(joints, axis=0)).max()
        ),
        "maximum_root_step_m": float(
            np.linalg.norm(np.diff(roots, axis=0), axis=1).max()
        ),
        "maximum_stance_contact_error_m": float(
            np.abs(ankle_clearance[support]).max()
        ),
        "minimum_sole_clearance_m": float(sole_clearance.min()),
        "unsupported_frame_count": int((~support.any(axis=1)).sum()),
        "swing_corrected_frame_count": corrected_frames,
        "maximum_stance_root_correction_m": float(
            np.abs(stance_height_shift).max()
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "transition.npz",
        joint_position=joints,
        root_position_world=roots,
        root_orientation_world_wxyz=quaternions,
        source_support_mask=support,
        phase_boundaries=np.asarray(
            (0, first_landing, second_landing, frame_count),
            dtype=np.int64,
        ),
        minimum_sole_clearance_by_frame=sole_clearance.min(axis=(1, 2)),
    )
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
