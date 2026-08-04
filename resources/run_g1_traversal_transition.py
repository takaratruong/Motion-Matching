#!/usr/bin/env python3
"""Build a supported turn between two validated stair traversals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_contact_segments import ANKLE_ORIGIN_SOLE_M
from mm_sonic.torch_g1_fk import MujocoG1FootKinematics
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_horizontal_terrain_retarget import (
    WideBoundG1TerrainRetargeter,
)
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument("--outgoing", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blend-frames", type=int, default=20)
    parser.add_argument("--junction-tail-frames", type=int, default=30)
    return parser


def _load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "joint_position",
            "root_position_world",
            "root_orientation_world_wxyz",
            "source_support_mask",
        )
        if any(name not in archive for name in required):
            raise ContractError("traversal artifact is incomplete")
        output = {name: np.asarray(archive[name]).copy() for name in required}
        if "phase_boundaries" in archive:
            output["phase_boundaries"] = np.asarray(
                archive["phase_boundaries"]
            ).copy()
    frames = len(output["joint_position"])
    if (
        output["joint_position"].shape != (frames, 29)
        or output["root_position_world"].shape != (frames, 3)
        or output["root_orientation_world_wxyz"].shape != (frames, 4)
        or output["source_support_mask"].shape != (frames, 2)
        or output["source_support_mask"].dtype != np.bool_
        or not all(np.isfinite(value).all() for value in output.values())
        or not bool(output["source_support_mask"].any(axis=1).all())
    ):
        raise ContractError("traversal artifact arrays are invalid")
    return output


def _quaternion_slerp(
    start_wxyz: object, stop_wxyz: object, fraction: object
) -> np.ndarray:
    start = np.asarray(start_wxyz, dtype=np.float64)
    stop = np.asarray(stop_wxyz, dtype=np.float64)
    alpha = np.asarray(fraction, dtype=np.float64)
    if (
        start.shape != (4,)
        or stop.shape != (4,)
        or alpha.ndim != 1
        or len(alpha) < 1
        or not all(np.isfinite(value).all() for value in (start, stop, alpha))
        or bool(((alpha < 0.0) | (alpha > 1.0)).any())
        or abs(float(np.linalg.norm(start)) - 1.0) > 1.0e-4
        or abs(float(np.linalg.norm(stop)) - 1.0) > 1.0e-4
    ):
        raise ContractError("traversal quaternion interpolation is invalid")
    dot = float(np.dot(start, stop))
    if dot < 0.0:
        stop = -stop
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        output = (1.0 - alpha[:, None]) * start + alpha[:, None] * stop
    else:
        angle = math.acos(dot)
        sine = math.sin(angle)
        output = (
            np.sin((1.0 - alpha) * angle)[:, None] / sine * start
            + np.sin(alpha * angle)[:, None] / sine * stop
        )
    output /= np.linalg.norm(output, axis=1, keepdims=True)
    return np.ascontiguousarray(output)


def _scene_vector(
    matcher_vector: np.ndarray, yaw_scene_from_matcher: float
) -> np.ndarray:
    cosine = math.cos(yaw_scene_from_matcher)
    sine = math.sin(yaw_scene_from_matcher)
    return np.array(
        (
            cosine * matcher_vector[0] - sine * matcher_vector[1],
            sine * matcher_vector[0] + cosine * matcher_vector[1],
        ),
        dtype=np.float64,
    )


def _junction_candidates(
    *,
    incoming: dict[str, np.ndarray],
    outgoing: dict[str, np.ndarray],
    incoming_feet: np.ndarray,
    outgoing_feet: np.ndarray,
    yaw_scene_from_matcher: float,
    tail_frames: int,
) -> list[tuple[int, int, int, np.ndarray]]:
    if tail_frames < 1:
        raise ContractError("traversal junction horizon is invalid")
    candidates = []
    start_in = max(0, len(incoming_feet) - tail_frames)
    start_out = max(0, len(outgoing_feet) - tail_frames)
    for incoming_frame in range(start_in, len(incoming_feet)):
        for outgoing_frame in range(start_out, len(outgoing_feet)):
            common = (
                incoming["source_support_mask"][incoming_frame]
                & outgoing["source_support_mask"][outgoing_frame]
            )
            for foot in np.flatnonzero(common):
                shift = (
                    incoming_feet[incoming_frame, foot, :2]
                    - outgoing_feet[outgoing_frame, foot, :2]
                )
                shift_scene = _scene_vector(
                    shift, yaw_scene_from_matcher
                )
                shifted_root = (
                    outgoing["root_position_world"][outgoing_frame, :2]
                    + shift
                )
                root_error = float(
                    np.linalg.norm(
                        incoming["root_position_world"][
                            incoming_frame, :2
                        ]
                        - shifted_root
                    )
                )
                joint_error = float(
                    np.linalg.norm(
                        incoming["joint_position"][incoming_frame]
                        - outgoing["joint_position"][outgoing_frame]
                    )
                )
                score = (
                    8.0 * abs(float(shift_scene[1]))
                    + root_error
                    + 0.04 * joint_error
                    + 0.002
                    * (
                        len(incoming_feet)
                        - 1
                        - incoming_frame
                        + len(outgoing_feet)
                        - 1
                        - outgoing_frame
                    )
                )
                candidates.append(
                    (
                        score,
                        incoming_frame,
                        outgoing_frame,
                        int(foot),
                        shift,
                    )
                )
    if not candidates:
        raise ContractError("no common supported traversal junction exists")
    candidates.sort(key=lambda candidate: candidate[0])
    return [
        (
            int(candidate[1]),
            int(candidate[2]),
            int(candidate[3]),
            np.ascontiguousarray(candidate[4]),
        )
        for candidate in candidates
    ]


def _select_junction(
    *,
    incoming: dict[str, np.ndarray],
    outgoing: dict[str, np.ndarray],
    incoming_feet: np.ndarray,
    outgoing_feet: np.ndarray,
    yaw_scene_from_matcher: float,
    tail_frames: int,
) -> tuple[int, int, int, np.ndarray]:
    return _junction_candidates(
        incoming=incoming,
        outgoing=outgoing,
        incoming_feet=incoming_feet,
        outgoing_feet=outgoing_feet,
        yaw_scene_from_matcher=yaw_scene_from_matcher,
        tail_frames=tail_frames,
    )[0]


def _terrain_clearance(
    *,
    joints: np.ndarray,
    roots: np.ndarray,
    quaternions: np.ndarray,
    alignment: object,
    target_grid: object,
    foot_kinematics: MujocoG1FootKinematics,
    sole_kinematics: MujocoG1SoleKinematics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ankles = foot_kinematics.foot_positions(joints, roots, quaternions)
    soles = sole_kinematics.sole_points(joints, roots, quaternions)
    ankle_scene = alignment.matcher_to_scene_xy(
        torch.tensor(ankles[..., :2], dtype=torch.float32)
    )
    sole_scene = alignment.matcher_to_scene_xy(
        torch.tensor(soles[..., :2], dtype=torch.float32)
    )
    ankle_surface = target_grid.sample_xy(ankle_scene).cpu().numpy()
    sole_surface = target_grid.sample_xy(sole_scene).cpu().numpy()
    return (
        ankles[..., 2] - ankle_surface - float(ANKLE_ORIGIN_SOLE_M),
        soles[..., 2] - sole_surface,
        ankles,
    )


def _correct_swing_collisions(
    *,
    arrays: dict[str, np.ndarray],
    alignment: object,
    target_grid: object,
    foot_kinematics: MujocoG1FootKinematics,
    sole_kinematics: MujocoG1SoleKinematics,
    retargeter: WideBoundG1TerrainRetargeter,
) -> int:
    corrected = 0
    for frame in range(len(arrays["joint_position"])):
        changed = False
        for iteration in range(3):
            _, sole_clearance, ankles = _terrain_clearance(
                joints=arrays["joint_position"][frame : frame + 1],
                roots=arrays["root_position_world"][frame : frame + 1],
                quaternions=arrays["root_orientation_world_wxyz"][
                    frame : frame + 1
                ],
                alignment=alignment,
                target_grid=target_grid,
                foot_kinematics=foot_kinematics,
                sole_kinematics=sole_kinematics,
            )
            per_foot = sole_clearance[0].min(axis=1)
            solve = (
                ~arrays["source_support_mask"][frame]
            ) & (per_foot < -0.025)
            if not bool(solve.any()):
                break
            targets = ankles[0].copy()
            targets[solve, 2] += 0.005 - per_foot[solve]
            (
                arrays["joint_position"][frame],
                arrays["root_position_world"][frame],
            ) = retargeter.solve_frame(
                joint_position=arrays["joint_position"][frame],
                root_position_world=arrays["root_position_world"][frame],
                root_orientation_world_wxyz=(
                    arrays["root_orientation_world_wxyz"][frame]
                ),
                solve_feet=solve,
                target_foot_position_world=targets,
                level_feet=solve & arrays["source_support_mask"][frame],
                initial_joint_position=(
                    None
                    if frame == 0
                    else arrays["joint_position"][frame]
                    if iteration > 0
                    else arrays["joint_position"][frame - 1]
                ),
                initial_root_position_world=(
                    None
                    if frame == 0
                    else arrays["root_position_world"][frame]
                    if iteration > 0
                    else arrays["root_position_world"][frame - 1]
                ),
            )
            changed = True
        corrected += int(changed)
    return corrected


def _build_turn(
    *,
    incoming: dict[str, np.ndarray],
    outgoing: dict[str, np.ndarray],
    incoming_frame: int,
    outgoing_frame: int,
    stance_foot: int,
    outgoing_shift_xy: np.ndarray,
    blend_frames: int,
    foot_kinematics: MujocoG1FootKinematics,
    retargeter: WideBoundG1TerrainRetargeter,
) -> tuple[dict[str, np.ndarray], int]:
    if blend_frames < 3:
        raise ContractError("traversal turn requires at least three frames")
    shifted_outgoing = {
        name: np.asarray(value).copy() for name, value in outgoing.items()
    }
    shifted_outgoing["root_position_world"][:, :2] += outgoing_shift_xy
    start_joint = incoming["joint_position"][incoming_frame]
    stop_joint = shifted_outgoing["joint_position"][outgoing_frame]
    start_root = incoming["root_position_world"][incoming_frame]
    stop_root = shifted_outgoing["root_position_world"][outgoing_frame]
    linear = np.linspace(0.0, 1.0, blend_frames + 2)[1:-1]
    smooth = linear * linear * (3.0 - 2.0 * linear)
    blend = {
        "joint_position": (
            (1.0 - smooth[:, None]) * start_joint
            + smooth[:, None] * stop_joint
        ),
        "root_position_world": (
            (1.0 - smooth[:, None]) * start_root
            + smooth[:, None] * stop_root
        ),
        "root_orientation_world_wxyz": _quaternion_slerp(
            incoming["root_orientation_world_wxyz"][incoming_frame],
            shifted_outgoing["root_orientation_world_wxyz"][
                outgoing_frame
            ],
            smooth,
        ),
        "source_support_mask": np.zeros(
            (blend_frames, 2), dtype=np.bool_
        ),
    }
    blend["source_support_mask"][:, stance_foot] = True
    stance_target = foot_kinematics.foot_positions(
        incoming["joint_position"][incoming_frame : incoming_frame + 1],
        incoming["root_position_world"][incoming_frame : incoming_frame + 1],
        incoming["root_orientation_world_wxyz"][
            incoming_frame : incoming_frame + 1
        ],
    )[0, stance_foot]
    for frame in range(blend_frames):
        feet = foot_kinematics.foot_positions(
            blend["joint_position"][frame : frame + 1],
            blend["root_position_world"][frame : frame + 1],
            blend["root_orientation_world_wxyz"][frame : frame + 1],
        )[0]
        targets = feet.copy()
        targets[stance_foot] = stance_target
        (
            blend["joint_position"][frame],
            blend["root_position_world"][frame],
        ) = retargeter.solve_frame(
            joint_position=blend["joint_position"][frame],
            root_position_world=blend["root_position_world"][frame],
            root_orientation_world_wxyz=(
                blend["root_orientation_world_wxyz"][frame]
            ),
            solve_feet=np.array(
                (stance_foot == 0, stance_foot == 1), dtype=np.bool_
            ),
            target_foot_position_world=targets,
            level_feet=np.array(
                (stance_foot == 0, stance_foot == 1), dtype=np.bool_
            ),
            initial_joint_position=(
                None
                if frame == 0
                else blend["joint_position"][frame - 1]
            ),
            initial_root_position_world=(
                None
                if frame == 0
                else blend["root_position_world"][frame - 1]
            ),
        )
    prefix = slice(0, incoming_frame + 1)
    # Reverse the outgoing motion so the intersection becomes its entrance.
    suffix_indices = np.arange(outgoing_frame - 1, -1, -1)
    result = {}
    for name in (
        "joint_position",
        "root_position_world",
        "root_orientation_world_wxyz",
        "source_support_mask",
    ):
        result[name] = np.ascontiguousarray(
            np.concatenate(
                (
                    incoming[name][prefix],
                    blend[name],
                    shifted_outgoing[name][suffix_indices],
                ),
                axis=0,
            )
        )
    return result, len(blend["joint_position"])


def main() -> int:
    args = _parser().parse_args()
    if args.blend_frames < 3 or args.junction_tail_frames < 1:
        raise ContractError("traversal transition thresholds are invalid")
    incoming = _load_artifact(args.incoming)
    outgoing = _load_artifact(args.outgoing)
    resolved = resolve_stair_config(
        args.target_dataset,
        load_experiment_config(args.config),
        device="cpu",
    )
    extension = resolved.measurement_extension
    foot_kinematics = MujocoG1FootKinematics(args.g1_xml)
    sole_kinematics = MujocoG1SoleKinematics(args.g1_xml)
    incoming_feet = foot_kinematics.foot_positions(
        incoming["joint_position"],
        incoming["root_position_world"],
        incoming["root_orientation_world_wxyz"],
    )
    outgoing_feet = foot_kinematics.foot_positions(
        outgoing["joint_position"],
        outgoing["root_position_world"],
        outgoing["root_orientation_world_wxyz"],
    )
    candidates = _junction_candidates(
        incoming=incoming,
        outgoing=outgoing,
        incoming_feet=incoming_feet,
        outgoing_feet=outgoing_feet,
        yaw_scene_from_matcher=float(
            extension.alignment.yaw_scene_from_matcher
        ),
        tail_frames=args.junction_tail_frames,
    )
    retargeter = WideBoundG1TerrainRetargeter(
        args.g1_xml,
        maximum_root_height_deviation_m=0.08,
        maximum_root_horizontal_deviation_m=0.12,
        maximum_target_error_m=0.005,
    )
    selected = None
    incoming_elevated_start = int(
        incoming.get("phase_boundaries", np.array((0,)))[1]
        if len(incoming.get("phase_boundaries", ())) >= 2
        else 0
    )
    outgoing_elevated_start = int(
        outgoing.get("phase_boundaries", np.array((0,)))[1]
        if len(outgoing.get("phase_boundaries", ())) >= 2
        else 0
    )
    for incoming_frame, outgoing_frame, stance_foot, shift_xy in candidates:
        if (
            incoming_frame < incoming_elevated_start
            or outgoing_frame < outgoing_elevated_start
        ):
            continue
        # Reject a junction before IK if translating its outgoing suffix
        # would move an otherwise-valid step into a neighboring riser.
        shifted_roots = outgoing["root_position_world"][
            : outgoing_frame + 1
        ].copy()
        shifted_roots[:, :2] += shift_xy
        suffix_ankle, suffix_sole, _ = _terrain_clearance(
            joints=outgoing["joint_position"][: outgoing_frame + 1],
            roots=shifted_roots,
            quaternions=outgoing["root_orientation_world_wxyz"][
                : outgoing_frame + 1
            ],
            alignment=extension.alignment,
            target_grid=extension.query_grid,
            foot_kinematics=foot_kinematics,
            sole_kinematics=sole_kinematics,
        )
        suffix_support = outgoing["source_support_mask"][
            : outgoing_frame + 1
        ]
        suffix_points = (
            (suffix_sole >= -0.025) & (suffix_sole <= 0.035)
        ).sum(axis=2)
        if (
            float(np.abs(suffix_ankle[suffix_support]).max()) > 0.020
            or float(suffix_sole.min()) < -0.025
            or bool((suffix_points[suffix_support] < 3).any())
        ):
            continue
        try:
            route, blend_count = _build_turn(
                incoming=incoming,
                outgoing=outgoing,
                incoming_frame=incoming_frame,
                outgoing_frame=outgoing_frame,
                stance_foot=stance_foot,
                outgoing_shift_xy=shift_xy,
                blend_frames=args.blend_frames,
                foot_kinematics=foot_kinematics,
                retargeter=retargeter,
            )
            corrected = _correct_swing_collisions(
                arrays=route,
                alignment=extension.alignment,
                target_grid=extension.query_grid,
                foot_kinematics=foot_kinematics,
                sole_kinematics=sole_kinematics,
                retargeter=retargeter,
            )
        except ContractError:
            continue
        ankle_clearance, sole_clearance, _ = _terrain_clearance(
            joints=route["joint_position"],
            roots=route["root_position_world"],
            quaternions=route["root_orientation_world_wxyz"],
            alignment=extension.alignment,
            target_grid=extension.query_grid,
            foot_kinematics=foot_kinematics,
            sole_kinematics=sole_kinematics,
        )
        support = route["source_support_mask"]
        maximum_stance_error = float(
            np.abs(ankle_clearance[support]).max()
        )
        minimum_sole_clearance = float(sole_clearance.min())
        supported_points = (
            (sole_clearance >= -0.025) & (sole_clearance <= 0.035)
        ).sum(axis=2)
        if (
            bool(support.any(axis=1).all())
            and maximum_stance_error <= 0.020
            and minimum_sole_clearance >= -0.025
            and not bool((supported_points[support] < 3).any())
        ):
            selected = (
                incoming_frame,
                outgoing_frame,
                stance_foot,
                shift_xy,
            )
            break
    if selected is None:
        raise ContractError(
            "no terrain-valid supported traversal junction exists"
        )
    metrics = {
        "schema": "g1-supported-traversal-transition/v1",
        "incoming_frame": incoming_frame,
        "outgoing_frame": outgoing_frame,
        "stance_foot": stance_foot,
        "blend_frame_count": blend_count,
        "frame_count": len(route["joint_position"]),
        "unsupported_frame_count": int(
            (~support.any(axis=1)).sum()
        ),
        "maximum_stance_contact_error_m": maximum_stance_error,
        "minimum_sole_clearance_m": minimum_sole_clearance,
        "swing_corrected_frame_count": corrected,
        "maximum_joint_step_rad": float(
            np.abs(np.diff(route["joint_position"], axis=0)).max()
        ),
        "maximum_root_step_m": float(
            np.linalg.norm(
                np.diff(route["root_position_world"], axis=0), axis=1
            ).max()
        ),
        "outgoing_shift_matcher_xy": shift_xy.tolist(),
        "outgoing_shift_scene_xy": _scene_vector(
            shift_xy,
            float(extension.alignment.yaw_scene_from_matcher),
        ).tolist(),
    }
    route["phase_boundaries"] = np.asarray(
        (
            0,
            incoming_frame,
            incoming_frame + blend_count,
            len(route["joint_position"]),
        ),
        dtype=np.int64,
    )
    route["minimum_sole_clearance_by_frame"] = sole_clearance.min(
        axis=(1, 2)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "transition.npz", **route)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
