#!/usr/bin/env python3
"""Synthesize one deterministic, speed-bounded stair motion connector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from mm_sonic.joints import ContractError, load_joint_contract
from mm_sonic.torch_g1_fk import MujocoG1FootKinematics
from mm_sonic.torch_stair_connector import (
    BoundedFootKinematics,
    StairConnectorBoundary,
    synthesize_double_support_connector,
)


DT_S = 0.02


def _boundary_from_artifact(
    arrays: Mapping[str, object],
    *,
    frame: int,
    side: Literal["incoming", "outgoing"],
    dt_s: float,
) -> StairConnectorBoundary:
    joints = np.asarray(arrays["joint_position"], dtype=np.float64)
    roots = np.asarray(arrays["root_position_world"], dtype=np.float64)
    qpos = np.asarray(arrays["qpos"], dtype=np.float64)
    feet = np.asarray(arrays["foot_position_world"], dtype=np.float64)
    if (
        joints.ndim != 2
        or joints.shape[1] != 29
        or roots.shape != (len(joints), 3)
        or qpos.ndim != 2
        or qpos.shape[0] != len(joints)
        or qpos.shape[1] < 7
        or feet.shape != (len(joints), 2, 3)
        or side not in ("incoming", "outgoing")
        or not 0 < frame < len(joints) - 1
        or not math.isfinite(float(dt_s))
        or float(dt_s) <= 0.0
    ):
        raise ContractError("stair connector artifact boundary is invalid")
    neighbor = frame - 1 if side == "incoming" else frame + 1
    sign = 1.0 if side == "incoming" else -1.0
    return StairConnectorBoundary(
        joint_position=joints[frame],
        joint_velocity=(joints[frame] - joints[neighbor])
        / float(dt_s)
        * sign,
        root_position_world=roots[frame],
        root_velocity_world=(roots[frame] - roots[neighbor])
        / float(dt_s)
        * sign,
        root_orientation_world_wxyz=qpos[frame, 3:7],
        foot_position_world=feet[frame],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--incoming-edge", required=True)
    parser.add_argument("--outgoing-edge", required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument(
        "--joint-contract",
        type=Path,
        default=Path("sonic/configs/g1_joint_contract.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=26)
    parser.add_argument("--maximum-joint-speed-rad-s", type=float, default=13.0)
    parser.add_argument("--maximum-foot-error-m", type=float, default=0.005)
    parser.add_argument("--context-frames", type=int, default=50)
    return parser


def _select_edge(edges: list[dict], identity: str) -> dict:
    matches = [
        edge
        for edge in edges
        if str(edge.get("edge_id", "")).startswith(identity)
    ]
    if len(matches) != 1:
        raise ContractError(
            f"stair connector edge identity is not unique: {identity}"
        )
    return matches[0]


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}
    except Exception as error:
        raise ContractError(f"cannot load connector source: {path}") from error


def _joint_limits(path: Path) -> tuple[np.ndarray, np.ndarray]:
    contract = load_joint_contract(path)
    lower = np.empty(29, dtype=np.float64)
    upper = np.empty(29, dtype=np.float64)
    for row in contract.rows:
        lower[row.target_index] = row.lower
        upper[row.target_index] = row.upper
    return lower, upper


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _aligned_outgoing(
    arrays: Mapping[str, np.ndarray],
    *,
    start: int,
    count: int,
    rotation_xy: np.ndarray,
    translation_xy: np.ndarray,
    yaw_rad: float,
) -> dict[str, np.ndarray]:
    stop = min(len(arrays["joint_position"]), start + count)
    selection = slice(start, stop)
    roots = np.asarray(arrays["root_position_world"][selection]).copy()
    roots[:, :2] = roots[:, :2] @ rotation_xy + translation_xy
    feet = np.asarray(arrays["foot_position_world"][selection]).copy()
    feet[:, :, :2] = feet[:, :, :2] @ rotation_xy + translation_xy
    half = 0.5 * yaw_rad
    rotation = np.asarray(
        (math.cos(half), 0.0, 0.0, math.sin(half)),
        dtype=np.float64,
    )
    orientations = np.asarray(arrays["qpos"][selection, 3:7]).copy()
    orientations = np.asarray(
        [_quaternion_multiply(rotation, value) for value in orientations]
    )
    orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)
    return {
        "joint_position": np.asarray(
            arrays["joint_position"][selection],
            dtype=np.float64,
        ),
        "root_position_world": roots,
        "root_orientation_world_wxyz": orientations,
        "foot_position_world": feet,
    }


def _deterministic_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(b"g1-stair-connector/v1")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _minimum_sole_clearance_by_frame(
    sole_points_world: object,
    surface_height_m: object,
) -> np.ndarray:
    sole = np.asarray(sole_points_world, dtype=np.float64)
    surface = np.asarray(surface_height_m, dtype=np.float64)
    if (
        sole.ndim != 4
        or sole.shape[1] != 2
        or sole.shape[3] != 3
        or surface.shape != sole.shape[:3]
        or not np.isfinite(sole).all()
        or not np.isfinite(surface).all()
    ):
        raise ContractError("stair connector sole clearance input is invalid")
    return np.ascontiguousarray(
        (sole[..., 2] - surface).min(axis=(1, 2))
    )


def main() -> int:
    args = _parser().parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    edges = graph.get("edges")
    sources = graph.get("source", {}).get("accepted_routes")
    if not isinstance(edges, list) or not isinstance(sources, list):
        raise ContractError("stair connector graph is invalid")
    incoming_edge = _select_edge(edges, args.incoming_edge)
    outgoing_edge = _select_edge(edges, args.outgoing_edge)
    if (
        (
            int(outgoing_edge["traversal_heading_bin"])
            - int(incoming_edge["traversal_heading_bin"])
        )
        % 8
        not in (2, 6)
    ):
        raise ContractError("stair connector pair is not a 90-degree change")
    source_by_sha = {
        item["route_deterministic_sha256"]: item
        for item in sources
    }
    incoming_source = source_by_sha[incoming_edge["source_artifact_sha256"]]
    outgoing_source = source_by_sha[outgoing_edge["source_artifact_sha256"]]
    incoming_arrays = _load_arrays(
        Path(incoming_source["route_directory"]) / "arrays.npz"
    )
    outgoing_arrays = _load_arrays(
        Path(outgoing_source["route_directory"]) / "arrays.npz"
    )
    incoming_frame = int(incoming_edge["source_end_frame_exclusive"]) - 1
    outgoing_frame = int(outgoing_edge["source_start_frame"])
    incoming = _boundary_from_artifact(
        incoming_arrays,
        frame=incoming_frame,
        side="incoming",
        dt_s=DT_S,
    )
    outgoing = _boundary_from_artifact(
        outgoing_arrays,
        frame=outgoing_frame,
        side="outgoing",
        dt_s=DT_S,
    )
    lower, upper = _joint_limits(args.joint_contract)
    kinematics = BoundedFootKinematics(
        MujocoG1FootKinematics(args.g1_xml)
    )
    connector = synthesize_double_support_connector(
        incoming,
        outgoing,
        kinematics=kinematics,
        frame_count=args.frame_count,
        dt_s=DT_S,
        maximum_joint_speed_rad_s=args.maximum_joint_speed_rad_s,
        maximum_foot_error_m=args.maximum_foot_error_m,
        joint_position_lower=lower,
        joint_position_upper=upper,
    )
    alignment = connector.outgoing_alignment
    outgoing_preview = _aligned_outgoing(
        outgoing_arrays,
        start=outgoing_frame,
        count=args.context_frames + 1,
        rotation_xy=alignment.rotation_xy,
        translation_xy=alignment.translation_xy,
        yaw_rad=alignment.yaw_rad,
    )
    joint_seam = (
        outgoing_preview["joint_position"][0]
        - connector.joint_position[-1]
    )
    foot_seam = (
        outgoing_preview["foot_position_world"][0]
        - connector.foot_position_world[-1]
    )
    joint_seam_speed = float(np.abs(joint_seam).max() / DT_S)
    foot_seam_error = float(
        np.linalg.norm(foot_seam, axis=1).max()
    )
    outgoing_future_speed = float(
        np.abs(
            outgoing_preview["joint_position"][1]
            - outgoing_preview["joint_position"][0]
        ).max()
        / DT_S
    )
    from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
    from mm_sonic.torch_terrain_rollout import (
        load_experiment_config,
        resolve_stair_config,
    )
    import torch

    resolved = resolve_stair_config(
        args.dataset,
        load_experiment_config(args.config),
        device="cpu",
    )
    sole_points = MujocoG1SoleKinematics(args.g1_xml).sole_points(
        connector.joint_position,
        connector.root_position_world,
        connector.root_orientation_world_wxyz,
    )
    sole_xy = torch.as_tensor(sole_points[..., :2], dtype=torch.float32)
    sole_surface = (
        resolved.measurement_extension.query_grid.sample_xy(
            resolved.measurement_extension.alignment.matcher_to_scene_xy(
                sole_xy
            )
        )
        .detach()
        .cpu()
        .numpy()
    )
    sole_clearance = _minimum_sole_clearance_by_frame(
        sole_points,
        sole_surface,
    )
    minimum_sole_clearance = float(sole_clearance.min())
    accepted = (
        connector.maximum_foot_error_m <= args.maximum_foot_error_m
        and connector.maximum_joint_speed_rad_s
        <= args.maximum_joint_speed_rad_s
        and foot_seam_error <= args.maximum_foot_error_m
        and joint_seam_speed <= args.maximum_joint_speed_rad_s
        and outgoing_future_speed <= args.maximum_joint_speed_rad_s
        and minimum_sole_clearance >= -0.025
    )
    connector_arrays = {
        "joint_position": connector.joint_position,
        "joint_velocity": connector.joint_velocity,
        "root_position_world": connector.root_position_world,
        "root_orientation_world_wxyz": (
            connector.root_orientation_world_wxyz
        ),
        "foot_position_world": connector.foot_position_world,
    }
    connector_sha256 = _deterministic_sha256(connector_arrays)
    start_context = max(0, incoming_frame - args.context_frames)
    incoming_selection = slice(start_context, incoming_frame + 1)
    preview = {
        "joint_position": np.concatenate(
            (
                incoming_arrays["joint_position"][incoming_selection],
                connector.joint_position[1:],
                outgoing_preview["joint_position"][1:],
            )
        ),
        "root_position_world": np.concatenate(
            (
                incoming_arrays["root_position_world"][incoming_selection],
                connector.root_position_world[1:],
                outgoing_preview["root_position_world"][1:],
            )
        ),
        "root_orientation_world_wxyz": np.concatenate(
            (
                incoming_arrays["qpos"][incoming_selection, 3:7],
                connector.root_orientation_world_wxyz[1:],
                outgoing_preview["root_orientation_world_wxyz"][1:],
            )
        ),
        "foot_position_world": np.concatenate(
            (
                incoming_arrays["foot_position_world"][incoming_selection],
                connector.foot_position_world[1:],
                outgoing_preview["foot_position_world"][1:],
            )
        ),
    }
    preview["phase"] = np.concatenate(
        (
            np.zeros(incoming_frame + 1 - start_context, dtype=np.int8),
            np.ones(args.frame_count - 1, dtype=np.int8),
            np.full(
                len(outgoing_preview["joint_position"]) - 1,
                2,
                dtype=np.int8,
            ),
        )
    )
    metrics = {
        "schema": "g1-stair-connector-metrics/v1",
        "accepted": accepted,
        "connector_sha256": connector_sha256,
        "incoming_edge_id": incoming_edge["edge_id"],
        "outgoing_edge_id": outgoing_edge["edge_id"],
        "incoming_heading_bin": incoming_edge["traversal_heading_bin"],
        "outgoing_heading_bin": outgoing_edge["traversal_heading_bin"],
        "frame_count": args.frame_count,
        "duration_s": (args.frame_count - 1) * DT_S,
        "maximum_joint_speed_rad_s": (
            connector.maximum_joint_speed_rad_s
        ),
        "maximum_foot_error_m": connector.maximum_foot_error_m,
        "alignment_residual_xy_m": alignment.maximum_residual_m,
        "outgoing_joint_seam_speed_rad_s": joint_seam_speed,
        "outgoing_foot_seam_error_m": foot_seam_error,
        "outgoing_future_speed_rad_s": outgoing_future_speed,
        "minimum_sole_clearance_m": minimum_sole_clearance,
        "terminal_joint_rms_deviation_rad": float(
            np.sqrt(np.mean(joint_seam * joint_seam))
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "connector.npz", **connector_arrays)
    np.savez_compressed(args.output / "preview.npz", **preview)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True))
    if not accepted:
        raise ContractError("stair connector failed the acceptance contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
