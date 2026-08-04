#!/usr/bin/env python3
"""Search a short offline graph walk of sole-valid stair pivot steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from mm_sonic.joints import ContractError, load_joint_contract
from mm_sonic.torch_g1_fk import (
    MujocoG1FootKinematics,
    target_state_qpos,
)
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_stair_connector import (
    BoundedFootKinematics,
    StairConnectorBoundary,
    single_step_sole_trajectory,
    synthesize_single_step_connector,
)
from mm_sonic.torch_stair_motion_graph import (
    StairContactNode,
    StairFoothold,
    StairMotionEdge,
    heading_bin_from_yaw,
    merge_stair_motion_graphs,
    stair_motion_graph_from_dict,
)
from mm_sonic.torch_stair_pivot_search import pivot_target_lattice
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)


DT_S = 0.02
SOLE_CLEARANCE_LIMIT_M = -0.025


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--incoming-edge", required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument(
        "--joint-contract",
        type=Path,
        default=Path("sonic/configs/g1_joint_contract.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    yaw = parser.add_mutually_exclusive_group(required=True)
    yaw.add_argument("--total-yaw-delta-rad", type=float)
    yaw.add_argument(
        "--step-yaw-delta-rad",
        type=float,
        action="append",
        help="Explicit per-step yaw deltas for a direction-changing path.",
    )
    parser.add_argument("--step-count", type=int, default=2)
    parser.add_argument("--frame-count-per-step", type=int, default=61)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--swing-clearance-m", type=float, default=0.08)
    parser.add_argument(
        "--root-height-offset-m",
        type=float,
        action="append",
        help=(
            "Per-step pelvis-height candidates. Repeat to search a lattice; "
            "the default is 0."
        ),
    )
    parser.add_argument("--maximum-joint-speed-rad-s", type=float, default=13.0)
    parser.add_argument(
        "--maximum-joint-acceleration-rad-s2",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--maximum-terminal-sole-contact-error-m",
        type=float,
        default=0.025,
    )
    parser.add_argument("--minimum-stance-width-m", type=float, default=0.18)
    parser.add_argument("--maximum-stance-width-m", type=float, default=0.40)
    parser.add_argument("--target-root-displacement-u-m", type=float)
    parser.add_argument("--target-root-displacement-v-m", type=float)
    parser.add_argument(
        "--step-target-root-u-m",
        type=float,
        action="append",
        help="Explicit cumulative stair-u root target after each step.",
    )
    parser.add_argument(
        "--step-target-root-v-m",
        type=float,
        action="append",
        help="Explicit cumulative stair-v root target after each step.",
    )
    parser.add_argument(
        "--maximum-terminal-displacement-error-m",
        type=float,
        default=0.03,
    )
    parser.add_argument("--offset-min-u-m", type=float, default=-0.24)
    parser.add_argument("--offset-max-u-m", type=float, default=0.08)
    parser.add_argument("--offset-min-v-m", type=float, default=-0.16)
    parser.add_argument("--offset-max-v-m", type=float, default=0.08)
    parser.add_argument("--offset-step-m", type=float, default=0.08)
    return parser


def _select_edge(edges: object, identity: str) -> dict:
    if not isinstance(edges, list):
        raise ContractError("stair pivot graph edges are invalid")
    matches = [
        edge
        for edge in edges
        if isinstance(edge, dict)
        and str(edge.get("edge_id", "")).startswith(identity)
    ]
    if len(matches) != 1:
        raise ContractError(
            f"stair pivot edge identity is not unique: {identity}"
        )
    return matches[0]


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}
    except Exception as error:
        raise ContractError(f"cannot load stair pivot source: {path}") from error


def _boundary(
    arrays: dict[str, np.ndarray],
    frame: int,
    *,
    stationary: bool = False,
) -> StairConnectorBoundary:
    joints = np.asarray(arrays["joint_position"], dtype=np.float64)
    roots = np.asarray(arrays["root_position_world"], dtype=np.float64)
    feet = np.asarray(arrays["foot_position_world"], dtype=np.float64)
    orientations = np.asarray(
        arrays["root_orientation_world_wxyz"]
        if "root_orientation_world_wxyz" in arrays
        else np.asarray(arrays["qpos"])[:, 3:7],
        dtype=np.float64,
    )
    if not 0 < frame < len(joints):
        raise ContractError("stair pivot boundary frame is invalid")
    return StairConnectorBoundary(
        joint_position=joints[frame],
        joint_velocity=(
            np.zeros(29, dtype=np.float64)
            if stationary
            else (
                np.asarray(arrays["joint_velocity"][frame], dtype=np.float64)
                if "joint_velocity" in arrays
                else (joints[frame] - joints[frame - 1]) / DT_S
            )
        ),
        root_position_world=roots[frame],
        root_velocity_world=(
            np.zeros(3, dtype=np.float64)
            if stationary
            else (roots[frame] - roots[frame - 1]) / DT_S
        ),
        root_orientation_world_wxyz=orientations[frame],
        foot_position_world=feet[frame],
    )


def _joint_limits(path: Path) -> tuple[np.ndarray, np.ndarray]:
    contract = load_joint_contract(path)
    lower = np.empty(29, dtype=np.float64)
    upper = np.empty(29, dtype=np.float64)
    for row in contract.rows:
        lower[row.target_index] = row.lower
        upper[row.target_index] = row.upper
    return lower, upper


def _offset_values(minimum: float, maximum: float, step: float) -> tuple[float, ...]:
    if (
        not all(math.isfinite(value) for value in (minimum, maximum, step))
        or step <= 0.0
        or minimum > maximum
    ):
        raise ContractError("stair pivot offset lattice is invalid")
    count = int(math.floor((maximum - minimum) / step + 1e-9))
    values = [minimum + index * step for index in range(count + 1)]
    if values[-1] < maximum - 1e-9:
        values.append(maximum)
    return tuple(float(value) for value in values)


def _valid_stance_width(
    foot_xy: object,
    *,
    minimum_m: float,
    maximum_m: float,
) -> bool:
    feet = np.asarray(foot_xy, dtype=np.float64)
    if (
        feet.shape != (2, 2)
        or not np.isfinite(feet).all()
        or not math.isfinite(float(minimum_m))
        or not math.isfinite(float(maximum_m))
        or float(minimum_m) <= 0.0
        or float(minimum_m) > float(maximum_m)
    ):
        raise ContractError("stair pivot stance-width inputs are invalid")
    distance = float(np.linalg.norm(feet[1] - feet[0]))
    return float(minimum_m) <= distance <= float(maximum_m)


def _stair_offsets_world(
    offsets_stair_uv: object,
    *,
    ascent_world_xy: object,
) -> tuple[tuple[float, float], ...]:
    offsets = np.asarray(offsets_stair_uv, dtype=np.float64)
    ascent = np.asarray(ascent_world_xy, dtype=np.float64)
    norm = float(np.linalg.norm(ascent))
    if (
        offsets.ndim != 2
        or offsets.shape[1] != 2
        or len(offsets) < 1
        or not np.isfinite(offsets).all()
        or ascent.shape != (2,)
        or not np.isfinite(ascent).all()
        or norm <= 1e-8
    ):
        raise ContractError("stair pivot coordinate frame is invalid")
    ascent /= norm
    lateral = np.asarray((-ascent[1], ascent[0]), dtype=np.float64)
    world = (
        offsets[:, :1] * ascent[None, :]
        + offsets[:, 1:] * lateral[None, :]
    )
    return tuple((float(value[0]), float(value[1])) for value in world)


def _world_displacement_to_stair(
    displacement_world_xy: object,
    *,
    ascent_world_xy: object,
) -> np.ndarray:
    displacement = np.asarray(displacement_world_xy, dtype=np.float64)
    ascent = np.asarray(ascent_world_xy, dtype=np.float64)
    norm = float(np.linalg.norm(ascent))
    if (
        displacement.shape != (2,)
        or not np.isfinite(displacement).all()
        or ascent.shape != (2,)
        or not np.isfinite(ascent).all()
        or norm <= 1e-8
    ):
        raise ContractError("stair pivot displacement frame is invalid")
    ascent /= norm
    lateral = np.asarray((-ascent[1], ascent[0]), dtype=np.float64)
    return np.asarray(
        (float(displacement @ ascent), float(displacement @ lateral)),
        dtype=np.float64,
    )


def _authenticated_graph_ascent_world_xy(
    graph_source: object,
    *,
    dataset_manifest_sha256: str,
) -> np.ndarray:
    if (
        not isinstance(graph_source, dict)
        or graph_source.get("dataset_manifest_sha256")
        != dataset_manifest_sha256
    ):
        raise ContractError("stair pivot graph dataset manifest is invalid")
    frame = graph_source.get("stair_frame")
    if not isinstance(frame, dict):
        raise ContractError("stair pivot graph frame is missing")
    try:
        origin = np.asarray(frame["origin_world_xy"], dtype=np.float64)
        yaw = float(frame["ascent_world_yaw"])
        dimensions = (
            float(frame["width_m"]),
            float(frame["tread_depth_m"]),
            float(frame["riser_height_m"]),
            float(frame["lateral_cell_m"]),
        )
        tread_count = int(frame["tread_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("stair pivot graph frame is invalid") from error
    if (
        origin.shape != (2,)
        or not np.isfinite(origin).all()
        or not math.isfinite(yaw)
        or not all(math.isfinite(value) and value > 0.0 for value in dimensions)
        or tread_count < 1
    ):
        raise ContractError("stair pivot graph frame is invalid")
    return np.asarray((math.cos(yaw), math.sin(yaw)), dtype=np.float64)


def _quantize(value: float, step: float) -> int:
    return int(math.floor(float(value) / float(step) + 0.5))


def _yaw_from_wxyz(quaternion: object) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _optimized_boundary_sha256(
    *,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    root_position_world: np.ndarray,
    root_orientation_world_wxyz: np.ndarray,
    foot_position_world: np.ndarray,
    foot_surface_height_m: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"g1-stair-optimized-boundary/v1")
    qpos = target_state_qpos(
        joint_position,
        root_position_world,
        root_orientation_world_wxyz,
    )
    for value in (
        qpos,
        joint_velocity,
        foot_position_world,
        foot_surface_height_m,
        np.asarray((True, True), dtype=np.bool_),
    ):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _verified_optimized_boundary_sha256(
    *,
    expected_sha256: str | None,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    root_position_world: np.ndarray,
    root_orientation_world_wxyz: np.ndarray,
    foot_position_world: np.ndarray,
    foot_surface_height_m: np.ndarray,
) -> str:
    actual = _optimized_boundary_sha256(
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        root_position_world=root_position_world,
        root_orientation_world_wxyz=root_orientation_world_wxyz,
        foot_position_world=foot_position_world,
        foot_surface_height_m=foot_surface_height_m,
    )
    if expected_sha256 is not None and actual != expected_sha256:
        raise ContractError(
            "stair pivot emitted start boundary does not match predecessor"
        )
    return actual


def _optimized_contact_node(
    *,
    root_position_world: np.ndarray,
    root_orientation_world_wxyz: np.ndarray,
    foot_position_world: np.ndarray,
    foot_surface_height_m: np.ndarray,
    graph_source: dict,
    last_landing_foot: int,
) -> StairContactNode:
    frame = graph_source["stair_frame"]
    ascent_yaw = float(frame["ascent_world_yaw"])
    ascent = np.asarray(
        (math.cos(ascent_yaw), math.sin(ascent_yaw)),
        dtype=np.float64,
    )
    origin = np.asarray(frame["origin_world_xy"], dtype=np.float64)
    root_stair = _world_displacement_to_stair(
        root_position_world[:2] - origin,
        ascent_world_xy=ascent,
    )
    feet_stair = np.stack(
        [
            _world_displacement_to_stair(
                foot_position_world[foot, :2] - origin,
                ascent_world_xy=ascent,
            )
            for foot in range(2)
        ]
    )
    riser = float(frame["riser_height_m"])
    lateral = float(frame["lateral_cell_m"])
    return StairContactNode(
        root_u_cell=_quantize(root_stair[0], float(frame["tread_depth_m"])),
        root_v_cell=_quantize(root_stair[1], lateral),
        root_height_level=max(0, _quantize(root_position_world[2], riser)),
        heading_bin=heading_bin_from_yaw(
            _yaw_from_wxyz(root_orientation_world_wxyz) - ascent_yaw
        ),
        velocity_bin=0,
        left_foothold=StairFoothold(
            _quantize(foot_surface_height_m[0], riser),
            _quantize(feet_stair[0, 1], lateral),
        ),
        right_foothold=StairFoothold(
            _quantize(foot_surface_height_m[1], riser),
            _quantize(feet_stair[1, 1], lateral),
        ),
        support_mask=3,
        last_landing_foot=last_landing_foot,
        gait_phase_bin=0 if last_landing_foot in (-1, 0) else 4,
    )


def _target_displacement(args: argparse.Namespace) -> np.ndarray | None:
    has_u = args.target_root_displacement_u_m is not None
    has_v = args.target_root_displacement_v_m is not None
    if has_u != has_v:
        raise ContractError(
            "stair pivot target displacement coordinates must be paired"
        )
    if not has_u:
        return None
    target = np.asarray(
        (
            args.target_root_displacement_u_m,
            args.target_root_displacement_v_m,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(target).all():
        raise ContractError("stair pivot target displacement is invalid")
    return target


def _step_schedule(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray | None]:
    if args.step_yaw_delta_rad is None:
        yaw = np.full(
            args.step_count,
            float(args.total_yaw_delta_rad) / float(args.step_count),
            dtype=np.float64,
        )
    else:
        yaw = np.asarray(args.step_yaw_delta_rad, dtype=np.float64)
        if yaw.shape != (args.step_count,):
            raise ContractError(
                "stair pivot yaw schedule must match step count"
            )
    if not np.isfinite(yaw).all():
        raise ContractError("stair pivot yaw schedule is invalid")

    has_step_u = args.step_target_root_u_m is not None
    has_step_v = args.step_target_root_v_m is not None
    if has_step_u != has_step_v:
        raise ContractError(
            "stair pivot step target coordinates must be paired"
        )
    target = _target_displacement(args)
    if has_step_u and target is not None:
        raise ContractError(
            "stair pivot aggregate and per-step targets are exclusive"
        )
    if has_step_u:
        targets = np.column_stack(
            (
                np.asarray(args.step_target_root_u_m, dtype=np.float64),
                np.asarray(args.step_target_root_v_m, dtype=np.float64),
            )
        )
        if targets.shape != (args.step_count, 2):
            raise ContractError(
                "stair pivot target schedule must match step count"
            )
    elif target is not None:
        targets = np.asarray(
            [
                target * float(index + 1) / float(args.step_count)
                for index in range(args.step_count)
            ],
            dtype=np.float64,
        )
    else:
        targets = None
    if targets is not None and not np.isfinite(targets).all():
        raise ContractError("stair pivot target schedule is invalid")
    return yaw, targets


def _connector_arrays(connector: object) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(connector, name), dtype=np.float64)
        for name in (
            "joint_position",
            "joint_velocity",
            "root_position_world",
            "root_orientation_world_wxyz",
            "foot_position_world",
        )
    }


def _pin_connector_start(
    arrays: dict[str, np.ndarray],
    boundary: StairConnectorBoundary,
) -> None:
    arrays["joint_position"][0] = boundary.joint_position
    arrays["joint_velocity"][0] = boundary.joint_velocity
    arrays["root_position_world"][0] = boundary.root_position_world
    arrays["root_orientation_world_wxyz"][0] = (
        boundary.root_orientation_world_wxyz
    )
    arrays["foot_position_world"][0] = boundary.foot_position_world


def _join(
    previous: dict[str, np.ndarray] | None,
    current: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if previous is None:
        joined = {name: value.copy() for name, value in current.items()}
    else:
        joined = {
            name: np.concatenate((previous[name], value[1:]))
            for name, value in current.items()
        }
    if "joint_position" in joined and "joint_velocity" in joined:
        joined["joint_velocity"] = np.gradient(
            joined["joint_position"],
            DT_S,
            axis=0,
            edge_order=2 if len(joined["joint_position"]) >= 3 else 1,
        )
    return joined


def _landing_segment_ranges(
    *,
    total_frame_count: int,
    frame_count_per_step: int,
    step_count: int,
) -> tuple[tuple[int, int], ...]:
    if frame_count_per_step < 2 or step_count < 1:
        raise ContractError("stair pivot segment frame count is invalid")
    expected = 1 + step_count * (frame_count_per_step - 1)
    if total_frame_count != expected:
        raise ContractError(
            "stair pivot joined frame count does not match its step schedule"
        )
    stride = frame_count_per_step - 1
    return tuple(
        (
            step_index * stride,
            step_index * stride + frame_count_per_step,
        )
        for step_index in range(step_count)
    )


def _maximum_joint_acceleration_rad_s2(
    joint_position: object,
    *,
    dt_s: float,
) -> float:
    joints = np.asarray(joint_position, dtype=np.float64)
    if (
        joints.ndim != 2
        or joints.shape[1] != 29
        or not np.isfinite(joints).all()
        or not math.isfinite(float(dt_s))
        or float(dt_s) <= 0.0
    ):
        raise ContractError("stair pivot acceleration inputs are invalid")
    if len(joints) < 3:
        return 0.0
    return float(
        np.abs(np.diff(joints, n=2, axis=0)).max()
        / (float(dt_s) * float(dt_s))
    )


def _sole_contact_metrics(
    sole_surface_residual_m: object,
) -> tuple[float, float]:
    residual = np.asarray(sole_surface_residual_m, dtype=np.float64)
    if (
        residual.ndim != 2
        or residual.shape[0] != 2
        or residual.shape[1] < 3
        or not np.isfinite(residual).all()
    ):
        raise ContractError("stair pivot sole contact inputs are invalid")
    return (
        float(np.abs(residual).max()),
        float(np.ptp(residual, axis=1).max()),
    )


def _sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(b"g1-stair-pivot-walk/v1")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_optimized_arrays(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, np.ndarray]:
    arrays = _load_arrays(path)
    shapes = {
        "joint_position": (29,),
        "joint_velocity": (29,),
        "root_position_world": (3,),
        "root_orientation_world_wxyz": (4,),
        "foot_position_world": (2, 3),
    }
    frame_count = None
    for name, trailing_shape in shapes.items():
        value = arrays.get(name)
        if (
            not isinstance(value, np.ndarray)
            or value.ndim != len(trailing_shape) + 1
            or value.shape[1:] != trailing_shape
            or len(value) < 2
            or not np.isfinite(value).all()
        ):
            raise ContractError("stair pivot optimized source arrays are invalid")
        if frame_count is None:
            frame_count = len(value)
        elif len(value) != frame_count:
            raise ContractError("stair pivot optimized source arrays are invalid")
    if _sha256(arrays) != expected_sha256:
        raise ContractError("stair pivot optimized source identity is invalid")
    return arrays


def main() -> int:
    args = _parser().parse_args()
    root_height_offsets = (
        (0.0,)
        if args.root_height_offset_m is None
        else tuple(float(value) for value in args.root_height_offset_m)
    )
    if (
        args.step_count < 1
        or args.frame_count_per_step < 3
        or args.beam_width < 1
        or not math.isfinite(args.swing_clearance_m)
        or args.swing_clearance_m <= 0.0
        or not root_height_offsets
        or not all(math.isfinite(value) for value in root_height_offsets)
        or not math.isfinite(args.maximum_joint_speed_rad_s)
        or args.maximum_joint_speed_rad_s <= 0.0
        or not math.isfinite(args.maximum_joint_acceleration_rad_s2)
        or args.maximum_joint_acceleration_rad_s2 <= 0.0
        or not math.isfinite(args.maximum_terminal_sole_contact_error_m)
        or args.maximum_terminal_sole_contact_error_m <= 0.0
        or not math.isfinite(args.minimum_stance_width_m)
        or not math.isfinite(args.maximum_stance_width_m)
        or args.minimum_stance_width_m <= 0.0
        or args.minimum_stance_width_m > args.maximum_stance_width_m
        or not math.isfinite(args.maximum_terminal_displacement_error_m)
        or args.maximum_terminal_displacement_error_m <= 0.0
    ):
        raise ContractError("stair pivot search inputs are invalid")
    yaw_schedule, target_schedule = _step_schedule(args)
    target_displacement = (
        target_schedule[-1] if target_schedule is not None else None
    )
    graph_document = json.loads(args.graph.read_text(encoding="utf-8"))
    original_graph = stair_motion_graph_from_dict(graph_document)
    graph = {
        **original_graph.to_dict(),
        "source": graph_document.get("source"),
    }
    edge = _select_edge(graph.get("edges"), args.incoming_edge)
    if (
        edge.get("minimum_sole_clearance_m") is None
        or float(edge["minimum_sole_clearance_m"]) < SOLE_CLEARANCE_LIMIT_M
        or edge.get("maximum_joint_speed_rad_s") is None
        or float(edge["maximum_joint_speed_rad_s"])
        > args.maximum_joint_speed_rad_s
        or (
            edge.get("artifact_kind", "source") == "optimized"
            and (
                edge.get("maximum_joint_acceleration_rad_s2") is None
                or float(edge["maximum_joint_acceleration_rad_s2"])
                > args.maximum_joint_acceleration_rad_s2
                or edge.get(
                    "maximum_terminal_sole_contact_error_m"
                )
                is None
                or float(
                    edge["maximum_terminal_sole_contact_error_m"]
                )
                > args.maximum_terminal_sole_contact_error_m
            )
        )
    ):
        raise ContractError("stair pivot incoming edge is not fully validated")
    graph_source = graph.get("source")
    sources = (
        graph_source.get("accepted_routes")
        if isinstance(graph_source, dict)
        else None
    )
    if not isinstance(sources, list):
        raise ContractError("stair pivot graph sources are invalid")
    seed_is_optimized = (
        edge.get("artifact_kind", "source") == "optimized"
    )
    if seed_is_optimized:
        optimized_sources = graph_source.get("optimized_artifacts", [])
        source = next(
            (
                item
                for item in optimized_sources
                if isinstance(item, dict)
                and item.get("deterministic_sha256")
                == edge["source_artifact_sha256"]
            ),
            None,
        )
        if not isinstance(source, dict):
            raise ContractError("stair pivot optimized source is missing")
        source_arrays = _load_optimized_arrays(
            Path(source["connector_path"]),
            expected_sha256=edge["source_artifact_sha256"],
        )
    else:
        source_by_sha = {
            item.get("route_deterministic_sha256"): item
            for item in sources
            if isinstance(item, dict)
        }
        source = source_by_sha.get(edge["source_artifact_sha256"])
        if not isinstance(source, dict):
            raise ContractError("stair pivot edge source is missing")
        source_arrays = _load_arrays(
            Path(source["route_directory"]) / "arrays.npz"
        )
    incoming_frame = int(edge["source_end_frame_exclusive"]) - 1
    initial_boundary = _boundary(
        source_arrays,
        incoming_frame,
        stationary=not seed_is_optimized,
    )
    if seed_is_optimized:
        validation_history = source_arrays["joint_position"][
            max(0, incoming_frame - 2) : incoming_frame + 1
        ].copy()
        if len(validation_history) < 3:
            validation_history = np.concatenate(
                (
                    np.repeat(
                        validation_history[:1],
                        3 - len(validation_history),
                        axis=0,
                    ),
                    validation_history,
                )
            )
    else:
        validation_history = np.repeat(
            initial_boundary.joint_position[None, :],
            3,
            axis=0,
        )
    lower, upper = _joint_limits(args.joint_contract)
    sole_kinematics = MujocoG1SoleKinematics(args.g1_xml)
    kinematics = BoundedFootKinematics(
        MujocoG1FootKinematics(args.g1_xml),
        sole_kinematics=sole_kinematics,
        maximum_function_evaluations=128,
    )
    resolved = resolve_stair_config(
        args.dataset,
        load_experiment_config(args.config),
        device="cpu",
    )
    ascent_world_xy = _authenticated_graph_ascent_world_xy(
        graph_source,
        dataset_manifest_sha256=resolved.dataset.manifest_sha256,
    )
    initial_root_world_xy = initial_boundary.root_position_world[:2].copy()

    def sample_surface(points_xy: np.ndarray) -> np.ndarray:
        points = torch.as_tensor(
            np.asarray(points_xy, dtype=np.float64).copy(),
            dtype=torch.float32,
        )
        return (
            resolved.measurement_extension.query_grid.sample_xy(
                resolved.measurement_extension.alignment.matcher_to_scene_xy(
                    points
                )
            )
            .detach()
            .cpu()
            .numpy()
        )

    u_values = _offset_values(
        args.offset_min_u_m,
        args.offset_max_u_m,
        args.offset_step_m,
    )
    v_values = _offset_values(
        args.offset_min_v_m,
        args.offset_max_v_m,
        args.offset_step_m,
    )
    offsets = _stair_offsets_world(
        tuple((u, v) for u in u_values for v in v_values),
        ascent_world_xy=ascent_world_xy,
    )
    beam = [
        {
            "boundary": initial_boundary,
            "arrays": None,
            "validation_joint_position": validation_history,
            "minimum_sole_clearance_m": math.inf,
            "maximum_joint_speed_rad_s": 0.0,
            "maximum_joint_acceleration_rad_s2": 0.0,
            "maximum_terminal_sole_contact_error_m": 0.0,
            "maximum_terminal_sole_contact_spread_m": 0.0,
            "last_pivot_foot": None,
            "steps": [],
            "displacement_error_m": (
                0.0
            ),
        }
    ]
    audit_steps = []
    for step_index in range(args.step_count):
        expanded = []
        audit = []
        for parent_index, state in enumerate(beam):
            candidates = pivot_target_lattice(
                state["boundary"].foot_position_world,
                yaw_delta_rad=float(yaw_schedule[step_index]),
                planar_offsets_m=offsets,
                sample_surface_height_m=sample_surface,
            )
            for candidate_index, candidate in enumerate(candidates):
                if (
                    state["last_pivot_foot"] is not None
                    and candidate.pivot_foot == state["last_pivot_foot"]
                ):
                    continue
                swing_foot = 1 - candidate.pivot_foot
                base_record = {
                    "step_index": step_index,
                    "parent_index": parent_index,
                    "candidate_index": candidate_index,
                    "pivot_foot": candidate.pivot_foot,
                    "planar_offset_m": list(candidate.planar_offset_m),
                    "swing_target_world": candidate.target_foot_position_world[
                        swing_foot
                    ].tolist(),
                }
                stance_width = float(
                    np.linalg.norm(
                        candidate.target_foot_position_world[1, :2]
                        - candidate.target_foot_position_world[0, :2]
                    )
                )
                base_record["stance_width_m"] = stance_width
                if not _valid_stance_width(
                    candidate.target_foot_position_world[:, :2],
                    minimum_m=args.minimum_stance_width_m,
                    maximum_m=args.maximum_stance_width_m,
                ):
                    base_record.update(
                        {
                            "solved": False,
                            "accepted": False,
                            "failure_type": "StanceWidthGate",
                            "failure": (
                                "candidate footholds violate stance-width "
                                "contract"
                            ),
                        }
                    )
                    audit.append(base_record)
                    continue
                boundary_soles = sole_kinematics.sole_points(
                    state["boundary"].joint_position[None, :],
                    state["boundary"].root_position_world[None, :],
                    state["boundary"].root_orientation_world_wxyz[None, :],
                )[0]
                terminal_foot_path = np.repeat(
                    state["boundary"].foot_position_world[None, :, :],
                    3,
                    axis=0,
                )
                terminal_foot_path[-1] = (
                    candidate.target_foot_position_world
                )
                terminal_soles = single_step_sole_trajectory(
                    boundary_soles,
                    state["boundary"].foot_position_world,
                    terminal_foot_path,
                    pivot_foot=candidate.pivot_foot,
                    yaw_delta_rad=float(yaw_schedule[step_index]),
                )[-1]
                terminal_surface = sample_surface(
                    terminal_soles[..., :2].reshape(-1, 2)
                ).reshape(terminal_soles.shape[:-1])
                (
                    target_contact_error,
                    target_contact_spread,
                ) = _sole_contact_metrics(
                    terminal_soles[..., 2] - terminal_surface
                )
                base_record.update(
                    {
                        "target_sole_contact_error_m": (
                            target_contact_error
                        ),
                        "target_sole_contact_spread_m": (
                            target_contact_spread
                        ),
                    }
                )
                if (
                    target_contact_error
                    > args.maximum_terminal_sole_contact_error_m
                ):
                    base_record.update(
                        {
                            "solved": False,
                            "accepted": False,
                            "failure_type": "TargetSoleContactGate",
                            "failure": (
                                "candidate footprint does not share a "
                                "supported terrain surface"
                            ),
                        }
                    )
                    audit.append(base_record)
                    continue
                for root_height_offset in root_height_offsets:
                    record = {
                        **base_record,
                        "root_height_offset_m": root_height_offset,
                    }
                    try:
                        connector = synthesize_single_step_connector(
                            state["boundary"],
                            swing_target_world=(
                                candidate.target_foot_position_world[swing_foot]
                            ),
                            pivot_foot=candidate.pivot_foot,
                            root_yaw_delta_rad=float(
                                yaw_schedule[step_index]
                            ),
                            kinematics=kinematics,
                            frame_count=args.frame_count_per_step,
                            dt_s=DT_S,
                            swing_clearance_m=args.swing_clearance_m,
                            root_height_offset_m=root_height_offset,
                            maximum_joint_speed_rad_s=(
                                args.maximum_joint_speed_rad_s
                            ),
                            maximum_joint_acceleration_rad_s2=(
                                args.maximum_joint_acceleration_rad_s2
                            ),
                            joint_position_lower=lower,
                            joint_position_upper=upper,
                        )
                        current = _connector_arrays(connector)
                        _pin_connector_start(current, state["boundary"])
                        sole = sole_kinematics.sole_points(
                            current["joint_position"],
                            current["root_position_world"],
                            current["root_orientation_world_wxyz"],
                        )
                        surface = sample_surface(
                            sole[..., :2].reshape(-1, 2)
                        ).reshape(sole.shape[:-1])
                        minimum_clearance = float(
                            (sole[..., 2] - surface).min()
                        )
                        (
                            terminal_contact_error,
                            terminal_contact_spread,
                        ) = _sole_contact_metrics(
                            sole[-1, ..., 2] - surface[-1]
                        )
                        joined = _join(state["arrays"], current)
                        validation_joint_position = np.concatenate(
                            (
                                state["validation_joint_position"],
                                current["joint_position"][1:],
                            ),
                            axis=0,
                        )
                        observed_speed = float(
                            np.abs(
                                np.diff(
                                    validation_joint_position,
                                    axis=0,
                                )
                            ).max()
                            / DT_S
                        )
                        observed_acceleration = (
                            _maximum_joint_acceleration_rad_s2(
                                validation_joint_position,
                                dt_s=DT_S,
                            )
                        )
                        root_displacement = _world_displacement_to_stair(
                            joined["root_position_world"][-1, :2]
                            - initial_root_world_xy,
                            ascent_world_xy=ascent_world_xy,
                        )
                        expected_displacement = (
                            target_schedule[step_index]
                            if target_schedule is not None
                            else root_displacement
                        )
                        displacement_error = float(
                            np.linalg.norm(
                                root_displacement - expected_displacement
                            )
                        )
                        record.update(
                            {
                                "solved": True,
                                "minimum_sole_clearance_m": minimum_clearance,
                                "maximum_joint_speed_rad_s": observed_speed,
                                "maximum_joint_acceleration_rad_s2": (
                                    observed_acceleration
                                ),
                                "terminal_sole_contact_error_m": (
                                    terminal_contact_error
                                ),
                                "terminal_sole_contact_spread_m": (
                                    terminal_contact_spread
                                ),
                                "root_displacement_stair_uv_m": (
                                    root_displacement.tolist()
                                ),
                                "displacement_error_m": displacement_error,
                            }
                        )
                        if (
                            minimum_clearance < SOLE_CLEARANCE_LIMIT_M
                            or observed_speed
                            > args.maximum_joint_speed_rad_s
                            or observed_acceleration
                            > args.maximum_joint_acceleration_rad_s2
                            or terminal_contact_error
                            > args.maximum_terminal_sole_contact_error_m
                        ):
                            record["accepted"] = False
                        else:
                            record["accepted"] = True
                            next_boundary = _boundary(
                                current,
                                len(current["joint_position"]) - 1,
                            )
                            expanded.append(
                                {
                                    "boundary": next_boundary,
                                    "arrays": joined,
                                    "validation_joint_position": (
                                        validation_joint_position
                                    ),
                                    "minimum_sole_clearance_m": min(
                                        state["minimum_sole_clearance_m"],
                                        minimum_clearance,
                                    ),
                                    "maximum_joint_speed_rad_s": (
                                        observed_speed
                                    ),
                                    "maximum_joint_acceleration_rad_s2": (
                                        observed_acceleration
                                    ),
                                    "maximum_terminal_sole_contact_error_m": max(
                                        state[
                                            "maximum_terminal_sole_contact_error_m"
                                        ],
                                        terminal_contact_error,
                                    ),
                                    "maximum_terminal_sole_contact_spread_m": max(
                                        state[
                                            "maximum_terminal_sole_contact_spread_m"
                                        ],
                                        terminal_contact_spread,
                                    ),
                                    "last_pivot_foot": candidate.pivot_foot,
                                    "displacement_error_m": (
                                        displacement_error
                                    ),
                                    "steps": [
                                        *state["steps"],
                                        record,
                                    ],
                                }
                            )
                    except Exception as error:
                        record.update(
                            {
                                "solved": False,
                                "accepted": False,
                                "failure_type": type(error).__name__,
                                "failure": str(error),
                            }
                        )
                    audit.append(record)
        audit_steps.append(audit)
        beam = sorted(
            expanded,
            key=lambda state: (
                -state["displacement_error_m"]
                if target_schedule is not None
                else 0.0,
                state["minimum_sole_clearance_m"],
                -state["maximum_joint_speed_rad_s"],
                -state["maximum_joint_acceleration_rad_s2"],
                -state["maximum_terminal_sole_contact_error_m"],
            ),
            reverse=True,
        )[: args.beam_width]
        if not beam:
            break

    accepted = (
        len(beam) > 0
        and len(beam[0]["steps"]) == args.step_count
        and (
            target_schedule is None
            or beam[0]["displacement_error_m"]
            <= args.maximum_terminal_displacement_error_m
        )
    )
    payload = {
        "schema": "g1-stair-pivot-search/v1",
        "accepted": accepted,
        "incoming_edge_id": edge["edge_id"],
        "initial_boundary_mode": (
            "optimized-edge-terminal"
            if seed_is_optimized
            else "stationary-pose-seed"
        ),
        "total_yaw_delta_rad": float(yaw_schedule.sum()),
        "step_yaw_delta_rad": yaw_schedule.tolist(),
        "step_count": args.step_count,
        "frame_count_per_step": args.frame_count_per_step,
        "root_height_offset_candidates_m": list(root_height_offsets),
        "maximum_joint_acceleration_limit_rad_s2": (
            args.maximum_joint_acceleration_rad_s2
        ),
        "maximum_terminal_sole_contact_error_limit_m": (
            args.maximum_terminal_sole_contact_error_m
        ),
        "beam_width": args.beam_width,
        "target_root_displacement_stair_uv_m": (
            target_displacement.tolist()
            if target_displacement is not None
            else None
        ),
        "step_target_root_stair_uv_m": (
            target_schedule.tolist()
            if target_schedule is not None
            else None
        ),
        "candidate_audit_by_step": audit_steps,
    }
    if beam:
        payload["best_attempt"] = {
            "completed_step_count": len(beam[0]["steps"]),
            "minimum_sole_clearance_m": beam[0][
                "minimum_sole_clearance_m"
            ],
            "maximum_joint_speed_rad_s": beam[0][
                "maximum_joint_speed_rad_s"
            ],
            "maximum_joint_acceleration_rad_s2": beam[0][
                "maximum_joint_acceleration_rad_s2"
            ],
            "maximum_terminal_sole_contact_error_m": beam[0][
                "maximum_terminal_sole_contact_error_m"
            ],
            "maximum_terminal_sole_contact_spread_m": beam[0][
                "maximum_terminal_sole_contact_spread_m"
            ],
            "terminal_displacement_error_m": beam[0][
                "displacement_error_m"
            ],
            "selected_steps": beam[0]["steps"],
        }
    args.output.mkdir(parents=True, exist_ok=False)
    if accepted:
        best = beam[0]
        arrays = best["arrays"]
        arrays["joint_velocity"][0] = initial_boundary.joint_velocity
        artifact_sha256 = _sha256(arrays)
        payload.update(
            {
                "deterministic_sha256": artifact_sha256,
                "minimum_sole_clearance_m": best[
                    "minimum_sole_clearance_m"
                ],
                "maximum_joint_speed_rad_s": best[
                    "maximum_joint_speed_rad_s"
                ],
                "maximum_joint_acceleration_rad_s2": best[
                    "maximum_joint_acceleration_rad_s2"
                ],
                "maximum_terminal_sole_contact_error_m": best[
                    "maximum_terminal_sole_contact_error_m"
                ],
                "maximum_terminal_sole_contact_spread_m": best[
                    "maximum_terminal_sole_contact_spread_m"
                ],
                "selected_steps": best["steps"],
                "terminal_displacement_error_m": best[
                    "displacement_error_m"
                ],
            }
        )
        np.savez_compressed(args.output / "connector.npz", **arrays)
        original_incoming = original_graph.edge_by_id[edge["edge_id"]]
        original_end_node = original_graph.node_by_id[
            original_incoming.end_node_id
        ]
        segment_ranges = _landing_segment_ranges(
            total_frame_count=len(arrays["joint_position"]),
            frame_count_per_step=args.frame_count_per_step,
            step_count=args.step_count,
        )
        boundary_indices = (
            segment_ranges[0][0],
            *(end_exclusive - 1 for _, end_exclusive in segment_ranges),
        )
        boundary_surfaces = sample_surface(
            arrays["foot_position_world"][
                np.asarray(boundary_indices), :, :2
            ].reshape(-1, 2)
        ).reshape(len(boundary_indices), 2)
        generated_nodes = {}
        generated_edges = []
        generated_artifacts = []
        last_landing_foot = original_end_node.last_landing_foot
        previous_end_boundary_sha256 = (
            original_incoming.end_boundary_sha256
            if seed_is_optimized
            else None
        )
        for step_index, (start, end_exclusive) in enumerate(segment_ranges):
            end = end_exclusive - 1
            segment_arrays = {
                name: value[start:end_exclusive].copy()
                for name, value in arrays.items()
            }
            segment_sha256 = _sha256(segment_arrays)
            segment_path = (
                args.output / f"connector-step-{step_index:03d}.npz"
            )
            np.savez_compressed(segment_path, **segment_arrays)
            start_node = _optimized_contact_node(
                root_position_world=arrays["root_position_world"][start],
                root_orientation_world_wxyz=(
                    arrays["root_orientation_world_wxyz"][start]
                ),
                foot_position_world=arrays["foot_position_world"][start],
                foot_surface_height_m=boundary_surfaces[step_index],
                graph_source=graph_source,
                last_landing_foot=last_landing_foot,
            )
            next_landing_foot = 1 - int(
                best["steps"][step_index]["pivot_foot"]
            )
            end_node = _optimized_contact_node(
                root_position_world=arrays["root_position_world"][end],
                root_orientation_world_wxyz=(
                    arrays["root_orientation_world_wxyz"][end]
                ),
                foot_position_world=arrays["foot_position_world"][end],
                foot_surface_height_m=boundary_surfaces[step_index + 1],
                graph_source=graph_source,
                last_landing_foot=next_landing_foot,
            )
            start_boundary_sha256 = (
                _verified_optimized_boundary_sha256(
                    expected_sha256=previous_end_boundary_sha256,
                    joint_position=arrays["joint_position"][start],
                    joint_velocity=arrays["joint_velocity"][start],
                    root_position_world=arrays["root_position_world"][start],
                    root_orientation_world_wxyz=(
                        arrays["root_orientation_world_wxyz"][start]
                    ),
                    foot_position_world=arrays["foot_position_world"][start],
                    foot_surface_height_m=boundary_surfaces[step_index],
                )
            )
            end_boundary_sha256 = _optimized_boundary_sha256(
                joint_position=arrays["joint_position"][end],
                joint_velocity=arrays["joint_velocity"][end],
                root_position_world=arrays["root_position_world"][end],
                root_orientation_world_wxyz=(
                    arrays["root_orientation_world_wxyz"][end]
                ),
                foot_position_world=arrays["foot_position_world"][end],
                foot_surface_height_m=boundary_surfaces[step_index + 1],
            )
            step_metrics = best["steps"][step_index]
            optimized_edge = StairMotionEdge(
                start_node_id=start_node.node_id,
                end_node_id=end_node.node_id,
                traversal_heading_bin=end_node.heading_bin,
                frame_count=len(segment_arrays["joint_position"]),
                source_artifact_sha256=segment_sha256,
                source_start_frame=0,
                source_end_frame_exclusive=len(
                    segment_arrays["joint_position"]
                ),
                start_boundary_sha256=start_boundary_sha256,
                end_boundary_sha256=end_boundary_sha256,
                provenance_sha256=artifact_sha256,
                exact_contact_valid=True,
                minimum_sole_clearance_m=step_metrics[
                    "minimum_sole_clearance_m"
                ],
                maximum_joint_speed_rad_s=step_metrics[
                    "maximum_joint_speed_rad_s"
                ],
                maximum_joint_acceleration_rad_s2=step_metrics[
                    "maximum_joint_acceleration_rad_s2"
                ],
                maximum_terminal_sole_contact_error_m=step_metrics[
                    "terminal_sole_contact_error_m"
                ],
                artifact_kind="optimized",
            )
            generated_nodes[start_node.node_id] = start_node
            generated_nodes[end_node.node_id] = end_node
            generated_edges.append(optimized_edge)
            generated_artifacts.append(
                {
                    "deterministic_sha256": segment_sha256,
                    "route_deterministic_sha256": artifact_sha256,
                    "connector_path": str(segment_path.resolve()),
                    "edge_id": optimized_edge.edge_id,
                    "pose_seed_edge_id": edge["edge_id"],
                    "step_index": step_index,
                }
            )
            last_landing_foot = next_landing_foot
            previous_end_boundary_sha256 = end_boundary_sha256
        optimized_graph = merge_stair_motion_graphs(
            (
                type(original_graph)(
                    original_graph.nodes,
                    tuple(
                        existing
                        for existing in original_graph.edges
                        if existing.artifact_kind == "optimized"
                    ),
                ),
                type(original_graph)(
                    tuple(generated_nodes.values()),
                    tuple(generated_edges),
                ),
            )
        )
        optimized_source = {
            **graph_source,
            "optimized_artifacts": [
                *graph_source.get("optimized_artifacts", []),
                *generated_artifacts,
            ],
        }
        optimized_payload = {
            **optimized_graph.to_dict(),
            "source": optimized_source,
            "coverage": optimized_graph.coverage_summary(),
        }
        (args.output / "graph.json").write_text(
            json.dumps(
                optimized_payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
    (args.output / "metrics.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "accepted": accepted,
                "minimum_sole_clearance_m": payload.get(
                    "minimum_sole_clearance_m"
                ),
                "maximum_joint_speed_rad_s": payload.get(
                    "maximum_joint_speed_rad_s"
                ),
                "maximum_joint_acceleration_rad_s2": payload.get(
                    "maximum_joint_acceleration_rad_s2"
                ),
                "maximum_terminal_sole_contact_error_m": payload.get(
                    "maximum_terminal_sole_contact_error_m"
                ),
            },
            sort_keys=True,
        )
    )
    if not accepted:
        raise ContractError("stair pivot search found no complete graph walk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
