"""Extract exact stair-relative contact edges from validated rollout traces."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping

import numpy as np

from .joints import ContractError
from .torch_contact_segments import source_support_mask
from .torch_stair_motion_graph import (
    StairContactNode,
    StairFoothold,
    StairMotionEdge,
    StairMotionGraph,
    heading_bin_from_yaw,
)
from .torch_terrain_omni_routes import StairFrame
from .torch_terrain_features import TerrainDataset


DT_S = 0.02
DEFAULT_LATERAL_CELL_M = 0.10
STOPPED_SPEED_MPS = 0.03
FAST_SPEED_MPS = 0.50


def aligned_source_support_mask(
    dataset: TerrainDataset,
    selected_clip_path: object,
    selected_source_frame: object,
) -> np.ndarray:
    """Resolve saved clip/frame provenance to authenticated source contacts."""

    if not isinstance(dataset, TerrainDataset):
        raise ContractError("stair graph contacts require a terrain dataset")
    paths = np.asarray(selected_clip_path)
    frames = np.asarray(selected_source_frame)
    if (
        paths.ndim != 1
        or frames.shape != paths.shape
        or frames.dtype.kind not in "iu"
    ):
        raise ContractError("stair graph saved source selection is invalid")
    index_by_path = {
        clip.relative_path: index
        for index, clip in enumerate(dataset.folder.clips)
    }
    cache: dict[int, np.ndarray] = {}
    output = np.empty((len(paths), 2), dtype=np.bool_)
    for output_frame, (raw_path, raw_frame) in enumerate(zip(paths, frames)):
        path = str(raw_path)
        if path not in index_by_path:
            raise ContractError(
                f"stair graph selected clip is not in the dataset: {path}"
            )
        clip_index = index_by_path[path]
        if clip_index not in cache:
            cache[clip_index] = (
                source_support_mask(dataset, clip_index)
                .detach()
                .cpu()
                .numpy()
            )
        source_frame = int(raw_frame)
        mask = cache[clip_index]
        if not 0 <= source_frame < len(mask):
            raise ContractError(
                f"stair graph selected source frame is out of range: {path}"
            )
        output[output_frame] = mask[source_frame]
    return output


def landing_decision_frames(support_mask: object) -> tuple[int, ...]:
    """Return the first frame of every double-support landing plateau."""

    support = np.asarray(support_mask)
    if (
        support.dtype != np.bool_
        or support.ndim != 2
        or support.shape[1] != 2
        or support.shape[0] < 1
    ):
        raise ContractError(
            "stair graph support mask must have boolean shape (T, 2)"
        )
    double = support.all(axis=1)
    onset = double.copy()
    onset[1:] &= ~double[:-1]
    return tuple(int(frame) for frame in np.flatnonzero(onset))


def _required_array(
    arrays: Mapping[str, object],
    name: str,
    *,
    frame_count: int,
    trailing_shape: tuple[int, ...],
    floating: bool = True,
) -> np.ndarray:
    if name not in arrays:
        raise ContractError(f"stair graph rollout array is missing: {name}")
    value = np.asarray(arrays[name])
    if value.shape != (frame_count, *trailing_shape):
        raise ContractError(f"stair graph rollout array has invalid shape: {name}")
    if floating:
        if value.dtype.kind not in "fc" or not np.isfinite(value).all():
            raise ContractError(
                f"stair graph rollout array must be finite floating point: {name}"
            )
    return value


def _stair_xy(
    world_xy: np.ndarray,
    stair_frame: StairFrame,
) -> np.ndarray:
    delta = np.asarray(world_xy, dtype=np.float64) - np.asarray(
        stair_frame.origin_world_xy,
        dtype=np.float64,
    )
    cosine = math.cos(stair_frame.ascent_world_yaw)
    sine = math.sin(stair_frame.ascent_world_yaw)
    return np.stack(
        (
            delta[..., 0] * cosine + delta[..., 1] * sine,
            -delta[..., 0] * sine + delta[..., 1] * cosine,
        ),
        axis=-1,
    )


def _quantize(value: float, cell_size: float) -> int:
    return int(math.floor(float(value) / float(cell_size) + 0.5))


def _hash_payload(
    *,
    schema: str,
    arrays: tuple[np.ndarray, ...],
    strings: tuple[str, ...] = (),
) -> str:
    digest = hashlib.sha256()
    digest.update(schema.encode("utf-8"))
    for value in arrays:
        owned = np.ascontiguousarray(value)
        digest.update(str(owned.dtype).encode("ascii"))
        digest.update(json.dumps(owned.shape).encode("ascii"))
        digest.update(owned.tobytes())
    digest.update(
        json.dumps(strings, ensure_ascii=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def _boundary_sha256(
    arrays: Mapping[str, np.ndarray],
    support: np.ndarray,
    frame: int,
) -> str:
    return _hash_payload(
        schema="g1-stair-boundary/v1",
        arrays=(
            arrays["qpos"][frame],
            arrays["joint_velocity"][frame],
            arrays["foot_position_world"][frame],
            arrays["foot_surface_height_m"][frame],
            support[frame],
        ),
    )


def _edge_provenance_sha256(
    arrays: Mapping[str, np.ndarray],
    support: np.ndarray,
    start: int,
    end: int,
) -> str:
    inclusive = slice(start, end + 1)
    return _hash_payload(
        schema="g1-stair-edge-provenance/v1",
        arrays=(
            arrays["qpos"][inclusive],
            arrays["root_position_world"][inclusive],
            arrays["foot_position_world"][inclusive],
            arrays["foot_surface_height_m"][inclusive],
            support[inclusive],
            arrays["selected_source_frame"][inclusive],
        ),
        strings=tuple(
            str(value) for value in arrays["selected_clip_path"][inclusive]
        ),
    )


def _last_landing_foot(support: np.ndarray, frame: int) -> int:
    if frame == 0:
        return -1
    entering = np.flatnonzero((~support[frame - 1]) & support[frame])
    return int(entering[0]) if len(entering) == 1 else -1


def _velocity_bin(root_world: np.ndarray, frame: int) -> int:
    if len(root_world) == 1:
        speed = 0.0
    elif frame == 0:
        speed = float(
            np.linalg.norm(root_world[1, :2] - root_world[0, :2]) / DT_S
        )
    else:
        speed = float(
            np.linalg.norm(
                root_world[frame, :2] - root_world[frame - 1, :2]
            )
            / DT_S
        )
    if speed <= STOPPED_SPEED_MPS:
        return 0
    return 1 if speed <= FAST_SPEED_MPS else 2


def _traversal_heading_bin(
    root_stair_xy: np.ndarray,
    command_stair_xy: np.ndarray,
    start: int,
    end: int,
) -> int:
    displacement = root_stair_xy[end] - root_stair_xy[start]
    if float(np.linalg.norm(displacement)) <= 0.02:
        displacement = command_stair_xy[start : end + 1].mean(axis=0)
    if float(np.linalg.norm(displacement)) <= 1e-8:
        return 0
    return heading_bin_from_yaw(
        math.atan2(float(displacement[1]), float(displacement[0]))
    )


def extract_stair_motion_graph(
    rollout_arrays: Mapping[str, object],
    *,
    support_mask: object,
    stair_frame: StairFrame,
    exact_contact_valid: bool,
    source_artifact_sha256: str,
    lateral_cell_m: float = DEFAULT_LATERAL_CELL_M,
    minimum_sole_clearance_by_frame: object | None = None,
    minimum_sole_clearance_m: float = -0.025,
    maximum_joint_speed_rad_s: float = 13.0,
) -> StairMotionGraph:
    """Extract landing-to-landing edges in one fixed global stair frame."""

    if exact_contact_valid is not True:
        raise ContractError(
            "stair graph extraction requires an exact-contact-validated rollout"
        )
    if (
        not isinstance(source_artifact_sha256, str)
        or len(source_artifact_sha256) != 64
    ):
        raise ContractError("stair graph source artifact identity is invalid")
    try:
        int(source_artifact_sha256, 16)
    except ValueError as error:
        raise ContractError(
            "stair graph source artifact identity is invalid"
        ) from error
    if not isinstance(stair_frame, StairFrame):
        raise ContractError("stair graph frame is invalid")
    if (
        not math.isfinite(float(lateral_cell_m))
        or float(lateral_cell_m) <= 0.0
        or stair_frame.tread_depth_m <= 0.0
        or stair_frame.riser_height_m <= 0.0
        or isinstance(maximum_joint_speed_rad_s, bool)
        or not isinstance(maximum_joint_speed_rad_s, (int, float))
        or not math.isfinite(float(maximum_joint_speed_rad_s))
        or float(maximum_joint_speed_rad_s) <= 0.0
    ):
        raise ContractError("stair graph quantization is invalid")
    support = np.asarray(support_mask)
    decisions = landing_decision_frames(support)
    frame_count = int(support.shape[0])
    sole_clearance = None
    if minimum_sole_clearance_by_frame is not None:
        sole_clearance = np.asarray(
            minimum_sole_clearance_by_frame,
            dtype=np.float64,
        )
        if (
            sole_clearance.shape != (frame_count,)
            or not np.isfinite(sole_clearance).all()
            or isinstance(minimum_sole_clearance_m, bool)
            or not isinstance(minimum_sole_clearance_m, (int, float))
            or not math.isfinite(float(minimum_sole_clearance_m))
        ):
            raise ContractError("stair graph sole clearance trace is invalid")
    qpos_value = np.asarray(rollout_arrays.get("qpos"))
    joint_velocity_value = np.asarray(
        rollout_arrays.get("joint_velocity")
    )
    if qpos_value.ndim != 2 or joint_velocity_value.ndim != 2:
        raise ContractError("stair graph rollout state arrays are missing")
    arrays = {
        "root_position_world": _required_array(
            rollout_arrays,
            "root_position_world",
            frame_count=frame_count,
            trailing_shape=(3,),
        ),
        "root_yaw_world": _required_array(
            rollout_arrays,
            "root_yaw_world",
            frame_count=frame_count,
            trailing_shape=(),
        ),
        "foot_position_world": _required_array(
            rollout_arrays,
            "foot_position_world",
            frame_count=frame_count,
            trailing_shape=(2, 3),
        ),
        "foot_surface_height_m": _required_array(
            rollout_arrays,
            "foot_surface_height_m",
            frame_count=frame_count,
            trailing_shape=(2,),
        ),
        "qpos": _required_array(
            rollout_arrays,
            "qpos",
            frame_count=frame_count,
            trailing_shape=(qpos_value.shape[-1],),
        ),
        "joint_velocity": _required_array(
            rollout_arrays,
            "joint_velocity",
            frame_count=frame_count,
            trailing_shape=(joint_velocity_value.shape[-1],),
        ),
        "joint_position": _required_array(
            rollout_arrays,
            "joint_position",
            frame_count=frame_count,
            trailing_shape=(29,),
        ),
        "command_velocity_world_xy": _required_array(
            rollout_arrays,
            "command_velocity_world_xy",
            frame_count=frame_count,
            trailing_shape=(2,),
        ),
    }
    for name in ("selected_clip_path", "selected_source_frame"):
        if name not in rollout_arrays:
            raise ContractError(f"stair graph rollout array is missing: {name}")
        value = np.asarray(rollout_arrays[name])
        if value.shape != (frame_count,):
            raise ContractError(
                f"stair graph rollout array has invalid shape: {name}"
            )
        arrays[name] = value
    if arrays["selected_source_frame"].dtype.kind not in "iu":
        raise ContractError(
            "stair graph selected source frames must use integers"
        )

    roots_stair = _stair_xy(arrays["root_position_world"][:, :2], stair_frame)
    feet_stair = _stair_xy(arrays["foot_position_world"][:, :, :2], stair_frame)
    commands_stair = _stair_xy(
        arrays["command_velocity_world_xy"]
        + np.asarray(stair_frame.origin_world_xy),
        stair_frame,
    )
    nodes_by_id: dict[str, StairContactNode] = {}
    node_by_frame: dict[int, StairContactNode] = {}
    for frame in decisions:
        landing_foot = _last_landing_foot(support, frame)
        node = StairContactNode(
            root_u_cell=_quantize(
                roots_stair[frame, 0], stair_frame.tread_depth_m
            ),
            root_v_cell=_quantize(roots_stair[frame, 1], lateral_cell_m),
            root_height_level=max(
                0,
                _quantize(
                    float(arrays["root_position_world"][frame, 2]),
                    stair_frame.riser_height_m,
                ),
            ),
            heading_bin=heading_bin_from_yaw(
                float(arrays["root_yaw_world"][frame])
                - stair_frame.ascent_world_yaw
            ),
            velocity_bin=_velocity_bin(arrays["root_position_world"], frame),
            left_foothold=StairFoothold(
                _quantize(
                    float(arrays["foot_surface_height_m"][frame, 0]),
                    stair_frame.riser_height_m,
                ),
                _quantize(float(feet_stair[frame, 0, 1]), lateral_cell_m),
            ),
            right_foothold=StairFoothold(
                _quantize(
                    float(arrays["foot_surface_height_m"][frame, 1]),
                    stair_frame.riser_height_m,
                ),
                _quantize(float(feet_stair[frame, 1, 1]), lateral_cell_m),
            ),
            support_mask=3,
            last_landing_foot=landing_foot,
            gait_phase_bin=0 if landing_foot in (-1, 0) else 4,
        )
        nodes_by_id.setdefault(node.node_id, node)
        node_by_frame[frame] = node

    raw_edges = []
    for start, end in zip(decisions, decisions[1:]):
        edge_joint_position = arrays["joint_position"][start : end + 1]
        observed_joint_speed = float(
            np.abs(
                np.diff(
                    edge_joint_position,
                    axis=0,
                )
            ).max()
            / DT_S
        )
        observed_joint_acceleration = (
            float(
                np.abs(
                    np.diff(edge_joint_position, n=2, axis=0)
                ).max()
                / (DT_S * DT_S)
            )
            if len(edge_joint_position) >= 3
            else 0.0
        )
        if (
            observed_joint_speed > float(maximum_joint_speed_rad_s)
            or (
                sole_clearance is not None
                and float(sole_clearance[start : end + 1].min())
                < float(minimum_sole_clearance_m)
            )
        ):
            continue
        raw_edges.append(
            StairMotionEdge(
            start_node_id=node_by_frame[start].node_id,
            end_node_id=node_by_frame[end].node_id,
            traversal_heading_bin=_traversal_heading_bin(
                roots_stair,
                commands_stair,
                start,
                end,
            ),
            frame_count=end - start + 1,
            source_artifact_sha256=source_artifact_sha256,
            source_start_frame=start,
            source_end_frame_exclusive=end + 1,
            start_boundary_sha256=_boundary_sha256(arrays, support, start),
            end_boundary_sha256=_boundary_sha256(arrays, support, end),
            provenance_sha256=_edge_provenance_sha256(
                arrays,
                support,
                start,
                end,
            ),
            exact_contact_valid=True,
            minimum_sole_clearance_m=(
                float(sole_clearance[start : end + 1].min())
                if sole_clearance is not None
                else None
            ),
            maximum_joint_speed_rad_s=observed_joint_speed,
            maximum_joint_acceleration_rad_s2=(
                observed_joint_acceleration
            ),
        )
        )
    edges_by_id = {edge.edge_id: edge for edge in raw_edges}
    return StairMotionGraph(
        tuple(nodes_by_id.values()),
        tuple(edges_by_id.values()),
    )
