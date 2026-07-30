"""Deterministic 50 Hz up/turn/down benchmark for Torch stair matching.

The benchmark uses only the motion matcher's transactional prepare/commit API.
It performs no physics integration, SONIC inference, rendering, or state
teleport.  Runtime timing is retained as diagnostic evidence but excluded from
the deterministic rollout identity.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import torch

from .joints import ContractError
from .torch_motion_features import resolve_torch_device
from .torch_motion_matcher import TorchMotionMatcher
from .torch_terrain_rollout import (
    ResolvedStairConfig,
    load_experiment_config,
    matcher_config_from_resolved,
    resolve_stair_config,
    terrain_transition_validator_from_resolved,
)


ASCENT_STOP = 280
REVERSAL_STOP = 380
STEP_COUNT = 640
TRANSITION_NEIGHBORHOOD_RADIUS = 4
_TIMING_ARRAYS = frozenset(("search_time_ns", "step_time_ns"))
_PHASES = ("ascent", "reversal", "descent")


@dataclass(frozen=True)
class DirectionalRollout:
    arrays: Mapping[str, np.ndarray]
    metrics: Mapping
    events: Sequence[Mapping]
    resolved_config: Mapping
    resolved_config_sha256: str
    deterministic_sha256: str


def directional_phase(step: int) -> str:
    """Return the frozen benchmark phase for one output-frame index."""

    if type(step) is not int or not 0 <= step < STEP_COUNT:
        raise ContractError(
            f"directional step must be an integer in [0, {STEP_COUNT - 1}]"
        )
    if step < ASCENT_STOP:
        return "ascent"
    if step < REVERSAL_STOP:
        return "reversal"
    return "descent"


def transition_neighborhood_mask(
    transitioned: np.ndarray,
    *,
    radius: int = TRANSITION_NEIGHBORHOOD_RADIUS,
) -> np.ndarray:
    """Mark output frames within ``radius`` of an accepted transition."""

    values = np.asarray(transitioned)
    if values.ndim != 1 or values.dtype.kind != "b":
        raise ContractError("transitioned must be a one-dimensional bool array")
    if type(radius) is not int or radius < 0:
        raise ContractError("transition neighborhood radius must be non-negative")
    mask = np.zeros(values.shape, dtype=np.bool_)
    for index in np.flatnonzero(values):
        start = max(0, int(index) - radius)
        stop = min(len(values), int(index) + radius + 1)
        mask[start:stop] = True
    return mask


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _owned_readonly(array: np.ndarray) -> np.ndarray:
    output = np.ascontiguousarray(array)
    output.setflags(write=False)
    return output


def _distribution(values: np.ndarray) -> dict:
    finite = np.asarray(values, np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    samples = [float(value) for value in finite]
    if not samples:
        return {
            "count": 0,
            "samples": [],
            "p50": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    return {
        "count": len(samples),
        "samples": samples,
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "maximum": float(finite.max()),
    }


def _required_frame_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    frame_count: int | None = None,
) -> np.ndarray:
    if name not in arrays:
        raise ContractError(f"directional rollout is missing array: {name}")
    value = np.asarray(arrays[name])
    if value.ndim < 1:
        raise ContractError(f"directional array must have a frame axis: {name}")
    if frame_count is not None and len(value) != frame_count:
        raise ContractError(f"directional array frame count differs: {name}")
    if value.dtype.kind in "fc" and not np.isfinite(value).all():
        raise ContractError(f"directional array must be finite: {name}")
    return value


def _phase_mask(frame_count: int, phase: str) -> np.ndarray:
    return np.fromiter(
        (directional_phase(index) == phase for index in range(frame_count)),
        dtype=np.bool_,
        count=frame_count,
    )


def _phase_metrics(
    *,
    phase_mask: np.ndarray,
    dt: float,
    joint_position: np.ndarray,
    root_position: np.ndarray,
    foot_position: np.ndarray,
    transition_neighborhood: np.ndarray,
    transitioned: np.ndarray,
    rejected: np.ndarray,
    cross_clip: np.ndarray,
    rescue: np.ndarray,
    rescue_rank: np.ndarray,
    clearance: np.ndarray,
    position_cost: np.ndarray,
    velocity_cost: np.ndarray,
    continuity_cost: np.ndarray,
) -> dict:
    frame_indices = np.flatnonzero(phase_mask)
    phase_joint_position = joint_position[phase_mask].astype(np.float64)
    phase_root_position = root_position[phase_mask].astype(np.float64)
    phase_foot_position = foot_position[phase_mask].astype(np.float64)
    phase_clearance = clearance[phase_mask]
    joint_acceleration = np.linalg.norm(
        np.diff(phase_joint_position, n=2, axis=0) / (dt * dt),
        axis=1,
    )
    joint_jerk = np.linalg.norm(
        np.diff(phase_joint_position, n=3, axis=0) / (dt * dt * dt),
        axis=1,
    )
    joint_jerk_frames = frame_indices[3:]
    root_jerk = np.linalg.norm(
        np.diff(phase_root_position, n=3, axis=0) / (dt * dt * dt),
        axis=1,
    )
    foot_speed = np.linalg.norm(
        np.diff(phase_foot_position[:, :, :2], axis=0) / dt,
        axis=2,
    )
    contact_foot = np.argmin(phase_clearance[1:], axis=1)
    contact_speed = foot_speed[
        np.arange(len(foot_speed), dtype=np.int64), contact_foot
    ]
    accepted_in_phase = transitioned & phase_mask
    neighborhood_jerk = transition_neighborhood[joint_jerk_frames]
    return {
        "frame_count": int(phase_mask.sum()),
        "first_frame": (
            int(frame_indices[0]) if frame_indices.size else None
        ),
        "last_frame": int(frame_indices[-1]) if frame_indices.size else None,
        "accepted_transition_count": int(accepted_in_phase.sum()),
        "rejected_transition_count": int((rejected & phase_mask).sum()),
        "cross_clip_transition_count": int(
            (cross_clip & phase_mask).sum()
        ),
        "terrain_safety_override_count": int(
            (rescue & phase_mask).sum()
        ),
        "terrain_safety_override_rank": _distribution(
            rescue_rank[rescue & phase_mask]
        ),
        "minimum_foot_clearance_m": (
            float(phase_clearance.min()) if phase_clearance.size else None
        ),
        "final_root_height_m": (
            float(phase_root_position[-1, 2])
            if len(phase_root_position)
            else None
        ),
        "joint_acceleration_rad_s2": _distribution(joint_acceleration),
        "joint_jerk_rad_s3": _distribution(joint_jerk),
        "transition_neighborhood_joint_jerk_rad_s3": _distribution(
            joint_jerk[neighborhood_jerk]
        ),
        "transition_neighborhood_joint_jerk_output_frames": [
            int(frame)
            for frame in joint_jerk_frames[neighborhood_jerk]
        ],
        "root_jerk_m_s3": _distribution(root_jerk),
        "contact_foot_speed_m_s": _distribution(contact_speed),
        "transition_position_cost": _distribution(
            position_cost[accepted_in_phase]
        ),
        "transition_velocity_cost": _distribution(
            velocity_cost[accepted_in_phase]
        ),
        "transition_continuity_cost": _distribution(
            continuity_cost[accepted_in_phase]
        ),
    }


def compute_directional_metrics(
    arrays: Mapping[str, np.ndarray],
    *,
    dt: float,
    position_weight: float,
    velocity_weight: float,
) -> dict:
    """Compute phase-separated deterministic quality diagnostics and gates."""

    if not math.isfinite(dt) or dt <= 0.0:
        raise ContractError("directional metric dt must be finite and positive")
    position_weight = _validate_weight(position_weight, "position weight")
    velocity_weight = _validate_weight(velocity_weight, "velocity weight")
    joint_position = _required_frame_array(arrays, "joint_position")
    if joint_position.ndim != 2:
        raise ContractError("joint_position must have shape [frames, joints]")
    frame_count = len(joint_position)
    if not 1 <= frame_count <= STEP_COUNT:
        raise ContractError(
            f"directional frame count must be in [1, {STEP_COUNT}]"
        )
    joint_velocity = _required_frame_array(
        arrays, "joint_velocity", frame_count
    )
    root_position = _required_frame_array(
        arrays, "root_position_world", frame_count
    )
    foot_position = _required_frame_array(
        arrays, "foot_position_world", frame_count
    )
    clearance = _required_frame_array(
        arrays, "foot_clearance_m", frame_count
    )
    selected_clip = _required_frame_array(
        arrays, "selected_clip_index", frame_count
    )
    previous_selected_clip = _required_frame_array(
        arrays, "previous_selected_clip_index", frame_count
    )
    transitioned = _required_frame_array(
        arrays, "transitioned", frame_count
    )
    rejected = _required_frame_array(
        arrays, "transition_rejected", frame_count
    )
    rescue = _required_frame_array(
        arrays, "terrain_safety_override", frame_count
    )
    rescue_rank = _required_frame_array(
        arrays, "terrain_safety_override_rank", frame_count
    )
    position_cost = _required_frame_array(
        arrays, "selected_transition_position_cost", frame_count
    )
    velocity_cost = _required_frame_array(
        arrays, "selected_transition_velocity_cost", frame_count
    )
    continuity_cost = _required_frame_array(
        arrays, "selected_transition_continuity_cost", frame_count
    )
    if joint_velocity.shape != joint_position.shape:
        raise ContractError(
            "joint_velocity must have the joint_position shape"
        )
    if root_position.shape != (frame_count, 3):
        raise ContractError("root_position_world must have shape [frames, 3]")
    if foot_position.shape != (frame_count, 2, 3):
        raise ContractError(
            "foot_position_world must have shape [frames, 2, 3]"
        )
    if clearance.shape != (frame_count, 2):
        raise ContractError(
            "foot_clearance_m must have shape [frames, 2]"
        )
    for name, value in (
        ("selected_clip_index", selected_clip),
        ("previous_selected_clip_index", previous_selected_clip),
        ("transitioned", transitioned),
        ("transition_rejected", rejected),
        ("terrain_safety_override", rescue),
        ("terrain_safety_override_rank", rescue_rank),
        ("selected_transition_position_cost", position_cost),
        ("selected_transition_velocity_cost", velocity_cost),
        ("selected_transition_continuity_cost", continuity_cost),
    ):
        if value.ndim != 1:
            raise ContractError(f"{name} must be one-dimensional")
    if any(
        value.dtype.kind != "b" for value in (transitioned, rejected, rescue)
    ):
        raise ContractError("transition and rescue diagnostics must be bool")

    transition_neighborhood = transition_neighborhood_mask(transitioned)
    cross_clip = transitioned & (
        selected_clip != previous_selected_clip
    )

    aggregate_mask = np.ones(frame_count, dtype=np.bool_)
    aggregate = _phase_metrics(
        phase_mask=aggregate_mask,
        dt=dt,
        joint_position=joint_position,
        root_position=root_position,
        foot_position=foot_position,
        transition_neighborhood=transition_neighborhood,
        transitioned=transitioned,
        rejected=rejected,
        cross_clip=cross_clip,
        rescue=rescue,
        rescue_rank=rescue_rank,
        clearance=clearance,
        position_cost=position_cost,
        velocity_cost=velocity_cost,
        continuity_cost=continuity_cost,
    )
    transition_frames = np.flatnonzero(transitioned)
    intervals = np.diff(transition_frames)
    aggregate["accepted_transition_interval_frames"] = _distribution(
        intervals
    )
    aggregate["accepted_transition_interval_s"] = _distribution(
        intervals.astype(np.float64) * dt
    )
    phase_metrics = {
        phase: _phase_metrics(
            phase_mask=_phase_mask(frame_count, phase),
            dt=dt,
            joint_position=joint_position,
            root_position=root_position,
            foot_position=foot_position,
            transition_neighborhood=transition_neighborhood,
            transitioned=transitioned,
            rejected=rejected,
            cross_clip=cross_clip,
            rescue=rescue,
            rescue_rank=rescue_rank,
            clearance=clearance,
            position_cost=position_cost,
            velocity_cost=velocity_cost,
            continuity_cost=continuity_cost,
        )
        for phase in _PHASES
    }

    def percentile_or_none(phase: str, diagnostic: str, field: str):
        return phase_metrics[phase][diagnostic][field]

    descent_p95 = percentile_or_none(
        "descent", "joint_jerk_rad_s3", "p95"
    )
    descent_transition_max = percentile_or_none(
        "descent",
        "transition_neighborhood_joint_jerk_rad_s3",
        "maximum",
    )
    ascent_p95 = percentile_or_none(
        "ascent", "joint_jerk_rad_s3", "p95"
    )
    minimum_clearance = aggregate["minimum_foot_clearance_m"]
    final_root_height = aggregate["final_root_height_m"]
    gates = {
        "completed_640_frames": frame_count == STEP_COUNT,
        "returned_to_lower_height": (
            final_root_height is not None and final_root_height <= 0.82
        ),
        "minimum_clearance": (
            minimum_clearance is not None and minimum_clearance >= -0.03
        ),
        "descent_transition_count": (
            phase_metrics["descent"]["accepted_transition_count"] <= 11
        ),
        "descent_joint_jerk_p95": (
            descent_p95 is not None and descent_p95 <= 16614.05
        ),
        "descent_transition_max_jerk": (
            descent_transition_max is None
            or descent_transition_max <= 58035.6
        ),
        "ascent_joint_jerk_p95": (
            ascent_p95 is not None and ascent_p95 <= 14427.81
        ),
    }
    baseline = position_weight == 0.0 and velocity_weight == 0.0
    return {
        "frame_count": frame_count,
        "dt_s": float(dt),
        "position_weight": position_weight,
        "velocity_weight": velocity_weight,
        "baseline": baseline,
        "qualified": (not baseline) and all(gates.values()),
        "gates": gates,
        "aggregate": aggregate,
        "phases": phase_metrics,
    }


def _validate_weight(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ContractError(f"{name} must be finite and non-negative")
    return float(value)


def _directional_rollout_hash(
    resolved_config_sha256: str,
    arrays: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"directional-up-turn-down/v1")
    digest.update(b"\x00")
    digest.update(resolved_config_sha256.encode("ascii"))
    for name in sorted(arrays):
        if name in _TIMING_ARRAYS:
            continue
        value = np.ascontiguousarray(arrays[name])
        digest.update(b"\x00")
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\x00")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\x00")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def run_directional_rollout(
    resolved: ResolvedStairConfig,
    *,
    device: str | torch.device,
    position_weight: float,
    velocity_weight: float,
) -> DirectionalRollout:
    """Run the frozen 640-frame dense up/turn/down command sequence."""

    if not isinstance(resolved, ResolvedStairConfig):
        raise ContractError("directional config must be resolved before running")
    position_weight = _validate_weight(position_weight, "position weight")
    velocity_weight = _validate_weight(velocity_weight, "velocity weight")
    resolved_device = resolve_torch_device(device)
    if resolved_device != resolved.device:
        raise ContractError("rollout device does not match resolved config")
    directional_config = deepcopy(dict(resolved.resolved_config))
    directional_config["matcher"]["transition_joint_position_weight"] = (
        position_weight
    )
    directional_config["matcher"]["transition_joint_velocity_weight"] = (
        velocity_weight
    )
    dt = float(directional_config["dt"])
    directional_config.update(
        {
            "active_condition": "dense",
            "active_encoder": "dense",
            "active_weight": float(
                directional_config["conditions"]["dense"]["weight"]
            ),
            "duration_s": STEP_COUNT * dt,
            "step_count": STEP_COUNT,
            "directional_step_count": STEP_COUNT,
            "directional_ascent_stop": ASCENT_STOP,
            "directional_reversal_stop": REVERSAL_STOP,
            "transition_neighborhood_radius": (
                TRANSITION_NEIGHBORHOOD_RADIUS
            ),
        }
    )
    matcher = TorchMotionMatcher.from_folder(
        resolved.dataset.root,
        device=str(resolved_device),
        config=matcher_config_from_resolved(directional_config),
        extension=resolved.measurement_extension,
        reset_clip_path=directional_config["reset_clip"],
        emitted_window_validator=terrain_transition_validator_from_resolved(
            resolved
        ),
    )
    reset = matcher.reset()
    direction = np.asarray(
        directional_config["reference_direction_matcher_xy"], np.float64
    )
    speed = float(directional_config["command_speed_mps"])
    clip_index = {
        clip.relative_path: index
        for index, clip in enumerate(resolved.dataset.folder.clips)
    }
    rows: dict[str, list] = {
        "time_s": [],
        "command_velocity_world_xy": [],
        "command_heading_world_yaw": [],
        "selected_frame": [],
        "selected_clip_index": [],
        "previous_selected_clip_index": [],
        "searched": [],
        "transitioned": [],
        "transition_rejected": [],
        "terrain_safety_override": [],
        "terrain_safety_override_rank": [],
        "motion_feature_cost": [],
        "terrain_feature_cost": [],
        "total_feature_cost": [],
        "selected_total_cost": [],
        "selected_transition_position_cost": [],
        "selected_transition_velocity_cost": [],
        "selected_transition_continuity_cost": [],
        "step_time_ns": [],
        "search_time_ns": [],
        "joint_position": [],
        "joint_velocity": [],
        "root_position_world": [],
        "root_orientation_world_wxyz": [],
        "foot_position_world": [],
        "foot_velocity_world": [],
        "foot_clearance_m": [],
    }
    events: list[dict] = []
    measurement = resolved.measurement_extension
    previous_clip_index = clip_index[
        reset.diagnostics.selected_clip_path
    ]
    for step in range(STEP_COUNT):
        requested_velocity = direction * speed
        if step >= ASCENT_STOP:
            requested_velocity = -requested_velocity
        heading = math.atan2(
            float(requested_velocity[1]), float(requested_velocity[0])
        )
        prepared = matcher.prepare_step(
            (
                float(requested_velocity[0]),
                float(requested_velocity[1]),
            ),
            heading,
            dt=dt,
        )
        result = matcher.commit(prepared)
        diagnostics = result.diagnostics
        body = (
            result.dense_feature_body_position_window[0]
            .detach()
            .to("cpu")
            .numpy()
        )
        body_velocity = (
            result.dense_feature_body_velocity_window[0]
            .detach()
            .to("cpu")
            .numpy()
        )
        foot = body[1:]
        foot_xy = torch.tensor(
            foot[:, :2], dtype=torch.float32, device=resolved_device
        )
        scene_xy = measurement.alignment.matcher_to_scene_xy(foot_xy)
        surface_height = (
            measurement.query_grid.sample_xy(scene_xy)
            .detach()
            .to("cpu")
            .numpy()
        )
        clearance = foot[:, 2] - surface_height

        rows["time_s"].append(np.float32(step * dt))
        rows["command_velocity_world_xy"].append(
            requested_velocity.astype(np.float32)
        )
        rows["command_heading_world_yaw"].append(np.float32(heading))
        rows["selected_frame"].append(np.int64(diagnostics.selected_frame))
        rows["selected_clip_index"].append(
            np.int32(clip_index[diagnostics.selected_clip_path])
        )
        rows["previous_selected_clip_index"].append(
            np.int32(previous_clip_index)
        )
        rows["searched"].append(np.bool_(diagnostics.searched))
        rows["transitioned"].append(np.bool_(diagnostics.transitioned))
        rows["transition_rejected"].append(
            np.bool_(diagnostics.transition_rejected)
        )
        rows["terrain_safety_override"].append(
            np.bool_(diagnostics.terrain_safety_override)
        )
        rows["terrain_safety_override_rank"].append(
            np.int32(diagnostics.terrain_safety_override_rank)
        )
        rows["motion_feature_cost"].append(
            np.float32(diagnostics.motion_feature_cost)
        )
        rows["terrain_feature_cost"].append(
            np.float32(diagnostics.extension_feature_cost)
        )
        rows["total_feature_cost"].append(
            np.float32(diagnostics.selected_feature_cost)
        )
        rows["selected_total_cost"].append(
            np.float32(diagnostics.selected_total_cost)
        )
        rows["selected_transition_position_cost"].append(
            np.float32(diagnostics.selected_transition_position_cost)
        )
        rows["selected_transition_velocity_cost"].append(
            np.float32(diagnostics.selected_transition_velocity_cost)
        )
        rows["selected_transition_continuity_cost"].append(
            np.float32(diagnostics.selected_transition_continuity_cost)
        )
        rows["step_time_ns"].append(np.int64(diagnostics.step_time_ns))
        rows["search_time_ns"].append(
            np.int64(
                -1
                if diagnostics.search_time_ns is None
                else diagnostics.search_time_ns
            )
        )
        rows["joint_position"].append(
            result.joint_position.detach().to("cpu").numpy()
        )
        rows["joint_velocity"].append(
            result.joint_velocity.detach().to("cpu").numpy()
        )
        rows["root_position_world"].append(
            result.root_position_world.detach().to("cpu").numpy()
        )
        rows["root_orientation_world_wxyz"].append(
            result.root_orientation_world_wxyz.detach().to("cpu").numpy()
        )
        rows["foot_position_world"].append(foot)
        rows["foot_velocity_world"].append(body_velocity[1:])
        rows["foot_clearance_m"].append(clearance.astype(np.float32))
        if (
            diagnostics.searched
            or diagnostics.transitioned
            or diagnostics.transition_rejected
            or diagnostics.terrain_safety_override
        ):
            events.append(
                {
                    "sequence": int(diagnostics.sequence),
                    "step": step,
                    "time_s": float(step * dt),
                    "phase": directional_phase(step),
                    "selected_clip_path": diagnostics.selected_clip_path,
                    "selected_frame": int(diagnostics.selected_frame),
                    "searched": bool(diagnostics.searched),
                    "transitioned": bool(diagnostics.transitioned),
                    "transition_rejected": bool(
                        diagnostics.transition_rejected
                    ),
                    "terrain_safety_override": bool(
                        diagnostics.terrain_safety_override
                    ),
                    "terrain_safety_override_rank": int(
                        diagnostics.terrain_safety_override_rank
                    ),
                    "motion_feature_cost": float(
                        diagnostics.motion_feature_cost
                    ),
                    "terrain_feature_cost": float(
                        diagnostics.extension_feature_cost
                    ),
                    "total_feature_cost": float(
                        diagnostics.selected_feature_cost
                    ),
                    "selected_total_cost": float(
                        diagnostics.selected_total_cost
                    ),
                    "selected_transition_position_cost": float(
                        diagnostics.selected_transition_position_cost
                    ),
                    "selected_transition_velocity_cost": float(
                        diagnostics.selected_transition_velocity_cost
                    ),
                    "selected_transition_continuity_cost": float(
                        diagnostics.selected_transition_continuity_cost
                    ),
                }
            )
        previous_clip_index = clip_index[
            diagnostics.selected_clip_path
        ]

    arrays = {
        name: _owned_readonly(np.asarray(values))
        for name, values in rows.items()
    }
    metrics = compute_directional_metrics(
        arrays,
        dt=dt,
        position_weight=position_weight,
        velocity_weight=velocity_weight,
    )
    metrics.update(
        {
            "dataset_manifest_sha256": resolved.dataset.manifest_sha256,
            "motion_inventory_sha256": matcher.motion_inventory_sha256,
            "device": str(resolved_device),
            "torch_version": torch.__version__,
            "cuda_device_name": (
                torch.cuda.get_device_name(resolved_device)
                if resolved_device.type == "cuda"
                else None
            ),
        }
    )
    resolved_hash = _json_sha256(directional_config)
    deterministic_hash = _directional_rollout_hash(
        resolved_hash, arrays
    )
    metrics["resolved_config_sha256"] = resolved_hash
    metrics["deterministic_sha256"] = deterministic_hash
    return DirectionalRollout(
        arrays=MappingProxyType(arrays),
        metrics=MappingProxyType(metrics),
        events=tuple(MappingProxyType(event) for event in events),
        resolved_config=MappingProxyType(directional_config),
        resolved_config_sha256=resolved_hash,
        deterministic_sha256=deterministic_hash,
    )


def _remove_real_directory(path: Path) -> None:
    for item in sorted(
        path.rglob("*"), key=lambda value: len(value.parts), reverse=True
    ):
        if item.is_symlink() or item.is_file():
            item.unlink()
        elif item.is_dir():
            item.rmdir()
    path.rmdir()


def _validated_lexical_output_path(output: str | Path) -> Path:
    path = Path(os.path.abspath(os.fspath(output)))
    if path.is_symlink():
        raise ContractError(
            "directional rollout output must not be a symlink"
        )
    return path


def save_directional_rollout(
    rollout: DirectionalRollout,
    output: str | Path,
) -> None:
    """Transactionally save pickle-free arrays and canonical JSON artifacts."""

    if not isinstance(rollout, DirectionalRollout):
        raise ContractError("only a DirectionalRollout can be saved")
    output = _validated_lexical_output_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    committed = False
    try:
        for name, value in rollout.arrays.items():
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ContractError(
                    f"directional array must be pickle-free: {name}"
                )
            if array.dtype.kind in "fc" and not np.isfinite(array).all():
                raise ContractError(
                    f"directional array must be finite: {name}"
                )
        np.savez(staging / "rollout.npz", **rollout.arrays)
        metrics = deepcopy(dict(rollout.metrics))
        metrics["rollout_npz_sha256"] = _file_sha256(
            staging / "rollout.npz"
        )
        (staging / "metrics.json").write_bytes(
            _canonical_json_bytes(metrics)
        )
        (staging / "events.jsonl").write_bytes(
            b"".join(
                _canonical_json_bytes(dict(event))
                for event in rollout.events
            )
        )
        (staging / "resolved_config.json").write_bytes(
            _canonical_json_bytes(dict(rollout.resolved_config))
        )
        if output.exists():
            if not output.is_dir() or output.is_symlink():
                raise ContractError(
                    "directional rollout output must be a real directory"
                )
            backup = output.with_name(output.name + ".previous")
            if backup.exists():
                raise ContractError(
                    "directional rollout backup path already exists"
                )
            os.replace(output, backup)
            try:
                os.replace(staging, output)
                committed = True
            finally:
                if committed:
                    _remove_real_directory(backup)
                elif backup.exists():
                    os.replace(backup, output)
        else:
            os.replace(staging, output)
            committed = True
    finally:
        if not committed and staging.exists():
            _remove_real_directory(staging)


def _argument_weight(value: str) -> float:
    try:
        weight = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "weight must be finite and non-negative"
        ) from error
    if not math.isfinite(weight) or weight < 0.0:
        raise argparse.ArgumentTypeError(
            "weight must be finite and non-negative"
        )
    return weight


def build_directional_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic 640-frame Torch stair up/turn/down "
            "benchmark."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--position-weight", type=_argument_weight, default=0.0
    )
    parser.add_argument(
        "--velocity-weight", type=_argument_weight, default=0.0
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_directional_argument_parser().parse_args(argv)
    output = _validated_lexical_output_path(args.output)
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    raw = load_experiment_config(args.config)
    resolved = resolve_stair_config(args.dataset, raw, device=device)
    rollout = run_directional_rollout(
        resolved,
        device=device,
        position_weight=args.position_weight,
        velocity_weight=args.velocity_weight,
    )
    save_directional_rollout(rollout, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "deterministic_sha256": rollout.deterministic_sha256,
                "metrics": dict(rollout.metrics),
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
