"""Deterministic 50 Hz kinematic A/B rollout for Torch stair matching.

This benchmark calls only the transactional motion-matcher API.  It performs no
physics integration, SONIC inference, rendering, or reconstructed state update.
Timing is recorded as diagnostic evidence but excluded from deterministic
identity.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import argparse
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
from .torch_motion_matcher import MatcherConfig, TorchMotionMatcher
from .torch_motion_features import resolve_torch_device
from .torch_terrain_features import (
    DENSE_FORWARD_M,
    DENSE_LATERAL_M,
    TerrainDataset,
    TerrainFeatureExtension,
    TerrainFootClearanceValidator,
)


EXPERIMENT_SCHEMA = "g1-torch-stair-small-experiment/v1"
CONDITIONS = ("flat", "legacy", "dense")
_TIMING_ARRAYS = frozenset(("search_time_ns", "step_time_ns"))
_MATCHER_INTEGER_FIELDS = frozenset(
    (
        "search_interval_steps",
        "exclusion_frames",
        "transition_window_candidate_count",
    )
)
_MATCHER_FLOAT_FIELDS = frozenset(
    (
        "acceleration_mps2",
        "deceleration_mps2",
        "yaw_rate_rad_s",
        "stop_speed_mps",
        "reversal_speed_mps",
        "transition_penalty",
        "inertialization_halflife_s",
        "transition_joint_position_weight",
        "transition_joint_velocity_weight",
        "transition_window_jerk_weight",
        "transition_settle_duration_s",
        "transition_settle_penalty",
    )
)
_MATCHER_NONNEGATIVE_FLOAT_FIELDS = frozenset(
    (
        "transition_settle_duration_s",
        "transition_settle_penalty",
        "transition_joint_position_weight",
        "transition_joint_velocity_weight",
        "transition_window_jerk_weight",
    )
)


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


def load_experiment_config(path: str | Path) -> dict:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ContractError(f"cannot load experiment config: {path}") from error
    _validate_base_config(value)
    return deepcopy(value)


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        suffix = " finite and positive" if positive else " finite"
        raise ContractError(f"{name} must be{suffix}")
    return float(value)


def _validate_base_config(config: object) -> None:
    if not isinstance(config, dict):
        raise ContractError("experiment config must be an object")
    if config.get("schema") != EXPERIMENT_SCHEMA:
        raise ContractError("experiment config schema is invalid")
    dt = _finite_number(config.get("dt"), "dt", positive=True)
    if abs(dt - 0.02) > 1e-12:
        raise ContractError("experiment dt must equal 0.02")
    duration = _finite_number(
        config.get("duration_s"), "duration_s", positive=True
    )
    steps = duration / dt
    if abs(steps - round(steps)) > 1e-9:
        raise ContractError("duration_s must contain an integral number of steps")
    matcher = config.get("matcher")
    expected_matcher_fields = (
        _MATCHER_INTEGER_FIELDS | _MATCHER_FLOAT_FIELDS
    )
    if (
        not isinstance(matcher, dict)
        or set(matcher) != expected_matcher_fields
    ):
        raise ContractError("experiment matcher fields are invalid")
    search_interval = matcher["search_interval_steps"]
    exclusion_frames = matcher["exclusion_frames"]
    if (
        type(search_interval) is not int
        or search_interval < 1
    ):
        raise ContractError(
            "matcher search_interval_steps must be a positive integer"
        )
    if type(exclusion_frames) is not int or exclusion_frames < 0:
        raise ContractError(
            "matcher exclusion_frames must be a non-negative integer"
        )
    candidate_count = matcher["transition_window_candidate_count"]
    if type(candidate_count) is not int or candidate_count < 1:
        raise ContractError(
            "matcher transition_window_candidate_count must be a "
            "positive integer"
        )
    for name in _MATCHER_FLOAT_FIELDS:
        value = _finite_number(
            matcher[name],
            f"matcher {name}",
            positive=name not in _MATCHER_NONNEGATIVE_FLOAT_FIELDS,
        )
        if name in _MATCHER_NONNEGATIVE_FLOAT_FIELDS and value < 0.0:
            raise ContractError(
                f"matcher {name} must be finite and non-negative"
            )
    preview_steps = config.get("terrain_transition_preview_steps")
    if (
        type(preview_steps) is not int
        or not 1 <= preview_steps <= 46
    ):
        raise ContractError(
            "terrain_transition_preview_steps must be an integer in [1, 46]"
        )
    for name in ("query_scene", "reset_clip"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ContractError(f"{name} must be a non-empty relative path")
    conditions = config.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(CONDITIONS):
        raise ContractError("experiment conditions must be exactly flat/legacy/dense")
    expected_encoder = {"flat": None, "legacy": "legacy", "dense": "dense"}
    for condition, encoder in expected_encoder.items():
        descriptor = conditions.get(condition)
        if not isinstance(descriptor, dict) or descriptor.get("encoder") != encoder:
            raise ContractError(f"condition {condition} encoder is invalid")
        weight = _finite_number(
            descriptor.get("weight"), f"{condition} weight"
        )
        if (condition == "flat" and weight != 0.0) or (
            condition != "flat" and weight <= 0.0
        ):
            raise ContractError(f"condition {condition} weight is invalid")
    acceptance = config.get("acceptance")
    required = (
        "latest_stair_selection_before_riser_m",
        "minimum_reference_horizontal_progress_ratio",
        "minimum_reference_root_height_gain_ratio",
        "minimum_foot_clearance_m",
        "maximum_penetration_integral_ratio_vs_flat",
    )
    if not isinstance(acceptance, dict) or set(acceptance) != set(required):
        raise ContractError("experiment acceptance thresholds are invalid")
    for name in required:
        _finite_number(acceptance[name], f"acceptance {name}")


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _inverse_rotate_xy(vector: np.ndarray, yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    x, y = (float(value) for value in vector)
    return np.array(
        [cosine * x + sine * y, -sine * x + cosine * y],
        np.float64,
    )


@dataclass(frozen=True)
class ResolvedStairConfig:
    dataset: TerrainDataset
    measurement_extension: TerrainFeatureExtension
    resolved_config: Mapping
    base_config_sha256: str
    device: torch.device


def matcher_config_from_resolved(config: Mapping) -> MatcherConfig:
    """Reconstruct the exact matcher parameters recorded by the experiment."""

    values = config["matcher"]
    return MatcherConfig(
        dt=float(config["dt"]),
        search_interval_steps=int(values["search_interval_steps"]),
        acceleration_mps2=float(values["acceleration_mps2"]),
        deceleration_mps2=float(values["deceleration_mps2"]),
        yaw_rate_rad_s=float(values["yaw_rate_rad_s"]),
        stop_speed_mps=float(values["stop_speed_mps"]),
        reversal_speed_mps=float(values["reversal_speed_mps"]),
        exclusion_frames=int(values["exclusion_frames"]),
        transition_penalty=float(values["transition_penalty"]),
        inertialization_halflife_s=float(
            values["inertialization_halflife_s"]
        ),
        transition_joint_position_weight=float(
            values["transition_joint_position_weight"]
        ),
        transition_joint_velocity_weight=float(
            values["transition_joint_velocity_weight"]
        ),
        transition_settle_duration_s=float(
            values["transition_settle_duration_s"]
        ),
        transition_settle_penalty=float(
            values["transition_settle_penalty"]
        ),
        transition_window_jerk_weight=float(
            values["transition_window_jerk_weight"]
        ),
        transition_window_candidate_count=int(
            values["transition_window_candidate_count"]
        ),
    )


def terrain_transition_validator_from_resolved(
    resolved: ResolvedStairConfig,
) -> TerrainFootClearanceValidator:
    """Build the dense condition's emitted-window clearance validator."""

    return TerrainFootClearanceValidator(
        extension=resolved.measurement_extension,
        preview_steps=int(
            resolved.resolved_config[
                "terrain_transition_preview_steps"
            ]
        ),
        minimum_clearance_m=float(
            resolved.resolved_config["acceptance"][
                "minimum_foot_clearance_m"
            ]
        ),
    )


def resolve_stair_config(
    dataset_root: str | Path,
    config: Mapping,
    *,
    device: str | torch.device,
) -> ResolvedStairConfig:
    raw = deepcopy(dict(config))
    _validate_base_config(raw)
    resolved_device = resolve_torch_device(device)
    dataset = TerrainDataset.load(dataset_root, device=resolved_device)
    query_scene = raw["query_scene"]
    matches = [
        clip
        for clip in dataset.folder.clips
        if clip.relative_path == query_scene
    ]
    if len(matches) != 1:
        raise ContractError(
            f"query scene must resolve exactly once: {query_scene}"
        )
    if sum(
        clip.relative_path == raw["reset_clip"]
        for clip in dataset.folder.clips
    ) != 1:
        raise ContractError(
            f"reset clip must resolve exactly once: {raw['reset_clip']}"
        )
    clip = matches[0]
    root = dataset.folder.layout.root_body_index
    root_position = clip.body_position_world[:, root].astype(np.float64)
    ascent_end = int(np.argmax(root_position[:, 2]))
    if ascent_end < 10:
        raise ContractError(
            "query scene must begin with a recorded stair ascent segment"
        )
    ascent_position = root_position[: ascent_end + 1]
    delta_xy = ascent_position[:, :2] - ascent_position[0, :2]
    endpoint = delta_xy[-1]
    endpoint_norm = float(np.linalg.norm(endpoint))
    if endpoint_norm <= 1e-6:
        principal = np.linalg.svd(delta_xy, full_matrices=False)[2][0]
        if float(delta_xy @ principal).max() < abs(float(delta_xy @ -principal).max()):
            principal = -principal
        direction_scene = principal
    else:
        direction_scene = endpoint / endpoint_norm
    projected = delta_xy @ direction_scene
    reference_progress = float(projected.max())
    reference_height_gain = float(
        ascent_position[:, 2].max() - ascent_position[0, 2]
    )
    if reference_progress <= 0.1 or reference_height_gain <= 0.01:
        raise ContractError(
            "query scene has insufficient horizontal or vertical reference motion"
        )
    source_duration = ascent_end / float(clip.fps)
    command_speed = reference_progress / source_duration
    source_yaw = _yaw_from_wxyz(
        clip.body_quaternion_world_wxyz[0, root]
    )
    direction_matcher = _inverse_rotate_xy(direction_scene, source_yaw)
    direction_matcher /= np.linalg.norm(direction_matcher)

    measurement = TerrainFeatureExtension.for_condition(
        dataset,
        condition="dense",
        query_scene=query_scene,
        weight=float(raw["conditions"]["dense"]["weight"]),
    )
    samples = np.linspace(0.0, reference_progress, 1001, dtype=np.float32)
    matcher_points = torch.tensor(
        samples[:, None] * direction_matcher[None, :],
        dtype=torch.float32,
        device=resolved_device,
    )
    scene_points = measurement.alignment.matcher_to_scene_xy(matcher_points)
    heights = measurement.query_grid.sample_xy(scene_points).cpu().numpy()
    base_height = float(heights[0])
    riser = np.flatnonzero(heights > base_height + 0.05)
    if not len(riser):
        raise ContractError("query scene has no first riser above 0.05 m")
    first_riser = float(samples[int(riser[0])])

    resolved = deepcopy(raw)
    resolved.update(
        {
            "dataset_root": str(dataset.root),
            "dataset_manifest_sha256": dataset.manifest_sha256,
            "motion_inventory_sha256": dataset.folder.inventory_sha256,
            "reference_direction_scene_xy": direction_scene.tolist(),
            "reference_direction_matcher_xy": direction_matcher.tolist(),
            "reference_horizontal_progress_m": reference_progress,
            "reference_root_height_gain_m": reference_height_gain,
            "reference_source_duration_s": source_duration,
            "reference_ascent_end_frame": ascent_end,
            "command_speed_mps": command_speed,
            "command_heading_matcher_yaw": math.atan2(
                float(direction_matcher[1]), float(direction_matcher[0])
            ),
            "first_riser_progress_m": first_riser,
            "upper_landing_progress_m": 0.9 * reference_progress,
            "upper_landing_height_gain_m": 0.8 * reference_height_gain,
            "step_count": int(round(float(raw["duration_s"]) / float(raw["dt"]))),
        }
    )
    return ResolvedStairConfig(
        dataset=dataset,
        measurement_extension=measurement,
        resolved_config=MappingProxyType(resolved),
        base_config_sha256=_json_sha256(raw),
        device=resolved_device,
    )


@dataclass(frozen=True)
class StairRollout:
    condition: str
    arrays: Mapping[str, np.ndarray]
    metrics: Mapping
    events: Sequence[Mapping]
    resolved_config: Mapping
    resolved_config_sha256: str
    deterministic_sha256: str


def _owned_readonly(array: np.ndarray) -> np.ndarray:
    output = np.ascontiguousarray(array)
    output.setflags(write=False)
    return output


def _deterministic_rollout_hash(
    condition: str,
    resolved_config_sha256: str,
    arrays: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(condition.encode("utf-8"))
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


def _percentiles(values: np.ndarray) -> dict:
    finite = np.asarray(values, np.float64)
    finite = finite[finite >= 0.0]
    if not len(finite):
        return {"p50": None, "p95": None, "p99": None, "maximum": None}
    return {
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "maximum": float(finite.max()),
    }


def run_stair_rollout(
    config: ResolvedStairConfig,
    condition: str,
    *,
    device: str | torch.device,
) -> StairRollout:
    if not isinstance(config, ResolvedStairConfig):
        raise ContractError("rollout config must be resolved before running")
    if condition not in CONDITIONS:
        raise ContractError("rollout condition must be flat, legacy, or dense")
    resolved_device = resolve_torch_device(device)
    if resolved_device != config.device:
        raise ContractError("rollout device does not match resolved config")
    descriptor = config.resolved_config["conditions"][condition]
    extension = None
    if descriptor["encoder"] is not None:
        extension = TerrainFeatureExtension.for_condition(
            config.dataset,
            condition=descriptor["encoder"],
            query_scene=config.resolved_config["query_scene"],
            weight=float(descriptor["weight"]),
        )
    matcher = TorchMotionMatcher.from_folder(
        config.dataset.root,
        device=str(resolved_device),
        config=matcher_config_from_resolved(config.resolved_config),
        extension=extension,
        reset_clip_path=config.resolved_config["reset_clip"],
        emitted_window_validator=(
            terrain_transition_validator_from_resolved(config)
            if condition == "dense"
            else None
        ),
    )
    reset = matcher.reset()
    initial_root = reset.root_position_world.detach().cpu().numpy()
    steps = int(config.resolved_config["step_count"])
    dt = float(config.resolved_config["dt"])
    direction = np.asarray(
        config.resolved_config["reference_direction_matcher_xy"], np.float64
    )
    speed = float(config.resolved_config["command_speed_mps"])
    command_velocity = direction * speed
    heading = float(config.resolved_config["command_heading_matcher_yaw"])

    rows: dict[str, list] = {
        "time_s": [],
        "command_velocity_world_xy": [],
        "command_heading_world_yaw": [],
        "selected_frame": [],
        "selected_clip_index": [],
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
        "root_position_world": [],
        "root_orientation_world_wxyz": [],
        "feature_body_position_world": [],
        "terrain_patch_position_world": [],
        "foot_clearance_m": [],
        "progress_m": [],
    }
    events: list[dict] = []
    clip_index = {
        clip.relative_path: index
        for index, clip in enumerate(config.dataset.folder.clips)
    }
    measurement = config.measurement_extension
    for step in range(steps):
        active_command = (
            step * dt
            < float(config.resolved_config["reference_source_duration_s"])
        )
        requested_velocity = (
            command_velocity if active_command else np.zeros(2, np.float64)
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
        body_xy = torch.tensor(
            body[:, :2], dtype=torch.float32, device=resolved_device
        )
        scene_xy = measurement.alignment.matcher_to_scene_xy(body_xy)
        surface_height = (
            measurement.query_grid.sample_xy(scene_xy).detach().cpu().numpy()
        )
        clearance = body[1:, 2] - surface_height[1:]
        root_position = result.root_position_world.detach().cpu().numpy()
        root_quaternion = result.root_orientation_world_wxyz
        w, x, y, z = root_quaternion
        root_yaw = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        patch_forward, patch_lateral = torch.meshgrid(
            torch.tensor(
                DENSE_FORWARD_M,
                dtype=torch.float32,
                device=resolved_device,
            ),
            torch.tensor(
                DENSE_LATERAL_M,
                dtype=torch.float32,
                device=resolved_device,
            ),
            indexing="ij",
        )
        cosine = torch.cos(root_yaw)
        sine = torch.sin(root_yaw)
        patch_x = cosine * patch_forward - sine * patch_lateral
        patch_y = sine * patch_forward + cosine * patch_lateral
        patch_matcher_xy = torch.stack(
            (patch_x, patch_y), dim=-1
        ).reshape(-1, 2) + result.root_position_world[:2]
        patch_scene_xy = measurement.alignment.matcher_to_scene_xy(
            patch_matcher_xy
        )
        patch_height = measurement.query_grid.sample_xy(patch_scene_xy)
        patch_position = torch.cat(
            (patch_matcher_xy, patch_height[:, None]), dim=1
        ).detach().cpu().numpy()
        progress = float((root_position[:2] - initial_root[:2]) @ direction)

        rows["time_s"].append(np.float32(step * dt))
        rows["command_velocity_world_xy"].append(
            requested_velocity.astype(np.float32)
        )
        rows["command_heading_world_yaw"].append(np.float32(heading))
        rows["selected_frame"].append(np.int64(diagnostics.selected_frame))
        rows["selected_clip_index"].append(
            np.int32(clip_index[diagnostics.selected_clip_path])
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
            result.joint_position.detach().cpu().numpy()
        )
        rows["root_position_world"].append(root_position)
        rows["root_orientation_world_wxyz"].append(
            root_quaternion.detach().cpu().numpy()
        )
        rows["feature_body_position_world"].append(body)
        rows["terrain_patch_position_world"].append(patch_position)
        rows["foot_clearance_m"].append(clearance.astype(np.float32))
        rows["progress_m"].append(np.float32(progress))
        if (
            diagnostics.searched
            or diagnostics.transitioned
            or diagnostics.transition_rejected
            or diagnostics.terrain_safety_override
        ):
            events.append(
                {
                    "sequence": diagnostics.sequence,
                    "time_s": step * dt,
                    "selected_clip_path": diagnostics.selected_clip_path,
                    "selected_frame": diagnostics.selected_frame,
                    "searched": diagnostics.searched,
                    "transitioned": diagnostics.transitioned,
                    "transition_rejected": (
                        diagnostics.transition_rejected
                    ),
                    "terrain_safety_override": (
                        diagnostics.terrain_safety_override
                    ),
                    "terrain_safety_override_rank": (
                        diagnostics.terrain_safety_override_rank
                    ),
                    "motion_feature_cost": diagnostics.motion_feature_cost,
                    "terrain_feature_cost": diagnostics.extension_feature_cost,
                    "total_feature_cost": diagnostics.selected_feature_cost,
                    "selected_total_cost": diagnostics.selected_total_cost,
                    "selected_transition_position_cost": (
                        diagnostics.selected_transition_position_cost
                    ),
                    "selected_transition_velocity_cost": (
                        diagnostics.selected_transition_velocity_cost
                    ),
                    "selected_transition_continuity_cost": (
                        diagnostics.selected_transition_continuity_cost
                    ),
                }
            )

    arrays = {
        name: _owned_readonly(np.asarray(value))
        for name, value in rows.items()
    }
    # Progress and gains are reported from the first committed 20 ms row, which
    # is also viewer frame zero and common to every condition.
    first_root = arrays["root_position_world"][0]
    arrays["progress_m"] = _owned_readonly(
        (
            arrays["root_position_world"][:, :2] - first_root[:2]
        )
        @ direction
    ).astype(np.float32)
    arrays["progress_m"].setflags(write=False)
    root_gain = arrays["root_position_world"][:, 2] - first_root[2]
    penetration = np.maximum(0.0, -arrays["foot_clearance_m"])
    stair_indices = [
        index
        for index, selected in enumerate(arrays["selected_clip_index"])
        if config.dataset.folder.clips[int(selected)].relative_path
        != config.resolved_config["reset_clip"]
    ]
    first_selection_progress = (
        float(arrays["progress_m"][stair_indices[0]])
        if stair_indices
        else None
    )
    maximum_progress = float(arrays["progress_m"].max())
    maximum_height_gain = float(root_gain.max())
    positive_rescue_ranks = arrays["terrain_safety_override_rank"][
        arrays["terrain_safety_override_rank"] > 0
    ]
    metrics = {
        "condition": condition,
        "dataset_manifest_sha256": config.dataset.manifest_sha256,
        "motion_inventory_sha256": matcher.motion_inventory_sha256,
        "encoder_dimension": 0 if extension is None else extension.dimension,
        "encoder_weight": float(descriptor["weight"]),
        "device": str(resolved_device),
        "torch_version": torch.__version__,
        "cuda_device_name": (
            torch.cuda.get_device_name(resolved_device)
            if resolved_device.type == "cuda"
            else None
        ),
        "step_count": steps,
        "rejected_transition_count": int(
            arrays["transition_rejected"].sum()
        ),
        "terrain_safety_override_count": int(
            arrays["terrain_safety_override"].sum()
        ),
        "maximum_terrain_safety_override_rank": int(
            arrays["terrain_safety_override_rank"].max()
        ),
        "p95_positive_terrain_safety_override_rank": (
            float(np.percentile(positive_rescue_ranks, 95))
            if positive_rescue_ranks.size
            else 0.0
        ),
        "first_stair_selection_progress_m": first_selection_progress,
        "maximum_progress_m": maximum_progress,
        "maximum_root_height_gain_m": maximum_height_gain,
        "minimum_foot_clearance_m": float(
            arrays["foot_clearance_m"].min()
        ),
        "integrated_foot_penetration_m_s": float(
            penetration.sum(dtype=np.float64) * dt
        ),
        "reached_upper_landing": bool(
            maximum_progress
            >= float(config.resolved_config["upper_landing_progress_m"])
            and maximum_height_gain
            >= float(config.resolved_config["upper_landing_height_gain_m"])
        ),
        "step_latency_ns": _percentiles(arrays["step_time_ns"]),
        "search_latency_ns": _percentiles(arrays["search_time_ns"]),
    }
    condition_config = deepcopy(dict(config.resolved_config))
    condition_config["active_condition"] = condition
    condition_config["active_encoder"] = descriptor["encoder"]
    condition_config["active_weight"] = float(descriptor["weight"])
    resolved_hash = _json_sha256(condition_config)
    deterministic_hash = _deterministic_rollout_hash(
        condition, resolved_hash, arrays
    )
    metrics["resolved_config_sha256"] = resolved_hash
    metrics["deterministic_sha256"] = deterministic_hash
    return StairRollout(
        condition=condition,
        arrays=MappingProxyType(arrays),
        metrics=MappingProxyType(metrics),
        events=tuple(MappingProxyType(event) for event in events),
        resolved_config=MappingProxyType(condition_config),
        resolved_config_sha256=resolved_hash,
        deterministic_sha256=deterministic_hash,
    )


def evaluate_dense_acceptance(
    dense_metrics: Mapping,
    flat_metrics: Mapping,
    resolved_config: Mapping,
) -> dict:
    acceptance = resolved_config["acceptance"]
    selection = dense_metrics.get("first_stair_selection_progress_m")
    first_riser = float(resolved_config["first_riser_progress_m"])
    latest = float(acceptance["latest_stair_selection_before_riser_m"])
    reference_progress = float(
        resolved_config["reference_horizontal_progress_m"]
    )
    reference_height = float(resolved_config["reference_root_height_gain_m"])
    progress_ok = float(dense_metrics["maximum_progress_m"]) >= (
        float(acceptance["minimum_reference_horizontal_progress_ratio"])
        * reference_progress
    )
    height_ok = float(dense_metrics["maximum_root_height_gain_m"]) >= (
        float(acceptance["minimum_reference_root_height_gain_ratio"])
        * reference_height
    )
    clearance_ok = float(dense_metrics["minimum_foot_clearance_m"]) >= float(
        acceptance["minimum_foot_clearance_m"]
    )
    landing_advantage = bool(dense_metrics.get("reached_upper_landing")) and not bool(
        flat_metrics.get("reached_upper_landing")
    )
    flat_penetration = float(
        flat_metrics["integrated_foot_penetration_m_s"]
    )
    dense_penetration = float(
        dense_metrics["integrated_foot_penetration_m_s"]
    )
    penetration_ratio_ok = (
        flat_penetration > 0.0
        and dense_penetration
        <= float(
            acceptance["maximum_penetration_integral_ratio_vs_flat"]
        )
        * flat_penetration
    )
    criteria = {
        "stair_selected_by_riser_window": (
            selection is not None
            and float(selection) <= first_riser - latest
        ),
        "horizontal_progress": progress_ok,
        "root_height_gain": height_ok,
        "foot_clearance": clearance_ok,
        "landing_or_penetration_improvement": (
            landing_advantage or penetration_ratio_ok
        ),
    }
    return {"passed": all(criteria.values()), "criteria": criteria}


def save_stair_rollout(rollout: StairRollout, output: str | Path) -> None:
    if not isinstance(rollout, StairRollout):
        raise ContractError("only a StairRollout can be saved")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    committed = False
    try:
        np.savez(staging / "rollout.npz", **rollout.arrays)
        metrics = dict(rollout.metrics)
        metrics["rollout_npz_sha256"] = _file_sha256(
            staging / "rollout.npz"
        )
        (staging / "metrics.json").write_bytes(_canonical_json_bytes(metrics))
        event_bytes = b"".join(
            _canonical_json_bytes(dict(event)) for event in rollout.events
        )
        (staging / "events.jsonl").write_bytes(event_bytes)
        (staging / "resolved_config.json").write_bytes(
            _canonical_json_bytes(dict(rollout.resolved_config))
        )
        if output.exists():
            if not output.is_dir() or output.is_symlink():
                raise ContractError("rollout output must be a real directory")
            backup = output.with_name(output.name + ".previous")
            if backup.exists():
                raise ContractError("rollout backup path already exists")
            os.replace(output, backup)
            try:
                os.replace(staging, output)
                committed = True
            finally:
                if committed:
                    for path in sorted(
                        backup.rglob("*"), key=lambda item: len(item.parts), reverse=True
                    ):
                        if path.is_file():
                            path.unlink()
                        elif path.is_dir():
                            path.rmdir()
                    backup.rmdir()
                elif backup.exists():
                    os.replace(backup, output)
        else:
            os.replace(staging, output)
            committed = True
    finally:
        if not committed and staging.exists():
            for path in sorted(
                staging.rglob("*"), key=lambda item: len(item.parts), reverse=True
            ):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_rollout_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the native 50 Hz Torch stair kinematics experiment."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_rollout_argument_parser().parse_args(argv)
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    raw = load_experiment_config(args.config)
    resolved = resolve_stair_config(args.dataset, raw, device=device)
    output = Path(args.output).resolve()
    runs: dict[str, StairRollout] = {}
    for condition in args.conditions:
        run = run_stair_rollout(resolved, condition, device=device)
        save_stair_rollout(run, output / condition)
        runs[condition] = run
        print(
            json.dumps(
                {
                    "condition": condition,
                    "output": str(output / condition),
                    "deterministic_sha256": run.deterministic_sha256,
                    "metrics": dict(run.metrics),
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
    if "flat" in runs and "dense" in runs:
        acceptance = evaluate_dense_acceptance(
            runs["dense"].metrics,
            runs["flat"].metrics,
            resolved.resolved_config,
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "acceptance.json").write_bytes(
            _canonical_json_bytes(acceptance)
        )
        print(json.dumps({"acceptance": acceptance}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
