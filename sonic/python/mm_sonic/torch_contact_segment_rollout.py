"""Deterministic authoritative-FK rollout for contact-segment matching."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .joints import ContractError
from .torch_contact_segments import (
    ContactSegmentIndex,
    TerrainContactSegmentPolicy,
)
from .torch_g1_fk import MujocoG1FootKinematics
from .torch_motion_matcher import TorchMotionMatcher
from .torch_terrain_rollout import (
    EXPERIMENT_SCHEMA,
    ResolvedStairConfig,
    matcher_config_from_resolved,
    resolve_stair_config,
    terrain_transition_validator_from_resolved,
)


CONTACT_EXPERIMENT_SCHEMA = "g1-torch-contact-segment-experiment/v1"
DEFAULT_ACCEPTANCE = MappingProxyType(
    {
        "maximum_unsupported_fraction": 0.15,
        "maximum_longest_unsupported_frames": 10,
        "maximum_stance_slide_m": 0.35,
        "minimum_foot_clearance_m": -0.03,
        "maximum_lost_source_support_fraction": 0.05,
        "require_sequential_commitment": True,
        "require_upper_landing": True,
    }
)


@dataclass(frozen=True)
class AcceptanceGate:
    name: str
    passed: bool
    observed: object
    threshold: object


@dataclass(frozen=True)
class ContactSegmentRollout:
    arrays: Mapping[str, np.ndarray]
    metrics: Mapping[str, object]
    gates: tuple[AcceptanceGate, ...]
    resolved_config: Mapping[str, object]

    @property
    def accepted(self) -> bool:
        return all(gate.passed for gate in self.gates)


def evaluate_contact_segment_acceptance(
    metrics: Mapping[str, object],
    thresholds: Mapping[str, object] = DEFAULT_ACCEPTANCE,
) -> tuple[AcceptanceGate, ...]:
    """Evaluate the frozen inclusive contact-segment acceptance table."""

    try:
        records = (
            AcceptanceGate(
                "unsupported_fraction",
                float(metrics["unsupported_fraction"])
                <= float(thresholds["maximum_unsupported_fraction"]),
                float(metrics["unsupported_fraction"]),
                float(thresholds["maximum_unsupported_fraction"]),
            ),
            AcceptanceGate(
                "longest_unsupported",
                int(metrics["longest_unsupported_frames"])
                <= int(thresholds["maximum_longest_unsupported_frames"]),
                int(metrics["longest_unsupported_frames"]),
                int(thresholds["maximum_longest_unsupported_frames"]),
            ),
            AcceptanceGate(
                "stance_slide",
                float(metrics["stance_slide_m"])
                <= float(thresholds["maximum_stance_slide_m"]),
                float(metrics["stance_slide_m"]),
                float(thresholds["maximum_stance_slide_m"]),
            ),
            AcceptanceGate(
                "minimum_clearance",
                float(metrics["minimum_foot_clearance_m"])
                >= float(thresholds["minimum_foot_clearance_m"]),
                float(metrics["minimum_foot_clearance_m"]),
                float(thresholds["minimum_foot_clearance_m"]),
            ),
            AcceptanceGate(
                "lost_source_support",
                float(metrics["lost_source_support_fraction"])
                <= float(
                    thresholds["maximum_lost_source_support_fraction"]
                ),
                float(metrics["lost_source_support_fraction"]),
                float(
                    thresholds["maximum_lost_source_support_fraction"]
                ),
            ),
            AcceptanceGate(
                "committed_sequence",
                (
                    int(metrics["committed_sequence_violations"]) == 0
                    if bool(thresholds["require_sequential_commitment"])
                    else True
                ),
                int(metrics["committed_sequence_violations"]),
                0,
            ),
            AcceptanceGate(
                "upper_landing",
                (
                    bool(metrics["reached_upper_landing"])
                    if bool(thresholds["require_upper_landing"])
                    else True
                ),
                bool(metrics["reached_upper_landing"]),
                True,
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("contact-segment acceptance metrics are invalid") from error
    return records


def load_contact_segment_config(path: str | Path) -> dict:
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise ContractError(f"cannot load contact-segment config: {path}") from error
    if not isinstance(config, dict) or config.get("schema") != CONTACT_EXPERIMENT_SCHEMA:
        raise ContractError("contact-segment experiment schema is invalid")
    contact = config.get("contact_segments")
    acceptance = config.get("acceptance")
    if (
        not isinstance(contact, dict)
        or set(contact)
        != {
            "minimum_frames",
            "maximum_frames",
            "maximum_scene_xy_mismatch_m",
            "entry_inertialization_halflife_s",
            "flat_support_transition_cost_weight",
            "maximum_unsupported_frames",
            "maximum_lost_source_support_fraction",
        }
        or not isinstance(acceptance, dict)
        or set(acceptance) != set(DEFAULT_ACCEPTANCE)
    ):
        raise ContractError("contact-segment experiment thresholds are invalid")
    minimum = contact["minimum_frames"]
    maximum = contact["maximum_frames"]
    if type(minimum) is not int or type(maximum) is not int or not 1 <= minimum <= maximum:
        raise ContractError("contact-segment frame bounds are invalid")
    mismatch = contact["maximum_scene_xy_mismatch_m"]
    if (
        isinstance(mismatch, bool)
        or not isinstance(mismatch, (int, float))
        or not math.isfinite(float(mismatch))
        or float(mismatch) <= 0.0
    ):
        raise ContractError("contact-segment scene mismatch limit is invalid")
    entry_halflife = contact["entry_inertialization_halflife_s"]
    if (
        isinstance(entry_halflife, bool)
        or not isinstance(entry_halflife, (int, float))
        or not math.isfinite(float(entry_halflife))
        or float(entry_halflife) <= 0.0
    ):
        raise ContractError("contact-segment entry inertialization is invalid")
    flat_support_weight = contact["flat_support_transition_cost_weight"]
    if (
        isinstance(flat_support_weight, bool)
        or not isinstance(flat_support_weight, (int, float))
        or not math.isfinite(float(flat_support_weight))
        or float(flat_support_weight) < 0.0
    ):
        raise ContractError("flat support transition cost weight is invalid")
    return deepcopy(config)


def resolve_contact_segment_config(
    dataset_root: str | Path,
    config: Mapping[str, object],
    *,
    device: str,
) -> ResolvedStairConfig:
    """Reuse authenticated terrain resolution while retaining contact gates."""

    raw = deepcopy(dict(config))
    if raw.get("schema") != CONTACT_EXPERIMENT_SCHEMA:
        raise ContractError("contact-segment experiment schema is invalid")
    base = deepcopy(raw)
    base["schema"] = EXPERIMENT_SCHEMA
    base.pop("contact_segments", None)
    base["acceptance"] = {
        "latest_stair_selection_before_riser_m": 0.2,
        "maximum_penetration_integral_ratio_vs_flat": 0.5,
        "minimum_foot_clearance_m": float(
            raw["acceptance"]["minimum_foot_clearance_m"]
        ),
        "minimum_reference_horizontal_progress_ratio": 0.9,
        "minimum_reference_root_height_gain_ratio": 0.8,
    }
    resolved = resolve_stair_config(dataset_root, base, device=device)
    merged = dict(resolved.resolved_config)
    merged["schema"] = CONTACT_EXPERIMENT_SCHEMA
    merged["contact_segments"] = deepcopy(raw["contact_segments"])
    merged["acceptance"] = deepcopy(raw["acceptance"])
    return ResolvedStairConfig(
        dataset=resolved.dataset,
        measurement_extension=resolved.measurement_extension,
        resolved_config=MappingProxyType(merged),
        base_config_sha256=hashlib.sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        device=resolved.device,
    )


def _longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _rollout_metrics(
    arrays: Mapping[str, np.ndarray], resolved: ResolvedStairConfig
) -> dict[str, object]:
    support = arrays["emitted_support_mask"]
    supported = support.any(axis=1)
    source_supported = arrays["source_support_mask"].any(axis=1)
    lost = source_supported & ~supported
    feet = arrays["foot_position_world"]
    slide = 0.0
    for frame in range(1, len(feet)):
        persistent = support[frame - 1] & support[frame]
        if persistent.any():
            slide += float(
                np.linalg.norm(
                    feet[frame, persistent, :2]
                    - feet[frame - 1, persistent, :2],
                    axis=1,
                ).sum()
            )
    violations = 0
    for frame in range(1, len(feet)):
        if (
            arrays["segment_committed"][frame - 1]
            and arrays["segment_committed"][frame]
            and arrays["segment_start_frame"][frame - 1]
            == arrays["segment_start_frame"][frame]
            and arrays["selected_frame"][frame]
            != arrays["selected_frame"][frame - 1] + 1
        ):
            violations += 1
    direction = np.asarray(
        resolved.resolved_config["reference_direction_matcher_xy"], np.float64
    )
    root = arrays["root_position_world"]
    progress = (root[:, :2] - root[0, :2]) @ direction
    height_gain = root[:, 2] - root[0, 2]
    reached = bool(
        np.any(
            (progress >= resolved.resolved_config["upper_landing_progress_m"])
            & (
                height_gain
                >= resolved.resolved_config["upper_landing_height_gain_m"]
            )
        )
    )
    return {
        "unsupported_fraction": float((~supported).mean()),
        "longest_unsupported_frames": _longest_false_run(supported),
        "stance_slide_m": slide,
        "minimum_foot_clearance_m": float(
            arrays["foot_clearance_m"].min()
        ),
        "lost_source_support_fraction": (
            0.0
            if not source_supported.any()
            else float(lost.sum() / source_supported.sum())
        ),
        "committed_sequence_violations": violations,
        "reached_upper_landing": reached,
    }


def run_contact_segment_rollout(
    resolved: ResolvedStairConfig,
    g1_xml: str | Path,
) -> ContactSegmentRollout:
    contact = resolved.resolved_config["contact_segments"]
    index = ContactSegmentIndex.from_dataset(
        resolved.dataset,
        minimum_frames=int(contact["minimum_frames"]),
        maximum_frames=int(contact["maximum_frames"]),
    )
    fk = MujocoG1FootKinematics(g1_xml)
    policy = TerrainContactSegmentPolicy(
        index=index,
        extension=resolved.measurement_extension,
        foot_kinematics=fk,
        maximum_scene_xy_mismatch_m=float(
            contact["maximum_scene_xy_mismatch_m"]
        ),
        entry_inertialization_halflife_s=float(
            contact["entry_inertialization_halflife_s"]
        ),
        flat_support_transition_cost_weight=float(
            contact["flat_support_transition_cost_weight"]
        ),
    )
    matcher = TorchMotionMatcher.from_folder(
        resolved.dataset.root,
        device=str(resolved.device),
        config=matcher_config_from_resolved(resolved.resolved_config),
        extension=resolved.measurement_extension,
        reset_clip_path=resolved.resolved_config["reset_clip"],
        emitted_window_validator=terrain_transition_validator_from_resolved(
            resolved
        ),
        contact_segment_policy=policy,
    )
    matcher.reset()
    direction = np.asarray(
        resolved.resolved_config["reference_direction_matcher_xy"], np.float64
    )
    velocity = direction * float(resolved.resolved_config["command_speed_mps"])
    heading = float(resolved.resolved_config["command_heading_matcher_yaw"])
    clip_by_path = {
        clip.relative_path: index
        for index, clip in enumerate(resolved.dataset.folder.clips)
    }
    rows = {
        name: []
        for name in (
            "joint_position",
            "root_position_world",
            "root_orientation_world_wxyz",
            "foot_position_world",
            "foot_surface_height_m",
            "foot_clearance_m",
            "source_support_mask",
            "emitted_support_mask",
            "segment_committed",
            "segment_start_frame",
            "segment_end_frame",
            "segment_entering_foot",
            "segment_vertical_offset_m",
            "selected_clip_index",
            "selected_frame",
            "step_time_ns",
            "search_time_ns",
        )
    }
    steps = int(resolved.resolved_config["step_count"])
    for _ in range(steps):
        result = matcher.step(tuple(velocity.tolist()), heading)
        diagnostic = result.diagnostics
        clip_index = clip_by_path[diagnostic.selected_clip_path]
        feet = fk.foot_positions(
            result.joint_position[None],
            result.root_position_world[None],
            result.root_orientation_world_wxyz[None],
        )[0]
        import torch

        feet_tensor = torch.tensor(
            feet, dtype=torch.float32, device=resolved.device
        )
        scene_xy = resolved.measurement_extension.alignment.matcher_to_scene_xy(
            feet_tensor[:, :2]
        )
        surface = (
            resolved.measurement_extension.query_grid.sample_xy(scene_xy)
            .detach()
            .cpu()
            .numpy()
        )
        clearance = feet[:, 2] - surface
        emitted = (
            np.abs(clearance - 0.035) <= 0.020
        )
        source = index.support_mask(clip_index)[diagnostic.selected_frame]
        rows["joint_position"].append(result.joint_position.cpu().numpy())
        rows["root_position_world"].append(
            result.root_position_world.cpu().numpy()
        )
        rows["root_orientation_world_wxyz"].append(
            result.root_orientation_world_wxyz.cpu().numpy()
        )
        rows["foot_position_world"].append(feet)
        rows["foot_surface_height_m"].append(surface)
        rows["foot_clearance_m"].append(clearance)
        rows["source_support_mask"].append(source.cpu().numpy())
        rows["emitted_support_mask"].append(emitted)
        rows["segment_committed"].append(diagnostic.segment_committed)
        rows["segment_start_frame"].append(
            -1 if diagnostic.segment_start_frame is None else diagnostic.segment_start_frame
        )
        rows["segment_end_frame"].append(
            -1 if diagnostic.segment_end_frame is None else diagnostic.segment_end_frame
        )
        rows["segment_entering_foot"].append(
            -1 if diagnostic.segment_entering_foot is None else diagnostic.segment_entering_foot
        )
        rows["segment_vertical_offset_m"].append(
            math.nan if diagnostic.segment_vertical_offset_m is None else diagnostic.segment_vertical_offset_m
        )
        rows["selected_clip_index"].append(clip_index)
        rows["selected_frame"].append(diagnostic.selected_frame)
        rows["step_time_ns"].append(diagnostic.step_time_ns)
        rows["search_time_ns"].append(
            -1 if diagnostic.search_time_ns is None else diagnostic.search_time_ns
        )
    arrays = {name: np.asarray(values) for name, values in rows.items()}
    vertical_speed = np.zeros_like(arrays["foot_clearance_m"])
    vertical_speed[1:] = (
        arrays["foot_position_world"][1:, :, 2]
        - arrays["foot_position_world"][:-1, :, 2]
    ) / float(resolved.resolved_config["dt"])
    arrays["emitted_support_mask"] = (
        (np.abs(arrays["foot_clearance_m"] - 0.035) <= 0.020)
        & (np.abs(vertical_speed) <= 0.12)
    )
    metrics = _rollout_metrics(arrays, resolved)
    gates = evaluate_contact_segment_acceptance(
        metrics, resolved.resolved_config["acceptance"]
    )
    return ContactSegmentRollout(
        arrays=MappingProxyType(arrays),
        metrics=MappingProxyType(metrics),
        gates=gates,
        resolved_config=resolved.resolved_config,
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_contact_segment_rollout(
    rollout: ContactSegmentRollout, output: str | Path
) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **rollout.arrays)
        _atomic_write(destination / "rollout.npz", temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)
    metrics = dict(rollout.metrics)
    metrics["accepted"] = rollout.accepted
    metrics["gates"] = [gate.__dict__ for gate in rollout.gates]
    _atomic_write(
        destination / "metrics.json",
        (json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    _atomic_write(
        destination / "resolved-config.json",
        (json.dumps(dict(rollout.resolved_config), sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    events = []
    committed = rollout.arrays["segment_committed"]
    starts = rollout.arrays["segment_start_frame"]
    for frame in range(len(committed)):
        if committed[frame] and (
            frame == 0
            or not committed[frame - 1]
            or starts[frame] != starts[frame - 1]
        ):
            events.append(
                json.dumps(
                    {
                        "event": "segment_entry",
                        "rollout_frame": frame,
                        "source_frame": int(
                            rollout.arrays["selected_frame"][frame]
                        ),
                        "segment_start_frame": int(starts[frame]),
                        "segment_end_frame": int(
                            rollout.arrays["segment_end_frame"][frame]
                        ),
                    },
                    sort_keys=True,
                )
            )
    _atomic_write(
        destination / "events.jsonl",
        (("\n".join(events) + "\n") if events else "").encode(),
    )
