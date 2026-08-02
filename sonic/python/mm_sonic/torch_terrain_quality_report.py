"""Deterministic reporting and orchestration for terrain motion quality."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import torch

from .torch_contact_oracle_actions import (
    ContactPhaseActionIndex,
    with_mirrored_actions,
)
from .torch_contact_oracle_search import load_contact_oracle_config
from .torch_contact_segments import ANKLE_ORIGIN_SOLE_M
from .torch_foothold_actions import plan_footholds
from .torch_terrain_action_quality import (
    NativeActionQuality,
    build_native_quality_index,
)
from .torch_terrain_quality_artifacts import analyze_saved_route
from .torch_terrain_quality_oracle import (
    QualityCandidateScore,
    QualityOracleResult,
    rank_quality_actions,
)
from .torch_terrain_quality_preview import QualityPreview, build_quality_preview
from .torch_terrain_quality_states import capture_quality_states


@dataclass(frozen=True)
class NativeCorpusSample:
    stance_drift_m: float
    boundary_motion: float

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in (self.stance_drift_m, self.boundary_motion)
        ):
            raise ValueError("native corpus sample is invalid")


@dataclass(frozen=True)
class NativeQualityThresholds:
    stance_drift_m: float
    boundary_motion: float
    sample_count: int

    def __post_init__(self) -> None:
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in (self.stance_drift_m, self.boundary_motion)
            )
            or type(self.sample_count) is not int
            or self.sample_count < 1
        ):
            raise ValueError("native quality thresholds are invalid")


def _median_plus_two_mad(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return median + 2.0 * mad


def derive_native_thresholds(
    samples: Sequence[NativeCorpusSample],
) -> NativeQualityThresholds:
    samples = tuple(samples)
    if not samples or any(not isinstance(value, NativeCorpusSample) for value in samples):
        raise ValueError("native threshold samples must be nonempty")
    return NativeQualityThresholds(
        stance_drift_m=_median_plus_two_mad(
            [value.stance_drift_m for value in samples]
        ),
        boundary_motion=_median_plus_two_mad(
            [value.boundary_motion for value in samples]
        ),
        sample_count=len(samples),
    )


@dataclass(frozen=True)
class StateClassificationEvidence:
    state_id: str
    source_acceptable: bool
    native_acceptable_count: int
    placed_acceptable_count: int
    composed_acceptable_count: int
    baseline_selected_acceptable: bool
    two_step_acceptable_count: int
    trigger_metrics: Mapping[str, float | int | bool]

    def __post_init__(self) -> None:
        counts = (
            self.native_acceptable_count,
            self.placed_acceptable_count,
            self.composed_acceptable_count,
            self.two_step_acceptable_count,
        )
        metrics = dict(self.trigger_metrics)
        if (
            not isinstance(self.state_id, str)
            or not self.state_id
            or type(self.source_acceptable) is not bool
            or type(self.baseline_selected_acceptable) is not bool
            or any(type(value) is not int or value < 0 for value in counts)
            or any(
                not isinstance(name, str)
                or not name
                or isinstance(value, str)
                or not isinstance(value, (int, float, bool))
                or (isinstance(value, float) and not math.isfinite(value))
                for name, value in metrics.items()
            )
        ):
            raise ValueError("terrain quality classification evidence is invalid")
        object.__setattr__(self, "trigger_metrics", MappingProxyType(metrics))


def classify_state(evidence: StateClassificationEvidence) -> str:
    """Return the earliest demonstrated failure boundary."""

    if not isinstance(evidence, StateClassificationEvidence):
        raise ValueError("terrain quality classification input is invalid")
    if not evidence.source_acceptable:
        return "source"
    if evidence.native_acceptable_count == 0:
        return "corpus"
    if evidence.placed_acceptable_count == 0:
        return "placement"
    if evidence.composed_acceptable_count == 0:
        return "composition"
    if not evidence.baseline_selected_acceptable:
        if evidence.two_step_acceptable_count == 0:
            return "representation"
        return "search"
    if evidence.two_step_acceptable_count == 0:
        return "representation"
    return "qualified"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _candidate_json(row: QualityCandidateScore) -> dict[str, object]:
    return {
        "action_index": row.action_index,
        "source_key": list(row.source_key),
        "landing_error_m": row.landing_error_m,
        "joint_position_boundary_error_rad": row.joint_position_boundary_error_rad,
        "joint_velocity_boundary_error_rad_s": row.joint_velocity_boundary_error_rad_s,
        "command_displacement_error_m": row.command_displacement_error_m,
        "command_facing_error_rad": row.command_facing_error_rad,
        "native_stance_drift_m": row.native_stance_drift_m,
        "clearance_margin_m": row.clearance_margin_m,
        "total_cost": row.total_cost,
        "exact_successor_index": row.exact_successor_index,
        "two_step_total_cost": row.two_step_total_cost,
    }


def _ranking_json(result: QualityOracleResult) -> dict[str, object]:
    return {
        "state_id": result.state_id,
        "evaluated_action_count": result.evaluated_action_count,
        "rejected_by_reason": dict(result.rejected_by_reason),
        "best_landing": [_candidate_json(row) for row in result.best_landing],
        "best_continuity": [_candidate_json(row) for row in result.best_continuity],
        "best_combined": [_candidate_json(row) for row in result.best_combined],
        "best_two_step": [_candidate_json(row) for row in result.best_two_step],
        "deterministic_sha256": result.deterministic_sha256,
    }


def _preview_metrics(preview: QualityPreview) -> dict[str, dict[str, float]]:
    output = {}
    for name in ("native", "placed", "composed"):
        stage = getattr(preview, name)
        output[name] = {
            "joint_boundary_jump_rad": stage.joint_boundary_jump_rad,
            "joint_velocity_boundary_jump_rad_s": (
                stage.joint_velocity_boundary_jump_rad_s
            ),
            "root_boundary_jump_m": stage.root_boundary_jump_m,
            "source_stance_drift_m": stage.source_stance_drift_m,
        }
    return output


def _load_routes(root: Path) -> tuple[tuple[str, Mapping[str, np.ndarray]], ...]:
    paths = sorted(root.glob("*/routes/*/arrays.npz"))
    if not paths:
        paths = sorted(root.glob("routes/*/arrays.npz"))
    if not paths:
        raise ValueError(f"baseline artifacts contain no route arrays: {root}")
    output = []
    names = set()
    for path in paths:
        route_name = path.parent.name
        if route_name in names:
            raise ValueError(f"duplicate baseline route: {route_name}")
        names.add(route_name)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        output.append((route_name, MappingProxyType(arrays)))
    return tuple(output)


def _native_thresholds(qualities: Sequence[NativeActionQuality]):
    return derive_native_thresholds(
        tuple(
            NativeCorpusSample(
                quality.source_stance_drift_m,
                max(quality.entry_joint_speed_norm, quality.terminal_joint_speed_norm),
            )
            for quality in qualities
        )
    )


def _render_contact_sheet(
    path: Path,
    state_id: str,
    previews: Mapping[int, QualityPreview],
) -> None:
    """Render deterministic root/foot contact panels from saved preview states."""

    if not previews:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    action_ids = sorted(previews)
    figure, axes = plt.subplots(
        len(action_ids), 3, figsize=(10.5, max(2.8, 2.5 * len(action_ids))), squeeze=False
    )
    sample_names = ("native", "placed", "composed")
    for row_index, action_index in enumerate(action_ids):
        preview = previews[action_index]
        for column, stage_name in enumerate(sample_names):
            axis = axes[row_index, column]
            stage = getattr(preview, stage_name)
            root = stage.qpos[:, :3]
            feet = stage.foot_position_world
            axis.plot(root[:, 0], root[:, 2], color="black", linewidth=1.2, label="root")
            for foot, color in ((0, "tab:blue"), (1, "tab:orange")):
                axis.plot(feet[:, foot, 0], feet[:, foot, 2], color=color, linewidth=1.0)
                supported = stage.source_support_mask[:, foot]
                axis.scatter(
                    feet[supported, foot, 0], feet[supported, foot, 2],
                    color=color, s=8,
                )
            axis.set_title(
                f"a{action_index} {stage_name}\n"
                f"drift={stage.source_stance_drift_m:.3f}m "
                f"jump={stage.joint_boundary_jump_rad:.2f}"
            )
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=0.2)
    figure.suptitle(f"terrain quality state {state_id[:12]}")
    figure.tight_layout()
    figure.savefig(path, dpi=120, metadata={"Software": "motion-quality-oracle"})
    plt.close(figure)


def run_quality_oracle(
    *,
    dataset: str,
    terrain_config: str,
    oracle_config: str,
    g1_xml: str,
    baseline_artifacts: str,
    output: str,
    device: str,
) -> dict[str, object]:
    """Run the complete frozen-state diagnostic and atomically publish evidence."""

    from .torch_contact_segment_rollout import (
        build_contact_segment_policy,
        load_contact_segment_config,
        resolve_contact_segment_config,
    )
    from .torch_g1_fk import MujocoG1FootKinematics
    from .torch_terrain_rollout import (
        load_experiment_config,
        resolve_stair_config,
    )

    output_path = Path(os.path.abspath(output))
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"terrain quality output already exists: {output_path}")
    baseline_root = Path(os.path.abspath(baseline_artifacts))
    baseline_resolved = resolve_stair_config(
        dataset, load_experiment_config(terrain_config), device=device
    )
    experiment = load_contact_oracle_config(oracle_config)
    contact_resolved = resolve_contact_segment_config(
        dataset,
        load_contact_segment_config(experiment.terrain_config),
        device=device,
    )
    foot_kinematics = MujocoG1FootKinematics(g1_xml)
    contact_policy = build_contact_segment_policy(contact_resolved, g1_xml)
    index = with_mirrored_actions(
        ContactPhaseActionIndex.from_dataset(
            contact_resolved.dataset, contact_policy.index, foot_kinematics
        )
    )
    qualities = build_native_quality_index(index)
    thresholds = _native_thresholds(qualities)
    measurement = baseline_resolved.measurement_extension

    def sample_surface(points: torch.Tensor) -> torch.Tensor:
        return measurement.query_grid.sample_xy(
            measurement.alignment.matcher_to_scene_xy(points)
        )

    def sample_surface_numpy(points: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(
            points, dtype=torch.float32, device=baseline_resolved.device
        )
        return sample_surface(query).detach().cpu().numpy()

    routes = _load_routes(baseline_root)
    route_records = []
    state_payloads = []
    preview_arrays: dict[str, Mapping[str, np.ndarray]] = {}
    for route_name, arrays in routes:
        route_quality = analyze_saved_route(
            baseline_resolved.dataset, route_name, arrays
        )
        states = capture_quality_states(
            route_name=route_name,
            arrays=arrays,
            source_support_mask=route_quality.source_support_mask,
            terrain_patch_sampler=sample_surface_numpy,
        )
        route_records.append(
            {
                "route_name": route_name,
                "state_count": len(states),
                "contact_agreement_fraction": (
                    route_quality.metrics.contact_agreement_fraction
                ),
                "complete_step_count": route_quality.metrics.complete_step_count,
                "route_sha256": route_quality.deterministic_sha256,
            }
        )
        for state in states:
            dtype = torch.float32
            state_feet = torch.as_tensor(
                np.array(state.foot_position_world, copy=True),
                dtype=dtype,
                device=baseline_resolved.device,
            )
            support = torch.as_tensor(
                np.array(state.source_support_mask, copy=True),
                dtype=torch.bool,
                device=baseline_resolved.device,
            )
            command = torch.as_tensor(
                np.array(state.command_velocity_world_xy, copy=True),
                dtype=dtype,
                device=baseline_resolved.device,
            )
            footholds = plan_footholds(
                foot_xy_m=state_feet[:, :2],
                support_mask=support,
                command_xy=command,
                sample_surface=sample_surface,
                reachable_forward_m=(0.20, 0.25, 0.30, 0.35, 0.40),
                lateral_samples_m=(-0.15, -0.075, 0.0, 0.075, 0.15),
                edge_margin_m=experiment.search.constraints.edge_margin_m,
                beam_width=experiment.search.foothold_beam_width,
            )
            if footholds.score.numel() == 0:
                evidence = StateClassificationEvidence(
                    state_id=state.state_id,
                    source_acceptable=True,
                    native_acceptable_count=0,
                    placed_acceptable_count=0,
                    composed_acceptable_count=0,
                    baseline_selected_acceptable=False,
                    two_step_acceptable_count=0,
                    trigger_metrics={"foothold_plan_count": 0},
                )
                state_payloads.append(
                    {
                        "state_id": state.state_id,
                        "route_name": route_name,
                        "route_frame": state.route_frame,
                        "reason": state.reason,
                        "classification": classify_state(evidence),
                        "evidence": dict(evidence.trigger_metrics),
                        "ranking": None,
                    }
                )
                continue
            landing_foot = int(footholds.landing_feet[0, 0].item())
            desired_xy = footholds.landing_xy_world_m[0, 0]
            desired_z = sample_surface(desired_xy[None])[0] + float(ANKLE_ORIGIN_SOLE_M)
            desired = torch.cat((desired_xy, desired_z.reshape(1))).detach().cpu().numpy()
            landing_frames = int(footholds.landing_frame_offsets[0, 0].item())
            command_target = (
                np.asarray(state.root_position_world[:2])
                + np.asarray(state.command_velocity_world_xy) * landing_frames * 0.02
            )
            ranking = rank_quality_actions(
                state=state,
                index=index,
                native_quality=qualities,
                desired_landing_world_xyz=desired,
                command_target_world_xy=command_target,
                sample_surface=sample_surface,
                constraints=experiment.search.constraints,
                top_k=5,
            )
            ranked_rows = (
                *ranking.best_landing,
                *ranking.best_continuity,
                *ranking.best_combined,
                *ranking.best_two_step,
            )
            selected_indices = sorted({row.action_index for row in ranked_rows})
            previews = {
                action_index: build_quality_preview(
                    state=state,
                    action=index.actions[action_index],
                    foot_kinematics=foot_kinematics,
                    inertialization_halflife_s=0.10,
                )
                for action_index in selected_indices
            }
            native_count = sum(
                tuple(bool(value) for value in quality.entry_support)
                == tuple(bool(value) for value in state.source_support_mask)
                and quality.source_stance_drift_m <= thresholds.stance_drift_m
                and max(
                    quality.entry_joint_speed_norm,
                    quality.terminal_joint_speed_norm,
                )
                <= thresholds.boundary_motion
                for quality in qualities
            )
            placed_count = ranking.evaluated_action_count - sum(
                ranking.rejected_by_reason.values()
            )
            composed_good = {
                action_index
                for action_index, preview in previews.items()
                if preview.composed.source_stance_drift_m
                <= max(0.01, thresholds.stance_drift_m)
                and preview.composed.joint_boundary_jump_rad
                <= experiment.search.constraints.maximum_joint_position_error_rad
                and preview.composed.joint_velocity_boundary_jump_rad_s
                <= experiment.search.constraints.maximum_joint_velocity_error_rad_s
            }
            clip_lookup = {
                clip_index: str(clip.relative_path)
                for clip_index, clip in enumerate(index.actions and contact_resolved.dataset.folder.clips)
            }
            baseline_good = any(
                action_index in composed_good
                and clip_lookup[index.actions[action_index].clip_index]
                == state.selected_clip_path
                and index.actions[action_index].start_frame
                <= state.selected_source_frame
                < index.actions[action_index].end_frame
                for action_index in selected_indices
            )
            two_step_count = sum(
                row.action_index in composed_good for row in ranking.best_two_step
            )
            evidence = StateClassificationEvidence(
                state_id=state.state_id,
                source_acceptable=(
                    route_quality.metrics.contact_agreement_fraction >= 0.5
                ),
                native_acceptable_count=native_count,
                placed_acceptable_count=placed_count,
                composed_acceptable_count=len(composed_good),
                baseline_selected_acceptable=baseline_good,
                two_step_acceptable_count=two_step_count,
                trigger_metrics={
                    "desired_landing_foot": landing_foot,
                    "native_stance_threshold_m": thresholds.stance_drift_m,
                    "native_boundary_threshold": thresholds.boundary_motion,
                    "planned_foothold_count": int(footholds.score.numel()),
                },
            )
            key_arrays = {}
            preview_summary = {}
            for action_index, preview in previews.items():
                preview_summary[str(action_index)] = _preview_metrics(preview)
                for stage_name in ("native", "placed", "composed"):
                    stage = getattr(preview, stage_name)
                    prefix = f"action_{action_index}_{stage_name}"
                    key_arrays[f"{prefix}_qpos"] = stage.qpos
                    key_arrays[f"{prefix}_feet"] = stage.foot_position_world
                    key_arrays[f"{prefix}_support"] = stage.source_support_mask
            preview_arrays[state.state_id] = MappingProxyType(key_arrays)
            state_payloads.append(
                {
                    "state_id": state.state_id,
                    "route_name": route_name,
                    "route_frame": state.route_frame,
                    "reason": state.reason,
                    "classification": classify_state(evidence),
                    "evidence": {
                        "source_acceptable": evidence.source_acceptable,
                        "native_acceptable_count": evidence.native_acceptable_count,
                        "placed_acceptable_count": evidence.placed_acceptable_count,
                        "composed_acceptable_count": evidence.composed_acceptable_count,
                        "baseline_selected_acceptable": evidence.baseline_selected_acceptable,
                        "two_step_acceptable_count": evidence.two_step_acceptable_count,
                        "trigger_metrics": dict(evidence.trigger_metrics),
                    },
                    "desired_landing_world_xyz": desired.tolist(),
                    "ranking": _ranking_json(ranking),
                    "preview_metrics": preview_summary,
                    "_previews": previews,
                }
            )

    identity_states = []
    for payload in state_payloads:
        identity_states.append({key: value for key, value in payload.items() if key != "_previews"})
    identity = {
        "schema": "g1-terrain-motion-quality-oracle/v1",
        "dataset_identity": baseline_resolved.dataset.manifest_sha256,
        "action_count": len(index.actions),
        "action_rejections": dict(index.inventory.rejected_by_reason),
        "native_thresholds": {
            "stance_drift_m": thresholds.stance_drift_m,
            "boundary_motion": thresholds.boundary_motion,
            "sample_count": thresholds.sample_count,
        },
        "routes": route_records,
        "states": identity_states,
    }
    digest = hashlib.sha256(_canonical(identity)).hexdigest()
    summary = {**identity, "deterministic_sha256": digest}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    try:
        states_dir = staging / "states"
        states_dir.mkdir()
        for payload in state_payloads:
            state_dir = states_dir / payload["state_id"]
            state_dir.mkdir()
            persisted = {key: value for key, value in payload.items() if key != "_previews"}
            (state_dir / "report.json").write_bytes(_canonical(persisted) + b"\n")
            arrays = preview_arrays.get(payload["state_id"])
            if arrays:
                np.savez_compressed(state_dir / "previews.npz", **arrays)
                _render_contact_sheet(
                    state_dir / "contact-sheet.png",
                    payload["state_id"],
                    payload["_previews"],
                )
        (staging / "summary.json").write_bytes(_canonical(summary) + b"\n")
        os.replace(staging, output_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "deterministic_sha256": digest,
        "state_count": len(state_payloads),
        "action_count": len(index.actions),
        "output": str(output_path),
    }
