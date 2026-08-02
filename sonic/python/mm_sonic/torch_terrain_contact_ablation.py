"""Offline orchestration for contact-space terrain composition ablations."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import torch

from .torch_contact_oracle_search import OracleConstraints
from .torch_terrain_action_quality import NativeActionQuality
from .torch_terrain_contact_quality import (
    ContactQualityAblationResult,
    build_contact_quality_ablation,
)
from .torch_terrain_quality_oracle import QualityActionCache
from .torch_terrain_quality_states import FrozenQualityState


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def selected_ranked_action_indices(
    state_payload: Mapping[str, object], *, maximum_count: int
) -> tuple[int, ...]:
    """Select a deterministic diverse prefix of saved exhaustive rankings."""

    if (
        not isinstance(state_payload, Mapping)
        or type(maximum_count) is not int
        or maximum_count < 1
    ):
        raise ValueError("contact ablation selection inputs are invalid")
    ranking = state_payload.get("ranking")
    if not isinstance(ranking, Mapping):
        return ()
    output: list[int] = []
    seen: set[int] = set()
    for name in (
        "best_combined",
        "best_landing",
        "best_continuity",
        "best_two_step",
    ):
        rows = ranking.get(name)
        if not isinstance(rows, list):
            raise ValueError("contact ablation saved ranking is invalid")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("contact ablation saved ranking row is invalid")
            action_index = row.get("action_index")
            if type(action_index) is not int or action_index < 0:
                raise ValueError("contact ablation action identity is invalid")
            if action_index not in seen:
                seen.add(action_index)
                output.append(action_index)
                if len(output) == maximum_count:
                    return tuple(output)
    return tuple(output)


def contact_ablation_passes(metrics: Mapping[str, object]) -> bool:
    """Apply the frozen correction acceptance thresholds."""

    if not isinstance(metrics, Mapping):
        raise ValueError("contact ablation metrics are invalid")
    names = (
        "placed_stance_drift_m",
        "projected_stance_drift_m",
        "projected_landing_error_m",
        "maximum_target_error_m",
        "maximum_root_correction_m",
    )
    try:
        values = {name: float(metrics[name]) for name in names}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("contact ablation metrics are incomplete") from error
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("contact ablation metrics are invalid")
    return bool(
        values["projected_stance_drift_m"] <= 0.03
        and values["projected_stance_drift_m"]
        <= values["placed_stance_drift_m"] + 0.01
        and values["projected_landing_error_m"] <= 0.12
        and values["maximum_target_error_m"] <= 0.005
        and values["maximum_root_correction_m"] <= 0.10
    )


def _rotate_batch_xy(values: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    while cosine.ndim < values.ndim - 1:
        cosine = cosine.unsqueeze(-1)
        sine = sine.unsqueeze(-1)
    return torch.stack(
        (
            cosine * values[..., 0] - sine * values[..., 1],
            sine * values[..., 0] + cosine * values[..., 1],
        ),
        dim=-1,
    )


def contact_anchored_prefilter(
    *,
    state: FrozenQualityState,
    cache: QualityActionCache,
    native_quality: tuple[NativeActionQuality, ...],
    desired_landing_foot: int,
    desired_landing_world_xyz: object,
    command_target_world_xy: object,
    constraints: OracleConstraints,
    maximum_count: int,
) -> tuple[int, ...]:
    """Rank every action after its entry contacts, rather than root, are fitted."""

    count = len(cache.source_keys)
    if (
        not isinstance(state, FrozenQualityState)
        or not isinstance(cache, QualityActionCache)
        or not isinstance(native_quality, tuple)
        or len(native_quality) != count
        or any(
            not isinstance(quality, NativeActionQuality)
            or quality.action_index != index
            for index, quality in enumerate(native_quality)
        )
        or type(desired_landing_foot) is not int
        or desired_landing_foot not in (0, 1)
        or not isinstance(constraints, OracleConstraints)
        or type(maximum_count) is not int
        or maximum_count < 1
    ):
        raise ValueError("contact anchored prefilter inputs are invalid")
    if count == 0:
        return ()
    desired_numpy = np.asarray(desired_landing_world_xyz, dtype=np.float64)
    command_numpy = np.asarray(command_target_world_xy, dtype=np.float64)
    if (
        desired_numpy.shape != (3,)
        or command_numpy.shape != (2,)
        or not np.isfinite(desired_numpy).all()
        or not np.isfinite(command_numpy).all()
    ):
        raise ValueError("contact anchored prefilter targets are invalid")

    reference = cache.entry_foot_local
    dtype, device = reference.dtype, reference.device
    state_feet = torch.as_tensor(
        np.array(state.foot_position_world, copy=True), dtype=dtype, device=device
    )
    state_root = torch.as_tensor(
        np.array(state.root_position_world, copy=True), dtype=dtype, device=device
    )
    desired = torch.as_tensor(desired_numpy, dtype=dtype, device=device)
    command = torch.as_tensor(command_numpy, dtype=dtype, device=device)
    state_support = torch.as_tensor(
        np.array(state.source_support_mask, copy=True),
        dtype=torch.bool,
        device=device,
    )
    support_indices = torch.nonzero(state_support, as_tuple=False).flatten()
    if support_indices.numel() not in (1, 2):
        return ()

    valid_geometry = torch.ones((count,), dtype=torch.bool, device=device)
    if support_indices.numel() == 1:
        yaw = torch.full(
            (count,), float(state.root_yaw_world), dtype=dtype, device=device
        )
    else:
        source_delta = reference[:, 1, :2] - reference[:, 0, :2]
        target_delta = state_feet[1, :2] - state_feet[0, :2]
        source_length = torch.linalg.vector_norm(source_delta, dim=1)
        valid_geometry &= source_length > 1e-6
        target_length = torch.linalg.vector_norm(target_delta)
        if float(target_length.item()) <= 1e-6:
            return ()
        source_angle = torch.atan2(source_delta[:, 1], source_delta[:, 0])
        target_angle = torch.atan2(target_delta[1], target_delta[0])
        yaw = torch.atan2(
            torch.sin(target_angle - source_angle),
            torch.cos(target_angle - source_angle),
        )

    rotated_entry_xy = _rotate_batch_xy(reference[..., :2], yaw)
    supported_entry_xy = rotated_entry_xy[:, support_indices]
    translation_xy = torch.mean(
        state_feet[support_indices, :2][None] - supported_entry_xy, dim=1
    )
    translation_z = torch.mean(
        state_feet[support_indices, 2][None]
        - reference[:, support_indices, 2],
        dim=1,
    )
    placed_entry = torch.cat(
        (
            rotated_entry_xy + translation_xy[:, None],
            reference[..., 2:] + translation_z[:, None, None],
        ),
        dim=2,
    )
    entry_error = torch.linalg.vector_norm(
        placed_entry[:, support_indices] - state_feet[support_indices][None], dim=2
    ).amax(dim=1)
    landing_xy = (
        _rotate_batch_xy(cache.terminal_foot_local[:, :2], yaw) + translation_xy
    )
    landing_z = cache.terminal_foot_local[:, 2] + translation_z
    landing = torch.cat((landing_xy, landing_z[:, None]), dim=1)
    terminal_root = (
        _rotate_batch_xy(cache.terminal_root_local_xy, yaw) + translation_xy
    )
    terminal_yaw = cache.terminal_yaw_local + yaw
    facing_error = torch.abs(
        torch.atan2(
            torch.sin(terminal_yaw - float(state.command_heading_world_yaw)),
            torch.cos(terminal_yaw - float(state.command_heading_world_yaw)),
        )
    )
    landing_error = torch.linalg.vector_norm(landing - desired[None], dim=1)
    command_error = torch.linalg.vector_norm(terminal_root - command[None], dim=1)
    placed_root_entry = torch.cat((translation_xy, translation_z[:, None]), dim=1)
    root_correction = torch.linalg.vector_norm(
        placed_root_entry - state_root[None], dim=1
    )
    joint_position_error = torch.linalg.vector_norm(
        cache.entry_joint_position
        - torch.as_tensor(
            np.array(state.joint_position, copy=True), dtype=dtype, device=device
        )[None],
        dim=1,
    )
    joint_velocity_error = torch.linalg.vector_norm(
        cache.entry_joint_velocity
        - torch.as_tensor(
            np.array(state.joint_velocity, copy=True), dtype=dtype, device=device
        )[None],
        dim=1,
    )
    native_stance = torch.tensor(
        [quality.source_stance_drift_m for quality in native_quality],
        dtype=dtype,
        device=device,
    )
    support_match = (cache.entry_support == state_support[None]).all(dim=1)
    feasible = (
        valid_geometry
        & support_match
        & (cache.swing_foot == desired_landing_foot)
        & (entry_error <= constraints.maximum_entry_foot_error_m)
        & (joint_position_error <= constraints.maximum_joint_position_error_rad)
        & (joint_velocity_error <= constraints.maximum_joint_velocity_error_rad_s)
        & (landing_error <= 0.50)
        & (command_error <= 1.0)
        & (facing_error <= np.pi)
        & (root_correction <= 0.30)
        & (native_stance <= 0.10)
    )
    cost = (
        10.0 * (landing_error / 0.12).square()
        + 3.0 * (command_error / 0.25).square()
        + 2.0 * (facing_error / 0.60).square()
        + (joint_position_error / constraints.maximum_joint_position_error_rad).square()
        + (joint_velocity_error / constraints.maximum_joint_velocity_error_rad_s).square()
        + 4.0 * (entry_error / constraints.maximum_entry_foot_error_m).square()
        + 2.0 * (root_correction / 0.15).square()
        + (native_stance / 0.03).square()
    )
    cost = torch.where(feasible, cost, torch.full_like(cost, torch.inf))
    order = torch.argsort(cost, stable=True)
    accepted = order[torch.isfinite(cost[order])][:maximum_count]
    return tuple(int(value) for value in accepted.detach().cpu().tolist())


def _metrics(result: ContactQualityAblationResult) -> dict[str, float]:
    return {
        name: float(getattr(result, name))
        for name in (
            "placement_entry_error_m",
            "placed_stance_drift_m",
            "unprojected_stance_drift_m",
            "projected_stance_drift_m",
            "projected_landing_error_m",
            "maximum_target_error_m",
            "maximum_root_correction_m",
            "maximum_joint_deformation_rad",
            "rms_joint_deformation_rad",
        )
    }


def _load_authenticated_summary(root: Path) -> dict[str, object]:
    try:
        summary = json.loads((root / "summary.json").read_text())
    except Exception as error:
        raise ValueError("cannot load terrain quality oracle summary") from error
    if (
        not isinstance(summary, dict)
        or summary.get("schema") != "g1-terrain-motion-quality-oracle/v1"
        or not isinstance(summary.get("states"), list)
        or not isinstance(summary.get("deterministic_sha256"), str)
    ):
        raise ValueError("terrain quality oracle summary is invalid")
    identity = {
        key: value for key, value in summary.items() if key != "deterministic_sha256"
    }
    if hashlib.sha256(_canonical(identity)).hexdigest() != summary["deterministic_sha256"]:
        raise ValueError("terrain quality oracle summary authentication failed")
    return summary


def _render_contact_sheet(
    path: Path,
    *,
    state: object,
    action_index: int,
    baseline_qpos: np.ndarray | None,
    baseline_feet: np.ndarray | None,
    result: ContactQualityAblationResult,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = []
    if baseline_qpos is not None and baseline_feet is not None:
        stages.append(("root composed", baseline_qpos, baseline_feet))
    stages.extend(
        (
            (
                "contact placed",
                result.unprojected_qpos,
                result.unprojected_foot_position_world,
            ),
            (
                "contact projected",
                result.projected_qpos,
                result.projected_foot_position_world,
            ),
        )
    )
    terrain = np.asarray(state.terrain_patch_world_xyh)
    figure, axes = plt.subplots(
        2, len(stages), figsize=(4.2 * len(stages), 7.0), squeeze=False
    )
    support = result.source_support_mask
    for column, (name, qpos, feet) in enumerate(stages):
        root = np.asarray(qpos)[:, :3]
        plan = axes[0, column]
        plan.scatter(
            terrain[:, 0], terrain[:, 1], c=terrain[:, 2], cmap="terrain", s=7
        )
        plan.plot(root[:, 0], root[:, 1], color="black", linewidth=1.2)
        for foot, color in ((0, "tab:blue"), (1, "tab:orange")):
            plan.plot(feet[:, foot, 0], feet[:, foot, 1], color=color)
            plan.scatter(
                feet[support[:, foot], foot, 0],
                feet[support[:, foot], foot, 1],
                color=color,
                s=10,
            )
        plan.set_title(name)
        plan.set_aspect("equal", adjustable="box")
        plan.grid(alpha=0.2)

        height = axes[1, column]
        time = np.arange(root.shape[0]) / 50.0
        height.plot(time, root[:, 2], color="black", label="root")
        for foot, color in ((0, "tab:blue"), (1, "tab:orange")):
            height.plot(time, feet[:, foot, 2], color=color)
            height.scatter(
                time[support[:, foot]],
                feet[support[:, foot], foot, 2],
                color=color,
                s=10,
            )
        height.set_xlabel("time (s)")
        height.set_ylabel("world height (m)")
        height.grid(alpha=0.2)
    figure.suptitle(
        f"{state.reason} frame {state.route_frame} action {action_index}\n"
        f"projected drift={result.projected_stance_drift_m:.4f}m "
        f"landing={result.projected_landing_error_m:.4f}m "
        f"joint warp={result.maximum_joint_deformation_rad:.3f}rad"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=130, metadata={"Software": "contact-ablation"})
    plt.close(figure)


def run_contact_ablation(
    *,
    dataset: str,
    terrain_config: str,
    oracle_config: str,
    g1_xml: str,
    baseline_artifacts: str,
    quality_oracle_artifacts: str,
    output: str,
    device: str,
    maximum_actions_per_state: int,
    projection_strategy: str,
    selection_strategy: str,
) -> dict[str, object]:
    """Evaluate contact composition on the authenticated frozen oracle states."""

    from .torch_contact_oracle_actions import (
        ContactPhaseActionIndex,
        with_mirrored_actions,
    )
    from .torch_contact_oracle_search import load_contact_oracle_config
    from .torch_contact_oracle_search import validate_placement
    from .torch_contact_segment_rollout import (
        build_contact_segment_policy,
        load_contact_segment_config,
        resolve_contact_segment_config,
    )
    from .torch_g1_fk import MujocoG1FootKinematics
    from .torch_foothold_actions import plan_footholds
    from .torch_terrain_action_quality import build_native_quality_index
    from .torch_terrain_contact_composition import place_action_contact_anchored
    from .torch_terrain_quality_artifacts import analyze_saved_route
    from .torch_terrain_quality_report import _load_routes
    from .torch_terrain_quality_oracle import build_quality_action_cache
    from .torch_terrain_quality_preview import quality_state_as_oracle
    from .torch_terrain_quality_states import capture_quality_states
    from .torch_terrain_rollout import load_experiment_config, resolve_stair_config

    if type(maximum_actions_per_state) is not int or maximum_actions_per_state < 1:
        raise ValueError("maximum actions per state must be positive")
    if projection_strategy not in ("joint-only", "stance-root"):
        raise ValueError("contact ablation projection strategy is invalid")
    if selection_strategy not in ("saved", "contact-anchored"):
        raise ValueError("contact ablation selection strategy is invalid")
    output_path = Path(os.path.abspath(output))
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"contact ablation output already exists: {output_path}")
    baseline_root = Path(os.path.abspath(baseline_artifacts))
    oracle_root = Path(os.path.abspath(quality_oracle_artifacts))
    oracle_summary = _load_authenticated_summary(oracle_root)

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
    if len(index.actions) != oracle_summary.get("action_count"):
        raise ValueError("contact ablation action inventory does not match oracle")
    native_quality = build_native_quality_index(index)
    quality_cache = build_quality_action_cache(index)

    measurement = baseline_resolved.measurement_extension

    def sample_surface(query: torch.Tensor) -> torch.Tensor:
        return measurement.query_grid.sample_xy(
            measurement.alignment.matcher_to_scene_xy(query)
        )

    def sample_surface_numpy(points: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(
            points, dtype=torch.float32, device=baseline_resolved.device
        )
        return sample_surface(query).detach().cpu().numpy()

    captured = {}
    for route_name, arrays in _load_routes(baseline_root):
        route_quality = analyze_saved_route(
            baseline_resolved.dataset, route_name, arrays
        )
        for state in capture_quality_states(
            route_name=route_name,
            arrays=arrays,
            source_support_mask=route_quality.source_support_mask,
            terrain_patch_sampler=sample_surface_numpy,
        ):
            captured[state.state_id] = state
    payloads = oracle_summary["states"]
    if not all(
        isinstance(payload, dict) and payload.get("state_id") in captured
        for payload in payloads
    ):
        raise ValueError("contact ablation frozen states do not match oracle")

    state_records = []
    state_arrays: dict[str, dict[str, np.ndarray]] = {}
    state_results: dict[str, dict[int, ContactQualityAblationResult]] = {}
    failure_counts: Counter[str] = Counter()
    for payload in payloads:
        state_id = payload["state_id"]
        state = captured[state_id]
        desired = payload.get("desired_landing_world_xyz")
        trigger = (
            payload.get("evidence", {}).get("trigger_metrics", {})
            if isinstance(payload.get("evidence"), dict)
            else {}
        )
        desired_foot = trigger.get("desired_landing_foot")
        if selection_strategy == "saved":
            selected = selected_ranked_action_indices(
                payload, maximum_count=maximum_actions_per_state
            )
        elif desired is None or type(desired_foot) is not int:
            selected = ()
        else:
            state_feet = torch.as_tensor(
                np.array(state.foot_position_world, copy=True),
                dtype=torch.float32,
                device=baseline_resolved.device,
            )
            state_support = torch.as_tensor(
                np.array(state.source_support_mask, copy=True),
                dtype=torch.bool,
                device=baseline_resolved.device,
            )
            state_command = torch.as_tensor(
                np.array(state.command_velocity_world_xy, copy=True),
                dtype=torch.float32,
                device=baseline_resolved.device,
            )
            footholds = plan_footholds(
                foot_xy_m=state_feet[:, :2],
                support_mask=state_support,
                command_xy=state_command,
                sample_surface=sample_surface,
                reachable_forward_m=(0.20, 0.25, 0.30, 0.35, 0.40),
                lateral_samples_m=(-0.15, -0.075, 0.0, 0.075, 0.15),
                edge_margin_m=experiment.search.constraints.edge_margin_m,
                beam_width=experiment.search.foothold_beam_width,
            )
            if footholds.score.numel() == 0:
                selected = ()
            else:
                landing_frames = int(
                    footholds.landing_frame_offsets[0, 0].item()
                )
                command_target = (
                    np.asarray(state.root_position_world[:2])
                    + np.asarray(state.command_velocity_world_xy)
                    * landing_frames
                    * 0.02
                )
                prefiltered = contact_anchored_prefilter(
                    state=state,
                    cache=quality_cache,
                    native_quality=native_quality,
                    desired_landing_foot=desired_foot,
                    desired_landing_world_xyz=desired,
                    command_target_world_xy=command_target,
                    constraints=experiment.search.constraints,
                    maximum_count=max(256, maximum_actions_per_state),
                )
                current = quality_state_as_oracle(
                    state, index.actions[0].joint_position
                )
                terrain_accepted = []
                for action_index in prefiltered:
                    placed = place_action_contact_anchored(
                        index.actions[action_index], current
                    ).placed
                    feasibility = validate_placement(
                        placed=placed,
                        state=current,
                        sample_surface=sample_surface,
                        constraints=experiment.search.constraints,
                    )
                    if feasibility.accepted:
                        terrain_accepted.append(action_index)
                        if len(terrain_accepted) == maximum_actions_per_state:
                            break
                selected = tuple(terrain_accepted)
        action_records = []
        arrays: dict[str, np.ndarray] = {}
        successful: dict[int, ContactQualityAblationResult] = {}
        for action_index in selected:
            if not 0 <= action_index < len(index.actions):
                raise ValueError("contact ablation ranked action is out of range")
            try:
                result = build_contact_quality_ablation(
                    state=state,
                    action=index.actions[action_index],
                    desired_landing_foot=desired_foot,
                    desired_landing_world_xyz=desired,
                    foot_kinematics=foot_kinematics,
                    projection_strategy=projection_strategy,
                )
            except ValueError as error:
                reason = str(error)
                failure_counts[reason] += 1
                action_records.append(
                    {
                        "action_index": action_index,
                        "success": False,
                        "failure_reason": reason,
                    }
                )
                continue
            metrics = _metrics(result)
            accepted = contact_ablation_passes(metrics)
            successful[action_index] = result
            action_records.append(
                {
                    "action_index": action_index,
                    "success": True,
                    "accepted": accepted,
                    "metrics": metrics,
                }
            )
            prefix = f"action_{action_index}"
            arrays[f"{prefix}_contact_qpos"] = result.unprojected_qpos
            arrays[f"{prefix}_projected_qpos"] = result.projected_qpos
            arrays[f"{prefix}_contact_feet"] = (
                result.unprojected_foot_position_world
            )
            arrays[f"{prefix}_projected_feet"] = (
                result.projected_foot_position_world
            )
            arrays[f"{prefix}_support"] = result.source_support_mask
        accepted_actions = [
            row["action_index"]
            for row in action_records
            if row.get("success") and row.get("accepted")
        ]
        best_action = None
        if successful:
            best_action = min(
                successful,
                key=lambda action_index: (
                    successful[action_index].projected_stance_drift_m,
                    successful[action_index].projected_landing_error_m,
                    successful[action_index].maximum_joint_deformation_rad,
                    action_index,
                ),
            )
        state_records.append(
            {
                "state_id": state_id,
                "route_name": state.route_name,
                "route_frame": state.route_frame,
                "reason": state.reason,
                "baseline_classification": payload.get("classification"),
                "selected_action_count": len(selected),
                "successful_projection_count": len(successful),
                "accepted_action_indices": accepted_actions,
                "best_action_index": best_action,
                "actions": action_records,
            }
        )
        if arrays:
            state_arrays[state_id] = arrays
            state_results[state_id] = successful

    identity = {
        "schema": "g1-terrain-contact-composition-ablation/v1",
        "source_oracle_sha256": oracle_summary["deterministic_sha256"],
        "dataset_identity": oracle_summary["dataset_identity"],
        "action_count": len(index.actions),
        "maximum_actions_per_state": maximum_actions_per_state,
        "projection_strategy": projection_strategy,
        "selection_strategy": selection_strategy,
        "state_count": len(state_records),
        "states_with_accepted_action": sum(
            bool(record["accepted_action_indices"]) for record in state_records
        ),
        "projection_failure_counts": dict(sorted(failure_counts.items())),
        "states": state_records,
    }
    digest = hashlib.sha256(_canonical(identity)).hexdigest()
    summary = {**identity, "deterministic_sha256": digest}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    )
    try:
        states_root = staging / "states"
        states_root.mkdir()
        record_by_id = {record["state_id"]: record for record in state_records}
        for state_id, arrays in state_arrays.items():
            state_dir = states_root / state_id
            state_dir.mkdir()
            (state_dir / "report.json").write_bytes(
                _canonical(record_by_id[state_id]) + b"\n"
            )
            np.savez_compressed(state_dir / "contact-previews.npz", **arrays)
            best_action = record_by_id[state_id]["best_action_index"]
            if best_action is not None:
                baseline_qpos = baseline_feet = None
                source_preview = oracle_root / "states" / state_id / "previews.npz"
                if source_preview.exists():
                    with np.load(source_preview, allow_pickle=False) as archive:
                        qpos_key = f"action_{best_action}_composed_qpos"
                        feet_key = f"action_{best_action}_composed_feet"
                        if qpos_key in archive and feet_key in archive:
                            baseline_qpos = np.array(archive[qpos_key], copy=True)
                            baseline_feet = np.array(archive[feet_key], copy=True)
                _render_contact_sheet(
                    state_dir / "contact-sheet.png",
                    state=captured[state_id],
                    action_index=best_action,
                    baseline_qpos=baseline_qpos,
                    baseline_feet=baseline_feet,
                    result=state_results[state_id][best_action],
                )
        (staging / "summary.json").write_bytes(_canonical(summary) + b"\n")
        os.replace(staging, output_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "deterministic_sha256": digest,
        "state_count": len(state_records),
        "states_with_accepted_action": identity["states_with_accepted_action"],
        "output": str(output_path),
    }
