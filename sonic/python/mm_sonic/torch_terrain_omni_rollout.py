"""Deterministic, renderer-independent same-stair rollout orchestration."""

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
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from .torch_terrain_omni_metrics import (
    OmniRouteMetrics,
    RescueEvent,
    evaluate_omni_route,
)
from .torch_terrain_omni_routes import OmniRoute, StairFrame, world_commands


DT_S = 0.02
_TIMING_ARRAYS = frozenset(("step_time_ns", "search_time_ns"))


@dataclass(frozen=True)
class KinematicSample:
    qpos: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    root_position_world: np.ndarray
    root_yaw_world: float
    foot_position_world: np.ndarray


@dataclass(frozen=True)
class RouteFailure:
    stage: str
    frame_index: int | None
    exception_type: str
    message: str


@dataclass(frozen=True)
class RouteOutcomeEvaluation:
    completed: bool
    segment_progress_ratio: tuple[tuple[str, float], ...]
    elevated_foot_sample_count: int
    final_heading_error_rad: float
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class OmniRouteRun:
    route: OmniRoute
    arrays: Mapping[str, np.ndarray]
    outcome: RouteOutcomeEvaluation
    metrics: OmniRouteMetrics | None
    completed_frames: int
    completed_without_exception: bool
    failure: RouteFailure | None
    deterministic_sha256: str


def evaluate_route_outcome(
    route: OmniRoute,
    arrays: Mapping[str, np.ndarray],
) -> RouteOutcomeEvaluation:
    """Evaluate the route's explicit progress, terrain, and terminal contract."""

    root = np.asarray(arrays["root_position_world"], dtype=np.float64)
    velocity = np.asarray(
        arrays["command_velocity_world_xy"], dtype=np.float64
    )
    segment_index = np.asarray(
        arrays["command_segment_index"], dtype=np.int64
    )
    surface = np.asarray(
        arrays["foot_surface_height_m"], dtype=np.float64
    )
    yaw = np.asarray(arrays["root_yaw_world"], dtype=np.float64)
    command_yaw = np.asarray(
        arrays["command_heading_world_yaw"], dtype=np.float64
    )
    frame_count = root.shape[0]
    if (
        root.shape != (frame_count, 3)
        or velocity.shape != (frame_count, 2)
        or segment_index.shape != (frame_count,)
        or surface.shape != (frame_count, 2)
        or yaw.shape != (frame_count,)
        or command_yaw.shape != (frame_count,)
        or not all(
            np.isfinite(value).all()
            for value in (root, velocity, surface, yaw, command_yaw)
        )
    ):
        raise ValueError("route outcome arrays are invalid")
    if frame_count == 0:
        return RouteOutcomeEvaluation(
            False, (), 0, math.inf, ("no-frames",)
        )

    reasons: list[str] = []
    ratios: list[tuple[str, float]] = []
    by_segment = {
        command.segment: index for index, command in enumerate(route.commands)
    }
    for segment in route.outcome.required_segments:
        index = by_segment[segment]
        frames = np.flatnonzero(segment_index == index)
        if frames.size < 2:
            ratio = 0.0
        else:
            command = np.mean(velocity[frames], axis=0)
            speed = float(np.linalg.norm(command))
            direction = command / speed if speed > 0.0 else np.zeros(2)
            progress = float(
                np.dot(
                    root[frames[-1], :2] - root[frames[0], :2],
                    direction,
                )
            )
            expected = speed * (frames.size - 1) * DT_S
            ratio = progress / expected if expected > 0.0 else 0.0
        ratios.append((segment, ratio))
        if ratio < route.outcome.min_segment_progress_ratio:
            reasons.append(f"segment:{segment}:progress")

    elevated_count = int(np.sum(surface > 0.05))
    if elevated_count < route.outcome.min_elevated_foot_samples:
        reasons.append("terrain:not-engaged")

    tail = slice(max(0, frame_count - 10), frame_count)
    if (
        route.outcome.final_surface == "flat"
        and bool(np.any(surface[tail] > 0.05))
    ):
        reasons.append("final-surface:not-flat")
    if (
        route.outcome.final_surface == "elevated"
        and not bool(np.any(surface[tail] > 0.05))
    ):
        reasons.append("final-surface:not-elevated")

    heading_error = np.abs(
        np.arctan2(
            np.sin(yaw[tail] - command_yaw[tail]),
            np.cos(yaw[tail] - command_yaw[tail]),
        )
    )
    final_heading_error = float(np.max(heading_error))
    tolerance = route.outcome.final_heading_error_max_rad
    if tolerance is not None and final_heading_error > tolerance:
        reasons.append("final-heading:error")

    return RouteOutcomeEvaluation(
        completed=not reasons,
        segment_progress_ratio=tuple(ratios),
        elevated_foot_sample_count=elevated_count,
        final_heading_error_rad=final_heading_error,
        failure_reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class OmniMatrix:
    runs: tuple[OmniRouteRun, ...]
    dataset_identity: str
    config_identity: str
    matrix_pass: bool
    deterministic_sha256: str


_ARRAY_SHAPES = {
    "command_velocity_world_xy": (2,),
    "command_heading_world_yaw": (),
    "command_segment_index": (),
    "qpos": (36,),
    "joint_position": (29,),
    "joint_velocity": (29,),
    "root_position_world": (3,),
    "root_yaw_world": (),
    "foot_position_world": (2, 3),
    "foot_surface_height_m": (2,),
    "selected_clip_path": (),
    "selected_source_frame": (),
    "terrain_rescue": (),
    "step_time_ns": (),
    "search_time_ns": (),
}

_ARRAY_DTYPES = {
    "command_velocity_world_xy": np.float64,
    "command_heading_world_yaw": np.float64,
    "command_segment_index": np.int32,
    "qpos": np.float64,
    "joint_position": np.float64,
    "joint_velocity": np.float64,
    "root_position_world": np.float64,
    "root_yaw_world": np.float64,
    "foot_position_world": np.float64,
    "foot_surface_height_m": np.float64,
    "selected_clip_path": np.dtype("<U256"),
    "selected_source_frame": np.int64,
    "terrain_rescue": np.bool_,
    "step_time_ns": np.int64,
    "search_time_ns": np.int64,
}


def _owned_readonly(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=_ARRAY_DTYPES[name])
    if array.shape != shape or (
        array.dtype.kind in "fc" and not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must have finite shape {shape}")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _finalize_rows(rows: dict[str, list]) -> Mapping[str, np.ndarray]:
    arrays = {}
    frame_count = len(rows["qpos"])
    for name, shape_tail in _ARRAY_SHAPES.items():
        if rows[name]:
            value = np.asarray(rows[name], dtype=_ARRAY_DTYPES[name])
        else:
            value = np.empty((0,) + shape_tail, dtype=_ARRAY_DTYPES[name])
        arrays[name] = _owned_readonly(
            value, (frame_count,) + shape_tail, name
        )
    return MappingProxyType(arrays)


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _hash_run(
    route: OmniRoute,
    arrays: Mapping[str, np.ndarray],
    failure: RouteFailure | None,
    dataset_identity: str,
    config_identity: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"g1-same-stair-omni-route/v2\0")
    digest.update(dataset_identity.encode("utf-8"))
    digest.update(b"\0")
    digest.update(config_identity.encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        _canonical_json(
            {
                "name": route.name,
                "required_outcome": route.required_outcome,
                "outcome_contract": {
                    "required_segments": route.outcome.required_segments,
                    "min_segment_progress_ratio": (
                        route.outcome.min_segment_progress_ratio
                    ),
                    "min_elevated_foot_samples": (
                        route.outcome.min_elevated_foot_samples
                    ),
                    "final_surface": route.outcome.final_surface,
                    "final_heading_error_max_rad": (
                        route.outcome.final_heading_error_max_rad
                    ),
                },
                "commands": [
                    {
                        "velocity_stair_xy": command.velocity_stair_xy,
                        "heading_stair_yaw": command.heading_stair_yaw,
                        "frames": command.frames,
                        "segment": command.segment,
                        "reset_before": command.reset_before,
                    }
                    for command in route.commands
                ],
                "failure": (
                    None
                    if failure is None
                    else {
                        "stage": failure.stage,
                        "frame_index": failure.frame_index,
                        "exception_type": failure.exception_type,
                        "message": failure.message,
                    }
                ),
            }
        )
    )
    for name in sorted(set(arrays) - _TIMING_ARRAYS):
        array = arrays[name]
        digest.update(b"\0")
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical_json(array.shape))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _failure(stage: str, frame_index: int | None, error: Exception) -> RouteFailure:
    return RouteFailure(stage, frame_index, type(error).__name__, str(error))


def _scalar_diagnostic(diagnostics, name: str, default):
    value = getattr(diagnostics, name, default)
    return default if value is None else value


def _run_route(
    route: OmniRoute,
    *,
    stair_frame: StairFrame,
    matcher_factory: Callable[[OmniRoute], object],
    kinematics: Callable[[object], KinematicSample],
    terrain_sampler: Callable[[np.ndarray], np.ndarray],
    dataset_identity: str,
    config_identity: str,
    maximum_step_time_ns: int | None,
) -> OmniRouteRun:
    rows = {name: [] for name in _ARRAY_SHAPES}
    failure = None
    stage = "setup"
    frame_index: int | None = None
    try:
        if (
            not route.commands
            or not route.commands[0].reset_before
            or any(command.reset_before for command in route.commands[1:])
        ):
            raise ValueError("route reset must be declared on its first command only")
        matcher = matcher_factory(route)
        stage = "reset"
        matcher.reset()
        expanded = world_commands(stair_frame, route)
        frame_index = 0
        for segment_index, command in enumerate(expanded):
            for _ in range(command.frames):
                stage = "prepare"
                prepared = matcher.prepare_step(
                    command.velocity_world_xy,
                    command.heading_world_yaw,
                    dt=DT_S,
                )
                stage = "commit"
                result = matcher.commit(prepared)
                stage = "kinematics"
                sample = kinematics(result)
                if not isinstance(sample, KinematicSample):
                    raise TypeError("kinematics must return KinematicSample")
                qpos = _owned_readonly(sample.qpos, (36,), "qpos")
                joint_position = _owned_readonly(
                    sample.joint_position, (29,), "joint_position"
                )
                joint_velocity = _owned_readonly(
                    sample.joint_velocity, (29,), "joint_velocity"
                )
                root = _owned_readonly(
                    sample.root_position_world, (3,), "root_position_world"
                )
                feet = _owned_readonly(
                    sample.foot_position_world, (2, 3), "foot_position_world"
                )
                root_yaw = float(sample.root_yaw_world)
                if not np.isfinite(root_yaw):
                    raise ValueError("root yaw must be finite")
                stage = "terrain"
                surface = _owned_readonly(
                    terrain_sampler(feet[:, :2]),
                    (2,),
                    "foot_surface_height_m",
                )
                diagnostics = result.diagnostics
                stage = "record"
                rows["command_velocity_world_xy"].append(
                    command.velocity_world_xy
                )
                rows["command_heading_world_yaw"].append(
                    command.heading_world_yaw
                )
                rows["command_segment_index"].append(segment_index)
                rows["qpos"].append(qpos)
                rows["joint_position"].append(joint_position)
                rows["joint_velocity"].append(joint_velocity)
                rows["root_position_world"].append(root)
                rows["root_yaw_world"].append(root_yaw)
                rows["foot_position_world"].append(feet)
                rows["foot_surface_height_m"].append(surface)
                rows["selected_clip_path"].append(
                    str(diagnostics.selected_clip_path)
                )
                rows["selected_source_frame"].append(
                    int(diagnostics.selected_frame)
                )
                rows["terrain_rescue"].append(
                    bool(
                        _scalar_diagnostic(
                            diagnostics, "terrain_safety_override", False
                        )
                    )
                )
                rows["step_time_ns"].append(
                    int(_scalar_diagnostic(diagnostics, "step_time_ns", -1))
                )
                rows["search_time_ns"].append(
                    int(_scalar_diagnostic(diagnostics, "search_time_ns", -1))
                )
                frame_index += 1
                step_time_ns = rows["step_time_ns"][-1]
                if (
                    maximum_step_time_ns is not None
                    and step_time_ns > maximum_step_time_ns
                ):
                    stage = "latency"
                    raise RuntimeError(
                        f"matcher step took {step_time_ns} ns, exceeding "
                        f"{maximum_step_time_ns} ns"
                    )
    except Exception as error:
        failure = _failure(stage, frame_index, error)

    arrays = _finalize_rows(rows)
    outcome = evaluate_route_outcome(route, arrays)
    metrics = None
    if arrays["qpos"].shape[0] > 0:
        rescue_events = tuple(
            RescueEvent(
                str(clip),
                int(source_frame),
                (
                    float(root[0]),
                    float(root[1]),
                ),
                bool(is_rescue),
            )
            for clip, source_frame, root, is_rescue in zip(
                arrays["selected_clip_path"],
                arrays["selected_source_frame"],
                arrays["root_position_world"],
                arrays["terrain_rescue"],
            )
            if is_rescue
        )
        metrics = evaluate_omni_route(
            root_xy=arrays["root_position_world"][:, :2],
            root_yaw=arrays["root_yaw_world"],
            foot_position_world=arrays["foot_position_world"],
            foot_surface_height_m=arrays["foot_surface_height_m"],
            command_velocity_world_xy=arrays[
                "command_velocity_world_xy"
            ],
            command_heading_world_yaw=arrays[
                "command_heading_world_yaw"
            ],
            selected_clip_id=arrays["selected_clip_path"],
            selected_source_frame=arrays["selected_source_frame"],
            rescue_events=rescue_events,
            required_outcome_completed=outcome.completed,
        )
    deterministic_hash = _hash_run(
        route, arrays, failure, dataset_identity, config_identity
    )
    return OmniRouteRun(
        route=route,
        arrays=arrays,
        outcome=outcome,
        metrics=metrics,
        completed_frames=int(arrays["qpos"].shape[0]),
        completed_without_exception=failure is None,
        failure=failure,
        deterministic_sha256=deterministic_hash,
    )


def run_omni_matrix(
    *,
    routes: Sequence[OmniRoute],
    stair_frame: StairFrame,
    matcher_factory: Callable[[OmniRoute], object],
    kinematics: Callable[[object], KinematicSample],
    terrain_sampler: Callable[[np.ndarray], np.ndarray],
    dataset_identity: str,
    config_identity: str,
    maximum_step_time_ns: int | None = None,
) -> OmniMatrix:
    """Run every route independently so one exception cannot abort the matrix."""

    if not dataset_identity or not config_identity:
        raise ValueError("dataset and config identities must be non-empty")
    if maximum_step_time_ns is not None and (
        type(maximum_step_time_ns) is not int or maximum_step_time_ns <= 0
    ):
        raise ValueError("maximum step time must be a positive integer")
    runs = tuple(
        _run_route(
            route,
            stair_frame=stair_frame,
            matcher_factory=matcher_factory,
            kinematics=kinematics,
            terrain_sampler=terrain_sampler,
            dataset_identity=dataset_identity,
            config_identity=config_identity,
            maximum_step_time_ns=maximum_step_time_ns,
        )
        for route in routes
    )
    digest = hashlib.sha256()
    digest.update(b"g1-same-stair-omni-matrix/v2")
    digest.update(b"\0")
    digest.update(dataset_identity.encode("utf-8"))
    digest.update(b"\0")
    digest.update(config_identity.encode("utf-8"))
    for run in runs:
        digest.update(b"\0")
        digest.update(run.route.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(run.deterministic_sha256.encode("ascii"))
    return OmniMatrix(
        runs=runs,
        dataset_identity=dataset_identity,
        config_identity=config_identity,
        matrix_pass=bool(runs) and all(
            run.completed_without_exception
            and run.outcome.completed
            for run in runs
        ),
        deterministic_sha256=digest.hexdigest(),
    )


def run_resolved_omni_matrix(
    resolved,
    *,
    g1_xml: str | Path,
    routes: Sequence[OmniRoute] | None = None,
    normalization_override=None,
    contact_segment_policy=None,
    foothold_action_policy=None,
    maximum_step_time_ns: int | None = None,
) -> OmniMatrix:
    """Run the real Torch matcher and measure native MuJoCo FK ankle origins."""

    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError("real omnidirectional rollout requires MuJoCo") from error

    from .torch_motion_matcher import TorchMotionMatcher
    from .torch_terrain_live_viewer import (
        apply_kinematic_state,
        build_kinematic_scene,
        matcher_result_qpos,
    )
    from .torch_terrain_omni_routes import same_stair_routes
    from .torch_terrain_rollout import (
        matcher_config_from_resolved,
        terrain_transition_validator_from_resolved,
    )

    config = resolved.resolved_config
    if float(config["dt"]) != DT_S:
        raise ValueError("omnidirectional rollout requires 50 Hz config")
    matcher = TorchMotionMatcher.from_folder(
        resolved.dataset.root,
        device=str(resolved.device),
        config=matcher_config_from_resolved(config),
        extension=resolved.measurement_extension,
        reset_clip_path=config["reset_clip"],
        emitted_window_validator=terrain_transition_validator_from_resolved(
            resolved
        ),
        normalization_override=normalization_override,
        contact_segment_policy=contact_segment_policy,
        foothold_action_policy=foothold_action_policy,
    )
    model, data = build_kinematic_scene(g1_xml, resolved)
    left_ankle = int(model.body("left_ankle_roll_link").id)
    right_ankle = int(model.body("right_ankle_roll_link").id)

    def kinematics(result) -> KinematicSample:
        qpos = matcher_result_qpos(result)
        apply_kinematic_state(mujoco, model, data, qpos)
        w, x, y, z = qpos[3:7]
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

        def cpu(value) -> np.ndarray:
            if isinstance(value, torch.Tensor):
                value = value.detach().to("cpu").numpy()
            return np.asarray(value, dtype=np.float64)

        return KinematicSample(
            qpos=qpos,
            joint_position=cpu(result.joint_position),
            joint_velocity=cpu(result.joint_velocity),
            root_position_world=cpu(result.root_position_world),
            root_yaw_world=yaw,
            foot_position_world=np.stack(
                (data.xpos[left_ankle], data.xpos[right_ankle])
            ).astype(np.float64, copy=True),
        )

    measurement = resolved.measurement_extension

    def terrain_sampler(matcher_xy: np.ndarray) -> np.ndarray:
        points = torch.tensor(
            matcher_xy,
            dtype=torch.float32,
            device=resolved.device,
        )
        scene_xy = measurement.alignment.matcher_to_scene_xy(points)
        return (
            measurement.query_grid.sample_xy(scene_xy)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
        )

    direction = np.asarray(
        config["reference_direction_matcher_xy"], dtype=np.float64
    )
    ascent_yaw = math.atan2(float(direction[1]), float(direction[0]))
    stair_frame = StairFrame(
        origin_world_xy=(0.0, 0.0),
        ascent_world_yaw=ascent_yaw,
        width_m=0.6223,
        tread_depth_m=0.3302,
        riser_height_m=0.1778,
        tread_count=3,
    )
    mean, scale = matcher.database.normalization.parameters_copy()
    digest = hashlib.sha256()
    digest.update(b"g1-motion-feature-normalization/v1")
    digest.update(mean.detach().to("cpu").numpy().tobytes())
    digest.update(scale.detach().to("cpu").numpy().tobytes())
    normalization_digest = digest.hexdigest()
    return run_omni_matrix(
        routes=tuple(same_stair_routes() if routes is None else routes),
        stair_frame=stair_frame,
        matcher_factory=lambda _route: matcher,
        kinematics=kinematics,
        terrain_sampler=terrain_sampler,
        dataset_identity=resolved.dataset.manifest_sha256,
        config_identity=(
            f"{resolved.base_config_sha256}:normalization:"
            f"{normalization_digest}"
        ),
        maximum_step_time_ns=maximum_step_time_ns,
    )


def fit_resolved_normalization(resolved):
    """Fit and return the exact feature normalization for one resolved corpus."""

    from .torch_motion_features import TorchMotionDatabase

    database = TorchMotionDatabase.from_folder(
        resolved.dataset.folder,
        device=resolved.device,
        extension=resolved.measurement_extension,
        reset_clip_path=resolved.resolved_config["reset_clip"],
    )
    return database.normalization


def _distribution_json(distribution) -> dict:
    return {
        "count": len(distribution.samples),
        "minimum": distribution.minimum,
        "maximum": distribution.maximum,
        "mean": distribution.mean,
        "p95": distribution.p95,
    }


def _metrics_json(metrics: OmniRouteMetrics | None) -> dict | None:
    if metrics is None:
        return None
    return {
        "stance_frame_count_per_foot": np.sum(
            metrics.stance_mask, axis=0
        ).astype(int).tolist(),
        "support_height_error_m": _distribution_json(
            metrics.support_height_error_m
        ),
        "support_height_difference_m": _distribution_json(
            metrics.support_height_difference_m
        ),
        "penetration_depth_m": _distribution_json(
            metrics.penetration_depth_m
        ),
        "stance_slide_m": {
            "per_foot": list(metrics.stance_slide_m.per_foot),
            "total": metrics.stance_slide_m.total,
        },
        "heading_error_rad": _distribution_json(metrics.heading_error_rad),
        "root_progress_m": metrics.root_progress_m,
        "root_jerk_m_s3": _distribution_json(metrics.root_jerk_m_s3),
        "root_velocity_error_mps": _distribution_json(
            metrics.root_velocity_error_mps
        ),
        "stalled_moving_frame_count": int(
            np.sum(metrics.stalled_moving_mask)
        ),
        "stalled_moving_fraction": metrics.stalled_moving_fraction,
        "longest_stall_frames": metrics.longest_stall_frames,
        "transition_count": metrics.transition_count,
        "rescue_cycle_count": len(metrics.rescue_cycles),
        "rescue_cycles": [
            {
                "event_indices": list(cycle.event_indices),
                "clip_pair": list(cycle.clip_pair),
                "root_progress_m": cycle.root_progress_m,
            }
            for cycle in metrics.rescue_cycles
        ],
        "required_outcome_completed": metrics.required_outcome_completed,
    }


def _failure_json(failure: RouteFailure | None) -> dict | None:
    if failure is None:
        return None
    return {
        "stage": failure.stage,
        "frame_index": failure.frame_index,
        "exception_type": failure.exception_type,
        "message": failure.message,
    }


def _outcome_json(outcome: RouteOutcomeEvaluation) -> dict:
    return {
        "completed": outcome.completed,
        "segment_progress_ratio": [
            {"segment": segment, "ratio": ratio}
            for segment, ratio in outcome.segment_progress_ratio
        ],
        "elevated_foot_sample_count": outcome.elevated_foot_sample_count,
        "final_heading_error_rad": outcome.final_heading_error_rad,
        "failure_reasons": list(outcome.failure_reasons),
    }


def save_omni_matrix(matrix: OmniMatrix, output: str | Path) -> None:
    """Publish one matrix atomically as pickle-free NPZ and canonical JSON."""

    if not isinstance(matrix, OmniMatrix):
        raise TypeError("matrix must be an OmniMatrix")
    output = Path(os.path.abspath(os.fspath(output)))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"omnidirectional output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        routes_dir = staging / "routes"
        routes_dir.mkdir()
        matrix_routes = []
        for run in matrix.runs:
            route_dir = routes_dir / run.route.name
            route_dir.mkdir()
            np.savez_compressed(route_dir / "arrays.npz", **run.arrays)
            metrics_payload = {
                "schema": "g1-same-stair-omni-route-metrics/v2",
                "route": run.route.name,
                "required_outcome": run.route.required_outcome,
                "outcome": _outcome_json(run.outcome),
                "completed_frames": run.completed_frames,
                "completed_without_exception": (
                    run.completed_without_exception
                ),
                "failure": _failure_json(run.failure),
                "deterministic_sha256": run.deterministic_sha256,
                "metrics": _metrics_json(run.metrics),
            }
            (route_dir / "metrics.json").write_bytes(
                _canonical_json(metrics_payload) + b"\n"
            )
            matrix_routes.append(
                {
                    "name": run.route.name,
                    "required_outcome": run.route.required_outcome,
                    "completed_frames": run.completed_frames,
                    "completed_without_exception": (
                        run.completed_without_exception
                    ),
                    "failure": _failure_json(run.failure),
                    "deterministic_sha256": run.deterministic_sha256,
                }
            )
        matrix_payload = {
            "schema": "g1-same-stair-omni-matrix/v2",
            "dataset_identity": matrix.dataset_identity,
            "config_identity": matrix.config_identity,
            "matrix_pass": matrix.matrix_pass,
            "deterministic_sha256": matrix.deterministic_sha256,
            "routes": matrix_routes,
        }
        (staging / "matrix.json").write_bytes(
            _canonical_json(matrix_payload) + b"\n"
        )
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
