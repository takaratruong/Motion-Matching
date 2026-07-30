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
_CHALLENGE_NAMES = (
    "rapid-reversal",
    "lateral-switch",
    "independent-octants",
    "stair-stop-restart",
    "upper-landing-side-exit",
    "seeded-random",
)
_CHALLENGE_SEED = 20260730
_CHALLENGE_DT = 0.02


@dataclass(frozen=True)
class DirectionalRollout:
    arrays: Mapping[str, np.ndarray]
    metrics: Mapping
    events: Sequence[Mapping]
    resolved_config: Mapping
    resolved_config_sha256: str
    deterministic_sha256: str


@dataclass(frozen=True)
class ChallengeCommand:
    velocity_world_xy: tuple[float, float]
    heading_world_yaw: float
    reset_before: bool = False

    def __post_init__(self) -> None:
        velocity = self.velocity_world_xy
        if (
            not isinstance(velocity, tuple)
            or len(velocity) != 2
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in velocity
            )
        ):
            raise ContractError(
                "challenge velocity must be a finite two-float tuple"
            )
        heading = self.heading_world_yaw
        if (
            not isinstance(heading, (int, float))
            or isinstance(heading, bool)
            or not math.isfinite(float(heading))
        ):
            raise ContractError("challenge heading must be finite")
        if type(self.reset_before) is not bool:
            raise ContractError("challenge reset flag must be bool")
        object.__setattr__(
            self,
            "velocity_world_xy",
            (float(velocity[0]), float(velocity[1])),
        )
        object.__setattr__(self, "heading_world_yaw", float(heading))


@dataclass(frozen=True)
class ChallengeScenario:
    name: str
    commands: tuple[ChallengeCommand, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ContractError("challenge scenario name must be non-empty")
        if (
            not isinstance(self.commands, tuple)
            or not self.commands
            or any(
                not isinstance(command, ChallengeCommand)
                for command in self.commands
            )
        ):
            raise ContractError(
                "challenge scenario commands must be a non-empty tuple"
            )


@dataclass(frozen=True)
class ChallengeScenarioRun:
    scenario: ChallengeScenario
    arrays: Mapping[str, np.ndarray]
    metrics: Mapping
    events: Sequence[Mapping]
    command_sha256: str
    deterministic_sha256: str


@dataclass(frozen=True)
class ChallengeMatrix:
    runs: tuple[ChallengeScenarioRun, ...]
    metrics: Mapping
    resolved_config: Mapping
    resolved_config_sha256: str
    deterministic_sha256: str


def _wrap_yaw(value: float) -> float:
    wrapped = math.atan2(math.sin(value), math.cos(value))
    return -math.pi if math.isclose(wrapped, math.pi) else wrapped


def _octant_vectors(
    speed: float,
    forward_heading_world_yaw: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            speed
            * math.cos(
                forward_heading_world_yaw + index * math.pi / 4.0
            ),
            speed
            * math.sin(
                forward_heading_world_yaw + index * math.pi / 4.0
            ),
        )
        for index in range(8)
    )


def _repeat_command(
    velocity: tuple[float, float],
    heading: float,
    count: int,
    *,
    reset_first: bool = False,
) -> tuple[ChallengeCommand, ...]:
    return tuple(
        ChallengeCommand(
            velocity,
            heading,
            reset_before=reset_first and index == 0,
        )
        for index in range(count)
    )


def _seeded_random_commands(
    *,
    seed: int,
    speed: float,
    forward_heading_world_yaw: float,
    max_episode_frames: int,
) -> tuple[ChallengeCommand, ...]:
    rng = np.random.default_rng(seed)
    velocities = ((0.0, 0.0),) + _octant_vectors(
        speed, forward_heading_world_yaw
    )
    headings = tuple(
        _wrap_yaw(
            forward_heading_world_yaw + index * math.pi / 4.0
        )
        for index in range(-4, 4)
    )
    commands: list[ChallengeCommand] = []
    for _episode in range(8):
        episode_length = 0
        previous_identity = None
        first_segment = True
        while max_episode_frames - episode_length >= 5:
            remaining = max_episode_frames - episode_length
            segment_length = int(rng.integers(5, min(40, remaining) + 1))
            while True:
                velocity = velocities[int(rng.integers(len(velocities)))]
                heading = headings[int(rng.integers(len(headings)))]
                identity = (velocity, heading)
                if identity != previous_identity:
                    break
            commands.extend(
                ChallengeCommand(
                    velocity,
                    heading,
                    reset_before=first_segment and offset == 0,
                )
                for offset in range(segment_length)
            )
            episode_length += segment_length
            previous_identity = identity
            first_segment = False
    return tuple(commands)


def challenge_scenarios(
    *,
    seed: int = _CHALLENGE_SEED,
    speed: float = 1.0,
    forward_heading_world_yaw: float = 0.0,
    max_episode_frames: int = 400,
) -> tuple[ChallengeScenario, ...]:
    """Return the frozen independent velocity/heading stress scenarios."""

    if type(seed) is not int:
        raise ContractError("challenge seed must be an integer")
    if (
        not isinstance(speed, (int, float))
        or isinstance(speed, bool)
        or not math.isfinite(float(speed))
    ):
        raise ContractError("challenge speed must be finite")
    if (
        not isinstance(forward_heading_world_yaw, (int, float))
        or isinstance(forward_heading_world_yaw, bool)
        or not math.isfinite(float(forward_heading_world_yaw))
    ):
        raise ContractError("challenge forward heading must be finite")
    if type(max_episode_frames) is not int or max_episode_frames <= 0:
        raise ContractError(
            "challenge episode frame bound must be a positive integer"
        )
    if max_episode_frames < 5:
        raise ContractError(
            "challenge episode frame bound must be at least five"
        )
    speed = float(speed)
    forward_heading_world_yaw = _wrap_yaw(
        float(forward_heading_world_yaw)
    )
    if forward_heading_world_yaw == 0.0:
        forward = (speed, 0.0)
        backward = (-speed, 0.0)
        right = (0.0, speed)
        left = (0.0, -speed)
    else:
        forward = (
            speed * math.cos(forward_heading_world_yaw),
            speed * math.sin(forward_heading_world_yaw),
        )
        backward = (-forward[0], -forward[1])
        right = (
            speed
            * math.cos(forward_heading_world_yaw + math.pi / 2.0),
            speed
            * math.sin(forward_heading_world_yaw + math.pi / 2.0),
        )
        left = (-right[0], -right[1])

    rapid = tuple(
        ChallengeCommand(
            forward if (index // 10) % 2 == 0 else backward,
            forward_heading_world_yaw,
            reset_before=index == 0,
        )
        for index in range(200)
    )
    lateral = tuple(
        ChallengeCommand(
            right if (index // 15) % 2 == 0 else left,
            forward_heading_world_yaw,
            reset_before=index == 0,
        )
        for index in range(240)
    )
    octants: list[ChallengeCommand] = []
    directions = _octant_vectors(speed, forward_heading_world_yaw)
    headings = tuple(
        _wrap_yaw(
            forward_heading_world_yaw + index * math.pi / 4.0
        )
        for index in range(8)
    )
    for index, velocity in enumerate(directions):
        octants.extend(
            _repeat_command(
                velocity,
                headings[(index + 2) % 8],
                30,
                reset_first=index == 0,
            )
        )
    stop_restart = (
        _repeat_command(
            forward,
            forward_heading_world_yaw,
            80,
            reset_first=True,
        )
        + _repeat_command(
            (0.0, 0.0), forward_heading_world_yaw, 20
        )
        + _repeat_command(forward, forward_heading_world_yaw, 100)
        + _repeat_command(
            (0.0, 0.0), forward_heading_world_yaw, 20
        )
        + _repeat_command(forward, forward_heading_world_yaw, 100)
        + _repeat_command(
            (0.0, 0.0), forward_heading_world_yaw, 20
        )
    )
    side_exit = (
        _repeat_command(
            forward,
            forward_heading_world_yaw,
            ASCENT_STOP,
            reset_first=True,
        )
        + _repeat_command(right, forward_heading_world_yaw, 120)
        + _repeat_command(
            forward,
            forward_heading_world_yaw,
            ASCENT_STOP,
            reset_first=True,
        )
        + _repeat_command(left, forward_heading_world_yaw, 120)
    )
    random_commands = _seeded_random_commands(
        seed=seed,
        speed=speed,
        forward_heading_world_yaw=forward_heading_world_yaw,
        max_episode_frames=max_episode_frames,
    )
    scenarios = (
        ChallengeScenario(_CHALLENGE_NAMES[0], rapid),
        ChallengeScenario(_CHALLENGE_NAMES[1], lateral),
        ChallengeScenario(_CHALLENGE_NAMES[2], tuple(octants)),
        ChallengeScenario(_CHALLENGE_NAMES[3], stop_restart),
        ChallengeScenario(_CHALLENGE_NAMES[4], side_exit),
        ChallengeScenario(_CHALLENGE_NAMES[5], random_commands),
    )
    names = tuple(scenario.name for scenario in scenarios)
    if names != _CHALLENGE_NAMES or len(set(names)) != len(names):
        raise ContractError("challenge scenario names must be unique and frozen")
    return scenarios


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


def _challenge_command_sha256(scenario: ChallengeScenario) -> str:
    return _json_sha256(
        {
            "name": scenario.name,
            "commands": [
                {
                    "velocity_world_xy": list(command.velocity_world_xy),
                    "heading_world_yaw": command.heading_world_yaw,
                    "reset_before": command.reset_before,
                }
                for command in scenario.commands
            ],
        }
    )


def _split_reset_episodes(reset_before: np.ndarray) -> tuple[slice, ...]:
    reset = np.asarray(reset_before)
    if reset.ndim != 1 or reset.dtype.kind != "b":
        raise ContractError("challenge reset_before must be one-dimensional bool")
    starts = [0]
    starts.extend(
        int(index)
        for index in np.flatnonzero(reset)
        if int(index) != 0
    )
    starts = sorted(set(starts))
    return tuple(
        slice(start, stop)
        for start, stop in zip(starts, starts[1:] + [len(reset)])
        if stop > start
    )


def _episode_differences(
    values: np.ndarray,
    episodes: Sequence[slice],
    *,
    order: int,
    dt: float,
    axis: int | tuple[int, ...],
) -> np.ndarray:
    samples = []
    for episode in episodes:
        episode_values = np.asarray(values[episode], np.float64)
        if len(episode_values) <= order:
            continue
        difference = np.diff(episode_values, n=order, axis=0) / (
            dt**order
        )
        samples.append(np.linalg.norm(difference, axis=axis).reshape(-1))
    if not samples:
        return np.empty(0, np.float64)
    return np.concatenate(samples)


def _compute_challenge_metrics(
    scenario: ChallengeScenario,
    arrays: Mapping[str, np.ndarray],
    *,
    dt: float,
    position_weight: float,
    velocity_weight: float,
    exception: Mapping | None,
) -> dict:
    frame_count = len(np.asarray(arrays.get("joint_position", ())))
    reset_before = np.asarray(
        arrays.get("reset_before", np.zeros(frame_count, np.bool_))
    )
    if len(reset_before) != frame_count:
        raise ContractError("challenge reset evidence frame count differs")
    episodes = _split_reset_episodes(reset_before)
    joint_position = np.asarray(
        arrays.get("joint_position", np.empty((0, 0))), np.float64
    )
    root_position = np.asarray(
        arrays.get("root_position_world", np.empty((0, 3))), np.float64
    )
    foot_position = np.asarray(
        arrays.get("foot_position_world", np.empty((0, 2, 3))),
        np.float64,
    )
    clearance = np.asarray(
        arrays.get("foot_clearance_m", np.empty((0, 2))), np.float64
    )
    joint_jerk = _episode_differences(
        joint_position,
        episodes,
        order=3,
        dt=dt,
        axis=1,
    )
    joint_acceleration = _episode_differences(
        joint_position,
        episodes,
        order=2,
        dt=dt,
        axis=1,
    )
    root_jerk = _episode_differences(
        root_position,
        episodes,
        order=3,
        dt=dt,
        axis=1,
    )
    foot_speed = _episode_differences(
        foot_position[:, :, :2],
        episodes,
        order=1,
        dt=dt,
        axis=2,
    )
    transitioned = np.asarray(
        arrays.get("transitioned", np.zeros(frame_count, np.bool_))
    )
    rejected = np.asarray(
        arrays.get("transition_rejected", np.zeros(frame_count, np.bool_))
    )
    rescue = np.asarray(
        arrays.get("terrain_safety_override", np.zeros(frame_count, np.bool_))
    )
    selected_clip = np.asarray(
        arrays.get("selected_clip_index", np.zeros(frame_count, np.int32))
    )
    previous_clip = np.asarray(
        arrays.get(
            "previous_selected_clip_index",
            np.zeros(frame_count, np.int32),
        )
    )
    rescue_rank = np.asarray(
        arrays.get(
            "terrain_safety_override_rank",
            np.zeros(frame_count, np.int32),
        )
    )
    position_cost = np.asarray(
        arrays.get(
            "selected_transition_position_cost",
            np.zeros(frame_count, np.float64),
        )
    )
    velocity_cost = np.asarray(
        arrays.get(
            "selected_transition_velocity_cost",
            np.zeros(frame_count, np.float64),
        )
    )
    continuity_cost = np.asarray(
        arrays.get(
            "selected_transition_continuity_cost",
            np.zeros(frame_count, np.float64),
        )
    )
    for name, value in (
        ("transitioned", transitioned),
        ("transition_rejected", rejected),
        ("terrain_safety_override", rescue),
        ("selected_clip_index", selected_clip),
        ("previous_selected_clip_index", previous_clip),
        ("terrain_safety_override_rank", rescue_rank),
        ("selected_transition_position_cost", position_cost),
        ("selected_transition_velocity_cost", velocity_cost),
        ("selected_transition_continuity_cost", continuity_cost),
    ):
        if value.ndim != 1 or len(value) != frame_count:
            raise ContractError(f"challenge array frame count differs: {name}")
    transition_intervals = []
    for episode in episodes:
        frames = np.flatnonzero(transitioned[episode])
        if len(frames) > 1:
            transition_intervals.append(np.diff(frames))
    intervals = (
        np.concatenate(transition_intervals)
        if transition_intervals
        else np.empty(0, np.int64)
    )
    jerk_distribution = _distribution(joint_jerk)
    minimum_clearance = (
        float(clearance.min()) if clearance.size else None
    )
    command_sha = _challenge_command_sha256(scenario)
    return {
        "scenario_name": scenario.name,
        "command_sha256": command_sha,
        "command_count": len(scenario.commands),
        "frame_count": frame_count,
        "episode_count": len(episodes),
        "dt_s": float(dt),
        "position_weight": position_weight,
        "velocity_weight": velocity_weight,
        "completed_without_exception": (
            exception is None and frame_count == len(scenario.commands)
        ),
        "exception": deepcopy(dict(exception)) if exception is not None else None,
        "minimum_clearance_m": minimum_clearance,
        "p95_joint_jerk_rad_s3": jerk_distribution["p95"],
        "aggregate": {
            "joint_acceleration_rad_s2": _distribution(
                joint_acceleration
            ),
            "joint_jerk_rad_s3": jerk_distribution,
            "root_jerk_m_s3": _distribution(root_jerk),
            "foot_speed_m_s": _distribution(foot_speed),
            "minimum_foot_clearance_m": minimum_clearance,
            "accepted_transition_count": int(transitioned.sum()),
            "rejected_transition_count": int(rejected.sum()),
            "cross_clip_transition_count": int(
                (transitioned & (selected_clip != previous_clip)).sum()
            ),
            "terrain_safety_override_count": int(rescue.sum()),
            "terrain_safety_override_rank": _distribution(
                rescue_rank[rescue]
            ),
            "accepted_transition_interval_frames": _distribution(
                intervals
            ),
            "accepted_transition_interval_s": _distribution(
                intervals.astype(np.float64) * dt
            ),
            "transition_position_cost": _distribution(
                position_cost[transitioned]
            ),
            "transition_velocity_cost": _distribution(
                velocity_cost[transitioned]
            ),
            "transition_continuity_cost": _distribution(
                continuity_cost[transitioned]
            ),
        },
    }


def _validated_challenge_metric_map(
    metrics: Mapping[str, Mapping],
    *,
    label: str,
) -> dict[str, dict]:
    if not isinstance(metrics, Mapping):
        raise ContractError(f"{label} challenge metrics must be a mapping")
    if set(metrics) != set(_CHALLENGE_NAMES):
        raise ContractError(
            f"{label} challenge metrics must contain all six scenarios"
        )
    validated = {}
    for name in _CHALLENGE_NAMES:
        value = metrics[name]
        if not isinstance(value, Mapping):
            raise ContractError(f"{label} scenario metrics must be mappings")
        metric = deepcopy(dict(value))
        if metric.get("scenario_name") != name:
            raise ContractError(f"{label} scenario name differs: {name}")
        command_sha = metric.get("command_sha256")
        if (
            not isinstance(command_sha, str)
            or len(command_sha) != 64
            or any(character not in "0123456789abcdef" for character in command_sha)
        ):
            raise ContractError(
                f"{label} command identity is invalid: {name}"
            )
        if type(metric.get("completed_without_exception")) is not bool:
            raise ContractError(
                f"{label} completion evidence is invalid: {name}"
            )
        clearance = metric.get("minimum_clearance_m")
        if (
            clearance is not None
            and (
                not isinstance(clearance, (int, float))
                or isinstance(clearance, bool)
                or not math.isfinite(float(clearance))
            )
        ):
            raise ContractError(
                f"{label} clearance evidence is invalid: {name}"
            )
        jerk = metric.get("p95_joint_jerk_rad_s3")
        if (
            jerk is not None
            and (
                not isinstance(jerk, (int, float))
                or isinstance(jerk, bool)
                or not math.isfinite(float(jerk))
                or float(jerk) < 0.0
            )
        ) or (
            jerk is None and metric["completed_without_exception"]
        ):
            raise ContractError(f"{label} jerk evidence is invalid: {name}")
        metric["minimum_clearance_m"] = (
            None if clearance is None else float(clearance)
        )
        metric["p95_joint_jerk_rad_s3"] = (
            None if jerk is None else float(jerk)
        )
        validated[name] = metric
    return validated


def evaluate_challenge_matrix(
    baseline_metrics: Mapping[str, Mapping],
    retained_metrics: Mapping[str, Mapping],
) -> dict:
    """Apply the frozen six-scenario baseline-relative acceptance rules."""

    baseline = _validated_challenge_metric_map(
        baseline_metrics, label="baseline"
    )
    retained = _validated_challenge_metric_map(
        retained_metrics, label="retained"
    )
    scenario_verdicts = {}
    for name in _CHALLENGE_NAMES:
        before = baseline[name]
        after = retained[name]
        if before["command_sha256"] != after["command_sha256"]:
            raise ContractError(
                f"challenge command identity differs from baseline: {name}"
            )
        baseline_jerk = before["p95_joint_jerk_rad_s3"]
        retained_jerk = after["p95_joint_jerk_rad_s3"]
        if baseline_jerk is None or retained_jerk is None:
            ratio = None
            nonregression = False
            material_improvement = False
        elif baseline_jerk == 0.0:
            ratio = None
            nonregression = retained_jerk == 0.0
            material_improvement = False
        else:
            ratio = retained_jerk / baseline_jerk
            nonregression = ratio <= 1.10
            material_improvement = ratio <= 0.85
        scenario_verdicts[name] = {
            "command_sha256": after["command_sha256"],
            "baseline_p95_joint_jerk_rad_s3": baseline_jerk,
            "retained_p95_joint_jerk_rad_s3": retained_jerk,
            "jerk_ratio": ratio,
            "completed_without_exception": after[
                "completed_without_exception"
            ],
            "minimum_clearance_m": after["minimum_clearance_m"],
            "scenario_nonregression": nonregression,
            "material_improvement": material_improvement,
            "exception": deepcopy(after.get("exception")),
        }
    material_count = sum(
        value["material_improvement"]
        for value in scenario_verdicts.values()
    )
    gates = {
        "completed_without_exception": all(
            value["completed_without_exception"]
            for value in scenario_verdicts.values()
        ),
        "minimum_clearance": all(
            value["minimum_clearance_m"] is not None
            and value["minimum_clearance_m"] >= -0.03
            for value in scenario_verdicts.values()
        ),
        "scenario_nonregression": all(
            value["scenario_nonregression"]
            for value in scenario_verdicts.values()
        ),
        "material_improvement_count": material_count >= 4,
    }
    return {
        "baseline": False,
        "matrix_pass": all(gates.values()),
        "material_improvement_count": material_count,
        "gates": gates,
        "scenarios": scenario_verdicts,
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


def _challenge_run_hash(
    resolved_config_sha256: str,
    command_sha256: str,
    arrays: Mapping[str, np.ndarray],
    exception: Mapping | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"independent-command-challenge/v1")
    digest.update(b"\x00")
    digest.update(resolved_config_sha256.encode("ascii"))
    digest.update(b"\x00")
    digest.update(command_sha256.encode("ascii"))
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
    if exception is not None:
        digest.update(b"\x00exception\x00")
        digest.update(_canonical_json_bytes(dict(exception)))
    return digest.hexdigest()


def _challenge_rows() -> dict[str, list]:
    return {
        "time_s": [],
        "command_velocity_world_xy": [],
        "command_heading_world_yaw": [],
        "reset_before": [],
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


def _finalize_challenge_run(
    *,
    scenario: ChallengeScenario,
    rows: Mapping[str, list],
    events: Sequence[Mapping],
    resolved_config_sha256: str,
    dt: float,
    position_weight: float,
    velocity_weight: float,
    exception: Mapping | None,
    metadata: Mapping | None = None,
) -> ChallengeScenarioRun:
    empty_specs = {
        "command_velocity_world_xy": ((0, 2), np.float32),
        "reset_before": ((0,), np.bool_),
        "searched": ((0,), np.bool_),
        "transitioned": ((0,), np.bool_),
        "transition_rejected": ((0,), np.bool_),
        "terrain_safety_override": ((0,), np.bool_),
        "joint_position": ((0, 0), np.float32),
        "joint_velocity": ((0, 0), np.float32),
        "root_position_world": ((0, 3), np.float32),
        "root_orientation_world_wxyz": ((0, 4), np.float32),
        "foot_position_world": ((0, 2, 3), np.float32),
        "foot_velocity_world": ((0, 2, 3), np.float32),
        "foot_clearance_m": ((0, 2), np.float32),
    }
    arrays = {
        name: _owned_readonly(
            np.asarray(values)
            if values
            else np.empty(
                *empty_specs.get(name, ((0,), np.float32))
            )
        )
        for name, values in rows.items()
    }
    command_sha = _challenge_command_sha256(scenario)
    metrics = _compute_challenge_metrics(
        scenario,
        arrays,
        dt=dt,
        position_weight=position_weight,
        velocity_weight=velocity_weight,
        exception=exception,
    )
    if metadata:
        metrics.update(deepcopy(dict(metadata)))
    deterministic_hash = _challenge_run_hash(
        resolved_config_sha256,
        command_sha,
        arrays,
        exception,
    )
    metrics["resolved_config_sha256"] = resolved_config_sha256
    metrics["deterministic_sha256"] = deterministic_hash
    return ChallengeScenarioRun(
        scenario=scenario,
        arrays=MappingProxyType(arrays),
        metrics=MappingProxyType(metrics),
        events=tuple(MappingProxyType(dict(event)) for event in events),
        command_sha256=command_sha,
        deterministic_sha256=deterministic_hash,
    )


class _ChallengeScenarioFailure(Exception):
    def __init__(
        self,
        cause: Exception,
        command_index: int,
        partial_run: ChallengeScenarioRun,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.command_index = command_index
        self.partial_run = partial_run


def _run_challenge_scenario(
    resolved: ResolvedStairConfig,
    *,
    resolved_device: torch.device,
    challenge_config: Mapping,
    resolved_config_sha256: str,
    scenario: ChallengeScenario,
    position_weight: float,
    velocity_weight: float,
) -> ChallengeScenarioRun:
    dt = float(challenge_config["dt"])
    matcher = TorchMotionMatcher.from_folder(
        resolved.dataset.root,
        device=str(resolved_device),
        config=matcher_config_from_resolved(challenge_config),
        extension=resolved.measurement_extension,
        reset_clip_path=challenge_config["reset_clip"],
        emitted_window_validator=terrain_transition_validator_from_resolved(
            resolved
        ),
    )
    clip_index = {
        clip.relative_path: index
        for index, clip in enumerate(resolved.dataset.folder.clips)
    }
    rows = _challenge_rows()
    events: list[dict] = []
    measurement = resolved.measurement_extension
    previous_clip_index = None
    metadata = {
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
    for command_index, command in enumerate(scenario.commands):
        try:
            if command.reset_before or previous_clip_index is None:
                reset = matcher.reset()
                previous_clip_index = clip_index[
                    reset.diagnostics.selected_clip_path
                ]
            prepared = matcher.prepare_step(
                command.velocity_world_xy,
                command.heading_world_yaw,
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

            rows["time_s"].append(np.float32(command_index * dt))
            rows["command_velocity_world_xy"].append(
                np.asarray(command.velocity_world_xy, np.float32)
            )
            rows["command_heading_world_yaw"].append(
                np.float32(command.heading_world_yaw)
            )
            rows["reset_before"].append(np.bool_(command.reset_before))
            rows["selected_frame"].append(
                np.int64(diagnostics.selected_frame)
            )
            rows["selected_clip_index"].append(
                np.int32(clip_index[diagnostics.selected_clip_path])
            )
            rows["previous_selected_clip_index"].append(
                np.int32(previous_clip_index)
            )
            rows["searched"].append(np.bool_(diagnostics.searched))
            rows["transitioned"].append(
                np.bool_(diagnostics.transitioned)
            )
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
                np.float32(
                    diagnostics.selected_transition_position_cost
                )
            )
            rows["selected_transition_velocity_cost"].append(
                np.float32(
                    diagnostics.selected_transition_velocity_cost
                )
            )
            rows["selected_transition_continuity_cost"].append(
                np.float32(
                    diagnostics.selected_transition_continuity_cost
                )
            )
            rows["step_time_ns"].append(
                np.int64(diagnostics.step_time_ns)
            )
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
                result.root_orientation_world_wxyz.detach()
                .to("cpu")
                .numpy()
            )
            rows["foot_position_world"].append(foot)
            rows["foot_velocity_world"].append(body_velocity[1:])
            rows["foot_clearance_m"].append(
                clearance.astype(np.float32)
            )
            if (
                diagnostics.searched
                or diagnostics.transitioned
                or diagnostics.transition_rejected
                or diagnostics.terrain_safety_override
            ):
                events.append(
                    {
                        "sequence": int(diagnostics.sequence),
                        "command_index": command_index,
                        "time_s": float(command_index * dt),
                        "scenario": scenario.name,
                        "selected_clip_path": (
                            diagnostics.selected_clip_path
                        ),
                        "selected_frame": int(
                            diagnostics.selected_frame
                        ),
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
        except Exception as error:
            exception = {
                "type": type(error).__name__,
                "message": str(error),
                "command_index": command_index,
            }
            partial = _finalize_challenge_run(
                scenario=scenario,
                rows=rows,
                events=events,
                resolved_config_sha256=resolved_config_sha256,
                dt=dt,
                position_weight=position_weight,
                velocity_weight=velocity_weight,
                exception=exception,
                metadata=metadata,
            )
            raise _ChallengeScenarioFailure(
                error, command_index, partial
            ) from error
    return _finalize_challenge_run(
        scenario=scenario,
        rows=rows,
        events=events,
        resolved_config_sha256=resolved_config_sha256,
        dt=dt,
        position_weight=position_weight,
        velocity_weight=velocity_weight,
        exception=None,
        metadata=metadata,
    )


def run_challenge_matrix(
    resolved: ResolvedStairConfig,
    *,
    device: str | torch.device,
    position_weight: float,
    velocity_weight: float,
) -> ChallengeMatrix:
    """Execute all six challenge scenarios, isolating scenario failures."""

    position_weight = _validate_weight(position_weight, "position weight")
    velocity_weight = _validate_weight(velocity_weight, "velocity weight")
    resolved_device = resolve_torch_device(device)
    if resolved_device != resolved.device:
        raise ContractError("rollout device does not match resolved config")
    challenge_config = deepcopy(dict(resolved.resolved_config))
    if float(challenge_config.get("dt", _CHALLENGE_DT)) != _CHALLENGE_DT:
        raise ContractError("challenge matrix requires dt=0.02")
    challenge_config["dt"] = _CHALLENGE_DT
    challenge_config.setdefault("matcher", {})
    challenge_config["matcher"]["transition_joint_position_weight"] = (
        position_weight
    )
    challenge_config["matcher"]["transition_joint_velocity_weight"] = (
        velocity_weight
    )
    challenge_config.update(
        {
            "active_condition": "dense",
            "active_encoder": "dense",
            "challenge_matrix": True,
            "challenge_scenario_names": list(_CHALLENGE_NAMES),
            "challenge_seed": _CHALLENGE_SEED,
        }
    )
    resolved_hash = _json_sha256(challenge_config)
    direction = np.asarray(
        challenge_config.get("reference_direction_matcher_xy"),
        np.float64,
    )
    if (
        direction.shape != (2,)
        or not np.isfinite(direction).all()
        or float(np.linalg.norm(direction)) <= 0.0
    ):
        raise ContractError(
            "challenge reference direction must be a finite nonzero 2-vector"
        )
    speed = challenge_config.get("command_speed_mps")
    forward_heading = math.atan2(
        float(direction[1]), float(direction[0])
    )
    runs = []
    for scenario in challenge_scenarios(
        speed=speed,
        forward_heading_world_yaw=forward_heading,
    ):
        try:
            run = _run_challenge_scenario(
                resolved,
                resolved_device=resolved_device,
                challenge_config=challenge_config,
                resolved_config_sha256=resolved_hash,
                scenario=scenario,
                position_weight=position_weight,
                velocity_weight=velocity_weight,
            )
        except _ChallengeScenarioFailure as error:
            run = error.partial_run
        except Exception as error:
            exception = {
                "type": type(error).__name__,
                "message": str(error),
                "command_index": int(
                    getattr(error, "_challenge_command_index", 0)
                ),
            }
            run = _finalize_challenge_run(
                scenario=scenario,
                rows=_challenge_rows(),
                events=(),
                resolved_config_sha256=resolved_hash,
                dt=_CHALLENGE_DT,
                position_weight=position_weight,
                velocity_weight=velocity_weight,
                exception=exception,
            )
        runs.append(run)
    matrix_digest = hashlib.sha256()
    matrix_digest.update(b"independent-command-challenge-matrix/v1")
    matrix_digest.update(b"\x00")
    matrix_digest.update(resolved_hash.encode("ascii"))
    for run in runs:
        matrix_digest.update(b"\x00")
        matrix_digest.update(run.scenario.name.encode("utf-8"))
        matrix_digest.update(b"\x00")
        matrix_digest.update(run.command_sha256.encode("ascii"))
        matrix_digest.update(b"\x00")
        matrix_digest.update(run.deterministic_sha256.encode("ascii"))
    deterministic_hash = matrix_digest.hexdigest()
    metrics = {
        "position_weight": position_weight,
        "velocity_weight": velocity_weight,
        "scenario_count": len(runs),
        "completed_scenario_count": sum(
            run.metrics["completed_without_exception"] for run in runs
        ),
        "resolved_config_sha256": resolved_hash,
        "deterministic_sha256": deterministic_hash,
    }
    return ChallengeMatrix(
        runs=tuple(runs),
        metrics=MappingProxyType(metrics),
        resolved_config=MappingProxyType(challenge_config),
        resolved_config_sha256=resolved_hash,
        deterministic_sha256=deterministic_hash,
    )


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


def _write_challenge_scenario_artifacts(
    run: ChallengeScenarioRun,
    output: Path,
) -> dict:
    output.mkdir()
    for name, value in run.arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ContractError(
                f"challenge array must be pickle-free: {name}"
            )
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ContractError(f"challenge array must be finite: {name}")
    np.savez(output / "rollout.npz", **run.arrays)
    metrics = deepcopy(dict(run.metrics))
    metrics.update(
        {
            "scenario_name": run.scenario.name,
            "command_sha256": run.command_sha256,
            "deterministic_sha256": run.deterministic_sha256,
            "rollout_npz_sha256": _file_sha256(output / "rollout.npz"),
        }
    )
    metrics_bytes = _canonical_json_bytes(metrics)
    (output / "metrics.json").write_bytes(metrics_bytes)
    events_bytes = b"".join(
        _canonical_json_bytes(dict(event)) for event in run.events
    )
    (output / "events.jsonl").write_bytes(events_bytes)
    commands = {
        "name": run.scenario.name,
        "commands": [
            {
                "velocity_world_xy": list(command.velocity_world_xy),
                "heading_world_yaw": command.heading_world_yaw,
                "reset_before": command.reset_before,
            }
            for command in run.scenario.commands
        ],
    }
    (output / "commands.json").write_bytes(
        _canonical_json_bytes(commands)
    )
    return {
        "command_sha256": run.command_sha256,
        "deterministic_sha256": run.deterministic_sha256,
        "metrics_json_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "events_jsonl_sha256": hashlib.sha256(events_bytes).hexdigest(),
        "rollout_npz_sha256": metrics["rollout_npz_sha256"],
        "commands_json_sha256": _file_sha256(output / "commands.json"),
    }


def _read_canonical_json(path: Path, label: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} artifact is missing or a symlink")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} artifact is invalid JSON") from error
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise ContractError(f"{label} artifact is not canonical JSON")
    return value


def _load_authenticated_challenge_baseline(
    baseline_root: str | Path,
) -> dict[str, dict]:
    root = Path(os.path.abspath(os.fspath(baseline_root)))
    if root.is_symlink() or not root.is_dir():
        raise ContractError("challenge baseline root must be a real directory")
    matrix = _read_canonical_json(
        root / "matrix.json", "challenge baseline matrix"
    )
    if matrix.get("baseline") is not True:
        raise ContractError("challenge baseline artifact is not a baseline")
    if tuple(matrix.get("scenario_names", ())) != _CHALLENGE_NAMES:
        raise ContractError("challenge baseline scenario names differ")
    resolved_config_path = root / "resolved_config.json"
    resolved_config = _read_canonical_json(
        resolved_config_path, "challenge baseline resolved config"
    )
    if (
        _file_sha256(resolved_config_path)
        != matrix.get("resolved_config_json_sha256")
        or _json_sha256(resolved_config)
        != matrix.get("resolved_config_sha256")
    ):
        raise ContractError(
            "challenge baseline resolved config authentication failed"
        )
    artifacts = matrix.get("scenario_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(
        _CHALLENGE_NAMES
    ):
        raise ContractError("challenge baseline artifact index differs")
    metrics_by_name = {}
    for name in _CHALLENGE_NAMES:
        scenario_root = root / name
        if scenario_root.is_symlink() or not scenario_root.is_dir():
            raise ContractError(
                f"challenge baseline scenario is not a real directory: {name}"
            )
        artifact = artifacts[name]
        if not isinstance(artifact, dict):
            raise ContractError(
                f"challenge baseline artifact index is invalid: {name}"
            )
        metrics_path = scenario_root / "metrics.json"
        metrics = _read_canonical_json(
            metrics_path, f"challenge baseline metrics: {name}"
        )
        if _file_sha256(metrics_path) != artifact.get(
            "metrics_json_sha256"
        ):
            raise ContractError(
                f"challenge baseline metrics authentication failed: {name}"
            )
        rollout_path = scenario_root / "rollout.npz"
        if rollout_path.is_symlink() or not rollout_path.is_file():
            raise ContractError(
                f"challenge baseline rollout is missing: {name}"
            )
        rollout_sha = _file_sha256(rollout_path)
        if (
            rollout_sha != artifact.get("rollout_npz_sha256")
            or rollout_sha != metrics.get("rollout_npz_sha256")
        ):
            raise ContractError(
                f"challenge baseline rollout authentication failed: {name}"
            )
        try:
            with np.load(rollout_path, allow_pickle=False) as archive:
                for array_name in archive.files:
                    array = archive[array_name]
                    if array.dtype.hasobject or (
                        array.dtype.kind in "fc"
                        and not np.isfinite(array).all()
                    ):
                        raise ContractError(
                            "challenge baseline rollout array is invalid: "
                            f"{name}/{array_name}"
                        )
        except (OSError, ValueError) as error:
            raise ContractError(
                f"challenge baseline rollout is invalid: {name}"
            ) from error
        events_path = scenario_root / "events.jsonl"
        if (
            events_path.is_symlink()
            or not events_path.is_file()
            or _file_sha256(events_path)
            != artifact.get("events_jsonl_sha256")
        ):
            raise ContractError(
                f"challenge baseline events authentication failed: {name}"
            )
        commands_path = scenario_root / "commands.json"
        commands = _read_canonical_json(
            commands_path, f"challenge baseline commands: {name}"
        )
        if _file_sha256(commands_path) != artifact.get(
            "commands_json_sha256"
        ):
            raise ContractError(
                f"challenge baseline command authentication failed: {name}"
            )
        if commands.get("name") != name:
            raise ContractError(
                f"challenge baseline command scenario differs: {name}"
            )
        scenario = ChallengeScenario(
            name,
            tuple(
                ChallengeCommand(
                    tuple(command["velocity_world_xy"]),
                    command["heading_world_yaw"],
                    command["reset_before"],
                )
                for command in commands.get("commands", ())
            ),
        )
        command_sha = _challenge_command_sha256(scenario)
        if (
            command_sha != artifact.get("command_sha256")
            or command_sha != metrics.get("command_sha256")
        ):
            raise ContractError(
                f"challenge baseline command identity failed: {name}"
            )
        if metrics.get("deterministic_sha256") != artifact.get(
            "deterministic_sha256"
        ):
            raise ContractError(
                f"challenge baseline result identity failed: {name}"
            )
        metrics_by_name[name] = metrics
    return _validated_challenge_metric_map(
        metrics_by_name, label="baseline"
    )


def save_challenge_matrix(
    matrix: ChallengeMatrix,
    output: str | Path,
    *,
    baseline_root: str | Path | None = None,
) -> dict:
    """Transactionally save all scenario evidence and the matrix verdict."""

    if not isinstance(matrix, ChallengeMatrix):
        raise ContractError("only a ChallengeMatrix can be saved")
    names = tuple(run.scenario.name for run in matrix.runs)
    if names != _CHALLENGE_NAMES or len(set(names)) != len(names):
        raise ContractError("challenge matrix must contain all six scenarios")
    current_metrics = {}
    for run in matrix.runs:
        metric = deepcopy(dict(run.metrics))
        metric.update(
            {
                "scenario_name": run.scenario.name,
                "command_sha256": run.command_sha256,
                "deterministic_sha256": run.deterministic_sha256,
            }
        )
        current_metrics[run.scenario.name] = metric
    current_metrics = _validated_challenge_metric_map(
        current_metrics, label="retained"
    )
    baseline_metrics = (
        None
        if baseline_root is None
        else _load_authenticated_challenge_baseline(baseline_root)
    )
    if baseline_metrics is None:
        verdict = {
            "baseline": True,
            "matrix_pass": None,
            "material_improvement_count": None,
            "gates": None,
            "scenarios": {
                name: {
                    "command_sha256": current_metrics[name][
                        "command_sha256"
                    ],
                    "completed_without_exception": current_metrics[name][
                        "completed_without_exception"
                    ],
                    "minimum_clearance_m": current_metrics[name][
                        "minimum_clearance_m"
                    ],
                    "p95_joint_jerk_rad_s3": current_metrics[name][
                        "p95_joint_jerk_rad_s3"
                    ],
                    "exception": deepcopy(
                        current_metrics[name].get("exception")
                    ),
                }
                for name in _CHALLENGE_NAMES
            },
        }
    else:
        verdict = evaluate_challenge_matrix(
            baseline_metrics, current_metrics
        )
    output = _validated_lexical_output_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    committed = False
    try:
        scenario_artifacts = {
            run.scenario.name: _write_challenge_scenario_artifacts(
                run, staging / run.scenario.name
            )
            for run in matrix.runs
        }
        (staging / "resolved_config.json").write_bytes(
            _canonical_json_bytes(dict(matrix.resolved_config))
        )
        top = deepcopy(verdict)
        top.update(
            {
                "scenario_names": list(_CHALLENGE_NAMES),
                "scenario_artifacts": scenario_artifacts,
                "position_weight": matrix.metrics.get("position_weight"),
                "velocity_weight": matrix.metrics.get("velocity_weight"),
                "resolved_config_sha256": matrix.resolved_config_sha256,
                "deterministic_sha256": matrix.deterministic_sha256,
                "resolved_config_json_sha256": _file_sha256(
                    staging / "resolved_config.json"
                ),
            }
        )
        (staging / "matrix.json").write_bytes(_canonical_json_bytes(top))
        if output.exists():
            if not output.is_dir() or output.is_symlink():
                raise ContractError(
                    "challenge matrix output must be a real directory"
                )
            backup = output.with_name(output.name + ".previous")
            if backup.exists():
                raise ContractError(
                    "challenge matrix backup path already exists"
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
    return top


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
        "--position-weight", type=_argument_weight
    )
    parser.add_argument(
        "--velocity-weight", type=_argument_weight
    )
    parser.add_argument(
        "--challenge-matrix",
        action="store_true",
        help="run the six-scenario independent-command stress matrix",
    )
    parser.add_argument(
        "--baseline-root",
        help="authenticated challenge baseline artifact root",
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
    needs_resolved_weight = (
        args.position_weight is None or args.velocity_weight is None
    )
    matcher_config = (
        resolved.resolved_config.get("matcher")
        if needs_resolved_weight
        else None
    )
    if needs_resolved_weight and not isinstance(matcher_config, Mapping):
        raise ContractError("resolved matcher config must be a mapping")
    position_weight = _validate_weight(
        (
            matcher_config.get("transition_joint_position_weight")
            if args.position_weight is None
            else args.position_weight
        ),
        "position weight",
    )
    velocity_weight = _validate_weight(
        (
            matcher_config.get("transition_joint_velocity_weight")
            if args.velocity_weight is None
            else args.velocity_weight
        ),
        "velocity weight",
    )
    if args.baseline_root is not None and not args.challenge_matrix:
        raise ContractError(
            "--baseline-root requires --challenge-matrix"
        )
    if args.challenge_matrix:
        matrix = run_challenge_matrix(
            resolved,
            device=device,
            position_weight=position_weight,
            velocity_weight=velocity_weight,
        )
        save_challenge_matrix(
            matrix,
            output,
            baseline_root=(
                None
                if args.baseline_root is None
                else Path(args.baseline_root)
            ),
        )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "deterministic_sha256": matrix.deterministic_sha256,
                    "metrics": dict(matrix.metrics),
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        return 0
    rollout = run_directional_rollout(
        resolved,
        device=device,
        position_weight=position_weight,
        velocity_weight=velocity_weight,
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
