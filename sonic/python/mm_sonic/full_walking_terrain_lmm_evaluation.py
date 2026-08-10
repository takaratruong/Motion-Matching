"""Deterministic coverage, speed, and planted-foot reducers for formal routes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_SAFETY_KEYS = (
    "candidate_exhaustion",
    "fallback",
    "joint_clamp",
    "native_limit_violation",
    "nonfinite",
    "out_of_range_successor",
)


def _finite_vector(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite vector")
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FormalRoute:
    """One continuous-time command authority for an authenticated scene."""

    scene_id: str
    duration_seconds: float
    command_times: np.ndarray
    command_speed: np.ndarray
    command_steering: np.ndarray

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id)
        duration = float(self.duration_seconds)
        times = _finite_vector(self.command_times, "command times")
        speed = _finite_vector(self.command_speed, "command speed")
        steering = _finite_vector(self.command_steering, "command steering")
        if not scene_id:
            raise ValueError("formal route scene ID must be non-empty")
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("formal route duration must be finite and nonnegative")
        if len(times) == 0 or len(speed) != len(times) or len(steering) != len(times):
            raise ValueError("formal route command vectors must be nonempty and aligned")
        if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("formal route command times must start at zero and increase")
        if times[-1] > duration:
            raise ValueError("formal route command time exceeds its duration")
        if np.any(np.abs(speed) > 1.0) or np.any(np.abs(steering) > 1.0):
            raise ValueError("formal route commands must be normalized to [-1,1]")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "duration_seconds", duration)
        object.__setattr__(self, "command_times", times)
        object.__setattr__(self, "command_speed", speed)
        object.__setattr__(self, "command_steering", steering)

    def sample_times(self, fps: float) -> np.ndarray:
        rate = float(fps)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("route sampling rate must be finite and positive")
        count = math.floor(self.duration_seconds * rate + 1.0e-9) + 1
        result = np.arange(count, dtype=np.float64) / rate
        result.setflags(write=False)
        return result

    def commands_at(self, sample_times: object) -> tuple[np.ndarray, np.ndarray]:
        times = _finite_vector(sample_times, "sample times")
        if len(times) and (times[0] < 0.0 or times[-1] > self.duration_seconds + 1e-9):
            raise ValueError("sample time lies outside the formal route")
        indices = np.searchsorted(self.command_times, times, side="right") - 1
        indices = np.maximum(indices, 0)
        speed = np.asarray(self.command_speed[indices], dtype=np.float64)
        steering = np.asarray(self.command_steering[indices], dtype=np.float64)
        speed.setflags(write=False)
        steering.setflags(write=False)
        return speed, steering


@dataclass
class SlipAccumulator:
    """Accumulate planar probe velocity only within continuing contact runs."""

    planted_speeds_mps: list[float] = field(default_factory=list)

    def update(
        self,
        previous_probes: np.ndarray,
        current_probes: np.ndarray,
        previous_contacts: np.ndarray,
        current_contacts: np.ndarray,
        *,
        dt: float,
    ) -> None:
        previous = np.asarray(previous_probes, dtype=np.float64)
        current = np.asarray(current_probes, dtype=np.float64)
        prior_contact = np.asarray(previous_contacts)
        contact = np.asarray(current_contacts)
        elapsed = float(dt)
        if (
            previous.ndim != 2
            or previous.shape[1] != 3
            or current.shape != previous.shape
            or prior_contact.shape != (len(previous),)
            or contact.shape != (len(previous),)
            or prior_contact.dtype.kind != "b"
            or contact.dtype.kind != "b"
        ):
            raise ValueError("probe/contact samples must align as [probe,xyz] and [probe]")
        if not np.isfinite(previous).all() or not np.isfinite(current).all():
            raise ValueError("support probes must be finite")
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError("slip timestep must be finite and positive")
        planted = np.logical_and(prior_contact, contact)
        speeds = np.linalg.norm(current[planted, :2] - previous[planted, :2], axis=1)
        self.planted_speeds_mps.extend(float(value / elapsed) for value in speeds)

    def summary(self) -> dict[str, int | float | None]:
        values = np.asarray(self.planted_speeds_mps, dtype=np.float64)
        if len(values) == 0:
            return {"sample_count": 0, "median_mps": None, "p95_mps": None}
        return {
            "sample_count": len(values),
            "median_mps": float(np.median(values)),
            "p95_mps": float(np.percentile(values, 95.0)),
        }


@dataclass(frozen=True)
class EvaluationSeries:
    """Post-``mj_forward`` observations returned by a route evaluator adapter."""

    probes: np.ndarray
    contacts: np.ndarray
    root_xy: np.ndarray
    desired_speed_mps: np.ndarray
    query_distances: np.ndarray
    canonical_source_ids: tuple[str, ...]
    forward_count: int
    safety_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        probes = np.asarray(self.probes, dtype=np.float64)
        contacts = np.asarray(self.contacts)
        root = np.asarray(self.root_xy, dtype=np.float64)
        desired = np.asarray(self.desired_speed_mps, dtype=np.float64)
        distance = np.asarray(self.query_distances, dtype=np.float64)
        frames = len(probes)
        if probes.ndim != 3 or probes.shape[2] != 3:
            raise ValueError("evaluation probes must have shape [frame,probe,xyz]")
        if contacts.shape != probes.shape[:2] or contacts.dtype.kind != "b":
            raise ValueError("evaluation contacts must be boolean [frame,probe]")
        if root.shape != (frames, 2) or desired.shape != (frames,) or distance.shape != (frames,):
            raise ValueError("evaluation root/speed/distance timelines must align")
        if len(self.canonical_source_ids) != frames:
            raise ValueError("canonical source timeline must align with evaluation frames")
        if not all(np.isfinite(value).all() for value in (probes, root, desired, distance)):
            raise ValueError("evaluation observations must be finite")
        if type(self.forward_count) is not int or self.forward_count != frames:
            raise ValueError("evaluation forward count must equal the observation count")
        safety = {name: int(self.safety_counts.get(name, 0)) for name in _SAFETY_KEYS}
        if any(value < 0 for value in safety.values()):
            raise ValueError("evaluation safety counters must be nonnegative")
        for name, value in (
            ("probes", probes),
            ("contacts", contacts.astype(np.bool_, copy=False)),
            ("root_xy", root),
            ("desired_speed_mps", desired),
            ("query_distances", distance),
        ):
            frozen = np.array(value, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        object.__setattr__(
            self, "canonical_source_ids", tuple(str(value) for value in self.canonical_source_ids)
        )
        object.__setattr__(self, "safety_counts", safety)


def _terrain_name(scene_id: str) -> str:
    lowered = scene_id.lower()
    for name in ("flat", "curb", "stair", "slope"):
        if name in lowered:
            return name
    if "ramp" in lowered:
        return "slope"
    return lowered


def _route_authority_sha256(routes: Sequence[FormalRoute]) -> str:
    payload = [
        {
            "scene_id": route.scene_id,
            "duration_seconds": route.duration_seconds,
            "command_times": route.command_times.tolist(),
            "command_speed": route.command_speed.tolist(),
            "command_steering": route.command_steering.tolist(),
        }
        for route in routes
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _evaluate(
    evaluator: Any,
    route: FormalRoute,
    times: np.ndarray,
    speed: np.ndarray,
    steering: np.ndarray,
    g1_xml: Path,
) -> EvaluationSeries:
    method = getattr(evaluator, "evaluate_formal_route", None)
    if not callable(method):
        raise TypeError(
            "formal comparison requires an evaluator adapter that performs "
            "MuJoCo forwarding and exposes evaluate_formal_route"
        )
    result = method(route, times, speed, steering, g1_xml=Path(g1_xml))
    if not isinstance(result, EvaluationSeries):
        raise TypeError("formal route evaluator must return EvaluationSeries")
    return result


def _reduce_route(series: EvaluationSeries, *, fps: float) -> dict[str, object]:
    slip = SlipAccumulator()
    dt = 1.0 / fps
    for frame in range(1, series.forward_count):
        slip.update(
            series.probes[frame - 1], series.probes[frame],
            series.contacts[frame - 1], series.contacts[frame], dt=dt,
        )
    realized = np.linalg.norm(np.diff(series.root_xy, axis=0), axis=1) / dt
    speed_error = np.abs(realized - np.abs(series.desired_speed_mps[1:]))
    identities = sorted(set(series.canonical_source_ids))
    return {
        "forward_count": series.forward_count,
        "slip": slip.summary(),
        "_slip_values": tuple(slip.planted_speeds_mps),
        "speed_absolute_errors_mps": speed_error.tolist(),
        "query_distances": series.query_distances.tolist(),
        "canonical_identities": identities,
        "canonical_identity_count": len(identities),
        "safety_counts": dict(series.safety_counts),
    }


def _aggregate(
    route_results: Sequence[tuple[FormalRoute, dict[str, object]]], authority_sha: str
) -> dict[str, object]:
    slip_values: list[float] = []
    speed_errors: list[float] = []
    distances: list[float] = []
    safety = {name: 0 for name in _SAFETY_KEYS}
    by_terrain: dict[str, dict[str, object]] = {}
    forward_count = 0
    for route, result in route_results:
        forward_count += int(result["forward_count"])
        speed_errors.extend(float(value) for value in result["speed_absolute_errors_mps"])
        distances.extend(float(value) for value in result["query_distances"])
        raw_slip = result.get("_slip_values", ())
        slip_values.extend(float(value) for value in raw_slip)
        for name in _SAFETY_KEYS:
            safety[name] += int(result["safety_counts"].get(name, 0))
        terrain = _terrain_name(route.scene_id)
        entry = by_terrain.setdefault(
            terrain,
            {"forward_count": 0, "canonical_identities": set()},
        )
        entry["forward_count"] = int(entry["forward_count"]) + int(result["forward_count"])
        entry["canonical_identities"].update(result["canonical_identities"])
    by_terrain_json: dict[str, dict[str, object]] = {}
    for terrain, entry in sorted(by_terrain.items()):
        identities = sorted(entry["canonical_identities"])
        by_terrain_json[terrain] = {
            "forward_count": entry["forward_count"],
            "canonical_identities": identities,
            "canonical_identity_count": len(identities),
        }
    slip_array = np.asarray(slip_values, dtype=np.float64)
    return {
        "forward_count": forward_count,
        "slip": {
            "sample_count": len(slip_array),
            "median_mps": float(np.median(slip_array)) if len(slip_array) else None,
            "p95_mps": float(np.percentile(slip_array, 95.0)) if len(slip_array) else None,
        },
        "speed_sample_count": len(speed_errors),
        "speed_mae_mps": float(np.mean(speed_errors)) if speed_errors else None,
        "query_sample_count": len(distances),
        "query_distance_p95": float(np.percentile(distances, 95.0)) if distances else None,
        "safety_counts": safety,
        "by_terrain": by_terrain_json,
        "route_command_authority_sha256": authority_sha,
    }


def _reduction(baseline: object, candidate: object) -> float:
    old, new = float(baseline), float(candidate)
    if old == 0.0:
        return 0.0 if new == 0.0 else -1.0
    return float((old - new) / old)


def compare_baseline_candidate(
    *,
    baseline: Any,
    candidate: Any,
    routes: Sequence[FormalRoute],
    g1_xml: Path,
) -> Mapping[str, object]:
    """Evaluate identical continuous routes at frozen 25 Hz and candidate 60 Hz."""

    route_values = tuple(routes)
    if not route_values or not all(isinstance(route, FormalRoute) for route in route_values):
        raise ValueError("formal comparison requires at least one FormalRoute")
    if float(getattr(baseline, "fps", math.nan)) != 25.0:
        raise ValueError("formal baseline evaluator must run at exactly 25 Hz")
    if float(getattr(candidate, "fps", math.nan)) != 60.0:
        raise ValueError("formal candidate evaluator must run at exactly 60 Hz")
    authority_sha = _route_authority_sha256(route_values)
    reduced: dict[str, list[tuple[FormalRoute, dict[str, object]]]] = {
        "baseline": [], "candidate": []
    }
    for route in route_values:
        for name, evaluator, fps in (
            ("baseline", baseline, 25.0), ("candidate", candidate, 60.0)
        ):
            times = route.sample_times(fps)
            speed, steering = route.commands_at(times)
            series = _evaluate(evaluator, route, times, speed, steering, Path(g1_xml))
            route_result = _reduce_route(series, fps=fps)
            reduced[name].append((route, route_result))
    baseline_result = _aggregate(reduced["baseline"], authority_sha)
    candidate_result = _aggregate(reduced["candidate"], authority_sha)
    comparison = {
        "query_distance_p95_reduction_fraction": _reduction(
            baseline_result["query_distance_p95"], candidate_result["query_distance_p95"]
        ),
        "slip_p95_reduction_fraction": _reduction(
            baseline_result["slip"]["p95_mps"], candidate_result["slip"]["p95_mps"]
        ),
    }
    return {
        "schema": "g1-full-walking-terrain-lmm-comparison/v1",
        "route_command_authority_sha256": authority_sha,
        "baseline": baseline_result,
        "candidate": candidate_result,
        "comparison": comparison,
    }


__all__ = (
    "EvaluationSeries",
    "FormalRoute",
    "SlipAccumulator",
    "compare_baseline_candidate",
)
