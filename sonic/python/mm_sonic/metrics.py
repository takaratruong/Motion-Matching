"""Pure SONIC delivery, tracking, contact, threshold, and verdict metrics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from .artifacts import RunBundle
from .coordinator import (
    LOGGER_JOINT_PERMUTATION,
    DeliveryAudit,
    DeliveryAuditEvidence,
    parse_official_target_row,
)
from .external import (
    CURRENT_FRAME_ADVANCEMENT_SHA256,
    VerifiedExternal,
    VerifiedGearCheckout,
)
from .joints import ContractError
from .timeline import CanonicalTargetBuffer
from .zmq_v1 import DecodedPoseV1, decode_pose_v1


PINNED_GEAR_COMMIT = "60de0df7ffedeef415fe58d435e92cc5b01ba3d9"
PINNED_CURRENT_FRAME_ADVANCEMENT_SHA256 = CURRENT_FRAME_ADVANCEMENT_SHA256
FORBIDDEN_CONTACT_GROUPS = frozenset(("pelvis", "knees", "torso", "hands"))
REGISTERED_TERRAIN_SCENE_IDS = (
    "grail-curb-low",
    "ramp-10-up-down",
    "stairs-shallow",
)
TARGET_RADIUS_M = 0.25
DURATION_MULTIPLIER = 1.25
MINIMUM_PELVIS_LOCAL_HEIGHT_M = 0.45
MINIMUM_PELVIS_UP_DOT = 0.5
REFERENCE_PENETRATION_LIMIT_M = 0.005
KNOWN_GOOD_METRIC_MULTIPLIER = 1.5
KNOWN_GOOD_ZERO_EPSILON = 1.0e-8


def _finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ContractError(f"{label} must be finite")
    return float(value)


def _finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        output = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must be numeric") from error
    if output.shape != shape or not np.all(np.isfinite(output)):
        raise ContractError(f"{label} must have shape {shape} with finite values")
    return output


def _frame_indices(value: object, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or raw.shape[0] <= 0
        or raw.dtype.kind not in "iu"
        or raw.dtype.kind == "b"
    ):
        raise ContractError(f"{label} frame indices must be a nonempty integer vector")
    indices = raw.astype(np.int64, copy=True)
    if np.any(indices < 0) or not np.all(np.diff(indices) == 1):
        raise ContractError(f"{label} frame indices must be unique and contiguous")
    return indices


@dataclass(frozen=True)
class TrackingMetrics:
    joint_position_rmse_rad: float
    pelvis_orientation_rms_rad: float
    joint_tracking_trace_rad: tuple[float, ...]
    pelvis_tracking_trace_rad: tuple[float, ...]


def tracking_metrics(
    target_frame_index: object,
    target_joint_position: object,
    target_pelvis_quat_wxyz: object,
    actual_frame_index: object,
    actual_joint_position: object,
    actual_pelvis_quat_wxyz: object,
) -> TrackingMetrics:
    """Score exact aligned 50 Hz rows without averaging per-joint RMSEs."""

    target_indices = _frame_indices(target_frame_index, "target")
    actual_indices = _frame_indices(actual_frame_index, "actual")
    if not np.array_equal(target_indices, actual_indices):
        raise ContractError("target and actual frame coverage must be exactly aligned")
    count = int(target_indices.shape[0])
    target_joint = _finite_array(
        target_joint_position, (count, 29), "target joint position"
    )
    actual_joint = _finite_array(
        actual_joint_position, (count, 29), "actual joint position"
    )
    target_quat = _finite_array(
        target_pelvis_quat_wxyz, (count, 4), "target pelvis quaternion"
    )
    actual_quat = _finite_array(
        actual_pelvis_quat_wxyz, (count, 4), "actual pelvis quaternion"
    )
    for label, quaternions in (
        ("target", target_quat),
        ("actual", actual_quat),
    ):
        norms = np.linalg.norm(quaternions, axis=1)
        if np.any(np.abs(norms - 1.0) > 1.0e-6):
            raise ContractError(f"{label} pelvis values must be unit quaternions")

    joint_error = actual_joint - target_joint
    joint_trace = np.sqrt(np.mean(np.square(joint_error), axis=1))
    joint_rmse = math.sqrt(float(np.mean(np.square(joint_error))))
    dots = np.sum(target_quat * actual_quat, axis=1)
    angles = 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
    pelvis_rms = math.sqrt(float(np.mean(np.square(angles))))
    return TrackingMetrics(
        joint_position_rmse_rad=joint_rmse,
        pelvis_orientation_rms_rad=pelvis_rms,
        joint_tracking_trace_rad=tuple(float(value) for value in joint_trace),
        pelvis_tracking_trace_rad=tuple(float(value) for value in angles),
    )


def minimum_local_pelvis_height(
    pelvis_position_mujoco_xyz: object,
    local_terrain_height_m: object,
) -> float:
    positions = np.asarray(pelvis_position_mujoco_xyz, dtype=np.float64)
    terrain = np.asarray(local_terrain_height_m, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[0] <= 0
        or positions.shape[1] != 3
        or terrain.shape != (positions.shape[0],)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(terrain))
    ):
        raise ContractError(
            "local pelvis height requires aligned finite xyz/terrain rows"
        )
    return float(np.min(positions[:, 2] - terrain))


def pelvis_up_dots(pelvis_quat_wxyz: object) -> np.ndarray:
    quaternions = np.asarray(pelvis_quat_wxyz, dtype=np.float64)
    if (
        quaternions.ndim != 2
        or quaternions.shape[0] <= 0
        or quaternions.shape[1] != 4
        or not np.all(np.isfinite(quaternions))
    ):
        raise ContractError("pelvis up-dot requires finite wxyz quaternion rows")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(np.abs(norms - 1.0) > 1.0e-6):
        raise ContractError("pelvis up-dot requires unit quaternions")
    _w, x, y, _z = np.moveaxis(quaternions, -1, 0)
    return 1.0 - 2.0 * (np.square(x) + np.square(y))


def _contact_groups(groups: Iterable[str]) -> tuple[str, ...]:
    values = tuple(groups)
    if any(type(value) is not str or not value for value in values):
        raise ContractError("contact groups must be nonempty strings")
    return values


def has_forbidden_contact(groups: Iterable[str]) -> bool:
    return any(group in FORBIDDEN_CONTACT_GROUPS for group in _contact_groups(groups))


def within_target_radius(distance_m: float) -> bool:
    distance = _finite(distance_m, "target distance")
    return distance >= 0.0 and distance <= TARGET_RADIUS_M


def within_duration(reached_time_s: float, nominal_duration_s: float) -> bool:
    reached = _finite(reached_time_s, "target time")
    nominal = _finite(nominal_duration_s, "nominal duration")
    return reached >= 0.0 and nominal > 0.0 and reached <= DURATION_MULTIPLIER * nominal


def has_reference_penetration(
    contact_groups: Iterable[str], penetration_depth_m: Iterable[float]
) -> bool:
    groups = _contact_groups(contact_groups)
    depths = tuple(_finite(value, "penetration depth") for value in penetration_depth_m)
    if len(groups) != len(depths) or any(value < 0.0 for value in depths):
        raise ContractError("penetration rows must align and be nonnegative")
    return any(
        group in FORBIDDEN_CONTACT_GROUPS
        and depth > REFERENCE_PENETRATION_LIMIT_M
        for group, depth in zip(groups, depths, strict=True)
    )


def within_known_good_ratio(candidate: float, known_good: float) -> bool:
    candidate_value = _finite(candidate, "candidate metric")
    known_good_value = _finite(known_good, "known-good metric")
    if candidate_value < 0.0 or known_good_value < 0.0:
        raise ContractError("tracking metrics must be nonnegative")
    if known_good_value <= KNOWN_GOOD_ZERO_EPSILON:
        return candidate_value <= KNOWN_GOOD_ZERO_EPSILON
    return candidate_value <= KNOWN_GOOD_METRIC_MULTIPLIER * known_good_value


def nearest_rank_summary(samples: Iterable[int | float]) -> dict[str, int | float]:
    values = tuple(samples)
    if not values:
        raise ContractError("timing summary requires at least one sample")
    for value in values:
        _finite(value, "timing sample")
        if float(value) < 0.0:
            raise ContractError("timing samples must be nonnegative")
    ordered = sorted(values)

    def nearest(percentile: float):
        rank = max(1, int(math.ceil(percentile * len(ordered))))
        return ordered[rank - 1]

    return {"p50": nearest(0.50), "p95": nearest(0.95), "p99": nearest(0.99)}


@dataclass(frozen=True)
class SecondaryMetrics:
    swing_foot_scuff_count: int
    minimum_foot_clearance_m: float
    horizontal_path_drift_m: float
    joint_tracking_trace_rad: tuple[float, ...]
    pelvis_tracking_trace_rad: tuple[float, ...]
    contact_impulses_ns: tuple[float, ...]
    policy_execution_timing_ns: Mapping[str, int | float] | None

    def __post_init__(self) -> None:
        if (
            type(self.swing_foot_scuff_count) is not int
            or self.swing_foot_scuff_count < 0
        ):
            raise ContractError("swing-foot scuff count must be nonnegative")
        for label, value in (
            ("minimum foot clearance", self.minimum_foot_clearance_m),
            ("horizontal path drift", self.horizontal_path_drift_m),
        ):
            _finite(value, label)
        if self.horizontal_path_drift_m < 0.0:
            raise ContractError("horizontal path drift must be nonnegative")
        for label in ("joint_tracking_trace_rad", "pelvis_tracking_trace_rad"):
            values = tuple(getattr(self, label))
            if not values:
                raise ContractError(f"{label} must not be empty")
            for value in values:
                if _finite(value, label) < 0.0:
                    raise ContractError(f"{label} must be nonnegative")
            object.__setattr__(self, label, values)
        impulses = tuple(self.contact_impulses_ns)
        for value in impulses:
            if _finite(value, "contact_impulses_ns") < 0.0:
                raise ContractError("contact_impulses_ns must be nonnegative")
        object.__setattr__(self, "contact_impulses_ns", impulses)
        if self.policy_execution_timing_ns is not None:
            timing = dict(self.policy_execution_timing_ns)
            if set(timing) != {"p50", "p95", "p99"}:
                raise ContractError("policy timing must contain p50/p95/p99")
            for value in timing.values():
                if _finite(value, "policy timing") < 0.0:
                    raise ContractError("policy timing must be nonnegative")
            if not (timing["p50"] <= timing["p95"] <= timing["p99"]):
                raise ContractError("policy timing quantiles must be ordered")
            object.__setattr__(
                self, "policy_execution_timing_ns", MappingProxyType(timing)
            )


@dataclass(frozen=True)
class DynamicTrialVerdict:
    integration_pass: bool
    dynamic_pass: bool
    checks: Mapping[str, bool]
    secondary: SecondaryMetrics


def evaluate_terrain_trial(
    *,
    integration_pass: bool,
    exact_frame_coverage: bool,
    target_distance_m: float,
    reached_time_s: float,
    nominal_duration_s: float,
    minimum_pelvis_local_height_m: float,
    minimum_pelvis_up_dot: float,
    contact_groups: Iterable[str],
    secondary: SecondaryMetrics,
) -> DynamicTrialVerdict:
    if type(integration_pass) is not bool or type(exact_frame_coverage) is not bool:
        raise ContractError("trial integration/coverage gates must be booleans")
    if not isinstance(secondary, SecondaryMetrics):
        raise ContractError("trial requires complete secondary metrics")
    height = _finite(minimum_pelvis_local_height_m, "minimum local pelvis height")
    up_dot = _finite(minimum_pelvis_up_dot, "minimum pelvis up-dot")
    checks = {
        "integration": integration_pass,
        "exact_frame_coverage": exact_frame_coverage,
        "target_radius": within_target_radius(target_distance_m),
        "duration": within_duration(reached_time_s, nominal_duration_s),
        "pelvis_local_height": height >= MINIMUM_PELVIS_LOCAL_HEIGHT_M,
        "pelvis_up_dot": up_dot >= MINIMUM_PELVIS_UP_DOT,
        "forbidden_contacts": not has_forbidden_contact(contact_groups),
    }
    return DynamicTrialVerdict(
        integration_pass=integration_pass,
        dynamic_pass=all(checks.values()),
        checks=MappingProxyType(checks),
        secondary=secondary,
    )


def evaluate_dynamic_trial(
    *,
    integration_pass: bool,
    exact_frame_coverage: bool,
    target_distance_m: float,
    reached_time_s: float,
    nominal_duration_s: float,
    minimum_pelvis_local_height_m: float,
    minimum_pelvis_up_dot: float,
    contact_groups: Iterable[str],
    secondary: SecondaryMetrics,
) -> DynamicTrialVerdict:
    """Backward-compatible name for the Stage C terrain trial evaluator."""

    return evaluate_terrain_trial(
        integration_pass=integration_pass,
        exact_frame_coverage=exact_frame_coverage,
        target_distance_m=target_distance_m,
        reached_time_s=reached_time_s,
        nominal_duration_s=nominal_duration_s,
        minimum_pelvis_local_height_m=minimum_pelvis_local_height_m,
        minimum_pelvis_up_dot=minimum_pelvis_up_dot,
        contact_groups=contact_groups,
        secondary=secondary,
    )


def evaluate_flat_trial(
    *,
    integration_pass: bool,
    exact_command_coverage: bool,
    exact_frame_coverage: bool,
    minimum_pelvis_local_height_m: float,
    minimum_pelvis_up_dot: float,
    contact_groups: Iterable[str],
    joint_position_rmse_rad: float,
    known_good_joint_position_rmse_rad: float,
    pelvis_orientation_rms_rad: float,
    known_good_pelvis_orientation_rms_rad: float,
    secondary: SecondaryMetrics,
) -> DynamicTrialVerdict:
    """Evaluate every registered Stage B safety, coverage, and tracking gate."""

    booleans = (
        integration_pass,
        exact_command_coverage,
        exact_frame_coverage,
    )
    if any(type(value) is not bool for value in booleans):
        raise ContractError("flat trial integration/coverage gates must be booleans")
    if not isinstance(secondary, SecondaryMetrics):
        raise ContractError("flat trial requires complete secondary metrics")
    height = _finite(minimum_pelvis_local_height_m, "minimum local pelvis height")
    up_dot = _finite(minimum_pelvis_up_dot, "minimum pelvis up-dot")
    checks = {
        "integration": integration_pass,
        "exact_command_coverage": exact_command_coverage,
        "exact_frame_coverage": exact_frame_coverage,
        "pelvis_local_height": height >= MINIMUM_PELVIS_LOCAL_HEIGHT_M,
        "pelvis_up_dot": up_dot >= MINIMUM_PELVIS_UP_DOT,
        "forbidden_contacts": not has_forbidden_contact(contact_groups),
        "joint_tracking_ratio": within_known_good_ratio(
            joint_position_rmse_rad,
            known_good_joint_position_rmse_rad,
        ),
        "pelvis_tracking_ratio": within_known_good_ratio(
            pelvis_orientation_rms_rad,
            known_good_pelvis_orientation_rms_rad,
        ),
    }
    return DynamicTrialVerdict(
        integration_pass=integration_pass,
        dynamic_pass=all(checks.values()),
        checks=MappingProxyType(checks),
        secondary=secondary,
    )


@dataclass(frozen=True)
class SceneHypothesisVerdict:
    aware_successes: int
    blind_successes: int
    pass_margin: int
    composition_feasible: bool
    awareness_effect: str
    hypothesis_supported: bool


@dataclass(frozen=True)
class OverallHypothesisVerdict:
    scene_verdicts: Mapping[str, SceneHypothesisVerdict]
    composition_feasible: bool
    awareness_effect: str
    hypothesis_supported: bool


def aggregate_scene_hypothesis(
    aware_successes: int, blind_successes: int
) -> SceneHypothesisVerdict:
    for label, value in (
        ("aware successes", aware_successes),
        ("blind successes", blind_successes),
    ):
        if type(value) is not int or value < 0 or value > 10:
            raise ContractError(f"{label} must be an integer in 0..10")
    margin = aware_successes - blind_successes
    aware_pass = aware_successes >= 8
    blind_pass = blind_successes >= 8
    if aware_pass and blind_pass:
        effect = "inconclusive"
    elif aware_pass and margin >= 3:
        effect = "supported"
    else:
        effect = "not_supported"
    return SceneHypothesisVerdict(
        aware_successes=aware_successes,
        blind_successes=blind_successes,
        pass_margin=margin,
        composition_feasible=aware_pass,
        awareness_effect=effect,
        hypothesis_supported=effect == "supported",
    )


def aggregate_overall_hypothesis(
    scene_verdicts: Mapping[str, SceneHypothesisVerdict],
) -> OverallHypothesisVerdict:
    """Require a verdict for every registered class; never average classes."""

    if not isinstance(scene_verdicts, Mapping):
        raise ContractError("overall hypothesis requires three registered classes")
    copied = dict(scene_verdicts)
    if set(copied) != set(REGISTERED_TERRAIN_SCENE_IDS) or any(
        not isinstance(value, SceneHypothesisVerdict) for value in copied.values()
    ):
        raise ContractError("overall hypothesis requires three registered classes")
    ordered = {
        scene_id: copied[scene_id] for scene_id in REGISTERED_TERRAIN_SCENE_IDS
    }
    composition = all(value.composition_feasible for value in ordered.values())
    if all(value.hypothesis_supported for value in ordered.values()):
        effect = "supported"
    elif any(value.awareness_effect == "not_supported" for value in ordered.values()):
        effect = "not_supported"
    else:
        effect = "inconclusive"
    return OverallHypothesisVerdict(
        scene_verdicts=MappingProxyType(ordered),
        composition_feasible=composition,
        awareness_effect=effect,
        hypothesis_supported=effect == "supported",
    )


def attribute_failure(
    *,
    stage: str,
    integration_pass: bool,
    known_good_file_pass: bool,
    known_good_stream_pass: bool,
    kinematic_pass: bool,
    dynamic_pass: bool,
) -> str | None:
    booleans = (
        integration_pass,
        known_good_file_pass,
        known_good_stream_pass,
        kinematic_pass,
        dynamic_pass,
    )
    if type(stage) is not str or stage not in {"A", "B", "C"}:
        raise ContractError("failure attribution stage must be A, B, or C")
    if any(type(value) is not bool for value in booleans):
        raise ContractError("failure attribution gates must be booleans")
    if not integration_pass:
        return "integration"
    if not known_good_file_pass:
        return "gear_setup_model_simulator_or_harness"
    if not known_good_stream_pass:
        return "bridge_protocol"
    if stage == "B":
        if not kinematic_pass:
            return "joint_or_coordinate_conversion"
        if not dynamic_pass:
            return "reference_distribution_or_sonic_tracking"
    if stage == "C":
        if not kinematic_pass:
            return "motion_matching_reference"
        if not dynamic_pass:
            return "low_level_tracking_contact_domain_mismatch"
    if stage == "A" and not dynamic_pass:
        return "gear_setup_model_simulator_or_harness"
    return None


def _canonical_bytes(buffer: CanonicalTargetBuffer) -> bytes:
    if not isinstance(buffer, CanonicalTargetBuffer):
        raise ContractError("delivery audit requires a canonical target buffer")
    expected_indices = np.arange(buffer.count, dtype=np.int64)
    if not np.array_equal(buffer.frame_index, expected_indices):
        raise ContractError("canonical delivery frame indices must be exactly 0..N-1")
    return b"".join(
        (
            np.asarray(buffer.joint_position, dtype="<f4").tobytes(order="C"),
            np.asarray(buffer.joint_velocity, dtype="<f4").tobytes(order="C"),
            np.asarray(buffer.body_quat_w, dtype="<f4").tobytes(order="C"),
            np.asarray(buffer.frame_index, dtype="<i8").tobytes(order="C"),
        )
    )


def _canonical_json_bytes(value: object, label: str) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ContractError(
            f"delivery transcript {label} is not canonical JSON"
        ) from error


class _EvidenceTranscript:
    """Length-delimited, domain-separated digest of every audited input."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.add("domain", b"mm-sonic-delivery-evidence/v1")

    def add(self, label: str, payload: bytes) -> None:
        encoded_label = label.encode("ascii")
        self._digest.update(b"\x01")
        self._digest.update(len(encoded_label).to_bytes(4, "big"))
        self._digest.update(encoded_label)
        self._digest.update(len(payload).to_bytes(8, "big"))
        self._digest.update(payload)

    def add_json(self, label: str, value: object) -> None:
        self.add(label, _canonical_json_bytes(value, label))

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _official_row(buffer: CanonicalTargetBuffer, index: int) -> bytes:
    permutation = np.asarray(LOGGER_JOINT_PERMUTATION, dtype=np.int64)
    values = np.concatenate(
        (
            np.zeros(3, dtype=np.float32),
            buffer.body_quat_w[index],
            buffer.joint_position[index, permutation],
        )
    ).astype(np.float32, copy=False)
    return (
        ",".join(format(float(value), ".6g") for value in values) + ",\n"
    ).encode("ascii")


def _relative_parts(relative: object) -> tuple[str, ...]:
    if type(relative) is not str or not relative:
        raise ContractError("delivery archive path must be nonempty")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ContractError("delivery archive path is not confined")
    return path.parts


def _read_regular_confined(run: RunBundle, relative: object) -> bytes:
    if not isinstance(run, RunBundle):
        raise ContractError("production delivery audit requires a RunBundle")
    parts = _relative_parts(relative)
    try:
        run._require_path_identity()
        retained = getattr(run, "_directory_fd", -1)
        if type(retained) is not int or retained < 0:
            raise ContractError("delivery run descriptor is unavailable")
        root_fd = os.dup(retained)
    except OSError as error:
        raise ContractError("cannot retain delivery run evidence") from error
    descriptor = root_fd
    try:
        for part in parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            if descriptor != root_fd:
                os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(
            parts[-1],
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=descriptor,
        )
        try:
            metadata = os.fstat(file_descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ContractError("delivery archive must be one regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_descriptor, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(file_descriptor)
    except OSError as error:
        raise ContractError("cannot read confined delivery evidence") from error
    finally:
        if descriptor != root_fd:
            os.close(descriptor)
        os.close(root_fd)


def _read_durable_readiness(
    evidence: DeliveryAuditEvidence,
    transcript: _EvidenceTranscript,
) -> Mapping[str, object]:
    if not isinstance(evidence.run, RunBundle):
        raise ContractError("production delivery audit requires a RunBundle")
    raw = _read_regular_confined(evidence.run, "readiness.jsonl")
    transcript.add("readiness.durable_jsonl", raw)
    if not raw.endswith(b"\n") or raw.endswith(b"\r\n") or raw.count(b"\n") != 1:
        raise ContractError("delivery audit requires one durable readiness record")

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ContractError("durable readiness record contains duplicate keys")
            output[key] = value
        return output

    try:
        envelope = json.loads(
            raw[:-1],
            object_pairs_hook=no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContractError(f"invalid durable readiness constant: {value}")
            ),
        )
        canonical = (
            json.dumps(
                envelope,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (UnicodeDecodeError, json.JSONDecodeError, UnicodeEncodeError) as error:
        raise ContractError("durable readiness record is invalid JSON") from error
    if (
        type(envelope) is not dict
        or set(envelope) != {"kind", "sequence", "recorded_utc", "record"}
        or envelope["kind"] != "readiness"
        or type(envelope["sequence"]) is not int
        or envelope["sequence"] <= 0
        or type(envelope["recorded_utc"]) is not str
        or not envelope["recorded_utc"]
        or type(envelope["record"]) is not dict
        or canonical != raw
    ):
        raise ContractError("durable readiness record has an invalid envelope")
    return envelope["record"]


def _read_publication(
    evidence: DeliveryAuditEvidence,
    publication: Mapping[str, object],
    phase: str,
    transcript: _EvidenceTranscript,
    transcript_label: str,
    archive_paths: set[str],
) -> DecodedPoseV1:
    expected_keys = {
        "first_frame_index",
        "last_frame_index",
        "phase",
        "attempt",
        "message_path",
        "digest_path",
        "sha256",
        "size",
        "local_send_completed",
    }
    record = dict(publication)
    transcript.add_json(f"{transcript_label}.metadata", record)
    if set(record) != expected_keys or record["local_send_completed"] is not True:
        raise ContractError("delivery publication metadata is incomplete")
    if record["phase"] != phase:
        raise ContractError("delivery publication phase changed")
    if phase == "timeline" and record["attempt"] is not None:
        raise ContractError("timeline delivery publication has a readiness attempt")
    if phase == "readiness" and (
        type(record["attempt"]) is not int or record["attempt"] <= 0
    ):
        raise ContractError("readiness delivery publication has an invalid attempt")
    if (
        type(record["first_frame_index"]) is not int
        or type(record["last_frame_index"]) is not int
        or record["first_frame_index"] < 0
        or record["last_frame_index"] < record["first_frame_index"]
    ):
        raise ContractError("delivery publication frame range is invalid")
    stem = (
        f"{record['first_frame_index']:06d}-{record['last_frame_index']:06d}"
    )
    if phase == "readiness":
        leaf_stem = f"attempt-{record['attempt']:06d}__{stem}"
        directory = "transmitted/readiness"
    else:
        leaf_stem = stem
        directory = "transmitted"
    expected_message_path = f"{directory}/{leaf_stem}.bin"
    expected_digest_path = f"{directory}/{leaf_stem}.sha256"
    if (
        record["message_path"] != expected_message_path
        or record["digest_path"] != expected_digest_path
    ):
        raise ContractError("delivery publication archive path changed")
    for path in (expected_message_path, expected_digest_path):
        if path in archive_paths:
            raise ContractError("delivery publication archive path is duplicated")
        archive_paths.add(path)
    if not isinstance(evidence.run, RunBundle):
        raise ContractError("production delivery audit requires a RunBundle")
    message = _read_regular_confined(evidence.run, record["message_path"])
    transcript.add(f"{transcript_label}.message", message)
    digest = hashlib.sha256(message).hexdigest()
    if (
        type(record["sha256"]) is not str
        or record["sha256"] != digest
        or type(record["size"]) is not int
        or record["size"] != len(message)
    ):
        raise ContractError("archived delivery digest or size changed")
    sidecar = _read_regular_confined(evidence.run, record["digest_path"])
    transcript.add(f"{transcript_label}.sidecar", sidecar)
    expected_sidecar = f"{digest}  {leaf_stem}.bin\n".encode("ascii")
    if sidecar != expected_sidecar:
        raise ContractError("archived delivery digest sidecar changed")
    decoded = decode_pose_v1(message, baseline_mode=True)
    if (
        type(record["first_frame_index"]) is not int
        or type(record["last_frame_index"]) is not int
        or record["first_frame_index"] != int(decoded.frame_index[0])
        or record["last_frame_index"] != int(decoded.frame_index[-1])
    ):
        raise ContractError("archived delivery frame range changed")
    return decoded


def _decoded_equals_canonical(
    decoded: DecodedPoseV1,
    canonical: CanonicalTargetBuffer,
    indices: np.ndarray,
) -> bool:
    if not np.array_equal(decoded.frame_index, indices):
        return False
    if np.any(indices < 0) or np.any(indices >= canonical.count):
        return False
    comparisons = (
        (decoded.joint_position, canonical.joint_position[indices], "<f4"),
        (decoded.joint_velocity, canonical.joint_velocity[indices], "<f4"),
        (decoded.body_quat_w, canonical.body_quat_w[indices], "<f4"),
    )
    return all(
        np.asarray(actual, dtype=dtype).tobytes(order="C")
        == np.asarray(expected, dtype=dtype).tobytes(order="C")
        for actual, expected, dtype in comparisons
    )


class ProductionDeliveryAuditor:
    """Recompute terminal delivery evidence from authenticated immutable inputs."""

    def __init__(
        self, verified_gear: VerifiedGearCheckout | VerifiedExternal
    ) -> None:
        if not isinstance(verified_gear, (VerifiedGearCheckout, VerifiedExternal)):
            raise ContractError("delivery auditor requires verified GEAR inputs")
        if verified_gear.gear_commit != PINNED_GEAR_COMMIT:
            raise ContractError("delivery auditor GEAR commit is not pinned")
        digest = verified_gear.hashes.get("gear:current_frame_advancement_source")
        if digest != PINNED_CURRENT_FRAME_ADVANCEMENT_SHA256:
            raise ContractError(
                "delivery auditor requires the pinned CurrentFrameAdvancement "
                "source hash"
            )
        self._current_frame_advancement_sha256 = digest
        self._gear_commit = verified_gear.gear_commit

    def audit(self, evidence: DeliveryAuditEvidence) -> DeliveryAudit:
        if not isinstance(evidence, DeliveryAuditEvidence):
            raise ContractError("delivery auditor requires immutable evidence")
        canonical = evidence.canonical_buffer
        canonical_bytes = _canonical_bytes(canonical)
        transcript = _EvidenceTranscript()
        transcript.add_json(
            "auditor.identity",
            {
                "gear_commit": self._gear_commit,
                "current_frame_advancement_sha256": (
                    self._current_frame_advancement_sha256
                ),
            },
        )
        transcript.add("session_id", evidence.session_id.encode("utf-8"))
        transcript.add_json(
            "canonical.layout",
            {
                "count": canonical.count,
                "joint_position_shape": list(canonical.joint_position.shape),
                "joint_velocity_shape": list(canonical.joint_velocity.shape),
                "body_quat_w_shape": list(canonical.body_quat_w.shape),
                "frame_index_shape": list(canonical.frame_index.shape),
                "float_dtype": "little-endian-f32",
                "index_dtype": "little-endian-i64",
            },
        )
        transcript.add("canonical.bytes", canonical_bytes)
        transcript.add("official_log_slice", evidence.official_log_slice)

        expected_readiness = {
            "session_id": evidence.readiness.session_id,
            "official_log_path": evidence.readiness.official_log_path,
            "log_identity": [
                evidence.readiness.log_device,
                evidence.readiness.log_inode,
            ],
            "readiness_log_start_offset": evidence.readiness.readiness_log_start_offset,
            "scoring_log_offset": evidence.readiness.scoring_log_offset,
            "readiness_attempts": evidence.readiness.readiness_attempts,
            "expected_row_sha256": evidence.readiness.expected_row_sha256,
        }
        transcript.add_json("readiness.evidence", expected_readiness)
        try:
            durable_readiness = dict(_read_durable_readiness(evidence, transcript))
            readiness_exact = (
                durable_readiness == expected_readiness
                and evidence.readiness.session_id == evidence.session_id
                and evidence.readiness.expected_row_sha256
                == hashlib.sha256(_official_row(canonical, 0)).hexdigest()
                and evidence.readiness.scoring_log_offset
                >= evidence.readiness.readiness_log_start_offset
            )
        except ContractError:
            readiness_exact = False

        transmitted_exact = readiness_exact
        decoded_indices: list[int] = []
        archive_paths: set[str] = set()
        transcript.add_json(
            "publication.counts",
            {
                "readiness": len(evidence.readiness_publications),
                "timeline": len(evidence.timeline_publications),
            },
        )
        if (
            len(evidence.readiness_publications)
            != evidence.readiness.readiness_attempts
        ):
            transmitted_exact = False
        for expected_attempt, publication in enumerate(
            evidence.readiness_publications, start=1
        ):
            try:
                decoded = _read_publication(
                    evidence,
                    publication,
                    "readiness",
                    transcript,
                    f"publication.readiness.{expected_attempt:06d}",
                    archive_paths,
                )
                if publication["attempt"] != expected_attempt:
                    transmitted_exact = False
                indices = np.array([0], dtype=np.int64)
                if not _decoded_equals_canonical(decoded, canonical, indices):
                    transmitted_exact = False
            except (ContractError, KeyError, IndexError, TypeError, ValueError):
                transmitted_exact = False
        decoded_indices.append(0)
        for publication_index, publication in enumerate(
            evidence.timeline_publications, start=1
        ):
            try:
                decoded = _read_publication(
                    evidence,
                    publication,
                    "timeline",
                    transcript,
                    f"publication.timeline.{publication_index:06d}",
                    archive_paths,
                )
                indices = np.asarray(decoded.frame_index, dtype=np.int64)
                decoded_indices.extend(int(value) for value in indices)
                if not _decoded_equals_canonical(decoded, canonical, indices):
                    transmitted_exact = False
            except (ContractError, KeyError, IndexError, TypeError, ValueError):
                transmitted_exact = False
        if decoded_indices != list(range(canonical.count)):
            transmitted_exact = False

        official_exact = readiness_exact
        official_rows = evidence.official_log_slice.splitlines(keepends=True)
        observed_rows = 1 + len(official_rows)
        if (
            not evidence.official_log_slice.endswith(b"\n")
            and evidence.official_log_slice
        ):
            official_exact = False
        if len(official_rows) != canonical.count - 1:
            official_exact = False
        for cursor, row in enumerate(official_rows, start=1):
            try:
                parsed = parse_official_target_row(row)
            except ContractError:
                official_exact = False
                continue
            if cursor >= canonical.count or parsed != _official_row(canonical, cursor):
                official_exact = False

        return DeliveryAudit(
            expected_rows=canonical.count,
            observed_rows=observed_rows,
            transmitted_indices_exact=transmitted_exact,
            official_rows_exact=official_exact,
            evidence_sha256=transcript.hexdigest(),
        )
