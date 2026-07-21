"""Fail-closed decoding for transactional motion-matching source data."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .joints import ContractError, JointContract, reorder_source_to_target


_BOUNDARY_COUNT = 11
_STEP_COUNT = 10
_SUPPORTED_SOURCE_INTERVALS = (5, 10)
_JOINT_COUNT = 29
_TERRAIN_SAMPLE_COUNT = 4
_SOURCE_RATE_HZ = 25
_QUATERNION_NORM_TOLERANCE = 1.0e-5
_COORDINATE_SIGNATURE = "holden-y-up-right-handed-forward-plus-z"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class InitialBoundary:
    session_id: str
    source_joint_names: tuple[str, ...]
    joint_position_source: np.ndarray
    joint_velocity_source: np.ndarray
    physical_pelvis_position_holden: np.ndarray
    physical_pelvis_orientation_holden: np.ndarray
    virtual_root_position_holden: np.ndarray
    virtual_root_orientation_holden: np.ndarray


@dataclass(frozen=True)
class SourceChunk:
    session_id: str
    candidate_id: str
    predecessor_id: str | None
    source_rate_hz: int
    source_intervals: int
    timestamps_s: np.ndarray
    source_joint_names: tuple[str, ...]
    target_joint_names: tuple[str, ...]
    joint_position_source: np.ndarray
    joint_velocity_source: np.ndarray
    physical_pelvis_position_holden: np.ndarray
    physical_pelvis_orientation_holden: np.ndarray
    virtual_root_position_holden: np.ndarray
    virtual_root_orientation_holden: np.ndarray
    selected_database_frame: np.ndarray
    candidate_preview_count: np.ndarray
    candidate_limit_rejection_count: np.ndarray
    first_rejected_database_frame: np.ndarray
    first_rejected_joint_index: np.ndarray
    first_rejected_joint_position: np.ndarray
    searched: np.ndarray
    transitioned: np.ndarray
    terrain_cost: np.ndarray
    terrain_values: np.ndarray
    terrain_points_holden: np.ndarray
    support_height: np.ndarray
    support_target: np.ndarray
    scene: Mapping[str, object]
    command: Mapping[str, object]
    artifacts: Mapping[str, str]


@dataclass(frozen=True)
class JointFeasibilityIdentity:
    schema: str
    frame_count: int
    raw_safe_count: int
    raw_unsafe_count: int
    search_safe_count: int
    joint_limit_violation_count: tuple[int, ...]
    mask_sha256: str


def _object_no_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ContractError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def loads_exact(text: str) -> object:
    return json.loads(
        text,
        object_pairs_hook=_object_no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ContractError(f"invalid JSON constant: {value}")),
    )


def _exact_object(
    value: object,
    expected: set[str],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise ContractError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractError(
            f"{label} keys differ: missing={missing}, extra={extra}"
        )
    return value


def _string(value: object, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        qualifier = "nonempty " if nonempty else ""
        raise ContractError(f"{label} must be a {qualifier}string")
    return value


def _fixed_integer(value: object, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise ContractError(f"{label} must equal {expected}")
    return value


def _supported_source_intervals(value: object, label: str) -> int:
    if type(value) is not int or value not in _SUPPORTED_SOURCE_INTERVALS:
        raise ContractError(
            f"{label} must be one of "
            + " or ".join(str(count) for count in _SUPPORTED_SOURCE_INTERVALS)
        )
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractError(f"{label} must be a nonnegative integer")
    return value


def _binary32(value: object, label: str) -> np.float32:
    if type(value) not in (int, float):
        raise ContractError(f"{label} must be a JSON number")
    if type(value) is int:
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                converted = np.float32(value)
        except (OverflowError, ValueError) as error:
            raise ContractError(
                f"{label} is not a finite binary32 number"
            ) from error
        if not np.isfinite(converted):
            raise ContractError(f"{label} does not round-trip through binary32")
        if int(converted) != value:
            raise ContractError(
                f"{label} integer does not recover an exact binary32 value"
            )
        return converted
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ContractError(f"{label} is not a finite binary32 number") from error
    if not math.isfinite(number):
        raise ContractError(f"{label} is not a finite binary32 number")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.float32(number)
    if not np.isfinite(converted):
        raise ContractError(f"{label} does not round-trip through binary32")
    canonical_decimal = float(format(float(converted), ".9g"))
    if float(converted) != number and canonical_decimal != number:
        raise ContractError(
            f"{label} does not recover an exact binary32 value"
        )
    return converted


def _float_array(
    value: object,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    flat: list[np.float32] = []

    def visit(candidate: object, depth: int, path: str) -> None:
        if depth == len(shape):
            flat.append(_binary32(candidate, path))
            return
        if type(candidate) is not list or len(candidate) != shape[depth]:
            raise ContractError(f"{path} must have length {shape[depth]}")
        for index, item in enumerate(candidate):
            visit(item, depth + 1, f"{path}[{index}]")

    visit(value, 0, label)
    output = np.asarray(flat, dtype=np.float32).reshape(shape).copy(order="C")
    output.flags.writeable = False
    return output


def _float64_array(
    value: object,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    flat: list[np.float64] = []

    def visit(candidate: object, depth: int, path: str) -> None:
        if depth == len(shape):
            if type(candidate) not in (int, float):
                raise ContractError(f"{path} must be a JSON number")
            try:
                number = float(candidate)
            except (OverflowError, ValueError) as error:
                raise ContractError(
                    f"{path} must be a finite binary64 number"
                ) from error
            if not math.isfinite(number):
                raise ContractError(f"{path} must be a finite binary64 number")
            if type(candidate) is int and int(number) != candidate:
                raise ContractError(
                    f"{path} integer does not recover an exact binary64 value"
                )
            flat.append(np.float64(number))
            return
        if type(candidate) is not list or len(candidate) != shape[depth]:
            raise ContractError(f"{path} must have length {shape[depth]}")
        for index, item in enumerate(candidate):
            visit(item, depth + 1, f"{path}[{index}]")

    visit(value, 0, label)
    output = np.asarray(flat, dtype=np.float64).reshape(shape).copy(order="C")
    output.flags.writeable = False
    return output


def _integer_array(value: object, size: int, label: str) -> np.ndarray:
    if type(value) is not list or len(value) != size:
        raise ContractError(f"{label} must have length {size}")
    limit = np.iinfo(np.int64).max
    output = np.empty(size, dtype=np.int64)
    for index, item in enumerate(value):
        if type(item) is not int or item < 0 or item > limit:
            raise ContractError(f"{label}[{index}] must be a nonnegative int64")
        output[index] = item
    output.flags.writeable = False
    return output


def _sentinel_integer_array(value: object, size: int, label: str) -> np.ndarray:
    if type(value) is not list or len(value) != size:
        raise ContractError(f"{label} must have length {size}")
    limit = np.iinfo(np.int64).max
    output = np.empty(size, dtype=np.int64)
    for index, item in enumerate(value):
        if type(item) is not int or item < -1 or item > limit:
            raise ContractError(f"{label}[{index}] must be an int64 at least -1")
        output[index] = item
    output.flags.writeable = False
    return output


def _boolean_array(value: object, size: int, label: str) -> np.ndarray:
    if type(value) is not list or len(value) != size:
        raise ContractError(f"{label} must have length {size}")
    output = np.empty(size, dtype=np.bool_)
    for index, item in enumerate(value):
        if type(item) is not bool:
            raise ContractError(f"{label}[{index}] must be a boolean")
        output[index] = item
    output.flags.writeable = False
    return output


def _joint_orders(
    contract: JointContract,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    # The Task 2 public reorder API performs the complete canonical validation.
    reorder_source_to_target(np.zeros(_JOINT_COUNT, dtype=np.float32), contract)
    source = tuple(row.source_joint for row in contract.rows)
    targets: list[str | None] = [None] * _JOINT_COUNT
    for row in contract.rows:
        if row.source_index < 0 or row.source_index >= _JOINT_COUNT:
            raise ContractError("joint contract source indices are invalid")
        if row.target_index < 0 or row.target_index >= _JOINT_COUNT:
            raise ContractError("joint contract target indices are invalid")
        if targets[row.target_index] is not None:
            raise ContractError("joint contract target indices are duplicated")
        targets[row.target_index] = row.target_name
    if len(set(source)) != _JOINT_COUNT or any(value is None for value in targets):
        raise ContractError("joint contract names are incomplete")
    return source, tuple(value for value in targets if value is not None)


def _joint_names(
    value: object,
    expected: tuple[str, ...],
    label: str,
    *,
    exact_order: bool,
) -> tuple[str, ...]:
    if type(value) is not list or len(value) != _JOINT_COUNT:
        raise ContractError(f"{label} must contain {_JOINT_COUNT} names")
    if any(type(item) is not str or not item for item in value):
        raise ContractError(f"{label} must contain nonempty strings")
    output = tuple(value)
    if len(set(output)) != _JOINT_COUNT:
        raise ContractError(f"{label} contains duplicate names")
    if exact_order:
        if output != expected:
            raise ContractError(f"{label} does not match checked target order")
    elif set(output) != set(expected):
        raise ContractError(f"{label} does not cover checked source names")
    return output


def _unit_quaternions(value: np.ndarray, label: str) -> np.ndarray:
    norms = np.linalg.norm(value.astype(np.float64), axis=-1)
    if np.any(np.abs(norms - 1.0) > _QUATERNION_NORM_TOLERANCE):
        raise ContractError(f"{label} contains a non-unit quaternion")
    return value


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256.fullmatch(result) is None:
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return result


def parse_joint_feasibility_identity(
    value: object,
) -> JointFeasibilityIdentity:
    """Validate and own an authenticated G1 database feasibility identity."""

    fields = {
        "schema",
        "frame_count",
        "raw_safe_count",
        "raw_unsafe_count",
        "search_safe_count",
        "joint_limit_violation_count",
        "mask_sha256",
    }
    source = _exact_object(value, fields, "joint feasibility identity")
    schema = source["schema"]
    if (
        type(schema) is not str
        or schema != "g1-joint-feasibility-certificate/v1"
    ):
        raise ContractError(
            "joint feasibility identity.schema must equal "
            "g1-joint-feasibility-certificate/v1"
        )
    frame_count = _nonnegative_integer(
        source["frame_count"], "joint feasibility identity.frame_count"
    )
    raw_safe_count = _nonnegative_integer(
        source["raw_safe_count"],
        "joint feasibility identity.raw_safe_count",
    )
    raw_unsafe_count = _nonnegative_integer(
        source["raw_unsafe_count"],
        "joint feasibility identity.raw_unsafe_count",
    )
    search_safe_count = _nonnegative_integer(
        source["search_safe_count"],
        "joint feasibility identity.search_safe_count",
    )
    if frame_count != raw_safe_count + raw_unsafe_count:
        raise ContractError(
            "joint feasibility identity frame count does not reconcile"
        )
    if search_safe_count <= 0 or search_safe_count > raw_safe_count:
        raise ContractError(
            "joint feasibility identity search-safe count is invalid"
        )
    violation_source = source["joint_limit_violation_count"]
    if type(violation_source) is not list or len(violation_source) != _JOINT_COUNT:
        raise ContractError(
            "joint feasibility identity.joint_limit_violation_count "
            f"must have length {_JOINT_COUNT}"
        )
    violation_count = tuple(
        _nonnegative_integer(
            item,
            "joint feasibility identity.joint_limit_violation_count"
            f"[{index}]",
        )
        for index, item in enumerate(violation_source)
    )
    if sum(violation_count) != raw_unsafe_count:
        raise ContractError(
            "joint feasibility identity joint-limit counts do not reconcile"
        )
    return JointFeasibilityIdentity(
        schema=schema,
        frame_count=frame_count,
        raw_safe_count=raw_safe_count,
        raw_unsafe_count=raw_unsafe_count,
        search_safe_count=search_safe_count,
        joint_limit_violation_count=violation_count,
        mask_sha256=_sha256(
            source["mask_sha256"],
            "joint feasibility identity.mask_sha256",
        ),
    )


def _parse_scene(value: object) -> Mapping[str, object]:
    fields = {
        "scene_id",
        "route_id",
        "terrain_weight",
        "coordinate_signature",
        "heightfield_sha256",
        "mesh_sha256",
        "walkability_sha256",
    }
    source = _exact_object(value, fields, "scene")
    output: dict[str, object] = {
        "scene_id": _string(source["scene_id"], "scene.scene_id"),
        "route_id": _string(source["route_id"], "scene.route_id"),
        "terrain_weight": _binary32(
            source["terrain_weight"], "scene.terrain_weight"
        ),
        "coordinate_signature": _string(
            source["coordinate_signature"], "scene.coordinate_signature"
        ),
        "heightfield_sha256": _sha256(
            source["heightfield_sha256"], "scene.heightfield_sha256"
        ),
        "mesh_sha256": _sha256(source["mesh_sha256"], "scene.mesh_sha256"),
        "walkability_sha256": _sha256(
            source["walkability_sha256"], "scene.walkability_sha256"
        ),
    }
    return MappingProxyType(output)


def _parse_shared_command_fields(
    source: dict[str, object],
    step_count: int,
) -> dict[str, object]:
    return {
        "requested_velocity_holden": _float_array(
            source["requested_velocity_holden"],
            (3,),
            "command.requested_velocity_holden",
        ),
        "desired_heading_holden_wxyz": _unit_quaternions(
            _float_array(
                source["desired_heading_holden_wxyz"],
                (4,),
                "command.desired_heading_holden_wxyz",
            ),
            "command.desired_heading_holden_wxyz",
        ),
        "applied_velocity_holden": _float_array(
            source["applied_velocity_holden"],
            (step_count, 3),
            "command.applied_velocity_holden",
        ),
    }


def _parse_command_v1(value: object, step_count: int) -> Mapping[str, object]:
    fields = {
        "requested_velocity_holden",
        "desired_heading_holden_wxyz",
        "applied_velocity_holden",
    }
    source = _exact_object(value, fields, "command")
    return MappingProxyType(_parse_shared_command_fields(source, step_count))


def _parse_command_v2(value: object, step_count: int) -> Mapping[str, object]:
    fields = {
        "requested_velocity_holden",
        "desired_heading_holden_wxyz",
        "applied_velocity_holden",
        "applied_heading_holden_wxyz",
    }
    source = _exact_object(value, fields, "command")
    output = dict(_parse_shared_command_fields(source, step_count))
    output["applied_heading_holden_wxyz"] = _unit_quaternions(
        _float_array(
            source["applied_heading_holden_wxyz"],
            (step_count, 4),
            "command.applied_heading_holden_wxyz",
        ),
        "command.applied_heading_holden_wxyz",
    )
    return MappingProxyType(output)


def _parse_artifacts(value: object) -> Mapping[str, str]:
    fields = {
        "build_commit",
        "joint_contract_sha256",
        "motion_manifest_sha256",
        "database_sha256",
        "terrain_features_sha256",
        "terrain_support_sha256",
        "scene_index_sha256",
        "skeleton_signature",
        "coordinate_signature",
    }
    source = _exact_object(value, fields, "artifacts")
    output = {
        "build_commit": _string(source["build_commit"], "artifacts.build_commit"),
        "joint_contract_sha256": _sha256(
            source["joint_contract_sha256"], "artifacts.joint_contract_sha256"
        ),
        "motion_manifest_sha256": _sha256(
            source["motion_manifest_sha256"], "artifacts.motion_manifest_sha256"
        ),
        "database_sha256": _sha256(
            source["database_sha256"], "artifacts.database_sha256"
        ),
        "terrain_features_sha256": _sha256(
            source["terrain_features_sha256"],
            "artifacts.terrain_features_sha256",
        ),
        "terrain_support_sha256": _sha256(
            source["terrain_support_sha256"],
            "artifacts.terrain_support_sha256",
        ),
        "scene_index_sha256": _sha256(
            source["scene_index_sha256"], "artifacts.scene_index_sha256"
        ),
        "skeleton_signature": _sha256(
            source["skeleton_signature"], "artifacts.skeleton_signature"
        ),
        "coordinate_signature": _string(
            source["coordinate_signature"], "artifacts.coordinate_signature"
        ),
    }
    return MappingProxyType(output)


def parse_initial_boundary(
    value: object,
    contract: JointContract,
) -> InitialBoundary:
    """Validate one reset boundary and detach it from its decoded JSON tree."""

    fields = {
        "session_id",
        "source_joint_names",
        "joint_position_source",
        "joint_velocity_source",
        "physical_pelvis_position_holden",
        "physical_pelvis_orientation_holden",
        "virtual_root_position_holden",
        "virtual_root_orientation_holden",
    }
    source = _exact_object(value, fields, "initial boundary")
    expected_source, _ = _joint_orders(contract)
    return InitialBoundary(
        session_id=_string(source["session_id"], "initial boundary.session_id"),
        source_joint_names=_joint_names(
            source["source_joint_names"],
            expected_source,
            "initial boundary.source_joint_names",
            exact_order=False,
        ),
        joint_position_source=_float_array(
            source["joint_position_source"],
            (_JOINT_COUNT,),
            "initial boundary.joint_position_source",
        ),
        joint_velocity_source=_float_array(
            source["joint_velocity_source"],
            (_JOINT_COUNT,),
            "initial boundary.joint_velocity_source",
        ),
        physical_pelvis_position_holden=_float_array(
            source["physical_pelvis_position_holden"],
            (3,),
            "initial boundary.physical_pelvis_position_holden",
        ),
        physical_pelvis_orientation_holden=_unit_quaternions(
            _float_array(
                source["physical_pelvis_orientation_holden"],
                (4,),
                "initial boundary.physical_pelvis_orientation_holden",
            ),
            "initial boundary.physical_pelvis_orientation_holden",
        ),
        virtual_root_position_holden=_float_array(
            source["virtual_root_position_holden"],
            (3,),
            "initial boundary.virtual_root_position_holden",
        ),
        virtual_root_orientation_holden=_unit_quaternions(
            _float_array(
                source["virtual_root_orientation_holden"],
                (4,),
                "initial boundary.virtual_root_orientation_holden",
            ),
            "initial boundary.virtual_root_orientation_holden",
        ),
    )


def parse_source_chunk(
    value: object,
    contract: JointContract,
) -> SourceChunk:
    """Validate one mm-chunk/v1 value and return immutable owned data."""

    fields = {
        "schema",
        "session_id",
        "candidate_id",
        "predecessor_id",
        "source_rate_hz",
        "source_intervals",
        "timestamps_s",
        "source_joint_names",
        "target_joint_names",
        "joint_position_source",
        "joint_velocity_source",
        "physical_pelvis_position_holden",
        "physical_pelvis_orientation_holden",
        "virtual_root_position_holden",
        "virtual_root_orientation_holden",
        "selected_database_frame",
        "candidate_preview_count",
        "candidate_limit_rejection_count",
        "first_rejected_database_frame",
        "first_rejected_joint_index",
        "first_rejected_joint_position",
        "searched",
        "transitioned",
        "terrain_cost",
        "terrain_values",
        "terrain_points_holden",
        "support_height",
        "support_target",
        "scene",
        "command",
        "artifacts",
    }
    source = _exact_object(value, fields, "source chunk")
    schema_value = source["schema"]
    if type(schema_value) is not str or schema_value not in (
        "mm-chunk/v1",
        "mm-chunk/v2",
    ):
        raise ContractError(
            "source chunk.schema must equal mm-chunk/v1 or mm-chunk/v2"
        )
    command_parser = (
        _parse_command_v1 if schema_value == "mm-chunk/v1" else _parse_command_v2
    )
    step_count = _supported_source_intervals(
        source["source_intervals"], "source chunk.source_intervals"
    )
    boundary_count = step_count + 1
    expected_source, expected_target = _joint_orders(contract)
    source_names = _joint_names(
        source["source_joint_names"],
        expected_source,
        "source chunk.source_joint_names",
        exact_order=False,
    )
    target_names = _joint_names(
        source["target_joint_names"],
        expected_target,
        "source chunk.target_joint_names",
        exact_order=True,
    )
    predecessor = source["predecessor_id"]
    if predecessor is not None:
        predecessor = _string(
            predecessor, "source chunk.predecessor_id", nonempty=False
        )
    timestamps = _float_array(
        source["timestamps_s"],
        (boundary_count,),
        "source chunk.timestamps_s",
    )
    if np.any(np.diff(timestamps.astype(np.float64)) <= 0.0):
        raise ContractError("source chunk.timestamps_s must increase strictly")
    expected_timestamps = np.array(
        [
            np.float32(index) / np.float32(_SOURCE_RATE_HZ)
            for index in range(boundary_count)
        ],
        dtype=np.float32,
    )
    if not np.array_equal(
        timestamps.view(np.uint32), expected_timestamps.view(np.uint32)
    ):
        raise ContractError("source chunk.timestamps_s is not the exact 25 Hz grid")
    physical_orientation = _unit_quaternions(
        _float_array(
            source["physical_pelvis_orientation_holden"],
            (boundary_count, 4),
            "source chunk.physical_pelvis_orientation_holden",
        ),
        "source chunk.physical_pelvis_orientation_holden",
    )
    virtual_orientation = _unit_quaternions(
        _float_array(
            source["virtual_root_orientation_holden"],
            (boundary_count, 4),
            "source chunk.virtual_root_orientation_holden",
        ),
        "source chunk.virtual_root_orientation_holden",
    )
    selected_database_frame = _integer_array(
        source["selected_database_frame"],
        step_count,
        "source chunk.selected_database_frame",
    )
    candidate_preview_count = _integer_array(
        source["candidate_preview_count"],
        step_count,
        "source chunk.candidate_preview_count",
    )
    candidate_limit_rejection_count = _integer_array(
        source["candidate_limit_rejection_count"],
        step_count,
        "source chunk.candidate_limit_rejection_count",
    )
    first_rejected_database_frame = _sentinel_integer_array(
        source["first_rejected_database_frame"],
        step_count,
        "source chunk.first_rejected_database_frame",
    )
    first_rejected_joint_index = _sentinel_integer_array(
        source["first_rejected_joint_index"],
        step_count,
        "source chunk.first_rejected_joint_index",
    )
    first_rejected_joint_position = _float64_array(
        source["first_rejected_joint_position"],
        (step_count,),
        "source chunk.first_rejected_joint_position",
    )
    for step in range(step_count):
        previews = int(candidate_preview_count[step])
        rejections = int(candidate_limit_rejection_count[step])
        rejected_frame = int(first_rejected_database_frame[step])
        rejected_joint = int(first_rejected_joint_index[step])
        rejected_position = first_rejected_joint_position[step]
        if rejections > previews:
            raise ContractError("candidate preview counts are inconsistent")
        if rejections == 0:
            if (
                rejected_frame != -1
                or rejected_joint != -1
                or int(rejected_position.view(np.uint64)) != 0
            ):
                raise ContractError(
                    "candidate rejection sentinels are inconsistent"
                )
            continue
        if (
            rejected_frame < 0
            or rejected_joint < 0
            or rejected_joint >= _JOINT_COUNT
        ):
            raise ContractError("candidate rejection indices are invalid")
        row = contract.rows[rejected_joint]
        position = float(rejected_position)
        if row.lower <= position <= row.upper:
            raise ContractError(
                "candidate rejected position is not outside its contract"
            )
        if int(selected_database_frame[step]) == rejected_frame:
            raise ContractError("selected frame equals first rejected frame")
    scene = _parse_scene(source["scene"])
    artifacts = _parse_artifacts(source["artifacts"])
    if scene["coordinate_signature"] != artifacts["coordinate_signature"]:
        raise ContractError("scene and artifact coordinate signatures disagree")
    if scene["coordinate_signature"] != _COORDINATE_SIGNATURE:
        raise ContractError(
            "coordinate signature must equal " + _COORDINATE_SIGNATURE
        )
    return SourceChunk(
        session_id=_string(source["session_id"], "source chunk.session_id"),
        candidate_id=_string(source["candidate_id"], "source chunk.candidate_id"),
        predecessor_id=predecessor,
        source_rate_hz=_fixed_integer(
            source["source_rate_hz"], _SOURCE_RATE_HZ, "source chunk.source_rate_hz"
        ),
        source_intervals=step_count,
        timestamps_s=timestamps,
        source_joint_names=source_names,
        target_joint_names=target_names,
        joint_position_source=_float_array(
            source["joint_position_source"],
            (boundary_count, _JOINT_COUNT),
            "source chunk.joint_position_source",
        ),
        joint_velocity_source=_float_array(
            source["joint_velocity_source"],
            (boundary_count, _JOINT_COUNT),
            "source chunk.joint_velocity_source",
        ),
        physical_pelvis_position_holden=_float_array(
            source["physical_pelvis_position_holden"],
            (boundary_count, 3),
            "source chunk.physical_pelvis_position_holden",
        ),
        physical_pelvis_orientation_holden=physical_orientation,
        virtual_root_position_holden=_float_array(
            source["virtual_root_position_holden"],
            (boundary_count, 3),
            "source chunk.virtual_root_position_holden",
        ),
        virtual_root_orientation_holden=virtual_orientation,
        selected_database_frame=selected_database_frame,
        candidate_preview_count=candidate_preview_count,
        candidate_limit_rejection_count=candidate_limit_rejection_count,
        first_rejected_database_frame=first_rejected_database_frame,
        first_rejected_joint_index=first_rejected_joint_index,
        first_rejected_joint_position=first_rejected_joint_position,
        searched=_boolean_array(
            source["searched"], step_count, "source chunk.searched"
        ),
        transitioned=_boolean_array(
            source["transitioned"], step_count, "source chunk.transitioned"
        ),
        terrain_cost=_float_array(
            source["terrain_cost"], (step_count,), "source chunk.terrain_cost"
        ),
        terrain_values=_float_array(
            source["terrain_values"],
            (step_count, _TERRAIN_SAMPLE_COUNT),
            "source chunk.terrain_values",
        ),
        terrain_points_holden=_float_array(
            source["terrain_points_holden"],
            (step_count, _TERRAIN_SAMPLE_COUNT, 3),
            "source chunk.terrain_points_holden",
        ),
        support_height=_float_array(
            source["support_height"],
            (step_count,),
            "source chunk.support_height",
        ),
        support_target=_float_array(
            source["support_target"],
            (step_count,),
            "source chunk.support_target",
        ),
        scene=scene,
        command=command_parser(source["command"], step_count),
        artifacts=artifacts,
    )
