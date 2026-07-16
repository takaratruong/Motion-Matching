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


def _parse_command(value: object) -> Mapping[str, object]:
    fields = {
        "requested_velocity_holden",
        "desired_heading_holden_wxyz",
        "applied_velocity_holden",
    }
    source = _exact_object(value, fields, "command")
    output = {
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
            (_STEP_COUNT, 3),
            "command.applied_velocity_holden",
        ),
    }
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
    if source["schema"] != "mm-chunk/v1" or type(source["schema"]) is not str:
        raise ContractError("source chunk.schema must equal mm-chunk/v1")
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
        (_BOUNDARY_COUNT,),
        "source chunk.timestamps_s",
    )
    if np.any(np.diff(timestamps.astype(np.float64)) <= 0.0):
        raise ContractError("source chunk.timestamps_s must increase strictly")
    expected_timestamps = np.array(
        [np.float32(index) / np.float32(_SOURCE_RATE_HZ) for index in range(11)],
        dtype=np.float32,
    )
    if not np.array_equal(
        timestamps.view(np.uint32), expected_timestamps.view(np.uint32)
    ):
        raise ContractError("source chunk.timestamps_s is not the exact 25 Hz grid")
    physical_orientation = _unit_quaternions(
        _float_array(
            source["physical_pelvis_orientation_holden"],
            (_BOUNDARY_COUNT, 4),
            "source chunk.physical_pelvis_orientation_holden",
        ),
        "source chunk.physical_pelvis_orientation_holden",
    )
    virtual_orientation = _unit_quaternions(
        _float_array(
            source["virtual_root_orientation_holden"],
            (_BOUNDARY_COUNT, 4),
            "source chunk.virtual_root_orientation_holden",
        ),
        "source chunk.virtual_root_orientation_holden",
    )
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
        source_intervals=_fixed_integer(
            source["source_intervals"], _STEP_COUNT, "source chunk.source_intervals"
        ),
        timestamps_s=timestamps,
        source_joint_names=source_names,
        target_joint_names=target_names,
        joint_position_source=_float_array(
            source["joint_position_source"],
            (_BOUNDARY_COUNT, _JOINT_COUNT),
            "source chunk.joint_position_source",
        ),
        joint_velocity_source=_float_array(
            source["joint_velocity_source"],
            (_BOUNDARY_COUNT, _JOINT_COUNT),
            "source chunk.joint_velocity_source",
        ),
        physical_pelvis_position_holden=_float_array(
            source["physical_pelvis_position_holden"],
            (_BOUNDARY_COUNT, 3),
            "source chunk.physical_pelvis_position_holden",
        ),
        physical_pelvis_orientation_holden=physical_orientation,
        virtual_root_position_holden=_float_array(
            source["virtual_root_position_holden"],
            (_BOUNDARY_COUNT, 3),
            "source chunk.virtual_root_position_holden",
        ),
        virtual_root_orientation_holden=virtual_orientation,
        selected_database_frame=_integer_array(
            source["selected_database_frame"],
            _STEP_COUNT,
            "source chunk.selected_database_frame",
        ),
        searched=_boolean_array(
            source["searched"], _STEP_COUNT, "source chunk.searched"
        ),
        transitioned=_boolean_array(
            source["transitioned"], _STEP_COUNT, "source chunk.transitioned"
        ),
        terrain_cost=_float_array(
            source["terrain_cost"], (_STEP_COUNT,), "source chunk.terrain_cost"
        ),
        terrain_values=_float_array(
            source["terrain_values"],
            (_STEP_COUNT, _TERRAIN_SAMPLE_COUNT),
            "source chunk.terrain_values",
        ),
        terrain_points_holden=_float_array(
            source["terrain_points_holden"],
            (_STEP_COUNT, _TERRAIN_SAMPLE_COUNT, 3),
            "source chunk.terrain_points_holden",
        ),
        support_height=_float_array(
            source["support_height"],
            (_STEP_COUNT,),
            "source chunk.support_height",
        ),
        support_target=_float_array(
            source["support_target"],
            (_STEP_COUNT,),
            "source chunk.support_target",
        ),
        scene=scene,
        command=_parse_command(source["command"]),
        artifacts=artifacts,
    )
