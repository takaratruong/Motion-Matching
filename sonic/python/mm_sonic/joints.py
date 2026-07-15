"""Certified name-driven G1 joint projection contract generation."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

TARGET_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

PINNED_TARGET_TO_SOURCE_PERMUTATION = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)

_BODY_TO_HOLDEN = {
    "pelvis": "Hips",
    "left_hip_pitch_link": "LeftHipPitch",
    "left_hip_roll_link": "LeftHipRoll",
    "left_hip_yaw_link": "LeftHipYaw",
    "left_knee_link": "LeftKnee",
    "left_ankle_pitch_link": "LeftAnkle",
    "left_ankle_roll_link": "LeftToe",
    "right_hip_pitch_link": "RightHipPitch",
    "right_hip_roll_link": "RightHipRoll",
    "right_hip_yaw_link": "RightHipYaw",
    "right_knee_link": "RightKnee",
    "right_ankle_pitch_link": "RightAnkle",
    "right_ankle_roll_link": "RightToe",
    "waist_yaw_link": "Spine",
    "waist_roll_link": "Spine1",
    "torso_link": "Spine2",
    "left_shoulder_pitch_link": "LeftShoulderPitch",
    "left_shoulder_roll_link": "LeftShoulderRoll",
    "left_shoulder_yaw_link": "LeftShoulderYaw",
    "left_elbow_link": "LeftElbow",
    "left_wrist_roll_link": "LeftWristRoll",
    "left_wrist_pitch_link": "LeftWristPitch",
    "left_wrist_yaw_link": "LeftWrist",
    "right_shoulder_pitch_link": "RightShoulderPitch",
    "right_shoulder_roll_link": "RightShoulderRoll",
    "right_shoulder_yaw_link": "RightShoulderYaw",
    "right_elbow_link": "RightElbow",
    "right_wrist_roll_link": "RightWristRoll",
    "right_wrist_pitch_link": "RightWristPitch",
    "right_wrist_yaw_link": "RightWrist",
}

_Q_ZUP_TO_HOLDEN = np.array(
    [2.0**-0.5, -(2.0**-0.5), 0.0, 0.0],
    np.float64,
)
_JOINT_COUNT = 29
_AXIS_PERTURBATION = 1.0e-4


class ContractError(ValueError):
    """A joint contract or one of its named sources is invalid."""


@dataclass(frozen=True)
class JointRow:
    source_index: int
    source_bone: str
    source_parent: str
    source_joint: str
    qpos_address: int
    axis_holden: tuple[float, float, float]
    static_local_holden_wxyz: tuple[float, float, float, float]
    sign: float
    zero_offset: float
    lower: float
    upper: float
    target_name: str
    target_index: int


@dataclass(frozen=True)
class JointContract:
    source_mjcf_sha256: str
    target_order_source_sha256: str
    rows: tuple[JointRow, ...]


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _read_file(path: Path | str, label: str) -> tuple[Path, bytes]:
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ContractError(f"{label} does not resolve: {candidate}") from error
    if not resolved.is_file():
        raise ContractError(f"{label} is not a file: {resolved}")
    try:
        return resolved, resolved.read_bytes()
    except OSError as error:
        raise ContractError(f"cannot read {label}: {resolved}") from error


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            rw * lw - rx * lx - ry * ly - rz * lz,
            rw * lx + rx * lw - ry * lz + rz * ly,
            rw * ly + rx * lz + ry * lw - rz * lx,
            rw * lz - rx * ly + ry * lx + rz * lw,
        ),
        axis=-1,
    )


def _quat_inverse(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value, np.float64).copy()
    output[..., 1:] *= -1.0
    return output


def _quat_rotate(rotation: np.ndarray, vector: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, np.float64)
    vector = np.asarray(vector, np.float64)
    xyz = rotation[..., 1:]
    first = 2.0 * np.cross(xyz, vector)
    return vector + rotation[..., :1] * first + np.cross(xyz, first)


def _normalized_vector(value: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(value, np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ContractError(f"{label} is not a finite three-vector")
    magnitude = float(np.linalg.norm(value))
    if magnitude < 1.0e-12:
        raise ContractError(f"{label} has zero magnitude")
    output = value / magnitude
    output[np.abs(output) < 1.0e-10] = 0.0
    output /= np.linalg.norm(output)
    return output


def _canonical_quaternion(value: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(value, np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ContractError(f"{label} is not a finite quaternion")
    magnitude = float(np.linalg.norm(value))
    if magnitude < 1.0e-12:
        raise ContractError(f"{label} has zero magnitude")
    output = value / magnitude
    output[np.abs(output) < 1.0e-15] = 0.0
    for component in output:
        if component != 0.0:
            if component < 0.0:
                output = -output
            break
    return output


def _quaternion_angle(left: np.ndarray, right: np.ndarray) -> float:
    delta = _quat_multiply(_quat_inverse(left), right)
    if delta[0] < 0.0:
        delta = -delta
    return float(
        2.0
        * math.atan2(
            float(np.linalg.norm(delta[1:])),
            abs(float(delta[0])),
        )
    )


def _scaled_angle_axis(value: np.ndarray) -> np.ndarray:
    value = _canonical_quaternion(value, "perturbed joint delta")
    vector_length = float(np.linalg.norm(value[1:]))
    if vector_length < 1.0e-15:
        return np.zeros(3, np.float64)
    angle = 2.0 * math.atan2(vector_length, float(value[0]))
    return value[1:] * (angle / vector_length)


def _basis_converted_global_quaternions(data: Any) -> np.ndarray:
    source = np.asarray(data.xquat, np.float64)
    basis = np.broadcast_to(_Q_ZUP_TO_HOLDEN, source.shape)
    inverse = np.broadcast_to(_quat_inverse(_Q_ZUP_TO_HOLDEN), source.shape)
    return _quat_multiply(_quat_multiply(basis, source), inverse)


def _local_quaternion(model: Any, globals_holden: np.ndarray, body_id: int) -> np.ndarray:
    parent_id = int(model.body_parentid[body_id])
    if parent_id == 0:
        return globals_holden[body_id]
    return _quat_multiply(
        _quat_inverse(globals_holden[parent_id]),
        globals_holden[body_id],
    )


def _ast_assignment(module: ast.Module, name: str) -> Any:
    matches = []
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in statement.targets
            ):
                matches.append(statement.value)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == name
        ):
            matches.append(statement.value)
    if len(matches) != 1:
        raise ContractError(
            f"target order source must define {name} exactly once"
        )
    try:
        return ast.literal_eval(matches[0])
    except (TypeError, ValueError) as error:
        raise ContractError(
            f"target order source {name} must be a literal"
        ) from error


def _target_body_order_and_permutation(
    contents: bytes,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    try:
        text = contents.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ContractError("target order source is not valid UTF-8 Python") from error
    body_values = _ast_assignment(module, "G1_ISAACLAB_JOINTS")
    permutation_values = _ast_assignment(
        module, "G1_MUJOCO_TO_ISAACLAB_DOF"
    )
    if (
        type(body_values) is not list
        or len(body_values) != _JOINT_COUNT + 1
        or any(type(value) is not str or not value for value in body_values)
        or body_values[0] != "pelvis"
        or len(set(body_values)) != len(body_values)
    ):
        raise ContractError("target body order is not the exact 30-name G1 order")
    if (
        type(permutation_values) is not list
        or any(type(value) is not int for value in permutation_values)
    ):
        raise ContractError("target permutation must be an integer list")
    permutation = tuple(permutation_values)
    if permutation != PINNED_TARGET_TO_SOURCE_PERMUTATION:
        raise ContractError("target permutation does not match the pinned G1 order")
    return tuple(body_values[1:]), permutation


def _source_model_contract(model: Any, mujoco_module: Any) -> tuple[int, ...]:
    hinge = tuple(
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id])
        == int(mujoco_module.mjtJoint.mjJNT_HINGE)
    )
    names = tuple(model.joint(joint_id).name for joint_id in hinge)
    if len(hinge) != _JOINT_COUNT or names != SOURCE_JOINT_ORDER:
        raise ContractError(
            "source MJCF hinge order must be the exact named 29-joint G1 order"
        )
    if len(set(names)) != _JOINT_COUNT:
        raise ContractError("source MJCF hinge names are not unique")
    non_hinge = [
        joint_id
        for joint_id in range(model.njnt)
        if joint_id not in hinge
    ]
    if (
        len(non_hinge) != 1
        or int(model.jnt_type[non_hinge[0]])
        != int(mujoco_module.mjtJoint.mjJNT_FREE)
    ):
        raise ContractError("source MJCF must have one free base plus 29 hinges")
    return hinge


def generate_joint_contract(
    source_mjcf: Path | str,
    target_order_source: Path | str,
) -> JointContract:
    """Generate the named contract from MuJoCo FK and a pinned GEAR order file."""

    source_path, source_contents = _read_file(source_mjcf, "source_mjcf")
    _, target_contents = _read_file(
        target_order_source, "target_order_source"
    )
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(source_path))
    except (ImportError, ValueError) as error:
        raise ContractError(f"cannot load source MJCF: {source_path}") from error
    hinge_ids = _source_model_contract(model, mujoco)
    target_bodies, pinned_permutation = _target_body_order_and_permutation(
        target_contents
    )

    joint_for_body: dict[str, str] = {}
    for joint_id in hinge_ids:
        body_id = int(model.jnt_bodyid[joint_id])
        body_name = model.body(body_id).name
        if body_name in joint_for_body:
            raise ContractError(f"multiple hinges are attached to {body_name}")
        joint_for_body[body_name] = model.joint(joint_id).name
    try:
        target_names = tuple(joint_for_body[name] for name in target_bodies)
    except KeyError as error:
        raise ContractError(
            f"target body has no named source hinge: {error.args[0]}"
        ) from error
    if target_names != TARGET_JOINT_ORDER:
        raise ContractError("target body names do not resolve to the pinned joint order")
    if tuple(SOURCE_JOINT_ORDER[index] for index in pinned_permutation) != target_names:
        raise ContractError("name-derived target order disagrees with pinned permutation")
    target_index_by_name = {
        name: index for index, name in enumerate(target_names)
    }

    zero_qpos = np.asarray(model.qpos0, np.float64).copy()
    for joint_id in hinge_ids:
        address = int(model.jnt_qposadr[joint_id])
        if zero_qpos[address] != 0.0:
            raise ContractError(
                f"source joint {model.joint(joint_id).name} has nonzero semantics"
            )
        if not bool(model.jnt_limited[joint_id]):
            raise ContractError(
                f"source joint {model.joint(joint_id).name} is not range limited"
            )

    data = mujoco.MjData(model)
    data.qpos[:] = zero_qpos
    mujoco.mj_forward(model, data)
    zero_globals = _basis_converted_global_quaternions(data)
    rows = []
    for source_index, joint_id in enumerate(hinge_ids):
        joint_name = model.joint(joint_id).name
        body_id = int(model.jnt_bodyid[joint_id])
        body_name = model.body(body_id).name
        parent_id = int(model.body_parentid[body_id])
        parent_name = model.body(parent_id).name
        if body_name not in _BODY_TO_HOLDEN or parent_name not in _BODY_TO_HOLDEN:
            raise ContractError(
                f"source joint {joint_name} has unmapped body hierarchy"
            )
        static_raw = _canonical_quaternion(
            _local_quaternion(model, zero_globals, body_id),
            f"{joint_name} static rotation",
        )

        perturbed_qpos = zero_qpos.copy()
        qpos_address = int(model.jnt_qposadr[joint_id])
        perturbed_qpos[qpos_address] += _AXIS_PERTURBATION
        data.qpos[:] = perturbed_qpos
        mujoco.mj_forward(model, data)
        perturbed_globals = _basis_converted_global_quaternions(data)
        perturbed_local = _canonical_quaternion(
            _local_quaternion(model, perturbed_globals, body_id),
            f"{joint_name} perturbed rotation",
        )
        delta = _quat_multiply(_quat_inverse(static_raw), perturbed_local)
        inferred_axis = _normalized_vector(
            _scaled_angle_axis(delta) / _AXIS_PERTURBATION,
            f"{joint_name} inferred axis",
        )
        expected_axis = _normalized_vector(
            _quat_rotate(_Q_ZUP_TO_HOLDEN, model.jnt_axis[joint_id]),
            f"{joint_name} MJCF axis",
        )
        if float(np.dot(inferred_axis, expected_axis)) < 1.0 - 1.0e-10:
            raise ContractError(
                f"{joint_name} perturbed Holden axis disagrees with MJCF axis"
            )
        half = 0.5 * _AXIS_PERTURBATION
        twist = np.concatenate(
            ([math.cos(half)], math.sin(half) * inferred_axis)
        )
        reconstruction = _quat_multiply(static_raw, twist)
        if _quaternion_angle(reconstruction, perturbed_local) > 1.0e-10:
            raise ContractError(
                f"{joint_name} perturbation does not reconstruct locally"
            )

        lower, upper = (
            float(value) for value in model.jnt_range[joint_id]
        )
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ContractError(f"{joint_name} has an invalid hinge range")
        rows.append(
            JointRow(
                source_index=source_index,
                source_bone=_BODY_TO_HOLDEN[body_name],
                source_parent=_BODY_TO_HOLDEN[parent_name],
                source_joint=joint_name,
                qpos_address=qpos_address,
                axis_holden=tuple(float(value) for value in inferred_axis),
                static_local_holden_wxyz=tuple(
                    float(value) for value in static_raw
                ),
                sign=1.0,
                zero_offset=0.0,
                lower=lower,
                upper=upper,
                target_name=joint_name,
                target_index=target_index_by_name[joint_name],
            )
        )

    contract = JointContract(
        source_mjcf_sha256=_sha256_bytes(source_contents),
        target_order_source_sha256=_sha256_bytes(target_contents),
        rows=tuple(rows),
    )
    _validate_contract(contract)
    return contract


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ContractError(f"{label} must be a finite number")
    return converted


def _validate_hash(value: str, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} must be a lowercase SHA-256")


def _validate_contract(contract: JointContract) -> None:
    if not isinstance(contract, JointContract):
        raise ContractError("contract must be a JointContract")
    _validate_hash(contract.source_mjcf_sha256, "source_mjcf_sha256")
    _validate_hash(
        contract.target_order_source_sha256,
        "target_order_source_sha256",
    )
    if len(contract.rows) != _JOINT_COUNT:
        raise ContractError(f"expected 29 contract rows, got {len(contract.rows)}")
    if tuple(row.source_index for row in contract.rows) != tuple(range(_JOINT_COUNT)):
        raise ContractError("contract rows are not in source-index order")
    if tuple(row.source_joint for row in contract.rows) != SOURCE_JOINT_ORDER:
        raise ContractError("contract source names do not match the G1 source order")
    if set(row.target_index for row in contract.rows) != set(range(_JOINT_COUNT)):
        raise ContractError("contract target indices are not a permutation")
    ordered_target = tuple(
        next(row.target_name for row in contract.rows if row.target_index == index)
        for index in range(_JOINT_COUNT)
    )
    if ordered_target != TARGET_JOINT_ORDER:
        raise ContractError("contract target names do not match the G1 target order")
    if len({row.qpos_address for row in contract.rows}) != _JOINT_COUNT:
        raise ContractError("contract qpos addresses are not unique")
    for row in contract.rows:
        if not isinstance(row, JointRow):
            raise ContractError("contract rows must be JointRow values")
        if (
            type(row.source_index) is not int
            or type(row.qpos_address) is not int
            or type(row.target_index) is not int
            or row.qpos_address < 0
        ):
            raise ContractError("contract row indices must be nonnegative integers")
        for value, label in (
            (row.source_bone, "source_bone"),
            (row.source_parent, "source_parent"),
            (row.source_joint, "source_joint"),
            (row.target_name, "target_name"),
        ):
            if type(value) is not str or not value:
                raise ContractError(f"{label} must be a nonempty string")
        if row.target_name != row.source_joint:
            raise ContractError("target mapping must be name driven")
        axis = np.asarray(row.axis_holden, np.float64)
        static = np.asarray(row.static_local_holden_wxyz, np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ContractError("axis_holden must contain three finite values")
        if static.shape != (4,) or not np.all(np.isfinite(static)):
            raise ContractError(
                "static_local_holden_wxyz must contain four finite values"
            )
        if abs(float(np.linalg.norm(axis)) - 1.0) > 1.0e-9:
            raise ContractError("axis_holden must be normalized")
        if abs(float(np.linalg.norm(static)) - 1.0) > 1.0e-9:
            raise ContractError("static_local_holden_wxyz must be normalized")
        first_nonzero = next((value for value in static if value != 0.0), None)
        if first_nonzero is None or first_nonzero < 0.0:
            raise ContractError("static quaternion sign is not canonical")
        sign = _finite_number(row.sign, "sign")
        zero = _finite_number(row.zero_offset, "zero_offset")
        lower = _finite_number(row.lower, "lower")
        upper = _finite_number(row.upper, "upper")
        if sign not in (-1.0, 1.0):
            raise ContractError("sign must be exactly -1 or 1")
        if lower > zero or zero > upper or lower >= upper:
            raise ContractError("zero offset and range are inconsistent")


def contract_json_bytes(contract: JointContract) -> bytes:
    """Encode a contract canonically with sorted compact JSON and one newline."""

    _validate_contract(contract)
    payload = {
        "source_mjcf_sha256": contract.source_mjcf_sha256,
        "target_order_source_sha256": contract.target_order_source_sha256,
        "rows": [asdict(row) for row in contract.rows],
    }
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ContractError("contract cannot be encoded as canonical JSON") from error
    return (text + "\n").encode("utf-8")


def write_joint_contract(contract: JointContract, path: Path | str) -> None:
    """Write one canonical contract file."""

    try:
        Path(path).write_bytes(contract_json_bytes(contract))
    except OSError as error:
        raise ContractError(f"cannot write joint contract: {path}") from error


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ContractError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ContractError(f"{label} keys are invalid")
    return value


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ContractError(f"{label} must be an integer")
    return value


def _number_tuple(value: Any, size: int, label: str) -> tuple[float, ...]:
    if type(value) is not list or len(value) != size:
        raise ContractError(f"{label} must contain {size} numbers")
    return tuple(_finite_number(item, label) for item in value)


def load_joint_contract(path: Path | str) -> JointContract:
    """Load and validate a canonical named joint contract."""

    _, contents = _read_file(path, "joint_contract")
    try:
        payload = json.loads(
            contents.decode("utf-8"), object_pairs_hook=_object_without_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("joint contract is not valid UTF-8 JSON") from error
    payload = _exact_keys(
        payload,
        {"source_mjcf_sha256", "target_order_source_sha256", "rows"},
        "joint contract",
    )
    raw_rows = payload["rows"]
    if type(raw_rows) is not list:
        raise ContractError("joint contract rows must be an array")
    row_keys = {
        "source_index",
        "source_bone",
        "source_parent",
        "source_joint",
        "qpos_address",
        "axis_holden",
        "static_local_holden_wxyz",
        "sign",
        "zero_offset",
        "lower",
        "upper",
        "target_name",
        "target_index",
    }
    rows = []
    for index, raw_row in enumerate(raw_rows):
        raw_row = _exact_keys(raw_row, row_keys, f"row {index}")
        for name in ("source_bone", "source_parent", "source_joint", "target_name"):
            if type(raw_row[name]) is not str:
                raise ContractError(f"row {index} {name} must be a string")
        rows.append(
            JointRow(
                source_index=_integer(raw_row["source_index"], "source_index"),
                source_bone=raw_row["source_bone"],
                source_parent=raw_row["source_parent"],
                source_joint=raw_row["source_joint"],
                qpos_address=_integer(raw_row["qpos_address"], "qpos_address"),
                axis_holden=_number_tuple(raw_row["axis_holden"], 3, "axis_holden"),
                static_local_holden_wxyz=_number_tuple(
                    raw_row["static_local_holden_wxyz"],
                    4,
                    "static_local_holden_wxyz",
                ),
                sign=_finite_number(raw_row["sign"], "sign"),
                zero_offset=_finite_number(raw_row["zero_offset"], "zero_offset"),
                lower=_finite_number(raw_row["lower"], "lower"),
                upper=_finite_number(raw_row["upper"], "upper"),
                target_name=raw_row["target_name"],
                target_index=_integer(raw_row["target_index"], "target_index"),
            )
        )
    contract = JointContract(
        source_mjcf_sha256=payload["source_mjcf_sha256"],
        target_order_source_sha256=payload["target_order_source_sha256"],
        rows=tuple(rows),
    )
    _validate_contract(contract)
    return contract


def reorder_source_to_target(
    values: np.ndarray,
    contract: JointContract,
) -> np.ndarray:
    """Reorder the final axis by registered names and target indices."""

    _validate_contract(contract)
    source = np.asarray(values)
    if source.ndim == 0 or source.shape[-1] != _JOINT_COUNT:
        raise ContractError(f"expected 29 source joints, got {source.shape}")
    output = np.empty_like(source)
    for row in contract.rows:
        output[..., row.target_index] = source[..., row.source_index]
    return output
