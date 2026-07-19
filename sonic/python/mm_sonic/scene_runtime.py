"""Shared GEAR scene identities and name-driven initial physics state.

This module holds the one shared definition of the pinned GEAR scene/robot
paths and hashes, the analytic scene registry path, and the construction of an
exact name-driven initial MuJoCo state from a validated motion-matching reset
boundary. It converts the physical Holden pelvis pose through the existing
Holden/MuJoCo transforms, maps validated source joints through the joint
contract names, places all fourteen Dex3 hand targets by their public joint
orders, and authenticates every registered dependency it reads.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from pathlib import Path

import numpy as np

from .hands import (
    Dex3HandTargets,
    LEFT_HAND_JOINT_ORDER,
    RIGHT_HAND_JOINT_ORDER,
    validate_hand_targets,
)
from .joints import JointContract
from .schema import InitialBoundary
from .scene import (
    PENETRATION_THRESHOLD_M,
    RegisteredScene,
    SceneError,
    _MAXIMUM_OBJ_BYTES,
    _load_authenticated_scene_model,
    _read_authenticated,
    penetration_exceeds_threshold,
)
from .transform import (
    holden_to_mujoco_quaternions,
    holden_to_mujoco_vectors,
    mujoco_to_holden_vectors,
)


SONIC_ROOT = Path(__file__).resolve().parents[2]
SCENE_REGISTRY_PATH = SONIC_ROOT / "configs/scene_registry.json"
GEAR_SCENE_RELATIVE = Path("gear_sonic_deploy/g1/scene_29dof_with_hand.xml")
GEAR_ROBOT_RELATIVE = Path("gear_sonic_deploy/g1/g1_29dof_with_hand.xml")
GEAR_SCENE_SHA256 = (
    "f8538904eb47cada1bfb2dcdc157099092aa63df4307d7e077b651b16bfb6c74"
)
GEAR_ROBOT_SHA256 = (
    "8b68d8f06674c5c10cd2cd89764b3cfba9fabba5080b55ea67ee1dd12cf630cd"
)

_INITIAL_BOUNDARY_HASH_DOMAIN = b"mm-sonic-initial-boundary/v1\0"
_JOINT_COUNT = 29


@dataclass(frozen=True)
class InitialPhysicsState:
    qpos: np.ndarray
    qpos_sha256: str
    initial_boundary_sha256: str
    maximum_forbidden_penetration_m: float
    pelvis_clearance_m: float
    left_foot_clearance_m: float
    right_foot_clearance_m: float


def _finite_float64(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    source = np.asarray(value)
    if source.shape != shape or source.dtype.kind not in "iuf":
        raise SceneError(f"{label} must have shape {shape}")
    output = np.ascontiguousarray(np.asarray(source, np.float64))
    if not np.all(np.isfinite(output)):
        raise SceneError(f"{label} must contain only finite values")
    return output


def _initial_boundary_sha256(initial: InitialBoundary) -> str:
    """Hash a deterministic canonical encoding of the reset boundary."""

    if not isinstance(initial, InitialBoundary):
        raise SceneError("initial boundary must be an InitialBoundary")
    names = tuple(initial.source_joint_names)
    if (
        len(names) != _JOINT_COUNT
        or any(type(name) is not str or not name for name in names)
        or len(set(names)) != _JOINT_COUNT
    ):
        raise SceneError("initial boundary source joint names are invalid")
    digest = hashlib.sha256()
    digest.update(_INITIAL_BOUNDARY_HASH_DOMAIN)
    digest.update(initial.session_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update("\0".join(names).encode("utf-8"))
    digest.update(b"\0")
    for label, shape, array in (
        ("joint_position_source", (_JOINT_COUNT,), initial.joint_position_source),
        ("joint_velocity_source", (_JOINT_COUNT,), initial.joint_velocity_source),
        (
            "physical_pelvis_position_holden",
            (3,),
            initial.physical_pelvis_position_holden,
        ),
        (
            "physical_pelvis_orientation_holden",
            (4,),
            initial.physical_pelvis_orientation_holden,
        ),
        (
            "virtual_root_position_holden",
            (3,),
            initial.virtual_root_position_holden,
        ),
        (
            "virtual_root_orientation_holden",
            (4,),
            initial.virtual_root_orientation_holden,
        ),
    ):
        canonical = _finite_float64(array, shape, f"initial boundary.{label}")
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(canonical, "<f8").tobytes(order="C"))
    return digest.hexdigest()


def _query_coordinate(value: float) -> float:
    """Canonicalize one binary32 query coordinate (normal-or-zero only)."""

    coordinate = np.float32(value)
    bits = int(coordinate.view(np.uint32))
    magnitude = bits & 0x7FFFFFFF
    exponent = magnitude & 0x7F800000
    if not (magnitude == 0 or (exponent != 0 and exponent != 0x7F800000)):
        raise SceneError("terrain query coordinate is not normal-or-zero binary32")
    if magnitude == 0:
        return 0.0
    return float(coordinate)


def _canonicalize_output(value: float) -> float:
    rounded = np.float32(value)
    bits = int(rounded.view(np.uint32))
    if (bits & 0x7F800000) == 0:
        return 0.0
    return float(rounded)


def _normal_or_positive_zero(value: np.float32) -> bool:
    bits = int(np.float32(value).view(np.uint32))
    exponent = bits & 0x7F800000
    return bits == 0 or (exponent != 0 and exponent != 0x7F800000)


def _axis_cell(
    value: float, origin: float, count: int, cell_size: float
) -> tuple[int, float] | None:
    difference = value - origin
    coordinate = difference / cell_size
    if not math.isfinite(coordinate):
        return None
    floored = math.floor(coordinate)
    if floored <= 0.0:
        index = 0
    elif floored >= float(count - 2):
        index = count - 2
    else:
        index = int(floored)
    while index > 0:
        node = origin + float(index) * cell_size
        if not (value < node):
            break
        index -= 1
    while index < count - 2:
        next_node = origin + float(index + 1) * cell_size
        if not (value >= next_node):
            break
        index += 1
    node = origin + float(index) * cell_size
    local_fraction = (value - node) / cell_size
    if not math.isfinite(local_fraction):
        return None
    if local_fraction <= 0.0:
        fraction = 0.0
    elif local_fraction >= 1.0:
        fraction = 1.0
    else:
        fraction = local_fraction
    return index, fraction


@dataclass(frozen=True)
class _Heightfield:
    nx: int
    nz: int
    origin_x: float
    origin_z: float
    cell_size: float
    exterior_height: float
    heights: np.ndarray


def _parse_heightfield(contents: bytes) -> _Heightfield:
    if len(contents) < 32:
        raise SceneError("terrain.bin has a truncated G1HF header")
    magic, version, nx, nz, origin_x, origin_z, cell, exterior = struct.unpack(
        "<4sIIIffff", contents[:32]
    )
    if magic != b"G1HF" or version != 2 or nx < 2 or nz < 2:
        raise SceneError("terrain.bin is not a valid G1HF/v2 heightfield")
    if 32 + int(nx) * int(nz) * 4 != len(contents):
        raise SceneError("terrain.bin G1HF payload length is invalid")
    heights = np.frombuffer(contents, dtype="<f4", offset=32)
    if not np.all(np.isfinite(heights)):
        raise SceneError("terrain.bin heights must be finite")
    return _Heightfield(
        nx=int(nx),
        nz=int(nz),
        origin_x=float(np.float32(origin_x)),
        origin_z=float(np.float32(origin_z)),
        cell_size=float(np.float32(cell)),
        exterior_height=float(np.float32(exterior)),
        heights=heights,
    )


def _sample_terrain_height(
    field: _Heightfield, x_holden: float, z_holden: float
) -> float:
    """Reproduce the fixed-diagonal ``tx >= tz`` interpolation in binary64."""

    x = float(_query_coordinate(x_holden))
    z = float(_query_coordinate(z_holden))
    maximum_x = field.origin_x + float(field.nx - 1) * field.cell_size
    maximum_z = field.origin_z + float(field.nz - 1) * field.cell_size
    if not (math.isfinite(maximum_x) and math.isfinite(maximum_z)):
        return field.exterior_height
    if (
        x < field.origin_x
        or x > maximum_x
        or z < field.origin_z
        or z > maximum_z
    ):
        return field.exterior_height
    cell_x = _axis_cell(x, field.origin_x, field.nx, field.cell_size)
    cell_z = _axis_cell(z, field.origin_z, field.nz, field.cell_size)
    if cell_x is None or cell_z is None:
        return field.exterior_height
    x0, tx = cell_x
    z0, tz = cell_z
    offset = z0 * field.nx + x0
    value00 = field.heights[offset]
    value10 = field.heights[offset + 1]
    value01 = field.heights[offset + field.nx]
    value11 = field.heights[offset + field.nx + 1]
    if not all(
        _normal_or_positive_zero(value)
        for value in (value00, value10, value01, value11)
    ):
        return field.exterior_height
    h00 = float(value00)
    h10 = float(value10)
    h01 = float(value01)
    h11 = float(value11)
    if tx >= tz:
        value = h00 + tx * (h10 - h00) + tz * (h11 - h10)
    else:
        value = h00 + tx * (h11 - h01) + tz * (h01 - h00)
    if not math.isfinite(value):
        return field.exterior_height
    return _canonicalize_output(value)


def _named_body_position(model, data, name: str) -> np.ndarray:
    try:
        body_id = int(model.body(name).id)
    except KeyError as error:
        raise SceneError(f"required GEAR body is missing: {name}") from error
    return np.asarray(data.xpos[body_id], np.float64)


def _body_descends_from(model, body_id: int, ancestor_id: int) -> bool:
    cursor = int(body_id)
    while cursor > 0:
        if cursor == ancestor_id:
            return True
        cursor = int(model.body_parentid[cursor])
    return False


def _foot_clearance(
    scene: RegisteredScene,
    field: _Heightfield | None,
    model,
    data,
    side: str,
) -> float:
    body_name = f"{side}_ankle_roll_link"
    try:
        ankle_id = int(model.body(body_name).id)
    except KeyError as error:
        raise SceneError(f"required GEAR body is missing: {body_name}") from error
    geom_ids: list[int] = []
    for raw_geom_id in scene.allowed_foot_geoms:
        geom_id = int(raw_geom_id)
        if geom_id < 0 or geom_id >= int(model.ngeom):
            raise SceneError("registered allowed foot geom is outside the model")
        if _body_descends_from(
            model, int(model.geom_bodyid[geom_id]), ankle_id
        ):
            geom_ids.append(geom_id)
    if not geom_ids:
        raise SceneError(f"registered {side} sole geom group resolved empty")
    return min(
        _surface_height(
            scene,
            field,
            np.asarray(data.geom_xpos[geom_id], np.float64),
        )
        for geom_id in geom_ids
    )


def _surface_height(
    scene: RegisteredScene,
    field: _Heightfield | None,
    position_mujoco: np.ndarray,
) -> float:
    holden = np.asarray(
        mujoco_to_holden_vectors(np.asarray(position_mujoco, np.float64)),
        np.float64,
    )
    if field is None:
        surface = 0.0
    else:
        surface = _sample_terrain_height(field, float(holden[0]), float(holden[2]))
    if not math.isfinite(surface):
        raise SceneError("terrain surface height is not finite")
    clearance = float(holden[1]) - float(surface)
    if not math.isfinite(clearance):
        raise SceneError("terrain clearance is not finite")
    return clearance


def initial_physics_state(
    scene: RegisteredScene,
    initial: InitialBoundary,
    contract: JointContract,
    hand_targets: Dex3HandTargets,
) -> InitialPhysicsState:
    """Build the exact name-driven initial MuJoCo state from the reset boundary."""

    from .transform import map_source_joints

    if not isinstance(scene, RegisteredScene):
        raise SceneError("registered scene is required")
    if not isinstance(initial, InitialBoundary):
        raise SceneError("validated initial boundary is required")
    if scene.transformed_bounds_mujoco is None:
        raise SceneError("registered scene is missing transformed bounds")
    validated_hands = validate_hand_targets(hand_targets)

    pelvis_position_holden = _finite_float64(
        initial.physical_pelvis_position_holden, (3,), "physical pelvis position"
    )
    pelvis_orientation_holden = _finite_float64(
        initial.physical_pelvis_orientation_holden,
        (4,),
        "physical pelvis orientation",
    )
    joint_position_source = _finite_float64(
        initial.joint_position_source, (_JOINT_COUNT,), "joint position source"
    )

    pelvis_position_mujoco = np.asarray(
        holden_to_mujoco_vectors(pelvis_position_holden), np.float64
    )
    pelvis_quaternion_mujoco = np.asarray(
        holden_to_mujoco_quaternions(pelvis_orientation_holden), np.float64
    ).reshape(4)
    target_joint_values = np.asarray(
        map_source_joints(
            joint_position_source, initial.source_joint_names, contract
        ),
        np.float64,
    )

    mujoco, model, terrain_geom = _load_authenticated_scene_model(scene)

    free_joints = [
        joint_id
        for joint_id in range(int(model.njnt))
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(free_joints) != 1:
        raise SceneError("initial-state model must contain exactly one free root")
    root_joint = free_joints[0]
    if (
        model.joint(root_joint).name != "floating_base_joint"
        or model.body(int(model.jnt_bodyid[root_joint])).name != "pelvis"
    ):
        raise SceneError("free root must be floating_base_joint on pelvis")
    root_address = int(model.jnt_qposadr[root_joint])

    qpos = np.array(model.qpos0, np.float64).copy()
    qpos[root_address : root_address + 3] = pelvis_position_mujoco
    qpos[root_address + 3 : root_address + 7] = pelvis_quaternion_mujoco

    used_addresses: set[int] = set()

    def _place(name: str, value: float) -> None:
        try:
            joint_id = int(model.joint(name).id)
        except KeyError as error:
            raise SceneError(f"initial-state model is missing joint {name}") from error
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise SceneError(f"joint {name} is not a hinge")
        address = int(model.jnt_qposadr[joint_id])
        if address in used_addresses:
            raise SceneError(f"duplicate qpos address for joint {name}")
        used_addresses.add(address)
        if bool(model.jnt_limited[joint_id]):
            lower, upper = (float(bound) for bound in model.jnt_range[joint_id])
            if value < lower or value > upper:
                raise SceneError(
                    f"joint {name} position {value} is outside range "
                    f"[{lower}, {upper}]"
                )
        qpos[address] = value

    for row in contract.rows:
        _place(row.target_name, float(target_joint_values[row.target_index]))
    hand_values = tuple(validated_hands.left) + tuple(validated_hands.right)
    for name, value in zip(
        LEFT_HAND_JOINT_ORDER + RIGHT_HAND_JOINT_ORDER, hand_values
    ):
        _place(name, float(value))

    if len(used_addresses) != _JOINT_COUNT + len(hand_values):
        raise SceneError("named initial joints did not resolve to unique addresses")
    if not np.all(np.isfinite(qpos)):
        raise SceneError("initial qpos contains nonfinite values")

    horizontal = np.asarray(scene.transformed_bounds_mujoco, np.float64)[:, :2]
    root_horizontal = pelvis_position_mujoco[:2]
    if np.any(root_horizontal < horizontal[0]) or np.any(
        root_horizontal > horizontal[1]
    ):
        raise SceneError("physical root is outside the registered scene domain")

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)

    field: _Heightfield | None = None
    if scene.source_heightfield is not None:
        _, heightfield_bytes, _ = _read_authenticated(
            scene.source_heightfield,
            scene.source_hashes.get("terrain_bin"),
            "registered terrain.bin",
            maximum_bytes=_MAXIMUM_OBJ_BYTES,
        )
        field = _parse_heightfield(heightfield_bytes)

    pelvis_clearance = _surface_height(
        scene, field, _named_body_position(model, data, "pelvis")
    )
    left_foot_clearance = _foot_clearance(
        scene, field, model, data, "left"
    )
    right_foot_clearance = _foot_clearance(
        scene, field, model, data, "right"
    )

    forbidden_geoms: set[int] = set()
    for group_name, geom_ids in scene.forbidden_geom_groups.items():
        for raw_geom_id in geom_ids:
            geom_id = int(raw_geom_id)
            if geom_id < 0 or geom_id >= int(model.ngeom):
                raise SceneError("registered forbidden geom is outside the model")
            forbidden_geoms.add(geom_id)
            clearance = _surface_height(
                scene,
                field,
                np.asarray(data.geom_xpos[geom_id], np.float64),
            )
            if penetration_exceeds_threshold(
                max(0.0, -clearance), PENETRATION_THRESHOLD_M
            ):
                geom_name = model.geom(geom_id).name or str(geom_id)
                raise SceneError(
                    f"registered forbidden geom {geom_name} in group "
                    f"{group_name} penetrates below authenticated terrain"
                )
    maximum_forbidden = 0.0
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 == terrain_geom and geom2 != terrain_geom:
            robot_geom = geom2
        elif geom2 == terrain_geom and geom1 != terrain_geom:
            robot_geom = geom1
        else:
            continue
        if robot_geom not in forbidden_geoms:
            continue
        penetration = max(0.0, -float(contact.dist))
        maximum_forbidden = max(maximum_forbidden, penetration)
        if penetration_exceeds_threshold(penetration, PENETRATION_THRESHOLD_M):
            raise SceneError(
                "registered forbidden geom penetrates terrain beyond "
                f"{PENETRATION_THRESHOLD_M} m"
            )

    canonical_qpos = np.ascontiguousarray(qpos, "<f8")
    canonical_qpos.flags.writeable = False
    qpos_sha256 = hashlib.sha256(canonical_qpos.tobytes(order="C")).hexdigest()

    return InitialPhysicsState(
        qpos=canonical_qpos,
        qpos_sha256=qpos_sha256,
        initial_boundary_sha256=_initial_boundary_sha256(initial),
        maximum_forbidden_penetration_m=float(maximum_forbidden),
        pelvis_clearance_m=float(pelvis_clearance),
        left_foot_clearance_m=float(left_foot_clearance),
        right_foot_clearance_m=float(right_foot_clearance),
    )
