"""Authenticated scene conversion, GEAR overlays, and kinematic replay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from .joints import TARGET_JOINT_ORDER


HOLDEN_COORDINATE_SIGNATURE = "holden-y-up-right-handed-forward-plus-z"
MUJOCO_COORDINATE_SIGNATURE = "mujoco-z-up-right-handed-forward-plus-x"
FLAT_SCENE_ID = "sonic-flat-baseline"
SCENE_REGISTRY_SCHEMA = "mm-sonic-scene-registry/v1"
FLAT_REGISTRY_SHA256 = (
    "85480e1e8a190d6b065749382172191850739c0aca8a8946182f97665d693d5b"
)
PENETRATION_THRESHOLD_M = 0.005
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCENE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAXIMUM_JSON_BYTES = 16 * 1024 * 1024
_MAXIMUM_OBJ_BYTES = 512 * 1024 * 1024

HOLDEN_TO_MUJOCO_MATRIX = np.array(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)
HOLDEN_TO_MUJOCO_MATRIX.flags.writeable = False


class SceneError(ValueError):
    """A scene artifact, transform, model, or replay contract failed."""


@dataclass(frozen=True, eq=False)
class ObjTransform:
    source_sha256: str
    transformed_sha256: str
    source_bounds_holden: np.ndarray
    transformed_bounds_mujoco: np.ndarray
    vertex_count: int
    normal_count: int
    face_count: int

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ObjTransform)
            and self.source_sha256 == other.source_sha256
            and self.transformed_sha256 == other.transformed_sha256
            and np.array_equal(
                self.source_bounds_holden, other.source_bounds_holden
            )
            and np.array_equal(
                self.transformed_bounds_mujoco,
                other.transformed_bounds_mujoco,
            )
            and self.vertex_count == other.vertex_count
            and self.normal_count == other.normal_count
            and self.face_count == other.face_count
        )


@dataclass(frozen=True)
class RegisteredScene:
    scene_id: str
    route_id: str | None
    source_kind: str
    source_mesh: Path | None
    source_heightfield: Path | None
    source_hashes: Mapping[str, str]
    coordinate_source: str
    coordinate_target: str
    transform_matrix: np.ndarray
    transformed_obj: Path | None
    gear_scene_xml: Path
    output_hashes: Mapping[str, str]
    allowed_foot_geoms: tuple[int, ...]
    forbidden_geom_groups: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class ReplayContact:
    frame_index: int
    terrain_geom: int
    robot_geom: int
    group: str
    distance_m: float
    penetration_m: float


@dataclass(frozen=True)
class KinematicReplayReport:
    frame_count: int
    qpos: np.ndarray
    contacts: tuple[ReplayContact, ...]
    allowed_foot_contacts: bool
    maximum_forbidden_penetration_m: float
    forbidden_penetration: bool
    threshold_m: float


@dataclass(frozen=True)
class _TerrainSource:
    scene_id: str
    route_id: str
    scene_json: Path
    terrain_bin: Path
    terrain_obj: Path
    source_hashes: Mapping[str, str]
    walkability_sha256: str
    mesh_bounds_holden: np.ndarray
    obj_bytes: bytes


@dataclass(frozen=True)
class _MMCatalogIdentity:
    root: Path
    manifest_sha256: str
    scene_index_sha256: str
    scene_index: Mapping[str, Any]


def _validate_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise SceneError(f"{label} must be a lowercase SHA-256")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as error:
            raise SceneError(f"cannot inspect {label}: {current}") from error
        if stat.S_ISLNK(status.st_mode):
            raise SceneError(f"{label} contains a symlink: {current}")


def _read_regular(
    path: Path | str,
    label: str,
    *,
    maximum_bytes: int,
) -> tuple[Path, bytes]:
    candidate = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_components(candidate, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise SceneError(f"cannot open {label}: {candidate}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SceneError(f"{label} must be a regular file: {candidate}")
        if before.st_nlink != 1:
            raise SceneError(f"{label} must not be hard linked: {candidate}")
        if before.st_size < 0 or before.st_size > maximum_bytes:
            raise SceneError(f"{label} has an invalid byte length: {candidate}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise SceneError(f"{label} changed while being read: {candidate}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SceneError(f"{label} grew while being read: {candidate}")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise SceneError(f"{label} changed while being read: {candidate}")
        return candidate, b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_authenticated(
    path: Path | str,
    expected_sha256: object,
    label: str,
    *,
    maximum_bytes: int,
) -> tuple[Path, bytes, str]:
    expected = _validate_sha256(expected_sha256, f"{label} expected SHA-256")
    resolved, contents = _read_regular(
        path, label, maximum_bytes=maximum_bytes
    )
    observed = _sha256_bytes(contents)
    if observed != expected:
        raise SceneError(
            f"{label} SHA-256 mismatch: expected {expected}, got {observed}"
        )
    return resolved, contents, observed


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise SceneError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _json_bytes(contents: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            contents.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SceneError(f"{label} contains invalid constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SceneError(f"{label} is not valid UTF-8 JSON") from error
    if type(value) is not dict:
        raise SceneError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        actual = set(value) if type(value) is dict else set()
        raise SceneError(
            f"{label} keys differ: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SceneError(f"{label} must be a finite number")
    return result


def _number_tuple(value: object, count: int, label: str) -> tuple[float, ...]:
    if type(value) is not list or len(value) != count:
        raise SceneError(f"{label} must contain exactly {count} numbers")
    return tuple(_finite_number(item, label) for item in value)


def _readonly_array(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    output = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy(order="C")
    output.flags.writeable = False
    return output


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def load_scene_registry(
    path: Path | str,
) -> Mapping[str, Mapping[str, object]]:
    """Load the one exact shared analytic scene registry."""

    _, contents, digest = _read_authenticated(
        path,
        FLAT_REGISTRY_SHA256,
        "scene registry",
        maximum_bytes=_MAXIMUM_JSON_BYTES,
    )
    if digest != FLAT_REGISTRY_SHA256:
        raise SceneError("scene registry digest changed")
    payload = _exact_keys(
        _json_bytes(contents, "scene registry"),
        {"schema", "scenes"},
        "scene registry",
    )
    if payload["schema"] != SCENE_REGISTRY_SCHEMA:
        raise SceneError("scene registry schema changed")
    scenes = _exact_keys(
        payload["scenes"], {FLAT_SCENE_ID}, "scene registry scenes"
    )
    entry = _exact_keys(
        scenes[FLAT_SCENE_ID],
        {
            "kind",
            "coordinate_signature",
            "bounds_xz",
            "spawn_position_holden",
            "spawn_yaw_holden",
            "height_m",
            "walkability_class",
        },
        "flat scene entry",
    )
    normalized: dict[str, object] = {
        "kind": entry["kind"],
        "coordinate_signature": entry["coordinate_signature"],
        "bounds_xz": _number_tuple(entry["bounds_xz"], 4, "bounds_xz"),
        "spawn_position_holden": _number_tuple(
            entry["spawn_position_holden"], 3, "spawn_position_holden"
        ),
        "spawn_yaw_holden": _finite_number(
            entry["spawn_yaw_holden"], "spawn_yaw_holden"
        ),
        "height_m": _finite_number(entry["height_m"], "height_m"),
        "walkability_class": entry["walkability_class"],
    }
    expected = {
        "kind": "analytic-flat",
        "coordinate_signature": HOLDEN_COORDINATE_SIGNATURE,
        "bounds_xz": (-10.0, -10.0, 10.0, 10.0),
        "spawn_position_holden": (0.0, 0.0, 0.0),
        "spawn_yaw_holden": 0.0,
        "height_m": 0.0,
        "walkability_class": 1,
    }
    if normalized != expected or type(entry["walkability_class"]) is not int:
        raise SceneError("sonic-flat-baseline analytic values changed")
    return MappingProxyType(
        {FLAT_SCENE_ID: MappingProxyType(normalized)}
    )


def _format_float32(value: np.float32) -> str:
    candidate = np.float32(value)
    if not np.isfinite(candidate):
        raise SceneError("OBJ transform produced a non-finite binary32 value")
    if candidate == np.float32(0.0):
        candidate = np.float32(0.0)
    return format(float(candidate), ".9g")


def _transform_vector(values: Sequence[str], label: str) -> np.ndarray:
    if len(values) != 3:
        raise SceneError(f"{label} must contain exactly three coordinates")
    try:
        source64 = np.asarray([float(value) for value in values], np.float64)
    except ValueError as error:
        raise SceneError(f"{label} must contain numeric coordinates") from error
    if not np.all(np.isfinite(source64)):
        raise SceneError(f"{label} coordinates must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        source = source64.astype(np.float32)
    if not np.all(np.isfinite(source)):
        raise SceneError(f"{label} coordinates must remain finite as binary32")
    target = np.array((source[0], -source[2], source[1]), np.float32)
    return target


def _obj_transformed_bytes(contents: bytes) -> tuple[bytes, ObjTransform]:
    if not contents.endswith(b"\n"):
        raise SceneError("OBJ must end in a newline")
    if b"\x00" in contents:
        raise SceneError("OBJ must not contain NUL bytes")
    try:
        text = contents.decode("ascii")
    except UnicodeDecodeError as error:
        raise SceneError("OBJ must use ASCII syntax") from error
    output_lines: list[str] = []
    source_vertices: list[np.ndarray] = []
    target_vertices: list[np.ndarray] = []
    normal_count = 0
    face_tokens: list[tuple[str, ...]] = []
    for line_number, line_with_newline in enumerate(
        text.splitlines(keepends=True), start=1
    ):
        line = line_with_newline[:-1]
        if line.endswith("\r"):
            raise SceneError("OBJ must use LF line endings")
        tokens = line.split()
        if not tokens or tokens[0].startswith("#"):
            output_lines.append(line + "\n")
            continue
        if tokens[0] == "v":
            target = _transform_vector(tokens[1:], f"OBJ vertex {line_number}")
            source = np.array((target[0], target[2], -target[1]), np.float32)
            source_vertices.append(source)
            target_vertices.append(target)
            output_lines.append(
                "v " + " ".join(_format_float32(value) for value in target) + "\n"
            )
        elif tokens[0] == "vn":
            target = _transform_vector(tokens[1:], f"OBJ normal {line_number}")
            if float(np.linalg.norm(target.astype(np.float64))) <= 0.0:
                raise SceneError(f"OBJ normal {line_number} must be nonzero")
            normal_count += 1
            output_lines.append(
                "vn "
                + " ".join(_format_float32(value) for value in target)
                + "\n"
            )
        elif tokens[0] == "f":
            if len(tokens) != 4:
                raise SceneError("OBJ faces must be triangles")
            face_tokens.append(tuple(tokens[1:]))
            output_lines.append(line + "\n")
        else:
            output_lines.append(line + "\n")
    if len(source_vertices) < 3 or not face_tokens:
        raise SceneError("OBJ must contain vertices and triangular faces")
    for face in face_tokens:
        for token in face:
            first = token.split("/", 1)[0]
            try:
                index = int(first)
            except ValueError as error:
                raise SceneError("OBJ face vertex index must be an integer") from error
            if index == 0 or abs(index) > len(source_vertices):
                raise SceneError("OBJ face vertex index is out of bounds")
    source_array = np.stack(source_vertices)
    target_array = np.stack(target_vertices)
    source_bounds = _readonly_array(
        np.stack((np.min(source_array, axis=0), np.max(source_array, axis=0))),
        np.float32,
    )
    target_bounds = _readonly_array(
        np.stack((np.min(target_array, axis=0), np.max(target_array, axis=0))),
        np.float32,
    )
    transformed = "".join(output_lines).encode("ascii")
    record = ObjTransform(
        source_sha256=_sha256_bytes(contents),
        transformed_sha256=_sha256_bytes(transformed),
        source_bounds_holden=source_bounds,
        transformed_bounds_mujoco=target_bounds,
        vertex_count=len(source_vertices),
        normal_count=normal_count,
        face_count=len(face_tokens),
    )
    return transformed, record


def _write_exclusive(path: Path, contents: bytes, label: str) -> None:
    if path.name in ("", ".", ".."):
        raise SceneError(f"{label} destination is invalid")
    _reject_symlink_components(path.parent, f"{label} destination parent")
    try:
        parent_status = os.stat(path.parent, follow_symlinks=False)
    except OSError as error:
        raise SceneError(f"{label} destination parent does not exist") from error
    if not stat.S_ISDIR(parent_status.st_mode):
        raise SceneError(f"{label} destination parent must be a directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags, 0o644)
        view = memoryview(contents)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SceneError(f"cannot write {label} destination")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as error:
        raise SceneError(f"{label} destination already exists or is unsafe") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def transform_obj(
    source_obj: Path | str,
    expected_source_sha256: str,
    destination_obj: Path | str,
) -> ObjTransform:
    """Transform authenticated OBJ vertices/normals into MuJoCo coordinates."""

    _, source, observed = _read_authenticated(
        source_obj,
        expected_source_sha256,
        "source OBJ",
        maximum_bytes=_MAXIMUM_OBJ_BYTES,
    )
    transformed, record = _obj_transformed_bytes(source)
    if observed != record.source_sha256:
        raise SceneError("source OBJ changed during transformation")
    _write_exclusive(Path(destination_obj), transformed, "OBJ")
    return record


def _safe_relative(value: object, expected: str, label: str) -> str:
    if type(value) is not str or value != expected:
        raise SceneError(f"{label} path must equal {expected}")
    parts = Path(value).parts
    if Path(value).is_absolute() or not parts or any(
        part in ("", ".", "..") for part in parts
    ):
        raise SceneError(f"{label} path is unsafe")
    return value


def _descriptor(value: object, label: str, expected_path: str) -> dict[str, Any]:
    descriptor = _exact_keys(
        value,
        {"path", "schema", "sha256"},
        label,
    )
    _safe_relative(descriptor["path"], expected_path, label)
    _validate_sha256(descriptor["sha256"], f"{label} SHA-256")
    return descriptor


def _validate_g1hf(contents: bytes, descriptor: Mapping[str, Any]) -> None:
    if len(contents) < 32:
        raise SceneError("terrain.bin has a truncated G1HF header")
    magic, version, nx, nz, origin_x, origin_z, cell, exterior = struct.unpack(
        "<4sIIIffff", contents[:32]
    )
    expected_length = 32 + int(nx) * int(nz) * 4
    if magic != b"G1HF" or version != 2 or nx < 2 or nz < 2:
        raise SceneError("terrain.bin is not a valid G1HF/v2 heightfield")
    if expected_length != len(contents):
        raise SceneError("terrain.bin G1HF payload length is invalid")
    values = np.frombuffer(contents, dtype="<f4", offset=32)
    if not np.all(np.isfinite(values)):
        raise SceneError("terrain.bin heights must be finite")
    expected = (
        descriptor.get("version"),
        descriptor.get("nx"),
        descriptor.get("nz"),
    )
    if expected != (2, int(nx), int(nz)):
        raise SceneError("terrain.bin dimensions disagree with scene.json")
    numeric = (
        ("origin_x", origin_x),
        ("origin_z", origin_z),
        ("cell_size_m", cell),
        ("exterior_height_m", exterior),
    )
    for name, observed in numeric:
        expected_value = _finite_number(descriptor.get(name), f"heightfield {name}")
        expected_bits = np.float32(expected_value).view(np.uint32)
        observed_bits = np.float32(observed).view(np.uint32)
        if expected_bits != observed_bits:
            raise SceneError(f"terrain.bin {name} disagrees with scene.json")


def _load_mm_catalog_identity(
    terrain_dir: Path | str,
) -> _MMCatalogIdentity:
    root = Path(terrain_dir)
    if not root.is_absolute():
        root = Path(os.path.abspath(os.fspath(root)))
    _reject_symlink_components(root, "terrain root")
    try:
        root_status = os.stat(root, follow_symlinks=False)
    except OSError as error:
        raise SceneError("terrain root is unavailable") from error
    if not stat.S_ISDIR(root_status.st_mode):
        raise SceneError("terrain root must be a directory")

    _, manifest_bytes = _read_regular(
        root / "manifest.json",
        "terrain manifest",
        maximum_bytes=_MAXIMUM_OBJ_BYTES,
    )
    manifest = _json_bytes(manifest_bytes, "terrain manifest")
    if manifest.get("schema") != "g1-terrain-artifacts/v2":
        raise SceneError("terrain manifest schema changed")
    scene_index_descriptor = _descriptor(
        manifest.get("scene_index"),
        "scene index descriptor",
        "scenes/index.json",
    )
    if scene_index_descriptor["schema"] != "g1-terrain-scene-index/v1":
        raise SceneError("scene index descriptor schema changed")
    _, index_bytes, index_sha = _read_authenticated(
        root / "scenes/index.json",
        scene_index_descriptor["sha256"],
        "scene index",
        maximum_bytes=_MAXIMUM_JSON_BYTES,
    )
    index = _json_bytes(index_bytes, "scene index")
    if (
        index.get("schema") != "g1-terrain-scene-index/v1"
        or index.get("coordinate_signature") != HOLDEN_COORDINATE_SIGNATURE
    ):
        raise SceneError("scene index identity changed")
    return _MMCatalogIdentity(
        root=root,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        scene_index_sha256=index_sha,
        scene_index=MappingProxyType(index),
    )


def _load_terrain_source(
    catalog: _MMCatalogIdentity,
    scene_id: str,
    route_id: str | None,
) -> _TerrainSource:
    if _SCENE_ID.fullmatch(scene_id) is None:
        raise SceneError("terrain scene ID is unsafe")
    if type(route_id) is not str or not route_id:
        raise SceneError("terrain scene route must be nonempty")
    root = catalog.root
    index = catalog.scene_index
    ids = index.get("scene_ids")
    descriptors = index.get("scenes")
    if (
        type(ids) is not list
        or type(descriptors) is not list
        or len(ids) != len(descriptors)
        or len(set(ids)) != len(ids)
        or scene_id not in ids
    ):
        raise SceneError("scene index does not exactly register the requested scene")
    position = ids.index(scene_id)
    scene_descriptor = _exact_keys(
        descriptors[position], {"id", "path", "sha256"}, "scene descriptor"
    )
    expected_scene_path = f"scenes/{scene_id}/scene.json"
    if scene_descriptor["id"] != scene_id:
        raise SceneError("scene descriptor ID changed")
    _safe_relative(scene_descriptor["path"], expected_scene_path, "scene descriptor")
    scene_path, scene_bytes, scene_sha = _read_authenticated(
        root / expected_scene_path,
        scene_descriptor["sha256"],
        "scene.json",
        maximum_bytes=_MAXIMUM_JSON_BYTES,
    )
    scene = _json_bytes(scene_bytes, "scene.json")
    if (
        scene.get("schema") != "g1-terrain-scene/v1"
        or scene.get("id") != scene_id
        or scene.get("coordinate_signature") != HOLDEN_COORDINATE_SIGNATURE
    ):
        raise SceneError("scene.json identity changed")
    heightfield = scene.get("heightfield")
    mesh = scene.get("mesh")
    walkability = scene.get("walkability")
    if (
        type(heightfield) is not dict
        or type(mesh) is not dict
        or type(walkability) is not dict
    ):
        raise SceneError("scene.json artifact descriptors are missing")
    height_path = _safe_relative(
        heightfield.get("path"), "terrain.bin", "heightfield"
    )
    mesh_path = _safe_relative(mesh.get("path"), "terrain.obj", "mesh")
    _safe_relative(walkability.get("path"), "walkability.bin", "walkability")
    if (
        heightfield.get("schema") != "G1HF/v2"
        or mesh.get("schema") != "obj/v1"
        or walkability.get("schema") != "G1WM/v1"
    ):
        raise SceneError("scene.json artifact schemas changed")
    terrain_path, terrain_bytes, terrain_sha = _read_authenticated(
        scene_path.parent / height_path,
        heightfield.get("sha256"),
        "terrain.bin",
        maximum_bytes=_MAXIMUM_OBJ_BYTES,
    )
    obj_path, obj_bytes, obj_sha = _read_authenticated(
        scene_path.parent / mesh_path,
        mesh.get("sha256"),
        "terrain.obj",
        maximum_bytes=_MAXIMUM_OBJ_BYTES,
    )
    _validate_g1hf(terrain_bytes, heightfield)
    transformed, obj_record = _obj_transformed_bytes(obj_bytes)
    del transformed
    bounds = scene.get("bounds")
    if type(bounds) is not dict:
        raise SceneError("scene.json mesh bounds are missing")
    expected_bounds = _readonly_array(
        (
            _number_tuple(bounds.get("mesh_min_xyz"), 3, "mesh minimum bounds"),
            _number_tuple(bounds.get("mesh_max_xyz"), 3, "mesh maximum bounds"),
        ),
        np.float32,
    )
    if not np.array_equal(obj_record.source_bounds_holden, expected_bounds):
        raise SceneError("terrain.obj mesh bounds disagree with scene.json")
    routes = scene.get("routes")
    if type(routes) is not list:
        raise SceneError("scene.json routes are missing")
    matching_routes = [
        route
        for route in routes
        if type(route) is dict and route.get("id") == route_id
    ]
    if len(matching_routes) != 1:
        raise SceneError(f"scene {scene_id} has no unique route {route_id}")
    walkability_sha = _validate_sha256(
        walkability.get("sha256"), "walkability expected SHA-256"
    )
    source_hashes = _immutable_mapping(
        {
            "manifest": catalog.manifest_sha256,
            "scene_index": catalog.scene_index_sha256,
            "scene_json": scene_sha,
            "terrain_bin": terrain_sha,
            "terrain_obj": obj_sha,
            "walkability": walkability_sha,
        }
    )
    return _TerrainSource(
        scene_id=scene_id,
        route_id=route_id,
        scene_json=scene_path,
        terrain_bin=terrain_path,
        terrain_obj=obj_path,
        source_hashes=source_hashes,
        walkability_sha256=walkability_sha,
        mesh_bounds_holden=expected_bounds,
        obj_bytes=obj_bytes,
    )


def _make_output_directory(
    path: Path | str,
    *,
    input_roots: Sequence[Path],
) -> Path:
    output = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_components(output.parent, "scene output parent")
    for root in input_roots:
        protected = Path(os.path.abspath(os.fspath(root)))
        try:
            output.relative_to(protected)
        except ValueError:
            continue
        raise SceneError(
            f"scene output overlaps an authenticated input tree: {protected}"
        )
    try:
        os.mkdir(output, 0o755)
    except OSError as error:
        raise SceneError("scene output must be a new non-symlink directory") from error
    return output


def _parse_xml_root(contents: bytes, label: str) -> ET.Element:
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SceneError(
            f"{label} must be UTF-8 and must not contain a DOCTYPE"
        ) from error
    if "<!DOCTYPE" in text.upper():
        raise SceneError(f"{label} must not contain a DOCTYPE")
    try:
        return ET.fromstring(contents)
    except (ET.ParseError, ValueError) as error:
        raise SceneError(f"{label} is malformed") from error


def _load_official_scene(
    official_scene_xml: Path | str,
    expected_official_scene_sha256: str,
    expected_robot_sha256: str,
    *,
    label_prefix: str = "",
) -> tuple[Path, ET.ElementTree, Path, bytes, str, str]:
    scene_path, scene_bytes, scene_sha = _read_authenticated(
        official_scene_xml,
        expected_official_scene_sha256,
        f"{label_prefix}official scene XML",
        maximum_bytes=_MAXIMUM_JSON_BYTES,
    )
    root = _parse_xml_root(scene_bytes, f"{label_prefix}official scene XML")
    if root.tag != "mujoco":
        raise SceneError("official scene XML root must be mujoco")
    includes = root.findall("include")
    if len(includes) != 1:
        raise SceneError("official scene must contain exactly one robot include")
    include = includes[0]
    if set(include.attrib) != {"file"}:
        raise SceneError("official robot include attributes changed")
    relative = include.attrib["file"]
    if (
        not relative
        or Path(relative).is_absolute()
        or len(Path(relative).parts) != 1
        or relative in (".", "..")
    ):
        raise SceneError("official robot include path is unsafe")
    robot_path, robot_bytes, robot_sha = _read_authenticated(
        scene_path.parent / relative,
        expected_robot_sha256,
        f"{label_prefix}robot include XML",
        maximum_bytes=_MAXIMUM_JSON_BYTES,
    )
    return (
        scene_path,
        ET.ElementTree(root),
        robot_path,
        robot_bytes,
        scene_sha,
        robot_sha,
    )


def _run_local_robot_bytes(robot_path: Path, robot_bytes: bytes) -> bytes:
    root = _parse_xml_root(robot_bytes, "official robot XML")
    if root.tag != "mujoco":
        raise SceneError("official robot XML root must be mujoco")
    compilers = root.findall("compiler")
    if len(compilers) != 1:
        raise SceneError("official robot must contain exactly one compiler")
    compiler = compilers[0]
    if compiler.attrib.get("meshdir") != "meshes":
        raise SceneError("official robot compiler meshdir must be exactly meshes")

    mesh_directory = robot_path.parent / "meshes"
    _reject_symlink_components(mesh_directory, "official robot mesh directory")
    try:
        mesh_status = os.stat(mesh_directory, follow_symlinks=False)
        resolved_mesh_directory = mesh_directory.resolve(strict=True)
        resolved_mesh_directory.relative_to(robot_path.parent)
    except (OSError, ValueError) as error:
        raise SceneError(
            "official robot mesh directory must remain inside its input directory"
        ) from error
    if not stat.S_ISDIR(mesh_status.st_mode):
        raise SceneError("official robot mesh directory must be a real directory")

    compiler.attrib["meshdir"] = str(resolved_mesh_directory)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
        short_empty_elements=True,
    ) + b"\n"


def _overlay_bytes(
    tree: ET.ElementTree,
    scene_id: str,
    transformed_obj: Path | None,
) -> bytes:
    root = tree.getroot()
    worldbodies = root.findall("worldbody")
    assets = root.findall("asset")
    if len(worldbodies) != 1 or len(assets) != 1:
        raise SceneError("official scene must have one asset and one worldbody")
    worldbody = worldbodies[0]
    asset = assets[0]
    floors = [
        element
        for element in worldbody.findall("geom")
        if element.attrib.get("name") == "floor"
        and element.attrib.get("type") == "plane"
    ]
    if len(floors) != 1:
        raise SceneError("official scene must have exactly one named infinite floor")
    if any(
        element.attrib.get("name") == "mm_terrain"
        for element in root.iter("geom")
    ):
        raise SceneError("official scene already defines mm_terrain")
    if scene_id == FLAT_SCENE_ID:
        if transformed_obj is not None:
            raise SceneError("flat scene must not have a transformed mesh")
        floors[0].attrib["name"] = "mm_terrain"
    else:
        if transformed_obj is None or not transformed_obj.is_absolute():
            raise SceneError("terrain scene requires an absolute transformed OBJ")
        worldbody.remove(floors[0])
        if any(
            element.attrib.get("name") == "mm_terrain_mesh"
            for element in root.iter("mesh")
        ):
            raise SceneError("official scene already defines mm_terrain_mesh")
        ET.SubElement(
            asset,
            "mesh",
            {"name": "mm_terrain_mesh", "file": str(transformed_obj)},
        )
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "mm_terrain",
                "type": "mesh",
                "mesh": "mm_terrain_mesh",
                "contype": "1",
                "conaffinity": "1",
            },
        )
    if len(
        [
            element
            for element in root.iter("geom")
            if element.attrib.get("name") == "mm_terrain"
        ]
    ) != 1:
        raise SceneError("generated scene must have exactly one mm_terrain geom")
    ET.indent(tree, space="  ")
    return ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
        short_empty_elements=True,
    ) + b"\n"


def _import_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as error:
        raise SceneError(
            "MuJoCo is required; install the integration extra"
        ) from error
    return mujoco


def _body_id(model: Any, name: str) -> int:
    try:
        body_id = int(model.body(name).id)
    except KeyError as error:
        raise SceneError(f"required GEAR body is missing: {name}") from error
    if body_id <= 0:
        raise SceneError(f"required GEAR body is invalid: {name}")
    return body_id


def _direct_body_geoms(model: Any, body_ids: set[int]) -> tuple[int, ...]:
    return tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in body_ids
    )


def _descendants(model: Any, root_body: int) -> set[int]:
    output: set[int] = set()
    for body_id in range(1, int(model.nbody)):
        cursor = body_id
        while cursor > 0:
            if cursor == root_body:
                output.add(body_id)
                break
            cursor = int(model.body_parentid[cursor])
    return output


def _resolve_contact_groups(
    model: Any,
) -> tuple[tuple[int, ...], Mapping[str, tuple[int, ...]]]:
    left_foot = _direct_body_geoms(
        model, _descendants(model, _body_id(model, "left_ankle_roll_link"))
    )
    right_foot = _direct_body_geoms(
        model, _descendants(model, _body_id(model, "right_ankle_roll_link"))
    )
    if not left_foot or not right_foot:
        raise SceneError("both allowed foot geom groups must resolve nonempty")
    allowed = tuple(sorted(set(left_foot) | set(right_foot)))
    pelvis = _direct_body_geoms(model, {_body_id(model, "pelvis")})
    knees = _direct_body_geoms(
        model,
        {
            _body_id(model, "left_knee_link"),
            _body_id(model, "right_knee_link"),
        },
    )
    torso = _direct_body_geoms(
        model,
        {
            _body_id(model, "waist_yaw_link"),
            _body_id(model, "waist_roll_link"),
            _body_id(model, "torso_link"),
        },
    )
    hand_bodies = {
        body_id
        for body_id in range(1, int(model.nbody))
        if any(
            token in model.body(body_id).name.lower()
            for token in ("wrist", "hand", "finger")
        )
    }
    hands = _direct_body_geoms(model, hand_bodies)
    groups = {
        "pelvis": tuple(sorted(set(pelvis))),
        "knees": tuple(sorted(set(knees))),
        "torso": tuple(sorted(set(torso))),
        "hands": tuple(sorted(set(hands))),
    }
    allowed_set = set(allowed)
    used: set[int] = set()
    for name, values in groups.items():
        current = set(values)
        if not current:
            raise SceneError(f"forbidden geom group {name} resolved empty")
        if current & allowed_set:
            raise SceneError(f"forbidden geom group {name} overlaps allowed feet")
        if current & used:
            raise SceneError(f"forbidden geom group {name} overlaps another group")
        used |= current
    return allowed, MappingProxyType(groups)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SceneError("scene registration cannot be encoded") from error


def register_scene(
    scene_id: str,
    route_id: str | None,
    *,
    registry_path: Path | str,
    terrain_dir: Path | str | None,
    official_scene_xml: Path | str,
    output_dir: Path | str,
    expected_official_scene_sha256: str,
    expected_robot_sha256: str,
    mm_hello_identity: Mapping[str, object] | None,
    mm_scene_identity: Mapping[str, object] | None,
) -> RegisteredScene:
    """Build one authenticated run-local GEAR scene and geom registry."""

    registry = load_scene_registry(registry_path)
    if terrain_dir is None:
        raise SceneError(
            "terrain_dir is required to authenticate the MM scene catalog"
        )
    catalog = _load_mm_catalog_identity(terrain_dir)
    transformed_path: Path | None = None
    transform_record: ObjTransform | None = None
    terrain: _TerrainSource | None = None
    source_mesh: Path | None = None
    source_heightfield: Path | None = None
    source_bounds: np.ndarray | None = None
    target_bounds: np.ndarray | None = None
    if scene_id == FLAT_SCENE_ID:
        if type(route_id) is not str or not route_id:
            raise SceneError("flat registered scene route must be nonempty")
        entry = registry[FLAT_SCENE_ID]
        source_kind = str(entry["kind"])
        source_hashes = _immutable_mapping(
            {
                "registry": FLAT_REGISTRY_SHA256,
                "manifest": catalog.manifest_sha256,
                "scene_index": catalog.scene_index_sha256,
            }
        )
    else:
        terrain = _load_terrain_source(catalog, scene_id, route_id)
        source_kind = "authenticated-terrain"
        source_hashes = terrain.source_hashes
        source_mesh = terrain.terrain_obj
        source_heightfield = terrain.terrain_bin

    _verify_mm_hello_identity(source_hashes, mm_hello_identity)
    _verify_mm_reset_identity(
        scene_id,
        route_id,
        source_hashes,
        mm_scene_identity,
    )

    (
        scene_path,
        tree,
        official_robot_path,
        official_robot_bytes,
        official_sha,
        official_robot_sha,
    ) = _load_official_scene(
        official_scene_xml,
        expected_official_scene_sha256,
        expected_robot_sha256,
    )
    generated_robot_bytes = _run_local_robot_bytes(
        official_robot_path,
        official_robot_bytes,
    )
    input_roots = [
        Path(os.path.abspath(os.fspath(registry_path))).parent,
        catalog.root,
        scene_path.parent,
    ]
    output = _make_output_directory(
        output_dir,
        input_roots=input_roots,
    )
    generated_robot_path = output / "gear_robot.xml"
    _write_exclusive(
        generated_robot_path,
        generated_robot_bytes,
        "run-local robot XML",
    )
    generated_robot_sha = _sha256_bytes(generated_robot_bytes)
    includes = tree.getroot().findall("include")
    if len(includes) != 1 or set(includes[0].attrib) != {"file"}:
        raise SceneError("official scene robot include changed after authentication")
    includes[0].attrib["file"] = str(generated_robot_path)
    if terrain is not None:
        transformed_path = output / "terrain.obj"
        transform_record = transform_obj(
            terrain.terrain_obj,
            terrain.source_hashes["terrain_obj"],
            transformed_path,
        )
        if not np.array_equal(
            transform_record.source_bounds_holden,
            terrain.mesh_bounds_holden,
        ):
            raise SceneError("transformed OBJ source bounds changed")
        source_bounds = transform_record.source_bounds_holden
        target_bounds = transform_record.transformed_bounds_mujoco
    overlay = _overlay_bytes(tree, scene_id, transformed_path)
    overlay_path = output / "gear_scene.xml"
    _write_exclusive(overlay_path, overlay, "GEAR scene XML")
    overlay_sha = _sha256_bytes(overlay)
    mujoco = _import_mujoco()
    try:
        model = mujoco.MjModel.from_xml_path(str(overlay_path))
    except (ValueError, OSError) as error:
        raise SceneError(f"generated GEAR scene does not load: {error}") from error
    terrain_geom_ids = [
        geom_id
        for geom_id in range(int(model.ngeom))
        if model.geom(geom_id).name == "mm_terrain"
    ]
    if len(terrain_geom_ids) != 1:
        raise SceneError("compiled GEAR scene must contain exactly one mm_terrain geom")
    allowed, forbidden = _resolve_contact_groups(model)

    output_hashes: dict[str, str] = {
        "official_scene": official_sha,
        "official_robot": official_robot_sha,
        "robot_include": generated_robot_sha,
        "gear_scene_xml": overlay_sha,
    }
    if transform_record is not None:
        output_hashes["transformed_obj"] = transform_record.transformed_sha256
    registration_payload: dict[str, Any] = {
        "schema": "mm-sonic-scene-registration/v1",
        "scene_id": scene_id,
        "route_id": route_id,
        "source_kind": source_kind,
        "source_hashes": dict(source_hashes),
        "coordinate_source": HOLDEN_COORDINATE_SIGNATURE,
        "coordinate_target": MUJOCO_COORDINATE_SIGNATURE,
        "transform_matrix": HOLDEN_TO_MUJOCO_MATRIX.tolist(),
        "source_bounds_holden": (
            None if source_bounds is None else source_bounds.tolist()
        ),
        "output_bounds_mujoco": (
            None if target_bounds is None else target_bounds.tolist()
        ),
        "official_scene": str(scene_path),
        "official_scene_sha256": official_sha,
        "official_robot_include": str(official_robot_path),
        "official_robot_include_sha256": official_robot_sha,
        "robot_include": str(generated_robot_path),
        "robot_include_sha256": generated_robot_sha,
        "gear_scene_sha256": overlay_sha,
        "transformed_obj_sha256": (
            None
            if transform_record is None
            else transform_record.transformed_sha256
        ),
        "allowed_foot_geoms": list(allowed),
        "forbidden_geom_groups": {
            name: list(values) for name, values in forbidden.items()
        },
    }
    registration_bytes = _canonical_json(registration_payload)
    _write_exclusive(
        output / "scene_registration.json",
        registration_bytes,
        "scene registration",
    )
    output_hashes["scene_registration"] = _sha256_bytes(registration_bytes)
    transform_matrix = HOLDEN_TO_MUJOCO_MATRIX.copy()
    transform_matrix.flags.writeable = False
    registered = RegisteredScene(
        scene_id=scene_id,
        route_id=route_id,
        source_kind=source_kind,
        source_mesh=source_mesh,
        source_heightfield=source_heightfield,
        source_hashes=_immutable_mapping(source_hashes),
        coordinate_source=HOLDEN_COORDINATE_SIGNATURE,
        coordinate_target=MUJOCO_COORDINATE_SIGNATURE,
        transform_matrix=transform_matrix,
        transformed_obj=transformed_path,
        gear_scene_xml=overlay_path,
        output_hashes=_immutable_mapping(output_hashes),
        allowed_foot_geoms=allowed,
        forbidden_geom_groups=forbidden,
    )
    verify_mm_scene_identity(
        registered,
        mm_hello_identity,
        mm_scene_identity,
    )
    return registered


def _verify_mm_hello_identity(
    source_hashes: Mapping[str, str],
    mm_hello_identity: Mapping[str, object] | None,
) -> None:
    mismatch = not isinstance(mm_hello_identity, Mapping) or (
        mm_hello_identity.get("coordinate_signature")
        != HOLDEN_COORDINATE_SIGNATURE
        or mm_hello_identity.get("motion_manifest_sha256")
        != source_hashes.get("manifest")
        or mm_hello_identity.get("scene_index_sha256")
        != source_hashes.get("scene_index")
    )
    if mismatch:
        raise SceneError(
            "MM hello identity does not match authenticated scene artifacts"
        )


def _verify_mm_reset_identity(
    scene_id: str,
    route_id: str | None,
    source_hashes: Mapping[str, str],
    mm_scene_identity: Mapping[str, object] | None,
) -> None:
    expected_hashes = (
        (
            FLAT_REGISTRY_SHA256,
            FLAT_REGISTRY_SHA256,
            FLAT_REGISTRY_SHA256,
        )
        if scene_id == FLAT_SCENE_ID
        else (
            source_hashes.get("terrain_bin"),
            source_hashes.get("terrain_obj"),
            source_hashes.get("walkability"),
        )
    )
    mismatch = not isinstance(mm_scene_identity, Mapping) or (
        mm_scene_identity.get("scene_id") != scene_id
        or mm_scene_identity.get("route_id") != route_id
        or mm_scene_identity.get("coordinate_signature")
        != HOLDEN_COORDINATE_SIGNATURE
        or tuple(
            mm_scene_identity.get(name)
            for name in (
                "heightfield_sha256",
                "mesh_sha256",
                "walkability_sha256",
            )
        )
        != expected_hashes
    )
    if mismatch:
        raise SceneError("MM scene identity does not match generated overlay")


def verify_mm_scene_identity(
    scene: RegisteredScene,
    mm_hello_identity: Mapping[str, object] | None,
    mm_scene_identity: Mapping[str, object] | None,
) -> None:
    """Bind MM hello/reset identities to authenticated scene artifacts."""

    if not isinstance(scene, RegisteredScene):
        raise SceneError("MM scene identity is invalid")
    _verify_mm_hello_identity(scene.source_hashes, mm_hello_identity)
    _verify_mm_reset_identity(
        scene.scene_id,
        scene.route_id,
        scene.source_hashes,
        mm_scene_identity,
    )


def penetration_exceeds_threshold(
    penetration_m: float,
    threshold_m: float = PENETRATION_THRESHOLD_M,
) -> bool:
    penetration = _finite_number(penetration_m, "penetration")
    threshold = _finite_number(threshold_m, "penetration threshold")
    if penetration < 0.0 or threshold < 0.0:
        raise SceneError("penetration and threshold must be nonnegative")
    return penetration > threshold


def _finite_matrix(
    value: object,
    shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    if value is None:
        raise SceneError(f"{label} diagnostics are required")
    source = np.asarray(value)
    if source.shape != shape or source.dtype.kind not in "iuf":
        raise SceneError(f"{label} must have shape {shape}")
    output = np.asarray(source, np.float64)
    if not np.all(np.isfinite(output)):
        raise SceneError(f"{label} must contain only finite values")
    return output


def replay_kinematic_reference(
    scene: RegisteredScene,
    target_joint_names: Sequence[str],
    joint_position: object,
    physical_pelvis_position_mujoco: object,
    physical_pelvis_quaternion_wxyz: object,
    *,
    penetration_threshold_m: float = PENETRATION_THRESHOLD_M,
) -> KinematicReplayReport:
    """Replay named joints and diagnostic physical pelvis through MuJoCo."""

    if not isinstance(scene, RegisteredScene):
        raise SceneError("registered scene is required")
    if tuple(target_joint_names) != tuple(TARGET_JOINT_ORDER):
        raise SceneError("target joint names must equal the pinned 29-name order")
    threshold = _finite_number(
        penetration_threshold_m, "penetration threshold"
    )
    if threshold < 0.0:
        raise SceneError("penetration threshold must be nonnegative")
    positions_source = np.asarray(joint_position)
    if positions_source.ndim != 2 or positions_source.shape[1] != 29:
        raise SceneError("joint_position must have shape [N,29]")
    if positions_source.dtype.kind not in "iuf":
        raise SceneError("joint_position must be numeric")
    positions = np.asarray(positions_source, np.float64)
    if not np.all(np.isfinite(positions)):
        raise SceneError("joint_position must contain only finite values")
    frames = positions.shape[0]
    if frames < 1:
        raise SceneError("kinematic replay requires at least one frame")
    pelvis_position = _finite_matrix(
        physical_pelvis_position_mujoco,
        (frames, 3),
        "physical pelvis position",
    )
    pelvis_quaternion = _finite_matrix(
        physical_pelvis_quaternion_wxyz,
        (frames, 4),
        "physical pelvis quaternion",
    ).copy()
    norms = np.linalg.norm(pelvis_quaternion, axis=1)
    if np.any(norms < 1.0e-12) or np.any(np.abs(norms - 1.0) > 1.0e-5):
        raise SceneError("physical pelvis quaternion must be unit length")
    pelvis_quaternion /= norms[:, None]

    overlay_path, overlay_bytes, overlay_sha = _read_authenticated(
        scene.gear_scene_xml,
        scene.output_hashes.get("gear_scene_xml"),
        "registered GEAR scene XML",
        maximum_bytes=_MAXIMUM_JSON_BYTES,
    )
    overlay_root = _parse_xml_root(overlay_bytes, "registered GEAR scene XML")
    includes = overlay_root.findall("include")
    if (
        overlay_root.tag != "mujoco"
        or len(includes) != 1
        or set(includes[0].attrib) != {"file"}
    ):
        raise SceneError("registered GEAR scene XML include identity changed")
    robot_include = Path(includes[0].attrib["file"])
    if not robot_include.is_absolute():
        raise SceneError("registered robot include must be absolute")
    robot_path, _, robot_sha = _read_authenticated(
        robot_include,
        scene.output_hashes.get("robot_include"),
        "registered robot include",
        maximum_bytes=_MAXIMUM_JSON_BYTES,
    )
    try:
        robot_path.relative_to(overlay_path.parent)
    except ValueError as error:
        raise SceneError("registered robot include escapes the scene output") from error

    if scene.transformed_obj is None:
        transformed_sha: str | None = None
        if "transformed_obj" in scene.output_hashes:
            raise SceneError("flat scene unexpectedly registers a transformed OBJ")
    else:
        transformed_path, _, transformed_sha = _read_authenticated(
            scene.transformed_obj,
            scene.output_hashes.get("transformed_obj"),
            "registered transformed OBJ",
            maximum_bytes=_MAXIMUM_OBJ_BYTES,
        )
        terrain_meshes = overlay_root.findall(
            "./asset/mesh[@name='mm_terrain_mesh']"
        )
        if (
            len(terrain_meshes) != 1
            or set(terrain_meshes[0].attrib) != {"name", "file"}
            or terrain_meshes[0].attrib["file"] != str(transformed_path)
        ):
            raise SceneError("registered transformed OBJ path identity changed")

    registration_path = overlay_path.parent / "scene_registration.json"
    _, registration_bytes, _ = _read_authenticated(
        registration_path,
        scene.output_hashes.get("scene_registration"),
        "registered scene registration",
        maximum_bytes=_MAXIMUM_JSON_BYTES,
    )
    registration = _json_bytes(registration_bytes, "registered scene registration")
    official_scene_value = registration.get("official_scene")
    official_robot_value = registration.get("official_robot_include")
    if (
        type(official_scene_value) is not str
        or not Path(official_scene_value).is_absolute()
        or type(official_robot_value) is not str
        or not Path(official_robot_value).is_absolute()
    ):
        raise SceneError("registered original scene identities changed")
    (
        official_scene_path,
        _,
        official_robot_path,
        _,
        official_scene_sha,
        official_robot_sha,
    ) = _load_official_scene(
        official_scene_value,
        scene.output_hashes.get("official_scene"),
        scene.output_hashes.get("official_robot"),
        label_prefix="registered ",
    )
    if (
        registration.get("gear_scene_sha256") != overlay_sha
        or official_scene_value != str(official_scene_path)
        or registration.get("official_scene_sha256") != official_scene_sha
        or official_robot_value != str(official_robot_path)
        or registration.get("official_robot_include_sha256")
        != official_robot_sha
        or registration.get("robot_include") != str(robot_path)
        or registration.get("robot_include_sha256") != robot_sha
        or registration.get("transformed_obj_sha256") != transformed_sha
    ):
        raise SceneError("registered scene registration identity changed")
    mujoco = _import_mujoco()
    try:
        model = mujoco.MjModel.from_xml_path(str(overlay_path))
    except (ValueError, OSError) as error:
        raise SceneError(f"registered GEAR scene does not load: {error}") from error
    allowed, forbidden = _resolve_contact_groups(model)
    if allowed != scene.allowed_foot_geoms or dict(forbidden) != dict(
        scene.forbidden_geom_groups
    ):
        raise SceneError("registered geom IDs do not match the replay model")
    try:
        terrain_geom = int(model.geom("mm_terrain").id)
    except KeyError as error:
        raise SceneError("replay model has no mm_terrain geom") from error

    free_joints = [
        joint_id
        for joint_id in range(int(model.njnt))
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(free_joints) != 1:
        raise SceneError("replay model must contain exactly one free root")
    root_joint = free_joints[0]
    if (
        model.joint(root_joint).name != "floating_base_joint"
        or model.body(int(model.jnt_bodyid[root_joint])).name != "pelvis"
    ):
        raise SceneError("free root must be floating_base_joint on pelvis")
    root_address = int(model.jnt_qposadr[root_joint])
    joint_addresses: list[int] = []
    for name in TARGET_JOINT_ORDER:
        try:
            joint_id = int(model.joint(name).id)
        except KeyError as error:
            raise SceneError(f"replay model is missing target joint {name}") from error
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise SceneError(f"target joint {name} is not a hinge")
        joint_addresses.append(int(model.jnt_qposadr[joint_id]))
    if len(set(joint_addresses)) != 29:
        raise SceneError("target joint qpos addresses are not unique")

    forbidden_lookup: dict[int, str] = {}
    for group_name, geom_ids in forbidden.items():
        for geom_id in geom_ids:
            forbidden_lookup[geom_id] = group_name
    allowed_set = set(allowed)
    qpos_rows = np.empty((frames, int(model.nq)), np.float64)
    contacts: list[ReplayContact] = []
    maximum_forbidden = 0.0
    violation = False
    data = mujoco.MjData(model)
    for frame in range(frames):
        data.qpos[:] = model.qpos0
        data.qpos[root_address : root_address + 3] = pelvis_position[frame]
        data.qpos[root_address + 3 : root_address + 7] = pelvis_quaternion[frame]
        data.qpos[np.asarray(joint_addresses, np.int64)] = positions[frame]
        qpos_rows[frame] = data.qpos
        mujoco.mj_forward(model, data)
        mujoco.mj_collision(model, data)
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
            distance = float(contact.dist)
            penetration = max(0.0, -distance)
            if abs(penetration - threshold) <= 1.0e-12:
                penetration = threshold
            if robot_geom in allowed_set:
                group = "allowed_feet"
            else:
                group = forbidden_lookup.get(robot_geom, "other")
            contacts.append(
                ReplayContact(
                    frame_index=frame,
                    terrain_geom=terrain_geom,
                    robot_geom=robot_geom,
                    group=group,
                    distance_m=distance,
                    penetration_m=penetration,
                )
            )
            if group in forbidden:
                maximum_forbidden = max(maximum_forbidden, penetration)
                if penetration_exceeds_threshold(penetration, threshold):
                    violation = True
    qpos_rows.flags.writeable = False
    return KinematicReplayReport(
        frame_count=frames,
        qpos=qpos_rows,
        contacts=tuple(contacts),
        allowed_foot_contacts=any(
            contact.group == "allowed_feet" for contact in contacts
        ),
        maximum_forbidden_penetration_m=maximum_forbidden,
        forbidden_penetration=violation,
        threshold_m=threshold,
    )
