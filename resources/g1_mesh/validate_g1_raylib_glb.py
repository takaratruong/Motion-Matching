"""Dependency-free structural validation for the certified G1 Raylib GLB."""

import hashlib
import json
import math
import pathlib
import struct


GLTF_JSON_CHUNK = 0x4E4F534A
GLTF_BIN_CHUNK = 0x004E4942
UNSIGNED_BYTE = 5121
UNSIGNED_SHORT = 5123
FLOAT = 5126

SOURCE_SIZE = 11348604
SOURCE_SHA256 = (
    "1546cb574d0c9296f8200c8df8f75d1618e89bf43a6b46b8414fdfc852a4815b"
)
BLENDER_VERSION = "4.0.2"
MAX_VERTICES = 65535
RIGID_WEIGHT_TOLERANCE = 1e-6

REQUIRED_ARTICULATED_BONES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

REQUIRED_ARTICULATED_PARENTS = {
    "pelvis": None,
    "left_hip_pitch_link": "pelvis",
    "left_hip_roll_link": "left_hip_pitch_link",
    "left_hip_yaw_link": "left_hip_roll_link",
    "left_knee_link": "left_hip_yaw_link",
    "left_ankle_pitch_link": "left_knee_link",
    "left_ankle_roll_link": "left_ankle_pitch_link",
    "right_hip_pitch_link": "pelvis",
    "right_hip_roll_link": "right_hip_pitch_link",
    "right_hip_yaw_link": "right_hip_roll_link",
    "right_knee_link": "right_hip_yaw_link",
    "right_ankle_pitch_link": "right_knee_link",
    "right_ankle_roll_link": "right_ankle_pitch_link",
    "waist_yaw_link": "pelvis",
    "waist_roll_link": "waist_yaw_link",
    "torso_link": "waist_roll_link",
    "left_shoulder_pitch_link": "torso_link",
    "left_shoulder_roll_link": "left_shoulder_pitch_link",
    "left_shoulder_yaw_link": "left_shoulder_roll_link",
    "left_elbow_link": "left_shoulder_yaw_link",
    "left_wrist_roll_link": "left_elbow_link",
    "left_wrist_pitch_link": "left_wrist_roll_link",
    "left_wrist_yaw_link": "left_wrist_pitch_link",
    "right_shoulder_pitch_link": "torso_link",
    "right_shoulder_roll_link": "right_shoulder_pitch_link",
    "right_shoulder_yaw_link": "right_shoulder_roll_link",
    "right_elbow_link": "right_shoulder_yaw_link",
    "right_wrist_roll_link": "right_elbow_link",
    "right_wrist_pitch_link": "right_wrist_roll_link",
    "right_wrist_yaw_link": "right_wrist_pitch_link",
}

EVIDENCE_FIELDS = (
    "bone_count",
    "skin_count",
    "vertex_count",
    "primitive_count",
    "maximum_primitive_vertices",
    "rigid_vertex_count",
    "height_m",
    "glb_sha256",
)

MANIFEST_FIELDS = frozenset(
    EVIDENCE_FIELDS
    + (
        "blender_version",
        "glb_size",
        "skin_names",
        "required_articulated_bones",
        "source_size",
        "source_sha256",
    )
)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path, label):
    try:
        return pathlib.Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read {label}") from error


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_document(data, label):
    try:
        document = json.loads(data, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"invalid {label} JSON root")
    return document


def _chunks(data):
    if len(data) < 12:
        raise ValueError("invalid GLB header")
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or total != len(data):
        raise ValueError("invalid GLB header")
    offset = 12
    result = {}
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError("invalid GLB chunks")
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        end = offset + length
        if end > len(data) or kind in result:
            raise ValueError("invalid GLB chunks")
        result[kind] = data[offset:end]
        offset = end
    if offset != len(data) or GLTF_JSON_CHUNK not in result:
        raise ValueError("invalid GLB chunks")
    return result


def _accessor_bytes(document, binary, index):
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride")
    return accessor, memoryview(binary)[start:], stride


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid {label}")
    return value


def _array(value, label):
    if not isinstance(value, list):
        raise ValueError(f"invalid {label}")
    return value


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}")
    return value


def _accessor_layout(document, binary, index, components, component_size, label):
    accessors = _array(document.get("accessors"), "accessors")
    views = _array(document.get("bufferViews"), "bufferViews")
    index = _integer(index, f"{label} accessor index")
    if index >= len(accessors):
        raise ValueError(f"invalid {label} accessor index")
    accessor = _object(accessors[index], f"{label} accessor")
    if "sparse" in accessor:
        raise ValueError(f"sparse {label} accessor is unsupported")
    view_index = _integer(accessor.get("bufferView"), f"{label} buffer view")
    if view_index >= len(views):
        raise ValueError(f"invalid {label} buffer view")
    view = _object(views[view_index], f"{label} buffer view")
    if _integer(view.get("buffer", 0), f"{label} buffer") != 0:
        raise ValueError(f"invalid {label} buffer")
    view_offset = _integer(view.get("byteOffset", 0), f"{label} view offset")
    view_length = _integer(view.get("byteLength"), f"{label} view length")
    accessor_offset = _integer(
        accessor.get("byteOffset", 0), f"{label} accessor offset"
    )
    count = _integer(accessor.get("count"), f"{label} count", minimum=1)
    element_size = components * component_size
    stride_value = view.get("byteStride")
    if stride_value is None:
        stride_value = element_size
    else:
        stride_value = _integer(stride_value, f"{label} stride", minimum=1)
    if stride_value < element_size or stride_value % component_size != 0:
        raise ValueError(f"invalid {label} stride")
    end_in_view = accessor_offset + (count - 1) * stride_value + element_size
    if end_in_view > view_length or view_offset + view_length > len(binary):
        raise ValueError(f"truncated {label} accessor")
    if (view_offset + accessor_offset) % component_size != 0:
        raise ValueError(f"misaligned {label} accessor")
    stored_accessor, values, stored_stride = _accessor_bytes(
        document, binary, index
    )
    if stored_accessor is not accessor or stored_stride != view.get("byteStride"):
        raise ValueError(f"invalid {label} accessor storage")
    return accessor, values, stride_value


def _node_parents(nodes):
    parents = [-1] * len(nodes)
    for parent_index, node_value in enumerate(nodes):
        node = _object(node_value, "node")
        children = node.get("children", [])
        children = _array(children, "node children")
        seen = set()
        for child_value in children:
            child = _integer(child_value, "child node")
            if child >= len(nodes) or child in seen or parents[child] != -1:
                raise ValueError("invalid GLB node hierarchy")
            seen.add(child)
            parents[child] = parent_index
    for start in range(len(nodes)):
        seen = set()
        node = start
        while node != -1:
            if node in seen:
                raise ValueError("cyclic GLB node hierarchy")
            seen.add(node)
            node = parents[node]
    return parents


def _joint_parent(node_index, joint_indices, parents):
    parent = parents[node_index]
    while parent != -1 and parent not in joint_indices:
        parent = parents[parent]
    return parent


def _position_bounds(accessor):
    minimum = accessor.get("min")
    maximum = accessor.get("max")
    if not isinstance(minimum, list) or not isinstance(maximum, list):
        raise ValueError("POSITION accessor bounds are required")
    if len(minimum) != 3 or len(maximum) != 3:
        raise ValueError("invalid POSITION accessor bounds")
    result_minimum = []
    result_maximum = []
    for low, high in zip(minimum, maximum):
        if (
            isinstance(low, bool)
            or isinstance(high, bool)
            or not isinstance(low, (int, float))
            or not isinstance(high, (int, float))
        ):
            raise ValueError("invalid POSITION accessor bounds")
        low = float(low)
        high = float(high)
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError("invalid POSITION accessor bounds")
        result_minimum.append(low)
        result_maximum.append(high)
    return result_minimum, result_maximum


def _inspect_g1_glb(data):
    chunks = _chunks(data)
    if GLTF_BIN_CHUNK not in chunks:
        raise ValueError("GLB binary chunk is required")
    document = _json_document(chunks[GLTF_JSON_CHUNK], "GLB")
    binary = chunks[GLTF_BIN_CHUNK]

    buffers = _array(document.get("buffers"), "buffers")
    if len(buffers) != 1:
        raise ValueError("GLB must contain exactly one binary buffer")
    buffer = _object(buffers[0], "buffer")
    if "uri" in buffer:
        raise ValueError("external GLB buffers are unsupported")
    binary_length = _integer(buffer.get("byteLength"), "buffer length")
    if binary_length > len(binary) or len(binary) - binary_length > 3:
        raise ValueError("invalid GLB binary length")

    nodes = _array(document.get("nodes"), "nodes")
    skins = _array(document.get("skins"), "skins")
    if len(skins) != 1:
        raise ValueError("G1 GLB must contain exactly one skin")
    skin = _object(skins[0], "skin")
    joints = _array(skin.get("joints"), "skin joints")
    if len(joints) != 39:
        raise ValueError("G1 skin must contain exactly 39 bones")
    joint_nodes = []
    for value in joints:
        index = _integer(value, "joint node")
        if index >= len(nodes) or index in joint_nodes:
            raise ValueError("invalid or duplicate skin joint")
        joint_nodes.append(index)
    skin_names = []
    for index in joint_nodes:
        name = _object(nodes[index], "joint node").get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("skin joints require names")
        skin_names.append(name)
    if len(set(skin_names)) != 39:
        raise ValueError("G1 skin joint names must be unique")

    parents = _node_parents(nodes)
    joint_indices = set(joint_nodes)
    joint_by_name = dict(zip(skin_names, joint_nodes))
    for name in REQUIRED_ARTICULATED_BONES:
        if name not in joint_by_name:
            raise ValueError(f"missing required articulated bone: {name}")
        expected_name = REQUIRED_ARTICULATED_PARENTS[name]
        parent_index = _joint_parent(
            joint_by_name[name], joint_indices, parents
        )
        expected_index = (
            -1 if expected_name is None else joint_by_name.get(expected_name, -2)
        )
        if parent_index != expected_index:
            raise ValueError(f"wrong articulated parent for {name}")

    meshes = _array(document.get("meshes"), "meshes")
    if not meshes:
        raise ValueError("G1 GLB contains no meshes")
    vertex_count = 0
    rigid_vertex_count = 0
    maximum_primitive_vertices = 0
    primitive_count = 0
    bounds_minimum = [math.inf, math.inf, math.inf]
    bounds_maximum = [-math.inf, -math.inf, -math.inf]

    for mesh_value in meshes:
        mesh = _object(mesh_value, "mesh")
        primitives = _array(mesh.get("primitives"), "mesh primitives")
        if not primitives:
            raise ValueError("G1 mesh contains no primitives")
        for primitive_value in primitives:
            primitive_count += 1
            primitive = _object(primitive_value, "mesh primitive")
            if primitive.get("mode", 4) != 4:
                raise ValueError("G1 mesh primitive must use triangles")
            attributes = _object(primitive.get("attributes"), "attributes")
            try:
                position_index = attributes["POSITION"]
                joints_index = attributes["JOINTS_0"]
                weights_index = attributes["WEIGHTS_0"]
                indices_index = primitive["indices"]
            except KeyError as error:
                raise ValueError("G1 primitive is missing skin attributes") from error

            position, _, _ = _accessor_layout(
                document, binary, position_index, 3, 4, "POSITION"
            )
            if position.get("componentType") != FLOAT or position.get("type") != "VEC3":
                raise ValueError("POSITION must be float VEC3")
            position_count = position["count"]
            if position_count > MAX_VERTICES:
                raise ValueError("G1 primitive exceeds Raylib u16 vertex limit")

            joint, _, _ = _accessor_layout(
                document,
                binary,
                joints_index,
                4,
                1
                if _object(
                    _array(document.get("accessors"), "accessors")[joints_index],
                    "JOINTS_0 accessor",
                ).get("componentType")
                == UNSIGNED_BYTE
                else 2,
                "JOINTS_0",
            )
            if (
                joint.get("componentType") not in (UNSIGNED_BYTE, UNSIGNED_SHORT)
                or joint.get("type") != "VEC4"
            ):
                raise ValueError("JOINTS_0 must be unsigned-byte/short VEC4")
            if joint.get("count") != position_count:
                raise ValueError("JOINTS_0 count does not match POSITION")

            weights, weight_bytes, weight_stride = _accessor_layout(
                document, binary, weights_index, 4, 4, "WEIGHTS_0"
            )
            if weights.get("componentType") != FLOAT or weights.get("type") != "VEC4":
                raise ValueError("WEIGHTS_0 must be float VEC4")
            if weights.get("count") != position_count:
                raise ValueError("WEIGHTS_0 count does not match POSITION")

            indices, _, _ = _accessor_layout(
                document, binary, indices_index, 1, 2, "indices"
            )
            if (
                indices.get("componentType") != UNSIGNED_SHORT
                or indices.get("type") != "SCALAR"
            ):
                raise ValueError("indices must be unsigned-short SCALAR")

            minimum, maximum = _position_bounds(position)
            for axis in range(3):
                bounds_minimum[axis] = min(bounds_minimum[axis], minimum[axis])
                bounds_maximum[axis] = max(bounds_maximum[axis], maximum[axis])

            for vertex in range(position_count):
                values = struct.unpack_from("<4f", weight_bytes, vertex * weight_stride)
                one_count = sum(
                    abs(value - 1.0) <= RIGID_WEIGHT_TOLERANCE
                    for value in values
                )
                zero_count = sum(
                    abs(value) <= RIGID_WEIGHT_TOLERANCE for value in values
                )
                if (
                    any(not math.isfinite(value) for value in values)
                    or one_count != 1
                    or zero_count != 3
                ):
                    raise ValueError("G1 GLB contains non-rigid skin weights")
                rigid_vertex_count += 1

            vertex_count += position_count
            maximum_primitive_vertices = max(
                maximum_primitive_vertices, position_count
            )

    if primitive_count != 35:
        raise ValueError("G1 GLB must contain exactly 35 rigid-link primitives")
    if vertex_count == 0:
        raise ValueError("G1 GLB contains no skinned vertices")
    height_m = max(
        high - low for low, high in zip(bounds_minimum, bounds_maximum)
    )
    if height_m < 0.8 or height_m > 1.6:
        raise ValueError("G1 GLB bounding-box extent is outside [0.8, 1.6] m")

    return {
        "bone_count": len(joints),
        "skin_count": len(skins),
        "vertex_count": vertex_count,
        "primitive_count": primitive_count,
        "maximum_primitive_vertices": maximum_primitive_vertices,
        "rigid_vertex_count": rigid_vertex_count,
        "height_m": height_m,
        "glb_sha256": _sha256(data),
        "skin_names": skin_names,
    }


def inspect_g1_glb(glb_path: pathlib.Path) -> dict[str, object]:
    """Return structural evidence for a certified G1 GLB or raise ValueError."""

    data = _read_bytes(glb_path, "GLB")
    try:
        return _inspect_g1_glb(data)
    except ValueError:
        raise
    except (IndexError, KeyError, TypeError, struct.error) as error:
        raise ValueError("invalid G1 GLB structure") from error


def validate_g1_glb(
    glb_path: pathlib.Path,
    manifest_path: pathlib.Path,
    source_path: pathlib.Path | None = None,
) -> dict[str, object]:
    """Return certified counts and bounds or raise ValueError."""

    report = inspect_g1_glb(glb_path)
    manifest = _json_document(_read_bytes(manifest_path, "manifest"), "manifest")
    if set(manifest) != MANIFEST_FIELDS:
        raise ValueError("G1 mesh manifest fields mismatch")

    for field in EVIDENCE_FIELDS:
        if manifest.get(field) != report[field]:
            if field == "glb_sha256":
                raise ValueError("GLB SHA-256 mismatch")
            raise ValueError(f"manifest {field} mismatch")
    glb_size = manifest.get("glb_size")
    if (
        isinstance(glb_size, bool)
        or not isinstance(glb_size, int)
        or glb_size != pathlib.Path(glb_path).stat().st_size
    ):
        raise ValueError("GLB size mismatch")
    if manifest.get("skin_names") != report["skin_names"]:
        raise ValueError("manifest skin names mismatch")
    if manifest.get("required_articulated_bones") != list(
        REQUIRED_ARTICULATED_BONES
    ):
        raise ValueError("manifest articulated bone names mismatch")
    if manifest.get("blender_version") != BLENDER_VERSION:
        raise ValueError("manifest Blender version mismatch")
    if manifest.get("source_size") != SOURCE_SIZE:
        raise ValueError("manifest source size mismatch")
    if manifest.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("manifest source SHA-256 mismatch")

    if source_path is not None:
        source_data = _read_bytes(source_path, "G1 FBX source")
        if len(source_data) != manifest["source_size"]:
            raise ValueError("G1 FBX source size mismatch")
        if _sha256(source_data) != manifest["source_sha256"]:
            raise ValueError("G1 FBX source SHA-256 mismatch")
    return report
