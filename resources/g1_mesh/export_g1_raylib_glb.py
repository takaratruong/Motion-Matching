"""Deterministically export the authenticated G1 FBX for Raylib."""

import argparse
import hashlib
import json
import math
import pathlib
import struct
import sys


SOURCE_SHA256 = "1546cb574d0c9296f8200c8df8f75d1618e89bf43a6b46b8414fdfc852a4815b"
MAX_VERTICES = 65535
TARGET_VERTICES = 52000
RIGID_GROUP_COUNT = 35
RIGID_WEIGHT_TOLERANCE = 1e-6
NONZERO_WEIGHT_TOLERANCE = 1e-8

# The motion database stores exact MuJoCo link-frame rotations, re-expressed
# from Z-up to Y-up by conjugation in g1_to_bvh.py.  The source FBX armature
# instead aimed each Blender bone from its link origin toward a convenient
# child/leaf tail.  Those head-to-tail bind axes differ from the physical link
# axes by 110-180 degrees and cannot consume the accepted global rotations
# directly.  Rebase the exported armature to the authoritative zero-pose link
# frames while leaving the rigid geometry untouched.
PRODUCTION_ZERO_POSE_IDENTITY = (0.0, 0.0, 0.0, 1.0)
PRODUCTION_ZERO_POSE_HIP = (
    0.0,
    0.0,
    0.08733857244071258,
    0.996178685660368,
)
PRODUCTION_ZERO_POSE_LEFT_SHOULDER = (
    0.13920101962536002,
    -0.00009868681391343435,
    -0.000013872201955782791,
    0.9902641396131313,
)
PRODUCTION_ZERO_POSE_LEFT_ARM = (
    0.000029274470605633596,
    -0.00009579579294906673,
    -0.000027471645031675433,
    0.9999999946057401,
)
PRODUCTION_ZERO_POSE_RIGHT_SHOULDER = (
    -0.13920101962536002,
    0.00009868681391343435,
    -0.000013872201955782777,
    0.9902641396131312,
)
PRODUCTION_ZERO_POSE_RIGHT_ARM = (
    -0.000029274470605633596,
    0.00009579579294906676,
    -0.0000274716450316754,
    0.9999999946057401,
)

PRODUCTION_ZERO_POSE_POSITIONS = {
    "pelvis": (0.0, 0.0, 0.0),
    "left_hip_pitch_link": (0.0, -0.1027, -0.064452),
    "left_hip_roll_link": (0.0, -0.133165, -0.116452),
    "left_hip_yaw_link": (0.046217661, -0.251041004, -0.116452),
    "left_knee_link": (-0.000002331, -0.439295752, -0.1186009),
    "left_ankle_pitch_link": (
        -0.000002331, -0.739305752, -0.118506455,
    ),
    "left_ankle_roll_link": (
        -0.000002331, -0.756863752, -0.118506455,
    ),
    "right_hip_pitch_link": (0.0, -0.1027, 0.064452),
    "right_hip_roll_link": (0.0, -0.133165, 0.116452),
    "right_hip_yaw_link": (0.046217661, -0.251041004, 0.116452),
    "right_knee_link": (-0.000002331, -0.439295752, 0.1186009),
    "right_ankle_pitch_link": (
        -0.000002331, -0.739305752, 0.118506455,
    ),
    "right_ankle_roll_link": (
        -0.000002331, -0.756863752, 0.118506455,
    ),
    "waist_yaw_link": (0.0, 0.0, 0.0),
    "waist_roll_link": (-0.0039635, 0.035, 0.0),
    "torso_link": (-0.0039635, 0.054, 0.0),
    "left_shoulder_pitch_link": (-0.0000072, 0.29178, -0.10022),
    "left_shoulder_roll_link": (
        0.000000374, 0.288961283, -0.140560443,
    ),
    "left_shoulder_yaw_link": (
        -0.0000041, 0.185761649, -0.146806486,
    ),
    "left_elbow_link": (0.015774476, 0.105242782, -0.146808176),
    "left_wrist_roll_link": (
        0.115774286, 0.095237398, -0.148677513,
    ),
    "left_wrist_pitch_link": (
        0.153774286, 0.09523531, -0.148670233,
    ),
    "left_wrist_yaw_link": (
        0.199774285, 0.095232782, -0.148661419,
    ),
    "right_shoulder_pitch_link": (-0.0000072, 0.29178, 0.10021),
    "right_shoulder_roll_link": (
        0.000000374, 0.288961283, 0.140550443,
    ),
    "right_shoulder_yaw_link": (
        -0.0000041, 0.185761649, 0.146796486,
    ),
    "right_elbow_link": (0.015774476, 0.105242782, 0.146798176),
    "right_wrist_roll_link": (
        0.115774286, 0.095237398, 0.148667513,
    ),
    "right_wrist_pitch_link": (
        0.153774286, 0.09523531, 0.148660233,
    ),
    "right_wrist_yaw_link": (
        0.199774285, 0.095232782, 0.148651419,
    ),
}


def production_zero_pose_rotations():
    output = {
        name: PRODUCTION_ZERO_POSE_IDENTITY
        for name in (
            "pelvis",
            "left_hip_pitch_link",
            "left_knee_link",
            "left_ankle_pitch_link",
            "left_ankle_roll_link",
            "right_hip_pitch_link",
            "right_knee_link",
            "right_ankle_pitch_link",
            "right_ankle_roll_link",
            "waist_yaw_link",
            "waist_roll_link",
            "torso_link",
        )
    }
    for name in (
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
    ):
        output[name] = PRODUCTION_ZERO_POSE_HIP
    output["left_shoulder_pitch_link"] = (
        PRODUCTION_ZERO_POSE_LEFT_SHOULDER
    )
    output["right_shoulder_pitch_link"] = (
        PRODUCTION_ZERO_POSE_RIGHT_SHOULDER
    )
    for name in (
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
    ):
        output[name] = PRODUCTION_ZERO_POSE_LEFT_ARM
    for name in (
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ):
        output[name] = PRODUCTION_ZERO_POSE_RIGHT_ARM
    if len(output) != 30:
        raise RuntimeError("production G1 bind reference must contain 30 links")
    return output


def _quaternion_normalize(value):
    length = math.sqrt(sum(component*component for component in value))
    if not math.isfinite(length) or length <= 0.0:
        raise RuntimeError("G1 bind quaternion is invalid")
    return tuple(component/length for component in value)


def _quaternion_conjugate(value):
    return (-value[0], -value[1], -value[2], value[3])


def _quaternion_multiply(first, second):
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    return _quaternion_normalize((
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ))


def _quaternion_rotate(rotation, value):
    x, y, z, w = rotation
    vx, vy, vz = value
    tx = 2.0*(y*vz - z*vy)
    ty = 2.0*(z*vx - x*vz)
    tz = 2.0*(x*vy - y*vx)
    return (
        vx + w*tx + (y*tz - z*ty),
        vy + w*ty + (z*tx - x*tz),
        vz + w*tz + (x*ty - y*tx),
    )


def _node_globals(document):
    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        raise RuntimeError("G1 GLB nodes are missing")
    parents = [-1] * len(nodes)
    for parent, node in enumerate(nodes):
        if "matrix" in node:
            raise RuntimeError("G1 GLB matrix nodes are unsupported")
        for child in node.get("children", []):
            if child < 0 or child >= len(nodes) or parents[child] != -1:
                raise RuntimeError("G1 GLB node hierarchy is invalid")
            parents[child] = parent

    positions = [None] * len(nodes)
    rotations = [None] * len(nodes)

    def visit(index):
        if positions[index] is not None:
            return positions[index], rotations[index]
        node = nodes[index]
        local_position = tuple(node.get("translation", (0.0, 0.0, 0.0)))
        local_rotation = _quaternion_normalize(
            tuple(node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
        )
        if len(local_position) != 3 or len(local_rotation) != 4:
            raise RuntimeError("G1 GLB node transform is invalid")
        parent = parents[index]
        if parent < 0:
            positions[index] = local_position
            rotations[index] = local_rotation
        else:
            parent_position, parent_rotation = visit(parent)
            rotated = _quaternion_rotate(parent_rotation, local_position)
            positions[index] = tuple(
                first + second
                for first, second in zip(parent_position, rotated)
            )
            rotations[index] = _quaternion_multiply(
                parent_rotation,
                local_rotation,
            )
        return positions[index], rotations[index]

    for index in range(len(nodes)):
        visit(index)
    return parents, positions, rotations


def _rigid_inverse_matrix(position, rotation):
    x, y, z, w = _quaternion_normalize(rotation)
    rotation_matrix = (
        (1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w), 2.0*(x*z + y*w)),
        (2.0*(x*y + z*w), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)),
        (2.0*(x*z - y*w), 2.0*(y*z + x*w), 1.0 - 2.0*(x*x + y*y)),
    )
    inverse_rotation = tuple(zip(*rotation_matrix))
    inverse_translation = tuple(
        -sum(inverse_rotation[row][axis]*position[axis] for axis in range(3))
        for row in range(3)
    )
    matrix = (
        (*inverse_rotation[0], inverse_translation[0]),
        (*inverse_rotation[1], inverse_translation[1]),
        (*inverse_rotation[2], inverse_translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )
    return tuple(matrix[row][column] for column in range(4) for row in range(4))


def _read_glb(path):
    data = path.read_bytes()
    if len(data) < 20:
        raise RuntimeError("G1 GLB is truncated")
    magic, version, total_length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2 or total_length != len(data):
        raise RuntimeError("G1 GLB header is invalid")
    offset = 12
    document = None
    binary = None
    while offset < len(data):
        if offset + 8 > len(data):
            raise RuntimeError("G1 GLB chunk header is truncated")
        length, kind = struct.unpack_from("<II", data, offset)
        begin = offset + 8
        end = begin + length
        if end > len(data):
            raise RuntimeError("G1 GLB chunk is truncated")
        payload = data[begin:end]
        if kind == 0x4E4F534A:
            if document is not None:
                raise RuntimeError("G1 GLB has duplicate JSON chunks")
            document = json.loads(payload.rstrip(b" \\x00"))
        elif kind == 0x004E4942:
            if binary is not None:
                raise RuntimeError("G1 GLB has duplicate BIN chunks")
            binary = bytearray(payload)
        else:
            raise RuntimeError("G1 GLB contains an unsupported chunk")
        offset = end
    if document is None or binary is None:
        raise RuntimeError("G1 GLB requires JSON and BIN chunks")
    return document, binary


def _write_glb(path, document, binary):
    json_bytes = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary = bytes(binary)
    if len(binary) % 4 != 0:
        raise RuntimeError("G1 GLB BIN chunk is not four-byte aligned")
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary)
    output = bytearray(struct.pack("<III", 0x46546C67, 2, total_length))
    output.extend(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    output.extend(json_bytes)
    output.extend(struct.pack("<II", len(binary), 0x004E4942))
    output.extend(binary)
    path.write_bytes(output)


def rebase_glb_to_production_links(path):
    rotations = production_zero_pose_rotations()
    positions = PRODUCTION_ZERO_POSE_POSITIONS
    if set(positions) != set(rotations):
        raise RuntimeError("G1 bind positions and rotations must name the same links")
    document, binary = _read_glb(path)
    nodes = document.get("nodes")
    skins = document.get("skins")
    if not isinstance(skins, list) or len(skins) != 1:
        raise RuntimeError("G1 GLB requires one skin")
    skin = skins[0]
    joints = skin.get("joints")
    if not isinstance(joints, list) or len(joints) != 39:
        raise RuntimeError("G1 GLB requires 39 skin joints")

    parents, original_positions, original_rotations = _node_globals(document)
    joints_by_name = {nodes[index].get("name"): index for index in joints}
    missing = sorted(set(rotations) - set(joints_by_name))
    if missing:
        raise RuntimeError(f"G1 GLB is missing mapped bones: {missing}")

    target_positions = {
        index: original_positions[index]
        for index in joints
    }
    target_rotations = {
        index: original_rotations[index]
        for index in joints
    }
    for name in rotations:
        index = joints_by_name[name]
        target_positions[index] = tuple(positions[name])
        target_rotations[index] = _quaternion_normalize(rotations[name])

    joint_set = set(joints)
    for index in joints:
        parent = parents[index]
        if parent in joint_set:
            parent_position = target_positions[parent]
            parent_rotation = target_rotations[parent]
        elif parent >= 0:
            parent_position = original_positions[parent]
            parent_rotation = original_rotations[parent]
        else:
            parent_position = (0.0, 0.0, 0.0)
            parent_rotation = (0.0, 0.0, 0.0, 1.0)
        inverse_parent = _quaternion_conjugate(parent_rotation)
        delta = tuple(
            target_positions[index][axis] - parent_position[axis]
            for axis in range(3)
        )
        local_position = _quaternion_rotate(inverse_parent, delta)
        local_rotation = _quaternion_multiply(
            inverse_parent,
            target_rotations[index],
        )
        node = nodes[index]
        node.pop("matrix", None)
        node["translation"] = list(local_position)
        node["rotation"] = list(local_rotation)
        node["scale"] = [1.0, 1.0, 1.0]

    accessors = document.get("accessors")
    views = document.get("bufferViews")
    accessor_index = skin.get("inverseBindMatrices")
    if (
        not isinstance(accessors, list) or
        not isinstance(views, list) or
        not isinstance(accessor_index, int) or
        accessor_index < 0 or accessor_index >= len(accessors)
    ):
        raise RuntimeError("G1 inverse-bind accessor is missing")
    accessor = accessors[accessor_index]
    if (
        accessor.get("componentType") != 5126 or
        accessor.get("type") != "MAT4" or
        accessor.get("count") != len(joints) or
        "sparse" in accessor
    ):
        raise RuntimeError("G1 inverse-bind accessor is unsupported")
    view_index = accessor.get("bufferView")
    if not isinstance(view_index, int) or view_index < 0 or view_index >= len(views):
        raise RuntimeError("G1 inverse-bind buffer view is missing")
    view = views[view_index]
    buffer_index = view.get("buffer", 0)
    if type(buffer_index) is not int or buffer_index != 0:
        raise RuntimeError("G1 inverse binds must use the GLB BIN chunk")
    view_offset = view.get("byteOffset", 0)
    view_length = view.get("byteLength")
    accessor_offset = accessor.get("byteOffset", 0)
    stride = view.get("byteStride", 64)
    layout_values = (view_offset, view_length, accessor_offset, stride)
    if any(type(value) is not int or value < 0 for value in layout_values):
        raise RuntimeError("G1 inverse-bind layout is invalid")
    if any(value % 4 != 0 for value in layout_values):
        raise RuntimeError("G1 inverse-bind layout is not four-byte aligned")
    if stride < 64:
        raise RuntimeError("G1 inverse-bind stride is invalid")
    if view_offset + view_length > len(binary):
        raise RuntimeError("G1 inverse-bind buffer view is out of bounds")
    end_in_view = accessor_offset + (len(joints) - 1)*stride + 64
    if end_in_view > view_length:
        raise RuntimeError(
            "G1 inverse-bind payload is outside its buffer view")
    begin = view_offset + accessor_offset
    for slot, index in enumerate(joints):
        offset = begin + slot*stride
        struct.pack_into(
            "<16f",
            binary,
            offset,
            *_rigid_inverse_matrix(
                target_positions[index],
                target_rotations[index],
            ),
        )

    # Recompute from the serialized node values and fail before writing if a
    # mapped transform or fixed attachment global was not preserved.
    _, rebased_positions, rebased_rotations = _node_globals(document)
    for index in joints:
        position_error = math.sqrt(sum(
            (rebased_positions[index][axis] - target_positions[index][axis])**2
            for axis in range(3)
        ))
        rotation_dot = abs(sum(
            rebased_rotations[index][axis]*target_rotations[index][axis]
            for axis in range(4)
        ))
        if position_error > 1e-6 or 1.0 - min(1.0, rotation_dot) > 1e-7:
            raise RuntimeError("G1 GLB bind rebase did not preserve a joint global")

    _write_glb(path, document, binary)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_rigid_groups(mesh):
    import bpy

    groups = list(mesh.vertex_groups)
    if len(groups) != RIGID_GROUP_COUNT:
        raise RuntimeError("decimated G1 mesh must contain 35 vertex groups")

    group_indices = {group.index for group in groups}
    vertex_counts = {group.index: 0 for group in groups}
    assignments = []
    for vertex in mesh.data.vertices:
        if any(not math.isfinite(value.weight) for value in vertex.groups):
            raise RuntimeError("decimated G1 mesh contains non-finite weights")
        weighted = [
            value
            for value in vertex.groups
            if value.weight > NONZERO_WEIGHT_TOLERANCE
        ]
        if (
            len(weighted) != 1
            or weighted[0].group not in group_indices
            or abs(weighted[0].weight - 1.0) > RIGID_WEIGHT_TOLERANCE
        ):
            raise RuntimeError("decimated G1 mesh contains non-rigid weights")
        group_index = weighted[0].group
        assignments.append(group_index)
        vertex_counts[group_index] += 1
    if any(count == 0 for count in vertex_counts.values()):
        raise RuntimeError("decimated G1 mesh contains an empty vertex group")

    corner_counts = {group.index: 0 for group in groups}
    for polygon in mesh.data.polygons:
        polygon_groups = {assignments[index] for index in polygon.vertices}
        if len(polygon_groups) != 1:
            raise RuntimeError("decimated G1 polygon spans rigid vertex groups")
        group_index = next(iter(polygon_groups))
        corner_counts[group_index] += polygon.loop_total
    if any(count == 0 for count in corner_counts.values()):
        raise RuntimeError("decimated G1 vertex group contains no polygons")
    if any(count > MAX_VERTICES for count in corner_counts.values()):
        raise RuntimeError("G1 rigid group exceeds Raylib u16 vertex limit")

    original_data = mesh.data
    collections = tuple(mesh.users_collection)
    if not collections:
        raise RuntimeError("decimated G1 mesh is not linked to a collection")
    outputs = []
    for group in sorted(groups, key=lambda value: value.name):
        value = mesh.copy()
        value.data = mesh.data.copy()
        value.name = f"G1_{group.name}"
        value.data.name = f"G1_{group.name}"
        for collection in collections:
            collection.objects.link(value)
        bpy.ops.object.select_all(action="DESELECT")
        value.select_set(True)
        bpy.context.view_layer.objects.active = value
        value.vertex_groups.active_index = group.index
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.vertex_group_select()
        bpy.ops.mesh.select_all(action="INVERT")
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")
        if not value.data.vertices or not value.data.polygons:
            raise RuntimeError("G1 rigid group split produced an empty mesh")
        outputs.append(value)
    bpy.data.objects.remove(mesh, do_unlink=True)
    bpy.data.meshes.remove(original_data)
    return outputs


def export(source, output):
    import bpy

    if source.stat().st_size != 11348604 or sha256(source) != SOURCE_SHA256:
        raise RuntimeError("G1 FBX source identity mismatch")
    if (bpy.context.scene.render.threads_mode != "FIXED" or
            bpy.context.scene.render.threads != 1):
        raise RuntimeError("G1 export requires Blender --threads 1")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(source))
    armatures = [value for value in bpy.data.objects if value.type == "ARMATURE"]
    meshes = [value for value in bpy.data.objects if value.type == "MESH"]
    if len(armatures) != 1 or len(meshes) != 1:
        raise RuntimeError("G1 FBX must contain one armature and one mesh")
    mesh = meshes[0]
    original_vertices = len(mesh.data.vertices)
    if original_vertices != 196692:
        raise RuntimeError("G1 FBX vertex count mismatch")
    modifier = mesh.modifiers.new("raylib_u16", "DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = TARGET_VERTICES / original_vertices
    mesh.modifiers.move(len(mesh.modifiers) - 1, 0)
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    split_rigid_groups(mesh)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        export_yup=True,
        export_skins=True,
        export_animations=False,
        export_apply=False,
    )
    rebase_glb_to_production_links(output)


def _certify(source, output):
    import bpy

    repository_root = pathlib.Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from resources.g1_mesh.validate_g1_raylib_glb import (
        EVIDENCE_FIELDS,
        REQUIRED_ARTICULATED_BONES,
        inspect_g1_glb,
        validate_g1_glb,
    )

    evidence = inspect_g1_glb(output)
    payload = {field: evidence[field] for field in EVIDENCE_FIELDS}
    payload.update(
        {
            "blender_version": bpy.app.version_string,
            "glb_size": output.stat().st_size,
            "skin_names": evidence["skin_names"],
            "required_articulated_bones": list(REQUIRED_ARTICULATED_BONES),
            "source_size": source.stat().st_size,
            "source_sha256": sha256(source),
        }
    )
    manifest_path = output.with_name("manifest.json")
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return validate_g1_glb(output, manifest_path, source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    source = args.source.resolve()
    output = args.output.resolve()
    manifest_path = output.with_name("manifest.json")
    try:
        output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        export(source, output)
        report = _certify(source, output)
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    except Exception:
        output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
