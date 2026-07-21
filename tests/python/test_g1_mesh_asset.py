import copy
import json
import math
import pathlib
import struct
import tempfile
import unittest

from resources.g1_mesh.export_g1_raylib_glb import (
    _read_glb,
    _write_glb,
    rebase_glb_to_production_links,
)
from resources.g1_mesh.validate_g1_raylib_glb import validate_g1_glb


ROOT = pathlib.Path(__file__).resolve().parents[2]
ASSET = ROOT / "resources/g1_mesh/g1_raylib.glb"
MANIFEST = ROOT / "resources/g1_mesh/manifest.json"
SOURCE = pathlib.Path("/home/ubuntu/projects/g1_mm/g1.fbx")


def _glb_document_and_binary(path):
    data = path.read_bytes()
    magic, version, total_length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2 or total_length != len(data):
        raise ValueError("invalid GLB header")
    offset = 12
    document = None
    binary = None
    while offset < len(data):
        length, chunk_type = struct.unpack_from("<II", data, offset)
        begin = offset + 8
        end = begin + length
        if end > len(data):
            raise ValueError("GLB chunk is truncated")
        payload = data[begin:end]
        if chunk_type == 0x4E4F534A:
            document = json.loads(payload.rstrip(b" \\x00"))
        elif chunk_type == 0x004E4942:
            binary = payload
        offset = end
    if document is None or binary is None:
        raise ValueError("GLB JSON or BIN chunk is missing")
    return document, binary


def _glb_document(path):
    document, _ = _glb_document_and_binary(path)
    return document


def _quaternion_multiply(first, second):
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )


def _global_node_rotations(document):
    nodes = document["nodes"]
    parents = [-1] * len(nodes)
    for parent, node in enumerate(nodes):
        for child in node.get("children", []):
            parents[child] = parent
    output = [None] * len(nodes)

    def visit(index):
        if output[index] is not None:
            return output[index]
        local = tuple(nodes[index].get("rotation", [0.0, 0.0, 0.0, 1.0]))
        output[index] = (
            local if parents[index] < 0
            else _quaternion_multiply(visit(parents[index]), local)
        )
        return output[index]

    for index in range(len(nodes)):
        visit(index)
    return output


def _rotate_vector(rotation, value):
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


def _global_node_positions(document, global_rotations):
    nodes = document["nodes"]
    parents = [-1] * len(nodes)
    for parent, node in enumerate(nodes):
        for child in node.get("children", []):
            parents[child] = parent
    output = [None] * len(nodes)

    def visit(index):
        if output[index] is not None:
            return output[index]
        local = tuple(nodes[index].get("translation", [0.0, 0.0, 0.0]))
        parent = parents[index]
        if parent < 0:
            output[index] = local
        else:
            parent_position = visit(parent)
            rotated = _rotate_vector(global_rotations[parent], local)
            output[index] = tuple(
                first + second
                for first, second in zip(parent_position, rotated)
            )
        return output[index]

    for index in range(len(nodes)):
        visit(index)
    return output


def _production_zero_pose_positions():
    return {
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


def _production_zero_pose_rotations():
    identity = (0.0, 0.0, 0.0, 1.0)
    hip = (0.0, 0.0, 0.08733857244071258, 0.996178685660368)
    left_shoulder = (
        0.13920101962536002,
        -0.00009868681391343435,
        -0.000013872201955782791,
        0.9902641396131313,
    )
    left_arm = (
        0.000029274470605633596,
        -0.00009579579294906673,
        -0.000027471645031675433,
        0.9999999946057401,
    )
    right_shoulder = (
        -0.13920101962536002,
        0.00009868681391343435,
        -0.000013872201955782777,
        0.9902641396131312,
    )
    right_arm = (
        -0.000029274470605633596,
        0.00009579579294906676,
        -0.0000274716450316754,
        0.9999999946057401,
    )
    result = {
        name: identity
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
        result[name] = hip
    result["left_shoulder_pitch_link"] = left_shoulder
    result["right_shoulder_pitch_link"] = right_shoulder
    for name in (
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
    ):
        result[name] = left_arm
    for name in (
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ):
        result[name] = right_arm
    return result


def _rigid_inverse_matrix(position, rotation):
    norm = math.sqrt(sum(value*value for value in rotation))
    x, y, z, w = (value/norm for value in rotation)
    matrix = (
        (
            1.0 - 2.0*(y*y + z*z),
            2.0*(x*y - z*w),
            2.0*(x*z + y*w),
        ),
        (
            2.0*(x*y + z*w),
            1.0 - 2.0*(x*x + z*z),
            2.0*(y*z - x*w),
        ),
        (
            2.0*(x*z - y*w),
            2.0*(y*z + x*w),
            1.0 - 2.0*(x*x + y*y),
        ),
    )
    inverse_rotation = tuple(zip(*matrix))
    inverse_translation = tuple(
        -sum(
            inverse_rotation[row][axis]*position[axis]
            for axis in range(3)
        )
        for row in range(3)
    )
    inverse = (
        (*inverse_rotation[0], inverse_translation[0]),
        (*inverse_rotation[1], inverse_translation[1]),
        (*inverse_rotation[2], inverse_translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )
    return tuple(
        inverse[row][column]
        for column in range(4)
        for row in range(4)
    )


class G1MeshAssetTests(unittest.TestCase):
    def test_mapped_bind_positions_match_production_link_origins(self):
        document = _glb_document(ASSET)
        global_rotations = _global_node_rotations(document)
        global_positions = _global_node_positions(document, global_rotations)
        nodes_by_name = {
            node.get("name"): index
            for index, node in enumerate(document["nodes"])
        }
        expected = _production_zero_pose_positions()
        self.assertEqual(len(expected), 30)
        for name, reference in expected.items():
            actual = global_positions[nodes_by_name[name]]
            error = math.sqrt(sum(
                (first - second)**2
                for first, second in zip(actual, reference)
            ))
            self.assertLess(
                error,
                0.0005,
                f"{name} bind origin differs by {error:.6f} metres",
            )

    def test_mapped_bind_rotations_match_production_link_frames(self):
        document = _glb_document(ASSET)
        global_rotations = _global_node_rotations(document)
        nodes_by_name = {
            node.get("name"): index
            for index, node in enumerate(document["nodes"])
        }
        expected = _production_zero_pose_rotations()
        self.assertEqual(len(expected), 30)
        for name, reference in expected.items():
            actual = global_rotations[nodes_by_name[name]]
            dot = abs(sum(a*b for a, b in zip(actual, reference)))
            dot = min(1.0, max(-1.0, dot))
            error_degrees = math.degrees(2.0*math.acos(dot))
            self.assertLess(
                error_degrees,
                0.05,
                f"{name} bind frame differs by {error_degrees:.6f} degrees",
            )

    def test_inverse_binds_cancel_all_joint_global_bind_transforms(self):
        document, binary = _glb_document_and_binary(ASSET)
        global_rotations = _global_node_rotations(document)
        global_positions = _global_node_positions(document, global_rotations)
        skin = document["skins"][0]
        accessor = document["accessors"][skin["inverseBindMatrices"]]
        view = document["bufferViews"][accessor["bufferView"]]
        self.assertEqual(accessor["componentType"], 5126)
        self.assertEqual(accessor["type"], "MAT4")
        self.assertEqual(accessor["count"], len(skin["joints"]))
        self.assertEqual(len(skin["joints"]), 39)
        begin = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        stride = view.get("byteStride", 64)
        for slot, node_index in enumerate(skin["joints"]):
            actual = struct.unpack_from("<16f", binary, begin + slot*stride)
            expected = _rigid_inverse_matrix(
                global_positions[node_index],
                global_rotations[node_index],
            )
            error = max(abs(first - second) for first, second in zip(
                actual,
                expected,
            ))
            name = document["nodes"][node_index].get("name", str(node_index))
            self.assertLess(
                error,
                0.00002,
                f"{name} inverse bind differs by {error:.9f}",
            )

    def test_rebase_rejects_inverse_binds_outside_their_buffer_view(self):
        with tempfile.TemporaryDirectory() as directory:
            glb = pathlib.Path(directory) / "g1.glb"
            glb.write_bytes(ASSET.read_bytes())
            document, binary = _read_glb(glb)
            skin = document["skins"][0]
            accessor = document["accessors"][skin["inverseBindMatrices"]]
            view = document["bufferViews"][accessor["bufferView"]]
            stride = view.get("byteStride", 64)
            view["byteLength"] = (
                accessor.get("byteOffset", 0) +
                (accessor["count"] - 1)*stride +
                60
            )
            _write_glb(glb, document, binary)
            with self.assertRaisesRegex(
                RuntimeError,
                "outside its buffer view",
            ):
                rebase_glb_to_production_links(glb)

    def test_committed_asset_is_raylib_compatible(self):
        report = validate_g1_glb(ASSET, MANIFEST, SOURCE)
        self.assertEqual(report["skin_count"], 1)
        self.assertEqual(report["bone_count"], 39)
        self.assertEqual(report["primitive_count"], 35)
        self.assertLessEqual(report["maximum_primitive_vertices"], 65535)
        self.assertEqual(report["rigid_vertex_count"], report["vertex_count"])
        self.assertGreaterEqual(report["height_m"], 0.8)
        self.assertLessEqual(report["height_m"], 1.6)

    def test_manifest_names_all_required_articulated_bones_once(self):
        manifest = json.loads(MANIFEST.read_text())
        required = manifest["required_articulated_bones"]
        self.assertEqual(len(required), 30)
        self.assertEqual(len(set(required)), 30)
        self.assertNotIn("Simulation", required)
        self.assertIn("left_ankle_pitch_link", required)
        self.assertIn("left_ankle_roll_link", required)
        self.assertIn("right_ankle_pitch_link", required)
        self.assertIn("right_ankle_roll_link", required)

    def test_corrupt_glb_and_manifest_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            glb = root / "g1.glb"
            manifest = root / "manifest.json"
            glb.write_bytes(ASSET.read_bytes()[:-1] + b"x")
            payload = copy.deepcopy(json.loads(MANIFEST.read_text()))
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "GLB SHA-256"):
                validate_g1_glb(glb, manifest)


if __name__ == "__main__":
    unittest.main()
