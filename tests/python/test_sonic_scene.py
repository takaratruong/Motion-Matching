from dataclasses import replace
import hashlib
import json
import os
import struct
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from mm_sonic.hands import LEFT_HAND_JOINT_ORDER, RIGHT_HAND_JOINT_ORDER
from mm_sonic.joints import TARGET_JOINT_ORDER
from mm_sonic.scene import (
    HOLDEN_TO_MUJOCO_MATRIX,
    MUJOCO_COORDINATE_SIGNATURE,
    SceneError,
    load_scene_registry,
    normalize_run_local_actuators,
    penetration_exceeds_threshold,
    register_scene,
    replay_kinematic_reference,
    transform_obj,
    verify_loaded_actuator_routing,
    verify_mm_scene_identity,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "sonic/configs/scene_registry.json"
REGISTRY_SHA256 = (
    "85480e1e8a190d6b065749382172191850739c0aca8a8946182f97665d693d5b"
)
HOLDEN_SIGNATURE = "holden-y-up-right-handed-forward-plus-z"
GEAR_COMMIT = "60de0df7ffedeef415fe58d435e92cc5b01ba3d9"
GEAR_SCENE_SHA256 = (
    "f8538904eb47cada1bfb2dcdc157099092aa63df4307d7e077b651b16bfb6c74"
)
GEAR_ROBOT_SHA256 = (
    "8b68d8f06674c5c10cd2cd89764b3cfba9fabba5080b55ea67ee1dd12cf630cd"
)
EXPECTED_G1_ACTUATOR_JOINT_ORDER = (
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
    *LEFT_HAND_JOINT_ORDER,
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    *RIGHT_HAND_JOINT_ORDER,
)


def _sha256(path_or_bytes):
    if isinstance(path_or_bytes, (str, Path)):
        value = Path(path_or_bytes).read_bytes()
    else:
        value = bytes(path_or_bytes)
    return hashlib.sha256(value).hexdigest()


def _write_json(path, value):
    data = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha256(data)


def _mm_hello_identity(terrain_root):
    manifest_path = terrain_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    index_path = terrain_root / manifest["scene_index"]["path"]
    index = json.loads(index_path.read_text())
    return {
        "coordinate_signature": index["coordinate_signature"],
        "motion_manifest_sha256": _sha256(manifest_path),
        "scene_index_sha256": _sha256(index_path),
        "joint_feasibility": {
            "schema": "g1-joint-feasibility-certificate/v1",
            "frame_count": 1,
            "raw_safe_count": 1,
            "raw_unsafe_count": 0,
            "search_safe_count": 1,
            "joint_limit_violation_count": [0] * 29,
            "mask_sha256": "a" * 64,
        },
    }


def _flat_scene_identity(route_id="flat-12s"):
    return {
        "scene_id": "sonic-flat-baseline",
        "route_id": route_id,
        "coordinate_signature": HOLDEN_SIGNATURE,
        "heightfield_sha256": REGISTRY_SHA256,
        "mesh_sha256": REGISTRY_SHA256,
        "walkability_sha256": REGISTRY_SHA256,
    }


def _terrain_scene_identity(terrain_root, scene_id, route_id):
    metadata = json.loads(
        (terrain_root / "scenes" / scene_id / "scene.json").read_text()
    )
    return {
        "scene_id": scene_id,
        "route_id": route_id,
        "coordinate_signature": metadata["coordinate_signature"],
        "heightfield_sha256": metadata["heightfield"]["sha256"],
        "mesh_sha256": metadata["mesh"]["sha256"],
        "walkability_sha256": metadata["walkability"]["sha256"],
    }


def _element_structure(element):
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(_element_structure(child) for child in element),
    )


def _compile_mm_server(executable):
    result = subprocess.run(
        (
            "g++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I.",
            '-DMM_CHUNK_BUILD_COMMIT="task9-test"',
            "-DMM_CHUNK_DEFAULT_JOINT_CONTRACT="
            f'"{ROOT / "sonic/configs/g1_joint_contract.json"}"',
            f'-DMM_CHUNK_DEFAULT_SCENE_REGISTRY="{REGISTRY}"',
            "sonic/cpp/mm_chunk_server.cpp",
            "-o",
            str(executable),
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)


def _exchange_mm_server(executable, terrain_root, requests):
    environment = os.environ.copy()
    environment["SONIC_TERRAIN_DIR"] = str(terrain_root)
    environment["SONIC_SCENE_REGISTRY"] = str(REGISTRY)
    process = subprocess.Popen(
        (str(executable),),
        cwd=ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    responses = []
    try:
        for request in requests:
            process.stdin.write(
                json.dumps(request, separators=(",", ":")) + "\n"
            )
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise AssertionError("MM server closed before responding")
            responses.append(json.loads(line))
        process.stdin.close()
        process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else ""
        if process.stderr is not None:
            process.stderr.close()
        return_code = process.wait(timeout=30)
        if return_code != 0:
            raise AssertionError(stderr)
        return responses
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)


def _synthetic_robot_xml():
    body_names = {
        "waist_yaw_joint": "waist_yaw_link",
        "waist_roll_joint": "waist_roll_link",
        "waist_pitch_joint": "torso_link",
    }
    bodies = []
    for index, joint_name in enumerate(reversed(TARGET_JOINT_ORDER)):
        body_name = body_names.get(
            joint_name, joint_name.removesuffix("_joint") + "_link"
        )
        if body_name == "left_ankle_roll_link":
            position = "0.2 0 -0.195"
        elif body_name == "right_ankle_roll_link":
            position = "-0.2 0 -0.195"
        else:
            position = f"{(index % 7) * 0.03 - 0.09:.9g} 0 1.5"
        descendant = ""
        if body_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            side = body_name.split("_", 1)[0]
            descendant = (
                f'<body name="{side}_sole_link">'
                f'<geom name="{side}_sole_geom" type="sphere" '
                'size="0.01" density="100"/></body>'
            )
        bodies.append(
            f'<body name="{body_name}" pos="{position}">'
            f'<joint name="{joint_name}" type="hinge" axis="0 1 0" '
            'range="-3 3" damping="0.01"/>'
            f'<geom name="{body_name}_geom" type="sphere" size="0.01" '
            'density="100"/>'
            + descendant
            + "</body>"
        )
    for side, reference in (("left", "0.31"), ("right", "-0.27")):
        bodies.append(
            f'<body name="{side}_hand_finger_link" pos="0 0 1.7">'
            f'<joint name="{side}_hand_finger_joint" type="hinge" '
            f'axis="1 0 0" range="-1 1" ref="{reference}"/>'
            f'<geom name="{side}_hand_finger_geom" type="sphere" '
            'size="0.01" density="100"/>'
            "</body>"
        )
    joint_names = list(reversed(TARGET_JOINT_ORDER)) + [
        "left_hand_finger_joint",
        "right_hand_finger_joint",
    ]
    # Emit one motor per named joint, intentionally in reverse document order,
    # so normalization must reorder them to MuJoCo joint traversal.
    motors = "".join(
        f'<motor name="{name}_motor" joint="{name}" gear="1"/>'
        for name in reversed(joint_names)
    )
    return (
        '<mujoco model="synthetic_g1">'
        '<compiler angle="radian" meshdir="meshes"/>'
        '<worldbody><body name="pelvis" pos="0 0 0.2">'
        '<joint name="floating_base_joint" type="free"/>'
        '<geom name="pelvis_geom" type="sphere" size="0.1" density="100"/>'
        + "".join(bodies)
        + "</body></worldbody>"
        + f"<actuator>{motors}</actuator>"
        + "</mujoco>\n"
    )


def _official_scene_fixture(root):
    inputs = root / "official-input"
    inputs.mkdir()
    (inputs / "meshes").mkdir()
    robot = inputs / "robot.xml"
    robot.write_text(_synthetic_robot_xml(), encoding="utf-8")
    scene = inputs / "official_scene.xml"
    scene.write_text(
        '<mujoco model="official">\n'
        '  <include file="robot.xml"/>\n'
        '  <asset><texture name="ground" type="2d" builtin="checker" '
        'width="16" height="16" rgb1=".2 .2 .2" rgb2=".3 .3 .3"/>'
        '<material name="ground" texture="ground"/></asset>\n'
        '  <worldbody><light name="sun" pos="0 0 3"/>'
        '<geom name="calibration_marker" type="sphere" pos="2 2 2" '
        'size=".01" rgba=".1 .2 .3 1" contype="0" conaffinity="0"/>'
        '<geom name="floor" type="plane" size="0 0 .05" '
        'material="ground"/></worldbody>\n'
        '</mujoco>\n',
        encoding="utf-8",
    )
    return scene, robot


def _asymmetric_obj(*, textured=True):
    texture = (
        "vt 0 0\nvt 1 0\nvt 0 1\n"
        "vn 0 0 1\n"
        "usemtl certified-rock\n"
        if textured
        else ""
    )
    suffix = "/1/1" if textured else ""
    second = ("/2/1", "/3/1") if textured else ("", "")
    return (
        "# asymmetric Holden tetrahedron\n"
        "v -1 0 -2\n"
        "v 2 0 -2\n"
        "v -1 3 -2\n"
        "v -1 0 4\n"
        + texture
        + f"f 1{suffix} 2{second[0]} 3{second[1]}\n"
        + f"f 1{suffix} 4{second[0]} 2{second[1]}\n"
        + f"f 1{suffix} 3{second[0]} 4{second[1]}\n"
        + f"f 2{suffix} 4{second[0]} 3{second[1]}\n"
    ).encode("ascii")


def _terrain_fixture(root, scene_id="synthetic-curb"):
    scene_dir = root / "scenes" / scene_id
    scene_dir.mkdir(parents=True)
    obj = scene_dir / "terrain.obj"
    obj.write_bytes(_asymmetric_obj(textured=False))
    terrain = scene_dir / "terrain.bin"
    terrain.write_bytes(
        struct.pack("<4sIIIffff", b"G1HF", 2, 2, 2, -1.0, -2.0, 3.0, 0.0)
        + struct.pack("<ffff", 0.0, 0.0, 0.0, 0.0)
    )
    walkability_sha = "b" * 64
    scene_payload = {
        "schema": "g1-terrain-scene/v1",
        "id": scene_id,
        "coordinate_signature": HOLDEN_SIGNATURE,
        "heightfield": {
            "path": "terrain.bin",
            "schema": "G1HF/v2",
            "sha256": _sha256(terrain),
            "version": 2,
            "nx": 2,
            "nz": 2,
            "origin_x": -1.0,
            "origin_z": -2.0,
            "cell_size_m": 3.0,
            "exterior_height_m": 0.0,
        },
        "mesh": {
            "path": "terrain.obj",
            "schema": "obj/v1",
            "sha256": _sha256(obj),
        },
        "walkability": {
            "path": "walkability.bin",
            "schema": "G1WM/v1",
            "sha256": walkability_sha,
            "version": 1,
            "nx": 2,
            "nz": 2,
        },
        "bounds": {
            "mesh_min_xyz": [-1.0, 0.0, -2.0],
            "mesh_max_xyz": [2.0, 3.0, 4.0],
        },
        "spawn": {"position": [0.0, 0.0, 0.0], "yaw_radians": 0.0},
        "routes": [
            {
                "id": "forward",
                "waypoints_xz": [[0.0, 0.0], [0.0, 1.0]],
                "walkability_class": 1,
            }
        ],
    }
    scene_json = scene_dir / "scene.json"
    scene_sha = _write_json(scene_json, scene_payload)
    index_payload = {
        "schema": "g1-terrain-scene-index/v1",
        "coordinate_signature": HOLDEN_SIGNATURE,
        "scene_ids": [scene_id],
        "scenes": [
            {
                "id": scene_id,
                "path": f"scenes/{scene_id}/scene.json",
                "sha256": scene_sha,
            }
        ],
    }
    index = root / "scenes/index.json"
    index_sha = _write_json(index, index_payload)
    _write_json(
        root / "manifest.json",
        {
            "schema": "g1-terrain-artifacts/v2",
            "scene_index": {
                "path": "scenes/index.json",
                "schema": "g1-terrain-scene-index/v1",
                "sha256": index_sha,
            },
        },
    )
    identity = {
        "scene_id": scene_id,
        "route_id": "forward",
        "coordinate_signature": HOLDEN_SIGNATURE,
        "heightfield_sha256": _sha256(terrain),
        "mesh_sha256": _sha256(obj),
        "walkability_sha256": walkability_sha,
    }
    return identity


class RegistryAndObjTests(unittest.TestCase):
    def test_committed_registry_is_the_only_exact_shared_definition(self):
        self.assertEqual(_sha256(REGISTRY), REGISTRY_SHA256)
        registry = load_scene_registry(REGISTRY)
        self.assertEqual(tuple(registry), ("sonic-flat-baseline",))
        self.assertEqual(
            registry["sonic-flat-baseline"],
            {
                "kind": "analytic-flat",
                "coordinate_signature": HOLDEN_SIGNATURE,
                "bounds_xz": (-10.0, -10.0, 10.0, 10.0),
                "spawn_position_holden": (0.0, 0.0, 0.0),
                "spawn_yaw_holden": 0.0,
                "height_m": 0.0,
                "walkability_class": 1,
            },
        )
        np.testing.assert_array_equal(
            HOLDEN_TO_MUJOCO_MATRIX,
            np.array(((1, 0, 0), (0, 0, -1), (0, 1, 0)), np.float64),
        )

    def test_obj_transform_is_deterministic_and_preserves_face_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.obj"
            source.write_bytes(_asymmetric_obj())
            first = root / "first.obj"
            second = root / "second.obj"
            first_record = transform_obj(source, _sha256(source), first)
            second_record = transform_obj(source, _sha256(source), second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_record, second_record)
            self.assertEqual(first_record.source_sha256, _sha256(source))
            self.assertEqual(first_record.transformed_sha256, _sha256(first))
            np.testing.assert_array_equal(
                first_record.source_bounds_holden,
                np.array(((-1, 0, -2), (2, 3, 4)), np.float32),
            )
            np.testing.assert_array_equal(
                first_record.transformed_bounds_mujoco,
                np.array(((-1, -4, 0), (2, 2, 3)), np.float32),
            )
            text = first.read_text(encoding="ascii")
            self.assertIn("v -1 2 0\n", text)
            self.assertIn("v -1 -4 0\n", text)
            self.assertIn("vn 0 -1 0\n", text)
            self.assertIn("usemtl certified-rock\n", text)
            source_faces = [
                line
                for line in source.read_text().splitlines()
                if line.startswith("f ")
            ]
            target_faces = [line for line in text.splitlines() if line.startswith("f ")]
            self.assertEqual(target_faces, source_faces)
            self.assertIn("/2/1", target_faces[0])

    def test_obj_hash_path_bounds_and_symlink_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.obj"
            source.write_bytes(_asymmetric_obj())
            with self.assertRaisesRegex(SceneError, "SHA-256 mismatch"):
                transform_obj(source, "0" * 64, root / "wrong.obj")
            link = root / "link.obj"
            link.symlink_to(source)
            with self.assertRaisesRegex(SceneError, "symlink"):
                transform_obj(link, _sha256(source), root / "linked.obj")
            destination = root / "destination.obj"
            destination.symlink_to(root / "outside.obj")
            with self.assertRaisesRegex(SceneError, "destination"):
                transform_obj(source, _sha256(source), destination)
            malformed = root / "malformed.obj"
            malformed.write_text("v 0 0 nan\nf 1 1 1\n", encoding="ascii")
            with self.assertRaisesRegex(SceneError, "finite"):
                transform_obj(malformed, _sha256(malformed), root / "bad.obj")


class ActuatorNormalizationTests(unittest.TestCase):
    def _register_flat_scene_with_synthetic_actuators(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        terrain_root = root / "terrain"
        _terrain_fixture(terrain_root)
        official, robot = _official_scene_fixture(root)
        return register_scene(
            "sonic-flat-baseline",
            "flat-12s",
            registry_path=REGISTRY,
            terrain_dir=terrain_root,
            official_scene_xml=official,
            output_dir=root / "flat-output",
            expected_official_scene_sha256=_sha256(official),
            expected_robot_sha256=_sha256(robot),
            mm_hello_identity=_mm_hello_identity(terrain_root),
            mm_scene_identity=_flat_scene_identity(),
        )

    def test_run_local_actuators_match_loaded_joint_traversal_with_sentinels(self):
        registered = self._register_flat_scene_with_synthetic_actuators()
        model = mujoco.MjModel.from_xml_path(str(registered.gear_scene_xml))
        order = verify_loaded_actuator_routing(model)
        self.assertEqual(order, tuple(model.joint(i).name for i in range(1, model.njnt)))
        sentinels = np.arange(1, model.nu + 1, dtype=np.float64)
        routed = np.zeros(model.nu, dtype=np.float64)
        for joint_id, sentinel in enumerate(sentinels, start=1):
            routed[joint_id - 1] = sentinel
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            self.assertEqual(routed[actuator_id], sentinels[joint_id - 1])

    def test_normalization_is_deterministic_and_preserves_motor_attributes(self):
        robot_bytes = _synthetic_robot_xml().encode("utf-8")
        first, order_first, sha_first = normalize_run_local_actuators(
            robot_bytes, label="robot"
        )
        second, order_second, sha_second = normalize_run_local_actuators(
            robot_bytes, label="robot"
        )
        self.assertEqual(first, second)
        self.assertEqual(order_first, order_second)
        self.assertEqual(sha_first, sha_second)
        self.assertRegex(sha_first, r"^[0-9a-f]{64}$")
        root = ET.fromstring(first)
        actuator = root.find("actuator")
        self.assertIsNotNone(actuator)
        motors = list(actuator)
        self.assertEqual(
            tuple(motor.attrib["joint"] for motor in motors), order_first
        )
        for motor in motors:
            self.assertEqual(motor.attrib["gear"], "1")
            self.assertTrue(motor.attrib["name"].endswith("_motor"))

    def test_missing_duplicate_and_unknown_motors_fail_closed(self):
        root = ET.fromstring(_synthetic_robot_xml())
        actuator = root.find("actuator")
        motors = list(actuator)
        # Duplicate motor for the same joint.
        duplicate = ET.SubElement(actuator, "motor")
        duplicate.attrib.update(motors[0].attrib)
        duplicate.attrib["name"] = "dup_motor"
        with self.assertRaisesRegex(SceneError, "motor"):
            normalize_run_local_actuators(
                ET.tostring(root, encoding="utf-8"), label="robot"
            )

        root = ET.fromstring(_synthetic_robot_xml())
        actuator = root.find("actuator")
        actuator.remove(list(actuator)[0])
        with self.assertRaisesRegex(SceneError, "motor"):
            normalize_run_local_actuators(
                ET.tostring(root, encoding="utf-8"), label="robot"
            )

        root = ET.fromstring(_synthetic_robot_xml())
        actuator = root.find("actuator")
        list(actuator)[0].attrib["joint"] = "nonexistent_joint"
        with self.assertRaisesRegex(SceneError, "joint"):
            normalize_run_local_actuators(
                ET.tostring(root, encoding="utf-8"), label="robot"
            )

    def test_loaded_verifier_requires_free_root_and_joint_transmissions(self):
        nonfree_root = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="root">'
            '<joint name="unactuated" type="hinge"/>'
            '<geom type="sphere" size="0.1" density="100"/>'
            '<body name="child"><joint name="actuated" type="hinge"/>'
            '<geom type="sphere" size="0.1" density="100"/>'
            '</body></body></worldbody><actuator>'
            '<motor name="motor" joint="actuated" gear="1"/>'
            '</actuator></mujoco>'
        )
        with self.assertRaisesRegex(SceneError, "free root"):
            verify_loaded_actuator_routing(nonfree_root)

        site_transmission = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="root">'
            '<freejoint name="floating_base_joint"/>'
            '<geom type="sphere" size="0.1" density="100"/>'
            '<site name="first_site" size="0.01"/>'
            '<body name="child"><joint name="actuated" type="hinge"/>'
            '<geom type="sphere" size="0.1" density="100"/>'
            '<site name="target_site" size="0.01"/>'
            '</body></body></worldbody><actuator>'
            '<motor name="motor" site="target_site" gear="1 0 0 0 0 0"/>'
            '</actuator></mujoco>'
        )
        with self.assertRaisesRegex(SceneError, "joint transmission"):
            verify_loaded_actuator_routing(site_transmission)


class OverlayAndAuthenticationTests(unittest.TestCase):
    def test_absolute_dotdot_output_cannot_enter_authenticated_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            terrain_root = root / "terrain"
            _terrain_fixture(terrain_root)
            official, robot = _official_scene_fixture(root)
            detour = root / "detour"
            detour.mkdir()
            canonical_output = official.parent / "escaped-output"
            disguised_output = detour / ".." / official.parent.name / canonical_output.name
            self.assertTrue(disguised_output.is_absolute())
            self.assertNotEqual(disguised_output, canonical_output)
            with self.assertRaisesRegex(SceneError, "output.*input"):
                register_scene(
                    "sonic-flat-baseline",
                    "flat-12s",
                    registry_path=REGISTRY,
                    terrain_dir=terrain_root,
                    official_scene_xml=official,
                    output_dir=disguised_output,
                    expected_official_scene_sha256=_sha256(official),
                    expected_robot_sha256=_sha256(robot),
                    mm_hello_identity=_mm_hello_identity(terrain_root),
                    mm_scene_identity=_flat_scene_identity(),
                )
            self.assertFalse(canonical_output.exists())

    def test_utf16_doctype_is_rejected_for_official_scene_and_robot(self):
        for target in ("scene", "robot"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                terrain_root = root / "terrain"
                _terrain_fixture(terrain_root)
                official, robot = _official_scene_fixture(root)
                selected = official if target == "scene" else robot
                document = (
                    '<?xml version="1.0" encoding="UTF-16"?>\n'
                    '<!DOCTYPE mujoco [<!ENTITY marker "authenticated">]>\n'
                    + selected.read_text(encoding="utf-8")
                )
                selected.write_bytes(document.encode("utf-16"))
                with self.assertRaisesRegex(SceneError, "DOCTYPE"):
                    register_scene(
                        "sonic-flat-baseline",
                        "flat-12s",
                        registry_path=REGISTRY,
                        terrain_dir=terrain_root,
                        official_scene_xml=official,
                        output_dir=root / "rejected-output",
                        expected_official_scene_sha256=_sha256(official),
                        expected_robot_sha256=_sha256(robot),
                        mm_hello_identity=_mm_hello_identity(terrain_root),
                        mm_scene_identity=_flat_scene_identity(),
                    )
                self.assertFalse((root / "rejected-output").exists())

    def test_flat_overlay_loads_with_absolute_robot_and_one_named_plane(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            terrain_root = root / "terrain"
            _terrain_fixture(terrain_root)
            official, robot = _official_scene_fixture(root)
            official_root = ET.parse(official).getroot()
            registered = register_scene(
                "sonic-flat-baseline",
                "flat-12s",
                registry_path=REGISTRY,
                terrain_dir=terrain_root,
                official_scene_xml=official,
                output_dir=root / "flat-output",
                expected_official_scene_sha256=_sha256(official),
                expected_robot_sha256=_sha256(robot),
                mm_hello_identity=_mm_hello_identity(terrain_root),
                mm_scene_identity=_flat_scene_identity(),
            )
            self.assertEqual(registered.scene_id, "sonic-flat-baseline")
            self.assertEqual(registered.route_id, "flat-12s")
            self.assertEqual(registered.source_kind, "analytic-flat")
            self.assertIsNone(registered.source_mesh)
            self.assertIsNone(registered.source_heightfield)
            self.assertEqual(registered.source_hashes["registry"], REGISTRY_SHA256)
            self.assertEqual(
                registered.source_hashes["manifest"],
                _sha256(terrain_root / "manifest.json"),
            )
            self.assertEqual(
                registered.source_hashes["scene_index"],
                _sha256(terrain_root / "scenes/index.json"),
            )
            self.assertEqual(registered.coordinate_source, HOLDEN_SIGNATURE)
            self.assertEqual(
                registered.coordinate_target, MUJOCO_COORDINATE_SIGNATURE
            )
            np.testing.assert_array_equal(
                registered.transform_matrix, HOLDEN_TO_MUJOCO_MATRIX
            )
            self.assertIsNone(registered.transformed_obj)
            root_xml = ET.parse(registered.gear_scene_xml).getroot()
            includes = root_xml.findall("include")
            self.assertEqual(len(includes), 1)
            generated_robot = Path(includes[0].attrib["file"])
            self.assertEqual(
                generated_robot,
                registered.gear_scene_xml.parent / "gear_robot.xml",
            )
            self.assertTrue(generated_robot.is_absolute())
            generated_compiler = ET.parse(generated_robot).getroot().find("compiler")
            self.assertIsNotNone(generated_compiler)
            assert generated_compiler is not None
            self.assertEqual(
                generated_compiler.attrib["meshdir"],
                str((robot.parent / "meshes").resolve()),
            )
            terrain_geoms = root_xml.findall("./worldbody/geom[@name='mm_terrain']")
            self.assertEqual(len(terrain_geoms), 1)
            self.assertEqual(terrain_geoms[0].attrib["type"], "plane")
            self.assertEqual(root_xml.findall("./worldbody/geom[@name='floor']"), [])
            original_floor = official_root.find("./worldbody/geom[@name='floor']")
            assert original_floor is not None
            expected_flat_attributes = dict(original_floor.attrib)
            expected_flat_attributes["name"] = "mm_terrain"
            self.assertEqual(terrain_geoms[0].attrib, expected_flat_attributes)
            self.assertEqual(
                tuple(
                    _element_structure(child)
                    for child in official_root.find("asset")
                ),
                tuple(
                    _element_structure(child)
                    for child in root_xml.find("asset")
                ),
            )
            self.assertEqual(
                tuple(
                    _element_structure(child)
                    for child in official_root.find("worldbody")
                    if child is not original_floor
                ),
                tuple(
                    _element_structure(child)
                    for child in root_xml.find("worldbody")
                    if child.attrib.get("name") != "mm_terrain"
                ),
            )
            model = mujoco.MjModel.from_xml_path(str(registered.gear_scene_xml))
            self.assertEqual(model.geom("mm_terrain").id >= 0, True)
            self.assertGreater(len(registered.allowed_foot_geoms), 0)
            self.assertEqual(
                set(registered.forbidden_geom_groups),
                {"pelvis", "knees", "torso", "hands"},
            )
            forbidden = set().union(
                *(set(ids) for ids in registered.forbidden_geom_groups.values())
            )
            self.assertTrue(forbidden)
            self.assertTrue(set(registered.allowed_foot_geoms).isdisjoint(forbidden))
            for name in ("left_sole_geom", "right_sole_geom"):
                self.assertIn(
                    model.geom(name).id, registered.allowed_foot_geoms
                )
            forbidden_groups = [
                set(ids)
                for ids in registered.forbidden_geom_groups.values()
            ]
            for index, first in enumerate(forbidden_groups):
                for second in forbidden_groups[index + 1 :]:
                    self.assertTrue(first.isdisjoint(second))
            for ids in registered.forbidden_geom_groups.values():
                self.assertTrue(ids)

    def test_authenticated_terrain_overlay_uses_only_published_obj_and_bin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            terrain_root = root / "terrain"
            identity = _terrain_fixture(terrain_root)
            official, robot = _official_scene_fixture(root)
            official_root = ET.parse(official).getroot()
            registered = register_scene(
                identity["scene_id"],
                identity["route_id"],
                registry_path=REGISTRY,
                terrain_dir=terrain_root,
                official_scene_xml=official,
                output_dir=root / "terrain-output",
                expected_official_scene_sha256=_sha256(official),
                expected_robot_sha256=_sha256(robot),
                mm_hello_identity=_mm_hello_identity(terrain_root),
                mm_scene_identity=identity,
            )
            source_dir = terrain_root / "scenes" / identity["scene_id"]
            self.assertEqual(registered.source_mesh, source_dir / "terrain.obj")
            self.assertEqual(
                registered.source_heightfield, source_dir / "terrain.bin"
            )
            self.assertEqual(
                set(registered.source_hashes),
                {
                    "manifest",
                    "scene_index",
                    "scene_json",
                    "terrain_bin",
                    "terrain_obj",
                    "walkability",
                },
            )
            self.assertEqual(
                registered.source_hashes["terrain_bin"],
                identity["heightfield_sha256"],
            )
            self.assertEqual(
                registered.source_hashes["terrain_obj"], identity["mesh_sha256"]
            )
            root_xml = ET.parse(registered.gear_scene_xml).getroot()
            self.assertEqual(root_xml.findall("./worldbody/geom[@name='floor']"), [])
            terrain_geoms = root_xml.findall("./worldbody/geom[@name='mm_terrain']")
            self.assertEqual(len(terrain_geoms), 1)
            self.assertEqual(terrain_geoms[0].attrib["type"], "mesh")
            self.assertEqual(
                len(root_xml.findall("./asset/mesh[@name='mm_terrain_mesh']")),
                1,
            )
            original_assets = tuple(
                _element_structure(child)
                for child in official_root.find("asset")
            )
            generated_assets = tuple(
                _element_structure(child) for child in root_xml.find("asset")
            )
            self.assertEqual(
                generated_assets[: len(original_assets)], original_assets
            )
            original_floor = official_root.find("./worldbody/geom[@name='floor']")
            assert original_floor is not None
            self.assertEqual(
                tuple(
                    _element_structure(child)
                    for child in official_root.find("worldbody")
                    if child is not original_floor
                ),
                tuple(
                    _element_structure(child)
                    for child in root_xml.find("worldbody")
                    if child.attrib.get("name") != "mm_terrain"
                ),
            )
            model = mujoco.MjModel.from_xml_path(str(registered.gear_scene_xml))
            self.assertEqual(model.geom("mm_terrain").id >= 0, True)
            expected_bounds = np.array(((-1, -4, 0), (2, 2, 3)), np.float32)
            registration_path = (
                registered.gear_scene_xml.parent / "scene_registration.json"
            )
            self.assertEqual(
                _sha256(registration_path),
                registered.output_hashes["scene_registration"],
            )
            registration = json.loads(registration_path.read_text())
            np.testing.assert_array_equal(
                np.asarray(registration["output_bounds_mujoco"], np.float32),
                expected_bounds,
            )
            assert registered.transformed_obj is not None
            registered.transformed_obj.write_bytes(
                registered.transformed_obj.read_bytes() + b"# tampered\n"
            )
            with self.assertRaisesRegex(
                SceneError, "registered transformed OBJ.*SHA-256"
            ):
                replay_kinematic_reference(
                    registered,
                    TARGET_JOINT_ORDER,
                    np.zeros((1, 29), np.float32),
                    np.array([[0.0, 0.0, 0.3]], np.float32),
                    np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
                )

    def test_terrain_hash_bounds_path_route_and_identity_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            terrain_root = root / "terrain"
            identity = _terrain_fixture(terrain_root)
            official, robot = _official_scene_fixture(root)

            bad_identity = dict(identity)
            bad_identity["scene_id"] = "other"
            with self.assertRaisesRegex(SceneError, "MM scene identity"):
                register_scene(
                    identity["scene_id"],
                    identity["route_id"],
                    registry_path=REGISTRY,
                    terrain_dir=terrain_root,
                    official_scene_xml=official,
                    output_dir=root / "bad-identity",
                    expected_official_scene_sha256=_sha256(official),
                    expected_robot_sha256=_sha256(robot),
                    mm_hello_identity=_mm_hello_identity(terrain_root),
                    mm_scene_identity=bad_identity,
                )

            bad_hello = _mm_hello_identity(terrain_root)
            bad_hello["scene_index_sha256"] = "0" * 64
            with self.assertRaisesRegex(SceneError, "MM hello identity"):
                register_scene(
                    identity["scene_id"],
                    identity["route_id"],
                    registry_path=REGISTRY,
                    terrain_dir=terrain_root,
                    official_scene_xml=official,
                    output_dir=root / "bad-hello",
                    expected_official_scene_sha256=_sha256(official),
                    expected_robot_sha256=_sha256(robot),
                    mm_hello_identity=bad_hello,
                    mm_scene_identity=identity,
                )

            with self.assertRaisesRegex(SceneError, "MM hello identity"):
                register_scene(
                    identity["scene_id"],
                    identity["route_id"],
                    registry_path=REGISTRY,
                    terrain_dir=terrain_root,
                    official_scene_xml=official,
                    output_dir=root / "missing-hello",
                    expected_official_scene_sha256=_sha256(official),
                    expected_robot_sha256=_sha256(robot),
                    mm_hello_identity=None,
                    mm_scene_identity=identity,
                )

            with self.assertRaisesRegex(SceneError, "route"):
                register_scene(
                    identity["scene_id"],
                    "missing-route",
                    registry_path=REGISTRY,
                    terrain_dir=terrain_root,
                    official_scene_xml=official,
                    output_dir=root / "bad-route",
                    expected_official_scene_sha256=_sha256(official),
                    expected_robot_sha256=_sha256(robot),
                    mm_hello_identity=_mm_hello_identity(terrain_root),
                    mm_scene_identity=identity,
                )

            external_output = terrain_root / "generated-overlay"
            with self.assertRaisesRegex(SceneError, "output.*input"):
                register_scene(
                    identity["scene_id"],
                    identity["route_id"],
                    registry_path=REGISTRY,
                    terrain_dir=terrain_root,
                    official_scene_xml=official,
                    output_dir=external_output,
                    expected_official_scene_sha256=_sha256(official),
                    expected_robot_sha256=_sha256(robot),
                    mm_hello_identity=_mm_hello_identity(terrain_root),
                    mm_scene_identity=identity,
                )
            self.assertFalse(external_output.exists())

            scene_path = terrain_root / "scenes" / identity["scene_id"] / "scene.json"
            original = json.loads(scene_path.read_text())
            original["bounds"]["mesh_max_xyz"][0] = 2.5
            _write_json(scene_path, original)
            index_path = terrain_root / "scenes/index.json"
            index = json.loads(index_path.read_text())
            index["scenes"][0]["sha256"] = _sha256(scene_path)
            index_sha = _write_json(index_path, index)
            manifest_path = terrain_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["scene_index"]["sha256"] = index_sha
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(SceneError, "mesh bounds"):
                register_scene(
                    identity["scene_id"],
                    identity["route_id"],
                    registry_path=REGISTRY,
                    terrain_dir=terrain_root,
                    official_scene_xml=official,
                    output_dir=root / "bad-bounds",
                    expected_official_scene_sha256=_sha256(official),
                    expected_robot_sha256=_sha256(robot),
                    mm_hello_identity=_mm_hello_identity(terrain_root),
                    mm_scene_identity=identity,
                )

            outside = root / "outside"
            outside.mkdir()
            output_link = root / "output-link"
            output_link.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(SceneError, "output"):
                register_scene(
                    "sonic-flat-baseline",
                    "flat-12s",
                    registry_path=REGISTRY,
                    terrain_dir=terrain_root,
                    official_scene_xml=official,
                    output_dir=output_link,
                    expected_official_scene_sha256=_sha256(official),
                    expected_robot_sha256=_sha256(robot),
                    mm_hello_identity=_mm_hello_identity(terrain_root),
                    mm_scene_identity=_flat_scene_identity(),
                )

    def test_official_scene_and_robot_hashes_are_authenticated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            terrain_root = root / "terrain"
            _terrain_fixture(terrain_root)
            official, robot = _official_scene_fixture(root)
            with self.assertRaisesRegex(SceneError, "official scene.*SHA-256"):
                register_scene(
                    "sonic-flat-baseline",
                    "flat-12s",
                    registry_path=REGISTRY,
                    terrain_dir=terrain_root,
                    official_scene_xml=official,
                    output_dir=root / "bad-official",
                    expected_official_scene_sha256="0" * 64,
                    expected_robot_sha256=_sha256(robot),
                    mm_hello_identity=_mm_hello_identity(terrain_root),
                    mm_scene_identity=_flat_scene_identity(),
                )
            with self.assertRaisesRegex(SceneError, "robot include.*SHA-256"):
                register_scene(
                    "sonic-flat-baseline",
                    "flat-12s",
                    registry_path=REGISTRY,
                    terrain_dir=terrain_root,
                    official_scene_xml=official,
                    output_dir=root / "bad-robot",
                    expected_official_scene_sha256=_sha256(official),
                    expected_robot_sha256="0" * 64,
                    mm_hello_identity=_mm_hello_identity(terrain_root),
                    mm_scene_identity=_flat_scene_identity(),
                )

    def test_robot_compiler_meshdir_and_real_mesh_directory_fail_closed(self):
        cases = (
            "missing-compiler",
            "duplicate-compiler",
            "wrong-meshdir",
            "doctype",
            "symlink-meshes",
        )
        for case in cases:
            with (
                self.subTest(case=case),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                terrain_root = root / "terrain"
                _terrain_fixture(terrain_root)
                official, robot = _official_scene_fixture(root)
                if case == "doctype":
                    robot.write_bytes(b"<!DOCTYPE mujoco []>\n" + robot.read_bytes())
                    pattern = "DOCTYPE"
                elif case == "symlink-meshes":
                    meshes = robot.parent / "meshes"
                    meshes.rmdir()
                    outside = root / "outside-meshes"
                    outside.mkdir()
                    meshes.symlink_to(outside, target_is_directory=True)
                    pattern = "symlink"
                else:
                    tree = ET.parse(robot)
                    compiler = tree.getroot().find("compiler")
                    assert compiler is not None
                    if case == "missing-compiler":
                        tree.getroot().remove(compiler)
                        pattern = "exactly one compiler"
                    elif case == "duplicate-compiler":
                        ET.SubElement(
                            tree.getroot(), "compiler", {"meshdir": "meshes"}
                        )
                        pattern = "exactly one compiler"
                    else:
                        compiler.attrib["meshdir"] = "other"
                        pattern = "meshdir.*exactly meshes"
                    tree.write(robot, encoding="utf-8", xml_declaration=True)

                with self.assertRaisesRegex(SceneError, pattern):
                    register_scene(
                        "sonic-flat-baseline",
                        "flat-12s",
                        registry_path=REGISTRY,
                        terrain_dir=terrain_root,
                        official_scene_xml=official,
                        output_dir=root / "rejected-output",
                        expected_official_scene_sha256=_sha256(official),
                        expected_robot_sha256=_sha256(robot),
                        mm_hello_identity=_mm_hello_identity(terrain_root),
                        mm_scene_identity=_flat_scene_identity(),
                    )
                self.assertFalse((root / "rejected-output").exists())


class KinematicReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.terrain_root = self.root / "terrain"
        _terrain_fixture(self.terrain_root)
        self.official, self.robot = _official_scene_fixture(self.root)
        self.registered = self._register_flat("registered")

    def _register_flat(self, output_name):
        return register_scene(
            "sonic-flat-baseline",
            "flat-12s",
            registry_path=REGISTRY,
            terrain_dir=self.terrain_root,
            official_scene_xml=self.official,
            output_dir=self.root / output_name,
            expected_official_scene_sha256=_sha256(self.official),
            expected_robot_sha256=_sha256(self.robot),
            mm_hello_identity=_mm_hello_identity(self.terrain_root),
            mm_scene_identity=_flat_scene_identity(),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _replay(self, root_height, dtype=np.float32, scene=None):
        selected_scene = self.registered if scene is None else scene
        joint_position = np.zeros((1, 29), dtype)
        joint_position[0, TARGET_JOINT_ORDER.index("left_knee_joint")] = 0.2
        return replay_kinematic_reference(
            selected_scene,
            TARGET_JOINT_ORDER,
            joint_position,
            np.array([[0.0, 0.0, root_height]], dtype),
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype),
        )

    def test_replay_uses_named_qpos_physical_pelvis_and_retains_hand_defaults(self):
        report = self._replay(0.3)
        self.assertEqual(report.frame_count, 1)
        self.assertFalse(report.forbidden_penetration)
        self.assertEqual(report.maximum_forbidden_penetration_m, 0.0)
        self.assertFalse(report.allowed_foot_contacts)
        self.assertEqual(report.contacts, ())
        model = mujoco.MjModel.from_xml_path(str(self.registered.gear_scene_xml))
        root_address = int(model.jnt_qposadr[model.joint("floating_base_joint").id])
        named_addresses = [
            int(model.jnt_qposadr[model.joint(name).id])
            for name in TARGET_JOINT_ORDER
        ]
        self.assertNotEqual(named_addresses, list(range(7, 36)))
        np.testing.assert_array_equal(
            report.qpos[0, root_address : root_address + 7],
            np.array(
                (0, 0, float(np.float32(0.3)), 1, 0, 0, 0), np.float64
            ),
        )
        left_knee = model.joint("left_knee_joint").id
        self.assertAlmostEqual(
            report.qpos[0, model.jnt_qposadr[left_knee]], 0.2, places=7
        )
        for name in ("left_hand_finger_joint", "right_hand_finger_joint"):
            joint_id = model.joint(name).id
            address = int(model.jnt_qposadr[joint_id])
            self.assertEqual(report.qpos[0, address], model.qpos0[address])
            self.assertNotEqual(report.qpos[0, address], 0.0)

    def test_allowed_foot_contact_is_not_forbidden(self):
        report = self._replay(0.2)
        self.assertFalse(report.forbidden_penetration)
        self.assertTrue(report.allowed_foot_contacts)
        self.assertTrue(
            all(contact.group == "allowed_feet" for contact in report.contacts)
        )

    def test_threshold_is_strictly_greater_than_point_zero_zero_five(self):
        self.assertFalse(penetration_exceeds_threshold(0.005, 0.005))
        self.assertTrue(penetration_exceeds_threshold(0.00501, 0.005))
        exact = self._replay(0.095, np.float64)
        self.assertAlmostEqual(
            exact.maximum_forbidden_penetration_m, 0.005, places=7
        )
        self.assertFalse(exact.forbidden_penetration)
        exceeded = self._replay(0.09499, np.float64)
        self.assertAlmostEqual(
            exceeded.maximum_forbidden_penetration_m, 0.00501, places=7
        )
        self.assertTrue(exceeded.forbidden_penetration)

    def test_replay_rejects_body_position_substitution_and_positional_joint_order(self):
        with self.assertRaisesRegex(SceneError, "physical pelvis"):
            replay_kinematic_reference(
                self.registered,
                TARGET_JOINT_ORDER,
                np.zeros((1, 29), np.float32),
                None,
                np.array([[1, 0, 0, 0]], np.float32),
            )
        reversed_names = tuple(reversed(TARGET_JOINT_ORDER))
        with self.assertRaisesRegex(SceneError, "target joint names"):
            replay_kinematic_reference(
                self.registered,
                reversed_names,
                np.zeros((1, 29), np.float32),
                np.array([[0, 0, 0.3]], np.float32),
                np.array([[1, 0, 0, 0]], np.float32),
            )

    def test_explicit_identity_verifier_rejects_overlay_mismatch(self):
        identity = _flat_scene_identity()
        hello = _mm_hello_identity(self.terrain_root)
        verify_mm_scene_identity(self.registered, hello, identity)
        identity["mesh_sha256"] = "0" * 64
        with self.assertRaisesRegex(SceneError, "MM scene identity"):
            verify_mm_scene_identity(self.registered, hello, identity)

        bad_hello = dict(hello)
        bad_hello["motion_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(SceneError, "MM hello identity"):
            verify_mm_scene_identity(
                self.registered, bad_hello, _flat_scene_identity()
            )

        malformed_feasibility = dict(hello)
        malformed_feasibility["joint_feasibility"] = {
            **hello["joint_feasibility"],
            "search_safe_count": 0,
        }
        with self.assertRaisesRegex(SceneError, "joint feasibility"):
            verify_mm_scene_identity(
                self.registered,
                malformed_feasibility,
                _flat_scene_identity(),
            )

    def test_replay_reauthenticates_every_registered_dependency(self):
        cases = (
            (
                "gear XML",
                lambda scene: scene.gear_scene_xml,
                "registered GEAR scene XML.*SHA-256",
            ),
            (
                "official scene",
                lambda scene: self.official,
                "registered official scene XML.*SHA-256",
            ),
            (
                "official robot include",
                lambda scene: self.robot,
                "registered robot include.*SHA-256",
            ),
            (
                "generated robot include",
                lambda scene: Path(
                    ET.parse(scene.gear_scene_xml).getroot().find("include").attrib[
                        "file"
                    ]
                ),
                "registered robot include.*SHA-256",
            ),
            (
                "registration sidecar",
                lambda scene: scene.gear_scene_xml.parent
                / "scene_registration.json",
                "registered scene registration.*SHA-256",
            ),
        )
        for index, (label, select_path, pattern) in enumerate(cases):
            with self.subTest(label=label):
                scene = self._register_flat(f"tamper-{index}")
                path = select_path(scene)
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"\n")
                    with self.assertRaisesRegex(SceneError, pattern):
                        self._replay(0.3, scene=scene)
                finally:
                    path.write_bytes(original)

        scene = self._register_flat("symlink-sidecar")
        sidecar = scene.gear_scene_xml.parent / "scene_registration.json"
        sidecar.unlink()
        sidecar.symlink_to(self.robot)
        with self.assertRaisesRegex(SceneError, "symlink"):
            self._replay(0.3, scene=scene)

    def test_replay_rejects_absolute_dotdot_robot_include_escape(self):
        scene = self._register_flat("dotdot-replay")
        overlay_tree = ET.parse(scene.gear_scene_xml)
        include = overlay_tree.getroot().find("include")
        assert include is not None
        generated_robot = Path(include.attrib["file"])
        outside_robot = self.root / "outside-robot.xml"
        outside_robot.write_bytes(generated_robot.read_bytes())
        detour = scene.gear_scene_xml.parent / "detour"
        detour.mkdir()
        include.attrib["file"] = str(
            detour / ".." / ".." / outside_robot.name
        )
        overlay_tree.write(
            scene.gear_scene_xml,
            encoding="utf-8",
            xml_declaration=True,
        )
        output_hashes = dict(scene.output_hashes)
        output_hashes["gear_scene_xml"] = _sha256(scene.gear_scene_xml)
        attacked = replace(scene, output_hashes=output_hashes)
        with self.assertRaisesRegex(SceneError, "robot include escapes"):
            self._replay(0.3, scene=attacked)


class ManifestSchemaTests(unittest.TestCase):
    def test_run_manifest_registers_complete_scene_evidence(self):
        schema = json.loads(
            (ROOT / "sonic/schemas/run_manifest_v1.schema.json").read_text()
        )
        scene = schema["properties"]["scene_registration"]
        self.assertEqual(
            set(scene["required"]),
            {
                "scene_id",
                "route_id",
                "source_kind",
                "source_hashes",
                "coordinate_source",
                "coordinate_target",
                "transform_matrix",
                "output_hashes",
                "allowed_foot_geoms",
                "forbidden_geom_groups",
            },
        )
        self.assertFalse(scene["additionalProperties"])
        self.assertEqual(
            set(scene["properties"]["forbidden_geom_groups"]["required"]),
            {"pelvis", "knees", "torso", "hands"},
        )
        self.assertFalse(
            scene["properties"]["forbidden_geom_groups"][
                "additionalProperties"
            ]
        )
        complete_rule = next(
            rule
            for rule in schema["allOf"]
            if rule.get("if", {}).get("properties", {}).get("status")
            == {"const": "complete"}
        )
        self.assertIn("scene_registration", complete_rule["then"]["required"])
        self.assertNotIn(
            "scene_registration",
            schema["$defs"]["terminalManifest"]["required"],
        )


class BuildGraphTests(unittest.TestCase):
    def test_projection_cli_public_target_builds_the_runtime_binary(self):
        makefile = (ROOT / "sonic/cpp/Makefile").read_text()
        self.assertIn(
            "g1_project_pose_cli: $(BUILD_DIR)/g1_project_pose_cli",
            makefile,
        )
        dependency_block = makefile.split(
            "$(BUILD_DIR)/g1_project_pose_cli:", 1
        )[1].split("\n\tmkdir", 1)[0]
        self.assertIn("g1_project_pose_cli.cpp", dependency_block)
        self.assertIn("g1_joint_contract_io.h", dependency_block)
        self.assertIn("g1_joint_projection.h", dependency_block)
        self.assertIn("../configs/g1_joint_contract.json", dependency_block)

    def test_server_build_tracks_and_embeds_the_flat_registry(self):
        makefile = (ROOT / "sonic/cpp/Makefile").read_text()
        self.assertIn(
            "SCENE_REGISTRY := $(abspath ../configs/scene_registry.json)",
            makefile,
        )
        dependency_block = makefile.split(
            "$(BUILD_DIR)/mm_chunk_server:", 1
        )[1].split("\n\tmkdir", 1)[0]
        self.assertIn("sonic_flat_scene.h", dependency_block)
        self.assertIn("../configs/scene_registry.json", dependency_block)
        self.assertIn(
            "-DMM_CHUNK_DEFAULT_SCENE_REGISTRY=\\\"$(SCENE_REGISTRY)\\\"",
            makefile,
        )


@unittest.skipUnless(
    os.environ.get("SONIC_TERRAIN_DIR"),
    "SONIC_TERRAIN_DIR is required for guarded real terrain registration",
)
class RealTerrainArtifactTests(unittest.TestCase):
    def test_curb_ramp_and_stairs_authenticate_transform_and_load(self):
        terrain_root = Path(os.environ["SONIC_TERRAIN_DIR"]).resolve(
            strict=True
        )
        scenes = (
            ("grail-curb-low", "curb-forward"),
            ("ramp-10-up-down", "up-landing-down"),
            ("stairs-shallow", "ascent-landing-descent"),
        )
        protected_paths = [
            terrain_root / "manifest.json",
            terrain_root / "scenes/index.json",
        ]
        for scene_id, _ in scenes:
            scene_dir = terrain_root / "scenes" / scene_id
            protected_paths.extend(
                (
                    scene_dir / "scene.json",
                    scene_dir / "terrain.bin",
                    scene_dir / "terrain.obj",
                )
            )
        before = {path: _sha256(path) for path in protected_paths}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            official, robot = _official_scene_fixture(root)
            hello_identity = _mm_hello_identity(terrain_root)

            for scene_id, route_id in scenes:
                with self.subTest(scene_id=scene_id):
                    scene_dir = terrain_root / "scenes" / scene_id
                    metadata = json.loads(
                        (scene_dir / "scene.json").read_text()
                    )
                    identity = _terrain_scene_identity(
                        terrain_root, scene_id, route_id
                    )
                    registered = register_scene(
                        scene_id,
                        route_id,
                        registry_path=REGISTRY,
                        terrain_dir=terrain_root,
                        official_scene_xml=official,
                        output_dir=root / scene_id,
                        expected_official_scene_sha256=_sha256(official),
                        expected_robot_sha256=_sha256(robot),
                        mm_hello_identity=hello_identity,
                        mm_scene_identity=identity,
                    )
                    self.assertEqual(
                        registered.source_mesh, scene_dir / "terrain.obj"
                    )
                    self.assertEqual(
                        registered.source_heightfield,
                        scene_dir / "terrain.bin",
                    )
                    self.assertEqual(
                        registered.source_hashes["terrain_obj"],
                        metadata["mesh"]["sha256"],
                    )
                    self.assertEqual(
                        registered.source_hashes["terrain_bin"],
                        metadata["heightfield"]["sha256"],
                    )
                    self.assertIsNotNone(registered.transformed_obj)
                    assert registered.transformed_obj is not None
                    self.assertEqual(
                        _sha256(registered.transformed_obj),
                        registered.output_hashes["transformed_obj"],
                    )
                    source_bounds = np.asarray(
                        (
                            metadata["bounds"]["mesh_min_xyz"],
                            metadata["bounds"]["mesh_max_xyz"],
                        ),
                        np.float32,
                    )
                    expected_bounds = np.asarray(
                        (
                            (
                                source_bounds[0, 0],
                                -source_bounds[1, 2],
                                source_bounds[0, 1],
                            ),
                            (
                                source_bounds[1, 0],
                                -source_bounds[0, 2],
                                source_bounds[1, 1],
                            ),
                        ),
                        np.float32,
                    )
                    registration = json.loads(
                        (
                            registered.gear_scene_xml.parent
                            / "scene_registration.json"
                        ).read_text()
                    )
                    np.testing.assert_array_equal(
                        np.asarray(
                            registration["source_bounds_holden"],
                            np.float32,
                        ),
                        source_bounds,
                    )
                    np.testing.assert_array_equal(
                        np.asarray(
                            registration["output_bounds_mujoco"],
                            np.float32,
                        ),
                        expected_bounds,
                    )
                    model = mujoco.MjModel.from_xml_path(
                        str(registered.gear_scene_xml)
                    )
                    self.assertGreaterEqual(model.geom("mm_terrain").id, 0)
                    self.assertTrue(registered.allowed_foot_geoms)
                    self.assertEqual(
                        set(registered.forbidden_geom_groups),
                        {"pelvis", "knees", "torso", "hands"},
                    )
                    self.assertTrue(
                        all(registered.forbidden_geom_groups.values())
                    )

        self.assertEqual(
            {path: _sha256(path) for path in protected_paths}, before
        )


@unittest.skipUnless(
    os.environ.get("SONIC_TERRAIN_DIR"),
    "SONIC_TERRAIN_DIR is required for guarded real MM terrain resets",
)
class RealTerrainServerIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.terrain_root = Path(os.environ["SONIC_TERRAIN_DIR"]).resolve(
            strict=True
        )
        cls.temporary = tempfile.TemporaryDirectory()
        cls.executable = Path(cls.temporary.name) / "mm_chunk_server"
        _compile_mm_server(cls.executable)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _exchange_resets(self, scenes):
        requests = [{"v": 1, "op": "hello", "request_id": "h"}]
        for index, (scene_id, route_id) in enumerate(scenes):
            requests.append(
                {
                    "v": 1,
                    "op": "reset",
                    "request_id": f"r{index}",
                    "session_id": f"real-scene-{index}",
                    "scene_id": scene_id,
                    "route_id": route_id,
                    "terrain_weight": 0.0,
                }
            )
        requests.append({"v": 1, "op": "close", "request_id": "c"})
        return _exchange_mm_server(
            self.executable, self.terrain_root, requests
        )

    def test_actual_server_returns_authenticated_curb_and_ramp_identities(self):
        scenes = (
            ("grail-curb-low", "curb-forward"),
            ("ramp-10-up-down", "up-landing-down"),
        )
        responses = self._exchange_resets(scenes)
        self.assertTrue(all(response["ok"] for response in responses), responses)
        for name in (
            "coordinate_signature",
            "motion_manifest_sha256",
            "scene_index_sha256",
        ):
            value = _mm_hello_identity(self.terrain_root)[name]
            self.assertEqual(responses[0]["data"].get(name), value)
        for index, (scene_id, route_id) in enumerate(scenes):
            actual = responses[index + 1]["data"]["scene"]
            expected = _terrain_scene_identity(
                self.terrain_root, scene_id, route_id
            )
            for name, value in expected.items():
                self.assertEqual(actual.get(name), value)

    def test_actual_server_returns_authenticated_stairs_identity(self):
        responses = self._exchange_resets(
            (("stairs-shallow", "ascent-landing-descent"),)
        )
        self.assertTrue(responses[0]["ok"], responses)
        stairs = responses[1]
        if (
            not stairs["ok"]
            and stairs.get("error", {}).get("code") == "reset_failed"
            and "scene route count changed"
            in stairs.get("error", {}).get("message", "")
        ):
            self.skipTest(
                "not_run: active terrain pack predates the authenticated "
                "directional stairs route contract"
            )
        self.assertTrue(stairs["ok"], responses)
        actual = stairs["data"]["scene"]
        expected = _terrain_scene_identity(
            self.terrain_root,
            "stairs-shallow",
            "ascent-landing-descent",
        )
        for name, value in expected.items():
            self.assertEqual(actual.get(name), value)
        self.assertTrue(responses[2]["ok"], responses)


@unittest.skipUnless(
    os.environ.get("SONIC_GEAR_CHECKOUT")
    and os.environ.get("SONIC_TERRAIN_DIR"),
    "SONIC_GEAR_CHECKOUT and SONIC_TERRAIN_DIR are required for official GEAR",
)
class OfficialGearIntegrationTests(unittest.TestCase):
    def test_run_local_robot_preserves_structure_and_mesh_provenance(self):
        checkout = Path(os.environ["SONIC_GEAR_CHECKOUT"]).resolve(strict=True)
        deploy = checkout / "gear_sonic_deploy/g1"
        official = deploy / "scene_29dof_with_hand.xml"
        robot = deploy / "g1_29dof_with_hand.xml"
        meshes = (deploy / "meshes").resolve(strict=True)
        self.assertEqual(_sha256(official), GEAR_SCENE_SHA256)
        self.assertEqual(_sha256(robot), GEAR_ROBOT_SHA256)

        original_root = ET.parse(robot).getroot()
        original_compiler = original_root.find("compiler")
        self.assertIsNotNone(original_compiler)
        assert original_compiler is not None
        self.assertEqual(original_compiler.attrib.get("meshdir"), "meshes")
        unresolved = [
            mesh.attrib["file"]
            for mesh in original_root.findall("./asset/mesh")
            if not (meshes / mesh.attrib["file"]).is_file()
            or (meshes / mesh.attrib["file"]).read_bytes().startswith(
                b"version https://git-lfs.github.com/spec/v1\n"
            )
        ]
        if unresolved:
            self.skipTest(
                "not_run: pinned official GEAR mesh assets are unresolved "
                f"({len(unresolved)} files; first={unresolved[0]})"
            )

        terrain_root = Path(os.environ["SONIC_TERRAIN_DIR"]).resolve(strict=True)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "official-flat"
            registered = register_scene(
                "sonic-flat-baseline",
                "flat-12s",
                registry_path=REGISTRY,
                terrain_dir=terrain_root,
                official_scene_xml=official,
                output_dir=output,
                expected_official_scene_sha256=GEAR_SCENE_SHA256,
                expected_robot_sha256=GEAR_ROBOT_SHA256,
                mm_hello_identity=_mm_hello_identity(terrain_root),
                mm_scene_identity=_flat_scene_identity(),
            )

            overlay_root = ET.parse(registered.gear_scene_xml).getroot()
            includes = overlay_root.findall("include")
            self.assertEqual(len(includes), 1)
            generated_robot = Path(includes[0].attrib["file"]).resolve(strict=True)
            self.assertEqual(generated_robot.parent, output.resolve(strict=True))
            self.assertFalse(generated_robot.is_symlink())
            self.assertNotEqual(generated_robot, robot)

            loaded_model = mujoco.MjModel.from_xml_path(
                str(registered.gear_scene_xml)
            )
            loaded_order = verify_loaded_actuator_routing(loaded_model)
            self.assertEqual(
                loaded_order,
                EXPECTED_G1_ACTUATOR_JOINT_ORDER,
            )
            self.assertEqual(
                loaded_model.joint(
                    int(loaded_model.actuator_trnid[22, 0])
                ).name,
                LEFT_HAND_JOINT_ORDER[0],
            )
            self.assertEqual(
                loaded_model.joint(
                    int(loaded_model.actuator_trnid[29, 0])
                ).name,
                "right_shoulder_pitch_joint",
            )

            generated_root = ET.parse(generated_robot).getroot()
            generated_compilers = generated_root.findall("compiler")
            self.assertEqual(len(generated_compilers), 1)
            self.assertEqual(generated_compilers[0].attrib.get("meshdir"), str(meshes))
            generated_compilers[0].attrib["meshdir"] = "meshes"
            for parsed_root in (generated_root, original_root):
                actuator_blocks = parsed_root.findall("actuator")
                self.assertEqual(len(actuator_blocks), 1)
                actuator = actuator_blocks[0]
                actuator[:] = sorted(
                    list(actuator),
                    key=lambda motor: motor.attrib.get("joint", ""),
                )
            self.assertEqual(
                _element_structure(generated_root),
                _element_structure(original_root),
            )

            self.assertEqual(
                set(registered.output_hashes),
                {
                    "official_scene",
                    "official_robot",
                    "gear_scene_xml",
                    "robot_include",
                    "scene_registration",
                },
            )
            self.assertEqual(
                registered.output_hashes["official_robot"], GEAR_ROBOT_SHA256
            )
            self.assertEqual(
                registered.output_hashes["robot_include"], _sha256(generated_robot)
            )
            registration = json.loads(
                (output / "scene_registration.json").read_text(encoding="utf-8")
            )
            self.assertEqual(registration["official_scene"], str(official))
            self.assertEqual(
                registration["official_scene_sha256"], GEAR_SCENE_SHA256
            )
            self.assertEqual(registration["official_robot_include"], str(robot))
            self.assertEqual(
                registration["official_robot_include_sha256"], GEAR_ROBOT_SHA256
            )
            self.assertEqual(registration["robot_include"], str(generated_robot))
            self.assertEqual(
                registration["robot_include_sha256"], _sha256(generated_robot)
            )

    def test_pinned_official_gear_or_explicit_lfs_not_run(self):
        checkout = Path(os.environ["SONIC_GEAR_CHECKOUT"]).resolve(
            strict=True
        )
        revision = subprocess.run(
            ("git", "-C", str(checkout), "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(revision.returncode, 0, revision.stderr)
        self.assertEqual(revision.stdout.strip(), GEAR_COMMIT)
        deploy = checkout / "gear_sonic_deploy/g1"
        official = deploy / "scene_29dof_with_hand.xml"
        robot = deploy / "g1_29dof_with_hand.xml"
        self.assertEqual(_sha256(official), GEAR_SCENE_SHA256)
        self.assertEqual(_sha256(robot), GEAR_ROBOT_SHA256)

        robot_root = ET.parse(robot).getroot()
        compiler = robot_root.find("compiler")
        mesh_dir = "" if compiler is None else compiler.attrib.get("meshdir", "")
        unresolved = []
        for mesh in robot_root.findall("./asset/mesh"):
            mesh_file = deploy / mesh_dir / mesh.attrib["file"]
            if not mesh_file.is_file():
                unresolved.append(f"missing:{mesh_file.name}")
                continue
            if mesh_file.read_bytes().startswith(
                b"version https://git-lfs.github.com/spec/v1\n"
            ):
                unresolved.append(f"lfs-pointer:{mesh_file.name}")
        if unresolved:
            self.skipTest(
                "not_run: pinned official GEAR mesh assets are unresolved "
                f"({len(unresolved)} files; first={unresolved[0]})"
            )

        terrain_root = Path(os.environ["SONIC_TERRAIN_DIR"]).resolve(
            strict=True
        )
        scenes = (
            ("sonic-flat-baseline", "flat-12s"),
            ("grail-curb-low", "curb-forward"),
            ("ramp-10-up-down", "up-landing-down"),
            ("stairs-shallow", "ascent-landing-descent"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "mm_chunk_server"
            _compile_mm_server(executable)
            requests = [{"v": 1, "op": "hello", "request_id": "h"}]
            for index, (scene_id, route_id) in enumerate(scenes):
                requests.append(
                    {
                        "v": 1,
                        "op": "reset",
                        "request_id": f"r{index}",
                        "session_id": f"official-scene-{index}",
                        "scene_id": scene_id,
                        "route_id": route_id,
                        "terrain_weight": 0.0,
                    }
                )
            requests.append({"v": 1, "op": "close", "request_id": "c"})
            responses = _exchange_mm_server(
                executable, terrain_root, requests
            )
            self.assertTrue(
                all(response["ok"] for response in responses), responses
            )
            hello_identity = responses[0]["data"]
            reset_identities = {
                scene_id: responses[index + 1]["data"]["scene"]
                for index, (scene_id, _) in enumerate(scenes)
            }
            for scene_id, route_id in scenes:
                registered = register_scene(
                    scene_id,
                    route_id,
                    registry_path=REGISTRY,
                    terrain_dir=terrain_root,
                    official_scene_xml=official,
                    output_dir=root / scene_id,
                    expected_official_scene_sha256=GEAR_SCENE_SHA256,
                    expected_robot_sha256=GEAR_ROBOT_SHA256,
                    mm_hello_identity=hello_identity,
                    mm_scene_identity=reset_identities[scene_id],
                )
                self.assertTrue(registered.allowed_foot_geoms)
                self.assertTrue(
                    all(registered.forbidden_geom_groups.values())
                )


@unittest.skipUnless(
    os.environ.get("SONIC_TERRAIN_DIR"),
    "SONIC_TERRAIN_DIR is required for the guarded real MM flat-scene test",
)
class FlatServerIntegrationTests(unittest.TestCase):
    def test_real_server_resets_and_generates_from_committed_flat_registry(self):
        terrain = Path(os.environ["SONIC_TERRAIN_DIR"]).resolve(strict=True)
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "mm_chunk_server"
            compile_result = subprocess.run(
                (
                    "g++",
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    "-I.",
                    f'-DMM_CHUNK_BUILD_COMMIT="task9-test"',
                    "-DMM_CHUNK_DEFAULT_JOINT_CONTRACT="
                    f'"{ROOT / "sonic/configs/g1_joint_contract.json"}"',
                    f'-DMM_CHUNK_DEFAULT_SCENE_REGISTRY="{REGISTRY}"',
                    "sonic/cpp/mm_chunk_server.cpp",
                    "-o",
                    str(executable),
                ),
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                compile_result.returncode,
                0,
                compile_result.stdout + compile_result.stderr,
            )
            environment = os.environ.copy()
            environment["SONIC_TERRAIN_DIR"] = str(terrain)
            environment["SONIC_SCENE_REGISTRY"] = str(REGISTRY)
            process = subprocess.Popen(
                (str(executable),),
                cwd=ROOT,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            requests = (
                {
                    "v": 1,
                    "op": "hello",
                    "request_id": "h",
                },
                {
                    "v": 1,
                    "op": "reset",
                    "request_id": "r",
                    "session_id": "flat-session",
                    "scene_id": "sonic-flat-baseline",
                    "route_id": "flat-12s",
                    "terrain_weight": 0.0,
                },
                {
                    "v": 1,
                    "op": "generate",
                    "request_id": "g",
                    "session_id": "flat-session",
                    "candidate_id": "flat-0",
                    "predecessor_id": None,
                    "source_intervals": 10,
                    "requested_velocity_holden": [0.0, 0.0, 0.5],
                    "desired_heading_holden_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                {
                    "v": 1,
                    "op": "abort",
                    "request_id": "a",
                    "session_id": "flat-session",
                    "candidate_id": "flat-0",
                },
                {"v": 1, "op": "close", "request_id": "c"},
            )
            assert process.stdin is not None
            assert process.stdout is not None
            responses = []
            for request in requests:
                process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                process.stdin.flush()
                responses.append(json.loads(process.stdout.readline()))
            process.stdin.close()
            process.stdout.close()
            stderr = process.stderr.read() if process.stderr is not None else ""
            if process.stderr is not None:
                process.stderr.close()
            self.assertEqual(process.wait(timeout=30), 0, stderr)
            self.assertTrue(all(response["ok"] for response in responses), responses)
            reset_scene = responses[1]["data"]["scene"]
            self.assertEqual(reset_scene["scene_id"], "sonic-flat-baseline")
            self.assertEqual(reset_scene["route_id"], "flat-12s")
            self.assertEqual(reset_scene["heightfield_sha256"], REGISTRY_SHA256)
            self.assertEqual(reset_scene["mesh_sha256"], REGISTRY_SHA256)
            self.assertEqual(reset_scene["walkability_sha256"], REGISTRY_SHA256)
            candidate = responses[2]["data"]
            self.assertEqual(len(candidate["terrain_values"]), 10)
            self.assertEqual(len(candidate["physical_pelvis_position_holden"]), 11)
            self.assertTrue(
                np.all(np.isfinite(np.asarray(candidate["terrain_values"], np.float64)))
            )


if __name__ == "__main__":
    unittest.main()
