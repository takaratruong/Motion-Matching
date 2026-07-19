import hashlib
import json
import math
import os
import struct
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from mm_sonic.hands import (
    LEFT_HAND_JOINT_ORDER,
    NEUTRAL_HAND_TARGETS,
    RIGHT_HAND_JOINT_ORDER,
)
from mm_sonic.coordinator import SessionConfig, SourceValidator
from mm_sonic.joints import SOURCE_JOINT_ORDER, TARGET_JOINT_ORDER, load_joint_contract
from mm_sonic.process import MMChunkClient
from mm_sonic.schema import InitialBoundary
from mm_sonic.scene import SceneError, register_scene
from mm_sonic.scene_runtime import (
    GEAR_ROBOT_RELATIVE,
    GEAR_ROBOT_SHA256,
    GEAR_SCENE_RELATIVE,
    GEAR_SCENE_SHA256,
    SCENE_REGISTRY_PATH,
    InitialPhysicsState,
    initial_physics_state,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "sonic/configs/scene_registry.json"
JOINT_CONTRACT = ROOT / "sonic/configs/g1_joint_contract.json"
HOLDEN_SIGNATURE = "holden-y-up-right-handed-forward-plus-z"
REGISTRY_SHA256 = (
    "85480e1e8a190d6b065749382172191850739c0aca8a8946182f97665d693d5b"
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


def _full_robot_xml():
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
                f'<body name="{side}_sole_link" pos="0 0 -0.05">'
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
    hand_joint_names = LEFT_HAND_JOINT_ORDER + RIGHT_HAND_JOINT_ORDER
    for joint_name in hand_joint_names:
        body_name = joint_name.removesuffix("_joint") + "_link"
        bodies.append(
            f'<body name="{body_name}" pos="0 0 1.7">'
            f'<joint name="{joint_name}" type="hinge" axis="1 0 0" '
            'range="-2 2"/>'
            f'<geom name="{body_name}_geom" type="sphere" size="0.01" '
            'density="100"/>'
            "</body>"
        )
    joint_names = list(reversed(TARGET_JOINT_ORDER)) + list(hand_joint_names)
    motors = "".join(
        f'<motor name="{name}_motor" joint="{name}" gear="1"/>'
        for name in reversed(joint_names)
    )
    return (
        '<mujoco model="synthetic_g1_hands">'
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
    robot.write_text(_full_robot_xml(), encoding="utf-8")
    scene = inputs / "official_scene.xml"
    scene.write_text(
        '<mujoco model="official">\n'
        '  <include file="robot.xml"/>\n'
        '  <asset><texture name="ground" type="2d" builtin="checker" '
        'width="16" height="16" rgb1=".2 .2 .2" rgb2=".3 .3 .3"/>'
        '<material name="ground" texture="ground"/></asset>\n'
        '  <worldbody><light name="sun" pos="0 0 3"/>'
        '<geom name="floor" type="plane" size="0 0 .05" '
        'material="ground"/></worldbody>\n'
        '</mujoco>\n',
        encoding="utf-8",
    )
    return scene, robot


def _asymmetric_obj():
    return (
        "# asymmetric Holden tetrahedron\n"
        "v -1 0 -2\n"
        "v 2 0 -2\n"
        "v -1 3 -2\n"
        "v -1 0 4\n"
        "f 1 2 3\n"
        "f 1 4 2\n"
        "f 1 3 4\n"
        "f 2 4 3\n"
    ).encode("ascii")


def _terrain_fixture(root, scene_id="synthetic-curb", *, shallow=False):
    scene_dir = root / "scenes" / scene_id
    scene_dir.mkdir(parents=True)
    obj = scene_dir / "terrain.obj"
    obj.write_bytes(_asymmetric_obj())
    terrain = scene_dir / "terrain.bin"
    cell_size = 0.1 if shallow else 1.0
    heights = (0.0,) * 6 if shallow else (0.0, 0.1, 0.2, 0.6, 0.8, 1.0)
    terrain.write_bytes(
        struct.pack(
            "<4sIIIffff",
            b"G1HF",
            2,
            3,
            2,
            -1.0,
            -0.5,
            cell_size,
            0.0,
        )
        + struct.pack("<ffffff", *heights)
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
            "nx": 3,
            "nz": 2,
            "origin_x": -1.0,
            "origin_z": -0.5,
            "cell_size_m": cell_size,
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
    return {
        "scene_id": scene_id,
        "route_id": "forward",
        "coordinate_signature": HOLDEN_SIGNATURE,
        "heightfield_sha256": _sha256(terrain),
        "mesh_sha256": _sha256(obj),
        "walkability_sha256": walkability_sha,
    }


def _mm_hello_identity(terrain_root):
    manifest_path = terrain_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    index_path = terrain_root / manifest["scene_index"]["path"]
    return {
        "coordinate_signature": HOLDEN_SIGNATURE,
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


def _initial_boundary(pelvis_holden):
    return InitialBoundary(
        session_id="reset-session",
        source_joint_names=SOURCE_JOINT_ORDER,
        joint_position_source=np.zeros(29, np.float64),
        joint_velocity_source=np.zeros(29, np.float64),
        physical_pelvis_position_holden=np.asarray(pelvis_holden, np.float64),
        physical_pelvis_orientation_holden=np.array((1.0, 0.0, 0.0, 0.0)),
        virtual_root_position_holden=np.zeros(3, np.float64),
        virtual_root_orientation_holden=np.array((1.0, 0.0, 0.0, 0.0)),
    )


class InitialPhysicsStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.terrain_root = self.root / "terrain"
        _terrain_fixture(self.terrain_root)
        self.official, self.robot = _official_scene_fixture(self.root)
        self.contract = load_joint_contract(JOINT_CONTRACT)
        # MuJoCo pelvis (1, 3, 2) comes from Holden (1, 2, -3).
        self.initial_boundary = _initial_boundary((1.0, 2.0, -3.0))
        self.registered = register_scene(
            "sonic-flat-baseline",
            "flat-12s",
            registry_path=REGISTRY,
            terrain_dir=self.terrain_root,
            official_scene_xml=self.official,
            output_dir=self.root / "flat-output",
            expected_official_scene_sha256=_sha256(self.official),
            expected_robot_sha256=_sha256(self.robot),
            mm_hello_identity=_mm_hello_identity(self.terrain_root),
            mm_scene_identity=_flat_scene_identity(),
        )

    def test_initial_state_uses_mm_root_named_joints_and_closed_hands(self) -> None:
        state = initial_physics_state(
            self.registered,
            self.initial_boundary,
            self.contract,
            NEUTRAL_HAND_TARGETS,
        )
        self.assertIsInstance(state, InitialPhysicsState)
        model = mujoco.MjModel.from_xml_path(str(self.registered.gear_scene_xml))
        root = int(model.jnt_qposadr[model.joint("floating_base_joint").id])
        np.testing.assert_allclose(state.qpos[root : root + 3], (1.0, 3.0, 2.0))
        for name, value in zip(LEFT_HAND_JOINT_ORDER, NEUTRAL_HAND_TARGETS.left):
            address = int(model.jnt_qposadr[model.joint(name).id])
            self.assertAlmostEqual(state.qpos[address], value)
        for name, value in zip(RIGHT_HAND_JOINT_ORDER, NEUTRAL_HAND_TARGETS.right):
            address = int(model.jnt_qposadr[model.joint(name).id])
            self.assertAlmostEqual(state.qpos[address], value)
        self.assertRegex(state.qpos_sha256, r"^[0-9a-f]{64}$")
        self.assertRegex(state.initial_boundary_sha256, r"^[0-9a-f]{64}$")
        self.assertTrue(math.isfinite(state.pelvis_clearance_m))
        self.assertTrue(math.isfinite(state.left_foot_clearance_m))
        self.assertTrue(math.isfinite(state.right_foot_clearance_m))
        self.assertAlmostEqual(state.left_foot_clearance_m, 1.755)
        self.assertAlmostEqual(state.right_foot_clearance_m, 1.755)
        self.assertTrue(math.isfinite(state.maximum_forbidden_penetration_m))

    def test_initial_state_rejects_root_outside_terrain_bounds(self) -> None:
        outside = replace(
            self.initial_boundary,
            physical_pelvis_position_holden=np.array((1000.0, 1.0, 1000.0)),
        )
        with self.assertRaisesRegex(SceneError, "scene domain"):
            initial_physics_state(
                self.registered, outside, self.contract, NEUTRAL_HAND_TARGETS
            )

    def test_initial_state_rejects_joint_limit_violation(self) -> None:
        # The synthetic hinges are limited to [-3, 3]; push one source joint past.
        positions = np.zeros(29, np.float64)
        positions[SOURCE_JOINT_ORDER.index("left_knee_joint")] = 9.0
        boundary = replace(
            self.initial_boundary, joint_position_source=positions
        )
        with self.assertRaisesRegex(SceneError, "outside range"):
            initial_physics_state(
                self.registered, boundary, self.contract, NEUTRAL_HAND_TARGETS
            )

    def test_initial_state_rejects_nonfinite_pelvis(self) -> None:
        boundary = replace(
            self.initial_boundary,
            physical_pelvis_position_holden=np.array((np.inf, 0.0, 0.0)),
        )
        with self.assertRaisesRegex(SceneError, "finite"):
            initial_physics_state(
                self.registered, boundary, self.contract, NEUTRAL_HAND_TARGETS
            )

    def test_initial_state_rejects_forbidden_geom_penetration(self) -> None:
        # Sinking the flat-plane root below the surface drives the pelvis geom
        # into the mm_terrain plane past the 0.005 m threshold.
        boundary = _initial_boundary((0.0, -0.5, 0.0))
        with self.assertRaisesRegex(SceneError, "penetrat"):
            initial_physics_state(
                self.registered, boundary, self.contract, NEUTRAL_HAND_TARGETS
            )

    def test_initial_state_hash_is_deterministic_over_le_float64_qpos(self) -> None:
        first = initial_physics_state(
            self.registered,
            self.initial_boundary,
            self.contract,
            NEUTRAL_HAND_TARGETS,
        )
        second = initial_physics_state(
            self.registered,
            self.initial_boundary,
            self.contract,
            NEUTRAL_HAND_TARGETS,
        )
        self.assertEqual(first.qpos_sha256, second.qpos_sha256)
        self.assertEqual(
            first.qpos_sha256,
            hashlib.sha256(
                np.ascontiguousarray(first.qpos, "<f8").tobytes()
            ).hexdigest(),
        )
        self.assertEqual(
            first.initial_boundary_sha256, second.initial_boundary_sha256
        )

    def test_initial_state_rejects_tampered_robot_include(self) -> None:
        robot_include = self.registered.gear_scene_xml.parent / "gear_robot.xml"
        robot_include.write_text(
            robot_include.read_text(encoding="utf-8") + "<!-- tampered -->\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SceneError, "registered robot include.*SHA-256"):
            initial_physics_state(
                self.registered,
                self.initial_boundary,
                self.contract,
                NEUTRAL_HAND_TARGETS,
            )


class TerrainInitialPhysicsStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.terrain_root = self.root / "terrain"
        self.identity = _terrain_fixture(self.terrain_root)
        self.official, self.robot = _official_scene_fixture(self.root)
        self.contract = load_joint_contract(JOINT_CONTRACT)
        self.registered = register_scene(
            self.identity["scene_id"],
            self.identity["route_id"],
            registry_path=REGISTRY,
            terrain_dir=self.terrain_root,
            official_scene_xml=self.official,
            output_dir=self.root / "terrain-output",
            expected_official_scene_sha256=_sha256(self.official),
            expected_robot_sha256=_sha256(self.robot),
            mm_hello_identity=_mm_hello_identity(self.terrain_root),
            mm_scene_identity=self.identity,
        )

    def test_terrain_initial_state_samples_rectangular_fixed_diagonal_exactly(self) -> None:
        # At Holden z=0, pelvis x=.5 lies exactly on the fixed diagonal while
        # left/right sole centers at x=.7/.3 exercise its opposite triangles.
        boundary = _initial_boundary((0.5, 5.0, 0.0))
        state = initial_physics_state(
            self.registered, boundary, self.contract, NEUTRAL_HAND_TARGETS
        )
        self.assertAlmostEqual(state.pelvis_clearance_m, 4.45, places=6)
        self.assertAlmostEqual(state.left_foot_clearance_m, 4.185, places=6)
        self.assertAlmostEqual(state.right_foot_clearance_m, 4.245, places=6)

    def test_terrain_initial_state_rejects_root_below_hfield_base(self) -> None:
        below_base = _initial_boundary((0.0, -100.0, 0.0))
        with self.assertRaisesRegex(SceneError, "below.*terrain"):
            initial_physics_state(
                self.registered,
                below_base,
                self.contract,
                NEUTRAL_HAND_TARGETS,
            )

    def test_rotated_forbidden_geoms_cannot_hide_below_shallow_hfield_base(
        self,
    ) -> None:
        terrain_root = self.root / "shallow-terrain"
        identity = _terrain_fixture(
            terrain_root,
            scene_id="shallow-base",
            shallow=True,
        )
        registered = register_scene(
            identity["scene_id"],
            identity["route_id"],
            registry_path=REGISTRY,
            terrain_dir=terrain_root,
            official_scene_xml=self.official,
            output_dir=self.root / "shallow-output",
            expected_official_scene_sha256=_sha256(self.official),
            expected_robot_sha256=_sha256(self.robot),
            mm_hello_identity=_mm_hello_identity(terrain_root),
            mm_scene_identity=identity,
        )
        inverted = replace(
            _initial_boundary((-0.9, 0.6, -0.45)),
            physical_pelvis_orientation_holden=np.array(
                (0.0, 1.0, 0.0, 0.0), np.float64
            ),
        )
        with self.assertRaisesRegex(SceneError, "forbidden geom.*below.*terrain"):
            initial_physics_state(
                registered,
                inverted,
                self.contract,
                NEUTRAL_HAND_TARGETS,
            )

    def test_terrain_initial_state_rejects_root_outside_transformed_bounds(self) -> None:
        outside = _initial_boundary((500.0, 1.0, 0.0))
        with self.assertRaisesRegex(SceneError, "scene domain"):
            initial_physics_state(
                self.registered, outside, self.contract, NEUTRAL_HAND_TARGETS
            )


class SharedConstantTests(unittest.TestCase):
    def test_shared_gear_identities_are_frozen(self) -> None:
        self.assertEqual(SCENE_REGISTRY_PATH, REGISTRY)
        self.assertEqual(
            GEAR_SCENE_RELATIVE,
            Path("gear_sonic_deploy/g1/scene_29dof_with_hand.xml"),
        )
        self.assertEqual(
            GEAR_ROBOT_RELATIVE,
            Path("gear_sonic_deploy/g1/g1_29dof_with_hand.xml"),
        )
        self.assertEqual(
            GEAR_SCENE_SHA256,
            "f8538904eb47cada1bfb2dcdc157099092aa63df4307d7e077b651b16bfb6c74",
        )
        self.assertEqual(
            GEAR_ROBOT_SHA256,
            "8b68d8f06674c5c10cd2cd89764b3cfba9fabba5080b55ea67ee1dd12cf630cd",
        )


@unittest.skipUnless(
    os.environ.get("SONIC_REAL_SCENE_CANARY") == "1",
    "real default-scene canary is opt-in",
)
class RealDefaultSceneCanaryTests(unittest.TestCase):
    def test_register_default_scene_and_initial_state(self) -> None:
        terrain_dir = Path(os.environ["SONIC_TERRAIN_DIR"]).resolve(strict=True)
        gear_checkout = Path(
            os.environ.get("SONIC_GEAR_CHECKOUT", "/tmp/groot-wbc-plan-inspect")
        ).resolve(strict=True)
        official_scene = gear_checkout / GEAR_SCENE_RELATIVE
        official_robot = gear_checkout / GEAR_ROBOT_RELATIVE
        self.assertEqual(_sha256(official_scene), GEAR_SCENE_SHA256)
        self.assertEqual(_sha256(official_robot), GEAR_ROBOT_SHA256)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "mm_chunk_server"
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
                    '-DMM_CHUNK_BUILD_COMMIT="scene-runtime-canary"',
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
            run_root = root / "run"
            run_root.mkdir()
            environment = dict(os.environ)
            environment["SONIC_TERRAIN_DIR"] = str(terrain_dir)
            environment["SONIC_SCENE_REGISTRY"] = str(REGISTRY)
            with MMChunkClient(
                run_root=run_root,
                command=(str(executable),),
                stdout_archive=run_root / "mm.stdout",
                stderr_archive=run_root / "mm.stderr",
                env=environment,
                cwd=ROOT,
            ) as client:
                hello = client.hello()
                reset = client.reset(
                    SessionConfig("grail-curb-default", "curb-forward", 4.0),
                    session_id="scene-runtime-canary",
                )

            contract = load_joint_contract(JOINT_CONTRACT)
            initial = SourceValidator(contract).validate_initial(reset)
            registered = register_scene(
                "grail-curb-default",
                "curb-forward",
                registry_path=REGISTRY,
                terrain_dir=terrain_dir,
                official_scene_xml=official_scene,
                output_dir=run_root / "scene",
                expected_official_scene_sha256=GEAR_SCENE_SHA256,
                expected_robot_sha256=GEAR_ROBOT_SHA256,
                mm_hello_identity=hello,
                mm_scene_identity=reset["scene"],
            )
            state = initial_physics_state(
                registered, initial, contract, NEUTRAL_HAND_TARGETS
            )
            model = mujoco.MjModel.from_xml_path(str(registered.gear_scene_xml))
            terrain_geom = int(model.geom("mm_terrain").id)
            self.assertEqual(
                int(model.geom_type[terrain_geom]),
                int(mujoco.mjtGeom.mjGEOM_HFIELD),
            )
            visual_geom = int(model.geom("mm_terrain_visual").id)
            self.assertEqual(
                int(model.geom_type[visual_geom]), int(mujoco.mjtGeom.mjGEOM_MESH)
            )
            self.assertEqual(int(model.geom_contype[visual_geom]), 0)
            self.assertEqual(int(model.geom_conaffinity[visual_geom]), 0)
            self.assertEqual(
                sum(
                    model.geom(index).name == "mm_terrain"
                    for index in range(int(model.ngeom))
                ),
                1,
            )
            self.assertEqual(state.qpos.shape, (int(model.nq),))
            self.assertTrue(np.all(np.isfinite(state.qpos)))
            root_address = int(
                model.jnt_qposadr[model.joint("floating_base_joint").id]
            )
            bounds = np.asarray(registered.transformed_bounds_mujoco)
            self.assertTrue(np.all(state.qpos[root_address : root_address + 2] >= bounds[0, :2]))
            self.assertTrue(np.all(state.qpos[root_address : root_address + 2] <= bounds[1, :2]))
            self.assertTrue(
                all(
                    math.isfinite(value)
                    for value in (
                        state.pelvis_clearance_m,
                        state.left_foot_clearance_m,
                        state.right_foot_clearance_m,
                    )
                )
            )
            print(
                "authenticated terrain collision + render mesh: "
                f"scene={registered.scene_id} collision_geom={terrain_geom} "
                f"visual_geom={visual_geom} "
                f"qpos_sha256={state.qpos_sha256}",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
