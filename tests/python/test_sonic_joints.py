import hashlib
import json
import os
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np

from mm_sonic.joints import (
    ContractError,
    PINNED_TARGET_TO_SOURCE_PERMUTATION,
    SOURCE_JOINT_ORDER,
    TARGET_JOINT_ORDER,
    contract_json_bytes,
    generate_joint_contract,
    load_joint_contract,
    reorder_source_to_target,
    write_joint_contract,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTERED_SOURCE_MJCF = Path(
    "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
)
REGISTERED_SOURCE_SHA256 = (
    "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376"
)
PINNED_GEAR_ROOT = Path("/tmp/groot-wbc-plan-inspect")
PINNED_GEAR_SCENE_RELATIVE = Path(
    "gear_sonic_deploy/g1/scene_29dof_with_hand.xml"
)
PINNED_GEAR_ROBOT_RELATIVE = Path(
    "gear_sonic_deploy/g1/g1_29dof_with_hand.xml"
)
PINNED_GEAR_SCENE_SHA256 = (
    "f8538904eb47cada1bfb2dcdc157099092aa63df4307d7e077b651b16bfb6c74"
)
PINNED_GEAR_ROBOT_SHA256 = (
    "8b68d8f06674c5c10cd2cd89764b3cfba9fabba5080b55ea67ee1dd12cf630cd"
)
TARGET_ORDER_RELATIVE = Path(
    "gear_sonic/envs/manager_env/robots/g1.py"
)
TARGET_ORDER_SOURCE_SHA256 = (
    "42f9b4394eb21512735da4b1ca78c2d0546c4221c7bdaf8b3a186d8dc5588ba9"
)
COMMITTED_CONTRACT = ROOT / "sonic/configs/g1_joint_contract.json"

EXPECTED_SOURCE_JOINTS = (
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

EXPECTED_TARGET_JOINTS = (
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

EXPECTED_TARGET_TO_SOURCE = (
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

EXPECTED_HOLDEN_BONES = (
    "LeftHipPitch",
    "LeftHipRoll",
    "LeftHipYaw",
    "LeftKnee",
    "LeftAnkle",
    "LeftToe",
    "RightHipPitch",
    "RightHipRoll",
    "RightHipYaw",
    "RightKnee",
    "RightAnkle",
    "RightToe",
    "Spine",
    "Spine1",
    "Spine2",
    "LeftShoulderPitch",
    "LeftShoulderRoll",
    "LeftShoulderYaw",
    "LeftElbow",
    "LeftWristRoll",
    "LeftWristPitch",
    "LeftWrist",
    "RightShoulderPitch",
    "RightShoulderRoll",
    "RightShoulderYaw",
    "RightElbow",
    "RightWristRoll",
    "RightWristPitch",
    "RightWrist",
)

EXPECTED_HOLDEN_PARENTS = (
    "Hips",
    "LeftHipPitch",
    "LeftHipRoll",
    "LeftHipYaw",
    "LeftKnee",
    "LeftAnkle",
    "Hips",
    "RightHipPitch",
    "RightHipRoll",
    "RightHipYaw",
    "RightKnee",
    "RightAnkle",
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "LeftShoulderPitch",
    "LeftShoulderRoll",
    "LeftShoulderYaw",
    "LeftElbow",
    "LeftWristRoll",
    "LeftWristPitch",
    "Spine2",
    "RightShoulderPitch",
    "RightShoulderRoll",
    "RightShoulderYaw",
    "RightElbow",
    "RightWristRoll",
    "RightWristPitch",
)

Q_ZUP_TO_HOLDEN = np.array(
    [2.0**-0.5, -(2.0**-0.5), 0.0, 0.0],
    np.float64,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quat_mul(left, right):
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


def _quat_inverse(value):
    value = np.asarray(value, np.float64)
    output = value.copy()
    output[..., 1:] *= -1.0
    return output


def _quat_rotate(rotation, vector):
    rotation = np.asarray(rotation, np.float64)
    vector = np.asarray(vector, np.float64)
    xyz = rotation[..., 1:]
    first = 2.0 * np.cross(xyz, vector)
    return vector + rotation[..., :1] * first + np.cross(xyz, first)


def _quat_from_angle_axis(angle, axis):
    angle = np.asarray(angle, np.float64)
    axis = np.asarray(axis, np.float64)
    return np.concatenate(
        (
            np.cos(0.5 * angle)[..., None],
            np.sin(0.5 * angle)[..., None] * axis,
        ),
        axis=-1,
    )


def _quat_distance(left, right):
    delta = _quat_mul(_quat_inverse(left), right)
    delta = np.where(delta[..., :1] < 0.0, -delta, delta)
    return 2.0 * np.arctan2(
        np.linalg.norm(delta[..., 1:], axis=-1),
        np.abs(delta[..., 0]),
    )


def _source_hinge_ids(model):
    return tuple(
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id])
        == int(mujoco.mjtJoint.mjJNT_HINGE)
    )


def _model_joint_names(model, joint_ids):
    return tuple(model.joint(joint_id).name for joint_id in joint_ids)


def _holden_inputs_from_source_fk(model, data, qpos):
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    body_ids = np.arange(1, model.nbody)
    source_positions = data.xpos[body_ids].copy()
    source_rotations = data.xquat[body_ids].copy()
    basis = np.broadcast_to(Q_ZUP_TO_HOLDEN, source_rotations.shape)
    basis_inverse = np.broadcast_to(
        _quat_inverse(Q_ZUP_TO_HOLDEN), source_rotations.shape
    )
    global_positions = np.zeros((31, 3), np.float64)
    global_rotations = np.zeros((31, 4), np.float64)
    global_rotations[:, 0] = 1.0
    global_positions[1:] = _quat_rotate(basis, source_positions)
    global_rotations[1:] = _quat_mul(
        _quat_mul(basis, source_rotations), basis_inverse
    )

    local_rotations = np.zeros((31, 4), np.float64)
    local_rotations[:, 0] = 1.0
    for body_id in body_ids:
        parent_id = int(model.body_parentid[body_id])
        if parent_id == 0:
            local_rotations[body_id] = global_rotations[body_id]
        else:
            local_rotations[body_id] = _quat_mul(
                _quat_inverse(global_rotations[parent_id]),
                global_rotations[body_id],
            )
    local_angular_velocities = np.zeros((31, 3), np.float64)
    return (
        local_rotations,
        local_angular_velocities,
        global_positions,
        global_rotations,
    )


def _float_tuple(values):
    return tuple(float(value) for value in values)


def _spec_joint_topology(spec):
    return tuple(
        (joint.name, joint.parent.name, int(joint.type))
        for joint in spec.joints
    )


def _spec_joint_semantics(spec):
    return tuple(
        (
            joint.name,
            (
                ("body", joint.parent.name),
                ("default_class", joint.classname.name),
                ("type", int(joint.type)),
                ("align", int(joint.align)),
                ("limited", int(joint.limited)),
                ("actfrclimited", int(joint.actfrclimited)),
                ("actgravcomp", int(joint.actgravcomp)),
                ("group", int(joint.group)),
                ("pos", _float_tuple(joint.pos)),
                ("axis", _float_tuple(joint.axis)),
                ("range", _float_tuple(joint.range)),
                ("ref", float(joint.ref)),
                ("springref", float(joint.springref)),
                ("stiffness", _float_tuple(joint.stiffness)),
                ("springdamper", _float_tuple(joint.springdamper)),
                ("damping", _float_tuple(joint.damping)),
                ("armature", float(joint.armature)),
                ("frictionloss", float(joint.frictionloss)),
                ("margin", float(joint.margin)),
                ("solref_limit", _float_tuple(joint.solref_limit)),
                ("solimp_limit", _float_tuple(joint.solimp_limit)),
                ("solref_friction", _float_tuple(joint.solref_friction)),
                ("solimp_friction", _float_tuple(joint.solimp_friction)),
                ("actfrcrange", _float_tuple(joint.actfrcrange)),
                ("userdata", _float_tuple(joint.userdata)),
            ),
        )
        for joint in spec.joints
    )


def _model_joint_topology(model):
    return tuple(
        (
            model.joint(joint_id).name,
            model.body(int(model.jnt_bodyid[joint_id])).name,
            int(model.jnt_type[joint_id]),
        )
        for joint_id in range(model.njnt)
    )


def _compile_pinned_gear_joint_model(scene_path):
    scene_root = ET.fromstring(scene_path.read_bytes())
    include_files = [
        element.attrib.get("file")
        for element in scene_root.findall("include")
    ]
    if include_files != ["g1_29dof_with_hand.xml"]:
        raise AssertionError(f"unexpected pinned GEAR include: {include_files}")

    spec = mujoco.MjSpec.from_file(str(scene_path))
    before_bodies = tuple(body.name for body in spec.bodies)
    before_joint_topology = _spec_joint_topology(spec)
    before_joint_semantics = _spec_joint_semantics(spec)
    joint_references = {
        joint.name: float(joint.ref) for joint in spec.joints if joint.name
    }
    mesh_geoms = [
        geom
        for geom in list(spec.geoms)
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH
    ]
    meshes = list(spec.meshes)
    if not mesh_geoms or not meshes:
        raise AssertionError("pinned GEAR scene unexpectedly has no mesh visuals")
    for geom in mesh_geoms:
        spec.delete(geom)
    for mesh in meshes:
        spec.delete(mesh)
    if tuple(body.name for body in spec.bodies) != before_bodies:
        raise AssertionError("visual stripping changed GEAR bodies")
    if _spec_joint_semantics(spec) != before_joint_semantics:
        raise AssertionError("visual stripping changed GEAR joint semantics")
    model = spec.compile()
    if _model_joint_topology(model) != before_joint_topology:
        raise AssertionError(
            "compiled GEAR joint topology differs from stripped spec"
        )
    return model, len(mesh_geoms), len(meshes), joint_references


class _MutatingSpecProxy:
    def __init__(self, spec, *, delete_mutator=None, compiled_mutator=None):
        self._spec = spec
        self._delete_mutator = delete_mutator
        self._compiled_mutator = compiled_mutator
        self._delete_mutated = False

    def __getattr__(self, name):
        return getattr(self._spec, name)

    def delete(self, element):
        result = self._spec.delete(element)
        if self._delete_mutator is not None and not self._delete_mutated:
            self._delete_mutator(self._spec)
            self._delete_mutated = True
        return result

    def compile(self):
        model = self._spec.compile()
        if self._compiled_mutator is not None:
            self._compiled_mutator(model)
        return model


def _invoke_projection_cli(cli, contract_path, poses):
    chunks = [str(len(poses))]
    for pose in poses:
        for values in pose:
            chunks.extend(format(float(value), ".17g") for value in values.flat)
    completed = subprocess.run(
        (str(cli), str(contract_path)),
        input=" ".join(chunks) + "\n",
        capture_output=True,
        check=False,
        text=True,
        timeout=90,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "projection CLI failed with code "
            f"{completed.returncode}: {completed.stderr}"
        )
    output = np.fromstring(completed.stdout, sep=" ")
    if output.size == 0 or int(output[0]) != len(poses):
        raise AssertionError(
            f"projection CLI returned an invalid pose count: {completed.stdout[:200]}"
        )
    values_per_pose = 29 + 29 + 29 + 3 + 4
    expected = 1 + len(poses) * values_per_pose
    if output.size != expected:
        raise AssertionError(
            f"projection CLI returned {output.size} values, expected {expected}"
        )
    return output[1:].reshape(len(poses), values_per_pose)


class SonicJointContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        requested_source = Path(
            os.environ.get("SONIC_SOURCE_MJCF", str(REGISTERED_SOURCE_MJCF))
        ).resolve(strict=True)
        cls.source_path = requested_source
        cls.gear_root = Path(
            os.environ.get("SONIC_GEAR_CHECKOUT", str(PINNED_GEAR_ROOT))
        ).resolve(strict=True)
        cls.gear_scene = cls.gear_root / PINNED_GEAR_SCENE_RELATIVE
        cls.gear_robot = cls.gear_root / PINNED_GEAR_ROBOT_RELATIVE
        cls.target_order_source = cls.gear_root / TARGET_ORDER_RELATIVE
        cls.source_model = mujoco.MjModel.from_xml_path(str(cls.source_path))

    def test_source_names_and_pinned_target_permutation_are_exact(self):
        self.assertEqual(self.source_path, REGISTERED_SOURCE_MJCF)
        self.assertEqual(_sha256(self.source_path), REGISTERED_SOURCE_SHA256)
        hinge_ids = _source_hinge_ids(self.source_model)
        source_names = _model_joint_names(self.source_model, hinge_ids)
        self.assertEqual(len(hinge_ids), 29)
        self.assertEqual(len(set(source_names)), 29)
        self.assertEqual(source_names, EXPECTED_SOURCE_JOINTS)
        self.assertEqual(SOURCE_JOINT_ORDER, EXPECTED_SOURCE_JOINTS)
        self.assertEqual(TARGET_JOINT_ORDER, EXPECTED_TARGET_JOINTS)
        self.assertEqual(
            PINNED_TARGET_TO_SOURCE_PERMUTATION,
            EXPECTED_TARGET_TO_SOURCE,
        )
        self.assertEqual(
            tuple(SOURCE_JOINT_ORDER[index] for index in EXPECTED_TARGET_TO_SOURCE),
            TARGET_JOINT_ORDER,
        )

    def test_generation_is_byte_deterministic_and_matches_committed_contract(self):
        first = generate_joint_contract(
            self.source_path, self.target_order_source
        )
        second = generate_joint_contract(
            self.source_path, self.target_order_source
        )
        first_bytes = contract_json_bytes(first)
        second_bytes = contract_json_bytes(second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertTrue(first_bytes.endswith(b"\n"))
        self.assertNotIn(b": ", first_bytes)
        self.assertNotIn(b", ", first_bytes)
        self.assertEqual(
            first.source_mjcf_sha256,
            REGISTERED_SOURCE_SHA256,
        )
        self.assertEqual(
            first.target_order_source_sha256,
            TARGET_ORDER_SOURCE_SHA256,
        )
        self.assertEqual(len(first.rows), 29)
        self.assertEqual(
            tuple(row.source_index for row in first.rows), tuple(range(29))
        )
        self.assertEqual(
            tuple(row.source_joint for row in first.rows),
            EXPECTED_SOURCE_JOINTS,
        )
        self.assertEqual(
            tuple(row.source_bone for row in first.rows),
            EXPECTED_HOLDEN_BONES,
        )
        self.assertEqual(
            tuple(row.source_parent for row in first.rows),
            EXPECTED_HOLDEN_PARENTS,
        )
        self.assertEqual(
            tuple(row.qpos_address for row in first.rows),
            tuple(range(7, 36)),
        )
        self.assertEqual(
            tuple(first.rows[index].target_index for index in EXPECTED_TARGET_TO_SOURCE),
            tuple(range(29)),
        )
        self.assertEqual(
            tuple(first.rows[index].target_name for index in EXPECTED_TARGET_TO_SOURCE),
            EXPECTED_TARGET_JOINTS,
        )
        for row in first.rows:
            self.assertAlmostEqual(np.linalg.norm(row.axis_holden), 1.0, places=12)
            self.assertAlmostEqual(
                np.linalg.norm(row.static_local_holden_wxyz), 1.0, places=12
            )
            first_nonzero = next(
                value
                for value in row.static_local_holden_wxyz
                if value != 0.0
            )
            self.assertGreater(first_nonzero, 0.0)
            self.assertEqual(row.sign, 1.0)
            self.assertEqual(row.zero_offset, 0.0)
            self.assertLess(row.lower, row.upper)

        with tempfile.TemporaryDirectory() as temporary:
            first_path = Path(temporary) / "first.json"
            second_path = Path(temporary) / "second.json"
            write_joint_contract(first, first_path)
            write_joint_contract(second, second_path)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertEqual(first_path.read_bytes(), first_bytes)
            self.assertEqual(load_joint_contract(first_path), first)

        self.assertEqual(COMMITTED_CONTRACT.read_bytes(), first_bytes)
        self.assertEqual(load_joint_contract(COMMITTED_CONTRACT), first)
        raw = json.loads(first_bytes)
        self.assertEqual(list(raw), sorted(raw))
        for row in raw["rows"]:
            self.assertEqual(list(row), sorted(row))

    def test_pinned_gear_with_hand_scene_has_identical_body_joint_semantics(self):
        self.assertEqual(
            self.gear_scene.resolve(strict=True),
            (PINNED_GEAR_ROOT / PINNED_GEAR_SCENE_RELATIVE).resolve(),
        )
        self.assertEqual(
            self.gear_robot.resolve(strict=True),
            (PINNED_GEAR_ROOT / PINNED_GEAR_ROBOT_RELATIVE).resolve(),
        )
        self.assertEqual(_sha256(self.gear_scene), PINNED_GEAR_SCENE_SHA256)
        self.assertEqual(_sha256(self.gear_robot), PINNED_GEAR_ROBOT_SHA256)
        gear_model, removed_geoms, removed_meshes, gear_references = (
            _compile_pinned_gear_joint_model(self.gear_scene)
        )
        source_spec = mujoco.MjSpec.from_file(str(self.source_path))
        source_references = {
            joint.name: float(joint.ref)
            for joint in source_spec.joints
            if joint.name
        }
        self.assertGreater(removed_geoms, 0)
        self.assertGreater(removed_meshes, 0)

        source_ids = _source_hinge_ids(self.source_model)
        source_by_name = {
            self.source_model.joint(joint_id).name: joint_id
            for joint_id in source_ids
        }
        source_free_ids = tuple(
            joint_id
            for joint_id in range(self.source_model.njnt)
            if self.source_model.jnt_type[joint_id]
            == mujoco.mjtJoint.mjJNT_FREE
        )
        gear_free_ids = tuple(
            joint_id
            for joint_id in range(gear_model.njnt)
            if gear_model.jnt_type[joint_id]
            == mujoco.mjtJoint.mjJNT_FREE
        )
        self.assertEqual(len(source_free_ids), 1)
        self.assertEqual(len(gear_free_ids), 1)
        source_free_id = source_free_ids[0]
        gear_free_id = gear_free_ids[0]
        self.assertEqual(
            (
                self.source_model.joint(source_free_id).name,
                self.source_model.body(
                    int(self.source_model.jnt_bodyid[source_free_id])
                ).name,
            ),
            ("floating_base_joint", "pelvis"),
        )
        self.assertEqual(
            (
                gear_model.joint(gear_free_id).name,
                gear_model.body(int(gear_model.jnt_bodyid[gear_free_id])).name,
            ),
            ("floating_base_joint", "pelvis"),
        )
        gear_ids = _source_hinge_ids(gear_model)
        gear_names = _model_joint_names(gear_model, gear_ids)
        gear_by_name = {gear_model.joint(name).name: gear_model.joint(name).id for name in gear_names}
        self.assertEqual(len(gear_names), len(set(gear_names)))
        self.assertEqual(set(EXPECTED_SOURCE_JOINTS) - set(gear_names), set())
        extras = set(gear_names) - set(EXPECTED_SOURCE_JOINTS)
        self.assertEqual(len(extras), 14)
        self.assertTrue(all("_hand_" in name for name in extras))

        for name in EXPECTED_SOURCE_JOINTS:
            source_id = source_by_name[name]
            gear_id = gear_by_name[name]
            self.assertEqual(
                gear_model.body(int(gear_model.jnt_bodyid[gear_id])).name,
                self.source_model.body(
                    int(self.source_model.jnt_bodyid[source_id])
                ).name,
                name,
            )
            self.assertEqual(
                int(self.source_model.jnt_type[source_id]),
                int(mujoco.mjtJoint.mjJNT_HINGE),
                name,
            )
            self.assertEqual(
                int(gear_model.jnt_type[gear_id]),
                int(mujoco.mjtJoint.mjJNT_HINGE),
                name,
            )
            np.testing.assert_array_equal(
                gear_model.jnt_axis[gear_id],
                self.source_model.jnt_axis[source_id],
                err_msg=name,
            )
            np.testing.assert_array_equal(
                gear_model.jnt_range[gear_id],
                self.source_model.jnt_range[source_id],
                err_msg=name,
            )
            self.assertEqual(
                int(gear_model.jnt_limited[gear_id]),
                int(self.source_model.jnt_limited[source_id]),
                name,
            )
            self.assertEqual(source_references[name], 0.0, name)
            self.assertEqual(gear_references[name], 0.0, name)
            self.assertEqual(
                float(
                    self.source_model.qpos0[
                        self.source_model.jnt_qposadr[source_id]
                    ]
                ),
                0.0,
                name,
            )
            self.assertEqual(
                float(gear_model.qpos0[gear_model.jnt_qposadr[gear_id]]),
                0.0,
                name,
            )

        print(
            "Pinned GEAR joint semantics: "
            "scene=scene_29dof_with_hand.xml body_joints=29 "
            "free_base=floating_base_joint@pelvis "
            f"extra_hand_joints={len(extras)} removed_mesh_geoms={removed_geoms} "
            f"removed_mesh_assets={removed_meshes}"
        )

    def test_joint_only_strip_rejects_joint_default_mutation(self):
        spec = mujoco.MjSpec.from_file(str(self.gear_scene))

        def mutate_joint_default(resolved_spec):
            joint = resolved_spec.joint("left_knee_joint")
            joint.damping[0] = float(joint.damping[0]) + 0.001

        proxy = _MutatingSpecProxy(
            spec,
            delete_mutator=mutate_joint_default,
        )
        with mock.patch.object(mujoco.MjSpec, "from_file", return_value=proxy):
            with self.assertRaisesRegex(
                AssertionError,
                "visual stripping changed GEAR joint semantics",
            ):
                _compile_pinned_gear_joint_model(self.gear_scene)

    def test_joint_only_strip_rejects_compiled_topology_mutations(self):
        def move_required_hinge(model):
            joint_id = model.joint("left_knee_joint").id
            model.jnt_bodyid[joint_id] = model.body("pelvis").id

        def relocate_free_base(model):
            joint_id = model.joint("floating_base_joint").id
            model.jnt_bodyid[joint_id] = model.body("left_hip_pitch_link").id

        def remove_free_base(model):
            joint_id = model.joint("floating_base_joint").id
            model.jnt_type[joint_id] = mujoco.mjtJoint.mjJNT_HINGE

        def add_extra_free_base(model):
            joint_id = model.joint("left_knee_joint").id
            model.jnt_type[joint_id] = mujoco.mjtJoint.mjJNT_FREE

        def replace_free_base(model):
            free_id = model.joint("floating_base_joint").id
            replacement_id = model.joint("left_knee_joint").id
            model.jnt_type[free_id] = mujoco.mjtJoint.mjJNT_HINGE
            model.jnt_type[replacement_id] = mujoco.mjtJoint.mjJNT_FREE

        cases = (
            ("wrong required hinge body", move_required_hinge),
            ("relocated free base", relocate_free_base),
            ("missing free base", remove_free_base),
            ("extra free base", add_extra_free_base),
            ("replaced free base", replace_free_base),
        )
        for label, mutator in cases:
            with self.subTest(label=label):
                spec = mujoco.MjSpec.from_file(str(self.gear_scene))
                proxy = _MutatingSpecProxy(
                    spec,
                    compiled_mutator=mutator,
                )
                with mock.patch.object(
                    mujoco.MjSpec,
                    "from_file",
                    return_value=proxy,
                ):
                    with self.assertRaisesRegex(
                        AssertionError,
                        "compiled GEAR joint topology differs from stripped spec",
                    ):
                        _compile_pinned_gear_joint_model(self.gear_scene)

    def test_source_to_target_reorder_is_name_driven(self):
        contract = load_joint_contract(COMMITTED_CONTRACT)
        source = np.arange(29, dtype=np.float64)
        expected = source[np.asarray(EXPECTED_TARGET_TO_SOURCE)]
        np.testing.assert_array_equal(
            reorder_source_to_target(source, contract), expected
        )
        batch = np.stack((source, source + 100.0))
        np.testing.assert_array_equal(
            reorder_source_to_target(batch, contract),
            batch[:, np.asarray(EXPECTED_TARGET_TO_SOURCE)],
        )
        with self.assertRaisesRegex(ContractError, "expected 29 source joints"):
            reorder_source_to_target(np.arange(28), contract)

    def test_cpp_projection_certifies_mujoco_samples(self):
        cli = Path(os.environ["SONIC_PROJECT_CLI"]).resolve(strict=True)
        contract = load_joint_contract(COMMITTED_CONTRACT)
        model = self.source_model
        data = mujoco.MjData(model)
        base = model.qpos0.copy()
        lower = np.array([row.lower for row in contract.rows], np.float64)
        upper = np.array([row.upper for row in contract.rows], np.float64)
        qpos_addresses = np.array(
            [row.qpos_address for row in contract.rows], np.int64
        )

        samples = []
        for values in (
            np.zeros(29, np.float64),
            lower + 1.0e-5,
            upper - 1.0e-5,
            0.5 * (lower + upper),
        ):
            qpos = base.copy()
            qpos[qpos_addresses] = values
            samples.append(qpos)
        rng = np.random.default_rng(20260715)
        for values in rng.uniform(
            lower + 1.0e-5,
            upper - 1.0e-5,
            size=(256, 29),
        ):
            qpos = base.copy()
            qpos[qpos_addresses] = values
            samples.append(qpos)
        self.assertEqual(len(samples), 260)

        poses = []
        expected_local = []
        expected_global_positions = []
        expected_global_rotations = []
        for qpos in samples:
            pose = _holden_inputs_from_source_fk(model, data, qpos)
            poses.append(pose)
            expected_local.append(pose[0])
            expected_global_positions.append(pose[2])
            expected_global_rotations.append(pose[3])
        output = _invoke_projection_cli(cli, COMMITTED_CONTRACT, poses)
        source_position = output[:, :29]
        source_velocity = output[:, 29:58]
        off_axis = output[:, 58:87]
        pelvis_position = output[:, 87:90]
        pelvis_rotation = output[:, 90:94]

        expected_source = np.stack(samples)[:, qpos_addresses]
        joint_errors = np.abs(source_position - expected_source)
        maximum_joint_error = float(np.max(joint_errors))
        self.assertLessEqual(maximum_joint_error, 1.0e-4)
        np.testing.assert_array_equal(source_velocity, np.zeros_like(source_velocity))
        self.assertLessEqual(float(np.max(off_axis)), 1.0e-4)

        reconstructed = np.empty((len(samples), 29, 4), np.float64)
        for source_index, row in enumerate(contract.rows):
            angle = (
                source_position[:, source_index] - row.zero_offset
            ) / row.sign
            twist = _quat_from_angle_axis(
                angle, np.asarray(row.axis_holden, np.float64)
            )
            static = np.broadcast_to(
                np.asarray(row.static_local_holden_wxyz, np.float64),
                twist.shape,
            )
            reconstructed[:, source_index] = _quat_mul(static, twist)
        source_bones = np.array(
            [EXPECTED_HOLDEN_BONES.index(row.source_bone) + 2 for row in contract.rows],
            np.int64,
        )
        expected_local_array = np.stack(expected_local)[:, source_bones]
        local_errors = _quat_distance(reconstructed, expected_local_array)
        maximum_local_error = float(np.max(local_errors))
        self.assertLessEqual(maximum_local_error, 1.0e-4)

        expected_target = expected_source[:, np.asarray(EXPECTED_TARGET_TO_SOURCE)]
        projected_target = reorder_source_to_target(source_position, contract)
        self.assertLessEqual(
            float(np.max(np.abs(projected_target - expected_target))),
            1.0e-4,
        )
        expected_global_positions = np.stack(expected_global_positions)
        expected_global_rotations = np.stack(expected_global_rotations)
        self.assertLessEqual(
            float(
                np.max(
                    np.abs(pelvis_position - expected_global_positions[:, 1])
                )
            ),
            1.0e-5,
        )
        self.assertLessEqual(
            float(
                np.max(
                    _quat_distance(
                        pelvis_rotation, expected_global_rotations[:, 1]
                    )
                )
            ),
            1.0e-5,
        )
        self.assertEqual(
            {row.source_joint for row in contract.rows},
            set(EXPECTED_SOURCE_JOINTS),
        )
        self.assertEqual(
            {row.target_name for row in contract.rows},
            set(EXPECTED_TARGET_JOINTS),
        )
        print(
            "SONIC G1 projection certification: "
            f"samples={len(samples)} exact_source_names=29 exact_target_names=29 "
            f"max_joint_angle_error_rad={maximum_joint_error:.9g} "
            f"max_reconstructed_local_rotation_error_rad={maximum_local_error:.9g} "
            f"max_off_axis_residual_rad={float(np.max(off_axis)):.9g}"
        )


if __name__ == "__main__":
    unittest.main()
