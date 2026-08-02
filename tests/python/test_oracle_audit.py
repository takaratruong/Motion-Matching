from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import unittest

import numpy as np

from mm_sonic.terrain_oracle.audit import (
    AuditThresholds,
    audit_clip,
    structural_model_sha256,
)
from mm_sonic.terrain_oracle.canonical import (
    CanonicalTerrainMesh,
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    TerrainBinding,
    derive_clip_kinematics,
)
from mm_sonic.terrain_oracle.contact import CanonicalMeshQuery
from mm_sonic.terrain_oracle.math3d import RigidTransform
from tests.python.terrain_oracle_test_utils import synthetic_canonical_clip


ROOT = Path(__file__).resolve().parents[2]
LEFT_ANKLE = ISAACLAB_BODY_NAMES.index("left_ankle_roll_link")
RIGHT_ANKLE = ISAACLAB_BODY_NAMES.index("right_ankle_roll_link")


class _Named:
    def __init__(self, identifier: int, name: str) -> None:
        self.id = identifier
        self.name = name


class _FakeG1Model:
    """Small mechanical model with deliberately noncanonical joint ordering."""

    def __init__(self) -> None:
        self._body_names = ("world", *ISAACLAB_BODY_NAMES)
        self._joint_names = ("floating_base_joint", *reversed(ISAACLAB_JOINT_NAMES))
        self._geom_names = (
            "pelvis_collision_mesh",
            *(
                f"{side}_sole_{corner}"
                for side in ("left", "right")
                for corner in range(4)
            ),
        )
        self.nbody = len(self._body_names)
        self.njnt = len(self._joint_names)
        self.ngeom = len(self._geom_names)
        self.nmesh = 1
        self.nq = 36
        self.nv = 35
        body_id = {name: index for index, name in enumerate(self._body_names)}

        self.body_parentid = np.zeros(self.nbody, dtype=np.int32)
        self.body_parentid[body_id["pelvis"]] = 0
        self.body_parentid[2:] = body_id["pelvis"]
        self.body_parentid[body_id["left_ankle_roll_link"]] = body_id[
            "left_ankle_pitch_link"
        ]
        self.body_parentid[body_id["right_ankle_roll_link"]] = body_id[
            "right_ankle_pitch_link"
        ]

        self.jnt_type = np.array((0, *(3 for _ in ISAACLAB_JOINT_NAMES)), np.int32)
        self.jnt_limited = np.array((0, *(1 for _ in ISAACLAB_JOINT_NAMES)), np.uint8)
        self.jnt_range = np.zeros((self.njnt, 2), np.float64)
        self.jnt_range[1:] = (-1.0, 1.0)
        self.jnt_qposadr = np.array((0, *range(7, 36)), np.int32)
        self.jnt_bodyid = np.empty(self.njnt, np.int32)
        self.jnt_bodyid[0] = body_id["pelvis"]
        for index, name in enumerate(self._joint_names[1:], start=1):
            link = name.removesuffix("_joint") + "_link"
            if name == "waist_pitch_joint":
                link = "torso_link"
            self.jnt_bodyid[index] = body_id[link]

        cube = np.array(
            [
                (-0.05, -0.05, -0.05),
                (0.05, -0.05, -0.05),
                (0.05, 0.05, -0.05),
                (-0.05, 0.05, -0.05),
                (-0.05, -0.05, 0.05),
                (0.05, -0.05, 0.05),
                (0.05, 0.05, 0.05),
                (-0.05, 0.05, 0.05),
            ],
            np.float64,
        )
        self.mesh_vert = cube
        self.mesh_face = np.array(
            [
                (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
                (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
            ],
            np.int32,
        )
        self.mesh_vertadr = np.array((0,), np.int32)
        self.mesh_vertnum = np.array((len(cube),), np.int32)
        self.mesh_faceadr = np.array((0,), np.int32)
        self.mesh_facenum = np.array((len(self.mesh_face),), np.int32)
        self.mesh_pos = np.zeros((1, 3), np.float64)
        self.mesh_quat = np.array(((1.0, 0.0, 0.0, 0.0),), np.float64)
        self.mesh_scale = np.ones((1, 3), np.float64)

        self.geom_bodyid = np.empty(self.ngeom, np.int32)
        self.geom_bodyid[0] = body_id["pelvis"]
        self.geom_bodyid[1:5] = body_id["left_ankle_roll_link"]
        self.geom_bodyid[5:9] = body_id["right_ankle_roll_link"]
        self.geom_type = np.array((7, *(2 for _ in range(8))), np.int32)
        self.geom_dataid = np.array((0, *(-1 for _ in range(8))), np.int32)
        self.geom_contype = np.ones(self.ngeom, np.int32)
        self.geom_conaffinity = np.ones(self.ngeom, np.int32)
        self.geom_pos = np.zeros((self.ngeom, 3), np.float64)
        corners = np.array(
            (
                (-0.10, 0.04, -0.03),
                (-0.10, -0.04, -0.03),
                (0.10, 0.04, -0.03),
                (0.10, -0.04, -0.03),
            ),
            np.float64,
        )
        self.geom_pos[1:5] = corners
        self.geom_pos[5:9] = corners
        self.geom_quat = np.zeros((self.ngeom, 4), np.float64)
        self.geom_quat[:, 0] = 1.0
        self.geom_size = np.zeros((self.ngeom, 3), np.float64)
        self.geom_size[1:, 0] = 0.005

    def _named(self, names: tuple[str, ...], value: int | str) -> _Named:
        if isinstance(value, str):
            try:
                value = names.index(value)
            except ValueError as error:
                raise KeyError(value) from error
        return _Named(int(value), names[int(value)])

    def body(self, value: int | str) -> _Named:
        return self._named(self._body_names, value)

    def joint(self, value: int | str) -> _Named:
        return self._named(self._joint_names, value)

    def geom(self, value: int | str) -> _Named:
        return self._named(self._geom_names, value)


def _identity_transform() -> RigidTransform:
    return RigidTransform(
        np.zeros(3, np.float32),
        np.array((1.0, 0.0, 0.0, 0.0), np.float32),
    )


def _plane_query(*, toe_only: bool = False) -> CanonicalMeshQuery:
    minimum_x = 0.05 if toe_only else -5.0
    mesh = CanonicalTerrainMesh(
        vertices_local=np.array(
            (
                (minimum_x, -5.0, 0.0),
                (5.0, -5.0, 0.0),
                (5.0, 5.0, 0.0),
                (minimum_x, 5.0, 0.0),
            ),
            np.float32,
        ),
        faces=np.array(((0, 1, 2), (0, 2, 3)), np.int32),
        valid_faces=np.array((True, True)),
        source_asset_sha256="a" * 64,
    )
    return CanonicalMeshQuery(mesh, _identity_transform())


def _finite_triangle_query() -> CanonicalMeshQuery:
    mesh = CanonicalTerrainMesh(
        vertices_local=np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            np.float32,
        ),
        faces=np.array(((0, 1, 2),), np.int32),
        valid_faces=np.array((True,)),
        source_asset_sha256="b" * 64,
    )
    return CanonicalMeshQuery(mesh, _identity_transform())


def _clean_clip(
    query: CanonicalMeshQuery, *, frames: int = 40
):
    clip = synthetic_canonical_clip(frames=frames)
    root_position = np.zeros((frames, 3), np.float32)
    root_position[:, 2] = 0.8
    body_position = np.zeros((frames, 30, 3), np.float32)
    body_position[:, :, 2] = 0.8
    body_position[:, LEFT_ANKLE] = (0.0, 0.20, 0.035)
    body_position[:, RIGHT_ANKLE] = (0.0, -0.20, 0.035)
    sole_position = np.empty((frames, 2, 3), np.float32)
    sole_position[:, 0] = (0.0, 0.20, 0.0)
    sole_position[:, 1] = (0.0, -0.20, 0.0)
    heel_position = sole_position + np.array((-0.10, 0.0, 0.0), np.float32)
    toe_position = sole_position + np.array((0.10, 0.0, 0.0), np.float32)
    binding = TerrainBinding(
        asset_path="/read-only/plane.obj",
        asset_size_bytes=128,
        asset_sha256=query.source_asset_sha256,
        asset_license_id="CC0-1.0",
        mesh_sha256=query.mesh_sha256,
        world_from_terrain=query.world_from_terrain,
        validity_mask_path=None,
    )
    return derive_clip_kinematics(
        replace(
            clip,
            root_position_world=root_position,
            body_position_world=body_position,
            sole_position_world=sole_position,
            heel_position_world=heel_position,
            toe_position_world=toe_position,
            contact=np.ones((frames, 2), np.float32),
            terrain=binding,
        )
    )


def _reason_codes(report: object) -> set[str]:
    return {reason.code for reason in report.reasons}


class OracleAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _FakeG1Model()
        self.query = _plane_query()
        self.clip = _clean_clip(self.query)

    def test_clean_clip_is_accepted_deterministic_and_immutable(self):
        """Catches nondeterministic reports or mutable accepted evidence."""

        first = audit_clip(self.clip, self.model, self.query)
        second = audit_clip(self.clip, self.model, self.query)

        self.assertEqual(first, second)
        self.assertEqual(first.status, "accepted")
        self.assertEqual(first.accepted_intervals, ((0, 40),))
        self.assertEqual(first.reasons, ())
        self.assertEqual(first.source_sha256, self.clip.source.source_sha256)
        self.assertEqual(len(first.model_sha256), 64)
        self.assertEqual(len(first.terrain_sha256), 64)
        self.assertEqual(first.to_dict(), second.to_dict())
        json.dumps(first.to_dict(), allow_nan=False, sort_keys=True)
        with self.assertRaises(FrozenInstanceError):
            first.status = "rejected"
        with self.assertRaises(TypeError):
            first.metrics["max_joint_speed_rad_s"] = 1.0
        with self.assertRaises(FrozenInstanceError):
            AuditThresholds().guard_frames = 9

    def test_antipodal_quaternions_are_continuous_but_real_jumps_are_not(self):
        """Catches component-wise quaternion jumps and unchecked body/sole traces."""

        antipodal = replace(
            self.clip,
            root_quaternion_world_wxyz=np.where(
                (np.arange(40) % 2)[:, None] == 0,
                self.clip.root_quaternion_world_wxyz,
                -self.clip.root_quaternion_world_wxyz,
            ),
            body_quaternion_world_wxyz=np.where(
                (np.arange(40) % 2)[:, None, None] == 0,
                self.clip.body_quaternion_world_wxyz,
                -self.clip.body_quaternion_world_wxyz,
            ),
            sole_quaternion_world_wxyz=np.where(
                (np.arange(40) % 2)[:, None, None] == 0,
                self.clip.sole_quaternion_world_wxyz,
                -self.clip.sole_quaternion_world_wxyz,
            ),
        )
        self.assertEqual(
            audit_clip(antipodal, self.model, self.query).status, "accepted"
        )

        jump = np.array(
            (np.cos(0.75), 0.0, np.sin(0.75), 0.0), np.float32
        )
        for field, index in (
            ("root_quaternion_world_wxyz", (slice(20, None), slice(None))),
            ("body_quaternion_world_wxyz", (slice(20, None), 29, slice(None))),
            ("sole_quaternion_world_wxyz", (slice(20, None), 1, slice(None))),
        ):
            with self.subTest(field=field):
                values = np.array(getattr(self.clip, field), copy=True)
                values[index] = jump
                report = audit_clip(
                    replace(self.clip, **{field: values}),
                    self.model,
                    self.query,
                )
                self.assertIn("quaternion_discontinuity", _reason_codes(report))

    def test_joint_limit_and_velocity_are_classified_separately_by_name(self):
        """Catches unchecked indices or conflation of position and speed limits."""

        limited = np.array(self.clip.joint_position, copy=True)
        limited[:, ISAACLAB_JOINT_NAMES.index("left_hip_pitch_joint")] = 1.2
        limit_report = audit_clip(
            derive_clip_kinematics(replace(self.clip, joint_position=limited)),
            self.model,
            self.query,
        )
        self.assertIn("joint_limit", _reason_codes(limit_report))
        self.assertNotIn("joint_velocity", _reason_codes(limit_report))

        fast = np.array(self.clip.joint_position, copy=True)
        fast[:, ISAACLAB_JOINT_NAMES.index("right_wrist_yaw_joint")] = (
            np.arange(40) * 0.9
        )
        velocity_report = audit_clip(
            derive_clip_kinematics(replace(self.clip, joint_position=fast)),
            self.model,
            self.query,
        )
        self.assertIn("joint_velocity", _reason_codes(velocity_report))

    def test_contact_support_penetration_and_skate_are_independent(self):
        """Catches trusting labels, height-only support, or body-origin collision."""

        mismatch = replace(self.clip, contact=np.zeros((40, 2), np.float32))
        self.assertIn(
            "contact_inconsistency",
            _reason_codes(audit_clip(mismatch, self.model, self.query)),
        )

        toe_query = _plane_query(toe_only=True)
        toe_clip = replace(
            self.clip,
            terrain=replace(
                self.clip.terrain,
                asset_sha256=toe_query.source_asset_sha256,
                mesh_sha256=toe_query.mesh_sha256,
            ),
        )
        toe_codes = _reason_codes(audit_clip(toe_clip, self.model, toe_query))
        self.assertIn("incomplete_sole_support", toe_codes)
        self.assertIn("contact_inconsistency", toe_codes)

        low_feet = np.array(self.clip.body_position_world, copy=True)
        low_feet[:, (LEFT_ANKLE, RIGHT_ANKLE), 2] = 0.025
        foot_report = audit_clip(
            derive_clip_kinematics(
                replace(self.clip, body_position_world=low_feet)
            ),
            self.model,
            self.query,
        )
        self.assertIn("foot_penetration", _reason_codes(foot_report))
        self.assertNotIn("body_penetration", _reason_codes(foot_report))

        low_pelvis = np.array(self.clip.body_position_world, copy=True)
        low_pelvis[:, 0, 2] = -0.02
        body_report = audit_clip(
            derive_clip_kinematics(
                replace(self.clip, body_position_world=low_pelvis)
            ),
            self.model,
            self.query,
        )
        self.assertIn("body_penetration", _reason_codes(body_report))

        skating = np.array(self.clip.body_position_world, copy=True)
        skating[:, (LEFT_ANKLE, RIGHT_ANKLE), 0] = (
            np.arange(40, dtype=np.float32)[:, None] * 0.01
        )
        skate_report = audit_clip(
            derive_clip_kinematics(
                replace(self.clip, body_position_world=skating)
            ),
            self.model,
            self.query,
        )
        self.assertIn("stance_skate", _reason_codes(skate_report))

    def test_derivative_mismatch_and_root_implausibility_are_distinct(self):
        """Catches trusting serialized derivatives or omitting root mechanics."""

        stored = np.array(self.clip.root_linear_velocity_world, copy=True)
        stored[10, 0] = 2.0
        derivative = audit_clip(
            replace(self.clip, root_linear_velocity_world=stored),
            self.model,
            self.query,
        )
        self.assertEqual(_reason_codes(derivative), {"derivative_discontinuity"})

        root = np.array(self.clip.root_position_world, copy=True)
        root[:, 0] = np.arange(40, dtype=np.float32) * 0.2
        implausible = audit_clip(
            derive_clip_kinematics(replace(self.clip, root_position_world=root)),
            self.model,
            self.query,
        )
        self.assertIn("root_plausibility", _reason_codes(implausible))

    def test_interval_trimming_is_guarded_half_open_and_all_bad_rejects(self):
        """Catches off-by-one guards, short fragment acceptance, and bad status."""

        joint = np.array(self.clip.joint_position, copy=True)
        joint[20, 0] = 1.2
        corrupt = audit_clip(
            replace(self.clip, joint_position=joint),
            self.model,
            self.query,
        )
        self.assertEqual(
            corrupt.status, "accepted_with_intervals_removed"
        )
        self.assertEqual(corrupt.accepted_intervals, ((0, 18), (23, 40)))
        self.assertEqual(
            [reason.frame_interval for reason in corrupt.reasons if reason.code == "joint_limit"],
            [(20, 21)],
        )

        all_bad_joint = np.array(self.clip.joint_position, copy=True)
        all_bad_joint[:, 0] = 1.2
        rejected = audit_clip(
            derive_clip_kinematics(
                replace(self.clip, joint_position=all_bad_joint)
            ),
            self.model,
            self.query,
        )
        self.assertEqual(rejected.status, "rejected")
        self.assertEqual(rejected.accepted_intervals, ())

    def test_combined_corruption_has_exact_expected_reason_set(self):
        """Catches cross-contamination among orientation, collision, and stance."""

        sole_quaternion = np.array(
            self.clip.sole_quaternion_world_wxyz, copy=True
        )
        sole_quaternion[20:, 1] = np.array(
            (np.cos(0.75), np.sin(0.75), 0.0, 0.0), np.float32
        )
        body_position = np.array(self.clip.body_position_world, copy=True)
        body_position[:, 0, 2] = -0.02
        body_position[:, (LEFT_ANKLE, RIGHT_ANKLE), 0] = (
            np.arange(40, dtype=np.float32)[:, None] * 0.01
        )
        corrupted = derive_clip_kinematics(
            replace(
                self.clip,
                sole_quaternion_world_wxyz=sole_quaternion,
                body_position_world=body_position,
            )
        )

        report = audit_clip(corrupted, self.model, self.query)

        self.assertEqual(report.status, "rejected")
        self.assertEqual(
            _reason_codes(report),
            {"quaternion_discontinuity", "body_penetration", "stance_skate"},
        )

    def test_model_query_and_terrain_provenance_mismatches_fail_closed(self):
        """Catches audit evidence produced against unrelated mechanics or terrain."""

        missing = audit_clip(replace(self.clip, terrain=None), self.model, self.query)
        self.assertEqual(missing.status, "rejected")
        self.assertEqual(_reason_codes(missing), {"terrain_registration"})

        other_query = _plane_query(toe_only=True)
        mismatched = audit_clip(self.clip, self.model, other_query)
        self.assertEqual(mismatched.status, "rejected")
        self.assertEqual(_reason_codes(mismatched), {"terrain_registration"})

        moved_query = CanonicalMeshQuery(
            self.query.mesh,
            RigidTransform(
                np.array((0.0, 0.0, 0.1), np.float32),
                np.array((1.0, 0.0, 0.0, 0.0), np.float32),
            ),
        )
        moved = audit_clip(self.clip, self.model, moved_query)
        self.assertEqual(moved.status, "rejected")
        self.assertEqual(_reason_codes(moved), {"terrain_registration"})

        bad_model = _FakeG1Model()
        bad_model._body_names = (
            *bad_model._body_names[:-1],
            "wrong_body_name",
        )
        wrong_model = audit_clip(self.clip, bad_model, self.query)
        self.assertEqual(wrong_model.status, "rejected")
        self.assertEqual(_reason_codes(wrong_model), {"model_registration"})

    def test_structural_model_hash_changes_with_used_mechanical_property(self):
        """Catches a model hash that covers names but not collision mechanics."""

        original = structural_model_sha256(self.model)
        changed = _FakeG1Model()
        changed.mesh_vert[0, 2] -= 0.001
        self.assertNotEqual(original, structural_model_sha256(changed))
        changed_range = _FakeG1Model()
        changed_range.jnt_range[1, 0] -= 0.001
        self.assertNotEqual(original, structural_model_sha256(changed_range))

    def test_mesh_collision_uses_exact_triangle_crossing_and_finite_broadphase(self):
        """Catches vertex-only collision and infinite supporting-plane false hits."""

        query = _finite_triangle_query()
        clip = _clean_clip(query)
        model = _FakeG1Model()
        model.mesh_vert = np.array(
            (
                (-0.20, 0.25, -0.02),
                (1.20, 0.25, -0.02),
                (0.25, 0.25, 0.10),
            ),
            np.float64,
        )
        model.mesh_face = np.array(((0, 1, 2),), np.int32)
        model.mesh_vertnum[:] = 3
        model.mesh_facenum[:] = 1
        body = np.array(clip.body_position_world, copy=True)
        body[:, 0] = 0.0
        crossing = derive_clip_kinematics(
            replace(clip, body_position_world=body)
        )
        self.assertIn(
            "body_penetration",
            _reason_codes(audit_clip(crossing, model, query)),
        )

        body[:, 0, 0] = 3.0
        outside = derive_clip_kinematics(
            replace(clip, body_position_world=body)
        )
        self.assertNotIn(
            "body_penetration",
            _reason_codes(audit_clip(outside, model, query)),
        )

    def test_schema_accepts_exact_report_and_rejects_invalid_documents(self):
        """Catches schema/report drift and permissive hashes, status, or intervals."""

        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest("jsonschema is unavailable")
        schema = json.loads(
            (
                ROOT / "sonic/schemas/terrain_oracle_audit_v1.schema.json"
            ).read_text("utf-8")
        )
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        document = audit_clip(self.clip, self.model, self.query).to_dict()
        validator.validate(document)
        mutations = []
        extra = json.loads(json.dumps(document))
        extra["extra"] = 1
        mutations.append(extra)
        bad_hash = json.loads(json.dumps(document))
        bad_hash["model_sha256"] = "A" * 64
        mutations.append(bad_hash)
        bad_status = json.loads(json.dumps(document))
        bad_status["status"] = "okay"
        mutations.append(bad_status)
        bad_interval = json.loads(json.dumps(document))
        bad_interval["accepted_intervals"] = [[-1, 10]]
        mutations.append(bad_interval)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assertFalse(validator.is_valid(mutation))


if __name__ == "__main__":
    unittest.main()
