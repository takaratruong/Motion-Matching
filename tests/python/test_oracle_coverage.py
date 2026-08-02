from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from mm_sonic.joints import ContractError
import mm_sonic.terrain_oracle.coverage as coverage_module
from mm_sonic.terrain_oracle.audit import AuditReason, AuditThresholds, ClipAudit
from mm_sonic.terrain_oracle.canonical import CanonicalTerrainMesh, TerrainBinding
from mm_sonic.terrain_oracle.contact import (
    CONTACT_RECONSTRUCTION_TAG,
    CanonicalMeshQuery,
)
from mm_sonic.terrain_oracle.coverage import (
    ACTION_CLASSES,
    AcceptedClipRecord,
    CoverageContribution,
    CoverageManifest,
    assign_grouped_splits,
    build_coverage,
    deduplicate,
    freeze_coverage,
)
from mm_sonic.terrain_oracle.math3d import RigidTransform
from mm_sonic.terrain_oracle.storage import write_clip
from mm_sonic.terrain_oracle.storage import ClipRecord, clip_digest
from tests.python.terrain_oracle_test_utils import synthetic_canonical_clip


ROOT = Path(__file__).resolve().parents[2]


def _hash_text(digest, value):
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _hash_array(digest, value, dtype):
    array = np.ascontiguousarray(value, dtype=dtype)
    _hash_text(digest, array.shape)
    digest.update(array.tobytes(order="C"))


def _query_hash(query):
    digest = hashlib.sha256(b"terrain-oracle-exact-query-surface/v1\0")
    _hash_text(digest, query.mesh_sha256)
    _hash_text(digest, query.source_asset_sha256)
    _hash_array(digest, query._triangles, np.float64)
    _hash_array(digest, query._normals, np.float64)
    _hash_array(digest, query._face_indices, np.int64)
    _hash_array(digest, query.world_from_terrain.translation_world, np.float64)
    _hash_array(
        digest,
        query.world_from_terrain.quaternion_world_from_local_wxyz,
        np.float64,
    )
    return digest.hexdigest()


class CoverageFixture:
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.mesh = CanonicalTerrainMesh(
            vertices_local=np.array(
                ((-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)),
                np.float32,
            ),
            faces=np.array(((0, 1, 2), (0, 2, 3)), np.int32),
            valid_faces=np.array((True, True)),
            source_asset_sha256="a" * 64,
        )
        self.transform = RigidTransform(
            translation_world=np.zeros(3, np.float32),
            quaternion_world_from_local_wxyz=np.array((1, 0, 0, 0), np.float32),
        )
        self.query = CanonicalMeshQuery(self.mesh, self.transform)
        self.model_sha = "b" * 64

    def tearDown(self):
        self._temporary.cleanup()

    def record(
        self,
        clip_id,
        *,
        source_sha=None,
        terrain_asset_sha=None,
        procedural_family_id=None,
        mirror_of=None,
        action_class="walk",
        intervals=None,
        x_speed=0.4,
        y_speed=0.0,
        yaw_rate=0.0,
        contact_pattern=None,
        z_profile=None,
        sole_yaw_by_leg=(0.0, 0.0),
        sole_z_by_leg=(0.0, 0.0),
        heel_z_offset_by_leg=(0.0, 0.0),
        toe_z_offset_by_leg=(0.0, 0.0),
    ):
        frames = 14
        clip = synthetic_canonical_clip(frames=frames)
        source_sha = source_sha or hashlib.sha256(clip_id.encode()).hexdigest()
        asset_sha = terrain_asset_sha or self.mesh.source_asset_sha256
        mesh = replace(self.mesh, source_asset_sha256=asset_sha)
        query = CanonicalMeshQuery(mesh, self.transform)
        timeline = np.arange(frames, dtype=np.float32) / np.float32(50.0)
        root = np.zeros((frames, 3), np.float32)
        root[:, 0] = timeline * np.float32(x_speed)
        root[:, 1] = timeline * np.float32(y_speed)
        root[:, 2] = np.float32(0.8)
        if z_profile is not None:
            root[:, 2] += np.asarray(z_profile, np.float32)
        yaw = timeline * np.float32(yaw_rate)
        quat = np.zeros((frames, 4), np.float32)
        quat[:, 0] = np.cos(yaw / 2)
        quat[:, 3] = np.sin(yaw / 2)
        contact = np.zeros((frames, 2), np.float32)
        if contact_pattern is None:
            contact[:, 0] = 1
        else:
            contact[:] = np.asarray(contact_pattern, np.float32)
        sole = np.zeros((frames, 2, 3), np.float32)
        sole[:, :, :2] = root[:, None, :2]
        sole[:, 0, 1] += 0.1
        sole[:, 1, 1] -= 0.1
        sole[:, :, 2] = np.asarray(sole_z_by_leg, np.float32)
        # Touchdowns carry independently hand-authored step displacement.
        sole[:, 0, 0] += np.arange(frames, dtype=np.float32) * 0.02
        sole[:, 1, 0] += np.arange(frames, dtype=np.float32) * 0.01
        heel = sole.copy()
        heel[:, :, 0] -= 0.1
        heel[:, :, 2] += np.asarray(heel_z_offset_by_leg, np.float32)
        toe = sole.copy()
        toe[:, :, 0] += 0.1
        toe[:, :, 2] += np.asarray(toe_z_offset_by_leg, np.float32)
        sole_quaternion = np.zeros((frames, 2, 4), np.float32)
        for leg in range(2):
            sole_quaternion[:, leg, 0] = np.cos(
                np.float32(sole_yaw_by_leg[leg]) / 2
            )
            sole_quaternion[:, leg, 3] = np.sin(
                np.float32(sole_yaw_by_leg[leg]) / 2
            )
        terrain = TerrainBinding(
            asset_path=f"/readonly/{asset_sha}.obj",
            asset_size_bytes=10,
            asset_sha256=asset_sha,
            asset_license_id="CC0-1.0",
            mesh_sha256=query.mesh_sha256,
            world_from_terrain=self.transform,
            validity_mask_path=None,
        )
        clip = replace(
            clip,
            clip_id=clip_id,
            source=replace(
                clip.source,
                source_path=f"synthetic://{clip_id}",
                source_sha256=source_sha,
            ),
            root_position_world=root,
            root_quaternion_world_wxyz=quat,
            root_linear_velocity_world=np.tile(
                np.array((x_speed, y_speed, 0), np.float32), (frames, 1)
            ),
            root_angular_velocity_world=np.tile(
                np.array((0, 0, yaw_rate), np.float32), (frames, 1)
            ),
            sole_position_world=sole,
            sole_quaternion_world_wxyz=sole_quaternion,
            heel_position_world=heel,
            toe_position_world=toe,
            contact=contact,
            terrain=terrain,
            mirror_of=mirror_of,
            action_tags=("motion", CONTACT_RECONSTRUCTION_TAG),
        )
        clip.validate()
        artifact = write_clip(
            self.root / f"artifact-{len(list(self.root.iterdir()))}" / "clips",
            clip,
        )
        intervals = tuple(intervals or ((0, frames),))
        status = "accepted" if intervals == ((0, frames),) else "accepted_with_intervals_removed"
        reasons = ()
        if status != "accepted":
            reasons = (
                AuditReason("joint_velocity", "error", (intervals[0][1], intervals[-1][0]), 2, 1),
            )
        audit = ClipAudit(
            schema="terrain-oracle-audit/v1",
            clip_id=clip_id,
            frame_count=frames,
            source_sha256=source_sha,
            clip_sha256=artifact.sha256,
            model_sha256=self.model_sha,
            terrain_sha256=_query_hash(query),
            status=status,
            accepted_intervals=intervals,
            thresholds=AuditThresholds().to_dict(),
            metrics={
                name: 0.0
                for name in (
                    "max_joint_limit_violation_rad", "max_joint_speed_rad_s",
                    "max_quaternion_step_rad", "max_root_speed_m_s",
                    "min_root_height_m", "max_root_height_m",
                    "max_root_acceleration_m_s2", "max_root_angular_speed_rad_s",
                    "max_derivative_error", "max_contact_disagreement",
                    "max_incomplete_sole_fraction", "max_stance_skate_m",
                    "max_foot_penetration_m", "max_body_penetration_m",
                    "max_body_fk_position_error_m",
                    "max_body_fk_orientation_error_rad",
                )
            },
            reasons=reasons,
        )
        return AcceptedClipRecord(
            clip=clip,
            clip_record=artifact,
            audit=audit,
            accepted_intervals=intervals,
            terrain_mesh=mesh,
            terrain_query=query,
            model_sha256=self.model_sha,
            action_class=action_class,
            procedural_family_id=procedural_family_id,
        )


class OracleCoverageTests(CoverageFixture, unittest.TestCase):
    def test_semantic_byte_copy_deduplicates_despite_changed_identity(self):
        original = self.record("walk", terrain_asset_sha="1" * 64)
        copied_clip = replace(
            original.clip,
            clip_id="copy",
            source=replace(
                original.clip.source,
                source_path="synthetic://copy",
                source_sha256="2" * 64,
            ),
        )
        artifact = write_clip(self.root / "copy" / "clips", copied_clip)
        audit = replace(
            original.audit,
            clip_id="copy",
            source_sha256="2" * 64,
            clip_sha256=artifact.sha256,
        )
        copied = replace(
            original,
            clip=copied_clip,
            clip_record=artifact,
            audit=audit,
        )
        result = deduplicate((copied, original, copied))
        self.assertEqual(len(result.unique_records), 1)
        self.assertEqual(result.semantic_digest_by_clip["walk"], result.semantic_digest_by_clip["copy"])

    def test_mirror_is_not_a_duplicate_but_never_crosses_split(self):
        original = self.record("walk", terrain_asset_sha="1" * 64)
        mirror = self.record(
            "walk__mirror",
            terrain_asset_sha="2" * 64,
            mirror_of="walk",
            y_speed=-0.2,
        )
        self.assertEqual(len(deduplicate((original, mirror)).unique_records), 2)
        split = assign_grouped_splits((mirror, original), "oracle-v1")
        self.assertEqual(split["walk"], split["walk__mirror"])

    def test_source_terrain_procedural_and_transitive_constraints_never_leak(self):
        a = self.record("a", source_sha="1" * 64, terrain_asset_sha="a" * 64)
        b = self.record("b", source_sha="1" * 64, terrain_asset_sha="b" * 64)
        c = self.record("c", source_sha="3" * 64, terrain_asset_sha="b" * 64)
        d = self.record("d", procedural_family_id="family", terrain_asset_sha="d" * 64)
        e = self.record("e", procedural_family_id="family", terrain_asset_sha="e" * 64)
        split = assign_grouped_splits((e, c, a, d, b), "oracle-v1")
        self.assertEqual(len({split[x] for x in ("a", "b", "c")}), 1)
        self.assertEqual(split["d"], split["e"])
        self.assertEqual(split.to_dict(), assign_grouped_splits((a, b, c, d, e), "oracle-v1").to_dict())

    def test_unrelated_groups_distribute_by_published_sha256_boundaries(self):
        records = tuple(
            self.record(f"independent-{i}", source_sha=f"{i + 1:064x}", terrain_asset_sha=f"{i + 100:064x}")
            for i in range(40)
        )
        split = assign_grouped_splits(records, "fixed-seed")
        self.assertGreaterEqual(len(set(split.assignments.values())), 2)
        self.assertEqual(split.proportions, (0.8, 0.1, 0.1))

    def test_conflicting_same_clip_id_and_invalid_seed_fail_closed(self):
        a = self.record("same", terrain_asset_sha="1" * 64)
        b = self.record("same", terrain_asset_sha="2" * 64, x_speed=0.9)
        with self.assertRaisesRegex(ContractError, "conflicting"):
            deduplicate((a, b))
        for seed in ("", 7, True):
            with self.subTest(seed=seed), self.assertRaises(ContractError):
                assign_grouped_splits((a,), seed)

    def test_rejected_mismatched_stale_or_outside_interval_evidence_is_rejected(self):
        record = self.record("checked", terrain_asset_sha="1" * 64)
        cases = (
            {"model_sha256": "f" * 64},
            {"accepted_intervals": ((0, 13),)},
            {"audit": replace(record.audit, source_sha256="f" * 64)},
            {"audit": replace(record.audit, status="rejected", accepted_intervals=(), reasons=(
                AuditReason("joint_velocity", "fatal", (0, 14), 2, 1),
            ))},
            {"clip": replace(record.clip, action_tags=("motion",))},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                replace(record, **changes)

    def test_audit_binds_exact_clip_artifact_and_canonical_record_path(self):
        record = self.record("artifact-bound", terrain_asset_sha="1" * 64)
        changed_root = np.array(record.clip.root_position_world, copy=True)
        changed_root[4, 0] += np.float32(0.125)
        changed_command = np.array(
            record.clip.commands.inferred_yaw_rate_rad_s, copy=True
        )
        changed_command[4] += np.float32(0.25)
        changed_contact = np.array(record.clip.contact, copy=True)
        changed_contact[4] = changed_contact[4, ::-1]
        changed_clips = (
            replace(record.clip, root_position_world=changed_root),
            replace(
                record.clip,
                commands=replace(
                    record.clip.commands,
                    inferred_yaw_rate_rad_s=changed_command,
                ),
            ),
            replace(record.clip, contact=changed_contact),
        )
        for changed in changed_clips:
            fresh = ClipRecord(
                relative_path=f"clips/{clip_digest(changed)}.npz",
                sha256=clip_digest(changed),
                clip_id=changed.clip_id,
                frame_count=changed.frame_count,
            )
            with self.subTest(digest=fresh.sha256), self.assertRaisesRegex(
                ContractError, "audit|artifact|digest"
            ):
                replace(record, clip=changed, clip_record=fresh)

        forged_path = replace(
            record.clip_record,
            relative_path=f"elsewhere/{record.clip_record.sha256}.npz",
        )
        with self.assertRaisesRegex(ContractError, "relative_path"):
            replace(record, clip_record=forged_path)

    def test_coverage_has_every_axis_and_preserves_ascent_descent_support(self):
        pattern = np.zeros((14, 2), np.float32)
        pattern[:4, 0] = 1
        pattern[4:8, 1] = 1
        pattern[8:10] = 1
        ascent = self.record(
            "up", action_class="ascent", contact_pattern=pattern,
            z_profile=np.linspace(0, .3, 14), x_speed=0.8, yaw_rate=0.5,
            terrain_asset_sha="1" * 64,
        )
        descent = self.record(
            "down", action_class="descent", contact_pattern=pattern[::-1],
            z_profile=np.linspace(.3, 0, 14), x_speed=-0.4, yaw_rate=-0.5,
            terrain_asset_sha="2" * 64,
        )
        manifest = build_coverage((ascent, descent))
        self.assertEqual(
            set(manifest.bin_definitions),
            {
                "movement_facing_offset", "planar_speed", "yaw_rate",
                "transition", "support_phase", "support_leg", "step_forward",
                "step_lateral", "step_vertical", "surface_normal",
                "contact_yaw", "swing_clearance", "duration", "action_class",
            },
        )
        self.assertIn("ascent", manifest.marginal_counts["action_class"])
        self.assertIn("descent", manifest.marginal_counts["action_class"])
        self.assertGreater(len(manifest.occupied_cells), 0)

    def test_repetition_byte_duplicates_and_input_order_do_not_change_coverage(self):
        up = self.record("up", action_class="ascent", terrain_asset_sha="1" * 64)
        down = self.record("down", action_class="descent", terrain_asset_sha="2" * 64, x_speed=-.4)
        one = build_coverage((up, down))
        many = build_coverage((down, up, up, down, up))
        self.assertEqual(one.to_dict(), many.to_dict())

    def test_lexically_earlier_identity_copy_does_not_change_manifest(self):
        original = self.record(
            "z_original", action_class="ascent", terrain_asset_sha="1" * 64
        )
        copied_clip = replace(
            original.clip,
            clip_id="a_copy",
            source=replace(
                original.clip.source,
                source_path="synthetic://identity-copy",
                source_sha256="f" * 64,
            ),
        )
        artifact = write_clip(self.root / "identity-copy" / "clips", copied_clip)
        copied = replace(
            original,
            clip=copied_clip,
            clip_record=artifact,
            audit=replace(
                original.audit,
                clip_id="a_copy",
                source_sha256="f" * 64,
                clip_sha256=artifact.sha256,
            ),
        )

        expected = build_coverage((original,))
        for records in (
            (original, copied),
            (copied, original),
            (copied, original, copied, original),
        ):
            with self.subTest(records=tuple(item.clip.clip_id for item in records)):
                actual = build_coverage(records)
                self.assertEqual(actual, expected)
                self.assertEqual(actual.to_dict(), expected.to_dict())
                self.assertEqual(actual.content_sha256, expected.content_sha256)

    def test_accepted_interval_gap_is_never_bridged(self):
        pattern = np.zeros((14, 2), np.float32)
        pattern[1:4, 0] = 1
        pattern[10:13, 0] = 1
        record = self.record(
            "gap", intervals=((0, 5), (9, 14)), contact_pattern=pattern,
            terrain_asset_sha="1" * 64,
        )
        manifest = build_coverage((record,))
        self.assertNotIn("reverse", manifest.marginal_counts["transition"])
        self.assertTrue(all(cell.interval in ((0, 5), (9, 14)) for cell in manifest.occupied_cells))

    def test_stationary_flight_and_nonunique_swing_publish_none(self):
        flight = np.zeros((14, 2), np.float32)
        record = self.record(
            "stationary-flight",
            x_speed=0.0,
            contact_pattern=flight,
            sole_z_by_leg=(0.2, 0.3),
            terrain_asset_sha="1" * 64,
        )
        manifest = build_coverage((record,))
        coordinates = [cell.to_dict()["coordinates"] for cell in manifest.occupied_cells]
        self.assertTrue(coordinates)
        self.assertEqual({row["movement_facing_offset"] for row in coordinates}, {"none"})
        self.assertEqual({row["surface_normal"] for row in coordinates}, {"none"})
        self.assertEqual({row["contact_yaw"] for row in coordinates}, {"none"})
        self.assertEqual({row["swing_clearance"] for row in coordinates}, {"none"})

    def test_right_contact_uses_right_yaw_and_double_support_emits_both_legs(self):
        right_only = np.zeros((14, 2), np.float32)
        right_only[:, 1] = 1
        right = self.record(
            "right-only",
            contact_pattern=right_only,
            sole_yaw_by_leg=(0.0, 1.2),
            terrain_asset_sha="1" * 64,
        )
        right_rows = [
            cell.to_dict()["coordinates"] for cell in build_coverage((right,)).occupied_cells
        ]
        self.assertEqual({row["support_leg"] for row in right_rows}, {"right"})
        self.assertNotIn("[0,0.392699)", {row["contact_yaw"] for row in right_rows})

        double = np.ones((14, 2), np.float32)
        both = self.record(
            "both",
            contact_pattern=double,
            sole_yaw_by_leg=(-1.2, 1.2),
            terrain_asset_sha="2" * 64,
        )
        both_rows = [
            cell.to_dict()["coordinates"] for cell in build_coverage((both,)).occupied_cells
        ]
        self.assertEqual({row["support_leg"] for row in both_rows}, {"left", "right"})
        self.assertGreaterEqual(len({row["contact_yaw"] for row in both_rows}), 2)

    def test_swing_clearance_uses_conservative_sole_heel_toe_geometry(self):
        left_support = np.zeros((14, 2), np.float32)
        left_support[:, 0] = 1
        record = self.record(
            "clearance-probes",
            contact_pattern=left_support,
            sole_z_by_leg=(0.0, 0.20),
            heel_z_offset_by_leg=(0.0, -0.19),
            toe_z_offset_by_leg=(0.0, 0.10),
            terrain_asset_sha="1" * 64,
        )
        rows = [
            cell.to_dict()["coordinates"] for cell in build_coverage((record,)).occupied_cells
        ]
        self.assertEqual({row["swing_clearance"] for row in rows}, {"[0,0.015)"})

    def test_simultaneous_touchdowns_preserve_independent_step_displacements(self):
        pattern = np.zeros((14, 2), np.float32)
        pattern[0:2] = 1
        pattern[10:12] = 1
        record = self.record(
            "two-touchdowns",
            contact_pattern=pattern,
            terrain_asset_sha="1" * 64,
        )
        rows = [
            cell.to_dict()["coordinates"] for cell in build_coverage((record,)).occupied_cells
        ]
        touchdown = [row for row in rows if row["step_forward"] != "none"]
        self.assertEqual({row["support_leg"] for row in touchdown}, {"left", "right"})
        self.assertGreaterEqual(len({row["step_forward"] for row in touchdown}), 2)

    def test_boundaries_use_deterministic_underflow_overflow_and_disconnected_components(self):
        slow = self.record("slow", x_speed=0.0, terrain_asset_sha="1" * 64)
        fast = self.record("fast", x_speed=99.0, yaw_rate=-99, terrain_asset_sha="2" * 64)
        manifest = build_coverage((fast, slow))
        speed = manifest.marginal_counts["planar_speed"]
        self.assertIn("[0,0.1)", speed)
        self.assertIn("[2.5,+inf)", speed)
        self.assertGreaterEqual(len(manifest.connected_components), 2)

    def test_manifest_hash_from_dict_schema_and_atomic_no_overwrite(self):
        manifest = build_coverage((self.record("one", terrain_asset_sha="1" * 64),))
        document = manifest.to_dict()
        self.assertEqual(CoverageManifest.from_dict(document), manifest)
        mutated = json.loads(json.dumps(document))
        mutated["marginal_counts"]["action_class"]["walk"] += 1
        with self.assertRaisesRegex(ContractError, "marginal|content hash"):
            CoverageManifest.from_dict(mutated)

        schema = json.loads(
            (ROOT / "sonic/schemas/terrain_oracle_coverage_v1.schema.json").read_text()
        )
        validator_roots = sorted(
            (ROOT / ".superpowers/sdd/2026-08-01-terrain-oracle-01-audited-corpus").glob(
                ".schema-validation.*"
            )
        )
        self.assertTrue(validator_roots, "Task 2 isolated schema validator is missing")
        invalid_documents = []
        invalid = json.loads(json.dumps(document))
        invalid["unknown"] = 1
        invalid_documents.append(invalid)
        invalid = json.loads(json.dumps(document))
        invalid["bin_definitions"]["planar_speed"]["edges"][1] = 0.11
        invalid_documents.append(invalid)
        invalid = json.loads(json.dumps(document))
        invalid["occupied_cells"][0]["coordinates"]["support_leg"] = "both"
        invalid_documents.append(invalid)
        invalid = json.loads(json.dumps(document))
        invalid["occupied_cells"][0]["contributions"][0]["unknown"] = 1
        invalid_documents.append(invalid)
        schema_path = self.root / "schema.json"
        valid_path = self.root / "valid.json"
        invalid_path = self.root / "invalid.json"
        reordered_document = json.loads(json.dumps(document))
        reordered_document["bin_definitions"] = dict(
            reversed(tuple(reordered_document["bin_definitions"].items()))
        )
        reordered_document["marginal_counts"] = dict(
            reversed(tuple(reordered_document["marginal_counts"].items()))
        )
        for cell in reordered_document["occupied_cells"]:
            cell["coordinates"] = dict(reversed(tuple(cell["coordinates"].items())))
        schema_path.write_text(json.dumps(schema))
        valid_path.write_text(json.dumps(reordered_document))
        invalid_path.write_text(json.dumps(invalid_documents))
        script = (
            "import json,sys\n"
            "from jsonschema import Draft202012Validator as V\n"
            "s=json.load(open(sys.argv[1])); good=json.load(open(sys.argv[2])); bad=json.load(open(sys.argv[3]))\n"
            "V.check_schema(s); v=V(s); v.validate(good)\n"
            "assert all(list(v.iter_errors(item)) for item in bad)\n"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(validator_roots[0])
        completed = subprocess.run(
            ["/move/u/bodow/miniconda3/envs/cloc3/bin/python", "-B", "-c", script,
             str(schema_path), str(valid_path), str(invalid_path)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

        destination = self.root / "freeze" / "coverage.json"
        freeze_coverage(destination, manifest)
        self.assertEqual(json.loads(destination.read_text()), document)
        with self.assertRaises(FileExistsError):
            freeze_coverage(destination, manifest)
        self.assertEqual(list(destination.parent.glob(f".{destination.name}.*")), [])

    def test_cells_publish_recomputable_canonical_contribution_evidence(self):
        record = self.record("evidence", terrain_asset_sha="1" * 64)
        manifest = build_coverage((record,))
        self.assertTrue(manifest.occupied_cells)
        for cell in manifest.occupied_cells:
            self.assertEqual(cell.contribution_count, len(cell.contributions))
            self.assertEqual(
                cell.dedup_group_ids,
                tuple(sorted({item.dedup_group_id for item in cell.contributions})),
            )
            self.assertEqual(
                cell.intervals,
                tuple(sorted({item.accepted_interval for item in cell.contributions})),
            )
            for contribution in cell.contributions:
                self.assertEqual(
                    contribution.semantic_digest,
                    deduplicate((record,)).semantic_digest_by_clip[record.clip.clip_id],
                )
        self.assertEqual(record.clip_record.sha256, record.audit.clip_sha256)
        self.assertEqual(record.clip.source.source_sha256, record.audit.source_sha256)

    def test_json_object_insertion_order_is_not_semantic(self):
        manifest = build_coverage((self.record("json-order", terrain_asset_sha="1" * 64),))
        document = manifest.to_dict()
        document["bin_definitions"] = dict(
            reversed(tuple(document["bin_definitions"].items()))
        )
        document["marginal_counts"] = dict(
            reversed(tuple(document["marginal_counts"].items()))
        )
        for cell in document["occupied_cells"]:
            cell["coordinates"] = dict(
                reversed(tuple(cell["coordinates"].items()))
            )

        restored = CoverageManifest.from_dict(document)
        self.assertEqual(restored, manifest)
        self.assertEqual(restored.to_dict(), manifest.to_dict())
        self.assertEqual(restored.content_sha256, manifest.content_sha256)

    def test_rehashed_manifest_relation_mutations_all_fail(self):
        manifest = build_coverage((self.record("relations", terrain_asset_sha="1" * 64),))
        original = manifest.to_dict()

        def rehash(document):
            payload = dict(document)
            payload.pop("content_sha256", None)
            document["content_sha256"] = hashlib.sha256(
                (
                    json.dumps(
                        payload,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("ascii")
            ).hexdigest()
            return document

        mutations = []
        changed = json.loads(json.dumps(original))
        changed["adjacency_rule"] = "forged"
        mutations.append(changed)
        changed = json.loads(json.dumps(original))
        changed["bin_definitions"]["planar_speed"]["edges"][1] = 0.11
        mutations.append(changed)
        changed = json.loads(json.dumps(original))
        changed["marginal_counts"]["action_class"]["walk"] += 1
        mutations.append(changed)
        changed = json.loads(json.dumps(original))
        changed["occupied_cells"][0]["contribution_count"] += 1
        mutations.append(changed)
        changed = json.loads(json.dumps(original))
        changed["occupied_cells"][0]["contributions"][0]["accepted_interval"][1] -= 1
        mutations.append(changed)
        changed = json.loads(json.dumps(original))
        changed["occupied_cells"][0]["contributions"][0]["contribution_id"] = "f" * 64
        mutations.append(changed)
        changed = json.loads(json.dumps(original))
        contribution = changed["occupied_cells"][0]["contributions"][0]
        contribution["semantic_digest"] = "f" * 64
        contribution["contribution_id"] = hashlib.sha256(
            (
                json.dumps(
                    {
                        "schema": "terrain-oracle-coverage-contribution/v1",
                        "semantic_digest": contribution["semantic_digest"],
                        "dedup_group_id": contribution["dedup_group_id"],
                        "accepted_interval": contribution["accepted_interval"],
                    },
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("ascii")
        ).hexdigest()
        mutations.append(changed)
        changed = json.loads(json.dumps(original))
        changed["occupied_cells"][0]["cell_id"] = "f" * 64
        mutations.append(changed)
        changed = json.loads(json.dumps(original))
        changed["connected_components"] = []
        mutations.append(changed)
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                CoverageManifest.from_dict(rehash(mutation))

    def test_freeze_refuses_symlink_and_cleans_temp_after_publish_exception(self):
        manifest = build_coverage((self.record("atomic", terrain_asset_sha="1" * 64),))
        parent = self.root / "atomic-freeze"
        parent.mkdir()
        sentinel = parent / "sentinel"
        sentinel.write_text("unchanged")
        destination = parent / "coverage.json"
        destination.symlink_to(sentinel)
        with self.assertRaises(FileExistsError):
            freeze_coverage(destination, manifest)
        self.assertTrue(destination.is_symlink())
        self.assertEqual(sentinel.read_text(), "unchanged")
        self.assertEqual(list(parent.glob(f".{destination.name}.*")), [])

        destination.unlink()
        with mock.patch.object(
            coverage_module,
            "_rename_noreplace",
            side_effect=OSError("injected publication failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected"):
                freeze_coverage(destination, manifest)
        self.assertFalse(destination.exists())
        self.assertEqual(list(parent.glob(f".{destination.name}.*")), [])

    def test_action_class_enum_is_closed(self):
        self.assertEqual(
            ACTION_CLASSES,
            ("walk", "run", "start", "stop", "reverse", "ascent", "descent", "turn", "sidestep", "other"),
        )
        with self.assertRaises(ContractError):
            self.record("bad-action", action_class="stairs", terrain_asset_sha="1" * 64)

    def test_reserved_mirror_suffix_requires_valid_explicit_original(self):
        original = self.record("plain", terrain_asset_sha="1" * 64)
        untagged = self.record("plain__mirror", terrain_asset_sha="2" * 64)
        missing = self.record(
            "missing__mirror", mirror_of="missing", terrain_asset_sha="3" * 64
        )
        nested = self.record(
            "plain__mirror__mirror",
            mirror_of="plain__mirror",
            terrain_asset_sha="4" * 64,
        )
        for records in ((original, untagged), (missing,), (original, nested)):
            with self.subTest(records=records), self.assertRaises(ContractError):
                assign_grouped_splits(records, "oracle-v1")

        tagged = self.record(
            "plain__mirror",
            mirror_of="plain",
            y_speed=-0.2,
            terrain_asset_sha="2" * 64,
        )
        split = assign_grouped_splits((tagged, original), "oracle-v1")
        self.assertEqual(split["plain"], split["plain__mirror"])

    def test_public_outputs_recursively_own_nested_inputs(self):
        records = (self.record("owned", terrain_asset_sha="1" * 64),)
        split = assign_grouped_splits(records, "seed")
        assignments = dict(split.assignments)
        groups = dict(split.group_ids)
        rebuilt_split = type(split)(
            split.seed, split.rule, list(split.proportions), assignments, groups
        )
        before_split = rebuilt_split.to_dict()
        assignments["owned"] = "test"
        groups["owned"] = "0" * 64
        self.assertEqual(rebuilt_split.to_dict(), before_split)
        with self.assertRaises(TypeError):
            rebuilt_split.assignments["owned"] = "test"

        manifest = build_coverage(records)
        definitions = json.loads(json.dumps(manifest.to_dict()["bin_definitions"]))
        marginals = json.loads(json.dumps(manifest.to_dict()["marginal_counts"]))
        rebuilt = CoverageManifest(
            manifest.schema,
            manifest.adjacency_rule,
            definitions,
            marginals,
            list(manifest.occupied_cells),
            [list(component) for component in manifest.connected_components],
            manifest.content_sha256,
        )
        before = rebuilt.to_dict()
        definitions["planar_speed"]["edges"][0] = -999
        marginals["action_class"]["walk"] = 999
        self.assertEqual(rebuilt.to_dict(), before)
        with self.assertRaises(TypeError):
            rebuilt.bin_definitions["planar_speed"]["kind"] = "changed"


if __name__ == "__main__":
    unittest.main()
