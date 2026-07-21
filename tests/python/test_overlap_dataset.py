import base64
import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from resources import quat as holden_quat
from resources.g1_interaction_builder.overlap_motion import (
    extract_interaction_pairs,
    extract_walking_windows,
)
from resources.g1_interaction_builder.schema import (
    G1_SKELETON,
    InteractionArtifact,
    InteractionHand,
    InteractionPhase,
)
from resources.g1_terrain_builder.schema import HoldenClip
from tools import build_g1_overlap_dataset


_CANONICAL_OFFSETS = np.frombuffer(base64.b64decode(
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACwJFBU0r08/4O9AGACpKmR+bzJ/VS9vc3MPDgx/r0AAMSkdE2gvd6XNb6P1Ay7AID9ozqamb7GEMY4AAAjJKrVj7wAwA4kAHi4JFFU0r1E/4M9ELQfLoCR+byr/VQ9eM3MPJgx/r0AgMEkSk2gvQCYNb6P1Aw7AABApNiZmb7GEMa4AIDwI4bVj7wAQJ2iAAAAAAAAAAAAAAAAP+CBux1cDz0AiAGkbQC3rNmlmzwAbImk2qOBO5F8cz4IQM29ALgMpd+aYrzGpRu9AACno+ZZ072Cd8y7WkuBPCvmpL2qSIwzXsvMPQrXI7y/c/e6tKQbPQAAQKTynn4zCGk8PQAAkKNm4gQz2aOBO418cz7EOs09AKDfpDCbYrzEpRs9AMAfJQxa0711eMw7WkuBPGDmpL0AEF+lTMzMPQrXI7y/c/c6fqUbPQAAJqVzE5myvmk8PQAAAKSIcomy"
), dtype="<f4").reshape(31, 3)


def _canonical_walking(frames: int = 500) -> HoldenClip:
    walking = HoldenClip.empty(frames=frames, bones=31)
    walking.positions[:] = _CANONICAL_OFFSETS
    walking.positions[:, 0, 0] = np.arange(frames, dtype=np.float32)
    return walking


def interaction_artifact_fixture(
    frames: int = 120, *, contact_offset: int = 29, lift_offset: int = 35,
) -> InteractionArtifact:
    """One right-hand clip whose Contact is at R+29 and Lift is post-window."""
    reach = 60
    positions = np.zeros((frames, 31, 3), np.float32)
    positions[:, 0, 0] = np.arange(frames, dtype=np.float32) * 0.01
    positions[:, 1, 0] = np.linspace(0.0, 0.08, frames, dtype=np.float32)
    rotations = np.zeros((frames, 31, 4), np.float32)
    rotations[..., 0] = 1.0
    phases = np.full(frames, int(InteractionPhase.APPROACH), np.uint8)
    if reach < frames:
        phases[reach:] = int(InteractionPhase.REACH)
    if reach + contact_offset < frames:
        phases[reach + contact_offset:] = int(InteractionPhase.CONTACT)
    if reach + lift_offset < frames:
        phases[reach + lift_offset:] = int(InteractionPhase.LIFT)
    if reach + lift_offset + 8 < frames:
        phases[reach + lift_offset + 8:] = int(InteractionPhase.HOLD)
    hand_contacts = np.zeros((frames, 2), np.uint8)
    hand_contacts[reach + contact_offset:, int(InteractionHand.RIGHT)] = 1
    object_positions = np.zeros((frames, 3), np.float32)
    object_positions[:, 0] = 1.0
    object_rotations = np.zeros((frames, 4), np.float32)
    object_rotations[:, 0] = 1.0
    return InteractionArtifact(
        fps=25,
        parents=G1_SKELETON.parents.copy(),
        range_starts=np.array([0], np.int32),
        range_stops=np.array([frames], np.int32),
        positions=positions,
        velocities=np.zeros((frames, 31, 3), np.float32),
        rotations=rotations,
        angular_velocities=np.zeros((frames, 31, 3), np.float32),
        foot_contacts=np.zeros((frames, 2), np.uint8),
        hand_contacts=hand_contacts,
        hand_dof=np.zeros((frames, 14), np.float32),
        hand_dof_velocities=np.zeros((frames, 14), np.float32),
        phases=phases,
        active_hands=np.array([int(InteractionHand.RIGHT)], np.uint8),
        time_to_contact=np.maximum(reach - np.arange(frames), 0).astype(np.float32) / 25.0,
        object_positions=object_positions,
        object_rotations=object_rotations,
        object_velocities=np.zeros((frames, 3), np.float32),
        object_angular_velocities=np.zeros((frames, 3), np.float32),
        table_positions=np.zeros((1, 3), np.float32),
        table_rotations=np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        table_sizes=np.ones((1, 3), np.float32),
        object_dimensions=np.array([[0.1, 0.2, 0.3]], np.float32),
        grasp_positions_object=np.array([[0.01, 0.02, 0.03]], np.float32),
        grasp_rotations_object=np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        approach_directions_object=np.array([[1.0, 0.0, 0.0]], np.float32),
        source_frames=np.arange(1000, 1000 + frames, dtype=np.int32),
    )


def _yaw_quaternion(angle: float) -> np.ndarray:
    return holden_quat.from_angle_axis(
        np.asarray(angle, np.float32), np.asarray([0.0, 1.0, 0.0], np.float32)
    ).astype(np.float32)


def _globally_transform_interaction(
    artifact: InteractionArtifact, *, translation: np.ndarray, yaw: float,
) -> InteractionArtifact:
    transformed = copy.deepcopy(artifact)
    rotation = _yaw_quaternion(yaw)
    transformed.positions[:, 0] = (
        holden_quat.mul_vec(rotation, transformed.positions[:, 0]) + translation
    )
    transformed.velocities[:, 0] = holden_quat.mul_vec(
        rotation, transformed.velocities[:, 0]
    )
    transformed.angular_velocities[:, 0] = holden_quat.mul_vec(
        rotation, transformed.angular_velocities[:, 0]
    )
    transformed.rotations[:, 0] = holden_quat.mul(rotation, transformed.rotations[:, 0])
    transformed.object_positions = (
        holden_quat.mul_vec(rotation, transformed.object_positions) + translation
    )
    transformed.object_rotations = holden_quat.mul(rotation, transformed.object_rotations)
    return transformed


def _globally_transform_walking(
    clip: HoldenClip, *, translation: np.ndarray, yaw: float,
) -> HoldenClip:
    transformed = copy.deepcopy(clip)
    rotation = _yaw_quaternion(yaw)
    transformed.positions[:, 0] = (
        holden_quat.mul_vec(rotation, transformed.positions[:, 0]) + translation
    )
    transformed.rotations[:, 0] = holden_quat.mul(rotation, transformed.rotations[:, 0])
    return transformed


class OverlapInteractionExtractionTests(unittest.TestCase):
    def test_interaction_pair_uses_real_contiguous_overlap(self):
        artifact = interaction_artifact_fixture()
        reach = 60

        rows = extract_interaction_pairs(artifact)

        self.assertEqual(rows.walk_windows.shape, (1, 50, 195))
        self.assertEqual(rows.pickup_windows.shape, (1, 50, 195))
        np.testing.assert_array_equal(
            rows.walk_windows[:, 30:50], rows.pickup_windows[:, 0:20]
        )
        np.testing.assert_array_equal(rows.source_walk_ranges[0], [reach - 50, reach])
        np.testing.assert_array_equal(rows.source_pickup_ranges[0], [reach - 20, reach + 30])
        np.testing.assert_array_equal(
            rows.source_continuation_ranges[0], [reach + 30, reach + 36]
        )
        np.testing.assert_array_equal(rows.continuation_sequence_indices, [0])
        self.assertEqual(rows.static_conditions.shape, (1, 25))
        self.assertEqual(rows.walk_temporal.shape, (1, 50, 9))
        self.assertEqual(rows.pickup_temporal.shape, (1, 50, 9))
        self.assertEqual(rows.rejection_counts, {})

    def test_interaction_motion_and_route_ignore_global_object_transform(self):
        artifact = interaction_artifact_fixture()
        artifact.object_rotations[:] = _yaw_quaternion(0.35)
        artifact.rotations[:, 0] = _yaw_quaternion(-0.20)
        transformed = _globally_transform_interaction(
            artifact, translation=np.asarray([7.0, 0.5, -3.0], np.float32), yaw=1.10
        )

        baseline = extract_interaction_pairs(artifact)
        equivalent = extract_interaction_pairs(transformed)

        np.testing.assert_allclose(baseline.walk_windows, equivalent.walk_windows, atol=2e-6)
        np.testing.assert_allclose(baseline.pickup_windows, equivalent.pickup_windows, atol=2e-6)
        np.testing.assert_allclose(baseline.walk_temporal, equivalent.walk_temporal, atol=2e-6)
        np.testing.assert_allclose(baseline.pickup_temporal, equivalent.pickup_temporal, atol=2e-6)
        self.assertEqual(
            baseline.walk_windows[:, 30:50].tobytes(),
            baseline.pickup_windows[:, :20].tobytes(),
        )
        self.assertEqual(
            equivalent.walk_windows[:, 30:50].tobytes(),
            equivalent.pickup_windows[:, :20].tobytes(),
        )

    def test_grasp_pose_uses_the_same_yaw_canonical_frame_as_motion(self):
        artifact = interaction_artifact_fixture()
        tilt = holden_quat.from_angle_axis(
            np.asarray(0.7, np.float32), np.asarray([1.0, 0.0, 0.0], np.float32)
        ).astype(np.float32)
        yaw = _yaw_quaternion(0.4)
        object_rotation = holden_quat.mul(yaw, tilt)
        artifact.object_rotations[:] = object_rotation
        artifact.grasp_positions_object[0] = np.asarray([0.01, 0.02, 0.03], np.float32)
        artifact.grasp_rotations_object[0] = holden_quat.from_angle_axis(
            np.asarray(-0.2, np.float32), np.asarray([0.0, 0.0, 1.0], np.float32)
        ).astype(np.float32)

        rows = extract_interaction_pairs(artifact)

        w, x, y, z = object_rotation
        canonical_yaw = _yaw_quaternion(
            float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))
        )
        residual = holden_quat.mul(holden_quat.inv(canonical_yaw), object_rotation)
        expected_position = holden_quat.mul_vec(
            residual, artifact.grasp_positions_object[0]
        )
        expected_rotation = holden_quat.mul(
            residual, artifact.grasp_rotations_object[0]
        )
        expected_matrix = holden_quat.to_xform(expected_rotation)
        expected_rotation6d = np.concatenate(
            (expected_matrix[:, 0], expected_matrix[:, 1]), axis=0
        )
        expected_approach = holden_quat.mul_vec(
            residual, artifact.approach_directions_object[0]
        )
        expected_approach_xz = expected_approach[[0, 2]] / np.linalg.norm(
            expected_approach[[0, 2]]
        )
        np.testing.assert_allclose(rows.static_conditions[0, 2:5], expected_position, atol=1e-6)
        np.testing.assert_allclose(rows.static_conditions[0, 5:11], expected_rotation6d, atol=1e-6)
        np.testing.assert_allclose(rows.static_conditions[0, 11:13], expected_approach_xz, atol=1e-6)

    def test_source_ranges_are_local_to_their_continuation_identity(self):
        first = interaction_artifact_fixture()
        second = copy.deepcopy(first)
        frame_fields = (
            "positions", "velocities", "rotations", "angular_velocities",
            "foot_contacts", "hand_contacts", "hand_dof", "hand_dof_velocities",
            "phases", "time_to_contact", "object_positions", "object_rotations",
            "object_velocities", "object_angular_velocities", "source_frames",
        )
        clip_fields = (
            "active_hands", "table_positions", "table_rotations", "table_sizes",
            "object_dimensions", "grasp_positions_object", "grasp_rotations_object",
            "approach_directions_object",
        )
        for name in frame_fields:
            setattr(second, name, np.concatenate((getattr(first, name), getattr(first, name))))
        for name in clip_fields:
            setattr(second, name, np.concatenate((getattr(first, name), getattr(first, name))))
        second.range_starts = np.array([0, 120], np.int32)
        second.range_stops = np.array([120, 240], np.int32)

        rows = extract_interaction_pairs(second)

        np.testing.assert_array_equal(rows.sequence_indices, [0, 1])
        np.testing.assert_array_equal(rows.source_walk_ranges[1], [10, 60])
        np.testing.assert_array_equal(rows.source_continuation_ranges[1], [90, 96])

    def test_lift_after_r_plus_29_is_accepted_when_continuation_reaches_it(self):
        rows = extract_interaction_pairs(interaction_artifact_fixture())

        self.assertEqual(len(rows.walk_windows), 1)
        self.assertGreater(rows.source_continuation_ranges[0, 1], 90)

    def test_rejection_reasons_cover_invalid_source_rows(self):
        cases = (
            ("missing_reach", lambda a: a.phases.__setitem__(slice(None), 0)),
            ("left_hand", lambda a: a.active_hands.__setitem__(0, int(InteractionHand.LEFT))),
            (
                "invalid_grasp",
                lambda a: a.approach_directions_object.__setitem__(0, 0.0),
            ),
        )
        for expected, mutate in cases:
            with self.subTest(expected=expected):
                artifact = interaction_artifact_fixture()
                mutate(artifact)
                rows = extract_interaction_pairs(artifact)
                self.assertEqual(len(rows.walk_windows), 0)
                self.assertEqual(rows.rejection_counts, {expected: 1})

        late_contact = extract_interaction_pairs(
            interaction_artifact_fixture(contact_offset=30, lift_offset=35)
        )
        self.assertEqual(
            late_contact.rejection_counts, {"contact_after_reach_window": 1}
        )
        in_window_lift = extract_interaction_pairs(
            interaction_artifact_fixture(contact_offset=20, lift_offset=25)
        )
        self.assertEqual(
            in_window_lift.rejection_counts, {"missing_continuation": 1}
        )
        short = extract_interaction_pairs(interaction_artifact_fixture(frames=89))
        self.assertEqual(short.rejection_counts, {"short_clip": 1})


class OverlapWalkingExtractionTests(unittest.TestCase):
    def test_walking_windows_are_complete_encoded_source_ranges(self):
        clip = HoldenClip.empty(frames=71, bones=31)
        clip.name = "native-g1"
        clip.positions[:, 0, 0] = np.arange(71, dtype=np.float32)
        rows = extract_walking_windows(clip, stride=10)

        self.assertEqual(rows.shape, (3, 50, 195))
        np.testing.assert_array_equal(
            rows[:, :, 0], np.broadcast_to(np.arange(-49, 1, dtype=np.float32), (3, 50))
        )

    def test_balancing_keeps_every_native_source_window(self):
        clip = HoldenClip.empty(frames=101, bones=31)
        clip.positions[:50, 0, 0] = np.arange(50, dtype=np.float32) * 0.01
        clip.positions[50:, 0, 0] = 0.5 + np.arange(51, dtype=np.float32) * 0.10
        starts = np.arange(0, 52, 10, dtype=np.int32)

        selected = build_g1_overlap_dataset._coverage_balanced_walking_indices(
            clip, stride=10, seed=8
        )

        np.testing.assert_array_equal(starts[np.unique(selected)], starts)
        self.assertGreaterEqual(len(selected), len(starts))

    def test_walking_goal_frame_rotates_route_and_motion(self):
        clip = HoldenClip.empty(frames=500, bones=31)
        clip.positions[:, 0, 0] = np.arange(500, dtype=np.float32) * 0.05
        clip.rotations[:, 0] = _yaw_quaternion(np.pi / 2.0)
        transformed = _globally_transform_walking(
            clip, translation=np.asarray([-5.0, 0.0, 8.0], np.float32), yaw=-0.70
        )

        window, _, route = build_g1_overlap_dataset._walking_row(clip, 0)
        equivalent_window, _, equivalent_route = build_g1_overlap_dataset._walking_row(
            transformed, 0
        )
        terminal = clip.rotations[49, 0]
        expected_route = holden_quat.inv_mul_vec(
            terminal, clip.positions[0, 0] - clip.positions[49, 0]
        )[[0, 2]]

        np.testing.assert_allclose(route[0, :2], expected_route, atol=2e-6)
        np.testing.assert_allclose(window, equivalent_window, atol=2e-6)
        np.testing.assert_allclose(route, equivalent_route, atol=2e-6)

    def test_walking_partitions_have_no_shared_source_frame_or_range(self):
        clip = HoldenClip.empty(frames=500, bones=31)
        clip.positions[:, 0, 0] = np.arange(500, dtype=np.float32) * 0.03
        partitions = build_g1_overlap_dataset._partitioned_walking_rows(
            clip, stride=10, seed=14
        )

        source_frames = {
            name: {
                frame
                for start, stop in rows[3]
                for frame in range(int(start), int(stop))
            }
            for name, rows in partitions.items()
        }
        source_ranges = {
            name: {tuple(int(value) for value in item) for item in rows[3]}
            for name, rows in partitions.items()
        }
        self.assertGreater(len(partitions["train"][0]), 8)
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertTrue(source_frames[left].isdisjoint(source_frames[right]))
            self.assertTrue(source_ranges[left].isdisjoint(source_ranges[right]))


class OverlapDatasetCliTests(unittest.TestCase):
    def test_canonical_offsets_require_the_frozen_native_digest(self):
        walking = HoldenClip.empty(frames=3, bones=31)
        walking.positions[:, 2, 1] = np.asarray([-0.1000, -0.0990, -0.1000], np.float32)

        with self.assertRaisesRegex(ValueError, "SHA256"):
            build_g1_overlap_dataset._canonical_skeleton_metadata(walking)

    def test_interaction_split_is_object_disjoint_and_complete(self):
        object_ids = np.asarray(["cup", "cup", "bottle", "bottle", "plate", "vase"])

        split = build_g1_overlap_dataset._split_interaction_by_object(object_ids, seed=9)

        partition_objects = {
            name: set(object_ids[indices].tolist()) for name, indices in split.items()
        }
        self.assertTrue(partition_objects["train"].isdisjoint(partition_objects["validation"]))
        self.assertTrue(partition_objects["train"].isdisjoint(partition_objects["test"]))
        self.assertTrue(partition_objects["validation"].isdisjoint(partition_objects["test"]))
        self.assertEqual(
            np.sort(np.concatenate(tuple(split.values()))).tolist(), list(range(len(object_ids)))
        )

    def test_script_entrypoint_imports_from_repository_root(self):
        repository = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "tools/build_g1_overlap_dataset.py", "--help"],
            cwd=repository, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--walking-source", result.stdout)

    def test_main_reports_a_compact_audit_summary(self):
        audit = {
            "interaction_rows": 3,
            "walking_rows": 2,
            "overlap_byte_identical": True,
            "continuations": [{"sequence_id": "large-record"}],
        }
        output = io.StringIO()
        with patch.object(build_g1_overlap_dataset, "build_dataset", return_value=audit), patch.object(
            sys, "argv", ["build_g1_overlap_dataset.py", "--walking-source", "a", "--g1-xml", "b", "--interaction-pack", "c", "--output", "d", "--seed", "1"]
        ), redirect_stdout(output):
            self.assertEqual(build_g1_overlap_dataset.main(), 0)
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["interaction_rows"], 3)
        self.assertNotIn("continuations", summary)

    def test_export_uses_manifest_object_identity_and_train_only_normalization(self):
        artifact = interaction_artifact_fixture()
        walking = _canonical_walking()
        manifest = {
            "clips": [{
                "sequence_id": "pickup_table__cup_2__001",
                "object_id": "cup_2",
                "range_start": 0,
                "range_stop": 120,
            }]
        }
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, xml = root / "native.npz", root / "g1.xml"
            pack, output = root / "pack", root / "dataset.npz"
            source.write_bytes(b"native")
            xml.write_text("<mujoco/>", encoding="utf-8")
            pack.mkdir()
            for name in (
                "interaction_database.bin", "interaction_features.bin", "manifest.json",
            ):
                (pack / name).write_bytes(name.encode("ascii"))
            with patch.object(build_g1_overlap_dataset, "load_native_g1_walk", return_value=walking), patch.object(
                build_g1_overlap_dataset, "read_artifact_set", return_value=(artifact, None, manifest, None, None)
            ):
                audit = build_g1_overlap_dataset.build_dataset(
                    source, xml, pack, output, seed=2026072101
                )
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".manifest.json").is_file())
            self.assertEqual(audit["interaction_rows"], 1)
            self.assertTrue(audit["overlap_byte_identical"])
            with np.load(output, allow_pickle=False) as dataset:
                np.testing.assert_array_equal(dataset["train_object_ids"], ["cup_2"])
                self.assertIn("normalization_mean", dataset)
                self.assertEqual(dataset["normalization_mean"].shape, (195,))
                self.assertGreater(float(dataset["normalization_scale"][3]), 0.0)
                np.testing.assert_array_equal(
                    dataset["skeleton_parents"], G1_SKELETON.parents,
                )
                np.testing.assert_array_equal(
                    dataset["skeleton_names"], G1_SKELETON.names,
                )
                self.assertEqual(dataset["skeleton_signature"].item(), G1_SKELETON.signature())
                self.assertEqual(dataset["canonical_local_offsets"].shape, (31, 3))
                np.testing.assert_array_equal(dataset["canonical_local_offsets"][:2], 0.0)
            self.assertEqual(audit["frame_schema"], 195)
            self.assertEqual(audit["skeleton_signature"], G1_SKELETON.signature())

    def test_failed_manifest_publication_restores_the_previous_output_pair(self):
        artifact = interaction_artifact_fixture()
        walking = _canonical_walking()
        manifest = {"clips": [{
            "sequence_id": "pickup_table__cup_2__001",
            "object_id": "cup_2",
            "range_start": 0,
            "range_stop": 120,
        }]}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, xml = root / "native.npz", root / "g1.xml"
            pack, output = root / "pack", root / "dataset.npz"
            source.write_bytes(b"native")
            xml.write_text("<mujoco/>", encoding="utf-8")
            pack.mkdir()
            for name in ("interaction_database.bin", "interaction_features.bin", "manifest.json"):
                (pack / name).write_bytes(name.encode("ascii"))
            output.write_bytes(b"old dataset")
            output.with_suffix(".manifest.json").write_bytes(b'{"old": true}\n')
            previous_dataset = output.read_bytes()
            previous_manifest = output.with_suffix(".manifest.json").read_bytes()
            real_replace = build_g1_overlap_dataset.os.replace
            failed = False

            def fail_manifest_replace(source_path, destination_path):
                nonlocal failed
                if not failed and Path(destination_path) == output.with_suffix(".manifest.json"):
                    failed = True
                    raise OSError("injected manifest publish failure")
                return real_replace(source_path, destination_path)

            with patch.object(build_g1_overlap_dataset, "load_native_g1_walk", return_value=walking), patch.object(
                build_g1_overlap_dataset, "read_artifact_set", return_value=(artifact, None, manifest, None, None)
            ), patch.object(build_g1_overlap_dataset.os, "replace", side_effect=fail_manifest_replace):
                with self.assertRaisesRegex(OSError, "injected manifest"):
                    build_g1_overlap_dataset.build_dataset(source, xml, pack, output, seed=2026072101)

            self.assertEqual(output.read_bytes(), previous_dataset)
            self.assertEqual(output.with_suffix(".manifest.json").read_bytes(), previous_manifest)


if __name__ == "__main__":
    unittest.main()
