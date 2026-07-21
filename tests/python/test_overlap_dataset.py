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


def interaction_artifact_fixture(
    frames: int = 120, *, contact_offset: int = 29, lift_offset: int = 35,
) -> InteractionArtifact:
    """One right-hand clip whose Contact is at R+29 and Lift is post-window."""
    reach = 60
    positions = np.zeros((frames, 31, 3), np.float32)
    positions[:, 0, 0] = np.arange(frames, dtype=np.float32) * 0.01
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


class OverlapInteractionExtractionTests(unittest.TestCase):
    def test_interaction_pair_uses_real_contiguous_overlap(self):
        artifact = interaction_artifact_fixture()
        reach = 60

        rows = extract_interaction_pairs(artifact)

        self.assertEqual(rows.walk_windows.shape, (1, 50, 192))
        self.assertEqual(rows.pickup_windows.shape, (1, 50, 192))
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

        self.assertEqual(rows.shape, (3, 50, 192))
        np.testing.assert_array_equal(
            rows[:, :, 0], np.broadcast_to(np.arange(50, dtype=np.float32), (3, 50))
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


class OverlapDatasetCliTests(unittest.TestCase):
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
        walking = HoldenClip.empty(frames=71, bones=31)
        walking.positions[:, 0, 0] = np.arange(71, dtype=np.float32)
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


if __name__ == "__main__":
    unittest.main()
