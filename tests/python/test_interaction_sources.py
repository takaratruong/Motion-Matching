from dataclasses import fields
from itertools import combinations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from resources.g1_interaction_builder.object_geometry import (
    build_dimension_catalog,
)
from resources.g1_interaction_builder.schema import (
    CanonicalInteractionClip,
    ConversionValidationError,
    EvaluationSplit,
    FeatureGroup,
    FeatureSet,
    G1_SKELETON,
    InteractionArtifact,
    InteractionBuildError,
    InteractionHand,
    InteractionPhase,
    InteractionValidationError,
    LabeledInteractionClip,
    PhaseConfig,
    RawInteractionClip,
    SourcePaths,
    SourceValidationError,
)
from resources.g1_interaction_builder.sources import (
    discover_source_paths,
    discover_source_paths_many,
    load_raw_interaction,
    object_id_from_sequence,
    sequence_parts,
)
from tests.python.interaction_fixture import write_source_fixture


def canonical_clip_fixture(frames: int = 3) -> CanonicalInteractionClip:
    local_rotations = np.tile(
        np.array([1, 0, 0, 0], np.float32), (frames, 31, 1)
    )
    hand_rotations = np.tile(
        np.array([1, 0, 0, 0], np.float32), (frames, 2, 1)
    )
    object_rotations = np.tile(
        np.array([1, 0, 0, 0], np.float32), (frames, 1)
    )
    return CanonicalInteractionClip(
        sequence_id="pickup_table__cup_2__001",
        object_id="cup_2",
        fps=25.0,
        positions=np.zeros((frames, 31, 3), np.float32),
        velocities=np.zeros((frames, 31, 3), np.float32),
        rotations=local_rotations,
        angular_velocities=np.zeros((frames, 31, 3), np.float32),
        foot_contacts=np.zeros((frames, 2), np.uint8),
        hand_contacts=np.zeros((frames, 2), np.uint8),
        hand_dof=np.zeros((frames, 14), np.float32),
        hand_dof_velocities=np.zeros((frames, 14), np.float32),
        hand_positions=np.zeros((frames, 2, 3), np.float32),
        hand_rotations=hand_rotations,
        object_positions=np.zeros((frames, 3), np.float32),
        object_rotations=object_rotations,
        object_velocities=np.zeros((frames, 3), np.float32),
        object_angular_velocities=np.zeros((frames, 3), np.float32),
        table_position=np.zeros(3, np.float32),
        table_rotation=np.array([1, 0, 0, 0], np.float32),
        table_size=np.ones(3, np.float32),
        object_dimensions=np.ones(3, np.float32),
        source_frames=np.arange(frames, dtype=np.int32),
    )


class InteractionSourceTests(unittest.TestCase):
    def test_frozen_schema_names_values_and_skeleton_signature(self):
        self.assertEqual([member.value for member in InteractionHand], [0, 1])
        self.assertEqual(
            [member.value for member in InteractionPhase], [0, 1, 2, 3, 4]
        )
        self.assertEqual(
            G1_SKELETON.signature(),
            "6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7",
        )
        expected_fields = {
            SourcePaths: (
                "sequence_id",
                "object_id",
                "robot",
                "objects",
                "meta",
                "object_usd",
            ),
            RawInteractionClip: (
                "sequence_id",
                "object_id",
                "fps",
                "qpos",
                "hand_dof",
                "object_positions",
                "object_rotations",
                "hand_contacts",
                "table_position",
                "table_rotation",
                "table_size",
                "object_dimensions",
                "source_frames",
            ),
            CanonicalInteractionClip: (
                "sequence_id",
                "object_id",
                "fps",
                "positions",
                "velocities",
                "rotations",
                "angular_velocities",
                "foot_contacts",
                "hand_contacts",
                "hand_dof",
                "hand_dof_velocities",
                "hand_positions",
                "hand_rotations",
                "object_positions",
                "object_rotations",
                "object_velocities",
                "object_angular_velocities",
                "table_position",
                "table_rotation",
                "table_size",
                "object_dimensions",
                "source_frames",
            ),
            LabeledInteractionClip: (
                "motion",
                "active_hand",
                "phases",
                "time_to_contact",
                "contact_frame",
                "lift_frame",
                "hold_frame",
                "support_height",
                "grasp_position_object",
                "grasp_rotation_object",
                "approach_direction_object",
            ),
            FeatureGroup: ("name", "start", "stop"),
            FeatureSet: ("values", "offsets", "scales", "groups"),
            EvaluationSplit: ("seed", "database_objects", "heldout_objects"),
            InteractionArtifact: (
                "fps",
                "parents",
                "range_starts",
                "range_stops",
                "positions",
                "velocities",
                "rotations",
                "angular_velocities",
                "foot_contacts",
                "hand_contacts",
                "hand_dof",
                "hand_dof_velocities",
                "phases",
                "active_hands",
                "time_to_contact",
                "object_positions",
                "object_rotations",
                "object_velocities",
                "object_angular_velocities",
                "table_positions",
                "table_rotations",
                "table_sizes",
                "object_dimensions",
                "grasp_positions_object",
                "grasp_rotations_object",
                "approach_directions_object",
                "source_frames",
            ),
        }
        for cls, names in expected_fields.items():
            with self.subTest(cls=cls.__name__):
                self.assertEqual(tuple(field.name for field in fields(cls)), names)
        self.assertEqual(PhaseConfig(), PhaseConfig())
        for cls in (
            SourceValidationError,
            ConversionValidationError,
            InteractionValidationError,
        ):
            with self.subTest(cls=cls.__name__):
                error = cls("code", "detail")
                self.assertIsInstance(error, InteractionBuildError)
                self.assertEqual(error.code, "code")
                self.assertEqual(str(error), "code: detail")

    def test_sequence_parser_accepts_exact_table_and_ground_ids(self):
        self.assertEqual(
            sequence_parts("pickup_table__cup_2__001"),
            ("pickup_table", "cup_2"),
        )
        self.assertEqual(
            sequence_parts("pickup_ground__cup_2__001"),
            ("pickup_ground", "cup_2"),
        )
        for invalid in (
            "pickup_ground__cup_2__01",
            "pickup_ground__cup_2__0001",
            "putdown_ground__cup_2__001",
        ):
            with self.subTest(sequence_id=invalid), self.assertRaisesRegex(
                ValueError, "invalid pickup sequence id"
            ):
                sequence_parts(invalid)
        self.assertEqual(
            object_id_from_sequence("pickup_ground__cup_2__001"), "cup_2"
        )

    def test_discovery_requires_complete_sorted_triplets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = write_source_fixture(root)
            found = discover_source_paths(root)
            self.assertEqual(found, [expected])
            expected.meta.unlink()
            with self.assertRaisesRegex(ValueError, "missing meta"):
                discover_source_paths(root)

    def test_discovery_sorts_and_reports_every_missing_modality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            later = write_source_fixture(
                root, sequence_id="pickup_table__cup_2__002"
            )
            earlier = write_source_fixture(
                root, sequence_id="pickup_table__cup_2__001"
            )
            self.assertEqual(
                [source.sequence_id for source in discover_source_paths(root)],
                [earlier.sequence_id, later.sequence_id],
            )
            earlier.meta.unlink()
            later.object_usd.unlink()
            with self.assertRaises(ValueError) as caught:
                discover_source_paths(root)
            message = str(caught.exception)
            self.assertIn(f"{earlier.sequence_id}: missing meta", message)
            self.assertIn(f"{later.sequence_id}: missing object_usd", message)

    def test_discovers_multiple_roots_in_sequence_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            table = write_source_fixture(
                base / "table", sequence_id="pickup_table__cup_2__001"
            )
            ground = write_source_fixture(
                base / "ground",
                sequence_id="pickup_ground__cup_2__001",
                ground=True,
            )
            found = discover_source_paths_many([base / "table", base / "ground"])
            self.assertEqual(
                [item.sequence_id for item in found],
                [ground.sequence_id, table.sequence_id],
            )

    def test_discovery_rejects_duplicate_sequences_across_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sequence_id = "pickup_ground__cup_2__001"
            write_source_fixture(base / "first", sequence_id, ground=True)
            write_source_fixture(base / "second", sequence_id, ground=True)
            with self.assertRaisesRegex(
                ValueError, "duplicate sequence ids across roots"
            ):
                discover_source_paths_many([base / "first", base / "second"])

    def test_loads_native_g1_hands_contacts_and_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_source_fixture(Path(tmp))
            clip = load_raw_interaction(
                paths, np.array([0.08, 0.12, 0.20], np.float32)
            )
            self.assertEqual(clip.qpos.shape, (25, 36))
            self.assertEqual(clip.hand_dof.shape, (25, 14))
            np.testing.assert_array_equal(clip.hand_contacts[:10], 0)
            np.testing.assert_array_equal(clip.hand_contacts[10:, 0], 1)
            np.testing.assert_allclose(clip.qpos[:, 3], 1.0)
            np.testing.assert_allclose(clip.table_size, [1.2, 0.8, 0.05])
            np.testing.assert_allclose(
                clip.object_dimensions, [0.08, 0.12, 0.20]
            )

    def test_nonempty_contact_points_not_dictionary_membership_define_contact(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_source_fixture(Path(tmp))
            clip = load_raw_interaction(paths, np.ones(3, np.float32))
            self.assertEqual(int(clip.hand_contacts[:10].sum()), 0)

    def test_loads_the_native_direct_meta_record_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_source_fixture(Path(tmp))
            wrapped = joblib.load(paths.meta)
            joblib.dump(wrapped[paths.sequence_id], paths.meta)
            clip = load_raw_interaction(paths, np.ones(3, np.float32))
            np.testing.assert_allclose(clip.table_position, [0.0, 0.0, 0.75])

    def test_ground_meta_synthesizes_exact_virtual_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_source_fixture(Path(tmp), ground=True)
            clip = load_raw_interaction(paths, np.ones(3, np.float32))
            np.testing.assert_array_equal(
                clip.table_position, np.array([0.0, 0.0, -0.02], np.float32)
            )
            np.testing.assert_array_equal(
                clip.table_size, np.array([20.0, 20.0, 0.04], np.float32)
            )
            np.testing.assert_array_equal(
                clip.table_rotation, np.array([1.0, 0.0, 0.0, 0.0], np.float32)
            )

    def test_loads_the_native_direct_ground_meta_record_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_source_fixture(Path(tmp), ground=True)
            wrapped = joblib.load(paths.meta)
            joblib.dump(wrapped[paths.sequence_id], paths.meta)
            clip = load_raw_interaction(paths, np.ones(3, np.float32))
            np.testing.assert_array_equal(
                clip.table_position, np.array([0.0, 0.0, -0.02], np.float32)
            )

    def test_ground_meta_rejects_any_tabletop_keys(self):
        tabletop_fields = {
            "table_pos": np.array([0.0, 0.0, 0.75], np.float32),
            "table_quat": np.array([0, 0, 0, 1], np.float32),
            "table_size": np.array([1.2, 0.8, 0.05], np.float32),
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_source_fixture(Path(tmp), ground=True)
            for count in range(1, len(tabletop_fields) + 1):
                for fields in combinations(tabletop_fields, count):
                    with self.subTest(fields=fields):
                        record = {"object_name": paths.object_id}
                        record.update(
                            {field: tabletop_fields[field] for field in fields}
                        )
                        joblib.dump(record, paths.meta)
                        with self.assertRaises(SourceValidationError) as caught:
                            load_raw_interaction(paths, np.ones(3, np.float32))
                        self.assertEqual(caught.exception.code, "invalid_source_record")

    def test_rejects_partial_native_meta_with_a_path_specific_source_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_source_fixture(Path(tmp))
            wrapped = joblib.load(paths.meta)
            partial = dict(wrapped[paths.sequence_id])
            partial.pop("table_size")
            joblib.dump(partial, paths.meta)
            with self.assertRaises(SourceValidationError) as caught:
                load_raw_interaction(paths, np.ones(3, np.float32))
            self.assertIn(str(paths.meta), str(caught.exception))

    def test_rejects_mismatched_source_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_source_fixture(Path(tmp))
            wrapped = joblib.load(paths.objects)
            wrapped[paths.sequence_id]["fps"] = 30.0
            joblib.dump(wrapped, paths.objects)
            with self.assertRaisesRegex(SourceValidationError, "fps"):
                load_raw_interaction(paths, np.ones(3, np.float32))

    def test_raw_validation_rejects_invalid_shapes_values_and_ordering(self):
        mutations = (
            ("qpos", np.zeros((25, 35), np.float32), "qpos"),
            ("fps", 0.0, "fps"),
            (
                "object_rotations",
                np.zeros((25, 4), np.float32),
                "quaternion",
            ),
            (
                "hand_contacts",
                np.zeros((25, 2), np.float32),
                "contact",
            ),
            (
                "source_frames",
                np.arange(25, dtype=np.int32)[::-1],
                "source frame",
            ),
            ("table_size", np.array([1.0, 0.0, 1.0]), "dimension"),
        )
        for attribute, value, message in mutations:
            with self.subTest(attribute=attribute), tempfile.TemporaryDirectory() as tmp:
                paths = write_source_fixture(Path(tmp))
                clip = load_raw_interaction(paths, np.ones(3, np.float32))
                setattr(clip, attribute, value)
                with self.assertRaisesRegex(
                    SourceValidationError, f"{paths.sequence_id}.*{message}"
                ):
                    clip.validate()

    def test_canonical_validation_enforces_the_frozen_array_contract(self):
        clip = canonical_clip_fixture()
        clip.validate()
        mutations = (
            ("positions", np.zeros((3, 30, 3), np.float32), "position"),
            (
                "hand_rotations",
                np.zeros((3, 2, 4), np.float32),
                "quaternion",
            ),
            ("foot_contacts", np.full((3, 2), 2, np.uint8), "contact"),
            ("object_dimensions", np.array([1.0, -1.0, 1.0]), "dimension"),
        )
        for attribute, value, message in mutations:
            with self.subTest(attribute=attribute):
                clip = canonical_clip_fixture()
                setattr(clip, attribute, value)
                with self.assertRaisesRegex(
                    ConversionValidationError,
                    f"{clip.sequence_id}.*{message}",
                ):
                    clip.validate()

    def test_dimension_catalog_deduplicates_consistent_object_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [
                write_source_fixture(
                    root, sequence_id="pickup_table__cup_2__001"
                ),
                write_source_fixture(
                    root, sequence_id="pickup_table__cup_2__002"
                ),
            ]
            expected = np.array([0.08, 0.12, 0.20], np.float32)
            with patch(
                "resources.g1_interaction_builder.object_geometry.read_usd_dimensions",
                return_value=expected.copy(),
            ):
                catalog = build_dimension_catalog(sources)
            self.assertEqual(tuple(catalog), ("cup_2",))
            np.testing.assert_array_equal(catalog["cup_2"], expected)

    def test_dimension_catalog_rejects_inconsistent_object_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [
                write_source_fixture(
                    root, sequence_id="pickup_table__cup_2__001"
                ),
                write_source_fixture(
                    root, sequence_id="pickup_table__cup_2__002"
                ),
            ]
            with patch(
                "resources.g1_interaction_builder.object_geometry.read_usd_dimensions",
                side_effect=(
                    np.array([0.08, 0.12, 0.20], np.float32),
                    np.array([0.08, 0.12, 0.22], np.float32),
                ),
            ), self.assertRaisesRegex(ValueError, "inconsistent USD bounds"):
                build_dimension_catalog(sources)


if __name__ == "__main__":
    unittest.main()
