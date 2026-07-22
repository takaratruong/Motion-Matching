import contextlib
import copy
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from resources import build_g1_interaction_database as build_cli
from resources import fetch_grail_pickup_table as fetch_cli
from resources import validate_g1_interaction_database as validate_cli
from resources.g1_interaction_builder import build as build_module
from resources.g1_interaction_builder.artifacts import (
    read_artifact_set,
    write_artifact_set,
)
from resources.g1_interaction_builder.schema import (
    ConversionValidationError,
    EvaluationSplit,
    G1_SKELETON,
    InteractionValidationError,
    PhaseConfig,
    SourceValidationError,
)
from tests.python.interaction_fixture import (
    canonical_pickup_fixture,
    labeled_clips_for_objects,
    write_source_fixture,
)


NUMERIC_REPORT = {
    "fk_max_error_m": 0.0002,
    "fk_rotation_max_error_degrees": 0.02,
    "duration_error_s": 0.0,
    "quaternion_norm_max_error": 0.00001,
}


KNOWN_REJECTION_CODES_BY_STAGE = {
    "source": {
        "fps_mismatch",
        "frame_count_mismatch",
        "invalid_contact",
        "invalid_dimensions",
        "invalid_fps",
        "invalid_quaternion",
        "invalid_shape",
        "invalid_source_frames",
        "invalid_source_record",
        "missing_field",
        "non_finite",
        "object_identity_mismatch",
    },
    "conversion": {
        "duration_error",
        "fk_error",
        "fk_rotation_error",
        "fps_mismatch",
        "frame_count_mismatch",
        "invalid_contact",
        "invalid_dimensions",
        "invalid_fps",
        "invalid_quaternion",
        "invalid_shape",
        "invalid_source_frames",
        "joint_limit_violation",
        "non_finite",
        "skeleton_mismatch",
    },
    "interaction": {
        "ambiguous_active_hand",
        "contact_lost_before_hold",
        "invalid_approach",
        "invalid_grasp",
        "no_five_centimeter_lift",
        "no_distinct_lift_phase",
        "no_stable_contact",
        "no_stable_hold",
    },
}
KNOWN_REJECTION_CODES = set().union(
    *KNOWN_REJECTION_CODES_BY_STAGE.values()
)


def artifact_snapshot(output: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir())
        if path.is_file()
    }


def rewrite_json(path: Path, update) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    update(value)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_sources(
    root: Path,
    identities: list[tuple[str, str]],
) -> list:
    return [
        write_source_fixture(root, sequence_id=sequence_id, object_id=object_id)
        for sequence_id, object_id in identities
    ]


def corrupt_object_frame_count(source) -> None:
    records = joblib.load(source.objects)
    record = records[source.sequence_id]
    record["root_pos"] = record["root_pos"][:-1]
    joblib.dump(records, source.objects)


def fake_convert(raw, kinematics, target_fps):
    del kinematics
    if target_fps != 25.0:
        raise AssertionError(f"unexpected target fps {target_fps}")
    motion = canonical_pickup_fixture()
    motion.sequence_id = raw.sequence_id
    motion.object_id = raw.object_id
    motion.object_dimensions = raw.object_dimensions.copy()
    return motion, G1_SKELETON, dict(NUMERIC_REPORT)


@contextlib.contextmanager
def fast_build_patches(*, converter=fake_convert):
    dimensions = np.array([0.08, 0.20, 0.12], np.float32)
    with patch.object(
        build_cli, "G1Kinematics", return_value=SimpleNamespace()
    ) as kinematics, patch.object(
        build_module,
        "read_usd_dimensions",
        return_value=dimensions.copy(),
    ) as dimension_reader, patch.object(
        build_module, "convert_interaction", side_effect=converter
    ) as conversion:
        yield kinematics, dimension_reader, conversion


def build_argv(
    source_root: Path,
    output: Path,
    *extra: str,
) -> list[str]:
    return [
        "--source-root",
        str(source_root),
        "--g1-xml",
        str(source_root / "g1.xml"),
        "--output",
        str(output),
        *extra,
    ]


def manifest_metadata_corruptions(artifact) -> tuple:
    active_hand = int(artifact.active_hands[0])

    def mutate_parent_value(manifest: dict) -> None:
        manifest["skeleton_parents"][0] = 0

    def mutate_parent_type(manifest: dict) -> None:
        manifest["skeleton_parents"][1] = float(
            manifest["skeleton_parents"][1]
        )

    def mutate_feature_name(manifest: dict) -> None:
        manifest["feature_names"][0] = "wrong_feature"

    def truncate_feature_names(manifest: dict) -> None:
        manifest["feature_names"].pop()

    def mutate_feature_group(manifest: dict) -> None:
        manifest["feature_groups"][0]["name"] = "wrong_group"

    def mutate_feature_group_type(manifest: dict) -> None:
        manifest["feature_groups"][0]["start"] = 0.0

    def mutate_phase_config(manifest: dict) -> None:
        manifest["phase_config"]["stable_source_samples"] = 4

    def mutate_split_seed(manifest: dict) -> None:
        manifest["split"]["seed"] += 1

    def mutate_database_count(manifest: dict) -> None:
        manifest["split"]["database_object_count"] += 1

    def mutate_heldout_count(manifest: dict) -> None:
        manifest["split"]["heldout_object_count"] += 1

    def mutate_dependency_keys(manifest: dict) -> None:
        manifest["dependency_versions"].pop(
            sorted(manifest["dependency_versions"])[0]
        )

    def mutate_dependency_value(manifest: dict) -> None:
        first = sorted(manifest["dependency_versions"])[0]
        manifest["dependency_versions"][first] = ""

    return (
        (
            "database magic",
            lambda value: value.__setitem__("database_magic", "BADDB"),
            r"manifest database_magic.*G1INTDB1",
        ),
        (
            "feature magic",
            lambda value: value.__setitem__("feature_magic", "BADFT"),
            r"manifest feature_magic.*G1INTFT1",
        ),
        (
            "skeleton names",
            lambda value: value["skeleton_names"].__setitem__(
                0, "WrongRoot"
            ),
            r"manifest skeleton_names.*exact G1 skeleton",
        ),
        (
            "skeleton parent value",
            mutate_parent_value,
            r"manifest skeleton_parents.*exact G1 skeleton",
        ),
        (
            "skeleton parent integer type",
            mutate_parent_type,
            r"manifest skeleton_parents.*integers",
        ),
        (
            "skeleton signature",
            lambda value: value.__setitem__(
                "skeleton_signature", "wrong"
            ),
            r"manifest skeleton_signature.*exact G1 skeleton",
        ),
        (
            "clip active hand integer type",
            lambda value: value["clips"][0].__setitem__(
                "active_hand", True
            ),
            r"manifest clip 0 active_hand.*integer 0 or 1",
        ),
        (
            "clip active hand binary mismatch",
            lambda value: value["clips"][0].__setitem__(
                "active_hand", 1 - active_hand
            ),
            r"manifest clip 0 active_hand.*database active_hands",
        ),
        (
            "empty source root",
            lambda value: value.__setitem__("source_root", ""),
            r"manifest source_root.*nonempty absolute path string",
        ),
        (
            "relative source root",
            lambda value: value.__setitem__(
                "source_root", "relative/source"
            ),
            r"manifest source_root.*nonempty absolute path string",
        ),
        (
            "dataset id",
            lambda value: value.__setitem__("dataset_id", "other/dataset"),
            r"manifest dataset_id.*nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL",
        ),
        (
            "phase config",
            mutate_phase_config,
            r"manifest phase_config.*exact schema-v1 PhaseConfig",
        ),
        (
            "feature name",
            mutate_feature_name,
            r"manifest feature_names.*exact 71 schema-v1 names",
        ),
        (
            "feature name count",
            truncate_feature_names,
            r"manifest feature_names.*exact 71 schema-v1 names",
        ),
        (
            "feature group",
            mutate_feature_group,
            r"manifest feature_groups.*exact schema-v1 groups",
        ),
        (
            "feature group integer type",
            mutate_feature_group_type,
            r"manifest feature_groups.*integer start and stop",
        ),
        (
            "split seed",
            mutate_split_seed,
            r"manifest split seed.*evaluation split seed",
        ),
        (
            "split database count",
            mutate_database_count,
            r"manifest split database_object_count.*evaluation split",
        ),
        (
            "split heldout count",
            mutate_heldout_count,
            r"manifest split heldout_object_count.*evaluation split",
        ),
        (
            "dependency key set",
            mutate_dependency_keys,
            r"manifest dependency_versions keys.*exactly",
        ),
        (
            "empty dependency version",
            mutate_dependency_value,
            r"manifest dependency_versions.*nonempty string",
        ),
        (
            "git commit syntax",
            lambda value: value.__setitem__("git_commit", "ABCDEF1"),
            r"manifest git_commit.*lowercase hexadecimal.*7.*64",
        ),
        (
            "diagnostic limit bool",
            lambda value: value.__setitem__("diagnostic_limit", True),
            r"manifest diagnostic_limit.*null or a positive integer",
        ),
        (
            "diagnostic limit zero",
            lambda value: value.__setitem__("diagnostic_limit", 0),
            r"manifest diagnostic_limit.*null or a positive integer",
        ),
        (
            "source date epoch bool",
            lambda value: value.__setitem__("source_date_epoch", False),
            r"manifest source_date_epoch.*null or an integer",
        ),
        (
            "source date epoch string",
            lambda value: value.__setitem__(
                "source_date_epoch", "1720950000"
            ),
            r"manifest source_date_epoch.*null or an integer",
        ),
    )


def publish_fast_valid_pack(
    test: unittest.TestCase,
    root: Path,
    output: Path,
) -> None:
    write_sources(
        root,
        [
            ("pickup_table__alpha__001", "alpha"),
            ("pickup_table__beta__001", "beta"),
        ],
    )
    with fast_build_patches(), contextlib.redirect_stdout(io.StringIO()):
        test.assertEqual(
            build_cli.main(
                build_argv(root, output, "--heldout-count", "1")
            ),
            0,
        )


class InteractionBuildUnitTests(unittest.TestCase):
    def test_object_dimensions_use_one_representative_per_category(self):
        sources = [
            SimpleNamespace(
                sequence_id="pickup_table__cup_2__002",
                object_id="cup_2",
                object_usd=Path("table-002.usd"),
            ),
            SimpleNamespace(
                sequence_id="pickup_ground__cup_2__001",
                object_id="cup_2",
                object_usd=Path("ground-001.usd"),
            ),
            SimpleNamespace(
                sequence_id="pickup_table__cup_2__001",
                object_id="cup_2",
                object_usd=Path("table-001.usd"),
            ),
            SimpleNamespace(
                sequence_id="pickup_ground__cup_2__002",
                object_id="cup_2",
                object_usd=Path("ground-002.usd"),
            ),
        ]
        dimensions = np.array([0.08, 0.20, 0.12], np.float32)
        with patch.object(
            build_module,
            "read_usd_dimensions",
            return_value=dimensions,
        ) as reader:
            result = build_module.build_object_dimensions(sources)

        np.testing.assert_array_equal(result["cup_2"], dimensions)
        self.assertEqual(
            [path.name for (path,), _ in reader.call_args_list],
            ["ground-001.usd", "table-001.usd"],
        )

    def test_object_dimensions_reject_inconsistent_cross_category_bounds(self):
        sources = [
            SimpleNamespace(
                sequence_id="pickup_table__cup_2__001",
                object_id="cup_2",
                object_usd=Path("table.usd"),
            ),
            SimpleNamespace(
                sequence_id="pickup_ground__cup_2__001",
                object_id="cup_2",
                object_usd=Path("ground.usd"),
            ),
        ]
        with patch.object(
            build_module,
            "read_usd_dimensions",
            side_effect=[
                np.array([0.08, 0.20, 0.12], np.float32),
                np.array([0.082, 0.20, 0.12], np.float32),
            ],
        ), self.assertRaisesRegex(
            ValueError, "inconsistent cross-category USD bounds"
        ):
            build_module.build_object_dimensions(sources)

    def test_rejection_code_schema_is_closed(self):
        self.assertEqual(
            set(build_module.all_schema_v1_rejection_codes()),
            KNOWN_REJECTION_CODES,
        )

    def test_rejection_stage_predicate_matches_the_frozen_table(self):
        for stage, reviewed in KNOWN_REJECTION_CODES_BY_STAGE.items():
            for code in KNOWN_REJECTION_CODES:
                with self.subTest(stage=stage, code=code):
                    self.assertEqual(
                        build_module.is_schema_v1_rejection(stage, code),
                        code in reviewed,
                    )
        self.assertFalse(
            build_module.is_schema_v1_rejection(
                "unknown_stage", "frame_count_mismatch"
            )
        )
        self.assertFalse(
            build_module.is_schema_v1_rejection(
                "source", "unknown_code"
            )
        )

    def test_every_rejection_code_is_emitted_at_a_reviewed_stage(self):
        source = SimpleNamespace(sequence_id="a", object_id="object")
        dimensions = {"object": np.ones(3, np.float32)}
        error_types = {
            "source": SourceValidationError,
            "conversion": ConversionValidationError,
            "interaction": InteractionValidationError,
        }
        for stage, codes in KNOWN_REJECTION_CODES_BY_STAGE.items():
            for code in sorted(codes):
                with self.subTest(
                    stage=stage, code=code
                ), contextlib.ExitStack() as stack:
                    error = error_types[stage](code, "reviewed exclusion")
                    if stage == "source":
                        stack.enter_context(
                            patch.object(
                                build_module,
                                "load_raw_interaction",
                                side_effect=error,
                            )
                        )
                    else:
                        stack.enter_context(
                            patch.object(
                                build_module,
                                "load_raw_interaction",
                                return_value=source,
                            )
                        )
                    if stage == "conversion":
                        stack.enter_context(
                            patch.object(
                                build_module,
                                "convert_interaction",
                                side_effect=error,
                            )
                        )
                    elif stage == "interaction":
                        stack.enter_context(
                            patch.object(
                                build_module,
                                "convert_interaction",
                                return_value=(source, G1_SKELETON, {}),
                            )
                        )
                        stack.enter_context(
                            patch.object(
                                build_module,
                                "derive_interaction_labels",
                                side_effect=error,
                            )
                        )

                    included, rejected, reports = (
                        build_module.build_labeled_clips(
                            [source], dimensions, SimpleNamespace()
                        )
                    )
                    self.assertEqual(included, [])
                    self.assertEqual(reports, [])
                    self.assertEqual(
                        [(item.stage, item.code) for item in rejected],
                        [(stage, code)],
                    )

    def test_known_code_at_an_unreviewed_stage_propagates(self):
        source = SimpleNamespace(sequence_id="a", object_id="object")
        dimensions = {"object": np.ones(3, np.float32)}
        error = SourceValidationError(
            "fk_error", "conversion code at source stage"
        )
        with patch.object(
            build_module, "load_raw_interaction", side_effect=error
        ), self.assertRaisesRegex(SourceValidationError, "fk_error"):
            build_module.build_labeled_clips(
                [source], dimensions, SimpleNamespace()
            )

    def test_unknown_typed_rejection_code_propagates(self):
        source = SimpleNamespace(sequence_id="a", object_id="object")
        dimensions = {"object": np.ones(3, np.float32)}
        error = SourceValidationError(
            "unreviewed_code", "must abort publication"
        )
        with patch.object(
            build_module, "load_raw_interaction", side_effect=error
        ), self.assertRaisesRegex(SourceValidationError, "unreviewed_code"):
            build_module.build_labeled_clips(
                [source], dimensions, SimpleNamespace()
            )

    def test_rejection_is_frozen_with_the_exact_field_order(self):
        self.assertTrue(
            build_module.Rejection.__dataclass_params__.frozen
        )
        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(build_module.Rejection)),
            ("sequence_id", "object_id", "stage", "code", "message"),
        )

    def test_per_source_orchestration_is_sorted_and_always_converts_at_25_hz(self):
        sources = [
            SimpleNamespace(sequence_id="z", object_id="z_object"),
            SimpleNamespace(sequence_id="a", object_id="a_object"),
        ]
        dimensions = {
            "a_object": np.ones(3, np.float32),
            "z_object": np.ones(3, np.float32),
        }
        events = []

        def load(source, dimension):
            events.append(("load", source.sequence_id))
            np.testing.assert_array_equal(
                dimension, dimensions[source.object_id]
            )
            return source

        def convert(raw, kinematics, target_fps):
            del kinematics
            events.append(("convert", raw.sequence_id, target_fps))
            return raw, G1_SKELETON, {"sequence": raw.sequence_id}

        def label(canonical):
            events.append(("label", canonical.sequence_id))
            return canonical

        with patch.object(
            build_module, "load_raw_interaction", side_effect=load
        ), patch.object(
            build_module, "convert_interaction", side_effect=convert
        ), patch.object(
            build_module, "derive_interaction_labels", side_effect=label
        ):
            included, rejected, reports = build_module.build_labeled_clips(
                sources, dimensions, SimpleNamespace()
            )

        self.assertEqual([item.sequence_id for item in included], ["a", "z"])
        self.assertEqual(rejected, [])
        self.assertEqual(reports, [{"sequence": "a"}, {"sequence": "z"}])
        self.assertEqual(
            events,
            [
                ("load", "a"),
                ("convert", "a", 25.0),
                ("label", "a"),
                ("load", "z"),
                ("convert", "z", 25.0),
                ("label", "z"),
            ],
        )

    def test_only_expected_typed_errors_become_sorted_rejections(self):
        sources = [
            SimpleNamespace(sequence_id="z", object_id="source_object"),
            SimpleNamespace(sequence_id="m", object_id="conversion_object"),
            SimpleNamespace(sequence_id="a", object_id="interaction_object"),
            SimpleNamespace(sequence_id="v", object_id="valid_object"),
        ]
        dimensions = {
            source.object_id: np.ones(3, np.float32) for source in sources
        }

        def load(source, dimension):
            del dimension
            if source.object_id == "source_object":
                raise SourceValidationError(
                    "frame_count_mismatch", "source frames disagree"
                )
            return source

        def convert(raw, kinematics, target_fps):
            del kinematics, target_fps
            if raw.object_id == "conversion_object":
                raise ConversionValidationError("fk_error", "bad FK")
            return raw, G1_SKELETON, {"ok": raw.sequence_id}

        def label(canonical):
            if canonical.object_id == "interaction_object":
                raise InteractionValidationError(
                    "no_stable_hold", "object never settles"
                )
            return canonical

        with patch.object(
            build_module, "load_raw_interaction", side_effect=load
        ), patch.object(
            build_module, "convert_interaction", side_effect=convert
        ), patch.object(
            build_module, "derive_interaction_labels", side_effect=label
        ):
            included, rejected, reports = build_module.build_labeled_clips(
                sources, dimensions, SimpleNamespace()
            )

        self.assertEqual([item.sequence_id for item in included], ["v"])
        self.assertEqual(reports, [{"ok": "v"}])
        self.assertEqual(
            [
                (item.sequence_id, item.object_id, item.stage, item.code)
                for item in rejected
            ],
            [
                ("a", "interaction_object", "interaction", "no_stable_hold"),
                ("m", "conversion_object", "conversion", "fk_error"),
                ("z", "source_object", "source", "frame_count_mismatch"),
            ],
        )
        self.assertEqual(
            rejected[0].message,
            "no_stable_hold: object never settles",
        )

    def test_unexpected_programming_and_base_exceptions_propagate(self):
        source = SimpleNamespace(sequence_id="a", object_id="object")
        dimensions = {"object": np.ones(3, np.float32)}
        for error in (
            RuntimeError("programming bug"),
            KeyboardInterrupt(),
            MemoryError("out of memory"),
        ):
            with self.subTest(error=type(error).__name__), patch.object(
                build_module, "load_raw_interaction", side_effect=error
            ), self.assertRaises(type(error)):
                build_module.build_labeled_clips(
                    [source], dimensions, SimpleNamespace()
                )

    def test_skeleton_mismatch_is_a_conversion_rejection(self):
        source = SimpleNamespace(sequence_id="a", object_id="object")
        dimensions = {"object": np.ones(3, np.float32)}
        wrong_skeleton = SimpleNamespace(signature=lambda: "wrong")
        with patch.object(
            build_module, "load_raw_interaction", return_value=source
        ), patch.object(
            build_module,
            "convert_interaction",
            return_value=(source, wrong_skeleton, {}),
        ):
            included, rejected, reports = build_module.build_labeled_clips(
                [source], dimensions, SimpleNamespace()
            )
        self.assertEqual(included, [])
        self.assertEqual(reports, [])
        self.assertEqual(rejected[0].stage, "conversion")
        self.assertEqual(rejected[0].code, "skeleton_mismatch")

    def test_prepare_database_uses_the_reviewed_single_artifact_boundary(self):
        clips = labeled_clips_for_objects(["database", "heldout"])
        split = EvaluationSplit(7, ("database",), ("heldout",))
        expected = ((clips[0],), object(), object())
        with patch.object(
            build_module, "split_objects", return_value=split
        ) as split_objects, patch.object(
            build_module, "prepare_artifacts", return_value=expected
        ) as prepare_artifacts:
            result = build_module.prepare_database(
                clips,
                heldout_count=1,
                seed=7,
                skeleton=G1_SKELETON,
            )

        self.assertEqual(result, (*expected, split))
        split_objects.assert_called_once_with(
            clips, heldout_count=1, seed=7
        )
        prepare_artifacts.assert_called_once_with(
            clips, split, G1_SKELETON
        )
        for forbidden in (
            "partition_clips",
            "assemble_database",
            "build_features",
            "build_database_features",
        ):
            self.assertFalse(hasattr(build_module, forbidden), forbidden)


class InteractionBuildCliTests(unittest.TestCase):
    def test_parse_args_has_exact_schema_v1_defaults(self):
        args = build_cli.parse_args(
            [
                "--source-root",
                "source",
                "--source-root",
                "other-source",
                "--g1-xml",
                "g1.xml",
            ]
        )
        self.assertEqual(
            args.source_root,
            [Path("source"), Path("other-source")],
        )
        self.assertEqual(args.g1_xml, Path("g1.xml"))
        self.assertEqual(args.target_fps, 25.0)
        self.assertEqual(args.heldout_count, 20)
        self.assertEqual(args.seed, 20260714)
        self.assertIsNone(args.limit)
        self.assertFalse(args.allow_rejections)
        self.assertEqual(args.output, Path("resources/g1_interaction"))

    def test_multiple_roots_publish_sorted_provenance_and_split_shared_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "data"
            table_root = base / "pickup_table"
            ground_root = base / "pickup_ground"
            output = Path(tmp) / "pack"
            write_source_fixture(
                table_root,
                sequence_id="pickup_table__cup_2__001",
                object_id="cup_2",
            )
            write_source_fixture(
                table_root,
                sequence_id="pickup_table__table_only__001",
                object_id="table_only",
            )
            write_source_fixture(
                ground_root,
                sequence_id="pickup_ground__cup_2__001",
                object_id="cup_2",
                ground=True,
            )
            write_source_fixture(
                ground_root,
                sequence_id="pickup_ground__ground_only__001",
                object_id="ground_only",
                ground=True,
            )
            argv = [
                "--source-root",
                str(ground_root),
                "--source-root",
                str(table_root),
                "--source-root",
                str(table_root),
                "--g1-xml",
                str(base / "g1.xml"),
                "--output",
                str(output),
                "--heldout-count",
                "1",
            ]
            with fast_build_patches(), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(build_cli.main(argv), 0)

            _, _, manifest, split, _ = read_artifact_set(output)
            self.assertEqual(
                manifest["source_roots"],
                sorted([str(table_root.resolve()), str(ground_root.resolve())]),
            )
            self.assertEqual(manifest["source_root"], str(base.resolve()))
            self.assertEqual(
                {clip["source_category"] for clip in manifest["clips"]},
                {"pickup_ground", "pickup_table"},
            )
            cup_sequences = {
                "pickup_table__cup_2__001",
                "pickup_ground__cup_2__001",
            }
            database_sequences = {
                clip["sequence_id"] for clip in manifest["clips"]
            }
            self.assertIn(
                database_sequences & cup_sequences,
                (set(), cup_sequences),
            )
            self.assertTrue(
                set(split["database_objects"]).isdisjoint(
                    split["heldout_objects"]
                )
            )

    def test_parse_args_rejects_non_schema_fps_and_nonpositive_counts(self):
        cases = (
            (["--target-fps", "50"], "schema v1 requires --target-fps 25"),
            (["--heldout-count", "0"], "--heldout-count must be positive"),
            (["--limit", "0"], "--limit must be positive"),
        )
        required = ["--source-root", "source", "--g1-xml", "g1.xml"]
        for extra, message in cases:
            with self.subTest(extra=extra), self.assertRaises(
                SystemExit
            ), contextlib.redirect_stderr(io.StringIO()) as stderr:
                build_cli.parse_args(required + extra)
            self.assertIn(message, stderr.getvalue())

    def test_source_shape_error_has_frozen_code_and_normal_mode_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output = Path(tmp) / "pack"
            sources = write_sources(
                root,
                [
                    ("pickup_table__alpha__001", "alpha"),
                    ("pickup_table__beta__001", "beta"),
                    ("pickup_table__broken__001", "broken"),
                ],
            )
            corrupt_object_frame_count(sources[-1])
            with fast_build_patches(), contextlib.redirect_stderr(
                io.StringIO()
            ) as stderr:
                result = build_cli.main(
                    build_argv(root, output, "--heldout-count", "1")
                )

            self.assertNotEqual(result, 0)
            self.assertFalse(output.exists())
            self.assertIn("frame_count_mismatch", stderr.getvalue())

    def test_allow_rejections_publishes_valid_partitions_and_exact_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output = Path(tmp) / "pack"
            sources = write_sources(
                root,
                [
                    ("pickup_table__alpha__001", "alpha"),
                    ("pickup_table__beta__001", "beta"),
                    ("pickup_table__broken__001", "broken"),
                ],
            )
            corrupt_object_frame_count(sources[-1])
            with fast_build_patches(), contextlib.redirect_stdout(io.StringIO()):
                result = build_cli.main(
                    build_argv(
                        root,
                        output,
                        "--heldout-count",
                        "1",
                        "--allow-rejections",
                    )
                )

            self.assertEqual(result, 0)
            artifact, features, manifest, split, report = read_artifact_set(output)
            self.assertEqual(set(output.iterdir()), {
                output / "interaction_database.bin",
                output / "interaction_features.bin",
                output / "manifest.json",
                output / "evaluation_split.json",
                output / "validation_report.json",
            })
            self.assertEqual(manifest["source_clips"], 3)
            self.assertEqual(manifest["included_clips"], 2)
            self.assertEqual(manifest["rejected_clips"], 1)
            self.assertEqual(
                (
                    report["source_clips"],
                    report["included_clips"],
                    report["rejected_clips"],
                ),
                (3, 2, 1),
            )
            self.assertEqual(len(manifest["clips"]), len(artifact.range_starts))
            self.assertLess(len(manifest["clips"]), manifest["included_clips"])
            self.assertEqual(report["included_frames"], len(artifact.positions))
            self.assertEqual(len(features.values), len(artifact.positions))
            self.assertEqual(len(split["heldout_objects"]), 1)
            self.assertEqual(
                report["rejections_by_code"], {"frame_count_mismatch": 1}
            )
            self.assertEqual(
                tuple(report["rejections"][0]),
                ("code", "message", "object_id", "sequence_id", "stage"),
            )
            self.assertEqual(
                set(report["rejections"][0]),
                {"sequence_id", "object_id", "stage", "code", "message"},
            )

    def test_rejection_histogram_is_exact_for_repeated_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output = Path(tmp) / "pack"
            sources = write_sources(
                root,
                [
                    ("pickup_table__alpha__001", "alpha"),
                    ("pickup_table__beta__001", "beta"),
                    ("pickup_table__broken_a__001", "broken_a"),
                    ("pickup_table__broken_b__001", "broken_b"),
                ],
            )
            corrupt_object_frame_count(sources[-2])
            corrupt_object_frame_count(sources[-1])
            with fast_build_patches(), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    build_cli.main(
                        build_argv(
                            root,
                            output,
                            "--heldout-count",
                            "1",
                            "--allow-rejections",
                        )
                    ),
                    0,
                )
            report = json.loads(
                (output / "validation_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["rejected_clips"], 2)
            self.assertEqual(
                report["rejections_by_code"], {"frame_count_mismatch": 2}
            )
            self.assertEqual(
                [item["sequence_id"] for item in report["rejections"]],
                [
                    "pickup_table__broken_a__001",
                    "pickup_table__broken_b__001",
                ],
            )

    def test_allow_rejections_refuses_empty_database_or_heldout_partition(self):
        cases = (
            (
                [("pickup_table__only__001", "only")],
                1,
            ),
            (
                [
                    ("pickup_table__alpha__001", "alpha"),
                    ("pickup_table__beta__001", "beta"),
                ],
                2,
            ),
        )
        for identities, heldout_count in cases:
            with self.subTest(
                identities=identities
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "source"
                output = Path(tmp) / "pack"
                write_sources(root, identities)
                with fast_build_patches(), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    result = build_cli.main(
                        build_argv(
                            root,
                            output,
                            "--heldout-count",
                            str(heldout_count),
                            "--allow-rejections",
                        )
                    )
                self.assertNotEqual(result, 0)
                self.assertFalse(output.exists())

    def test_limit_selects_first_lexicographic_sources_and_records_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output = Path(tmp) / "pack"
            write_sources(
                root,
                [
                    ("pickup_table__zulu__001", "zulu"),
                    ("pickup_table__alpha__001", "alpha"),
                    ("pickup_table__middle__001", "middle"),
                ],
            )
            seen = []

            def recording_convert(raw, kinematics, target_fps):
                seen.append(raw.sequence_id)
                return fake_convert(raw, kinematics, target_fps)

            with fast_build_patches(
                converter=recording_convert
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    build_cli.main(
                        build_argv(
                            root,
                            output,
                            "--limit",
                            "2",
                            "--heldout-count",
                            "1",
                        )
                    ),
                    0,
                )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                seen,
                [
                    "pickup_table__alpha__001",
                    "pickup_table__middle__001",
                ],
            )
            self.assertEqual(manifest["source_clips"], 2)
            self.assertEqual(manifest["diagnostic_limit"], 2)

    def test_object_dimensions_are_read_once_per_selected_object_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output = Path(tmp) / "pack"
            write_sources(
                root,
                [
                    ("pickup_table__alpha__001", "alpha"),
                    ("pickup_table__alpha__002", "alpha"),
                    ("pickup_table__beta__001", "beta"),
                ],
            )
            with fast_build_patches() as (
                _,
                dimension_reader,
                _,
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    build_cli.main(
                        build_argv(root, output, "--heldout-count", "1")
                    ),
                    0,
                )
            self.assertEqual(dimension_reader.call_count, 2)
            self.assertEqual(
                [path.stem for (path,), _ in dimension_reader.call_args_list],
                [
                    "pickup_table__alpha__001",
                    "pickup_table__beta__001",
                ],
            )

    def test_corpus_path_accepts_valid_fortran_converted_array_layout(self):
        def fortran_contact_convert(raw, kinematics, target_fps):
            motion, skeleton, report = fake_convert(
                raw, kinematics, target_fps
            )
            motion.foot_contacts = np.asfortranarray(
                motion.foot_contacts
            )
            self.assertFalse(motion.foot_contacts.flags.c_contiguous)
            return motion, skeleton, report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output = Path(tmp) / "pack"
            write_sources(
                root,
                [
                    ("pickup_table__alpha__001", "alpha"),
                    ("pickup_table__beta__001", "beta"),
                ],
            )
            with fast_build_patches(
                converter=fortran_contact_convert
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    build_cli.main(
                        build_argv(root, output, "--heldout-count", "1")
                    ),
                    0,
                )
            artifact, _, _, _, _ = read_artifact_set(output)
            self.assertTrue(artifact.foot_contacts.flags.c_contiguous)

    def test_manifest_contains_frozen_identity_order_and_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output = Path(tmp) / "pack"
            write_sources(
                root,
                [
                    ("pickup_table__alpha__001", "alpha"),
                    ("pickup_table__beta__001", "beta"),
                    ("pickup_table__charlie__001", "charlie"),
                ],
            )
            with fast_build_patches(), patch.dict(
                os.environ, {}, clear=False
            ), contextlib.redirect_stdout(io.StringIO()):
                os.environ.pop("SOURCE_DATE_EPOCH", None)
                self.assertEqual(
                    build_cli.main(
                        build_argv(root, output, "--heldout-count", "1")
                    ),
                    0,
                )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["database_magic"], "G1INTDB1")
            self.assertEqual(manifest["feature_magic"], "G1INTFT1")
            self.assertEqual(manifest["target_fps"], 25.0)
            self.assertEqual(manifest["source_root"], str(root.resolve()))
            self.assertNotIn("source_roots", manifest)
            self.assertEqual(manifest["dataset_id"], fetch_cli.DATASET_ID)
            self.assertEqual(manifest["skeleton_names"], list(G1_SKELETON.names))
            self.assertEqual(
                manifest["skeleton_parents"], G1_SKELETON.parents.tolist()
            )
            self.assertEqual(
                manifest["skeleton_signature"], G1_SKELETON.signature()
            )
            self.assertEqual(len(manifest["feature_names"]), 71)
            self.assertEqual(
                [(group["name"], group["start"], group["stop"])
                 for group in manifest["feature_groups"]],
                [
                    ("pose", 0, 33),
                    ("trajectory", 33, 45),
                    ("grasp", 45, 57),
                    ("root_target", 57, 65),
                    ("context", 65, 71),
                ],
            )
            self.assertEqual(manifest["split"]["seed"], 20260714)
            self.assertEqual(manifest["split"]["heldout_object_count"], 1)
            self.assertEqual(manifest["split"]["database_object_count"], 2)
            self.assertEqual(
                manifest["phase_config"], dataclasses.asdict(PhaseConfig())
            )
            self.assertEqual(manifest["diagnostic_limit"], None)
            self.assertIsNone(manifest["source_date_epoch"])
            self.assertRegex(manifest["git_commit"], r"^[0-9a-f]{40}$")
            self.assertEqual(
                set(manifest["dependency_versions"]),
                {
                    "huggingface-hub",
                    "joblib",
                    "mujoco",
                    "numpy",
                    "scipy",
                    "usd-core",
                },
            )
            self.assertTrue(
                all(
                    isinstance(version, str) and version
                    for version in manifest["dependency_versions"].values()
                )
            )
            sequence_ids = [clip["sequence_id"] for clip in manifest["clips"]]
            self.assertEqual(sequence_ids, sorted(sequence_ids))
            self.assertEqual(len(sequence_ids), len(set(sequence_ids)))
            for clip in manifest["clips"]:
                self.assertEqual(
                    set(clip),
                    {
                        "sequence_id",
                        "object_id",
                        "active_hand",
                        "range_start",
                        "range_stop",
                    },
                )
                self.assertNotIn("source_category", clip)

    def test_source_date_epoch_is_null_when_absent_and_integer_when_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            write_sources(
                root,
                [
                    ("pickup_table__alpha__001", "alpha"),
                    ("pickup_table__beta__001", "beta"),
                ],
            )
            absent = Path(tmp) / "absent"
            present = Path(tmp) / "present"
            with fast_build_patches(), patch.dict(
                os.environ, {}, clear=False
            ), contextlib.redirect_stdout(io.StringIO()):
                os.environ.pop("SOURCE_DATE_EPOCH", None)
                self.assertEqual(
                    build_cli.main(
                        build_argv(root, absent, "--heldout-count", "1")
                    ),
                    0,
                )
            with fast_build_patches(), patch.dict(
                os.environ, {"SOURCE_DATE_EPOCH": "1720950000"}, clear=False
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    build_cli.main(
                        build_argv(root, present, "--heldout-count", "1")
                    ),
                    0,
                )
            absent_value = json.loads(
                (absent / "manifest.json").read_text(encoding="utf-8")
            )["source_date_epoch"]
            present_value = json.loads(
                (present / "manifest.json").read_text(encoding="utf-8")
            )["source_date_epoch"]
            self.assertIsNone(absent_value)
            self.assertEqual(present_value, 1720950000)
            self.assertIs(type(present_value), int)

    def test_invalid_source_date_epoch_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output = Path(tmp) / "pack"
            write_sources(
                root,
                [
                    ("pickup_table__alpha__001", "alpha"),
                    ("pickup_table__beta__001", "beta"),
                ],
            )
            with fast_build_patches(), patch.dict(
                os.environ, {"SOURCE_DATE_EPOCH": "not-an-integer"}, clear=False
            ), self.assertRaisesRegex(ValueError, "SOURCE_DATE_EPOCH"):
                build_cli.main(
                    build_argv(root, output, "--heldout-count", "1")
                )
            self.assertFalse(output.exists())

    def test_repeated_builds_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            write_sources(
                root,
                [
                    ("pickup_table__alpha__001", "alpha"),
                    ("pickup_table__beta__001", "beta"),
                    ("pickup_table__charlie__001", "charlie"),
                ],
            )
            with fast_build_patches(), patch.dict(
                os.environ, {"SOURCE_DATE_EPOCH": "1720950000"}, clear=False
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    build_cli.main(
                        build_argv(root, first, "--heldout-count", "1")
                    ),
                    0,
                )
            with fast_build_patches(), patch.dict(
                os.environ, {"SOURCE_DATE_EPOCH": "1720950000"}, clear=False
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    build_cli.main(
                        build_argv(root, second, "--heldout-count", "1")
                    ),
                    0,
                )
            self.assertEqual(artifact_snapshot(first), artifact_snapshot(second))

    def test_failed_rebuild_preserves_prior_complete_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output = Path(tmp) / "pack"
            sources = write_sources(
                root,
                [
                    ("pickup_table__alpha__001", "alpha"),
                    ("pickup_table__beta__001", "beta"),
                ],
            )
            with fast_build_patches(), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    build_cli.main(
                        build_argv(root, output, "--heldout-count", "1")
                    ),
                    0,
                )
            before = artifact_snapshot(output)
            corrupt_object_frame_count(sources[0])
            with fast_build_patches(), contextlib.redirect_stderr(io.StringIO()):
                self.assertNotEqual(
                    build_cli.main(
                        build_argv(root, output, "--heldout-count", "1")
                    ),
                    0,
                )
            self.assertEqual(before, artifact_snapshot(output))
            read_artifact_set(output)


class InteractionManifestMetadataTests(unittest.TestCase):
    def test_writer_rejects_every_corrupted_required_metadata_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            baseline = Path(tmp) / "baseline"
            publish_fast_valid_pack(self, root, baseline)
            artifact, features, manifest, split_value, report = (
                read_artifact_set(baseline)
            )
            split = EvaluationSplit(
                seed=split_value["seed"],
                database_objects=tuple(split_value["database_objects"]),
                heldout_objects=tuple(split_value["heldout_objects"]),
            )

            for index, (label, mutation, message) in enumerate(
                manifest_metadata_corruptions(artifact)
            ):
                with self.subTest(label=label):
                    invalid = copy.deepcopy(manifest)
                    mutation(invalid)
                    output = Path(tmp) / f"writer-corrupt-{index}"
                    with self.assertRaisesRegex(ValueError, message):
                        write_artifact_set(
                            output,
                            artifact,
                            features,
                            split,
                            invalid,
                            report,
                        )
                    self.assertFalse(output.exists())

    def test_standalone_validator_rejects_every_corrupted_metadata_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            baseline = Path(tmp) / "baseline"
            publish_fast_valid_pack(self, root, baseline)
            artifact, _, _, _, _ = read_artifact_set(baseline)

            for index, (label, mutation, message) in enumerate(
                manifest_metadata_corruptions(artifact)
            ):
                with self.subTest(label=label):
                    corrupt = Path(tmp) / f"reader-corrupt-{index}"
                    shutil.copytree(baseline, corrupt)
                    rewrite_json(corrupt / "manifest.json", mutation)
                    with self.assertRaisesRegex(ValueError, message):
                        validate_cli.validate(corrupt)


class InteractionValidatorTests(unittest.TestCase):
    def test_validator_rereads_all_files_hashes_binaries_and_prints_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output = Path(tmp) / "pack"
            write_sources(
                root,
                [
                    ("pickup_table__alpha__001", "alpha"),
                    ("pickup_table__beta__001", "beta"),
                ],
            )
            with fast_build_patches(), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    build_cli.main(
                        build_argv(root, output, "--heldout-count", "1")
                    ),
                    0,
                )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    validate_cli.main(["--input", str(output)]), 0
                )
            lines = stdout.getvalue().splitlines()
            self.assertEqual(len(lines), 1)
            db_hash = hashlib.sha256(
                (output / "interaction_database.bin").read_bytes()
            ).hexdigest()
            feature_hash = hashlib.sha256(
                (output / "interaction_features.bin").read_bytes()
            ).hexdigest()
            self.assertEqual(
                lines[0],
                "VALID schema=1 fps=25 bones=31 features=71 clips=1 "
                "frames=75 heldout_objects=1 "
                f"db_sha256={db_hash} feature_sha256={feature_hash}",
            )


class InteractionFetchTests(unittest.TestCase):
    def test_fetch_contract_is_constrained_to_one_dataset_and_exact_patterns(self):
        self.assertEqual(
            fetch_cli.DATASET_ID,
            "nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL",
        )
        self.assertEqual(
            list(fetch_cli.ALLOW_PATTERNS),
            [
                "data/pickup_table/robot/*.pkl",
                "data/pickup_table/objects/*.pkl",
                "data/pickup_table/meta/*.pkl",
                "data/pickup_table/object_usd/*.usd",
                "data/pickup_table/object_usd/textures/*",
            ],
        )

    def test_dry_run_subprocess_needs_no_client_network_or_output_path(self):
        secret = "hf_super_secret_token"
        guard = """
import builtins
import runpy
import socket
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "huggingface_hub" or name.startswith("huggingface_hub."):
        raise ModuleNotFoundError("huggingface_hub deliberately unavailable")
    return real_import(name, globals, locals, fromlist, level)

def reject_network(*args, **kwargs):
    raise AssertionError("network access attempted during dry-run")

builtins.__import__ = guarded_import
socket.create_connection = reject_network
socket.socket.connect = reject_network
sys.argv = [
    "resources.fetch_grail_pickup_table",
    "--output",
    sys.argv[1],
    "--dry-run",
]
runpy.run_module("resources.fetch_grail_pickup_table", run_name="__main__")
"""
        expected = "\n".join(
            [
                f"dataset={fetch_cli.DATASET_ID}",
                *(f"allow_pattern={item}" for item in fetch_cli.ALLOW_PATTERNS),
            ]
        ) + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "grail-dry-run"
            environment = os.environ.copy()
            environment["HF_TOKEN"] = secret
            result = subprocess.run(
                [sys.executable, "-c", guard, str(output)],
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, expected)
            self.assertEqual(result.stderr, "")
            self.assertFalse(output.exists())
            self.assertNotIn(secret, result.stdout + result.stderr)

    def test_mocked_download_uses_only_exact_arguments_and_keeps_token_secret(self):
        secret = "hf_explicit_secret"
        stdout = io.StringIO()
        with patch.object(
            fetch_cli, "_snapshot_download"
        ) as download, contextlib.redirect_stdout(stdout):
            self.assertEqual(
                fetch_cli.main(
                    [
                        "--output",
                        "/tmp/grail",
                        "--token",
                        secret,
                    ]
                ),
                0,
            )
        download.assert_called_once_with(
            repo_id=fetch_cli.DATASET_ID,
            repo_type="dataset",
            local_dir=Path("/tmp/grail"),
            allow_patterns=list(fetch_cli.ALLOW_PATTERNS),
            token=secret,
        )
        self.assertNotIn(secret, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
