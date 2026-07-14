import contextlib
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
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
from resources.g1_interaction_builder.artifacts import read_artifact_set
from resources.g1_interaction_builder.schema import (
    ConversionValidationError,
    EvaluationSplit,
    G1_SKELETON,
    InteractionValidationError,
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


def artifact_snapshot(output: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir())
        if path.is_file()
    }


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


class InteractionBuildUnitTests(unittest.TestCase):
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
            ["--source-root", "source", "--g1-xml", "g1.xml"]
        )
        self.assertEqual(args.source_root, Path("source"))
        self.assertEqual(args.g1_xml, Path("g1.xml"))
        self.assertEqual(args.target_fps, 25.0)
        self.assertEqual(args.heldout_count, 20)
        self.assertEqual(args.seed, 20260714)
        self.assertIsNone(args.limit)
        self.assertFalse(args.allow_rejections)
        self.assertEqual(args.output, Path("resources/g1_interaction"))

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
            self.assertEqual(manifest["phase_config"]["stable_source_samples"], 3)
            self.assertEqual(manifest["diagnostic_limit"], None)
            self.assertIsNone(manifest["source_date_epoch"])
            self.assertRegex(manifest["git_commit"], r"^[0-9a-f]{40}$")
            self.assertTrue(manifest["dependency_versions"])
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

    def test_dry_run_never_downloads_or_prints_token(self):
        secret = "hf_super_secret_token"
        stdout = io.StringIO()
        with patch.dict(os.environ, {"HF_TOKEN": secret}), patch.object(
            fetch_cli, "snapshot_download"
        ) as download, contextlib.redirect_stdout(stdout):
            self.assertEqual(
                fetch_cli.main(["--output", "/tmp/grail", "--dry-run"]),
                0,
            )
        download.assert_not_called()
        output = stdout.getvalue()
        self.assertIn(fetch_cli.DATASET_ID, output)
        for pattern in fetch_cli.ALLOW_PATTERNS:
            self.assertIn(pattern, output)
        self.assertNotIn(secret, output)

    def test_mocked_download_uses_only_exact_arguments_and_keeps_token_secret(self):
        secret = "hf_explicit_secret"
        stdout = io.StringIO()
        with patch.object(
            fetch_cli, "snapshot_download"
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
