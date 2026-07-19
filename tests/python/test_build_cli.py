import gc
import io
import inspect
import json
import os
import struct
import subprocess
import tempfile
import unittest
import weakref
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np

from resources import build_g1_terrain_database as builder
from resources.g1_terrain_builder.artifacts import (
    read_support_sidecar,
    read_terrain_sidecar,
    read_walkability,
)
from resources.g1_terrain_builder.database import read_holden_database
from resources.g1_terrain_builder.scenes import REQUIRED_SCENE_IDS
from resources.g1_terrain_builder.schema import HoldenClip, SkeletonSpec


PYTHON = "/home/ubuntu/miniconda3/envs/diffsim/bin/python"


class BuildCliTests(unittest.TestCase):
    def test_grail_limit_measures_complete_corpus_before_motion_selection(self):
        args = SimpleNamespace(grail_glob="clips/*.pkl", grail_limit=1)
        paths = [
            "clips/e.pkl", "clips/c.pkl", "clips/a.pkl",
            "clips/d.pkl", "clips/b.pkl",
        ]
        heights = {
            "a": 0.10, "b": 0.20, "c": 0.30, "d": 0.40, "e": 0.50,
        }
        selected = {
            "grail-curb-default": "a",
            "grail-curb-low": "b",
            "grail-curb-medium": "c",
            "grail-curb-high": "d",
        }
        measured = []

        class Terrain:
            def __init__(self, base):
                self.base = base

            def footprint(self):
                measured.append(self.base)
                return {"height": heights[self.base]}

        with (
            mock.patch.object(builder.glob, "glob", return_value=paths),
            mock.patch.object(builder, "_require_file") as require_file,
            mock.patch.object(
                builder.GrailTerrain, "from_base",
                side_effect=lambda base: Terrain(base)),
            mock.patch.object(
                builder, "select_grail_scene_bases",
                return_value=selected, create=True) as select,
        ):
            observed = builder._inspect_grail_corpus(args)

        all_paths, motion_paths, path_by_base, observed_heights, observed_selected = \
            observed
        self.assertEqual(all_paths, tuple(sorted(paths)))
        self.assertEqual(motion_paths, ("clips/a.pkl",))
        self.assertEqual(tuple(path_by_base), ("a", "b", "c", "d", "e"))
        self.assertEqual(measured, ["a", "b", "c", "d", "e"])
        self.assertEqual(observed_heights, heights)
        self.assertEqual(observed_selected, selected)
        self.assertEqual(
            [call.args for call in require_file.call_args_list],
            [(path, "GRAIL clip") for path in sorted(paths)])
        select.assert_called_once_with(heights)

    def test_duplicate_grail_basename_fails_before_surface_measurement(self):
        args = SimpleNamespace(grail_glob="clips/*.pkl", grail_limit=1)
        with (
            mock.patch.object(
                builder.glob, "glob",
                return_value=["clips/a/same.pkl", "clips/b/same.pkl"]),
            mock.patch.object(builder, "_require_file"),
            mock.patch.object(
                builder.GrailTerrain, "from_base") as from_base,
            self.assertRaisesRegex(ValueError, "duplicate GRAIL.*base"),
        ):
            builder._inspect_grail_corpus(args)
        from_base.assert_not_called()

    def test_empty_grail_corpus_is_rejected_even_at_limit_zero(self):
        args = SimpleNamespace(grail_glob="missing/*.pkl", grail_limit=0)
        with (
            mock.patch.object(builder.glob, "glob", return_value=[]),
            self.assertRaisesRegex(FileNotFoundError, "matched no clips"),
        ):
            builder._inspect_grail_corpus(args)

    def test_assembly_streams_sources_and_excludes_route_only_clips(self):
        args = SimpleNamespace(
            output="unused", grail_glob="clips/*.pkl", grail_limit=1,
            g1_xml="g1.xml", takara="takara.npz", remap="remap.npy",
        )
        bases = ("motion", "route-a", "route-b", "route-c")
        all_paths = tuple(f"clips/{base}.pkl" for base in bases)
        path_by_base = dict(zip(bases, all_paths))
        measured_heights = {
            "motion": 0.10, "route-a": 0.20,
            "route-b": 0.30, "route-c": 0.40,
        }
        selected = {
            "grail-curb-default": "motion",
            "grail-curb-low": "route-a",
            "grail-curb-medium": "route-b",
            "grail-curb-high": "route-c",
        }
        skeleton = SkeletonSpec(
            tuple(f"bone-{index}" for index in range(31)),
            np.arange(-1, 30, dtype=np.int32))
        source_refs = []
        clip_refs = []
        load_events = []

        class Source:
            pass

        def source(name, terrain_id):
            value = Source()
            value.name = name
            value.terrain_id = terrain_id
            value.fps = 25.0
            value.qpos = np.zeros((2, 36), np.float32)
            value.source_frames = np.arange(2, dtype=np.int64)
            source_refs.append(weakref.ref(value))
            return value

        def load_takara(path, remap):
            load_events.append("takara_walk_50hz")
            return source("takara_walk_50hz", "flat")

        def load_grail(path):
            gc.collect()
            self.assertTrue(all(reference() is None for reference in source_refs))
            base = os.path.splitext(os.path.basename(path))[0]
            load_events.append(base)
            return source(base, base)

        def clip_for(value):
            clip = HoldenClip.empty(2, 31)
            clip.name = value.name
            clip.terrain_id = value.terrain_id
            clip_refs.append((clip.name, weakref.ref(clip)))
            return clip

        report = {
            "fk_max_error_m": 0.0,
            "duration_error_s": 0.0,
            "quaternion_norm_max_error": 0.0,
        }

        def finalize(value, terrain, kinematics):
            return clip_for(value), skeleton, dict(report)

        def convert(value, kinematics, output_fps):
            return clip_for(value), skeleton, dict(report)

        def all_definitions(observed_heights, clips_by_terrain):
            self.assertEqual(observed_heights, measured_heights)
            self.assertEqual(set(clips_by_terrain), set(bases))
            return ("all-definitions",)

        scene_pack = SimpleNamespace(scenes=tuple(range(14)))

        with (
            mock.patch.object(builder, "_require_file"),
            mock.patch.object(
                builder, "_inspect_grail_corpus",
                return_value=(
                    all_paths, all_paths[:1], path_by_base,
                    measured_heights, selected)),
            mock.patch.object(builder, "G1Kinematics", return_value=object()),
            mock.patch.object(builder, "load_takara", side_effect=load_takara),
            mock.patch.object(builder, "load_grail", side_effect=load_grail),
            mock.patch.object(builder, "FlatTerrain", return_value=object()),
            mock.patch.object(
                builder.GrailTerrain, "from_base", return_value=object()),
            mock.patch.object(builder, "finalize_clip", new=finalize),
            mock.patch.object(
                builder, "convert_source_clip", new=convert),
            mock.patch.object(
                builder, "all_scene_definitions",
                new=all_definitions, create=True),
            mock.patch.object(
                builder, "build_scene_pack",
                new=lambda definitions: scene_pack, create=True),
        ):
            artifacts, manifest, observed_pack = \
                builder._assemble_candidate(args)

        gc.collect()
        self.assertIs(observed_pack, scene_pack)
        self.assertEqual(load_events, [
            "takara_walk_50hz", "motion", "route-a", "route-b", "route-c",
        ])
        self.assertEqual(
            [entry["name"] for entry in manifest["sources"]],
            ["takara_walk_50hz", "motion"])
        self.assertEqual(manifest["total_clips"], 2)
        self.assertEqual(manifest["grail_clips"], 1)
        self.assertEqual(set(manifest), {
            "schema", "output_fps", "feature_dimensions",
            "terrain_dimensions", "support_dimensions",
            "terrain_feature_distances_m", "total_clips", "grail_clips",
            "skipped_clips", "database_frames", "diagnostic_mode",
            "sources", "skeleton", "contact", "surface", "validation",
        })
        self.assertEqual(manifest["schema"], "g1-terrain-artifacts/v2")
        self.assertEqual(manifest["support_dimensions"], 3)
        self.assertEqual(
            manifest["terrain_feature_distances_m"], [0.25, 0.5, 0.75, 1.0])
        self.assertEqual(set(manifest["surface"]), {"semantics", "signature"})
        self.assertEqual(set(manifest["validation"]), {
            "schema", "fk_max_error_m", "duration_error_s",
            "quaternion_norm_max_error",
        })
        self.assertEqual(
            manifest["validation"]["schema"],
            "g1-terrain-validation/v1")
        self.assertEqual(len(artifacts.range_starts), 2)
        self.assertEqual(len(artifacts.positions), 4)
        self.assertTrue(all(reference() is None for reference in source_refs))
        self.assertTrue(all(reference() is None for _, reference in clip_refs))

    def test_loaded_grail_identity_must_match_validated_basename(self):
        valid = SimpleNamespace(name="terrain-a", terrain_id="terrain-a")
        builder._require_loaded_grail_source(valid, "terrain-a")
        cases = (
            (SimpleNamespace(name="wrong", terrain_id="terrain-a"), "name"),
            (SimpleNamespace(name="terrain-a", terrain_id="wrong"), "terrain"),
        )
        for source, field in cases:
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "loaded source name/terrain identity changed",
            ):
                builder._require_loaded_grail_source(source, "terrain-a")

    def test_candidate_validator_argv_is_exact_in_both_modes(self):
        base = dict(
            output="unused", grail_glob="/data/grail/*.pkl",
            g1_xml="/models/g1.xml", takara="/motion/takara.npz",
            remap="/motion/remap.npy",
        )
        validator = os.path.join(
            builder.REPOSITORY_ROOT,
            "resources", "validate_g1_terrain_database.py")
        diagnostic_argv = [
            builder.sys.executable, validator, "/tmp/candidate"]
        cases = (
            (0, diagnostic_argv),
            (1, diagnostic_argv),
            (None, [
                builder.sys.executable, validator, "/tmp/candidate",
                "--full-source-validation",
                "--grail-glob", "/data/grail/*.pkl",
                "--g1-xml", "/models/g1.xml",
                "--takara", "/motion/takara.npz",
                "--remap", "/motion/remap.npy",
            ]),
        )
        for limit, expected in cases:
            args = SimpleNamespace(grail_limit=limit, **base)
            with self.subTest(limit=limit), mock.patch.object(
                subprocess, "run",
            ) as run:
                callback = builder._candidate_validation_policy(args)
                callback("/tmp/candidate")
                run.assert_called_once_with(expected, check=True)

    def test_candidate_validator_nonzero_is_a_value_error(self):
        args = SimpleNamespace(
            output="unused", grail_glob="/data/grail/*.pkl", grail_limit=1,
            g1_xml="/models/g1.xml", takara="/motion/takara.npz",
            remap="/motion/remap.npy",
        )
        failure = subprocess.CalledProcessError(7, ["validator"])
        with mock.patch.object(
            subprocess, "run", side_effect=failure,
        ), self.assertRaisesRegex(
            ValueError, "candidate validator failed.*status 7",
        ):
            builder._run_candidate_validator("/tmp/candidate", args)

    def test_task7_legacy_builder_still_lacks_explicit_index_arguments(self):
        self.assertEqual(
            tuple(inspect.signature(builder.publish_artifacts).parameters),
            (
                "output_dir", "artifacts", "manifest_base", "scene_pack",
                "motion_index", "terrain_banks", "validate_candidate",
            ),
        )
        args = SimpleNamespace(
            output="published", grail_glob="clips/*.pkl", grail_limit=1,
            g1_xml="g1.xml", takara="takara.npz", remap="remap.npy",
        )
        artifacts = object()
        manifest_base = {"schema": "g1-terrain-artifacts/v2"}
        scene_pack = object()
        callback = lambda staging: None
        finalized = {"schema": "g1-terrain-artifacts/v2"}
        with (
            mock.patch.object(
                builder, "_assemble_candidate",
                return_value=(artifacts, manifest_base, scene_pack)) as assemble,
            mock.patch.object(
                builder, "_candidate_validation_policy",
                return_value=callback, create=True) as policy,
            mock.patch.object(
                builder, "publish_artifacts",
                return_value=finalized) as publish,
        ):
            observed = builder.build_artifacts(args)

        self.assertIs(observed, finalized)
        assemble.assert_called_once_with(args)
        policy.assert_called_once_with(args)
        publish.assert_called_once_with(
            "published", artifacts, manifest_base, scene_pack, callback)

    def test_parser_removes_the_obsolete_runtime_terrain_option(self):
        parser = builder._parser()
        self.assertNotIn("runtime_terrain", vars(parser.parse_args([])))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--runtime-terrain", "obsolete"])

    def test_success_summary_reports_the_complete_scene_count(self):
        manifest = {
            "schema": "g1-terrain-artifacts/v2",
            "database_frames": 7,
            "total_clips": 2,
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(builder, "build_artifacts", return_value=manifest),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            status = builder.main(["--output", "/tmp/candidate"])
        self.assertEqual(status, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(
            stdout.getvalue(),
            "BUILT g1-terrain-artifacts/v2 frames=7 clips=2 scenes=14 "
            "output=/tmp/candidate\n")

    @unittest.skip(
        "Task 7 must make the legacy single-curb builder emit v3 indexes"
    )
    def test_one_grail_clip_builds_and_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "g1_terrain")
            built = subprocess.run([
                PYTHON, "resources/build_g1_terrain_database.py",
                "--output", output, "--grail-limit", "1",
            ], check=True, text=True, capture_output=True)
            validated = subprocess.run([
                PYTHON, "resources/validate_g1_terrain_database.py",
                output,
            ], check=True, text=True, capture_output=True)
            with open(
                os.path.join(output, "manifest.json"), encoding="utf-8",
            ) as stream:
                manifest = json.load(stream)
            with open(
                os.path.join(output, "scenes", "index.json"),
                encoding="utf-8",
            ) as stream:
                index = json.load(stream)
            database = read_holden_database(
                os.path.join(output, "database.bin"))
            terrain_features = read_terrain_sidecar(
                os.path.join(output, "terrain_features.bin"))
            terrain_support = read_support_sidecar(
                os.path.join(output, "terrain_support.bin"))
            scene_headers = {}
            walkability_shapes = {}
            for scene_id in REQUIRED_SCENE_IDS:
                scene_root = os.path.join(output, "scenes", scene_id)
                with open(os.path.join(scene_root, "terrain.bin"), "rb") as stream:
                    scene_headers[scene_id] = struct.unpack(
                        "<4sIII4f", stream.read(32))
                walkability_shapes[scene_id] = read_walkability(
                    os.path.join(scene_root, "walkability.bin")).shape
            grail_provenance = {}
            for scene_id in REQUIRED_SCENE_IDS[:4]:
                with open(
                    os.path.join(output, "scenes", scene_id, "scene.json"),
                    encoding="utf-8",
                ) as stream:
                    grail_provenance[scene_id] = json.load(stream)[
                        "provenance"]["source_ids"][0]

        self.assertEqual(manifest["schema"], "g1-terrain-artifacts/v2")
        self.assertEqual(built.stderr, "")
        self.assertIn(
            "VALID g1-terrain-artifacts/v2", built.stdout)
        self.assertIn(
            "BUILT g1-terrain-artifacts/v2", built.stdout)
        self.assertIn("clips=2 scenes=14", built.stdout)
        self.assertEqual(manifest["output_fps"], 25.0)
        self.assertEqual(manifest["feature_dimensions"], 31)
        self.assertEqual(manifest["terrain_dimensions"], 4)
        self.assertEqual(manifest["support_dimensions"], 3)
        self.assertEqual(manifest["grail_clips"], 1)
        self.assertEqual(manifest["total_clips"], 2)
        self.assertEqual(manifest["skipped_clips"], 0)
        self.assertTrue(manifest["diagnostic_mode"])
        self.assertEqual(set(manifest["validation"]), {
            "schema", "duration_error_s", "fk_max_error_m",
            "quaternion_norm_max_error",
        })
        self.assertEqual(len(manifest["skeleton"]["names"]), 31)
        self.assertEqual(database.positions.shape[1], 31)
        self.assertEqual(len(database.positions), len(terrain_features))
        self.assertEqual(len(database.positions), len(terrain_support))
        self.assertFalse(os.path.exists(os.path.join(output, "terrain.bin")))
        self.assertFalse(os.path.exists(os.path.join(output, "terrain.obj")))
        self.assertEqual(index["scene_ids"], list(REQUIRED_SCENE_IDS))
        self.assertEqual(
            [descriptor["id"] for descriptor in index["scenes"]],
            list(REQUIRED_SCENE_IDS))
        for scene_id in REQUIRED_SCENE_IDS:
            self.assertEqual(scene_headers[scene_id][:2], (b"G1HF", 2))
            self.assertGreaterEqual(walkability_shapes[scene_id][0], 2)
            self.assertGreaterEqual(walkability_shapes[scene_id][1], 2)
        self.assertEqual(grail_provenance, {
            "grail-curb-default": "terrain_curbs__curb_000__000",
            "grail-curb-low": "terrain_curbs__curb_186__004",
            "grail-curb-medium": "terrain_curbs__curb_022__001",
            "grail-curb-high": "terrain_curbs__curb_165__006",
        })
        takara_stop = manifest["sources"][0]["range_stop"]
        np.testing.assert_array_equal(
            terrain_support[:takara_stop],
            np.zeros((takara_stop, 3), np.float32))
        self.assertEqual(validated.stderr, "")
        self.assertEqual(
            validated.stdout,
            f"VALID g1-terrain-artifacts/v2 frames={len(database.positions)} "
            "clips=2 bones=31 terrain_dims=4 support_dims=3 scenes=14 "
            "source_rows=0\n",
        )
        cursor = 0
        for source in manifest["sources"]:
            self.assertEqual(source["range_start"], cursor)
            self.assertEqual(
                source["range_stop"] - source["range_start"],
                source["output_frames"])
            self.assertEqual(
                len(source["source_frame_map"]), source["output_frames"])
            self.assertEqual(source["range_stop"], cursor + source["output_frames"])
            cursor = source["range_stop"]
        self.assertEqual(cursor, len(database.positions))

    def test_invalid_build_arguments_fail_before_publication(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "missing")
            cases = (
                ("negative_limit", ["--grail-limit", "-1"], "non-negative"),
                ("empty_positive_glob", [
                    "--grail-glob", os.path.join(td, "none", "*.pkl"),
                    "--grail-limit", "1",
                ], "matched no clips"),
                ("empty_zero_glob", [
                    "--grail-glob", os.path.join(td, "none", "*.pkl"),
                    "--grail-limit", "0",
                ], "matched no clips"),
                ("empty_full_glob", [
                    "--grail-glob", os.path.join(td, "none", "*.pkl"),
                ], "matched no clips"),
                ("missing_g1_xml", [
                    "--g1-xml", missing, "--grail-limit", "0",
                ], "missing G1 XML"),
            )
            for name, arguments, message in cases:
                output = os.path.join(td, name)
                result = subprocess.run([
                    PYTHON, "resources/build_g1_terrain_database.py",
                    "--output", output, *arguments,
                ], text=True, capture_output=True)
                with self.subTest(name=name):
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(os.path.exists(output))

    def test_duplicate_source_names_are_rejected_before_conversion(self):
        args = SimpleNamespace(
            output="unused", grail_glob="clips/*.pkl", grail_limit=1,
            g1_xml="g1.xml", takara="takara.npz", remap="remap.npy",
        )
        collision = "clips/takara_walk_50hz.pkl"
        with (
            mock.patch.object(builder, "_require_file"),
            mock.patch.object(
                builder, "_inspect_grail_corpus", return_value=(
                    (collision,), (collision,),
                    {"takara_walk_50hz": collision},
                    {"takara_walk_50hz": 0.1},
                    {"grail-curb-default": "takara_walk_50hz"},
                )),
            mock.patch.object(builder, "finalize_clip") as finalize,
        ):
            with self.assertRaisesRegex(ValueError, "duplicate source names"):
                builder._assemble_candidate(args)
        finalize.assert_not_called()

    def test_skeleton_changes_are_rejected_before_combination(self):
        args = SimpleNamespace(
            output="unused", grail_glob="clips/*.pkl", grail_limit=1,
            g1_xml="g1.xml", takara="takara.npz", remap="remap.npy",
        )
        takara = SimpleNamespace(
            name="takara_walk_50hz", terrain_id="flat", fps=25.0,
            qpos=np.zeros((3, 36), np.float32),
        )
        grail = SimpleNamespace(
            name="grail", terrain_id="grail", fps=25.0,
            qpos=np.zeros((3, 36), np.float32),
        )
        clip_a = HoldenClip.empty(3, 2)
        clip_b = HoldenClip.empty(3, 2)
        skeleton_a = SkeletonSpec(
            ("Simulation", "Hips"), np.array([-1, 0], np.int32))
        skeleton_b = SkeletonSpec(
            ("Simulation", "Changed"), np.array([-1, 0], np.int32))
        report = {
            "fk_max_error_m": 0.0,
            "duration_error_s": 0.0,
            "quaternion_norm_max_error": 0.0,
        }
        grail_path = "clips/grail.pkl"
        with (
            mock.patch.object(builder, "_require_file"),
            mock.patch.object(
                builder, "_inspect_grail_corpus", return_value=(
                    (grail_path,), (grail_path,), {"grail": grail_path},
                    {"grail": 0.1}, {
                        "grail-curb-default": "grail",
                        "grail-curb-low": "grail",
                        "grail-curb-medium": "grail",
                        "grail-curb-high": "grail",
                    },
                )),
            mock.patch.object(builder, "G1Kinematics"),
            mock.patch.object(builder, "load_takara", return_value=takara),
            mock.patch.object(builder, "load_grail", return_value=grail),
            mock.patch.object(
                builder.GrailTerrain, "from_base", return_value=object()),
            mock.patch.object(
                builder, "finalize_clip",
                side_effect=((clip_a, skeleton_a, report),
                             (clip_b, skeleton_b, report))),
            mock.patch.object(builder, "combine_clips") as combine,
        ):
            with self.assertRaisesRegex(ValueError, "grail.*skeleton signature"):
                builder._assemble_candidate(args)
        combine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
