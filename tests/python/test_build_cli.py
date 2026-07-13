import json
import os
import struct
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from resources import build_g1_terrain_database as builder
from resources.g1_terrain_builder.artifacts import read_terrain_sidecar
from resources.g1_terrain_builder.database import read_holden_database
from resources.g1_terrain_builder.schema import HoldenClip, SkeletonSpec


PYTHON = "/home/ubuntu/miniconda3/envs/diffsim/bin/python"


class BuildCliTests(unittest.TestCase):
    def test_one_grail_clip_builds_and_validates(self):
        with tempfile.TemporaryDirectory() as td:
            output = os.path.join(td, "g1_terrain")
            built = subprocess.run([
                PYTHON, "resources/build_g1_terrain_database.py",
                "--output", output,
                "--grail-limit", "1",
            ], check=True, text=True, capture_output=True)
            validated = subprocess.run([
                PYTHON, "resources/validate_g1_terrain_database.py",
                output,
            ], check=True, text=True, capture_output=True)
            manifest_path = os.path.join(output, "manifest.json")
            with open(manifest_path, encoding="utf-8") as stream:
                manifest = json.load(stream)
            database = read_holden_database(os.path.join(output, "database.bin"))
            terrain_features = read_terrain_sidecar(os.path.join(
                output, "terrain_features.bin"))
            with open(os.path.join(output, "terrain.bin"), "rb") as stream:
                terrain_header = struct.unpack("<4sIII4f", stream.read(32))
            corruption_results = []
            corruptions = (
                ("output_fps", lambda m: m.__setitem__("output_fps", 60.0),
                 "output_fps"),
                ("source_map", lambda m: m["sources"][0]["source_frame_map"].pop(),
                 "source frame map"),
                ("skeleton_parents",
                 lambda m: m["skeleton"]["parents"].__setitem__(1, -1),
                 "skeleton parents"),
            )
            for name, corrupt, message in corruptions:
                changed = json.loads(json.dumps(manifest))
                corrupt(changed)
                with open(manifest_path, "w", encoding="utf-8") as stream:
                    json.dump(changed, stream)
                result = subprocess.run([
                    PYTHON, "resources/validate_g1_terrain_database.py", output,
                ], text=True, capture_output=True)
                corruption_results.append((name, message, result))
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            artifact_corruptions = (
                ("database", "database.bin", lambda b: b[:-1], "database.bin"),
                ("terrain_features", "terrain_features.bin", lambda b: b[:-1],
                 "terrain_features.bin"),
                ("terrain_payload", "terrain.bin", lambda b: b + b"x",
                 "terrain.bin"),
                ("terrain_obj", "terrain.obj", lambda b: b + b"malformed\n",
                 "terrain.obj"),
                ("terrain_obj_nonfinite", "terrain.obj",
                 lambda b: b + b"v nan 0 0\n", "vertex must be finite"),
                ("terrain_obj_bad_face", "terrain.obj",
                 lambda b: b + b"f 1 2 9999999\n", "face index"),
                ("validation_json", "validation.json", lambda b: b"{}",
                 "validation.json"),
            )
            for name, filename, corrupt, message in artifact_corruptions:
                path = os.path.join(output, filename)
                with open(path, "rb") as stream:
                    original = stream.read()
                with open(path, "wb") as stream:
                    stream.write(corrupt(original))
                result = subprocess.run([
                    PYTHON, "resources/validate_g1_terrain_database.py", output,
                ], text=True, capture_output=True)
                corruption_results.append((name, message, result))
                with open(path, "wb") as stream:
                    stream.write(original)
            terrain_path = os.path.join(output, "terrain.bin")
            with open(terrain_path, "rb") as stream:
                terrain_payload = stream.read()
            coordinated_corruptions = (
                ("coordinated_cell_size", 24, "cell_size", 0.04,
                 "cell_size_m"),
                ("coordinated_exterior_height", 28, "exterior_height", 1.0,
                 "exterior height"),
            )
            for name, offset, metadata_key, value, message in (
                    coordinated_corruptions):
                changed_manifest = json.loads(json.dumps(manifest))
                changed_manifest["terrain"]["heightfield"][metadata_key] = value
                changed_payload = bytearray(terrain_payload)
                struct.pack_into("<f", changed_payload, offset, value)
                with open(manifest_path, "w", encoding="utf-8") as stream:
                    json.dump(changed_manifest, stream)
                with open(terrain_path, "wb") as stream:
                    stream.write(changed_payload)
                result = subprocess.run([
                    PYTHON, "resources/validate_g1_terrain_database.py", output,
                ], text=True, capture_output=True)
                corruption_results.append((name, message, result))
            shifted_manifest = json.loads(json.dumps(manifest))
            shifted_payload = bytearray(terrain_payload)
            shifted_origin_x = terrain_header[4] + 4.0
            shifted_manifest["terrain"]["heightfield"][
                "origin_x"] = shifted_origin_x
            struct.pack_into("<f", shifted_payload, 16, shifted_origin_x)
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump(shifted_manifest, stream)
            with open(terrain_path, "wb") as stream:
                stream.write(shifted_payload)
            result = subprocess.run([
                PYTHON, "resources/validate_g1_terrain_database.py", output,
            ], text=True, capture_output=True)
            corruption_results.append((
                "heightfield_obj_coverage",
                "terrain.bin domain does not cover terrain.obj XZ bounds",
                result,
            ))
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            with open(terrain_path, "wb") as stream:
                stream.write(terrain_payload)
        self.assertEqual(manifest["schema"], "g1-terrain-artifacts/v1")
        self.assertEqual(built.stderr, "")
        self.assertIn("BUILT g1-terrain-artifacts/v1", built.stdout)
        self.assertEqual(manifest["output_fps"], 25.0)
        self.assertEqual(manifest["feature_dimensions"], 31)
        self.assertEqual(manifest["terrain_dimensions"], 4)
        self.assertEqual(manifest["grail_clips"], 1)
        self.assertEqual(manifest["total_clips"], 2)
        self.assertEqual(manifest["skipped_clips"], 0)
        self.assertTrue(manifest["diagnostic_mode"])
        self.assertEqual(len(manifest["skeleton"]["names"]), 31)
        self.assertEqual(database.positions.shape[1], 31)
        self.assertEqual(len(database.positions), len(terrain_features))
        self.assertEqual(terrain_header[:2], (b"G1HF", 1))
        self.assertGreaterEqual(terrain_header[2], 2)
        self.assertGreaterEqual(terrain_header[3], 2)
        self.assertEqual(validated.stderr, "")
        self.assertEqual(
            validated.stdout,
            f"VALID g1-terrain-artifacts/v1 frames={len(database.positions)} "
            "clips=2 bones=31 terrain_dims=4\n",
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
        for name, message, result in corruption_results:
            with self.subTest(name=name):
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_invalid_build_arguments_fail_before_publication(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "missing")
            cases = (
                ("negative_limit", ["--grail-limit", "-1"], "non-negative"),
                ("empty_positive_glob", [
                    "--grail-glob", os.path.join(td, "none", "*.pkl"),
                    "--grail-limit", "1",
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
            runtime_terrain="terrain",
        )
        duplicate = SimpleNamespace(name="duplicate")
        with (
            mock.patch.object(builder, "_require_file"),
            mock.patch.object(builder.glob, "glob", return_value=["clip.pkl"]),
            mock.patch.object(builder, "G1Kinematics"),
            mock.patch.object(builder, "load_takara", return_value=duplicate),
            mock.patch.object(builder, "load_grail", return_value=duplicate),
            mock.patch.object(builder, "finalize_clip") as finalize,
        ):
            with self.assertRaisesRegex(ValueError, "duplicate source names"):
                builder.build_artifacts(args)
        finalize.assert_not_called()

    def test_skeleton_changes_are_rejected_before_combination(self):
        args = SimpleNamespace(
            output="unused", grail_glob="clips/*.pkl", grail_limit=1,
            g1_xml="g1.xml", takara="takara.npz", remap="remap.npy",
            runtime_terrain="terrain",
        )
        takara = SimpleNamespace(
            name="takara_walk_50hz", terrain_id="flat", fps=25.0,
            qpos=np.zeros((3, 36), np.float32),
        )
        grail = SimpleNamespace(
            name="grail", terrain_id="flat", fps=25.0,
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
        with (
            mock.patch.object(builder, "_require_file"),
            mock.patch.object(builder.glob, "glob", return_value=["clip.pkl"]),
            mock.patch.object(builder, "G1Kinematics"),
            mock.patch.object(builder, "load_takara", return_value=takara),
            mock.patch.object(builder, "load_grail", return_value=grail),
            mock.patch.object(
                builder, "finalize_clip",
                side_effect=((clip_a, skeleton_a, report),
                             (clip_b, skeleton_b, report))),
            mock.patch.object(builder, "combine_clips") as combine,
        ):
            with self.assertRaisesRegex(ValueError, "grail.*skeleton signature"):
                builder.build_artifacts(args)
        combine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
