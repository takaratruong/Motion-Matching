import json
import os
import struct
import subprocess
import tempfile
import unittest

from resources.g1_terrain_builder.artifacts import read_terrain_sidecar
from resources.g1_terrain_builder.database import read_holden_database


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


if __name__ == "__main__":
    unittest.main()
