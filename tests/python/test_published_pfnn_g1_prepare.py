from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np


RELEASED_DEMO = Path("/home/ubuntu/datasets/pfnn/pfnn/demo")
RESOURCES = Path("sonic/resources/published_pfnn_g1").resolve()
RELEASED_CPP_SHA256 = (
    "6deb74a9af58874e2b9c88cca760f9aa1c9e6d4db10fa10ae02b39609e3d3154"
)


class PublishedPFNNG1PrepareTests(unittest.TestCase):
    def test_rejects_tampered_released_source_before_preparation(self) -> None:
        from mm_sonic.published_pfnn_g1_prepare import prepare_exporter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tampered_demo = root / "source"
            tampered_demo.mkdir()
            (tampered_demo / "pfnn.cpp").write_text("// not the release\n")
            with self.assertRaisesRegex(ValueError, "released pfnn.cpp SHA"):
                prepare_exporter(tampered_demo, root / "cache", RESOURCES)

    def test_checked_in_patch_applies_to_the_authenticated_release(self) -> None:
        self.assertEqual(
            hashlib.sha256((RELEASED_DEMO / "pfnn.cpp").read_bytes()).hexdigest(),
            RELEASED_CPP_SHA256,
        )
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "demo"
            copy.mkdir()
            shutil.copy2(RELEASED_DEMO / "pfnn.cpp", copy / "pfnn.cpp")
            subprocess.run(
                ["patch", "-p1", "--input", str(RESOURCES / "pfnn_export.patch")],
                cwd=copy,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            patched = (copy / "pfnn.cpp").read_text()

        self.assertIn("PFNNXFM", patched)
        self.assertIn("--export", patched)
        self.assertIn("--world", patched)

    def test_authenticated_exporter_is_copied_patched_and_compiled_directly(self) -> None:
        from mm_sonic import published_pfnn_g1_prepare as prepare

        real_run = subprocess.run
        compile_command = [
            "g++", "-std=gnu++11", "-Wall", "-O3", "-ffast-math",
            "pfnn.cpp", "-lGL", "-lGLEW", "-lSDL2", "-g", "-o",
            "pfnn_export",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            shutil.copy2(RELEASED_DEMO / "pfnn.cpp", source / "pfnn.cpp")

            def run(command, **kwargs):
                if command[0] == "patch":
                    return real_run(command, **kwargs)
                self.assertEqual(command, compile_command)
                self.assertEqual(Path(kwargs["cwd"]), root / "cache")
                self.assertTrue(kwargs["check"])
                (root / "cache" / "pfnn_export").touch()
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(prepare.subprocess, "run", side_effect=run):
                executable = prepare.prepare_exporter(source, root / "cache", RESOURCES)

            self.assertEqual(executable, root / "cache" / "pfnn_export")
            patched = (root / "cache" / "pfnn.cpp").read_text()
            self.assertIn("PFNNXFM", patched)
            self.assertIn("--world", patched)

    def test_prepares_scene_six_terrain_as_cached_triangle_mesh(self) -> None:
        from mm_sonic.published_pfnn_g1_prepare import prepare_terrain
        from mm_sonic.published_pfnn_g1_scenes import load_scenes

        with tempfile.TemporaryDirectory() as directory:
            mesh = prepare_terrain(
                load_scenes()[6], RELEASED_DEMO, Path(directory)
            )
            repeated = prepare_terrain(
                load_scenes()[6], RELEASED_DEMO, Path(directory)
            )
            with np.load(mesh) as archive:
                self.assertEqual(archive["faces"].shape[1], 3)
                self.assertEqual(archive["vertices"].shape[1], 3)

        self.assertEqual(repeated, mesh)


if __name__ == "__main__":
    unittest.main()
