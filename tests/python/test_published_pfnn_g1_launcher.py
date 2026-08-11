from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from mm_sonic.published_pfnn_g1_launcher import (
    _wait_for_expected_world,
    build_launch_spec,
    main,
    process_identity,
    stop_recorded_process,
)
from mm_sonic.published_pfnn_g1_scenes import load_scenes


class PublishedPFNNG1LauncherTests(unittest.TestCase):
    def test_scene_six_commands_bind_world_five_and_matching_terrain(self) -> None:
        scene = load_scenes()[6]
        terrain = Path("/cache/hmap_urban_001_smooth-deadbeef.npz")
        spec = build_launch_spec(
            scene=scene,
            python=Path("/venv/python"),
            exporter=Path("/cache/pfnn_export"),
            fifo=Path("/run/frames.fifo"),
            terrain=terrain,
            scene_xml=Path("/gear/scene_29dof.xml"),
            gmr_root=Path("/gmr"),
        )

        self.assertEqual(spec.terrain_path, terrain)
        self.assertEqual(spec.exporter_command[-4:], ("--world", "5", "--export", "/run/frames.fifo"))
        self.assertIn("--expected-world", spec.viewer_command)
        expected_index = spec.viewer_command.index("--expected-world")
        self.assertEqual(spec.viewer_command[expected_index + 1], "5")

    def test_stop_requires_exact_process_identity(self) -> None:
        process = subprocess.Popen(["sleep", "60"])
        identity = process_identity(process.pid)
        try:
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                stop_recorded_process(replace(identity, start_ticks=identity.start_ticks + 1))
            self.assertIsNone(process.poll())
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                stop_recorded_process(replace(identity, command_sha256="0" * 64))
            self.assertIsNone(process.poll())
            self.assertTrue(stop_recorded_process(identity, timeout_seconds=2.0))
            process.wait(timeout=1.0)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=1.0)

    def test_switch_dry_run_does_not_touch_runtime_or_cache(self) -> None:
        identity = process_identity(os.getpid())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            receipt = {
                "schema": "published-pfnn-g1-live/v1",
                "processes": {
                    name: asdict(identity) for name in ("exporter", "bridge", "viewer")
                },
            }
            active = runtime / "active.json"
            active.write_text(json.dumps(receipt))
            original = active.read_bytes()

            with redirect_stdout(io.StringIO()):
                main(
                    [
                        "switch",
                        "--dry-run",
                        "--scene",
                        "6",
                        "--runtime-root",
                        str(runtime),
                        "--cache-root",
                        str(root / "cache"),
                    ]
                )

            self.assertEqual(active.read_bytes(), original)
            self.assertFalse((root / "cache").exists())
            self.assertFalse((runtime / "runs").exists())

    def test_status_rejects_receipt_without_exact_process_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "active.json").write_text(
                json.dumps(
                    {"schema": "published-pfnn-g1-live/v1", "processes": {}}
                )
            )
            with self.assertRaisesRegex(ValueError, "exact process identities"):
                main(["status", "--runtime-root", str(runtime)])

    def test_startup_waits_for_matching_exported_world(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "viewer.log"
            log.write_text(
                "frame_count=1 root=(0,0,0) frame=9 world=5 phase=0.1\n"
            )
            _wait_for_expected_world(log, 5, (), timeout_seconds=0.1)
            with self.assertRaisesRegex(RuntimeError, "did not report world 4"):
                _wait_for_expected_world(log, 4, (), timeout_seconds=0.01)


if __name__ == "__main__":
    unittest.main()
