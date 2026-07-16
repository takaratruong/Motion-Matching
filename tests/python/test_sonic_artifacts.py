from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from mm_sonic.artifacts import RunBundle, verify_run_inventory
from mm_sonic.joints import ContractError


def terminal_metadata() -> dict[str, object]:
    return {
        "external": {
            "gear_checkout": "/readonly/gear",
            "gear_commit": "1" * 40,
            "gear_dirty": False,
            "policy": "/readonly/policy.onnx",
            "observation_config": "/readonly/observation.yaml",
            "encoder": None,
            "terrain_dir": "/readonly/terrain",
            "source_mjcf": "/readonly/g1.xml",
            "hashes": {"gear:known_good_reference": "2" * 64},
        },
        "repositories": {
            "motion_matching": {"commit": "3" * 40, "dirty": False},
            "gear_sonic": {"commit": "1" * 40, "dirty": False},
        },
        "artifact_hashes": {
            "policy": "4" * 64,
            "encoder": None,
            "observation_config": "5" * 64,
            "model": "6" * 64,
            "source_mjcf": "7" * 64,
            "motion": "8" * 64,
            "terrain": "9" * 64,
            "joint_map": "a" * 64,
            "scene": "b" * 64,
        },
        "command_script": {"id": "flat-12s", "sha256": "c" * 64},
        "perturbation": {
            "id": "nominal",
            "lateral_offset_m": 0.0,
            "yaw_offset_rad": 0.0,
        },
        "coordinate_transform": {
            "source": "holden-y-up-right-handed-forward-plus-z",
            "target": "mujoco-z-up-right-handed-forward-plus-x",
            "matrix": [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        },
        "processes": [
            {"name": "mm", "argv": ["sonic/build/mm_chunk_server"]},
            {"name": "gear", "argv": ["g1_deploy", "--input-type", "zmq"]},
        ],
    }


class RunBundleCreationTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def test_create_is_exclusive_and_starts_with_a_fsynced_running_manifest(self):
        bundle = RunBundle.create(self.root, "stage-a", "run-0001")
        self.assertEqual(bundle.path, self.root.resolve() / "stage-a" / "run-0001")
        self.assertEqual(bundle.status, "running")
        manifest = json.loads((bundle.path / "manifest.json").read_text("utf-8"))
        self.assertEqual(
            set(manifest),
            {"schema", "experiment_id", "run_id", "status", "created_utc"},
        )
        self.assertEqual(manifest["schema"], "mm-sonic-run-manifest/v1")
        self.assertEqual(manifest["status"], "running")
        with self.assertRaisesRegex(ContractError, "already exists"):
            RunBundle.create(self.root, "stage-a", "run-0001")

    def test_ids_and_all_output_paths_are_confined_and_symlinks_are_rejected(self):
        for experiment_id, run_id in (
            ("../escape", "run"),
            ("stage", "../escape"),
            ("/absolute", "run"),
            ("stage", "/absolute"),
        ):
            with self.subTest(experiment_id=experiment_id, run_id=run_id):
                with self.assertRaisesRegex(ContractError, "identifier"):
                    RunBundle.create(self.root, experiment_id, run_id)

        bundle = RunBundle.create(self.root, "stage", "safe")
        outside = self.root / "outside"
        outside.mkdir()
        (bundle.path / "linked").symlink_to(outside, target_is_directory=True)
        for path in ("../outside.txt", "/tmp/outside.txt", "linked/escaped.txt"):
            with self.subTest(path=path), self.assertRaises(ContractError):
                bundle.write_text(path, "must not escape\n")
        self.assertFalse((outside / "escaped.txt").exists())

    def test_inputs_are_opened_read_only_and_cannot_be_output_targets(self):
        input_tree = self.root / "input-tree"
        input_tree.mkdir()
        source = input_tree / "source.bin"
        source.write_bytes(b"authenticated input\n")
        bundle = RunBundle.create(self.root / "runs", "stage", "run")

        with bundle.open_input(source) as stream:
            self.assertEqual(stream.read(), b"authenticated input\n")
            self.assertFalse(stream.writable())
        with self.assertRaisesRegex(ContractError, "relative"):
            bundle.write_bytes(source, b"corruption")
        self.assertEqual(source.read_bytes(), b"authenticated input\n")

        alias = self.root / "input-alias"
        alias.symlink_to(source)
        with self.assertRaisesRegex(ContractError, "symlink"):
            bundle.open_input(alias)


class RunBundleLifecycleTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.bundle = RunBundle.create(self.root, "experiment", "run")

    def tearDown(self):
        self._temporary.cleanup()

    def test_candidate_abort_accept_and_timing_records_are_incremental_jsonl(self):
        records = (
            (self.bundle.record_candidate, {"candidate_id": "c0", "raw": {"v": 1}}, "candidates.jsonl"),
            (self.bundle.record_abort, {"candidate_id": "c0", "reason": "validation"}, "aborts.jsonl"),
            (self.bundle.record_candidate, {"candidate_id": "c1", "raw": {"v": 1}}, "candidates.jsonl"),
            (self.bundle.record_accept, {"candidate_id": "c1", "frames": [1, 20]}, "accepts.jsonl"),
            (self.bundle.record_timing, {"name": "resample", "duration_ns": 1234}, "timings.jsonl"),
        )
        expected_counts: dict[str, int] = {}
        for writer, record, filename in records:
            writer(record)
            expected_counts[filename] = expected_counts.get(filename, 0) + 1
            raw = (self.bundle.path / filename).read_bytes()
            self.assertTrue(raw.endswith(b"\n"))
            decoded = [json.loads(line) for line in raw.splitlines()]
            self.assertEqual(len(decoded), expected_counts[filename])
            self.assertEqual(decoded[-1]["record"], record)
            self.assertEqual(decoded[-1]["sequence"], sum(expected_counts.values()))

    def test_terminal_statuses_write_inventory_and_make_all_evidence_read_only(self):
        for index, terminal in enumerate(("complete", "failed", "not_run")):
            with self.subTest(terminal=terminal):
                bundle = RunBundle.create(self.root, f"experiment-{index}", "run")
                bundle.update_manifest(terminal_metadata())
                bundle.record_candidate({"candidate_id": "c0"})
                bundle.write_text("logs/child.stderr", "diagnostic\n")
                inventory = bundle.finalize(
                    terminal,
                    outcome={"reason": terminal, "integration_pass": terminal == "complete"},
                )
                self.assertEqual(bundle.status, terminal)
                manifest = json.loads((bundle.path / "manifest.json").read_text("utf-8"))
                self.assertEqual(manifest["status"], terminal)
                self.assertEqual(manifest["outcome"]["reason"], terminal)
                self.assertEqual(manifest["evidence"]["inventory_sha256"], hashlib.sha256(
                    (bundle.path / "inventory.json").read_bytes()
                ).hexdigest())
                self.assertIn("logs/child.stderr", inventory["files"])
                self.assertTrue(verify_run_inventory(bundle.path))
                for relative in (*inventory["files"], "inventory.json", "manifest.json"):
                    mode = (bundle.path / relative).stat().st_mode
                    self.assertFalse(mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH), relative)
                with self.assertRaisesRegex(ContractError, "finalized"):
                    bundle.finalize(terminal, outcome={})
                with self.assertRaisesRegex(ContractError, "finalized"):
                    bundle.record_timing({"name": "late", "duration_ns": 1})

    def test_failed_run_preserves_partial_candidate_and_abort_evidence(self):
        self.bundle.update_manifest(terminal_metadata())
        self.bundle.record_candidate({"candidate_id": "c-partial", "source_rows": 11})
        self.bundle.record_abort({"candidate_id": "c-partial", "error": "bad quaternion"})
        inventory = self.bundle.finalize(
            "failed",
            outcome={"failure_layer": "validation", "error": "bad quaternion"},
        )
        self.assertIn("candidates.jsonl", inventory["files"])
        self.assertIn("aborts.jsonl", inventory["files"])
        self.assertIn(b"c-partial", (self.bundle.path / "candidates.jsonl").read_bytes())
        self.assertIn(b"bad quaternion", (self.bundle.path / "aborts.jsonl").read_bytes())

    def test_inventory_detects_any_post_finalization_edit(self):
        self.bundle.update_manifest(terminal_metadata())
        self.bundle.write_text("evidence.txt", "original\n")
        self.bundle.finalize("complete", outcome={"integration_pass": True})
        evidence = self.bundle.path / "evidence.txt"
        evidence.chmod(0o644)
        evidence.write_text("edited\n", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "SHA-256"):
            verify_run_inventory(self.bundle.path)

    def test_inventory_detects_unregistered_evidence_added_after_finalization(self):
        self.bundle.update_manifest(terminal_metadata())
        self.bundle.write_text("evidence.txt", "registered\n")
        self.bundle.finalize("complete", outcome={"integration_pass": True})
        (self.bundle.path / "unregistered.txt").write_text(
            "not inventoried\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ContractError, "unregistered"):
            verify_run_inventory(self.bundle.path)

    def test_finalize_requires_complete_identity_metadata_and_valid_status(self):
        with self.assertRaisesRegex(ContractError, "terminal status"):
            self.bundle.finalize("running", outcome={})
        with self.assertRaisesRegex(ContractError, "manifest.*missing"):
            self.bundle.finalize("failed", outcome={"reason": "fixture"})
        self.bundle.update_manifest(terminal_metadata())
        with self.assertRaisesRegex(ContractError, "JSON"):
            self.bundle.record_timing({"name": "bad", "duration_ns": float("nan")})

        invalid = terminal_metadata()
        invalid["artifact_hashes"] = dict(invalid["artifact_hashes"])
        invalid["artifact_hashes"]["policy"] = "not-a-sha256"
        other = RunBundle.create(self.root, "experiment-invalid", "run")
        other.update_manifest(invalid)
        with self.assertRaisesRegex(ContractError, "artifact_hashes"):
            other.finalize("failed", outcome={"reason": "fixture"})


class RunManifestSchemaTests(unittest.TestCase):
    def test_terminal_schema_registers_every_reproducibility_category(self):
        root = Path(__file__).resolve().parents[2]
        schema = json.loads(
            (root / "sonic" / "schemas" / "run_manifest_v1.schema.json").read_text("utf-8")
        )
        terminal_required = set(schema["$defs"]["terminalManifest"]["required"])
        self.assertTrue(
            {
                "external",
                "repositories",
                "artifact_hashes",
                "command_script",
                "perturbation",
                "coordinate_transform",
                "processes",
                "outcome",
                "evidence",
                "finalized_utc",
            }.issubset(terminal_required)
        )


if __name__ == "__main__":
    unittest.main()
