from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from jsonschema import Draft202012Validator

from mm_sonic.artifacts import RunBundle, verify_run_inventory
from mm_sonic.coordinator import (
    AbortDecisionRecord,
    AcceptedChunk,
    DeliveryAudit,
    PreflightResult,
    RejectionRecord,
    TerminalVerdict,
    TimingRecord,
)
from mm_sonic.joints import ContractError
from mm_sonic.process import AdvanceResult


def scene_registration_metadata() -> dict[str, object]:
    return {
        "scene_id": "sonic-flat-baseline",
        "route_id": "flat-12s",
        "source_kind": "analytic-flat",
        "source_hashes": {
            "registry": "d" * 64,
            "manifest": "e" * 64,
            "scene_index": "f" * 64,
        },
        "coordinate_source": "holden-y-up-right-handed-forward-plus-z",
        "coordinate_target": "mujoco-z-up-right-handed-forward-plus-x",
        "transform_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        "output_hashes": {
            "gear_scene_xml": "1" * 64,
            "scene_registration": "2" * 64,
        },
        "allowed_foot_geoms": [11, 12],
        "forbidden_geom_groups": {
            "pelvis": [1],
            "knees": [2, 3],
            "torso": [4, 5, 6],
            "hands": [7, 8, 9, 10],
        },
    }


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
        "scene_registration": scene_registration_metadata(),
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

    def _finalized_bundle(self, experiment_id: str) -> RunBundle:
        bundle = RunBundle.create(self.root, experiment_id, "run")
        bundle.update_manifest(terminal_metadata())
        bundle.write_text("logs/evidence.txt", "registered\n")
        bundle.finalize("complete", outcome={"integration_pass": True})
        return bundle

    @staticmethod
    def _rewrite_manifest(bundle: RunBundle, mutate) -> None:
        path = bundle.path / "manifest.json"
        path.chmod(0o644)
        manifest = json.loads(path.read_text("utf-8"))
        mutate(manifest)
        path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )

    @staticmethod
    def _rewrite_inventory(bundle: RunBundle, mutate) -> None:
        path = bundle.path / "inventory.json"
        path.chmod(0o644)
        inventory = json.loads(path.read_text("utf-8"))
        mutate(inventory)
        path.write_text(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )

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

    def test_coordinator_records_are_structured_and_terminal_verdict_is_exclusive(self):
        readiness = PreflightResult(
            session_id="s0",
            readiness_log_start_offset=0,
            scoring_log_offset=321,
            readiness_attempts=2,
            official_log_path="/run/target.csv",
            log_device=11,
            log_inode=22,
            expected_row_sha256="0" * 64,
        )
        self.bundle.write_readiness(readiness)
        source = SimpleNamespace(
            session_id="s0",
            candidate_id="s0:candidate:000000",
            predecessor_id=None,
            timestamps_s=[index / 25.0 for index in range(11)],
        )
        target = SimpleNamespace(
            accepted_chunk_id="s0:target:1-20",
            source_candidate_id=source.candidate_id,
            frame_index=list(range(1, 21)),
            hashes={
                "canonical_target_sha256": "1" * 64,
                "diagnostic_sha256": "2" * 64,
            },
        )
        self.bundle.write_prepared(source, target)
        rejection = RejectionRecord(
            candidate_id=source.candidate_id,
            failure_site="source_validation",
            error_type="ContractError",
            error_message="bad source",
            mm_alive=True,
            abort_required=True,
            abort_sent=False,
            timeline_abort_required=False,
        )
        self.bundle.write_rejection(rejection)
        self.bundle.write_abort_decision(
            AbortDecisionRecord(
                candidate_id=source.candidate_id,
                mm_abort_required=True,
                mm_abort_sent=True,
                timeline_abort_required=False,
                timeline_aborted=False,
                errors=(),
            )
        )
        timing = TimingRecord(
            candidate_id=source.candidate_id,
            durations_ns={
                "mm_generation": 11,
                "projection_validation": 13,
                "resampling": None,
                "artifact_enqueue": None,
                "publication": None,
                "simulation_advance": None,
            },
            failed_stage="source_validation",
            wall_started_ns=100,
            wall_finished_ns=200,
        )
        self.bundle.write_timing(timing)

        accepted_target = SimpleNamespace(
            accepted_chunk_id="s0:target:1-20",
            source_candidate_id=source.candidate_id,
            buffer=SimpleNamespace(frame_index=list(range(1, 21))),
            hashes=target.hashes,
        )
        accepted = AcceptedChunk(
            target=accepted_target,
            advance=AdvanceResult(80, 0.0, 0.4, 20, 80),
            timings_ns={
                "mm_generation": 1,
                "projection_validation": 2,
                "resampling": 3,
                "artifact_enqueue": 4,
                "publication": 5,
                "simulation_advance": 6,
            },
            wall_started_ns=300,
            wall_finished_ns=400,
        )
        self.bundle.write_accepted(accepted)

        verdict = TerminalVerdict(
            status="failed",
            failure_phase="pre_commit",
            failure_site="source_validation",
            error_type="ContractError",
            error_message="bad source",
            accepted_chunks=0,
            simulation_paused=True,
            delivery_audit_required=True,
            delivery_audit_completed=False,
            delivery_audit=None,
        )
        self.bundle.write_terminal_verdict(verdict)

        candidate = json.loads(
            (self.bundle.path / "candidates.jsonl").read_text("ascii").splitlines()[0]
        )["record"]
        self.assertEqual(candidate["candidate_id"], source.candidate_id)
        self.assertEqual(candidate["target_frames"], [1, 20])
        persisted_readiness = json.loads(
            (self.bundle.path / "readiness.jsonl").read_text("ascii").splitlines()[0]
        )["record"]
        self.assertEqual(persisted_readiness["session_id"], "s0")
        self.assertEqual(persisted_readiness["scoring_log_offset"], 321)
        self.assertEqual(persisted_readiness["log_identity"], [11, 22])
        abort = json.loads(
            (self.bundle.path / "aborts.jsonl").read_text("ascii").splitlines()[0]
        )["record"]
        self.assertTrue(abort["mm_abort_sent"])
        terminal_path = self.bundle.path / "terminal-verdict.json"
        terminal = json.loads(terminal_path.read_text("ascii"))
        self.assertEqual(terminal["schema"], "mm-sonic-terminal-verdict/v1")
        self.assertTrue(terminal["delivery_audit_required"])
        self.assertFalse(terminal["delivery_audit_completed"])
        self.assertNotIn("ack", terminal_path.read_text("ascii").lower())
        with self.assertRaisesRegex(ContractError, "already exists"):
            self.bundle.write_terminal_verdict(verdict)

    def test_prepared_record_rejects_nonintegral_and_noncontiguous_frames(self):
        source = SimpleNamespace(
            session_id="s0",
            candidate_id="c0",
            predecessor_id=None,
            timestamps_s=[index / 25.0 for index in range(11)],
        )
        for frames in ([1.5, 2.5], [1, 3], [2, 1], [True, 1]):
            target = SimpleNamespace(
                accepted_chunk_id="target",
                source_candidate_id="c0",
                frame_index=frames,
                hashes={},
            )
            with self.subTest(frames=frames):
                with self.assertRaisesRegex(ContractError, "frame indices"):
                    self.bundle.write_prepared(source, target)
        self.assertFalse((self.bundle.path / "candidates.jsonl").exists())

    def test_incremental_records_reject_hard_linked_external_targets(self):
        victim = self.root / "external-candidates.jsonl"
        victim.write_bytes(b'{"external":true}\n')
        os.link(victim, self.bundle.path / "candidates.jsonl")

        with self.assertRaisesRegex(ContractError, "hard link"):
            self.bundle.record_candidate({"candidate_id": "must-not-append"})
        self.assertEqual(victim.read_bytes(), b'{"external":true}\n')
        self.assertEqual(self.bundle.status, "running")

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

    def test_inventory_authenticates_terminal_status_and_outcome(self):
        cases = (
            ("status", lambda value: value.update({"status": "failed"})),
            (
                "outcome",
                lambda value: value.update(
                    {"outcome": {"integration_pass": False, "tampered": True}}
                ),
            ),
        )
        for index, (label, mutate) in enumerate(cases):
            with self.subTest(label=label):
                bundle = self._finalized_bundle(f"manifest-tamper-{index}")
                self._rewrite_manifest(bundle, mutate)
                with self.assertRaisesRegex(
                    ContractError, "terminal manifest core SHA-256"
                ):
                    verify_run_inventory(bundle.path)

    def test_inventory_rejects_nonterminal_manifest_status(self):
        bundle = self._finalized_bundle("nonterminal-status")
        self._rewrite_manifest(
            bundle, lambda value: value.update({"status": "running"})
        )
        with self.assertRaisesRegex(ContractError, "terminal manifest status"):
            verify_run_inventory(bundle.path)

    def test_inventory_rejects_unsupported_manifest_and_inventory_schemas(self):
        manifest_bundle = self._finalized_bundle("manifest-schema")
        self._rewrite_manifest(
            manifest_bundle,
            lambda value: value.update({"schema": "unsupported-manifest"}),
        )
        with self.assertRaisesRegex(ContractError, "manifest.*unsupported schema"):
            verify_run_inventory(manifest_bundle.path)

        inventory_bundle = self._finalized_bundle("inventory-schema")
        self._rewrite_inventory(
            inventory_bundle,
            lambda value: value.update({"schema": "unsupported-inventory"}),
        )
        with self.assertRaisesRegex(ContractError, "inventory.*unsupported schema"):
            verify_run_inventory(inventory_bundle.path)

    def test_inventory_validates_declared_file_count(self):
        bundle = self._finalized_bundle("file-count-tamper")

        def mutate(value):
            value["evidence"]["file_count"] += 1

        self._rewrite_manifest(bundle, mutate)
        with self.assertRaisesRegex(ContractError, "file_count"):
            verify_run_inventory(bundle.path)

    def test_inventory_rejects_fifo_and_unregistered_directory_entries(self):
        fifo_bundle = self._finalized_bundle("fifo-entry")
        os.mkfifo(fifo_bundle.path / "unexpected.fifo")
        with self.assertRaisesRegex(ContractError, "special entry"):
            verify_run_inventory(fifo_bundle.path)

        directory_bundle = self._finalized_bundle("directory-entry")
        (directory_bundle.path / "unexpected-empty-directory").mkdir()
        with self.assertRaisesRegex(ContractError, "unregistered director"):
            verify_run_inventory(directory_bundle.path)

    def test_finalize_rejects_replaced_run_path_without_touching_symlink_target(self):
        self.bundle.update_manifest(terminal_metadata())
        self.bundle.write_text("evidence.txt", "registered\n")
        victim = self.root / "input-tree"
        victim.mkdir()
        victim_file = victim / "input.bin"
        victim_file.write_bytes(b"read-only input identity\n")
        victim_file.chmod(0o666)
        expected_mode = stat.S_IMODE(victim_file.stat().st_mode)

        original = self.bundle.path
        moved = original.with_name("run-moved")
        original.rename(moved)
        original.symlink_to(victim, target_is_directory=True)

        with self.assertRaisesRegex(ContractError, "run path identity"):
            self.bundle.finalize("complete", outcome={"integration_pass": True})
        self.assertEqual(stat.S_IMODE(victim_file.stat().st_mode), expected_mode)
        self.assertEqual(victim_file.read_bytes(), b"read-only input identity\n")
        self.assertFalse((victim / "inventory.json").exists())

    def test_finalize_rejects_hard_link_alias_without_changing_input_inode(self):
        self.bundle.update_manifest(terminal_metadata())
        victim = self.root / "input.bin"
        victim.write_bytes(b"external input\n")
        victim.chmod(0o666)
        expected_mode = stat.S_IMODE(victim.stat().st_mode)
        os.link(victim, self.bundle.path / "hard-linked-input.bin")

        with self.assertRaisesRegex(ContractError, "hard link"):
            self.bundle.finalize("failed", outcome={"failure_layer": "fixture"})
        self.assertEqual(stat.S_IMODE(victim.stat().st_mode), expected_mode)
        self.assertEqual(victim.read_bytes(), b"external input\n")

    def test_finalize_rejects_unregistered_empty_directories(self):
        self.bundle.update_manifest(terminal_metadata())
        (self.bundle.path / "unexpected-empty-directory").mkdir()

        with self.assertRaisesRegex(ContractError, "unregistered director"):
            self.bundle.finalize("failed", outcome={"reason": "fixture"})
        self.assertEqual(self.bundle.status, "finalizing")

    def test_sealing_failure_is_permanently_fail_closed_on_disk_and_in_api(self):
        self.bundle.update_manifest(terminal_metadata())
        self.bundle.write_text("evidence.txt", "registered\n")

        with mock.patch.object(
            self.bundle,
            "_seal_evidence",
            side_effect=OSError("injected sealing failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected sealing failure"):
                self.bundle.finalize(
                    "complete", outcome={"integration_pass": True}
                )

        manifest = json.loads(
            (self.bundle.path / "manifest.json").read_text("utf-8")
        )
        self.assertEqual(self.bundle.status, "finalizing")
        self.assertEqual(manifest["status"], "finalizing")
        self.assertNotIn("outcome", manifest)
        self.assertNotIn("evidence", manifest)
        self.assertNotIn("finalized_utc", manifest)
        with self.assertRaisesRegex(ContractError, "finalizing"):
            self.bundle.write_text("late.txt", "forbidden\n")
        with self.assertRaisesRegex(ContractError, "finalizing"):
            self.bundle.record_timing({"name": "late", "duration_ns": 1})
        with self.assertRaisesRegex(ContractError, "finalizing"):
            self.bundle.finalize("failed", outcome={"reason": "retry"})
        self.assertFalse((self.bundle.path / "late.txt").exists())

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

    def test_finalize_rejects_every_malformed_manifest_category_before_transition(self):
        cases = (
            (
                "external",
                lambda value: value["external"].update(
                    {"gear_dirty": "not-a-boolean"}
                ),
            ),
            (
                "repositories",
                lambda value: value["repositories"]["motion_matching"].update(
                    {"commit": "short"}
                ),
            ),
            (
                "command_script",
                lambda value: value["command_script"].update(
                    {"sha256": "not-a-sha256"}
                ),
            ),
            (
                "perturbation",
                lambda value: value["perturbation"].update(
                    {"lateral_offset_m": "not-a-number"}
                ),
            ),
            (
                "coordinate_transform",
                lambda value: value["coordinate_transform"].update(
                    {"matrix": [[1.0, 0.0], [0.0, 1.0]]}
                ),
            ),
            (
                "processes",
                lambda value: value.update(
                    {"processes": [{"name": "mm", "argv": []}]}
                ),
            ),
            (
                "processes",
                lambda value: value.update({"processes": []}),
            ),
        )
        for index, (label, mutate) in enumerate(cases):
            with self.subTest(label=label):
                metadata = terminal_metadata()
                mutate(metadata)
                bundle = RunBundle.create(
                    self.root, f"malformed-{index}", "run"
                )
                bundle.update_manifest(metadata)
                with self.assertRaisesRegex(ContractError, label):
                    bundle.finalize("failed", outcome={"reason": "fixture"})
                self.assertEqual(bundle.status, "running")
                manifest = json.loads(
                    (bundle.path / "manifest.json").read_text("utf-8")
                )
                self.assertEqual(manifest["status"], "running")

    def test_complete_requires_every_mandatory_artifact_hash_but_not_encoder(self):
        mandatory = (
            "policy",
            "observation_config",
            "model",
            "source_mjcf",
            "motion",
            "terrain",
            "joint_map",
            "scene",
        )
        for index, name in enumerate(mandatory):
            with self.subTest(name=name):
                metadata = terminal_metadata()
                metadata["artifact_hashes"] = dict(metadata["artifact_hashes"])
                metadata["artifact_hashes"][name] = None
                bundle = RunBundle.create(
                    self.root, f"complete-null-{index}", "run"
                )
                bundle.update_manifest(metadata)
                with self.assertRaisesRegex(
                    ContractError, rf"complete.*{name}"
                ):
                    bundle.finalize("complete", outcome={"integration_pass": True})
                self.assertEqual(bundle.status, "running")

        encoder_optional = RunBundle.create(
            self.root, "complete-encoder-null", "run"
        )
        encoder_optional.update_manifest(terminal_metadata())
        encoder_optional.finalize(
            "complete", outcome={"integration_pass": True}
        )
        self.assertEqual(encoder_optional.status, "complete")

    def test_failed_and_not_run_can_record_unavailable_artifact_hashes_as_null(self):
        for index, terminal in enumerate(("failed", "not_run")):
            with self.subTest(terminal=terminal):
                metadata = terminal_metadata()
                metadata["artifact_hashes"] = {
                    key: None for key in metadata["artifact_hashes"]
                }
                bundle = RunBundle.create(
                    self.root, f"unavailable-{index}", "run"
                )
                bundle.update_manifest(metadata)
                bundle.finalize(terminal, outcome={"reason": "unavailable"})
                self.assertEqual(bundle.status, terminal)

    def test_scene_registration_is_complete_only_and_always_validated(self):
        missing = terminal_metadata()
        missing.pop("scene_registration")
        complete = RunBundle.create(
            self.root, "missing-complete-scene", "run"
        )
        complete.update_manifest(missing)
        with self.assertRaisesRegex(
            ContractError, "complete.*scene_registration"
        ):
            complete.finalize("complete", outcome={"integration_pass": True})

        for index, terminal in enumerate(("failed", "not_run")):
            with self.subTest(terminal=terminal):
                bundle = RunBundle.create(
                    self.root, f"missing-scene-{index}", "run"
                )
                bundle.update_manifest(missing)
                bundle.finalize(terminal, outcome={"reason": terminal})
                self.assertEqual(bundle.status, terminal)

        malformed = terminal_metadata()
        malformed["scene_registration"] = scene_registration_metadata()
        malformed_scene = malformed["scene_registration"]
        assert isinstance(malformed_scene, dict)
        forbidden = malformed_scene["forbidden_geom_groups"]
        assert isinstance(forbidden, dict)
        forbidden["pelvis"] = [11]
        failed = RunBundle.create(self.root, "malformed-scene", "run")
        failed.update_manifest(malformed)
        with self.assertRaisesRegex(
            ContractError, "scene_registration.*overlap"
        ):
            failed.finalize("failed", outcome={"reason": "fixture"})

        registered = terminal_metadata()
        registered_scene = registered["scene_registration"]
        assert isinstance(registered_scene, dict)
        registered_scene["terrain_geoms"] = [0]
        complete_with_terrain = RunBundle.create(
            self.root, "registered-terrain-geoms", "run"
        )
        complete_with_terrain.update_manifest(registered)
        complete_with_terrain.finalize(
            "complete", outcome={"integration_pass": True}
        )

        overlapping = terminal_metadata()
        overlapping_scene = overlapping["scene_registration"]
        assert isinstance(overlapping_scene, dict)
        overlapping_scene["terrain_geoms"] = [11]
        overlap = RunBundle.create(self.root, "terrain-overlap", "run")
        overlap.update_manifest(overlapping)
        with self.assertRaisesRegex(ContractError, "geom groups overlap"):
            overlap.finalize("complete", outcome={"integration_pass": True})


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

    def test_schema_has_a_complete_only_nonnull_artifact_rule(self):
        root = Path(__file__).resolve().parents[2]
        schema = json.loads(
            (root / "sonic" / "schemas" / "run_manifest_v1.schema.json").read_text("utf-8")
        )
        self.assertIn("finalizing", schema["properties"]["status"]["enum"])
        complete_rules = [
            rule
            for rule in schema["allOf"]
            if rule.get("if", {}).get("properties", {}).get("status")
            == {"const": "complete"}
        ]
        self.assertEqual(len(complete_rules), 1)
        self.assertEqual(
            complete_rules[0]["then"]["properties"]["artifact_hashes"]["$ref"],
            "#/$defs/completeArtifactHashes",
        )
        complete_hashes = schema["$defs"]["completeArtifactHashes"]
        for name in (
            "policy",
            "observation_config",
            "model",
            "source_mjcf",
            "motion",
            "terrain",
            "joint_map",
            "scene",
        ):
            self.assertEqual(
                complete_hashes["properties"][name], {"$ref": "#/$defs/sha256"}
            )
        self.assertEqual(
            schema["properties"]["processes"]["items"]["properties"]["argv"][
                "minItems"
            ],
            1,
        )
        self.assertEqual(schema["properties"]["processes"]["minItems"], 1)

    def test_schema_allows_unauthenticated_identity_only_for_not_run(self):
        root = Path(__file__).resolve().parents[2]
        schema = json.loads(
            (root / "sonic" / "schemas" / "run_manifest_v1.schema.json").read_text("utf-8")
        )
        validator = Draft202012Validator(schema)
        with tempfile.TemporaryDirectory() as temporary:
            bundle = RunBundle.create(temporary, "not-run-null", "run")
            metadata = terminal_metadata()
            metadata.pop("scene_registration")
            metadata["external"]["gear_commit"] = None
            metadata["external"]["gear_dirty"] = None
            metadata["external"]["hashes"] = {"policy": None}
            metadata["repositories"] = {
                "motion_matching": {"commit": None, "dirty": None},
                "gear_sonic": {"commit": None, "dirty": None},
            }
            bundle.update_manifest(metadata)
            bundle.finalize("not_run", outcome={"reason": "missing"})
            manifest = json.loads(
                (bundle.path / "manifest.json").read_text("ascii")
            )
            validator.validate(manifest)
            manifest["status"] = "failed"
            self.assertTrue(tuple(validator.iter_errors(manifest)))


if __name__ == "__main__":
    unittest.main()
