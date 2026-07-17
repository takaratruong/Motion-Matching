from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mm_sonic import cli as cli_module
from mm_sonic import metrics as metrics_module
from mm_sonic.artifacts import RunBundle, verify_run_inventory
from mm_sonic.cli import GateResult, StageAContext, StageARequest, main
from mm_sonic.commands import CommandSample, command_script_bytes, flat_command_script
from mm_sonic.coordinator import Coordinator, SessionConfig, expected_official_target_row
from mm_sonic.external import ExternalInputs
from mm_sonic.joints import ContractError
from mm_sonic.metrics import validate_stage_a_prerequisite
from mm_sonic.process import GearProcess, ProcessError, _RemoteMMError
from mm_sonic.schema import parse_joint_feasibility_identity
from mm_sonic.timeline import CanonicalTargetBuffer
from mm_sonic.zmq_v1 import PosePublisher, encode_pose_v1


GATES = (
    "external_identity",
    "joint_projection_round_trip",
    "basis_and_scene_alignment",
    "flat_mm_kinematic_replay",
    "known_good_file_dynamic",
    "known_good_stream_delivery",
    "known_good_stream_dynamic",
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _joint_feasibility_identity(*, mask_sha256: str | None = None):
    violations = [0] * 29
    for index, count in {
        4: 20,
        5: 938,
        9: 4,
        10: 14,
        11: 52,
        14: 12,
        18: 13,
        25: 10,
    }.items():
        violations[index] = count
    return {
        "schema": "g1-joint-feasibility-certificate/v1",
        "frame_count": 459682,
        "raw_safe_count": 458619,
        "raw_unsafe_count": 1063,
        "search_safe_count": 458000,
        "joint_limit_violation_count": violations,
        "mask_sha256": "a" * 64 if mask_sha256 is None else mask_sha256,
    }


def _joint_feasibility_hello(identity=None):
    source = _joint_feasibility_identity() if identity is None else identity
    return {
        "joint_feasibility": {
            **source,
            "joint_limit_violation_count": list(
                source["joint_limit_violation_count"]
            ),
        }
    }


def _canonical(count: int = 21) -> CanonicalTargetBuffer:
    joint_position = np.zeros((count, 29), dtype="<f4")
    joint_position[:, 0] = np.arange(count, dtype=np.float32) / 10.0
    joint_velocity = np.zeros((count, 29), dtype="<f4")
    body_quat_w = np.zeros((count, 4), dtype="<f4")
    body_quat_w[:, 0] = 1.0
    return CanonicalTargetBuffer(
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        body_quat_w=body_quat_w,
        frame_index=np.arange(count, dtype="<i8"),
    )


def _write_pinned_known_good_fixture(root: Path) -> dict[str, np.ndarray]:
    """Write the resolved 455-frame/14-body layout declared by pinned info."""

    frames = 455
    bodies = 14
    frame = np.arange(frames, dtype=np.float32)[:, None]
    joint = np.arange(29, dtype=np.float32)[None, :]
    body = np.arange(bodies, dtype=np.float32)[None, :, None]
    xyz = np.arange(3, dtype=np.float32)[None, None, :]
    joint_position = np.ascontiguousarray(frame / 1000.0 + joint / 100.0, dtype="<f4")
    joint_velocity = np.ascontiguousarray(-frame / 2000.0 + joint / 200.0, dtype="<f4")
    body_position = np.ascontiguousarray(
        frame[:, None, :] / 1000.0 + body / 10.0 + xyz / 100.0,
        dtype="<f4",
    )
    body_quaternion = np.zeros((frames, bodies, 4), dtype="<f4")
    body_quaternion[..., 0] = np.float32(1.0)
    body_quaternion[..., 3] = np.asarray(body[..., 0] / 100.0, dtype=np.float32)
    body_quaternion /= np.linalg.norm(
        body_quaternion.astype(np.float64), axis=2, keepdims=True
    ).astype(np.float32)
    body_linear_velocity = np.ascontiguousarray(
        frame[:, None, :] / 3000.0 + body / 30.0 + xyz / 300.0,
        dtype="<f4",
    )
    body_angular_velocity = np.ascontiguousarray(
        -frame[:, None, :] / 4000.0 + body / 40.0 + xyz / 400.0,
        dtype="<f4",
    )

    def write_csv(name: str, values: np.ndarray, prefix: str) -> None:
        flattened = np.asarray(values, dtype="<f4").reshape(frames, -1)
        header = ",".join(f"{prefix}_{index}" for index in range(flattened.shape[1]))
        rows = [
            ",".join(format(float(value), ".9g") for value in row)
            for row in flattened
        ]
        (root / name).write_text(header + "\n" + "\n".join(rows) + "\n", encoding="ascii")

    write_csv("joint_pos.csv", joint_position, "joint")
    write_csv("joint_vel.csv", joint_velocity, "joint_vel")
    write_csv("body_pos.csv", body_position, "body_pos")
    write_csv("body_quat.csv", body_quaternion, "body_quat")
    write_csv("body_lin_vel.csv", body_linear_velocity, "body_lin_vel")
    write_csv("body_ang_vel.csv", body_angular_velocity, "body_ang_vel")
    (root / "info.txt").write_text(
        "Motion Information: walking_quip_360_R_002__A428\n"
        "==================================================\n\n"
        "joint_pos:\n  Shape: (455, 29)\n  Dtype: float32\n\n"
        "joint_vel:\n  Shape: (455, 29)\n  Dtype: float32\n\n"
        "body_pos_w:\n  Shape: (455, 14, 3)\n  Dtype: float32\n\n"
        "body_quat_w:\n  Shape: (455, 14, 4)\n  Dtype: float32\n\n"
        "body_lin_vel_w:\n  Shape: (455, 14, 3)\n  Dtype: float32\n\n"
        "body_ang_vel_w:\n  Shape: (455, 14, 3)\n  Dtype: float32\n\n"
        "_body_indexes:\n  Shape: (14,)\n  Dtype: int64\n\n"
        "time_step_total:\n  Shape: ()\n  Dtype: int64\n  Sample: [455]\n",
        encoding="ascii",
    )
    (root / "metadata.txt").write_text(
        "Metadata for: walking_quip_360_R_002__A428\n"
        "==============================\n\n"
        "Body part indexes:\n"
        "[ 0  4 10 18  5 11 19  9 16 22 28 17 23 29]\n\n"
        "Total timesteps: 455\n\n"
        "Data arrays summary:\n"
        "  joint_pos: (455, 29) (float32)\n"
        "  joint_vel: (455, 29) (float32)\n"
        "  body_pos_w: (455, 14, 3) (float32)\n"
        "  body_quat_w: (455, 14, 4) (float32)\n"
        "  body_lin_vel_w: (455, 14, 3) (float32)\n"
        "  body_ang_vel_w: (455, 14, 3) (float32)\n"
        "  _body_indexes: (14,) (int64)\n"
        "  time_step_total: () (int64)\n",
        encoding="ascii",
    )
    return {
        "joint_position": joint_position,
        "joint_velocity": joint_velocity,
        "body_position": body_position,
        "body_quaternion": body_quaternion,
        "body_linear_velocity": body_linear_velocity,
        "body_angular_velocity": body_angular_velocity,
    }


def _scene_registration() -> dict[str, object]:
    return {
        "scene_id": "sonic-flat-baseline",
        "route_id": "flat-12s",
        "source_kind": "analytic-flat",
        "source_hashes": {
            "registry": _sha("registry"),
            "manifest": _sha("manifest"),
            "scene_index": _sha("scene-index"),
        },
        "coordinate_source": "holden-y-up-right-handed-forward-plus-z",
        "coordinate_target": "mujoco-z-up-right-handed-forward-plus-x",
        "transform_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        "output_hashes": {
            "gear_scene_xml": _sha("gear-scene"),
            "scene_registration": _sha("scene-registration"),
        },
        "allowed_foot_geoms": [11, 12],
        "forbidden_geom_groups": {
            "pelvis": [1],
            "knees": [2, 3],
            "torso": [4, 5],
            "hands": [6, 7],
        },
    }


class CompleteStageAFake:
    """Complete fake only at the policy/simulator/process boundary."""

    def __init__(
        self,
        *,
        failure_gate: str | None = None,
        failure_status: str = "integration_failure",
        stream_reference_mismatch: bool = False,
        deferred_scene_identity: bool = False,
    ) -> None:
        self.failure_gate = failure_gate
        self.failure_status = failure_status
        self.stream_reference_mismatch = stream_reference_mismatch
        self.deferred_scene_identity = deferred_scene_identity

    def authenticate(self, request, inputs) -> StageAContext:
        artifact_hashes = {
            "policy": _sha("policy"),
            "encoder": None,
            "observation_config": _sha("observation"),
            "model": (
                None if self.deferred_scene_identity else _sha("model")
            ),
            "source_mjcf": _sha("source-mjcf"),
            "motion": _sha("motion"),
            "terrain": _sha("terrain"),
            "joint_map": _sha("joint-map"),
            "scene": (
                None if self.deferred_scene_identity else _sha("scene")
            ),
        }
        return StageAContext(
            gear_commit="1" * 40,
            gear_dirty=False,
            external_hashes=MappingProxyType(
                {
                    "policy": artifact_hashes["policy"],
                    "observation_config": artifact_hashes[
                        "observation_config"
                    ],
                    "source_mjcf": artifact_hashes["source_mjcf"],
                    "terrain_dir": artifact_hashes["terrain"],
                }
            ),
            motion_matching_commit="2" * 40,
            motion_matching_dirty=False,
            artifact_hashes=MappingProxyType(artifact_hashes),
            known_good_reference=inputs.gear_checkout / "known-good",
            processes=(
                MappingProxyType(
                    {
                        "name": "policy-simulator",
                        "argv": ("official-policy-runner", "--headless"),
                    }
                ),
            ),
        )

    def run_gate(self, name, request, context, bundle) -> GateResult:
        delivery_audit_sha256 = None
        gate_payload: dict[str, object] = {"gate": name}
        if name == "known_good_stream_delivery":
            audit_core = {
                "consumer_marker_count": 25,
                "logical_exact": True,
                "receipt_fence_exact": True,
            }
            delivery_audit_sha256 = hashlib.sha256(
                cli_module._canonical_json_bytes(
                    audit_core, "fake delivery audit"
                )
            ).hexdigest()
            gate_payload["delivery_audit"] = {
                **audit_core,
                "evidence_sha256": delivery_audit_sha256,
                "readiness_publications": 1,
                "logical_publications": 22,
                "padding_publications": 1,
                "receipt_fence_publications": 1,
                "consumer_markers": 25,
            }
        payload = cli_module._canonical_json_bytes(gate_payload, "fake gate")
        relative = f"gates/{name}.json"
        bundle.write_bytes(relative, payload)
        output_hash = hashlib.sha256(payload).hexdigest()
        if name == self.failure_gate:
            return GateResult(
                status=self.failure_status,
                reason=f"injected {self.failure_status}",
                outputs=MappingProxyType({relative: output_hash}),
            )

        identity: dict[str, str] = {}
        scene = None
        metrics: dict[str, object] = {}
        artifact_hashes: dict[str, str] = {}
        processes: tuple[MappingProxyType, ...] = ()
        if name == "external_identity":
            identity.update(
                {
                    "policy": _sha("policy"),
                    "encoder": _sha("encoder-absent"),
                    "observation_config": _sha("observation"),
                    "external_commit": "1" * 40,
                }
            )
        elif name == "basis_and_scene_alignment":
            identity.update(
                {
                    "official_model_xml": _sha("official-model"),
                    "generated_flat_scene": _sha("gear-scene"),
                    "initial_qpos": _sha("initial-qpos"),
                }
            )
            scene = _scene_registration()
            metrics["joint_feasibility"] = _joint_feasibility_identity()
            if self.deferred_scene_identity:
                artifact_hashes.update(
                    {"model": _sha("model"), "scene": _sha("scene")}
                )
                processes = (
                    MappingProxyType(
                        {
                            "name": "gated-simulator-import-preflight",
                            "argv": [
                                "python",
                                "-m",
                                "mm_sonic.gated_sim",
                                "--import-preflight",
                            ],
                            "executable_sha256": _sha("python"),
                        }
                    ),
                )
        elif name == "flat_mm_kinematic_replay":
            identity["mm_reference_buffer"] = _sha("mm-reference")
        elif name == "known_good_file_dynamic":
            identity["known_good_reference_buffer"] = _sha("known-good")
            metrics = {
                "joint_position_rmse_rad": 0.02,
                "pelvis_orientation_rms_rad": 0.03,
            }
        elif name == "known_good_stream_delivery":
            identity["known_good_reference_buffer"] = _sha(
                "different-known-good"
                if self.stream_reference_mismatch
                else "known-good"
            )

        evidence_hashes = {f"{name}_sha256": output_hash}
        if delivery_audit_sha256 is not None:
            evidence_hashes["delivery_audit_sha256"] = delivery_audit_sha256
        return GateResult(
            status="pass",
            identity=MappingProxyType(identity),
            evidence_hashes=MappingProxyType(evidence_hashes),
            metrics=MappingProxyType(metrics),
            outputs=MappingProxyType({relative: output_hash}),
            scene_registration=scene,
            artifact_hashes=MappingProxyType(artifact_hashes),
            processes=processes,
        )


class SonicCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.gear = self.root / "gear"
        self.gear.mkdir()
        (self.gear / "known-good").mkdir()
        self.policy = self.root / "policy.onnx"
        self.policy.write_bytes(b"real policy fixture")
        self.observation = self.root / "observation.yaml"
        self.observation.write_text("observations: []\n", encoding="ascii")
        self.source_mjcf = self.root / "g1.xml"
        self.source_mjcf.write_text("<mujoco/>\n", encoding="ascii")
        self.terrain = self.root / "terrain"
        self.terrain.mkdir()
        (self.terrain / "manifest.json").write_text("{}\n", encoding="ascii")
        self.output = self.root / "runs"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _common(self, *, output: Path | None = None) -> list[str]:
        return [
            "--gear-checkout",
            str(self.gear),
            "--policy",
            str(self.policy),
            "--observation-config",
            str(self.observation),
            "--source-mjcf",
            str(self.source_mjcf),
            "--terrain-dir",
            str(self.terrain),
            "--output-root",
            str(self.output if output is None else output),
        ]

    def _run(
        self,
        command: list[str],
        fake: CompleteStageAFake | None = None,
        *,
        environ: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        code = main(
            command,
            operations=CompleteStageAFake() if fake is None else fake,
            environ={} if environ is None else environ,
            stdout=stdout,
            stderr=stderr,
        )
        return code, stdout.getvalue(), stderr.getvalue()

    def _only_bundle(self, output: Path | None = None) -> Path:
        stage_root = (self.output if output is None else output) / "stage-a"
        bundles = tuple(stage_root.iterdir())
        self.assertEqual(len(bundles), 1)
        return bundles[0]

    @staticmethod
    def _evidence(bundle: Path) -> dict[str, object]:
        return json.loads((bundle / "stage-a-evidence.json").read_text("ascii"))

    def _stream_preload_audit_fixture(self, label: str):
        bundle = RunBundle.create(self.root / f"preload-{label}", "audit", label)
        canonical = _canonical(441)
        enabled = b"\xe2\x9c\x93 cold startup\nZMQ STREAMING MODE: ENABLED\n"
        left = b"Delta heading left: 0.1 rad\n"
        right = b"Delta heading right: 0 rad\n"
        stdout_path = bundle.write_bytes(
            "dynamic/stream/gear.stdout", enabled + left + right
        )
        post_enable_fence = MappingProxyType(
            {
                "boundary": len(enabled),
                "end_offset": len(enabled + left + right),
                "key_sequence": "qe",
                "left_line": "Delta heading left: 0.1 rad",
                "right_line": "Delta heading right: 0 rad",
                "semantics": (
                    "post-enable-reset-tail-complete-with-net-zero-heading"
                ),
            }
        )

        class Publisher:
            def prepare(_self, buffer, *, phase, attempt):
                return SimpleNamespace(
                    buffer=buffer,
                    phase=phase,
                    attempt=attempt,
                )

            def send_prepared(_self, prepared):
                archived = dict(
                    bundle.archive_transmission(
                        encode_pose_v1(prepared.buffer),
                        first_frame_index=int(prepared.buffer.frame_index[0]),
                        last_frame_index=int(prepared.buffer.frame_index[-1]),
                        phase=prepared.phase,
                        attempt=prepared.attempt,
                    )
                )
                archived["local_send_completed"] = True
                return MappingProxyType(archived)

        class Gear:
            wait_for_control_ready = True
            control_active = False
            input_prepared = True

            @staticmethod
            def group_is_stopped():
                return False

            @staticmethod
            def require_alive():
                return None

            @staticmethod
            def publication_boundary():
                return len(stdout_path.read_bytes())

            @staticmethod
            def wait_for_stream_processing(
                after_offset,
                *,
                frame_count,
                global_start,
                merged_count,
            ):
                copied = merged_count - frame_count
                start_line = (
                    "[ZMQEndpointInterface] *** Starting ZMQ processing ***"
                )
                processing_line = (
                    f"[StreamedMotionMerger] Processing {frame_count} frames, "
                    f"incoming_frame_start={global_start}, frame_step=1"
                )
                merged_line = (
                    f"[StreamedMotionMerger] Merged motion: {merged_count} frames "
                    f"(copied: {copied} + incoming: {frame_count})"
                )
                end_line = (
                    "[ZMQEndpointInterface] "
                    "*** End of ZMQ decoding processing ***"
                )
                start_bytes = f"{start_line}\n".encode("ascii")
                diagnostics = (
                    "[ZMQEndpointInterface] Protocol version: 1\n"
                    + (
                        "[ZMQEndpointInterface] Protocol version 1 established\n"
                        if global_start == 0
                        else ""
                    )
                ).encode("ascii")
                processing_bytes = f"{processing_line}\n".encode("ascii")
                between = b"[ZMQEndpointInterface] decoded protocol-v1 fields\n"
                merged_bytes = f"{merged_line}\n".encode("ascii")
                merged_to_end = (
                    "[ZMQEndpointInterface] active_protocol_version_=1\n"
                    "[ZMQEndpointInterface] result.motion->GetEncodeMode()=0\n"
                    "[ZMQEndpointInterface] motion name: streamed\n"
                    "[ZMQEndpointInterface] Merged streamed data: accepted\n"
                ).encode("ascii")
                end_bytes = f"{end_line}\n".encode("ascii")
                start_range = (after_offset, after_offset + len(start_bytes))
                processing_range = (
                    start_range[1] + len(diagnostics),
                    start_range[1] + len(diagnostics) + len(processing_bytes),
                )
                merged_range = (
                    processing_range[1] + len(between),
                    processing_range[1] + len(between) + len(merged_bytes),
                )
                end_range = (
                    merged_range[1] + len(merged_to_end),
                    merged_range[1] + len(merged_to_end) + len(end_bytes),
                )
                lines = (
                    start_bytes
                    + diagnostics
                    + processing_bytes
                    + between
                    + merged_bytes
                    + merged_to_end
                    + end_bytes
                )
                with stdout_path.open("ab") as output:
                    output.write(lines)
                    output.flush()
                return MappingProxyType(
                    {
                        "boundary": after_offset,
                        "end_offset": end_range[1],
                        "start_range": start_range,
                        "processing_range": processing_range,
                        "merged_range": merged_range,
                        "end_range": end_range,
                        "frame_count": frame_count,
                        "global_start": global_start,
                        "merged_count": merged_count,
                        "start_line": start_line,
                        "processing_line": processing_line,
                        "merged_line": merged_line,
                        "end_line": end_line,
                    }
                )

        evidence = cli_module._preload_known_good_stream(
            canonical,
            Publisher(),
            Gear(),
            post_enable_fence,
        )
        return bundle, canonical, stdout_path, evidence

    def _reseal_stage_a_bundle(
        self,
        source: Path,
        label: str,
        *,
        mutate_evidence=None,
        mutate_file=None,
        mutate_manifest=None,
        outcome=None,
    ) -> Path:
        source_manifest = json.loads((source / "manifest.json").read_text("ascii"))
        source_inventory = json.loads((source / "inventory.json").read_text("ascii"))
        evidence = json.loads(
            (source / "stage-a-evidence.json").read_text("ascii")
        )
        if mutate_evidence is not None:
            mutate_evidence(evidence)
        bundle = RunBundle.create(self.root / "resealed", "stage-a", label)
        contents_by_relative = {}
        for relative in source_inventory["files"]:
            if relative == "stage-a-evidence.json":
                continue
            else:
                contents = (source / relative).read_bytes()
            if mutate_file is not None:
                contents = mutate_file(relative, contents)
            contents_by_relative[relative] = contents
        evidence_contents = cli_module._canonical_json_bytes(
            evidence, "Stage A evidence"
        )
        if mutate_file is not None:
            evidence_contents = mutate_file(
                "stage-a-evidence.json", evidence_contents
            )
        contents_by_relative["stage-a-evidence.json"] = evidence_contents
        for relative in source_inventory["files"]:
            bundle.write_bytes(relative, contents_by_relative[relative])
        immutable = {
            "schema",
            "experiment_id",
            "run_id",
            "status",
            "created_utc",
            "finalization_started_utc",
            "finalized_utc",
            "outcome",
            "evidence",
        }
        metadata = {
            key: value
            for key, value in source_manifest.items()
            if key not in immutable
        }
        if mutate_manifest is not None:
            mutate_manifest(metadata)
        bundle.update_manifest(metadata)
        bundle.finalize(
            "complete",
            outcome=(source_manifest["outcome"] if outcome is None else outcome),
        )
        path = bundle.path
        bundle.__del__()
        self.assertTrue(verify_run_inventory(path))
        return path

    def test_required_paths_are_never_inferred_from_environment(self) -> None:
        environment = {
            "SONIC_GEAR_CHECKOUT": str(self.gear),
            "SONIC_POLICY": str(self.policy),
            "SONIC_OBS_CONFIG": str(self.observation),
            "SONIC_SOURCE_MJCF": str(self.source_mjcf),
            "SONIC_TERRAIN_DIR": str(self.terrain),
        }
        code, _stdout, stderr = self._run(["preflight"], environ=environment)
        self.assertEqual(code, 2)
        self.assertIn("--gear-checkout", stderr)
        self.assertFalse(self.output.exists())

    def test_tilde_aliases_on_every_path_option_reject_before_bundle_creation(
        self,
    ) -> None:
        options = (
            "--gear-checkout",
            "--policy",
            "--observation-config",
            "--encoder",
            "--source-mjcf",
            "--terrain-dir",
            "--output-root",
        )
        for option in options:
            for form in ("split", "equals"):
                for alias in ("~", "~/x", "~user/x"):
                    with self.subTest(option=option, form=form, alias=alias):
                        output = self.root / "tilde-output"
                        valid = {
                            "--gear-checkout": str(self.gear),
                            "--policy": str(self.policy),
                            "--observation-config": str(self.observation),
                            "--encoder": str(self.policy),
                            "--source-mjcf": str(self.source_mjcf),
                            "--terrain-dir": str(self.terrain),
                            "--output-root": str(output),
                        }
                        valid[option] = alias
                        ordered = [
                            "--gear-checkout",
                            "--policy",
                            "--observation-config",
                        ]
                        if option == "--encoder":
                            ordered.append("--encoder")
                        ordered += [
                            "--source-mjcf",
                            "--terrain-dir",
                            "--output-root",
                        ]
                        command = ["stage-a", "--mode", "known-good-stream"]
                        for flag in ordered:
                            value = valid[flag]
                            if flag == option and form == "equals":
                                command.append(f"{flag}={value}")
                            else:
                                command.extend([flag, value])

                        with patch.object(
                            cli_module.RunBundle, "create"
                        ) as create_bundle:
                            code, _stdout, stderr = self._run(command)

                        self.assertEqual(code, 2)
                        self.assertIn("user expansion", stderr)
                        create_bundle.assert_not_called()
                        if option != "--output-root":
                            self.assertFalse(output.exists())

    def test_output_root_must_be_confined_away_from_external_inputs(self) -> None:
        nested = self.gear / "forbidden-output"
        code, _stdout, stderr = self._run(
            ["preflight", *self._common(output=nested)]
        )
        self.assertEqual(code, 2)
        self.assertIn("output_root", stderr)
        self.assertFalse(nested.exists())

    def test_output_root_cannot_create_a_claimed_missing_input_path(self) -> None:
        missing_policy = self.root / "missing-policy.onnx"
        command = ["preflight", *self._common(output=missing_policy)]
        policy_index = command.index("--policy") + 1
        command[policy_index] = str(missing_policy)
        code, _stdout, stderr = self._run(command)
        self.assertEqual(code, 2)
        self.assertIn("output_root", stderr)
        self.assertFalse(missing_policy.exists())

        nested_policy_output = missing_policy / "runs"
        command = ["preflight", *self._common(output=nested_policy_output)]
        policy_index = command.index("--policy") + 1
        command[policy_index] = str(missing_policy)
        code, _stdout, stderr = self._run(command)
        self.assertEqual(code, 2)
        self.assertIn("output_root", stderr)
        self.assertFalse(missing_policy.exists())

        missing_gear = self.root / "missing-gear"
        nested = missing_gear / "runs"
        command = ["preflight", *self._common(output=nested)]
        gear_index = command.index("--gear-checkout") + 1
        command[gear_index] = str(missing_gear)
        code, _stdout, stderr = self._run(command)
        self.assertEqual(code, 2)
        self.assertIn("output_root", stderr)
        self.assertFalse(missing_gear.exists())

    def test_preflight_passes_only_its_three_nonstepping_gates(self) -> None:
        argv = ["preflight", *self._common()]
        code, stdout, stderr = self._run(argv)
        self.assertEqual((code, stderr), (0, ""))
        bundle = self._only_bundle()
        evidence = self._evidence(bundle)
        self.assertEqual(
            [gate["status"] for gate in evidence["gates"]],
            ["pass", "pass", "pass", "pending", "pending", "pending", "pending"],
        )
        self.assertEqual(evidence["command_status"], "pass")
        self.assertEqual(evidence["stage_a_status"], "incomplete")
        self.assertTrue(verify_run_inventory(bundle))
        summary = json.loads(stdout)
        self.assertEqual(Path(summary["evidence"]), bundle / "stage-a-evidence.json")
        manifest = json.loads((bundle / "manifest.json").read_text("ascii"))
        self.assertEqual(
            manifest["command_script"],
            {
                "id": "stage-a-preflight-no-trial",
                "sha256": hashlib.sha256(
                    b"mm-sonic-stage-a/preflight/no-trial\n"
                ).hexdigest(),
            },
        )

    def test_capable_preflight_merges_late_scene_and_process_identity(self) -> None:
        fake = CompleteStageAFake(deferred_scene_identity=True)
        code, _stdout, stderr = self._run(
            ["preflight", *self._common()], fake
        )
        self.assertEqual((code, stderr), (0, ""))
        bundle = self._only_bundle()
        manifest = json.loads((bundle / "manifest.json").read_text("ascii"))
        self.assertEqual(manifest["artifact_hashes"]["model"], _sha("model"))
        self.assertEqual(manifest["artifact_hashes"]["scene"], _sha("scene"))
        self.assertEqual(
            manifest["processes"][-1]["name"],
            "gated-simulator-import-preflight",
        )
        expected_feasibility = _joint_feasibility_identity()
        self.assertEqual(manifest["joint_feasibility"], expected_feasibility)
        evidence = self._evidence(bundle)
        self.assertEqual(
            evidence["gates"][2]["metrics"]["joint_feasibility"],
            expected_feasibility,
        )
        self.assertEqual(
            evidence["metrics"]["basis_and_scene_alignment"][
                "joint_feasibility"
            ],
            expected_feasibility,
        )
        self.assertTrue(verify_run_inventory(bundle))

    def test_exact_argv_and_only_the_safe_environment_allowlist_are_recorded(self) -> None:
        argv = ["stage-a", "--mode", "mm-reference", *self._common()]
        environment = {
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": "sonic/python",
            "CUDA_VISIBLE_DEVICES": "0",
            "LD_LIBRARY_PATH": "/safe/runtime",
            "HOME": "/must/not/be/recorded",
            "AWS_SECRET_ACCESS_KEY": "must-not-be-recorded",
            "SONIC_POLICY": "/must/not/be-read",
        }
        code, _stdout, stderr = self._run(argv, environ=environment)
        self.assertEqual((code, stderr), (0, ""))
        manifest = json.loads(
            (self._only_bundle() / "manifest.json").read_text("ascii")
        )
        cli_process = manifest["processes"][0]
        self.assertEqual(cli_process["argv"], ["mm_sonic.cli", *argv])
        self.assertEqual(
            cli_process["environment"],
            {
                "allowlist": [
                    "CUDA_VISIBLE_DEVICES",
                    "LD_LIBRARY_PATH",
                    "PATH",
                    "PYTHONPATH",
                ],
                "values": {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "LD_LIBRARY_PATH": "/safe/runtime",
                    "PATH": "/usr/bin:/bin",
                    "PYTHONPATH": "sonic/python",
                },
            },
        )
        serialized = json.dumps(manifest)
        self.assertNotIn("AWS_SECRET", serialized)
        self.assertNotIn("must-not-be-recorded", serialized)
        expected_script = command_script_bytes(
            scene_id="sonic-flat-baseline",
            route_id="flat-12s",
            commands=flat_command_script(),
        )
        self.assertEqual(
            manifest["command_script"],
            {
                "id": "flat-12s",
                "sha256": hashlib.sha256(expected_script).hexdigest(),
            },
        )

    def test_not_run_is_result_free_immutable_evidence_and_exit_four(self) -> None:
        fake = CompleteStageAFake(
            failure_gate="basis_and_scene_alignment",
            failure_status="not_run",
        )
        code, _stdout, stderr = self._run(
            ["stage-a", "--mode", "known-good-stream", *self._common()],
            fake,
        )
        self.assertEqual((code, stderr), (4, ""))
        bundle = self._only_bundle()
        evidence = self._evidence(bundle)
        self.assertEqual(evidence["command_status"], "not_run")
        self.assertEqual(evidence["stage_a_status"], "not_run")
        self.assertEqual(evidence["metrics"], {})
        self.assertNotIn("verdict", evidence)
        self.assertNotIn("feasibility", json.dumps(evidence).lower())
        self.assertEqual(
            [gate["status"] for gate in evidence["gates"]],
            ["pass", "pass", "not_run", "not_run", "not_run", "not_run", "not_run"],
        )
        self.assertTrue(verify_run_inventory(bundle))
        for path in (bundle / "manifest.json", bundle / "stage-a-evidence.json"):
            self.assertFalse(path.stat().st_mode & stat.S_IWUSR)

    def test_missing_explicit_capabilities_write_unfabricated_not_run(self) -> None:
        cases = (
            ("--policy", "missing-policy.onnx"),
            ("--observation-config", "missing-observation.yaml"),
            ("--source-mjcf", "missing-g1.xml"),
            ("--terrain-dir", "missing-terrain"),
        )
        for index, (flag, leaf) in enumerate(cases):
            with self.subTest(flag=flag):
                output = self.root / f"missing-capability-{index}"
                command = ["preflight", *self._common(output=output)]
                command[command.index(flag) + 1] = str(self.root / leaf)
                code, stdout, stderr = self._run(command)
                self.assertEqual((code, stderr), (4, ""))
                bundle = self._only_bundle(output)
                evidence = self._evidence(bundle)
                self.assertEqual(evidence["command_status"], "not_run")
                self.assertEqual(evidence["identity"], {})
                self.assertEqual(evidence["metrics"], {})
                self.assertRegex(evidence["invocation_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(evidence["diagnosis_sha256"], r"^[0-9a-f]{64}$")
                summary = json.loads(stdout)
                self.assertEqual(Path(summary["evidence"]), bundle / "stage-a-evidence.json")
                manifest = json.loads((bundle / "manifest.json").read_text("ascii"))
                self.assertEqual(manifest["status"], "not_run")
                self.assertIsNone(manifest["external"]["gear_commit"])
                self.assertIsNone(manifest["repositories"]["gear_sonic"]["commit"])
                self.assertTrue(
                    all(value is None for value in manifest["external"]["hashes"].values())
                )
                self.assertTrue(verify_run_inventory(bundle))
                for path in bundle.rglob("*"):
                    if path.is_file():
                        self.assertFalse(path.stat().st_mode & stat.S_IWUSR)

    def test_authentication_mismatch_is_exit_two_without_fabricated_identity(
        self,
    ) -> None:
        class WrongAuthentication(CompleteStageAFake):
            def authenticate(self, request, inputs):
                raise cli_module.ExternalInputError(
                    "pinned GEAR commit does not match"
                )

        code, stdout, stderr = self._run(
            ["preflight", *self._common()], WrongAuthentication()
        )
        self.assertEqual(code, 2)
        self.assertIn("pinned GEAR commit", stderr)
        bundle = self._only_bundle()
        evidence = self._evidence(bundle)
        self.assertEqual(evidence["command_status"], "integration_failure")
        self.assertEqual(evidence["stage_a_status"], "failed")
        self.assertEqual(evidence["identity"], {})
        self.assertEqual(evidence["metrics"], {})
        summary = json.loads(stdout)
        self.assertEqual(summary["status"], "integration_failure")
        manifest = json.loads((bundle / "manifest.json").read_text("ascii"))
        self.assertEqual(manifest["status"], "not_run")
        self.assertIsNone(manifest["external"]["gear_commit"])
        self.assertIsNone(manifest["repositories"]["gear_sonic"]["commit"])
        self.assertTrue(
            all(value is None for value in manifest["external"]["hashes"].values())
        )
        self.assertTrue(verify_run_inventory(bundle))
        for path in bundle.rglob("*"):
            if path.is_file():
                self.assertFalse(path.stat().st_mode & stat.S_IWUSR)

    def test_source_mjcf_lfs_pointer_is_result_free_exit_four_before_parsing(
        self,
    ) -> None:
        self.source_mjcf.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\n"
            + b"oid sha256:"
            + b"0" * 64
            + b"\nsize 4096\n"
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        code = main(
            ["preflight", *self._common()],
            operations=cli_module.DefaultStageAOperations(),
            environ={},
            stdout=stdout,
            stderr=stderr,
        )
        self.assertEqual((code, stderr.getvalue()), (4, ""))
        bundle = self._only_bundle()
        evidence = self._evidence(bundle)
        self.assertEqual(evidence["command_status"], "not_run")
        self.assertEqual(evidence["stage_a_status"], "not_run")
        self.assertEqual(evidence["identity"], {})
        self.assertEqual(evidence["metrics"], {})
        diagnosis = json.loads(
            (bundle / "pre-execution-diagnosis.json").read_text("ascii")
        )
        self.assertIn("Git-LFS", diagnosis["reason"])
        self.assertIn(str(self.source_mjcf), diagnosis["reason"])
        manifest = json.loads((bundle / "manifest.json").read_text("ascii"))
        self.assertEqual(manifest["status"], "not_run")
        self.assertTrue(
            all(value is None for value in manifest["external"]["hashes"].values())
        )
        self.assertTrue(verify_run_inventory(bundle))
        for path in bundle.rglob("*"):
            if path.is_file():
                self.assertFalse(path.stat().st_mode & stat.S_IWUSR)

    def test_integration_and_scientific_failures_use_distinct_exit_codes(self) -> None:
        cases = (
            ("external_identity", "integration_failure", "preflight", 2),
            (
                "flat_mm_kinematic_replay",
                "scientific_failure",
                "mm-reference",
                3,
            ),
        )
        for index, (gate, status_value, mode, expected) in enumerate(cases):
            with self.subTest(status=status_value):
                output = self.root / f"runs-{index}"
                fake = CompleteStageAFake(
                    failure_gate=gate, failure_status=status_value
                )
                command = (
                    ["preflight", *self._common(output=output)]
                    if mode == "preflight"
                    else [
                        "stage-a",
                        "--mode",
                        mode,
                        *self._common(output=output),
                    ]
                )
                code, _stdout, stderr = self._run(command, fake)
                self.assertEqual((code, stderr), (expected, ""))
                evidence = self._evidence(self._only_bundle(output))
                self.assertEqual(evidence["command_status"], status_value)

    def test_finalization_failure_is_reported_once_without_retry_masking(self):
        calls = []

        def fail_after_transition(bundle, status, *, outcome):
            calls.append((status, dict(outcome)))
            bundle._status = "finalizing"
            raise ContractError("primary finalization failure")

        with patch.object(
            cli_module.RunBundle,
            "finalize",
            autospec=True,
            side_effect=fail_after_transition,
        ):
            code, stdout, stderr = self._run(
                ["preflight", *self._common()]
            )

        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("primary finalization failure", stderr)
        self.assertNotIn("already finalized or finalizing", stderr)
        self.assertEqual(len(calls), 1)

    def test_pretransition_finalization_failure_is_not_retried_or_masked(self):
        calls = []

        def fail_before_transition(_bundle, status, *, outcome):
            calls.append((status, dict(outcome)))
            raise ContractError("primary pretransition finalization failure")

        with patch.object(
            cli_module.RunBundle,
            "finalize",
            autospec=True,
            side_effect=fail_before_transition,
        ):
            code, stdout, stderr = self._run(
                ["preflight", *self._common()]
            )

        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("primary pretransition finalization failure", stderr)
        self.assertNotIn("evidence output already exists", stderr)
        self.assertEqual(len(calls), 1)

    def test_bundle_rename_cannot_hide_written_evidence_and_mask_finalization(self):
        calls = []

        def fail_after_bundle_rename(bundle, status, *, outcome):
            calls.append((status, dict(outcome)))
            bundle.path.rename(bundle.path.with_name(f"{bundle.path.name}-moved"))
            raise ContractError("primary renamed-bundle finalization failure")

        with patch.object(
            cli_module.RunBundle,
            "finalize",
            autospec=True,
            side_effect=fail_after_bundle_rename,
        ):
            code, stdout, stderr = self._run(
                ["preflight", *self._common()]
            )

        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("primary renamed-bundle finalization failure", stderr)
        self.assertNotIn("evidence output already exists", stderr)
        self.assertEqual(len(calls), 1)

    def test_each_mode_runs_only_its_ordered_prerequisite_prefix(self) -> None:
        cases = (
            ("mm-reference", 4),
            ("known-good-file", 5),
            ("known-good-stream", 7),
        )
        for index, (mode, passed) in enumerate(cases):
            with self.subTest(mode=mode):
                output = self.root / f"mode-{index}"
                code, _stdout, stderr = self._run(
                    [
                        "stage-a",
                        "--mode",
                        mode,
                        *self._common(output=output),
                    ]
                )
                self.assertEqual((code, stderr), (0, ""))
                statuses = [
                    gate["status"]
                    for gate in self._evidence(self._only_bundle(output))[
                        "gates"
                    ]
                ]
                self.assertEqual(statuses, ["pass"] * passed + ["pending"] * (7 - passed))

    def test_stream_reference_mismatch_prevents_dynamic_launch(self) -> None:
        fake = CompleteStageAFake(stream_reference_mismatch=True)
        code, _stdout, stderr = self._run(
            ["stage-a", "--mode", "known-good-stream", *self._common()],
            fake,
        )
        self.assertEqual((code, stderr), (2, ""))
        evidence = self._evidence(self._only_bundle())
        self.assertEqual(
            evidence["gates"][5]["status"], "integration_failure"
        )
        self.assertIn("identity", evidence["gates"][5]["reason"])
        self.assertEqual(evidence["gates"][6]["status"], "not_run")
        self.assertFalse(
            (self._only_bundle() / "gates/known_good_stream_dynamic.json").exists()
        )

    def test_startup_failure_retains_attempted_process_and_trt_cache(self) -> None:
        class StartupFailureFake(CompleteStageAFake):
            def run_gate(self, name, request, context, bundle):
                if name != "known_good_file_dynamic":
                    return super().run_gate(name, request, context, bundle)
                relative = "runtime-inputs/file/policy_policy.trt"
                payload = b"partial TensorRT cache\n"
                bundle.write_bytes(relative, payload)
                executable = Path(sys.executable).resolve(strict=True)
                return GateResult(
                    status="integration_failure",
                    reason="ProcessError: synthetic startup failure",
                    outputs=MappingProxyType(
                        {relative: hashlib.sha256(payload).hexdigest()}
                    ),
                    processes=(
                        MappingProxyType(
                            {
                                "name": "g1_deploy_onnx_ref-known-good-file",
                                "argv": [
                                    str(executable),
                                    "lo",
                                    "/run/runtime-inputs/file/policy.onnx",
                                    "/run/known-good/reference-base",
                                ],
                                "executable_sha256": hashlib.sha256(
                                    executable.read_bytes()
                                ).hexdigest(),
                                "environment": {
                                    "allowlist": list(request.environment),
                                    "values": dict(request.environment),
                                },
                                "outputs": [relative],
                            }
                        ),
                    ),
                )

        environment = {"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "0"}
        code, _stdout, stderr = self._run(
            ["stage-a", "--mode", "known-good-file", *self._common()],
            StartupFailureFake(),
            environ=environment,
        )
        self.assertEqual((code, stderr), (2, ""))
        bundle = self._only_bundle()
        evidence = self._evidence(bundle)
        cache_hash = hashlib.sha256(b"partial TensorRT cache\n").hexdigest()
        self.assertEqual(
            evidence["gates"][4]["outputs"],
            {"runtime-inputs/file/policy_policy.trt": cache_hash},
        )
        manifest = json.loads((bundle / "manifest.json").read_text("ascii"))
        process = manifest["processes"][-1]
        self.assertEqual(process["name"], "g1_deploy_onnx_ref-known-good-file")
        self.assertEqual(
            process["outputs"], ["runtime-inputs/file/policy_policy.trt"]
        )
        self.assertEqual(
            process["environment"]["values"], environment
        )
        self.assertRegex(process["executable_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(verify_run_inventory(bundle))

    def test_stage_a_prerequisite_is_inventory_and_identity_bound(self) -> None:
        file_output = self.root / "file-only"
        file_command = [
            "stage-a",
            "--mode",
            "known-good-file",
            *self._common(output=file_output),
        ]
        self.assertEqual(self._run(file_command)[0], 0)
        file_evidence = self._only_bundle(file_output) / "stage-a-evidence.json"
        with self.assertRaisesRegex(ContractError, "stream"):
            validate_stage_a_prerequisite(file_evidence, expected_identity=None)

        stream_output = self.root / "stream"
        stream_command = [
            "stage-a",
            "--mode",
            "known-good-stream",
            *self._common(output=stream_output),
        ]
        self.assertEqual(self._run(stream_command)[0], 0)
        stream_evidence = (
            self._only_bundle(stream_output) / "stage-a-evidence.json"
        )
        evidence = self._evidence(self._only_bundle(stream_output))
        validated = validate_stage_a_prerequisite(
            stream_evidence,
            expected_identity=evidence["identity"],
        )
        self.assertEqual(validated["stage_a_status"], "pass")
        mismatched = dict(evidence["identity"])
        mismatched["policy"] = _sha("other-policy")
        with self.assertRaisesRegex(ContractError, "identity"):
            validate_stage_a_prerequisite(
                stream_evidence, expected_identity=mismatched
            )

    def test_stage_a_prerequisite_reconstructs_every_sealed_claim(self) -> None:
        command = [
            "stage-a",
            "--mode",
            "known-good-stream",
            *self._common(),
        ]
        self.assertEqual(self._run(command)[0], 0)
        source = self._only_bundle()
        expected_identity = self._evidence(source)["identity"]

        cases = []

        def evidence_case(label, mutation):
            cases.append(
                (
                    label,
                    self._reseal_stage_a_bundle(
                        source, label, mutate_evidence=mutation
                    ),
                )
            )

        evidence_case(
            "stale-registry",
            lambda value: value.__setitem__("registry_sha256", "0" * 64),
        )
        evidence_case(
            "forged-command",
            lambda value: value.__setitem__("command", "preflight"),
        )
        evidence_case(
            "forged-mode",
            lambda value: value.__setitem__("mode", "known-good-file"),
        )
        evidence_case(
            "forged-argv",
            lambda value: value.__setitem__(
                "argv", ["mm_sonic.cli", "stage-a", "--mode", "known-good-file"]
            ),
        )
        evidence_case(
            "empty-evidence-hashes",
            lambda value: value["gates"][0].__setitem__("evidence_hashes", {}),
        )
        evidence_case(
            "gate-extra-key",
            lambda value: value["gates"][0].__setitem__("forged", True),
        )
        evidence_case(
            "passing-reason",
            lambda value: value["gates"][0].__setitem__(
                "reason", "pass cannot have a reason"
            ),
        )

        def drop_top_output(value):
            value["outputs"].pop(next(iter(value["outputs"])))

        evidence_case("top-output-mismatch", drop_top_output)

        def conflict_identity(value):
            value["gates"][1]["identity"]["policy"] = _sha("conflict")

        evidence_case("identity-conflict", conflict_identity)
        evidence_case(
            "metrics-mismatch",
            lambda value: value.__setitem__("metrics", {}),
        )

        def changed_gate_output(relative, contents):
            if relative == "gates/external_identity.json":
                return contents + b"forged\n"
            return contents

        cases.append(
            (
                "changed-output-digest",
                self._reseal_stage_a_bundle(
                    source,
                    "changed-output-digest",
                    mutate_file=changed_gate_output,
                ),
            )
        )
        cases.append(
            (
                "terminal-outcome-mismatch",
                self._reseal_stage_a_bundle(
                    source,
                    "terminal-outcome-mismatch",
                    outcome={
                        "status": "integration_failure",
                        "stage_a_status": "failed",
                    },
                ),
            )
        )

        def forged_manifest_process(metadata):
            metadata["processes"][0]["argv"] = [
                "mm_sonic.cli",
                "stage-a",
                "--mode",
                "known-good-file",
            ]

        cases.append(
            (
                "manifest-argv-mismatch",
                self._reseal_stage_a_bundle(
                    source,
                    "manifest-argv-mismatch",
                    mutate_manifest=forged_manifest_process,
                ),
            )
        )

        evidence_case(
            "arbitrary-evidence-hash-key",
            lambda value: value["gates"][0].__setitem__(
                "evidence_hashes", {"arbitrary_sha256": "0" * 64}
            ),
        )
        evidence_case(
            "primary-evidence-hash-mismatch",
            lambda value: value["gates"][0]["evidence_hashes"].__setitem__(
                "external_identity_sha256", "0" * 64
            ),
        )
        for gate_index, gate_name in enumerate(GATES[1:], start=1):
            evidence_case(
                f"{gate_name}-primary-hash-mismatch",
                lambda value, gate_index=gate_index, gate_name=gate_name: value[
                    "gates"
                ][gate_index]["evidence_hashes"].__setitem__(
                    f"{gate_name}_sha256", "0" * 64
                ),
            )
        evidence_case(
            "stream-delivery-missing-audit-hash",
            lambda value: value["gates"][5]["evidence_hashes"].pop(
                "delivery_audit_sha256"
            ),
        )
        evidence_case(
            "stream-delivery-extra-hash",
            lambda value: value["gates"][5]["evidence_hashes"].__setitem__(
                "forged_sha256", "0" * 64
            ),
        )

        def reseal_mutated_delivery(label, mutate_delivery):
            captured = {}

            def capture_evidence(value):
                captured["evidence"] = value

            def mutate_primary(relative, contents):
                if relative != "gates/known_good_stream_delivery.json":
                    return contents
                payload = json.loads(contents.decode("ascii"))
                mutate_delivery(payload["delivery_audit"])
                changed = cli_module._canonical_json_bytes(
                    payload, "mutated stream delivery"
                )
                digest = hashlib.sha256(changed).hexdigest()
                evidence = captured["evidence"]
                gate = evidence["gates"][5]
                gate["outputs"][relative] = digest
                gate["evidence_hashes"][
                    "known_good_stream_delivery_sha256"
                ] = digest
                evidence["outputs"][relative] = digest
                return changed

            return self._reseal_stage_a_bundle(
                source,
                label,
                mutate_evidence=capture_evidence,
                mutate_file=mutate_primary,
            )

        cases.append(
            (
                "delivery-audit-embedded-hash-mismatch",
                reseal_mutated_delivery(
                    "delivery-audit-embedded-hash-mismatch",
                    lambda audit: audit.__setitem__(
                        "evidence_sha256", "0" * 64
                    ),
                ),
            )
        )
        cases.append(
            (
                "delivery-audit-recomputed-hash-mismatch",
                reseal_mutated_delivery(
                    "delivery-audit-recomputed-hash-mismatch",
                    lambda audit: audit.__setitem__("logical_exact", False),
                ),
            )
        )

        source_argv = list(self._evidence(source)["argv"])

        def reseal_paired_argv(label, argv):
            return self._reseal_stage_a_bundle(
                source,
                label,
                mutate_evidence=lambda value: value.__setitem__("argv", argv),
                mutate_manifest=lambda metadata: metadata["processes"][0].__setitem__(
                    "argv", argv
                ),
            )

        policy_index = source_argv.index("--policy") + 1
        invalid_argv = {
            "argv-missing-required": [
                "mm_sonic.cli",
                "stage-a",
                "--mode",
                "known-good-stream",
            ],
            "argv-empty-required": [
                *source_argv[: policy_index - 1],
                "--policy=",
                *source_argv[policy_index + 1 :],
            ],
            "argv-duplicate-required": [
                *source_argv,
                "--policy",
                source_argv[policy_index],
            ],
            "argv-unknown-option": [*source_argv, "--unknown", "value"],
            "argv-positional": [*source_argv, "stray-positional"],
            "argv-duplicate-encoder": [
                *source_argv,
                "--encoder",
                str(self.policy),
                "--encoder",
                str(self.policy),
            ],
        }
        mismatched_policy = list(source_argv)
        mismatched_policy[policy_index] = str(self.source_mjcf)
        invalid_argv["argv-manifest-path-mismatch"] = mismatched_policy
        output_index = source_argv.index("--output-root") + 1
        mismatched_output = list(source_argv)
        mismatched_output[output_index] = str(self.root)
        invalid_argv["argv-output-root-mismatch"] = mismatched_output
        for label, argv in invalid_argv.items():
            cases.append((label, reseal_paired_argv(label, argv)))

        def mismatch_cli_environment(metadata):
            metadata["processes"][0]["environment"] = {
                "allowlist": [
                    "CUDA_VISIBLE_DEVICES",
                    "LD_LIBRARY_PATH",
                    "PATH",
                    "PYTHONPATH",
                ],
                "values": {"PATH": "/forged/bin"},
            }

        cases.append(
            (
                "manifest-environment-mismatch",
                self._reseal_stage_a_bundle(
                    source,
                    "manifest-environment-mismatch",
                    mutate_manifest=mismatch_cli_environment,
                ),
            )
        )
        for label, bundle in cases:
            with self.subTest(label=label):
                with self.assertRaises(ContractError):
                    validate_stage_a_prerequisite(
                        bundle / "stage-a-evidence.json",
                        expected_identity=expected_identity,
                    )

    def test_stage_a_prerequisite_accepts_relative_equals_form_invocation(self) -> None:
        output = self.root / "relative-equals-output"
        invocation_cwd = self.root / "invocation-cwd"
        validation_cwd = self.root / "nested" / "validation-cwd"
        invocation_cwd.mkdir()
        validation_cwd.mkdir(parents=True)
        relative = lambda path: os.path.relpath(path, invocation_cwd)
        command = [
            "stage-a",
            "--mode=known-good-stream",
            f"--gear-checkout={relative(self.gear)}",
            "--policy",
            relative(self.policy),
            f"--observation-config={relative(self.observation)}",
            "--source-mjcf",
            relative(self.source_mjcf),
            f"--terrain-dir={relative(self.terrain)}",
            f"--output-root={relative(output)}",
        ]
        original_cwd = Path.cwd()
        try:
            os.chdir(invocation_cwd)
            self.assertEqual(self._run(command)[0], 0)
        finally:
            os.chdir(original_cwd)
        bundle = self._only_bundle(output)
        try:
            os.chdir(validation_cwd)
            validated = validate_stage_a_prerequisite(
                bundle / "stage-a-evidence.json",
                expected_identity=self._evidence(bundle)["identity"],
            )
        finally:
            os.chdir(original_cwd)
        self.assertEqual(validated["argv"], ["mm_sonic.cli", *command])
        self.assertEqual(validated["invocation_cwd"], str(invocation_cwd))
        manifest = json.loads((bundle / "manifest.json").read_text("ascii"))
        self.assertEqual(
            manifest["processes"][0]["invocation_cwd"], str(invocation_cwd)
        )

    def test_stage_a_prerequisite_rejects_invocation_cwd_forgeries(self) -> None:
        output = self.root / "cwd-forgery-output"
        invocation_cwd = self.root / "cwd-forgery-invocation"
        validation_cwd = self.root / "nested" / "cwd-forgery-validation"
        alternate_cwd = self.root / "cwd-forgery-alternate"
        invocation_cwd.mkdir()
        validation_cwd.mkdir(parents=True)
        alternate_cwd.mkdir()
        relative = lambda path: os.path.relpath(path, invocation_cwd)
        command = [
            "stage-a",
            "--mode=known-good-stream",
            f"--gear-checkout={relative(self.gear)}",
            "--policy",
            relative(self.policy),
            f"--observation-config={relative(self.observation)}",
            "--source-mjcf",
            relative(self.source_mjcf),
            f"--terrain-dir={relative(self.terrain)}",
            f"--output-root={relative(output)}",
        ]
        original_cwd = Path.cwd()
        try:
            os.chdir(invocation_cwd)
            self.assertEqual(self._run(command)[0], 0)
        finally:
            os.chdir(original_cwd)
        source = self._only_bundle(output)
        expected_identity = self._evidence(source)["identity"]
        noncanonical = f"{invocation_cwd}/../{invocation_cwd.name}"

        cases = (
            (
                "cwd-evidence-only",
                self._reseal_stage_a_bundle(
                    source,
                    "cwd-evidence-only",
                    mutate_evidence=lambda value: value.__setitem__(
                        "invocation_cwd", str(alternate_cwd)
                    ),
                ),
            ),
            (
                "cwd-manifest-only",
                self._reseal_stage_a_bundle(
                    source,
                    "cwd-manifest-only",
                    mutate_manifest=lambda metadata: metadata["processes"][0].__setitem__(
                        "invocation_cwd", str(alternate_cwd)
                    ),
                ),
            ),
            (
                "cwd-paired-substitution",
                self._reseal_stage_a_bundle(
                    source,
                    "cwd-paired-substitution",
                    mutate_evidence=lambda value: value.__setitem__(
                        "invocation_cwd", str(alternate_cwd)
                    ),
                    mutate_manifest=lambda metadata: metadata["processes"][0].__setitem__(
                        "invocation_cwd", str(alternate_cwd)
                    ),
                ),
            ),
            (
                "cwd-paired-noncanonical",
                self._reseal_stage_a_bundle(
                    source,
                    "cwd-paired-noncanonical",
                    mutate_evidence=lambda value: value.__setitem__(
                        "invocation_cwd", noncanonical
                    ),
                    mutate_manifest=lambda metadata: metadata["processes"][0].__setitem__(
                        "invocation_cwd", noncanonical
                    ),
                ),
            ),
        )
        try:
            os.chdir(validation_cwd)
            for label, bundle in cases:
                with self.subTest(label=label), self.assertRaises(ContractError):
                    validate_stage_a_prerequisite(
                        bundle / "stage-a-evidence.json",
                        expected_identity=expected_identity,
                    )
        finally:
            os.chdir(original_cwd)

    def test_stage_a_path_resolution_rejects_user_expansion_aliases(self) -> None:
        with self.assertRaises(ContractError):
            metrics_module._stage_a_invocation_cwd("~")
        with self.assertRaises(ContractError):
            metrics_module._stage_a_resolved_path(
                "~", "Stage A test path", invocation_cwd=self.root
            )

    def test_stage_a_invocation_cwd_accepts_only_the_canonical_sealed_string(
        self,
    ) -> None:
        canonical = self.root / "canonical-cwd"
        canonical.mkdir()
        sealed = str(canonical.resolve(strict=True))

        accepted = metrics_module._stage_a_invocation_cwd(sealed)
        self.assertEqual(str(accepted), sealed)

        # A relative argv path is resolved against the sealed invocation cwd,
        # not the later validator cwd, so the same string authenticates the
        # canonical file even after chdir into an unrelated directory.
        validation_cwd = self.root / "nested" / "validator-cwd"
        validation_cwd.mkdir(parents=True)
        relative_target = os.path.relpath(self.policy, canonical)
        original_cwd = Path.cwd()
        try:
            os.chdir(validation_cwd)
            resolved = metrics_module._stage_a_resolved_path(
                relative_target,
                "Stage A policy",
                invocation_cwd=accepted,
            )
        finally:
            os.chdir(original_cwd)
        self.assertEqual(resolved, self.policy.resolve(strict=True))

        parent_segment = f"{canonical}/../{canonical.name}"
        self.assertEqual(Path(parent_segment).resolve(strict=True), canonical)
        noncanonical = {
            "relative": canonical.name,
            "trailing-slash": f"{sealed}/",
            "parent-segment": parent_segment,
            "tilde": "~",
            "tilde-child": f"~/{canonical.name}",
        }
        for label, value in noncanonical.items():
            with self.subTest(label=label):
                with self.assertRaises(ContractError):
                    metrics_module._stage_a_invocation_cwd(value)

    def test_stage_a_registry_has_the_exact_ordered_gate_contract(self) -> None:
        registry = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "sonic/configs/experiments/stage_a.json"
            ).read_text("ascii")
        )
        self.assertEqual(registry["schema"], "mm-sonic-stage-a/v1")
        self.assertEqual(registry["gates"], list(GATES))
        projection = registry["known_good_projection"]
        self.assertEqual(
            projection["rule"], dict(cli_module._KNOWN_GOOD_PROJECTION_RULE)
        )
        self.assertEqual(
            projection["rule_sha256"], cli_module._projection_rule_sha256()
        )
        self.assertEqual(
            projection["source_sha256_keys"],
            sorted(
                f"gear:known_good_reference/{name}"
                for name in cli_module.KNOWN_GOOD_REFERENCE_FILES
            ),
        )
        self.assertEqual(
            projection["rule"]["root_body_linear_velocity"],
            "positive-zero-f32",
        )
        self.assertEqual(
            projection["rule"]["root_body_angular_velocity"],
            "positive-zero-f32",
        )


class ProductionAdapterBoundaryTests(unittest.TestCase):
    _stream_preload_audit_fixture = SonicCLITests._stream_preload_audit_fixture

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _attempt_provenance_fixture(self):
        sonic_root = self.root / "sonic"
        mm_server = sonic_root / "build/mm_chunk_server"
        mm_server.parent.mkdir(parents=True)
        mm_server.write_bytes(b"authenticated MM server\n")
        mm_server.chmod(0o755)
        gear = self.root / "gear"
        gear_binary = gear / cli_module._GEAR_BINARY_RELATIVE
        gear_binary.parent.mkdir(parents=True)
        gear_binary.write_bytes(b"authenticated GEAR binary\n")
        gear_binary.chmod(0o755)
        policy = self.root / "policy.onnx"
        observation = self.root / "observation.yaml"
        terrain = self.root / "terrain"
        source = self.root / "source.xml"
        policy.write_bytes(b"policy\n")
        observation.write_bytes(b"observations: []\n")
        terrain.mkdir()
        source.write_bytes(b"<mujoco/>\n")
        inputs = ExternalInputs(
            gear_checkout=gear.resolve(),
            policy=policy.resolve(),
            observation_config=observation.resolve(),
            encoder=None,
            terrain_dir=terrain.resolve(),
            source_mjcf=source.resolve(),
        )
        context = StageAContext(
            gear_commit="1" * 40,
            gear_dirty=False,
            external_hashes=MappingProxyType({}),
            motion_matching_commit="2" * 40,
            motion_matching_dirty=False,
            artifact_hashes=MappingProxyType({}),
            known_good_reference=gear,
            processes=(),
        )
        operations = cli_module.DefaultStageAOperations()
        operations._runtime[id(context)] = {
            "verified_external": SimpleNamespace(inputs=inputs),
            "joint_feasibility": parse_joint_feasibility_identity(
                _joint_feasibility_identity()
            ),
        }
        environment = {"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "0"}
        request = StageARequest(
            command="stage-a",
            mode="known-good-stream",
            argv=(),
            namespace=argparse.Namespace(),
            environment=MappingProxyType(environment),
        )
        return sonic_root, inputs, context, operations, request, environment

    def test_joint_projection_timeout_retains_attempt_and_partial_output(self) -> None:
        (
            sonic_root,
            inputs,
            context,
            operations,
            request,
            environment,
        ) = self._attempt_provenance_fixture()
        projection = sonic_root / "build/g1_project_pose_cli"
        projection.write_bytes(b"authenticated projection CLI\n")
        projection.chmod(0o755)
        committed = cli_module._JOINT_CONTRACT_PATH.read_bytes()
        generated = SimpleNamespace(
            source_mjcf_sha256=_sha("source-mjcf"),
            target_order_source_sha256=_sha("target-order"),
        )
        observed = []

        def timed_out(
            source_mjcf,
            projection_cli,
            child_environment,
            *,
            before_invoke,
        ):
            self.assertEqual(source_mjcf, inputs.source_mjcf)
            self.assertEqual(projection_cli, projection)
            self.assertEqual(child_environment, environment)
            before_invoke()
            observed.extend(
                dict(value)
                for value in operations._state(context)[
                    "active_gate_attempt"
                ]["processes"]
            )
            raise subprocess.TimeoutExpired(
                (str(projection_cli), str(cli_module._JOINT_CONTRACT_PATH)),
                90,
                output=b"projection partial stdout\n",
                stderr=b"projection partial stderr\n",
            )

        bundle = RunBundle.create(self.root / "runs", "joint-attempt", "run")
        try:
            with (
                patch.object(cli_module, "_SONIC_ROOT", sonic_root),
                patch.object(
                    cli_module, "generate_joint_contract", return_value=generated
                ),
                patch.object(
                    cli_module, "contract_json_bytes", return_value=committed
                ),
                patch.object(
                    cli_module, "_certify_joint_projection", side_effect=timed_out
                ),
            ):
                result = operations.run_gate(
                    "joint_projection_round_trip", request, context, bundle
                )
            self.assertEqual(result.status, "integration_failure")
            self.assertEqual(len(observed), 1)
            self.assertEqual(len(result.processes), 1)
            process = result.processes[0]
            self.assertEqual(process["name"], "g1_project_pose_cli")
            self.assertEqual(
                process["argv"],
                [str(projection), str(cli_module._JOINT_CONTRACT_PATH)],
            )
            self.assertEqual(process["environment"]["values"], environment)
            self.assertEqual(
                process["outputs"],
                [
                    "gates/joint_projection.stdout",
                    "gates/joint_projection.stderr",
                ],
            )
            self.assertEqual(
                (bundle.path / "gates/joint_projection.stdout").read_bytes(),
                b"projection partial stdout\n",
            )
            self.assertEqual(
                (bundle.path / "gates/joint_projection.stderr").read_bytes(),
                b"projection partial stderr\n",
            )
            self.assertEqual(
                set(result.outputs),
                {
                    "gates/joint_projection.stdout",
                    "gates/joint_projection.stderr",
                },
            )
        finally:
            bundle.__del__()

    def test_linkage_timeout_retains_attempt_before_subprocess_run(self) -> None:
        (
            sonic_root,
            inputs,
            context,
            operations,
            request,
            environment,
        ) = self._attempt_provenance_fixture()
        observed = []

        def timeout(*args, **kwargs):
            observed.extend(
                dict(value)
                for value in operations._state(context)[
                    "active_gate_attempt"
                ]["processes"]
            )
            raise subprocess.TimeoutExpired(
                args[0],
                kwargs["timeout"],
                output=b"linkage partial stdout\n",
                stderr=b"linkage partial stderr\n",
            )

        bundle = RunBundle.create(self.root / "runs", "linkage-attempt", "run")
        try:
            with (
                patch.object(cli_module, "_SONIC_ROOT", sonic_root),
                patch.object(
                    cli_module, "_find_unresolved_git_lfs_capabilities", return_value=()
                ),
                patch.object(
                    cli_module.shutil, "which", return_value=sys.executable
                ),
                patch.object(cli_module.subprocess, "run", side_effect=timeout),
            ):
                result = operations.run_gate(
                    "basis_and_scene_alignment", request, context, bundle
                )
            self.assertEqual(result.status, "not_run")
            self.assertEqual(len(observed), 1)
            self.assertEqual(len(result.processes), 1)
            process = result.processes[0]
            self.assertEqual(
                process["name"], "gear-cuda-tensorrt-linkage-preflight"
            )
            self.assertEqual(
                process["argv"],
                [str(Path(sys.executable).resolve()), str(inputs.gear_checkout / cli_module._GEAR_BINARY_RELATIVE)],
            )
            self.assertEqual(process["environment"]["values"], environment)
            self.assertEqual(
                process["outputs"],
                ["preflight/gear-linkage.stdout", "preflight/gear-linkage.stderr"],
            )
            self.assertEqual(
                (bundle.path / "preflight/gear-linkage.stdout").read_bytes(),
                b"linkage partial stdout\n",
            )
            self.assertEqual(
                (bundle.path / "preflight/gear-linkage.stderr").read_bytes(),
                b"linkage partial stderr\n",
            )
        finally:
            bundle.__del__()

    def test_import_nonzero_is_registered_before_subprocess_run(self) -> None:
        (
            sonic_root,
            _inputs,
            context,
            operations,
            request,
            environment,
        ) = self._attempt_provenance_fixture()
        observed = []
        linked = subprocess.CompletedProcess(
            args=(),
            returncode=0,
            stdout=(
                b"libnvinfer.so.10 => /opt/libnvinfer.so.10\n"
                b"libcudart.so.12 => /opt/libcudart.so.12\n"
            ),
            stderr=b"",
        )

        def import_failure(*_args, **_kwargs):
            observed.extend(
                dict(value)
                for value in operations._state(context)[
                    "active_gate_attempt"
                ]["processes"]
            )
            return subprocess.CompletedProcess(
                args=(), returncode=7, stdout=b"", stderr=b"import failed\n"
            )

        calls = 0

        def run_probe(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return linked
            return import_failure(*args, **kwargs)

        bundle = RunBundle.create(self.root / "runs", "import-attempt", "run")
        try:
            with (
                patch.object(cli_module, "_SONIC_ROOT", sonic_root),
                patch.object(
                    cli_module, "_find_unresolved_git_lfs_capabilities", return_value=()
                ),
                patch.object(
                    cli_module.shutil, "which", return_value=sys.executable
                ),
                patch.object(
                    cli_module.subprocess,
                    "run",
                    side_effect=run_probe,
                ),
                patch.object(cli_module, "_cuda_device_count", return_value=1),
            ):
                result = operations.run_gate(
                    "basis_and_scene_alignment", request, context, bundle
                )
            self.assertEqual(result.status, "not_run")
            import_attempts = [
                value
                for value in observed
                if value["name"] == "gated-simulator-import-preflight"
            ]
            self.assertEqual(len(import_attempts), 1)
            self.assertEqual(import_attempts[0]["environment"]["values"], environment)
            self.assertRegex(import_attempts[0]["module_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                import_attempts[0]["outputs"],
                [
                    "preflight/gated-sim-import.stdout",
                    "preflight/gated-sim-import.stderr",
                ],
            )
        finally:
            bundle.__del__()

    def test_mm_preflight_attempt_precedes_constructor_hello_and_reset(self) -> None:
        (
            sonic_root,
            _inputs,
            context,
            operations,
            request,
            _environment,
        ) = self._attempt_provenance_fixture()
        linked = subprocess.CompletedProcess(
            args=(),
            returncode=0,
            stdout=(
                b"libnvinfer.so.10 => /opt/libnvinfer.so.10\n"
                b"libcudart.so.12 => /opt/libcudart.so.12\n"
            ),
            stderr=b"",
        )
        imported = subprocess.CompletedProcess(
            args=(), returncode=0, stdout=b"imported\n", stderr=b""
        )
        observed: dict[str, list[dict[str, object]]] = {}

        def snapshot(label):
            observed[label] = [
                dict(value)
                for value in operations._state(context)[
                    "active_gate_attempt"
                ]["processes"]
            ]

        class FailingMMClient:
            def __init__(self, **_kwargs):
                snapshot("constructor")

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def hello(self):
                snapshot("hello")
                return _joint_feasibility_hello()

            def reset(self, *_args, **_kwargs):
                snapshot("reset")
                raise ProcessError("synthetic MM reset failure")

        bundle = RunBundle.create(self.root / "runs", "mm-attempt", "run")
        try:
            with (
                patch.object(cli_module, "_SONIC_ROOT", sonic_root),
                patch.object(
                    cli_module, "_find_unresolved_git_lfs_capabilities", return_value=()
                ),
                patch.object(
                    cli_module.shutil, "which", return_value=sys.executable
                ),
                patch.object(
                    cli_module.subprocess, "run", side_effect=[linked, imported]
                ),
                patch.object(cli_module, "_cuda_device_count", return_value=1),
                patch.object(cli_module, "MMChunkClient", FailingMMClient),
            ):
                result = operations.run_gate(
                    "basis_and_scene_alignment", request, context, bundle
                )
            self.assertEqual(result.status, "integration_failure")
            for label in ("constructor", "hello", "reset"):
                mm_attempts = [
                    value
                    for value in observed[label]
                    if value["name"] == "mm_chunk_server-preflight"
                ]
                self.assertEqual(len(mm_attempts), 1, label)
                self.assertEqual(
                    mm_attempts[0]["outputs"],
                    ["preflight/mm.stdout", "preflight/mm.stderr"],
                )
            self.assertEqual(
                [value["name"] for value in result.processes],
                [
                    "gear-cuda-tensorrt-linkage-preflight",
                    "gated-simulator-import-preflight",
                    "mm_chunk_server-preflight",
                ],
            )
        finally:
            bundle.__del__()

    def test_preflight_joint_feasibility_is_parsed_before_reset_and_retained(self):
        (
            sonic_root,
            _inputs,
            context,
            operations,
            request,
            _environment,
        ) = self._attempt_provenance_fixture()
        linked = subprocess.CompletedProcess(
            args=(),
            returncode=0,
            stdout=(
                b"libnvinfer.so.10 => /opt/libnvinfer.so.10\n"
                b"libcudart.so.12 => /opt/libcudart.so.12\n"
            ),
            stderr=b"",
        )
        imported = subprocess.CompletedProcess(
            args=(), returncode=0, stdout=b"imported\n", stderr=b""
        )
        scene_xml = self.root / "preflight-scene.xml"
        scene_xml.write_bytes(b"<mujoco/>\n")
        scene = cli_module.RegisteredScene(
            scene_id="sonic-flat-baseline",
            route_id="flat-12s",
            source_kind="analytic-flat",
            source_mesh=None,
            source_heightfield=None,
            source_hashes=MappingProxyType(
                {"manifest": _sha("manifest"), "scene_index": _sha("index")}
            ),
            coordinate_source="holden-y-up-right-handed-forward-plus-z",
            coordinate_target="mujoco-z-up-right-handed-forward-plus-x",
            transform_matrix=np.eye(3, dtype=np.float64),
            transformed_obj=None,
            gear_scene_xml=scene_xml,
            output_hashes=MappingProxyType(
                {
                    "gear_scene_xml": _sha("gear-scene"),
                    "scene_registration": _sha("scene-registration"),
                }
            ),
            allowed_foot_geoms=(1,),
            forbidden_geom_groups=MappingProxyType(
                {"pelvis": (2,), "knees": (3,), "torso": (4,), "hands": (5,)}
            ),
        )
        preflight = _joint_feasibility_identity()
        events: list[str] = []

        class FakeMMClient:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def hello(self):
                events.append("hello")
                return _joint_feasibility_hello(preflight)

            def reset(self, *_args, **_kwargs):
                events.append("reset")
                return {"scene": {}}

        bundle = RunBundle.create(self.root / "runs", "preflight-feasibility", "run")
        try:
            with (
                patch.object(cli_module, "_SONIC_ROOT", sonic_root),
                patch.object(
                    cli_module, "_find_unresolved_git_lfs_capabilities", return_value=()
                ),
                patch.object(cli_module.shutil, "which", return_value=sys.executable),
                patch.object(
                    cli_module.subprocess, "run", side_effect=[linked, imported]
                ),
                patch.object(cli_module, "_cuda_device_count", return_value=1),
                patch.object(cli_module, "MMChunkClient", FakeMMClient),
                patch.object(cli_module, "register_scene", return_value=scene),
                patch.object(
                    cli_module,
                    "_model_initial_state",
                    return_value=(
                        np.zeros(36, dtype=np.float64),
                        np.arange(29, dtype=np.int64),
                        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                    ),
                ),
            ):
                result = operations.run_gate(
                    "basis_and_scene_alignment", request, context, bundle
                )
            self.assertEqual(result.status, "pass")
            self.assertEqual(events, ["hello", "reset"])
            expected = parse_joint_feasibility_identity(preflight)
            self.assertEqual(
                operations._state(context)["joint_feasibility"], expected
            )
            expected_payload = _joint_feasibility_identity()
            self.assertEqual(
                result.metrics["joint_feasibility"], expected_payload
            )
            payload = json.loads(
                (bundle.path / "gates/basis_and_scene_alignment.json").read_text(
                    "ascii"
                )
            )
            self.assertEqual(payload["joint_feasibility"], expected_payload)
        finally:
            bundle.__del__()

    def test_mm_reference_attempt_precedes_constructor_hello_and_reset(self) -> None:
        (
            sonic_root,
            _inputs,
            context,
            operations,
            request,
            _environment,
        ) = self._attempt_provenance_fixture()
        scene_xml = self.root / "registered-scene.xml"
        scene_xml.write_bytes(b"<mujoco/>\n")
        operations._runtime[id(context)]["scene"] = cli_module.RegisteredScene(
            scene_id="sonic-flat-baseline",
            route_id="flat-12s",
            source_kind="analytic-flat",
            source_mesh=None,
            source_heightfield=None,
            source_hashes=MappingProxyType({}),
            coordinate_source="holden-y-up-right-handed-forward-plus-z",
            coordinate_target="mujoco-z-up-right-handed-forward-plus-x",
            transform_matrix=np.eye(3, dtype=np.float64),
            transformed_obj=None,
            gear_scene_xml=scene_xml,
            output_hashes=MappingProxyType({}),
            allowed_foot_geoms=(),
            forbidden_geom_groups=MappingProxyType({}),
        )
        observed: dict[str, list[dict[str, object]]] = {}

        def snapshot(label):
            observed[label] = [
                dict(value)
                for value in operations._state(context)[
                    "active_gate_attempt"
                ]["processes"]
            ]

        class FailingMMClient:
            def __init__(self, **_kwargs):
                snapshot("constructor")

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def hello(self):
                snapshot("hello")
                return _joint_feasibility_hello()

            def reset(self, *_args, **_kwargs):
                snapshot("reset")
                raise ProcessError("synthetic reference reset failure")

        bundle = RunBundle.create(self.root / "runs", "mm-reference-attempt", "run")
        try:
            with (
                patch.object(cli_module, "_SONIC_ROOT", sonic_root),
                patch.object(cli_module, "load_joint_contract", return_value=object()),
                patch.object(cli_module, "SourceValidator", return_value=object()),
                patch.object(cli_module, "MMChunkClient", FailingMMClient),
            ):
                result = operations.run_gate(
                    "flat_mm_kinematic_replay", request, context, bundle
                )
            self.assertEqual(result.status, "integration_failure")
            for label in ("constructor", "hello", "reset"):
                attempts = [
                    value
                    for value in observed[label]
                    if value["name"] == "mm_chunk_server-flat-reference"
                ]
                self.assertEqual(len(attempts), 1, label)
                self.assertEqual(
                    attempts[0]["outputs"],
                    ["reference/mm.stdout", "reference/mm.stderr"],
                )
            self.assertEqual(
                [value["name"] for value in result.processes],
                ["mm_chunk_server-flat-reference"],
            )
        finally:
            bundle.__del__()

    @unittest.skipUnless(
        importlib.util.find_spec("zmq") is not None,
        "pyzmq is unavailable in this interpreter",
    )
    def test_stream_uses_resolved_ephemeral_loopback_endpoint_in_gear_argv(
        self,
    ) -> None:
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", 5556))
        blocker.listen(1)
        bundle = RunBundle.create(self.root / "runs", "endpoint", "run")
        publisher = None
        try:
            publisher = PosePublisher(
                "tcp://127.0.0.1:*",
                bundle=bundle,
            )
            host, port = cli_module._parse_local_zmq_endpoint(
                publisher.endpoint
            )
            command = cli_module._stream_gear_command(
                ("/authenticated/g1_deploy_onnx_ref", "lo"),
                publisher.endpoint,
            )
            self.assertEqual(host, "127.0.0.1")
            self.assertNotEqual(port, 5556)
            self.assertEqual(
                command[-4:],
                ("--zmq-host", host, "--zmq-port", str(port)),
            )
        finally:
            if publisher is not None:
                publisher.close()
            blocker.close()
            bundle.__del__()

    def test_known_good_initial_token_passes_real_coordinator_session_check(
        self,
    ) -> None:
        target_log = self.root / "target.csv"
        known_good = cli_module._KnownGoodTimeline(_canonical())
        timeline = SimpleNamespace(initial_buffer=known_good.initial_buffer)

        class Publisher:
            def prepare(self, buffer, *, phase, attempt):
                return SimpleNamespace(
                    buffer=buffer,
                    phase=phase,
                    attempt=attempt,
                )

            def send_prepared(self, prepared):
                target_log.write_bytes(
                    expected_official_target_row(prepared.buffer)
                )
                return {
                    "local_send_completed": True,
                    "phase": prepared.phase,
                }

            def close(self):
                pass

        class Stream:
            def start(self):
                pass

            def require_alive(self):
                pass

            def close(self):
                pass

        class Gate:
            is_paused = False

            def pause(self):
                self.is_paused = True

            def require_paused(self):
                if not self.is_paused:
                    raise AssertionError("gate was not paused")

            def close(self):
                pass

        class Evidence:
            def __init__(self):
                self.readiness = None
                self.terminal = None

            def write_readiness(self, value):
                self.readiness = value

            def write_terminal_verdict(self, value):
                self.terminal = value

        evidence = Evidence()
        coordinator = Coordinator(
            mm=cli_module._KnownGoodMM(),
            validator=cli_module._KnownGoodValidator(),
            timeline_factory=lambda _initial: timeline,
            run=evidence,
            publisher=Publisher(),
            stream=Stream(),
            gate=Gate(),
            target_motion_logfile=target_log,
            steps_per_chunk=80,
            session_id_factory=lambda: "known-good-session",
            readiness_wait=lambda: None,
        )
        try:
            result = coordinator.preflight(
                SessionConfig(
                    "sonic-flat-baseline",
                    "flat-12s",
                    0.0,
                )
            )
            self.assertEqual(result.session_id, "known-good-session")
            self.assertEqual(evidence.readiness, result)
        finally:
            coordinator.close()

    def test_bootstrap_waits_for_cold_load_then_enforces_sim_step_cap(
        self,
    ) -> None:
        startup_ready = threading.Event()
        operation_done = threading.Event()

        class Simulator:
            sim_dt = 0.01

            def __init__(self):
                self.steps = 0

            def advance(self, steps):
                if not startup_ready.is_set():
                    raise AssertionError("physics advanced during cold load")
                self.steps += steps
                operation_done.set()

        simulator = Simulator()

        def operation():
            time.sleep(0.02)
            startup_ready.set()
            operation_done.wait(1.0)

        advanced = cli_module._drive_simulator_until(
            operation,
            simulator,
            label="phased-test",
            ready_for_bootstrap=startup_ready.is_set,
            cold_start_maximum_seconds=0.2,
            maximum_seconds=0.05,
        )
        self.assertEqual((advanced, simulator.steps), (5, 5))

        blocked = threading.Event()
        cancelled = threading.Event()
        capped = Simulator()
        startup_ready.set()
        try:
            with self.assertRaisesRegex(ProcessError, "bootstrap"):
                cli_module._drive_simulator_until(
                    lambda: (
                        cancelled.wait(1.0),
                        blocked.set(),
                    ),
                    capped,
                    label="bounded-test",
                    ready_for_bootstrap=lambda: True,
                    cold_start_maximum_seconds=0.2,
                    maximum_seconds=0.05,
                    cancellation=cancelled,
                )
        finally:
            blocked.set()
        self.assertTrue(cancelled.is_set())
        self.assertLessEqual(capped.steps, 5)

    def test_bootstrap_timeout_joins_start_worker_across_popen_races(self) -> None:
        class Simulator:
            sim_dt = 0.01

            def advance(self, _steps):
                raise AssertionError("cold startup must not advance physics")

        def make_gear(stem, command, cancellation):
            return GearProcess(
                run_root=self.root,
                command=command,
                target_motion_logfile=self.root / f"{stem}.target",
                logs_dir=self.root / f"{stem}.logs",
                stdout_archive=self.root / f"{stem}.stdout",
                stderr_archive=self.root / f"{stem}.stderr",
                startup_markers=("NEVER READY",),
                active_markers=("CONTROL", "STREAM"),
                cancelled=cancellation.is_set,
                readiness_timeout_s=1.0,
                readiness_poll_s=0.005,
                stop_grace_s=0.02,
                term_grace_s=0.02,
                kill_grace_s=0.1,
            )

        before_publication_cancel = threading.Event()
        before = make_gear(
            "before-popen",
            [sys.executable, "-c", "raise SystemExit(0)"],
            before_publication_cancel,
        )

        def blocked_popen(*_args, **_kwargs):
            before_publication_cancel.wait(1.0)
            raise OSError("cancelled before Popen publication")

        with patch("mm_sonic.process.subprocess.Popen", side_effect=blocked_popen):
            with self.assertRaisesRegex(ProcessError, "cold startup"):
                cli_module._drive_simulator_until(
                    before.start,
                    Simulator(),
                    label="before-popen-race",
                    ready_for_bootstrap=lambda: False,
                    cold_start_maximum_seconds=0.03,
                    maximum_seconds=0.03,
                    cancellation=before_publication_cancel,
                )
        self.assertTrue(before_publication_cancel.is_set())

        child = self.root / "after-popen.py"
        child.write_text(
            "import time\nwhile True:\n    time.sleep(1.0)\n",
            encoding="utf-8",
        )
        after_publication_cancel = threading.Event()
        after = make_gear(
            "after-popen",
            [sys.executable, "-u", str(child)],
            after_publication_cancel,
        )
        with self.assertRaisesRegex(ProcessError, "cold startup"):
            cli_module._drive_simulator_until(
                after.start,
                Simulator(),
                label="after-popen-race",
                ready_for_bootstrap=lambda: False,
                cold_start_maximum_seconds=0.03,
                maximum_seconds=0.03,
                cancellation=after_publication_cancel,
            )
        self.assertTrue(after_publication_cancel.is_set())
        self.assertIsNotNone(after.returncode)
        from mm_sonic import process as process_module

        self.assertEqual(process_module._linux_group_states(after.pgid), {})
        self.assertFalse(
            any(
                thread.name.startswith("gear-")
                and thread.name.endswith("bootstrap")
                for thread in threading.enumerate()
            )
        )

    def test_loaded_file_target_requires_the_exact_zero_based_prefix(self) -> None:
        canonical = _canonical(3)
        rows = tuple(
            cli_module._official_target_row(canonical, index)
            for index in range(canonical.count)
        )
        self.assertEqual(cli_module._validate_target_prefix(b"", rows), 0)
        self.assertEqual(
            cli_module._validate_target_prefix(b"".join(rows), rows), 3
        )
        with self.assertRaisesRegex(ContractError, "canonical prefix"):
            cli_module._validate_target_prefix(rows[0] + rows[2], rows)
        with self.assertRaisesRegex(ContractError, "overshoot"):
            cli_module._validate_target_prefix(b"".join(rows + rows[-1:]), rows)

    def _remote_generation_error_case(
        self,
        *,
        code: str,
        message: str,
        label: str,
        run_identity: dict[str, object] | None = None,
        remote_error: bool = True,
        stderr_message: str = "",
    ) -> tuple[
        GateResult,
        dict[str, object] | None,
        tuple[str, ...],
        tuple[str, ...],
    ]:
        case_root = self.root / label
        case_root.mkdir()
        sonic_root = case_root / "sonic"
        mm_server = sonic_root / "build/mm_chunk_server"
        mm_server.parent.mkdir(parents=True)
        mm_server.write_bytes(b"authenticated mm server")
        mm_server.chmod(0o755)

        gear = case_root / "gear"
        terrain = case_root / "terrain"
        gear.mkdir()
        terrain.mkdir()
        policy = case_root / "policy.onnx"
        observation = case_root / "observation.yaml"
        source = case_root / "source.xml"
        policy.write_bytes(b"policy")
        observation.write_bytes(b"observations: []\n")
        source.write_bytes(b"<mujoco/>\n")
        inputs = ExternalInputs(
            gear_checkout=gear.resolve(),
            policy=policy.resolve(),
            observation_config=observation.resolve(),
            encoder=None,
            terrain_dir=terrain.resolve(),
            source_mjcf=source.resolve(),
        )
        context = StageAContext(
            gear_commit="1" * 40,
            gear_dirty=False,
            external_hashes=MappingProxyType({}),
            motion_matching_commit="2" * 40,
            motion_matching_dirty=False,
            artifact_hashes=MappingProxyType({}),
            known_good_reference=gear,
            processes=(),
        )
        scene_xml = case_root / "scene.xml"
        scene_xml.write_bytes(b"<mujoco/>\n")
        scene = cli_module.RegisteredScene(
            scene_id="sonic-flat-baseline",
            route_id="flat-12s",
            source_kind="analytic-flat",
            source_mesh=None,
            source_heightfield=None,
            source_hashes=MappingProxyType({}),
            coordinate_source="holden-y-up-right-handed-forward-plus-z",
            coordinate_target="mujoco-z-up-right-handed-forward-plus-x",
            transform_matrix=np.eye(3, dtype=np.float64),
            transformed_obj=None,
            gear_scene_xml=scene_xml,
            output_hashes=MappingProxyType({}),
            allowed_foot_geoms=(),
            forbidden_geom_groups=MappingProxyType({}),
        )
        initial = SimpleNamespace(
            physical_pelvis_position_holden=np.zeros(3, dtype=np.float32),
            virtual_root_position_holden=np.zeros(3, dtype=np.float32),
            virtual_root_orientation_holden=np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32
            ),
        )

        class FakeTimeline:
            def __init__(self, _initial, _contract):
                self.last_accepted_candidate_id = None
                self._count = 1

            @property
            def canonical_buffer(self):
                return SimpleNamespace(count=self._count)

            def prepare(self, value):
                return value

            def commit(self, value):
                self.last_accepted_candidate_id = value.candidate_id
                self._count += 20
                return SimpleNamespace(
                    physical_pelvis_position=np.zeros(
                        (20, 3), dtype=np.float32
                    ),
                    virtual_root_position=np.zeros((20, 3), dtype=np.float32),
                    virtual_root_quat_w=np.tile(
                        np.array(
                            [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32
                        ),
                        (20, 1),
                    ),
                )

        class FakeValidator:
            def validate_initial(self, _value):
                return initial

            def validate_source(self, value):
                return SimpleNamespace(candidate_id=value["candidate_id"])

        preflight_identity = _joint_feasibility_identity()
        run_feasibility = (
            _joint_feasibility_identity()
            if run_identity is None
            else run_identity
        )
        aborted: list[str] = []
        events: list[str] = []

        class FakeMMClient:
            def __init__(
                self, *, stdout_archive, stderr_archive, **_kwargs
            ):
                Path(stdout_archive).write_bytes(b"reset accepted\n")
                Path(stderr_archive).write_text(stderr_message, encoding="ascii")
                self.outstanding_candidate_id = None

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def hello(self):
                events.append("hello")
                return _joint_feasibility_hello(run_feasibility)

            def reset(self, _config, *, session_id):
                events.append("reset")
                self.session_id = session_id
                return {"scene": {}}

            def generate(self, _command, *, candidate_id, **_kwargs):
                events.append("generate")
                chunk = int(candidate_id.rsplit(":", 1)[1])
                if chunk == 5:
                    self.outstanding_candidate_id = None
                    if remote_error:
                        raise _RemoteMMError(code, message)
                    raise ProcessError(message)
                self.outstanding_candidate_id = candidate_id
                return {"candidate_id": candidate_id}

            def commit(self, candidate_id):
                if self.outstanding_candidate_id != candidate_id:
                    raise AssertionError("fake candidate ownership changed")
                self.outstanding_candidate_id = None

            def abort(self, candidate_id):
                aborted.append(candidate_id)
                self.outstanding_candidate_id = None

        contract = SimpleNamespace(
            rows=(
                SimpleNamespace(
                    source_joint="left_ankle_roll_joint",
                    lower=-0.2618,
                    upper=0.2618,
                ),
            )
        )
        operations = cli_module.DefaultStageAOperations()
        operations._runtime[id(context)] = {
            "scene": scene,
            "verified_external": SimpleNamespace(inputs=inputs),
            "joint_feasibility": parse_joint_feasibility_identity(
                preflight_identity
            ),
        }
        request = StageARequest(
            command="stage-a",
            mode="mm-reference",
            argv=(),
            namespace=argparse.Namespace(),
            environment=MappingProxyType({}),
        )
        bundle = RunBundle.create(case_root / "runs", "stage-a", label)
        try:
            with (
                patch.object(cli_module, "_SONIC_ROOT", sonic_root),
                patch.object(
                    cli_module, "load_joint_contract", return_value=contract
                ),
                patch.object(
                    cli_module, "SourceValidator", return_value=FakeValidator()
                ),
                patch.object(cli_module, "MMChunkClient", FakeMMClient),
                patch.object(cli_module, "verify_mm_scene_identity"),
                patch.object(cli_module, "TargetTimeline", FakeTimeline),
            ):
                result = operations.run_gate(
                    "flat_mm_kinematic_replay", request, context, bundle
                )
            evidence_path = bundle.path / "gates/flat_mm_kinematic_replay.json"
            evidence = (
                json.loads(evidence_path.read_text("ascii"))
                if evidence_path.exists()
                else None
            )
            return result, evidence, tuple(aborted), tuple(events)
        finally:
            bundle.__del__()

    def test_remote_joint_limit_failure_is_scientific_at_source_chunk(self):
        result, evidence, aborted, _events = self._remote_generation_error_case(
            code="generation_failed",
            message=(
                "joint left_ankle_roll_joint position -0.307408422 is outside "
                "range [-0.261799991, 0.261799991]"
            ),
            label="remote-joint-limit",
        )
        self.assertEqual(result.status, "scientific_failure")
        self.assertEqual(aborted, ())
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(
            {
                key: evidence[key]
                for key in (
                    "status",
                    "failed_chunk",
                    "failure_boundary",
                    "initial_boundary_accepted",
                    "completed_chunks",
                    "partial_frame_count",
                    "required_chunks",
                    "required_frame_count",
                )
            },
            {
                "status": "scientific_failure",
                "failed_chunk": 5,
                "failure_boundary": "source_chunk",
                "initial_boundary_accepted": True,
                "completed_chunks": 5,
                "partial_frame_count": 101,
                "required_chunks": 30,
                "required_frame_count": 601,
            },
        )

    def test_remote_joint_limit_lookalikes_remain_integration_failures(self):
        limit = (
            "joint left_ankle_roll_joint position -0.307408422 is outside "
            "range [-0.261799991, 0.261799991]"
        )
        cases = (
            ("generation_failed", "database runtime invariant failed"),
            ("invalid_candidate", limit),
            (
                "generation_failed",
                "joint invented_joint position -0.307408422 is outside range "
                "[-0.261799991, 0.261799991]",
            ),
            ("generation_failed", limit + " trailing"),
            (
                "generation_failed",
                "joint left_ankle_roll_joint position -0.2 is outside range "
                "[-0.261799991, 0.261799991]",
            ),
        )
        for index, (code, message) in enumerate(cases):
            with self.subTest(code=code, message=message):
                result, evidence, aborted, _events = self._remote_generation_error_case(
                    code=code,
                    message=message,
                    label=f"remote-lookalike-{index}",
                )
                self.assertEqual(result.status, "integration_failure")
                self.assertIsNone(evidence)
                self.assertEqual(aborted, ())

    def test_exact_remote_no_safe_candidate_is_scientific(self):
        message = "no joint-limit-safe database candidate"
        result, evidence, aborted, events = self._remote_generation_error_case(
            code="generation_failed",
            message=message,
            label="remote-no-safe-candidate",
        )
        self.assertEqual(result.status, "scientific_failure")
        self.assertEqual(aborted, ())
        self.assertEqual(events[:2], ("hello", "reset"))
        self.assertEqual(events.count("generate"), 6)
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(evidence["failure_boundary"], "source_chunk")
        self.assertIn(message, evidence["reason"])

    def test_no_safe_candidate_text_lookalikes_are_integration_failures(self):
        message = "no joint-limit-safe database candidate"
        cases = (
            ("invalid_candidate", message, True, ""),
            ("generation_failed", "prefix " + message, True, ""),
            ("generation_failed", message + " suffix", True, ""),
            ("generation_failed", message, False, ""),
            ("generation_failed", "child process exited", False, message),
        )
        for index, (code, observed, remote, stderr) in enumerate(cases):
            with self.subTest(code=code, observed=observed, remote=remote):
                result, evidence, aborted, _events = (
                    self._remote_generation_error_case(
                        code=code,
                        message=observed,
                        label=f"remote-no-safe-lookalike-{index}",
                        remote_error=remote,
                        stderr_message=stderr,
                    )
                )
                self.assertEqual(result.status, "integration_failure")
                self.assertIsNone(evidence)
                self.assertEqual(aborted, ())

    def test_exact_remote_no_inertialized_safe_candidate_is_scientific(self):
        message = "no inertialized-joint-safe database candidate"
        result, evidence, aborted, events = self._remote_generation_error_case(
            code="generation_failed",
            message=message,
            label="remote-no-inertialized-safe-candidate",
        )
        self.assertEqual(result.status, "scientific_failure")
        self.assertEqual(aborted, ())
        self.assertEqual(events[:2], ("hello", "reset"))
        self.assertEqual(events.count("generate"), 6)
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(evidence["failure_boundary"], "source_chunk")
        self.assertIn(message, evidence["reason"])

    def test_no_inertialized_safe_candidate_lookalikes_are_integration_failures(self):
        message = "no inertialized-joint-safe database candidate"
        cases = (
            ("invalid_candidate", message, True, ""),
            ("generation_failed", "prefix " + message, True, ""),
            ("generation_failed", message + " suffix", True, ""),
            ("generation_failed", message.upper(), True, ""),
            (
                "generation_failed",
                "no inertialized_joint_safe database candidate",
                True,
                "",
            ),
            ("generation_failed", message, False, ""),
            ("generation_failed", "child process exited", False, message),
        )
        for index, (code, observed, remote, stderr) in enumerate(cases):
            with self.subTest(code=code, observed=observed, remote=remote):
                result, evidence, aborted, _events = (
                    self._remote_generation_error_case(
                        code=code,
                        message=observed,
                        label=f"remote-no-inertialized-safe-lookalike-{index}",
                        remote_error=remote,
                        stderr_message=stderr,
                    )
                )
                self.assertEqual(result.status, "integration_failure")
                self.assertIsNone(evidence)
                self.assertEqual(aborted, ())

    def test_run_feasibility_drift_fails_before_reset_or_chunk_acceptance(self):
        count_drift = _joint_feasibility_identity()
        count_drift["frame_count"] += 1
        count_drift["raw_safe_count"] += 1

        vector_drift = _joint_feasibility_identity()
        vector_drift["joint_limit_violation_count"][5] -= 1
        vector_drift["joint_limit_violation_count"][11] += 1

        digest_drift = _joint_feasibility_identity(mask_sha256="b" * 64)
        malformed = _joint_feasibility_identity()
        malformed["search_safe_count"] = 0
        for label, identity in (
            ("count", count_drift),
            ("vector", vector_drift),
            ("digest", digest_drift),
            ("malformed", malformed),
        ):
            with self.subTest(label=label):
                result, evidence, aborted, events = (
                    self._remote_generation_error_case(
                        code="generation_failed",
                        message="no joint-limit-safe database candidate",
                        label=f"run-feasibility-drift-{label}",
                        run_identity=identity,
                    )
                )
                self.assertEqual(result.status, "integration_failure")
                self.assertIsNone(evidence)
                self.assertEqual(aborted, ())
                self.assertEqual(events, ("hello",))

    def test_initial_joint_limit_failure_is_scientific_before_chunk_zero(
        self,
    ) -> None:
        sonic_root = self.root / "sonic"
        mm_server = sonic_root / "build/mm_chunk_server"
        mm_server.parent.mkdir(parents=True)
        mm_server.write_bytes(b"authenticated mm server")
        mm_server.chmod(0o755)

        gear = self.root / "gear"
        gear.mkdir()
        policy = self.root / "policy.onnx"
        policy.write_bytes(b"policy")
        observation = self.root / "observation.yaml"
        observation.write_bytes(b"observations: []\n")
        terrain = self.root / "terrain"
        terrain.mkdir()
        source = self.root / "source.xml"
        source.write_bytes(b"<mujoco/>\n")
        inputs = ExternalInputs(
            gear_checkout=gear.resolve(),
            policy=policy.resolve(),
            observation_config=observation.resolve(),
            encoder=None,
            terrain_dir=terrain.resolve(),
            source_mjcf=source.resolve(),
        )
        context = StageAContext(
            gear_commit="1" * 40,
            gear_dirty=False,
            external_hashes=MappingProxyType({}),
            motion_matching_commit="2" * 40,
            motion_matching_dirty=False,
            artifact_hashes=MappingProxyType({}),
            known_good_reference=gear,
            processes=(),
        )
        scene_xml = self.root / "scene.xml"
        scene_xml.write_bytes(b"<mujoco/>\n")
        scene = cli_module.RegisteredScene(
            scene_id="sonic-flat-baseline",
            route_id="flat-12s",
            source_kind="analytic-flat",
            source_mesh=None,
            source_heightfield=None,
            source_hashes=MappingProxyType({}),
            coordinate_source="holden-y-up-right-handed-forward-plus-z",
            coordinate_target="mujoco-z-up-right-handed-forward-plus-x",
            transform_matrix=np.eye(3, dtype=np.float64),
            transformed_obj=None,
            gear_scene_xml=scene_xml,
            output_hashes=MappingProxyType({}),
            allowed_foot_geoms=(),
            forbidden_geom_groups=MappingProxyType({}),
        )
        generated = False

        class FakeMMClient:
            outstanding_candidate_id = None

            def __init__(self, *, stdout_archive, stderr_archive, **_kwargs):
                Path(stdout_archive).write_bytes(b"reset accepted\n")
                Path(stderr_archive).write_bytes(b"")

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def hello(self):
                return _joint_feasibility_hello()

            def reset(self, _config, *, session_id):
                self.session_id = session_id
                return {"scene": {}}

            def generate(self, *_args, **_kwargs):
                nonlocal generated
                generated = True
                raise AssertionError("chunk generation crossed the initial boundary")

        operations = cli_module.DefaultStageAOperations()
        operations._runtime[id(context)] = {
            "scene": scene,
            "verified_external": SimpleNamespace(inputs=inputs),
            "joint_feasibility": parse_joint_feasibility_identity(
                _joint_feasibility_identity()
            ),
        }
        bundle = RunBundle.create(self.root / "runs", "initial-limit", "run")
        request = StageARequest(
            command="stage-a",
            mode="mm-reference",
            argv=(),
            namespace=argparse.Namespace(),
            environment=MappingProxyType({}),
        )
        validator = SimpleNamespace(validate_initial=lambda _reset: object())
        try:
            with (
                patch.object(cli_module, "_SONIC_ROOT", sonic_root),
                patch.object(cli_module, "load_joint_contract", return_value=object()),
                patch.object(cli_module, "SourceValidator", return_value=validator),
                patch.object(cli_module, "MMChunkClient", FakeMMClient),
                patch.object(cli_module, "verify_mm_scene_identity"),
                patch.object(
                    cli_module,
                    "TargetTimeline",
                    side_effect=ContractError("initial joint limit exceeded"),
                ),
            ):
                result = operations.run_gate(
                    "flat_mm_kinematic_replay", request, context, bundle
                )
            self.assertEqual(result.status, "scientific_failure")
            self.assertFalse(generated)
            evidence = json.loads(
                (bundle.path / "gates/flat_mm_kinematic_replay.json").read_text(
                    "ascii"
                )
            )
            self.assertEqual(evidence["failed_chunk"], 0)
            self.assertEqual(evidence["failure_boundary"], "initial_boundary")
            self.assertFalse(evidence["initial_boundary_accepted"])
            self.assertEqual(evidence["completed_chunks"], 0)
            self.assertEqual(evidence["partial_frame_count"], 0)
            cli_module._validate_gate_result(result, bundle)
        finally:
            bundle.__del__()

    def test_known_good_decode_rejects_values_the_stream_cannot_carry(self) -> None:
        reference = self.root / "reference"
        reference.mkdir()

        def write_csv(name, width, value="0"):
            row = ",".join([value] * width)
            (reference / name).write_text(
                "header\n" + "\n".join([row] * 21) + "\n",
                encoding="ascii",
            )

        write_csv("joint_pos.csv", 29)
        write_csv("joint_vel.csv", 29)
        quat_pair = "1,0,0,0,1,0,0,0"
        (reference / "body_quat.csv").write_text(
            "header\n" + "\n".join([quat_pair] * 21) + "\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(ContractError, "root-only"):
            cli_module._decode_known_good_reference(reference)

        quaternion = "1,0,0,0"
        (reference / "body_quat.csv").write_text(
            "header\n" + "\n".join([quaternion] * 21) + "\n",
            encoding="ascii",
        )
        canonical = cli_module._decode_known_good_reference(reference)
        write_csv("body_pos.csv", 6)
        with self.assertRaisesRegex(ContractError, "root-only"):
            cli_module._decode_known_good_body_position(
                reference, canonical.count
            )

        write_csv("body_pos.csv", 3, value="0.1")
        body_position = cli_module._decode_known_good_body_position(
            reference, canonical.count
        )
        with self.assertRaisesRegex(ContractError, "body positions"):
            cli_module._require_streamable_body_position(body_position)

    def _lfs_runtime_boundary_fixture(self):
        pointer = (
            b"version https://git-lfs.github.com/spec/v1\n"
            + b"oid sha256:"
            + b"0" * 64
            + b"\nsize 123\n"
        )
        root = self.root / "lfs-runtime-boundary"
        gear = root / "gear"
        terrain = root / "terrain"
        gear.mkdir(parents=True)
        terrain.mkdir()
        policy = root / "explicit/policy.onnx"
        observation = root / "explicit/observation.yaml"
        source = root / "explicit/source.xml"
        encoder = root / "explicit/encoder.onnx"

        lock = json.loads(cli_module._LOCK_PATH.read_text(encoding="utf-8"))
        locked_sources = tuple(
            gear / lock[key]
            for key in (
                "joint_names_source",
                "policy_parameters_source",
                "zmq_example_source",
                "zmq_decoder_source",
                "stream_merger_source",
                "current_frame_advancement_source",
            )
        )
        known_good = tuple(
            gear / lock["known_good_reference"] / name
            for name in cli_module.KNOWN_GOOD_REFERENCE_FILES
        )
        official_runtime = (
            gear / cli_module._GEAR_MODEL_RELATIVE,
            gear / cli_module._GEAR_ROBOT_RELATIVE,
            gear / cli_module._GEAR_BINARY_RELATIVE,
            gear / "gear_sonic_deploy/g1/meshes/nested/runtime-mesh.STL",
        )
        terrain_payload = terrain / "nested/heightfield.bin"
        required = (
            policy,
            observation,
            source,
            encoder,
            *locked_sources,
            *known_good,
            *official_runtime,
            terrain_payload,
        )
        for target in required:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"resolved runtime payload\n")
        inputs = ExternalInputs(
            gear_checkout=gear,
            policy=policy,
            observation_config=observation,
            encoder=encoder,
            terrain_dir=terrain,
            source_mjcf=source,
        )
        return inputs, required, pointer

    def test_lfs_capability_scan_covers_exact_runtime_boundary(self) -> None:
        inputs, required, pointer = self._lfs_runtime_boundary_fixture()
        self.assertEqual(len(required), 23)

        for target in required:
            with self.subTest(target=target):
                target.write_bytes(pointer)
                try:
                    self.assertEqual(
                        cli_module._find_unresolved_git_lfs_capabilities(inputs),
                        (target,),
                    )
                finally:
                    target.write_bytes(b"resolved runtime payload\n")

        self.assertEqual(
            cli_module._find_unresolved_git_lfs_capabilities(
                replace(inputs, encoder=None)
            ),
            (),
        )

        duplicated = required[4]
        duplicated.write_bytes(pointer)
        try:
            self.assertEqual(
                cli_module._find_unresolved_git_lfs_capabilities(
                    replace(inputs, policy=duplicated, encoder=None)
                ),
                (duplicated,),
            )
        finally:
            duplicated.write_bytes(b"resolved runtime payload\n")

        simultaneous = (required[0], required[-1])
        for target in simultaneous:
            target.write_bytes(pointer)
        try:
            self.assertEqual(
                cli_module._find_unresolved_git_lfs_capabilities(inputs),
                simultaneous,
            )
        finally:
            for target in simultaneous:
                target.write_bytes(b"resolved runtime payload\n")

    def test_lfs_capability_scan_excludes_unrelated_gear_payloads(self) -> None:
        inputs, _required, pointer = self._lfs_runtime_boundary_fixture()
        gear = inputs.gear_checkout
        excluded = (
            gear / "datasets/unrelated-corpus.bin",
            gear / "checkpoints/excluded-training.ckpt",
            gear / "gear_sonic_deploy/policy/training/excluded-policy.onnx",
            gear / "gear_sonic_deploy/g1/images/preview.png",
            gear / ".git/objects/not-a-runtime-payload",
        )
        for target in excluded:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(pointer)

        self.assertEqual(
            cli_module._find_unresolved_git_lfs_capabilities(inputs),
            (),
        )
        explicit_training_policy = replace(inputs, policy=excluded[2], encoder=None)
        self.assertEqual(
            cli_module._find_unresolved_git_lfs_capabilities(
                explicit_training_policy
            ),
            (excluded[2],),
        )

    def test_lfs_capability_scan_is_symlink_safe_and_fail_closed(self) -> None:
        inputs, _required, pointer = self._lfs_runtime_boundary_fixture()
        outside_file = self.root / "outside-pointer.bin"
        outside_file.write_bytes(pointer)
        outside_directory = self.root / "outside-directory"
        outside_directory.mkdir()
        (outside_directory / "nested-pointer.bin").write_bytes(pointer)
        meshes = inputs.gear_checkout / "gear_sonic_deploy/g1/meshes"
        (meshes / "linked-file.bin").symlink_to(outside_file)
        (meshes / "linked-directory").symlink_to(
            outside_directory,
            target_is_directory=True,
        )
        self.assertEqual(
            cli_module._find_unresolved_git_lfs_capabilities(inputs),
            (),
        )

        def denied_walk(root, *, topdown, onerror, followlinks):
            self.assertIn(root, (meshes, inputs.terrain_dir))
            self.assertTrue(topdown)
            self.assertFalse(followlinks)
            onerror(PermissionError(f"synthetic traversal denial: {root}"))
            return ()

        with patch.object(cli_module.os, "walk", side_effect=denied_walk):
            with self.assertRaisesRegex(
                cli_module.CapabilityUnavailable,
                "cannot scan official GEAR mesh assets",
            ):
                cli_module._find_unresolved_git_lfs_capabilities(inputs)

    def test_authenticated_455_by_14_source_projects_to_explicit_441_by_1_bundle(
        self,
    ) -> None:
        source = self.root / "official-known-good"
        source.mkdir()
        expected = _write_pinned_known_good_fixture(source)
        source_hashes = {
            f"gear:known_good_reference/{name}": hashlib.sha256(
                (source / name).read_bytes()
            ).hexdigest()
            for name in cli_module.KNOWN_GOOD_REFERENCE_FILES
        }
        context = StageAContext(
            gear_commit="1" * 40,
            gear_dirty=False,
            external_hashes=MappingProxyType(source_hashes),
            motion_matching_commit="2" * 40,
            motion_matching_dirty=False,
            artifact_hashes=MappingProxyType({}),
            known_good_reference=source,
            processes=(),
        )
        bundle = RunBundle.create(self.root / "runs", "projection", "run")
        try:
            projection = cli_module.DefaultStageAOperations()._materialize_known_good_projection(
                context, bundle
            )
            canonical = cli_module._decode_known_good_reference(
                projection.reference_leaf
            )
            body_position = cli_module._decode_known_good_body_position(
                projection.reference_leaf, canonical.count
            )
            self.assertEqual(canonical.count, 441)
            np.testing.assert_array_equal(
                canonical.joint_position.view(np.uint32),
                expected["joint_position"][:441].view(np.uint32),
            )
            np.testing.assert_array_equal(
                canonical.joint_velocity.view(np.uint32),
                expected["joint_velocity"][:441].view(np.uint32),
            )
            np.testing.assert_array_equal(
                canonical.body_quat_w.view(np.uint32),
                expected["body_quaternion"][:441, 0].view(np.uint32),
            )
            self.assertEqual(body_position.shape, (441, 3))
            self.assertTrue(np.all(body_position.view(np.uint32) == 0))
            projected_linear = cli_module._read_numeric_csv(
                projection.reference_leaf / "body_lin_vel.csv", columns=3
            )
            projected_angular = cli_module._read_numeric_csv(
                projection.reference_leaf / "body_ang_vel.csv", columns=3
            )
            self.assertTrue(
                np.any(
                    expected["body_linear_velocity"][:441, 0].view(np.uint32)
                    != 0
                )
            )
            self.assertTrue(
                np.any(
                    expected["body_angular_velocity"][:441, 0].view(np.uint32)
                    != 0
                )
            )
            self.assertTrue(np.all(projected_linear.view(np.uint32) == 0))
            self.assertTrue(np.all(projected_angular.view(np.uint32) == 0))
            positive_zero_sha256 = hashlib.sha256(bytes(441 * 3 * 4)).hexdigest()
            self.assertEqual(
                projection.evidence["body_position_sha256"],
                positive_zero_sha256,
            )
            self.assertEqual(
                projection.evidence["body_linear_velocity_sha256"],
                positive_zero_sha256,
            )
            self.assertEqual(
                projection.evidence["body_angular_velocity_sha256"],
                positive_zero_sha256,
            )
            expected_linear_csv = (
                b"body_0_vel_x,body_0_vel_y,body_0_vel_z\n"
                + b"0,0,0\n" * 441
            )
            expected_angular_csv = (
                b"body_0_angvel_x,body_0_angvel_y,body_0_angvel_z\n"
                + b"0,0,0\n" * 441
            )
            self.assertEqual(
                projection.evidence["projected_files_sha256"][
                    "body_lin_vel.csv"
                ],
                hashlib.sha256(expected_linear_csv).hexdigest(),
            )
            self.assertEqual(
                projection.evidence["projected_files_sha256"][
                    "body_ang_vel.csv"
                ],
                hashlib.sha256(expected_angular_csv).hexdigest(),
            )
            self.assertIn("[0]", (projection.reference_leaf / "metadata.txt").read_text("ascii"))
            self.assertEqual(
                projection.evidence["source"],
                {"frame_count": 455, "body_count": 14, "sha256": source_hashes},
            )
            self.assertEqual(
                projection.evidence["selection"],
                {
                    "frame_range": [0, 440],
                    "frame_count": 441,
                    "body_indexes": [0],
                    "omitted_tail_range": [441, 454],
                    "omitted_tail_count": 14,
                },
            )
            self.assertRegex(projection.evidence["rule_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(projection.evidence["projection_sha256"], r"^[0-9a-f]{64}$")
        finally:
            bundle.__del__()

    def test_projection_registry_declares_cold_async_preload_contract(self) -> None:
        rule = dict(cli_module._KNOWN_GOOD_PROJECTION_RULE)
        self.assertEqual(rule["control_clock"], "wall-clock-50hz-asynchronous-to-simulator")
        self.assertEqual(rule["stream_readiness_frame_range"], [0, 0])
        self.assertEqual(rule["stream_readiness_publication_count"], 1)
        self.assertEqual(rule["stream_logical_frame_range"], [1, 440])
        self.assertEqual(rule["stream_logical_publication_count"], 22)
        self.assertEqual(rule["stream_padding_frame_range"], [441, 486])
        self.assertEqual(rule["stream_padding_publication_count"], 1)
        self.assertEqual(rule["stream_receipt_fence_frame_range"], [487, 487])
        self.assertEqual(rule["stream_receipt_fence_publication_count"], 1)
        self.assertEqual(rule["stream_post_enable_fence_key_sequence"], "qe")
        self.assertEqual(
            rule["stream_post_enable_fence_left_line"],
            "Delta heading left: 0.1 rad",
        )
        self.assertEqual(
            rule["stream_post_enable_fence_right_line"],
            "Delta heading right: 0 rad",
        )
        self.assertEqual(
            rule["stream_post_enable_fence_semantics"],
            "post-enable-reset-tail-complete-with-net-zero-heading",
        )
        self.assertEqual(
            rule["stream_stdout_observation_offsets"],
            "raw-utf8-stdout-byte-offsets-after-archive-flush",
        )
        self.assertEqual(
            rule["stream_consumer_transcript_path"],
            "dynamic/stream/preload-consumer-transcript.json",
        )
        self.assertEqual(
            rule["stream_consumer_transcript_binding"],
            "frozen-prefix-with-independent-exact-event-byte-ranges",
        )
        self.assertEqual(
            rule["stream_consumer_event_ranges"],
            ["start", "processing", "merged", "end"],
        )
        self.assertEqual(
            rule["stream_consumer_intervening_bytes"],
            "allowed-and-sha256-bound",
        )
        self.assertEqual(
            rule["stream_consumer_final_tail"],
            "final-end-authenticated-and-post-end-bytes-sha256-bound",
        )
        self.assertEqual(
            rule["stream_consumer_publication_boundaries"],
            "after-prior-end-and-no-later-than-current-start",
        )
        self.assertEqual(
            rule["stream_consumer_merged_to_end_diagnostics"],
            "arbitrary-and-sha256-bound",
        )
        self.assertEqual(
            rule["consumer_causal_fence"],
            "publication-N+1-authenticated-start-to-end-event-fences-publication-N",
        )
        self.assertNotIn("stream_transport_publication_count", rule)
        self.assertEqual(rule["target_coverage_row_count"], 441)
        self.assertNotIn("stream_lookahead_frame_count", rule)
        self.assertNotIn("scored_activation_contents", rule)

    def test_preload_audit_reconstructs_byte_exact_stdout_prefix(self) -> None:
        bundle, canonical, stdout_path, evidence = (
            self._stream_preload_audit_fixture("exact")
        )
        try:
            frozen_prefix = stdout_path.read_bytes()
            audit = cli_module._audit_known_good_stream_preload(
                canonical, bundle, evidence
            )
            transcript_path = (
                bundle.path
                / "dynamic/stream/preload-consumer-transcript.json"
            )
            transcript = json.loads(transcript_path.read_text("ascii"))
            self.assertEqual(
                transcript["schema"],
                "mm-sonic-preload-consumer-transcript/v1",
            )
            self.assertEqual(transcript["stdout_prefix_length"], len(frozen_prefix))
            self.assertEqual(
                transcript["stdout_prefix_sha256"],
                hashlib.sha256(frozen_prefix).hexdigest(),
            )
            self.assertEqual(len(transcript["consumer_triplets"]), 25)
            first = transcript["consumer_triplets"][0]
            self.assertEqual(set(first["event_sha256"]), {
                "start", "processing", "merged", "end"
            })
            merged_end = first["merged_range"][1]
            end_start = first["end_range"][0]
            expected_merged_to_end = (
                b"[ZMQEndpointInterface] active_protocol_version_=1\n"
                b"[ZMQEndpointInterface] result.motion->GetEncodeMode()=0\n"
                b"[ZMQEndpointInterface] motion name: streamed\n"
                b"[ZMQEndpointInterface] Merged streamed data: accepted\n"
            )
            self.assertEqual(
                first["intervening_ranges"]["merged_to_end"],
                [merged_end, end_start],
            )
            self.assertEqual(
                frozen_prefix[merged_end:end_start], expected_merged_to_end
            )
            self.assertEqual(
                first["intervening_sha256"]["merged_to_end"],
                hashlib.sha256(
                    frozen_prefix[merged_end:end_start]
                ).hexdigest(),
            )
            self.assertEqual(
                transcript["post_enable_fence"], dict(evidence.post_enable_fence)
            )
            self.assertEqual(
                audit["consumer_transcript_path"],
                "dynamic/stream/preload-consumer-transcript.json",
            )
            self.assertEqual(
                audit["consumer_transcript_sha256"],
                hashlib.sha256(transcript_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                cli_module._snapshot_outputs(bundle)[
                    "dynamic/stream/preload-consumer-transcript.json"
                ],
                audit["consumer_transcript_sha256"],
            )
            self.assertEqual(audit["stdout_prefix_length"], len(frozen_prefix))
            self.assertEqual(
                audit["stdout_prefix_sha256"],
                hashlib.sha256(frozen_prefix).hexdigest(),
            )
            with stdout_path.open("ab") as output:
                output.write(b"CONTROL later append\n")
            self.assertEqual(
                hashlib.sha256(
                    stdout_path.read_bytes()[: audit["stdout_prefix_length"]]
                ).hexdigest(),
                audit["stdout_prefix_sha256"],
            )
        finally:
            bundle.__del__()

    def test_preload_audit_rejects_marker_dict_or_stdout_forgery(self) -> None:
        def expect_rejected(label, mutate):
            bundle, canonical, stdout_path, evidence = (
                self._stream_preload_audit_fixture(label)
            )
            try:
                changed = mutate(stdout_path, evidence)
                with self.assertRaises(ContractError):
                    cli_module._audit_known_good_stream_preload(
                        canonical, bundle, evidence if changed is None else changed
                    )
            finally:
                bundle.__del__()

        def forged_marker(_stdout_path, evidence):
            markers = list(evidence.consumer_markers)
            first = dict(markers[0])
            first["boundary"] += 1
            markers[0] = MappingProxyType(first)
            return replace(evidence, consumer_markers=tuple(markers))

        def invalid_marker_type(_stdout_path, evidence):
            return replace(
                evidence,
                consumer_markers=("forged", *evidence.consumer_markers[1:]),
            )

        def missing_stdout(stdout_path, _evidence):
            stdout_path.unlink()

        def truncated_stdout(stdout_path, _evidence):
            stdout_path.write_bytes(stdout_path.read_bytes()[:-1])

        def extra_starting(stdout_path, _evidence):
            with stdout_path.open("ab") as output:
                output.write(
                    b"[ZMQEndpointInterface] *** Starting ZMQ processing ***\n"
                )

        def extra_ending(stdout_path, _evidence):
            with stdout_path.open("ab") as output:
                output.write(
                    b"[ZMQEndpointInterface] "
                    b"*** End of ZMQ decoding processing ***\n"
                )

        def reordered_stdout(stdout_path, evidence):
            raw = stdout_path.read_bytes()
            marker = evidence.consumer_markers[0]
            processing_start, processing_end = marker["processing_range"]
            merged_start, merged_end = marker["merged_range"]
            stdout_path.write_bytes(
                raw[:processing_start]
                + raw[merged_start:merged_end]
                + raw[processing_end:merged_start]
                + raw[processing_start:processing_end]
                + raw[merged_end:]
            )

        for label, mutate in (
            ("forged-marker", forged_marker),
            ("invalid-marker-type", invalid_marker_type),
            ("missing-stdout", missing_stdout),
            ("truncated-stdout", truncated_stdout),
            ("extra-starting", extra_starting),
            ("extra-ending", extra_ending),
            ("reordered-stdout", reordered_stdout),
        ):
            with self.subTest(label=label):
                expect_rejected(label, mutate)

    def test_known_good_timeline_retains_only_logical_chunk_transactions(
        self,
    ) -> None:
        canonical = _canonical(441)
        timeline = cli_module._KnownGoodTimeline(canonical)
        source = cli_module._KnownGoodSource(
            session_id="s",
            candidate_id="c",
            predecessor_id=None,
            timestamps_s=tuple(index / 25.0 for index in range(11)),
        )
        prepared = timeline.prepare(source)
        accepted = timeline.commit(prepared)
        self.assertEqual(accepted.frame_index.tolist(), list(range(1, 21)))
        self.assertEqual(
            timeline.canonical_buffer.frame_index.tolist(), list(range(21))
        )
        self.assertFalse(hasattr(timeline, "readiness_buffer"))
        self.assertFalse(hasattr(timeline, "transport_lookahead"))

    def test_stage_a_has_no_marker_coupled_stream_resume_adapter(self) -> None:
        self.assertFalse(hasattr(cli_module, "_StartedGearStream"))
        self.assertFalse(hasattr(GearProcess, "wait_for_stream_publication"))

    def test_runtime_inputs_are_staged_and_launch_parity_detects_mutation(
        self,
    ) -> None:
        gear = self.root / "gear"
        binary = gear / cli_module._GEAR_BINARY_RELATIVE
        model = gear / cli_module._GEAR_MODEL_RELATIVE
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"official executable")
        binary.chmod(0o755)
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"official model")
        policy = self.root / "policy.onnx"
        policy.write_bytes(b"policy")
        observation = self.root / "observation.yaml"
        observation.write_bytes(b"observation")
        encoder = self.root / "encoder.onnx"
        encoder.write_bytes(b"encoder")
        terrain = self.root / "terrain"
        terrain.mkdir()
        source = self.root / "source.xml"
        source.write_bytes(b"source")
        inputs = ExternalInputs(
            gear_checkout=gear.resolve(),
            policy=policy.resolve(),
            observation_config=observation.resolve(),
            encoder=encoder.resolve(),
            terrain_dir=terrain.resolve(),
            source_mjcf=source.resolve(),
        )
        hashes = {
            "policy": hashlib.sha256(b"policy").hexdigest(),
            "observation_config": hashlib.sha256(b"observation").hexdigest(),
            "encoder": hashlib.sha256(b"encoder").hexdigest(),
        }
        context = StageAContext(
            gear_commit="1" * 40,
            gear_dirty=False,
            external_hashes=MappingProxyType(hashes),
            motion_matching_commit="2" * 40,
            motion_matching_dirty=False,
            artifact_hashes=MappingProxyType({}),
            known_good_reference=self.root,
            processes=(),
        )
        operations = cli_module.DefaultStageAOperations()
        operations._runtime[id(context)] = {
            "verified_external": SimpleNamespace(inputs=inputs)
        }
        bundle = RunBundle.create(self.root / "runs", "staging", "run")
        generated_scene = bundle.path / "scene.xml"
        bundle.write_bytes("scene.xml", b"generated scene")
        scene = SimpleNamespace(gear_scene_xml=generated_scene)
        try:
            runtime = operations._stage_gear_runtime_inputs(
                context, bundle, "file"
            )
            request = StageARequest(
                command="stage-a",
                mode="known-good-file",
                argv=(),
                namespace=argparse.Namespace(),
                environment=MappingProxyType({}),
            )
            command = operations._gear_command(
                request,
                context,
                self.root / "reference",
                runtime,
            )
            self.assertIn(str(runtime.policy), command)
            self.assertNotIn(str(inputs.policy), command)
            self.assertTrue(all(Path(value).is_absolute() for value in (
                command[0], command[2], command[3], command[5], command[7]
            )))
            file_identity = cli_module._dynamic_launch_identity(
                inputs,
                runtime,
                scene,
                np.zeros(36, dtype=np.float64),
                _canonical(),
                np.zeros((21, 3), dtype="<f4"),
            )
            self.assertEqual(
                cli_module._require_launch_parity(
                    file_identity,
                    cli_module._dynamic_launch_identity(
                        inputs,
                        runtime,
                        scene,
                        np.zeros(36, dtype=np.float64),
                        _canonical(),
                        np.zeros((21, 3), dtype="<f4"),
                    ),
                )["equal"],
                True,
            )
            policy.write_bytes(b"mutated")
            with self.assertRaisesRegex(ContractError, "parity"):
                cli_module._require_launch_parity(
                    file_identity,
                    cli_module._dynamic_launch_identity(
                        inputs,
                        runtime,
                        scene,
                        np.zeros(36, dtype=np.float64),
                        _canonical(),
                        np.zeros((21, 3), dtype="<f4"),
                    ),
                )
        finally:
            bundle.__del__()

    def test_default_gate_recovers_attempted_provenance_after_startup_error(
        self,
    ) -> None:
        context = StageAContext(
            gear_commit="1" * 40,
            gear_dirty=False,
            external_hashes=MappingProxyType({}),
            motion_matching_commit="2" * 40,
            motion_matching_dirty=False,
            artifact_hashes=MappingProxyType({}),
            known_good_reference=self.root,
            processes=(),
        )
        operations = cli_module.DefaultStageAOperations()
        operations._runtime[id(context)] = {}
        bundle = RunBundle.create(self.root / "runs", "provenance", "run")
        executable = Path(sys.executable).resolve(strict=True)
        environment = {"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "0"}
        argv = (str(executable), "--synthetic-startup")

        def failing_handler(_request, active_context, active_bundle):
            operations._register_process_attempt(
                active_context,
                cli_module._process_record(
                    "g1_deploy_onnx_ref-known-good-file",
                    argv,
                    executable=executable,
                    environment=environment,
                    outputs=(),
                ),
            )
            active_bundle.write_bytes(
                "dynamic/file/gear.stdout", b"startup failed\n"
            )
            active_bundle.write_bytes(
                "runtime-inputs/file/policy_policy.trt", b"partial trt\n"
            )
            raise ProcessError("child exited before startup marker")

        operations._known_good_file_dynamic = failing_handler
        request = StageARequest(
            command="stage-a",
            mode="known-good-file",
            argv=(),
            namespace=argparse.Namespace(),
            environment=MappingProxyType(environment),
        )
        try:
            result = operations.run_gate(
                "known_good_file_dynamic", request, context, bundle
            )
            self.assertEqual(result.status, "integration_failure")
            self.assertIn("startup marker", result.reason)
            self.assertEqual(len(result.processes), 1)
            process = result.processes[0]
            self.assertEqual(process["argv"], list(argv))
            self.assertEqual(process["environment"]["values"], environment)
            self.assertRegex(process["executable_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                process["outputs"],
                [
                    "dynamic/file/gear.stdout",
                    "runtime-inputs/file/policy_policy.trt",
                ],
            )
            self.assertEqual(
                result.outputs["runtime-inputs/file/policy_policy.trt"],
                hashlib.sha256(b"partial trt\n").hexdigest(),
            )
            cli_module._validate_gate_result(result, bundle)
        finally:
            bundle.__del__()

    def test_dynamic_partial_constructors_close_all_prior_resources(self) -> None:
        gear_checkout = self.root / "gear"
        (gear_checkout / "gear_sonic_deploy").mkdir(parents=True)
        policy = self.root / "policy.onnx"
        observation = self.root / "observation.yaml"
        source = self.root / "source.xml"
        terrain = self.root / "terrain"
        policy.write_bytes(b"policy")
        observation.write_bytes(b"observations: []\n")
        source.write_bytes(b"<mujoco/>\n")
        terrain.mkdir()
        inputs = ExternalInputs(
            gear_checkout=gear_checkout,
            policy=policy,
            observation_config=observation,
            encoder=None,
            terrain_dir=terrain,
            source_mjcf=source,
        )
        context = StageAContext(
            gear_commit="1" * 40,
            gear_dirty=False,
            external_hashes=MappingProxyType({}),
            motion_matching_commit="2" * 40,
            motion_matching_dirty=False,
            artifact_hashes=MappingProxyType({}),
            known_good_reference=gear_checkout,
            processes=(),
        )
        scene_xml = self.root / "scene.xml"
        scene_xml.write_bytes(b"<mujoco/>\n")
        scene = cli_module.RegisteredScene(
            scene_id="sonic-flat-baseline",
            route_id="flat-12s",
            source_kind="analytic-flat",
            source_mesh=None,
            source_heightfield=None,
            source_hashes=MappingProxyType({}),
            coordinate_source="holden-y-up-right-handed-forward-plus-z",
            coordinate_target="mujoco-z-up-right-handed-forward-plus-x",
            transform_matrix=np.eye(3, dtype=np.float64),
            transformed_obj=None,
            gear_scene_xml=scene_xml,
            output_hashes=MappingProxyType({}),
            allowed_foot_geoms=(),
            forbidden_geom_groups=MappingProxyType({}),
        )
        canonical = _canonical()
        body_position = np.zeros((canonical.count, 3), dtype="<f4")
        projection_leaf = self.root / "projected/known_good"
        projection = cli_module._KnownGoodProjection(
            reference_base=projection_leaf.parent,
            reference_leaf=projection_leaf,
            canonical=canonical,
            body_position=body_position,
            evidence=MappingProxyType({}),
        )
        runtime_inputs = cli_module._GearRuntimeInputs(
            policy=policy,
            observation_config=observation,
            encoder=None,
        )
        operations = cli_module.DefaultStageAOperations()
        operations._runtime[id(context)] = {
            "verified_external": SimpleNamespace(inputs=inputs),
            "scene": scene,
            "initial_qpos": np.zeros(36, dtype=np.float64),
            "joint_addresses": np.arange(29, dtype=np.int64),
            "initial_pelvis_quaternion": np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            ),
        }
        request = StageARequest(
            command="stage-a",
            mode="known-good-stream",
            argv=(),
            namespace=argparse.Namespace(),
            environment=MappingProxyType({}),
        )
        bundle = RunBundle.create(self.root / "runs", "cleanup", "run")

        class Gear:
            argv = (str(Path(sys.executable).resolve()),)

            def __init__(self):
                self.close_calls = 0

            def close(self):
                self.close_calls += 1

        file_gear = Gear()
        file_constructor_attempts = []
        file_simulator_attempts = []

        def construct_file_gear(**_kwargs):
            file_constructor_attempts.extend(
                dict(value)
                for value in operations._state(context)[
                    "active_gate_attempt"
                ]["processes"]
            )
            return file_gear

        def fail_file_simulator(**_kwargs):
            file_simulator_attempts.extend(
                dict(value)
                for value in operations._state(context)[
                    "active_gate_attempt"
                ]["processes"]
            )
            raise ProcessError("simulator constructor failed")

        try:
            with (
                patch.object(
                    operations,
                    "_materialize_known_good_projection",
                    return_value=projection,
                ),
                patch.object(
                    operations,
                    "_stage_gear_runtime_inputs",
                    return_value=runtime_inputs,
                ),
                patch.object(
                    operations,
                    "_gear_command",
                    return_value=(str(Path(sys.executable).resolve()),),
                ),
                patch.object(
                    cli_module,
                    "_dynamic_launch_identity",
                    return_value=MappingProxyType(
                        {"fields": MappingProxyType({}), "sha256": _sha("launch")}
                    ),
                ),
                patch.object(cli_module, "GearProcess", side_effect=construct_file_gear),
                patch.object(
                    cli_module,
                    "GatedSimulatorClient",
                    side_effect=fail_file_simulator,
                ),
            ):
                file_result = operations.run_gate(
                    "known_good_file_dynamic", request, context, bundle
                )
            self.assertEqual(file_result.status, "integration_failure")
            self.assertEqual(file_gear.close_calls, 1)
            self.assertEqual(
                [value["name"] for value in file_constructor_attempts],
                ["g1_deploy_onnx_ref-known-good-file"],
            )
            self.assertEqual(
                [value["name"] for value in file_simulator_attempts],
                [
                    "g1_deploy_onnx_ref-known-good-file",
                    "gated-simulator-known-good-file",
                ],
            )

            reference_sha256 = cli_module._canonical_target_sha256(canonical)
            file_digest = _sha("projected-file")
            source_hashes = {
                f"gear:known_good_reference/{name}": file_digest
                for name in cli_module.KNOWN_GOOD_REFERENCE_FILES
            }
            projection_evidence = MappingProxyType(
                {
                    "projected_files_sha256": {
                        name: file_digest
                        for name in cli_module.KNOWN_GOOD_REFERENCE_FILES
                    },
                    "source": {"sha256": source_hashes},
                    "canonical_target_sha256": reference_sha256,
                    "rule_sha256": _sha("projection-rule"),
                    "projection_sha256": _sha("projection"),
                }
            )
            projection = cli_module._KnownGoodProjection(
                reference_base=projection_leaf.parent,
                reference_leaf=projection_leaf,
                canonical=canonical,
                body_position=body_position,
                evidence=projection_evidence,
            )
            state = operations._runtime[id(context)]
            state.update(
                {
                    "known_good_reference_base": projection.reference_base,
                    "known_good_projection": projection,
                    "known_good_reference": canonical,
                    "known_good_body_position": body_position,
                    "known_good_file_launch_identity": MappingProxyType(
                        {"fields": MappingProxyType({}), "sha256": _sha("launch")}
                    ),
                }
            )
            projection_evidence_sha256 = hashlib.sha256(
                cli_module._canonical_json_bytes(
                    dict(projection_evidence), "known-good projection evidence"
                )
            ).hexdigest()
            real_sha256_file = cli_module._sha256_file

            def staged_sha256(path):
                candidate = Path(path)
                if candidate.name in cli_module.KNOWN_GOOD_REFERENCE_FILES:
                    return file_digest
                if candidate.name == "projection.json":
                    return projection_evidence_sha256
                return real_sha256_file(candidate)

            class Publisher:
                endpoint = "tcp://127.0.0.1:44555"

                def __init__(self):
                    self.close_calls = 0

                def close(self):
                    self.close_calls += 1

            stream_publisher = Publisher()
            stream_constructor_attempts = []

            def fail_stream_gear(**_kwargs):
                stream_constructor_attempts.extend(
                    dict(value)
                    for value in operations._state(context)[
                        "active_gate_attempt"
                    ]["processes"]
                )
                raise ProcessError("GEAR constructor failed")

            with (
                patch.object(
                    cli_module,
                    "_decode_known_good_reference",
                    return_value=canonical,
                ),
                patch.object(
                    cli_module,
                    "_decode_known_good_body_position",
                    return_value=body_position,
                ),
                patch.object(cli_module, "_sha256_file", side_effect=staged_sha256),
                patch.object(
                    cli_module,
                    "_dynamic_launch_identity",
                    return_value=MappingProxyType(
                        {"fields": MappingProxyType({}), "sha256": _sha("launch")}
                    ),
                ),
                patch.object(
                    cli_module,
                    "_require_launch_parity",
                    return_value=MappingProxyType({"equal": True}),
                ),
                patch.object(
                    operations,
                    "_gear_command",
                    return_value=(str(Path(sys.executable).resolve()),),
                ),
                patch.object(
                    operations,
                    "_stage_gear_runtime_inputs",
                    return_value=runtime_inputs,
                ),
                patch.object(
                    cli_module, "PosePublisher", return_value=stream_publisher
                ),
                patch.object(
                    cli_module,
                    "GearProcess",
                    side_effect=fail_stream_gear,
                ),
            ):
                stream_result = operations.run_gate(
                    "known_good_stream_delivery", request, context, bundle
                )
            self.assertEqual(stream_result.status, "integration_failure")
            self.assertEqual(stream_publisher.close_calls, 1)
            self.assertEqual(
                [value["name"] for value in stream_constructor_attempts],
                ["g1_deploy_onnx_ref-known-good-stream"],
            )

            stream_publisher = Publisher()
            stream_gear = Gear()
            stream_simulator_attempts = []

            def fail_stream_simulator(**_kwargs):
                stream_simulator_attempts.extend(
                    dict(value)
                    for value in operations._state(context)[
                        "active_gate_attempt"
                    ]["processes"]
                )
                raise ProcessError("simulator constructor failed")

            with (
                patch.object(
                    cli_module,
                    "_decode_known_good_reference",
                    return_value=canonical,
                ),
                patch.object(
                    cli_module,
                    "_decode_known_good_body_position",
                    return_value=body_position,
                ),
                patch.object(cli_module, "_sha256_file", side_effect=staged_sha256),
                patch.object(
                    cli_module,
                    "_dynamic_launch_identity",
                    return_value=MappingProxyType(
                        {"fields": MappingProxyType({}), "sha256": _sha("launch")}
                    ),
                ),
                patch.object(
                    cli_module,
                    "_require_launch_parity",
                    return_value=MappingProxyType({"equal": True}),
                ),
                patch.object(
                    operations,
                    "_gear_command",
                    return_value=(str(Path(sys.executable).resolve()),),
                ),
                patch.object(
                    operations,
                    "_stage_gear_runtime_inputs",
                    return_value=runtime_inputs,
                ),
                patch.object(
                    cli_module, "PosePublisher", return_value=stream_publisher
                ),
                patch.object(cli_module, "GearProcess", return_value=stream_gear),
                patch.object(
                    cli_module,
                    "GatedSimulatorClient",
                    side_effect=fail_stream_simulator,
                ),
            ):
                simulator_result = operations.run_gate(
                    "known_good_stream_delivery", request, context, bundle
                )
            self.assertEqual(simulator_result.status, "integration_failure")
            self.assertEqual(stream_gear.close_calls, 1)
            self.assertEqual(stream_publisher.close_calls, 1)
            self.assertEqual(
                [value["name"] for value in stream_simulator_attempts],
                [
                    "g1_deploy_onnx_ref-known-good-stream",
                    "gated-simulator-known-good-stream",
                ],
            )
        finally:
            bundle.__del__()

    def test_process_output_ownership_is_disjoint(self) -> None:
        outputs = {
            "dynamic/file/target.csv": _sha("target"),
            "dynamic/file/gear.stdout": _sha("gear-out"),
            "dynamic/file/gear.stderr": _sha("gear-err"),
            "dynamic/file/gear-logs/state.csv": _sha("gear-log"),
            "dynamic/file/simulator.stdout": _sha("sim-out"),
            "dynamic/file/simulator.stderr": _sha("sim-err"),
            "dynamic/file/bootstrap-sim-logs/state.jsonl": _sha("boot"),
            "dynamic/file/scored-sim-logs/state.jsonl": _sha("score"),
            "runtime-inputs/file/policy_policy.trt": _sha("file-policy-cache"),
            "runtime-inputs/stream/policy_policy.trt": _sha("stream-policy-cache"),
            "runtime-inputs/file/policy.onnx": _sha("staged-file-policy"),
            "transmitted/000001-000020.bin": _sha("publisher"),
        }
        gear, simulator = cli_module._dynamic_process_outputs(outputs, "file")
        self.assertFalse(set(gear) & set(simulator))
        self.assertEqual(
            set(gear),
            {
                "dynamic/file/target.csv",
                "dynamic/file/gear.stdout",
                "dynamic/file/gear.stderr",
                "dynamic/file/gear-logs/state.csv",
                "runtime-inputs/file/policy_policy.trt",
            },
        )
        self.assertEqual(
            set(simulator),
            {
                "dynamic/file/simulator.stdout",
                "dynamic/file/simulator.stderr",
                "dynamic/file/bootstrap-sim-logs/state.jsonl",
                "dynamic/file/scored-sim-logs/state.jsonl",
            },
        )
        stream_gear, stream_simulator = cli_module._dynamic_process_outputs(
            outputs, "stream"
        )
        self.assertEqual(
            set(stream_gear), {"runtime-inputs/stream/policy_policy.trt"}
        )
        self.assertEqual(stream_simulator, ())
        self.assertFalse(set(gear) & set(stream_gear))

    def test_file_then_stream_runtime_staging_owns_disjoint_trt_caches(
        self,
    ) -> None:
        policy = self.root / "policy.onnx"
        observation = self.root / "observation.yaml"
        source = self.root / "source.xml"
        terrain = self.root / "terrain"
        gear = self.root / "gear"
        policy.write_bytes(b"policy-v1\n")
        observation.write_bytes(b"observations: []\n")
        source.write_bytes(b"<mujoco/>\n")
        terrain.mkdir()
        gear.mkdir()
        inputs = ExternalInputs(
            gear_checkout=gear,
            policy=policy,
            observation_config=observation,
            encoder=None,
            terrain_dir=terrain,
            source_mjcf=source,
        )
        hashes = {
            "policy": hashlib.sha256(policy.read_bytes()).hexdigest(),
            "observation_config": hashlib.sha256(
                observation.read_bytes()
            ).hexdigest(),
        }
        context = StageAContext(
            gear_commit="1" * 40,
            gear_dirty=False,
            external_hashes=MappingProxyType(hashes),
            motion_matching_commit="2" * 40,
            motion_matching_dirty=False,
            artifact_hashes=MappingProxyType({}),
            known_good_reference=gear,
            processes=(),
        )
        operations = cli_module.DefaultStageAOperations()
        operations._runtime[id(context)] = {
            "verified_external": SimpleNamespace(inputs=inputs)
        }
        bundle = RunBundle.create(self.root / "runs", "trt-staging", "run")
        try:
            file_runtime = operations._stage_gear_runtime_inputs(
                context, bundle, "file"
            )
            (file_runtime.policy.parent / "policy_policy.trt").write_bytes(
                b"file TensorRT cache\n"
            )
            stream_runtime = operations._stage_gear_runtime_inputs(
                context, bundle, "stream"
            )
            (stream_runtime.policy.parent / "policy_policy.trt").write_bytes(
                b"stream TensorRT cache\n"
            )
            self.assertEqual(
                file_runtime.policy.relative_to(bundle.path).as_posix(),
                "runtime-inputs/file/policy.onnx",
            )
            self.assertEqual(
                stream_runtime.policy.relative_to(bundle.path).as_posix(),
                "runtime-inputs/stream/policy.onnx",
            )
            outputs = cli_module._snapshot_outputs(bundle)
            file_outputs, _ = cli_module._dynamic_process_outputs(outputs, "file")
            stream_outputs, _ = cli_module._dynamic_process_outputs(
                outputs, "stream"
            )
            self.assertEqual(
                file_outputs, ("runtime-inputs/file/policy_policy.trt",)
            )
            self.assertEqual(
                stream_outputs, ("runtime-inputs/stream/policy_policy.trt",)
            )
            self.assertFalse(set(file_outputs) & set(stream_outputs))
        finally:
            bundle.__del__()

    def test_stream_start_failure_never_claims_file_mode_trt_cache(self) -> None:
        context = StageAContext(
            gear_commit="1" * 40,
            gear_dirty=False,
            external_hashes=MappingProxyType({}),
            motion_matching_commit="2" * 40,
            motion_matching_dirty=False,
            artifact_hashes=MappingProxyType({}),
            known_good_reference=self.root,
            processes=(),
        )
        operations = cli_module.DefaultStageAOperations()
        operations._runtime[id(context)] = {}
        bundle = RunBundle.create(self.root / "runs", "trt-failure", "run")
        executable = Path(sys.executable).resolve(strict=True)
        environment = {"PATH": "/usr/bin:/bin"}
        bundle.write_bytes(
            "runtime-inputs/file/policy_policy.trt", b"completed file cache\n"
        )

        def failing_stream(_request, active_context, active_bundle):
            operations._register_process_attempt(
                active_context,
                cli_module._process_record(
                    "g1_deploy_onnx_ref-known-good-stream",
                    (str(executable), "--stream"),
                    executable=executable,
                    environment=environment,
                    outputs=("runtime-inputs/stream/",),
                ),
            )
            active_bundle.write_bytes(
                "runtime-inputs/stream/policy_policy.trt",
                b"partial stream cache\n",
            )
            raise ProcessError("synthetic stream startup failure")

        operations._known_good_stream_delivery = failing_stream
        request = StageARequest(
            command="stage-a",
            mode="known-good-stream",
            argv=(),
            namespace=argparse.Namespace(),
            environment=MappingProxyType(environment),
        )
        try:
            result = operations.run_gate(
                "known_good_stream_delivery", request, context, bundle
            )
            self.assertEqual(result.status, "integration_failure")
            self.assertEqual(
                result.processes[0]["outputs"],
                ["runtime-inputs/stream/policy_policy.trt"],
            )
            self.assertNotIn(
                "runtime-inputs/file/policy_policy.trt",
                result.processes[0]["outputs"],
            )
            self.assertEqual(
                set(result.outputs),
                {"runtime-inputs/stream/policy_policy.trt"},
            )
        finally:
            bundle.__del__()

    def test_gear_runtime_linkage_probe_is_cuda_tensorrt_specific(self) -> None:
        resolved = (
            "libnvinfer.so.10 => /opt/tensorrt/libnvinfer.so.10\n"
            "libcudart.so.12 => /usr/local/cuda/lib64/libcudart.so.12\n"
        )
        self.assertEqual(
            cli_module._validate_gear_linkage(resolved),
            ("libcudart", "libnvinfer"),
        )
        with self.assertRaisesRegex(ContractError, "TensorRT"):
            cli_module._validate_gear_linkage(
                "libcudart.so.12 => /usr/local/cuda/libcudart.so.12\n"
            )

        class Call:
            def __init__(self, result, count=None):
                self.result = result
                self.count = count

            def __call__(self, argument):
                if self.count is not None:
                    argument._obj.value = self.count
                return self.result

        driver = SimpleNamespace(
            cuInit=Call(0),
            cuDeviceGetCount=Call(0, count=2),
        )
        self.assertEqual(
            cli_module._cuda_device_count(lambda _name: driver),
            2,
        )
        no_device = SimpleNamespace(
            cuInit=Call(0),
            cuDeviceGetCount=Call(0, count=0),
        )
        with self.assertRaisesRegex(ContractError, "no available device"):
            cli_module._cuda_device_count(lambda _name: no_device)


if __name__ == "__main__":
    unittest.main()
