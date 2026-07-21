from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mm_sonic import cli as cli_module
from mm_sonic.artifacts import RunBundle, verify_run_inventory
from mm_sonic.cli import GateResult, StageARequest
from mm_sonic.commands import flat_command_script
from mm_sonic.external import ExternalInputs
from mm_sonic.joints import ContractError
from mm_sonic.reference import ReferenceDiagnostics
from mm_sonic.scene import RegisteredScene
from mm_sonic.stage_b import (
    FlatSimulatorSafety,
    build_stream_transport_plan,
    canonicalize_sonic_transport,
    load_flat_simulator_safety,
    validate_stage_b_coverage,
)
from mm_sonic.timeline import CanonicalTargetBuffer
from tests.python.test_sonic_cli import CompleteStageAFake, _sha


def canonical(count: int = 601) -> CanonicalTargetBuffer:
    joint_position = np.zeros((count, 29), dtype="<f4")
    joint_velocity = np.zeros((count, 29), dtype="<f4")
    body_quat_w = np.zeros((count, 4), dtype="<f4")
    body_quat_w[:, 0] = 1.0
    return CanonicalTargetBuffer(
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        body_quat_w=body_quat_w,
        frame_index=np.arange(count, dtype="<i8"),
    )


def terminal_stop_fence(
    control_steps: int = 3000,
    sim_dt_s: float = 0.004,
) -> MappingProxyType:
    simulator_steps = control_steps + 1
    return MappingProxyType(
        {
            "target_rows": 601,
            "required_control_steps": control_steps,
            "requested_control_steps": control_steps,
            "control_drive_steps": control_steps,
            "control_drive_duration_s": control_steps * sim_dt_s,
            "simulator_steps": simulator_steps,
            "simulator_duration_s": simulator_steps * sim_dt_s,
            "state_rows": simulator_steps // round(0.02 / sim_dt_s),
            "contact_rows": simulator_steps,
            "gear_stopped_before_audit": True,
            "post_stop_snapshot_stable": True,
            "final_snapshot_stable": True,
            "exact": True,
        }
    )


class StageBContractTests(unittest.TestCase):
    def test_parser_registers_stage_b_with_authenticated_prerequisite(self) -> None:
        namespace = cli_module._parser().parse_args(
            [
                "stage-b",
                "--stage-a-evidence",
                "/evidence/stage-a-evidence.json",
                "--gear-checkout",
                "/gear",
                "--policy",
                "/policy.onnx",
                "--observation-config",
                "/observation.yaml",
                "--source-mjcf",
                "/g1.xml",
                "--terrain-dir",
                "/terrain",
                "--output-root",
                "/runs",
            ]
        )
        self.assertEqual(namespace.command, "stage-b")
        self.assertEqual(
            namespace.stage_a_evidence,
            "/evidence/stage-a-evidence.json",
        )

    def test_601_frame_transport_has_30_logical_chunks_and_future_fence(self) -> None:
        plan = build_stream_transport_plan(canonical())

        np.testing.assert_array_equal(plan.readiness.frame_index, [0])
        self.assertEqual(len(plan.logical), 30)
        self.assertTrue(all(buffer.count == 20 for buffer in plan.logical))
        np.testing.assert_array_equal(
            np.concatenate([buffer.frame_index for buffer in plan.logical]),
            np.arange(1, 601, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            plan.padding.frame_index,
            np.arange(601, 647, dtype=np.int64),
        )
        np.testing.assert_array_equal(plan.receipt_fence.frame_index, [647])
        self.assertEqual(plan.publication_count, 33)

    def test_sonic_transport_canonicalizes_only_float32_subnormals(self) -> None:
        source = canonical(2)
        joint_position = source.joint_position.copy()
        joint_velocity = source.joint_velocity.copy()
        body_quat = source.body_quat_w.copy()
        smallest = np.nextafter(np.float32(0.0), np.float32(1.0))
        joint_position[0, 0] = smallest
        joint_position[0, 1] = -smallest
        joint_velocity[1, 2] = np.finfo(np.float32).tiny
        source = CanonicalTargetBuffer(
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            body_quat_w=body_quat,
            frame_index=np.arange(2, dtype="<i8"),
        )

        result = canonicalize_sonic_transport(source)

        self.assertEqual(result.subnormal_count, 2)
        self.assertEqual(result.maximum_subnormal_magnitude, float(smallest))
        self.assertEqual(result.buffer.joint_position[0, 0], 0.0)
        self.assertEqual(result.buffer.joint_position[0, 1], 0.0)
        self.assertEqual(
            result.buffer.joint_velocity[1, 2], np.finfo(np.float32).tiny
        )
        self.assertEqual(source.joint_position[0, 0], smallest)

    def test_transport_plan_preserves_stage_a_441_frame_layout(self) -> None:
        plan = build_stream_transport_plan(canonical(441))
        self.assertEqual(len(plan.logical), 22)
        self.assertEqual(plan.publication_count, 25)
        np.testing.assert_array_equal(
            plan.padding.frame_index,
            np.arange(441, 487, dtype=np.int64),
        )
        np.testing.assert_array_equal(plan.receipt_fence.frame_index, [487])

    def test_exact_stage_b_coverage_is_30_commands_601_frames_12_seconds(self) -> None:
        result = validate_stage_b_coverage(
            commands=flat_command_script(),
            canonical=canonical(),
            accepted_command_indices=tuple(range(30)),
            observed_target_rows=601,
            control_drive_steps=3000,
            control_drive_duration_s=12.0,
            sim_dt_s=0.004,
            control_lead_rows=16,
            terminal_stop_fence=terminal_stop_fence(),
        )
        self.assertEqual(result.command_count, 30)
        self.assertEqual(result.frame_count, 601)
        self.assertEqual(result.reference_duration_s, 12.0)
        self.assertTrue(result.exact_command_coverage)
        self.assertTrue(result.exact_frame_coverage)
        self.assertTrue(result.exact_control_duration)
        self.assertTrue(result.terminal_stop_fence_exact)

        cases = (
            {
                "commands": flat_command_script()[:-1],
                "canonical": canonical(),
                "accepted_command_indices": tuple(range(29)),
                "observed_target_rows": 601,
                "control_drive_steps": 3000,
                "control_drive_duration_s": 12.0,
                "sim_dt_s": 0.004,
                "control_lead_rows": 16,
                "terminal_stop_fence": terminal_stop_fence(),
            },
            {
                "commands": flat_command_script(),
                "canonical": canonical(581),
                "accepted_command_indices": tuple(range(30)),
                "observed_target_rows": 581,
                "control_drive_steps": 3000,
                "control_drive_duration_s": 12.0,
                "sim_dt_s": 0.004,
                "control_lead_rows": 16,
                "terminal_stop_fence": terminal_stop_fence(),
            },
            {
                "commands": flat_command_script(),
                "canonical": canonical(),
                "accepted_command_indices": (*range(29), 28),
                "observed_target_rows": 601,
                "control_drive_steps": 3000,
                "control_drive_duration_s": 12.0,
                "sim_dt_s": 0.004,
                "control_lead_rows": 16,
                "terminal_stop_fence": terminal_stop_fence(),
            },
            {
                "commands": flat_command_script(),
                "canonical": canonical(),
                "accepted_command_indices": tuple(range(30)),
                "observed_target_rows": 600,
                "control_drive_steps": 3000,
                "control_drive_duration_s": 12.0,
                "sim_dt_s": 0.004,
                "control_lead_rows": 16,
                "terminal_stop_fence": terminal_stop_fence(),
            },
            {
                "commands": flat_command_script(),
                "canonical": canonical(),
                "accepted_command_indices": tuple(range(30)),
                "observed_target_rows": 601,
                "control_drive_steps": 2999,
                "control_drive_duration_s": 11.996,
                "sim_dt_s": 0.004,
                "control_lead_rows": 16,
                "terminal_stop_fence": terminal_stop_fence(),
            },
            {
                "commands": flat_command_script(),
                "canonical": canonical(),
                "accepted_command_indices": tuple(range(30)),
                "observed_target_rows": 601,
                "control_drive_steps": 3000,
                "control_drive_duration_s": 12.0,
                "sim_dt_s": 0.004,
                "control_lead_rows": 15,
                "terminal_stop_fence": terminal_stop_fence(),
            },
            {
                "commands": flat_command_script(),
                "canonical": canonical(),
                "accepted_command_indices": tuple(range(30)),
                "observed_target_rows": 601,
                "control_drive_steps": 3000,
                "control_drive_duration_s": 12.0,
                "sim_dt_s": 0.004,
                "control_lead_rows": 16,
                "terminal_stop_fence": terminal_stop_fence(2999),
            },
        )
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(ContractError):
                validate_stage_b_coverage(**fields)

        for name, value in (
            ("requested_control_steps", 3000.0),
            ("control_drive_duration_s", math.nan),
            ("simulator_duration_s", math.inf),
        ):
            damaged_fence = dict(terminal_stop_fence())
            damaged_fence[name] = value
            with self.subTest(fence_field=name), self.assertRaises(ContractError):
                validate_stage_b_coverage(
                    commands=flat_command_script(),
                    canonical=canonical(),
                    accepted_command_indices=tuple(range(30)),
                    observed_target_rows=601,
                    control_drive_steps=3000,
                    control_drive_duration_s=12.0,
                    sim_dt_s=0.004,
                    control_lead_rows=16,
                    terminal_stop_fence=damaged_fence,
                )

    def test_flat_simulator_safety_uses_registered_geom_ids_not_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states = (
                {
                    "step": 4,
                    "sim_time_s": 0.02,
                    "state": {
                        "pelvis_position_m": [0.0, 0.0, 0.51],
                        "pelvis_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "pelvis_up": [0.0, 0.0, 1.0],
                    },
                },
                {
                    "step": 8,
                    "sim_time_s": 0.04,
                    "state": {
                        "pelvis_position_m": [0.1, 0.0, 0.49],
                        "pelvis_quaternion_wxyz": [
                            math.sqrt(0.9),
                            math.sqrt(0.1),
                            0.0,
                            0.0,
                        ],
                        "pelvis_up": [0.0, -0.6, 0.8],
                    },
                },
            )
            contacts = (
                {
                    "step": 1,
                    "sim_time_s": 0.005,
                    "contacts": [
                        {
                            "geom1_id": 10,
                            "geom2_id": 100,
                            "geom1": "left_foot",
                            "geom2": "floor",
                            "distance_m": -0.001,
                            "force": [0.0, 0.0, 20.0, 0.0, 0.0, 0.0],
                        }
                    ],
                },
                {
                    "step": 2,
                    "sim_time_s": 0.01,
                    "contacts": [
                        {
                            "geom1_id": 1,
                            "geom2_id": 100,
                            "geom1": "misleading_safe_name",
                            "geom2": "floor",
                            "distance_m": -0.002,
                            "force": [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                        }
                    ],
                },
                {
                    "step": 3,
                    "sim_time_s": 0.015,
                    "contacts": [
                        {
                            "geom1_id": 2,
                            "geom2_id": 5,
                            "geom1": "knee_self_contact",
                            "geom2": "hand_self_contact",
                            "distance_m": -0.003,
                            "force": [0.0, 0.0, 30.0, 0.0, 0.0, 0.0],
                        }
                    ],
                },
            ) + tuple(
                {
                    "step": step,
                    "sim_time_s": step * 0.005,
                    "contacts": [],
                }
                for step in range(4, 9)
            )
            (root / "state.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in states),
                encoding="ascii",
            )
            (root / "contacts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in contacts),
                encoding="ascii",
            )

            result = load_flat_simulator_safety(
                root,
                terrain_geoms=(100,),
                allowed_foot_geoms=(10,),
                forbidden_geom_groups={
                    "pelvis": (1,),
                    "knees": (2, 3),
                    "torso": (4,),
                    "hands": (5, 6),
                },
                sim_dt_s=0.005,
                expected_final_pelvis_xy=(0.1, 0.0),
                expected_simulator_steps=8,
                expected_state_rows=2,
                expected_contact_rows=8,
            )

        self.assertEqual(result.state_rows, 2)
        self.assertEqual(result.contact_rows, 8)
        self.assertEqual(result.minimum_pelvis_local_height_m, 0.49)
        self.assertAlmostEqual(result.minimum_pelvis_up_dot, 0.8)
        self.assertEqual(result.forbidden_contact_groups, ("pelvis",))
        self.assertEqual(result.swing_foot_scuff_count, 1)
        self.assertEqual(result.horizontal_path_drift_m, 0.0)
        self.assertEqual(len(result.contact_impulses_ns), 2)
        self.assertAlmostEqual(result.contact_impulses_ns[0], 0.1)

    def test_flat_simulator_safety_rejects_truncated_complete_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = {
                "step": 4,
                "sim_time_s": 0.02,
                "state": {
                    "pelvis_position_m": [0.0, 0.0, 0.8],
                    "pelvis_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "pelvis_up": [0.0, 0.0, 1.0],
                },
            }
            contact = {
                "step": 1,
                "sim_time_s": 0.005,
                "contacts": [],
            }
            (root / "state.jsonl").write_text(
                json.dumps(state) + "\n", encoding="ascii"
            )
            (root / "contacts.jsonl").write_text(
                json.dumps(contact) + "\n", encoding="ascii"
            )

            with self.assertRaisesRegex(ContractError, "coverage"):
                load_flat_simulator_safety(
                    root,
                    terrain_geoms=(100,),
                    allowed_foot_geoms=(10,),
                    forbidden_geom_groups={
                        "pelvis": (1,),
                        "knees": (2,),
                        "torso": (3,),
                        "hands": (4,),
                    },
                    sim_dt_s=0.005,
                    expected_final_pelvis_xy=(0.0, 0.0),
                    expected_simulator_steps=8,
                    expected_state_rows=2,
                    expected_contact_rows=8,
                )


class StageBDefaultDynamicTests(unittest.TestCase):
    def test_default_dynamic_streams_all_601_frames_and_reports_scored_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gear_checkout = root / "gear"
            terrain = root / "terrain"
            gear_checkout.mkdir()
            terrain.mkdir()
            policy = root / "policy.onnx"
            observation = root / "observation.yaml"
            source_mjcf = root / "g1.xml"
            for path in (policy, observation, source_mjcf):
                path.write_bytes(path.name.encode("ascii"))
            inputs = ExternalInputs(
                gear_checkout=gear_checkout,
                policy=policy,
                observation_config=observation,
                encoder=None,
                terrain_dir=terrain,
                source_mjcf=source_mjcf,
            )
            request = StageARequest(
                command="stage-b",
                mode=None,
                argv=("stage-b",),
                namespace=SimpleNamespace(),
                environment=MappingProxyType({}),
                invocation_cwd=root,
            )
            context = CompleteStageAFake().authenticate(request, inputs)
            scene_xml = root / "scene.xml"
            scene_xml.write_text("<mujoco/>\n", encoding="ascii")
            scene = RegisteredScene(
                scene_id="sonic-flat-baseline",
                route_id="flat-12s",
                source_kind="analytic-flat",
                source_mesh=None,
                source_heightfield=None,
                source_hashes=MappingProxyType({"registry": _sha("registry")}),
                coordinate_source="holden",
                coordinate_target="mujoco",
                transform_matrix=np.eye(3),
                transformed_obj=None,
                gear_scene_xml=scene_xml,
                output_hashes=MappingProxyType({"gear_scene_xml": _sha("scene")}),
                allowed_foot_geoms=(10, 11),
                forbidden_geom_groups=MappingProxyType(
                    {
                        "pelvis": (1,),
                        "knees": (2, 3),
                        "torso": (4,),
                        "hands": (5, 6),
                    }
                ),
                terrain_geoms=(0,),
            )
            raw_reference = canonical()
            joint_position = raw_reference.joint_position.copy()
            joint_position[13, 18] = np.nextafter(
                np.float32(0.0), np.float32(1.0)
            )
            reference = CanonicalTargetBuffer(
                joint_position=joint_position,
                joint_velocity=raw_reference.joint_velocity,
                body_quat_w=raw_reference.body_quat_w,
                frame_index=raw_reference.frame_index,
            )
            diagnostics = ReferenceDiagnostics(
                physical_pelvis_position=np.column_stack(
                    (
                        np.linspace(0.0, 1.0, 601),
                        np.zeros(601),
                        np.full(601, 0.8),
                    )
                ),
                virtual_root_position=np.zeros((601, 3)),
                virtual_root_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (601, 1)),
            )
            operations = cli_module.DefaultStageAOperations()
            operations._runtime[id(context)] = {
                "verified_external": SimpleNamespace(inputs=inputs),
                "scene": scene,
                "initial_qpos": np.zeros(36),
                "mm_reference": reference,
                "mm_diagnostics": diagnostics,
            }
            bundle = RunBundle.create(root / "runs", "stage-b", "unit")
            projection = cli_module._KnownGoodProjection(
                reference_base=root,
                reference_leaf=root,
                canonical=canonical(441),
                body_position=np.zeros((441, 3), dtype="<f4"),
                evidence=MappingProxyType({"projection_sha256": _sha("projection")}),
            )
            runtime = cli_module._GearRuntimeInputs(
                policy=policy,
                observation_config=observation,
                encoder=None,
            )
            coverage = cli_module._TargetCoverageResult(
                target_rows=601,
                control_drive_steps=3000,
                control_drive_duration_s=12.0,
                simulator_steps=3000,
                simulator_duration_s=12.0,
                state_rows=600,
                contact_rows=3000,
                wall_duration_s=12.5,
                target_device=1,
                target_inode=2,
                target_sha256=_sha("target"),
                terminal_stop_fence=terminal_stop_fence(),
            )
            execution = SimpleNamespace(
                bootstrap_steps=3,
                bootstrap=MappingProxyType({"ready": True}),
                wait_maintenance_steps=4,
                wait_maintenance=MappingProxyType({"ready": True}),
                wait_epoch=SimpleNamespace(
                    target_rows=0,
                    q_rows=0,
                    base_rows=0,
                    target_device=3,
                    target_inode=4,
                    target_sha256=_sha("wait"),
                ),
                prime=MappingProxyType({"prime": {"steps": 0}}),
                coverage=coverage,
                target_audit=MappingProxyType({"exact": True}),
                metrics=MappingProxyType(
                    {
                        "frame_count": 601,
                        "joint_position_rmse_rad": 0.02,
                        "pelvis_orientation_rms_rad": 0.03,
                        "joint_tracking_trace_rad": [0.02] * 601,
                        "pelvis_tracking_trace_rad": [0.03] * 601,
                        "pairing": "same-control-tick-positional",
                        "tick_period_ms": {"count": 600},
                    }
                ),
                preload_audit=MappingProxyType(
                    {
                        "logical_frames": 600,
                        "consumer_marker_count": 33,
                        "exact": True,
                    }
                ),
            )
            safety = FlatSimulatorSafety(
                state_rows=600,
                contact_rows=3001,
                minimum_pelvis_local_height_m=0.6,
                minimum_pelvis_up_dot=0.9,
                forbidden_contact_groups=(),
                swing_foot_scuff_count=0,
                minimum_foot_clearance_m=0.0,
                horizontal_path_drift_m=0.05,
                contact_impulses_ns=(0.1,),
            )

            class Publisher:
                endpoint = "tcp://127.0.0.1:43123"

                def __init__(self, *_args, **_kwargs):
                    self.closed = False

                def close(self):
                    self.closed = True

            class Gear:
                def __init__(self, *, command, **_kwargs):
                    self.argv = tuple(command)

                def close(self):
                    pass

            class Simulator:
                def __init__(self, **_kwargs):
                    self.command = (
                        sys.executable,
                        "-m",
                        "mm_sonic.gated_sim",
                        "--unpaced-physics",
                    )
                    self.sim_dt = 0.004

                def close(self):
                    pass

            calls = []

            def execute(**kwargs):
                calls.append(kwargs)
                return execution

            with (
                patch.object(
                    operations,
                    "_materialize_known_good_projection",
                    return_value=projection,
                ),
                patch.object(
                    operations,
                    "_stage_gear_runtime_inputs",
                    return_value=runtime,
                ),
                patch.object(
                    operations,
                    "_gear_command",
                    return_value=(str(Path(sys.executable).resolve(strict=True)),),
                ),
                patch.object(cli_module, "PosePublisher", Publisher),
                patch.object(cli_module, "GearProcess", Gear),
                patch.object(cli_module, "GatedSimulatorClient", Simulator),
                patch.object(
                    cli_module,
                    "_execute_known_good_scoring_epoch",
                    side_effect=execute,
                ),
                patch.object(
                    cli_module,
                    "load_flat_simulator_safety",
                    return_value=safety,
                ) as safety_loader,
            ):
                result = operations.run_gate(
                    "stage_b_dynamic", request, context, bundle
                )

            self.assertEqual(result.status, "pass", result.reason)
            evidence = json.loads(
                (bundle.path / "gates/stage_b_dynamic.json").read_text("ascii")
            )
            self.assertEqual(evidence["fresh_low_state_prime"]["prime"]["steps"], 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["canonical"].count, 601)
            self.assertEqual(calls[0]["control_lead_rows"], 16)
            self.assertEqual(calls[0]["canonical"].joint_position[13, 18], 0.0)
            np.testing.assert_array_equal(
                calls[0]["body_position"], np.zeros((601, 3), dtype="<f4")
            )
            safety_loader.assert_called_once_with(
                bundle.path / "dynamic/stream/scored-sim-logs",
                terrain_geoms=(0,),
                allowed_foot_geoms=(10, 11),
                forbidden_geom_groups=scene.forbidden_geom_groups,
                sim_dt_s=0.004,
                expected_final_pelvis_xy=(1.0, 0.0),
                expected_simulator_steps=3000,
                expected_state_rows=600,
                expected_contact_rows=3000,
            )
            self.assertEqual(result.metrics["coverage"]["command_count"], 30)
            self.assertEqual(result.metrics["coverage"]["frame_count"], 601)
            self.assertTrue(
                result.metrics["coverage"]["terminal_stop_fence_exact"]
            )
            fence = result.metrics["coverage"]["terminal_stop_fence"]
            self.assertEqual(
                fence["requested_control_steps"],
                3000,
            )
            self.assertEqual(
                result.metrics["tracking"]["joint_tracking_trace_rad"],
                [0.02] * 601,
            )
            self.assertEqual(
                result.metrics["secondary"]["horizontal_path_drift_m"],
                0.05,
            )
            self.assertEqual(
                result.metrics["transport_normalization"]["subnormal_count"],
                1,
            )
            self.assertIn("--unpaced-physics", result.processes[1]["argv"])
            self.assertIn("gates/stage_b_dynamic.json", result.outputs)


class StageBCLITests(unittest.TestCase):
    class Fake(CompleteStageAFake):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls: list[str] = []
            self.dynamic_called = False

        def run_gate(self, name, request, context, bundle):
            self.calls.append(name)
            if name != "stage_b_dynamic":
                result = super().run_gate(name, request, context, bundle)
                if name != "basis_and_scene_alignment" or result.status != "pass":
                    return result
                robot_relative = "scene/gear_robot.xml"
                scene_relative = "scene/gear_scene.xml"
                robot = b'<mujoco model="g1"><worldbody/></mujoco>\n'
                robot_path = bundle.path / robot_relative
                scene = (
                    '<?xml version="1.0" encoding="utf-8"?>\n'
                    '<mujoco model="g1 scene">\n'
                    f'  <include file="{robot_path}" />\n'
                    '  <worldbody><geom name="mm_terrain" type="plane" '
                    'size="0 0 0.05" /></worldbody>\n'
                    '</mujoco>\n'
                ).encode("utf-8")
                bundle.write_bytes(robot_relative, robot)
                bundle.write_bytes(scene_relative, scene)
                robot_sha256 = hashlib.sha256(robot).hexdigest()
                scene_sha256 = hashlib.sha256(scene).hexdigest()
                registration = dict(result.scene_registration)
                output_hashes = dict(registration["output_hashes"])
                output_hashes.update(
                    {
                        "gear_scene_xml": scene_sha256,
                        "robot_include": robot_sha256,
                        "official_scene": _sha("official-scene"),
                        "official_robot": _sha("official-robot"),
                    }
                )
                registration["output_hashes"] = output_hashes
                registration["terrain_geoms"] = [0]
                identity = dict(result.identity)
                identity["generated_flat_scene"] = scene_sha256
                outputs = dict(result.outputs)
                outputs[robot_relative] = robot_sha256
                outputs[scene_relative] = scene_sha256
                return GateResult(
                    status=result.status,
                    reason=result.reason,
                    identity=MappingProxyType(identity),
                    evidence_hashes=result.evidence_hashes,
                    metrics=result.metrics,
                    outputs=MappingProxyType(outputs),
                    scene_registration=MappingProxyType(registration),
                    artifact_hashes=result.artifact_hashes,
                    processes=result.processes,
                )
            self.dynamic_called = True
            payload = cli_module._canonical_json_bytes(
                {"stage": "B", "integration": "pass"},
                "fake Stage B dynamic",
            )
            relative = "gates/stage_b_dynamic.json"
            bundle.write_bytes(relative, payload)
            digest = cli_module._sha256_bytes(payload)
            return GateResult(
                status="pass",
                evidence_hashes=MappingProxyType(
                    {"stage_b_dynamic_sha256": digest}
                ),
                metrics=MappingProxyType(
                    {
                        "coverage": {
                            "command_count": 30,
                            "accepted_command_indices": list(range(30)),
                            "frame_count": 601,
                            "observed_target_rows": 601,
                            "reference_duration_s": 12.0,
                            "exact_command_coverage": True,
                            "exact_frame_coverage": True,
                            "exact_control_duration": True,
                            "terminal_stop_fence_exact": True,
                        },
                        "tracking": {
                            "joint_position_rmse_rad": 0.02,
                            "pelvis_orientation_rms_rad": 0.03,
                            "joint_tracking_trace_rad": [0.02] * 601,
                            "pelvis_tracking_trace_rad": [0.03] * 601,
                        },
                        "safety": {
                            "minimum_pelvis_local_height_m": 0.6,
                            "minimum_pelvis_up_dot": 0.9,
                            "forbidden_contact_groups": [],
                            "state_rows": 500,
                            "contact_rows": 2000,
                            "expected_state_rows": 500,
                            "expected_contact_rows": 2000,
                            "expected_simulator_steps": 2000,
                            "exact_log_coverage": True,
                        },
                        "secondary": {
                            "swing_foot_scuff_count": 0,
                            "minimum_foot_clearance_m": 0.0,
                            "horizontal_path_drift_m": 0.1,
                            "contact_impulses_ns": [],
                            "policy_execution_timing_ns": None,
                        },
                        "timings": {"wall_duration_s": 12.1},
                        "delivery_audit": {
                            "logical_frames": 600,
                            "consumer_marker_count": 33,
                            "exact": True,
                        },
                    }
                ),
                outputs=MappingProxyType({relative: digest}),
            )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.gear = self.root / "gear"
        self.gear.mkdir()
        (self.gear / "known-good").mkdir()
        self.policy = self.root / "policy.onnx"
        self.policy.write_bytes(b"policy")
        self.observation = self.root / "observation.yaml"
        self.observation.write_text("observations: []\n", encoding="ascii")
        self.source_mjcf = self.root / "g1.xml"
        self.source_mjcf.write_text("<mujoco/>\n", encoding="ascii")
        self.terrain = self.root / "terrain"
        self.terrain.mkdir()
        (self.terrain / "manifest.json").write_text("{}\n", encoding="ascii")
        self.stage_a_root = self.root / "stage-a-runs"
        code = cli_module.main(
            [
                "stage-a",
                "--mode",
                "known-good-stream",
                *self.common(self.stage_a_root),
            ],
            operations=self.Fake(),
            environ={},
        )
        self.assertEqual(code, 0)
        bundles = tuple((self.stage_a_root / "stage-a").iterdir())
        self.assertEqual(len(bundles), 1)
        self.stage_a_evidence = bundles[0] / "stage-a-evidence.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def common(self, output: Path) -> list[str]:
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
            str(output),
        ]

    def run_stage_b(self, fake) -> tuple[int, Path, dict[str, object]]:
        output = self.root / f"stage-b-{len(tuple(self.root.iterdir()))}"
        code = cli_module.main(
            [
                "stage-b",
                "--stage-a-evidence",
                str(self.stage_a_evidence),
                *self.common(output),
            ],
            operations=fake,
            environ={},
        )
        bundles = tuple((output / "stage-b").iterdir())
        self.assertEqual(len(bundles), 1)
        evidence = json.loads(
            (bundles[0] / "stage-b-evidence.json").read_text("ascii")
        )
        return code, bundles[0], evidence

    def test_stage_b_runs_kinematic_before_dynamic_and_seals_pass(self) -> None:
        fake = self.Fake()

        code, bundle, evidence = self.run_stage_b(fake)

        self.assertEqual(code, 0)
        self.assertEqual(
            fake.calls,
            [
                "external_identity",
                "joint_projection_round_trip",
                "basis_and_scene_alignment",
                "flat_mm_kinematic_replay",
                "stage_b_dynamic",
            ],
        )
        self.assertEqual(evidence["schema"], "mm-sonic-trial-verdict/v1")
        self.assertEqual(evidence["stage"], "B")
        self.assertTrue(evidence["integration_pass"])
        self.assertTrue(evidence["kinematic_pass"])
        self.assertTrue(evidence["dynamic_pass"])
        self.assertIsNone(evidence["failure_layer"])
        self.assertEqual(evidence["expected_frames"], 601)
        self.assertEqual(evidence["expected_sim_time_s"], 12.0)
        self.assertTrue(verify_run_inventory(bundle))

    def test_stage_b_identity_mismatch_blocks_dynamic_launch(self) -> None:
        class Mismatch(self.Fake):
            def run_gate(inner_self, name, request, context, bundle):
                result = super().run_gate(name, request, context, bundle)
                if name == "flat_mm_kinematic_replay" and result.status == "pass":
                    return GateResult(
                        status="pass",
                        identity=MappingProxyType(
                            {"mm_reference_buffer": _sha("different-reference")}
                        ),
                        evidence_hashes=result.evidence_hashes,
                        outputs=result.outputs,
                    )
                return result

        fake = Mismatch()

        code, _bundle, evidence = self.run_stage_b(fake)

        self.assertEqual(code, 2)
        self.assertFalse(fake.dynamic_called)
        self.assertFalse(evidence["integration_pass"])
        self.assertTrue(evidence["kinematic_pass"])
        self.assertFalse(evidence["dynamic_pass"])
        self.assertEqual(evidence["failure_layer"], "prerequisite")

    def test_stage_b_accepts_path_only_generated_scene_hash_change(self) -> None:
        fake = self.Fake()

        code, _bundle, evidence = self.run_stage_b(fake)

        self.assertEqual(code, 0)
        self.assertTrue(fake.dynamic_called)
        self.assertTrue(evidence["dynamic_pass"])
        self.assertIn(
            "scene_semantic_sha256", evidence["evidence_hashes"]
        )

    def test_stage_b_rejects_semantic_generated_scene_change(self) -> None:
        class SemanticSceneChange(self.Fake):
            def run_gate(inner_self, name, request, context, bundle):
                result = super().run_gate(name, request, context, bundle)
                if name != "basis_and_scene_alignment" or result.status != "pass":
                    return result
                relative = "scene/gear_scene.xml"
                path = bundle.path / relative
                changed = path.read_bytes().replace(
                    b'type="plane"', b'type="sphere"'
                )
                path.write_bytes(changed)
                changed_hash = hashlib.sha256(changed).hexdigest()
                registration = dict(result.scene_registration)
                output_hashes = dict(registration["output_hashes"])
                output_hashes["gear_scene_xml"] = changed_hash
                registration["output_hashes"] = output_hashes
                identity = dict(result.identity)
                identity["generated_flat_scene"] = changed_hash
                outputs = dict(result.outputs)
                outputs[relative] = changed_hash
                return GateResult(
                    status=result.status,
                    reason=result.reason,
                    identity=MappingProxyType(identity),
                    evidence_hashes=result.evidence_hashes,
                    metrics=result.metrics,
                    outputs=MappingProxyType(outputs),
                    scene_registration=MappingProxyType(registration),
                    artifact_hashes=result.artifact_hashes,
                    processes=result.processes,
                )

        fake = SemanticSceneChange()

        code, _bundle, evidence = self.run_stage_b(fake)

        self.assertEqual(code, cli_module.EXIT_CONFIGURATION)
        self.assertFalse(fake.dynamic_called)
        self.assertEqual(evidence["failure_layer"], "prerequisite")

    def test_stage_b_kinematic_failure_never_launches_sonic(self) -> None:
        fake = self.Fake(
            failure_gate="flat_mm_kinematic_replay",
            failure_status="scientific_failure",
        )

        code, _bundle, evidence = self.run_stage_b(fake)

        self.assertEqual(code, 3)
        self.assertFalse(fake.dynamic_called)
        self.assertFalse(evidence["kinematic_pass"])
        self.assertFalse(evidence["dynamic_pass"])
        self.assertEqual(evidence["failure_layer"], "reference")

    def test_stage_b_missing_prerequisite_seals_fail_closed_not_run(self) -> None:
        output = self.root / "stage-b-missing-prerequisite"

        code = cli_module.main(
            [
                "stage-b",
                "--stage-a-evidence",
                str(self.root / "missing-stage-a-evidence.json"),
                *self.common(output),
            ],
            operations=self.Fake(),
            environ={},
        )

        self.assertEqual(code, cli_module.EXIT_NOT_RUN)
        bundles = tuple((output / "stage-b").iterdir())
        self.assertEqual(len(bundles), 1)
        evidence = json.loads(
            (bundles[0] / "stage-b-evidence.json").read_text("ascii")
        )
        self.assertFalse(evidence["integration_pass"])
        self.assertFalse(evidence["kinematic_pass"])
        self.assertFalse(evidence["dynamic_pass"])
        self.assertEqual(evidence["failure_layer"], "prerequisite")
        self.assertTrue(verify_run_inventory(bundles[0]))


if __name__ == "__main__":
    unittest.main()
