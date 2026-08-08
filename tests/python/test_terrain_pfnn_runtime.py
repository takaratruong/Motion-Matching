from __future__ import annotations

import math
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import mm_sonic.evaluate_terrain_pfnn as evaluate_module
import mm_sonic.train_terrain_pfnn as train_module
import mm_sonic.terrain_pfnn.training as training_module
import mm_sonic.terrain_pfnn.runtime as runtime_module

from mm_sonic.evaluate_terrain_pfnn import (
    _decode_usd_mesh,
    _load_decoded_grail_npz,
    _load_decoded_usd_npz,
    KnownSlopeTerrain,
    SourceAlignedTerrain,
    known_train_command,
    select_known_train_grail_record,
    evaluate,
)
from mm_sonic.train_terrain_pfnn import fitted_subset_metadata, validated_normal_selection
from mm_sonic.terrain_pfnn.dataset import pfnn_input_sha256
from mm_sonic.terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT, TRAJECTORY_TIMES_S
from mm_sonic.terrain_pfnn.recurrence import (
    PlannedTrajectory,
    advance_recurrent_state,
    initialize_recurrent_state,
    pack_recurrent_input,
    plan_recurrent_trajectory,
)
from mm_sonic.terrain_pfnn.runtime import (
    ClosedLoopRecorder,
    ClosedLoopScenario,
    ClosedLoopValidationResult,
    NativeG1RuntimeGeometry,
    PFNNRuntimeFrame,
    RuntimeGeometryObservation,
    TerrainPFNNRuntime,
    TerrainSample,
    evaluate_closed_loop_scenarios,
    make_validation_scenario,
    validation_identity_set_receipt,
)
from mm_sonic.terrain_pfnn.training import FittedTransitionReport, finite_runtime_seed
from mm_sonic.terrain_oracle.canonical import CanonicalTerrainMesh
from mm_sonic.terrain_oracle.math3d import RigidTransform


MODEL_PATH = Path(
    "/home/ubuntu/projects/gear-sonic-pinned-60de0df/"
    "motionbricks/assets/skeletons/g1/g1_29dof.xml"
)


class FakeKinematics:
    kinematic_signature_sha256 = "kinematic-signature"
    joint_limits = torch.tensor([[-2.0, 2.0]] * 29, dtype=torch.float64)


class FakeCheckpoint:
    def __init__(self) -> None:
        self.kinematic_signature_sha256 = FakeKinematics.kinematic_signature_sha256
        self.joint_limits = FakeKinematics.joint_limits.clone()
        self.runtime_seed = finite_runtime_seed()
        self.phase_advance_q99 = 0.2
        self.normalization = {
            "x_mean": torch.zeros(INPUT_LAYOUT.size, dtype=torch.float32),
            "x_std": torch.ones(INPUT_LAYOUT.size, dtype=torch.float32),
            "y_mean": torch.zeros(OUTPUT_LAYOUT.size, dtype=torch.float32),
            "y_std": torch.ones(OUTPUT_LAYOUT.size, dtype=torch.float32),
        }
        self.dataset_digest = "dataset"


def physical_output(
    *,
    body_position: float = 0.0,
    body_velocity: float = 0.0,
    planar_velocity: tuple[float, float] = (0.0, 0.0),
    phase_advance: float = 0.1,
    joints: float = 0.0,
    contact_logits: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    output = torch.zeros(OUTPUT_LAYOUT.size, dtype=torch.float32)
    trajectory = output[OUTPUT_LAYOUT["trajectory_position"]].reshape(12, 2)
    trajectory[:, 0] = torch.as_tensor(TRAJECTORY_TIMES_S, dtype=torch.float32) * 0.12
    direction = output[OUTPUT_LAYOUT["trajectory_direction"]].reshape(12, 2)
    direction[:, 0] = 1.0
    output[OUTPUT_LAYOUT["body_position"]] = body_position
    output[OUTPUT_LAYOUT["body_velocity"]] = body_velocity
    output[OUTPUT_LAYOUT["root_height"]] = 0.8
    output[OUTPUT_LAYOUT["joint_position"]] = joints
    output[OUTPUT_LAYOUT["root_planar_velocity"]] = torch.tensor(planar_velocity)
    output[OUTPUT_LAYOUT["phase_advance"]] = phase_advance
    output[OUTPUT_LAYOUT["contact_logit"]] = torch.tensor(contact_logits)
    return output


class FakeModel(torch.nn.Module):
    def __init__(self, outputs: list[torch.Tensor]) -> None:
        super().__init__()
        self.outputs = list(outputs)
        self.inputs: list[torch.Tensor] = []
        self.phases: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x.detach().cpu().clone())
        self.phases.append(phase.detach().cpu().clone())
        if not self.outputs:
            raise RuntimeError("no deterministic output remains")
        return self.outputs.pop(0).reshape(1, -1).to(x.device)


def terrain_with_grade(degrees: float):
    tangent = math.tan(math.radians(degrees))

    def sample(xy: np.ndarray) -> TerrainSample:
        point = np.asarray(xy, dtype=np.float64)
        return TerrainSample(
            height_m=float(tangent * point[0]),
            gradient_xy=np.array((tangent, 0.0), dtype=np.float64),
        )

    return sample


class TerrainPFNNRuntimeTests(unittest.TestCase):
    @staticmethod
    def bind_identity_bootstrap(checkpoint: FakeCheckpoint) -> np.ndarray:
        seed = checkpoint.runtime_seed
        raw = np.zeros(INPUT_LAYOUT.size, dtype=np.float32)
        raw[INPUT_LAYOUT["trajectory_position"]] = np.asarray(
            seed["trajectory_position"]
        ).reshape(-1)
        raw[INPUT_LAYOUT["trajectory_direction"]] = np.asarray(
            seed["trajectory_direction"]
        ).reshape(-1)
        raw[INPUT_LAYOUT["terrain_height"]] = np.asarray(
            seed["terrain_height"]
        ).reshape(-1)
        raw[INPUT_LAYOUT["semantic_intent"]] = np.asarray(
            seed["semantic_intent"]
        ).reshape(-1)
        raw[INPUT_LAYOUT["previous_body_position"]] = np.asarray(
            seed["body_position"]
        ).reshape(-1)
        raw[INPUT_LAYOUT["previous_body_velocity"]] = np.asarray(
            seed["body_velocity"]
        ).reshape(-1)
        normalized = raw.copy()
        for field in ("previous_body_position", "previous_body_velocity"):
            normalized[INPUT_LAYOUT[field]] *= 0.1
        seed["normalized_input"] = torch.as_tensor(normalized.copy())
        seed["normalized_input_sha256"] = pfnn_input_sha256(normalized)
        return normalized

    def make_runtime(
        self,
        model: FakeModel,
        *,
        terrain=terrain_with_grade(0.0),
        checkpoint: FakeCheckpoint | None = None,
        enforce_motion_envelope: bool = True,
        command_driven_root: bool = False,
    ) -> TerrainPFNNRuntime:
        return TerrainPFNNRuntime(
            checkpoint=FakeCheckpoint() if checkpoint is None else checkpoint,
            kinematics=FakeKinematics(),
            model=model,
            height_and_grade_at=terrain,
            enforce_motion_envelope=enforce_motion_envelope,
            command_driven_root=command_driven_root,
        )

    def test_checkpoint_seed_is_the_exact_reached_bootstrap_state(self) -> None:
        checkpoint = FakeCheckpoint()
        seed = checkpoint.runtime_seed
        seed["phase"] = torch.tensor(1.25, dtype=torch.float32)
        seed["root_height"] = torch.tensor(0.91, dtype=torch.float32)
        seed["root_tilt"] = torch.tensor((0.05, -0.02), dtype=torch.float32)
        seed["joint_position"] = torch.linspace(-0.2, 0.2, 29)
        seed["body_position"] = torch.arange(90, dtype=torch.float32).reshape(30, 3) / 100.0
        seed["body_velocity"] = -seed["body_position"]
        seed["trajectory_position"] = torch.column_stack(
            (torch.linspace(-1.0, 1.0, 12), torch.full((12,), 0.25))
        )
        runtime = self.make_runtime(FakeModel([physical_output()]), checkpoint=checkpoint)
        frame = runtime.frame
        np.testing.assert_array_equal(
            frame.root_position_world,
            (0.0, 0.0, float(seed["root_height"])),
        )
        np.testing.assert_array_equal(
            frame.joint_position_isaaclab,
            seed["joint_position"].numpy(),
        )
        self.assertEqual(frame.phase, float(seed["phase"]))
        np.testing.assert_allclose(
            frame.trajectory.position_world_xy,
            seed["trajectory_position"].numpy(),
            atol=1.0e-7,
        )
        np.testing.assert_array_equal(runtime.previous_body_position, seed["body_position"])
        np.testing.assert_array_equal(runtime.previous_body_velocity, seed["body_velocity"])

    def test_first_inference_defers_command_blend_and_matches_seed_receipt(self) -> None:
        checkpoint = FakeCheckpoint()
        trajectory = checkpoint.runtime_seed["trajectory_position"].reshape(12, 2)
        trajectory[:, 0] = torch.as_tensor(TRAJECTORY_TIMES_S, dtype=torch.float32)
        checkpoint.runtime_seed["semantic_intent"][:] = torch.tensor((0.0, 1.0))
        expected = self.bind_identity_bootstrap(checkpoint)
        model = FakeModel([physical_output(joints=0.1)])
        runtime = self.make_runtime(model, checkpoint=checkpoint)
        frame = runtime.step(np.array((-0.8, 0.0)), camera_yaw=0.0)
        self.assertNotIn("hold_reason", frame.diagnostics)
        np.testing.assert_array_equal(model.inputs[0][0].numpy(), expected)
        np.testing.assert_array_equal(
            model.inputs[0][0, INPUT_LAYOUT["semantic_intent"]].reshape(12, 2),
            np.tile((0.0, 1.0), (12, 1)),
        )

    def test_second_input_contains_first_prediction_not_teacher_state(self) -> None:
        model = FakeModel(
            [
                physical_output(body_position=1.5, body_velocity=-2.0),
                physical_output(body_position=0.25, body_velocity=0.5),
            ]
        )
        runtime = self.make_runtime(model)
        runtime.step(np.zeros(2), camera_yaw=0.0)
        runtime.step(np.zeros(2), camera_yaw=0.0)
        second = model.inputs[1][0].numpy()
        np.testing.assert_allclose(
            second[INPUT_LAYOUT["previous_body_position"]], 0.15, atol=1.0e-7
        )
        np.testing.assert_allclose(
            second[INPUT_LAYOUT["previous_body_velocity"]], -0.20, atol=1.0e-7
        )
        self.assertTrue(
            np.any(np.abs(second[INPUT_LAYOUT["trajectory_position"]]) > 1.0e-5)
        )

    def test_tick_one_input_matches_direct_shared_recurrence_kernel(self) -> None:
        checkpoint = FakeCheckpoint()
        seed = checkpoint.runtime_seed
        seed_position = seed["trajectory_position"].reshape(12, 2)
        seed_position[:, 0] = torch.as_tensor(TRAJECTORY_TIMES_S, dtype=torch.float32)
        self.bind_identity_bootstrap(checkpoint)
        first_prediction = physical_output(body_position=0.4, body_velocity=-0.2)
        model = FakeModel([first_prediction, physical_output()])
        runtime = self.make_runtime(model, checkpoint=checkpoint)

        runtime.step(np.zeros(2), camera_yaw=0.0)
        runtime.step(np.array((0.3, 0.0)), camera_yaw=0.0)
        captured_x_tick1 = model.inputs[1][0].numpy()

        state = initialize_recurrent_state(
            trajectory_position_local=seed["trajectory_position"].reshape(1, 12, 2),
            trajectory_direction_local=seed["trajectory_direction"].reshape(1, 12, 2),
            semantic_intent=seed["semantic_intent"].reshape(1, 12, 2),
            previous_body_position_local=seed["body_position"].reshape(1, 30, 3),
            previous_body_velocity_local=seed["body_velocity"].reshape(1, 30, 3),
            phase=seed["phase"].reshape(1),
            root_world_xy=seed["world_xy"].reshape(1, 2),
            root_yaw_world=seed["world_yaw"].reshape(1),
        )
        bootstrap_planned = PlannedTrajectory(
            position_world_xy=state.predicted_position_world_xy,
            direction_world_xy=state.predicted_direction_world_xy,
            semantic_intent=seed["semantic_intent"].reshape(1, 12, 2),
        )
        state = advance_recurrent_state(
            state,
            bootstrap_planned,
            first_prediction.reshape(1, -1),
            phase_advance_cap=torch.tensor((0.3,), dtype=torch.float32),
        )
        planned = plan_recurrent_trajectory(
            state, torch.tensor(((0.3, 0.0),), dtype=torch.float32)
        )
        direct_x_tick1 = pack_recurrent_input(
            state=state,
            planned=planned,
            terrain_height=torch.zeros((1, 12, 3), dtype=torch.float32),
            x_mean=checkpoint.normalization["x_mean"],
            x_std=checkpoint.normalization["x_std"],
        )[0].numpy()

        np.testing.assert_allclose(captured_x_tick1, direct_x_tick1, atol=3e-6, rtol=0.0)
        self.assertEqual(runtime.frame.diagnostics.get("hold_reason"), None)

    def test_future_blend_uses_normalized_knot_time_and_interval_velocity(self) -> None:
        checkpoint = FakeCheckpoint()
        trajectory = checkpoint.runtime_seed["trajectory_position"].reshape(12, 2)
        trajectory[:, 0] = torch.as_tensor(TRAJECTORY_TIMES_S, dtype=torch.float32)
        self.bind_identity_bootstrap(checkpoint)
        model = FakeModel([physical_output(), physical_output()])
        runtime = self.make_runtime(model, checkpoint=checkpoint)
        runtime.step(np.zeros(2), camera_yaw=0.0)
        runtime.step(np.array((0.4, 0.0)), camera_yaw=0.0)
        packed = model.inputs[1][0, INPUT_LAYOUT["trajectory_position"]].reshape(12, 2)
        # The author's non-responsive PFNN blend uses 1-(1-u)^0.5.
        weight = 1.0 - math.sqrt(1.0 - 0.2)
        expected_velocity = (1.0 - weight) * 0.12 + weight * 0.4
        self.assertAlmostEqual(float(packed[7, 0]), expected_velocity / 6.0, places=6)
        self.assertAlmostEqual(float(packed[6, 0]), 0.0, places=7)

    def test_unsupported_future_is_collapsed_before_packing_and_model_is_stationary(self) -> None:
        calls: list[np.ndarray] = []

        def terrain(xy: np.ndarray) -> TerrainSample:
            point = np.asarray(xy, dtype=np.float64)
            calls.append(point.copy())
            degrees = 21.0 if point[0] > 0.01 else 0.0
            tangent = math.tan(math.radians(degrees))
            return TerrainSample(0.0, np.array((tangent, 0.0)))

        model = FakeModel([physical_output(), physical_output()])
        runtime = self.make_runtime(model, terrain=terrain)
        runtime.step(np.zeros(2), camera_yaw=0.0)
        frame = runtime.step(np.array((0.35, 0.0)), camera_yaw=0.0)
        self.assertEqual(len(model.inputs), 2)
        self.assertFalse(frame.supported)
        self.assertEqual(frame.diagnostics["desired_speed_m_s"], 0.0)
        packed = model.inputs[1][0].numpy()
        future = packed[INPUT_LAYOUT["trajectory_position"]].reshape(12, 2)[6:]
        np.testing.assert_allclose(future, 0.0, atol=1.0e-7)
        terrain_feature = packed[INPUT_LAYOUT["terrain_height"]]
        self.assertTrue(np.isfinite(terrain_feature).all())
        self.assertLessEqual(float(np.max(np.abs(terrain_feature))), 1.0e-7)
        self.assertTrue(any(point[0] > 0.01 for point in calls))

    def test_nineteen_degree_calls_model_advances_phase_and_uses_raw_logits(self) -> None:
        checkpoint = FakeCheckpoint()
        checkpoint.runtime_seed["phase"] = torch.tensor(
            2.0 * math.pi - 0.1, dtype=torch.float32
        )
        model = FakeModel(
            [physical_output(phase_advance=0.9, contact_logits=(-2.0, 0.0, 2.0, 4.0))]
        )
        runtime = self.make_runtime(model, terrain=terrain_with_grade(19.0), checkpoint=checkpoint)
        frame = runtime.step(np.zeros(2), camera_yaw=0.0)
        self.assertEqual(len(model.inputs), 1)
        self.assertTrue(frame.supported)
        self.assertAlmostEqual(frame.phase, 0.2, places=5)  # q99 cap is 0.3
        expected = torch.sigmoid(torch.tensor((-2.0, 0.0, 2.0, 4.0))).numpy()
        np.testing.assert_allclose(frame.contact_probability, expected, atol=1.0e-7)

    def test_invalid_prediction_holds_every_runtime_state_transactionally(self) -> None:
        model = FakeModel(
            [
                physical_output(body_position=0.3),
                physical_output(joints=1.0),  # step exceeds 0.25 rad
                physical_output(body_position=0.6),
            ]
        )
        runtime = self.make_runtime(model)
        accepted = runtime.step(np.zeros(2), camera_yaw=0.0)
        accepted_recurrent_state = runtime._recurrent_state
        held = runtime.step(np.zeros(2), camera_yaw=0.0)
        self.assertEqual(held.diagnostics["hold_reason"], "joint_step")
        self.assertIs(runtime._recurrent_state, accepted_recurrent_state)
        np.testing.assert_array_equal(held.root_position_world, accepted.root_position_world)
        np.testing.assert_array_equal(held.joint_position_isaaclab, accepted.joint_position_isaaclab)
        self.assertEqual(held.phase, accepted.phase)
        third = runtime.step(np.zeros(2), camera_yaw=0.0)
        np.testing.assert_array_equal(model.inputs[1], model.inputs[2])
        self.assertEqual(third.diagnostics["hold_count"], 1)

    def test_preview_mode_commits_finite_pose_and_reports_envelope_violation(self) -> None:
        model = FakeModel(
            [physical_output(joints=0.1), physical_output(joints=1.0)]
        )
        runtime = self.make_runtime(model, enforce_motion_envelope=False)

        first = runtime.step(np.zeros(2), camera_yaw=0.0)
        second = runtime.step(np.zeros(2), camera_yaw=0.0)

        self.assertNotIn("hold_reason", second.diagnostics)
        self.assertIn("joint_step", second.diagnostics["preview_envelope_violations"])
        self.assertEqual(second.diagnostics["hold_count"], 0)
        self.assertGreater(
            float(np.max(np.abs(second.joint_position_isaaclab - first.joint_position_isaaclab))),
            0.25,
        )

    def test_command_driven_preview_root_tracks_command_and_stops_on_release(self) -> None:
        model = FakeModel(
            [
                physical_output(planar_velocity=(0.0, 1.0)),
                physical_output(planar_velocity=(0.0, 1.0)),
            ]
        )
        runtime = self.make_runtime(
            model,
            enforce_motion_envelope=False,
            command_driven_root=True,
        )

        moving = runtime.step(np.asarray((0.6, 0.0)), camera_yaw=0.0)
        stopped = runtime.step(np.zeros(2), camera_yaw=0.0)

        self.assertAlmostEqual(moving.root_position_world[0], 0.6 / 30.0, places=7)
        self.assertAlmostEqual(moving.root_position_world[1], 0.0, places=7)
        np.testing.assert_allclose(
            stopped.root_position_world[:2], moving.root_position_world[:2], atol=1.0e-7
        )
        np.testing.assert_array_equal(
            stopped.joint_position_isaaclab, moving.joint_position_isaaclab
        )
        self.assertEqual(stopped.phase, moving.phase)
        self.assertTrue(stopped.diagnostics["idle_pose_held"])
        self.assertTrue(moving.diagnostics["command_driven_root"])

    def test_current_missing_terrain_holds_without_calling_model(self) -> None:
        model = FakeModel([physical_output()])
        calls = 0

        def disappears_after_bootstrap(_xy: np.ndarray) -> TerrainSample | None:
            nonlocal calls
            calls += 1
            return TerrainSample(0.0, np.zeros(2)) if calls == 1 else None

        runtime = self.make_runtime(model, terrain=disappears_after_bootstrap)
        frame = runtime.step(np.zeros(2), camera_yaw=0.0)
        self.assertEqual(model.inputs, [])
        self.assertIn("terrain", frame.diagnostics["hold_reason"])

    def test_predicted_root_terrain_query_uses_exact_torch_commit_coordinate(self) -> None:
        boundary = 1.0 / 30.0

        def terrain(xy: np.ndarray) -> TerrainSample:
            point = np.asarray(xy, dtype=np.float64)
            degrees = 21.0 if point[0] > boundary else 0.0
            tangent = math.tan(math.radians(degrees))
            return TerrainSample(0.0, np.array((tangent, 0.0)))

        model = FakeModel([physical_output(planar_velocity=(1.0, 0.0))])
        runtime = self.make_runtime(model, terrain=terrain)
        initial_state = runtime._recurrent_state

        frame = runtime.step(np.zeros(2), camera_yaw=0.0)

        self.assertEqual(frame.diagnostics["hold_reason"], "unsupported_predicted_root_terrain")
        self.assertIs(runtime._recurrent_state, initial_state)
        np.testing.assert_array_equal(frame.root_position_world[:2], (0.0, 0.0))

    def test_seed_world_xy_and_yaw_are_honored(self) -> None:
        checkpoint = FakeCheckpoint()
        checkpoint.runtime_seed["world_xy"] = torch.tensor((2.0, -1.0))
        checkpoint.runtime_seed["world_yaw"] = torch.tensor(math.pi / 2.0)
        local = checkpoint.runtime_seed["trajectory_position"].reshape(12, 2)
        local[:, 0] = torch.linspace(-0.2, 0.8, 12)
        local_direction = checkpoint.runtime_seed["trajectory_direction"].reshape(12, 2)
        local_direction[:] = torch.tensor((1.0, 0.0))
        self.bind_identity_bootstrap(checkpoint)
        runtime = self.make_runtime(FakeModel([physical_output()]), checkpoint=checkpoint)
        np.testing.assert_allclose(runtime.frame.root_position_world[:2], (2.0, -1.0))
        expected_position = np.column_stack(
            (2.0 - local[:, 1].numpy(), -1.0 + local[:, 0].numpy())
        )
        np.testing.assert_allclose(
            runtime.frame.trajectory.position_world_xy, expected_position, atol=1.0e-7
        )
        np.testing.assert_allclose(
            runtime.frame.trajectory.direction_world_xy,
            np.tile((0.0, 1.0), (12, 1)),
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            runtime.frame.root_quaternion_world_wxyz,
            (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
            atol=1.0e-7,
        )


class ClosedLoopMetricsTests(unittest.TestCase):
    @staticmethod
    def frame(index: int, *, phase: float, supported: bool = True) -> PFNNRuntimeFrame:
        direction = np.zeros((12, 2), np.float64)
        direction[:, 0] = 1.0
        from mm_sonic.terrain_pfnn.runtime import PFNNTrajectoryState

        return PFNNRuntimeFrame(
            root_position_world=np.array((0.01 * index, 0.0, 0.8)),
            root_quaternion_world_wxyz=np.array((1.0, 0.0, 0.0, 0.0)),
            joint_position_isaaclab=np.zeros(29),
            phase=phase,
            contact_probability=np.ones(4),
            trajectory=PFNNTrajectoryState(
                np.zeros((12, 2)), direction, np.tile((0.0, 1.0), (12, 1))
            ),
            supported=supported,
            diagnostics={"hold_count": 0, "phase_advance": 0.1},
        )

    def test_recorder_emits_global_and_all_segment_gates_and_detects_freeze(self) -> None:
        recorder = ClosedLoopRecorder(joint_limits=np.array([[-2.0, 2.0]] * 29))
        observation = RuntimeGeometryObservation(
            maximum_sole_penetration_m=0.01,
            maximum_forbidden_body_penetration_m=0.0,
            forbidden_geom_names=(),
            stance_sole_speeds_m_s=(0.05, 0.06),
        )
        phase = 0.0
        segment_names = ("flat", "ascent", "summit", "descent", "landing")
        for index in range(600):
            if index >= 30:
                phase = (phase + 0.1) % (2.0 * math.pi)
            recorder.record(
                self.frame(index, phase=phase),
                desired_velocity_world=np.array((0.3, 0.0)),
                terrain_sample=TerrainSample(0.0, np.zeros(2), segment_names[index % 5]),
                geometry=observation,
                traversal_direction="forward" if index < 300 else "backward",
            )
        report = recorder.finalize()
        self.assertEqual(report["schema"], "mm-sonic-terrain-pfnn-closed-loop/v1")
        self.assertEqual(set(report["segments"]), set(segment_names))
        self.assertIn("gates", report["segments"]["flat"])
        self.assertTrue(report["gates"]["finite_20_seconds"])
        self.assertGreaterEqual(report["global"]["walking_freeze_count"], 1)
        self.assertFalse(report["gates"]["no_phase_freeze"])
        self.assertEqual(report["global"]["maximum_sole_penetration_m"], 0.01)

    def test_zero_predicted_stance_samples_and_stationary_tagged_traversal_fail(self) -> None:
        recorder = ClosedLoopRecorder(joint_limits=np.array([[-2.0, 2.0]] * 29))
        no_stance = RuntimeGeometryObservation(0.0, 0.0, (), ())
        grade = math.tan(math.radians(19.0))
        for index in range(600):
            recorder.record(
                self.frame(0, phase=(0.1 * index) % (2.0 * math.pi)),
                desired_velocity_world=np.array((0.3, 0.0)),
                terrain_sample=TerrainSample(0.0, np.array((grade, 0.0))),
                geometry=no_stance,
                traversal_direction="forward" if index < 300 else "backward",
            )
        report = recorder.finalize()
        self.assertEqual(report["global"]["predicted_stance_sample_count"], 0)
        self.assertFalse(report["gates"]["stance_sole_speed_within_limit"])
        self.assertFalse(report["gates"]["traverses_18_9_degrees_both_directions"])
        self.assertEqual(
            report["traversal"]["maximum_grade_degrees_by_direction"],
            {"forward": 0.0, "backward": 0.0},
        )

    def test_realized_displacement_drives_bidirectional_traversal_without_tags(self) -> None:
        recorder = ClosedLoopRecorder(joint_limits=np.array([[-2.0, 2.0]] * 29))
        observation = RuntimeGeometryObservation(0.0, 0.0, (), (0.02,))
        tangent = math.tan(math.radians(19.0))
        positions = [*range(300), *range(300, 0, -1)]
        for tick, position in enumerate(positions):
            recorder.record(
                self.frame(position, phase=(0.1 * tick) % (2.0 * math.pi)),
                desired_velocity_world=np.array((0.3 if tick < 300 else -0.3, 0.0)),
                terrain_sample=TerrainSample(0.0, np.array((tangent, 0.0))),
                geometry=observation,
                traversal_direction=None,
            )
        report = recorder.finalize()
        self.assertTrue(report["gates"]["traverses_18_9_degrees_both_directions"])
        self.assertGreater(report["traversal"]["realized_displacement_m"]["forward"], 1.0)
        self.assertGreater(report["traversal"]["realized_displacement_m"]["backward"], 1.0)

    def test_validation_penalty_counts_boolean_failures_and_numeric_excess(self) -> None:
        metrics = {
            "gates": {
                "finite_20_seconds": True,
                "no_phase_reversal": False,
                "root_translation_step_within_limit": False,
            },
            "numeric_gates": {
                "root_translation_step_m": {
                    "value": 0.09,
                    "limit": 0.06,
                    "gate": "root_translation_step_within_limit",
                },
                "joint_step_rad": {"value": 0.10, "limit": 0.25},
            },
            "dataset_digest_sha256": "data",
            "kinematic_signature_sha256": "kin",
        }
        result = ClosedLoopValidationResult.from_metrics(metrics, split="validation")
        self.assertEqual(result.hard_failures, 3)
        self.assertAlmostEqual(result.normalized_excess, 0.5)
        self.assertEqual(result.failure_penalty, 3000.5)

    def test_validation_scenario_aggregate_retains_selection_provenance(self) -> None:
        calls: list[int] = []

        manifest = {
            "dataset_digest_sha256": "d" * 64,
            "split_identities": {
                "train": ["train-a"],
                "validation": ["val-a", "val-b"],
                "test": ["test-a"],
            },
        }
        signature = "e" * 64
        identity_receipt = validation_identity_set_receipt(manifest)

        def scenario(index: int) -> ClosedLoopScenario:
            def evaluate() -> dict[str, object]:
                calls.append(index)
                return {
                    "gates": {"finite_20_seconds": True},
                    "numeric_gates": {},
                }

            return make_validation_scenario(
                manifest=manifest,
                kinematic_signature_sha256=signature,
                scenario_id=f"hill-{index}",
                terrain_source_receipt={
                    "kind": "procedural_three_hill",
                    "source_sha256": f"{index + 1:064x}",
                    "version": "task8/v1",
                },
                evaluator=evaluate,
            )

        result = evaluate_closed_loop_scenarios(
            split="validation",
            scenarios=(scenario(0), scenario(1)),
            expected_dataset_digest="d" * 64,
            expected_kinematic_signature_sha256=signature,
            expected_identity_set_sha256=identity_receipt["receipt_sha256"],
        )
        self.assertEqual(calls, [0, 1])
        self.assertEqual(result.metrics["scenario_count"], 2)
        self.assertEqual(result.metrics["dataset_digest_sha256"], "d" * 64)
        self.assertEqual(result.metrics["kinematic_signature_sha256"], signature)
        self.assertEqual(
            result.metrics["validation_identity_set_sha256"],
            identity_receipt["receipt_sha256"],
        )
        selection, promote = validated_normal_selection(
            one_step_score=2.5,
            result=result,
            dataset_digest="d" * 64,
            kinematic_signature_sha256=signature,
            validation_identity_set_sha256=identity_receipt["receipt_sha256"],
            expected_scenario_provenance_sha256=(
                result.metrics["scenario_provenance_sha256"]
            ),
        )
        self.assertTrue(promote)
        self.assertEqual(selection["validation_score"], 2.5)

    def test_validation_scenario_provenance_rejects_before_callbacks_open(self) -> None:
        calls: list[str] = []
        manifest = {
            "dataset_digest_sha256": "d" * 64,
            "split_identities": {
                "train": ["train"], "validation": ["val"], "test": ["test"]
            },
        }
        signature = "e" * 64
        receipt = validation_identity_set_receipt(manifest)

        def callback() -> dict[str, object]:
            calls.append("opened")
            return {"gates": {"finite_20_seconds": True}, "numeric_gates": {}}

        scenario = make_validation_scenario(
            manifest=manifest,
            kinematic_signature_sha256=signature,
            scenario_id="sealed-hill",
            terrain_source_receipt={
                "kind": "procedural_three_hill",
                "source_sha256": "a" * 64,
                "version": "task8/v1",
            },
            evaluator=callback,
        )
        forged = replace(
            scenario,
            provenance={**scenario.provenance, "dataset_digest_sha256": "f" * 64},
        )
        with self.assertRaisesRegex(ValueError, "provenance"):
            evaluate_closed_loop_scenarios(
                split="validation",
                scenarios=(scenario, forged),
                expected_dataset_digest="d" * 64,
                expected_kinematic_signature_sha256=signature,
                expected_identity_set_sha256=receipt["receipt_sha256"],
            )
        self.assertEqual(calls, [])

    def test_validation_rejects_train_or_test_before_opening_any_scenario(self) -> None:
        calls: list[str] = []

        def scenario() -> dict[str, object]:
            calls.append("opened")
            return {"gates": {}, "numeric_gates": {}}

        for split in ("train", "test"):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, "validation"):
                    evaluate_closed_loop_scenarios(
                        split=split,
                        scenarios=(scenario,),
                        expected_dataset_digest="d" * 64,
                        expected_kinematic_signature_sha256="e" * 64,
                        expected_identity_set_sha256="f" * 64,
                    )
        self.assertEqual(calls, [])

    @unittest.skipUnless(MODEL_PATH.is_file(), "native G1 model is unavailable")
    def test_native_geometry_uses_exact_eight_sole_spheres(self) -> None:
        geometry = NativeG1RuntimeGeometry.from_mjcf(MODEL_PATH)
        observation = geometry.observe(
            root_position_world=np.array((0.0, 0.0, 0.8)),
            root_quaternion_world_wxyz=np.array((1.0, 0.0, 0.0, 0.0)),
            joint_position_isaaclab=np.zeros(29),
            contact_probability=np.ones(4),
            height_and_grade_at=terrain_with_grade(0.0),
        )
        self.assertEqual(geometry.sole_geom_count, 8)
        self.assertTrue(math.isfinite(observation.maximum_sole_penetration_m))
        self.assertTrue(math.isfinite(observation.maximum_forbidden_body_penetration_m))
        self.assertFalse(any("ankle_roll" in name for name in observation.forbidden_geom_names))
        collision = {
            geom
            for geom in range(geometry._model.ngeom)
            if int(geometry._model.geom_bodyid[geom]) in geometry._body_ids
            and (
                int(geometry._model.geom_contype[geom])
                or int(geometry._model.geom_conaffinity[geom])
            )
        }
        self.assertFalse(geometry._sole_set & set(geometry._forbidden_geom_ids))
        self.assertEqual(geometry._sole_set | set(geometry._forbidden_geom_ids), collision)

    @unittest.skipUnless(MODEL_PATH.is_file(), "native G1 model is unavailable")
    def test_native_mesh_audit_queries_every_transformed_collision_vertex(self) -> None:
        import mujoco

        geometry = NativeG1RuntimeGeometry.from_mjcf(MODEL_PATH)
        mesh_geom = next(
            geom
            for geom in geometry._forbidden_geom_ids
            if int(geometry._model.geom_type[geom])
            == int(mujoco.mjtGeom.mjGEOM_MESH)
        )
        mesh_id = int(geometry._model.geom_dataid[mesh_geom])
        vertex_count = int(geometry._model.mesh_vertnum[mesh_id])
        samples = geometry._geom_samples(mesh_geom)
        self.assertEqual(len(samples), vertex_count)

    def test_primitive_collision_surfaces_catch_uphill_side_penetration(self) -> None:
        tangent = math.tan(math.radians(19.0))
        normal = np.array((tangent, 0.0, -1.0))
        rotation = np.eye(3)
        position = np.zeros(3)
        sphere = runtime_module._deterministic_collision_surface_samples(
            "sphere", np.array((0.2, 0.0, 0.0)), rotation, position
        )
        capsule = runtime_module._deterministic_collision_surface_samples(
            "capsule", np.array((0.1, 0.3, 0.0)), rotation, position
        )
        box = runtime_module._deterministic_collision_surface_samples(
            "box", np.array((0.2, 0.1, 0.1)), rotation, position
        )
        self.assertGreater(np.max(sphere @ normal), 0.2)
        self.assertGreater(np.max(capsule @ normal), 0.1)
        self.assertAlmostEqual(
            float(np.max(box @ normal)), 0.2 * tangent + 0.1, places=7
        )

    def test_vectorized_collision_penetration_matches_every_scalar_slope_sample(self) -> None:
        mesh = CanonicalTerrainMesh(
            vertices_local=np.asarray(
                ((-2.0, -2.0, -1.0), (2.0, -2.0, 1.0),
                 (2.0, 2.0, 1.0), (-2.0, 2.0, -1.0)),
                dtype=np.float32,
            ),
            faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
            valid_faces=np.ones(2, dtype=np.bool_),
            source_asset_sha256="f" * 64,
        )
        terrain = SourceAlignedTerrain(
            mesh,
            world_from_mesh=RigidTransform(
                np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0))
            ),
            source_root_xy=(0.0, 0.0), source_root_yaw=0.0,
            source_support_height=0.0,
        )
        points = np.asarray(
            [
                (x, y, 0.05 + 0.01 * ((index % 5) - 2))
                for index, (x, y) in enumerate(
                    (tuple(value) for value in np.random.default_rng(7).uniform(-1.9, 1.9, (257, 2)))
                )
            ],
            dtype=np.float64,
        )
        scalar = max(
            0.0,
            max(terrain(point[:2]).height_m - point[2] for point in points),
        )
        heights = terrain.collision_heights_at(points[:, :2])
        self.assertEqual(heights.shape, (len(points),))
        vectorized = NativeG1RuntimeGeometry._penetration(points, terrain)
        self.assertAlmostEqual(vectorized, scalar, places=12)

    def test_known_train_source_and_twenty_second_script_are_exact(self) -> None:
        manifest = {
            "source_records": {
                "z-grail": {"source_kind": "grail", "split": "train"},
                "a-test": {"source_kind": "grail", "split": "test"},
                "b-grail": {"source_kind": "grail", "split": "train"},
                "a-lafan": {"source_kind": "lafan", "split": "train"},
            }
        }
        name, record = select_known_train_grail_record(manifest, split="train")
        self.assertEqual(name, "b-grail")
        self.assertEqual(record["split"], "train")
        expected = {
            0: ((0.2, 0.1), "forward"),
            59: ((0.2, 0.1), "forward"),
            60: ((0.35, 0.0), "forward"),
            359: ((0.35, 0.0), "forward"),
            360: ((0.0, 0.0), None),
            419: ((0.0, 0.0), None),
            420: ((-0.35, 0.0), "backward"),
            539: ((-0.35, 0.0), "backward"),
            540: ((0.0, 0.0), None),
            599: ((0.0, 0.0), None),
        }
        for tick, (velocity, direction) in expected.items():
            with self.subTest(tick=tick):
                actual_velocity, actual_direction = known_train_command(
                    tick, (0.2, 0.1)
                )
                np.testing.assert_array_equal(actual_velocity, velocity)
                self.assertEqual(actual_direction, direction)
        with self.assertRaisesRegex(ValueError, "600"):
            known_train_command(600, (0.2, 0.1))

    def test_usd_decode_is_numeric_hash_bound_and_fails_closed(self) -> None:
        usd_text = """#usda 1.0
(
    defaultPrim = "Terrain"
    metersPerUnit = 1
    upAxis = "Z"
)
def Mesh "Terrain" {
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(0,0,0), (1,0,0), (1,1,0), (0,1,0)]
    uniform token subdivisionScheme = "none"
}
"""
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            usd = root / "terrain.usda"
            usd.write_text(usd_text)
            digest = hashlib.sha256(usd.read_bytes()).hexdigest()
            mesh = _decode_usd_mesh(usd, expected_sha256=digest)
            self.assertEqual(mesh.vertices_local.shape, (4, 3))
            self.assertEqual(mesh.faces.shape, (2, 3))
            with self.assertRaisesRegex(ValueError, "hash"):
                _decode_usd_mesh(usd, expected_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "helper"):
                _decode_usd_mesh(
                    usd,
                    expected_sha256=digest,
                    python_path=Path("/bin/false"),
                )
            tampered = root / "tampered.npz"
            np.savez(
                tampered,
                schema=np.asarray("mm-sonic-usd-mesh/v1"),
                source_sha256=np.asarray("f" * 64),
                vertices=np.zeros((3, 3), np.float32),
                faces=np.asarray(((0, 1, 2),), np.int32),
                valid_faces=np.ones(1, dtype=np.bool_),
            )
            with self.assertRaisesRegex(ValueError, "hash"):
                _load_decoded_usd_npz(tampered, expected_sha256=digest)

    def test_known_slope_wrapper_has_bounded_flat_apron_then_exact_flank(self) -> None:
        tangent = math.tan(math.radians(10.0))
        mesh = CanonicalTerrainMesh(
            vertices_local=np.asarray(
                (
                    (0.0, -0.5, 0.0),
                    (0.0, 0.5, 0.0),
                    (1.0, -0.5, tangent),
                    (1.0, 0.5, tangent),
                ),
                dtype=np.float32,
            ),
            faces=np.asarray(((0, 2, 3), (0, 3, 1)), dtype=np.int32),
            valid_faces=np.ones(2, dtype=np.bool_),
            source_asset_sha256="a" * 64,
        )
        terrain = KnownSlopeTerrain.from_mesh(mesh)
        origin = terrain(np.array((0.0, 0.0)))
        self.assertIsNotNone(origin)
        self.assertEqual(origin.segment, "flat")
        np.testing.assert_array_equal(origin.gradient_xy, (0.0, 0.0))
        self.assertEqual(terrain.alignment_report["approach"], "synthetic_flat_apron")
        boundary = terrain(np.array((0.5, 0.0)))
        self.assertIsNotNone(boundary)
        self.assertAlmostEqual(boundary.height_m, 0.0, places=6)
        flank = terrain(np.array((0.75, 0.0)))
        self.assertIsNotNone(flank)
        self.assertAlmostEqual(flank.absolute_grade_degrees, 10.0, places=4)
        self.assertIsNone(terrain(np.array((-0.251, 0.0))))

    def test_source_aligned_terrain_rotates_gradient_and_preserves_anchor(self) -> None:
        mesh = CanonicalTerrainMesh(
            vertices_local=np.asarray(
                ((-2.0, -2.0, -2.0), (2.0, -2.0, 2.0),
                 (2.0, 2.0, 2.0), (-2.0, 2.0, -2.0)),
                dtype=np.float32,
            ),
            faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
            valid_faces=np.ones(2, dtype=np.bool_),
            source_asset_sha256="a" * 64,
        )
        terrain = SourceAlignedTerrain(
            mesh,
            world_from_mesh=RigidTransform(
                np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0))
            ),
            source_root_xy=(0.0, 0.0),
            source_root_yaw=math.pi / 2.0,
            source_support_height=0.0,
        )
        origin = terrain(np.zeros(2))
        forward = terrain(np.array((0.2, 0.0)))
        self.assertAlmostEqual(origin.height_m, 0.0, places=7)
        self.assertAlmostEqual(forward.height_m, 0.0, places=7)
        np.testing.assert_allclose(origin.gradient_xy, (0.0, -1.0), atol=1.0e-7)
        self.assertIsNone(terrain(np.array((3.0, 0.0))))

    def test_source_aligned_seed_selects_supported_flat_and_crosses_exact_flank(self) -> None:
        rise = math.tan(math.radians(10.0))
        vertices = np.asarray(
            [
                (-2.0, -1.0, 0.0), (-2.0, 1.0, 0.0),
                (0.0, -1.0, 0.0), (0.0, 1.0, 0.0),
                (1.0, -1.0, rise), (1.0, 1.0, rise),
                (2.0, -1.0, rise), (2.0, 1.0, rise),
            ],
            dtype=np.float32,
        )
        mesh = CanonicalTerrainMesh(
            vertices_local=vertices,
            faces=np.asarray(
                (
                    (0, 2, 3), (0, 3, 1),
                    (2, 4, 5), (2, 5, 3),
                    (4, 6, 7), (4, 7, 5),
                ),
                dtype=np.int32,
            ),
            valid_faces=np.ones(6, dtype=np.bool_),
            source_asset_sha256="a" * 64,
        )
        positions = np.column_stack((np.linspace(-0.2, 0.2, 12), np.zeros(12)))
        directions = np.tile((1.0, 0.0), (12, 1))
        terrain = SourceAlignedTerrain.from_supported_flat_seed(
            mesh,
            world_from_mesh=RigidTransform(
                np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0))
            ),
            runtime_root_xy=(3.0, -4.0),
            runtime_root_yaw=math.pi / 3.0,
            seed_trajectory_position=positions,
            seed_trajectory_direction=directions,
            expected_relative_terrain=np.zeros((12, 3)),
        )
        self.assertEqual(terrain.alignment_report["approach"], "supported_flat_mesh")
        self.assertLessEqual(
            terrain.alignment_report["maximum_seed_probe_error_m"], 1.0e-4
        )
        self.assertGreaterEqual(
            terrain.alignment_report["native_maximum_grade_degrees"], 9.99
        )
        self.assertIsNotNone(terrain(np.array((3.0, -4.0))))
        self.assertIsNone(terrain(np.array((30.0, -4.0))))

    def test_supported_plateau_route_turns_before_boundary_without_post_flank_flat(self) -> None:
        rise = math.tan(math.radians(12.0)) * 1.5
        mesh = CanonicalTerrainMesh(
            vertices_local=np.asarray(
                (
                    (-1.0, -0.7, 0.0), (0.0, -0.7, 0.0),
                    (-1.0, 0.7, 0.0), (0.0, 0.7, 0.0),
                    (1.5, -0.7, rise), (1.5, 0.7, rise),
                ),
                dtype=np.float32,
            ),
            faces=np.asarray(
                ((0, 1, 3), (0, 3, 2), (1, 4, 5), (1, 5, 3)),
                dtype=np.int32,
            ),
            valid_faces=np.ones(4, dtype=np.bool_),
            source_asset_sha256="b" * 64,
        )
        positions = np.column_stack((np.linspace(-0.1, 0.05, 12), np.zeros(12)))
        terrain = SourceAlignedTerrain.from_supported_flat_seed(
            mesh,
            world_from_mesh=RigidTransform(
                np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0))
            ),
            runtime_root_xy=(0.0, 0.0),
            runtime_root_yaw=0.0,
            seed_trajectory_position=positions,
            seed_trajectory_direction=np.tile((1.0, 0.0), (12, 1)),
            expected_relative_terrain=np.zeros((12, 3)),
        )
        report = terrain.alignment_report
        self.assertFalse(report["has_post_flank_continuation"])
        self.assertGreater(report["turnaround_runtime_x_m"], report["flank_entry_runtime_x_m"])
        self.assertLess(report["turnaround_runtime_x_m"], report["supported_route_extent_m"])
        self.assertIsNotNone(
            terrain(np.array((report["turnaround_runtime_x_m"], 0.0)))
        )
        self.assertIsNone(
            terrain(np.array((report["turnaround_runtime_x_m"] + 0.01, 0.0)))
        )

        short_mesh = CanonicalTerrainMesh(
            vertices_local=np.asarray(
                (
                    (-0.35, -0.3, 0.0), (0.0, -0.3, 0.0),
                    (-0.35, 0.3, 0.0), (0.0, 0.3, 0.0),
                    (0.15, -0.3, rise), (0.15, 0.3, rise),
                ),
                dtype=np.float32,
            ),
            faces=np.asarray(
                ((0, 1, 3), (0, 3, 2), (1, 4, 5), (1, 5, 3)),
                dtype=np.int32,
            ),
            valid_faces=np.ones(4, dtype=np.bool_),
            source_asset_sha256="c" * 64,
        )
        with self.assertRaisesRegex(ValueError, "turnaround|route"):
            SourceAlignedTerrain.from_supported_flat_seed(
                short_mesh,
                world_from_mesh=RigidTransform(
                    np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0))
                ),
                runtime_root_xy=(0.0, 0.0), runtime_root_yaw=0.0,
                seed_trajectory_position=positions,
                seed_trajectory_direction=np.tile((1.0, 0.0), (12, 1)),
                expected_relative_terrain=np.zeros((12, 3)),
            )

    def test_grail_pair_npz_rejects_robot_or_frame_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pair.npz"
            fields = {
                "schema": np.asarray("mm-sonic-grail-pair/v1"),
                "source_sha256": np.asarray("a" * 64),
                "robot_sha256": np.asarray("b" * 64),
                "vertices": np.asarray(
                    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                    np.float32,
                ),
                "faces": np.asarray(((0, 1, 2),), np.int32),
                "valid_faces": np.ones(1, np.bool_),
                "anchor_root_xy": np.zeros(2, np.float32),
                "anchor_yaw": np.asarray(0.0, np.float32),
                "anchor_support": np.asarray(0.0, np.float32),
                "anchor_center_frame": np.asarray(82, np.int32),
            }
            np.savez(path, **fields)
            decoded = _load_decoded_grail_npz(
                path,
                expected_terrain_sha256="a" * 64,
                expected_robot_sha256="b" * 64,
                expected_center_frame=82,
            )
            self.assertEqual(decoded.anchor_center_frame, 82)
            with self.assertRaisesRegex(ValueError, "provenance"):
                _load_decoded_grail_npz(
                    path,
                    expected_terrain_sha256="a" * 64,
                    expected_robot_sha256="c" * 64,
                    expected_center_frame=82,
                )

    def test_normal_best_requires_evaluated_validation_score_and_provenance(self) -> None:
        data = "a" * 64
        signature = "b" * 64
        identity = "c" * 64
        scenario = "d" * 64
        metrics = {
            "gates": {"finite_20_seconds": True},
            "numeric_gates": {},
            "dataset_digest_sha256": data,
            "kinematic_signature_sha256": signature,
            "validation_identity_set_sha256": identity,
            "scenario_provenance_sha256": scenario,
        }
        result = ClosedLoopValidationResult.from_metrics(metrics, split="validation")
        selection, promote = validated_normal_selection(
            one_step_score=2.5,
            result=result,
            dataset_digest=data,
            kinematic_signature_sha256=signature,
            validation_identity_set_sha256=identity,
            expected_scenario_provenance_sha256=scenario,
        )
        self.assertTrue(promote)
        self.assertEqual(selection["closed_loop_score"], 0.0)
        self.assertEqual(selection["validation_score"], 2.5)
        for changed in (None, "wrong-data", "wrong-kin", "wrong-id", "wrong-scenario"):
            with self.subTest(changed=changed):
                if changed is None:
                    bad_result = ClosedLoopValidationResult(
                        evaluated=False,
                        split="validation",
                        hard_failures=0,
                        normalized_excess=0.0,
                        metrics=metrics,
                    )
                    actual_data, actual_signature = data, signature
                    actual_identity, actual_scenario = identity, scenario
                else:
                    bad_result = result
                    actual_data = "e" * 64 if "data" in changed else data
                    actual_signature = "e" * 64 if "kin" in changed else signature
                    actual_identity = "e" * 64 if "id" in changed else identity
                    actual_scenario = "e" * 64 if "scenario" in changed else scenario
                selection, promote = validated_normal_selection(
                    one_step_score=2.5,
                    result=bad_result,
                    dataset_digest=actual_data,
                    kinematic_signature_sha256=actual_signature,
                    validation_identity_set_sha256=actual_identity,
                    expected_scenario_provenance_sha256=actual_scenario,
                )
                self.assertIsNone(selection)
                self.assertFalse(promote)

    def test_fitted_transition_rejection_prevents_known_terrain_callback(self) -> None:
        manifest = {
            "dataset_digest_sha256": "a" * 64,
            "split_identities": {
                "train": ["slope_000"], "validation": ["slope_002"],
            },
        }

        class Dataset:
            split = "train"
            x_mean = np.zeros(INPUT_LAYOUT.size, np.float32)
            x_std = np.ones(INPUT_LAYOUT.size, np.float32)
            y_mean = np.zeros(OUTPUT_LAYOUT.size, np.float32)
            y_std = np.ones(OUTPUT_LAYOUT.size, np.float32)

            def __init__(self) -> None:
                classes = ("flat", "flat", "ascent", "descent", "transition")
                self.rows = [
                    self.row(center, classes[index])
                    for index, center in enumerate(range(10, 15))
                ]

            @staticmethod
            def row(center: int, terrain_class: str) -> dict[str, object]:
                y = np.zeros(OUTPUT_LAYOUT.size, np.float32)
                y[OUTPUT_LAYOUT["trajectory_direction"]] = np.tile((1.0, 0.0), 12)
                y[OUTPUT_LAYOUT["root_height"]] = 0.8
                y[OUTPUT_LAYOUT["phase_advance"]] = 0.1
                return {
                    "x": np.zeros(INPUT_LAYOUT.size, np.float32),
                    "y": y,
                    "phase": np.float32(0.1 * center),
                    "clip_id": "terrain_slopes__slope_000__000",
                    "center_frame": center,
                    "split_identity": "slope_000",
                    "split": "train",
                    "sequence_lane": "motion",
                    "terrain_class": terrain_class,
                }

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict[str, object]:
                return self.rows[index]

        dataset = Dataset()
        fitted_subset = fitted_subset_metadata(dataset, range(len(dataset)))
        active_pair_receipt = train_module.fitted_transition_pair_receipt(dataset)
        active_target_audit = training_module.fitted_target_envelope_audit(
            train_module.materialize_transition_pairs(dataset),
            normalization=dataset,
            joint_limits=torch.tensor([[-2.0, 2.0]] * 29, dtype=torch.float64),
            phase_advance_cap=1.5 * 0.2,
            contract=training_module.DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE,
        )
        maxima = {
            "absolute_output": 0.251,
            "root_translation_step_m": 0.0,
            "root_rotation_step_rad": 0.0,
            "joint_step_rad": 0.251,
            "joint_limit_excess_rad": 0.0,
            "root_height_m": 0.8,
            "root_quaternion_norm_error": 0.0,
            "trajectory_direction_norm_deviation": 0.0,
            "phase_advance_rad": 0.1,
        }
        failure = {
            "clip_id": "terrain_slopes__slope_000__000",
            "sequence_lane": "motion",
            "center_frame": 11,
            "field": "joint_position",
            "joint": "left_hip_pitch_joint",
            "value": 0.251,
            "limit": 0.25,
        }
        base = {
            "schema": "mm-sonic-fitted-transition-report/v1",
            "accepted": False,
            "sample_count": 1,
            "maxima": maxima,
            "first_failure": failure,
            "rows_sha256": "d" * 64,
        }
        rejected = FittedTransitionReport(
            accepted=False,
            sample_count=1,
            maxima=maxima,
            first_failure=failure,
            rows_sha256="d" * 64,
            report_sha256=hashlib.sha256(json.dumps(
                base, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode()).hexdigest(),
        )

        class Kinematics:
            kinematic_signature_sha256 = "kin"
            joint_limits = torch.tensor([[-2.0, 2.0]] * 29, dtype=torch.float64)

            def to(self, _device):
                return self

        class Checkpoint:
            kinematic_signature_sha256 = "kin"
            joint_limits = Kinematics.joint_limits.clone()
            train_identities = ("slope_000",)
            validation_identities = ("slope_002",)
            normalization = {
                "x_mean": torch.zeros(INPUT_LAYOUT.size),
                "x_std": torch.ones(INPUT_LAYOUT.size),
                "y_mean": torch.zeros(OUTPUT_LAYOUT.size),
                "y_std": torch.ones(OUTPUT_LAYOUT.size),
            }
            loss_weights = {}
            phase_advance_q99 = 0.2
            selection = {"one_step_score": 1.0, "provisional": True}

            def __init__(self) -> None:
                self.fitted_subset = fitted_subset
                self.physical_envelope_objective = (
                    training_module.physical_envelope_objective_payload(
                        training_module.DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE
                    )
                )
                self.physical_envelope_objective_sha256 = hashlib.sha256(
                    json.dumps(
                        self.physical_envelope_objective,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                self.fitted_pair_receipt = active_pair_receipt
                self.target_envelope_audit = active_target_audit

            @staticmethod
            def build_model():
                return torch.nn.Linear(INPUT_LAYOUT.size, OUTPUT_LAYOUT.size)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(json.dumps(manifest))
            checkpoint_path = root / "checkpoint.pt"
            checkpoint_path.write_bytes(b"candidate")
            output_path = root / "evaluation.json"
            checkpoint = Checkpoint()
            transition_devices: list[str] = []

            def transition_evaluation(
                model: torch.nn.Module, *args: object, **kwargs: object
            ) -> FittedTransitionReport:
                transition_devices.append(next(model.parameters()).device.type)
                return rejected

            with (
                patch(
                    "mm_sonic.evaluate_terrain_pfnn.TorchG1ForwardKinematics.from_mjcf",
                    return_value=Kinematics(),
                ),
                patch(
                    "mm_sonic.evaluate_terrain_pfnn._inspect_checkpoint",
                    return_value=checkpoint,
                ),
                patch(
                    "mm_sonic.evaluate_terrain_pfnn._finalize_inspected_checkpoint",
                    side_effect=lambda value: value,
                ),
                patch(
                    "mm_sonic.evaluate_terrain_pfnn.PFNNShardDataset",
                    return_value=dataset,
                ),
                patch(
                    "mm_sonic.evaluate_terrain_pfnn.one_step_metrics",
                    return_value={"one_step_score": 1.0, "samples": 2},
                ),
                patch(
                    "mm_sonic.evaluate_terrain_pfnn.evaluate_fitted_transition_envelope",
                    side_effect=transition_evaluation,
                ),
                patch(
                    "mm_sonic.evaluate_terrain_pfnn.run_known_train_rollout",
                    side_effect=AssertionError("known terrain callback opened"),
                ) as rollout,
            ):
                report = evaluate(
                    checkpoint_path=checkpoint_path,
                    dataset_path=root / "manifest.json",
                    model_path=MODEL_PATH,
                    split="train",
                    output_path=output_path,
                    sealed_test=False,
                    run_directory=None,
                    batch_size=2,
                    device="meta",
                    closed_loop_seconds=20.0,
                    promote_pipeline_checkpoint=True,
                )
                checkpoint.selection = None
                missing_selection_report = evaluate(
                    checkpoint_path=checkpoint_path,
                    dataset_path=root / "manifest.json",
                    model_path=MODEL_PATH,
                    split="train",
                    output_path=output_path,
                    sealed_test=False,
                    run_directory=None,
                    batch_size=2,
                    device="meta",
                    closed_loop_seconds=20.0,
                    promote_pipeline_checkpoint=False,
                )
            rollout.assert_not_called()
            self.assertEqual(transition_devices, ["cpu", "cpu"])
            self.assertEqual(
                report["closed_loop"]["status"],
                "rejected_fitted_transition_envelope",
            )
            self.assertEqual(report["fitted_transition_report"], rejected.to_dict())
            self.assertEqual(
                report["fitted_transition_report_sha256"], rejected.report_sha256
            )
            self.assertIsNone(report["pipeline_promotion"]["receipt"])
            self.assertFalse(report["pipeline_promotion"]["promoted"])
            self.assertEqual(
                missing_selection_report["closed_loop"]["status"],
                "rejected_fitted_transition_envelope",
            )
            self.assertIsNone(missing_selection_report["pipeline_promotion"])
            self.assertFalse((root / "best.pt").exists())

    def test_evaluator_cli_fails_closed_on_transition_rejection(self) -> None:
        rejected = {
            "fitted_transition_report": {"accepted": False},
            "closed_loop": {
                "status": "rejected_fitted_transition_envelope",
                "accepted": False,
            },
        }
        with (
            patch.object(evaluate_module, "evaluate", return_value=rejected),
            patch("builtins.print"),
        ):
            self.assertEqual(evaluate_module.main([
                "--checkpoint", "candidate.pt",
                "--dataset", "dataset",
                "--split", "train",
                "--closed-loop-seconds", "20",
            ]), 2)

    def test_active_pair_mismatch_rejects_before_evaluator_model_and_known_callback(
        self,
    ) -> None:
        manifest = {
            "dataset_digest_sha256": "a" * 64,
            "split_identities": {
                "train": ["slope_000"],
                "validation": ["slope_002"],
            },
        }

        class Dataset:
            split = "train"
            x_mean = np.zeros(INPUT_LAYOUT.size, np.float32)
            x_std = np.ones(INPUT_LAYOUT.size, np.float32)
            y_mean = np.zeros(OUTPUT_LAYOUT.size, np.float32)
            y_std = np.ones(OUTPUT_LAYOUT.size, np.float32)

            def __init__(self) -> None:
                classes = ("flat", "flat", "ascent", "descent", "transition")
                self.rows = [
                    self.row(center, classes[index])
                    for index, center in enumerate(range(10, 15))
                ]

            @staticmethod
            def row(center: int, terrain_class: str) -> dict[str, object]:
                y = np.zeros(OUTPUT_LAYOUT.size, np.float32)
                y[OUTPUT_LAYOUT["trajectory_direction"]] = np.tile(
                    (1.0, 0.0), 12
                )
                y[OUTPUT_LAYOUT["root_height"]] = 0.8
                y[OUTPUT_LAYOUT["phase_advance"]] = 0.1
                return {
                    "x": np.zeros(INPUT_LAYOUT.size, np.float32),
                    "y": y,
                    "phase": np.float32(0.1 * center),
                    "clip_id": "terrain_slopes__slope_000__000",
                    "center_frame": center,
                    "split_identity": "slope_000",
                    "split": "train",
                    "sequence_lane": "motion",
                    "terrain_class": terrain_class,
                }

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict[str, object]:
                return self.rows[index]

        class Kinematics:
            kinematic_signature_sha256 = "kin"
            joint_limits = torch.tensor([[-2.0, 2.0]] * 29, dtype=torch.float64)

            def to(self, _device: object) -> Kinematics:
                return self

        dataset = Dataset()
        fitted_subset = fitted_subset_metadata(dataset, range(len(dataset)))
        pair_receipt = train_module.fitted_transition_pair_receipt(dataset)
        pair_dataset = train_module.materialize_transition_pairs(dataset)
        target_audit = training_module.fitted_target_envelope_audit(
            pair_dataset,
            normalization=dataset,
            joint_limits=Kinematics.joint_limits,
            phase_advance_cap=1.5 * 0.2,
            contract=training_module.DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE,
        )
        tampered_receipt = dict(pair_receipt)
        tampered_receipt["pairs_sha256"] = "0" * 64
        base = {
            name: value
            for name, value in tampered_receipt.items()
            if name != "receipt_sha256"
        }
        tampered_receipt["receipt_sha256"] = hashlib.sha256(
            json.dumps(base, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(json.dumps(manifest))
            checkpoint_path = root / "checkpoint.pt"
            model = training_module.PhaseFunctionedNetwork(
                hidden_size=8, dropout_probability=0.0
            )
            optimizer = torch.optim.Adam(
                model.parameters(), lr=1.0e-3, weight_decay=0.0
            )
            training_module.save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                dataset,
                dataset_digest=manifest["dataset_digest_sha256"],
                kinematic_signature_sha256="kin",
                runtime_seed=finite_runtime_seed(),
                physical_envelope_objective=(
                    training_module.DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE
                ),
                fitted_pair_receipt=tampered_receipt,
                target_envelope_audit=target_audit,
                step=1,
                joint_limits=Kinematics.joint_limits,
                phase_advance_q99=0.2,
                train_identities=("slope_000",),
                validation_identities=("slope_002",),
            )
            with patch(
                "mm_sonic.evaluate_terrain_pfnn.TorchG1ForwardKinematics.from_mjcf",
                return_value=Kinematics(),
            ), patch(
                "mm_sonic.evaluate_terrain_pfnn.PFNNShardDataset",
                return_value=dataset,
            ), patch.object(
                training_module.PhaseFunctionedNetwork,
                "load_state_dict",
                side_effect=AssertionError("evaluator model restore was reached"),
            ) as model_restore, patch.object(
                torch.optim.Adam,
                "load_state_dict",
                side_effect=AssertionError("evaluator Adam restore was reached"),
            ) as adam_restore, patch(
                "mm_sonic.evaluate_terrain_pfnn.run_known_train_rollout",
                side_effect=AssertionError("known terrain callback opened"),
            ) as callback:
                with self.assertRaisesRegex(ValueError, "active physical envelope"):
                    evaluate(
                        checkpoint_path=checkpoint_path,
                        dataset_path=root / "manifest.json",
                        model_path=MODEL_PATH,
                        split="train",
                        output_path=root / "evaluation.json",
                        sealed_test=False,
                        run_directory=None,
                        batch_size=2,
                        device="cpu",
                        closed_loop_seconds=20.0,
                        promote_pipeline_checkpoint=True,
                    )
                model_restore.assert_not_called()
                adam_restore.assert_not_called()
                callback.assert_not_called()

    def test_public_evaluator_allows_only_explicit_twenty_second_train_gate(self) -> None:
        manifest = {
            "dataset_digest_sha256": "a" * 64,
            "split_identities": {"train": ["train-id"], "validation": ["val-id"]},
        }

        class Dataset:
            split = "train"
            x_mean = np.zeros(INPUT_LAYOUT.size, np.float32)
            x_std = np.ones(INPUT_LAYOUT.size, np.float32)
            y_mean = np.zeros(OUTPUT_LAYOUT.size, np.float32)
            y_std = np.ones(OUTPUT_LAYOUT.size, np.float32)

            def __init__(self) -> None:
                classes = ("flat", "flat", "ascent", "descent", "transition")
                self.rows = []
                for index, center in enumerate(range(10, 15)):
                    y = np.zeros(OUTPUT_LAYOUT.size, np.float32)
                    y[OUTPUT_LAYOUT["phase_advance"]] = 0.1
                    self.rows.append(
                        {
                            "x": np.zeros(INPUT_LAYOUT.size, np.float32),
                            "y": y,
                            "phase": np.float32(0.1),
                            "clip_id": "train-id",
                            "center_frame": center,
                            "split_identity": "train-id",
                            "split": "train",
                            "sequence_lane": "motion",
                            "terrain_class": classes[index],
                        }
                    )

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict[str, object]:
                return self.rows[index]

        class Kinematics:
            kinematic_signature_sha256 = "kin"
            joint_limits = torch.tensor([[-2.0, 2.0]] * 29, dtype=torch.float64)

            def to(self, _device):
                return self

        dataset = Dataset()
        active_pair_receipt = train_module.fitted_transition_pair_receipt(dataset)
        active_target_audit = training_module.fitted_target_envelope_audit(
            train_module.materialize_transition_pairs(dataset),
            normalization=dataset,
            joint_limits=Kinematics.joint_limits,
            phase_advance_cap=1.5 * 0.2,
            contract=training_module.DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE,
        )

        class Checkpoint:
            kinematic_signature_sha256 = "kin"
            joint_limits = Kinematics.joint_limits.clone()
            train_identities = ("train-id",)
            validation_identities = ("val-id",)
            normalization = {
                "x_mean": torch.zeros(INPUT_LAYOUT.size),
                "x_std": torch.ones(INPUT_LAYOUT.size),
                "y_mean": torch.zeros(OUTPUT_LAYOUT.size),
                "y_std": torch.ones(OUTPUT_LAYOUT.size),
            }
            loss_weights = {}
            phase_advance_q99 = 0.2

            def __init__(self) -> None:
                self.physical_envelope_objective = (
                    training_module.physical_envelope_objective_payload(
                        training_module.DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE
                    )
                )
                self.physical_envelope_objective_sha256 = hashlib.sha256(
                    json.dumps(
                        self.physical_envelope_objective,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                self.fitted_pair_receipt = active_pair_receipt
                self.target_envelope_audit = active_target_audit

            @staticmethod
            def build_model():
                return torch.nn.Identity()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(__import__("json").dumps(manifest))
            checkpoint_path = root / "best.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            output = root / "evaluation.json"
            closed_loop = {
                "schema": "mm-sonic-terrain-pfnn-closed-loop/v1",
                "known_train_gate": {"accepted": True},
            }
            with (
                patch(
                    "mm_sonic.evaluate_terrain_pfnn.TorchG1ForwardKinematics.from_mjcf",
                    return_value=Kinematics(),
                ),
                patch(
                    "mm_sonic.evaluate_terrain_pfnn._inspect_checkpoint",
                    return_value=Checkpoint(),
                ),
                patch(
                    "mm_sonic.evaluate_terrain_pfnn._finalize_inspected_checkpoint",
                    side_effect=lambda value: value,
                ),
                patch(
                    "mm_sonic.evaluate_terrain_pfnn.PFNNShardDataset",
                    return_value=dataset,
                ),
                patch(
                    "mm_sonic.evaluate_terrain_pfnn.one_step_metrics",
                    return_value={"one_step_score": 1.0, "samples": 1},
                ),
                patch(
                    "mm_sonic.evaluate_terrain_pfnn.run_known_train_rollout",
                    return_value=closed_loop,
                ) as rollout,
            ):
                report = evaluate(
                    checkpoint_path=checkpoint_path,
                    dataset_path=root / "manifest.json",
                    model_path=MODEL_PATH,
                    split="train",
                    output_path=output,
                    sealed_test=False,
                    run_directory=None,
                    batch_size=1,
                    device="cpu",
                    closed_loop_seconds=20.0,
                )
                rollout.assert_called_once()
                rollout.reset_mock()
                with self.assertRaisesRegex(ValueError, "provisional fitted"):
                    evaluate(
                        checkpoint_path=checkpoint_path,
                        dataset_path=root / "manifest.json",
                        model_path=MODEL_PATH,
                        split="train",
                        output_path=output,
                        sealed_test=False,
                        run_directory=None,
                        batch_size=1,
                        device="cpu",
                        closed_loop_seconds=20.0,
                        promote_pipeline_checkpoint=True,
                    )
                rollout.assert_not_called()
            self.assertEqual(report["closed_loop"], closed_loop)
            with self.assertRaisesRegex(ValueError, "20"):
                evaluate(
                    checkpoint_path=checkpoint_path,
                    dataset_path=root / "manifest.json",
                    model_path=MODEL_PATH,
                    split="train",
                    output_path=output,
                    sealed_test=False,
                    run_directory=None,
                    batch_size=1,
                    device="cpu",
                    closed_loop_seconds=None,
                )


if __name__ == "__main__":
    unittest.main()
