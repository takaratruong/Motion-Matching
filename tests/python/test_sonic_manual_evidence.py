from __future__ import annotations

import hashlib
import math
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mm_sonic.commands import CommandSample, flat_command_script
from mm_sonic.hands import (
    LEFT_HAND_JOINT_ORDER,
    NEUTRAL_HAND_TARGETS,
    RIGHT_HAND_JOINT_ORDER,
    hand_targets_record,
)
from mm_sonic.joints import ContractError, TARGET_JOINT_ORDER
from mm_sonic.scene import normalize_run_local_actuators
from mm_sonic.manual_evidence import (
    GEAR_FALL_MARKER,
    MANUAL_COMMAND_SCHEMA,
    MMCommandReplay,
    ManualEvidence,
    audit_manual_bundle,
    audit_manual_run,
    authenticate_pose_stream,
    command_phase_report,
    environment_control_record,
    load_canonical_target_npz,
    manual_command_artifact_bytes,
    manual_summary_v4_bytes,
    manual_summary_v5_bytes,
    parse_environment_control,
    parse_manual_command_artifact,
    parse_manual_summary_v4,
    parse_manual_summary_v5,
    scan_for_fall_marker,
    state_metrics,
)
from mm_sonic.scene import (
    HOLDEN_COORDINATE_SIGNATURE,
    HOLDEN_TO_MUJOCO_MATRIX,
    MUJOCO_COORDINATE_SIGNATURE,
)
from mm_sonic.replay_video import ReplayState
from mm_sonic.timeline import CanonicalTargetBuffer
from mm_sonic.zmq_v1 import encode_pose_v1


# The 14 hand joints occupy qpos slots 36..49 in the manual state fixtures.
_HAND_QPOS_ADDRESSES = tuple(range(36, 50))
_HAND_JOINT_RANGES = tuple((-2.0, 2.0) for _ in range(14))
_HAND_TARGET_VECTOR = np.asarray(
    NEUTRAL_HAND_TARGETS.left + NEUTRAL_HAND_TARGETS.right, dtype=np.float64
)


def _stand(index: int) -> CommandSample:
    return CommandSample(index, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def _forward(index: int, speed: float = 0.5) -> CommandSample:
    return CommandSample(index, (speed, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _state(index: int, x: float, y: float, z: float, yaw: float) -> ReplayState:
    qpos = np.zeros(50, dtype=np.float64)
    qpos[0] = x
    qpos[1] = y
    qpos[2] = z
    qpos[3:7] = _yaw_quat(yaw)
    # Settle the 14 Dex3 joints at their commanded target plus 0.05 rad so the
    # final-second median absolute tracking error stays under the 0.20 bound.
    qpos[36:50] = _HAND_TARGET_VECTOR + 0.05
    return ReplayState(4 * (index + 1), 0.02 * (index + 1), qpos, np.zeros(49))


class ManualCommandArtifactTests(unittest.TestCase):
    def test_serializes_committed_commands_in_chunk_order(self) -> None:
        commands = [_stand(0), _stand(1), _forward(2), _forward(3)]

        data = manual_command_artifact_bytes(
            mode="script",
            preload_chunks=2,
            commands=commands,
            hand_targets=NEUTRAL_HAND_TARGETS,
        )
        parsed = parse_manual_command_artifact(data)

        self.assertEqual(parsed.schema, MANUAL_COMMAND_SCHEMA)
        self.assertEqual(parsed.mode, "script")
        self.assertEqual(parsed.preload_chunks, 2)
        self.assertEqual(
            [command.chunk_index for command in parsed.commands], [0, 1, 2, 3]
        )
        self.assertEqual(parsed.commands[2].requested_velocity_mujoco, (0.5, 0.0, 0.0))
        document = json.loads(data)
        self.assertEqual(
            [entry["origin"] for entry in document["commands"]],
            ["preload", "preload", "operator", "operator"],
        )

    def test_serialization_is_deterministic(self) -> None:
        commands = [_stand(0), _forward(1)]

        first = manual_command_artifact_bytes(
            mode="interactive",
            preload_chunks=1,
            commands=commands,
            hand_targets=NEUTRAL_HAND_TARGETS,
        )
        second = manual_command_artifact_bytes(
            mode="interactive",
            preload_chunks=1,
            commands=commands,
            hand_targets=NEUTRAL_HAND_TARGETS,
        )

        self.assertEqual(first, second)

    def test_rejects_reordered_chunk_indices(self) -> None:
        commands = [_stand(0), _stand(2), _forward(1)]

        with self.assertRaisesRegex(ContractError, "chunk indices"):
            manual_command_artifact_bytes(
                mode="script",
                preload_chunks=1,
                commands=commands,
                hand_targets=NEUTRAL_HAND_TARGETS,
            )


def _buffer(first_index: int, count: int) -> CanonicalTargetBuffer:
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1))
    return CanonicalTargetBuffer(
        joint_position=np.zeros((count, 29), dtype=np.float32),
        joint_velocity=np.zeros((count, 29), dtype=np.float32),
        body_quat_w=quat,
        frame_index=np.arange(first_index, first_index + count, dtype=np.int64),
    )


class FallMarkerTests(unittest.TestCase):
    def test_detects_real_simulator_fall_marker_line(self) -> None:
        log = b"Warning: Robot has fallen, height: 0.195 m\n"

        self.assertTrue(scan_for_fall_marker(log))
        self.assertEqual(GEAR_FALL_MARKER, "Warning: Robot has fallen")

    def test_absent_fall_marker_returns_false(self) -> None:
        self.assertFalse(scan_for_fall_marker(b"healthy run\nno issues\n"))


class CanonicalNpzTests(unittest.TestCase):
    def test_loads_matching_canonical_buffer(self) -> None:
        import io

        buffer = _buffer(0, 601)
        stream = io.BytesIO()
        np.savez(
            stream,
            joint_position=np.ascontiguousarray(buffer.joint_position, dtype="<f4"),
            joint_velocity=np.ascontiguousarray(buffer.joint_velocity, dtype="<f4"),
            body_quat_w=np.ascontiguousarray(buffer.body_quat_w, dtype="<f4"),
            frame_index=np.ascontiguousarray(buffer.frame_index, dtype="<i8"),
        )

        loaded = load_canonical_target_npz(stream.getvalue())

        self.assertEqual(loaded.count, 601)

    def test_rejects_npz_with_unexpected_arrays(self) -> None:
        import io

        stream = io.BytesIO()
        np.savez(stream, unexpected=np.zeros(3))
        with self.assertRaisesRegex(ContractError, "canonical"):
            load_canonical_target_npz(stream.getvalue())


class PoseStreamTests(unittest.TestCase):
    def test_authenticates_contiguous_frames(self) -> None:
        frames = [encode_pose_v1(_buffer(0, 1)), encode_pose_v1(_buffer(1, 20))]

        buffer, hand_exact = authenticate_pose_stream(frames, expected_first_frame=0)

        self.assertEqual(buffer.count, 21)
        np.testing.assert_array_equal(buffer.frame_index, np.arange(21))
        self.assertFalse(hand_exact)

    def test_reports_exact_hand_transport_for_neutral_frames(self) -> None:
        frames = [
            encode_pose_v1(_buffer(0, 1), hand_targets=NEUTRAL_HAND_TARGETS),
            encode_pose_v1(_buffer(1, 20), hand_targets=NEUTRAL_HAND_TARGETS),
        ]

        buffer, hand_exact = authenticate_pose_stream(
            frames, expected_first_frame=0, hand_targets=NEUTRAL_HAND_TARGETS
        )

        self.assertEqual(buffer.count, 21)
        self.assertTrue(hand_exact)

    def test_rejects_noncontiguous_frames(self) -> None:
        frames = [encode_pose_v1(_buffer(0, 1)), encode_pose_v1(_buffer(5, 20))]

        with self.assertRaisesRegex(ContractError, "contiguous"):
            authenticate_pose_stream(frames, expected_first_frame=0)

    def test_rejects_duplicate_frames(self) -> None:
        frames = [encode_pose_v1(_buffer(0, 1)), encode_pose_v1(_buffer(1, 20)),
                  encode_pose_v1(_buffer(1, 20))]

        with self.assertRaisesRegex(ContractError, "contiguous"):
            authenticate_pose_stream(frames, expected_first_frame=0)


class StateMetricsTests(unittest.TestCase):
    def test_reports_boundary_metrics_for_forward_then_stop(self) -> None:
        states = []
        # 10 frames moving forward 0.1 m each, then 5 frames standing still.
        for index in range(10):
            states.append(_state(index, 0.1 * index, 0.0, 0.75 - 0.01 * index, 0.0))
        for offset in range(5):
            states.append(_state(10 + offset, 0.9, 0.0, 0.66, 0.5))

        metrics = state_metrics(tuple(states), final_stop_seconds=0.06)

        self.assertAlmostEqual(metrics.minimum_root_height_m, 0.66, places=6)
        self.assertAlmostEqual(metrics.minimum_pelvis_up_dot, 1.0, places=6)
        self.assertAlmostEqual(metrics.path_distance_m, 0.9, places=6)
        self.assertAlmostEqual(metrics.yaw_change_rad, 0.5, places=6)
        self.assertAlmostEqual(
            metrics.final_stop_displacement_m, 0.0, places=6
        )

    def test_rejects_empty_state_stream(self) -> None:
        with self.assertRaisesRegex(ContractError, "state"):
            state_metrics((), final_stop_seconds=2.0)

    def test_yaw_change_reports_net_turn_not_cumulative_gait_jitter(self) -> None:
        yaws = (0.0, 0.5, 0.0, 0.5)
        states = tuple(
            _state(index, 0.0, 0.0, 0.75, yaw)
            for index, yaw in enumerate(yaws)
        )

        metrics = state_metrics(states, final_stop_seconds=0.02)

        self.assertAlmostEqual(metrics.yaw_change_rad, 0.5, places=6)


class CommandPhaseTests(unittest.TestCase):
    def test_flat_script_contains_ordered_phases(self) -> None:
        report = command_phase_report(flat_command_script())

        self.assertTrue(report.has_forward)
        self.assertTrue(report.has_heading_change)
        self.assertTrue(report.has_final_stand)
        self.assertTrue(report.phases_ordered)

    def test_missing_forward_phase_is_reported(self) -> None:
        commands = tuple(_stand(index) for index in range(6))

        report = command_phase_report(commands)

        self.assertFalse(report.has_forward)
        self.assertFalse(report.phases_ordered)

    def test_missing_final_stand_is_reported(self) -> None:
        commands = (_stand(0), _forward(1), _forward(2))

        report = command_phase_report(commands)

        self.assertTrue(report.has_forward)
        self.assertFalse(report.has_final_stand)
        self.assertFalse(report.phases_ordered)

    def test_backward_motion_does_not_count_as_forward(self) -> None:
        commands = tuple(
            CommandSample(index, (-0.5, 0.0, 0.0), _yaw_quat(0.1 * index))
            if index < 5
            else _stand(index)
            for index in range(10)
        )

        report = command_phase_report(commands)

        self.assertFalse(report.has_forward)
        self.assertFalse(report.phases_ordered)

    def test_one_zero_command_is_not_a_two_second_final_stand(self) -> None:
        commands = tuple(
            CommandSample(index, (0.5, 0.0, 0.0), _yaw_quat(0.1 * index))
            if index < 9
            else _stand(index)
            for index in range(10)
        )

        report = command_phase_report(commands)

        self.assertFalse(report.has_final_stand)
        self.assertFalse(report.phases_ordered)


def _environment_control_fields() -> dict:
    return {
        "scene_id": "grail-curb-default",
        "route_id": "curb-forward",
        "terrain_weight": 4.0,
        "input_source": "x11",
        "mapper_version": "holden-control/v1",
        "camera_sequence": 7,
        "mm_hello_identity": {
            "coordinate_signature": HOLDEN_COORDINATE_SIGNATURE,
            "motion_manifest_sha256": "a" * 64,
            "scene_index_sha256": "b" * 64,
        },
        "mm_scene_identity": {
            "scene_id": "grail-curb-default",
            "route_id": "curb-forward",
            "coordinate_signature": HOLDEN_COORDINATE_SIGNATURE,
            "heightfield_sha256": "c" * 64,
            "mesh_sha256": "d" * 64,
            "walkability_sha256": "e" * 64,
        },
        "source_hashes": {
            "manifest": "a" * 64,
            "scene_index": "b" * 64,
            "terrain_bin": "c" * 64,
            "terrain_obj": "d" * 64,
            "walkability": "e" * 64,
        },
        "output_hashes": {
            "gear_scene_xml": "f" * 64,
            "transformed_obj": "0" * 64,
        },
        "coordinate_source": HOLDEN_COORDINATE_SIGNATURE,
        "coordinate_target": MUJOCO_COORDINATE_SIGNATURE,
        "transform_matrix": HOLDEN_TO_MUJOCO_MATRIX.tolist(),
        "source_bounds_holden": [[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]],
        "transformed_bounds_mujoco": [[-3.0, -1.0, -2.0], [3.0, 1.0, 2.0]],
        "initial_boundary_sha256": "1" * 64,
        "initial_qpos_sha256": "2" * 64,
    }


class EnvironmentControlCodecTests(unittest.TestCase):
    def test_round_trips_all_bound_fields(self) -> None:
        fields = _environment_control_fields()

        record = environment_control_record(**fields)
        parsed = parse_environment_control(record)

        self.assertEqual(parsed["scene_id"], "grail-curb-default")
        self.assertEqual(parsed["route_id"], "curb-forward")
        self.assertEqual(parsed["terrain_weight"], 4.0)
        self.assertEqual(parsed["input_source"], "x11")
        self.assertEqual(parsed["mapper_version"], "holden-control/v1")
        self.assertEqual(parsed["camera_sequence"], 7)
        self.assertEqual(parsed["initial_qpos_sha256"], "2" * 64)
        self.assertEqual(
            parsed["transform_matrix"], HOLDEN_TO_MUJOCO_MATRIX.tolist()
        )

    def test_rejects_unknown_field(self) -> None:
        record = environment_control_record(**_environment_control_fields())
        record["unexpected"] = 1
        with self.assertRaisesRegex(ContractError, "environment control"):
            parse_environment_control(record)

    def test_rejects_missing_field(self) -> None:
        record = environment_control_record(**_environment_control_fields())
        del record["camera_sequence"]
        with self.assertRaisesRegex(ContractError, "environment control"):
            parse_environment_control(record)

    def test_rejects_mutated_digest(self) -> None:
        record = environment_control_record(**_environment_control_fields())
        record["initial_qpos_sha256"] = "z" * 64
        with self.assertRaisesRegex(ContractError, "initial_qpos_sha256"):
            parse_environment_control(record)

    def test_rejects_nonfinite_transform_entry(self) -> None:
        fields = _environment_control_fields()
        matrix = [list(row) for row in fields["transform_matrix"]]
        matrix[0][0] = float("inf")
        fields["transform_matrix"] = matrix
        with self.assertRaisesRegex(ContractError, "finite"):
            environment_control_record(**fields)

    def test_rejects_unknown_input_source(self) -> None:
        fields = _environment_control_fields()
        fields["input_source"] = "gamepad"
        with self.assertRaisesRegex(ContractError, "input_source"):
            environment_control_record(**fields)

    def test_rejects_negative_camera_sequence(self) -> None:
        fields = _environment_control_fields()
        fields["camera_sequence"] = -1
        with self.assertRaisesRegex(ContractError, "camera_sequence"):
            environment_control_record(**fields)

    def test_rejects_malformed_bounds_shape(self) -> None:
        fields = _environment_control_fields()
        fields["source_bounds_holden"] = [[0.0, 0.0, 0.0]]
        with self.assertRaisesRegex(ContractError, "source_bounds_holden"):
            environment_control_record(**fields)


def _v4_summary_fields(run_root: str) -> dict:
    commands = _manual_commands()
    command_bytes = manual_command_artifact_bytes(
        mode="interactive",
        preload_chunks=2,
        commands=commands,
        hand_targets=NEUTRAL_HAND_TARGETS,
    )
    return {
        "mode": "interactive",
        "run_root": run_root,
        "preload_chunks": 2,
        "generated_chunks": len(commands),
        "lookahead_seconds": 0.8,
        "command_bytes": command_bytes,
        "hand_targets": NEUTRAL_HAND_TARGETS,
        "environment_control": environment_control_record(
            **_environment_control_fields()
        ),
        "snapshot": {
            "contact_rows": 12,
            "sim_time_s": 3.5,
            "state_rows": 10,
            "steps": 12,
        },
    }


class ManualSummaryV4Tests(unittest.TestCase):
    def test_round_trips_terrain_summary(self) -> None:
        fields = _v4_summary_fields("/runs/manual-x")

        data = manual_summary_v4_bytes(**fields)
        parsed = parse_manual_summary_v4(data)

        self.assertEqual(parsed["schema"], "mm-sonic-manual-demo/v4")
        self.assertEqual(parsed["mode"], "interactive")
        self.assertEqual(parsed["preload_chunks"], 2)
        self.assertEqual(
            parsed["environment_control"]["scene_id"], "grail-curb-default"
        )

    def test_rejects_v3_summary_as_v4(self) -> None:
        fields = _v4_summary_fields("/runs/manual-x")
        data = manual_summary_v4_bytes(**fields)
        document = json.loads(data)
        document["schema"] = "mm-sonic-manual-demo/v3"
        with self.assertRaisesRegex(ContractError, "schema"):
            parse_manual_summary_v4(
                (json.dumps(document, sort_keys=True) + "\n").encode("ascii")
            )

    def test_rejects_mutated_environment_digest(self) -> None:
        fields = _v4_summary_fields("/runs/manual-x")
        data = manual_summary_v4_bytes(**fields)
        document = json.loads(data)
        document["environment_control"]["initial_qpos_sha256"] = "z" * 64
        with self.assertRaisesRegex(ContractError, "initial_qpos_sha256"):
            parse_manual_summary_v4(
                (json.dumps(document, sort_keys=True) + "\n").encode("ascii")
            )

    def test_rejects_unknown_top_level_field(self) -> None:
        fields = _v4_summary_fields("/runs/manual-x")
        data = manual_summary_v4_bytes(**fields)
        document = json.loads(data)
        document["surprise"] = 1
        with self.assertRaisesRegex(ContractError, "manual summary"):
            parse_manual_summary_v4(
                (json.dumps(document, sort_keys=True) + "\n").encode("ascii")
            )


def _movement_model_record(profile: str = "holden-v1") -> dict:
    return {
        "profile": profile,
        "acceleration_mps2": 1.5,
        "deceleration_mps2": 2.0,
        "directional_acceleration": False,
        "turn_strength": False,
    }


class ManualSummaryV5Tests(unittest.TestCase):
    def _v5_fields(self, run_root: str, profile: str = "holden-v1") -> dict:
        fields = _v4_summary_fields(run_root)
        fields["movement_model"] = _movement_model_record(profile)
        return fields

    def test_round_trips_selected_movement_model(self) -> None:
        fields = self._v5_fields("/runs/manual-x")
        data = manual_summary_v5_bytes(**fields)
        parsed = parse_manual_summary_v5(data)
        self.assertEqual(parsed["schema"], "mm-sonic-manual-demo/v5")
        self.assertEqual(
            parsed["movement_model"], _movement_model_record("holden-v1")
        )

    def test_records_raw_default_profile(self) -> None:
        fields = self._v5_fields("/runs/manual-x", profile="raw")
        parsed = parse_manual_summary_v5(manual_summary_v5_bytes(**fields))
        self.assertEqual(parsed["movement_model"]["profile"], "raw")

    def test_v4_parser_still_reads_old_summaries(self) -> None:
        fields = _v4_summary_fields("/runs/manual-x")
        parsed = parse_manual_summary_v4(manual_summary_v4_bytes(**fields))
        self.assertEqual(parsed["schema"], "mm-sonic-manual-demo/v4")

    def test_rejects_invalid_movement_model(self) -> None:
        fields = self._v5_fields("/runs/manual-x")
        fields["movement_model"] = {"profile": "other"}
        with self.assertRaisesRegex(ContractError, "movement_model"):
            manual_summary_v5_bytes(**fields)


def _npz_bytes(buffer: CanonicalTargetBuffer) -> bytes:
    import io

    stream = io.BytesIO()
    np.savez(
        stream,
        joint_position=np.ascontiguousarray(buffer.joint_position, dtype="<f4"),
        joint_velocity=np.ascontiguousarray(buffer.joint_velocity, dtype="<f4"),
        body_quat_w=np.ascontiguousarray(buffer.body_quat_w, dtype="<f4"),
        frame_index=np.ascontiguousarray(buffer.frame_index, dtype="<i8"),
    )
    return stream.getvalue()


def _manual_commands() -> tuple[CommandSample, ...]:
    commands: list[CommandSample] = []
    for index in range(34):
        if index < 4 or index >= 20:
            speed = 0.0
        else:
            speed = 0.5
        yaw = 0.0 if index < 12 else min(0.7, 0.1 * (index - 11))
        commands.append(
            CommandSample(index, (speed, 0.0, 0.0), _yaw_quat(yaw))
        )
    return tuple(commands)


def _valid_evidence(**overrides: object) -> dict:
    pose_frames = [encode_pose_v1(_buffer(0, 1), hand_targets=NEUTRAL_HAND_TARGETS)]
    for chunk in range(34):
        pose_frames.append(
            encode_pose_v1(_buffer(1 + chunk * 20, 20), hand_targets=NEUTRAL_HAND_TARGETS)
        )
    states = []
    for index in range(600):
        if index < 300:
            x = 0.004 * index
            yaw = 0.0
        else:
            x = 1.2
            yaw = 0.5
        states.append(_state(index, x, 0.0, 0.7, yaw))
    evidence = {
        "states": tuple(states),
        "pose_frames": pose_frames,
        "mm_replay": MMCommandReplay(
            baseline_canonical_parity=True,
            command_buffer=_buffer(0, 681),
        ),
        "command_artifact": manual_command_artifact_bytes(
            mode="interactive",
            preload_chunks=4,
            commands=_manual_commands(),
            hand_targets=NEUTRAL_HAND_TARGETS,
        ),
        "gear_log": b"startup ok\nCONTROL active\nclean shutdown\n",
        "hand_targets": NEUTRAL_HAND_TARGETS,
        "hand_qpos_addresses": _HAND_QPOS_ADDRESSES,
        "hand_joint_ranges": _HAND_JOINT_RANGES,
        "actuator_routing_pass": True,
    }
    evidence.update(overrides)
    return evidence


class AuditManualRunTests(unittest.TestCase):
    def test_valid_evidence_passes_all_boundaries(self) -> None:
        result = audit_manual_run(**_valid_evidence())

        self.assertIsInstance(result, ManualEvidence)
        self.assertTrue(result.passed)
        self.assertTrue(result.state_rows_valid)
        self.assertTrue(result.pose_stream_valid)
        self.assertTrue(result.baseline_canonical_parity)
        self.assertTrue(result.command_replay_parity)
        self.assertTrue(result.no_fall_marker)
        self.assertTrue(result.command_phases.phases_ordered)
        self.assertAlmostEqual(result.metrics.minimum_root_height_m, 0.7, places=6)

    def test_wrong_state_row_count_fails_closed(self) -> None:
        evidence = _valid_evidence()
        evidence["states"] = evidence["states"][:599]
        with self.assertRaisesRegex(ContractError, "600"):
            audit_manual_run(**evidence)

    def test_command_replay_mismatch_fails_closed(self) -> None:
        corrupted = _buffer(0, 681)
        corrupted.joint_position.flags.writeable = True
        corrupted.joint_position[5, 0] = np.float32(1.0)
        result = audit_manual_run(
            **_valid_evidence(
                mm_replay=MMCommandReplay(
                    baseline_canonical_parity=True,
                    command_buffer=corrupted,
                )
            )
        )

        self.assertFalse(result.command_replay_parity)
        self.assertFalse(result.passed)

    def test_unqualified_mm_server_fails_closed(self) -> None:
        result = audit_manual_run(
            **_valid_evidence(
                mm_replay=MMCommandReplay(
                    baseline_canonical_parity=False,
                    command_buffer=_buffer(0, 681),
                )
            )
        )

        self.assertFalse(result.baseline_canonical_parity)
        self.assertFalse(result.passed)

    def test_fall_marker_fails_closed(self) -> None:
        log = f"startup\n{GEAR_FALL_MARKER} humanoid fell\n".encode()
        result = audit_manual_run(**_valid_evidence(gear_log=log))

        self.assertFalse(result.no_fall_marker)
        self.assertFalse(result.passed)

    def test_missing_turn_phase_fails_closed(self) -> None:
        commands = tuple(
            _stand(index) if index < 4 or index >= 20 else _forward(index)
            for index in range(34)
        )
        artifact = manual_command_artifact_bytes(
            mode="interactive",
            preload_chunks=4,
            commands=commands,
            hand_targets=NEUTRAL_HAND_TARGETS,
        )
        result = audit_manual_run(**_valid_evidence(command_artifact=artifact))

        self.assertFalse(result.command_phases.has_heading_change)
        self.assertFalse(result.passed)

    def test_noncontiguous_pose_stream_fails_closed(self) -> None:
        evidence = _valid_evidence()
        frames = list(evidence["pose_frames"])
        frames[10] = encode_pose_v1(_buffer(999, 20), hand_targets=NEUTRAL_HAND_TARGETS)
        evidence["pose_frames"] = frames
        with self.assertRaisesRegex(ContractError, "contiguous"):
            audit_manual_run(**evidence)

    def test_missing_hand_field_fails_closed(self) -> None:
        evidence = _valid_evidence()
        frames = list(evidence["pose_frames"])
        frames[10] = encode_pose_v1(_buffer(1 + 9 * 20, 20))  # legacy, no hands
        evidence["pose_frames"] = frames
        with self.assertRaisesRegex(ContractError, "hand fields"):
            audit_manual_run(**evidence)

    def test_changed_hand_bit_fails_closed(self) -> None:
        evidence = _valid_evidence()
        from mm_sonic.hands import Dex3HandTargets

        wrong_left = list(NEUTRAL_HAND_TARGETS.left)
        wrong_left[1] = 0.2
        wrong = Dex3HandTargets(
            profile=NEUTRAL_HAND_TARGETS.profile,
            left=tuple(wrong_left),
            right=NEUTRAL_HAND_TARGETS.right,
        )
        frames = list(evidence["pose_frames"])
        frames[10] = encode_pose_v1(_buffer(1 + 9 * 20, 20), hand_targets=wrong)
        evidence["pose_frames"] = frames
        result = audit_manual_run(**evidence)
        self.assertFalse(result.hand_transport_exact)
        self.assertFalse(result.passed)

    def test_hand_tracking_error_above_bound_fails_closed(self) -> None:
        evidence = _valid_evidence()
        states = list(evidence["states"])
        drifted = []
        for state in states:
            qpos = state.qpos.copy()
            qpos[36:50] = _HAND_TARGET_VECTOR + 0.5
            drifted.append(ReplayState(state.step, state.sim_time_s, qpos, state.qvel))
        evidence["states"] = tuple(drifted)
        result = audit_manual_run(**evidence)
        self.assertFalse(result.hand_tracking.passed)
        self.assertFalse(result.passed)
        self.assertGreater(
            max(result.hand_tracking.median_absolute_error_rad), 0.20
        )

    def test_unexpected_limit_pinning_fails_closed(self) -> None:
        evidence = _valid_evidence()
        states = list(evidence["states"])
        pinned = []
        for state in states:
            qpos = state.qpos.copy()
            # Joint index 0 target is 0.0; pin it to the +2.0 range limit.
            qpos[36] = 2.0
            pinned.append(ReplayState(state.step, state.sim_time_s, qpos, state.qvel))
        evidence["states"] = tuple(pinned)
        result = audit_manual_run(**evidence)
        self.assertFalse(result.hand_tracking.limit_pass)
        self.assertFalse(result.passed)

    def test_actuator_routing_failure_fails_closed(self) -> None:
        result = audit_manual_run(**_valid_evidence(actuator_routing_pass=False))
        self.assertFalse(result.actuator_routing_pass)
        self.assertFalse(result.passed)


def _store_pose(path: Path, buffer: CanonicalTargetBuffer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    message = encode_pose_v1(buffer, hand_targets=NEUTRAL_HAND_TARGETS)
    path.write_bytes(message)
    path.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(message).hexdigest()}  {path.name}\n",
        encoding="ascii",
    )


def _manual_scene_files(run: Path) -> dict[str, str]:
    """Write a loadable run-local scene whose hand joints occupy qpos 36..49.

    A free root plus 29 body hinges then 14 hand hinges yields nq == 50 with the
    hand joints last, matching the synthetic state fixture. Motors are emitted in
    reverse order so normalization must reorder them to joint traversal.
    """

    scene_dir = run / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    joint_order = list(TARGET_JOINT_ORDER) + list(LEFT_HAND_JOINT_ORDER) + list(
        RIGHT_HAND_JOINT_ORDER
    )
    bodies = []
    for index, joint in enumerate(joint_order):
        bodies.append(
            f'<body name="{joint}_link" pos="{0.05 * index:.4g} 0 1.0">'
            f'<joint name="{joint}" type="hinge" axis="0 1 0" range="-2 2"/>'
            f'<geom name="{joint}_geom" type="sphere" size="0.01" density="100"/>'
            "</body>"
        )
    motors = "".join(
        f'<motor name="{joint}_motor" joint="{joint}" gear="1"/>'
        for joint in reversed(joint_order)
    )
    robot_xml = (
        '<mujoco model="manual_synthetic">'
        '<compiler angle="radian"/>'
        '<worldbody><body name="pelvis" pos="0 0 0.2">'
        '<freejoint name="floating_base_joint"/>'
        '<geom name="pelvis_geom" type="sphere" size="0.1" density="100"/>'
        + "".join(bodies)
        + "</body></worldbody>"
        + f"<actuator>{motors}</actuator>"
        + "</mujoco>\n"
    ).encode("utf-8")
    robot_bytes, _, actuator_sha = normalize_run_local_actuators(
        robot_xml, label="run-local robot XML"
    )
    robot_path = scene_dir / "gear_robot.xml"
    robot_path.write_bytes(robot_bytes)
    scene_bytes = (
        '<mujoco model="manual_scene">'
        f'<include file="{robot_path}"/>'
        '<worldbody><geom name="floor" type="plane" size="0 0 .05"/>'
        "</worldbody></mujoco>\n"
    ).encode("utf-8")
    scene_path = scene_dir / "gear_scene.xml"
    scene_path.write_bytes(scene_bytes)
    return {
        "gear_scene_sha256": hashlib.sha256(scene_bytes).hexdigest(),
        "gear_robot_sha256": hashlib.sha256(robot_bytes).hexdigest(),
        "actuator_joint_order_sha256": actuator_sha,
    }


def _write_valid_bundle(root: Path) -> tuple[Path, Path]:
    run = root / "manual-run"
    run.mkdir()
    commands = _manual_commands()
    command_bytes = manual_command_artifact_bytes(
        mode="interactive",
        preload_chunks=4,
        commands=commands,
        hand_targets=NEUTRAL_HAND_TARGETS,
    )
    (run / "manual-commands.json").write_bytes(command_bytes)

    state_dir = run / "scored-sim-logs"
    state_dir.mkdir()
    rows: list[str] = []
    for index in range(600):
        x = min(4.0, 0.01 * index)
        yaw = min(0.75, max(0.0, 0.003 * (index - 150)))
        state = _state(index, x, 0.0, 0.75, yaw)
        rows.append(
            json.dumps(
                {
                    "step": state.step,
                    "sim_time_s": state.sim_time_s,
                    "state": {
                        "qpos": state.qpos.tolist(),
                        "qvel": state.qvel.tolist(),
                    },
                },
                separators=(",", ":"),
            )
        )
    (state_dir / "state.jsonl").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (run / "simulator.stderr").write_bytes(b"")

    _store_pose(
        run
        / "transmitted/readiness/attempt-000001__000000-000000.bin",
        _buffer(0, 1),
    )
    for chunk in range(34):
        first = 1 + chunk * 20
        directory = (
            run / "transmitted/logical"
            if chunk < 4
            else run / "transmitted"
        )
        _store_pose(
            directory / f"{first:06d}-{first + 19:06d}.bin",
            _buffer(first, 20),
        )

    canonical = root / "canonical_target.npz"
    canonical.write_bytes(_npz_bytes(_buffer(0, 601)))
    scene_control = _manual_scene_files(run)
    summary = {
        "schema": "mm-sonic-manual-demo/v3",
        "mode": "interactive",
        "run_root": str(run.resolve()),
        "preload_chunks": 4,
        "generated_chunks": 34,
        "lookahead_seconds": 1.6,
        "command_artifact": {
            "path": "manual-commands.json",
            "sha256": hashlib.sha256(command_bytes).hexdigest(),
        },
        "hand_control": hand_targets_record(NEUTRAL_HAND_TARGETS),
        "scene_control": scene_control,
        "snapshot": {
            "contact_rows": 2401,
            "sim_time_s": 12.005,
            "state_rows": 600,
            "steps": 2401,
        },
    }
    (run / "manual-summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="ascii",
    )
    return run, canonical


class ManualBundleAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run, self.canonical = _write_valid_bundle(self.root)
        self.replay = MMCommandReplay(
            baseline_canonical_parity=True,
            command_buffer=_buffer(0, 681),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_authenticates_the_real_manual_bundle_layout(self) -> None:
        result = audit_manual_bundle(self.run, self.canonical, replay=self.replay)

        self.assertTrue(result.passed)
        self.assertEqual(result.pose_frame_count, 681)
        self.assertEqual(len(result.command_artifact.commands), 34)
        self.assertTrue(result.root_height_pass)
        self.assertTrue(result.upright_pass)
        self.assertTrue(result.path_pass)
        self.assertTrue(result.yaw_pass)
        self.assertTrue(result.final_stop_pass)

    def test_authenticates_the_planned_script_mode_bundle(self) -> None:
        command_bytes = manual_command_artifact_bytes(
            mode="script",
            preload_chunks=4,
            commands=_manual_commands(),
            hand_targets=NEUTRAL_HAND_TARGETS,
        )
        (self.run / "manual-commands.json").write_bytes(command_bytes)
        summary_path = self.run / "manual-summary.json"
        summary = json.loads(summary_path.read_text("ascii"))
        summary["mode"] = "script"
        summary["command_artifact"]["sha256"] = hashlib.sha256(
            command_bytes
        ).hexdigest()
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="ascii",
        )

        result = audit_manual_bundle(self.run, self.canonical, replay=self.replay)

        self.assertTrue(result.passed)
        self.assertEqual(result.command_artifact.mode, "script")

    def test_rejects_a_malformed_state_cadence(self) -> None:
        path = self.run / "scored-sim-logs/state.jsonl"
        rows = path.read_text("utf-8").splitlines()
        first = json.loads(rows[0])
        first["sim_time_s"] = 0.03
        rows[0] = json.dumps(first, separators=(",", ":"))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ContractError, "time sequence"):
            audit_manual_bundle(self.run, self.canonical, replay=self.replay)

    def test_rejects_a_missing_live_archive(self) -> None:
        (self.run / "transmitted/000081-000100.bin").unlink()
        (self.run / "transmitted/000081-000100.sha256").unlink()

        with self.assertRaisesRegex(ContractError, "archive layout"):
            audit_manual_bundle(self.run, self.canonical, replay=self.replay)

    def test_rejects_a_forged_archive_digest(self) -> None:
        path = self.run / "transmitted/000081-000100.sha256"
        path.write_text(f"{'0' * 64}  000081-000100.bin\n", encoding="ascii")

        with self.assertRaisesRegex(ContractError, "digest"):
            audit_manual_bundle(self.run, self.canonical, replay=self.replay)

    def test_real_fall_marker_fails_the_bundle(self) -> None:
        (self.run / "simulator.stderr").write_text(
            "Warning: Robot has fallen, height: 0.195 m\n",
            encoding="ascii",
        )

        result = audit_manual_bundle(self.run, self.canonical, replay=self.replay)

        self.assertFalse(result.no_fall_marker)
        self.assertFalse(result.passed)

    def test_rejects_a_summary_command_hash_forgery(self) -> None:
        path = self.run / "manual-summary.json"
        summary = json.loads(path.read_text("ascii"))
        summary["command_artifact"]["sha256"] = "0" * 64
        path.write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="ascii",
        )

        with self.assertRaisesRegex(ContractError, "command artifact"):
            audit_manual_bundle(self.run, self.canonical, replay=self.replay)

    def test_rejects_v2_summary_with_clear_version_error(self) -> None:
        path = self.run / "manual-summary.json"
        summary = json.loads(path.read_text("ascii"))
        summary["schema"] = "mm-sonic-manual-demo/v2"
        summary.pop("hand_control")
        summary.pop("scene_control")
        path.write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="ascii",
        )

        with self.assertRaisesRegex(ContractError, "unsupported.*schema"):
            audit_manual_bundle(self.run, self.canonical, replay=self.replay)


if __name__ == "__main__":
    unittest.main()
