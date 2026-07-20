"""Live MM-only four-reset direction gate: pure scoring/report boundary.

These synthetic tests validate the gate's pure scoring core and its JSON-safe
report oracle only. They are NOT evidence that live Motion Matching follows a
command; the controller runs the real four-reset live probe separately.
"""

from __future__ import annotations

from dataclasses import replace
import os
from types import MappingProxyType, SimpleNamespace
import unittest

import numpy as np

from mm_sonic.direction_probe import (
    REGISTERED_DIRECTION_AXES_MUJOCO,
    REGISTERED_MINIMUM_PROJECTION_M,
)
from mm_sonic.joints import ContractError, load_joint_contract
from mm_sonic.timeline import TargetTimeline
from mm_sonic.transform import mujoco_to_holden_vectors
from mm_sonic.commands import CommandSample
from mm_sonic.coordinator import SessionConfig
from mm_sonic.mm_direction_gate import (
    DirectionScore,
    build_gate_parser,
    direction_command,
    direction_gate_summary,
    hash_initial_boundary,
    main,
    run_live_direction,
    run_live_direction_gate,
    score_target_direction,
    write_gate_artifact,
)
from tests.python.test_sonic_resample import (
    JOINT_CONTRACT,
    initial_from_chunk,
    make_source_chunk,
    readonly,
)


_TARGET_ROWS = 20


def base_target():
    contract = load_joint_contract(JOINT_CONTRACT)
    chunk = make_source_chunk(contract)
    timeline = TargetTimeline(initial_from_chunk(chunk), contract)
    return timeline.prepare(chunk).target


def _target_from_velocity(velocity_mujoco, *, per_frame):
    """A synthetic target travelling ``per_frame`` metres along a velocity axis."""
    axis = np.asarray(velocity_mujoco, np.float64)
    horizontal = np.array([axis[0], axis[1], 0.0], np.float64)
    norm = float(np.linalg.norm(horizontal))
    unit = horizontal / norm if norm > 0.0 else horizontal
    steps = np.arange(_TARGET_ROWS, dtype=np.float64)[:, np.newaxis]
    delta = unit * per_frame
    path = (steps * delta[np.newaxis, :]).astype(np.float32)
    holden = mujoco_to_holden_vectors(tuple(float(v) for v in axis))
    command = MappingProxyType(
        {
            "requested_velocity_holden": readonly(np.asarray(holden, np.float64)),
            "desired_heading_holden_wxyz": readonly([1.0, 0.0, 0.0, 0.0]),
            "applied_velocity_holden": readonly(np.zeros((10, 3))),
        }
    )
    return replace(base_target(), virtual_root_position=path, command=command)


def direction_target(direction, *, per_frame, speed):
    """Target travelling along ``direction`` with a matching command."""
    axis = REGISTERED_DIRECTION_AXES_MUJOCO[direction]
    velocity_mujoco = (axis[0] * speed, axis[1] * speed, 0.0)
    return _target_from_velocity(velocity_mujoco, per_frame=per_frame)


class ScoreTargetDirectionTests(unittest.TestCase):
    def test_forward_target_scores_a_passing_direction_score(self):
        target = direction_target("forward", per_frame=0.02, speed=0.5)
        score = score_target_direction("forward", target)

        self.assertIsInstance(score, DirectionScore)
        self.assertEqual(score.direction, "forward")
        self.assertEqual(
            score.minimum_projection_m, REGISTERED_MINIMUM_PROJECTION_M["forward"]
        )
        # 0.02 m per frame across 19 gaps clears the 0.05 m floor along +x.
        self.assertAlmostEqual(score.signed_projection_m, 0.02 * 19, places=5)
        self.assertAlmostEqual(score.orthogonal_projection_m, 0.0, places=5)
        np.testing.assert_allclose(
            score.root_displacement_mujoco,
            (0.02 * 19, 0.0, 0.0),
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            score.requested_velocity_mujoco, (0.5, 0.0, 0.0), atol=1.0e-6
        )
        self.assertIs(score.passed, True)

    def test_pure_score_does_not_require_embedded_command_metadata(self):
        target = SimpleNamespace(
            virtual_root_position=np.asarray(
                ((1.0, 2.0, 0.8), (0.8, 2.0, 5.8)), dtype=np.float32
            )
        )

        score = score_target_direction("backward", target)

        self.assertIs(score.passed, True)
        self.assertAlmostEqual(score.signed_projection_m, 0.2, places=6)
        self.assertEqual(score.requested_velocity_mujoco, (-1.0, 0.0, 0.0))


class DirectionCommandTests(unittest.TestCase):
    def test_each_direction_requests_its_axis_at_speed_with_identity_heading(self):
        cases = {
            "forward": (0.5, 0.0, 0.0),
            "backward": (-0.5, 0.0, 0.0),
            "left": (0.0, 0.5, 0.0),
            "right": (0.0, -0.5, 0.0),
        }
        for direction, expected in cases.items():
            command = direction_command(direction, 0.5, chunk_index=0)
            self.assertIsInstance(command, CommandSample)
            self.assertEqual(command.chunk_index, 0)
            self.assertEqual(command.requested_velocity_mujoco, expected)
            self.assertEqual(
                command.desired_heading_mujoco_wxyz, (1.0, 0.0, 0.0, 0.0)
            )

    def test_rejects_unregistered_direction(self):
        with self.assertRaises(ContractError):
            direction_command("upward", 0.5, chunk_index=0)


def _real_initial():
    contract = load_joint_contract(JOINT_CONTRACT)
    return initial_from_chunk(make_source_chunk(contract))


class HashInitialBoundaryTests(unittest.TestCase):
    def test_hash_is_deterministic_lowercase_sha256(self):
        first = hash_initial_boundary(_real_initial())
        second = hash_initial_boundary(_real_initial())
        self.assertEqual(first, second)
        self.assertRegex(first, r"^[0-9a-f]{64}$")

    def test_hash_changes_when_a_boundary_value_changes(self):
        base = _real_initial()
        moved = replace(
            base,
            virtual_root_position_holden=readonly(
                np.asarray(base.virtual_root_position_holden, np.float64) + 1.0
            ),
        )
        self.assertNotEqual(
            hash_initial_boundary(base), hash_initial_boundary(moved)
        )


class FakeClient:
    """Records the live boundary call ordering for one direction."""

    def __init__(self, events, target_for, *, reset_id="reset"):
        self.events = events
        self._target_for = target_for
        self._reset_id = reset_id
        self.session_id = None
        self.outstanding_candidate_id = None
        self.active_candidate_id = None
        self.closed = False
        self.stdout_archive = "/runs/mm.stdout"
        self.stderr_archive = "/runs/mm.stderr"

    def hello(self):
        self.events.append("hello")
        return {"protocol_version": 1, "build_commit": "abc123"}

    def reset(self, config, *, session_id):
        self.events.append("reset")
        self.session_id = session_id
        return {
            "session_id": session_id,
            "initial_boundary": self._reset_id,
            "scene": {"scene_id": config.scene_id, "route_id": config.route_id},
        }

    def generate(self, command, *, session_id, candidate_id, predecessor_id, source_intervals):
        self.events.append("generate")
        assert source_intervals == 10, source_intervals
        assert predecessor_id is None, predecessor_id
        self.outstanding_candidate_id = candidate_id
        return {"candidate_id": candidate_id, "command": command}

    def commit(self, candidate_id):
        self.events.append("commit")
        raise AssertionError("direction gate must never commit a candidate")

    def abort(self, candidate_id):
        self.events.append("abort")
        assert candidate_id == self.outstanding_candidate_id
        self.outstanding_candidate_id = None

    def close(self):
        self.events.append("close")
        self.closed = True


class FakeValidator:
    def __init__(self, events, initial):
        self.events = events
        self._initial = initial

    def validate_initial(self, raw):
        self.events.append("validate_initial")
        return self._initial

    def validate_source(self, raw):
        self.events.append("validate_source")
        return SimpleNamespace(candidate_id=raw["candidate_id"], command=raw["command"])


class FakeTimeline:
    def __init__(self, events, target=None, *, per_frame=0.02):
        self.events = events
        self._target = target
        self._per_frame = per_frame
        self.last_accepted_candidate_id = None
        self._pending = None

    def prepare(self, source):
        self.events.append("prepare")
        self._pending = source.candidate_id
        target = self._target
        if target is None:
            # Model MM travelling along the commanded velocity axis.
            target = _target_from_velocity(
                source.command.requested_velocity_mujoco, per_frame=self._per_frame
            )
        return SimpleNamespace(
            source_candidate_id=source.candidate_id, target=target
        )

    def commit(self, prepared):
        self.events.append("timeline_commit")
        raise AssertionError("direction gate must never commit a timeline candidate")

    def abort(self, candidate_id):
        self.events.append("timeline_abort")
        assert candidate_id == self._pending
        self._pending = None


class RunLiveDirectionTests(unittest.TestCase):
    def _harness(self, direction):
        events = []
        initial = _real_initial()
        target = direction_target(direction, per_frame=0.02, speed=0.5)
        client = FakeClient(events, target)
        validator = FakeValidator(events, initial)
        timeline = FakeTimeline(events, target)
        result = run_live_direction(
            direction,
            client=client,
            validator=validator,
            timeline_factory=lambda boundary: timeline,
            config=SessionConfig("sonic-flat-baseline", "flat-12s", 0.0),
            session_id="gate-session",
            speed_mps=0.5,
        )
        return events, result, client, timeline

    def test_orders_reset_validate_generate_score_then_abort_and_close(self):
        events, result, client, timeline = self._harness("forward")
        self.assertEqual(
            events,
            [
                "hello",
                "reset",
                "validate_initial",
                "generate",
                "validate_source",
                "prepare",
                "timeline_abort",
                "abort",
                "close",
            ],
        )
        score = result.score
        self.assertIsInstance(score, DirectionScore)
        self.assertEqual(score.direction, "forward")
        self.assertIs(score.passed, True)
        self.assertEqual(result.reset_sha256, hash_initial_boundary(_real_initial()))
        # Preserved evidence: candidate id, hello/scene identity, and log paths.
        self.assertEqual(result.candidate_id, "gate-session:candidate:000000")
        self.assertEqual(result.hello["build_commit"], "abc123")
        self.assertEqual(result.scene["scene_id"], "sonic-flat-baseline")
        self.assertEqual(result.stdout_path, "/runs/mm.stdout")
        self.assertEqual(result.stderr_path, "/runs/mm.stderr")
        self.assertIs(client.closed, True)
        self.assertIsNone(client.outstanding_candidate_id)

    def test_client_is_closed_even_when_generation_raises(self):
        events = []
        initial = _real_initial()

        class RaisingClient(FakeClient):
            def generate(self, command, **kwargs):
                self.events.append("generate")
                raise RuntimeError("boom")

        client = RaisingClient(events, direction_target("left", per_frame=0.02, speed=0.5))
        validator = FakeValidator(events, initial)
        with self.assertRaises(RuntimeError):
            run_live_direction(
                "left",
                client=client,
                validator=validator,
                timeline_factory=lambda boundary: FakeTimeline(events, None),
                config=SessionConfig("sonic-flat-baseline", "flat-12s", 0.0),
                session_id="gate-session",
                speed_mps=0.5,
            )
        self.assertIs(client.closed, True)
        self.assertEqual(events[-1], "close")


class RunLiveDirectionGateTests(unittest.TestCase):
    def test_four_fresh_clients_from_identical_reset_pass(self):
        events = []
        initial = _real_initial()
        created_sessions = []

        def client_factory(direction, session_id):
            created_sessions.append((direction, session_id))
            return FakeClient(events, direction_target(direction, per_frame=0.02, speed=0.5))

        summary = run_live_direction_gate(
            client_factory=client_factory,
            validator_factory=lambda: FakeValidator(events, initial),
            timeline_factory=lambda boundary: FakeTimeline(
                events, None
            ),
            config=SessionConfig("sonic-flat-baseline", "flat-12s", 0.0),
            session_prefix="gate",
            speed_mps=0.5,
        )
        self.assertEqual(summary["schema"], "mm-direction-gate/v1")
        self.assertIs(summary["passed"], True)
        self.assertIs(summary["same_reset"], True)
        # Exactly four fresh isolated sessions, one per direction.
        self.assertEqual(len(created_sessions), 4)
        self.assertEqual(
            {direction for direction, _ in created_sessions}, set(_ALL_DIRECTIONS)
        )
        self.assertEqual(len({session for _, session in created_sessions}), 4)
        # Per-direction evidence is preserved in the JSON-safe summary.
        for direction in _ALL_DIRECTIONS:
            entry = summary["scores"][direction]
            self.assertEqual(
                entry["candidate_id"], f"gate:{direction}:candidate:000000"
            )
            self.assertEqual(entry["stdout_path"], "/runs/mm.stdout")
            self.assertEqual(entry["stderr_path"], "/runs/mm.stderr")
            self.assertEqual(entry["hello"]["build_commit"], "abc123")
            self.assertIn("requested_velocity_mujoco", entry)


_ALL_DIRECTIONS = ("forward", "backward", "left", "right")


def scores_all_passing():
    return {
        direction: score_target_direction(
            direction, direction_target(direction, per_frame=0.02, speed=0.5)
        )
        for direction in _ALL_DIRECTIONS
    }


def identical_reset_hashes(digest="a" * 64):
    return {direction: digest for direction in _ALL_DIRECTIONS}


class DirectionGateSummaryTests(unittest.TestCase):
    def test_all_pass_from_identical_reset_is_a_passing_json_safe_summary(self):
        import json

        summary = direction_gate_summary(
            scores_all_passing(), identical_reset_hashes()
        )
        # JSON-safe: round-trips without custom encoders.
        self.assertEqual(json.loads(json.dumps(summary)), summary)
        self.assertEqual(summary["schema"], "mm-direction-gate/v1")
        self.assertIs(summary["same_reset"], True)
        self.assertIs(summary["passed"], True)
        for direction in _ALL_DIRECTIONS:
            entry = summary["scores"][direction]
            self.assertEqual(entry["direction"], direction)
            self.assertIs(entry["passed"], True)
            self.assertEqual(entry["reset_sha256"], "a" * 64)

    def test_ordered_sequences_are_accepted_by_the_public_api(self):
        scores = [scores_all_passing()[direction] for direction in _ALL_DIRECTIONS]
        summary = direction_gate_summary(scores, ["a" * 64] * 4)
        self.assertIs(summary["passed"], True)

    def test_duplicate_direction_in_sequence_fails_without_hiding_evidence(self):
        scores = [scores_all_passing()[direction] for direction in _ALL_DIRECTIONS]
        scores[-1] = scores[1]
        summary = direction_gate_summary(scores, ["a" * 64] * 4)
        self.assertIs(summary["passed"], False)
        self.assertEqual(summary["scores"]["right"]["direction"], "backward")

    def test_mismatched_reset_hashes_fail_and_clear_same_reset(self):
        hashes = identical_reset_hashes()
        hashes["right"] = "b" * 64
        summary = direction_gate_summary(scores_all_passing(), hashes)
        self.assertIs(summary["same_reset"], False)
        self.assertIs(summary["passed"], False)

    def test_one_failing_direction_fails_the_summary(self):
        scores = scores_all_passing()
        # Neutral travel does not clear the floor: force a failing score.
        scores["left"] = score_target_direction(
            "left", direction_target("left", per_frame=0.0, speed=0.5)
        )
        self.assertIs(scores["left"].passed, False)
        summary = direction_gate_summary(scores, identical_reset_hashes())
        self.assertIs(summary["same_reset"], True)
        self.assertIs(summary["passed"], False)

    def test_missing_direction_is_rejected(self):
        scores = scores_all_passing()
        del scores["backward"]
        hashes = identical_reset_hashes()
        del hashes["backward"]
        with self.assertRaises(ContractError):
            direction_gate_summary(scores, hashes)

    def test_lowercase_sha256_reset_hashes_are_required(self):
        with self.assertRaises(ContractError):
            direction_gate_summary(
                scores_all_passing(), identical_reset_hashes("A" * 64)
            )


class GateParserTests(unittest.TestCase):
    def test_production_defaults_match_flat_diagnostic_environment(self):
        args = build_gate_parser().parse_args(
            ["--mm-server", "/bin/mm", "--output-dir", "/tmp/out"]
        )
        self.assertEqual(args.mm_server, "/bin/mm")
        self.assertEqual(args.output_dir, "/tmp/out")
        self.assertEqual(args.scene_id, "sonic-flat-baseline")
        self.assertEqual(args.route_id, "flat-12s")
        self.assertEqual(args.terrain_weight, 0.0)
        self.assertEqual(args.speed_mps, 0.5)
        self.assertTrue(args.terrain_dir)

    def test_overrides_are_accepted(self):
        args = build_gate_parser().parse_args(
            [
                "--mm-server", "/bin/mm",
                "--output-dir", "/tmp/out",
                "--terrain-dir", "/data/terrain",
                "--scene-id", "custom",
                "--route-id", "route-9",
                "--terrain-weight", "4.0",
                "--speed-mps", "0.8",
            ]
        )
        self.assertEqual(args.scene_id, "custom")
        self.assertEqual(args.route_id, "route-9")
        self.assertEqual(args.terrain_weight, 4.0)
        self.assertEqual(args.speed_mps, 0.8)
        self.assertEqual(args.terrain_dir, "/data/terrain")


class WriteGateArtifactTests(unittest.TestCase):
    def test_writes_json_atomically_and_returns_absolute_path(self):
        import json
        import tempfile

        summary = direction_gate_summary(
            scores_all_passing(), identical_reset_hashes()
        )
        with tempfile.TemporaryDirectory() as output_dir:
            path = write_gate_artifact(summary, output_dir)
            self.assertTrue(os.path.isabs(path))
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), summary)
            # No temporary siblings survive the atomic write.
            siblings = os.listdir(output_dir)
            self.assertEqual(len(siblings), 1, siblings)


class MainTests(unittest.TestCase):
    def _run_main(self, output_dir, *, run_gate):
        import io
        import contextlib

        argv = [
            "--mm-server", "/bin/mm",
            "--output-dir", output_dir,
            "--terrain-dir", "/data/terrain",
        ]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(argv, run_gate=run_gate)
        return exit_code, stdout.getvalue().strip()

    def test_exit_zero_and_prints_absolute_artifact_path_when_all_pass(self):
        import json
        import tempfile

        def run_gate(**kwargs):
            self.assertEqual(kwargs["config"].scene_id, "sonic-flat-baseline")
            self.assertEqual(kwargs["speed_mps"], 0.5)
            return direction_gate_summary(
                scores_all_passing(), identical_reset_hashes()
            )

        with tempfile.TemporaryDirectory() as output_dir:
            exit_code, printed = self._run_main(output_dir, run_gate=run_gate)
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.isabs(printed))
            with open(printed, encoding="utf-8") as handle:
                self.assertIs(json.load(handle)["passed"], True)

    def test_exit_nonzero_but_still_writes_evidence_when_a_direction_fails(self):
        import json
        import tempfile

        def run_gate(**kwargs):
            hashes = identical_reset_hashes()
            hashes["right"] = "b" * 64  # differing reset -> gate fails
            return direction_gate_summary(scores_all_passing(), hashes)

        with tempfile.TemporaryDirectory() as output_dir:
            exit_code, printed = self._run_main(output_dir, run_gate=run_gate)
            self.assertNotEqual(exit_code, 0)
            self.assertTrue(os.path.isabs(printed))
            with open(printed, encoding="utf-8") as handle:
                self.assertIs(json.load(handle)["passed"], False)


if __name__ == "__main__":
    unittest.main()
