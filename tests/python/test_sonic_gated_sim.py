import io
import json
import math
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import types
import unittest
from unittest.mock import patch

import numpy as np

from mm_sonic import gated_sim
from mm_sonic.gated_sim import (
    ExternalGearBackend,
    GatedSimulatorRunner,
    ProtocolError,
    _normalize_onscreen_viewer,
    physical_qpos_with_perturbation,
    serve_jsonl,
)


class FakeBackend:
    def __init__(self, *, nq=36, sim_dt=0.005, wall_clock_pacing=False):
        self._model = SimpleNamespace(nq=nq)
        self._data = SimpleNamespace(
            time=0.0,
            qpos=np.zeros(nq, dtype=np.float64),
        )
        self._sim_dt = sim_dt
        self._wall_clock_pacing = wall_clock_pacing
        self.reset_calls = []
        self.step_calls = 0
        self.sample_calls = 0
        self.prime_calls = 0
        self.refresh_calls = 0
        self.low_command_received = False
        self.low_command_q_target = np.zeros(29, dtype=np.float32)
        self.camera_calls = []
        self.closed = False

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @property
    def sim_dt(self):
        return self._sim_dt

    @property
    def wall_clock_pacing(self):
        return self._wall_clock_pacing

    def reset_from_qpos(
        self, qpos, lateral_offset_m, yaw_offset_rad, *, elastic_band_enabled
    ):
        self.reset_calls.append(
            (qpos.copy(), lateral_offset_m, yaw_offset_rad, elastic_band_enabled)
        )
        self._data.qpos[:] = physical_qpos_with_perturbation(
            qpos,
            lateral_offset_m=lateral_offset_m,
            yaw_offset_rad=yaw_offset_rad,
        )
        self._data.time = 0.0

    def step(self):
        self.step_calls += 1
        self._data.time = self.step_calls * self._sim_dt

    def sample(self):
        self.sample_calls += 1
        return {
            "qpos": self._data.qpos.copy(),
            "contacts": [
                {
                    "geom1": "left_foot",
                    "geom2": "floor",
                    "distance_m": -0.001,
                }
            ],
        }

    def prime_low_state(self):
        self.prime_calls += 1

    def refresh_low_state(self):
        self.refresh_calls += 1

    def low_command_snapshot(self):
        return {
            "received": self.low_command_received,
            "q_target": (
                self.low_command_q_target.copy()
                if self.low_command_received
                else None
            ),
        }

    def set_camera(self, azimuth_deg, elevation_deg, distance_m):
        self.camera_calls.append((azimuth_deg, elevation_deg, distance_m))

    def close(self):
        self.closed = True


class GatedSimulatorRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.scene = self.root / "gear_scene.xml"
        self.scene.write_text("<mujoco/>", encoding="utf-8")
        self.log_dir = self.root / "sim"
        self.backends = []

        def factory(scene_xml):
            self.assertEqual(scene_xml, self.scene.resolve())
            backend = FakeBackend()
            self.backends.append(backend)
            return backend

        self.runner = GatedSimulatorRunner(factory, run_root=self.root)

    def tearDown(self):
        self.runner.close()
        self.temporary.cleanup()

    def reset(self, qpos=None, *, lateral=0.0, yaw=0.0, elastic_band_enabled=True):
        if qpos is None:
            qpos = np.zeros(36, dtype=np.float64)
            qpos[2] = 0.8
            qpos[3] = 1.0
            qpos[7:] = np.linspace(-0.4, 0.4, 29)
        return self.runner.reset(
            scene_xml=self.scene,
            initial_qpos=qpos,
            lateral_offset_m=lateral,
            yaw_offset_rad=yaw,
            log_dir=self.log_dir,
            elastic_band_enabled=elastic_band_enabled,
        )

    def test_reset_requires_and_echoes_a_strict_band_boolean(self):
        first = self.reset(elastic_band_enabled=True)
        self.assertIs(first["elastic_band_enabled"], True)
        self.assertIs(self.backends[-1].reset_calls[-1][3], True)

        second = self.runner.reset(
            scene_xml=self.scene,
            initial_qpos=np.r_[np.zeros(3), 1.0, np.zeros(32)],
            lateral_offset_m=0.0,
            yaw_offset_rad=0.0,
            log_dir=self.root / "scored-band-sim",
            elastic_band_enabled=False,
        )
        self.assertIs(second["elastic_band_enabled"], False)
        self.assertIs(self.backends[-1].reset_calls[-1][3], False)

    def test_reset_rejects_nonboolean_band_state_without_touching_backend(self):
        for value in (0, 1, None, "true"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ProtocolError, "elastic_band_enabled"):
                    self.runner.reset(
                        scene_xml=self.scene,
                        initial_qpos=np.r_[np.zeros(3), 1.0, np.zeros(32)],
                        lateral_offset_m=0.0,
                        yaw_offset_rad=0.0,
                        log_dir=self.root / f"band-{value}-sim",
                        elastic_band_enabled=value,
                    )
        self.assertEqual(self.backends, [])

    def test_reset_is_idle_and_requires_exact_nq_values(self):
        with self.assertRaisesRegex(ProtocolError, "exactly 36"):
            self.reset(np.array([0.0]))
        self.assertEqual(self.backends[0].step_calls, 0)

        result = self.reset()
        self.assertEqual(result["nq"], 36)
        self.assertEqual(result["sim_dt_s"], 0.005)
        self.assertEqual(self.backends[-1].step_calls, 0)
        self.assertEqual(self.backends[-1].sample_calls, 0)

    def test_same_scene_reset_reuses_backend_and_rotates_log_epoch(self):
        first_logs = self.log_dir
        second_logs = self.root / "scored-sim"
        initial = np.zeros(36, dtype=np.float64)
        initial[2] = 0.8
        initial[3] = 1.0

        self.reset(initial)
        self.runner.advance(1)
        self.runner.reset(
            scene_xml=self.scene,
            initial_qpos=initial,
            lateral_offset_m=0.0,
            yaw_offset_rad=0.0,
            log_dir=second_logs,
            elastic_band_enabled=True,
        )

        self.assertEqual(len(self.backends), 1)
        self.assertEqual(len(self.backends[0].reset_calls), 2)
        self.assertFalse(self.backends[0].closed)
        self.assertTrue((first_logs / "contacts.jsonl").is_file())
        self.assertTrue((second_logs / "contacts.jsonl").is_file())
        self.assertEqual(self.runner.snapshot()["steps"], 0)

    def test_same_path_scene_replacement_requires_a_fresh_child(self):
        initial = np.zeros(36, dtype=np.float64)
        initial[2] = 0.8
        initial[3] = 1.0
        original_scene = self.root / "original-scene.xml"

        self.reset(initial)
        self.runner.advance(1)
        self.scene.rename(original_scene)
        self.scene.write_text("<mujoco model='replacement'/>", encoding="utf-8")

        with self.assertRaisesRegex(
            ProtocolError,
            "scene identity/path changed; fresh simulator process required",
        ):
            self.runner.reset(
                scene_xml=self.scene,
                initial_qpos=initial,
                lateral_offset_m=0.0,
                yaw_offset_rad=0.0,
                log_dir=self.root / "replacement-scene-logs",
                elastic_band_enabled=True,
            )

        self.assertEqual(len(self.backends), 1)
        self.assertFalse(self.backends[0].closed)
        self.assertTrue((self.log_dir / "contacts.jsonl").is_file())
        self.assertFalse((self.root / "replacement-scene-logs").exists())
        self.assertTrue(original_scene.is_file())
        self.assertTrue(self.scene.is_file())

    def test_in_place_scene_mutation_requires_a_fresh_child(self):
        initial = np.zeros(36, dtype=np.float64)
        initial[2] = 0.8
        initial[3] = 1.0

        self.reset(initial)
        original_inode = self.scene.stat().st_ino
        self.scene.write_text("<mujoco model='mutated'/>", encoding="utf-8")
        self.assertEqual(self.scene.stat().st_ino, original_inode)

        with self.assertRaisesRegex(
            ProtocolError,
            "scene identity/path changed; fresh simulator process required",
        ):
            self.runner.reset(
                scene_xml=self.scene,
                initial_qpos=initial,
                lateral_offset_m=0.0,
                yaw_offset_rad=0.0,
                log_dir=self.root / "mutated-scene-logs",
                elastic_band_enabled=True,
            )

        self.assertEqual(len(self.backends), 1)
        self.assertFalse(self.backends[0].closed)
        self.assertFalse((self.root / "mutated-scene-logs").exists())

    def test_different_scene_path_requires_a_fresh_child(self):
        initial = np.zeros(36, dtype=np.float64)
        initial[2] = 0.8
        initial[3] = 1.0
        other_scene = self.root / "other-scene.xml"
        other_scene.write_text("<mujoco model='other'/>", encoding="utf-8")

        self.reset(initial)
        with self.assertRaisesRegex(
            ProtocolError,
            "scene identity/path changed; fresh simulator process required",
        ):
            self.runner.reset(
                scene_xml=other_scene,
                initial_qpos=initial,
                lateral_offset_m=0.0,
                yaw_offset_rad=0.0,
                log_dir=self.root / "other-scene-logs",
                elastic_band_enabled=True,
            )

        self.assertEqual(len(self.backends), 1)
        self.assertFalse(self.backends[0].closed)
        self.assertFalse((self.root / "other-scene-logs").exists())

    def test_reset_perturbs_only_copied_physical_horizontal_root_and_yaw(self):
        initial = np.zeros(36, dtype=np.float64)
        initial[:7] = [1.25, -0.5, 0.82, 1.0, 0.0, 0.0, 0.0]
        initial[7:] = np.linspace(-1.0, 1.0, 29)
        unchanged_reference = initial.copy()

        self.reset(initial, lateral=0.06, yaw=math.radians(4.0))

        np.testing.assert_array_equal(initial, unchanged_reference)
        physical = self.backends[-1].data.qpos
        self.assertEqual(physical[0], unchanged_reference[0])
        self.assertAlmostEqual(physical[1], unchanged_reference[1] + 0.06)
        self.assertEqual(physical[2], unchanged_reference[2])
        np.testing.assert_array_equal(physical[7:], unchanged_reference[7:])
        expected_yaw = np.array(
            [
                math.cos(math.radians(2.0)),
                0.0,
                0.0,
                math.sin(math.radians(2.0)),
            ]
        )
        np.testing.assert_allclose(physical[3:7], expected_yaw, atol=1e-15)

    def test_advance_steps_exactly_and_logs_contacts_each_step_and_state_at_50hz(self):
        self.reset()
        backend = self.backends[-1]

        result = self.runner.advance(80)

        self.assertEqual(result.steps, 80)
        self.assertEqual(result.sim_time_start_s, 0.0)
        self.assertAlmostEqual(result.sim_time_end_s, 0.4, places=15)
        self.assertEqual(result.contact_rows, 80)
        self.assertEqual(result.state_rows, 20)
        self.assertEqual(backend.step_calls, 80)
        self.assertEqual(backend.sample_calls, 80)

        contact_rows = [
            json.loads(line)
            for line in (self.log_dir / "contacts.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        state_rows = [
            json.loads(line)
            for line in (self.log_dir / "state.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual([row["step"] for row in contact_rows], list(range(1, 81)))
        self.assertEqual(
            [row["step"] for row in state_rows],
            list(range(4, 81, 4)),
        )
        self.assertEqual(contact_rows[-1]["contacts"][0]["geom2"], "floor")
        self.assertAlmostEqual(state_rows[-1]["sim_time_s"], 0.4, places=15)

    def test_pacing_deadline_includes_sampling_and_logging_without_drift(self):
        self.reset()
        self.backends[-1]._wall_clock_pacing = True

        with (
            patch.object(
                gated_sim.time,
                "monotonic",
                side_effect=[100.0, 100.004, 100.009],
            ),
            patch.object(gated_sim.time, "sleep") as sleep,
        ):
            self.runner.advance(2)

        self.assertEqual(sleep.call_count, 2)
        self.assertAlmostEqual(sleep.call_args_list[0].args[0], 0.001, places=12)
        self.assertAlmostEqual(sleep.call_args_list[1].args[0], 0.001, places=12)

    def test_prime_low_state_publishes_without_stepping_or_changing_evidence(self):
        with self.assertRaisesRegex(ProtocolError, "reset is required"):
            self.runner.prime_low_state()

        self.reset()
        backend = self.backends[-1]
        before_snapshot = self.runner.snapshot()
        before_qpos = backend.data.qpos.copy()
        before_time = backend.data.time

        result = self.runner.prime_low_state()

        self.assertEqual(
            result,
            {
                "published": True,
                "steps": 0,
                "sim_time_s": 0.0,
                "state_rows": 0,
                "contact_rows": 0,
            },
        )
        self.assertEqual(backend.prime_calls, 1)
        self.assertEqual(backend.step_calls, 0)
        self.assertEqual(backend.sample_calls, 0)
        np.testing.assert_array_equal(backend.data.qpos, before_qpos)
        self.assertEqual(backend.data.time, before_time)
        self.assertEqual(self.runner.snapshot(), before_snapshot)

    def test_refresh_low_state_preserves_existing_epoch_evidence(self):
        self.reset()
        self.runner.advance(4)
        backend = self.backends[-1]
        before = self.runner.snapshot()

        result = self.runner.refresh_low_state()

        self.assertEqual(result, {"published": True, **before})
        self.assertEqual(backend.refresh_calls, 1)
        self.assertEqual(self.runner.snapshot(), before)

    def test_low_command_snapshot_is_receiver_owned_and_does_not_step(self):
        with self.assertRaisesRegex(ProtocolError, "reset is required"):
            self.runner.low_command_snapshot()

        self.reset()
        backend = self.backends[-1]
        self.assertEqual(
            self.runner.low_command_snapshot(),
            {"received": False, "q_target": None},
        )

        target = np.linspace(-0.7, 0.7, 29, dtype=np.float32)
        backend.low_command_received = True
        backend.low_command_q_target[:] = target
        received = self.runner.low_command_snapshot()
        backend.low_command_q_target[:] = 99.0

        self.assertIs(received["received"], True)
        self.assertEqual(tuple(received["q_target"]), tuple(float(x) for x in target))
        self.assertEqual(backend.step_calls, 0)
        self.assertEqual(backend.sample_calls, 0)
        self.assertEqual(self.runner.snapshot()["steps"], 0)

    def test_camera_request_applies_without_advancing_physics(self):
        self.reset()
        backend = self.backends[-1]
        self.runner.advance(4)
        before = self.runner.snapshot()

        applied = self.runner.set_camera(
            sequence=7,
            azimuth_deg=45.0,
            elevation_deg=-20.0,
            distance_m=4.0,
        )

        self.assertEqual(
            applied,
            {
                "sequence": 7,
                "azimuth_deg": 45.0,
                "elevation_deg": -20.0,
                "distance_m": 4.0,
            },
        )
        self.assertEqual(self.runner.snapshot(), before)
        self.assertEqual(backend.camera_calls[-1], (45.0, -20.0, 4.0))
        self.assertEqual(backend.step_calls, 4)
        self.assertEqual(backend.sample_calls, 4)

    def test_camera_accepts_positional_arguments(self):
        self.reset()
        backend = self.backends[-1]

        applied = self.runner.set_camera(7, 45.0, -20.0, 4.0)

        self.assertEqual(
            applied,
            {
                "sequence": 7,
                "azimuth_deg": 45.0,
                "elevation_deg": -20.0,
                "distance_m": 4.0,
            },
        )
        self.assertEqual(backend.camera_calls[-1], (45.0, -20.0, 4.0))

    def test_camera_accepts_inclusive_distance_bounds(self):
        self.reset()
        backend = self.backends[-1]
        near = self.runner.set_camera(
            sequence=0, azimuth_deg=0.0, elevation_deg=0.0, distance_m=0.1
        )
        far = self.runner.set_camera(
            sequence=1, azimuth_deg=0.0, elevation_deg=0.0, distance_m=100.0
        )
        self.assertEqual(near["distance_m"], 0.1)
        self.assertEqual(far["distance_m"], 100.0)
        self.assertEqual(backend.camera_calls[0], (0.0, 0.0, 0.1))
        self.assertEqual(backend.camera_calls[1], (0.0, 0.0, 100.0))

    def test_camera_rejects_boolean_or_negative_sequence(self):
        self.reset()
        backend = self.backends[-1]
        for sequence in (True, False, -1):
            with self.subTest(sequence=sequence):
                with self.assertRaisesRegex(ProtocolError, "sequence"):
                    self.runner.set_camera(
                        sequence=sequence,
                        azimuth_deg=0.0,
                        elevation_deg=0.0,
                        distance_m=1.0,
                    )
        self.assertEqual(backend.camera_calls, [])

    def test_camera_rejects_nonfinite_angles(self):
        self.reset()
        backend = self.backends[-1]
        for azimuth, elevation in (
            (math.inf, 0.0),
            (0.0, math.nan),
        ):
            with self.subTest(azimuth=azimuth, elevation=elevation):
                with self.assertRaisesRegex(ProtocolError, "finite"):
                    self.runner.set_camera(
                        sequence=0,
                        azimuth_deg=azimuth,
                        elevation_deg=elevation,
                        distance_m=1.0,
                    )
        self.assertEqual(backend.camera_calls, [])

    def test_camera_rejects_distance_outside_inclusive_range(self):
        self.reset()
        backend = self.backends[-1]
        for distance in (0.0, 0.09, 100.01, math.inf):
            with self.subTest(distance=distance):
                with self.assertRaisesRegex(ProtocolError, "distance_m"):
                    self.runner.set_camera(
                        sequence=0,
                        azimuth_deg=0.0,
                        elevation_deg=0.0,
                        distance_m=distance,
                    )
        self.assertEqual(backend.camera_calls, [])

    def test_sampling_phase_and_row_counts_continue_across_advances(self):
        self.reset()
        first = self.runner.advance(3)
        second = self.runner.advance(5)
        snapshot = self.runner.snapshot()

        self.assertEqual(first.state_rows, 0)
        self.assertEqual(second.state_rows, 2)
        self.assertEqual(snapshot["steps"], 8)
        self.assertEqual(snapshot["state_rows"], 2)
        self.assertEqual(snapshot["contact_rows"], 8)
        self.assertAlmostEqual(snapshot["sim_time_s"], 0.04, places=15)

    def test_rejects_nonpositive_or_boolean_steps_without_stepping(self):
        self.reset()
        backend = self.backends[-1]
        for value in (0, -1, True, 1.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ProtocolError, "positive integer"):
                    self.runner.advance(value)
        self.assertEqual(backend.step_calls, 0)

    def test_requires_sim_dt_to_divide_50hz_period_exactly(self):
        bad = FakeBackend(sim_dt=0.003)
        runner = GatedSimulatorRunner(lambda _scene: bad, run_root=self.root)
        try:
            with self.assertRaisesRegex(ProtocolError, "50 Hz"):
                runner.reset(
                    scene_xml=self.scene,
                    initial_qpos=np.r_[np.zeros(3), 1.0, np.zeros(32)],
                    lateral_offset_m=0.0,
                    yaw_offset_rad=0.0,
                    log_dir=self.log_dir / "bad",
                    elastic_band_enabled=True,
                )
            self.assertEqual(bad.step_calls, 0)
        finally:
            runner.close()

    def test_reset_rejects_relative_scene_and_log_paths(self):
        qpos = np.zeros(36)
        qpos[3] = 1.0
        original_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            with self.assertRaisesRegex(ProtocolError, "absolute"):
                self.runner.reset(
                    scene_xml=Path("gear_scene.xml"),
                    initial_qpos=qpos,
                    lateral_offset_m=0.0,
                    yaw_offset_rad=0.0,
                    log_dir=self.log_dir,
                    elastic_band_enabled=True,
                )
            with self.assertRaisesRegex(ProtocolError, "absolute"):
                self.runner.reset(
                    scene_xml=self.scene,
                    initial_qpos=qpos,
                    lateral_offset_m=0.0,
                    yaw_offset_rad=0.0,
                    log_dir=Path("relative-log"),
                    elastic_band_enabled=True,
                )
        finally:
            os.chdir(original_cwd)

    def test_run_root_rejects_escape_symlink_and_preexisting_log_directory(self):
        qpos = np.zeros(36)
        qpos[3] = 1.0
        runner = GatedSimulatorRunner(
            lambda _scene: FakeBackend(),
            run_root=self.root,
        )
        try:
            with tempfile.TemporaryDirectory() as outside_text:
                outside = Path(outside_text)
                outside_scene = outside / "scene.xml"
                outside_scene.write_text("<mujoco/>", encoding="utf-8")
                with self.assertRaisesRegex(ProtocolError, "run_root"):
                    runner.reset(
                        scene_xml=outside_scene,
                        initial_qpos=qpos,
                        lateral_offset_m=0.0,
                        yaw_offset_rad=0.0,
                        log_dir=self.root / "outside-scene-logs",
                        elastic_band_enabled=True,
                    )

                linked_scene = self.root / "linked-scene.xml"
                linked_scene.symlink_to(outside_scene)
                with self.assertRaisesRegex(ProtocolError, "symlink"):
                    runner.reset(
                        scene_xml=linked_scene,
                        initial_qpos=qpos,
                        lateral_offset_m=0.0,
                        yaw_offset_rad=0.0,
                        log_dir=self.root / "linked-scene-logs",
                        elastic_band_enabled=True,
                    )

            existing_logs = self.root / "existing-sim-logs"
            existing_logs.mkdir()
            marker = existing_logs / "keep"
            marker.write_text("original", encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "exists"):
                runner.reset(
                    scene_xml=self.scene,
                    initial_qpos=qpos,
                    lateral_offset_m=0.0,
                    yaw_offset_rad=0.0,
                    log_dir=existing_logs,
                    elastic_band_enabled=True,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "original")
        finally:
            runner.close()


class GatedSimulatorProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.scene = self.root / "scene.xml"
        self.scene.write_text("<mujoco/>", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def run_server(self, requests):
        backend = FakeBackend()
        stdin = io.StringIO("".join(line + "\n" for line in requests))
        stdout = io.StringIO()
        serve_jsonl(
            run_root=self.root,
            backend_factory=lambda _scene: backend,
            input_stream=stdin,
            output_stream=stdout,
        )
        return [json.loads(line) for line in stdout.getvalue().splitlines()], backend

    def test_exact_protocol_rejects_duplicate_and_unknown_keys(self):
        responses, backend = self.run_server(
            [
                '{"v":1,"op":"hello","request_id":"g0","request_id":"evil"}',
                '{"v":1,"op":"hello","request_id":"g1","extra":0}',
                '{"v":1,"op":"close","request_id":"g2"}',
            ]
        )
        self.assertFalse(responses[0]["ok"])
        self.assertEqual(responses[0]["error"]["code"], "invalid_request")
        self.assertIn("duplicate JSON key", responses[0]["error"]["message"])
        self.assertFalse(responses[1]["ok"])
        self.assertIn("keys differ", responses[1]["error"]["message"])
        self.assertTrue(responses[2]["ok"])
        self.assertFalse(backend.closed)  # hello/close never construct the backend

    def test_one_value_reset_example_is_rejected_before_any_step(self):
        reset = json.dumps(
            {
                "v": 1,
                "op": "reset",
                "request_id": "g1",
                "scene_xml": str(self.scene),
                "initial_qpos": [0.0],
                "lateral_offset_m": 0.0,
                "yaw_offset_rad": 0.0,
                "log_dir": str(self.root / "sim"),
                "elastic_band_enabled": True,
            },
            separators=(",", ":"),
        )
        responses, backend = self.run_server(
            [
                '{"v":1,"op":"hello","request_id":"g0"}',
                reset,
                '{"v":1,"op":"close","request_id":"g2"}',
            ]
        )
        self.assertTrue(responses[0]["ok"])
        self.assertFalse(responses[1]["ok"])
        self.assertIn("exactly 36", responses[1]["error"]["message"])
        self.assertEqual(backend.step_calls, 0)

    def test_prime_low_state_protocol_is_exact_and_does_not_advance(self):
        qpos = [0.0] * 36
        qpos[2] = 0.8
        qpos[3] = 1.0
        reset = json.dumps(
            {
                "v": 1,
                "op": "reset",
                "request_id": "g0",
                "scene_xml": str(self.scene),
                "initial_qpos": qpos,
                "lateral_offset_m": 0.0,
                "yaw_offset_rad": 0.0,
                "log_dir": str(self.root / "prime-sim"),
                "elastic_band_enabled": False,
            },
            separators=(",", ":"),
        )
        responses, backend = self.run_server(
            [
                reset,
                '{"v":1,"op":"prime_low_state","request_id":"g1"}',
                '{"v":1,"op":"snapshot","request_id":"g2"}',
                '{"v":1,"op":"low_command","request_id":"g3"}',
                '{"v":1,"op":"low_command","request_id":"g4","extra":0}',
                '{"v":1,"op":"prime_low_state","request_id":"g5","extra":0}',
                '{"v":1,"op":"close","request_id":"g6"}',
            ]
        )

        self.assertTrue(responses[1]["ok"])
        self.assertEqual(
            responses[1]["data"],
            {
                "published": True,
                "steps": 0,
                "sim_time_s": 0.0,
                "state_rows": 0,
                "contact_rows": 0,
            },
        )
        self.assertEqual(
            responses[2]["data"],
            {
                "steps": 0,
                "sim_time_s": 0.0,
                "state_rows": 0,
                "contact_rows": 0,
            },
        )
        self.assertEqual(
            responses[3]["data"],
            {"received": False, "q_target": None},
        )
        self.assertFalse(responses[4]["ok"])
        self.assertIn("keys differ", responses[4]["error"]["message"])
        self.assertFalse(responses[5]["ok"])
        self.assertIn("keys differ", responses[5]["error"]["message"])
        self.assertEqual(backend.prime_calls, 1)
        self.assertEqual(backend.step_calls, 0)

    def test_refresh_low_state_protocol_preserves_current_epoch(self):
        qpos = [0.0] * 36
        qpos[2] = 0.8
        qpos[3] = 1.0
        reset = json.dumps(
            {
                "v": 1,
                "op": "reset",
                "request_id": "r0",
                "scene_xml": str(self.scene),
                "initial_qpos": qpos,
                "lateral_offset_m": 0.0,
                "yaw_offset_rad": 0.0,
                "log_dir": str(self.root / "refresh-sim"),
                "elastic_band_enabled": False,
            },
            separators=(",", ":"),
        )
        responses, backend = self.run_server(
            [
                reset,
                '{"v":1,"op":"advance","request_id":"r1","steps":4}',
                '{"v":1,"op":"refresh_low_state","request_id":"r2"}',
                '{"v":1,"op":"snapshot","request_id":"r3"}',
                '{"v":1,"op":"refresh_low_state","request_id":"r4","extra":0}',
                '{"v":1,"op":"close","request_id":"r5"}',
            ]
        )

        expected = {
            "steps": 4,
            "sim_time_s": 0.02,
            "state_rows": 1,
            "contact_rows": 4,
        }
        self.assertEqual(responses[2]["data"], {"published": True, **expected})
        self.assertEqual(responses[3]["data"], expected)
        self.assertEqual(backend.refresh_calls, 1)
        self.assertEqual(backend.step_calls, 4)
        self.assertFalse(responses[4]["ok"])
        self.assertIn("keys differ", responses[4]["error"]["message"])

    def test_reset_request_echoes_band_boolean_and_rejects_nonboolean(self):
        def reset_request(band):
            return json.dumps(
                {
                    "v": 1,
                    "op": "reset",
                    "request_id": "g1",
                    "scene_xml": str(self.scene),
                    "initial_qpos": [0.0] * 3 + [1.0] + [0.0] * 32,
                    "lateral_offset_m": 0.0,
                    "yaw_offset_rad": 0.0,
                    "log_dir": str(self.root / "band-sim"),
                    "elastic_band_enabled": band,
                },
                separators=(",", ":"),
            )

        responses, backend = self.run_server(
            [
                '{"v":1,"op":"hello","request_id":"g0"}',
                reset_request(False),
                '{"v":1,"op":"close","request_id":"g2"}',
            ]
        )
        self.assertTrue(responses[1]["ok"])
        self.assertIs(responses[1]["data"]["elastic_band_enabled"], False)
        self.assertIs(backend.reset_calls[-1][3], False)

        rejected, rejected_backend = self.run_server(
            [
                '{"v":1,"op":"hello","request_id":"g0"}',
                reset_request(1),
                '{"v":1,"op":"close","request_id":"g2"}',
            ]
        )
        self.assertFalse(rejected[1]["ok"])
        self.assertIn(
            "elastic_band_enabled", rejected[1]["error"]["message"]
        )
        self.assertEqual(rejected_backend.reset_calls, [])

    def test_camera_request_requires_exact_keys_and_echoes_values(self):
        reset = json.dumps(
            {
                "v": 1,
                "op": "reset",
                "request_id": "g1",
                "scene_xml": str(self.scene),
                "initial_qpos": [0.0] * 3 + [1.0] + [0.0] * 32,
                "lateral_offset_m": 0.0,
                "yaw_offset_rad": 0.0,
                "log_dir": str(self.root / "cam-sim"),
                "elastic_band_enabled": True,
            },
            separators=(",", ":"),
        )
        camera = json.dumps(
            {
                "v": 1,
                "op": "camera",
                "request_id": "g2",
                "sequence": 5,
                "azimuth_deg": 30.0,
                "elevation_deg": -15.0,
                "distance_m": 3.5,
            },
            separators=(",", ":"),
        )
        responses, backend = self.run_server(
            [
                '{"v":1,"op":"hello","request_id":"g0"}',
                reset,
                camera,
                '{"v":1,"op":"camera","request_id":"g3","sequence":5,'
                '"azimuth_deg":30.0,"elevation_deg":-15.0}',
                '{"v":1,"op":"camera","request_id":"g4","sequence":5,'
                '"azimuth_deg":30.0,"elevation_deg":-15.0,"distance_m":3.5,'
                '"extra":0}',
                '{"v":1,"op":"close","request_id":"g5"}',
            ]
        )
        self.assertTrue(responses[2]["ok"])
        self.assertEqual(
            responses[2]["data"],
            {
                "sequence": 5,
                "azimuth_deg": 30.0,
                "elevation_deg": -15.0,
                "distance_m": 3.5,
            },
        )
        self.assertEqual(backend.camera_calls[-1], (30.0, -15.0, 3.5))
        self.assertFalse(responses[3]["ok"])
        self.assertIn("keys differ", responses[3]["error"]["message"])
        self.assertFalse(responses[4]["ok"])
        self.assertIn("keys differ", responses[4]["error"]["message"])
        self.assertEqual(backend.step_calls, 0)

    def test_advance_before_reset_is_rejected_without_constructing_backend(self):
        constructed = []
        stdout = io.StringIO()
        serve_jsonl(
            run_root=self.root,
            backend_factory=lambda _scene: constructed.append(True),
            input_stream=io.StringIO(
                '{"v":1,"op":"advance","request_id":"g0","steps":1}\n'
                '{"v":1,"op":"close","request_id":"g1"}\n'
            ),
            output_stream=stdout,
        )
        responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertFalse(responses[0]["ok"])
        self.assertIn("reset", responses[0]["error"]["message"])
        self.assertEqual(constructed, [])


class ExternalGearBackendBoundaryTests(unittest.TestCase):
    def test_paced_backend_leaves_wall_clock_sleep_to_whole_step_runner(self):
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._wall_clock_pacing = True
        backend._simulator = SimpleNamespace(
            sim_dt=0.005,
            sim_env=SimpleNamespace(viewer=None, sim_step=lambda: None),
        )

        with patch.object(gated_sim.time, "sleep") as sleep:
            backend.step()

        sleep.assert_not_called()

    def test_unpaced_backend_advances_without_wall_clock_sleep(self):
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._wall_clock_pacing = False
        backend._simulator = SimpleNamespace(
            sim_dt=0.005,
            sim_env=SimpleNamespace(sim_step=lambda: None),
        )

        with patch.object(gated_sim.time, "sleep") as sleep:
            backend.step()

        sleep.assert_not_called()

    def test_prime_low_state_clears_receipt_and_publishes_without_step(self):
        events = []
        obs = {"time": 0.0}
        qpos = np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])
        bridge = SimpleNamespace(
            reset=lambda: events.append("receipt-reset"),
            PublishLowState=lambda value: events.append(("publish", value)),
        )
        sim_env = SimpleNamespace(
            mj_data=SimpleNamespace(time=0.0, qpos=qpos.copy()),
            unitree_bridge=bridge,
            prepare_obs=lambda: events.append("prepare") or obs,
            sim_step=lambda: events.append("step"),
        )
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._simulator = SimpleNamespace(sim_dt=0.005, sim_env=sim_env)

        backend.prime_low_state()

        self.assertEqual(
            events,
            ["receipt-reset", "prepare", ("publish", obs)],
        )
        self.assertIs(sim_env.obs, obs)
        self.assertEqual(sim_env.mj_data.time, 0.0)
        np.testing.assert_array_equal(sim_env.mj_data.qpos, qpos)

    def test_refresh_low_state_publishes_without_reset_or_step(self):
        events = []
        obs = {"time": 0.4}
        bridge = SimpleNamespace(
            reset=lambda: events.append("receipt-reset"),
            PublishLowState=lambda value: events.append(("publish", value)),
        )
        sim_env = SimpleNamespace(
            mj_data=SimpleNamespace(time=0.4),
            unitree_bridge=bridge,
            prepare_obs=lambda: events.append("prepare") or obs,
            sim_step=lambda: events.append("step"),
        )
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._simulator = SimpleNamespace(sim_dt=0.005, sim_env=sim_env)

        backend.refresh_low_state()

        self.assertEqual(events, ["prepare", ("publish", obs)])
        self.assertIs(sim_env.obs, obs)
        self.assertEqual(sim_env.mj_data.time, 0.4)

    def test_low_command_snapshot_copies_exact_receiver_target(self):
        target = np.linspace(-0.5, 0.5, 29, dtype=np.float32)
        bridge = SimpleNamespace(
            low_cmd_lock=threading.Lock(),
            low_cmd_received=False,
            num_body_motor=29,
            low_cmd=SimpleNamespace(
                motor_cmd=[SimpleNamespace(q=float(value)) for value in target]
            ),
        )
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._simulator = SimpleNamespace(
            sim_env=SimpleNamespace(unitree_bridge=bridge)
        )

        self.assertEqual(
            backend.low_command_snapshot(),
            {"received": False, "q_target": None},
        )
        bridge.low_cmd_received = True
        received = backend.low_command_snapshot()
        bridge.low_cmd.motor_cmd[0].q = 99.0

        self.assertIs(received["received"], True)
        np.testing.assert_array_equal(received["q_target"], target)

    def test_contact_samples_retain_authoritative_geom_ids(self):
        class FakeMujoco:
            class mjtObj:
                mjOBJ_GEOM = 5

            @staticmethod
            def mj_contactForce(_model, _data, _index, force):
                force[:] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]

            @staticmethod
            def mj_id2name(_model, _kind, geom_id):
                return {7: "left_foot", 91: "floor"}[geom_id]

        pelvis = SimpleNamespace(
            xpos=np.array([0.0, 0.0, 0.8]),
            xquat=np.array([1.0, 0.0, 0.0, 0.0]),
            xmat=np.eye(3).reshape(-1),
        )
        data = SimpleNamespace(
            ncon=1,
            contact=[
                SimpleNamespace(
                    geom1=7,
                    geom2=91,
                    dist=-0.001,
                    pos=np.zeros(3),
                    frame=np.eye(3).reshape(-1),
                )
            ],
            qpos=np.zeros(36),
            qvel=np.zeros(35),
            qacc=np.zeros(35),
            ctrl=np.zeros(29),
            actuator_force=np.zeros(29),
            body=lambda name: pelvis if name == "pelvis" else None,
        )
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._bindings = SimpleNamespace(mujoco=FakeMujoco)
        backend._simulator = SimpleNamespace(
            sim_env=SimpleNamespace(mj_model=SimpleNamespace(), mj_data=data)
        )

        sample = backend.sample()

        self.assertEqual(sample["contacts"][0]["geom1_id"], 7)
        self.assertEqual(sample["contacts"][0]["geom2_id"], 91)
        self.assertEqual(sample["contacts"][0]["geom1"], "left_foot")
        self.assertEqual(sample["contacts"][0]["geom2"], "floor")

    def test_protocol_backend_forces_headless_nonrendering_simulator(self):
        captured = {}

        class ConfigLoader:
            env_name = "default"

            @staticmethod
            def load_wbc_yaml():
                return {
                    "ENABLE_ONSCREEN": True,
                    "ENABLE_OFFSCREEN": True,
                }

        def base_simulator(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

        bindings = SimpleNamespace(
            sim_loop_config=ConfigLoader,
            base_simulator=base_simulator,
        )
        with tempfile.TemporaryDirectory() as root_text:
            scene = Path(root_text) / "scene.xml"
            scene.write_text("<mujoco/>\n", encoding="utf-8")
            with patch.object(
                gated_sim,
                "load_external_bindings",
                return_value=bindings,
            ):
                ExternalGearBackend("/authenticated/gear", scene)

        self.assertEqual(captured["config"]["ROBOT_SCENE"], str(scene))
        self.assertFalse(captured["onscreen"])
        self.assertFalse(captured["offscreen"])
        self.assertFalse(captured["enable_image_publish"])

    def test_protocol_backend_forwards_explicit_onscreen_to_base_simulator(self):
        for onscreen in (False, True):
            with self.subTest(onscreen=onscreen):
                captured = {}

                class ConfigLoader:
                    env_name = "default"

                    @staticmethod
                    def load_wbc_yaml():
                        return {}

                viewer = SimpleNamespace(
                    opt=SimpleNamespace(flags=[0] * 31, geomgroup=[0] * 6),
                    is_running=lambda: True,
                )

                def base_simulator(**kwargs):
                    captured.update(kwargs)
                    return SimpleNamespace(sim_env=SimpleNamespace(viewer=viewer))

                bindings = SimpleNamespace(
                    sim_loop_config=ConfigLoader,
                    base_simulator=base_simulator,
                    mujoco=SimpleNamespace(
                        mjtVisFlag=SimpleNamespace(mjVIS_STATIC=22),
                    ),
                )
                with tempfile.TemporaryDirectory() as root_text:
                    scene = Path(root_text) / "scene.xml"
                    scene.write_text("<mujoco/>\n", encoding="utf-8")
                    with patch.object(
                        gated_sim,
                        "load_external_bindings",
                        return_value=bindings,
                    ):
                        ExternalGearBackend(
                            "/authenticated/gear", scene, onscreen=onscreen
                        )

                self.assertIs(captured["onscreen"], onscreen)
                self.assertFalse(captured["offscreen"])
                self.assertFalse(captured["enable_image_publish"])

    def test_protocol_backend_rejects_nonboolean_onscreen(self):
        with tempfile.TemporaryDirectory() as root_text:
            scene = Path(root_text) / "scene.xml"
            scene.write_text("<mujoco/>\n", encoding="utf-8")
            for value in (0, 1, None, "true"):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(
                        ProtocolError, "onscreen must be a boolean"
                    ):
                        ExternalGearBackend(
                            "/authenticated/gear", scene, onscreen=value
                        )

    def test_visible_backend_syncs_viewer_at_50hz_not_every_physics_step(self):
        events = []
        sim_env = SimpleNamespace(
            viewer=SimpleNamespace(is_running=lambda: True),
            sim_step=lambda: events.append("step"),
            update_viewer=lambda: events.append("sync"),
        )
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._wall_clock_pacing = False
        backend._simulator = SimpleNamespace(sim_dt=0.005, sim_env=sim_env)

        for _ in range(4):
            backend.step()

        self.assertEqual(events, ["step", "step", "step", "step", "sync"])

    def test_closed_visible_viewer_stops_before_physics_or_sync(self):
        events = []
        sim_env = SimpleNamespace(
            viewer=SimpleNamespace(is_running=lambda: False),
            sim_step=lambda: events.append("step"),
            update_viewer=lambda: events.append("sync"),
        )
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._wall_clock_pacing = False
        backend._simulator = SimpleNamespace(sim_dt=0.005, sim_env=sim_env)

        with self.assertRaisesRegex(ProtocolError, "viewer is closed"):
            backend.step()

        self.assertEqual(events, [])

    def test_headless_backend_steps_once_without_viewer_sync(self):
        events = []
        sim_env = SimpleNamespace(
            viewer=None,
            sim_step=lambda: events.append("step"),
            update_viewer=lambda: events.append("sync"),
        )
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._wall_clock_pacing = False
        backend._simulator = SimpleNamespace(sim_dt=0.005, sim_env=sim_env)

        backend.step()

        self.assertEqual(events, ["step"])

    def test_set_camera_assigns_viewer_fields_and_syncs_once(self):
        events = []
        cam = SimpleNamespace(azimuth=0.0, elevation=0.0, distance=0.0)
        viewer = SimpleNamespace(
            cam=cam,
            is_running=lambda: True,
            sync=lambda: events.append("sync"),
        )
        sim_env = SimpleNamespace(viewer=viewer)
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._simulator = SimpleNamespace(sim_env=sim_env)

        backend.set_camera(45.0, -20.0, 4.0)

        self.assertEqual(cam.azimuth, 45.0)
        self.assertEqual(cam.elevation, -20.0)
        self.assertEqual(cam.distance, 4.0)
        self.assertEqual(events, ["sync"])

    def test_set_camera_requires_a_running_passive_viewer(self):
        cam = SimpleNamespace(azimuth=0.0, elevation=0.0, distance=0.0)
        for viewer in (None, SimpleNamespace(cam=cam, is_running=lambda: False)):
            with self.subTest(viewer=viewer):
                sim_env = SimpleNamespace(viewer=viewer)
                backend = ExternalGearBackend.__new__(ExternalGearBackend)
                backend._simulator = SimpleNamespace(sim_env=sim_env)
                with self.assertRaisesRegex(ProtocolError, "viewer"):
                    backend.set_camera(45.0, -20.0, 4.0)

    def test_freeze_on_fall_must_be_a_boolean(self):
        with tempfile.TemporaryDirectory() as root_text:
            scene = Path(root_text) / "scene.xml"
            scene.write_text("<mujoco/>\n", encoding="utf-8")
            for value in (0, 1, None, "true"):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(
                        ProtocolError, "freeze_on_fall must be a boolean"
                    ):
                        ExternalGearBackend(
                            "/authenticated/gear",
                            scene,
                            onscreen=True,
                            freeze_on_fall=value,
                        )

    def test_freeze_on_fall_requires_onscreen(self):
        with tempfile.TemporaryDirectory() as root_text:
            scene = Path(root_text) / "scene.xml"
            scene.write_text("<mujoco/>\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ProtocolError, "freeze_on_fall requires onscreen"
            ):
                ExternalGearBackend(
                    "/authenticated/gear",
                    scene,
                    onscreen=False,
                    freeze_on_fall=True,
                )

    def _freeze_bindings(self, sim_env):
        class ConfigLoader:
            env_name = "default"

            @staticmethod
            def load_wbc_yaml():
                return {}

        def base_simulator(**_kwargs):
            return SimpleNamespace(sim_dt=0.005, sim_env=sim_env)

        def mj_forward(model, data):
            events = getattr(sim_env, "forward_events", None)
            if events is not None:
                events.append((model, data))
            # Model the real API recomputing acceleration so the adapter must
            # explicitly restore the frozen acceleration contract afterward.
            data.qacc[:] = 7.0

        return SimpleNamespace(
            sim_loop_config=ConfigLoader,
            base_simulator=base_simulator,
            mujoco=SimpleNamespace(
                mj_forward=mj_forward,
                mjtVisFlag=SimpleNamespace(mjVIS_STATIC=22),
            ),
        )

    @staticmethod
    def _attach_running_viewer(sim_env):
        """Give an onscreen fixture the running viewer startup normalization needs."""

        sim_env.viewer = SimpleNamespace(
            opt=SimpleNamespace(flags=[0] * 31, geomgroup=[0] * 6),
            is_running=lambda: True,
        )
        return sim_env

    def _construct_freeze_backend(self, sim_env, *, freeze_on_fall):
        bindings = self._freeze_bindings(sim_env)
        with tempfile.TemporaryDirectory() as root_text:
            scene = Path(root_text) / "scene.xml"
            scene.write_text("<mujoco/>\n", encoding="utf-8")
            with (
                patch.object(
                    gated_sim, "load_external_bindings", return_value=bindings
                ),
                redirect_stdout(io.StringIO()),
            ):
                return ExternalGearBackend(
                    "/authenticated/gear",
                    scene,
                    onscreen=True,
                    freeze_on_fall=freeze_on_fall,
                )

    def test_default_backend_does_not_replace_official_fall_callback(self):
        original = lambda: None
        sim_env = self._attach_running_viewer(
            SimpleNamespace(check_fall=original)
        )

        backend = self._construct_freeze_backend(sim_env, freeze_on_fall=False)

        self.assertIs(backend._simulator.sim_env.check_fall, original)

    def test_freeze_latches_first_fall_and_suppresses_official_reset(self):
        reset_calls = []
        mj_data = SimpleNamespace(
            qpos=np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]),
            qvel=np.ones(6),
            qacc=np.ones(6),
            ctrl=np.ones(3),
            time=0.0,
        )

        class SimEnv:
            def reset(self_inner):
                reset_calls.append(True)

            def check_fall(self_inner):
                if self_inner.mj_data.qpos[2] < 0.2:
                    self_inner.reset()

            def sim_step(self_inner):
                self_inner.check_fall()

        sim_env = SimEnv()
        sim_env.mj_data = mj_data
        sim_env.mj_model = object()
        sim_env.forward_events = []
        self._attach_running_viewer(sim_env)
        fallen_qpos = mj_data.qpos.copy()
        stderr = io.StringIO()

        backend = self._construct_freeze_backend(sim_env, freeze_on_fall=True)
        with redirect_stderr(stderr):
            sim_env.sim_step()

        self.assertEqual(reset_calls, [])
        self.assertTrue(backend._frozen)
        self.assertTrue(np.array_equal(mj_data.qpos, fallen_qpos))
        self.assertTrue(np.array_equal(mj_data.qvel, np.zeros(6)))
        self.assertTrue(np.array_equal(mj_data.qacc, np.zeros(6)))
        self.assertTrue(np.array_equal(mj_data.ctrl, np.zeros(3)))
        self.assertEqual(
            sim_env.forward_events,
            [(sim_env.mj_model, sim_env.mj_data)],
        )
        self.assertNotEqual(stderr.getvalue(), "")

    def test_freeze_logs_once_and_does_not_relatch(self):
        mj_data = SimpleNamespace(
            qpos=np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]),
            qvel=np.ones(6),
            qacc=np.ones(6),
            ctrl=np.ones(3),
            time=0.0,
        )

        class SimEnv:
            def reset(self_inner):
                raise AssertionError("official reset must not run")

            def check_fall(self_inner):
                if self_inner.mj_data.qpos[2] < 0.2:
                    self_inner.reset()

        sim_env = SimEnv()
        sim_env.mj_data = mj_data
        sim_env.mj_model = object()
        self._attach_running_viewer(sim_env)

        backend = self._construct_freeze_backend(sim_env, freeze_on_fall=True)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            sim_env.check_fall()
            first = stderr.getvalue()
            sim_env.check_fall()
            second = stderr.getvalue()

        self.assertEqual(first, second)
        self.assertTrue(backend._frozen)

    def _frozen_sim_env(self, *, viewer_running=True, joystick=True):
        events = []
        obs_token = object()
        bridge = SimpleNamespace(
            joystick=joystick,
            PublishLowState=lambda obs: events.append(("low", obs)),
            PublishWirelessController=lambda: events.append(("wireless",)),
        )
        sim_env = SimpleNamespace(
            viewer=SimpleNamespace(is_running=lambda: viewer_running),
            mj_data=SimpleNamespace(time=1.0),
            unitree_bridge=bridge,
            prepare_obs=lambda: events.append(("prepare",)) or obs_token,
            sim_step=lambda: events.append(("step",)),
            update_viewer=lambda: events.append(("sync",)),
        )
        return sim_env, events, obs_token

    def test_frozen_step_republishes_lowstate_and_syncs_at_50hz(self):
        sim_env, events, obs_token = self._frozen_sim_env()
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._wall_clock_pacing = False
        backend._freeze_on_fall = True
        backend._frozen = True
        backend._simulator = SimpleNamespace(sim_dt=0.005, sim_env=sim_env)

        for _ in range(4):
            backend.step()

        self.assertNotIn(("step",), events)
        self.assertEqual(
            events,
            [
                ("prepare",),
                ("low", obs_token),
                ("wireless",),
                ("prepare",),
                ("low", obs_token),
                ("wireless",),
                ("prepare",),
                ("low", obs_token),
                ("wireless",),
                ("prepare",),
                ("low", obs_token),
                ("wireless",),
                ("sync",),
            ],
        )
        self.assertIs(sim_env.obs, obs_token)
        self.assertAlmostEqual(sim_env.mj_data.time, 1.02)

    def test_frozen_step_skips_wireless_when_joystick_is_falsey(self):
        sim_env, events, obs = self._frozen_sim_env(joystick=None)
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._wall_clock_pacing = False
        backend._freeze_on_fall = True
        backend._frozen = True
        backend._simulator = SimpleNamespace(sim_dt=0.005, sim_env=sim_env)

        backend.step()

        self.assertIn(("low", obs), events)
        self.assertNotIn(("wireless",), events)
        self.assertNotIn(("step",), events)

    def test_frozen_step_rejects_a_closed_viewer_before_publishing(self):
        sim_env, events, _obs = self._frozen_sim_env(viewer_running=False)
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._wall_clock_pacing = False
        backend._freeze_on_fall = True
        backend._frozen = True
        backend._simulator = SimpleNamespace(sim_dt=0.005, sim_env=sim_env)

        with self.assertRaisesRegex(ProtocolError, "viewer is closed"):
            backend.step()

        self.assertEqual(events, [])
        self.assertEqual(sim_env.mj_data.time, 1.0)

    def test_freeze_on_fall_flag_defaults_off_and_parses_opt_in(self):
        parser = gated_sim._parser()
        self.assertFalse(
            parser.parse_args(["--gear-checkout", "/gear"]).freeze_on_fall
        )
        self.assertTrue(
            parser.parse_args(
                ["--gear-checkout", "/gear", "--onscreen", "--freeze-on-fall"]
            ).freeze_on_fall
        )

    def test_production_main_forwards_freeze_on_fall_through_backend_factory(self):
        for argv_extra, expected in (
            (["--onscreen"], False),
            (["--onscreen", "--freeze-on-fall"], True),
        ):
            with self.subTest(expected=expected):
                captured = {}

                def fake_backend(
                    checkout, scene, *, wall_clock_pacing, onscreen, freeze_on_fall
                ):
                    captured["freeze_on_fall"] = freeze_on_fall
                    return SimpleNamespace()

                def fake_server(**kwargs):
                    kwargs["backend_factory"](Path("/gear/scene.xml"))

                with (
                    patch.object(
                        gated_sim, "_verified_checkout", return_value=Path("/gear")
                    ),
                    patch.object(
                        gated_sim, "serve_jsonl", side_effect=fake_server
                    ),
                    patch.object(gated_sim, "ExternalGearBackend", fake_backend),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                ):
                    result = gated_sim.main(
                        [
                            "--gear-checkout",
                            "/gear",
                            "--run-root",
                            str(Path.cwd()),
                            *argv_extra,
                        ]
                    )

                self.assertEqual(result, 0)
                self.assertIs(captured["freeze_on_fall"], expected)

    def test_onscreen_flag_defaults_headless_and_parses_opt_in(self):
        parser = gated_sim._parser()
        self.assertFalse(parser.parse_args(["--gear-checkout", "/gear"]).onscreen)
        self.assertTrue(
            parser.parse_args(
                ["--gear-checkout", "/gear", "--onscreen"]
            ).onscreen
        )

    def test_production_main_forwards_onscreen_through_backend_factory(self):
        for argv_extra, expected in (([], False), (["--onscreen"], True)):
            with self.subTest(expected=expected):
                captured = {}

                def fake_backend(
                    checkout, scene, *, wall_clock_pacing, onscreen,
                    freeze_on_fall,
                ):
                    captured["onscreen"] = onscreen
                    captured["wall_clock_pacing"] = wall_clock_pacing
                    return SimpleNamespace()

                def fake_server(**kwargs):
                    kwargs["backend_factory"](Path("/gear/scene.xml"))

                with (
                    patch.object(
                        gated_sim, "_verified_checkout", return_value=Path("/gear")
                    ),
                    patch.object(
                        gated_sim, "serve_jsonl", side_effect=fake_server
                    ),
                    patch.object(gated_sim, "ExternalGearBackend", fake_backend),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                ):
                    result = gated_sim.main(
                        [
                            "--gear-checkout",
                            "/gear",
                            "--run-root",
                            str(Path.cwd()),
                            *argv_extra,
                        ]
                    )

                self.assertEqual(result, 0)
                self.assertIs(captured["onscreen"], expected)
                self.assertTrue(captured["wall_clock_pacing"])

    def test_production_main_reserves_stdout_for_jsonl_protocol(self):
        protocol_line = '{"v":1,"ok":true}\n'

        def fake_server(**kwargs):
            print("background DDS diagnostic")
            kwargs.get("output_stream", sys.stdout).write(protocol_line)

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(gated_sim, "_verified_checkout", return_value=Path("/gear")),
            patch.object(gated_sim, "serve_jsonl", side_effect=fake_server),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            result = gated_sim.main(
                [
                    "--gear-checkout",
                    "/gear",
                    "--run-root",
                    str(Path.cwd()),
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue(), protocol_line)
        self.assertIn("background DDS diagnostic", stderr.getvalue())

    def test_official_step_and_close_diagnostics_cannot_corrupt_jsonl_stdout(self):
        class PrintingEnvironment:
            def sim_step(self):
                print("official step diagnostic")

        class PrintingSimulator:
            sim_dt = 0.0
            sim_env = PrintingEnvironment()

            def close(self):
                print("official close diagnostic")

        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._simulator = PrintingSimulator()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            backend.step()
            backend.close()

        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("official step diagnostic", stderr.getvalue())
        self.assertIn("official close diagnostic", stderr.getvalue())

    def test_external_reset_uses_independent_physical_qpos_oracle(self):
        class FakeMujoco:
            def __init__(self):
                self.reset_calls = 0
                self.forward_calls = 0

            def mj_resetData(self, _model, data):
                self.reset_calls += 1
                data.qpos[:] = -99.0
                data.qvel[:] = -99.0
                data.ctrl[:] = -99.0

            def mj_forward(self, _model, _data):
                self.forward_calls += 1

        model = SimpleNamespace(nq=36)
        data = SimpleNamespace(
            qpos=np.zeros(36, dtype=np.float64),
            qvel=np.ones(35, dtype=np.float64),
            ctrl=np.ones(29, dtype=np.float64),
        )
        mujoco = FakeMujoco()
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._bindings = SimpleNamespace(mujoco=mujoco)
        backend._simulator = SimpleNamespace(
            sim_env=SimpleNamespace(
                mj_model=model,
                mj_data=data,
                elastic_band=SimpleNamespace(
                    enable=False,
                    point=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                    kp_ang=1000.0,
                ),
            )
        )
        initial = np.zeros(36, dtype=np.float64)
        initial[:7] = [1.2, -0.4, 0.81, 1.0, 0.0, 0.0, 0.0]
        initial[7:] = np.linspace(-0.7, 0.7, 29)
        unchanged = initial.copy()

        backend.reset_from_qpos(
            initial,
            lateral_offset_m=0.09,
            yaw_offset_rad=math.pi / 2.0,
            elastic_band_enabled=True,
        )

        np.testing.assert_array_equal(initial, unchanged)
        np.testing.assert_allclose(data.qpos[:3], [1.2, -0.31, 0.81], atol=1e-15)
        np.testing.assert_allclose(
            data.qpos[3:7],
            [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
            atol=1e-15,
        )
        np.testing.assert_array_equal(data.qpos[7:], unchanged[7:])
        np.testing.assert_array_equal(data.qvel, np.zeros(35))
        np.testing.assert_array_equal(data.ctrl, np.zeros(29))
        self.assertEqual(mujoco.reset_calls, 1)
        self.assertEqual(mujoco.forward_calls, 1)

    def _band_backend(self, band):
        events = []

        class FakeMujoco:
            def mj_resetData(self, _model, _data):
                events.append(("reset", getattr(band, "enable", "<missing>")))

            def mj_forward(self, _model, _data):
                events.append("forward")

        model = SimpleNamespace(nq=36)
        data = SimpleNamespace(
            qpos=np.zeros(36, dtype=np.float64),
            qvel=np.zeros(35, dtype=np.float64),
            ctrl=np.zeros(29, dtype=np.float64),
        )
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._bindings = SimpleNamespace(mujoco=FakeMujoco())
        sim_env = SimpleNamespace(mj_model=model, mj_data=data)
        if band is not None:
            sim_env.elastic_band = band
        backend._simulator = SimpleNamespace(sim_env=sim_env)
        return backend, events

    def test_external_reset_applies_band_state_before_mujoco_reset(self):
        band = SimpleNamespace(
            enable=True,
            point=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            kp_ang=1000.0,
        )
        backend, events = self._band_backend(band)
        qpos = np.r_[np.zeros(3), 1.0, np.zeros(32)]

        backend.reset_from_qpos(
            qpos, 0.0, 0.0, elastic_band_enabled=False
        )
        self.assertFalse(band.enable)
        self.assertEqual(events[0], ("reset", False))

        backend.reset_from_qpos(
            qpos, 0.0, 0.0, elastic_band_enabled=True
        )
        self.assertTrue(band.enable)
        self.assertEqual(events[2], ("reset", True))

    def test_external_reset_centers_enabled_band_on_physical_spawn_xy(self):
        band = SimpleNamespace(
            enable=False,
            point=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            kp_ang=1000.0,
        )
        backend, _events = self._band_backend(band)
        qpos = np.r_[
            np.array([1.2, -0.4, 0.81, 1.0, 0.0, 0.0, 0.0]),
            np.zeros(29),
        ]

        backend.reset_from_qpos(
            qpos,
            lateral_offset_m=0.09,
            yaw_offset_rad=0.0,
            elastic_band_enabled=True,
        )

        np.testing.assert_allclose(
            band.point,
            [1.2, -0.31, 1.0],
            rtol=0.0,
            atol=1.0e-15,
        )
        self.assertEqual(band.kp_ang, 0.0)

    def test_external_reset_rejects_missing_band_or_nonboolean_state(self):
        qpos = np.r_[np.zeros(3), 1.0, np.zeros(32)]
        backend, _events = self._band_backend(None)
        with self.assertRaisesRegex(ProtocolError, "elastic_band"):
            backend.reset_from_qpos(qpos, 0.0, 0.0, elastic_band_enabled=True)

        backend, _events = self._band_backend(SimpleNamespace(enable=False))
        with self.assertRaisesRegex(ProtocolError, "elastic_band_enabled"):
            backend.reset_from_qpos(qpos, 0.0, 0.0, elastic_band_enabled=1)


def _external_module_names():
    return tuple(
        name
        for name in sys.modules
        if name in ("gear_sonic", "unitree_sdk2py")
        or name.startswith("gear_sonic.")
        or name.startswith("unitree_sdk2py.")
    )


class OnscreenViewerNormalizationTests(unittest.TestCase):
    @staticmethod
    def _bindings():
        return SimpleNamespace(
            mujoco=SimpleNamespace(
                mjtVisFlag=SimpleNamespace(mjVIS_STATIC=22),
            )
        )

    @staticmethod
    def _running_sim_env():
        viewer = SimpleNamespace(
            opt=SimpleNamespace(flags=[0] * 31, geomgroup=[0] * 6),
            is_running=lambda: True,
        )
        return SimpleNamespace(viewer=viewer)

    def test_onscreen_backend_enables_static_terrain_presentation(self) -> None:
        sim_env = self._running_sim_env()
        _normalize_onscreen_viewer(sim_env, self._bindings().mujoco)
        self.assertEqual(sim_env.viewer.opt.flags[22], 1)
        self.assertEqual(sim_env.viewer.opt.geomgroup[2], 1)

    def test_normalization_does_not_step_or_sync(self) -> None:
        events: list[str] = []
        viewer = SimpleNamespace(
            opt=SimpleNamespace(flags=[0] * 31, geomgroup=[0] * 6),
            is_running=lambda: True,
        )
        sim_env = SimpleNamespace(
            viewer=viewer,
            sim_step=lambda: events.append("step"),
            update_viewer=lambda: events.append("sync"),
        )
        _normalize_onscreen_viewer(sim_env, self._bindings().mujoco)
        self.assertEqual(events, [])

    def test_rejects_missing_viewer(self) -> None:
        sim_env = SimpleNamespace(viewer=None)
        with self.assertRaisesRegex(ProtocolError, "viewer"):
            _normalize_onscreen_viewer(sim_env, self._bindings().mujoco)

    def test_rejects_closed_viewer(self) -> None:
        viewer = SimpleNamespace(
            opt=SimpleNamespace(flags=[0] * 31, geomgroup=[0] * 6),
            is_running=lambda: False,
        )
        sim_env = SimpleNamespace(viewer=viewer)
        with self.assertRaisesRegex(ProtocolError, "viewer"):
            _normalize_onscreen_viewer(sim_env, self._bindings().mujoco)

    def test_rejects_malformed_option_arrays(self) -> None:
        flags = [0] * 31
        viewer = SimpleNamespace(
            opt=SimpleNamespace(flags=flags, geomgroup=[0]),
            is_running=lambda: True,
        )
        sim_env = SimpleNamespace(viewer=viewer)
        with self.assertRaisesRegex(ProtocolError, "viewer presentation"):
            _normalize_onscreen_viewer(sim_env, self._bindings().mujoco)
        self.assertEqual(flags[22], 0)

    def test_rejects_missing_static_flag_identity_without_partial_mutation(
        self,
    ) -> None:
        sim_env = self._running_sim_env()
        malformed_mujoco = SimpleNamespace(mjtVisFlag=SimpleNamespace())

        with self.assertRaisesRegex(ProtocolError, "viewer presentation"):
            _normalize_onscreen_viewer(sim_env, malformed_mujoco)

        self.assertEqual(sim_env.viewer.opt.flags, [0] * 31)
        self.assertEqual(sim_env.viewer.opt.geomgroup, [0] * 6)

    def test_constructor_normalizes_once_only_when_onscreen(self) -> None:
        for onscreen in (False, True):
            with self.subTest(onscreen=onscreen):
                calls: list[object] = []

                class ConfigLoader:
                    env_name = "default"

                    @staticmethod
                    def load_wbc_yaml():
                        return {}

                sim_env = self._running_sim_env()
                simulator = SimpleNamespace(sim_env=sim_env)

                def base_simulator(**kwargs):
                    return simulator

                bindings = SimpleNamespace(
                    sim_loop_config=ConfigLoader,
                    base_simulator=base_simulator,
                    mujoco=self._bindings().mujoco,
                )
                with tempfile.TemporaryDirectory() as root_text:
                    scene = Path(root_text) / "scene.xml"
                    scene.write_text("<mujoco/>\n", encoding="utf-8")
                    with patch.object(
                        gated_sim,
                        "load_external_bindings",
                        return_value=bindings,
                    ), patch.object(
                        gated_sim,
                        "_normalize_onscreen_viewer",
                        side_effect=lambda *a: calls.append(a),
                    ):
                        ExternalGearBackend(
                            "/authenticated/gear", scene, onscreen=onscreen
                        )
                if onscreen:
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0], (sim_env, bindings.mujoco))
                else:
                    self.assertEqual(calls, [])

    def test_constructor_closes_simulator_when_normalization_fails(self) -> None:
        close_calls: list[bool] = []

        class ConfigLoader:
            env_name = "default"

            @staticmethod
            def load_wbc_yaml():
                return {}

        simulator = SimpleNamespace(
            sim_env=self._running_sim_env(),
            close=lambda: close_calls.append(True),
        )
        bindings = SimpleNamespace(
            sim_loop_config=ConfigLoader,
            base_simulator=lambda **kwargs: simulator,
            mujoco=self._bindings().mujoco,
        )
        with tempfile.TemporaryDirectory() as root_text:
            scene = Path(root_text) / "scene.xml"
            scene.write_text("<mujoco/>\n", encoding="utf-8")
            with patch.object(
                gated_sim,
                "load_external_bindings",
                return_value=bindings,
            ), patch.object(
                gated_sim,
                "_normalize_onscreen_viewer",
                side_effect=ProtocolError("viewer presentation failed"),
            ):
                with self.assertRaisesRegex(ProtocolError, "viewer presentation"):
                    ExternalGearBackend(
                        "/authenticated/gear", scene, onscreen=True
                    )

        self.assertEqual(close_calls, [True])


class ExternalCheckoutProvenanceTests(unittest.TestCase):
    def _seed_gear(self, checkout):
        package = checkout / "gear_sonic" / "utils" / "mujoco_sim"
        package.mkdir(parents=True)
        for init in (
            checkout / "gear_sonic" / "__init__.py",
            checkout / "gear_sonic" / "utils" / "__init__.py",
            package / "__init__.py",
        ):
            init.write_text("", encoding="utf-8")
        (package / "base_sim.py").write_text(
            "import unitree_sdk2py\n"
            "import unitree_sdk2py.b2\n"
            "import unitree_sdk2py.core.channel as channel\n"
            "unitree_sdk2py.GEAR_IMPORT_OBSERVED = True\n"
            "channel.GEAR_IMPORT_OBSERVED = True\n"
            "class BaseSimulator:\n    pass\n",
            encoding="utf-8",
        )
        (package / "configs.py").write_text(
            "class SimLoopConfig:\n    pass\n",
            encoding="utf-8",
        )

    def _seed_unitree(self, source_dir):
        package = source_dir / "unitree_sdk2py"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            "VERSION = 'synthetic'\n", encoding="utf-8"
        )
        core = package / "core"
        core.mkdir()
        (core / "__init__.py").write_text("", encoding="utf-8")
        (core / "channel.py").write_text(
            "VERSION = 'synthetic-channel'\n", encoding="utf-8"
        )
        # Match the pinned SDK's implicit namespace package layout.
        (package / "b2").mkdir()
        return package

    def _commit(self, checkout):
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "config",
                "user.email",
                "test@example.com",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "config", "user.name", "Test"],
            check=True,
        )
        subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "commit", "-qm", "synthetic"],
            check=True,
        )
        return subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
        ).strip()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkout = self.root / "gear"
        self.checkout.mkdir()
        self._seed_gear(self.checkout)
        self._seed_unitree(
            self.checkout / "external_dependencies" / "unitree_sdk2_python"
        )
        self.commit = self._commit(self.checkout)
        self.saved_modules = {
            name: sys.modules[name] for name in _external_module_names()
        }
        for name in self.saved_modules:
            sys.modules.pop(name, None)

    def tearDown(self):
        for name in _external_module_names():
            sys.modules.pop(name, None)
        sys.modules.update(self.saved_modules)
        for entry in tuple(sys.path):
            if entry.startswith(str(self.root)):
                sys.path.remove(entry)
        self.temporary.cleanup()

    def test_checkout_must_be_clean_including_untracked_files(self):
        with patch.object(gated_sim, "PINNED_GEAR_COMMIT", self.commit):
            (self.checkout / "untracked.py").write_text("shadow = True\n")
            with self.assertRaisesRegex(ProtocolError, "clean"):
                gated_sim._verified_checkout(self.checkout)
            (self.checkout / "untracked.py").unlink()
            tracked = self.checkout / "gear_sonic" / "__init__.py"
            tracked.write_text("edited = True\n", encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "clean"):
                gated_sim._verified_checkout(self.checkout)

    def test_preloaded_external_module_origin_must_be_inside_checkout(self):
        with patch.object(gated_sim, "PINNED_GEAR_COMMIT", self.commit):
            bindings = gated_sim.load_external_bindings(self.checkout)
            self.assertEqual(bindings.checkout, self.checkout.resolve())

            module_name = "gear_sonic.utils.mujoco_sim.base_sim"
            shadow = types.ModuleType(module_name)
            shadow.__file__ = str(self.root / "shadow_base_sim.py")
            shadow.BaseSimulator = type("BaseSimulator", (), {})
            shadow.BaseSimulator.__module__ = module_name
            sys.modules[module_name] = shadow
            with self.assertRaisesRegex(ProtocolError, "origin"):
                gated_sim.load_external_bindings(self.checkout)

    def test_unitree_source_is_resolved_from_pinned_checkout_without_caller_path(self):
        unitree_source = str(
            self.checkout / "external_dependencies" / "unitree_sdk2_python"
        )
        # The isolated caller supplies no Unitree Python path of its own.
        self.assertNotIn(unitree_source, sys.path)
        self.assertNotIn("unitree_sdk2py", sys.modules)

        with patch.object(gated_sim, "PINNED_GEAR_COMMIT", self.commit):
            bindings = gated_sim.load_external_bindings(self.checkout)

        unitree = sys.modules["unitree_sdk2py"]
        origin = Path(unitree.__file__).resolve()
        expected = (
            self.checkout
            / "external_dependencies"
            / "unitree_sdk2_python"
            / "unitree_sdk2py"
            / "__init__.py"
        ).resolve()
        self.assertEqual(origin, expected)
        self.assertEqual(bindings.checkout, self.checkout.resolve())

    def test_missing_unitree_source_directory_is_rejected(self):
        checkout = self.root / "gear_no_unitree"
        checkout.mkdir()
        self._seed_gear(checkout)
        commit = self._commit(checkout)
        with patch.object(gated_sim, "PINNED_GEAR_COMMIT", commit):
            with self.assertRaisesRegex(ProtocolError, "unitree"):
                gated_sim.load_external_bindings(checkout)

    def test_escaping_symlink_unitree_source_is_rejected(self):
        checkout = self.root / "gear_symlinked_unitree"
        checkout.mkdir()
        self._seed_gear(checkout)
        outside = self.root / "outside_unitree"
        self._seed_unitree(outside)
        source = checkout / "external_dependencies" / "unitree_sdk2_python"
        source.parent.mkdir(parents=True)
        # Commit an escaping symlink; validation must refuse to follow it.
        source.symlink_to(outside, target_is_directory=True)
        commit = self._commit(checkout)
        with patch.object(gated_sim, "PINNED_GEAR_COMMIT", commit):
            with self.assertRaisesRegex(ProtocolError, "symlink"):
                gated_sim.load_external_bindings(checkout)

    def test_wrong_origin_unitree_module_is_rejected(self):
        with patch.object(gated_sim, "PINNED_GEAR_COMMIT", self.commit):
            outside = self.root / "shadow_unitree"
            outside.mkdir()
            shadow_file = outside / "__init__.py"
            shadow_file.write_text("", encoding="utf-8")
            shadow = types.ModuleType("unitree_sdk2py")
            shadow.__file__ = str(shadow_file)
            sys.modules["unitree_sdk2py"] = shadow
            with self.assertRaisesRegex(ProtocolError, "origin"):
                gated_sim.load_external_bindings(self.checkout)
            self.assertFalse(hasattr(shadow, "GEAR_IMPORT_OBSERVED"))

    def test_wrong_origin_cached_unitree_child_is_rejected_before_gear_import(self):
        unitree_source = str(
            self.checkout / "external_dependencies" / "unitree_sdk2_python"
        )
        sys.path.insert(0, unitree_source)
        try:
            __import__("unitree_sdk2py")
        finally:
            sys.path.remove(unitree_source)
        pinned_root = sys.modules["unitree_sdk2py"]

        outside = self.root / "shadow_unitree" / "core"
        outside.mkdir(parents=True)
        core_file = outside / "__init__.py"
        channel_file = outside / "channel.py"
        core_file.write_text("", encoding="utf-8")
        channel_file.write_text("", encoding="utf-8")
        shadow_core = types.ModuleType("unitree_sdk2py.core")
        shadow_core.__file__ = str(core_file)
        shadow_core.__path__ = [str(outside)]
        shadow_channel = types.ModuleType("unitree_sdk2py.core.channel")
        shadow_channel.__file__ = str(channel_file)
        sys.modules["unitree_sdk2py.core"] = shadow_core
        sys.modules["unitree_sdk2py.core.channel"] = shadow_channel
        pinned_root.core = shadow_core
        shadow_core.channel = shadow_channel

        with patch.object(gated_sim, "PINNED_GEAR_COMMIT", self.commit):
            with self.assertRaisesRegex(ProtocolError, "origin"):
                gated_sim.load_external_bindings(self.checkout)
        self.assertFalse(hasattr(shadow_channel, "GEAR_IMPORT_OBSERVED"))


if __name__ == "__main__":
    unittest.main()
