import io
import json
import math
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
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
    physical_qpos_with_perturbation,
    serve_jsonl,
)


class FakeBackend:
    def __init__(self, *, nq=36, sim_dt=0.005):
        self._model = SimpleNamespace(nq=nq)
        self._data = SimpleNamespace(
            time=0.0,
            qpos=np.zeros(nq, dtype=np.float64),
        )
        self._sim_dt = sim_dt
        self.reset_calls = []
        self.step_calls = 0
        self.sample_calls = 0
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

    def test_visible_backend_syncs_viewer_once_after_single_step(self):
        events = []
        sim_env = SimpleNamespace(
            viewer=SimpleNamespace(is_running=lambda: True),
            sim_step=lambda: events.append("step"),
            update_viewer=lambda: events.append("sync"),
        )
        backend = ExternalGearBackend.__new__(ExternalGearBackend)
        backend._wall_clock_pacing = False
        backend._simulator = SimpleNamespace(sim_dt=0.005, sim_env=sim_env)

        backend.step()

        self.assertEqual(events, ["step", "sync"])

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

                def fake_backend(checkout, scene, *, wall_clock_pacing, onscreen):
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
                elastic_band=SimpleNamespace(enable=False),
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
        band = SimpleNamespace(enable=True)
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
