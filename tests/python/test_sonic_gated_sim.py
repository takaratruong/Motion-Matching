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

    def reset_from_qpos(self, qpos, lateral_offset_m, yaw_offset_rad):
        self.reset_calls.append(
            (qpos.copy(), lateral_offset_m, yaw_offset_rad)
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

    def reset(self, qpos=None, *, lateral=0.0, yaw=0.0):
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
        )

    def test_reset_is_idle_and_requires_exact_nq_values(self):
        with self.assertRaisesRegex(ProtocolError, "exactly 36"):
            self.reset(np.array([0.0]))
        self.assertEqual(self.backends[0].step_calls, 0)

        result = self.reset()
        self.assertEqual(result["nq"], 36)
        self.assertEqual(result["sim_dt_s"], 0.005)
        self.assertEqual(self.backends[-1].step_calls, 0)
        self.assertEqual(self.backends[-1].sample_calls, 0)

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
                )
            with self.assertRaisesRegex(ProtocolError, "absolute"):
                self.runner.reset(
                    scene_xml=self.scene,
                    initial_qpos=qpos,
                    lateral_offset_m=0.0,
                    yaw_offset_rad=0.0,
                    log_dir=Path("relative-log"),
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
            sim_env=SimpleNamespace(mj_model=model, mj_data=data)
        )
        initial = np.zeros(36, dtype=np.float64)
        initial[:7] = [1.2, -0.4, 0.81, 1.0, 0.0, 0.0, 0.0]
        initial[7:] = np.linspace(-0.7, 0.7, 29)
        unchanged = initial.copy()

        backend.reset_from_qpos(
            initial,
            lateral_offset_m=0.09,
            yaw_offset_rad=math.pi / 2.0,
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


class ExternalCheckoutProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkout = self.root / "gear"
        package = self.checkout / "gear_sonic" / "utils" / "mujoco_sim"
        package.mkdir(parents=True)
        for init in (
            self.checkout / "gear_sonic" / "__init__.py",
            self.checkout / "gear_sonic" / "utils" / "__init__.py",
            package / "__init__.py",
        ):
            init.write_text("", encoding="utf-8")
        (package / "base_sim.py").write_text(
            "class BaseSimulator:\n    pass\n",
            encoding="utf-8",
        )
        (package / "configs.py").write_text(
            "class SimLoopConfig:\n    pass\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q", str(self.checkout)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.checkout),
                "config",
                "user.email",
                "test@example.com",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.checkout), "config", "user.name", "Test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.checkout), "add", "."],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.checkout), "commit", "-qm", "synthetic"],
            check=True,
        )
        self.commit = subprocess.check_output(
            ["git", "-C", str(self.checkout), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        self.saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "gear_sonic" or name.startswith("gear_sonic.")
        }
        for name in self.saved_modules:
            sys.modules.pop(name, None)

    def tearDown(self):
        for name in tuple(sys.modules):
            if name == "gear_sonic" or name.startswith("gear_sonic."):
                sys.modules.pop(name, None)
        sys.modules.update(self.saved_modules)
        while str(self.checkout) in sys.path:
            sys.path.remove(str(self.checkout))
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


if __name__ == "__main__":
    unittest.main()
