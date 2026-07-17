import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from mm_sonic.commands import CommandSample
from mm_sonic.coordinator import SessionConfig
from mm_sonic.process import (
    AdvanceResult,
    ChildProcessDied,
    GatedSimulatorClient,
    GearProcess,
    MMChunkClient,
    OperatorCancelled,
    ProcessError,
    ProcessProtocolError,
    SimulationPolicyGate,
    _RemoteMMError,
)


def process_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class TemporaryScriptCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def script(self, name, source):
        path = self.root / name
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        return path


class MMChunkClientRemoteErrorTests(TemporaryScriptCase):
    def test_valid_generation_error_retains_structure_and_clears_candidate(self):
        message = (
            "joint left_ankle_roll_joint position -0.307408422 is outside "
            "range [-0.261799991, 0.261799991]"
        )
        child = self.script(
            "mm_remote_error.py",
            r'''
            import json
            import sys

            for line in sys.stdin:
                request = json.loads(line)
                op = request["op"]
                if op == "hello":
                    response = {
                        "v": 1,
                        "ok": True,
                        "op": op,
                        "request_id": request["request_id"],
                        "data": {"protocol_version": 1},
                    }
                elif op == "reset":
                    response = {
                        "v": 1,
                        "ok": True,
                        "op": op,
                        "request_id": request["request_id"],
                        "data": {
                            "session_id": request["session_id"],
                            "active_candidate_id": None,
                            "scene": {},
                            "initial_boundary": {},
                        },
                    }
                elif op == "generate":
                    response = {
                        "v": 1,
                        "ok": False,
                        "op": op,
                        "request_id": request["request_id"],
                        "error": {
                            "code": "generation_failed",
                            "message": (
                                "joint left_ankle_roll_joint position "
                                "-0.307408422 is outside range "
                                "[-0.261799991, 0.261799991]"
                            ),
                        },
                    }
                elif op == "close":
                    response = {
                        "v": 1,
                        "ok": True,
                        "op": op,
                        "request_id": request["request_id"],
                        "data": {},
                    }
                print(json.dumps(response, separators=(",", ":")), flush=True)
                if op == "close":
                    break
            ''',
        )
        client = MMChunkClient(
            run_root=self.root,
            command=(sys.executable, "-u", str(child)),
            stdout_archive=self.root / "mm.stdout.jsonl",
            stderr_archive=self.root / "mm.stderr.log",
            poll_interval_s=0.01,
            stop_grace_s=0.05,
            term_grace_s=0.05,
            kill_grace_s=0.05,
        )
        try:
            client.hello()
            client.reset(
                SessionConfig("sonic-flat-baseline", "flat-12s", 0.0),
                session_id="session-remote-error",
            )
            with self.assertRaises(_RemoteMMError) as raised:
                client.generate(
                    CommandSample(
                        chunk_index=0,
                        requested_velocity_mujoco=(0.5, 0.0, 0.0),
                        desired_heading_mujoco_wxyz=(1.0, 0.0, 0.0, 0.0),
                    ),
                    session_id="session-remote-error",
                    candidate_id="candidate-remote-error",
                    predecessor_id=None,
                    source_intervals=10,
                )
            self.assertEqual(raised.exception.code, "generation_failed")
            self.assertEqual(raised.exception.message, message)
            self.assertEqual(
                str(raised.exception), f"generation_failed: {message}"
            )
            self.assertIsNone(client.outstanding_candidate_id)
        finally:
            client.close()


class GatedSimulatorClientTests(TemporaryScriptCase):
    def client(self, script, *, cancelled=None, stem="sim"):
        return GatedSimulatorClient(
            run_root=self.root,
            command=[sys.executable, "-u", str(script)],
            stdout_archive=self.root / f"{stem}.stdout.jsonl",
            stderr_archive=self.root / f"{stem}.stderr.log",
            cancelled=cancelled,
            poll_interval_s=0.01,
            stop_grace_s=0.05,
            term_grace_s=0.05,
        )

    def test_strict_jsonl_round_trip_and_exact_step_result(self):
        child = self.script(
            "sim_child.py",
            r'''
            import json
            import sys

            for line in sys.stdin:
                request = json.loads(line)
                op = request["op"]
                data = {}
                if op == "hello":
                    data = {"protocol": "gated-sim/v1"}
                elif op == "reset":
                    data = {"nq": 36, "sim_dt_s": 0.005, "sim_time_s": 0.0}
                elif op == "advance":
                    steps = request["steps"]
                    data = {
                        "steps": steps,
                        "sim_time_start_s": 0.0,
                        "sim_time_end_s": steps * 0.005,
                        "state_rows": steps // 4,
                        "contact_rows": steps,
                    }
                elif op == "snapshot":
                    data = {
                        "steps": 80,
                        "sim_time_s": 0.4,
                        "state_rows": 20,
                        "contact_rows": 80,
                    }
                elif op == "close":
                    data = {"closed": True}
                response = {
                    "v": 1,
                    "ok": True,
                    "op": op,
                    "request_id": request["request_id"],
                    "data": data,
                }
                print(json.dumps(response, separators=(",", ":")), flush=True)
                if op == "close":
                    break
            ''',
        )
        client = self.client(child)
        try:
            self.assertEqual(client.hello()["protocol"], "gated-sim/v1")
            scene = self.root / "scene.xml"
            scene.write_text("<mujoco/>", encoding="utf-8")
            reset = client.reset(
                scene_xml=scene,
                initial_qpos=np.zeros(36),
                lateral_offset_m=0.0,
                yaw_offset_rad=0.0,
                log_dir=self.root / "logs",
            )
            self.assertEqual(reset["nq"], 36)
            self.assertEqual(client.sim_dt, reset["sim_dt_s"])
            result = client.advance(80)
            self.assertEqual(
                result,
                AdvanceResult(
                    steps=80,
                    sim_time_start_s=0.0,
                    sim_time_end_s=0.4,
                    state_rows=20,
                    contact_rows=80,
                ),
            )
            self.assertEqual(client.snapshot()["sim_time_s"], 0.4)
        finally:
            client.close()
        archived = (self.root / "sim.stdout.jsonl").read_text(encoding="utf-8")
        self.assertIn('"op":"advance"', archived)
        self.assertIn('"op":"close"', archived)

    def test_close_strictly_validates_response_then_guarantees_cleanup(self):
        child = self.script(
            "malformed_close.py",
            r'''
            import json
            import sys
            import time

            request = json.loads(sys.stdin.readline())
            print(json.dumps({
                "v": 1,
                "ok": True,
                "op": "close",
                "request_id": request["request_id"],
                "data": {"closed": False},
            }), flush=True)
            time.sleep(60)
            ''',
        )
        client = self.client(child)
        pid = client.pid
        pgid = client.pgid
        with self.assertRaisesRegex(ProcessProtocolError, "closed"):
            client.close()
        self.assertFalse(process_exists(pid))
        with self.assertRaises(ProcessLookupError):
            os.killpg(pgid, 0)
        self.assertIn(
            '"closed": false',
            (self.root / "sim.stdout.jsonl").read_text(encoding="utf-8"),
        )

    def test_close_missing_response_is_bounded_and_raised_after_cleanup(self):
        child = self.script(
            "missing_close.py",
            r'''
            import sys
            import time
            sys.stdin.readline()
            time.sleep(60)
            ''',
        )
        client = self.client(child)
        pid = client.pid
        pgid = client.pgid
        with self.assertRaisesRegex(ProcessError, "close response"):
            client.close()
        self.assertFalse(process_exists(pid))
        with self.assertRaises(ProcessLookupError):
            os.killpg(pgid, 0)

    def test_close_drains_and_archives_residual_stdout_before_raising(self):
        child = self.script(
            "residual_close.py",
            r'''
            import json
            import sys

            request = json.loads(sys.stdin.readline())
            print(json.dumps({
                "v": 1,
                "ok": True,
                "op": "close",
                "request_id": request["request_id"],
                "data": {"closed": True},
            }), flush=True)
            print("non-protocol residual", flush=True)
            ''',
        )
        client = self.client(child)
        with self.assertRaisesRegex(ProcessProtocolError, "residual"):
            client.close()
        archived = (self.root / "sim.stdout.jsonl").read_text(encoding="utf-8")
        self.assertIn('"closed": true', archived)
        self.assertIn("non-protocol residual", archived)

    def test_close_archives_unread_stdout_when_leader_already_died(self):
        child = self.script(
            "dead_unread_stdout.py",
            r'''
            print("unread post-mortem stdout", flush=True)
            raise SystemExit(29)
            ''',
        )
        client = self.client(child)
        deadline = time.monotonic() + 1.0
        while client.returncode is None and time.monotonic() < deadline:
            time.sleep(0.01)
        with self.assertRaisesRegex(ProcessProtocolError, "residual"):
            client.close()
        self.assertIn(
            "unread post-mortem stdout",
            (self.root / "sim.stdout.jsonl").read_text(encoding="utf-8"),
        )

    def test_close_archives_stdout_emitted_by_term_handler(self):
        child = self.script(
            "term_stdout.py",
            r'''
            import signal
            import sys
            import time

            def on_term(_signum, _frame):
                print("stdout from SIGTERM handler", flush=True)
                raise SystemExit(0)

            signal.signal(signal.SIGTERM, on_term)
            sys.stdin.readline()
            while True:
                time.sleep(1)
            ''',
        )
        client = self.client(child)
        with self.assertRaisesRegex(ProcessError, "close response"):
            client.close()
        self.assertIn(
            "stdout from SIGTERM handler",
            (self.root / "sim.stdout.jsonl").read_text(encoding="utf-8"),
        )

    def test_duplicate_response_key_is_rejected(self):
        child = self.script(
            "duplicate.py",
            r'''
            import json
            import sys
            sys.stdin.readline()
            print(
                '{"v":1,"ok":true,"op":"hello","request_id":"g0",'
                '"request_id":"wrong","data":{}}',
                flush=True,
            )
            close = json.loads(sys.stdin.readline())
            print(json.dumps({
                "v": 1,
                "ok": True,
                "op": "close",
                "request_id": close["request_id"],
                "data": {"closed": True},
            }), flush=True)
            ''',
        )
        client = self.client(child)
        try:
            with self.assertRaisesRegex(ProcessProtocolError, "duplicate JSON key"):
                client.hello()
        finally:
            client.close()

    def test_child_death_reports_stderr_and_archives_it(self):
        child = self.script(
            "dead.py",
            r'''
            import sys
            sys.stdin.readline()
            print("simulator exploded", file=sys.stderr, flush=True)
            raise SystemExit(17)
            ''',
        )
        client = self.client(child)
        try:
            with self.assertRaisesRegex(ChildProcessDied, "17") as caught:
                client.hello()
            self.assertIn("simulator exploded", str(caught.exception))
        finally:
            client.close()
        self.assertIn(
            "simulator exploded",
            (self.root / "sim.stderr.log").read_text(encoding="utf-8"),
        )

    def test_dead_leader_cleanup_kills_same_verified_group_descendant(self):
        grandchild_path = self.root / "sim-grandchild.pid"
        child = self.script(
            "dead_with_descendant.py",
            r'''
            import os
            from pathlib import Path
            import signal
            import subprocess
            import sys

            descendant = subprocess.Popen([
                sys.executable,
                "-u",
                "-c",
                "import signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "time.sleep(60)",
            ])
            Path(os.environ["GRANDCHILD_PATH"]).write_text(str(descendant.pid))
            sys.stdin.readline()
            raise SystemExit(19)
            ''',
        )
        old = os.environ.get("GRANDCHILD_PATH")
        os.environ["GRANDCHILD_PATH"] = str(grandchild_path)
        try:
            client = GatedSimulatorClient(
                run_root=self.root,
                command=[sys.executable, "-u", str(child)],
                stdout_archive=self.root / "orphan.stdout",
                stderr_archive=self.root / "orphan.stderr",
                env=dict(os.environ),
                poll_interval_s=0.01,
                stop_grace_s=0.05,
                term_grace_s=0.05,
            )
        finally:
            if old is None:
                os.environ.pop("GRANDCHILD_PATH", None)
            else:
                os.environ["GRANDCHILD_PATH"] = old
        pgid = client.pgid
        try:
            with self.assertRaisesRegex(ChildProcessDied, "19"):
                client.hello()
            descendant = int(grandchild_path.read_text(encoding="utf-8"))
            client.close()
            survived = process_exists(descendant)
        finally:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 1.0
            while process_exists(descendant) and time.monotonic() < deadline:
                time.sleep(0.01)
        self.assertFalse(survived)
        with self.assertRaises(ProcessLookupError):
            os.killpg(pgid, 0)

    def test_verification_failure_cleans_expected_new_group_before_constructor_exits(
        self,
    ):
        leader_path = self.root / "unverified-leader.pid"
        descendant_path = self.root / "unverified-descendant.pid"
        child = self.script(
            "unverified_group.py",
            r'''
            import os
            from pathlib import Path
            import signal
            import subprocess
            import sys

            descendant = subprocess.Popen([
                sys.executable,
                "-u",
                "-c",
                "import signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "time.sleep(60)",
            ])
            Path(os.environ["LEADER_PATH"]).write_text(str(os.getpid()))
            Path(os.environ["DESCENDANT_PATH"]).write_text(str(descendant.pid))
            ''',
        )
        env = dict(
            os.environ,
            LEADER_PATH=str(leader_path),
            DESCENDANT_PATH=str(descendant_path),
        )

        def fail_after_leader_exit(process):
            process.wait(timeout=2.0)
            raise ChildProcessDied("synthetic pre-verification leader exit")

        leader = None
        descendant = None
        try:
            with (
                patch(
                    "mm_sonic.process._verify_new_process_group",
                    side_effect=fail_after_leader_exit,
                ),
                self.assertRaisesRegex(ChildProcessDied, "pre-verification"),
            ):
                GatedSimulatorClient(
                    run_root=self.root,
                    command=[sys.executable, "-u", str(child)],
                    stdout_archive=self.root / "unverified.stdout",
                    stderr_archive=self.root / "unverified.stderr",
                    env=env,
                    term_grace_s=0.05,
                    kill_grace_s=0.2,
                )
            leader = int(leader_path.read_text(encoding="utf-8"))
            descendant = int(descendant_path.read_text(encoding="utf-8"))
            self.assertFalse(process_exists(descendant))
            with self.assertRaises(ProcessLookupError):
                os.killpg(leader, 0)
        finally:
            if leader is not None:
                try:
                    os.killpg(leader, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if descendant is not None:
                deadline = time.monotonic() + 1.0
                while process_exists(descendant) and time.monotonic() < deadline:
                    time.sleep(0.01)

    def test_reset_response_fields_are_strictly_typed(self):
        child = self.script(
            "bad_reset_response.py",
            r'''
            import json
            import sys
            request = json.loads(sys.stdin.readline())
            print(json.dumps({
                "v": 1,
                "ok": True,
                "op": "reset",
                "request_id": request["request_id"],
                "data": {"nq": True, "sim_dt_s": "0.005", "sim_time_s": 0.0},
            }), flush=True)
            close = json.loads(sys.stdin.readline())
            print(json.dumps({
                "v": 1,
                "ok": True,
                "op": "close",
                "request_id": close["request_id"],
                "data": {"closed": True},
            }), flush=True)
            ''',
        )
        client = self.client(child)
        try:
            scene = self.root / "scene.xml"
            scene.write_text("<mujoco/>", encoding="utf-8")
            with self.assertRaises(ProcessProtocolError):
                client.reset(
                    scene_xml=scene,
                    initial_qpos=np.zeros(36),
                    lateral_offset_m=0.0,
                    yaw_offset_rad=0.0,
                    log_dir=self.root / "logs",
                )
        finally:
            client.close()

    def test_wait_has_operator_cancellation_but_no_generation_deadline(self):
        delayed = self.script(
            "delayed.py",
            r'''
            import json
            import sys
            import time
            request = json.loads(sys.stdin.readline())
            time.sleep(0.15)
            print(json.dumps({
                "v": 1,
                "ok": True,
                "op": "hello",
                "request_id": request["request_id"],
                "data": {"protocol": "gated-sim/v1"},
            }), flush=True)
            close = json.loads(sys.stdin.readline())
            print(json.dumps({
                "v": 1,
                "ok": True,
                "op": "close",
                "request_id": close["request_id"],
                "data": {"closed": True},
            }), flush=True)
            ''',
        )
        client = self.client(delayed)
        try:
            self.assertEqual(client.hello()["protocol"], "gated-sim/v1")
        finally:
            client.close()

        blocked = self.script(
            "blocked.py",
            r'''
            import sys
            import time
            sys.stdin.readline()
            while True:
                time.sleep(1)
            ''',
        )
        cancelled = threading.Event()
        client = self.client(
            blocked,
            cancelled=cancelled.is_set,
            stem="blocked",
        )
        timer = threading.Timer(0.08, cancelled.set)
        timer.start()
        try:
            with self.assertRaises(OperatorCancelled):
                client.hello()
        finally:
            timer.cancel()
            pid = client.pid
            pgid = client.pgid
            with self.assertRaises(OperatorCancelled):
                client.close()
        self.assertFalse(process_exists(pid))
        with self.assertRaises(ProcessLookupError):
            os.killpg(pgid, 0)

    def test_run_root_rejects_escape_symlink_and_preexisting_archives(self):
        child = self.script("unused_client.py", "raise SystemExit(0)\n")
        with tempfile.TemporaryDirectory() as outside_text:
            outside = Path(outside_text)
            with self.assertRaisesRegex(ProcessError, "run_root"):
                GatedSimulatorClient(
                    run_root=self.root,
                    command=[sys.executable, str(child)],
                    stdout_archive=outside / "escape.stdout",
                    stderr_archive=self.root / "escape.stderr",
                )

            (self.root / "linked").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ProcessError, "symlink"):
                GatedSimulatorClient(
                    run_root=self.root,
                    command=[sys.executable, str(child)],
                    stdout_archive=self.root / "linked" / "sim.stdout",
                    stderr_archive=self.root / "linked.stderr",
                )

            existing = self.root / "existing.stdout"
            existing.write_text("do not truncate", encoding="utf-8")
            with self.assertRaisesRegex(ProcessError, "exists"):
                GatedSimulatorClient(
                    run_root=self.root,
                    command=[sys.executable, str(child)],
                    stdout_archive=existing,
                    stderr_archive=self.root / "existing.stderr",
                )
            self.assertEqual(existing.read_text(encoding="utf-8"), "do not truncate")

    def test_reset_paths_are_confined_before_jsonl_request(self):
        child = self.script(
            "reset_path_peer.py",
            r'''
            import json
            import sys
            for line in sys.stdin:
                request = json.loads(line)
                if request["op"] == "reset":
                    data = {"nq": 36, "sim_dt_s": 0.005, "sim_time_s": 0.0}
                else:
                    data = {"closed": True}
                print(json.dumps({
                    "v": 1,
                    "ok": True,
                    "op": request["op"],
                    "request_id": request["request_id"],
                    "data": data,
                }), flush=True)
                if request["op"] == "close":
                    break
            ''',
        )
        client = GatedSimulatorClient(
            run_root=self.root,
            command=[sys.executable, "-u", str(child)],
            stdout_archive=self.root / "reset-path.stdout",
            stderr_archive=self.root / "reset-path.stderr",
            stop_grace_s=0.1,
        )
        qpos = np.r_[np.zeros(3), 1.0, np.zeros(32)]
        try:
            with tempfile.TemporaryDirectory() as outside_text:
                outside_scene = Path(outside_text) / "scene.xml"
                outside_scene.write_text("<mujoco/>", encoding="utf-8")
                with self.assertRaisesRegex(ProcessError, "run_root"):
                    client.reset(
                        scene_xml=outside_scene,
                        initial_qpos=qpos,
                        lateral_offset_m=0.0,
                        yaw_offset_rad=0.0,
                        log_dir=self.root / "sim-a",
                    )

            existing_logs = self.root / "existing-reset-logs"
            existing_logs.mkdir()
            with self.assertRaisesRegex(ProcessError, "exists"):
                client.reset(
                    scene_xml=self.script("scene.xml", "<mujoco/>"),
                    initial_qpos=qpos,
                    lateral_offset_m=0.0,
                    yaw_offset_rad=0.0,
                    log_dir=existing_logs,
                )
        finally:
            client.close()


class GearProcessTests(TemporaryScriptCase):
    def gear(self, child, *, active=("CONTROL READY", "STREAM READY"), **kwargs):
        logs_dir = kwargs.pop("logs_dir", self.root / "gear-logs")
        readiness_poll_s = kwargs.pop("readiness_poll_s", 0.01)
        return GearProcess(
            run_root=self.root,
            command=[sys.executable, "-u", str(child)],
            target_motion_logfile=self.root / "target.csv",
            logs_dir=logs_dir,
            stdout_archive=self.root / "gear.stdout.log",
            stderr_archive=self.root / "gear.stderr.log",
            startup_markers=("BOOT READY",),
            active_markers=active,
            wait_for_control_marker="BOOT READY",
            readiness_poll_s=readiness_poll_s,
            stop_grace_s=0.05,
            term_grace_s=0.05,
            kill_grace_s=0.2,
            **kwargs,
        )

    def test_launches_under_pty_in_verified_new_group_with_exact_official_flags(self):
        argv_path = self.root / "argv.json"
        keys_path = self.root / "keys.bin"
        child = self.script(
            "gear_child.py",
            r'''
            import json
            import os
            from pathlib import Path
            import sys
            import termios

            Path(os.environ["ARGV_PATH"]).write_text(json.dumps(sys.argv[1:]))
            print(f"BOOT READY tty={os.isatty(0)}", flush=True)
            print("gear stderr", file=sys.stderr, flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            print("BOOT READY", flush=True)
            keys = os.read(0, 1)
            print("CONTROL READY", flush=True)
            keys += os.read(0, 1)
            Path(os.environ["KEYS_PATH"]).write_bytes(keys)
            print("STREAM READY", flush=True)
            while True:
                key = os.read(0, 1)
                if key.lower() == b"o":
                    break
            ''',
        )
        env = dict(os.environ, ARGV_PATH=str(argv_path), KEYS_PATH=str(keys_path))
        gear = self.gear(child, env=env)
        try:
            gear.start()
            self.assertNotEqual(gear.pgid, os.getpgrp())
            self.assertEqual(gear.pgid, gear.pid)
            self.assertEqual(os.getpgid(gear.pid), gear.pgid)
            self.assertEqual(keys_path.read_bytes(), b"]\n")
            argv = json.loads(argv_path.read_text(encoding="utf-8"))
            self.assertIn("--input-type", argv)
            self.assertEqual(argv[argv.index("--input-type") + 1], "zmq")
            self.assertEqual(argv.count("--target-motion-logfile"), 1)
            self.assertEqual(argv.count("--logs-dir"), 1)
            self.assertEqual(argv.count("--enable-csv-logs"), 1)
            self.assertEqual(argv.count("--zmq-verbose"), 1)
            self.assertEqual(argv.count("--disable-crc-check"), 1)
            self.assertNotIn("--zmq-conflate", argv)
        finally:
            gear.close()
        self.assertIn(
            "BOOT READY tty=True",
            (self.root / "gear.stdout.log").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "gear stderr",
            (self.root / "gear.stderr.log").read_text(encoding="utf-8"),
        )
        self.assertEqual(gear.cleanup_history[0], "official-stop-key")
        self.assertFalse(process_exists(gear.pid))

    def test_close_removes_only_its_empty_logs_directory(self):
        child = self.script("unused_logs.py", "raise SystemExit(0)\n")
        gear = self.gear(child)
        logs = gear.logs_dir
        self.assertTrue(logs.is_dir())

        gear.close()

        self.assertFalse(logs.exists())

        preserved = self.gear(child)
        marker = preserved.logs_dir / "retained.csv"
        marker.write_text("evidence\n", encoding="utf-8")
        preserved.close()
        self.assertEqual(marker.read_text(encoding="utf-8"), "evidence\n")

    def test_close_cannot_follow_replaced_logs_parent_outside_run_root(self):
        child = self.script("unused_swapped_logs.py", "raise SystemExit(0)\n")
        parent = self.root / "managed"
        logs = parent / "gear-logs"
        gear = self.gear(child, logs_dir=logs)
        moved_parent = self.root / "managed-original"
        parent.rename(moved_parent)

        with tempfile.TemporaryDirectory() as outside_text:
            outside = Path(outside_text)
            outside_logs = outside / "gear-logs"
            outside_logs.mkdir()
            parent.symlink_to(outside, target_is_directory=True)

            gear.close()

            self.assertTrue(outside_logs.is_dir())
            self.assertFalse((moved_parent / "gear-logs").exists())

    def test_close_retains_a_replacement_logs_leaf_with_different_inode(self):
        child = self.script("unused_replaced_logs.py", "raise SystemExit(0)\n")
        gear = self.gear(child)
        original = self.root / "original-gear-logs"
        gear.logs_dir.rename(original)
        gear.logs_dir.mkdir()

        gear.close()

        self.assertTrue(original.is_dir())
        self.assertTrue(gear.logs_dir.is_dir())

    def test_owned_logs_leaf_descriptor_remains_live_until_close(self):
        child = self.script("unused_live_logs_fd.py", "raise SystemExit(0)\n")
        gear = self.gear(child)
        leaf_fd = gear._owned_logs_directory.leaf_fd
        parent_fd = gear._owned_logs_directory.parent_fd
        original = os.fstat(leaf_fd)
        gear.logs_dir.rmdir()
        gear.logs_dir.mkdir()
        replacement = gear.logs_dir.stat(follow_symlinks=False)

        self.assertNotEqual(
            (replacement.st_dev, replacement.st_ino),
            (original.st_dev, original.st_ino),
        )
        gear.close()

        self.assertTrue(gear.logs_dir.is_dir())
        for descriptor in (leaf_fd, parent_fd):
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_invalid_poll_configuration_does_not_create_logs_directory(self):
        child = self.script("unused_invalid_poll.py", "raise SystemExit(0)\n")

        with self.assertRaisesRegex(ValueError, "poll intervals"):
            self.gear(child, readiness_poll_s=0.0)

        self.assertFalse((self.root / "gear-logs").exists())

    def test_environment_is_copied_before_any_logs_parent_is_created(self):
        child = self.script("unused_environment_order.py", "raise SystemExit(0)\n")
        parent = self.root / "late-parent"
        observations = []

        class ObservedEnvironment:
            def keys(self):
                observations.append(parent.exists())
                return ("PATH",)

            @staticmethod
            def __getitem__(_key):
                return "/usr/bin"

        gear = self.gear(
            child,
            logs_dir=parent / "gear-logs",
            env=ObservedEnvironment(),
        )
        try:
            self.assertEqual(observations, [False])
        finally:
            gear.close()

    def test_logs_leaf_open_failure_rolls_back_created_directory(self):
        from mm_sonic import process as process_module

        child = self.script("unused_leaf_open_failure.py", "raise SystemExit(0)\n")
        real_open = process_module.os.open

        def fail_leaf_open(path, flags, mode=0o777, *, dir_fd=None):
            if (
                isinstance(path, str)
                and path.startswith(process_module._OWNED_DIRECTORY_STAGING_PREFIX)
                and dir_fd is not None
            ):
                raise OSError("synthetic leaf open failure")
            return real_open(path, flags, mode, dir_fd=dir_fd)

        with (
            patch.object(process_module.os, "open", side_effect=fail_leaf_open),
            self.assertRaisesRegex(ProcessError, "cannot create GEAR logs"),
        ):
            self.gear(child)

        self.assertFalse((self.root / "gear-logs").exists())

    def test_logs_staging_leaf_swap_after_open_is_rejected_without_deletion(self):
        from mm_sonic import process as process_module

        child = self.script("unused_leaf_swap.py", "raise SystemExit(0)\n")
        real_stat = process_module.os.stat
        created = self.root / "wrapper-created-logs"
        replacement = None
        swapped = False

        def swap_leaf_stat(path, *args, dir_fd=None, **kwargs):
            nonlocal replacement, swapped
            if (
                isinstance(path, str)
                and path.startswith(process_module._OWNED_DIRECTORY_STAGING_PREFIX)
                and dir_fd is not None
                and not swapped
            ):
                swapped = True
                replacement = self.root / path
                replacement.rename(created)
                replacement.mkdir()
            return real_stat(path, *args, dir_fd=dir_fd, **kwargs)

        with (
            patch.object(process_module.os, "stat", side_effect=swap_leaf_stat),
            self.assertRaisesRegex(ProcessError, "identity changed"),
        ):
            self.gear(child)

        self.assertTrue(created.is_dir())
        assert replacement is not None
        self.assertTrue(replacement.is_dir())

    def test_logs_leaf_swap_before_first_public_stat_is_rejected_without_deletion(
        self,
    ):
        from mm_sonic import process as process_module

        real_stat = process_module.os.stat
        created = self.root / "wrapper-created-before-public-stat"
        replacement = self.root / "gear-logs"
        swapped = False
        owned = None

        def swap_leaf_stat(path, *args, dir_fd=None, **kwargs):
            nonlocal swapped
            if path == "gear-logs" and dir_fd is not None and not swapped:
                swapped = True
                replacement.rename(created)
                replacement.mkdir()
            return real_stat(path, *args, dir_fd=dir_fd, **kwargs)

        try:
            with patch.object(
                process_module.os,
                "stat",
                side_effect=swap_leaf_stat,
            ):
                with self.assertRaisesRegex(ProcessError, "identity changed"):
                    _, owned = process_module._create_owned_directory(
                        self.root,
                        replacement,
                        "GEAR logs",
                    )
        finally:
            if owned is not None:
                os.close(owned.leaf_fd)
                os.close(owned.parent_fd)

        self.assertTrue(swapped)
        self.assertTrue(created.is_dir())
        self.assertTrue(replacement.is_dir())

    def test_logs_leaf_swap_on_open_failure_preserves_both_directories(self):
        from mm_sonic import process as process_module

        child = self.script("unused_leaf_swap_failure.py", "raise SystemExit(0)\n")
        real_stat = process_module.os.stat
        created = self.root / "wrapper-created-on-failure"
        replacement = None
        swapped = False

        def swap_then_fail(path, *args, dir_fd=None, **kwargs):
            nonlocal replacement, swapped
            if (
                isinstance(path, str)
                and path.startswith(process_module._OWNED_DIRECTORY_STAGING_PREFIX)
                and dir_fd is not None
                and not swapped
            ):
                swapped = True
                replacement = self.root / path
                replacement.rename(created)
                replacement.mkdir()
                raise OSError("synthetic swapped leaf stat failure")
            return real_stat(path, *args, dir_fd=dir_fd, **kwargs)

        with (
            patch.object(process_module.os, "stat", side_effect=swap_then_fail),
            self.assertRaisesRegex(ProcessError, "cannot create GEAR logs"),
        ):
            self.gear(child)

        self.assertTrue(created.is_dir())
        assert replacement is not None
        self.assertTrue(replacement.is_dir())

    def test_command_is_materialized_once_before_validation_or_output_creation(self):
        child = self.script("unused_one_shot_command.py", "raise SystemExit(0)\n")

        class OneShotCommand:
            def __init__(self):
                self.iterations = 0

            def __len__(self):
                return 2

            def __iter__(self):
                self.iterations += 1
                if self.iterations > 1:
                    raise RuntimeError("command was iterated more than once")
                yield sys.executable
                yield str(child)

        command = OneShotCommand()
        gear = GearProcess(
            run_root=self.root,
            command=command,
            target_motion_logfile=self.root / "one-shot" / "target.csv",
            logs_dir=self.root / "one-shot" / "logs",
            stdout_archive=self.root / "one-shot" / "out",
            stderr_archive=self.root / "one-shot" / "err",
        )
        try:
            self.assertEqual(command.iterations, 1)
            self.assertEqual(gear.argv[:2], (sys.executable, str(child)))
        finally:
            gear.close()

    def test_default_markers_follow_literal_official_control_then_stream_order(self):
        keys_path = self.root / "default-marker-keys.bin"
        child = self.script(
            "official_default_markers.py",
            r'''
            import os
            from pathlib import Path
            import termios

            print("Initialized ZMQ endpoint interface", flush=True)
            print("Init Done", flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            keys = os.read(0, 1)
            print(
                "[Control] DEBUG: operator_state.start=true, "
                "transitioning to CONTROL state",
                flush=True,
            )
            keys += os.read(0, 1)
            Path(os.environ["KEYS_PATH"]).write_bytes(keys)
            print("ZMQ STREAMING MODE: ENABLED", flush=True)
            while os.read(0, 1).lower() != b"o":
                pass
            ''',
        )
        gear = GearProcess(
            run_root=self.root,
            command=[sys.executable, "-u", str(child)],
            target_motion_logfile=self.root / "default-target.csv",
            logs_dir=self.root / "default-logs",
            stdout_archive=self.root / "default.stdout",
            stderr_archive=self.root / "default.stderr",
            readiness_timeout_s=0.15,
            readiness_poll_s=0.005,
            stop_grace_s=0.05,
            term_grace_s=0.05,
            kill_grace_s=0.2,
            env=dict(os.environ, KEYS_PATH=str(keys_path)),
        )
        try:
            gear.start()
            self.assertEqual(keys_path.read_bytes(), b"]\n")
        finally:
            gear.close()

    def test_loaded_motion_profile_uses_authenticated_keyboard_playback(self):
        argv_path = self.root / "loaded-motion-argv.json"
        keys_path = self.root / "loaded-motion-keys.bin"
        child = self.script(
            "official_loaded_motion_markers.py",
            r'''
            import json
            import os
            from pathlib import Path
            import sys
            import termios

            Path(os.environ["ARGV_PATH"]).write_text(json.dumps(sys.argv[1:]))
            print("✓ Motion data loaded successfully!", flush=True)
            print("Started with motion: walk.csv (paused at frame 0)", flush=True)
            print("Initialized keyboard input interface (default)", flush=True)
            print("Init Done", flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            keys = os.read(0, 1)
            print(
                "[Control] DEBUG: operator_state.start=true, "
                "transitioning to CONTROL state",
                flush=True,
            )
            keys += os.read(0, 1)
            Path(os.environ["KEYS_PATH"]).write_bytes(keys)
            print("Playing motion 0 from frame 0 to end (walk.csv)", flush=True)
            while os.read(0, 1).lower() != b"o":
                pass
            ''',
        )
        target = self.root / "loaded-target.csv"
        logs = self.root / "loaded-logs"
        gear = GearProcess(
            run_root=self.root,
            command=[sys.executable, "-u", str(child), "/official/reference"],
            target_motion_logfile=target,
            logs_dir=logs,
            stdout_archive=self.root / "loaded.stdout",
            stderr_archive=self.root / "loaded.stderr",
            launch_profile="loaded_motion",
            readiness_timeout_s=0.15,
            readiness_poll_s=0.005,
            stop_grace_s=0.05,
            term_grace_s=0.05,
            kill_grace_s=0.2,
            env=dict(
                os.environ,
                ARGV_PATH=str(argv_path),
                KEYS_PATH=str(keys_path),
            ),
        )
        try:
            self.assertFalse(gear.startup_markers_ready)
            gear.start()
            self.assertTrue(gear.startup_markers_ready)
            self.assertEqual(keys_path.read_bytes(), b"]t")
            self.assertEqual(
                json.loads(argv_path.read_text(encoding="utf-8")),
                [
                    "/official/reference",
                    "--input-type",
                    "keyboard",
                    "--target-motion-logfile",
                    str(target),
                    "--logs-dir",
                    str(logs),
                    "--enable-csv-logs",
                    "--disable-crc-check",
                ],
            )
            self.assertTrue(gear.ready)
        finally:
            gear.close()
        self.assertEqual(gear.cleanup_history[0], "official-stop-key")
        self.assertFalse(process_exists(gear.pid))

    def test_launch_profile_is_closed_and_default_remains_zmq_stream(self):
        child = self.script("unused_profile.py", "raise SystemExit(0)\n")
        common = dict(
            run_root=self.root,
            command=[sys.executable, str(child)],
            target_motion_logfile=self.root / "profile-target.csv",
            logs_dir=self.root / "profile-logs",
            stdout_archive=self.root / "profile.stdout",
            stderr_archive=self.root / "profile.stderr",
        )
        with self.assertRaisesRegex(ValueError, "launch_profile"):
            GearProcess(**common, launch_profile="arbitrary")

        gear = GearProcess(**common)
        self.addCleanup(gear.close)
        self.assertEqual(gear.launch_profile, "zmq_stream")
        self.assertEqual(gear.argv[-9:], (
            "--input-type",
            "zmq",
            "--target-motion-logfile",
            str(self.root / "profile-target.csv"),
            "--logs-dir",
            str(self.root / "profile-logs"),
            "--enable-csv-logs",
            "--disable-crc-check",
            "--zmq-verbose",
        ))

    def test_loaded_motion_rejects_nonzero_start_frame_before_activation(self):
        child = self.script(
            "loaded_motion_wrong_frame.py",
            r'''
            import os
            import termios

            print("✓ Motion data loaded successfully!", flush=True)
            print("Started with motion: walk.csv (paused at frame 1)", flush=True)
            print("Initialized keyboard input interface (default)", flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            os.read(0, 1)
            print(
                "[Control] DEBUG: operator_state.start=true, "
                "transitioning to CONTROL state",
                flush=True,
            )
            os.read(0, 1)
            print("Playing motion 0 from frame 0 to end (walk.csv)", flush=True)
            while os.read(0, 1).lower() != b"o":
                pass
            ''',
        )
        gear = GearProcess(
            run_root=self.root,
            command=[sys.executable, "-u", str(child), "/official/reference"],
            target_motion_logfile=self.root / "wrong-frame-target.csv",
            logs_dir=self.root / "wrong-frame-logs",
            stdout_archive=self.root / "wrong-frame.stdout",
            stderr_archive=self.root / "wrong-frame.stderr",
            launch_profile="loaded_motion",
            readiness_timeout_s=0.05,
            readiness_poll_s=0.005,
            stop_grace_s=0.05,
            term_grace_s=0.05,
            kill_grace_s=0.2,
        )
        try:
            with self.assertRaisesRegex(ProcessError, "readiness"):
                gear.start()
        finally:
            gear.close()
        self.assertFalse(process_exists(gear.pid))

    def test_loaded_motion_phased_activation_resumes_without_marker_restop(self):
        keys_path = self.root / "scoring-reset-keys.bin"
        child = self.script(
            "loaded_motion_scoring_reset.py",
            r'''
            import os
            from pathlib import Path
            import termios

            print("✓ Motion data loaded successfully!", flush=True)
            print("Started with motion: walk.csv (paused at frame 0)", flush=True)
            print("Initialized keyboard input interface (default)", flush=True)
            print("Init Done", flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            keys = os.read(0, 1)
            print("Reset motion 0 to frame 0 (paused)", flush=True)
            keys += os.read(0, 1)
            print("Playing motion 0 from frame 0 to end (walk.csv)", flush=True)
            keys += os.read(0, 1)
            Path(os.environ["KEYS_PATH"]).write_bytes(keys)
            print(
                "[Control] DEBUG: operator_state.start=true, "
                "transitioning to CONTROL state",
                flush=True,
            )
            while os.read(0, 1).lower() != b"o":
                pass
            ''',
        )
        gear = GearProcess(
            run_root=self.root,
            command=[sys.executable, "-u", str(child), "/official/reference"],
            target_motion_logfile=self.root / "scoring-reset-target.csv",
            logs_dir=self.root / "scoring-reset-logs",
            stdout_archive=self.root / "scoring-reset.stdout",
            stderr_archive=self.root / "scoring-reset.stderr",
            launch_profile="loaded_motion",
            readiness_timeout_s=0.15,
            readiness_poll_s=0.005,
            signal_poll_s=0.005,
            stop_grace_s=0.05,
            term_grace_s=0.05,
            kill_grace_s=0.2,
            env=dict(os.environ, KEYS_PATH=str(keys_path)),
        )
        try:
            gear.start_to_wait_for_control()
            self.assertTrue(gear.wait_for_control_ready)
            gear.prepare_loaded_motion_for_scoring()
            gear.stop_group()
            gear.activate_loaded_motion_for_scoring()
            self.assertTrue(gear.control_active)
            self.assertTrue(gear.group_is_resumed())
            self.assertEqual(keys_path.read_bytes(), b"rt]")
        finally:
            gear.close()
        self.assertFalse(process_exists(gear.pid))

    def test_archive_write_failure_is_surfaced_after_child_cleanup(self):
        child = self.script(
            "archive_failure.py",
            r'''
            import os
            import termios
            print("BOOT READY", flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            while True:
                os.read(0, 1)
            ''',
        )
        gear = self.gear(child, readiness_timeout_s=0.1)
        from mm_sonic import process as process_module

        real_open = process_module._open_exclusive_binary_output

        class FailingArchive:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            @property
            def closed(self):
                return self.wrapped.closed

            def write(self, _chunk):
                raise OSError("synthetic archive write failure")

            def flush(self):
                self.wrapped.flush()

            def close(self):
                self.wrapped.close()

        def injected_open(run_root, value, label):
            opened = real_open(run_root, value, label)
            if label == "GEAR stdout archive":
                return FailingArchive(opened)
            return opened

        with patch(
            "mm_sonic.process._open_exclusive_binary_output",
            side_effect=injected_open,
        ):
            with self.assertRaisesRegex(ProcessError, "archive.*write failure"):
                gear.start()
        self.assertFalse(process_exists(gear.pid))

    def test_close_reports_reader_that_does_not_drain_before_bound(self):
        child = self.script(
            "incomplete_reader.py",
            r'''
            import os
            import termios
            print("BOOT READY", flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            os.read(0, 1)
            print("CONTROL READY", flush=True)
            os.read(0, 1)
            print("STREAM READY", flush=True)
            while os.read(0, 1).lower() != b"o":
                pass
            ''',
        )
        gear = self.gear(child)
        gear.start()
        release_reader = threading.Event()
        stuck = threading.Thread(target=release_reader.wait, daemon=True)
        stuck.start()
        gear._reader_threads.append(stuck)
        try:
            with self.assertRaisesRegex(ProcessError, "did not drain"):
                gear.close()
        finally:
            release_reader.set()
            stuck.join(timeout=1.0)
        self.assertFalse(process_exists(gear.pid))

    def test_preemitted_readiness_markers_cannot_satisfy_key_handshake(self):
        child = self.script(
            "preemitted.py",
            r'''
            import os
            import termios

            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            print("BOOT READY", flush=True)
            print("CONTROL READY", flush=True)
            print("STREAM READY", flush=True)
            os.read(0, 2)
            while os.read(0, 1).lower() != b"o":
                pass
            ''',
        )
        gear = self.gear(child, readiness_timeout_s=0.1)
        try:
            with self.assertRaisesRegex(ProcessError, "readiness"):
                gear.start()
        finally:
            gear.close()

    def test_out_of_order_stream_marker_cannot_satisfy_newline_handshake(self):
        child = self.script(
            "out_of_order.py",
            r'''
            import os
            import termios

            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            print("BOOT READY", flush=True)
            os.read(0, 1)
            print("STREAM READY", flush=True)
            print("CONTROL READY", flush=True)
            os.read(0, 1)
            while os.read(0, 1).lower() != b"o":
                pass
            ''',
        )
        gear = self.gear(child, readiness_timeout_s=0.1)
        try:
            with self.assertRaisesRegex(ProcessError, "readiness"):
                gear.start()
        finally:
            gear.close()

    def test_stream_preparation_waits_for_post_enable_reset_fence(self):
        release = self.root / "release-reset-tail"
        enabled = self.root / "enabled"
        keys = self.root / "post-enable.keys"
        child = self.script(
            "post_enable_reset_fence.py",
            r'''
            import os
            from pathlib import Path
            import termios
            import time

            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            print("BOOT READY", flush=True)
            assert os.read(0, 1) == b"\n"
            print("STREAM READY", flush=True)
            Path(os.environ["ENABLED_PATH"]).write_bytes(b"enabled")
            release = Path(os.environ["RELEASE_PATH"])
            while not release.exists():
                time.sleep(0.001)
            keys = os.read(0, 2)
            Path(os.environ["KEYS_PATH"]).write_bytes(keys)
            if keys == b"qe":
                print("Delta heading left: 0.1 rad", flush=True)
                print("Delta heading right: 0 rad", flush=True)
            while os.read(0, 1).lower() != b"o":
                pass
            ''',
        )
        gear = self.gear(
            child,
            readiness_timeout_s=0.5,
            env=dict(
                os.environ,
                ENABLED_PATH=str(enabled),
                RELEASE_PATH=str(release),
                KEYS_PATH=str(keys),
            ),
        )
        result = []
        errors = []
        try:
            gear.start_to_wait_for_control()

            def prepare():
                try:
                    result.append(gear.enable_stream_for_preload())
                except BaseException as error:
                    errors.append(error)

            worker = threading.Thread(target=prepare)
            worker.start()
            deadline = time.monotonic() + 1.0
            while not enabled.exists() and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertTrue(enabled.exists())
            self.assertTrue(worker.is_alive())
            self.assertFalse(gear.input_prepared)
            release.write_bytes(b"release")
            worker.join(timeout=1.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(keys.read_bytes(), b"qe")
            self.assertTrue(gear.input_prepared)
            fence = result[0]
            self.assertEqual(
                dict(fence),
                {
                    "boundary": fence["boundary"],
                    "end_offset": fence["end_offset"],
                    "key_sequence": "qe",
                    "left_line": "Delta heading left: 0.1 rad",
                    "right_line": "Delta heading right: 0 rad",
                    "semantics": (
                        "post-enable-reset-tail-complete-with-net-zero-heading"
                    ),
                },
            )
            self.assertLess(fence["boundary"], fence["end_offset"])
        finally:
            release.touch(exist_ok=True)
            gear.close()

    def test_stream_preparation_requires_authenticated_cold_wait(self):
        child = self.script("never_started.py", "raise SystemExit(0)\n")
        gear = self.gear(child)
        try:
            with self.assertRaisesRegex(ProcessError, "WAIT_FOR_CONTROL"):
                gear.enable_stream_for_preload()
            self.assertFalse(gear.input_prepared)
        finally:
            gear.close()

    def test_stream_preparation_rejects_missing_or_reordered_fence_lines(self):
        for label, lines in (
            ("missing", ("Delta heading left: 0.1 rad",)),
            (
                "reordered",
                (
                    "Delta heading right: 0 rad",
                    "Delta heading left: 0.1 rad",
                ),
            ),
        ):
            with self.subTest(label=label):
                child = self.script(
                    f"post_enable_{label}.py",
                    f'''
                    import os
                    import termios

                    attrs = termios.tcgetattr(0)
                    attrs[3] &= ~(termios.ICANON | termios.ECHO)
                    termios.tcsetattr(0, termios.TCSANOW, attrs)
                    print("BOOT READY", flush=True)
                    assert os.read(0, 1) == b"\\n"
                    print("STREAM READY", flush=True)
                    assert os.read(0, 2) == b"qe"
                    for line in {lines!r}:
                        print(line, flush=True)
                    raise SystemExit(0)
                    ''',
                )
                gear = GearProcess(
                    run_root=self.root,
                    command=[sys.executable, "-u", str(child)],
                    target_motion_logfile=self.root / f"{label}.target.csv",
                    logs_dir=self.root / f"{label}-logs",
                    stdout_archive=self.root / f"{label}.stdout",
                    stderr_archive=self.root / f"{label}.stderr",
                    startup_markers=("BOOT READY",),
                    active_markers=("CONTROL READY", "STREAM READY"),
                    wait_for_control_marker="BOOT READY",
                    readiness_timeout_s=0.1,
                    readiness_poll_s=0.005,
                    stop_grace_s=0.05,
                    term_grace_s=0.05,
                    kill_grace_s=0.2,
                )
                try:
                    gear.start_to_wait_for_control()
                    with self.assertRaises(ProcessError):
                        gear.enable_stream_for_preload()
                    self.assertFalse(gear.input_prepared)
                finally:
                    gear.close()

    def test_raw_utf8_observation_offsets_equal_archived_byte_offsets(self):
        child = self.script(
            "raw_utf8_offsets.py",
            r'''
            import os
            import termios
            import time

            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            os.write(2, "✓ stderr noise\n".encode("utf-8"))
            time.sleep(0.02)
            os.write(1, b"\xe2")
            time.sleep(0.01)
            os.write(1, b"\x9c\x93 BOOT READY\n")
            assert os.read(0, 1) == b"\n"
            print("STREAM READY", flush=True)
            assert os.read(0, 2) == b"qe"
            print("Delta heading left: 0.1 rad", flush=True)
            print("Delta heading right: 0 rad", flush=True)
            while os.read(0, 1).lower() != b"o":
                pass
            ''',
        )
        gear = GearProcess(
            run_root=self.root,
            command=[sys.executable, "-u", str(child)],
            target_motion_logfile=self.root / "utf8.target.csv",
            logs_dir=self.root / "utf8-logs",
            stdout_archive=self.root / "utf8.stdout",
            stderr_archive=self.root / "utf8.stderr",
            startup_markers=("✓ BOOT READY",),
            active_markers=("CONTROL READY", "STREAM READY"),
            wait_for_control_marker="✓ BOOT READY",
            readiness_timeout_s=0.2,
            readiness_poll_s=0.005,
            stop_grace_s=0.05,
            term_grace_s=0.05,
            kill_grace_s=0.2,
        )
        try:
            gear.start_to_wait_for_control()
            fence = gear.enable_stream_for_preload()
            boundary = gear.publication_boundary()
            archived = (self.root / "utf8.stdout").read_bytes()
            self.assertEqual(boundary, len(archived))
            self.assertIn(
                "✓ stderr noise".encode("utf-8"),
                (self.root / "utf8.stderr").read_bytes(),
            )
            self.assertEqual(fence["end_offset"], boundary)
            self.assertEqual(archived[: fence["boundary"]].decode("utf-8").splitlines()[-1], "STREAM READY")
        finally:
            gear.close()

    def test_wait_preload_requires_exact_authenticated_processing_transcript(self):
        child = self.script(
            "wait_preload_processing.py",
            r'''
            import os
            import termios

            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            print("BOOT READY", flush=True)
            os.read(0, 1)
            print("STREAM READY", flush=True)
            assert os.read(0, 2) == b"qe"
            print("Delta heading left: 0.1 rad", flush=True)
            print("Delta heading right: 0 rad", flush=True)
            os.read(0, 1)
            print(
                "[ZMQEndpointInterface] *** Starting ZMQ processing ***",
                flush=True,
            )
            print("[ZMQEndpointInterface] Protocol version: 1", flush=True)
            print("[ZMQEndpointInterface] Protocol version 1 established", flush=True)
            print(
                "[StreamedMotionMerger] Processing 20 frames, "
                "incoming_frame_start=1, frame_step=1",
                flush=True,
            )
            print(
                "[StreamedMotionMerger] Merged motion: 21 frames "
                "(copied: 1 + incoming: 20)",
                flush=True,
            )
            print("[ZMQEndpointInterface] active_protocol_version_=1", flush=True)
            print("[ZMQEndpointInterface] result.motion->GetEncodeMode()=0", flush=True)
            print("[ZMQEndpointInterface] motion name: streamed", flush=True)
            print(
                "[ZMQEndpointInterface] Merged streamed data: 21 current-rate "
                "frames, window [0..20] (message-index), frame_step=1, "
                "frame_offset_adjustment=0, did_catchup=0",
                flush=True,
            )
            print(
                "[ZMQEndpointInterface] *** End of ZMQ decoding processing ***",
                flush=True,
            )
            while os.read(0, 1).lower() != b"o":
                pass
            ''',
        )
        gear = self.gear(child, readiness_timeout_s=0.1)
        try:
            gear.start_to_wait_for_control()
            gear.enable_stream_for_preload()
            boundary = gear.publication_boundary()
            gear.write_keys(b"x")
            marker = gear.wait_for_stream_processing(
                boundary,
                frame_count=20,
                global_start=1,
                merged_count=21,
            )
            self.assertEqual(marker["frame_count"], 20)
            self.assertEqual(marker["global_start"], 1)
            self.assertEqual(marker["merged_count"], 21)
            archived = (self.root / "gear.stdout.log").read_bytes()
            for key, line in (
                ("start_range", marker["start_line"]),
                ("processing_range", marker["processing_line"]),
                ("merged_range", marker["merged_line"]),
                ("end_range", marker["end_line"]),
            ):
                start, end = marker[key]
                self.assertEqual(archived[start:end], f"{line}\n".encode("ascii"))
            self.assertLess(marker["start_range"][1], marker["processing_range"][0])
            self.assertLess(marker["merged_range"][1], marker["end_range"][0])
            self.assertEqual(marker["end_offset"], marker["end_range"][1])
        finally:
            gear.close()

    def test_wait_preload_rejects_publication_without_exact_decoder_end(self):
        child = self.script(
            "wait_preload_missing_end.py",
            r'''
            import os
            import termios

            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            print("BOOT READY", flush=True)
            os.read(0, 1)
            print("STREAM READY", flush=True)
            assert os.read(0, 2) == b"qe"
            print("Delta heading left: 0.1 rad", flush=True)
            print("Delta heading right: 0 rad", flush=True)
            os.read(0, 1)
            print(
                "[ZMQEndpointInterface] *** Starting ZMQ processing ***",
                flush=True,
            )
            print(
                "[StreamedMotionMerger] Processing 20 frames, "
                "incoming_frame_start=1, frame_step=1",
                flush=True,
            )
            print(
                "[StreamedMotionMerger] Merged motion: 21 frames "
                "(copied: 1 + incoming: 20)",
                flush=True,
            )
            print("[ZMQEndpointInterface] one post-merge diagnostic", flush=True)
            ''',
        )
        gear = self.gear(child, readiness_timeout_s=0.1)
        try:
            gear.start_to_wait_for_control()
            gear.enable_stream_for_preload()
            boundary = gear.publication_boundary()
            gear.write_keys(b"x")
            with self.assertRaises(ProcessError):
                gear.wait_for_stream_processing(
                    boundary,
                    frame_count=20,
                    global_start=1,
                    merged_count=21,
                )
        finally:
            gear.close()

    def test_wait_preload_stays_blocked_until_exact_delayed_decoder_end(self):
        child = self.script(
            "wait_preload_delayed_end.py",
            r'''
            import os
            import termios

            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            print("BOOT READY", flush=True)
            os.read(0, 1)
            print("STREAM READY", flush=True)
            assert os.read(0, 2) == b"qe"
            print("Delta heading left: 0.1 rad", flush=True)
            print("Delta heading right: 0 rad", flush=True)
            os.read(0, 1)
            print(
                "[ZMQEndpointInterface] *** Starting ZMQ processing ***",
                flush=True,
            )
            print("[ZMQEndpointInterface] Protocol version: 1", flush=True)
            print("[ZMQEndpointInterface] Protocol version 1 established", flush=True)
            print(
                "[StreamedMotionMerger] Processing 20 frames, "
                "incoming_frame_start=1, frame_step=1",
                flush=True,
            )
            print(
                "[StreamedMotionMerger] Merged motion: 21 frames "
                "(copied: 1 + incoming: 20)",
                flush=True,
            )
            print("[ZMQEndpointInterface] active_protocol_version_=1", flush=True)
            print("[ZMQEndpointInterface] motion name: streamed", flush=True)
            os.read(0, 1)
            print(
                "[ZMQEndpointInterface] *** End of ZMQ decoding processing ***",
                flush=True,
            )
            while os.read(0, 1).lower() != b"o":
                pass
            ''',
        )
        gear = self.gear(child, readiness_timeout_s=2.0)
        waiter = None
        marker_result = []
        wait_errors = []
        archive = self.root / "gear.stdout.log"
        merged_to_end_diagnostics = (
            b"[ZMQEndpointInterface] active_protocol_version_=1\n"
            b"[ZMQEndpointInterface] motion name: streamed\n"
        )
        end_line = (
            "[ZMQEndpointInterface] *** End of ZMQ decoding processing ***"
        )
        try:
            gear.start_to_wait_for_control()
            gear.enable_stream_for_preload()
            boundary = gear.publication_boundary()
            end_wait_requested = threading.Event()
            wait_exact_line_range = gear._wait_exact_line_range

            def observe_exact_line_wait(line, *, after_offset):
                if line == f"{end_line}\n":
                    end_wait_requested.set()
                return wait_exact_line_range(line, after_offset=after_offset)

            gear._wait_exact_line_range = observe_exact_line_wait

            def wait_for_end():
                try:
                    marker_result.append(
                        gear.wait_for_stream_processing(
                            boundary,
                            frame_count=20,
                            global_start=1,
                            merged_count=21,
                        )
                    )
                except BaseException as error:  # noqa: BLE001
                    wait_errors.append(error)

            waiter = threading.Thread(target=wait_for_end)
            waiter.start()
            gear.write_keys(b"x")
            self.assertTrue(end_wait_requested.wait(timeout=1.0))

            deadline = time.monotonic() + 1.0
            while (
                merged_to_end_diagnostics not in archive.read_bytes()
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            # The full Start/processing/Merged sequence plus a post-merge
            # diagnostic is now archived, but the pinned End line is still
            # withheld behind the second PTY byte.
            self.assertIn(merged_to_end_diagnostics, archive.read_bytes())
            self.assertTrue(waiter.is_alive())
            self.assertEqual(marker_result, [])
            self.assertEqual(wait_errors, [])

            gear.write_keys(b"\n")
            waiter.join(timeout=1.0)
            self.assertFalse(waiter.is_alive())
            self.assertEqual(wait_errors, [])

            marker = marker_result[0]
            archived = archive.read_bytes()
            end_start, end_end = marker["end_range"]
            self.assertEqual(
                marker["end_line"],
                end_line,
            )
            self.assertEqual(
                archived[end_start:end_end],
                f"{marker['end_line']}\n".encode("ascii"),
            )
            self.assertEqual(marker["end_offset"], end_end)
            self.assertLessEqual(
                marker["start_range"][1], marker["processing_range"][0]
            )
            self.assertLessEqual(
                marker["processing_range"][1], marker["merged_range"][0]
            )
            self.assertLess(marker["merged_range"][1], end_start)
            self.assertEqual(
                archived[marker["merged_range"][1] : end_start],
                merged_to_end_diagnostics,
            )
        finally:
            try:
                gear.close()
            finally:
                if waiter is not None:
                    waiter.join(timeout=1.0)

    def test_rejects_conflate_or_duplicate_managed_flags(self):
        child = self.script("unused.py", "raise SystemExit(0)\n")
        for flag in (
            "--zmq-conflate",
            "--input-type",
            "--logs-dir",
            "--zmq-conflate=true",
            "--input-type=file",
            "--target-motion-logfile=/tmp/override.csv",
            "--logs-dir=/tmp/override",
            "--enable-csv-logs=false",
            "--disable-crc-check=false",
            "--zmq-verbose=false",
        ):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, "managed GEAR flag"):
                    GearProcess(
                        run_root=self.root,
                        command=[sys.executable, str(child), flag],
                        target_motion_logfile=self.root / "target.csv",
                        logs_dir=self.root / "logs",
                        stdout_archive=self.root / "out",
                        stderr_archive=self.root / "err",
                    )

    def test_failed_exec_closes_pty_and_archive_descriptors(self):
        gear = GearProcess(
            run_root=self.root,
            command=[str(self.root / "missing-gear-executable")],
            target_motion_logfile=self.root / "target.csv",
            logs_dir=self.root / "logs",
            stdout_archive=self.root / "out",
            stderr_archive=self.root / "err",
        )
        try:
            with self.assertRaises(FileNotFoundError):
                gear.start()
            self.assertIsNone(gear._master_fd)
            self.assertTrue(gear._stdout_file.closed)
            self.assertTrue(gear._stderr_file.closed)
            with self.assertRaisesRegex(ProcessError, "closed"):
                gear.start()
        finally:
            gear.close()

    def test_run_root_rejects_escape_symlink_and_preexisting_outputs(self):
        child = self.script("unused_gear.py", "raise SystemExit(0)\n")
        with tempfile.TemporaryDirectory() as outside_text:
            outside = Path(outside_text)
            with self.assertRaisesRegex(ProcessError, "run_root"):
                GearProcess(
                    run_root=self.root,
                    command=[sys.executable, str(child)],
                    target_motion_logfile=outside / "target.csv",
                    logs_dir=self.root / "logs-a",
                    stdout_archive=self.root / "out-a",
                    stderr_archive=self.root / "err-a",
                )

            (self.root / "linked").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ProcessError, "symlink"):
                GearProcess(
                    run_root=self.root,
                    command=[sys.executable, str(child)],
                    target_motion_logfile=self.root / "linked" / "target.csv",
                    logs_dir=self.root / "logs-b",
                    stdout_archive=self.root / "out-b",
                    stderr_archive=self.root / "err-b",
                )

            preexisting_logs = self.root / "preexisting-logs"
            preexisting_logs.mkdir()
            marker = preexisting_logs / "keep"
            marker.write_text("evidence", encoding="utf-8")
            with self.assertRaisesRegex(ProcessError, "exists"):
                GearProcess(
                    run_root=self.root,
                    command=[sys.executable, str(child)],
                    target_motion_logfile=self.root / "target-c.csv",
                    logs_dir=preexisting_logs,
                    stdout_archive=self.root / "out-c",
                    stderr_archive=self.root / "err-c",
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "evidence")

    def test_rejects_shell_unsafe_run_root_and_logs_before_any_spawn_or_write(
        self,
    ):
        child = self.script("unused_shell_path.py", "raise SystemExit(0)\n")
        sentinel = self.root / "shell-injection-sentinel"
        for name in ("run root", "run;touch-shell-injection-sentinel"):
            with self.subTest(name=name):
                unsafe_root = self.root / name
                unsafe_root.mkdir()
                with patch("mm_sonic.process.subprocess.Popen") as popen:
                    with self.assertRaisesRegex(ProcessError, "shell-safe"):
                        GearProcess(
                            run_root=unsafe_root,
                            command=[sys.executable, str(child)],
                            target_motion_logfile=unsafe_root / "target.csv",
                            logs_dir=unsafe_root / "gear-logs",
                            stdout_archive=unsafe_root / "out",
                            stderr_archive=unsafe_root / "err",
                        )
                popen.assert_not_called()
                self.assertEqual(tuple(unsafe_root.iterdir()), ())
                self.assertFalse(sentinel.exists())

    def test_rejects_shell_unsafe_logs_path_before_creating_it(self):
        child = self.script("unused_shell_logs.py", "raise SystemExit(0)\n")
        unsafe_logs = self.root / "gear logs;touch-owned"
        with patch("mm_sonic.process.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ProcessError, "shell-safe"):
                GearProcess(
                    run_root=self.root,
                    command=[sys.executable, str(child)],
                    target_motion_logfile=self.root / "target-safe.csv",
                    logs_dir=unsafe_logs,
                    stdout_archive=self.root / "out-safe",
                    stderr_archive=self.root / "err-safe",
                )
        popen.assert_not_called()
        self.assertFalse(unsafe_logs.exists())

    def test_cleanup_escalates_term_then_kill_and_leaves_no_group_orphan(self):
        grandchild_path = self.root / "grandchild.pid"
        child = self.script(
            "stubborn.py",
            r'''
            import os
            from pathlib import Path
            import signal
            import subprocess
            import sys
            import termios
            import time

            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            grandchild = subprocess.Popen([
                sys.executable,
                "-u",
                "-c",
                "import signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "time.sleep(60)",
            ])
            Path(os.environ["GRANDCHILD_PATH"]).write_text(str(grandchild.pid))
            print("BOOT READY", flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            os.read(0, 1)
            print("CONTROL READY", flush=True)
            os.read(0, 1)
            print("STREAM READY", flush=True)
            while True:
                time.sleep(1)
            ''',
        )
        env = dict(os.environ, GRANDCHILD_PATH=str(grandchild_path))
        gear = self.gear(child, env=env)
        gear.start()
        pid = gear.pid
        pgid = gear.pgid
        grandchild = int(grandchild_path.read_text(encoding="utf-8"))
        gear.close()

        self.assertEqual(
            gear.cleanup_history,
            ["official-stop-key", "SIGTERM", "SIGKILL"],
        )
        self.assertFalse(process_exists(pid))
        self.assertFalse(process_exists(grandchild))
        with self.assertRaises(ProcessLookupError):
            os.killpg(pgid, 0)

    def test_stop_group_stops_descendant_even_after_verified_leader_dies(self):
        descendant_path = self.root / "leaderless-descendant.pid"
        child = self.script(
            "leader_dies.py",
            r'''
            import os
            from pathlib import Path
            import subprocess
            import sys
            import termios

            print("BOOT READY", flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            os.read(0, 1)
            print("CONTROL READY", flush=True)
            os.read(0, 1)
            print("STREAM READY", flush=True)
            descendant = subprocess.Popen([
                sys.executable, "-u", "-c", "import time; time.sleep(60)"
            ])
            Path(os.environ["DESCENDANT_PATH"]).write_text(str(descendant.pid))
            ''',
        )
        gear = self.gear(
            child,
            env=dict(os.environ, DESCENDANT_PATH=str(descendant_path)),
        )
        gear.start()
        pgid = gear.pgid
        deadline = time.monotonic() + 1.0
        while gear.returncode is None and time.monotonic() < deadline:
            time.sleep(0.01)
        descendant = int(descendant_path.read_text(encoding="utf-8"))
        try:
            with self.assertRaises(ChildProcessDied):
                gear.stop_group()
            self.assertTrue(gear.group_is_stopped())
        finally:
            gear.close()
        self.assertFalse(process_exists(descendant))
        with self.assertRaises(ProcessLookupError):
            os.killpg(pgid, 0)

    def test_continue_waits_until_every_live_group_member_is_resumed(self):
        child = self.script(
            "continue_verification.py",
            r'''
            import os
            import termios
            print("BOOT READY", flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            os.read(0, 1)
            print("CONTROL READY", flush=True)
            os.read(0, 1)
            print("STREAM READY", flush=True)
            while os.read(0, 1).lower() != b"o":
                pass
            ''',
        )
        gear = self.gear(child)
        gear.start()
        gear.stop_group()
        calls = []

        def synthetic_states(_pgid):
            calls.append(True)
            if len(calls) == 1:
                return {gear.pid: "T", gear.pid + 1: "T"}
            if len(calls) < 4:
                return {gear.pid: "R", gear.pid + 1: "T"}
            return {gear.pid: "R", gear.pid + 1: "S"}

        try:
            with (
                patch(
                    "mm_sonic.process._linux_group_states",
                    side_effect=synthetic_states,
                ),
                patch.object(gear, "_send_group_signal", return_value=True),
            ):
                gear.continue_group()
            self.assertGreaterEqual(len(calls), 4)
        finally:
            gear.close()

    def test_detects_unexpected_gear_death(self):
        child = self.script(
            "dies.py",
            r'''
            import os
            import termios
            print("BOOT READY", flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            os.read(0, 1)
            print("CONTROL READY", flush=True)
            os.read(0, 1)
            print("STREAM READY", flush=True)
            raise SystemExit(23)
            ''',
        )
        gear = self.gear(child)
        try:
            gear.start()
            deadline = time.monotonic() + 1.0
            while gear.returncode is None and time.monotonic() < deadline:
                time.sleep(0.01)
            with self.assertRaisesRegex(ChildProcessDied, "23"):
                gear.require_alive()
        finally:
            gear.close()


class FakeSimulatorClient:
    def __init__(self, sim_dt=0.005, delta_error=0.0):
        self.sim_dt = sim_dt
        self.delta_error = delta_error
        self.steps = []
        self.running_checks = []
        self.gear = None
        self.closed = False

    def advance(self, steps):
        self.steps.append(steps)
        if self.gear is not None:
            self.running_checks.append(not self.gear.group_is_stopped())
        return AdvanceResult(
            steps=steps,
            sim_time_start_s=1.25,
            sim_time_end_s=1.25 + steps * self.sim_dt + self.delta_error,
            state_rows=steps // 4,
            contact_rows=steps,
        )

    def require_alive(self):
        return None

    def close(self):
        self.closed = True


class SimulationPolicyGateTests(TemporaryScriptCase):
    def setUp(self):
        super().setUp()
        self.child = self.script(
            "running.py",
            r'''
            import os
            import termios
            import time
            print("BOOT READY", flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            os.read(0, 1)
            print("CONTROL READY", flush=True)
            os.read(0, 1)
            print("STREAM READY", flush=True)
            while True:
                key = os.read(0, 1)
                if key.lower() == b"o":
                    break
                time.sleep(0.001)
            ''',
        )
        self.gear = GearProcess(
            run_root=self.root,
            command=[sys.executable, "-u", str(self.child)],
            target_motion_logfile=self.root / "target.csv",
            logs_dir=self.root / "logs",
            stdout_archive=self.root / "stdout",
            stderr_archive=self.root / "stderr",
            startup_markers=("BOOT READY",),
            active_markers=("CONTROL READY", "STREAM READY"),
            wait_for_control_marker="BOOT READY",
            readiness_poll_s=0.005,
            signal_poll_s=0.005,
            stop_grace_s=0.05,
            term_grace_s=0.05,
            kill_grace_s=0.2,
        )
        self.gear.start()

    def tearDown(self):
        self.gear.close()
        super().tearDown()

    def test_pause_and_advance_order_whole_group_around_exact_steps(self):
        simulator = FakeSimulatorClient()
        simulator.gear = self.gear
        gate = SimulationPolicyGate(self.gear, simulator)

        gate.pause()
        self.assertTrue(gate.is_paused)
        self.assertTrue(self.gear.group_is_stopped())
        result = gate.advance(0.4)

        self.assertEqual(result.steps, 80)
        self.assertEqual(simulator.steps, [80])
        self.assertEqual(simulator.running_checks, [True])
        self.assertTrue(gate.is_paused)
        self.assertTrue(self.gear.group_is_stopped())
        self.assertEqual(
            self.gear.signal_history[-3:],
            [signal.SIGSTOP, signal.SIGCONT, signal.SIGSTOP],
        )

    def test_duration_must_derive_exact_positive_integer_before_resume(self):
        simulator = FakeSimulatorClient()
        gate = SimulationPolicyGate(self.gear, simulator)
        gate.pause()
        before = list(self.gear.signal_history)
        for duration in (0.0, -0.4, 0.40000000000002, True):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(ValueError, "exact positive integer"):
                    gate.advance(duration)
        self.assertEqual(self.gear.signal_history, before)
        self.assertEqual(simulator.steps, [])

    def test_mujoco_time_is_authoritative_and_mismatch_repauses(self):
        simulator = FakeSimulatorClient(delta_error=2e-12)
        gate = SimulationPolicyGate(self.gear, simulator)
        gate.pause()
        with self.assertRaisesRegex(ProcessProtocolError, "MuJoCo time") as raised:
            gate.advance(0.4)
        self.assertEqual(raised.exception.failure_site, "simulator_advance")
        self.assertTrue(gate.is_paused)
        self.assertTrue(self.gear.group_is_stopped())

    def test_release_steps_rejects_idle_gate_and_noninteger_steps(self):
        simulator = FakeSimulatorClient()
        gate = SimulationPolicyGate(self.gear, simulator)
        with self.assertRaisesRegex(RuntimeError, "paused"):
            gate.release_steps(80)
        gate.pause()
        for steps in (0, -1, True, 1.0):
            with self.subTest(steps=steps):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    gate.release_steps(steps)

    def test_pause_stops_gear_before_reporting_simulator_death(self):
        class DeadSimulator(FakeSimulatorClient):
            def require_alive(self):
                raise ChildProcessDied("simulator already died")

        gate = SimulationPolicyGate(self.gear, DeadSimulator())
        with self.assertRaisesRegex(ChildProcessDied, "simulator"):
            gate.pause()
        self.assertTrue(self.gear.group_is_stopped())

    def test_continue_failure_is_always_guarded_by_restop(self):
        simulator = FakeSimulatorClient()
        gate = SimulationPolicyGate(self.gear, simulator)
        gate.pause()
        original_continue = self.gear.continue_group

        def resume_then_fail():
            original_continue()
            raise ProcessError("synthetic CONT verification failure")

        self.gear.continue_group = resume_then_fail
        with self.assertRaisesRegex(ProcessError, "CONT verification") as raised:
            gate.release_steps(80)
        self.assertEqual(raised.exception.failure_site, "process_resume")
        self.assertTrue(self.gear.group_is_stopped())
        self.assertEqual(simulator.steps, [])

    def test_simulator_request_failure_is_tagged_after_verified_resume(self):
        class FailingSimulator(FakeSimulatorClient):
            def advance(self, steps):
                self.steps.append(steps)
                raise ProcessProtocolError("synthetic advance failure")

        simulator = FailingSimulator()
        gate = SimulationPolicyGate(self.gear, simulator)
        gate.pause()
        with self.assertRaisesRegex(ProcessProtocolError, "advance failure") as raised:
            gate.release_steps(80)
        self.assertEqual(raised.exception.failure_site, "simulator_advance")
        self.assertTrue(gate.is_paused)
        self.assertTrue(self.gear.group_is_stopped())

    def test_gate_binds_and_rechecks_simulator_reported_sim_dt(self):
        simulator = FakeSimulatorClient(sim_dt=0.01)
        gate = SimulationPolicyGate(self.gear, simulator)
        gate.pause()

        result = gate.advance(0.4)
        self.assertEqual(result.steps, 40)
        simulator.sim_dt = 0.005
        before = list(self.gear.signal_history)
        with self.assertRaisesRegex(ProcessProtocolError, "SIMULATE_DT changed"):
            gate.advance(0.4)
        self.assertEqual(self.gear.signal_history, before)
        self.assertTrue(self.gear.group_is_stopped())


if __name__ == "__main__":
    unittest.main()
