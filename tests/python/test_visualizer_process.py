import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

from resources import g1_visualizer_process as visualizer_process


class FakeClock:
    def __init__(self, on_sleep=None):
        self.now = 0.0
        self.sleep_durations = []
        self.on_sleep = on_sleep

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.sleep_durations.append(duration)
        self.now += duration
        if self.on_sleep is not None:
            self.on_sleep(len(self.sleep_durations))


class SyntheticProcTree:
    def __init__(self, root):
        self.root = Path(root)

    def make_executable(self, name, contents=b"executable"):
        path = self.root.parent / name
        path.write_bytes(contents)
        path.chmod(0o755)
        return path.resolve()

    def add_process(self, pid, start_time, executable, cmdline):
        process = self.root / str(pid)
        process.mkdir(parents=True)
        self.set_start_time(pid, start_time)
        (process / "exe").symlink_to(Path(executable))
        (process / "cmdline").write_bytes(cmdline)

    def set_start_time(self, pid, start_time):
        process = self.root / str(pid)
        process.mkdir(parents=True, exist_ok=True)
        # The command deliberately contains whitespace and ')' so parsing must
        # locate the final command delimiter instead of splitting naively.
        fields_4_through_21 = [str(value) for value in range(4, 22)]
        stat = " ".join([
            str(pid), "(visualizer ) worker)", "S",
            *fields_4_through_21, str(start_time), "0", "0",
        ])
        (process / "stat").write_text(stat + "\n", encoding="ascii")

    def set_executable(self, pid, executable):
        link = self.root / str(pid) / "exe"
        link.unlink()
        link.symlink_to(Path(executable))

    def set_cmdline(self, pid, cmdline):
        (self.root / str(pid) / "cmdline").write_bytes(cmdline)

    def remove_process(self, pid):
        process = self.root / str(pid)
        for entry in process.iterdir():
            entry.unlink()
        process.rmdir()


class VisualizerProcessTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.proc_root = Path(self.temporary.name) / "proc"
        self.proc_root.mkdir()
        self.proc = SyntheticProcTree(self.proc_root)
        self.target = self.proc.make_executable("controller-target")
        self.other = self.proc.make_executable("controller-other")

    def identity_for(self, pid=123, start_time=456,
                     cmdline=b"controller\0--scene\0mixed\xff"):
        self.proc.add_process(pid, start_time, self.target, cmdline)
        return visualizer_process.read_process_identity(
            pid, proc_root=self.proc_root)

    def assert_no_signal_after_mutation(self, mutate):
        identity = self.identity_for()
        mutate(identity)
        calls = []
        with self.assertRaises(visualizer_process.ProcessIdentityError):
            visualizer_process.signal_identity(
                identity,
                "TERM",
                proc_root=self.proc_root,
                kill_callback=lambda pid, signum: calls.append((pid, signum)),
            )
        self.assertEqual(calls, [])

    def test_parse_proc_stat_reads_field_22_with_adversarial_command(self):
        self.proc.add_process(
            812, 998877, self.target, b"controller\0")
        raw = (self.proc_root / "812" / "stat").read_bytes()

        self.assertEqual(
            visualizer_process.parse_proc_stat_start_time(
                raw, expected_pid=812),
            998877,
        )
        with self.assertRaises(ValueError):
            visualizer_process.parse_proc_stat_start_time(
                raw, expected_pid=813)
        with self.assertRaises(ValueError):
            visualizer_process.parse_proc_stat_start_time(b"812 (bad) S")

    def test_full_identity_rejects_pid_reuse_during_the_read(self):
        self.proc.add_process(813, 1000, self.target, b"controller\0")

        with mock.patch.object(
                visualizer_process,
                "_read_start_time",
                side_effect=[1000, 1001]) as start_time_reader:
            with self.assertRaisesRegex(
                    visualizer_process.ProcessIdentityError,
                    "changed while reading identity"):
                visualizer_process.read_process_identity(
                    813, proc_root=self.proc_root)
        self.assertEqual(start_time_reader.call_count, 2)

    def test_parse_and_compare_identity_are_strict_and_pure(self):
        identity = self.identity_for(cmdline=b"controller\0--raw\0\xff")
        document = visualizer_process.canonical_identity_bytes(identity)

        parsed = visualizer_process.parse_identity(document)
        self.assertEqual(parsed, identity)
        self.assertIsNot(parsed, identity)
        self.assertTrue(visualizer_process.compare_identities(parsed, identity))

        changed = copy.deepcopy(identity)
        changed["cmdline_hex"] = b"controller\0--changed\0".hex()
        self.assertFalse(
            visualizer_process.compare_identities(identity, changed))

        malformed = copy.deepcopy(identity)
        malformed["extra"] = 1
        with self.assertRaises(ValueError):
            visualizer_process.parse_identity(malformed)
        malformed = copy.deepcopy(identity)
        malformed["pid"] = True
        with self.assertRaises(ValueError):
            visualizer_process.parse_identity(malformed)
        malformed = copy.deepcopy(identity)
        malformed["cmdline_hex"] = "FF"
        with self.assertRaises(ValueError):
            visualizer_process.parse_identity(malformed)
        with self.assertRaises(ValueError):
            visualizer_process.parse_identity(
                '{"pid":1,"pid":2}')

    def test_canonical_documents_are_sorted_compact_and_newline_terminated(self):
        identity = self.identity_for(pid=7, start_time=8,
                                     cmdline=b"raw\0bytes\xff")
        expected = (
            json.dumps(identity, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("ascii")

        self.assertEqual(
            visualizer_process.canonical_identity_bytes(identity), expected)
        self.assertEqual(
            visualizer_process.canonical_identity_bytes(None), b"null\n")

        output = Path(self.temporary.name) / "identity.json"
        visualizer_process.write_identity(output, identity)
        self.assertEqual(output.read_bytes(), expected)

    def test_discover_requires_an_absolute_resolved_executable(self):
        with self.assertRaises(ValueError):
            visualizer_process.discover(
                "controller-target", proc_root=self.proc_root)

        alias = Path(self.temporary.name) / "controller-alias"
        alias.symlink_to(self.target)
        self.proc.add_process(14, 90, self.target, b"controller\0")
        identity = visualizer_process.discover(
            alias.absolute(), proc_root=self.proc_root)
        self.assertEqual(identity["executable"], str(self.target))

    def test_discover_returns_none_for_zero_exact_matches(self):
        self.proc.add_process(10, 100, self.other, b"other\0")

        self.assertIsNone(visualizer_process.discover(
            self.target, proc_root=self.proc_root))

    def test_discover_returns_full_raw_identity_for_one_exact_match(self):
        cmdline = b"controller\0--terrain\0mixed\xff\0"
        self.proc.add_process(123, 456, self.target, cmdline)
        executable_stat = self.target.stat()

        self.assertEqual(
            visualizer_process.discover(
                self.target, proc_root=self.proc_root),
            {
                "pid": 123,
                "start_time": 456,
                "executable": str(self.target),
                "executable_device": executable_stat.st_dev,
                "executable_inode": executable_stat.st_ino,
                "cmdline_hex": cmdline.hex(),
            },
        )

    def test_discover_rejects_more_than_one_exact_match(self):
        self.proc.add_process(31, 301, self.target, b"first\0")
        self.proc.add_process(32, 302, self.target, b"second\0")

        with self.assertRaisesRegex(
                visualizer_process.ProcessIdentityError,
                "more than one"):
            visualizer_process.discover(
                self.target, proc_root=self.proc_root)

    def test_discover_rejects_torn_identity_during_exec(self):
        self.proc.add_process(33, 303, self.target, b"before-exec\0")
        before = visualizer_process.read_process_identity(
            33, proc_root=self.proc_root)
        after = copy.deepcopy(before)
        after["executable"] = str(self.other)
        after["executable_inode"] = self.other.stat().st_ino
        after["cmdline_hex"] = b"after-exec\0".hex()
        torn = copy.deepcopy(after)
        torn["executable"] = str(self.target)

        with mock.patch.object(
                visualizer_process,
                "read_process_identity",
                side_effect=[torn, after]) as identity_reader:
            with self.assertRaisesRegex(
                    visualizer_process.ProcessIdentityError,
                    "changed during discovery"):
                visualizer_process.discover(
                    self.target, proc_root=self.proc_root)
        self.assertEqual(identity_reader.call_count, 2)

    def test_capture_wait_rejects_a_missing_pid_immediately(self):
        clock = FakeClock()

        with self.assertRaisesRegex(
                visualizer_process.ProcessIdentityError, "missing"):
            visualizer_process.capture_wait(
                404,
                self.target,
                timeout_ms=2000,
                proc_root=self.proc_root,
                monotonic=clock.monotonic,
                sleeper=clock.sleep,
            )
        self.assertEqual(clock.sleep_durations, [])

    def test_capture_wait_times_out_without_oversleeping_on_wrong_executable(self):
        self.proc.add_process(41, 410, self.other, b"other\0")
        clock = FakeClock()

        with self.assertRaises(visualizer_process.ProcessCaptureTimeout):
            visualizer_process.capture_wait(
                41,
                self.target,
                timeout_ms=25,
                proc_root=self.proc_root,
                monotonic=clock.monotonic,
                sleeper=clock.sleep,
                poll_interval_ms=10,
            )

        self.assertAlmostEqual(sum(clock.sleep_durations), 0.025)
        self.assertLessEqual(max(clock.sleep_durations), 0.010)

    def test_capture_wait_rejects_timeout_outside_fixed_bound(self):
        self.proc.add_process(42, 420, self.other, b"other\0")

        for timeout_ms in (-1,
                           visualizer_process.MAX_CAPTURE_TIMEOUT_MS + 1):
            with self.subTest(timeout_ms=timeout_ms):
                with self.assertRaises(ValueError):
                    visualizer_process.capture_wait(
                        42,
                        self.target,
                        timeout_ms=timeout_ms,
                        proc_root=self.proc_root,
                    )

    def test_capture_wait_rejects_disappearance_of_the_pinned_process(self):
        self.proc.add_process(51, 510, self.other, b"other\0")
        clock = FakeClock(
            on_sleep=lambda unused: self.proc.remove_process(51))

        with self.assertRaisesRegex(
                visualizer_process.ProcessIdentityError, "disappeared"):
            visualizer_process.capture_wait(
                51,
                self.target,
                timeout_ms=100,
                proc_root=self.proc_root,
                monotonic=clock.monotonic,
                sleeper=clock.sleep,
            )

    def test_capture_wait_rejects_changed_pinned_start_time_during_exec(self):
        self.proc.add_process(61, 610, self.other, b"launcher\0")

        def simulate_replacement(unused):
            self.proc.set_start_time(61, 611)
            self.proc.set_executable(61, self.target)
            self.proc.set_cmdline(61, b"controller\0")

        clock = FakeClock(on_sleep=simulate_replacement)
        with self.assertRaisesRegex(
                visualizer_process.ProcessIdentityError,
                "start time changed"):
            visualizer_process.capture_wait(
                61,
                self.target,
                timeout_ms=100,
                proc_root=self.proc_root,
                monotonic=clock.monotonic,
                sleeper=clock.sleep,
            )

    def test_capture_wait_never_accepts_a_matching_later_pid(self):
        self.proc.add_process(62, 620, self.other, b"launcher\0")

        def add_unrelated_match(sleep_count):
            if sleep_count == 1:
                self.proc.add_process(
                    63, 630, self.target, b"controller\0")

        clock = FakeClock(on_sleep=add_unrelated_match)
        with self.assertRaises(visualizer_process.ProcessCaptureTimeout):
            visualizer_process.capture_wait(
                62,
                self.target,
                timeout_ms=25,
                proc_root=self.proc_root,
                monotonic=clock.monotonic,
                sleeper=clock.sleep,
                poll_interval_ms=10,
            )

    def test_capture_wait_matches_discover_after_unchanged_exec(self):
        self.proc.add_process(71, 710, self.other, b"launcher\0--wait\0")

        def simulate_exec(unused):
            self.proc.set_executable(71, self.target)
            self.proc.set_cmdline(71, b"controller\0--terrain\0mixed\0")

        clock = FakeClock(on_sleep=simulate_exec)
        captured = visualizer_process.capture_wait(
            71,
            self.target,
            timeout_ms=100,
            proc_root=self.proc_root,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
        )
        discovered = visualizer_process.discover(
            self.target, proc_root=self.proc_root)

        self.assertTrue(visualizer_process.compare_identities(
            captured, discovered))
        self.assertEqual(
            visualizer_process.canonical_identity_bytes(captured),
            visualizer_process.canonical_identity_bytes(discovered),
        )

    def test_capture_wait_rejects_a_change_between_full_identity_reads(self):
        self.proc.add_process(81, 810, self.target, b"controller\0")
        first = visualizer_process.discover(
            self.target, proc_root=self.proc_root)
        second = copy.deepcopy(first)
        second["cmdline_hex"] = b"controller\0--mutated\0".hex()

        with mock.patch.object(
                visualizer_process,
                "read_process_identity",
                side_effect=[first, second]) as reader:
            with self.assertRaisesRegex(
                    visualizer_process.ProcessIdentityError,
                    "changed between identity reads"):
                visualizer_process.capture_wait(
                    81,
                    self.target,
                    timeout_ms=0,
                    proc_root=self.proc_root,
                )
        self.assertEqual(reader.call_count, 2)

    def test_signal_revalidates_identity_and_maps_only_named_signals(self):
        signal_names = {
            "STOP": signal.SIGSTOP,
            "CONT": signal.SIGCONT,
            "TERM": signal.SIGTERM,
            "KILL": signal.SIGKILL,
        }
        for offset, (name, expected_signal) in enumerate(
                signal_names.items(), start=1):
            with self.subTest(name=name):
                identity = self.identity_for(
                    pid=100 + offset, start_time=200 + offset)
                calls = []
                self.assertTrue(visualizer_process.signal_identity(
                    identity,
                    name,
                    proc_root=self.proc_root,
                    kill_callback=lambda pid, signum: calls.append(
                        (pid, signum)),
                ))
                self.assertEqual(
                    calls, [(100 + offset, expected_signal)])

        with self.assertRaises(ValueError):
            visualizer_process.signal_identity(
                None, "USR1", proc_root=self.proc_root,
                kill_callback=lambda unused_pid, unused_signal: None)

    def test_signal_treats_null_or_vanished_identity_as_successful_noop(self):
        calls = []
        self.assertFalse(visualizer_process.signal_identity(
            None,
            "TERM",
            proc_root=self.proc_root,
            kill_callback=lambda pid, signum: calls.append((pid, signum)),
        ))

        identity = self.identity_for(pid=901, start_time=902)
        self.proc.remove_process(901)
        self.assertFalse(visualizer_process.signal_identity(
            identity,
            "TERM",
            proc_root=self.proc_root,
            kill_callback=lambda pid, signum: calls.append((pid, signum)),
        ))
        self.assertEqual(calls, [])

    def test_signal_rejects_changed_start_time_before_kill(self):
        self.assert_no_signal_after_mutation(
            lambda unused: self.proc.set_start_time(123, 457))

    def test_signal_rejects_changed_executable_path_before_kill(self):
        alias = Path(self.temporary.name) / "controller-hardlink"
        os.link(self.target, alias)
        target_stat = self.target.stat()
        alias_stat = alias.stat()
        self.assertEqual(target_stat.st_dev, alias_stat.st_dev)
        self.assertEqual(target_stat.st_ino, alias_stat.st_ino)

        self.assert_no_signal_after_mutation(
            lambda unused: self.proc.set_executable(123, alias))

    def test_signal_rejects_changed_executable_inode_before_kill(self):
        def replace_executable(identity):
            replacement = self.proc.make_executable(
                "controller-replacement", b"new inode")
            os.replace(replacement, self.target)
            replaced_stat = self.target.stat()
            self.assertEqual(
                replaced_stat.st_dev, identity["executable_device"])
            self.assertNotEqual(
                replaced_stat.st_ino, identity["executable_inode"])

        self.assert_no_signal_after_mutation(replace_executable)

    def test_signal_rejects_changed_executable_device_before_kill(self):
        identity = self.identity_for()
        changed = copy.deepcopy(identity)
        changed["executable_device"] += 1
        calls = []

        with mock.patch.object(
                visualizer_process,
                "read_process_identity",
                return_value=changed) as identity_reader:
            with self.assertRaises(visualizer_process.ProcessIdentityError):
                visualizer_process.signal_identity(
                    identity,
                    "TERM",
                    proc_root=self.proc_root,
                    kill_callback=lambda pid, signum: calls.append(
                        (pid, signum)),
                )
        self.assertEqual(identity_reader.call_count, 2)
        self.assertEqual(calls, [])

    def test_signal_rejects_changed_raw_command_bytes_before_kill(self):
        self.assert_no_signal_after_mutation(
            lambda unused: self.proc.set_cmdline(
                123, b"controller\0--different\0"))

    def test_signal_rejects_change_between_validation_reads_before_kill(self):
        identity = self.identity_for()
        changed = copy.deepcopy(identity)
        changed["cmdline_hex"] = b"controller\0--changed-during-read\0".hex()
        calls = []

        with mock.patch.object(
                visualizer_process,
                "read_process_identity",
                side_effect=[identity, changed]) as reader:
            with self.assertRaises(visualizer_process.ProcessIdentityError):
                visualizer_process.signal_identity(
                    identity,
                    "TERM",
                    proc_root=self.proc_root,
                    kill_callback=lambda pid, signum: calls.append(
                        (pid, signum)),
                )
        self.assertEqual(reader.call_count, 2)
        self.assertEqual(calls, [])

    def test_signal_rejects_missing_live_field_before_kill(self):
        self.assert_no_signal_after_mutation(
            lambda unused: (self.proc_root / "123" / "cmdline").unlink())

    def test_signal_rejects_malformed_saved_identity_before_kill(self):
        identity = self.identity_for()
        del identity["executable_inode"]
        calls = []

        with self.assertRaises(ValueError):
            visualizer_process.signal_identity(
                identity,
                "TERM",
                proc_root=self.proc_root,
                kill_callback=lambda pid, signum: calls.append((pid, signum)),
            )
        self.assertEqual(calls, [])

    def test_signal_treats_disappearance_at_kill_as_successful_noop(self):
        identity = self.identity_for()
        calls = []

        def vanished_kill(pid, signum):
            calls.append((pid, signum))
            raise ProcessLookupError(pid)

        self.assertFalse(visualizer_process.signal_identity(
            identity,
            "TERM",
            proc_root=self.proc_root,
            kill_callback=vanished_kill,
        ))
        self.assertEqual(calls, [(123, signal.SIGTERM)])

    def test_signal_file_reads_canonical_null_and_identity_documents(self):
        output = Path(self.temporary.name) / "saved.json"
        visualizer_process.write_identity(output, None)
        calls = []
        self.assertFalse(visualizer_process.signal_identity_file(
            output,
            "STOP",
            proc_root=self.proc_root,
            kill_callback=lambda pid, signum: calls.append((pid, signum)),
        ))

        identity = self.identity_for()
        visualizer_process.write_identity(output, identity)
        self.assertTrue(visualizer_process.signal_identity_file(
            output,
            "CONT",
            proc_root=self.proc_root,
            kill_callback=lambda pid, signum: calls.append((pid, signum)),
        ))
        self.assertEqual(calls, [(123, signal.SIGCONT)])

    def test_cli_fixes_proc_root_and_kill_callback(self):
        output = Path(self.temporary.name) / "cli.json"
        identity_path = Path(self.temporary.name) / "input.json"
        visualizer_process.write_identity(identity_path, None)

        with mock.patch.object(
                visualizer_process, "discover", return_value=None) as discover:
            self.assertEqual(visualizer_process.main([
                "discover", "--executable", str(self.target),
                "--output", str(output),
            ]), 0)
        self.assertEqual(discover.call_args.kwargs["proc_root"], Path("/proc"))
        self.assertEqual(output.read_bytes(), b"null\n")

        captured_identity = self.identity_for()
        with mock.patch.object(
                visualizer_process,
                "capture_wait",
                return_value=captured_identity) as capture_wait:
            self.assertEqual(visualizer_process.main([
                "capture-wait", "--pid", "123",
                "--executable", str(self.target),
                "--timeout-ms", "2000", "--output", str(output),
            ]), 0)
        self.assertEqual(
            capture_wait.call_args.kwargs["proc_root"], Path("/proc"))
        self.assertEqual(
            output.read_bytes(),
            visualizer_process.canonical_identity_bytes(captured_identity),
        )

        with mock.patch.object(
                visualizer_process,
                "signal_identity_file",
                return_value=False) as signal_file:
            self.assertEqual(visualizer_process.main([
                "signal", "--identity", str(identity_path),
                "--signal", "TERM",
            ]), 0)
        self.assertEqual(
            signal_file.call_args.kwargs["proc_root"], Path("/proc"))
        self.assertIs(signal_file.call_args.kwargs["kill_callback"], os.kill)


if __name__ == "__main__":
    unittest.main()
