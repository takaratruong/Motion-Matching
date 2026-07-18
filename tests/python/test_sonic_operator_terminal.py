from __future__ import annotations

import os
import pty
import termios
import time
import unittest

from mm_sonic.joints import ContractError
from mm_sonic.operator import OperatorState
from mm_sonic.operator_terminal import (
    TerminalInputReader,
    TerminalKeyBuffer,
    operator_state_from_keys,
)


def _sample_until(buffer: TerminalKeyBuffer, timeout: float = 1.0) -> OperatorState:
    """Poll the buffer until the reader thread has fed a non-empty state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = buffer.sample()
        if state != OperatorState():
            return state
        time.sleep(0.005)
    return OperatorState()


class OperatorStateFromKeysTests(unittest.TestCase):
    def test_case_insensitive_keys_map_to_complete_state(self) -> None:
        state = operator_state_from_keys("wSaD qeX")
        self.assertEqual(
            (
                state.forward,
                state.backward,
                state.left,
                state.right,
                state.heading_left,
                state.heading_right,
                state.stand,
                state.terminate,
            ),
            (True,) * 8,
        )
        self.assertIsInstance(state, OperatorState)

    def test_repetition_is_idempotent(self) -> None:
        self.assertEqual(
            operator_state_from_keys("www"),
            operator_state_from_keys("w"),
        )

    def test_carriage_return_and_line_feed_are_ignored(self) -> None:
        self.assertEqual(
            operator_state_from_keys("\r\nw\r\n"),
            OperatorState(forward=True),
        )
        self.assertEqual(operator_state_from_keys("\r\n"), OperatorState())

    def test_unknown_character_raises_contract_error(self) -> None:
        with self.assertRaises(ContractError):
            operator_state_from_keys("z")


class TerminalKeyBufferTests(unittest.TestCase):
    def test_multiple_feeds_union_and_sample_clears(self) -> None:
        buffer = TerminalKeyBuffer()
        buffer.feed("w")
        buffer.feed("d")
        self.assertEqual(
            buffer.sample(),
            OperatorState(forward=True, right=True),
        )
        self.assertEqual(buffer.sample(), OperatorState())

    def test_invalid_feed_is_atomic_and_leaves_prior_keys_intact(self) -> None:
        buffer = TerminalKeyBuffer()
        buffer.feed("w")
        with self.assertRaises(ContractError):
            buffer.feed("dz")
        self.assertEqual(buffer.sample(), OperatorState(forward=True))


class TerminalInputReaderTests(unittest.TestCase):
    def test_rejects_non_tty_before_starting_thread(self) -> None:
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, read_fd)
        self.addCleanup(os.close, write_fd)
        buffer = TerminalKeyBuffer()
        with self.assertRaises(ContractError):
            TerminalInputReader(read_fd, buffer)

    def test_feeds_captured_keys_and_restores_attributes(self) -> None:
        controller_fd, follower_fd = pty.openpty()
        self.addCleanup(os.close, controller_fd)
        self.addCleanup(os.close, follower_fd)
        original = termios.tcgetattr(follower_fd)
        buffer = TerminalKeyBuffer()

        with TerminalInputReader(follower_fd, buffer):
            os.write(controller_fd, b"w")
            state = _sample_until(buffer)
            self.assertEqual(state, OperatorState(forward=True))
            self.assertNotEqual(termios.tcgetattr(follower_fd), original)

        self.assertEqual(termios.tcgetattr(follower_fd), original)

    def test_restores_attributes_on_exception(self) -> None:
        controller_fd, follower_fd = pty.openpty()
        self.addCleanup(os.close, controller_fd)
        self.addCleanup(os.close, follower_fd)
        original = termios.tcgetattr(follower_fd)
        buffer = TerminalKeyBuffer()

        with self.assertRaises(RuntimeError):
            with TerminalInputReader(follower_fd, buffer):
                raise RuntimeError("boom")

        self.assertEqual(termios.tcgetattr(follower_fd), original)

    def test_surfaces_reader_failure_to_caller(self) -> None:
        controller_fd, follower_fd = pty.openpty()
        self.addCleanup(os.close, follower_fd)
        buffer = TerminalKeyBuffer()

        with self.assertRaises(ContractError):
            with TerminalInputReader(follower_fd, buffer) as reader:
                os.close(controller_fd)
                reader.wait_closed(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
