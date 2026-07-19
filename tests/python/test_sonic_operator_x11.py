from __future__ import annotations

import os
import threading
import time
import unittest

from mm_sonic.commands import CommandSample
from mm_sonic.holden_control import (
    CameraState,
    HoldenControlMapper,
    MappedControlState,
    NormalizedControlState,
)
from mm_sonic.joints import ContractError
from mm_sonic.operator_x11 import (
    KEYSYMS,
    BoundaryControlMailbox,
    ContinuousControlLoop,
    KeyLevels,
    X11KeyStateProvider,
    normalized_state_from_pressed,
)


class FakeProvider:
    def __init__(self, frames: list[KeyLevels]) -> None:
        self.frames = frames

    def sample(self) -> KeyLevels:
        return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]


class SteadyProvider:
    """Return one healthy sample forever without raising or exhausting."""

    def __init__(self, levels: KeyLevels) -> None:
        self._levels = levels

    def sample(self) -> KeyLevels:
        return self._levels


class RaisingProvider:
    def sample(self) -> KeyLevels:
        raise RuntimeError("device gone")


class AbsentProvider:
    def sample(self) -> None:
        return None


def _mapper() -> HoldenControlMapper:
    return HoldenControlMapper(initial_heading_yaw_rad=0.0)


class KeysymRegistryTests(unittest.TestCase):
    def test_registry_has_exact_frozen_keysyms(self) -> None:
        self.assertEqual(
            KEYSYMS,
            {
                "W": 0x0077,
                "A": 0x0061,
                "S": 0x0073,
                "D": 0x0064,
                "Q": 0x0071,
                "E": 0x0065,
                "X": 0x0078,
                "SPACE": 0x0020,
                "LEFT": 0xFF51,
                "UP": 0xFF52,
                "RIGHT": 0xFF53,
                "DOWN": 0xFF54,
                "LEFT_SHIFT": 0xFFE1,
                "LEFT_CTRL": 0xFFE3,
            },
        )


class NormalizedStateFromPressedTests(unittest.TestCase):
    def test_maps_pressed_keys_to_normalized_axes(self) -> None:
        state = normalized_state_from_pressed(
            frozenset({"D", "S", "RIGHT", "DOWN", "LEFT_CTRL", "LEFT_SHIFT", "Q", "SPACE", "X"})
        )
        self.assertEqual(state.left_x, 1.0)
        self.assertEqual(state.left_z, 1.0)
        self.assertEqual(state.right_x, 1.0)
        self.assertEqual(state.right_z, 1.0)
        self.assertTrue(state.strafe)
        self.assertTrue(state.walk)
        self.assertEqual(state.zoom, 1.0)
        self.assertTrue(state.stand)
        self.assertTrue(state.terminate)

    def test_opposed_keys_cancel(self) -> None:
        state = normalized_state_from_pressed(frozenset({"W", "S", "A", "D"}))
        self.assertEqual(state.left_x, 0.0)
        self.assertEqual(state.left_z, 0.0)

    def test_empty_pressed_is_neutral(self) -> None:
        self.assertEqual(normalized_state_from_pressed(frozenset()), NormalizedControlState())


class KeyLevelsTests(unittest.TestCase):
    def test_rejects_unknown_key(self) -> None:
        with self.assertRaises(ContractError):
            KeyLevels(focused=True, pressed=frozenset({"Z"}))

    def test_rejects_nonbool_focus(self) -> None:
        with self.assertRaises(ContractError):
            KeyLevels(focused=1, pressed=frozenset())


class ContinuousControlLoopTests(unittest.TestCase):
    def test_hold_release_and_focus_loss_publish_exact_transitions(self) -> None:
        events: list[str] = []
        provider = FakeProvider([
            KeyLevels(focused=True, pressed=frozenset({"W"})),
            KeyLevels(focused=True, pressed=frozenset({"W", "LEFT_CTRL"})),
            KeyLevels(focused=False, pressed=frozenset()),
        ])
        loop = ContinuousControlLoop(
            provider,
            _mapper(),
            event_sink=events.append,
            period_s=0.001,
        )
        with loop:
            self.assertTrue(loop.wait_for_sequence(3, timeout_s=1.0))
        self.assertIn("KEY W DOWN -> forward", events)
        self.assertIn("KEY LEFT_CTRL DOWN -> strafe", events)
        self.assertIn("FOCUS LOST -> neutral", events)

    def test_healthy_unfocused_samples_keep_loop_alive_and_neutral(self) -> None:
        # Regression (decision-001): a successful focused=False sample is fresh
        # input, not a stale/absent sample. Many such samples over >0.1s must
        # keep the loop alive and continuously publish neutral state.
        provider = SteadyProvider(KeyLevels(focused=False, pressed=frozenset()))
        loop = ContinuousControlLoop(provider, _mapper(), period_s=0.001)
        with loop:
            self.assertTrue(loop.wait_for_sequence(150, timeout_s=2.0))
            command, mapped = loop.mailbox.sample(0)
        self.assertIsNotNone(command)
        self.assertEqual(command.requested_velocity_mujoco, (0.0, 0.0, 0.0))
        self.assertEqual(mapped.velocity_mujoco, (0.0, 0.0, 0.0))

    def test_provider_exception_is_fatal(self) -> None:
        loop = ContinuousControlLoop(RaisingProvider(), _mapper(), period_s=0.001)
        with self.assertRaises(ContractError):
            with loop:
                loop.wait_for_sequence(1, timeout_s=1.0)

    def test_absent_samples_beyond_staleness_are_fatal(self) -> None:
        loop = ContinuousControlLoop(AbsentProvider(), _mapper(), period_s=0.001)
        with self.assertRaises(ContractError):
            with loop:
                time.sleep(0.2)

    def test_x_rising_edge_sets_cancel_event(self) -> None:
        event = threading.Event()
        provider = FakeProvider([
            KeyLevels(focused=True, pressed=frozenset()),
            KeyLevels(focused=True, pressed=frozenset({"X"})),
        ])
        loop = ContinuousControlLoop(
            provider,
            _mapper(),
            cancel_event=event,
            period_s=0.001,
        )
        with loop:
            self.assertTrue(loop.wait_for_sequence(2, timeout_s=1.0))
            self.assertTrue(event.wait(timeout=1.0))
        self.assertTrue(event.is_set())


class BoundaryControlMailboxTests(unittest.TestCase):
    @staticmethod
    def _mapped(*, stand: bool, terminate: bool) -> MappedControlState:
        return MappedControlState(
            velocity_mujoco=(0.5, 0.0, 0.0),
            desired_heading_mujoco_wxyz=(1.0, 0.0, 0.0, 0.0),
            camera=CameraState(0, 0.0, 0.4, 3.0),
            strafe=False,
            walk_blend=0.0,
            stand=stand,
            terminate=terminate,
        )

    def test_space_and_x_edges_survive_release_until_consumed(self) -> None:
        mailbox = BoundaryControlMailbox()
        mailbox.publish(self._mapped(stand=True, terminate=False))
        mailbox.publish(self._mapped(stand=False, terminate=False))
        first, _ = mailbox.sample(0)
        self.assertEqual(first.requested_velocity_mujoco, (0.0, 0.0, 0.0))
        mailbox.publish(self._mapped(stand=False, terminate=True))
        mailbox.publish(self._mapped(stand=False, terminate=False))
        terminated, _ = mailbox.sample(1)
        self.assertIsNone(terminated)

    def test_sample_requires_monotonic_sequence(self) -> None:
        mailbox = BoundaryControlMailbox()
        mailbox.publish(self._mapped(stand=False, terminate=False))
        mailbox.sample(0)
        with self.assertRaises(ContractError):
            mailbox.sample(0)

    def test_sample_before_publish_is_rejected(self) -> None:
        mailbox = BoundaryControlMailbox()
        with self.assertRaises(ContractError):
            mailbox.sample(0)


class X11KeyStateProviderTests(unittest.TestCase):
    def test_missing_display_hard_fails(self) -> None:
        previous = os.environ.pop("DISPLAY", None)
        try:
            with self.assertRaises(ContractError):
                X11KeyStateProvider()
        finally:
            if previous is not None:
                os.environ["DISPLAY"] = previous


if __name__ == "__main__":
    unittest.main()
