from __future__ import annotations

import unittest

from mm_sonic.commands import CommandSample
from mm_sonic.coordinator import CandidateSuperseded
from mm_sonic.joints import ContractError
from mm_sonic.operator_x11 import IntentSnapshot
from mm_sonic.responsive_scheduler import ResponsiveScheduler


def _command(chunk_index: int = 0, speed: float = 0.5) -> CommandSample:
    return CommandSample(
        chunk_index=chunk_index,
        requested_velocity_mujoco=(speed, 0.0, 0.0),
        desired_heading_mujoco_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


class FakeMailbox:
    """Serves scripted IntentSnapshots per sample_intent call."""

    def __init__(self, snapshots: list[IntentSnapshot]) -> None:
        self._snapshots = list(snapshots)
        self.sampled_indices: list[int] = []
        self.current_revision = 1

    def sample_intent(self, chunk_index: int):
        if type(chunk_index) is not int or chunk_index < 0:
            raise ContractError("chunk_index must be a nonnegative integer")
        self.sampled_indices.append(chunk_index)
        snapshot = self._snapshots.pop(0)
        self.current_revision = snapshot.revision
        return snapshot, None


class RecordingCoordinator:
    """Records predicate results and can inject supersession/failures."""

    def __init__(self, script: list[str]) -> None:
        # Each entry: "accept", "supersede", or "raise".
        self._script = list(script)
        self.calls: list[tuple[int | None, bool]] = []
        self.release_calls = 0

    def run_one_chunk(self, command=None, *, command_is_current=None):
        # Evaluate the bound predicate exactly as the real coordinator would,
        # immediately before publication.
        predicate_value = None
        if command_is_current is not None:
            predicate_value = command_is_current(command)
        action = self._script.pop(0)
        chunk = command.chunk_index if command is not None else None
        self.calls.append((chunk, bool(predicate_value)))
        if action == "supersede":
            candidate = command if command is not None else _command()
            raise CandidateSuperseded("session:candidate:000000", candidate)
        if action == "raise":
            raise RuntimeError("terminal failure")
        self.release_calls += 1
        return object()


class ResponsiveSchedulerTests(unittest.TestCase):
    def test_accepts_current_snapshot_and_binds_exact_revision(self) -> None:
        snapshot = IntentSnapshot(revision=3, observed_ns=100, command=_command())
        mailbox = FakeMailbox([snapshot])
        coordinator = RecordingCoordinator(["accept"])
        scheduler = ResponsiveScheduler(coordinator, mailbox)

        result = scheduler.run_one_prefix(0)

        self.assertIsNotNone(result)
        # The predicate bound the exact sampled revision and returned current.
        self.assertEqual(coordinator.calls, [(0, True)])
        self.assertEqual(mailbox.sampled_indices, [0])

    def test_supersession_retries_same_chunk_index_with_newest_revision(self) -> None:
        stale = IntentSnapshot(revision=1, observed_ns=100, command=_command())
        fresh = IntentSnapshot(revision=2, observed_ns=200, command=_command())
        mailbox = FakeMailbox([stale, fresh])
        coordinator = RecordingCoordinator(["supersede", "accept"])
        scheduler = ResponsiveScheduler(coordinator, mailbox)

        result = scheduler.run_one_prefix(0)

        self.assertIsNotNone(result)
        # Same chunk index re-sampled after supersession.
        self.assertEqual(mailbox.sampled_indices, [0, 0])
        self.assertEqual(len(coordinator.calls), 2)

    def test_terminate_snapshot_returns_none_without_calling_coordinator(self) -> None:
        snapshot = IntentSnapshot(revision=1, observed_ns=100, command=None)
        mailbox = FakeMailbox([snapshot])
        coordinator = RecordingCoordinator([])
        scheduler = ResponsiveScheduler(coordinator, mailbox)

        result = scheduler.run_one_prefix(0)

        self.assertIsNone(result)
        self.assertEqual(coordinator.calls, [])
        self.assertEqual(coordinator.release_calls, 0)

    def test_other_failures_propagate_without_retry(self) -> None:
        snapshot = IntentSnapshot(revision=1, observed_ns=100, command=_command())
        mailbox = FakeMailbox([snapshot])
        coordinator = RecordingCoordinator(["raise"])
        scheduler = ResponsiveScheduler(coordinator, mailbox)

        with self.assertRaises(RuntimeError):
            scheduler.run_one_prefix(0)
        # No retry after a non-supersession failure.
        self.assertEqual(mailbox.sampled_indices, [0])
        self.assertEqual(coordinator.release_calls, 0)

    def test_predicate_currency_reflects_newer_revision(self) -> None:
        # If the mailbox advances past the bound revision during generation,
        # the bound predicate must report the candidate as stale.
        snapshot = IntentSnapshot(revision=1, observed_ns=100, command=_command())

        class AdvancingMailbox(FakeMailbox):
            def __init__(self) -> None:
                super().__init__([snapshot, snapshot])
                self.current_revision = 1

        mailbox = AdvancingMailbox()

        class PredicateCapturingCoordinator(RecordingCoordinator):
            def run_one_chunk(self, command=None, *, command_is_current=None):
                # Simulate a newer revision arriving mid-generation.
                mailbox.current_revision = 2
                return super().run_one_chunk(
                    command, command_is_current=command_is_current
                )

        coordinator = PredicateCapturingCoordinator(["supersede", "accept"])
        scheduler = ResponsiveScheduler(coordinator, mailbox)

        result = scheduler.run_one_prefix(0)
        self.assertIsNotNone(result)
        # First call bound revision 1 but the live revision advanced to 2, so
        # the predicate reported stale.
        self.assertFalse(coordinator.calls[0][1])

    def test_requires_run_one_chunk_and_sample_intent(self) -> None:
        with self.assertRaises(ContractError):
            ResponsiveScheduler(object(), FakeMailbox([]))
        with self.assertRaises(ContractError):
            ResponsiveScheduler(RecordingCoordinator([]), object())


class ScheduledPrefixResultTests(unittest.TestCase):
    def test_accepted_prefix_preserves_snapshot_and_sampled_ns(self) -> None:
        from mm_sonic.responsive_scheduler import ScheduledPrefix

        snapshot = IntentSnapshot(revision=4, observed_ns=123, command=_command())
        mailbox = FakeMailbox([snapshot])
        coordinator = RecordingCoordinator(["accept"])
        clock = iter([777])
        scheduler = ResponsiveScheduler(
            coordinator, mailbox, monotonic_ns=lambda: next(clock)
        )

        result = scheduler.run_one_prefix(0)

        self.assertIsInstance(result, ScheduledPrefix)
        # The exact successful snapshot object is preserved.
        self.assertIs(result.snapshot, snapshot)
        self.assertEqual(result.sampled_ns, 777)
        self.assertIsNotNone(result.accepted)

    def test_on_sample_callback_fires_for_each_sample_with_snapshot(self) -> None:
        stale = IntentSnapshot(revision=1, observed_ns=100, command=_command())
        fresh = IntentSnapshot(revision=2, observed_ns=200, command=_command())
        mailbox = FakeMailbox([stale, fresh])
        coordinator = RecordingCoordinator(["supersede", "accept"])
        seen: list[tuple[int, object]] = []
        scheduler = ResponsiveScheduler(
            coordinator,
            mailbox,
            on_sample=lambda snapshot, mapped: seen.append(
                (snapshot.revision, mapped)
            ),
        )

        result = scheduler.run_one_prefix(0)

        self.assertIsNotNone(result)
        # on_sample fired once per sample, including the superseded retry.
        self.assertEqual([revision for revision, _ in seen], [1, 2])

    def test_on_sample_fires_before_terminate_returns_none(self) -> None:
        snapshot = IntentSnapshot(revision=1, observed_ns=100, command=None)
        mailbox = FakeMailbox([snapshot])
        coordinator = RecordingCoordinator([])
        seen: list[object] = []
        scheduler = ResponsiveScheduler(
            coordinator, mailbox, on_sample=lambda s, m: seen.append(s)
        )

        result = scheduler.run_one_prefix(0)

        self.assertIsNone(result)
        # Camera delivery still happens on the terminate boundary.
        self.assertEqual(len(seen), 1)

    def test_ordinary_coordinator_compat_without_new_kwargs(self) -> None:
        # A plain Coordinator (no monotonic_ns/on_sample) still drives.
        snapshot = IntentSnapshot(revision=1, observed_ns=100, command=_command())
        mailbox = FakeMailbox([snapshot])
        coordinator = RecordingCoordinator(["accept"])
        scheduler = ResponsiveScheduler(coordinator, mailbox)

        result = scheduler.run_one_prefix(0)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
