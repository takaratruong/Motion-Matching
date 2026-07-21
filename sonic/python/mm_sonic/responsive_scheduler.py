"""Serial same-prefix responsive scheduler for the SONIC coordinator.

The scheduler is a thin orchestrator over the existing
``Coordinator.run_one_chunk``.  It samples one immutable
``IntentSnapshot(revision, observed_ns, command)`` from the boundary mailbox,
binds that exact revision into a ``command_is_current`` predicate, and retries
the same chunk index only when the coordinator reports a typed
``CandidateSuperseded`` outcome.  Every other failure propagates unchanged, and
a terminate snapshot returns ``None`` without touching the coordinator.

It is deliberately serial: exactly one in-flight MM request, no speculative
generation, and no change to the append-only publication or any default
coordinator behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from .coordinator import CandidateSuperseded
from .joints import ContractError


@dataclass(frozen=True)
class ScheduledPrefix:
    """One accepted responsive prefix bound to the intent that produced it.

    Preserves the exact successful ``IntentSnapshot`` and the monotonic
    ``sampled_ns`` timestamp read at its sample time, alongside the accepted
    result the coordinator/committer returned.
    """

    snapshot: object
    sampled_ns: int
    accepted: object


class ResponsiveScheduler:
    """Drive one committed prefix per operator boundary, retrying supersession."""

    def __init__(
        self,
        coordinator: object,
        mailbox: object,
        *,
        on_sample: Callable[[object, object], None] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not hasattr(coordinator, "run_one_chunk"):
            raise ContractError("scheduler requires a coordinator with run_one_chunk")
        if not hasattr(mailbox, "sample_intent"):
            raise ContractError("scheduler requires a mailbox with sample_intent")
        if on_sample is not None and not callable(on_sample):
            raise ContractError("scheduler on_sample must be callable or None")
        if not callable(monotonic_ns):
            raise ContractError("scheduler monotonic_ns must be callable")
        self._coordinator = coordinator
        self._mailbox = mailbox
        self._on_sample = on_sample
        self._monotonic_ns = monotonic_ns

    def run_one_prefix(self, chunk_index: int) -> ScheduledPrefix | None:
        """Sample the latest intent and commit one prefix, retrying supersession.

        Returns a :class:`ScheduledPrefix` binding the accepted result to the
        exact successful snapshot and its monotonic sampled timestamp, or
        ``None`` when the sampled intent is a terminate snapshot (no command).
        Retries the same ``chunk_index`` only after a typed
        ``CandidateSuperseded`` outcome, always re-sampling the newest revision.
        Any other exception propagates unchanged.  When provided, ``on_sample``
        fires for every sample (including terminate and superseded retries) so
        camera delivery stays synchronized with the boundary.
        """

        if type(chunk_index) is not int or chunk_index < 0:
            raise ContractError("scheduler chunk_index must be a nonnegative integer")
        while True:
            snapshot, mapped = self._mailbox.sample_intent(chunk_index)
            # The mailbox records the input observation while sampling.  Read
            # this boundary timestamp afterwards so the trace cannot claim the
            # sample preceded the observation that produced it.
            sampled_ns = self._monotonic_ns()
            if self._on_sample is not None:
                self._on_sample(snapshot, mapped)
            if snapshot.command is None:
                return None
            bound_revision = snapshot.revision

            def command_is_current(_command: object) -> bool:
                # The candidate is current only while no newer locomotion
                # revision has superseded the exact revision this snapshot
                # bound at sample time.
                return self._mailbox.current_revision <= bound_revision

            try:
                accepted = self._coordinator.run_one_chunk(
                    snapshot.command,
                    command_is_current=command_is_current,
                )
            except CandidateSuperseded:
                # Retry the same chunk index using the newest revision.
                continue
            return ScheduledPrefix(
                snapshot=snapshot, sampled_ns=sampled_ns, accepted=accepted
            )
