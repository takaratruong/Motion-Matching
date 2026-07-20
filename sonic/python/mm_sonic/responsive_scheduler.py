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

from .coordinator import CandidateSuperseded
from .joints import ContractError


class ResponsiveScheduler:
    """Drive one committed prefix per operator boundary, retrying supersession."""

    def __init__(self, coordinator: object, mailbox: object) -> None:
        if not hasattr(coordinator, "run_one_chunk"):
            raise ContractError("scheduler requires a coordinator with run_one_chunk")
        if not hasattr(mailbox, "sample_intent"):
            raise ContractError("scheduler requires a mailbox with sample_intent")
        self._coordinator = coordinator
        self._mailbox = mailbox

    def run_one_prefix(self, chunk_index: int) -> object | None:
        """Sample the latest intent and commit one prefix, retrying supersession.

        Returns the accepted chunk, or ``None`` when the sampled intent is a
        terminate snapshot (no command).  Retries the same ``chunk_index`` only
        after a typed ``CandidateSuperseded`` outcome, always re-sampling the
        newest revision.  Any other exception propagates unchanged.
        """

        if type(chunk_index) is not int or chunk_index < 0:
            raise ContractError("scheduler chunk_index must be a nonnegative integer")
        while True:
            snapshot, _mapped = self._mailbox.sample_intent(chunk_index)
            if snapshot.command is None:
                return None
            bound_revision = snapshot.revision

            def command_is_current(_command: object) -> bool:
                # The candidate is current only while no newer locomotion
                # revision has superseded the exact revision this snapshot
                # bound at sample time.
                return self._mailbox.current_revision <= bound_revision

            try:
                return self._coordinator.run_one_chunk(
                    snapshot.command,
                    command_is_current=command_is_current,
                )
            except CandidateSuperseded:
                # Retry the same chunk index using the newest revision.
                continue
