"""Read-only bounded reachability diagnostics for matcher transitions."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Generic, Iterable, TypeVar


StateT = TypeVar("StateT")
CommandT = TypeVar("CommandT")


@dataclass(frozen=True)
class ReachabilityLimits:
    max_depth: int = 2
    beam_width: int = 8
    max_expanded_states: int = 64
    max_source_advance_frames: int = 15

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("max_depth", self.max_depth, 2),
            ("beam_width", self.beam_width, 8),
            ("max_expanded_states", self.max_expanded_states, 64),
            (
                "max_source_advance_frames",
                self.max_source_advance_frames,
                15,
            ),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")


@dataclass(frozen=True)
class TransitionReachabilityDiagnostic:
    command_change_frame: int
    planner_compute_ns: int
    expanded_state_count: int
    depth_used: int
    reachable: bool
    first_clip_index: int | None
    first_frame_index: int | None
    terminal_clip_index: int | None
    terminal_frame_index: int | None
    budget_exhausted: bool


@dataclass(frozen=True)
class _FrontierState(Generic[StateT]):
    state: StateT
    first_identity: tuple[int, int]


def bounded_transition_reachability(
    start: StateT,
    *,
    command: CommandT,
    command_change_frame: int,
    expand: Callable[[StateT, CommandT, int], Iterable[StateT]],
    is_safe: Callable[[StateT, CommandT], bool],
    is_terminal: Callable[[StateT, CommandT], bool],
    identity: Callable[[StateT], tuple[int, int]],
    limits: ReachabilityLimits = ReachabilityLimits(),
) -> TransitionReachabilityDiagnostic:
    """Search ranked transition successors without mutating matcher state."""

    if type(command_change_frame) is not int or command_change_frame < 0:
        raise ValueError("command_change_frame must be a non-negative integer")
    if not isinstance(limits, ReachabilityLimits):
        raise TypeError("limits must be ReachabilityLimits")
    for name, callback in (
        ("expand", expand),
        ("is_safe", is_safe),
        ("is_terminal", is_terminal),
        ("identity", identity),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")

    started_ns = time.perf_counter_ns()
    frontier: list[_FrontierState[StateT]] = []
    expanded_count = 0
    depth_used = 0

    def diagnostic(
        *,
        reachable: bool,
        depth: int,
        first: tuple[int, int] | None = None,
        terminal: tuple[int, int] | None = None,
        budget_exhausted: bool = False,
    ) -> TransitionReachabilityDiagnostic:
        return TransitionReachabilityDiagnostic(
            command_change_frame=command_change_frame,
            planner_compute_ns=time.perf_counter_ns() - started_ns,
            expanded_state_count=expanded_count,
            depth_used=depth,
            reachable=reachable,
            first_clip_index=None if first is None else first[0],
            first_frame_index=None if first is None else first[1],
            terminal_clip_index=None if terminal is None else terminal[0],
            terminal_frame_index=None if terminal is None else terminal[1],
            budget_exhausted=budget_exhausted,
        )

    parents: tuple[tuple[StateT, tuple[int, int] | None], ...] = (
        (start, None),
    )
    for depth in range(1, limits.max_depth + 1):
        depth_used = depth
        frontier.clear()
        active = [
            (
                iter(
                    expand(
                        parent,
                        command,
                        limits.max_source_advance_frames,
                    )
                ),
                inherited_first,
            )
            for parent, inherited_first in parents
        ]
        while active and expanded_count < limits.max_expanded_states:
            next_active = []
            for iterator, inherited_first in active:
                if expanded_count >= limits.max_expanded_states:
                    break
                try:
                    child = next(iterator)
                except StopIteration:
                    continue
                next_active.append((iterator, inherited_first))
                expanded_count += 1
                safe = is_safe(child, command)
                if type(safe) is not bool:
                    raise TypeError("is_safe must return exact bool")
                if not safe:
                    continue
                child_identity = identity(child)
                if (
                    type(child_identity) is not tuple
                    or len(child_identity) != 2
                    or any(type(value) is not int for value in child_identity)
                ):
                    raise TypeError(
                        "identity must return an exact pair of integers"
                    )
                first = (
                    child_identity
                    if inherited_first is None
                    else inherited_first
                )
                terminal = is_terminal(child, command)
                if type(terminal) is not bool:
                    raise TypeError("is_terminal must return exact bool")
                if terminal:
                    return diagnostic(
                        reachable=True,
                        depth=depth,
                        first=first,
                        terminal=child_identity,
                    )
                if len(frontier) < limits.beam_width:
                    frontier.append(_FrontierState(child, first))
            active = next_active
        if expanded_count >= limits.max_expanded_states:
            return diagnostic(
                reachable=False,
                depth=depth,
                budget_exhausted=True,
            )
        if not frontier:
            break
        parents = tuple(
            (entry.state, entry.first_identity) for entry in frontier
        )
    return diagnostic(reachable=False, depth=depth_used)
