"""Global search over rigid motion windows placed along a desired root path."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .joints import ContractError


@dataclass(frozen=True, order=True)
class RootPathMotionCandidate:
    """One environment-certified source window placed on a root path."""

    candidate_id: str
    covered_start_m: float
    covered_stop_m: float
    placement_cost: float
    frame_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or isinstance(self.covered_start_m, bool)
            or not isinstance(self.covered_start_m, (int, float))
            or isinstance(self.covered_stop_m, bool)
            or not isinstance(self.covered_stop_m, (int, float))
            or isinstance(self.placement_cost, bool)
            or not isinstance(self.placement_cost, (int, float))
            or not all(
                math.isfinite(float(value))
                for value in (
                    self.covered_start_m,
                    self.covered_stop_m,
                    self.placement_cost,
                )
            )
            or float(self.covered_start_m) < 0.0
            or float(self.covered_stop_m)
            <= float(self.covered_start_m)
            or float(self.placement_cost) < 0.0
            or type(self.frame_count) is not int
            or self.frame_count < 2
        ):
            raise ContractError("root-path motion candidate is invalid")


@dataclass(frozen=True, order=True)
class RootPathMotionTransition:
    """A certified supported-frame transition between two placed windows."""

    first_candidate_id: str
    second_candidate_id: str
    transition_cost: float
    first_frame: int = 0
    second_frame: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.first_candidate_id, str)
            or not self.first_candidate_id
            or not isinstance(self.second_candidate_id, str)
            or not self.second_candidate_id
            or self.first_candidate_id == self.second_candidate_id
            or isinstance(self.transition_cost, bool)
            or not isinstance(self.transition_cost, (int, float))
            or not math.isfinite(float(self.transition_cost))
            or float(self.transition_cost) < 0.0
            or type(self.first_frame) is not int
            or self.first_frame < 0
            or type(self.second_frame) is not int
            or self.second_frame < 0
        ):
            raise ContractError("root-path motion transition is invalid")


@dataclass(frozen=True)
class RootPathMotionChain:
    candidate_ids: tuple[str, ...]
    covered_stop_m: float
    total_cost: float


@dataclass(frozen=True)
class OrderedLevelMotionCandidate:
    """A placed path candidate annotated with actual touchdown levels."""

    candidate: RootPathMotionCandidate
    level_indices: tuple[int, ...]
    touchdown_feet: tuple[int, ...]
    touchdown_frames: tuple[int, ...]
    terminal_double_support: bool = False

    def __post_init__(self) -> None:
        count = len(self.level_indices)
        if (
            not isinstance(self.candidate, RootPathMotionCandidate)
            or not isinstance(self.level_indices, tuple)
            or not isinstance(self.touchdown_feet, tuple)
            or not isinstance(self.touchdown_frames, tuple)
            or count < 1
            or len(self.touchdown_feet) != count
            or len(self.touchdown_frames) != count
            or any(type(value) is not int or value < 0 for value in self.level_indices)
            or any(
                next_value < value
                for value, next_value in zip(
                    self.level_indices, self.level_indices[1:]
                )
            )
            or any(type(value) is not int or value not in (0, 1) for value in self.touchdown_feet)
            or any(
                type(value) is not int
                or not 0 <= value < self.candidate.frame_count
                for value in self.touchdown_frames
            )
            or any(
                next_value <= value
                for value, next_value in zip(
                    self.touchdown_frames, self.touchdown_frames[1:]
                )
            )
            or type(self.terminal_double_support) is not bool
        ):
            raise ContractError("ordered-level motion candidate is invalid")


def _advance_ordered_levels(
    *,
    current_level: int,
    final_feet: tuple[int, ...],
    candidate: OrderedLevelMotionCandidate,
    start_frame: int,
    stop_frame: int,
    required_level_count: int,
) -> tuple[int, tuple[int, ...]] | None:
    level = current_level
    tail = final_feet
    final_level = required_level_count - 1
    for event_level, event_foot, event_frame in zip(
        candidate.level_indices,
        candidate.touchdown_feet,
        candidate.touchdown_frames,
    ):
        if not start_frame <= event_frame < stop_frame:
            continue
        if event_level > level + 1:
            return None
        if event_level == level + 1:
            level = event_level
        if event_level == final_level and level == final_level:
            tail = (tail + (event_foot,))[-2:]
    return level, tail


def root_path_segment_starts(
    *,
    path_length_m: float,
    segment_length_m: float,
    stride_m: float,
) -> tuple[float, ...]:
    """Return regular search anchors plus the exact final segment anchor."""

    values = (path_length_m, segment_length_m, stride_m)
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in values
        )
        or float(segment_length_m) >= float(path_length_m)
        or float(stride_m) > float(segment_length_m)
    ):
        raise ContractError("root-path segment options are invalid")
    final_start = float(path_length_m) - float(segment_length_m)
    count = int(math.floor(final_start / float(stride_m) + 1.0e-9))
    starts = [round(index * float(stride_m), 9) for index in range(count + 1)]
    starts.append(round(final_start, 9))
    return tuple(sorted(set(starts)))


def root_path_search_anchors(
    *,
    path_length_m: float,
    stride_m: float,
    minimum_window_length_m: float,
) -> tuple[float, ...]:
    """Cover all possible starts, including shorter windows near path end."""

    values = (path_length_m, stride_m, minimum_window_length_m)
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in values
        )
        or float(minimum_window_length_m) >= float(path_length_m)
        or float(stride_m) > float(path_length_m)
    ):
        raise ContractError("root-path search anchor options are invalid")
    final_start = (
        float(path_length_m) - float(minimum_window_length_m)
    )
    count = int(math.floor(final_start / float(stride_m) + 1.0e-9))
    anchors = [
        round(index * float(stride_m), 9)
        for index in range(count + 1)
    ]
    anchors.append(round(final_start, 9))
    return tuple(sorted(set(anchors)))


def select_root_path_motion_chain(
    *,
    candidates: Sequence[RootPathMotionCandidate],
    transitions: Sequence[RootPathMotionTransition],
    path_length_m: float,
    maximum_coverage_gap_m: float = 0.05,
    start_tolerance_m: float = 0.05,
    minimum_segment_frames: int = 1,
    fragmentation_cost: float = 0.0,
) -> RootPathMotionChain:
    """Find the minimum-cost compatible chain that covers the full path.

    Candidates are already rigidly placed and environment-certified. Explicit
    transitions encode pose, velocity, and support compatibility. The search
    therefore cannot invent a splice merely because two intervals overlap.
    """

    owned_candidates = tuple(candidates)
    owned_transitions = tuple(transitions)
    if (
        not owned_candidates
        or any(
            not isinstance(item, RootPathMotionCandidate)
            for item in owned_candidates
        )
        or any(
            not isinstance(item, RootPathMotionTransition)
            for item in owned_transitions
        )
        or isinstance(path_length_m, bool)
        or not isinstance(path_length_m, (int, float))
        or not math.isfinite(float(path_length_m))
        or float(path_length_m) <= 0.0
        or isinstance(maximum_coverage_gap_m, bool)
        or not isinstance(maximum_coverage_gap_m, (int, float))
        or not math.isfinite(float(maximum_coverage_gap_m))
        or float(maximum_coverage_gap_m) < 0.0
        or isinstance(start_tolerance_m, bool)
        or not isinstance(start_tolerance_m, (int, float))
        or not math.isfinite(float(start_tolerance_m))
        or float(start_tolerance_m) < 0.0
        or type(minimum_segment_frames) is not int
        or minimum_segment_frames < 1
        or isinstance(fragmentation_cost, bool)
        or not isinstance(fragmentation_cost, (int, float))
        or not math.isfinite(float(fragmentation_cost))
        or float(fragmentation_cost) < 0.0
    ):
        raise ContractError("root-path motion graph input is invalid")
    by_id = {item.candidate_id: item for item in owned_candidates}
    if len(by_id) != len(owned_candidates):
        raise ContractError("root-path candidate ids are not unique")
    incoming: dict[str, list[RootPathMotionTransition]] = {
        item.candidate_id: [] for item in owned_candidates
    }
    transition_keys: set[tuple[str, str]] = set()
    for transition in owned_transitions:
        key = (
            transition.first_candidate_id,
            transition.second_candidate_id,
        )
        if (
            key in transition_keys
            or key[0] not in by_id
            or key[1] not in by_id
            or transition.first_frame >= by_id[key[0]].frame_count
            or transition.second_frame >= by_id[key[1]].frame_count
        ):
            raise ContractError("root-path transition endpoints are invalid")
        transition_keys.add(key)
        incoming[key[1]].append(transition)

    ordered = sorted(
        owned_candidates,
        key=lambda item: (
            item.covered_stop_m,
            item.covered_start_m,
            item.candidate_id,
        ),
    )
    best: dict[
        tuple[str, int], tuple[float, tuple[str, ...]]
    ] = {}
    for candidate in ordered:
        if candidate.covered_start_m <= float(start_tolerance_m):
            best[(candidate.candidate_id, 0)] = (
                float(candidate.placement_cost),
                (candidate.candidate_id,),
            )
        for transition in incoming[candidate.candidate_id]:
            previous = by_id[transition.first_candidate_id]
            if (
                candidate.covered_stop_m
                <= previous.covered_stop_m + 1.0e-9
                or candidate.covered_start_m
                > previous.covered_stop_m
                + float(maximum_coverage_gap_m)
            ):
                continue
            options = []
            for (
                previous_candidate_id,
                previous_incoming_frame,
            ), previous_best in best.items():
                if (
                    previous_candidate_id != previous.candidate_id
                    or previous_incoming_frame
                    + int(minimum_segment_frames)
                    > transition.first_frame + 1
                ):
                    continue
                options.append(
                    (
                        previous_best[0]
                        + float(transition.transition_cost)
                        + float(fragmentation_cost)
                        + float(candidate.placement_cost),
                        previous_best[1] + (candidate.candidate_id,),
                    )
                )
            if not options:
                continue
            state = (candidate.candidate_id, transition.second_frame)
            selected = min(options, key=lambda item: (item[0], item[1]))
            existing = best.get(state)
            if existing is None or selected < existing:
                best[state] = selected

    complete = [
        (
            cost_and_chain[0],
            cost_and_chain[1],
            candidate.covered_stop_m,
        )
        for candidate in ordered
        if candidate.covered_stop_m
        >= float(path_length_m) - float(start_tolerance_m)
        for (candidate_id, incoming_frame), cost_and_chain in best.items()
        if candidate_id == candidate.candidate_id
        and candidate.frame_count - incoming_frame
        >= int(minimum_segment_frames)
    ]
    if not complete:
        raise ContractError("no complete root-path chain")
    total_cost, candidate_ids, covered_stop = min(
        complete, key=lambda item: (item[0], item[1])
    )
    return RootPathMotionChain(
        candidate_ids=candidate_ids,
        covered_stop_m=float(covered_stop),
        total_cost=float(total_cost),
    )


def select_ordered_level_motion_chain(
    *,
    candidates: Sequence[OrderedLevelMotionCandidate],
    transitions: Sequence[RootPathMotionTransition],
    path_length_m: float,
    required_level_count: int,
    maximum_coverage_gap_m: float = 0.05,
    start_tolerance_m: float = 0.05,
    minimum_segment_frames: int = 1,
    fragmentation_cost: float = 0.0,
) -> RootPathMotionChain:
    """Select a complete path chain without skipping required terrain levels."""

    owned = tuple(candidates)
    transition_items = tuple(transitions)
    if (
        not owned
        or any(not isinstance(item, OrderedLevelMotionCandidate) for item in owned)
        or any(
            not isinstance(item, RootPathMotionTransition)
            for item in transition_items
        )
        or type(required_level_count) is not int
        or required_level_count < 2
        or not math.isfinite(float(path_length_m))
        or path_length_m <= 0.0
        or not math.isfinite(float(maximum_coverage_gap_m))
        or maximum_coverage_gap_m < 0.0
        or not math.isfinite(float(start_tolerance_m))
        or start_tolerance_m < 0.0
        or type(minimum_segment_frames) is not int
        or minimum_segment_frames < 1
        or isinstance(fragmentation_cost, bool)
        or not isinstance(fragmentation_cost, (int, float))
        or not math.isfinite(float(fragmentation_cost))
        or float(fragmentation_cost) < 0.0
    ):
        raise ContractError("ordered-level root-path graph input is invalid")
    by_id = {
        item.candidate.candidate_id: item for item in owned
    }
    if len(by_id) != len(owned):
        raise ContractError("ordered-level candidate ids are not unique")
    incoming: dict[str, list[RootPathMotionTransition]] = {
        candidate_id: [] for candidate_id in by_id
    }
    transition_keys = set()
    for transition in transition_items:
        key = (
            transition.first_candidate_id,
            transition.second_candidate_id,
        )
        if (
            key in transition_keys
            or key[0] not in by_id
            or key[1] not in by_id
            or transition.first_frame
            >= by_id[key[0]].candidate.frame_count
            or transition.second_frame
            >= by_id[key[1]].candidate.frame_count
        ):
            raise ContractError(
                "ordered-level transition endpoints are invalid"
            )
        transition_keys.add(key)
        incoming[key[1]].append(transition)

    ordered = sorted(
        owned,
        key=lambda item: (
            item.candidate.covered_stop_m,
            item.candidate.covered_start_m,
            item.candidate.candidate_id,
        ),
    )
    best: dict[
        tuple[str, int, int, tuple[int, ...]],
        tuple[float, tuple[str, ...]],
    ] = {}
    for item in ordered:
        candidate = item.candidate
        if candidate.covered_start_m <= start_tolerance_m:
            state = (candidate.candidate_id, 0, -1, ())
            best[state] = (
                float(candidate.placement_cost),
                (candidate.candidate_id,),
            )
        for transition in incoming[candidate.candidate_id]:
            previous_item = by_id[transition.first_candidate_id]
            previous = previous_item.candidate
            if (
                candidate.covered_stop_m
                <= previous.covered_stop_m + 1.0e-9
                or candidate.covered_start_m
                > previous.covered_stop_m + maximum_coverage_gap_m
            ):
                continue
            options = []
            for (
                previous_id,
                previous_incoming,
                current_level,
                final_feet,
            ), previous_best in best.items():
                if (
                    previous_id != previous.candidate_id
                    or previous_incoming + minimum_segment_frames
                    > transition.first_frame + 1
                ):
                    continue
                advanced = _advance_ordered_levels(
                    current_level=current_level,
                    final_feet=final_feet,
                    candidate=previous_item,
                    start_frame=previous_incoming,
                    stop_frame=transition.first_frame + 1,
                    required_level_count=required_level_count,
                )
                if advanced is None:
                    continue
                options.append(
                    (
                        previous_best[0]
                        + float(transition.transition_cost)
                        + float(fragmentation_cost)
                        + float(candidate.placement_cost),
                        previous_best[1] + (candidate.candidate_id,),
                        advanced,
                    )
                )
            for cost, chain, (level, feet) in options:
                state = (
                    candidate.candidate_id,
                    transition.second_frame,
                    level,
                    feet,
                )
                existing = best.get(state)
                value = (cost, chain)
                if existing is None or value < existing:
                    best[state] = value

    complete = []
    final_level = required_level_count - 1
    for item in ordered:
        candidate = item.candidate
        if candidate.covered_stop_m < path_length_m - start_tolerance_m:
            continue
        for (
            candidate_id,
            incoming_frame,
            current_level,
            final_feet,
        ), cost_and_chain in best.items():
            if (
                candidate_id != candidate.candidate_id
                or candidate.frame_count - incoming_frame
                < minimum_segment_frames
            ):
                continue
            advanced = _advance_ordered_levels(
                current_level=current_level,
                final_feet=final_feet,
                candidate=item,
                start_frame=incoming_frame,
                stop_frame=candidate.frame_count,
                required_level_count=required_level_count,
            )
            if (
                advanced is None
                or advanced[0] != final_level
                or len(advanced[1]) < 2
                or advanced[1][-1] == advanced[1][-2]
                or not item.terminal_double_support
            ):
                continue
            complete.append(
                (
                    cost_and_chain[0],
                    cost_and_chain[1],
                    candidate.covered_stop_m,
                )
            )
    if not complete:
        raise ContractError("no complete ordered-level root-path chain")
    total_cost, candidate_ids, covered_stop = min(
        complete, key=lambda item: (item[0], item[1])
    )
    return RootPathMotionChain(
        candidate_ids=candidate_ids,
        covered_stop_m=float(covered_stop),
        total_cost=float(total_cost),
    )
