"""Layered contact-action search for constant-heading terrain paths."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .joints import ContractError
from .torch_foothold_actions import (
    FootholdAction,
    FootholdActionIndex,
    FootholdTransitionGraph,
    foothold_action_descriptor_cost,
)
from .torch_terrain_footprint_candidates import (
    TerrainFootprintCandidate,
    TerrainFootprintLayer,
)


def _finite_nonnegative(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ContractError(f"heading footprint search {label} is invalid")
    return float(value)


@dataclass(frozen=True)
class FootprintActionEdge:
    action_key: tuple[int, int]
    candidate_indices: tuple[int, int]
    descriptor_cost: float
    transition_cost: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.action_key, tuple)
            or len(self.action_key) != 2
            or any(type(value) is not int or value < 0 for value in self.action_key)
            or not isinstance(self.candidate_indices, tuple)
            or len(self.candidate_indices) != 2
            or any(
                type(value) is not int or value < 0
                for value in self.candidate_indices
            )
        ):
            raise ContractError("heading footprint edge metadata is invalid")
        object.__setattr__(
            self,
            "descriptor_cost",
            _finite_nonnegative(self.descriptor_cost, "descriptor cost"),
        )
        object.__setattr__(
            self,
            "transition_cost",
            _finite_nonnegative(self.transition_cost, "transition cost"),
        )


@dataclass(frozen=True)
class HeadingFootprintPlan:
    heading_scene_xy: torch.Tensor
    footprints: tuple[TerrainFootprintCandidate, ...]
    edges: tuple[FootprintActionEdge, ...]
    action_keys: tuple[tuple[int, int], ...]
    total_cost: float

    def __post_init__(self) -> None:
        heading = self.heading_scene_xy
        if (
            not isinstance(heading, torch.Tensor)
            or tuple(heading.shape) != (2,)
            or not heading.dtype.is_floating_point
            or not torch.isfinite(heading).all()
            or abs(float(torch.linalg.vector_norm(heading).item()) - 1.0)
            > 1.0e-5
            or not isinstance(self.footprints, tuple)
            or any(
                not isinstance(item, TerrainFootprintCandidate)
                for item in self.footprints
            )
            or not isinstance(self.edges, tuple)
            or any(not isinstance(item, FootprintActionEdge) for item in self.edges)
            or not isinstance(self.action_keys, tuple)
            or self.action_keys != tuple(edge.action_key for edge in self.edges)
            or len(self.footprints) != 2 * len(self.edges)
        ):
            raise ContractError("heading footprint plan is invalid")
        object.__setattr__(self, "heading_scene_xy", heading.detach().clone())
        object.__setattr__(
            self,
            "total_cost",
            _finite_nonnegative(self.total_cost, "total cost"),
        )


class HeadingPlanningFailure(ContractError):
    def __init__(
        self,
        *,
        code: str,
        step_index: int,
        candidate_count: int,
        action_count: int,
        reasons: tuple[str, ...],
    ) -> None:
        if (
            code not in ("no_motion_coverage", "sequence_dead_end")
            or type(step_index) is not int
            or step_index < 0
            or type(candidate_count) is not int
            or candidate_count < 0
            or type(action_count) is not int
            or action_count < 0
            or not isinstance(reasons, tuple)
            or not reasons
            or any(not isinstance(item, str) or not item for item in reasons)
        ):
            raise ContractError("heading planning failure is invalid")
        self.code = code
        self.step_index = step_index
        self.candidate_count = candidate_count
        self.action_count = action_count
        self.reasons = reasons
        super().__init__(
            f"{code} at step {step_index}: {'; '.join(reasons)}"
        )


def action_edge_cost(
    *,
    descriptor_cost: float,
    placement_cost: float,
    transition_cost: float,
    yaw_delta_rad: float,
    maximum_yaw_delta_rad: float,
) -> float:
    """Combine normalized contact, placement, boundary, and heading costs."""

    descriptor = _finite_nonnegative(descriptor_cost, "descriptor cost")
    placement = _finite_nonnegative(placement_cost, "placement cost")
    transition = _finite_nonnegative(transition_cost, "transition cost")
    yaw = _finite_nonnegative(abs(yaw_delta_rad), "yaw delta")
    maximum_yaw = _finite_nonnegative(
        maximum_yaw_delta_rad, "maximum yaw delta"
    )
    if maximum_yaw <= 0.0 or yaw > maximum_yaw:
        return math.inf
    return (
        descriptor
        + placement
        + transition
        + (yaw / maximum_yaw) ** 2
    )


@dataclass(frozen=True)
class _BeamState:
    root_heading_xy: torch.Tensor
    foot_surface_height_m: torch.Tensor
    contact_frame: int
    footprints: tuple[TerrainFootprintCandidate, ...]
    edges: tuple[FootprintActionEdge, ...]
    action_indices: tuple[int, ...]
    candidate_order: tuple[int, ...]
    total_cost: float


def _transition_cost(
    graph: FootholdTransitionGraph,
    source_index: int,
    target_index: int,
    maximum_position_error_rad: float,
    maximum_velocity_error_rad_s: float,
) -> float:
    assert graph.terminal_joint_position is not None
    assert graph.terminal_joint_velocity is not None
    position = torch.linalg.vector_norm(
        graph.terminal_joint_position[source_index]
        - graph.entry_joint_position[target_index]
    )
    velocity = torch.linalg.vector_norm(
        graph.terminal_joint_velocity[source_index]
        - graph.entry_joint_velocity[target_index]
    )
    return float(
        (
            torch.square(position / maximum_position_error_rad)
            + torch.square(velocity / maximum_velocity_error_rad_s)
        ).item()
    )


def search_heading_footprint_plan(
    *,
    layers: tuple[TerrainFootprintLayer, ...],
    heading_scene_xy: torch.Tensor,
    action_index: FootholdActionIndex,
    transition_graph: FootholdTransitionGraph | None,
    beam_width: int,
    xy_tolerance_m: float,
    height_tolerance_m: float,
    timing_tolerance_frames: int,
    maximum_yaw_delta_rad: float,
    maximum_transition_position_error_rad: float,
    maximum_transition_velocity_error_rad_s: float,
    start_root_heading_xy: torch.Tensor | None = None,
    start_foot_surface_height_m: torch.Tensor | None = None,
    current_joint_position: torch.Tensor | None = None,
    current_joint_velocity: torch.Tensor | None = None,
) -> HeadingFootprintPlan:
    """Select a complete, transition-compatible action chain."""

    if (
        not isinstance(layers, tuple)
        or not layers
        or len(layers) % 2
        or any(
            not isinstance(layer, TerrainFootprintLayer)
            or layer.step_index != index
            or not layer.candidates
            for index, layer in enumerate(layers)
        )
        or any(
            left.nominal.foot == right.nominal.foot
            for left, right in zip(layers, layers[1:])
        )
        or not isinstance(action_index, FootholdActionIndex)
        or (
            transition_graph is not None
            and not isinstance(transition_graph, FootholdTransitionGraph)
        )
        or type(beam_width) is not int
        or beam_width < 1
        or type(timing_tolerance_frames) is not int
        or timing_tolerance_frames < 0
    ):
        raise ContractError("heading footprint search input is invalid")
    scalar_limits = (
        xy_tolerance_m,
        height_tolerance_m,
        maximum_yaw_delta_rad,
        maximum_transition_position_error_rad,
        maximum_transition_velocity_error_rad_s,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in scalar_limits
    ):
        raise ContractError("heading footprint search limits are invalid")

    first = layers[0].nominal.center_heading_xy
    heading = heading_scene_xy
    if (
        not isinstance(heading, torch.Tensor)
        or tuple(heading.shape) != (2,)
        or heading.device != first.device
        or heading.dtype != first.dtype
        or not torch.isfinite(heading).all()
        or float(torch.linalg.vector_norm(heading).item()) <= 1.0e-6
    ):
        raise ContractError("heading footprint search heading is invalid")
    heading = heading / torch.linalg.vector_norm(heading)
    device = first.device
    dtype = first.dtype

    if start_root_heading_xy is None:
        root = torch.zeros(2, dtype=dtype, device=device)
    else:
        root = start_root_heading_xy
    if start_foot_surface_height_m is None:
        surface = torch.zeros(2, dtype=dtype, device=device)
    else:
        surface = start_foot_surface_height_m
    if any(
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (2,)
        or value.device != device
        or value.dtype != dtype
        or not torch.isfinite(value).all()
        for value in (root, surface)
    ):
        raise ContractError("heading footprint search start state is invalid")

    actions = action_index.actions
    if any(
        action.landing_xy_start_frame_m.device != device
        or action.landing_xy_start_frame_m.dtype != dtype
        for action in actions
    ):
        raise ContractError("heading footprint search action device is invalid")
    graph_rows: dict[tuple[int, int], int] = {}
    pair_eligible = None
    entry_eligible = None
    if transition_graph is not None:
        graph_rows = {
            key: index
            for index, key in enumerate(transition_graph.action_keys)
        }
        if any(
            (action.clip_index, action.start_frame) not in graph_rows
            for action in actions
        ):
            raise ContractError(
                "heading footprint search graph omits an action"
            )
        pair_eligible = transition_graph.pair_eligibility(
            maximum_position_error_rad=maximum_transition_position_error_rad,
            maximum_velocity_error_rad_s=(
                maximum_transition_velocity_error_rad_s
            ),
        )
        if (current_joint_position is None) != (
            current_joint_velocity is None
        ):
            raise ContractError(
                "heading footprint search current state is incomplete"
            )
        if current_joint_position is not None:
            entry_eligible = transition_graph.entry_eligibility(
                current_joint_position=current_joint_position,
                current_joint_velocity=current_joint_velocity,
                maximum_position_error_rad=(
                    maximum_transition_position_error_rad
                ),
                maximum_velocity_error_rad_s=(
                    maximum_transition_velocity_error_rad_s
                ),
            )
    elif current_joint_position is not None or current_joint_velocity is not None:
        raise ContractError(
            "heading footprint search current state requires a graph"
        )

    beam = (
        _BeamState(
            root_heading_xy=root.detach().clone(),
            foot_surface_height_m=surface.detach().clone(),
            contact_frame=0,
            footprints=(),
            edges=(),
            action_indices=(),
            candidate_order=(),
            total_cost=0.0,
        ),
    )
    for layer_index in range(0, len(layers), 2):
        first_layer = layers[layer_index]
        second_layer = layers[layer_index + 1]
        expanded: list[_BeamState] = []
        descriptor_match = False
        graph_rejection = False
        for state in beam:
            for first_candidate_index, first_candidate in enumerate(
                first_layer.candidates
            ):
                for second_candidate_index, second_candidate in enumerate(
                    second_layer.candidates
                ):
                    candidates = (first_candidate, second_candidate)
                    landing_feet = torch.tensor(
                        tuple(item.foot for item in candidates),
                        dtype=torch.int64,
                        device=device,
                    )
                    landing_xy = torch.stack(
                        [
                            item.center_heading_xy
                            - state.root_heading_xy
                            for item in candidates
                        ]
                    )
                    landing_height = torch.tensor(
                        tuple(
                            item.surface_height_m
                            - float(
                                state.foot_surface_height_m[item.foot].item()
                            )
                            for item in candidates
                        ),
                        dtype=dtype,
                        device=device,
                    )
                    landing_timing = torch.tensor(
                        tuple(
                            layer.nominal.contact_frame
                            - state.contact_frame
                            for layer in (first_layer, second_layer)
                        ),
                        dtype=torch.int64,
                        device=device,
                    )
                    for next_action_index, action in enumerate(actions):
                        descriptor = foothold_action_descriptor_cost(
                            landing_feet=landing_feet,
                            landing_xy_heading_m=landing_xy,
                            landing_height_delta_m=landing_height,
                            landing_frame_offsets=landing_timing,
                            action=action,
                            xy_tolerance_m=xy_tolerance_m,
                            height_tolerance_m=height_tolerance_m,
                            timing_tolerance_frames=timing_tolerance_frames,
                        )
                        if not bool(torch.isfinite(descriptor).item()):
                            continue
                        yaw_delta = float(
                            action.root_yaw_delta_rad[-1].item()
                        )
                        if abs(yaw_delta) > maximum_yaw_delta_rad:
                            continue
                        descriptor_match = True
                        next_graph_row = (
                            graph_rows[
                                (action.clip_index, action.start_frame)
                            ]
                            if transition_graph is not None
                            else -1
                        )
                        transition_cost = 0.0
                        if entry_eligible is not None and not state.edges:
                            if not bool(entry_eligible[next_graph_row].item()):
                                graph_rejection = True
                                continue
                        if pair_eligible is not None and state.edges:
                            previous_action = actions[
                                state.action_indices[-1]
                            ]
                            previous_graph_row = graph_rows[
                                (
                                    previous_action.clip_index,
                                    previous_action.start_frame,
                                )
                            ]
                            if not bool(
                                pair_eligible[
                                    previous_graph_row, next_graph_row
                                ].item()
                            ):
                                graph_rejection = True
                                continue
                            assert transition_graph is not None
                            transition_cost = _transition_cost(
                                transition_graph,
                                previous_graph_row,
                                next_graph_row,
                                maximum_transition_position_error_rad,
                                maximum_transition_velocity_error_rad_s,
                            )
                        placement_cost = sum(
                            item.placement_cost for item in candidates
                        )
                        edge_increment = action_edge_cost(
                            descriptor_cost=float(descriptor.item()),
                            placement_cost=placement_cost,
                            transition_cost=transition_cost,
                            yaw_delta_rad=yaw_delta,
                            maximum_yaw_delta_rad=maximum_yaw_delta_rad,
                        )
                        if not math.isfinite(edge_increment):
                            continue
                        new_surface = state.foot_surface_height_m.clone()
                        for item in candidates:
                            new_surface[item.foot] = item.surface_height_m
                        edge = FootprintActionEdge(
                            action_key=(
                                action.clip_index, action.start_frame
                            ),
                            candidate_indices=(
                                first_candidate_index,
                                second_candidate_index,
                            ),
                            descriptor_cost=float(descriptor.item()),
                            transition_cost=transition_cost,
                        )
                        expanded.append(
                            _BeamState(
                                root_heading_xy=(
                                    state.root_heading_xy
                                    + action.root_displacement_m[-1]
                                ),
                                foot_surface_height_m=new_surface,
                                contact_frame=(
                                    second_layer.nominal.contact_frame
                                ),
                                footprints=(
                                    *state.footprints,
                                    first_candidate,
                                    second_candidate,
                                ),
                                edges=(*state.edges, edge),
                                action_indices=(
                                    *state.action_indices,
                                    next_action_index,
                                ),
                                candidate_order=(
                                    *state.candidate_order,
                                    first_candidate_index,
                                    second_candidate_index,
                                ),
                                total_cost=(
                                    state.total_cost + edge_increment
                                ),
                            )
                        )
        if not expanded:
            if descriptor_match and graph_rejection:
                raise HeadingPlanningFailure(
                    code="sequence_dead_end",
                    step_index=first_layer.step_index,
                    candidate_count=(
                        len(first_layer.candidates)
                        * len(second_layer.candidates)
                    ),
                    action_count=len(actions),
                    reasons=(
                        "transition graph removed every beam state",
                    ),
                )
            raise HeadingPlanningFailure(
                code="no_motion_coverage",
                step_index=first_layer.step_index,
                candidate_count=(
                    len(first_layer.candidates)
                    * len(second_layer.candidates)
                ),
                action_count=len(actions),
                reasons=(
                    "no contact action satisfies descriptor tolerances",
                ),
            )
        expanded.sort(
            key=lambda state: (
                state.total_cost,
                tuple(edge.action_key for edge in state.edges),
                state.candidate_order,
            )
        )
        beam = tuple(expanded[:beam_width])

    best = beam[0]
    return HeadingFootprintPlan(
        heading_scene_xy=heading,
        footprints=best.footprints,
        edges=best.edges,
        action_keys=tuple(edge.action_key for edge in best.edges),
        total_cost=best.total_cost,
    )
