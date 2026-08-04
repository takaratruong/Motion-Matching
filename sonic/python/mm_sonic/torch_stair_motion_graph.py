"""Stair-relative contact-state graph primitives for offline motion synthesis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping, Sequence

from .joints import ContractError


HEADING_BIN_COUNT = 8
_SHA256_HEX_LENGTH = 64


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def heading_bin_from_yaw(yaw_relative_to_ascent_rad: float) -> int:
    """Quantize a wrapped stair-relative yaw to the nearest 45-degree bin."""

    if (
        isinstance(yaw_relative_to_ascent_rad, bool)
        or not isinstance(yaw_relative_to_ascent_rad, (int, float))
        or not math.isfinite(float(yaw_relative_to_ascent_rad))
    ):
        raise ContractError("stair graph heading yaw is invalid")
    bin_width = 2.0 * math.pi / HEADING_BIN_COUNT
    return int(math.floor(float(yaw_relative_to_ascent_rad) / bin_width + 0.5)) % (
        HEADING_BIN_COUNT
    )


@dataclass(frozen=True, order=True)
class StairFoothold:
    tread_index: int
    lateral_cell: int

    def __post_init__(self) -> None:
        if (
            type(self.tread_index) is not int
            or type(self.lateral_cell) is not int
        ):
            raise ContractError("stair graph foothold is invalid")


@dataclass(frozen=True, order=True)
class StairContactNode:
    root_u_cell: int
    root_v_cell: int
    root_height_level: int
    heading_bin: int
    velocity_bin: int
    left_foothold: StairFoothold
    right_foothold: StairFoothold
    support_mask: int
    last_landing_foot: int
    gait_phase_bin: int

    def __post_init__(self) -> None:
        if (
            type(self.root_u_cell) is not int
            or type(self.root_v_cell) is not int
            or type(self.root_height_level) is not int
            or self.root_height_level < 0
            or type(self.heading_bin) is not int
            or not 0 <= self.heading_bin < HEADING_BIN_COUNT
            or type(self.velocity_bin) is not int
            or not 0 <= self.velocity_bin <= 2
            or not isinstance(self.left_foothold, StairFoothold)
            or not isinstance(self.right_foothold, StairFoothold)
            or type(self.support_mask) is not int
            or not 1 <= self.support_mask <= 3
            or type(self.last_landing_foot) is not int
            or self.last_landing_foot not in (-1, 0, 1)
            or type(self.gait_phase_bin) is not int
            or not 0 <= self.gait_phase_bin < HEADING_BIN_COUNT
        ):
            raise ContractError("stair graph contact node is invalid")

    @property
    def node_id(self) -> str:
        return _canonical_sha256(
            {
                "schema": "g1-stair-contact-node/v1",
                **asdict(self),
            }
        )


@dataclass(frozen=True, order=True)
class StairMotionEdge:
    start_node_id: str
    end_node_id: str
    traversal_heading_bin: int
    frame_count: int
    source_artifact_sha256: str
    source_start_frame: int
    source_end_frame_exclusive: int
    start_boundary_sha256: str
    end_boundary_sha256: str
    provenance_sha256: str
    exact_contact_valid: bool
    minimum_sole_clearance_m: float | None = None
    maximum_joint_speed_rad_s: float | None = None
    maximum_joint_acceleration_rad_s2: float | None = None
    maximum_terminal_sole_contact_error_m: float | None = None
    artifact_kind: str = "source"

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.start_node_id)
            or not _is_sha256(self.end_node_id)
            or type(self.traversal_heading_bin) is not int
            or not 0 <= self.traversal_heading_bin < HEADING_BIN_COUNT
            or type(self.frame_count) is not int
            or self.frame_count < 1
            or not _is_sha256(self.source_artifact_sha256)
            or type(self.source_start_frame) is not int
            or self.source_start_frame < 0
            or type(self.source_end_frame_exclusive) is not int
            or self.source_end_frame_exclusive <= self.source_start_frame
            or not _is_sha256(self.start_boundary_sha256)
            or not _is_sha256(self.end_boundary_sha256)
            or not _is_sha256(self.provenance_sha256)
            or type(self.exact_contact_valid) is not bool
            or (
                self.minimum_sole_clearance_m is not None
                and (
                    isinstance(self.minimum_sole_clearance_m, bool)
                    or not isinstance(
                        self.minimum_sole_clearance_m,
                        (int, float),
                    )
                    or not math.isfinite(
                        float(self.minimum_sole_clearance_m)
                    )
                )
            )
            or (
                self.maximum_joint_speed_rad_s is not None
                and (
                    isinstance(self.maximum_joint_speed_rad_s, bool)
                    or not isinstance(
                        self.maximum_joint_speed_rad_s,
                        (int, float),
                    )
                    or not math.isfinite(
                        float(self.maximum_joint_speed_rad_s)
                    )
                    or float(self.maximum_joint_speed_rad_s) < 0.0
                )
            )
            or (
                self.maximum_joint_acceleration_rad_s2 is not None
                and (
                    isinstance(
                        self.maximum_joint_acceleration_rad_s2,
                        bool,
                    )
                    or not isinstance(
                        self.maximum_joint_acceleration_rad_s2,
                        (int, float),
                    )
                    or not math.isfinite(
                        float(self.maximum_joint_acceleration_rad_s2)
                    )
                    or float(self.maximum_joint_acceleration_rad_s2) < 0.0
                )
            )
            or (
                self.maximum_terminal_sole_contact_error_m is not None
                and (
                    isinstance(
                        self.maximum_terminal_sole_contact_error_m,
                        bool,
                    )
                    or not isinstance(
                        self.maximum_terminal_sole_contact_error_m,
                        (int, float),
                    )
                    or not math.isfinite(
                        float(
                            self.maximum_terminal_sole_contact_error_m
                        )
                    )
                    or float(
                        self.maximum_terminal_sole_contact_error_m
                    )
                    < 0.0
                )
            )
            or self.artifact_kind not in ("source", "optimized")
        ):
            raise ContractError("stair graph motion edge is invalid")

    @property
    def edge_id(self) -> str:
        return _canonical_sha256(
            {
                "schema": "g1-stair-motion-edge/v3",
                **asdict(self),
            }
        )


class StairMotionGraph:
    """Immutable directed graph with eight-way outgoing coverage."""

    def __init__(
        self,
        nodes: Sequence[StairContactNode],
        edges: Sequence[StairMotionEdge],
    ) -> None:
        owned_nodes = tuple(nodes)
        owned_edges = tuple(edges)
        if any(not isinstance(node, StairContactNode) for node in owned_nodes):
            raise ContractError("stair graph nodes are invalid")
        if any(not isinstance(edge, StairMotionEdge) for edge in owned_edges):
            raise ContractError("stair graph edges are invalid")
        by_id = {node.node_id: node for node in owned_nodes}
        if len(by_id) != len(owned_nodes):
            raise ContractError("stair graph contains duplicate nodes")
        known = frozenset(by_id)
        if any(
            edge.start_node_id not in known or edge.end_node_id not in known
            for edge in owned_edges
        ):
            raise ContractError("stair graph edge references an unknown node")
        if any(not edge.exact_contact_valid for edge in owned_edges):
            raise ContractError("stair graph contains an invalid contact edge")
        edge_ids = {edge.edge_id for edge in owned_edges}
        if len(edge_ids) != len(owned_edges):
            raise ContractError("stair graph contains duplicate edges")
        adjacency: dict[str, list[StairMotionEdge]] = {
            node_id: [] for node_id in known
        }
        for edge in owned_edges:
            adjacency[edge.start_node_id].append(edge)
        self.nodes = tuple(sorted(owned_nodes, key=lambda node: node.node_id))
        self.edges = tuple(sorted(owned_edges, key=lambda edge: edge.edge_id))
        self.node_by_id = MappingProxyType(by_id)
        self.edge_by_id = MappingProxyType(
            {edge.edge_id: edge for edge in owned_edges}
        )
        self.edges_by_start = MappingProxyType(
            {
                node_id: tuple(
                    sorted(values, key=lambda edge: edge.edge_id)
                )
                for node_id, values in adjacency.items()
            }
        )
        by_start_boundary: dict[str, list[StairMotionEdge]] = {}
        for edge in owned_edges:
            by_start_boundary.setdefault(
                edge.start_boundary_sha256,
                [],
            ).append(edge)
        self.edges_by_start_boundary = MappingProxyType(
            {
                boundary: tuple(
                    sorted(values, key=lambda edge: edge.edge_id)
                )
                for boundary, values in by_start_boundary.items()
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "g1-stair-motion-graph/v3",
            "nodes": [
                {
                    "node_id": node.node_id,
                    **asdict(node),
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "edge_id": edge.edge_id,
                    **asdict(edge),
                }
                for edge in self.edges
            ],
        }

    def direction_coverage(self, node_id: str) -> tuple[bool, ...]:
        if node_id not in self.node_by_id:
            raise ContractError("stair graph coverage node is unknown")
        covered = [False] * HEADING_BIN_COUNT
        for edge in self.edges_by_start[node_id]:
            covered[edge.traversal_heading_bin] = True
        return tuple(covered)

    def exact_successor_edges(
        self,
        incoming_edge_id: str,
    ) -> tuple[StairMotionEdge, ...]:
        if incoming_edge_id not in self.edge_by_id:
            raise ContractError("stair graph incoming edge is unknown")
        incoming = self.edge_by_id[incoming_edge_id]
        return self.edges_by_start_boundary.get(
            incoming.end_boundary_sha256,
            (),
        )

    def quantized_successor_edges(
        self,
        incoming_edge_id: str,
    ) -> tuple[StairMotionEdge, ...]:
        """Return edges that start at the incoming edge's contact-state cell."""

        if incoming_edge_id not in self.edge_by_id:
            raise ContractError("stair graph incoming edge is unknown")
        incoming = self.edge_by_id[incoming_edge_id]
        return self.edges_by_start[incoming.end_node_id]

    def exact_missing_directions(
        self,
        incoming_edge_id: str,
    ) -> tuple[int, ...]:
        covered = {
            edge.traversal_heading_bin
            for edge in self.exact_successor_edges(incoming_edge_id)
        }
        return tuple(
            direction
            for direction in range(HEADING_BIN_COUNT)
            if direction not in covered
        )

    def quantized_transition_coverage_matrix(
        self,
    ) -> tuple[tuple[int, ...], ...]:
        matrix = [
            [0 for _ in range(HEADING_BIN_COUNT)]
            for _ in range(HEADING_BIN_COUNT)
        ]
        for incoming in self.edges:
            for outgoing in self.quantized_successor_edges(incoming.edge_id):
                matrix[incoming.traversal_heading_bin][
                    outgoing.traversal_heading_bin
                ] += 1
        return tuple(tuple(row) for row in matrix)

    def exact_transition_coverage_matrix(self) -> tuple[tuple[int, ...], ...]:
        matrix = [
            [0 for _ in range(HEADING_BIN_COUNT)]
            for _ in range(HEADING_BIN_COUNT)
        ]
        for incoming in self.edges:
            for outgoing in self.exact_successor_edges(incoming.edge_id):
                matrix[incoming.traversal_heading_bin][
                    outgoing.traversal_heading_bin
                ] += 1
        return tuple(tuple(row) for row in matrix)

    def missing_directions(self, node_id: str) -> tuple[int, ...]:
        return tuple(
            index
            for index, present in enumerate(
                self.direction_coverage(node_id)
            )
            if not present
        )

    def coverage_summary(self) -> dict[str, object]:
        direction_edge_count = [
            sum(
                edge.traversal_heading_bin == direction
                for edge in self.edges
            )
            for direction in range(HEADING_BIN_COUNT)
        ]
        option_histogram = [0] * (HEADING_BIN_COUNT + 1)
        for node in self.nodes:
            option_count = sum(self.direction_coverage(node.node_id))
            option_histogram[option_count] += 1
        exact_transitions = self.exact_transition_coverage_matrix()
        quantized_transitions = self.quantized_transition_coverage_matrix()
        measured_clearances = sorted(
            float(edge.minimum_sole_clearance_m)
            for edge in self.edges
            if edge.minimum_sole_clearance_m is not None
        )
        measured_joint_speeds = sorted(
            float(edge.maximum_joint_speed_rad_s)
            for edge in self.edges
            if edge.maximum_joint_speed_rad_s is not None
        )
        measured_joint_accelerations = sorted(
            float(edge.maximum_joint_acceleration_rad_s2)
            for edge in self.edges
            if edge.maximum_joint_acceleration_rad_s2 is not None
        )
        return {
            "direction_edge_count": direction_edge_count,
            "covered_direction_count": sum(
                count > 0 for count in direction_edge_count
            ),
            "node_direction_option_histogram": option_histogram,
            "nodes_with_multiple_directions": sum(option_histogram[2:]),
            "fully_omnidirectional_node_count": option_histogram[
                HEADING_BIN_COUNT
            ],
            "exact_transition_pair_count": sum(
                count > 0
                for row in exact_transitions
                for count in row
            ),
            "exact_transition_coverage_matrix": [
                list(row) for row in exact_transitions
            ],
            "quantized_transition_pair_count": sum(
                count > 0
                for row in quantized_transitions
                for count in row
            ),
            "quantized_transition_coverage_matrix": [
                list(row) for row in quantized_transitions
            ],
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "sole_clearance_edge_count": len(measured_clearances),
            "minimum_edge_sole_clearance_m": (
                measured_clearances[0] if measured_clearances else None
            ),
            "joint_speed_edge_count": len(measured_joint_speeds),
            "maximum_edge_joint_speed_rad_s": (
                measured_joint_speeds[-1]
                if measured_joint_speeds
                else None
            ),
            "joint_acceleration_edge_count": len(
                measured_joint_accelerations
            ),
            "maximum_edge_joint_acceleration_rad_s2": (
                measured_joint_accelerations[-1]
                if measured_joint_accelerations
                else None
            ),
        }


def stair_motion_graph_from_dict(value: object) -> StairMotionGraph:
    """Reconstruct and authenticate a serialized stair-motion graph."""

    if (
        not isinstance(value, dict)
        or value.get("schema") != "g1-stair-motion-graph/v3"
        or not isinstance(value.get("nodes"), list)
        or not isinstance(value.get("edges"), list)
    ):
        raise ContractError("serialized stair graph is invalid")
    try:
        nodes = []
        for raw in value["nodes"]:
            fields = dict(raw)
            claimed_id = fields.pop("node_id")
            fields["left_foothold"] = StairFoothold(
                **fields["left_foothold"]
            )
            fields["right_foothold"] = StairFoothold(
                **fields["right_foothold"]
            )
            node = StairContactNode(**fields)
            if node.node_id != claimed_id:
                raise ContractError(
                    "serialized stair graph node identity is invalid"
                )
            nodes.append(node)
        edges = []
        for raw in value["edges"]:
            fields = dict(raw)
            claimed_id = fields.pop("edge_id")
            edge = StairMotionEdge(**fields)
            if edge.edge_id != claimed_id:
                raise ContractError(
                    "serialized stair graph edge identity is invalid"
                )
            edges.append(edge)
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("serialized stair graph is invalid") from error
    return StairMotionGraph(tuple(nodes), tuple(edges))


def merge_stair_motion_graphs(
    graphs: Sequence[StairMotionGraph],
) -> StairMotionGraph:
    """Merge exact graph shards without discarding boundary-pose variants."""

    owned = tuple(graphs)
    if any(not isinstance(graph, StairMotionGraph) for graph in owned):
        raise ContractError("stair graph merge inputs are invalid")
    nodes = {
        node.node_id: node
        for graph in owned
        for node in graph.nodes
    }
    edges = {
        edge.edge_id: edge
        for graph in owned
        for edge in graph.edges
    }
    return StairMotionGraph(tuple(nodes.values()), tuple(edges.values()))
