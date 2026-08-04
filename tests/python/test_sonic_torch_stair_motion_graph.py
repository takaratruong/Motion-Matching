import math
import unittest

from mm_sonic.joints import ContractError
from mm_sonic.torch_stair_motion_graph import (
    StairContactNode,
    StairFoothold,
    StairMotionEdge,
    StairMotionGraph,
    heading_bin_from_yaw,
    merge_stair_motion_graphs,
    stair_motion_graph_from_dict,
)


def _node(*, root_u_cell=2, heading_bin=0, left_tread=1, right_tread=1):
    return StairContactNode(
        root_u_cell=root_u_cell,
        root_v_cell=0,
        root_height_level=1,
        heading_bin=heading_bin,
        velocity_bin=1,
        left_foothold=StairFoothold(left_tread, -1),
        right_foothold=StairFoothold(right_tread, 1),
        support_mask=3,
        last_landing_foot=0,
        gait_phase_bin=0,
    )


class StairMotionGraphTest(unittest.TestCase):
    def test_edge_rejects_nonfinite_sole_clearance(self):
        with self.assertRaisesRegex(ContractError, "motion edge"):
            StairMotionEdge(
                start_node_id="a" * 64,
                end_node_id="b" * 64,
                traversal_heading_bin=0,
                frame_count=10,
                source_artifact_sha256="c" * 64,
                source_start_frame=0,
                source_end_frame_exclusive=11,
                start_boundary_sha256="d" * 64,
                end_boundary_sha256="e" * 64,
                provenance_sha256="f" * 64,
                exact_contact_valid=True,
                minimum_sole_clearance_m=float("nan"),
            )

    def test_heading_quantization_is_eight_way_and_wrapped(self):
        self.assertEqual(heading_bin_from_yaw(0.0), 0)
        self.assertEqual(heading_bin_from_yaw(math.pi / 4), 1)
        self.assertEqual(heading_bin_from_yaw(math.pi), 4)
        self.assertEqual(heading_bin_from_yaw(-math.pi / 2), 6)
        self.assertEqual(heading_bin_from_yaw(2 * math.pi), 0)

    def test_node_identity_preserves_footholds_and_contact_state(self):
        level = _node()
        split = _node(right_tread=2)
        turned = _node(heading_bin=2)

        self.assertNotEqual(level.node_id, split.node_id)
        self.assertNotEqual(level.node_id, turned.node_id)
        self.assertEqual(level.node_id, _node().node_id)

    def test_graph_reports_direction_coverage_per_contact_node(self):
        start = _node()
        forward = _node(root_u_cell=3, heading_bin=0)
        lateral = _node(heading_bin=2, right_tread=2)
        edges = (
            StairMotionEdge(
                start_node_id=start.node_id,
                end_node_id=forward.node_id,
                traversal_heading_bin=0,
                frame_count=25,
                source_artifact_sha256="1" * 64,
                source_start_frame=0,
                source_end_frame_exclusive=25,
                start_boundary_sha256="c" * 64,
                end_boundary_sha256="d" * 64,
                provenance_sha256="a" * 64,
                exact_contact_valid=True,
            ),
            StairMotionEdge(
                start_node_id=start.node_id,
                end_node_id=lateral.node_id,
                traversal_heading_bin=2,
                frame_count=18,
                source_artifact_sha256="2" * 64,
                source_start_frame=5,
                source_end_frame_exclusive=23,
                start_boundary_sha256="e" * 64,
                end_boundary_sha256="f" * 64,
                provenance_sha256="b" * 64,
                exact_contact_valid=True,
            ),
        )
        graph = StairMotionGraph((start, forward, lateral), edges)

        self.assertEqual(
            graph.direction_coverage(start.node_id),
            (True, False, True, False, False, False, False, False),
        )
        self.assertEqual(
            graph.missing_directions(start.node_id),
            (1, 3, 4, 5, 6, 7),
        )
        self.assertEqual(
            graph.coverage_summary()["node_direction_option_histogram"],
            [2, 0, 1, 0, 0, 0, 0, 0, 0],
        )
        restored = stair_motion_graph_from_dict(graph.to_dict())
        self.assertEqual(restored.to_dict(), graph.to_dict())

    def test_graph_rejects_edges_with_unknown_nodes(self):
        start = _node()
        edge = StairMotionEdge(
            start_node_id=start.node_id,
            end_node_id="f" * 64,
            traversal_heading_bin=0,
            frame_count=25,
            source_artifact_sha256="1" * 64,
            source_start_frame=0,
            source_end_frame_exclusive=25,
            start_boundary_sha256="c" * 64,
            end_boundary_sha256="d" * 64,
            provenance_sha256="a" * 64,
            exact_contact_valid=True,
        )

        with self.assertRaisesRegex(ContractError, "unknown"):
            StairMotionGraph((start,), (edge,))

    def test_edge_identity_preserves_exact_boundary_poses(self):
        start = _node()
        end = _node(root_u_cell=3)
        common = dict(
            start_node_id=start.node_id,
            end_node_id=end.node_id,
            traversal_heading_bin=0,
            frame_count=25,
            source_artifact_sha256="1" * 64,
            source_start_frame=0,
            source_end_frame_exclusive=25,
            end_boundary_sha256="d" * 64,
            provenance_sha256="a" * 64,
            exact_contact_valid=True,
        )
        first = StairMotionEdge(
            start_boundary_sha256="b" * 64,
            **common,
        )
        second = StairMotionEdge(
            start_boundary_sha256="c" * 64,
            **common,
        )

        self.assertNotEqual(first.edge_id, second.edge_id)

    def test_merge_deduplicates_shared_nodes_and_preserves_edge_variants(self):
        start = _node()
        end = _node(root_u_cell=3)
        common = dict(
            start_node_id=start.node_id,
            end_node_id=end.node_id,
            traversal_heading_bin=0,
            frame_count=25,
            source_artifact_sha256="1" * 64,
            source_start_frame=0,
            source_end_frame_exclusive=25,
            end_boundary_sha256="d" * 64,
            exact_contact_valid=True,
        )
        first = StairMotionEdge(
            start_boundary_sha256="b" * 64,
            provenance_sha256="a" * 64,
            **common,
        )
        second = StairMotionEdge(
            start_boundary_sha256="c" * 64,
            provenance_sha256="e" * 64,
            **common,
        )

        merged = merge_stair_motion_graphs(
            (
                StairMotionGraph((start, end), (first,)),
                StairMotionGraph((start, end), (second,)),
            )
        )

        self.assertEqual(len(merged.nodes), 2)
        self.assertEqual(len(merged.edges), 2)

    def test_exact_boundary_successor_connects_distinct_quantized_nodes(self):
        start = _node(root_u_cell=1)
        incoming_end = _node(root_u_cell=2, heading_bin=0)
        outgoing_start = _node(root_u_cell=2, heading_bin=2)
        end = _node(root_u_cell=3, heading_bin=2)
        incoming = StairMotionEdge(
            start_node_id=start.node_id,
            end_node_id=incoming_end.node_id,
            traversal_heading_bin=0,
            frame_count=20,
            source_artifact_sha256="1" * 64,
            source_start_frame=0,
            source_end_frame_exclusive=21,
            start_boundary_sha256="a" * 64,
            end_boundary_sha256="b" * 64,
            provenance_sha256="c" * 64,
            exact_contact_valid=True,
        )
        outgoing = StairMotionEdge(
            start_node_id=outgoing_start.node_id,
            end_node_id=end.node_id,
            traversal_heading_bin=2,
            frame_count=20,
            source_artifact_sha256="2" * 64,
            source_start_frame=0,
            source_end_frame_exclusive=21,
            start_boundary_sha256="b" * 64,
            end_boundary_sha256="d" * 64,
            provenance_sha256="e" * 64,
            exact_contact_valid=True,
        )
        graph = StairMotionGraph(
            (start, incoming_end, outgoing_start, end),
            (incoming, outgoing),
        )

        self.assertEqual(
            graph.exact_successor_edges(incoming.edge_id),
            (outgoing,),
        )
        self.assertEqual(
            graph.exact_missing_directions(incoming.edge_id),
            (0, 1, 3, 4, 5, 6, 7),
        )
        matrix = graph.exact_transition_coverage_matrix()
        self.assertEqual(matrix[0][2], 1)
        self.assertEqual(sum(sum(row) for row in matrix), 1)

    def test_quantized_successor_exposes_connector_candidate(self):
        start = _node(root_u_cell=1)
        shared = _node(root_u_cell=2)
        end = _node(root_u_cell=3, heading_bin=2)
        incoming = StairMotionEdge(
            start_node_id=start.node_id,
            end_node_id=shared.node_id,
            traversal_heading_bin=0,
            frame_count=20,
            source_artifact_sha256="1" * 64,
            source_start_frame=0,
            source_end_frame_exclusive=21,
            start_boundary_sha256="a" * 64,
            end_boundary_sha256="b" * 64,
            provenance_sha256="c" * 64,
            exact_contact_valid=True,
        )
        outgoing = StairMotionEdge(
            start_node_id=shared.node_id,
            end_node_id=end.node_id,
            traversal_heading_bin=2,
            frame_count=20,
            source_artifact_sha256="2" * 64,
            source_start_frame=0,
            source_end_frame_exclusive=21,
            start_boundary_sha256="d" * 64,
            end_boundary_sha256="e" * 64,
            provenance_sha256="f" * 64,
            exact_contact_valid=True,
        )
        graph = StairMotionGraph(
            (start, shared, end),
            (incoming, outgoing),
        )

        self.assertEqual(
            graph.quantized_successor_edges(incoming.edge_id),
            (outgoing,),
        )
        matrix = graph.quantized_transition_coverage_matrix()
        self.assertEqual(matrix[0][2], 1)
        self.assertEqual(
            graph.coverage_summary()["quantized_transition_pair_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
