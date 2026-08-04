import math
import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_stair_motion_graph_extract import (
    extract_stair_motion_graph,
    landing_decision_frames,
)
from mm_sonic.torch_terrain_omni_routes import StairFrame


def _synthetic_rollout():
    frames = 7
    root = np.zeros((frames, 3), dtype=np.float64)
    root[:, 0] = np.linspace(10.0, 10.6, frames)
    root[:, 1] = 20.0
    root[:, 2] = 0.80
    feet = np.zeros((frames, 2, 3), dtype=np.float64)
    feet[:, 0, :2] = (10.0, 19.9)
    feet[:, 1, :2] = (10.0, 20.1)
    feet[3:, :, 0] += 0.30
    feet[6:, :, 0] += 0.30
    surface = np.zeros((frames, 2), dtype=np.float64)
    surface[3:, 0] = 0.20
    surface[6:, 1] = 0.20
    support = np.asarray(
        [
            [True, True],
            [False, True],
            [False, True],
            [True, True],
            [True, False],
            [True, False],
            [True, True],
        ],
        dtype=np.bool_,
    )
    arrays = {
        "root_position_world": root,
        "root_yaw_world": np.zeros(frames, dtype=np.float64),
        "foot_position_world": feet,
        "foot_surface_height_m": surface,
        "qpos": np.arange(frames * 6, dtype=np.float64).reshape(frames, 6),
        "joint_position": np.zeros((frames, 29), dtype=np.float64),
        "joint_velocity": np.zeros((frames, 3), dtype=np.float64),
        "selected_clip_path": np.asarray(["clip.npz"] * frames),
        "selected_source_frame": np.arange(frames, dtype=np.int64),
        "command_velocity_world_xy": np.tile((1.0, 0.0), (frames, 1)),
    }
    stair = StairFrame(
        origin_world_xy=(10.0, 20.0),
        ascent_world_yaw=0.0,
        width_m=1.0,
        tread_depth_m=0.30,
        riser_height_m=0.20,
        tread_count=4,
    )
    return arrays, support, stair


class StairMotionGraphExtractTest(unittest.TestCase):
    def test_landing_frames_collapse_double_support_plateaus(self):
        _, support, _ = _synthetic_rollout()

        self.assertEqual(landing_decision_frames(support), (0, 3, 6))

    def test_extracts_contact_edges_in_global_stair_frame(self):
        arrays, support, stair = _synthetic_rollout()

        graph = extract_stair_motion_graph(
            arrays,
            support_mask=support,
            stair_frame=stair,
            exact_contact_valid=True,
            source_artifact_sha256="a" * 64,
        )

        self.assertEqual(len(graph.nodes), 3)
        self.assertEqual(len(graph.edges), 2)
        self.assertEqual(
            tuple(edge.traversal_heading_bin for edge in graph.edges),
            (0, 0),
        )
        ordered = sorted(graph.nodes, key=lambda node: node.root_u_cell)
        self.assertEqual(
            tuple(node.root_u_cell for node in ordered),
            (0, 1, 2),
        )
        self.assertEqual(ordered[1].left_foothold.tread_index, 1)
        self.assertEqual(ordered[2].right_foothold.tread_index, 1)
        self.assertNotEqual(
            graph.edges[0].start_boundary_sha256,
            graph.edges[0].end_boundary_sha256,
        )
        self.assertEqual(graph.edges[0].frame_count, 4)
        self.assertEqual(graph.edges[0].source_artifact_sha256, "a" * 64)
        self.assertTrue(
            all(node.root_height_level == 4 for node in graph.nodes)
        )

    def test_rotated_stair_still_quantizes_ascent_as_heading_zero(self):
        arrays, support, stair = _synthetic_rollout()
        stair = StairFrame(
            origin_world_xy=stair.origin_world_xy,
            ascent_world_yaw=math.pi / 2,
            width_m=stair.width_m,
            tread_depth_m=stair.tread_depth_m,
            riser_height_m=stair.riser_height_m,
            tread_count=stair.tread_count,
        )
        root = arrays["root_position_world"].copy()
        root[:, 0] = 10.0
        root[:, 1] = np.linspace(20.0, 20.6, len(root))
        feet = arrays["foot_position_world"].copy()
        relative = feet[:, :, :2] - (10.0, 20.0)
        feet[:, :, 0] = 10.0 - relative[:, :, 1]
        feet[:, :, 1] = 20.0 + relative[:, :, 0]
        arrays = {
            **arrays,
            "root_position_world": root,
            "root_yaw_world": np.full(len(root), math.pi / 2),
            "foot_position_world": feet,
            "command_velocity_world_xy": np.tile((0.0, 1.0), (len(root), 1)),
        }

        graph = extract_stair_motion_graph(
            arrays,
            support_mask=support,
            stair_frame=stair,
            exact_contact_valid=True,
            source_artifact_sha256="a" * 64,
        )

        self.assertTrue(all(node.heading_bin == 0 for node in graph.nodes))
        self.assertTrue(
            all(edge.traversal_heading_bin == 0 for edge in graph.edges)
        )

    def test_refuses_unvalidated_rollout(self):
        arrays, support, stair = _synthetic_rollout()

        with self.assertRaisesRegex(ContractError, "exact-contact"):
            extract_stair_motion_graph(
                arrays,
                support_mask=support,
                stair_frame=stair,
                exact_contact_valid=False,
                source_artifact_sha256="a" * 64,
            )

    def test_missing_dynamic_state_array_raises_contract_error(self):
        arrays, support, stair = _synthetic_rollout()
        del arrays["qpos"]

        with self.assertRaisesRegex(ContractError, "state arrays"):
            extract_stair_motion_graph(
                arrays,
                support_mask=support,
                stair_frame=stair,
                exact_contact_valid=True,
                source_artifact_sha256="a" * 64,
            )

    def test_boundary_identity_ignores_clip_provenance(self):
        arrays, support, stair = _synthetic_rollout()
        first = extract_stair_motion_graph(
            arrays,
            support_mask=support,
            stair_frame=stair,
            exact_contact_valid=True,
            source_artifact_sha256="a" * 64,
        )
        changed = {
            **arrays,
            "selected_clip_path": np.asarray(["other.npz"] * len(support)),
            "selected_source_frame": np.arange(
                100,
                100 + len(support),
                dtype=np.int64,
            ),
        }
        second = extract_stair_motion_graph(
            changed,
            support_mask=support,
            stair_frame=stair,
            exact_contact_valid=True,
            source_artifact_sha256="b" * 64,
        )
        first_edge = next(
            edge for edge in first.edges if edge.source_start_frame == 0
        )
        second_edge = next(
            edge for edge in second.edges if edge.source_start_frame == 0
        )

        self.assertEqual(
            first_edge.start_boundary_sha256,
            second_edge.start_boundary_sha256,
        )
        self.assertEqual(
            first_edge.end_boundary_sha256,
            second_edge.end_boundary_sha256,
        )
        self.assertNotEqual(
            first_edge.provenance_sha256,
            second_edge.provenance_sha256,
        )

    def test_rejects_edge_with_any_full_sole_penetration_frame(self):
        arrays, support, stair = _synthetic_rollout()
        clearance = np.zeros(len(support), dtype=np.float64)
        clearance[1] = -0.20

        graph = extract_stair_motion_graph(
            arrays,
            support_mask=support,
            stair_frame=stair,
            exact_contact_valid=True,
            source_artifact_sha256="a" * 64,
            minimum_sole_clearance_by_frame=clearance,
            minimum_sole_clearance_m=-0.025,
        )

        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges[0].source_start_frame, 3)
        self.assertEqual(graph.edges[0].minimum_sole_clearance_m, 0.0)

    def test_rejects_edge_with_output_joint_speed_spike(self):
        arrays, support, stair = _synthetic_rollout()
        joints = arrays["joint_position"].copy()
        joints[1, 0] = 0.80
        arrays = {**arrays, "joint_position": joints}

        graph = extract_stair_motion_graph(
            arrays,
            support_mask=support,
            stair_frame=stair,
            exact_contact_valid=True,
            source_artifact_sha256="a" * 64,
            minimum_sole_clearance_by_frame=np.zeros(len(support)),
            minimum_sole_clearance_m=-0.025,
            maximum_joint_speed_rad_s=13.0,
        )

        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges[0].source_start_frame, 3)
        self.assertEqual(graph.edges[0].maximum_joint_speed_rad_s, 0.0)


if __name__ == "__main__":
    unittest.main()
