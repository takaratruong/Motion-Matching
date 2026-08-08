import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mm_sonic.joints import ContractError
from mm_sonic.torch_object_motion_field import (
    motion_field_intersections,
    rasterized_motion_field,
)

_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "run_g1_motion_field_transition_graph.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_g1_motion_field_transition_graph", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class MotionFieldTransitionGraphTests(unittest.TestCase):
    @staticmethod
    def lines():
        return rasterized_motion_field(
            elevated_scene_xy=[
                (-0.5, -0.5),
                (-0.5, 0.5),
                (0.5, -0.5),
                (0.5, 0.5),
            ],
            heading_degrees=(0.0, 45.0, -45.0),
            spacing_m=0.20,
            approach_margin_m=0.50,
            exit_margin_m=0.50,
        )

    def test_registers_validated_edge_from_actual_scene_endpoints(self):
        lines = self.lines()
        source = [line for line in lines if line.heading_degrees == 0.0][2]
        target = [line for line in lines if line.heading_degrees == 45.0][3]
        source_point = np.asarray(source.start_scene_xy) + 0.4 * (
            np.asarray(source.stop_scene_xy) - source.start_scene_xy
        )
        target_point = np.asarray(target.start_scene_xy) + 0.6 * (
            np.asarray(target.stop_scene_xy) - target.start_scene_xy
        )
        graph = _MODULE.build_transition_graph(
            lines=lines,
            intersections=motion_field_intersections(lines),
            edge_records=(
                {
                    "edge_id": "turn-left",
                    "from_heading_degrees": 0.0,
                    "to_heading_degrees": 45.0,
                    "start_scene_xy": source_point,
                    "end_scene_xy": target_point,
                    "artifact": "turn-left.npz",
                    "artifact_sha256": "a" * 64,
                    "validation": "turn-left-validation.json",
                    "frame_count": 42,
                    "metrics": {"minimum_sole_clearance_m": -0.01},
                    "provenance": [{"clip_index": 7, "start_frame": 10}],
                },
            ),
            maximum_lane_error_m=0.10,
        )

        self.assertEqual(graph["edge_count"], 1)
        edge = graph["edges"][0]
        self.assertEqual(edge["from_line_id"], source.line_id)
        self.assertEqual(edge["to_line_id"], target.line_id)
        self.assertEqual(edge["intersection"]["status"], "associated")
        self.assertEqual(edge["artifact_sha256"], "a" * 64)

    def test_rejects_edge_too_far_from_assigned_lane(self):
        lines = self.lines()
        with self.assertRaisesRegex(ContractError, "lane error"):
            _MODULE.build_transition_graph(
                lines=lines,
                intersections=motion_field_intersections(lines),
                edge_records=(
                    {
                        "edge_id": "bad",
                        "from_heading_degrees": 0.0,
                        "to_heading_degrees": 45.0,
                        "start_scene_xy": (8.0, 8.0),
                        "end_scene_xy": (8.0, 8.0),
                        "artifact": "bad.npz",
                        "artifact_sha256": "b" * 64,
                        "validation": "bad.json",
                        "frame_count": 2,
                        "metrics": {},
                        "provenance": [],
                    },
                ),
                maximum_lane_error_m=0.10,
            )

    def test_load_edge_record_rejects_stale_validation_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "edge.npz"
            np.savez(
                artifact,
                joint_position=np.zeros((2, 29)),
                root_position_world=np.zeros((2, 3)),
                root_orientation_world_wxyz=np.tile((1.0, 0.0, 0.0, 0.0), (2, 1)),
                source_support_mask=np.ones((2, 2), dtype=bool),
            )
            validation = root / "validation.json"
            validation.write_text(
                '{"validated": true, "input": "'
                + str(artifact.resolve())
                + '", "artifact_sha256": "'
                + "0" * 64
                + '", "frame_count": 2, "metrics": {}}'
            )

            with self.assertRaisesRegex(ContractError, "hash mismatch"):
                _MODULE.load_edge_record(
                    specification={
                        "edge_id": "edge",
                        "artifact": str(artifact),
                        "validation": str(validation),
                        "from_heading_degrees": 0.0,
                        "to_heading_degrees": 45.0,
                        "observed_heading_change_degrees": 45.0,
                        "provenance": [],
                    },
                    matcher_to_scene_xy=lambda points: points,
                )

    def test_load_edge_record_rejects_wrong_terrain_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "edge.npz"
            np.savez(
                artifact,
                joint_position=np.zeros((2, 29)),
                root_position_world=np.zeros((2, 3)),
                root_orientation_world_wxyz=np.tile((1.0, 0.0, 0.0, 0.0), (2, 1)),
                source_support_mask=np.ones((2, 2), dtype=bool),
            )
            import hashlib
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            validation = root / "validation.json"
            validation.write_text(
                '{"validated": true, "input": "'
                + str(artifact.resolve())
                + '", "artifact_sha256": "'
                + digest
                + '", "query_scene": "wrong/motion.npz", '
                + '"frame_count": 2, "metrics": {}}'
            )

            with self.assertRaisesRegex(ContractError, "terrain scene mismatch"):
                _MODULE.load_edge_record(
                    specification={
                        "edge_id": "edge",
                        "artifact": str(artifact),
                        "validation": str(validation),
                        "from_heading_degrees": 0.0,
                        "to_heading_degrees": 45.0,
                        "observed_heading_change_degrees": 45.0,
                        "provenance": [],
                    },
                    matcher_to_scene_xy=lambda points: points,
                    expected_query_scene="right/motion.npz",
                )


if __name__ == "__main__":
    unittest.main()
