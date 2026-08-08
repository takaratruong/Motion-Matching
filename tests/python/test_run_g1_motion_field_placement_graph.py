import importlib.util
import unittest
from pathlib import Path

_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "run_g1_motion_field_placement_graph.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_g1_motion_field_placement_graph", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class MotionFieldPlacementGraphTests(unittest.TestCase):
    def test_combines_admitted_and_rejected_cells_without_hiding_failures(self):
        reports = (
            {
                "template_edge_id": "positive",
                "query_scene": "scene",
                "candidate_count": 2,
                "accepted_count": 1,
                "records": [
                    {
                        "index": 0,
                        "from_line_id": "h0",
                        "to_line_id": "p0",
                        "validated": True,
                        "artifact": "/tmp/a.npz",
                        "artifact_sha256": "a" * 64,
                        "rejections": [],
                    },
                    {
                        "index": 1,
                        "from_line_id": "h1",
                        "to_line_id": "p1",
                        "validated": False,
                        "rejections": ["terrain_penetration"],
                    },
                ],
            },
        )

        graph = _MODULE.build_placement_graph(reports=reports)

        self.assertEqual(graph["candidate_cell_count"], 2)
        self.assertEqual(graph["edge_count"], 1)
        self.assertEqual(graph["rejected_cell_count"], 1)
        self.assertEqual(graph["cells"][0]["status"], "admitted")
        self.assertEqual(graph["cells"][1]["status"], "rejected")
        self.assertEqual(graph["occupied_line_ids"], ["h0", "p0"])


if __name__ == "__main__":
    unittest.main()
