import importlib.util
import unittest
from pathlib import Path

from mm_sonic.joints import ContractError
from mm_sonic.torch_object_motion_field import rasterized_motion_field

_PATH = Path(__file__).resolve().parents[2] / "resources" / "run_g1_object_motion_field.py"
_SPEC = importlib.util.spec_from_file_location("run_g1_object_motion_field", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class ObjectMotionFieldRunnerTests(unittest.TestCase):
    @staticmethod
    def lines():
        return rasterized_motion_field(
            elevated_scene_xy=[(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)],
            heading_degrees=(0.0, 90.0),
            spacing_m=0.20,
            approach_margin_m=0.25,
            exit_margin_m=0.25,
        )

    def test_manifest_keeps_missing_and_rejected_cells_explicit(self):
        lines = self.lines()
        records = (
            {
                "line_id": lines[0].line_id,
                "heading_degrees": lines[0].heading_degrees,
                "artifact": "valid.npz",
                "classification": "staircase_intersecting",
                "independently_validated": True,
            },
            {
                "line_id": lines[1].line_id,
                "heading_degrees": lines[1].heading_degrees,
                "artifact": "flat.npz",
                "classification": "flat_only_excluded",
                "independently_validated": True,
            },
        )

        manifest = _MODULE.build_motion_field_manifest(lines=lines, route_records=records)

        self.assertEqual(len(manifest["lines"]), len(lines))
        by_id = {item["line_id"]: item for item in manifest["lines"]}
        self.assertEqual(by_id[lines[0].line_id]["status"], "admitted")
        self.assertEqual(by_id[lines[1].line_id]["status"], "rejected")
        self.assertEqual(by_id[lines[2].line_id]["status"], "missing")
        self.assertEqual(manifest["admitted_line_ids"], [lines[0].line_id])

    def test_manifest_rejects_heading_mismatch_and_unknown_lines(self):
        lines = self.lines()
        mismatch = ({
            "line_id": lines[0].line_id,
            "heading_degrees": 45.0,
            "artifact": "wrong.npz",
            "classification": "staircase_intersecting",
            "independently_validated": True,
        },)

        manifest = _MODULE.build_motion_field_manifest(lines=lines, route_records=mismatch)
        self.assertEqual(manifest["lines"][0]["status"], "rejected")
        self.assertEqual(manifest["lines"][0]["reason"], "heading_mismatch")

        unknown = (dict(mismatch[0], line_id="not-a-field-line"),)
        with self.assertRaisesRegex(ContractError, "unknown line"):
            _MODULE.build_motion_field_manifest(lines=lines, route_records=unknown)


if __name__ == "__main__":
    unittest.main()
