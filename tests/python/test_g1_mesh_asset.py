import copy
import json
import pathlib
import tempfile
import unittest

from resources.g1_mesh.validate_g1_raylib_glb import validate_g1_glb


ROOT = pathlib.Path(__file__).resolve().parents[2]
ASSET = ROOT / "resources/g1_mesh/g1_raylib.glb"
MANIFEST = ROOT / "resources/g1_mesh/manifest.json"
SOURCE = pathlib.Path("/home/ubuntu/projects/g1_mm/g1.fbx")


class G1MeshAssetTests(unittest.TestCase):
    def test_committed_asset_is_raylib_compatible(self):
        report = validate_g1_glb(ASSET, MANIFEST, SOURCE)
        self.assertEqual(report["skin_count"], 1)
        self.assertEqual(report["bone_count"], 39)
        self.assertEqual(report["primitive_count"], 35)
        self.assertLessEqual(report["maximum_primitive_vertices"], 65535)
        self.assertEqual(report["rigid_vertex_count"], report["vertex_count"])
        self.assertGreaterEqual(report["height_m"], 0.8)
        self.assertLessEqual(report["height_m"], 1.6)

    def test_manifest_names_all_required_articulated_bones_once(self):
        manifest = json.loads(MANIFEST.read_text())
        required = manifest["required_articulated_bones"]
        self.assertEqual(len(required), 30)
        self.assertEqual(len(set(required)), 30)
        self.assertNotIn("Simulation", required)
        self.assertIn("left_ankle_pitch_link", required)
        self.assertIn("left_ankle_roll_link", required)
        self.assertIn("right_ankle_pitch_link", required)
        self.assertIn("right_ankle_roll_link", required)

    def test_corrupt_glb_and_manifest_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            glb = root / "g1.glb"
            manifest = root / "manifest.json"
            glb.write_bytes(ASSET.read_bytes()[:-1] + b"x")
            payload = copy.deepcopy(json.loads(MANIFEST.read_text()))
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "GLB SHA-256"):
                validate_g1_glb(glb, manifest)


if __name__ == "__main__":
    unittest.main()
