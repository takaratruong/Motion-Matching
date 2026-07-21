import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "controller.cpp").read_text(encoding="utf-8")
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")


class NativeG1ControllerTests(unittest.TestCase):
    def test_controller_is_native_g1(self):
        self.assertIn("g1_controller_state state", SOURCE)
        self.assertIn("G1_BoneCount", SOURCE)
        self.assertIn("g1_skeleton_validate", SOURCE)
        self.assertNotIn("FlatControllerPose", SOURCE)
        self.assertNotIn("collapse_interaction_pose", SOURCE)
        self.assertNotIn("expand_flat_controller_pose", SOURCE)

    def test_controller_uses_raylib_6_model_api(self):
        self.assertIn("IsModelValid(", SOURCE)
        self.assertIn("model.skeleton.bones", SOURCE)
        self.assertIn("model.skeleton.bindPose", SOURCE)
        self.assertNotIn("IsModelReady(", SOURCE)

    def test_clearance_kernel_is_compiled_without_fast_math(self):
        self.assertIn(
            "G1_CLEARANCE_OBJECT := build/g1_clearance.o", MAKEFILE)
        self.assertIn(
            "$(G1_CLEARANCE_OBJECT): g1_clearance.cpp", MAKEFILE)
        self.assertNotIn(
            "NATIVE_G1_SOURCES := g1_clearance.cpp", MAKEFILE)


if __name__ == "__main__":
    unittest.main()
