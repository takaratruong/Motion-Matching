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

    def test_native_controller_links_diffusion_without_flat_adapter(self):
        self.assertIn('#include "interaction_native_g1_bridge.h"', SOURCE)
        self.assertIn("interaction::SmartPickupController", SOURCE)
        self.assertIn("interaction::RuntimeInput", SOURCE)
        self.assertIn("capture_native_g1_snapshot(", SOURCE)
        self.assertIn("INTERACTION_NATIVE_G1_SOURCES :=", MAKEFILE)
        controller_sources = MAKEFILE.split(
            "INTERACTION_NATIVE_G1_SOURCES :=", 1)[1].split(
                "HEADER =", 1)[0]
        self.assertNotIn("interaction_controller_adapter.cpp", controller_sources)
        self.assertNotIn("interaction_target_rig_ik.cpp", controller_sources)
        self.assertNotIn("locomotion_controller_update.cpp", controller_sources)

    def test_f_stops_before_native_approach_and_runtime_update(self):
        update_loop = SOURCE.split("auto update_func", 1)[1]
        ordered_tokens = (
            "IsKeyPressed(KEY_F)",
            "smart_pickup_controller.pre_step(",
            "G1CommandIntent command_intent",
            "simulation_positions_update(",
            "capture_native_g1_snapshot(",
            "smart_pickup_controller.post_step(",
            "interaction_runtime.update(interaction_input)",
        )
        for token in ordered_tokens:
            self.assertIn(token, update_loop)
        positions = [update_loop.index(token) for token in ordered_tokens]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("MM_INTERACTION_FUNNEL_CHECKPOINT", SOURCE)
        self.assertIn("MM_INTERACTION_FUNNEL_WORKER", SOURCE)


if __name__ == "__main__":
    unittest.main()
