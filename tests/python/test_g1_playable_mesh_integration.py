import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "controller.cpp").read_text(encoding="utf-8")


class G1PlayableMeshIntegrationTests(unittest.TestCase):
    def test_classic_ui_and_control_contract_remains(self):
        for text in (
            '"terrain scene / runtime"',
            '"controls"',
            '"WASD / left stick - move"',
            '"Arrows / right stick - camera"',
            '"Left trigger - strafe"',
            '"run sideways speed"',
            '"walk sideways speed"',
            "G1CommandIntent command_intent;",
            "predicted_desired_headings",
        ):
            self.assertIn(text, SOURCE)
        self.assertNotIn("Transactional terrain IK", SOURCE)
        self.assertNotIn("frame_runtime", SOURCE)

    def test_mesh_uses_only_the_final_stable_global_pose(self):
        self.assertIn('#include "g1_mesh_renderer.h"', SOURCE)
        self.assertEqual(
            SOURCE.count(
                "if (::IsKeyPressed(KEY_M)) show_g1_mesh = !show_g1_mesh;"
            ),
            1,
        )
        self.assertEqual(
            SOURCE.count(
                "if (::IsKeyPressed(KEY_B)) show_g1_bones = !show_g1_bones;"
            ),
            1,
        )
        self.assertIn("bool show_g1_mesh = true;", SOURCE)
        self.assertIn("bool show_g1_bones = true;", SOURCE)

        update = SOURCE.index("::g1_mesh_renderer_update(")
        update_end = SOURCE.index(");", update)
        update_call = SOURCE[update:update_end]
        self.assertIn("state.global_bone_positions", update_call)
        self.assertIn("state.global_bone_rotations", update_call)
        self.assertNotIn("adjusted_bone_positions", update_call)
        self.assertNotIn("adjusted_bone_rotations", update_call)

        begin = SOURCE.index("BeginMode3D(camera);", update)
        draw = SOURCE.index(
            "::g1_mesh_renderer_draw(g1_mesh_renderer);", begin
        )
        end = SOURCE.index("EndMode3D();", draw)
        self.assertLess(update, begin)
        self.assertLess(begin, draw)
        self.assertLess(draw, end)

    def test_mesh_lifecycle_uses_existing_normal_cleanup(self):
        load = SOURCE.index("::g1_mesh_renderer_load(")
        loop = SOURCE.index("auto update_func = [&]()")
        cleanup = SOURCE.index("auto normal_cleanup = [&]()")
        unload = SOURCE.index(
            "::g1_mesh_renderer_unload(g1_mesh_renderer);", cleanup
        )
        terrain_unload = SOURCE.index("model_unloader(terrain_model);", unload)
        close = SOURCE.index("CloseWindow();", terrain_unload)
        self.assertLess(load, loop)
        self.assertLess(cleanup, unload)
        self.assertLess(unload, terrain_unload)
        self.assertLess(terrain_unload, close)
        self.assertEqual(
            SOURCE.count("::g1_mesh_renderer_unload(g1_mesh_renderer);"), 1
        )


if __name__ == "__main__":
    unittest.main()
