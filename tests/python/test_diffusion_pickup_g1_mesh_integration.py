import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "controller.cpp").read_text(encoding="utf-8")


class DiffusionPickupG1MeshIntegrationTests(unittest.TestCase):
    def test_mesh_consumes_only_final_global_pose(self):
        self.assertIn('#include "g1_mesh_renderer.h"', SOURCE)
        self.assertIn("bool show_g1_mesh = true;", SOURCE)
        self.assertIn("bool show_g1_bones = false;", SOURCE)
        self.assertEqual(SOURCE.count("IsKeyPressed(KEY_M)"), 1)
        self.assertEqual(SOURCE.count("IsKeyPressed(KEY_B)"), 1)

        update = SOURCE.index("::g1_mesh_renderer_update(")
        update_end = SOURCE.index(");", update)
        call = SOURCE[update:update_end]
        self.assertIn("global_bone_positions", call)
        self.assertIn("global_bone_rotations", call)
        self.assertNotIn("adjusted_bone", call)

        begin = SOURCE.index("BeginMode3D(camera);", update)
        draw = SOURCE.index(
            "::g1_mesh_renderer_draw(g1_mesh_renderer);", begin
        )
        end = SOURCE.index("EndMode3D();", draw)
        self.assertLess(update, begin)
        self.assertLess(begin, draw)
        self.assertLess(draw, end)

    def test_mesh_has_explicit_lifecycle(self):
        load = SOURCE.index("::g1_mesh_renderer_load(")
        loop = SOURCE.index("auto update_func = [&]()")
        unload = SOURCE.index("::g1_mesh_renderer_unload(g1_mesh_renderer);")
        normal_unload = SOURCE.index("unload_g1_mesh();", loop)
        close = SOURCE.index("CloseWindow();", normal_unload)
        self.assertLess(load, loop)
        self.assertLess(unload, load)
        self.assertLess(loop, normal_unload)
        self.assertLess(normal_unload, close)
        self.assertEqual(
            SOURCE.count("::g1_mesh_renderer_unload(g1_mesh_renderer);"), 1
        )
        self.assertEqual(SOURCE.count("unload_g1_mesh();"), 3)


if __name__ == "__main__":
    unittest.main()
