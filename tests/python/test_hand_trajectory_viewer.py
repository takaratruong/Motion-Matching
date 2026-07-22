import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "hand_trajectory_viewer.cpp"
TRAJECTORY_SOURCE_PATH = ROOT / "interaction_hand_trajectories.cpp"
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")


class HandTrajectoryViewerTests(unittest.TestCase):
    def source(self):
        self.assertTrue(SOURCE_PATH.exists(), "viewer source is missing")
        return SOURCE_PATH.read_text(encoding="utf-8")

    def test_has_complete_object_controls(self):
        source = self.source()
        for key in (
            "KEY_Q", "KEY_E", "KEY_R", "KEY_F", "KEY_Z", "KEY_C",
            "KEY_LEFT", "KEY_RIGHT", "KEY_UP", "KEY_DOWN",
            "KEY_W", "KEY_S", "KEY_PAGE_UP", "KEY_PAGE_DOWN",
            "KEY_LEFT_BRACKET", "KEY_RIGHT_BRACKET", "KEY_SLASH",
            "KEY_ENTER",
            "KEY_V", "KEY_BACKSPACE",
        ):
            self.assertIn(key, source)

    def test_rebuilds_world_grasp_shaped_valid_set_live(self):
        source = self.source()
        self.assertIn("select_hand_trajectories(", source)
        self.assertIn("shape_hand_trajectory(", source)
        self.assertIn("evaluate_shaped_trajectory_feasibility(", source)
        self.assertIn("rebuild_valid_trajectories(", source)
        self.assertIn("if (grasp_changed)", source)
        grasp_change = source.split("if (grasp_changed)", 1)[1].split(
            "if (IsKeyPressed(KEY_ENTER))", 1
        )[0]
        self.assertNotIn("rebuild_valid_trajectories(", grasp_change)
        self.assertIn("search_stale = true", grasp_change)
        enter_search = source.split("if (IsKeyPressed(KEY_ENTER))", 1)[1]
        self.assertIn("rebuild_valid_trajectories(", enter_search)
        self.assertIn("search_stale = false", enter_search)
        self.assertIn("grasp_world_position", source)
        self.assertIn("grasp_world_rotation", source)
        self.assertIn("ik_rejected", source)
        self.assertIn("object_rejected", source)
        self.assertIn("table_rejected", source)
        self.assertIn("valid.empty()", source)
        self.assertIn("constrain_grasp_orientation", source)
        self.assertNotIn("previous_clip", source)
        self.assertNotIn("preserved", source)
        self.assertIn("selected_index = 0U", enter_search)
        self.assertIn("[: previous  / or ]: next  Enter: rerun", source)
        self.assertIn("SEARCH STALE - press Enter", source)

        trajectory_source = TRAJECTORY_SOURCE_PATH.read_text(encoding="utf-8")
        self.assertIn("hand_trajectory_scene_alignment(", trajectory_source)

    def test_builds_recorded_table_and_renders_rejections(self):
        source = self.source()
        self.assertIn("ShelfGeometry", source)
        self.assertIn("make_recorded_table_geometry(", source)
        self.assertNotIn("make_shelf(", source)
        self.assertIn("show_rejected", source)
        self.assertIn("DrawLine3D", source)

    def test_cycles_and_animates_selected_g1_kinematics(self):
        source = self.source()
        self.assertIn("selected_index", source)
        self.assertIn("animation_seconds", source)
        self.assertNotIn("hand_trajectory_world_mapping(", source)
        self.assertIn("shaped.poses", source)
        self.assertIn("world_pose(", source)
        self.assertIn("g1_skeleton::kParents", source)
        self.assertIn("DrawCylinderEx(", source)

    def test_is_independent_of_controller_diffusion_mesh_and_terrain(self):
        source = self.source()
        for forbidden in (
            '#include "interaction_offline_overlap.h"',
            '#include "interaction_learned_pickup_backend.h"',
            '#include "g1_mesh_renderer.h"',
            '#include "terrain_runtime.h"',
            "LoadModel(", "TakeScreenshot(",
        ):
            self.assertNotIn(forbidden, source)

    def test_makefile_has_standalone_target(self):
        self.assertIn("hand_trajectory_viewer:", MAKEFILE)
        target = MAKEFILE.split("hand_trajectory_viewer:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("hand_trajectory_viewer.cpp", target)
        self.assertIn("interaction_hand_trajectories.cpp", target)


if __name__ == "__main__":
    unittest.main()
