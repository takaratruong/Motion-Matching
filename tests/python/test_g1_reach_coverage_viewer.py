import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
VIEWER = ROOT / "g1_reach_coverage_viewer.cpp"
PROBE = ROOT / "g1_reach_coverage_probe.cpp"
REACH_COVERAGE = ROOT / "reach_coverage.cpp"
TRAJECTORY_HEADER = ROOT / "interaction_hand_trajectories.h"
MAKEFILE = ROOT / "Makefile"
MESH_TEST = ROOT / "tests" / "cpp" / "test_g1_mesh_renderer.cpp"


class G1ReachCoverageViewerTests(unittest.TestCase):
    def source(self, path):
        self.assertTrue(path.exists(), f"missing {path.name}")
        return path.read_text(encoding="utf-8")

    def test_flat_viewer_contract(self):
        source = self.source(VIEWER)
        for required in (
            "reach::load_pack(", "reach::search_all(",
            "reach::regenerate(", "KEY_ENTER", "KEY_SLASH",
            "KEY_LEFT_BRACKET", "KEY_RIGHT_BRACKET", "KEY_G",
            "KEY_Q", "KEY_E", "KEY_R", "KEY_F", "KEY_Z", "KEY_C",
            "KEY_LEFT", "KEY_RIGHT", "KEY_UP", "KEY_DOWN", "KEY_W",
            "KEY_S", "KEY_V", "KEY_BACKSPACE", "DrawCylinderEx(",
            "DrawLine3D(", "CAPTURED LEFT", "MIRRORED RIGHT",
            "POSITION ERROR", "APPROACH AXIS", "FULL ORIENTATION",
            "OBJECT COLLISION", "ENVIRONMENT COLLISION",
            "SEARCH STALE - press Enter", "--object-size",
            "selected_evaluation",
            "REJECTED - motion shown for diagnosis",
            "ClearBackground(Color{238, 241, 245, 255})",
            "DrawSphereWires(", "DARKBLUE", "SKYBLUE", "LIME",
            "grasp_approach_local", "use_coverage_environment = true",
            "evaluation_hand(pack, evaluation)",
            "kWristContactOffset", "SEARCH INCOMPLETE",
            "RAW INSTANCES", "PROCESSED", "YAW PLACEMENT",
            "PRESS ENTER TO SEARCH", "displayed", "accepted",
            "std::optional<reach::SearchResult>",
            "results = std::move(completed)",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "terrain_runtime", "TakeScreenshot(",
            "controller.cpp", "reach::select_candidates(",
            "use_coverage_environment = false",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn(
            "evaluation.rejection == reach::Rejection::None ? LIME : ORANGE",
            source,
        )
        self.assertIn("object_collision_observed", source)
        self.assertIn("environment_collision_observed", source)
        self.assertIn("OBSERVED OBJECT COLLISION", source)
        self.assertIn("OBSERVED ENVIRONMENT COLLISION", source)
        self.assertIn(
            "results.has_value() && !results->accepted.empty() && !stale",
            source,
        )

    def test_rotated_object_has_no_axis_aligned_solid(self):
        source = self.source(VIEWER)
        self.assertEqual(source.count("DrawCubeV("), 1)

    def test_rejected_candidate_is_individually_inspectable(self):
        source = self.source(VIEWER)
        for required in (
            "rejected_indices", "selected_rejected",
            "KEY_COMMA", "KEY_PERIOD",
        ):
            self.assertIn(required, source)

    def test_lowered_scene_and_mouse_camera_contract(self):
        source = self.source(VIEWER)
        for required in (
            "kTableTop = 0.65F", "kTableCenterY = 0.62F",
            "OrbitCameraState", "update_orbit_camera(",
            "GetMouseDelta()", "GetMouseWheelMove()",
            "MOUSE_BUTTON_LEFT", "MOUSE_BUTTON_MIDDLE",
            "std::clamp", "camera.target",
            "std::remainder", "kCameraTargetLimit",
        ):
            self.assertIn(required, source)
        self.assertIn("state.target.x = std::clamp", source)
        self.assertIn("state.target.y = std::clamp", source)
        self.assertIn("state.target.z = std::clamp", source)
        self.assertNotIn("constexpr float table_top = 0.77F", source)

    def test_mesh_uses_the_exact_selected_world_pose_and_safe_lifecycle(self):
        source = self.source(VIEWER)
        for required in (
            '#include "g1_mesh_renderer.h"',
            '"resources/g1_mesh/g1_raylib.glb"',
            "bool show_g1_mesh = true", "bool show_g1_bones = false",
            "IsKeyPressed(KEY_M)", "IsKeyPressed(KEY_B)",
            "::g1_mesh_renderer_load(", "::g1_mesh_renderer_update(",
            "::g1_mesh_renderer_draw(", "::g1_mesh_renderer_unload(",
            "mesh_world_pose.positions", "mesh_world_pose.rotations",
            "G1 MESH DISABLED", "show_g1_bones = true",
        ):
            self.assertIn(required, source)
        update = source.index("::g1_mesh_renderer_update(")
        begin_drawing = source.index("BeginDrawing();", update)
        begin_3d = source.index("BeginMode3D(camera);", begin_drawing)
        draw = source.index("::g1_mesh_renderer_draw(", begin_3d)
        end_3d = source.index("EndMode3D();", draw)
        unload = source.index("::g1_mesh_renderer_unload(")
        close = source.index("CloseWindow();", unload)
        init = source.index("InitWindow(")
        ready = source.index("IsWindowReady()", init)
        load = source.index("::g1_mesh_renderer_load(", ready)
        self.assertLess(init, ready)
        self.assertLess(ready, load)
        self.assertLess(update, begin_drawing)
        self.assertLess(begin_3d, draw)
        self.assertLess(draw, end_3d)
        self.assertLess(unload, close)

    def test_renderer_test_exercises_certified_glb_runtime_lifecycle(self):
        source = self.source(MESH_TEST)
        for required in (
            "int main(int argc, char** argv)", "FLAG_WINDOW_HIDDEN",
            "IsWindowReady()", "g1_mesh_renderer_load(",
            "g1_mesh_renderer_update(", "g1_mesh_renderer_draw(",
            "g1_mesh_renderer_unload(", "BeginDrawing()", "CloseWindow()",
        ):
            self.assertIn(required, source)

    def test_reach_collision_uses_final_contact_only(self):
        self.assertNotIn(
            "active_object_contact_window_samples",
            self.source(REACH_COVERAGE),
        )
        self.assertNotIn(
            "active_object_contact_window_samples",
            self.source(TRAJECTORY_HEADER),
        )

    def test_reach_shaping_uses_bilateral_posture_ik(self):
        source = self.source(REACH_COVERAGE)
        shaped = source.split("Evaluation shape_candidate(", 1)[1].split(
            "Evaluation evaluate_candidate(", 1
        )[0]
        for required in (
            '#include "interaction_posture_ik.h"',
            "solve_hand_posture_ik_task_priority(",
            "decompose_upper_body(",
            "kPostureCorrectionSeconds = 0.6F",
        ):
            self.assertIn(required, source)
        self.assertNotIn("solve_hand_ik(", shaped)

    def test_probe_contract(self):
        source = self.source(PROBE)
        for required in (
            "reach::load_pack(", "reach::search_all(", "--json",
            "shared_grasps", "open_space", "table", "shelf",
            "below_table", "lower_table", "raw_instances",
            "processed_instances", "root_azimuth_sectors",
            "elapsed_seconds", "complete", "4608", "0.001F", "30",
            "observed_collisions",
            "coverage_demonstrated", "search_integrity_passed", "null",
            "kTableTop = 0.65F", "supported_center", "coverage.boxes",
        ):
            self.assertIn(required, source)
        for forbidden in ("own_query(", "position_perturbations"):
            self.assertNotIn(forbidden, source)

    def test_probe_enforces_the_measured_open_space_baseline(self):
        source = self.source(PROBE)
        for required in (
            "constexpr size_t kMinimumOpenAccepted = 94U;",
            "constexpr size_t kMinimumOpenAcceptedPerHand = 47U;",
            "constexpr size_t kMinimumOpenAzimuthSectors = 5U;",
            "report.accepted >= kMinimumOpenAccepted",
            "report.hands[0] >= kMinimumOpenAcceptedPerHand",
            "report.hands[1] >= kMinimumOpenAcceptedPerHand",
            "report.root_azimuth_sectors >= kMinimumOpenAzimuthSectors",
        ):
            self.assertIn(required, source)

    def test_makefile_has_standalone_targets(self):
        makefile = self.source(MAKEFILE)
        self.assertIn("g1_reach_coverage_viewer:", makefile)
        self.assertIn("g1_reach_coverage_probe:", makefile)
        self.assertIn("reach_coverage.cpp", makefile)
        self.assertIn("reach_search.cpp", makefile)
        self.assertIn("reach_database.cpp", makefile)
        self.assertIn("g1_mesh_renderer.h", makefile)
        self.assertIn("resources/g1_mesh/g1_raylib.glb", makefile)
        probe_target = makefile.split("g1_reach_coverage_probe:", 1)[1].split(
            "\n\n", 1
        )[0]
        self.assertIn("-pthread", probe_target)


if __name__ == "__main__":
    unittest.main()
