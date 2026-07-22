import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "hand_trajectory_viewer.cpp"
TRAJECTORY_SOURCE_PATH = ROOT / "interaction_hand_trajectories.cpp"
AUDIT_SOURCE_PATH = ROOT / "interaction_reuse_audit.cpp"
AUDIT_PROBE_SOURCE_PATH = ROOT / "interaction_reuse_audit_probe.cpp"
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
        self.assertIn("environment_rejected", source)
        self.assertIn("valid.empty()", source)
        self.assertIn("orientation_mode", source)
        self.assertNotIn("constrain_grasp_orientation", source)
        self.assertNotIn("previous_clip", source)
        self.assertNotIn("preserved", source)
        self.assertIn("selected_index = 0U", enter_search)
        self.assertIn("[: previous  / or ]: next  Enter: rerun", source)
        self.assertIn("SEARCH STALE - press Enter", source)

        trajectory_source = TRAJECTORY_SOURCE_PATH.read_text(encoding="utf-8")
        self.assertIn("hand_trajectory_scene_alignment(", trajectory_source)
        self.assertLess(
            trajectory_source.index("if (position_error >"),
            trajectory_source.index(
                "for (int32_t frame = trajectory.start_frame;"
            ),
        )

    def test_builds_recorded_table_and_renders_rejections(self):
        source = self.source()
        self.assertIn("EnvironmentGeometry", source)
        self.assertIn("make_coverage_environment(", source)
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
        self.assertIn("selected.source.start_frame", source)
        self.assertIn('case 0U: return "APPROACH"', source)
        self.assertIn("kBackgroundPathStride", source)
        self.assertIn("draw_path(selected, LIME, 1U, true)", source)
        self.assertIn(
            "std::vector<interaction::Pose>{}.swap(shaped.poses)", source
        )
        self.assertNotIn("shaped.poses.clear()", source)
        self.assertIn("shape_selected_animation(", source)
        self.assertIn("selected_animation.poses[sample]", source)
        self.assertNotIn("selected.shaped.poses[sample]", source)

    def test_loads_compact_database_and_displays_support(self):
        source = self.source()
        self.assertIn('#include "interaction_trajectory_database.h"', source)
        self.assertIn("load_trajectory_database(", source)
        self.assertNotIn("interaction::load_database(", source)
        self.assertIn("SupportKind::Table", source)
        self.assertIn("SupportKind::Ground", source)
        self.assertIn('"GROUND"', source)
        self.assertIn('"TABLE"', source)
        self.assertIn("support_name(selected.source.support)", source)

    def test_ground_support_uses_no_table_geometry(self):
        source = self.source()
        self.assertIn(
            "EnvironmentGeometry support_geometry(", source
        )
        self.assertIn(
            "if (support == interaction::SupportKind::Ground) {\n"
            "        return {};\n"
            "    }",
            source,
        )
        self.assertNotIn("no_support_collision_geometry(", source)
        self.assertIn("environment.boxes", source)

        helper = source.split(
            "EnvironmentGeometry support_geometry(", 1
        )[1].split("CanonicalGrasp canonical_grasp(", 1)[0]
        ground_branch = helper.split(
            "if (support == interaction::SupportKind::Ground)", 1
        )[1].split("interaction::make_coverage_environment(", 1)[0]
        self.assertIn("return {}", ground_branch)
        self.assertNotIn("make_coverage_environment", ground_branch)

    def test_uses_exact_first_axis_fallback_and_all_approach_sides(self):
        source = self.source()
        self.assertIn("kTargetValidTrajectories = 12U", source)
        self.assertIn("GraspOrientationMode::ExactPose", source)
        self.assertIn("GraspOrientationMode::ApproachAxis", source)
        self.assertNotIn("kSceneFront", source)
        self.assertNotIn("starts_on_allowed_side(", source)
        self.assertNotIn("wrong_side", source)
        self.assertIn("exact_compatible", source)
        self.assertIn("fallback_compatible", source)
        self.assertIn('"EXACT"', source)
        self.assertIn('"AXIS-FALLBACK"', source)

    def test_ranks_refined_fallbacks_and_adds_viewer_clearance(self):
        source = self.source()
        self.assertIn("achieved_orientation_error_radians", source)
        self.assertIn("orientation_error_degrees", source)
        self.assertIn("orient %.1f deg", source)
        self.assertIn("std::stable_sort", source)
        self.assertIn("viewer_collision_config()", source)
        for radius in ("joint_radius_m", "limb_radius_m", "torso_radius_m"):
            self.assertIn(f"config.{radius} += 0.02F", source)
        self.assertIn("const TrajectoryCollisionConfig& collision_config", source)
        self.assertIn("environment, collision_config", source)

        comparator = source.split("std::stable_sort", 1)[1].split(
            "return result;", 1
        )[0]
        self.assertIn("TrajectoryMatchTier::AxisFallback", comparator)
        self.assertIn("achieved_orientation_error_radians", comparator)
        self.assertIn("source.cost", comparator)
        self.assertIn("source.clip", comparator)

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
        self.assertIn("interaction_trajectory_database.h", target)
        self.assertIn("MIXED_INTERACTION_PACK", MAKEFILE)
        self.assertIn("mixed-interaction-pack", MAKEFILE)

    def test_exposes_bounded_exhaustive_reuse_audit(self):
        source = self.source()
        for contract in (
            "KEY_A",
            "audit_reusable_hand_trajectories(",
            "ReuseAuditStatus::Incomplete",
            "deadline_milliseconds = 30000U",
            "worker_count = 4U",
            "Contact accepted",
            "fully shaped",
            "object rejected",
            "furniture rejected",
            "reusable",
            "AUDIT INCOMPLETE",
            "AUDIT STALE",
        ):
            self.assertIn(contract, source)

        grasp_change = source.split("if (grasp_changed)", 1)[1].split(
            "if (IsKeyPressed(KEY_ENTER))", 1
        )[0]
        self.assertIn("audit_stale = true", grasp_change)
        enter_search = source.split("if (IsKeyPressed(KEY_ENTER))", 1)[1].split(
            "if (IsKeyPressed(KEY_A))", 1
        )[0]
        self.assertNotIn("audit_reusable_hand_trajectories(", enter_search)

        self.assertIn("interaction_reuse_audit.cpp", MAKEFILE)
        self.assertIn("interaction_reuse_audit.h", MAKEFILE)
        self.assertIn("interaction_reuse_audit_probe:", MAKEFILE)

    def test_audit_deadline_and_hud_contract_are_truthful(self):
        viewer = self.source()
        audit_source = AUDIT_SOURCE_PATH.read_text(encoding="utf-8")
        probe_source = AUDIT_PROBE_SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn("AUDIT INCOMPLETE / STALE", viewer)
        self.assertIn("claim_mutex", audit_source)
        self.assertNotIn("std::atomic<size_t> next", audit_source)
        self.assertIn(
            "result.elapsed_milliseconds = elapsed_milliseconds(start)",
            audit_source,
        )
        self.assertGreater(
            audit_source.rindex(
                "result.elapsed_milliseconds = elapsed_milliseconds(start)"
            ),
            audit_source.index("std::stable_sort("),
        )
        self.assertIn("result.elapsed_milliseconds <= 30000U", probe_source)


if __name__ == "__main__":
    unittest.main()
