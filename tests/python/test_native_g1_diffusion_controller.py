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

    def test_one_final_g1_pose_fans_out_to_every_consumer(self):
        self.assertIn("const interaction::Pose final_g1_pose", SOURCE)
        self.assertIn(
            "interaction::WorldPose final_g1_world_pose", SOURCE)
        self.assertIn("validate_final_g1_grasp(", SOURCE)

        final_block = SOURCE.split(
            "const interaction::Pose final_g1_pose", 1)[1]
        self.assertIn("write_native_g1_pose(\n            final_g1_pose", final_block)

        mesh_block = SOURCE.split(
            "if (!::g1_mesh_renderer_update", 1)[1].split("// Render", 1)[0]
        self.assertIn("final_g1_world_pose.positions", mesh_block)
        self.assertIn("final_g1_world_pose.rotations", mesh_block)
        self.assertNotIn("state.global_bone_positions", mesh_block)

        skeleton_block = SOURCE.split(
            "// G1: no skinned mesh", 1)[1].split(
                "// Draw matched features", 1)[0]
        self.assertIn("final_g1_world_pose.positions", skeleton_block)
        self.assertIn("g1_skeleton::kParents", skeleton_block)
        self.assertNotIn("state.global_bone_positions", skeleton_block)

    def test_native_carry_places_and_refreshes_for_repickup(self):
        self.assertIn("controller_place_staging_stick(", SOURCE)
        self.assertIn("interaction_runtime.preview_place(", SOURCE)
        self.assertIn("interaction::PlaceRequest native_place_request", SOURCE)
        self.assertIn("native_place_request.selection_id =", SOURCE)
        self.assertIn(
            "interaction_input.place_request = pending_native_place_request",
            SOURCE,
        )
        self.assertIn("interaction_input.reset_pressed =", SOURCE)
        self.assertNotIn("native_g1_pose_handoff.reset();", SOURCE)
        self.assertIn("rendered_destination_surface", SOURCE)
        self.assertIn("native_place_preview->staging_root_world.position", SOURCE)

        carry_block = SOURCE.split(
            "interaction_runtime.state() ==\n"
            "                interaction::RuntimeState::Carry", 1)[1]
        self.assertIn("++interaction_next_request_id", carry_block)
        self.assertIn("interaction::arrival_facing_stick(", carry_block)

        refresh_block = SOURCE.split(
            "interaction_output = interaction_runtime.update(interaction_input)",
            1,
        )[1]
        self.assertIn(
            "interaction_registry.find_by_id(\n"
            "                interaction_scene_target_handle.id)",
            refresh_block,
        )

    def test_reset_releases_without_bypassing_handoff(self):
        self.assertIn("const bool scene_reset_requested = pending_reset", SOURCE)
        self.assertIn("bool interaction_reset_pending = false", SOURCE)
        self.assertIn("interaction_reset_pending = true", SOURCE)
        self.assertIn(
            "interaction_output.diagnostics.state ==\n"
            "                interaction::RuntimeState::Locomotion",
            SOURCE,
        )
        self.assertIn(
            "smart_pickup_cancel_pressed || smart_pickup_reset_pressed",
            SOURCE,
        )
        self.assertNotIn("native_g1_pose_handoff.reset();", SOURCE)

    def test_final_pose_uses_dedicated_channels(self):
        self.assertIn("array1d<vec3> final_g1_local_positions", SOURCE)
        write_block = SOURCE.split(
            "interaction::write_native_g1_pose(\n            final_g1_pose",
            1,
        )[1].split("interaction::WorldPose final_g1_world_pose", 1)[0]
        self.assertIn("final_g1_local_velocities", write_block)
        self.assertIn("final_g1_local_angular_velocities", write_block)
        self.assertNotIn("state.bone_velocities", write_block)
        self.assertNotIn("state.bone_angular_velocities", write_block)


if __name__ == "__main__":
    unittest.main()
