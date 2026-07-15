#include "interaction_runtime.h"
#include "tests/cpp/interaction_runtime_fixture.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr float kDt = 1.0F / 25.0F;
constexpr int kMaximumUpdates = 600;

interaction::RuntimeFixture make_runtime_fixture() {
    return interaction::make_place_runtime_fixture();
}

bool exact(vec3 left, vec3 right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool exact(quat left, quat right) {
    return left.w == right.w && left.x == right.x &&
           left.y == right.y && left.z == right.z;
}

bool exact(
    const interaction::Transform& left,
    const interaction::Transform& right) {
    return exact(left.position, right.position) &&
           exact(left.rotation, right.rotation);
}

bool exact(
    const interaction::Pose& left,
    const interaction::Pose& right) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        if (!exact(left.positions[bone], right.positions[bone]) ||
            !exact(left.velocities[bone], right.velocities[bone]) ||
            !exact(left.rotations[bone], right.rotations[bone]) ||
            !exact(
                left.angular_velocities[bone],
                right.angular_velocities[bone])) {
            return false;
        }
    }
    return left.hand_dof == right.hand_dof &&
           left.hand_dof_velocities == right.hand_dof_velocities &&
           left.foot_contacts == right.foot_contacts;
}

bool exact(
    const interaction::GraspAffordance& left,
    const interaction::GraspAffordance& right) {
    return left.id == right.id && left.hand == right.hand &&
           exact(left.hand_in_object, right.hand_in_object) &&
           exact(
               left.approach_direction_object,
               right.approach_direction_object) &&
           left.clearance_radius == right.clearance_radius;
}

bool exact(
    const interaction::InteractionTarget& left,
    const interaction::InteractionTarget& right) {
    if (left.handle != right.handle ||
        !exact(left.object_world, right.object_world) ||
        left.object_profile_id != right.object_profile_id ||
        !exact(
            left.object_bounds.center_object,
            right.object_bounds.center_object) ||
        !exact(
            left.object_bounds.half_extents_object,
            right.object_bounds.half_extents_object) ||
        !exact(left.object_dimensions, right.object_dimensions) ||
        !exact(left.table_world, right.table_world) ||
        !exact(left.table_size, right.table_size) ||
        left.state != right.state ||
        left.owner_request != right.owner_request ||
        left.affordances.size() != right.affordances.size()) {
        return false;
    }
    for (size_t index = 0; index < left.affordances.size(); ++index) {
        if (!exact(left.affordances[index], right.affordances[index])) {
            return false;
        }
    }
    return true;
}

bool exact(
    const interaction::PlaceAffordance& left,
    const interaction::PlaceAffordance& right) {
    return left.id == right.id &&
           exact(left.object_in_surface, right.object_in_surface) &&
           exact(left.support_point_object, right.support_point_object) &&
           exact(
               left.approach_direction_surface,
               right.approach_direction_surface) &&
           left.clearance_radius == right.clearance_radius;
}

bool exact(
    const interaction::PlacementSurface& left,
    const interaction::PlacementSurface& right) {
    if (!(left.handle == right.handle) ||
        !exact(left.surface_world, right.surface_world) ||
        !exact(left.support_volume_world, right.support_volume_world) ||
        !exact(left.support_volume_size, right.support_volume_size) ||
        left.half_extent_x_m != right.half_extent_x_m ||
        left.half_extent_z_m != right.half_extent_z_m ||
        left.overhead_clearance_m != right.overhead_clearance_m ||
        left.affordances.size() != right.affordances.size()) {
        return false;
    }
    for (size_t index = 0; index < left.affordances.size(); ++index) {
        if (!exact(left.affordances[index], right.affordances[index])) {
            return false;
        }
    }
    return true;
}

bool exact(
    const interaction::IKConfig& left,
    const interaction::IKConfig& right) {
    return left.maximum_request_position_m ==
               right.maximum_request_position_m &&
           left.maximum_request_orientation_radians ==
               right.maximum_request_orientation_radians &&
           left.accepted_position_m == right.accepted_position_m &&
           left.accepted_orientation_radians ==
               right.accepted_orientation_radians &&
           left.damping == right.damping &&
           left.finite_difference_radians ==
               right.finite_difference_radians &&
           left.orientation_scale_m_per_radian ==
               right.orientation_scale_m_per_radian &&
           left.maximum_step_radians == right.maximum_step_radians &&
           left.maximum_iterations == right.maximum_iterations;
}

bool exact(
    const interaction::PlaceCandidate& left,
    const interaction::PlaceCandidate& right) {
    return left.mode == right.mode && left.source_id == right.source_id &&
           left.selection_id == right.selection_id &&
           left.timing.canonical_fps == right.timing.canonical_fps &&
           left.timing.playback_speed == right.timing.playback_speed &&
           left.timing.entry_blend_seconds ==
               right.timing.entry_blend_seconds &&
           left.timing.reversed_commit_seconds ==
               right.timing.reversed_commit_seconds &&
           left.timing.maximum_alignment_seconds ==
               right.timing.maximum_alignment_seconds &&
           left.match.maximum_entry_root_error_m ==
               right.match.maximum_entry_root_error_m &&
           left.match.maximum_entry_yaw_error_radians ==
               right.match.maximum_entry_yaw_error_radians &&
           exact(left.ik, right.ik) && left.clip == right.clip &&
           left.entry_frame == right.entry_frame &&
           left.commit_frame == right.commit_frame &&
           left.release_frame == right.release_frame &&
           left.stop_frame == right.stop_frame &&
           left.direction == right.direction &&
           exact(left.scene_from_source, right.scene_from_source) &&
           exact(left.staging_root_world, right.staging_root_world) &&
           exact(left.entry_root_offset, right.entry_root_offset) &&
           left.entry_yaw_offset == right.entry_yaw_offset &&
           left.total_cost == right.total_cost;
}

bool exact(
    const interaction::PlaceStagingPreview& left,
    const interaction::PlaceStagingPreview& right) {
    return left.accepted == right.accepted && left.ready == right.ready &&
           left.reason == right.reason &&
           exact(left.candidate, right.candidate) &&
           exact(left.ik, right.ik) &&
           left.ik_config_fingerprint == right.ik_config_fingerprint &&
           exact(left.staging_root_world, right.staging_root_world) &&
           left.root_error_m == right.root_error_m &&
           left.yaw_error_radians == right.yaw_error_radians;
}

bool exact(
    const interaction::PlacementFit& left,
    const interaction::PlacementFit& right) {
    return left.accepted == right.accepted && left.reason == right.reason &&
           left.support_gap_m == right.support_gap_m &&
           left.lowest_corner_m == right.lowest_corner_m &&
           left.highest_corner_m == right.highest_corner_m &&
           left.footprint_valid == right.footprint_valid &&
           left.overhead_valid == right.overhead_valid;
}

bool exact(
    const interaction::RuntimePlaceDiagnostics& left,
    const interaction::RuntimePlaceDiagnostics& right) {
    return left.surface == right.surface &&
           left.affordance_id == right.affordance_id &&
           left.mode == right.mode &&
           left.selection_id == right.selection_id &&
           left.preview_available == right.preview_available &&
           exact(left.preview, right.preview) &&
           left.candidate_certified == right.candidate_certified &&
           left.preflight_config_identity ==
               right.preflight_config_identity &&
           exact(left.effective_ik, right.effective_ik) &&
           left.ik_config_fingerprint == right.ik_config_fingerprint &&
           exact(left.requested_goal_world, right.requested_goal_world) &&
           exact(left.requested_fit, right.requested_fit) &&
           exact(left.actual_fit, right.actual_fit) &&
           left.clip == right.clip &&
           left.source_frame == right.source_frame &&
           left.source_frame_exact == right.source_frame_exact &&
           left.commit_frame == right.commit_frame &&
           left.phase == right.phase &&
           left.time_to_release_seconds == right.time_to_release_seconds &&
           left.committed == right.committed &&
           left.release_due == right.release_due &&
           left.released == right.released &&
           left.support_sweep_clear == right.support_sweep_clear &&
           left.support_position_error_m == right.support_position_error_m &&
           left.support_orientation_error_radians ==
               right.support_orientation_error_radians &&
           left.requested_root_correction_m ==
               right.requested_root_correction_m &&
           left.applied_root_correction_m ==
               right.applied_root_correction_m &&
           left.requested_yaw_correction_radians ==
               right.requested_yaw_correction_radians &&
           left.applied_yaw_correction_radians ==
               right.applied_yaw_correction_radians &&
           left.requested_hand_correction_m ==
               right.requested_hand_correction_m &&
           left.applied_hand_correction_m ==
               right.applied_hand_correction_m &&
           left.requested_hand_orientation_radians ==
               right.requested_hand_orientation_radians &&
           left.applied_hand_orientation_radians ==
               right.applied_hand_orientation_radians &&
           left.reason == right.reason;
}

bool exact(
    const interaction::RuntimeDiagnostics& left,
    const interaction::RuntimeDiagnostics& right) {
    return left.state == right.state &&
           left.result == right.result &&
           left.reason == right.reason &&
           left.target == right.target &&
           left.object_state == right.object_state &&
           left.affordance_id == right.affordance_id &&
           left.clip == right.clip &&
           left.frame == right.frame &&
           left.phase == right.phase &&
           left.hand == right.hand &&
           left.total_cost == right.total_cost &&
           left.group_costs == right.group_costs &&
           left.requested_root_correction_m ==
               right.requested_root_correction_m &&
           left.applied_root_correction_m ==
               right.applied_root_correction_m &&
           left.requested_yaw_correction_radians ==
               right.requested_yaw_correction_radians &&
           left.applied_yaw_correction_radians ==
               right.applied_yaw_correction_radians &&
           left.playback_speed == right.playback_speed &&
           left.hand_position_error_m == right.hand_position_error_m &&
           left.hand_orientation_error_radians ==
               right.hand_orientation_error_radians &&
           left.hand_constraint_weight == right.hand_constraint_weight &&
           left.attached == right.attached &&
           left.recorded_carry == right.recorded_carry &&
           left.inactive_arm_targets_locomotion ==
               right.inactive_arm_targets_locomotion &&
           left.inactive_arm_tracks_locomotion ==
               right.inactive_arm_tracks_locomotion &&
           left.pack_available == right.pack_available &&
           exact(left.place, right.place);
}

bool exact(
    const interaction::RuntimeOutput& left,
    const interaction::RuntimeOutput& right) {
    return left.owns_pose == right.owns_pose &&
           left.suppress_steering == right.suppress_steering &&
           exact(left.pose, right.pose) &&
           exact(left.object_world, right.object_world) &&
           exact(left.diagnostics, right.diagnostics);
}

interaction::RuntimeInput idle_input(
    const interaction::LocomotionSnapshot& locomotion) {
    interaction::RuntimeInput input{};
    input.dt = kDt;
    input.locomotion = locomotion;
    return input;
}

interaction::RuntimeInput interact_input(
    const interaction::LocomotionSnapshot& locomotion,
    std::optional<interaction::PickRequest> request) {
    interaction::RuntimeInput input = idle_input(locomotion);
    input.interact_pressed = true;
    input.pick_request = request;
    return input;
}

interaction::RuntimeInput place_interact_input(
    const interaction::LocomotionSnapshot& locomotion,
    std::optional<interaction::PlaceRequest> request) {
    interaction::RuntimeInput input = idle_input(locomotion);
    input.interact_pressed = true;
    input.place_request = request;
    return input;
}

interaction::RuntimeInput cancel_input(
    const interaction::LocomotionSnapshot& locomotion) {
    interaction::RuntimeInput input = idle_input(locomotion);
    input.cancel_pressed = true;
    return input;
}

interaction::RuntimeInput reset_input(
    const interaction::LocomotionSnapshot& locomotion) {
    interaction::RuntimeInput input = idle_input(locomotion);
    input.reset_pressed = true;
    return input;
}

interaction::RuntimeOutput advance(
    interaction::InteractionRuntime& runtime,
    const interaction::LocomotionSnapshot& locomotion) {
    return runtime.update(idle_input(locomotion));
}

void assert_valid_hand_constraint_weight(
    const interaction::RuntimeDiagnostics& diagnostics) {
    assert(std::isfinite(diagnostics.hand_constraint_weight));
    assert(diagnostics.hand_constraint_weight >= 0.0F);
    assert(diagnostics.hand_constraint_weight <= 1.0F);
}

interaction::RuntimeOutput advance_until(
    interaction::InteractionRuntime& runtime,
    const interaction::LocomotionSnapshot& locomotion,
    interaction::RuntimeState expected,
    interaction::RuntimeState permitted_before) {
    for (int update = 0; update < kMaximumUpdates; ++update) {
        const interaction::RuntimeOutput output = advance(runtime, locomotion);
        if (output.diagnostics.state == expected) return output;
        assert(output.diagnostics.state == permitted_before);
    }
    assert(false && "interaction runtime did not reach expected state");
    return {};
}

interaction::RuntimeOutput enter_align(
    interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture) {
    using namespace interaction;
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    return output;
}

interaction::RuntimeOutput enter_carry_before_first_update(
    interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture) {
    using namespace interaction;
    RuntimeOutput output = enter_align(runtime, fixture);
    assert(!output.diagnostics.inactive_arm_tracks_locomotion);
    for (int update = 0;
         update < kMaximumUpdates &&
         output.diagnostics.state != RuntimeState::Carry;
         ++update) {
        assert(!output.diagnostics.inactive_arm_tracks_locomotion);
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(!output.diagnostics.inactive_arm_tracks_locomotion);
    return output;
}

interaction::PlaceRequest place_request_for(
    const interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture,
    uint64_t request_id = 1001U) {
    const interaction::PlaceStagingPreview preview = runtime.preview_place(
        fixture.surface, fixture.place_affordance_id);
    assert(preview.accepted);
    interaction::PlaceRequest request{};
    request.held_target = fixture.request.target;
    request.surface = fixture.surface;
    request.affordance_id = fixture.place_affordance_id;
    request.request_id = request_id;
    request.selection_id = preview.candidate.selection_id;
    return request;
}

interaction::RuntimeOutput enter_place_align(
    interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture,
    interaction::PlaceRequest request) {
    using namespace interaction;
    RuntimeOutput output = runtime.update(place_interact_input(
        fixture.locomotion, request));
    assert(output.diagnostics.state == RuntimeState::PlacePreflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::PlaceAlign);
    assert(output.diagnostics.attached);
    return output;
}

interaction::RuntimeOutput enter_ready_place_align(
    interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture,
    uint64_t request_id = 1001U) {
    return enter_place_align(
        runtime,
        fixture,
        place_request_for(runtime, fixture, request_id));
}

void assert_held_by_original_owner(
    const interaction::RuntimeFixture& fixture) {
    const interaction::InteractionTarget* held =
        fixture.registry.find(fixture.request.target);
    assert(held != nullptr);
    assert(held->state == interaction::ObjectState::Held);
    assert(held->owner_request == fixture.request.request_id);
}

void assert_fresh_carry_next_sample(
    interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture,
    const interaction::RuntimeOutput& recovery,
    const interaction::RuntimeConfig& config = interaction::RuntimeConfig{}) {
    using namespace interaction;
    const InteractionTarget* target = fixture.registry.find(
        fixture.request.target);
    assert(target != nullptr);
    const GraspAffordance* affordance = fixture.registry.find_affordance(
        fixture.request.target, fixture.request.affordance_id);
    assert(affordance != nullptr);
    CarryController expected(
        fixture.database,
        fixture.features,
        classify_carry_ranges(fixture.database, config.carry),
        config.carry,
        config.ik);
    expected.start(
        recovery.pose,
        affordance->hand,
        *affordance,
        recovery.object_world);
    const Pose expected_pose = expected.update(fixture.locomotion, kDt);
    const Transform expected_object = expected.object_world();

    const RuntimeOutput next = advance(runtime, fixture.locomotion);
    assert(next.diagnostics.state == RuntimeState::Carry);
    assert(exact(next.pose, expected_pose));
    assert(exact(next.object_world, expected_object));
    assert(next.diagnostics.attached);
    assert_held_by_original_owner(fixture);
}

void assert_preflight_rejection_preserves_carry(
    interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture,
    const interaction::RuntimeInput& edge,
    interaction::Reason expected_reason) {
    using namespace interaction;
    InteractionRuntime paused_control = runtime;
    const RuntimeDiagnostics before = runtime.diagnostics();
    const InteractionTarget target_before =
        *fixture.registry.find(fixture.request.target);

    RuntimeOutput output = runtime.update(edge);
    assert(output.diagnostics.state == RuntimeState::PlacePreflight);
    const Pose frozen_pose = output.pose;
    const Transform frozen_object = output.object_world;
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == expected_reason);
    assert(exact(output.pose, frozen_pose));
    assert(exact(output.object_world, frozen_object));
    assert(exact(*fixture.registry.find(fixture.request.target), target_before));
    assert(before.state == RuntimeState::Carry);

    const RuntimeOutput trial_next = advance(runtime, fixture.locomotion);
    const RuntimeOutput control_next = advance(
        paused_control, fixture.locomotion);
    assert(exact(trial_next.pose, control_next.pose));
    assert(exact(trial_next.object_world, control_next.object_world));
    assert(trial_next.diagnostics.state == RuntimeState::Carry);
    assert_held_by_original_owner(fixture);
}

void assert_free(
    const interaction::TargetRegistry& registry,
    interaction::TargetHandle handle) {
    const interaction::InteractionTarget* target = registry.find(handle);
    assert(target != nullptr);
    assert(target->state == interaction::ObjectState::Free);
    assert(target->owner_request == 0U);
}

interaction::RuntimeDiagnostics run_rejected(
    interaction::RuntimeFixture fixture) {
    using namespace interaction;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(!output.diagnostics.attached);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, fixture.request.target);
    return output.diagnostics;
}

interaction::RuntimeDiagnostics run_post_commit_failure(
    interaction::RuntimeFixture fixture) {
    using namespace interaction;
    RuntimeConfig config{};
    config.ik.accepted_position_m =
        config.ik.maximum_request_position_m;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);

    bool saw_pickup_replay = false;
    bool saw_final_owned_frame = false;
    Reason latched_reason = Reason::None;
    int32_t previous_frame = output.diagnostics.frame;
    const int32_t expected_final = fixture.database.range_stops.at(1) - 1;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        output = advance(runtime, fixture.locomotion);
        if (output.diagnostics.frame >= 0 && previous_frame >= 0) {
            assert(output.diagnostics.frame >= previous_frame);
        }
        previous_frame = output.diagnostics.frame;
        if (output.diagnostics.state == RuntimeState::PickupReplay) {
            saw_pickup_replay = true;
        }
        if (output.diagnostics.result == ResultCode::Failed) {
            if (latched_reason == Reason::None) {
                latched_reason = output.diagnostics.reason;
            }
            assert(output.diagnostics.reason == latched_reason);
        }
        assert(!output.diagnostics.attached);
        if (output.diagnostics.frame == expected_final &&
            output.diagnostics.state != RuntimeState::Locomotion) {
            assert(output.owns_pose);
            saw_final_owned_frame = true;
        }
        if (output.diagnostics.state == RuntimeState::Locomotion) {
            assert(saw_pickup_replay);
            assert(saw_final_owned_frame);
            assert(output.diagnostics.result == ResultCode::Failed);
            assert(output.diagnostics.reason == latched_reason);
            assert(!output.owns_pose && !output.suppress_steering);
            assert_free(fixture.registry, fixture.request.target);
            return output.diagnostics;
        }
    }
    assert(false && "post-commit failure did not return to locomotion");
    return {};
}

interaction::RuntimeFixture post_attach_contact_loss_fixture() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    constexpr int32_t kFirstPostContactFrame =
        runtime_fixture_detail::kFramesPerClip +
        runtime_fixture_detail::kContactLocalFrame + 1;
    fixture.database.hand_contacts.at(
        static_cast<size_t>(kFirstPostContactFrame) * 2U + 1U) = 0U;
    return fixture;
}

interaction::RuntimeFixture post_success_contact_loss_fixture(
    int32_t frames_after_hold = 1) {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    const int32_t loss_frame =
        runtime_fixture_detail::kFramesPerClip +
        runtime_fixture_detail::kHoldLocalFrame + frames_after_hold;
    fixture.database.hand_contacts.at(
        static_cast<size_t>(loss_frame) * 2U + 1U) = 0U;
    return fixture;
}

interaction::RuntimeFixture post_ik_table_crossing_fixture() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionTarget* target = fixture.registry.find(fixture.request.target);
    assert(target != nullptr);
    target->table_size.z = 0.01F;
    GraspAffordance& affordance = target->affordances.at(0);
    affordance.clearance_radius = 0.005F;
    affordance.hand_in_object.position.y -= 0.06F;
    affordance.hand_in_object.position.z += 0.04F;
    return fixture;
}

interaction::RuntimeFixture post_ik_object_crossing_fixture() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionTarget* target = fixture.registry.find(fixture.request.target);
    assert(target != nullptr);
    target->table_world.position.y = 0.0F;
    target->table_size.y = 0.10F;
    target->affordances.at(0).hand_in_object.position.y -= 0.06F;
    target->affordances.at(0).hand_in_object.position.z += 0.04F;
    return fixture;
}

interaction::RuntimeFixture curved_precontact_clearance_fixture() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionTarget* target = fixture.registry.find(fixture.request.target);
    assert(target != nullptr);
    GraspAffordance& affordance = target->affordances.at(0);
    affordance.hand_in_object.position.x += 0.08F;
    affordance.clearance_radius = 0.001F;

    constexpr int32_t kIntermediateFrame =
        runtime_fixture_detail::kFramesPerClip + 18;
    vec3 authored = runtime_fixture_detail::read_bone_position(
        fixture.database, kIntermediateFrame, kRightHandBone);
    authored.z += 0.10F;
    runtime_fixture_detail::write_bone_position(
        fixture.database, kIntermediateFrame, kRightHandBone, authored);
    runtime_fixture_detail::recompute_velocities(fixture.database);

    Pose corrected = pose_at_frame(fixture.database, kIntermediateFrame);
    const WorldPose sampled_world = world_pose(corrected);
    const Transform sampled_hand{
        sampled_world.positions[kRightHandBone],
        sampled_world.rotations[kRightHandBone],
    };
    const WorldPose contact_world = world_pose(pose_at_frame(
        fixture.database,
        runtime_fixture_detail::kFramesPerClip +
            runtime_fixture_detail::kContactLocalFrame));
    const Transform source_contact{
        contact_world.positions[kRightHandBone],
        contact_world.rotations[kRightHandBone],
    };
    const Transform target_hand = compose(
        target->object_world, affordance.hand_in_object);
    constexpr float kReachAlpha = 8.0F / 15.0F;
    Transform requested = sampled_hand;
    requested.position = sampled_hand.position + kReachAlpha *
        (target_hand.position - source_contact.position);
    requested.rotation = quat_nlerp_shortest(
        sampled_hand.rotation,
        quat_normalize(quat_mul(
            quat_mul(
                target_hand.rotation,
                quat_inv(source_contact.rotation)),
            sampled_hand.rotation)),
        kReachAlpha);
    IKConfig ik_config{};
    ik_config.accepted_position_m = 0.01F;
    ik_config.maximum_iterations = 64;
    const IKResult ik = solve_hand_ik(
        corrected, Hand::Right, requested, ik_config);
    assert(ik.accepted);
    const vec3 corrected_hand = world_pose(corrected).positions[
        kRightHandBone];
    target->table_world = {corrected_hand, quat()};
    target->table_size = vec3(0.004F, 0.004F, 0.004F);

    const RawQuery raw = build_raw_query(query_input_for(
        fixture, *target, affordance));
    fixture.features.offsets.assign(raw.begin(), raw.end());
    return fixture;
}

interaction::RuntimeFixture reverse_place_runtime_fixture() {
    using namespace interaction;
    using namespace interaction::runtime_fixture_detail;
    RuntimeFixture fixture = make_runtime_fixture();
    fixture.place_library.recorded.clear();
    for (int32_t local = kContactLocalFrame + 1;
         local <= kHoldLocalFrame;
         ++local) {
        const int32_t frame = kFramesPerClip + local;
        const float alpha = static_cast<float>(
            local - (kContactLocalFrame + 1)) /
            static_cast<float>(
                kHoldLocalFrame - (kContactLocalFrame + 1));
        vec3 object = read_vec3(
            fixture.database.object_positions,
            static_cast<size_t>(frame));
        object.y = 0.80F + 0.15F * alpha;
        write_vec3(
            fixture.database.object_positions,
            static_cast<size_t>(frame),
            object);
        const vec3 root = read_bone_position(
            fixture.database, frame, g1_skeleton::Simulation);
        write_bone_position(
            fixture.database, frame, kRightHandBone, object - root);
        write_bone_position(
            fixture.database, frame, kLeftHandBone, object - root);
    }
    recompute_velocities(fixture.database);
    return fixture;
}

void assert_invalid_runtime_config(interaction::RuntimeConfig config) {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    bool threw = false;
    try {
        InteractionRuntime runtime(
            fixture.database, fixture.features, fixture.registry, config);
        (void)runtime;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    assert_free(fixture.registry, fixture.request.target);
}

void assert_both_runtime_constructors_reject_without_mutation(
    interaction::RuntimeConfig config) {
    using namespace interaction;
    for (bool placement_enabled : {false, true}) {
        RuntimeFixture fixture = make_runtime_fixture();
        const InteractionTarget target_before =
            *fixture.registry.find(fixture.request.target);
        const PlacementSurface surface_before =
            *fixture.surface_registry.find(fixture.surface);
        bool threw = false;
        try {
            if (placement_enabled) {
                InteractionRuntime runtime(
                    fixture.database,
                    fixture.features,
                    fixture.registry,
                    fixture.surface_registry,
                    fixture.place_library,
                    config);
                (void)runtime;
            } else {
                InteractionRuntime runtime(
                    fixture.database,
                    fixture.features,
                    fixture.registry,
                    config);
                (void)runtime;
            }
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
        assert(exact(
            *fixture.registry.find(fixture.request.target), target_before));
        assert(exact(
            *fixture.surface_registry.find(fixture.surface), surface_before));
    }
}

void assert_both_runtime_constructors_accept(
    interaction::RuntimeConfig config) {
    using namespace interaction;
    for (bool placement_enabled : {false, true}) {
        RuntimeFixture fixture = make_runtime_fixture();
        if (placement_enabled) {
            InteractionRuntime runtime(
                fixture.database,
                fixture.features,
                fixture.registry,
                fixture.surface_registry,
                fixture.place_library,
                config);
            (void)runtime;
        } else {
            InteractionRuntime runtime(
                fixture.database,
                fixture.features,
                fixture.registry,
                config);
            (void)runtime;
        }
        assert_free(fixture.registry, fixture.request.target);
    }
}

void test_frozen_public_contract_and_defaults() {
    using namespace interaction;

    static_assert(std::is_same_v<
        std::underlying_type_t<RuntimeState>, uint8_t>);
    static_assert(static_cast<uint8_t>(RuntimeState::Disabled) == 0U);
    static_assert(static_cast<uint8_t>(RuntimeState::Locomotion) == 1U);
    static_assert(static_cast<uint8_t>(RuntimeState::Preflight) == 2U);
    static_assert(static_cast<uint8_t>(RuntimeState::Align) == 3U);
    static_assert(static_cast<uint8_t>(RuntimeState::PickupReplay) == 4U);
    static_assert(static_cast<uint8_t>(RuntimeState::Hold) == 5U);
    static_assert(static_cast<uint8_t>(RuntimeState::Carry) == 6U);
    static_assert(static_cast<uint8_t>(RuntimeState::PlacePreflight) == 7U);
    static_assert(static_cast<uint8_t>(RuntimeState::PlaceAlign) == 8U);
    static_assert(static_cast<uint8_t>(RuntimeState::PlaceReplay) == 9U);
    static_assert(static_cast<uint8_t>(RuntimeState::PlaceRelease) == 10U);

    static_assert(std::is_same_v<decltype(RuntimeConfig{}.matcher), MatchConfig>);
    static_assert(std::is_same_v<
        decltype(RuntimeConfig{}.playback), PlaybackConfig>);
    static_assert(std::is_same_v<decltype(RuntimeConfig{}.ik), IKConfig>);
    static_assert(std::is_same_v<
        decltype(RuntimeConfig{}.attachment), AttachmentConfig>);
    static_assert(std::is_same_v<decltype(RuntimeConfig{}.carry), CarryConfig>);
    static_assert(std::is_same_v<
        decltype(RuntimeConfig{}.place), PlaceControllerConfig>);
    static_assert(std::is_same_v<decltype(RuntimeInput{}.dt), float>);
    static_assert(std::is_same_v<
        decltype(RuntimeInput{}.locomotion), LocomotionSnapshot>);
    static_assert(std::is_same_v<
        decltype(RuntimeInput{}.pick_request), std::optional<PickRequest>>);
    static_assert(std::is_same_v<
        decltype(RuntimeInput{}.place_request), std::optional<PlaceRequest>>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.state), RuntimeState>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.result), ResultCode>);
    static_assert(std::is_same_v<decltype(RuntimeDiagnostics{}.reason), Reason>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.target), TargetHandle>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.object_state), ObjectState>);
    static_assert(std::is_same_v<decltype(RuntimeOutput{}.pose), Pose>);
    static_assert(std::is_same_v<
        decltype(RuntimeOutput{}.object_world), Transform>);
    static_assert(std::is_same_v<
        decltype(RuntimeOutput{}.diagnostics), RuntimeDiagnostics>);
    static_assert(std::is_constructible_v<
        InteractionRuntime,
        const Database&,
        const Features&,
        TargetRegistry&,
        RuntimeConfig>);
    static_assert(std::is_constructible_v<
        InteractionRuntime,
        const Database&,
        const Features&,
        TargetRegistry&,
        PlacementSurfaceRegistry&,
        const PlaceMotionLibrary&,
        RuntimeConfig>);
    static_assert(std::is_same_v<
        decltype(std::declval<const InteractionRuntime&>().preview_place(
            SurfaceHandle{}, uint32_t{})),
        PlaceStagingPreview>);
    static_assert(std::is_same_v<
        decltype(&InteractionRuntime::disabled),
        InteractionRuntime (*)(Reason)>);
    static_assert(std::is_same_v<
        decltype(&InteractionRuntime::state),
        RuntimeState (InteractionRuntime::*)() const>);
    static_assert(std::is_same_v<
        decltype(&InteractionRuntime::diagnostics),
        const RuntimeDiagnostics& (InteractionRuntime::*)() const>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.hand_constraint_weight), float>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.inactive_arm_targets_locomotion),
        bool>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.inactive_arm_tracks_locomotion), bool>);
    static_assert(std::is_same_v<
        decltype(&InteractionRuntime::update),
        RuntimeOutput (InteractionRuntime::*)(const RuntimeInput&)>);

    const PlaybackConfig playback{};
    assert(playback.canonical_fps == 25.0F);
    assert(playback.speed == 1.0F);
    assert(playback.minimum_speed == 0.85F);
    assert(playback.maximum_speed == 1.15F);
    assert(playback.entry_blend_seconds == 0.25F);
    assert(playback.maximum_alignment_seconds == 1.00F);
    assert(playback.commit_horizon_seconds == 0.50F);

    const RuntimeInput input{};
    assert(input.dt == 0.0F);
    assert(!input.interact_pressed && !input.pick_request.has_value());
    assert(!input.cancel_pressed && !input.reset_pressed);

    const RuntimeDiagnostics diagnostics{};
    assert(diagnostics.state == RuntimeState::Disabled);
    assert(diagnostics.result == ResultCode::None);
    assert(diagnostics.reason == Reason::None);
    assert(diagnostics.object_state == ObjectState::Free);
    assert(diagnostics.clip == -1 && diagnostics.frame == -1);
    assert(diagnostics.hand_constraint_weight == 0.0F);
    assert(!diagnostics.attached && !diagnostics.recorded_carry);
    assert(!diagnostics.inactive_arm_targets_locomotion);
    assert(!diagnostics.inactive_arm_tracks_locomotion);
    assert(!diagnostics.pack_available);

    const RuntimeOutput output{};
    assert(!output.owns_pose && !output.suppress_steering);
}

void test_place_preview_and_collapsed_success_lifecycle() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        fixture.surface_registry,
        fixture.place_library,
        RuntimeConfig{});

    const RuntimeDiagnostics before_preview = runtime.diagnostics();
    const InteractionTarget before_target =
        *fixture.registry.find(fixture.request.target);
    const PlacementSurface before_surface =
        *fixture.surface_registry.find(fixture.surface);
    const PlaceStagingPreview before_carry = runtime.preview_place(
        fixture.surface, fixture.place_affordance_id);
    assert(!before_carry.accepted);
    assert(before_carry.reason == Reason::OutOfRange);
    assert(exact(runtime.diagnostics(), before_preview));
    assert(exact(
        *fixture.registry.find(fixture.request.target), before_target));
    assert(exact(
        *fixture.surface_registry.find(fixture.surface), before_surface));

    RuntimeOutput output = enter_carry_before_first_update(runtime, fixture);
    const RuntimeOutput frozen = output;
    const RuntimeDiagnostics carry_diagnostics = runtime.diagnostics();
    const InteractionTarget carry_target_before =
        *fixture.registry.find(fixture.request.target);
    const PlacementSurface carry_surface_before =
        *fixture.surface_registry.find(fixture.surface);
    const PlaceStagingPreview first = runtime.preview_place(
        fixture.surface, fixture.place_affordance_id);
    const PlaceStagingPreview second = runtime.preview_place(
        fixture.surface, fixture.place_affordance_id);
    assert(first.accepted);
    assert(first.ready);
    assert(first.candidate.selection_id != 0U);
    assert(first.ik_config_fingerprint != 0U);
    assert(exact(first, second));
    assert(exact(runtime.diagnostics(), carry_diagnostics));
    assert(exact(
        *fixture.registry.find(fixture.request.target), carry_target_before));
    assert(exact(
        *fixture.surface_registry.find(fixture.surface), carry_surface_before));

    PlaceRequest request{};
    request.held_target = fixture.request.target;
    request.surface = fixture.surface;
    request.affordance_id = fixture.place_affordance_id;
    request.request_id = 1001U;
    request.selection_id = first.candidate.selection_id;
    output = runtime.update(place_interact_input(fixture.locomotion, request));
    assert(output.diagnostics.state == RuntimeState::PlacePreflight);
    assert(output.owns_pose && output.suppress_steering);
    assert(output.diagnostics.attached);
    assert(exact(output.pose, frozen.pose));
    assert(exact(output.object_world, frozen.object_world));
    assert(exact(output.diagnostics.place.preview, first));

    std::vector<RuntimeState> collapsed{RuntimeState::Carry};
    RuntimeState previous = RuntimeState::Carry;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        if (output.diagnostics.state != previous) {
            collapsed.push_back(output.diagnostics.state);
            previous = output.diagnostics.state;
        }
        if (output.diagnostics.state == RuntimeState::PlacePreflight ||
            output.diagnostics.state == RuntimeState::PlaceAlign ||
            output.diagnostics.state == RuntimeState::PlaceReplay ||
            output.diagnostics.state == RuntimeState::PlaceRelease) {
            assert(output.owns_pose);
            assert(output.suppress_steering);
        }
        if (output.diagnostics.state == RuntimeState::PlacePreflight) {
            assert(output.diagnostics.place.selection_id ==
                   first.candidate.selection_id);
        }
        if (output.diagnostics.state == RuntimeState::PlaceReplay) {
            assert(output.diagnostics.attached);
            assert(output.diagnostics.object_state == ObjectState::Held);
        }
        if (output.diagnostics.state == RuntimeState::PlaceRelease) {
            assert(!output.diagnostics.attached);
            assert(output.diagnostics.object_state == ObjectState::Free);
            assert(output.diagnostics.target.id == fixture.request.target.id);
            assert(output.diagnostics.target.generation ==
                   fixture.request.target.generation + 1U);
            const InteractionTarget* placed = fixture.registry.find(
                output.diagnostics.target);
            assert(placed != nullptr);
            assert(exact(placed->object_world, output.object_world));
            assert(exact(
                placed->table_world,
                before_surface.support_volume_world));
            assert(exact(
                placed->table_size,
                before_surface.support_volume_size));
        }
        if (output.diagnostics.state == RuntimeState::Locomotion) break;
        output = advance(runtime, fixture.locomotion);
    }
    if (collapsed.empty() || collapsed.back() != RuntimeState::Locomotion) {
        collapsed.push_back(output.diagnostics.state);
    }
    const std::vector<RuntimeState> expected = {
        RuntimeState::Carry,
        RuntimeState::PlacePreflight,
        RuntimeState::PlaceAlign,
        RuntimeState::PlaceReplay,
        RuntimeState::PlaceRelease,
        RuntimeState::Locomotion,
    };
    assert(collapsed == expected);
    assert(output.diagnostics.result == ResultCode::Succeeded);
    assert(output.diagnostics.reason == Reason::None);
    assert(!output.owns_pose && !output.suppress_steering);
    const InteractionTarget released_before_preview =
        *fixture.registry.find(output.diagnostics.target);
    const PlacementSurface surface_before_preview =
        *fixture.surface_registry.find(fixture.surface);
    const RuntimeDiagnostics diagnostics_before_preview =
        runtime.diagnostics();
    const PlaceStagingPreview after_release = runtime.preview_place(
        fixture.surface, fixture.place_affordance_id);
    assert(!after_release.accepted);
    assert(after_release.reason == Reason::OutOfRange);
    assert(exact(runtime.diagnostics(), diagnostics_before_preview));
    assert(exact(
        *fixture.registry.find(output.diagnostics.target),
        released_before_preview));
    assert(exact(
        *fixture.surface_registry.find(fixture.surface),
        surface_before_preview));

    const TargetHandle placed_handle = output.diagnostics.target;
    const std::optional<TargetHandle> resolved =
        fixture.registry.resolve_single_target(
            fixture.locomotion.pose.positions[g1_skeleton::Simulation],
            5.0F);
    assert(resolved.has_value());
    assert(*resolved == placed_handle);
    const InteractionTarget* placed = fixture.registry.find(placed_handle);
    const GraspAffordance* placed_affordance =
        fixture.registry.find_affordance(
            placed_handle, fixture.request.affordance_id);
    assert(placed != nullptr && placed_affordance != nullptr);
    const QueryInput destination_query = query_input_for(
        fixture, *placed, *placed_affordance);
    assert(exact(
        destination_query.table_world,
        before_surface.support_volume_world));
    assert(exact(
        destination_query.table_size,
        before_surface.support_volume_size));
    assert(!exact(destination_query.table_world, before_target.table_world));

    const PickRequest repick{
        placed_handle,
        fixture.request.affordance_id,
        1002U,
    };
    output = runtime.update(interact_input(fixture.locomotion, repick));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    assert(output.diagnostics.target == placed_handle);
}

void test_runtime_validates_complete_place_and_ik_configuration() {
    using namespace interaction;
    const float infinity = std::numeric_limits<float>::infinity();
    const float orientation_cap = 0.436332313F;

    const auto rejects_place = [](const auto& mutate) {
        RuntimeConfig config{};
        mutate(config.place);
        assert_both_runtime_constructors_reject_without_mutation(config);
    };
    rejects_place([](PlaceControllerConfig& value) {
        value.timing.canonical_fps = 0.0F;
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.timing.playback_speed = 0.0F;
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.timing.entry_blend_seconds = 0.0F;
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.timing.reversed_commit_seconds = 0.0F;
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.timing.maximum_alignment_seconds = 0.0F;
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.timing.maximum_alignment_seconds = std::nextafter(
            value.timing.entry_blend_seconds, 0.0F);
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.timing.maximum_alignment_seconds = std::nextafter(
            value.timing.reversed_commit_seconds, 0.0F);
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.match.maximum_entry_root_error_m = 0.0F;
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.match.maximum_entry_root_error_m = std::nextafter(
            0.25F, std::numeric_limits<float>::infinity());
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.match.maximum_entry_yaw_error_radians = 0.0F;
    });
    rejects_place([orientation_cap](PlaceControllerConfig& value) {
        value.match.maximum_entry_yaw_error_radians = std::nextafter(
            orientation_cap, std::numeric_limits<float>::infinity());
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.release_position_m = 0.0F;
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.release_position_m = std::nextafter(
            0.02F, std::numeric_limits<float>::infinity());
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.release_orientation_radians = 0.0F;
    });
    rejects_place([orientation_cap](PlaceControllerConfig& value) {
        value.release_orientation_radians = std::nextafter(
            orientation_cap, std::numeric_limits<float>::infinity());
    });

    using IKMember = float IKConfig::*;
    const std::array<IKMember, 8> ik_fields = {
        &IKConfig::maximum_request_position_m,
        &IKConfig::maximum_request_orientation_radians,
        &IKConfig::accepted_position_m,
        &IKConfig::accepted_orientation_radians,
        &IKConfig::damping,
        &IKConfig::finite_difference_radians,
        &IKConfig::orientation_scale_m_per_radian,
        &IKConfig::maximum_step_radians,
    };
    for (size_t index = 0; index < ik_fields.size(); ++index) {
        RuntimeConfig config{};
        config.ik.*ik_fields[index] = index < 4U ? -0.01F : 0.0F;
        assert_both_runtime_constructors_reject_without_mutation(config);

        config = RuntimeConfig{};
        config.ik.*ik_fields[index] = infinity;
        assert_both_runtime_constructors_reject_without_mutation(config);
    }
    {
        RuntimeConfig config{};
        config.ik.maximum_iterations = -1;
        assert_both_runtime_constructors_reject_without_mutation(config);
    }

    const float position_cap = 0.12F;
    for (IKMember capped : {
             &IKConfig::maximum_request_position_m,
             &IKConfig::maximum_request_orientation_radians}) {
        RuntimeConfig config{};
        config.ik.*capped = std::nextafter(
            capped == &IKConfig::maximum_request_position_m
                ? position_cap
                : orientation_cap,
            infinity);
        assert_both_runtime_constructors_reject_without_mutation(config);
    }

    RuntimeConfig cap_config{};
    cap_config.ik.maximum_request_position_m = position_cap;
    cap_config.ik.maximum_request_orientation_radians = orientation_cap;
    assert_both_runtime_constructors_accept(cap_config);
    RuntimeConfig tighter_config{};
    tighter_config.ik.maximum_request_position_m = 0.10F;
    tighter_config.ik.maximum_request_orientation_radians = 0.40F;
    assert_both_runtime_constructors_accept(tighter_config);

    for (const RuntimeConfig& config : {cap_config, tighter_config}) {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            fixture.surface_registry,
            fixture.place_library,
            config);
        enter_carry_before_first_update(runtime, fixture);
        const PlaceStagingPreview preview = runtime.preview_place(
            fixture.surface, fixture.place_affordance_id);
        assert(preview.accepted);
        assert(exact(preview.ik, config.ik));
        assert(preview.ik.maximum_request_position_m ==
               config.ik.maximum_request_position_m);
        assert(preview.ik.maximum_request_orientation_radians ==
               config.ik.maximum_request_orientation_radians);
    }
}

void test_runtime_forwards_nondefault_place_config_exactly() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    RuntimeConfig config{};
    config.place.timing.playback_speed = 0.85F;
    config.place.timing.entry_blend_seconds = 0.20F;
    config.place.timing.reversed_commit_seconds = 0.40F;
    config.place.timing.maximum_alignment_seconds = 0.90F;
    config.place.match.maximum_entry_root_error_m = 0.20F;
    config.place.match.maximum_entry_yaw_error_radians = 0.30F;
    config.place.release_position_m = 0.015F;
    config.place.release_orientation_radians = 0.10F;

    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        fixture.surface_registry,
        fixture.place_library,
        config);
    enter_carry_before_first_update(runtime, fixture);
    const PlaceStagingPreview preview = runtime.preview_place(
        fixture.surface, fixture.place_affordance_id);
    assert(preview.accepted && preview.ready);
    assert(preview.candidate.timing.canonical_fps ==
           config.place.timing.canonical_fps);
    assert(preview.candidate.timing.playback_speed ==
           config.place.timing.playback_speed);
    assert(preview.candidate.timing.entry_blend_seconds ==
           config.place.timing.entry_blend_seconds);
    assert(preview.candidate.timing.reversed_commit_seconds ==
           config.place.timing.reversed_commit_seconds);
    assert(preview.candidate.timing.maximum_alignment_seconds ==
           config.place.timing.maximum_alignment_seconds);
    assert(preview.candidate.match.maximum_entry_root_error_m ==
           config.place.match.maximum_entry_root_error_m);
    assert(preview.candidate.match.maximum_entry_yaw_error_radians ==
           config.place.match.maximum_entry_yaw_error_radians);

    RuntimeOutput output = enter_place_align(
        runtime, fixture, place_request_for(runtime, fixture));
    assert(output.diagnostics.place.preflight_config_identity);
    assert(exact(output.diagnostics.place.preview.candidate, preview.candidate));
    const int32_t commit = preview.candidate.commit_frame;
    double previous_source = output.diagnostics.place.source_frame_exact;
    int ticks_to_commit = 0;
    while (output.diagnostics.state == RuntimeState::PlaceAlign) {
        output = advance(runtime, fixture.locomotion);
        ++ticks_to_commit;
        assert(output.diagnostics.place.source_frame_exact >= previous_source);
        previous_source = output.diagnostics.place.source_frame_exact;
    }
    assert(output.diagnostics.state == RuntimeState::PlaceReplay);
    assert(output.diagnostics.place.source_frame_exact >= commit);
    assert(ticks_to_commit == static_cast<int>(std::ceil(
        static_cast<double>(commit - preview.candidate.entry_frame) /
        config.place.timing.playback_speed)));

    RecordedPlaceClip& clip = fixture.place_library.recorded.front();
    clip.poses.at(static_cast<size_t>(clip.release_frame))
        .positions[kRightHandBone].x += 0.016F;
    RuntimeOutput last_safe = output;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        output = advance(runtime, fixture.locomotion);
        if (output.diagnostics.state == RuntimeState::Carry) break;
        last_safe = output;
    }
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.result == ResultCode::Failed);
    assert(output.diagnostics.reason == Reason::ReleasePosition);
    assert(output.diagnostics.hand_position_error_m >
           config.place.release_position_m);
    assert(output.diagnostics.hand_position_error_m < 0.02F);
    assert(output.diagnostics.attached);
    assert(exact(output.pose, last_safe.pose));
    assert(exact(output.object_world, last_safe.object_world));
    assert_fresh_carry_next_sample(runtime, fixture, output, config);
}

void test_runtime_ik_identity_binds_every_scalar_and_iteration() {
    using namespace interaction;
    const auto preview_for = [](RuntimeConfig config) {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            fixture.surface_registry,
            fixture.place_library,
            config);
        enter_carry_before_first_update(runtime, fixture);
        const PlaceStagingPreview preview = runtime.preview_place(
            fixture.surface, fixture.place_affordance_id);
        assert(preview.accepted && preview.ready);
        assert(exact(preview.ik, config.ik));
        assert(preview.ik_config_fingerprint != 0U);
        return preview;
    };

    const RuntimeConfig baseline_config{};
    const PlaceStagingPreview baseline = preview_for(baseline_config);
    const PlaceStagingPreview duplicate = preview_for(baseline_config);
    assert(baseline.ik_config_fingerprint ==
           duplicate.ik_config_fingerprint);
    assert(baseline.candidate.selection_id ==
           duplicate.candidate.selection_id);

    using IKMember = float IKConfig::*;
    const std::array<IKMember, 8> fields = {
        &IKConfig::maximum_request_position_m,
        &IKConfig::maximum_request_orientation_radians,
        &IKConfig::accepted_position_m,
        &IKConfig::accepted_orientation_radians,
        &IKConfig::damping,
        &IKConfig::finite_difference_radians,
        &IKConfig::orientation_scale_m_per_radian,
        &IKConfig::maximum_step_radians,
    };
    for (size_t index = 0; index < fields.size(); ++index) {
        RuntimeConfig perturbed = baseline_config;
        float& scalar = perturbed.ik.*fields[index];
        scalar = std::nextafter(
            scalar,
            index < 2U
                ? 0.0F
                : std::numeric_limits<float>::infinity());
        const PlaceStagingPreview changed = preview_for(perturbed);
        assert(changed.ik_config_fingerprint !=
               baseline.ik_config_fingerprint);
        assert(changed.candidate.selection_id !=
               baseline.candidate.selection_id);
    }
    RuntimeConfig iterated = baseline_config;
    ++iterated.ik.maximum_iterations;
    const PlaceStagingPreview iteration_changed = preview_for(iterated);
    assert(iteration_changed.ik_config_fingerprint !=
           baseline.ik_config_fingerprint);
    assert(iteration_changed.candidate.selection_id !=
           baseline.candidate.selection_id);

    RuntimeFixture fixture_a = make_runtime_fixture();
    RuntimeFixture fixture_b = make_runtime_fixture();
    RuntimeConfig config_b{};
    config_b.ik.damping = std::nextafter(
        config_b.ik.damping, std::numeric_limits<float>::infinity());
    InteractionRuntime runtime_a(
        fixture_a.database,
        fixture_a.features,
        fixture_a.registry,
        fixture_a.surface_registry,
        fixture_a.place_library,
        RuntimeConfig{});
    InteractionRuntime runtime_b(
        fixture_b.database,
        fixture_b.features,
        fixture_b.registry,
        fixture_b.surface_registry,
        fixture_b.place_library,
        config_b);
    enter_carry_before_first_update(runtime_a, fixture_a);
    enter_carry_before_first_update(runtime_b, fixture_b);
    const PlaceStagingPreview preview_a = runtime_a.preview_place(
        fixture_a.surface, fixture_a.place_affordance_id);
    const PlaceStagingPreview preview_b = runtime_b.preview_place(
        fixture_b.surface, fixture_b.place_affordance_id);
    assert(preview_a.candidate.selection_id !=
           preview_b.candidate.selection_id);
    PlaceRequest stale = place_request_for(runtime_b, fixture_b);
    stale.selection_id = preview_a.candidate.selection_id;
    assert_preflight_rejection_preserves_carry(
        runtime_b,
        fixture_b,
        place_interact_input(fixture_b.locomotion, stale),
        Reason::TargetChanged);
}

void test_place_preflight_rejections_preserve_frozen_carry() {
    using namespace interaction;
    const auto make_enabled = [](RuntimeFixture& fixture) {
        return InteractionRuntime(
            fixture.database,
            fixture.features,
            fixture.registry,
            fixture.surface_registry,
            fixture.place_library,
            RuntimeConfig{});
    };

    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, std::nullopt),
            Reason::TargetUnavailable);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        PlaceRequest wrong = place_request_for(runtime, fixture);
        ++wrong.held_target.id;
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, wrong),
            Reason::TargetChanged);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        PlaceRequest stale = place_request_for(runtime, fixture);
        ++stale.selection_id;
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, stale),
            Reason::TargetChanged);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        PlaceRequest missing_action_id = place_request_for(runtime, fixture);
        missing_action_id.request_id = 0U;
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, missing_action_id),
            Reason::TargetUnavailable);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        fixture.surface = fixture.surface_registry.upsert(
            runtime_fixture_detail::make_placement_surface(
                900U, vec3(0.55F, 0.65F, 3.0F)));
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        const PlaceStagingPreview far = runtime.preview_place(
            fixture.surface, fixture.place_affordance_id);
        assert(far.accepted);
        assert(!far.ready);
        assert(far.root_error_m > 0.25F);
        PlaceRequest request{};
        request.held_target = fixture.request.target;
        request.surface = fixture.surface;
        request.affordance_id = fixture.place_affordance_id;
        request.request_id = 1001U;
        request.selection_id = far.candidate.selection_id;
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, request),
            Reason::CorrectionLimit);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        fixture.surface = fixture.surface_registry.upsert(
            runtime_fixture_detail::make_placement_surface(
                900U, vec3(0.55F, 0.65F, 3.0F)));
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        const PlaceStagingPreview far_snapshot = runtime.preview_place(
            fixture.surface, fixture.place_affordance_id);
        assert(far_snapshot.accepted && !far_snapshot.ready);
        advance(runtime, fixture.locomotion);
        const PlaceStagingPreview moved_snapshot = runtime.preview_place(
            fixture.surface, fixture.place_affordance_id);
        assert(moved_snapshot.accepted);
        assert(moved_snapshot.candidate.selection_id !=
               far_snapshot.candidate.selection_id);
        PlaceRequest stale_far{};
        stale_far.held_target = fixture.request.target;
        stale_far.surface = fixture.surface;
        stale_far.affordance_id = fixture.place_affordance_id;
        stale_far.request_id = 1001U;
        stale_far.selection_id = far_snapshot.candidate.selection_id;
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, stale_far),
            Reason::TargetChanged);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            RuntimeConfig{});
        enter_carry_before_first_update(runtime, fixture);
        const RuntimeDiagnostics before = runtime.diagnostics();
        const InteractionTarget target_before =
            *fixture.registry.find(fixture.request.target);
        const PlacementSurface surface_before =
            *fixture.surface_registry.find(fixture.surface);
        const PlaceStagingPreview unavailable = runtime.preview_place(
            fixture.surface, fixture.place_affordance_id);
        assert(!unavailable.accepted);
        assert(unavailable.reason == Reason::PackUnavailable);
        assert(exact(runtime.diagnostics(), before));
        assert(exact(
            *fixture.registry.find(fixture.request.target), target_before));
        assert(exact(
            *fixture.surface_registry.find(fixture.surface), surface_before));
        PlaceRequest request{};
        request.held_target = fixture.request.target;
        request.surface = fixture.surface;
        request.affordance_id = fixture.place_affordance_id;
        request.request_id = 1001U;
        request.selection_id = 1U;
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, request),
            Reason::PackUnavailable);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        const RuntimeOutput carry = enter_carry_before_first_update(
            runtime, fixture);
        const int32_t pickup_clip = carry.diagnostics.clip;
        assert(pickup_clip >= 0);
        fixture.place_library.recorded.clear();
        fixture.database.active_hands.at(
            static_cast<size_t>(pickup_clip)) = 0U;
        const PlaceStagingPreview rejected = runtime.preview_place(
            fixture.surface, fixture.place_affordance_id);
        assert(!rejected.accepted);
        assert(rejected.reason == Reason::NoCandidate);
        PlaceRequest request{};
        request.held_target = fixture.request.target;
        request.surface = fixture.surface;
        request.affordance_id = fixture.place_affordance_id;
        request.request_id = 1001U;
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, request),
            Reason::NoCandidate);
    }

    const auto assert_mutated_snapshot_rejects = [](
        RuntimeFixture& fixture,
        InteractionRuntime& runtime,
        bool mutate_target,
        bool mutate_library,
        bool mutate_surface,
        Reason expected_reason) {
        InteractionRuntime paused_control = runtime;
        const PlaceRequest request = place_request_for(runtime, fixture);
        RuntimeOutput output = runtime.update(place_interact_input(
            fixture.locomotion, request));
        assert(output.diagnostics.state == RuntimeState::PlacePreflight);
        const Pose frozen_pose = output.pose;
        const Transform frozen_object = output.object_world;
        if (mutate_target) {
            InteractionTarget* held = fixture.registry.find(
                fixture.request.target);
            assert(held != nullptr);
            held->object_dimensions.x = std::nextafter(
                held->object_dimensions.x,
                std::numeric_limits<float>::infinity());
        }
        if (mutate_library) {
            fixture.place_library.recorded.front()
                .object_poses.front().position.x += 0.001F;
        }
        if (mutate_surface) {
            PlacementSurface replacement =
                runtime_fixture_detail::make_placement_surface();
            replacement.surface_world.position.x += 0.01F;
            replacement.support_volume_world.position.x += 0.01F;
            fixture.surface_registry.upsert(replacement);
        }
        output = advance(runtime, fixture.locomotion);
        assert(output.diagnostics.state == RuntimeState::Carry);
        assert(output.diagnostics.result == ResultCode::Rejected);
        assert(output.diagnostics.reason == expected_reason);
        assert(output.diagnostics.attached);
        assert(exact(output.pose, frozen_pose));
        assert(exact(output.object_world, frozen_object));
        assert_held_by_original_owner(fixture);
        const RuntimeOutput trial_next = advance(runtime, fixture.locomotion);
        const RuntimeOutput control_next = advance(
            paused_control, fixture.locomotion);
        assert(exact(trial_next.pose, control_next.pose));
        assert(exact(trial_next.object_world, control_next.object_world));
    };
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        assert_mutated_snapshot_rejects(
            fixture, runtime, true, false, false, Reason::TargetChanged);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        assert_mutated_snapshot_rejects(
            fixture, runtime, false, true, false, Reason::TargetChanged);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        assert_mutated_snapshot_rejects(
            fixture, runtime, false, false, true, Reason::SurfaceChanged);
    }
}

void test_post_begin_place_failures_reconstruct_fresh_carry() {
    using namespace interaction;
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            fixture.surface_registry,
            fixture.place_library,
            RuntimeConfig{});
        enter_carry_before_first_update(runtime, fixture);
        RuntimeOutput output = enter_ready_place_align(runtime, fixture);
        for (int update = 0; update < 4; ++update) {
            output = advance(runtime, fixture.locomotion);
            assert(output.diagnostics.state == RuntimeState::PlaceAlign);
        }
        InteractionRuntime uncancelled = runtime;
        output = runtime.update(cancel_input(fixture.locomotion));
        const RuntimeOutput corresponding = advance(
            uncancelled, fixture.locomotion);
        assert(output.diagnostics.state == RuntimeState::Carry);
        assert(output.diagnostics.result == ResultCode::Cancelled);
        assert(output.diagnostics.reason == Reason::Cancelled);
        assert(output.diagnostics.attached);
        assert(exact(output.pose, corresponding.pose));
        assert(exact(output.object_world, corresponding.object_world));
        assert_fresh_carry_next_sample(runtime, fixture, output);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            fixture.surface_registry,
            fixture.place_library,
            RuntimeConfig{});
        enter_carry_before_first_update(runtime, fixture);
        RuntimeOutput output = enter_ready_place_align(runtime, fixture);
        while (output.diagnostics.state == RuntimeState::PlaceAlign) {
            output = advance(runtime, fixture.locomotion);
        }
        assert(output.diagnostics.state == RuntimeState::PlaceReplay);
        assert(output.diagnostics.place.committed);
        fixture.place_library.recorded.front()
            .poses.at(static_cast<size_t>(
                fixture.place_library.recorded.front().release_frame))
            .positions[kRightHandBone].x += 0.021F;
        RuntimeOutput last_safe = output;
        for (int update = 0; update < kMaximumUpdates; ++update) {
            output = advance(runtime, fixture.locomotion);
            if (output.diagnostics.state == RuntimeState::Carry) break;
            last_safe = output;
        }
        assert(output.diagnostics.state == RuntimeState::Carry);
        assert(output.diagnostics.result == ResultCode::Failed);
        assert(output.diagnostics.reason == Reason::ReleasePosition);
        assert(output.diagnostics.hand_position_error_m > 0.02F);
        assert(output.diagnostics.attached);
        assert(output.diagnostics.object_state == ObjectState::Held);
        assert(exact(output.pose, last_safe.pose));
        assert(exact(output.object_world, last_safe.object_world));
        assert_held_by_original_owner(fixture);
        assert_fresh_carry_next_sample(runtime, fixture, output);
    }
    {
        RuntimeFixture trial_fixture = make_runtime_fixture();
        RuntimeFixture control_fixture = make_runtime_fixture();
        InteractionRuntime trial(
            trial_fixture.database,
            trial_fixture.features,
            trial_fixture.registry,
            trial_fixture.surface_registry,
            trial_fixture.place_library,
            RuntimeConfig{});
        InteractionRuntime control(
            control_fixture.database,
            control_fixture.features,
            control_fixture.registry,
            control_fixture.surface_registry,
            control_fixture.place_library,
            RuntimeConfig{});
        enter_carry_before_first_update(trial, trial_fixture);
        enter_carry_before_first_update(control, control_fixture);
        RuntimeOutput trial_output = enter_ready_place_align(
            trial, trial_fixture);
        RuntimeOutput control_output = enter_ready_place_align(
            control, control_fixture);
        assert(exact(trial_output, control_output));

        PlacementSurface replacement =
            runtime_fixture_detail::make_placement_surface();
        replacement.surface_world.position.x += 0.01F;
        replacement.support_volume_world.position.x += 0.01F;
        const SurfaceHandle replacement_handle =
            trial_fixture.surface_registry.upsert(replacement);
        assert(replacement_handle.generation ==
               trial_fixture.surface.generation + 1U);
        const InteractionTarget held_before =
            *trial_fixture.registry.find(trial_fixture.request.target);
        for (int update = 0; update < kMaximumUpdates; ++update) {
            trial_output = advance(trial, trial_fixture.locomotion);
            control_output = advance(control, control_fixture.locomotion);
            if (trial_output.diagnostics.state == RuntimeState::Carry) break;
            assert(exact(trial_output.pose, control_output.pose));
            assert(exact(trial_output.object_world, control_output.object_world));
        }
        assert(trial_output.diagnostics.state == RuntimeState::Carry);
        assert(trial_output.diagnostics.result == ResultCode::Failed);
        assert(trial_output.diagnostics.reason == Reason::SurfaceChanged);
        assert(trial_output.diagnostics.attached);
        assert(control_output.diagnostics.state == RuntimeState::PlaceRelease);
        assert(exact(trial_output.pose, control_output.pose));
        assert(exact(trial_output.object_world, control_output.object_world));
        assert(exact(
            *trial_fixture.registry.find(trial_fixture.request.target),
            held_before));
        assert_fresh_carry_next_sample(
            trial, trial_fixture, trial_output);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        fixture.registry = TargetRegistry{};
        InteractionTarget target = runtime_fixture_detail::make_target();
        target.handle.generation = std::numeric_limits<uint32_t>::max();
        const TargetHandle maximum = fixture.registry.upsert(target);
        assert(maximum.generation == std::numeric_limits<uint32_t>::max());
        fixture.request = {
            maximum,
            runtime_fixture_detail::kAffordanceId,
            runtime_fixture_detail::kRequestId,
        };
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            fixture.surface_registry,
            fixture.place_library,
            RuntimeConfig{});
        enter_carry_before_first_update(runtime, fixture);
        RuntimeOutput output = enter_ready_place_align(runtime, fixture);
        const InteractionTarget held_before =
            *fixture.registry.find(fixture.request.target);
        for (int update = 0; update < kMaximumUpdates; ++update) {
            output = advance(runtime, fixture.locomotion);
            if (output.diagnostics.state == RuntimeState::Carry) break;
        }
        assert(output.diagnostics.state == RuntimeState::Carry);
        assert(output.diagnostics.result == ResultCode::Failed);
        assert(output.diagnostics.reason == Reason::TargetChanged);
        assert(output.diagnostics.attached);
        assert(output.diagnostics.target == maximum);
        assert(exact(
            *fixture.registry.find(fixture.request.target), held_before));
        assert(exact(
            fixture.registry.find(fixture.request.target)->table_world,
            held_before.table_world));
        assert(exact(
            fixture.registry.find(fixture.request.target)->table_size,
            held_before.table_size));
        assert_fresh_carry_next_sample(runtime, fixture, output);
    }
}

void test_place_cancellation_boundaries_for_recorded_and_reverse() {
    using namespace interaction;
    for (bool reverse : {false, true}) {
        for (int relative_to_commit : {-1, 0, 1}) {
            RuntimeFixture trial_fixture = reverse
                ? reverse_place_runtime_fixture()
                : make_runtime_fixture();
            RuntimeFixture control_fixture = reverse
                ? reverse_place_runtime_fixture()
                : make_runtime_fixture();
            RuntimeConfig runtime_config{};
            if (reverse) {
                runtime_config.ik.accepted_position_m =
                    runtime_config.ik.maximum_request_position_m;
                runtime_config.ik.accepted_orientation_radians =
                    runtime_config.ik.maximum_request_orientation_radians;
            }
            InteractionRuntime trial(
                trial_fixture.database,
                trial_fixture.features,
                trial_fixture.registry,
                trial_fixture.surface_registry,
                trial_fixture.place_library,
                runtime_config);
            InteractionRuntime control(
                control_fixture.database,
                control_fixture.features,
                control_fixture.registry,
                control_fixture.surface_registry,
                control_fixture.place_library,
                runtime_config);
            enter_carry_before_first_update(trial, trial_fixture);
            enter_carry_before_first_update(control, control_fixture);
            const PlaceStagingPreview preview = trial.preview_place(
                trial_fixture.surface,
                trial_fixture.place_affordance_id);
            assert(preview.accepted && preview.ready);
            assert(preview.candidate.mode ==
                   (reverse
                        ? PlaceMotionMode::ReversedPickup
                        : PlaceMotionMode::RecordedPlace));
            assert(preview.candidate.direction == (reverse ? -1 : 1));
            RuntimeOutput trial_output = enter_ready_place_align(
                trial, trial_fixture);
            RuntimeOutput control_output = enter_ready_place_align(
                control, control_fixture);
            assert(exact(trial_output, control_output));

            const int32_t direction = preview.candidate.direction;
            const double target_source =
                preview.candidate.commit_frame +
                relative_to_commit * direction;
            const double source_before_cancel = target_source - direction;
            for (int update = 0; update < kMaximumUpdates; ++update) {
                if (trial_output.diagnostics.place.source_frame_exact ==
                    source_before_cancel) {
                    break;
                }
                if (direction > 0) {
                    assert(
                        trial_output.diagnostics.place.source_frame_exact <
                        source_before_cancel);
                } else {
                    assert(
                        trial_output.diagnostics.place.source_frame_exact >
                        source_before_cancel);
                }
                trial_output = advance(trial, trial_fixture.locomotion);
                control_output = advance(control, control_fixture.locomotion);
                assert(exact(trial_output, control_output));
            }
            assert(trial_output.diagnostics.place.source_frame_exact ==
                   source_before_cancel);

            trial_output = trial.update(cancel_input(
                trial_fixture.locomotion));
            control_output = advance(control, control_fixture.locomotion);
            assert(control_output.diagnostics.place.source_frame_exact ==
                   target_source);
            if (relative_to_commit < 0) {
                assert(trial_output.diagnostics.state == RuntimeState::Carry);
                assert(trial_output.diagnostics.result ==
                       ResultCode::Cancelled);
                assert(trial_output.diagnostics.reason == Reason::Cancelled);
                assert(trial_output.diagnostics.attached);
                assert(control_output.diagnostics.state ==
                       RuntimeState::PlaceAlign);
                assert(exact(trial_output.pose, control_output.pose));
                assert(exact(
                    trial_output.object_world,
                    control_output.object_world));
                assert_fresh_carry_next_sample(
                    trial, trial_fixture, trial_output, runtime_config);
            } else {
                assert(exact(trial_output, control_output));
                assert(trial_output.diagnostics.state ==
                       RuntimeState::PlaceReplay);
                assert(trial_output.diagnostics.attached);
                assert(trial_output.diagnostics.object_state ==
                       ObjectState::Held);
                assert(trial_output.diagnostics.place.committed);
                assert(trial_output.diagnostics.reason == Reason::None);
            }
        }
    }
}

void test_reset_and_duplicate_edges_are_ignored_in_every_place_state() {
    using namespace interaction;
    for (bool reset : {false, true}) {
        RuntimeFixture trial_fixture = make_runtime_fixture();
        RuntimeFixture control_fixture = make_runtime_fixture();
        InteractionRuntime trial(
            trial_fixture.database,
            trial_fixture.features,
            trial_fixture.registry,
            trial_fixture.surface_registry,
            trial_fixture.place_library,
            RuntimeConfig{});
        InteractionRuntime control(
            control_fixture.database,
            control_fixture.features,
            control_fixture.registry,
            control_fixture.surface_registry,
            control_fixture.place_library,
            RuntimeConfig{});
        enter_carry_before_first_update(trial, trial_fixture);
        enter_carry_before_first_update(control, control_fixture);
        const PlaceRequest request = place_request_for(
            trial, trial_fixture, reset ? 2001U : 2002U);
        RuntimeOutput trial_output = trial.update(place_interact_input(
            trial_fixture.locomotion, request));
        RuntimeOutput control_output = control.update(place_interact_input(
            control_fixture.locomotion, request));
        assert(exact(trial_output, control_output));
        assert(trial_output.diagnostics.state ==
               RuntimeState::PlacePreflight);

        uint32_t observed_states = 0U;
        for (int update = 0; update < kMaximumUpdates; ++update) {
            switch (trial_output.diagnostics.state) {
                case RuntimeState::PlacePreflight:
                    observed_states |= 1U << 0U;
                    break;
                case RuntimeState::PlaceAlign:
                    observed_states |= 1U << 1U;
                    break;
                case RuntimeState::PlaceReplay:
                    observed_states |= 1U << 2U;
                    break;
                case RuntimeState::PlaceRelease:
                    observed_states |= 1U << 3U;
                    break;
                case RuntimeState::Locomotion:
                    break;
                default:
                    assert(false && "unexpected state in place equivalence");
            }
            if (trial_output.diagnostics.state == RuntimeState::Locomotion) {
                break;
            }

            RuntimeInput trial_input = idle_input(trial_fixture.locomotion);
            if (reset) {
                trial_input.reset_pressed = true;
            } else {
                trial_input.interact_pressed = true;
                trial_input.place_request = request;
            }
            trial_output = trial.update(trial_input);
            control_output = advance(control, control_fixture.locomotion);
            assert(exact(trial_output, control_output));
        }
        assert(observed_states == 0x0FU);
        assert(trial_output.diagnostics.state == RuntimeState::Locomotion);
        assert(trial_output.diagnostics.result == ResultCode::Succeeded);
        assert(trial_output.diagnostics.reason == Reason::None);
    }
}

void test_recorded_and_reverse_place_diagnostics_are_deterministic() {
    using namespace interaction;
    for (bool reverse : {false, true}) {
        RuntimeFixture first_fixture = reverse
            ? reverse_place_runtime_fixture()
            : make_runtime_fixture();
        RuntimeFixture second_fixture = reverse
            ? reverse_place_runtime_fixture()
            : make_runtime_fixture();
        RuntimeConfig config{};
        if (reverse) {
            config.ik.accepted_position_m =
                config.ik.maximum_request_position_m;
            config.ik.accepted_orientation_radians =
                config.ik.maximum_request_orientation_radians;
        }
        InteractionRuntime first(
            first_fixture.database,
            first_fixture.features,
            first_fixture.registry,
            first_fixture.surface_registry,
            first_fixture.place_library,
            config);
        InteractionRuntime second(
            second_fixture.database,
            second_fixture.features,
            second_fixture.registry,
            second_fixture.surface_registry,
            second_fixture.place_library,
            config);
        enter_carry_before_first_update(first, first_fixture);
        enter_carry_before_first_update(second, second_fixture);
        const PlaceStagingPreview preview = first.preview_place(
            first_fixture.surface, first_fixture.place_affordance_id);
        const PlaceStagingPreview duplicate = second.preview_place(
            second_fixture.surface, second_fixture.place_affordance_id);
        assert(preview.accepted && preview.ready);
        assert(exact(preview, duplicate));
        const PlaceMotionMode expected_mode = reverse
            ? PlaceMotionMode::ReversedPickup
            : PlaceMotionMode::RecordedPlace;
        assert(preview.candidate.mode == expected_mode);

        const PlaceRequest request = place_request_for(first, first_fixture);
        RuntimeOutput first_output = first.update(place_interact_input(
            first_fixture.locomotion, request));
        RuntimeOutput second_output = second.update(place_interact_input(
            second_fixture.locomotion, request));
        assert(exact(first_output, second_output));
        double previous_source =
            first_output.diagnostics.place.source_frame_exact;
        bool have_source = false;
        bool saw_align = false;
        bool saw_replay = false;
        bool saw_release = false;
        int stop_sample_count = 0;
        for (int update = 0; update < kMaximumUpdates; ++update) {
            const RuntimeState state = first_output.diagnostics.state;
            if (state == RuntimeState::Locomotion) break;
            assert(state == RuntimeState::PlacePreflight ||
                   state == RuntimeState::PlaceAlign ||
                   state == RuntimeState::PlaceReplay ||
                   state == RuntimeState::PlaceRelease);
            assert(first_output.owns_pose);
            assert(first_output.suppress_steering);
            assert(first_output.diagnostics.place.mode == expected_mode);
            assert(first_output.diagnostics.place.selection_id ==
                   preview.candidate.selection_id);
            assert(first_output.diagnostics.place.ik_config_fingerprint ==
                   preview.ik_config_fingerprint);
            assert(exact(
                first_output.diagnostics.place.effective_ik,
                config.ik));
            assert(first_output.diagnostics.place.candidate_certified);
            if (state != RuntimeState::PlacePreflight) {
                assert(first_output.diagnostics.place.preflight_config_identity);
            }
            for (float correction : {
                     first_output.diagnostics.place
                         .requested_root_correction_m,
                     first_output.diagnostics.place.applied_root_correction_m,
                     first_output.diagnostics.place
                         .requested_yaw_correction_radians,
                     first_output.diagnostics.place
                         .applied_yaw_correction_radians,
                     first_output.diagnostics.place
                         .requested_hand_correction_m,
                     first_output.diagnostics.place.applied_hand_correction_m,
                     first_output.diagnostics.place
                         .requested_hand_orientation_radians,
                     first_output.diagnostics.place
                         .applied_hand_orientation_radians}) {
                assert(std::isfinite(correction));
                assert(correction >= 0.0F);
            }
            if (state == RuntimeState::PlaceAlign ||
                state == RuntimeState::PlaceReplay ||
                state == RuntimeState::PlaceRelease) {
                const double source =
                    first_output.diagnostics.place.source_frame_exact;
                assert(std::isfinite(source));
                if (have_source) {
                    if (preview.candidate.direction > 0) {
                        assert(source >= previous_source);
                    } else {
                        assert(source <= previous_source);
                    }
                }
                previous_source = source;
                have_source = true;
                if (source == preview.candidate.stop_frame) {
                    ++stop_sample_count;
                }
            }
            saw_align = saw_align || state == RuntimeState::PlaceAlign;
            saw_replay = saw_replay || state == RuntimeState::PlaceReplay;
            saw_release = saw_release || state == RuntimeState::PlaceRelease;
            if (state == RuntimeState::PlaceRelease) {
                assert(first_output.diagnostics.place.released);
                assert(!first_output.diagnostics.attached);
                assert(first_output.diagnostics.place
                           .requested_root_correction_m == 0.0F);
                assert(first_output.diagnostics.place
                           .applied_root_correction_m == 0.0F);
                assert(first_output.diagnostics.place
                           .requested_yaw_correction_radians == 0.0F);
                assert(first_output.diagnostics.place
                           .applied_yaw_correction_radians == 0.0F);
                assert(first_output.diagnostics.place
                           .requested_hand_correction_m == 0.0F);
                assert(first_output.diagnostics.place
                           .applied_hand_correction_m == 0.0F);
                assert(first_output.diagnostics.place
                           .requested_hand_orientation_radians == 0.0F);
                assert(first_output.diagnostics.place
                           .applied_hand_orientation_radians == 0.0F);
            }
            first_output = advance(first, first_fixture.locomotion);
            second_output = advance(second, second_fixture.locomotion);
            assert(exact(first_output, second_output));
        }
        assert(first_output.diagnostics.state == RuntimeState::Locomotion);
        assert(first_output.diagnostics.result == ResultCode::Succeeded);
        assert(first_output.diagnostics.reason == Reason::None);
        assert(saw_align && saw_replay && saw_release);
        assert(stop_sample_count == 1);
    }
}

void test_disabled_is_permanent_pose_passthrough() {
    using namespace interaction;
    const RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime = InteractionRuntime::disabled(
        Reason::PackUnavailable);

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(runtime.state() == RuntimeState::Disabled);
    assert(output.diagnostics.state == RuntimeState::Disabled);
    assert(output.diagnostics.reason == Reason::PackUnavailable);
    assert(!output.diagnostics.pack_available);
    assert(!output.owns_pose && !output.suppress_steering);
    assert(exact(output.pose, fixture.locomotion.pose));

    output = runtime.update(reset_input(fixture.locomotion));
    assert(output.diagnostics.state == RuntimeState::Disabled);
    assert(output.diagnostics.reason == Reason::PackUnavailable);
    assert(!output.owns_pose && !output.suppress_steering);
    assert(exact(output.pose, fixture.locomotion.pose));

    for (float arbitrary_dt : {
             0.0F,
             -1.0F,
             std::numeric_limits<float>::quiet_NaN(),
             std::numeric_limits<float>::infinity()}) {
        RuntimeInput arbitrary = interact_input(
            fixture.locomotion, fixture.request);
        arbitrary.dt = arbitrary_dt;
        output = runtime.update(arbitrary);
        assert(output.diagnostics.state == RuntimeState::Disabled);
        assert(output.diagnostics.reason == Reason::PackUnavailable);
        assert(!output.owns_pose && !output.suppress_steering);
        assert(exact(output.pose, fixture.locomotion.pose));
    }
}

void test_enabled_runtime_rejects_noncanonical_dt_before_mutation() {
    using namespace interaction;
    const std::array<float, 6> invalid_steps = {
        0.0F,
        1.0F / 60.0F,
        0.12F,
        -kDt,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
    };
    for (float invalid_dt : invalid_steps) {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            RuntimeConfig{});
        const RuntimeDiagnostics before = runtime.diagnostics();
        RuntimeInput invalid = interact_input(
            fixture.locomotion, fixture.request);
        invalid.dt = invalid_dt;
        bool threw = false;
        try {
            (void)runtime.update(invalid);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
        assert(runtime.state() == RuntimeState::Locomotion);
        assert(exact(runtime.diagnostics(), before));
        assert_free(fixture.registry, fixture.request.target);

        const RuntimeOutput retried = runtime.update(interact_input(
            fixture.locomotion, fixture.request));
        assert(retried.diagnostics.state == RuntimeState::Preflight);
        assert_free(fixture.registry, fixture.request.target);
    }
}

void test_noncanonical_dt_cannot_mutate_owned_runtime_state() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    RuntimeFixture reference_fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        RuntimeConfig{});
    InteractionRuntime reference(
        reference_fixture.database,
        reference_fixture.features,
        reference_fixture.registry,
        RuntimeConfig{});
    const RuntimeOutput before = enter_align(runtime, fixture);
    const RuntimeOutput reference_before = enter_align(
        reference, reference_fixture);
    assert(exact(before.diagnostics, reference_before.diagnostics));
    const RuntimeDiagnostics diagnostics_before = runtime.diagnostics();
    const InteractionTarget* target_before = fixture.registry.find(
        fixture.request.target);
    assert(target_before != nullptr);
    const Transform object_before = target_before->object_world;

    RuntimeInput invalid = cancel_input(fixture.locomotion);
    invalid.dt = 0.12F;
    bool threw = false;
    try {
        (void)runtime.update(invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    assert(runtime.state() == RuntimeState::Align);
    assert(exact(runtime.diagnostics(), diagnostics_before));
    const InteractionTarget* target_after = fixture.registry.find(
        fixture.request.target);
    assert(target_after != nullptr);
    assert(target_after->state == ObjectState::Targeted);
    assert(target_after->owner_request == fixture.request.request_id);
    assert(exact(target_after->object_world, object_before));

    const RuntimeOutput after = advance(runtime, fixture.locomotion);
    const RuntimeOutput reference_after = advance(
        reference, reference_fixture.locomotion);
    assert(exact(after.diagnostics, reference_after.diagnostics));
    assert(exact(after.pose, reference_after.pose));
    assert(exact(after.object_world, reference_after.object_world));
}

void test_interact_without_explicit_request_is_rejected_after_preflight() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, std::nullopt));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    assert(output.diagnostics.result == ResultCode::None);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, fixture.request.target);

    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == Reason::TargetUnavailable);
    assert(!output.diagnostics.attached);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, fixture.request.target);
}

void test_success_order_carry_and_reset() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    RuntimeConfig config{};
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    assert(runtime.state() == RuntimeState::Locomotion);

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, fixture.request.target);

    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    assert(output.diagnostics.result == ResultCode::Accepted);
    assert(output.diagnostics.reason == Reason::None);
    assert(output.diagnostics.target == fixture.request.target);
    assert(output.diagnostics.affordance_id == fixture.request.affordance_id);
    assert(output.diagnostics.object_state == ObjectState::Targeted);
    assert(output.diagnostics.pack_available);
    assert(output.owns_pose && !output.suppress_steering);
    const InteractionTarget* reserved = fixture.registry.find(
        fixture.request.target);
    assert(reserved != nullptr);
    assert(reserved->state == ObjectState::Targeted);
    assert(reserved->owner_request == fixture.request.request_id);

    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::PickupReplay,
        RuntimeState::Align);
    assert(output.diagnostics.result == ResultCode::Accepted);
    assert(output.owns_pose && output.suppress_steering);

    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Hold,
        RuntimeState::PickupReplay);
    assert(output.diagnostics.attached);
    assert(output.diagnostics.object_state == ObjectState::Attached);
    assert(output.owns_pose && output.suppress_steering);

    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Carry,
        RuntimeState::Hold);
    assert(output.diagnostics.result == ResultCode::Succeeded);
    assert(output.diagnostics.reason == Reason::None);
    assert(output.diagnostics.attached);
    assert(output.diagnostics.object_state == ObjectState::Held);
    assert(output.owns_pose && !output.suppress_steering);

    const uint32_t generation_before_reset =
        runtime.diagnostics().target.generation;
    output = runtime.update(reset_input(fixture.locomotion));
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Reset);
    assert(output.diagnostics.reason == Reason::Reset);
    assert(!output.diagnostics.attached);
    assert(!output.diagnostics.inactive_arm_tracks_locomotion);
    assert(!output.owns_pose && !output.suppress_steering);
    assert(exact(output.object_world, fixture.original_object_world));

    const std::optional<TargetHandle> restored =
        fixture.registry.resolve_single_target(vec3(0.0F, 0.0F, 3.0F), 1.0F);
    assert(restored.has_value());
    assert(restored->generation == generation_before_reset + 1U);
    const InteractionTarget* restored_target = fixture.registry.find(*restored);
    assert(restored_target != nullptr);
    assert(restored_target->state == ObjectState::Free);
    assert(restored_target->owner_request == 0U);
    assert(exact(restored_target->object_world, fixture.original_object_world));

    output = advance(runtime, fixture.locomotion);
    const std::optional<TargetHandle> after_idle =
        fixture.registry.resolve_single_target(vec3(0.0F, 0.0F, 3.0F), 1.0F);
    assert(after_idle == restored);
    assert(output.diagnostics.result == ResultCode::Reset);
}

void test_layered_carry_publishes_inactive_arm_authority_transactionally() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    RuntimeConfig config{};
    config.carry.minimum_root_displacement_m = 10.0F;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);

    RuntimeOutput output = enter_carry_before_first_update(runtime, fixture);
    assert(!output.diagnostics.inactive_arm_targets_locomotion);
    LocomotionSnapshot carry_locomotion = fixture.locomotion;
    carry_locomotion.pose = output.pose;
    for (int update = 0;
         update < 32 &&
         !output.diagnostics.inactive_arm_tracks_locomotion;
         ++update) {
        output = advance(runtime, carry_locomotion);
        if (!output.diagnostics.inactive_arm_tracks_locomotion) {
            assert(output.diagnostics.state == RuntimeState::Carry);
            assert(!output.diagnostics.recorded_carry);
            assert(output.diagnostics.inactive_arm_targets_locomotion);
        }
    }
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(!output.diagnostics.recorded_carry);
    assert(output.diagnostics.inactive_arm_targets_locomotion);
    assert(output.diagnostics.inactive_arm_tracks_locomotion);

    const RuntimeDiagnostics before_failure = runtime.diagnostics();
    RuntimeInput invalid = idle_input(carry_locomotion);
    invalid.locomotion.pose.positions[g1_skeleton::Simulation].x =
        std::numeric_limits<float>::quiet_NaN();
    bool threw = false;
    try {
        (void)runtime.update(invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    assert(exact(runtime.diagnostics(), before_failure));

    output = advance(runtime, carry_locomotion);
    assert(output.diagnostics.inactive_arm_tracks_locomotion);
    output = runtime.update(reset_input(carry_locomotion));
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(!output.diagnostics.inactive_arm_targets_locomotion);
    assert(!output.diagnostics.inactive_arm_tracks_locomotion);
}

void test_recorded_and_weighted_carry_never_publish_inactive_arm_authority() {
    using namespace interaction;
    for (bool weighted_layered : {false, true}) {
        RuntimeFixture fixture = make_runtime_fixture();
        RuntimeConfig config{};
        if (weighted_layered) {
            config.carry.minimum_root_displacement_m = 10.0F;
            config.carry.inactive_arm_weight = 0.25F;
        }
        InteractionRuntime runtime(
            fixture.database, fixture.features, fixture.registry, config);
        RuntimeOutput output = enter_carry_before_first_update(runtime, fixture);
        assert(!output.diagnostics.inactive_arm_targets_locomotion);
        LocomotionSnapshot carry_locomotion = fixture.locomotion;
        carry_locomotion.pose = output.pose;
        bool saw_recorded = false;
        for (int update = 0; update < 24; ++update) {
            output = advance(runtime, carry_locomotion);
            assert(output.diagnostics.state == RuntimeState::Carry);
            assert(!output.diagnostics.inactive_arm_tracks_locomotion);
            assert(
                output.diagnostics.inactive_arm_targets_locomotion ==
                (!weighted_layered && !output.diagnostics.recorded_carry));
            saw_recorded = saw_recorded || output.diagnostics.recorded_carry;
        }
        if (!weighted_layered) assert(saw_recorded);
    }
}

void test_hand_constraint_weight_tracks_authored_reach_and_attachment() {
    using namespace interaction;
    using namespace interaction::runtime_fixture_detail;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    assert_valid_hand_constraint_weight(output.diagnostics);
    assert(output.diagnostics.hand_constraint_weight == 0.0F);

    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    constexpr int32_t kEntryFrame = kFramesPerClip + 10;
    constexpr int32_t kContactFrame =
        kFramesPerClip + kContactLocalFrame;
    assert(output.diagnostics.frame == kEntryFrame);
    assert(output.diagnostics.hand_constraint_weight == 0.0F);

    float previous_weight = output.diagnostics.hand_constraint_weight;
    bool saw_precontact_pickup_replay = false;
    bool saw_contact = false;
    bool saw_attached_pickup_replay = false;
    bool saw_hold = false;
    bool saw_carry = false;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        output = advance(runtime, fixture.locomotion);
        assert_valid_hand_constraint_weight(output.diagnostics);

        if (output.diagnostics.frame <= kContactFrame &&
            (output.diagnostics.state == RuntimeState::Align ||
             output.diagnostics.state == RuntimeState::PickupReplay)) {
            const float expected = std::clamp(
                static_cast<float>(
                    output.diagnostics.frame - kEntryFrame) /
                    static_cast<float>(kContactFrame - kEntryFrame),
                0.0F,
                1.0F);
            assert(output.diagnostics.hand_constraint_weight == expected);
            assert(output.diagnostics.hand_constraint_weight >=
                   previous_weight);
            previous_weight = output.diagnostics.hand_constraint_weight;
            if (output.diagnostics.state == RuntimeState::PickupReplay &&
                output.diagnostics.frame < kContactFrame) {
                saw_precontact_pickup_replay = true;
            }
            if (output.diagnostics.frame == kContactFrame) {
                saw_contact = true;
                assert(output.diagnostics.phase == Phase::Contact);
                assert(output.diagnostics.hand_constraint_weight == 1.0F);
            }
        }

        if (output.diagnostics.attached) {
            assert(output.diagnostics.hand_constraint_weight == 1.0F);
            if (output.diagnostics.state == RuntimeState::PickupReplay) {
                saw_attached_pickup_replay = true;
            }
        }
        if (output.diagnostics.state == RuntimeState::Hold) {
            saw_hold = true;
            assert(output.diagnostics.hand_constraint_weight == 1.0F);
        }
        if (output.diagnostics.state == RuntimeState::Carry) {
            saw_carry = true;
            assert(output.diagnostics.hand_constraint_weight == 1.0F);
            break;
        }
    }
    assert(saw_precontact_pickup_replay);
    assert(saw_contact);
    assert(saw_attached_pickup_replay);
    assert(saw_hold);
    assert(saw_carry);
}

void test_hand_constraint_weight_resets_on_cancel_failure_and_reset() {
    using namespace interaction;

    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            RuntimeConfig{});
        RuntimeOutput output = enter_align(runtime, fixture);
        while (output.diagnostics.hand_constraint_weight == 0.0F) {
            output = advance(runtime, fixture.locomotion);
            assert(output.diagnostics.state == RuntimeState::Align);
        }
        output = runtime.update(cancel_input(fixture.locomotion));
        assert(output.diagnostics.state == RuntimeState::Locomotion);
        assert(output.diagnostics.result == ResultCode::Cancelled);
        assert(output.diagnostics.hand_constraint_weight == 0.0F);
    }

    {
        RuntimeFixture fixture = contact_failure_fixture();
        RuntimeConfig config{};
        config.ik.accepted_position_m =
            config.ik.maximum_request_position_m;
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            config);
        RuntimeOutput output = enter_align(runtime, fixture);
        bool saw_failure = false;
        for (int update = 0; update < kMaximumUpdates; ++update) {
            output = advance(runtime, fixture.locomotion);
            assert_valid_hand_constraint_weight(output.diagnostics);
            if (output.diagnostics.result == ResultCode::Failed) {
                saw_failure = true;
                assert(output.diagnostics.hand_constraint_weight == 0.0F);
                break;
            }
        }
        assert(saw_failure);
    }

    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            RuntimeConfig{});
        RuntimeOutput output = enter_align(runtime, fixture);
        output = advance_until(
            runtime,
            fixture.locomotion,
            RuntimeState::PickupReplay,
            RuntimeState::Align);
        output = advance_until(
            runtime,
            fixture.locomotion,
            RuntimeState::Hold,
            RuntimeState::PickupReplay);
        output = advance_until(
            runtime,
            fixture.locomotion,
            RuntimeState::Carry,
            RuntimeState::Hold);
        assert(output.diagnostics.hand_constraint_weight == 1.0F);
        output = runtime.update(reset_input(fixture.locomotion));
        assert(output.diagnostics.state == RuntimeState::Locomotion);
        assert(output.diagnostics.result == ResultCode::Reset);
        assert(output.diagnostics.hand_constraint_weight == 0.0F);
        output = advance(runtime, fixture.locomotion);
        assert(output.diagnostics.hand_constraint_weight == 0.0F);
    }
}

void test_carry_reset_preserves_a_newer_authoritative_generation() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        RuntimeConfig{});
    RuntimeOutput output = enter_align(runtime, fixture);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::PickupReplay,
        RuntimeState::Align);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Hold,
        RuntimeState::PickupReplay);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Carry,
        RuntimeState::Hold);
    assert(output.diagnostics.attached);

    Transform authoritative = fixture.original_object_world;
    authoritative.position.x += 0.45F;
    authoritative.position.z -= 0.20F;
    const TargetHandle newer = fixture.registry.replace_pose(
        fixture.request.target.id, authoritative);
    assert(newer.generation == fixture.request.target.generation + 1U);

    output = runtime.update(reset_input(fixture.locomotion));
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Failed);
    assert(output.diagnostics.reason == Reason::TargetChanged);
    assert(!output.diagnostics.attached);
    assert(!output.diagnostics.inactive_arm_tracks_locomotion);
    assert(!output.owns_pose && !output.suppress_steering);
    assert(exact(output.object_world, authoritative));

    const InteractionTarget* preserved = fixture.registry.find(newer);
    assert(preserved != nullptr);
    assert(preserved->state == ObjectState::Free);
    assert(preserved->owner_request == 0U);
    assert(exact(preserved->object_world, authoritative));

    const RuntimeOutput idle = advance(runtime, fixture.locomotion);
    assert(idle.diagnostics.result == ResultCode::Failed);
    assert(idle.diagnostics.reason == Reason::TargetChanged);
    assert(exact(idle.object_world, authoritative));
    assert(fixture.registry.find(newer) != nullptr);
}

void test_carry_update_preserves_a_newer_authoritative_generation() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        RuntimeConfig{});
    RuntimeOutput output = enter_align(runtime, fixture);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::PickupReplay,
        RuntimeState::Align);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Hold,
        RuntimeState::PickupReplay);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Carry,
        RuntimeState::Hold);
    assert(output.diagnostics.attached);
    const InteractionTarget* held = fixture.registry.find(
        fixture.request.target);
    assert(held != nullptr);
    assert(held->state == ObjectState::Held);
    assert(held->owner_request == fixture.request.request_id);

    Transform authoritative = fixture.original_object_world;
    authoritative.position.x -= 0.35F;
    authoritative.position.z += 0.25F;
    const TargetHandle newer = fixture.registry.replace_pose(
        fixture.request.target.id, authoritative);
    assert(newer.generation == fixture.request.target.generation + 1U);
    LocomotionSnapshot current = fixture.locomotion;
    current.pose.positions[g1_skeleton::Hips].x += 0.17F;

    output = advance(runtime, current);

    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Failed);
    assert(output.diagnostics.reason == Reason::TargetChanged);
    assert(output.diagnostics.target == newer);
    assert(output.diagnostics.object_state == ObjectState::Free);
    assert(!output.diagnostics.attached);
    assert(!output.diagnostics.recorded_carry);
    assert(!output.diagnostics.inactive_arm_tracks_locomotion);
    assert(!output.owns_pose && !output.suppress_steering);
    assert(exact(output.pose, current.pose));
    assert(exact(output.object_world, authoritative));

    const InteractionTarget* preserved = fixture.registry.find(newer);
    assert(preserved != nullptr);
    assert(preserved->state == ObjectState::Free);
    assert(preserved->owner_request == 0U);
    assert(exact(preserved->object_world, authoritative));

    const RuntimeOutput idle = advance(runtime, current);
    assert(idle.diagnostics.target == newer);
    assert(exact(idle.object_world, authoritative));
    assert(fixture.registry.find(newer) != nullptr);
}

void test_align_cancel_releases_reservation() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    assert(fixture.registry.validate(
        fixture.request.target, fixture.request.request_id));

    output = runtime.update(cancel_input(fixture.locomotion));
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Cancelled);
    assert(output.diagnostics.reason == Reason::Cancelled);
    assert(!output.diagnostics.attached);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, fixture.request.target);
}

void test_entry_blends_the_whole_pose_for_quarter_second() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    fixture.locomotion.pose.positions[g1_skeleton::Hips].x += 0.30F;
    const float locomotion_hips_x =
        fixture.locomotion.pose.positions[g1_skeleton::Hips].x;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(exact(output.pose, fixture.locomotion.pose));
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    assert(exact(output.pose, fixture.locomotion.pose));

    for (int update = 0; update < 3; ++update) {
        output = advance(runtime, fixture.locomotion);
    }
    const float midpoint_x =
        output.pose.positions[g1_skeleton::Hips].x;
    assert(midpoint_x < locomotion_hips_x);
    assert(midpoint_x > 0.0F);

    for (int update = 3; update < 7; ++update) {
        output = advance(runtime, fixture.locomotion);
    }
    assert(std::abs(
        output.pose.positions[g1_skeleton::Hips].x) < 1.0e-6F);
}

void test_entry_blend_continues_after_early_commit() {
    using namespace interaction;
    RuntimeFixture early_fixture = make_runtime_fixture();
    RuntimeFixture align_fixture = make_runtime_fixture();
    early_fixture.locomotion.pose.positions[g1_skeleton::Hips].x += 0.30F;
    align_fixture.locomotion = early_fixture.locomotion;

    RuntimeConfig early_config{};
    early_config.playback.entry_blend_seconds = 0.50F;
    early_config.playback.commit_horizon_seconds = kDt;
    RuntimeConfig align_config = early_config;
    align_config.playback.commit_horizon_seconds = 0.50F;

    InteractionRuntime early(
        early_fixture.database,
        early_fixture.features,
        early_fixture.registry,
        early_config);
    InteractionRuntime align(
        align_fixture.database,
        align_fixture.features,
        align_fixture.registry,
        align_config);
    RuntimeOutput early_output = enter_align(early, early_fixture);
    RuntimeOutput align_output = enter_align(align, align_fixture);

    early_output = advance(early, early_fixture.locomotion);
    align_output = advance(align, align_fixture.locomotion);
    assert(early_output.diagnostics.state == RuntimeState::PickupReplay);
    assert(align_output.diagnostics.state == RuntimeState::Align);
    assert(exact(early_output.pose, align_output.pose));

    early_output = advance(early, early_fixture.locomotion);
    align_output = advance(align, align_fixture.locomotion);
    assert(early_output.diagnostics.state == RuntimeState::PickupReplay);
    assert(align_output.diagnostics.state == RuntimeState::Align);
    assert(exact(early_output.pose, align_output.pose));
}

void test_cancel_is_ignored_after_commit_and_source_frame_is_monotonic() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);

    int32_t previous_frame = output.diagnostics.frame;
    while (output.diagnostics.state == RuntimeState::Align) {
        output = advance(runtime, fixture.locomotion);
        assert(output.diagnostics.frame >= previous_frame);
        previous_frame = output.diagnostics.frame;
    }
    assert(output.diagnostics.state == RuntimeState::PickupReplay);
    assert(output.diagnostics.frame >= previous_frame);
    const int32_t committed_frame = output.diagnostics.frame;

    output = runtime.update(cancel_input(fixture.locomotion));
    assert(output.diagnostics.state == RuntimeState::PickupReplay);
    assert(output.diagnostics.result == ResultCode::Accepted);
    assert(output.diagnostics.reason == Reason::None);
    assert(output.diagnostics.frame >= committed_frame);
    assert(output.owns_pose && output.suppress_steering);
    assert(fixture.registry.validate(
        fixture.request.target, fixture.request.request_id));
}

void test_commit_at_contact_preserves_the_one_shot_crossing() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    RuntimeConfig config{};
    constexpr int32_t kEntryFrame =
        runtime_fixture_detail::kFramesPerClip + 10;
    config.playback.commit_horizon_seconds =
        fixture.database.time_to_contact.at(kEntryFrame);
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::PickupReplay,
        RuntimeState::Align);
    assert(output.diagnostics.frame >=
        runtime_fixture_detail::kFramesPerClip +
        runtime_fixture_detail::kContactLocalFrame);

    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Hold,
        RuntimeState::PickupReplay);
    assert(output.diagnostics.attached);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Carry,
        RuntimeState::Hold);
    assert(output.diagnostics.result == ResultCode::Succeeded);
}

void test_nonunit_playback_speed_commits_by_source_contact() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    RuntimeConfig config{};
    config.playback.speed = 1.10F;
    config.playback.commit_horizon_seconds = 1.0F;
    config.playback.maximum_alignment_seconds = 1.0F;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    RuntimeOutput output = enter_align(runtime, fixture);

    constexpr int32_t kContactFrame =
        runtime_fixture_detail::kFramesPerClip +
        runtime_fixture_detail::kContactLocalFrame;
    for (int update = 0; update < 13; ++update) {
        output = advance(runtime, fixture.locomotion);
        assert(output.diagnostics.state == RuntimeState::Align);
        assert(output.diagnostics.frame < kContactFrame);
    }

    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::PickupReplay);
    assert(output.diagnostics.frame == kContactFrame);
    assert(output.diagnostics.playback_speed == config.playback.speed);
}

void test_canonical_updates_gate_the_exact_contact_pose() {
    using namespace interaction;
    RuntimeFixture fixture = contact_failure_fixture();
    RuntimeConfig config{};
    config.ik.accepted_position_m =
        config.ik.maximum_request_position_m;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);

    int32_t previous_frame = output.diagnostics.frame;
    bool saw_failed_contact = false;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        output = advance(runtime, fixture.locomotion);
        assert(output.diagnostics.frame >= previous_frame);
        previous_frame = output.diagnostics.frame;
        if (output.diagnostics.result == ResultCode::Failed) {
            assert(output.diagnostics.reason == Reason::ContactPosition);
            assert(!output.diagnostics.attached);
            saw_failed_contact = true;
            break;
        }
        assert(output.diagnostics.state != RuntimeState::Carry);
    }
    assert(saw_failed_contact);
}

void test_canonical_updates_cannot_skip_post_attach_contact_loss() {
    using namespace interaction;
    RuntimeFixture fixture = post_attach_contact_loss_fixture();
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        RuntimeConfig{});
    RuntimeOutput output = enter_align(runtime, fixture);

    for (int update = 0;
         update < kMaximumUpdates &&
         output.diagnostics.result != ResultCode::Failed &&
         output.diagnostics.state != RuntimeState::Carry;
         ++update) {
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.result == ResultCode::Failed);
    assert(output.diagnostics.reason == Reason::LostContact);
    assert(!output.diagnostics.attached);
}

void test_pickup_success_precedes_later_canonical_contact_loss() {
    using namespace interaction;
    RuntimeFixture fixture = post_success_contact_loss_fixture();
    RuntimeConfig config{};
    config.attachment.required_lift_m = 0.0F;
    config.attachment.required_hold_seconds = 0.50F;
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        config);
    RuntimeOutput output = enter_align(runtime, fixture);

    for (int update = 0;
         update < kMaximumUpdates &&
         output.diagnostics.state != RuntimeState::Carry &&
         output.diagnostics.result != ResultCode::Failed;
         ++update) {
        output = advance(runtime, fixture.locomotion);
    }

    constexpr int32_t kFirstHoldFrame =
        runtime_fixture_detail::kFramesPerClip +
        runtime_fixture_detail::kHoldLocalFrame;
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.result == ResultCode::Succeeded);
    assert(output.diagnostics.frame == kFirstHoldFrame);
    assert(output.diagnostics.object_state == ObjectState::Held);
}

void test_hold_interval_precedes_its_canonical_stop_frame_event() {
    using namespace interaction;
    RuntimeFixture fixture = post_success_contact_loss_fixture(0);
    RuntimeConfig config{};
    config.attachment.required_lift_m = 0.0F;
    config.attachment.required_hold_seconds = 0.58F;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    RuntimeOutput output = enter_align(runtime, fixture);
    for (int update = 0;
         update < kMaximumUpdates &&
         output.diagnostics.state != RuntimeState::Carry &&
         output.diagnostics.result != ResultCode::Failed;
         ++update) {
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.result == ResultCode::Succeeded);
    assert(output.diagnostics.frame ==
        runtime_fixture_detail::kFramesPerClip +
            runtime_fixture_detail::kHoldLocalFrame);
}

void test_post_ik_contact_sweep_cannot_cross_the_expanded_table() {
    using namespace interaction;
    RuntimeFixture fixture = post_ik_table_crossing_fixture();
    const InteractionTarget* authored = fixture.registry.find(
        fixture.request.target);
    const GraspAffordance* affordance = fixture.registry.find_affordance(
        fixture.request.target, fixture.request.affordance_id);
    assert(authored != nullptr && affordance != nullptr);
    const Transform target_hand = compose(
        authored->object_world, affordance->hand_in_object);
    const Transform entry_hand = {
        world_pose(fixture.locomotion.pose).positions[kRightHandBone],
        world_pose(fixture.locomotion.pose).rotations[kRightHandBone],
    };
    const float expanded_half_z =
        0.5F * authored->table_size.z + affordance->clearance_radius;
    assert(entry_hand.position.z <
        authored->table_world.position.z - expanded_half_z);
    assert(target_hand.position.z >
        authored->table_world.position.z + expanded_half_z);

    RuntimeConfig config{};
    config.ik.accepted_position_m = 0.01F;
    config.ik.maximum_iterations = 64;
    config.attachment.maximum_position_error_m = 0.02F;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);

    bool saw_blocked = false;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        output = advance(runtime, fixture.locomotion);
        if (output.diagnostics.result == ResultCode::Failed) {
            assert(output.diagnostics.reason == Reason::BlockedPath);
            assert(!output.diagnostics.attached);
            saw_blocked = true;
            break;
        }
        assert(output.diagnostics.state != RuntimeState::Carry);
    }
    assert(saw_blocked);
}

void test_post_ik_precontact_sweep_cannot_cross_the_expanded_object() {
    using namespace interaction;
    RuntimeFixture fixture = post_ik_object_crossing_fixture();
    const InteractionTarget* authored = fixture.registry.find(
        fixture.request.target);
    const GraspAffordance* affordance = fixture.registry.find_affordance(
        fixture.request.target, fixture.request.affordance_id);
    assert(authored != nullptr && affordance != nullptr);
    const Transform target_hand = compose(
        authored->object_world, affordance->hand_in_object);
    const float expanded_half_z =
        0.5F * authored->object_dimensions.z +
        affordance->clearance_radius;
    assert(target_hand.position.z >
        authored->object_world.position.z - expanded_half_z);
    assert(target_hand.position.z <
        authored->object_world.position.z + expanded_half_z);

    constexpr int32_t kPreContact =
        runtime_fixture_detail::kFramesPerClip +
        runtime_fixture_detail::kContactLocalFrame - 1;
    const float recorded_precontact_z = world_pose(
        pose_at_frame(fixture.database, kPreContact)).positions[
            kRightHandBone].z;
    const float correction_alpha =
        static_cast<float>(runtime_fixture_detail::kContactLocalFrame - 1 - 10) /
        static_cast<float>(runtime_fixture_detail::kContactLocalFrame - 10);
    const float corrected_precontact_z = recorded_precontact_z +
        correction_alpha * (
            target_hand.position.z - authored->object_world.position.z);
    assert(recorded_precontact_z <
        authored->object_world.position.z - expanded_half_z);
    assert(corrected_precontact_z >
        authored->object_world.position.z - expanded_half_z);
    assert(corrected_precontact_z <
        authored->object_world.position.z + expanded_half_z);

    RuntimeConfig config{};
    config.ik.accepted_position_m = 0.01F;
    config.ik.maximum_iterations = 64;
    config.attachment.maximum_position_error_m = 0.02F;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);

    bool saw_blocked = false;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        output = advance(runtime, fixture.locomotion);
        if (output.diagnostics.result == ResultCode::Failed) {
            assert(output.diagnostics.reason == Reason::BlockedPath);
            assert(!output.diagnostics.attached);
            saw_blocked = true;
            break;
        }
        assert(output.diagnostics.state != RuntimeState::Carry);
    }
    assert(saw_blocked);
}

void test_canonical_clearance_visits_curved_precontact_samples() {
    using namespace interaction;
    RuntimeFixture fixture = curved_precontact_clearance_fixture();
    const InteractionTarget* target = fixture.registry.find(
        fixture.request.target);
    const GraspAffordance* affordance = fixture.registry.find_affordance(
        fixture.request.target, fixture.request.affordance_id);
    assert(target != nullptr && affordance != nullptr);
    const float expanded_half_x =
        0.5F * target->table_size.x + affordance->clearance_radius;
    constexpr int32_t kIntermediateFrame =
        runtime_fixture_detail::kFramesPerClip + 18;
    const float authored_hand_x = world_pose(pose_at_frame(
        fixture.database, kIntermediateFrame)).positions[
            kRightHandBone].x;
    assert(std::abs(authored_hand_x - target->table_world.position.x) >
        expanded_half_x);

    RuntimeConfig config{};
    config.ik.accepted_position_m = 0.01F;
    config.ik.maximum_iterations = 64;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    RuntimeOutput output = enter_align(runtime, fixture);
    for (int update = 0;
         update < kMaximumUpdates &&
         output.diagnostics.result != ResultCode::Failed;
         ++update) {
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.result == ResultCode::Failed);
    assert(output.diagnostics.reason == Reason::BlockedPath);
    assert(!output.diagnostics.attached);
}

void test_stale_generation_is_rejected() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    const PickRequest stale_request = fixture.request;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, stale_request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    const TargetHandle replacement = fixture.registry.replace_pose(
        stale_request.target.id, fixture.original_object_world);
    assert(replacement.generation == stale_request.target.generation + 1U);

    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == Reason::TargetChanged);
    assert(!output.diagnostics.attached);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, replacement);
}

void test_align_rejects_profile_or_bounds_snapshot_changes() {
    using namespace interaction;

    for (int mutation = 0; mutation < 2; ++mutation) {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            RuntimeConfig{});
        RuntimeOutput output = enter_align(runtime, fixture);
        assert(output.diagnostics.state == RuntimeState::Align);

        InteractionTarget* authoritative = fixture.registry.find(
            fixture.request.target);
        assert(authoritative != nullptr);
        if (mutation == 0) {
            ++authoritative->object_profile_id;
        } else {
            authoritative->object_bounds.center_object.z += 0.001F;
        }

        output = advance(runtime, fixture.locomotion);
        assert(output.diagnostics.state == RuntimeState::Locomotion);
        assert(output.diagnostics.result == ResultCode::Rejected);
        assert(output.diagnostics.reason == Reason::TargetChanged);
        assert(!output.diagnostics.attached);
        assert(!output.owns_pose && !output.suppress_steering);
        const InteractionTarget* released = fixture.registry.find(
            fixture.request.target);
        assert(released != nullptr);
        assert(released->state == ObjectState::Free);
        assert(released->owner_request == 0U);
    }
}

void test_failed_reservation_never_releases_an_existing_owner() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    assert(fixture.registry.reserve(
        fixture.request.target, fixture.request.request_id));
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == Reason::TargetChanged);

    const InteractionTarget* target = fixture.registry.find(
        fixture.request.target);
    assert(target != nullptr);
    assert(target->state == ObjectState::Targeted);
    assert(target->owner_request == fixture.request.request_id);
}

void test_runtime_config_is_validated_before_reservation() {
    using namespace interaction;
    RuntimeConfig config{};
    config.matcher.maximum_approach_m = -1.0F;
    assert_invalid_runtime_config(config);

    config = RuntimeConfig{};
    config.matcher.group_weights.fill(0.0F);
    assert_invalid_runtime_config(config);

    config = RuntimeConfig{};
    config.playback.canonical_fps = 30.0F;
    assert_invalid_runtime_config(config);

    config = RuntimeConfig{};
    config.playback.speed = 2.0F;
    assert_invalid_runtime_config(config);

    config = RuntimeConfig{};
    config.ik.damping = 0.0F;
    assert_invalid_runtime_config(config);

    config = RuntimeConfig{};
    config.attachment.required_hold_seconds = -1.0F;
    assert_invalid_runtime_config(config);

    config = RuntimeConfig{};
    config.carry.search_interval_seconds = 0.0F;
    assert_invalid_runtime_config(config);
}

void test_exception_after_reservation_rolls_back_and_can_retry() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);

    const std::vector<float> offsets = fixture.features.offsets;
    fixture.features.offsets.clear();
    bool threw = false;
    try {
        (void)advance(runtime, fixture.locomotion);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
    assert(runtime.state() == RuntimeState::Preflight);
    assert_free(fixture.registry, fixture.request.target);

    fixture.features.offsets = offsets;
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    assert(output.diagnostics.result == ResultCode::Accepted);
    assert(fixture.registry.validate(
        fixture.request.target, fixture.request.request_id));
}

void test_high_cost_is_rejected_as_poor_match_and_stays_terminal() {
    using namespace interaction;
    RuntimeFixture fixture = high_cost_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == Reason::PoorMatch);
    assert_free(fixture.registry, fixture.request.target);

    const RuntimeDiagnostics terminal = output.diagnostics;
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == terminal.state);
    assert(output.diagnostics.result == terminal.result);
    assert(output.diagnostics.reason == terminal.reason);
    assert(output.diagnostics.target == terminal.target);
    assert(output.diagnostics.affordance_id == terminal.affordance_id);
    assert(output.diagnostics.clip == terminal.clip);
    assert(output.diagnostics.frame == terminal.frame);
    assert(!output.owns_pose && !output.suppress_steering);

    assert(run_rejected(high_cost_fixture()).reason == Reason::PoorMatch);
}

void test_contact_failure_is_one_shot_latched_and_drains_final_frame() {
    using namespace interaction;
    const RuntimeDiagnostics diagnostics =
        run_post_commit_failure(contact_failure_fixture());
    assert(diagnostics.reason == Reason::ContactPosition);
    assert(!diagnostics.attached);
    assert(diagnostics.frame ==
        runtime_fixture_detail::kFramesPerClip * 2 - 1);
}

void test_hold_extends_the_frozen_final_pose_without_resetting_timer() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    RuntimeConfig config{};
    config.attachment.required_hold_seconds = 2.0F;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::PickupReplay,
        RuntimeState::Align);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Hold,
        RuntimeState::PickupReplay);

    int hold_updates = 0;
    bool saw_final_frame = false;
    bool saw_frozen_extension = false;
    Pose final_pose{};
    while (output.diagnostics.state == RuntimeState::Hold) {
        if (output.diagnostics.frame ==
                fixture.database.range_stops.at(1) - 1 &&
            output.diagnostics.state != RuntimeState::Locomotion) {
            if (!saw_final_frame) {
                final_pose = output.pose;
                saw_final_frame = true;
            } else {
                assert(exact(output.pose, final_pose));
                saw_frozen_extension = true;
            }
        }
        ++hold_updates;
        assert(hold_updates < kMaximumUpdates);
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.result == ResultCode::Succeeded);
    assert(saw_final_frame && saw_frozen_extension);
    assert(static_cast<float>(hold_updates) * kDt <
        config.attachment.required_hold_seconds);
}

void test_post_attach_failure_preserves_object_and_frees_registry() {
    using namespace interaction;
    RuntimeFixture fixture = post_attach_contact_loss_fixture();
    const TargetHandle original_handle = fixture.request.target;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);

    bool saw_attached = false;
    bool saw_failure = false;
    bool saw_final_owned_frame = false;
    Transform last_attached_object{};
    for (int update = 0; update < kMaximumUpdates; ++update) {
        output = advance(runtime, fixture.locomotion);
        if (output.diagnostics.attached) {
            assert(!saw_failure);
            saw_attached = true;
            last_attached_object = output.object_world;
        }
        if (output.diagnostics.result == ResultCode::Failed) {
            saw_failure = true;
            assert(output.diagnostics.reason == Reason::LostContact);
            assert(!output.diagnostics.attached);
        }
        if (output.diagnostics.frame ==
                fixture.database.range_stops.at(1) - 1 &&
            output.diagnostics.state != RuntimeState::Locomotion) {
            assert(output.owns_pose);
            saw_final_owned_frame = true;
        }
        if (output.diagnostics.state == RuntimeState::Locomotion) break;
    }
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Failed);
    assert(output.diagnostics.reason == Reason::LostContact);
    assert(saw_attached && saw_failure && saw_final_owned_frame);
    assert(!output.diagnostics.attached);
    assert(exact(output.object_world, last_attached_object));

    const std::optional<TargetHandle> replacement =
        fixture.registry.resolve_single_target(
            last_attached_object.position, 1.0F);
    assert(replacement.has_value());
    assert(replacement->id == original_handle.id);
    assert(replacement->generation == original_handle.generation + 1U);
    const InteractionTarget* target = fixture.registry.find(*replacement);
    assert(target != nullptr);
    assert(target->state == ObjectState::Free);
    assert(target->owner_request == 0U);
    assert(exact(target->object_world, last_attached_object));
}

void test_post_attach_target_change_never_clobbers_a_newer_generation() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::PickupReplay,
        RuntimeState::Align);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Hold,
        RuntimeState::PickupReplay);
    assert(output.diagnostics.attached);

    Transform authoritative = fixture.original_object_world;
    authoritative.position.x += 0.30F;
    authoritative.position.z += 0.20F;
    const TargetHandle newer = fixture.registry.replace_pose(
        fixture.request.target.id, authoritative);
    assert(newer.generation == fixture.request.target.generation + 1U);

    bool saw_final_owned_frame = false;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        output = advance(runtime, fixture.locomotion);
        if (output.diagnostics.frame ==
                fixture.database.range_stops.at(1) - 1 &&
            output.diagnostics.state != RuntimeState::Locomotion) {
            assert(output.owns_pose);
            saw_final_owned_frame = true;
        }
        if (output.diagnostics.state == RuntimeState::Locomotion) break;
    }
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Failed);
    assert(output.diagnostics.reason == Reason::TargetChanged);
    assert(saw_final_owned_frame);

    const InteractionTarget* preserved = fixture.registry.find(newer);
    assert(preserved != nullptr);
    assert(preserved->state == ObjectState::Free);
    assert(preserved->owner_request == 0U);
    assert(exact(preserved->object_world, authoritative));
}

}  // namespace

int main() {
    test_frozen_public_contract_and_defaults();
    test_place_preview_and_collapsed_success_lifecycle();
    test_runtime_validates_complete_place_and_ik_configuration();
    test_runtime_forwards_nondefault_place_config_exactly();
    test_runtime_ik_identity_binds_every_scalar_and_iteration();
    test_place_preflight_rejections_preserve_frozen_carry();
    test_post_begin_place_failures_reconstruct_fresh_carry();
    test_place_cancellation_boundaries_for_recorded_and_reverse();
    test_reset_and_duplicate_edges_are_ignored_in_every_place_state();
    test_recorded_and_reverse_place_diagnostics_are_deterministic();
    test_disabled_is_permanent_pose_passthrough();
    test_enabled_runtime_rejects_noncanonical_dt_before_mutation();
    test_noncanonical_dt_cannot_mutate_owned_runtime_state();
    test_interact_without_explicit_request_is_rejected_after_preflight();
    test_success_order_carry_and_reset();
    test_layered_carry_publishes_inactive_arm_authority_transactionally();
    test_recorded_and_weighted_carry_never_publish_inactive_arm_authority();
    test_hand_constraint_weight_tracks_authored_reach_and_attachment();
    test_hand_constraint_weight_resets_on_cancel_failure_and_reset();
    test_carry_reset_preserves_a_newer_authoritative_generation();
    test_carry_update_preserves_a_newer_authoritative_generation();
    test_align_cancel_releases_reservation();
    test_entry_blends_the_whole_pose_for_quarter_second();
    test_entry_blend_continues_after_early_commit();
    test_cancel_is_ignored_after_commit_and_source_frame_is_monotonic();
    test_commit_at_contact_preserves_the_one_shot_crossing();
    test_nonunit_playback_speed_commits_by_source_contact();
    test_canonical_updates_gate_the_exact_contact_pose();
    test_canonical_updates_cannot_skip_post_attach_contact_loss();
    test_pickup_success_precedes_later_canonical_contact_loss();
    test_hold_interval_precedes_its_canonical_stop_frame_event();
    test_post_ik_contact_sweep_cannot_cross_the_expanded_table();
    test_post_ik_precontact_sweep_cannot_cross_the_expanded_object();
    test_canonical_clearance_visits_curved_precontact_samples();
    test_stale_generation_is_rejected();
    test_align_rejects_profile_or_bounds_snapshot_changes();
    test_failed_reservation_never_releases_an_existing_owner();
    test_runtime_config_is_validated_before_reservation();
    test_exception_after_reservation_rolls_back_and_can_retry();
    test_high_cost_is_rejected_as_poor_match_and_stays_terminal();
    test_contact_failure_is_one_shot_latched_and_drains_final_frame();
    test_hold_extends_the_frozen_final_pose_without_resetting_timer();
    test_post_attach_failure_preserves_object_and_frees_registry();
    test_post_attach_target_change_never_clobbers_a_newer_generation();
}
