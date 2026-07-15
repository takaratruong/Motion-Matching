#include "interaction_runtime.h"
#include "tests/cpp/interaction_runtime_fixture.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

constexpr float kDt = 1.0F / 25.0F;
constexpr int kMaximumUpdates = 600;

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
           left.inactive_arm_tracks_locomotion ==
               right.inactive_arm_tracks_locomotion &&
           left.pack_available == right.pack_available;
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

    static_assert(std::is_same_v<decltype(RuntimeConfig{}.matcher), MatchConfig>);
    static_assert(std::is_same_v<
        decltype(RuntimeConfig{}.playback), PlaybackConfig>);
    static_assert(std::is_same_v<decltype(RuntimeConfig{}.ik), IKConfig>);
    static_assert(std::is_same_v<
        decltype(RuntimeConfig{}.attachment), AttachmentConfig>);
    static_assert(std::is_same_v<decltype(RuntimeConfig{}.carry), CarryConfig>);
    static_assert(std::is_same_v<decltype(RuntimeInput{}.dt), float>);
    static_assert(std::is_same_v<
        decltype(RuntimeInput{}.locomotion), LocomotionSnapshot>);
    static_assert(std::is_same_v<
        decltype(RuntimeInput{}.pick_request), std::optional<PickRequest>>);
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
    assert(!diagnostics.inactive_arm_tracks_locomotion);
    assert(!diagnostics.pack_available);

    const RuntimeOutput output{};
    assert(!output.owns_pose && !output.suppress_steering);
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
        }
    }
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(!output.diagnostics.recorded_carry);
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
        LocomotionSnapshot carry_locomotion = fixture.locomotion;
        carry_locomotion.pose = output.pose;
        bool saw_recorded = false;
        for (int update = 0; update < 24; ++update) {
            output = advance(runtime, carry_locomotion);
            assert(output.diagnostics.state == RuntimeState::Carry);
            assert(!output.diagnostics.inactive_arm_tracks_locomotion);
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
    test_failed_reservation_never_releases_an_existing_owner();
    test_runtime_config_is_validated_before_reservation();
    test_exception_after_reservation_rolls_back_and_can_retry();
    test_high_cost_is_rejected_as_poor_match_and_stays_terminal();
    test_contact_failure_is_one_shot_latched_and_drains_final_frame();
    test_hold_extends_the_frozen_final_pose_without_resetting_timer();
    test_post_attach_failure_preserves_object_and_frees_registry();
    test_post_attach_target_change_never_clobbers_a_newer_generation();
}
