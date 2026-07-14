#include "interaction_controller_adapter.h"
#include "tests/cpp/interaction_runtime_fixture.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using interaction::ControllerInteractionAdapter;
using interaction::ControllerInteractionEdges;
using interaction::ControllerInteractionFrameHandoff;
using interaction::ControllerInteractionFrameState;
using interaction::ControllerInteractionSceneHandoff;
using interaction::ControllerInteractionSceneState;
using interaction::ControllerInteractionScheduler;
using interaction::Database;
using interaction::Features;
using interaction::GraspAffordance;
using interaction::Hand;
using interaction::InteractionTarget;
using interaction::LocomotionSnapshot;
using interaction::ObjectState;
using interaction::Phase;
using interaction::PickRequest;
using interaction::Pose;
using interaction::Reason;
using interaction::ResultCode;
using interaction::RuntimeInput;
using interaction::RuntimeOutput;
using interaction::RuntimeState;
using interaction::TargetHandle;
using interaction::Transform;

bool float_bits_equal(float left, float right) {
    return std::memcmp(&left, &right, sizeof(float)) == 0;
}

bool vec_bits_equal(vec3 left, vec3 right) {
    return float_bits_equal(left.x, right.x) &&
           float_bits_equal(left.y, right.y) &&
           float_bits_equal(left.z, right.z);
}

bool quat_bits_equal(quat left, quat right) {
    return float_bits_equal(left.w, right.w) &&
           float_bits_equal(left.x, right.x) &&
           float_bits_equal(left.y, right.y) &&
           float_bits_equal(left.z, right.z);
}

bool pose_bits_equal(const Pose& left, const Pose& right) {
    for (size_t bone = 0; bone < left.positions.size(); ++bone) {
        if (!vec_bits_equal(left.positions[bone], right.positions[bone]) ||
            !vec_bits_equal(left.velocities[bone], right.velocities[bone]) ||
            !quat_bits_equal(left.rotations[bone], right.rotations[bone]) ||
            !vec_bits_equal(
                left.angular_velocities[bone],
                right.angular_velocities[bone])) {
            return false;
        }
    }
    for (size_t joint = 0; joint < left.hand_dof.size(); ++joint) {
        if (!float_bits_equal(left.hand_dof[joint], right.hand_dof[joint]) ||
            !float_bits_equal(
                left.hand_dof_velocities[joint],
                right.hand_dof_velocities[joint])) {
            return false;
        }
    }
    return left.foot_contacts == right.foot_contacts;
}

bool transform_bits_equal(const Transform& left, const Transform& right) {
    return vec_bits_equal(left.position, right.position) &&
           quat_bits_equal(left.rotation, right.rotation);
}

bool output_fields_equal(const RuntimeOutput& left, const RuntimeOutput& right) {
    const auto& left_diagnostics = left.diagnostics;
    const auto& right_diagnostics = right.diagnostics;
    if (left.owns_pose != right.owns_pose ||
        left.suppress_steering != right.suppress_steering ||
        !pose_bits_equal(left.pose, right.pose) ||
        !transform_bits_equal(left.object_world, right.object_world) ||
        left_diagnostics.state != right_diagnostics.state ||
        left_diagnostics.result != right_diagnostics.result ||
        left_diagnostics.reason != right_diagnostics.reason ||
        left_diagnostics.target != right_diagnostics.target ||
        left_diagnostics.object_state != right_diagnostics.object_state ||
        left_diagnostics.affordance_id != right_diagnostics.affordance_id ||
        left_diagnostics.clip != right_diagnostics.clip ||
        left_diagnostics.frame != right_diagnostics.frame ||
        left_diagnostics.phase != right_diagnostics.phase ||
        left_diagnostics.hand != right_diagnostics.hand ||
        !float_bits_equal(
            left_diagnostics.total_cost,
            right_diagnostics.total_cost) ||
        left_diagnostics.group_costs != right_diagnostics.group_costs ||
        !float_bits_equal(
            left_diagnostics.requested_root_correction_m,
            right_diagnostics.requested_root_correction_m) ||
        !float_bits_equal(
            left_diagnostics.applied_root_correction_m,
            right_diagnostics.applied_root_correction_m) ||
        !float_bits_equal(
            left_diagnostics.requested_yaw_correction_radians,
            right_diagnostics.requested_yaw_correction_radians) ||
        !float_bits_equal(
            left_diagnostics.applied_yaw_correction_radians,
            right_diagnostics.applied_yaw_correction_radians) ||
        !float_bits_equal(
            left_diagnostics.playback_speed,
            right_diagnostics.playback_speed) ||
        !float_bits_equal(
            left_diagnostics.hand_position_error_m,
            right_diagnostics.hand_position_error_m) ||
        !float_bits_equal(
            left_diagnostics.hand_orientation_error_radians,
            right_diagnostics.hand_orientation_error_radians) ||
        left_diagnostics.attached != right_diagnostics.attached ||
        left_diagnostics.recorded_carry != right_diagnostics.recorded_carry ||
        left_diagnostics.pack_available != right_diagnostics.pack_available) {
        return false;
    }
    return true;
}

bool near(float left, float right, float tolerance = 1.0e-5F) {
    return std::fabs(left - right) <= tolerance;
}

void assert_vec_near(vec3 actual, vec3 expected, float tolerance = 1.0e-5F) {
    assert(near(actual.x, expected.x, tolerance));
    assert(near(actual.y, expected.y, tolerance));
    assert(near(actual.z, expected.z, tolerance));
}

void assert_same_rotation(
    quat actual,
    quat expected,
    float tolerance = 1.0e-5F) {
    const float orientation_dot = std::fabs(quat_dot(actual, expected));
    assert(near(orientation_dot, 1.0F, tolerance));
}

Pose make_pose(float base, bool opposite_quaternion_sign = false) {
    Pose pose;
    for (size_t bone = 0; bone < pose.positions.size(); ++bone) {
        const float offset = static_cast<float>(bone) * 0.01F;
        pose.positions[bone] =
            vec3(base + offset, base + 1.0F + offset, base + 2.0F + offset);
        pose.velocities[bone] =
            vec3(base + 3.0F + offset, base + 4.0F, base + 5.0F - offset);
        const float half_angle = 0.25F + 0.001F * static_cast<float>(bone);
        quat rotation(cosf(half_angle), 0.0F, sinf(half_angle), 0.0F);
        pose.rotations[bone] = opposite_quaternion_sign ? -rotation : rotation;
        pose.angular_velocities[bone] =
            vec3(base + 6.0F, base + 7.0F + offset, base + 8.0F);
    }
    for (size_t joint = 0; joint < pose.hand_dof.size(); ++joint) {
        const float offset = static_cast<float>(joint) * 0.02F;
        pose.hand_dof[joint] = base + 9.0F + offset;
        pose.hand_dof_velocities[joint] = base + 10.0F - offset;
    }
    pose.foot_contacts = {
        static_cast<uint8_t>(base > 0.0F),
        static_cast<uint8_t>(base <= 0.0F)};
    return pose;
}

LocomotionSnapshot make_snapshot(float base) {
    LocomotionSnapshot snapshot;
    snapshot.pose = make_pose(base);
    for (size_t index = 0; index < snapshot.future_root_positions.size(); ++index) {
        const float value = base + static_cast<float>(index);
        snapshot.future_root_positions[index] = vec3(value, 0.0F, -value);
        snapshot.future_root_rotations[index] =
            quat_from_angle_axis(value * 0.1F, vec3(0.0F, 1.0F, 0.0F));
    }
    return snapshot;
}

RuntimeOutput make_complete_output(int serial, RuntimeState state) {
    RuntimeOutput output;
    output.owns_pose = (serial % 2) != 0;
    output.suppress_steering = (serial % 3) != 0;
    output.pose = make_pose(static_cast<float>(serial));
    output.object_world.position =
        vec3(static_cast<float>(serial), 2.0F, -3.0F);
    output.object_world.rotation =
        quat_from_angle_axis(0.2F * static_cast<float>(serial), vec3(0, 1, 0));
    output.diagnostics.state = state;
    output.diagnostics.result = ResultCode::Accepted;
    output.diagnostics.reason = Reason::None;
    output.diagnostics.target = {
        static_cast<uint64_t>(100 + serial),
        static_cast<uint32_t>(10 + serial)};
    output.diagnostics.object_state = ObjectState::Targeted;
    output.diagnostics.affordance_id = static_cast<uint32_t>(20 + serial);
    output.diagnostics.clip = serial;
    output.diagnostics.frame = 30 + serial;
    output.diagnostics.phase = Phase::Reach;
    output.diagnostics.hand = Hand::Left;
    output.diagnostics.total_cost = 1.0F + static_cast<float>(serial);
    output.diagnostics.group_costs = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    output.diagnostics.requested_root_correction_m = 0.11F;
    output.diagnostics.applied_root_correction_m = 0.09F;
    output.diagnostics.requested_yaw_correction_radians = 0.21F;
    output.diagnostics.applied_yaw_correction_radians = 0.19F;
    output.diagnostics.playback_speed = 1.05F;
    output.diagnostics.hand_position_error_m = 0.02F;
    output.diagnostics.hand_orientation_error_radians = 0.03F;
    output.diagnostics.attached = true;
    output.diagnostics.recorded_carry = true;
    output.diagnostics.pack_available = true;
    return output;
}

std::string read_text(const std::string& path) {
    std::ifstream input(path);
    assert(input.good());
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

void test_exact_constants() {
    static_assert(
        interaction::kControllerStepSeconds == 1.0F / 60.0F,
        "controller step must be exact binary32 1/60");
    static_assert(
        interaction::kInteractionRuntimeStepSeconds == 1.0F / 25.0F,
        "runtime step must be exact binary32 1/25");
}

void test_scheduler_cadence_and_cache() {
    ControllerInteractionScheduler scheduler;
    assert(scheduler.phase() == 0);

    int snapshot_calls = 0;
    int resolver_calls = 0;
    int update_calls = 0;
    std::vector<int> due_ticks;
    RuntimeOutput newest{};

    for (int tick = 1; tick <= 60; ++tick) {
        const int calls_before = update_calls;
        const RuntimeOutput before = scheduler.cached_output();
        const RuntimeOutput& observed = scheduler.tick(
            {},
            [&]() {
                ++snapshot_calls;
                return make_snapshot(static_cast<float>(tick));
            },
            [&](const LocomotionSnapshot&) -> std::optional<PickRequest> {
                ++resolver_calls;
                return std::nullopt;
            },
            [&](const RuntimeInput& input) {
                ++update_calls;
                due_ticks.push_back(tick);
                assert(input.dt == 1.0F / 25.0F);
                assert(!input.interact_pressed);
                assert(!input.pick_request.has_value());
                newest = make_complete_output(
                    update_calls,
                    (update_calls % 2) == 0
                        ? RuntimeState::Disabled
                        : RuntimeState::Locomotion);
                return newest;
            });

        assert(update_calls - calls_before <= 1);
        if (update_calls == calls_before) {
            assert(output_fields_equal(observed, before));
        } else {
            assert(output_fields_equal(observed, newest));
        }
    }

    assert(update_calls == 25);
    assert(snapshot_calls == 25);
    assert(resolver_calls == 0);
    assert(scheduler.phase() == 0);
    const std::vector<int> expected_first_block = {3, 5, 8, 10, 12};
    assert(std::vector<int>(due_ticks.begin(), due_ticks.begin() + 5) ==
           expected_first_block);
    for (size_t block = 0; block < 5; ++block) {
        for (size_t index = 0; index < expected_first_block.size(); ++index) {
            assert(due_ticks[block * 5 + index] ==
                   static_cast<int>(block * 12) + expected_first_block[index]);
        }
    }
}

void test_edges_latch_coalesce_and_clear_after_delivery() {
    ControllerInteractionScheduler scheduler;
    int snapshot_calls = 0;
    int resolver_calls = 0;
    int update_calls = 0;
    std::vector<RuntimeInput> delivered;
    const PickRequest request{{42, 7}, 9, 1234};

    const auto snapshot_provider = [&]() {
        ++snapshot_calls;
        return make_snapshot(4.0F);
    };
    const auto resolver = [&](const LocomotionSnapshot& snapshot) {
        ++resolver_calls;
        assert(pose_bits_equal(snapshot.pose, make_pose(4.0F)));
        return std::optional<PickRequest>{request};
    };
    const auto update = [&](const RuntimeInput& input) {
        ++update_calls;
        delivered.push_back(input);
        return make_complete_output(update_calls, RuntimeState::Locomotion);
    };

    scheduler.tick({true, false, false}, snapshot_provider, resolver, update);
    scheduler.tick({true, true, false}, snapshot_provider, resolver, update);
    assert(snapshot_calls == 0);
    assert(resolver_calls == 0);
    assert(update_calls == 0);

    scheduler.tick({false, false, true}, snapshot_provider, resolver, update);
    assert(snapshot_calls == 1);
    assert(resolver_calls == 1);
    assert(update_calls == 1);
    assert(delivered[0].interact_pressed);
    assert(delivered[0].cancel_pressed);
    assert(delivered[0].reset_pressed);
    assert(delivered[0].pick_request.has_value());
    assert(delivered[0].pick_request->target == request.target);
    assert(delivered[0].pick_request->affordance_id == request.affordance_id);
    assert(delivered[0].pick_request->request_id == request.request_id);

    scheduler.tick({}, snapshot_provider, resolver, update);
    scheduler.tick({}, snapshot_provider, resolver, update);
    assert(snapshot_calls == 2);
    assert(resolver_calls == 1);
    assert(update_calls == 2);
    assert(!delivered[1].interact_pressed);
    assert(!delivered[1].cancel_pressed);
    assert(!delivered[1].reset_pressed);
    assert(!delivered[1].pick_request.has_value());

    ControllerInteractionScheduler due_edge_scheduler;
    due_edge_scheduler.tick({}, snapshot_provider, resolver, update);
    due_edge_scheduler.tick({}, snapshot_provider, resolver, update);
    due_edge_scheduler.tick(
        {true, true, true}, snapshot_provider, resolver, update);
    const RuntimeInput& due_edge_input = delivered.back();
    assert(due_edge_input.interact_pressed);
    assert(due_edge_input.cancel_pressed);
    assert(due_edge_input.reset_pressed);
}

void test_cache_changes_only_after_successful_due_delivery() {
    ControllerInteractionScheduler scheduler;
    const auto snapshot_provider = [] { return make_snapshot(2.0F); };
    const auto resolver = [](const LocomotionSnapshot&) {
        return std::optional<PickRequest>{PickRequest{{8, 2}, 5, 99}};
    };
    int calls = 0;

    scheduler.tick({true, false, false}, snapshot_provider, resolver,
                   [&](const RuntimeInput&) {
                       ++calls;
                       return make_complete_output(1, RuntimeState::Locomotion);
                   });
    scheduler.tick({}, snapshot_provider, resolver,
                   [&](const RuntimeInput&) {
                       ++calls;
                       return make_complete_output(1, RuntimeState::Locomotion);
                   });
    assert(calls == 0);
    const RuntimeOutput initial{};
    assert(output_fields_equal(scheduler.cached_output(), initial));

    bool threw = false;
    try {
        scheduler.tick({}, snapshot_provider, resolver,
                       [&](const RuntimeInput& input) -> RuntimeOutput {
                           ++calls;
                           assert(input.interact_pressed);
                           throw std::runtime_error("delivery failed");
                       });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    assert(calls == 1);
    assert(output_fields_equal(scheduler.cached_output(), initial));

    scheduler.tick({}, snapshot_provider, resolver,
                   [&](const RuntimeInput&) {
                       ++calls;
                       return make_complete_output(2, RuntimeState::Disabled);
                   });
    const RuntimeOutput held = scheduler.cached_output();
    scheduler.tick({}, snapshot_provider, resolver,
                   [&](const RuntimeInput& input) {
                       ++calls;
                       assert(input.interact_pressed);
                       return make_complete_output(3, RuntimeState::Locomotion);
                   });
    assert(calls == 2);
    assert(!output_fields_equal(scheduler.cached_output(), held));
}

void test_adapter_blends_every_pose_channel_and_contacts() {
    ControllerInteractionAdapter adapter;
    const Pose locomotion = make_pose(-2.0F);
    RuntimeOutput idle;
    const Pose idle_output =
        adapter.apply(locomotion, idle, interaction::kControllerStepSeconds);
    assert(pose_bits_equal(idle_output, locomotion));

    RuntimeOutput owned;
    owned.owns_pose = true;
    owned.pose = make_pose(4.0F);
    const Pose first =
        adapter.apply(locomotion, owned, interaction::kControllerStepSeconds);
    assert(!pose_bits_equal(first, locomotion));
    assert(!pose_bits_equal(first, owned.pose));

    const float alpha = interaction::kControllerStepSeconds / 0.25F;
    assert_vec_near(
        first.positions[5],
        lerp(locomotion.positions[5], owned.pose.positions[5], alpha));
    assert_vec_near(
        first.velocities[6],
        lerp(locomotion.velocities[6], owned.pose.velocities[6], alpha));
    assert_same_rotation(
        first.rotations[7],
        quat_nlerp_shortest(
            locomotion.rotations[7], owned.pose.rotations[7], alpha));
    assert_vec_near(
        first.angular_velocities[8],
        lerp(
            locomotion.angular_velocities[8],
            owned.pose.angular_velocities[8],
            alpha));
    assert(near(
        first.hand_dof[4],
        lerpf(locomotion.hand_dof[4], owned.pose.hand_dof[4], alpha)));
    assert(near(
        first.hand_dof_velocities[9],
        lerpf(
            locomotion.hand_dof_velocities[9],
            owned.pose.hand_dof_velocities[9],
            alpha)));

    Pose blended = first;
    for (int tick = 2; tick <= 7; ++tick) {
        blended =
            adapter.apply(locomotion, owned, interaction::kControllerStepSeconds);
        assert(blended.foot_contacts == locomotion.foot_contacts);
    }
    blended = adapter.apply(
        locomotion, owned, interaction::kControllerStepSeconds);
    assert(blended.foot_contacts == owned.pose.foot_contacts);

    for (int tick = 9; tick <= 15; ++tick) {
        blended =
            adapter.apply(locomotion, owned, interaction::kControllerStepSeconds);
    }
    assert(pose_bits_equal(blended, owned.pose));
    const Pose after_settlement =
        adapter.apply(locomotion, owned, interaction::kControllerStepSeconds);
    assert(pose_bits_equal(after_settlement, owned.pose));
}

void test_adapter_shortest_antipodal_rotation() {
    ControllerInteractionAdapter adapter;
    Pose source{};
    Pose target{};
    const quat positive_target =
        quat_from_angle_axis(0.75F, vec3(0.0F, 1.0F, 0.0F));
    target.rotations.fill(-positive_target);
    RuntimeOutput owned;
    owned.owns_pose = true;
    owned.pose = target;

    const Pose first =
        adapter.apply(source, owned, interaction::kControllerStepSeconds);
    for (quat rotation : first.rotations) {
        assert(near(quat_length(rotation), 1.0F, 2.0e-5F));
        assert(rotation.w > 0.0F);
        assert(rotation.y > 0.0F);
    }
}

void test_adapter_exit_reentry_and_reset_capture_fresh_locomotion() {
    ControllerInteractionAdapter adapter;
    RuntimeOutput owned;
    owned.owns_pose = true;
    owned.pose = make_pose(9.0F);

    const Pose first_source = make_pose(1.0F);
    (void)adapter.apply(
        first_source, owned, interaction::kControllerStepSeconds);

    RuntimeOutput idle;
    const Pose current_locomotion = make_pose(3.0F);
    const Pose exited = adapter.apply(
        current_locomotion, idle, interaction::kControllerStepSeconds);
    assert(pose_bits_equal(exited, current_locomotion));

    const Pose reentered = adapter.apply(
        current_locomotion, owned, interaction::kControllerStepSeconds);
    const float alpha = interaction::kControllerStepSeconds / 0.25F;
    assert_vec_near(
        reentered.positions[0],
        lerp(
            current_locomotion.positions[0],
            owned.pose.positions[0],
            alpha));

    adapter.reset();
    const Pose reset_source = make_pose(5.0F);
    const Pose after_reset = adapter.apply(
        reset_source, owned, interaction::kControllerStepSeconds);
    assert_vec_near(
        after_reset.positions[0],
        lerp(reset_source.positions[0], owned.pose.positions[0], alpha));
}

void test_frame_handoff_preserves_fresh_complete_nonowned_pose_for_60_frames() {
    ControllerInteractionFrameHandoff disabled_handoff;
    ControllerInteractionFrameHandoff loaded_handoff;
    ControllerInteractionScheduler disabled_scheduler;
    ControllerInteractionScheduler loaded_scheduler;
    interaction::InteractionRuntime disabled_runtime =
        interaction::InteractionRuntime::disabled(
            Reason::PackUnavailable);
    interaction::RuntimeFixture loaded_fixture =
        interaction::make_runtime_fixture();
    interaction::InteractionRuntime loaded_runtime(
        loaded_fixture.database,
        loaded_fixture.features,
        loaded_fixture.registry,
        interaction::RuntimeConfig{});
    int disabled_runtime_calls = 0;
    int loaded_runtime_calls = 0;
    Pose previous{};

    for (int tick = 1; tick <= 60; ++tick) {
        const Pose fresh = make_pose(20.0F + static_cast<float>(tick));
        if (tick != 1) {
            assert(!pose_bits_equal(fresh, previous));
        }
        previous = fresh;

        const RuntimeOutput& disabled_output = disabled_scheduler.tick(
            {},
            [&] { return make_snapshot(static_cast<float>(tick)); },
            [](const LocomotionSnapshot&) -> std::optional<PickRequest> {
                return std::nullopt;
            },
            [&](const RuntimeInput& input) {
                ++disabled_runtime_calls;
                assert(input.dt == interaction::kInteractionRuntimeStepSeconds);
                return disabled_runtime.update(input);
            });
        const RuntimeOutput& loaded_output = loaded_scheduler.tick(
            {},
            [&] { return make_snapshot(static_cast<float>(tick)); },
            [](const LocomotionSnapshot&) -> std::optional<PickRequest> {
                return std::nullopt;
            },
            [&](const RuntimeInput& input) {
                ++loaded_runtime_calls;
                assert(input.dt == interaction::kInteractionRuntimeStepSeconds);
                const RuntimeOutput output = loaded_runtime.update(input);
                assert(output.diagnostics.state == RuntimeState::Locomotion);
                assert(output.diagnostics.pack_available);
                assert(!output.owns_pose);
                return output;
            });
        const ControllerInteractionFrameState disabled_frame =
            disabled_handoff.apply(
                fresh,
                disabled_output,
                interaction::kControllerStepSeconds);
        const ControllerInteractionFrameState loaded_frame =
            loaded_handoff.apply(
                fresh,
                loaded_output,
                interaction::kControllerStepSeconds);
        assert(pose_bits_equal(disabled_frame.pose, fresh));
        assert(pose_bits_equal(loaded_frame.pose, fresh));
        assert(!disabled_frame.owns_pose);
        assert(!loaded_frame.owns_pose);
        assert(!disabled_frame.synchronize_simulation_root);
        assert(!loaded_frame.synchronize_simulation_root);
    }

    assert(disabled_runtime_calls == 25);
    assert(loaded_runtime_calls == 25);
    assert(disabled_scheduler.phase() == 0);
    assert(loaded_scheduler.phase() == 0);
}

void test_frame_handoff_retains_channels_and_exposes_root_sync() {
    for (float value : interaction::kFlatControllerRestHandDof) {
        assert(float_bits_equal(value, 0.0F));
    }
    for (float value : interaction::kFlatControllerRestHandDofVelocities) {
        assert(float_bits_equal(value, 0.0F));
    }

    ControllerInteractionFrameHandoff handoff;
    Pose locomotion = make_pose(-4.0F);
    locomotion.hand_dof = interaction::kFlatControllerRestHandDof;
    locomotion.hand_dof_velocities =
        interaction::kFlatControllerRestHandDofVelocities;
    RuntimeOutput owned;
    owned.owns_pose = true;
    owned.pose = make_pose(8.0F, true);

    ControllerInteractionFrameState frame{};
    for (int tick = 1; tick <= 15; ++tick) {
        frame = handoff.apply(
            locomotion, owned, interaction::kControllerStepSeconds);
        assert(frame.owns_pose);
        assert(frame.synchronize_simulation_root);
        assert(vec_bits_equal(
            frame.simulation_root_position, frame.pose.positions[0]));
        assert(quat_bits_equal(
            frame.simulation_root_rotation, frame.pose.rotations[0]));
    }
    assert(pose_bits_equal(frame.pose, owned.pose));
    assert(frame.pose.hand_dof == owned.pose.hand_dof);
    assert(frame.pose.hand_dof_velocities ==
           owned.pose.hand_dof_velocities);
    assert(frame.pose.foot_contacts == owned.pose.foot_contacts);

    vec3 simulation_position(91.0F, 92.0F, 93.0F);
    quat simulation_rotation = quat_from_angle_axis(
        0.3F, vec3(0.0F, 1.0F, 0.0F));
    if (frame.synchronize_simulation_root) {
        simulation_position = frame.simulation_root_position;
        simulation_rotation = frame.simulation_root_rotation;
    }
    assert(vec_bits_equal(simulation_position, frame.pose.positions[0]));
    assert(quat_bits_equal(simulation_rotation, frame.pose.rotations[0]));

    const Pose fresh_nonowned = make_pose(31.0F);
    RuntimeOutput nonowned;
    const ControllerInteractionFrameState released = handoff.apply(
        fresh_nonowned, nonowned, interaction::kControllerStepSeconds);
    assert(pose_bits_equal(released.pose, fresh_nonowned));
    assert(!released.synchronize_simulation_root);
}

void test_scene_handoff_retains_attached_pose_until_registry_reclaims_authority() {
    ControllerInteractionSceneHandoff handoff;
    InteractionTarget target;
    target.handle = {41, 3};
    target.state = ObjectState::Attached;
    target.object_world = {
        vec3(2.0F, 0.8F, 3.0F),
        quat_from_angle_axis(0.2F, vec3(0.0F, 1.0F, 0.0F))};
    const Transform authored_fallback = target.object_world;

    RuntimeOutput attached;
    attached.diagnostics.target = target.handle;
    attached.diagnostics.attached = true;
    attached.object_world = {
        vec3(2.5F, 1.2F, 3.4F),
        quat_from_angle_axis(0.7F, vec3(0.0F, 1.0F, 0.0F))};
    const ControllerInteractionSceneState attached_scene = handoff.apply(
        &target, attached, authored_fallback);
    assert(attached_scene.runtime_authority);
    assert(transform_bits_equal(
        attached_scene.object_world, attached.object_world));

    target.state = ObjectState::Held;
    RuntimeOutput post_failure = attached;
    post_failure.diagnostics.attached = false;
    post_failure.object_world = {
        vec3(2.8F, 1.4F, 3.7F),
        quat_from_angle_axis(0.9F, vec3(0.0F, 1.0F, 0.0F))};
    const ControllerInteractionSceneState held_scene = handoff.apply(
        &target, post_failure, authored_fallback);
    assert(held_scene.runtime_authority);
    assert(transform_bits_equal(
        held_scene.object_world, post_failure.object_world));

    InteractionTarget replacement = target;
    replacement.handle = {41, 9};
    replacement.state = ObjectState::Free;
    replacement.object_world = {
        vec3(-3.0F, 0.9F, 4.0F),
        quat_from_angle_axis(-0.5F, vec3(0.0F, 1.0F, 0.0F))};
    const ControllerInteractionSceneState free_scene = handoff.apply(
        &replacement, post_failure, authored_fallback);
    assert(!free_scene.runtime_authority);
    assert(transform_bits_equal(
        free_scene.object_world, replacement.object_world));

    replacement.state = ObjectState::Targeted;
    replacement.object_world.position.x += 0.25F;
    const ControllerInteractionSceneState targeted_scene = handoff.apply(
        &replacement, post_failure, authored_fallback);
    assert(!targeted_scene.runtime_authority);
    assert(transform_bits_equal(
        targeted_scene.object_world, replacement.object_world));
}

void test_carry_label_is_only_specific_during_carry() {
    RuntimeOutput output;
    output.diagnostics.recorded_carry = true;
    output.diagnostics.state = RuntimeState::Hold;
    assert(std::string(interaction::controller_carry_mode_label(output)) ==
           "none");
    output.diagnostics.state = RuntimeState::Carry;
    assert(std::string(interaction::controller_carry_mode_label(output)) ==
           "recorded");
    output.diagnostics.recorded_carry = false;
    assert(std::string(interaction::controller_carry_mode_label(output)) ==
           "layered");
}

Transform transform_from_arrays(
    const std::vector<float>& positions,
    const std::vector<float>& rotations,
    size_t index) {
    return {
        vec3(
            positions[index * 3],
            positions[index * 3 + 1],
            positions[index * 3 + 2]),
        quat(
            rotations[index * 4],
            rotations[index * 4 + 1],
            rotations[index * 4 + 2],
            rotations[index * 4 + 3])};
}

void test_demo_target_preserves_object_in_table_transform() {
    Database database;
    database.clip_count = 1;
    database.frame_count = 4;
    database.range_starts = {0};
    database.range_stops = {4};
    database.phases = {
        static_cast<uint8_t>(Phase::Approach),
        static_cast<uint8_t>(Phase::Reach),
        static_cast<uint8_t>(Phase::Contact),
        static_cast<uint8_t>(Phase::Lift)};
    database.active_hands = {static_cast<uint8_t>(Hand::Left)};

    const Transform source_table{
        vec3(2.0F, 0.8F, -1.0F),
        quat_from_angle_axis(0.65F, vec3(0.0F, 1.0F, 0.0F))};
    const Transform object_in_table{
        vec3(0.35F, 0.45F, -0.22F),
        quat_from_angle_axis(-0.4F, vec3(0.0F, 1.0F, 0.0F))};
    const Transform source_object = compose(source_table, object_in_table);

    database.table_positions = {
        source_table.position.x,
        source_table.position.y,
        source_table.position.z};
    database.table_rotations = {
        source_table.rotation.w,
        source_table.rotation.x,
        source_table.rotation.y,
        source_table.rotation.z};
    database.table_sizes = {1.4F, 0.1F, 0.8F};
    database.object_dimensions = {0.12F, 0.25F, 0.16F};
    database.grasp_positions_object = {0.01F, 0.04F, -0.02F};
    const quat grasp_rotation =
        quat_from_angle_axis(0.2F, vec3(1.0F, 0.0F, 0.0F));
    database.grasp_rotations_object = {
        grasp_rotation.w,
        grasp_rotation.x,
        grasp_rotation.y,
        grasp_rotation.z};
    database.approach_directions_object = {0.0F, 0.0F, 1.0F};
    database.object_positions.resize(12, 0.0F);
    database.object_rotations.resize(16, 0.0F);
    for (size_t frame = 0; frame < 4; ++frame) {
        const Transform object = frame == 1
            ? source_object
            : Transform{vec3(20.0F + static_cast<float>(frame), 0, 0), quat()};
        database.object_positions[frame * 3] = object.position.x;
        database.object_positions[frame * 3 + 1] = object.position.y;
        database.object_positions[frame * 3 + 2] = object.position.z;
        database.object_rotations[frame * 4] = object.rotation.w;
        database.object_rotations[frame * 4 + 1] = object.rotation.x;
        database.object_rotations[frame * 4 + 2] = object.rotation.y;
        database.object_rotations[frame * 4 + 3] = object.rotation.z;
    }

    const InteractionTarget target =
        interaction::make_controller_demo_target(database);
    assert(target.handle.id != 0U);
    assert(target.handle.generation != 0U);
    assert(near(target.table_world.position.x, 0.0F));
    assert(near(target.table_world.position.y, source_table.position.y));
    assert(near(target.table_world.position.z, 3.0F));
    assert_same_rotation(target.table_world.rotation, source_table.rotation);
    const Transform relocated_local =
        compose(inverse(target.table_world), target.object_world);
    assert_vec_near(relocated_local.position, object_in_table.position);
    assert_same_rotation(relocated_local.rotation, object_in_table.rotation);
    assert(target.affordances.size() == 1);
    assert(target.affordances[0].hand == Hand::Left);
    assert(target.state == ObjectState::Free);

    const Transform selected_source = transform_from_arrays(
        database.object_positions, database.object_rotations, 1);
    assert_vec_near(selected_source.position, source_object.position);
}

void test_cross_pack_frame_count_validation() {
    Database database;
    Features features;
    database.frame_count = 25;
    features.frame_count = 24;
    bool threw = false;
    try {
        interaction::validate_controller_interaction_pack(database, features);
    } catch (const interaction::FormatError& error) {
        threw = true;
        assert(std::string(error.what()).find("frame count") !=
               std::string::npos);
    }
    assert(threw);
    features.frame_count = database.frame_count;
    interaction::validate_controller_interaction_pack(database, features);
}

void test_debug_draw_uses_real_correction_geometry_and_complete_text() {
    const std::string debug = read_text("interaction_debug_draw.h");
    assert(debug.find("requested_root_correction_m * right") ==
           std::string::npos);
    assert(debug.find(
               "displayed_pose.positions[0] - locomotion_pose.positions[0]") !=
           std::string::npos);
    assert(debug.find("target=%llu:%u affordance=%u") != std::string::npos);
    assert(debug.find("groups=[%.3f %.3f %.3f %.3f %.3f]") !=
           std::string::npos);
    assert(debug.find("root req/app=%.3f/%.3f") != std::string::npos);
    assert(debug.find("yaw req/app=%.3f/%.3f") != std::string::npos);
    assert(debug.find("hand pos/orient=%.3f/%.3f") != std::string::npos);
    assert(debug.find("object=%s carry=%s") != std::string::npos);
    assert(debug.find("controller_carry_mode_label(output)") !=
           std::string::npos);
}

void test_controller_and_make_clock_policy() {
    const std::string controller = read_text("controller.cpp");
    assert(controller.find("SetTargetFPS(60);") != std::string::npos);
    assert(controller.find(
               "const float dt = interaction::kControllerStepSeconds;") !=
           std::string::npos);
    assert(controller.find("ControllerInteractionScheduler") !=
           std::string::npos);
    assert(controller.find("interaction_scheduler.tick(") != std::string::npos);
    assert(controller.find("interaction_frame_handoff.apply(") !=
           std::string::npos);
    assert(controller.find("interaction_scene_handoff.apply(") !=
           std::string::npos);
    assert(controller.find("interaction_registry.find_by_id(") !=
           std::string::npos);
    assert(controller.find("++next_handle.generation") == std::string::npos);
    assert(controller.find("interaction::kControllerStepSeconds") !=
           std::string::npos);
    assert(controller.find("GetFrameTime(") == std::string::npos);
    assert(controller.find("runtime_accumulator") == std::string::npos);
    assert(controller.find("runtime_output_alpha") == std::string::npos);
    assert(controller.find(
               "emscripten_set_main_loop_arg(update_callback, &u, 60, 1);") !=
           std::string::npos);
    assert(controller.find(
               "emscripten_set_main_loop_arg(update_callback, &u, 0, 1);") ==
           std::string::npos);
    assert(controller.find("InteractionRuntime::disabled(") !=
           std::string::npos);
    assert(controller.find("catch (const interaction::FormatError&") !=
           std::string::npos);

    const std::string makefile = read_text("Makefile");
    const size_t sources_begin = makefile.find("INTERACTION_SOURCES :=");
    const size_t sources_end = makefile.find("HEADER =", sources_begin);
    assert(sources_begin != std::string::npos);
    assert(sources_end != std::string::npos);
    const std::string controller_sources =
        makefile.substr(sources_begin, sources_end - sources_begin);
    assert(controller_sources.find("interaction_runtime.cpp") !=
           std::string::npos);
    assert(controller_sources.find("interaction_controller_adapter.cpp") !=
           std::string::npos);
    assert(controller_sources.find("interaction_pose.cpp") !=
           std::string::npos);
    assert(controller_sources.find("interaction_target.cpp") !=
           std::string::npos);
    assert(controller_sources.find("_probe.cpp") == std::string::npos);
    assert(controller_sources.find("SOURCE := controller.cpp") !=
           std::string::npos);
    assert(makefile.find("CONTROLLER_CXXFLAGS := -std=c++17") !=
           std::string::npos);
}

}  // namespace

int main() {
    test_exact_constants();
    test_scheduler_cadence_and_cache();
    test_edges_latch_coalesce_and_clear_after_delivery();
    test_cache_changes_only_after_successful_due_delivery();
    test_adapter_blends_every_pose_channel_and_contacts();
    test_adapter_shortest_antipodal_rotation();
    test_adapter_exit_reentry_and_reset_capture_fresh_locomotion();
    test_frame_handoff_preserves_fresh_complete_nonowned_pose_for_60_frames();
    test_frame_handoff_retains_channels_and_exposes_root_sync();
    test_scene_handoff_retains_attached_pose_until_registry_reclaims_authority();
    test_carry_label_is_only_specific_during_carry();
    test_demo_target_preserves_object_in_table_transform();
    test_cross_pack_frame_count_validation();
    test_debug_draw_uses_real_correction_geometry_and_complete_text();
    test_controller_and_make_clock_policy();
    return 0;
}
