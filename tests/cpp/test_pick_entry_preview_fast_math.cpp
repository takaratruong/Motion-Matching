#ifndef NDEBUG
#error "Pickup preview fast-math canary requires NDEBUG"
#endif

#ifndef __FAST_MATH__
#error "Pickup preview fast-math canary requires -ffast-math"
#endif

#include "interaction_runtime.h"
#include "tests/cpp/interaction_runtime_fixture.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

uint32_t float_bits(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float float_from_bits(uint32_t bits) {
    float value = 0.0F;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool same_bits(float left, float right) {
    return float_bits(left) == float_bits(right);
}

bool same_bits(vec3 left, vec3 right) {
    return same_bits(left.x, right.x) && same_bits(left.y, right.y) &&
           same_bits(left.z, right.z);
}

bool same_bits(quat left, quat right) {
    return same_bits(left.w, right.w) && same_bits(left.x, right.x) &&
           same_bits(left.y, right.y) && same_bits(left.z, right.z);
}

bool same_bits(
    const interaction::LocomotionSnapshot& left,
    const interaction::LocomotionSnapshot& right) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        if (!same_bits(left.pose.positions[bone], right.pose.positions[bone]) ||
            !same_bits(left.pose.velocities[bone], right.pose.velocities[bone]) ||
            !same_bits(left.pose.rotations[bone], right.pose.rotations[bone]) ||
            !same_bits(
                left.pose.angular_velocities[bone],
                right.pose.angular_velocities[bone])) {
            return false;
        }
    }
    for (size_t joint = 0; joint < left.pose.hand_dof.size(); ++joint) {
        if (!same_bits(left.pose.hand_dof[joint], right.pose.hand_dof[joint]) ||
            !same_bits(
                left.pose.hand_dof_velocities[joint],
                right.pose.hand_dof_velocities[joint])) {
            return false;
        }
    }
    if (left.pose.foot_contacts != right.pose.foot_contacts) return false;
    for (size_t index = 0; index < left.future_root_positions.size(); ++index) {
        if (!same_bits(
                left.future_root_positions[index],
                right.future_root_positions[index]) ||
            !same_bits(
                left.future_root_rotations[index],
                right.future_root_rotations[index])) {
            return false;
        }
    }
    return true;
}

bool same_bits(interaction::Transform left, interaction::Transform right) {
    return same_bits(left.position, right.position) &&
           same_bits(left.rotation, right.rotation);
}

bool same_target(
    const interaction::InteractionTarget& left,
    const interaction::InteractionTarget& right) {
    if (left.handle != right.handle ||
        !same_bits(left.object_world, right.object_world) ||
        left.object_profile_id != right.object_profile_id ||
        !same_bits(
            left.object_bounds.center_object,
            right.object_bounds.center_object) ||
        !same_bits(
            left.object_bounds.half_extents_object,
            right.object_bounds.half_extents_object) ||
        !same_bits(left.object_dimensions, right.object_dimensions) ||
        !same_bits(left.table_world, right.table_world) ||
        !same_bits(left.table_size, right.table_size) ||
        left.state != right.state ||
        left.owner_request != right.owner_request ||
        left.affordances.size() != right.affordances.size()) {
        return false;
    }
    for (size_t index = 0; index < left.affordances.size(); ++index) {
        const interaction::GraspAffordance& a = left.affordances[index];
        const interaction::GraspAffordance& b = right.affordances[index];
        if (a.id != b.id || a.hand != b.hand ||
            !same_bits(a.hand_in_object, b.hand_in_object) ||
            !same_bits(
                a.approach_direction_object,
                b.approach_direction_object) ||
            !same_bits(a.clearance_radius, b.clearance_radius)) {
            return false;
        }
    }
    return true;
}

bool same_candidate(
    const interaction::MatchCandidate& left,
    const interaction::MatchCandidate& right) {
    if (left.clip != right.clip ||
        left.entry_frame != right.entry_frame ||
        left.contact_frame != right.contact_frame ||
        left.lift_frame != right.lift_frame ||
        left.hold_frame != right.hold_frame ||
        !same_bits(left.scene_from_source, right.scene_from_source) ||
        !same_bits(left.entry_root_offset, right.entry_root_offset) ||
        !same_bits(left.entry_yaw_offset, right.entry_yaw_offset) ||
        !same_bits(left.total_cost, right.total_cost)) {
        return false;
    }
    for (size_t group = 0; group < left.group_costs.size(); ++group) {
        if (!same_bits(left.group_costs[group], right.group_costs[group])) {
            return false;
        }
    }
    return true;
}

bool same_preview(
    const interaction::PickEntryPreview& left,
    const interaction::PickEntryPreview& right) {
    return left.path_feasible == right.path_feasible &&
        left.match_ready == right.match_ready &&
        left.path_reason == right.path_reason &&
        left.match_reason == right.match_reason &&
        same_bits(left.prospective_root.world_x, right.prospective_root.world_x) &&
        same_bits(left.prospective_root.world_z, right.prospective_root.world_z) &&
        same_bits(
            left.prospective_root.world_yaw_radians,
            right.prospective_root.world_yaw_radians) &&
        left.feasible_entry_frame == right.feasible_entry_frame &&
        left.contact_frame == right.contact_frame &&
        same_bits(left.total_cost, right.total_cost) &&
        same_candidate(left.match_candidate, right.match_candidate);
}

struct Observation {
    interaction::RuntimeState state{};
    std::array<unsigned char, sizeof(interaction::RuntimeDiagnostics)>
        diagnostics{};
    interaction::InteractionTarget target{};
    interaction::LocomotionSnapshot caller{};
};

Observation capture_observation(
    const interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture,
    const interaction::LocomotionSnapshot& caller) {
    Observation result{};
    result.state = runtime.state();
    std::memcpy(
        result.diagnostics.data(),
        &runtime.diagnostics(),
        result.diagnostics.size());
    const interaction::InteractionTarget* target =
        fixture.registry.find(fixture.request.target);
    require(target != nullptr, "preview fixture target disappeared");
    result.target = *target;
    result.caller = caller;
    return result;
}

bool same_observation(const Observation& left, const Observation& right) {
    return left.state == right.state &&
        left.diagnostics == right.diagnostics &&
        same_target(left.target, right.target) &&
        same_bits(left.caller, right.caller);
}

std::vector<float*> snapshot_float_fields(
    interaction::LocomotionSnapshot& snapshot) {
    std::vector<float*> fields;
    const auto append_vec3 = [&](vec3& value) {
        fields.push_back(&value.x);
        fields.push_back(&value.y);
        fields.push_back(&value.z);
    };
    const auto append_quat = [&](quat& value) {
        fields.push_back(&value.w);
        fields.push_back(&value.x);
        fields.push_back(&value.y);
        fields.push_back(&value.z);
    };
    for (vec3& value : snapshot.pose.positions) append_vec3(value);
    for (vec3& value : snapshot.pose.velocities) append_vec3(value);
    for (quat& value : snapshot.pose.rotations) append_quat(value);
    for (vec3& value : snapshot.pose.angular_velocities) append_vec3(value);
    for (float& value : snapshot.pose.hand_dof) fields.push_back(&value);
    for (float& value : snapshot.pose.hand_dof_velocities) {
        fields.push_back(&value);
    }
    for (vec3& value : snapshot.future_root_positions) append_vec3(value);
    for (quat& value : snapshot.future_root_rotations) append_quat(value);
    return fields;
}

float yaw_radians(quat rotation) {
    rotation = rotation / quat_length(rotation);
    const vec3 forward = quat_mul_vec3(
        rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(forward.x, forward.z);
}

float shortest_angle(float angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

interaction::PickEntryRoot live_root(
    const interaction::LocomotionSnapshot& snapshot) {
    const size_t root = static_cast<size_t>(g1_skeleton::Simulation);
    return {
        snapshot.pose.positions[root].x,
        snapshot.pose.positions[root].z,
        yaw_radians(snapshot.pose.rotations[root]),
    };
}

bool near(float left, float right, float tolerance = 2.0e-4F) {
    return std::fabs(left - right) <= tolerance;
}

bool near(vec3 left, vec3 right, float tolerance = 2.0e-4F) {
    return near(left.x, right.x, tolerance) &&
           near(left.y, right.y, tolerance) &&
           near(left.z, right.z, tolerance);
}

void require_out_of_range(
    const interaction::PickEntryPreview& preview,
    interaction::PickEntryRoot requested) {
    require(!preview.path_feasible, "invalid preview became path feasible");
    require(!preview.match_ready, "invalid preview became match ready");
    require(
        preview.path_reason == interaction::Reason::OutOfRange,
        "invalid preview lost path OutOfRange");
    require(
        preview.match_reason == interaction::Reason::OutOfRange,
        "invalid preview lost match OutOfRange");
    require(
        same_bits(preview.prospective_root.world_x, requested.world_x) &&
        same_bits(preview.prospective_root.world_z, requested.world_z) &&
        same_bits(
            preview.prospective_root.world_yaw_radians,
            requested.world_yaw_radians),
        "invalid preview changed prospective root bits");
    require(
        preview.feasible_entry_frame == -1 && preview.contact_frame == -1,
        "invalid preview populated frames");
    require(
        same_bits(preview.total_cost, 0.0F),
        "invalid preview populated cost");
    require(
        same_candidate(preview.match_candidate, interaction::MatchCandidate{}),
        "invalid preview populated candidate");
}

void test_nontrivial_rigid_map() {
    using namespace interaction;
    constexpr size_t root = static_cast<size_t>(g1_skeleton::Simulation);
    const LocomotionSnapshot input = make_pick_snapshot_fixture();
    const LocomotionSnapshot before = input;
    const PickEntryRoot requested{2.25F, -1.75F, 0.70F};
    const float delta_yaw = shortest_angle(
        requested.world_yaw_radians - yaw_radians(input.pose.rotations[root]));
    const quat delta = quat_from_angle_axis(
        delta_yaw, vec3(0.0F, 1.0F, 0.0F));
    const runtime_detail::PickSnapshotMap mapped =
        runtime_detail::map_pick_entry_snapshot(input, requested);

    require(mapped.accepted, "nontrivial rigid map was rejected");
    require(mapped.reason == Reason::None, "accepted rigid map has reason");
    require(same_bits(input, before), "rigid map mutated its caller snapshot");
    require(
        same_bits(mapped.snapshot.pose.positions[root].x, requested.world_x) &&
        same_bits(mapped.snapshot.pose.positions[root].y, input.pose.positions[root].y) &&
        same_bits(mapped.snapshot.pose.positions[root].z, requested.world_z),
        "rigid map did not install requested planar root");
    require(
        near(shortest_angle(
            yaw_radians(mapped.snapshot.pose.rotations[root]) -
            requested.world_yaw_radians), 0.0F),
        "rigid map did not install requested yaw");
    require(
        near(
            mapped.snapshot.pose.velocities[root],
            quat_mul_vec3(delta, input.pose.velocities[root])),
        "rigid map did not rotate root velocity");
    const vec3 mapped_root(
        requested.world_x,
        input.pose.positions[root].y,
        requested.world_z);
    for (size_t index = 0; index < input.future_root_positions.size(); ++index) {
        require(
            near(
                mapped.snapshot.future_root_positions[index],
                mapped_root + quat_mul_vec3(
                    delta,
                    input.future_root_positions[index] -
                        input.pose.positions[root])),
            "rigid map did not transform future root position");
    }
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        if (bone == root) continue;
        require(
            same_bits(
                mapped.snapshot.pose.positions[bone],
                input.pose.positions[bone]) &&
            same_bits(
                mapped.snapshot.pose.velocities[bone],
                input.pose.velocities[bone]) &&
            same_bits(
                mapped.snapshot.pose.rotations[bone],
                input.pose.rotations[bone]) &&
            same_bits(
                mapped.snapshot.pose.angular_velocities[bone],
                input.pose.angular_velocities[bone]),
            "rigid map changed a child-local channel");
    }
}

void test_bit_pattern_nonfinite_rejection() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});
    const LocomotionSnapshot baseline = fixture.locomotion;
    const PickEntryRoot valid_root = live_root(baseline);
    constexpr std::array<uint32_t, 4> invalid_bits{
        0x7f800000U,
        0xff800000U,
        0x7fc00001U,
        0x7f800001U,
    };

    LocomotionSnapshot field_source = baseline;
    const size_t field_count = snapshot_float_fields(field_source).size();
    for (size_t field = 0; field < field_count; ++field) {
        for (uint32_t bits : invalid_bits) {
            LocomotionSnapshot invalid = baseline;
            *snapshot_float_fields(invalid).at(field) = float_from_bits(bits);
            const Observation before = capture_observation(
                runtime, fixture, invalid);
            const PickEntryPreview preview = runtime.preview_pick(
                invalid,
                valid_root,
                fixture.request.target,
                fixture.request.affordance_id);
            require_out_of_range(preview, valid_root);
            require(
                same_observation(
                    before,
                    capture_observation(runtime, fixture, invalid)),
                "invalid snapshot preview mutated runtime or registry");
        }
    }

    for (size_t field = 0; field < 3U; ++field) {
        for (uint32_t bits : invalid_bits) {
            PickEntryRoot invalid = valid_root;
            std::array<float*, 3> fields{
                &invalid.world_x,
                &invalid.world_z,
                &invalid.world_yaw_radians,
            };
            *fields[field] = float_from_bits(bits);
            const Observation before = capture_observation(
                runtime, fixture, baseline);
            const PickEntryPreview preview = runtime.preview_pick(
                baseline,
                invalid,
                fixture.request.target,
                fixture.request.affordance_id);
            require_out_of_range(preview, invalid);
            require(
                same_observation(
                    before,
                    capture_observation(runtime, fixture, baseline)),
                "invalid root preview mutated runtime or registry");
        }
    }
}

interaction::PickEntryPreview preview_twice_without_mutation(
    interaction::RuntimeFixture& fixture,
    interaction::InteractionRuntime& runtime) {
    const interaction::PickEntryRoot root = live_root(fixture.locomotion);
    const Observation before = capture_observation(
        runtime, fixture, fixture.locomotion);
    const interaction::PickEntryPreview first = runtime.preview_pick(
        fixture.locomotion,
        root,
        fixture.request.target,
        fixture.request.affordance_id);
    const Observation middle = capture_observation(
        runtime, fixture, fixture.locomotion);
    const interaction::PickEntryPreview second = runtime.preview_pick(
        fixture.locomotion,
        root,
        fixture.request.target,
        fixture.request.affordance_id);
    const Observation after = capture_observation(
        runtime, fixture, fixture.locomotion);
    require(same_preview(first, second), "repeated preview was not exact");
    require(
        same_observation(before, middle) && same_observation(before, after),
        "preview mutated runtime, registry, diagnostics, or caller");
    return first;
}

void test_preview_path_match_and_no_mutation() {
    using namespace interaction;
    RuntimeFixture feasible_fixture = make_runtime_fixture();
    InteractionRuntime feasible_runtime(
        feasible_fixture.database,
        feasible_fixture.features,
        feasible_fixture.registry,
        RuntimeConfig{});
    const PickEntryPreview feasible = preview_twice_without_mutation(
        feasible_fixture, feasible_runtime);
    require(feasible.path_feasible, "feasible preview lost its path");
    require(feasible.match_ready, "feasible preview was not match ready");
    require(
        feasible.path_reason == Reason::None &&
        feasible.match_reason == Reason::None,
        "feasible preview reported a reason");

    RuntimeFixture blocked_fixture = make_runtime_fixture();
    InteractionTarget* blocked_target = blocked_fixture.registry.find(
        blocked_fixture.request.target);
    require(blocked_target != nullptr, "blocked fixture target missing");
    blocked_target->table_size.z = 1.60F;
    InteractionRuntime blocked_runtime(
        blocked_fixture.database,
        blocked_fixture.features,
        blocked_fixture.registry,
        RuntimeConfig{});
    const PickEntryPreview blocked = preview_twice_without_mutation(
        blocked_fixture, blocked_runtime);
    require(!blocked.path_feasible, "blocked preview became path feasible");
    require(!blocked.match_ready, "blocked preview became match ready");
    require(
        blocked.path_reason == Reason::BlockedPath &&
        blocked.match_reason == Reason::BlockedPath,
        "blocked preview lost BlockedPath");

    RuntimeFixture costly_fixture = high_cost_fixture();
    InteractionRuntime costly_runtime(
        costly_fixture.database,
        costly_fixture.features,
        costly_fixture.registry,
        RuntimeConfig{});
    const PickEntryPreview costly = preview_twice_without_mutation(
        costly_fixture, costly_runtime);
    require(costly.path_feasible, "over-cost preview lost feasible path");
    require(!costly.match_ready, "over-cost preview became match ready");
    require(
        costly.path_reason == Reason::None &&
        costly.match_reason == Reason::PoorMatch,
        "over-cost preview lost PoorMatch separation");
    require(
        near(costly.total_cost, 106.40F, 1.0e-4F),
        "over-cost preview lost its finite diagnostic cost");
}

void test_nonunit_fractional_endpoint_certification() {
    using namespace interaction;
    constexpr float kDt = 1.0F / 25.0F;
    RuntimeConfig config{};
    config.playback.speed = 0.85F;
    const auto idle = [=](const LocomotionSnapshot& locomotion) {
        RuntimeInput input{};
        input.dt = kDt;
        input.locomotion = locomotion;
        return input;
    };
    const auto interact = [=](const RuntimeFixture& fixture) {
        RuntimeInput input{};
        input.dt = kDt;
        input.locomotion = fixture.locomotion;
        input.interact_pressed = true;
        input.pick_request = fixture.request;
        return input;
    };

    RuntimeFixture trace_fixture = make_nonunit_fractional_arc_fixture();
    InteractionRuntime trace_runtime(
        trace_fixture.database,
        trace_fixture.features,
        trace_fixture.registry,
        config);
    RuntimeOutput trace = trace_runtime.update(interact(trace_fixture));
    require(
        trace.diagnostics.state == RuntimeState::Preflight,
        "nonunit trace did not enter Preflight");
    trace = trace_runtime.update(idle(trace_fixture.locomotion));
    require(
        trace.diagnostics.state == RuntimeState::Align,
        "nonunit trace did not enter Align");
    for (int update = 0;
         update < 20 && trace.diagnostics.state == RuntimeState::Align;
         ++update) {
        trace = trace_runtime.update(idle(trace_fixture.locomotion));
    }
    require(
        trace.diagnostics.state == RuntimeState::PickupReplay,
        "nonunit trace did not reach PickupReplay");
    trace = trace_runtime.update(idle(trace_fixture.locomotion));
    constexpr int32_t kFractionalLeftFrame =
        runtime_fixture_detail::kFramesPerClip + 21;
    require(
        trace.diagnostics.frame == kFractionalLeftFrame,
        "nonunit trace did not land on the fractional source interval");
    const vec3 fractional_hand = world_pose(trace.pose).positions[
        kRightHandBone];

    RuntimeFixture fixture = make_nonunit_fractional_arc_fixture();
    set_fractional_endpoint_table(fixture, fractional_hand);
    const MatchInput match_input = match_input_for(fixture);
    const MatchResult authored = select_whole_clip(
        match_input, config.matcher);
    require(authored.accepted, "nonunit authored path was not clear");
    require(
        authored.candidate.entry_frame ==
            runtime_fixture_detail::kFramesPerClip + 10,
        "nonunit authored matcher did not choose Reach");
    const runtime_detail::RealizedPickTransitionEvaluation realized =
        runtime_detail::evaluate_realized_pick_transition(
            fixture.database,
            fixture.locomotion.pose,
            authored.candidate,
            match_input.target,
            match_input.affordance,
            config.playback,
            config.ik);
    require(
        !realized.feasible && realized.reason == Reason::BlockedPath,
        "nonunit certifier skipped the fractional wall endpoint");

    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        config);
    const PickEntryPreview preview = runtime.preview_pick(
        fixture.locomotion,
        live_root(fixture.locomotion),
        fixture.request.target,
        fixture.request.affordance_id);
    require(
        preview.path_feasible && preview.match_ready,
        "nonunit preview did not find a safe fallback");
    require(
        fixture.database.phases.at(
            static_cast<size_t>(preview.match_candidate.entry_frame)) ==
            static_cast<uint8_t>(Phase::Approach),
        "nonunit preview did not fall back to Approach");

    RuntimeOutput output = runtime.update(interact(fixture));
    require(
        output.diagnostics.state == RuntimeState::Preflight,
        "nonunit execution did not enter Preflight");
    output = runtime.update(idle(fixture.locomotion));
    require(
        output.diagnostics.state == RuntimeState::Align &&
            output.diagnostics.frame ==
                preview.match_candidate.entry_frame,
        "nonunit commit diverged from preview");
    for (int update = 0;
         update < 600 && output.diagnostics.state != RuntimeState::Carry &&
         output.diagnostics.result != ResultCode::Failed;
         ++update) {
        output = runtime.update(idle(fixture.locomotion));
    }
    require(
        output.diagnostics.state == RuntimeState::Carry &&
            output.diagnostics.result == ResultCode::Succeeded &&
            output.diagnostics.attached,
        "nonunit safe fallback did not reach attached Carry");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 || std::string_view(argv[1]) != "--fast-math-canary") {
        return 2;
    }
    try {
        test_nontrivial_rigid_map();
        test_bit_pattern_nonfinite_rejection();
        test_preview_path_match_and_no_mutation();
        test_nonunit_fractional_endpoint_certification();
    } catch (...) {
        return 1;
    }
    return 0;
}
