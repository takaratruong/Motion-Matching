#include "interaction_matcher.h"
#include "tests/cpp/interaction_runtime_fixture.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

bool near(float left, float right, float tolerance = 1.0e-6F) {
    return std::abs(left - right) <= tolerance;
}

bool near(vec3 left, vec3 right, float tolerance = 1.0e-6F) {
    return near(left.x, right.x, tolerance) &&
           near(left.y, right.y, tolerance) &&
           near(left.z, right.z, tolerance);
}

bool near(
    const std::array<float, 5>& left,
    const std::array<float, 5>& right,
    float tolerance = 1.0e-6F) {
    for (size_t index = 0; index < left.size(); ++index) {
        if (!near(left[index], right[index], tolerance)) return false;
    }
    return true;
}

void set_target_affordance(
    interaction::MatchInput& input,
    const interaction::GraspAffordance& affordance) {
    input.affordance = affordance;
    input.request.affordance_id = affordance.id;
    input.target.affordances = {affordance};
}

void test_frozen_contract_and_fixture_invariants() {
    using namespace interaction;

    static_assert(std::is_same_v<std::underlying_type_t<Phase>, uint8_t>);
    static_assert(static_cast<uint8_t>(Phase::Approach) == 0U);
    static_assert(static_cast<uint8_t>(Phase::Reach) == 1U);
    static_assert(static_cast<uint8_t>(Phase::Contact) == 2U);
    static_assert(static_cast<uint8_t>(Phase::Lift) == 3U);
    static_assert(static_cast<uint8_t>(Phase::Hold) == 4U);
    static_assert(std::is_same_v<std::underlying_type_t<Reason>, uint8_t>);

    const MatchConfig config{};
    assert(near(config.maximum_approach_m, 1.0F));
    assert(near(config.maximum_root_correction_m, 0.25F));
    assert(near(config.maximum_yaw_correction_radians, 0.436332313F));
    assert(near(config.maximum_hand_correction_m, 0.12F));
    assert(near(config.maximum_hand_orientation_radians, 0.436332313F));
    assert(near(config.group_weights, {1.0F, 1.0F, 2.0F, 2.0F, 1.0F}));
    assert(near(config.maximum_cost, 9.0F));

    const RuntimeFixture fixture = make_runtime_fixture();
    const Database& database = fixture.database;
    detail::validate_database(database);
    detail::validate_features(fixture.features);
    assert(database.frame_count == 150U);
    assert(database.clip_count == 2U);
    assert((database.range_starts == std::vector<int32_t>{0, 75}));
    assert((database.range_stops == std::vector<int32_t>{75, 150}));
    assert((database.active_hands == std::vector<uint8_t>{0U, 1U}));
    for (int32_t clip = 0; clip < 2; ++clip) {
        const int32_t start = database.range_starts.at(clip);
        for (int32_t local = 0; local < 75; ++local) {
            const Phase expected = local < 10 ? Phase::Approach
                : local < 25 ? Phase::Reach
                : local < 30 ? Phase::Contact
                : local < 40 ? Phase::Lift
                : Phase::Hold;
            assert(database.phases.at(start + local) ==
                   static_cast<uint8_t>(expected));
        }
        assert(near(
            runtime_fixture_detail::read_vec3(
                database.object_positions, start + 29).y,
            0.75F));
        assert(near(
            runtime_fixture_detail::read_vec3(
                database.object_positions, start + 39).y,
            0.95F));
    }
    const vec3 left_hold_start = pose_at_frame(database, 40)
        .positions[g1_skeleton::Simulation];
    const vec3 left_hold_stop = pose_at_frame(database, 74)
        .positions[g1_skeleton::Simulation];
    assert(near(left_hold_start, left_hold_stop));
    const vec3 right_hold_start = pose_at_frame(database, 115)
        .positions[g1_skeleton::Simulation];
    const vec3 right_hold_stop = pose_at_frame(database, 149)
        .positions[g1_skeleton::Simulation];
    assert(near(right_hold_stop.x - right_hold_start.x, 0.40F));
    const vec3 right_object_start = runtime_fixture_detail::read_vec3(
        database.object_positions, 115);
    const vec3 right_object_stop = runtime_fixture_detail::read_vec3(
        database.object_positions, 149);
    assert(near(right_object_stop.x - right_object_start.x, 0.40F));

    const MatchInput input = valid_input();
    assert(input.database != nullptr);
    assert(input.features != nullptr);
    assert(input.database->frame_count == 150U);
    for (float value : input.query) assert(near(value, 0.0F));
    const MatchInput wrong_hand = wrong_hand_input();
    detail::validate_database(*wrong_hand.database);
    detail::validate_features(*wrong_hand.features);

    const InteractionTarget* target =
        fixture.registry.find(fixture.request.target);
    const GraspAffordance* affordance = fixture.registry.find_affordance(
        fixture.request.target, fixture.request.affordance_id);
    assert(target != nullptr && affordance != nullptr);
    const RawQuery raw = build_raw_query(
        query_input_for(fixture, *target, *affordance));
    assert(std::equal(
        raw.begin(), raw.end(), fixture.features.offsets.begin()));
    const NormalizedQuery normalized = normalize_query(raw, fixture.features);
    for (float value : normalized) assert(near(value, 0.0F));

    const RuntimeFixture costly = high_cost_fixture();
    assert(costly.features.values.size() == fixture.features.values.size());
    for (size_t index = 0; index < costly.features.values.size(); ++index) {
        assert(near(
            costly.features.values[index] - fixture.features.values[index],
            10.0F));
    }

    const RuntimeFixture failed_contact = contact_failure_fixture();
    const WorldPose original_contact = world_pose(pose_at_frame(database, 100));
    const WorldPose changed_contact = world_pose(
        pose_at_frame(failed_contact.database, 100));
    assert(near(
        changed_contact.positions[kRightHandBone].x -
            original_contact.positions[kRightHandBone].x,
        0.05F));
    assert(near(
        world_pose(pose_at_frame(failed_contact.database, 99))
            .positions[kRightHandBone],
        world_pose(pose_at_frame(database, 99)).positions[kRightHandBone]));
}

void test_selects_valid_first_reach_with_exact_group_costs() {
    using namespace interaction;
    const MatchResult accepted = select_whole_clip(valid_input(), MatchConfig{});
    assert(accepted.accepted);
    assert(accepted.reason == Reason::None);
    assert(accepted.candidate.clip == 1);
    assert(accepted.candidate.entry_frame == 85);
    assert(accepted.candidate.entry_frame < accepted.candidate.contact_frame);
    assert(accepted.candidate.contact_frame == 100);
    assert(accepted.candidate.contact_frame < accepted.candidate.lift_frame);
    assert(accepted.candidate.lift_frame == 105);
    assert(accepted.candidate.lift_frame < accepted.candidate.hold_frame);
    assert(accepted.candidate.hold_frame == 115);
    assert(near(accepted.candidate.scene_from_source.position, vec3()));
    assert(near(accepted.candidate.entry_root_offset, vec3()));
    assert(near(accepted.candidate.entry_yaw_offset, 0.0F));
    assert(near(
        accepted.candidate.group_costs,
        std::array<float, 5>{0.01F, 0.04F, 0.09F, 0.16F, 0.25F}));
    assert(near(accepted.candidate.total_cost, 0.80F / 7.0F));
    assert(accepted.candidate.total_cost <= 9.0F);
}

void test_rejects_each_frozen_gate() {
    using namespace interaction;
    assert(select_whole_clip(wrong_hand_input(), {}).reason ==
           Reason::NoCandidate);
    assert(select_whole_clip(out_of_range_input(), {}).reason ==
           Reason::OutOfRange);
    assert(select_whole_clip(excessive_root_input(), {}).reason ==
           Reason::CorrectionLimit);
    assert(select_whole_clip(blocked_table_input(), {}).reason ==
           Reason::BlockedPath);
    assert(select_whole_clip(high_cost_input(), {}).reason ==
           Reason::PoorMatch);
}

void test_validates_exact_request_target_and_affordance() {
    using namespace interaction;
    MatchInput input = valid_input();
    input.database = nullptr;
    assert(select_whole_clip(input, {}).reason == Reason::PackUnavailable);

    input = valid_input();
    ++input.request.target.generation;
    assert(select_whole_clip(input, {}).reason == Reason::TargetChanged);

    input = valid_input();
    ++input.request.affordance_id;
    assert(select_whole_clip(input, {}).reason == Reason::TargetUnavailable);

    input = valid_input();
    input.target.affordances.clear();
    assert(select_whole_clip(input, {}).reason == Reason::TargetUnavailable);
}

void test_uses_planar_approach_alignment_and_root_correction() {
    using namespace interaction;
    MatchInput input = valid_input();
    input.target.object_world.position.y += 0.10F;
    MatchResult result = select_whole_clip(input, {});
    assert(result.accepted);
    assert(near(result.candidate.scene_from_source.position.y, 0.0F));
    assert(near(result.candidate.entry_root_offset.y, 0.0F));

    input = valid_input();
    input.locomotion.pose.positions[g1_skeleton::Simulation].y += 50.0F;
    result = select_whole_clip(input, {});
    assert(result.accepted);
    assert(near(result.candidate.entry_root_offset.y, 0.0F));

    input = valid_input();
    input.target.object_world.position.x = 0.20F;
    input.target.object_world.position.z = 2.80F;
    input.target.object_world.rotation = quat_from_angle_axis(
        0.20F, vec3(0.0F, 1.0F, 0.0F));
    result = select_whole_clip(input, {});
    assert(result.accepted);
    const Transform source_object = {
        runtime_fixture_detail::read_vec3(
            input.database->object_positions, 99),
        quat(),
    };
    const Transform mapped_object = compose(
        result.candidate.scene_from_source, source_object);
    assert(near(mapped_object.position.x, input.target.object_world.position.x));
    assert(near(mapped_object.position.z, input.target.object_world.position.z));
    assert(near(
        quat_angle_between(
            mapped_object.rotation,
            input.target.object_world.rotation),
        0.0F));

    input = valid_input();
    GraspAffordance shifted = input.affordance;
    shifted.hand_in_object.position.x += 0.121F;
    set_target_affordance(input, shifted);
    assert(select_whole_clip(input, {}).reason == Reason::CorrectionLimit);

    input = valid_input();
    const quat excessive_yaw = quat_from_angle_axis(
        0.436332313F + 0.01F, vec3(0.0F, 1.0F, 0.0F));
    input.target.object_world.rotation = excessive_yaw;
    MatchConfig yaw_config{};
    yaw_config.maximum_root_correction_m = 1.0F;
    assert(select_whole_clip(input, yaw_config).reason ==
           Reason::CorrectionLimit);

    input = valid_input();
    GraspAffordance rotated = input.affordance;
    rotated.hand_in_object.rotation = quat_from_angle_axis(
        0.436332313F + 0.01F, vec3(1.0F, 0.0F, 0.0F));
    set_target_affordance(input, rotated);
    assert(select_whole_clip(input, {}).reason == Reason::CorrectionLimit);
}

void test_prefers_reach_then_falls_back_to_nearest_phase_set() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    MatchInput input = match_input_for(fixture);

    for (size_t dimension = 0; dimension < kFeatureDimension; ++dimension) {
        fixture.features.values.at(85U * kFeatureDimension + dimension) =
            20.0F;
    }
    MatchResult result = select_whole_clip(input, {});
    assert(result.accepted);
    assert(result.candidate.entry_frame == 75);
    assert(fixture.database.phases.at(result.candidate.entry_frame) ==
           static_cast<uint8_t>(Phase::Approach));

    fixture = make_runtime_fixture();
    input = match_input_for(fixture);
    for (int32_t frame = 75; frame < 85; ++frame) {
        runtime_fixture_detail::set_group_row(
            fixture.features, frame, {0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
    }
    result = select_whole_clip(input, {});
    assert(result.accepted);
    assert(result.candidate.entry_frame == 85);
}

void test_ties_choose_lowest_clip_then_earliest_entry() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    fixture.database.active_hands.at(0) = 1U;
    MatchInput input = match_input_for(fixture);
    MatchResult result = select_whole_clip(input, {});
    assert(result.accepted);
    assert(result.candidate.clip == 0);
    assert(result.candidate.entry_frame == 10);

    for (size_t dimension = 0; dimension < kFeatureDimension; ++dimension) {
        fixture.features.values.at(10U * kFeatureDimension + dimension) =
            20.0F;
        fixture.features.values.at(85U * kFeatureDimension + dimension) =
            20.0F;
    }
    result = select_whole_clip(input, {});
    assert(result.accepted);
    assert(result.candidate.clip == 0);
    assert(result.candidate.entry_frame == 0);
}

void test_reads_serialized_groups_and_configured_weights() {
    using namespace interaction;
    MatchConfig config{};
    config.group_weights = {0.0F, 0.0F, 1.0F, 0.0F, 0.0F};
    const MatchResult result = select_whole_clip(valid_input(), config);
    assert(result.accepted);
    assert(near(result.candidate.total_cost, 0.09F));
}

void test_clearance_uses_segments_and_exempts_only_final_contact() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    MatchInput input = match_input_for(fixture);
    assert(select_whole_clip(input, {}).accepted);

    RuntimeFixture post_contact = make_runtime_fixture();
    MatchInput post_contact_input = match_input_for(post_contact);
    vec3 post_contact_hand = runtime_fixture_detail::read_bone_position(
        post_contact.database, 101, kRightHandBone);
    post_contact_hand.y = 0.60F;
    runtime_fixture_detail::write_bone_position(
        post_contact.database, 101, kRightHandBone, post_contact_hand);
    runtime_fixture_detail::recompute_velocities(post_contact.database);
    assert(select_whole_clip(post_contact_input, {}).reason ==
           Reason::BlockedPath);

    constexpr int32_t kLastPrecontact = 99;
    vec3 local_hand = runtime_fixture_detail::read_bone_position(
        fixture.database, kLastPrecontact, kRightHandBone);
    local_hand.z = 0.95F;
    runtime_fixture_detail::write_bone_position(
        fixture.database, kLastPrecontact, kRightHandBone, local_hand);
    runtime_fixture_detail::recompute_velocities(fixture.database);
    assert(select_whole_clip(input, {}).reason == Reason::BlockedPath);
}

void test_clearance_applies_entry_translation_to_root_and_hand_path() {
    using namespace interaction;
    MatchInput root_control = valid_input();
    root_control.target.table_size.z = 1.04F;
    assert(select_whole_clip(root_control, {}).accepted);

    MatchInput root_input = valid_input();
    root_input.locomotion.pose.positions[g1_skeleton::Simulation].z += 0.24F;
    root_input.target.table_size.z = 1.04F;
    assert(select_whole_clip(root_input, {}).reason == Reason::BlockedPath);

    MatchInput hand_control = valid_input();
    hand_control.target.table_world.position = vec3(0.20F, 0.36F, 2.50F);
    hand_control.target.table_size = vec3(0.04F, 0.72F, 0.04F);
    assert(select_whole_clip(hand_control, {}).accepted);

    MatchInput hand_input = hand_control;
    hand_input.locomotion.pose.positions[g1_skeleton::Simulation].x += 0.20F;
    hand_input.locomotion.pose.positions[g1_skeleton::Simulation].z += 0.10F;
    assert(select_whole_clip(hand_input, {}).reason == Reason::BlockedPath);
}

void test_clearance_applies_entry_yaw_to_hand_path() {
    using namespace interaction;
    MatchInput control = valid_input();
    control.target.table_world.position = vec3(0.08F, 0.36F, 2.39F);
    control.target.table_size = vec3(0.04F, 0.72F, 0.04F);
    assert(select_whole_clip(control, {}).accepted);

    MatchInput input = control;
    input.locomotion.pose.rotations[g1_skeleton::Simulation] =
        quat_from_angle_axis(0.20F, vec3(0.0F, 1.0F, 0.0F));
    assert(select_whole_clip(input, {}).reason == Reason::BlockedPath);
}

void test_failure_precedence_is_deterministic() {
    using namespace interaction;
    RuntimeFixture fixture = high_cost_fixture();
    fixture.database.active_hands.at(0) = 1U;
    for (int32_t frame = 40; frame < 75; ++frame) {
        fixture.database.phases.at(frame) = static_cast<uint8_t>(Phase::Lift);
    }
    MatchInput input = match_input_for(fixture);
    assert(select_whole_clip(input, {}).reason == Reason::OutOfRange);
}

}  // namespace

int main() {
    test_frozen_contract_and_fixture_invariants();
    test_selects_valid_first_reach_with_exact_group_costs();
    test_rejects_each_frozen_gate();
    test_validates_exact_request_target_and_affordance();
    test_uses_planar_approach_alignment_and_root_correction();
    test_prefers_reach_then_falls_back_to_nearest_phase_set();
    test_ties_choose_lowest_clip_then_earliest_entry();
    test_reads_serialized_groups_and_configured_weights();
    test_clearance_uses_segments_and_exempts_only_final_contact();
    test_clearance_applies_entry_translation_to_root_and_hand_path();
    test_clearance_applies_entry_yaw_to_hand_path();
    test_failure_precedence_is_deterministic();
    return 0;
}
