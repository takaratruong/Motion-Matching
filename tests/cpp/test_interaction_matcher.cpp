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

bool near(quat left, quat right, float tolerance = 1.0e-6F) {
    return near(left.w, right.w, tolerance) &&
           near(left.x, right.x, tolerance) &&
           near(left.y, right.y, tolerance) &&
           near(left.z, right.z, tolerance);
}

bool near(
    const interaction::Transform& left,
    const interaction::Transform& right,
    float tolerance = 1.0e-6F) {
    return near(left.position, right.position, tolerance) &&
           near(left.rotation, right.rotation, tolerance);
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

interaction::matcher_detail::PickEvaluationInput evaluation_input(
    const interaction::MatchInput& input) {
    return {
        input.database,
        input.features,
        input.query,
        input.locomotion,
        input.target,
        input.affordance,
    };
}

void assert_same_match_result(
    const interaction::MatchResult& left,
    const interaction::MatchResult& right) {
    assert(left.accepted == right.accepted);
    assert(left.reason == right.reason);
    if (!left.accepted) return;
    assert(left.candidate.clip == right.candidate.clip);
    assert(left.candidate.entry_frame == right.candidate.entry_frame);
    assert(left.candidate.contact_frame == right.candidate.contact_frame);
    assert(left.candidate.lift_frame == right.candidate.lift_frame);
    assert(left.candidate.hold_frame == right.candidate.hold_frame);
    assert(near(
        left.candidate.scene_from_source,
        right.candidate.scene_from_source));
    assert(near(
        left.candidate.entry_root_offset,
        right.candidate.entry_root_offset));
    assert(near(
        left.candidate.entry_yaw_offset,
        right.candidate.entry_yaw_offset));
    assert(near(left.candidate.total_cost, right.candidate.total_cost));
    assert(near(left.candidate.group_costs, right.candidate.group_costs));
}

interaction::matcher_detail::PickEvaluation evaluate(
    const interaction::MatchInput& input,
    const interaction::MatchConfig& config = {}) {
    return interaction::matcher_detail::evaluate_pick_entries(
        evaluation_input(input), config);
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

void test_rejects_authored_slot_metadata_mismatch() {
    using namespace interaction;

    MatchInput input = valid_input();
    input.target.affordances.front().interaction_slots = {
        {3U, -0.41F, -0.22F, 1.10F},
        {9U, 0.18F, -0.39F, 0.20F},
    };
    input.affordance = input.target.affordances.front();
    input.affordance.interaction_slots[0].root_x_object_m += 0.001F;

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
    assert(evaluate(input).path_feasible);

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
    assert(!evaluate(post_contact_input).path_feasible);
    assert(evaluate(post_contact_input).path_reason == Reason::BlockedPath);

    constexpr int32_t kLastPrecontact = 99;
    vec3 local_hand = runtime_fixture_detail::read_bone_position(
        fixture.database, kLastPrecontact, kRightHandBone);
    local_hand.z = 0.95F;
    runtime_fixture_detail::write_bone_position(
        fixture.database, kLastPrecontact, kRightHandBone, local_hand);
    runtime_fixture_detail::recompute_velocities(fixture.database);
    assert(select_whole_clip(input, {}).reason == Reason::BlockedPath);
    assert(!evaluate(input).path_feasible);
    assert(evaluate(input).path_reason == Reason::BlockedPath);
}

void test_clearance_applies_entry_translation_to_root_and_hand_path() {
    using namespace interaction;
    MatchInput root_control = valid_input();
    root_control.target.table_size.z = 1.04F;
    assert(select_whole_clip(root_control, {}).accepted);
    assert(evaluate(root_control).path_feasible);

    MatchInput root_input = valid_input();
    root_input.locomotion.pose.positions[g1_skeleton::Simulation].z += 0.24F;
    root_input.target.table_size.z = 1.04F;
    assert(select_whole_clip(root_input, {}).reason == Reason::BlockedPath);
    assert(!evaluate(root_input).path_feasible);
    assert(evaluate(root_input).path_reason == Reason::BlockedPath);

    MatchInput hand_control = valid_input();
    hand_control.target.table_world.position = vec3(0.20F, 0.36F, 2.50F);
    hand_control.target.table_size = vec3(0.04F, 0.72F, 0.04F);
    assert(select_whole_clip(hand_control, {}).accepted);
    assert(evaluate(hand_control).path_feasible);

    MatchInput hand_input = hand_control;
    hand_input.locomotion.pose.positions[g1_skeleton::Simulation].x += 0.20F;
    hand_input.locomotion.pose.positions[g1_skeleton::Simulation].z += 0.10F;
    assert(select_whole_clip(hand_input, {}).reason == Reason::BlockedPath);
    assert(!evaluate(hand_input).path_feasible);
    assert(evaluate(hand_input).path_reason == Reason::BlockedPath);
}

void test_root_clearance_tolerates_only_subcentimeter_proxy_penetration() {
    using namespace interaction;

    const auto input_with_proxy_penetration = [](float penetration_m) {
        constexpr float kRootToTableCenterM = 1.0F;
        constexpr float kNominalRootClearanceM = 0.25F;
        MatchInput input = valid_input();
        input.target.table_size.z = 2.0F * (
            kRootToTableCenterM - kNominalRootClearanceM + penetration_m);
        return input;
    };

    const MatchInput shallow = input_with_proxy_penetration(0.009F);
    assert(select_whole_clip(shallow, {}).accepted);
    assert(evaluate(shallow).path_feasible);

    const MatchInput deep = input_with_proxy_penetration(0.011F);
    assert(select_whole_clip(deep, {}).reason == Reason::BlockedPath);
    assert(!evaluate(deep).path_feasible);
    assert(evaluate(deep).path_reason == Reason::BlockedPath);
}

void test_clearance_applies_entry_yaw_to_hand_path() {
    using namespace interaction;
    MatchInput control = valid_input();
    control.target.table_world.position = vec3(0.08F, 0.36F, 2.39F);
    control.target.table_size = vec3(0.04F, 0.72F, 0.04F);
    assert(select_whole_clip(control, {}).accepted);
    assert(evaluate(control).path_feasible);

    MatchInput input = control;
    input.locomotion.pose.rotations[g1_skeleton::Simulation] =
        quat_from_angle_axis(0.20F, vec3(0.0F, 1.0F, 0.0F));
    assert(select_whole_clip(input, {}).reason == Reason::BlockedPath);
    assert(!evaluate(input).path_feasible);
    assert(evaluate(input).path_reason == Reason::BlockedPath);
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

void test_shared_pick_evaluation_preserves_select_behavior() {
    using namespace interaction;

    const auto assert_parity = [](const MatchInput& input) {
        const MatchResult selected = select_whole_clip(input, MatchConfig{});
        const matcher_detail::PickEvaluation evaluated = evaluate(input);
        assert_same_match_result(selected, evaluated.selection);
        assert(evaluated.match_ready == selected.accepted);
        assert(evaluated.match_reason == selected.reason);
    };

    assert_parity(valid_input());
    assert_parity(wrong_hand_input());
    assert_parity(out_of_range_input());
    assert_parity(excessive_root_input());
    assert_parity(blocked_table_input());
    assert_parity(high_cost_input());

    RuntimeFixture tied = make_runtime_fixture();
    tied.database.active_hands.at(0) = 1U;
    MatchInput tied_input = match_input_for(tied);
    const matcher_detail::PickEvaluation tied_evaluation = evaluate(tied_input);
    assert_same_match_result(
        select_whole_clip(tied_input, {}), tied_evaluation.selection);
    assert(tied_evaluation.selection.candidate.clip == 0);
    assert(tied_evaluation.selection.candidate.entry_frame == 10);

    MatchInput missing_pack = valid_input();
    missing_pack.database = nullptr;
    missing_pack.request = {};
    assert(select_whole_clip(missing_pack, {}).reason ==
           Reason::PackUnavailable);

    missing_pack = valid_input();
    missing_pack.features = nullptr;
    ++missing_pack.request.target.generation;
    assert(select_whole_clip(missing_pack, {}).reason ==
           Reason::PackUnavailable);

    MatchInput invalid_request = valid_input();
    invalid_request.request.request_id = 0U;
    assert(select_whole_clip(invalid_request, {}).reason ==
           Reason::TargetUnavailable);
}

void test_shared_pick_evaluation_separates_path_from_cost_readiness() {
    using namespace interaction;

    const matcher_detail::PickEvaluation accepted = evaluate(valid_input());
    assert(accepted.path_feasible);
    assert(accepted.path_reason == Reason::None);
    assert(accepted.match_ready);
    assert(accepted.match_reason == Reason::None);
    assert(accepted.selection.accepted);
    assert(accepted.feasible_entry_frame ==
           accepted.selection.candidate.entry_frame);
    assert(accepted.contact_frame ==
           accepted.selection.candidate.contact_frame);
    assert(accepted.total_cost_available);
    assert(near(accepted.total_cost, 0.80F / 7.0F));

    const matcher_detail::PickEvaluation over_cost = evaluate(
        high_cost_input());
    assert(over_cost.path_feasible);
    assert(over_cost.path_reason == Reason::None);
    assert(!over_cost.match_ready);
    assert(over_cost.match_reason == Reason::PoorMatch);
    assert(!over_cost.selection.accepted);
    assert(over_cost.selection.reason == Reason::PoorMatch);
    assert(over_cost.feasible_entry_frame == 85);
    assert(over_cost.contact_frame == 100);
    assert(over_cost.total_cost_available);
    assert(near(over_cost.total_cost, 106.40F, 1.0e-4F));
}

void test_shared_pick_evaluation_preserves_hard_reason_order() {
    using namespace interaction;

    const matcher_detail::PickEvaluation no_candidate = evaluate(
        wrong_hand_input());
    assert(!no_candidate.path_feasible);
    assert(no_candidate.path_reason == Reason::NoCandidate);

    const matcher_detail::PickEvaluation out_of_range = evaluate(
        out_of_range_input());
    assert(!out_of_range.path_feasible);
    assert(out_of_range.path_reason == Reason::OutOfRange);

    const matcher_detail::PickEvaluation correction = evaluate(
        excessive_root_input());
    assert(!correction.path_feasible);
    assert(correction.path_reason == Reason::CorrectionLimit);

    const matcher_detail::PickEvaluation blocked = evaluate(
        blocked_table_input());
    assert(!blocked.path_feasible);
    assert(blocked.path_reason == Reason::BlockedPath);

    RuntimeFixture out_of_range_and_correction = make_runtime_fixture();
    out_of_range_and_correction.database.active_hands.at(0) = 1U;
    for (int32_t frame = 40; frame < 75; ++frame) {
        out_of_range_and_correction.database.phases.at(frame) =
            static_cast<uint8_t>(Phase::Lift);
    }
    MatchInput priority_input = match_input_for(out_of_range_and_correction);
    priority_input.locomotion.pose.positions[g1_skeleton::Simulation].z +=
        0.30F;
    const matcher_detail::PickEvaluation out_of_range_first = evaluate(
        priority_input);
    assert(!out_of_range_first.path_feasible);
    assert(out_of_range_first.path_reason == Reason::OutOfRange);

    RuntimeFixture correction_and_blocked = make_runtime_fixture();
    correction_and_blocked.database.active_hands.at(0) = 1U;
    vec3 shifted_root = runtime_fixture_detail::read_bone_position(
        correction_and_blocked.database, 10, g1_skeleton::Simulation);
    shifted_root.x += 0.30F;
    runtime_fixture_detail::write_bone_position(
        correction_and_blocked.database,
        10,
        g1_skeleton::Simulation,
        shifted_root);
    MatchInput correction_first_input = match_input_for(
        correction_and_blocked);
    correction_first_input.target.table_size.z = 1.60F;
    const matcher_detail::PickEvaluation correction_first = evaluate(
        correction_first_input);
    assert(!correction_first.path_feasible);
    assert(correction_first.path_reason == Reason::CorrectionLimit);
}

void test_shared_pick_evaluation_reports_first_feasible_entry() {
    using namespace interaction;

    RuntimeFixture fixture = make_runtime_fixture();
    MatchInput input = match_input_for(fixture);
    for (size_t dimension = 0; dimension < kFeatureDimension; ++dimension) {
        fixture.features.values.at(85U * kFeatureDimension + dimension) =
            20.0F;
    }
    const matcher_detail::PickEvaluation evaluation = evaluate(input);
    assert(evaluation.path_feasible);
    assert(evaluation.feasible_entry_frame == 85);
    assert(evaluation.contact_frame == 100);
    assert(evaluation.match_ready);
    assert(evaluation.selection.candidate.entry_frame == 75);
    assert(fixture.database.phases.at(
               evaluation.selection.candidate.entry_frame) ==
           static_cast<uint8_t>(Phase::Approach));

    fixture = make_runtime_fixture();
    input = match_input_for(fixture);
    for (int32_t frame = 75; frame < 85; ++frame) {
        runtime_fixture_detail::set_group_row(
            fixture.features,
            frame,
            {0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
    }
    const matcher_detail::PickEvaluation reach_short_circuit = evaluate(input);
    assert(reach_short_circuit.match_ready);
    assert(reach_short_circuit.feasible_entry_frame == 85);
    assert(reach_short_circuit.selection.candidate.entry_frame == 85);
}

void test_candidate_feasibility_rejection_continues_from_reach_to_approach() {
    using namespace interaction;

    RuntimeFixture fixture = make_runtime_fixture();
    const MatchInput input = match_input_for(fixture);
    bool saw_reach = false;
    bool saw_approach = false;
    const matcher_detail::PickEvaluation evaluated =
        matcher_detail::evaluate_pick_entries(
            evaluation_input(input),
            MatchConfig{},
            [&](const MatchCandidate& candidate) {
                assert(candidate.clip == 1);
                assert(candidate.contact_frame == 100);
                assert(candidate.lift_frame == 105);
                assert(candidate.hold_frame == 115);
                const Phase phase = static_cast<Phase>(
                    fixture.database.phases.at(
                        static_cast<size_t>(candidate.entry_frame)));
                if (phase == Phase::Reach) {
                    saw_reach = true;
                    return Reason::BlockedPath;
                }
                assert(phase == Phase::Approach);
                saw_approach = true;
                return Reason::None;
            });

    assert(saw_reach);
    assert(saw_approach);
    assert(evaluated.path_feasible);
    assert(evaluated.path_reason == Reason::None);
    assert(evaluated.match_ready);
    assert(evaluated.match_reason == Reason::None);
    assert(evaluated.selection.accepted);
    assert(evaluated.selection.candidate.entry_frame == 75);
    assert(fixture.database.phases.at(
               static_cast<size_t>(
                   evaluated.selection.candidate.entry_frame)) ==
           static_cast<uint8_t>(Phase::Approach));

    const MatchResult unfiltered = select_whole_clip(input, MatchConfig{});
    assert(unfiltered.accepted);
    assert(unfiltered.candidate.entry_frame == 85);

    const matcher_detail::PickEvaluation all_blocked =
        matcher_detail::evaluate_pick_entries(
            evaluation_input(input),
            MatchConfig{},
            [](const MatchCandidate&) { return Reason::BlockedPath; });
    assert(!all_blocked.path_feasible);
    assert(all_blocked.path_reason == Reason::BlockedPath);
    assert(!all_blocked.match_ready);
    assert(all_blocked.match_reason == Reason::BlockedPath);
    assert(!all_blocked.selection.accepted);
    assert(all_blocked.selection.reason == Reason::BlockedPath);
}

void test_candidate_feasibility_out_of_range_exception_propagates() {
    using namespace interaction;

    const MatchInput input = valid_input();
    bool propagated = false;
    try {
        (void)matcher_detail::evaluate_pick_entries(
            evaluation_input(input),
            MatchConfig{},
            [](const MatchCandidate&) -> Reason {
                throw std::out_of_range(
                    "candidate feasibility callback failure");
            });
    } catch (const std::out_of_range&) {
        propagated = true;
    }
    assert(propagated);
}

void test_shared_pick_evaluation_preserves_match_reason_priority_and_cost_availability() {
    using namespace interaction;

    RuntimeFixture malformed_features = make_runtime_fixture();
    malformed_features.features.values.clear();
    const MatchInput malformed_input = match_input_for(malformed_features);
    const matcher_detail::PickEvaluation malformed = evaluate(malformed_input);
    assert(malformed.path_feasible);
    assert(malformed.path_reason == Reason::None);
    assert(!malformed.match_ready);
    assert(malformed.match_reason == Reason::OutOfRange);
    assert(malformed.selection.reason == Reason::OutOfRange);
    assert(malformed.feasible_entry_frame == 85);
    assert(malformed.contact_frame == 100);
    assert(!malformed.total_cost_available);
    assert_same_match_result(
        select_whole_clip(malformed_input, {}), malformed.selection);

    const matcher_detail::PickEvaluation over_cost = evaluate(
        high_cost_input());
    assert(over_cost.path_feasible);
    assert(!over_cost.match_ready);
    assert(over_cost.total_cost_available);
    assert(over_cost.match_reason == Reason::PoorMatch);

    RuntimeFixture mixed = high_cost_fixture();
    mixed.database.active_hands.at(0) = 1U;
    mixed.features.values.resize(
        75U * static_cast<size_t>(kFeatureDimension));
    const MatchInput mixed_input = match_input_for(mixed);
    const matcher_detail::PickEvaluation mixed_evaluation = evaluate(
        mixed_input);
    assert(mixed_evaluation.path_feasible);
    assert(mixed_evaluation.path_reason == Reason::None);
    assert(!mixed_evaluation.match_ready);
    assert(mixed_evaluation.total_cost_available);
    assert(mixed_evaluation.match_reason == Reason::OutOfRange);
    assert(mixed_evaluation.selection.reason == Reason::OutOfRange);
    assert(mixed_evaluation.feasible_entry_frame == 10);
    assert(mixed_evaluation.contact_frame == 25);
    assert(near(mixed_evaluation.total_cost, 106.40F, 1.0e-4F));
    assert_same_match_result(
        select_whole_clip(mixed_input, {}), mixed_evaluation.selection);
}

void test_preview_match_reason_preserves_retryable_poor_match() {
    using namespace interaction;
    RuntimeFixture fixture = high_cost_fixture();
    fixture.database.active_hands.at(0) = 1U;
    const MatchInput input = match_input_for(fixture);

    const matcher_detail::PickEvaluation result =
        matcher_detail::evaluate_pick_entries(
            evaluation_input(input),
            MatchConfig{},
            [](const MatchCandidate& candidate) {
                return candidate.clip == 0
                    ? Reason::CorrectionLimit
                    : Reason::None;
            });

    assert(result.path_feasible);
    assert(result.path_reason == Reason::None);
    assert(!result.match_ready);
    assert(result.feasible_entry_frame == 85);
    assert(result.contact_frame == 100);
    assert(result.total_cost_available);
    assert(near(result.total_cost, 106.40F, 1.0e-4F));
    assert(result.match_reason == Reason::PoorMatch);
    assert(result.selection.reason == Reason::CorrectionLimit);
}

}  // namespace

int main() {
    test_frozen_contract_and_fixture_invariants();
    test_selects_valid_first_reach_with_exact_group_costs();
    test_rejects_each_frozen_gate();
    test_validates_exact_request_target_and_affordance();
    test_rejects_authored_slot_metadata_mismatch();
    test_uses_planar_approach_alignment_and_root_correction();
    test_prefers_reach_then_falls_back_to_nearest_phase_set();
    test_ties_choose_lowest_clip_then_earliest_entry();
    test_reads_serialized_groups_and_configured_weights();
    test_clearance_uses_segments_and_exempts_only_final_contact();
    test_clearance_applies_entry_translation_to_root_and_hand_path();
    test_root_clearance_tolerates_only_subcentimeter_proxy_penetration();
    test_clearance_applies_entry_yaw_to_hand_path();
    test_failure_precedence_is_deterministic();
    test_shared_pick_evaluation_preserves_select_behavior();
    test_shared_pick_evaluation_separates_path_from_cost_readiness();
    test_shared_pick_evaluation_preserves_hard_reason_order();
    test_shared_pick_evaluation_reports_first_feasible_entry();
    test_candidate_feasibility_rejection_continues_from_reach_to_approach();
    test_candidate_feasibility_out_of_range_exception_propagates();
    test_shared_pick_evaluation_preserves_match_reason_priority_and_cost_availability();
    test_preview_match_reason_preserves_retryable_poor_match();
    return 0;
}
