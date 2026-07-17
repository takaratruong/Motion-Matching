#include "interaction_pick_assist.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

float float_from_bits(uint32_t bits) {
    float value = 0.0F;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool finite_from_bits(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

void require_invalid_config(
    const interaction::PickAssistConfig& config,
    const char* message) {
    try {
        const interaction::ControllerPickAssist assist(config);
        (void)assist;
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

void require_near(
    float actual,
    float expected,
    float tolerance,
    const char* message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

void require_near(
    vec3 actual,
    vec3 expected,
    float tolerance,
    const char* message) {
    require_near(actual.x, expected.x, tolerance, message);
    require_near(actual.y, expected.y, tolerance, message);
    require_near(actual.z, expected.z, tolerance, message);
}

interaction::InteractionTarget make_target() {
    interaction::InteractionTarget target{};
    target.handle = {41U, 3U};
    target.object_world = {vec3(0.0F, 0.75F, 0.30F), quat()};
    target.object_dimensions = vec3(0.20F, 0.30F, 0.20F);
    target.object_bounds = {vec3(), target.object_dimensions * 0.5F};
    target.state = interaction::ObjectState::Free;
    interaction::GraspAffordance affordance{};
    affordance.id = 7U;
    affordance.hand = interaction::Hand::Right;
    affordance.approach_direction_object = vec3(0.0F, 0.0F, 1.0F);
    affordance.clearance_radius = 0.04F;
    target.affordances.push_back(affordance);
    return target;
}

interaction::InteractionTarget make_frozen_slot_target() {
    interaction::InteractionTarget target{};
    target.handle = {51U, 4U};
    target.object_world = {vec3(0.0F, 0.80F, 0.0F), quat()};
    target.object_profile_id = 101U;
    target.object_bounds = {vec3(), vec3(0.05F, 0.10F, 0.05F)};
    target.object_dimensions = vec3(0.10F, 0.20F, 0.10F);
    target.table_world = {vec3(100.0F, 0.0F, 100.0F), quat()};
    target.table_size = vec3(1.0F, 0.10F, 1.0F);
    target.state = interaction::ObjectState::Free;
    target.owner_request = 0U;

    interaction::GraspAffordance affordance{};
    affordance.id = 7U;
    affordance.hand = interaction::Hand::Right;
    affordance.hand_in_object = {vec3(), quat()};
    affordance.approach_direction_object = vec3(0.0F, 0.0F, 1.0F);
    affordance.clearance_radius = 0.04F;
    affordance.interaction_slots = {
        {12U, 0.70F, 0.0F, 0.0F},
        {9U, 0.0F, 0.50F, 0.0F},
        {11U, 0.0F, 0.80F, 0.0F},
    };
    target.affordances.push_back(affordance);
    return target;
}

interaction::PickAssistStart make_frozen_slot_start(
    const interaction::InteractionTarget& target) {
    interaction::PickAssistStart start{};
    start.target_snapshot = target;
    start.affordance_id = target.affordances.front().id;
    start.root_world = {vec3(0.0F, 0.0F, 0.0F), quat()};
    return start;
}

struct Scenario {
    interaction::InteractionTarget target = make_target();
    interaction::PickAssistStart start{};
    interaction::PickAssistObservation observation{};

    Scenario() {
        start.target = target.handle;
        start.affordance_id = target.affordances.front().id;
        start.hand = target.affordances.front().hand;
        start.object_world = target.object_world;
        start.root_world = {vec3(0.0F, 0.0F, -0.90F), quat()};
        start.reach_waypoint = {vec3(0.0F, 0.0F, -0.10F), quat()};
        start.slots = interaction::make_pick_entry_slots(
            start.reach_waypoint, target);
        observation.target = &target;
        observation.displayed_root = start.root_world;
        observation.snapshot_fingerprint = 91U;
        observation.preview_snapshot_fingerprint = 91U;
    }
};

vec3 reach_entry_point(const Scenario& scenario) {
    vec3 forward = quat_mul_vec3(
        scenario.start.reach_waypoint.rotation,
        vec3(0.0F, 0.0F, 1.0F));
    forward.y = 0.0F;
    return scenario.start.reach_waypoint.position - 0.60F * normalize(forward);
}

interaction::PickEntryPreview ready_preview(
    const interaction::PickEntrySlot& slot) {
    interaction::PickEntryPreview preview{};
    preview.path_feasible = true;
    preview.match_ready = true;
    preview.prospective_root = slot.prospective_root;
    return preview;
}

std::array<interaction::PickEntryPreview, 2> ready_previews(
    const Scenario& scenario) {
    return {
        ready_preview(scenario.start.slots.ordered[0]),
        ready_preview(scenario.start.slots.ordered[1]),
    };
}

bool is_zero(vec3 value) {
    return value.x == 0.0F && value.y == 0.0F && value.z == 0.0F;
}

void require_final_preview_diagnostics_cleared(
    const interaction::PickAssistDiagnostics& diagnostics) {
    const auto& preview = diagnostics.final_preview;
    require(
        !preview.available && !preview.all_preview_roots_finite &&
            !preview.fingerprint_equal && !preview.path_feasible &&
            preview.path_reason == interaction::Reason::None &&
            !preview.match_ready &&
            preview.match_reason == interaction::Reason::None &&
            !preview.prospective_root_equal &&
            preview.feasible_entry_frame == -1 &&
            preview.contact_frame == -1 && preview.total_cost == 0.0F,
        "final-preview diagnostics were not cleared");
}

void enter_final_approach(
    interaction::ControllerPickAssist& assist,
    Scenario& scenario,
    size_t slot_index) {
    require(assist.begin(scenario.start), "final fixture begin failed");
    scenario.observation.displayed_root.position = reach_entry_point(scenario);
    assist.observe(scenario.observation);
    std::array<interaction::PickEntryPreview, 2> previews{};
    previews[slot_index] =
        ready_preview(scenario.start.slots.ordered[slot_index]);
    scenario.observation.previews = previews;
    assist.observe(scenario.observation);
    scenario.observation.previews.reset();
    require(
        assist.diagnostics().state ==
            interaction::PickAssistState::FinalApproach &&
        assist.diagnostics().selected_slot == static_cast<int>(slot_index),
        "final fixture did not freeze requested slot");
}

void enter_final_preview(
    interaction::ControllerPickAssist& assist,
    Scenario& scenario,
    size_t slot_index = 0U) {
    enter_final_approach(assist, scenario, slot_index);
    scenario.observation.displayed_root =
        scenario.start.slots.ordered[slot_index].waypoint;
    scenario.observation.simulation_velocity = vec3();
    scenario.observation.displayed_planar_speed_mps = 0.0F;
    assist.observe(scenario.observation);
    for (uint32_t tick = 0U; tick < 5U; ++tick) {
        assist.observe(scenario.observation);
    }
    require(
        assist.diagnostics().state ==
            interaction::PickAssistState::FinalPreview,
        "final-preview fixture did not settle");
}

void test_idle_does_not_override_input() {
    interaction::ControllerPickAssist assist;
    const interaction::PickAssistOutput output = assist.observe({});
    require(!output.override_steering, "Idle overrode steering");
    require(!output.needs_preview, "Idle requested a preview");
    require(!output.submit_interact, "Idle submitted interaction");
    require(!assist.active(), "Idle reported active");
    require(!assist.owns_manual_interact(),
        "Idle owned manual interaction");
    require(
        assist.diagnostics().state == interaction::PickAssistState::Idle,
        "Idle diagnostics reported the wrong state");
}

void test_begin_selects_and_freezes_one_authored_slot() {
    const interaction::InteractionTarget target = make_frozen_slot_target();
    const interaction::PickAssistStart start = make_frozen_slot_start(target);
    interaction::ControllerPickAssist assist;

    require(assist.begin(start, &target), "valid frozen-slot begin failed");
    const interaction::PickAssistDiagnostics& diagnostics =
        assist.diagnostics();
    require(
        diagnostics.state == interaction::PickAssistState::SlotApproach,
        "valid begin did not enter SlotApproach");
    require(
        diagnostics.selected_slot_id == 9U,
        "deterministic begin did not select authored slot ID 9");
    require(
        diagnostics.slot_selection.selected_index.has_value(),
        "valid begin did not retain its selected diagnostic index");
    require(
        diagnostics.route_length_m ==
            diagnostics.slot_selection.ordered[
                *diagnostics.slot_selection.selected_index].route_length_m,
        "begin route did not retain selected-slot provenance");
}

void test_slot_approach_observe_is_inert_before_legacy_preflight() {
    const interaction::InteractionTarget target = make_frozen_slot_target();
    const interaction::PickAssistStart start = make_frozen_slot_start(target);
    interaction::ControllerPickAssist assist;

    require(assist.begin(start, &target), "SlotApproach begin failed");
    interaction::PickAssistObservation invalid_observation{};
    invalid_observation.runtime_state = interaction::RuntimeState::Preflight;

    const interaction::PickAssistOutput output =
        assist.observe(invalid_observation);
    require(
        !output.override_steering && is_zero(output.left_stick) &&
            is_zero(output.right_stick) && !output.force_strafe &&
            !output.stationary_constraint && !output.needs_preview &&
            !output.submit_interact,
        "SlotApproach processed an invalid legacy observation");
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::SlotApproach &&
            assist.diagnostics().reason == interaction::PickAssistReason::None,
        "invalid legacy observation changed SlotApproach diagnostics");
    require(assist.active() && assist.owns_manual_interact(),
        "inert SlotApproach observation released ownership");
    require(!assist.take_submission(1U).has_value(),
        "inert SlotApproach observation produced a submission");
}

void require_frozen_slot_begin_failure(
    interaction::InteractionTarget target,
    interaction::PickAssistStart start,
    interaction::PickAssistReason expected,
    const char* message) {
    start.target_snapshot = target;
    interaction::ControllerPickAssist assist;
    require(!assist.begin(start, &target), message);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason == expected,
        "frozen-slot begin failure reported the wrong stable reason");
    require(
        !assist.active() && !assist.owns_manual_interact(),
        "failed frozen-slot begin retained input ownership");
}

void test_begin_maps_every_aggregate_no_winner_reason() {
    {
        interaction::InteractionTarget target = make_frozen_slot_target();
        target.affordances.front().interaction_slots.clear();
        require_frozen_slot_begin_failure(
            target,
            make_frozen_slot_start(target),
            interaction::PickAssistReason::NoAuthoredSlot,
            "begin accepted a target with no authored slots");
    }
    {
        interaction::InteractionTarget target = make_frozen_slot_target();
        target.object_world.rotation =
            quat(0.5F, 0.5F, 0.5F, -0.5F);
        require_frozen_slot_begin_failure(
            target,
            make_frozen_slot_start(target),
            interaction::PickAssistReason::InvalidGeometry,
            "begin accepted invalid slot mapping geometry");
    }
    {
        interaction::InteractionTarget target = make_frozen_slot_target();
        target.affordances.front().interaction_slots = {
            {20U, 0.0F, 1.10F, 0.0F},
            {21U, 0.0F, 1.20F, 0.0F},
        };
        require_frozen_slot_begin_failure(
            target,
            make_frozen_slot_start(target),
            interaction::PickAssistReason::OutsideTravelEnvelope,
            "begin accepted only outside-envelope slots");
    }
    {
        interaction::InteractionTarget target = make_frozen_slot_target();
        interaction::PickAssistStart start = make_frozen_slot_start(target);
        start.obstacles.push_back({
            vec3(0.0F, 0.0F, 0.0F),
            vec3(0.10F, 0.10F, 0.10F),
        });
        require_frozen_slot_begin_failure(
            target,
            start,
            interaction::PickAssistReason::AllSlotsBlocked,
            "begin accepted an all-blocked slot set");
    }
}

void test_slot_reason_mapping_is_exhaustive_and_same_named() {
    struct Mapping {
        interaction::PickSlotReason slot;
        interaction::PickAssistReason assist;
    };
    const Mapping mappings[] = {
        {interaction::PickSlotReason::None,
         interaction::PickAssistReason::None},
        {interaction::PickSlotReason::NoAuthoredSlot,
         interaction::PickAssistReason::NoAuthoredSlot},
        {interaction::PickSlotReason::InvalidGeometry,
         interaction::PickAssistReason::InvalidGeometry},
        {interaction::PickSlotReason::OutsideTravelEnvelope,
         interaction::PickAssistReason::OutsideTravelEnvelope},
        {interaction::PickSlotReason::TableBlocked,
         interaction::PickAssistReason::TableBlocked},
        {interaction::PickSlotReason::ObstacleBlocked,
         interaction::PickAssistReason::ObstacleBlocked},
        {interaction::PickSlotReason::AllSlotsBlocked,
         interaction::PickAssistReason::AllSlotsBlocked},
    };
    for (const Mapping& mapping : mappings) {
        require(
            interaction::pick_assist_reason_from_slot_reason(mapping.slot) ==
                mapping.assist,
            "pick-slot reason did not map one-to-one to assist reason");
    }
}

void test_failed_begin_can_immediately_begin_a_valid_attempt() {
    interaction::InteractionTarget invalid = make_frozen_slot_target();
    invalid.affordances.front().interaction_slots.clear();
    interaction::ControllerPickAssist assist;
    require(
        !assist.begin(make_frozen_slot_start(invalid), &invalid),
        "retry fixture unexpectedly accepted its invalid attempt");
    require(
        !assist.active() && !assist.owns_manual_interact(),
        "retry fixture retained ownership after failure");

    const interaction::InteractionTarget valid = make_frozen_slot_target();
    require(
        assist.begin(make_frozen_slot_start(valid), &valid),
        "failed assist could not immediately begin a valid attempt");
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::SlotApproach &&
            assist.diagnostics().reason == interaction::PickAssistReason::None &&
            assist.diagnostics().selected_slot_id == 9U,
        "valid retry did not replace the failed attempt diagnostics");
}

void test_diagnostic_names_are_stable() {
    const std::array<const char*, 9> states{
        "Idle", "CoarseApproach", "Preview", "FinalApproach",
        "Settling", "FinalPreview", "ReadyToSubmit", "Submitted",
        "Failed"};
    for (size_t i = 0; i < states.size(); ++i) {
        require(
            std::string(interaction::pick_assist_state_name(
                static_cast<interaction::PickAssistState>(i))) == states[i],
            "state diagnostic name changed");
    }
    const std::array<const char*, 10> reasons{
        "None", "Cancelled", "TargetUnavailable", "TargetChanged",
        "RuntimeChanged", "OutsideTravelEnvelope", "NoFeasibleEntry",
        "PreviewDeadline", "ArrivalDeadline", "FinalPreviewRejected"};
    for (size_t i = 0; i < reasons.size(); ++i) {
        require(
            std::string(interaction::pick_assist_reason_name(
                static_cast<interaction::PickAssistReason>(i))) == reasons[i],
            "reason diagnostic name changed");
    }
}

void test_constructor_rejects_unsafe_or_nonfinite_config() {
    const float nan = float_from_bits(0x7fc00001U);
    const float infinity = float_from_bits(0x7f800000U);
    interaction::PickAssistConfig config{};

    config.maximum_assisted_path_m = 1.0001F;
    require_invalid_config(config, "assisted path above 1.00 m was accepted");
    config = {};
    config.maximum_assisted_path_m = 0.0F;
    require_invalid_config(config, "zero assisted path was accepted");
    config = {};
    config.maximum_assisted_path_m = nan;
    require_invalid_config(config, "NaN assisted path was accepted");
    config = {};
    config.maximum_assisted_path_m = infinity;
    require_invalid_config(config, "infinite assisted path was accepted");

    config = {};
    config.reach_entry_distance_m = nan;
    require_invalid_config(config, "NaN entry distance was accepted");
    config = {};
    config.reach_entry_tolerance_m = infinity;
    require_invalid_config(config, "infinite entry tolerance was accepted");
    config = {};
    config.maximum_settle_position_error_m = 0.0F;
    require_invalid_config(config, "zero settle position bound was accepted");
    config = {};
    config.maximum_settle_displayed_speed_mps = nan;
    require_invalid_config(config, "NaN settle speed bound was accepted");

    config = {};
    config.arrival.slow_radius_m = nan;
    require_invalid_config(config, "NaN arrival slow radius was accepted");
    config = {};
    config.arrival.latch_position_error_m = 0.0F;
    require_invalid_config(config, "zero arrival position bound was accepted");
    config = {};
    config.arrival.latch_simulation_speed_mps = infinity;
    require_invalid_config(config, "infinite arrival speed bound was accepted");
    config = {};
    config.arrival.maximum_yaw_error_radians = nan;
    require_invalid_config(config, "NaN arrival yaw bound was accepted");
    config = {};
    config.arrival.minimum_standoff_m = 0.0F;
    require_invalid_config(config, "zero minimum standoff was accepted");
    config = {};
    config.arrival.maximum_standoff_m = infinity;
    require_invalid_config(config, "infinite maximum standoff was accepted");
    config = {};
    config.arrival.maximum_standoff_m =
        config.arrival.minimum_standoff_m;
    require_invalid_config(config, "empty standoff interval was accepted");

    config = {};
    config.required_settle_ticks = 0U;
    require_invalid_config(config, "zero settle tick count was accepted");
    config = {};
    config.maximum_preview_ticks = 0U;
    require_invalid_config(config, "zero preview tick count was accepted");
    config = {};
    config.maximum_arrival_ticks = 0U;
    require_invalid_config(config, "zero arrival tick count was accepted");

    config = {};
    config.maximum_assisted_path_m = 0.25F;
    const interaction::ControllerPickAssist smaller_cap(config);
    require(
        smaller_cap.diagnostics().state == interaction::PickAssistState::Idle,
        "valid smaller assisted-path cap was rejected");
}

void test_begin_latches_planar_routes_and_enforces_inclusive_envelope() {
    Scenario scenario;
    interaction::ControllerPickAssist assist;
    require(assist.begin(scenario.start), "valid route was rejected");
    require(
        assist.diagnostics().state ==
            interaction::PickAssistState::CoarseApproach,
        "begin did not enter CoarseApproach");
    require(assist.active() && assist.owns_manual_interact(),
        "CoarseApproach did not own the attempt");
    require(
        assist.diagnostics().target == scenario.start.target &&
        assist.diagnostics().affordance_id == scenario.start.affordance_id,
        "begin did not latch request identity");
    const vec3 entry = reach_entry_point(scenario);
    for (size_t i = 0; i < 2U; ++i) {
        const vec3 slot = scenario.start.slots.ordered[i].waypoint.position;
        const float expected = std::hypot(
            scenario.start.root_world.position.x - entry.x,
            scenario.start.root_world.position.z - entry.z) +
            std::hypot(slot.x - entry.x, slot.z - entry.z);
        require_near(
            assist.diagnostics().slot_route_lengths_m[i], expected,
            2.0e-5F, "begin computed a nonplanar route length");
        require(assist.diagnostics().slot_permitted[i],
            "valid default route was not permitted");
    }
    require_near(
        assist.diagnostics().route_length_m,
        std::min(
            assist.diagnostics().slot_route_lengths_m[0],
            assist.diagnostics().slot_route_lengths_m[1]),
        2.0e-5F,
        "preselection route was not the minimum permitted route");

    Scenario boundary;
    boundary.start.root_world.position = vec3(0.0F, 100.0F, 0.0F);
    boundary.start.reach_waypoint = {vec3(0.0F, -50.0F, 0.60F), quat()};
    boundary.start.slots.ordered[0].waypoint.position =
        vec3(0.0F, 90.0F, 1.00002F);
    boundary.start.slots.ordered[1].waypoint.position =
        vec3(0.0F, -90.0F, 1.001F);
    interaction::ControllerPickAssist boundary_assist;
    require(boundary_assist.begin(boundary.start),
        "1.00 m plus 2e-5 route tolerance was rejected");
    require(boundary_assist.diagnostics().slot_permitted[0],
        "inclusive tolerance route was not permitted");
    require(!boundary_assist.diagnostics().slot_permitted[1],
        "1.001 m route was permitted");

    boundary.start.slots.ordered[0].waypoint.position.z = 1.001F;
    interaction::ControllerPickAssist rejected;
    require(!rejected.begin(boundary.start),
        "two out-of-envelope routes were accepted");
    require(
        rejected.diagnostics().state == interaction::PickAssistState::Failed &&
        rejected.diagnostics().reason ==
            interaction::PickAssistReason::OutsideTravelEnvelope,
        "outside travel envelope did not fail visibly");
    require(!rejected.active(), "Failed route remained active");
    require(rejected.begin(scenario.start),
        "begin was not retryable from Failed");
}

void require_nonfinite_begin_rejected(
    Scenario& scenario,
    const char* message) {
    interaction::ControllerPickAssist assist;
    require(!assist.begin(scenario.start), message);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed &&
        assist.diagnostics().reason ==
            interaction::PickAssistReason::OutsideTravelEnvelope,
        "nonfinite begin did not fail closed in the travel envelope");
    require(
        finite_from_bits(assist.diagnostics().slot_route_lengths_m[0]) &&
        finite_from_bits(assist.diagnostics().slot_route_lengths_m[1]) &&
        finite_from_bits(assist.diagnostics().route_length_m),
        "nonfinite begin leaked nonfinite route diagnostics");
}

void test_begin_rejects_nonfinite_start_and_derived_routes() {
    const float nan = float_from_bits(0x7fc00001U);
    const float infinity = float_from_bits(0x7f800000U);
    const float largest = float_from_bits(0x7f7fffffU);
    {
        Scenario scenario;
        scenario.start.object_world.position.x = nan;
        require_nonfinite_begin_rejected(
            scenario, "nonfinite latched object pose was accepted");
    }
    {
        Scenario scenario;
        scenario.start.root_world.rotation.w = infinity;
        require_nonfinite_begin_rejected(
            scenario, "nonfinite start root was accepted");
    }
    {
        Scenario scenario;
        scenario.start.reach_waypoint.position.z = nan;
        require_nonfinite_begin_rejected(
            scenario, "nonfinite Reach waypoint was accepted");
    }
    {
        Scenario scenario;
        scenario.start.slots.ordered[0].waypoint.position.x = infinity;
        require_nonfinite_begin_rejected(
            scenario, "nonfinite slot waypoint was accepted");
    }
    {
        Scenario scenario;
        scenario.start.slots.ordered[0]
            .prospective_root.world_yaw_radians = nan;
        require_nonfinite_begin_rejected(
            scenario, "nonfinite prospective root was accepted at begin");
    }
    {
        Scenario scenario;
        scenario.start.slots.ordered[1].hand_score = nan;
        require_nonfinite_begin_rejected(
            scenario, "nonfinite slot hand score was accepted");
    }
    {
        Scenario scenario;
        scenario.start.slots.clearance_chord_m = infinity;
        require_nonfinite_begin_rejected(
            scenario, "nonfinite slot metadata was accepted");
    }
    {
        Scenario scenario;
        scenario.start.root_world.position.x = largest;
        scenario.start.reach_waypoint.position.x = -largest;
        require_nonfinite_begin_rejected(
            scenario, "overflowing derived route was accepted or exposed");
    }
}

void test_coarse_steering_requests_current_preview_and_freezes_slot() {
    Scenario scenario;
    interaction::ControllerPickAssist assist;
    require(assist.begin(scenario.start), "coarse fixture begin failed");
    scenario.observation.camera_azimuth = 0.50F * PIf;
    interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    vec3 world_direction =
        reach_entry_point(scenario) - scenario.start.root_world.position;
    world_direction.y = 0.0F;
    world_direction = normalize(world_direction);
    const vec3 expected_coarse = quat_inv_mul_vec3(
        quat_from_angle_axis(
            scenario.observation.camera_azimuth,
            vec3(0.0F, 1.0F, 0.0F)),
        world_direction);
    require(output.override_steering && !output.force_strafe,
        "CoarseApproach did not own ordinary steering");
    require_near(output.left_stick, expected_coarse, 2.0e-5F,
        "coarse command was not camera relative");
    require_near(output.right_stick, vec3(), 0.0F,
        "coarse command drove the facing stick");

    scenario.observation.displayed_root.position = reach_entry_point(scenario);
    output = assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Preview &&
        output.override_steering && output.needs_preview,
        "entry observation did not request Preview");
    require_near(output.left_stick, vec3(), 0.0F,
        "Preview did not hold zero steering");

    std::array<interaction::PickEntryPreview, 2> previews{};
    previews[1] = ready_preview(scenario.start.slots.ordered[1]);
    scenario.observation.previews = previews;
    scenario.observation.preview_snapshot_fingerprint = 90U;
    output = assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Preview &&
        assist.diagnostics().selected_slot == -1 && output.needs_preview,
        "stale preview snapshot changed selection");

    scenario.observation.preview_snapshot_fingerprint =
        scenario.observation.snapshot_fingerprint;
    previews[0].prospective_root =
        scenario.start.slots.ordered[0].prospective_root;
    previews[1].prospective_root =
        scenario.start.slots.ordered[0].prospective_root;
    scenario.observation.previews = previews;
    output = assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Preview &&
        assist.diagnostics().selected_slot == -1 && output.needs_preview,
        "wrong-slot initial preview provenance changed selection");
    previews[1] = ready_preview(scenario.start.slots.ordered[1]);
    scenario.observation.previews = previews;
    output = assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
            interaction::PickAssistState::FinalApproach &&
        assist.diagnostics().selected_slot == 1,
        "current eligible preview did not freeze its slot");
    require_near(
        assist.diagnostics().route_length_m,
        assist.diagnostics().slot_route_lengths_m[1],
        2.0e-5F,
        "selection did not freeze its route provenance");
    const vec3 expected_navigation = interaction::arrival_navigation_stick(
        scenario.start.slots.ordered[1].waypoint.position,
        scenario.observation.displayed_root.position,
        scenario.observation.camera_azimuth);
    vec3 target_forward = quat_mul_vec3(
        scenario.start.slots.ordered[1].waypoint.rotation,
        vec3(0.0F, 0.0F, 1.0F));
    target_forward.y = 0.0F;
    const vec3 expected_facing = interaction::arrival_facing_stick(
        target_forward, scenario.observation.camera_azimuth);
    require(output.override_steering && output.force_strafe,
        "FinalApproach transition did not own strafe steering");
    require_near(output.left_stick, expected_navigation, 2.0e-5F,
        "FinalApproach did not use arrival navigation helper");
    require_near(output.right_stick, expected_facing, 2.0e-5F,
        "FinalApproach did not use arrival facing helper");

    previews[0] = ready_preview(scenario.start.slots.ordered[0]);
    scenario.observation.previews = previews;
    assist.observe(scenario.observation);
    require(assist.diagnostics().selected_slot == 1,
        "later preview switched the frozen slot");
}

void test_preview_filters_asymmetric_caller_supplied_slot_permissions() {
    Scenario scenario;
    interaction::GraspAffordance unrelated =
        scenario.target.affordances.front();
    unrelated.id = 99U;
    scenario.target.affordances.push_back(unrelated);

    const vec3 entry = reach_entry_point(scenario);
    scenario.start.slots.ordered[0].waypoint.position =
        entry + vec3(0.0F, 0.0F, 2.0F);
    scenario.start.slots.ordered[0].prospective_root.world_x =
        scenario.start.slots.ordered[0].waypoint.position.x;
    scenario.start.slots.ordered[0].prospective_root.world_z =
        scenario.start.slots.ordered[0].waypoint.position.z;
    scenario.start.slots.ordered[0].hand_score = 100.0F;

    interaction::ControllerPickAssist assist;
    require(assist.begin(scenario.start),
        "one permitted asymmetric route was rejected");
    require(!assist.diagnostics().slot_permitted[0] &&
            assist.diagnostics().slot_permitted[1],
        "asymmetric routes produced wrong permission bits");
    scenario.observation.displayed_root.position = entry;
    interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Preview &&
        output.needs_preview,
        "caller-supplied asymmetric slots failed before Preview");

    scenario.observation.previews = ready_previews(scenario);
    output = assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
            interaction::PickAssistState::FinalApproach &&
        assist.diagnostics().selected_slot == 1,
        "unpermitted higher-score slot won preview selection");
    require(output.override_steering && output.force_strafe,
        "permitted slot did not begin FinalApproach");
}

void test_brake_latch_is_one_way_and_settle_uses_displayed_stability() {
    Scenario scenario;
    interaction::ControllerPickAssist assist;
    enter_final_approach(assist, scenario, 0U);
    const interaction::PickEntrySlot& slot =
        scenario.start.slots.ordered[0];
    scenario.observation.displayed_root = slot.waypoint;
    scenario.observation.simulation_velocity = vec3();
    scenario.observation.displayed_planar_speed_mps = 0.0F;
    interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Settling,
        "arrival-ready observation did not latch braking");
    require(assist.diagnostics().settle_ticks == 0U,
        "brake-latch observation counted as settle tick one");
    require(output.override_steering && output.stationary_constraint &&
            is_zero(output.left_stick) && is_zero(output.right_stick),
        "brake latch did not produce zero-stick stationary output");

    scenario.observation.displayed_root.position = reach_entry_point(scenario);
    scenario.observation.simulation_velocity = vec3(2.0F, 0.0F, 0.0F);
    scenario.observation.displayed_planar_speed_mps = 2.0F;
    output = assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Settling &&
        assist.diagnostics().settle_ticks == 0U,
        "post-brake drift re-enabled approach or counted settling");
    require(output.override_steering && output.stationary_constraint &&
            is_zero(output.left_stick) && is_zero(output.right_stick),
        "post-brake drift re-enabled steering");

    scenario.observation.displayed_root = slot.waypoint;
    scenario.observation.simulation_velocity = vec3(9.0F, 0.0F, 0.0F);
    scenario.observation.displayed_planar_speed_mps = 0.10F;
    assist.observe(scenario.observation);
    require(assist.diagnostics().settle_ticks == 1U,
        "displayed-speed boundary did not count as stable");
    scenario.observation.displayed_planar_speed_mps = 0.1001F;
    assist.observe(scenario.observation);
    require(assist.diagnostics().settle_ticks == 0U,
        "displayed speed above 0.10 m/s did not reset settling");

    vec3 radial = slot.waypoint.position - scenario.start.object_world.position;
    radial.y = 0.0F;
    const vec3 tangent = normalize(
        cross(vec3(0.0F, 1.0F, 0.0F), radial));
    scenario.observation.displayed_planar_speed_mps = 0.0F;
    scenario.observation.displayed_root = slot.waypoint;
    scenario.observation.displayed_root.position =
        slot.waypoint.position + 0.1501F * tangent;
    assist.observe(scenario.observation);
    require(assist.diagnostics().settle_ticks == 0U,
        "position error above 0.15 m did not reset settling");
    scenario.observation.displayed_root.position =
        slot.waypoint.position + 0.15F * tangent;
    assist.observe(scenario.observation);
    require(assist.diagnostics().settle_ticks == 1U,
        "position-error boundary did not count as stable");

    scenario.observation.displayed_root = slot.waypoint;
    scenario.observation.displayed_root.position =
        scenario.start.object_world.position + 0.46F * normalize(radial);
    assist.observe(scenario.observation);
    require(assist.diagnostics().settle_ticks == 0U,
        "standoff outside arrival contract did not reset settling");
    scenario.observation.displayed_root = slot.waypoint;
    scenario.observation.displayed_root.rotation = quat_from_angle_axis(
        21.0F * PIf / 180.0F, vec3(0.0F, 1.0F, 0.0F));
    assist.observe(scenario.observation);
    require(assist.diagnostics().settle_ticks == 0U,
        "yaw outside arrival contract did not reset settling");

    scenario.observation.displayed_root = slot.waypoint;
    for (uint32_t tick = 1U; tick <= 5U; ++tick) {
        output = assist.observe(scenario.observation);
        require(assist.diagnostics().settle_ticks == tick,
            "stable post-latch settle ticks were not consecutive");
        require(output.override_steering && output.stationary_constraint &&
                is_zero(output.left_stick) && is_zero(output.right_stick),
            "stable settle tick released zero-stick stationary output");
        if (tick < 5U) {
            require(
                assist.diagnostics().state ==
                    interaction::PickAssistState::Settling,
                "settling completed before five full post-latch ticks");
        }
    }
    require(
        assist.diagnostics().state ==
            interaction::PickAssistState::FinalPreview &&
        output.needs_preview,
        "five stable ticks did not request FinalPreview");
}

void test_final_preview_certifies_frozen_slot_and_submits_once() {
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        enter_final_preview(assist, scenario);
        scenario.observation.previews = ready_previews(scenario);
        scenario.observation.preview_snapshot_fingerprint = 90U;
        assist.observe(scenario.observation);
        require(
            assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::FinalPreviewRejected,
            "stale FinalPreview fingerprint was accepted");
    }
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        enter_final_preview(assist, scenario);
        std::array<interaction::PickEntryPreview, 2> previews =
            ready_previews(scenario);
        previews[0].prospective_root =
            scenario.start.slots.ordered[1].prospective_root;
        scenario.observation.previews = previews;
        assist.observe(scenario.observation);
        require(
            assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::FinalPreviewRejected,
            "wrong-slot FinalPreview provenance was accepted");
    }
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        enter_final_preview(assist, scenario);
        std::array<interaction::PickEntryPreview, 2> previews =
            ready_previews(scenario);
        previews[0].match_ready = false;
        scenario.observation.previews = previews;
        assist.observe(scenario.observation);
        require(
            assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().selected_slot == 0,
            "FinalPreview switched to the other eligible slot");
    }

    Scenario scenario;
    interaction::ControllerPickAssist assist;
    enter_final_preview(assist, scenario);
    std::array<interaction::PickEntryPreview, 2> successful_previews =
        ready_previews(scenario);
    successful_previews[0].feasible_entry_frame = 114;
    successful_previews[0].contact_frame = 139;
    successful_previews[0].total_cost = 8.25F;
    scenario.observation.previews = successful_previews;
    interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
            interaction::PickAssistState::ReadyToSubmit &&
        output.submit_interact,
        "certifying observation did not emit immediate submit pulse");
    require(output.override_steering && output.stationary_constraint,
        "submit pulse released braking before resolver consumption");
    require(assist.active() && assist.owns_manual_interact(),
        "ReadyToSubmit did not own the synthetic-submit window");
    {
        const auto& preview = assist.diagnostics().final_preview;
        require(
            preview.available && preview.all_preview_roots_finite &&
                preview.fingerprint_equal && preview.path_feasible &&
                preview.path_reason == interaction::Reason::None &&
                preview.match_ready &&
                preview.match_reason == interaction::Reason::None &&
                preview.prospective_root_equal &&
                preview.feasible_entry_frame == 114 &&
                preview.contact_frame == 139,
            "successful final-preview diagnostics were not captured");
        require_near(
            preview.total_cost,
            8.25F,
            1.0e-6F,
            "successful final-preview cost was not captured");
    }

    output = assist.observe(scenario.observation);
    require(!output.submit_interact,
        "ReadyToSubmit emitted more than one submit pulse");
    require(!assist.take_submission(0U).has_value(),
        "zero request id consumed submission");
    require(
        assist.diagnostics().state ==
            interaction::PickAssistState::ReadyToSubmit,
        "zero request id changed submission state");
    const std::optional<interaction::PickRequest> request =
        assist.take_submission(7U);
    require(request.has_value(), "nonzero request id did not submit");
    require(
        request->target == scenario.start.target &&
        request->affordance_id == scenario.start.affordance_id &&
        request->request_id == 7U,
        "submission did not preserve latched identity");
    require(
        assist.diagnostics().state == interaction::PickAssistState::Submitted &&
        !assist.active() && !assist.owns_manual_interact(),
        "submission did not become terminal and non-active");
    require(
        assist.diagnostics().final_preview.available &&
            assist.diagnostics().final_preview.feasible_entry_frame == 114 &&
            assist.diagnostics().final_preview.contact_frame == 139,
        "submission cleared the final-preview evidence");
    require(!assist.take_submission(8U).has_value(),
        "submission was consumed twice");

    scenario.target.state = interaction::ObjectState::Targeted;
    scenario.observation.runtime_state = interaction::RuntimeState::Preflight;
    output = assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Submitted &&
        assist.diagnostics().reason == interaction::PickAssistReason::None,
        "expected post-submit Targeted/Preflight transition failed");
    require(!output.override_steering && !output.submit_interact,
        "Submitted continued driving controller output");
    assist.cancel();
    require(
        assist.diagnostics().state == interaction::PickAssistState::Submitted,
        "cancel changed Submitted");
}

void test_final_preview_diagnostics_capture_rejection_and_retry_reset() {
    const float infinity = float_from_bits(0x7f800000U);
    Scenario scenario;
    interaction::ControllerPickAssist assist;
    enter_final_preview(assist, scenario);
    std::array<interaction::PickEntryPreview, 2> previews =
        ready_previews(scenario);
    previews[0].path_feasible = false;
    previews[0].path_reason = interaction::Reason::BlockedPath;
    previews[0].match_ready = false;
    previews[0].match_reason = interaction::Reason::PoorMatch;
    previews[0].prospective_root.world_x = std::nextafter(
        previews[0].prospective_root.world_x, infinity);
    previews[0].feasible_entry_frame = 114;
    previews[0].contact_frame = 139;
    previews[0].total_cost = 12.50F;
    scenario.observation.preview_snapshot_fingerprint = 90U;
    scenario.observation.previews = previews;

    assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::FinalPreviewRejected,
        "diagnostic rejection fixture was not rejected");
    {
        const auto& preview = assist.diagnostics().final_preview;
        require(
            preview.available && preview.all_preview_roots_finite &&
                !preview.fingerprint_equal && !preview.path_feasible &&
                preview.path_reason == interaction::Reason::BlockedPath &&
                !preview.match_ready &&
                preview.match_reason == interaction::Reason::PoorMatch &&
                !preview.prospective_root_equal &&
                preview.feasible_entry_frame == 114 &&
                preview.contact_frame == 139,
            "rejected final-preview facts were not captured");
        require_near(
            preview.total_cost,
            12.50F,
            1.0e-6F,
            "rejected final-preview cost was not captured");
    }

    scenario.observation.previews.reset();
    scenario.observation.preview_snapshot_fingerprint =
        scenario.observation.snapshot_fingerprint;
    require(assist.begin(scenario.start),
        "retry did not restart a failed diagnostic fixture");
    require_final_preview_diagnostics_cleared(assist.diagnostics());
}

void test_preview_root_provenance_rejects_one_bit_perturbations() {
    const float infinity = float_from_bits(0x7f800000U);
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        require(assist.begin(scenario.start),
            "exact initial provenance fixture begin failed");
        scenario.observation.displayed_root.position =
            reach_entry_point(scenario);
        assist.observe(scenario.observation);
        std::array<interaction::PickEntryPreview, 2> previews{};
        previews[0] = ready_preview(scenario.start.slots.ordered[0]);
        previews[0].prospective_root.world_x = std::nextafter(
            previews[0].prospective_root.world_x, infinity);
        scenario.observation.previews = previews;
        const interaction::PickAssistOutput output =
            assist.observe(scenario.observation);
        require(
            assist.diagnostics().state == interaction::PickAssistState::Preview &&
            assist.diagnostics().selected_slot == -1 && output.needs_preview,
            "one-bit initial preview root perturbation was consumed");
    }
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        enter_final_preview(assist, scenario);
        std::array<interaction::PickEntryPreview, 2> previews =
            ready_previews(scenario);
        previews[0].prospective_root.world_z = std::nextafter(
            previews[0].prospective_root.world_z, infinity);
        scenario.observation.previews = previews;
        assist.observe(scenario.observation);
        require(
            assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::FinalPreviewRejected,
            "one-bit final preview root perturbation was certified");
    }
}

void require_cancel_cleared(interaction::ControllerPickAssist& assist) {
    assist.cancel();
    const interaction::PickAssistDiagnostics& diagnostics =
        assist.diagnostics();
    require(
        diagnostics.state == interaction::PickAssistState::Idle &&
        diagnostics.reason == interaction::PickAssistReason::Cancelled,
        "cancel did not return pre-submit state to visible Idle");
    require(
        diagnostics.target == interaction::TargetHandle{} &&
        diagnostics.affordance_id == 0U &&
        diagnostics.selected_slot == -1 &&
        diagnostics.settle_ticks == 0U,
        "cancel did not clear latched identity/counters");
    require(
        diagnostics.slot_route_lengths_m == std::array<float, 2>{} &&
        diagnostics.slot_permitted == std::array<bool, 2>{} &&
        diagnostics.route_length_m == 0.0F &&
        diagnostics.root_error_m == 0.0F &&
        diagnostics.yaw_error_radians == 0.0F &&
        diagnostics.speed_mps == 0.0F,
        "cancel did not clear route/motion diagnostics");
    require_final_preview_diagnostics_cleared(diagnostics);
    require(!assist.active() && !assist.owns_manual_interact(),
        "cancelled assist remained active");
    require(!assist.observe({}).override_steering,
        "cancel did not release override on the same tick");
}

void test_cancel_clears_every_pre_submit_phase() {
    for (interaction::PickAssistState requested : {
             interaction::PickAssistState::CoarseApproach,
             interaction::PickAssistState::Preview,
             interaction::PickAssistState::FinalApproach,
             interaction::PickAssistState::Settling,
             interaction::PickAssistState::FinalPreview,
             interaction::PickAssistState::ReadyToSubmit}) {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        switch (requested) {
        case interaction::PickAssistState::CoarseApproach:
            require(assist.begin(scenario.start), "cancel fixture begin failed");
            break;
        case interaction::PickAssistState::Preview:
            require(assist.begin(scenario.start), "cancel fixture begin failed");
            scenario.observation.displayed_root.position =
                reach_entry_point(scenario);
            assist.observe(scenario.observation);
            break;
        case interaction::PickAssistState::FinalApproach:
            enter_final_approach(assist, scenario, 0U);
            break;
        case interaction::PickAssistState::Settling:
            enter_final_approach(assist, scenario, 0U);
            scenario.observation.displayed_root =
                scenario.start.slots.ordered[0].waypoint;
            assist.observe(scenario.observation);
            break;
        case interaction::PickAssistState::FinalPreview:
            enter_final_preview(assist, scenario);
            break;
        case interaction::PickAssistState::ReadyToSubmit:
            enter_final_preview(assist, scenario);
            scenario.observation.previews = ready_previews(scenario);
            assist.observe(scenario.observation);
            break;
        default:
            throw std::runtime_error("invalid cancel fixture state");
        }
        require(assist.diagnostics().state == requested,
            "cancel fixture reached wrong phase");
        require_cancel_cleared(assist);
    }

    Scenario scenario;
    scenario.start.root_world.position.z = -5.0F;
    interaction::ControllerPickAssist failed;
    require(!failed.begin(scenario.start),
        "failed cancel fixture unexpectedly began");
    require(failed.diagnostics().state == interaction::PickAssistState::Failed,
        "failed cancel fixture did not fail");
    require_cancel_cleared(failed);
}

void expect_observation_failure(
    Scenario& scenario,
    interaction::PickAssistReason reason,
    const char* message) {
    interaction::ControllerPickAssist assist;
    require(assist.begin(scenario.start), "failure fixture begin failed");
    assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed,
        message);
    require(assist.diagnostics().reason == reason,
        "observation failure reported wrong reason");
    require(!assist.take_submission(9U).has_value(),
        "failed observation produced submission");
}

void test_target_by_id_and_runtime_changes_fail_before_submission() {
    {
        Scenario scenario;
        scenario.observation.target = nullptr;
        expect_observation_failure(
            scenario, interaction::PickAssistReason::TargetUnavailable,
            "missing stable-ID target did not fail");
    }
    {
        Scenario scenario;
        scenario.target.handle.id += 1U;
        expect_observation_failure(
            scenario, interaction::PickAssistReason::TargetUnavailable,
            "different stable target ID did not fail unavailable");
    }
    {
        Scenario scenario;
        scenario.target.handle.generation += 1U;
        expect_observation_failure(
            scenario, interaction::PickAssistReason::TargetChanged,
            "target generation change did not fail");
    }
    {
        Scenario scenario;
        scenario.target.state = interaction::ObjectState::Targeted;
        expect_observation_failure(
            scenario, interaction::PickAssistReason::TargetChanged,
            "pre-submit target state change did not fail");
    }
    {
        Scenario scenario;
        scenario.target.object_world.position.z = std::nextafter(
            scenario.target.object_world.position.z, 1.0F);
        expect_observation_failure(
            scenario, interaction::PickAssistReason::TargetChanged,
            "one-bit target pose change did not fail");
    }
    {
        Scenario scenario;
        scenario.target.affordances.front().id += 1U;
        expect_observation_failure(
            scenario, interaction::PickAssistReason::TargetChanged,
            "affordance ID change did not fail");
    }
    {
        Scenario scenario;
        scenario.target.affordances.front().hand = interaction::Hand::Left;
        expect_observation_failure(
            scenario, interaction::PickAssistReason::TargetChanged,
            "affordance hand change did not fail");
    }
    {
        Scenario scenario;
        scenario.observation.runtime_state =
            interaction::RuntimeState::Preflight;
        expect_observation_failure(
            scenario, interaction::PickAssistReason::RuntimeChanged,
            "pre-submit runtime transition did not fail");
    }
}

void require_failed_not_ready(
    const interaction::ControllerPickAssist& assist,
    const char* message) {
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed,
        message);
    require(
        assist.diagnostics().state !=
            interaction::PickAssistState::ReadyToSubmit,
        "nonfinite evidence reached ReadyToSubmit");
    require(
        finite_from_bits(assist.diagnostics().root_error_m) &&
            finite_from_bits(assist.diagnostics().yaw_error_radians) &&
            finite_from_bits(assist.diagnostics().speed_mps),
        "nonfinite evidence escaped through failure diagnostics");
}

void test_nonfinite_target_observation_and_preview_fail_closed() {
    const float nan = float_from_bits(0x7fc00001U);
    const float infinity = float_from_bits(0x7f800000U);
    const float largest = float_from_bits(0x7f7fffffU);
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        require(assist.begin(scenario.start), "nonfinite target fixture failed");
        scenario.target.object_world.position.z = nan;
        assist.observe(scenario.observation);
        require_failed_not_ready(
            assist, "nonfinite target pose did not fail closed");
        require(
            assist.diagnostics().reason ==
                interaction::PickAssistReason::TargetChanged,
            "nonfinite target pose reported wrong reason");
    }
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        require(assist.begin(scenario.start), "nonfinite root fixture failed");
        scenario.observation.displayed_root.position.x = nan;
        assist.observe(scenario.observation);
        require_failed_not_ready(
            assist, "nonfinite displayed root did not fail closed");
    }
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        require(assist.begin(scenario.start), "nonfinite camera fixture failed");
        scenario.observation.camera_azimuth = infinity;
        assist.observe(scenario.observation);
        require_failed_not_ready(
            assist, "nonfinite camera azimuth did not fail closed");
    }
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        require(assist.begin(scenario.start), "initial preview fixture failed");
        scenario.observation.displayed_root.position =
            reach_entry_point(scenario);
        assist.observe(scenario.observation);
        std::array<interaction::PickEntryPreview, 2> previews =
            ready_previews(scenario);
        previews[0].prospective_root.world_x = nan;
        scenario.observation.previews = previews;
        assist.observe(scenario.observation);
        require_failed_not_ready(
            assist, "nonfinite initial preview root did not fail closed");
    }
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        enter_final_approach(assist, scenario, 0U);
        scenario.observation.displayed_root =
            scenario.start.slots.ordered[0].waypoint;
        scenario.observation.simulation_velocity.x = nan;
        assist.observe(scenario.observation);
        require_failed_not_ready(
            assist, "nonfinite simulation velocity did not fail closed");
    }
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        enter_final_approach(assist, scenario, 0U);
        scenario.observation.displayed_root =
            scenario.start.slots.ordered[0].waypoint;
        assist.observe(scenario.observation);
        scenario.observation.displayed_planar_speed_mps = nan;
        assist.observe(scenario.observation);
        require_failed_not_ready(
            assist, "nonfinite displayed settle speed did not fail closed");
    }
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        enter_final_approach(assist, scenario, 0U);
        scenario.observation.displayed_root =
            scenario.start.slots.ordered[0].waypoint;
        assist.observe(scenario.observation);
        scenario.observation.displayed_root.rotation.w = infinity;
        assist.observe(scenario.observation);
        require_failed_not_ready(
            assist, "nonfinite displayed yaw evidence did not fail closed");
    }
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        enter_final_approach(assist, scenario, 0U);
        scenario.observation.displayed_root =
            scenario.start.slots.ordered[0].waypoint;
        scenario.observation.displayed_root.rotation =
            quat(largest, largest, largest, -largest);
        assist.observe(scenario.observation);
        require_failed_not_ready(
            assist, "overflowing derived yaw evidence did not fail closed");
    }
    {
        Scenario scenario;
        scenario.start.object_world.position.x = -largest;
        scenario.target.object_world = scenario.start.object_world;
        scenario.observation.target = &scenario.target;
        interaction::ControllerPickAssist assist;
        enter_final_approach(assist, scenario, 0U);
        scenario.observation.displayed_root.position.x = largest;
        assist.observe(scenario.observation);
        require_failed_not_ready(
            assist, "overflowing derived arrival metric did not fail closed");
    }
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        enter_final_preview(assist, scenario);
        std::array<interaction::PickEntryPreview, 2> previews =
            ready_previews(scenario);
        previews[0].prospective_root.world_yaw_radians = infinity;
        scenario.observation.previews = previews;
        assist.observe(scenario.observation);
        require_failed_not_ready(
            assist, "nonfinite final preview root did not fail closed");
        require(
            assist.diagnostics().reason ==
                interaction::PickAssistReason::FinalPreviewRejected,
            "nonfinite final preview reported wrong reason");
        require(
            assist.diagnostics().final_preview.available &&
                !assist.diagnostics().final_preview
                     .all_preview_roots_finite &&
                !assist.diagnostics().final_preview
                     .prospective_root_equal,
            "nonfinite final preview facts were not captured before failure");
    }
}

void test_preview_and_arrival_deadlines_fail_without_submission() {
    {
        Scenario scenario;
        interaction::ControllerPickAssist assist;
        require(assist.begin(scenario.start), "no-feasible fixture begin failed");
        scenario.observation.displayed_root.position =
            reach_entry_point(scenario);
        assist.observe(scenario.observation);
        std::array<interaction::PickEntryPreview, 2> previews =
            ready_previews(scenario);
        previews[0].path_feasible = false;
        previews[0].match_ready = false;
        previews[1].path_feasible = false;
        previews[1].match_ready = false;
        scenario.observation.previews = previews;
        assist.observe(scenario.observation);
        require(
            assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::NoFeasibleEntry,
            "two current infeasible permitted previews did not fail");
        require(!assist.take_submission(4U).has_value(),
            "no-feasible failure produced submission");
    }
    {
        interaction::PickAssistConfig config{};
        config.maximum_preview_ticks = 3U;
        Scenario scenario;
        interaction::ControllerPickAssist assist(config);
        require(assist.begin(scenario.start), "preview deadline begin failed");
        scenario.observation.displayed_root.position =
            reach_entry_point(scenario);
        assist.observe(scenario.observation);
        std::array<interaction::PickEntryPreview, 2> previews =
            ready_previews(scenario);
        previews[0].match_ready = false;
        previews[1].match_ready = false;
        scenario.observation.previews = previews;
        for (uint32_t tick = 1U; tick <= 3U; ++tick) {
            assist.observe(scenario.observation);
            if (tick < 3U) {
                require(
                    assist.diagnostics().state ==
                        interaction::PickAssistState::Preview,
                    "preview deadline fired early");
            }
        }
        require(
            assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::PreviewDeadline,
            "preview deadline did not fail visibly");
        require(!assist.take_submission(4U).has_value(),
            "preview deadline produced submission");
    }
    {
        interaction::PickAssistConfig config{};
        config.maximum_arrival_ticks = 3U;
        Scenario scenario;
        interaction::ControllerPickAssist assist(config);
        enter_final_approach(assist, scenario, 0U);
        scenario.observation.displayed_root.position =
            reach_entry_point(scenario);
        for (uint32_t tick = 1U; tick <= 3U; ++tick) {
            assist.observe(scenario.observation);
            if (tick < 3U) {
                require(
                    assist.diagnostics().state ==
                        interaction::PickAssistState::FinalApproach,
                    "arrival deadline fired early");
            }
        }
        require(
            assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::ArrivalDeadline,
            "FinalApproach deadline did not fail visibly");
    }
    {
        interaction::PickAssistConfig config{};
        config.required_settle_ticks = 1U;
        config.maximum_arrival_ticks = 3U;
        Scenario scenario;
        interaction::ControllerPickAssist assist(config);
        enter_final_approach(assist, scenario, 0U);
        scenario.observation.displayed_root =
            scenario.start.slots.ordered[0].waypoint;
        assist.observe(scenario.observation);
        require(
            assist.diagnostics().state == interaction::PickAssistState::Settling,
            "cross-phase deadline did not latch Settling");
        assist.observe(scenario.observation);
        require(
            assist.diagnostics().state ==
                interaction::PickAssistState::FinalPreview,
            "cross-phase deadline did not reach FinalPreview");
        scenario.observation.previews.reset();
        assist.observe(scenario.observation);
        require(
            assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::ArrivalDeadline,
            "arrival deadline did not advance through all three phases");
        require(!assist.take_submission(6U).has_value(),
            "arrival deadline produced submission");
    }
}

void test_final_preview_deadline_precedes_valid_certification() {
    {
        interaction::PickAssistConfig config{};
        config.required_settle_ticks = 1U;
        config.maximum_arrival_ticks = 3U;
        Scenario scenario;
        interaction::ControllerPickAssist assist(config);
        enter_final_approach(assist, scenario, 0U);
        scenario.observation.displayed_root =
            scenario.start.slots.ordered[0].waypoint;
        assist.observe(scenario.observation);
        assist.observe(scenario.observation);
        require(
            assist.diagnostics().state ==
                interaction::PickAssistState::FinalPreview,
            "deadline-boundary fixture did not reach FinalPreview");
        scenario.observation.previews = ready_previews(scenario);
        const interaction::PickAssistOutput output =
            assist.observe(scenario.observation);
        require(
            assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::ArrivalDeadline,
            "valid preview certified on the exact arrival deadline tick");
        require(!output.submit_interact &&
                !assist.take_submission(12U).has_value(),
            "deadline-boundary preview produced a submission");
    }
    {
        interaction::PickAssistConfig config{};
        config.required_settle_ticks = 1U;
        config.maximum_arrival_ticks = 4U;
        Scenario scenario;
        interaction::ControllerPickAssist assist(config);
        enter_final_approach(assist, scenario, 0U);
        scenario.observation.displayed_root =
            scenario.start.slots.ordered[0].waypoint;
        assist.observe(scenario.observation);
        assist.observe(scenario.observation);
        scenario.observation.previews = ready_previews(scenario);
        const interaction::PickAssistOutput output =
            assist.observe(scenario.observation);
        require(
            assist.diagnostics().state ==
                interaction::PickAssistState::ReadyToSubmit &&
            output.submit_interact,
            "valid preview before arrival deadline was rejected");
    }
}

}  // namespace

int main() {
    try {
        test_idle_does_not_override_input();
        test_begin_selects_and_freezes_one_authored_slot();
        test_slot_approach_observe_is_inert_before_legacy_preflight();
        test_begin_maps_every_aggregate_no_winner_reason();
        test_slot_reason_mapping_is_exhaustive_and_same_named();
        test_failed_begin_can_immediately_begin_a_valid_attempt();
        test_diagnostic_names_are_stable();
        test_constructor_rejects_unsafe_or_nonfinite_config();
        test_begin_latches_planar_routes_and_enforces_inclusive_envelope();
        test_begin_rejects_nonfinite_start_and_derived_routes();
        test_coarse_steering_requests_current_preview_and_freezes_slot();
        test_preview_filters_asymmetric_caller_supplied_slot_permissions();
        test_brake_latch_is_one_way_and_settle_uses_displayed_stability();
        test_final_preview_certifies_frozen_slot_and_submits_once();
        test_final_preview_diagnostics_capture_rejection_and_retry_reset();
        test_preview_root_provenance_rejects_one_bit_perturbations();
        test_cancel_clears_every_pre_submit_phase();
        test_target_by_id_and_runtime_changes_fail_before_submission();
        test_nonfinite_target_observation_and_preview_fail_closed();
        test_preview_and_arrival_deadlines_fail_without_submission();
        test_final_preview_deadline_precedes_valid_certification();
        std::cout << "interaction_pick_assist tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "interaction_pick_assist test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
