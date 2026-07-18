#include "interaction_pick_assist.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

#define DEFINE_PUBLIC_MEMBER_TRAIT(trait_name, member_name) \
    template <typename Type, typename = void> \
    struct trait_name : std::false_type {}; \
    template <typename Type> \
    struct trait_name< \
        Type, \
        std::void_t<decltype(&Type::member_name)>> : std::true_type {};

DEFINE_PUBLIC_MEMBER_TRAIT(
    has_reach_entry_distance_m,
    reach_entry_distance_m)
DEFINE_PUBLIC_MEMBER_TRAIT(
    has_reach_entry_tolerance_m,
    reach_entry_tolerance_m)
DEFINE_PUBLIC_MEMBER_TRAIT(
    has_maximum_preview_ticks,
    maximum_preview_ticks)
DEFINE_PUBLIC_MEMBER_TRAIT(has_start_target, target)
DEFINE_PUBLIC_MEMBER_TRAIT(has_start_hand, hand)
DEFINE_PUBLIC_MEMBER_TRAIT(has_start_object_world, object_world)
DEFINE_PUBLIC_MEMBER_TRAIT(has_start_reach_waypoint, reach_waypoint)
DEFINE_PUBLIC_MEMBER_TRAIT(has_start_slots, slots)
DEFINE_PUBLIC_MEMBER_TRAIT(has_observation_previews, previews)
DEFINE_PUBLIC_MEMBER_TRAIT(has_diagnostics_selected_slot, selected_slot)
DEFINE_PUBLIC_MEMBER_TRAIT(
    has_diagnostics_slot_route_lengths_m,
    slot_route_lengths_m)
DEFINE_PUBLIC_MEMBER_TRAIT(
    has_diagnostics_slot_permitted,
    slot_permitted)

#undef DEFINE_PUBLIC_MEMBER_TRAIT

template <typename Type, typename = void>
struct has_one_argument_begin : std::false_type {};

template <typename Type>
struct has_one_argument_begin<
    Type,
    std::void_t<decltype(static_cast<bool (Type::*)(
        const interaction::PickAssistStart&)>(&Type::begin))>>
    : std::true_type {};

static_assert(
    !has_reach_entry_distance_m<interaction::PickAssistConfig>::value,
    "PickAssistConfig::reach_entry_distance_m must be absent");
static_assert(
    !has_reach_entry_tolerance_m<interaction::PickAssistConfig>::value,
    "PickAssistConfig::reach_entry_tolerance_m must be absent");
static_assert(
    !has_maximum_preview_ticks<interaction::PickAssistConfig>::value,
    "PickAssistConfig::maximum_preview_ticks must be absent");
static_assert(
    !has_start_target<interaction::PickAssistStart>::value,
    "PickAssistStart::target must be absent");
static_assert(
    !has_start_hand<interaction::PickAssistStart>::value,
    "PickAssistStart::hand must be absent");
static_assert(
    !has_start_object_world<interaction::PickAssistStart>::value,
    "PickAssistStart::object_world must be absent");
static_assert(
    !has_start_reach_waypoint<interaction::PickAssistStart>::value,
    "PickAssistStart::reach_waypoint must be absent");
static_assert(
    !has_start_slots<interaction::PickAssistStart>::value,
    "PickAssistStart::slots must be absent");
static_assert(
    !has_observation_previews<interaction::PickAssistObservation>::value,
    "PickAssistObservation::previews must be absent");
static_assert(
    !has_diagnostics_selected_slot<
        interaction::PickAssistDiagnostics>::value,
    "PickAssistDiagnostics::selected_slot must be absent");
static_assert(
    !has_diagnostics_slot_route_lengths_m<
        interaction::PickAssistDiagnostics>::value,
    "PickAssistDiagnostics::slot_route_lengths_m must be absent");
static_assert(
    !has_diagnostics_slot_permitted<
        interaction::PickAssistDiagnostics>::value,
    "PickAssistDiagnostics::slot_permitted must be absent");
static_assert(
    !has_one_argument_begin<interaction::ControllerPickAssist>::value,
    "ControllerPickAssist::begin(start) must be absent");

static_assert(
    static_cast<uint8_t>(interaction::PickAssistState::Idle) == 0U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistState::SlotApproach) == 1U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistState::Settling) == 2U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistState::FinalPreview) == 3U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistState::ReadyToSubmit) == 4U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistState::Submitted) == 5U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistState::Failed) == 6U);

static_assert(
    static_cast<uint8_t>(interaction::PickAssistReason::None) == 0U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistReason::Cancelled) == 1U);
static_assert(
    static_cast<uint8_t>(
        interaction::PickAssistReason::TargetUnavailable) == 2U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistReason::TargetChanged) == 3U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistReason::SlotChanged) == 4U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistReason::RuntimeChanged) == 5U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistReason::NoAuthoredSlot) == 6U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistReason::InvalidGeometry) == 7U);
static_assert(
    static_cast<uint8_t>(
        interaction::PickAssistReason::OutsideTravelEnvelope) == 8U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistReason::TableBlocked) == 9U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistReason::ObstacleBlocked) ==
        10U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistReason::AllSlotsBlocked) ==
        11U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistReason::ArrivalDeadline) ==
        12U);
static_assert(
    static_cast<uint8_t>(interaction::PickAssistReason::PoorMatch) == 13U);
static_assert(
    static_cast<uint8_t>(
        interaction::PickAssistReason::FinalPreviewRejected) == 14U);

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

float float_from_bits(uint32_t bits) {
    float value = 0.0F;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
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

bool same_float_bits_exact(float left, float right) {
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

bool same_pick_entry_root_bits_exact(
    interaction::PickEntryRoot left,
    interaction::PickEntryRoot right) {
    return same_float_bits_exact(left.world_x, right.world_x) &&
        same_float_bits_exact(left.world_z, right.world_z) &&
        same_float_bits_exact(
            left.world_yaw_radians, right.world_yaw_radians);
}

bool same_vec3_bits_exact(vec3 left, vec3 right) {
    return same_float_bits_exact(left.x, right.x) &&
        same_float_bits_exact(left.y, right.y) &&
        same_float_bits_exact(left.z, right.z);
}

bool same_transform_bits_exact(
    interaction::Transform left,
    interaction::Transform right) {
    return same_float_bits_exact(left.position.x, right.position.x) &&
        same_float_bits_exact(left.position.y, right.position.y) &&
        same_float_bits_exact(left.position.z, right.position.z) &&
        same_float_bits_exact(left.rotation.w, right.rotation.w) &&
        same_float_bits_exact(left.rotation.x, right.rotation.x) &&
        same_float_bits_exact(left.rotation.y, right.rotation.y) &&
        same_float_bits_exact(left.rotation.z, right.rotation.z);
}

void require_frozen_slot_provenance_unchanged(
    const interaction::PickAssistDiagnostics& before,
    const interaction::PickAssistDiagnostics& after,
    const char* message) {
    require(
        before.slot_selection.selected_index.has_value(),
        "frozen provenance fixture had no selected index");
    const size_t selected = *before.slot_selection.selected_index;
    require(
        selected < before.slot_selection.ordered.size() &&
            selected < after.slot_selection.ordered.size() &&
            after.selected_slot_id == before.selected_slot_id &&
            after.slot_selection.selected_index ==
                before.slot_selection.selected_index &&
            after.slot_selection.reason == before.slot_selection.reason &&
            after.slot_selection.ordered.size() ==
                before.slot_selection.ordered.size() &&
            after.slot_selection.ordered[selected].id ==
                before.slot_selection.ordered[selected].id &&
            same_transform_bits_exact(
                after.slot_selection.ordered[selected].root_world,
                before.slot_selection.ordered[selected].root_world) &&
            same_float_bits_exact(
                after.route_length_m,
                before.route_length_m) &&
            same_float_bits_exact(
                after.slot_selection.ordered[selected].route_length_m,
                before.slot_selection.ordered[selected].route_length_m) &&
            after.slot_selection.ordered[selected].route_millimetres ==
                before.slot_selection.ordered[selected].route_millimetres,
        message);
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

struct FrozenSlotScenario {
    interaction::InteractionTarget target = make_frozen_slot_target();
    interaction::PickAssistStart start = make_frozen_slot_start(target);
    interaction::PickAssistObservation observation{};

    FrozenSlotScenario() {
        start.root_world = {vec3(-0.80F, 0.0F, 0.50F), quat()};
        observation.target = &target;
        observation.displayed_root = start.root_world;
    }
};

bool is_zero(vec3 value) {
    return value.x == 0.0F && value.y == 0.0F && value.z == 0.0F;
}

void require_zero_pick_assist_output(
    const interaction::PickAssistOutput& output,
    const char* message) {
    require(
        !output.override_steering && is_zero(output.left_stick) &&
            is_zero(output.right_stick) && !output.force_strafe &&
            !output.stationary_constraint && !output.needs_preview &&
            !output.preview_root.has_value() && !output.submit_interact,
        message);
}

void require_stationary_preview_request(
    const interaction::PickAssistOutput& output,
    interaction::PickEntryRoot expected_root,
    const char* message) {
    require(
        output.override_steering && output.force_strafe &&
            output.stationary_constraint &&
            is_zero(output.left_stick) && is_zero(output.right_stick) &&
            output.needs_preview && output.preview_root.has_value() &&
            same_pick_entry_root_bits_exact(
                *output.preview_root, expected_root) &&
            !output.submit_interact,
        message);
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

void test_slot_approach_emits_far_camera_relative_steering() {
    const auto same_float_bits = [](float left, float right) {
        return std::memcmp(&left, &right, sizeof(left)) == 0;
    };
    const auto same_vec3_bits = [&](vec3 left, vec3 right) {
        return same_float_bits(left.x, right.x) &&
            same_float_bits(left.y, right.y) &&
            same_float_bits(left.z, right.z);
    };
    const auto same_quat_bits = [&](quat left, quat right) {
        return same_float_bits(left.w, right.w) &&
            same_float_bits(left.x, right.x) &&
            same_float_bits(left.y, right.y) &&
            same_float_bits(left.z, right.z);
    };
    const auto same_transform_bits = [&](
        interaction::Transform left,
        interaction::Transform right) {
        return same_vec3_bits(left.position, right.position) &&
            same_quat_bits(left.rotation, right.rotation);
    };
    const auto same_mapped_slot = [&](
        const interaction::MappedPickSlot& left,
        const interaction::MappedPickSlot& right) {
        return left.id == right.id &&
            same_transform_bits(left.root_world, right.root_world) &&
            same_float_bits(left.route_length_m, right.route_length_m) &&
            same_float_bits(
                left.heading_change_radians,
                right.heading_change_radians) &&
            same_float_bits(
                left.object_origin_distance_m,
                right.object_origin_distance_m) &&
            same_float_bits(
                left.object_bounds_center_distance_m,
                right.object_bounds_center_distance_m) &&
            left.route_millimetres == right.route_millimetres &&
            left.heading_milliradians == right.heading_milliradians &&
            left.reason == right.reason &&
            left.obstacle_index == right.obstacle_index;
    };
    const auto same_frozen_provenance = [&](
        const interaction::PickAssistDiagnostics& left,
        const interaction::PickAssistDiagnostics& right) {
        if (left.selected_slot_id != right.selected_slot_id ||
            left.slot_selection.selected_index !=
                right.slot_selection.selected_index ||
            left.slot_selection.reason != right.slot_selection.reason ||
            left.slot_selection.ordered.size() !=
                right.slot_selection.ordered.size() ||
            !left.slot_selection.selected_index.has_value() ||
            !same_float_bits(left.route_length_m, right.route_length_m)) {
            return false;
        }
        const size_t selected = *left.slot_selection.selected_index;
        return selected < left.slot_selection.ordered.size() &&
            same_mapped_slot(
                left.slot_selection.ordered[selected],
                right.slot_selection.ordered[selected]);
    };

    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist;

    require(
        assist.begin(scenario.start, &scenario.target),
        "far-approach begin failed");
    require(
        assist.diagnostics().slot_selection.selected_index.has_value(),
        "far-approach begin did not retain its selected index");
    const interaction::MappedPickSlot frozen_slot =
        assist.diagnostics().slot_selection.ordered[
            *assist.diagnostics().slot_selection.selected_index];
    require(
        frozen_slot.id == 9U && frozen_slot.route_length_m == 0.80F,
        "far-approach fixture did not select the exact 0.80 m slot");

    const interaction::InteractionTarget target_before = scenario.target;
    const interaction::Transform target_pose_before =
        scenario.target.object_world;
    const interaction::Transform start_pose_before =
        scenario.start.root_world;
    const interaction::Transform displayed_root_before =
        scenario.observation.displayed_root;
    const interaction::PickAssistDiagnostics diagnostics_before =
        assist.diagnostics();
    const interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    require(
        output.override_steering &&
            output.left_stick.x == 1.0F &&
            output.left_stick.y == 0.0F &&
            output.left_stick.z == 0.0F,
        "far SlotApproach did not emit ordinary camera-relative steering");
    require(
        !output.needs_preview && is_zero(output.right_stick) &&
            !output.force_strafe && !output.stationary_constraint &&
            !output.submit_interact,
        "far SlotApproach emitted preview, facing, strafe, stationary, or submission output");
    require(
        interaction::same_interaction_target_snapshot(
            scenario.target, target_before) &&
            interaction::same_interaction_target_snapshot(
                scenario.start.target_snapshot,
                target_before) &&
            same_transform_bits(
                scenario.target.object_world, target_pose_before) &&
            same_transform_bits(scenario.start.root_world, start_pose_before) &&
            same_transform_bits(
                scenario.observation.displayed_root,
                displayed_root_before),
        "far SlotApproach mutated an input target or pose");
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::SlotApproach &&
            same_frozen_provenance(
                assist.diagnostics(), diagnostics_before),
        "far SlotApproach changed its frozen slot provenance");

    FrozenSlotScenario rotated_scenario;
    rotated_scenario.observation.camera_azimuth = 0.50F * PIf;
    interaction::ControllerPickAssist rotated_assist;
    require(
        rotated_assist.begin(
            rotated_scenario.start, &rotated_scenario.target),
        "rotated far-approach begin failed");
    const interaction::PickAssistDiagnostics rotated_before =
        rotated_assist.diagnostics();
    require(
        rotated_before.route_length_m == 0.80F &&
            same_frozen_provenance(rotated_before, diagnostics_before),
        "rotated far-approach fixture did not freeze the same exact 0.80 m slot");
    const interaction::InteractionTarget rotated_target_before =
        rotated_scenario.target;
    const interaction::Transform rotated_target_pose_before =
        rotated_scenario.target.object_world;
    const interaction::Transform rotated_start_pose_before =
        rotated_scenario.start.root_world;
    const interaction::Transform rotated_displayed_root_before =
        rotated_scenario.observation.displayed_root;

    const interaction::PickAssistOutput rotated_output =
        rotated_assist.observe(rotated_scenario.observation);
    const vec3 expected_rotated = quat_inv_mul_vec3(
        quat_from_angle_axis(
            rotated_scenario.observation.camera_azimuth,
            vec3(0.0F, 1.0F, 0.0F)),
        vec3(1.0F, 0.0F, 0.0F));
    require(
        rotated_output.override_steering &&
            !same_vec3_bits(rotated_output.left_stick, output.left_stick),
        "camera azimuth did not change far camera-relative stick coordinates");
    require_near(
        rotated_output.left_stick,
        expected_rotated,
        2.0e-5F,
        "rotated far SlotApproach command was not camera relative");
    require(
        !rotated_output.needs_preview &&
            is_zero(rotated_output.right_stick) &&
            !rotated_output.force_strafe &&
            !rotated_output.stationary_constraint &&
            !rotated_output.submit_interact,
        "rotated far SlotApproach changed output beyond left-stick coordinates");
    require(
        interaction::same_interaction_target_snapshot(
            rotated_scenario.target, rotated_target_before) &&
            interaction::same_interaction_target_snapshot(
                rotated_scenario.start.target_snapshot,
                rotated_target_before) &&
            same_transform_bits(
                rotated_scenario.target.object_world,
                rotated_target_pose_before) &&
            same_transform_bits(
                rotated_scenario.start.root_world,
                rotated_start_pose_before) &&
            same_transform_bits(
                rotated_scenario.observation.displayed_root,
                rotated_displayed_root_before),
        "rotated far SlotApproach mutated an input target or pose");
    require(
        rotated_assist.diagnostics().state ==
                interaction::PickAssistState::SlotApproach &&
            same_frozen_provenance(
                rotated_assist.diagnostics(), rotated_before) &&
            same_frozen_provenance(
                rotated_assist.diagnostics(), assist.diagnostics()),
        "camera azimuth changed frozen slot ID, index, root, or route provenance");
}

void test_slot_approach_runtime_change_precedes_target_and_metrics() {
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist;

    require(
        assist.begin(scenario.start, &scenario.target),
        "runtime-precedence fixture begin failed");
    const interaction::PickAssistDiagnostics frozen_before =
        assist.diagnostics();
    scenario.observation.runtime_state =
        interaction::RuntimeState::Preflight;
    scenario.observation.target = nullptr;
    scenario.observation.displayed_root.position.x =
        float_from_bits(0x7fc00001U);

    const interaction::PickAssistOutput failed_output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::RuntimeChanged,
        "runtime precedence did not fail SlotApproach with RuntimeChanged");
    require_zero_pick_assist_output(
        failed_output,
        "runtime precedence emitted output while failing");
    require(
        !assist.active() && !assist.owns_manual_interact() &&
            !assist.take_submission(301U).has_value(),
        "runtime precedence retained ownership or submitted");
    require_frozen_slot_provenance_unchanged(
        frozen_before,
        assist.diagnostics(),
        "runtime precedence changed frozen slot provenance");

    const interaction::PickAssistOutput terminal_output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::RuntimeChanged,
        "runtime precedence failure was not terminal and stable");
    require_zero_pick_assist_output(
        terminal_output,
        "stable runtime failure emitted output");
    require(
        !assist.active() && !assist.owns_manual_interact() &&
            !assist.take_submission(302U).has_value(),
        "stable runtime failure reacquired ownership or submitted");
    require_frozen_slot_provenance_unchanged(
        frozen_before,
        assist.diagnostics(),
        "stable runtime failure changed frozen slot provenance");
}

using FrozenSlotMutation = void (*)(FrozenSlotScenario&);

struct FrozenSlotObservationFailureCase {
    const char* name;
    interaction::PickAssistReason expected_reason;
    FrozenSlotMutation mutate;
};

interaction::GraspInteractionSlot& selected_authored_slot(
    FrozenSlotScenario& scenario) {
    std::vector<interaction::GraspInteractionSlot>& slots =
        scenario.target.affordances.front().interaction_slots;
    const auto selected = std::find_if(
        slots.begin(), slots.end(), [](const auto& slot) {
            return slot.id == 9U;
        });
    require(
        selected != slots.end(),
        "identity fixture had no selected authored slot ID 9");
    return *selected;
}

void expect_frozen_slot_observation_failure(
    const FrozenSlotObservationFailureCase& test_case) {
    const auto require_case = [&](bool condition, const char* detail) {
        if (!condition) {
            throw std::runtime_error(
                std::string(test_case.name) + ": " + detail);
        }
    };

    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist;
    require_case(
        assist.begin(scenario.start, &scenario.target),
        "fixture begin failed");
    const interaction::PickAssistDiagnostics frozen_before =
        assist.diagnostics();
    require_case(
        frozen_before.state ==
                interaction::PickAssistState::SlotApproach &&
            frozen_before.reason == interaction::PickAssistReason::None &&
            frozen_before.selected_slot_id == 9U &&
            frozen_before.slot_selection.selected_index.has_value() &&
            frozen_before.settle_ticks == 0U &&
            same_float_bits_exact(
                frozen_before.assisted_travel_m, 0.0F),
        "fixture did not freeze selected slot ID 9");

    test_case.mutate(scenario);
    const interaction::PickAssistOutput failed_output =
        assist.observe(scenario.observation);
    const interaction::PickAssistDiagnostics failed =
        assist.diagnostics();
    require_case(
        failed.state == interaction::PickAssistState::Failed,
        "observation did not enter Failed");
    require_case(
        failed.reason == test_case.expected_reason,
        "observation reported the wrong stable reason");
    if (test_case.expected_reason ==
        interaction::PickAssistReason::SlotChanged) {
        require_case(
            failed.reason != interaction::PickAssistReason::TargetChanged,
            "slot identity change was classified as TargetChanged");
    }
    const std::string failed_output_message =
        std::string(test_case.name) +
        ": failing observation emitted output";
    require_zero_pick_assist_output(
        failed_output, failed_output_message.c_str());
    require_case(
        !assist.active() && !assist.owns_manual_interact() &&
            !assist.take_submission(401U).has_value(),
        "failure retained ownership or produced a submission");
    const std::string failed_provenance_message =
        std::string(test_case.name) +
        ": failure changed frozen slot ID, root, or index";
    require_frozen_slot_provenance_unchanged(
        frozen_before,
        failed,
        failed_provenance_message.c_str());
    require_case(
        failed.settle_ticks == frozen_before.settle_ticks &&
            same_float_bits_exact(failed.assisted_travel_m, 0.0F),
        "failure advanced settling or accumulated assisted travel");

    const interaction::PickAssistOutput terminal_output =
        assist.observe(scenario.observation);
    const interaction::PickAssistDiagnostics terminal =
        assist.diagnostics();
    require_case(
        terminal.state == interaction::PickAssistState::Failed &&
            terminal.reason == test_case.expected_reason &&
            terminal.reason == failed.reason,
        "second observation changed the terminal state or reason");
    const std::string terminal_output_message =
        std::string(test_case.name) +
        ": stable terminal observation emitted output";
    require_zero_pick_assist_output(
        terminal_output, terminal_output_message.c_str());
    require_case(
        !assist.active() && !assist.owns_manual_interact() &&
            !assist.take_submission(402U).has_value(),
        "stable terminal failure reacquired ownership or submitted");
    const std::string terminal_provenance_message =
        std::string(test_case.name) +
        ": stable terminal failure changed frozen provenance";
    require_frozen_slot_provenance_unchanged(
        frozen_before,
        terminal,
        terminal_provenance_message.c_str());
    require_case(
        terminal.settle_ticks == frozen_before.settle_ticks &&
            same_float_bits_exact(terminal.assisted_travel_m, 0.0F),
        "stable terminal failure advanced settling or assisted travel");
}

void test_slot_approach_target_unavailable_failures_are_stable() {
    const std::array<FrozenSlotObservationFailureCase, 2> cases{{
        {
            "null current target",
            interaction::PickAssistReason::TargetUnavailable,
            [](FrozenSlotScenario& scenario) {
                scenario.observation.target = nullptr;
            },
        },
        {
            "different current target handle ID",
            interaction::PickAssistReason::TargetUnavailable,
            [](FrozenSlotScenario& scenario) {
                ++scenario.target.handle.id;
            },
        },
    }};
    for (const FrozenSlotObservationFailureCase& test_case : cases) {
        expect_frozen_slot_observation_failure(test_case);
    }
}

void test_slot_approach_target_snapshot_failures_are_stable() {
    const std::array<FrozenSlotObservationFailureCase, 8> cases{{
        {
            "same-ID generation mutation",
            interaction::PickAssistReason::TargetChanged,
            [](FrozenSlotScenario& scenario) {
                ++scenario.target.handle.generation;
            },
        },
        {
            "same-ID object-pose mutation",
            interaction::PickAssistReason::TargetChanged,
            [](FrozenSlotScenario& scenario) {
                scenario.target.object_world.position.x = std::nextafter(
                    scenario.target.object_world.position.x, 1.0F);
            },
        },
        {
            "same-ID targeted-state mutation",
            interaction::PickAssistReason::TargetChanged,
            [](FrozenSlotScenario& scenario) {
                scenario.target.state = interaction::ObjectState::Targeted;
            },
        },
        {
            "same-ID owner-request mutation",
            interaction::PickAssistReason::TargetChanged,
            [](FrozenSlotScenario& scenario) {
                scenario.target.owner_request = 811U;
            },
        },
        {
            "same-ID selected-affordance grasp mutation",
            interaction::PickAssistReason::TargetChanged,
            [](FrozenSlotScenario& scenario) {
                interaction::GraspAffordance& affordance =
                    scenario.target.affordances.front();
                affordance.hand_in_object.position.x = std::nextafter(
                    affordance.hand_in_object.position.x, 1.0F);
            },
        },
        {
            "combined generation and selected-slot mutation",
            interaction::PickAssistReason::TargetChanged,
            [](FrozenSlotScenario& scenario) {
                ++scenario.target.handle.generation;
                interaction::GraspInteractionSlot& slot =
                    selected_authored_slot(scenario);
                slot.root_x_object_m = std::nextafter(
                    slot.root_x_object_m, 1.0F);
            },
        },
        {
            "combined targeted state and selected-slot mutation",
            interaction::PickAssistReason::TargetChanged,
            [](FrozenSlotScenario& scenario) {
                scenario.target.state = interaction::ObjectState::Targeted;
                interaction::GraspInteractionSlot& slot =
                    selected_authored_slot(scenario);
                slot.root_x_object_m = std::nextafter(
                    slot.root_x_object_m, 1.0F);
            },
        },
        {
            "combined owner request and selected-slot mutation",
            interaction::PickAssistReason::TargetChanged,
            [](FrozenSlotScenario& scenario) {
                scenario.target.owner_request = 811U;
                interaction::GraspInteractionSlot& slot =
                    selected_authored_slot(scenario);
                slot.root_x_object_m = std::nextafter(
                    slot.root_x_object_m, 1.0F);
            },
        },
    }};
    for (const FrozenSlotObservationFailureCase& test_case : cases) {
        expect_frozen_slot_observation_failure(test_case);
    }
}

void test_slot_approach_slot_identity_failures_precede_snapshot_mismatch() {
    const std::array<FrozenSlotObservationFailureCase, 7> cases{{
        {
            "selected authored slot geometry mutation",
            interaction::PickAssistReason::SlotChanged,
            [](FrozenSlotScenario& scenario) {
                interaction::GraspInteractionSlot& slot =
                    selected_authored_slot(scenario);
                slot.root_x_object_m = std::nextafter(
                    slot.root_x_object_m, 1.0F);
            },
        },
        {
            "selected authored slot ID mutation",
            interaction::PickAssistReason::SlotChanged,
            [](FrozenSlotScenario& scenario) {
                selected_authored_slot(scenario).id = 109U;
            },
        },
        {
            "ordered authored slot vector mutation",
            interaction::PickAssistReason::SlotChanged,
            [](FrozenSlotScenario& scenario) {
                std::vector<interaction::GraspInteractionSlot>& slots =
                    scenario.target.affordances.front().interaction_slots;
                require(
                    slots.size() == 3U && slots[1].id == 9U,
                    "slot-order fixture did not retain selected index 1");
                std::swap(slots.front(), slots.back());
            },
        },
        {
            "combined selected-slot and object-pose mutation",
            interaction::PickAssistReason::SlotChanged,
            [](FrozenSlotScenario& scenario) {
                interaction::GraspInteractionSlot& slot =
                    selected_authored_slot(scenario);
                slot.root_z_object_m = std::nextafter(
                    slot.root_z_object_m, 1.0F);
                scenario.target.object_world.position.z = std::nextafter(
                    scenario.target.object_world.position.z, 1.0F);
            },
        },
        {
            "selected authored slot ID changed to zero",
            interaction::PickAssistReason::SlotChanged,
            [](FrozenSlotScenario& scenario) {
                selected_authored_slot(scenario).id = 0U;
            },
        },
        {
            "selected authored slot ID changed to a duplicate",
            interaction::PickAssistReason::SlotChanged,
            [](FrozenSlotScenario& scenario) {
                interaction::GraspInteractionSlot& slot =
                    selected_authored_slot(scenario);
                slot.id = scenario.target.affordances.front()
                    .interaction_slots.front().id;
            },
        },
        {
            "selected authored slot geometry changed to NaN",
            interaction::PickAssistReason::SlotChanged,
            [](FrozenSlotScenario& scenario) {
                selected_authored_slot(scenario).root_x_object_m =
                    float_from_bits(0x7fc00001U);
            },
        },
    }};
    for (const FrozenSlotObservationFailureCase& test_case : cases) {
        expect_frozen_slot_observation_failure(test_case);
    }
}

void test_slot_approach_nonfinite_observation_failures_are_stable() {
    const std::array<FrozenSlotObservationFailureCase, 5> cases{{
        {
            "nonfinite displayed root position",
            interaction::PickAssistReason::OutsideTravelEnvelope,
            [](FrozenSlotScenario& scenario) {
                scenario.observation.displayed_root.position.x =
                    float_from_bits(0x7fc00001U);
            },
        },
        {
            "nonfinite simulation velocity",
            interaction::PickAssistReason::OutsideTravelEnvelope,
            [](FrozenSlotScenario& scenario) {
                scenario.observation.simulation_velocity.z =
                    float_from_bits(0x7fc00001U);
            },
        },
        {
            "NaN displayed planar speed",
            interaction::PickAssistReason::OutsideTravelEnvelope,
            [](FrozenSlotScenario& scenario) {
                scenario.observation.displayed_planar_speed_mps =
                    float_from_bits(0x7fc00001U);
            },
        },
        {
            "negative displayed planar speed",
            interaction::PickAssistReason::OutsideTravelEnvelope,
            [](FrozenSlotScenario& scenario) {
                scenario.observation.displayed_planar_speed_mps = -0.01F;
            },
        },
        {
            "nonfinite camera azimuth",
            interaction::PickAssistReason::OutsideTravelEnvelope,
            [](FrozenSlotScenario& scenario) {
                scenario.observation.camera_azimuth =
                    float_from_bits(0x7f800000U);
            },
        },
    }};
    for (const FrozenSlotObservationFailureCase& test_case : cases) {
        expect_frozen_slot_observation_failure(test_case);
    }
}

void test_slot_approach_unchanged_identity_still_steers() {
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist;
    require(
        assist.begin(scenario.start, &scenario.target),
        "unchanged-identity fixture begin failed");
    const interaction::PickAssistDiagnostics frozen_before =
        assist.diagnostics();

    const interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    require(
        output.override_steering && !is_zero(output.left_stick) &&
            is_zero(output.right_stick) && !output.force_strafe &&
            !output.stationary_constraint && !output.needs_preview &&
            !output.submit_interact,
        "unchanged target and slot identity did not keep far steering active");
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::SlotApproach &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            assist.active() && assist.owns_manual_interact() &&
            !assist.take_submission(403U).has_value(),
        "unchanged target and slot identity changed ownership or state");
    require_frozen_slot_provenance_unchanged(
        frozen_before,
        assist.diagnostics(),
        "unchanged observation changed frozen slot provenance");
}

void test_slot_approach_emits_slow_radius_arrival_steering() {
    const interaction::PickAssistConfig config{};
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist(config);

    require(
        assist.begin(scenario.start, &scenario.target),
        "slow-radius approach begin failed");
    const interaction::PickAssistDiagnostics frozen_before =
        assist.diagnostics();
    require(
        frozen_before.state == interaction::PickAssistState::SlotApproach &&
            frozen_before.reason == interaction::PickAssistReason::None &&
            frozen_before.selected_slot_id == 9U &&
            frozen_before.slot_selection.selected_index.has_value(),
        "slow-radius fixture did not select slot ID 9");
    const interaction::Transform frozen_root =
        frozen_before.slot_selection.ordered[
            *frozen_before.slot_selection.selected_index].root_world;
    require(
        frozen_root.position.x == 0.0F &&
            frozen_root.position.y == 0.0F &&
            frozen_root.position.z == 0.50F,
        "slow-radius fixture froze an unexpected root");

    scenario.observation.displayed_root.position =
        vec3(-0.30F, 0.0F, 0.50F);
    scenario.observation.camera_azimuth = 0.25F * PIf;
    const float root_error_m = static_cast<float>(std::hypot(
        static_cast<double>(
            frozen_root.position.x -
            scenario.observation.displayed_root.position.x),
        static_cast<double>(
            frozen_root.position.z -
            scenario.observation.displayed_root.position.z)));
    require(
        root_error_m < config.arrival.slow_radius_m &&
            root_error_m > config.arrival.latch_position_error_m,
        "slow-radius fixture was not between slow and arrival radii");

    const vec3 expected_left = interaction::arrival_navigation_stick(
        frozen_root.position,
        scenario.observation.displayed_root.position,
        scenario.observation.camera_azimuth,
        config.arrival);
    vec3 frozen_forward = quat_mul_vec3(
        frozen_root.rotation, vec3(0.0F, 0.0F, 1.0F));
    frozen_forward.y = 0.0F;
    const vec3 expected_right = interaction::arrival_facing_stick(
        frozen_forward, scenario.observation.camera_azimuth);

    const interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    require(
        output.override_steering && output.force_strafe,
        "slow-radius SlotApproach did not own strafe steering");
    require(
        same_vec3_bits_exact(output.left_stick, expected_left),
        "slow-radius SlotApproach did not use arrival navigation steering");
    require(
        same_vec3_bits_exact(output.right_stick, expected_right),
        "slow-radius SlotApproach did not use frozen-root facing steering");
    require(
        !output.needs_preview && !output.stationary_constraint &&
            !output.submit_interact,
        "slow-radius SlotApproach emitted preview, stationary, or submission output");
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::SlotApproach &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None,
        "slow-radius SlotApproach changed state or reason");
    require_frozen_slot_provenance_unchanged(
        frozen_before,
        assist.diagnostics(),
        "slow-radius SlotApproach changed frozen slot provenance");
}

void test_slot_approach_latches_inclusive_arrival_boundaries() {
    const interaction::PickAssistConfig config{};
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist(config);

    require(
        assist.begin(scenario.start, &scenario.target),
        "inclusive arrival-latch fixture begin failed");
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::SlotApproach &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            assist.diagnostics().slot_selection.selected_index.has_value(),
        "inclusive arrival-latch fixture did not freeze a slot");
    const interaction::MappedPickSlot frozen_slot =
        assist.diagnostics().slot_selection.ordered[
            *assist.diagnostics().slot_selection.selected_index];

    constexpr float position_error_m = 0.03F;
    constexpr float simulation_speed_mps = 0.05F;
    constexpr float yaw_error_radians = 20.0F * PIf / 180.0F;
    scenario.observation.displayed_root = frozen_slot.root_world;
    scenario.observation.displayed_root.position.x =
        frozen_slot.root_world.position.x - position_error_m;
    scenario.observation.displayed_root.rotation = quat_from_angle_axis(
        yaw_error_radians, vec3(0.0F, 1.0F, 0.0F));
    scenario.observation.simulation_velocity =
        vec3(simulation_speed_mps, 0.0F, 0.0F);

    const auto planar_distance = [](vec3 left, vec3 right) {
        return static_cast<float>(std::hypot(
            static_cast<double>(left.x) - static_cast<double>(right.x),
            static_cast<double>(left.z) - static_cast<double>(right.z)));
    };
    const auto planar_speed = [](vec3 velocity) {
        return static_cast<float>(std::hypot(
            static_cast<double>(velocity.x),
            static_cast<double>(velocity.z)));
    };
    const auto planar_yaw = [](quat rotation) {
        const vec3 forward = quat_mul_vec3(
            rotation, vec3(0.0F, 0.0F, 1.0F));
        return std::atan2(forward.x, forward.z);
    };
    const auto wrapped_yaw_error = [&](quat left, quat right) {
        const float difference = planar_yaw(left) - planar_yaw(right);
        return std::abs(std::atan2(
            std::sin(difference), std::cos(difference)));
    };
    const float derived_root_error_m = planar_distance(
        scenario.observation.displayed_root.position,
        frozen_slot.root_world.position);
    const float derived_simulation_speed_mps =
        planar_speed(scenario.observation.simulation_velocity);
    const float derived_yaw_error_radians = wrapped_yaw_error(
        scenario.observation.displayed_root.rotation,
        frozen_slot.root_world.rotation);
    require(
        same_float_bits_exact(
            derived_root_error_m,
            config.arrival.latch_position_error_m) &&
            same_float_bits_exact(
                derived_simulation_speed_mps,
                config.arrival.latch_simulation_speed_mps) &&
            same_float_bits_exact(
                derived_yaw_error_radians,
                config.arrival.maximum_yaw_error_radians),
        "inclusive arrival fixture metrics were not exactly on all three boundaries");
    const interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Settling &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            assist.diagnostics().settle_ticks == 0U,
        "inclusive frozen-slot arrival boundaries did not latch Settling");
    require(
        output.override_steering && output.force_strafe &&
            output.stationary_constraint && is_zero(output.left_stick) &&
            is_zero(output.right_stick),
        "inclusive arrival latch did not emit zero-stick stationary braking");
    require(
        !output.needs_preview && !output.submit_interact &&
            !assist.take_submission(74U).has_value(),
        "inclusive arrival latch previewed or submitted on the latch tick");
}

interaction::PickAssistDiagnostics latch_frozen_slot_settling(
    interaction::ControllerPickAssist& assist,
    FrozenSlotScenario& scenario,
    const interaction::PickAssistConfig& config) {
    require(
        assist.begin(scenario.start, &scenario.target),
        "frozen Settling fixture begin failed");
    const interaction::PickAssistDiagnostics frozen_before =
        assist.diagnostics();
    require(
        frozen_before.state ==
                interaction::PickAssistState::SlotApproach &&
            frozen_before.reason == interaction::PickAssistReason::None &&
            frozen_before.selected_slot_id == 9U &&
            frozen_before.slot_selection.selected_index.has_value(),
        "frozen Settling fixture did not select slot ID 9");
    const interaction::MappedPickSlot frozen_slot =
        frozen_before.slot_selection.ordered[
            *frozen_before.slot_selection.selected_index];

    scenario.observation.displayed_root = frozen_slot.root_world;
    scenario.observation.displayed_root.position.x =
        frozen_slot.root_world.position.x -
        config.arrival.latch_position_error_m;
    scenario.observation.displayed_root.rotation = quat_from_angle_axis(
        config.arrival.maximum_yaw_error_radians,
        vec3(0.0F, 1.0F, 0.0F));
    scenario.observation.simulation_velocity = vec3(
        config.arrival.latch_simulation_speed_mps, 0.0F, 0.0F);
    scenario.observation.displayed_planar_speed_mps = 0.0F;

    const float expected_travel_m = static_cast<float>(std::hypot(
        static_cast<double>(scenario.observation.displayed_root.position.x) -
            static_cast<double>(scenario.start.root_world.position.x),
        static_cast<double>(scenario.observation.displayed_root.position.z) -
            static_cast<double>(scenario.start.root_world.position.z)));
    const interaction::PickAssistOutput latch_output =
        assist.observe(scenario.observation);
    const interaction::PickAssistDiagnostics latched =
        assist.diagnostics();
    require(
        latched.state == interaction::PickAssistState::Settling &&
            latched.reason == interaction::PickAssistReason::None &&
            latched.settle_ticks == 0U,
        "valid inclusive frozen metrics did not latch Settling");
    require(
        latch_output.override_steering && latch_output.force_strafe &&
            latch_output.stationary_constraint &&
            is_zero(latch_output.left_stick) &&
            is_zero(latch_output.right_stick) &&
            !latch_output.needs_preview && !latch_output.submit_interact,
        "frozen Settling latch did not emit stationary zero-stick braking");
    require(
        assist.active() && assist.owns_manual_interact() &&
            !assist.take_submission(501U).has_value(),
        "frozen Settling latch lost ownership or submitted");
    require(
        same_float_bits_exact(
            latched.root_error_m,
            config.arrival.latch_position_error_m) &&
            same_float_bits_exact(
                latched.speed_mps,
                config.arrival.latch_simulation_speed_mps) &&
            same_float_bits_exact(
                latched.yaw_error_radians,
                config.arrival.maximum_yaw_error_radians) &&
            same_float_bits_exact(
                latched.assisted_travel_m, expected_travel_m),
        "frozen Settling latch did not retain inclusive metrics and travel");
    require_frozen_slot_provenance_unchanged(
        frozen_before,
        latched,
        "frozen Settling latch changed frozen provenance");
    return latched;
}

void test_frozen_slot_settling_revalidates_selected_slot() {
    const interaction::PickAssistConfig config{};
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist(config);
    const interaction::PickAssistDiagnostics latched =
        latch_frozen_slot_settling(assist, scenario, config);

    interaction::GraspInteractionSlot& selected =
        selected_authored_slot(scenario);
    selected.root_x_object_m = std::nextafter(
        selected.root_x_object_m, 1.0F);
    const interaction::PickAssistOutput failed_output =
        assist.observe(scenario.observation);
    const interaction::PickAssistDiagnostics failed =
        assist.diagnostics();
    require(
        failed.state == interaction::PickAssistState::Failed &&
            failed.reason == interaction::PickAssistReason::SlotChanged,
        "frozen Settling selected-slot mutation did not fail SlotChanged");
    require_zero_pick_assist_output(
        failed_output,
        "frozen Settling selected-slot failure emitted output");
    require(
        !assist.active() && !assist.owns_manual_interact() &&
            !assist.take_submission(502U).has_value(),
        "frozen Settling selected-slot failure retained ownership or submitted");
    require(
        failed.settle_ticks == latched.settle_ticks &&
            same_float_bits_exact(
                failed.assisted_travel_m, latched.assisted_travel_m),
        "frozen Settling selected-slot failure advanced settling or travel");
    require_frozen_slot_provenance_unchanged(
        latched,
        failed,
        "frozen Settling selected-slot failure changed frozen provenance");

    const interaction::PickAssistOutput terminal_output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::SlotChanged,
        "frozen Settling selected-slot failure was not terminal");
    require_zero_pick_assist_output(
        terminal_output,
        "terminal frozen Settling selected-slot failure emitted output");
    require_frozen_slot_provenance_unchanged(
        latched,
        assist.diagnostics(),
        "terminal frozen Settling selected-slot failure changed provenance");
}

void test_frozen_slot_settling_enforces_consecutive_travel() {
    const interaction::PickAssistConfig config{};
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist(config);
    const interaction::PickAssistDiagnostics latched =
        latch_frozen_slot_settling(assist, scenario, config);

    const vec3 previous_position =
        scenario.observation.displayed_root.position;
    scenario.observation.displayed_root.position.x -= 0.25F;
    const float next_segment_m = static_cast<float>(std::hypot(
        static_cast<double>(scenario.observation.displayed_root.position.x) -
            static_cast<double>(previous_position.x),
        static_cast<double>(scenario.observation.displayed_root.position.z) -
            static_cast<double>(previous_position.z)));
    const float expected_travel_m =
        latched.assisted_travel_m + next_segment_m;
    require(
        latched.assisted_travel_m < 1.00002F &&
            expected_travel_m > 1.00002F,
        "frozen Settling travel fixture did not cross 1.00002 m");

    const interaction::PickAssistOutput failed_output =
        assist.observe(scenario.observation);
    const interaction::PickAssistDiagnostics failed =
        assist.diagnostics();
    require(
        failed.state == interaction::PickAssistState::Failed &&
            failed.reason ==
                interaction::PickAssistReason::OutsideTravelEnvelope,
        "frozen Settling travel overshoot did not fail the envelope");
    require_zero_pick_assist_output(
        failed_output,
        "frozen Settling travel overshoot emitted output");
    require(
        !assist.active() && !assist.owns_manual_interact() &&
            !assist.take_submission(503U).has_value(),
        "frozen Settling travel overshoot retained ownership or submitted");
    require(
        failed.settle_ticks == latched.settle_ticks &&
            same_float_bits_exact(
                failed.assisted_travel_m, expected_travel_m),
        "frozen Settling travel overshoot changed settling or accumulation");
    require_frozen_slot_provenance_unchanged(
        latched,
        failed,
        "frozen Settling travel overshoot changed frozen provenance");

    const interaction::PickAssistOutput terminal_output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::OutsideTravelEnvelope &&
            same_float_bits_exact(
                assist.diagnostics().assisted_travel_m,
                expected_travel_m),
        "frozen Settling travel failure was not terminal and stable");
    require_zero_pick_assist_output(
        terminal_output,
        "terminal frozen Settling travel failure emitted output");
    require_frozen_slot_provenance_unchanged(
        latched,
        assist.diagnostics(),
        "terminal frozen Settling travel failure changed provenance");
}

void test_frozen_slot_settling_keeps_stationary_braking() {
    const interaction::PickAssistConfig config{};
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist(config);
    const interaction::PickAssistDiagnostics latched =
        latch_frozen_slot_settling(assist, scenario, config);

    const interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    const interaction::PickAssistDiagnostics settled =
        assist.diagnostics();
    require(
        settled.state == interaction::PickAssistState::Settling &&
            settled.reason == interaction::PickAssistReason::None &&
            settled.settle_ticks == 1U,
        "valid frozen Settling next tick changed state or did not advance exactly once");
    require(
        output.override_steering && output.force_strafe &&
            output.stationary_constraint && is_zero(output.left_stick) &&
            is_zero(output.right_stick) && !output.needs_preview &&
            !output.submit_interact,
        "valid frozen Settling next tick did not keep stationary braking");
    require(
        assist.active() && assist.owns_manual_interact() &&
            !assist.take_submission(504U).has_value(),
        "valid frozen Settling next tick lost ownership or submitted");
    require(
        same_float_bits_exact(
            settled.root_error_m,
            config.arrival.latch_position_error_m) &&
            same_float_bits_exact(
                settled.speed_mps,
                config.arrival.latch_simulation_speed_mps) &&
            same_float_bits_exact(
                settled.yaw_error_radians,
                config.arrival.maximum_yaw_error_radians) &&
            same_float_bits_exact(
                settled.assisted_travel_m, latched.assisted_travel_m),
        "valid frozen Settling next tick changed frozen metrics or travel");
    require_frozen_slot_provenance_unchanged(
        latched,
        settled,
        "valid frozen Settling next tick changed frozen provenance");
}

void test_frozen_slot_settling_requests_exact_frozen_preview_root() {
    const interaction::PickAssistConfig config{};
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist(config);
    const interaction::PickAssistDiagnostics latched =
        latch_frozen_slot_settling(assist, scenario, config);
    require(
        latched.slot_selection.selected_index.has_value(),
        "frozen settle-preview fixture lost its selected index");
    const interaction::MappedPickSlot frozen_slot =
        latched.slot_selection.ordered[
            *latched.slot_selection.selected_index];
    const vec3 frozen_forward = quat_mul_vec3(
        frozen_slot.root_world.rotation,
        vec3(0.0F, 0.0F, 1.0F));
    const interaction::PickEntryRoot expected_preview_root{
        frozen_slot.root_world.position.x,
        frozen_slot.root_world.position.z,
        std::atan2(frozen_forward.x, frozen_forward.z),
    };
    const float infinity = std::numeric_limits<float>::infinity();

    const auto configure_stable_boundary_observation = [&] {
        scenario.observation.displayed_root = frozen_slot.root_world;
        scenario.observation.displayed_root.position.x =
            frozen_slot.root_world.position.x -
            config.maximum_settle_position_error_m;
        scenario.observation.displayed_root.rotation = quat_mul(
            quat_from_angle_axis(
                config.arrival.maximum_yaw_error_radians,
                vec3(0.0F, 1.0F, 0.0F)),
            frozen_slot.root_world.rotation);
        scenario.observation.displayed_planar_speed_mps =
            config.maximum_settle_displayed_speed_mps;
        scenario.observation.simulation_velocity =
            vec3(3.0F, 0.0F, 4.0F);
    };
    const auto require_stationary_output = [](
        const interaction::PickAssistOutput& output,
        bool needs_preview,
        const char* message) {
        require(
            output.override_steering && output.force_strafe &&
                output.stationary_constraint &&
                is_zero(output.left_stick) &&
                is_zero(output.right_stick) &&
                output.needs_preview == needs_preview &&
                !output.submit_interact,
            message);
    };

    configure_stable_boundary_observation();
    const float derived_root_error_m = static_cast<float>(std::hypot(
        static_cast<double>(
            scenario.observation.displayed_root.position.x) -
            static_cast<double>(frozen_slot.root_world.position.x),
        static_cast<double>(
            scenario.observation.displayed_root.position.z) -
            static_cast<double>(frozen_slot.root_world.position.z)));
    const auto planar_yaw = [](quat rotation) {
        const vec3 forward = quat_mul_vec3(
            rotation, vec3(0.0F, 0.0F, 1.0F));
        return std::atan2(forward.x, forward.z);
    };
    const float yaw_difference =
        planar_yaw(scenario.observation.displayed_root.rotation) -
        planar_yaw(frozen_slot.root_world.rotation);
    const float derived_yaw_error_radians = std::abs(std::atan2(
        std::sin(yaw_difference), std::cos(yaw_difference)));
    require(
        same_float_bits_exact(
            derived_root_error_m,
            config.maximum_settle_position_error_m) &&
            same_float_bits_exact(
                scenario.observation.displayed_planar_speed_mps,
                config.maximum_settle_displayed_speed_mps) &&
            same_float_bits_exact(
                derived_yaw_error_radians,
                config.arrival.maximum_yaw_error_radians),
        "frozen settle-preview fixture was not exactly on all settle bounds");

    for (uint32_t bound = 0U; bound < 3U; ++bound) {
        configure_stable_boundary_observation();
        const interaction::PickAssistOutput stable_output =
            assist.observe(scenario.observation);
        require(
            assist.diagnostics().state ==
                    interaction::PickAssistState::Settling &&
                assist.diagnostics().reason ==
                    interaction::PickAssistReason::None &&
                assist.diagnostics().settle_ticks == 1U,
            "stable settle boundary did not begin a consecutive count");
        require_stationary_output(
            stable_output,
            false,
            "stable settle boundary did not keep no-preview braking");

        if (bound == 0U) {
            const float root_error_above = std::nextafter(
                config.maximum_settle_position_error_m, infinity);
            scenario.observation.displayed_root.position.x =
                frozen_slot.root_world.position.x - root_error_above;
        } else if (bound == 1U) {
            scenario.observation.displayed_planar_speed_mps =
                std::nextafter(
                    config.maximum_settle_displayed_speed_mps,
                    infinity);
        } else {
            const float yaw_error_above = std::nextafter(
                config.arrival.maximum_yaw_error_radians, infinity);
            scenario.observation.displayed_root.rotation = quat_mul(
                quat_from_angle_axis(
                    yaw_error_above, vec3(0.0F, 1.0F, 0.0F)),
                frozen_slot.root_world.rotation);
        }
        const interaction::PickAssistOutput unstable_output =
            assist.observe(scenario.observation);
        require(
            assist.diagnostics().state ==
                    interaction::PickAssistState::Settling &&
                assist.diagnostics().reason ==
                    interaction::PickAssistReason::None &&
                assist.diagnostics().settle_ticks == 0U,
            "first representable settle-bound overshoot did not reset the consecutive count");
        require_stationary_output(
            unstable_output,
            false,
            "unstable settle boundary did not keep no-preview braking");
    }

    configure_stable_boundary_observation();
    interaction::PickAssistOutput output{};
    for (uint32_t tick = 1U; tick <= 5U; ++tick) {
        output = assist.observe(scenario.observation);
        require(
            assist.diagnostics().settle_ticks == tick,
            "stable frozen settle ticks were not consecutive");
        if (tick < 5U) {
            require(
                assist.diagnostics().state ==
                        interaction::PickAssistState::Settling &&
                    assist.diagnostics().reason ==
                        interaction::PickAssistReason::None,
                "frozen settling completed before five stable ticks");
            require_stationary_output(
                output,
                false,
                "pre-final frozen settle tick requested a preview");
        }
    }
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::FinalPreview &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            assist.diagnostics().settle_ticks == 5U,
        "fifth stable frozen settle tick did not enter FinalPreview");
    require_stationary_output(
        output,
        true,
        "fifth stable frozen settle tick did not request stationary preview");
    require(
        output.preview_root.has_value() &&
            same_float_bits_exact(
                output.preview_root->world_x,
                expected_preview_root.world_x) &&
            same_float_bits_exact(
                output.preview_root->world_z,
                expected_preview_root.world_z) &&
            same_float_bits_exact(
                output.preview_root->world_yaw_radians,
                expected_preview_root.world_yaw_radians),
        "frozen FinalPreview request did not publish the exact frozen root");
}

interaction::PickEntryRoot enter_frozen_final_preview(
    interaction::ControllerPickAssist& assist,
    FrozenSlotScenario& scenario,
    const interaction::PickAssistConfig& config = {}) {
    const interaction::PickAssistDiagnostics latched =
        latch_frozen_slot_settling(assist, scenario, config);
    require(
        latched.slot_selection.selected_index.has_value(),
        "frozen FinalPreview helper lost its selected index");
    const interaction::MappedPickSlot frozen_slot =
        latched.slot_selection.ordered[
            *latched.slot_selection.selected_index];
    const vec3 frozen_forward = quat_mul_vec3(
        frozen_slot.root_world.rotation,
        vec3(0.0F, 0.0F, 1.0F));
    const interaction::PickEntryRoot expected_root{
        frozen_slot.root_world.position.x,
        frozen_slot.root_world.position.z,
        std::atan2(frozen_forward.x, frozen_forward.z),
    };

    scenario.observation.displayed_root = frozen_slot.root_world;
    scenario.observation.simulation_velocity = vec3();
    scenario.observation.displayed_planar_speed_mps = 0.0F;
    interaction::PickAssistOutput output{};
    for (uint32_t tick = 1U;
         tick <= config.required_settle_ticks;
         ++tick) {
        output = assist.observe(scenario.observation);
        require(
            assist.diagnostics().settle_ticks == tick,
            "frozen FinalPreview helper lost a stable settle tick");
        if (tick < config.required_settle_ticks) {
            require(
                assist.diagnostics().state ==
                        interaction::PickAssistState::Settling &&
                    assist.diagnostics().reason ==
                        interaction::PickAssistReason::None &&
                    output.override_steering && output.force_strafe &&
                    output.stationary_constraint &&
                    is_zero(output.left_stick) &&
                    is_zero(output.right_stick) &&
                    !output.needs_preview &&
                    !output.preview_root.has_value() &&
                    !output.submit_interact,
                "frozen FinalPreview helper previewed before settling completed");
        }
    }
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::FinalPreview &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            assist.diagnostics().settle_ticks ==
                config.required_settle_ticks,
        "frozen FinalPreview helper did not complete settling exactly once");
    require_stationary_preview_request(
        output,
        expected_root,
        "frozen FinalPreview helper did not request the exact frozen root");
    require(
        assist.active() && assist.owns_manual_interact() &&
            !assist.take_submission(60U).has_value(),
        "frozen FinalPreview helper lost ownership or submitted");
    require_frozen_slot_provenance_unchanged(
        latched,
        assist.diagnostics(),
        "frozen FinalPreview helper changed frozen provenance");
    return *output.preview_root;
}

interaction::PickEntryPreview certified_frozen_preview(
    interaction::PickEntryRoot root) {
    interaction::PickEntryPreview preview{};
    preview.path_feasible = true;
    preview.path_reason = interaction::Reason::None;
    preview.match_ready = true;
    preview.match_reason = interaction::Reason::None;
    preview.prospective_root = root;
    preview.feasible_entry_frame = 114;
    preview.contact_frame = 139;
    preview.total_cost = 8.25F;
    return preview;
}

interaction::PickAssistDiagnostics enter_frozen_ready_to_submit(
    interaction::ControllerPickAssist& assist,
    FrozenSlotScenario& scenario,
    const interaction::PickAssistConfig& config = {}) {
    const interaction::PickEntryRoot frozen_root =
        enter_frozen_final_preview(assist, scenario, config);
    const interaction::PickAssistDiagnostics before = assist.diagnostics();
    scenario.observation.snapshot_fingerprint = 1701U;
    scenario.observation.preview_snapshot_fingerprint = 1701U;
    scenario.observation.preview = certified_frozen_preview(frozen_root);

    const interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::ReadyToSubmit &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            output.override_steering && output.force_strafe &&
            output.stationary_constraint &&
            is_zero(output.left_stick) && is_zero(output.right_stick) &&
            !output.needs_preview && !output.preview_root.has_value() &&
            output.submit_interact,
        "frozen ReadyToSubmit helper did not certify exactly once");
    require(
        assist.active() && assist.owns_manual_interact(),
        "frozen ReadyToSubmit helper lost pre-handoff ownership");
    require_frozen_slot_provenance_unchanged(
        before,
        assist.diagnostics(),
        "frozen ReadyToSubmit helper changed frozen provenance");
    return assist.diagnostics();
}

void test_frozen_final_preview_missing_singular_preview_rerequests() {
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist;
    const interaction::PickEntryRoot frozen_root =
        enter_frozen_final_preview(assist, scenario);
    const interaction::PickAssistDiagnostics before = assist.diagnostics();

    scenario.observation.preview.reset();
    const interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::FinalPreview &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None,
        "missing singular frozen preview changed FinalPreview outcome");
    require_stationary_preview_request(
        output,
        frozen_root,
        "missing singular frozen preview did not re-request the frozen root");
    require(
        assist.active() && assist.owns_manual_interact() &&
            !assist.take_submission(61U).has_value(),
        "missing singular frozen preview lost ownership or submitted");
    require_frozen_slot_provenance_unchanged(
        before,
        assist.diagnostics(),
        "missing singular frozen preview changed frozen provenance");
    require_final_preview_diagnostics_cleared(assist.diagnostics());
}

void test_frozen_final_preview_poor_match_rerequests_then_recovers() {
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist;
    const interaction::PickEntryRoot frozen_root =
        enter_frozen_final_preview(assist, scenario);
    const interaction::PickAssistDiagnostics before = assist.diagnostics();
    scenario.observation.snapshot_fingerprint = 901U;
    scenario.observation.preview_snapshot_fingerprint = 901U;

    interaction::PickEntryPreview poor_match =
        certified_frozen_preview(frozen_root);
    poor_match.match_ready = false;
    poor_match.match_reason = interaction::Reason::PoorMatch;
    poor_match.total_cost = 12.50F;
    scenario.observation.preview = poor_match;
    interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::FinalPreview &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None,
        "sole PoorMatch preview did not remain retryable");
    require_stationary_preview_request(
        output,
        frozen_root,
        "sole PoorMatch preview did not re-request the same frozen root");
    {
        const auto& diagnostics = assist.diagnostics().final_preview;
        require(
            diagnostics.available &&
                diagnostics.all_preview_roots_finite &&
                diagnostics.fingerprint_equal &&
                diagnostics.path_feasible &&
                diagnostics.path_reason == interaction::Reason::None &&
                !diagnostics.match_ready &&
                diagnostics.match_reason ==
                    interaction::Reason::PoorMatch &&
                diagnostics.prospective_root_equal &&
                diagnostics.feasible_entry_frame == 114 &&
                diagnostics.contact_frame == 139 &&
                same_float_bits_exact(
                    diagnostics.total_cost, poor_match.total_cost),
            "sole PoorMatch preview diagnostics were not captured exactly");
    }
    require(
        assist.active() && assist.owns_manual_interact() &&
            !assist.take_submission(62U).has_value(),
        "sole PoorMatch preview lost ownership or submitted");
    require_frozen_slot_provenance_unchanged(
        before,
        assist.diagnostics(),
        "sole PoorMatch preview changed frozen provenance");

    interaction::PickEntryPreview certified =
        certified_frozen_preview(frozen_root);
    certified.feasible_entry_frame = 115;
    certified.contact_frame = 140;
    certified.total_cost = 7.50F;
    scenario.observation.preview = certified;
    output = assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::ReadyToSubmit &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            output.override_steering && output.force_strafe &&
            output.stationary_constraint &&
            is_zero(output.left_stick) && is_zero(output.right_stick) &&
            !output.needs_preview && !output.preview_root.has_value() &&
            output.submit_interact,
        "certified singular preview did not recover PoorMatch to ReadyToSubmit");
    {
        const auto& diagnostics = assist.diagnostics().final_preview;
        require(
            diagnostics.available &&
                diagnostics.all_preview_roots_finite &&
                diagnostics.fingerprint_equal &&
                diagnostics.path_feasible &&
                diagnostics.path_reason == interaction::Reason::None &&
                diagnostics.match_ready &&
                diagnostics.match_reason == interaction::Reason::None &&
                diagnostics.prospective_root_equal &&
                diagnostics.feasible_entry_frame == 115 &&
                diagnostics.contact_frame == 140 &&
                same_float_bits_exact(
                    diagnostics.total_cost, certified.total_cost),
            "recovered certified preview diagnostics were not captured exactly");
    }
}

enum class FrozenFinalPreviewDefect : uint8_t {
    BlockedPath,
    CorrectionLimit,
    TargetUnavailable,
    NonfiniteRoot,
    FingerprintMismatch,
    DifferentRoot,
};

struct FrozenFinalPreviewHardCase {
    const char* name;
    FrozenFinalPreviewDefect defect;
};

void test_frozen_final_preview_hard_rejections_are_terminal() {
    const std::array<FrozenFinalPreviewHardCase, 6> cases{{
        {"BlockedPath preview reason", FrozenFinalPreviewDefect::BlockedPath},
        {"CorrectionLimit preview reason",
            FrozenFinalPreviewDefect::CorrectionLimit},
        {"TargetUnavailable preview reason",
            FrozenFinalPreviewDefect::TargetUnavailable},
        {"nonfinite preview root", FrozenFinalPreviewDefect::NonfiniteRoot},
        {"preview fingerprint mismatch",
            FrozenFinalPreviewDefect::FingerprintMismatch},
        {"one-ULP different prospective root",
            FrozenFinalPreviewDefect::DifferentRoot},
    }};
    const float infinity = float_from_bits(0x7f800000U);

    for (size_t case_index = 0U; case_index < cases.size(); ++case_index) {
        const FrozenFinalPreviewHardCase& test_case = cases[case_index];
        const auto require_case = [&](bool condition, const char* detail) {
            if (!condition) {
                throw std::runtime_error(
                    std::string(test_case.name) + ": " + detail);
            }
        };
        FrozenSlotScenario scenario;
        interaction::ControllerPickAssist assist;
        const interaction::PickEntryRoot frozen_root =
            enter_frozen_final_preview(assist, scenario);
        const interaction::PickAssistDiagnostics before =
            assist.diagnostics();
        scenario.observation.snapshot_fingerprint = 901U;
        scenario.observation.preview_snapshot_fingerprint = 901U;
        interaction::PickEntryPreview preview =
            certified_frozen_preview(frozen_root);
        bool expected_finite = true;
        bool expected_fingerprint_equal = true;
        bool expected_root_equal = true;

        switch (test_case.defect) {
        case FrozenFinalPreviewDefect::BlockedPath:
            preview.path_feasible = false;
            preview.path_reason = interaction::Reason::BlockedPath;
            preview.match_ready = false;
            break;
        case FrozenFinalPreviewDefect::CorrectionLimit:
            preview.match_ready = false;
            preview.match_reason = interaction::Reason::CorrectionLimit;
            break;
        case FrozenFinalPreviewDefect::TargetUnavailable:
            preview.path_feasible = false;
            preview.path_reason = interaction::Reason::TargetUnavailable;
            preview.match_ready = false;
            break;
        case FrozenFinalPreviewDefect::NonfiniteRoot:
            preview.prospective_root.world_x =
                float_from_bits(0x7fc00001U);
            expected_finite = false;
            expected_root_equal = false;
            break;
        case FrozenFinalPreviewDefect::FingerprintMismatch:
            scenario.observation.preview_snapshot_fingerprint = 902U;
            expected_fingerprint_equal = false;
            break;
        case FrozenFinalPreviewDefect::DifferentRoot:
            preview.prospective_root.world_x = std::nextafter(
                preview.prospective_root.world_x, infinity);
            expected_root_equal = false;
            break;
        }
        require_case(
            scenario.observation.target == &scenario.target,
            "fixture invalidated the live observation target");
        scenario.observation.preview = preview;
        const interaction::PickAssistOutput output =
            assist.observe(scenario.observation);
        require_case(
            assist.diagnostics().state ==
                    interaction::PickAssistState::Failed &&
                assist.diagnostics().reason ==
                    interaction::PickAssistReason::FinalPreviewRejected,
            "hard preview outcome was not rejected immediately");
        require_zero_pick_assist_output(output, test_case.name);
        require_case(
            !assist.active() && !assist.owns_manual_interact() &&
                !assist.take_submission(
                    63U + static_cast<uint64_t>(case_index)).has_value(),
            "hard preview rejection retained ownership or submitted");
        require_frozen_slot_provenance_unchanged(
            before,
            assist.diagnostics(),
            test_case.name);
        const auto& diagnostics = assist.diagnostics().final_preview;
        require_case(
            diagnostics.available &&
                diagnostics.all_preview_roots_finite == expected_finite &&
                diagnostics.fingerprint_equal ==
                    expected_fingerprint_equal &&
                diagnostics.path_feasible == preview.path_feasible &&
                diagnostics.path_reason == preview.path_reason &&
                diagnostics.match_ready == preview.match_ready &&
                diagnostics.match_reason == preview.match_reason &&
                diagnostics.prospective_root_equal == expected_root_equal &&
                diagnostics.feasible_entry_frame ==
                    preview.feasible_entry_frame &&
                diagnostics.contact_frame == preview.contact_frame &&
                same_float_bits_exact(
                    diagnostics.total_cost, preview.total_cost),
            "hard preview diagnostics did not capture every supplied fact");
    }
}

void test_frozen_final_preview_certifies_and_submits_exactly_once() {
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist;
    const interaction::PickEntryRoot frozen_root =
        enter_frozen_final_preview(assist, scenario);
    const interaction::PickAssistDiagnostics before = assist.diagnostics();
    scenario.observation.snapshot_fingerprint = 901U;
    scenario.observation.preview_snapshot_fingerprint = 901U;
    const interaction::PickEntryPreview certified =
        certified_frozen_preview(frozen_root);
    scenario.observation.preview = certified;

    interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::ReadyToSubmit &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            output.override_steering && output.force_strafe &&
            output.stationary_constraint &&
            is_zero(output.left_stick) && is_zero(output.right_stick) &&
            !output.needs_preview && !output.preview_root.has_value() &&
            output.submit_interact,
        "certified frozen preview did not emit one stationary submit pulse");
    require(
        assist.active() && assist.owns_manual_interact(),
        "certified frozen preview did not retain ReadyToSubmit ownership");
    require_frozen_slot_provenance_unchanged(
        before,
        assist.diagnostics(),
        "certified frozen preview changed frozen provenance");

    output = assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::ReadyToSubmit &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            output.override_steering && output.force_strafe &&
            output.stationary_constraint &&
            is_zero(output.left_stick) && is_zero(output.right_stick) &&
            !output.needs_preview && !output.preview_root.has_value() &&
            !output.submit_interact,
        "ReadyToSubmit repeated the frozen submit pulse");

    const std::optional<interaction::PickRequest> first =
        assist.take_submission(71U);
    const std::optional<interaction::PickRequest> second =
        assist.take_submission(71U);
    require(
        first.has_value() && !second.has_value() &&
            first->target == scenario.target.handle &&
            first->affordance_id ==
                scenario.target.affordances.front().id &&
            first->request_id == 71U &&
            assist.diagnostics().state ==
                interaction::PickAssistState::Submitted &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            !assist.active() && !assist.owns_manual_interact(),
        "take_submission(71) did not yield exactly one frozen PickRequest");

    assist.cancel();
    require(
        first.has_value() && first->target == scenario.target.handle &&
            first->affordance_id ==
                scenario.target.affordances.front().id &&
            first->request_id == 71U &&
            assist.diagnostics().state ==
                interaction::PickAssistState::Submitted &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            !assist.take_submission(71U).has_value(),
        "cancel after Submitted revoked or created a frozen request");
}

enum class FrozenReadyPreflightMutation : uint8_t {
    Runtime,
    MissingTarget,
    SlotGeometry,
    NonfiniteMetric,
};

struct FrozenReadyPreflightCase {
    const char* name;
    FrozenReadyPreflightMutation mutation;
    interaction::PickAssistReason expected_reason;
};

void test_frozen_ready_to_submit_runs_shared_preflight() {
    const std::array<FrozenReadyPreflightCase, 4> cases{{
        {"runtime changed", FrozenReadyPreflightMutation::Runtime,
            interaction::PickAssistReason::RuntimeChanged},
        {"target unavailable", FrozenReadyPreflightMutation::MissingTarget,
            interaction::PickAssistReason::TargetUnavailable},
        {"selected slot changed", FrozenReadyPreflightMutation::SlotGeometry,
            interaction::PickAssistReason::SlotChanged},
        {"displayed metric nonfinite",
            FrozenReadyPreflightMutation::NonfiniteMetric,
            interaction::PickAssistReason::OutsideTravelEnvelope},
    }};

    for (size_t case_index = 0U; case_index < cases.size(); ++case_index) {
        const FrozenReadyPreflightCase& test_case = cases[case_index];
        const auto require_case = [&](bool condition, const char* detail) {
            if (!condition) {
                throw std::runtime_error(
                    std::string(test_case.name) + ": " + detail);
            }
        };
        FrozenSlotScenario scenario;
        interaction::ControllerPickAssist assist;
        const interaction::PickAssistDiagnostics frozen =
            enter_frozen_ready_to_submit(assist, scenario);

        switch (test_case.mutation) {
        case FrozenReadyPreflightMutation::Runtime:
            scenario.observation.runtime_state =
                interaction::RuntimeState::Preflight;
            break;
        case FrozenReadyPreflightMutation::MissingTarget:
            scenario.observation.target = nullptr;
            break;
        case FrozenReadyPreflightMutation::SlotGeometry: {
            interaction::GraspInteractionSlot& selected =
                selected_authored_slot(scenario);
            selected.root_x_object_m = std::nextafter(
                selected.root_x_object_m, 1.0F);
            break;
        }
        case FrozenReadyPreflightMutation::NonfiniteMetric:
            scenario.observation.displayed_planar_speed_mps =
                float_from_bits(0x7fc00001U);
            break;
        }

        const interaction::PickAssistOutput output =
            assist.observe(scenario.observation);
        require_case(
            assist.diagnostics().state ==
                    interaction::PickAssistState::Failed &&
                assist.diagnostics().reason == test_case.expected_reason,
            "ReadyToSubmit did not fail through frozen shared preflight");
        require_zero_pick_assist_output(output, test_case.name);
        require_case(
            !assist.active() && !assist.owns_manual_interact() &&
                !assist.take_submission(
                    1702U + static_cast<uint64_t>(case_index)).has_value(),
            "ReadyToSubmit preflight failure retained ownership or request");
        require_frozen_slot_provenance_unchanged(
            frozen,
            assist.diagnostics(),
            test_case.name);
    }
}

void test_frozen_final_preview_missing_preview_hits_arrival_deadline() {
    interaction::PickAssistConfig config{};
    config.maximum_arrival_ticks = config.required_settle_ticks + 3U;
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist(config);
    const interaction::PickEntryRoot frozen_root =
        enter_frozen_final_preview(assist, scenario, config);
    const interaction::PickAssistDiagnostics before = assist.diagnostics();

    scenario.observation.preview.reset();
    for (uint32_t retry = 1U; retry <= 2U; ++retry) {
        const interaction::PickAssistOutput output =
            assist.observe(scenario.observation);
        require(
            assist.diagnostics().state ==
                    interaction::PickAssistState::FinalPreview &&
                assist.diagnostics().reason ==
                    interaction::PickAssistReason::None,
            "missing frozen preview reached the arrival deadline early");
        require_stationary_preview_request(
            output,
            frozen_root,
            "pre-deadline missing frozen preview did not re-request");
        require_frozen_slot_provenance_unchanged(
            before,
            assist.diagnostics(),
            "pre-deadline missing frozen preview changed frozen provenance");
    }

    const interaction::PickAssistOutput deadline_output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::ArrivalDeadline,
        "third missing frozen preview did not hit the inclusive arrival deadline");
    require_zero_pick_assist_output(
        deadline_output,
        "missing frozen preview deadline emitted output or another request");
    require(
        !assist.active() && !assist.owns_manual_interact() &&
            !assist.take_submission(72U).has_value(),
        "missing frozen preview deadline retained ownership or submitted");
    require_frozen_slot_provenance_unchanged(
        before,
        assist.diagnostics(),
        "missing frozen preview deadline changed frozen provenance");
}

void test_frozen_final_preview_poor_match_deadline_memory_survives_missing() {
    interaction::PickAssistConfig config{};
    config.maximum_arrival_ticks = config.required_settle_ticks + 3U;
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist(config);
    const interaction::PickEntryRoot frozen_root =
        enter_frozen_final_preview(assist, scenario, config);
    const interaction::PickAssistDiagnostics before = assist.diagnostics();
    scenario.observation.snapshot_fingerprint = 901U;
    scenario.observation.preview_snapshot_fingerprint = 901U;

    interaction::PickEntryPreview poor_match =
        certified_frozen_preview(frozen_root);
    poor_match.match_ready = false;
    poor_match.match_reason = interaction::Reason::PoorMatch;
    scenario.observation.preview = poor_match;
    interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::FinalPreview &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None,
        "sole PoorMatch reached the frozen arrival deadline early");
    require_stationary_preview_request(
        output,
        frozen_root,
        "pre-deadline PoorMatch did not re-request the frozen preview");

    scenario.observation.preview.reset();
    output = assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::FinalPreview &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None,
        "missing preview after PoorMatch reached the arrival deadline early");
    require_stationary_preview_request(
        output,
        frozen_root,
        "missing preview after PoorMatch did not re-request");
    {
        const auto& diagnostics = assist.diagnostics().final_preview;
        require(
            diagnostics.available && diagnostics.path_feasible &&
                diagnostics.path_reason == interaction::Reason::None &&
                !diagnostics.match_ready &&
                diagnostics.match_reason == interaction::Reason::PoorMatch,
            "missing preview forgot the prior sole PoorMatch diagnostics");
    }

    const interaction::PickAssistOutput deadline_output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::PoorMatch,
        "frozen preview deadline forgot the prior sole PoorMatch outcome");
    require_zero_pick_assist_output(
        deadline_output,
        "PoorMatch deadline emitted output or another preview request");
    require(
        !assist.active() && !assist.owns_manual_interact() &&
            !assist.take_submission(73U).has_value(),
        "PoorMatch deadline retained ownership or submitted");
    require_frozen_slot_provenance_unchanged(
        before,
        assist.diagnostics(),
        "PoorMatch deadline changed frozen provenance");
}

void test_frozen_final_preview_deadline_precedes_certification() {
    interaction::PickAssistConfig config{};
    config.maximum_arrival_ticks = config.required_settle_ticks + 3U;
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist(config);
    const interaction::PickEntryRoot frozen_root =
        enter_frozen_final_preview(assist, scenario, config);
    const interaction::PickAssistDiagnostics before = assist.diagnostics();
    scenario.observation.snapshot_fingerprint = 901U;
    scenario.observation.preview_snapshot_fingerprint = 901U;

    scenario.observation.preview.reset();
    for (uint32_t retry = 1U; retry <= 2U; ++retry) {
        const interaction::PickAssistOutput output =
            assist.observe(scenario.observation);
        require(
            assist.diagnostics().state ==
                    interaction::PickAssistState::FinalPreview &&
                assist.diagnostics().reason ==
                    interaction::PickAssistReason::None,
            "certification-boundary fixture reached the deadline early");
        require_stationary_preview_request(
            output,
            frozen_root,
            "certification-boundary fixture did not re-request");
    }
    require_final_preview_diagnostics_cleared(assist.diagnostics());

    scenario.observation.preview = certified_frozen_preview(frozen_root);
    const interaction::PickAssistOutput deadline_output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::ArrivalDeadline,
        "certified frozen preview was processed on the inclusive deadline");
    require_zero_pick_assist_output(
        deadline_output,
        "deadline-boundary certification emitted output or submitted");
    require_final_preview_diagnostics_cleared(assist.diagnostics());
    require(
        !assist.active() && !assist.owns_manual_interact() &&
            !assist.take_submission(74U).has_value(),
        "deadline-boundary certification retained ownership or submitted");
    require_frozen_slot_provenance_unchanged(
        before,
        assist.diagnostics(),
        "deadline-boundary certification changed frozen provenance");
}

void test_slot_approach_rejects_adjacent_arrival_overshoots() {
    const interaction::PickAssistConfig config{};
    const float infinity = std::numeric_limits<float>::infinity();
    const auto planar_distance = [](vec3 left, vec3 right) {
        return static_cast<float>(std::hypot(
            static_cast<double>(left.x) - static_cast<double>(right.x),
            static_cast<double>(left.z) - static_cast<double>(right.z)));
    };
    const auto planar_speed = [](vec3 velocity) {
        return static_cast<float>(std::hypot(
            static_cast<double>(velocity.x),
            static_cast<double>(velocity.z)));
    };
    const auto planar_yaw = [](quat rotation) {
        const vec3 forward = quat_mul_vec3(
            rotation, vec3(0.0F, 0.0F, 1.0F));
        return std::atan2(forward.x, forward.z);
    };
    const auto wrapped_yaw_error = [&](quat left, quat right) {
        const float difference = planar_yaw(left) - planar_yaw(right);
        return std::abs(std::atan2(
            std::sin(difference), std::cos(difference)));
    };

    {
        FrozenSlotScenario scenario;
        interaction::ControllerPickAssist assist(config);
        require(
            assist.begin(scenario.start, &scenario.target) &&
                assist.diagnostics().slot_selection.selected_index.has_value(),
            "adjacent-position arrival fixture begin failed");
        const interaction::MappedPickSlot frozen_slot =
            assist.diagnostics().slot_selection.ordered[
                *assist.diagnostics().slot_selection.selected_index];
        const float position_error_above = std::nextafter(
            config.arrival.latch_position_error_m, infinity);
        scenario.observation.displayed_root = frozen_slot.root_world;
        scenario.observation.displayed_root.position.x =
            frozen_slot.root_world.position.x - position_error_above;
        scenario.observation.displayed_root.rotation = quat_from_angle_axis(
            config.arrival.maximum_yaw_error_radians,
            vec3(0.0F, 1.0F, 0.0F));
        scenario.observation.simulation_velocity = vec3(
            config.arrival.latch_simulation_speed_mps, 0.0F, 0.0F);

        const float derived_root_error_m = planar_distance(
            scenario.observation.displayed_root.position,
            frozen_slot.root_world.position);
        const float derived_simulation_speed_mps =
            planar_speed(scenario.observation.simulation_velocity);
        const float derived_yaw_error_radians = wrapped_yaw_error(
            scenario.observation.displayed_root.rotation,
            frozen_slot.root_world.rotation);
        require(
            same_float_bits_exact(
                derived_root_error_m, position_error_above) &&
                derived_root_error_m >
                    config.arrival.latch_position_error_m &&
                same_float_bits_exact(
                    derived_simulation_speed_mps,
                    config.arrival.latch_simulation_speed_mps) &&
                same_float_bits_exact(
                    derived_yaw_error_radians,
                    config.arrival.maximum_yaw_error_radians),
            "adjacent-position fixture changed more than its position metric");
        const interaction::PickAssistOutput output =
            assist.observe(scenario.observation);
        require(
            assist.diagnostics().state ==
                    interaction::PickAssistState::SlotApproach &&
                assist.diagnostics().reason ==
                    interaction::PickAssistReason::None &&
                assist.diagnostics().settle_ticks == 0U,
            "first representable position overshoot latched arrival");
        require(
            output.override_steering && !output.stationary_constraint &&
                !output.needs_preview && !output.submit_interact &&
                !assist.take_submission(75U).has_value(),
            "first representable position overshoot latched braking or submitted");
    }

    {
        FrozenSlotScenario scenario;
        interaction::ControllerPickAssist assist(config);
        require(
            assist.begin(scenario.start, &scenario.target) &&
                assist.diagnostics().slot_selection.selected_index.has_value(),
            "adjacent-speed arrival fixture begin failed");
        const interaction::MappedPickSlot frozen_slot =
            assist.diagnostics().slot_selection.ordered[
                *assist.diagnostics().slot_selection.selected_index];
        const float simulation_speed_above = std::nextafter(
            config.arrival.latch_simulation_speed_mps, infinity);
        scenario.observation.displayed_root = frozen_slot.root_world;
        scenario.observation.displayed_root.position.x =
            frozen_slot.root_world.position.x -
            config.arrival.latch_position_error_m;
        scenario.observation.displayed_root.rotation = quat_from_angle_axis(
            config.arrival.maximum_yaw_error_radians,
            vec3(0.0F, 1.0F, 0.0F));
        scenario.observation.simulation_velocity =
            vec3(simulation_speed_above, 0.0F, 0.0F);

        const float derived_root_error_m = planar_distance(
            scenario.observation.displayed_root.position,
            frozen_slot.root_world.position);
        const float derived_simulation_speed_mps =
            planar_speed(scenario.observation.simulation_velocity);
        const float derived_yaw_error_radians = wrapped_yaw_error(
            scenario.observation.displayed_root.rotation,
            frozen_slot.root_world.rotation);
        require(
            same_float_bits_exact(
                derived_root_error_m,
                config.arrival.latch_position_error_m) &&
                same_float_bits_exact(
                    derived_simulation_speed_mps,
                    simulation_speed_above) &&
                derived_simulation_speed_mps >
                    config.arrival.latch_simulation_speed_mps &&
                same_float_bits_exact(
                    derived_yaw_error_radians,
                    config.arrival.maximum_yaw_error_radians),
            "adjacent-speed fixture changed more than its simulation-speed metric");
        const interaction::PickAssistOutput output =
            assist.observe(scenario.observation);
        require(
            assist.diagnostics().state ==
                    interaction::PickAssistState::SlotApproach &&
                assist.diagnostics().reason ==
                    interaction::PickAssistReason::None &&
                assist.diagnostics().settle_ticks == 0U,
            "first representable simulation-speed overshoot latched arrival");
        require(
            output.override_steering && !output.stationary_constraint &&
                !output.needs_preview && !output.submit_interact &&
                !assist.take_submission(76U).has_value(),
            "first representable simulation-speed overshoot latched braking or submitted");
    }

    {
        FrozenSlotScenario scenario;
        interaction::ControllerPickAssist assist(config);
        require(
            assist.begin(scenario.start, &scenario.target) &&
                assist.diagnostics().slot_selection.selected_index.has_value(),
            "adjacent-yaw arrival fixture begin failed");
        const interaction::MappedPickSlot frozen_slot =
            assist.diagnostics().slot_selection.ordered[
                *assist.diagnostics().slot_selection.selected_index];
        const float yaw_angle_above = std::nextafter(
            config.arrival.maximum_yaw_error_radians, infinity);
        scenario.observation.displayed_root = frozen_slot.root_world;
        scenario.observation.displayed_root.position.x =
            frozen_slot.root_world.position.x -
            config.arrival.latch_position_error_m;
        scenario.observation.displayed_root.rotation = quat_from_angle_axis(
            yaw_angle_above, vec3(0.0F, 1.0F, 0.0F));
        scenario.observation.simulation_velocity = vec3(
            config.arrival.latch_simulation_speed_mps, 0.0F, 0.0F);

        const float derived_root_error_m = planar_distance(
            scenario.observation.displayed_root.position,
            frozen_slot.root_world.position);
        const float derived_simulation_speed_mps =
            planar_speed(scenario.observation.simulation_velocity);
        const float derived_yaw_error_radians = wrapped_yaw_error(
            scenario.observation.displayed_root.rotation,
            frozen_slot.root_world.rotation);
        const float first_derived_yaw_above = std::nextafter(
            config.arrival.maximum_yaw_error_radians, infinity);
        require(
            same_float_bits_exact(
                derived_root_error_m,
                config.arrival.latch_position_error_m) &&
                same_float_bits_exact(
                    derived_simulation_speed_mps,
                    config.arrival.latch_simulation_speed_mps) &&
                same_float_bits_exact(
                    derived_yaw_error_radians,
                    first_derived_yaw_above) &&
                derived_yaw_error_radians >
                    config.arrival.maximum_yaw_error_radians,
            "one-ULP yaw quaternion did not produce the first derived yaw above the bound");
        const interaction::PickAssistOutput output =
            assist.observe(scenario.observation);
        require(
            assist.diagnostics().state ==
                    interaction::PickAssistState::SlotApproach &&
                assist.diagnostics().reason ==
                    interaction::PickAssistReason::None &&
                assist.diagnostics().settle_ticks == 0U,
            "first representable derived-yaw overshoot latched arrival");
        require(
            output.override_steering && !output.stationary_constraint &&
                !output.needs_preview && !output.submit_interact &&
                !assist.take_submission(77U).has_value(),
            "first representable derived-yaw overshoot latched braking or submitted");
    }
}

void test_slot_approach_accumulates_inclusive_travel_and_rejects_overshoot() {
    const auto configure_travel_scenario = [](FrozenSlotScenario& scenario) {
        scenario.target.object_world.position.x = 0.80F;
        scenario.start.target_snapshot = scenario.target;
        scenario.start.root_world = {
            vec3(0.0F, 0.0F, 0.50F), quat()};
        scenario.observation.displayed_root = scenario.start.root_world;
        scenario.observation.target = &scenario.target;
    };
    const auto planar_endpoint_distance = [](vec3 left, vec3 right) {
        return static_cast<float>(std::hypot(
            static_cast<double>(left.x) - static_cast<double>(right.x),
            static_cast<double>(left.z) - static_cast<double>(right.z)));
    };
    const auto xyz_endpoint_distance = [](vec3 left, vec3 right) {
        return static_cast<float>(std::hypot(
            static_cast<double>(left.x) - static_cast<double>(right.x),
            static_cast<double>(left.y) - static_cast<double>(right.y),
            static_cast<double>(left.z) - static_cast<double>(right.z)));
    };
    const auto same_float_bits = [](float left, float right) {
        return std::memcmp(&left, &right, sizeof(left)) == 0;
    };
    const auto same_transform_bits = [&](
        interaction::Transform left,
        interaction::Transform right) {
        return same_float_bits(left.position.x, right.position.x) &&
            same_float_bits(left.position.y, right.position.y) &&
            same_float_bits(left.position.z, right.position.z) &&
            same_float_bits(left.rotation.w, right.rotation.w) &&
            same_float_bits(left.rotation.x, right.rotation.x) &&
            same_float_bits(left.rotation.y, right.rotation.y) &&
            same_float_bits(left.rotation.z, right.rotation.z);
    };

    FrozenSlotScenario inclusive_scenario;
    configure_travel_scenario(inclusive_scenario);
    interaction::ControllerPickAssist inclusive_assist;
    require(
        inclusive_assist.begin(
            inclusive_scenario.start, &inclusive_scenario.target),
        "inclusive-travel fixture begin failed");
    require(
        inclusive_assist.diagnostics().selected_slot_id == 9U &&
            inclusive_assist.diagnostics()
                .slot_selection.selected_index.has_value(),
        "inclusive-travel fixture did not select slot ID 9");
    const size_t inclusive_index =
        *inclusive_assist.diagnostics().slot_selection.selected_index;
    const interaction::MappedPickSlot& inclusive_slot =
        inclusive_assist.diagnostics().slot_selection.ordered[
            inclusive_index];
    require(
        inclusive_slot.root_world.position.x == 0.80F &&
            inclusive_slot.root_world.position.z == 0.50F,
        "inclusive-travel fixture did not map slot ID 9 to (0.80, 0.50)");

    constexpr float inclusive_travel_cap_m = 1.00002F;
    const vec3 start_endpoint =
        inclusive_scenario.start.root_world.position;
    inclusive_scenario.observation.displayed_root.position =
        vec3(0.50F, 10.0F, 0.50F);
    const vec3 middle_endpoint =
        inclusive_scenario.observation.displayed_root.position;
    inclusive_assist.observe(inclusive_scenario.observation);
    require(
        inclusive_assist.diagnostics().state ==
                interaction::PickAssistState::SlotApproach &&
            inclusive_assist.diagnostics().reason ==
                interaction::PickAssistReason::None,
        "first inclusive-travel endpoint left SlotApproach");

    inclusive_scenario.observation.displayed_root.position =
        vec3(0.50F, -10.0F, inclusive_travel_cap_m);
    const vec3 boundary_endpoint =
        inclusive_scenario.observation.displayed_root.position;
    inclusive_assist.observe(inclusive_scenario.observation);
    const float expected_assisted_travel_m =
        planar_endpoint_distance(start_endpoint, middle_endpoint) +
        planar_endpoint_distance(middle_endpoint, boundary_endpoint);
    const float start_to_boundary_planar_m =
        planar_endpoint_distance(start_endpoint, boundary_endpoint);
    const float expected_xyz_travel_m =
        xyz_endpoint_distance(start_endpoint, middle_endpoint) +
        xyz_endpoint_distance(middle_endpoint, boundary_endpoint);
    require(
        same_float_bits(
            expected_assisted_travel_m, inclusive_travel_cap_m),
        "inclusive endpoint arithmetic did not equal exactly 1.00002 m");
    require(
        !same_float_bits(
            start_to_boundary_planar_m, expected_assisted_travel_m),
        "inclusive endpoint arithmetic matched start-to-current planar distance");
    require(
        expected_xyz_travel_m > inclusive_travel_cap_m,
        "inclusive endpoint XYZ path did not exceed the travel cap");
    require(
        same_float_bits(
            inclusive_assist.diagnostics().assisted_travel_m,
            expected_assisted_travel_m),
        "inclusive travel did not accumulate consecutive planar endpoint distances");
    require(
        inclusive_assist.diagnostics().state ==
                interaction::PickAssistState::SlotApproach &&
            inclusive_assist.diagnostics().reason ==
                interaction::PickAssistReason::None,
        "exactly 1.00002 m assisted travel left SlotApproach");

    FrozenSlotScenario overshoot_scenario;
    configure_travel_scenario(overshoot_scenario);
    interaction::ControllerPickAssist overshoot_assist;
    require(
        overshoot_assist.begin(
            overshoot_scenario.start, &overshoot_scenario.target),
        "overshoot fixture begin failed");
    const interaction::PickAssistDiagnostics frozen_before =
        overshoot_assist.diagnostics();
    require(
        frozen_before.selected_slot_id == 9U &&
            frozen_before.slot_selection.selected_index.has_value(),
        "overshoot fixture did not select slot ID 9");
    const size_t frozen_index =
        *frozen_before.slot_selection.selected_index;
    const interaction::MappedPickSlot frozen_slot =
        frozen_before.slot_selection.ordered[frozen_index];
    const float first_overshoot = std::nextafter(
        inclusive_travel_cap_m,
        std::numeric_limits<float>::infinity());
    const vec3 overshoot_start_endpoint =
        overshoot_scenario.start.root_world.position;
    const vec3 overshoot_middle_endpoint(0.50F, 10.0F, 0.50F);
    const vec3 overshoot_endpoint(
        0.50F, -10.0F, first_overshoot);
    const float expected_overshoot_travel_m =
        planar_endpoint_distance(
            overshoot_start_endpoint, overshoot_middle_endpoint) +
        planar_endpoint_distance(
            overshoot_middle_endpoint, overshoot_endpoint);
    const float overshoot_start_to_current_planar_m =
        planar_endpoint_distance(
            overshoot_start_endpoint, overshoot_endpoint);
    const float expected_overshoot_xyz_travel_m =
        xyz_endpoint_distance(
            overshoot_start_endpoint, overshoot_middle_endpoint) +
        xyz_endpoint_distance(
            overshoot_middle_endpoint, overshoot_endpoint);
    require(
        same_float_bits(expected_overshoot_travel_m, first_overshoot),
        "overshoot endpoint arithmetic was not the first float above 1.00002 m");
    require(
        !same_float_bits(
            overshoot_start_to_current_planar_m,
            expected_overshoot_travel_m),
        "overshoot endpoint arithmetic matched start-to-current planar distance");
    require(
        expected_overshoot_xyz_travel_m > inclusive_travel_cap_m,
        "overshoot endpoint XYZ path did not exceed the travel cap");

    overshoot_scenario.observation.displayed_root.position =
        overshoot_middle_endpoint;
    overshoot_assist.observe(overshoot_scenario.observation);
    require(
        overshoot_assist.diagnostics().state ==
                interaction::PickAssistState::SlotApproach &&
            overshoot_assist.diagnostics().reason ==
                interaction::PickAssistReason::None,
        "first overshoot-fixture endpoint left SlotApproach");
    overshoot_scenario.observation.displayed_root.position =
        overshoot_endpoint;
    const interaction::PickAssistOutput overshoot_output =
        overshoot_assist.observe(overshoot_scenario.observation);
    const interaction::PickAssistDiagnostics& failed =
        overshoot_assist.diagnostics();
    require(
        same_float_bits(
            failed.assisted_travel_m, expected_overshoot_travel_m),
        "overshoot travel did not accumulate consecutive planar endpoint distances");
    require(
        failed.state == interaction::PickAssistState::Failed &&
            failed.reason ==
                interaction::PickAssistReason::OutsideTravelEnvelope,
        "first representable assisted-travel overshoot did not fail visibly");
    require(
        !overshoot_assist.active() &&
            !overshoot_assist.owns_manual_interact() &&
            !overshoot_output.submit_interact &&
            !overshoot_assist.take_submission(71U).has_value(),
        "assisted-travel overshoot retained ownership or submitted");
    require(
        failed.selected_slot_id == frozen_before.selected_slot_id &&
            failed.slot_selection.selected_index ==
                frozen_before.slot_selection.selected_index &&
            failed.slot_selection.ordered.size() ==
                frozen_before.slot_selection.ordered.size() &&
            same_float_bits(
                failed.route_length_m, frozen_before.route_length_m) &&
            same_transform_bits(
                failed.slot_selection.ordered[frozen_index].root_world,
                frozen_slot.root_world) &&
            same_float_bits(
                failed.slot_selection.ordered[frozen_index].route_length_m,
                frozen_slot.route_length_m),
        "assisted-travel overshoot reselected or changed frozen diagnostics");
}

void test_slot_approach_revalidates_frozen_route_against_table() {
    FrozenSlotScenario scenario;
    scenario.target.object_world = {vec3(), quat()};
    scenario.target.affordances.front().interaction_slots = {
        {9U, 0.80F, 0.0F, 0.0F},
        {11U, -0.80F, 0.0F, 0.0F},
        {12U, 0.0F, -0.90F, 0.0F},
    };
    scenario.target.table_world = {
        vec3(0.40F, 0.0F, 0.30F), quat()};
    scenario.target.table_size = vec3(0.02F, 0.10F, 0.02F);
    scenario.start.root_world = {vec3(), quat()};
    scenario.start.target_snapshot = scenario.target;
    scenario.observation.displayed_root = scenario.start.root_world;
    interaction::ControllerPickAssist assist;

    require(
        assist.begin(scenario.start, &scenario.target),
        "table-revalidation fixture begin failed");
    const interaction::PickAssistDiagnostics frozen_before =
        assist.diagnostics();
    require(
        frozen_before.state == interaction::PickAssistState::SlotApproach &&
            frozen_before.selected_slot_id == 9U &&
            frozen_before.slot_selection.selected_index.has_value(),
        "table-revalidation fixture did not select slot ID 9");
    const size_t selected =
        *frozen_before.slot_selection.selected_index;
    const interaction::MappedPickSlot& frozen_slot =
        frozen_before.slot_selection.ordered[selected];
    require(
        frozen_slot.root_world.position.x == 0.80F &&
            frozen_slot.root_world.position.y == 0.0F &&
            frozen_slot.root_world.position.z == 0.0F &&
            frozen_slot.route_length_m == 0.80F,
        "table-revalidation fixture froze unexpected slot geometry");

    scenario.observation.displayed_root.position =
        vec3(0.0F, 0.0F, 0.60F);
    const interaction::Transform current_root =
        scenario.observation.displayed_root;
    const auto alternate = std::find_if(
        frozen_before.slot_selection.ordered.begin(),
        frozen_before.slot_selection.ordered.end(),
        [](const interaction::MappedPickSlot& slot) {
            return slot.id == 11U;
        });
    require(
        alternate != frozen_before.slot_selection.ordered.end(),
        "table-revalidation fixture had no alternate slot ID 11");
    require(
        alternate->root_world.position.x == -0.80F &&
            alternate->root_world.position.y == 0.0F &&
            alternate->root_world.position.z == 0.0F &&
            alternate->route_length_m == frozen_slot.route_length_m,
        "table-revalidation fixture mapped an unexpected alternate endpoint");
    require(
        interaction::revalidate_frozen_pick_slot(
            current_root,
            frozen_slot.root_world,
            scenario.start.target_snapshot,
            scenario.start.obstacles) ==
                interaction::PickSlotReason::TableBlocked,
        "exact selected frozen table route was not TableBlocked");
    require(
        interaction::revalidate_frozen_pick_slot(
            current_root,
            current_root,
            scenario.start.target_snapshot,
            scenario.start.obstacles) == interaction::PickSlotReason::None,
        "current table endpoint was not clear");
    require(
        interaction::revalidate_frozen_pick_slot(
            current_root,
            alternate->root_world,
            scenario.start.target_snapshot,
            scenario.start.obstacles) == interaction::PickSlotReason::None,
        "alternate authored table endpoint was not clear");
    const interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    const interaction::PickAssistDiagnostics& failed =
        assist.diagnostics();
    require(
        failed.state == interaction::PickAssistState::Failed &&
            failed.reason == interaction::PickAssistReason::TableBlocked &&
            same_float_bits_exact(failed.assisted_travel_m, 0.60F),
        "newly blocked frozen route did not fail immediately with TableBlocked");
    require(
        !assist.active() && !assist.owns_manual_interact() &&
            !output.submit_interact &&
            !assist.take_submission(72U).has_value(),
        "table-blocked frozen route retained ownership or submitted");
    require_frozen_slot_provenance_unchanged(
        frozen_before,
        failed,
        "table-blocked frozen route reselected or changed frozen diagnostics");
}

void test_slot_approach_revalidates_frozen_route_against_obstacles() {
    FrozenSlotScenario scenario;
    scenario.target.object_world = {vec3(), quat()};
    scenario.target.affordances.front().interaction_slots = {
        {9U, 0.80F, 0.0F, 0.0F},
        {11U, -0.80F, 0.0F, 0.0F},
        {12U, 0.0F, -0.90F, 0.0F},
    };
    scenario.start.root_world = {
        vec3(0.0F, 10.0F, 0.0F), quat()};
    scenario.start.target_snapshot = scenario.target;
    scenario.start.obstacles.push_back({
        vec3(0.64F, 0.0F, 0.62F),
        vec3(0.02F, 0.10F, 0.02F),
    });
    scenario.observation.displayed_root = scenario.start.root_world;
    interaction::ControllerPickAssist assist;

    require(
        assist.begin(scenario.start, &scenario.target),
        "obstacle-revalidation fixture begin failed");
    const interaction::PickAssistDiagnostics frozen_before =
        assist.diagnostics();
    require(
        frozen_before.state == interaction::PickAssistState::SlotApproach &&
            frozen_before.selected_slot_id == 9U &&
            frozen_before.slot_selection.selected_index.has_value(),
        "obstacle-revalidation fixture did not select slot ID 9");
    const size_t selected =
        *frozen_before.slot_selection.selected_index;
    const interaction::MappedPickSlot& frozen_slot =
        frozen_before.slot_selection.ordered[selected];
    require(
        frozen_slot.root_world.position.x == 0.80F &&
            frozen_slot.root_world.position.y == 10.0F &&
            frozen_slot.root_world.position.z == 0.0F &&
            frozen_slot.route_length_m == 0.80F,
        "obstacle-revalidation fixture froze unexpected slot geometry");

    scenario.observation.displayed_root.position =
        vec3(0.0F, 0.0F, 0.60F);
    const interaction::Transform current_root =
        scenario.observation.displayed_root;
    const auto alternate = std::find_if(
        frozen_before.slot_selection.ordered.begin(),
        frozen_before.slot_selection.ordered.end(),
        [](const interaction::MappedPickSlot& slot) {
            return slot.id == 11U;
        });
    require(
        alternate != frozen_before.slot_selection.ordered.end(),
        "obstacle-revalidation fixture had no alternate slot ID 11");
    require(
        alternate->root_world.position.x == -0.80F &&
            alternate->root_world.position.y == 10.0F &&
            alternate->root_world.position.z == 0.0F &&
            alternate->route_length_m == frozen_slot.route_length_m,
        "obstacle-revalidation fixture mapped an unexpected alternate endpoint");
    require(
        interaction::revalidate_frozen_pick_slot(
            current_root,
            frozen_slot.root_world,
            scenario.start.target_snapshot,
            scenario.start.obstacles) ==
                interaction::PickSlotReason::ObstacleBlocked,
        "exact selected frozen obstacle route was not ObstacleBlocked");
    require(
        interaction::revalidate_frozen_pick_slot(
            current_root,
            current_root,
            scenario.start.target_snapshot,
            scenario.start.obstacles) == interaction::PickSlotReason::None,
        "current obstacle endpoint was not clear");
    require(
        interaction::revalidate_frozen_pick_slot(
            current_root,
            alternate->root_world,
            scenario.start.target_snapshot,
            scenario.start.obstacles) == interaction::PickSlotReason::None,
        "alternate authored obstacle endpoint was not clear");
    const interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    const interaction::PickAssistDiagnostics& failed =
        assist.diagnostics();
    require(
        failed.state == interaction::PickAssistState::Failed &&
            failed.reason == interaction::PickAssistReason::ObstacleBlocked &&
            same_float_bits_exact(failed.assisted_travel_m, 0.60F),
        "newly blocked frozen route did not fail immediately with ObstacleBlocked");
    require(
        !assist.active() && !assist.owns_manual_interact() &&
            !output.submit_interact &&
            !assist.take_submission(73U).has_value(),
        "obstacle-blocked frozen route retained ownership or submitted");
    require_frozen_slot_provenance_unchanged(
        frozen_before,
        failed,
        "obstacle-blocked frozen route reselected or changed frozen diagnostics");
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
    const std::array<const char*, 7> states{
        "Idle", "SlotApproach", "Settling", "FinalPreview",
        "ReadyToSubmit", "Submitted", "Failed"};
    for (size_t i = 0; i < states.size(); ++i) {
        require(
            std::string(interaction::pick_assist_state_name(
                static_cast<interaction::PickAssistState>(i))) == states[i],
            "state diagnostic name changed");
    }
    const std::array<const char*, 15> reasons{
        "None", "Cancelled", "TargetUnavailable", "TargetChanged",
        "SlotChanged", "RuntimeChanged", "NoAuthoredSlot",
        "InvalidGeometry", "OutsideTravelEnvelope", "TableBlocked",
        "ObstacleBlocked", "AllSlotsBlocked", "ArrivalDeadline",
        "PoorMatch", "FinalPreviewRejected"};
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
    config.maximum_arrival_ticks = 0U;
    require_invalid_config(config, "zero arrival tick count was accepted");

    config = {};
    config.maximum_assisted_path_m = 0.25F;
    const interaction::ControllerPickAssist smaller_cap(config);
    require(
        smaller_cap.diagnostics().state == interaction::PickAssistState::Idle,
        "valid smaller assisted-path cap was rejected");
}

void test_frozen_slot_cancel_clears_state_and_restarts() {
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist;
    require(
        assist.begin(scenario.start, &scenario.target),
        "frozen-slot cancel fixture begin failed");
    const interaction::PickAssistDiagnostics frozen_before =
        assist.diagnostics();

    scenario.observation.displayed_root.position.x = -0.60F;
    const interaction::PickAssistOutput steering_output =
        assist.observe(scenario.observation);
    require(
        steering_output.override_steering &&
            !is_zero(steering_output.left_stick) &&
            !steering_output.needs_preview &&
            !steering_output.stationary_constraint &&
            !steering_output.submit_interact &&
            assist.diagnostics().assisted_travel_m > 0.0F,
        "frozen-slot cancel fixture did not produce steering and travel");
    require_frozen_slot_provenance_unchanged(
        frozen_before,
        assist.diagnostics(),
        "frozen-slot cancel fixture changed provenance before cancellation");

    assist.cancel();
    require_zero_pick_assist_output(
        assist.observe(scenario.observation),
        "frozen-slot cancel emitted same-tick output");
    require_zero_pick_assist_output(
        assist.observe(scenario.observation),
        "frozen-slot cancel emitted subsequent output");

    const interaction::PickAssistDiagnostics& cancelled =
        assist.diagnostics();
    require(
        cancelled.state == interaction::PickAssistState::Idle &&
            cancelled.reason == interaction::PickAssistReason::Cancelled,
        "frozen-slot cancel did not remain visibly Idle/Cancelled");
    require(
        cancelled.target == interaction::TargetHandle{} &&
            cancelled.affordance_id == 0U &&
            cancelled.selected_slot_id == 0U,
        "frozen-slot cancel retained identity diagnostics");
    require(
        cancelled.slot_selection.ordered.empty() &&
            !cancelled.slot_selection.selected_index.has_value() &&
            cancelled.slot_selection.reason ==
                interaction::PickSlotReason::NoAuthoredSlot,
        "frozen-slot cancel retained selection diagnostics");
    require(
        cancelled.route_length_m == 0.0F &&
            cancelled.assisted_travel_m == 0.0F &&
            cancelled.object_origin_distance_m == 0.0F &&
            cancelled.object_bounds_center_distance_m == 0.0F &&
            cancelled.root_error_m == 0.0F &&
            cancelled.yaw_error_radians == 0.0F &&
            cancelled.speed_mps == 0.0F &&
            cancelled.settle_ticks == 0U,
        "frozen-slot cancel retained travel or motion diagnostics");
    require_final_preview_diagnostics_cleared(cancelled);
    require(
        !assist.active() && !assist.owns_manual_interact() &&
            !assist.take_submission(10U).has_value(),
        "frozen-slot cancel retained ownership or submission");

    FrozenSlotScenario restarted;
    require(
        assist.begin(restarted.start, &restarted.target) &&
            assist.diagnostics().state ==
                interaction::PickAssistState::SlotApproach &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None,
        "frozen-slot cancel did not permit an immediate valid restart");
}

}  // namespace

int main() {
    try {
        test_idle_does_not_override_input();
        test_begin_selects_and_freezes_one_authored_slot();
        test_slot_approach_emits_far_camera_relative_steering();
        test_slot_approach_runtime_change_precedes_target_and_metrics();
        test_slot_approach_target_unavailable_failures_are_stable();
        test_slot_approach_target_snapshot_failures_are_stable();
        test_slot_approach_slot_identity_failures_precede_snapshot_mismatch();
        test_slot_approach_nonfinite_observation_failures_are_stable();
        test_slot_approach_unchanged_identity_still_steers();
        test_slot_approach_emits_slow_radius_arrival_steering();
        test_slot_approach_latches_inclusive_arrival_boundaries();
        test_frozen_slot_settling_revalidates_selected_slot();
        test_frozen_slot_settling_enforces_consecutive_travel();
        test_frozen_slot_settling_keeps_stationary_braking();
        test_frozen_slot_settling_requests_exact_frozen_preview_root();
        test_frozen_final_preview_missing_singular_preview_rerequests();
        test_frozen_final_preview_poor_match_rerequests_then_recovers();
        test_frozen_final_preview_hard_rejections_are_terminal();
        test_frozen_final_preview_certifies_and_submits_exactly_once();
        test_frozen_ready_to_submit_runs_shared_preflight();
        test_frozen_final_preview_missing_preview_hits_arrival_deadline();
        test_frozen_final_preview_poor_match_deadline_memory_survives_missing();
        test_frozen_final_preview_deadline_precedes_certification();
        test_slot_approach_rejects_adjacent_arrival_overshoots();
        test_slot_approach_accumulates_inclusive_travel_and_rejects_overshoot();
        test_slot_approach_revalidates_frozen_route_against_table();
        test_slot_approach_revalidates_frozen_route_against_obstacles();
        test_begin_maps_every_aggregate_no_winner_reason();
        test_slot_reason_mapping_is_exhaustive_and_same_named();
        test_failed_begin_can_immediately_begin_a_valid_attempt();
        test_diagnostic_names_are_stable();
        test_constructor_rejects_unsafe_or_nonfinite_config();
        test_frozen_slot_cancel_clears_state_and_restarts();
        std::cout << "interaction_pick_assist tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "interaction_pick_assist test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
