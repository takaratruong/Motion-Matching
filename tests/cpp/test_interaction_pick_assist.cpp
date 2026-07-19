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
#include <utility>

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

#define DEFINE_ENUMERATOR_TRAIT(trait_name, enumerator_name) \
    template <typename Type, typename = void> \
    struct trait_name : std::false_type {}; \
    template <typename Type> \
    struct trait_name< \
        Type, \
        std::void_t<decltype(Type::enumerator_name)>> : std::true_type {};

DEFINE_ENUMERATOR_TRAIT(has_state_coarse_approach, CoarseApproach)
DEFINE_ENUMERATOR_TRAIT(has_state_preview, Preview)
DEFINE_ENUMERATOR_TRAIT(has_state_final_approach, FinalApproach)
DEFINE_ENUMERATOR_TRAIT(has_reason_no_feasible_entry, NoFeasibleEntry)
DEFINE_ENUMERATOR_TRAIT(has_reason_preview_deadline, PreviewDeadline)

#undef DEFINE_ENUMERATOR_TRAIT

template <typename Type, typename = void>
struct has_one_argument_begin : std::false_type {};

template <typename Type>
struct has_one_argument_begin<
    Type,
    std::void_t<decltype(std::declval<Type&>().begin(
        std::declval<const interaction::PickAssistStart&>()))>>
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
    !has_state_coarse_approach<interaction::PickAssistState>::value,
    "PickAssistState::CoarseApproach must be absent");
static_assert(
    !has_state_preview<interaction::PickAssistState>::value,
    "PickAssistState::Preview must be absent");
static_assert(
    !has_state_final_approach<interaction::PickAssistState>::value,
    "PickAssistState::FinalApproach must be absent");
static_assert(
    !has_reason_no_feasible_entry<interaction::PickAssistReason>::value,
    "PickAssistReason::NoFeasibleEntry must be absent");
static_assert(
    !has_reason_preview_deadline<interaction::PickAssistReason>::value,
    "PickAssistReason::PreviewDeadline must be absent");

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
    static_cast<uint8_t>(
        interaction::PickAssistState::SlotSelectionPreview) == 7U);

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
static_assert(
    static_cast<uint8_t>(
        interaction::PickAssistReason::SelectionPreviewRejected) == 15U);

static_assert(std::is_same_v<
    decltype(interaction::PickAssistOutput{}.preview_requests),
    std::vector<interaction::PickAssistPreviewRequest>>);
static_assert(std::is_same_v<
    decltype(interaction::PickAssistObservation{}.preview_results),
    std::vector<interaction::PickAssistPreviewResult>>);
static_assert(std::is_same_v<
    decltype(interaction::PickAssistDiagnostics{}.frozen_slot_index),
    std::optional<size_t>>);
static_assert(std::is_same_v<
    decltype(interaction::PickAssistDiagnostics{}.selection_previews),
    std::vector<interaction::PickAssistSelectionPreviewDiagnostics>>);

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

interaction::PickEntryRoot mapped_entry_root(
    const interaction::MappedPickSlot& slot) {
    const vec3 forward = quat_mul_vec3(
        slot.root_world.rotation, vec3(0.0F, 0.0F, 1.0F));
    return {
        slot.root_world.position.x,
        slot.root_world.position.z,
        std::atan2(forward.x, forward.z),
    };
}

interaction::PickAssistObservation make_selection_observation(
    const interaction::InteractionTarget& target,
    interaction::Transform displayed_root) {
    interaction::PickAssistObservation observation{};
    observation.target = &target;
    observation.displayed_root = displayed_root;
    return observation;
}

enum class SelectionPreviewKind : uint8_t {
    Ready,
    PoorMatch,
    CorrectionLimit,
    BlockedPath,
    Missing,
};

std::optional<interaction::PickEntryPreview> selection_preview(
    interaction::PickEntryRoot root,
    SelectionPreviewKind kind,
    float cost = 1.0F) {
    if (kind == SelectionPreviewKind::Missing) return std::nullopt;
    interaction::PickEntryPreview preview{};
    preview.path_feasible = true;
    preview.path_reason = interaction::Reason::None;
    preview.match_ready = true;
    preview.match_reason = interaction::Reason::None;
    preview.prospective_root = root;
    preview.feasible_entry_frame = 114;
    preview.contact_frame = 139;
    preview.total_cost = cost;
    switch (kind) {
    case SelectionPreviewKind::Ready:
    case SelectionPreviewKind::Missing:
        break;
    case SelectionPreviewKind::PoorMatch:
        preview.match_ready = false;
        preview.match_reason = interaction::Reason::PoorMatch;
        break;
    case SelectionPreviewKind::CorrectionLimit:
        preview.match_ready = false;
        preview.match_reason = interaction::Reason::CorrectionLimit;
        break;
    case SelectionPreviewKind::BlockedPath:
        preview.path_feasible = false;
        preview.path_reason = interaction::Reason::BlockedPath;
        preview.match_ready = false;
        preview.match_reason = interaction::Reason::BlockedPath;
        break;
    }
    return preview;
}

void set_selection_results(
    interaction::PickAssistObservation& observation,
    const interaction::PickAssistOutput& requested,
    const std::vector<SelectionPreviewKind>& kinds,
    uint64_t fingerprint,
    const std::vector<float>& costs = {}) {
    require(
        kinds.size() <= requested.preview_requests.size(),
        "selection-result fixture supplied too many result kinds");
    require(
        costs.empty() || costs.size() == kinds.size(),
        "selection-result fixture cost count changed");
    observation.snapshot_fingerprint = fingerprint;
    observation.preview_snapshot_fingerprint = fingerprint;
    observation.preview_results.clear();
    for (size_t index = 0U; index < kinds.size(); ++index) {
        const interaction::PickAssistPreviewRequest& request =
            requested.preview_requests[index];
        observation.preview_results.push_back({
            request,
            selection_preview(
                request.root,
                kinds[index],
                costs.empty() ? 1.0F : costs[index]),
        });
    }
}

void require_ranked_selection_requests(
    const interaction::PickAssistOutput& output,
    const interaction::PickAssistDiagnostics& diagnostics,
    const char* message) {
    const std::vector<size_t>& ranked =
        diagnostics.slot_selection.ranked_eligible_indices;
    require(
        output.override_steering && output.force_strafe &&
            output.stationary_constraint && is_zero(output.left_stick) &&
            is_zero(output.right_stick) && !output.submit_interact &&
            output.preview_requests.size() == ranked.size(),
        message);
    for (size_t rank = 0U; rank < ranked.size(); ++rank) {
        const interaction::MappedPickSlot& slot =
            diagnostics.slot_selection.ordered[ranked[rank]];
        require(
            output.preview_requests[rank].slot_id == slot.id &&
                same_pick_entry_root_bits_exact(
                    output.preview_requests[rank].root,
                    mapped_entry_root(slot)),
            message);
    }
}

bool begin_and_certify_geometry_winner(
    interaction::ControllerPickAssist& assist,
    const interaction::PickAssistStart& start,
    const interaction::InteractionTarget* target) {
    if (!assist.begin(start, target) || target == nullptr) return false;
    interaction::PickAssistObservation observation =
        make_selection_observation(*target, start.root_world);
    const interaction::PickAssistOutput requested =
        assist.observe(observation);
    if (requested.preview_requests.empty()) return false;
    set_selection_results(
        observation,
        requested,
        std::vector<SelectionPreviewKind>(
            requested.preview_requests.size(),
            SelectionPreviewKind::Ready),
        0x5e1ec7U);
    (void)assist.observe(observation);
    return assist.diagnostics().frozen_slot_index.has_value();
}

void require_zero_pick_assist_output(
    const interaction::PickAssistOutput& output,
    const char* message) {
    require(
        !output.override_steering && is_zero(output.left_stick) &&
            is_zero(output.right_stick) && !output.force_strafe &&
            !output.stationary_constraint &&
            output.preview_requests.empty() && !output.submit_interact,
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
            output.preview_requests.size() == 1U &&
            output.preview_requests.front().slot_id == 9U &&
            same_pick_entry_root_bits_exact(
                output.preview_requests.front().root, expected_root) &&
            !output.submit_interact,
        message);
}

interaction::PickAssistPreviewRequest frozen_preview_request(
    interaction::PickEntryRoot root) {
    return {9U, root};
}

void set_frozen_preview_result(
    interaction::PickAssistObservation& observation,
    interaction::PickEntryRoot request_root,
    std::optional<interaction::PickEntryPreview> preview) {
    observation.preview_results = {{
        frozen_preview_request(request_root), std::move(preview)}};
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
    require(output.preview_requests.empty(), "Idle requested a preview");
    require(!output.submit_interact, "Idle submitted interaction");
    require(!assist.active(), "Idle reported active");
    require(!assist.owns_manual_interact(),
        "Idle owned manual interaction");
    require(
        assist.diagnostics().state == interaction::PickAssistState::Idle,
        "Idle diagnostics reported the wrong state");
}

void test_begin_defers_freeze_and_requests_every_ranked_entry() {
    const interaction::InteractionTarget target = make_frozen_slot_target();
    const interaction::PickAssistStart start = make_frozen_slot_start(target);
    interaction::ControllerPickAssist assist;

    require(assist.begin(start, &target), "selection-preview begin failed");
    const interaction::PickAssistDiagnostics& began = assist.diagnostics();
    require(
        began.state == interaction::PickAssistState::SlotSelectionPreview &&
            began.reason == interaction::PickAssistReason::None &&
            began.selected_slot_id == 0U &&
            !began.frozen_slot_index.has_value() &&
            began.slot_selection.selected_index.has_value() &&
            began.slot_selection.ranked_eligible_indices.size() == 3U &&
            began.route_length_m == 0.0F &&
            began.object_origin_distance_m == 0.0F &&
            began.object_bounds_center_distance_m == 0.0F,
        "begin froze geometry-only selection before motion certification");
    const size_t legacy = *began.slot_selection.selected_index;
    require(
        began.slot_selection.ordered[legacy].id == 9U,
        "begin changed the legacy geometry-only winner diagnostic");

    interaction::PickAssistObservation observation =
        make_selection_observation(target, start.root_world);
    observation.snapshot_fingerprint = 0x101U;
    const interaction::PickAssistOutput output = assist.observe(observation);
    require_ranked_selection_requests(
        output,
        assist.diagnostics(),
        "activation did not emit every ranked geometry-eligible request");
    const interaction::PickAssistDiagnostics& requested =
        assist.diagnostics();
    require(
        requested.state ==
                interaction::PickAssistState::SlotSelectionPreview &&
            requested.selected_slot_id == 0U &&
            !requested.frozen_slot_index.has_value() &&
            requested.selection_preview_epochs == 1U &&
            requested.selection_preview_calls == 0U &&
            !requested.selection_poor_match_observed &&
            requested.selection_previews.size() ==
                output.preview_requests.size(),
        "activation changed freeze state or selection counters");
    for (size_t index = 0U;
         index < requested.selection_previews.size();
         ++index) {
        const auto& preview = requested.selection_previews[index];
        require(
            preview.request.slot_id ==
                    output.preview_requests[index].slot_id &&
                same_pick_entry_root_bits_exact(
                    preview.request.root,
                    output.preview_requests[index].root) &&
                preview.observation_snapshot_fingerprint == 0x101U &&
                preview.preview_snapshot_fingerprint == 0U &&
                !preview.available &&
                !preview.prospective_root.has_value(),
            "initial selection diagnostics did not retain request provenance");
    }
}

void test_selection_freezes_first_ranked_ready_entry_not_lowest_cost() {
    const interaction::InteractionTarget target = make_frozen_slot_target();
    const interaction::PickAssistStart start = make_frozen_slot_start(target);
    interaction::ControllerPickAssist assist;
    require(assist.begin(start, &target), "ranked-ready begin failed");
    interaction::PickAssistObservation observation =
        make_selection_observation(target, start.root_world);
    const interaction::PickAssistOutput requested =
        assist.observe(observation);
    require(
        requested.preview_requests.size() == 3U &&
            requested.preview_requests[0].slot_id == 9U &&
            requested.preview_requests[1].slot_id == 12U &&
            requested.preview_requests[2].slot_id == 11U,
        "ranked-ready fixture did not request 9,12,11 geometry order");
    set_selection_results(
        observation,
        requested,
        {
            SelectionPreviewKind::CorrectionLimit,
            SelectionPreviewKind::Ready,
            SelectionPreviewKind::Ready,
        },
        0x202U,
        {0.01F, 99.0F, 0.10F});

    const interaction::PickAssistOutput output =
        assist.observe(observation);
    const interaction::PickAssistDiagnostics& frozen =
        assist.diagnostics();
    require(
        frozen.state == interaction::PickAssistState::SlotApproach &&
            frozen.reason == interaction::PickAssistReason::None &&
            frozen.frozen_slot_index.has_value() &&
            *frozen.frozen_slot_index == 0U &&
            frozen.selected_slot_id == 12U &&
            frozen.slot_selection.selected_index.has_value() &&
            frozen.slot_selection.ordered[
                *frozen.slot_selection.selected_index].id == 9U &&
            frozen.selection_preview_epochs == 1U &&
            frozen.selection_preview_calls == 3U &&
            output.preview_requests.empty() &&
            output.override_steering,
        "selection did not freeze the first ranked ready entry exactly once");
    require(
        frozen.selection_previews.size() == 3U &&
            !frozen.selection_previews[0].match_ready &&
            frozen.selection_previews[0].match_reason ==
                interaction::Reason::CorrectionLimit &&
            frozen.selection_previews[1].match_ready &&
            frozen.selection_previews[1].total_cost == 99.0F &&
            frozen.selection_previews[2].match_ready &&
            frozen.selection_previews[2].total_cost == 0.10F,
        "selection diagnostics lost ranked-ready cost evidence");
    const std::optional<size_t> frozen_slot_index =
        frozen.frozen_slot_index;
    const uint32_t frozen_slot_id = frozen.selected_slot_id;

    interaction::PickAssistObservation later =
        make_selection_observation(target, start.root_world);
    set_selection_results(
        later,
        requested,
        std::vector<SelectionPreviewKind>(
            requested.preview_requests.size(),
            SelectionPreviewKind::Ready),
        0x203U);
    (void)assist.observe(later);
    require(
        assist.diagnostics().frozen_slot_index == frozen_slot_index &&
            assist.diagnostics().selected_slot_id == frozen_slot_id,
        "post-freeze preview evidence switched the selected slot");
}

void test_selection_incomplete_batches_retry_atomically_then_deadline() {
    interaction::PickAssistConfig config{};
    config.maximum_arrival_ticks = 3U;
    const interaction::InteractionTarget target = make_frozen_slot_target();
    const interaction::PickAssistStart start = make_frozen_slot_start(target);
    interaction::ControllerPickAssist assist(config);
    require(assist.begin(start, &target), "incomplete-batch begin failed");
    interaction::PickAssistObservation observation =
        make_selection_observation(target, start.root_world);
    interaction::PickAssistOutput requested = assist.observe(observation);

    set_selection_results(
        observation,
        requested,
        {SelectionPreviewKind::Ready},
        0x301U);
    interaction::PickAssistOutput output = assist.observe(observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::SlotSelectionPreview &&
            !assist.diagnostics().frozen_slot_index.has_value() &&
            assist.diagnostics().selection_preview_epochs == 2U &&
            assist.diagnostics().selection_preview_calls == 1U,
        "partial ready batch selected a partial winner");
    require_ranked_selection_requests(
        output,
        assist.diagnostics(),
        "partial ready batch did not re-request the entire batch");

    requested = output;
    observation.preview_results.clear();
    observation.snapshot_fingerprint = 0x302U;
    observation.preview_snapshot_fingerprint = 0U;
    output = assist.observe(observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::SlotSelectionPreview &&
            assist.diagnostics().selection_preview_epochs == 3U &&
            assist.diagnostics().selection_preview_calls == 1U,
        "empty callback epoch reached the deadline early");
    require_ranked_selection_requests(
        output,
        assist.diagnostics(),
        "empty callback epoch did not re-request all entries");

    observation.preview_results.clear();
    observation.snapshot_fingerprint = 0x303U;
    output = assist.observe(observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::ArrivalDeadline &&
            !assist.diagnostics().frozen_slot_index.has_value() &&
            output.preview_requests.empty(),
        "third incomplete selection epoch did not fail ArrivalDeadline");
}

void test_selection_mixed_poor_match_retries_until_poor_match_deadline() {
    interaction::PickAssistConfig config{};
    config.maximum_arrival_ticks = 3U;
    const interaction::InteractionTarget target = make_frozen_slot_target();
    const interaction::PickAssistStart start = make_frozen_slot_start(target);
    interaction::ControllerPickAssist assist(config);
    require(assist.begin(start, &target), "PoorMatch selection begin failed");
    interaction::PickAssistObservation observation =
        make_selection_observation(target, start.root_world);
    interaction::PickAssistOutput requested = assist.observe(observation);

    set_selection_results(
        observation,
        requested,
        {
            SelectionPreviewKind::CorrectionLimit,
            SelectionPreviewKind::PoorMatch,
            SelectionPreviewKind::BlockedPath,
        },
        0x401U);
    interaction::PickAssistOutput output = assist.observe(observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::SlotSelectionPreview &&
            assist.diagnostics().selection_poor_match_observed &&
            !assist.diagnostics().frozen_slot_index.has_value(),
        "mixed hard/PoorMatch epoch did not retry atomically");
    require_ranked_selection_requests(
        output,
        assist.diagnostics(),
        "mixed hard/PoorMatch epoch did not request every entry again");

    for (uint64_t fingerprint : {0x402U, 0x403U}) {
        requested = output;
        set_selection_results(
            observation,
            requested,
            std::vector<SelectionPreviewKind>(
                requested.preview_requests.size(),
                SelectionPreviewKind::PoorMatch),
            fingerprint);
        output = assist.observe(observation);
    }
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::PoorMatch &&
            assist.diagnostics().selection_preview_epochs == 3U &&
            assist.diagnostics().selection_preview_calls == 9U &&
            !assist.diagnostics().frozen_slot_index.has_value() &&
            output.preview_requests.empty(),
        "PoorMatch epochs did not terminate with stable PoorMatch reason");
}

void test_selection_complete_hard_batch_rejects() {
    const interaction::InteractionTarget target = make_frozen_slot_target();
    const interaction::PickAssistStart start = make_frozen_slot_start(target);
    interaction::ControllerPickAssist assist;
    require(assist.begin(start, &target), "hard-batch begin failed");
    interaction::PickAssistObservation observation =
        make_selection_observation(target, start.root_world);
    const interaction::PickAssistOutput requested =
        assist.observe(observation);
    set_selection_results(
        observation,
        requested,
        {
            SelectionPreviewKind::CorrectionLimit,
            SelectionPreviewKind::BlockedPath,
            SelectionPreviewKind::CorrectionLimit,
        },
        0x501U);
    const interaction::PickAssistOutput output = assist.observe(observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::SelectionPreviewRejected &&
            !assist.diagnostics().frozen_slot_index.has_value() &&
            output.preview_requests.empty(),
        "complete hard-rejected batch did not fail closed");
}

void test_selection_malformed_batches_fail_closed() {
    enum class Defect : uint8_t {
        Misordered,
        Duplicate,
        WrongSlot,
        WrongRequestRoot,
        WrongFingerprint,
        NonfiniteProspectiveRoot,
        NanCost,
        InfiniteCost,
        ExtraResult,
    };
    const std::array<Defect, 9> defects{
        Defect::Misordered,
        Defect::Duplicate,
        Defect::WrongSlot,
        Defect::WrongRequestRoot,
        Defect::WrongFingerprint,
        Defect::NonfiniteProspectiveRoot,
        Defect::NanCost,
        Defect::InfiniteCost,
        Defect::ExtraResult,
    };
    for (Defect defect : defects) {
        const interaction::InteractionTarget target =
            make_frozen_slot_target();
        const interaction::PickAssistStart start =
            make_frozen_slot_start(target);
        interaction::ControllerPickAssist assist;
        require(assist.begin(start, &target), "malformed-batch begin failed");
        interaction::PickAssistObservation observation =
            make_selection_observation(target, start.root_world);
        const interaction::PickAssistOutput requested =
            assist.observe(observation);
        set_selection_results(
            observation,
            requested,
            std::vector<SelectionPreviewKind>(
                requested.preview_requests.size(),
                SelectionPreviewKind::Ready),
            0x601U);
        switch (defect) {
        case Defect::Misordered:
            std::swap(
                observation.preview_results[0],
                observation.preview_results[1]);
            break;
        case Defect::Duplicate:
            observation.preview_results[1] =
                observation.preview_results[0];
            break;
        case Defect::WrongSlot:
            ++observation.preview_results[0].request.slot_id;
            break;
        case Defect::WrongRequestRoot:
            observation.preview_results[0].request.root.world_x =
                std::nextafter(
                    observation.preview_results[0]
                        .request.root.world_x,
                    std::numeric_limits<float>::infinity());
            break;
        case Defect::WrongFingerprint:
            observation.preview_snapshot_fingerprint = 0x602U;
            break;
        case Defect::NonfiniteProspectiveRoot:
            observation.preview_results[0]
                .preview->prospective_root.world_x =
                    float_from_bits(0x7fc00001U);
            break;
        case Defect::NanCost:
            observation.preview_results[0].preview->total_cost =
                float_from_bits(0x7fc00001U);
            break;
        case Defect::InfiniteCost:
            observation.preview_results[0].preview->total_cost =
                float_from_bits(0x7f800000U);
            break;
        case Defect::ExtraResult:
            observation.preview_results.push_back(
                observation.preview_results.back());
            break;
        }
        const interaction::PickAssistOutput output =
            assist.observe(observation);
        require(
            assist.diagnostics().state ==
                    interaction::PickAssistState::Failed &&
                assist.diagnostics().reason ==
                    interaction::PickAssistReason::SelectionPreviewRejected &&
                !assist.diagnostics().frozen_slot_index.has_value() &&
                output.preview_requests.empty(),
            "malformed selection batch did not fail closed");
    }

    const interaction::InteractionTarget target =
        make_frozen_slot_target();
    const interaction::PickAssistStart start =
        make_frozen_slot_start(target);
    interaction::ControllerPickAssist assist;
    require(
        assist.begin(start, &target),
        "partial-malformed selection begin failed");
    interaction::PickAssistObservation observation =
        make_selection_observation(target, start.root_world);
    const interaction::PickAssistOutput requested =
        assist.observe(observation);
    set_selection_results(
        observation,
        requested,
        {SelectionPreviewKind::Ready},
        0x603U);
    observation.preview_results.front()
        .preview->prospective_root.world_x =
            float_from_bits(0x7fc00001U);
    const interaction::PickAssistOutput partial_malformed =
        assist.observe(observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::SelectionPreviewRejected &&
            !assist.diagnostics().frozen_slot_index.has_value() &&
            partial_malformed.preview_requests.empty(),
        "partial batch hid malformed present evidence behind retry");
}

void test_selection_latches_poor_match_before_ready_deadline() {
    interaction::PickAssistConfig config{};
    config.maximum_arrival_ticks = 1U;
    const interaction::InteractionTarget target = make_frozen_slot_target();
    const interaction::PickAssistStart start = make_frozen_slot_start(target);
    interaction::ControllerPickAssist assist(config);
    require(
        assist.begin(start, &target),
        "ready/PoorMatch deadline begin failed");
    interaction::PickAssistObservation observation =
        make_selection_observation(target, start.root_world);
    const interaction::PickAssistOutput requested =
        assist.observe(observation);
    set_selection_results(
        observation,
        requested,
        {
            SelectionPreviewKind::Ready,
            SelectionPreviewKind::PoorMatch,
            SelectionPreviewKind::CorrectionLimit,
        },
        0x604U);

    const interaction::PickAssistOutput output = assist.observe(observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::PoorMatch &&
            assist.diagnostics().selection_poor_match_observed &&
            !assist.diagnostics().frozen_slot_index.has_value() &&
            output.preview_requests.empty(),
        "ready/PoorMatch deadline did not retain truthful PoorMatch evidence");
}

void test_selection_target_affordance_and_slot_mutations_fail_closed() {
    enum class Mutation : uint8_t { Target, Affordance, Slot };
    for (Mutation mutation :
         {Mutation::Target, Mutation::Affordance, Mutation::Slot}) {
        interaction::InteractionTarget target = make_frozen_slot_target();
        const interaction::PickAssistStart start =
            make_frozen_slot_start(target);
        interaction::ControllerPickAssist assist;
        require(assist.begin(start, &target), "mutation begin failed");
        interaction::PickAssistObservation observation =
            make_selection_observation(target, start.root_world);
        const interaction::PickAssistOutput requested =
            assist.observe(observation);
        set_selection_results(
            observation,
            requested,
            std::vector<SelectionPreviewKind>(
                requested.preview_requests.size(),
                SelectionPreviewKind::Ready),
            0x701U);
        switch (mutation) {
        case Mutation::Target:
            ++target.handle.generation;
            break;
        case Mutation::Affordance:
            target.affordances.front().hand_in_object.position.x =
                std::nextafter(0.0F, 1.0F);
            break;
        case Mutation::Slot:
            target.affordances.front().interaction_slots.front()
                .root_x_object_m = std::nextafter(0.70F, 1.0F);
            break;
        }
        const interaction::PickAssistOutput output =
            assist.observe(observation);
        const interaction::PickAssistReason expected =
            mutation == Mutation::Slot
                ? interaction::PickAssistReason::SlotChanged
                : interaction::PickAssistReason::TargetChanged;
        require(
            assist.diagnostics().state ==
                    interaction::PickAssistState::Failed &&
                assist.diagnostics().reason == expected &&
                !assist.diagnostics().frozen_slot_index.has_value() &&
                output.preview_requests.empty(),
            "selection provenance mutation did not fail with stable reason");
    }
}

void test_selection_revalidates_each_candidate_route_before_freeze() {
    for (bool obstacle_case : {false, true}) {
        interaction::InteractionTarget target = make_frozen_slot_target();
        target.object_world = {vec3(), quat()};
        target.affordances.front().interaction_slots = {
            {9U, 0.80F, 0.0F, 0.0F},
            {11U, -0.80F, 0.0F, 0.0F},
            {12U, 0.0F, -0.90F, 0.0F},
        };
        interaction::PickAssistStart start =
            make_frozen_slot_start(target);
        start.root_world = obstacle_case
            ? interaction::Transform{vec3(0.0F, 10.0F, 0.0F), quat()}
            : interaction::Transform{vec3(), quat()};
        if (obstacle_case) {
            start.obstacles.push_back({
                vec3(0.64F, 0.0F, 0.62F),
                vec3(0.02F, 0.10F, 0.02F),
            });
        } else {
            target.table_world = {
                vec3(0.40F, 0.0F, 0.30F), quat()};
            target.table_size = vec3(0.02F, 0.10F, 0.02F);
            start.target_snapshot = target;
        }
        interaction::ControllerPickAssist assist;
        require(assist.begin(start, &target), "route-revalidation begin failed");
        interaction::PickAssistObservation observation =
            make_selection_observation(target, start.root_world);
        const interaction::PickAssistOutput requested =
            assist.observe(observation);
        require(
            requested.preview_requests.size() == 3U &&
                requested.preview_requests[0].slot_id == 9U &&
                requested.preview_requests[1].slot_id == 11U,
            "route-revalidation fixture did not rank slots 9 then 11");
        observation.displayed_root.position =
            vec3(0.0F, 0.0F, 0.60F);
        set_selection_results(
            observation,
            requested,
            std::vector<SelectionPreviewKind>(
                requested.preview_requests.size(),
                SelectionPreviewKind::Ready),
            obstacle_case ? 0x802U : 0x801U);
        const interaction::PickAssistOutput output =
            assist.observe(observation);
        require(
            assist.diagnostics().frozen_slot_index.has_value() &&
                assist.diagnostics().selected_slot_id == 11U &&
                assist.diagnostics().state ==
                    interaction::PickAssistState::SlotApproach &&
                output.preview_requests.empty(),
            obstacle_case
                ? "obstacle-blocked candidate certified over clear alternate"
                : "table-blocked candidate certified over clear alternate");
    }
}

void test_selection_epochs_do_not_consume_frozen_arrival_budget() {
    interaction::PickAssistConfig config{};
    config.required_settle_ticks = 2U;
    config.maximum_arrival_ticks = 4U;
    const interaction::InteractionTarget target = make_frozen_slot_target();
    const interaction::PickAssistStart start = make_frozen_slot_start(target);
    interaction::ControllerPickAssist assist(config);
    require(assist.begin(start, &target), "arrival-reset begin failed");
    interaction::PickAssistObservation observation =
        make_selection_observation(target, start.root_world);
    interaction::PickAssistOutput requested = assist.observe(observation);
    for (uint64_t fingerprint : {0x901U, 0x902U}) {
        set_selection_results(
            observation,
            requested,
            std::vector<SelectionPreviewKind>(
                requested.preview_requests.size(),
                SelectionPreviewKind::PoorMatch),
            fingerprint);
        requested = assist.observe(observation);
        require_ranked_selection_requests(
            requested,
            assist.diagnostics(),
            "pre-freeze PoorMatch did not retain full request batch");
    }
    set_selection_results(
        observation,
        requested,
        std::vector<SelectionPreviewKind>(
            requested.preview_requests.size(),
            SelectionPreviewKind::Ready),
        0x903U);
    (void)assist.observe(observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::SlotApproach &&
            assist.diagnostics().selected_slot_id == 9U,
        "arrival-reset fixture did not freeze after two preview retries");
    const size_t frozen_index = *assist.diagnostics().frozen_slot_index;
    observation.preview_results.clear();
    observation.displayed_root = assist.diagnostics()
        .slot_selection.ordered[frozen_index].root_world;
    (void)assist.observe(observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::Settling,
        "arrival-reset fixture did not latch frozen root");
    (void)assist.observe(observation);
    const interaction::PickAssistOutput final_settle =
        assist.observe(observation);
    require(
        assist.diagnostics().state == interaction::PickAssistState::FinalPreview &&
            assist.diagnostics().reason == interaction::PickAssistReason::None &&
            final_settle.preview_requests.size() == 1U,
        "selection epochs leaked into the frozen arrival deadline budget");
}

void test_begin_selects_and_freezes_one_authored_slot() {
    const interaction::InteractionTarget target = make_frozen_slot_target();
    const interaction::PickAssistStart start = make_frozen_slot_start(target);
    interaction::ControllerPickAssist assist;

    require(
        begin_and_certify_geometry_winner(
            assist, start, &target),
        "valid frozen-slot begin/certification failed");
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

void test_begin_rejects_duplicate_affordance_ids_immediately() {
    interaction::InteractionTarget duplicate_target =
        make_frozen_slot_target();
    interaction::GraspAffordance duplicate_affordance =
        duplicate_target.affordances.front();
    duplicate_affordance.interaction_slots = {
        {99U, 0.25F, 0.25F, 0.50F},
    };
    duplicate_target.affordances.push_back(duplicate_affordance);
    const interaction::PickAssistStart duplicate_start =
        make_frozen_slot_start(duplicate_target);
    interaction::ControllerPickAssist assist;

    require(
        !assist.begin(duplicate_start, &duplicate_target),
        "duplicate affordance IDs were accepted by begin");
    require(
        assist.diagnostics().state == interaction::PickAssistState::Failed &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::TargetChanged,
        "duplicate affordance IDs did not fail begin as TargetChanged");
    require(
        !assist.active() && !assist.owns_manual_interact() &&
            !assist.take_submission(404U).has_value(),
        "duplicate-affordance begin failure retained ownership or submitted");

    const interaction::InteractionTarget unique_target =
        make_frozen_slot_target();
    const interaction::PickAssistStart unique_start =
        make_frozen_slot_start(unique_target);
    require(
        begin_and_certify_geometry_winner(
            assist, unique_start, &unique_target) &&
            assist.diagnostics().state ==
                interaction::PickAssistState::SlotApproach &&
            assist.diagnostics().reason == interaction::PickAssistReason::None &&
            assist.active() && assist.owns_manual_interact(),
        "duplicate-affordance begin failure prevented a valid unique restart");
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
        begin_and_certify_geometry_winner(
            assist, scenario.start, &scenario.target),
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
        output.preview_requests.empty() && is_zero(output.right_stick) &&
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
        begin_and_certify_geometry_winner(
            rotated_assist,
            rotated_scenario.start,
            &rotated_scenario.target),
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
        rotated_output.preview_requests.empty() &&
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
        begin_and_certify_geometry_winner(
            assist, scenario.start, &scenario.target),
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
        begin_and_certify_geometry_winner(
            assist, scenario.start, &scenario.target),
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
        begin_and_certify_geometry_winner(
            assist, scenario.start, &scenario.target),
        "unchanged-identity fixture begin failed");
    const interaction::PickAssistDiagnostics frozen_before =
        assist.diagnostics();

    const interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    require(
        output.override_steering && !is_zero(output.left_stick) &&
            is_zero(output.right_stick) && !output.force_strafe &&
            !output.stationary_constraint &&
            output.preview_requests.empty() &&
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
        begin_and_certify_geometry_winner(
            assist, scenario.start, &scenario.target),
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
        output.preview_requests.empty() && !output.stationary_constraint &&
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
        begin_and_certify_geometry_winner(
            assist, scenario.start, &scenario.target),
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
        output.preview_requests.empty() && !output.submit_interact &&
            !assist.take_submission(74U).has_value(),
        "inclusive arrival latch previewed or submitted on the latch tick");
}

interaction::PickAssistDiagnostics latch_frozen_slot_settling(
    interaction::ControllerPickAssist& assist,
    FrozenSlotScenario& scenario,
    const interaction::PickAssistConfig& config) {
    require(
        begin_and_certify_geometry_winner(
            assist, scenario.start, &scenario.target),
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
            latch_output.preview_requests.empty() &&
            !latch_output.submit_interact,
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

void test_frozen_slot_settling_records_consecutive_travel_without_failure() {
    const interaction::PickAssistConfig config{};
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist(config);
    const interaction::PickAssistDiagnostics latched =
        latch_frozen_slot_settling(assist, scenario, config);

    assist.observe(scenario.observation);
    const interaction::PickAssistDiagnostics settling = assist.diagnostics();
    require(
        settling.state == interaction::PickAssistState::Settling &&
            settling.reason == interaction::PickAssistReason::None &&
            settling.settle_ticks == 1U,
        "frozen Settling travel fixture did not advance its stable tick");
    require_frozen_slot_provenance_unchanged(
        latched,
        settling,
        "stable frozen Settling tick changed frozen provenance");

    const vec3 previous_position =
        scenario.observation.displayed_root.position;
    scenario.observation.displayed_root.position.x -= 0.25F;
    const float next_segment_m = static_cast<float>(std::hypot(
        static_cast<double>(scenario.observation.displayed_root.position.x) -
            static_cast<double>(previous_position.x),
        static_cast<double>(scenario.observation.displayed_root.position.z) -
            static_cast<double>(previous_position.z)));
    const float expected_travel_m =
        settling.assisted_travel_m + next_segment_m;
    require(
        settling.assisted_travel_m < 1.00002F &&
            expected_travel_m > 1.00002F,
        "frozen Settling travel fixture did not cross 1.00002 m");

    const interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    const interaction::PickAssistDiagnostics continued =
        assist.diagnostics();
    require(
        continued.state == interaction::PickAssistState::Settling &&
            continued.reason == interaction::PickAssistReason::None &&
            continued.settle_ticks == 0U,
        "finite frozen Settling detour failed or retained a nonconsecutive tick");
    require(
        output.override_steering && output.force_strafe &&
            output.stationary_constraint && is_zero(output.left_stick) &&
            is_zero(output.right_stick) &&
            output.preview_requests.empty() &&
            !output.submit_interact,
        "finite frozen Settling detour did not retain active braking");
    require(
        assist.active() && assist.owns_manual_interact() &&
            !assist.take_submission(503U).has_value(),
        "finite frozen Settling detour lost ownership or submitted");
    require_frozen_slot_provenance_unchanged(
        settling,
        continued,
        "finite frozen Settling detour changed frozen provenance");
    require(
        same_float_bits_exact(
            continued.assisted_travel_m, expected_travel_m),
        "frozen Settling did not retain cumulative planar telemetry");
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
            is_zero(output.right_stick) &&
            output.preview_requests.empty() &&
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
        bool requests_preview,
        const char* message) {
        require(
            output.override_steering && output.force_strafe &&
                output.stationary_constraint &&
                is_zero(output.left_stick) &&
                is_zero(output.right_stick) &&
                output.preview_requests.empty() == !requests_preview &&
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
        output.preview_requests.size() == 1U &&
            output.preview_requests.front().slot_id == 9U &&
            same_float_bits_exact(
                output.preview_requests.front().root.world_x,
                expected_preview_root.world_x) &&
            same_float_bits_exact(
                output.preview_requests.front().root.world_z,
                expected_preview_root.world_z) &&
            same_float_bits_exact(
                output.preview_requests.front().root.world_yaw_radians,
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
                    output.preview_requests.empty() &&
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
    return output.preview_requests.front().root;
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
    set_frozen_preview_result(
        scenario.observation,
        frozen_root,
        certified_frozen_preview(frozen_root));

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
            output.preview_requests.empty() &&
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

void test_frozen_final_preview_missing_batch_rerequests() {
    FrozenSlotScenario scenario;
    interaction::ControllerPickAssist assist;
    const interaction::PickEntryRoot frozen_root =
        enter_frozen_final_preview(assist, scenario);
    const interaction::PickAssistDiagnostics before = assist.diagnostics();

    scenario.observation.preview_results.clear();
    const interaction::PickAssistOutput output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::FinalPreview &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None,
        "missing frozen preview batch changed FinalPreview outcome");
    require_stationary_preview_request(
        output,
        frozen_root,
        "missing frozen preview batch did not re-request the frozen root");
    require(
        assist.active() && assist.owns_manual_interact() &&
            !assist.take_submission(61U).has_value(),
        "missing frozen preview batch lost ownership or submitted");
    require_frozen_slot_provenance_unchanged(
        before,
        assist.diagnostics(),
        "missing frozen preview batch changed frozen provenance");
    require_final_preview_diagnostics_cleared(assist.diagnostics());

    set_frozen_preview_result(
        scenario.observation, frozen_root, std::nullopt);
    const interaction::PickAssistOutput null_result_output =
        assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::FinalPreview &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None,
        "exact singleton null preview result changed FinalPreview outcome");
    require_stationary_preview_request(
        null_result_output,
        frozen_root,
        "exact singleton null preview result did not re-request");
    require_final_preview_diagnostics_cleared(assist.diagnostics());
}

void test_frozen_final_preview_rejects_nonexact_result_batch() {
    enum class Defect : uint8_t {
        DuplicateResult,
        WrongSlot,
        WrongRequestRoot,
    };
    const std::array<Defect, 3> defects{
        Defect::DuplicateResult,
        Defect::WrongSlot,
        Defect::WrongRequestRoot,
    };

    for (Defect defect : defects) {
        FrozenSlotScenario scenario;
        interaction::ControllerPickAssist assist;
        const interaction::PickEntryRoot frozen_root =
            enter_frozen_final_preview(assist, scenario);
        scenario.observation.snapshot_fingerprint = 1801U;
        scenario.observation.preview_snapshot_fingerprint = 1801U;
        set_frozen_preview_result(
            scenario.observation,
            frozen_root,
            certified_frozen_preview(frozen_root));

        switch (defect) {
        case Defect::DuplicateResult:
            scenario.observation.preview_results.push_back(
                scenario.observation.preview_results.front());
            break;
        case Defect::WrongSlot:
            scenario.observation.preview_results.front().request.slot_id =
                109U;
            break;
        case Defect::WrongRequestRoot:
            scenario.observation.preview_results.front()
                .request.root.world_x = std::nextafter(
                    frozen_root.world_x,
                    std::numeric_limits<float>::infinity());
            break;
        }

        const interaction::PickAssistOutput output =
            assist.observe(scenario.observation);
        require(
            assist.diagnostics().state ==
                    interaction::PickAssistState::Failed &&
                assist.diagnostics().reason ==
                    interaction::PickAssistReason::FinalPreviewRejected,
            "nonexact FinalPreview result batch was consumed");
        require_zero_pick_assist_output(
            output,
            "nonexact FinalPreview result batch emitted output");
        require(
            !assist.take_submission(1801U).has_value(),
            "nonexact FinalPreview result batch produced a submission");
    }
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
    set_frozen_preview_result(
        scenario.observation, frozen_root, poor_match);
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
    set_frozen_preview_result(
        scenario.observation, frozen_root, certified);
    output = assist.observe(scenario.observation);
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::ReadyToSubmit &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            output.override_steering && output.force_strafe &&
            output.stationary_constraint &&
            is_zero(output.left_stick) && is_zero(output.right_stick) &&
            output.preview_requests.empty() &&
            output.submit_interact,
        "certified singleton preview did not recover PoorMatch to ReadyToSubmit");
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
        set_frozen_preview_result(
            scenario.observation, frozen_root, preview);
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
    set_frozen_preview_result(
        scenario.observation, frozen_root, certified);

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
            output.preview_requests.empty() &&
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
            output.preview_requests.empty() &&
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

    scenario.observation.preview_results.clear();
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
    set_frozen_preview_result(
        scenario.observation, frozen_root, poor_match);
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

    scenario.observation.preview_results.clear();
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

    scenario.observation.preview_results.clear();
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

    set_frozen_preview_result(
        scenario.observation,
        frozen_root,
        certified_frozen_preview(frozen_root));
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
            begin_and_certify_geometry_winner(
                assist, scenario.start, &scenario.target) &&
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
                output.preview_requests.empty() &&
                !output.submit_interact &&
                !assist.take_submission(75U).has_value(),
            "first representable position overshoot latched braking or submitted");
    }

    {
        FrozenSlotScenario scenario;
        interaction::ControllerPickAssist assist(config);
        require(
            begin_and_certify_geometry_winner(
                assist, scenario.start, &scenario.target) &&
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
                output.preview_requests.empty() &&
                !output.submit_interact &&
                !assist.take_submission(76U).has_value(),
            "first representable simulation-speed overshoot latched braking or submitted");
    }

    {
        FrozenSlotScenario scenario;
        interaction::ControllerPickAssist assist(config);
        require(
            begin_and_certify_geometry_winner(
                assist, scenario.start, &scenario.target) &&
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
                output.preview_requests.empty() &&
                !output.submit_interact &&
                !assist.take_submission(77U).has_value(),
            "first representable derived-yaw overshoot latched braking or submitted");
    }
}

void test_slot_approach_records_cumulative_travel_without_failure() {
    FrozenSlotScenario scenario;
    scenario.target.object_world.position.x = 0.80F;
    scenario.start.target_snapshot = scenario.target;
    scenario.start.root_world = {
        vec3(0.0F, 0.0F, 0.50F), quat()};
    scenario.observation.displayed_root = scenario.start.root_world;
    scenario.observation.target = &scenario.target;

    interaction::ControllerPickAssist assist;
    require(
        begin_and_certify_geometry_winner(
            assist, scenario.start, &scenario.target),
        "cumulative-travel fixture begin failed");
    const interaction::PickAssistDiagnostics frozen_before =
        assist.diagnostics();
    require(
        frozen_before.selected_slot_id == 9U &&
            frozen_before.slot_selection.selected_index.has_value(),
        "cumulative-travel fixture did not select slot ID 9");
    const interaction::MappedPickSlot& frozen_slot =
        frozen_before.slot_selection.ordered[
            *frozen_before.slot_selection.selected_index];
    require(
        same_float_bits_exact(
            frozen_slot.root_world.position.x, 0.80F) &&
            same_float_bits_exact(
                frozen_slot.root_world.position.z, 0.50F),
        "cumulative-travel fixture did not freeze slot 9 at (0.8, 0.5)");

    scenario.observation.displayed_root.position.x = 0.50F;
    assist.observe(scenario.observation);
    scenario.observation.displayed_root.position.x = 0.0F;
    assist.observe(scenario.observation);
    scenario.observation.displayed_root.position.x = 0.50F;
    const interaction::PickAssistOutput output =
        assist.observe(scenario.observation);

    require(same_float_bits_exact(
                assist.diagnostics().assisted_travel_m, 1.50F),
            "approach did not retain cumulative planar telemetry");
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::SlotApproach &&
            assist.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            assist.active() && output.override_steering &&
            !output.submit_interact,
        "finite admitted detour was treated as a terminal travel envelope");
    require_frozen_slot_provenance_unchanged(
        frozen_before,
        assist.diagnostics(),
        "finite admitted detour changed frozen provenance");
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
        begin_and_certify_geometry_winner(
            assist, scenario.start, &scenario.target),
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
        begin_and_certify_geometry_winner(
            assist, scenario.start, &scenario.target),
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
        begin_and_certify_geometry_winner(
            assist, make_frozen_slot_start(valid), &valid),
        "failed assist could not immediately begin a valid attempt");
    require(
        assist.diagnostics().state ==
                interaction::PickAssistState::SlotApproach &&
            assist.diagnostics().reason == interaction::PickAssistReason::None &&
            assist.diagnostics().selected_slot_id == 9U,
        "valid retry did not replace the failed attempt diagnostics");
}

void test_diagnostic_names_are_stable() {
    const std::array<const char*, 8> states{
        "Idle", "SlotApproach", "Settling", "FinalPreview",
        "ReadyToSubmit", "Submitted", "Failed", "SlotSelectionPreview"};
    for (size_t i = 0; i < states.size(); ++i) {
        require(
            std::string(interaction::pick_assist_state_name(
                static_cast<interaction::PickAssistState>(i))) == states[i],
            "state diagnostic name changed");
    }
    const std::array<const char*, 16> reasons{
        "None", "Cancelled", "TargetUnavailable", "TargetChanged",
        "SlotChanged", "RuntimeChanged", "NoAuthoredSlot",
        "InvalidGeometry", "OutsideTravelEnvelope", "TableBlocked",
        "ObstacleBlocked", "AllSlotsBlocked", "ArrivalDeadline",
        "PoorMatch", "FinalPreviewRejected", "SelectionPreviewRejected"};
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
        begin_and_certify_geometry_winner(
            assist, scenario.start, &scenario.target),
        "frozen-slot cancel fixture begin failed");
    const interaction::PickAssistDiagnostics frozen_before =
        assist.diagnostics();

    scenario.observation.displayed_root.position.x = -0.60F;
    const interaction::PickAssistOutput steering_output =
        assist.observe(scenario.observation);
    require(
        steering_output.override_steering &&
            !is_zero(steering_output.left_stick) &&
            steering_output.preview_requests.empty() &&
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
        begin_and_certify_geometry_winner(
            assist, restarted.start, &restarted.target) &&
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
        test_begin_defers_freeze_and_requests_every_ranked_entry();
        test_selection_freezes_first_ranked_ready_entry_not_lowest_cost();
        test_selection_incomplete_batches_retry_atomically_then_deadline();
        test_selection_mixed_poor_match_retries_until_poor_match_deadline();
        test_selection_complete_hard_batch_rejects();
        test_selection_malformed_batches_fail_closed();
        test_selection_latches_poor_match_before_ready_deadline();
        test_selection_target_affordance_and_slot_mutations_fail_closed();
        test_selection_revalidates_each_candidate_route_before_freeze();
        test_selection_epochs_do_not_consume_frozen_arrival_budget();
        test_begin_selects_and_freezes_one_authored_slot();
        test_begin_rejects_duplicate_affordance_ids_immediately();
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
        test_frozen_slot_settling_records_consecutive_travel_without_failure();
        test_frozen_slot_settling_keeps_stationary_braking();
        test_frozen_slot_settling_requests_exact_frozen_preview_root();
        test_frozen_final_preview_missing_batch_rerequests();
        test_frozen_final_preview_rejects_nonexact_result_batch();
        test_frozen_final_preview_poor_match_rerequests_then_recovers();
        test_frozen_final_preview_hard_rejections_are_terminal();
        test_frozen_final_preview_certifies_and_submits_exactly_once();
        test_frozen_ready_to_submit_runs_shared_preflight();
        test_frozen_final_preview_missing_preview_hits_arrival_deadline();
        test_frozen_final_preview_poor_match_deadline_memory_survives_missing();
        test_frozen_final_preview_deadline_precedes_certification();
        test_slot_approach_rejects_adjacent_arrival_overshoots();
        test_slot_approach_records_cumulative_travel_without_failure();
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
