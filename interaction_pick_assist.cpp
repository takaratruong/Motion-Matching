#include "interaction_pick_assist.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace interaction {
namespace {

constexpr float kTravelEnvelopeToleranceM = 2.0e-5F;

uint32_t float_bits(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool is_finite(float value) {
    return (float_bits(value) & 0x7f800000U) != 0x7f800000U;
}

bool same_finite_float_bits(float left, float right) {
    return is_finite(left) && is_finite(right) &&
        float_bits(left) == float_bits(right);
}

bool is_finite(vec3 value) {
    return is_finite(value.x) && is_finite(value.y) && is_finite(value.z);
}

bool is_finite(quat value) {
    return is_finite(value.w) && is_finite(value.x) &&
        is_finite(value.y) && is_finite(value.z);
}

bool is_finite(Transform value) {
    return is_finite(value.position) && is_finite(value.rotation);
}

const GraspAffordance* find_unique_affordance(
    const InteractionTarget& target,
    uint32_t affordance_id) {
    const GraspAffordance* selected = nullptr;
    for (const GraspAffordance& affordance : target.affordances) {
        if (affordance.id != affordance_id) continue;
        if (selected != nullptr) return nullptr;
        selected = &affordance;
    }
    return selected;
}

bool same_ordered_authored_slots(
    const GraspAffordance& frozen,
    const GraspAffordance& live) {
    if (frozen.interaction_slots.size() !=
        live.interaction_slots.size()) {
        return false;
    }
    for (size_t index = 0U;
         index < frozen.interaction_slots.size();
         ++index) {
        const GraspInteractionSlot& left =
            frozen.interaction_slots[index];
        const GraspInteractionSlot& right =
            live.interaction_slots[index];
        if (left.id != right.id ||
            !same_finite_float_bits(
                left.root_x_object_m, right.root_x_object_m) ||
            !same_finite_float_bits(
                left.root_z_object_m, right.root_z_object_m) ||
            !same_finite_float_bits(
                left.root_yaw_object_radians,
                right.root_yaw_object_radians)) {
            return false;
        }
    }
    return true;
}

bool is_finite(PickEntryRoot value) {
    return is_finite(value.world_x) && is_finite(value.world_z) &&
        is_finite(value.world_yaw_radians);
}

bool is_finite(const PickEntrySlot& slot) {
    return is_finite(slot.waypoint) &&
        is_finite(slot.prospective_root) && is_finite(slot.hand_score);
}

bool is_finite(const PickAssistStart& start) {
    if (!is_finite(start.object_world) || !is_finite(start.root_world) ||
        !is_finite(start.reach_waypoint) ||
        !is_finite(start.slots.clearance_chord_m) ||
        !is_finite(start.slots.preserved_standoff_m)) {
        return false;
    }
    for (const PickEntrySlot& slot : start.slots.ordered) {
        if (!is_finite(slot)) return false;
    }
    return true;
}

bool is_positive_finite(float value) {
    return is_finite(value) && value > 0.0F;
}

void validate_config(const PickAssistConfig& config) {
    const bool valid_scalars =
        is_positive_finite(config.maximum_assisted_path_m) &&
        config.maximum_assisted_path_m <= 1.00F &&
        is_positive_finite(config.reach_entry_distance_m) &&
        is_positive_finite(config.reach_entry_tolerance_m) &&
        is_positive_finite(config.maximum_settle_position_error_m) &&
        is_positive_finite(config.maximum_settle_displayed_speed_mps) &&
        is_positive_finite(config.arrival.slow_radius_m) &&
        is_positive_finite(config.arrival.latch_position_error_m) &&
        is_positive_finite(config.arrival.latch_simulation_speed_mps) &&
        is_positive_finite(config.arrival.maximum_yaw_error_radians) &&
        is_positive_finite(config.arrival.minimum_standoff_m) &&
        is_positive_finite(config.arrival.maximum_standoff_m) &&
        config.arrival.maximum_standoff_m >
            config.arrival.minimum_standoff_m;
    const bool valid_ticks = config.required_settle_ticks != 0U &&
        config.maximum_preview_ticks != 0U &&
        config.maximum_arrival_ticks != 0U;
    if (!valid_scalars || !valid_ticks) {
        throw std::invalid_argument("invalid pick-assist config");
    }
}

float planar_distance(vec3 left, vec3 right) {
    return static_cast<float>(std::hypot(
        static_cast<double>(left.x) - static_cast<double>(right.x),
        static_cast<double>(left.z) - static_cast<double>(right.z)));
}

bool same_entry_root(PickEntryRoot left, PickEntryRoot right) {
    return is_finite(left) && is_finite(right) &&
        float_bits(left.world_x) == float_bits(right.world_x) &&
        float_bits(left.world_z) == float_bits(right.world_z) &&
        float_bits(left.world_yaw_radians) ==
            float_bits(right.world_yaw_radians);
}

bool same_transform(Transform left, Transform right) {
    return is_finite(left) && is_finite(right) &&
        left.position.x == right.position.x &&
        left.position.y == right.position.y &&
        left.position.z == right.position.z &&
        left.rotation.w == right.rotation.w &&
        left.rotation.x == right.rotation.x &&
        left.rotation.y == right.rotation.y &&
        left.rotation.z == right.rotation.z;
}

bool observation_metrics_are_finite(
    const PickAssistObservation& observation) {
    return is_finite(observation.displayed_root) &&
        is_finite(observation.simulation_velocity) &&
        is_finite(observation.displayed_planar_speed_mps) &&
        observation.displayed_planar_speed_mps >= 0.0F &&
        is_finite(observation.camera_azimuth);
}

bool preview_roots_are_finite(
    const std::array<PickEntryPreview, 2>& previews) {
    for (const PickEntryPreview& preview : previews) {
        if (!is_finite(preview.prospective_root)) return false;
    }
    return true;
}

void capture_final_preview_diagnostics(
    PickAssistDiagnostics& diagnostics,
    const PickAssistStart& start,
    const PickAssistObservation& observation) {
    if (diagnostics.state != PickAssistState::FinalPreview ||
        !observation.previews.has_value() ||
        diagnostics.selected_slot < 0 ||
        diagnostics.selected_slot >=
            static_cast<int>(observation.previews->size())) {
        return;
    }

    const size_t selected = static_cast<size_t>(diagnostics.selected_slot);
    const PickEntryPreview& preview = observation.previews->at(selected);
    PickAssistFinalPreviewDiagnostics captured{};
    captured.available = true;
    captured.all_preview_roots_finite =
        preview_roots_are_finite(*observation.previews);
    captured.fingerprint_equal =
        observation.preview_snapshot_fingerprint ==
        observation.snapshot_fingerprint;
    captured.path_feasible = preview.path_feasible;
    captured.path_reason = preview.path_reason;
    captured.match_ready = preview.match_ready;
    captured.match_reason = preview.match_reason;
    captured.prospective_root_equal = same_entry_root(
        preview.prospective_root,
        start.slots.ordered[selected].prospective_root);
    captured.feasible_entry_frame = preview.feasible_entry_frame;
    captured.contact_frame = preview.contact_frame;
    captured.total_cost = preview.total_cost;
    diagnostics.final_preview = captured;
}

void capture_frozen_final_preview_diagnostics(
    PickAssistDiagnostics& diagnostics,
    PickEntryRoot frozen_preview_root,
    const PickAssistObservation& observation) {
    if (diagnostics.state != PickAssistState::FinalPreview ||
        !observation.preview.has_value()) {
        return;
    }

    const PickEntryPreview& preview = *observation.preview;
    PickAssistFinalPreviewDiagnostics captured{};
    captured.available = true;
    captured.all_preview_roots_finite =
        is_finite(preview.prospective_root);
    captured.fingerprint_equal =
        observation.preview_snapshot_fingerprint ==
        observation.snapshot_fingerprint;
    captured.path_feasible = preview.path_feasible;
    captured.path_reason = preview.path_reason;
    captured.match_ready = preview.match_ready;
    captured.match_reason = preview.match_reason;
    captured.prospective_root_equal = same_entry_root(
        preview.prospective_root, frozen_preview_root);
    captured.feasible_entry_frame = preview.feasible_entry_frame;
    captured.contact_frame = preview.contact_frame;
    captured.total_cost = preview.total_cost;
    diagnostics.final_preview = captured;
}

PickAssistOutput fail_output(
    PickAssistDiagnostics& diagnostics,
    PickAssistReason reason) {
    if (!is_finite(diagnostics.root_error_m)) {
        diagnostics.root_error_m = 0.0F;
    }
    if (!is_finite(diagnostics.yaw_error_radians)) {
        diagnostics.yaw_error_radians = 0.0F;
    }
    if (!is_finite(diagnostics.speed_mps)) {
        diagnostics.speed_mps = 0.0F;
    }
    diagnostics.state = PickAssistState::Failed;
    diagnostics.reason = reason;
    return {};
}

bool target_metadata_matches(
    const PickAssistStart& start,
    const InteractionTarget& target) {
    if (target.handle.generation != start.target.generation ||
        target.state != ObjectState::Free ||
        !same_transform(target.object_world, start.object_world)) {
        return false;
    }
    for (const GraspAffordance& affordance : target.affordances) {
        if (affordance.id == start.affordance_id) {
            return affordance.hand == start.hand;
        }
    }
    return false;
}

vec3 ordinary_navigation_stick(
    vec3 target,
    vec3 current,
    float camera_azimuth) {
    vec3 direction = target - current;
    direction.y = 0.0F;
    const float distance = static_cast<float>(std::hypot(
        static_cast<double>(direction.x),
        static_cast<double>(direction.z)));
    if (distance <= 1.0e-5F) return {};
    direction = direction / distance;
    const quat camera_basis = quat_from_angle_axis(
        camera_azimuth, vec3(0.0F, 1.0F, 0.0F));
    return quat_inv_mul_vec3(camera_basis, direction);
}

float yaw_radians(quat rotation) {
    const vec3 forward = quat_mul_vec3(
        rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(forward.x, forward.z);
}

float yaw_error(float left, float right) {
    return std::abs(std::atan2(
        std::sin(left - right), std::cos(left - right)));
}

float planar_speed(vec3 velocity) {
    return static_cast<float>(std::hypot(
        static_cast<double>(velocity.x),
        static_cast<double>(velocity.z)));
}

PickAssistOutput braking_output(
    bool needs_preview = false,
    std::optional<PickEntryRoot> preview_root = {}) {
    PickAssistOutput output{};
    output.override_steering = true;
    output.force_strafe = true;
    output.stationary_constraint = true;
    output.needs_preview = needs_preview;
    output.preview_root = preview_root;
    return output;
}

PickAssistReason frozen_arrival_deadline_reason(
    bool poor_match_observed) {
    return poor_match_observed
        ? PickAssistReason::PoorMatch
        : PickAssistReason::ArrivalDeadline;
}

}  // namespace

ControllerPickAssist::ControllerPickAssist(PickAssistConfig config)
    : config_(config) {
    validate_config(config_);
}

void ControllerPickAssist::cancel() {
    frozen_slot_attempt_ = false;
    frozen_preview_root_.reset();
    poor_match_observed_ = false;
    if (diagnostics_.state == PickAssistState::Submitted) return;
    start_ = {};
    frozen_slot_ = {};
    previous_observed_root_ = {};
    entry_point_ = {};
    preview_ticks_ = 0U;
    arrival_ticks_ = 0U;
    diagnostics_ = {};
    diagnostics_.reason = PickAssistReason::Cancelled;
}

bool ControllerPickAssist::begin(const PickAssistStart& start) {
    if (active()) return false;

    frozen_slot_attempt_ = false;
    frozen_preview_root_.reset();
    poor_match_observed_ = false;
    previous_observed_root_ = {};
    start_ = start;
    preview_ticks_ = 0U;
    arrival_ticks_ = 0U;
    diagnostics_ = {};
    diagnostics_.target = start.target;
    diagnostics_.affordance_id = start.affordance_id;

    if (!is_finite(start)) {
        diagnostics_.state = PickAssistState::Failed;
        diagnostics_.reason = PickAssistReason::OutsideTravelEnvelope;
        return false;
    }

    vec3 forward = quat_mul_vec3(
        start.reach_waypoint.rotation,
        vec3(0.0F, 0.0F, 1.0F));
    forward.y = 0.0F;
    if (!is_finite(forward)) {
        diagnostics_.state = PickAssistState::Failed;
        diagnostics_.reason = PickAssistReason::OutsideTravelEnvelope;
        return false;
    }
    const float forward_length = static_cast<float>(std::hypot(
        static_cast<double>(forward.x),
        static_cast<double>(forward.z)));
    if (!is_finite(forward_length) || forward_length <= 1.0e-5F) {
        diagnostics_.state = PickAssistState::Failed;
        diagnostics_.reason = PickAssistReason::OutsideTravelEnvelope;
        return false;
    }
    forward = forward / forward_length;
    entry_point_ = start.reach_waypoint.position -
        config_.reach_entry_distance_m * forward;
    if (!is_finite(entry_point_)) {
        diagnostics_.state = PickAssistState::Failed;
        diagnostics_.reason = PickAssistReason::OutsideTravelEnvelope;
        return false;
    }

    float minimum_permitted = 0.0F;
    bool have_permitted = false;
    for (size_t i = 0; i < start.slots.ordered.size(); ++i) {
        const float root_to_entry =
            planar_distance(start.root_world.position, entry_point_);
        const float entry_to_slot = planar_distance(
            entry_point_, start.slots.ordered[i].waypoint.position);
        if (!is_finite(root_to_entry) || !is_finite(entry_to_slot)) {
            diagnostics_.state = PickAssistState::Failed;
            diagnostics_.reason = PickAssistReason::OutsideTravelEnvelope;
            return false;
        }
        const float route = root_to_entry + entry_to_slot;
        if (!is_finite(route)) {
            diagnostics_.state = PickAssistState::Failed;
            diagnostics_.reason = PickAssistReason::OutsideTravelEnvelope;
            return false;
        }
        const bool permitted = route <=
            config_.maximum_assisted_path_m + kTravelEnvelopeToleranceM;
        diagnostics_.slot_route_lengths_m[i] = route;
        diagnostics_.slot_permitted[i] = permitted;
        if (permitted && (!have_permitted || route < minimum_permitted)) {
            minimum_permitted = route;
            have_permitted = true;
        }
    }
    if (!have_permitted) {
        diagnostics_.state = PickAssistState::Failed;
        diagnostics_.reason = PickAssistReason::OutsideTravelEnvelope;
        return false;
    }

    diagnostics_.route_length_m = minimum_permitted;
    diagnostics_.state = PickAssistState::CoarseApproach;
    return true;
}

bool ControllerPickAssist::begin(
    const PickAssistStart& start,
    const InteractionTarget* post_step_target) {
    if (active()) return false;

    start_ = {};
    frozen_slot_ = {};
    frozen_preview_root_.reset();
    frozen_slot_attempt_ = false;
    poor_match_observed_ = false;
    previous_observed_root_ = {};
    entry_point_ = {};
    preview_ticks_ = 0U;
    arrival_ticks_ = 0U;
    diagnostics_ = {};
    diagnostics_.target = start.target_snapshot.handle;
    diagnostics_.affordance_id = start.affordance_id;

    const auto fail_begin = [this](PickAssistReason reason) {
        start_ = {};
        frozen_slot_ = {};
        frozen_preview_root_.reset();
        frozen_slot_attempt_ = false;
        poor_match_observed_ = false;
        previous_observed_root_ = {};
        entry_point_ = {};
        preview_ticks_ = 0U;
        arrival_ticks_ = 0U;
        diagnostics_.state = PickAssistState::Failed;
        diagnostics_.reason = reason;
        return false;
    };

    if (post_step_target == nullptr ||
        post_step_target->handle.id != start.target_snapshot.handle.id) {
        return fail_begin(PickAssistReason::TargetUnavailable);
    }
    if (start.target_snapshot.state != ObjectState::Free ||
        start.target_snapshot.owner_request != 0U ||
        !same_interaction_target_snapshot(
            start.target_snapshot, *post_step_target)) {
        return fail_begin(PickAssistReason::TargetChanged);
    }

    const GraspAffordance* selected_affordance = nullptr;
    for (const GraspAffordance& affordance :
         start.target_snapshot.affordances) {
        if (affordance.id == start.affordance_id) {
            selected_affordance = &affordance;
            break;
        }
    }
    if (selected_affordance == nullptr) {
        return fail_begin(PickAssistReason::TargetChanged);
    }

    PickSlotConfig slot_config{};
    slot_config.maximum_direct_travel_m =
        config_.maximum_assisted_path_m;
    diagnostics_.slot_selection = select_pick_slot(
        start.root_world,
        start.target_snapshot,
        *selected_affordance,
        start.obstacles,
        slot_config);
    if (!diagnostics_.slot_selection.selected_index.has_value()) {
        return fail_begin(pick_assist_reason_from_slot_reason(
            diagnostics_.slot_selection.reason));
    }

    const size_t selected_index =
        *diagnostics_.slot_selection.selected_index;
    frozen_slot_ = diagnostics_.slot_selection.ordered[selected_index];
    const PickEntryRoot frozen_preview_root{
        frozen_slot_.root_world.position.x,
        frozen_slot_.root_world.position.z,
        yaw_radians(frozen_slot_.root_world.rotation),
    };
    if (!is_finite(frozen_preview_root)) {
        return fail_begin(PickAssistReason::OutsideTravelEnvelope);
    }
    frozen_preview_root_ = frozen_preview_root;
    start_ = start;
    start_.target = start.target_snapshot.handle;
    start_.hand = selected_affordance->hand;
    start_.object_world = start.target_snapshot.object_world;
    previous_observed_root_ = start.root_world;

    diagnostics_.selected_slot = static_cast<int>(selected_index);
    diagnostics_.selected_slot_id = frozen_slot_.id;
    diagnostics_.route_length_m = frozen_slot_.route_length_m;
    diagnostics_.object_origin_distance_m =
        frozen_slot_.object_origin_distance_m;
    diagnostics_.object_bounds_center_distance_m =
        frozen_slot_.object_bounds_center_distance_m;
    diagnostics_.state = PickAssistState::SlotApproach;
    frozen_slot_attempt_ = true;
    return true;
}

PickAssistOutput ControllerPickAssist::observe(
    const PickAssistObservation& observation) {
    PickAssistOutput output{};
    const bool frozen_slot_phase = frozen_slot_attempt_ &&
        (diagnostics_.state == PickAssistState::SlotApproach ||
            diagnostics_.state == PickAssistState::Settling ||
            diagnostics_.state == PickAssistState::FinalPreview);
    const bool frozen_slot_settling = frozen_slot_attempt_ &&
        diagnostics_.state == PickAssistState::Settling;
    const bool frozen_slot_final_preview = frozen_slot_attempt_ &&
        diagnostics_.state == PickAssistState::FinalPreview;
    if (frozen_slot_phase) {
        if (observation.runtime_state != RuntimeState::Locomotion) {
            return fail_output(
                diagnostics_, PickAssistReason::RuntimeChanged);
        }
        if (observation.target == nullptr ||
            observation.target->handle.id != start_.target.id) {
            return fail_output(
                diagnostics_, PickAssistReason::TargetUnavailable);
        }
        if (observation.target->handle.generation !=
                start_.target.generation ||
            observation.target->state != ObjectState::Free ||
            observation.target->owner_request != 0U) {
            return fail_output(
                diagnostics_, PickAssistReason::TargetChanged);
        }
        const GraspAffordance* frozen_affordance =
            find_unique_affordance(
                start_.target_snapshot, start_.affordance_id);
        const GraspAffordance* live_affordance =
            find_unique_affordance(
                *observation.target, start_.affordance_id);
        if (frozen_affordance == nullptr ||
            live_affordance == nullptr) {
            return fail_output(
                diagnostics_, PickAssistReason::TargetChanged);
        }
        if (!same_ordered_authored_slots(
                *frozen_affordance, *live_affordance)) {
            return fail_output(
                diagnostics_, PickAssistReason::SlotChanged);
        }
        if (!same_interaction_target_snapshot(
                start_.target_snapshot, *observation.target)) {
            return fail_output(
                diagnostics_, PickAssistReason::TargetChanged);
        }
        if (!observation_metrics_are_finite(observation)) {
            return fail_output(
                diagnostics_, PickAssistReason::OutsideTravelEnvelope);
        }
        const float travel_segment_m = planar_distance(
            previous_observed_root_.position,
            observation.displayed_root.position);
        diagnostics_.assisted_travel_m += travel_segment_m;
        previous_observed_root_ = observation.displayed_root;
        if (!is_finite(travel_segment_m) ||
            !is_finite(diagnostics_.assisted_travel_m) ||
            diagnostics_.assisted_travel_m >
                config_.maximum_assisted_path_m +
                    kTravelEnvelopeToleranceM) {
            previous_observed_root_ = {};
            return fail_output(
                diagnostics_, PickAssistReason::OutsideTravelEnvelope);
        }
        PickSlotConfig slot_config{};
        slot_config.maximum_direct_travel_m =
            config_.maximum_assisted_path_m;
        const PickSlotReason slot_reason = revalidate_frozen_pick_slot(
            observation.displayed_root,
            frozen_slot_.root_world,
            start_.target_snapshot,
            start_.obstacles,
            slot_config);
        if (slot_reason != PickSlotReason::None) {
            return fail_output(
                diagnostics_,
                pick_assist_reason_from_slot_reason(slot_reason));
        }
        diagnostics_.root_error_m = planar_distance(
            observation.displayed_root.position,
            frozen_slot_.root_world.position);
        diagnostics_.speed_mps = planar_speed(
            observation.simulation_velocity);
        diagnostics_.yaw_error_radians = yaw_error(
            yaw_radians(observation.displayed_root.rotation),
            yaw_radians(frozen_slot_.root_world.rotation));
        if (!is_finite(diagnostics_.root_error_m) ||
            !is_finite(diagnostics_.speed_mps) ||
            !is_finite(diagnostics_.yaw_error_radians)) {
            return fail_output(
                diagnostics_, PickAssistReason::OutsideTravelEnvelope);
        }
        if (frozen_slot_settling) {
            ++arrival_ticks_;
            const bool stable =
                diagnostics_.root_error_m <=
                    config_.maximum_settle_position_error_m &&
                observation.displayed_planar_speed_mps <=
                    config_.maximum_settle_displayed_speed_mps &&
                diagnostics_.yaw_error_radians <=
                    config_.arrival.maximum_yaw_error_radians;
            if (stable) {
                ++diagnostics_.settle_ticks;
            } else {
                diagnostics_.settle_ticks = 0U;
            }
            if (arrival_ticks_ >= config_.maximum_arrival_ticks) {
                return fail_output(
                    diagnostics_,
                    frozen_arrival_deadline_reason(
                        poor_match_observed_));
            }
            if (diagnostics_.settle_ticks ==
                config_.required_settle_ticks) {
                diagnostics_.state = PickAssistState::FinalPreview;
                return braking_output(true, frozen_preview_root_);
            }
            return braking_output();
        }
        if (frozen_slot_final_preview) {
            ++arrival_ticks_;
            if (arrival_ticks_ >= config_.maximum_arrival_ticks) {
                return fail_output(
                    diagnostics_,
                    frozen_arrival_deadline_reason(
                        poor_match_observed_));
            }
            if (!observation.preview.has_value()) {
                return braking_output(true, frozen_preview_root_);
            }
            capture_frozen_final_preview_diagnostics(
                diagnostics_, *frozen_preview_root_, observation);
            const PickAssistFinalPreviewDiagnostics& preview =
                diagnostics_.final_preview;
            const bool retryable_poor_match =
                preview.all_preview_roots_finite &&
                preview.fingerprint_equal &&
                preview.prospective_root_equal &&
                preview.path_feasible &&
                preview.path_reason == Reason::None &&
                !preview.match_ready &&
                preview.match_reason == Reason::PoorMatch;
            if (retryable_poor_match) {
                poor_match_observed_ = true;
                return braking_output(true, frozen_preview_root_);
            }
            const bool certified =
                preview.all_preview_roots_finite &&
                preview.fingerprint_equal &&
                preview.prospective_root_equal &&
                preview.path_feasible && preview.match_ready;
            if (!certified) {
                return fail_output(
                    diagnostics_,
                    PickAssistReason::FinalPreviewRejected);
            }
            diagnostics_.state = PickAssistState::ReadyToSubmit;
            PickAssistOutput ready = braking_output();
            ready.submit_interact = true;
            return ready;
        }
        if (diagnostics_.root_error_m <=
                config_.arrival.latch_position_error_m &&
            diagnostics_.speed_mps <=
                config_.arrival.latch_simulation_speed_mps &&
            diagnostics_.yaw_error_radians <=
                config_.arrival.maximum_yaw_error_radians) {
            diagnostics_.state = PickAssistState::Settling;
            diagnostics_.settle_ticks = 0U;
            arrival_ticks_ = 0U;
            return braking_output();
        }
        if (diagnostics_.root_error_m <=
            config_.arrival.slow_radius_m) {
            output.override_steering = true;
            output.left_stick = arrival_navigation_stick(
                frozen_slot_.root_world.position,
                observation.displayed_root.position,
                observation.camera_azimuth,
                config_.arrival);
            output.right_stick = arrival_facing_stick(
                quat_mul_vec3(
                    frozen_slot_.root_world.rotation,
                    vec3(0.0F, 0.0F, 1.0F)),
                observation.camera_azimuth);
            output.force_strafe = true;
            return output;
        }
        output.override_steering = true;
        output.left_stick = ordinary_navigation_stick(
            frozen_slot_.root_world.position,
            observation.displayed_root.position,
            observation.camera_azimuth);
        return output;
    }
    if (active()) {
        if (observation.runtime_state != RuntimeState::Locomotion) {
            diagnostics_.state = PickAssistState::Failed;
            diagnostics_.reason = PickAssistReason::RuntimeChanged;
            return output;
        }
        if (observation.target == nullptr ||
            observation.target->handle.id != start_.target.id) {
            diagnostics_.state = PickAssistState::Failed;
            diagnostics_.reason = PickAssistReason::TargetUnavailable;
            return output;
        }
        if (!target_metadata_matches(start_, *observation.target)) {
            diagnostics_.state = PickAssistState::Failed;
            diagnostics_.reason = PickAssistReason::TargetChanged;
            return output;
        }
        capture_final_preview_diagnostics(
            diagnostics_, start_, observation);
        if (!observation_metrics_are_finite(observation)) {
            return fail_output(
                diagnostics_, PickAssistReason::OutsideTravelEnvelope);
        }
        if (observation.previews.has_value() &&
            !preview_roots_are_finite(*observation.previews)) {
            const PickAssistReason reason =
                diagnostics_.state == PickAssistState::FinalPreview
                ? PickAssistReason::FinalPreviewRejected
                : diagnostics_.state == PickAssistState::Preview
                ? PickAssistReason::NoFeasibleEntry
                : PickAssistReason::OutsideTravelEnvelope;
            return fail_output(diagnostics_, reason);
        }
    }
    switch (diagnostics_.state) {
    case PickAssistState::Idle:
    case PickAssistState::Submitted:
    case PickAssistState::Failed:
        return output;
    case PickAssistState::SlotApproach:
        return output;
    case PickAssistState::CoarseApproach: {
        diagnostics_.root_error_m = planar_distance(
            observation.displayed_root.position, entry_point_);
        diagnostics_.speed_mps = observation.displayed_planar_speed_mps;
        if (!is_finite(diagnostics_.root_error_m)) {
            return fail_output(
                diagnostics_, PickAssistReason::OutsideTravelEnvelope);
        }
        output.override_steering = true;
        if (diagnostics_.root_error_m <= config_.reach_entry_tolerance_m) {
            diagnostics_.state = PickAssistState::Preview;
            preview_ticks_ = 0U;
            output.needs_preview = true;
            return output;
        }
        output.left_stick = ordinary_navigation_stick(
            entry_point_, observation.displayed_root.position,
            observation.camera_azimuth);
        if (!is_finite(output.left_stick)) {
            return fail_output(
                diagnostics_, PickAssistReason::OutsideTravelEnvelope);
        }
        return output;
    }
    case PickAssistState::Preview: {
        output.override_steering = true;
        output.needs_preview = true;
        ++preview_ticks_;
        if (!observation.previews.has_value() ||
            observation.preview_snapshot_fingerprint !=
                observation.snapshot_fingerprint) {
            if (preview_ticks_ >= config_.maximum_preview_ticks) {
                diagnostics_.state = PickAssistState::Failed;
                diagnostics_.reason = PickAssistReason::PreviewDeadline;
                return {};
            }
            return output;
        }
        std::array<bool, 2> current_permitted =
            diagnostics_.slot_permitted;
        for (size_t i = 0; i < current_permitted.size(); ++i) {
            current_permitted[i] = current_permitted[i] &&
                same_entry_root(
                    observation.previews->at(i).prospective_root,
                    start_.slots.ordered[i].prospective_root);
        }
        const std::optional<size_t> selected = choose_pick_entry_slot(
            start_.slots,
            *observation.previews,
            current_permitted,
            start_.hand);
        if (!selected.has_value()) {
            bool have_current_permitted = false;
            bool all_permitted_are_current = true;
            bool any_path_feasible = false;
            for (size_t i = 0; i < current_permitted.size(); ++i) {
                if (diagnostics_.slot_permitted[i] &&
                    !current_permitted[i]) {
                    all_permitted_are_current = false;
                }
                if (current_permitted[i]) {
                    have_current_permitted = true;
                    any_path_feasible = any_path_feasible ||
                        observation.previews->at(i).path_feasible;
                }
            }
            if (have_current_permitted && all_permitted_are_current &&
                !any_path_feasible) {
                diagnostics_.state = PickAssistState::Failed;
                diagnostics_.reason = PickAssistReason::NoFeasibleEntry;
                return {};
            }
            if (preview_ticks_ >= config_.maximum_preview_ticks) {
                diagnostics_.state = PickAssistState::Failed;
                diagnostics_.reason = PickAssistReason::PreviewDeadline;
                return {};
            }
            return output;
        }
        diagnostics_.selected_slot = static_cast<int>(*selected);
        diagnostics_.route_length_m =
            diagnostics_.slot_route_lengths_m[*selected];
        diagnostics_.state = PickAssistState::FinalApproach;
        arrival_ticks_ = 0U;
        output.needs_preview = false;

        const PickEntrySlot& slot = start_.slots.ordered[*selected];
        diagnostics_.root_error_m = planar_distance(
            observation.displayed_root.position, slot.waypoint.position);
        if (!is_finite(diagnostics_.root_error_m)) {
            return fail_output(
                diagnostics_, PickAssistReason::OutsideTravelEnvelope);
        }
        output.force_strafe = true;
        output.left_stick = arrival_navigation_stick(
            slot.waypoint.position,
            observation.displayed_root.position,
            observation.camera_azimuth,
            config_.arrival);
        vec3 forward = quat_mul_vec3(
            slot.waypoint.rotation, vec3(0.0F, 0.0F, 1.0F));
        forward.y = 0.0F;
        if (!is_finite(forward)) {
            return fail_output(
                diagnostics_, PickAssistReason::OutsideTravelEnvelope);
        }
        output.right_stick = arrival_facing_stick(
            forward, observation.camera_azimuth);
        if (!is_finite(output.left_stick) ||
            !is_finite(output.right_stick)) {
            return fail_output(
                diagnostics_, PickAssistReason::OutsideTravelEnvelope);
        }
        return output;
    }
    case PickAssistState::FinalApproach: {
        ++arrival_ticks_;
        const PickEntrySlot& slot = start_.slots.ordered[
            static_cast<size_t>(diagnostics_.selected_slot)];
        diagnostics_.root_error_m = planar_distance(
            observation.displayed_root.position, slot.waypoint.position);
        diagnostics_.yaw_error_radians = yaw_error(
            yaw_radians(observation.displayed_root.rotation),
            slot.prospective_root.world_yaw_radians);
        diagnostics_.speed_mps = observation.displayed_planar_speed_mps;
        const float standoff_m = planar_distance(
            observation.displayed_root.position,
            start_.object_world.position);
        const float simulation_speed_mps =
            planar_speed(observation.simulation_velocity);
        if (!is_finite(diagnostics_.root_error_m) ||
            !is_finite(diagnostics_.yaw_error_radians) ||
            !is_finite(standoff_m) || !is_finite(simulation_speed_mps)) {
            return fail_output(
                diagnostics_, PickAssistReason::OutsideTravelEnvelope);
        }
        if (arrival_ready(
                diagnostics_.root_error_m,
                simulation_speed_mps,
                diagnostics_.yaw_error_radians,
                standoff_m,
                config_.arrival)) {
            diagnostics_.state = PickAssistState::Settling;
            diagnostics_.settle_ticks = 0U;
            if (arrival_ticks_ >= config_.maximum_arrival_ticks) {
                diagnostics_.state = PickAssistState::Failed;
                diagnostics_.reason = PickAssistReason::ArrivalDeadline;
                return {};
            }
            return braking_output();
        }
        output.override_steering = true;
        output.force_strafe = true;
        output.left_stick = arrival_navigation_stick(
            slot.waypoint.position,
            observation.displayed_root.position,
            observation.camera_azimuth,
            config_.arrival);
        vec3 forward = quat_mul_vec3(
            slot.waypoint.rotation, vec3(0.0F, 0.0F, 1.0F));
        forward.y = 0.0F;
        if (!is_finite(forward)) {
            return fail_output(
                diagnostics_, PickAssistReason::OutsideTravelEnvelope);
        }
        output.right_stick = arrival_facing_stick(
            forward, observation.camera_azimuth);
        if (!is_finite(output.left_stick) ||
            !is_finite(output.right_stick)) {
            return fail_output(
                diagnostics_, PickAssistReason::OutsideTravelEnvelope);
        }
        if (arrival_ticks_ >= config_.maximum_arrival_ticks) {
            diagnostics_.state = PickAssistState::Failed;
            diagnostics_.reason = PickAssistReason::ArrivalDeadline;
            return {};
        }
        return output;
    }
    case PickAssistState::Settling: {
        ++arrival_ticks_;
        const PickEntrySlot& slot = start_.slots.ordered[
            static_cast<size_t>(diagnostics_.selected_slot)];
        diagnostics_.root_error_m = planar_distance(
            observation.displayed_root.position, slot.waypoint.position);
        diagnostics_.yaw_error_radians = yaw_error(
            yaw_radians(observation.displayed_root.rotation),
            slot.prospective_root.world_yaw_radians);
        diagnostics_.speed_mps = observation.displayed_planar_speed_mps;
        const float standoff_m = planar_distance(
            observation.displayed_root.position,
            start_.object_world.position);
        if (!is_finite(diagnostics_.root_error_m) ||
            !is_finite(diagnostics_.yaw_error_radians) ||
            !is_finite(standoff_m)) {
            return fail_output(
                diagnostics_, PickAssistReason::OutsideTravelEnvelope);
        }
        const bool stable =
            diagnostics_.root_error_m <=
                config_.maximum_settle_position_error_m &&
            observation.displayed_planar_speed_mps <=
                config_.maximum_settle_displayed_speed_mps &&
            diagnostics_.yaw_error_radians <=
                config_.arrival.maximum_yaw_error_radians &&
            standoff_m >= config_.arrival.minimum_standoff_m &&
            standoff_m <= config_.arrival.maximum_standoff_m;
        if (stable) {
            ++diagnostics_.settle_ticks;
        } else {
            diagnostics_.settle_ticks = 0U;
        }
        if (diagnostics_.settle_ticks >= config_.required_settle_ticks) {
            diagnostics_.state = PickAssistState::FinalPreview;
            if (arrival_ticks_ >= config_.maximum_arrival_ticks) {
                diagnostics_.state = PickAssistState::Failed;
                diagnostics_.reason = PickAssistReason::ArrivalDeadline;
                return {};
            }
            return braking_output(true);
        }
        if (arrival_ticks_ >= config_.maximum_arrival_ticks) {
            diagnostics_.state = PickAssistState::Failed;
            diagnostics_.reason = PickAssistReason::ArrivalDeadline;
            return {};
        }
        return braking_output();
    }
    case PickAssistState::FinalPreview: {
        ++arrival_ticks_;
        if (arrival_ticks_ >= config_.maximum_arrival_ticks) {
            return fail_output(
                diagnostics_, PickAssistReason::ArrivalDeadline);
        }
        if (!observation.previews.has_value()) {
            return braking_output(true);
        }
        const size_t selected =
            static_cast<size_t>(diagnostics_.selected_slot);
        const PickEntryPreview& preview = observation.previews->at(selected);
        const bool certified =
            observation.preview_snapshot_fingerprint ==
                observation.snapshot_fingerprint &&
            preview.path_feasible && preview.match_ready &&
            same_entry_root(
                preview.prospective_root,
                start_.slots.ordered[selected].prospective_root);
        if (!certified) {
            diagnostics_.state = PickAssistState::Failed;
            diagnostics_.reason = PickAssistReason::FinalPreviewRejected;
            return {};
        }
        diagnostics_.state = PickAssistState::ReadyToSubmit;
        PickAssistOutput ready = braking_output();
        ready.submit_interact = true;
        return ready;
    }
    case PickAssistState::ReadyToSubmit:
        return braking_output();
    }
    return output;
}

std::optional<PickRequest> ControllerPickAssist::take_submission(
    uint64_t request_id) {
    if (diagnostics_.state != PickAssistState::ReadyToSubmit ||
        request_id == 0U) {
        return std::nullopt;
    }
    const PickRequest request{
        start_.target,
        start_.affordance_id,
        request_id,
    };
    diagnostics_.state = PickAssistState::Submitted;
    return request;
}

bool ControllerPickAssist::active() const {
    switch (diagnostics_.state) {
    case PickAssistState::CoarseApproach:
    case PickAssistState::Preview:
    case PickAssistState::FinalApproach:
    case PickAssistState::Settling:
    case PickAssistState::FinalPreview:
    case PickAssistState::ReadyToSubmit:
    case PickAssistState::SlotApproach:
        return true;
    case PickAssistState::Idle:
    case PickAssistState::Submitted:
    case PickAssistState::Failed:
        return false;
    }
    return false;
}

bool ControllerPickAssist::owns_manual_interact() const {
    return active();
}

const PickAssistDiagnostics& ControllerPickAssist::diagnostics() const {
    return diagnostics_;
}

const char* pick_assist_state_name(PickAssistState state) {
    switch (state) {
    case PickAssistState::Idle: return "Idle";
    case PickAssistState::CoarseApproach: return "CoarseApproach";
    case PickAssistState::Preview: return "Preview";
    case PickAssistState::FinalApproach: return "FinalApproach";
    case PickAssistState::Settling: return "Settling";
    case PickAssistState::FinalPreview: return "FinalPreview";
    case PickAssistState::ReadyToSubmit: return "ReadyToSubmit";
    case PickAssistState::Submitted: return "Submitted";
    case PickAssistState::Failed: return "Failed";
    case PickAssistState::SlotApproach: return "SlotApproach";
    }
    throw std::runtime_error("invalid pick-assist state");
}

const char* pick_assist_reason_name(PickAssistReason reason) {
    switch (reason) {
    case PickAssistReason::None: return "None";
    case PickAssistReason::Cancelled: return "Cancelled";
    case PickAssistReason::TargetUnavailable: return "TargetUnavailable";
    case PickAssistReason::TargetChanged: return "TargetChanged";
    case PickAssistReason::RuntimeChanged: return "RuntimeChanged";
    case PickAssistReason::OutsideTravelEnvelope:
        return "OutsideTravelEnvelope";
    case PickAssistReason::NoFeasibleEntry: return "NoFeasibleEntry";
    case PickAssistReason::PreviewDeadline: return "PreviewDeadline";
    case PickAssistReason::ArrivalDeadline: return "ArrivalDeadline";
    case PickAssistReason::FinalPreviewRejected:
        return "FinalPreviewRejected";
    case PickAssistReason::SlotChanged: return "SlotChanged";
    case PickAssistReason::NoAuthoredSlot: return "NoAuthoredSlot";
    case PickAssistReason::InvalidGeometry: return "InvalidGeometry";
    case PickAssistReason::TableBlocked: return "TableBlocked";
    case PickAssistReason::ObstacleBlocked: return "ObstacleBlocked";
    case PickAssistReason::AllSlotsBlocked: return "AllSlotsBlocked";
    case PickAssistReason::PoorMatch: return "PoorMatch";
    }
    throw std::runtime_error("invalid pick-assist reason");
}

PickAssistReason pick_assist_reason_from_slot_reason(PickSlotReason reason) {
    switch (reason) {
    case PickSlotReason::None:
        return PickAssistReason::None;
    case PickSlotReason::NoAuthoredSlot:
        return PickAssistReason::NoAuthoredSlot;
    case PickSlotReason::InvalidGeometry:
        return PickAssistReason::InvalidGeometry;
    case PickSlotReason::OutsideTravelEnvelope:
        return PickAssistReason::OutsideTravelEnvelope;
    case PickSlotReason::TableBlocked:
        return PickAssistReason::TableBlocked;
    case PickSlotReason::ObstacleBlocked:
        return PickAssistReason::ObstacleBlocked;
    case PickSlotReason::AllSlotsBlocked:
        return PickAssistReason::AllSlotsBlocked;
    }
    throw std::runtime_error("invalid pick-slot reason");
}

}  // namespace interaction
