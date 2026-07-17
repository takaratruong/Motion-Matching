#include "interaction_pick_approach.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace interaction {
namespace {

constexpr float kReachPositionErrorM = 0.15F;
constexpr float kInteractionClearanceEpsilonM = 0.001F;
constexpr float kStandoffMinimumM = 0.35F;
constexpr float kStandoffMaximumM = 0.45F;
constexpr float kWaypointInvariantToleranceM = 2.0e-5F;

bool is_finite(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

bool is_finite(vec3 value) {
    return is_finite(value.x) &&
        is_finite(value.y) &&
        is_finite(value.z);
}

bool is_finite(quat value) {
    return is_finite(value.w) &&
        is_finite(value.x) &&
        is_finite(value.y) &&
        is_finite(value.z);
}

vec3 read_vec3(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 3U;
    return vec3(
        values.at(offset),
        values.at(offset + 1U),
        values.at(offset + 2U));
}

quat read_quat(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 4U;
    return quat(
        values.at(offset),
        values.at(offset + 1U),
        values.at(offset + 2U),
        values.at(offset + 3U));
}

float yaw_radians(quat rotation) {
    const vec3 facing =
        quat_mul_vec3(rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(facing.x, facing.z);
}

float shortest_angle(float angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

Transform scene_alignment(
    const Transform& source_object,
    const Transform& target_object) {
    const float yaw = shortest_angle(
        yaw_radians(target_object.rotation) -
        yaw_radians(source_object.rotation));
    const quat rotation =
        quat_from_angle_axis(yaw, vec3(0.0F, 1.0F, 0.0F));
    const vec3 rotated_source =
        quat_mul_vec3(rotation, source_object.position);
    return {
        vec3(
            target_object.position.x - rotated_source.x,
            0.0F,
            target_object.position.z - rotated_source.z),
        rotation};
}

}  // namespace

Transform make_pick_reach_waypoint(
    const Database& database,
    const InteractionTarget& target,
    uint32_t clip_index) {
    if (database.clip_count == 0U || clip_index >= database.clip_count ||
        clip_index >= database.range_starts.size() ||
        clip_index >= database.range_stops.size() ||
        database.bone_count != g1_skeleton::BoneCount) {
        throw std::runtime_error(
            "pick-entry pack has no valid clip root row");
    }
    const int32_t start = database.range_starts.at(clip_index);
    const int32_t stop = database.range_stops.at(clip_index);
    int32_t reach = -1;
    int32_t contact = -1;
    for (int32_t frame = start; frame < stop; ++frame) {
        const uint8_t phase = database.phases.at(
            static_cast<size_t>(frame));
        if (reach < 0 &&
            phase == static_cast<uint8_t>(Phase::Reach)) {
            reach = frame;
        }
        if (contact < 0 &&
            phase == static_cast<uint8_t>(Phase::Contact)) {
            contact = frame;
        }
    }
    if (reach < start || contact <= reach || contact - 1 < start) {
        throw std::runtime_error(
            "pick-entry clip has no Reach waypoint");
    }

    const Transform source_object{
        read_vec3(
            database.object_positions,
            static_cast<size_t>(contact - 1)),
        read_quat(
            database.object_rotations,
            static_cast<size_t>(contact - 1))};
    const Transform scene_from_source =
        scene_alignment(source_object, target.object_world);
    constexpr size_t root = g1_skeleton::Simulation;
    const size_t root_row =
        static_cast<size_t>(reach) * database.bone_count + root;
    const Transform source_root{
        read_vec3(database.positions, root_row),
        read_quat(database.rotations, root_row)};
    return compose(scene_from_source, source_root);
}

const char* pick_entry_slot_name(PickEntrySlotIdentity identity) {
    switch (identity) {
    case PickEntrySlotIdentity::Plus: return "Plus";
    case PickEntrySlotIdentity::Minus: return "Minus";
    }
    throw std::runtime_error(
        "pick-entry preview has an invalid slot identity");
}

PickEntrySlots make_pick_entry_slots(
    const Transform& reach_waypoint,
    const InteractionTarget& target) {
    if (target.affordances.size() != 1U) {
        throw std::runtime_error(
            "pick-entry waypoint requires one affordance");
    }
    const GraspAffordance& affordance = target.affordances.front();
    if (affordance.hand != Hand::Right &&
        affordance.hand != Hand::Left) {
        throw std::runtime_error(
            "pick-entry waypoint requires a valid hand");
    }
    const Transform stable_reach_waypoint = reach_waypoint;
    const float approach_object_length = length(
        affordance.approach_direction_object);
    const float object_rotation_norm =
        quat_length(target.object_world.rotation);
    const float reach_rotation_norm = quat_length(reach_waypoint.rotation);
    if (!is_finite(reach_waypoint.position) ||
        !is_finite(reach_waypoint.rotation) ||
        !is_finite(target.object_world.position) ||
        !is_finite(target.object_world.rotation) ||
        !is_finite(target.object_dimensions) ||
        !is_finite(affordance.approach_direction_object) ||
        !is_finite(affordance.clearance_radius) ||
        !is_finite(approach_object_length) ||
        !is_finite(object_rotation_norm) ||
        !is_finite(reach_rotation_norm) ||
        target.object_dimensions.x <= 0.0F ||
        target.object_dimensions.y <= 0.0F ||
        target.object_dimensions.z <= 0.0F ||
        affordance.clearance_radius < 0.0F ||
        std::abs(approach_object_length - 1.0F) > 2.0e-5F ||
        std::abs(affordance.approach_direction_object.y) > 2.0e-5F ||
        std::abs(object_rotation_norm - 1.0F) > 1.0e-3F ||
        std::abs(reach_rotation_norm - 1.0F) > 1.0e-3F) {
        throw std::runtime_error(
            "pick-entry waypoint has invalid geometry");
    }
    vec3 approach_world = quat_mul_vec3(
        target.object_world.rotation,
        affordance.approach_direction_object);
    approach_world.y = 0.0F;
    const float approach_world_length = length(approach_world);
    if (!is_finite(approach_world_length) ||
        approach_world_length <= 1.0e-5F) {
        throw std::runtime_error(
            "pick-entry waypoint has no planar approach");
    }
    approach_world = approach_world / approach_world_length;
    const vec3 right_of_approach =
        cross(vec3(0.0F, 1.0F, 0.0F), approach_world);
    const vec3 lateral_object = quat_mul_vec3(
        quat_inv(target.object_world.rotation), right_of_approach);
    const float object_support_radius_m = 0.5F * (
        std::abs(lateral_object.x) * target.object_dimensions.x +
        std::abs(lateral_object.y) * target.object_dimensions.y +
        std::abs(lateral_object.z) * target.object_dimensions.z);
    const float clearance_chord_m = object_support_radius_m +
        affordance.clearance_radius +
        kInteractionClearanceEpsilonM;
    const vec3 object = target.object_world.position;
    const vec3 reach_radius(
        reach_waypoint.position.x - object.x,
        0.0F,
        reach_waypoint.position.z - object.z);
    const float standoff_m = length(reach_radius);
    if (!is_finite(lateral_object) ||
        !is_finite(object_support_radius_m) ||
        !is_finite(clearance_chord_m) ||
        !is_finite(standoff_m) ||
        standoff_m < kStandoffMinimumM ||
        standoff_m > kStandoffMaximumM ||
        clearance_chord_m <= 0.0F ||
        clearance_chord_m > kReachPositionErrorM ||
        clearance_chord_m > 2.0F * standoff_m) {
        throw std::runtime_error(
            "pick-entry waypoint has invalid standoff or chord");
    }
    const float arc = 2.0F * std::asin(
        clearance_chord_m / (2.0F * standoff_m));
    const float hand_sign = affordance.hand == Hand::Right
        ? 1.0F
        : -1.0F;
    const vec3 hand_lateral = hand_sign * right_of_approach;
    if (!is_finite(arc) || !is_finite(hand_lateral)) {
        throw std::runtime_error(
            "pick-entry slots arc is non-finite");
    }

    const auto make_slot = [&](PickEntrySlotIdentity identity, float sign) {
        const vec3 radial = quat_mul_vec3(
            quat_from_angle_axis(
                sign * arc, vec3(0.0F, 1.0F, 0.0F)),
            reach_radius);
        const vec3 position(
            object.x + radial.x,
            stable_reach_waypoint.position.y,
            object.z + radial.z);
        const quat rotation = stable_reach_waypoint.rotation;
        const float hand_score = dot(
            position - stable_reach_waypoint.position, hand_lateral);
        return PickEntrySlot{
            identity,
            Transform{position, rotation},
            PickEntryRoot{
                position.x,
                position.z,
                yaw_radians(stable_reach_waypoint.rotation)},
            hand_score};
    };

    const PickEntrySlots slots{
        std::array<PickEntrySlot, 2>{
            make_slot(PickEntrySlotIdentity::Plus, +1.0F),
            make_slot(PickEntrySlotIdentity::Minus, -1.0F)},
        clearance_chord_m,
        standoff_m};
    const float expected_yaw =
        yaw_radians(stable_reach_waypoint.rotation);
    for (const PickEntrySlot& slot : slots.ordered) {
        vec3 chord_delta =
            slot.waypoint.position - stable_reach_waypoint.position;
        chord_delta.y = 0.0F;
        vec3 candidate_radius = slot.waypoint.position - object;
        candidate_radius.y = 0.0F;
        const float chord_error_m = std::abs(
            length(chord_delta) - clearance_chord_m);
        const float standoff_error_m = std::abs(
            length(candidate_radius) - standoff_m);
        const float height_error_m = std::abs(
            slot.waypoint.position.y -
            stable_reach_waypoint.position.y);
        const float rotation_same_sign_error = std::sqrt(
            (slot.waypoint.rotation.w -
             stable_reach_waypoint.rotation.w) *
                (slot.waypoint.rotation.w -
                 stable_reach_waypoint.rotation.w) +
            (slot.waypoint.rotation.x -
             stable_reach_waypoint.rotation.x) *
                (slot.waypoint.rotation.x -
                 stable_reach_waypoint.rotation.x) +
            (slot.waypoint.rotation.y -
             stable_reach_waypoint.rotation.y) *
                (slot.waypoint.rotation.y -
                 stable_reach_waypoint.rotation.y) +
            (slot.waypoint.rotation.z -
             stable_reach_waypoint.rotation.z) *
                (slot.waypoint.rotation.z -
                 stable_reach_waypoint.rotation.z));
        const float rotation_opposite_sign_error = std::sqrt(
            (slot.waypoint.rotation.w +
             stable_reach_waypoint.rotation.w) *
                (slot.waypoint.rotation.w +
                 stable_reach_waypoint.rotation.w) +
            (slot.waypoint.rotation.x +
             stable_reach_waypoint.rotation.x) *
                (slot.waypoint.rotation.x +
                 stable_reach_waypoint.rotation.x) +
            (slot.waypoint.rotation.y +
             stable_reach_waypoint.rotation.y) *
                (slot.waypoint.rotation.y +
                 stable_reach_waypoint.rotation.y) +
            (slot.waypoint.rotation.z +
             stable_reach_waypoint.rotation.z) *
                (slot.waypoint.rotation.z +
                 stable_reach_waypoint.rotation.z));
        const float rotation_copy_error = std::min(
            rotation_same_sign_error, rotation_opposite_sign_error);
        const float yaw_error_radians = std::abs(
            shortest_angle(
                slot.prospective_root.world_yaw_radians - expected_yaw));
        if (!is_finite(slot.waypoint.position) ||
            !is_finite(slot.waypoint.rotation) ||
            !is_finite(slot.prospective_root.world_x) ||
            !is_finite(slot.prospective_root.world_z) ||
            !is_finite(slot.prospective_root.world_yaw_radians) ||
            !is_finite(slot.hand_score) ||
            !is_finite(chord_error_m) ||
            !is_finite(standoff_error_m) ||
            !is_finite(height_error_m) ||
            !is_finite(rotation_copy_error) ||
            !is_finite(yaw_error_radians) ||
            length(candidate_radius) < kStandoffMinimumM ||
            length(candidate_radius) > kStandoffMaximumM ||
            chord_error_m > kWaypointInvariantToleranceM ||
            standoff_error_m > kWaypointInvariantToleranceM ||
            height_error_m > kWaypointInvariantToleranceM ||
            rotation_copy_error > kWaypointInvariantToleranceM ||
            yaw_error_radians > kWaypointInvariantToleranceM) {
            throw std::runtime_error(
                "pick-entry slots arc invariant failed");
        }
    }
    return slots;
}

std::optional<size_t> choose_pick_entry_slot(
    const PickEntrySlots& slots,
    const std::array<PickEntryPreview, 2>& previews,
    const std::array<bool, 2>& permitted,
    Hand active_hand) {
    std::array<size_t, 2> eligible_slots{};
    size_t count = 0U;
    for (size_t i = 0; i < 2; ++i) {
        const bool eligible = permitted[i] &&
            previews[i].path_feasible && previews[i].match_ready;
        if (eligible) {
            eligible_slots[count++] = i;
        }
    }
    if (count == 0U) {
        return std::nullopt;
    }
    if (count == 1U) {
        return eligible_slots[0];
    }
    if (slots.ordered[0].hand_score > slots.ordered[1].hand_score) {
        return 0U;
    }
    if (slots.ordered[1].hand_score > slots.ordered[0].hand_score) {
        return 1U;
    }
    return active_hand == Hand::Right ? 0U : 1U;
}

}  // namespace interaction
