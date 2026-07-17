#pragma once

#include "interaction_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pick_entry_oracle {

inline constexpr float kInteractionClearanceEpsilonM = 0.001F;
inline constexpr float kMaximumReachPositionErrorM = 0.15F;
inline constexpr float kMinimumStandoffM = 0.35F;
inline constexpr float kMaximumStandoffM = 0.45F;

struct OracleRoots {
    interaction::PickEntryRoot reach{};
    interaction::PickEntryRoot minus{};
    interaction::PickEntryRoot plus{};
};

inline vec3 read_vec3(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 3U;
    return vec3(
        values.at(offset),
        values.at(offset + 1U),
        values.at(offset + 2U));
}

inline quat read_quat(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 4U;
    return quat(
        values.at(offset),
        values.at(offset + 1U),
        values.at(offset + 2U),
        values.at(offset + 3U));
}

inline interaction::Transform frame_transform(
    const std::vector<float>& positions,
    const std::vector<float>& rotations,
    size_t index) {
    return {read_vec3(positions, index), read_quat(rotations, index)};
}

inline int32_t first_phase_frame(
    const interaction::Database& database,
    interaction::Phase phase) {
    if (database.clip_count == 0U || database.range_starts.empty() ||
        database.range_stops.empty()) {
        throw std::runtime_error("pick-entry oracle pack has no clip 0");
    }
    const int32_t start = database.range_starts.at(0U);
    const int32_t stop = database.range_stops.at(0U);
    for (int32_t frame = start; frame < stop; ++frame) {
        if (database.phases.at(static_cast<size_t>(frame)) ==
            static_cast<uint8_t>(phase)) {
            return frame;
        }
    }
    throw std::runtime_error(
        "pick-entry oracle pack is missing a required phase");
}

inline float yaw_radians(quat rotation) {
    const float magnitude = quat_length(rotation);
    if (!std::isfinite(magnitude) || !(magnitude > 1.0e-6F)) {
        throw std::runtime_error("pick-entry oracle rotation is invalid");
    }
    rotation = rotation / magnitude;
    const vec3 forward = quat_mul_vec3(
        rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(forward.x, forward.z);
}

inline float shortest_angle(float angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

inline interaction::Transform scene_alignment(
    interaction::Transform source_object,
    interaction::Transform target_object) {
    const float yaw = shortest_angle(
        yaw_radians(target_object.rotation) -
        yaw_radians(source_object.rotation));
    const quat rotation = quat_from_angle_axis(
        yaw, vec3(0.0F, 1.0F, 0.0F));
    const vec3 rotated_source = quat_mul_vec3(
        rotation, source_object.position);
    return {
        vec3(
            target_object.position.x - rotated_source.x,
            0.0F,
            target_object.position.z - rotated_source.z),
        rotation,
    };
}

inline interaction::Transform reach_waypoint(
    const interaction::Database& database,
    const interaction::InteractionTarget& target) {
    const int32_t reach = first_phase_frame(
        database, interaction::Phase::Reach);
    const int32_t contact = first_phase_frame(
        database, interaction::Phase::Contact);
    if (contact <= reach) {
        throw std::runtime_error(
            "pick-entry oracle contact does not follow Reach");
    }
    const interaction::Transform source_object = frame_transform(
        database.object_positions,
        database.object_rotations,
        static_cast<size_t>(contact - 1));
    const interaction::Transform mapping = scene_alignment(
        source_object, target.object_world);
    const interaction::Pose pose = interaction::pose_at_frame(database, reach);
    constexpr size_t root =
        static_cast<size_t>(g1_skeleton::Simulation);
    return interaction::compose(
        mapping, {pose.positions[root], pose.rotations[root]});
}

inline OracleRoots make_oracle_roots(
    const interaction::Database& database,
    const interaction::InteractionTarget& target) {
    if (target.affordances.size() != 1U) {
        throw std::runtime_error(
            "pick-entry oracle target must have one affordance");
    }
    const interaction::Transform reach = reach_waypoint(database, target);
    const interaction::GraspAffordance& affordance =
        target.affordances.front();

    vec3 approach_world = quat_mul_vec3(
        target.object_world.rotation,
        affordance.approach_direction_object);
    approach_world.y = 0.0F;
    const float approach_length = length(approach_world);
    if (!std::isfinite(approach_length) ||
        !(approach_length > 1.0e-5F)) {
        throw std::runtime_error(
            "pick-entry oracle target has no planar approach");
    }
    approach_world = approach_world / approach_length;
    const vec3 right_of_approach = cross(
        vec3(0.0F, 1.0F, 0.0F), approach_world);
    const vec3 lateral_object = quat_mul_vec3(
        quat_inv(target.object_world.rotation), right_of_approach);
    const float object_support_radius_m = 0.5F * (
        std::fabs(lateral_object.x) * target.object_dimensions.x +
        std::fabs(lateral_object.y) * target.object_dimensions.y +
        std::fabs(lateral_object.z) * target.object_dimensions.z);
    const float clearance_chord_m = object_support_radius_m +
        affordance.clearance_radius + kInteractionClearanceEpsilonM;
    const vec3 reach_radius(
        reach.position.x - target.object_world.position.x,
        0.0F,
        reach.position.z - target.object_world.position.z);
    const float standoff_m = length(reach_radius);
    if (!std::isfinite(clearance_chord_m) ||
        !std::isfinite(standoff_m) ||
        !(clearance_chord_m > 0.0F) ||
        clearance_chord_m > kMaximumReachPositionErrorM ||
        standoff_m < kMinimumStandoffM ||
        standoff_m > kMaximumStandoffM ||
        clearance_chord_m > 2.0F * standoff_m) {
        throw std::runtime_error(
            "pick-entry oracle arc geometry is outside approved bounds");
    }
    const float arc = 2.0F * std::asin(
        clearance_chord_m / (2.0F * standoff_m));
    const vec3 minus_radius = quat_mul_vec3(
        quat_from_angle_axis(-arc, vec3(0.0F, 1.0F, 0.0F)),
        reach_radius);
    const vec3 plus_radius = quat_mul_vec3(
        quat_from_angle_axis(+arc, vec3(0.0F, 1.0F, 0.0F)),
        reach_radius);
    const float reach_yaw = yaw_radians(reach.rotation);
    return {
        {reach.position.x, reach.position.z, reach_yaw},
        {
            target.object_world.position.x + minus_radius.x,
            target.object_world.position.z + minus_radius.z,
            reach_yaw,
        },
        {
            target.object_world.position.x + plus_radius.x,
            target.object_world.position.z + plus_radius.z,
            reach_yaw,
        },
    };
}

}  // namespace pick_entry_oracle
