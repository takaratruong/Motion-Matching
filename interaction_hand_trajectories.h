#pragma once

#include "interaction_target.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace interaction {

struct HandTrajectoryQuery {
    Transform object_world{};
    vec3 object_dimensions{};
    Hand hand = Hand::Right;
    vec3 grasp_world_position{};
    std::optional<quat> grasp_world_rotation;
};

struct HandTrajectoryConfig {
    float maximum_grasp_position_error_m = 0.20F;
    float maximum_grasp_orientation_error_radians = 0.785398163F;
    float maximum_log_dimension_error = 0.75F;
    size_t maximum_compatible_clips = 4096U;
};

struct HandTrajectory {
    int32_t clip = -1;
    int32_t reach_frame = -1;
    int32_t contact_frame = -1;
    int32_t lift_frame = -1;
    size_t contact_point = 0U;
    float cost = 0.0F;
    Transform source_object{};
    std::vector<Transform> hands_in_source_object;
    std::vector<vec3> elbows_in_source_object;
};

struct MappedHandTrajectory {
    std::vector<Transform> hands;
    std::vector<vec3> elbows;
};

struct OrientedBox {
    Transform world{};
    vec3 dimensions{};
};

struct ShelfGeometry {
    std::array<OrientedBox, 5> boxes{};
};

enum class TrajectoryFeasibilityReason : uint8_t {
    None = 0U,
    ObjectCollision = 1U,
    ShelfCollision = 2U,
};

struct TrajectoryFeasibility {
    TrajectoryFeasibilityReason reason = TrajectoryFeasibilityReason::None;
    size_t sample = 0U;
};

struct TrajectoryCollisionConfig {
    float wrist_radius_m = 0.04F;
    float forearm_radius_m = 0.035F;
};

ShelfGeometry make_recorded_table_geometry(
    const Transform& table_world,
    vec3 table_dimensions,
    float leg_thickness_m = 0.04F);

std::vector<HandTrajectory> select_hand_trajectories(
    const Database& database,
    const HandTrajectoryQuery& query,
    const HandTrajectoryConfig& config = HandTrajectoryConfig{});

MappedHandTrajectory map_hand_trajectory(
    const HandTrajectory& trajectory,
    const HandTrajectoryQuery& query);

Transform hand_trajectory_world_mapping(
    const HandTrajectory& trajectory,
    const HandTrajectoryQuery& query);

TrajectoryFeasibility evaluate_trajectory_feasibility(
    const MappedHandTrajectory& trajectory,
    size_t contact_point,
    const OrientedBox& object,
    const ShelfGeometry& shelf,
    const TrajectoryCollisionConfig& config = TrajectoryCollisionConfig{});

}  // namespace interaction
