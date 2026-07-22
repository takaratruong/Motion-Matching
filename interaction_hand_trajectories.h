#pragma once

#include "interaction_ik.h"
#include "interaction_target.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace interaction {

enum class GraspOrientationMode : uint8_t {
    ExactPose = 0U,
    ApproachAxis = 1U,
    PositionOnly = 2U,
};

enum class TrajectoryMatchTier : uint8_t {
    Exact = 0U,
    AxisFallback = 1U,
    PositionOnly = 2U,
};

struct HandTrajectoryQuery {
    Transform object_world{};
    vec3 object_dimensions{};
    Hand hand = Hand::Right;
    vec3 grasp_world_position{};
    quat grasp_world_rotation{};
    vec3 approach_world_direction{1.0F, 0.0F, 0.0F};
    GraspOrientationMode orientation_mode = GraspOrientationMode::ExactPose;
};

struct HandTrajectoryConfig {
    float maximum_grasp_position_error_m = 0.12F;
    float maximum_grasp_orientation_error_radians = 0.436332313F;
    size_t maximum_compatible_clips = 4096U;
};

enum class SupportKind : uint8_t {
    Table = 0U,
    Ground = 1U,
};

struct HandTrajectory {
    int32_t clip = -1;
    int32_t start_frame = -1;
    int32_t reach_frame = -1;
    int32_t contact_frame = -1;
    int32_t lift_frame = -1;
    size_t reach_point = 0U;
    size_t contact_point = 0U;
    float cost = 0.0F;
    TrajectoryMatchTier match_tier = TrajectoryMatchTier::Exact;
    SupportKind support = SupportKind::Table;
    Transform source_object{};
    vec3 start_root_in_source_object{};
    vec3 source_approach_direction_object{1.0F, 0.0F, 0.0F};
    std::vector<Transform> hands_in_source_object;
    std::vector<vec3> elbows_in_source_object;
};

struct MappedHandTrajectory {
    std::vector<Transform> hands;
    std::vector<vec3> elbows;
};

struct ShapedHandTrajectory {
    std::vector<Pose> poses;
    MappedHandTrajectory path;
    float achieved_orientation_error_radians = 0.0F;
    bool contact_accepted = false;
    Reason reason = Reason::None;
};

struct OrientedBox {
    Transform world{};
    vec3 dimensions{};
};

struct EnvironmentGeometry {
    std::vector<OrientedBox> boxes;
};

enum class TrajectoryFeasibilityReason : uint8_t {
    None = 0U,
    ObjectCollision = 1U,
    EnvironmentCollision = 2U,
};

struct TrajectoryFeasibility {
    TrajectoryFeasibilityReason reason = TrajectoryFeasibilityReason::None;
    size_t sample = 0U;
};

struct TrajectoryCollisionConfig {
    float wrist_radius_m = 0.04F;
    float forearm_radius_m = 0.035F;
    float joint_radius_m = 0.035F;
    float limb_radius_m = 0.045F;
    float torso_radius_m = 0.10F;
};

EnvironmentGeometry make_recorded_table_geometry(
    const Transform& table_world,
    vec3 table_dimensions,
    float leg_thickness_m = 0.04F);

EnvironmentGeometry make_coverage_environment(
    const Transform& table_world,
    vec3 table_dimensions,
    float leg_thickness_m = 0.04F);

SupportKind support_kind(const Database& database, size_t clip);

std::vector<HandTrajectory> select_hand_trajectories(
    const Database& database,
    const HandTrajectoryQuery& query,
    const HandTrajectoryConfig& config = HandTrajectoryConfig{});

MappedHandTrajectory map_hand_trajectory(
    const HandTrajectory& trajectory,
    const HandTrajectoryQuery& query);

Transform hand_trajectory_scene_alignment(
    const HandTrajectory& trajectory,
    const HandTrajectoryQuery& query);

bool starts_on_allowed_side(
    const HandTrajectory& trajectory,
    const HandTrajectoryQuery& query,
    vec3 scene_front,
    float minimum_dot = 0.0F);

ShapedHandTrajectory shape_hand_trajectory(
    const Database& database,
    const HandTrajectory& trajectory,
    const HandTrajectoryQuery& query,
    const IKConfig& config = IKConfig{});

TrajectoryFeasibility evaluate_trajectory_feasibility(
    const MappedHandTrajectory& trajectory,
    size_t contact_point,
    const OrientedBox& object,
    const EnvironmentGeometry& environment,
    const TrajectoryCollisionConfig& config = TrajectoryCollisionConfig{});

TrajectoryFeasibility evaluate_shaped_trajectory_feasibility(
    const ShapedHandTrajectory& trajectory,
    size_t contact_point,
    Hand hand,
    const OrientedBox& object,
    const EnvironmentGeometry& environment,
    const TrajectoryCollisionConfig& config = TrajectoryCollisionConfig{});

}  // namespace interaction
