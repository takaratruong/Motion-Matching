#pragma once

// Straight-in grasp approach interfaces are implemented test-first by the
// supervised feature job.

#include "interaction_hand_trajectories.h"
#include "interaction_posture_ik.h"
#include "reach_coverage.h"
#include "vec.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace reach {

struct StraightApproachConfig {
    float corridor_length_m = 0.20F;
    float corridor_radius_m = 0.03F;
    float close_distance_m = 0.03F;
    float maximum_backward_step_m = 0.002F;
    float maximum_blend_seconds = 0.20F;
    float maximum_pregrasp_orientation_radians = 0.174532925F;
    float open_gripper_radius_m = 0.065F;
    float closed_gripper_radius_m = 0.040F;
    std::array<float, 7U> open_active_hand_dof{};
    bool use_explicit_closed_hand_dof = false;
    std::array<float, 7U> closed_active_hand_dof{};
};

struct StraightApproachQuality {
    bool finite = false;
    bool reaches_pregrasp_plane = false;
    float maximum_lateral_m = 0.0F;
    float rms_lateral_m = 0.0F;
    float backward_ratio = 0.0F;
    float maximum_angle_radians = 0.0F;
    float rms_angle_radians = 0.0F;
    size_t corridor_start = 0U;
};

bool valid_straight_approach_config(
    const StraightApproachConfig& config);

StraightApproachQuality measure_straight_approach(
    const std::vector<vec3>& wrist_path,
    vec3 contact_world,
    vec3 approach_world,
    const StraightApproachConfig& config = {});

bool straight_approach_quality_less(
    const StraightApproachQuality& left,
    const StraightApproachQuality& right);

enum class CorridorRetargetFailure : uint8_t {
    None = 0U,
    NoPregraspCoverage,
    InvalidInput,
    InvalidSolver,
    OutsideCorridor,
    BackwardMotion,
    ObjectCollision,
    EnvironmentCollision,
};

struct CorridorRetargetResult {
    bool accepted = false;
    CorridorRetargetFailure failure =
        CorridorRetargetFailure::InvalidInput;
    size_t failure_sample = 0U;
    size_t blend_start = 0U;
    size_t corridor_start = 0U;
    float active_arm_deformation = 0.0F;
    StraightApproachQuality quality{};
    std::vector<interaction::Pose> poses;
};

CorridorRetargetResult retarget_straight_approach(
    const std::vector<interaction::Pose>& source,
    Hand hand,
    const interaction::Transform& hand_world,
    vec3 approach_world,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    float fps,
    const StraightApproachConfig& config = {},
    const interaction::PostureIKConfig& ik_config = {},
    const interaction::TrajectoryCollisionConfig& collision_config = {});

}  // namespace reach
