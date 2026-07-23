#pragma once

#include "g1_arm_joint_metadata.h"
#include "interaction_ik.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace interaction {

inline constexpr size_t kLeftUpperBodyJointCount = 10U;
using LeftUpperBodyAngles =
    std::array<float, kLeftUpperBodyJointCount>;

struct ElbowPole {
    bool valid = false;
    vec3 shoulder{};
    vec3 axis{};
    vec3 direction{};
};

struct PostureIKConfig {
    float accepted_position_m = 0.001F;
    float accepted_orientation_radians = 0.261799388F;
    float initial_damping = 0.05F;
    float minimum_damping = 0.0001F;
    float maximum_damping = 100.0F;
    float finite_difference_radians = 0.001F;
    float orientation_scale_m_per_radian = 0.10F;
    float elbow_pole_scale_m = 0.04F;
    LeftUpperBodyAngles source_scale_m_per_radian{
        0.10F, 0.15F, 0.15F,
        0.020F, 0.020F, 0.020F, 0.015F,
        0.005F, 0.005F, 0.005F};
    LeftUpperBodyAngles temporal_scale_m_per_radian{
        0.030F, 0.030F, 0.030F,
        0.015F, 0.015F, 0.015F, 0.015F,
        0.010F, 0.010F, 0.010F};
    float maximum_step_radians = 0.10F;
    int32_t maximum_iterations = 30;
};

struct PostureIKResult {
    bool accepted = false;
    Reason reason = Reason::None;
    bool joint_limit_saturated = false;
    bool elbow_pole_degenerate = false;
    float position_error_m = 0.0F;
    float orientation_error_radians = 0.0F;
    float elbow_pole_error_radians = 0.0F;
    double objective = 0.0;
    int32_t iterations = 0;
    LeftUpperBodyAngles joint_angles{};
    std::array<bool, kLeftUpperBodyJointCount> at_limit{};
};

LeftUpperBodyAngles decompose_left_upper_body(const Pose& pose);
void apply_left_upper_body(Pose& pose, const LeftUpperBodyAngles& angles);
ElbowPole left_elbow_pole(const Pose& pose);
float transported_elbow_pole_error(
    const ElbowPole& reference,
    const ElbowPole& current);

PostureIKResult solve_left_hand_posture_ik(
    Pose& pose,
    Transform target_hand_world,
    const Pose& source_pose,
    const LeftUpperBodyAngles& temporal_seed,
    const PostureIKConfig& config = PostureIKConfig{});

}  // namespace interaction
