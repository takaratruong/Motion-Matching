#pragma once

#include "interaction_matcher.h"

#include <array>
#include <cstdint>

namespace interaction {

struct IKConfig {
    float maximum_request_position_m = 0.12F;
    float maximum_request_orientation_radians = 0.436332313F;
    float accepted_position_m = 0.04F;
    float accepted_orientation_radians = 0.261799388F;
    float damping = 0.05F;
    float finite_difference_radians = 0.001F;
    float orientation_scale_m_per_radian = 0.25F;
    float maximum_step_radians = 0.10F;
    int32_t maximum_iterations = 8;
};

struct IKResult {
    bool accepted = false;
    Reason reason = Reason::None;
    float position_error_m = 0.0F;
    float orientation_error_radians = 0.0F;
    std::array<float, 7> joint_angles{};
};

IKResult solve_hand_ik(
    Pose& pose,
    Hand hand,
    Transform target_hand_world,
    const IKConfig& config);

}  // namespace interaction
