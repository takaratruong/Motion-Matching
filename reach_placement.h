#pragma once

#include "reach_motion.h"

#include <cstddef>
#include <cstdint>

namespace reach {

inline constexpr uint8_t kYawPlacementCount = 12U;

float placement_yaw(uint8_t yaw_index);

interaction::Transform contact_alignment(
    const Pack& pack,
    size_t clip,
    uint8_t yaw_index,
    vec3 target_position);

interaction::Pose place_pose(
    const Pack& pack,
    size_t clip,
    uint8_t yaw_index,
    vec3 target_position,
    int32_t database_frame);

vec3 place_direction(vec3 direction, uint8_t yaw_index);

}  // namespace reach
