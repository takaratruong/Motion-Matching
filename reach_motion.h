#pragma once

#include "interaction_pose.h"
#include "reach_database.h"

#include <cstddef>
#include <cstdint>

namespace reach {

interaction::Pose pose_at_frame(const Database& database, int32_t frame);
interaction::Transform endpoint_transform(
    const Database& database,
    size_t clip);
vec3 approach_direction(const Database& database, size_t clip);

}  // namespace reach

