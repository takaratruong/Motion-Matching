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
int32_t clip_contact_frame(const Database& database, size_t clip);
int32_t clip_return_start(const Database& database, size_t clip);
int32_t clip_return_stop(const Database& database, size_t clip);

}  // namespace reach
