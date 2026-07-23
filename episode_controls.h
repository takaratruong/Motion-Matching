#pragma once

#include "g1_flat_motion_matcher.h"

#include <cmath>

namespace episode {

struct DigitalLocomotionInput {
    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
};

inline LocomotionCommand camera_relative_command(
    vec3 camera_forward,
    quat current_heading,
    const DigitalLocomotionInput& input,
    float speed) {
    camera_forward.y = 0.0F;
    if (length(camera_forward) < 1.0e-5F) {
        camera_forward = vec3(0.0F, 0.0F, 1.0F);
    }
    camera_forward = normalize(camera_forward);
    const vec3 screen_right = normalize(cross(
        camera_forward, vec3(0.0F, 1.0F, 0.0F)));
    vec3 direction{};
    if (input.forward) direction = direction + camera_forward;
    if (input.backward) direction = direction - camera_forward;
    if (input.right) direction = direction + screen_right;
    if (input.left) direction = direction - screen_right;
    if (length(direction) < 1.0e-5F) {
        return {vec3(), current_heading};
    }
    direction = normalize(direction);
    const float yaw = std::atan2(direction.x, direction.z);
    return {
        direction * speed,
        quat_from_angle_axis(yaw, vec3(0.0F, 1.0F, 0.0F)),
    };
}

}  // namespace episode
