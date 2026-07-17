#pragma once

#include "vec.h"

namespace interaction {

struct ArrivalConfig {
    float slow_radius_m = 0.60F;
    float latch_position_error_m = 0.03F;
    float latch_simulation_speed_mps = 0.05F;
    float maximum_yaw_error_radians = 20.0F * PIf / 180.0F;
    float minimum_standoff_m = 0.35F;
    float maximum_standoff_m = 0.45F;
};

vec3 arrival_navigation_stick(
    vec3 target_world,
    vec3 current_world,
    float camera_azimuth,
    const ArrivalConfig& config = {});

vec3 arrival_facing_stick(
    vec3 target_forward_world,
    float camera_azimuth);

bool arrival_ready(
    float position_error_m,
    float planar_simulation_speed_mps,
    float yaw_error_radians,
    float standoff_m,
    const ArrivalConfig& config = {});

}  // namespace interaction
