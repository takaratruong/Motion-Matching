#pragma once

#include <cstdlib>

#include "array.h"
#include "quat.h"
#include "vec.h"

vec3 desired_velocity_update(
    vec3 gamepadstick_left,
    float camera_azimuth,
    quat simulation_rotation,
    float fwrd_speed,
    float side_speed,
    float back_speed);

quat desired_rotation_update(
    quat desired_rotation,
    vec3 gamepadstick_left,
    vec3 gamepadstick_right,
    float camera_azimuth,
    bool desired_strafe,
    vec3 desired_velocity);

vec3 simulation_collide_obstacles(
    vec3 prev_pos,
    vec3 next_pos,
    slice1d<vec3> obstacles_positions,
    slice1d<vec3> obstacles_scales,
    float radius = 0.6F);

void simulation_positions_update(
    vec3& position,
    vec3& velocity,
    vec3& acceleration,
    vec3 desired_velocity,
    float halflife,
    float dt,
    slice1d<vec3> obstacles_positions,
    slice1d<vec3> obstacles_scales);

void simulation_rotations_update(
    quat& rotation,
    vec3& angular_velocity,
    quat desired_rotation,
    float halflife,
    float dt);
