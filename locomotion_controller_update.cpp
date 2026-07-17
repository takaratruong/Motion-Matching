#include "locomotion_controller_update.h"

#include "spring.h"

#include <cmath>

vec3 desired_velocity_update(
    const vec3 gamepadstick_left,
    const float camera_azimuth,
    const quat simulation_rotation,
    const float fwrd_speed,
    const float side_speed,
    const float back_speed)
{
    // Find stick position in world space by rotating using camera azimuth
    vec3 global_stick_direction = quat_mul_vec3(
        quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0)), gamepadstick_left);

    // Find stick position local to current facing direction
    vec3 local_stick_direction = quat_inv_mul_vec3(
        simulation_rotation, global_stick_direction);

    // Scale stick by forward, sideways and backwards speeds
    vec3 local_desired_velocity = local_stick_direction.z > 0.0 ?
        vec3(side_speed, 0.0f, fwrd_speed) * local_stick_direction :
        vec3(side_speed, 0.0f, back_speed) * local_stick_direction;

    // Re-orientate into the world space
    return quat_mul_vec3(simulation_rotation, local_desired_velocity);
}

quat desired_rotation_update(
    const quat desired_rotation,
    const vec3 gamepadstick_left,
    const vec3 gamepadstick_right,
    const float camera_azimuth,
    const bool desired_strafe,
    const vec3 desired_velocity)
{
    quat desired_rotation_curr = desired_rotation;

    // If strafe is active then desired direction is coming from right
    // stick as long as that stick is being used, otherwise we assume
    // forward facing
    if (desired_strafe)
    {
        vec3 desired_direction = quat_mul_vec3(quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0)), vec3(0, 0, -1));

        if (length(gamepadstick_right) > 0.01f)
        {
            desired_direction = quat_mul_vec3(quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0)), normalize(gamepadstick_right));
        }

        return quat_from_angle_axis(atan2f(desired_direction.x, desired_direction.z), vec3(0, 1, 0));
    }

    // If strafe is not active the desired direction comes from the left
    // stick as long as that stick is being used
    else if (length(gamepadstick_left) > 0.01f)
    {
        vec3 desired_direction = normalize(desired_velocity);
        return quat_from_angle_axis(atan2f(desired_direction.x, desired_direction.z), vec3(0, 1, 0));
    }

    // Otherwise desired direction remains the same
    else
    {
        return desired_rotation_curr;
    }
}

// Collide against the obscales which are
// essentially bounding boxes of a given size
vec3 simulation_collide_obstacles(
    const vec3 prev_pos,
    const vec3 next_pos,
    const slice1d<vec3> obstacles_positions,
    const slice1d<vec3> obstacles_scales,
    const float radius)
{
    vec3 dx = next_pos - prev_pos;
    vec3 proj_pos = prev_pos;

    // Substep because I'm too lazy to implement CCD
    int substeps = 1 + (int)(length(dx) * 5.0f);

    for (int j = 0; j < substeps; j++)
    {
        proj_pos = proj_pos + dx / substeps;

        for (int i = 0; i < obstacles_positions.size; i++)
        {
            // Find nearest point inside obscale and push out
            vec3 nearest = clamp(proj_pos,
              obstacles_positions(i) - 0.5f * obstacles_scales(i),
              obstacles_positions(i) + 0.5f * obstacles_scales(i));

            if (length(nearest - proj_pos) < radius)
            {
                proj_pos = radius * normalize(proj_pos - nearest) + nearest;
            }
        }
    }

    return proj_pos;
}

// Taken from https://theorangeduck.com/page/spring-roll-call#controllers
void simulation_positions_update(
    vec3& position,
    vec3& velocity,
    vec3& acceleration,
    const vec3 desired_velocity,
    const float halflife,
    const float dt,
    const slice1d<vec3> obstacles_positions,
    const slice1d<vec3> obstacles_scales)
{
    float y = halflife_to_damping(halflife) / 2.0f;
    vec3 j0 = velocity - desired_velocity;
    vec3 j1 = acceleration + j0*y;
    float eydt = fast_negexpf(y*dt);

    vec3 position_prev = position;

    position = eydt*(((-j1)/(y*y)) + ((-j0 - j1*dt)/y)) +
        (j1/(y*y)) + j0/y + desired_velocity * dt + position_prev;
    velocity = eydt*(j0 + j1*dt) + desired_velocity;
    acceleration = eydt*(acceleration - j1*y*dt);

    position = simulation_collide_obstacles(
        position_prev,
        position,
        obstacles_positions,
        obstacles_scales);
}

void simulation_rotations_update(
    quat& rotation,
    vec3& angular_velocity,
    const quat desired_rotation,
    const float halflife,
    const float dt)
{
    simple_spring_damper_exact(
        rotation,
        angular_velocity,
        desired_rotation,
        halflife, dt);
}
