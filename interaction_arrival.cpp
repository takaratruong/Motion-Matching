#include "interaction_arrival.h"

#include "quat.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace interaction {
namespace {

bool finite(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

bool finite(vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

void validate_config(const ArrivalConfig& config) {
    const bool positive_limits =
        finite(config.slow_radius_m) && config.slow_radius_m > 0.0F &&
        finite(config.latch_position_error_m) &&
        config.latch_position_error_m > 0.0F &&
        finite(config.latch_simulation_speed_mps) &&
        config.latch_simulation_speed_mps > 0.0F &&
        finite(config.maximum_yaw_error_radians) &&
        config.maximum_yaw_error_radians > 0.0F &&
        finite(config.minimum_standoff_m) &&
        config.minimum_standoff_m > 0.0F &&
        finite(config.maximum_standoff_m) &&
        config.maximum_standoff_m > 0.0F;
    if (!positive_limits ||
        !(config.maximum_standoff_m > config.minimum_standoff_m)) {
        throw std::invalid_argument("interaction arrival invalid config");
    }
}

void validate_camera(float camera_azimuth) {
    if (!finite(camera_azimuth)) {
        throw std::invalid_argument(
            "interaction arrival invalid camera azimuth");
    }
}

vec3 inverse_camera_yaw(vec3 world_direction, float camera_azimuth) {
    const quat camera_control_basis = quat_from_angle_axis(
        camera_azimuth, vec3(0.0F, 1.0F, 0.0F));
    return quat_inv_mul_vec3(camera_control_basis, world_direction);
}

}  // namespace

vec3 arrival_navigation_stick(
    vec3 target_world,
    vec3 current_world,
    float camera_azimuth,
    const ArrivalConfig& config) {
    validate_config(config);
    if (!finite(target_world) || !finite(current_world)) {
        throw std::invalid_argument(
            "interaction arrival invalid navigation vector");
    }
    validate_camera(camera_azimuth);

    const double delta_x = static_cast<double>(target_world.x) -
        static_cast<double>(current_world.x);
    const double delta_z = static_cast<double>(target_world.z) -
        static_cast<double>(current_world.z);
    if (delta_x == 0.0 && delta_z == 0.0) return vec3();

    const double planar_distance = std::sqrt(
        delta_x * delta_x + delta_z * delta_z);
    const float magnitude = clampf(
        static_cast<float>(
            planar_distance / static_cast<double>(config.slow_radius_m)),
        0.0F,
        1.0F);
    const double scale = static_cast<double>(magnitude) / planar_distance;
    const vec3 world_command(
        static_cast<float>(delta_x * scale),
        0.0F,
        static_cast<float>(delta_z * scale));
    return inverse_camera_yaw(world_command, camera_azimuth);
}

vec3 arrival_facing_stick(
    vec3 target_forward_world,
    float camera_azimuth) {
    if (!finite(target_forward_world)) {
        throw std::invalid_argument(
            "interaction arrival invalid facing vector");
    }
    validate_camera(camera_azimuth);

    const double planar_x = static_cast<double>(target_forward_world.x);
    const double planar_z = static_cast<double>(target_forward_world.z);
    if (planar_x == 0.0 && planar_z == 0.0) {
        throw std::invalid_argument(
            "interaction arrival facing vector must be nonzero");
    }
    const double planar_length = std::sqrt(
        planar_x * planar_x + planar_z * planar_z);
    const double inverse_length = 1.0 / planar_length;
    const vec3 world_direction(
        static_cast<float>(planar_x * inverse_length),
        0.0F,
        static_cast<float>(planar_z * inverse_length));
    return inverse_camera_yaw(world_direction, camera_azimuth);
}

bool arrival_ready(
    float position_error_m,
    float planar_simulation_speed_mps,
    float yaw_error_radians,
    float standoff_m,
    const ArrivalConfig& config) {
    validate_config(config);
    if (!finite(position_error_m) || position_error_m < 0.0F ||
        !finite(planar_simulation_speed_mps) ||
        planar_simulation_speed_mps < 0.0F ||
        !finite(yaw_error_radians) || yaw_error_radians < 0.0F ||
        !finite(standoff_m) || standoff_m < 0.0F) {
        throw std::invalid_argument(
            "interaction arrival invalid readiness metric");
    }

    return position_error_m <= config.latch_position_error_m &&
        planar_simulation_speed_mps <=
            config.latch_simulation_speed_mps &&
        yaw_error_radians <= config.maximum_yaw_error_radians &&
        standoff_m >= config.minimum_standoff_m &&
        standoff_m <= config.maximum_standoff_m;
}

}  // namespace interaction
