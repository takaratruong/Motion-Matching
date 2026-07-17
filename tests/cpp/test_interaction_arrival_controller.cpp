#include "interaction_arrival.h"
#include "locomotion_controller_update.h"

#include "array.h"
#include "quat.h"
#include "vec.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace {

constexpr float kDt = 0.04F;
constexpr float kSimulationVelocityHalflife = 0.27F;
constexpr float kSimulationRotationHalflife = 0.27F;
constexpr int kMaximumTicks = 250;
constexpr int kMaximumCarryStagingTicks = 150;
constexpr float kPlaceStagingRootToleranceM = 0.25F;
constexpr float kPlaceStagingSlowRadiusM = 0.75F;
constexpr float kPlaceStagingMinimumInput = 0.20F;
constexpr float kAutodemoStagingDistanceM = 1.202609658F;
constexpr float kAutodemoStagingYawRadians = 0.343F * PIf / 180.0F;

static_assert(kDt == 1.0F / 25.0F);

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

float planar_length(vec3 value) {
    return std::hypot(value.x, value.z);
}

float planar_yaw_error(quat rotation, quat target) {
    const vec3 rotation_forward =
        quat_mul_vec3(rotation, vec3(0.0F, 0.0F, 1.0F));
    const vec3 target_forward =
        quat_mul_vec3(target, vec3(0.0F, 0.0F, 1.0F));
    const float rotation_yaw =
        std::atan2(rotation_forward.x, rotation_forward.z);
    const float target_yaw =
        std::atan2(target_forward.x, target_forward.z);
    return std::abs(std::atan2(
        std::sin(rotation_yaw - target_yaw),
        std::cos(rotation_yaw - target_yaw)));
}

vec3 place_staging_stick(
    vec3 target,
    vec3 displayed_root,
    float camera_azimuth) {
    vec3 root_error_world = target - displayed_root;
    root_error_world.y = 0.0F;
    const float root_error_m = planar_length(root_error_world);
    if (root_error_m <= 0.02F) return vec3();

    const float position_weight = clampf(
        root_error_m / kPlaceStagingSlowRadiusM,
        kPlaceStagingMinimumInput,
        1.0F);
    const vec3 world_command =
        position_weight * normalize(root_error_world);
    const quat camera_control_basis = quat_from_angle_axis(
        camera_azimuth, vec3(0.0F, 1.0F, 0.0F));
    return quat_inv_mul_vec3(camera_control_basis, world_command);
}

vec3 displayed_root_with_clamped_lag(
    vec3 simulation_root,
    vec3 staging_origin,
    float lag_cap_m) {
    vec3 travel = simulation_root - staging_origin;
    travel.y = 0.0F;
    const float travel_m = planar_length(travel);
    if (travel_m <= 0.0F) return simulation_root;
    const float lag_m = std::min(travel_m, lag_cap_m);
    return simulation_root - lag_m * normalize(travel);
}

struct CarryStagingResult {
    bool ready = false;
    int command_ticks = 0;
    float final_displayed_error_m = 0.0F;
};

CarryStagingResult simulate_autodemo_carry_staging(
    float outer_input_scale,
    float displayed_lag_cap_m) {
    const vec3 staging_origin;
    const vec3 staging_root(0.0F, 0.0F, kAutodemoStagingDistanceM);
    vec3 simulation_root = staging_origin;
    vec3 simulation_velocity;
    vec3 simulation_acceleration;
    const quat simulation_rotation = quat_from_angle_axis(
        kAutodemoStagingYawRadians, vec3(0.0F, 1.0F, 0.0F));
    array1d<vec3> obstacles_positions;
    array1d<vec3> obstacles_scales;

    CarryStagingResult result;
    for (int command_tick = 0;
         command_tick <= kMaximumCarryStagingTicks;
         ++command_tick) {
        const vec3 displayed_root = displayed_root_with_clamped_lag(
            simulation_root, staging_origin, displayed_lag_cap_m);
        result.final_displayed_error_m =
            planar_length(staging_root - displayed_root);
        if (result.final_displayed_error_m <=
            kPlaceStagingRootToleranceM) {
            result.ready = true;
            result.command_ticks = command_tick;
            return result;
        }
        if (command_tick == kMaximumCarryStagingTicks) break;

        const vec3 gamepadstick_left = outer_input_scale *
            place_staging_stick(staging_root, displayed_root, 0.0F);
        const vec3 desired_velocity = desired_velocity_update(
            gamepadstick_left,
            0.0F,
            simulation_rotation,
            0.9F,
            0.6F,
            0.6F);
        simulation_positions_update(
            simulation_root,
            simulation_velocity,
            simulation_acceleration,
            desired_velocity,
            kSimulationVelocityHalflife,
            kDt,
            obstacles_positions,
            obstacles_scales);
    }
    result.command_ticks = kMaximumCarryStagingTicks;
    return result;
}

void test_residual_velocity_converges_without_early_latch() {
    const interaction::ArrivalConfig config{};
    const vec3 target(0.0F, 0.0F, 0.0F);
    vec3 position(0.262F, 0.0F, 0.0F);
    vec3 velocity(0.67F, 0.0F, 0.0F);
    vec3 acceleration;
    const quat rotation;
    array1d<vec3> obstacles_positions;
    array1d<vec3> obstacles_scales;

    require(
        !interaction::arrival_ready(
            planar_length(position - target),
            planar_length(velocity),
            0.0F,
            0.40F,
            config),
        "residual-velocity case latched at its initial state");

    bool ready = false;
    bool saw_nonzero_correction = false;
    for (int tick = 0; tick < kMaximumTicks; ++tick) {
        const float position_error = planar_length(position - target);
        const float speed = planar_length(velocity);
        ready = interaction::arrival_ready(
            position_error, speed, 0.0F, 0.40F, config);
        if (ready) {
            require(
                position_error <= config.latch_position_error_m,
                "arrival latched before the position bound");
            require(
                speed <= config.latch_simulation_speed_mps,
                "arrival latched before the simulation-speed bound");
            break;
        }

        const vec3 navigation_stick =
            interaction::arrival_navigation_stick(
                target, position, 0.0F, config);
        require(
            planar_length(navigation_stick) > 0.0F,
            "arrival correction became zero before readiness");
        saw_nonzero_correction = true;
        const vec3 desired_velocity = desired_velocity_update(
            navigation_stick,
            0.0F,
            rotation,
            0.9F,
            0.6F,
            0.6F);
        simulation_positions_update(
            position,
            velocity,
            acceleration,
            desired_velocity,
            kSimulationVelocityHalflife,
            kDt,
            obstacles_positions,
            obstacles_scales);
    }

    require(saw_nonzero_correction, "arrival never issued a correction");
    require(ready, "residual-velocity arrival exceeded 250 ticks");
    require(
        planar_length(position - target) <= config.latch_position_error_m,
        "residual-velocity arrival missed the final position bound");
    require(
        planar_length(velocity) <= config.latch_simulation_speed_mps,
        "residual-velocity arrival missed the final speed bound");
}

void test_right_stick_strafe_seam_corrects_adjacent_yaw() {
    const interaction::ArrivalConfig config{};
    const float initial_yaw = std::nextafter(
        config.maximum_yaw_error_radians,
        std::numeric_limits<float>::infinity());
    quat rotation = quat_from_angle_axis(
        initial_yaw, vec3(0.0F, 1.0F, 0.0F));
    vec3 angular_velocity;
    const quat target_rotation;
    const vec3 target_forward(0.0F, 0.0F, 1.0F);
    const vec3 zero_stick;

    require(
        !interaction::arrival_ready(
            0.0F, 0.0F, initial_yaw, 0.40F, config),
        "one-ULP-above yaw latched before facing correction");

    bool ready = false;
    bool saw_facing_authority = false;
    for (int tick = 0; tick < kMaximumTicks; ++tick) {
        const vec3 facing_stick =
            interaction::arrival_facing_stick(target_forward, 0.0F);
        require(
            planar_length(facing_stick) > 0.01F,
            "arrival facing stick lost authority at zero position error");

        const quat without_strafe = desired_rotation_update(
            rotation,
            zero_stick,
            facing_stick,
            0.0F,
            false,
            vec3());
        require(
            planar_yaw_error(without_strafe, rotation) <= 1.0e-6F,
            "right stick unexpectedly controlled facing without strafe");

        const quat desired_rotation = desired_rotation_update(
            rotation,
            zero_stick,
            facing_stick,
            0.0F,
            true,
            vec3());
        require(
            planar_yaw_error(desired_rotation, target_rotation) <= 1.0e-6F,
            "right-stick/strafe seam did not target waypoint facing");
        saw_facing_authority = true;

        simulation_rotations_update(
            rotation,
            angular_velocity,
            desired_rotation,
            kSimulationRotationHalflife,
            kDt);
        const float yaw_error =
            planar_yaw_error(rotation, target_rotation);
        ready = interaction::arrival_ready(
            0.0F, 0.0F, yaw_error, 0.40F, config);
        if (ready) {
            require(
                yaw_error <= config.maximum_yaw_error_radians,
                "yaw arrival latched above its inclusive bound");
            break;
        }
    }

    require(saw_facing_authority, "yaw case never exercised facing authority");
    require(ready, "one-ULP-above yaw arrival exceeded 250 ticks");
}

void test_autodemo_carry_staging_reaches_ready_preview_by_150_ticks() {
    for (const float displayed_lag_cap_m : {0.10F, 0.15F}) {
        const CarryStagingResult quarter_scale =
            simulate_autodemo_carry_staging(0.25F, displayed_lag_cap_m);
        require(
            !quarter_scale.ready,
            "quarter-scale staging unexpectedly met the 150-tick bound");

        const CarryStagingResult full_scale =
            simulate_autodemo_carry_staging(1.0F, displayed_lag_cap_m);
        require(
            full_scale.ready,
            "full-scale staging missed the 150-tick bound");
        require(
            full_scale.command_ticks >= 25 &&
                full_scale.command_ticks <= kMaximumCarryStagingTicks,
            "full-scale staging violated the evidence tick envelope");
        require(
            full_scale.final_displayed_error_m <=
                kPlaceStagingRootToleranceM,
            "full-scale staging became ready outside the root tolerance");
    }
}

}  // namespace

int main() {
    try {
        test_residual_velocity_converges_without_early_latch();
        test_right_stick_strafe_seam_corrects_adjacent_yaw();
        test_autodemo_carry_staging_reaches_ready_preview_by_150_ticks();
        std::puts("interaction_arrival_controller tests passed");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "interaction_arrival_controller FAILED: %s\n",
            error.what());
        return EXIT_FAILURE;
    }
}
