#include "interaction_funnel_capture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace interaction {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr std::array<double, 16> kCandidateOffsetDegrees{
    0.0,
    22.5, -22.5,
    45.0, -45.0,
    67.5, -67.5,
    90.0, -90.0,
    112.5, -112.5,
    135.0, -135.0,
    157.5, -157.5,
    180.0,
};

bool finite_transform(const Transform& value) {
    const float values[]{
        value.position.x, value.position.y, value.position.z,
        value.rotation.w, value.rotation.x,
        value.rotation.y, value.rotation.z,
    };
    for (float component : values) {
        if (!std::isfinite(component)) return false;
    }
    return true;
}

}  // namespace

FunnelCaptureSelection select_funnel_capture(
    Transform live_root,
    const InteractionTarget& target,
    const std::vector<PickNavigationObstacle>& obstacles,
    const FunnelCaptureConfig& config) {
    FunnelCaptureSelection selection{};
    if (!finite_transform(live_root) ||
        !finite_transform(target.object_world) ||
        !std::isfinite(config.minimum_object_radius_m) ||
        !std::isfinite(config.maximum_object_radius_m) ||
        !(config.minimum_object_radius_m > 0.0F) ||
        config.maximum_object_radius_m < config.minimum_object_radius_m) {
        return selection;
    }

    const double bearing_x =
        static_cast<double>(live_root.position.x) -
        static_cast<double>(target.object_world.position.x);
    const double bearing_z =
        static_cast<double>(live_root.position.z) -
        static_cast<double>(target.object_world.position.z);
    const double bearing_length = std::sqrt(
        bearing_x * bearing_x + bearing_z * bearing_z);
    if (!(bearing_length > std::numeric_limits<double>::epsilon())) {
        return selection;
    }
    const double unit_x = bearing_x / bearing_length;
    const double unit_z = bearing_z / bearing_length;
    const double radius = std::clamp(
        bearing_length,
        static_cast<double>(config.minimum_object_radius_m),
        static_cast<double>(config.maximum_object_radius_m));

    bool saw_blocked = false;
    bool saw_outside = false;
    for (size_t index = 0U; index < kCandidateOffsetDegrees.size(); ++index) {
        const double angle = kCandidateOffsetDegrees[index] * kPi / 180.0;
        const double sine = std::sin(angle);
        const double cosine = std::cos(angle);
        const double direction_x = cosine * unit_x + sine * unit_z;
        const double direction_z = -sine * unit_x + cosine * unit_z;
        const float world_x = static_cast<float>(
            static_cast<double>(target.object_world.position.x) +
            radius * direction_x);
        const float world_z = static_cast<float>(
            static_cast<double>(target.object_world.position.z) +
            radius * direction_z);
        const float facing_yaw = std::atan2(
            target.object_world.position.x - world_x,
            target.object_world.position.z - world_z);
        const Transform candidate{
            vec3(world_x, live_root.position.y, world_z),
            quat_from_angle_axis(
                facing_yaw, vec3(0.0F, 1.0F, 0.0F)),
        };
        const PickSlotReason reason = revalidate_frozen_pick_slot(
            live_root, candidate, target, obstacles, config.route);
        if (reason == PickSlotReason::None) {
            selection.target_world = candidate;
            selection.candidate_index = static_cast<int>(index);
            selection.reason = PickSlotReason::None;
            return selection;
        }
        saw_blocked = saw_blocked ||
            reason == PickSlotReason::TableBlocked ||
            reason == PickSlotReason::ObstacleBlocked ||
            reason == PickSlotReason::AllSlotsBlocked;
        saw_outside = saw_outside ||
            reason == PickSlotReason::OutsideTravelEnvelope;
    }
    selection.reason = saw_blocked
        ? PickSlotReason::AllSlotsBlocked
        : (saw_outside
               ? PickSlotReason::OutsideTravelEnvelope
               : PickSlotReason::InvalidGeometry);
    return selection;
}

std::optional<std::array<float, 24>> build_funnel_condition(
    Transform entry_root_world,
    vec3 simulation_velocity_world,
    const InteractionTarget& target,
    const GraspAffordance& affordance) {
    if (!finite_transform(entry_root_world) ||
        !finite_transform(target.object_world) ||
        !finite_transform(affordance.hand_in_object) ||
        !std::isfinite(simulation_velocity_world.x) ||
        !std::isfinite(simulation_velocity_world.z) ||
        !std::isfinite(target.object_dimensions.x) ||
        !std::isfinite(target.object_dimensions.y) ||
        !std::isfinite(target.object_dimensions.z) ||
        !std::isfinite(target.table_world.position.y)) {
        return std::nullopt;
    }
    const double approach_x = affordance.approach_direction_object.x;
    const double approach_z = affordance.approach_direction_object.z;
    const double approach_norm = std::sqrt(
        approach_x * approach_x + approach_z * approach_z);
    if (!(approach_norm > 1.0e-8)) return std::nullopt;

    const vec3 object_forward_raw = quat_mul_vec3(
        target.object_world.rotation, vec3(0.0F, 0.0F, 1.0F));
    const vec3 root_forward_raw = quat_mul_vec3(
        entry_root_world.rotation, vec3(0.0F, 0.0F, 1.0F));
    const double object_length = std::sqrt(
        static_cast<double>(object_forward_raw.x) * object_forward_raw.x +
        static_cast<double>(object_forward_raw.z) * object_forward_raw.z);
    const double root_length = std::sqrt(
        static_cast<double>(root_forward_raw.x) * root_forward_raw.x +
        static_cast<double>(root_forward_raw.z) * root_forward_raw.z);
    if (!(object_length > 0.0) || !(root_length > 0.0)) return std::nullopt;
    const double object_sin = object_forward_raw.x / object_length;
    const double object_cos = object_forward_raw.z / object_length;
    const double root_sin = root_forward_raw.x / root_length;
    const double root_cos = root_forward_raw.z / root_length;

    const quat grasp = affordance.hand_in_object.rotation;
    std::array<float, 24> condition{};
    condition[0] = affordance.hand == Hand::Left ? 1.0F : 0.0F;
    condition[1] = affordance.hand == Hand::Right ? 1.0F : 0.0F;
    condition[2] = affordance.hand_in_object.position.x;
    condition[3] = affordance.hand_in_object.position.y;
    condition[4] = affordance.hand_in_object.position.z;
    condition[5] = 1.0F - 2.0F * (grasp.y * grasp.y + grasp.z * grasp.z);
    condition[6] = 2.0F * (grasp.x * grasp.y + grasp.z * grasp.w);
    condition[7] = 2.0F * (grasp.x * grasp.z - grasp.y * grasp.w);
    condition[8] = 2.0F * (grasp.x * grasp.y - grasp.z * grasp.w);
    condition[9] = 1.0F - 2.0F * (grasp.x * grasp.x + grasp.z * grasp.z);
    condition[10] = 2.0F * (grasp.y * grasp.z + grasp.x * grasp.w);
    condition[11] = static_cast<float>(approach_x / approach_norm);
    condition[12] = static_cast<float>(approach_z / approach_norm);
    condition[13] = target.object_dimensions.x;
    condition[14] = target.object_dimensions.y;
    condition[15] = target.object_dimensions.z;
    condition[16] =
        target.object_world.position.y - target.table_world.position.y;
    condition[17] = affordance.hand_in_object.position.y;

    const double delta_x =
        static_cast<double>(entry_root_world.position.x) -
        target.object_world.position.x;
    const double delta_z =
        static_cast<double>(entry_root_world.position.z) -
        target.object_world.position.z;
    condition[18] = static_cast<float>(object_cos * delta_x + object_sin * delta_z);
    condition[19] = static_cast<float>(-object_sin * delta_x + object_cos * delta_z);
    condition[20] = static_cast<float>(root_sin * object_cos - root_cos * object_sin);
    condition[21] = static_cast<float>(root_cos * object_cos + root_sin * object_sin);
    condition[22] = static_cast<float>(
        object_cos * simulation_velocity_world.x +
        object_sin * simulation_velocity_world.z);
    condition[23] = static_cast<float>(
        -object_sin * simulation_velocity_world.x +
        object_cos * simulation_velocity_world.z);
    for (float value : condition) {
        if (!std::isfinite(value)) return std::nullopt;
    }
    return condition;
}

}  // namespace interaction
