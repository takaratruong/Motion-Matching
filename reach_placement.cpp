#include "reach_placement.h"

#include <cmath>
#include <stdexcept>

namespace reach {
namespace {

constexpr float kTwoPi = 6.28318530717958647692F;

bool finite(vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

quat placement_rotation(uint8_t yaw_index) {
    return quat_from_angle_axis(
        placement_yaw(yaw_index), vec3(0.0F, 1.0F, 0.0F));
}

}  // namespace

float placement_yaw(uint8_t yaw_index) {
    if (yaw_index >= kYawPlacementCount) {
        throw std::out_of_range("reach yaw placement index outside range");
    }
    return kTwoPi * static_cast<float>(yaw_index) /
           static_cast<float>(kYawPlacementCount);
}

interaction::Transform contact_alignment(
    const Pack& pack,
    size_t clip,
    uint8_t yaw_index,
    vec3 target_position) {
    if (!finite(target_position)) {
        throw std::invalid_argument("reach placement target is non-finite");
    }
    const quat rotation = placement_rotation(yaw_index);
    const vec3 endpoint = endpoint_transform(pack.database, clip).position;
    return {
        target_position - quat_mul_vec3(rotation, endpoint),
        rotation,
    };
}

interaction::Pose place_pose(
    const Pack& pack,
    size_t clip,
    uint8_t yaw_index,
    vec3 target_position,
    int32_t database_frame) {
    interaction::Pose pose = pose_at_frame(pack.database, database_frame);
    const interaction::Transform alignment = contact_alignment(
        pack, clip, yaw_index, target_position);
    const size_t root = static_cast<size_t>(g1_skeleton::Simulation);
    const interaction::Transform placed_root = interaction::compose(
        alignment, {pose.positions[root], pose.rotations[root]});
    pose.positions[root] = placed_root.position;
    pose.rotations[root] = placed_root.rotation;
    return pose;
}

vec3 place_direction(vec3 direction, uint8_t yaw_index) {
    if (!finite(direction)) {
        throw std::invalid_argument("reach placement direction is non-finite");
    }
    return quat_mul_vec3(placement_rotation(yaw_index), direction);
}

}  // namespace reach
