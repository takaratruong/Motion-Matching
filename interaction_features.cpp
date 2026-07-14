#include "interaction_features.h"

#include <array>
#include <cassert>
#include <cstddef>

namespace interaction {
namespace {

void append(RawQuery& query, size_t& dimension, vec3 value) {
    query[dimension++] = value.x;
    query[dimension++] = value.y;
    query[dimension++] = value.z;
}

}  // namespace

RawQuery build_raw_query(const QueryInput& input) {
    const WorldPose world = world_pose(input.locomotion.pose);
    const vec3 root_position = world.positions[g1_skeleton::Simulation];
    const quat root_rotation = world.rotations[g1_skeleton::Simulation];
    const quat inverse_root = quat_inv(root_rotation);
    const quat inverse_grasp = quat_inv(input.grasp_world.rotation);
    const size_t hand_bone =
        input.hand == Hand::Left ? kLeftHandBone : kRightHandBone;
    const std::array<size_t, 5> pose_bones = {
        g1_skeleton::LeftToe,
        g1_skeleton::RightToe,
        g1_skeleton::Hips,
        g1_skeleton::Spine2,
        hand_bone,
    };

    RawQuery query{};
    size_t dimension = 0;
    for (size_t bone : pose_bones) {
        append(
            query,
            dimension,
            quat_mul_vec3(
                inverse_root, world.positions[bone] - root_position));
    }
    for (size_t bone : pose_bones) {
        append(
            query,
            dimension,
            quat_mul_vec3(inverse_root, world.velocities[bone]));
    }

    const vec3 root_velocity = quat_mul_vec3(
        inverse_root, world.velocities[g1_skeleton::Simulation]);
    query[dimension++] = root_velocity.x;
    query[dimension++] = root_velocity.z;
    query[dimension++] =
        world.angular_velocities[g1_skeleton::Simulation].y;

    for (const vec3 future_root : input.locomotion.future_root_positions) {
        const vec3 delta = quat_mul_vec3(
            inverse_root, future_root - root_position);
        query[dimension++] = delta.x;
        query[dimension++] = delta.z;
    }
    for (const quat future_rotation : input.locomotion.future_root_rotations) {
        const vec3 facing_world = quat_mul_vec3(
            future_rotation, vec3(0.0F, 0.0F, 1.0F));
        const vec3 facing_local = quat_mul_vec3(inverse_root, facing_world);
        query[dimension++] = facing_local.x;
        query[dimension++] = facing_local.z;
    }

    const vec3 hand_position = world.positions[hand_bone];
    const quat hand_rotation = world.rotations[hand_bone];
    append(
        query,
        dimension,
        quat_mul_vec3(
            inverse_grasp, hand_position - input.grasp_world.position));
    append(
        query,
        dimension,
        quat_to_scaled_angle_axis(
            quat_abs(quat_mul(inverse_grasp, hand_rotation)), 1.0e-5F));
    append(
        query,
        dimension,
        quat_mul_vec3(
            inverse_grasp,
            world.velocities[hand_bone] - input.grasp_linear_velocity));
    append(
        query,
        dimension,
        quat_mul_vec3(
            inverse_grasp,
            world.angular_velocities[hand_bone] -
                input.grasp_angular_velocity));

    append(
        query,
        dimension,
        quat_mul_vec3(
            inverse_grasp, root_position - input.grasp_world.position));
    const vec3 root_facing_world = quat_mul_vec3(
        root_rotation, vec3(0.0F, 0.0F, 1.0F));
    const vec3 root_facing_grasp =
        quat_mul_vec3(inverse_grasp, root_facing_world);
    query[dimension++] = root_facing_grasp.x;
    query[dimension++] = root_facing_grasp.z;
    append(
        query,
        dimension,
        quat_mul_vec3(
            inverse_grasp,
            world.velocities[g1_skeleton::Simulation] -
                input.grasp_linear_velocity));

    const float table_top =
        input.table_world.position.y + 0.5F * input.table_size.y;
    query[dimension++] = input.grasp_world.position.y - table_top;
    query[dimension++] = input.approach_direction_object.x;
    query[dimension++] = input.approach_direction_object.z;
    append(query, dimension, input.object_dimensions);

    assert(dimension == query.size());
    return query;
}

NormalizedQuery normalize_query(
    const RawQuery& raw,
    const Features& features) {
    NormalizedQuery normalized{};
    for (size_t dimension = 0; dimension < normalized.size(); ++dimension) {
        normalized[dimension] =
            (raw[dimension] - features.offsets.at(dimension)) /
            features.scales.at(dimension);
    }
    return normalized;
}

}  // namespace interaction
