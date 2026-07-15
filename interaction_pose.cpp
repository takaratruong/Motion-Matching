#include "interaction_pose.h"
#include "interaction_rotation_gate.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace interaction {
namespace {

constexpr size_t kVectorComponents = 3U;
constexpr size_t kQuaternionComponents = 4U;

size_t frame_index(const Database& database, int32_t frame) {
    if (frame < 0 || static_cast<uint32_t>(frame) >= database.frame_count) {
        throw std::out_of_range("interaction pose frame outside database");
    }
    return static_cast<size_t>(frame);
}

vec3 read_vec3(
    const std::vector<float>& values,
    size_t frame,
    size_t bone) {
    const size_t offset =
        (frame * g1_skeleton::BoneCount + bone) * kVectorComponents;
    return vec3(
        values.at(offset), values.at(offset + 1U), values.at(offset + 2U));
}

quat read_quat(
    const std::vector<float>& values,
    size_t frame,
    size_t bone) {
    const size_t offset =
        (frame * g1_skeleton::BoneCount + bone) * kQuaternionComponents;
    return quat(
        values.at(offset), values.at(offset + 1U), values.at(offset + 2U),
        values.at(offset + 3U));
}

}  // namespace

Transform compose(const Transform& parent, const Transform& local) {
    return {
        parent.position + quat_mul_vec3(parent.rotation, local.position),
        quat_normalize(quat_mul(parent.rotation, local.rotation)),
    };
}

Transform inverse(const Transform& value) {
    const quat rotation = quat_inv(value.rotation);
    return {quat_mul_vec3(rotation, -value.position), rotation};
}

Pose pose_at_frame(const Database& database, int32_t frame) {
    const size_t source_frame = frame_index(database, frame);
    Pose pose{};
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        pose.positions[bone] =
            read_vec3(database.positions, source_frame, bone);
        pose.velocities[bone] =
            read_vec3(database.velocities, source_frame, bone);
        pose.rotations[bone] =
            read_quat(database.rotations, source_frame, bone);
        pose.angular_velocities[bone] =
            read_vec3(database.angular_velocities, source_frame, bone);
    }
    for (size_t dof = 0; dof < pose.hand_dof.size(); ++dof) {
        const size_t offset = source_frame * pose.hand_dof.size() + dof;
        pose.hand_dof[dof] = database.hand_dof.at(offset);
        pose.hand_dof_velocities[dof] =
            database.hand_dof_velocities.at(offset);
    }
    for (size_t foot = 0; foot < pose.foot_contacts.size(); ++foot) {
        const size_t offset = source_frame * pose.foot_contacts.size() + foot;
        pose.foot_contacts[foot] = database.foot_contacts.at(offset);
    }
    return pose;
}

Pose interpolate_pose(const Pose& left, const Pose& right, float alpha) {
    Pose pose{};
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        pose.positions[bone] =
            lerp(left.positions[bone], right.positions[bone], alpha);
        pose.velocities[bone] =
            lerp(left.velocities[bone], right.velocities[bone], alpha);
        pose.rotations[bone] = quat_nlerp_shortest(
            left.rotations[bone], right.rotations[bone], alpha);
        pose.angular_velocities[bone] = lerp(
            left.angular_velocities[bone],
            right.angular_velocities[bone],
            alpha);
    }
    for (size_t dof = 0; dof < pose.hand_dof.size(); ++dof) {
        pose.hand_dof[dof] =
            lerpf(left.hand_dof[dof], right.hand_dof[dof], alpha);
        pose.hand_dof_velocities[dof] = lerpf(
            left.hand_dof_velocities[dof],
            right.hand_dof_velocities[dof],
            alpha);
    }
    pose.foot_contacts = alpha < 0.5F
        ? left.foot_contacts
        : right.foot_contacts;
    return pose;
}

Pose sample_pose(
    const Database& database,
    uint32_t clip,
    float seconds_from_entry) {
    const int32_t start = database.range_starts.at(clip);
    const int32_t stop = database.range_stops.at(clip);
    const float frame = seconds_from_entry * 25.0F + start;
    if (frame < start || frame > static_cast<float>(stop - 1)) {
        throw std::out_of_range("interaction sample outside clip range");
    }
    const int32_t left = static_cast<int32_t>(std::floor(frame));
    const int32_t right = std::min(left + 1, stop - 1);
    return interpolate_pose(
        pose_at_frame(database, left),
        pose_at_frame(database, right),
        frame - static_cast<float>(left));
}

WorldPose world_pose(const Pose& pose) {
    WorldPose world{};
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        const int32_t parent = g1_skeleton::kParents[bone];
        if (parent < 0) {
            world.positions[bone] = pose.positions[bone];
            world.velocities[bone] = pose.velocities[bone];
            world.rotations[bone] = pose.rotations[bone];
            world.angular_velocities[bone] = pose.angular_velocities[bone];
            continue;
        }

        const size_t parent_bone = static_cast<size_t>(parent);
        const vec3 offset = quat_mul_vec3(
            world.rotations[parent_bone], pose.positions[bone]);
        world.positions[bone] = world.positions[parent_bone] + offset;
        world.rotations[bone] = quat_normalize(quat_mul(
            world.rotations[parent_bone], pose.rotations[bone]));
        world.velocities[bone] =
            world.velocities[parent_bone] +
            cross(world.angular_velocities[parent_bone], offset) +
            quat_mul_vec3(
                world.rotations[parent_bone], pose.velocities[bone]);
        world.angular_velocities[bone] =
            world.angular_velocities[parent_bone] +
            quat_mul_vec3(
                world.rotations[parent_bone],
                pose.angular_velocities[bone]);
    }
    return world;
}

rotation_gate::Rotation world_rotation_evidence(
    const Pose& pose,
    size_t bone) {
    if (bone >= g1_skeleton::BoneCount) {
        throw std::out_of_range("interaction pose bone index out of range");
    }
    std::array<size_t, g1_skeleton::BoneCount> chain{};
    size_t count = 0U;
    size_t current = bone;
    while (true) {
        chain[count++] = current;
        const int32_t parent = g1_skeleton::kParents[current];
        if (parent < 0) break;
        current = static_cast<size_t>(parent);
    }

    rotation_gate::Rotation rotation = rotation_gate::from_quat(
        pose.rotations[chain[count - 1U]]);
    while (count > 1U) {
        --count;
        rotation = rotation_gate::multiply(
            rotation,
            rotation_gate::from_quat(
                pose.rotations[chain[count - 1U]]));
    }
    return rotation;
}

rotation_gate::Rotation world_rotation_evidence(
    const Pose& pose,
    size_t bone,
    const rotation_gate::Rotation& root_rotation_evidence) {
    if (bone >= g1_skeleton::BoneCount) {
        throw std::out_of_range("interaction pose bone index out of range");
    }
    std::array<size_t, g1_skeleton::BoneCount> chain{};
    size_t count = 0U;
    size_t current = bone;
    while (true) {
        chain[count++] = current;
        const int32_t parent = g1_skeleton::kParents[current];
        if (parent < 0) break;
        current = static_cast<size_t>(parent);
    }

    rotation_gate::Rotation rotation = root_rotation_evidence;
    while (count > 1U) {
        --count;
        rotation = rotation_gate::multiply(
            rotation,
            rotation_gate::from_quat(
                pose.rotations[chain[count - 1U]]));
    }
    return rotation;
}

}  // namespace interaction
