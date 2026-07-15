#pragma once

#include "g1_skeleton.h"
#include "interaction_database.h"
#include "quat.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace interaction {

struct Transform {
    vec3 position;
    quat rotation;
};

struct Pose {
    std::array<vec3, g1_skeleton::BoneCount> positions{};
    std::array<vec3, g1_skeleton::BoneCount> velocities{};
    std::array<quat, g1_skeleton::BoneCount> rotations{};
    std::array<vec3, g1_skeleton::BoneCount> angular_velocities{};
    std::array<float, 14> hand_dof{};
    std::array<float, 14> hand_dof_velocities{};
    std::array<uint8_t, 2> foot_contacts{};
};

struct WorldPose {
    std::array<vec3, g1_skeleton::BoneCount> positions{};
    std::array<vec3, g1_skeleton::BoneCount> velocities{};
    std::array<quat, g1_skeleton::BoneCount> rotations{};
    std::array<vec3, g1_skeleton::BoneCount> angular_velocities{};
};

struct LocomotionSnapshot {
    Pose pose;
    std::array<vec3, 3> future_root_positions{};
    std::array<quat, 3> future_root_rotations{};
};

Transform compose(const Transform& parent, const Transform& local);
Transform inverse(const Transform& value);
Pose pose_at_frame(const Database& database, int32_t frame);
Pose interpolate_pose(const Pose& left, const Pose& right, float alpha);
Pose sample_pose(
    const Database& database,
    uint32_t clip,
    float seconds_from_entry);
WorldPose world_pose(const Pose& pose);
quat raw_world_rotation(const Pose& pose, size_t bone);

}  // namespace interaction
