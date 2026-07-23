#pragma once

#include "interaction_posture_ik.h"

#include <cassert>
#include <cmath>
#include <cstddef>

namespace g1_posture_fixture {

constexpr float kTolerance = 2.0e-5F;

inline bool near(
    float left,
    float right,
    float tolerance = kTolerance) {
    return std::abs(left - right) <= tolerance;
}

inline const interaction::HingeJoint& upper_body_metadata(size_t joint) {
    return joint < interaction::kWaist.size()
        ? interaction::kWaist[joint]
        : interaction::kLeftArm[joint - interaction::kWaist.size()];
}

inline interaction::Pose make_pose() {
    interaction::Pose pose{};
    pose.positions[g1_skeleton::Simulation] = vec3(0.20F, 0.0F, -0.10F);
    pose.rotations[g1_skeleton::Simulation] =
        quat_from_angle_axis(0.17F, vec3(0.0F, 1.0F, 0.0F));
    pose.positions[g1_skeleton::Hips] = vec3(0.0F, 0.82F, 0.0F);
    pose.positions[g1_skeleton::Spine] = vec3(0.0F, 0.10F, 0.0F);
    pose.positions[g1_skeleton::Spine1] = vec3(0.0F, 0.10F, 0.0F);
    pose.positions[g1_skeleton::Spine2] = vec3(0.0F, 0.10F, 0.0F);

    pose.positions[g1_skeleton::LeftShoulderPitch] =
        vec3(0.0039563F, 0.23778F, -0.10022F);
    pose.positions[g1_skeleton::LeftShoulderRoll] =
        vec3(0.0F, -0.013831F, -0.038F);
    pose.positions[g1_skeleton::LeftShoulderYaw] =
        vec3(0.0F, -0.1032F, -0.00624F);
    pose.positions[g1_skeleton::LeftElbow] =
        vec3(0.015783F, -0.080518F, 0.0F);
    pose.positions[g1_skeleton::LeftWristRoll] =
        vec3(0.10F, -0.01F, -0.00188791F);
    pose.positions[g1_skeleton::LeftWristPitch] =
        vec3(0.038F, 0.0F, 0.0F);
    pose.positions[g1_skeleton::LeftWrist] =
        vec3(0.046F, 0.0F, 0.0F);

    pose.positions[g1_skeleton::RightShoulderPitch] =
        vec3(0.0039563F, 0.23778F, 0.10021F);
    pose.positions[g1_skeleton::RightShoulderRoll] =
        vec3(0.0F, -0.013831F, 0.038F);
    pose.positions[g1_skeleton::RightShoulderYaw] =
        vec3(0.0F, -0.1032F, 0.00624F);
    pose.positions[g1_skeleton::RightElbow] =
        vec3(0.015783F, -0.080518F, 0.0F);
    pose.positions[g1_skeleton::RightWristRoll] =
        vec3(0.10F, -0.01F, 0.00188791F);
    pose.positions[g1_skeleton::RightWristPitch] =
        vec3(0.038F, 0.0F, 0.0F);
    pose.positions[g1_skeleton::RightWrist] =
        vec3(0.046F, 0.0F, 0.0F);

    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        const float value = static_cast<float>(bone);
        pose.velocities[bone] =
            vec3(value + 0.1F, value + 0.2F, value + 0.3F);
        pose.angular_velocities[bone] =
            vec3(value + 0.4F, value + 0.5F, value + 0.6F);
    }
    for (size_t dof = 0U; dof < pose.hand_dof.size(); ++dof) {
        pose.hand_dof[dof] = static_cast<float>(dof) + 0.25F;
        pose.hand_dof_velocities[dof] = static_cast<float>(dof) + 0.75F;
    }
    pose.foot_contacts = {1U, 0U};

    const interaction::LeftUpperBodyAngles angles{
        0.05F, -0.03F, 0.04F,
        0.10F, 0.35F, -0.20F, 0.80F, 0.10F, -0.15F, 0.05F};
    interaction::apply_left_upper_body(pose, angles);
    return pose;
}

inline interaction::Transform left_hand_transform(
    const interaction::Pose& pose) {
    const interaction::WorldPose world = interaction::world_pose(pose);
    return {
        world.positions[g1_skeleton::LeftWrist],
        world.rotations[g1_skeleton::LeftWrist],
    };
}

inline void assert_finite_bounded(
    const interaction::LeftUpperBodyAngles& angles) {
    for (size_t joint = 0U; joint < angles.size(); ++joint) {
        const interaction::HingeJoint& item = upper_body_metadata(joint);
        assert(std::isfinite(angles[joint]));
        assert(angles[joint] >= item.lower - kTolerance);
        assert(angles[joint] <= item.upper + kTolerance);
    }
}

inline bool owned_rotation(size_t bone) {
    for (size_t joint = 0U;
         joint < interaction::kLeftUpperBodyJointCount;
         ++joint) {
        if (static_cast<size_t>(upper_body_metadata(joint).bone) == bone) {
            return true;
        }
    }
    return false;
}

inline bool exact(vec3 left, vec3 right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

inline bool exact(quat left, quat right) {
    return left.w == right.w && left.x == right.x &&
           left.y == right.y && left.z == right.z;
}

inline void assert_non_owned_local_channels_equal(
    const interaction::Pose& before,
    const interaction::Pose& after) {
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        assert(exact(before.positions[bone], after.positions[bone]));
        assert(exact(before.velocities[bone], after.velocities[bone]));
        assert(exact(
            before.angular_velocities[bone],
            after.angular_velocities[bone]));
        if (!owned_rotation(bone)) {
            assert(exact(before.rotations[bone], after.rotations[bone]));
        }
    }
    assert(before.hand_dof == after.hand_dof);
    assert(before.hand_dof_velocities == after.hand_dof_velocities);
    assert(before.foot_contacts == after.foot_contacts);
}

inline bool same_pose(
    const interaction::Pose& left,
    const interaction::Pose& right) {
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        if (!exact(left.positions[bone], right.positions[bone]) ||
            !exact(left.velocities[bone], right.velocities[bone]) ||
            !exact(left.rotations[bone], right.rotations[bone]) ||
            !exact(
                left.angular_velocities[bone],
                right.angular_velocities[bone])) {
            return false;
        }
    }
    return left.hand_dof == right.hand_dof &&
           left.hand_dof_velocities == right.hand_dof_velocities &&
           left.foot_contacts == right.foot_contacts;
}

inline void assert_finite_pose(const interaction::Pose& pose) {
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        assert(std::isfinite(pose.positions[bone].x));
        assert(std::isfinite(pose.positions[bone].y));
        assert(std::isfinite(pose.positions[bone].z));
        assert(std::isfinite(pose.rotations[bone].w));
        assert(std::isfinite(pose.rotations[bone].x));
        assert(std::isfinite(pose.rotations[bone].y));
        assert(std::isfinite(pose.rotations[bone].z));
    }
}

}  // namespace g1_posture_fixture
