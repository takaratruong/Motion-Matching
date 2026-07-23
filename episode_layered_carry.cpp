#include "episode_layered_carry.h"

#include "g1_arm_joint_metadata.h"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace episode {
namespace {

interaction::Transform hand_transform(
    const interaction::Pose& pose,
    interaction::Hand hand) {
    const interaction::WorldPose world =
        interaction::world_pose(pose);
    const size_t wrist = hand == interaction::Hand::Left
        ? g1_skeleton::LeftWrist
        : g1_skeleton::RightWrist;
    return {world.positions[wrist], world.rotations[wrist]};
}

const std::array<interaction::HingeJoint, 7>& arm(
    interaction::Hand hand) {
    return hand == interaction::Hand::Left
        ? interaction::kLeftArm
        : interaction::kRightArm;
}

}  // namespace

void LayeredCarry::start(
    const interaction::Pose& final_hold_pose,
    const interaction::Pose& nominal_carry_pose,
    interaction::Hand hand,
    interaction::Transform hand_in_object,
    interaction::Transform object_world) {
    nominal_carry_pose_ = nominal_carry_pose;
    last_safe_pose_ = final_hold_pose;
    hand_ = hand;
    hand_in_object_ = hand_in_object;
    object_world_ = object_world;
    temporal_seed_ =
        interaction::decompose_upper_body(
            nominal_carry_pose, hand);
    started_ = true;
    last_solve_accepted_ = true;
    last_position_error_m_ = 0.0F;
    last_orientation_error_radians_ = 0.0F;
}

const interaction::Pose& LayeredCarry::update(
    const interaction::LocomotionSnapshot& locomotion) {
    if (!started_) {
        throw std::logic_error("layered carry was not started");
    }
    interaction::Pose requested = locomotion.pose;
    for (const interaction::HingeJoint& joint : arm(hand_)) {
        const size_t bone = static_cast<size_t>(joint.bone);
        requested.rotations[bone] =
            nominal_carry_pose_.rotations[bone];
    }
    constexpr std::array<size_t, 3> spine = {
        g1_skeleton::Spine,
        g1_skeleton::Spine1,
        g1_skeleton::Spine2,
    };
    for (const size_t bone : spine) {
        requested.rotations[bone] = quat_nlerp_shortest(
            locomotion.pose.rotations[bone],
            nominal_carry_pose_.rotations[bone],
            0.25F);
    }
    const interaction::Transform requested_hand =
        hand_transform(requested, hand_);
    const interaction::Transform desired_object =
        interaction::compose(
            requested_hand,
            interaction::inverse(hand_in_object_));
    const interaction::Transform target_hand =
        interaction::compose(desired_object, hand_in_object_);

    interaction::Pose solved = requested;
    interaction::PostureIKConfig carry_ik_config{};
    carry_ik_config.accepted_position_m = 0.01F;
    carry_ik_config.accepted_orientation_radians = 0.34906585F;
    const interaction::PostureIKResult result =
        interaction::solve_hand_posture_ik_task_priority(
            solved,
            hand_,
            target_hand,
            requested,
            temporal_seed_,
            carry_ik_config);
    last_solve_accepted_ = result.accepted;
    last_position_error_m_ = result.position_error_m;
    last_orientation_error_radians_ =
        result.orientation_error_radians;
    if (result.accepted) {
        last_safe_pose_ = solved;
        temporal_seed_ = result.joint_angles;
    } else {
        interaction::Pose fallback = requested;
        for (const interaction::HingeJoint& joint : arm(hand_)) {
            const size_t bone =
                static_cast<size_t>(joint.bone);
            fallback.rotations[bone] =
                last_safe_pose_.rotations[bone];
        }
        last_safe_pose_ = fallback;
    }
    object_world_ = interaction::compose(
        hand_transform(last_safe_pose_, hand_),
        interaction::inverse(hand_in_object_));
    return last_safe_pose_;
}

const interaction::Pose& LayeredCarry::pose() const {
    return last_safe_pose_;
}

interaction::Transform LayeredCarry::object_world() const {
    return object_world_;
}

bool LayeredCarry::last_solve_accepted() const {
    return last_solve_accepted_;
}

float LayeredCarry::last_position_error_m() const {
    return last_position_error_m_;
}

float LayeredCarry::last_orientation_error_radians() const {
    return last_orientation_error_radians_;
}

}  // namespace episode
