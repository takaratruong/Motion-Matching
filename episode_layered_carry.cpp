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
    const interaction::Pose& nominal_return_pose,
    interaction::Hand hand,
    interaction::Transform hand_in_object) {
    nominal_return_pose_ = nominal_return_pose;
    pose_ = nominal_return_pose;
    hand_ = hand;
    hand_in_object_ = hand_in_object;
    object_world_ = interaction::compose(
        hand_transform(pose_, hand_),
        interaction::inverse(hand_in_object_));
    started_ = true;
}

const interaction::Pose& LayeredCarry::update(
    const interaction::LocomotionSnapshot& locomotion) {
    if (!started_) {
        throw std::logic_error("layered carry was not started");
    }
    pose_ = locomotion.pose;
    for (const interaction::HingeJoint& joint : arm(hand_)) {
        const size_t bone = static_cast<size_t>(joint.bone);
        pose_.rotations[bone] =
            nominal_return_pose_.rotations[bone];
    }
    constexpr std::array<size_t, 3> spine = {
        g1_skeleton::Spine,
        g1_skeleton::Spine1,
        g1_skeleton::Spine2,
    };
    for (const size_t bone : spine) {
        pose_.rotations[bone] = quat_nlerp_shortest(
            locomotion.pose.rotations[bone],
            nominal_return_pose_.rotations[bone],
            0.25F);
    }
    object_world_ = interaction::compose(
        hand_transform(pose_, hand_),
        interaction::inverse(hand_in_object_));
    return pose_;
}

const interaction::Pose& LayeredCarry::pose() const {
    return pose_;
}

interaction::Transform LayeredCarry::object_world() const {
    return object_world_;
}

}  // namespace episode
