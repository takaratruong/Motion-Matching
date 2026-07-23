#include "episode_layered_carry.h"

#include "g1_arm_joint_metadata.h"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace episode {
namespace {

interaction::Transform root_transform(
    const interaction::Pose& pose) {
    return {
        pose.positions[g1_skeleton::Simulation],
        pose.rotations[g1_skeleton::Simulation],
    };
}

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
    interaction::Hand hand,
    interaction::Transform hand_in_object,
    interaction::Transform object_world) {
    final_hold_pose_ = final_hold_pose;
    last_safe_pose_ = final_hold_pose;
    hand_ = hand;
    hand_in_object_ = hand_in_object;
    object_in_root_ = interaction::compose(
        interaction::inverse(root_transform(final_hold_pose)),
        object_world);
    object_world_ = object_world;
    temporal_seed_ =
        interaction::decompose_upper_body(final_hold_pose, hand);
    started_ = true;
    last_solve_accepted_ = true;
}

const interaction::Pose& LayeredCarry::update(
    const interaction::LocomotionSnapshot& locomotion) {
    if (!started_) {
        throw std::logic_error("layered carry was not started");
    }
    const interaction::Transform desired_object =
        interaction::compose(
            root_transform(locomotion.pose), object_in_root_);
    const interaction::Transform target_hand =
        interaction::compose(desired_object, hand_in_object_);
    interaction::Pose requested = locomotion.pose;
    for (const interaction::HingeJoint& joint : arm(hand_)) {
        const size_t bone = static_cast<size_t>(joint.bone);
        requested.rotations[bone] =
            final_hold_pose_.rotations[bone];
    }
    constexpr std::array<size_t, 3> spine = {
        g1_skeleton::Spine,
        g1_skeleton::Spine1,
        g1_skeleton::Spine2,
    };
    for (const size_t bone : spine) {
        requested.rotations[bone] = quat_nlerp_shortest(
            locomotion.pose.rotations[bone],
            final_hold_pose_.rotations[bone],
            0.25F);
    }

    interaction::Pose solved = requested;
    const interaction::PostureIKResult result =
        interaction::solve_hand_posture_ik_task_priority(
            solved,
            hand_,
            target_hand,
            requested,
            temporal_seed_);
    last_solve_accepted_ = result.accepted;
    if (result.accepted) {
        last_safe_pose_ = solved;
        temporal_seed_ = result.joint_angles;
    } else {
        const interaction::Transform live_root =
            root_transform(locomotion.pose);
        last_safe_pose_.positions[g1_skeleton::Simulation] =
            live_root.position;
        last_safe_pose_.rotations[g1_skeleton::Simulation] =
            live_root.rotation;
        last_safe_pose_.velocities[g1_skeleton::Simulation] =
            locomotion.pose.velocities[g1_skeleton::Simulation];
        last_safe_pose_.angular_velocities[g1_skeleton::Simulation] =
            locomotion.pose.angular_velocities[
                g1_skeleton::Simulation];
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

}  // namespace episode
