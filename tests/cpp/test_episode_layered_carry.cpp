#include "episode_layered_carry.h"

#include "g1_arm_joint_metadata.h"
#include "g1_flat_motion_matcher.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool finite_pose(const interaction::Pose& pose) {
    for (const vec3 position : pose.positions) {
        if (!std::isfinite(position.x) ||
            !std::isfinite(position.y) ||
            !std::isfinite(position.z)) {
            return false;
        }
    }
    for (const quat rotation : pose.rotations) {
        if (!std::isfinite(rotation.w) ||
            !std::isfinite(rotation.x) ||
            !std::isfinite(rotation.y) ||
            !std::isfinite(rotation.z)) {
            return false;
        }
    }
    return true;
}

size_t wrist_bone(interaction::Hand hand) {
    return hand == interaction::Hand::Left
        ? g1_skeleton::LeftWrist
        : g1_skeleton::RightWrist;
}

void test_hand(interaction::Hand hand) {
    episode::FlatMotionMatcher matcher(
        "build/g1-episode/walking_database.bin");
    const interaction::Pose hold = matcher.snapshot().pose;
    const interaction::WorldPose hold_world =
        interaction::world_pose(hold);
    const size_t wrist = wrist_bone(hand);
    const interaction::Transform hand_world{
        hold_world.positions[wrist],
        hold_world.rotations[wrist],
    };
    const interaction::Transform object_world{
        hand_world.position + vec3(0.0F, -0.05F, 0.0F),
        hand_world.rotation,
    };
    const interaction::Transform hand_in_object =
        interaction::compose(
            interaction::inverse(object_world), hand_world);

    episode::LayeredCarry carry;
    carry.start(hold, hand, hand_in_object);
    const vec3 start_root =
        hold.positions[g1_skeleton::Simulation];
    float maximum_knee_change = 0.0F;
    float maximum_inactive_arm_change = 0.0F;
    interaction::Pose previous = hold;
    for (int tick = 0; tick < 100; ++tick) {
        matcher.update(
            {vec3(0.0F, 0.0F, 0.22F), quat()},
            1.0F / 25.0F);
        const interaction::Pose& pose =
            carry.update(matcher.snapshot());
        const std::array<interaction::HingeJoint, 7>& selected_arm =
            hand == interaction::Hand::Left
                ? interaction::kLeftArm
                : interaction::kRightArm;
        for (const interaction::HingeJoint& joint : selected_arm) {
            const size_t bone = static_cast<size_t>(joint.bone);
            require(
                quat_angle_between(
                    pose.rotations[bone], hold.rotations[bone]) < 0.002F,
                "selected arm did not retain the nominal return layer");
        }
        maximum_knee_change = std::max(
            maximum_knee_change,
            quat_angle_between(
                pose.rotations[g1_skeleton::LeftKnee],
                previous.rotations[g1_skeleton::LeftKnee]));
        const size_t inactive_shoulder =
            hand == interaction::Hand::Left
                ? g1_skeleton::RightShoulderPitch
                : g1_skeleton::LeftShoulderPitch;
        maximum_inactive_arm_change = std::max(
            maximum_inactive_arm_change,
            quat_angle_between(
                pose.rotations[inactive_shoulder],
                previous.rotations[inactive_shoulder]));
        const interaction::WorldPose world =
            interaction::world_pose(pose);
        const interaction::Transform solved_hand{
            world.positions[wrist],
            world.rotations[wrist],
        };
        const interaction::Transform published_object =
            interaction::compose(
                solved_hand,
                interaction::inverse(hand_in_object));
        require(
            length(
                published_object.position -
                carry.object_world().position) < 0.005F,
            "object was not wrist-authoritative");
        require(
            finite_pose(pose),
            "layered carry published a non-finite pose");
        previous = pose;
    }
    require(
        length(
            previous.positions[g1_skeleton::Simulation] -
            start_root) > 0.10F,
        "layered carry did not retain walking root motion");
    require(
        maximum_knee_change > 0.005F,
        "layered carry froze the legs");
    require(
        maximum_inactive_arm_change > 0.001F,
        "layered carry froze the inactive arm");
}

}  // namespace

int main() {
    try {
        test_hand(interaction::Hand::Left);
        test_hand(interaction::Hand::Right);
        std::cout << "episode layered carry PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "episode layered carry FAILED: "
                  << error.what() << '\n';
        return 1;
    }
}
