#include "episode_layered_carry.h"

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
    carry.start(
        hold, hold, hand, hand_in_object, object_world);
    const vec3 start_root =
        hold.positions[g1_skeleton::Simulation];
    float maximum_knee_change = 0.0F;
    float maximum_inactive_arm_change = 0.0F;
    interaction::Pose previous = hold;
    for (int tick = 0; tick < 150; ++tick) {
        matcher.update(
            {vec3(0.0F, 0.0F, 0.22F), quat()},
            1.0F / 25.0F);
        const interaction::Pose& pose =
            carry.update(matcher.snapshot());
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
        require(
            carry.last_position_error_m() < 0.01F,
            "accepted carry exceeded position error bound");
        require(
            carry.last_orientation_error_radians() < 0.30F,
            "accepted carry exceeded orientation error bound");
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

void test_nominal_carry_posture_replaces_reach_arm() {
    episode::FlatMotionMatcher matcher(
        "build/g1-episode/walking_database.bin");
    const interaction::Pose nominal =
        episode::load_native_g1_reference_pose(
            "build/g1-episode/carry_left_database.bin");
    require(
        finite_pose(nominal),
        "carry database did not provide a finite posture reference");
    interaction::Pose reach = nominal;
    reach.rotations[g1_skeleton::LeftShoulderPitch] =
        quat_normalize(quat_mul(
            quat_from_angle_axis(
                0.45F, vec3(0.0F, 0.0F, -1.0F)),
            reach.rotations[g1_skeleton::LeftShoulderPitch]));
    const interaction::WorldPose world =
        interaction::world_pose(reach);
    const interaction::Transform hand_world{
        world.positions[g1_skeleton::LeftWrist],
        world.rotations[g1_skeleton::LeftWrist],
    };
    const interaction::Transform object_world{
        hand_world.position + vec3(0.0F, -0.05F, 0.0F),
        hand_world.rotation,
    };
    const interaction::Transform hand_in_object =
        interaction::compose(
            interaction::inverse(object_world), hand_world);
    episode::LayeredCarry carry;
    carry.start(
        reach,
        nominal,
        interaction::Hand::Left,
        hand_in_object,
        object_world);
    const interaction::Pose& pose =
        carry.update(matcher.snapshot());
    require(
        carry.last_solve_accepted(),
        "nominal carry posture was not accepted");
    require(
        quat_angle_between(
            pose.rotations[g1_skeleton::LeftShoulderPitch],
            nominal.rotations[
                g1_skeleton::LeftShoulderPitch]) <
            quat_angle_between(
                reach.rotations[g1_skeleton::LeftShoulderPitch],
                nominal.rotations[
                    g1_skeleton::LeftShoulderPitch]),
        "active arm did not return toward nominal carry posture");
}

}  // namespace

int main() {
    try {
        test_hand(interaction::Hand::Left);
        test_hand(interaction::Hand::Right);
        test_nominal_carry_posture_replaces_reach_arm();
        std::cout << "episode layered carry PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "episode layered carry FAILED: "
                  << error.what() << '\n';
        return 1;
    }
}
