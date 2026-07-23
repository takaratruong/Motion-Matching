#include "g1_flat_motion_matcher.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool finite_pose(const interaction::Pose& pose) {
    for (const vec3 position : pose.positions) {
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z)) return false;
    }
    for (const quat rotation : pose.rotations) {
        if (!std::isfinite(rotation.w) || !std::isfinite(rotation.x) ||
            !std::isfinite(rotation.y) || !std::isfinite(rotation.z)) {
            return false;
        }
    }
    return true;
}

float heading_error(quat left, quat right) {
    vec3 left_forward = quat_mul_vec3(left, vec3(0.0F, 0.0F, 1.0F));
    vec3 right_forward = quat_mul_vec3(right, vec3(0.0F, 0.0F, 1.0F));
    left_forward.y = 0.0F;
    right_forward.y = 0.0F;
    return std::acos(clampf(
        dot(normalize(left_forward), normalize(right_forward)),
        -1.0F,
        1.0F));
}

void test_heading_command_converges(
    const std::filesystem::path& walking_database) {
    const std::filesystem::path pack("build/g1-episode");
    episode::FlatMotionMatcher matcher(
        walking_database, pack / "carry_left_database.bin");
    const quat start = matcher.snapshot().pose.rotations[0];
    const quat target = quat_mul(
        quat_from_angle_axis(
            0.5F * 3.14159265358979323846F,
            vec3(0.0F, 1.0F, 0.0F)),
        start);
    const float initial_error = heading_error(start, target);
    for (int tick = 0; tick < 100; ++tick) {
        matcher.update({vec3(), target}, 1.0F / 25.0F);
    }
    const float final_error = heading_error(
        matcher.snapshot().pose.rotations[0], target);
    require(
        final_error < 0.5F * initial_error,
        "heading command diverged: initial " +
            std::to_string(initial_error) + " final " +
            std::to_string(final_error));
}

void test_real_motion_and_switch_preserve_root(
    const std::filesystem::path& walking_database) {
    const std::filesystem::path pack("build/g1-episode");
    episode::FlatMotionMatcher matcher(
        walking_database, pack / "carry_left_database.bin");
    const vec3 start = matcher.snapshot().pose.positions[0];
    vec3 previous = start;
    float maximum_step = 0.0F;
    episode::FlatSkeletonWorldPose previous_flat =
        matcher.flat_skeleton();
    float maximum_flat_joint_step = 0.0F;
    const episode::LocomotionCommand command{
        vec3(0.0F, 0.0F, 0.60F), quat()};
    for (int tick = 0; tick < 100; ++tick) {
        matcher.update(command, 1.0F / 25.0F);
        const interaction::Pose& pose = matcher.snapshot().pose;
        require(finite_pose(pose), "matcher published a non-finite pose");
        require(
            matcher.flat_skeleton().valid,
            "23-bone matcher did not publish its visible skeleton");
        maximum_step = std::max(
            maximum_step, length(pose.positions[0] - previous));
        for (size_t bone = 1U;
             bone < episode::kFlatSkeletonBoneCount;
             ++bone) {
            maximum_flat_joint_step = std::max(
                maximum_flat_joint_step,
                length(
                    matcher.flat_skeleton().positions[bone] -
                    previous_flat.positions[bone]));
        }
        previous_flat = matcher.flat_skeleton();
        previous = pose.positions[0];
    }
    require(
        length(previous - start) > 0.20F,
        "forward command did not move the root");
    require(
        maximum_step < 0.20F,
        "walking root step was discontinuous: " +
            std::to_string(maximum_step));
    require(
        maximum_flat_joint_step < 0.30F,
        "visible flat skeleton popped between matcher clips: " +
            std::to_string(maximum_flat_joint_step));

    const vec3 before_switch = matcher.snapshot().pose.positions[0];
    matcher.switch_database(pack / "carry_left_database.bin");
    require(
        length(matcher.snapshot().pose.positions[0] - before_switch) <
            1.0e-5F,
        "left carry switch moved the live root");
    matcher.switch_database(pack / "carry_right_database.bin");
    require(
        length(matcher.snapshot().pose.positions[0] - before_switch) <
            1.0e-5F,
        "right carry switch moved the live root");
    matcher.reset_database(walking_database);
    require(
        length(
            matcher.snapshot().pose.positions[0] -
            vec3(0.0F, 0.0F, -1.40F)) < 1.0e-5F,
        "walking reset did not restore the scene spawn");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path walking = argc >= 2
            ? std::filesystem::path(argv[1])
            : std::filesystem::path(
                  "build/g1-episode/walking_database.bin");
        test_real_motion_and_switch_preserve_root(walking);
        test_heading_command_converges(walking);
        std::cout << "flat G1 motion matcher PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "flat G1 motion matcher FAILED: " << error.what()
                  << '\n';
        return 1;
    }
}
