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

void test_real_motion_and_switch_preserve_root() {
    const std::filesystem::path pack("build/g1-episode");
    episode::FlatMotionMatcher matcher(pack / "walking_database.bin");
    const vec3 start = matcher.snapshot().pose.positions[0];
    vec3 previous = start;
    float maximum_step = 0.0F;
    const episode::LocomotionCommand command{
        vec3(0.0F, 0.0F, 0.60F), quat()};
    for (int tick = 0; tick < 100; ++tick) {
        matcher.update(command, 1.0F / 25.0F);
        const interaction::Pose& pose = matcher.snapshot().pose;
        require(finite_pose(pose), "matcher published a non-finite pose");
        maximum_step = std::max(
            maximum_step, length(pose.positions[0] - previous));
        previous = pose.positions[0];
    }
    require(
        length(previous - start) > 0.20F,
        "forward command did not move the root");
    require(
        maximum_step < 0.20F,
        "walking root step was discontinuous: " +
            std::to_string(maximum_step));

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
}

}  // namespace

int main() {
    try {
        test_real_motion_and_switch_preserve_root();
        std::cout << "flat G1 motion matcher PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "flat G1 motion matcher FAILED: " << error.what()
                  << '\n';
        return 1;
    }
}
