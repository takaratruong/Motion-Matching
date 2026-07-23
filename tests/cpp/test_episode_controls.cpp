#include "episode_controls.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    try {
        const vec3 camera_forward =
            normalize(vec3(0.6F, 0.0F, 0.8F));
        const vec3 screen_right = normalize(cross(
            camera_forward, vec3(0.0F, 1.0F, 0.0F)));
        const quat heading{};
        const auto right = episode::camera_relative_command(
            camera_forward,
            heading,
            {false, false, false, true},
            0.22F);
        const auto left = episode::camera_relative_command(
            camera_forward,
            heading,
            {false, false, true, false},
            0.22F);
        const auto diagonal = episode::camera_relative_command(
            camera_forward,
            heading,
            {true, false, false, true},
            0.22F);
        require(
            dot(right.desired_velocity_world, screen_right) > 0.21F,
            "D did not move screen-right");
        require(
            dot(left.desired_velocity_world, screen_right) < -0.21F,
            "A did not move screen-left");
        require(
            std::abs(
                length(diagonal.desired_velocity_world) - 0.22F) <
                1.0e-5F,
            "diagonal command was not normalized");
        const auto idle = episode::camera_relative_command(
            camera_forward, heading, {}, 0.22F);
        require(
            length(idle.desired_velocity_world) == 0.0F,
            "idle command moved");
        std::cout << "episode controls PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "episode controls FAILED: " << error.what()
                  << '\n';
        return 1;
    }
}
