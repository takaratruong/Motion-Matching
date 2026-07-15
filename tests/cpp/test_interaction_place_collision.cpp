#include "interaction_place_collision.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

#define TEST_CHECK(condition) \
    require(static_cast<bool>(condition), #condition)

interaction::PlacementSurface make_support(vec3 size) {
    interaction::PlacementSurface surface{};
    surface.support_volume_world = {vec3(), quat()};
    surface.support_volume_size = size;
    return surface;
}

interaction::ObjectLocalBounds centered_bounds(vec3 half_extents) {
    return {vec3(), half_extents};
}

float minimum_x(const std::array<vec3, 8>& corners) {
    float value = std::numeric_limits<float>::infinity();
    for (vec3 corner : corners) value = std::min(value, corner.x);
    return value;
}

void test_touching_is_collision_and_next_float_is_clear() {
    using namespace interaction;
    using namespace interaction::place_collision;

    const PlacementSurface surface = make_support(vec3(2.0F, 2.0F, 2.0F));
    const ObjectLocalBounds bounds = centered_bounds(
        vec3(0.5F, 0.5F, 0.5F));
    Transform object{vec3(1.5F, 0.0F, 0.0F), quat()};

    TEST_CHECK(object_intersects_support(object, bounds, surface));

    object.position.x = std::nextafter(
        object.position.x, std::numeric_limits<float>::infinity());
    TEST_CHECK(!object_intersects_support(object, bounds, surface));
}

void test_same_side_separated_endpoints_and_interval_are_clear() {
    using namespace interaction;
    using namespace interaction::place_collision;

    const PlacementSurface surface = make_support(vec3(1.0F, 1.0F, 1.0F));
    const ObjectLocalBounds bounds = centered_bounds(
        vec3(0.1F, 0.1F, 0.1F));
    const Transform previous{vec3(1.0F, 0.0F, 0.0F), quat()};
    const Transform current{vec3(2.0F, 0.0F, 0.0F), quat()};

    TEST_CHECK(!object_intersects_support(previous, bounds, surface));
    TEST_CHECK(!object_intersects_support(current, bounds, surface));
    TEST_CHECK(!swept_interval_intersects_support(
        previous, current, bounds, surface));
}

void test_translational_sweep_catches_clear_endpoints_crossing_support() {
    using namespace interaction;
    using namespace interaction::place_collision;

    const PlacementSurface surface = make_support(vec3(1.0F, 1.0F, 1.0F));
    const ObjectLocalBounds bounds = centered_bounds(
        vec3(0.1F, 0.1F, 0.1F));
    const Transform previous{vec3(-1.0F, 0.0F, 0.0F), quat()};
    const Transform current{vec3(1.0F, 0.0F, 0.0F), quat()};

    TEST_CHECK(!object_intersects_support(previous, bounds, surface));
    TEST_CHECK(!object_intersects_support(current, bounds, surface));
    TEST_CHECK(swept_interval_intersects_support(
        previous, current, bounds, surface));
}

void test_rotating_off_center_bounds_use_angular_inflation() {
    using namespace interaction;
    using namespace interaction::place_collision;

    const PlacementSurface surface = make_support(vec3(1.0F, 4.0F, 4.0F));
    const ObjectLocalBounds bounds{
        vec3(1.0F, 0.0F, 0.0F), vec3(0.1F, 0.1F, 0.1F)};
    constexpr float quarter_turn_half = 0.785398163F;
    const Transform previous{
        vec3(),
        quat_from_angle_axis(
            -quarter_turn_half, vec3(0.0F, 1.0F, 0.0F))};
    const Transform current{
        vec3(),
        quat_from_angle_axis(
            quarter_turn_half, vec3(0.0F, 1.0F, 0.0F))};

    TEST_CHECK(!object_intersects_support(previous, bounds, surface));
    TEST_CHECK(!object_intersects_support(current, bounds, surface));
    TEST_CHECK(minimum_x(object_corners_in_support(
        previous, bounds, surface)) > 0.5F);
    TEST_CHECK(minimum_x(object_corners_in_support(
        current, bounds, surface)) > 0.5F);
    TEST_CHECK(std::abs(
        maximum_corner_radius(bounds) - std::sqrt(1.23)) < 1.0e-6);
    TEST_CHECK(swept_interval_intersects_support(
        previous, current, bounds, surface));
}

}  // namespace

int main() {
    test_touching_is_collision_and_next_float_is_clear();
    test_same_side_separated_endpoints_and_interval_are_clear();
    test_translational_sweep_catches_clear_endpoints_crossing_support();
    test_rotating_off_center_bounds_use_angular_inflation();
    return 0;
}
