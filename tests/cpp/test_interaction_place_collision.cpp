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

double dot_double(vec3 left, vec3 right) {
    return static_cast<double>(left.x) * right.x +
           static_cast<double>(left.y) * right.y +
           static_cast<double>(left.z) * right.z;
}

quat normalized_for_oracle(quat value) {
    const double norm = std::sqrt(
        static_cast<double>(value.w) * value.w +
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z);
    const float inverse_norm = static_cast<float>(1.0 / norm);
    return value * inverse_norm;
}

std::array<vec3, 3> rotation_axes_for_oracle(quat rotation) {
    rotation = normalized_for_oracle(rotation);
    return {
        quat_mul_vec3(rotation, vec3(1.0F, 0.0F, 0.0F)),
        quat_mul_vec3(rotation, vec3(0.0F, 1.0F, 0.0F)),
        quat_mul_vec3(rotation, vec3(0.0F, 0.0F, 1.0F)),
    };
}

double projected_radius_for_oracle(
    vec3 half_extents,
    const std::array<vec3, 3>& axes,
    vec3 axis) {
    return static_cast<double>(half_extents.x) *
               std::abs(dot_double(axes[0], axis)) +
           static_cast<double>(half_extents.y) *
               std::abs(dot_double(axes[1], axis)) +
           static_cast<double>(half_extents.z) *
               std::abs(dot_double(axes[2], axis));
}

bool separated_for_oracle(
    vec3 center_delta,
    vec3 axis,
    vec3 left_half,
    const std::array<vec3, 3>& left_axes,
    vec3 right_half,
    const std::array<vec3, 3>& right_axes) {
    const double squared_length = dot_double(axis, axis);
    if (!(squared_length > 1.0e-20)) return false;
    axis = axis * static_cast<float>(1.0 / std::sqrt(squared_length));
    return std::abs(dot_double(center_delta, axis)) >
        projected_radius_for_oracle(left_half, left_axes, axis) +
        projected_radius_for_oracle(right_half, right_axes, axis);
}

bool face_axes_overlap_for_oracle(
    interaction::Transform object_world,
    interaction::ObjectLocalBounds bounds,
    const interaction::PlacementSurface& surface) {
    const std::array<vec3, 3> support_axes = rotation_axes_for_oracle(
        surface.support_volume_world.rotation);
    const std::array<vec3, 3> object_axes = rotation_axes_for_oracle(
        object_world.rotation);
    const vec3 support_half = 0.5F * surface.support_volume_size;
    const vec3 object_center = object_world.position + quat_mul_vec3(
        object_world.rotation, bounds.center_object);
    const vec3 center_delta = object_center -
        surface.support_volume_world.position;
    for (vec3 axis : support_axes) {
        if (separated_for_oracle(
                center_delta,
                axis,
                support_half,
                support_axes,
                bounds.half_extents_object,
                object_axes)) {
            return false;
        }
    }
    for (vec3 axis : object_axes) {
        if (separated_for_oracle(
                center_delta,
                axis,
                support_half,
                support_axes,
                bounds.half_extents_object,
                object_axes)) {
            return false;
        }
    }
    return true;
}

size_t separating_cross_axis_count_for_oracle(
    interaction::Transform object_world,
    interaction::ObjectLocalBounds bounds,
    const interaction::PlacementSurface& surface) {
    const std::array<vec3, 3> support_axes = rotation_axes_for_oracle(
        surface.support_volume_world.rotation);
    const std::array<vec3, 3> object_axes = rotation_axes_for_oracle(
        object_world.rotation);
    const vec3 support_half = 0.5F * surface.support_volume_size;
    const vec3 object_center = object_world.position + quat_mul_vec3(
        object_world.rotation, bounds.center_object);
    const vec3 center_delta = object_center -
        surface.support_volume_world.position;
    size_t count = 0U;
    for (vec3 support_axis : support_axes) {
        for (vec3 object_axis : object_axes) {
            if (separated_for_oracle(
                    center_delta,
                    cross(support_axis, object_axis),
                    support_half,
                    support_axes,
                    bounds.half_extents_object,
                    object_axes)) {
                ++count;
            }
        }
    }
    return count;
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

void test_single_cross_axis_separation_is_not_a_collision() {
    using namespace interaction;
    using namespace interaction::place_collision;

    const PlacementSurface surface = make_support(vec3(2.0F, 1.1F, 0.5F));
    const ObjectLocalBounds bounds{
        vec3(), vec3(0.75F, 0.35F, 0.20F)};
    const Transform object{
        vec3(0.352325611F, 0.845091259F, -0.874350897F),
        quat(0.683911081F, 0.039219663F, 0.454193084F, -0.569592920F)};

    TEST_CHECK(face_axes_overlap_for_oracle(object, bounds, surface));
    TEST_CHECK(separating_cross_axis_count_for_oracle(
        object, bounds, surface) == 1U);
    TEST_CHECK(!object_intersects_support(object, bounds, surface));
}

void test_support_local_corners_are_common_rigid_transform_invariant() {
    using namespace interaction;
    using namespace interaction::place_collision;

    PlacementSurface surface = make_support(vec3(1.0F, 1.2F, 0.8F));
    surface.support_volume_world = {
        vec3(0.20F, -0.30F, 0.40F),
        quat_from_angle_axis(0.55F, vec3(0.0F, 1.0F, 0.0F))};
    const ObjectLocalBounds bounds{
        vec3(0.07F, -0.03F, 0.02F), vec3(0.20F, 0.10F, 0.15F)};
    const Transform object{
        vec3(0.70F, 0.10F, -0.20F),
        quat_from_angle_axis(-0.35F, vec3(1.0F, 0.0F, 0.0F))};
    const std::array<vec3, 8> before = object_corners_in_support(
        object, bounds, surface);

    const Transform common{
        vec3(1.20F, -0.40F, 0.80F),
        quat_from_angle_axis(1.05F, vec3(0.0F, 1.0F, 0.0F))};
    PlacementSurface transformed_surface = surface;
    transformed_surface.support_volume_world = compose(
        common, surface.support_volume_world);
    const Transform transformed_object = compose(common, object);
    const std::array<vec3, 8> after = object_corners_in_support(
        transformed_object, bounds, transformed_surface);

    for (size_t index = 0U; index < before.size(); ++index) {
        TEST_CHECK(std::abs(before[index].x - after[index].x) < 1.0e-5F);
        TEST_CHECK(std::abs(before[index].y - after[index].y) < 1.0e-5F);
        TEST_CHECK(std::abs(before[index].z - after[index].z) < 1.0e-5F);
    }
}

void test_quaternion_sign_equivalent_endpoints_have_zero_inflation() {
    using namespace interaction;
    using namespace interaction::place_collision;

    const PlacementSurface surface = make_support(vec3(1.0F, 2.0F, 2.0F));
    const ObjectLocalBounds bounds = centered_bounds(
        vec3(0.1F, 0.1F, 0.1F));
    const quat rotation = quat_from_angle_axis(
        0.785398163F, vec3(0.0F, 1.0F, 0.0F));
    const Transform previous{vec3(0.80F, 0.0F, 0.0F), rotation};
    const Transform current{
        previous.position,
        quat(-rotation.w, -rotation.x, -rotation.y, -rotation.z)};
    const std::array<vec3, 8> previous_corners =
        object_corners_in_support(previous, bounds, surface);
    const std::array<vec3, 8> current_corners =
        object_corners_in_support(current, bounds, surface);

    TEST_CHECK(!object_intersects_support(previous, bounds, surface));
    TEST_CHECK(!object_intersects_support(current, bounds, surface));
    for (size_t index = 0U; index < previous_corners.size(); ++index) {
        TEST_CHECK(std::abs(
            previous_corners[index].x - current_corners[index].x) < 1.0e-6F);
        TEST_CHECK(std::abs(
            previous_corners[index].y - current_corners[index].y) < 1.0e-6F);
        TEST_CHECK(std::abs(
            previous_corners[index].z - current_corners[index].z) < 1.0e-6F);
    }
    const double long_arc_expansion = 2.0 * maximum_corner_radius(bounds);
    TEST_CHECK(
        minimum_x(previous_corners) - long_arc_expansion <= 0.5);
    TEST_CHECK(!swept_interval_intersects_support(
        previous, current, bounds, surface));
}

}  // namespace

int main() {
    test_touching_is_collision_and_next_float_is_clear();
    test_same_side_separated_endpoints_and_interval_are_clear();
    test_translational_sweep_catches_clear_endpoints_crossing_support();
    test_rotating_off_center_bounds_use_angular_inflation();
    test_single_cross_axis_separation_is_not_a_collision();
    test_support_local_corners_are_common_rigid_transform_invariant();
    test_quaternion_sign_equivalent_endpoints_have_zero_inflation();
    return 0;
}
