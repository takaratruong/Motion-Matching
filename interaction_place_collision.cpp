#include "interaction_place_collision.h"

#include "interaction_pose.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace interaction {
namespace place_collision {
namespace {

constexpr float kMinimumNorm = 1.0e-12F;

quat normalized(quat value) {
    const double norm = std::sqrt(
        static_cast<double>(value.w) * value.w +
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z);
    if (!(norm > kMinimumNorm) || !std::isfinite(norm)) return quat();
    const float inverse_norm = static_cast<float>(1.0 / norm);
    return value * inverse_norm;
}

double rotation_error(quat left, quat right) {
    left = normalized(left);
    right = normalized(right);
    const double cosine = std::clamp(
        std::abs(
            static_cast<double>(left.w) * right.w +
            static_cast<double>(left.x) * right.x +
            static_cast<double>(left.y) * right.y +
            static_cast<double>(left.z) * right.z),
        0.0,
        1.0);
    return 2.0 * std::acos(cosine);
}

double dot_double(vec3 left, vec3 right) {
    return static_cast<double>(left.x) * right.x +
           static_cast<double>(left.y) * right.y +
           static_cast<double>(left.z) * right.z;
}

std::array<vec3, 3> rotation_axes(quat rotation) {
    rotation = normalized(rotation);
    return {
        quat_mul_vec3(rotation, vec3(1.0F, 0.0F, 0.0F)),
        quat_mul_vec3(rotation, vec3(0.0F, 1.0F, 0.0F)),
        quat_mul_vec3(rotation, vec3(0.0F, 0.0F, 1.0F)),
    };
}

double projected_radius(
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

bool separated_on_axis(
    vec3 center_delta,
    vec3 axis,
    vec3 left_half,
    const std::array<vec3, 3>& left_axes,
    vec3 right_half,
    const std::array<vec3, 3>& right_axes) {
    const double squared_length = dot_double(axis, axis);
    if (!(squared_length > 1.0e-20)) return false;
    axis = axis * static_cast<float>(1.0 / std::sqrt(squared_length));
    const double center_projection = std::abs(dot_double(center_delta, axis));
    return center_projection >
        projected_radius(left_half, left_axes, axis) +
        projected_radius(right_half, right_axes, axis);
}

}  // namespace

bool object_intersects_support(
    Transform object_world,
    ObjectLocalBounds bounds,
    const PlacementSurface& surface) {
    const std::array<vec3, 3> support_axes = rotation_axes(
        surface.support_volume_world.rotation);
    const std::array<vec3, 3> object_axes = rotation_axes(
        object_world.rotation);
    const vec3 support_half = 0.5F * surface.support_volume_size;
    const vec3 object_center = object_world.position + quat_mul_vec3(
        object_world.rotation, bounds.center_object);
    const vec3 center_delta = object_center -
        surface.support_volume_world.position;
    for (vec3 axis : support_axes) {
        if (separated_on_axis(
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
        if (separated_on_axis(
                center_delta,
                axis,
                support_half,
                support_axes,
                bounds.half_extents_object,
                object_axes)) {
            return false;
        }
    }
    for (vec3 support_axis : support_axes) {
        for (vec3 object_axis : object_axes) {
            if (separated_on_axis(
                    center_delta,
                    cross(support_axis, object_axis),
                    support_half,
                    support_axes,
                    bounds.half_extents_object,
                    object_axes)) {
                return false;
            }
        }
    }
    return true;
}

std::array<vec3, 8> object_corners_in_support(
    Transform object_world,
    ObjectLocalBounds bounds,
    const PlacementSurface& surface) {
    std::array<vec3, 8> corners{};
    const Transform support_from_world = inverse(
        surface.support_volume_world);
    size_t index = 0U;
    for (int sign_x : {-1, 1}) {
        for (int sign_y : {-1, 1}) {
            for (int sign_z : {-1, 1}) {
                const vec3 local = bounds.center_object + vec3(
                    static_cast<float>(sign_x) *
                        bounds.half_extents_object.x,
                    static_cast<float>(sign_y) *
                        bounds.half_extents_object.y,
                    static_cast<float>(sign_z) *
                        bounds.half_extents_object.z);
                const vec3 world = object_world.position + quat_mul_vec3(
                    object_world.rotation, local);
                corners[index++] = support_from_world.position +
                    quat_mul_vec3(support_from_world.rotation, world);
            }
        }
    }
    return corners;
}

double maximum_corner_radius(ObjectLocalBounds bounds) {
    double maximum = 0.0;
    for (int sign_x : {-1, 1}) {
        for (int sign_y : {-1, 1}) {
            for (int sign_z : {-1, 1}) {
                const vec3 corner = bounds.center_object + vec3(
                    static_cast<float>(sign_x) *
                        bounds.half_extents_object.x,
                    static_cast<float>(sign_y) *
                        bounds.half_extents_object.y,
                    static_cast<float>(sign_z) *
                        bounds.half_extents_object.z);
                maximum = std::max(
                    maximum,
                    std::sqrt(dot_double(corner, corner)));
            }
        }
    }
    return maximum;
}

bool swept_interval_intersects_support(
    Transform previous,
    Transform current,
    ObjectLocalBounds bounds,
    const PlacementSurface& surface) {
    const std::array<vec3, 8> previous_corners =
        object_corners_in_support(previous, bounds, surface);
    const std::array<vec3, 8> current_corners =
        object_corners_in_support(current, bounds, surface);
    vec3 minimum = previous_corners.front();
    vec3 maximum = previous_corners.front();
    const auto accumulate = [&](vec3 corner) {
        minimum.x = std::min(minimum.x, corner.x);
        minimum.y = std::min(minimum.y, corner.y);
        minimum.z = std::min(minimum.z, corner.z);
        maximum.x = std::max(maximum.x, corner.x);
        maximum.y = std::max(maximum.y, corner.y);
        maximum.z = std::max(maximum.z, corner.z);
    };
    for (vec3 corner : previous_corners) accumulate(corner);
    for (vec3 corner : current_corners) accumulate(corner);
    const double angle = rotation_error(
        previous.rotation, current.rotation);
    const float angular_expansion = static_cast<float>(
        maximum_corner_radius(bounds) * (1.0 - std::cos(0.5 * angle)));
    minimum = minimum - vec3(
        angular_expansion, angular_expansion, angular_expansion);
    maximum = maximum + vec3(
        angular_expansion, angular_expansion, angular_expansion);
    const vec3 support_half = 0.5F * surface.support_volume_size;
    return minimum.x <= support_half.x && maximum.x >= -support_half.x &&
           minimum.y <= support_half.y && maximum.y >= -support_half.y &&
           minimum.z <= support_half.z && maximum.z >= -support_half.z;
}

}  // namespace place_collision
}  // namespace interaction
