#pragma once

#include "quat.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace interaction {
namespace rotation_gate {

struct Rotation {
    double w = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Measure {
    double vector_squared = 0.0;
    double scalar_squared = 0.0;
    double radians = 0.0;
    bool valid = false;
};

inline bool finite_bits(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

inline bool finite_bits(double value) {
    uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7ff0000000000000ULL) !=
        0x7ff0000000000000ULL;
}

inline bool finite_bits(quat value) {
    return finite_bits(value.w) && finite_bits(value.x) &&
           finite_bits(value.y) && finite_bits(value.z);
}

inline bool finite_bits(const Rotation& value) {
    return finite_bits(value.w) && finite_bits(value.x) &&
           finite_bits(value.y) && finite_bits(value.z);
}

inline Rotation from_quat(quat value) {
    return {value.w, value.x, value.y, value.z};
}

inline Rotation multiply(Rotation left, Rotation right) {
    return {
        right.w * left.w - right.x * left.x -
            right.y * left.y - right.z * left.z,
        right.w * left.x + right.x * left.w -
            right.y * left.z + right.z * left.y,
        right.w * left.y + right.x * left.z +
            right.y * left.w - right.z * left.x,
        right.w * left.z - right.x * left.y +
            right.y * left.x + right.z * left.w,
    };
}

inline Rotation inverse(Rotation value) {
    return {-value.w, value.x, value.y, value.z};
}

inline Measure measure(Rotation target, Rotation current) {
    if (!finite_bits(target) || !finite_bits(current)) return {};

    const double relative_w =
        target.w * current.w + target.x * current.x +
        target.y * current.y + target.z * current.z;
    const double relative_x =
        -target.w * current.x + target.x * current.w -
        target.y * current.z + target.z * current.y;
    const double relative_y =
        -target.w * current.y + target.x * current.z +
        target.y * current.w - target.z * current.x;
    const double relative_z =
        -target.w * current.z - target.x * current.y +
        target.y * current.x + target.z * current.w;
    const double vector_squared =
        relative_x * relative_x +
        relative_y * relative_y +
        relative_z * relative_z;
    const double scalar_squared = relative_w * relative_w;
    const double norm_squared = vector_squared + scalar_squared;
    if (!finite_bits(vector_squared) || !finite_bits(scalar_squared) ||
        !(norm_squared > 0.0) || !finite_bits(norm_squared)) {
        return {};
    }

    const double radians = 2.0 * std::atan2(
        std::sqrt(vector_squared), std::abs(relative_w));
    if (!finite_bits(radians)) return {};
    return {vector_squared, scalar_squared, radians, true};
}

inline Measure measure(quat target, quat current) {
    return measure(from_quat(target), from_quat(current));
}

inline double half_tangent_squared(const Measure& value) {
    if (!value.valid) return std::numeric_limits<double>::max();
    if (value.scalar_squared == 0.0) {
        return std::numeric_limits<double>::max();
    }
    const double result = value.vector_squared / value.scalar_squared;
    return finite_bits(result)
        ? result
        : std::numeric_limits<double>::max();
}

inline double encoded_half_tangent_squared(float radians) {
    const float half = 0.5F * radians;
    const float sine = std::sin(half);
    const float cosine = std::cos(half);
    const double sine_squared =
        static_cast<double>(sine) * static_cast<double>(sine);
    const double cosine_squared =
        static_cast<double>(cosine) * static_cast<double>(cosine);
    if (cosine_squared == 0.0) {
        return std::numeric_limits<double>::max();
    }
    const double result = sine_squared / cosine_squared;
    return finite_bits(result)
        ? result
        : std::numeric_limits<double>::max();
}

inline double maximum_half_tangent_squared(float maximum_radians) {
    const double exact = encoded_half_tangent_squared(maximum_radians);
    const float next_radians = std::nextafter(
        maximum_radians, std::numeric_limits<float>::infinity());
    const double next = encoded_half_tangent_squared(next_radians);
    if (finite_bits(exact) && finite_bits(next) && next > exact) {
        return exact + 0.5 * (next - exact);
    }
    return exact;
}

inline bool within(const Measure& value, float maximum_radians) {
    constexpr float kPi = 3.141592654F;
    if (!value.valid || !finite_bits(maximum_radians) ||
        maximum_radians < 0.0F) {
        return false;
    }
    if (maximum_radians >= kPi) return true;
    return half_tangent_squared(value) <=
        maximum_half_tangent_squared(maximum_radians);
}

inline bool within(
    const Rotation& target,
    const Rotation& current,
    float maximum_radians) {
    return within(measure(target, current), maximum_radians);
}

inline bool within(
    quat target,
    quat current,
    float maximum_radians) {
    return within(measure(target, current), maximum_radians);
}

inline float radians(const Measure& value) {
    if (!value.valid ||
        value.radians > std::numeric_limits<float>::max()) {
        return std::numeric_limits<float>::max();
    }
    return static_cast<float>(value.radians);
}

inline float classified_radians(
    const Measure& value,
    float maximum_radians) {
    float result = radians(value);
    if (within(value, maximum_radians)) {
        if (result > maximum_radians) result = maximum_radians;
        return result;
    }
    if (!(result > maximum_radians) &&
        finite_bits(maximum_radians)) {
        result = std::nextafter(
            maximum_radians, std::numeric_limits<float>::infinity());
    }
    return result;
}

}  // namespace rotation_gate
}  // namespace interaction
