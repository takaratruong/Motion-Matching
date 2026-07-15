#pragma once

#include "quat.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace interaction {
namespace rotation_gate {

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

inline Measure measure(quat target, quat current) {
    if (!finite_bits(target) || !finite_bits(current)) return {};

    const double tw = target.w;
    const double tx = target.x;
    const double ty = target.y;
    const double tz = target.z;
    const double cw = current.w;
    const double cx = current.x;
    const double cy = current.y;
    const double cz = current.z;

    const double relative_w =
        tw * cw + tx * cx + ty * cy + tz * cz;
    const double relative_x =
        -tw * cx + tx * cw - ty * cz + tz * cy;
    const double relative_y =
        -tw * cy + tx * cz + ty * cw - tz * cx;
    const double relative_z =
        -tw * cz - tx * cy + ty * cx + tz * cw;
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

inline double maximum_half_tangent_squared(float maximum_radians) {
    const double tangent = std::tan(
        0.5 * static_cast<double>(maximum_radians));
    const double result = tangent * tangent;
    return finite_bits(result)
        ? result
        : std::numeric_limits<double>::max();
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
