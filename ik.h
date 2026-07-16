#pragma once

#include "terrain_runtime.h"

#include <cfloat>
#include <cmath>
#include <limits>

struct IKTargetProjection
{
    bool reachable = false;
    vec3 clamped_target;
    float raw_distance_m = 0.0f;
    float clamped_distance_m = 0.0f;
    float minimum_distance_m = 0.0f;
    float maximum_distance_m = 0.0f;
};

struct IKReachShell
{
    double minimum_distance_m = 0.0;
    double maximum_distance_m = 0.0;
    float minimum_distance_f32_m = 0.0f;
    float maximum_distance_f32_m = 0.0f;
};

bool ik_effective_reach_shell(
    IKReachShell& output,
    const vec3& root,
    const vec3& middle,
    const vec3& end,
    float reach_buffer_m);

struct IKClampResult
{
    quat value;
    float requested_radians = 0.0f;
    float actual_radians = 0.0f;
    bool limited = false;
};

static constexpr uint32_t IKClampBoundaryAttemptCapacity = 64U;

#if defined(G1_IK_ENABLE_TEST_SEAMS)
static constexpr uint32_t IKClampBoundaryAuditCapacity = 64U;

struct IKClampBoundaryAuditAttempt
{
    double interpolation_angle = 0.0;
    double precise_angle = 0.0;
    float rounded_angle = 0.0f;
    bool accepted = false;
};

struct IKClampBoundaryAudit
{
    IKClampBoundaryAuditAttempt
        attempts[IKClampBoundaryAuditCapacity] = {};
    uint32_t attempt_count = 0U;
    bool finite_exhausted = false;
};

static_assert(
    IKClampBoundaryAttemptCapacity ==
        static_cast<uint32_t>(
            sizeof(((IKClampBoundaryAudit*)nullptr)->attempts) /
            sizeof(((IKClampBoundaryAudit*)nullptr)->attempts[0])),
    "IK clamp production frontier must equal audit capacity");
#endif

struct IKBendSelection
{
    vec3 direction;
    float current_projection_length = 0.0f;
    bool used_current_projection = false;
    bool used_hinge_fallback = false;
    bool used_safe_perpendicular = false;
    bool sign_flipped = false;
};

struct IKTwoBoneResult
{
    bool applied = false;
    bool reachable = false;
    bool correction_limited = false;
    quat root_local;
    quat middle_local;
    float root_correction_radians = 0.0f;
    float middle_correction_radians = 0.0f;
    IKTargetProjection target;
    IKBendSelection bend;
};

static inline bool ik_float_is_runtime_value(float value)
{
    return terrain_float_is_normal_or_zero_query(value);
}

static inline bool ik_vec3_is_runtime_value(vec3 value)
{
    return ik_float_is_runtime_value(value.x) &&
           ik_float_is_runtime_value(value.y) &&
           ik_float_is_runtime_value(value.z);
}

static inline bool ik_quat_components_are_runtime_values(quat value)
{
    return ik_float_is_runtime_value(value.w) &&
           ik_float_is_runtime_value(value.x) &&
           ik_float_is_runtime_value(value.y) &&
           ik_float_is_runtime_value(value.z);
}

static inline bool ik_checked_binary32_commit(
    double source, float& output)
{
    if (!terrain_double_is_finite(source)) {
        return false;
    }
    const volatile double materialized = source;
    const volatile float rounded = static_cast<float>(materialized);
    const uint32_t bits = terrain_float_bits(rounded);
    const uint32_t magnitude = bits & UINT32_C(0x7fffffff);
    const uint32_t exponent = magnitude & UINT32_C(0x7f800000);
    if (exponent == UINT32_C(0x7f800000)) {
        return false;
    }
    if (magnitude == 0) {
        if (source != 0.0) {
            return false;
        }
        output = 0.0f;
        return true;
    }
    if (exponent == 0) {
        return false;
    }
    output = rounded;
    return true;
}

static inline bool ik_checked_norm_components(
    double& precise_length,
    float& rounded_length,
    const double* components,
    int count,
    bool require_binary32_squared_length)
{
    double maximum = 0.0;
    for (int i = 0; i < count; ++i) {
        if (!terrain_double_is_finite(components[i])) {
            return false;
        }
        const double absolute = fabs(components[i]);
        maximum = absolute > maximum ? absolute : maximum;
    }
    if (maximum == 0.0) {
        precise_length = 0.0;
        rounded_length = 0.0f;
        return true;
    }

    double square_sum = 0.0;
    for (int i = 0; i < count; ++i) {
        const volatile double scaled = components[i] / maximum;
        const volatile double square = scaled * scaled;
        const volatile double sum = square_sum + square;
        square_sum = sum;
    }
    if (!terrain_double_is_finite(square_sum) || square_sum <= 0.0) {
        return false;
    }
    const volatile double scaled_length = sqrt(square_sum);
    const volatile double length = maximum * scaled_length;
    const volatile double squared_maximum = maximum * maximum;
    const volatile double squared_length = squared_maximum * square_sum;
    if (!terrain_double_is_finite(length) || length <= 0.0 ||
        !terrain_double_is_finite(squared_length) ||
        (require_binary32_squared_length &&
         squared_length > static_cast<double>(FLT_MAX)) ||
        !ik_checked_binary32_commit(length, rounded_length)) {
        return false;
    }
    precise_length = length;
    return true;
}

static inline bool ik_checked_vec3_norm(
    double& precise_length,
    float& rounded_length,
    vec3 value,
    bool require_binary32_squared_length = true)
{
    if (!ik_vec3_is_runtime_value(value)) {
        return false;
    }
    const double components[3] = {
        static_cast<double>(value.x),
        static_cast<double>(value.y),
        static_cast<double>(value.z)
    };
    return ik_checked_norm_components(
        precise_length, rounded_length, components, 3,
        require_binary32_squared_length);
}

static inline bool ik_checked_quat_norm(
    double& precise_length, float& rounded_length, quat value)
{
    if (!ik_quat_components_are_runtime_values(value)) {
        return false;
    }
    const double components[4] = {
        static_cast<double>(value.w),
        static_cast<double>(value.x),
        static_cast<double>(value.y),
        static_cast<double>(value.z)
    };
    return ik_checked_norm_components(
        precise_length, rounded_length, components, 4, true);
}

static inline bool ik_vec3_is_unit(vec3 value)
{
    double length = 0.0;
    float rounded = 0.0f;
    return ik_checked_vec3_norm(length, rounded, value) &&
           fabs(length - 1.0) <= 2.0e-5;
}

static inline bool ik_quat_is_unit(quat value)
{
    double length = 0.0;
    float rounded = 0.0f;
    return ik_checked_quat_norm(length, rounded, value) &&
           fabs(length - 1.0) <= 2.0e-5;
}

static inline bool ik_checked_vec3_from_double(
    vec3& output, double x, double y, double z)
{
    vec3 candidate;
    if (!ik_checked_binary32_commit(x, candidate.x) ||
        !ik_checked_binary32_commit(y, candidate.y) ||
        !ik_checked_binary32_commit(z, candidate.z) ||
        !ik_vec3_is_runtime_value(candidate)) {
        return false;
    }
    output = candidate;
    return true;
}

static inline bool ik_checked_vec3_subtract(
    vec3& output, vec3 left, vec3 right)
{
    if (!ik_vec3_is_runtime_value(left) ||
        !ik_vec3_is_runtime_value(right)) {
        return false;
    }
    const volatile double x =
        static_cast<double>(left.x) - static_cast<double>(right.x);
    const volatile double y =
        static_cast<double>(left.y) - static_cast<double>(right.y);
    const volatile double z =
        static_cast<double>(left.z) - static_cast<double>(right.z);
    return ik_checked_vec3_from_double(output, x, y, z);
}

static inline bool ik_checked_vec3_add(
    vec3& output, vec3 left, vec3 right)
{
    if (!ik_vec3_is_runtime_value(left) ||
        !ik_vec3_is_runtime_value(right)) {
        return false;
    }
    const volatile double x =
        static_cast<double>(left.x) + static_cast<double>(right.x);
    const volatile double y =
        static_cast<double>(left.y) + static_cast<double>(right.y);
    const volatile double z =
        static_cast<double>(left.z) + static_cast<double>(right.z);
    return ik_checked_vec3_from_double(output, x, y, z);
}

static inline bool ik_checked_vec3_scale(
    vec3& output, vec3 value, double scale)
{
    if (!ik_vec3_is_runtime_value(value) ||
        !terrain_double_is_finite(scale)) {
        return false;
    }
    const volatile double x = static_cast<double>(value.x) * scale;
    const volatile double y = static_cast<double>(value.y) * scale;
    const volatile double z = static_cast<double>(value.z) * scale;
    return ik_checked_vec3_from_double(output, x, y, z);
}

static inline bool ik_checked_dot(double& output, vec3 left, vec3 right)
{
    if (!ik_vec3_is_runtime_value(left) ||
        !ik_vec3_is_runtime_value(right)) {
        return false;
    }
    const volatile double x =
        static_cast<double>(left.x) * static_cast<double>(right.x);
    const volatile double y =
        static_cast<double>(left.y) * static_cast<double>(right.y);
    const volatile double z =
        static_cast<double>(left.z) * static_cast<double>(right.z);
    const volatile double first_sum = x + y;
    const volatile double final_sum = first_sum + z;
    if (!terrain_double_is_finite(final_sum)) {
        return false;
    }
    output = final_sum;
    return true;
}

static inline bool ik_checked_cross(vec3& output, vec3 left, vec3 right)
{
    if (!ik_vec3_is_runtime_value(left) ||
        !ik_vec3_is_runtime_value(right)) {
        return false;
    }
    const volatile double x =
        static_cast<double>(left.y) * static_cast<double>(right.z) -
        static_cast<double>(left.z) * static_cast<double>(right.y);
    const volatile double y =
        static_cast<double>(left.z) * static_cast<double>(right.x) -
        static_cast<double>(left.x) * static_cast<double>(right.z);
    const volatile double z =
        static_cast<double>(left.x) * static_cast<double>(right.y) -
        static_cast<double>(left.y) * static_cast<double>(right.x);
    return ik_checked_vec3_from_double(output, x, y, z);
}

static inline bool ik_checked_normalize(vec3& output, vec3 value)
{
    double length = 0.0;
    float rounded_length = 0.0f;
    if (!ik_checked_vec3_norm(
            length, rounded_length, value) || length <= 0.0) {
        return false;
    }
    vec3 candidate;
    if (!ik_checked_vec3_from_double(
            candidate,
            static_cast<double>(value.x) / length,
            static_cast<double>(value.y) / length,
            static_cast<double>(value.z) / length) ||
        !ik_vec3_is_unit(candidate)) {
        return false;
    }
    output = candidate;
    return true;
}

static inline bool ik_checked_distance_precise(
    double& precise_output,
    float& rounded_output,
    vec3 left,
    vec3 right)
{
    if (!ik_vec3_is_runtime_value(left) ||
        !ik_vec3_is_runtime_value(right)) {
        return false;
    }
    const double components[3] = {
        static_cast<double>(left.x) - static_cast<double>(right.x),
        static_cast<double>(left.y) - static_cast<double>(right.y),
        static_cast<double>(left.z) - static_cast<double>(right.z)
    };
    double precise = 0.0;
    float rounded = 0.0f;
    if (!ik_checked_norm_components(
            precise, rounded, components, 3, true)) {
        return false;
    }
    precise_output = precise;
    rounded_output = rounded;
    return true;
}

static inline bool ik_checked_distance(
    float& output, vec3 left, vec3 right)
{
    double precise = 0.0;
    float rounded = 0.0f;
    if (!ik_checked_distance_precise(
            precise, rounded, left, right)) {
        return false;
    }
    output = rounded;
    return true;
}

static inline bool ik_checked_quat_from_double(
    quat& output, double w, double x, double y, double z)
{
    const double components[4] = {w, x, y, z};
    double length = 0.0;
    float rounded_length = 0.0f;
    if (!ik_checked_norm_components(
            length, rounded_length, components, 4, true) ||
        length <= 0.0) {
        return false;
    }
    quat candidate;
    if (!ik_checked_binary32_commit(w / length, candidate.w) ||
        !ik_checked_binary32_commit(x / length, candidate.x) ||
        !ik_checked_binary32_commit(y / length, candidate.y) ||
        !ik_checked_binary32_commit(z / length, candidate.z) ||
        !ik_quat_is_unit(candidate)) {
        return false;
    }
    output = candidate;
    return true;
}

static inline bool ik_checked_quat_normalize(quat& output, quat value)
{
    if (!ik_quat_components_are_runtime_values(value)) {
        return false;
    }
    return ik_checked_quat_from_double(
        output,
        static_cast<double>(value.w),
        static_cast<double>(value.x),
        static_cast<double>(value.y),
        static_cast<double>(value.z));
}

static inline bool ik_checked_quat_multiply(
    quat& output, quat left, quat right)
{
    if (!ik_quat_is_unit(left) || !ik_quat_is_unit(right)) {
        return false;
    }
    const volatile double w =
        static_cast<double>(right.w) * static_cast<double>(left.w) -
        static_cast<double>(right.x) * static_cast<double>(left.x) -
        static_cast<double>(right.y) * static_cast<double>(left.y) -
        static_cast<double>(right.z) * static_cast<double>(left.z);
    const volatile double x =
        static_cast<double>(right.w) * static_cast<double>(left.x) +
        static_cast<double>(right.x) * static_cast<double>(left.w) -
        static_cast<double>(right.y) * static_cast<double>(left.z) +
        static_cast<double>(right.z) * static_cast<double>(left.y);
    const volatile double y =
        static_cast<double>(right.w) * static_cast<double>(left.y) +
        static_cast<double>(right.x) * static_cast<double>(left.z) +
        static_cast<double>(right.y) * static_cast<double>(left.w) -
        static_cast<double>(right.z) * static_cast<double>(left.x);
    const volatile double z =
        static_cast<double>(right.w) * static_cast<double>(left.z) -
        static_cast<double>(right.x) * static_cast<double>(left.y) +
        static_cast<double>(right.y) * static_cast<double>(left.x) +
        static_cast<double>(right.z) * static_cast<double>(left.w);
    return ik_checked_quat_from_double(output, w, x, y, z);
}

static inline quat ik_quat_inverse_unit(quat value)
{
    return quat(-value.w, value.x, value.y, value.z);
}

static inline bool ik_checked_quat_inverse_multiply(
    quat& output, quat left, quat right)
{
    if (!ik_quat_is_unit(left) || !ik_quat_is_unit(right)) {
        return false;
    }
    return ik_checked_quat_multiply(
        output, ik_quat_inverse_unit(left), right);
}

static inline bool ik_checked_quat_rotate(
    vec3& output, quat rotation, vec3 value)
{
    if (!ik_quat_is_unit(rotation) ||
        !ik_vec3_is_runtime_value(value)) {
        return false;
    }
    const double qx = static_cast<double>(rotation.x);
    const double qy = static_cast<double>(rotation.y);
    const double qz = static_cast<double>(rotation.z);
    const double vx = static_cast<double>(value.x);
    const double vy = static_cast<double>(value.y);
    const double vz = static_cast<double>(value.z);
    const volatile double tx = 2.0 * (qy * vz - qz * vy);
    const volatile double ty = 2.0 * (qz * vx - qx * vz);
    const volatile double tz = 2.0 * (qx * vy - qy * vx);
    const volatile double cross_x = qy * tz - qz * ty;
    const volatile double cross_y = qz * tx - qx * tz;
    const volatile double cross_z = qx * ty - qy * tx;
    const volatile double output_x =
        vx + static_cast<double>(rotation.w) * tx + cross_x;
    const volatile double output_y =
        vy + static_cast<double>(rotation.w) * ty + cross_y;
    const volatile double output_z =
        vz + static_cast<double>(rotation.w) * tz + cross_z;
    return ik_checked_vec3_from_double(
        output, output_x, output_y, output_z);
}

static inline bool ik_checked_quat_angle(
    double& precise_angle, float& rounded_angle,
    quat left, quat right)
{
    if (!ik_quat_is_unit(left) || !ik_quat_is_unit(right)) {
        return false;
    }
    if (terrain_float_bits(left.w) == terrain_float_bits(right.w) &&
        terrain_float_bits(left.x) == terrain_float_bits(right.x) &&
        terrain_float_bits(left.y) == terrain_float_bits(right.y) &&
        terrain_float_bits(left.z) == terrain_float_bits(right.z)) {
        precise_angle = 0.0;
        rounded_angle = 0.0f;
        return true;
    }
    quat normalized_left;
    quat normalized_right;
    if (!ik_checked_quat_normalize(normalized_left, left) ||
        !ik_checked_quat_normalize(normalized_right, right)) {
        return false;
    }
    const volatile double dot_value =
        static_cast<double>(normalized_left.w) *
            static_cast<double>(normalized_right.w) +
        static_cast<double>(normalized_left.x) *
            static_cast<double>(normalized_right.x) +
        static_cast<double>(normalized_left.y) *
            static_cast<double>(normalized_right.y) +
        static_cast<double>(normalized_left.z) *
            static_cast<double>(normalized_right.z);
    if (!terrain_double_is_finite(dot_value)) {
        return false;
    }
    double shortest_dot = fabs(dot_value);
    shortest_dot = shortest_dot > 1.0 ? 1.0 : shortest_dot;
    const volatile double angle = 2.0 * acos(shortest_dot);
    if (!terrain_double_is_finite(angle) ||
        !ik_checked_binary32_commit(angle, rounded_angle)) {
        return false;
    }
    precise_angle = angle;
    return true;
}

static inline bool ik_checked_safe_perpendicular(
    vec3& output, vec3 direction)
{
    if (!ik_vec3_is_unit(direction)) {
        return false;
    }
    const vec3 axis = fabs(static_cast<double>(direction.x)) < 0.75
        ? vec3(1.0f, 0.0f, 0.0f)
        : vec3(0.0f, 0.0f, 1.0f);
    vec3 crossed;
    return ik_checked_cross(crossed, direction, axis) &&
           ik_checked_normalize(output, crossed);
}

static inline bool ik_checked_quat_between(
    quat& output, vec3 from, vec3 to)
{
    if (!ik_vec3_is_unit(from) || !ik_vec3_is_unit(to)) {
        return false;
    }
    double cosine = 0.0;
    if (!ik_checked_dot(cosine, from, to)) {
        return false;
    }
    cosine = cosine < -1.0 ? -1.0 : (cosine > 1.0 ? 1.0 : cosine);
    if (cosine < -0.999999) {
        vec3 axis;
        if (!ik_checked_safe_perpendicular(axis, from)) {
            return false;
        }
        quat candidate(0.0f, axis.x, axis.y, axis.z);
        if (!ik_quat_is_unit(candidate)) {
            return false;
        }
        output = candidate;
        return true;
    }
    if (cosine > 0.999999) {
        output = quat();
        return true;
    }
    vec3 crossed;
    return ik_checked_cross(crossed, from, to) &&
           ik_checked_quat_from_double(
               output, 1.0 + cosine,
               static_cast<double>(crossed.x),
               static_cast<double>(crossed.y),
               static_cast<double>(crossed.z));
}

static inline bool ik_clamp_quat_bits_equal(
    quat left, quat right)
{
    return terrain_float_bits(left.w) == terrain_float_bits(right.w) &&
           terrain_float_bits(left.x) == terrain_float_bits(right.x) &&
           terrain_float_bits(left.y) == terrain_float_bits(right.y) &&
           terrain_float_bits(left.z) == terrain_float_bits(right.z);
}

static inline bool ik_clamp_local_delta(
    IKClampResult& output,
    quat baseline,
    quat desired,
    float maximum_radians
#if defined(G1_IK_ENABLE_TEST_SEAMS)
    , uint32_t attempt_limit = IKClampBoundaryAttemptCapacity,
    IKClampBoundaryAudit* audit = nullptr
#endif
    )
{
#if defined(G1_IK_ENABLE_TEST_SEAMS)
    if (attempt_limit > IKClampBoundaryAttemptCapacity) {
        return false;
    }
#else
    const uint32_t attempt_limit =
        IKClampBoundaryAttemptCapacity;
#endif
    if (!ik_quat_is_unit(baseline) || !ik_quat_is_unit(desired) ||
        !terrain_float_is_positive_normal(maximum_radians) ||
        maximum_radians > PIf) {
        return false;
    }

#if defined(G1_IK_ENABLE_TEST_SEAMS)
    IKClampBoundaryAudit candidate_audit = {};
#endif

    if (ik_clamp_quat_bits_equal(baseline, desired)) {
        IKClampResult candidate = {};
        candidate.value = baseline;
        candidate.requested_radians = 0.0f;
        candidate.actual_radians = 0.0f;
        candidate.limited = false;
        output = candidate;
#if defined(G1_IK_ENABLE_TEST_SEAMS)
        if (audit != nullptr) {
            *audit = candidate_audit;
        }
#endif
        return true;
    }

    quat normalized_baseline;
    quat normalized_desired;
    if (!ik_checked_quat_normalize(normalized_baseline, baseline) ||
        !ik_checked_quat_normalize(normalized_desired, desired)) {
        return false;
    }

    const volatile double raw_dot =
        static_cast<double>(normalized_baseline.w) *
            static_cast<double>(normalized_desired.w) +
        static_cast<double>(normalized_baseline.x) *
            static_cast<double>(normalized_desired.x) +
        static_cast<double>(normalized_baseline.y) *
            static_cast<double>(normalized_desired.y) +
        static_cast<double>(normalized_baseline.z) *
            static_cast<double>(normalized_desired.z);
    if (!terrain_double_is_finite(raw_dot)) {
        return false;
    }
    quat shortest = raw_dot < 0.0
        ? -normalized_desired
        : normalized_desired;
    quat canonical_desired;
    if (!ik_checked_quat_normalize(
            canonical_desired, shortest)) {
        return false;
    }
    double cosine = fabs(raw_dot);
    cosine = cosine > 1.0 ? 1.0 : cosine;
    const volatile double theta = acos(cosine);
    const volatile double construction_requested = 2.0 * theta;
    if (!terrain_double_is_finite(theta) ||
        !terrain_double_is_finite(construction_requested)) {
        return false;
    }

    double precise_requested = 0.0;
    float requested = 0.0f;
    if (!ik_checked_quat_angle(
            precise_requested, requested,
            baseline, canonical_desired)) {
        return false;
    }

    const bool limited =
        precise_requested > static_cast<double>(maximum_radians);
    if (!limited) {
        IKClampResult candidate = {};
        candidate.value = canonical_desired;
        candidate.requested_radians = requested;
        candidate.actual_radians = requested;
        candidate.limited = false;
        output = candidate;
#if defined(G1_IK_ENABLE_TEST_SEAMS)
        if (audit != nullptr) {
            *audit = candidate_audit;
        }
#endif
        return true;
    }

    if (construction_requested <= 0.0) {
        return false;
    }
    const volatile double sine_theta = sin(theta);
    if (!terrain_double_is_finite(sine_theta) ||
        sine_theta <= 0.0) {
        return false;
    }
    double interpolation_angle =
        static_cast<double>(maximum_radians);
    quat candidate_value;
    double precise_actual = 0.0;
    float actual = 0.0f;
    bool bounded_candidate_found = false;
    for (uint32_t attempt = 0U;
         attempt < attempt_limit; ++attempt) {
        const volatile double alpha =
            interpolation_angle / construction_requested;
        if (!terrain_double_is_finite(alpha) || alpha <= 0.0 ||
            alpha >= 1.0) {
            return false;
        }
        const volatile double baseline_weight =
            sin((1.0 - alpha) * theta) / sine_theta;
        const volatile double desired_weight =
            sin(alpha * theta) / sine_theta;
        if (!ik_checked_quat_from_double(
                candidate_value,
                baseline_weight *
                        static_cast<double>(normalized_baseline.w) +
                    desired_weight * static_cast<double>(shortest.w),
                baseline_weight *
                        static_cast<double>(normalized_baseline.x) +
                    desired_weight * static_cast<double>(shortest.x),
                baseline_weight *
                        static_cast<double>(normalized_baseline.y) +
                    desired_weight * static_cast<double>(shortest.y),
                baseline_weight *
                        static_cast<double>(normalized_baseline.z) +
                    desired_weight * static_cast<double>(shortest.z)) ||
            !ik_checked_quat_angle(
                precise_actual, actual,
                baseline, candidate_value)) {
            return false;
        }
        const bool accepted =
            precise_actual > 0.0 && actual > 0.0f &&
            precise_actual <=
                static_cast<double>(maximum_radians) &&
            actual <= maximum_radians;
#if defined(G1_IK_ENABLE_TEST_SEAMS)
        candidate_audit.attempts[attempt].interpolation_angle =
            interpolation_angle;
        candidate_audit.attempts[attempt].precise_angle =
            precise_actual;
        candidate_audit.attempts[attempt].rounded_angle = actual;
        candidate_audit.attempts[attempt].accepted = accepted;
        candidate_audit.attempt_count = attempt + 1U;
#endif
        if (accepted) {
            bounded_candidate_found = true;
            break;
        }

        const float materialized_interpolation =
            static_cast<float>(interpolation_angle);
        const double lower_binary32_angle = static_cast<double>(
            std::nextafter(materialized_interpolation, 0.0f));
        if (!terrain_double_is_finite(lower_binary32_angle) ||
            lower_binary32_angle <= 0.0 ||
            lower_binary32_angle >= interpolation_angle) {
            break;
        }

        const double measured =
            precise_actual > static_cast<double>(actual)
                ? precise_actual
                : static_cast<double>(actual);
        if (!terrain_double_is_finite(measured) || measured < 0.0) {
            return false;
        }
        double next_angle = lower_binary32_angle;
        if (measured > static_cast<double>(maximum_radians)) {
            next_angle = interpolation_angle *
                (static_cast<double>(maximum_radians) / measured);
            if (!terrain_double_is_finite(next_angle) ||
                next_angle <= 0.0) {
                return false;
            }
            if (next_angle >= lower_binary32_angle) {
                next_angle = lower_binary32_angle;
            }
        }
        if (next_angle >= interpolation_angle) {
            next_angle = std::nextafter(interpolation_angle, 0.0);
        }
        if (!terrain_double_is_finite(next_angle) ||
            next_angle <= 0.0 ||
            next_angle >= interpolation_angle) {
            break;
        }
        interpolation_angle = next_angle;
    }

    if (!bounded_candidate_found) {
#if defined(G1_IK_ENABLE_TEST_SEAMS)
        candidate_audit.finite_exhausted = true;
        if (audit != nullptr) {
            *audit = candidate_audit;
        }
#endif
        return false;
    }

    if (precise_actual <= 0.0 || actual <= 0.0f ||
        precise_actual > static_cast<double>(maximum_radians) ||
        actual > maximum_radians) {
        return false;
    }
    IKClampResult candidate = {};
    candidate.value = candidate_value;
    candidate.requested_radians = requested;
    candidate.actual_radians = actual;
    candidate.limited = true;
    output = candidate;
#if defined(G1_IK_ENABLE_TEST_SEAMS)
    if (audit != nullptr) {
        *audit = candidate_audit;
    }
#endif
    return true;
}

#if defined(G1_IK_ENABLE_TEST_SEAMS)
static inline bool ik_clamp_local_delta_audited_for_test(
    IKClampResult& output,
    IKClampBoundaryAudit& audit,
    quat baseline,
    quat desired,
    float maximum_radians,
    uint32_t attempt_limit)
{
    const uintptr_t output_begin =
        reinterpret_cast<uintptr_t>(&output);
    const uintptr_t audit_begin =
        reinterpret_cast<uintptr_t>(&audit);
    if (output_begin > UINTPTR_MAX - sizeof(output) ||
        audit_begin > UINTPTR_MAX - sizeof(audit)) {
        return false;
    }
    const uintptr_t output_end =
        output_begin + sizeof(output);
    const uintptr_t audit_end =
        audit_begin + sizeof(audit);
    if (output_begin < audit_end &&
        audit_begin < output_end) {
        return false;
    }
    return ik_clamp_local_delta(
        output, baseline, desired, maximum_radians,
        attempt_limit, &audit);
}
#endif

static inline bool ik_checked_radial_target(
    vec3& output,
    vec3 root,
    vec3 direction,
    double distance)
{
    if (!ik_vec3_is_runtime_value(root) ||
        !ik_vec3_is_unit(direction) ||
        !terrain_double_is_finite(distance) || distance < 0.0) {
        return false;
    }
    return ik_checked_vec3_from_double(
        output,
        static_cast<double>(root.x) +
            static_cast<double>(direction.x) * distance,
        static_cast<double>(root.y) +
            static_cast<double>(direction.y) * distance,
        static_cast<double>(root.z) +
            static_cast<double>(direction.z) * distance);
}

static inline bool ik_checked_materialize_shell_target(
    vec3& output,
    double& precise_distance,
    float& rounded_distance,
    vec3 root,
    vec3 direction,
    double requested_distance,
    double minimum_distance,
    double maximum_distance)
{
    if (!ik_vec3_is_runtime_value(root) ||
        !ik_vec3_is_unit(direction) ||
        !terrain_double_is_finite(requested_distance) ||
        !terrain_double_is_finite(minimum_distance) ||
        !terrain_double_is_finite(maximum_distance) ||
        requested_distance < 0.0 || minimum_distance < 0.0 ||
        maximum_distance < minimum_distance ||
        requested_distance < minimum_distance ||
        requested_distance > maximum_distance) {
        return false;
    }

    double distance = requested_distance;
    for (int attempt = 0; attempt < 64; ++attempt) {
        vec3 candidate;
        double measured = 0.0;
        float measured_float = 0.0f;
        if (!ik_checked_radial_target(
                candidate, root, direction, distance) ||
            !ik_checked_distance_precise(
                measured, measured_float, candidate, root)) {
            return false;
        }
        if (measured >= minimum_distance &&
            measured <= maximum_distance) {
            output = candidate;
            precise_distance = measured;
            rounded_distance = measured_float;
            return true;
        }

        double corrected = distance;
        if (measured > maximum_distance) {
            if (measured <= 0.0) {
                return false;
            }
            corrected = distance * (maximum_distance / measured);
            float rounded_corrected = 0.0f;
            if (!ik_checked_binary32_commit(
                    corrected, rounded_corrected)) {
                return false;
            }
            corrected = static_cast<double>(std::nextafter(
                rounded_corrected, 0.0f));
            if (corrected < minimum_distance) {
                corrected = minimum_distance;
            }
            if (corrected >= distance) {
                corrected = std::nextafter(distance, 0.0);
            }
        } else {
            if (measured > 0.0) {
                corrected = distance * (minimum_distance / measured);
            } else {
                corrected = minimum_distance;
            }
            float rounded_corrected = 0.0f;
            if (!ik_checked_binary32_commit(
                    corrected, rounded_corrected)) {
                return false;
            }
            corrected = static_cast<double>(std::nextafter(
                rounded_corrected,
                std::numeric_limits<float>::infinity()));
            if (corrected > maximum_distance) {
                corrected = maximum_distance;
            }
            if (corrected <= distance) {
                corrected = std::nextafter(
                    distance, std::numeric_limits<double>::infinity());
            }
        }
        if (!terrain_double_is_finite(corrected) ||
            corrected < minimum_distance ||
            corrected > maximum_distance || corrected == distance) {
            return false;
        }
        distance = corrected;
    }
    return false;
}

bool ik_project_target(
    IKTargetProjection& output,
    vec3 root,
    vec3 middle,
    vec3 end,
    vec3 requested,
    float reach_buffer_m);

static inline bool ik_select_bend_direction(
    IKBendSelection& output,
    vec3 current_upper,
    vec3 target_direction,
    vec3 hinge_axis_world)
{
    if (!ik_vec3_is_runtime_value(current_upper) ||
        !ik_vec3_is_unit(target_direction) ||
        !ik_vec3_is_unit(hinge_axis_world)) {
        return false;
    }
    double along = 0.0;
    if (!ik_checked_dot(along, current_upper, target_direction)) {
        return false;
    }
    const volatile double projection_x =
        static_cast<double>(current_upper.x) -
        static_cast<double>(target_direction.x) * along;
    const volatile double projection_y =
        static_cast<double>(current_upper.y) -
        static_cast<double>(target_direction.y) * along;
    const volatile double projection_z =
        static_cast<double>(current_upper.z) -
        static_cast<double>(target_direction.z) * along;
    const double projection_components[3] = {
        projection_x, projection_y, projection_z
    };
    double projection_length = 0.0;
    float projection_float = 0.0f;
    if (!ik_checked_norm_components(
            projection_length, projection_float,
            projection_components, 3, true)) {
        return false;
    }

    IKBendSelection candidate = {};
    candidate.current_projection_length = projection_float;
    if (projection_length >= static_cast<double>(1.0e-6f)) {
        if (!ik_checked_vec3_from_double(
                candidate.direction,
                projection_x / projection_length,
                projection_y / projection_length,
                projection_z / projection_length) ||
            !ik_vec3_is_unit(candidate.direction)) {
            return false;
        }
        candidate.used_current_projection = true;
    } else {
        const volatile double fallback_x =
            static_cast<double>(hinge_axis_world.y) *
                    static_cast<double>(target_direction.z) -
                static_cast<double>(hinge_axis_world.z) *
                    static_cast<double>(target_direction.y);
        const volatile double fallback_y =
            static_cast<double>(hinge_axis_world.z) *
                    static_cast<double>(target_direction.x) -
                static_cast<double>(hinge_axis_world.x) *
                    static_cast<double>(target_direction.z);
        const volatile double fallback_z =
            static_cast<double>(hinge_axis_world.x) *
                    static_cast<double>(target_direction.y) -
                static_cast<double>(hinge_axis_world.y) *
                    static_cast<double>(target_direction.x);
        const double fallback_components[3] = {
            fallback_x, fallback_y, fallback_z
        };
        double fallback_length = 0.0;
        float fallback_float = 0.0f;
        if (!ik_checked_norm_components(
                fallback_length, fallback_float,
                fallback_components, 3, true)) {
            return false;
        }
        candidate.used_hinge_fallback = true;
        if (fallback_length >= static_cast<double>(1.0e-6f)) {
            if (!ik_checked_vec3_from_double(
                    candidate.direction,
                    fallback_x / fallback_length,
                    fallback_y / fallback_length,
                    fallback_z / fallback_length) ||
                !ik_vec3_is_unit(candidate.direction)) {
                return false;
            }
        } else {
            if (!ik_checked_safe_perpendicular(
                    candidate.direction, target_direction)) {
                return false;
            }
            candidate.used_safe_perpendicular = true;
        }

        const volatile double agreement_x =
            static_cast<double>(candidate.direction.x) * projection_x;
        const volatile double agreement_y =
            static_cast<double>(candidate.direction.y) * projection_y;
        const volatile double agreement_z =
            static_cast<double>(candidate.direction.z) * projection_z;
        const volatile double agreement_xy = agreement_x + agreement_y;
        const volatile double agreement = agreement_xy + agreement_z;
        if (!terrain_double_is_finite(agreement)) {
            return false;
        }
        if (projection_length > static_cast<double>(1.0e-8f) &&
            agreement < 0.0) {
            candidate.direction = vec3(
                -candidate.direction.x,
                -candidate.direction.y,
                -candidate.direction.z);
            candidate.sign_flipped = true;
        }
    }
    if (!ik_vec3_is_unit(candidate.direction)) {
        return false;
    }
    output = candidate;
    return true;
}

static inline bool ik_two_bone_bounded(
    IKTwoBoneResult& output,
    quat root_local_before,
    quat middle_local_before,
    vec3 root,
    vec3 middle,
    vec3 end,
    vec3 requested_target,
    vec3 hinge_axis_world,
    quat root_global,
    quat middle_global,
    quat root_parent_global,
    float reach_buffer_m,
    float maximum_correction_radians)
{
    if (!ik_quat_is_unit(root_local_before) ||
        !ik_quat_is_unit(middle_local_before) ||
        !ik_quat_is_unit(root_global) ||
        !ik_quat_is_unit(middle_global) ||
        !ik_quat_is_unit(root_parent_global) ||
        !ik_vec3_is_unit(hinge_axis_world) ||
        !terrain_float_is_positive_normal(maximum_correction_radians) ||
        maximum_correction_radians > PIf) {
        return false;
    }

    IKTargetProjection projection = {};
    if (!ik_project_target(
            projection, root, middle, end,
            requested_target, reach_buffer_m)) {
        return false;
    }
    vec3 current_upper;
    vec3 current_lower;
    double upper = 0.0;
    double lower = 0.0;
    float upper_float = 0.0f;
    float lower_float = 0.0f;
    if (!ik_checked_vec3_subtract(current_upper, middle, root) ||
        !ik_checked_vec3_subtract(current_lower, end, middle) ||
        !ik_checked_distance_precise(
            upper, upper_float, middle, root) ||
        !ik_checked_distance_precise(
            lower, lower_float, end, middle)) {
        return false;
    }

    double projected_motion = 0.0;
    double requested_motion = 0.0;
    float projected_motion_float = 0.0f;
    float requested_motion_float = 0.0f;
    if (!ik_checked_distance_precise(
            projected_motion, projected_motion_float,
            projection.clamped_target, end) ||
        !ik_checked_distance_precise(
            requested_motion, requested_motion_float,
            requested_target, end)) {
        return false;
    }
    if (projected_motion <= static_cast<double>(1.0e-6f)) {
        double current_distance = 0.0;
        float current_float = 0.0f;
        if (!ik_checked_distance_precise(
                current_distance, current_float, end, root)) {
            return false;
        }
        vec3 current_direction(0.0f, -1.0f, 0.0f);
        if (current_distance > 1.0e-7) {
            if (!ik_checked_vec3_from_double(
                    current_direction,
                    (static_cast<double>(end.x) -
                        static_cast<double>(root.x)) /
                        current_distance,
                    (static_cast<double>(end.y) -
                        static_cast<double>(root.y)) /
                        current_distance,
                    (static_cast<double>(end.z) -
                        static_cast<double>(root.z)) /
                        current_distance) ||
                !ik_vec3_is_unit(current_direction)) {
                return false;
            }
        }
        IKBendSelection current_bend = {};
        if (!ik_select_bend_direction(
                current_bend, current_upper, current_direction,
                hinge_axis_world)) {
            return false;
        }
        projection.clamped_target = end;
        projection.clamped_distance_m = current_float;
        if (requested_motion <= static_cast<double>(1.0e-6f)) {
            projection.reachable = true;
        }
        IKTwoBoneResult candidate = {};
        candidate.applied = true;
        candidate.reachable = projection.reachable;
        candidate.correction_limited = false;
        candidate.root_local = root_local_before;
        candidate.middle_local = middle_local_before;
        candidate.root_correction_radians = 0.0f;
        candidate.middle_correction_radians = 0.0f;
        candidate.target = projection;
        candidate.bend = current_bend;
        output = candidate;
        return true;
    }

    double target_distance = 0.0;
    float target_float = 0.0f;
    if (!ik_checked_distance_precise(
            target_distance, target_float,
            projection.clamped_target, root) ||
        target_distance <= 0.0) {
        return false;
    }
    vec3 target_direction;
    if (!ik_checked_vec3_from_double(
            target_direction,
            (static_cast<double>(projection.clamped_target.x) -
                static_cast<double>(root.x)) / target_distance,
            (static_cast<double>(projection.clamped_target.y) -
                static_cast<double>(root.y)) / target_distance,
            (static_cast<double>(projection.clamped_target.z) -
                static_cast<double>(root.z)) / target_distance) ||
        !ik_vec3_is_unit(target_direction)) {
        return false;
    }
    IKBendSelection bend = {};
    if (!ik_select_bend_direction(
            bend, current_upper, target_direction,
            hinge_axis_world)) {
        return false;
    }

    const double upper_d = upper;
    const double lower_d = lower;
    const double geometric_minimum = fabs(upper_d - lower_d);
    const double geometric_maximum = upper_d + lower_d;
    const double projected_distance = target_distance;
    const double target_d =
        projected_distance < geometric_minimum
            ? geometric_minimum
            : (projected_distance > geometric_maximum
                   ? geometric_maximum
                   : projected_distance);
    if (target_d <= 0.0) {
        return false;
    }
    const volatile double upper_square = upper_d * upper_d;
    const volatile double lower_square = lower_d * lower_d;
    const volatile double target_square = target_d * target_d;
    const volatile double numerator =
        upper_square + target_square - lower_square;
    const volatile double denominator = 2.0 * target_d;
    if (!terrain_double_is_finite(numerator) ||
        !terrain_double_is_finite(denominator) || denominator <= 0.0) {
        return false;
    }
    const volatile double knee_along = numerator / denominator;
    const volatile double along_square = knee_along * knee_along;
    double radicand = upper_square - along_square;
    const volatile double radicand_scale =
        upper_square + lower_square + target_square;
    const volatile double radicand_tolerance =
        64.0 * DBL_EPSILON * radicand_scale;
    if (radicand < 0.0 &&
        radicand >= -radicand_tolerance) {
        radicand = 0.0;
    }
    if (!terrain_double_is_finite(knee_along) ||
        !terrain_double_is_finite(radicand) || radicand < 0.0 ||
        radicand > static_cast<double>(FLT_MAX)) {
        return false;
    }
    const volatile double knee_height = sqrt(radicand);
    if (!terrain_double_is_finite(knee_height)) {
        return false;
    }

    vec3 desired_knee;
    if (!ik_checked_vec3_from_double(
            desired_knee,
            static_cast<double>(root.x) +
                static_cast<double>(target_direction.x) * knee_along +
                static_cast<double>(bend.direction.x) * knee_height,
            static_cast<double>(root.y) +
                static_cast<double>(target_direction.y) * knee_along +
                static_cast<double>(bend.direction.y) * knee_height,
            static_cast<double>(root.z) +
                static_cast<double>(target_direction.z) * knee_along +
                static_cast<double>(bend.direction.z) * knee_height)) {
        return false;
    }
    vec3 desired_upper;
    vec3 desired_lower;
    vec3 current_upper_direction;
    vec3 desired_upper_direction;
    vec3 desired_lower_direction;
    if (!ik_checked_vec3_subtract(desired_upper, desired_knee, root) ||
        !ik_checked_vec3_subtract(
            desired_lower, projection.clamped_target, desired_knee) ||
        !ik_checked_normalize(current_upper_direction, current_upper) ||
        !ik_checked_normalize(desired_upper_direction, desired_upper) ||
        !ik_checked_normalize(desired_lower_direction, desired_lower)) {
        return false;
    }

    quat root_delta;
    quat desired_root_global;
    vec3 root_rotated_lower;
    vec3 root_rotated_lower_direction;
    quat middle_delta;
    quat rotated_middle_global;
    quat desired_middle_global;
    quat desired_root_local;
    quat desired_middle_local;
    if (!ik_checked_quat_between(
            root_delta,
            current_upper_direction, desired_upper_direction) ||
        !ik_checked_quat_multiply(
            desired_root_global, root_delta, root_global) ||
        !ik_checked_quat_rotate(
            root_rotated_lower, root_delta, current_lower) ||
        !ik_checked_normalize(
            root_rotated_lower_direction, root_rotated_lower) ||
        !ik_checked_quat_between(
            middle_delta,
            root_rotated_lower_direction, desired_lower_direction) ||
        !ik_checked_quat_multiply(
            rotated_middle_global, root_delta, middle_global) ||
        !ik_checked_quat_multiply(
            desired_middle_global,
            middle_delta, rotated_middle_global) ||
        !ik_checked_quat_inverse_multiply(
            desired_root_local,
            root_parent_global, desired_root_global) ||
        !ik_checked_quat_inverse_multiply(
            desired_middle_local,
            desired_root_global, desired_middle_global)) {
        return false;
    }

    IKClampResult root_clamp = {};
    IKClampResult middle_clamp = {};
    if (!ik_clamp_local_delta(
            root_clamp, root_local_before,
            desired_root_local, maximum_correction_radians) ||
        !ik_clamp_local_delta(
            middle_clamp, middle_local_before,
            desired_middle_local, maximum_correction_radians)) {
        return false;
    }

    IKTwoBoneResult candidate = {};
    candidate.applied = true;
    candidate.reachable = projection.reachable;
    candidate.correction_limited =
        root_clamp.limited || middle_clamp.limited;
    candidate.root_local = root_clamp.value;
    candidate.middle_local = middle_clamp.value;
    candidate.root_correction_radians = root_clamp.actual_radians;
    candidate.middle_correction_radians = middle_clamp.actual_radians;
    candidate.target = projection;
    candidate.bend = bend;
    output = candidate;
    return true;
}
