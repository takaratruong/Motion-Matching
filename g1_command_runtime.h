#pragma once

#include "ik.h"
#include "terrain_runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

static constexpr int G1CommandTrajectorySampleCount = 4;

struct G1CommandIntent
{
    vec3 requested_velocity;
    quat desired_heading;
};

struct G1CommandSnapshot
{
    G1CommandIntent intent;
    vec3 applied_velocity;
    vec3 predicted_desired_velocities[G1CommandTrajectorySampleCount];
    vec3 predicted_root_positions[G1CommandTrajectorySampleCount];
    quat predicted_root_rotations[G1CommandTrajectorySampleCount];
    quat predicted_desired_headings[G1CommandTrajectorySampleCount];
};

struct G1TestHeadingOverride
{
    bool active = false;
    quat heading;
};

static inline bool g1_command_memory_range(
    uintptr_t& begin,
    uintptr_t& end,
    const void* memory,
    std::size_t size)
{
    if (memory == NULL || size == 0) {
        begin = 0;
        end = 0;
        return memory != NULL || size == 0;
    }
    begin = reinterpret_cast<uintptr_t>(memory);
    if (begin > std::numeric_limits<uintptr_t>::max() - size) {
        return false;
    }
    end = begin + size;
    return true;
}

static inline bool g1_command_memory_overlaps(
    const void* first,
    std::size_t first_size,
    const void* second,
    std::size_t second_size)
{
    uintptr_t first_begin = 0;
    uintptr_t first_end = 0;
    uintptr_t second_begin = 0;
    uintptr_t second_end = 0;
    if (first == NULL || second == NULL || first_size == 0 || second_size == 0) {
        return false;
    }
    if (!g1_command_memory_range(
            first_begin, first_end, first, first_size) ||
        !g1_command_memory_range(
            second_begin, second_end, second, second_size)) {
        return true;
    }
    return first_begin < second_end && second_begin < first_end;
}

static inline bool g1_command_failure(
    const void* output,
    std::size_t output_size,
    char* error,
    int error_capacity,
    const char* message)
{
    const bool diagnostic_alias =
        error != NULL && error_capacity > 0 &&
        g1_command_memory_overlaps(
            output,
            output_size,
            error,
            static_cast<std::size_t>(error_capacity));
    if (!diagnostic_alias) {
        terrain_error(error, error_capacity, "%s", message);
    }
    return false;
}

static inline bool g1_command_vec3_is_finite(vec3 value)
{
    return terrain_float_is_finite(value.x) &&
           terrain_float_is_finite(value.y) &&
           terrain_float_is_finite(value.z);
}

static inline bool g1_command_vec3_is_canonical(vec3 value)
{
    return terrain_float_is_normal_or_positive_zero(value.x) &&
           terrain_float_is_normal_or_positive_zero(value.y) &&
           terrain_float_is_normal_or_positive_zero(value.z);
}

static inline vec3 g1_command_vec3_canonicalize(vec3 value)
{
    return vec3(
        terrain_runtime_canonicalize_output(value.x),
        terrain_runtime_canonicalize_output(value.y),
        terrain_runtime_canonicalize_output(value.z));
}

static inline bool g1_test_heading_override_parse(
    G1TestHeadingOverride& output,
    const char* text,
    char* error,
    int error_capacity)
{
    if (error != NULL && error_capacity > 0 &&
        g1_command_memory_overlaps(
            &output,
            sizeof(output),
            error,
            static_cast<std::size_t>(error_capacity))) {
        return false;
    }

    G1TestHeadingOverride candidate;
    if (text == NULL) {
        output = candidate;
        return true;
    }

    static const struct
    {
        const char* text;
        quat heading;
    } values[] = {
        {"forward", quat(1.0f, 0.0f, 0.0f, 0.0f)},
        {"backward", quat(0.0f, 0.0f, 1.0f, 0.0f)},
        {"positive-x", quat(0.707106769f, 0.0f, 0.707106769f, 0.0f)},
        {"negative-x", quat(0.707106769f, 0.0f, -0.707106769f, 0.0f)},
        {"diagonal-positive-x",
         quat(0.923879504f, 0.0f, 0.382683426f, 0.0f)},
        {"diagonal-negative-x",
         quat(0.923879504f, 0.0f, -0.382683426f, 0.0f)},
    };
    for (const auto& value : values) {
        if (std::strcmp(text, value.text) == 0) {
            candidate.active = true;
            candidate.heading = value.heading;
            output = candidate;
            return true;
        }
    }
    return g1_command_failure(
        &output,
        sizeof(output),
        error,
        error_capacity,
        "MM_TEST_HEADING must be one of the six exact named headings");
}

static inline bool g1_command_snapshot_is_valid(
    const G1CommandSnapshot& value)
{
    if (!g1_command_vec3_is_canonical(value.intent.requested_velocity) ||
        !ik_quat_is_unit(value.intent.desired_heading) ||
        !g1_command_vec3_is_canonical(value.applied_velocity)) {
        return false;
    }
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        if (!g1_command_vec3_is_canonical(
                value.predicted_desired_velocities[index]) ||
            !g1_command_vec3_is_canonical(
                value.predicted_root_positions[index]) ||
            !ik_quat_is_unit(value.predicted_root_rotations[index]) ||
            !ik_quat_is_unit(value.predicted_desired_headings[index])) {
            return false;
        }
    }
    return true;
}

template<typename T>
static inline bool g1_command_slice_aliases_output(
    const G1CommandSnapshot& output,
    const slice1d<T> values)
{
    if (values.size != G1CommandTrajectorySampleCount || values.data == NULL) {
        return false;
    }
    return g1_command_memory_overlaps(
        &output,
        sizeof(output),
        values.data,
        static_cast<std::size_t>(values.size) * sizeof(T));
}

static inline bool g1_command_snapshot_build(
    G1CommandSnapshot& output,
    G1CommandIntent intent,
    vec3 applied_velocity,
    const slice1d<vec3> predicted_desired_velocities,
    const slice1d<vec3> predicted_root_positions,
    const slice1d<quat> predicted_root_rotations,
    const slice1d<quat> predicted_desired_headings,
    char* error,
    int error_capacity)
{
    if (error != NULL && error_capacity > 0 &&
        g1_command_memory_overlaps(
            &output,
            sizeof(output),
            error,
            static_cast<std::size_t>(error_capacity))) {
        return false;
    }
    if (predicted_desired_velocities.size !=
            G1CommandTrajectorySampleCount ||
        predicted_root_positions.size != G1CommandTrajectorySampleCount ||
        predicted_root_rotations.size != G1CommandTrajectorySampleCount ||
        predicted_desired_headings.size != G1CommandTrajectorySampleCount ||
        predicted_desired_velocities.data == NULL ||
        predicted_root_positions.data == NULL ||
        predicted_root_rotations.data == NULL ||
        predicted_desired_headings.data == NULL) {
        return g1_command_failure(
            &output,
            sizeof(output),
            error,
            error_capacity,
            "G1 command snapshot requires four stored trajectory samples");
    }
    if (g1_command_slice_aliases_output(
            output, predicted_desired_velocities) ||
        g1_command_slice_aliases_output(output, predicted_root_positions) ||
        g1_command_slice_aliases_output(output, predicted_root_rotations) ||
        g1_command_slice_aliases_output(output, predicted_desired_headings)) {
        return g1_command_failure(
            &output,
            sizeof(output),
            error,
            error_capacity,
            "G1 command snapshot output must not alias trajectory inputs");
    }
    if (!g1_command_vec3_is_finite(intent.requested_velocity) ||
        !ik_quat_is_unit(intent.desired_heading) ||
        !g1_command_vec3_is_finite(applied_velocity)) {
        return g1_command_failure(
            &output,
            sizeof(output),
            error,
            error_capacity,
            "G1 command intent or applied velocity is invalid");
    }

    G1CommandSnapshot candidate;
    candidate.intent.requested_velocity =
        g1_command_vec3_canonicalize(intent.requested_velocity);
    candidate.intent.desired_heading = intent.desired_heading;
    candidate.applied_velocity =
        g1_command_vec3_canonicalize(applied_velocity);
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        const vec3 velocity = predicted_desired_velocities(index);
        const vec3 position = predicted_root_positions(index);
        const quat rotation = predicted_root_rotations(index);
        const quat heading = predicted_desired_headings(index);
        if (!g1_command_vec3_is_finite(velocity) ||
            !g1_command_vec3_is_finite(position) ||
            !ik_quat_is_unit(rotation) || !ik_quat_is_unit(heading)) {
            return g1_command_failure(
                &output,
                sizeof(output),
                error,
                error_capacity,
                "G1 command trajectory sample is invalid");
        }
        candidate.predicted_desired_velocities[index] =
            g1_command_vec3_canonicalize(velocity);
        candidate.predicted_root_positions[index] =
            g1_command_vec3_canonicalize(position);
        candidate.predicted_root_rotations[index] = rotation;
        candidate.predicted_desired_headings[index] = heading;
    }
    if (!g1_command_snapshot_is_valid(candidate)) {
        return g1_command_failure(
            &output,
            sizeof(output),
            error,
            error_capacity,
            "G1 command snapshot candidate failed publication validation");
    }
    output = candidate;
    return true;
}
