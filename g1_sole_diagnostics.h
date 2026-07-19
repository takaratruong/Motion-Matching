#pragma once

#include "g1_kinematic_contract.h"
#include "g1_surface_query.h"

#include <cfloat>
#include <cmath>
#include <cstdarg>
#include <cstdio>

enum G1SoleDiagnosticStatus
{
    G1SoleDiagnosticOk = 0,
    G1SoleDiagnosticOutsideDomain,
    G1SoleDiagnosticInvalidInput,
    G1SoleDiagnosticInvalidField,
    G1SoleDiagnosticInvalidSurface,
    G1SoleDiagnosticArithmeticFailure,
};

struct G1SoleDiagnosticSnapshot
{
    vec3 world_points[2][4];
    float surface_heights[2][4] = {};
    float clearances[2][4] = {};
    vec3 sole_centroids[2];
    float minimum_clearance[2] = {};
    float global_minimum_clearance = 0.0f;
    float stance_slip[2] = {};
    bool contact[2] = {};
    bool slip_reset[2] = {};
    int scene_generation = 0;
};

struct G1SoleDiagnosticState
{
    vec3 previous_centroid[2];
    float accumulated_slip[2] = {};
    bool has_previous_centroid[2] = {};
    bool previous_contact[2] = {};
    bool has_scene_generation = false;
    int scene_generation = 0;
};

static inline G1SoleDiagnosticStatus g1_sole_diagnostic_error(
    G1SoleDiagnosticStatus status,
    char* output,
    int capacity,
    const char* format,
    ...)
{
    if (output != NULL && capacity > 0) {
        va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(
            output, static_cast<size_t>(capacity), format, arguments);
        va_end(arguments);
    }
    return status;
}

static inline bool g1_sole_vec3_is_finite(vec3 value)
{
    return terrain_float_is_finite(value.x) &&
           terrain_float_is_finite(value.y) &&
           terrain_float_is_finite(value.z);
}

static inline bool g1_sole_quat_is_unit(quat value)
{
    if (!terrain_float_is_finite(value.w) ||
        !terrain_float_is_finite(value.x) ||
        !terrain_float_is_finite(value.y) ||
        !terrain_float_is_finite(value.z)) {
        return false;
    }
    const volatile double square_w =
        static_cast<double>(value.w) * static_cast<double>(value.w);
    const volatile double square_x =
        static_cast<double>(value.x) * static_cast<double>(value.x);
    const volatile double square_y =
        static_cast<double>(value.y) * static_cast<double>(value.y);
    const volatile double square_z =
        static_cast<double>(value.z) * static_cast<double>(value.z);
    const volatile double first_sum = square_w + square_x;
    const volatile double second_sum = square_y + square_z;
    const volatile double square_sum = first_sum + second_sum;
    return terrain_double_is_finite(square_sum) &&
           std::fabs(square_sum - 1.0) <= 2.0e-5;
}

static inline bool g1_sole_vec3_is_exact(vec3 left, vec3 right)
{
    return terrain_float_bits(left.x) == terrain_float_bits(right.x) &&
           terrain_float_bits(left.y) == terrain_float_bits(right.y) &&
           terrain_float_bits(left.z) == terrain_float_bits(right.z);
}

static inline bool g1_sole_leg_configs_are_authoritative(
    const G1LegConfig legs[2])
{
    if (legs == NULL) return false;
    const G1LegConfig expected[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    for (int foot = 0; foot < 2; ++foot) {
        if (legs[foot].ankle != expected[foot].ankle) return false;
        for (int probe = 0; probe < 4; ++probe) {
            if (!g1_sole_vec3_is_finite(
                    legs[foot].sole_points_local[probe]) ||
                !g1_sole_vec3_is_exact(
                    legs[foot].sole_points_local[probe],
                    expected[foot].sole_points_local[probe])) {
                return false;
            }
        }
    }
    return true;
}

static inline bool g1_sole_round_output(double value, float& output)
{
    if (!terrain_double_is_finite(value) ||
        value < -static_cast<double>(FLT_MAX) ||
        value > static_cast<double>(FLT_MAX)) {
        return false;
    }
    const volatile float rounded = static_cast<float>(value);
    if (!terrain_float_is_finite(rounded)) return false;
    output = terrain_runtime_canonicalize_output(rounded);
    return true;
}

static inline bool g1_sole_world_point(
    vec3& output,
    vec3 ankle_position,
    quat ankle_rotation,
    vec3 local_point)
{
    const vec3 rotated = quat_mul_vec3(ankle_rotation, local_point);
    if (!g1_sole_vec3_is_finite(rotated)) return false;
    return g1_sole_round_output(
               static_cast<double>(ankle_position.x) + rotated.x,
               output.x) &&
           g1_sole_round_output(
               static_cast<double>(ankle_position.y) + rotated.y,
               output.y) &&
           g1_sole_round_output(
               static_cast<double>(ankle_position.z) + rotated.z,
               output.z);
}

static inline bool g1_sole_centroid(
    vec3& output,
    const vec3 points[4])
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    for (int probe = 0; probe < 4; ++probe) {
        if (!g1_sole_vec3_is_finite(points[probe])) return false;
        x += static_cast<double>(points[probe].x);
        y += static_cast<double>(points[probe].y);
        z += static_cast<double>(points[probe].z);
    }
    return g1_sole_round_output(x * 0.25, output.x) &&
           g1_sole_round_output(y * 0.25, output.y) &&
           g1_sole_round_output(z * 0.25, output.z);
}

static inline bool g1_sole_horizontal_distance(
    float& output,
    vec3 left,
    vec3 right)
{
    const volatile double dx =
        static_cast<double>(left.x) - static_cast<double>(right.x);
    const volatile double dz =
        static_cast<double>(left.z) - static_cast<double>(right.z);
    const volatile double square_x = dx * dx;
    const volatile double square_z = dz * dz;
    const volatile double square_sum = square_x + square_z;
    if (!terrain_double_is_finite(square_sum) || square_sum < 0.0) {
        return false;
    }
    const volatile double distance = std::sqrt(square_sum);
    return g1_sole_round_output(distance, output);
}

// Read-only physical sole observation. The caller's output and slip history
// are committed together only after every pose transform and G1HF/v2 query
// succeeds, so a controlled failure cannot publish a partial row or advance
// the accepted stance history.
static inline G1SoleDiagnosticStatus g1_sole_diagnostics_observe(
    G1SoleDiagnosticSnapshot& output,
    G1SoleDiagnosticState& state,
    const heightfield& field,
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    const G1LegConfig legs[2],
    const bool contacts[2],
    int scene_generation,
    bool explicit_reset,
    char* error,
    int error_capacity)
{
    if (error_capacity < 0 || contacts == NULL || scene_generation < 0 ||
        global_positions.data == NULL || global_rotations.data == NULL ||
        global_positions.size != G1_BoneCount ||
        global_rotations.size != G1_BoneCount) {
        return g1_sole_diagnostic_error(
            G1SoleDiagnosticInvalidInput,
            error,
            error_capacity,
            "physical sole diagnostics: invalid pose shape, contacts, or scene generation");
    }
    if (field.version != 2 || !terrain_heightfield_is_queryable(field)) {
        return g1_sole_diagnostic_error(
            G1SoleDiagnosticInvalidField,
            error,
            error_capacity,
            "physical sole diagnostics: invalid G1HF/v2 surface field");
    }
    if (!g1_sole_leg_configs_are_authoritative(legs)) {
        return g1_sole_diagnostic_error(
            G1SoleDiagnosticInvalidInput,
            error,
            error_capacity,
            "physical sole diagnostics: non-authoritative or non-finite sole probe configuration");
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (!g1_sole_vec3_is_finite(global_positions(bone)) ||
            !g1_sole_quat_is_unit(global_rotations(bone))) {
            return g1_sole_diagnostic_error(
                G1SoleDiagnosticInvalidInput,
                error,
                error_capacity,
                "physical sole diagnostics: final global pose is non-finite or invalid");
        }
    }

    G1SoleDiagnosticSnapshot candidate = {};
    candidate.scene_generation = scene_generation;
    for (int foot = 0; foot < 2; ++foot) {
        candidate.contact[foot] = contacts[foot];
        candidate.minimum_clearance[foot] = FLT_MAX;
        const int ankle = legs[foot].ankle;
        for (int probe = 0; probe < 4; ++probe) {
            if (!g1_sole_world_point(
                    candidate.world_points[foot][probe],
                    global_positions(ankle),
                    global_rotations(ankle),
                    legs[foot].sole_points_local[probe])) {
                return g1_sole_diagnostic_error(
                    G1SoleDiagnosticArithmeticFailure,
                    error,
                    error_capacity,
                    "physical sole diagnostics: sole transform arithmetic failed");
            }
            G1SurfaceSample surface;
            const G1SurfaceQueryStatus query_status = g1_surface_query_v2(
                surface,
                field,
                candidate.world_points[foot][probe].x,
                candidate.world_points[foot][probe].z);
            if (query_status == G1SurfaceQueryOutside) {
                return g1_sole_diagnostic_error(
                    G1SoleDiagnosticOutsideDomain,
                    error,
                    error_capacity,
                    "physical sole diagnostics: sole query is outside the G1HF/v2 domain");
            }
            if (query_status != G1SurfaceQueryValid) {
                return g1_sole_diagnostic_error(
                    G1SoleDiagnosticInvalidSurface,
                    error,
                    error_capacity,
                    "physical sole diagnostics: invalid G1HF/v2 surface query");
            }
            candidate.surface_heights[foot][probe] = surface.height;
            if (!g1_sole_round_output(
                    static_cast<double>(
                        candidate.world_points[foot][probe].y) -
                        static_cast<double>(surface.height),
                    candidate.clearances[foot][probe])) {
                return g1_sole_diagnostic_error(
                    G1SoleDiagnosticArithmeticFailure,
                    error,
                    error_capacity,
                    "physical sole diagnostics: clearance arithmetic failed");
            }
            if (candidate.clearances[foot][probe] <
                candidate.minimum_clearance[foot]) {
                candidate.minimum_clearance[foot] =
                    candidate.clearances[foot][probe];
            }
        }
        if (!g1_sole_centroid(
                candidate.sole_centroids[foot],
                candidate.world_points[foot])) {
            return g1_sole_diagnostic_error(
                G1SoleDiagnosticArithmeticFailure,
                error,
                error_capacity,
                "physical sole diagnostics: sole centroid arithmetic failed");
        }
    }
    candidate.global_minimum_clearance =
        candidate.minimum_clearance[0] < candidate.minimum_clearance[1]
            ? candidate.minimum_clearance[0]
            : candidate.minimum_clearance[1];

    G1SoleDiagnosticState state_candidate = state;
    const bool generation_reset =
        !state_candidate.has_scene_generation ||
        state_candidate.scene_generation != scene_generation;
    const bool reset_all = explicit_reset || generation_reset;
    if (reset_all) {
        for (int foot = 0; foot < 2; ++foot) {
            state_candidate.has_previous_centroid[foot] = false;
            state_candidate.previous_contact[foot] = false;
            state_candidate.accumulated_slip[foot] = 0.0f;
        }
    }
    state_candidate.has_scene_generation = true;
    state_candidate.scene_generation = scene_generation;

    for (int foot = 0; foot < 2; ++foot) {
        const bool contact_edge =
            state_candidate.previous_contact[foot] != contacts[foot];
        candidate.slip_reset[foot] = reset_all || contact_edge;
        if (!contacts[foot]) {
            state_candidate.accumulated_slip[foot] = 0.0f;
            state_candidate.has_previous_centroid[foot] = false;
            candidate.stance_slip[foot] = 0.0f;
        } else if (!state_candidate.has_previous_centroid[foot] ||
                   !state_candidate.previous_contact[foot]) {
            state_candidate.accumulated_slip[foot] = 0.0f;
            state_candidate.previous_centroid[foot] =
                candidate.sole_centroids[foot];
            state_candidate.has_previous_centroid[foot] = true;
            candidate.stance_slip[foot] = 0.0f;
            candidate.slip_reset[foot] = true;
        } else {
            float displacement = 0.0f;
            if (!g1_sole_horizontal_distance(
                    displacement,
                    candidate.sole_centroids[foot],
                    state_candidate.previous_centroid[foot]) ||
                !g1_sole_round_output(
                    static_cast<double>(
                        state_candidate.accumulated_slip[foot]) +
                        static_cast<double>(displacement),
                    state_candidate.accumulated_slip[foot])) {
                return g1_sole_diagnostic_error(
                    G1SoleDiagnosticArithmeticFailure,
                    error,
                    error_capacity,
                    "physical sole diagnostics: stance slip arithmetic failed");
            }
            state_candidate.previous_centroid[foot] =
                candidate.sole_centroids[foot];
            candidate.stance_slip[foot] =
                state_candidate.accumulated_slip[foot];
        }
        state_candidate.previous_contact[foot] = contacts[foot];
    }

    output = candidate;
    state = state_candidate;
    return G1SoleDiagnosticOk;
}
