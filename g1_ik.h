#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "database.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "g1_kinematic_contract.h"
#include "g1_skeleton.h"
#include "ik.h"
#include "g1_surface_query.h"

#include <cstdarg>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>

static inline bool g1_ik_error(
    char* output, int capacity, const char* format, ...)
{
    if (output != NULL && capacity > 0) {
        va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(
            output, static_cast<size_t>(capacity), format, arguments);
        va_end(arguments);
    }
    return false;
}

static inline bool g1_leg_vec3_is_finite(const vec3 value)
{
    return terrain_float_is_finite(value.x) &&
           terrain_float_is_finite(value.y) &&
           terrain_float_is_finite(value.z);
}

static inline bool g1_leg_vec3_is_exact(
    const vec3 value, const vec3 expected)
{
    return value.x == expected.x &&
           value.y == expected.y &&
           value.z == expected.z;
}

static inline float g1_leg_length_squared(const vec3 value)
{
    return dot(value, value);
}

static inline bool g1_leg_axis_is_unit(const vec3 value)
{
    const float squared = g1_leg_length_squared(value);
    return terrain_float_is_finite(squared) &&
           std::fabs(squared - 1.0f) <= 1.0e-5f;
}

static inline bool g1_leg_database_shape_validate(
    const database& db, char* error, int error_capacity)
{
    if (db.bone_positions.rows <= 0 ||
        db.bone_positions.cols != G1_BoneCount ||
        db.bone_positions.data == NULL) {
        return g1_ik_error(
            error,
            error_capacity,
            "G1 IK position pose shape mismatch: expected rows>0 cols=%d",
            G1_BoneCount);
    }
    if (db.bone_rotations.rows != db.bone_positions.rows ||
        db.bone_rotations.rows <= 0 ||
        db.bone_rotations.cols != G1_BoneCount ||
        db.bone_rotations.data == NULL) {
        return g1_ik_error(
            error,
            error_capacity,
            "G1 IK rotation pose shape mismatch: expected %dx%d",
            db.bone_positions.rows,
            G1_BoneCount);
    }
    if (db.bone_parents.size != G1_BoneCount ||
        db.bone_parents.data == NULL) {
        return g1_ik_error(
            error,
            error_capacity,
            "G1 IK parent shape mismatch: expected %d entries",
            G1_BoneCount);
    }
    return true;
}

static inline bool g1_leg_database_local_basis_validate(
    const database& db, char* error, int error_capacity)
{
    const int bones[] = {
        G1_LeftKnee,
        G1_LeftAnkle,
        G1_LeftToe,
        G1_RightKnee,
        G1_RightAnkle,
        G1_RightToe
    };
    const char* const names[] = {
        "LeftKnee",
        "LeftAnkle",
        "LeftToe",
        "RightKnee",
        "RightAnkle",
        "RightToe"
    };
    const vec3 expected_offsets[] = {
        vec3(-0.078273f, -0.17734f, -0.0021489f),
        vec3(0.0f, -0.30001f, +0.000094445f),
        vec3(0.0f, -0.017558f, 0.0f),
        vec3(-0.078273f, -0.17734f, +0.0021489f),
        vec3(0.0f, -0.30001f, -0.000094445f),
        vec3(0.0f, -0.017558f, 0.0f)
    };
    const float tolerance_m = 1.0e-6f;
    const int count = static_cast<int>(sizeof(bones) / sizeof(bones[0]));
    for (int frame = 0; frame < db.bone_positions.rows; ++frame) {
        for (int index = 0; index < count; ++index) {
            const vec3 actual = db.bone_positions(frame, bones[index]);
            const vec3 expected = expected_offsets[index];
            if (!g1_leg_vec3_is_finite(actual) ||
                std::fabs(actual.x - expected.x) > tolerance_m ||
                std::fabs(actual.y - expected.y) > tolerance_m ||
                std::fabs(actual.z - expected.z) > tolerance_m) {
                return g1_ik_error(
                    error,
                    error_capacity,
                    "G1 IK local basis mismatch at frame %d bone %s",
                    frame,
                    names[index]);
            }
        }
    }
    return true;
}

static inline bool g1_leg_measured_geometry_matches(
    const G1LegConfig& value, const G1LegConfig& expected)
{
    if (!g1_leg_vec3_is_exact(
            value.knee_hinge_axis_local,
            expected.knee_hinge_axis_local) ||
        !g1_leg_vec3_is_exact(
            value.foot_forward_local,
            expected.foot_forward_local) ||
        !g1_leg_vec3_is_exact(
            value.sole_normal_local,
            expected.sole_normal_local) ||
        value.foot_sphere_radius_m != expected.foot_sphere_radius_m ||
        !g1_leg_vec3_is_exact(
            value.thigh_start_local, expected.thigh_start_local) ||
        !g1_leg_vec3_is_exact(
            value.thigh_end_local, expected.thigh_end_local) ||
        value.thigh_radius_m != expected.thigh_radius_m ||
        !g1_leg_vec3_is_exact(
            value.shin_start_local, expected.shin_start_local) ||
        !g1_leg_vec3_is_exact(
            value.shin_end_local, expected.shin_end_local) ||
        value.shin_radius_m != expected.shin_radius_m ||
        value.reach_buffer_m != expected.reach_buffer_m ||
        value.planted_clearance_m != expected.planted_clearance_m ||
        value.swing_clearance_m != expected.swing_clearance_m ||
        value.max_swing_lift_m != expected.max_swing_lift_m ||
        value.max_correction_radians != expected.max_correction_radians) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (!g1_leg_vec3_is_exact(
                value.foot_sphere_centers_local[i],
                expected.foot_sphere_centers_local[i]) ||
            !g1_leg_vec3_is_exact(
                value.sole_points_local[i],
                expected.sole_points_local[i])) {
            return false;
        }
    }
    return true;
}

static inline bool g1_leg_geometry_validate(
    const G1LegConfig& config,
    const G1LegConfig& expected,
    char* error,
    int error_capacity)
{
    const vec3 vectors[] = {
        config.knee_hinge_axis_local,
        config.foot_forward_local,
        config.sole_normal_local,
        config.foot_sphere_centers_local[0],
        config.foot_sphere_centers_local[1],
        config.foot_sphere_centers_local[2],
        config.foot_sphere_centers_local[3],
        config.sole_points_local[0],
        config.sole_points_local[1],
        config.sole_points_local[2],
        config.sole_points_local[3],
        config.thigh_start_local,
        config.thigh_end_local,
        config.shin_start_local,
        config.shin_end_local
    };
    for (const vec3 value : vectors) {
        if (!g1_leg_vec3_is_finite(value)) {
            return g1_ik_error(
                error,
                error_capacity,
                "%s leg: non-finite geometry",
                config.name);
        }
    }
    const float scalars[] = {
        config.foot_sphere_radius_m,
        config.thigh_radius_m,
        config.shin_radius_m,
        config.reach_buffer_m,
        config.planted_clearance_m,
        config.swing_clearance_m,
        config.max_swing_lift_m,
        config.max_correction_radians
    };
    for (const float value : scalars) {
        if (!terrain_float_is_finite(value)) {
            return g1_ik_error(
                error,
                error_capacity,
                "%s leg: non-finite geometry bound",
                config.name);
        }
        if (value <= 0.0f) {
            return g1_ik_error(
                error,
                error_capacity,
                "%s leg: geometry bounds must be positive",
                config.name);
        }
    }
    if (!g1_leg_axis_is_unit(config.knee_hinge_axis_local) ||
        !g1_leg_axis_is_unit(config.foot_forward_local) ||
        !g1_leg_axis_is_unit(config.sole_normal_local)) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s leg: local axes must be unit length",
            config.name);
    }
    const float orthogonal_tolerance = 1.0e-5f;
    if (std::fabs(dot(
            config.knee_hinge_axis_local,
            config.foot_forward_local)) > orthogonal_tolerance ||
        std::fabs(dot(
            config.knee_hinge_axis_local,
            config.sole_normal_local)) > orthogonal_tolerance ||
        std::fabs(dot(
            config.foot_forward_local,
            config.sole_normal_local)) > orthogonal_tolerance) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s leg: local axes must be orthogonal",
            config.name);
    }
    if (!g1_leg_vec3_is_exact(
            config.knee_hinge_axis_local,
            vec3(0.0f, 0.0f, -1.0f)) ||
        !g1_leg_vec3_is_exact(
            config.foot_forward_local,
            vec3(1.0f, 0.0f, 0.0f)) ||
        !g1_leg_vec3_is_exact(
            config.sole_normal_local,
            vec3(0.0f, 1.0f, 0.0f))) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s leg: local basis does not match the mapped XML basis",
            config.name);
    }

    const float thigh_length_squared = g1_leg_length_squared(
        config.thigh_end_local - config.thigh_start_local);
    const float shin_length_squared = g1_leg_length_squared(
        config.shin_end_local - config.shin_start_local);
    if (thigh_length_squared <= 1.0e-12f ||
        shin_length_squared <= 1.0e-12f ||
        config.thigh_radius_m * config.thigh_radius_m >=
            thigh_length_squared ||
        config.shin_radius_m * config.shin_radius_m >=
            shin_length_squared) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s leg: capsule geometry is degenerate or out of bounds",
            config.name);
    }

    const float sole_tolerance_squared = 1.0e-12f;
    for (int i = 0; i < 4; ++i) {
        const vec3 expected_point =
            config.foot_sphere_centers_local[i] -
            config.sole_normal_local * config.foot_sphere_radius_m;
        if (g1_leg_length_squared(
                config.sole_points_local[i] - expected_point) >
            sole_tolerance_squared ||
            std::fabs(dot(
                config.sole_points_local[i] - config.sole_points_local[0],
                config.sole_normal_local)) > 1.0e-6f) {
            return g1_ik_error(
                error,
                error_capacity,
                "%s leg: invalid sole probe relationship",
                config.name);
        }
        for (int j = 0; j < i; ++j) {
            if (g1_leg_length_squared(
                    config.foot_sphere_centers_local[i] -
                    config.foot_sphere_centers_local[j]) <=
                    sole_tolerance_squared ||
                g1_leg_length_squared(
                    config.sole_points_local[i] -
                    config.sole_points_local[j]) <=
                    sole_tolerance_squared) {
                return g1_ik_error(
                    error,
                    error_capacity,
                    "%s leg: duplicate sole probe relationship",
                    config.name);
            }
        }
    }

    if (!g1_leg_measured_geometry_matches(config, expected)) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s leg: measured geometry contract mismatch",
            config.name);
    }
    return true;
}

static inline bool g1_leg_config_validate(
    const database& db,
    const G1LegConfig& config,
    char* error,
    int error_capacity)
{
    if (!g1_leg_database_shape_validate(db, error, error_capacity)) {
        return false;
    }
    if (config.name == NULL || config.name[0] == '\0') {
        return g1_ik_error(
            error, error_capacity, "G1 leg config has an invalid name");
    }
    const bool is_left = std::strcmp(config.name, "left") == 0;
    const bool is_right = std::strcmp(config.name, "right") == 0;
    if (!is_left && !is_right) {
        return g1_ik_error(
            error,
            error_capacity,
            "G1 leg config has malformed name '%s'",
            config.name);
    }
    const G1LegConfig expected =
        is_left ? g1_left_leg_config() : g1_right_leg_config();
    if (config.hip != expected.hip ||
        config.knee != expected.knee ||
        config.ankle != expected.ankle ||
        config.contact != expected.contact) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s leg: invalid named bone contract",
            config.name);
    }
    if (!g1_leg_geometry_validate(
            config, expected, error, error_capacity)) {
        return false;
    }
    const int expected_hip_parent =
        is_left ? G1_LeftHipRoll : G1_RightHipRoll;
    if (db.bone_parents(config.hip) != expected_hip_parent) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s %s parent mismatch",
            config.name,
            is_left ? "LeftHipYaw" : "RightHipYaw");
    }
    if (db.bone_parents(config.knee) != config.hip) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s %s parent mismatch",
            config.name,
            is_left ? "LeftKnee" : "RightKnee");
    }
    if (db.bone_parents(config.ankle) != config.knee) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s ankle parent mismatch",
            config.name);
    }
    if (db.bone_parents(config.contact) != config.ankle) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s contact parent mismatch",
            config.name);
    }
    return true;
}

static inline bool g1_leg_configs_validate(
    const database& db, char* error, int error_capacity)
{
    if (!g1_leg_database_shape_validate(db, error, error_capacity)) {
        return false;
    }
    if (!g1_leg_database_local_basis_validate(
            db, error, error_capacity)) {
        return false;
    }
    if (!g1_leg_config_validate(
            db, g1_left_leg_config(), error, error_capacity) ||
        !g1_leg_config_validate(
            db, g1_right_leg_config(), error, error_capacity)) {
        return false;
    }
    return g1_skeleton_validate(db, error, error_capacity);
}

struct G1SurfaceTarget
{
    vec3 point;
    vec3 normal;
};

static const int G1_FootReleaseMaximumFrames = 25;

struct G1FootLockState
{
    bool initialized = false;
    bool contact = false;
    bool locked = false;
    bool position_active = false;
    bool releasing = false;
    int release_frames = 0;
    vec3 previous_input;
    vec3 lock_point;
    vec3 output_position;
    vec3 output_velocity;
    vec3 offset_position;
    vec3 offset_velocity;
};

struct G1FootTarget
{
    bool locked = false;
    bool position_active = false;
    bool releasing = false;
    bool drift_limit_exceeded = false;
    G1SurfaceTarget surface;
    vec3 desired_sole_normal;
    vec3 sole_center;
    float horizontal_drift_m = 0.0f;
};

struct G1PhysicalSolePositionTarget
{
    quat contact_rotation;
    vec3 contact_origin;
    vec3 ankle_target;
};

bool g1_physical_sole_position_target(
    G1PhysicalSolePositionTarget& output,
    const vec3& current_contact_origin,
    const quat& current_contact_rotation,
    const vec3& current_ankle_origin,
    const G1LegConfig& config,
    const vec3& desired_sole_center,
    const vec3& desired_sole_normal,
    char* error,
    int error_capacity);

static inline vec3 g1_ik_vec3_canonicalize(vec3 value)
{
    return vec3(
        terrain_runtime_canonicalize_output(value.x),
        terrain_runtime_canonicalize_output(value.y),
        terrain_runtime_canonicalize_output(value.z));
}

static inline bool g1_ik_vec3_is_zero(vec3 value)
{
    return (terrain_float_bits(value.x) & UINT32_C(0x7fffffff)) == 0 &&
           (terrain_float_bits(value.y) & UINT32_C(0x7fffffff)) == 0 &&
           (terrain_float_bits(value.z) & UINT32_C(0x7fffffff)) == 0;
}

static inline bool g1_ik_vec3_bits_equal(vec3 left, vec3 right)
{
    return terrain_float_bits(left.x) == terrain_float_bits(right.x) &&
           terrain_float_bits(left.y) == terrain_float_bits(right.y) &&
           terrain_float_bits(left.z) == terrain_float_bits(right.z);
}

static inline bool g1_ik_surface_target_is_valid(
    const G1SurfaceTarget& target)
{
    return g1_ik_vec3_is_runtime_value(target.point) &&
           g1_ik_surface_normal_is_valid(target.normal);
}

static inline bool g1_surface_target_sample(
    G1SurfaceTarget& output,
    const heightfield& field,
    float x,
    float z,
    float clearance,
    char* error,
    int error_capacity)
{
    if (!g1_ik_float_is_runtime_value(clearance) || clearance < 0.0f) {
        return g1_ik_error(
            error, error_capacity,
            "G1 IK surface query has invalid clearance");
    }
    G1SurfaceSample sample = {};
    const G1SurfaceQueryStatus status =
        g1_surface_query_v2(sample, field, x, z);
    if (status == G1SurfaceQueryOutside) {
        return g1_ik_error(
            error, error_capacity,
            "G1 IK surface query is outside the G1HF/v2 domain");
    }
    if (status != G1SurfaceQueryValid) {
        return g1_ik_error(
            error, error_capacity,
            "G1 IK surface query has invalid G1HF/v2 input or data");
    }

    float canonical_x = 0.0f;
    float canonical_z = 0.0f;
    float target_y = 0.0f;
    const volatile double promoted_y =
        static_cast<double>(sample.height) +
        static_cast<double>(clearance);
    if (!terrain_v2_query_coordinate(x, canonical_x) ||
        !terrain_v2_query_coordinate(z, canonical_z) ||
        !terrain_v2_round_output(promoted_y, target_y)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 IK surface query height plus clearance is invalid");
    }
    G1SurfaceTarget candidate = {};
    candidate.point = vec3(canonical_x, target_y, canonical_z);
    candidate.normal = sample.normal;
    if (!g1_ik_surface_target_is_valid(candidate)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 IK surface query produced an invalid target");
    }
    output = candidate;
    return true;
}

static inline bool g1_foot_runtime_config_validate(
    const G1LegConfig& config, char* error, int error_capacity)
{
    if (config.name == NULL || config.name[0] == '\0') {
        return g1_ik_error(
            error, error_capacity,
            "G1 planted-foot config has an invalid name");
    }
    const bool is_left = std::strcmp(config.name, "left") == 0;
    const bool is_right = std::strcmp(config.name, "right") == 0;
    if (!is_left && !is_right) {
        return g1_ik_error(
            error, error_capacity,
            "G1 planted-foot config has a malformed name");
    }
    const G1LegConfig expected =
        is_left ? g1_left_leg_config() : g1_right_leg_config();
    if (config.hip != expected.hip ||
        config.knee != expected.knee ||
        config.ankle != expected.ankle ||
        config.contact != expected.contact ||
        !g1_leg_measured_geometry_matches(config, expected)) {
        return g1_ik_error(
            error, error_capacity,
            "%s planted-foot config contract mismatch",
            config.name);
    }
    return true;
}

static inline bool g1_foot_lock_state_is_valid(
    const G1FootLockState& state)
{
    if (!state.initialized || state.contact != state.locked ||
        state.position_active != (state.locked || state.releasing) ||
        (state.locked && state.releasing) ||
        (state.releasing && state.contact) ||
        state.release_frames < 0 ||
        state.release_frames >= G1_FootReleaseMaximumFrames ||
        (!state.releasing && state.release_frames != 0) ||
        !g1_ik_vec3_is_runtime_value(state.previous_input) ||
        !g1_ik_vec3_is_runtime_value(state.lock_point) ||
        !g1_ik_vec3_is_runtime_value(state.output_position) ||
        !g1_ik_vec3_is_runtime_value(state.output_velocity) ||
        !g1_ik_vec3_is_runtime_value(state.offset_position) ||
        !g1_ik_vec3_is_runtime_value(state.offset_velocity)) {
        return false;
    }
    return state.position_active ||
           (g1_ik_vec3_is_zero(state.offset_position) &&
            g1_ik_vec3_is_zero(state.offset_velocity));
}

static inline bool g1_foot_target_is_valid(const G1FootTarget& target)
{
    if (target.position_active != (target.locked || target.releasing) ||
        (target.locked && target.releasing) ||
        (target.drift_limit_exceeded && !target.locked) ||
        !g1_ik_vec3_is_runtime_value(target.surface.point) ||
        !g1_ik_vec3_is_runtime_value(target.surface.normal) ||
        !g1_ik_vec3_is_runtime_value(target.desired_sole_normal) ||
        !g1_ik_vec3_is_runtime_value(target.sole_center) ||
        !g1_ik_float_is_runtime_value(target.horizontal_drift_m) ||
        target.horizontal_drift_m < 0.0f ||
        (!target.locked && target.horizontal_drift_m != 0.0f)) {
        return false;
    }
    const bool has_surface_normal =
        !g1_ik_vec3_is_zero(target.surface.normal);
    const bool has_desired_normal =
        !g1_ik_vec3_is_zero(target.desired_sole_normal);
    if (!has_surface_normal || !has_desired_normal) {
        return !target.locked && !target.position_active &&
               !target.releasing && !target.drift_limit_exceeded &&
               !has_surface_normal && !has_desired_normal &&
               g1_ik_vec3_bits_equal(target.surface.point, vec3()) &&
               g1_ik_vec3_bits_equal(target.surface.normal, vec3()) &&
               g1_ik_vec3_bits_equal(
                   target.desired_sole_normal, vec3()) &&
               g1_ik_vec3_bits_equal(target.sole_center, vec3()) &&
               terrain_float_bits(target.horizontal_drift_m) == 0U;
    }
    return g1_ik_surface_target_is_valid(target.surface) &&
           g1_ik_surface_normal_is_valid(
               target.desired_sole_normal);
}

static inline bool g1_foot_lock_reset(
    G1FootLockState& state,
    vec3 initial_sole_center,
    char* error,
    int error_capacity)
{
    if (!g1_ik_vec3_is_runtime_value(initial_sole_center)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 planted-foot reset requires a finite normal-or-zero center");
    }
    const vec3 canonical =
        g1_ik_vec3_canonicalize(initial_sole_center);
    G1FootLockState candidate = {};
    candidate.initialized = true;
    candidate.previous_input = canonical;
    candidate.lock_point = canonical;
    candidate.output_position = canonical;
    if (!g1_foot_lock_state_is_valid(candidate)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 planted-foot reset produced invalid state");
    }
    state = candidate;
    return true;
}

static inline bool g1_foot_checked_velocity_component(
    float& output, float current, float previous, float dt)
{
    const volatile double difference =
        static_cast<double>(current) - static_cast<double>(previous);
    const volatile double velocity =
        difference / static_cast<double>(dt);
    return terrain_v2_round_output(velocity, output);
}

static inline bool g1_foot_checked_input_velocity(
    vec3& output, vec3 current, vec3 previous, float dt)
{
    vec3 candidate;
    if (!g1_foot_checked_velocity_component(
            candidate.x, current.x, previous.x, dt) ||
        !g1_foot_checked_velocity_component(
            candidate.y, current.y, previous.y, dt) ||
        !g1_foot_checked_velocity_component(
            candidate.z, current.z, previous.z, dt)) {
        return false;
    }
    output = candidate;
    return true;
}

static inline bool g1_foot_checked_transition_component(
    float& offset_position,
    float& offset_velocity,
    float output_position,
    float output_velocity,
    float target_position,
    float target_velocity)
{
    float next_position = 0.0f;
    float next_velocity = 0.0f;
    if (!terrain_f32_sub(
            next_position, output_position, target_position) ||
        !terrain_f32_sub(
            next_velocity, output_velocity, target_velocity)) {
        return false;
    }
    offset_position = next_position;
    offset_velocity = next_velocity;
    return true;
}

static inline bool g1_foot_checked_transition(
    vec3& offset_position,
    vec3& offset_velocity,
    vec3 output_position,
    vec3 output_velocity,
    vec3 target_position,
    vec3 target_velocity)
{
    vec3 next_position;
    vec3 next_velocity;
    if (!g1_foot_checked_transition_component(
            next_position.x, next_velocity.x,
            output_position.x, output_velocity.x,
            target_position.x, target_velocity.x) ||
        !g1_foot_checked_transition_component(
            next_position.y, next_velocity.y,
            output_position.y, output_velocity.y,
            target_position.y, target_velocity.y) ||
        !g1_foot_checked_transition_component(
            next_position.z, next_velocity.z,
            output_position.z, output_velocity.z,
            target_position.z, target_velocity.z)) {
        return false;
    }
    offset_position = next_position;
    offset_velocity = next_velocity;
    return true;
}

static inline bool g1_foot_checked_spring_component(
    float& output_position,
    float& output_velocity,
    float& offset_position,
    float& offset_velocity,
    float target_position,
    float target_velocity,
    float y,
    float eydt,
    float dt)
{
    float position_times_y = 0.0f;
    float j1 = 0.0f;
    float j1_times_dt = 0.0f;
    float position_sum = 0.0f;
    float next_offset_position = 0.0f;
    float j1_times_y = 0.0f;
    float velocity_difference = 0.0f;
    float next_offset_velocity = 0.0f;
    float next_output_position = 0.0f;
    float next_output_velocity = 0.0f;
    if (!terrain_f32_mul(position_times_y, offset_position, y) ||
        !terrain_f32_add(j1, offset_velocity, position_times_y) ||
        !terrain_f32_mul(j1_times_dt, j1, dt) ||
        !terrain_f32_add(position_sum, offset_position, j1_times_dt) ||
        !terrain_f32_mul(next_offset_position, eydt, position_sum) ||
        !terrain_f32_mul(j1_times_y, j1, y) ||
        !terrain_f32_mul(j1_times_y, j1_times_y, dt) ||
        !terrain_f32_sub(
            velocity_difference, offset_velocity, j1_times_y) ||
        !terrain_f32_mul(
            next_offset_velocity, eydt, velocity_difference) ||
        !terrain_f32_add(
            next_output_position, target_position,
            next_offset_position) ||
        !terrain_f32_add(
            next_output_velocity, target_velocity,
            next_offset_velocity)) {
        return false;
    }
    offset_position = next_offset_position;
    offset_velocity = next_offset_velocity;
    output_position = next_output_position;
    output_velocity = next_output_velocity;
    return true;
}

static inline bool g1_foot_checked_spring_update(
    vec3& output_position,
    vec3& output_velocity,
    vec3& offset_position,
    vec3& offset_velocity,
    vec3 target_position,
    vec3 target_velocity,
    float dt)
{
    const float half_life = 0.10f;
    const float y = halflife_to_damping(half_life) / 2.0f;
    float y_times_dt = 0.0f;
    if (!g1_ik_float_is_runtime_value(y) ||
        !terrain_f32_mul(y_times_dt, y, dt)) {
        return false;
    }
    float eydt = fast_negexpf(y_times_dt);
    if (!terrain_f32_accept(eydt, eydt)) {
        return false;
    }

    vec3 next_output_position = output_position;
    vec3 next_output_velocity = output_velocity;
    vec3 next_offset_position = offset_position;
    vec3 next_offset_velocity = offset_velocity;
    if (!g1_foot_checked_spring_component(
            next_output_position.x, next_output_velocity.x,
            next_offset_position.x, next_offset_velocity.x,
            target_position.x, target_velocity.x, y, eydt, dt) ||
        !g1_foot_checked_spring_component(
            next_output_position.y, next_output_velocity.y,
            next_offset_position.y, next_offset_velocity.y,
            target_position.y, target_velocity.y, y, eydt, dt) ||
        !g1_foot_checked_spring_component(
            next_output_position.z, next_output_velocity.z,
            next_offset_position.z, next_offset_velocity.z,
            target_position.z, target_velocity.z, y, eydt, dt)) {
        return false;
    }
    output_position = next_output_position;
    output_velocity = next_output_velocity;
    offset_position = next_offset_position;
    offset_velocity = next_offset_velocity;
    return true;
}

static inline bool g1_foot_release_is_settled(
    const G1FootLockState& state)
{
    const double position_tolerance = 1.0e-4;
    const double velocity_tolerance = 1.0e-3;
    return fabs(static_cast<double>(state.offset_position.x)) <=
               position_tolerance &&
           fabs(static_cast<double>(state.offset_position.y)) <=
               position_tolerance &&
           fabs(static_cast<double>(state.offset_position.z)) <=
               position_tolerance &&
           fabs(static_cast<double>(state.offset_velocity.x)) <=
               velocity_tolerance &&
           fabs(static_cast<double>(state.offset_velocity.y)) <=
               velocity_tolerance &&
           fabs(static_cast<double>(state.offset_velocity.z)) <=
               velocity_tolerance;
}

static inline bool g1_foot_horizontal_drift(
    float& rounded_distance,
    bool& limit_exceeded,
    vec3 input,
    vec3 lock_point)
{
    const volatile double dx =
        static_cast<double>(input.x) -
        static_cast<double>(lock_point.x);
    const volatile double dz =
        static_cast<double>(input.z) -
        static_cast<double>(lock_point.z);
    const volatile double square_x = dx * dx;
    const volatile double square_z = dz * dz;
    const volatile double square_sum = square_x + square_z;
    if (!terrain_double_is_finite(square_sum) || square_sum < 0.0) {
        return false;
    }
    const volatile double distance = sqrt(square_sum);
    if (!terrain_double_is_finite(distance) ||
        !terrain_v2_round_output(distance, rounded_distance)) {
        return false;
    }
    limit_exceeded = distance > static_cast<double>(0.20f);
    return true;
}

static inline bool g1_foot_lock_update(
    G1FootLockState& state,
    G1FootTarget& output,
    const heightfield& field,
    const G1LegConfig& config,
    vec3 input_sole_center,
    bool input_contact,
    float dt,
    char* error,
    int error_capacity)
{
    if (!g1_foot_lock_state_is_valid(state)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 planted-foot update has invalid lock state");
    }
    if (!g1_foot_target_is_valid(output)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 planted-foot update has invalid prior output");
    }
    if (!g1_foot_runtime_config_validate(
            config, error, error_capacity)) {
        return false;
    }
    if (!g1_ik_vec3_is_runtime_value(input_sole_center)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 planted-foot update has invalid input center");
    }
    if (!g1_dt_is_exact_25_hz(dt)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 planted-foot update requires exact binary32 25 Hz dt");
    }

    G1SurfaceTarget validated_locked_surface = {};
    if (state.locked) {
        if (!g1_surface_target_sample(
                validated_locked_surface, field,
                state.lock_point.x, state.lock_point.z,
                config.planted_clearance_m,
                error, error_capacity)) {
            return false;
        }
        if (!g1_ik_vec3_bits_equal(
                state.lock_point, validated_locked_surface.point)) {
            return g1_ik_error(
                error, error_capacity,
                "G1 planted-foot lock point does not match checked terrain");
        }
    }

    G1FootLockState next = state;
    const vec3 canonical_input =
        g1_ik_vec3_canonicalize(input_sole_center);
    vec3 input_velocity;
    if (!g1_foot_checked_input_velocity(
            input_velocity, canonical_input, next.previous_input, dt)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 planted-foot input velocity overflowed");
    }

    const bool rising_edge = !next.contact && input_contact;
    const bool falling_edge = next.contact && !input_contact;
    G1SurfaceTarget sampled = {};
    if (rising_edge) {
        if (!g1_surface_target_sample(
                sampled, field, canonical_input.x, canonical_input.z,
                config.planted_clearance_m,
                error, error_capacity)) {
            return false;
        }
        if (!g1_foot_checked_transition(
                next.offset_position, next.offset_velocity,
                next.output_position, next.output_velocity,
                sampled.point, vec3())) {
            return g1_ik_error(
                error, error_capacity,
                "G1 planted-foot rising transition overflowed");
        }
        next.lock_point = sampled.point;
        next.locked = true;
        next.position_active = true;
        next.releasing = false;
        next.release_frames = 0;
    } else if (falling_edge) {
        if (!g1_foot_checked_transition(
                next.offset_position, next.offset_velocity,
                next.output_position, next.output_velocity,
                canonical_input, input_velocity)) {
            return g1_ik_error(
                error, error_capacity,
                "G1 planted-foot release transition overflowed");
        }
        next.locked = false;
        next.position_active = true;
        next.releasing = true;
        next.release_frames = 0;
    } else if (next.locked) {
        if (!g1_foot_checked_spring_update(
                next.output_position, next.output_velocity,
                next.offset_position, next.offset_velocity,
                next.lock_point, vec3(), dt)) {
            return g1_ik_error(
                error, error_capacity,
                "G1 planted-foot lock spring overflowed");
        }
    } else if (next.releasing) {
        if (!g1_foot_checked_spring_update(
                next.output_position, next.output_velocity,
                next.offset_position, next.offset_velocity,
                canonical_input, input_velocity, dt)) {
            return g1_ik_error(
                error, error_capacity,
                "G1 planted-foot release spring overflowed");
        }
        ++next.release_frames;
        // Consume at least two stable release updates even when the event
        // begins at zero offset, so release is observably multi-frame.
        if ((next.release_frames >= 2 &&
             g1_foot_release_is_settled(next)) ||
            next.release_frames >= G1_FootReleaseMaximumFrames) {
            next.position_active = false;
            next.releasing = false;
            next.release_frames = 0;
            next.output_position = canonical_input;
            next.output_velocity = input_velocity;
            next.offset_position = vec3();
            next.offset_velocity = vec3();
        }
    } else {
        next.output_position = canonical_input;
        next.output_velocity = input_velocity;
        next.offset_position = vec3();
        next.offset_velocity = vec3();
    }
    next.contact = input_contact;
    next.previous_input = canonical_input;

    if (next.locked && state.locked) {
        sampled = validated_locked_surface;
    } else if (next.locked) {
        if (!g1_surface_target_sample(
                sampled, field, next.lock_point.x, next.lock_point.z,
                config.planted_clearance_m,
                error, error_capacity)) {
            return false;
        }
    } else if (!g1_surface_target_sample(
                   sampled, field,
                   canonical_input.x, canonical_input.z,
                   config.planted_clearance_m,
                   error, error_capacity)) {
        return false;
    }

    G1FootTarget candidate = {};
    candidate.locked = next.locked;
    candidate.position_active = next.position_active;
    candidate.releasing = next.releasing;
    candidate.surface = sampled;
    candidate.desired_sole_normal = sampled.normal;
    candidate.sole_center = next.locked
        ? next.lock_point
        : next.output_position;
    if (next.locked &&
        !g1_foot_horizontal_drift(
            candidate.horizontal_drift_m,
            candidate.drift_limit_exceeded,
            canonical_input, next.lock_point)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 planted-foot horizontal drift overflowed");
    }

    if (!g1_foot_lock_state_is_valid(next) ||
        !g1_foot_target_is_valid(candidate) ||
        candidate.locked != next.locked ||
        candidate.position_active != next.position_active ||
        candidate.releasing != next.releasing ||
        !g1_ik_vec3_is_runtime_value(candidate.sole_center)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 planted-foot update produced invalid state");
    }
    state = next;
    output = candidate;
    return true;
}

enum G1LegIterationProvenance
{
    G1LegIterationNone = 0,
    G1LegIterationDirect1,
    G1LegIterationContact1,
    G1LegIterationContact2,
    G1LegIterationContact3,
    G1LegIterationContact4,
    G1LegIterationBaselineFallback1,
};

struct G1LegSolveResult
{
    bool applied = false;
    bool reachable = false;
    bool correction_limited = false;
    bool safe_stop_requested = false;
    int iterations = 0;
    G1LegIterationProvenance iteration_provenance =
        G1LegIterationNone;
    vec3 requested_ankle_target;
    vec3 clamped_ankle_target;
    vec3 hinge_axis_world;
    vec3 bend_direction;
    bool bend_used_current_projection = false;
    bool bend_used_hinge_fallback = false;
    bool bend_used_safe_perpendicular = false;
    bool bend_sign_flipped = false;
    float raw_distance_m = 0.0f;
    float clamped_distance_m = 0.0f;
    float max_correction_radians = 0.0f;
    float contact_residual_m = FLT_MAX;
};

static constexpr int G1ContactSolveMaximumIterations = 4;

static inline G1LegIterationProvenance
g1_ik_contact_iteration_provenance(int iterations)
{
    switch (iterations) {
    case 1: return G1LegIterationContact1;
    case 2: return G1LegIterationContact2;
    case 3: return G1LegIterationContact3;
    case 4: return G1LegIterationContact4;
    default: return G1LegIterationNone;
    }
}

static inline bool g1_ik_memory_ranges_overlap(
    const void* left,
    size_t left_bytes,
    const void* right,
    size_t right_bytes)
{
    if (left == NULL || right == NULL ||
        left_bytes == 0 || right_bytes == 0) {
        return false;
    }
    const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left);
    const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right);
    if (left_begin > UINTPTR_MAX - left_bytes ||
        right_begin > UINTPTR_MAX - right_bytes) {
        return true;
    }
    const uintptr_t left_end = left_begin + left_bytes;
    const uintptr_t right_end = right_begin + right_bytes;
    return left_begin < right_end && right_begin < left_end;
}

static inline bool g1_ik_checked_physical_sole_centroid(
    vec3& output,
    vec3 contact_origin,
    quat contact_rotation,
    const G1LegConfig& config)
{
    if (g1_ik_memory_ranges_overlap(
            &output, sizeof(output), &config, sizeof(config)) ||
        !g1_foot_runtime_config_validate(config, NULL, 0) ||
        !g1_ik_vec3_is_runtime_value(contact_origin) ||
        !ik_quat_is_unit(contact_rotation)) {
        return false;
    }

    volatile double sum_x = 0.0;
    volatile double sum_y = 0.0;
    volatile double sum_z = 0.0;
    for (int probe = 0; probe < 4; ++probe) {
        vec3 offset;
        vec3 point;
        if (!ik_checked_quat_rotate(
                offset,
                contact_rotation,
                config.sole_points_local[probe]) ||
            !ik_checked_vec3_add(point, contact_origin, offset) ||
            !g1_ik_vec3_is_runtime_value(point)) {
            return false;
        }
        sum_x = sum_x + static_cast<double>(point.x);
        sum_y = sum_y + static_cast<double>(point.y);
        sum_z = sum_z + static_cast<double>(point.z);
    }

    const volatile double centroid_x = sum_x / 4.0;
    const volatile double centroid_y = sum_y / 4.0;
    const volatile double centroid_z = sum_z / 4.0;
    vec3 candidate;
    if (!terrain_v2_round_output(centroid_x, candidate.x) ||
        !terrain_v2_round_output(centroid_y, candidate.y) ||
        !terrain_v2_round_output(centroid_z, candidate.z) ||
        !g1_ik_vec3_is_runtime_value(candidate)) {
        return false;
    }
    output = candidate;
    return true;
}

static inline bool g1_ik_parent_topology_validate(
    const slice1d<int> parents,
    char* error,
    int error_capacity)
{
    static const int expected[G1_BoneCount] = {
        -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
    };
    if (parents.size != G1_BoneCount || parents.data == NULL) {
        return g1_ik_error(
            error, error_capacity,
            "G1 named IK requires exactly 31 non-null parents");
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (parents.data[bone] != expected[bone]) {
            return g1_ik_error(
                error, error_capacity,
                "G1 named IK parent topology mismatch at bone %d",
                bone);
        }
    }
    return true;
}

static inline bool g1_ik_pose_inputs_validate(
    const slice1d<quat> output_rotations,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const G1LegConfig& config,
    char* error,
    int error_capacity)
{
    if (output_rotations.size != G1_BoneCount ||
        output_rotations.data == NULL ||
        local_positions.size != G1_BoneCount ||
        local_positions.data == NULL ||
        baseline_rotations.size != G1_BoneCount ||
        baseline_rotations.data == NULL) {
        return g1_ik_error(
            error, error_capacity,
            "G1 named IK requires non-null exact 31-bone pose slices");
    }
    const size_t rotation_bytes =
        static_cast<size_t>(G1_BoneCount) * sizeof(quat);
    if (g1_ik_memory_ranges_overlap(
            output_rotations.data, rotation_bytes,
            baseline_rotations.data, rotation_bytes)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 named IK working and baseline rotations must not alias");
    }
    if (!g1_ik_parent_topology_validate(
            parents, error, error_capacity) ||
        !g1_foot_runtime_config_validate(
            config, error, error_capacity)) {
        return false;
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (!g1_ik_vec3_is_runtime_value(local_positions.data[bone])) {
            return g1_ik_error(
                error, error_capacity,
                "G1 named IK local position is invalid at bone %d",
                bone);
        }
        if (!ik_quat_is_unit(baseline_rotations.data[bone])) {
            return g1_ik_error(
                error, error_capacity,
                "G1 named IK baseline quaternion is not finite/unit at bone %d",
                bone);
        }
        if (!ik_quat_is_unit(output_rotations.data[bone])) {
            return g1_ik_error(
                error, error_capacity,
                "G1 named IK working quaternion is not finite/unit at bone %d",
                bone);
        }
    }
    return true;
}

static inline bool g1_ik_checked_forward_kinematics(
    const slice1d<vec3> output_positions,
    const slice1d<quat> output_rotations,
    const slice1d<vec3> local_positions,
    const slice1d<quat> local_rotations,
    const slice1d<int> parents,
    char* error,
    int error_capacity)
{
    if (output_positions.size != G1_BoneCount ||
        output_positions.data == NULL ||
        output_rotations.size != G1_BoneCount ||
        output_rotations.data == NULL ||
        local_positions.size != G1_BoneCount ||
        local_positions.data == NULL ||
        local_rotations.size != G1_BoneCount ||
        local_rotations.data == NULL ||
        !g1_ik_parent_topology_validate(
            parents, error, error_capacity)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 checked FK requires complete non-null 31-bone slices");
    }
    const size_t position_bytes =
        static_cast<size_t>(G1_BoneCount) * sizeof(vec3);
    const size_t rotation_bytes =
        static_cast<size_t>(G1_BoneCount) * sizeof(quat);
    const size_t parent_bytes =
        static_cast<size_t>(G1_BoneCount) * sizeof(int);
    if (g1_ik_memory_ranges_overlap(
            output_positions.data, position_bytes,
            output_rotations.data, rotation_bytes)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 checked FK output ranges must not overlap");
    }
    if (g1_ik_memory_ranges_overlap(
            output_positions.data, position_bytes,
            local_positions.data, position_bytes) ||
        g1_ik_memory_ranges_overlap(
            output_positions.data, position_bytes,
            local_rotations.data, rotation_bytes) ||
        g1_ik_memory_ranges_overlap(
            output_positions.data, position_bytes,
            parents.data, parent_bytes) ||
        g1_ik_memory_ranges_overlap(
            output_rotations.data, rotation_bytes,
            local_positions.data, position_bytes) ||
        g1_ik_memory_ranges_overlap(
            output_rotations.data, rotation_bytes,
            local_rotations.data, rotation_bytes) ||
        g1_ik_memory_ranges_overlap(
            output_rotations.data, rotation_bytes,
            parents.data, parent_bytes)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 checked FK outputs must not alias const inputs");
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (!g1_ik_vec3_is_runtime_value(local_positions.data[bone]) ||
            !ik_quat_is_unit(local_rotations.data[bone])) {
            return g1_ik_error(
                error, error_capacity,
                "G1 checked FK has invalid local pose at bone %d",
                bone);
        }
    }

    array1d<vec3> candidate_positions(G1_BoneCount);
    array1d<quat> candidate_rotations(G1_BoneCount);
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        const int parent = parents.data[bone];
        if (parent == -1) {
            candidate_positions(bone) = local_positions.data[bone];
            candidate_rotations(bone) = local_rotations.data[bone];
        } else {
            vec3 rotated_local;
            if (!ik_checked_quat_rotate(
                    rotated_local,
                    candidate_rotations(parent),
                    local_positions.data[bone]) ||
                !ik_checked_vec3_add(
                    candidate_positions(bone),
                    candidate_positions(parent), rotated_local) ||
                !ik_checked_quat_multiply(
                    candidate_rotations(bone),
                    candidate_rotations(parent),
                    local_rotations.data[bone])) {
                return g1_ik_error(
                    error, error_capacity,
                    "G1 checked FK overflowed at bone %d", bone);
            }
        }
        if (!g1_ik_vec3_is_runtime_value(candidate_positions(bone)) ||
            !ik_quat_is_unit(candidate_rotations(bone))) {
            return g1_ik_error(
                error, error_capacity,
                "G1 checked FK produced invalid global at bone %d",
                bone);
        }
    }
    std::memcpy(
        output_positions.data, candidate_positions.data,
        position_bytes);
    std::memcpy(
        output_rotations.data, candidate_rotations.data,
        rotation_bytes);
    return true;
}

static inline bool g1_ik_leg_result_is_valid(
    const G1LegSolveResult& result)
{
    bool iteration_provenance_is_coherent = false;
    switch (result.iteration_provenance) {
    case G1LegIterationDirect1:
        iteration_provenance_is_coherent =
            result.iterations == 1 &&
            terrain_float_bits(result.contact_residual_m) ==
                terrain_float_bits(FLT_MAX);
        break;
    case G1LegIterationContact1:
    case G1LegIterationContact2:
    case G1LegIterationContact3:
    case G1LegIterationContact4:
        iteration_provenance_is_coherent =
            result.iterations ==
                static_cast<int>(result.iteration_provenance) -
                    static_cast<int>(G1LegIterationContact1) + 1 &&
            terrain_float_bits(result.contact_residual_m) !=
                terrain_float_bits(FLT_MAX);
        break;
    case G1LegIterationBaselineFallback1:
        iteration_provenance_is_coherent =
            result.iterations == 1 &&
            terrain_float_bits(result.contact_residual_m) !=
                terrain_float_bits(FLT_MAX) &&
            result.reachable &&
            !result.correction_limited &&
            terrain_float_bits(result.max_correction_radians) == 0U &&
            g1_ik_vec3_bits_equal(
                result.requested_ankle_target,
                result.clamped_ankle_target) &&
            terrain_float_bits(result.raw_distance_m) ==
                terrain_float_bits(result.clamped_distance_m);
        break;
    case G1LegIterationNone:
    default:
        break;
    }
    return result.applied && result.iterations >= 1 &&
           result.iterations <= G1ContactSolveMaximumIterations &&
           iteration_provenance_is_coherent &&
           g1_ik_vec3_is_runtime_value(result.requested_ankle_target) &&
           g1_ik_vec3_is_runtime_value(result.clamped_ankle_target) &&
           ik_vec3_is_unit(result.hinge_axis_world) &&
           ik_vec3_is_unit(result.bend_direction) &&
           g1_ik_float_is_runtime_value(result.raw_distance_m) &&
           result.raw_distance_m >= 0.0f &&
           g1_ik_float_is_runtime_value(result.clamped_distance_m) &&
           result.clamped_distance_m >= 0.0f &&
           g1_ik_float_is_runtime_value(result.max_correction_radians) &&
           result.max_correction_radians >= 0.0f &&
           g1_ik_float_is_runtime_value(result.contact_residual_m) &&
           result.contact_residual_m >= 0.0f;
}

static inline bool g1_apply_named_position_ik(
    slice1d<quat> output_rotations,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const G1LegConfig& config,
    vec3 requested_ankle_target,
    G1LegSolveResult& output,
    char* error,
    int error_capacity)
{
    if (!g1_ik_pose_inputs_validate(
            output_rotations, local_positions,
            baseline_rotations, parents, config,
            error, error_capacity)) {
        return false;
    }
    if (!g1_ik_vec3_is_runtime_value(requested_ankle_target)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 named leg IK target is invalid");
    }

    array1d<vec3> global_positions(G1_BoneCount);
    array1d<quat> global_rotations(G1_BoneCount);
    if (!g1_ik_checked_forward_kinematics(
            global_positions, global_rotations,
            local_positions, baseline_rotations, parents,
            error, error_capacity)) {
        return false;
    }

    const int hip_parent = parents.data[config.hip];
    vec3 hinge_axis_world;
    if (hip_parent < 0 || hip_parent >= G1_BoneCount ||
        !ik_checked_quat_rotate(
            hinge_axis_world,
            global_rotations(config.knee),
            config.knee_hinge_axis_local) ||
        !ik_vec3_is_unit(hinge_axis_world)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 named leg IK could not map the knee hinge");
    }

    IKTwoBoneResult solve = {};
    if (!ik_two_bone_bounded(
            solve,
            baseline_rotations(config.hip),
            baseline_rotations(config.knee),
            global_positions(config.hip),
            global_positions(config.knee),
            global_positions(config.ankle),
            requested_ankle_target,
            hinge_axis_world,
            global_rotations(config.hip),
            global_rotations(config.knee),
            global_rotations(hip_parent),
            config.reach_buffer_m,
            config.max_correction_radians)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 named leg IK rejected checked two-bone math");
    }

    array1d<quat> candidate_pose(output_rotations);
    candidate_pose(config.hip) = solve.root_local;
    candidate_pose(config.knee) = solve.middle_local;
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (!ik_quat_is_unit(candidate_pose(bone))) {
            return g1_ik_error(
                error, error_capacity,
                "G1 named leg IK produced invalid local rotation");
        }
    }
    array1d<vec3> candidate_global_positions(G1_BoneCount);
    array1d<quat> candidate_global_rotations(G1_BoneCount);
    if (!g1_ik_checked_forward_kinematics(
            candidate_global_positions, candidate_global_rotations,
            local_positions, candidate_pose, parents,
            error, error_capacity)) {
        return false;
    }

    G1LegSolveResult candidate = {};
    candidate.applied = true;
    candidate.reachable = solve.reachable;
    candidate.correction_limited = solve.correction_limited;
    candidate.safe_stop_requested =
        !solve.reachable || solve.correction_limited;
    candidate.iterations = 1;
    candidate.iteration_provenance = G1LegIterationDirect1;
    candidate.requested_ankle_target = requested_ankle_target;
    candidate.clamped_ankle_target = solve.target.clamped_target;
    candidate.hinge_axis_world = hinge_axis_world;
    candidate.bend_direction = solve.bend.direction;
    candidate.bend_used_current_projection =
        solve.bend.used_current_projection;
    candidate.bend_used_hinge_fallback =
        solve.bend.used_hinge_fallback;
    candidate.bend_used_safe_perpendicular =
        solve.bend.used_safe_perpendicular;
    candidate.bend_sign_flipped = solve.bend.sign_flipped;
    candidate.raw_distance_m = solve.target.raw_distance_m;
    candidate.clamped_distance_m = solve.target.clamped_distance_m;
    candidate.max_correction_radians =
        solve.root_correction_radians > solve.middle_correction_radians
            ? solve.root_correction_radians
            : solve.middle_correction_radians;
    candidate.contact_residual_m = FLT_MAX;
    if (!g1_ik_leg_result_is_valid(candidate) ||
        candidate.max_correction_radians >
            config.max_correction_radians) {
        return g1_ik_error(
            error, error_capacity,
            "G1 named leg IK produced invalid result");
    }

    std::memcpy(
        output_rotations.data, candidate_pose.data,
        static_cast<size_t>(G1_BoneCount) * sizeof(quat));
    output = candidate;
    return true;
}

static inline bool g1_ik_contact_residual_is_converged_precise(
    double residual_m)
{
    return terrain_double_is_finite(residual_m) &&
           residual_m >= 0.0 &&
           residual_m <= static_cast<double>(0.005f);
}

static inline bool g1_ik_contact_residual_is_converged(float residual_m)
{
    return g1_ik_float_is_runtime_value(residual_m) &&
           g1_ik_contact_residual_is_converged_precise(
               static_cast<double>(residual_m));
}

static inline bool g1_ik_contact_iterations_have_valid_provenance(
    const G1LegSolveResult& result)
{
    return g1_ik_leg_result_is_valid(result) &&
           (result.iteration_provenance ==
                G1LegIterationContact1 ||
            result.iteration_provenance ==
                G1LegIterationContact2 ||
            result.iteration_provenance ==
                G1LegIterationContact3 ||
            result.iteration_provenance ==
                G1LegIterationContact4 ||
            result.iteration_provenance ==
                G1LegIterationBaselineFallback1);
}

static inline bool g1_apply_named_contact_position_ik_from_initial_target(
    slice1d<quat> output_rotations,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const G1LegConfig& config,
    vec3 desired_contact,
    vec3 ankle_target,
    G1LegSolveResult& output,
    char* error,
    int error_capacity)
{
    if (!g1_ik_pose_inputs_validate(
            output_rotations, local_positions,
            baseline_rotations, parents, config,
            error, error_capacity)) {
        return false;
    }
    if (!g1_ik_vec3_is_runtime_value(desired_contact) ||
        !g1_ik_vec3_is_runtime_value(ankle_target)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 contact residual target is invalid");
    }

    array1d<quat> candidate_pose(output_rotations);
    array1d<vec3> candidate_global_positions(G1_BoneCount);
    array1d<quat> candidate_global_rotations(G1_BoneCount);
    G1LegSolveResult aggregate = {};
    bool all_reachable = true;
    bool any_limited = false;
    float maximum_correction = 0.0f;
    double residual_precise_m = DBL_MAX;
    float residual_m = FLT_MAX;
    for (int iteration = 0;
         iteration < G1ContactSolveMaximumIterations;
         ++iteration) {
        G1LegSolveResult solve = {};
        if (!g1_apply_named_position_ik(
                candidate_pose, local_positions, baseline_rotations,
                parents, config, ankle_target,
                solve, error, error_capacity)) {
            return false;
        }
        all_reachable = all_reachable && solve.reachable;
        any_limited = any_limited || solve.correction_limited;
        maximum_correction =
            solve.max_correction_radians > maximum_correction
                ? solve.max_correction_radians
                : maximum_correction;
        aggregate = solve;
        aggregate.iterations = iteration + 1;
        aggregate.iteration_provenance =
            g1_ik_contact_iteration_provenance(
                aggregate.iterations);
        if (!g1_ik_checked_forward_kinematics(
                candidate_global_positions,
                candidate_global_rotations,
                local_positions, candidate_pose, parents,
                error, error_capacity) ||
            !ik_checked_distance_precise(
                residual_precise_m, residual_m,
                candidate_global_positions(config.contact),
                desired_contact)) {
            return g1_ik_error(
                error, error_capacity,
                "G1 contact residual FK/math became invalid");
        }
        aggregate.contact_residual_m = residual_m;
        if (g1_ik_contact_residual_is_converged_precise(
                residual_precise_m)) {
            break;
        }
        if (iteration + 1 < G1ContactSolveMaximumIterations) {
            vec3 residual;
            vec3 next_ankle_target;
            if (!ik_checked_vec3_subtract(
                    residual, desired_contact,
                    candidate_global_positions(config.contact)) ||
                !ik_checked_vec3_add(
                    next_ankle_target, ankle_target, residual)) {
                return g1_ik_error(
                    error, error_capacity,
                    "G1 contact residual update overflowed");
            }
            ankle_target = next_ankle_target;
        }
    }

    if (!g1_ik_checked_forward_kinematics(
            candidate_global_positions, candidate_global_rotations,
            local_positions, candidate_pose, parents,
            error, error_capacity) ||
        !ik_checked_distance_precise(
            residual_precise_m, residual_m,
            candidate_global_positions(config.contact),
            desired_contact)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 contact final residual could not be verified");
    }
    aggregate.reachable = all_reachable;
    aggregate.correction_limited = any_limited;
    aggregate.max_correction_radians = maximum_correction;
    aggregate.contact_residual_m = residual_m;
    aggregate.safe_stop_requested =
        aggregate.safe_stop_requested || !all_reachable || any_limited ||
        !g1_ik_contact_residual_is_converged_precise(
            residual_precise_m);
    if (!g1_ik_leg_result_is_valid(aggregate) ||
        aggregate.max_correction_radians >
            config.max_correction_radians) {
        return g1_ik_error(
            error, error_capacity,
            "G1 contact residual produced invalid result");
    }

    std::memcpy(
        output_rotations.data, candidate_pose.data,
        static_cast<size_t>(G1_BoneCount) * sizeof(quat));
    output = aggregate;
    return true;
}

static inline bool g1_apply_named_contact_position_ik(
    slice1d<quat> output_rotations,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const G1LegConfig& config,
    vec3 desired_contact,
    G1LegSolveResult& output,
    char* error,
    int error_capacity)
{
    if (!g1_ik_pose_inputs_validate(
            output_rotations, local_positions,
            baseline_rotations, parents, config,
            error, error_capacity)) {
        return false;
    }
    if (!g1_ik_vec3_is_runtime_value(desired_contact)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 contact residual target is invalid");
    }

    array1d<vec3> baseline_global_positions(G1_BoneCount);
    array1d<quat> baseline_global_rotations(G1_BoneCount);
    if (!g1_ik_checked_forward_kinematics(
            baseline_global_positions, baseline_global_rotations,
            local_positions, baseline_rotations, parents,
            error, error_capacity)) {
        return false;
    }
    vec3 contact_offset;
    vec3 ankle_target;
    if (!ik_checked_vec3_subtract(
            contact_offset,
            baseline_global_positions(config.contact),
            baseline_global_positions(config.ankle)) ||
        !ik_checked_vec3_subtract(
            ankle_target, desired_contact, contact_offset)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 contact residual initial target overflowed");
    }
    return g1_apply_named_contact_position_ik_from_initial_target(
        output_rotations,
        local_positions,
        baseline_rotations,
        parents,
        config,
        desired_contact,
        ankle_target,
        output,
        error,
        error_capacity);
}

struct G1FootOrientationResult
{
    bool applied = false;
    bool correction_limited = false;
    bool safe_stop_requested = false;
    quat target_global_rotation;
    float requested_correction_radians = 0.0f;
    float correction_radians = 0.0f;
};

static inline bool g1_ik_checked_quat_from_xy(
    quat& output, vec3 forward, vec3 up)
{
    if (!ik_vec3_is_unit(forward) || !ik_vec3_is_unit(up)) {
        return false;
    }
    double forward_up = 0.0;
    vec3 side;
    vec3 corrected_forward;
    if (!ik_checked_dot(forward_up, forward, up) ||
        fabs(forward_up) > 2.0e-5 ||
        !ik_checked_cross(side, forward, up) ||
        !ik_checked_normalize(side, side) ||
        !ik_checked_cross(corrected_forward, up, side) ||
        !ik_checked_normalize(corrected_forward, corrected_forward)) {
        return false;
    }

    const double c0x = static_cast<double>(corrected_forward.x);
    const double c0y = static_cast<double>(corrected_forward.y);
    const double c0z = static_cast<double>(corrected_forward.z);
    const double c1x = static_cast<double>(up.x);
    const double c1y = static_cast<double>(up.y);
    const double c1z = static_cast<double>(up.z);
    const double c2x = static_cast<double>(side.x);
    const double c2y = static_cast<double>(side.y);
    const double c2z = static_cast<double>(side.z);
    quat candidate;
    if (c2z < 0.0) {
        if (c0x > c1y) {
            if (!ik_checked_quat_from_double(
                    candidate,
                    c1z - c2y,
                    1.0 + c0x - c1y - c2z,
                    c0y + c1x,
                    c2x + c0z)) {
                return false;
            }
        } else if (!ik_checked_quat_from_double(
                       candidate,
                       c2x - c0z,
                       c0y + c1x,
                       1.0 - c0x + c1y - c2z,
                       c1z + c2y)) {
            return false;
        }
    } else if (c0x < -c1y) {
        if (!ik_checked_quat_from_double(
                candidate,
                c0y - c1x,
                c2x + c0z,
                c1z + c2y,
                1.0 - c0x - c1y + c2z)) {
            return false;
        }
    } else if (!ik_checked_quat_from_double(
                   candidate,
                   1.0 + c0x + c1y + c2z,
                   c1z - c2y,
                   c2x - c0z,
                   c0y - c1x)) {
        return false;
    }

    vec3 mapped_forward;
    vec3 mapped_up;
    double forward_alignment = 0.0;
    double up_alignment = 0.0;
    if (!ik_checked_quat_rotate(
            mapped_forward, candidate, vec3(1.0f, 0.0f, 0.0f)) ||
        !ik_checked_quat_rotate(
            mapped_up, candidate, vec3(0.0f, 1.0f, 0.0f)) ||
        !ik_checked_dot(
            forward_alignment, mapped_forward, corrected_forward) ||
        !ik_checked_dot(up_alignment, mapped_up, up) ||
        forward_alignment < 0.99999 || up_alignment < 0.99999) {
        return false;
    }
    output = candidate;
    return true;
}

static inline bool g1_surface_aligned_foot_rotation(
    quat& output,
    quat current_global_rotation,
    const G1LegConfig& config,
    vec3 surface_normal,
    char* error,
    int error_capacity)
{
    const bool error_alias = error != NULL && error_capacity > 0 &&
        (g1_ik_memory_ranges_overlap(
             error, static_cast<size_t>(error_capacity),
             &output, sizeof(output)) ||
         g1_ik_memory_ranges_overlap(
             error, static_cast<size_t>(error_capacity),
             &config, sizeof(config)));
    if (error_alias) {
        return false;
    }
    if (g1_ik_memory_ranges_overlap(
            &output, sizeof(output), &config, sizeof(config))) {
        return g1_ik_error(
            error, error_capacity,
            "G1 foot orientation output must not alias its config");
    }
    if (!g1_foot_runtime_config_validate(
            config, error, error_capacity)) {
        return false;
    }
    if (!ik_quat_is_unit(current_global_rotation)) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation received a non-finite/non-unit rotation",
            config.name);
    }
    if (!g1_ik_surface_normal_is_valid(surface_normal)) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation requires an upward unit surface normal",
            config.name);
    }

    quat normalized_current_global_rotation;
    if (!ik_checked_quat_normalize(
            normalized_current_global_rotation,
            current_global_rotation)) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation could not normalize its rotation",
            config.name);
    }
    vec3 up;
    vec3 current_forward;
    double projection_dot = 0.0;
    vec3 normal_component;
    vec3 projected_forward;
    double projected_length = 0.0;
    float rounded_projected_length = 0.0f;
    if (!ik_checked_normalize(up, surface_normal) ||
        !ik_checked_quat_rotate(
            current_forward,
            normalized_current_global_rotation,
            config.foot_forward_local) ||
        !ik_checked_dot(projection_dot, current_forward, up) ||
        !ik_checked_vec3_scale(
            normal_component, up, projection_dot) ||
        !ik_checked_vec3_subtract(
            projected_forward, current_forward, normal_component) ||
        !ik_checked_vec3_norm(
            projected_length,
            rounded_projected_length,
            projected_forward)) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation heading projection overflowed",
            config.name);
    }
    vec3 forward;
    if (projected_length < 1.0e-6) {
        const vec3 fallback_axis =
            fabs(static_cast<double>(up.x)) < 0.75
                ? vec3(1.0f, 0.0f, 0.0f)
                : vec3(0.0f, 0.0f, 1.0f);
        double fallback_dot = 0.0;
        vec3 fallback_normal_component;
        vec3 fallback_projected;
        if (!ik_checked_dot(fallback_dot, fallback_axis, up) ||
            !ik_checked_vec3_scale(
                fallback_normal_component, up, fallback_dot) ||
            !ik_checked_vec3_subtract(
                fallback_projected,
                fallback_axis,
                fallback_normal_component) ||
            !ik_checked_normalize(forward, fallback_projected)) {
            return g1_ik_error(
                error, error_capacity,
                "%s foot orientation fallback heading is invalid",
                config.name);
        }
    } else if (!ik_checked_normalize(forward, projected_forward)) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation could not normalize its heading",
            config.name);
    }

    quat candidate;
    if (!g1_ik_checked_quat_from_xy(candidate, forward, up)) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation produced an invalid target frame",
            config.name);
    }
    output = candidate;
    return true;
}

static inline bool g1_ik_orientation_aliases_inputs(
    const slice1d<quat> output_rotations,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    G1FootOrientationResult& output,
    const G1LegConfig& config,
    char* error,
    int error_capacity)
{
    const size_t rotation_bytes =
        static_cast<size_t>(G1_BoneCount) * sizeof(quat);
    const size_t position_bytes =
        static_cast<size_t>(G1_BoneCount) * sizeof(vec3);
    const size_t parent_bytes =
        static_cast<size_t>(G1_BoneCount) * sizeof(int);
    const bool pose_alias =
        g1_ik_memory_ranges_overlap(
            output_rotations.data, rotation_bytes,
            baseline_rotations.data, rotation_bytes) ||
        g1_ik_memory_ranges_overlap(
            output_rotations.data, rotation_bytes,
            local_positions.data, position_bytes) ||
        g1_ik_memory_ranges_overlap(
            output_rotations.data, rotation_bytes,
            parents.data, parent_bytes) ||
        g1_ik_memory_ranges_overlap(
            output_rotations.data, rotation_bytes,
            &config, sizeof(config));
    const bool result_alias =
        g1_ik_memory_ranges_overlap(
            &output, sizeof(output),
            output_rotations.data, rotation_bytes) ||
        g1_ik_memory_ranges_overlap(
            &output, sizeof(output),
            baseline_rotations.data, rotation_bytes) ||
        g1_ik_memory_ranges_overlap(
            &output, sizeof(output),
            local_positions.data, position_bytes) ||
        g1_ik_memory_ranges_overlap(
            &output, sizeof(output),
            parents.data, parent_bytes) ||
        g1_ik_memory_ranges_overlap(
            &output, sizeof(output), &config, sizeof(config));
    bool error_alias = false;
    if (error != NULL && error_capacity > 0) {
        const size_t error_bytes = static_cast<size_t>(error_capacity);
        error_alias =
            g1_ik_memory_ranges_overlap(
                error, error_bytes,
                output_rotations.data, rotation_bytes) ||
            g1_ik_memory_ranges_overlap(
                error, error_bytes,
                baseline_rotations.data, rotation_bytes) ||
            g1_ik_memory_ranges_overlap(
                error, error_bytes,
                local_positions.data, position_bytes) ||
            g1_ik_memory_ranges_overlap(
                error, error_bytes,
                parents.data, parent_bytes) ||
            g1_ik_memory_ranges_overlap(
                error, error_bytes, &output, sizeof(output)) ||
            g1_ik_memory_ranges_overlap(
                error, error_bytes, &config, sizeof(config));
    }
    if (error_alias) {
        return true;
    }
    if (pose_alias || result_alias) {
        g1_ik_error(
            error, error_capacity,
            "G1 foot orientation mutable outputs must not alias inputs");
        return true;
    }
    return false;
}

static inline bool g1_ik_orientation_error_aliases_memory(
    const slice1d<quat> output_rotations,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    G1FootOrientationResult& output,
    const G1LegConfig& config,
    char* error,
    int error_capacity)
{
    if (error == NULL || error_capacity <= 0) {
        return false;
    }
    const size_t error_bytes = static_cast<size_t>(error_capacity);
    const size_t output_bytes = output_rotations.size > 0
        ? static_cast<size_t>(output_rotations.size) * sizeof(quat)
        : 0;
    const size_t position_bytes = local_positions.size > 0
        ? static_cast<size_t>(local_positions.size) * sizeof(vec3)
        : 0;
    const size_t baseline_bytes = baseline_rotations.size > 0
        ? static_cast<size_t>(baseline_rotations.size) * sizeof(quat)
        : 0;
    const size_t parent_bytes = parents.size > 0
        ? static_cast<size_t>(parents.size) * sizeof(int)
        : 0;
    return g1_ik_memory_ranges_overlap(
               error, error_bytes,
               output_rotations.data, output_bytes) ||
           g1_ik_memory_ranges_overlap(
               error, error_bytes,
               local_positions.data, position_bytes) ||
           g1_ik_memory_ranges_overlap(
               error, error_bytes,
               baseline_rotations.data, baseline_bytes) ||
           g1_ik_memory_ranges_overlap(
               error, error_bytes, parents.data, parent_bytes) ||
           g1_ik_memory_ranges_overlap(
               error, error_bytes, &output, sizeof(output)) ||
           g1_ik_memory_ranges_overlap(
               error, error_bytes, &config, sizeof(config));
}

static inline bool g1_apply_named_foot_orientation(
    slice1d<quat> output_rotations,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const G1LegConfig& config,
    vec3 surface_normal,
    G1FootOrientationResult& output,
    char* error,
    int error_capacity)
{
    if (g1_ik_orientation_error_aliases_memory(
            output_rotations, local_positions, baseline_rotations,
            parents, output, config, error, error_capacity)) {
        return false;
    }
    if (output_rotations.size != G1_BoneCount ||
        output_rotations.data == NULL ||
        local_positions.size != G1_BoneCount ||
        local_positions.data == NULL ||
        baseline_rotations.size != G1_BoneCount ||
        baseline_rotations.data == NULL ||
        parents.size != G1_BoneCount ||
        parents.data == NULL) {
        return g1_ik_error(
            error, error_capacity,
            "G1 foot orientation requires non-null exact 31-bone slices");
    }
    if (g1_ik_orientation_aliases_inputs(
            output_rotations, local_positions, baseline_rotations,
            parents, output, config, error, error_capacity)) {
        return false;
    }
    if (!g1_ik_pose_inputs_validate(
            output_rotations, local_positions, baseline_rotations,
            parents, config, error, error_capacity)) {
        return false;
    }
    if (!g1_ik_surface_normal_is_valid(surface_normal)) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation requires an upward unit surface normal",
            config.name);
    }

    array1d<quat> candidate_pose(output_rotations);
    array1d<vec3> current_global_positions(G1_BoneCount);
    array1d<quat> current_global_rotations(G1_BoneCount);
    if (!g1_ik_checked_forward_kinematics(
            current_global_positions, current_global_rotations,
            local_positions, candidate_pose, parents,
            error, error_capacity)) {
        return false;
    }

    quat target_global;
    if (!g1_surface_aligned_foot_rotation(
            target_global,
            current_global_rotations(config.contact),
            config, surface_normal,
            error, error_capacity)) {
        return false;
    }
    const int parent = parents.data[config.contact];
    if (parent < 0 || parent >= G1_BoneCount) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation contact has an invalid parent",
            config.name);
    }
    quat desired_local;
    if (!ik_checked_quat_inverse_multiply(
            desired_local,
            current_global_rotations(parent),
            target_global)) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation could not materialize a local target",
            config.name);
    }
    IKClampResult bounded = {};
    if (!ik_clamp_local_delta(
            bounded,
            baseline_rotations.data[config.contact],
            desired_local,
            config.max_correction_radians)) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation could not bound its local correction",
            config.name);
    }
    candidate_pose(config.contact) = bounded.value;

    array1d<vec3> verified_global_positions(G1_BoneCount);
    array1d<quat> verified_global_rotations(G1_BoneCount);
    if (!g1_ik_checked_forward_kinematics(
            verified_global_positions, verified_global_rotations,
            local_positions, candidate_pose, parents,
            error, error_capacity)) {
        return false;
    }

    G1FootOrientationResult candidate = {};
    candidate.applied = true;
    candidate.correction_limited = bounded.limited;
    candidate.safe_stop_requested = bounded.limited;
    candidate.target_global_rotation = target_global;
    candidate.requested_correction_radians = bounded.requested_radians;
    candidate.correction_radians = bounded.actual_radians;
    if (!ik_quat_is_unit(candidate.target_global_rotation) ||
        !g1_ik_float_is_runtime_value(
            candidate.requested_correction_radians) ||
        candidate.requested_correction_radians < 0.0f ||
        !g1_ik_float_is_runtime_value(candidate.correction_radians) ||
        candidate.correction_radians < 0.0f ||
        candidate.correction_radians > config.max_correction_radians ||
        candidate.safe_stop_requested != candidate.correction_limited) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation produced an invalid bounded result",
            config.name);
    }

    std::memcpy(
        output_rotations.data, candidate_pose.data,
        static_cast<size_t>(G1_BoneCount) * sizeof(quat));
    output = candidate;
    return true;
}

static inline bool g1_apply_named_physical_sole_ik(
    slice1d<quat> output_rotations,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const G1LegConfig& config,
    vec3 desired_sole_center,
    vec3 surface_normal,
    G1LegSolveResult& position_output,
    G1FootOrientationResult& orientation_output,
    char* error,
    int error_capacity)
{
    const size_t rotation_bytes =
        static_cast<size_t>(G1_BoneCount) * sizeof(quat);
    const size_t position_bytes =
        static_cast<size_t>(G1_BoneCount) * sizeof(vec3);
    const size_t parent_bytes =
        static_cast<size_t>(G1_BoneCount) * sizeof(int);
    const auto result_aliases = [
        &output_rotations,
        &local_positions,
        &baseline_rotations,
        &parents,
        &config,
        rotation_bytes,
        position_bytes,
        parent_bytes](const void* result, size_t bytes) {
        return g1_ik_memory_ranges_overlap(
                   result, bytes,
                   output_rotations.data, rotation_bytes) ||
               g1_ik_memory_ranges_overlap(
                   result, bytes,
                   local_positions.data, position_bytes) ||
               g1_ik_memory_ranges_overlap(
                   result, bytes,
                   baseline_rotations.data, rotation_bytes) ||
               g1_ik_memory_ranges_overlap(
                   result, bytes,
                   parents.data, parent_bytes) ||
               g1_ik_memory_ranges_overlap(
                   result, bytes, &config, sizeof(config));
    };
    const bool outputs_alias =
        result_aliases(&position_output, sizeof(position_output)) ||
        result_aliases(&orientation_output, sizeof(orientation_output)) ||
        g1_ik_memory_ranges_overlap(
            &position_output, sizeof(position_output),
            &orientation_output, sizeof(orientation_output));
    bool error_alias = false;
    if (error != NULL && error_capacity > 0) {
        const size_t error_bytes = static_cast<size_t>(error_capacity);
        error_alias =
            g1_ik_memory_ranges_overlap(
                error, error_bytes,
                output_rotations.data, rotation_bytes) ||
            g1_ik_memory_ranges_overlap(
                error, error_bytes,
                local_positions.data, position_bytes) ||
            g1_ik_memory_ranges_overlap(
                error, error_bytes,
                baseline_rotations.data, rotation_bytes) ||
            g1_ik_memory_ranges_overlap(
                error, error_bytes,
                parents.data, parent_bytes) ||
            g1_ik_memory_ranges_overlap(
                error, error_bytes,
                &position_output, sizeof(position_output)) ||
            g1_ik_memory_ranges_overlap(
                error, error_bytes,
                &orientation_output, sizeof(orientation_output)) ||
            g1_ik_memory_ranges_overlap(
                error, error_bytes, &config, sizeof(config));
    }
    if (error_alias) return false;
    if (outputs_alias) {
        return g1_ik_error(
            error, error_capacity,
            "G1 physical sole IK outputs must not alias inputs");
    }
    if (error_capacity < 0 ||
        !g1_ik_pose_inputs_validate(
            output_rotations,
            local_positions,
            baseline_rotations,
            parents,
            config,
            error,
            error_capacity) ||
        !g1_ik_vec3_is_runtime_value(desired_sole_center) ||
        !g1_ik_surface_normal_is_valid(surface_normal)) {
        return false;
    }

    array1d<vec3> baseline_global_positions(G1_BoneCount);
    array1d<quat> baseline_global_rotations(G1_BoneCount);
    if (!g1_ik_checked_forward_kinematics(
            baseline_global_positions,
            baseline_global_rotations,
            local_positions,
            baseline_rotations,
            parents,
            error,
            error_capacity)) {
        return false;
    }

    G1PhysicalSolePositionTarget physical_target = {};
    if (!g1_physical_sole_position_target(
            physical_target,
            baseline_global_positions(config.contact),
            baseline_global_rotations(config.contact),
            baseline_global_positions(config.ankle),
            config,
            desired_sole_center,
            surface_normal,
            error,
            error_capacity)) {
        return false;
    }

    array1d<quat> candidate_pose(baseline_rotations);
    G1LegSolveResult position_candidate = {};
    if (!g1_apply_named_contact_position_ik_from_initial_target(
            candidate_pose,
            local_positions,
            baseline_rotations,
            parents,
            config,
            physical_target.contact_origin,
            physical_target.ankle_target,
            position_candidate,
            error,
            error_capacity)) {
        return false;
    }

    array1d<vec3> position_global_positions(G1_BoneCount);
    array1d<quat> position_global_rotations(G1_BoneCount);
    if (!g1_ik_checked_forward_kinematics(
            position_global_positions,
            position_global_rotations,
            local_positions,
            candidate_pose,
            parents,
            error,
            error_capacity)) {
        return false;
    }
    const int contact_parent = parents.data[config.contact];
    quat desired_contact_local;
    if (contact_parent < 0 || contact_parent >= G1_BoneCount ||
        !ik_checked_quat_inverse_multiply(
            desired_contact_local,
            position_global_rotations(contact_parent),
            physical_target.contact_rotation)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 physical sole IK could not materialize frozen rotation");
    }
    IKClampResult bounded_orientation = {};
    if (!ik_clamp_local_delta(
            bounded_orientation,
            baseline_rotations.data[config.contact],
            desired_contact_local,
            config.max_correction_radians)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 physical sole IK could not bound frozen rotation");
    }
    candidate_pose(config.contact) = bounded_orientation.value;

    G1FootOrientationResult orientation_candidate = {};
    orientation_candidate.applied = true;
    orientation_candidate.correction_limited =
        bounded_orientation.limited;
    orientation_candidate.safe_stop_requested =
        bounded_orientation.limited;
    orientation_candidate.target_global_rotation =
        physical_target.contact_rotation;
    orientation_candidate.requested_correction_radians =
        bounded_orientation.requested_radians;
    orientation_candidate.correction_radians =
        bounded_orientation.actual_radians;

    array1d<vec3> final_global_positions(G1_BoneCount);
    array1d<quat> final_global_rotations(G1_BoneCount);
    vec3 final_sole_center;
    double sole_residual_precise_m = DBL_MAX;
    float sole_residual_m = FLT_MAX;
    vec3 normalized_surface_normal;
    vec3 final_sole_normal;
    vec3 final_forward;
    vec3 frozen_forward;
    double normal_alignment = 0.0;
    double heading_alignment = 0.0;
    if (!g1_ik_checked_forward_kinematics(
            final_global_positions,
            final_global_rotations,
            local_positions,
            candidate_pose,
            parents,
            error,
            error_capacity) ||
        !g1_ik_checked_physical_sole_centroid(
            final_sole_center,
            final_global_positions(config.contact),
            final_global_rotations(config.contact),
            config) ||
        !ik_checked_distance_precise(
            sole_residual_precise_m,
            sole_residual_m,
            final_sole_center,
            desired_sole_center) ||
        !ik_checked_normalize(
            normalized_surface_normal,
            surface_normal) ||
        !ik_checked_quat_rotate(
            final_sole_normal,
            final_global_rotations(config.contact),
            config.sole_normal_local) ||
        !ik_checked_quat_rotate(
            final_forward,
            final_global_rotations(config.contact),
            config.foot_forward_local) ||
        !ik_checked_quat_rotate(
            frozen_forward,
            physical_target.contact_rotation,
            config.foot_forward_local) ||
        !ik_checked_dot(
            normal_alignment,
            final_sole_normal,
            normalized_surface_normal) ||
        !ik_checked_dot(
            heading_alignment,
            final_forward,
            frozen_forward)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 physical sole IK final verification failed");
    }

    position_candidate.contact_residual_m = sole_residual_m;
    position_candidate.safe_stop_requested =
        !position_candidate.reachable ||
        position_candidate.correction_limited ||
        !g1_ik_contact_residual_is_converged_precise(
            sole_residual_precise_m);
    if (position_candidate.reachable &&
        !position_candidate.correction_limited) {
        position_candidate.clamped_ankle_target =
            position_candidate.requested_ankle_target;
        position_candidate.clamped_distance_m =
            position_candidate.raw_distance_m;
    }
    if ((!orientation_candidate.correction_limited &&
         (normal_alignment < 0.99999 ||
          heading_alignment < 0.99999)) ||
        !g1_ik_leg_result_is_valid(position_candidate) ||
        position_candidate.max_correction_radians >
            config.max_correction_radians ||
        !ik_quat_is_unit(
            orientation_candidate.target_global_rotation) ||
        !g1_ik_float_is_runtime_value(
            orientation_candidate.requested_correction_radians) ||
        orientation_candidate.requested_correction_radians < 0.0f ||
        !g1_ik_float_is_runtime_value(
            orientation_candidate.correction_radians) ||
        orientation_candidate.correction_radians < 0.0f ||
        orientation_candidate.correction_radians >
            config.max_correction_radians ||
        orientation_candidate.safe_stop_requested !=
            orientation_candidate.correction_limited) {
        return g1_ik_error(
            error, error_capacity,
            "G1 physical sole IK produced invalid diagnostics");
    }

    std::memcpy(
        output_rotations.data,
        candidate_pose.data,
        rotation_bytes);
    position_output = position_candidate;
    orientation_output = orientation_candidate;
    return true;
}
