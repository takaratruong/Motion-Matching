#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "database.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "g1_skeleton.h"
#include "terrain_runtime.h"

#include <cstdarg>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>

struct G1LegConfig
{
    const char* name;
    int hip;
    int knee;
    int ankle;
    int contact;
    vec3 knee_hinge_axis_local;
    vec3 foot_forward_local;
    vec3 sole_normal_local;
    vec3 foot_sphere_centers_local[4];
    vec3 sole_points_local[4];
    float foot_sphere_radius_m;
    vec3 thigh_start_local;
    vec3 thigh_end_local;
    float thigh_radius_m;
    vec3 shin_start_local;
    vec3 shin_end_local;
    float shin_radius_m;
    float reach_buffer_m;
    float planted_clearance_m;
    float swing_clearance_m;
    float max_swing_lift_m;
    float max_correction_radians;
};

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

static inline G1LegConfig g1_leg_config(
    const char* name, int hip, int knee, int ankle, int contact)
{
    G1LegConfig config = {};
    config.name = name;
    config.hip = hip;
    config.knee = knee;
    config.ankle = ankle;
    config.contact = contact;
    config.knee_hinge_axis_local = vec3(0.0f, 0.0f, -1.0f);
    config.foot_forward_local = vec3(1.0f, 0.0f, 0.0f);
    config.sole_normal_local = vec3(0.0f, 1.0f, 0.0f);
    config.foot_sphere_radius_m = 0.02f;
    config.foot_sphere_centers_local[0] =
        vec3(-0.05f, -0.03f, -0.025f);
    config.foot_sphere_centers_local[1] =
        vec3(-0.05f, -0.03f, +0.025f);
    config.foot_sphere_centers_local[2] =
        vec3(+0.12f, -0.03f, -0.030f);
    config.foot_sphere_centers_local[3] =
        vec3(+0.12f, -0.03f, +0.030f);
    for (int i = 0; i < 4; ++i) {
        config.sole_points_local[i] =
            config.foot_sphere_centers_local[i] -
            config.sole_normal_local * config.foot_sphere_radius_m;
    }
    config.thigh_start_local = vec3(0.0f, -0.02f, 0.0f);
    config.thigh_end_local = vec3(-0.078f, -0.17f, 0.0f);
    config.thigh_radius_m = 0.05f;
    config.shin_start_local = vec3(0.0f, -0.05f, 0.0f);
    config.shin_end_local = vec3(0.0f, -0.28f, 0.0f);
    config.shin_radius_m = 0.04f;
    config.reach_buffer_m = 0.015f;
    config.planted_clearance_m = 0.005f;
    config.swing_clearance_m = 0.015f;
    config.max_swing_lift_m = 0.08f;
    config.max_correction_radians = 0.35f;
    return config;
}

static inline G1LegConfig g1_left_leg_config()
{
    return g1_leg_config(
        "left", G1_LeftHipYaw, G1_LeftKnee, G1_LeftAnkle, G1_LeftToe);
}

static inline G1LegConfig g1_right_leg_config()
{
    return g1_leg_config(
        "right", G1_RightHipYaw, G1_RightKnee, G1_RightAnkle, G1_RightToe);
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
