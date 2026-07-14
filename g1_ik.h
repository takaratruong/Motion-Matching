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

enum G1SurfaceQueryStatus
{
    G1SurfaceQueryValid = 0,
    G1SurfaceQueryOutside = 1,
    G1SurfaceQueryInvalid = 2
};

struct G1SurfaceSample
{
    float height = 0.0f;
    vec3 normal;
};

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
    vec3 sole_center;
    float horizontal_drift_m = 0.0f;
};

static inline bool g1_ik_float_is_runtime_value(float value)
{
    return terrain_float_is_normal_or_zero_query(value);
}

static inline bool g1_ik_vec3_is_runtime_value(vec3 value)
{
    return g1_ik_float_is_runtime_value(value.x) &&
           g1_ik_float_is_runtime_value(value.y) &&
           g1_ik_float_is_runtime_value(value.z);
}

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

static inline bool g1_ik_dt_is_exact_25_hz(float dt)
{
    // Exact binary32 encoding of 1.0f / 25.0f (0.04f).
    return terrain_float_bits(dt) == UINT32_C(0x3d23d70a);
}

static inline bool g1_ik_surface_normal_is_valid(vec3 normal)
{
    if (!g1_ik_vec3_is_runtime_value(normal) || normal.y <= 0.0f) {
        return false;
    }
    const volatile double square_x =
        static_cast<double>(normal.x) * static_cast<double>(normal.x);
    const volatile double square_y =
        static_cast<double>(normal.y) * static_cast<double>(normal.y);
    const volatile double square_z =
        static_cast<double>(normal.z) * static_cast<double>(normal.z);
    const volatile double first_sum = square_x + square_y;
    const volatile double square_sum = first_sum + square_z;
    return terrain_double_is_finite(square_sum) &&
           fabs(square_sum - 1.0) <= 2.0e-5;
}

static inline bool g1_ik_surface_target_is_valid(
    const G1SurfaceTarget& target)
{
    return g1_ik_vec3_is_runtime_value(target.point) &&
           g1_ik_surface_normal_is_valid(target.normal);
}

// This is the fail-closed low-level query used by both the lock runtime and
// later collision probes. It intentionally never returns exterior_height.
static inline G1SurfaceQueryStatus g1_surface_query_v2(
    G1SurfaceSample& output,
    const heightfield& field,
    float input_x,
    float input_z)
{
    if (field.version != 2 || !terrain_heightfield_is_queryable(field)) {
        return G1SurfaceQueryInvalid;
    }

    float canonical_x = 0.0f;
    float canonical_z = 0.0f;
    if (!terrain_v2_query_coordinate(input_x, canonical_x) ||
        !terrain_v2_query_coordinate(input_z, canonical_z)) {
        return G1SurfaceQueryInvalid;
    }

    const volatile double maximum_x_product =
        static_cast<double>(field.nx - 1) *
        static_cast<double>(field.cell_size);
    const volatile double maximum_z_product =
        static_cast<double>(field.nz - 1) *
        static_cast<double>(field.cell_size);
    const volatile double maximum_x =
        static_cast<double>(field.origin_x) + maximum_x_product;
    const volatile double maximum_z =
        static_cast<double>(field.origin_z) + maximum_z_product;
    if (!terrain_double_is_finite(maximum_x) ||
        !terrain_double_is_finite(maximum_z)) {
        return G1SurfaceQueryInvalid;
    }
    const double x = static_cast<double>(canonical_x);
    const double z = static_cast<double>(canonical_z);
    if (x < static_cast<double>(field.origin_x) || x > maximum_x ||
        z < static_cast<double>(field.origin_z) || z > maximum_z) {
        return G1SurfaceQueryOutside;
    }

    heightfield_cell cell = {};
    if (!terrain_v2_locate_cell(
            field, canonical_x, canonical_z, cell)) {
        return G1SurfaceQueryInvalid;
    }
    double h00 = 0.0;
    double h10 = 0.0;
    double h01 = 0.0;
    double h11 = 0.0;
    if (!terrain_v2_cell_heights(
            field, cell, h00, h10, h01, h11)) {
        return G1SurfaceQueryInvalid;
    }

    double height = 0.0;
    double slope_x = 0.0;
    double slope_z = 0.0;
    if (cell.tx >= cell.tz) {
        const volatile double difference_x = h10 - h00;
        const volatile double x_term = cell.tx * difference_x;
        const volatile double first_sum = h00 + x_term;
        const volatile double difference_z = h11 - h10;
        const volatile double z_term = cell.tz * difference_z;
        const volatile double final_sum = first_sum + z_term;
        const volatile double divided_x =
            difference_x / static_cast<double>(field.cell_size);
        const volatile double divided_z =
            difference_z / static_cast<double>(field.cell_size);
        height = final_sum;
        slope_x = divided_x;
        slope_z = divided_z;
    } else {
        const volatile double difference_x = h11 - h01;
        const volatile double x_term = cell.tx * difference_x;
        const volatile double first_sum = h00 + x_term;
        const volatile double difference_z = h01 - h00;
        const volatile double z_term = cell.tz * difference_z;
        const volatile double final_sum = first_sum + z_term;
        const volatile double divided_x =
            difference_x / static_cast<double>(field.cell_size);
        const volatile double divided_z =
            difference_z / static_cast<double>(field.cell_size);
        height = final_sum;
        slope_x = divided_x;
        slope_z = divided_z;
    }
    if (!terrain_double_is_finite(slope_x) ||
        !terrain_double_is_finite(slope_z)) {
        return G1SurfaceQueryInvalid;
    }

    const volatile double normal_x = -slope_x;
    const volatile double normal_y = 1.0;
    const volatile double normal_z = -slope_z;
    const double absolute_x = fabs(normal_x);
    const double absolute_y = fabs(normal_y);
    const double absolute_z = fabs(normal_z);
    const double maximum_xy =
        absolute_x > absolute_y ? absolute_x : absolute_y;
    const double maximum =
        maximum_xy > absolute_z ? maximum_xy : absolute_z;
    if (!terrain_double_is_finite(maximum) || maximum <= 0.0) {
        return G1SurfaceQueryInvalid;
    }
    const volatile double scaled_x = normal_x / maximum;
    const volatile double scaled_y = normal_y / maximum;
    const volatile double scaled_z = normal_z / maximum;
    const volatile double square_x = scaled_x * scaled_x;
    const volatile double square_y = scaled_y * scaled_y;
    const volatile double square_z = scaled_z * scaled_z;
    const volatile double first_square_sum = square_x + square_y;
    const volatile double square_sum = first_square_sum + square_z;
    if (!terrain_double_is_finite(square_sum) || square_sum <= 0.0) {
        return G1SurfaceQueryInvalid;
    }
    const volatile double normal_length = sqrt(square_sum);
    if (!terrain_double_is_finite(normal_length) ||
        normal_length <= 0.0) {
        return G1SurfaceQueryInvalid;
    }
    const volatile double unit_x = scaled_x / normal_length;
    const volatile double unit_y = scaled_y / normal_length;
    const volatile double unit_z = scaled_z / normal_length;

    G1SurfaceSample candidate = {};
    if (!terrain_v2_round_output(height, candidate.height) ||
        !terrain_v2_round_output(unit_x, candidate.normal.x) ||
        !terrain_v2_round_output(unit_y, candidate.normal.y) ||
        !terrain_v2_round_output(unit_z, candidate.normal.z) ||
        !g1_ik_surface_normal_is_valid(candidate.normal)) {
        return G1SurfaceQueryInvalid;
    }
    output = candidate;
    return G1SurfaceQueryValid;
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
        !g1_ik_vec3_is_runtime_value(target.sole_center) ||
        !g1_ik_float_is_runtime_value(target.horizontal_drift_m) ||
        target.horizontal_drift_m < 0.0f ||
        (!target.locked && target.horizontal_drift_m != 0.0f)) {
        return false;
    }
    const bool has_surface_normal =
        !g1_ik_vec3_is_zero(target.surface.normal);
    return (!target.position_active && !has_surface_normal) ||
           (has_surface_normal &&
            g1_ik_surface_target_is_valid(target.surface));
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
    if (!g1_ik_dt_is_exact_25_hz(dt)) {
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
    candidate.sole_center = next.output_position;
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
