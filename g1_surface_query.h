#pragma once

#include "terrain_runtime.h"

#include <cmath>

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
