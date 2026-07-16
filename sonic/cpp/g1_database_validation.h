#pragma once

#include "database.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>

static inline bool g1_database_validation_error(
    char* output,
    int capacity,
    const char* format,
    ...)
{
    if (output != nullptr && capacity > 0) {
        va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(
            output,
            static_cast<std::size_t>(capacity),
            format,
            arguments);
        va_end(arguments);
    }
    return false;
}

template<typename T>
static inline bool g1_database_animation_shape_is_valid(
    const array2d<T>& values,
    int frames,
    int bones,
    const char* label,
    char* error,
    int capacity)
{
    if (values.rows != frames || values.cols != bones ||
        values.data == nullptr) {
        return g1_database_validation_error(
            error,
            capacity,
            "G1 database %s shape mismatch: expected %dx%d, got %dx%d",
            label,
            frames,
            bones,
            values.rows,
            values.cols);
    }
    return true;
}

static inline bool g1_database_validate(
    const database& db,
    char* error,
    int capacity)
{
    const int frames = db.nframes();
    const int bones = db.nbones();
    if (frames <= 0) {
        return g1_database_validation_error(
            error, capacity, "G1 database has no frames");
    }
    if (!g1_database_animation_shape_is_valid(
            db.bone_positions,
            frames,
            bones,
            "bone_positions",
            error,
            capacity) ||
        !g1_database_animation_shape_is_valid(
            db.bone_velocities,
            frames,
            bones,
            "bone_velocities",
            error,
            capacity) ||
        !g1_database_animation_shape_is_valid(
            db.bone_rotations,
            frames,
            bones,
            "bone_rotations",
            error,
            capacity) ||
        !g1_database_animation_shape_is_valid(
            db.bone_angular_velocities,
            frames,
            bones,
            "bone_angular_velocities",
            error,
            capacity)) {
        return false;
    }
    if (db.contact_states.rows != frames || db.contact_states.cols != 2 ||
        db.contact_states.data == nullptr) {
        return g1_database_validation_error(
            error,
            capacity,
            "G1 database contact shape mismatch: expected %dx2, got %dx%d",
            frames,
            db.contact_states.rows,
            db.contact_states.cols);
    }
    if (db.range_starts.size <= 0 ||
        db.range_stops.size != db.range_starts.size ||
        db.range_starts.data == nullptr || db.range_stops.data == nullptr) {
        return g1_database_validation_error(
            error,
            capacity,
            "G1 database range arrays must be nonempty and equal-sized");
    }
    int expected_start = 0;
    for (int range = 0; range < db.nranges(); ++range) {
        const int start = db.range_starts(range);
        const int stop = db.range_stops(range);
        if (start != expected_start || stop <= start || stop > frames) {
            return g1_database_validation_error(
                error,
                capacity,
                "G1 database range %d is not contiguous/in-bounds: "
                "expected start %d, got [%d,%d) for %d frames",
                range,
                expected_start,
                start,
                stop,
                frames);
        }
        expected_start = stop;
    }
    if (expected_start != frames) {
        return g1_database_validation_error(
            error,
            capacity,
            "G1 database ranges stop at %d instead of covering %d frames",
            expected_start,
            frames);
    }
    return true;
}

static inline bool g1_matching_feature_value_is_safe(float value)
{
    const std::uint32_t magnitude =
        feature_float_bits(value) & UINT32_C(0x7fffffff);
    return feature_float_is_finite(value) &&
           magnitude != UINT32_C(0x7f7fffff);
}

static inline bool g1_matching_features_validate(
    const database& db,
    char* error,
    int capacity)
{
    static constexpr int expected_features = 31;
    if (db.features.rows != db.nframes() ||
        db.features.cols != expected_features || db.features.data == nullptr ||
        db.features_offset.size != expected_features ||
        db.features_scale.size != expected_features ||
        db.features_offset.data == nullptr ||
        db.features_scale.data == nullptr) {
        return g1_database_validation_error(
            error,
            capacity,
            "G1 matching feature build failed: expected %dx%d features, "
            "got %dx%d",
            db.nframes(),
            expected_features,
            db.features.rows,
            db.features.cols);
    }
    for (int feature = 0; feature < expected_features; ++feature) {
        if (!g1_matching_feature_value_is_safe(
                db.features_offset(feature)) ||
            !feature_float_is_positive_finite(db.features_scale(feature))) {
            return g1_database_validation_error(
                error,
                capacity,
                "G1 matching feature %d has invalid offset/scale",
                feature);
        }
    }
    for (int index = 0; index < db.features.rows * db.features.cols;
         ++index) {
        if (!g1_matching_feature_value_is_safe(db.features.data[index])) {
            return g1_database_validation_error(
                error,
                capacity,
                "G1 matching feature row payload is invalid at value %d",
                index);
        }
    }

    const int small_rows =
        (db.nframes() + BOUND_SM_SIZE - 1) / BOUND_SM_SIZE;
    const int large_rows =
        (db.nframes() + BOUND_LR_SIZE - 1) / BOUND_LR_SIZE;
    const array2d<float>* bounds[4] = {
        &db.bound_sm_min,
        &db.bound_sm_max,
        &db.bound_lr_min,
        &db.bound_lr_max,
    };
    const int expected_rows[4] = {
        small_rows, small_rows, large_rows, large_rows};
    for (int bound = 0; bound < 4; ++bound) {
        if (bounds[bound]->rows != expected_rows[bound] ||
            bounds[bound]->cols != expected_features ||
            bounds[bound]->data == nullptr) {
            return g1_database_validation_error(
                error,
                capacity,
                "G1 matching bound %d shape mismatch: expected %dx%d, "
                "got %dx%d",
                bound,
                expected_rows[bound],
                expected_features,
                bounds[bound]->rows,
                bounds[bound]->cols);
        }
        for (int index = 0;
             index < bounds[bound]->rows * bounds[bound]->cols;
             ++index) {
            if (!g1_matching_feature_value_is_safe(
                    bounds[bound]->data[index])) {
                return g1_database_validation_error(
                    error,
                    capacity,
                    "G1 matching bound %d has invalid value at %d",
                    bound,
                    index);
            }
        }
    }
    for (int index = 0; index < small_rows * expected_features; ++index) {
        if (db.bound_sm_min.data[index] > db.bound_sm_max.data[index]) {
            return g1_database_validation_error(
                error, capacity, "G1 small matching bounds are inverted");
        }
    }
    for (int index = 0; index < large_rows * expected_features; ++index) {
        if (db.bound_lr_min.data[index] > db.bound_lr_max.data[index]) {
            return g1_database_validation_error(
                error, capacity, "G1 large matching bounds are inverted");
        }
    }
    return true;
}
