#pragma once

#include "terrain_runtime.h"

#include <math.h>
#include <stdint.h>

enum motion_bank_family
{
    MOTION_BANK_FAMILY_NONE = 0,
    MOTION_BANK_FAMILY_FLAT,
    MOTION_BANK_FAMILY_CURB,
    MOTION_BANK_FAMILY_SLOPE,
    MOTION_BANK_FAMILY_STAIR,
};

static inline const char* motion_bank_family_name(motion_bank_family family)
{
    switch (family) {
    case MOTION_BANK_FAMILY_NONE: return "none";
    case MOTION_BANK_FAMILY_FLAT: return "flat";
    case MOTION_BANK_FAMILY_CURB: return "curb";
    case MOTION_BANK_FAMILY_SLOPE: return "slope";
    case MOTION_BANK_FAMILY_STAIR: return "stair";
    }
    return "unknown";
}

struct motion_bank_classification
{
    motion_bank_family family = MOTION_BANK_FAMILY_NONE;
    int elevation_mode = 0;
    double confidence = 0.0;
    double step_height_m = 0.0;
    double grade_degrees = 0.0;
};

enum motion_bank_classification_status
{
    motion_bank_classification_ok = 0,
    motion_bank_classification_invalid_input,
    motion_bank_classification_invalid_count,
    motion_bank_classification_invalid_value,
    motion_bank_classification_invalid_domain,
    motion_bank_classification_invalid_normal,
};

static inline const char* motion_bank_classification_status_name(
    motion_bank_classification_status status)
{
    switch (status) {
    case motion_bank_classification_ok: return "ok";
    case motion_bank_classification_invalid_input: return "invalid_input";
    case motion_bank_classification_invalid_count: return "invalid_count";
    case motion_bank_classification_invalid_value: return "invalid_value";
    case motion_bank_classification_invalid_domain: return "invalid_domain";
    case motion_bank_classification_invalid_normal: return "invalid_normal";
    }
    return "unknown";
}

// These are the named Task-4 production thresholds. They describe the fixed
// one-metre terrain profile and are not tuned from runtime route outcomes.
static const int MOTION_BANK_PROFILE_MIN_SAMPLES = 9;
static const int MOTION_BANK_PROFILE_MAX_SAMPLES = 64;
static const double MOTION_BANK_PROFILE_DOMAIN_START_M = 0.0;
static const double MOTION_BANK_PROFILE_DOMAIN_STOP_M = 1.0;
static const double MOTION_BANK_PROFILE_DOMAIN_TOLERANCE_M = 1e-9;
static const double MOTION_BANK_PROFILE_MAX_SAMPLE_SPACING_M = 0.05;
static const double MOTION_BANK_PROFILE_MIN_NORMAL_LENGTH = 1e-12;
static const double MOTION_BANK_PROFILE_MIN_UP_NORMAL = 1e-6;
static const double MOTION_BANK_FLAT_MAX_ABS_GRADE_DEGREES = 2.0;
static const double MOTION_BANK_FLAT_MAX_LEVEL_RANGE_M = 0.04;
static const double MOTION_BANK_DISCONTINUITY_MIN_STEP_M = 0.06;
static const int MOTION_BANK_STAIR_MIN_STEP_COUNT = 2;
static const double MOTION_BANK_STAIR_MIN_TREAD_M = 0.15;
static const double MOTION_BANK_STAIR_MAX_TREAD_M = 0.45;
static const double MOTION_BANK_ELEVATION_MIN_CHANGE_M = 0.03;
static const double MOTION_BANK_CONFIDENCE_REFERENCE_STEP_M = 0.12;
static const double MOTION_BANK_CONFIDENCE_REFERENCE_SLOPE_DEGREES = 10.0;
static const double MOTION_BANK_CONFIDENCE_REFERENCE_STAIR_STEPS = 3.0;
static const double MOTION_BANK_CONFIDENCE_MAX_LINEAR_RESIDUAL_M = 0.02;

static inline double motion_bank_min(double left, double right)
{
    return left < right ? left : right;
}

static inline double motion_bank_max(double left, double right)
{
    return left > right ? left : right;
}

static inline double motion_bank_clamp_unit(double value)
{
    if (value <= 0.0) return 0.0;
    if (value >= 1.0) return 1.0;
    return value;
}

static inline double motion_bank_canonicalize_double_zero(double value)
{
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7fffffffffffffff)) == 0 ? 0.0 : value;
}

static inline double motion_bank_median(
    const double* values, int count)
{
    double sorted[MOTION_BANK_PROFILE_MAX_SAMPLES];
    for (int i = 0; i < count; ++i) {
        double value = values[i];
        int destination = i;
        while (destination > 0 && sorted[destination - 1] > value) {
            sorted[destination] = sorted[destination - 1];
            --destination;
        }
        sorted[destination] = value;
    }
    if ((count & 1) != 0) return sorted[count / 2];
    const volatile double sum =
        sorted[count / 2 - 1] + sorted[count / 2];
    return sum * 0.5;
}

static inline int motion_bank_signed_elevation(double value)
{
    if (fabs(value) < MOTION_BANK_ELEVATION_MIN_CHANGE_M) return 0;
    return value > 0.0 ? 1 : -1;
}

static inline bool motion_bank_classify_profile(
    motion_bank_classification& output,
    motion_bank_classification_status& status,
    const double* distances_m,
    const double* heights_m,
    const terrain_profile_normal_v2* surface_normals,
    int count)
{
    if (distances_m == NULL || heights_m == NULL ||
        surface_normals == NULL) {
        status = motion_bank_classification_invalid_input;
        return false;
    }
    if (count < MOTION_BANK_PROFILE_MIN_SAMPLES ||
        count > MOTION_BANK_PROFILE_MAX_SAMPLES) {
        status = motion_bank_classification_invalid_count;
        return false;
    }

    double relative_heights[MOTION_BANK_PROFILE_MAX_SAMPLES];
    double surface_grades[MOTION_BANK_PROFILE_MAX_SAMPLES];
    double sum_distances = 0.0;
    double sum_heights = 0.0;
    for (int i = 0; i < count; ++i) {
        const terrain_profile_normal_v2 normal = surface_normals[i];
        if (!terrain_double_is_finite(distances_m[i]) ||
            !terrain_double_is_finite(heights_m[i]) ||
            !terrain_double_is_finite(normal.x) ||
            !terrain_double_is_finite(normal.y) ||
            !terrain_double_is_finite(normal.z)) {
            status = motion_bank_classification_invalid_value;
            return false;
        }
        if (i > 0) {
            const volatile double interval =
                distances_m[i] - distances_m[i - 1];
            if (!terrain_double_is_finite(interval) || interval <= 0.0 ||
                interval >
                    MOTION_BANK_PROFILE_MAX_SAMPLE_SPACING_M +
                    MOTION_BANK_PROFILE_DOMAIN_TOLERANCE_M) {
                status = motion_bank_classification_invalid_domain;
                return false;
            }
        }

        const volatile double square_x = normal.x * normal.x;
        const volatile double square_y = normal.y * normal.y;
        const volatile double square_z = normal.z * normal.z;
        const volatile double first_square_sum = square_x + square_y;
        const volatile double square_sum = first_square_sum + square_z;
        if (!terrain_double_is_finite(square_sum) || square_sum < 0.0) {
            status = motion_bank_classification_invalid_normal;
            return false;
        }
        const volatile double length = sqrt(square_sum);
        if (!terrain_double_is_finite(length) ||
            length < MOTION_BANK_PROFILE_MIN_NORMAL_LENGTH) {
            status = motion_bank_classification_invalid_normal;
            return false;
        }
        const volatile double normalized_x = normal.x / length;
        const volatile double normalized_y = normal.y / length;
        const volatile double normalized_z = normal.z / length;
        if (!terrain_double_is_finite(normalized_x) ||
            !terrain_double_is_finite(normalized_y) ||
            !terrain_double_is_finite(normalized_z) ||
            normalized_y < MOTION_BANK_PROFILE_MIN_UP_NORMAL) {
            status = motion_bank_classification_invalid_normal;
            return false;
        }
        const volatile double horizontal_square_x =
            normalized_x * normalized_x;
        const volatile double horizontal_square_z =
            normalized_z * normalized_z;
        const volatile double horizontal_square_sum =
            horizontal_square_x + horizontal_square_z;
        const volatile double horizontal_length =
            sqrt(horizontal_square_sum);
        const volatile double surface_grade =
            horizontal_length / normalized_y;
        if (!terrain_double_is_finite(surface_grade)) {
            status = motion_bank_classification_invalid_normal;
            return false;
        }
        surface_grades[i] = surface_grade;

        const volatile double relative =
            heights_m[i] - heights_m[0];
        if (!terrain_double_is_finite(relative)) {
            status = motion_bank_classification_invalid_value;
            return false;
        }
        relative_heights[i] = relative;
        const volatile double next_distance_sum =
            sum_distances + distances_m[i];
        const volatile double next_height_sum = sum_heights + relative;
        if (!terrain_double_is_finite(next_distance_sum) ||
            !terrain_double_is_finite(next_height_sum)) {
            status = motion_bank_classification_invalid_value;
            return false;
        }
        sum_distances = next_distance_sum;
        sum_heights = next_height_sum;
    }
    if (fabs(distances_m[0] - MOTION_BANK_PROFILE_DOMAIN_START_M) >
            MOTION_BANK_PROFILE_DOMAIN_TOLERANCE_M ||
        fabs(distances_m[count - 1] - MOTION_BANK_PROFILE_DOMAIN_STOP_M) >
            MOTION_BANK_PROFILE_DOMAIN_TOLERANCE_M) {
        status = motion_bank_classification_invalid_domain;
        return false;
    }

    const volatile double mean_distance =
        sum_distances / static_cast<double>(count);
    const volatile double mean_height =
        sum_heights / static_cast<double>(count);
    double numerator = 0.0;
    double denominator = 0.0;
    for (int i = 0; i < count; ++i) {
        const volatile double centered_distance =
            distances_m[i] - mean_distance;
        const volatile double centered_height =
            relative_heights[i] - mean_height;
        const volatile double numerator_term =
            centered_distance * centered_height;
        const volatile double denominator_term =
            centered_distance * centered_distance;
        const volatile double next_numerator = numerator + numerator_term;
        const volatile double next_denominator =
            denominator + denominator_term;
        numerator = next_numerator;
        denominator = next_denominator;
    }
    if (!terrain_double_is_finite(numerator) ||
        !terrain_double_is_finite(denominator) || denominator <= 0.0) {
        status = motion_bank_classification_invalid_domain;
        return false;
    }
    const volatile double grade = numerator / denominator;
    const volatile double grade_degrees =
        atan(grade) * (180.0 / 3.14159265358979323846264338327950288);
    const double median_surface_grade =
        motion_bank_median(surface_grades, count);
    const volatile double surface_grade_degrees =
        atan(median_surface_grade) *
        (180.0 / 3.14159265358979323846264338327950288);
    if (!terrain_double_is_finite(grade) ||
        !terrain_double_is_finite(grade_degrees) ||
        !terrain_double_is_finite(surface_grade_degrees)) {
        status = motion_bank_classification_invalid_value;
        return false;
    }

    double step_positions[MOTION_BANK_PROFILE_MAX_SAMPLES];
    double steps[MOTION_BANK_PROFILE_MAX_SAMPLES];
    int step_count = 0;
    int candidate = 0;
    while (candidate < count - 1) {
        const volatile double delta =
            relative_heights[candidate + 1] -
            relative_heights[candidate];
        if (fabs(delta) < MOTION_BANK_DISCONTINUITY_MIN_STEP_M) {
            ++candidate;
            continue;
        }
        const int start = candidate;
        int stop = candidate;
        double step = 0.0;
        while (stop < count - 1) {
            const volatile double grouped_delta =
                relative_heights[stop + 1] - relative_heights[stop];
            if (fabs(grouped_delta) <
                MOTION_BANK_DISCONTINUITY_MIN_STEP_M) {
                break;
            }
            const volatile double next_step = step + grouped_delta;
            step = next_step;
            ++stop;
        }
        steps[step_count] = step;
        const volatile double position_sum =
            distances_m[start] + distances_m[stop];
        step_positions[step_count] = 0.5 * position_sum;
        ++step_count;
        candidate = stop;
    }

    motion_bank_classification result = {};
    if (step_count > 0) {
        double step_height = motion_bank_median(steps, step_count);
        const bool positive = step_height > 0.0;
        bool same_direction = true;
        for (int i = 0; i < step_count; ++i) {
            if ((steps[i] > 0.0) != positive) same_direction = false;
        }
        double tread_widths[MOTION_BANK_PROFILE_MAX_SAMPLES];
        bool repeated_treads =
            step_count >= MOTION_BANK_STAIR_MIN_STEP_COUNT &&
            same_direction;
        for (int i = 0; i < step_count - 1; ++i) {
            const volatile double width =
                step_positions[i + 1] - step_positions[i];
            tread_widths[i] = width;
            if (width < MOTION_BANK_STAIR_MIN_TREAD_M ||
                width > MOTION_BANK_STAIR_MAX_TREAD_M) {
                repeated_treads = false;
            }
        }
        if (repeated_treads && step_count > 1) {
            double minimum_width = tread_widths[0];
            double maximum_width = tread_widths[0];
            for (int i = 1; i < step_count - 1; ++i) {
                minimum_width = motion_bank_min(
                    minimum_width, tread_widths[i]);
                maximum_width = motion_bank_max(
                    maximum_width, tread_widths[i]);
            }
            const double tread_span =
                MOTION_BANK_STAIR_MAX_TREAD_M -
                MOTION_BANK_STAIR_MIN_TREAD_M;
            const double regularity = 1.0 - motion_bank_clamp_unit(
                (maximum_width - minimum_width) / tread_span);
            double confidence = motion_bank_min(
                1.0,
                static_cast<double>(step_count) /
                    MOTION_BANK_CONFIDENCE_REFERENCE_STAIR_STEPS);
            confidence = motion_bank_min(
                confidence,
                fabs(step_height) /
                    MOTION_BANK_CONFIDENCE_REFERENCE_STEP_M);
            confidence = motion_bank_min(confidence, regularity);
            result.family = MOTION_BANK_FAMILY_STAIR;
            result.elevation_mode = motion_bank_signed_elevation(step_height);
            result.confidence = confidence;
            result.step_height_m = step_height;
            result.grade_degrees = grade_degrees;
        } else {
            int largest = 0;
            for (int i = 1; i < step_count; ++i) {
                if (fabs(steps[i]) > fabs(steps[largest])) largest = i;
            }
            step_height = steps[largest];
            result.family = MOTION_BANK_FAMILY_CURB;
            result.elevation_mode = motion_bank_signed_elevation(step_height);
            result.confidence = motion_bank_min(
                1.0,
                fabs(step_height) /
                    MOTION_BANK_CONFIDENCE_REFERENCE_STEP_M);
            result.step_height_m = step_height;
            result.grade_degrees = grade_degrees;
        }
    } else {
        double minimum_height = relative_heights[0];
        double maximum_height = relative_heights[0];
        for (int i = 1; i < count; ++i) {
            minimum_height = motion_bank_min(
                minimum_height, relative_heights[i]);
            maximum_height = motion_bank_max(
                maximum_height, relative_heights[i]);
        }
        const double level_range = maximum_height - minimum_height;
        if (level_range <= MOTION_BANK_FLAT_MAX_LEVEL_RANGE_M &&
            fabs(grade_degrees) <=
                MOTION_BANK_FLAT_MAX_ABS_GRADE_DEGREES &&
            surface_grade_degrees <=
                MOTION_BANK_FLAT_MAX_ABS_GRADE_DEGREES) {
            const double range_margin = 1.0 - motion_bank_clamp_unit(
                level_range / MOTION_BANK_FLAT_MAX_LEVEL_RANGE_M);
            const double grade_margin = 1.0 - motion_bank_clamp_unit(
                motion_bank_max(
                    fabs(grade_degrees), surface_grade_degrees) /
                MOTION_BANK_FLAT_MAX_ABS_GRADE_DEGREES);
            result.family = MOTION_BANK_FAMILY_FLAT;
            result.elevation_mode = 0;
            result.confidence = motion_bank_min(range_margin, grade_margin);
            result.step_height_m = 0.0;
            result.grade_degrees = grade_degrees;
        } else {
            double residual = 0.0;
            for (int i = 0; i < count; ++i) {
                const volatile double fitted =
                    grade * distances_m[i] + relative_heights[0];
                const volatile double difference =
                    relative_heights[i] - fitted;
                residual = motion_bank_max(residual, fabs(difference));
            }
            const double residual_confidence =
                1.0 - motion_bank_clamp_unit(
                    residual /
                    MOTION_BANK_CONFIDENCE_MAX_LINEAR_RESIDUAL_M);
            const double angle_confidence = motion_bank_clamp_unit(
                motion_bank_max(
                    fabs(grade_degrees), surface_grade_degrees) /
                MOTION_BANK_CONFIDENCE_REFERENCE_SLOPE_DEGREES);
            const double normal_difference = motion_bank_max(
                0.0, fabs(grade_degrees) - surface_grade_degrees);
            const double normal_agreement =
                1.0 - motion_bank_clamp_unit(
                    normal_difference /
                    MOTION_BANK_CONFIDENCE_REFERENCE_SLOPE_DEGREES);
            result.family = MOTION_BANK_FAMILY_SLOPE;
            result.elevation_mode = motion_bank_signed_elevation(
                grade * (MOTION_BANK_PROFILE_DOMAIN_STOP_M -
                         MOTION_BANK_PROFILE_DOMAIN_START_M));
            result.confidence = motion_bank_min(
                angle_confidence,
                motion_bank_min(residual_confidence, normal_agreement));
            result.step_height_m = 0.0;
            result.grade_degrees = grade_degrees;
        }
    }

    if (!terrain_double_is_finite(result.confidence) ||
        !terrain_double_is_finite(result.step_height_m) ||
        !terrain_double_is_finite(result.grade_degrees)) {
        status = motion_bank_classification_invalid_value;
        return false;
    }
    result.confidence = motion_bank_canonicalize_double_zero(
        result.confidence);
    result.step_height_m = motion_bank_canonicalize_double_zero(
        result.step_height_m);
    result.grade_degrees = motion_bank_canonicalize_double_zero(
        result.grade_degrees);
    output = result;
    status = motion_bank_classification_ok;
    return true;
}

enum motion_bank_transition_reason
{
    motion_bank_transition_uninitialized = 0,
    motion_bank_transition_initial_confident,
    motion_bank_transition_retained_same,
    motion_bank_transition_pending_started,
    motion_bank_transition_pending_advanced,
    motion_bank_transition_confirmed,
    motion_bank_transition_pending_cleared_invalid,
    motion_bank_transition_pending_cleared_low_confidence,
};

static inline const char* motion_bank_transition_reason_name(
    motion_bank_transition_reason reason)
{
    switch (reason) {
    case motion_bank_transition_uninitialized: return "uninitialized";
    case motion_bank_transition_initial_confident: return "initial_confident";
    case motion_bank_transition_retained_same: return "retained_same";
    case motion_bank_transition_pending_started: return "pending_started";
    case motion_bank_transition_pending_advanced: return "pending_advanced";
    case motion_bank_transition_confirmed: return "confirmed";
    case motion_bank_transition_pending_cleared_invalid:
        return "pending_cleared_invalid";
    case motion_bank_transition_pending_cleared_low_confidence:
        return "pending_cleared_low_confidence";
    }
    return "unknown";
}

static const double MOTION_BANK_TRANSITION_MIN_CONFIDENCE = 0.60;
// At the fixed 25 Hz controller rate, two consecutive frames are 80 ms.
static const int MOTION_BANK_TRANSITION_REQUIRED_FRAMES = 2;

struct motion_bank_state
{
    motion_bank_family current_family = MOTION_BANK_FAMILY_NONE;
    motion_bank_family pending_family = MOTION_BANK_FAMILY_NONE;
    int pending_count = 0;
    bool transitioned = false;
    motion_bank_transition_reason reason =
        motion_bank_transition_uninitialized;
};

static inline void motion_bank_state_reset(motion_bank_state& state)
{
    state.current_family = MOTION_BANK_FAMILY_NONE;
    state.pending_family = MOTION_BANK_FAMILY_NONE;
    state.pending_count = 0;
    state.transitioned = false;
    state.reason = motion_bank_transition_uninitialized;
}

static inline bool motion_bank_observation_is_valid(
    const motion_bank_classification& observation)
{
    return observation.family >= MOTION_BANK_FAMILY_FLAT &&
           observation.family <= MOTION_BANK_FAMILY_STAIR &&
           observation.elevation_mode >= -1 &&
           observation.elevation_mode <= 1 &&
           terrain_double_is_finite(observation.confidence) &&
           observation.confidence >= 0.0 && observation.confidence <= 1.0 &&
           terrain_double_is_finite(observation.step_height_m) &&
           terrain_double_is_finite(observation.grade_degrees);
}

static inline void motion_bank_state_clear_pending(motion_bank_state& state)
{
    state.pending_family = MOTION_BANK_FAMILY_NONE;
    state.pending_count = 0;
}

static inline void motion_bank_state_observe(
    motion_bank_state& state,
    bool observation_valid,
    const motion_bank_classification& observation)
{
    state.transitioned = false;
    if (!observation_valid ||
        !motion_bank_observation_is_valid(observation)) {
        motion_bank_state_clear_pending(state);
        state.reason = motion_bank_transition_pending_cleared_invalid;
        return;
    }
    if (state.current_family == MOTION_BANK_FAMILY_NONE) {
        motion_bank_state_clear_pending(state);
        if (observation.confidence >=
            MOTION_BANK_TRANSITION_MIN_CONFIDENCE) {
            state.current_family = observation.family;
            state.transitioned = true;
            state.reason = motion_bank_transition_initial_confident;
        } else {
            state.reason =
                motion_bank_transition_pending_cleared_low_confidence;
        }
        return;
    }
    if (observation.family == state.current_family) {
        motion_bank_state_clear_pending(state);
        state.reason = motion_bank_transition_retained_same;
        return;
    }
    if (observation.confidence < MOTION_BANK_TRANSITION_MIN_CONFIDENCE) {
        motion_bank_state_clear_pending(state);
        state.reason =
            motion_bank_transition_pending_cleared_low_confidence;
        return;
    }
    if (state.pending_family != observation.family) {
        state.pending_family = observation.family;
        state.pending_count = 1;
        state.reason = motion_bank_transition_pending_started;
        return;
    }
    ++state.pending_count;
    if (state.pending_count < MOTION_BANK_TRANSITION_REQUIRED_FRAMES) {
        state.reason = motion_bank_transition_pending_advanced;
        return;
    }
    state.current_family = state.pending_family;
    motion_bank_state_clear_pending(state);
    state.transitioned = true;
    state.reason = motion_bank_transition_confirmed;
}
