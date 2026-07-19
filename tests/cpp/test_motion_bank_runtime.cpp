#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>
#include <new>

static bool allocation_counting = false;
static size_t allocation_count = 0;

void* operator new(size_t size)
{
    if (allocation_counting) ++allocation_count;
    void* pointer = malloc(size == 0 ? 1 : size);
    if (pointer == NULL) throw std::bad_alloc();
    return pointer;
}

void* operator new[](size_t size)
{
    if (allocation_counting) ++allocation_count;
    void* pointer = malloc(size == 0 ? 1 : size);
    if (pointer == NULL) throw std::bad_alloc();
    return pointer;
}

void operator delete(void* pointer) noexcept { free(pointer); }
void operator delete[](void* pointer) noexcept { free(pointer); }
void operator delete(void* pointer, size_t) noexcept { free(pointer); }
void operator delete[](void* pointer, size_t) noexcept { free(pointer); }

#include "motion_bank_runtime.h"

static void check(bool condition, const char* message)
{
    if (!condition) {
        fprintf(stderr, "motion bank runtime test failed: %s\n", message);
        abort();
    }
}

static uint32_t float_bits(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static double double_from_bits(uint64_t bits)
{
    double value = 0.0;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void check_close(double actual, double expected, double tolerance,
                        const char* message)
{
    check(terrain_double_is_finite(actual), message);
    check(fabs(actual - expected) <= tolerance, message);
}

enum oracle_surface
{
    oracle_flat,
    oracle_longitudinal_slope,
    oracle_cross_slope,
    oracle_curb_up,
    oracle_curb_down,
    oracle_stair_up,
    oracle_stair_down,
    oracle_curved_plane,
};

static double ten_degree_grade()
{
    return tan(10.0 * 3.14159265358979323846264338327950288 / 180.0);
}

static void initialize_oracle_field(heightfield& field, oracle_surface surface)
{
    field.version = 2;
    field.nx = 101;
    field.nz = surface == oracle_curved_plane ? 101 : 51;
    field.origin_x = -0.5f;
    field.origin_z = -0.5f;
    field.cell_size = 0.02f;
    field.exterior_height = 37.0f;
    field.heights.resize(field.nx * field.nz);

    const double grade = ten_degree_grade();
    for (int z = 0; z < field.nz; ++z) {
        const float node_z = field.origin_z +
            static_cast<float>(z) * field.cell_size;
        for (int x = 0; x < field.nx; ++x) {
            const float node_x = field.origin_x +
                static_cast<float>(x) * field.cell_size;
            double height = 1.0;
            if (surface == oracle_longitudinal_slope) {
                height += grade * static_cast<double>(node_x);
            } else if (surface == oracle_cross_slope) {
                height += grade * static_cast<double>(node_z);
            } else if (surface == oracle_curb_up ||
                       surface == oracle_curb_down) {
                const double sign = surface == oracle_curb_up ? 1.0 : -1.0;
                if (node_x >= 0.5f) height += sign * 0.12;
            } else if (surface == oracle_stair_up ||
                       surface == oracle_stair_down) {
                const double sign = surface == oracle_stair_up ? 1.0 : -1.0;
                const float edges[3] = {0.20f, 0.52f, 0.84f};
                for (int edge = 0; edge < 3; ++edge) {
                    if (node_x >= edges[edge]) height += sign * 0.12;
                }
            } else if (surface == oracle_curved_plane) {
                height = 10.0 + static_cast<double>(node_x) +
                    10.0 * static_cast<double>(node_z);
            }
            field.heights(z * field.nx + x) = static_cast<float>(height);
        }
    }
}

static void check_descriptor_bits(
    oracle_surface surface, const uint32_t expected[12], const char* message)
{
    heightfield field;
    initialize_oracle_field(field, surface);
    vec3 centerline[2] = {
        vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f)};
    const vec3 centerline_before[2] = {centerline[0], centerline[1]};
    vec3 heading(1.0f, 0.0f, 0.0f);
    const vec3 heading_before = heading;
    terrain_descriptor_v2 descriptor = {};
    terrain_profile_status status = terrain_profile_invalid_sample;

    check(terrain_descriptor_sample_v2(
              descriptor, status, field,
              slice1d<vec3>(2, centerline), heading),
          message);
    check(status == terrain_profile_ok, message);
    for (int i = 0; i < 12; ++i) {
        check(float_bits(descriptor.values[i]) == expected[i], message);
    }
    check(memcmp(centerline, centerline_before, sizeof(centerline)) == 0,
          "descriptor does not mutate predicted travel");
    check(memcmp(&heading, &heading_before, sizeof(heading)) == 0,
          "descriptor does not mutate independent heading");
    check(float_bits(descriptor.root_point.y) == float_bits(1.0f),
          "descriptor records root provenance height");
    for (int i = 0; i < 8; ++i) {
        check(float_bits(descriptor.center_points[i].y) !=
                  float_bits(field.exterior_height),
              "descriptor records in-domain center provenance");
    }
}

static void test_locked_python_descriptor_oracles()
{
    static const uint32_t flat[12] = {};
    static const uint32_t long_slope[12] = {
        UINT32_C(0x3cb48f40), UINT32_C(0x3d348f20),
        UINT32_C(0x3d876b50), UINT32_C(0x3db48f10),
        UINT32_C(0x3de1b2e0), UINT32_C(0x3e076b50),
        UINT32_C(0x3e1dfd30), UINT32_C(0x3e348f10),
        0, 0, 0, 0,
    };
    static const uint32_t cross_slope[12] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        UINT32_C(0x3d568350), UINT32_C(0x3d568350),
        UINT32_C(0x3d568350), UINT32_C(0x3d568350),
    };
    static const uint32_t curb_up[12] = {
        0, 0, 0, UINT32_C(0x3df5c290), UINT32_C(0x3df5c290),
        UINT32_C(0x3df5c290), UINT32_C(0x3df5c290),
        UINT32_C(0x3df5c290), 0, 0, 0, 0,
    };
    static const uint32_t curb_down[12] = {
        0, 0, 0, UINT32_C(0xbdf5c290), UINT32_C(0xbdf5c290),
        UINT32_C(0xbdf5c290), UINT32_C(0xbdf5c290),
        UINT32_C(0xbdf5c290), 0, 0, 0, 0,
    };
    static const uint32_t stair_up[12] = {
        0, UINT32_C(0x3df5c290), UINT32_C(0x3df5c290),
        UINT32_C(0x3df5c2a0), UINT32_C(0x3e75c290),
        UINT32_C(0x3e75c290), UINT32_C(0x3eb851ec),
        UINT32_C(0x3eb851ec), 0, 0, 0, 0,
    };
    static const uint32_t stair_down[12] = {
        0, UINT32_C(0xbdf5c290), UINT32_C(0xbdf5c290),
        UINT32_C(0xbdf5c2a0), UINT32_C(0xbe75c290),
        UINT32_C(0xbe75c290), UINT32_C(0xbeb851ec),
        UINT32_C(0xbeb851ec), 0, 0, 0, 0,
    };
    check_descriptor_bits(oracle_flat, flat, "flat descriptor Python oracle");
    check_descriptor_bits(oracle_longitudinal_slope, long_slope,
                          "longitudinal slope descriptor Python oracle");
    check_descriptor_bits(oracle_cross_slope, cross_slope,
                          "cross-slope descriptor Python oracle");
    check_descriptor_bits(oracle_curb_up, curb_up,
                          "curb-up descriptor Python oracle");
    check_descriptor_bits(oracle_curb_down, curb_down,
                          "curb-down descriptor Python oracle");
    check_descriptor_bits(oracle_stair_up, stair_up,
                          "stair-up descriptor Python oracle");
    check_descriptor_bits(oracle_stair_down, stair_down,
                          "stair-down descriptor Python oracle");
}

static void test_curved_travel_and_independent_heading()
{
    heightfield field;
    initialize_oracle_field(field, oracle_curved_plane);
    vec3 centerline[3] = {
        vec3(0.0f, 4.0f, 0.0f),
        vec3(0.5f, 5.0f, 0.0f),
        vec3(0.5f, 6.0f, 0.5f),
    };
    vec3 before[3];
    memcpy(before, centerline, sizeof(before));
    vec3 heading(0.0f, 7.0f, 1.0f);
    const vec3 heading_before = heading;
    terrain_descriptor_v2 descriptor = {};
    terrain_profile_status status = terrain_profile_invalid_sample;
    check(terrain_descriptor_sample_v2(
              descriptor, status, field,
              slice1d<vec3>(3, centerline), heading),
          "curved descriptor succeeds");
    const double expected_longitudinal[8] = {
        0.125, 0.25, 0.375, 0.5, 1.75, 3.0, 4.25, 5.5};
    for (int i = 0; i < 8; ++i) {
        check_close(descriptor.values[i], expected_longitudinal[i], 2e-6,
                    "descriptor follows curved arc distance");
    }
    for (int i = 8; i < 12; ++i) {
        check_close(descriptor.values[i], -2.0 * 0.148506455, 2e-6,
                    "corridor retains independent heading");
    }
    check(memcmp(centerline, before, sizeof(before)) == 0,
          "curved descriptor preserves travel bytes");
    check(memcmp(&heading, &heading_before, sizeof(heading)) == 0,
          "curved descriptor preserves heading bytes");
}

static void fill_sentinel(terrain_descriptor_v2& descriptor)
{
    memset(&descriptor, 0xa5, sizeof(descriptor));
}

static void check_descriptor_failure_preserves_output(
    const heightfield& field, slice1d<vec3> centerline, vec3 heading,
    terrain_profile_status expected_status, const char* message)
{
    terrain_descriptor_v2 descriptor;
    fill_sentinel(descriptor);
    unsigned char before[sizeof(descriptor)];
    memcpy(before, &descriptor, sizeof(before));
    terrain_profile_status status = terrain_profile_ok;
    check(!terrain_descriptor_sample_v2(
              descriptor, status, field, centerline, heading), message);
    check(status == expected_status, message);
    check(memcmp(before, &descriptor, sizeof(before)) == 0, message);
}

static void test_descriptor_explicit_failures_are_transactional()
{
    heightfield field;
    initialize_oracle_field(field, oracle_flat);
    vec3 valid[2] = {
        vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f)};
    vec3 short_line[2] = {
        vec3(0.0f, 0.0f, 0.0f), vec3(0.9f, 0.0f, 0.0f)};
    check_descriptor_failure_preserves_output(
        field, slice1d<vec3>(2, short_line), vec3(1.0f, 0.0f, 0.0f),
        terrain_profile_invalid_centerline, "short centerline is explicit");

    uint32_t nan_bits = UINT32_C(0x7fc00001);
    memcpy(&valid[1].x, &nan_bits, sizeof(nan_bits));
    check_descriptor_failure_preserves_output(
        field, slice1d<vec3>(2, valid), vec3(1.0f, 0.0f, 0.0f),
        terrain_profile_invalid_centerline, "NaN centerline is explicit");
    valid[1].x = 1.0f;

    uint32_t subnormal_bits = UINT32_C(0x00000001);
    memcpy(&valid[0].x, &subnormal_bits, sizeof(subnormal_bits));
    check_descriptor_failure_preserves_output(
        field, slice1d<vec3>(2, valid), vec3(1.0f, 0.0f, 0.0f),
        terrain_profile_invalid_centerline,
        "subnormal centerline is explicit under fast math");
    valid[0].x = 0.0f;
    check_descriptor_failure_preserves_output(
        field, slice1d<vec3>(2, valid), vec3(0.0f, 0.0f, 0.0f),
        terrain_profile_invalid_heading, "degenerate heading is explicit");
    vec3 subnormal_heading(0.0f, 0.0f, 0.0f);
    memcpy(&subnormal_heading.x, &subnormal_bits, sizeof(subnormal_bits));
    check_descriptor_failure_preserves_output(
        field, slice1d<vec3>(2, valid), subnormal_heading,
        terrain_profile_invalid_heading,
        "subnormal heading is explicit under fast math");

    vec3 outside[2] = {
        vec3(0.6f, 0.0f, 0.0f), vec3(1.6f, 0.0f, 0.0f)};
    check_descriptor_failure_preserves_output(
        field, slice1d<vec3>(2, outside), vec3(1.0f, 0.0f, 0.0f),
        terrain_profile_out_of_domain,
        "exterior height never proves descriptor domain");

    const float saved = field.heights(25 * field.nx + 25);
    field.heights(25 * field.nx + 25) = -0.0f;
    check_descriptor_failure_preserves_output(
        field, slice1d<vec3>(2, valid), vec3(1.0f, 0.0f, 0.0f),
        terrain_profile_invalid_sample,
        "invalid selected cell is not classified flat");
    field.heights(25 * field.nx + 25) = saved;

    const uint32_t saved_version = field.version;
    field.version = 1;
    check_descriptor_failure_preserves_output(
        field, slice1d<vec3>(2, valid), vec3(1.0f, 0.0f, 0.0f),
        terrain_profile_invalid_field, "non-v2 field is explicit");
    field.version = saved_version;

    check(strcmp(terrain_profile_status_name(terrain_profile_out_of_domain),
                 "out_of_domain") == 0,
          "terrain failure status has stable literal");
}

struct profile_fixture
{
    double distances[51];
    double heights[51];
    terrain_profile_normal_v2 normals[51];
};

static void initialize_profile(profile_fixture& profile, oracle_surface surface)
{
    const double grade = ten_degree_grade();
    for (int i = 0; i < 51; ++i) {
        const double distance = static_cast<double>(i) / 50.0;
        profile.distances[i] = distance;
        profile.heights[i] = 0.0;
        double normal_x = 0.0;
        double normal_y = 1.0;
        double normal_z = 0.0;
        if (surface == oracle_longitudinal_slope) {
            profile.heights[i] = grade * distance;
            normal_x = -grade;
        } else if (surface == oracle_cross_slope) {
            normal_z = -grade;
        } else if (surface == oracle_curb_up ||
                   surface == oracle_curb_down) {
            const double sign = surface == oracle_curb_up ? 1.0 : -1.0;
            if (distance >= 0.5) profile.heights[i] = sign * 0.12;
        } else if (surface == oracle_stair_up ||
                   surface == oracle_stair_down) {
            const double sign = surface == oracle_stair_up ? 1.0 : -1.0;
            const double edges[3] = {0.20, 0.52, 0.84};
            for (int edge = 0; edge < 3; ++edge) {
                if (distance >= edges[edge]) profile.heights[i] += sign * 0.12;
            }
        }
        const double length = sqrt(
            normal_x * normal_x + normal_y * normal_y +
            normal_z * normal_z);
        profile.normals[i].x = normal_x / length;
        profile.normals[i].y = normal_y / length;
        profile.normals[i].z = normal_z / length;
    }
}

struct classification_oracle
{
    oracle_surface surface;
    motion_bank_family family;
    int elevation_mode;
    uint64_t confidence_bits;
    uint64_t step_bits;
    uint64_t grade_bits;
};

static void test_locked_python_classifier_oracles()
{
    static const classification_oracle oracles[] = {
        {oracle_flat, MOTION_BANK_FAMILY_FLAT, 0,
         UINT64_C(0x3ff0000000000000), 0, 0},
        {oracle_longitudinal_slope, MOTION_BANK_FAMILY_SLOPE, 1,
         UINT64_C(0x3fefffffffffffce), 0,
         UINT64_C(0x4024000000000000)},
        {oracle_cross_slope, MOTION_BANK_FAMILY_SLOPE, 0,
         UINT64_C(0x3ff0000000000000), 0, 0},
        {oracle_curb_up, MOTION_BANK_FAMILY_CURB, 1,
         UINT64_C(0x3ff0000000000000),
         UINT64_C(0x3fbeb851eb851ec0),
         UINT64_C(0x40240415edb47036)},
        {oracle_curb_down, MOTION_BANK_FAMILY_CURB, -1,
         UINT64_C(0x3ff0000000000000),
         UINT64_C(0xbfbeb851eb851eb8),
         UINT64_C(0xc0240415edb47031)},
        {oracle_stair_up, MOTION_BANK_FAMILY_STAIR, 1,
         UINT64_C(0x3feffffffffffff8),
         UINT64_C(0x3fbeb851eb851eb0),
         UINT64_C(0x40355376eeb1670f)},
        {oracle_stair_down, MOTION_BANK_FAMILY_STAIR, -1,
         UINT64_C(0x3feffffffffffffe),
         UINT64_C(0xbfbeb851eb851eb8),
         UINT64_C(0xc0355376eeb16712)},
    };
    for (size_t i = 0; i < sizeof(oracles) / sizeof(oracles[0]); ++i) {
        profile_fixture profile;
        initialize_profile(profile, oracles[i].surface);
        motion_bank_classification classification = {};
        motion_bank_classification_status status =
            motion_bank_classification_invalid_value;
        check(motion_bank_classify_profile(
                  classification, status, profile.distances,
                  profile.heights, profile.normals, 51),
              "locked classifier oracle succeeds");
        check(status == motion_bank_classification_ok,
              "classifier returns explicit success");
        check(classification.family == oracles[i].family,
              "classifier family matches Python production");
        check(classification.elevation_mode == oracles[i].elevation_mode,
              "classifier elevation matches Python production");
        check_close(classification.confidence,
                    double_from_bits(oracles[i].confidence_bits), 2e-12,
                    "classifier confidence matches Python production");
        check_close(classification.step_height_m,
                    double_from_bits(oracles[i].step_bits), 2e-12,
                    "classifier step matches Python production");
        check_close(classification.grade_degrees,
                    double_from_bits(oracles[i].grade_bits), 2e-12,
                    "classifier grade matches Python production");
    }
    check(strcmp(motion_bank_family_name(MOTION_BANK_FAMILY_STAIR),
                 "stair") == 0,
          "bank family has stable literal");
}

static void test_classifier_rejects_bad_profiles_transactionally()
{
    profile_fixture profile;
    initialize_profile(profile, oracle_flat);
    motion_bank_classification output;
    memset(&output, 0x5a, sizeof(output));
    unsigned char before[sizeof(output)];
    memcpy(before, &output, sizeof(before));
    motion_bank_classification_status status = motion_bank_classification_ok;

    profile.distances[2] = profile.distances[1];
    check(!motion_bank_classify_profile(
              output, status, profile.distances, profile.heights,
              profile.normals, 51),
          "duplicate distance fails");
    check(status == motion_bank_classification_invalid_domain,
          "duplicate distance has explicit status");
    check(memcmp(before, &output, sizeof(before)) == 0,
          "classifier failure preserves output");
    initialize_profile(profile, oracle_flat);

    profile.distances[50] = 0.9;
    check(!motion_bank_classify_profile(
              output, status, profile.distances, profile.heights,
              profile.normals, 51),
          "short profile domain fails");
    check(status == motion_bank_classification_invalid_domain,
          "short profile has explicit status");
    initialize_profile(profile, oracle_flat);

    uint64_t nan_bits = UINT64_C(0x7ff8000000000001);
    memcpy(&profile.heights[9], &nan_bits, sizeof(nan_bits));
    check(!motion_bank_classify_profile(
              output, status, profile.distances, profile.heights,
              profile.normals, 51),
          "nonfinite profile fails");
    check(status == motion_bank_classification_invalid_value,
          "nonfinite profile has explicit status");
    initialize_profile(profile, oracle_flat);

    profile.normals[4].x = 0.0;
    profile.normals[4].y = 0.0;
    profile.normals[4].z = 0.0;
    check(!motion_bank_classify_profile(
              output, status, profile.distances, profile.heights,
              profile.normals, 51),
          "degenerate normal fails");
    check(status == motion_bank_classification_invalid_normal,
          "degenerate normal has explicit status");
    check(memcmp(before, &output, sizeof(before)) == 0,
          "all classifier failures preserve output");

    check(!motion_bank_classify_profile(
              output, status, profile.distances, profile.heights,
              profile.normals, 8),
          "sparse profile fails");
    check(status == motion_bank_classification_invalid_count,
          "sparse profile has explicit status");
    check(strcmp(motion_bank_classification_status_name(status),
                 "invalid_count") == 0,
          "classifier status has stable literal");
}

static motion_bank_classification observation(
    motion_bank_family family, double confidence, int elevation_mode = 2)
{
    motion_bank_classification value = {};
    value.family = family;
    value.elevation_mode = elevation_mode == 2
        ? (family == MOTION_BANK_FAMILY_FLAT ? 0 : 1)
        : elevation_mode;
    value.confidence = confidence;
    value.step_height_m = family == MOTION_BANK_FAMILY_CURB ||
                          family == MOTION_BANK_FAMILY_STAIR
        ? 0.12 : 0.0;
    value.grade_degrees = family == MOTION_BANK_FAMILY_SLOPE ? 10.0 : 0.0;
    return value;
}

static void test_fixed_two_frame_hysteresis()
{
    motion_bank_state state;
    motion_bank_state_reset(state);
    check(state.current_family == MOTION_BANK_FAMILY_NONE &&
              state.pending_family == MOTION_BANK_FAMILY_NONE &&
              state.current_elevation_mode == 0 &&
              state.pending_elevation_mode == 0 &&
              state.pending_count == 0 && !state.transitioned &&
              state.reason == motion_bank_transition_uninitialized,
          "bank state reset is explicit");

    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_STAIR, 1.0));
    check(state.current_family == MOTION_BANK_FAMILY_STAIR &&
              state.current_elevation_mode == 1 &&
              state.transitioned &&
              state.reason == motion_bank_transition_initial_confident,
          "initial confident family enters immediately");
    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_STAIR, 0.1));
    check(state.current_family == MOTION_BANK_FAMILY_STAIR &&
              !state.transitioned && state.pending_count == 0 &&
              state.reason == motion_bank_transition_retained_same,
          "same family remains stable without confidence gate");

    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_CURB, 0.99));
    check(state.current_family == MOTION_BANK_FAMILY_STAIR &&
              state.pending_family == MOTION_BANK_FAMILY_CURB &&
              state.pending_count == 1 && !state.transitioned,
          "one isolated curb cannot release stair");
    motion_bank_state_observe(
        state, false, observation(MOTION_BANK_FAMILY_FLAT, 1.0));
    check(state.current_family == MOTION_BANK_FAMILY_STAIR &&
              state.pending_family == MOTION_BANK_FAMILY_NONE &&
              state.pending_count == 0 &&
              state.reason == motion_bank_transition_pending_cleared_invalid,
          "invalid observation clears pending without inventing bank");

    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_SLOPE, 0.60));
    check(state.pending_family == MOTION_BANK_FAMILY_SLOPE &&
              state.pending_count == 1,
          "fixed confidence boundary begins pending family");
    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_CURB, 0.9));
    check(state.current_family == MOTION_BANK_FAMILY_STAIR &&
              state.pending_family == MOTION_BANK_FAMILY_CURB &&
              state.pending_count == 1,
          "different observation restarts consecutive count");
    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_CURB, 0.59));
    check(state.current_family == MOTION_BANK_FAMILY_STAIR &&
              state.pending_family == MOTION_BANK_FAMILY_NONE &&
              state.pending_count == 0 &&
              state.reason ==
                  motion_bank_transition_pending_cleared_low_confidence,
          "low confidence clears pending without releasing stair");

    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_FLAT, 1.0));
    check(state.current_family == MOTION_BANK_FAMILY_STAIR &&
              state.pending_count == 1,
          "first confident flat is pending");
    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_FLAT, 1.0));
    check(state.current_family == MOTION_BANK_FAMILY_FLAT &&
              state.current_elevation_mode == 0 &&
              state.pending_family == MOTION_BANK_FAMILY_NONE &&
              state.pending_count == 0 && state.transitioned &&
              state.reason == motion_bank_transition_confirmed,
          "two confident flat frames release stair at 25 Hz");
    check(strcmp(motion_bank_transition_reason_name(state.reason),
                 "confirmed") == 0,
          "transition reason has stable literal");

    motion_bank_state_reset(state);
    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_FLAT, 0.59));
    check(state.current_family == MOTION_BANK_FAMILY_NONE &&
              state.reason ==
                  motion_bank_transition_pending_cleared_low_confidence,
          "low-confidence startup does not invent flat");
}

static void test_family_and_elevation_hysterize_atomically()
{
    motion_bank_state state;
    motion_bank_state_reset(state);
    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_FLAT, 1.0, 0));
    check(state.current_family == MOTION_BANK_FAMILY_FLAT &&
              state.current_elevation_mode == 0,
          "confident flat initializes the accepted family/elevation pair");

    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_SLOPE, 0.0, 1));
    check(state.current_family == MOTION_BANK_FAMILY_FLAT &&
              state.current_elevation_mode == 0 &&
              state.pending_family == MOTION_BANK_FAMILY_NONE &&
              state.pending_elevation_mode == 0 &&
              !state.transitioned,
          "low-confidence slope-up retains accepted flat/level atomically");

    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_SLOPE, 1.0, 1));
    check(state.current_family == MOTION_BANK_FAMILY_FLAT &&
              state.current_elevation_mode == 0 &&
              state.pending_family == MOTION_BANK_FAMILY_SLOPE &&
              state.pending_elevation_mode == 1 &&
              state.pending_count == 1 && !state.transitioned,
          "first confident slope-up observation only starts the pair pending");
    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_SLOPE, 1.0, 1));
    check(state.current_family == MOTION_BANK_FAMILY_SLOPE &&
              state.current_elevation_mode == 1 &&
              state.pending_family == MOTION_BANK_FAMILY_NONE &&
              state.pending_elevation_mode == 0 &&
              state.pending_count == 0 && state.transitioned,
          "second confident observation atomically accepts slope-up");

    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_SLOPE, 1.0, -1));
    check(state.current_family == MOTION_BANK_FAMILY_SLOPE &&
              state.current_elevation_mode == 1 &&
              state.pending_family == MOTION_BANK_FAMILY_SLOPE &&
              state.pending_elevation_mode == -1 &&
              state.pending_count == 1 && !state.transitioned,
          "same-family elevation reversal is hysterized as a pair change");
    motion_bank_state_observe(
        state, true, observation(MOTION_BANK_FAMILY_SLOPE, 1.0, -1));
    check(state.current_family == MOTION_BANK_FAMILY_SLOPE &&
              state.current_elevation_mode == -1 && state.transitioned,
          "same-family elevation reversal commits only after confirmation");
}

static void test_dense_sampling_and_hot_calls_allocate_nothing()
{
    heightfield field;
    initialize_oracle_field(field, oracle_longitudinal_slope);
    vec3 centerline[2] = {
        vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f)};
    terrain_dense_profile_v2 profile = {};
    terrain_profile_status terrain_status = terrain_profile_invalid_sample;
    check(terrain_dense_profile_sample_v2(
              profile, terrain_status, field,
              slice1d<vec3>(2, centerline)),
          "dense runtime profile samples one-metre path");
    check(profile.distances_m[0] == 0.0 &&
              profile.distances_m[50] == 1.0,
          "dense profile locks full domain");
    for (int i = 1; i < 51; ++i) {
        check(profile.distances_m[i] > profile.distances_m[i - 1] &&
                  profile.distances_m[i] - profile.distances_m[i - 1] <=
                      0.05,
              "dense profile spacing satisfies classifier");
    }
    motion_bank_classification classification = {};
    motion_bank_classification_status classification_status =
        motion_bank_classification_invalid_value;
    check(motion_bank_classify_profile(
              classification, classification_status, profile.distances_m,
              profile.heights_m, profile.normals, 51),
          "runtime dense profile feeds classifier");
    check(classification.family == MOTION_BANK_FAMILY_SLOPE &&
              classification.elevation_mode == 1,
          "runtime dense slope selects slope-up bank");

    terrain_descriptor_v2 descriptor = {};
    motion_bank_state state;
    motion_bank_state_reset(state);
    allocation_count = 0;
    allocation_counting = true;
    const bool descriptor_ok = terrain_descriptor_sample_v2(
        descriptor, terrain_status, field,
        slice1d<vec3>(2, centerline), vec3(1.0f, 0.0f, 0.0f));
    const bool profile_ok = terrain_dense_profile_sample_v2(
        profile, terrain_status, field, slice1d<vec3>(2, centerline));
    const bool classifier_ok = motion_bank_classify_profile(
        classification, classification_status, profile.distances_m,
        profile.heights_m, profile.normals, 51);
    motion_bank_state_observe(state, true, classification);
    allocation_counting = false;
    check(descriptor_ok && profile_ok && classifier_ok,
          "hot calls succeed during allocation audit");
    check(allocation_count == 0,
          "descriptor classifier and hysteresis allocate no heap memory");
}

int main()
{
    test_locked_python_descriptor_oracles();
    test_curved_travel_and_independent_heading();
    test_descriptor_explicit_failures_are_transactional();
    test_locked_python_classifier_oracles();
    test_classifier_rejects_bad_profiles_transactionally();
    test_fixed_two_frame_hysteresis();
    test_family_and_elevation_hysterize_atomically();
    test_dense_sampling_and_hot_calls_allocate_nothing();
    printf("motion bank runtime tests passed\n");
    return 0;
}
