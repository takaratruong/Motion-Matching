#include "support_runtime.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "support runtime test failed: %s\n", message);
        std::exit(1);
    }
}

static void close(float actual, float expected, float tolerance, const char* message)
{
    check(terrain_float_is_finite(actual) &&
          std::fabs(actual - expected) <= tolerance, message);
}

static uint32_t float_bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static support_observation observation(
    float root, float left, float right, bool left_contact, bool right_contact)
{
    support_observation out = {};
    out.source_height[0] = out.source_height[1] = out.source_height[2] = 0.0f;
    out.runtime_height[0] = root;
    out.runtime_height[1] = left;
    out.runtime_height[2] = right;
    out.delta[0] = root;
    out.delta[1] = left;
    out.delta[2] = right;
    out.contact[0] = left_contact;
    out.contact[1] = right_contact;
    return out;
}

static void check_observation_bits_equal(
    const support_observation& actual,
    const support_observation& expected,
    const char* message)
{
    bool equal = true;
    for (int i = 0; i < 3; ++i) {
        equal = equal &&
            float_bits(actual.source_height[i]) ==
                float_bits(expected.source_height[i]) &&
            float_bits(actual.runtime_height[i]) ==
                float_bits(expected.runtime_height[i]) &&
            float_bits(actual.delta[i]) == float_bits(expected.delta[i]);
    }
    equal = equal && actual.contact[0] == expected.contact[0] &&
        actual.contact[1] == expected.contact[1];
    check(equal, message);
}

static void test_observation_build_maps_g1_support_and_checked_v2_sampling()
{
    terrain_support_set support;
    support.values.resize(2, 3);
    support.values(0, 0) = 91.0f;
    support.values(0, 1) = 92.0f;
    support.values(0, 2) = 93.0f;
    support.values(1, 0) = 0.25f;
    support.values(1, 1) = 1.50f;
    support.values(1, 2) = -2.0f;

    heightfield terrain;
    terrain.version = 2;
    terrain.nx = 2;
    terrain.nz = 2;
    terrain.origin_x = 10.0f;
    terrain.origin_z = 20.0f;
    terrain.cell_size = 2.0f;
    terrain.exterior_height = -17.0f;
    terrain.heights.resize(4);
    terrain.heights(0) = 1.0f;
    terrain.heights(1) = 5.0f;
    terrain.heights(2) = 9.0f;
    terrain.heights(3) = 21.0f;

    const vec3 root(10.5f, 2.0f, 20.25f);
    const vec3 left_toe(11.0f, 3.0f, 21.5f);
    const vec3 right_toe(11.5f, 4.0f, 20.5f);
    const vec3 points[3] = {root, left_toe, right_toe};
    support_observation out = {};
    char error[256] = {};
    check(support_observation_build(
              out, support, 1, terrain, root, left_toe, right_toe,
              true, false, error, sizeof(error)),
          error);

    close(out.source_height[0], 0.25f, 0.0f, "G1SP root column");
    close(out.source_height[1], 1.50f, 0.0f, "G1SP left-toe column");
    close(out.source_height[2], -2.0f, 0.0f, "G1SP right-toe column");
    close(out.runtime_height[0], 4.0f, 0.0f,
          "asymmetric v2 root sample");
    close(out.runtime_height[1], 13.0f, 0.0f,
          "asymmetric v2 left-toe sample");
    close(out.runtime_height[2], 8.0f, 0.0f,
          "asymmetric v2 right-toe sample");
    for (int i = 0; i < 3; ++i)
        close(out.runtime_height[i],
              heightfield_sample_v2(terrain, points[i].x, points[i].z),
              0.0f, "observation uses checked v2 sample result");
    close(out.delta[0], 3.75f, 0.0f, "root support delta");
    close(out.delta[1], 11.50f, 0.0f, "left-toe support delta");
    close(out.delta[2], 10.0f, 0.0f, "right-toe support delta");
    check(out.contact[0] && !out.contact[1], "recorded observation contacts");

    terrain.version = 1;
    terrain.exterior_height = -23.0f;
    check(support_observation_build(
              out, support, 1, terrain, root, left_toe, right_toe,
              false, true, error, sizeof(error)),
          error);
    for (int i = 0; i < 3; ++i)
        close(out.runtime_height[i], -23.0f, 0.0f,
              "checked v2 sampling rejects wrong in-memory version");
    check(!out.contact[0] && out.contact[1],
          "checked-sample observation contacts");

    terrain.version = 2;
    terrain.exterior_height = -17.0f;
    check(support_observation_build(
              out, support, 1, terrain, root, left_toe, right_toe,
              true, false, error, sizeof(error)),
          error);
    const support_observation before = out;

    support.values(1, 1) = std::numeric_limits<float>::quiet_NaN();
    error[0] = '\0';
    check(!support_observation_build(
              out, support, 1, terrain, root, left_toe, right_toe,
              true, false, error, sizeof(error)),
          "nonfinite G1SP observation rejection");
    check_observation_bits_equal(
        out, before, "nonfinite G1SP observation transaction");
    check(std::strstr(error, "finite") != NULL,
          "nonfinite G1SP observation diagnostic");
    support.values(1, 1) = 1.50f;

    error[0] = '\0';
    check(!support_observation_build(
              out, support, 1, terrain, root,
              vec3(std::numeric_limits<float>::quiet_NaN(),
                   left_toe.y, left_toe.z),
              right_toe, true, false, error, sizeof(error)),
          "nonfinite FK observation rejection");
    check_observation_bits_equal(
        out, before, "nonfinite FK observation transaction");
    check(std::strstr(error, "finite") != NULL,
          "nonfinite FK observation diagnostic");

    error[0] = '\0';
    check(!support_observation_build(
              out, support, support.values.rows, terrain,
              root, left_toe, right_toe, true, false,
              error, sizeof(error)),
          "out-of-range observation rejection");
    check_observation_bits_equal(
        out, before, "out-of-range observation transaction");
    check(std::strstr(error, "range") != NULL,
          "out-of-range observation diagnostic");
}

static void test_contact_choice_airborne_hold_and_root_fallback()
{
    const float dt = 1.0f / 25.0f;
    char error[256] = {};
    support_frame_state state;
    support_frame_reset(state, 0.30f);

    check(support_frame_update(state,
          observation(0.10f, 0.24f, 0.38f, true, false), false,
          dt, error, sizeof(error)), error);
    check(state.source == support_left, "left support source");
    close(state.nominal_height, 0.24f, 1e-6f, "left support target");

    check(support_frame_update(state,
          observation(0.10f, 0.24f, 0.38f, true, true), false,
          dt, error, sizeof(error)), error);
    check(state.source == support_both, "double support source");
    close(state.nominal_height, 0.31f, 1e-6f, "double support mean");

    const float held = state.height;
    const float held_velocity = state.velocity;
    for (int frame = 0; frame < 2; ++frame) {
        check(support_frame_update(state,
              observation(0.0f, 0.0f, 0.0f, false, false), false,
              dt, error, sizeof(error)), error);
        check(state.source == support_held, "two-frame airborne hold source");
        close(state.nominal_height, 0.31f, 1e-6f, "two-frame airborne hold target");
        check(float_bits(state.height) == float_bits(held),
              "two-frame airborne hold preserves height bits");
        check(float_bits(state.velocity) == float_bits(held_velocity),
              "two-frame airborne hold preserves velocity bits");
    }
    check(state.height > 0.0f && state.height <= held + 1e-6f,
          "hold remains bounded");
    check(support_frame_update(state,
          observation(0.0f, 0.0f, 0.0f, false, false), false,
          dt, error, sizeof(error)), error);
    check(state.source == support_airborne_root, "third-frame root fallback");
    check(state.height > 0.0f && state.height < held,
          "root fallback starts continuously");
    check(float_bits(state.height) != float_bits(held) ||
          float_bits(state.velocity) != float_bits(held_velocity),
          "third frame starts root fallback decay");
}

static void test_rebase_preserves_height_and_vertical_velocity()
{
    support_frame_state state;
    support_frame_reset(state, 0.36f);
    state.height = 0.41f;
    state.velocity = 0.17f;
    support_frame_rebase(state, -0.12f, -0.03f);
    close(state.nominal_height + state.offset_height,
          0.41f, 1e-7f, "rebase position continuity");
    close(state.nominal_velocity + state.offset_velocity,
          0.17f, 1e-7f, "rebase velocity continuity");
}

static void test_update_rebases_source_frame_transition()
{
    const float dt = 1.0f / 25.0f;
    const float target = 0.395f;
    support_frame_state state;
    support_frame_reset(state, 0.40f);
    state.velocity = 0.12f;
    state.nominal_height = 0.39f;
    state.nominal_velocity = -0.04f;
    state.offset_height = state.height - state.nominal_height;
    state.offset_velocity = state.velocity - state.nominal_velocity;
    state.source = support_left;

    float expected_offset_height = state.height - target;
    float expected_offset_velocity = state.velocity;
    decay_spring_damper_exact(
        expected_offset_height, expected_offset_velocity, 0.10f, dt);

    char error[256] = {};
    check(support_frame_update(state,
          observation(0.0f, target, 0.0f, true, false), true,
          dt, error, sizeof(error)), error);
    check(state.source == support_left, "source-frame transition keeps contact source");
    close(state.nominal_height, target, 0.0f,
          "source-frame transition nominal target");
    check(float_bits(state.nominal_velocity) == float_bits(0.0f),
          "source-frame transition zeroes nominal velocity");
    close(state.offset_height, expected_offset_height, 1e-7f,
          "source-frame transition rebases height before decay");
    close(state.offset_velocity, expected_offset_velocity, 1e-7f,
          "source-frame transition rebases velocity before decay");
    close(state.height, target + expected_offset_height, 1e-7f,
          "source-frame transition decays continuously from prior height");
    close(state.velocity, expected_offset_velocity, 1e-7f,
          "source-frame transition decays continuously from prior velocity");
}

static void test_update_rebases_contact_source_transition()
{
    const float dt = 1.0f / 25.0f;
    const float target = 0.31f;
    support_frame_state state;
    support_frame_reset(state, 0.27f);
    state.velocity = -0.08f;
    state.nominal_height = 0.30f;
    state.nominal_velocity = 0.02f;
    state.offset_height = state.height - state.nominal_height;
    state.offset_velocity = state.velocity - state.nominal_velocity;
    state.source = support_left;

    float expected_offset_height = state.height - target;
    float expected_offset_velocity = state.velocity;
    decay_spring_damper_exact(
        expected_offset_height, expected_offset_velocity, 0.10f, dt);

    char error[256] = {};
    check(support_frame_update(state,
          observation(0.0f, 0.30f, 0.32f, true, true), false,
          dt, error, sizeof(error)), error);
    check(state.source == support_both, "contact transition selects both support");
    close(state.nominal_height, target, 1e-7f,
          "contact transition nominal target");
    check(float_bits(state.nominal_velocity) == float_bits(0.0f),
          "contact transition zeroes nominal velocity");
    close(state.offset_height, expected_offset_height, 1e-7f,
          "contact transition rebases height before decay");
    close(state.offset_velocity, expected_offset_velocity, 1e-7f,
          "contact transition rebases velocity before decay");
    close(state.height, target + expected_offset_height, 1e-7f,
          "contact transition decays continuously from prior height");
    close(state.velocity, expected_offset_velocity, 1e-7f,
          "contact transition decays continuously from prior velocity");
}

static void test_update_rebases_only_above_nominal_discontinuity_boundary()
{
    const float dt = 1.0f / 25.0f;
    const float boundary = 0.02f;
    char error[256] = {};

    support_frame_state exact;
    support_frame_reset(exact, 0.0f);
    exact.source = support_left;
    check(support_frame_update(exact,
          observation(0.0f, boundary, 0.0f, true, false), false,
          dt, error, sizeof(error)), error);
    close(exact.nominal_height, boundary, 0.0f,
          "exact discontinuity boundary nominal target");
    close(exact.nominal_velocity, boundary / dt, 1e-7f,
          "exact discontinuity boundary keeps sampled velocity");
    check(float_bits(exact.offset_height) == float_bits(0.0f) &&
          float_bits(exact.offset_velocity) == float_bits(0.0f),
          "exact discontinuity boundary does not rebase offsets");
    close(exact.height, boundary, 0.0f,
          "exact discontinuity boundary does not rebase height");

    const float above = std::nextafter(
        boundary, std::numeric_limits<float>::infinity());
    check(above > boundary, "above-boundary fixture");
    support_frame_state discontinuous;
    support_frame_reset(discontinuous, 0.0f);
    discontinuous.source = support_left;
    float expected_offset_height = -above;
    float expected_offset_velocity = 0.0f;
    decay_spring_damper_exact(
        expected_offset_height, expected_offset_velocity, 0.10f, dt);

    check(support_frame_update(discontinuous,
          observation(0.0f, above, 0.0f, true, false), false,
          dt, error, sizeof(error)), error);
    close(discontinuous.nominal_height, above, 0.0f,
          "above-boundary discontinuity nominal target");
    check(float_bits(discontinuous.nominal_velocity) == float_bits(0.0f),
          "above-boundary discontinuity zeroes nominal velocity");
    close(discontinuous.offset_height, expected_offset_height, 1e-7f,
          "above-boundary discontinuity rebases height before decay");
    close(discontinuous.offset_velocity, expected_offset_velocity, 1e-7f,
          "above-boundary discontinuity rebases velocity before decay");
    close(discontinuous.height, above + expected_offset_height, 1e-7f,
          "above-boundary discontinuity decays from prior height");
    close(discontinuous.velocity, expected_offset_velocity, 1e-7f,
          "above-boundary discontinuity decays from prior velocity");
    check(float_bits(discontinuous.height) != float_bits(above),
          "above-boundary discontinuity avoids a target snap");
}

static void test_same_source_airborne_discontinuities_do_not_create_velocity()
{
    const float dt = 1.0f / 25.0f;
    const float targets[] = {-0.0885f, 0.1600f};
    char error[256] = {};
    support_frame_state state;
    support_frame_reset(state, 0.0f);
    state.source = support_airborne_root;
    state.airborne_frames = 2;

    float previous_height = state.height;
    for (const float target : targets) {
        check(support_frame_update(state,
              observation(target, 0.0f, 0.0f, false, false), false,
              dt, error, sizeof(error)), error);
        check(state.source == support_airborne_root,
              "observed discontinuity remains same-source airborne root");
        check(float_bits(state.nominal_velocity) == float_bits(0.0f),
              "same-source discontinuity zeroes nominal velocity");
        check(terrain_float_is_finite(state.height) &&
              terrain_float_is_finite(state.velocity),
              "same-source discontinuity output remains finite");
        const float lower = std::fmin(previous_height, target) - 1e-6f;
        const float upper = std::fmax(previous_height, target) + 1e-6f;
        check(state.height >= lower && state.height <= upper,
              "same-source discontinuity first frame remains bounded");
        check(std::fabs(state.velocity) < 1.0f,
              "same-source discontinuity avoids artificial velocity spike");
        previous_height = state.height;
    }
}

static void test_elevated_landing_ramp_and_descent_do_not_decay_to_zero()
{
    const float dt = 1.0f / 25.0f;
    char error[256] = {};
    support_frame_state state;
    support_frame_reset(state, 0.36f);
    for (int frame = 0; frame < 75; ++frame) {
        check(support_frame_update(state,
              observation(0.36f, 0.36f, 0.36f, true, true), false,
              dt, error, sizeof(error)), error);
        close(state.height, 0.36f, 1e-6f, "three-second elevated landing");
    }
    float previous = state.height;
    for (int frame = 1; frame <= 36; ++frame) {
        const float height = 0.36f + 0.01f * static_cast<float>(frame);
        check(support_frame_update(state,
              observation(height, height, height, true, true), false,
              dt, error, sizeof(error)), error);
        check(state.height >= previous, "ramp support is monotonic");
        previous = state.height;
    }
    for (int frame = 35; frame >= 0; --frame) {
        const float height = 0.01f * static_cast<float>(frame);
        check(support_frame_update(state,
              observation(height, height, height, true, true), false,
              dt, error, sizeof(error)), error);
    }
    for (int frame = 0; frame < 50; ++frame)
        check(support_frame_update(state,
              observation(0.0f, 0.0f, 0.0f, true, true), false,
              dt, error, sizeof(error)), error);
    close(state.height, 0.0f, 0.02f, "descent returns to base level");
}

static void test_nonfinite_update_is_transactional()
{
    support_frame_state state;
    support_frame_reset(state, 0.25f);
    const support_frame_state before = state;
    support_observation bad = observation(0, 0, 0, true, false);
    bad.delta[1] = std::numeric_limits<float>::quiet_NaN();
    char error[256] = {};
    check(!support_frame_update(
          state, bad, false, 1.0f / 25.0f, error, sizeof(error)),
          "nonfinite support rejection");
    check(state.height == before.height && state.velocity == before.velocity &&
          state.nominal_height == before.nominal_height &&
          state.nominal_velocity == before.nominal_velocity &&
          state.offset_height == before.offset_height &&
          state.offset_velocity == before.offset_velocity &&
          state.airborne_frames == before.airborne_frames &&
          state.source == before.source &&
          state.initialized == before.initialized,
          "nonfinite support transaction");
    check(std::strstr(error, "finite") != NULL, "nonfinite diagnostic");
}

static void test_pose_and_root_helpers_are_horizontal_only()
{
    array1d<vec3> input(3), output(3);
    input(0) = vec3(1.0f, -7.0f, 2.0f);
    input(1) = vec3(0.0f, 0.8f, 0.0f);
    input(2) = vec3(0.1f, -0.2f, 0.3f);
    support_pose_apply(output, input, 0.42f);
    close(output(0).y, 0.42f, 0.0f, "support on Simulation Y");
    check(output(0).x == input(0).x && output(0).z == input(0).z,
          "support preserves root XZ");
    check(output(1).x == input(1).x && output(1).y == input(1).y &&
          output(1).z == input(1).z, "support preserves local child pose");

    const vec3 character(1.0f, -0.0f, 1.0f);
    const vec3 simulation(2.0f, -99.0f, 2.0f);
    vec3 result = horizontal_adjust_character_position(
        character, simulation, 0.1f, 1.0f / 25.0f);
    check(float_bits(result.y) == float_bits(character.y),
          "plain adjustment preserves Y bits");
    result = horizontal_adjust_character_position_by_velocity(
        character, vec3(0.5f, 1000.0f, 0.0f), simulation,
        0.5f, 0.1f, 1.0f / 25.0f);
    check(float_bits(result.y) == float_bits(character.y),
          "velocity adjustment preserves Y bits");
    result = horizontal_clamp_character_position(character, simulation, 0.15f);
    check(float_bits(result.y) == float_bits(character.y),
          "clamp preserves Y bits");
    close(std::sqrt((result.x - simulation.x) * (result.x - simulation.x) +
                    (result.z - simulation.z) * (result.z - simulation.z)),
          0.15f, 1e-6f, "horizontal clamp radius");
}

int main()
{
    test_observation_build_maps_g1_support_and_checked_v2_sampling();
    test_contact_choice_airborne_hold_and_root_fallback();
    test_rebase_preserves_height_and_vertical_velocity();
    test_update_rebases_source_frame_transition();
    test_update_rebases_contact_source_transition();
    test_update_rebases_only_above_nominal_discontinuity_boundary();
    test_same_source_airborne_discontinuities_do_not_create_velocity();
    test_elevated_landing_ramp_and_descent_do_not_decay_to_zero();
    test_nonfinite_update_is_transactional();
    test_pose_and_root_helpers_are_horizontal_only();
    return 0;
}
