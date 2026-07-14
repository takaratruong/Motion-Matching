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
    for (int frame = 0; frame < 2; ++frame) {
        check(support_frame_update(state,
              observation(0.0f, 0.0f, 0.0f, false, false), false,
              dt, error, sizeof(error)), error);
        check(state.source == support_held, "two-frame airborne hold source");
        close(state.nominal_height, 0.31f, 1e-6f, "two-frame airborne hold target");
    }
    check(state.height > 0.0f && state.height <= held + 1e-6f,
          "hold remains bounded");
    check(support_frame_update(state,
          observation(0.0f, 0.0f, 0.0f, false, false), false,
          dt, error, sizeof(error)), error);
    check(state.source == support_airborne_root, "third-frame root fallback");
    check(state.height > 0.0f && state.height < held,
          "root fallback starts continuously");
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
    test_contact_choice_airborne_hold_and_root_fallback();
    test_rebase_preserves_height_and_vertical_velocity();
    test_elevated_landing_ramp_and_descent_do_not_decay_to_zero();
    test_nonfinite_update_is_transactional();
    test_pose_and_root_helpers_are_horizontal_only();
    return 0;
}
