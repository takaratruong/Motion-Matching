#include "sonic/cpp/g1_movement_model.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static void check(bool condition, const char* expression, int line)
{
    if (!condition) {
        std::fprintf(
            stderr,
            "G1 movement model check failed at line %d: %s\n",
            line,
            expression);
        std::exit(1);
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

static bool nearly(float value, float expected)
{
    return std::fabs(value - expected) <= 1.0e-4f;
}

static bool finite(vec3 v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

static bool same_bits(vec3 left, vec3 right)
{
    return std::memcmp(&left, &right, sizeof(vec3)) == 0;
}

static void test_public_behavior()
{
    vec3 velocity;
    char error[256] = {};
    const g1_movement_model_config config = g1_movement_model_fixed_config();
    CHECK(g1_movement_model_step(
        velocity, vec3(), vec3(0.9f, 0.0f, 0.0f), 0.04f,
        G1MovementHoldenV1, config, error, sizeof(error)));
    CHECK(nearly(velocity.x, 0.06f));

    velocity = vec3(0.9f, 0.0f, 0.0f);
    vec3 first_brake;
    CHECK(g1_movement_model_step(
        first_brake, velocity, vec3(-0.9f, 0.0f, 0.0f), 0.04f,
        G1MovementHoldenV1, config, error, sizeof(error)));
    CHECK(nearly(first_brake.x, 0.82f));
    velocity = first_brake;
    bool crossed_zero = false;
    for (int i = 0; i < 32; ++i) {
        vec3 next;
        CHECK(g1_movement_model_step(
            next, velocity, vec3(-0.9f, 0.0f, 0.0f), 0.04f,
            G1MovementHoldenV1, config, error, sizeof(error)));
        CHECK(next.x <= velocity.x + 1.0e-6f);
        CHECK(length(next - velocity) <= 0.060001f);
        crossed_zero = crossed_zero || next.x <= 0.0f;
        velocity = next;
    }
    CHECK(crossed_zero);
    CHECK(nearly(velocity.x, -0.9f));

    vec3 lateral;
    CHECK(g1_movement_model_step(
        lateral, vec3(0.6f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.6f), 0.04f,
        G1MovementHoldenV1, config, error, sizeof(error)));
    CHECK(finite(lateral));
    CHECK(length(lateral - vec3(0.6f, 0.0f, 0.0f)) <= 0.080001f);

    array1d<vec3> predicted;
    predicted.resize(4);
    const vec3 persistent(0.3f, 0.0f, 0.0f);
    CHECK(g1_movement_model_predict(
        predicted, persistent, vec3(-0.9f, 0.0f, 0.0f), 1.0f / 3.0f,
        G1MovementHoldenV1, config, error, sizeof(error)));
    CHECK(same_bits(persistent, vec3(0.3f, 0.0f, 0.0f)));
    CHECK(predicted(0).x < 0.3f);
    CHECK(predicted(3).x <= predicted(2).x);

    vec3 raw;
    CHECK(g1_movement_model_step(
        raw, vec3(0.4f, 0.0f, 0.0f), vec3(-0.9f, 0.0f, 0.0f), 0.04f,
        G1MovementRaw, config, error, sizeof(error)));
    CHECK(same_bits(raw, vec3(-0.9f, 0.0f, 0.0f)));
}

static void require_step_rejects(
    vec3 current,
    vec3 target,
    float dt,
    g1_movement_model_profile profile,
    const g1_movement_model_config& config,
    const char* message)
{
    vec3 output(1.25f, -2.5f, 3.75f);
    const vec3 before = output;
    char error[256] = {};
    const bool ok = g1_movement_model_step(
        output, current, target, dt, profile, config,
        error, sizeof(error));
    check(!ok, message, __LINE__);
    check(same_bits(output, before),
          "rejected step must not mutate caller output", __LINE__);
}

static void test_invalid_inputs_fail_closed()
{
    const g1_movement_model_config config = g1_movement_model_fixed_config();
    const vec3 good_current(0.1f, 0.0f, 0.0f);
    const vec3 good_target(0.5f, 0.0f, 0.0f);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    require_step_rejects(
        good_current, good_target, 0.0f,
        G1MovementHoldenV1, config, "zero dt is rejected");
    require_step_rejects(
        good_current, good_target, -0.04f,
        G1MovementHoldenV1, config, "negative dt is rejected");
    require_step_rejects(
        good_current, good_target, nan,
        G1MovementHoldenV1, config, "non-finite dt is rejected");
    require_step_rejects(
        vec3(nan, 0.0f, 0.0f), good_target, 0.04f,
        G1MovementHoldenV1, config, "non-finite current is rejected");
    require_step_rejects(
        good_current, vec3(0.0f, inf, 0.0f), 0.04f,
        G1MovementHoldenV1, config, "non-finite target is rejected");
    require_step_rejects(
        good_current, good_target, 0.04f,
        static_cast<g1_movement_model_profile>(42), config,
        "invalid profile enum is rejected");

    g1_movement_model_config zero_acceleration = config;
    zero_acceleration.acceleration = 0.0f;
    require_step_rejects(
        good_current, good_target, 0.04f,
        G1MovementHoldenV1, zero_acceleration,
        "zero acceleration is rejected");

    g1_movement_model_config negative_deceleration = config;
    negative_deceleration.deceleration = -2.0f;
    require_step_rejects(
        good_current, good_target, 0.04f,
        G1MovementHoldenV1, negative_deceleration,
        "negative deceleration is rejected");

    g1_movement_model_config nan_acceleration = config;
    nan_acceleration.acceleration = nan;
    require_step_rejects(
        good_current, good_target, 0.04f,
        G1MovementHoldenV1, nan_acceleration,
        "non-finite acceleration is rejected");

    g1_movement_model_config inf_deceleration = config;
    inf_deceleration.deceleration = inf;
    require_step_rejects(
        good_current, good_target, 0.04f,
        G1MovementHoldenV1, inf_deceleration,
        "non-finite deceleration is rejected");
}

static void test_predict_fails_closed_without_mutation()
{
    const g1_movement_model_config config = g1_movement_model_fixed_config();
    array1d<vec3> predicted;
    predicted.resize(4);
    const vec3 seed(9.0f, 8.0f, 7.0f);
    predicted.set(seed);
    char error[256] = {};
    const bool ok = g1_movement_model_predict(
        predicted, vec3(0.3f, 0.0f, 0.0f), vec3(-0.9f, 0.0f, 0.0f),
        -1.0f / 3.0f, G1MovementHoldenV1, config, error, sizeof(error));
    check(!ok, "predict rejects non-positive sample_dt", __LINE__);
    for (int i = 0; i < predicted.size; ++i) {
        check(same_bits(predicted(i), seed),
              "rejected predict must not mutate caller slice", __LINE__);
    }
}

static void require_step_parity(
    vec3 current,
    vec3 target,
    float dt,
    const g1_movement_model_config& config)
{
    char error[256] = {};
    vec3 velocity_only;
    vec3 turn_profile;
    CHECK(g1_movement_model_step(
        velocity_only, current, target, dt,
        G1MovementHoldenV1, config, error, sizeof(error)));
    CHECK(g1_movement_model_step(
        turn_profile, current, target, dt,
        G1MovementHoldenTurnV1, config, error, sizeof(error)));
    CHECK(same_bits(velocity_only, turn_profile));
}

// The turn profile must shape translational velocity bit-identically to
// holden-v1; only the heading is capped elsewhere.
static void test_turn_profile_velocity_matches_holden_v1()
{
    const g1_movement_model_config config = g1_movement_model_fixed_config();

    require_step_parity(vec3(), vec3(0.9f, 0.0f, 0.0f), 0.04f, config);
    require_step_parity(
        vec3(0.9f, 0.0f, 0.0f), vec3(-0.9f, 0.0f, 0.0f), 0.04f, config);
    require_step_parity(
        vec3(0.6f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.6f), 0.04f, config);
    require_step_parity(
        vec3(0.4f, 0.0f, 0.0f), vec3(-0.9f, 0.0f, 0.0f), 0.04f, config);

    array1d<vec3> velocity_only;
    array1d<vec3> turn_profile;
    velocity_only.resize(4);
    turn_profile.resize(4);
    char error[256] = {};
    const vec3 persistent(0.3f, 0.0f, 0.0f);
    CHECK(g1_movement_model_predict(
        velocity_only, persistent, vec3(-0.9f, 0.0f, 0.0f), 1.0f / 3.0f,
        G1MovementHoldenV1, config, error, sizeof(error)));
    CHECK(g1_movement_model_predict(
        turn_profile, persistent, vec3(-0.9f, 0.0f, 0.0f), 1.0f / 3.0f,
        G1MovementHoldenTurnV1, config, error, sizeof(error)));
    for (int i = 0; i < 4; ++i) {
        CHECK(same_bits(velocity_only(i), turn_profile(i)));
    }
}

int main()
{
    test_public_behavior();
    test_invalid_inputs_fail_closed();
    test_predict_fails_closed_without_mutation();
    test_turn_profile_velocity_matches_holden_v1();
    std::printf("G1 movement model tests passed\n");
    return 0;
}
