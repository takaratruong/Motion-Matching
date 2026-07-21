#include "sonic/cpp/g1_turn_model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static void check(bool condition, const char* expression, int line)
{
    if (!condition) {
        std::fprintf(
            stderr,
            "G1 turn model check failed at line %d: %s\n",
            line,
            expression);
        std::exit(1);
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

static bool nearly(float value, float expected, float tolerance)
{
    return std::fabs(value - expected) <= tolerance;
}

static quat yaw_quat(float angle)
{
    return quat_from_angle_axis(angle, vec3(0.0f, 1.0f, 0.0f));
}

// Planar yaw about +Y in degrees, in the same (-180, 180] wrap the limiter uses.
static float yaw_degrees(quat q)
{
    return 2.0f * std::atan2(q.y, q.w) * 180.0f / PIf;
}

static float wrapped_yaw_degrees(quat q)
{
    float degrees = yaw_degrees(q);
    while (degrees <= -180.0f) degrees += 360.0f;
    while (degrees > 180.0f) degrees -= 360.0f;
    return degrees;
}

static bool same_quat_bits(quat left, quat right)
{
    return std::memcmp(&left, &right, sizeof(quat)) == 0;
}

static void test_exact_stepping_and_reversal()
{
    char error[256] = {};
    const g1_turn_model_config config = g1_turn_model_fixed_config();
    quat current = yaw_quat(0.0f);
    quat next = quat(9.0f, 8.0f, 7.0f, 6.0f);
    CHECK(g1_turn_model_step(
        next, current, yaw_quat(PIf), 0.04f,
        G1MovementHoldenTurnV1, config, error, sizeof(error)));
    CHECK(nearly(yaw_degrees(next), 4.8f, 1.0e-4f));
    CHECK(nearly(quat_length(next), 1.0f, 1.0e-6f));

    int steps = 1;
    current = next;
    while (yaw_degrees(current) < 179.999f && steps < 100) {
        CHECK(g1_turn_model_step(
            next, current, yaw_quat(PIf), 0.04f,
            G1MovementHoldenTurnV1, config, error, sizeof(error)));
        CHECK(yaw_degrees(next) + 1.0e-4f >= yaw_degrees(current));
        current = next;
        ++steps;
    }
    CHECK(steps >= 38);
    CHECK(nearly(steps * 0.04f, 1.52f, 1.0e-5f));

    const quat small = yaw_quat(2.0f * PIf / 180.0f);
    CHECK(g1_turn_model_step(
        next, yaw_quat(0.0f), small, 0.04f,
        G1MovementHoldenTurnV1, config, error, sizeof(error)));
    CHECK(same_quat_bits(next, small));
}

static void test_wrap_and_tie()
{
    char error[256] = {};
    const g1_turn_model_config config = g1_turn_model_fixed_config();
    quat next = quat(9.0f, 8.0f, 7.0f, 6.0f);

    // +179 to -179 takes the +2-degree shortest path.
    CHECK(g1_turn_model_step(
        next, yaw_quat(179.0f * PIf / 180.0f),
        yaw_quat(-179.0f * PIf / 180.0f), 0.04f,
        G1MovementHoldenTurnV1, config, error, sizeof(error)));
    CHECK(nearly(wrapped_yaw_degrees(next), -179.0f, 1.0e-4f));

    // The exact 180-degree tie is positive.
    CHECK(g1_turn_model_step(
        next, yaw_quat(0.0f), yaw_quat(-PIf), 0.04f,
        G1MovementHoldenTurnV1, config, error, sizeof(error)));
    CHECK(nearly(yaw_degrees(next), 4.8f, 1.0e-4f));
}

static void test_prediction_from_local_copy()
{
    char error[256] = {};
    const g1_turn_model_config config = g1_turn_model_fixed_config();
    quat predictions_storage[4];
    const quat persistent = yaw_quat(0.0f);
    CHECK(g1_turn_model_predict(
        slice1d<quat>(4, predictions_storage), persistent, yaw_quat(PIf),
        1.0f / 3.0f, G1MovementHoldenTurnV1, config,
        error, sizeof(error)));
    CHECK(same_quat_bits(persistent, yaw_quat(0.0f)));
    CHECK(nearly(yaw_degrees(predictions_storage[0]), 40.0f, 1.0e-4f));
    CHECK(nearly(yaw_degrees(predictions_storage[1]), 80.0f, 1.0e-4f));
    CHECK(nearly(yaw_degrees(predictions_storage[2]), 120.0f, 1.0e-4f));
    CHECK(nearly(yaw_degrees(predictions_storage[3]), 160.0f, 1.0e-4f));
}

static void test_historical_profiles_pass_through_exactly()
{
    char error[256] = {};
    const g1_turn_model_config config = g1_turn_model_fixed_config();
    // A valid non-planar unit quaternion (rotation about a tilted axis).
    const quat non_planar = quat_from_angle_axis(
        0.7f, normalize(vec3(0.3f, 0.5f, 0.8f)));
    CHECK(nearly(quat_length(non_planar), 1.0f, 1.0e-6f));

    quat next = quat(9.0f, 8.0f, 7.0f, 6.0f);
    CHECK(g1_turn_model_step(
        next, yaw_quat(0.0f), non_planar, 0.04f,
        G1MovementRaw, config, error, sizeof(error)));
    CHECK(same_quat_bits(next, non_planar));

    next = quat(9.0f, 8.0f, 7.0f, 6.0f);
    CHECK(g1_turn_model_step(
        next, yaw_quat(0.0f), non_planar, 0.04f,
        G1MovementHoldenV1, config, error, sizeof(error)));
    CHECK(same_quat_bits(next, non_planar));
}

static void require_step_rejects(
    quat current,
    quat target,
    float dt,
    g1_movement_model_profile profile,
    const g1_turn_model_config& config,
    const char* message)
{
    quat output(1.25f, -2.5f, 3.75f, -4.0f);
    const quat before = output;
    char error[256] = {};
    const bool ok = g1_turn_model_step(
        output, current, target, dt, profile, config, error, sizeof(error));
    check(!ok, message, __LINE__);
    check(same_quat_bits(output, before),
          "rejected step must not mutate caller output", __LINE__);
}

static void test_turn_profile_invalid_inputs_fail_closed()
{
    const g1_turn_model_config config = g1_turn_model_fixed_config();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const quat planar = yaw_quat(0.0f);
    const quat target = yaw_quat(PIf);

    // Non-planar unit quaternion is rejected under the turn profile.
    const quat non_planar = quat_from_angle_axis(
        0.7f, normalize(vec3(0.3f, 0.5f, 0.8f)));
    require_step_rejects(
        non_planar, target, 0.04f, G1MovementHoldenTurnV1, config,
        "non-planar current heading is rejected");
    require_step_rejects(
        planar, non_planar, 0.04f, G1MovementHoldenTurnV1, config,
        "non-planar target heading is rejected");

    // Non-unit quaternion is rejected.
    require_step_rejects(
        quat(2.0f, 0.0f, 0.5f, 0.0f), target, 0.04f,
        G1MovementHoldenTurnV1, config, "non-unit current heading is rejected");

    // NaN component is rejected.
    require_step_rejects(
        quat(nan, 0.0f, 0.0f, 0.0f), target, 0.04f,
        G1MovementHoldenTurnV1, config, "NaN heading component is rejected");

    // dt validation.
    require_step_rejects(
        planar, target, 0.0f, G1MovementHoldenTurnV1, config,
        "zero dt is rejected");
    require_step_rejects(
        planar, target, -0.04f, G1MovementHoldenTurnV1, config,
        "negative dt is rejected");
    require_step_rejects(
        planar, target, inf, G1MovementHoldenTurnV1, config,
        "non-finite dt is rejected");

    // Invalid enum.
    require_step_rejects(
        planar, target, 0.04f,
        static_cast<g1_movement_model_profile>(42), config,
        "invalid profile enum is rejected");

    // Invalid rate configurations.
    g1_turn_model_config zero_rate = config;
    zero_rate.max_yaw_rate_radians = 0.0f;
    require_step_rejects(
        planar, target, 0.04f, G1MovementHoldenTurnV1, zero_rate,
        "zero yaw rate is rejected");
    g1_turn_model_config negative_rate = config;
    negative_rate.max_yaw_rate_radians = -1.0f;
    require_step_rejects(
        planar, target, 0.04f, G1MovementHoldenTurnV1, negative_rate,
        "negative yaw rate is rejected");
    g1_turn_model_config nan_rate = config;
    nan_rate.max_yaw_rate_radians = nan;
    require_step_rejects(
        planar, target, 0.04f, G1MovementHoldenTurnV1, nan_rate,
        "non-finite yaw rate is rejected");
}

static void test_fixed_config_rate()
{
    const g1_turn_model_config config = g1_turn_model_fixed_config();
    CHECK(nearly(config.max_yaw_rate_radians, 120.0f * PIf / 180.0f, 1.0e-6f));
}

int main()
{
    test_fixed_config_rate();
    test_exact_stepping_and_reversal();
    test_wrap_and_tie();
    test_prediction_from_local_copy();
    test_historical_profiles_pass_through_exactly();
    test_turn_profile_invalid_inputs_fail_closed();
    std::printf("G1 turn model tests passed\n");
    return 0;
}
