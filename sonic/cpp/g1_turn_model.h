#pragma once

// Opt-in fixed-rate desired-heading (yaw) limiter.
//
// A pure, header-only unit that bounds the angular speed of the desired
// heading presented to motion matching and runtime application. `raw` and
// `holden-v1` preserve the existing unit-quaternion contract and return the
// requested heading exactly. `holden-turn-v1` additionally requires a
// planar-yaw heading, computes the signed shortest yaw delta in `(-pi, pi]`,
// maps an exact 180-degree tie to positive pi, and advances the current
// heading by at most `max_yaw_rate_radians * dt`. All functions validate their
// inputs and the configuration and never mutate caller outputs on failure.

#include "vec.h"
#include "quat.h"
#include "array.h"
#include "common.h"
#include "g1_movement_model.h"

#include <cmath>

// A `holden-turn-v1` heading is planar when its off-yaw components are within
// this tolerance of zero. Shared with the repository's quaternion tolerance.
static const float G1_TURN_PLANAR_TOLERANCE = 1.0e-5f;

struct g1_turn_model_config
{
    float max_yaw_rate_radians;
};

static inline g1_turn_model_config g1_turn_model_fixed_config()
{
    g1_turn_model_config config;
    config.max_yaw_rate_radians = 120.0f * PIf / 180.0f;
    return config;
}

static inline bool g1_turn_model_fail(
    char* error,
    int capacity,
    const char* message)
{
    return g1_movement_model_fail(error, capacity, message);
}

static inline bool g1_turn_model_quat_is_finite(quat q)
{
    return std::isfinite(q.w) && std::isfinite(q.x)
        && std::isfinite(q.y) && std::isfinite(q.z);
}

// Wrap `angle` into the half-open interval `(-pi, pi]`. The exact +/-pi tie
// therefore maps to positive pi, which advances toward positive yaw.
static inline float g1_turn_wrap_signed(float angle)
{
    float wrapped = std::fmod(angle + PIf, 2.0f * PIf);
    if (wrapped <= 0.0f) wrapped += 2.0f * PIf;
    return wrapped - PIf; // (-pi, pi]
}

static inline bool g1_turn_model_validate_config(
    const g1_turn_model_config& config,
    char* error,
    int capacity)
{
    if (!std::isfinite(config.max_yaw_rate_radians)
            || config.max_yaw_rate_radians <= 0.0f) {
        return g1_turn_model_fail(
            error, capacity,
            "turn model max yaw rate must be finite and positive");
    }
    return true;
}

static inline bool g1_turn_model_validate_common(
    float dt,
    quat current,
    quat target,
    g1_movement_model_profile profile,
    const g1_turn_model_config& config,
    char* error,
    int capacity)
{
    if (profile != G1MovementRaw && profile != G1MovementHoldenV1
            && profile != G1MovementHoldenTurnV1) {
        return g1_turn_model_fail(
            error, capacity, "turn model profile is unknown");
    }
    if (!std::isfinite(dt) || dt <= 0.0f) {
        return g1_turn_model_fail(
            error, capacity, "turn model dt must be finite and positive");
    }
    if (!g1_turn_model_quat_is_finite(current)) {
        return g1_turn_model_fail(
            error, capacity, "turn model current heading must be finite");
    }
    if (!g1_turn_model_quat_is_finite(target)) {
        return g1_turn_model_fail(
            error, capacity, "turn model target heading must be finite");
    }
    if (std::fabs(quat_length(current) - 1.0f) > G1_TURN_PLANAR_TOLERANCE) {
        return g1_turn_model_fail(
            error, capacity, "turn model current heading must be unit");
    }
    if (std::fabs(quat_length(target) - 1.0f) > G1_TURN_PLANAR_TOLERANCE) {
        return g1_turn_model_fail(
            error, capacity, "turn model target heading must be unit");
    }
    if (!g1_turn_model_validate_config(config, error, capacity)) {
        return false;
    }
    return true;
}

static inline bool g1_turn_model_require_planar(
    quat q,
    char* error,
    int capacity)
{
    if (std::fabs(q.x) > G1_TURN_PLANAR_TOLERANCE
            || std::fabs(q.z) > G1_TURN_PLANAR_TOLERANCE) {
        return g1_turn_model_fail(
            error, capacity, "turn model heading must be planar yaw");
    }
    return true;
}

// Limit `current` one step of duration `dt` toward `target`. Writes the
// bounded heading to `output` only after every input is validated.
static inline bool g1_turn_model_step(
    quat& output,
    quat current,
    quat target,
    float dt,
    g1_movement_model_profile profile,
    const g1_turn_model_config& config,
    char* error,
    int capacity)
{
    if (!g1_turn_model_validate_common(
            dt, current, target, profile, config, error, capacity)) {
        return false;
    }
    if (profile != G1MovementHoldenTurnV1) {
        output = target;
        return true;
    }
    if (!g1_turn_model_require_planar(current, error, capacity)
            || !g1_turn_model_require_planar(target, error, capacity)) {
        return false;
    }
    const float current_yaw = 2.0f * std::atan2(current.y, current.w);
    const float target_yaw = 2.0f * std::atan2(target.y, target.w);
    const float delta = g1_turn_wrap_signed(target_yaw - current_yaw);
    const float maximum = config.max_yaw_rate_radians * dt;
    if (std::fabs(delta) <= maximum) {
        output = target;
        return true;
    }
    const float applied_yaw = current_yaw + std::copysign(maximum, delta);
    output = quat_from_angle_axis(applied_yaw, vec3(0.0f, 1.0f, 0.0f));
    return true;
}

// Predict a sequence of bounded headings from a local copy of `current`
// advancing toward `target` at `sample_dt` per sample. The persistent state
// (`current` is passed by value) is never mutated, and the output slice is
// left untouched when any input or intermediate step is invalid.
static inline bool g1_turn_model_predict(
    slice1d<quat> output,
    quat current,
    quat target,
    float sample_dt,
    g1_movement_model_profile profile,
    const g1_turn_model_config& config,
    char* error,
    int capacity)
{
    if (!g1_turn_model_validate_common(
            sample_dt, current, target, profile, config, error, capacity)) {
        return false;
    }
    if (profile == G1MovementHoldenTurnV1
            && (!g1_turn_model_require_planar(current, error, capacity)
                || !g1_turn_model_require_planar(target, error, capacity))) {
        return false;
    }
    array1d<quat> predicted_local;
    predicted_local.resize(output.size);
    quat predicted = current;
    for (int i = 0; i < output.size; ++i) {
        quat next;
        if (!g1_turn_model_step(
                next, predicted, target, sample_dt,
                profile, config, error, capacity)) {
            return false;
        }
        predicted = next;
        predicted_local(i) = predicted;
    }
    for (int i = 0; i < output.size; ++i) {
        output(i) = predicted_local(i);
    }
    return true;
}
