#pragma once

// Opt-in acceleration/deceleration movement model.
//
// A pure, header-only unit that shapes a persistent intermediate velocity
// toward a traversability-limited command. `raw` returns the command
// unchanged (existing behavior). `holden-v1` limits the per-step change
// magnitude by a fixed acceleration when speeding up and a fixed
// deceleration otherwise, so an abrupt reversal brakes through zero without
// overshoot. All functions validate their inputs and the configuration and
// never mutate caller outputs on failure.

#include "vec.h"
#include "array.h"

#include <cmath>

enum g1_movement_model_profile
{
    G1MovementRaw = 0,
    G1MovementHoldenV1 = 1,
    G1MovementHoldenTurnV1 = 2
};

struct g1_movement_model_config
{
    float acceleration;
    float deceleration;
    bool directional_acceleration;
    bool turn_strength;
};

static inline g1_movement_model_config g1_movement_model_fixed_config()
{
    g1_movement_model_config config;
    config.acceleration = 1.5f;
    config.deceleration = 2.0f;
    config.directional_acceleration = false;
    config.turn_strength = false;
    return config;
}

static inline bool g1_movement_model_vec_is_finite(vec3 v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

static inline bool g1_movement_model_fail(
    char* error,
    int capacity,
    const char* message)
{
    if (error != nullptr && capacity > 0) {
        int i = 0;
        for (; message[i] != '\0' && i < capacity - 1; ++i) {
            error[i] = message[i];
        }
        error[i] = '\0';
    }
    return false;
}

static inline bool g1_movement_model_validate(
    float dt,
    vec3 current,
    vec3 target,
    g1_movement_model_profile profile,
    const g1_movement_model_config& config,
    char* error,
    int capacity)
{
    if (profile != G1MovementRaw && profile != G1MovementHoldenV1
            && profile != G1MovementHoldenTurnV1) {
        return g1_movement_model_fail(
            error, capacity, "movement model profile is unknown");
    }
    if (!std::isfinite(dt) || dt <= 0.0f) {
        return g1_movement_model_fail(
            error, capacity, "movement model dt must be finite and positive");
    }
    if (!g1_movement_model_vec_is_finite(current)) {
        return g1_movement_model_fail(
            error, capacity, "movement model current velocity must be finite");
    }
    if (!g1_movement_model_vec_is_finite(target)) {
        return g1_movement_model_fail(
            error, capacity, "movement model target velocity must be finite");
    }
    if (!std::isfinite(config.acceleration) || config.acceleration <= 0.0f) {
        return g1_movement_model_fail(
            error, capacity,
            "movement model acceleration must be finite and positive");
    }
    if (!std::isfinite(config.deceleration) || config.deceleration <= 0.0f) {
        return g1_movement_model_fail(
            error, capacity,
            "movement model deceleration must be finite and positive");
    }
    return true;
}

// Advance `current` one step of duration `dt` toward `target`. Writes the
// shaped velocity to `output` only after every input is validated.
static inline bool g1_movement_model_step(
    vec3& output,
    vec3 current,
    vec3 target,
    float dt,
    g1_movement_model_profile profile,
    const g1_movement_model_config& config,
    char* error,
    int capacity)
{
    if (!g1_movement_model_validate(
            dt, current, target, profile, config, error, capacity)) {
        return false;
    }
    if (profile == G1MovementRaw) {
        output = target;
        return true;
    }
    const float current_speed = length(current);
    const float target_speed = length(target);
    const float limit = target_speed > current_speed + 1.0e-6f
        ? config.acceleration
        : config.deceleration;
    const vec3 difference = target - current;
    const float distance = length(difference);
    const float maximum_change = limit * dt;
    output = distance <= maximum_change
        ? target
        : current + difference * (maximum_change / distance);
    return true;
}

// Predict a sequence of intermediate velocities from a local copy of
// `current` advancing toward `target` at `sample_dt` per sample. The
// persistent state (`current` is passed by value) is never mutated, and the
// output slice is left untouched when any input is invalid.
static inline bool g1_movement_model_predict(
    slice1d<vec3> output,
    vec3 current,
    vec3 target,
    float sample_dt,
    g1_movement_model_profile profile,
    const g1_movement_model_config& config,
    char* error,
    int capacity)
{
    if (!g1_movement_model_validate(
            sample_dt, current, target, profile, config, error, capacity)) {
        return false;
    }
    vec3 predicted = current;
    for (int i = 0; i < output.size; ++i) {
        vec3 next;
        if (!g1_movement_model_step(
                next, predicted, target, sample_dt,
                profile, config, error, capacity)) {
            return false;
        }
        predicted = next;
        output(i) = predicted;
    }
    return true;
}
