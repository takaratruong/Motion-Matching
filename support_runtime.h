#pragma once

#include <cstdlib>

#include "array.h"
#include "spring.h"
#include "terrain_runtime.h"
#include "vec.h"

#include <cmath>
#include <cstdio>

enum support_source
{
    support_root,
    support_left,
    support_right,
    support_both,
    support_held,
    support_airborne_root,
};

static inline const char* support_source_name(const support_source source)
{
    switch (source) {
    case support_root: return "root";
    case support_left: return "left";
    case support_right: return "right";
    case support_both: return "both";
    case support_held: return "held";
    default: return "airborne-root";
    }
}

struct support_observation
{
    float source_height[3] = {};
    float runtime_height[3] = {};
    float delta[3] = {};
    bool contact[2] = {};
};

struct support_frame_state
{
    float height = 0.0f;
    float velocity = 0.0f;
    float nominal_height = 0.0f;
    float nominal_velocity = 0.0f;
    float offset_height = 0.0f;
    float offset_velocity = 0.0f;
    int airborne_frames = 0;
    support_source source = support_root;
    bool initialized = false;
};

static inline bool support_error(
    char* error, const int capacity, const char* message)
{
    if (error != NULL && capacity > 0)
        std::snprintf(error, static_cast<size_t>(capacity), "%s", message);
    return false;
}

static inline bool support_observation_is_finite(
    const support_observation& observation)
{
    for (int i = 0; i < 3; ++i)
        if (!terrain_float_is_finite(observation.source_height[i]) ||
            !terrain_float_is_finite(observation.runtime_height[i]) ||
            !terrain_float_is_finite(observation.delta[i])) return false;
    return true;
}

static inline bool support_observation_build(
    support_observation& out,
    const terrain_support_set& support,
    const int frame,
    const heightfield& terrain,
    const vec3 root,
    const vec3 left_toe,
    const vec3 right_toe,
    const bool left_contact,
    const bool right_contact,
    char* error,
    const int error_capacity)
{
    if (frame < 0 || frame >= support.values.rows || support.values.cols != 3)
        return support_error(error, error_capacity, "support frame is out of range");
    support_observation candidate = {};
    const vec3 points[3] = {root, left_toe, right_toe};
    for (int i = 0; i < 3; ++i) {
        if (!terrain_float_is_finite(points[i].x) ||
            !terrain_float_is_finite(points[i].y) ||
            !terrain_float_is_finite(points[i].z))
            return support_error(error, error_capacity,
                "support FK points must contain only finite values");
        candidate.source_height[i] = support.values(frame, i);
        candidate.runtime_height[i] =
            heightfield_sample_v2(terrain, points[i].x, points[i].z);
        candidate.delta[i] =
            candidate.runtime_height[i] - candidate.source_height[i];
    }
    candidate.contact[0] = left_contact;
    candidate.contact[1] = right_contact;
    if (!support_observation_is_finite(candidate))
        return support_error(error, error_capacity,
            "support observation must contain only finite values");
    out = candidate;
    return true;
}

static inline void support_frame_reset(
    support_frame_state& state, const float initial_height)
{
    state = support_frame_state();
    state.height = initial_height;
    state.nominal_height = initial_height;
    state.initialized = true;
}

static inline void support_frame_rebase(
    support_frame_state& state,
    const float new_nominal_height,
    const float new_nominal_velocity)
{
    state.offset_height = state.height - new_nominal_height;
    state.offset_velocity = state.velocity - new_nominal_velocity;
    state.nominal_height = new_nominal_height;
    state.nominal_velocity = new_nominal_velocity;
}

static inline bool support_frame_update(
    support_frame_state& state,
    const support_observation& observation,
    const bool source_frame_changed,
    const float dt,
    char* error,
    const int error_capacity)
{
    if (!support_observation_is_finite(observation) ||
        !terrain_float_is_finite(dt) || dt <= 0.0f)
        return support_error(error, error_capacity,
            "support update inputs must be finite with positive dt");
    support_frame_state next = state;
    if (!next.initialized) support_frame_reset(next, observation.delta[0]);

    float target = next.nominal_height;
    support_source source = support_held;
    if (observation.contact[0] && observation.contact[1]) {
        next.airborne_frames = 0;
        target = 0.5f * (observation.delta[1] + observation.delta[2]);
        source = support_both;
    } else if (observation.contact[0]) {
        next.airborne_frames = 0;
        target = observation.delta[1];
        source = support_left;
    } else if (observation.contact[1]) {
        next.airborne_frames = 0;
        target = observation.delta[2];
        source = support_right;
    } else {
        ++next.airborne_frames;
        if (next.airborne_frames <= 2) {
            target = next.nominal_height;
            source = support_held;
        } else {
            target = observation.delta[0];
            source = support_airborne_root;
        }
    }
    if (!terrain_float_is_finite(target))
        return support_error(error, error_capacity,
            "support target must be finite");
    const bool source_changed = source_frame_changed || source != next.source;
    const float target_delta = target - next.nominal_height;
    const bool discontinuity = source_changed ||
        std::fabs(target_delta) > 0.02f;
    const float target_velocity = source == support_held || discontinuity
        ? 0.0f
        : target_delta / dt;
    if (discontinuity)
        support_frame_rebase(next, target, target_velocity);
    else {
        next.nominal_height = target;
        next.nominal_velocity = target_velocity;
    }
    next.source = source;
    if (source != support_held)
        decay_spring_damper_exact(
            next.offset_height, next.offset_velocity, 0.10f, dt);
    next.height = next.nominal_height + next.offset_height;
    next.velocity = next.nominal_velocity + next.offset_velocity;
    if (!terrain_float_is_finite(next.height) ||
        !terrain_float_is_finite(next.velocity))
        return support_error(error, error_capacity,
            "support output must remain finite");
    state = next;
    return true;
}

static inline void support_pose_apply(
    const slice1d<vec3> output,
    const slice1d<vec3> inertialized,
    const float support_height)
{
    if (output.size != inertialized.size || output.size <= 0) return;
    for (int i = 0; i < output.size; ++i) output(i) = inertialized(i);
    output(0).y = support_height;
}

static inline float horizontal_length(const vec3 value)
{
    return std::sqrt(value.x * value.x + value.z * value.z);
}

static inline vec3 horizontal_adjust_character_position(
    const vec3 character,
    const vec3 simulation,
    const float halflife,
    const float dt)
{
    const vec3 difference(simulation.x - character.x, 0.0f,
                          simulation.z - character.z);
    const vec3 adjustment = damp_adjustment_exact(difference, halflife, dt);
    return vec3(character.x + adjustment.x, character.y,
                character.z + adjustment.z);
}

static inline vec3 horizontal_adjust_character_position_by_velocity(
    const vec3 character,
    const vec3 character_velocity,
    const vec3 simulation,
    const float max_adjustment_ratio,
    const float halflife,
    const float dt)
{
    vec3 adjustment = damp_adjustment_exact(
        vec3(simulation.x - character.x, 0.0f,
             simulation.z - character.z), halflife, dt);
    const float length_now = horizontal_length(adjustment);
    const float maximum = max_adjustment_ratio *
        horizontal_length(character_velocity) * dt;
    if (length_now > maximum && length_now > 1e-8f)
        adjustment = adjustment * (maximum / length_now);
    return vec3(character.x + adjustment.x, character.y,
                character.z + adjustment.z);
}

static inline vec3 horizontal_clamp_character_position(
    const vec3 character, const vec3 simulation, const float maximum)
{
    vec3 difference(character.x - simulation.x, 0.0f,
                    character.z - simulation.z);
    const float distance = horizontal_length(difference);
    if (distance <= maximum || distance <= 1e-8f) return character;
    difference = difference * (maximum / distance);
    return vec3(simulation.x + difference.x, character.y,
                simulation.z + difference.z);
}
