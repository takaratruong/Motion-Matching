#pragma once

#include "scene_runtime.h"

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

struct deterministic_route_sample
{
    vec3 command;
    int waypoint = 0;
    bool complete = false;
};

static inline bool deterministic_route_segment_frames(
    int& frames,
    const std::pair<float, float>& start,
    const std::pair<float, float>& stop,
    float dt,
    float speed)
{
    float maximum_step = 0.0f;
    return terrain_f32_mul(maximum_step, speed, dt) &&
           scene_route_sample_count(frames, start, stop, maximum_step);
}

static inline bool deterministic_route_hold_frames(
    int& frames, float seconds, float dt)
{
    float ratio = 0.0f;
    if (!terrain_f32_div(ratio, seconds, dt)) return false;
    const float rounded = ceilf(ratio);
    if (!terrain_float_is_finite(rounded) || rounded < 0.0f ||
        static_cast<double>(rounded) > static_cast<double>(INT_MAX))
    {
        return false;
    }
    frames = static_cast<int>(rounded);
    return true;
}

static inline bool deterministic_route_inputs_valid(
    const scene_route& route, float dt, float speed)
{
    if (!terrain_float_is_finite(dt) || dt <= 0.0f ||
        !terrain_float_is_finite(speed) || speed <= 0.0f ||
        route.waypoints_xz.size() < 2 ||
        route.waypoints_xz.size() > static_cast<size_t>(INT_MAX) ||
        !terrain_float_is_finite(route.landing_hold_seconds) ||
        route.landing_hold_seconds < 0.0f ||
        (route.landing_hold_seconds > 0.0f &&
         route.waypoints_xz.size() < 4))
    {
        return false;
    }
    for (const std::pair<float, float>& waypoint : route.waypoints_xz) {
        if (!terrain_float_is_normal_or_zero_query(waypoint.first) ||
            !terrain_float_is_normal_or_zero_query(waypoint.second))
        {
            return false;
        }
    }
    return true;
}

static inline bool deterministic_route_schedule_frames(
    int& total, const scene_route& route, float dt, float speed)
{
    if (!deterministic_route_inputs_valid(route, dt, speed)) return false;
    int64_t candidate = 0;
    for (size_t i = 0; i + 1 < route.waypoints_xz.size(); ++i) {
        float dx = 0.0f;
        float dz = 0.0f;
        float length = 0.0f;
        int frames = 0;
        if (!scene_route_segment(
                dx, dz, length,
                route.waypoints_xz[i], route.waypoints_xz[i + 1]) ||
            length <= 1.0e-6f ||
            !deterministic_route_segment_frames(
                frames,
                route.waypoints_xz[i], route.waypoints_xz[i + 1],
                dt, speed))
        {
            return false;
        }
        candidate += frames;
        if (candidate > INT_MAX) return false;
        if (i + 1 == 2 && route.landing_hold_seconds > 0.0f) {
            int hold = 0;
            if (!deterministic_route_hold_frames(
                    hold, route.landing_hold_seconds, dt))
            {
                return false;
            }
            candidate += hold;
            if (candidate > INT_MAX) return false;
        }
    }
    total = static_cast<int>(candidate);
    return true;
}

static inline bool deterministic_route_command(
    deterministic_route_sample& out,
    const scene_route& route,
    int frame,
    float dt,
    float speed,
    char* error,
    int capacity)
{
    if (frame < 0 || !deterministic_route_inputs_valid(route, dt, speed)) {
        return scene_error(
            error,
            capacity,
            "route '%s': invalid frame, dt, speed, hold, or waypoint count",
            route.id.c_str());
    }

    int total = 0;
    if (!deterministic_route_schedule_frames(total, route, dt, speed)) {
        return scene_error(
            error,
            capacity,
            "route '%s': invalid binary32 segment or schedule timing",
            route.id.c_str());
    }

    int64_t cursor = 0;
    for (size_t i = 0; i + 1 < route.waypoints_xz.size(); ++i) {
        float dx = 0.0f;
        float dz = 0.0f;
        float length = 0.0f;
        float unit_x = 0.0f;
        float unit_z = 0.0f;
        float command_x = 0.0f;
        float command_z = 0.0f;
        int frames = 0;
        if (!scene_route_segment(
                dx, dz, length,
                route.waypoints_xz[i], route.waypoints_xz[i + 1]) ||
            length <= 1.0e-6f ||
            !deterministic_route_segment_frames(
                frames,
                route.waypoints_xz[i], route.waypoints_xz[i + 1],
                dt, speed))
        {
            return scene_error(
                error, capacity,
                "route '%s': segment %zu has invalid binary32 timing",
                route.id.c_str(), i);
        }
        if (static_cast<int64_t>(frame) < cursor + frames) {
            if (!terrain_f32_div(unit_x, dx, length) ||
                !terrain_f32_div(unit_z, dz, length) ||
                !terrain_f32_mul(command_x, speed, unit_x) ||
                !terrain_f32_mul(command_z, speed, unit_z))
            {
                return scene_error(
                    error, capacity,
                    "route '%s': segment %zu command overflow",
                    route.id.c_str(), i);
            }
            deterministic_route_sample sample;
            sample.command = vec3(command_x, 0.0f, command_z);
            sample.waypoint = static_cast<int>(i) + 1;
            out = sample;
            return true;
        }
        cursor += frames;
        if (i + 1 == 2 && route.landing_hold_seconds > 0.0f) {
            int hold = 0;
            if (!deterministic_route_hold_frames(
                    hold, route.landing_hold_seconds, dt))
            {
                return scene_error(
                    error, capacity,
                    "route '%s': invalid landing hold",
                    route.id.c_str());
            }
            if (static_cast<int64_t>(frame) < cursor + hold) {
                deterministic_route_sample sample;
                sample.waypoint = 2;
                out = sample;
                return true;
            }
            cursor += hold;
        }
    }

    deterministic_route_sample sample;
    sample.waypoint = static_cast<int>(route.waypoints_xz.size()) - 1;
    sample.complete = true;
    out = sample;
    return true;
}

static inline int deterministic_route_motion_frames(
    const scene_route& route, float dt = 0.04f, float speed = 0.50f)
{
    int total = 0;
    return deterministic_route_schedule_frames(total, route, dt, speed)
        ? total : -1;
}

static inline float deterministic_route_target_height(
    const scene_route& route, const heightfield& terrain)
{
    struct bin
    {
        int key = 0;
        int count = 0;
        float sum = 0.0f;
    };

    if (!deterministic_route_inputs_valid(route, 0.04f, 0.50f) ||
        !terrain_heightfield_is_queryable(terrain))
    {
        return NAN;
    }
    const float base = heightfield_sample_v2(
        terrain,
        route.waypoints_xz.front().first,
        route.waypoints_xz.front().second);
    float half_cell = 0.0f;
    if (!terrain_float_is_finite(base) ||
        !terrain_f32_div(half_cell, terrain.cell_size, 2.0f))
    {
        return NAN;
    }

    std::vector<bin> bins;
    for (size_t i = 0; i + 1 < route.waypoints_xz.size(); ++i) {
        int steps = 0;
        if (!scene_route_sample_count(
                steps,
                route.waypoints_xz[i], route.waypoints_xz[i + 1],
                half_cell))
        {
            return NAN;
        }
        for (int64_t sample_index = 0;
             sample_index <= static_cast<int64_t>(steps);
             ++sample_index)
        {
            const int step = static_cast<int>(sample_index);
            float x = 0.0f;
            float z = 0.0f;
            if (!scene_binary32_lerp(
                    x,
                    route.waypoints_xz[i].first,
                    route.waypoints_xz[i + 1].first,
                    step,
                    steps) ||
                !scene_binary32_lerp(
                    z,
                    route.waypoints_xz[i].second,
                    route.waypoints_xz[i + 1].second,
                    step,
                    steps))
            {
                return NAN;
            }
            const float height = heightfield_sample_v2(terrain, x, z);
            float difference = 0.0f;
            if (!terrain_float_is_finite(height) ||
                !terrain_f32_sub(difference, height, base))
            {
                return NAN;
            }
            if (fabsf(difference) <= 0.02f) continue;

            float scaled_height = 0.0f;
            if (!terrain_f32_mul(scaled_height, height, 100.0f) ||
                static_cast<double>(scaled_height) <
                    static_cast<double>(INT_MIN) ||
                static_cast<double>(scaled_height) >
                    static_cast<double>(INT_MAX))
            {
                return NAN;
            }
            const long rounded_key = lroundf(scaled_height);
            if (rounded_key < INT_MIN || rounded_key > INT_MAX) return NAN;
            const int key = static_cast<int>(rounded_key);
            size_t found = 0;
            while (found < bins.size() && bins[found].key != key) ++found;
            if (found == bins.size()) {
                bin value;
                value.key = key;
                bins.push_back(value);
            }
            if (bins[found].count == INT_MAX ||
                !terrain_f32_add(
                    bins[found].sum, bins[found].sum, height))
            {
                return NAN;
            }
            ++bins[found].count;
        }
    }
    if (bins.empty()) return base;

    size_t best = 0;
    for (size_t i = 1; i < bins.size(); ++i) {
        if (bins[i].count > bins[best].count ||
            (bins[i].count == bins[best].count &&
             bins[i].key > bins[best].key))
        {
            best = i;
        }
    }
    float target = 0.0f;
    return terrain_f32_div(
               target,
               bins[best].sum,
               static_cast<float>(bins[best].count))
        ? target : NAN;
}
