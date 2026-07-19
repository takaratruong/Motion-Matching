#pragma once

#include "g1_controller_state.h"
#include "g1_sole_diagnostics.h"
#include "route_runtime.h"

struct g1_runtime_diagnostic_snapshot
{
    const char* source_name = "";
    const char* source_terrain = "";
    int source_index = 0;
    float continuation_cost = 0.0f;
    float source_root_height = 0.0f;
    float source_left_toe_height = 0.0f;
    float source_right_toe_height = 0.0f;
    float runtime_support_root_height = 0.0f;
    float runtime_support_left_toe_height = 0.0f;
    float runtime_support_right_toe_height = 0.0f;
    float support_root_delta = 0.0f;
    float support_left_toe_delta = 0.0f;
    float support_right_toe_delta = 0.0f;
    float support_height = 0.0f;
    float support_velocity = 0.0f;
    const char* support_source = "";
    int airborne_frames = 0;
    bool left_contact = false;
    bool right_contact = false;
    float support_retargeted_hips_y = 0.0f;
    float ik_adjusted_hips_y = 0.0f;
    float simulation_x = 0.0f;
    float simulation_z = 0.0f;
    int walkability_class = 1;
    bool blocked = false;
    const char* blocked_reason = "";
    float blocked_distance = 0.0f;
    float blocked_point_x = 0.0f;
    float blocked_point_z = 0.0f;
    float commanded_speed = 0.0f;
    float applied_speed = 0.0f;
    int route_waypoint = 0;
    bool route_complete = false;
    float route_target_height = 0.0f;
    int scene_generation = 0;
    int scene_frame = 0;
    int scene_reset_count = 0;
    bool scene_switch_failed = false;
    int motion_pack_load_count = 0;
    int model_load_count = 0;
    int model_unload_count = 0;
    int live_model_count = 0;
    float sole_clearance[2][4] = {};
    float sole_minimum_clearance[2] = {};
    float sole_global_minimum_clearance = 0.0f;
    float stance_slip[2] = {};
    bool stance_slip_reset[2] = {};
};

static inline bool g1_runtime_sole_snapshot_is_valid(
    const G1SoleDiagnosticSnapshot& sole,
    int scene_generation,
    bool left_contact,
    bool right_contact)
{
    if (sole.scene_generation != scene_generation ||
        sole.contact[0] != left_contact ||
        sole.contact[1] != right_contact) {
        return false;
    }
    float foot_minima[2] = {FLT_MAX, FLT_MAX};
    for (int foot = 0; foot < 2; ++foot) {
        if (!terrain_float_is_finite(sole.stance_slip[foot]) ||
            sole.stance_slip[foot] < 0.0f ||
            (sole.slip_reset[foot] &&
             sole.stance_slip[foot] != 0.0f) ||
            (!sole.contact[foot] && sole.stance_slip[foot] != 0.0f)) {
            return false;
        }
        for (int probe = 0; probe < 4; ++probe) {
            if (!g1_sole_vec3_is_finite(sole.world_points[foot][probe]) ||
                !terrain_float_is_finite(
                    sole.surface_heights[foot][probe]) ||
                !terrain_float_is_finite(sole.clearances[foot][probe])) {
                return false;
            }
            if (sole.clearances[foot][probe] < foot_minima[foot]) {
                foot_minima[foot] = sole.clearances[foot][probe];
            }
        }
        if (sole.minimum_clearance[foot] != foot_minima[foot]) return false;
    }
    const float global_minimum = foot_minima[0] < foot_minima[1]
        ? foot_minima[0] : foot_minima[1];
    return sole.global_minimum_clearance == global_minimum;
}

static inline bool g1_runtime_support_state_is_finite(
    const support_frame_state& state)
{
    return terrain_float_is_finite(state.height) &&
           terrain_float_is_finite(state.velocity) &&
           terrain_float_is_finite(state.nominal_height) &&
           terrain_float_is_finite(state.nominal_velocity) &&
           terrain_float_is_finite(state.offset_height) &&
           terrain_float_is_finite(state.offset_velocity) &&
           state.airborne_frames >= 0 &&
           state.source >= support_root && state.source <= support_airborne_root;
}

static inline bool g1_runtime_traversal_is_finite(
    const traversability_diagnostics& traversal)
{
    return traversal.walkability_class >= 0 &&
           traversal.walkability_class <= 2 &&
           traversal.reason >= walkability_clear &&
           traversal.reason <= walkability_nonfinite &&
           terrain_float_is_finite(traversal.distance) &&
           terrain_float_is_finite(traversal.commanded_speed) &&
           terrain_float_is_finite(traversal.applied_speed) &&
           terrain_float_is_finite(traversal.point.x) &&
           terrain_float_is_finite(traversal.point.y) &&
           terrain_float_is_finite(traversal.point.z);
}

static inline bool g1_runtime_snapshot_is_valid(
    const g1_runtime_diagnostic_snapshot& snapshot)
{
    const float values[] = {
        snapshot.continuation_cost,
        snapshot.source_root_height,
        snapshot.source_left_toe_height,
        snapshot.source_right_toe_height,
        snapshot.runtime_support_root_height,
        snapshot.runtime_support_left_toe_height,
        snapshot.runtime_support_right_toe_height,
        snapshot.support_root_delta,
        snapshot.support_left_toe_delta,
        snapshot.support_right_toe_delta,
        snapshot.support_height,
        snapshot.support_velocity,
        snapshot.support_retargeted_hips_y,
        snapshot.ik_adjusted_hips_y,
        snapshot.simulation_x,
        snapshot.simulation_z,
        snapshot.blocked_distance,
        snapshot.blocked_point_x,
        snapshot.blocked_point_z,
        snapshot.commanded_speed,
        snapshot.applied_speed,
        snapshot.route_target_height,
        snapshot.sole_minimum_clearance[0],
        snapshot.sole_minimum_clearance[1],
        snapshot.sole_global_minimum_clearance,
        snapshot.stance_slip[0],
        snapshot.stance_slip[1],
    };
    for (float value : values) {
        if (!terrain_float_is_finite(value)) return false;
    }
    return snapshot.source_name != NULL && snapshot.source_name[0] != '\0' &&
           snapshot.source_terrain != NULL &&
           snapshot.source_terrain[0] != '\0' &&
           snapshot.support_source != NULL &&
           snapshot.support_source[0] != '\0' &&
           snapshot.blocked_reason != NULL &&
           snapshot.blocked_reason[0] != '\0' &&
           snapshot.source_index >= 0 && snapshot.airborne_frames >= 0 &&
           snapshot.walkability_class >= 0 &&
           snapshot.walkability_class <= 2 &&
           snapshot.route_waypoint >= 0 && snapshot.scene_generation >= 0 &&
           snapshot.scene_frame >= 0 && snapshot.scene_reset_count >= 0 &&
           snapshot.motion_pack_load_count >= 0 &&
           snapshot.model_load_count >= 0 &&
           snapshot.model_unload_count >= 0 &&
           snapshot.live_model_count == 1;
}

static inline bool g1_runtime_diagnostics_build(
    g1_runtime_diagnostic_snapshot& out,
    const motion_pack_manifest& manifest,
    const g1_controller_state& state,
    const traversability_diagnostics& traversal,
    const deterministic_route_sample& route,
    float route_target_height,
    int scene_generation,
    int scene_reset_count,
    bool scene_switch_failed,
    int motion_pack_load_count,
    int model_load_count,
    int model_unload_count,
    char* error,
    int capacity)
{
    const int source_index = motion_source_for_frame(manifest, state.frame_index);
    if (source_index < 0 ||
        source_index >= static_cast<int>(manifest.sources.size()))
    {
        return scene_error(
            error, capacity,
            "runtime diagnostics: database frame %d has no motion source",
            state.frame_index);
    }
    const motion_source_record& source =
        manifest.sources[static_cast<size_t>(source_index)];
    if (source.name.empty() || source.terrain_id.empty()) {
        return scene_error(
            error, capacity,
            "runtime diagnostics: motion source %d has empty metadata",
            source_index);
    }
    if (state.global_bone_positions.data == NULL ||
        state.global_bone_positions.size <= G1_Hips ||
        state.walkability_class < 0 || state.walkability_class > 2 ||
        state.scene_frame < 0 || route.waypoint < 0 ||
        !support_observation_is_finite(state.support_observation_now) ||
        !g1_runtime_support_state_is_finite(state.support) ||
        !g1_runtime_traversal_is_finite(traversal) ||
        !terrain_float_is_finite(state.incumbent_cost) ||
        !terrain_float_is_finite(state.simulation_position.x) ||
        !terrain_float_is_finite(state.simulation_position.y) ||
        !terrain_float_is_finite(state.simulation_position.z) ||
        !terrain_float_is_finite(state.global_bone_positions(G1_Hips).x) ||
        !terrain_float_is_finite(state.global_bone_positions(G1_Hips).y) ||
        !terrain_float_is_finite(state.global_bone_positions(G1_Hips).z) ||
        !terrain_float_is_finite(route.command.x) ||
        !terrain_float_is_finite(route.command.y) ||
        !terrain_float_is_finite(route.command.z) ||
        !terrain_float_is_finite(route_target_height))
    {
        return scene_error(
            error, capacity,
            "runtime diagnostics: non-finite or invalid runtime state");
    }
    if (scene_generation < 0 || scene_reset_count < 0 ||
        motion_pack_load_count < 0 || model_load_count < 0 ||
        model_unload_count < 0 || model_unload_count > model_load_count)
    {
        return scene_error(
            error, capacity,
            "runtime diagnostics: counters must be nonnegative and ordered");
    }
    const int live_model_count = model_load_count - model_unload_count;
    if (live_model_count != 1) {
        return scene_error(
            error, capacity,
            "runtime diagnostics: expected exactly one live model, got %d",
            live_model_count);
    }

    g1_runtime_diagnostic_snapshot candidate;
    candidate.source_name = source.name.c_str();
    candidate.source_terrain = source.terrain_id.c_str();
    candidate.source_index = source_index;
    candidate.continuation_cost = state.incumbent_cost;
    candidate.source_root_height =
        state.support_observation_now.source_height[0];
    candidate.source_left_toe_height =
        state.support_observation_now.source_height[1];
    candidate.source_right_toe_height =
        state.support_observation_now.source_height[2];
    candidate.runtime_support_root_height =
        state.support_observation_now.runtime_height[0];
    candidate.runtime_support_left_toe_height =
        state.support_observation_now.runtime_height[1];
    candidate.runtime_support_right_toe_height =
        state.support_observation_now.runtime_height[2];
    candidate.support_root_delta = state.support_observation_now.delta[0];
    candidate.support_left_toe_delta = state.support_observation_now.delta[1];
    candidate.support_right_toe_delta = state.support_observation_now.delta[2];
    candidate.support_height = state.support.height;
    candidate.support_velocity = state.support.velocity;
    candidate.support_source = support_source_name(state.support.source);
    candidate.airborne_frames = state.support.airborne_frames;
    candidate.left_contact = state.support_observation_now.contact[0];
    candidate.right_contact = state.support_observation_now.contact[1];
    candidate.support_retargeted_hips_y =
        state.global_bone_positions(G1_Hips).y;
    candidate.ik_adjusted_hips_y = candidate.support_retargeted_hips_y;
    candidate.simulation_x = state.simulation_position.x;
    candidate.simulation_z = state.simulation_position.z;
    candidate.walkability_class = state.walkability_class;
    candidate.blocked = traversal.blocked;
    candidate.blocked_reason = walkability_reason_name(traversal.reason);
    candidate.blocked_distance = traversal.distance;
    candidate.blocked_point_x = traversal.point.x;
    candidate.blocked_point_z = traversal.point.z;
    candidate.commanded_speed = traversal.commanded_speed;
    candidate.applied_speed = traversal.applied_speed;
    candidate.route_waypoint = route.waypoint;
    candidate.route_complete = route.complete;
    candidate.route_target_height = route_target_height;
    candidate.scene_generation = scene_generation;
    candidate.scene_frame = state.scene_frame;
    candidate.scene_reset_count = scene_reset_count;
    candidate.scene_switch_failed = scene_switch_failed;
    candidate.motion_pack_load_count = motion_pack_load_count;
    candidate.model_load_count = model_load_count;
    candidate.model_unload_count = model_unload_count;
    candidate.live_model_count = live_model_count;
    if (!g1_runtime_snapshot_is_valid(candidate)) {
        return scene_error(
            error, capacity,
            "runtime diagnostics: snapshot validation failed");
    }
    out = candidate;
    return true;
}

static inline bool g1_runtime_diagnostics_attach_sole(
    g1_runtime_diagnostic_snapshot& out,
    const G1SoleDiagnosticSnapshot& sole,
    char* error,
    int capacity)
{
    if (!g1_runtime_snapshot_is_valid(out) ||
        !g1_runtime_sole_snapshot_is_valid(
            sole,
            out.scene_generation,
            out.left_contact,
            out.right_contact)) {
        return scene_error(
            error,
            capacity,
            "runtime diagnostics: invalid physical sole snapshot");
    }
    g1_runtime_diagnostic_snapshot candidate = out;
    for (int foot = 0; foot < 2; ++foot) {
        for (int probe = 0; probe < 4; ++probe) {
            candidate.sole_clearance[foot][probe] =
                sole.clearances[foot][probe];
        }
        candidate.sole_minimum_clearance[foot] =
            sole.minimum_clearance[foot];
        candidate.stance_slip[foot] = sole.stance_slip[foot];
        candidate.stance_slip_reset[foot] = sole.slip_reset[foot];
    }
    candidate.sole_global_minimum_clearance =
        sole.global_minimum_clearance;
    if (!g1_runtime_snapshot_is_valid(candidate)) {
        return scene_error(
            error,
            capacity,
            "runtime diagnostics: physical sole snapshot validation failed");
    }
    out = candidate;
    return true;
}
