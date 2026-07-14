#pragma once

#include <stdlib.h>
#include "array.h"
#include "vec.h"
#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static inline bool motion_match_query_is_finite_31d(
    const slice1d<float> query)
{
    if (query.size != 31) return false;
    for (int i = 0; i < 31; ++i) {
        uint32_t bits = 0;
        memcpy(&bits, &query(i), sizeof(bits));
        if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000))
            return false;
    }
    return true;
}

static inline bool motion_match_query_bits_hex(
    char* output, int capacity, const slice1d<float> query)
{
    if (output == NULL || capacity < 31 * 8 + 1 ||
        !motion_match_query_is_finite_31d(query))
        return false;
    for (int i = 0; i < 31; ++i) {
        uint32_t bits = 0;
        memcpy(&bits, &query(i), sizeof(bits));
        snprintf(output + i * 8, 9, "%08x", (unsigned)bits);
    }
    output[31 * 8] = '\0';
    return true;
}

struct motion_match_pose_diagnostic
{
    float hips_y = 0.0f;
    float hips_clearance = 0.0f;
    float left_toe_clearance = 0.0f;
    float right_toe_clearance = 0.0f;
    // Minimum over Hips plus both knees, ankles, and toes (seven probes).
    float minimum_clearance = 0.0f;
};

struct motion_match_log_row
{
    int frame = 0;
    float fixed_dt = 0.04f;
    const char* scene_id = "grail-curb-default";
    const char* mode = "live";
    const char* route = "manual";
    const char* query_bits_hex = "";
    int query_database_frame = 0;
    int query_range = 0;
    int selected_database_frame = 0;
    int database_frame = 0;
    int range = 0;
    int source_range = 0;
    bool searched = false;
    bool transitioned = false;
    float incumbent_cost = 0.0f;
    float selected_cost = 0.0f;
    float selected_terrain_error = 0.0f;
    float effective_terrain_weight = 0.0f;
    float terrain[4] = {};
    vec3 terrain_points[4] = {};
    motion_match_pose_diagnostic raw_selected;
    motion_match_pose_diagnostic inertialized;
    motion_match_pose_diagnostic rendered;
    float hips_inertial_offset_y = 0.0f;
    float runtime_root_surface_height = 0.0f;
    float runtime_left_toe_surface_height = 0.0f;
    float runtime_right_toe_surface_height = 0.0f;
    float adjustment_xz = 0.0f;
    float adjustment_y = 0.0f;
    float clamp_xz = 0.0f;
    float clamp_y = 0.0f;
    bool matching_enabled = true;
    bool adjustment_enabled = true;
    bool clamping_enabled = true;
    bool support_retargeting_enabled = false;
    bool ik_enabled = false;
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
};

struct motion_match_log
{
    FILE* file = NULL;
    const char* path = NULL;

    bool io_error(
        char* error, int error_capacity,
        const char* action, const int saved_errno) const
    {
        if (error != NULL && error_capacity > 0) {
            snprintf(
                error, (size_t)error_capacity, "%s: cannot %s motion log (%s)",
                path != NULL ? path : "<disabled>", action,
                strerror(saved_errno != 0 ? saved_errno : EIO));
        }
        return false;
    }

    bool open(const char* log_path, char* error, int error_capacity)
    {
        if (log_path == NULL) return true;
        path = log_path;
        file = fopen(path, "w");
        if (file == NULL) {
            return io_error(error, error_capacity, "open", errno);
        }
        const bool header_ok = fprintf(file,
            "frame,fixed_dt,scene_id,mode,route,query_bits_hex,"
            "query_database_frame,query_range,selected_database_frame,"
            "database_frame,range,source_range,searched,transitioned,"
            "incumbent_cost,selected_cost,selected_terrain_error,"
            "effective_terrain_weight,terrain0,terrain1,terrain2,terrain3,"
            "terrain_point0_x,terrain_point0_y,terrain_point0_z,"
            "terrain_point1_x,terrain_point1_y,terrain_point1_z,"
            "terrain_point2_x,terrain_point2_y,terrain_point2_z,"
            "terrain_point3_x,terrain_point3_y,terrain_point3_z,"
            "raw_selected_hips_y,inertialized_hips_y,rendered_hips_y,"
            "hips_inertial_offset_y,runtime_root_surface_height,"
            "runtime_left_toe_surface_height,runtime_right_toe_surface_height,"
            "raw_selected_hips_clearance,raw_selected_left_toe_clearance,"
            "raw_selected_right_toe_clearance,raw_selected_min_clearance,"
            "inertialized_hips_clearance,inertialized_left_toe_clearance,"
            "inertialized_right_toe_clearance,inertialized_min_clearance,"
            "rendered_hips_clearance,rendered_left_toe_clearance,"
            "rendered_right_toe_clearance,rendered_min_clearance,"
            "adjustment_xz,adjustment_y,clamp_xz,clamp_y,matching_enabled,"
            "adjustment_enabled,clamping_enabled,support_retargeting_enabled,"
            "ik_enabled"
            ",source_name,source_terrain,source_index,continuation_cost,"
            "source_root_height,source_left_toe_height,source_right_toe_height,"
            "runtime_support_root_height,runtime_support_left_toe_height,"
            "runtime_support_right_toe_height,support_root_delta,"
            "support_left_toe_delta,support_right_toe_delta,support_height,"
            "support_velocity,support_source,airborne_frames,left_contact,"
            "right_contact,support_retargeted_hips_y,ik_adjusted_hips_y,"
            "simulation_x,simulation_z,walkability_class,blocked,"
            "blocked_reason,blocked_distance,blocked_point_x,blocked_point_z,"
            "commanded_speed,applied_speed,route_waypoint,route_complete,"
            "route_target_height,scene_generation,scene_frame,"
            "scene_reset_count,scene_switch_failed,motion_pack_load_count,"
            "model_load_count,model_unload_count,live_model_count\n") >= 0;
        if (!header_ok || fflush(file) != 0) {
            const int saved_errno = errno;
            fclose(file);
            file = NULL;
            return io_error(error, error_capacity, "initialize", saved_errno);
        }
        return true;
    }

    bool write(
        const motion_match_log_row& r,
        char* error, const int error_capacity)
    {
        if (file == NULL) return true;
        bool ok = fprintf(file,
            "%d,%.9g,%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,"
            "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g",
            r.frame, r.fixed_dt, r.scene_id, r.mode, r.route,
            r.query_bits_hex,
            r.query_database_frame, r.query_range, r.selected_database_frame,
            r.database_frame, r.range, r.source_range,
            (int)r.searched, (int)r.transitioned,
            r.incumbent_cost, r.selected_cost, r.selected_terrain_error,
            r.effective_terrain_weight, r.terrain[0], r.terrain[1],
            r.terrain[2], r.terrain[3]) >= 0;
        for (int i = 0; ok && i < 4; ++i) {
            ok = fprintf(file, ",%.9g,%.9g,%.9g",
                    r.terrain_points[i].x,
                    r.terrain_points[i].y,
                    r.terrain_points[i].z) >= 0;
        }
        if (ok) ok = fprintf(file,
            ",%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
            "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
            "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
            "%d,%d,%d,%d,%d",
            r.raw_selected.hips_y, r.inertialized.hips_y,
            r.rendered.hips_y, r.hips_inertial_offset_y,
            r.runtime_root_surface_height,
            r.runtime_left_toe_surface_height,
            r.runtime_right_toe_surface_height,
            r.raw_selected.hips_clearance,
            r.raw_selected.left_toe_clearance,
            r.raw_selected.right_toe_clearance,
            r.raw_selected.minimum_clearance,
            r.inertialized.hips_clearance,
            r.inertialized.left_toe_clearance,
            r.inertialized.right_toe_clearance,
            r.inertialized.minimum_clearance,
            r.rendered.hips_clearance,
            r.rendered.left_toe_clearance,
            r.rendered.right_toe_clearance,
            r.rendered.minimum_clearance,
            r.adjustment_xz, r.adjustment_y, r.clamp_xz, r.clamp_y,
            (int)r.matching_enabled, (int)r.adjustment_enabled,
            (int)r.clamping_enabled, (int)r.support_retargeting_enabled,
            (int)r.ik_enabled) >= 0;
        if (ok) ok = fprintf(file,
            ",%s,%s,%d,%.9g,"
            "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
            "%s,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%d,%d,%s,"
            "%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d,%.9g,%d,%d,%d,%d,%d,%d,%d,%d\n",
            r.source_name, r.source_terrain, r.source_index,
            r.continuation_cost,
            r.source_root_height, r.source_left_toe_height,
            r.source_right_toe_height,
            r.runtime_support_root_height,
            r.runtime_support_left_toe_height,
            r.runtime_support_right_toe_height,
            r.support_root_delta, r.support_left_toe_delta,
            r.support_right_toe_delta, r.support_height, r.support_velocity,
            r.support_source, r.airborne_frames,
            (int)r.left_contact, (int)r.right_contact,
            r.support_retargeted_hips_y, r.ik_adjusted_hips_y,
            r.simulation_x, r.simulation_z,
            r.walkability_class, (int)r.blocked, r.blocked_reason,
            r.blocked_distance, r.blocked_point_x, r.blocked_point_z,
            r.commanded_speed, r.applied_speed,
            r.route_waypoint, (int)r.route_complete, r.route_target_height,
            r.scene_generation, r.scene_frame, r.scene_reset_count,
            (int)r.scene_switch_failed, r.motion_pack_load_count,
            r.model_load_count, r.model_unload_count, r.live_model_count) >= 0;
        if (ok) ok = fflush(file) == 0;
        return ok ? true : io_error(error, error_capacity, "write", errno);
    }

    bool close(char* error, const int error_capacity)
    {
        if (file == NULL) return true;
        int saved_errno = 0;
        if (fflush(file) != 0) saved_errno = errno;
        if (fclose(file) != 0 && saved_errno == 0) saved_errno = errno;
        file = NULL;
        if (saved_errno != 0) {
            const bool result = io_error(
                error, error_capacity, "close", saved_errno);
            path = NULL;
            return result;
        }
        path = NULL;
        return true;
    }
};
