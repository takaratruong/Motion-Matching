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

struct motion_match_clearance_work_diagnostic
{
    uint32_t point_queries = 0;
    uint32_t cells_visited = 0;
    uint32_t primitive_triangle_pairs = 0;
    uint32_t face_patches = 0;
    uint32_t candidate_tests = 0;
    uint32_t subdivision_nodes = 0;
};

struct motion_match_swing_diagnostic
{
    uint32_t candidates_evaluated = 0;
    uint32_t selected_index = UINT32_MAX;
    uint32_t selected_lift_bits = 0;
    uint32_t materialized_command_y_bits = 0;
    char actual_sphere_center_bits_hex[12 * 8 + 1] =
        "000000000000000000000000000000000000000000000000"
        "000000000000000000000000000000000000000000000000";
    const char* selected_clearance_status = "invalid-input";
    bool selected_controller_constraints_passed = false;
    bool selected_clearance_certified = false;
    double lower_margin = 0.0;
    double witness_upper_margin = 0.0;
    motion_match_clearance_work_diagnostic selected_work;
    motion_match_clearance_work_diagnostic total_work;
};

struct motion_match_ik_diagnostic
{
    bool applied = false;
    bool safe_stop_requested = false;
    const char* stop_reason = "none";
    float max_correction = 0.0f;
    float actual_simulation_speed = 0.0f;
    bool candidate_rejected = false;
    const char* candidate_clearance_status = "invalid-input";
    double candidate_toe_clearance[2] = {};
    double candidate_foot_clearance[2] = {};
    double candidate_minimum_clearance = 0.0;
    bool recorded_contact[2] = {};
    bool locked[2] = {};
    float observed_lock_drift[2] = {};
    float lock_drift[2] = {};
    float sole_normal_alignment[2] = {};
    float contact_residual[2] = {};
    float target_height[2] = {};
    vec3 target_normal[2] = {};
    motion_match_swing_diagnostic swing[2];
    bool reachable[2] = {};
    double knee_clearance[2] = {};
    double ankle_clearance[2] = {};
    double toe_clearance[2] = {};
    double foot_clearance[2] = {};
    double shin_clearance[2] = {};
    double thigh_clearance[2] = {};
    double hips_clearance = 0.0;
    double minimum_clearance = 0.0;
};

struct motion_match_landing_diagnostic
{
    bool expected = false;
    bool patch_ready = false;
    uint32_t sample = UINT32_MAX;
    const char* surface_status = "invalid";
    int walkability_class = 0;
    vec3 center;
    float height = 0.0f;
    vec3 normal;
    double patch_maximum_residual = 0.0;
};

struct motion_match_rejected_foot_diagnostic
{
    motion_match_landing_diagnostic landing;
    vec3 target;
    vec3 target_normal;
    bool reachable = false;
    bool correction_limited = false;
    const char* selected_clearance_status = "invalid-input";
    double selected_lower_margin = 0.0;
    double selected_witness_upper = 0.0;
};

struct motion_match_directional_diagnostic
{
    vec3 requested_velocity;
    vec3 applied_velocity;
    char desired_heading_bits_hex[4 * 8 + 1] =
        "00000000000000000000000000000000";
    char predicted_heading_bits_hex[4 * 4 * 8 + 1] =
        "00000000000000000000000000000000"
        "00000000000000000000000000000000"
        "00000000000000000000000000000000"
        "00000000000000000000000000000000";
    double simulation_heading_error_deg = 0.0;
    double rendered_heading_error_deg = 0.0;
    const char* footprint_status = "invalid-input";
    bool footprint_blocked = false;
    const char* footprint_blocked_reason = "clear";
    float footprint_root_height = 0.0f;
    float footprint_min_height[2] = {};
    float footprint_max_height[2] = {};
    double maximum_root_split[2] = {};
    bool footprint_multilevel[2] = {};
    motion_match_landing_diagnostic landing[2];
    uint32_t footprint_sweeps = 0;
    uint32_t footprint_surface_queries = 0;
    uint32_t footprint_node_visits = 0;
    bool frame_rejected = false;
    const char* frame_rejection_stage = "none";
    bool ik_safe_stop_latched = false;
    bool rejected_attempted_footprint_available = false;
    bool rejected_attempted_ik_available = false;
    const char* rejected_stop_reason = "none";
    bool rejected_attempted_pose_available = false;
    const char* rejected_pose_status = "invalid-input";
    double rejected_pose_minimum_clearance = 0.0;
    motion_match_rejected_foot_diagnostic rejected[2];
    char accepted_state_digest_hex[16 + 1] = "0000000000000000";
};

struct motion_match_log_row
{
    int frame = 0;
    float fixed_dt = 0.04f;
    const char* scene_id = "grail-curb-default";
    const char* mode = "live";
    const char* route = "manual";
    const char* query_bits_hex = "";
    char query_bits_storage[31 * 8 + 1] = {};
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
    motion_match_ik_diagnostic ik;
    motion_match_directional_diagnostic directional;
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
        bool header_ok = fprintf(file,
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
            "model_load_count,model_unload_count,live_model_count") >= 0;
        if (header_ok) header_ok = fputs(
            ",ik_applied,ik_safe_stop_requested,ik_stop_reason,"
            "max_ik_correction,actual_simulation_speed,ik_candidate_rejected,"
            "ik_candidate_clearance_status,left_candidate_toe_clearance,"
            "left_candidate_foot_clearance,right_candidate_toe_clearance,"
            "right_candidate_foot_clearance,ik_candidate_minimum_clearance,"
            "left_recorded_contact,right_recorded_contact,left_locked,"
            "right_locked,left_observed_lock_drift,right_observed_lock_drift,"
            "left_lock_drift,right_lock_drift,left_sole_normal_alignment,"
            "right_sole_normal_alignment,left_contact_residual,"
            "right_contact_residual,left_target_height,right_target_height,"
            "left_target_normal_x,left_target_normal_y,left_target_normal_z,"
            "right_target_normal_x,right_target_normal_y,"
            "right_target_normal_z,"
            "left_swing_candidates_evaluated,left_swing_selected_index,"
            "left_swing_selected_lift_bits,"
            "left_swing_materialized_command_y_bits,"
            "left_swing_actual_sphere_center_bits_hex,"
            "left_swing_selected_clearance_status,"
            "left_swing_selected_controller_constraints_passed,"
            "left_swing_selected_clearance_certified,"
            "left_swing_lower_margin,left_swing_witness_upper_margin,"
            "left_swing_selected_work_point_queries,"
            "left_swing_selected_work_cells_visited,"
            "left_swing_selected_work_primitive_triangle_pairs,"
            "left_swing_selected_work_face_patches,"
            "left_swing_selected_work_candidate_tests,"
            "left_swing_selected_work_subdivision_nodes,"
            "left_swing_total_work_point_queries,"
            "left_swing_total_work_cells_visited,"
            "left_swing_total_work_primitive_triangle_pairs,"
            "left_swing_total_work_face_patches,"
            "left_swing_total_work_candidate_tests,"
            "left_swing_total_work_subdivision_nodes,"
            "right_swing_candidates_evaluated,right_swing_selected_index,"
            "right_swing_selected_lift_bits,"
            "right_swing_materialized_command_y_bits,"
            "right_swing_actual_sphere_center_bits_hex,"
            "right_swing_selected_clearance_status,"
            "right_swing_selected_controller_constraints_passed,"
            "right_swing_selected_clearance_certified,"
            "right_swing_lower_margin,right_swing_witness_upper_margin,"
            "right_swing_selected_work_point_queries,"
            "right_swing_selected_work_cells_visited,"
            "right_swing_selected_work_primitive_triangle_pairs,"
            "right_swing_selected_work_face_patches,"
            "right_swing_selected_work_candidate_tests,"
            "right_swing_selected_work_subdivision_nodes,"
            "right_swing_total_work_point_queries,"
            "right_swing_total_work_cells_visited,"
            "right_swing_total_work_primitive_triangle_pairs,"
            "right_swing_total_work_face_patches,"
            "right_swing_total_work_candidate_tests,"
            "right_swing_total_work_subdivision_nodes,"
            "left_reachable,right_reachable,left_knee_clearance,"
            "left_ankle_clearance,left_toe_clearance,left_foot_clearance,"
            "left_shin_clearance,left_thigh_clearance,right_knee_clearance,"
            "right_ankle_clearance,right_toe_clearance,right_foot_clearance,"
            "right_shin_clearance,right_thigh_clearance,ik_hips_clearance,"
            "ik_minimum_clearance", file) >= 0;
        if (header_ok) header_ok = fputs(
            ",requested_velocity_x,requested_velocity_y,"
            "requested_velocity_z,applied_velocity_x,applied_velocity_y,"
            "applied_velocity_z,desired_heading_bits_hex,"
            "predicted_heading_bits_hex,simulation_heading_error_deg,"
            "rendered_heading_error_deg,footprint_status,footprint_blocked,"
            "footprint_blocked_reason,footprint_root_height,"
            "left_footprint_min_height,left_footprint_max_height,"
            "right_footprint_min_height,right_footprint_max_height,"
            "left_maximum_root_split,right_maximum_root_split,"
            "left_footprint_multilevel,right_footprint_multilevel,"
            "left_landing_expected,left_landing_patch_ready,"
            "left_landing_sample,left_landing_surface_status,"
            "left_landing_walkability_class,"
            "left_predicted_landing_center_x,"
            "left_predicted_landing_center_y,"
            "left_predicted_landing_center_z,left_predicted_landing_height,"
            "left_predicted_landing_normal_x,"
            "left_predicted_landing_normal_y,"
            "left_predicted_landing_normal_z,"
            "left_landing_patch_maximum_residual,"
            "right_landing_expected,right_landing_patch_ready,"
            "right_landing_sample,right_landing_surface_status,"
            "right_landing_walkability_class,"
            "right_predicted_landing_center_x,"
            "right_predicted_landing_center_y,"
            "right_predicted_landing_center_z,"
            "right_predicted_landing_height,"
            "right_predicted_landing_normal_x,"
            "right_predicted_landing_normal_y,"
            "right_predicted_landing_normal_z,"
            "right_landing_patch_maximum_residual,"
            "footprint_sweeps,footprint_surface_queries,"
            "footprint_node_visits,frame_rejected,frame_rejection_stage,"
            "ik_safe_stop_latched,rejected_attempted_footprint_available,"
            "rejected_attempted_ik_available,rejected_stop_reason,"
            "rejected_attempted_pose_available,rejected_pose_status,"
            "rejected_pose_minimum_clearance,"
            "rejected_left_landing_expected,"
            "rejected_left_landing_patch_ready,rejected_left_landing_sample,"
            "rejected_left_landing_center_x,"
            "rejected_left_landing_center_y,"
            "rejected_left_landing_center_z,"
            "rejected_left_landing_surface_status,"
            "rejected_left_landing_surface_height,"
            "rejected_left_landing_surface_normal_x,"
            "rejected_left_landing_surface_normal_y,"
            "rejected_left_landing_surface_normal_z,"
            "rejected_left_landing_walkability_class,"
            "rejected_left_landing_patch_maximum_residual,"
            "rejected_left_target_x,rejected_left_target_y,"
            "rejected_left_target_z,rejected_left_target_normal_x,"
            "rejected_left_target_normal_y,rejected_left_target_normal_z,"
            "rejected_left_reachable,rejected_left_correction_limited,"
            "rejected_left_selected_clearance_status,"
            "rejected_left_selected_lower_margin,"
            "rejected_left_selected_witness_upper,"
            "rejected_right_landing_expected,"
            "rejected_right_landing_patch_ready,"
            "rejected_right_landing_sample,"
            "rejected_right_landing_center_x,"
            "rejected_right_landing_center_y,"
            "rejected_right_landing_center_z,"
            "rejected_right_landing_surface_status,"
            "rejected_right_landing_surface_height,"
            "rejected_right_landing_surface_normal_x,"
            "rejected_right_landing_surface_normal_y,"
            "rejected_right_landing_surface_normal_z,"
            "rejected_right_landing_walkability_class,"
            "rejected_right_landing_patch_maximum_residual,"
            "rejected_right_target_x,rejected_right_target_y,"
            "rejected_right_target_z,rejected_right_target_normal_x,"
            "rejected_right_target_normal_y,rejected_right_target_normal_z,"
            "rejected_right_reachable,rejected_right_correction_limited,"
            "rejected_right_selected_clearance_status,"
            "rejected_right_selected_lower_margin,"
            "rejected_right_selected_witness_upper,"
            "accepted_state_digest_hex\n", file) >= 0;
        if (!header_ok || fflush(file) != 0) {
            const int saved_errno = errno;
            fclose(file);
            file = NULL;
            return io_error(error, error_capacity, "initialize", saved_errno);
        }
        return true;
    }

    bool write_work(const motion_match_clearance_work_diagnostic& work)
    {
        return fprintf(
            file, ",%u,%u,%u,%u,%u,%u",
            (unsigned)work.point_queries,
            (unsigned)work.cells_visited,
            (unsigned)work.primitive_triangle_pairs,
            (unsigned)work.face_patches,
            (unsigned)work.candidate_tests,
            (unsigned)work.subdivision_nodes) >= 0;
    }

    bool write_swing(const motion_match_swing_diagnostic& swing)
    {
        bool ok = fprintf(
            file, ",%u,%u,%u,%u,%s,%s,%d,%d,%.17g,%.17g",
            (unsigned)swing.candidates_evaluated,
            (unsigned)swing.selected_index,
            (unsigned)swing.selected_lift_bits,
            (unsigned)swing.materialized_command_y_bits,
            swing.actual_sphere_center_bits_hex,
            swing.selected_clearance_status,
            (int)swing.selected_controller_constraints_passed,
            (int)swing.selected_clearance_certified,
            swing.lower_margin,
            swing.witness_upper_margin) >= 0;
        if (ok) ok = write_work(swing.selected_work);
        if (ok) ok = write_work(swing.total_work);
        return ok;
    }

    bool write_ik(const motion_match_ik_diagnostic& ik)
    {
        bool ok = fprintf(
            file,
            ",%d,%d,%s,%.9g,%.9g,%d,%s,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,"
            "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
            "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g",
            (int)ik.applied,
            (int)ik.safe_stop_requested,
            ik.stop_reason,
            ik.max_correction,
            ik.actual_simulation_speed,
            (int)ik.candidate_rejected,
            ik.candidate_clearance_status,
            ik.candidate_toe_clearance[0],
            ik.candidate_foot_clearance[0],
            ik.candidate_toe_clearance[1],
            ik.candidate_foot_clearance[1],
            ik.candidate_minimum_clearance,
            (int)ik.recorded_contact[0],
            (int)ik.recorded_contact[1],
            (int)ik.locked[0],
            (int)ik.locked[1],
            ik.observed_lock_drift[0],
            ik.observed_lock_drift[1],
            ik.lock_drift[0],
            ik.lock_drift[1],
            ik.sole_normal_alignment[0],
            ik.sole_normal_alignment[1],
            ik.contact_residual[0],
            ik.contact_residual[1],
            ik.target_height[0],
            ik.target_height[1],
            ik.target_normal[0].x,
            ik.target_normal[0].y,
            ik.target_normal[0].z,
            ik.target_normal[1].x,
            ik.target_normal[1].y,
            ik.target_normal[1].z) >= 0;
        if (ok) ok = write_swing(ik.swing[0]);
        if (ok) ok = write_swing(ik.swing[1]);
        if (ok) ok = fprintf(
            file,
            ",%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
            (int)ik.reachable[0],
            (int)ik.reachable[1],
            ik.knee_clearance[0],
            ik.ankle_clearance[0],
            ik.toe_clearance[0],
            ik.foot_clearance[0],
            ik.shin_clearance[0],
            ik.thigh_clearance[0],
            ik.knee_clearance[1],
            ik.ankle_clearance[1],
            ik.toe_clearance[1],
            ik.foot_clearance[1],
            ik.shin_clearance[1],
            ik.thigh_clearance[1],
            ik.hips_clearance,
            ik.minimum_clearance) >= 0;
        return ok;
    }

    bool write_landing(const motion_match_landing_diagnostic& landing)
    {
        return fprintf(
            file,
            ",%d,%d,%u,%s,%d,%.9g,%.9g,%.9g,%.9g,"
            "%.9g,%.9g,%.9g,%.17g",
            (int)landing.expected,
            (int)landing.patch_ready,
            (unsigned)landing.sample,
            landing.surface_status,
            landing.walkability_class,
            landing.center.x,
            landing.center.y,
            landing.center.z,
            landing.height,
            landing.normal.x,
            landing.normal.y,
            landing.normal.z,
            landing.patch_maximum_residual) >= 0;
    }

    bool write_rejected_foot(
        const motion_match_rejected_foot_diagnostic& rejected)
    {
        return fprintf(
            file,
            ",%d,%d,%u,%.9g,%.9g,%.9g,%s,%.9g,%.9g,%.9g,%.9g,%d,"
            "%.17g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d,%s,%.17g,%.17g",
            (int)rejected.landing.expected,
            (int)rejected.landing.patch_ready,
            (unsigned)rejected.landing.sample,
            rejected.landing.center.x,
            rejected.landing.center.y,
            rejected.landing.center.z,
            rejected.landing.surface_status,
            rejected.landing.height,
            rejected.landing.normal.x,
            rejected.landing.normal.y,
            rejected.landing.normal.z,
            rejected.landing.walkability_class,
            rejected.landing.patch_maximum_residual,
            rejected.target.x,
            rejected.target.y,
            rejected.target.z,
            rejected.target_normal.x,
            rejected.target_normal.y,
            rejected.target_normal.z,
            (int)rejected.reachable,
            (int)rejected.correction_limited,
            rejected.selected_clearance_status,
            rejected.selected_lower_margin,
            rejected.selected_witness_upper) >= 0;
    }

    bool write_directional(
        const motion_match_directional_diagnostic& directional)
    {
        bool ok = fprintf(
            file,
            ",%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%s,%s,%.17g,%.17g,"
            "%s,%d,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.17g,%.17g,%d,%d",
            directional.requested_velocity.x,
            directional.requested_velocity.y,
            directional.requested_velocity.z,
            directional.applied_velocity.x,
            directional.applied_velocity.y,
            directional.applied_velocity.z,
            directional.desired_heading_bits_hex,
            directional.predicted_heading_bits_hex,
            directional.simulation_heading_error_deg,
            directional.rendered_heading_error_deg,
            directional.footprint_status,
            (int)directional.footprint_blocked,
            directional.footprint_blocked_reason,
            directional.footprint_root_height,
            directional.footprint_min_height[0],
            directional.footprint_max_height[0],
            directional.footprint_min_height[1],
            directional.footprint_max_height[1],
            directional.maximum_root_split[0],
            directional.maximum_root_split[1],
            (int)directional.footprint_multilevel[0],
            (int)directional.footprint_multilevel[1]) >= 0;
        if (ok) ok = write_landing(directional.landing[0]);
        if (ok) ok = write_landing(directional.landing[1]);
        if (ok) ok = fprintf(
            file, ",%u,%u,%u,%d,%s,%d,%d,%d,%s,%d,%s,%.17g",
            (unsigned)directional.footprint_sweeps,
            (unsigned)directional.footprint_surface_queries,
            (unsigned)directional.footprint_node_visits,
            (int)directional.frame_rejected,
            directional.frame_rejection_stage,
            (int)directional.ik_safe_stop_latched,
            (int)directional.rejected_attempted_footprint_available,
            (int)directional.rejected_attempted_ik_available,
            directional.rejected_stop_reason,
            (int)directional.rejected_attempted_pose_available,
            directional.rejected_pose_status,
            directional.rejected_pose_minimum_clearance) >= 0;
        if (ok) ok = write_rejected_foot(directional.rejected[0]);
        if (ok) ok = write_rejected_foot(directional.rejected[1]);
        if (ok) ok = fprintf(
            file, ",%s\n", directional.accepted_state_digest_hex) >= 0;
        return ok;
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
            "%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d,%.9g,%d,%d,%d,%d,%d,%d,%d,%d",
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
        if (ok) ok = write_ik(r.ik);
        if (ok) ok = write_directional(r.directional);
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
