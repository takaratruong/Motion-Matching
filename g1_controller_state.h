#pragma once

#include "scene_runtime.h"
#include "support_runtime.h"

#include <cfloat>
#include <utility>

struct g1_controller_state
{
    int frame_index = 0;
    int scene_frame = 0;
    float search_time = 0.10f;
    float search_timer = 0.10f;
    float force_search_timer = 0.10f;

    array1d<vec3> curr_bone_positions;
    array1d<vec3> curr_bone_velocities;
    array1d<vec3> trns_bone_positions;
    array1d<vec3> trns_bone_velocities;
    array1d<quat> curr_bone_rotations;
    array1d<quat> trns_bone_rotations;
    array1d<vec3> curr_bone_angular_velocities;
    array1d<vec3> trns_bone_angular_velocities;
    array1d<bool> curr_bone_contacts;
    array1d<bool> trns_bone_contacts;

    array1d<vec3> bone_positions;
    array1d<vec3> bone_velocities;
    array1d<vec3> bone_angular_velocities;
    array1d<quat> bone_rotations;
    array1d<vec3> bone_offset_positions;
    array1d<vec3> bone_offset_velocities;
    array1d<vec3> bone_offset_angular_velocities;
    array1d<quat> bone_offset_rotations;

    array1d<vec3> adjusted_bone_positions;
    array1d<vec3> global_bone_positions;
    array1d<vec3> global_bone_velocities;
    array1d<quat> adjusted_bone_rotations;
    array1d<quat> global_bone_rotations;
    array1d<vec3> global_bone_angular_velocities;
    array1d<bool> global_bone_computed;

    vec3 transition_src_position;
    vec3 transition_dst_position;
    quat transition_src_rotation;
    quat transition_dst_rotation;

    vec3 desired_velocity;
    vec3 desired_velocity_change_curr;
    vec3 desired_velocity_change_prev;
    quat desired_rotation;
    vec3 desired_rotation_change_curr;
    vec3 desired_rotation_change_prev;
    float desired_gait = 0.0f;
    float desired_gait_velocity = 0.0f;

    vec3 simulation_position;
    vec3 simulation_velocity;
    vec3 simulation_acceleration;
    quat simulation_rotation;
    vec3 simulation_angular_velocity;

    array1d<vec3> trajectory_desired_velocities;
    array1d<vec3> trajectory_positions;
    array1d<vec3> trajectory_velocities;
    array1d<vec3> trajectory_accelerations;
    array1d<vec3> trajectory_angular_velocities;
    array1d<quat> trajectory_desired_rotations;
    array1d<quat> trajectory_rotations;

    array1d<int> contact_bones;
    array1d<bool> contact_states;
    array1d<bool> contact_locks;
    array1d<vec3> contact_positions;
    array1d<vec3> contact_velocities;
    array1d<vec3> contact_points;
    array1d<vec3> contact_targets;
    array1d<vec3> contact_offset_positions;
    array1d<vec3> contact_offset_velocities;

    support_frame_state support;
    support_observation support_observation_now;
    float traversal_speed_scale = 1.0f;
    float traversal_speed_scale_velocity = 0.0f;
    bool blocked = false;
    int walkability_class = 1;
    float blocked_distance = FLT_MAX;
    vec3 blocked_point;

    // Open-loop route cursor; route_frames advances only after its row writes.
    int route_index = 0;
    int route_waypoint = 1;
    int route_frames = 0;

    float camera_azimuth = 0.0f;
    float camera_altitude = 0.4f;
    float camera_distance = 4.0f;

    bool searched = false;
    bool transitioned = false;
    float incumbent_cost = 0.0f;
    float selected_cost = 0.0f;
    float selected_terrain_error = 0.0f;
    float adjustment_xz = 0.0f;
    float adjustment_y = 0.0f;
    float clamp_xz = 0.0f;
    float clamp_y = 0.0f;
};

static inline void g1_controller_state_seed_first_frame_desired_velocity(
    g1_controller_state& state)
{
    if (state.scene_frame == 0)
    {
        state.trajectory_desired_velocities.set(state.desired_velocity);
    }
}

template<typename T>
static inline void g1_swap(array1d<T>& first, array1d<T>& second)
{
    std::swap(first.size, second.size);
    std::swap(first.data, second.data);
}

static inline void g1_controller_state_swap(
    g1_controller_state& first, g1_controller_state& second)
{
    using std::swap;

    swap(first.frame_index, second.frame_index);
    swap(first.scene_frame, second.scene_frame);
    swap(first.search_time, second.search_time);
    swap(first.search_timer, second.search_timer);
    swap(first.force_search_timer, second.force_search_timer);

#define G1_SWAP_ARRAY(name) g1_swap(first.name, second.name)
    G1_SWAP_ARRAY(curr_bone_positions);
    G1_SWAP_ARRAY(curr_bone_velocities);
    G1_SWAP_ARRAY(trns_bone_positions);
    G1_SWAP_ARRAY(trns_bone_velocities);
    G1_SWAP_ARRAY(curr_bone_rotations);
    G1_SWAP_ARRAY(trns_bone_rotations);
    G1_SWAP_ARRAY(curr_bone_angular_velocities);
    G1_SWAP_ARRAY(trns_bone_angular_velocities);
    G1_SWAP_ARRAY(curr_bone_contacts);
    G1_SWAP_ARRAY(trns_bone_contacts);
    G1_SWAP_ARRAY(bone_positions);
    G1_SWAP_ARRAY(bone_velocities);
    G1_SWAP_ARRAY(bone_angular_velocities);
    G1_SWAP_ARRAY(bone_rotations);
    G1_SWAP_ARRAY(bone_offset_positions);
    G1_SWAP_ARRAY(bone_offset_velocities);
    G1_SWAP_ARRAY(bone_offset_angular_velocities);
    G1_SWAP_ARRAY(bone_offset_rotations);
    G1_SWAP_ARRAY(adjusted_bone_positions);
    G1_SWAP_ARRAY(global_bone_positions);
    G1_SWAP_ARRAY(global_bone_velocities);
    G1_SWAP_ARRAY(adjusted_bone_rotations);
    G1_SWAP_ARRAY(global_bone_rotations);
    G1_SWAP_ARRAY(global_bone_angular_velocities);
    G1_SWAP_ARRAY(global_bone_computed);
    G1_SWAP_ARRAY(trajectory_desired_velocities);
    G1_SWAP_ARRAY(trajectory_positions);
    G1_SWAP_ARRAY(trajectory_velocities);
    G1_SWAP_ARRAY(trajectory_accelerations);
    G1_SWAP_ARRAY(trajectory_angular_velocities);
    G1_SWAP_ARRAY(trajectory_desired_rotations);
    G1_SWAP_ARRAY(trajectory_rotations);
    G1_SWAP_ARRAY(contact_bones);
    G1_SWAP_ARRAY(contact_states);
    G1_SWAP_ARRAY(contact_locks);
    G1_SWAP_ARRAY(contact_positions);
    G1_SWAP_ARRAY(contact_velocities);
    G1_SWAP_ARRAY(contact_points);
    G1_SWAP_ARRAY(contact_targets);
    G1_SWAP_ARRAY(contact_offset_positions);
    G1_SWAP_ARRAY(contact_offset_velocities);
#undef G1_SWAP_ARRAY

    swap(first.transition_src_position, second.transition_src_position);
    swap(first.transition_dst_position, second.transition_dst_position);
    swap(first.transition_src_rotation, second.transition_src_rotation);
    swap(first.transition_dst_rotation, second.transition_dst_rotation);
    swap(first.desired_velocity, second.desired_velocity);
    swap(first.desired_velocity_change_curr, second.desired_velocity_change_curr);
    swap(first.desired_velocity_change_prev, second.desired_velocity_change_prev);
    swap(first.desired_rotation, second.desired_rotation);
    swap(first.desired_rotation_change_curr, second.desired_rotation_change_curr);
    swap(first.desired_rotation_change_prev, second.desired_rotation_change_prev);
    swap(first.desired_gait, second.desired_gait);
    swap(first.desired_gait_velocity, second.desired_gait_velocity);
    swap(first.simulation_position, second.simulation_position);
    swap(first.simulation_velocity, second.simulation_velocity);
    swap(first.simulation_acceleration, second.simulation_acceleration);
    swap(first.simulation_rotation, second.simulation_rotation);
    swap(first.simulation_angular_velocity, second.simulation_angular_velocity);
    swap(first.support, second.support);
    swap(first.support_observation_now, second.support_observation_now);
    swap(first.traversal_speed_scale, second.traversal_speed_scale);
    swap(first.traversal_speed_scale_velocity,
         second.traversal_speed_scale_velocity);
    swap(first.blocked, second.blocked);
    swap(first.walkability_class, second.walkability_class);
    swap(first.blocked_distance, second.blocked_distance);
    swap(first.blocked_point, second.blocked_point);
    swap(first.route_index, second.route_index);
    swap(first.route_waypoint, second.route_waypoint);
    swap(first.route_frames, second.route_frames);
    swap(first.camera_azimuth, second.camera_azimuth);
    swap(first.camera_altitude, second.camera_altitude);
    swap(first.camera_distance, second.camera_distance);
    swap(first.searched, second.searched);
    swap(first.transitioned, second.transitioned);
    swap(first.incumbent_cost, second.incumbent_cost);
    swap(first.selected_cost, second.selected_cost);
    swap(first.selected_terrain_error, second.selected_terrain_error);
    swap(first.adjustment_xz, second.adjustment_xz);
    swap(first.adjustment_y, second.adjustment_y);
    swap(first.clamp_xz, second.clamp_xz);
    swap(first.clamp_y, second.clamp_y);
}

static inline bool g1_controller_state_reset(
    g1_controller_state& out,
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene,
    char* error,
    const int capacity)
{
    if (db.nframes() <= 0 || db.nbones() != G1_BoneCount ||
        db.nranges() <= 0 || support.values.rows != db.nframes() ||
        support.values.cols != 3)
    {
        return scene_error(
            error, capacity, "controller reset: invalid database/support shapes");
    }

    const vec3 spawn = scene.metadata.spawn_position;
    if (!terrain_float_is_finite(spawn.x) ||
        !terrain_float_is_finite(spawn.y) ||
        !terrain_float_is_finite(spawn.z) ||
        !terrain_float_is_finite(scene.metadata.spawn_yaw) ||
        !scene_inside(scene.metadata.playable_bounds, spawn.x, spawn.z))
    {
        return scene_error(
            error,
            capacity,
            "controller reset: scene '%s' spawn is invalid",
            scene.metadata.id.c_str());
    }

    g1_controller_state candidate;
    candidate.frame_index = db.range_starts(0);
    candidate.curr_bone_positions = db.bone_positions(candidate.frame_index);
    candidate.curr_bone_velocities = db.bone_velocities(candidate.frame_index);
    candidate.curr_bone_rotations = db.bone_rotations(candidate.frame_index);
    candidate.curr_bone_angular_velocities =
        db.bone_angular_velocities(candidate.frame_index);
    candidate.curr_bone_contacts = db.contact_states(candidate.frame_index);

    candidate.trns_bone_positions = candidate.curr_bone_positions;
    candidate.trns_bone_velocities = candidate.curr_bone_velocities;
    candidate.trns_bone_rotations = candidate.curr_bone_rotations;
    candidate.trns_bone_angular_velocities =
        candidate.curr_bone_angular_velocities;
    candidate.trns_bone_contacts = candidate.curr_bone_contacts;

    candidate.bone_positions = candidate.curr_bone_positions;
    candidate.bone_velocities = candidate.curr_bone_velocities;
    candidate.bone_rotations = candidate.curr_bone_rotations;
    candidate.bone_angular_velocities =
        candidate.curr_bone_angular_velocities;

    const int bones = db.nbones();
    candidate.bone_offset_positions.resize(bones);
    candidate.bone_offset_positions.set(vec3());
    candidate.bone_offset_velocities.resize(bones);
    candidate.bone_offset_velocities.set(vec3());
    candidate.bone_offset_rotations.resize(bones);
    candidate.bone_offset_rotations.set(quat());
    candidate.bone_offset_angular_velocities.resize(bones);
    candidate.bone_offset_angular_velocities.set(vec3());

    candidate.transition_src_position = candidate.curr_bone_positions(0);
    candidate.transition_src_rotation = candidate.curr_bone_rotations(0);
    candidate.transition_dst_position = vec3(spawn.x, 0.0f, spawn.z);
    candidate.transition_dst_rotation = quat_from_angle_axis(
        scene.metadata.spawn_yaw, vec3(0.0f, 1.0f, 0.0f));
    candidate.bone_positions(0) = candidate.transition_dst_position;
    candidate.bone_rotations(0) = candidate.transition_dst_rotation;
    candidate.bone_velocities(0) = vec3();
    candidate.bone_angular_velocities(0) = vec3();

    candidate.adjusted_bone_positions = candidate.bone_positions;
    candidate.adjusted_bone_rotations = candidate.bone_rotations;
    candidate.global_bone_positions.resize(bones);
    candidate.global_bone_positions.set(vec3());
    candidate.global_bone_velocities.resize(bones);
    candidate.global_bone_velocities.set(vec3());
    candidate.global_bone_rotations.resize(bones);
    candidate.global_bone_rotations.set(quat());
    candidate.global_bone_angular_velocities.resize(bones);
    candidate.global_bone_angular_velocities.set(vec3());
    candidate.global_bone_computed.resize(bones);
    candidate.global_bone_computed.zero();

    candidate.simulation_position = vec3(spawn.x, 0.0f, spawn.z);
    candidate.simulation_rotation = candidate.transition_dst_rotation;
    candidate.desired_rotation = candidate.simulation_rotation;

    candidate.trajectory_desired_velocities.resize(4);
    candidate.trajectory_desired_velocities.set(vec3());
    candidate.trajectory_positions.resize(4);
    candidate.trajectory_positions.set(candidate.simulation_position);
    candidate.trajectory_velocities.resize(4);
    candidate.trajectory_velocities.set(vec3());
    candidate.trajectory_accelerations.resize(4);
    candidate.trajectory_accelerations.set(vec3());
    candidate.trajectory_angular_velocities.resize(4);
    candidate.trajectory_angular_velocities.set(vec3());
    candidate.trajectory_desired_rotations.resize(4);
    candidate.trajectory_desired_rotations.set(candidate.simulation_rotation);
    candidate.trajectory_rotations.resize(4);
    candidate.trajectory_rotations.set(candidate.simulation_rotation);

    candidate.contact_bones.resize(2);
    candidate.contact_bones(0) = G1_LeftToe;
    candidate.contact_bones(1) = G1_RightToe;
    candidate.contact_states.resize(2);
    candidate.contact_states.zero();
    candidate.contact_locks.resize(2);
    candidate.contact_locks.zero();
    candidate.contact_positions.resize(2);
    candidate.contact_positions.set(vec3());
    candidate.contact_velocities.resize(2);
    candidate.contact_velocities.set(vec3());
    candidate.contact_points.resize(2);
    candidate.contact_points.set(vec3());
    candidate.contact_targets.resize(2);
    candidate.contact_targets.set(vec3());
    candidate.contact_offset_positions.resize(2);
    candidate.contact_offset_positions.set(vec3());
    candidate.contact_offset_velocities.resize(2);
    candidate.contact_offset_velocities.set(vec3());

    const float runtime_height = heightfield_sample_v2(
        scene.terrain, spawn.x, spawn.z);
    const float initial_support =
        runtime_height - support.values(candidate.frame_index, 0);
    if (!terrain_float_is_finite(initial_support))
    {
        return scene_error(
            error,
            capacity,
            "controller reset: non-finite initial support for scene '%s'",
            scene.metadata.id.c_str());
    }
    support_frame_reset(candidate.support, initial_support);
    support_pose_apply(
        candidate.adjusted_bone_positions,
        candidate.bone_positions,
        candidate.support.height);
    candidate.search_timer = candidate.search_time;
    candidate.force_search_timer = candidate.search_time;
    candidate.camera_azimuth = scene.metadata.spawn_yaw;
    candidate.blocked_distance = FLT_MAX;

    g1_controller_state_swap(out, candidate);
    return true;
}
