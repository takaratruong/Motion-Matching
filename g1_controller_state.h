#pragma once

#include "scene_runtime.h"
#include "g1_command_runtime.h"
#include "g1_ik_runtime.h"
#include "support_runtime.h"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

struct g1_controller_state
{
    g1_controller_state() = default;
    g1_controller_state(const g1_controller_state&) = delete;
    g1_controller_state& operator=(const g1_controller_state&) = delete;
    g1_controller_state(g1_controller_state&&) = delete;
    g1_controller_state& operator=(g1_controller_state&&) = delete;

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

    G1FootprintStatus footprint_status = G1FootprintInvalidInput;
    G1FootprintObservation footprint;
    G1IkState ik;
    G1IkFrameResult ik_frame;
    array1d<vec3> ik_bone_positions;
    array1d<quat> ik_bone_rotations;
    array1d<vec3> ik_global_bone_positions;
    array1d<quat> ik_global_bone_rotations;
    array1d<vec3> ik_candidate_bone_positions;
    array1d<quat> ik_candidate_bone_rotations;
    array1d<vec3> ik_candidate_global_bone_positions;
    array1d<quat> ik_candidate_global_bone_rotations;
    G1PoseClearance ik_clearance;
    G1PoseClearance ik_candidate_clearance;
    G1ClearanceStatus ik_candidate_clearance_status =
        G1ClearanceInvalidInput;
    bool ik_candidate_rejected = false;

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
    G1CommandSnapshot command;

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

static inline float g1_idle_match_transition_cost(
    const float command_speed,
    const float planar_simulation_speed)
{
    return command_speed >= 0.0f &&
           planar_simulation_speed >= 0.0f &&
           command_speed <= 1.0e-4f &&
           planar_simulation_speed <= 0.05f
        ? 1.0f
        : 0.0f;
}

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
    G1_SWAP_ARRAY(ik_bone_positions);
    G1_SWAP_ARRAY(ik_bone_rotations);
    G1_SWAP_ARRAY(ik_global_bone_positions);
    G1_SWAP_ARRAY(ik_global_bone_rotations);
    G1_SWAP_ARRAY(ik_candidate_bone_positions);
    G1_SWAP_ARRAY(ik_candidate_bone_rotations);
    G1_SWAP_ARRAY(ik_candidate_global_bone_positions);
    G1_SWAP_ARRAY(ik_candidate_global_bone_rotations);
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
    swap(first.command, second.command);
    swap(first.footprint_status, second.footprint_status);
    swap(first.footprint, second.footprint);
    swap(first.ik, second.ik);
    swap(first.ik_frame, second.ik_frame);
    swap(first.ik_clearance, second.ik_clearance);
    swap(first.ik_candidate_clearance, second.ik_candidate_clearance);
    swap(first.ik_candidate_clearance_status,
         second.ik_candidate_clearance_status);
    swap(first.ik_candidate_rejected, second.ik_candidate_rejected);
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

struct g1_controller_state_memory_range
{
    const void* data = NULL;
    std::size_t bytes = 0;
};

static inline bool g1_controller_state_ranges_overlap(
    const g1_controller_state_memory_range& first,
    const g1_controller_state_memory_range& second)
{
    return g1_ik_memory_ranges_overlap(
        first.data, first.bytes, second.data, second.bytes);
}

template<class T>
static inline bool g1_controller_state_append_range(
    g1_controller_state_memory_range* ranges,
    int& count,
    int capacity,
    const array1d<T>& values,
    int expected_size)
{
    if (ranges == NULL || count < 0 || count >= capacity ||
        expected_size <= 0 || values.size != expected_size ||
        values.data == NULL ||
        static_cast<std::size_t>(expected_size) >
            std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return false;
    }
    ranges[count].data = values.data;
    ranges[count].bytes =
        static_cast<std::size_t>(expected_size) * sizeof(T);
    ++count;
    return true;
}

static inline bool g1_controller_state_storage_ranges(
    const g1_controller_state& state,
    g1_controller_state_memory_range* ranges,
    int& count,
    int capacity)
{
    count = 0;
#define G1_STATE_RANGE(name, size) \
    if (!g1_controller_state_append_range( \
            ranges, count, capacity, state.name, size)) return false
    G1_STATE_RANGE(curr_bone_positions, G1_BoneCount);
    G1_STATE_RANGE(curr_bone_velocities, G1_BoneCount);
    G1_STATE_RANGE(trns_bone_positions, G1_BoneCount);
    G1_STATE_RANGE(trns_bone_velocities, G1_BoneCount);
    G1_STATE_RANGE(curr_bone_rotations, G1_BoneCount);
    G1_STATE_RANGE(trns_bone_rotations, G1_BoneCount);
    G1_STATE_RANGE(curr_bone_angular_velocities, G1_BoneCount);
    G1_STATE_RANGE(trns_bone_angular_velocities, G1_BoneCount);
    G1_STATE_RANGE(curr_bone_contacts, 2);
    G1_STATE_RANGE(trns_bone_contacts, 2);
    G1_STATE_RANGE(bone_positions, G1_BoneCount);
    G1_STATE_RANGE(bone_velocities, G1_BoneCount);
    G1_STATE_RANGE(bone_angular_velocities, G1_BoneCount);
    G1_STATE_RANGE(bone_rotations, G1_BoneCount);
    G1_STATE_RANGE(bone_offset_positions, G1_BoneCount);
    G1_STATE_RANGE(bone_offset_velocities, G1_BoneCount);
    G1_STATE_RANGE(bone_offset_angular_velocities, G1_BoneCount);
    G1_STATE_RANGE(bone_offset_rotations, G1_BoneCount);
    G1_STATE_RANGE(adjusted_bone_positions, G1_BoneCount);
    G1_STATE_RANGE(global_bone_positions, G1_BoneCount);
    G1_STATE_RANGE(global_bone_velocities, G1_BoneCount);
    G1_STATE_RANGE(adjusted_bone_rotations, G1_BoneCount);
    G1_STATE_RANGE(global_bone_rotations, G1_BoneCount);
    G1_STATE_RANGE(global_bone_angular_velocities, G1_BoneCount);
    G1_STATE_RANGE(global_bone_computed, G1_BoneCount);
    G1_STATE_RANGE(trajectory_desired_velocities,
                   G1CommandTrajectorySampleCount);
    G1_STATE_RANGE(trajectory_positions, G1CommandTrajectorySampleCount);
    G1_STATE_RANGE(trajectory_velocities, G1CommandTrajectorySampleCount);
    G1_STATE_RANGE(trajectory_accelerations, G1CommandTrajectorySampleCount);
    G1_STATE_RANGE(trajectory_angular_velocities,
                   G1CommandTrajectorySampleCount);
    G1_STATE_RANGE(trajectory_desired_rotations,
                   G1CommandTrajectorySampleCount);
    G1_STATE_RANGE(trajectory_rotations, G1CommandTrajectorySampleCount);
    G1_STATE_RANGE(contact_bones, 2);
    G1_STATE_RANGE(contact_states, 2);
    G1_STATE_RANGE(contact_locks, 2);
    G1_STATE_RANGE(contact_positions, 2);
    G1_STATE_RANGE(contact_velocities, 2);
    G1_STATE_RANGE(contact_points, 2);
    G1_STATE_RANGE(contact_targets, 2);
    G1_STATE_RANGE(contact_offset_positions, 2);
    G1_STATE_RANGE(contact_offset_velocities, 2);
    G1_STATE_RANGE(ik_bone_positions, G1_BoneCount);
    G1_STATE_RANGE(ik_bone_rotations, G1_BoneCount);
    G1_STATE_RANGE(ik_global_bone_positions, G1_BoneCount);
    G1_STATE_RANGE(ik_global_bone_rotations, G1_BoneCount);
    G1_STATE_RANGE(ik_candidate_bone_positions, G1_BoneCount);
    G1_STATE_RANGE(ik_candidate_bone_rotations, G1_BoneCount);
    G1_STATE_RANGE(ik_candidate_global_bone_positions, G1_BoneCount);
    G1_STATE_RANGE(ik_candidate_global_bone_rotations, G1_BoneCount);
#undef G1_STATE_RANGE
    const g1_controller_state_memory_range object_range = {
        &state, sizeof(state)
    };
    for (int index = 0; index < count; ++index) {
        if (g1_controller_state_ranges_overlap(
                object_range, ranges[index])) {
            return false;
        }
    }
    return true;
}

static inline bool g1_controller_state_ranges_are_disjoint(
    const g1_controller_state_memory_range* ranges,
    int count)
{
    if (ranges == NULL || count <= 0) return false;
    for (int first = 0; first < count; ++first) {
        for (int second = first + 1; second < count; ++second) {
            if (g1_controller_state_ranges_overlap(
                    ranges[first], ranges[second])) {
                return false;
            }
        }
    }
    return true;
}

static inline bool g1_controller_state_storage_is_valid(
    const g1_controller_state& state)
{
    g1_controller_state_memory_range ranges[64] = {};
    int count = 0;
    return g1_controller_state_storage_ranges(
               state, ranges, count,
               static_cast<int>(sizeof(ranges) / sizeof(ranges[0]))) &&
           g1_controller_state_ranges_are_disjoint(ranges, count);
}

template<class T>
static inline bool g1_controller_state_append_flexible_owner_range(
    g1_controller_state_memory_range* ranges,
    int& count,
    int capacity,
    const array1d<T>& values)
{
    if (ranges == NULL || count < 0 || count > capacity ||
        values.size < 0) {
        return false;
    }
    if (values.size == 0) return values.data == NULL;
    if (values.data == NULL || count == capacity ||
        static_cast<std::size_t>(values.size) >
            std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return false;
    }
    ranges[count] = {
        values.data,
        static_cast<std::size_t>(values.size) * sizeof(T)
    };
    ++count;
    return true;
}

static inline bool g1_controller_state_reset_output_ranges(
    const g1_controller_state& state,
    g1_controller_state_memory_range* ranges,
    int& count,
    int capacity)
{
    count = 0;
#define G1_RESET_OUTPUT_RANGE(name) \
    if (!g1_controller_state_append_flexible_owner_range( \
            ranges, count, capacity, state.name)) return false
    G1_RESET_OUTPUT_RANGE(curr_bone_positions);
    G1_RESET_OUTPUT_RANGE(curr_bone_velocities);
    G1_RESET_OUTPUT_RANGE(trns_bone_positions);
    G1_RESET_OUTPUT_RANGE(trns_bone_velocities);
    G1_RESET_OUTPUT_RANGE(curr_bone_rotations);
    G1_RESET_OUTPUT_RANGE(trns_bone_rotations);
    G1_RESET_OUTPUT_RANGE(curr_bone_angular_velocities);
    G1_RESET_OUTPUT_RANGE(trns_bone_angular_velocities);
    G1_RESET_OUTPUT_RANGE(curr_bone_contacts);
    G1_RESET_OUTPUT_RANGE(trns_bone_contacts);
    G1_RESET_OUTPUT_RANGE(bone_positions);
    G1_RESET_OUTPUT_RANGE(bone_velocities);
    G1_RESET_OUTPUT_RANGE(bone_angular_velocities);
    G1_RESET_OUTPUT_RANGE(bone_rotations);
    G1_RESET_OUTPUT_RANGE(bone_offset_positions);
    G1_RESET_OUTPUT_RANGE(bone_offset_velocities);
    G1_RESET_OUTPUT_RANGE(bone_offset_angular_velocities);
    G1_RESET_OUTPUT_RANGE(bone_offset_rotations);
    G1_RESET_OUTPUT_RANGE(adjusted_bone_positions);
    G1_RESET_OUTPUT_RANGE(global_bone_positions);
    G1_RESET_OUTPUT_RANGE(global_bone_velocities);
    G1_RESET_OUTPUT_RANGE(adjusted_bone_rotations);
    G1_RESET_OUTPUT_RANGE(global_bone_rotations);
    G1_RESET_OUTPUT_RANGE(global_bone_angular_velocities);
    G1_RESET_OUTPUT_RANGE(global_bone_computed);
    G1_RESET_OUTPUT_RANGE(ik_bone_positions);
    G1_RESET_OUTPUT_RANGE(ik_bone_rotations);
    G1_RESET_OUTPUT_RANGE(ik_global_bone_positions);
    G1_RESET_OUTPUT_RANGE(ik_global_bone_rotations);
    G1_RESET_OUTPUT_RANGE(ik_candidate_bone_positions);
    G1_RESET_OUTPUT_RANGE(ik_candidate_bone_rotations);
    G1_RESET_OUTPUT_RANGE(ik_candidate_global_bone_positions);
    G1_RESET_OUTPUT_RANGE(ik_candidate_global_bone_rotations);
    G1_RESET_OUTPUT_RANGE(trajectory_desired_velocities);
    G1_RESET_OUTPUT_RANGE(trajectory_positions);
    G1_RESET_OUTPUT_RANGE(trajectory_velocities);
    G1_RESET_OUTPUT_RANGE(trajectory_accelerations);
    G1_RESET_OUTPUT_RANGE(trajectory_angular_velocities);
    G1_RESET_OUTPUT_RANGE(trajectory_desired_rotations);
    G1_RESET_OUTPUT_RANGE(trajectory_rotations);
    G1_RESET_OUTPUT_RANGE(contact_bones);
    G1_RESET_OUTPUT_RANGE(contact_states);
    G1_RESET_OUTPUT_RANGE(contact_locks);
    G1_RESET_OUTPUT_RANGE(contact_positions);
    G1_RESET_OUTPUT_RANGE(contact_velocities);
    G1_RESET_OUTPUT_RANGE(contact_points);
    G1_RESET_OUTPUT_RANGE(contact_targets);
    G1_RESET_OUTPUT_RANGE(contact_offset_positions);
    G1_RESET_OUTPUT_RANGE(contact_offset_velocities);
#undef G1_RESET_OUTPUT_RANGE
    const g1_controller_state_memory_range object = {
        &state, sizeof(state)
    };
    for (int index = 0; index < count; ++index) {
        if (g1_controller_state_ranges_overlap(object, ranges[index])) {
            return false;
        }
    }
    return count == 0 ||
           g1_controller_state_ranges_are_disjoint(ranges, count);
}

template<class T>
static inline bool g1_controller_state_source_owner_overlaps(
    const g1_controller_state_memory_range& test,
    const array1d<T>& values)
{
    if (values.size == 0) return values.data != NULL;
    if (values.size < 0 || values.data == NULL ||
        static_cast<std::size_t>(values.size) >
            std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return true;
    }
    const g1_controller_state_memory_range owner = {
        values.data,
        static_cast<std::size_t>(values.size) * sizeof(T)
    };
    return g1_controller_state_ranges_overlap(test, owner);
}

template<class T>
static inline bool g1_controller_state_source_owner_overlaps(
    const g1_controller_state_memory_range& test,
    const array2d<T>& values)
{
    if (values.rows == 0 && values.cols == 0) {
        return values.data != NULL;
    }
    if (values.rows <= 0 || values.cols <= 0 || values.data == NULL) {
        return true;
    }
    std::size_t elements = 0U;
    if (!terrain_size_multiply(
            static_cast<std::size_t>(values.rows),
            static_cast<std::size_t>(values.cols),
            elements) ||
        elements > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return true;
    }
    const g1_controller_state_memory_range owner = {
        values.data, elements * sizeof(T)
    };
    return g1_controller_state_ranges_overlap(test, owner);
}

static inline bool g1_controller_state_source_owner_overlaps(
    const g1_controller_state_memory_range& test,
    const std::string& value)
{
    if (value.capacity() == std::numeric_limits<std::size_t>::max() ||
        value.data() == NULL) {
        return true;
    }
    const g1_controller_state_memory_range owner = {
        value.data(), value.capacity() + 1U
    };
    return g1_controller_state_ranges_overlap(test, owner);
}

template<class T>
static inline bool g1_controller_state_vector_owner_overlaps(
    const g1_controller_state_memory_range& test,
    const std::vector<T>& values)
{
    if (values.capacity() == 0U) return false;
    if (values.data() == NULL ||
        values.capacity() >
            std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return true;
    }
    const g1_controller_state_memory_range owner = {
        values.data(), values.capacity() * sizeof(T)
    };
    return g1_controller_state_ranges_overlap(test, owner);
}

static inline bool g1_controller_state_string_vector_overlaps(
    const g1_controller_state_memory_range& test,
    const std::vector<std::string>& values)
{
    if (g1_controller_state_vector_owner_overlaps(test, values)) {
        return true;
    }
    for (const std::string& value : values) {
        if (g1_controller_state_source_owner_overlaps(test, value)) {
            return true;
        }
    }
    return false;
}

static inline bool g1_controller_state_artifact_overlaps(
    const g1_controller_state_memory_range& test,
    const artifact_reference& artifact)
{
    return g1_controller_state_source_owner_overlaps(test, artifact.path) ||
           g1_controller_state_source_owner_overlaps(test, artifact.schema) ||
           g1_controller_state_source_owner_overlaps(test, artifact.sha256) ||
           g1_controller_state_string_vector_overlaps(
               test, artifact.columns);
}

static inline bool g1_controller_state_region_vector_overlaps(
    const g1_controller_state_memory_range& test,
    const std::vector<scene_region>& regions)
{
    if (g1_controller_state_vector_owner_overlaps(test, regions)) {
        return true;
    }
    for (const scene_region& region : regions) {
        if (g1_controller_state_source_owner_overlaps(test, region.id)) {
            return true;
        }
    }
    return false;
}

static inline bool g1_controller_state_route_vector_overlaps(
    const g1_controller_state_memory_range& test,
    const std::vector<scene_route>& routes)
{
    if (g1_controller_state_vector_owner_overlaps(test, routes)) {
        return true;
    }
    for (const scene_route& route : routes) {
        if (g1_controller_state_source_owner_overlaps(test, route.id) ||
            g1_controller_state_source_owner_overlaps(
                test, route.expected_outcome) ||
            g1_controller_state_vector_owner_overlaps(
                test, route.waypoints_xz)) {
            return true;
        }
    }
    return false;
}

static inline bool g1_controller_state_scene_dynamic_overlaps(
    const g1_controller_state_memory_range& test,
    const scene_pack& scene)
{
    const scene_metadata& metadata = scene.metadata;
    return g1_controller_state_source_owner_overlaps(test, metadata.id) ||
           g1_controller_state_source_owner_overlaps(test, metadata.label) ||
           g1_controller_state_source_owner_overlaps(
               test, metadata.provenance_kind) ||
           g1_controller_state_string_vector_overlaps(
               test, metadata.provenance_source_ids) ||
           g1_controller_state_source_owner_overlaps(
               test, metadata.coordinate_signature) ||
           g1_controller_state_source_owner_overlaps(
               test, metadata.surface_signature) ||
           g1_controller_state_artifact_overlaps(
               test, metadata.heightfield) ||
           g1_controller_state_artifact_overlaps(test, metadata.mesh) ||
           g1_controller_state_artifact_overlaps(
               test, metadata.walkability) ||
           g1_controller_state_region_vector_overlaps(
               test, metadata.certified_regions) ||
           g1_controller_state_region_vector_overlaps(
               test, metadata.stress_regions) ||
           g1_controller_state_region_vector_overlaps(
               test, metadata.blocked_regions) ||
           g1_controller_state_route_vector_overlaps(
               test, metadata.routes) ||
           g1_controller_state_source_owner_overlaps(
               test, scene.terrain.heights) ||
           g1_controller_state_source_owner_overlaps(
               test, scene.walkability.cells) ||
           g1_controller_state_source_owner_overlaps(
               test, scene.scene_path) ||
           g1_controller_state_source_owner_overlaps(
               test, scene.terrain_path) ||
           g1_controller_state_source_owner_overlaps(
               test, scene.mesh_path) ||
           g1_controller_state_source_owner_overlaps(
               test, scene.walkability_path);
}

static inline bool g1_controller_state_source_storage_overlaps(
    const g1_controller_state_memory_range& test,
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene)
{
    const g1_controller_state_memory_range objects[] = {
        {&db, sizeof(db)},
        {&support, sizeof(support)},
        {&scene, sizeof(scene)}
    };
    for (const g1_controller_state_memory_range& object : objects) {
        if (g1_controller_state_ranges_overlap(test, object)) return true;
    }
#define G1_SOURCE_OWNER(owner) \
    if (g1_controller_state_source_owner_overlaps( \
            test, owner)) return true
    G1_SOURCE_OWNER(db.bone_positions);
    G1_SOURCE_OWNER(db.bone_velocities);
    G1_SOURCE_OWNER(db.bone_rotations);
    G1_SOURCE_OWNER(db.bone_angular_velocities);
    G1_SOURCE_OWNER(db.bone_parents);
    G1_SOURCE_OWNER(db.range_starts);
    G1_SOURCE_OWNER(db.range_stops);
    G1_SOURCE_OWNER(db.features);
    G1_SOURCE_OWNER(db.features_offset);
    G1_SOURCE_OWNER(db.features_scale);
    G1_SOURCE_OWNER(db.terrain_features);
    G1_SOURCE_OWNER(db.contact_states);
    G1_SOURCE_OWNER(db.bound_sm_min);
    G1_SOURCE_OWNER(db.bound_sm_max);
    G1_SOURCE_OWNER(db.bound_lr_min);
    G1_SOURCE_OWNER(db.bound_lr_max);
    G1_SOURCE_OWNER(support.values);
#undef G1_SOURCE_OWNER
    return g1_controller_state_scene_dynamic_overlaps(test, scene);
}

static inline bool
g1_controller_state_reset_output_is_disjoint_from_sources(
    const g1_controller_state& output,
    const g1_controller_state_memory_range* output_ranges,
    int output_count,
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene)
{
    if (output_ranges == NULL || output_count < 0) return false;
    const g1_controller_state_memory_range output_object = {
        &output, sizeof(output)
    };
    if (g1_controller_state_source_storage_overlaps(
            output_object, db, support, scene)) {
        return false;
    }
    for (int index = 0; index < output_count; ++index) {
        if (g1_controller_state_source_storage_overlaps(
                output_ranges[index], db, support, scene)) {
            return false;
        }
    }
    return true;
}

template<class T>
static inline bool g1_controller_state_diagnostic_aliases_array(
    const array1d<T>& values,
    const g1_controller_state_memory_range& diagnostic)
{
    if (values.size < 0 ||
        static_cast<std::size_t>(values.size) >
            std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return true;
    }
    if (values.size == 0 || values.data == NULL) return false;
    const g1_controller_state_memory_range memory = {
        values.data,
        static_cast<std::size_t>(values.size) * sizeof(T)
    };
    return g1_controller_state_ranges_overlap(diagnostic, memory);
}

static inline bool g1_controller_state_diagnostic_aliases_state(
    const g1_controller_state& state,
    char* error,
    int error_capacity)
{
    if (error_capacity < 0) return true;
    if (error == NULL || error_capacity == 0) return false;
    const g1_controller_state_memory_range diagnostic = {
        error, static_cast<std::size_t>(error_capacity)
    };
    const g1_controller_state_memory_range object = {
        &state, sizeof(state)
    };
    if (g1_controller_state_ranges_overlap(diagnostic, object)) {
        return true;
    }
#define G1_DIAGNOSTIC_ARRAY(name) \
    if (g1_controller_state_diagnostic_aliases_array( \
            state.name, diagnostic)) return true
    G1_DIAGNOSTIC_ARRAY(curr_bone_positions);
    G1_DIAGNOSTIC_ARRAY(curr_bone_velocities);
    G1_DIAGNOSTIC_ARRAY(trns_bone_positions);
    G1_DIAGNOSTIC_ARRAY(trns_bone_velocities);
    G1_DIAGNOSTIC_ARRAY(curr_bone_rotations);
    G1_DIAGNOSTIC_ARRAY(trns_bone_rotations);
    G1_DIAGNOSTIC_ARRAY(curr_bone_angular_velocities);
    G1_DIAGNOSTIC_ARRAY(trns_bone_angular_velocities);
    G1_DIAGNOSTIC_ARRAY(curr_bone_contacts);
    G1_DIAGNOSTIC_ARRAY(trns_bone_contacts);
    G1_DIAGNOSTIC_ARRAY(bone_positions);
    G1_DIAGNOSTIC_ARRAY(bone_velocities);
    G1_DIAGNOSTIC_ARRAY(bone_angular_velocities);
    G1_DIAGNOSTIC_ARRAY(bone_rotations);
    G1_DIAGNOSTIC_ARRAY(bone_offset_positions);
    G1_DIAGNOSTIC_ARRAY(bone_offset_velocities);
    G1_DIAGNOSTIC_ARRAY(bone_offset_angular_velocities);
    G1_DIAGNOSTIC_ARRAY(bone_offset_rotations);
    G1_DIAGNOSTIC_ARRAY(adjusted_bone_positions);
    G1_DIAGNOSTIC_ARRAY(global_bone_positions);
    G1_DIAGNOSTIC_ARRAY(global_bone_velocities);
    G1_DIAGNOSTIC_ARRAY(adjusted_bone_rotations);
    G1_DIAGNOSTIC_ARRAY(global_bone_rotations);
    G1_DIAGNOSTIC_ARRAY(global_bone_angular_velocities);
    G1_DIAGNOSTIC_ARRAY(global_bone_computed);
    G1_DIAGNOSTIC_ARRAY(trajectory_desired_velocities);
    G1_DIAGNOSTIC_ARRAY(trajectory_positions);
    G1_DIAGNOSTIC_ARRAY(trajectory_velocities);
    G1_DIAGNOSTIC_ARRAY(trajectory_accelerations);
    G1_DIAGNOSTIC_ARRAY(trajectory_angular_velocities);
    G1_DIAGNOSTIC_ARRAY(trajectory_desired_rotations);
    G1_DIAGNOSTIC_ARRAY(trajectory_rotations);
    G1_DIAGNOSTIC_ARRAY(contact_bones);
    G1_DIAGNOSTIC_ARRAY(contact_states);
    G1_DIAGNOSTIC_ARRAY(contact_locks);
    G1_DIAGNOSTIC_ARRAY(contact_positions);
    G1_DIAGNOSTIC_ARRAY(contact_velocities);
    G1_DIAGNOSTIC_ARRAY(contact_points);
    G1_DIAGNOSTIC_ARRAY(contact_targets);
    G1_DIAGNOSTIC_ARRAY(contact_offset_positions);
    G1_DIAGNOSTIC_ARRAY(contact_offset_velocities);
    G1_DIAGNOSTIC_ARRAY(ik_bone_positions);
    G1_DIAGNOSTIC_ARRAY(ik_bone_rotations);
    G1_DIAGNOSTIC_ARRAY(ik_global_bone_positions);
    G1_DIAGNOSTIC_ARRAY(ik_global_bone_rotations);
    G1_DIAGNOSTIC_ARRAY(ik_candidate_bone_positions);
    G1_DIAGNOSTIC_ARRAY(ik_candidate_bone_rotations);
    G1_DIAGNOSTIC_ARRAY(ik_candidate_global_bone_positions);
    G1_DIAGNOSTIC_ARRAY(ik_candidate_global_bone_rotations);
#undef G1_DIAGNOSTIC_ARRAY
    return false;
}

static inline bool g1_controller_reset_diagnostic_aliases_sources(
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene,
    char* error,
    int error_capacity)
{
    if (error_capacity < 0) return true;
    if (error == NULL || error_capacity == 0) return false;
    const g1_controller_state_memory_range diagnostic = {
        error, static_cast<std::size_t>(error_capacity)
    };
    return g1_controller_state_source_storage_overlaps(
        diagnostic, db, support, scene);
}

static inline bool g1_controller_state_vec3_array_is_finite(
    const array1d<vec3>& values)
{
    for (int index = 0; index < values.size; ++index) {
        if (!g1_ik_vec3_is_runtime_value(values(index))) return false;
    }
    return true;
}

static inline bool g1_controller_state_quat_array_is_valid(
    const array1d<quat>& values)
{
    for (int index = 0; index < values.size; ++index) {
        if (!ik_quat_is_unit(values(index))) return false;
    }
    return true;
}

static inline bool g1_controller_state_clearance_result_is_valid(
    const G1ClearanceResult& value)
{
    const double finite_values[] = {
        value.lower_bound_m,
        value.witness_upper_m,
        value.witness.body_x,
        value.witness.body_y,
        value.witness.body_z,
        value.witness.surface_x,
        value.witness.surface_y,
        value.witness.surface_z,
        value.witness.segment_parameter,
        value.witness.terrain_weight_0,
        value.witness.terrain_weight_1,
        value.witness.terrain_weight_2
    };
    for (double finite_value : finite_values) {
        if (!terrain_double_is_finite(finite_value)) return false;
    }
    const G1ClearanceBudget limits = g1_pose_clearance_budget();
    const double terrain_weight_sum =
        value.witness.terrain_weight_0 +
        value.witness.terrain_weight_1 +
        value.witness.terrain_weight_2;
    bool candidate_subindex_is_valid = false;
    switch (value.witness.candidate_kind) {
    case 0U:
    case 3U:
        candidate_subindex_is_valid =
            value.witness.candidate_subindex == 0U;
        break;
    case 1U:
        candidate_subindex_is_valid =
            value.witness.candidate_subindex <= 8U;
        break;
    case 2U:
        candidate_subindex_is_valid =
            static_cast<uint64_t>(value.witness.candidate_subindex) <
            static_cast<uint64_t>(limits.maximum_face_patches) +
                static_cast<uint64_t>(limits.maximum_subdivision_nodes);
        break;
    default:
        break;
    }
    return value.lower_bound_m <= value.witness_upper_m &&
           value.witness_upper_m <=
               value.lower_bound_m +
                   G1ClearanceMaximumCertificateWidthM &&
           value.witness.segment_parameter >= 0.0 &&
           value.witness.segment_parameter <= 1.0 &&
           value.witness.terrain_weight_0 >= 0.0 &&
           value.witness.terrain_weight_0 <= 1.0 &&
           value.witness.terrain_weight_1 >= 0.0 &&
           value.witness.terrain_weight_1 <= 1.0 &&
           value.witness.terrain_weight_2 >= 0.0 &&
           value.witness.terrain_weight_2 <= 1.0 &&
           terrain_weight_sum - 1.0 >= -1.0e-12 &&
           terrain_weight_sum - 1.0 <= 1.0e-12 &&
           value.witness.cell_x >= 0 && value.witness.cell_z >= 0 &&
           value.witness.terrain_triangle_index <= 1U &&
           value.witness.patch_index < limits.maximum_face_patches &&
           value.witness.candidate_kind <= 3U &&
           candidate_subindex_is_valid &&
           value.witness.primitive_index < 19U &&
           value.work.point_queries <= limits.maximum_point_queries &&
           value.work.cells_visited <= limits.maximum_cells &&
           value.work.primitive_triangle_pairs <=
               limits.maximum_primitive_triangle_pairs &&
           value.work.face_patches <= limits.maximum_face_patches &&
           value.work.candidate_tests <= limits.maximum_candidate_tests &&
           value.work.subdivision_nodes <=
               limits.maximum_subdivision_nodes;
}

static inline bool g1_controller_state_clearance_work_is_zero(
    const G1ClearanceWork& work)
{
    return work.point_queries == 0U && work.cells_visited == 0U &&
           work.primitive_triangle_pairs == 0U &&
           work.face_patches == 0U && work.candidate_tests == 0U &&
           work.subdivision_nodes == 0U;
}

static inline bool g1_controller_state_clearance_work_is_bounded(
    const G1ClearanceWork& work,
    const G1ClearanceBudget& limits)
{
    return work.point_queries <= limits.maximum_point_queries &&
           work.cells_visited <= limits.maximum_cells &&
           work.primitive_triangle_pairs <=
               limits.maximum_primitive_triangle_pairs &&
           work.face_patches <= limits.maximum_face_patches &&
           work.candidate_tests <= limits.maximum_candidate_tests &&
           work.subdivision_nodes <= limits.maximum_subdivision_nodes;
}

static inline bool g1_controller_state_vec3_bits_equal(
    vec3 left,
    vec3 right);
static inline uint64_t g1_controller_state_double_bits(double value);

static inline bool g1_controller_state_leg_result_is_successful(
    const G1LegSolveResult& value)
{
    const int provenance_count =
        (value.bend_used_current_projection ? 1 : 0) +
        (value.bend_used_hinge_fallback ? 1 : 0);
    return g1_ik_leg_result_is_valid(value) &&
           value.reachable &&
           !value.correction_limited && !value.safe_stop_requested &&
           value.iterations >= 1 && value.iterations <= 4 &&
           provenance_count == 1 &&
           (!value.bend_used_safe_perpendicular ||
            value.bend_used_hinge_fallback) &&
           g1_controller_state_vec3_bits_equal(
               value.requested_ankle_target,
               value.clamped_ankle_target) &&
           terrain_float_bits(value.raw_distance_m) ==
               terrain_float_bits(value.clamped_distance_m) &&
           value.max_correction_radians <= 0.35f &&
           g1_ik_contact_residual_is_converged(
               value.contact_residual_m);
}

static inline bool g1_controller_state_orientation_is_successful(
    const G1FootOrientationResult& value)
{
    return value.applied && !value.correction_limited &&
           !value.safe_stop_requested &&
           ik_quat_is_unit(value.target_global_rotation) &&
           g1_ik_float_is_runtime_value(
               value.requested_correction_radians) &&
           value.requested_correction_radians >= 0.0f &&
           g1_ik_float_is_runtime_value(value.correction_radians) &&
           value.correction_radians >= 0.0f &&
           value.correction_radians <= 0.35f &&
           terrain_float_bits(value.requested_correction_radians) ==
               terrain_float_bits(value.correction_radians);
}

static inline bool g1_controller_state_swing_candidate_is_canonical(
    const G1SwingCandidateDiagnostic& value)
{
    if (value.candidate_index != G1SwingNoCandidate ||
        value.lift_bits != 0U ||
        value.materialized_command_y_bits != 0U ||
        value.clearance_status != G1ClearanceInvalidInput ||
        value.controller_constraints_passed ||
        value.clearance_certified ||
        g1_controller_state_double_bits(value.lower_margin_m) != 0U ||
        g1_controller_state_double_bits(
            value.witness_upper_margin_m) != 0U ||
        !g1_controller_state_clearance_work_is_zero(
            value.clearance_work)) {
        return false;
    }
    for (int probe = 0; probe < 4; ++probe) {
        for (int axis = 0; axis < 3; ++axis) {
            if (value.actual_sphere_center_bits[probe][axis] != 0U) {
                return false;
            }
        }
    }
    return true;
}

static inline bool g1_controller_state_swing_candidate_is_selected(
    const G1SwingCandidateDiagnostic& value,
    uint32_t selected_index)
{
    if (selected_index >= G1SwingLiftCandidateCount ||
        value.candidate_index != selected_index ||
        value.lift_bits != G1SwingLiftCandidateBits[selected_index] ||
        value.clearance_status != G1ClearanceOk ||
        !value.controller_constraints_passed ||
        !value.clearance_certified ||
        !terrain_double_is_finite(value.lower_margin_m) ||
        value.lower_margin_m < 0.0 ||
        !terrain_double_is_finite(value.witness_upper_margin_m) ||
        value.witness_upper_margin_m < value.lower_margin_m ||
        value.witness_upper_margin_m >
            value.lower_margin_m +
                G1ClearanceMaximumCertificateWidthM ||
        !g1_controller_state_clearance_work_is_bounded(
            value.clearance_work,
            g1_swing_foot_clearance_budget())) {
        return false;
    }
    float materialized_y = 0.0f;
    std::memcpy(
        &materialized_y,
        &value.materialized_command_y_bits,
        sizeof(materialized_y));
    if (!g1_ik_float_is_runtime_value(materialized_y)) return false;
    for (int probe = 0; probe < 4; ++probe) {
        for (int axis = 0; axis < 3; ++axis) {
            float coordinate = 0.0f;
            std::memcpy(
                &coordinate,
                &value.actual_sphere_center_bits[probe][axis],
                sizeof(coordinate));
            if (!g1_ik_float_is_runtime_value(coordinate)) return false;
        }
    }
    return true;
}

static inline bool g1_controller_state_swing_selection_is_valid(
    const G1SwingSelectionDiagnostic& value,
    bool recorded_contact)
{
    if (recorded_contact) {
        return value.candidates_evaluated == 0U &&
               value.selected_index == G1SwingNoCandidate &&
               g1_controller_state_swing_candidate_is_canonical(
                   value.selected) &&
               g1_controller_state_clearance_work_is_zero(
                   value.total_clearance_work);
    }
    if (value.candidates_evaluated == 0U ||
        value.candidates_evaluated > G1SwingLiftCandidateCount ||
        value.selected_index >= G1SwingLiftCandidateCount ||
        value.candidates_evaluated != value.selected_index + 1U ||
        !g1_controller_state_swing_candidate_is_selected(
            value.selected, value.selected_index)) {
        return false;
    }
    const G1ClearanceWork& selected = value.selected.clearance_work;
    const G1ClearanceWork& total = value.total_clearance_work;
    const G1ClearanceBudget limits =
        g1_swing_foot_clearance_budget();
#define G1_SWING_TOTAL(name, limit_name) \
    if (total.name < selected.name || \
        static_cast<uint64_t>(total.name) > \
            static_cast<uint64_t>(limits.limit_name) * \
                static_cast<uint64_t>(value.candidates_evaluated)) { \
        return false; \
    }
    G1_SWING_TOTAL(point_queries, maximum_point_queries);
    G1_SWING_TOTAL(cells_visited, maximum_cells);
    G1_SWING_TOTAL(
        primitive_triangle_pairs,
        maximum_primitive_triangle_pairs);
    G1_SWING_TOTAL(face_patches, maximum_face_patches);
    G1_SWING_TOTAL(candidate_tests, maximum_candidate_tests);
    G1_SWING_TOTAL(subdivision_nodes, maximum_subdivision_nodes);
#undef G1_SWING_TOTAL
    return true;
}

static inline bool g1_controller_state_defensive_swing_is_valid(
    const G1SwingClearanceValidation& value,
    bool recorded_contact)
{
    if (recorded_contact) {
        return g1_controller_state_double_bits(value.lower_margin_m) == 0U &&
               g1_controller_state_double_bits(value.witness_upper_m) == 0U &&
               !value.sweep_evaluated &&
               g1_controller_state_clearance_work_is_zero(value.work);
    }
    return terrain_double_is_finite(value.lower_margin_m) &&
           value.lower_margin_m >= 0.0 &&
           terrain_double_is_finite(value.witness_upper_m) &&
           value.witness_upper_m >= value.lower_margin_m &&
           value.witness_upper_m <=
               value.lower_margin_m +
                   G1ClearanceMaximumCertificateWidthM &&
           value.sweep_evaluated &&
           g1_controller_state_clearance_work_is_bounded(
               value.work, g1_swing_foot_clearance_budget());
}

static inline bool g1_controller_state_ik_frame_is_valid(
    const G1IkFrameResult& value,
    const slice1d<bool> contacts,
    const G1FootprintObservation& footprint)
{
    if (contacts.size != 2 || contacts.data == NULL) {
        return false;
    }
    if (!value.applied) {
        G1IkFrameTransaction transaction;
        transaction.candidate_result = value;
        return g1_ik_runtime_is_disabled_noop(transaction);
    }
    if (!g1_root_reach_plan_is_valid(value.root_reach) ||
        value.root_reach.active !=
            (contacts(0) || contacts(1)) ||
        (value.root_reach.active &&
         !value.root_reach.common_interval_found) ||
        value.safe_stop_requested || value.stop_reason != G1IkStopNone ||
        !g1_ik_float_is_runtime_value(value.max_correction_radians) ||
        value.max_correction_radians < 0.0f ||
        value.max_correction_radians > 0.35f) {
        return false;
    }
    float maximum_correction = 0.0f;
    for (int foot = 0; foot < 2; ++foot) {
        const G1FootFrameResult& result = value.feet[foot];
        if (contacts.size != 2 || contacts.data == NULL ||
            result.recorded_contact != contacts(foot) ||
            (footprint.feet[foot].landing_expected &&
             !footprint.feet[foot].landing_patch_ready) ||
            !g1_foot_target_is_valid(result.target) ||
            !g1_controller_state_swing_selection_is_valid(
                result.swing_selection, result.recorded_contact) ||
            !g1_controller_state_leg_result_is_successful(
                result.position) ||
            !g1_controller_state_orientation_is_successful(
                result.orientation) ||
            !g1_controller_state_defensive_swing_is_valid(
                result.defensive_swing, result.recorded_contact)) {
            return false;
        }
        maximum_correction = maxf(
            maximum_correction,
            maxf(
                result.position.max_correction_radians,
                result.orientation.correction_radians));
    }
    return g1_ik_runtime_float_bits(maximum_correction) ==
           g1_ik_runtime_float_bits(value.max_correction_radians);
}

static inline bool
g1_controller_state_root_reach_local_pose_is_valid(
    const g1_controller_state& state)
{
    if (state.adjusted_bone_positions.size != G1_BoneCount ||
        state.ik_bone_positions.size != G1_BoneCount) {
        return false;
    }
    float expected_root_y = 0.0f;
    if (!g1_apply_root_reach_plan_y(
            expected_root_y,
            state.adjusted_bone_positions(G1_Simulation).y,
            state.ik_frame.root_reach) ||
        terrain_float_bits(
            state.ik_bone_positions(G1_Simulation).y) !=
            terrain_float_bits(expected_root_y) ||
        terrain_float_bits(
            state.ik_bone_positions(G1_Simulation).x) !=
            terrain_float_bits(
                state.adjusted_bone_positions(G1_Simulation).x) ||
        terrain_float_bits(
            state.ik_bone_positions(G1_Simulation).z) !=
            terrain_float_bits(
                state.adjusted_bone_positions(G1_Simulation).z)) {
        return false;
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (bone != G1_Simulation) {
            const vec3 accepted = state.ik_bone_positions(bone);
            const vec3 baseline = state.adjusted_bone_positions(bone);
            if (terrain_float_bits(accepted.x) !=
                    terrain_float_bits(baseline.x) ||
                terrain_float_bits(accepted.y) !=
                    terrain_float_bits(baseline.y) ||
                terrain_float_bits(accepted.z) !=
                    terrain_float_bits(baseline.z)) {
                return false;
            }
        }
    }
    return true;
}

static inline bool g1_controller_state_pose_clearance_is_valid(
    const G1PoseClearance& value)
{
    struct leaf_contract
    {
        const G1ClearanceResult* result;
        uint32_t first_primitive;
        uint32_t last_primitive;
        bool point_special;
    };
    const leaf_contract leaves[] = {
        {&value.hips, 0U, 0U, true},
        {&value.left.knee, 1U, 1U, true},
        {&value.left.ankle, 2U, 2U, true},
        {&value.left.toe, 3U, 3U, true},
        {&value.left.foot, 4U, 7U, false},
        {&value.left.thigh, 8U, 8U, false},
        {&value.left.shin, 9U, 9U, false},
        {&value.right.knee, 10U, 10U, true},
        {&value.right.ankle, 11U, 11U, true},
        {&value.right.toe, 12U, 12U, true},
        {&value.right.foot, 13U, 16U, false},
        {&value.right.thigh, 17U, 17U, false},
        {&value.right.shin, 18U, 18U, false}
    };
    for (const leaf_contract& leaf : leaves) {
        if (!g1_controller_state_clearance_result_is_valid(*leaf.result) ||
            leaf.result->witness.primitive_index < leaf.first_primitive ||
            leaf.result->witness.primitive_index > leaf.last_primitive) {
            return false;
        }
        if (leaf.point_special) {
            if (leaf.result->witness.candidate_kind != 3U ||
                leaf.result->witness.patch_index != 0U ||
                leaf.result->witness.candidate_subindex != 0U) {
                return false;
            }
        } else if (leaf.result->witness.candidate_kind > 2U ||
                   leaf.result->witness.patch_index >=
                       G1ClearancePatchesPerPair) {
            return false;
        }
    }
    const G1ClearanceResult* summaries[] = {
        &value.left.minimum, &value.right.minimum, &value.minimum
    };
    for (const G1ClearanceResult* summary : summaries) {
        if (!g1_controller_state_clearance_result_is_valid(*summary)) {
            return false;
        }
    }
    return true;
}

static inline bool g1_controller_state_vec3_bits_equal(
    vec3 left,
    vec3 right)
{
    return terrain_float_bits(left.x) == terrain_float_bits(right.x) &&
           terrain_float_bits(left.y) == terrain_float_bits(right.y) &&
           terrain_float_bits(left.z) == terrain_float_bits(right.z);
}

static inline bool g1_controller_state_quat_bits_equal(
    quat left,
    quat right)
{
    return terrain_float_bits(left.w) == terrain_float_bits(right.w) &&
           terrain_float_bits(left.x) == terrain_float_bits(right.x) &&
           terrain_float_bits(left.y) == terrain_float_bits(right.y) &&
           terrain_float_bits(left.z) == terrain_float_bits(right.z);
}

static inline bool
g1_controller_state_applied_ik_endpoints_are_authenticated(
    const g1_controller_state& state)
{
    if (!state.ik_frame.applied) return true;
    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    for (int foot = 0; foot < 2; ++foot) {
        const bool contact = state.curr_bone_contacts(foot);
        const G1FootIkState& runtime = state.ik.feet[foot];
        const G1FootFrameResult& result = state.ik_frame.feet[foot];
        vec3 current_sole_center;
        vec3 current_sole_normal;
        if (!g1_ik_checked_physical_sole_centroid(
                current_sole_center,
                state.global_bone_positions(
                    configs[foot].contact),
                state.global_bone_rotations(
                    configs[foot].contact),
                configs[foot]) ||
            !ik_checked_quat_rotate(
                current_sole_normal,
                state.global_bone_rotations(
                    configs[foot].contact),
                configs[foot].sole_normal_local) ||
            !g1_ik_surface_normal_is_valid(
                current_sole_normal)) {
            return false;
        }
        if (runtime.lock.contact != contact ||
            runtime.lock.locked != contact ||
            result.recorded_contact != contact ||
            result.target.locked != runtime.lock.locked ||
            result.target.position_active !=
                runtime.lock.position_active ||
            result.target.releasing != runtime.lock.releasing ||
            !g1_controller_state_vec3_bits_equal(
                runtime.lock.previous_input,
                g1_ik_vec3_canonicalize(
                    current_sole_center)) ||
            !g1_controller_state_vec3_bits_equal(
                runtime.baseline_sole_normal,
                current_sole_normal)) {
            return false;
        }
        if (contact) {
            float expected_horizontal_drift = 0.0f;
            bool expected_drift_limit_exceeded = false;
            if (!g1_controller_state_vec3_bits_equal(
                    result.target.sole_center,
                    runtime.lock.lock_point) ||
                !g1_controller_state_vec3_bits_equal(
                    result.target.surface.point,
                    runtime.lock.lock_point) ||
                !g1_controller_state_vec3_bits_equal(
                    result.target.desired_sole_normal,
                    result.target.surface.normal) ||
                !g1_foot_horizontal_drift(
                    expected_horizontal_drift,
                    expected_drift_limit_exceeded,
                    runtime.lock.previous_input,
                    runtime.lock.lock_point) ||
                terrain_float_bits(
                    result.target.horizontal_drift_m) !=
                    terrain_float_bits(
                        expected_horizontal_drift) ||
                result.target.drift_limit_exceeded !=
                    expected_drift_limit_exceeded) {
                return false;
            }
        } else if (!g1_controller_state_vec3_bits_equal(
                       result.target.sole_center,
                       runtime.lock.output_position) ||
                   terrain_float_bits(
                       result.target.surface.point.x) !=
                       terrain_float_bits(
                           runtime.lock.previous_input.x) ||
                   terrain_float_bits(
                       result.target.surface.point.z) !=
                       terrain_float_bits(
                           runtime.lock.previous_input.z) ||
                   !g1_controller_state_vec3_bits_equal(
                       result.target.desired_sole_normal,
                       current_sole_normal) ||
                   terrain_float_bits(
                       result.target.horizontal_drift_m) != 0U ||
                   result.target.drift_limit_exceeded) {
            return false;
        }
        G1FootTarget applied_target = result.target;
        if (!contact) {
            const G1SwingCandidateDiagnostic& selected =
                result.swing_selection.selected;
            float lift = 0.0f;
            float expected_y = 0.0f;
            std::memcpy(&lift, &selected.lift_bits, sizeof(lift));
            if (g1_apply_swing_lift_y(
                    expected_y,
                    result.target.sole_center.y,
                    lift,
                    NULL,
                    0) != G1ClearanceOk ||
                terrain_float_bits(expected_y) !=
                    selected.materialized_command_y_bits) {
                return false;
            }
            applied_target.sole_center.y = expected_y;
        }
        vec3 final_centers[4] = {};
        if (!g1_ik_runtime_compute_foot_centers(
                final_centers,
                state.ik_global_bone_positions,
                state.ik_global_bone_rotations,
                configs[foot])) {
            return false;
        }
        if (!g1_ik_runtime_controller_constraints_pass(
                result.position,
                result.orientation,
                state.ik_global_bone_positions(
                    configs[foot].contact),
                state.ik_global_bone_rotations(
                    configs[foot].contact),
                applied_target,
                configs[foot])) {
            return false;
        }
        for (int probe = 0; probe < 4; ++probe) {
            if (!g1_controller_state_vec3_bits_equal(
                    state.ik.feet[foot].swing
                        .previous_sphere_centers[probe],
                    final_centers[probe])) {
                return false;
            }
            if (!state.ik_frame.feet[foot].recorded_contact) {
                const uint32_t expected_bits[3] = {
                    terrain_float_bits(final_centers[probe].x),
                    terrain_float_bits(final_centers[probe].y),
                    terrain_float_bits(final_centers[probe].z)
                };
                for (int axis = 0; axis < 3; ++axis) {
                    if (state.ik_frame.feet[foot]
                            .swing_selection.selected
                            .actual_sphere_center_bits[probe][axis] !=
                        expected_bits[axis]) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static inline bool g1_controller_state_surface_bits_equal(
    const G1SurfaceSample& left,
    const G1SurfaceSample& right)
{
    return terrain_float_bits(left.height) ==
               terrain_float_bits(right.height) &&
           g1_controller_state_vec3_bits_equal(
               left.normal, right.normal);
}

static inline bool g1_controller_state_surface_is_positive_zero(
    const G1SurfaceSample& value)
{
    return terrain_float_bits(value.height) == 0U &&
           terrain_float_bits(value.normal.x) == 0U &&
           terrain_float_bits(value.normal.y) == 0U &&
           terrain_float_bits(value.normal.z) == 0U;
}

static inline bool g1_controller_state_vec3_is_positive_zero(vec3 value)
{
    return terrain_float_bits(value.x) == 0U &&
           terrain_float_bits(value.y) == 0U &&
           terrain_float_bits(value.z) == 0U;
}

static inline uint64_t g1_controller_state_double_bits(double value)
{
    uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static inline bool g1_controller_state_clearance_work_equal(
    const G1ClearanceWork& left,
    const G1ClearanceWork& right)
{
    return left.point_queries == right.point_queries &&
           left.cells_visited == right.cells_visited &&
           left.primitive_triangle_pairs ==
               right.primitive_triangle_pairs &&
           left.face_patches == right.face_patches &&
           left.candidate_tests == right.candidate_tests &&
           left.subdivision_nodes == right.subdivision_nodes;
}

static inline bool g1_controller_state_clearance_witness_equal(
    const G1ClearanceWitness& left,
    const G1ClearanceWitness& right)
{
    return g1_controller_state_double_bits(left.body_x) ==
               g1_controller_state_double_bits(right.body_x) &&
           g1_controller_state_double_bits(left.body_y) ==
               g1_controller_state_double_bits(right.body_y) &&
           g1_controller_state_double_bits(left.body_z) ==
               g1_controller_state_double_bits(right.body_z) &&
           g1_controller_state_double_bits(left.surface_x) ==
               g1_controller_state_double_bits(right.surface_x) &&
           g1_controller_state_double_bits(left.surface_y) ==
               g1_controller_state_double_bits(right.surface_y) &&
           g1_controller_state_double_bits(left.surface_z) ==
               g1_controller_state_double_bits(right.surface_z) &&
           g1_controller_state_double_bits(left.segment_parameter) ==
               g1_controller_state_double_bits(
                   right.segment_parameter) &&
           g1_controller_state_double_bits(left.terrain_weight_0) ==
               g1_controller_state_double_bits(
                   right.terrain_weight_0) &&
           g1_controller_state_double_bits(left.terrain_weight_1) ==
               g1_controller_state_double_bits(
                   right.terrain_weight_1) &&
           g1_controller_state_double_bits(left.terrain_weight_2) ==
               g1_controller_state_double_bits(
                   right.terrain_weight_2) &&
           left.primitive_index == right.primitive_index &&
           left.cell_x == right.cell_x && left.cell_z == right.cell_z &&
           left.terrain_triangle_index ==
               right.terrain_triangle_index &&
           left.patch_index == right.patch_index &&
           left.candidate_kind == right.candidate_kind &&
           left.candidate_subindex == right.candidate_subindex;
}

static inline bool g1_controller_state_clearance_result_equal(
    const G1ClearanceResult& left,
    const G1ClearanceResult& right)
{
    return g1_controller_state_double_bits(left.lower_bound_m) ==
               g1_controller_state_double_bits(right.lower_bound_m) &&
           g1_controller_state_double_bits(left.witness_upper_m) ==
               g1_controller_state_double_bits(right.witness_upper_m) &&
           g1_controller_state_clearance_witness_equal(
               left.witness, right.witness) &&
           g1_controller_state_clearance_work_equal(
               left.work, right.work);
}

static inline bool g1_controller_state_clearance_witness_key_less(
    const G1ClearanceWitness& left,
    const G1ClearanceWitness& right)
{
    if (left.primitive_index != right.primitive_index) {
        return left.primitive_index < right.primitive_index;
    }
    if (left.cell_z != right.cell_z) return left.cell_z < right.cell_z;
    if (left.cell_x != right.cell_x) return left.cell_x < right.cell_x;
    if (left.terrain_triangle_index !=
        right.terrain_triangle_index) {
        return left.terrain_triangle_index <
               right.terrain_triangle_index;
    }
    if (left.patch_index != right.patch_index) {
        return left.patch_index < right.patch_index;
    }
    if (left.candidate_kind != right.candidate_kind) {
        return left.candidate_kind < right.candidate_kind;
    }
    return left.candidate_subindex < right.candidate_subindex;
}

static inline bool g1_controller_state_checked_add_work(
    G1ClearanceWork& total,
    const G1ClearanceWork& value)
{
#define G1_ADD_WORK(name) \
    if (value.name > UINT32_MAX - total.name) return false; \
    total.name += value.name
    G1_ADD_WORK(point_queries);
    G1_ADD_WORK(cells_visited);
    G1_ADD_WORK(primitive_triangle_pairs);
    G1_ADD_WORK(face_patches);
    G1_ADD_WORK(candidate_tests);
    G1_ADD_WORK(subdivision_nodes);
#undef G1_ADD_WORK
    return true;
}

static inline bool g1_controller_state_clearance_summary_is_valid(
    const G1ClearanceResult& summary,
    const G1ClearanceResult* const* components,
    int component_count)
{
    if (components == NULL || component_count <= 0 ||
        components[0] == NULL) {
        return false;
    }
    double minimum_lower = components[0]->lower_bound_m;
    const G1ClearanceResult* canonical_witness = components[0];
    G1ClearanceWork total = {};
    for (int index = 0; index < component_count; ++index) {
        if (components[index] == NULL ||
            !g1_controller_state_checked_add_work(
                total, components[index]->work)) {
            return false;
        }
        if (components[index]->lower_bound_m < minimum_lower) {
            minimum_lower = components[index]->lower_bound_m;
        }
        if (components[index]->witness_upper_m <
                canonical_witness->witness_upper_m ||
            (g1_controller_state_double_bits(
                 components[index]->witness_upper_m) ==
                 g1_controller_state_double_bits(
                     canonical_witness->witness_upper_m) &&
             g1_controller_state_clearance_witness_key_less(
                 components[index]->witness,
                 canonical_witness->witness))) {
            canonical_witness = components[index];
        }
    }
    if (g1_controller_state_double_bits(summary.lower_bound_m) !=
            g1_controller_state_double_bits(minimum_lower) ||
        g1_controller_state_double_bits(summary.witness_upper_m) !=
            g1_controller_state_double_bits(
                canonical_witness->witness_upper_m) ||
        !g1_controller_state_clearance_work_equal(
            summary.work, total) ||
        !g1_controller_state_clearance_witness_equal(
            summary.witness, canonical_witness->witness)) {
        return false;
    }
    return true;
}

static inline bool g1_controller_state_pose_clearance_is_coherent(
    const G1PoseClearance& value)
{
    const G1ClearanceResult* left_components[] = {
        &value.left.knee, &value.left.ankle, &value.left.toe,
        &value.left.foot, &value.left.thigh, &value.left.shin
    };
    const G1ClearanceResult* right_components[] = {
        &value.right.knee, &value.right.ankle, &value.right.toe,
        &value.right.foot, &value.right.thigh, &value.right.shin
    };
    const G1ClearanceResult* pose_components[] = {
        &value.hips, &value.left.minimum, &value.right.minimum
    };
    return g1_controller_state_pose_clearance_is_valid(value) &&
           g1_controller_state_clearance_summary_is_valid(
               value.left.minimum,
               left_components,
               static_cast<int>(sizeof(left_components) /
                                sizeof(left_components[0]))) &&
           g1_controller_state_clearance_summary_is_valid(
               value.right.minimum,
               right_components,
               static_cast<int>(sizeof(right_components) /
                                sizeof(right_components[0]))) &&
           g1_controller_state_clearance_summary_is_valid(
               value.minimum,
               pose_components,
               static_cast<int>(sizeof(pose_components) /
                                sizeof(pose_components[0])));
}

static inline bool g1_controller_state_pose_clearance_meets_thresholds(
    const G1PoseClearance& value)
{
    return value.left.toe.lower_bound_m >= -0.005 &&
           value.left.foot.lower_bound_m >= -0.005 &&
           value.right.toe.lower_bound_m >= -0.005 &&
           value.right.foot.lower_bound_m >= -0.005 &&
           value.minimum.lower_bound_m >= -0.01;
}

static inline bool g1_controller_state_pose_clearance_equal(
    const G1PoseClearance& left,
    const G1PoseClearance& right)
{
    const G1ClearanceResult* left_results[] = {
        &left.hips,
        &left.left.knee, &left.left.ankle, &left.left.toe,
        &left.left.foot, &left.left.thigh, &left.left.shin,
        &left.left.minimum,
        &left.right.knee, &left.right.ankle, &left.right.toe,
        &left.right.foot, &left.right.thigh, &left.right.shin,
        &left.right.minimum, &left.minimum
    };
    const G1ClearanceResult* right_results[] = {
        &right.hips,
        &right.left.knee, &right.left.ankle, &right.left.toe,
        &right.left.foot, &right.left.thigh, &right.left.shin,
        &right.left.minimum,
        &right.right.knee, &right.right.ankle, &right.right.toe,
        &right.right.foot, &right.right.thigh, &right.right.shin,
        &right.right.minimum, &right.minimum
    };
    for (int index = 0;
         index < static_cast<int>(sizeof(left_results) /
                                sizeof(left_results[0]));
         ++index) {
        if (!g1_controller_state_clearance_result_equal(
                *left_results[index], *right_results[index])) {
            return false;
        }
    }
    return true;
}

static inline bool g1_controller_state_pose_fk_matches(
    const array1d<vec3>& local_positions,
    const array1d<quat>& local_rotations,
    const array1d<vec3>& global_positions,
    const array1d<quat>& global_rotations)
{
    static const int parents[G1_BoneCount] = {
        -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
    };
    if (local_positions.size != G1_BoneCount ||
        local_rotations.size != G1_BoneCount ||
        global_positions.size != G1_BoneCount ||
        global_rotations.size != G1_BoneCount ||
        local_positions.data == NULL || local_rotations.data == NULL ||
        global_positions.data == NULL || global_rotations.data == NULL) {
        return false;
    }
    vec3 expected_positions[G1_BoneCount] = {};
    quat expected_rotations[G1_BoneCount] = {};
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (!g1_ik_vec3_is_runtime_value(local_positions(bone)) ||
            !ik_quat_is_unit(local_rotations(bone))) {
            return false;
        }
        const int parent = parents[bone];
        if (parent == -1) {
            expected_positions[bone] = local_positions(bone);
            expected_rotations[bone] = local_rotations(bone);
        } else {
            vec3 rotated_local;
            if (!ik_checked_quat_rotate(
                    rotated_local,
                    expected_rotations[parent],
                    local_positions(bone)) ||
                !ik_checked_vec3_add(
                    expected_positions[bone],
                    expected_positions[parent],
                    rotated_local) ||
                !ik_checked_quat_multiply(
                    expected_rotations[bone],
                    expected_rotations[parent],
                    local_rotations(bone))) {
                return false;
            }
        }
        if (!g1_controller_state_vec3_bits_equal(
                expected_positions[bone], global_positions(bone)) ||
            !g1_controller_state_quat_bits_equal(
                expected_rotations[bone], global_rotations(bone))) {
            return false;
        }
    }
    return true;
}

static inline bool g1_controller_state_pose_arrays_bits_equal(
    const array1d<vec3>& left_positions,
    const array1d<quat>& left_rotations,
    const array1d<vec3>& right_positions,
    const array1d<quat>& right_rotations)
{
    if (left_positions.size != G1_BoneCount ||
        left_rotations.size != G1_BoneCount ||
        right_positions.size != G1_BoneCount ||
        right_rotations.size != G1_BoneCount) {
        return false;
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (!g1_controller_state_vec3_bits_equal(
                left_positions(bone), right_positions(bone)) ||
            !g1_controller_state_quat_bits_equal(
                left_rotations(bone), right_rotations(bone))) {
            return false;
        }
    }
    return true;
}

static inline bool g1_controller_state_accepted_footprint_is_valid(
    const g1_controller_state& state)
{
    if (state.footprint_status != G1FootprintOk ||
        state.footprint.blocked ||
        state.footprint.blocked_reason != walkability_clear) {
        return false;
    }
    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    vec3 current_centers[2][4] = {};
    for (int foot = 0; foot < 2; ++foot) {
        if (!g1_ik_runtime_compute_foot_centers(
                current_centers[foot],
                state.global_bone_positions,
                state.global_bone_rotations,
                configs[foot])) {
            return false;
        }
    }
    if (!g1_ik_runtime_footprint_validate(
            state.footprint,
            state.curr_bone_contacts,
            current_centers,
            NULL,
            0)) {
        return false;
    }

    const G1FootprintObservation& footprint = state.footprint;
    uint32_t landing_count = 0U;
    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        const G1FootprintFootObservation& foot =
            footprint.feet[foot_index];
        float aggregate_minimum =
            foot.probes[0].corridor_minimum_height;
        float aggregate_maximum =
            foot.probes[0].corridor_maximum_height;
        int aggregate_class =
            foot.probes[0].encountered_walkability_class;
        double maximum_root_split = 0.0;
        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            const G1FootprintProbe& probe = foot.probes[probe_index];
            if (!g1_controller_state_vec3_bits_equal(
                    probe.predicted_sphere_centers[0],
                    probe.current_sphere_center) ||
                !g1_controller_state_vec3_bits_equal(
                    probe.predicted_sole_points[0],
                    probe.current_sole_point) ||
                probe.predicted_surface_status[0] !=
                    G1SurfaceQueryValid ||
                !g1_controller_state_surface_bits_equal(
                    probe.predicted_surfaces[0],
                    probe.current_surface)) {
                return false;
            }
            for (int sample = 0;
                 sample < G1CommandTrajectorySampleCount;
                 ++sample) {
                if (probe.predicted_surface_status[sample] !=
                        G1SurfaceQueryValid ||
                    !g1_ik_runtime_surface_sample_is_valid(
                        probe.predicted_surfaces[sample])) {
                    return false;
                }
            }
            if (probe.corridor_minimum_height < aggregate_minimum) {
                aggregate_minimum = probe.corridor_minimum_height;
            }
            if (probe.corridor_maximum_height > aggregate_maximum) {
                aggregate_maximum = probe.corridor_maximum_height;
            }
            aggregate_class = g1_footprint_class_merge(
                aggregate_class,
                probe.encountered_walkability_class);
            const double differences[] = {
                static_cast<double>(footprint.root_surface.height) -
                    static_cast<double>(probe.current_surface.height),
                static_cast<double>(footprint.root_surface.height) -
                    static_cast<double>(probe.corridor_minimum_height),
                static_cast<double>(footprint.root_surface.height) -
                    static_cast<double>(probe.corridor_maximum_height)
            };
            for (double difference : differences) {
                if (!terrain_double_is_finite(difference)) return false;
                if (difference < 0.0) difference = -difference;
                if (difference > maximum_root_split) {
                    maximum_root_split = difference;
                }
            }
        }
        if (terrain_float_bits(foot.corridor_minimum_height) !=
                terrain_float_bits(aggregate_minimum) ||
            terrain_float_bits(foot.corridor_maximum_height) !=
                terrain_float_bits(aggregate_maximum) ||
            foot.encountered_walkability_class != aggregate_class ||
            g1_controller_state_double_bits(
                foot.maximum_root_split_m) !=
                g1_controller_state_double_bits(maximum_root_split) ||
            foot.multilevel !=
                (maximum_root_split >= static_cast<double>(0.04f))) {
            return false;
        }

        if (!foot.landing_expected) {
            if (foot.landing_patch_ready ||
                foot.landing_sample != UINT32_MAX ||
                !g1_controller_state_vec3_is_positive_zero(
                    foot.predicted_landing_sole_center) ||
                foot.predicted_landing_surface_status !=
                    G1SurfaceQueryInvalid ||
                !g1_controller_state_surface_is_positive_zero(
                    foot.predicted_landing_surface) ||
                foot.predicted_landing_walkability_class != 0 ||
                g1_controller_state_double_bits(
                    foot.landing_patch_maximum_residual_m) != 0U) {
                return false;
            }
            for (int probe = 0; probe < 4; ++probe) {
                if (!g1_controller_state_surface_is_positive_zero(
                        foot.probes[probe].selected_landing_surface)) {
                    return false;
                }
            }
            continue;
        }

        ++landing_count;
        if (foot.current_contact ||
            foot.landing_sample == 0U ||
            foot.landing_sample >= G1CommandTrajectorySampleCount ||
            foot.predicted_landing_surface_status !=
                G1SurfaceQueryValid ||
            !g1_ik_runtime_surface_sample_is_valid(
                foot.predicted_landing_surface) ||
            foot.predicted_landing_walkability_class == 0) {
            return false;
        }
        volatile double sum_x = 0.0;
        volatile double sum_y = 0.0;
        volatile double sum_z = 0.0;
        bool all_class_one =
            foot.predicted_landing_walkability_class == 1;
        double maximum_residual = 0.0;
        const uint32_t sample = foot.landing_sample;
        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            const G1FootprintProbe& probe = foot.probes[probe_index];
            if (!g1_controller_state_surface_bits_equal(
                    probe.selected_landing_surface,
                    probe.predicted_surfaces[sample])) {
                return false;
            }
            const vec3 sole = probe.predicted_sole_points[sample];
            sum_x = sum_x + static_cast<double>(sole.x);
            sum_y = sum_y + static_cast<double>(sole.y);
            sum_z = sum_z + static_cast<double>(sole.z);
            all_class_one = all_class_one &&
                probe.encountered_walkability_class == 1 &&
                probe.predicted_surface_status[sample] ==
                    G1SurfaceQueryValid;
            const volatile double difference_x =
                static_cast<double>(sole.x) -
                static_cast<double>(
                    foot.predicted_landing_sole_center.x);
            const volatile double difference_y =
                static_cast<double>(
                    probe.predicted_surfaces[sample].height) -
                static_cast<double>(
                    foot.predicted_landing_surface.height);
            const volatile double difference_z =
                static_cast<double>(sole.z) -
                static_cast<double>(
                    foot.predicted_landing_sole_center.z);
            const volatile double x_term = difference_x *
                static_cast<double>(
                    foot.predicted_landing_surface.normal.x);
            const volatile double y_term = difference_y *
                static_cast<double>(
                    foot.predicted_landing_surface.normal.y);
            const volatile double z_term = difference_z *
                static_cast<double>(
                    foot.predicted_landing_surface.normal.z);
            const volatile double first_sum = x_term + y_term;
            double residual = first_sum + z_term;
            if (!terrain_double_is_finite(residual)) return false;
            if (residual < 0.0) residual = -residual;
            if (residual > maximum_residual) {
                maximum_residual = residual;
            }
        }
        vec3 centroid;
        if (!terrain_v2_round_output(sum_x / 4.0, centroid.x) ||
            !terrain_v2_round_output(sum_y / 4.0, centroid.y) ||
            !terrain_v2_round_output(sum_z / 4.0, centroid.z) ||
            !g1_controller_state_vec3_bits_equal(
                centroid, foot.predicted_landing_sole_center) ||
            g1_controller_state_double_bits(maximum_residual) !=
                g1_controller_state_double_bits(
                    foot.landing_patch_maximum_residual_m)) {
            return false;
        }
        const uint32_t residual_limit_bits = UINT32_C(0x3ba3d70a);
        float residual_limit = 0.0f;
        std::memcpy(
            &residual_limit,
            &residual_limit_bits,
            sizeof(residual_limit));
        const bool expected_ready = all_class_one &&
            maximum_residual <= static_cast<double>(residual_limit);
        if (foot.landing_patch_ready != expected_ready) {
            return false;
        }
    }
    return footprint.work.sweeps == 24U &&
           footprint.work.surface_queries == 33U + landing_count;
}

static inline bool g1_controller_state_scalar_values_are_valid(
    const g1_controller_state& state)
{
    const float finite_values[] = {
        state.search_time,
        state.search_timer,
        state.force_search_timer,
        state.desired_gait,
        state.desired_gait_velocity,
        state.traversal_speed_scale,
        state.traversal_speed_scale_velocity,
        state.blocked_distance,
        state.camera_azimuth,
        state.camera_altitude,
        state.camera_distance,
        state.incumbent_cost,
        state.selected_cost,
        state.selected_terrain_error,
        state.adjustment_xz,
        state.adjustment_y,
        state.clamp_xz,
        state.clamp_y
    };
    for (float value : finite_values) {
        if (!terrain_float_is_finite(value)) return false;
    }
    const vec3 finite_vec3s[] = {
        state.transition_src_position,
        state.transition_dst_position,
        state.desired_velocity,
        state.desired_velocity_change_curr,
        state.desired_velocity_change_prev,
        state.desired_rotation_change_curr,
        state.desired_rotation_change_prev,
        state.simulation_position,
        state.simulation_velocity,
        state.simulation_acceleration,
        state.simulation_angular_velocity,
        state.blocked_point
    };
    for (vec3 value : finite_vec3s) {
        if (!g1_ik_vec3_is_runtime_value(value)) return false;
    }
    return state.frame_index >= 0 && state.scene_frame >= 0 &&
           state.search_time >= 0.0f && state.search_time <= 10.0f &&
           ik_quat_is_unit(state.transition_src_rotation) &&
           ik_quat_is_unit(state.transition_dst_rotation) &&
           ik_quat_is_unit(state.desired_rotation) &&
           ik_quat_is_unit(state.simulation_rotation) &&
           state.support.initialized &&
           terrain_float_is_finite(state.support.height) &&
           terrain_float_is_finite(state.support.velocity) &&
           terrain_float_is_finite(state.support.nominal_height) &&
           terrain_float_is_finite(state.support.nominal_velocity) &&
           terrain_float_is_finite(state.support.offset_height) &&
           terrain_float_is_finite(state.support.offset_velocity) &&
           state.support.airborne_frames >= 0 &&
           state.support.source >= support_root &&
           state.support.source <= support_airborne_root &&
           support_observation_is_finite(state.support_observation_now) &&
           state.traversal_speed_scale >= 0.0f &&
           state.traversal_speed_scale <= 1.0f &&
           state.walkability_class >= 0 && state.walkability_class <= 2 &&
           state.blocked_distance >= 0.0f &&
           state.route_index >= -1 && state.route_waypoint >= 0 &&
           state.route_frames >= 0 && state.camera_altitude >= 0.0f &&
           state.camera_distance > 0.0f &&
           state.incumbent_cost >= 0.0f && state.selected_cost >= 0.0f &&
           state.selected_terrain_error >= 0.0f &&
           state.adjustment_xz >= 0.0f && state.clamp_xz >= 0.0f &&
           terrain_float_bits(state.adjustment_y) == 0U &&
           terrain_float_bits(state.clamp_y) == 0U &&
           g1_command_snapshot_is_valid(state.command);
}

static inline bool g1_controller_state_is_valid(
    const g1_controller_state& state)
{
    if (!g1_controller_state_storage_is_valid(state) ||
        !g1_controller_state_scalar_values_are_valid(state) ||
        !g1_ik_runtime_state_is_valid(state.ik) ||
        !g1_controller_state_accepted_footprint_is_valid(state) ||
        !g1_controller_state_ik_frame_is_valid(
            state.ik_frame,
            state.curr_bone_contacts,
            state.footprint) ||
        !g1_controller_state_root_reach_local_pose_is_valid(state) ||
        state.ik_candidate_clearance_status != G1ClearanceOk ||
        state.ik_candidate_rejected ||
        !g1_controller_state_pose_clearance_is_coherent(
            state.ik_clearance) ||
        !g1_controller_state_pose_clearance_is_coherent(
            state.ik_candidate_clearance) ||
        !g1_controller_state_pose_clearance_meets_thresholds(
            state.ik_clearance) ||
        !g1_controller_state_pose_clearance_meets_thresholds(
            state.ik_candidate_clearance) ||
        !g1_controller_state_pose_clearance_equal(
            state.ik_clearance, state.ik_candidate_clearance)) {
        return false;
    }

    const array1d<vec3>* vec_arrays[] = {
        &state.curr_bone_positions, &state.curr_bone_velocities,
        &state.trns_bone_positions, &state.trns_bone_velocities,
        &state.curr_bone_angular_velocities,
        &state.trns_bone_angular_velocities,
        &state.bone_positions, &state.bone_velocities,
        &state.bone_angular_velocities,
        &state.bone_offset_positions, &state.bone_offset_velocities,
        &state.bone_offset_angular_velocities,
        &state.adjusted_bone_positions, &state.global_bone_positions,
        &state.global_bone_velocities,
        &state.global_bone_angular_velocities,
        &state.trajectory_desired_velocities, &state.trajectory_positions,
        &state.trajectory_velocities, &state.trajectory_accelerations,
        &state.trajectory_angular_velocities,
        &state.contact_positions, &state.contact_velocities,
        &state.contact_points, &state.contact_targets,
        &state.contact_offset_positions, &state.contact_offset_velocities,
        &state.ik_bone_positions, &state.ik_global_bone_positions,
        &state.ik_candidate_bone_positions,
        &state.ik_candidate_global_bone_positions
    };
    for (const array1d<vec3>* values : vec_arrays) {
        if (!g1_controller_state_vec3_array_is_finite(*values)) return false;
    }
    const array1d<quat>* quat_arrays[] = {
        &state.curr_bone_rotations, &state.trns_bone_rotations,
        &state.bone_rotations, &state.bone_offset_rotations,
        &state.adjusted_bone_rotations, &state.global_bone_rotations,
        &state.trajectory_desired_rotations, &state.trajectory_rotations,
        &state.ik_bone_rotations, &state.ik_global_bone_rotations,
        &state.ik_candidate_bone_rotations,
        &state.ik_candidate_global_bone_rotations
    };
    for (const array1d<quat>* values : quat_arrays) {
        if (!g1_controller_state_quat_array_is_valid(*values)) return false;
    }
    if (state.contact_bones(0) != G1_LeftToe ||
        state.contact_bones(1) != G1_RightToe) {
        return false;
    }
    if (!g1_controller_state_pose_fk_matches(
            state.adjusted_bone_positions,
            state.adjusted_bone_rotations,
            state.global_bone_positions,
            state.global_bone_rotations) ||
        !g1_controller_state_pose_fk_matches(
            state.ik_bone_positions,
            state.ik_bone_rotations,
            state.ik_global_bone_positions,
            state.ik_global_bone_rotations) ||
        !g1_controller_state_pose_fk_matches(
            state.ik_candidate_bone_positions,
            state.ik_candidate_bone_rotations,
            state.ik_candidate_global_bone_positions,
            state.ik_candidate_global_bone_rotations) ||
        !g1_controller_state_pose_arrays_bits_equal(
            state.ik_bone_positions,
            state.ik_bone_rotations,
            state.ik_candidate_bone_positions,
            state.ik_candidate_bone_rotations) ||
        !g1_controller_state_pose_arrays_bits_equal(
            state.ik_global_bone_positions,
            state.ik_global_bone_rotations,
            state.ik_candidate_global_bone_positions,
            state.ik_candidate_global_bone_rotations) ||
        !g1_controller_state_applied_ik_endpoints_are_authenticated(
            state) ||
        !g1_controller_state_vec3_bits_equal(
            state.desired_velocity,
            state.command.applied_velocity) ||
        !g1_controller_state_quat_bits_equal(
            state.desired_rotation,
            state.command.intent.desired_heading)) {
        return false;
    }
    for (int sample = 0;
         sample < G1CommandTrajectorySampleCount;
         ++sample) {
        if (!g1_controller_state_vec3_bits_equal(
                state.trajectory_desired_velocities(sample),
                state.command.predicted_desired_velocities[sample]) ||
            !g1_controller_state_vec3_bits_equal(
                state.trajectory_positions(sample),
                state.command.predicted_root_positions[sample]) ||
            !g1_controller_state_quat_bits_equal(
                state.trajectory_rotations(sample),
                state.command.predicted_root_rotations[sample]) ||
            !g1_controller_state_quat_bits_equal(
                state.trajectory_desired_rotations(sample),
                state.command.predicted_desired_headings[sample])) {
            return false;
        }
    }
    if (state.scene_frame > 0 &&
        (state.support_observation_now.contact[0] !=
             state.curr_bone_contacts(0) ||
         state.support_observation_now.contact[1] !=
             state.curr_bone_contacts(1))) {
        return false;
    }
    return true;
}

static inline bool g1_controller_state_copy(
    g1_controller_state& output,
    const g1_controller_state& input,
    char* error,
    int error_capacity)
{
    g1_controller_state_memory_range input_ranges[64] = {};
    g1_controller_state_memory_range output_ranges[64] = {};
    int input_count = 0;
    int output_count = 0;
    const int range_capacity =
        static_cast<int>(sizeof(input_ranges) / sizeof(input_ranges[0]));
    if (error_capacity < 0 || &output == &input ||
        !g1_controller_state_storage_ranges(
            input, input_ranges, input_count, range_capacity) ||
        !g1_controller_state_storage_ranges(
            output, output_ranges, output_count, range_capacity) ||
        input_count != output_count ||
        !g1_controller_state_ranges_are_disjoint(
            output_ranges, output_count)) {
        return false;
    }
    const g1_controller_state_memory_range diagnostic = {
        error,
        error != NULL && error_capacity > 0
            ? static_cast<std::size_t>(error_capacity)
            : 0U
    };
    if (diagnostic.bytes > 0U) {
        const g1_controller_state_memory_range input_object = {
            &input, sizeof(input)
        };
        const g1_controller_state_memory_range output_object = {
            &output, sizeof(output)
        };
        if (g1_controller_state_ranges_overlap(diagnostic, input_object) ||
            g1_controller_state_ranges_overlap(diagnostic, output_object)) {
            return false;
        }
        for (int index = 0; index < input_count; ++index) {
            if (g1_controller_state_ranges_overlap(
                    diagnostic, input_ranges[index]) ||
                g1_controller_state_ranges_overlap(
                    diagnostic, output_ranges[index])) {
                return false;
            }
        }
    }
    const g1_controller_state_memory_range input_object = {
        &input, sizeof(input)
    };
    const g1_controller_state_memory_range output_object = {
        &output, sizeof(output)
    };
    if (g1_controller_state_ranges_overlap(
            input_object, output_object)) {
        return false;
    }
    for (int index = 0; index < input_count; ++index) {
        if (g1_controller_state_ranges_overlap(
                input_ranges[index], output_object) ||
            g1_controller_state_ranges_overlap(
                output_ranges[index], input_object)) {
            return false;
        }
    }
    if (!g1_controller_state_is_valid(input)) {
        return scene_error(
            error, error_capacity,
            "controller copy: invalid source semantics");
    }
    for (int destination = 0; destination < output_count; ++destination) {
        for (int source = 0; source < input_count; ++source) {
            if (g1_controller_state_ranges_overlap(
                    output_ranges[destination], input_ranges[source])) {
                return scene_error(
                    error, error_capacity,
                    "controller copy: source and destination storage alias");
            }
        }
    }

#define G1_COPY_ARRAY(name) \
    std::memcpy(output.name.data, input.name.data, \
                static_cast<std::size_t>(input.name.size) * \
                    sizeof(*input.name.data))
    G1_COPY_ARRAY(curr_bone_positions);
    G1_COPY_ARRAY(curr_bone_velocities);
    G1_COPY_ARRAY(trns_bone_positions);
    G1_COPY_ARRAY(trns_bone_velocities);
    G1_COPY_ARRAY(curr_bone_rotations);
    G1_COPY_ARRAY(trns_bone_rotations);
    G1_COPY_ARRAY(curr_bone_angular_velocities);
    G1_COPY_ARRAY(trns_bone_angular_velocities);
    G1_COPY_ARRAY(curr_bone_contacts);
    G1_COPY_ARRAY(trns_bone_contacts);
    G1_COPY_ARRAY(bone_positions);
    G1_COPY_ARRAY(bone_velocities);
    G1_COPY_ARRAY(bone_angular_velocities);
    G1_COPY_ARRAY(bone_rotations);
    G1_COPY_ARRAY(bone_offset_positions);
    G1_COPY_ARRAY(bone_offset_velocities);
    G1_COPY_ARRAY(bone_offset_angular_velocities);
    G1_COPY_ARRAY(bone_offset_rotations);
    G1_COPY_ARRAY(adjusted_bone_positions);
    G1_COPY_ARRAY(global_bone_positions);
    G1_COPY_ARRAY(global_bone_velocities);
    G1_COPY_ARRAY(adjusted_bone_rotations);
    G1_COPY_ARRAY(global_bone_rotations);
    G1_COPY_ARRAY(global_bone_angular_velocities);
    G1_COPY_ARRAY(global_bone_computed);
    G1_COPY_ARRAY(trajectory_desired_velocities);
    G1_COPY_ARRAY(trajectory_positions);
    G1_COPY_ARRAY(trajectory_velocities);
    G1_COPY_ARRAY(trajectory_accelerations);
    G1_COPY_ARRAY(trajectory_angular_velocities);
    G1_COPY_ARRAY(trajectory_desired_rotations);
    G1_COPY_ARRAY(trajectory_rotations);
    G1_COPY_ARRAY(contact_bones);
    G1_COPY_ARRAY(contact_states);
    G1_COPY_ARRAY(contact_locks);
    G1_COPY_ARRAY(contact_positions);
    G1_COPY_ARRAY(contact_velocities);
    G1_COPY_ARRAY(contact_points);
    G1_COPY_ARRAY(contact_targets);
    G1_COPY_ARRAY(contact_offset_positions);
    G1_COPY_ARRAY(contact_offset_velocities);
    G1_COPY_ARRAY(ik_bone_positions);
    G1_COPY_ARRAY(ik_bone_rotations);
    G1_COPY_ARRAY(ik_global_bone_positions);
    G1_COPY_ARRAY(ik_global_bone_rotations);
    G1_COPY_ARRAY(ik_candidate_bone_positions);
    G1_COPY_ARRAY(ik_candidate_bone_rotations);
    G1_COPY_ARRAY(ik_candidate_global_bone_positions);
    G1_COPY_ARRAY(ik_candidate_global_bone_rotations);
#undef G1_COPY_ARRAY

    output.frame_index = input.frame_index;
    output.scene_frame = input.scene_frame;
    output.search_time = input.search_time;
    output.search_timer = input.search_timer;
    output.force_search_timer = input.force_search_timer;
    output.transition_src_position = input.transition_src_position;
    output.transition_dst_position = input.transition_dst_position;
    output.transition_src_rotation = input.transition_src_rotation;
    output.transition_dst_rotation = input.transition_dst_rotation;
    output.desired_velocity = input.desired_velocity;
    output.desired_velocity_change_curr = input.desired_velocity_change_curr;
    output.desired_velocity_change_prev = input.desired_velocity_change_prev;
    output.desired_rotation = input.desired_rotation;
    output.desired_rotation_change_curr = input.desired_rotation_change_curr;
    output.desired_rotation_change_prev = input.desired_rotation_change_prev;
    output.desired_gait = input.desired_gait;
    output.desired_gait_velocity = input.desired_gait_velocity;
    output.simulation_position = input.simulation_position;
    output.simulation_velocity = input.simulation_velocity;
    output.simulation_acceleration = input.simulation_acceleration;
    output.simulation_rotation = input.simulation_rotation;
    output.simulation_angular_velocity = input.simulation_angular_velocity;
    output.command = input.command;
    output.footprint_status = input.footprint_status;
    output.footprint = input.footprint;
    output.ik = input.ik;
    output.ik_frame = input.ik_frame;
    output.ik_clearance = input.ik_clearance;
    output.ik_candidate_clearance = input.ik_candidate_clearance;
    output.ik_candidate_clearance_status =
        input.ik_candidate_clearance_status;
    output.ik_candidate_rejected = input.ik_candidate_rejected;
    output.support = input.support;
    output.support_observation_now = input.support_observation_now;
    output.traversal_speed_scale = input.traversal_speed_scale;
    output.traversal_speed_scale_velocity =
        input.traversal_speed_scale_velocity;
    output.blocked = input.blocked;
    output.walkability_class = input.walkability_class;
    output.blocked_distance = input.blocked_distance;
    output.blocked_point = input.blocked_point;
    output.route_index = input.route_index;
    output.route_waypoint = input.route_waypoint;
    output.route_frames = input.route_frames;
    output.camera_azimuth = input.camera_azimuth;
    output.camera_altitude = input.camera_altitude;
    output.camera_distance = input.camera_distance;
    output.searched = input.searched;
    output.transitioned = input.transitioned;
    output.incumbent_cost = input.incumbent_cost;
    output.selected_cost = input.selected_cost;
    output.selected_terrain_error = input.selected_terrain_error;
    output.adjustment_xz = input.adjustment_xz;
    output.adjustment_y = input.adjustment_y;
    output.clamp_xz = input.clamp_xz;
    output.clamp_y = input.clamp_y;
    return true;
}

static inline bool g1_controller_reset_sources_have_valid_shapes(
    const database& db,
    const terrain_support_set& support)
{
    const int frames = db.bone_positions.rows;
    if (frames <= 0 ||
        db.bone_positions.cols != G1_BoneCount ||
        db.bone_positions.data == NULL ||
        db.bone_velocities.rows != frames ||
        db.bone_velocities.cols != G1_BoneCount ||
        db.bone_velocities.data == NULL ||
        db.bone_rotations.rows != frames ||
        db.bone_rotations.cols != G1_BoneCount ||
        db.bone_rotations.data == NULL ||
        db.bone_angular_velocities.rows != frames ||
        db.bone_angular_velocities.cols != G1_BoneCount ||
        db.bone_angular_velocities.data == NULL ||
        db.contact_states.rows != frames ||
        db.contact_states.cols != 2 ||
        db.contact_states.data == NULL ||
        db.bone_parents.size != G1_BoneCount ||
        db.bone_parents.data == NULL ||
        db.range_starts.size <= 0 ||
        db.range_starts.size != db.range_stops.size ||
        db.range_starts.data == NULL || db.range_stops.data == NULL ||
        support.values.rows != frames || support.values.cols != 3 ||
        support.values.data == NULL) {
        return false;
    }
    int previous_stop = 0;
    for (int range = 0; range < db.range_starts.size; ++range) {
        const int start = db.range_starts(range);
        const int stop = db.range_stops(range);
        if (start != previous_stop || start < 0 ||
            start >= stop || stop > frames) {
            return false;
        }
        previous_stop = stop;
    }
    return previous_stop == frames;
}

static inline bool g1_controller_state_reset_configured(
    g1_controller_state& out,
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene,
    float initial_search_time,
    bool ik_enabled,
    float dt,
    float trajectory_sample_time,
    char* error,
    const int capacity)
{
    (void)ik_enabled;
    g1_controller_state_memory_range output_ranges[64] = {};
    int output_range_count = 0;
    if (!g1_controller_state_reset_output_ranges(
            out,
            output_ranges,
            output_range_count,
            static_cast<int>(sizeof(output_ranges) /
                             sizeof(output_ranges[0]))) ||
        g1_controller_state_diagnostic_aliases_state(
            out, error, capacity) ||
        g1_controller_reset_diagnostic_aliases_sources(
            db, support, scene, error, capacity)) {
        return false;
    }
    if (!g1_controller_reset_sources_have_valid_shapes(db, support) ||
        scene.terrain.version != 2 ||
        !terrain_heightfield_is_queryable(scene.terrain) ||
        !walkability_grid_matches_heightfield(
            scene.walkability, scene.terrain) ||
        !g1_ik_parent_topology_validate(db.bone_parents, error, capacity) ||
        !terrain_float_is_finite(initial_search_time) ||
        initial_search_time < 0.0f || initial_search_time > 10.0f ||
        !g1_dt_is_exact_25_hz(dt) ||
        !terrain_float_is_positive_normal(trajectory_sample_time))
    {
        return scene_error(
            error, capacity, "controller reset: invalid database/support shapes");
    }
    if (!g1_controller_state_reset_output_is_disjoint_from_sources(
            out,
            output_ranges,
            output_range_count,
            db,
            support,
            scene)) {
        return false;
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
    candidate.search_time = initial_search_time;
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

    candidate.trajectory_desired_velocities.resize(
        G1CommandTrajectorySampleCount);
    candidate.trajectory_desired_velocities.set(vec3());
    candidate.trajectory_positions.resize(G1CommandTrajectorySampleCount);
    candidate.trajectory_positions.set(candidate.simulation_position);
    candidate.trajectory_velocities.resize(G1CommandTrajectorySampleCount);
    candidate.trajectory_velocities.set(vec3());
    candidate.trajectory_accelerations.resize(G1CommandTrajectorySampleCount);
    candidate.trajectory_accelerations.set(vec3());
    candidate.trajectory_angular_velocities.resize(
        G1CommandTrajectorySampleCount);
    candidate.trajectory_angular_velocities.set(vec3());
    candidate.trajectory_desired_rotations.resize(
        G1CommandTrajectorySampleCount);
    candidate.trajectory_desired_rotations.set(candidate.simulation_rotation);
    candidate.trajectory_rotations.resize(G1CommandTrajectorySampleCount);
    candidate.trajectory_rotations.set(candidate.simulation_rotation);

    G1CommandIntent command_intent;
    command_intent.requested_velocity = vec3();
    command_intent.desired_heading = candidate.simulation_rotation;
    if (!g1_command_snapshot_build(
            candidate.command,
            command_intent,
            vec3(),
            candidate.trajectory_desired_velocities,
            candidate.trajectory_positions,
            candidate.trajectory_rotations,
            candidate.trajectory_desired_rotations,
            error,
            capacity)) {
        return false;
    }

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
    if (!g1_ik_checked_forward_kinematics(
            candidate.global_bone_positions,
            candidate.global_bone_rotations,
            candidate.adjusted_bone_positions,
            candidate.adjusted_bone_rotations,
            db.bone_parents,
            error,
            capacity)) {
        return false;
    }

    candidate.ik_bone_positions = candidate.adjusted_bone_positions;
    candidate.ik_bone_rotations = candidate.adjusted_bone_rotations;
    candidate.ik_global_bone_positions = candidate.global_bone_positions;
    candidate.ik_global_bone_rotations = candidate.global_bone_rotations;
    candidate.ik_candidate_bone_positions =
        candidate.adjusted_bone_positions;
    candidate.ik_candidate_bone_rotations =
        candidate.adjusted_bone_rotations;
    candidate.ik_candidate_global_bone_positions =
        candidate.global_bone_positions;
    candidate.ik_candidate_global_bone_rotations =
        candidate.global_bone_rotations;
    if (!g1_ik_state_reset(
            candidate.ik,
            candidate.adjusted_bone_positions,
            candidate.adjusted_bone_rotations,
            db.bone_parents,
            error,
            capacity)) {
        return false;
    }

    G1FootContactSchedule contact_schedule;
    if (!g1_foot_contact_schedule_build(
            contact_schedule,
            db.contact_states,
            db.range_starts,
            db.range_stops,
            candidate.frame_index,
            candidate.curr_bone_contacts(0),
            candidate.curr_bone_contacts(1),
            dt,
            trajectory_sample_time,
            error,
            capacity)) {
        return false;
    }
    candidate.footprint_status = g1_footprint_observe_v2(
        candidate.footprint,
        g1_footprint_budget(),
        scene.terrain,
        scene.walkability,
        candidate.command,
        contact_schedule,
        candidate.global_bone_positions,
        candidate.global_bone_rotations,
        error,
        capacity);
    if (candidate.footprint_status != G1FootprintOk ||
        candidate.footprint.blocked) {
        return scene_error(
            error,
            capacity,
            "controller reset: initial footprint is not certified");
    }

    const G1ClearanceStatus clearance_status =
        g1_measure_pose_clearance(
            candidate.ik_clearance,
            g1_pose_clearance_budget(),
            scene.terrain,
            candidate.ik_global_bone_positions,
            candidate.ik_global_bone_rotations,
            error,
            capacity);
    if (clearance_status != G1ClearanceOk) {
        return false;
    }
    if (candidate.ik_clearance.left.toe.lower_bound_m < -0.005 ||
        candidate.ik_clearance.left.foot.lower_bound_m < -0.005 ||
        candidate.ik_clearance.right.toe.lower_bound_m < -0.005 ||
        candidate.ik_clearance.right.foot.lower_bound_m < -0.005 ||
        candidate.ik_clearance.minimum.lower_bound_m < -0.01) {
        return scene_error(
            error,
            capacity,
            "controller reset: initial pose clearance margins are below "
            "threshold (left-toe %.9g left-foot %.9g right-toe %.9g "
            "right-foot %.9g minimum %.9g)",
            candidate.ik_clearance.left.toe.lower_bound_m,
            candidate.ik_clearance.left.foot.lower_bound_m,
            candidate.ik_clearance.right.toe.lower_bound_m,
            candidate.ik_clearance.right.foot.lower_bound_m,
            candidate.ik_clearance.minimum.lower_bound_m);
    }
    candidate.ik_candidate_clearance = candidate.ik_clearance;
    candidate.ik_candidate_clearance_status = G1ClearanceOk;
    candidate.ik_candidate_rejected = false;
    candidate.search_timer = candidate.search_time;
    candidate.force_search_timer = candidate.search_time;
    candidate.camera_azimuth = scene.metadata.spawn_yaw;
    candidate.blocked_distance = FLT_MAX;

    if (!g1_controller_state_is_valid(candidate)) {
        return scene_error(
            error,
            capacity,
            "controller reset: certified state is invalid");
    }

    g1_controller_state_swap(out, candidate);
    return true;
}

static inline bool g1_controller_state_reset(
    g1_controller_state& out,
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene,
    char* error,
    const int capacity)
{
    return g1_controller_state_reset_configured(
        out,
        db,
        support,
        scene,
        0.10f,
        false,
        1.0f / 25.0f,
        1.0f / 3.0f,
        error,
        capacity);
}
