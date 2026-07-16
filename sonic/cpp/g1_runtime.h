#pragma once

#include "g1_runtime_diagnostics.h"
#include "motion_match_log.h"

#include <cassert>
#include <cfloat>
#include <cmath>

enum g1_runtime_command_mode
{
    G1RuntimeVisual,
    G1RuntimeRoute,
    G1RuntimeDirect
};

struct g1_runtime_step_request
{
    g1_runtime_command_mode mode = G1RuntimeDirect;
    vec3 requested_velocity_holden;
    quat desired_heading_holden;
    bool matching_enabled = true;
};

struct g1_runtime_step_result
{
    int query_database_frame = -1;
    int selected_database_frame = -1;
    int query_range = -1;
    float query[31] = {};
    terrain_centerline_snapshot terrain;
    traversability_diagnostics traversal;
    motion_match_pose_diagnostic raw_selected;
    motion_match_pose_diagnostic inertialized;
    motion_match_pose_diagnostic projected;
};

struct g1_runtime_config
{
    float dt = 0.04f;
    float trajectory_sample_time = 1.0f / 3.0f;
    float inertialize_blending_halflife = 0.10f;
    float desired_velocity_change_threshold = 50.0f;
    float desired_rotation_change_threshold = 50.0f;
    float simulation_velocity_halflife = 0.27f;
    float simulation_rotation_halflife = 0.27f;
    float adjustment_position_halflife = 0.10f;
    float adjustment_rotation_halflife = 0.20f;
    float adjustment_position_max_ratio = 0.50f;
    float adjustment_rotation_max_ratio = 0.50f;
    float clamping_max_distance = 0.15f;
    float clamping_max_angle = 0.5f * PIf;
    bool synchronization_enabled = false;
    bool adjustment_enabled = true;
    bool adjustment_by_velocity_enabled = true;
    bool clamping_enabled = true;
};

static motion_match_pose_diagnostic g1_pose_diagnostic(
    const slice1d<vec3> local_positions,
    const slice1d<quat> local_rotations,
    const slice1d<int> parents,
    const heightfield& terrain)
{
    array1d<vec3> positions(local_positions.size);
    array1d<quat> rotations(local_rotations.size);
    forward_kinematics_full(
        positions, rotations, local_positions, local_rotations, parents);
    motion_match_pose_diagnostic out;
    out.hips_y = positions(G1_Hips).y;
    out.hips_clearance = positions(G1_Hips).y - heightfield_sample_v2(
        terrain, positions(G1_Hips).x, positions(G1_Hips).z);
    out.left_toe_clearance = positions(G1_LeftToe).y - heightfield_sample_v2(
        terrain, positions(G1_LeftToe).x, positions(G1_LeftToe).z);
    out.right_toe_clearance = positions(G1_RightToe).y - heightfield_sample_v2(
        terrain, positions(G1_RightToe).x, positions(G1_RightToe).z);
    out.minimum_clearance = out.hips_clearance;
    const int probes[] = {
        G1_LeftKnee, G1_RightKnee, G1_LeftAnkle, G1_RightAnkle,
        G1_LeftToe, G1_RightToe
    };
    for (int i = 0; i < 6; ++i) {
        const vec3 p = positions(probes[i]);
        out.minimum_clearance = minf(
            out.minimum_clearance,
            p.y - heightfield_sample_v2(terrain, p.x, p.z));
    }
    return out;
}

static bool g1_pose_diagnostic_is_finite(
    const motion_match_pose_diagnostic& diagnostic)
{
    return terrain_float_is_finite(diagnostic.hips_y) &&
           terrain_float_is_finite(diagnostic.hips_clearance) &&
           terrain_float_is_finite(diagnostic.left_toe_clearance) &&
           terrain_float_is_finite(diagnostic.right_toe_clearance) &&
           terrain_float_is_finite(diagnostic.minimum_clearance);
}

static int g1_active_range(const database& db, const int frame)
{
    for (int range = 0; range < db.nranges(); ++range)
    {
        if (frame >= db.range_starts(range) && frame < db.range_stops(range))
        {
            return range;
        }
    }
    return -1;
}

#if defined(__GNUC__) && !defined(__clang__)
#define G1_RUNTIME_COMPILER_BOUNDARY __attribute__((noinline, noclone))
#elif defined(__clang__)
#define G1_RUNTIME_COMPILER_BOUNDARY __attribute__((noinline))
#elif defined(_MSC_VER)
#define G1_RUNTIME_COMPILER_BOUNDARY __declspec(noinline)
#else
#define G1_RUNTIME_COMPILER_BOUNDARY
#endif

inline G1_RUNTIME_COMPILER_BOUNDARY void inertialize_root_adjust(
    vec3& offset_position,
    vec3& transition_src_position,
    quat& transition_src_rotation,
    vec3& transition_dst_position,
    quat& transition_dst_rotation,
    vec3& position,
    quat& rotation,
    const vec3 input_position,
    const quat input_rotation)
{
    vec3 position_difference = input_position - position;
    position = position_difference + position;
    transition_dst_position = position_difference + transition_dst_position;
    transition_src_position = transition_src_position + quat_mul_vec3(
        transition_src_rotation,
        quat_inv_mul_vec3(
            transition_dst_rotation,
            position - offset_position - transition_dst_position));
    transition_dst_position = position;
    offset_position = vec3();
    quat rotation_difference = quat_normalize(
        quat_mul_inv(input_rotation, rotation));
    rotation = quat_mul(rotation_difference, rotation);
    transition_dst_rotation = quat_mul(
        rotation_difference, transition_dst_rotation);
}

inline void inertialize_pose_reset(
    slice1d<vec3> bone_offset_positions,
    slice1d<vec3> bone_offset_velocities,
    slice1d<quat> bone_offset_rotations,
    slice1d<vec3> bone_offset_angular_velocities,
    vec3& transition_src_position,
    quat& transition_src_rotation,
    vec3& transition_dst_position,
    quat& transition_dst_rotation,
    const vec3 root_position,
    const quat root_rotation)
{
    bone_offset_positions.zero();
    bone_offset_velocities.zero();
    bone_offset_rotations.set(quat());
    bone_offset_angular_velocities.zero();
    transition_src_position = root_position;
    transition_src_rotation = root_rotation;
    transition_dst_position = vec3();
    transition_dst_rotation = quat();
}

inline G1_RUNTIME_COMPILER_BOUNDARY void inertialize_pose_transition(
    slice1d<vec3> bone_offset_positions,
    slice1d<vec3> bone_offset_velocities,
    slice1d<quat> bone_offset_rotations,
    slice1d<vec3> bone_offset_angular_velocities,
    vec3& transition_src_position,
    quat& transition_src_rotation,
    vec3& transition_dst_position,
    quat& transition_dst_rotation,
    const vec3 root_position,
    const vec3 root_velocity,
    const quat root_rotation,
    const vec3 root_angular_velocity,
    const slice1d<vec3> bone_src_positions,
    const slice1d<vec3> bone_src_velocities,
    const slice1d<quat> bone_src_rotations,
    const slice1d<vec3> bone_src_angular_velocities,
    const slice1d<vec3> bone_dst_positions,
    const slice1d<vec3> bone_dst_velocities,
    const slice1d<quat> bone_dst_rotations,
    const slice1d<vec3> bone_dst_angular_velocities)
{
    transition_dst_position = root_position;
    transition_dst_rotation = root_rotation;
    transition_src_position = bone_dst_positions(0);
    transition_src_rotation = bone_dst_rotations(0);
    vec3 world_space_dst_velocity = quat_mul_vec3(
        transition_dst_rotation,
        quat_inv_mul_vec3(
            transition_src_rotation, bone_dst_velocities(0)));
    vec3 world_space_dst_angular_velocity = quat_mul_vec3(
        transition_dst_rotation,
        quat_inv_mul_vec3(
            transition_src_rotation, bone_dst_angular_velocities(0)));
    inertialize_transition(
        bone_offset_positions(0),
        bone_offset_velocities(0),
        root_position,
        root_velocity,
        root_position,
        world_space_dst_velocity);
    inertialize_transition(
        bone_offset_rotations(0),
        bone_offset_angular_velocities(0),
        root_rotation,
        root_angular_velocity,
        root_rotation,
        world_space_dst_angular_velocity);
    for (int i = 1; i < bone_offset_positions.size; i++)
    {
        inertialize_transition(
            bone_offset_positions(i),
            bone_offset_velocities(i),
            bone_src_positions(i),
            bone_src_velocities(i),
            bone_dst_positions(i),
            bone_dst_velocities(i));
        inertialize_transition(
            bone_offset_rotations(i),
            bone_offset_angular_velocities(i),
            bone_src_rotations(i),
            bone_src_angular_velocities(i),
            bone_dst_rotations(i),
            bone_dst_angular_velocities(i));
    }
}

inline G1_RUNTIME_COMPILER_BOUNDARY void inertialize_pose_update(
    slice1d<vec3> bone_positions,
    slice1d<vec3> bone_velocities,
    slice1d<quat> bone_rotations,
    slice1d<vec3> bone_angular_velocities,
    slice1d<vec3> bone_offset_positions,
    slice1d<vec3> bone_offset_velocities,
    slice1d<quat> bone_offset_rotations,
    slice1d<vec3> bone_offset_angular_velocities,
    const slice1d<vec3> bone_input_positions,
    const slice1d<vec3> bone_input_velocities,
    const slice1d<quat> bone_input_rotations,
    const slice1d<vec3> bone_input_angular_velocities,
    const vec3 transition_src_position,
    const quat transition_src_rotation,
    const vec3 transition_dst_position,
    const quat transition_dst_rotation,
    const float halflife,
    const float dt)
{
    vec3 world_space_position = quat_mul_vec3(
        transition_dst_rotation,
        quat_inv_mul_vec3(
            transition_src_rotation,
            bone_input_positions(0) - transition_src_position)) +
        transition_dst_position;
    vec3 world_space_velocity = quat_mul_vec3(
        transition_dst_rotation,
        quat_inv_mul_vec3(
            transition_src_rotation, bone_input_velocities(0)));
    quat world_space_rotation = quat_normalize(quat_mul(
        transition_dst_rotation,
        quat_inv_mul(transition_src_rotation, bone_input_rotations(0))));
    vec3 world_space_angular_velocity = quat_mul_vec3(
        transition_dst_rotation,
        quat_inv_mul_vec3(
            transition_src_rotation,
            bone_input_angular_velocities(0)));
    inertialize_update(
        bone_positions(0),
        bone_velocities(0),
        bone_offset_positions(0),
        bone_offset_velocities(0),
        world_space_position,
        world_space_velocity,
        halflife,
        dt);
    inertialize_update(
        bone_rotations(0),
        bone_angular_velocities(0),
        bone_offset_rotations(0),
        bone_offset_angular_velocities(0),
        world_space_rotation,
        world_space_angular_velocity,
        halflife,
        dt);
    for (int i = 1; i < bone_positions.size; i++)
    {
        inertialize_update(
            bone_positions(i),
            bone_velocities(i),
            bone_offset_positions(i),
            bone_offset_velocities(i),
            bone_input_positions(i),
            bone_input_velocities(i),
            halflife,
            dt);
        inertialize_update(
            bone_rotations(i),
            bone_angular_velocities(i),
            bone_offset_rotations(i),
            bone_offset_angular_velocities(i),
            bone_input_rotations(i),
            bone_input_angular_velocities(i),
            halflife,
            dt);
    }
}

inline void query_copy_denormalized_feature(
    slice1d<float> query,
    int& offset,
    const int size,
    const slice1d<float> features,
    const slice1d<float> features_offset,
    const slice1d<float> features_scale)
{
    for (int i = 0; i < size; i++)
    {
        query(offset + i) = features(offset + i) *
            features_scale(offset + i) + features_offset(offset + i);
    }
    offset += size;
}

inline G1_RUNTIME_COMPILER_BOUNDARY void
query_compute_trajectory_position_feature(
    slice1d<float> query,
    int& offset,
    const vec3 root_position,
    const quat root_rotation,
    const slice1d<vec3> trajectory_positions)
{
    vec3 traj0 = quat_inv_mul_vec3(
        root_rotation, trajectory_positions(1) - root_position);
    vec3 traj1 = quat_inv_mul_vec3(
        root_rotation, trajectory_positions(2) - root_position);
    vec3 traj2 = quat_inv_mul_vec3(
        root_rotation, trajectory_positions(3) - root_position);
    query(offset + 0) = traj0.x;
    query(offset + 1) = traj0.z;
    query(offset + 2) = traj1.x;
    query(offset + 3) = traj1.z;
    query(offset + 4) = traj2.x;
    query(offset + 5) = traj2.z;
    offset += 6;
}

inline G1_RUNTIME_COMPILER_BOUNDARY void
query_compute_trajectory_direction_feature(
    slice1d<float> query,
    int& offset,
    const quat root_rotation,
    const slice1d<quat> trajectory_rotations)
{
    vec3 traj0 = quat_inv_mul_vec3(
        root_rotation,
        quat_mul_vec3(trajectory_rotations(1), vec3(0, 0, 1)));
    vec3 traj1 = quat_inv_mul_vec3(
        root_rotation,
        quat_mul_vec3(trajectory_rotations(2), vec3(0, 0, 1)));
    vec3 traj2 = quat_inv_mul_vec3(
        root_rotation,
        quat_mul_vec3(trajectory_rotations(3), vec3(0, 0, 1)));
    query(offset + 0) = traj0.x;
    query(offset + 1) = traj0.z;
    query(offset + 2) = traj1.x;
    query(offset + 3) = traj1.z;
    query(offset + 4) = traj2.x;
    query(offset + 5) = traj2.z;
    offset += 6;
}

inline G1_RUNTIME_COMPILER_BOUNDARY void simulation_positions_update(
    vec3& position,
    vec3& velocity,
    vec3& acceleration,
    const vec3 desired_velocity,
    const float halflife,
    const float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f;
    vec3 j0 = velocity - desired_velocity;
    vec3 j1 = acceleration + j0*y;
    float eydt = fast_negexpf(y*dt);
    vec3 position_prev = position;
    position = eydt*(((-j1)/(y*y)) + ((-j0 - j1*dt)/y)) +
        (j1/(y*y)) + j0/y + desired_velocity * dt + position_prev;
    velocity = eydt*(j0 + j1*dt) + desired_velocity;
    acceleration = eydt*(acceleration - j1*y*dt);
}

inline void simulation_rotations_update(
    quat& rotation,
    vec3& angular_velocity,
    const quat desired_rotation,
    const float halflife,
    const float dt)
{
    simple_spring_damper_exact(
        rotation,
        angular_velocity,
        desired_rotation,
        halflife, dt);
}

inline G1_RUNTIME_COMPILER_BOUNDARY void trajectory_positions_predict(
    slice1d<vec3> positions,
    slice1d<vec3> velocities,
    slice1d<vec3> accelerations,
    const vec3 position,
    const vec3 velocity,
    const vec3 acceleration,
    const slice1d<vec3> desired_velocities,
    const float halflife,
    const float dt)
{
    positions(0) = position;
    velocities(0) = velocity;
    accelerations(0) = acceleration;
    for (int i = 1; i < positions.size; i++)
    {
        positions(i) = positions(i-1);
        velocities(i) = velocities(i-1);
        accelerations(i) = accelerations(i-1);
        simulation_positions_update(
            positions(i),
            velocities(i),
            accelerations(i),
            desired_velocities(i),
            halflife,
            dt);
    }
}

inline G1_RUNTIME_COMPILER_BOUNDARY void trajectory_rotations_predict(
    slice1d<quat> rotations,
    slice1d<vec3> angular_velocities,
    const quat rotation,
    const vec3 angular_velocity,
    const slice1d<quat> desired_rotations,
    const float halflife,
    const float dt)
{
    rotations.set(rotation);
    angular_velocities.set(angular_velocity);
    for (int i = 1; i < rotations.size; i++)
    {
        simulation_rotations_update(
            rotations(i),
            angular_velocities(i),
            desired_rotations(i),
            halflife,
            i * dt);
    }
}

inline G1_RUNTIME_COMPILER_BOUNDARY quat adjust_character_rotation(
    const quat character_rotation,
    const quat simulation_rotation,
    const float halflife,
    const float dt)
{
    quat difference_rotation = quat_abs(quat_normalize(
        quat_mul_inv(simulation_rotation, character_rotation)));
    quat adjustment_rotation = damp_adjustment_exact(
        difference_rotation,
        halflife,
        dt);
    return quat_mul(adjustment_rotation, character_rotation);
}

inline G1_RUNTIME_COMPILER_BOUNDARY quat
adjust_character_rotation_by_velocity(
    const quat character_rotation,
    const vec3 character_angular_velocity,
    const quat simulation_rotation,
    const float max_adjustment_ratio,
    const float halflife,
    const float dt)
{
    quat adjustment_rotation = damp_adjustment_exact(
        quat_abs(quat_normalize(quat_mul_inv(
            simulation_rotation, character_rotation))),
        halflife,
        dt);
    float max_length = max_adjustment_ratio *
        length(character_angular_velocity) * dt;
    if (length(quat_to_scaled_angle_axis(adjustment_rotation)) > max_length)
    {
        adjustment_rotation = quat_from_scaled_angle_axis(max_length *
            normalize(quat_to_scaled_angle_axis(adjustment_rotation)));
    }
    return quat_mul(adjustment_rotation, character_rotation);
}

inline G1_RUNTIME_COMPILER_BOUNDARY quat clamp_character_rotation(
    const quat character_rotation,
    const quat simulation_rotation,
    const float max_angle)
{
    if (quat_angle_between(character_rotation, simulation_rotation) >
        max_angle)
    {
        quat diff = quat_abs(quat_mul_inv(
            character_rotation, simulation_rotation));
        float diff_angle;
        vec3 diff_axis;
        quat_to_angle_axis(diff, diff_angle, diff_axis);
        diff_angle = clampf(diff_angle, -max_angle, max_angle);
        return quat_mul(
            quat_from_angle_axis(diff_angle, diff_axis),
            simulation_rotation);
    }
    else
    {
        return character_rotation;
    }
}

#undef G1_RUNTIME_COMPILER_BOUNDARY

static inline bool g1_runtime_config_is_valid(
    const g1_runtime_config& config)
{
    const float positive[] = {
        config.dt,
        config.trajectory_sample_time,
        config.inertialize_blending_halflife,
        config.desired_velocity_change_threshold,
        config.desired_rotation_change_threshold,
        config.simulation_velocity_halflife,
        config.simulation_rotation_halflife,
        config.adjustment_position_halflife,
        config.adjustment_rotation_halflife,
        config.clamping_max_angle,
    };
    for (float value : positive) {
        if (!terrain_float_is_finite(value) || value <= 0.0f) return false;
    }
    const float nonnegative[] = {
        config.adjustment_position_max_ratio,
        config.adjustment_rotation_max_ratio,
        config.clamping_max_distance,
    };
    for (float value : nonnegative) {
        if (!terrain_float_is_finite(value) || value < 0.0f) return false;
    }
    return true;
}

static inline bool g1_runtime_request_is_valid(
    const g1_runtime_step_request& request)
{
    return request.mode >= G1RuntimeVisual &&
           request.mode <= G1RuntimeDirect &&
           g1_command_vec3_is_finite(request.requested_velocity_holden) &&
           ik_quat_is_unit(request.desired_heading_holden);
}

static inline bool g1_runtime_artifacts_are_valid(
    const g1_controller_state& state,
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene)
{
    return g1_controller_state_is_valid_shape(state) &&
           db.nframes() > 0 && db.nbones() == G1_BoneCount &&
           db.nfeatures() == 31 && db.nranges() > 0 &&
           db.ncontacts() == 2 &&
           db.bone_parents.size == G1_BoneCount &&
           db.bone_parents.data != NULL &&
           db.features.data != NULL &&
           db.features_offset.size == 31 &&
           db.features_offset.data != NULL &&
           db.features_scale.size == 31 &&
           db.features_scale.data != NULL &&
           db.terrain_features.rows == db.nframes() &&
           db.terrain_features.cols == 4 &&
           support.values.rows == db.nframes() &&
           support.values.cols == 3 &&
           state.frame_index >= 0 && state.frame_index < db.nframes() &&
           g1_active_range(db, state.frame_index) >= 0 &&
           scene.terrain.version == 2 &&
           terrain_heightfield_is_queryable(scene.terrain) &&
           walkability_grid_matches_heightfield(
               scene.walkability, scene.terrain);
}

template<typename PredictionBuilder>
static inline bool g1_runtime_step(
    g1_runtime_step_result& out,
    g1_controller_state& state,
    database& db,
    const terrain_support_set& support_rows,
    const scene_pack& active_scene,
    const g1_runtime_step_request& request,
    const g1_runtime_config& config,
    PredictionBuilder prediction_builder,
    char* error,
    int capacity)
{
    if (request.mode < G1RuntimeVisual || request.mode > G1RuntimeDirect) {
        return scene_error(
            error, capacity, "runtime step command mode is invalid");
    }
    if (!g1_command_vec3_is_finite(request.requested_velocity_holden)) {
        return scene_error(
            error, capacity, "runtime step requested velocity is non-finite");
    }
    if (!ik_quat_is_unit(request.desired_heading_holden)) {
        return scene_error(
            error, capacity, "runtime step requires a unit heading");
    }
    if (!terrain_float_is_finite(config.dt) || config.dt <= 0.0f) {
        return scene_error(
            error, capacity, "runtime step config dt must be positive");
    }
    if (!g1_runtime_config_is_valid(config)) {
        return scene_error(
            error, capacity, "runtime step config is invalid");
    }
    if (!g1_runtime_artifacts_are_valid(
            state, db, support_rows, active_scene)) {
        return scene_error(
            error, capacity, "runtime step artifact or state shape is invalid");
    }

    g1_controller_state next;
    if (!g1_controller_state_clone(next, state, error, capacity)) {
        return false;
    }
    g1_runtime_step_result result;
    const float dt = config.dt;
    const vec3 commanded_velocity = request.requested_velocity_holden;
    const quat desired_rotation_curr = request.desired_heading_holden;

    next.transitioned = false;
    next.adjustment_xz = 0.0f;
    next.adjustment_y = 0.0f;
    next.clamp_xz = 0.0f;
    next.clamp_y = 0.0f;

    traversability_diagnostics traversal = {};
    vec3 desired_velocity_curr = traversability_limit_command(
        next.traversal_speed_scale,
        next.traversal_speed_scale_velocity,
        traversal,
        active_scene.walkability,
        active_scene.terrain,
        next.simulation_position,
        commanded_velocity,
        dt);
    if (!g1_runtime_traversal_is_finite(traversal) ||
        !terrain_float_is_finite(desired_velocity_curr.x) ||
        !terrain_float_is_finite(desired_velocity_curr.y) ||
        !terrain_float_is_finite(desired_velocity_curr.z))
    {
        return scene_error(
            error,
            capacity,
            "command traversal produced non-finite diagnostics");
    }
    traversability_stop_blocked_planar_dynamics(
        traversal,
        next.simulation_velocity,
        next.simulation_acceleration);
    next.blocked = traversal.blocked;
    next.walkability_class = traversal.walkability_class;
    next.blocked_distance = traversal.distance;
    next.blocked_point = traversal.point;

    next.desired_velocity_change_prev = next.desired_velocity_change_curr;
    next.desired_velocity_change_curr =
        (desired_velocity_curr - next.desired_velocity) / dt;
    next.desired_velocity = desired_velocity_curr;
    g1_controller_state_seed_first_frame_desired_velocity(next);

    next.desired_rotation_change_prev = next.desired_rotation_change_curr;
    next.desired_rotation_change_curr = quat_to_scaled_angle_axis(quat_abs(
        quat_mul_inv(desired_rotation_curr, next.desired_rotation))) / dt;
    next.desired_rotation = desired_rotation_curr;

    bool force_search = false;
    if (next.force_search_timer <= 0.0f && (
        (length(next.desired_velocity_change_prev) >=
             config.desired_velocity_change_threshold &&
         length(next.desired_velocity_change_curr) <
             config.desired_velocity_change_threshold) ||
        (length(next.desired_rotation_change_prev) >=
             config.desired_rotation_change_threshold &&
         length(next.desired_rotation_change_curr) <
             config.desired_rotation_change_threshold)))
    {
        force_search = true;
        next.force_search_timer = next.search_time;
    }
    else if (next.force_search_timer > 0)
    {
        next.force_search_timer -= dt;
    }

    G1CommandIntent command_intent;
    command_intent.requested_velocity = commanded_velocity;
    command_intent.desired_heading = desired_rotation_curr;

    G1CommandFramePrediction frame_seed = {};
    frame_seed.command = next.command;
    for (int index = 0;
         index < G1CommandTrajectorySampleCount;
         ++index) {
        frame_seed.command.predicted_desired_velocities[index] =
            next.trajectory_desired_velocities(index);
        frame_seed.command.predicted_root_positions[index] =
            next.trajectory_positions(index);
        frame_seed.command.predicted_root_rotations[index] =
            next.trajectory_rotations(index);
        frame_seed.command.predicted_desired_headings[index] =
            next.trajectory_desired_rotations(index);
        frame_seed.predicted_root_velocities[index] =
            next.trajectory_velocities(index);
        frame_seed.predicted_root_accelerations[index] =
            next.trajectory_accelerations(index);
        frame_seed.predicted_root_angular_velocities[index] =
            next.trajectory_angular_velocities(index);
    }

    G1CommandFramePredictionRequest frame_request;
    frame_request.route_mode = request.mode != G1RuntimeVisual;
    frame_request.intent = command_intent;
    frame_request.applied_velocity = desired_velocity_curr;

    G1CommandFramePrediction frame_prediction;
    if (!prediction_builder(
            frame_prediction,
            frame_seed,
            frame_request,
            next,
            config,
            error,
            capacity)) {
        return false;
    }

    next.command = frame_prediction.command;
    for (int index = 0;
         index < G1CommandTrajectorySampleCount;
         ++index) {
        next.trajectory_desired_velocities(index) =
            frame_prediction.command.predicted_desired_velocities[index];
        next.trajectory_positions(index) =
            frame_prediction.command.predicted_root_positions[index];
        next.trajectory_rotations(index) =
            frame_prediction.command.predicted_root_rotations[index];
        next.trajectory_desired_rotations(index) =
            frame_prediction.command.predicted_desired_headings[index];
        next.trajectory_velocities(index) =
            frame_prediction.predicted_root_velocities[index];
        next.trajectory_accelerations(index) =
            frame_prediction.predicted_root_accelerations[index];
        next.trajectory_angular_velocities(index) =
            frame_prediction.predicted_root_angular_velocities[index];
    }
    force_search = force_search || frame_prediction.force_search;

    array1d<float> query(db.nfeatures());
    slice1d<float> query_features = db.features(next.frame_index);
    int offset = 0;
    query_copy_denormalized_feature(
        query, offset, 3, query_features,
        db.features_offset, db.features_scale);
    query_copy_denormalized_feature(
        query, offset, 3, query_features,
        db.features_offset, db.features_scale);
    query_copy_denormalized_feature(
        query, offset, 3, query_features,
        db.features_offset, db.features_scale);
    query_copy_denormalized_feature(
        query, offset, 3, query_features,
        db.features_offset, db.features_scale);
    query_copy_denormalized_feature(
        query, offset, 3, query_features,
        db.features_offset, db.features_scale);
    query_compute_trajectory_position_feature(
        query,
        offset,
        next.bone_positions(0),
        next.bone_rotations(0),
        next.trajectory_positions);
    query_compute_trajectory_direction_feature(
        query,
        offset,
        next.bone_rotations(0),
        next.trajectory_rotations);

    const int query_database_frame = next.frame_index;
    const int query_range = g1_active_range(db, query_database_frame);
    if (active_scene.terrain.version != 2 ||
        !terrain_heightfield_is_queryable(active_scene.terrain) ||
        !terrain_centerline_inputs_are_valid(
            next.bone_positions(0),
            next.trajectory_positions,
            next.trajectory_rotations)) {
        return scene_error(
            error, capacity, "terrain centerline inputs are invalid");
    }
    terrain_centerline_snapshot terrain_query_snapshot = {};
    terrain_centerline_snapshot_compute_v2(
        terrain_query_snapshot,
        active_scene.terrain,
        next.bone_positions(0),
        next.trajectory_positions,
        next.trajectory_rotations);
    if (!terrain_centerline_snapshot_apply_walkability_v2(
            terrain_query_snapshot,
            active_scene.terrain,
            active_scene.walkability,
            next.bone_positions(0),
            next.simulation_position,
            0.20f)) {
        return scene_error(
            error,
            capacity,
            "terrain centerline walkability mask is invalid");
    }
    for (int terrain_feature = 0; terrain_feature < 4; ++terrain_feature) {
        const vec3 point = terrain_query_snapshot.points[terrain_feature];
        if (!terrain_float_is_finite(
                terrain_query_snapshot.values[terrain_feature]) ||
            !terrain_float_is_finite(point.x) ||
            !terrain_float_is_finite(point.y) ||
            !terrain_float_is_finite(point.z)) {
            return scene_error(
                error,
                capacity,
                "terrain query snapshot %d is invalid",
                terrain_feature);
        }
    }
    for (int terrain_feature = 0; terrain_feature < 4; ++terrain_feature)
    {
        query(offset++) = terrain_query_snapshot.values[terrain_feature];
    }

    assert(offset == db.nfeatures());
    if (!motion_match_query_is_finite_31d(query)) {
        return scene_error(
            error, capacity, "expected exactly 31 finite query values");
    }

    bool end_of_anim = database_trajectory_index_clamp(
        db, next.frame_index, 1) == next.frame_index;
    next.searched = request.matching_enabled &&
        (force_search || next.search_timer <= 0.0f || end_of_anim);
    next.incumbent_cost = 0.0f;
    next.selected_cost = 0.0f;
    next.selected_terrain_error = 0.0f;
    next.incumbent_cost = end_of_anim
        ? FLT_MAX : database_frame_cost(db, next.frame_index, query);
    next.selected_cost = next.incumbent_cost;
    next.selected_terrain_error = database_raw_terrain_error(
        db, next.frame_index, query);
    int selected_database_frame = query_database_frame;

    if (next.searched)
    {
        const int prior_index = next.frame_index;
        int best_index = end_of_anim ? -1 : prior_index;
        float best_cost = FLT_MAX;
        const float transition_cost =
            g1_idle_match_transition_cost(
                traversal.commanded_speed,
                walkability_xz_length(next.simulation_velocity));
        database_search(
            best_index,
            best_cost,
            db,
            query,
            transition_cost);
        selected_database_frame = best_index;
        if (best_index != prior_index) {
            next.selected_cost = best_cost;
            next.selected_terrain_error = database_raw_terrain_error(
                db, best_index, query);
        }
        if (best_index != prior_index)
        {
            next.transitioned = true;
            next.trns_bone_positions = db.bone_positions(best_index);
            next.trns_bone_velocities = db.bone_velocities(best_index);
            next.trns_bone_rotations = db.bone_rotations(best_index);
            next.trns_bone_angular_velocities =
                db.bone_angular_velocities(best_index);
            inertialize_pose_transition(
                next.bone_offset_positions,
                next.bone_offset_velocities,
                next.bone_offset_rotations,
                next.bone_offset_angular_velocities,
                next.transition_src_position,
                next.transition_src_rotation,
                next.transition_dst_position,
                next.transition_dst_rotation,
                next.bone_positions(0),
                next.bone_velocities(0),
                next.bone_rotations(0),
                next.bone_angular_velocities(0),
                next.curr_bone_positions,
                next.curr_bone_velocities,
                next.curr_bone_rotations,
                next.curr_bone_angular_velocities,
                next.trns_bone_positions,
                next.trns_bone_velocities,
                next.trns_bone_rotations,
                next.trns_bone_angular_velocities);
            next.frame_index = best_index;
        }
        next.search_timer = next.search_time;
    }

    next.search_timer -= dt;
    next.frame_index = database_trajectory_index_clamp(
        db, next.frame_index, 1);
    next.curr_bone_positions = db.bone_positions(next.frame_index);
    next.curr_bone_velocities = db.bone_velocities(next.frame_index);
    next.curr_bone_rotations = db.bone_rotations(next.frame_index);
    next.curr_bone_angular_velocities =
        db.bone_angular_velocities(next.frame_index);
    next.curr_bone_contacts = db.contact_states(next.frame_index);

    inertialize_pose_update(
        next.bone_positions,
        next.bone_velocities,
        next.bone_rotations,
        next.bone_angular_velocities,
        next.bone_offset_positions,
        next.bone_offset_velocities,
        next.bone_offset_rotations,
        next.bone_offset_angular_velocities,
        next.curr_bone_positions,
        next.curr_bone_velocities,
        next.curr_bone_rotations,
        next.curr_bone_angular_velocities,
        next.transition_src_position,
        next.transition_src_rotation,
        next.transition_dst_position,
        next.transition_dst_rotation,
        config.inertialize_blending_halflife,
        dt);

    array1d<vec3> raw_selected_positions(next.curr_bone_positions);
    array1d<quat> raw_selected_rotations(next.curr_bone_rotations);
    raw_selected_positions(0) = next.bone_positions(0);
    raw_selected_rotations(0) = next.bone_rotations(0);
    const motion_match_pose_diagnostic raw_selected_diagnostic =
        g1_pose_diagnostic(
            raw_selected_positions,
            raw_selected_rotations,
            db.bone_parents,
            active_scene.terrain);
    const motion_match_pose_diagnostic inertialized_diagnostic =
        g1_pose_diagnostic(
            next.bone_positions,
            next.bone_rotations,
            db.bone_parents,
            active_scene.terrain);
    if (!g1_pose_diagnostic_is_finite(raw_selected_diagnostic) ||
        !g1_pose_diagnostic_is_finite(inertialized_diagnostic) ||
        !terrain_float_is_finite(next.incumbent_cost) ||
        !terrain_float_is_finite(next.selected_cost) ||
        !terrain_float_is_finite(next.selected_terrain_error))
    {
        return scene_error(
            error,
            capacity,
            "motion costs or pose diagnostics are non-finite");
    }

    const vec3 simulation_before = next.simulation_position;
    simulation_positions_update(
        next.simulation_position,
        next.simulation_velocity,
        next.simulation_acceleration,
        next.desired_velocity,
        config.simulation_velocity_halflife,
        dt);
    const walkability_sweep_result integrated_traversal =
        traversability_preflight_step(
            simulation_before,
            next.simulation_position,
            next.simulation_velocity,
            next.simulation_acceleration,
            active_scene.walkability,
            active_scene.terrain,
            0.20f);
    if (integrated_traversal.blocked) {
        traversability_apply_sweep_result(
            simulation_before,
            next.simulation_position,
            next.simulation_velocity,
            next.simulation_acceleration,
            traversal,
            integrated_traversal);
    }
    walkability_reason current_reason = walkability_clear;
    const int current_walkability_class = walkability_footprint_class(
        active_scene.walkability,
        active_scene.terrain,
        next.simulation_position.x,
        next.simulation_position.z,
        0.20f,
        current_reason);
    next.blocked = traversal.blocked;
    next.blocked_distance = traversal.distance;
    next.blocked_point = traversal.point;
    next.walkability_class = current_walkability_class;
    if (!g1_runtime_traversal_is_finite(traversal) ||
        next.walkability_class < 0 || next.walkability_class > 2 ||
        !terrain_float_is_finite(next.simulation_position.x) ||
        !terrain_float_is_finite(next.simulation_position.y) ||
        !terrain_float_is_finite(next.simulation_position.z))
    {
        return scene_error(
            error,
            capacity,
            "integrated traversal or simulation state is invalid");
    }

    simulation_rotations_update(
        next.simulation_rotation,
        next.simulation_angular_velocity,
        next.desired_rotation,
        config.simulation_rotation_halflife,
        dt);

    forward_kinematics_full(
        next.global_bone_positions,
        next.global_bone_rotations,
        next.bone_positions,
        next.bone_rotations,
        db.bone_parents);
    if (!support_observation_build_walkable(
            next.support_observation_now,
            support_rows,
            next.frame_index,
            active_scene.terrain,
            active_scene.walkability,
            next.global_bone_positions(G1_Simulation),
            next.global_bone_positions(G1_LeftToe),
            next.global_bone_positions(G1_RightToe),
            next.curr_bone_contacts(0),
            next.curr_bone_contacts(1),
            error,
            capacity) ||
        !support_frame_update(
            next.support,
            next.support_observation_now,
            next.transitioned,
            dt,
            error,
            capacity))
    {
        return false;
    }
    if (!support_observation_is_finite(next.support_observation_now) ||
        !g1_runtime_support_state_is_finite(next.support))
    {
        return scene_error(
            error,
            capacity,
            "support observation or state is non-finite");
    }

    if (config.synchronization_enabled)
    {
        const float synchronization_data_factor = 1.0f;
        vec3 synchronized_position = lerp(
            next.simulation_position,
            next.bone_positions(0),
            synchronization_data_factor);
        quat synchronized_rotation = quat_nlerp_shortest(
            next.simulation_rotation,
            next.bone_rotations(0),
            synchronization_data_factor);
        next.simulation_position = synchronized_position;
        next.simulation_rotation = synchronized_rotation;
        inertialize_root_adjust(
            next.bone_offset_positions(0),
            next.transition_src_position,
            next.transition_src_rotation,
            next.transition_dst_position,
            next.transition_dst_rotation,
            next.bone_positions(0),
            next.bone_rotations(0),
            synchronized_position,
            synchronized_rotation);
    }

    if (!config.synchronization_enabled && config.adjustment_enabled)
    {
        const vec3 before_adjustment = next.bone_positions(G1_Simulation);
        vec3 adjusted_position;
        quat adjusted_rotation = next.bone_rotations(G1_Simulation);
        if (config.adjustment_by_velocity_enabled)
        {
            adjusted_position = horizontal_adjust_character_position_by_velocity(
                before_adjustment,
                next.bone_velocities(G1_Simulation),
                next.simulation_position,
                config.adjustment_position_max_ratio,
                config.adjustment_position_halflife,
                dt);
            adjusted_rotation = adjust_character_rotation_by_velocity(
                next.bone_rotations(G1_Simulation),
                next.bone_angular_velocities(G1_Simulation),
                next.simulation_rotation,
                config.adjustment_rotation_max_ratio,
                config.adjustment_rotation_halflife,
                dt);
        }
        else
        {
            adjusted_position = horizontal_adjust_character_position(
                before_adjustment,
                next.simulation_position,
                config.adjustment_position_halflife,
                dt);
            adjusted_rotation = adjust_character_rotation(
                next.bone_rotations(G1_Simulation),
                next.simulation_rotation,
                config.adjustment_rotation_halflife,
                dt);
        }
        next.adjustment_xz = horizontal_length(
            adjusted_position - before_adjustment);
        next.adjustment_y = adjusted_position.y - before_adjustment.y;
        inertialize_root_adjust(
            next.bone_offset_positions(G1_Simulation),
            next.transition_src_position,
            next.transition_src_rotation,
            next.transition_dst_position,
            next.transition_dst_rotation,
            next.bone_positions(G1_Simulation),
            next.bone_rotations(G1_Simulation),
            adjusted_position,
            adjusted_rotation);
    }

    if (!config.synchronization_enabled && config.clamping_enabled)
    {
        const vec3 before_clamp = next.bone_positions(G1_Simulation);
        vec3 adjusted_position = horizontal_clamp_character_position(
            before_clamp,
            next.simulation_position,
            config.clamping_max_distance);
        quat adjusted_rotation = next.bone_rotations(G1_Simulation);
        adjusted_rotation = clamp_character_rotation(
            adjusted_rotation,
            next.simulation_rotation,
            config.clamping_max_angle);
        next.clamp_xz = horizontal_length(
            adjusted_position - before_clamp);
        next.clamp_y = adjusted_position.y - before_clamp.y;
        inertialize_root_adjust(
            next.bone_offset_positions(G1_Simulation),
            next.transition_src_position,
            next.transition_src_rotation,
            next.transition_dst_position,
            next.transition_dst_rotation,
            next.bone_positions(G1_Simulation),
            next.bone_rotations(G1_Simulation),
            adjusted_position,
            adjusted_rotation);
    }

    if (next.adjustment_y != 0.0f || next.clamp_y != 0.0f)
    {
        return scene_error(
            error, capacity, "horizontal-root invariant failed");
    }

    next.adjusted_bone_rotations = next.bone_rotations;
    support_pose_apply(
        next.adjusted_bone_positions,
        next.bone_positions,
        next.support.height);
    forward_kinematics_full(
        next.global_bone_positions,
        next.global_bone_rotations,
        next.adjusted_bone_positions,
        next.adjusted_bone_rotations,
        db.bone_parents);
    const motion_match_pose_diagnostic projected_diagnostic =
        g1_pose_diagnostic(
            next.adjusted_bone_positions,
            next.adjusted_bone_rotations,
            db.bone_parents,
            active_scene.terrain);
    if (!g1_pose_diagnostic_is_finite(projected_diagnostic)) {
        return scene_error(
            error,
            capacity,
            "final support-retargeted pose diagnostic is non-finite");
    }

    result.query_database_frame = query_database_frame;
    result.selected_database_frame = selected_database_frame;
    result.query_range = query_range;
    for (int index = 0; index < 31; ++index) {
        result.query[index] = query(index);
    }
    result.terrain = terrain_query_snapshot;
    result.traversal = traversal;
    result.raw_selected = raw_selected_diagnostic;
    result.inertialized = inertialized_diagnostic;
    result.projected = projected_diagnostic;
    g1_controller_state_swap(state, next);
    out = result;
    return true;
}

static inline bool g1_runtime_step(
    g1_runtime_step_result& out,
    g1_controller_state& state,
    database& db,
    const terrain_support_set& support_rows,
    const scene_pack& active_scene,
    const g1_runtime_step_request& request,
    const g1_runtime_config& config,
    char* error,
    int capacity)
{
    if (request.mode != G1RuntimeDirect) {
        return scene_error(
            error,
            capacity,
            "renderer-free default predictor requires direct mode");
    }
    const auto direct_prediction_builder =
        [&](G1CommandFramePrediction& frame_prediction,
            const G1CommandFramePrediction& frame_seed,
            const G1CommandFramePredictionRequest& frame_request,
            g1_controller_state& next,
            const g1_runtime_config& runtime_config,
            char* prediction_error,
            int prediction_capacity)
        {
            G1CommandFramePredictionRequest direct_request = frame_request;
            direct_request.route_mode = true;
            direct_request.heading_override.active = true;
            direct_request.heading_override.heading =
                direct_request.intent.desired_heading;
            return g1_command_frame_prediction_build(
                frame_prediction,
                frame_seed,
                direct_request,
                [&](slice1d<vec3> desired_velocities,
                    bool& route_force_search,
                    char*,
                    int) {
                    for (int index = 0;
                         index < G1CommandTrajectorySampleCount;
                         ++index) {
                        desired_velocities(index) =
                            direct_request.intent.requested_velocity;
                    }
                    route_force_search = false;
                    return true;
                },
                [](slice1d<quat>,
                   const slice1d<vec3>,
                   char*,
                   int) {
                    return true;
                },
                [&](slice1d<quat> rotations,
                    slice1d<vec3> angular_velocities,
                    const slice1d<quat> desired_rotations,
                    const slice1d<vec3>,
                    char*,
                    int) {
                    trajectory_rotations_predict(
                        rotations,
                        angular_velocities,
                        next.simulation_rotation,
                        next.simulation_angular_velocity,
                        desired_rotations,
                        runtime_config.simulation_rotation_halflife,
                        runtime_config.trajectory_sample_time);
                    return true;
                },
                [](slice1d<vec3>,
                   const slice1d<quat>,
                   char*,
                   int) {
                    return true;
                },
                [&](slice1d<vec3> positions,
                    slice1d<vec3> velocities,
                    slice1d<vec3> accelerations,
                    const slice1d<vec3> desired_velocities,
                    char*,
                    int) {
                    trajectory_positions_predict(
                        positions,
                        velocities,
                        accelerations,
                        next.simulation_position,
                        next.simulation_velocity,
                        next.simulation_acceleration,
                        desired_velocities,
                        runtime_config.simulation_velocity_halflife,
                        runtime_config.trajectory_sample_time);
                    return true;
                },
                prediction_error,
                prediction_capacity);
        };
    return g1_runtime_step(
        out,
        state,
        db,
        support_rows,
        active_scene,
        request,
        config,
        direct_prediction_builder,
        error,
        capacity);
}
