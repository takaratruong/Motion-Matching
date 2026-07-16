#include "scene_switch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static void check(bool value, const char* message)
{
    if (!value)
    {
        std::fprintf(stderr, "scene switch test failed: %s\n", message);
        std::exit(1);
    }
}

struct fake_model
{
    int id = 0;
};

enum fake_model_mode
{
    fake_model_ready,
    fake_model_unallocated_failure,
    fake_model_allocated_failure
};

struct fake_models
{
    int loads = 0;
    int unloads = 0;
    int live = 0;
    int last_loaded_id = 0;
    int last_unloaded_id = 0;
    fake_model_mode mode = fake_model_ready;
    std::vector<int> unloaded_ids;
};

static void fixture_db(database& db)
{
    static const int parents[G1_BoneCount] = {
        -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
    };
    static constexpr int frames = 32;
    db.bone_positions.resize(frames, G1_BoneCount);
    db.bone_velocities.resize(frames, G1_BoneCount);
    db.bone_rotations.resize(frames, G1_BoneCount);
    db.bone_angular_velocities.resize(frames, G1_BoneCount);
    db.bone_parents.resize(G1_BoneCount);
    db.contact_states.resize(frames, 2);
    db.range_starts.resize(1);
    db.range_stops.resize(1);
    db.features.resize(frames, 31);
    db.features_offset.resize(31);
    db.features_scale.resize(31);
    db.terrain_features.resize(frames, 4);
    db.bone_positions.set(vec3());
    db.bone_velocities.set(vec3());
    db.bone_rotations.set(quat());
    db.bone_angular_velocities.set(vec3());
    db.contact_states.zero();
    db.features.zero();
    db.features_offset.zero();
    db.features_scale.set(1.0f);
    db.terrain_features.zero();
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        db.bone_parents(bone) = parents[bone];
    }
    for (int frame = 0; frame < frames; ++frame) {
        db.bone_positions(frame, G1_LeftKnee) =
            vec3(-0.078273f, -0.17734f, -0.0021489f);
        db.bone_positions(frame, G1_LeftAnkle) =
            vec3(0.0f, -0.30001f, +0.000094445f);
        db.bone_positions(frame, G1_LeftToe) =
            vec3(0.0f, -0.017558f, 0.0f);
        db.bone_positions(frame, G1_RightKnee) =
            vec3(-0.078273f, -0.17734f, +0.0021489f);
        db.bone_positions(frame, G1_RightAnkle) =
            vec3(0.0f, -0.30001f, -0.000094445f);
        db.bone_positions(frame, G1_RightToe) =
            vec3(0.0f, -0.017558f, 0.0f);
    }
    db.range_starts(0) = 0;
    db.range_stops(0) = frames;
    database_build_bounds(db);
}

static scene_pack fixture_scene(const char* id, float x)
{
    scene_pack scene;
    scene.metadata.id = id;
    scene.metadata.spawn_position = vec3(x, 0.0f, 2.0f);
    scene.metadata.spawn_yaw = 0.37f;
    scene.metadata.playable_bounds = {-3.0f, -3.0f, 7.0f, 7.0f};
    scene_route route;
    route.id = "pair-route";
    route.expected_outcome = "traverse";
    route.walkability_class = 1;
    route.waypoints_xz = {{x, 2.0f}, {x + 1.0f, 2.0f}};
    scene.metadata.routes.push_back(route);
    scene.terrain.version = 2;
    scene.terrain.nx = 41;
    scene.terrain.nz = 41;
    scene.terrain.origin_x = -4.0f;
    scene.terrain.origin_z = -4.0f;
    scene.terrain.cell_size = 0.25f;
    scene.terrain.exterior_height = -10.0f;
    scene.terrain.heights.resize(41 * 41);
    scene.terrain.heights.zero();
    scene.walkability.nx = 41;
    scene.walkability.nz = 41;
    scene.walkability.cells.resize(41 * 41);
    scene.walkability.cells.set(1);
    scene.mesh_path = std::string(id) + ".obj";
    return scene;
}

#define G1_FOR_EACH_STATE_ARRAY(MACRO) \
    MACRO(curr_bone_positions); \
    MACRO(curr_bone_velocities); \
    MACRO(trns_bone_positions); \
    MACRO(trns_bone_velocities); \
    MACRO(curr_bone_rotations); \
    MACRO(trns_bone_rotations); \
    MACRO(curr_bone_angular_velocities); \
    MACRO(trns_bone_angular_velocities); \
    MACRO(curr_bone_contacts); \
    MACRO(trns_bone_contacts); \
    MACRO(bone_positions); \
    MACRO(bone_velocities); \
    MACRO(bone_angular_velocities); \
    MACRO(bone_rotations); \
    MACRO(bone_offset_positions); \
    MACRO(bone_offset_velocities); \
    MACRO(bone_offset_angular_velocities); \
    MACRO(bone_offset_rotations); \
    MACRO(adjusted_bone_positions); \
    MACRO(global_bone_positions); \
    MACRO(global_bone_velocities); \
    MACRO(adjusted_bone_rotations); \
    MACRO(global_bone_rotations); \
    MACRO(global_bone_angular_velocities); \
    MACRO(global_bone_computed); \
    MACRO(trajectory_desired_velocities); \
    MACRO(trajectory_positions); \
    MACRO(trajectory_velocities); \
    MACRO(trajectory_accelerations); \
    MACRO(trajectory_angular_velocities); \
    MACRO(trajectory_desired_rotations); \
    MACRO(trajectory_rotations); \
    MACRO(contact_bones); \
    MACRO(contact_states); \
    MACRO(contact_locks); \
    MACRO(contact_positions); \
    MACRO(contact_velocities); \
    MACRO(contact_points); \
    MACRO(contact_targets); \
    MACRO(contact_offset_positions); \
    MACRO(contact_offset_velocities); \
    MACRO(ik_bone_positions); \
    MACRO(ik_bone_rotations); \
    MACRO(ik_global_bone_positions); \
    MACRO(ik_global_bone_rotations); \
    MACRO(ik_candidate_bone_positions); \
    MACRO(ik_candidate_bone_rotations); \
    MACRO(ik_candidate_global_bone_positions); \
    MACRO(ik_candidate_global_bone_rotations)

static void logical_hash_word(uint64_t& hash, uint64_t value)
{
    for (int byte = 0; byte < 8; ++byte) {
        hash ^= (value >> (byte * 8)) & UINT64_C(0xff);
        hash *= UINT64_C(1099511628211);
    }
}

static void logical_hash_value(uint64_t& hash, bool value)
{
    logical_hash_word(hash, value ? 1U : 0U);
}

static void logical_hash_value(uint64_t& hash, int value)
{
    logical_hash_word(hash, static_cast<uint32_t>(value));
}

static void logical_hash_value(uint64_t& hash, uint32_t value)
{
    logical_hash_word(hash, value);
}

static void logical_hash_value(uint64_t& hash, float value)
{
    logical_hash_word(hash, terrain_float_bits(value));
}

static void logical_hash_value(uint64_t& hash, double value)
{
    uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    logical_hash_word(hash, bits);
}

static void logical_hash_value(uint64_t& hash, vec3 value)
{
    logical_hash_value(hash, value.x);
    logical_hash_value(hash, value.y);
    logical_hash_value(hash, value.z);
}

static void logical_hash_value(uint64_t& hash, quat value)
{
    logical_hash_value(hash, value.w);
    logical_hash_value(hash, value.x);
    logical_hash_value(hash, value.y);
    logical_hash_value(hash, value.z);
}

template<class T>
static void logical_hash_array(uint64_t& hash, const array1d<T>& values)
{
    logical_hash_value(hash, values.size);
    for (int index = 0; index < values.size; ++index) {
        logical_hash_value(hash, values(index));
    }
}

static void logical_hash_surface(
    uint64_t& hash, const G1SurfaceSample& value)
{
    logical_hash_value(hash, value.height);
    logical_hash_value(hash, value.normal);
}

static void logical_hash_clearance_work(
    uint64_t& hash, const G1ClearanceWork& value)
{
    logical_hash_value(hash, value.point_queries);
    logical_hash_value(hash, value.cells_visited);
    logical_hash_value(hash, value.primitive_triangle_pairs);
    logical_hash_value(hash, value.face_patches);
    logical_hash_value(hash, value.candidate_tests);
    logical_hash_value(hash, value.subdivision_nodes);
}

static void logical_hash_clearance_result(
    uint64_t& hash, const G1ClearanceResult& value)
{
    logical_hash_value(hash, value.lower_bound_m);
    logical_hash_value(hash, value.witness_upper_m);
    logical_hash_value(hash, value.witness.body_x);
    logical_hash_value(hash, value.witness.body_y);
    logical_hash_value(hash, value.witness.body_z);
    logical_hash_value(hash, value.witness.surface_x);
    logical_hash_value(hash, value.witness.surface_y);
    logical_hash_value(hash, value.witness.surface_z);
    logical_hash_value(hash, value.witness.segment_parameter);
    logical_hash_value(hash, value.witness.terrain_weight_0);
    logical_hash_value(hash, value.witness.terrain_weight_1);
    logical_hash_value(hash, value.witness.terrain_weight_2);
    logical_hash_value(hash, value.witness.primitive_index);
    logical_hash_value(hash, value.witness.cell_x);
    logical_hash_value(hash, value.witness.cell_z);
    logical_hash_value(hash, value.witness.terrain_triangle_index);
    logical_hash_value(hash, value.witness.patch_index);
    logical_hash_value(hash, value.witness.candidate_kind);
    logical_hash_value(hash, value.witness.candidate_subindex);
    logical_hash_clearance_work(hash, value.work);
}

static void logical_hash_leg_clearance(
    uint64_t& hash, const G1LegClearance& value)
{
    logical_hash_clearance_result(hash, value.knee);
    logical_hash_clearance_result(hash, value.ankle);
    logical_hash_clearance_result(hash, value.toe);
    logical_hash_clearance_result(hash, value.foot);
    logical_hash_clearance_result(hash, value.thigh);
    logical_hash_clearance_result(hash, value.shin);
    logical_hash_clearance_result(hash, value.minimum);
}

static void logical_hash_pose_clearance(
    uint64_t& hash, const G1PoseClearance& value)
{
    logical_hash_clearance_result(hash, value.hips);
    logical_hash_leg_clearance(hash, value.left);
    logical_hash_leg_clearance(hash, value.right);
    logical_hash_clearance_result(hash, value.minimum);
}

static void logical_hash_intent(uint64_t& hash, const G1CommandIntent& value)
{
    logical_hash_value(hash, value.requested_velocity);
    logical_hash_value(hash, value.desired_heading);
}

static void logical_hash_command(
    uint64_t& hash, const G1CommandSnapshot& value)
{
    logical_hash_intent(hash, value.intent);
    logical_hash_value(hash, value.applied_velocity);
    for (int sample = 0;
         sample < G1CommandTrajectorySampleCount;
         ++sample) {
        logical_hash_value(hash, value.predicted_desired_velocities[sample]);
        logical_hash_value(hash, value.predicted_root_positions[sample]);
        logical_hash_value(hash, value.predicted_root_rotations[sample]);
        logical_hash_value(hash, value.predicted_desired_headings[sample]);
    }
}

static void logical_hash_support(
    uint64_t& hash, const support_frame_state& value)
{
    logical_hash_value(hash, value.height);
    logical_hash_value(hash, value.velocity);
    logical_hash_value(hash, value.nominal_height);
    logical_hash_value(hash, value.nominal_velocity);
    logical_hash_value(hash, value.offset_height);
    logical_hash_value(hash, value.offset_velocity);
    logical_hash_value(hash, value.airborne_frames);
    logical_hash_value(hash, static_cast<int>(value.source));
    logical_hash_value(hash, value.initialized);
}

static void logical_hash_support_observation(
    uint64_t& hash, const support_observation& value)
{
    for (int index = 0; index < 3; ++index) {
        logical_hash_value(hash, value.source_height[index]);
        logical_hash_value(hash, value.runtime_height[index]);
        logical_hash_value(hash, value.delta[index]);
    }
    logical_hash_value(hash, value.contact[0]);
    logical_hash_value(hash, value.contact[1]);
}

static void logical_hash_footprint(
    uint64_t& hash, const G1FootprintObservation& value)
{
    logical_hash_surface(hash, value.root_surface);
    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        const G1FootprintFootObservation& foot = value.feet[foot_index];
        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            const G1FootprintProbe& probe = foot.probes[probe_index];
            logical_hash_value(hash, probe.current_sphere_center);
            logical_hash_value(hash, probe.current_sole_point);
            logical_hash_surface(hash, probe.current_surface);
            for (int sample = 0;
                 sample < G1CommandTrajectorySampleCount;
                 ++sample) {
                logical_hash_value(hash, probe.predicted_sphere_centers[sample]);
                logical_hash_value(hash, probe.predicted_sole_points[sample]);
                logical_hash_value(
                    hash,
                    static_cast<int>(probe.predicted_surface_status[sample]));
                logical_hash_surface(hash, probe.predicted_surfaces[sample]);
            }
            logical_hash_surface(hash, probe.selected_landing_surface);
            logical_hash_value(hash, probe.corridor_minimum_height);
            logical_hash_value(hash, probe.corridor_maximum_height);
            logical_hash_value(hash, probe.encountered_walkability_class);
        }
        logical_hash_value(hash, foot.current_contact);
        logical_hash_value(hash, foot.landing_expected);
        logical_hash_value(hash, foot.landing_patch_ready);
        logical_hash_value(hash, foot.landing_sample);
        logical_hash_value(hash, foot.predicted_landing_sole_center);
        logical_hash_value(
            hash, static_cast<int>(foot.predicted_landing_surface_status));
        logical_hash_surface(hash, foot.predicted_landing_surface);
        logical_hash_value(hash, foot.predicted_landing_walkability_class);
        logical_hash_value(hash, foot.landing_patch_maximum_residual_m);
        logical_hash_value(hash, foot.corridor_minimum_height);
        logical_hash_value(hash, foot.corridor_maximum_height);
        logical_hash_value(hash, foot.maximum_root_split_m);
        logical_hash_value(hash, foot.encountered_walkability_class);
        logical_hash_value(hash, foot.multilevel);
    }
    logical_hash_value(hash, value.blocked);
    logical_hash_value(hash, static_cast<int>(value.blocked_reason));
    logical_hash_value(hash, value.work.sweeps);
    logical_hash_value(hash, value.work.surface_queries);
    logical_hash_value(hash, value.work.node_visits);
}

static void logical_hash_lock(uint64_t& hash, const G1FootLockState& value)
{
    logical_hash_value(hash, value.initialized);
    logical_hash_value(hash, value.contact);
    logical_hash_value(hash, value.locked);
    logical_hash_value(hash, value.position_active);
    logical_hash_value(hash, value.releasing);
    logical_hash_value(hash, value.release_frames);
    logical_hash_value(hash, value.previous_input);
    logical_hash_value(hash, value.lock_point);
    logical_hash_value(hash, value.output_position);
    logical_hash_value(hash, value.output_velocity);
    logical_hash_value(hash, value.offset_position);
    logical_hash_value(hash, value.offset_velocity);
}

static void logical_hash_ik_state(uint64_t& hash, const G1IkState& value)
{
    logical_hash_value(hash, value.initialized);
    for (int foot = 0; foot < 2; ++foot) {
        logical_hash_lock(hash, value.feet[foot].lock);
        logical_hash_value(hash, value.feet[foot].swing.initialized);
        for (int probe = 0; probe < 4; ++probe) {
            logical_hash_value(
                hash, value.feet[foot].swing.previous_sphere_centers[probe]);
        }
        logical_hash_value(
            hash, value.feet[foot].baseline_sole_normal);
    }
}

static void logical_hash_target(uint64_t& hash, const G1FootTarget& value)
{
    logical_hash_value(hash, value.locked);
    logical_hash_value(hash, value.position_active);
    logical_hash_value(hash, value.releasing);
    logical_hash_value(hash, value.drift_limit_exceeded);
    logical_hash_value(hash, value.surface.point);
    logical_hash_value(hash, value.surface.normal);
    logical_hash_value(hash, value.desired_sole_normal);
    logical_hash_value(hash, value.sole_center);
    logical_hash_value(hash, value.horizontal_drift_m);
}

static void logical_hash_swing_candidate(
    uint64_t& hash, const G1SwingCandidateDiagnostic& value)
{
    logical_hash_value(hash, value.candidate_index);
    logical_hash_value(hash, value.lift_bits);
    logical_hash_value(hash, value.materialized_command_y_bits);
    for (int probe = 0; probe < 4; ++probe) {
        for (int axis = 0; axis < 3; ++axis) {
            logical_hash_value(
                hash, value.actual_sphere_center_bits[probe][axis]);
        }
    }
    logical_hash_value(hash, static_cast<int>(value.clearance_status));
    logical_hash_value(hash, value.controller_constraints_passed);
    logical_hash_value(hash, value.clearance_certified);
    logical_hash_value(hash, value.lower_margin_m);
    logical_hash_value(hash, value.witness_upper_margin_m);
    logical_hash_clearance_work(hash, value.clearance_work);
}

static void logical_hash_frame_result(
    uint64_t& hash, const G1IkFrameResult& value)
{
    logical_hash_value(hash, value.applied);
    logical_hash_value(hash, value.safe_stop_requested);
    logical_hash_value(hash, static_cast<int>(value.stop_reason));
    logical_hash_value(hash, value.max_correction_radians);
    logical_hash_value(hash, value.root_reach.active);
    logical_hash_value(hash, value.root_reach.common_interval_found);
    logical_hash_value(hash, value.root_reach.applied);
    logical_hash_value(hash, value.root_reach.root_y_delta_m);
    for (int foot = 0; foot < 2; ++foot) {
        const G1FootFrameResult& result = value.feet[foot];
        logical_hash_value(hash, result.recorded_contact);
        logical_hash_target(hash, result.target);
        logical_hash_value(hash, result.swing_selection.candidates_evaluated);
        logical_hash_value(hash, result.swing_selection.selected_index);
        logical_hash_swing_candidate(hash, result.swing_selection.selected);
        logical_hash_clearance_work(
            hash, result.swing_selection.total_clearance_work);
        logical_hash_value(hash, result.defensive_swing.lower_margin_m);
        logical_hash_value(hash, result.defensive_swing.witness_upper_m);
        logical_hash_value(hash, result.defensive_swing.sweep_evaluated);
        logical_hash_clearance_work(hash, result.defensive_swing.work);
        const G1LegSolveResult& position = result.position;
        logical_hash_value(hash, position.applied);
        logical_hash_value(hash, position.reachable);
        logical_hash_value(hash, position.correction_limited);
        logical_hash_value(hash, position.safe_stop_requested);
        logical_hash_value(hash, position.iterations);
        logical_hash_value(
            hash, static_cast<int>(position.iteration_provenance));
        logical_hash_value(hash, position.requested_ankle_target);
        logical_hash_value(hash, position.clamped_ankle_target);
        logical_hash_value(hash, position.hinge_axis_world);
        logical_hash_value(hash, position.bend_direction);
        logical_hash_value(hash, position.bend_used_current_projection);
        logical_hash_value(hash, position.bend_used_hinge_fallback);
        logical_hash_value(hash, position.bend_used_safe_perpendicular);
        logical_hash_value(hash, position.bend_sign_flipped);
        logical_hash_value(hash, position.raw_distance_m);
        logical_hash_value(hash, position.clamped_distance_m);
        logical_hash_value(hash, position.max_correction_radians);
        logical_hash_value(hash, position.contact_residual_m);
        const G1FootOrientationResult& orientation = result.orientation;
        logical_hash_value(hash, orientation.applied);
        logical_hash_value(hash, orientation.correction_limited);
        logical_hash_value(hash, orientation.safe_stop_requested);
        logical_hash_value(hash, orientation.target_global_rotation);
        logical_hash_value(hash, orientation.requested_correction_radians);
        logical_hash_value(hash, orientation.correction_radians);
    }
}

static uint64_t state_logical_digest(const g1_controller_state& state)
{
    uint64_t hash = UINT64_C(1469598103934665603);
#define HASH_VALUE(name) logical_hash_value(hash, state.name)
#define HASH_ARRAY(name) logical_hash_array(hash, state.name)
    HASH_VALUE(frame_index);
    HASH_VALUE(scene_frame);
    HASH_VALUE(search_time);
    HASH_VALUE(search_timer);
    HASH_VALUE(force_search_timer);
    HASH_ARRAY(curr_bone_positions);
    HASH_ARRAY(curr_bone_velocities);
    HASH_ARRAY(trns_bone_positions);
    HASH_ARRAY(trns_bone_velocities);
    HASH_ARRAY(curr_bone_rotations);
    HASH_ARRAY(trns_bone_rotations);
    HASH_ARRAY(curr_bone_angular_velocities);
    HASH_ARRAY(trns_bone_angular_velocities);
    HASH_ARRAY(curr_bone_contacts);
    HASH_ARRAY(trns_bone_contacts);
    HASH_ARRAY(bone_positions);
    HASH_ARRAY(bone_velocities);
    HASH_ARRAY(bone_angular_velocities);
    HASH_ARRAY(bone_rotations);
    HASH_ARRAY(bone_offset_positions);
    HASH_ARRAY(bone_offset_velocities);
    HASH_ARRAY(bone_offset_angular_velocities);
    HASH_ARRAY(bone_offset_rotations);
    HASH_ARRAY(adjusted_bone_positions);
    HASH_ARRAY(global_bone_positions);
    HASH_ARRAY(global_bone_velocities);
    HASH_ARRAY(adjusted_bone_rotations);
    HASH_ARRAY(global_bone_rotations);
    HASH_ARRAY(global_bone_angular_velocities);
    HASH_ARRAY(global_bone_computed);
    HASH_ARRAY(ik_bone_positions);
    HASH_ARRAY(ik_bone_rotations);
    HASH_ARRAY(ik_global_bone_positions);
    HASH_ARRAY(ik_global_bone_rotations);
    HASH_ARRAY(ik_candidate_bone_positions);
    HASH_ARRAY(ik_candidate_bone_rotations);
    HASH_ARRAY(ik_candidate_global_bone_positions);
    HASH_ARRAY(ik_candidate_global_bone_rotations);
    HASH_ARRAY(trajectory_desired_velocities);
    HASH_ARRAY(trajectory_positions);
    HASH_ARRAY(trajectory_velocities);
    HASH_ARRAY(trajectory_accelerations);
    HASH_ARRAY(trajectory_angular_velocities);
    HASH_ARRAY(trajectory_desired_rotations);
    HASH_ARRAY(trajectory_rotations);
    HASH_ARRAY(contact_bones);
    HASH_ARRAY(contact_states);
    HASH_ARRAY(contact_locks);
    HASH_ARRAY(contact_positions);
    HASH_ARRAY(contact_velocities);
    HASH_ARRAY(contact_points);
    HASH_ARRAY(contact_targets);
    HASH_ARRAY(contact_offset_positions);
    HASH_ARRAY(contact_offset_velocities);
    HASH_VALUE(transition_src_position);
    HASH_VALUE(transition_dst_position);
    HASH_VALUE(transition_src_rotation);
    HASH_VALUE(transition_dst_rotation);
    HASH_VALUE(desired_velocity);
    HASH_VALUE(desired_velocity_change_curr);
    HASH_VALUE(desired_velocity_change_prev);
    HASH_VALUE(desired_rotation);
    HASH_VALUE(desired_rotation_change_curr);
    HASH_VALUE(desired_rotation_change_prev);
    HASH_VALUE(desired_gait);
    HASH_VALUE(desired_gait_velocity);
    HASH_VALUE(simulation_position);
    HASH_VALUE(simulation_velocity);
    HASH_VALUE(simulation_acceleration);
    HASH_VALUE(simulation_rotation);
    HASH_VALUE(simulation_angular_velocity);
    logical_hash_command(hash, state.command);
    HASH_VALUE(footprint_status);
    logical_hash_footprint(hash, state.footprint);
    logical_hash_ik_state(hash, state.ik);
    logical_hash_frame_result(hash, state.ik_frame);
    logical_hash_pose_clearance(hash, state.ik_clearance);
    logical_hash_pose_clearance(hash, state.ik_candidate_clearance);
    HASH_VALUE(ik_candidate_clearance_status);
    HASH_VALUE(ik_candidate_rejected);
    logical_hash_support(hash, state.support);
    logical_hash_support_observation(hash, state.support_observation_now);
    HASH_VALUE(traversal_speed_scale);
    HASH_VALUE(traversal_speed_scale_velocity);
    HASH_VALUE(blocked);
    HASH_VALUE(walkability_class);
    HASH_VALUE(blocked_distance);
    HASH_VALUE(blocked_point);
    HASH_VALUE(route_index);
    HASH_VALUE(route_waypoint);
    HASH_VALUE(route_frames);
    HASH_VALUE(camera_azimuth);
    HASH_VALUE(camera_altitude);
    HASH_VALUE(camera_distance);
    HASH_VALUE(searched);
    HASH_VALUE(transitioned);
    HASH_VALUE(incumbent_cost);
    HASH_VALUE(selected_cost);
    HASH_VALUE(selected_terrain_error);
    HASH_VALUE(adjustment_xz);
    HASH_VALUE(adjustment_y);
    HASH_VALUE(clamp_xz);
    HASH_VALUE(clamp_y);
#undef HASH_ARRAY
#undef HASH_VALUE
    return hash;
}

static uint64_t hash_bytes(
    uint64_t hash, const void* data, std::size_t bytes)
{
    const unsigned char* values =
        static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index < bytes; ++index) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t state_digest(const g1_controller_state& state)
{
    return state_logical_digest(state);
}

static g1_controller_state& runtime_state(
    G1FrameRuntime& runtime, int index)
{
    switch (index) {
    case 0: return runtime.accepted_state;
    case 1: return runtime.working_state;
    case 2: return runtime.candidates.common_state;
    case 3: return runtime.candidates.raw_state;
    default: return runtime.candidates.ik_state;
    }
}

static const g1_controller_state& runtime_state(
    const G1FrameRuntime& runtime, int index)
{
    switch (index) {
    case 0: return runtime.accepted_state;
    case 1: return runtime.working_state;
    case 2: return runtime.candidates.common_state;
    case 3: return runtime.candidates.raw_state;
    default: return runtime.candidates.ik_state;
    }
}

struct runtime_storage_snapshot
{
    uint64_t digest[5] = {};
    const void* data[5][49] = {};
    std::size_t bytes[5][49] = {};
};

static runtime_storage_snapshot runtime_storage(
    const G1FrameRuntime& runtime)
{
    runtime_storage_snapshot output;
    for (int state_index = 0; state_index < 5; ++state_index) {
        const g1_controller_state& state =
            runtime_state(runtime, state_index);
        output.digest[state_index] = state_digest(state);
        g1_controller_state_memory_range ranges[64] = {};
        int count = 0;
        check(g1_controller_state_storage_ranges(
                  state, ranges, count, 64) && count == 49,
              "runtime storage snapshot covers all 245 dynamic owners");
        for (int range = 0; range < count; ++range) {
            output.data[state_index][range] = ranges[range].data;
            output.bytes[state_index][range] = ranges[range].bytes;
        }
    }
    return output;
}

static bool same_runtime_storage(
    const runtime_storage_snapshot& first,
    const runtime_storage_snapshot& second)
{
    return std::memcmp(&first, &second, sizeof(first)) == 0;
}

static bool snapshot_storage_is_disjoint_from_runtime(
    const runtime_storage_snapshot& snapshot,
    const G1FrameRuntime& runtime)
{
    const runtime_storage_snapshot active = runtime_storage(runtime);
    for (int old_state = 0; old_state < 5; ++old_state) {
        for (int old_range = 0; old_range < 49; ++old_range) {
            for (int new_state = 0; new_state < 5; ++new_state) {
                for (int new_range = 0; new_range < 49; ++new_range) {
                    if (g1_ik_memory_ranges_overlap(
                            snapshot.data[old_state][old_range],
                            snapshot.bytes[old_state][old_range],
                            active.data[new_state][new_range],
                            active.bytes[new_state][new_range])) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static uint64_t publication_digest(const G1FramePublication& publication)
{
    return hash_bytes(
        UINT64_C(1469598103934665603),
        &publication,
        sizeof(publication));
}

static uint64_t accepted_diagnostic_digest(
    const G1FrameAcceptedDiagnostic& diagnostic)
{
    return hash_bytes(
        UINT64_C(1469598103934665603),
        &diagnostic,
        sizeof(diagnostic));
}

struct transaction_fixture
{
    database db;
    terrain_support_set support;
    scene_catalog catalog;
    motion_pack_manifest manifest;
    scene_pack active_scene = fixture_scene("one", 2.0f);
    G1FrameRuntime runtime;
    G1FrameResetConfig config;
    int active_index = 0;
    fake_model active_model{1};
    fake_models models;
    int scene_loads = 0;
    int last_scene_target = -1;
    bool scene_failure = false;
    bool reset_failure = false;
    bool verify_precommit = false;
    bool verify_commit_before_old_unload = false;

    int snapshot_index = -1;
    std::string snapshot_scene_id;
    std::string snapshot_mesh_path;
    const float* snapshot_heights = NULL;
    int snapshot_model_id = 0;
    runtime_storage_snapshot snapshot_runtime;
    uint64_t snapshot_publication_digest = 0;
    uint64_t snapshot_accepted_diagnostic_digest = 0;

    transaction_fixture()
    {
        fixture_db(db);
        support.values.resize(db.nframes(), 3);
        support.values.set(-1.0f);
        catalog.ids = {"one", "two"};
        config.route_mode = true;
        config.route_id = "pair-route";
        config.ik_enabled = true;
        config.initial_search_time = 0.375f;
        char error[512] = {};
        check(g1_frame_runtime_reset(
                  runtime,
                  db,
                  support,
                  active_scene,
                  config,
                  error,
                  static_cast<int>(sizeof(error))),
              error);
        poison_runtime_states();
        runtime.publication.ik_safe_stop_latched = true;
        runtime.publication.presentation_frame = 71;
        runtime.accepted_diagnostic.ready = true;
        runtime.accepted_diagnostic.presentation_frame = 53;
        runtime.accepted_diagnostic.scene_frame =
            runtime.accepted_state.scene_frame - 1;
        models.live = 1;
        capture_active();
    }

    void poison_runtime_states()
    {
        for (int index = 0; index < 5; ++index) {
            g1_controller_state& state = runtime_state(runtime, index);
            state.scene_frame = 17 + index * 3;
            state.route_frames = 5 + index * 2;
            state.simulation_position.x = 2.0f +
                0.125f * static_cast<float>(index);
            state.simulation_position.z = 2.0f -
                0.0625f * static_cast<float>(index);
            G1FootLockState& lock = state.ik.feet[0].lock;
            lock.contact = false;
            lock.locked = false;
            lock.position_active = true;
            lock.releasing = true;
            lock.release_frames = index + 1;
            check(g1_controller_state_is_valid(state),
                  "each poisoned live state remains independently valid");
        }
    }

    void capture_active()
    {
        snapshot_index = active_index;
        snapshot_scene_id = active_scene.metadata.id;
        snapshot_mesh_path = active_scene.mesh_path;
        snapshot_heights = active_scene.terrain.heights.data;
        snapshot_model_id = active_model.id;
        snapshot_runtime = runtime_storage(runtime);
        snapshot_publication_digest = publication_digest(runtime.publication);
        snapshot_accepted_diagnostic_digest =
            accepted_diagnostic_digest(runtime.accepted_diagnostic);
    }

    bool active_matches_snapshot() const
    {
        return active_index == snapshot_index &&
               active_scene.metadata.id == snapshot_scene_id &&
               active_scene.mesh_path == snapshot_mesh_path &&
               active_scene.terrain.heights.data == snapshot_heights &&
               active_model.id == snapshot_model_id &&
               same_runtime_storage(
                   runtime_storage(runtime), snapshot_runtime) &&
               publication_digest(runtime.publication) ==
                   snapshot_publication_digest &&
               accepted_diagnostic_digest(runtime.accepted_diagnostic) ==
                   snapshot_accepted_diagnostic_digest;
    }

    bool switch_to(int target, char* error, int capacity)
    {
        auto load_scene = [&](scene_pack& out, int index, char* message, int cap)
        {
            ++scene_loads;
            last_scene_target = index;
            if (index < 0 || index >= static_cast<int>(catalog.ids.size()))
            {
                return scene_error(
                    message, cap, "fake scene target %d is out of range", index);
            }
            if (scene_failure)
            {
                return scene_error(
                    message,
                    cap,
                    "fake scene load failed for '%s'",
                    catalog.ids[static_cast<size_t>(index)].c_str());
            }
            out = fixture_scene(
                catalog.ids[static_cast<size_t>(index)].c_str(),
                2.0f);
            if (reset_failure)
            {
                out.metadata.spawn_position.x =
                    std::numeric_limits<float>::quiet_NaN();
            }
            return true;
        };
        auto load_model = [&](fake_model& out, const char* path,
                              char* message, int cap)
        {
            if (verify_precommit)
            {
                check(active_matches_snapshot(),
                      "active objects stay unchanged until candidate is ready");
            }
            ++models.loads;
            if (models.mode == fake_model_unallocated_failure)
            {
                scene_error(message, cap, "%s: fake model was not allocated", path);
                return scene_model_load_result{false, false};
            }
            out.id = 100 + models.loads;
            models.last_loaded_id = out.id;
            ++models.live;
            if (models.mode == fake_model_allocated_failure)
            {
                scene_error(message, cap, "%s: fake model is not ready", path);
                return scene_model_load_result{true, false};
            }
            return scene_model_load_result{true, true};
        };
        auto unload = [&](fake_model& model)
        {
            if (model.id == 0) return;
            if (verify_commit_before_old_unload &&
                model.id == snapshot_model_id)
            {
                bool all_candidate_states_installed =
                    scene_frame_runtime_reset_candidate_is_valid(
                        runtime, config);
                for (int index = 0; index < 5; ++index) {
                    all_candidate_states_installed =
                        all_candidate_states_installed &&
                        g1_frame_reset_candidate_is_valid(
                            runtime_state(runtime, index),
                            config.initial_search_time) &&
                        (index == 0 ||
                         g1_frame_controller_states_equal(
                             runtime.accepted_state,
                             runtime_state(runtime, index)));
                }
                check(active_index == target &&
                          active_scene.metadata.id ==
                              catalog.ids[static_cast<size_t>(target)] &&
                          all_candidate_states_installed &&
                          snapshot_storage_is_disjoint_from_runtime(
                              snapshot_runtime, runtime) &&
                          active_model.id == models.last_loaded_id,
                      "scene, all five states, model, and index commit before old unload");
            }
            ++models.unloads;
            --models.live;
            models.last_unloaded_id = model.id;
            models.unloaded_ids.push_back(model.id);
            model.id = 0;
        };
        return scene_switch_transaction(
            active_scene,
            runtime,
            active_model,
            active_index,
            target,
            db,
            support,
            config,
            load_scene,
            load_model,
            unload,
            error,
            capacity);
    }

    void unload_active()
    {
        if (active_model.id == 0) return;
        ++models.unloads;
        --models.live;
        models.last_unloaded_id = active_model.id;
        models.unloaded_ids.push_back(active_model.id);
        active_model.id = 0;
    }
};

static void check_reset_state_member(
    const g1_controller_state& state,
    const database& db,
    const scene_pack& scene,
    const G1FrameResetConfig& config,
    const char* message)
{
    check(g1_frame_reset_candidate_is_valid(
              state, config.initial_search_time),
          message);
    check(g1_ik_vec3_is_zero(state.command.intent.requested_velocity) &&
              g1_ik_vec3_is_zero(state.command.applied_velocity) &&
              g1_frame_quat_bits_equal(
                  state.command.intent.desired_heading,
                  quat_from_angle_axis(
                      scene.metadata.spawn_yaw,
                      vec3(0.0f, 1.0f, 0.0f))),
          "reset state independently owns zero travel and spawn heading");
    check(state.footprint_status == G1FootprintOk &&
              !state.footprint.blocked && state.ik.initialized &&
              g1_ik_runtime_state_is_valid(state.ik) &&
              state.ik_candidate_clearance_status == G1ClearanceOk &&
              !state.ik_candidate_rejected &&
              state.ik_clearance.minimum.lower_bound_m >= -0.01,
          "reset state independently owns certified footprint, IK, and clearance");

    array1d<vec3> support_positions(G1_BoneCount);
    support_pose_apply(
        support_positions, state.bone_positions, state.support.height);
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        check(g1_frame_vec3_bits_equal(
                  support_positions(bone),
                  state.adjusted_bone_positions(bone)) &&
                  g1_frame_quat_bits_equal(
                      state.bone_rotations(bone),
                      state.adjusted_bone_rotations(bone)),
              "reset state independently owns support-retargeted baseline");
    }

    array1d<vec3> checked_positions(G1_BoneCount);
    array1d<quat> checked_rotations(G1_BoneCount);
    char error[512] = {};
    check(g1_ik_checked_forward_kinematics(
              checked_positions,
              checked_rotations,
              state.adjusted_bone_positions,
              state.adjusted_bone_rotations,
              db.bone_parents,
              error,
              static_cast<int>(sizeof(error))),
          error);
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        check(g1_frame_vec3_bits_equal(
                  checked_positions(bone),
                  state.global_bone_positions(bone)) &&
                  g1_frame_quat_bits_equal(
                      checked_rotations(bone),
                      state.global_bone_rotations(bone)) &&
                  g1_frame_vec3_bits_equal(
                      checked_positions(bone),
                      state.ik_global_bone_positions(bone)) &&
                  g1_frame_quat_bits_equal(
                      checked_rotations(bone),
                      state.ik_global_bone_rotations(bone)),
              "reset state independently owns checked final FK");
    }
}

static void check_reset_runtime_unit(
    const transaction_fixture& fixture)
{
    g1_controller_state_memory_range ranges[5][64] = {};
    int counts[5] = {};
    for (int index = 0; index < 5; ++index) {
        const g1_controller_state& state =
            runtime_state(fixture.runtime, index);
        check(g1_controller_state_storage_ranges(
                  state, ranges[index], counts[index], 64) &&
                  counts[index] == 49,
              "each reset state owns exactly 49 heap-backed arrays");
        if (index != 0) {
            check(g1_frame_controller_states_equal(
                      fixture.runtime.accepted_state, state),
                  "all five reset states are logically equal");
        }
        check(state.route_index == 0 && state.route_waypoint == 1 &&
                  state.route_frames == 0,
              "all five reset states own the canonical route cursor");
        check_reset_state_member(
            state,
            fixture.db,
            fixture.active_scene,
            fixture.config,
            "each reset state passes every independent certification gate");
    }
    for (int first = 0; first < 5; ++first) {
        for (int second = first + 1; second < 5; ++second) {
            check(&runtime_state(fixture.runtime, first) !=
                      &runtime_state(fixture.runtime, second) &&
                      g1_frame_state_range_sets_are_disjoint(
                          runtime_state(fixture.runtime, first),
                          ranges[first],
                          counts[first],
                          runtime_state(fixture.runtime, second),
                          ranges[second],
                          counts[second]),
                  "all ten reset-state object and heap pairs are disjoint");
        }
    }
    check(g1_frame_publication_is_valid(fixture.runtime.publication) &&
              !fixture.runtime.publication.rejection.rejected &&
              !fixture.runtime.publication.ik_safe_stop_latched &&
              fixture.runtime.publication.presentation_frame == 0 &&
              g1_frame_accepted_diagnostic_is_valid(
                  fixture.runtime.accepted_diagnostic) &&
              !fixture.runtime.accepted_diagnostic.ready,
          "successful reset/switch clears rejection, latch, and diagnostics");
}

template<class T>
using state_array_member = array1d<T> g1_controller_state::*;

template<class T>
static void check_live_preflight_rejects_member_alias(
    transaction_fixture& fixture,
    state_array_member<T> member)
{
    char error[512] = {};
    for (int first = 0; first < 5; ++first) {
        for (int second = first + 1; second < 5; ++second) {
            array1d<T>& target = runtime_state(fixture.runtime, second).*member;
            const array1d<T>& peer =
                runtime_state(fixture.runtime, first).*member;
            T* const saved = target.data;
            target.data = peer.data;
            const bool accepted = scene_frame_runtime_live_storage_preflight(
                fixture.runtime,
                fixture.db,
                fixture.support,
                fixture.active_scene,
                fixture.config,
                &fixture.active_model,
                sizeof(fixture.active_model),
                &fixture.active_index,
                sizeof(fixture.active_index),
                error,
                static_cast<int>(sizeof(error)));
            target.data = saved;
            check(!accepted && fixture.active_matches_snapshot(),
                  "live preflight rejects each owner alias across all ten state pairs without mutation");
        }
    }
}

template<class T>
static void check_candidate_isolation_rejects_member_alias(
    transaction_fixture& fixture,
    G1FrameRuntime& candidate,
    state_array_member<T> member)
{
    char error[512] = {};
    for (int live_index = 0; live_index < 5; ++live_index) {
        for (int candidate_index = 0; candidate_index < 5; ++candidate_index) {
            array1d<T>& target =
                runtime_state(candidate, candidate_index).*member;
            const array1d<T>& peer =
                runtime_state(fixture.runtime, live_index).*member;
            T* const saved = target.data;
            target.data = peer.data;
            const bool isolated = scene_frame_runtime_candidate_is_isolated(
                fixture.runtime,
                candidate,
                fixture.db,
                fixture.support,
                fixture.active_scene,
                fixture.active_scene,
                fixture.config,
                &fixture.active_model,
                sizeof(fixture.active_model),
                &fixture.active_index,
                sizeof(fixture.active_index),
                error,
                static_cast<int>(sizeof(error)));
            target.data = saved;
            check(!isolated && fixture.active_matches_snapshot(),
                  "candidate isolation rejects each owner alias across all 25 live/candidate pairs without publication");
        }
    }
}

static void test_five_state_reset_is_valid_equal_and_pairwise_disjoint()
{
    transaction_fixture fixture;
    char error[512] = {};
    check(scene_reset_current(
              fixture.runtime,
              fixture.db,
              fixture.support,
              fixture.active_scene,
              fixture.config,
              error,
              static_cast<int>(sizeof(error))),
          error);
    check_reset_runtime_unit(fixture);
    fixture.unload_active();
}

static void test_five_state_live_preflight_rejects_each_array_alias()
{
    transaction_fixture fixture;
#define CHECK_LIVE_ALIAS(name) \
    check_live_preflight_rejects_member_alias( \
        fixture, &g1_controller_state::name)
    G1_FOR_EACH_STATE_ARRAY(CHECK_LIVE_ALIAS);
#undef CHECK_LIVE_ALIAS
    fixture.unload_active();
}

static void test_five_state_candidate_isolation_covers_all_ten_states()
{
    transaction_fixture fixture;
    G1FrameRuntime candidate;
    char error[512] = {};
    check(g1_frame_runtime_reset(
              candidate,
              fixture.db,
              fixture.support,
              fixture.active_scene,
              fixture.config,
              error,
              static_cast<int>(sizeof(error))),
          error);
    check(scene_frame_runtime_candidate_is_isolated(
              fixture.runtime,
              candidate,
              fixture.db,
              fixture.support,
              fixture.active_scene,
              fixture.active_scene,
              fixture.config,
              &fixture.active_model,
              sizeof(fixture.active_model),
              &fixture.active_index,
              sizeof(fixture.active_index),
              error,
              static_cast<int>(sizeof(error))),
          "ten-state baseline is isolated");

    const g1_controller_state* states[10] = {};
    g1_controller_state_memory_range ranges[10][64] = {};
    int counts[10] = {};
    for (int index = 0; index < 5; ++index) {
        states[index] = &runtime_state(fixture.runtime, index);
        states[index + 5] = &runtime_state(candidate, index);
    }
    for (int index = 0; index < 10; ++index) {
        check(g1_controller_state_storage_ranges(
                  *states[index], ranges[index], counts[index], 64) &&
                  counts[index] == 49,
              "all ten states expose all 49 owners");
    }
    for (int first = 0; first < 10; ++first) {
        for (int second = first + 1; second < 10; ++second) {
            check(states[first] != states[second] &&
                      g1_frame_state_range_sets_are_disjoint(
                          *states[first], ranges[first], counts[first],
                          *states[second], ranges[second], counts[second]),
                  "all 45 ten-state heap-set pairs are disjoint");
        }
    }

    g1_controller_state& candidate_common =
        runtime_state(candidate, 2);
    vec3* const saved_positions =
        candidate_common.curr_bone_positions.data;
    candidate_common.curr_bone_positions.data =
        candidate_common.curr_bone_velocities.data + 1;
    const bool within_state_isolated =
        scene_frame_runtime_candidate_is_isolated(
            fixture.runtime,
            candidate,
            fixture.db,
            fixture.support,
            fixture.active_scene,
            fixture.active_scene,
            fixture.config,
            &fixture.active_model,
            sizeof(fixture.active_model),
            &fixture.active_index,
            sizeof(fixture.active_index),
            error,
            static_cast<int>(sizeof(error)));
    candidate_common.curr_bone_positions.data = saved_positions;
    check(!within_state_isolated && fixture.active_matches_snapshot(),
          "candidate isolation rejects within-state partial owner aliases without publication");

#define CHECK_CANDIDATE_ALIAS(name) \
    check_candidate_isolation_rejects_member_alias( \
        fixture, candidate, &g1_controller_state::name)
    G1_FOR_EACH_STATE_ARRAY(CHECK_CANDIDATE_ALIAS);
#undef CHECK_CANDIDATE_ALIAS
    fixture.unload_active();
}

static void test_scene_load_failure_and_invalid_target_skip_model_allocation()
{
    transaction_fixture fixture;
    char error[512] = {};
    const int prior_loads = fixture.models.loads;
    const int prior_unloads = fixture.models.unloads;

    fixture.scene_failure = true;
    fixture.capture_active();
    check(!fixture.switch_to(1, error, static_cast<int>(sizeof(error))),
          "scene-load failure is reported");
    check(std::string(error).find("fake scene load failed") != std::string::npos,
          "scene-load failure reason is preserved");
    check(fixture.models.loads == prior_loads &&
              fixture.models.unloads == prior_unloads &&
              fixture.models.live == 1,
          "scene-load failure allocates and unloads no candidate model");
    check(fixture.active_matches_snapshot(),
          "scene-load failure preserves every active object");

    fixture.scene_failure = false;
    fixture.capture_active();
    error[0] = '\0';
    check(!fixture.switch_to(7, error, static_cast<int>(sizeof(error))),
          "invalid target is rejected");
    check(fixture.last_scene_target == 7 &&
              std::string(error).find("target 7") != std::string::npos,
          "invalid target is handled by the scene loader");
    check(fixture.models.loads == prior_loads &&
              fixture.models.unloads == prior_unloads &&
              fixture.models.live == 1 && fixture.active_matches_snapshot(),
          "invalid target preserves active objects without model allocation");
    fixture.unload_active();
    check(fixture.models.live == 0, "scene failure fixture cleans up active model");
}

static void test_reset_failure_preserves_active_objects_and_skips_model_load()
{
    transaction_fixture fixture;
    char error[512] = {};
    fixture.reset_failure = true;
    fixture.capture_active();
    const int prior_loads = fixture.models.loads;
    const int prior_unloads = fixture.models.unloads;

    check(!fixture.switch_to(1, error, static_cast<int>(sizeof(error))),
          "candidate reset failure is reported");
    check(error[0] != '\0',
          "candidate reset failure preserves an actionable reason");
    check(fixture.scene_loads == 1 && fixture.models.loads == prior_loads &&
              fixture.models.unloads == prior_unloads &&
              fixture.models.live == 1,
          "candidate reset failure occurs before model load");
    check(fixture.active_matches_snapshot(),
          "candidate reset failure preserves every active object");
    fixture.unload_active();
    check(fixture.models.live == 0, "reset failure fixture cleans up active model");
}

static void test_model_failures_cleanup_only_allocated_candidate()
{
    transaction_fixture fixture;
    char error[512] = {};

    fixture.models.mode = fake_model_unallocated_failure;
    fixture.capture_active();
    const int prior_unloads = fixture.models.unloads;
    check(!fixture.switch_to(1, error, static_cast<int>(sizeof(error))),
          "unallocated model failure is reported");
    check(fixture.models.unloads == prior_unloads && fixture.models.live == 1,
          "unallocated model failure does not call unload");
    check(fixture.active_matches_snapshot(),
          "unallocated model failure preserves every active object");

    fixture.models.mode = fake_model_allocated_failure;
    fixture.capture_active();
    const int candidate_unloads = fixture.models.unloads;
    error[0] = '\0';
    check(!fixture.switch_to(1, error, static_cast<int>(sizeof(error))),
          "allocated-but-not-ready model failure is reported");
    check(error[0] != '\0',
          "allocated model failure preserves an actionable reason");
    check(fixture.models.unloads == candidate_unloads + 1 &&
              fixture.models.last_unloaded_id == fixture.models.last_loaded_id &&
              fixture.models.live == 1,
          "allocated-but-not-ready candidate is unloaded exactly once");
    check(fixture.active_matches_snapshot(),
          "allocated model failure preserves every active object");
    fixture.unload_active();
    check(fixture.models.live == 0, "model failure fixture cleans up active model");
}

static void test_five_state_success_swaps_every_state_atomically()
{
    transaction_fixture fixture;
    char error[512] = {};
    fixture.capture_active();
    const int old_model_id = fixture.active_model.id;
    fixture.verify_precommit = true;
    fixture.verify_commit_before_old_unload = true;

    check(fixture.switch_to(1, error, static_cast<int>(sizeof(error))), error);
    bool all_five_installed = true;
    for (int index = 0; index < 5; ++index) {
        const g1_controller_state& state =
            runtime_state(fixture.runtime, index);
        all_five_installed = all_five_installed &&
            state.scene_frame == 0 &&
            state.simulation_position.x == 2.0f;
    }
    check(fixture.active_index == 1 &&
              fixture.active_scene.metadata.id == "two" &&
              fixture.active_scene.mesh_path == "two.obj" &&
              all_five_installed &&
              fixture.active_model.id == fixture.models.last_loaded_id,
          "successful transaction commits all five candidate states and peer objects");
    check_reset_runtime_unit(fixture);
    check(std::count(
              fixture.models.unloaded_ids.begin(),
              fixture.models.unloaded_ids.end(),
              old_model_id) == 1 &&
              fixture.models.unloads == 1 && fixture.models.live == 1,
          "successful transaction unloads the old model exactly once");
    fixture.unload_active();
    check(fixture.models.live == 0, "success fixture cleans up active model");
}

static void test_reset_changes_only_state_and_is_transactional()
{
    transaction_fixture fixture;
    char error[512] = {};
    const int scene_loads = fixture.scene_loads;
    const int model_loads = fixture.models.loads;
    const int model_unloads = fixture.models.unloads;
    const int model_id = fixture.active_model.id;
    const std::string scene_id = fixture.active_scene.metadata.id;
    const std::string mesh_path = fixture.active_scene.mesh_path;
    const float* const heights = fixture.active_scene.terrain.heights.data;

    fixture.runtime.accepted_state.scene_frame = 91;
    fixture.runtime.accepted_state.route_frames = 19;
    fixture.runtime.working_state.scene_frame = 92;
    fixture.runtime.working_state.route_frames = 20;
    check(scene_reset_current(
              fixture.runtime,
              fixture.db,
              fixture.support,
              fixture.active_scene,
              fixture.config,
              error,
              static_cast<int>(sizeof(error))),
          error);
    check_reset_runtime_unit(fixture);
    check(fixture.scene_loads == scene_loads &&
              fixture.models.loads == model_loads &&
              fixture.models.unloads == model_unloads &&
              fixture.active_model.id == model_id &&
              fixture.active_scene.metadata.id == scene_id &&
              fixture.active_scene.mesh_path == mesh_path &&
              fixture.active_scene.terrain.heights.data == heights,
          "reset does not reload or replace scene or model");

    fixture.poison_runtime_states();
    fixture.runtime.publication.ik_safe_stop_latched = true;
    fixture.runtime.publication.presentation_frame = 83;
    fixture.capture_active();
    terrain_support_set bad_support;
    bad_support.values = fixture.support.values;
    bad_support.values(0, 0) = std::numeric_limits<float>::quiet_NaN();
    error[0] = '\0';
    check(!scene_reset_current(
              fixture.runtime,
              fixture.db,
              bad_support,
              fixture.active_scene,
              fixture.config,
              error,
              static_cast<int>(sizeof(error))),
          "failed current-scene reset is reported");
    check(error[0] != '\0',
          "failed current-scene reset preserves an actionable reason");
    check(fixture.active_matches_snapshot() &&
              fixture.scene_loads == scene_loads &&
              fixture.models.loads == model_loads &&
              fixture.models.unloads == model_unloads,
          "failed reset preserves state, scene, and model");
    fixture.unload_active();
    check(fixture.models.live == 0, "reset fixture cleans up active model");
}

static void test_repeated_switches_leave_exactly_one_live_model()
{
    transaction_fixture fixture;
    char error[512] = {};
    for (int cycle = 0; cycle < 20; ++cycle)
    {
        const int target = 1 - fixture.active_index;
        check(fixture.switch_to(target, error, static_cast<int>(sizeof(error))),
              error);
        check(fixture.models.live == 1,
              "each successful cycle leaves exactly one live model");
    }
    check(fixture.models.loads == 20 && fixture.models.unloads == 20 &&
              fixture.models.live == 1,
          "repeated cycles unload every superseded model exactly once");
    fixture.unload_active();
    check(fixture.models.live == 0, "normal final unload releases last model");
}

static void test_each_candidate_member_gate_is_independent_and_nonpublishing()
{
    transaction_fixture fixture;
    fixture.capture_active();
    G1FrameRuntime candidate;
    char error[512] = {};
    for (int index = 0; index < 5; ++index) {
        check(g1_frame_runtime_reset(
                  candidate,
                  fixture.db,
                  fixture.support,
                  fixture.active_scene,
                  fixture.config,
                  error,
                  static_cast<int>(sizeof(error))),
              error);
        runtime_state(candidate, index).footprint_status =
            G1FootprintInvalidInput;
        check(!scene_frame_runtime_reset_candidate_is_valid(
                  candidate, fixture.config),
              "each poisoned five-state candidate gate is rejected");
        check(fixture.active_matches_snapshot(),
              "candidate validation cannot publish into the live unit");
    }

    G1FrameResetConfig invalid_config = fixture.config;
    invalid_config.dt = std::nextafter(
        fixture.config.dt, std::numeric_limits<float>::infinity());
    error[0] = '\0';
    check(!scene_reset_current(
              fixture.runtime,
              fixture.db,
              fixture.support,
              fixture.active_scene,
              invalid_config,
              error,
              static_cast<int>(sizeof(error))),
          "malformed pair-reset configuration is rejected");
    check(fixture.active_matches_snapshot(),
          "malformed pair-reset configuration preserves the whole live unit");
    fixture.unload_active();
}

static void test_five_state_switch_failure_preserves_every_live_owner()
{
    test_scene_load_failure_and_invalid_target_skip_model_allocation();
    test_reset_failure_preserves_active_objects_and_skips_model_load();
    test_model_failures_cleanup_only_allocated_candidate();
    test_each_candidate_member_gate_is_independent_and_nonpublishing();
}

static void test_five_state_reset_failure_preserves_every_live_owner()
{
    transaction_fixture fixture;
    char error[512] = {};
    fixture.poison_runtime_states();
    fixture.capture_active();

    terrain_support_set bad_support;
    bad_support.values = fixture.support.values;
    bad_support.values(0, 0) =
        std::numeric_limits<float>::quiet_NaN();
    check(!scene_reset_current(
              fixture.runtime,
              fixture.db,
              bad_support,
              fixture.active_scene,
              fixture.config,
              error,
              static_cast<int>(sizeof(error))) &&
              fixture.active_matches_snapshot(),
          "artifact failure preserves all five live state values and 245 owners");

    G1FrameResetConfig malformed = fixture.config;
    malformed.dt = std::nextafter(
        malformed.dt, std::numeric_limits<float>::infinity());
    error[0] = '\0';
    check(!scene_reset_current(
              fixture.runtime,
              fixture.db,
              fixture.support,
              fixture.active_scene,
              malformed,
              error,
              static_cast<int>(sizeof(error))) &&
              fixture.active_matches_snapshot(),
          "malformed reset config preserves all five live state values and 245 owners");
    fixture.unload_active();
}

int main()
{
    test_five_state_reset_is_valid_equal_and_pairwise_disjoint();
    test_five_state_live_preflight_rejects_each_array_alias();
    test_five_state_candidate_isolation_covers_all_ten_states();
    test_five_state_success_swaps_every_state_atomically();
    test_five_state_switch_failure_preserves_every_live_owner();
    test_five_state_reset_failure_preserves_every_live_owner();
    test_reset_changes_only_state_and_is_transactional();
    test_repeated_switches_leave_exactly_one_live_model();
    return 0;
}
