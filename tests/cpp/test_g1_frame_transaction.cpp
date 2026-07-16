#define G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM
#include "g1_frame_transaction.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

static void check(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "G1 frame transaction test failed: %s\n", message);
        std::exit(1);
    }
}

static_assert(G1FrameStageInputRouteCommand == 0,
              "input/route command is the first transaction stage");
static_assert(G1FrameStagePoseCertificate + 1 == G1FrameStageCount,
              "pose certification is the final transaction stage");
static_assert(std::is_same<G1FrameStageRunner,
    G1FrameStageOutcome (*)(
        G1FrameTransactionStage,
        g1_controller_state&,
        G1FrameTransactionScratch&,
        const G1FrameExternalInputs&,
        char*, int)>::value,
    "the runner has only working state, scratch, and immutable external input");

static void make_database(database& db, int frames = 32)
{
    static const int parents[G1_BoneCount] = {
        -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
    };
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

static scene_pack make_scene()
{
    scene_pack scene;
    scene.metadata.id = "transaction-flat";
    scene.metadata.spawn_position = vec3(2.0f, 0.0f, 2.0f);
    scene.metadata.spawn_yaw = 0.37f;
    scene.metadata.playable_bounds = {-3.0f, -3.0f, 7.0f, 7.0f};
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
    return scene;
}

static void test_runtime_reset_publishes_mode_specific_route_cursor()
{
    database db;
    make_database(db);
    terrain_support_set support;
    support.values.resize(db.nframes(), 3);
    support.values.set(-1.0f);
    scene_pack scene = make_scene();
    G1FrameRuntime runtime;
    G1FrameResetConfig config;
    char error[512] = {};

    check(g1_frame_runtime_reset(
              runtime, db, support, scene, config,
              error, static_cast<int>(sizeof(error))),
          error);
    check(runtime.accepted_state.route_index == -1 &&
              runtime.accepted_state.route_waypoint == 0 &&
              runtime.accepted_state.route_frames == 0 &&
              runtime.working_state.route_index == -1 &&
              runtime.working_state.route_waypoint == 0 &&
              runtime.working_state.route_frames == 0,
          "typed non-route reset publishes the canonical inactive cursor to both states");

    scene_route route;
    route.id = "typed-route";
    route.expected_outcome = "pass";
    route.walkability_class = 1;
    route.waypoints_xz.push_back(std::make_pair(2.0f, 2.0f));
    route.waypoints_xz.push_back(std::make_pair(3.0f, 2.0f));
    scene.metadata.routes.push_back(route);
    config.route_mode = true;
    config.route_id = scene.metadata.routes[0].id.c_str();
    check(g1_frame_runtime_reset(
              runtime, db, support, scene, config,
              error, static_cast<int>(sizeof(error))),
          error);
    check(runtime.accepted_state.route_index == 0 &&
              runtime.accepted_state.route_waypoint == 1 &&
              runtime.accepted_state.route_frames == 0 &&
              runtime.working_state.route_index == 0 &&
              runtime.working_state.route_waypoint == 1 &&
              runtime.working_state.route_frames == 0,
          "typed route reset publishes the resolved initial cursor to both states");

    config.route_mode = false;
    config.route_id = nullptr;
    check(g1_frame_runtime_reset(
              runtime, db, support, scene, config,
              error, static_cast<int>(sizeof(error))),
          error);
    check(runtime.accepted_state.route_index == -1 &&
              runtime.accepted_state.route_waypoint == 0 &&
              runtime.accepted_state.route_frames == 0 &&
              runtime.working_state.route_index == -1 &&
              runtime.working_state.route_waypoint == 0 &&
              runtime.working_state.route_frames == 0,
          "typed reset deactivates a previously routed cursor atomically");
}

static void seed_nonzero_array_tails(g1_controller_state& state)
{
    state.bone_offset_velocities(G1_BoneCount - 1) =
        vec3(0.001f, -0.002f, 0.003f);
    state.global_bone_velocities(G1_BoneCount - 1) =
        vec3(-0.004f, 0.005f, -0.006f);
    state.trajectory_accelerations(G1CommandTrajectorySampleCount - 1) =
        vec3(0.007f, 0.008f, -0.009f);
    state.contact_offset_velocities(1) =
        vec3(-0.010f, 0.011f, 0.012f);
}

struct fixture
{
    database db;
    terrain_support_set support;
    scene_pack scene = make_scene();
    G1FrameRuntime runtime;
    G1FrameExternalInputs external;

    fixture()
    {
        make_database(db);
        support.values.resize(db.nframes(), 3);
        support.values.set(-1.0f);
        G1FrameResetConfig config;
        config.initial_search_time = 0.375f;
        char error[512] = {};
        check(g1_frame_runtime_reset(
                  runtime, db, support, scene, config,
                  error, static_cast<int>(sizeof(error))),
              error);
        seed_nonzero_array_tails(runtime.accepted_state);
        seed_nonzero_array_tails(runtime.working_state);
        runtime.working_state.camera_azimuth += 0.0625f;
        check(g1_controller_state_is_valid(runtime.accepted_state) &&
                  g1_controller_state_is_valid(runtime.working_state) &&
                  terrain_float_bits(runtime.accepted_state.camera_azimuth) !=
                      terrain_float_bits(
                          runtime.working_state.camera_azimuth),
              "fixture starts with valid but observably distinct accepted and working states");
        external.db = &db;
        external.support = &support;
        external.scene = &scene;
        external.input.move_stick = vec3(0.25f, -0.0f, -0.5f);
        external.input.look_stick = vec3();
        external.input.gait_target = 0.625f;
        external.input.camera_zoom_axis = -0.125f;
        external.input.scripted_azimuth_delta = 0.03125f;
        external.input.desired_strafe = true;
        external.input.presentation_frame = 41;
        external.tuning.initial_search_time = config.initial_search_time;
        external.tuning.ik_enabled = config.ik_enabled;
        external.tuning.effective_terrain_weight = 0.75f;
    }
};

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

template<class T>
static uint64_t value_logical_digest(const T& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_value(hash, value);
    return hash;
}

static uint64_t command_logical_digest(const G1CommandSnapshot& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_command(hash, value);
    return hash;
}

static uint64_t footprint_logical_digest(
    const G1FootprintObservation& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_footprint(hash, value);
    return hash;
}

static uint64_t ik_state_logical_digest(const G1IkState& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_ik_state(hash, value);
    return hash;
}

static uint64_t ik_frame_logical_digest(const G1IkFrameResult& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_frame_result(hash, value);
    return hash;
}

static uint64_t pose_clearance_logical_digest(
    const G1PoseClearance& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_pose_clearance(hash, value);
    return hash;
}

static uint64_t support_logical_digest(const support_frame_state& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_support(hash, value);
    return hash;
}

static uint64_t support_observation_logical_digest(
    const support_observation& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_support_observation(hash, value);
    return hash;
}

static float flip_low_unit_normal_bit(float value)
{
    uint32_t bits = terrain_float_bits(value) ^ UINT32_C(1);
    float output = 0.0f;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

static G1FootTarget active_test_target()
{
    G1FootTarget target;
    target.surface.point = vec3(1.0f, 0.0f, 2.0f);
    target.surface.normal = vec3(0.0f, 1.0f, 0.0f);
    target.desired_sole_normal = vec3(0.0f, 1.0f, 0.0f);
    target.sole_center = vec3(1.0f, 0.125f, 2.0f);
    check(g1_foot_target_is_valid(target),
          "active target fixture is structurally valid");
    return target;
}

static void test_orientation_authenticates_desired_sole_normal()
{
    const G1LegConfig config = g1_left_leg_config();
    G1FootOrientationResult orientation;
    orientation.applied = true;
    orientation.target_global_rotation = quat();

    G1FootTarget terrain_differs = active_test_target();
    terrain_differs.surface.normal = vec3(0.6f, 0.8f, 0.0f);
    check(g1_foot_target_is_valid(terrain_differs),
          "terrain-evidence normal may differ from the desired current-sole normal");
    check(g1_frame_orientation_is_complete(
              orientation, terrain_differs, config),
          "orientation completeness follows desired sole normal, not terrain evidence");

    G1FootTarget forged_desired = active_test_target();
    forged_desired.desired_sole_normal = vec3(0.6f, 0.8f, 0.0f);
    check(g1_foot_target_is_valid(forged_desired),
          "forged desired normal remains structurally valid");
    check(!g1_frame_orientation_is_complete(
              orientation, forged_desired, config),
          "orientation completeness rejects a structurally valid wrong desired normal");
}

static void test_contact_iteration_provenance_is_authenticated()
{
    database db;
    make_database(db, 1);
    db.bone_positions(0, G1_Simulation) =
        vec3(2.0f, 1.0f, 2.0f);
    db.bone_rotations(0, G1_LeftToe) =
        quat_from_angle_axis(
            -0.11f, vec3(0.0f, 0.0f, 1.0f));
    const G1LegConfig config = g1_left_leg_config();
    array1d<vec3> global_positions(G1_BoneCount);
    array1d<quat> global_rotations(G1_BoneCount);
    char error[512] = {};
    check(g1_ik_checked_forward_kinematics(
              global_positions,
              global_rotations,
              db.bone_positions(0),
              db.bone_rotations(0),
              db.bone_parents,
              error, static_cast<int>(sizeof(error))),
          error);
    vec3 current_sole;
    check(g1_ik_checked_physical_sole_centroid(
              current_sole,
              global_positions(config.contact),
              global_rotations(config.contact),
              config),
          "fallback fixture materializes its tilted sole");
    db.bone_positions(0, G1_Simulation).y +=
        config.planted_clearance_m - current_sole.y;
    check(g1_ik_checked_forward_kinematics(
              global_positions,
              global_rotations,
              db.bone_positions(0),
              db.bone_rotations(0),
              db.bone_parents,
              error, static_cast<int>(sizeof(error))) &&
              g1_ik_checked_physical_sole_centroid(
                  current_sole,
                  global_positions(config.contact),
                  global_rotations(config.contact),
                  config),
          error);

    const scene_pack scene = make_scene();
    G1FootTarget target;
    target.locked = true;
    target.position_active = true;
    check(g1_surface_target_sample(
              target.surface,
              scene.terrain,
              current_sole.x,
              current_sole.z,
              config.planted_clearance_m,
              error, static_cast<int>(sizeof(error))),
          error);
    target.desired_sole_normal = target.surface.normal;
    target.sole_center = target.surface.point;
    check(g1_foot_target_is_valid(target),
          "fallback fixture owns a valid planted target");

    db.bone_positions(0, G1_Simulation).x += 0.0008f;
    db.bone_positions(0, G1_Simulation).y += 0.00483f;
    check(g1_ik_checked_forward_kinematics(
              global_positions,
              global_rotations,
              db.bone_positions(0),
              db.bone_rotations(0),
              db.bone_parents,
              error, static_cast<int>(sizeof(error))) &&
              g1_ik_checked_physical_sole_centroid(
                  current_sole,
                  global_positions(config.contact),
                  global_rotations(config.contact),
                  config),
          error);
    double baseline_residual_precise =
        std::numeric_limits<double>::max();
    float baseline_residual =
        std::numeric_limits<float>::max();
    check(ik_checked_distance_precise(
              baseline_residual_precise,
              baseline_residual,
              current_sole,
              target.sole_center) &&
              baseline_residual_precise > 0.0 &&
              g1_ik_contact_residual_is_converged_precise(
                  baseline_residual_precise) &&
              g1_ik_contact_residual_is_converged(
                  baseline_residual),
          "fallback fixture starts inside the accepted baseline residual");

    G1IkRuntimeStagedCandidate staged;
    check(g1_ik_runtime_stage_recorded_contact(
              staged,
              db.bone_positions(0),
              db.bone_rotations(0),
              db.bone_parents,
              config,
              target,
              error, static_cast<int>(sizeof(error))),
          error);
    check(!staged.passes &&
              staged.position.applied &&
              staged.position.reachable &&
              !staged.position.correction_limited &&
              staged.position.safe_stop_requested &&
              staged.position.iterations == 1 &&
              staged.position.iteration_provenance ==
                  G1LegIterationBaselineFallback1 &&
              terrain_float_bits(
                  staged.position.max_correction_radians) == 0U &&
              staged.position.contact_residual_m > 0.005f &&
              staged.orientation.applied &&
              !staged.orientation.safe_stop_requested,
          "real recorded-contact fallback exposes its final orientation-adjusted residual stop");
    check(g1_frame_leg_result_is_complete(
              staged.position, config),
          "frame publication accepts the complete one-iteration fallback diagnostic");

    G1IkFrameResult fallback_rejection;
    fallback_rejection.safe_stop_requested = true;
    fallback_rejection.stop_reason = G1IkStopTargetUnreachable;
    fallback_rejection.feet[0].recorded_contact = true;
    fallback_rejection.feet[0].target = target;
    fallback_rejection.feet[0].position = staged.position;
    fallback_rejection.feet[0].orientation = staged.orientation;
    check(g1_frame_rejected_ik_result_is_valid(fallback_rejection),
          "frame publication accepts the authentic baseline fallback rejection");
    for (int forged_iterations = 2;
         forged_iterations <= G1ContactSolveMaximumIterations;
         ++forged_iterations) {
        G1IkFrameResult forged = fallback_rejection;
        forged.feet[0].position.iterations = forged_iterations;
        check(!g1_frame_rejected_ik_result_is_valid(forged),
              "frame publication rejects every in-range fallback iteration forgery");
    }
    G1IkFrameResult boolean_spoof = fallback_rejection;
    boolean_spoof.feet[0].position.iterations =
        G1ContactSolveMaximumIterations;
    boolean_spoof.feet[0].position.reachable = false;
    check(!g1_frame_rejected_ik_result_is_valid(boolean_spoof),
          "frame publication cannot authenticate fallback provenance from a flipped boolean");

    G1FootTarget ordinary_target = target;
    ordinary_target.surface.point.y += 3.0f;
    ordinary_target.sole_center.y += 3.0f;
    check(g1_foot_target_is_valid(ordinary_target),
          "ordinary rejection target remains structurally valid");
    G1IkRuntimeStagedCandidate ordinary;
    check(g1_ik_runtime_stage_recorded_contact(
              ordinary,
              db.bone_positions(0),
              db.bone_rotations(0),
              db.bone_parents,
              config,
              ordinary_target,
              error, static_cast<int>(sizeof(error))),
          error);
    check(!ordinary.passes &&
              ordinary.position.applied &&
              ordinary.position.iterations ==
                  G1ContactSolveMaximumIterations &&
              ordinary.position.iteration_provenance ==
                  G1LegIterationContact4 &&
              !g1_ik_contact_residual_is_converged(
                  ordinary.position.contact_residual_m),
          "ordinary target-unreachable solve authentically exhausts refinement");
    G1IkFrameResult ordinary_rejection;
    ordinary_rejection.safe_stop_requested = true;
    ordinary_rejection.stop_reason = G1IkStopTargetUnreachable;
    ordinary_rejection.feet[0].recorded_contact = true;
    ordinary_rejection.feet[0].target = ordinary_target;
    ordinary_rejection.feet[0].position = ordinary.position;
    ordinary_rejection.feet[0].orientation = ordinary.orientation;
    check(g1_frame_rejected_ik_result_is_valid(ordinary_rejection),
          "frame publication accepts the authentic exhausted rejection");
    for (int forged_iterations = 1;
         forged_iterations < G1ContactSolveMaximumIterations;
         ++forged_iterations) {
        G1IkFrameResult forged = ordinary_rejection;
        forged.feet[0].position.iterations = forged_iterations;
        check(!g1_frame_rejected_ik_result_is_valid(forged),
              "frame publication rejects every in-range ordinary iteration forgery");
    }
}

static void test_ready_landing_is_validation_only_lookahead()
{
    const G1FootTarget target = active_test_target();
    G1FootprintFootObservation lookahead;
    lookahead.landing_expected = true;
    lookahead.landing_patch_ready = true;
    lookahead.landing_sample = 2U;
    lookahead.predicted_landing_sole_center =
        vec3(5.0f, 0.75f, -3.0f);
    lookahead.predicted_landing_surface_status = G1SurfaceQueryValid;
    lookahead.predicted_landing_surface.height = 0.625f;
    lookahead.predicted_landing_surface.normal =
        vec3(0.6f, 0.8f, 0.0f);
    lookahead.predicted_landing_walkability_class = 1;
    check(g1_frame_landing_lookahead_is_valid(
              target, lookahead, g1_left_leg_config()),
          "ready future landing validates lookahead without snapping the current target");

    lookahead.landing_patch_ready = false;
    check(!g1_frame_landing_lookahead_is_valid(
              target, lookahead, g1_left_leg_config()),
          "unready future landing is rejected independently of current target geometry");
}

static void test_target_and_ik_state_normals_are_transaction_owned()
{
    const G1FootTarget target = active_test_target();
    G1FootTarget target_forgery = target;
    target_forgery.desired_sole_normal.y =
        flip_low_unit_normal_bit(target_forgery.desired_sole_normal.y);
    check(g1_foot_target_is_valid(target_forgery) &&
              terrain_float_bits(target.desired_sole_normal.y) !=
                  terrain_float_bits(
                      target_forgery.desired_sole_normal.y),
          "one-bit desired-normal target forgery is structurally valid");
    uint64_t target_hash = UINT64_C(1469598103934665603);
    uint64_t target_forgery_hash = UINT64_C(1469598103934665603);
    logical_hash_target(target_hash, target);
    logical_hash_target(target_forgery_hash, target_forgery);
    check(target_hash != target_forgery_hash,
          "target logical digest owns desired sole normal bits");
    check(!g1_frame_target_equal(target, target_forgery),
          "target equality rejects a one-bit desired sole normal forgery");

    fixture value;
    const G1IkState ik_state = value.runtime.accepted_state.ik;
    G1IkState ik_state_forgery = ik_state;
    ik_state_forgery.feet[0].baseline_sole_normal.y =
        flip_low_unit_normal_bit(
            ik_state_forgery.feet[0].baseline_sole_normal.y);
    check(g1_ik_runtime_state_is_valid(ik_state) &&
              g1_ik_runtime_state_is_valid(ik_state_forgery),
          "one-bit baseline-normal IK-state forgery is structurally valid");
    uint64_t ik_hash = UINT64_C(1469598103934665603);
    uint64_t ik_forgery_hash = UINT64_C(1469598103934665603);
    logical_hash_ik_state(ik_hash, ik_state);
    logical_hash_ik_state(ik_forgery_hash, ik_state_forgery);
    check(ik_hash != ik_forgery_hash,
          "IK-state logical digest owns baseline sole normal bits");
    check(!g1_frame_ik_state_equal(ik_state, ik_state_forgery),
          "IK-state equality rejects a one-bit baseline sole normal forgery");

    value.runtime.accepted_state.ik = ik_state_forgery;
    check(g1_controller_state_is_valid(value.runtime.accepted_state),
          "forged baseline normal remains valid in a canonical frame state");
    char error[512] = {};
    check(g1_controller_state_copy(
              value.runtime.working_state,
              value.runtime.accepted_state,
              error, static_cast<int>(sizeof(error))),
          error);
    check(g1_frame_ik_state_equal(
              value.runtime.accepted_state.ik,
              value.runtime.working_state.ik) &&
              terrain_float_bits(
                  value.runtime.working_state.ik.feet[0]
                      .baseline_sole_normal.y) ==
                  terrain_float_bits(
                      value.runtime.accepted_state.ik.feet[0]
                          .baseline_sole_normal.y),
          "checked frame-state copy preserves baseline sole normal bits");
}

struct StateStorageIdentity
{
    const void* data = NULL;
    int size = 0;
    std::size_t bytes = 0U;
};

struct StateStorageIdentities
{
    StateStorageIdentity owners[49];
};

static StateStorageIdentities state_storage_identities(
    const g1_controller_state& state)
{
    StateStorageIdentities output = {};
    int count = 0;
    const auto add = [&output, &count](
                         const void* data, int size,
                         std::size_t element_size) {
        check(count < 49, "state storage identity capacity is exact");
        output.owners[count].data = data;
        output.owners[count].size = size;
        output.owners[count].bytes = size > 0
            ? static_cast<std::size_t>(size) * element_size
            : 0U;
        ++count;
    };
#define ADD_ARRAY(name) \
    add(state.name.data, state.name.size, sizeof(*state.name.data))
    ADD_ARRAY(curr_bone_positions);
    ADD_ARRAY(curr_bone_velocities);
    ADD_ARRAY(trns_bone_positions);
    ADD_ARRAY(trns_bone_velocities);
    ADD_ARRAY(curr_bone_rotations);
    ADD_ARRAY(trns_bone_rotations);
    ADD_ARRAY(curr_bone_angular_velocities);
    ADD_ARRAY(trns_bone_angular_velocities);
    ADD_ARRAY(curr_bone_contacts);
    ADD_ARRAY(trns_bone_contacts);
    ADD_ARRAY(bone_positions);
    ADD_ARRAY(bone_velocities);
    ADD_ARRAY(bone_angular_velocities);
    ADD_ARRAY(bone_rotations);
    ADD_ARRAY(bone_offset_positions);
    ADD_ARRAY(bone_offset_velocities);
    ADD_ARRAY(bone_offset_angular_velocities);
    ADD_ARRAY(bone_offset_rotations);
    ADD_ARRAY(adjusted_bone_positions);
    ADD_ARRAY(global_bone_positions);
    ADD_ARRAY(global_bone_velocities);
    ADD_ARRAY(adjusted_bone_rotations);
    ADD_ARRAY(global_bone_rotations);
    ADD_ARRAY(global_bone_angular_velocities);
    ADD_ARRAY(global_bone_computed);
    ADD_ARRAY(trajectory_desired_velocities);
    ADD_ARRAY(trajectory_positions);
    ADD_ARRAY(trajectory_velocities);
    ADD_ARRAY(trajectory_accelerations);
    ADD_ARRAY(trajectory_angular_velocities);
    ADD_ARRAY(trajectory_desired_rotations);
    ADD_ARRAY(trajectory_rotations);
    ADD_ARRAY(contact_bones);
    ADD_ARRAY(contact_states);
    ADD_ARRAY(contact_locks);
    ADD_ARRAY(contact_positions);
    ADD_ARRAY(contact_velocities);
    ADD_ARRAY(contact_points);
    ADD_ARRAY(contact_targets);
    ADD_ARRAY(contact_offset_positions);
    ADD_ARRAY(contact_offset_velocities);
    ADD_ARRAY(ik_bone_positions);
    ADD_ARRAY(ik_bone_rotations);
    ADD_ARRAY(ik_global_bone_positions);
    ADD_ARRAY(ik_global_bone_rotations);
    ADD_ARRAY(ik_candidate_bone_positions);
    ADD_ARRAY(ik_candidate_bone_rotations);
    ADD_ARRAY(ik_candidate_global_bone_positions);
    ADD_ARRAY(ik_candidate_global_bone_rotations);
#undef ADD_ARRAY
    check(count == 49, "state storage snapshot covers exactly 49 owners");
    return output;
}

static bool same_storage_identities(
    const StateStorageIdentities& first,
    const StateStorageIdentities& second)
{
    for (int index = 0; index < 49; ++index) {
        if (first.owners[index].data != second.owners[index].data ||
            first.owners[index].size != second.owners[index].size ||
            first.owners[index].bytes != second.owners[index].bytes) {
            return false;
        }
    }
    return true;
}

static bool storage_identities_are_mutually_disjoint(
    const StateStorageIdentities& accepted,
    const StateStorageIdentities& working)
{
    for (int first = 0; first < 49; ++first) {
        if (accepted.owners[first].data == NULL ||
            accepted.owners[first].bytes == 0U ||
            working.owners[first].data == NULL ||
            working.owners[first].bytes == 0U) {
            return false;
        }
        for (int second = first + 1; second < 49; ++second) {
            if (g1_ik_memory_ranges_overlap(
                    accepted.owners[first].data,
                    accepted.owners[first].bytes,
                    accepted.owners[second].data,
                    accepted.owners[second].bytes) ||
                g1_ik_memory_ranges_overlap(
                    working.owners[first].data,
                    working.owners[first].bytes,
                    working.owners[second].data,
                    working.owners[second].bytes)) {
                return false;
            }
        }
        for (int second = 0; second < 49; ++second) {
            if (g1_ik_memory_ranges_overlap(
                    accepted.owners[first].data,
                    accepted.owners[first].bytes,
                    working.owners[second].data,
                    working.owners[second].bytes)) {
                return false;
            }
        }
    }
    return true;
}

static void logical_hash_route(
    uint64_t& hash, const deterministic_route_sample& value)
{
    logical_hash_value(hash, value.command);
    logical_hash_value(hash, value.waypoint);
    logical_hash_value(hash, value.complete);
}

static void logical_hash_traversal(
    uint64_t& hash, const traversability_diagnostics& value)
{
    logical_hash_value(hash, value.blocked);
    logical_hash_value(hash, value.walkability_class);
    logical_hash_value(hash, static_cast<int>(value.reason));
    logical_hash_value(hash, value.distance);
    logical_hash_value(hash, value.commanded_speed);
    logical_hash_value(hash, value.applied_speed);
    logical_hash_value(hash, value.point);
}

static void logical_hash_centerline(
    uint64_t& hash, const terrain_centerline_snapshot& value)
{
    for (int index = 0; index < 4; ++index) {
        logical_hash_value(hash, value.values[index]);
        logical_hash_value(hash, value.points[index]);
    }
}

static void logical_hash_pose_diagnostic(
    uint64_t& hash, const motion_match_pose_diagnostic& value)
{
    logical_hash_value(hash, value.hips_y);
    logical_hash_value(hash, value.hips_clearance);
    logical_hash_value(hash, value.left_toe_clearance);
    logical_hash_value(hash, value.right_toe_clearance);
    logical_hash_value(hash, value.minimum_clearance);
}

static void logical_hash_rejection(
    uint64_t& hash, const G1FrameRejectionDiagnostic& value)
{
    logical_hash_value(hash, value.rejected);
    logical_hash_value(hash, static_cast<int>(value.stage));
    logical_hash_value(hash, static_cast<int>(value.stop_reason));
    logical_hash_value(hash, value.attempted_footprint_available);
    logical_hash_value(hash, static_cast<int>(value.footprint_status));
    logical_hash_footprint(hash, value.attempted_footprint);
    logical_hash_value(hash, value.attempted_ik_available);
    logical_hash_frame_result(hash, value.ik_frame);
    logical_hash_value(hash, value.attempted_pose_available);
    logical_hash_value(hash, static_cast<int>(value.pose_status));
    logical_hash_pose_clearance(hash, value.pose_clearance);
}

static uint64_t publication_logical_digest(
    const G1FramePublication& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_intent(hash, value.requested_intent);
    logical_hash_rejection(hash, value.rejection);
    logical_hash_value(hash, value.ik_safe_stop_latched);
    logical_hash_value(hash, value.presentation_frame);
    return hash;
}

static uint64_t diagnostic_logical_digest(
    const G1FrameAcceptedDiagnostic& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_value(hash, value.ready);
    logical_hash_value(hash, value.presentation_frame);
    logical_hash_value(hash, value.scene_frame);
    logical_hash_route(hash, value.route);
    logical_hash_traversal(hash, value.traversal);
    for (int dimension = 0; dimension < 31; ++dimension) {
        logical_hash_value(hash, value.query[dimension]);
    }
    logical_hash_value(hash, value.query_database_frame);
    logical_hash_value(hash, value.query_range);
    logical_hash_value(hash, value.selected_database_frame);
    logical_hash_centerline(hash, value.terrain_query);
    logical_hash_pose_diagnostic(hash, value.raw_selected);
    logical_hash_pose_diagnostic(hash, value.inertialized);
    logical_hash_pose_diagnostic(hash, value.support_retargeted);
    logical_hash_pose_diagnostic(hash, value.rendered);
    logical_hash_value(hash, value.matching_enabled);
    logical_hash_value(hash, value.adjustment_enabled);
    logical_hash_value(hash, value.clamping_enabled);
    logical_hash_value(hash, value.ik_enabled);
    logical_hash_value(hash, value.effective_terrain_weight);
    return hash;
}

static bool same_intent_bits(
    const G1CommandIntent& first, const G1CommandIntent& second)
{
    uint64_t first_hash = UINT64_C(1469598103934665603);
    uint64_t second_hash = UINT64_C(1469598103934665603);
    logical_hash_intent(first_hash, first);
    logical_hash_intent(second_hash, second);
    return first_hash == second_hash;
}

enum HostileRunnerMode
{
    HostileRunnerNone = 0,
    HostileRunnerMissingIntent,
    HostileRunnerNonfiniteIntent,
    HostileRunnerMissingRejection,
    HostileRunnerMalformedRejection,
    HostileRunnerNoncanonicalRejection,
    HostileRunnerInvalidFinalState,
    HostileRunnerMissingAcceptedDiagnostic,
    HostileRunnerMalformedAcceptedDiagnostic,
};

struct TransactionTrace
{
    int runner_calls[G1FrameStageCount] = {};
    int hook_calls[G1FrameStageCount] = {};
    G1FrameTransactionStage runner_order[G1FrameStageCount] = {};
    G1FrameTransactionStage hook_order[G1FrameStageCount] = {};
    int runner_total = 0;
    int hook_total = 0;
    G1FrameTransactionStage runner_outcome_stage = G1FrameStageCount;
    G1FrameStageOutcome runner_outcome = G1FrameStageContinue;
    bool input_hook_ready = false;
    bool input_hook_velocity_exact = false;
    bool input_hook_heading_exact = false;
    bool prior_latch_seen = false;
    bool latched_handoff_velocity_exact = false;
    bool latched_handoff_cancel = false;
    bool latched_handoff_force_search = false;
    HostileRunnerMode hostile_mode = HostileRunnerNone;
    bool require_exact_initial_copy = false;
    uint64_t expected_initial_copy_digest = 0U;
    bool exact_initial_copy_seen = false;
    const g1_controller_state* success_candidate = NULL;
    bool mutate_working_ik_provenance_after_transcript = false;
    bool mutate_disabled_ik_result_after_transcript = false;
};

static TransactionTrace trace;

static void reset_trace()
{
    trace = TransactionTrace{};
}

static G1FrameRejectionDiagnostic finite_rejection()
{
    G1FrameRejectionDiagnostic rejection;
    rejection.rejected = true;
    rejection.stage = G1FrameRejectFootprint;
    rejection.stop_reason = G1IkStopFootprintOutsideDomain;
    rejection.footprint_status = G1FootprintOutsideDomain;
    return rejection;
}

static motion_match_pose_diagnostic pose_diagnostic(float base)
{
    motion_match_pose_diagnostic value;
    value.hips_y = base;
    value.hips_clearance = base + 0.01f;
    value.left_toe_clearance = base + 0.02f;
    value.right_toe_clearance = base + 0.03f;
    value.minimum_clearance = base + 0.04f;
    return value;
}

static G1FrameAcceptedDiagnostic accepted_diagnostic_candidate(
    const g1_controller_state& state,
    const G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external)
{
    G1FrameAcceptedDiagnostic accepted;
    accepted.ready = true;
    accepted.presentation_frame = external.input.presentation_frame;
    accepted.scene_frame = state.scene_frame;
    accepted.route.command = scratch.traversal_input;
    accepted.route.waypoint = state.route_waypoint;
    accepted.route.complete = false;
    accepted.traversal.blocked = false;
    accepted.traversal.walkability_class = state.walkability_class;
    accepted.traversal.reason = walkability_clear;
    accepted.traversal.distance = 0.125f;
    accepted.traversal.commanded_speed = 0.50f;
    accepted.traversal.applied_speed = 0.375f;
    accepted.traversal.point = vec3(0.25f, 0.50f, 0.75f);
    for (int dimension = 0; dimension < 31; ++dimension) {
        accepted.query[dimension] =
            0.001f * static_cast<float>(dimension + 1);
    }
    accepted.query_database_frame = state.frame_index;
    accepted.query_range = 0;
    accepted.selected_database_frame = state.frame_index;
    for (int sample = 0; sample < 4; ++sample) {
        const float base = 0.10f * static_cast<float>(sample + 1);
        accepted.terrain_query.values[sample] = base;
        accepted.terrain_query.points[sample] =
            vec3(base + 0.01f, base + 0.02f, base + 0.03f);
    }
    accepted.raw_selected = pose_diagnostic(1.0f);
    accepted.inertialized = pose_diagnostic(2.0f);
    accepted.support_retargeted = pose_diagnostic(3.0f);
    accepted.rendered = pose_diagnostic(4.0f);
    accepted.matching_enabled = true;
    accepted.adjustment_enabled = external.tuning.adjustment_enabled;
    accepted.clamping_enabled = external.tuning.clamping_enabled;
    accepted.ik_enabled = external.tuning.ik_enabled;
    accepted.effective_terrain_weight =
        external.tuning.effective_terrain_weight;
    return accepted;
}

static void apply_latch_to_working_state(
    g1_controller_state& state,
    const G1IkSafeStopHandoff& handoff)
{
    if (!handoff.cancel_planar_inertia) return;
    state.simulation_velocity.x = 0.0f;
    state.simulation_velocity.z = 0.0f;
    state.simulation_acceleration.x = 0.0f;
    state.simulation_acceleration.z = 0.0f;
}

static void apply_success_tail(
    g1_controller_state& state,
    bool prior_safe_stop_latched)
{
    ++state.scene_frame;
    if (!prior_safe_stop_latched) state.route_frames += 3;
    state.camera_azimuth += 0.125f;
    state.selected_cost += 0.25f;
}

static void apply_stage_marker(
    g1_controller_state& state,
    G1FrameTransactionStage stage)
{
    switch (stage) {
    case G1FrameStageInputRouteCommand:
        state.camera_azimuth += 0.001f;
        break;
    case G1FrameStageMatcherSearch:
        state.searched = true;
        break;
    case G1FrameStageInertialization:
        state.incumbent_cost += 0.003f;
        break;
    case G1FrameStageSimulationUpdate:
        state.simulation_position.x += 0.004f;
        break;
    case G1FrameStageSupportObservation:
        state.support_observation_now.delta[0] += 0.005f;
        break;
    case G1FrameStageSupportRetarget:
        state.adjustment_xz += 0.006f;
        break;
    case G1FrameStageContactUpdate:
        state.contact_points(0).x += 0.007f;
        break;
    case G1FrameStageFootprintObservation:
        state.blocked_distance = 0.875f;
        break;
    case G1FrameStageFirstFootIk:
        state.selected_terrain_error += 0.009f;
        break;
    case G1FrameStageSecondFootIk:
        state.clamp_xz += 0.010f;
        break;
    case G1FrameStageFinalFk:
        state.camera_altitude += 0.011f;
        break;
    case G1FrameStagePoseCertificate:
        state.camera_distance += 0.012f;
        break;
    default:
        break;
    }
}

static G1FrameStageOutcome test_runner(
    G1FrameTransactionStage stage,
    g1_controller_state& state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error, int error_capacity)
{
    const int stage_index = static_cast<int>(stage);
    if (stage_index < 0 || stage_index >= G1FrameStageCount ||
        trace.runner_total >= G1FrameStageCount) {
        return G1FrameStageGlobalError;
    }
    ++trace.runner_calls[stage_index];
    trace.runner_order[trace.runner_total++] = stage;
    scratch.query[stage_index] =
        10.0f + static_cast<float>(stage_index);

    if (stage == G1FrameStageInputRouteCommand &&
        trace.require_exact_initial_copy) {
        trace.exact_initial_copy_seen =
            state_logical_digest(state) ==
                trace.expected_initial_copy_digest;
        if (!trace.exact_initial_copy_seen) {
            return G1FrameStageGlobalError;
        }
    }
    if (stage == G1FrameStageInputRouteCommand &&
        trace.success_candidate != NULL &&
        !g1_controller_state_copy(
            state, *trace.success_candidate, error, error_capacity)) {
        return G1FrameStageGlobalError;
    }

    if (stage == G1FrameStageInputRouteCommand) {
        trace.prior_latch_seen = scratch.prior_safe_stop_latched;
        scratch.commanded_velocity = external.input.move_stick;
        if (!g1_ik_safe_stop_handoff(
                scratch.safe_stop_handoff,
                scratch.prior_safe_stop_latched,
                scratch.commanded_velocity,
                error, error_capacity)) {
            return G1FrameStageGlobalError;
        }
        apply_latch_to_working_state(state, scratch.safe_stop_handoff);
        scratch.traversal_input = scratch.safe_stop_handoff.applied_velocity;
        scratch.force_search = scratch.safe_stop_handoff.force_search;
        scratch.requested_intent.requested_velocity = external.input.move_stick;
        scratch.requested_intent.desired_heading = state.desired_rotation;
        scratch.requested_intent_ready = true;
    } else if (stage == G1FrameStagePoseCertificate) {
        scratch.accepted_diagnostic_candidate =
            accepted_diagnostic_candidate(state, scratch, external);
        scratch.accepted_diagnostic_ready = true;
    }
    if (stage == G1FrameStageFinalFk) {
        scratch.ik_transaction = G1IkFrameTransaction{};
        scratch.ik_transaction.initialized = true;
        scratch.ik_transaction.next_foot = 2U;
        scratch.ik_transaction.candidate_state = state.ik;
        scratch.ik_transaction.candidate_result = state.ik_frame;
        for (int foot = 0; foot < 2; ++foot) {
            scratch.ik_transaction.staged_iteration_provenance[foot] =
                state.ik_frame.feet[foot].position.applied
                    ? state.ik_frame.feet[foot]
                          .position.iteration_provenance
                    : G1LegIterationNone;
        }
    }
    apply_stage_marker(state, stage);
    if (stage == G1FrameStagePoseCertificate) {
        apply_success_tail(state, scratch.prior_safe_stop_latched);
        if (trace.mutate_working_ik_provenance_after_transcript) {
            state.ik_frame.feet[0].position.iteration_provenance =
                G1LegIterationBaselineFallback1;
        }
        if (trace.mutate_disabled_ik_result_after_transcript) {
            state.ik_frame.max_correction_radians = -0.0f;
        }
    }

    if (stage == trace.runner_outcome_stage) {
        if (trace.runner_outcome == G1FrameStageFiniteReject) {
            scratch.rejection = finite_rejection();
        }
        return trace.runner_outcome;
    }
    return G1FrameStageContinue;
}

static G1FrameStageOutcome hostile_runner(
    G1FrameTransactionStage stage,
    g1_controller_state& state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error, int error_capacity)
{
    const G1FrameStageOutcome ordinary = test_runner(
        stage, state, scratch, external, error, error_capacity);
    if (ordinary != G1FrameStageContinue) return ordinary;

    if (stage == G1FrameStageInputRouteCommand) {
        if (trace.hostile_mode == HostileRunnerMissingIntent) {
            scratch.requested_intent_ready = false;
        } else if (trace.hostile_mode == HostileRunnerNonfiniteIntent) {
            scratch.requested_intent.requested_velocity.x =
                std::numeric_limits<float>::quiet_NaN();
        }
    }
    if (stage == G1FrameStageFootprintObservation) {
        if (trace.hostile_mode == HostileRunnerMissingRejection) {
            scratch.rejection = G1FrameRejectionDiagnostic{};
            return G1FrameStageFiniteReject;
        }
        if (trace.hostile_mode == HostileRunnerMalformedRejection) {
            scratch.rejection = finite_rejection();
            scratch.rejection.stage = G1FrameRejectNone;
            return G1FrameStageFiniteReject;
        }
        if (trace.hostile_mode == HostileRunnerNoncanonicalRejection) {
            scratch.rejection = finite_rejection();
            scratch.rejection.attempted_footprint.root_surface.height = 1.0f;
            return G1FrameStageFiniteReject;
        }
    }
    if (stage == G1FrameStagePoseCertificate) {
        if (trace.hostile_mode == HostileRunnerInvalidFinalState) {
            state.search_timer =
                std::numeric_limits<float>::quiet_NaN();
        } else if (trace.hostile_mode ==
                   HostileRunnerMissingAcceptedDiagnostic) {
            scratch.accepted_diagnostic_ready = false;
        } else if (trace.hostile_mode ==
                   HostileRunnerMalformedAcceptedDiagnostic) {
            scratch.accepted_diagnostic_candidate.query[0] =
                std::numeric_limits<float>::quiet_NaN();
        }
    }
    return G1FrameStageContinue;
}

static G1FrameInjectedOutcome injection_hook(
    G1FrameTransactionStage stage,
    const g1_controller_state& working_state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    const G1FrameTransactionTestControl& control,
    char*, int)
{
    const int stage_index = static_cast<int>(stage);
    if (stage_index < 0 || stage_index >= G1FrameStageCount ||
        trace.hook_total >= G1FrameStageCount) {
        return G1FrameInjectGlobalError;
    }
    ++trace.hook_calls[stage_index];
    trace.hook_order[trace.hook_total++] = stage;
    if (stage == G1FrameStageInputRouteCommand) {
        G1CommandIntent expected;
        expected.requested_velocity = external.input.move_stick;
        expected.desired_heading = working_state.desired_rotation;
        trace.input_hook_ready = scratch.requested_intent_ready;
        trace.input_hook_velocity_exact =
            terrain_float_bits(scratch.commanded_velocity.x) ==
                terrain_float_bits(external.input.move_stick.x) &&
            terrain_float_bits(scratch.commanded_velocity.y) ==
                terrain_float_bits(external.input.move_stick.y) &&
            terrain_float_bits(scratch.commanded_velocity.z) ==
                terrain_float_bits(external.input.move_stick.z);
        trace.input_hook_heading_exact =
            same_intent_bits(scratch.requested_intent, expected);
        if (scratch.prior_safe_stop_latched) {
            trace.latched_handoff_velocity_exact =
                terrain_float_bits(
                    scratch.safe_stop_handoff.applied_velocity.x) == 0U &&
                terrain_float_bits(
                    scratch.safe_stop_handoff.applied_velocity.y) ==
                    terrain_float_bits(external.input.move_stick.y) &&
                terrain_float_bits(
                    scratch.safe_stop_handoff.applied_velocity.z) == 0U;
            trace.latched_handoff_cancel =
                scratch.safe_stop_handoff.cancel_planar_inertia;
            trace.latched_handoff_force_search =
                scratch.safe_stop_handoff.force_search;
        }
    }
    if (stage != control.injected_stage) {
        return G1FrameInjectContinue;
    }
    if (control.injected_outcome == G1FrameInjectFiniteReject) {
        scratch.rejection = finite_rejection();
    }
    return control.injected_outcome;
}

static void check_finite_publication(
    const G1FrameRuntime& runtime,
    const G1FramePublication& before,
    const G1CommandIntent& expected_intent,
    int expected_frame)
{
    G1FramePublication expected = before;
    expected.requested_intent = expected_intent;
    expected.rejection = finite_rejection();
    expected.ik_safe_stop_latched = true;
    expected.presentation_frame = expected_frame;
    check(publication_logical_digest(runtime.publication) ==
              publication_logical_digest(expected),
          "finite publication changes only the exact four-field whitelist");
}

struct RuntimeEvidence
{
    uint64_t accepted = 0U;
    uint64_t working = 0U;
    StateStorageIdentities accepted_storage;
    StateStorageIdentities working_storage;
    uint64_t publication = 0U;
    uint64_t diagnostic = 0U;
};

static RuntimeEvidence runtime_evidence(const G1FrameRuntime& runtime)
{
    RuntimeEvidence output;
    output.accepted = state_logical_digest(runtime.accepted_state);
    output.working = state_logical_digest(runtime.working_state);
    output.accepted_storage =
        state_storage_identities(runtime.accepted_state);
    output.working_storage =
        state_storage_identities(runtime.working_state);
    output.publication = publication_logical_digest(runtime.publication);
    output.diagnostic =
        diagnostic_logical_digest(runtime.accepted_diagnostic);
    return output;
}

static RuntimeEvidence preflight_runtime_evidence(
    const G1FrameRuntime& runtime)
{
    check(g1_controller_state_is_valid(runtime.working_state),
          "every preflight starts from a semantically valid working destination");
    check(state_logical_digest(runtime.working_state) !=
              state_logical_digest(runtime.accepted_state),
          "every preflight starts from a working value logically distinct from accepted");
    return runtime_evidence(runtime);
}

static bool same_runtime_evidence(
    const RuntimeEvidence& first,
    const RuntimeEvidence& second)
{
    return first.accepted == second.accepted &&
           first.working == second.working &&
           same_storage_identities(
               first.accepted_storage, second.accepted_storage) &&
           same_storage_identities(
               first.working_storage, second.working_storage) &&
           first.publication == second.publication &&
           first.diagnostic == second.diagnostic;
}

static bool same_published_runtime_evidence(
    const RuntimeEvidence& first,
    const RuntimeEvidence& second)
{
    return first.accepted == second.accepted &&
           same_storage_identities(
               first.accepted_storage, second.accepted_storage) &&
           same_storage_identities(
               first.working_storage, second.working_storage) &&
           first.publication == second.publication &&
           first.diagnostic == second.diagnostic;
}

static void check_finite_preservation(
    const G1FrameRuntime& runtime,
    const RuntimeEvidence& before,
    const char* message)
{
    check(state_logical_digest(runtime.accepted_state) == before.accepted &&
              same_storage_identities(
                  state_storage_identities(runtime.accepted_state),
                  before.accepted_storage) &&
              same_storage_identities(
                  state_storage_identities(runtime.working_state),
                  before.working_storage) &&
              diagnostic_logical_digest(runtime.accepted_diagnostic) ==
                  before.diagnostic,
          message);
}

static void check_global_preservation(
    const G1FrameRuntime& runtime,
    const RuntimeEvidence& before,
    const char* message)
{
    check(state_logical_digest(runtime.accepted_state) == before.accepted &&
              same_storage_identities(
                  state_storage_identities(runtime.accepted_state),
                  before.accepted_storage) &&
              same_storage_identities(
                  state_storage_identities(runtime.working_state),
                  before.working_storage) &&
              publication_logical_digest(runtime.publication) ==
                  before.publication &&
              diagnostic_logical_digest(runtime.accepted_diagnostic) ==
                  before.diagnostic,
          message);
}

static void check_preflight_preservation(
    const G1FrameRuntime& runtime,
    const RuntimeEvidence& before,
    const char* message)
{
    check_global_preservation(runtime, before, message);
    check(state_logical_digest(runtime.working_state) == before.working,
          message);
}

static void check_stage_trace(
    int last_runner_stage,
    int last_hook_stage,
    const char* message)
{
    check(trace.runner_total == last_runner_stage + 1 &&
              trace.hook_total == last_hook_stage + 1,
          message);
    for (int stage = 0; stage < G1FrameStageCount; ++stage) {
        const int expected_runner = stage <= last_runner_stage ? 1 : 0;
        const int expected_hook = stage <= last_hook_stage ? 1 : 0;
        check(trace.runner_calls[stage] == expected_runner &&
                  trace.hook_calls[stage] == expected_hook,
              message);
    }
    for (int index = 0; index <= last_runner_stage; ++index) {
        check(trace.runner_order[index] ==
                  static_cast<G1FrameTransactionStage>(index),
              message);
    }
    for (int index = 0; index <= last_hook_stage; ++index) {
        check(trace.hook_order[index] ==
                  static_cast<G1FrameTransactionStage>(index),
              message);
    }
}

static G1CommandIntent expected_requested_intent(const fixture& value)
{
    G1CommandIntent expected;
    expected.requested_velocity = value.external.input.move_stick;
    expected.desired_heading = value.runtime.accepted_state.desired_rotation;
    return expected;
}

static void test_seam_injection_at_every_stage_is_atomic_and_ordered()
{
    for (int stage = 0; stage < G1FrameStageCount; ++stage) {
        fixture finite;
        const RuntimeEvidence finite_before =
            runtime_evidence(finite.runtime);
        const G1FramePublication publication_before =
            finite.runtime.publication;
        const G1CommandIntent expected_intent =
            expected_requested_intent(finite);
        G1FrameTransactionTestSeam seam;
        seam.hook = injection_hook;
        seam.control.injected_stage =
            static_cast<G1FrameTransactionStage>(stage);
        seam.control.injected_outcome = G1FrameInjectFiniteReject;
        reset_trace();
        char error[512] = {};
        check(g1_frame_transaction_run(
                  finite.runtime, test_runner, finite.external, &seam,
                  error, static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionFiniteRejected,
              error);
        check_finite_preservation(
            finite.runtime, finite_before,
            "finite seam rejection rolls back accepted logic, both storage partitions, and diagnostic");
        check_finite_publication(
            finite.runtime, publication_before, expected_intent,
            finite.external.input.presentation_frame);
        check_stage_trace(
            stage, stage,
            "finite seam injection runs each real prior stage and no later stage");

        fixture global;
        const RuntimeEvidence global_before =
            runtime_evidence(global.runtime);
        seam.control.injected_outcome = G1FrameInjectGlobalError;
        reset_trace();
        check(g1_frame_transaction_run(
                  global.runtime, test_runner, global.external, &seam,
                  error, static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionGlobalError,
              "global seam injection returns controlled status");
        check_global_preservation(
            global.runtime, global_before,
            "global seam injection preserves the complete runtime publication unit");
        check_stage_trace(
            stage, stage,
            "global seam injection runs each real prior stage and no later stage");
    }
}

static void test_runner_outcomes_precede_hook_and_stop_later_stages()
{
    for (int stage = 0; stage < G1FrameStageCount; ++stage) {
        fixture finite;
        const RuntimeEvidence finite_before =
            runtime_evidence(finite.runtime);
        const G1FramePublication publication_before =
            finite.runtime.publication;
        G1FrameTransactionTestSeam seam;
        seam.hook = injection_hook;
        reset_trace();
        trace.runner_outcome_stage =
            static_cast<G1FrameTransactionStage>(stage);
        trace.runner_outcome = G1FrameStageFiniteReject;
        char error[512] = {};
        check(g1_frame_transaction_run(
                  finite.runtime, test_runner, finite.external, &seam,
                  error, static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionFiniteRejected,
              error);
        check_finite_preservation(
            finite.runtime, finite_before,
            "runner finite outcome preserves accepted owners and both storage partitions");
        check_finite_publication(
            finite.runtime, publication_before,
            expected_requested_intent(finite),
            finite.external.input.presentation_frame);
        check_stage_trace(
            stage, stage - 1,
            "hook follows Continue only and never follows runner finite outcome");

        fixture global;
        const RuntimeEvidence global_before =
            runtime_evidence(global.runtime);
        reset_trace();
        trace.runner_outcome_stage =
            static_cast<G1FrameTransactionStage>(stage);
        trace.runner_outcome = G1FrameStageGlobalError;
        check(g1_frame_transaction_run(
                  global.runtime, test_runner, global.external, &seam,
                  error, static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionGlobalError,
              "runner global outcome returns controlled status");
        check_global_preservation(
            global.runtime, global_before,
            "runner global outcome preserves all accepted/publication owners");
        check_stage_trace(
            stage, stage - 1,
            "hook follows Continue only and never follows runner global outcome");
    }
}

static void test_hostile_runner_outputs_are_rejected_before_publication()
{
    for (int raw_mode = HostileRunnerMissingIntent;
         raw_mode <= HostileRunnerMalformedAcceptedDiagnostic;
         ++raw_mode) {
        fixture value;
        const RuntimeEvidence before = runtime_evidence(value.runtime);
        G1FrameTransactionTestSeam seam;
        seam.hook = injection_hook;
        reset_trace();
        trace.hostile_mode = static_cast<HostileRunnerMode>(raw_mode);
        char error[512] = {};
        check(g1_frame_transaction_run(
                  value.runtime, hostile_runner, value.external, &seam,
                  error, static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionGlobalError,
              "invalid runner output is promoted to a controlled global error");
        check_global_preservation(
            value.runtime, before,
            "invalid runner output cannot publish, swap, or replace accepted diagnostics");
        if (raw_mode >= HostileRunnerMissingRejection &&
            raw_mode <= HostileRunnerNoncanonicalRejection) {
            check_stage_trace(
                G1FrameStageFootprintObservation,
                G1FrameStageContactUpdate,
                "invalid finite result stops before its hook and every later stage");
        } else {
            check_stage_trace(
                G1FrameStageCount - 1,
                G1FrameStageCount - 1,
                "invalid success candidate is detected after the complete ordered run");
        }
    }
}

static scene_pack make_success_candidate_scene()
{
    scene_pack scene = make_scene();
    scene.metadata.id = "transaction-success-candidate";
    scene.metadata.spawn_position = vec3(3.0f, 0.0f, 3.0f);
    scene.metadata.spawn_yaw = -0.53f;
    return scene;
}

static void materialize_enabled_ik_success_candidate(
    g1_controller_state& candidate,
    const fixture& value,
    const scene_pack& candidate_scene,
    char* error,
    int error_capacity)
{
    check(g1_ik_frame_evaluate(
              candidate.ik_bone_positions,
              candidate.ik_bone_rotations,
              candidate.ik,
              candidate.adjusted_bone_positions,
              candidate.adjusted_bone_rotations,
              value.db.bone_parents,
              candidate.curr_bone_contacts,
              candidate_scene.terrain,
              candidate.footprint,
              true,
              1.0f / 25.0f,
              candidate.ik_frame,
              error,
              error_capacity),
          error);
    check(candidate.ik_frame.applied &&
              !candidate.ik_frame.safe_stop_requested &&
              candidate.ik_frame.stop_reason == G1IkStopNone,
          "full success candidate contains a real accepted enabled-IK result");
    check(g1_ik_checked_forward_kinematics(
              candidate.ik_global_bone_positions,
              candidate.ik_global_bone_rotations,
              candidate.ik_bone_positions,
              candidate.ik_bone_rotations,
              value.db.bone_parents,
              error,
              error_capacity),
          error);
    std::memcpy(
        candidate.ik_candidate_bone_positions.data,
        candidate.ik_bone_positions.data,
        static_cast<std::size_t>(G1_BoneCount) * sizeof(vec3));
    std::memcpy(
        candidate.ik_candidate_bone_rotations.data,
        candidate.ik_bone_rotations.data,
        static_cast<std::size_t>(G1_BoneCount) * sizeof(quat));
    std::memcpy(
        candidate.ik_candidate_global_bone_positions.data,
        candidate.ik_global_bone_positions.data,
        static_cast<std::size_t>(G1_BoneCount) * sizeof(vec3));
    std::memcpy(
        candidate.ik_candidate_global_bone_rotations.data,
        candidate.ik_global_bone_rotations.data,
        static_cast<std::size_t>(G1_BoneCount) * sizeof(quat));
    check(g1_measure_pose_clearance(
              candidate.ik_clearance,
              g1_pose_clearance_budget(),
              candidate_scene.terrain,
              candidate.ik_global_bone_positions,
              candidate.ik_global_bone_rotations,
              error,
              error_capacity) == G1ClearanceOk,
          error);
    candidate.ik_candidate_clearance = candidate.ik_clearance;
    candidate.ik_candidate_clearance_status = G1ClearanceOk;
    candidate.ik_candidate_rejected = false;
}

static void build_valid_full_success_candidate(
    g1_controller_state& candidate,
    scene_pack& candidate_scene,
    const fixture& value,
    char* error,
    int error_capacity)
{
    check(g1_controller_state_reset_configured(
              candidate, value.db, value.support, candidate_scene,
              value.external.tuning.initial_search_time,
              false, 1.0f / 25.0f, 1.0f / 3.0f,
              error, error_capacity),
          error);
    materialize_enabled_ik_success_candidate(
        candidate, value, candidate_scene, error, error_capacity);
    seed_nonzero_array_tails(candidate);
    candidate.frame_index = 7;
    candidate.scene_frame = 5;
    candidate.search_timer = 0.50f;
    candidate.force_search_timer = 0.75f;
    candidate.transition_src_position = vec3(0.11f, -0.12f, 0.13f);
    candidate.transition_src_rotation = quat_from_angle_axis(
        0.23f, vec3(0.0f, 1.0f, 0.0f));
    candidate.desired_velocity = vec3(0.20f, 0.03f, -0.10f);
    candidate.desired_velocity_change_curr =
        vec3(0.04f, -0.05f, 0.06f);
    candidate.desired_velocity_change_prev =
        vec3(-0.07f, 0.08f, -0.09f);
    candidate.desired_rotation_change_curr =
        vec3(0.10f, 0.11f, 0.12f);
    candidate.desired_rotation_change_prev =
        vec3(-0.13f, -0.14f, -0.15f);
    candidate.desired_gait = 0.30f;
    candidate.desired_gait_velocity = -0.16f;
    candidate.simulation_velocity = vec3(0.17f, -0.18f, 0.19f);
    candidate.simulation_acceleration = vec3(-0.20f, 0.21f, -0.22f);
    candidate.simulation_angular_velocity =
        vec3(0.23f, -0.24f, 0.25f);
    candidate.command.intent.requested_velocity =
        vec3(-0.26f, 0.27f, 0.28f);
    candidate.command.applied_velocity = candidate.desired_velocity;
    candidate.support.velocity = 0.29f;
    candidate.support.nominal_height += 0.30f;
    candidate.support.nominal_velocity = -0.31f;
    candidate.support.offset_height = 0.32f;
    candidate.support.offset_velocity = -0.33f;
    candidate.support.airborne_frames = 4;
    candidate.support.source = support_held;
    for (int index = 0; index < 3; ++index) {
        const float base = 0.34f + 0.03f * static_cast<float>(index);
        candidate.support_observation_now.source_height[index] = base;
        candidate.support_observation_now.runtime_height[index] = base + 0.01f;
        candidate.support_observation_now.delta[index] = base + 0.02f;
    }
    candidate.support_observation_now.contact[0] =
        candidate.curr_bone_contacts(0);
    candidate.support_observation_now.contact[1] =
        candidate.curr_bone_contacts(1);
    candidate.traversal_speed_scale = 0.72f;
    candidate.traversal_speed_scale_velocity = -0.43f;
    candidate.blocked = true;
    candidate.walkability_class = 2;
    candidate.blocked_distance = 0.44f;
    candidate.blocked_point = vec3(-0.45f, 0.46f, -0.47f);
    candidate.route_index = 1;
    candidate.route_waypoint = 3;
    candidate.route_frames = 17;
    candidate.camera_altitude = 0.58f;
    candidate.camera_distance = 5.9f;
    candidate.searched = true;
    candidate.transitioned = true;
    candidate.incumbent_cost = 0.60f;
    candidate.selected_cost = 0.61f;
    candidate.selected_terrain_error = 0.62f;
    candidate.adjustment_xz = 0.63f;
    candidate.clamp_xz = 0.64f;
    check(g1_controller_state_is_valid(candidate),
          "full success candidate remains independently certified after every legal mutation");
    check(state_logical_digest(candidate) !=
              state_logical_digest(value.runtime.accepted_state),
          "full success candidate is observably distinct from the prior accepted state");
}

static void build_expected_success(
    g1_controller_state& expected_state,
    G1FrameAcceptedDiagnostic& expected_diagnostic,
    const fixture& value,
    const g1_controller_state* success_candidate,
    char* error,
    int error_capacity)
{
    check(g1_controller_state_reset(
              expected_state, value.db, value.support, value.scene,
              error, error_capacity),
          error);
    check(g1_controller_state_copy(
              expected_state,
              success_candidate != NULL
                  ? *success_candidate
                  : value.runtime.accepted_state,
              error, error_capacity),
          error);
    G1FrameTransactionScratch scratch;
    scratch.prior_safe_stop_latched =
        value.runtime.publication.ik_safe_stop_latched;
    scratch.commanded_velocity = value.external.input.move_stick;
    check(g1_ik_safe_stop_handoff(
              scratch.safe_stop_handoff,
              scratch.prior_safe_stop_latched,
              scratch.commanded_velocity,
              error, error_capacity),
          error);
    apply_latch_to_working_state(expected_state, scratch.safe_stop_handoff);
    scratch.traversal_input = scratch.safe_stop_handoff.applied_velocity;
    scratch.requested_intent.requested_velocity =
        value.external.input.move_stick;
    scratch.requested_intent.desired_heading = expected_state.desired_rotation;
    scratch.requested_intent_ready = true;
    for (int stage = 0; stage < G1FrameStageCount; ++stage) {
        const G1FrameTransactionStage typed_stage =
            static_cast<G1FrameTransactionStage>(stage);
        if (typed_stage == G1FrameStagePoseCertificate) {
            expected_diagnostic = accepted_diagnostic_candidate(
                expected_state, scratch, value.external);
        }
        apply_stage_marker(expected_state, typed_stage);
    }
    apply_success_tail(
        expected_state, scratch.prior_safe_stop_latched);
    check(g1_controller_state_is_valid(expected_state),
          "independently constructed success state is certified");
}

static void test_success_iteration_provenance_is_checked_before_publication()
{
    fixture value;
    scene_pack candidate_scene = make_success_candidate_scene();
    g1_controller_state success_candidate;
    char error[512] = {};
    build_valid_full_success_candidate(
        success_candidate, candidate_scene, value,
        error, static_cast<int>(sizeof(error)));
    check(success_candidate.ik_frame.feet[0].position.iterations == 1 &&
              success_candidate.ik_frame.feet[0]
                      .position.iteration_provenance ==
                  G1LegIterationContact1 &&
              terrain_float_bits(
                  success_candidate.ik_frame.feet[0]
                      .position.max_correction_radians) == 0U,
          "accepted publication fixture owns an authentic one-pass identity solve");

    G1FrameTransactionTestSeam seam;
    seam.hook = injection_hook;
    for (int forged_iterations = 2;
         forged_iterations <= G1ContactSolveMaximumIterations;
         ++forged_iterations) {
        success_candidate.ik_frame.feet[0].position.iterations =
            forged_iterations;
        check(!g1_controller_state_is_valid(success_candidate) &&
                  !g1_frame_success_iteration_provenance_is_valid(
                      success_candidate.ik_frame),
              "forged accepted iteration fails structural and publication provenance");
        const RuntimeEvidence before = runtime_evidence(value.runtime);
        reset_trace();
        trace.success_candidate = &success_candidate;
        check(g1_frame_transaction_run(
                  value.runtime, test_runner, value.external, &seam,
                  error, static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionGlobalError,
              "accepted publication rejects every in-range identity iteration forgery");
        check_global_preservation(
            value.runtime, before,
            "accepted iteration forgery cannot publish or replace accepted owners");
        check_stage_trace(
            G1FrameStageInputRouteCommand, -1,
            "accepted iteration forgery is rejected before any publication hook");
    }
    success_candidate.ik_frame.feet[0].position.iterations = 1;
    success_candidate.ik_frame.feet[0].position.iteration_provenance =
        G1LegIterationDirect1;
    check(!g1_controller_state_is_valid(success_candidate) &&
              !g1_frame_success_iteration_provenance_is_valid(
                  success_candidate.ik_frame),
          "completed Contact1 publication rejects enum-only Direct1 relabeling");

    success_candidate.ik_frame.feet[0].position.iteration_provenance =
        G1LegIterationContact1;
    success_candidate.ik_frame.feet[0]
        .position.iteration_provenance =
            G1LegIterationBaselineFallback1;
    check(g1_controller_state_is_valid(success_candidate) &&
              g1_frame_success_iteration_provenance_is_valid(
                  success_candidate.ik_frame),
          "identity Contact1-to-fallback relabel remains structurally valid before producer authentication");
    success_candidate.ik_frame.feet[0].position.iteration_provenance =
        G1LegIterationContact1;
    const RuntimeEvidence before = runtime_evidence(value.runtime);
    reset_trace();
    trace.success_candidate = &success_candidate;
    trace.mutate_working_ik_provenance_after_transcript = true;
    check(g1_frame_transaction_run(
              value.runtime, test_runner, value.external, &seam,
              error, static_cast<int>(sizeof(error))) ==
              G1FrameTransactionGlobalError,
          "accepted publication rejects a structurally valid provenance relabel after FinalFk evidence");
    check_global_preservation(
        value.runtime, before,
        "accepted provenance relabel cannot publish against immutable FinalFk evidence");
    check_stage_trace(
        G1FrameStageCount - 1, G1FrameStageCount - 1,
        "accepted provenance relabel is authenticated at the final publication boundary");

    fixture disabled;
    G1IkFrameTransaction disabled_producer;
    disabled_producer.initialized = true;
    disabled_producer.next_foot = 2U;
    disabled_producer.candidate_state =
        disabled.runtime.accepted_state.ik;
    disabled_producer.candidate_result =
        disabled.runtime.accepted_state.ik_frame;
    check(!disabled.runtime.accepted_state.ik_frame.applied &&
              g1_ik_runtime_is_disabled_noop(disabled_producer) &&
              g1_frame_success_ik_matches_producer(
                  disabled.runtime.accepted_state.ik_frame,
                  disabled_producer),
          "canonical disabled success authenticates to its genuine producer transaction");
    G1IkFrameResult mutated_disabled_result =
        disabled.runtime.accepted_state.ik_frame;
    mutated_disabled_result.max_correction_radians = -0.0f;
    check(!g1_frame_success_ik_matches_producer(
              mutated_disabled_result, disabled_producer),
          "disabled success has no bypass for a one-field result mutation");
    const RuntimeEvidence disabled_before =
        runtime_evidence(disabled.runtime);
    reset_trace();
    trace.mutate_disabled_ik_result_after_transcript = true;
    check(g1_frame_transaction_run(
              disabled.runtime, test_runner, disabled.external, &seam,
              error, static_cast<int>(sizeof(error))) ==
              G1FrameTransactionGlobalError,
          "disabled accepted publication rejects a one-field working-result mutation");
    check_global_preservation(
        disabled.runtime, disabled_before,
        "disabled working-result mutation cannot publish or swap accepted owners");
    check_stage_trace(
        G1FrameStageCount - 1, G1FrameStageCount - 1,
        "disabled working-result mutation is rejected at final publication");
}

static void check_success_swap_observability(
    const g1_controller_state& prior_accepted,
    const g1_controller_state& published_candidate)
{
#define REQUIRE_CHANGED_VALUE(name) \
    check(value_logical_digest(prior_accepted.name) != \
              value_logical_digest(published_candidate.name), \
          "valid success candidate changes scalar owner " #name)
#define REQUIRE_CHANGED_AGGREGATE(name, digest) \
    check(digest(prior_accepted.name) != digest(published_candidate.name), \
          "valid success candidate changes aggregate owner " #name)
    REQUIRE_CHANGED_VALUE(frame_index);
    REQUIRE_CHANGED_VALUE(scene_frame);
    REQUIRE_CHANGED_VALUE(search_timer);
    REQUIRE_CHANGED_VALUE(force_search_timer);
    REQUIRE_CHANGED_VALUE(transition_src_position);
    REQUIRE_CHANGED_VALUE(transition_dst_position);
    REQUIRE_CHANGED_VALUE(transition_src_rotation);
    REQUIRE_CHANGED_VALUE(transition_dst_rotation);
    REQUIRE_CHANGED_VALUE(desired_velocity);
    REQUIRE_CHANGED_VALUE(desired_velocity_change_curr);
    REQUIRE_CHANGED_VALUE(desired_velocity_change_prev);
    REQUIRE_CHANGED_VALUE(desired_rotation);
    REQUIRE_CHANGED_VALUE(desired_rotation_change_curr);
    REQUIRE_CHANGED_VALUE(desired_rotation_change_prev);
    REQUIRE_CHANGED_VALUE(desired_gait);
    REQUIRE_CHANGED_VALUE(desired_gait_velocity);
    REQUIRE_CHANGED_VALUE(simulation_position);
    REQUIRE_CHANGED_VALUE(simulation_velocity);
    REQUIRE_CHANGED_VALUE(simulation_acceleration);
    REQUIRE_CHANGED_VALUE(simulation_rotation);
    REQUIRE_CHANGED_VALUE(simulation_angular_velocity);
    REQUIRE_CHANGED_AGGREGATE(command, command_logical_digest);
    REQUIRE_CHANGED_AGGREGATE(footprint, footprint_logical_digest);
    REQUIRE_CHANGED_AGGREGATE(ik, ik_state_logical_digest);
    REQUIRE_CHANGED_AGGREGATE(ik_frame, ik_frame_logical_digest);
    REQUIRE_CHANGED_AGGREGATE(
        ik_clearance, pose_clearance_logical_digest);
    REQUIRE_CHANGED_AGGREGATE(
        ik_candidate_clearance, pose_clearance_logical_digest);
    REQUIRE_CHANGED_AGGREGATE(support, support_logical_digest);
    REQUIRE_CHANGED_AGGREGATE(
        support_observation_now, support_observation_logical_digest);
    REQUIRE_CHANGED_VALUE(traversal_speed_scale);
    REQUIRE_CHANGED_VALUE(traversal_speed_scale_velocity);
    REQUIRE_CHANGED_VALUE(blocked);
    REQUIRE_CHANGED_VALUE(walkability_class);
    REQUIRE_CHANGED_VALUE(blocked_distance);
    REQUIRE_CHANGED_VALUE(blocked_point);
    REQUIRE_CHANGED_VALUE(route_index);
    REQUIRE_CHANGED_VALUE(route_waypoint);
    REQUIRE_CHANGED_VALUE(route_frames);
    REQUIRE_CHANGED_VALUE(camera_azimuth);
    REQUIRE_CHANGED_VALUE(camera_altitude);
    REQUIRE_CHANGED_VALUE(camera_distance);
    REQUIRE_CHANGED_VALUE(searched);
    REQUIRE_CHANGED_VALUE(transitioned);
    REQUIRE_CHANGED_VALUE(incumbent_cost);
    REQUIRE_CHANGED_VALUE(selected_cost);
    REQUIRE_CHANGED_VALUE(selected_terrain_error);
    REQUIRE_CHANGED_VALUE(adjustment_xz);
    REQUIRE_CHANGED_VALUE(clamp_xz);
#undef REQUIRE_CHANGED_AGGREGATE
#undef REQUIRE_CHANGED_VALUE

    check(value_logical_digest(prior_accepted.search_time) ==
              value_logical_digest(published_candidate.search_time) &&
              prior_accepted.footprint_status == G1FootprintOk &&
              published_candidate.footprint_status == G1FootprintOk &&
              prior_accepted.ik_candidate_clearance_status ==
                  G1ClearanceOk &&
              published_candidate.ik_candidate_clearance_status ==
                  G1ClearanceOk &&
              !prior_accepted.ik_candidate_rejected &&
              !published_candidate.ik_candidate_rejected &&
              terrain_float_bits(prior_accepted.adjustment_y) == 0U &&
              terrain_float_bits(published_candidate.adjustment_y) == 0U &&
              terrain_float_bits(prior_accepted.clamp_y) == 0U &&
              terrain_float_bits(published_candidate.clamp_y) == 0U,
          "only immutable search time and five validity-derived constant owners remain equal");
}

static void test_success_swaps_every_owner_and_publishes_complete_diagnostic()
{
    fixture value;
    const uint64_t accepted_before =
        state_logical_digest(value.runtime.accepted_state);
    const StateStorageIdentities accepted_storage_before =
        state_storage_identities(value.runtime.accepted_state);
    const StateStorageIdentities working_storage_before =
        state_storage_identities(value.runtime.working_state);
    check(storage_identities_are_mutually_disjoint(
              accepted_storage_before, working_storage_before),
          "all 98 reset-owned buffers are mutually disjoint before the swap oracle");
    scene_pack candidate_scene = make_success_candidate_scene();
    g1_controller_state success_candidate;
    g1_controller_state expected_state;
    G1FrameAcceptedDiagnostic expected_diagnostic;
    char error[512] = {};
    build_valid_full_success_candidate(
        success_candidate, candidate_scene, value,
        error, static_cast<int>(sizeof(error)));
    build_expected_success(
        expected_state, expected_diagnostic, value, &success_candidate,
        error, static_cast<int>(sizeof(error)));
    check_success_swap_observability(
        value.runtime.accepted_state, expected_state);

    G1FrameTransactionTestSeam seam;
    seam.hook = injection_hook;
    reset_trace();
    trace.require_exact_initial_copy = true;
    trace.expected_initial_copy_digest = accepted_before;
    trace.success_candidate = &success_candidate;
    check(g1_frame_transaction_run(
              value.runtime, test_runner, value.external, &seam,
              error, static_cast<int>(sizeof(error))) ==
              G1FrameTransactionAccepted,
          error);
    check_stage_trace(
        G1FrameStageCount - 1, G1FrameStageCount - 1,
        "accepted transaction runs all 12 stages exactly once in order");
    check(trace.exact_initial_copy_seen,
          "runner observes a complete accepted-to-working copy before any stage mutation");
    check(g1_controller_state_is_valid(value.runtime.accepted_state) &&
              state_logical_digest(value.runtime.accepted_state) ==
                  state_logical_digest(expected_state),
          "accepted state equals the independently constructed certified result");
    check(state_logical_digest(value.runtime.working_state) == accepted_before,
          "single success swap leaves the prior accepted value in working state");
    check(same_storage_identities(
              state_storage_identities(value.runtime.accepted_state),
              working_storage_before) &&
          same_storage_identities(
              state_storage_identities(value.runtime.working_state),
              accepted_storage_before),
          "success performs one exact 49-owner accepted/working identity swap");
    check(diagnostic_logical_digest(value.runtime.accepted_diagnostic) ==
              diagnostic_logical_digest(expected_diagnostic),
          "success publishes every accepted diagnostic field exactly");
    G1FramePublication expected_publication;
    expected_publication.requested_intent.requested_velocity =
        value.external.input.move_stick;
    expected_publication.requested_intent.desired_heading =
        success_candidate.desired_rotation;
    expected_publication.presentation_frame =
        value.external.input.presentation_frame;
    check(publication_logical_digest(value.runtime.publication) ==
              publication_logical_digest(expected_publication),
          "success publishes exact intent/frame and clears rejection/latch");
    check(terrain_float_bits(
              value.runtime.accepted_state
                  .bone_offset_velocities(G1_BoneCount - 1).x) ==
              terrain_float_bits(0.001f) &&
          terrain_float_bits(
              value.runtime.accepted_state
                  .trajectory_accelerations(
                      G1CommandTrajectorySampleCount - 1).z) ==
              terrain_float_bits(-0.009f) &&
          terrain_float_bits(
              value.runtime.accepted_state
                  .contact_offset_velocities(1).z) ==
              terrain_float_bits(0.012f),
          "accepted success preserves nonzero tails across disparate owners");
}

static float poison_float(int salt)
{
    uint32_t bits = UINT32_C(0x7fc00000) |
        static_cast<uint32_t>(salt & 0x003fffff);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static void poison_array_values(array1d<vec3>& values, int& salt)
{
    for (int index = 0; index < values.size; ++index) {
        const float x = poison_float(salt++);
        const float y = poison_float(salt++);
        const float z = poison_float(salt++);
        values(index) = vec3(x, y, z);
    }
}

static void poison_array_values(array1d<quat>& values, int& salt)
{
    for (int index = 0; index < values.size; ++index) {
        const float w = poison_float(salt++);
        const float x = poison_float(salt++);
        const float y = poison_float(salt++);
        const float z = poison_float(salt++);
        values(index) = quat(w, x, y, z);
    }
}

static void poison_array_values(array1d<bool>& values, int& salt)
{
    for (int index = 0; index < values.size; ++index) {
        values(index) = ((salt++ + index) & 1) != 0;
    }
}

static void poison_array_values(array1d<int>& values, int& salt)
{
    for (int index = 0; index < values.size; ++index) {
        values(index) = -100000 - salt++ - index;
    }
}

static vec3 poison_vec3(int& salt)
{
    const float x = poison_float(salt++);
    const float y = poison_float(salt++);
    const float z = poison_float(salt++);
    return vec3(x, y, z);
}

static quat poison_quat(int& salt)
{
    const float w = poison_float(salt++);
    const float x = poison_float(salt++);
    const float y = poison_float(salt++);
    const float z = poison_float(salt++);
    return quat(w, x, y, z);
}

static void poison_surface(G1SurfaceSample& value, int& salt)
{
    value.height = poison_float(salt++);
    value.normal = poison_vec3(salt);
}

static void poison_clearance_work(G1ClearanceWork& value, int& salt)
{
    value.point_queries = static_cast<uint32_t>(salt++);
    value.cells_visited = static_cast<uint32_t>(salt++);
    value.primitive_triangle_pairs = static_cast<uint32_t>(salt++);
    value.face_patches = static_cast<uint32_t>(salt++);
    value.candidate_tests = static_cast<uint32_t>(salt++);
    value.subdivision_nodes = static_cast<uint32_t>(salt++);
}

static void poison_clearance_result(G1ClearanceResult& value, int& salt)
{
    value.lower_bound_m = -1000.0 - static_cast<double>(salt++);
    value.witness_upper_m = -1000.0 - static_cast<double>(salt++);
    value.witness.body_x = -1000.0 - static_cast<double>(salt++);
    value.witness.body_y = -1000.0 - static_cast<double>(salt++);
    value.witness.body_z = -1000.0 - static_cast<double>(salt++);
    value.witness.surface_x = -1000.0 - static_cast<double>(salt++);
    value.witness.surface_y = -1000.0 - static_cast<double>(salt++);
    value.witness.surface_z = -1000.0 - static_cast<double>(salt++);
    value.witness.segment_parameter =
        -1000.0 - static_cast<double>(salt++);
    value.witness.terrain_weight_0 =
        -1000.0 - static_cast<double>(salt++);
    value.witness.terrain_weight_1 =
        -1000.0 - static_cast<double>(salt++);
    value.witness.terrain_weight_2 =
        -1000.0 - static_cast<double>(salt++);
    value.witness.primitive_index = static_cast<uint32_t>(salt++);
    value.witness.cell_x = -salt++;
    value.witness.cell_z = -salt++;
    value.witness.terrain_triangle_index = static_cast<uint32_t>(salt++);
    value.witness.patch_index = static_cast<uint32_t>(salt++);
    value.witness.candidate_kind = static_cast<uint32_t>(salt++);
    value.witness.candidate_subindex = static_cast<uint32_t>(salt++);
    poison_clearance_work(value.work, salt);
}

static void poison_leg_clearance(G1LegClearance& value, int& salt)
{
    poison_clearance_result(value.knee, salt);
    poison_clearance_result(value.ankle, salt);
    poison_clearance_result(value.toe, salt);
    poison_clearance_result(value.foot, salt);
    poison_clearance_result(value.thigh, salt);
    poison_clearance_result(value.shin, salt);
    poison_clearance_result(value.minimum, salt);
}

static void poison_pose_clearance(G1PoseClearance& value, int& salt)
{
    poison_clearance_result(value.hips, salt);
    poison_leg_clearance(value.left, salt);
    poison_leg_clearance(value.right, salt);
    poison_clearance_result(value.minimum, salt);
}

static void poison_footprint(
    G1FootprintObservation& value, int& salt)
{
    poison_surface(value.root_surface, salt);
    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        G1FootprintFootObservation& foot = value.feet[foot_index];
        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            G1FootprintProbe& probe = foot.probes[probe_index];
            probe.current_sphere_center = poison_vec3(salt);
            probe.current_sole_point = poison_vec3(salt);
            poison_surface(probe.current_surface, salt);
            for (int sample = 0;
                 sample < G1CommandTrajectorySampleCount;
                 ++sample) {
                probe.predicted_sphere_centers[sample] = poison_vec3(salt);
                probe.predicted_sole_points[sample] = poison_vec3(salt);
                probe.predicted_surface_status[sample] =
                    static_cast<G1SurfaceQueryStatus>(20 + salt++);
                poison_surface(probe.predicted_surfaces[sample], salt);
            }
            poison_surface(probe.selected_landing_surface, salt);
            probe.corridor_minimum_height = poison_float(salt++);
            probe.corridor_maximum_height = poison_float(salt++);
            probe.encountered_walkability_class = -salt++;
        }
        foot.current_contact = ((salt++ + foot_index) & 1) != 0;
        foot.landing_expected = ((salt++ + foot_index) & 1) != 0;
        foot.landing_patch_ready = ((salt++ + foot_index) & 1) != 0;
        foot.landing_sample = static_cast<uint32_t>(salt++);
        foot.predicted_landing_sole_center = poison_vec3(salt);
        foot.predicted_landing_surface_status =
            static_cast<G1SurfaceQueryStatus>(40 + salt++);
        poison_surface(foot.predicted_landing_surface, salt);
        foot.predicted_landing_walkability_class = -salt++;
        foot.landing_patch_maximum_residual_m =
            -2000.0 - static_cast<double>(salt++);
        foot.corridor_minimum_height = poison_float(salt++);
        foot.corridor_maximum_height = poison_float(salt++);
        foot.maximum_root_split_m =
            -2000.0 - static_cast<double>(salt++);
        foot.encountered_walkability_class = -salt++;
        foot.multilevel = ((salt++ + foot_index) & 1) != 0;
    }
    value.blocked = true;
    value.blocked_reason = static_cast<walkability_reason>(20 + salt++);
    value.work.sweeps = static_cast<uint32_t>(salt++);
    value.work.surface_queries = static_cast<uint32_t>(salt++);
    value.work.node_visits = static_cast<uint32_t>(salt++);
}

static void poison_lock(G1FootLockState& value, int& salt)
{
    value.initialized = ((salt++) & 1) != 0;
    value.contact = ((salt++) & 1) != 0;
    value.locked = ((salt++) & 1) != 0;
    value.position_active = ((salt++) & 1) != 0;
    value.releasing = ((salt++) & 1) != 0;
    value.release_frames = -salt++;
    value.previous_input = poison_vec3(salt);
    value.lock_point = poison_vec3(salt);
    value.output_position = poison_vec3(salt);
    value.output_velocity = poison_vec3(salt);
    value.offset_position = poison_vec3(salt);
    value.offset_velocity = poison_vec3(salt);
}

static void poison_ik_state(G1IkState& value, int& salt)
{
    value.initialized = false;
    for (int foot = 0; foot < 2; ++foot) {
        poison_lock(value.feet[foot].lock, salt);
        value.feet[foot].swing.initialized = ((salt++) & 1) != 0;
        for (int probe = 0; probe < 4; ++probe) {
            value.feet[foot].swing.previous_sphere_centers[probe] =
                poison_vec3(salt);
        }
        value.feet[foot].baseline_sole_normal = poison_vec3(salt);
    }
}

static void poison_target(G1FootTarget& value, int& salt)
{
    value.locked = ((salt++) & 1) != 0;
    value.position_active = ((salt++) & 1) != 0;
    value.releasing = ((salt++) & 1) != 0;
    value.drift_limit_exceeded = ((salt++) & 1) != 0;
    value.surface.point = poison_vec3(salt);
    value.surface.normal = poison_vec3(salt);
    value.desired_sole_normal = poison_vec3(salt);
    value.sole_center = poison_vec3(salt);
    value.horizontal_drift_m = poison_float(salt++);
}

static void poison_swing_candidate(
    G1SwingCandidateDiagnostic& value, int& salt)
{
    value.candidate_index = static_cast<uint32_t>(salt++);
    value.lift_bits = static_cast<uint32_t>(salt++);
    value.materialized_command_y_bits = static_cast<uint32_t>(salt++);
    for (int probe = 0; probe < 4; ++probe) {
        for (int axis = 0; axis < 3; ++axis) {
            value.actual_sphere_center_bits[probe][axis] =
                static_cast<uint32_t>(salt++);
        }
    }
    value.clearance_status = static_cast<G1ClearanceStatus>(50 + salt++);
    value.controller_constraints_passed = ((salt++) & 1) != 0;
    value.clearance_certified = ((salt++) & 1) != 0;
    value.lower_margin_m = -3000.0 - static_cast<double>(salt++);
    value.witness_upper_margin_m = -3000.0 - static_cast<double>(salt++);
    poison_clearance_work(value.clearance_work, salt);
}

static void poison_leg_result(G1LegSolveResult& value, int& salt)
{
    value.applied = ((salt++) & 1) != 0;
    value.reachable = ((salt++) & 1) != 0;
    value.correction_limited = ((salt++) & 1) != 0;
    value.safe_stop_requested = ((salt++) & 1) != 0;
    value.iterations = -salt++;
    value.iteration_provenance =
        ((salt++) & 1) != 0
            ? G1LegIterationContact4
            : G1LegIterationBaselineFallback1;
    value.requested_ankle_target = poison_vec3(salt);
    value.clamped_ankle_target = poison_vec3(salt);
    value.hinge_axis_world = poison_vec3(salt);
    value.bend_direction = poison_vec3(salt);
    value.bend_used_current_projection = ((salt++) & 1) != 0;
    value.bend_used_hinge_fallback = ((salt++) & 1) != 0;
    value.bend_used_safe_perpendicular = ((salt++) & 1) != 0;
    value.bend_sign_flipped = ((salt++) & 1) != 0;
    value.raw_distance_m = poison_float(salt++);
    value.clamped_distance_m = poison_float(salt++);
    value.max_correction_radians = poison_float(salt++);
    value.contact_residual_m = poison_float(salt++);
}

static void poison_orientation(
    G1FootOrientationResult& value, int& salt)
{
    value.applied = ((salt++) & 1) != 0;
    value.correction_limited = ((salt++) & 1) != 0;
    value.safe_stop_requested = ((salt++) & 1) != 0;
    value.target_global_rotation = poison_quat(salt);
    value.requested_correction_radians = poison_float(salt++);
    value.correction_radians = poison_float(salt++);
}

static void poison_frame_result(G1IkFrameResult& value, int& salt)
{
    value.applied = true;
    value.safe_stop_requested = true;
    value.stop_reason = static_cast<G1IkStopReason>(50 + salt++);
    value.max_correction_radians = poison_float(salt++);
    for (int foot = 0; foot < 2; ++foot) {
        G1FootFrameResult& result = value.feet[foot];
        result.recorded_contact = ((salt++ + foot) & 1) != 0;
        poison_target(result.target, salt);
        result.swing_selection.candidates_evaluated =
            static_cast<uint32_t>(salt++);
        result.swing_selection.selected_index =
            static_cast<uint32_t>(salt++);
        poison_swing_candidate(result.swing_selection.selected, salt);
        poison_clearance_work(
            result.swing_selection.total_clearance_work, salt);
        result.defensive_swing.lower_margin_m =
            -4000.0 - static_cast<double>(salt++);
        result.defensive_swing.witness_upper_m =
            -4000.0 - static_cast<double>(salt++);
        result.defensive_swing.sweep_evaluated = ((salt++) & 1) != 0;
        poison_clearance_work(result.defensive_swing.work, salt);
        poison_leg_result(result.position, salt);
        poison_orientation(result.orientation, salt);
    }
}

static void poison_every_working_value(g1_controller_state& state)
{
    int salt = 1;
#define POISON_ARRAY(name) poison_array_values(state.name, salt)
    POISON_ARRAY(curr_bone_positions);
    POISON_ARRAY(curr_bone_velocities);
    POISON_ARRAY(trns_bone_positions);
    POISON_ARRAY(trns_bone_velocities);
    POISON_ARRAY(curr_bone_rotations);
    POISON_ARRAY(trns_bone_rotations);
    POISON_ARRAY(curr_bone_angular_velocities);
    POISON_ARRAY(trns_bone_angular_velocities);
    POISON_ARRAY(curr_bone_contacts);
    POISON_ARRAY(trns_bone_contacts);
    POISON_ARRAY(bone_positions);
    POISON_ARRAY(bone_velocities);
    POISON_ARRAY(bone_angular_velocities);
    POISON_ARRAY(bone_rotations);
    POISON_ARRAY(bone_offset_positions);
    POISON_ARRAY(bone_offset_velocities);
    POISON_ARRAY(bone_offset_angular_velocities);
    POISON_ARRAY(bone_offset_rotations);
    POISON_ARRAY(adjusted_bone_positions);
    POISON_ARRAY(global_bone_positions);
    POISON_ARRAY(global_bone_velocities);
    POISON_ARRAY(adjusted_bone_rotations);
    POISON_ARRAY(global_bone_rotations);
    POISON_ARRAY(global_bone_angular_velocities);
    POISON_ARRAY(global_bone_computed);
    POISON_ARRAY(trajectory_desired_velocities);
    POISON_ARRAY(trajectory_positions);
    POISON_ARRAY(trajectory_velocities);
    POISON_ARRAY(trajectory_accelerations);
    POISON_ARRAY(trajectory_angular_velocities);
    POISON_ARRAY(trajectory_desired_rotations);
    POISON_ARRAY(trajectory_rotations);
    POISON_ARRAY(contact_bones);
    POISON_ARRAY(contact_states);
    POISON_ARRAY(contact_locks);
    POISON_ARRAY(contact_positions);
    POISON_ARRAY(contact_velocities);
    POISON_ARRAY(contact_points);
    POISON_ARRAY(contact_targets);
    POISON_ARRAY(contact_offset_positions);
    POISON_ARRAY(contact_offset_velocities);
    POISON_ARRAY(ik_bone_positions);
    POISON_ARRAY(ik_bone_rotations);
    POISON_ARRAY(ik_global_bone_positions);
    POISON_ARRAY(ik_global_bone_rotations);
    POISON_ARRAY(ik_candidate_bone_positions);
    POISON_ARRAY(ik_candidate_bone_rotations);
    POISON_ARRAY(ik_candidate_global_bone_positions);
    POISON_ARRAY(ik_candidate_global_bone_rotations);
#undef POISON_ARRAY
    state.frame_index = -salt++;
    state.scene_frame = -salt++;
    state.search_time = poison_float(salt++);
    state.search_timer = poison_float(salt++);
    state.force_search_timer = poison_float(salt++);
    state.transition_src_position = poison_vec3(salt);
    state.transition_dst_position = poison_vec3(salt);
    state.transition_src_rotation = poison_quat(salt);
    state.transition_dst_rotation = poison_quat(salt);
    state.desired_velocity = poison_vec3(salt);
    state.desired_velocity_change_curr = poison_vec3(salt);
    state.desired_velocity_change_prev = poison_vec3(salt);
    state.desired_rotation = poison_quat(salt);
    state.desired_rotation_change_curr = poison_vec3(salt);
    state.desired_rotation_change_prev = poison_vec3(salt);
    state.desired_gait = poison_float(salt++);
    state.desired_gait_velocity = poison_float(salt++);
    state.simulation_position = poison_vec3(salt);
    state.simulation_velocity = poison_vec3(salt);
    state.simulation_acceleration = poison_vec3(salt);
    state.simulation_rotation = poison_quat(salt);
    state.simulation_angular_velocity = poison_vec3(salt);
    state.command = G1CommandSnapshot{};
    state.command.intent.requested_velocity = poison_vec3(salt);
    state.command.intent.desired_heading = poison_quat(salt);
    state.command.applied_velocity = poison_vec3(salt);
    for (int sample = 0;
         sample < G1CommandTrajectorySampleCount;
         ++sample) {
        state.command.predicted_desired_velocities[sample] =
            poison_vec3(salt);
        state.command.predicted_root_positions[sample] = poison_vec3(salt);
        state.command.predicted_root_rotations[sample] = poison_quat(salt);
        state.command.predicted_desired_headings[sample] = poison_quat(salt);
    }
    state.footprint_status = G1FootprintInvalidField;
    poison_footprint(state.footprint, salt);
    poison_ik_state(state.ik, salt);
    poison_frame_result(state.ik_frame, salt);
    poison_pose_clearance(state.ik_clearance, salt);
    poison_pose_clearance(state.ik_candidate_clearance, salt);
    state.ik_candidate_clearance_status = G1ClearanceInvalidField;
    state.ik_candidate_rejected = true;
    state.support = support_frame_state{};
    state.support.height = poison_float(salt++);
    state.support.velocity = poison_float(salt++);
    state.support.nominal_height = poison_float(salt++);
    state.support.nominal_velocity = poison_float(salt++);
    state.support.offset_height = poison_float(salt++);
    state.support.offset_velocity = poison_float(salt++);
    state.support.airborne_frames = -salt++;
    state.support.source = static_cast<support_source>(20 + salt++);
    state.support.initialized = false;
    state.support_observation_now = support_observation{};
    for (int index = 0; index < 3; ++index) {
        state.support_observation_now.source_height[index] =
            poison_float(salt++);
        state.support_observation_now.runtime_height[index] =
            poison_float(salt++);
        state.support_observation_now.delta[index] = poison_float(salt++);
    }
    state.support_observation_now.contact[0] = true;
    state.support_observation_now.contact[1] = false;
    state.traversal_speed_scale = poison_float(salt++);
    state.traversal_speed_scale_velocity = poison_float(salt++);
    state.blocked = true;
    state.walkability_class = -salt++;
    state.blocked_distance = poison_float(salt++);
    state.blocked_point = poison_vec3(salt);
    state.route_index = -salt++;
    state.route_waypoint = -salt++;
    state.route_frames = -salt++;
    state.camera_azimuth = poison_float(salt++);
    state.camera_altitude = poison_float(salt++);
    state.camera_distance = poison_float(salt++);
    state.searched = true;
    state.transitioned = true;
    state.incumbent_cost = poison_float(salt++);
    state.selected_cost = poison_float(salt++);
    state.selected_terrain_error = poison_float(salt++);
    state.adjustment_xz = poison_float(salt++);
    state.adjustment_y = poison_float(salt++);
    state.clamp_xz = poison_float(salt++);
    state.clamp_y = poison_float(salt++);
}

static void test_dirty_working_storage_is_overwritten_completely()
{
    fixture clean;
    fixture dirty;
    const StateStorageIdentities clean_working_before =
        state_storage_identities(clean.runtime.working_state);
    const StateStorageIdentities dirty_working_before =
        state_storage_identities(dirty.runtime.working_state);
    poison_every_working_value(dirty.runtime.working_state);
    check(g1_controller_state_storage_is_valid(
              dirty.runtime.working_state) &&
          same_storage_identities(
              state_storage_identities(dirty.runtime.working_state),
              dirty_working_before) &&
          !g1_frame_vec3_bits_equal(
              dirty.runtime.working_state.bone_positions(0),
              dirty.runtime.accepted_state.bone_positions(0)) &&
          !g1_frame_vec3_bits_equal(
              dirty.runtime.working_state.ik.feet[0]
                  .baseline_sole_normal,
              dirty.runtime.accepted_state.ik.feet[0]
                  .baseline_sole_normal) &&
          !g1_frame_vec3_bits_equal(
              dirty.runtime.working_state.ik_frame.feet[0]
                  .target.desired_sole_normal,
              dirty.runtime.accepted_state.ik_frame.feet[0]
                  .target.desired_sole_normal) &&
          !g1_frame_float_bits_equal(
              dirty.runtime.working_state.camera_azimuth,
              dirty.runtime.accepted_state.camera_azimuth) &&
          dirty.runtime.working_state.route_frames !=
              dirty.runtime.accepted_state.route_frames,
          "dirty destination keeps exact storage while every owner is poisoned");
    G1FrameTransactionTestSeam seam;
    seam.hook = injection_hook;
    char error[512] = {};
    reset_trace();
    check(g1_frame_transaction_run(
              clean.runtime, test_runner, clean.external, &seam,
              error, static_cast<int>(sizeof(error))) ==
              G1FrameTransactionAccepted,
          error);
    reset_trace();
    check(g1_frame_transaction_run(
              dirty.runtime, test_runner, dirty.external, &seam,
              error, static_cast<int>(sizeof(error))) ==
              G1FrameTransactionAccepted,
          error);
    check(state_logical_digest(dirty.runtime.accepted_state) ==
              state_logical_digest(clean.runtime.accepted_state) &&
          diagnostic_logical_digest(dirty.runtime.accepted_diagnostic) ==
              diagnostic_logical_digest(clean.runtime.accepted_diagnostic) &&
          publication_logical_digest(dirty.runtime.publication) ==
              publication_logical_digest(clean.runtime.publication),
          "clean and fully dirty destinations publish identical complete values");
    check(same_storage_identities(
              state_storage_identities(clean.runtime.accepted_state),
              clean_working_before) &&
          same_storage_identities(
              state_storage_identities(dirty.runtime.accepted_state),
              dirty_working_before),
          "clean and dirty successes each publish their original working storage");
}

static void test_input_checkpoint_publishes_ready_intent_only()
{
    fixture value;
    const RuntimeEvidence before = runtime_evidence(value.runtime);
    const G1FramePublication publication_before = value.runtime.publication;
    G1FrameTransactionTestSeam seam;
    seam.hook = injection_hook;
    seam.control.injected_stage = G1FrameStageInputRouteCommand;
    seam.control.injected_outcome = G1FrameInjectFiniteReject;
    char error[512] = {};
    reset_trace();
    check(g1_frame_transaction_run(
              value.runtime, test_runner, value.external, &seam,
              error, static_cast<int>(sizeof(error))) ==
              G1FrameTransactionFiniteRejected,
          error);
    check_finite_preservation(
        value.runtime, before,
        "input checkpoint leaves accepted logic, both storage partitions, and diagnostic exact");
    check(trace.input_hook_ready &&
              trace.input_hook_velocity_exact &&
              trace.input_hook_heading_exact,
          "input hook sees ready exact velocity and independent heading bits");
    check_finite_publication(
        value.runtime, publication_before,
        expected_requested_intent(value),
        value.external.input.presentation_frame);
}

static void expect_preflight_global(
    fixture& value,
    G1FrameStageRunner runner,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity,
    const char* message)
{
    const RuntimeEvidence before = preflight_runtime_evidence(value.runtime);
    G1FrameTransactionTestSeam seam;
    seam.hook = injection_hook;
    reset_trace();
    check(g1_frame_transaction_run(
              value.runtime, runner, external, &seam,
              error, error_capacity) ==
              G1FrameTransactionGlobalError,
          message);
    check(trace.runner_total == 0 && trace.hook_total == 0,
          "preflight failure occurs before runner or hook");
    check_preflight_preservation(value.runtime, before, message);
    check(same_runtime_evidence(runtime_evidence(value.runtime), before),
          "preflight rejection preserves the complete accepted/working runtime value");
}

static void check_restored_storage_preflight_failure(
    fixture& value,
    const RuntimeEvidence& before,
    G1FrameTransactionStatus status,
    const char* message)
{
    check(status == G1FrameTransactionGlobalError,
          message);
    check(trace.runner_total == 0 && trace.hook_total == 0,
          "storage preflight failure occurs before runner or hook");
    check_preflight_preservation(value.runtime, before, message);
}

static G1FrameTransactionStatus run_storage_preflight_case(
    fixture& value,
    char* error,
    int error_capacity)
{
    G1FrameTransactionTestSeam seam;
    seam.hook = injection_hook;
    reset_trace();
    return g1_frame_transaction_run(
        value.runtime, test_runner, value.external, &seam,
        error, error_capacity);
}

template<class T>
using G1StateArrayMember = array1d<T> g1_controller_state::*;

template<class T>
static void exercise_owner_side_preflight_matrix(
    G1StateArrayMember<T> member,
    int owner_index,
    bool accepted_side)
{
    char error[512] = {};
    {
        fixture value;
        const RuntimeEvidence before =
            preflight_runtime_evidence(value.runtime);
        g1_controller_state& side = accepted_side
            ? value.runtime.accepted_state
            : value.runtime.working_state;
        array1d<T>& owner = side.*member;
        T* const saved = owner.data;
        owner.data = NULL;
        const G1FrameTransactionStatus status =
            run_storage_preflight_case(
                value, error, static_cast<int>(sizeof(error)));
        owner.data = saved;
        check_restored_storage_preflight_failure(
            value, before, status,
            "every null accepted/working owner is rejected before any write");
    }
    {
        fixture value;
        const RuntimeEvidence before =
            preflight_runtime_evidence(value.runtime);
        g1_controller_state& side = accepted_side
            ? value.runtime.accepted_state
            : value.runtime.working_state;
        array1d<T>& owner = side.*member;
        const int saved = owner.size;
        check(saved > 1,
              "every transaction owner has a positive reducible canonical size");
        --owner.size;
        const G1FrameTransactionStatus status =
            run_storage_preflight_case(
                value, error, static_cast<int>(sizeof(error)));
        owner.size = saved;
        check_restored_storage_preflight_failure(
            value, before, status,
            "every wrong-size accepted/working owner is rejected before any write");
    }
    {
        fixture value;
        const RuntimeEvidence before =
            preflight_runtime_evidence(value.runtime);
        g1_controller_state& side = accepted_side
            ? value.runtime.accepted_state
            : value.runtime.working_state;
        array1d<T>& owner = side.*member;
        const StateStorageIdentities identities =
            state_storage_identities(side);
        const int peer_index = owner_index == 0 ? 1 : 0;
        check(owner.data == identities.owners[owner_index].data &&
                  identities.owners[peer_index].bytes > sizeof(T),
              "typed owner index agrees with the exact 49-owner identity table");
        T* const saved = owner.data;
        unsigned char* const peer_bytes =
            static_cast<unsigned char*>(const_cast<void*>(
                identities.owners[peer_index].data));
        const uintptr_t partial_address =
            reinterpret_cast<uintptr_t>(peer_bytes + sizeof(T));
        check(partial_address % alignof(T) == 0U,
              "partial-overlap pointer preserves the target element alignment");
        owner.data = reinterpret_cast<T*>(partial_address);
        check(owner.data != identities.owners[peer_index].data &&
                  g1_ik_memory_ranges_overlap(
                      owner.data,
                      static_cast<std::size_t>(owner.size) * sizeof(T),
                      identities.owners[peer_index].data,
                      identities.owners[peer_index].bytes),
              "within-state mutation is a genuine partial owner overlap");
        const G1FrameTransactionStatus status =
            run_storage_preflight_case(
                value, error, static_cast<int>(sizeof(error)));
        owner.data = saved;
        check_restored_storage_preflight_failure(
            value, before, status,
            "every partial within-state owner overlap is rejected before any write");
    }
}

template<class T>
static void exercise_owner_cross_state_preflight_matrix(
    G1StateArrayMember<T> member,
    int owner_index)
{
    char error[512] = {};
    for (int accepted_target = 0; accepted_target < 2; ++accepted_target) {
        fixture value;
        const RuntimeEvidence before =
            preflight_runtime_evidence(value.runtime);
        g1_controller_state& target = accepted_target != 0
            ? value.runtime.accepted_state
            : value.runtime.working_state;
        g1_controller_state& peer = accepted_target != 0
            ? value.runtime.working_state
            : value.runtime.accepted_state;
        array1d<T>& target_owner = target.*member;
        const array1d<T>& peer_owner = peer.*member;
        T* const saved = target_owner.data;
        target_owner.data = peer_owner.data;
        const G1FrameTransactionStatus status =
            run_storage_preflight_case(
                value, error, static_cast<int>(sizeof(error)));
        target_owner.data = saved;
        check_restored_storage_preflight_failure(
            value, before, status,
            "every accepted/working owner overlap is rejected before any write");
    }
    {
        fixture value;
        const RuntimeEvidence before =
            preflight_runtime_evidence(value.runtime);
        g1_controller_state& target = value.runtime.working_state;
        const g1_controller_state& peer = value.runtime.accepted_state;
        array1d<T>& target_owner = target.*member;
        const StateStorageIdentities target_identities =
            state_storage_identities(target);
        const StateStorageIdentities peer_identities =
            state_storage_identities(peer);
        const std::size_t target_bytes =
            static_cast<std::size_t>(target_owner.size) * sizeof(T);
        int peer_index = -1;
        for (int offset = 1; offset < 49; ++offset) {
            const int candidate_index = (owner_index + offset) % 49;
            const uintptr_t candidate_address =
                reinterpret_cast<uintptr_t>(
                    peer_identities.owners[candidate_index].data);
            if (peer_identities.owners[candidate_index].bytes >=
                    target_bytes &&
                candidate_address % alignof(T) == 0U) {
                peer_index = candidate_index;
                break;
            }
        }
        check(peer_index >= 0 && peer_index != owner_index &&
                  target_owner.data ==
                      target_identities.owners[owner_index].data &&
                  peer_identities.owners[peer_index].bytes >= target_bytes,
              "cross-state cross-owner cycle selects a different size-safe owner index");
        T* const saved = target_owner.data;
        const uintptr_t peer_address = reinterpret_cast<uintptr_t>(
            peer_identities.owners[peer_index].data);
        check(peer_address % alignof(T) == 0U,
              "cross-state cross-owner pointer preserves target alignment");
        target_owner.data = reinterpret_cast<T*>(peer_address);
        check(target_owner.data ==
                  peer_identities.owners[peer_index].data &&
                  g1_ik_memory_ranges_overlap(
                      target_owner.data,
                      target_bytes,
                      peer_identities.owners[peer_index].data,
                      peer_identities.owners[peer_index].bytes),
              "cross-state different-index mutation is a genuine full-extent-safe overlap");
        const G1FrameTransactionStatus status =
            run_storage_preflight_case(
                value, error, static_cast<int>(sizeof(error)));
        target_owner.data = saved;
        check_restored_storage_preflight_failure(
            value, before, status,
            "every working target rejects one size-safe cross-state different-owner overlap before semantic dereference");
    }
}

template<class T>
static void exercise_owner_error_overlap_preflight_matrix(
    G1StateArrayMember<T> member)
{
    for (int accepted_side = 0; accepted_side < 2; ++accepted_side) {
        fixture value;
        g1_controller_state& side = accepted_side != 0
            ? value.runtime.accepted_state
            : value.runtime.working_state;
        const array1d<T>& owner = side.*member;
        expect_preflight_global(
            value, test_runner, value.external,
            reinterpret_cast<char*>(owner.data),
            static_cast<int>(sizeof(T)),
            "error overlap with every accepted/working array owner is rejected");
    }
}

template<class T>
static void exercise_owner_preflight_matrix(
    G1StateArrayMember<T> member,
    int owner_index)
{
    exercise_owner_side_preflight_matrix(member, owner_index, false);
    exercise_owner_side_preflight_matrix(member, owner_index, true);
    exercise_owner_cross_state_preflight_matrix(member, owner_index);
    exercise_owner_error_overlap_preflight_matrix(member);
}

static void exercise_all_49_owner_preflight_matrices()
{
    int owner_index = 0;
#define EXERCISE_OWNER(name) \
    exercise_owner_preflight_matrix( \
        &g1_controller_state::name, owner_index++)
    EXERCISE_OWNER(curr_bone_positions);
    EXERCISE_OWNER(curr_bone_velocities);
    EXERCISE_OWNER(trns_bone_positions);
    EXERCISE_OWNER(trns_bone_velocities);
    EXERCISE_OWNER(curr_bone_rotations);
    EXERCISE_OWNER(trns_bone_rotations);
    EXERCISE_OWNER(curr_bone_angular_velocities);
    EXERCISE_OWNER(trns_bone_angular_velocities);
    EXERCISE_OWNER(curr_bone_contacts);
    EXERCISE_OWNER(trns_bone_contacts);
    EXERCISE_OWNER(bone_positions);
    EXERCISE_OWNER(bone_velocities);
    EXERCISE_OWNER(bone_angular_velocities);
    EXERCISE_OWNER(bone_rotations);
    EXERCISE_OWNER(bone_offset_positions);
    EXERCISE_OWNER(bone_offset_velocities);
    EXERCISE_OWNER(bone_offset_angular_velocities);
    EXERCISE_OWNER(bone_offset_rotations);
    EXERCISE_OWNER(adjusted_bone_positions);
    EXERCISE_OWNER(global_bone_positions);
    EXERCISE_OWNER(global_bone_velocities);
    EXERCISE_OWNER(adjusted_bone_rotations);
    EXERCISE_OWNER(global_bone_rotations);
    EXERCISE_OWNER(global_bone_angular_velocities);
    EXERCISE_OWNER(global_bone_computed);
    EXERCISE_OWNER(trajectory_desired_velocities);
    EXERCISE_OWNER(trajectory_positions);
    EXERCISE_OWNER(trajectory_velocities);
    EXERCISE_OWNER(trajectory_accelerations);
    EXERCISE_OWNER(trajectory_angular_velocities);
    EXERCISE_OWNER(trajectory_desired_rotations);
    EXERCISE_OWNER(trajectory_rotations);
    EXERCISE_OWNER(contact_bones);
    EXERCISE_OWNER(contact_states);
    EXERCISE_OWNER(contact_locks);
    EXERCISE_OWNER(contact_positions);
    EXERCISE_OWNER(contact_velocities);
    EXERCISE_OWNER(contact_points);
    EXERCISE_OWNER(contact_targets);
    EXERCISE_OWNER(contact_offset_positions);
    EXERCISE_OWNER(contact_offset_velocities);
    EXERCISE_OWNER(ik_bone_positions);
    EXERCISE_OWNER(ik_bone_rotations);
    EXERCISE_OWNER(ik_global_bone_positions);
    EXERCISE_OWNER(ik_global_bone_rotations);
    EXERCISE_OWNER(ik_candidate_bone_positions);
    EXERCISE_OWNER(ik_candidate_bone_rotations);
    EXERCISE_OWNER(ik_candidate_global_bone_positions);
    EXERCISE_OWNER(ik_candidate_global_bone_rotations);
#undef EXERCISE_OWNER
    check(owner_index == 49,
          "preflight mutation matrix covers exactly all 49 owning arrays");
}

static void invalidate_external_artifact_shape(fixture& value, int invalid)
{
    switch (invalid) {
    case 0: --value.db.bone_positions.cols; break;
    case 1: --value.db.bone_velocities.rows; break;
    case 2: --value.db.bone_rotations.cols; break;
    case 3: --value.db.bone_angular_velocities.rows; break;
    case 4: --value.db.contact_states.cols; break;
    case 5: --value.db.bone_parents.size; break;
    case 6: --value.db.range_stops(0); break;
    case 7: --value.db.features.rows; break;
    case 8: --value.db.features.cols; break;
    case 9: --value.db.features_offset.size; break;
    case 10: --value.db.features_scale.size; break;
    case 11: --value.db.terrain_features.rows; break;
    case 12: --value.db.terrain_features.cols; break;
    case 13: --value.support.values.rows; break;
    case 14: --value.support.values.cols; break;
    case 15: value.scene.terrain.version = 1; break;
    case 16: --value.scene.terrain.nx; break;
    case 17:
        value.scene.terrain.cell_size =
            std::numeric_limits<float>::quiet_NaN();
        break;
    case 18: --value.scene.terrain.heights.size; break;
    case 19: --value.scene.walkability.nx; break;
    case 20: --value.scene.walkability.cells.size; break;
    case 21:
        value.scene.metadata.spawn_position.x =
            std::numeric_limits<float>::quiet_NaN();
        break;
    default:
        value.scene.metadata.playable_bounds.max_x = 1.0f;
        break;
    }
}

static scene_route transaction_test_route(
    const char* id, float second_x, float second_z)
{
    scene_route route;
    route.id = id;
    route.expected_outcome = "pass";
    route.walkability_class = 1;
    route.waypoints_xz.push_back(std::make_pair(2.0f, 2.0f));
    route.waypoints_xz.push_back(std::make_pair(second_x, second_z));
    return route;
}

static void configure_route_fixture(fixture& value)
{
    value.scene.metadata.routes.clear();
    value.scene.metadata.routes.push_back(
        transaction_test_route("transaction-route-a", 3.0f, 2.0f));
    value.scene.metadata.routes.push_back(
        transaction_test_route("transaction-route-b", 2.0f, 3.0f));
    G1FrameResetConfig config;
    config.route_mode = true;
    config.route_id = value.scene.metadata.routes[0].id.c_str();
    config.initial_search_time = value.external.tuning.initial_search_time;
    config.ik_enabled = value.external.tuning.ik_enabled;
    char error[512] = {};
    check(g1_frame_runtime_reset(
              value.runtime, value.db, value.support, value.scene, config,
              error, static_cast<int>(sizeof(error))),
          error);
    value.runtime.working_state.camera_azimuth += 0.0625f;
    check(g1_controller_state_is_valid(value.runtime.working_state) &&
              state_logical_digest(value.runtime.working_state) !=
                  state_logical_digest(value.runtime.accepted_state),
          "route preflight reset preserves a valid distinct working destination");
    value.external.scene = &value.scene;
    value.external.route = &value.scene.metadata.routes[0];
    value.external.tuning.mode = G1_TestRoute;
    value.external.tuning.frame_limit = 100;
}

static void check_route_fixture_relation(
    const fixture& value,
    bool require_equal_route_frames,
    const char* message)
{
    check(value.external.scene == &value.scene &&
              value.external.route != NULL &&
              value.external.tuning.mode == G1_TestRoute &&
              value.external.tuning.frame_limit > 0,
          message);
    int member_index = -1;
    for (std::size_t index = 0;
         index < value.scene.metadata.routes.size();
         ++index) {
        if (value.external.route ==
            &value.scene.metadata.routes[index]) {
            member_index = static_cast<int>(index);
        }
    }
    check(member_index >= 0 &&
              !value.external.route->id.empty() &&
              value.external.route->waypoints_xz.size() >= 2U,
          message);
    const g1_controller_state* states[] = {
        &value.runtime.accepted_state,
        &value.runtime.working_state,
    };
    for (const g1_controller_state* state : states) {
        check(state->route_index == member_index &&
                  state->route_waypoint >= 1 &&
                  static_cast<std::size_t>(state->route_waypoint) <
                      value.external.route->waypoints_xz.size() &&
                  state->route_frames >= 0 &&
                  value.scene.metadata.routes[state->route_index].id ==
                      value.external.route->id,
              message);
    }
    check(value.runtime.accepted_state.route_index ==
              value.runtime.working_state.route_index &&
              value.runtime.accepted_state.route_waypoint ==
                  value.runtime.working_state.route_waypoint &&
              (!require_equal_route_frames ||
               value.runtime.accepted_state.route_frames ==
                   value.runtime.working_state.route_frames),
          message);
}

using G1TuningFloatMember = float G1FrameTuning::*;

static const G1TuningFloatMember all_tuning_float_members[] = {
    &G1FrameTuning::dt,
    &G1FrameTuning::trajectory_sample_time,
    &G1FrameTuning::route_speed,
    &G1FrameTuning::future_speed_scale,
    &G1FrameTuning::walkability_radius,
    &G1FrameTuning::effective_terrain_weight,
    &G1FrameTuning::initial_search_time,
    &G1FrameTuning::inertialize_blending_halflife,
    &G1FrameTuning::desired_velocity_change_threshold,
    &G1FrameTuning::desired_rotation_change_threshold,
    &G1FrameTuning::simulation_velocity_halflife,
    &G1FrameTuning::simulation_rotation_halflife,
    &G1FrameTuning::simulation_run_forward_speed,
    &G1FrameTuning::simulation_run_side_speed,
    &G1FrameTuning::simulation_run_back_speed,
    &G1FrameTuning::simulation_walk_forward_speed,
    &G1FrameTuning::simulation_walk_side_speed,
    &G1FrameTuning::simulation_walk_back_speed,
    &G1FrameTuning::synchronization_data_factor,
    &G1FrameTuning::adjustment_position_halflife,
    &G1FrameTuning::adjustment_rotation_halflife,
    &G1FrameTuning::adjustment_position_max_ratio,
    &G1FrameTuning::adjustment_rotation_max_ratio,
    &G1FrameTuning::clamping_max_distance,
    &G1FrameTuning::clamping_max_angle,
    &G1FrameTuning::contact_unlock_radius,
    &G1FrameTuning::contact_foot_height,
    &G1FrameTuning::contact_blending_halflife,
};

static void invalidate_input_snapshot(
    G1FrameExternalInputs& external, int invalid)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    switch (invalid) {
    case 0: external.input.move_stick.x = nan; break;
    case 1: external.input.move_stick.y = nan; break;
    case 2: external.input.move_stick.z = nan; break;
    case 3: external.input.look_stick.x = nan; break;
    case 4: external.input.look_stick.y = nan; break;
    case 5: external.input.look_stick.z = nan; break;
    case 6: external.input.gait_target = nan; break;
    case 7: external.input.camera_zoom_axis = nan; break;
    case 8: external.input.scripted_azimuth_delta = nan; break;
    case 9: external.input.presentation_frame = -1; break;
    case 10: external.input.move_stick.x = 2.0f; break;
    case 11: external.input.look_stick.z = -2.0f; break;
    case 12: external.input.gait_target = -0.01f; break;
    case 13: external.input.gait_target = 1.01f; break;
    case 14: external.heading_override.heading.w = nan; break;
    case 15: external.heading_override.heading.x = nan; break;
    case 16: external.heading_override.heading.y = nan; break;
    case 17: external.heading_override.heading.z = nan; break;
    case 18: external.heading_override.heading = quat(2.0f, 0.0f, 0.0f, 0.0f); break;
    case 19:
        external.heading_override.active = true;
        external.heading_override.heading.w = nan;
        break;
    case 20:
        external.heading_override.active = true;
        external.heading_override.heading.x = nan;
        break;
    case 21:
        external.heading_override.active = true;
        external.heading_override.heading.y = nan;
        break;
    case 22:
        external.heading_override.active = true;
        external.heading_override.heading.z = nan;
        break;
    default:
        external.heading_override.active = true;
        external.heading_override.heading = quat(2.0f, 0.0f, 0.0f, 0.0f);
        break;
    }
}

static void invalidate_tuning_boundary(
    G1FrameExternalInputs& external, int invalid)
{
    switch (invalid) {
    case 0: external.tuning.dt = 0.05f; break;
    case 1: external.tuning.trajectory_sample_time = 0.0f; break;
    case 2: external.tuning.route_speed = 0.0f; break;
    case 3: external.tuning.future_speed_scale = -0.01f; break;
    case 4: external.tuning.walkability_radius = -0.01f; break;
    case 5: external.tuning.effective_terrain_weight = -0.01f; break;
    case 6: external.tuning.effective_terrain_weight = 10.01f; break;
    case 7: external.tuning.initial_search_time = -0.01f; break;
    case 8: external.tuning.initial_search_time = 10.01f; break;
    case 9: external.tuning.inertialize_blending_halflife = 0.0f; break;
    case 10: external.tuning.desired_velocity_change_threshold = -0.01f; break;
    case 11: external.tuning.desired_rotation_change_threshold = -0.01f; break;
    case 12: external.tuning.simulation_velocity_halflife = 0.0f; break;
    case 13: external.tuning.simulation_rotation_halflife = 0.0f; break;
    case 14: external.tuning.simulation_run_forward_speed = -0.01f; break;
    case 15: external.tuning.simulation_run_side_speed = -0.01f; break;
    case 16: external.tuning.simulation_run_back_speed = -0.01f; break;
    case 17: external.tuning.simulation_walk_forward_speed = -0.01f; break;
    case 18: external.tuning.simulation_walk_side_speed = -0.01f; break;
    case 19: external.tuning.simulation_walk_back_speed = -0.01f; break;
    case 20: external.tuning.synchronization_data_factor = -0.01f; break;
    case 21: external.tuning.adjustment_position_halflife = 0.0f; break;
    case 22: external.tuning.adjustment_rotation_halflife = 0.0f; break;
    case 23: external.tuning.adjustment_position_max_ratio = -0.01f; break;
    case 24: external.tuning.adjustment_position_max_ratio = 1.01f; break;
    case 25: external.tuning.adjustment_rotation_max_ratio = -0.01f; break;
    case 26: external.tuning.adjustment_rotation_max_ratio = 1.01f; break;
    case 27: external.tuning.clamping_max_distance = -0.01f; break;
    case 28: external.tuning.clamping_max_angle = -0.01f; break;
    case 29: external.tuning.contact_unlock_radius = -0.01f; break;
    case 30: external.tuning.contact_foot_height = -0.01f; break;
    case 31: external.tuning.contact_blending_halflife = 0.0f; break;
    default: external.tuning.future_speed_scale = 1.01f; break;
    }
}

static void test_coordinator_preflight_rejects_invalid_inputs_and_aliases()
{
    char error[512] = {};
    {
        fixture value;
        expect_preflight_global(
            value, NULL, value.external,
            error, static_cast<int>(sizeof(error)),
            "null stage runner is rejected atomically");
    }
    {
        fixture value;
        value.runtime.accepted_state.search_time =
            std::numeric_limits<float>::quiet_NaN();
        expect_preflight_global(
            value, test_runner, value.external,
            error, static_cast<int>(sizeof(error)),
            "invalid accepted source semantics are rejected atomically");
    }
    {
        fixture value;
        value.runtime.publication.rejection.rejected = true;
        expect_preflight_global(
            value, test_runner, value.external,
            error, static_cast<int>(sizeof(error)),
            "invalid incoming publication is rejected atomically");
    }
    {
        fixture value;
        value.runtime.accepted_diagnostic.ready = true;
        value.runtime.accepted_diagnostic.presentation_frame = -1;
        expect_preflight_global(
            value, test_runner, value.external,
            error, static_cast<int>(sizeof(error)),
            "invalid accepted diagnostic is rejected atomically");
    }

    for (int invalid = 0; invalid < 3; ++invalid) {
        fixture value;
        G1FrameExternalInputs external = value.external;
        switch (invalid) {
        case 0: external.db = NULL; break;
        case 1: external.support = NULL; break;
        default: external.scene = NULL; break;
        }
        expect_preflight_global(
            value, test_runner, external,
            error, static_cast<int>(sizeof(error)),
            "every null immutable artifact pointer is rejected atomically");
    }

    for (int invalid = 0; invalid < 23; ++invalid) {
        fixture value;
        invalidate_external_artifact_shape(value, invalid);
        expect_preflight_global(
            value, test_runner, value.external,
            error, static_cast<int>(sizeof(error)),
            "every malformed database/support/scene shape is rejected atomically");
    }

    for (int invalid = 0; invalid < 24; ++invalid) {
        fixture value;
        G1FrameExternalInputs external = value.external;
        invalidate_input_snapshot(external, invalid);
        expect_preflight_global(
            value, test_runner, external,
            error, static_cast<int>(sizeof(error)),
            "every invalid sampled input and heading value is rejected atomically");
    }

    for (const G1TuningFloatMember member : all_tuning_float_members) {
        fixture value;
        G1FrameExternalInputs external = value.external;
        external.tuning.*member =
            std::numeric_limits<float>::quiet_NaN();
        expect_preflight_global(
            value, test_runner, external,
            error, static_cast<int>(sizeof(error)),
            "every nonfinite tuning scalar is rejected atomically");
    }

    for (int invalid = 0; invalid < 33; ++invalid) {
        fixture value;
        G1FrameExternalInputs external = value.external;
        invalidate_tuning_boundary(external, invalid);
        expect_preflight_global(
            value, test_runner, external,
            error, static_cast<int>(sizeof(error)),
            "every tuning range and exact-25-Hz boundary is rejected atomically");
    }

    for (int invalid = 0; invalid < 5; ++invalid) {
        fixture value;
        G1FrameExternalInputs external = value.external;
        switch (invalid) {
        case 0:
            external.tuning.frame_limit = -1;
            break;
        case 1:
            external.tuning.mode = G1_TestSequential;
            external.tuning.frame_limit = 0;
            break;
        case 2:
            external.tuning.mode = G1_TestSceneCycle;
            external.tuning.frame_limit = 14;
            external.tuning.scene_dwell_frames = 0;
            break;
        case 3:
            external.tuning.mode = G1_TestSceneCycle;
            external.tuning.scene_dwell_frames = 7;
            external.tuning.frame_limit = 14 * 7 - 1;
            break;
        default:
            external.tuning.mode = G1_TestSceneCycle;
            external.tuning.scene_dwell_frames =
                std::numeric_limits<int>::max() / 14 + 1;
            external.tuning.frame_limit =
                std::numeric_limits<int>::max();
            break;
        }
        expect_preflight_global(
            value, test_runner, external,
            error, static_cast<int>(sizeof(error)),
            "invalid frame constraints are rejected atomically");
    }

    {
        fixture value;
        G1FrameExternalInputs external = value.external;
        external.tuning.mode = G1_TestSceneCycle;
        external.tuning.scene_dwell_frames = 7;
        external.tuning.frame_limit = 14 * 7;
        G1FrameTransactionTestSeam seam;
        seam.hook = injection_hook;
        char boundary_error[512] = {};
        reset_trace();
        check(g1_frame_transaction_run(
                  value.runtime, test_runner, external, &seam,
                  boundary_error,
                  static_cast<int>(sizeof(boundary_error))) ==
                  G1FrameTransactionAccepted,
              "scene-cycle exact 14*dwell frame boundary is accepted");
        check_stage_trace(
            G1FrameStageCount - 1, G1FrameStageCount - 1,
            "valid scene-cycle boundary reaches every ordered stage");
    }

    {
        fixture value;
        G1FrameExternalInputs external = value.external;
        external.tuning.initial_search_time = 0.50f;
        expect_preflight_global(
            value, test_runner, external,
            error, static_cast<int>(sizeof(error)),
            "finite search-time bits must agree with accepted immutable search time");
    }

    {
        fixture value;
        configure_route_fixture(value);
        check_route_fixture_relation(
            value, true,
            "fresh route fixture has exact id, membership, index, and cursor agreement in both states");
        check(value.runtime.accepted_state.route_index == 0 &&
                  value.runtime.accepted_state.route_waypoint == 1 &&
                  value.runtime.accepted_state.route_frames == 0 &&
                  value.runtime.working_state.route_index == 0 &&
                  value.runtime.working_state.route_waypoint == 1 &&
                  value.runtime.working_state.route_frames == 0 &&
                  value.scene.metadata.routes[0].id ==
                      value.external.route->id,
              "fresh route reset installs exact (index, waypoint, frames) = (0, 1, 0) in both states");
        const int route_index_before =
            value.runtime.accepted_state.route_index;
        const int route_waypoint_before =
            value.runtime.accepted_state.route_waypoint;
        const int route_frames_before =
            value.runtime.accepted_state.route_frames;
        G1FrameTransactionTestSeam seam;
        seam.hook = injection_hook;
        char route_error[512] = {};
        reset_trace();
        check(g1_frame_transaction_run(
                  value.runtime, test_runner, value.external, &seam,
                  route_error, static_cast<int>(sizeof(route_error))) ==
                  G1FrameTransactionAccepted,
              "unmodified valid route fixture is accepted");
        check_stage_trace(
            G1FrameStageCount - 1, G1FrameStageCount - 1,
            "valid route baseline reaches all 12 stages exactly once");
        check(g1_controller_state_is_valid(
                  value.runtime.accepted_state) &&
                  g1_controller_state_is_valid(
                      value.runtime.working_state),
              "valid route success preserves two certified state values");
        check_route_fixture_relation(
            value, false,
            "accepted route success preserves id, membership, index, and waypoint relation");
        check(value.runtime.accepted_state.route_index ==
                  route_index_before &&
                  value.runtime.accepted_state.route_waypoint ==
                      route_waypoint_before &&
                  value.runtime.accepted_state.route_frames ==
                      route_frames_before + 3 &&
                  value.runtime.working_state.route_index ==
                      route_index_before &&
                  value.runtime.working_state.route_waypoint ==
                      route_waypoint_before &&
                  value.runtime.working_state.route_frames ==
                      route_frames_before,
              "route success advances only the published route frame cursor and leaves the prior accepted cursor in working");
    }

    for (int invalid = 0; invalid < 7; ++invalid) {
        fixture value;
        scene_route unrelated = transaction_test_route(
            "transaction-route-a", 3.0f, 2.0f);
        configure_route_fixture(value);
        G1FrameExternalInputs external = value.external;
        check(external.tuning.mode == G1_TestRoute &&
                  external.tuning.frame_limit > 0 &&
                  external.route == &value.scene.metadata.routes[0] &&
                  external.route->waypoints_xz.size() >= 2U,
              "each route mutation starts from one complete valid route-mode baseline");
        if (invalid == 0) {
            external.route = NULL;
        } else if (invalid == 1) {
            external.tuning.mode = G1_TestLive;
        } else if (invalid == 2) {
            external.route = &unrelated;
        } else if (invalid == 3) {
            external.route = &value.scene.metadata.routes[1];
        } else if (invalid == 4) {
            value.scene.metadata.routes[0].id.clear();
        } else if (invalid == 5) {
            value.runtime.accepted_state.route_index = 1;
        } else {
            value.scene.metadata.routes[0].waypoints_xz.pop_back();
        }
        expect_preflight_global(
            value, test_runner, external,
            error, static_cast<int>(sizeof(error)),
            "route mode, membership, id, index, and shape must agree atomically");
    }

    exercise_all_49_owner_preflight_matrices();

    for (int overlap = 0; overlap < 4; ++overlap) {
        fixture value;
        char* overlapping_error = NULL;
        int overlapping_capacity = 0;
        switch (overlap) {
        case 0:
            overlapping_error = reinterpret_cast<char*>(
                &value.runtime.accepted_state);
            overlapping_capacity =
                static_cast<int>(sizeof(value.runtime.accepted_state));
            break;
        case 1:
            overlapping_error = reinterpret_cast<char*>(
                &value.runtime.working_state);
            overlapping_capacity =
                static_cast<int>(sizeof(value.runtime.working_state));
            break;
        case 2:
            overlapping_error = reinterpret_cast<char*>(
                &value.runtime.publication);
            overlapping_capacity =
                static_cast<int>(sizeof(value.runtime.publication));
            break;
        default:
            overlapping_error = reinterpret_cast<char*>(
                &value.runtime.accepted_diagnostic);
            overlapping_capacity =
                static_cast<int>(sizeof(value.runtime.accepted_diagnostic));
            break;
        }
        expect_preflight_global(
            value, test_runner, value.external,
            overlapping_error, overlapping_capacity,
            "error overlap with any transaction-owned region is rejected");
    }
}

static void test_safe_stop_latch_lifecycle()
{
    fixture value;
    value.runtime.accepted_state.simulation_velocity =
        vec3(0.75f, -0.50f, -0.25f);
    value.runtime.accepted_state.simulation_acceleration =
        vec3(-0.125f, 0.625f, 0.375f);
    value.runtime.accepted_state.route_frames = 9;
    check(g1_controller_state_is_valid(value.runtime.accepted_state),
          "latch fixture accepted state is valid");
    const uint32_t velocity_y = terrain_float_bits(
        value.runtime.accepted_state.simulation_velocity.y);
    const uint32_t acceleration_y = terrain_float_bits(
        value.runtime.accepted_state.simulation_acceleration.y);
    const uint64_t accepted_before =
        state_logical_digest(value.runtime.accepted_state);
    const RuntimeEvidence before_first_finite =
        runtime_evidence(value.runtime);

    G1FrameTransactionTestSeam seam;
    seam.hook = injection_hook;
    seam.control.injected_stage = G1FrameStageMatcherSearch;
    seam.control.injected_outcome = G1FrameInjectFiniteReject;
    char error[512] = {};
    reset_trace();
    check(g1_frame_transaction_run(
              value.runtime, test_runner, value.external, &seam,
              error, static_cast<int>(sizeof(error))) ==
              G1FrameTransactionFiniteRejected &&
          value.runtime.publication.ik_safe_stop_latched &&
          state_logical_digest(value.runtime.accepted_state) ==
              accepted_before,
          "finite rejection sets the next-frame latch without accepting motion");
    check_finite_preservation(
        value.runtime, before_first_finite,
        "latching finite rejection preserves accepted owners and both storage partitions");

    const RuntimeEvidence latched_before = runtime_evidence(value.runtime);
    reset_trace();
    check(g1_frame_transaction_run(
              value.runtime, test_runner, value.external, &seam,
              error, static_cast<int>(sizeof(error))) ==
              G1FrameTransactionFiniteRejected &&
          trace.prior_latch_seen &&
          value.runtime.publication.ik_safe_stop_latched,
          "another finite retry receives and relatches the incoming latch");
    check(same_published_runtime_evidence(
              runtime_evidence(value.runtime), latched_before),
          "second finite retry preserves the complete latched publication unit and both storage partitions");

    seam.control.injected_outcome = G1FrameInjectGlobalError;
    const RuntimeEvidence before_global = runtime_evidence(value.runtime);
    reset_trace();
    check(g1_frame_transaction_run(
              value.runtime, test_runner, value.external, &seam,
              error, static_cast<int>(sizeof(error))) ==
              G1FrameTransactionGlobalError,
          "latched global retry returns controlled status");
    check_global_preservation(
        value.runtime, before_global,
        "global retry preserves incoming latch and every accepted owner");

    seam.control.injected_stage = G1FrameStageCount;
    seam.control.injected_outcome = G1FrameInjectContinue;
    reset_trace();
    check(g1_frame_transaction_run(
              value.runtime, test_runner, value.external, &seam,
              error, static_cast<int>(sizeof(error))) ==
              G1FrameTransactionAccepted,
          error);
    check(trace.prior_latch_seen &&
              trace.latched_handoff_velocity_exact &&
              trace.latched_handoff_cancel &&
              trace.latched_handoff_force_search &&
              !value.runtime.publication.ik_safe_stop_latched &&
              terrain_float_bits(
                  value.runtime.accepted_state.simulation_velocity.x) == 0U &&
              terrain_float_bits(
                  value.runtime.accepted_state.simulation_velocity.z) == 0U &&
              terrain_float_bits(
                  value.runtime.accepted_state.simulation_acceleration.x) == 0U &&
              terrain_float_bits(
                  value.runtime.accepted_state.simulation_acceleration.z) == 0U &&
              terrain_float_bits(
                  value.runtime.accepted_state.simulation_velocity.y) ==
                  velocity_y &&
              terrain_float_bits(
                  value.runtime.accepted_state.simulation_acceleration.y) ==
                  acceleration_y &&
              value.runtime.accepted_state.route_frames == 9 &&
              same_intent_bits(
                  value.runtime.publication.requested_intent,
                  expected_requested_intent(value)),
          "accepted retry receives the exact safe-stop handoff, consumes the "
          "latch, preserves Y and the complete heading, and freezes route");
    check(latched_before.publication !=
              publication_logical_digest(value.runtime.publication),
          "accepted retry replaces the finite publication atomically");

    const int frozen_route_frames =
        value.runtime.accepted_state.route_frames;
    const int consumed_scene_frame =
        value.runtime.accepted_state.scene_frame;
    reset_trace();
    check(g1_frame_transaction_run(
              value.runtime, test_runner, value.external, &seam,
              error, static_cast<int>(sizeof(error))) ==
              G1FrameTransactionAccepted,
          error);
    check(!trace.prior_latch_seen &&
              !value.runtime.publication.ik_safe_stop_latched &&
              value.runtime.accepted_state.route_frames ==
                  frozen_route_frames + 3 &&
              value.runtime.accepted_state.scene_frame ==
                  consumed_scene_frame + 1,
          "the next unlatched accepted frame resumes route advancement, so the stop lasts exactly one accepted frame");
    check_stage_trace(
        G1FrameStageCount - 1, G1FrameStageCount - 1,
        "post-latch frame resumes the complete ordered transaction exactly once");
}

int main()
{
    test_orientation_authenticates_desired_sole_normal();
    test_contact_iteration_provenance_is_authenticated();
    test_ready_landing_is_validation_only_lookahead();
    test_target_and_ik_state_normals_are_transaction_owned();
    test_runtime_reset_publishes_mode_specific_route_cursor();
    test_seam_injection_at_every_stage_is_atomic_and_ordered();
    test_runner_outcomes_precede_hook_and_stop_later_stages();
    test_hostile_runner_outputs_are_rejected_before_publication();
    test_input_checkpoint_publishes_ready_intent_only();
    test_success_iteration_provenance_is_checked_before_publication();
    test_success_swaps_every_owner_and_publishes_complete_diagnostic();
    test_dirty_working_storage_is_overwritten_completely();
    test_coordinator_preflight_rejects_invalid_inputs_and_aliases();
    test_safe_stop_latch_lifecycle();
    return 0;
}
