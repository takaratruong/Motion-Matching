#include "g1_controller_frame_runtime.h"

#include <algorithm>
#include <cmath>
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
static_assert(G1FrameStageRawBegin ==
                  G1FrameStageFootprintObservation + 1,
              "raw certification follows the common candidate phase");
static_assert(G1FrameStageIkBegin ==
                  G1FrameStageRawPoseCertificate + 1,
              "IK certification follows raw certification");
static_assert(G1FrameStageAcceptedFinalize + 1 == G1FrameStageCount,
              "accepted finalization is the final transaction stage");
static_assert(std::is_same<G1FrameStageRunner,
    G1FrameStageOutcome (*)(
        G1FrameTransactionStage,
        g1_controller_state&,
        G1FrameTransactionScratch&,
        const G1FrameExternalInputs&,
        char*, int)>::value,
    "the runner has only working state, scratch, and immutable external input");
static_assert(std::is_same<G1RecoveryProvider,
    G1RecoveryProviderStatus (*)(
        G1RecoveryCandidateSet&,
        const G1RecoveryRequest&,
        char*, int)>::value,
    "the real coordinator consumes the strict recovery provider unchanged");

using G1FrameCoordinator = G1FrameTransactionStatus (*)(
    G1FrameRuntime&,
    G1FrameStageRunner,
    G1RecoveryProvider,
    const G1FrameExternalInputs&,
    const G1FrameTransactionTestSeam*,
    char*,
    int);

static G1FrameTransactionStatus g1_frame_transaction_run(
    G1FrameRuntime& runtime,
    G1FrameStageRunner runner,
    const G1FrameExternalInputs& external,
    const G1FrameTransactionTestSeam* seam,
    char* error,
    int error_capacity)
{
    const G1FrameCoordinator coordinator =
        static_cast<G1FrameCoordinator>(&::g1_frame_transaction_run);
    return coordinator(
        runtime,
        runner,
        ::g1_recovery_candidates_build,
        external,
        seam,
        error,
        error_capacity);
}

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
        check(g1_controller_state_is_valid(runtime.accepted_state) &&
                  g1_controller_state_is_valid(runtime.working_state),
              "nonzero tail fixture remains a certified state pair");
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

static uint64_t footprint_logical_digest(
    const G1FootprintObservation& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_footprint(hash, value);
    return hash;
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

static void test_production_root_reach_digest_ownership()
{
    G1IkFrameResult baseline = {};
    baseline.root_reach.active = true;
    baseline.root_reach.common_interval_found = true;
    baseline.root_reach.applied = true;
    baseline.root_reach.root_y_delta_m = -0.01f;
    check(g1_root_reach_plan_is_valid(baseline.root_reach),
          "production digest ownership starts from a valid root plan");
    const uint64_t digest = ik_frame_logical_digest(baseline);
    const auto require_owned = [&baseline, digest](
        const G1IkFrameResult& mutated,
        const char* message) {
        check(!g1_frame_ik_result_equal(baseline, mutated) &&
                  ik_frame_logical_digest(mutated) != digest,
              message);
    };
    G1IkFrameResult mutated = baseline;
    mutated.root_reach.active = false;
    require_owned(mutated, "production digest owns root-plan active");
    mutated = baseline;
    mutated.root_reach.common_interval_found = false;
    require_owned(mutated, "production digest owns root-plan common interval");
    mutated = baseline;
    mutated.root_reach.applied = false;
    require_owned(mutated, "production digest owns root-plan applied");
    mutated = baseline;
    mutated.root_reach.root_y_delta_m = std::nextafter(
        baseline.root_reach.root_y_delta_m,
        -std::numeric_limits<float>::infinity());
    require_owned(mutated, "production digest owns exact root-plan delta bits");
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

struct StateStorageIdentity
{
    const void* data = NULL;
    int size = 0;
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
    const auto add = [&output, &count](const void* data, int size) {
        check(count < 49, "state storage identity capacity is exact");
        output.owners[count].data = data;
        output.owners[count].size = size;
        ++count;
    };
#define ADD_ARRAY(name) add(state.name.data, state.name.size)
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
            first.owners[index].size != second.owners[index].size) {
            return false;
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

static uint64_t rejection_logical_digest(
    const G1FrameRejectionDiagnostic& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_rejection(hash, value);
    return hash;
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

#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

static_assert(std::is_same<decltype(&g1_controller_frame_stage_run),
                           G1FrameStageRunner>::value,
              "the controller exports the exact typed production runner");

// Mandatory GREEN provenance remains an external build gate: compile
// controller.cpp with G1_CONTROLLER_NO_MAIN under strict warnings, link this
// fixture against that object, then use nm -C to prove exactly one exported
// g1_controller_frame_stage_run and no transaction-test seam in production.

static bool same_vec3_bits(vec3 first, vec3 second)
{
    return terrain_float_bits(first.x) == terrain_float_bits(second.x) &&
           terrain_float_bits(first.y) == terrain_float_bits(second.y) &&
           terrain_float_bits(first.z) == terrain_float_bits(second.z);
}

static bool same_quat_bits(quat first, quat second)
{
    return terrain_float_bits(first.w) == terrain_float_bits(second.w) &&
           terrain_float_bits(first.x) == terrain_float_bits(second.x) &&
           terrain_float_bits(first.y) == terrain_float_bits(second.y) &&
           terrain_float_bits(first.z) == terrain_float_bits(second.z);
}

static bool same_float_bits(float first, float second)
{
    return terrain_float_bits(first) == terrain_float_bits(second);
}

static float production_float_from_bits(uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static void test_fixed_nonassociative_root_reach_fk_ownership()
{
    const float a = production_float_from_bits(UINT32_C(0x3f000000));
    const float b = production_float_from_bits(UINT32_C(0x3f000001));
    const float d = production_float_from_bits(UINT32_C(0xbc80e8f0));
    database db;
    make_database(db, 1);
    array1d<vec3> support_local = db.bone_positions(0);
    array1d<quat> local_rotations = db.bone_rotations(0);
    support_local(G1_Simulation) = vec3(0.0f, a, 0.0f);
    support_local(G1_Hips) = vec3(0.0f, b, 0.0f);
    local_rotations(G1_Simulation) = quat();
    local_rotations(G1_Hips) = quat();

    const G1RootReachPlan plan = {true, true, true, d};
    check(g1_root_reach_plan_is_valid(plan),
          "the fixed multilevel fixture uses the certified Task-2 plan form");
    array1d<vec3> accepted_local = support_local;
    float expected_root_y = 0.0f;
    check(g1_apply_root_reach_plan_y(
              expected_root_y,
              support_local(G1_Simulation).y,
              plan),
          "the fixed multilevel fixture applies root Y only through the strict helper");
    accepted_local(G1_Simulation).y = expected_root_y;

    array1d<vec3> support_global(G1_BoneCount);
    array1d<quat> support_global_rotations(G1_BoneCount);
    array1d<vec3> accepted_global(G1_BoneCount);
    array1d<quat> accepted_global_rotations(G1_BoneCount);
    char error[512] = {};
    check(g1_ik_checked_forward_kinematics(
              support_global,
              support_global_rotations,
              support_local,
              local_rotations,
              db.bone_parents,
              error,
              static_cast<int>(sizeof(error))) &&
              g1_ik_checked_forward_kinematics(
                  accepted_global,
                  accepted_global_rotations,
                  accepted_local,
                  local_rotations,
                  db.bone_parents,
                  error,
                  static_cast<int>(sizeof(error))),
          error);

    float reassociated_hips_y = 0.0f;
    check(terrain_f32_add(
              reassociated_hips_y,
              support_global(G1_Hips).y,
              plan.root_y_delta_m),
          "the fixed multilevel fixture materializes the forbidden reassociation for comparison");
    G1FrameAcceptedDiagnostic diagnostic = {};
    diagnostic.support_retargeted.hips_y =
        support_global(G1_Hips).y;
    diagnostic.rendered.hips_y = accepted_global(G1_Hips).y;
    check(terrain_float_bits(
              accepted_local(G1_Simulation).y) ==
              terrain_float_bits(expected_root_y) &&
              terrain_float_bits(
                  diagnostic.support_retargeted.hips_y) ==
                  terrain_float_bits(support_global(G1_Hips).y) &&
              terrain_float_bits(
                  diagnostic.rendered.hips_y) ==
                  terrain_float_bits(accepted_global(G1_Hips).y) &&
              terrain_float_bits(accepted_global(G1_Hips).y) ==
                  UINT32_C(0x3f7bf8ba) &&
              terrain_float_bits(reassociated_hips_y) ==
                  UINT32_C(0x3f7bf8b8) &&
              terrain_float_bits(accepted_global(G1_Hips).y) !=
                  terrain_float_bits(reassociated_hips_y),
          "authoritative checked FK owns 0x3f7bf8ba while reassociated global-Hips plus delta is the distinct 0x3f7bf8b8");

    g1_controller_state working_state;
    working_state.scene_frame = 1;
    working_state.global_bone_positions.resize(G1_BoneCount);
    working_state.ik_global_bone_positions.resize(G1_BoneCount);
    working_state.global_bone_positions.set(vec3());
    working_state.ik_global_bone_positions.set(vec3());
    working_state.global_bone_positions(G1_Hips) =
        support_global(G1_Hips);
    working_state.ik_global_bone_positions(G1_Hips) =
        accepted_global(G1_Hips);
    G1FrameExternalInputs external = {};
    external.db = &db;
    external.tuning.mode = G1_TestSequential;
    diagnostic.scene_frame = 0;
    diagnostic.query_database_frame = 0;
    diagnostic.query_range = 0;
    diagnostic.selected_database_frame = 0;
    diagnostic.matching_enabled = false;
    check(g1_frame_accepted_diagnostic_matches_success(
              diagnostic, working_state, external),
          "the fixed multilevel diagnostic passes the production success matcher through both authoritative FK owners");
    const G1FrameAcceptedDiagnostic exact_diagnostic = diagnostic;
    diagnostic.support_retargeted.hips_y = std::nextafter(
        exact_diagnostic.support_retargeted.hips_y,
        std::numeric_limits<float>::infinity());
    check(!g1_frame_accepted_diagnostic_matches_success(
              diagnostic, working_state, external),
          "the production success matcher rejects a one-ULP support Hips diagnostic mutation");
    diagnostic = exact_diagnostic;
    diagnostic.rendered.hips_y = std::nextafter(
        exact_diagnostic.rendered.hips_y,
        std::numeric_limits<float>::infinity());
    check(!g1_frame_accepted_diagnostic_matches_success(
              diagnostic, working_state, external),
          "the production success matcher rejects a one-ULP rendered Hips diagnostic mutation");
    diagnostic = exact_diagnostic;
    working_state.global_bone_positions(G1_Hips).y = std::nextafter(
        support_global(G1_Hips).y,
        std::numeric_limits<float>::infinity());
    check(!g1_frame_accepted_diagnostic_matches_success(
              diagnostic, working_state, external),
          "the production success matcher rejects a one-ULP authoritative support FK mutation");
    working_state.global_bone_positions(G1_Hips) =
        support_global(G1_Hips);
    working_state.ik_global_bone_positions(G1_Hips).y =
        std::nextafter(
            accepted_global(G1_Hips).y,
            std::numeric_limits<float>::infinity());
    check(!g1_frame_accepted_diagnostic_matches_success(
              diagnostic, working_state, external),
          "the production success matcher rejects a one-ULP authoritative rendered FK mutation");
}

static bool same_external_lowering(
    const G1FrameExternalInputs& actual,
    const G1FrameExternalInputs& expected)
{
    return actual.db == expected.db &&
           actual.support == expected.support &&
           actual.scene == expected.scene &&
           actual.route == expected.route &&
           actual.heading_override.active == expected.heading_override.active &&
           same_quat_bits(
               actual.heading_override.heading,
               expected.heading_override.heading) &&
           same_vec3_bits(actual.input.move_stick, expected.input.move_stick) &&
           same_vec3_bits(actual.input.look_stick, expected.input.look_stick) &&
           same_float_bits(
               actual.input.gait_target, expected.input.gait_target) &&
           same_float_bits(
               actual.input.camera_zoom_axis,
               expected.input.camera_zoom_axis) &&
           same_float_bits(
               actual.input.scripted_azimuth_delta,
               expected.input.scripted_azimuth_delta) &&
           actual.input.desired_strafe == expected.input.desired_strafe &&
           actual.input.presentation_frame ==
               expected.input.presentation_frame &&
           actual.tuning.mode == expected.tuning.mode &&
           actual.tuning.frame_limit == expected.tuning.frame_limit &&
           actual.tuning.scene_dwell_frames ==
               expected.tuning.scene_dwell_frames &&
           same_float_bits(actual.tuning.dt, expected.tuning.dt) &&
           same_float_bits(
               actual.tuning.trajectory_sample_time,
               expected.tuning.trajectory_sample_time) &&
           same_float_bits(
               actual.tuning.route_speed, expected.tuning.route_speed) &&
           same_float_bits(
               actual.tuning.future_speed_scale,
               expected.tuning.future_speed_scale) &&
           same_float_bits(
               actual.tuning.walkability_radius,
               expected.tuning.walkability_radius) &&
           same_float_bits(
               actual.tuning.effective_terrain_weight,
               expected.tuning.effective_terrain_weight) &&
           same_float_bits(
               actual.tuning.initial_search_time,
               expected.tuning.initial_search_time) &&
           same_float_bits(
               actual.tuning.inertialize_blending_halflife,
               expected.tuning.inertialize_blending_halflife) &&
           same_float_bits(
               actual.tuning.desired_velocity_change_threshold,
               expected.tuning.desired_velocity_change_threshold) &&
           same_float_bits(
               actual.tuning.desired_rotation_change_threshold,
               expected.tuning.desired_rotation_change_threshold) &&
           same_float_bits(
               actual.tuning.simulation_velocity_halflife,
               expected.tuning.simulation_velocity_halflife) &&
           same_float_bits(
               actual.tuning.simulation_rotation_halflife,
               expected.tuning.simulation_rotation_halflife) &&
           same_float_bits(
               actual.tuning.simulation_run_forward_speed,
               expected.tuning.simulation_run_forward_speed) &&
           same_float_bits(
               actual.tuning.simulation_run_side_speed,
               expected.tuning.simulation_run_side_speed) &&
           same_float_bits(
               actual.tuning.simulation_run_back_speed,
               expected.tuning.simulation_run_back_speed) &&
           same_float_bits(
               actual.tuning.simulation_walk_forward_speed,
               expected.tuning.simulation_walk_forward_speed) &&
           same_float_bits(
               actual.tuning.simulation_walk_side_speed,
               expected.tuning.simulation_walk_side_speed) &&
           same_float_bits(
               actual.tuning.simulation_walk_back_speed,
               expected.tuning.simulation_walk_back_speed) &&
           actual.tuning.synchronization_enabled ==
               expected.tuning.synchronization_enabled &&
           same_float_bits(
               actual.tuning.synchronization_data_factor,
               expected.tuning.synchronization_data_factor) &&
           actual.tuning.adjustment_enabled ==
               expected.tuning.adjustment_enabled &&
           actual.tuning.adjustment_by_velocity_enabled ==
               expected.tuning.adjustment_by_velocity_enabled &&
           same_float_bits(
               actual.tuning.adjustment_position_halflife,
               expected.tuning.adjustment_position_halflife) &&
           same_float_bits(
               actual.tuning.adjustment_rotation_halflife,
               expected.tuning.adjustment_rotation_halflife) &&
           same_float_bits(
               actual.tuning.adjustment_position_max_ratio,
               expected.tuning.adjustment_position_max_ratio) &&
           same_float_bits(
               actual.tuning.adjustment_rotation_max_ratio,
               expected.tuning.adjustment_rotation_max_ratio) &&
           actual.tuning.clamping_enabled ==
               expected.tuning.clamping_enabled &&
           same_float_bits(
               actual.tuning.clamping_max_distance,
               expected.tuning.clamping_max_distance) &&
           same_float_bits(
               actual.tuning.clamping_max_angle,
               expected.tuning.clamping_max_angle) &&
           same_float_bits(
               actual.tuning.contact_unlock_radius,
               expected.tuning.contact_unlock_radius) &&
           same_float_bits(
               actual.tuning.contact_foot_height,
               expected.tuning.contact_foot_height) &&
           same_float_bits(
               actual.tuning.contact_blending_halflife,
               expected.tuning.contact_blending_halflife) &&
           actual.tuning.ik_enabled == expected.tuning.ik_enabled;
}

static void logical_hash_string(uint64_t& hash, const std::string& value)
{
    logical_hash_word(hash, static_cast<uint64_t>(value.size()));
    for (const char character : value) {
        logical_hash_word(
            hash, static_cast<unsigned char>(character));
    }
}

template<class T>
static void logical_hash_array2d(uint64_t& hash, const array2d<T>& values)
{
    logical_hash_value(hash, values.rows);
    logical_hash_value(hash, values.cols);
    for (int row = 0; row < values.rows; ++row) {
        for (int column = 0; column < values.cols; ++column) {
            logical_hash_value(hash, values(row, column));
        }
    }
}

static uint64_t database_logical_digest(const database& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_array2d(hash, value.bone_positions);
    logical_hash_array2d(hash, value.bone_velocities);
    logical_hash_array2d(hash, value.bone_rotations);
    logical_hash_array2d(hash, value.bone_angular_velocities);
    logical_hash_array(hash, value.bone_parents);
    logical_hash_array(hash, value.range_starts);
    logical_hash_array(hash, value.range_stops);
    logical_hash_array2d(hash, value.features);
    logical_hash_array(hash, value.features_offset);
    logical_hash_array(hash, value.features_scale);
    logical_hash_array2d(hash, value.terrain_features);
    logical_hash_array2d(hash, value.contact_states);
    logical_hash_array2d(hash, value.bound_sm_min);
    logical_hash_array2d(hash, value.bound_sm_max);
    logical_hash_array2d(hash, value.bound_lr_min);
    logical_hash_array2d(hash, value.bound_lr_max);
    return hash;
}

static uint64_t support_logical_digest(const terrain_support_set& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_array2d(hash, value.values);
    return hash;
}

static void logical_hash_bounds2(uint64_t& hash, const bounds2& value)
{
    logical_hash_value(hash, value.min_x);
    logical_hash_value(hash, value.min_z);
    logical_hash_value(hash, value.max_x);
    logical_hash_value(hash, value.max_z);
}

static void logical_hash_point3d(uint64_t& hash, const point3d& value)
{
    logical_hash_value(hash, value.x);
    logical_hash_value(hash, value.y);
    logical_hash_value(hash, value.z);
}

static void logical_hash_bounds3d(uint64_t& hash, const bounds3d& value)
{
    logical_hash_point3d(hash, value.minimum);
    logical_hash_point3d(hash, value.maximum);
}

static void logical_hash_artifact(
    uint64_t& hash, const artifact_reference& value)
{
    logical_hash_string(hash, value.path);
    logical_hash_string(hash, value.schema);
    logical_hash_string(hash, value.sha256);
    logical_hash_value(hash, value.version);
    logical_hash_value(hash, value.dimensions);
    logical_hash_word(hash, static_cast<uint64_t>(value.columns.size()));
    for (const std::string& column : value.columns) {
        logical_hash_string(hash, column);
    }
}

static void logical_hash_scene_route(
    uint64_t& hash, const scene_route& value)
{
    logical_hash_string(hash, value.id);
    logical_hash_string(hash, value.expected_outcome);
    logical_hash_value(hash, value.walkability_class);
    logical_hash_value(hash, value.landing_hold_seconds);
    logical_hash_word(
        hash, static_cast<uint64_t>(value.waypoints_xz.size()));
    for (const std::pair<float, float>& waypoint : value.waypoints_xz) {
        logical_hash_value(hash, waypoint.first);
        logical_hash_value(hash, waypoint.second);
    }
}

static uint64_t route_logical_digest(const scene_route* value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_value(hash, value != NULL);
    if (value != NULL) logical_hash_scene_route(hash, *value);
    return hash;
}

static uint64_t scene_logical_digest(const scene_pack& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    const scene_metadata& metadata = value.metadata;
    logical_hash_string(hash, metadata.id);
    logical_hash_string(hash, metadata.label);
    logical_hash_string(hash, metadata.provenance_kind);
    logical_hash_word(
        hash,
        static_cast<uint64_t>(metadata.provenance_source_ids.size()));
    for (const std::string& id : metadata.provenance_source_ids) {
        logical_hash_string(hash, id);
    }
    logical_hash_string(hash, metadata.coordinate_signature);
    logical_hash_string(hash, metadata.surface_signature);
    logical_hash_artifact(hash, metadata.heightfield);
    logical_hash_artifact(hash, metadata.mesh);
    logical_hash_artifact(hash, metadata.walkability);
    logical_hash_value(hash, metadata.heightfield_nx);
    logical_hash_value(hash, metadata.heightfield_nz);
    logical_hash_value(hash, metadata.walkability_nx);
    logical_hash_value(hash, metadata.walkability_nz);
    logical_hash_value(hash, metadata.heightfield_origin_x);
    logical_hash_value(hash, metadata.heightfield_origin_z);
    logical_hash_value(hash, metadata.heightfield_cell_size);
    logical_hash_value(hash, metadata.heightfield_exterior_height);
    logical_hash_bounds3d(hash, metadata.mesh_bounds);
    logical_hash_bounds3d(hash, metadata.heightfield_bounds);
    logical_hash_bounds2(hash, metadata.playable_bounds);
    logical_hash_bounds2(hash, metadata.lookahead_bounds);
    logical_hash_value(hash, metadata.spawn_position);
    logical_hash_value(hash, metadata.spawn_yaw);
    const std::vector<scene_region>* region_sets[] = {
        &metadata.certified_regions,
        &metadata.stress_regions,
        &metadata.blocked_regions,
    };
    for (const std::vector<scene_region>* regions : region_sets) {
        logical_hash_word(hash, static_cast<uint64_t>(regions->size()));
        for (const scene_region& region : *regions) {
            logical_hash_string(hash, region.id);
            logical_hash_bounds2(hash, region.bounds);
        }
    }
    logical_hash_word(
        hash, static_cast<uint64_t>(metadata.routes.size()));
    for (const scene_route& route : metadata.routes) {
        logical_hash_scene_route(hash, route);
    }
    logical_hash_value(hash, value.terrain.version);
    logical_hash_value(hash, value.terrain.nx);
    logical_hash_value(hash, value.terrain.nz);
    logical_hash_value(hash, value.terrain.origin_x);
    logical_hash_value(hash, value.terrain.origin_z);
    logical_hash_value(hash, value.terrain.cell_size);
    logical_hash_value(hash, value.terrain.exterior_height);
    logical_hash_array(hash, value.terrain.heights);
    logical_hash_value(hash, value.walkability.nx);
    logical_hash_value(hash, value.walkability.nz);
    logical_hash_array(hash, value.walkability.cells);
    logical_hash_string(hash, value.scene_path);
    logical_hash_string(hash, value.terrain_path);
    logical_hash_string(hash, value.mesh_path);
    logical_hash_string(hash, value.walkability_path);
    return hash;
}

struct ConstArtifactEvidence
{
    uint64_t db = 0U;
    uint64_t support = 0U;
    uint64_t scene = 0U;
    uint64_t route = 0U;
    G1FrameExternalInputs external;
};

static ConstArtifactEvidence const_artifact_evidence(
    const G1FrameExternalInputs& external)
{
    ConstArtifactEvidence evidence;
    evidence.db = database_logical_digest(*external.db);
    evidence.support = support_logical_digest(*external.support);
    evidence.scene = scene_logical_digest(*external.scene);
    evidence.route = route_logical_digest(external.route);
    evidence.external = external;
    return evidence;
}

static void check_const_artifacts(
    const G1FrameExternalInputs& external,
    const ConstArtifactEvidence& before,
    const char* message)
{
    check(database_logical_digest(*external.db) == before.db &&
              support_logical_digest(*external.support) == before.support &&
              scene_logical_digest(*external.scene) == before.scene &&
              route_logical_digest(external.route) == before.route &&
              same_external_lowering(external, before.external),
          message);
}

struct ProductionTrace
{
    int calls[G1FrameStageCount] = {};
    G1FrameTransactionStage order[G1FrameStageCount] = {};
    int total = 0;
    bool immutable_context_exact = true;
    bool requested_intent_ready = false;
    G1CommandIntent requested_intent;
    bool scripted_azimuth_applied = false;
    bool footprint_and_begin_ready = false;
    bool support_local_pose_ready = false;
    bool begin_root_only_ownership_exact = false;
    vec3 support_local_positions[G1_BoneCount] = {};
    quat support_local_rotations[G1_BoneCount] = {};
    bool foot0_completed = false;
    bool foot1_completed = false;
    bool final_fk_completed = false;
    G1IkState final_candidate_state;
    G1IkFrameResult final_candidate_result;
    vec3 final_local_positions[G1_BoneCount] = {};
    quat final_local_rotations[G1_BoneCount] = {};
    vec3 final_global_positions[G1_BoneCount] = {};
    quat final_global_rotations[G1_BoneCount] = {};
    bool accepted_diagnostic_ready = false;
    G1FrameAcceptedDiagnostic accepted_diagnostic_candidate;
    bool down_step_schedule_exact = false;
    bool prior_latch_seen = false;
    G1IkSafeStopHandoff latch_handoff;
};

static ProductionTrace production_trace;
static G1FrameExternalInputs expected_external_value;
static bool expected_external_ready = false;
static float expected_input_camera_azimuth = 0.0f;
static bool expect_down_step_schedule = false;

static void reset_production_trace(
    const G1FrameExternalInputs& external,
    float input_camera_azimuth)
{
    production_trace = ProductionTrace{};
    expected_external_value = external;
    expected_external_ready = true;
    expected_input_camera_azimuth = input_camera_azimuth;
}

struct ProductionRunnerProxyTrace
{
    int calls[G1FrameStageCount] = {};
    G1FrameTransactionStage order[G1FrameStageCount] = {};
    G1FrameStageOutcome outcomes[G1FrameStageCount] = {};
    int total = 0;
    bool terminal_ready = false;
    G1FrameTransactionStage terminal_stage = G1FrameStageCount;
    G1FrameStageOutcome terminal_outcome = G1FrameStageContinue;
};

static ProductionRunnerProxyTrace production_runner_proxy_trace;

static void reset_production_runner_proxy_trace()
{
    production_runner_proxy_trace = ProductionRunnerProxyTrace{};
}

static G1FrameStageOutcome recording_production_runner(
    G1FrameTransactionStage stage,
    g1_controller_state& working_state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity)
{
    const int stage_index = static_cast<int>(stage);
    if (stage_index < 0 || stage_index >= G1FrameStageCount ||
        production_runner_proxy_trace.total >= G1FrameStageCount ||
        production_runner_proxy_trace.calls[stage_index] != 0) {
        return G1FrameStageGlobalError;
    }
    const int record = production_runner_proxy_trace.total++;
    ++production_runner_proxy_trace.calls[stage_index];
    production_runner_proxy_trace.order[record] = stage;
    const G1FrameStageOutcome outcome =
        ::g1_controller_frame_stage_run(
            stage,
            working_state,
            scratch,
            external,
            error,
            error_capacity);
    production_runner_proxy_trace.outcomes[record] = outcome;
    if (outcome != G1FrameStageContinue) {
        production_runner_proxy_trace.terminal_ready = true;
        production_runner_proxy_trace.terminal_stage = stage;
        production_runner_proxy_trace.terminal_outcome = outcome;
    }
    return outcome;
}

static_assert(std::is_same<
                  decltype(&recording_production_runner),
                  G1FrameStageRunner>::value,
              "the recording proxy has the exact production runner type");

static G1FrameRejectionDiagnostic production_finite_rejection()
{
    G1FrameRejectionDiagnostic rejection;
    rejection.rejected = true;
    rejection.stage = G1FrameRejectFootprint;
    rejection.stop_reason = G1IkStopFootprintOutsideDomain;
    rejection.footprint_status = G1FootprintOutsideDomain;
    return rejection;
}

static G1FrameInjectedOutcome production_hook(
    G1FrameTransactionStage stage,
    const g1_controller_state& working_state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    const G1FrameTransactionTestControl& control,
    char*, int)
{
    const int index = static_cast<int>(stage);
    if (index < 0 || index >= G1FrameStageCount ||
        production_trace.total >= G1FrameStageCount) {
        return G1FrameInjectGlobalError;
    }
    ++production_trace.calls[index];
    production_trace.order[production_trace.total++] = stage;
    production_trace.immutable_context_exact =
        production_trace.immutable_context_exact &&
        expected_external_ready &&
        same_external_lowering(external, expected_external_value);

    if (stage == G1FrameStageInputRouteCommand) {
        production_trace.requested_intent_ready =
            scratch.requested_intent_ready;
        production_trace.requested_intent = scratch.requested_intent;
        production_trace.scripted_azimuth_applied = same_float_bits(
            working_state.camera_azimuth,
            expected_input_camera_azimuth +
                external.input.scripted_azimuth_delta);
        production_trace.prior_latch_seen =
            scratch.prior_safe_stop_latched;
        production_trace.latch_handoff = scratch.safe_stop_handoff;
    } else if (stage == G1FrameStageContactUpdate) {
        production_trace.support_local_pose_ready =
            working_state.adjusted_bone_positions.size == G1_BoneCount &&
            working_state.adjusted_bone_rotations.size == G1_BoneCount;
        if (production_trace.support_local_pose_ready) {
            for (int bone = 0; bone < G1_BoneCount; ++bone) {
                production_trace.support_local_positions[bone] =
                    working_state.adjusted_bone_positions(bone);
                production_trace.support_local_rotations[bone] =
                    working_state.adjusted_bone_rotations(bone);
            }
        }
    } else if (stage == G1FrameStageFootprintObservation) {
        production_trace.footprint_and_begin_ready =
            scratch.footprint_status == G1FootprintOk;
        if (expect_down_step_schedule) {
            const bool expected[4] = {false, false, true, true};
            production_trace.down_step_schedule_exact = true;
            for (int sample = 0; sample < 4; ++sample) {
                production_trace.down_step_schedule_exact =
                    production_trace.down_step_schedule_exact &&
                    scratch.contact_schedule.contact[0][sample] ==
                        expected[sample];
            }
        }
    } else if (stage == G1FrameStageIkBegin) {
        production_trace.footprint_and_begin_ready =
            production_trace.footprint_and_begin_ready &&
            scratch.raw_certificate.ik_transaction.initialized &&
            scratch.ik_certificate.ik_transaction.initialized;
        float expected_root_y = 0.0f;
        production_trace.begin_root_only_ownership_exact =
            production_trace.support_local_pose_ready &&
            g1_apply_root_reach_plan_y(
                expected_root_y,
                production_trace
                    .support_local_positions[G1_Simulation].y,
                scratch.ik_certificate.ik_transaction
                    .candidate_result.root_reach) &&
            terrain_float_bits(
                working_state.ik_candidate_bone_positions(
                    G1_Simulation).y) ==
                terrain_float_bits(expected_root_y);
        for (int bone = 0;
             production_trace.begin_root_only_ownership_exact &&
             bone < G1_BoneCount;
             ++bone) {
            const vec3 actual_position =
                working_state.ik_candidate_bone_positions(bone);
            const vec3 baseline_position =
                production_trace.support_local_positions[bone];
            production_trace.begin_root_only_ownership_exact =
                same_quat_bits(
                    working_state.ik_candidate_bone_rotations(bone),
                    production_trace.support_local_rotations[bone]) &&
                (bone == G1_Simulation
                     ? terrain_float_bits(actual_position.x) ==
                           terrain_float_bits(baseline_position.x) &&
                           terrain_float_bits(actual_position.z) ==
                           terrain_float_bits(baseline_position.z)
                     : same_vec3_bits(
                           actual_position, baseline_position));
        }
    } else if (stage == G1FrameStageIkFirstFoot) {
        production_trace.foot0_completed =
            scratch.ik_certificate.ik_transaction.next_foot == 1;
    } else if (stage == G1FrameStageIkSecondFoot) {
        production_trace.foot1_completed =
            scratch.ik_certificate.ik_transaction.next_foot == 2;
    } else if (stage == G1FrameStageRawFinalFk ||
               stage == G1FrameStageIkFinalFk) {
        const bool visible_stage = external.tuning.ik_enabled
            ? stage == G1FrameStageIkFinalFk
            : stage == G1FrameStageRawFinalFk;
        const G1FrameBranchCertificateScratch& certificate =
            stage == G1FrameStageIkFinalFk
                ? scratch.ik_certificate
                : scratch.raw_certificate;
        const bool branch_enabled = stage == G1FrameStageIkFinalFk;
        const bool ik_application_exact = branch_enabled
            ? certificate.ik_transaction.candidate_result.applied &&
                working_state.ik_frame.applied
            : !certificate.ik_transaction.candidate_result.applied &&
                !working_state.ik_frame.applied &&
                g1_ik_runtime_is_disabled_noop(
                    certificate.ik_transaction);
        production_trace.final_fk_completed =
            production_trace.final_fk_completed ||
            (visible_stage &&
             certificate.ik_transaction.initialized &&
             certificate.ik_transaction.next_foot == 2 &&
             ik_application_exact &&
             ik_state_logical_digest(working_state.ik) ==
                 ik_state_logical_digest(
                     certificate.ik_transaction.candidate_state) &&
             ik_frame_logical_digest(working_state.ik_frame) ==
                 ik_frame_logical_digest(
                     certificate.ik_transaction.candidate_result) &&
             working_state.ik_bone_positions.size == G1_BoneCount &&
             working_state.ik_bone_rotations.size == G1_BoneCount &&
             working_state.ik_global_bone_positions.size == G1_BoneCount &&
             working_state.ik_global_bone_rotations.size == G1_BoneCount);
        if (stage == G1FrameStageIkFinalFk &&
            certificate.ik_transaction.initialized &&
            certificate.ik_transaction.next_foot == 2) {
            production_trace.final_candidate_state =
                certificate.ik_transaction.candidate_state;
        }
        if (visible_stage && production_trace.final_fk_completed) {
            production_trace.final_candidate_result =
                certificate.ik_transaction.candidate_result;
            for (int bone = 0; bone < G1_BoneCount; ++bone) {
                production_trace.final_local_positions[bone] =
                    working_state.ik_bone_positions(bone);
                production_trace.final_local_rotations[bone] =
                    working_state.ik_bone_rotations(bone);
                production_trace.final_global_positions[bone] =
                    working_state.ik_global_bone_positions(bone);
                production_trace.final_global_rotations[bone] =
                    working_state.ik_global_bone_rotations(bone);
            }
        }
    } else if (stage == G1FrameStageAcceptedFinalize) {
        G1FrameAcceptedDiagnostic expected;
        expected.ready = true;
        expected.presentation_frame =
            external.input.presentation_frame;
        expected.scene_frame = working_state.scene_frame - 1;
        expected.route = scratch.route_sample;
        expected.traversal = scratch.traversal;
        for (int dimension = 0; dimension < 31; ++dimension) {
            expected.query[dimension] = scratch.query[dimension];
        }
        expected.query_database_frame =
            scratch.query_database_frame;
        expected.query_range = scratch.query_range;
        expected.selected_database_frame =
            scratch.selected_database_frame;
        expected.terrain_query = scratch.terrain_query;
        expected.raw_selected = scratch.raw_selected_diagnostic;
        expected.inertialized = scratch.inertialized_diagnostic;
        expected.support_retargeted =
            scratch.support_retargeted_diagnostic;
        expected.rendered = scratch.rendered_diagnostic;
        expected.matching_enabled =
            external.tuning.mode != G1_TestSequential;
        expected.adjustment_enabled =
            external.tuning.adjustment_enabled;
        expected.clamping_enabled =
            external.tuning.clamping_enabled;
        expected.ik_enabled = external.tuning.ik_enabled;
        expected.effective_terrain_weight =
            external.tuning.effective_terrain_weight;
        production_trace.accepted_diagnostic_ready =
            scratch.accepted_diagnostic_ready &&
            diagnostic_logical_digest(
                scratch.accepted_diagnostic_candidate) ==
                diagnostic_logical_digest(expected);
        if (production_trace.accepted_diagnostic_ready) {
            production_trace.accepted_diagnostic_candidate = expected;
        }
    }

    if (stage != control.injected_stage) {
        return G1FrameInjectContinue;
    }
    if (control.injected_outcome == G1FrameInjectFiniteReject) {
        scratch.rejection = production_finite_rejection();
    }
    return control.injected_outcome;
}

static void check_production_trace_through(
    int last_stage, const char* message)
{
    check(production_trace.total == last_stage + 1, message);
    for (int stage = 0; stage < G1FrameStageCount; ++stage) {
        check(production_trace.calls[stage] ==
                  (stage <= last_stage ? 1 : 0),
              message);
    }
    for (int stage = 0; stage <= last_stage; ++stage) {
        check(production_trace.order[stage] ==
                  static_cast<G1FrameTransactionStage>(stage),
              message);
    }
}

static void check_complete_success_publication(
    const G1FrameRuntime& runtime,
    const G1FrameExternalInputs& external,
    const char* message)
{
    G1FramePublication expected_publication;
    expected_publication.requested_intent =
        production_trace.requested_intent;
    expected_publication.presentation_frame =
        external.input.presentation_frame;
    check(production_trace.requested_intent_ready &&
              production_trace.accepted_diagnostic_ready &&
              publication_logical_digest(runtime.publication) ==
                  publication_logical_digest(expected_publication) &&
              diagnostic_logical_digest(runtime.accepted_diagnostic) ==
                  diagnostic_logical_digest(
                      production_trace.accepted_diagnostic_candidate) &&
              terrain_float_bits(
                  runtime.accepted_diagnostic
                      .support_retargeted.hips_y) ==
                  terrain_float_bits(
                      runtime.accepted_state
                          .global_bone_positions(G1_Hips).y) &&
              terrain_float_bits(
                  runtime.accepted_diagnostic.rendered.hips_y) ==
                  terrain_float_bits(
                      runtime.accepted_state
                          .ik_global_bone_positions(G1_Hips).y),
          message);
}

static void check_complete_accepted_final_fk(
    const fixture& value, const char* message)
{
    const g1_controller_state& accepted =
        value.runtime.accepted_state;
    check(production_trace.final_fk_completed &&
              ik_state_logical_digest(accepted.ik) ==
                  ik_state_logical_digest(
                      production_trace.final_candidate_state) &&
              ik_frame_logical_digest(accepted.ik_frame) ==
                  ik_frame_logical_digest(
                      production_trace.final_candidate_result),
          message);

    float expected_root_y = 0.0f;
    check(g1_apply_root_reach_plan_y(
              expected_root_y,
              accepted.adjusted_bone_positions(G1_Simulation).y,
              accepted.ik_frame.root_reach) &&
              terrain_float_bits(
                  accepted.ik_bone_positions(G1_Simulation).y) ==
                  terrain_float_bits(expected_root_y),
          message);
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        const vec3 baseline = accepted.adjusted_bone_positions(bone);
        const vec3 final_local = accepted.ik_bone_positions(bone);
        check(bone == G1_Simulation
                  ? terrain_float_bits(final_local.x) ==
                        terrain_float_bits(baseline.x) &&
                        terrain_float_bits(final_local.z) ==
                        terrain_float_bits(baseline.z)
                  : same_vec3_bits(final_local, baseline),
              message);
    }

    array1d<vec3> independent_positions(G1_BoneCount);
    array1d<quat> independent_rotations(G1_BoneCount);
    char error[1024] = {};
    check(g1_ik_checked_forward_kinematics(
              independent_positions,
              independent_rotations,
              accepted.ik_bone_positions,
              accepted.ik_bone_rotations,
              value.db.bone_parents,
              error,
              static_cast<int>(sizeof(error))),
          error);
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        check(same_vec3_bits(
                  accepted.ik_bone_positions(bone),
                  production_trace.final_local_positions[bone]) &&
                  same_quat_bits(
                      accepted.ik_bone_rotations(bone),
                      production_trace.final_local_rotations[bone]) &&
                  same_vec3_bits(
                      accepted.ik_global_bone_positions(bone),
                      production_trace.final_global_positions[bone]) &&
                  same_quat_bits(
                      accepted.ik_global_bone_rotations(bone),
                      production_trace.final_global_rotations[bone]) &&
                  same_vec3_bits(
                      accepted.ik_global_bone_positions(bone),
                      independent_positions(bone)) &&
                  same_quat_bits(
                      accepted.ik_global_bone_rotations(bone),
                      independent_rotations(bone)),
              message);
    }
}

struct ProductionEvidence
{
    uint64_t accepted = 0U;
    uint64_t working = 0U;
    StateStorageIdentities accepted_storage;
    StateStorageIdentities working_storage;
    uint64_t publication = 0U;
    uint64_t diagnostic = 0U;
};

static ProductionEvidence production_evidence(
    const G1FrameRuntime& runtime)
{
    ProductionEvidence evidence;
    evidence.accepted = state_logical_digest(runtime.accepted_state);
    evidence.working = state_logical_digest(runtime.working_state);
    evidence.accepted_storage =
        state_storage_identities(runtime.accepted_state);
    evidence.working_storage =
        state_storage_identities(runtime.working_state);
    evidence.publication = publication_logical_digest(runtime.publication);
    evidence.diagnostic =
        diagnostic_logical_digest(runtime.accepted_diagnostic);
    return evidence;
}

static void check_production_global_preservation(
    const G1FrameRuntime& runtime,
    const ProductionEvidence& before,
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

static void check_production_preflight_preservation(
    const G1FrameRuntime& runtime,
    const ProductionEvidence& before,
    const char* message)
{
    check_production_global_preservation(runtime, before, message);
    check(state_logical_digest(runtime.working_state) == before.working,
          message);
}

static scene_route production_route(
    const char* id, float destination_x, float destination_z)
{
    scene_route route;
    route.id = id;
    route.expected_outcome = "pass";
    route.walkability_class = 1;
    route.waypoints_xz.push_back(std::make_pair(2.0f, 2.0f));
    route.waypoints_xz.push_back(
        std::make_pair(destination_x, destination_z));
    return route;
}

static void configure_production_mode(fixture& value, g1_test_mode mode)
{
    value.external.input.move_stick = vec3(0.375f, 0.0f, -0.625f);
    value.external.input.look_stick = vec3(0.125f, 0.0f, -0.25f);
    value.external.input.gait_target = 0.75f;
    value.external.input.camera_zoom_axis = -0.125f;
    value.external.input.scripted_azimuth_delta = 0.03125f;
    value.external.input.desired_strafe = true;
    value.external.input.presentation_frame = 73;
    value.external.heading_override.active =
        mode != G1_TestRoute && mode != G1_TestLive;
    value.external.heading_override.heading =
        quat(0.923879504f, 0.0f, 0.382683426f, 0.0f);

    value.external.tuning.mode = mode;
    value.external.tuning.frame_limit =
        mode == G1_TestLive
            ? 0
            : mode == G1_TestSceneCycle ? 272 : 96;
    value.external.tuning.scene_dwell_frames = 17;
    value.external.tuning.dt = 1.0f / 25.0f;
    value.external.tuning.trajectory_sample_time = 0.25f;
    value.external.tuning.route_speed = 0.4375f;
    value.external.tuning.future_speed_scale = 0.875f;
    value.external.tuning.walkability_radius = 0.1875f;
    value.external.tuning.effective_terrain_weight =
        mode == G1_TestFlat || mode == G1_TestSequential
            ? 0.0f
            : mode == G1_TestLive
                ? 0.25f
                : mode == G1_TestRoute
                    ? 0.625f
                    : mode == G1_TestSceneCycle ? 0.50f : 0.8125f;
    value.external.tuning.initial_search_time = 0.375f;
    value.external.tuning.inertialize_blending_halflife = 0.0625f;
    value.external.tuning.desired_velocity_change_threshold = 37.5f;
    value.external.tuning.desired_rotation_change_threshold = 43.75f;
    value.external.tuning.simulation_velocity_halflife = 0.3125f;
    value.external.tuning.simulation_rotation_halflife = 0.1875f;
    value.external.tuning.simulation_run_forward_speed = 0.875f;
    value.external.tuning.simulation_run_side_speed = 0.5625f;
    value.external.tuning.simulation_run_back_speed = 0.53125f;
    value.external.tuning.simulation_walk_forward_speed = 0.46875f;
    value.external.tuning.simulation_walk_side_speed = 0.34375f;
    value.external.tuning.simulation_walk_back_speed = 0.328125f;
    value.external.tuning.synchronization_enabled = true;
    value.external.tuning.synchronization_data_factor = 0.9375f;
    value.external.tuning.adjustment_enabled = true;
    value.external.tuning.adjustment_by_velocity_enabled = false;
    value.external.tuning.adjustment_position_halflife = 0.09375f;
    value.external.tuning.adjustment_rotation_halflife = 0.15625f;
    value.external.tuning.adjustment_position_max_ratio = 0.4375f;
    value.external.tuning.adjustment_rotation_max_ratio = 0.40625f;
    value.external.tuning.clamping_enabled = true;
    value.external.tuning.clamping_max_distance = 0.1375f;
    value.external.tuning.clamping_max_angle = 0.75f;
    value.external.tuning.contact_unlock_radius = 0.1875f;
    value.external.tuning.contact_foot_height = 0.01875f;
    value.external.tuning.contact_blending_halflife = 0.078125f;
    value.external.tuning.ik_enabled = false;
    value.external.route = NULL;

    if (mode == G1_TestTerrain || mode == G1_TestRoute ||
        mode == G1_TestSceneCycle) {
        for (int z = 0; z < value.scene.terrain.nz; ++z) {
            for (int x = 0; x < value.scene.terrain.nx; ++x) {
                const float world_x = value.scene.terrain.origin_x +
                    value.scene.terrain.cell_size * static_cast<float>(x);
                value.scene.terrain.heights(
                    z * value.scene.terrain.nx + x) =
                    0.01f * (world_x - 2.0f);
            }
        }
        for (int frame = 0; frame < value.db.nframes(); ++frame) {
            for (int feature = 0; feature < 4; ++feature) {
                value.db.terrain_features(frame, feature) =
                    0.015625f * static_cast<float>(feature + 1);
            }
        }
    }

    if (mode == G1_TestRoute) {
        value.scene.metadata.routes.push_back(
            production_route("production-route", 4.0f, 2.0f));
    }
    G1FrameResetConfig config;
    config.route_mode = mode == G1_TestRoute;
    config.route_id = mode == G1_TestRoute
        ? value.scene.metadata.routes[0].id.c_str()
        : NULL;
    config.ik_enabled = value.external.tuning.ik_enabled;
    config.dt = value.external.tuning.dt;
    config.trajectory_sample_time =
        value.external.tuning.trajectory_sample_time;
    config.initial_search_time = value.external.tuning.initial_search_time;
    char error[512] = {};
    check(g1_frame_runtime_reset(
              value.runtime, value.db, value.support, value.scene,
              config, error, static_cast<int>(sizeof(error))),
          error);
    value.external.db = &value.db;
    value.external.support = &value.support;
    value.external.scene = &value.scene;
    value.external.route = mode == G1_TestRoute
        ? &value.scene.metadata.routes[0]
        : NULL;
}

static vec3 live_velocity_oracle(
    const G1FrameExternalInputs& external,
    float camera_azimuth,
    quat simulation_rotation,
    float gait)
{
    const float forward = lerpf(
        external.tuning.simulation_run_forward_speed,
        external.tuning.simulation_walk_forward_speed,
        gait);
    const float side = lerpf(
        external.tuning.simulation_run_side_speed,
        external.tuning.simulation_walk_side_speed,
        gait);
    const float back = lerpf(
        external.tuning.simulation_run_back_speed,
        external.tuning.simulation_walk_back_speed,
        gait);
    const vec3 world_stick = quat_mul_vec3(
        quat_from_angle_axis(camera_azimuth, vec3(0.0f, 1.0f, 0.0f)),
        external.input.move_stick);
    const vec3 local_stick =
        quat_inv_mul_vec3(simulation_rotation, world_stick);
    const vec3 scaled =
        (local_stick.z > 0.0f
             ? vec3(side, 0.0f, forward)
             : vec3(side, 0.0f, back)) *
        local_stick;
    return quat_mul_vec3(simulation_rotation, scaled);
}

static quat live_strafe_heading_oracle(
    const G1FrameExternalInputs& external,
    float camera_azimuth)
{
    vec3 direction = quat_mul_vec3(
        quat_from_angle_axis(camera_azimuth, vec3(0.0f, 1.0f, 0.0f)),
        vec3(0.0f, 0.0f, -1.0f));
    if (length(external.input.look_stick) > 0.01f) {
        direction = quat_mul_vec3(
            quat_from_angle_axis(
                camera_azimuth, vec3(0.0f, 1.0f, 0.0f)),
            normalize(external.input.look_stick));
    }
    return quat_from_angle_axis(
        atan2f(direction.x, direction.z), vec3(0.0f, 1.0f, 0.0f));
}

static void test_real_runner_covers_all_six_modes_and_lowerings()
{
    const g1_test_mode modes[] = {
        G1_TestLive,
        G1_TestSequential,
        G1_TestFlat,
        G1_TestTerrain,
        G1_TestRoute,
        G1_TestSceneCycle,
    };
    for (const g1_test_mode mode : modes) {
        fixture value;
        configure_production_mode(value, mode);
        const StateStorageIdentities accepted_before =
            state_storage_identities(value.runtime.accepted_state);
        const StateStorageIdentities working_before =
            state_storage_identities(value.runtime.working_state);
        const float camera_before =
            value.runtime.accepted_state.camera_azimuth;
        const float altitude_before =
            value.runtime.accepted_state.camera_altitude;
        const float distance_before =
            value.runtime.accepted_state.camera_distance;
        const float gait_before = value.runtime.accepted_state.desired_gait;
        const float gait_velocity_before =
            value.runtime.accepted_state.desired_gait_velocity;
        const quat simulation_rotation_before =
            value.runtime.accepted_state.simulation_rotation;
        const int frame_index_before =
            value.runtime.accepted_state.frame_index;
        const int route_frames_before =
            value.runtime.accepted_state.route_frames;
        const ConstArtifactEvidence artifacts_before =
            const_artifact_evidence(value.external);
        G1FrameTransactionTestSeam seam;
        seam.hook = production_hook;
        char error[1024] = {};
        reset_production_trace(value.external, camera_before);
        check(g1_frame_transaction_run(
                  value.runtime,
                  g1_controller_frame_stage_run,
                  value.external,
                  &seam,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionAccepted,
              error);
        check_production_trace_through(
            G1FrameStageCount - 1,
            "the real runner executes all twelve stages once in enum order");
        check(production_trace.immutable_context_exact &&
                  production_trace.requested_intent_ready &&
                  production_trace.scripted_azimuth_applied &&
                  production_trace.support_local_pose_ready &&
                  production_trace.footprint_and_begin_ready &&
                  production_trace.begin_root_only_ownership_exact &&
                  production_trace.foot0_completed &&
                  production_trace.foot1_completed &&
                  production_trace.final_fk_completed &&
                  production_trace.accepted_diagnostic_ready,
              "the typed external snapshot remains immutable through every real stage");
        check_complete_success_publication(
            value.runtime,
            value.external,
            "real success publishes the complete canonical publication and exact accepted diagnostic candidate");
        check_complete_accepted_final_fk(
            value,
            "real success publishes the completed IK transaction and exact independently checked final FK");
        check(g1_controller_state_is_valid(value.runtime.accepted_state) &&
                  value.runtime.accepted_diagnostic.ready &&
                  value.runtime.accepted_diagnostic.presentation_frame ==
                      value.external.input.presentation_frame &&
                  value.runtime.accepted_diagnostic.scene_frame ==
                      value.runtime.accepted_state.scene_frame - 1 &&
                  value.runtime.accepted_diagnostic.adjustment_enabled ==
                      value.external.tuning.adjustment_enabled &&
                  value.runtime.accepted_diagnostic.clamping_enabled ==
                      value.external.tuning.clamping_enabled &&
                  value.runtime.accepted_diagnostic.ik_enabled ==
                      value.external.tuning.ik_enabled &&
                  same_float_bits(
                      value.runtime.accepted_diagnostic
                          .effective_terrain_weight,
                      value.external.tuning.effective_terrain_weight),
              "the real success publishes a complete value-only accepted diagnostic");
        check(same_intent_bits(
                  value.runtime.publication.requested_intent,
                  production_trace.requested_intent) &&
                  value.runtime.publication.presentation_frame ==
                      value.external.input.presentation_frame &&
                  !value.runtime.publication.rejection.rejected &&
                  !value.runtime.publication.ik_safe_stop_latched,
              "real success publishes exact intent/frame and clears rejection/latch");
        if (value.external.heading_override.active) {
            check(same_quat_bits(
                      value.runtime.publication.requested_intent
                          .desired_heading,
                      value.external.heading_override.heading),
                  "the independent heading override reaches publication unchanged");
        }
        if (mode == G1_TestLive) {
            float expected_gait = gait_before;
            float expected_gait_velocity = gait_velocity_before;
            simple_spring_damper_exact(
                expected_gait,
                expected_gait_velocity,
                value.external.input.gait_target,
                0.10f,
                value.external.tuning.dt);
            const float command_azimuth = camera_before +
                value.external.input.scripted_azimuth_delta;
            const G1CommandIntent expected_intent = {
                live_velocity_oracle(
                    value.external,
                    command_azimuth,
                    simulation_rotation_before,
                    expected_gait),
                live_strafe_heading_oracle(
                    value.external, command_azimuth)
            };
            const float expected_distance = clampf(
                distance_before +
                    10.0f * value.external.tuning.dt *
                        value.external.input.camera_zoom_axis,
                0.1f,
                100.0f);
            check(same_intent_bits(
                      value.runtime.publication.requested_intent,
                      expected_intent) &&
                      same_float_bits(
                          value.runtime.accepted_state.desired_gait,
                          expected_gait) &&
                      same_float_bits(
                          value.runtime.accepted_state
                              .desired_gait_velocity,
                          expected_gait_velocity) &&
                      same_float_bits(
                          value.runtime.accepted_state.camera_altitude,
                          altitude_before) &&
                      same_float_bits(
                          value.runtime.accepted_state.camera_distance,
                          expected_distance),
                  "live mode independently authenticates move/look/gait/zoom/strafe lowering");
        } else if (mode == G1_TestSequential) {
            check(!value.runtime.accepted_diagnostic.matching_enabled &&
                      !value.runtime.accepted_state.searched &&
                      value.runtime.accepted_state.frame_index ==
                          database_trajectory_index_clamp(
                              value.db, frame_index_before, 1) &&
                      same_float_bits(
                          value.runtime.accepted_diagnostic
                              .effective_terrain_weight,
                          0.0f),
                  "sequential mode certifies continuation without matcher search");
        } else if (mode == G1_TestFlat) {
            bool all_flat = true;
            for (int sample = 0; sample < 4; ++sample) {
                all_flat = all_flat && same_float_bits(
                    value.runtime.accepted_diagnostic
                        .terrain_query.values[sample],
                    0.0f);
            }
            check(value.runtime.accepted_diagnostic.matching_enabled &&
                      all_flat &&
                      same_float_bits(
                          value.runtime.accepted_diagnostic
                              .effective_terrain_weight,
                          0.0f),
                  "flat mode publishes an exact zero terrain query and weight");
        } else if (mode == G1_TestTerrain) {
            bool has_nonzero_terrain = false;
            for (int sample = 0; sample < 4; ++sample) {
                has_nonzero_terrain = has_nonzero_terrain ||
                    !same_float_bits(
                        value.runtime.accepted_diagnostic
                            .terrain_query.values[sample],
                        0.0f);
            }
            check(value.runtime.accepted_diagnostic.matching_enabled &&
                      has_nonzero_terrain &&
                      same_float_bits(
                          value.runtime.accepted_diagnostic
                              .effective_terrain_weight,
                          0.8125f),
                  "terrain mode publishes a distinct nonzero terrain query and weight");
        } else if (mode == G1_TestRoute) {
            const vec3 route_command =
                value.runtime.accepted_diagnostic.route.command;
            const quat expected_route_heading = quat_from_angle_axis(
                atan2f(route_command.x, route_command.z),
                vec3(0.0f, 1.0f, 0.0f));
            check(value.external.route ==
                      &value.scene.metadata.routes[0] &&
                      length(route_command) > 0.0f &&
                      same_vec3_bits(
                          route_command,
                          value.runtime.publication.requested_intent
                              .requested_velocity) &&
                      same_quat_bits(
                          expected_route_heading,
                          value.runtime.publication.requested_intent
                              .desired_heading) &&
                      value.runtime.accepted_state.route_frames >
                          route_frames_before,
                  "route mode consumes the resolved route, advances its cursor, and derives route heading");
        } else {
            check(value.external.route == NULL &&
                      value.external.tuning.scene_dwell_frames == 17 &&
                      value.runtime.accepted_diagnostic.matching_enabled &&
                      value.runtime.accepted_state.scene_frame == 1,
                  "scene-cycle mode runs one accepted route-null frame under its explicit dwell contract");
        }
        check(same_storage_identities(
                  state_storage_identities(value.runtime.accepted_state),
                  working_before) &&
                  same_storage_identities(
                      state_storage_identities(value.runtime.working_state),
                      accepted_before),
              "real success performs exactly one complete 49-owner swap");
        check_const_artifacts(
            value.external,
            artifacts_before,
            "the real runner cannot mutate const DB/support/scene/route/external artifacts");
    }
}

static void test_scene_cycle_reaches_exact_dwell_boundary()
{
    fixture scene_cycle;
    fixture terrain_control;
    configure_production_mode(scene_cycle, G1_TestSceneCycle);
    configure_production_mode(terrain_control, G1_TestTerrain);
    scene_cycle.external.tuning.effective_terrain_weight =
        terrain_control.external.tuning.effective_terrain_weight;
    const int dwell = scene_cycle.external.tuning.scene_dwell_frames;
    check(dwell == 17 &&
              scene_cycle.runtime.accepted_state.scene_frame == 0 &&
              terrain_control.runtime.accepted_state.scene_frame == 0,
          "scene-cycle and same-weight Terrain controls begin before the exact dwell boundary");

    fixture* cases[] = {&scene_cycle, &terrain_control};
    for (int frame = 0; frame < dwell; ++frame) {
        for (fixture* value : cases) {
            const int scene_frame_before =
                value->runtime.accepted_state.scene_frame;
            value->external.input.presentation_frame = 500 + frame;
            const ConstArtifactEvidence artifacts_before =
                const_artifact_evidence(value->external);
            G1FrameTransactionTestSeam seam;
            seam.hook = production_hook;
            char error[1024] = {};
            reset_production_trace(
                value->external,
                value->runtime.accepted_state.camera_azimuth);
            check(g1_frame_transaction_run(
                      value->runtime,
                      g1_controller_frame_stage_run,
                      value->external,
                      &seam,
                      error,
                      static_cast<int>(sizeof(error))) ==
                      G1FrameTransactionAccepted,
                  error);
            check(value->runtime.accepted_state.scene_frame ==
                      scene_frame_before + 1 &&
                      value->runtime.accepted_diagnostic.scene_frame ==
                          scene_frame_before,
                  "each real SceneCycle/Terrain transaction advances exactly one accepted scene frame");
            check_production_trace_through(
                G1FrameStageAcceptedFinalize,
                "each dwell transaction completes all twenty real stages");
            check_complete_success_publication(
                value->runtime,
                value->external,
                "dwell run publishes a complete canonical accepted frame");
            check_complete_accepted_final_fk(
                *value,
                "dwell run publishes independently checked final FK values");
            check_const_artifacts(
                value->external,
                artifacts_before,
                "dwell run cannot perform caller-owned scene switching inside the runner");
        }
        if (frame + 1 < dwell) {
            check(scene_cycle.runtime.accepted_state.scene_frame < dwell,
                  "SceneCycle remains strictly before its dwell boundary until the final accepted frame");
        }
    }
    check(scene_cycle.runtime.accepted_state.scene_frame == dwell &&
              terrain_control.runtime.accepted_state.scene_frame == dwell &&
              same_float_bits(
                  scene_cycle.external.tuning.effective_terrain_weight,
                  terrain_control.external.tuning
                      .effective_terrain_weight),
          "real same-weight SceneCycle and Terrain transactions reach the exact dwell state; outer mode scheduling distinguishes the pending switch");
}

static void test_search_inertialization_and_simulation_tuning_oracles()
{
    {
        fixture value;
        configure_production_mode(value, G1_TestLive);
        value.runtime.accepted_state.search_timer = 0.0f;
        value.runtime.accepted_state.force_search_timer = 0.0f;
        check(g1_controller_state_is_valid(value.runtime.accepted_state),
              "search-time oracle source remains valid");
        float expected_timer = 0.0f;
        check(terrain_f32_sub(
                  expected_timer,
                  value.external.tuning.initial_search_time,
                  value.external.tuning.dt),
              "search-time oracle subtraction is representable");
        G1FrameTransactionTestSeam seam;
        seam.hook = production_hook;
        char error[1024] = {};
        const ConstArtifactEvidence artifacts_before =
            const_artifact_evidence(value.external);
        reset_production_trace(
            value.external, value.runtime.accepted_state.camera_azimuth);
        check(g1_frame_transaction_run(
                  value.runtime,
                  g1_controller_frame_stage_run,
                  value.external,
                  &seam,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionAccepted,
              error);
        check_production_trace_through(
            G1FrameStageAcceptedFinalize,
            "search-time oracle completes every real stage");
        check_complete_success_publication(
            value.runtime,
            value.external,
            "search-time oracle publishes the complete canonical success payload");
        check_complete_accepted_final_fk(
            value,
            "search-time oracle publishes exact independently checked final FK");
        check_const_artifacts(
            value.external,
            artifacts_before,
            "search-time oracle preserves every immutable production artifact");
        check(same_float_bits(
                  value.runtime.accepted_state.search_time, 0.375f) &&
                  same_float_bits(
                      value.runtime.accepted_state.search_timer,
                      expected_timer) &&
                  same_float_bits(
                      value.runtime.accepted_state.force_search_timer,
                      expected_timer),
              "validated MM_SEARCHT bits seed both real search timers exactly");
    }

    {
        fixture value;
        configure_production_mode(value, G1_TestSequential);
        const int bone = G1_BoneCount - 1;
        value.runtime.accepted_state.bone_offset_positions(bone) =
            vec3(0.03125f, -0.046875f, 0.0625f);
        value.runtime.accepted_state.bone_offset_velocities(bone) =
            vec3(-0.015625f, 0.0234375f, -0.0078125f);
        value.runtime.accepted_state.simulation_rotation = quat();
        value.runtime.accepted_state.simulation_angular_velocity =
            vec3(0.0f, 0.125f, 0.0f);
        check(g1_controller_state_is_valid(value.runtime.accepted_state),
              "inertial/simulation oracle source remains valid");

        vec3 expected_position;
        vec3 expected_velocity;
        vec3 expected_offset =
            value.runtime.accepted_state.bone_offset_positions(bone);
        vec3 expected_offset_velocity =
            value.runtime.accepted_state.bone_offset_velocities(bone);
        inertialize_update(
            expected_position,
            expected_velocity,
            expected_offset,
            expected_offset_velocity,
            value.db.bone_positions(1, bone),
            value.db.bone_velocities(1, bone),
            value.external.tuning.inertialize_blending_halflife,
            value.external.tuning.dt);
        quat expected_rotation =
            value.runtime.accepted_state.simulation_rotation;
        vec3 expected_angular_velocity =
            value.runtime.accepted_state.simulation_angular_velocity;
        simple_spring_damper_exact(
            expected_rotation,
            expected_angular_velocity,
            value.external.heading_override.heading,
            value.external.tuning.simulation_rotation_halflife,
            value.external.tuning.dt);

        G1FrameTransactionTestSeam seam;
        seam.hook = production_hook;
        char error[1024] = {};
        const ConstArtifactEvidence artifacts_before =
            const_artifact_evidence(value.external);
        reset_production_trace(
            value.external, value.runtime.accepted_state.camera_azimuth);
        check(g1_frame_transaction_run(
                  value.runtime,
                  g1_controller_frame_stage_run,
                  value.external,
                  &seam,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionAccepted,
              error);
        check_production_trace_through(
            G1FrameStageAcceptedFinalize,
            "inertial/simulation oracle completes every real stage");
        check_complete_success_publication(
            value.runtime,
            value.external,
            "inertial/simulation oracle publishes the complete canonical success payload");
        check_complete_accepted_final_fk(
            value,
            "inertial/simulation oracle publishes exact independently checked final FK");
        check_const_artifacts(
            value.external,
            artifacts_before,
            "inertial/simulation oracle preserves every immutable production artifact");
        check(same_vec3_bits(
                  value.runtime.accepted_state.bone_positions(bone),
                  expected_position) &&
                  same_vec3_bits(
                      value.runtime.accepted_state.bone_velocities(bone),
                      expected_velocity) &&
                  same_vec3_bits(
                      value.runtime.accepted_state
                          .bone_offset_positions(bone),
                      expected_offset) &&
                  same_vec3_bits(
                      value.runtime.accepted_state
                          .bone_offset_velocities(bone),
                      expected_offset_velocity),
              "MM_HALFLIFE drives the exact real inertializer update");
        check(same_quat_bits(
                  value.runtime.accepted_state.simulation_rotation,
                  expected_rotation) &&
                  same_vec3_bits(
                      value.runtime.accepted_state
                          .simulation_angular_velocity,
                      expected_angular_velocity),
              "MM_SIMROT_HL drives the exact real simulation spring update");
    }
}

struct ContactHistoryReference
{
    bool state = false;
    bool lock = false;
    vec3 position;
    vec3 velocity;
    vec3 point;
    vec3 target;
    vec3 offset_position;
    vec3 offset_velocity;
};

static ContactHistoryReference contact_history_reference(
    const g1_controller_state& state, int foot)
{
    ContactHistoryReference output;
    output.state = state.contact_states(foot);
    output.lock = state.contact_locks(foot);
    output.position = state.contact_positions(foot);
    output.velocity = state.contact_velocities(foot);
    output.point = state.contact_points(foot);
    output.target = state.contact_targets(foot);
    output.offset_position = state.contact_offset_positions(foot);
    output.offset_velocity = state.contact_offset_velocities(foot);
    return output;
}

static float contact_history_reference_fast_negexp(float x)
{
    return 1.0f /
        (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);
}

static void contact_history_reference_decay(
    vec3& value,
    vec3& velocity,
    float halflife,
    float dt)
{
    const float damping = (4.0f * LN2f) / (halflife + 1.0e-5f);
    const float y = damping / 2.0f;
    const vec3 j1 = velocity + value * y;
    const float eydt = contact_history_reference_fast_negexp(y * dt);
    value = eydt * (value + j1 * dt);
    velocity = eydt * (velocity - j1 * y * dt);
}

static void contact_history_reference_transition(
    vec3& offset,
    vec3& offset_velocity,
    vec3 source,
    vec3 source_velocity,
    vec3 destination,
    vec3 destination_velocity)
{
    offset = source + offset - destination;
    offset_velocity = source_velocity + offset_velocity -
        destination_velocity;
}

__attribute__((always_inline)) static inline void
contact_history_reference_advance(
    ContactHistoryReference& value,
    vec3 input_position,
    bool input_state,
    float unlock_radius,
    float foot_height,
    float blending_halflife,
    float dt)
{
#if defined(__FAST_MATH__)
    // The production fast build contracts scalar vector division to one
    // reciprocal followed by three products. Spell that permitted
    // reassociation explicitly so the independent oracle remains bit-exact
    // across translation units instead of depending on inliner choices.
    const float reciprocal_dt = 1.0f / (dt + 1.0e-8f);
    const vec3 input_velocity =
        (input_position - value.target) * reciprocal_dt;
#else
    const vec3 input_velocity =
        (input_position - value.target) / (dt + 1.0e-8f);
#endif
    value.target = input_position;
    contact_history_reference_decay(
        value.offset_position,
        value.offset_velocity,
        blending_halflife,
        dt);
    value.position = (value.lock ? value.point : input_position) +
        value.offset_position;
    value.velocity = (value.lock ? vec3() : input_velocity) +
        value.offset_velocity;
    const bool unlock = value.lock &&
        length(value.point - input_position) > unlock_radius;
    if (!value.state && input_state) {
        value.lock = true;
        value.point = value.position;
        value.point.y = foot_height;
        contact_history_reference_transition(
            value.offset_position,
            value.offset_velocity,
            input_position,
            input_velocity,
            value.point,
            vec3());
    } else if ((value.lock && value.state && !input_state) || unlock) {
        value.lock = false;
        contact_history_reference_transition(
            value.offset_position,
            value.offset_velocity,
            value.point,
            vec3(),
            input_position,
            input_velocity);
    }
    value.state = input_state;
}

static bool same_contact_history_bits(
    const ContactHistoryReference& expected,
    const g1_controller_state& actual,
    int foot)
{
    return expected.state == actual.contact_states(foot) &&
           expected.lock == actual.contact_locks(foot) &&
           same_vec3_bits(expected.position, actual.contact_positions(foot)) &&
           same_vec3_bits(expected.velocity, actual.contact_velocities(foot)) &&
           same_vec3_bits(expected.point, actual.contact_points(foot)) &&
           same_vec3_bits(expected.target, actual.contact_targets(foot)) &&
           same_vec3_bits(
               expected.offset_position,
               actual.contact_offset_positions(foot)) &&
           same_vec3_bits(
               expected.offset_velocity,
               actual.contact_offset_velocities(foot));
}

struct ContactTuningOutcome
{
    float rising_point_y = 0.0f;
    bool lock_after_far_target = false;
    vec3 offset_after_decay;
};

static G1FrameStageOutcome run_real_stage_prefix(
    fixture& value,
    G1FrameTransactionScratch& scratch,
    G1FrameTransactionStage last_stage,
    G1FrameTransactionStage& terminal_stage,
    char* error,
    int error_capacity);

static ContactTuningOutcome run_contact_tuning_sequence(
    float unlock_radius,
    float foot_height,
    float blending_halflife,
    bool axis_aligned_boundary = false)
{
    fixture value;
    for (int frame = 0; frame < value.db.nframes(); ++frame) {
        const bool contact = frame == 1 || frame == 2;
        value.db.contact_states(frame, 0) = contact;
        value.db.contact_states(frame, 1) = false;
        if (frame >= 2) {
            value.db.bone_positions(frame, G1_LeftToe).x += 0.375f;
        }
    }
    if (axis_aligned_boundary) {
        value.scene.metadata.spawn_yaw = 0.0f;
    }
    configure_production_mode(value, G1_TestSequential);
    value.external.tuning.contact_unlock_radius = unlock_radius;
    value.external.tuning.contact_foot_height = foot_height;
    value.external.tuning.contact_blending_halflife = blending_halflife;
    ContactHistoryReference reference[2] = {
        contact_history_reference(value.runtime.accepted_state, 0),
        contact_history_reference(value.runtime.accepted_state, 1),
    };
    ContactTuningOutcome outcome;
    char error[1024] = {};
    for (int frame = 0; frame < 4; ++frame) {
        value.external.input.presentation_frame = 300 + frame;
        const ConstArtifactEvidence artifacts_before =
            const_artifact_evidence(value.external);
        G1FrameTransactionScratch scratch;
        G1FrameTransactionStage terminal = G1FrameStageCount;
        check(run_real_stage_prefix(
                  value,
                  scratch,
                  G1FrameStageFootprintObservation,
                  terminal,
                  error,
                  static_cast<int>(sizeof(error))) ==
                      G1FrameStageContinue &&
                  terminal == G1FrameStageFootprintObservation,
              error[0] != '\0'
                  ? error
                  : "contact-tuning oracle completes the authenticated real "
                    "common candidate prefix");
        check(g1_controller_state_is_valid(value.runtime.working_state) &&
                  g1_controller_state_copy(
                      value.runtime.accepted_state,
                      value.runtime.working_state,
                      error,
                      static_cast<int>(sizeof(error))),
              error[0] != '\0'
                  ? error
                  : "contact-tuning oracle commits a valid common candidate "
                    "projection for the next history step");
        check_const_artifacts(
            value.external,
            artifacts_before,
            "each contact-tuning transaction preserves its specialized immutable database and artifacts");
        for (int foot = 0; foot < 2; ++foot) {
            const int contact_bone =
                value.runtime.accepted_state.contact_bones(foot);
            check(contact_bone >= 0 && contact_bone < G1_BoneCount,
                  "contact reference uses an authenticated configured contact bone");
            const vec3 independent_input_position =
                value.runtime.accepted_state
                    .global_bone_positions(contact_bone);
            contact_history_reference_advance(
                reference[foot],
                independent_input_position,
                value.runtime.accepted_state.curr_bone_contacts(foot),
                unlock_radius,
                foot_height,
                blending_halflife,
                value.external.tuning.dt);
            check(same_contact_history_bits(
                      reference[foot],
                      value.runtime.accepted_state,
                      foot),
                  "contact unlock radius, foot height, and blend half-life drive the exact independent history update");
        }
        if (frame == 0) {
            outcome.rising_point_y =
                value.runtime.accepted_state.contact_points(0).y;
        } else if (frame == 1) {
            outcome.lock_after_far_target =
                value.runtime.accepted_state.contact_locks(0);
        } else if (frame == 3) {
            outcome.offset_after_decay =
                value.runtime.accepted_state
                    .contact_offset_positions(0);
        }
    }
    return outcome;
}

static void test_contact_tuning_fields_drive_real_history()
{
    const float unlock_just_below = std::nextafter(0.375f, 0.0f);
    const float unlock_just_above = std::nextafter(
        0.375f, std::numeric_limits<float>::infinity());
    fixture boundary_baseline;
    boundary_baseline.scene.metadata.spawn_yaw = 0.0f;
    configure_production_mode(boundary_baseline, G1_TestSequential);
    const int boundary_contact_bone =
        boundary_baseline.runtime.accepted_state.contact_bones(0);
    const float boundary_foot_height = boundary_baseline.runtime
        .accepted_state.global_bone_positions(boundary_contact_bone).y;
    const ContactTuningOutcome just_below_unlock =
        run_contact_tuning_sequence(
            unlock_just_below, boundary_foot_height, 0.0625f, true);
    const ContactTuningOutcome just_above_unlock =
        run_contact_tuning_sequence(
            unlock_just_above, boundary_foot_height, 0.0625f, true);
    check(!just_below_unlock.lock_after_far_target &&
              just_above_unlock.lock_after_far_target,
          "the exact 0.375f displacement unlock boundary consumes the "
          "unscaled contact radius bits");

    const ContactTuningOutcome low_unlock =
        run_contact_tuning_sequence(0.05f, 0.015625f, 0.0625f);
    const ContactTuningOutcome high_unlock =
        run_contact_tuning_sequence(0.75f, 0.015625f, 0.0625f);
    check(!low_unlock.lock_after_far_target &&
              high_unlock.lock_after_far_target,
          "contact_unlock_radius independently controls real lock release");

    const ContactTuningOutcome low_foot =
        run_contact_tuning_sequence(0.75f, 0.015625f, 0.0625f);
    const ContactTuningOutcome high_foot =
        run_contact_tuning_sequence(0.75f, 0.03125f, 0.0625f);
    check(same_float_bits(low_foot.rising_point_y, 0.015625f) &&
              same_float_bits(high_foot.rising_point_y, 0.03125f),
          "contact_foot_height independently fixes the real rising lock height");

    const ContactTuningOutcome fast_blend =
        run_contact_tuning_sequence(0.05f, 0.015625f, 0.03125f);
    const ContactTuningOutcome slow_blend =
        run_contact_tuning_sequence(0.05f, 0.015625f, 0.25f);
    check(!same_vec3_bits(
              fast_blend.offset_after_decay,
              slow_blend.offset_after_decay),
          "contact_blending_halflife independently changes real history decay");
}

static G1FrameStageOutcome run_real_stage_prefix(
    fixture& value,
    G1FrameTransactionScratch& scratch,
    G1FrameTransactionStage last_stage,
    G1FrameTransactionStage& terminal_stage,
    char* error,
    int error_capacity)
{
    if (!g1_controller_state_copy(
            value.runtime.working_state,
            value.runtime.accepted_state,
            error,
            error_capacity)) {
        terminal_stage = G1FrameStageCount;
        return G1FrameStageGlobalError;
    }
    for (int stage = 0; stage <= static_cast<int>(last_stage); ++stage) {
        terminal_stage = static_cast<G1FrameTransactionStage>(stage);
        if (terminal_stage == G1FrameStageCandidateApply) {
            scratch.active_candidate = scratch.slot_zero_record;
        } else if (terminal_stage == G1FrameStageRawBegin) {
            scratch.rejection_branch = G1FrameCertificateRaw;
        } else if (terminal_stage == G1FrameStageIkBegin) {
            scratch.rejection_branch = G1FrameCertificateIk;
        }
        const G1FrameStageOutcome outcome =
            g1_controller_frame_stage_run(
                terminal_stage,
                value.runtime.working_state,
                scratch,
                value.external,
                error,
                error_capacity);
        if (outcome != G1FrameStageContinue) return outcome;
    }
    terminal_stage = last_stage;
    return G1FrameStageContinue;
}

static void check_real_rejection_snapshot_table(
    const G1IkFrameTransaction& transaction,
    G1IkRejectionCheckpoint expected)
{
    char error[1024] = {};
    for (int checkpoint = G1IkRejectionAfterBegin;
         checkpoint <= G1IkRejectionAfterFoot1;
         ++checkpoint) {
        G1IkFrameResult output;
        output.max_correction_radians = 0.125f;
        output.stop_reason = G1IkStopPoseClearanceRejected;
        const uint64_t before = ik_frame_logical_digest(output);
        const bool accepted = g1_ik_frame_rejection_snapshot(
            output,
            transaction,
            static_cast<G1IkRejectionCheckpoint>(checkpoint),
            error,
            static_cast<int>(sizeof(error)));
        if (checkpoint == static_cast<int>(expected)) {
            check(accepted &&
                      ik_frame_logical_digest(output) ==
                          ik_frame_logical_digest(
                              transaction.candidate_result),
                  "the exact real IK checkpoint snapshot succeeds");
        } else {
            check(!accepted && ik_frame_logical_digest(output) == before,
                  "both wrong real IK checkpoints fail without output mutation");
        }
    }
}

static void reset_ik_enabled_fixture(fixture& value, g1_test_mode mode)
{
    configure_production_mode(value, mode);
    value.external.tuning.ik_enabled = true;
    G1FrameResetConfig config;
    config.route_mode = mode == G1_TestRoute;
    config.route_id = mode == G1_TestRoute
        ? value.scene.metadata.routes[0].id.c_str()
        : NULL;
    config.ik_enabled = true;
    config.dt = value.external.tuning.dt;
    config.trajectory_sample_time =
        value.external.tuning.trajectory_sample_time;
    config.initial_search_time = value.external.tuning.initial_search_time;
    char error[1024] = {};
    check(g1_frame_runtime_reset(
              value.runtime,
              value.db,
              value.support,
              value.scene,
              config,
              error,
              static_cast<int>(sizeof(error))),
          error);
    value.external.db = &value.db;
    value.external.support = &value.support;
    value.external.scene = &value.scene;
    value.external.route = mode == G1_TestRoute
        ? &value.scene.metadata.routes[0]
        : NULL;
}

static void force_swing_history_below_terrain(
    g1_controller_state& state, int foot)
{
    state.ik.feet[foot].swing.initialized = true;
    for (int probe = 0; probe < 4; ++probe) {
        state.ik.feet[foot].swing.previous_sphere_centers[probe] =
            vec3(
                state.simulation_position.x,
                -0.75f - 0.01f * static_cast<float>(probe),
                state.simulation_position.z);
    }
}

static void check_real_coordinator_ik_rejection(
    fixture& coordinator_value,
    const fixture& direct_value,
    const ProductionEvidence& direct_before,
    const ConstArtifactEvidence& direct_artifacts_before,
    G1FrameTransactionStage terminal_stage,
    const G1FrameRejectionDiagnostic& direct_rejection)
{
    check(direct_rejection.rejected &&
              direct_rejection.attempted_footprint_available &&
              direct_rejection.attempted_ik_available,
          "direct real prefix provides complete typed rejection evidence");
    check(state_logical_digest(direct_value.runtime.accepted_state) ==
                  direct_before.accepted &&
              same_storage_identities(
                  state_storage_identities(
                      direct_value.runtime.accepted_state),
                  direct_before.accepted_storage) &&
              publication_logical_digest(
                  direct_value.runtime.publication) ==
                  direct_before.publication &&
              diagnostic_logical_digest(
                  direct_value.runtime.accepted_diagnostic) ==
                  direct_before.diagnostic,
          "direct-prefix fixture preserves its pristine accepted/publication baseline");
    check_const_artifacts(
        direct_value.external,
        direct_artifacts_before,
        "direct-prefix fixture preserves its independent artifact baseline");
    const ProductionEvidence before =
        production_evidence(coordinator_value.runtime);
    const G1FramePublication publication_before =
        coordinator_value.runtime.publication;
    const ConstArtifactEvidence artifacts_before =
        const_artifact_evidence(coordinator_value.external);
    G1FrameTransactionTestSeam seam;
    seam.hook = production_hook;
    char error[1024] = {};
    reset_production_trace(
        coordinator_value.external,
        coordinator_value.runtime.accepted_state.camera_azimuth);
    reset_production_runner_proxy_trace();
    check(g1_frame_transaction_run(
              coordinator_value.runtime,
              recording_production_runner,
              coordinator_value.external,
              &seam,
              error,
              static_cast<int>(sizeof(error))) ==
              G1FrameTransactionFiniteRejected,
          error);
    check(production_runner_proxy_trace.total ==
                  static_cast<int>(terminal_stage) + 1 &&
              production_runner_proxy_trace.terminal_ready &&
              production_runner_proxy_trace.terminal_stage ==
                  terminal_stage &&
              production_runner_proxy_trace.terminal_outcome ==
                  G1FrameStageFiniteReject,
          "recording proxy includes the genuine terminal finite-reject stage and outcome");
    for (int stage = 0; stage < G1FrameStageCount; ++stage) {
        const bool entered = stage <= static_cast<int>(terminal_stage);
        check(production_runner_proxy_trace.calls[stage] ==
                      (entered ? 1 : 0),
              "recording proxy enters each genuine prefix stage exactly once");
        if (entered) {
            check(production_runner_proxy_trace.order[stage] ==
                          static_cast<G1FrameTransactionStage>(stage) &&
                      production_runner_proxy_trace.outcomes[stage] ==
                          (stage == static_cast<int>(terminal_stage)
                               ? G1FrameStageFiniteReject
                               : G1FrameStageContinue),
                  "recording proxy records every forwarded stage outcome in order");
        }
    }
    check_production_trace_through(
        static_cast<int>(terminal_stage) - 1,
        "the hook trace ends immediately before the real rejecting stage");
    for (int stage = static_cast<int>(terminal_stage);
         stage < G1FrameStageCount;
         ++stage) {
        check(production_trace.calls[stage] == 0,
              "the real rejecting stage and every later stage stop before the hook");
    }
    const G1FrameRejectionDiagnostic& published_rejection =
        coordinator_value.runtime.publication.rejection;
    G1FramePublication expected_publication = publication_before;
    expected_publication.requested_intent =
        production_trace.requested_intent;
    expected_publication.rejection = direct_rejection;
    expected_publication.ik_safe_stop_latched = true;
    expected_publication.presentation_frame =
        coordinator_value.external.input.presentation_frame;
    check(state_logical_digest(coordinator_value.runtime.accepted_state) ==
              before.accepted &&
              same_storage_identities(
                  state_storage_identities(
                      coordinator_value.runtime.accepted_state),
                  before.accepted_storage) &&
              diagnostic_logical_digest(
                  coordinator_value.runtime.accepted_diagnostic) ==
                  before.diagnostic &&
              rejection_logical_digest(published_rejection) ==
                  rejection_logical_digest(direct_rejection) &&
              footprint_logical_digest(
                  published_rejection.attempted_footprint) ==
                  footprint_logical_digest(
                      direct_rejection.attempted_footprint) &&
              ik_frame_logical_digest(published_rejection.ik_frame) ==
                  ik_frame_logical_digest(direct_rejection.ik_frame) &&
              publication_logical_digest(
                  coordinator_value.runtime.publication) ==
                  publication_logical_digest(expected_publication),
          "genuine production IK stop publishes the exact direct-prefix rejection payload and whitelist only");
    check_const_artifacts(
        coordinator_value.external,
        artifacts_before,
        "genuine production IK rejection preserves immutable artifacts");
}

static void check_real_ik_accepting_control(
    fixture& direct_value,
    fixture& coordinator_value,
    G1FrameTransactionStage last_stage,
    uint32_t expected_next_foot,
    const char* message)
{
    const ProductionEvidence direct_before =
        production_evidence(direct_value.runtime);
    const ConstArtifactEvidence direct_artifacts_before =
        const_artifact_evidence(direct_value.external);
    G1FrameTransactionScratch scratch;
    G1FrameTransactionStage terminal = G1FrameStageCount;
    char error[1024] = {};
    check(run_real_stage_prefix(
              direct_value,
              scratch,
              last_stage,
              terminal,
              error,
              static_cast<int>(sizeof(error))) ==
              G1FrameStageContinue &&
              terminal == last_stage &&
              scratch.ik_certificate.ik_transaction.initialized &&
              scratch.ik_certificate.ik_transaction.next_foot ==
                  expected_next_foot &&
              !scratch.ik_certificate.ik_transaction.candidate_result
                   .safe_stop_requested,
          message);

    check(state_logical_digest(direct_value.runtime.accepted_state) ==
                  direct_before.accepted &&
              publication_logical_digest(
                  direct_value.runtime.publication) ==
                  direct_before.publication &&
              diagnostic_logical_digest(
                  direct_value.runtime.accepted_diagnostic) ==
                  direct_before.diagnostic,
          "accepting direct-prefix fixture preserves its pristine accepted baseline");
    check_const_artifacts(
        direct_value.external,
        direct_artifacts_before,
        "accepting direct-prefix fixture preserves its independent artifacts");

    const ConstArtifactEvidence artifacts_before =
        const_artifact_evidence(coordinator_value.external);
    G1FrameTransactionTestSeam seam;
    seam.hook = production_hook;
    reset_production_trace(
        coordinator_value.external,
        coordinator_value.runtime.accepted_state.camera_azimuth);
    check(g1_frame_transaction_run(
              coordinator_value.runtime,
              g1_controller_frame_stage_run,
              coordinator_value.external,
              &seam,
              error,
              static_cast<int>(sizeof(error))) ==
              G1FrameTransactionAccepted &&
              !coordinator_value.runtime.publication.rejection.rejected &&
              !coordinator_value.runtime.publication.ik_safe_stop_latched &&
              coordinator_value.runtime.accepted_diagnostic.ready,
          message);
    check_production_trace_through(
        G1FrameStageAcceptedFinalize, message);
    check_complete_success_publication(
        coordinator_value.runtime, coordinator_value.external, message);
    check_complete_accepted_final_fk(coordinator_value, message);
    check_const_artifacts(
        coordinator_value.external, artifacts_before, message);
}

static void test_genuine_foot0_and_foot1_safe_stops()
{
    {
        fixture control_direct;
        fixture control_coordinator;
        for (int frame = 0;
             frame < control_direct.db.nframes();
             ++frame) {
            control_direct.db.contact_states(frame, 0) = false;
            control_direct.db.contact_states(frame, 1) = false;
            control_coordinator.db.contact_states(frame, 0) = false;
            control_coordinator.db.contact_states(frame, 1) = false;
        }
        reset_ik_enabled_fixture(control_direct, G1_TestSequential);
        reset_ik_enabled_fixture(
            control_coordinator, G1_TestSequential);
        check_real_ik_accepting_control(
            control_direct,
            control_coordinator,
            G1FrameStageIkFirstFoot,
            1,
            "the real first-foot stage and coordinator accept when the "
            "hostile foot-0 swing history is absent");

        fixture direct;
        fixture coordinator;
        for (int frame = 0; frame < direct.db.nframes(); ++frame) {
            direct.db.contact_states(frame, 0) = false;
            direct.db.contact_states(frame, 1) = false;
            coordinator.db.contact_states(frame, 0) = false;
            coordinator.db.contact_states(frame, 1) = false;
        }
        reset_ik_enabled_fixture(direct, G1_TestSequential);
        reset_ik_enabled_fixture(coordinator, G1_TestSequential);
        force_swing_history_below_terrain(
            direct.runtime.accepted_state, 0);
        force_swing_history_below_terrain(
            coordinator.runtime.accepted_state, 0);
        check(g1_controller_state_is_valid(direct.runtime.accepted_state) &&
                  g1_controller_state_is_valid(
                      coordinator.runtime.accepted_state),
              "foot-0 real stop source state is valid");
        const ProductionEvidence direct_before =
            production_evidence(direct.runtime);
        const ConstArtifactEvidence direct_artifacts_before =
            const_artifact_evidence(direct.external);
        G1FrameTransactionScratch scratch;
        G1FrameTransactionStage terminal = G1FrameStageCount;
        char error[1024] = {};
        check(run_real_stage_prefix(
                  direct,
                  scratch,
                  G1FrameStageIkFirstFoot,
                  terminal,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameStageFiniteReject &&
                  terminal == G1FrameStageIkFirstFoot &&
                  scratch.ik_certificate.ik_transaction.next_foot == 1 &&
                  scratch.ik_certificate.ik_transaction.candidate_result
                      .safe_stop_requested &&
                  scratch.ik_certificate.ik_transaction
                          .candidate_result.stop_reason ==
                      G1IkStopNoSwingCandidate &&
                  g1_root_reach_plan_is_valid(
                      scratch.ik_certificate.ik_transaction.candidate_result
                          .root_reach) &&
                  !scratch.ik_certificate.ik_transaction.candidate_result
                       .root_reach.active &&
                  scratch.ik_certificate.ik_transaction
                      .candidate_result.feet[1]
                      .swing_selection.candidates_evaluated == 0 &&
                  !scratch.ik_certificate.ik_transaction
                       .candidate_result.feet[1]
                       .position.applied,
              "real foot 0 exhausts candidates, advances to cursor 1, and stops before foot 1");
        check_real_rejection_snapshot_table(
            scratch.ik_certificate.ik_transaction,
            G1IkRejectionAfterFoot0);
        check(scratch.rejection.stage == G1FrameRejectIkCandidate,
              "real foot-0 finite diagnostic names the first-foot stage");
        check_real_coordinator_ik_rejection(
            coordinator,
            direct,
            direct_before,
            direct_artifacts_before,
            G1FrameStageIkFirstFoot,
            scratch.rejection);
    }

    {
        fixture control_direct;
        fixture control_coordinator;
        for (int frame = 0;
             frame < control_direct.db.nframes();
             ++frame) {
            control_direct.db.contact_states(frame, 0) = false;
            control_direct.db.contact_states(frame, 1) = false;
            control_coordinator.db.contact_states(frame, 0) = false;
            control_coordinator.db.contact_states(frame, 1) = false;
        }
        reset_ik_enabled_fixture(control_direct, G1_TestSequential);
        reset_ik_enabled_fixture(
            control_coordinator, G1_TestSequential);
        check_real_ik_accepting_control(
            control_direct,
            control_coordinator,
            G1FrameStageIkSecondFoot,
            2,
            "the real second-foot stage and coordinator accept when the "
            "hostile foot-1 swing history is absent");

        fixture direct;
        fixture coordinator;
        for (int frame = 0; frame < direct.db.nframes(); ++frame) {
            direct.db.contact_states(frame, 0) = false;
            direct.db.contact_states(frame, 1) = false;
            coordinator.db.contact_states(frame, 0) = false;
            coordinator.db.contact_states(frame, 1) = false;
        }
        reset_ik_enabled_fixture(direct, G1_TestSequential);
        reset_ik_enabled_fixture(coordinator, G1_TestSequential);
        force_swing_history_below_terrain(
            direct.runtime.accepted_state, 1);
        force_swing_history_below_terrain(
            coordinator.runtime.accepted_state, 1);
        check(g1_controller_state_is_valid(direct.runtime.accepted_state) &&
                  g1_controller_state_is_valid(
                      coordinator.runtime.accepted_state),
              "foot-1 real stop source state is valid");
        const ProductionEvidence direct_before =
            production_evidence(direct.runtime);
        const ConstArtifactEvidence direct_artifacts_before =
            const_artifact_evidence(direct.external);
        G1FrameTransactionScratch scratch;
        G1FrameTransactionStage terminal = G1FrameStageCount;
        char error[1024] = {};
        check(run_real_stage_prefix(
                  direct,
                  scratch,
                  G1FrameStageIkSecondFoot,
                  terminal,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameStageFiniteReject &&
                  terminal == G1FrameStageIkSecondFoot &&
                  scratch.ik_certificate.ik_transaction.next_foot == 2 &&
                  !scratch.ik_certificate.ik_transaction.candidate_result
                       .feet[0].position.safe_stop_requested &&
                  scratch.ik_certificate.ik_transaction.candidate_result
                      .safe_stop_requested &&
                  scratch.ik_certificate.ik_transaction
                          .candidate_result.stop_reason ==
                      G1IkStopNoSwingCandidate &&
                  g1_root_reach_plan_is_valid(
                      scratch.ik_certificate.ik_transaction.candidate_result
                          .root_reach) &&
                  !scratch.ik_certificate.ik_transaction.candidate_result
                       .root_reach.active &&
                  !scratch.ik_certificate.ik_transaction
                       .candidate_result.applied,
              "real foot 0 completes before foot 1 exhausts candidates at cursor 2");
        check_real_rejection_snapshot_table(
            scratch.ik_certificate.ik_transaction,
            G1IkRejectionAfterFoot1);
        check(scratch.rejection.stage == G1FrameRejectIkCandidate,
              "real foot-1 finite diagnostic names the second-foot stage");
        check_real_coordinator_ik_rejection(
            coordinator,
            direct,
            direct_before,
            direct_artifacts_before,
            G1FrameStageIkSecondFoot,
            scratch.rejection);
    }
}

static void reset_begin_stop_runtime(fixture& value)
{
    G1FrameResetConfig config;
    config.route_mode = true;
    config.route_id = value.scene.metadata.routes[0].id.c_str();
    config.ik_enabled = true;
    config.dt = value.external.tuning.dt;
    config.trajectory_sample_time =
        value.external.tuning.trajectory_sample_time;
    config.initial_search_time = value.external.tuning.initial_search_time;
    char error[1024] = {};
    check(g1_frame_runtime_reset(
              value.runtime,
              value.db,
              value.support,
              value.scene,
              config,
              error,
              static_cast<int>(sizeof(error))),
          error);
    value.external.db = &value.db;
    value.external.support = &value.support;
    value.external.scene = &value.scene;
    value.external.route = &value.scene.metadata.routes[0];
}

static void align_begin_fixture_source_support(fixture& value)
{
    array1d<vec3> source_global_positions(G1_BoneCount);
    array1d<quat> source_global_rotations(G1_BoneCount);
    vec3 left_sole;
    vec3 right_sole;
    char error[1024] = {};
    check(g1_ik_checked_forward_kinematics(
              source_global_positions,
              source_global_rotations,
              value.db.bone_positions(0),
              value.db.bone_rotations(0),
              value.db.bone_parents,
              error,
              static_cast<int>(sizeof(error))) &&
              g1_ik_checked_physical_sole_centroid(
                  left_sole,
                  source_global_positions(G1_LeftToe),
                  source_global_rotations(G1_LeftToe),
                  g1_left_leg_config()) &&
              g1_ik_checked_physical_sole_centroid(
                  right_sole,
                  source_global_positions(G1_RightToe),
                  source_global_rotations(G1_RightToe),
                  g1_right_leg_config()) &&
              same_float_bits(left_sole.y, right_sole.y),
          error[0] == '\0'
              ? "the accepting-control source pose has symmetric physical soles"
              : error);
    const float source_surface_height =
        left_sole.y - g1_left_leg_config().planted_clearance_m;
    check(terrain_float_is_normal_or_zero_query(source_surface_height),
          "the accepting-control source support height is representable");
    value.support.values.set(source_surface_height);
}

static void shift_aligned_support_to_first_common_word(
    fixture& value)
{
    const float aligned_support = value.support.values(0, 0);
    check(terrain_float_bits(aligned_support) ==
              UINT32_C(0xbf08efbb),
          "common-plan fixture freezes its aligned support word");
    const float shifted_support = std::nextafter(
        aligned_support,
        std::numeric_limits<float>::infinity());
    check(terrain_float_bits(shifted_support) ==
              UINT32_C(0xbf08efba),
          "common-plan fixture uses the first accepting positive ULP");
    value.support.values.set(shifted_support);
}

static void configure_quantized_all_contact_fixture(
    fixture& value, bool planner_common)
{
    for (int frame = 0; frame < value.db.nframes(); ++frame) {
        value.db.contact_states(frame, 0) = true;
        value.db.contact_states(frame, 1) = true;
    }
    align_begin_fixture_source_support(value);
    if (planner_common) {
        shift_aligned_support_to_first_common_word(value);
    } else {
        check(terrain_float_bits(value.support.values(0, 0)) ==
                  UINT32_C(0xbf08efbb),
              "quantized terminal fixture retains the unshifted support word");
    }
    reset_ik_enabled_fixture(value, G1_TestFlat);
}

static void lift_begin_accepting_control_airborne_foot(fixture& value)
{
    for (int frame = 0; frame < value.db.nframes(); ++frame) {
        value.db.bone_positions(frame, G1_LeftHipYaw).y += 0.02f;
    }
}

static void configure_begin_stop_variant(fixture& value, int variant)
{
    configure_production_mode(value, G1_TestRoute);
    for (int frame = 0; frame < value.db.nframes(); ++frame) {
        value.db.contact_states(frame, 0) = false;
        value.db.contact_states(frame, 1) = true;
    }
    for (int current = 0; current <= 2; ++current) {
        value.db.contact_states(current, 0) = false;
        value.db.contact_states(current + 9, 0) = false;
        value.db.contact_states(current + 17, 0) = true;
        value.db.contact_states(current + 25, 0) = true;
    }
    const float drop = -0.08f - 0.04f * static_cast<float>(variant);
    for (int z = 0; z < value.scene.terrain.nz; ++z) {
        for (int x = 0; x < value.scene.terrain.nx; ++x) {
            const int checker = (x + z + variant) & 1;
            const float world_x = value.scene.terrain.origin_x +
                value.scene.terrain.cell_size * static_cast<float>(x);
            value.scene.terrain.heights(
                z * value.scene.terrain.nx + x) =
                world_x >= 2.0f && checker != 0 ? drop : 0.0f;
        }
    }
    value.external.tuning.trajectory_sample_time = 1.0f / 3.0f;
    value.external.tuning.ik_enabled = true;
    align_begin_fixture_source_support(value);
    reset_begin_stop_runtime(value);
}

static void test_genuine_begin_time_safe_stop()
{
    bool found = false;
    for (int variant = 0; variant < 8 && !found; ++variant) {
        fixture direct;
        fixture coordinator;
        configure_begin_stop_variant(direct, variant);
        configure_begin_stop_variant(coordinator, variant);
        const ProductionEvidence direct_before =
            production_evidence(direct.runtime);
        const ConstArtifactEvidence direct_artifacts_before =
            const_artifact_evidence(direct.external);
        G1FrameTransactionScratch scratch;
        G1FrameTransactionStage terminal = G1FrameStageCount;
        char error[1024] = {};
        const G1FrameStageOutcome outcome = run_real_stage_prefix(
            direct,
            scratch,
            G1FrameStageIkBegin,
            terminal,
            error,
            static_cast<int>(sizeof(error)));
        if (outcome != G1FrameStageFiniteReject ||
            terminal != G1FrameStageIkBegin ||
            !scratch.ik_certificate.ik_transaction.initialized ||
            scratch.ik_certificate.ik_transaction.next_foot != 0 ||
            scratch.ik_certificate.ik_transaction
                    .candidate_result.stop_reason !=
                G1IkStopLandingPatchUnavailable) {
            continue;
        }
        found = true;
        {
            fixture control_direct;
            fixture control_coordinator;
            configure_begin_stop_variant(control_direct, variant);
            configure_begin_stop_variant(
                control_coordinator, variant);
            control_direct.scene.terrain.heights.zero();
            control_coordinator.scene.terrain.heights.zero();
            lift_begin_accepting_control_airborne_foot(
                control_direct);
            lift_begin_accepting_control_airborne_foot(
                control_coordinator);
            shift_aligned_support_to_first_common_word(
                control_direct);
            shift_aligned_support_to_first_common_word(
                control_coordinator);
            reset_begin_stop_runtime(control_direct);
            reset_begin_stop_runtime(control_coordinator);
            check_real_ik_accepting_control(
                control_direct,
                control_coordinator,
                G1FrameStageIkBegin,
                0,
                "the real begin stage and coordinator accept when only the "
                "hostile unavailable-patch terrain is removed");
            const g1_controller_state& accepted =
                control_coordinator.runtime.accepted_state;
            vec3 accepted_right_sole;
            check(g1_ik_checked_physical_sole_centroid(
                      accepted_right_sole,
                      accepted.global_bone_positions(G1_RightToe),
                      accepted.global_bone_rotations(G1_RightToe),
                      g1_right_leg_config()) &&
                      terrain_float_bits(
                          control_coordinator.support.values(0, 0)) ==
                          UINT32_C(0xbf08efba) &&
                      terrain_float_bits(
                          accepted.adjusted_bone_positions(
                              G1_Simulation).y) ==
                          UINT32_C(0x3f08efba) &&
                      terrain_float_bits(accepted_right_sole.y) ==
                          UINT32_C(0x3ba3d688) &&
                      accepted.ik_frame.root_reach.active &&
                      accepted.ik_frame.root_reach
                          .common_interval_found &&
                      !accepted.ik_frame.root_reach.applied &&
                      terrain_float_bits(
                          accepted.ik_frame.root_reach
                              .root_y_delta_m) == 0U,
                  "terrain-removed control freezes the first common support, root, sole, and plan words");
        }
        check(variant == 0 &&
                  terrain_float_bits(direct.support.values(0, 0)) ==
                      UINT32_C(0xbf08efbb) &&
                  scratch.ik_certificate.ik_transaction.candidate_result
                  .safe_stop_requested &&
                  g1_root_reach_plan_is_valid(
                      scratch.ik_certificate.ik_transaction.candidate_result
                          .root_reach) &&
                  !scratch.ik_certificate.ik_transaction.candidate_result
                       .root_reach.active &&
                  scratch.ik_certificate.ik_transaction
                      .candidate_result.feet[0]
                      .swing_selection.candidates_evaluated == 0 &&
                  scratch.ik_certificate.ik_transaction
                      .candidate_result.feet[1]
                      .swing_selection.candidates_evaluated == 0 &&
                  scratch.rejection.stage == G1FrameRejectLandingPatch &&
                  scratch.rejection.attempted_footprint_available &&
                  scratch.rejection.attempted_ik_available,
              "successful real begin requests unavailable-patch stop before either foot");
        check_real_rejection_snapshot_table(
            scratch.ik_certificate.ik_transaction,
            G1IkRejectionAfterBegin);
        check_real_coordinator_ik_rejection(
            coordinator,
            direct,
            direct_before,
            direct_artifacts_before,
            G1FrameStageIkBegin,
            scratch.rejection);
    }
    check(found,
          "deterministic multilevel variants contain a genuine begin-time unavailable landing patch");
}

static uint64_t pose_clearance_logical_digest(
    const G1PoseClearance& value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    logical_hash_pose_clearance(hash, value);
    return hash;
}

static void test_real_pose_certificate_classification()
{
    {
        fixture value;
        configure_production_mode(value, G1_TestFlat);
        G1FrameTransactionScratch scratch;
        G1FrameTransactionStage terminal = G1FrameStageCount;
        char error[1024] = {};
        check(run_real_stage_prefix(
                  value,
                  scratch,
                  G1FrameStageIkFinalFk,
                  terminal,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameStageContinue,
              error);
        value.scene.terrain.origin_x = 100.0f;
        const G1FrameStageOutcome outcome =
            g1_controller_frame_stage_run(
                G1FrameStageIkPoseCertificate,
                value.runtime.working_state,
                scratch,
                value.external,
                error,
                static_cast<int>(sizeof(error)));
        check(outcome == G1FrameStageFiniteReject &&
                  scratch.rejection.rejected &&
                  scratch.rejection.stage ==
                      G1FrameRejectPoseCertificate &&
                  scratch.rejection.pose_status ==
                      G1ClearanceOutsideDomain &&
                  !scratch.rejection.attempted_pose_available &&
                  pose_clearance_logical_digest(
                      scratch.rejection.pose_clearance) ==
                      pose_clearance_logical_digest(G1PoseClearance{}),
              "real outside-domain pose certificate is finite with canonical unavailable pose");
    }

    {
        fixture value;
        configure_production_mode(value, G1_TestFlat);
        G1FrameTransactionScratch scratch;
        G1FrameTransactionStage terminal = G1FrameStageCount;
        char error[1024] = {};
        check(run_real_stage_prefix(
                  value,
                  scratch,
                  G1FrameStageIkFinalFk,
                  terminal,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameStageContinue,
              error);
        G1PoseClearance baseline;
        check(g1_measure_pose_clearance(
                  baseline,
                  g1_pose_clearance_budget(),
                  value.scene.terrain,
                  value.runtime.working_state.ik_global_bone_positions,
                  value.runtime.working_state.ik_global_bone_rotations,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1ClearanceOk,
              error);
        const double first_rejection_height = std::min(
            std::min(
                baseline.left.toe.lower_bound_m + 0.005,
                baseline.left.foot.lower_bound_m + 0.005),
            std::min(
                std::min(
                    baseline.right.toe.lower_bound_m + 0.005,
                    baseline.right.foot.lower_bound_m + 0.005),
                baseline.minimum.lower_bound_m + 0.01));
        const float raise = std::nextafter(
            static_cast<float>(first_rejection_height),
            std::numeric_limits<float>::infinity());
        for (int cell = 0; cell < value.scene.terrain.heights.size; ++cell) {
            value.scene.terrain.heights(cell) = raise;
        }
        const G1FrameStageOutcome outcome =
            g1_controller_frame_stage_run(
                G1FrameStageIkPoseCertificate,
                value.runtime.working_state,
                scratch,
                value.external,
                error,
                static_cast<int>(sizeof(error)));
        const G1PoseClearance& pose = scratch.rejection.pose_clearance;
        const bool threshold_failed =
            pose.left.toe.lower_bound_m < -0.005 ||
            pose.left.foot.lower_bound_m < -0.005 ||
            pose.right.toe.lower_bound_m < -0.005 ||
            pose.right.foot.lower_bound_m < -0.005 ||
            pose.minimum.lower_bound_m < -0.01;
        check(scratch.rejection.stage == G1FrameRejectPoseCertificate &&
                  outcome == G1FrameStageFiniteReject &&
                  scratch.rejection.pose_status == G1ClearanceOk &&
                  scratch.rejection.attempted_pose_available &&
                  threshold_failed &&
                  pose_clearance_logical_digest(pose) ==
                      pose_clearance_logical_digest(
                          scratch.ik_certificate.pose_clearance),
              "real Ok certificate rejected by controller thresholds publishes the complete certificate");
    }

    {
        fixture value;
        configure_production_mode(value, G1_TestFlat);
        G1FrameTransactionScratch scratch;
        G1FrameTransactionStage terminal = G1FrameStageCount;
        char error[1024] = {};
        check(run_real_stage_prefix(
                  value,
                  scratch,
                  G1FrameStageIkFinalFk,
                  terminal,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameStageContinue,
              error);
        for (int cell = 0; cell < value.scene.terrain.heights.size; ++cell) {
            value.scene.terrain.heights(cell) =
                std::numeric_limits<float>::quiet_NaN();
        }
        const uint64_t rejection_before = [&scratch]() {
            uint64_t hash = UINT64_C(1469598103934665603);
            logical_hash_rejection(hash, scratch.rejection);
            return hash;
        }();
        check(g1_controller_frame_stage_run(
                  G1FrameStageIkPoseCertificate,
                  value.runtime.working_state,
                  scratch,
                  value.external,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameStageGlobalError,
              "real invalid-field pose classification is a controlled global error");
        uint64_t rejection_after = UINT64_C(1469598103934665603);
        logical_hash_rejection(rejection_after, scratch.rejection);
        check(rejection_after == rejection_before,
              "global pose classification does not publish a finite diagnostic");
    }
}

static std::string read_source_file(const char* path)
{
    std::ifstream input(path, std::ios::binary);
    check(input.good(), "controller source is readable by production guard");
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

enum CppTokenKind
{
    CppTokenIdentifier,
    CppTokenNumber,
    CppTokenString,
    CppTokenCharacter,
    CppTokenPunctuation,
};

struct CppToken
{
    CppTokenKind kind = CppTokenPunctuation;
    std::string text;
    std::size_t begin = 0;
    std::size_t end = 0;
    int line = 1;
    bool directive = false;
};

static bool cpp_identifier_start(char value)
{
    return (value >= 'a' && value <= 'z') ||
           (value >= 'A' && value <= 'Z') || value == '_';
}

static bool cpp_identifier_continue(char value)
{
    return cpp_identifier_start(value) ||
           (value >= '0' && value <= '9');
}

static std::vector<CppToken> tokenize_cpp_source(
    const std::string& source,
    std::string& error)
{
    std::vector<CppToken> tokens;
    std::size_t position = 0;
    int line = 1;
    bool line_has_token = false;
    bool line_is_directive = false;
    const auto push = [&tokens, &source, &line, &line_is_directive](
                          CppTokenKind kind,
                          std::size_t begin,
                          std::size_t end) {
        CppToken token;
        token.kind = kind;
        token.text = source.substr(begin, end - begin);
        token.begin = begin;
        token.end = end;
        token.line = line;
        token.directive = line_is_directive;
        tokens.push_back(token);
    };
    while (position < source.size()) {
        const char value = source[position];
        const char next = position + 1 < source.size()
            ? source[position + 1]
            : '\0';
        if (value == '\n') {
            ++line;
            ++position;
            line_has_token = false;
            line_is_directive = false;
            continue;
        }
        if (value == ' ' || value == '\t' || value == '\r' ||
            value == '\f' || value == '\v') {
            ++position;
            continue;
        }
        if (value == '/' && next == '/') {
            position += 2;
            while (position < source.size() && source[position] != '\n') {
                ++position;
            }
            continue;
        }
        if (value == '/' && next == '*') {
            position += 2;
            bool closed = false;
            while (position + 1 < source.size()) {
                if (source[position] == '\n') {
                    ++line;
                    line_has_token = false;
                    line_is_directive = false;
                }
                if (source[position] == '*' &&
                    source[position + 1] == '/') {
                    position += 2;
                    closed = true;
                    break;
                }
                ++position;
            }
            if (!closed) {
                error = "unterminated block comment";
                return {};
            }
            continue;
        }
        if (!line_has_token && value == '#') {
            line_is_directive = true;
        }
        line_has_token = true;
        if (cpp_identifier_start(value) &&
            !(value == 'R' && next == '"')) {
            const std::size_t begin = position++;
            while (position < source.size() &&
                   cpp_identifier_continue(source[position])) {
                ++position;
            }
            push(CppTokenIdentifier, begin, position);
            continue;
        }
        if (value >= '0' && value <= '9') {
            const std::size_t begin = position++;
            while (position < source.size()) {
                const char current = source[position];
                if (cpp_identifier_continue(current) || current == '.' ||
                    current == '\'') {
                    ++position;
                } else if ((current == '+' || current == '-') &&
                           position > begin &&
                           (source[position - 1] == 'e' ||
                            source[position - 1] == 'E' ||
                            source[position - 1] == 'p' ||
                            source[position - 1] == 'P')) {
                    ++position;
                } else {
                    break;
                }
            }
            push(CppTokenNumber, begin, position);
            continue;
        }
        if (value == 'R' && next == '"') {
            const std::size_t begin = position;
            const std::size_t delimiter_begin = position + 2;
            const std::size_t opening =
                source.find('(', delimiter_begin);
            if (opening == std::string::npos ||
                opening - delimiter_begin > 16) {
                error = "invalid raw string delimiter";
                return {};
            }
            const std::string delimiter = source.substr(
                delimiter_begin, opening - delimiter_begin);
            if (delimiter.find_first_of(" \\)\t\r\n") !=
                std::string::npos) {
                error = "invalid raw string delimiter";
                return {};
            }
            const std::string terminator = ")" + delimiter + "\"";
            const std::size_t closing = source.find(
                terminator, opening + 1);
            if (closing == std::string::npos) {
                error = "unterminated raw string literal";
                return {};
            }
            for (std::size_t index = begin;
                 index < closing + terminator.size();
                 ++index) {
                if (source[index] == '\n') ++line;
            }
            position = closing + terminator.size();
            push(CppTokenString, begin, position);
            continue;
        }
        if (value == '"' || value == '\'') {
            const char quote = value;
            const std::size_t begin = position++;
            bool escaped = false;
            bool closed = false;
            while (position < source.size()) {
                const char current = source[position++];
                if (current == '\n') {
                    error = "newline in ordinary literal";
                    return {};
                }
                if (escaped) {
                    escaped = false;
                } else if (current == '\\') {
                    escaped = true;
                } else if (current == quote) {
                    closed = true;
                    break;
                }
            }
            if (!closed) {
                error = "unterminated ordinary literal";
                return {};
            }
            push(
                quote == '"' ? CppTokenString : CppTokenCharacter,
                begin,
                position);
            continue;
        }
        static const char* multi_tokens[] = {
            "%:%:", "<<=", ">>=", "->*", "...", "::", "->",
            "&&", "||", "==", "!=", ">=", "<=", "++", "--",
            "+=", "-=", "*=", "/=", "%=", "<<", ">>", "##",
            "[[", "]]",
        };
        bool matched = false;
        for (const char* candidate : multi_tokens) {
            const std::size_t length = std::strlen(candidate);
            if (source.compare(position, length, candidate) == 0) {
                push(CppTokenPunctuation, position, position + length);
                position += length;
                matched = true;
                break;
            }
        }
        if (matched) continue;
        push(CppTokenPunctuation, position, position + 1);
        ++position;
    }
    return tokens;
}

static std::size_t cpp_matching_token(
    const std::vector<CppToken>& tokens,
    std::size_t opening,
    const char* open_text,
    const char* close_text)
{
    if (opening >= tokens.size() ||
        tokens[opening].text != open_text) {
        return std::string::npos;
    }
    int depth = 0;
    for (std::size_t index = opening; index < tokens.size(); ++index) {
        if (tokens[index].directive) continue;
        if (tokens[index].text == open_text) {
            ++depth;
        } else if (tokens[index].text == close_text) {
            --depth;
            if (depth == 0) return index;
            if (depth < 0) return std::string::npos;
        }
    }
    return std::string::npos;
}

static std::size_t cpp_matching_token_backward(
    const std::vector<CppToken>& tokens,
    std::size_t closing,
    const char* open_text,
    const char* close_text)
{
    if (closing >= tokens.size() ||
        tokens[closing].text != close_text) {
        return std::string::npos;
    }
    int depth = 0;
    for (std::size_t index = closing + 1; index-- > 0;) {
        if (tokens[index].directive) continue;
        if (tokens[index].text == close_text) {
            ++depth;
        } else if (tokens[index].text == open_text) {
            --depth;
            if (depth == 0) return index;
            if (depth < 0) return std::string::npos;
        }
        if (index == 0) break;
    }
    return std::string::npos;
}

static bool cpp_token_sequence_at(
    const std::vector<CppToken>& tokens,
    std::size_t position,
    const std::vector<std::string>& expected)
{
    if (position + expected.size() > tokens.size()) return false;
    for (std::size_t offset = 0; offset < expected.size(); ++offset) {
        if (tokens[position + offset].directive ||
            tokens[position + offset].text != expected[offset]) {
            return false;
        }
    }
    return true;
}

static std::vector<std::size_t> cpp_find_token_sequence(
    const std::vector<CppToken>& tokens,
    std::size_t begin,
    std::size_t end,
    const std::vector<std::string>& expected)
{
    std::vector<std::size_t> positions;
    const std::size_t limit = std::min(end, tokens.size());
    for (std::size_t index = begin; index < limit; ++index) {
        if (index + expected.size() <= limit &&
            cpp_token_sequence_at(tokens, index, expected)) {
            positions.push_back(index);
        }
    }
    return positions;
}

static std::string cpp_compact_tokens(
    const std::vector<CppToken>& tokens,
    std::size_t begin,
    std::size_t end)
{
    std::string output;
    for (std::size_t index = begin;
         index < end && index < tokens.size();
         ++index) {
        if (!tokens[index].directive) output += tokens[index].text;
    }
    return output;
}

static std::string mask_source_noncode(
    const std::string& source, bool mask_literals)
{
    enum ScanState
    {
        ScanCode,
        ScanLineComment,
        ScanBlockComment,
        ScanString,
        ScanCharacter,
        ScanRawString,
    };
    std::string output = source;
    ScanState state = ScanCode;
    bool escaped = false;
    std::string raw_terminator;
    for (std::size_t index = 0; index < source.size(); ++index) {
        const char current = source[index];
        const char next = index + 1 < source.size()
            ? source[index + 1]
            : '\0';
        if (state == ScanCode) {
            if (current == '/' && next == '/') {
                output[index] = ' ';
                output[index + 1] = ' ';
                ++index;
                state = ScanLineComment;
            } else if (current == '/' && next == '*') {
                output[index] = ' ';
                output[index + 1] = ' ';
                ++index;
                state = ScanBlockComment;
            } else if (current == 'R' && next == '"') {
                const std::size_t delimiter_begin = index + 2;
                const std::size_t opening =
                    source.find('(', delimiter_begin);
                const bool raw_prefix_is_valid =
                    opening != std::string::npos &&
                    opening - delimiter_begin <= 16 &&
                    source.substr(
                        delimiter_begin,
                        opening - delimiter_begin).find_first_of(
                            " \\)\t\r\n") == std::string::npos;
                if (raw_prefix_is_valid) {
                    const std::string delimiter = source.substr(
                        delimiter_begin, opening - delimiter_begin);
                    raw_terminator = ")" + delimiter + "\"";
                    if (mask_literals) {
                        for (std::size_t position = index;
                             position <= opening;
                             ++position) {
                            output[position] = ' ';
                        }
                    }
                    index = opening;
                    state = ScanRawString;
                }
            } else if (current == '"') {
                if (mask_literals) output[index] = ' ';
                state = ScanString;
                escaped = false;
            } else if (current == '\'') {
                if (mask_literals) output[index] = ' ';
                state = ScanCharacter;
                escaped = false;
            }
        } else if (state == ScanLineComment) {
            if (current == '\n') {
                state = ScanCode;
            } else {
                output[index] = ' ';
            }
        } else if (state == ScanBlockComment) {
            if (current == '*' && next == '/') {
                output[index] = ' ';
                output[index + 1] = ' ';
                ++index;
                state = ScanCode;
            } else if (current != '\n') {
                output[index] = ' ';
            }
        } else if (state == ScanRawString) {
            if (source.compare(
                    index,
                    raw_terminator.size(),
                    raw_terminator) == 0) {
                if (mask_literals) {
                    for (std::size_t position = index;
                         position < index + raw_terminator.size();
                         ++position) {
                        output[position] = ' ';
                    }
                }
                index += raw_terminator.size() - 1;
                state = ScanCode;
            } else if (mask_literals && current != '\n') {
                output[index] = ' ';
            }
        } else {
            if (mask_literals && current != '\n') output[index] = ' ';
            if (escaped) {
                escaped = false;
            } else if (current == '\\') {
                escaped = true;
            } else if ((state == ScanString && current == '"') ||
                       (state == ScanCharacter && current == '\'')) {
                state = ScanCode;
            }
        }
    }
    check(state == ScanCode || state == ScanLineComment,
          "controller source comments and literals are lexically balanced");
    return output;
}


static std::string compact_source_text(const std::string& source)
{
    std::string compact;
    compact.reserve(source.size());
    for (const char value : source) {
        if (value != ' ' && value != '\t' && value != '\r' &&
            value != '\n') {
            compact.push_back(value);
        }
    }
    return compact;
}

static std::size_t source_occurrence_count(
    const std::string& source, const std::string& token)
{
    std::size_t count = 0;
    std::size_t cursor = 0;
    while ((cursor = source.find(token, cursor)) != std::string::npos) {
        ++count;
        cursor += token.size();
    }
    return count;
}

static std::size_t source_balanced_group_end(
    const std::string& masked_code,
    std::size_t search_begin,
    char opening,
    char closing,
    const char* message)
{
    const std::size_t group_begin =
        masked_code.find(opening, search_begin);
    check(group_begin != std::string::npos, message);
    int depth = 0;
    for (std::size_t position = group_begin;
         position < masked_code.size();
         ++position) {
        if (masked_code[position] == opening) {
            ++depth;
        } else if (masked_code[position] == closing) {
            --depth;
            check(depth >= 0, message);
            if (depth == 0) return position + 1;
        }
    }
    check(false, message);
    return std::string::npos;
}

static std::string active_call_compact_original(
    const std::string& masked_code,
    const std::string& comments_masked_source,
    std::size_t call_begin)
{
    const std::size_t call_end = source_balanced_group_end(
        masked_code,
        call_begin,
        '(',
        ')',
        "active source call has balanced arguments");
    return compact_source_text(comments_masked_source.substr(
        call_begin, call_end - call_begin));
}

static std::size_t exact_active_call_count(
    const std::string& masked_code,
    const std::string& comments_masked_source,
    const std::string& active_token,
    const std::string& exact_compact_call,
    std::size_t* first_position = nullptr)
{
    std::size_t count = 0;
    std::size_t cursor = 0;
    while ((cursor = masked_code.find(
                active_token, cursor)) != std::string::npos) {
        const bool identifier_prefix = cursor > 0 &&
            ((masked_code[cursor - 1] >= 'a' &&
              masked_code[cursor - 1] <= 'z') ||
             (masked_code[cursor - 1] >= 'A' &&
              masked_code[cursor - 1] <= 'Z') ||
             (masked_code[cursor - 1] >= '0' &&
              masked_code[cursor - 1] <= '9') ||
             masked_code[cursor - 1] == '_' ||
             masked_code[cursor - 1] == ':' ||
             masked_code[cursor - 1] == '.' ||
             masked_code[cursor - 1] == '>');
        if (!identifier_prefix && active_call_compact_original(
                masked_code,
                comments_masked_source,
                cursor) == exact_compact_call) {
            if (count == 0 && first_position != nullptr) {
                *first_position = cursor;
            }
            ++count;
        }
        cursor += active_token.size();
    }
    return count;
}

static bool exact_no_main_excluding_opening(
    const std::string& directive)
{
    std::string error;
    const std::vector<CppToken> tokens =
        tokenize_cpp_source(directive, error);
    if (!error.empty()) return false;
    const std::vector<std::string> ifndef_pattern = {
        "#", "ifndef", "G1_CONTROLLER_NO_MAIN",
    };
    const std::vector<std::string> if_pattern = {
        "#", "if", "!", "defined", "(",
        "G1_CONTROLLER_NO_MAIN", ")",
    };
    const auto exact_directive = [&](
        const std::vector<std::string>& pattern) {
        if (tokens.size() != pattern.size()) return false;
        for (std::size_t index = 0; index < pattern.size(); ++index) {
            if (!tokens[index].directive ||
                tokens[index].text != pattern[index]) {
                return false;
            }
        }
        return true;
    };
    return exact_directive(ifndef_pattern) ||
           exact_directive(if_pattern);
}

struct NoMainSourceView
{
    std::string visible;
    std::vector<unsigned char> excluded;
};

static bool build_no_main_source_view(
    const std::string& source,
    NoMainSourceView& view,
    std::string& error)
{
    struct Conditional
    {
        bool exact_no_main = false;
    };
    if (source.find("\\\n") != std::string::npos ||
        source.find("\\\r\n") != std::string::npos) {
        error = "translation-phase line splice is forbidden";
        return false;
    }
    if (source.find("%:") != std::string::npos) {
        error = "preprocessor directive digraph is forbidden";
        return false;
    }
    std::string token_error;
    const std::vector<CppToken> tokens =
        tokenize_cpp_source(source, token_error);
    if (!token_error.empty()) {
        error = token_error;
        return false;
    }
    view.visible = source;
    view.excluded.clear();
    view.excluded.resize(source.size(), 0U);
    std::vector<Conditional> conditionals;
    std::size_t line_start = 0;
    int line_number = 1;
    std::size_t token_cursor = 0;
    while (line_start < source.size()) {
        const std::size_t line_end = source.find('\n', line_start);
        const std::size_t end = line_end == std::string::npos
            ? source.size()
            : line_end;
        std::vector<std::string> directive_tokens;
        while (token_cursor < tokens.size() &&
               tokens[token_cursor].line < line_number) {
            ++token_cursor;
        }
        std::size_t line_token = token_cursor;
        while (line_token < tokens.size() &&
               tokens[line_token].line == line_number) {
            if (tokens[line_token].directive) {
                directive_tokens.push_back(tokens[line_token].text);
            }
            ++line_token;
        }

        bool in_exact_region = false;
        for (const Conditional& conditional : conditionals) {
            in_exact_region = in_exact_region ||
                conditional.exact_no_main;
        }
        if (in_exact_region) {
            for (std::size_t position = line_start;
                 position < end;
                 ++position) {
                if (view.visible[position] != '\n') {
                    view.visible[position] = ' ';
                }
                view.excluded[position] = 1U;
            }
        }

        const bool directive_ready =
            directive_tokens.size() >= 2 &&
            directive_tokens[0] == "#";
        const bool opens_conditional = directive_ready &&
            (directive_tokens[1] == "if" ||
             directive_tokens[1] == "ifdef" ||
             directive_tokens[1] == "ifndef");
        if (opens_conditional) {
            Conditional conditional;
            std::string directive;
            for (const std::string& token : directive_tokens) {
                directive += token;
                directive.push_back(' ');
            }
            conditional.exact_no_main =
                exact_no_main_excluding_opening(directive);
            conditionals.push_back(conditional);
        } else if (directive_ready &&
                   (directive_tokens[1] == "else" ||
                    directive_tokens[1] == "elif")) {
            if (conditionals.empty()) {
                error = "unmatched preprocessor branch";
                return false;
            }
            if (conditionals.back().exact_no_main) {
                error = "exact no-main guard cannot have else/elif";
                return false;
            }
        } else if (directive_ready &&
                   directive_tokens[1] == "endif") {
            if (conditionals.empty()) {
                error = "unmatched preprocessor endif";
                return false;
            }
            conditionals.pop_back();
        }
        if (line_end == std::string::npos) break;
        line_start = line_end + 1;
        ++line_number;
    }
    if (!conditionals.empty()) {
        error = "unclosed preprocessor conditional";
        return false;
    }
    return true;
}

static NoMainSourceView no_main_source_view(
    const std::string& source)
{
    NoMainSourceView view;
    std::string error;
    check(build_no_main_source_view(source, view, error),
          error.empty() ? "no-main source view is valid" : error.c_str());
    return view;
}

static bool position_is_inside_no_main_guard(
    const NoMainSourceView& view, std::size_t target)
{
    return target < view.excluded.size() && view.excluded[target] != 0U;
}

struct CppFunctionDefinition
{
    std::string name;
    std::size_t declaration_begin = 0;
    std::size_t name_index = 0;
    std::size_t parameters_begin = 0;
    std::size_t parameters_end = 0;
    std::size_t body_begin = 0;
    std::size_t body_end = 0;
    bool internal_linkage = false;
};

static bool cpp_range_has_token(
    const std::vector<CppToken>& tokens,
    std::size_t begin,
    std::size_t end,
    const char* text)
{
    for (std::size_t index = begin;
         index < end && index < tokens.size();
         ++index) {
        if (!tokens[index].directive && tokens[index].text == text) {
            return true;
        }
    }
    return false;
}

static bool cpp_range_has_indirect_callable_syntax(
    const std::vector<CppToken>& tokens,
    std::size_t begin,
    std::size_t end)
{
    const std::size_t limit = std::min(end, tokens.size());
    for (std::size_t index = begin; index < limit; ++index) {
        if (tokens[index].directive) continue;
        if (tokens[index].kind == CppTokenIdentifier &&
            tokens[index].text != "static_cast" &&
            tokens[index].text != "const_cast" &&
            tokens[index].text != "reinterpret_cast" &&
            tokens[index].text != "dynamic_cast" &&
            index + 3 < limit && tokens[index + 1].text == "<") {
            int angle_depth = 0;
            for (std::size_t cursor = index + 1;
                 cursor < limit;
                 ++cursor) {
                if (tokens[cursor].text == "<") {
                    ++angle_depth;
                } else if (tokens[cursor].text == ">") {
                    --angle_depth;
                } else if (tokens[cursor].text == ">>") {
                    angle_depth -= 2;
                } else if (angle_depth == 1 &&
                           (tokens[cursor].text == ";" ||
                            tokens[cursor].text == "{" ||
                            tokens[cursor].text == "}")) {
                    break;
                }
                if (angle_depth <= 0) {
                    std::size_t after = cursor + 1;
                    while (after < limit && tokens[after].directive) ++after;
                    if (angle_depth == 0 && after < limit &&
                        tokens[after].text == "(") {
                        return true;
                    }
                    break;
                }
            }
        }
        if (tokens[index].text == ")") {
            std::size_t next = index + 1;
            while (next < limit && tokens[next].directive) ++next;
            if (next < limit && tokens[next].text == "(") return true;
        }
        if (tokens[index].text != "(" || index + 2 >= limit) continue;
        std::size_t cursor = index + 1;
        while (cursor < limit &&
               (tokens[cursor].text == "const" ||
                tokens[cursor].text == "volatile")) {
            ++cursor;
        }
        if (cursor >= limit ||
            (tokens[cursor].text != "*" &&
             tokens[cursor].text != "&" &&
             tokens[cursor].text != "&&")) {
            continue;
        }
        const std::size_t close = cpp_matching_token(
            tokens, index, "(", ")");
        if (close == std::string::npos || close + 1 >= limit) continue;
        std::size_t after = close + 1;
        while (after < limit && tokens[after].directive) ++after;
        if (after < limit && tokens[after].text == "(") return true;
    }
    return false;
}

static bool collect_cpp_top_level(
    const std::vector<CppToken>& tokens,
    std::vector<CppFunctionDefinition>& functions,
    std::vector<unsigned char>& covered,
    std::string& error)
{
    covered.assign(tokens.size(), 0U);
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index].directive) covered[index] = 1U;
        if (tokens[index].text == "[[" ||
            tokens[index].text == "]]" ||
            tokens[index].text == "__attribute__") {
            error = "attributes are unsupported in audited no-main code";
            return false;
        }
    }
    int depth = 0;
    std::size_t segment_begin = 0;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index].directive) continue;
        if (tokens[index].text == "{") {
            if (depth != 0) {
                ++depth;
                continue;
            }
            const std::size_t close = cpp_matching_token(
                tokens, index, "{", "}");
            if (close == std::string::npos) {
                error = "unbalanced top-level brace";
                return false;
            }
            std::size_t previous = index;
            while (previous > segment_begin &&
                   tokens[previous - 1].directive) {
                --previous;
            }
            const bool function_candidate = previous > segment_begin &&
                tokens[previous - 1].text == ")";
            if (function_candidate) {
                const std::size_t parameters_end = previous - 1;
                const std::size_t parameters_begin =
                    cpp_matching_token_backward(
                        tokens, parameters_end, "(", ")");
                if (parameters_begin == std::string::npos ||
                    parameters_begin == 0 ||
                    tokens[parameters_begin - 1].kind !=
                        CppTokenIdentifier) {
                    error = "unsupported top-level function declarator";
                    return false;
                }
                CppFunctionDefinition function;
                function.name = tokens[parameters_begin - 1].text;
                function.declaration_begin = segment_begin;
                function.name_index = parameters_begin - 1;
                function.parameters_begin = parameters_begin;
                function.parameters_end = parameters_end;
                function.body_begin = index;
                function.body_end = close;
                function.internal_linkage = cpp_range_has_token(
                    tokens,
                    segment_begin,
                    function.name_index,
                    "static");
                functions.push_back(function);
                for (std::size_t token = segment_begin;
                     token <= close;
                     ++token) {
                    covered[token] = 1U;
                }
                index = close;
                segment_begin = close + 1;
                continue;
            }
            error = "non-function braced root survives no-main closure";
            return false;
        } else if (tokens[index].text == "}") {
            if (depth == 0) {
                error = "unmatched top-level closing brace";
                return false;
            }
            --depth;
        } else if (tokens[index].text == ";" && depth == 0) {
            std::size_t first = segment_begin;
            while (first < index &&
                   (covered[first] || tokens[first].directive)) {
                ++first;
            }
            if (first < index) {
                if (cpp_range_has_indirect_callable_syntax(
                        tokens, first, index + 1)) {
                    error = "function-pointer root survives no-main closure";
                    return false;
                }
                const bool allowed_declaration =
                    tokens[first].text == "using" ||
                    tokens[first].text == "typedef" ||
                    tokens[first].text == "static_assert" ||
                    tokens[first].text == "struct" ||
                    tokens[first].text == "class" ||
                    tokens[first].text == "enum" ||
                    cpp_range_has_token(
                        tokens, first, index, "constexpr");
                if (!allowed_declaration) {
                    error = "mutable or unsupported file-scope data/root";
                    return false;
                }
                for (std::size_t token = first;
                     token <= index;
                     ++token) {
                    covered[token] = 1U;
                }
            }
            segment_begin = index + 1;
        }
    }
    return true;
}

static std::string cpp_call_key(
    const std::vector<CppToken>& tokens, std::size_t name_index)
{
    if (name_index > 0 && tokens[name_index - 1].text == ".") {
        return "." + tokens[name_index].text;
    }
    if (name_index > 0 && tokens[name_index - 1].text == "->") {
        return "->" + tokens[name_index].text;
    }
    if (name_index > 0 && tokens[name_index - 1].text == "::") {
        if (name_index > 1 &&
            tokens[name_index - 2].kind == CppTokenIdentifier &&
            tokens[name_index - 2].text != "return") {
            return tokens[name_index - 2].text + "::" +
                tokens[name_index].text;
        }
        return "::" + tokens[name_index].text;
    }
    return tokens[name_index].text;
}

struct CppCheckedIndexOwner
{
    std::string name;
    int dimensions = 0;
    std::size_t declaration_name = 0;
};

static bool cpp_collect_checked_index_owners(
    const std::vector<CppToken>& tokens,
    std::size_t begin,
    std::size_t end,
    std::vector<CppCheckedIndexOwner>& owners,
    std::string& error)
{
    const std::size_t limit = std::min(end, tokens.size());
    for (std::size_t index = begin; index + 3 < limit; ++index) {
        if (tokens[index].directive ||
            tokens[index].kind != CppTokenIdentifier) {
            continue;
        }
        int dimensions = 0;
        if (tokens[index].text == "array1d" ||
            tokens[index].text == "slice1d") {
            dimensions = 1;
        } else if (tokens[index].text == "array2d" ||
                   tokens[index].text == "slice2d") {
            dimensions = 2;
        } else {
            continue;
        }
        if (tokens[index + 1].text != "<") continue;
        int angle_depth = 0;
        std::size_t close = std::string::npos;
        for (std::size_t cursor = index + 1; cursor < limit; ++cursor) {
            if (tokens[cursor].text == "<") {
                ++angle_depth;
            } else if (tokens[cursor].text == ">") {
                --angle_depth;
            } else if (tokens[cursor].text == ">>") {
                angle_depth -= 2;
            }
            if (angle_depth == 0) {
                close = cursor;
                break;
            }
            if (angle_depth < 0 || tokens[cursor].text == ";" ||
                tokens[cursor].text == "{" ||
                tokens[cursor].text == "}") {
                break;
            }
        }
        if (close == std::string::npos) continue;
        std::size_t name = close + 1;
        while (name < limit &&
               (tokens[name].text == "const" ||
                tokens[name].text == "volatile" ||
                tokens[name].text == "*" ||
                tokens[name].text == "&" ||
                tokens[name].text == "&&")) {
            ++name;
        }
        if (name >= limit ||
            tokens[name].kind != CppTokenIdentifier) {
            continue;
        }
        const std::size_t after = name + 1;
        if (after < limit &&
            tokens[after].text != "," &&
            tokens[after].text != ")" &&
            tokens[after].text != ";" &&
            tokens[after].text != "=" &&
            tokens[after].text != "{" &&
            tokens[after].text != "(" &&
            tokens[after].text != "[") {
            continue;
        }
        for (const CppCheckedIndexOwner& owner : owners) {
            if (owner.name == tokens[name].text) {
                error = "duplicate/shadowed checked-index owner in no-main closure: " +
                    owner.name;
                return false;
            }
        }
        CppCheckedIndexOwner owner;
        owner.name = tokens[name].text;
        owner.dimensions = dimensions;
        owner.declaration_name = name;
        owners.push_back(owner);
        index = name;
    }
    return true;
}

static int cpp_direct_call_argument_count(
    const std::vector<CppToken>& tokens,
    std::size_t opening,
    std::size_t closing)
{
    if (opening >= closing || closing > tokens.size()) return -1;
    if (opening + 1 == closing) return 0;
    int parentheses = 0;
    int brackets = 0;
    int braces = 0;
    int arguments = 1;
    for (std::size_t index = opening + 1; index < closing; ++index) {
        if (tokens[index].directive) return -1;
        if (tokens[index].text == "(") {
            ++parentheses;
        } else if (tokens[index].text == ")") {
            --parentheses;
        } else if (tokens[index].text == "[") {
            ++brackets;
        } else if (tokens[index].text == "]") {
            --brackets;
        } else if (tokens[index].text == "{") {
            ++braces;
        } else if (tokens[index].text == "}") {
            --braces;
        } else if (tokens[index].text == "," &&
                   parentheses == 0 && brackets == 0 && braces == 0) {
            ++arguments;
        }
        if (parentheses < 0 || brackets < 0 || braces < 0) return -1;
    }
    return parentheses == 0 && brackets == 0 && braces == 0
        ? arguments
        : -1;
}

static bool analyze_no_main_root_closure(
    const std::string& source,
    const std::string& root_name,
    const std::set<std::string>& trusted_calls,
    std::string& error)
{
    if (source.find("%:") != std::string::npos ||
        source.find("<%") != std::string::npos ||
        source.find("%>") != std::string::npos ||
        source.find("\\\n") != std::string::npos ||
        source.find("\\\r\n") != std::string::npos) {
        error = "closure source contains a forbidden preprocessor spelling";
        return false;
    }
    std::vector<CppToken> tokens = tokenize_cpp_source(source, error);
    if (!error.empty()) return false;
    for (const CppToken& token : tokens) {
        if (!token.directive &&
            (token.text == "operator" || token.text == "template")) {
            error = "template/operator callable forms are unsupported in no-main closure";
            return false;
        }
    }
    std::vector<CppFunctionDefinition> functions;
    std::vector<unsigned char> covered;
    if (!collect_cpp_top_level(tokens, functions, covered, error)) {
        return false;
    }
    int root = -1;
    std::set<std::string> local_names;
    for (std::size_t index = 0; index < functions.size(); ++index) {
        if (!local_names.insert(functions[index].name).second) {
            error = "overloaded/duplicate local root is unsupported";
            return false;
        }
        if (functions[index].name == root_name) {
            if (root >= 0 || functions[index].internal_linkage) {
                error = "runner root must be the unique external definition";
                return false;
            }
            root = static_cast<int>(index);
        } else if (!functions[index].internal_linkage) {
            error = "non-runner external function root survives no-main";
            return false;
        }
    }
    if (root < 0) {
        error = "runner root is missing from no-main closure";
        return false;
    }

    const std::set<std::string> language_calls = {
        "if", "for", "while", "switch", "sizeof", "alignof",
        "decltype", "noexcept", "static_assert", "return",
    };
    std::vector<std::vector<int>> edges(functions.size());
    for (std::size_t function_index = 0;
         function_index < functions.size();
         ++function_index) {
        const CppFunctionDefinition& function =
            functions[function_index];
        if (cpp_range_has_indirect_callable_syntax(
                tokens,
                function.declaration_begin,
                function.body_begin)) {
            error = "function-pointer/indirect callable declarator in no-main closure";
            return false;
        }
        if (cpp_range_has_indirect_callable_syntax(
                tokens,
                function.body_begin + 1,
                function.body_end)) {
            error = "template/parenthesized/function-pointer invocation in no-main closure";
            return false;
        }
        std::vector<CppCheckedIndexOwner> checked_index_owners;
        if (!cpp_collect_checked_index_owners(
                tokens,
                function.parameters_begin + 1,
                function.parameters_end,
                checked_index_owners,
                error) ||
            !cpp_collect_checked_index_owners(
                tokens,
                function.body_begin + 1,
                function.body_end,
                checked_index_owners,
                error)) {
            return false;
        }
        for (std::size_t index = function.body_begin + 1;
             index < function.body_end;
             ++index) {
            if (tokens[index].directive) {
                error = "preprocessor branch inside no-main function";
                return false;
            }
            if (tokens[index].text == "const_cast" ||
                tokens[index].text == "reinterpret_cast" ||
                tokens[index].text == "dynamic_cast") {
                error = "cast can escape immutable no-main authority";
                return false;
            }
            if (tokens[index].text == "extern") {
                error = "extern/nonlocal object declaration in no-main closure";
                return false;
            }
            const std::set<std::string> unsupported_transfers = {
                "goto", "throw", "try", "catch", "co_await",
                "co_yield", "co_return", "asm", "__asm__",
                "setjmp", "longjmp",
            };
            if (unsupported_transfers.count(tokens[index].text) != 0U) {
                error = "unsupported control transfer in no-main closure";
                return false;
            }
            if (cpp_range_has_indirect_callable_syntax(
                    tokens, index, std::min(index + 6, function.body_end))) {
                error = "parenthesized/function-pointer invocation in no-main closure";
                return false;
            }
            if (tokens[index].text == "static" &&
                (index + 1 >= function.body_end ||
                 tokens[index + 1].text != "constexpr")) {
                error = "mutable static local in no-main closure";
                return false;
            }
            if (tokens[index].text == "[" ) {
                const std::size_t close = cpp_matching_token(
                    tokens, index, "[", "]");
                if (close != std::string::npos &&
                    close + 1 < function.body_end &&
                    (tokens[close + 1].text == "{" ||
                     tokens[close + 1].text == "(")) {
                    error = "lambda/local callable in no-main closure";
                    return false;
                }
            }
            if (tokens[index].kind != CppTokenIdentifier ||
                index + 1 >= function.body_end ||
                tokens[index + 1].text != "(" ||
                language_calls.count(tokens[index].text) != 0U) {
                continue;
            }
            const std::string key = cpp_call_key(tokens, index);
            int local_target = -1;
            for (std::size_t candidate = 0;
                 candidate < functions.size();
                 ++candidate) {
                if (functions[candidate].name == tokens[index].text) {
                    local_target = static_cast<int>(candidate);
                    break;
                }
            }
            if (local_target >= 0 &&
                (key == tokens[index].text ||
                 key == "::" + tokens[index].text)) {
                edges[function_index].push_back(local_target);
            } else {
                const CppCheckedIndexOwner* checked_owner = nullptr;
                for (const CppCheckedIndexOwner& owner :
                     checked_index_owners) {
                    if (key == owner.name) {
                        checked_owner = &owner;
                        break;
                    }
                }
                if (checked_owner != nullptr) {
                    if (index == checked_owner->declaration_name) continue;
                    const std::size_t closing = cpp_matching_token(
                        tokens, index + 1, "(", ")");
                    if (closing == std::string::npos ||
                        closing >= function.body_end ||
                        cpp_direct_call_argument_count(
                            tokens, index + 1, closing) !=
                            checked_owner->dimensions) {
                        error = "checked-index owner has invalid arity in no-main closure: " +
                            checked_owner->name;
                        return false;
                    }
                    continue;
                }
                if (trusted_calls.count(key) != 0U) continue;
                error = "untrusted external/header call in no-main closure: " +
                    key;
                return false;
            }
        }
    }

    std::vector<unsigned char> reachable(functions.size(), 0U);
    std::vector<int> pending(1, root);
    while (!pending.empty()) {
        const int current = pending.back();
        pending.pop_back();
        if (reachable[static_cast<std::size_t>(current)] != 0U) continue;
        reachable[static_cast<std::size_t>(current)] = 1U;
        for (const int target : edges[static_cast<std::size_t>(current)]) {
            pending.push_back(target);
        }
    }
    for (std::size_t index = 0; index < functions.size(); ++index) {
        if (reachable[index] == 0U) {
            error = "orphan static helper survives strict no-main closure: " +
                functions[index].name;
            return false;
        }
    }
    return true;
}

static bool build_authenticated_production_no_main_closure_source(
    const NoMainSourceView& no_main_view,
    std::string& output,
    std::string& error)
{
    const std::vector<CppToken> tokens = tokenize_cpp_source(
        no_main_view.visible, error);
    if (!error.empty()) return false;
    const std::vector<std::size_t> query_boundaries =
        cpp_find_token_sequence(
            tokens,
            0,
            tokens.size(),
            {"[[", "gnu", "::", "noinline", "]]", "static", "void",
             "g1_runner_build_query", "("});
    if (query_boundaries.size() != 1) {
        error = "legacy query builder does not have one exact authenticated out-of-line boundary";
        return false;
    }
    output = no_main_view.visible;
    const std::size_t attribute_begin =
        tokens[query_boundaries[0]].begin;
    const std::size_t attribute_end =
        tokens[query_boundaries[0] + 4].end;
    output.replace(
        attribute_begin,
        attribute_end - attribute_begin,
        attribute_end - attribute_begin,
        ' ');
    return true;
}

static std::set<std::string> production_trusted_no_main_calls()
{
    return {
        "::atan2f",
        "::clampf",
        "::database_search",
        "::database_trajectory_index_clamp",
        "::deterministic_route_command",
        "::dot",
        "::fabsf",
        "::fmaxf",
        "::fminf",
        "::g1_footprint_observe_v2",
        "::g1_foot_contact_schedule_build",
        "::g1_footprint_budget",
        "::g1_frame_candidate_record_is_valid",
        "::g1_idle_match_transition_cost",
        "::g1_ik_checked_forward_kinematics",
        "::g1_ik_frame_begin",
        "::g1_ik_frame_finish",
        "::g1_ik_frame_rejection_snapshot",
        "::g1_ik_frame_stage_foot",
        "::g1_ik_safe_stop_handoff",
        "::g1_measure_pose_clearance",
        "::g1_pose_clearance_budget",
        "::heightfield_sample_v2",
        "::isfinite",
        "::length",
        "::normalize",
        "::quat_abs",
        "::quat_from_angle_axis",
        "::quat_inv",
        "::quat_mul",
        "::quat_mul_inv",
        "::quat_mul_vec3",
        "::snprintf",
        "::support_observation_build",
        "::support_pose_apply",
        "::terrain_centerline_snapshot_compute_v2",
        "::terrain_centerline_query",
        "::terrain_f32_add",
        "::terrain_f32_mul",
        "::terrain_f32_sub",
        "::terrain_float_bits",
        "::walkability_xz_length",
        "std::max",
        "std::min",
        "array1d",
        "quat",
        "slice1d",
        "vec3",
        ".adjusted_bone_positions",
        ".adjusted_bone_rotations",
        ".bone_angular_velocities",
        ".bone_offset_angular_velocities",
        ".bone_offset_positions",
        ".bone_offset_rotations",
        ".bone_offset_velocities",
        ".bone_parents",
        ".bone_positions",
        ".bone_rotations",
        ".bone_velocities",
        ".contact_bones",
        ".contact_locks",
        ".contact_offset_positions",
        ".contact_offset_velocities",
        ".contact_points",
        ".contact_positions",
        ".contact_states",
        ".contact_targets",
        ".contact_velocities",
        ".curr_bone_angular_velocities",
        ".curr_bone_contacts",
        ".curr_bone_positions",
        ".curr_bone_rotations",
        ".curr_bone_velocities",
        ".features",
        ".features_offset",
        ".features_scale",
        ".global_bone_angular_velocities",
        ".global_bone_computed",
        ".global_bone_positions",
        ".global_bone_rotations",
        ".global_bone_velocities",
        ".ik_bone_positions",
        ".ik_bone_rotations",
        ".ik_candidate_bone_positions",
        ".ik_candidate_bone_rotations",
        ".ik_candidate_global_bone_positions",
        ".ik_candidate_global_bone_rotations",
        ".ik_global_bone_positions",
        ".ik_global_bone_rotations",
        ".nframes",
        ".nbones",
        ".range_starts",
        ".range_stops",
        ".terrain_features",
        ".trajectory_accelerations",
        ".trajectory_angular_velocities",
        ".trajectory_desired_rotations",
        ".trajectory_desired_velocities",
        ".trajectory_positions",
        ".trajectory_rotations",
        ".trajectory_velocities",
        ".trns_bone_angular_velocities",
        ".trns_bone_contacts",
        ".trns_bone_positions",
        ".trns_bone_rotations",
        ".trns_bone_velocities",
        "->bone_angular_velocities",
        "->bone_parents",
        "->bone_positions",
        "->bone_rotations",
        "->bone_velocities",
        "->contact_states",
        "->features",
        "->features_offset",
        "->features_scale",
        "->nframes",
        "->nbones",
        "->range_starts",
        "->range_stops",
        "->terrain_features",
        "->values",
    };
}

static std::string authenticated_call_identifier(const std::string& key)
{
    std::size_t begin = key.size();
    while (begin > 0 &&
           (cpp_identifier_continue(key[begin - 1]))) {
        --begin;
    }
    return key.substr(begin);
}

static std::set<std::string> production_authenticated_source_names()
{
    std::set<std::string> names;
    const std::set<std::string> trusted =
        production_trusted_no_main_calls();
    for (const std::string& call : trusted) {
        const std::string identifier = authenticated_call_identifier(call);
        if (!identifier.empty()) names.insert(identifier);
    }
    const char* authenticated_calls[] = {
        "BeginDrawing", "BeginMode3D", "ClearBackground",
        "DrawCylinderEx", "DrawGrid", "DrawLine3D", "DrawModel",
        "DrawSphereWires", "EndDrawing", "EndMode3D", "GetTime",
        "WindowShouldClose", "contact_update", "controlled_runtime_error",
        "deterministic_route_command", "draw_g1_skeleton",
        "g1_candidate_audit_config_parse",
        "g1_apply_pending_scene_reset", "g1_build_accepted_log_row",
        "g1_controller_frame_stage_run", "g1_frame_transaction_run",
        "g1_parse_halflife", "g1_parse_ik_enabled",
        "g1_parse_search_time", "g1_parse_strafe_enabled",
        "gamepad_get_stick", "getenv", "inertialize_pose_update",
        "isfinite", "main", "motion_match_pose_snapshot",
        "simulation_positions_update", "simulation_rotations_update",
        "strcmp", "strtof", "traversability_command",
        "traversability_query", "update_g1_camera_from_accepted", "write",
        "write_requested",
    };
    for (const char* call : authenticated_calls) names.insert(call);
    const char* stage_and_status_names[] = {
        "G1FrameStageInputRouteCommand", "G1FrameStageMatcherSearch",
        "G1FrameStageInertialization", "G1FrameStageSimulationUpdate",
        "G1FrameStageSupportObservation", "G1FrameStageSupportRetarget",
        "G1FrameStageContactUpdate", "G1FrameStageFootprintObservation",
        "G1FrameStageFirstFootIk", "G1FrameStageSecondFootIk",
        "G1FrameStageFinalFk", "G1FrameStagePoseCertificate",
        "G1FrameStageCount", "G1FrameStageContinue",
        "G1FrameStageFiniteReject", "G1FrameStageGlobalError",
        "G1FrameTransactionAccepted", "G1FrameTransactionFiniteRejected",
        "G1FrameTransactionGlobalError", "G1FrameInjectContinue",
        "G1FrameInjectFiniteReject", "G1FrameInjectGlobalError",
    };
    for (const char* name : stage_and_status_names) names.insert(name);
    const char* bindings_and_owners[] = {
        "accepted_diagnostic", "accepted_state", "active_scene_index",
        "artifact_error", "camera", "candidate_audit_config",
        "candidate_audit_environment", "candidate_audit_evidence_ok",
        "candidate_audit_log", "candidate_audit_ok",
        "const", "controller_exit_code", "database", "db",
        "controller_exit_requested",
        "deterministic_log", "error", "error_capacity", "external",
        "frame_external", "frame_runtime", "frame_status", "log_context",
        "log_ok", "log_row", "log_row_ok", "parents",
        "pending_reset", "pending_scene_index", "positions",
        "process_config", "publication", "rotations", "run_stage",
        "scratch", "stage", "working_state",
        "G1CandidateAuditConfig", "G1CandidateAuditLog",
        "G1FrameAcceptedDiagnostic", "G1FramePublication",
        "G1FrameTransactionStatus", "g1_controller_state",
    };
    for (const char* name : bindings_and_owners) names.insert(name);
    const char* authenticated_input_and_parser_tokens[] = {
        "ERANGE", "NULL", "camera_zoom_axis", "contact_blending_halflife",
        "contact_foot_height", "contact_unlock_radius", "desired_strafe",
        "effective_terrain_weight", "end", "environ", "__environ",
        "errno", "gait_target",
        "ik_enabled", "inertialize_blending_halflife",
        "initial_search_time", "look_stick", "mode", "move_stick",
        "mode_name", "output", "parsed", "presentation_frame", "route",
        "route_speed",
        "scene_dwell_frames", "scripted_azimuth_delta",
        "simulation_rotation_halflife", "test_heading", "text",
        "trajectory_sample_time",
    };
    for (const char* name : authenticated_input_and_parser_tokens) {
        names.insert(name);
    }
    return names;
}

static bool audit_controller_macros(
    const std::vector<CppToken>& tokens,
    const std::set<std::string>& audited_names,
    std::string& error)
{
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index].text == "##" ||
            tokens[index].text == "%:%:") {
            error = "token paste is forbidden in audited controller source";
            return false;
        }
        if (!tokens[index].directive || tokens[index].text != "#" ||
            index + 2 >= tokens.size() ||
            tokens[index + 1].line != tokens[index].line ||
            tokens[index + 2].line != tokens[index].line) {
            continue;
        }
        const std::string& action = tokens[index + 1].text;
        if ((action == "define" || action == "undef") &&
            audited_names.count(tokens[index + 2].text) != 0U) {
            error = "audited identifier cannot be macro-defined or undefined: " +
                tokens[index + 2].text;
            return false;
        }
    }
    return true;
}

static std::vector<int> cpp_brace_depth_before(
    const std::vector<CppToken>& tokens)
{
    std::vector<int> depth(tokens.size(), 0);
    int current = 0;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        depth[index] = current;
        if (tokens[index].directive) continue;
        if (tokens[index].text == "{") {
            ++current;
        } else if (tokens[index].text == "}") {
            --current;
        }
    }
    return depth;
}

static bool cpp_top_level_function_declaration(
    const std::vector<CppToken>& tokens,
    std::size_t begin,
    std::size_t end)
{
    std::size_t first_assignment = end;
    for (std::size_t index = begin; index < end; ++index) {
        if (!tokens[index].directive && tokens[index].text == "=") {
            first_assignment = index;
            break;
        }
    }
    for (std::size_t index = begin + 1;
         index < end && index < first_assignment;
         ++index) {
        if (tokens[index].directive || tokens[index].text != "(" ||
            tokens[index - 1].kind != CppTokenIdentifier) {
            continue;
        }
        const std::size_t close = cpp_matching_token(
            tokens, index, "(", ")");
        if (close == std::string::npos || close >= end) continue;
        bool direct_initializer = false;
        for (std::size_t argument = index + 1;
             argument < close;
             ++argument) {
            direct_initializer = direct_initializer ||
                tokens[argument].kind == CppTokenNumber ||
                tokens[argument].kind == CppTokenString ||
                tokens[argument].kind == CppTokenCharacter ||
                tokens[argument].text == "true" ||
                tokens[argument].text == "false" ||
                tokens[argument].text == "nullptr" ||
                tokens[argument].text == "NULL";
        }
        if (!direct_initializer) return true;
    }
    return false;
}

static bool audit_project_header_scope(
    const std::string& path,
    const std::vector<CppToken>& tokens,
    std::size_t begin,
    std::size_t end,
    bool inside_type,
    std::string& error)
{
    std::size_t segment_begin = begin;
    int local_brace_depth = 0;
    for (std::size_t index = begin; index < end; ++index) {
        if (tokens[index].directive) continue;
        if (tokens[index].text == "{") {
            if (local_brace_depth != 0) {
                ++local_brace_depth;
                continue;
            }
            const std::size_t close = cpp_matching_token(
                tokens, index, "{", "}");
            if (close == std::string::npos || close >= end) {
                error = path + ": unbalanced nested header root";
                return false;
            }
            std::size_t previous = index;
            while (previous > segment_begin &&
                   tokens[previous - 1].directive) {
                --previous;
            }
            const bool function_root = previous > segment_begin &&
                tokens[previous - 1].text == ")";
            const bool namespace_root = cpp_range_has_token(
                tokens, segment_begin, index, "namespace");
            const bool struct_root = cpp_range_has_token(
                tokens, segment_begin, index, "struct");
            const bool class_root = cpp_range_has_token(
                tokens, segment_begin, index, "class");
            const bool union_root = cpp_range_has_token(
                tokens, segment_begin, index, "union");
            const bool enum_root = cpp_range_has_token(
                tokens, segment_begin, index, "enum");
            if (namespace_root || struct_root || class_root || union_root) {
                if (!audit_project_header_scope(
                        path,
                        tokens,
                        index + 1,
                        close,
                        !namespace_root,
                        error)) {
                    return false;
                }
                segment_begin = close + 1;
            } else if (function_root || enum_root) {
                segment_begin = close + 1;
            }
            index = close;
            continue;
        }
        if (tokens[index].text == "}") {
            if (local_brace_depth == 0) {
                error = path + ": unmatched nested header brace";
                return false;
            }
            --local_brace_depth;
            continue;
        }
        if (tokens[index].text != ";" || local_brace_depth != 0) continue;
        std::size_t first = segment_begin;
        while (first < index &&
               (tokens[first].directive ||
                tokens[first].text == "public" ||
                tokens[first].text == "private" ||
                tokens[first].text == "protected" ||
                tokens[first].text == ":")) {
            ++first;
        }
        if (first >= index) {
            segment_begin = index + 1;
            continue;
        }
        if (cpp_range_has_indirect_callable_syntax(
                tokens, first, index + 1) ||
            cpp_range_has_token(tokens, first, index, "function")) {
            error = path + ": function-pointer/callable nested root: " +
                cpp_compact_tokens(tokens, first, index + 1).substr(0, 160);
            return false;
        }
        const bool type_declaration =
            tokens[first].text == "using" ||
            tokens[first].text == "typedef" ||
            tokens[first].text == "static_assert" ||
            tokens[first].text == "struct" ||
            tokens[first].text == "class" ||
            tokens[first].text == "enum" ||
            tokens[first].text == "union";
        if (!type_declaration &&
            !cpp_top_level_function_declaration(tokens, first, index)) {
            const bool is_extern = cpp_range_has_token(
                tokens, first, index, "extern");
            const bool is_static = cpp_range_has_token(
                tokens, first, index, "static");
            const bool is_constexpr = cpp_range_has_token(
                tokens, first, index, "constexpr");
            const bool has_const = cpp_range_has_token(
                tokens, first, index, "const");
            bool mutable_pointer = false;
            for (std::size_t token = first; token < index; ++token) {
                if (tokens[token].text != "*") continue;
                const bool pointer_const = token + 1 < index &&
                    tokens[token + 1].text == "const";
                mutable_pointer = mutable_pointer || !pointer_const;
            }
            const bool mutable_object =
                !is_constexpr && (!has_const || mutable_pointer);
            if (is_extern ||
                ((!inside_type || is_static) && mutable_object)) {
                error = path +
                    ": mutable/extern namespace or inline-static object root";
                return false;
            }
        }
        segment_begin = index + 1;
    }
    return true;
}

static bool project_header_has_forbidden_line_splice(
    const std::string& source)
{
    bool macro_continuation = false;
    std::size_t line_begin = 0;
    while (line_begin < source.size()) {
        const std::size_t newline = source.find('\n', line_begin);
        const std::size_t line_end = newline == std::string::npos
            ? source.size()
            : newline;
        std::size_t content_end = line_end;
        if (content_end > line_begin &&
            source[content_end - 1] == '\r') {
            --content_end;
        }
        const bool spliced = content_end > line_begin &&
            source[content_end - 1] == '\\';
        if (spliced) --content_end;
        std::size_t cursor = line_begin;
        while (cursor < content_end &&
               (source[cursor] == ' ' || source[cursor] == '\t')) {
            ++cursor;
        }
        bool starts_macro = false;
        if (cursor < content_end && source[cursor] == '#') {
            ++cursor;
            while (cursor < content_end &&
                   (source[cursor] == ' ' || source[cursor] == '\t')) {
                ++cursor;
            }
            const std::size_t action_begin = cursor;
            while (cursor < content_end &&
                   cpp_identifier_continue(source[cursor])) {
                ++cursor;
            }
            const std::string action = source.substr(
                action_begin, cursor - action_begin);
            starts_macro = action == "define" || action == "undef";
            if (starts_macro && spliced) {
                while (cursor < content_end &&
                       (source[cursor] == ' ' || source[cursor] == '\t')) {
                    ++cursor;
                }
                const std::size_t name_begin = cursor;
                while (cursor < content_end &&
                       cpp_identifier_continue(source[cursor])) {
                    ++cursor;
                }
                if (name_begin == cursor || cursor == content_end) {
                    return true;
                }
            }
        }
        if (spliced && !starts_macro && !macro_continuation) {
            return true;
        }
        macro_continuation = spliced &&
            (starts_macro || macro_continuation);
        if (!spliced) macro_continuation = false;
        if (newline == std::string::npos) break;
        line_begin = newline + 1;
    }
    return false;
}

static bool audit_project_header_roots(
    const std::string& path,
    const std::string& source,
    const std::set<std::string>& authenticated_names,
    std::string& error)
{
    const std::string masked_source = mask_source_noncode(source, true);
    if (masked_source.find("%:") != std::string::npos ||
        masked_source.find("<%") != std::string::npos ||
        masked_source.find("%>") != std::string::npos ||
        masked_source.find("%:%:") != std::string::npos ||
        project_header_has_forbidden_line_splice(source)) {
        error = path + ": forbidden digraph/token-paste spelling";
        return false;
    }
    std::string token_error;
    const std::vector<CppToken> tokens =
        tokenize_cpp_source(source, token_error);
    if (!token_error.empty()) {
        error = path + ": " + token_error;
        return false;
    }
    if (!audit_controller_macros(
            tokens, authenticated_names, token_error)) {
        error = path + ": " + token_error;
        return false;
    }
    const char* forbidden_environment_roots[] = {
        "__environ", "environ", "getenv", "putenv", "secure_getenv",
        "setenv", "unsetenv",
    };
    for (const char* name : forbidden_environment_roots) {
        for (const CppToken& token : tokens) {
            if (!token.directive && token.kind == CppTokenIdentifier &&
                token.text == name) {
                error = path +
                    ": owned header contains environment authority";
                return false;
            }
        }
    }
    return audit_project_header_scope(
        path, tokens, 0, tokens.size(), false, error);
}

static bool audit_direct_project_headers(
    const std::vector<CppToken>& controller_tokens,
    const std::set<std::string>& authenticated_names,
    std::string& error)
{
    const std::set<std::string> trusted_external_headers = {
        "raygui.h", "raylib.h", "raymath.h",
    };
    std::set<std::string> audited_paths;
    for (std::size_t index = 0; index + 2 < controller_tokens.size(); ++index) {
        if (!controller_tokens[index].directive ||
            controller_tokens[index].text != "#" ||
            controller_tokens[index + 1].text != "include" ||
            controller_tokens[index + 2].kind != CppTokenString ||
            controller_tokens[index + 1].line != controller_tokens[index].line ||
            controller_tokens[index + 2].line != controller_tokens[index].line) {
            continue;
        }
        const std::string literal = controller_tokens[index + 2].text;
        if (literal.size() < 2) {
            error = "invalid quoted direct include";
            return false;
        }
        const std::string path = literal.substr(1, literal.size() - 2);
        if (trusted_external_headers.count(path) != 0U) continue;
        if (path.empty() || path[0] == '/' ||
            path.find("..") != std::string::npos) {
            error = "direct project header path escapes repository: " + path;
            return false;
        }
        std::ifstream input(path.c_str(), std::ios::binary);
        if (!input.good()) {
            error = "quoted direct project header is unreadable: " + path;
            return false;
        }
        if (!audited_paths.insert(path).second) continue;
        const std::string source{
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
        if (!audit_project_header_roots(
                path, source, authenticated_names, error)) {
            return false;
        }
    }
    return !audited_paths.empty();
}

static std::vector<std::size_t> cpp_find_token_sequence_at_depth(
    const std::vector<CppToken>& tokens,
    const std::vector<int>& depth,
    std::size_t begin,
    std::size_t end,
    const std::vector<std::string>& expected,
    int required_depth)
{
    std::vector<std::size_t> all = cpp_find_token_sequence(
        tokens, begin, end, expected);
    std::vector<std::size_t> filtered;
    for (const std::size_t position : all) {
        if (position < depth.size() &&
            depth[position] == required_depth) {
            filtered.push_back(position);
        }
    }
    return filtered;
}

struct CppCallRecord
{
    std::size_t name = 0;
    std::size_t opening = 0;
    std::size_t closing = 0;
    std::string key;
};

static std::vector<CppCallRecord> cpp_calls_named(
    const std::vector<CppToken>& tokens,
    std::size_t begin,
    std::size_t end,
    const std::string& identifier)
{
    std::vector<CppCallRecord> calls;
    const std::size_t limit = std::min(end, tokens.size());
    for (std::size_t index = begin;
         index + 1 < limit;
         ++index) {
        if (tokens[index].directive ||
            tokens[index].kind != CppTokenIdentifier ||
            tokens[index].text != identifier ||
            tokens[index + 1].text != "(") {
            continue;
        }
        const std::size_t closing = cpp_matching_token(
            tokens, index + 1, "(", ")");
        if (closing == std::string::npos || closing >= limit) continue;
        CppCallRecord call;
        call.name = index;
        call.opening = index + 1;
        call.closing = closing;
        call.key = cpp_call_key(tokens, index);
        calls.push_back(call);
    }
    return calls;
}

static bool cpp_statement_range(
    const std::vector<CppToken>& tokens,
    const std::vector<int>& depth,
    std::size_t contained,
    std::size_t scope_begin,
    std::size_t scope_end,
    std::size_t& statement_begin,
    std::size_t& statement_end)
{
    if (contained >= tokens.size() || contained >= depth.size()) return false;
    const int statement_depth = depth[contained];
    statement_begin = contained;
    while (statement_begin > scope_begin) {
        const std::size_t previous = statement_begin - 1;
        if (!tokens[previous].directive &&
            ((depth[previous] == statement_depth &&
              (tokens[previous].text == ";" ||
               tokens[previous].text == "{")) ||
             (depth[previous] == statement_depth + 1 &&
              tokens[previous].text == "}"))) {
            break;
        }
        --statement_begin;
    }
    statement_end = contained;
    while (statement_end < scope_end && statement_end < tokens.size()) {
        if (!tokens[statement_end].directive &&
            depth[statement_end] == statement_depth &&
            tokens[statement_end].text == ";") {
            return true;
        }
        ++statement_end;
    }
    return false;
}

static bool cpp_exact_statement_containing(
    const std::vector<CppToken>& tokens,
    const std::vector<int>& depth,
    std::size_t contained,
    std::size_t scope_begin,
    std::size_t scope_end,
    const std::string& exact_compact,
    std::size_t& statement_begin,
    std::size_t& statement_end)
{
    return cpp_statement_range(
               tokens,
               depth,
               contained,
               scope_begin,
               scope_end,
               statement_begin,
               statement_end) &&
           cpp_compact_tokens(
               tokens, statement_begin, statement_end + 1) ==
           exact_compact;
}

static std::size_t cpp_direct_assignment_count(
    const std::vector<CppToken>& tokens,
    std::size_t begin,
    std::size_t end,
    const std::string& identifier)
{
    std::size_t count = 0;
    const std::size_t limit = std::min(end, tokens.size());
    for (std::size_t index = begin; index + 1 < limit; ++index) {
        if (!tokens[index].directive &&
            tokens[index].text == identifier &&
            (tokens[index + 1].text == "=" ||
             tokens[index + 1].text == "+=" ||
             tokens[index + 1].text == "-=" ||
             tokens[index + 1].text == "*=" ||
             tokens[index + 1].text == "/=" ||
             tokens[index + 1].text == "%=")) {
            ++count;
        }
    }
    return count;
}

static std::size_t cpp_identifier_count(
    const std::vector<CppToken>& tokens,
    std::size_t begin,
    std::size_t end,
    const std::string& identifier)
{
    std::size_t count = 0;
    const std::size_t limit = std::min(end, tokens.size());
    for (std::size_t index = begin; index < limit; ++index) {
        if (!tokens[index].directive &&
            tokens[index].kind == CppTokenIdentifier &&
            tokens[index].text == identifier) {
            ++count;
        }
    }
    return count;
}

static bool cpp_exit_code_occurrences_are_owner_bound(
    const std::vector<CppToken>& tokens,
    const std::vector<int>& depth)
{
    const std::vector<std::size_t> declarations =
        cpp_find_token_sequence(
            tokens,
            0,
            tokens.size(),
            {"int", "controller_exit_code", "=", "0", ";"});
    const std::vector<std::size_t> fatal_writes =
        cpp_find_token_sequence(
            tokens,
            0,
            tokens.size(),
            {"controller_exit_code", "=", "2", ";"});
    const std::vector<std::size_t> cleanup_reads =
        cpp_find_token_sequence(
            tokens,
            0,
            tokens.size(),
            {"cleanup", ".", "exit_code", "=",
             "controller_exit_code", ";"});
    const std::vector<std::size_t> returns =
        cpp_find_token_sequence(
            tokens,
            0,
            tokens.size(),
            {"return", "controller_exit_code", ";"});
    if (declarations.size() != 1 || fatal_writes.size() != 11 ||
        cleanup_reads.size() != 1 || returns.size() != 3 ||
        declarations[0] >= depth.size() || depth[declarations[0]] != 1) {
        return false;
    }

    std::vector<unsigned char> authenticated(tokens.size(), 0U);
    const auto authenticate_statement = [
        &tokens, &depth, &authenticated](
        std::size_t identifier,
        const char* exact_compact) {
        std::size_t statement_begin = 0;
        std::size_t statement_end = 0;
        if (identifier >= authenticated.size() ||
            authenticated[identifier] != 0U ||
            !cpp_exact_statement_containing(
                tokens,
                depth,
                identifier,
                0,
                tokens.size(),
                exact_compact,
                statement_begin,
                statement_end)) {
            return false;
        }
        authenticated[identifier] = 1U;
        return true;
    };

    if (!authenticate_statement(
            declarations[0] + 1,
            "intcontroller_exit_code=0;")) {
        return false;
    }
    for (const std::size_t write : fatal_writes) {
        if (!authenticate_statement(
                write, "controller_exit_code=2;")) {
            return false;
        }
    }
    if (!authenticate_statement(
            cleanup_reads[0] + 4,
            "cleanup.exit_code=controller_exit_code;")) {
        return false;
    }
    for (const std::size_t value_return : returns) {
        if (!authenticate_statement(
                value_return + 1,
                "returncontroller_exit_code;")) {
            return false;
        }
    }

    std::size_t authenticated_count = 0;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (authenticated[index] != 0U) ++authenticated_count;
        if (!tokens[index].directive &&
            tokens[index].kind == CppTokenIdentifier &&
            tokens[index].text == "controller_exit_code" &&
            authenticated[index] == 0U) {
            return false;
        }
    }
    return authenticated_count == 16;
}

static std::size_t cpp_member_path_write_count(
    const std::vector<CppToken>& tokens,
    std::size_t begin,
    std::size_t end,
    const std::vector<std::string>& path)
{
    std::size_t count = 0;
    const std::vector<std::size_t> matches = cpp_find_token_sequence(
        tokens, begin, end, path);
    for (const std::size_t match : matches) {
        std::size_t cursor = match + path.size();
        int bracket_depth = 0;
        for (; cursor < end && cursor < tokens.size(); ++cursor) {
            const std::string& token = tokens[cursor].text;
            if (token == "[" || token == "(") {
                ++bracket_depth;
                continue;
            }
            if (token == "]" || token == ")") {
                if (bracket_depth == 0) break;
                --bracket_depth;
                continue;
            }
            if (bracket_depth != 0) continue;
            if (token == "=" || token == "+=" || token == "-=" ||
                token == "*=" || token == "/=" || token == "%=" ||
                token == "++" || token == "--") {
                ++count;
                break;
            }
            if (token == ";" || token == "," || token == "?" ||
                token == ":" || token == "==" || token == "!=" ||
                token == "<=" || token == ">=" || token == "&&" ||
                token == "||") {
                break;
            }
        }
    }
    return count;
}

static bool cpp_range_has_nested_loop(
    const std::vector<CppToken>& tokens,
    std::size_t begin,
    std::size_t end)
{
    return cpp_range_has_token(tokens, begin, end, "for") ||
           cpp_range_has_token(tokens, begin, end, "while") ||
           cpp_range_has_token(tokens, begin, end, "do");
}

static bool audit_whole_main_runtime_ownership(
    const std::vector<CppToken>& tokens,
    std::string& error)
{
    const std::vector<std::size_t> declarations = cpp_find_token_sequence(
        tokens,
        0,
        tokens.size(),
        {"G1FrameRuntime", "frame_runtime", ";"});
    if (declarations.size() != 1) {
        error = "main must own one direct G1FrameRuntime object";
        return false;
    }
    const std::size_t declaration_name = declarations[0] + 1;
    const char* allowed_call_names[] = {
        "draw_g1_skeleton", "g1_apply_pending_scene_reset",
        "g1_build_accepted_log_row", "g1_build_task7_log_suffix",
        "g1_frame_runtime_reset",
        "g1_frame_transaction_run", "update_g1_camera_from_accepted",
    };
    std::vector<std::pair<std::size_t, std::size_t>> allowed_arguments;
    for (const char* name : allowed_call_names) {
        const std::vector<CppCallRecord> calls = cpp_calls_named(
            tokens, 0, tokens.size(), name);
        for (const CppCallRecord& call : calls) {
            allowed_arguments.push_back(
                std::make_pair(call.opening + 1, call.closing));
        }
    }
    const std::vector<CppCallRecord> candidate_audit_calls =
        cpp_calls_named(tokens, 0, tokens.size(), "write_requested");
    if (candidate_audit_calls.size() > 1) {
        error = "main may own at most one candidate audit call";
        return false;
    }
    if (!candidate_audit_calls.empty()) {
        const CppCallRecord& call = candidate_audit_calls[0];
        if (call.key != ".write_requested" || call.name < 2 ||
            tokens[call.name - 2].text != "candidate_audit_log") {
            error = "main candidate audit must use its exact direct owner";
            return false;
        }
        allowed_arguments.push_back(
            std::make_pair(call.opening + 1, call.closing));
    }
    const std::vector<std::size_t> scene_frame_paths =
        cpp_find_token_sequence(
            tokens,
            0,
            tokens.size(),
            {"frame_runtime", ".", "accepted_state", ".",
             "scene_frame"});
    if (scene_frame_paths.size() != 1) {
        error = "main SceneCycle gate must have one exact runtime path";
        return false;
    }
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index].directive ||
            tokens[index].text != "frame_runtime") {
            continue;
        }
        if (index > 0 &&
            (tokens[index - 1].text == "&" ||
             tokens[index - 1].text == "*")) {
            error = "main may not take or dereference a runtime alias";
            return false;
        }
        bool allowed = index == declaration_name ||
            index == scene_frame_paths[0];
        for (const std::pair<std::size_t, std::size_t>& arguments :
             allowed_arguments) {
            allowed = allowed ||
                (index >= arguments.first && index < arguments.second);
        }
        if (!allowed) {
            error = "unapproved whole-main frame_runtime occurrence/alias";
            return false;
        }
    }
    return true;
}

static void check_structural_main_update_contract(
    const std::string& main_body_raw)
{
    check(main_body_raw.find("%:") == std::string::npos &&
              main_body_raw.find("<%") == std::string::npos &&
              main_body_raw.find("%>") == std::string::npos,
          "raw main body has no directive/brace digraph spelling");
    std::string error;
    const std::vector<CppToken> tokens =
        tokenize_cpp_source(main_body_raw, error);
    check(error.empty() && !tokens.empty() &&
              tokens[0].text == "{",
          error.empty() ? "main body tokenizes" : error.c_str());
    const std::vector<int> depth = cpp_brace_depth_before(tokens);
    const std::vector<std::string> update_condition = {
        "while", "(", "!", "::", "WindowShouldClose", "(", ")",
        "&&", "!", "controller_exit_requested", ")",
    };
    const std::vector<std::size_t> update_loops =
        cpp_find_token_sequence_at_depth(
            tokens,
            depth,
            0,
            tokens.size(),
            update_condition,
            1);
    check(update_loops.size() == 1,
          "main has one global-qualified ordinary outer update loop");
    const std::size_t update = update_loops[0];
    const std::size_t update_open = update + update_condition.size();
    check(update_open < tokens.size() &&
              tokens[update_open].text == "{",
          "outer update condition is followed directly by its body");
    const std::size_t update_close = cpp_matching_token(
        tokens, update_open, "{", "}");
    check(update_close != std::string::npos,
          "outer update body is balanced");
    const int update_depth = depth[update_open] + 1;
    std::string loop_probe_error;
    const std::vector<CppToken> dead_loop_probe = tokenize_cpp_source(
        "{while(true){::g1_frame_transaction_run();}}",
        loop_probe_error);
    check(loop_probe_error.empty() &&
              cpp_range_has_nested_loop(
                  dead_loop_probe, 1, dead_loop_probe.size() - 1) &&
              !cpp_range_has_nested_loop(
                  tokens, update_open + 1, update_close),
          "live update rejects synthetic and real nested-loop reachability wrappers");

    const std::vector<std::size_t> exit_code_declarations =
        cpp_find_token_sequence_at_depth(
            tokens,
            depth,
            0,
            update,
            {"int", "controller_exit_code", "=", "0", ";"},
            1);
    const std::vector<std::size_t> exit_code_tail =
        cpp_find_token_sequence_at_depth(
            tokens,
            depth,
            update_close + 1,
            tokens.size(),
            {"normal_cleanup", "(", ")", ";", "return",
             "controller_exit_code", ";"},
            1);
    const std::size_t loop_exit_code_identifiers =
        cpp_identifier_count(
            tokens,
            update_open + 1,
            update_close,
            "controller_exit_code");
    const std::size_t tail_exit_code_identifiers =
        cpp_identifier_count(
            tokens,
            update_close + 1,
            tokens.size(),
            "controller_exit_code");
    const std::vector<std::size_t> exit_code_zero_writes =
        cpp_find_token_sequence(
            tokens,
            0,
            tokens.size(),
            {"controller_exit_code", "=", "0", ";"});
    const std::vector<std::size_t> exit_code_fatal_writes =
        cpp_find_token_sequence(
            tokens,
            0,
            tokens.size(),
            {"controller_exit_code", "=", "2", ";"});
    const std::vector<std::size_t> exit_code_cleanup_reads =
        cpp_find_token_sequence(
            tokens,
            0,
            tokens.size(),
            {"cleanup", ".", "exit_code", "=",
             "controller_exit_code", ";"});
    const std::vector<std::size_t> exit_code_returns =
        cpp_find_token_sequence(
            tokens,
            0,
            tokens.size(),
            {"return", "controller_exit_code", ";"});
    check(exit_code_declarations.size() == 1 &&
              cpp_exit_code_occurrences_are_owner_bound(
                  tokens, depth) &&
              cpp_identifier_count(
                  tokens, 0, tokens.size(),
                  "controller_exit_code") == 16 &&
              cpp_direct_assignment_count(
                  tokens, 0, tokens.size(),
                  "controller_exit_code") == 12 &&
              exit_code_zero_writes.size() == 1 &&
              exit_code_zero_writes[0] ==
                  exit_code_declarations[0] + 1 &&
              exit_code_fatal_writes.size() == 11 &&
              exit_code_cleanup_reads.size() == 1 &&
              exit_code_returns.size() == 3 &&
              1 + exit_code_fatal_writes.size() +
                      exit_code_cleanup_reads.size() +
                      exit_code_returns.size() ==
                      16 &&
              exit_code_tail.size() == 1 &&
              exit_code_tail[0] == update_close + 1 &&
              exit_code_tail[0] + 7 == tokens.size() - 1 &&
              loop_exit_code_identifiers == 6 &&
              cpp_direct_assignment_count(
                  tokens,
                  update_open + 1,
                  update_close,
                  "controller_exit_code") == 6 &&
              tail_exit_code_identifiers == 1 &&
              exit_code_declarations.size() +
                      loop_exit_code_identifiers +
                      tail_exit_code_identifiers ==
                      8,
          "all exit-status occurrences are authenticated as one initialization, eleven fatal writes, one cleanup report, and three cleanup-ordered returns");

    const std::string exit_write =
        "controller_exit_code = 2;";
    const std::string shadowing_exit_write =
        "int controller_exit_code = 2;";
    std::string shadowed_main = main_body_raw;
    const std::size_t shadowed_write =
        shadowed_main.find(exit_write);
    check(shadowed_write != std::string::npos,
          "exit-code shadow mutation finds a real fatal write");
    shadowed_main.replace(
        shadowed_write,
        exit_write.size(),
        shadowing_exit_write);
    std::string shadow_token_error;
    const std::vector<CppToken> shadow_tokens =
        tokenize_cpp_source(shadowed_main, shadow_token_error);
    const std::vector<int> shadow_depth =
        cpp_brace_depth_before(shadow_tokens);
    check(shadow_token_error.empty() &&
              !cpp_exit_code_occurrences_are_owner_bound(
                  shadow_tokens, shadow_depth),
          "owner-aware exit-code audit rejects a count-preserving local shadow declaration");

    const std::string consumption_block =
        "if(pending_reset){"
        "constintrequested_scene_index=pending_scene_index;"
        "constboolscene_reset_ok=::g1_apply_pending_scene_reset("
        "requested_scene_index,active_scene_index,frame_runtime,"
        "scene_reset_context,"
        "artifact_error,static_cast<int>(sizeof(artifact_error)));"
        "if(!scene_reset_ok){::controlled_runtime_error(artifact_error);"
        "controller_exit_code=2;controller_exit_requested=true;break;}"
        "pending_reset=false;pending_scene_index=-1;}";
    check(update_open + 1 < tokens.size() &&
              tokens[update_open + 1].text == "if" &&
              depth[update_open + 1] == update_depth,
          "pending scene consumption is the first direct update statement");
    const std::size_t consumption_condition_open = update_open + 2;
    const std::size_t consumption_condition_close = cpp_matching_token(
        tokens, consumption_condition_open, "(", ")");
    check(consumption_condition_close != std::string::npos &&
              consumption_condition_close + 1 < tokens.size() &&
              tokens[consumption_condition_close + 1].text == "{",
          "pending scene consumption condition/block is balanced");
    const std::size_t consumption_close = cpp_matching_token(
        tokens, consumption_condition_close + 1, "{", "}");
    check(consumption_close != std::string::npos &&
              cpp_compact_tokens(
                  tokens, update_open + 1, consumption_close + 1) ==
                  consumption_block,
          "pending scene is consumed/reset exactly once before frame sampling");
    const std::vector<CppCallRecord> reset_helper_calls = cpp_calls_named(
        tokens,
        update_open + 1,
        consumption_close,
        "g1_apply_pending_scene_reset");
    const std::vector<std::size_t> pending_clear = cpp_find_token_sequence(
        tokens,
        update_open + 1,
        consumption_close,
        {"pending_reset", "=", "false", ";"});
    const std::vector<std::size_t> pending_index_clear =
        cpp_find_token_sequence(
            tokens,
            update_open + 1,
            consumption_close,
            {"pending_scene_index", "=", "-", "1", ";"});
    check(reset_helper_calls.size() == 1 &&
              pending_clear.size() == 1 &&
              pending_index_clear.size() == 1 &&
              reset_helper_calls[0].closing < pending_clear[0] &&
              pending_clear[0] < pending_index_clear[0],
          "pending reset/switch succeeds before its request flags are cleared");

    const std::vector<CppCallRecord> coordinator_calls =
        cpp_calls_named(
            tokens,
            update_open + 1,
            update_close,
            "g1_frame_transaction_run");
    check(coordinator_calls.size() == 1 &&
              coordinator_calls[0].key ==
                  "::g1_frame_transaction_run" &&
              depth[coordinator_calls[0].name] == update_depth,
          "outer update has exactly one global-qualified top-level coordinator call");
    std::size_t coordinator_statement = 0;
    std::size_t coordinator_statement_end = 0;
    const std::string exact_coordinator =
        "constG1FrameTransactionStatusframe_status="
        "::g1_frame_transaction_run(frame_runtime,"
        "::g1_controller_frame_stage_run,"
        "::g1_recovery_candidates_build,frame_external,"
        "test_seam_pointer,artifact_error,"
        "static_cast<int>(sizeof(artifact_error)));";
    check(cpp_exact_statement_containing(
              tokens,
              depth,
              coordinator_calls[0].name,
              update_open + 1,
              update_close,
              exact_coordinator,
              coordinator_statement,
              coordinator_statement_end),
          "coordinator result is bound once to the exact immutable frame_status declaration and arguments");

    const std::vector<std::string> global_failure_prefix = {
        "if", "(", "frame_status", "==",
        "G1FrameTransactionGlobalError", ")", "{",
    };
    const std::vector<std::size_t> global_failures =
        cpp_find_token_sequence_at_depth(
            tokens,
            depth,
            coordinator_statement_end + 1,
            update_close,
            global_failure_prefix,
            update_depth);
    check(global_failures.size() == 1 &&
              global_failures[0] == coordinator_statement_end + 1,
          "the exact global-error gate immediately follows the coordinator");
    const std::size_t global_failure_close = cpp_matching_token(
        tokens,
        global_failures[0] + global_failure_prefix.size() - 1,
        "{",
        "}");
    const std::string exact_global_failure =
        "if(frame_status==G1FrameTransactionGlobalError){"
        "::controlled_runtime_error(artifact_error);"
        "controller_exit_code=2;controller_exit_requested=true;break;}";
    check(global_failure_close != std::string::npos &&
              cpp_compact_tokens(
                  tokens,
                  global_failures[0],
                  global_failure_close + 1) == exact_global_failure,
          "global coordinator failure performs one controlled fatal exit before all observation");

    const std::vector<CppCallRecord> builder_calls =
        cpp_calls_named(
            tokens,
            global_failure_close + 1,
            update_close,
            "g1_build_accepted_log_row");
    check(builder_calls.size() == 1 &&
              builder_calls[0].key ==
                  "::g1_build_accepted_log_row" &&
              depth[builder_calls[0].name] == update_depth,
          "one top-level accepted log-row builder follows the coordinator");
    std::size_t builder_statement = 0;
    std::size_t builder_statement_end = 0;
    const std::string exact_builder =
        "constboollog_row_ok=::g1_build_accepted_log_row("
        "log_row,frame_runtime.accepted_state,"
        "frame_runtime.accepted_diagnostic,frame_runtime.publication,"
        "log_context,artifact_error,"
        "static_cast<int>(sizeof(artifact_error)));";
    check(cpp_exact_statement_containing(
              tokens,
              depth,
              builder_calls[0].name,
              update_open + 1,
              update_close,
              exact_builder,
              builder_statement,
              builder_statement_end),
          "log builder has exact accepted-owner/context/error arguments and a checked bool result");
    check(builder_statement == global_failure_close + 1,
          "accepted-row construction is the direct global-error fallthrough");

    const std::vector<std::string> row_failure_prefix = {
        "if", "(", "!", "log_row_ok", ")", "{",
    };
    const std::vector<std::size_t> row_failures =
        cpp_find_token_sequence_at_depth(
            tokens,
            depth,
            builder_statement_end + 1,
            update_close,
            row_failure_prefix,
            update_depth);
    check(row_failures.size() == 1,
          "log-row builder result has one direct top-level failure block");
    const std::size_t row_failure_close = cpp_matching_token(
        tokens,
        row_failures[0] + row_failure_prefix.size() - 1,
        "{",
        "}");
    const std::string exact_failure =
        "if(!log_row_ok){::controlled_runtime_error(artifact_error);"
        "controller_exit_code=2;controller_exit_requested=true;break;}";
    check(row_failure_close != std::string::npos &&
              cpp_compact_tokens(
                  tokens, row_failures[0], row_failure_close + 1) ==
                  exact_failure,
          "failed log-row construction performs controlled fatal exit before publication/render");
    check(row_failures[0] == builder_statement_end + 1,
          "the checked builder has no unchecked or alternate fallthrough statement");

    const std::vector<CppCallRecord> suffix_calls = cpp_calls_named(
        tokens,
        row_failure_close + 1,
        update_close,
        "g1_build_task7_log_suffix");
    check(suffix_calls.size() == 1 &&
              suffix_calls[0].key ==
                  "::g1_build_task7_log_suffix" &&
              depth[suffix_calls[0].name] == update_depth,
          "one top-level Task-7 suffix builder follows the base row");
    std::size_t suffix_statement = 0;
    std::size_t suffix_statement_end = 0;
    const std::string exact_suffix =
        "constboollog_suffix_ok=::g1_build_task7_log_suffix("
        "log_row,frame_runtime.accepted_state,"
        "frame_runtime.accepted_diagnostic,frame_runtime.publication,"
        "artifact_error,static_cast<int>(sizeof(artifact_error)));";
    check(cpp_exact_statement_containing(
              tokens,
              depth,
              suffix_calls[0].name,
              update_open + 1,
              update_close,
              exact_suffix,
              suffix_statement,
              suffix_statement_end) &&
              suffix_statement == row_failure_close + 1,
          "Task-7 suffix reads only accepted/publication owners and is direct fallthrough");
    const std::vector<std::string> suffix_failure_prefix = {
        "if", "(", "!", "log_suffix_ok", ")", "{",
    };
    const std::vector<std::size_t> suffix_failures =
        cpp_find_token_sequence_at_depth(
            tokens,
            depth,
            suffix_statement_end + 1,
            update_close,
            suffix_failure_prefix,
            update_depth);
    check(suffix_failures.size() == 1,
          "Task-7 suffix result has one direct failure block");
    const std::size_t suffix_failure_close = cpp_matching_token(
        tokens,
        suffix_failures[0] + suffix_failure_prefix.size() - 1,
        "{",
        "}");
    const std::string exact_suffix_failure =
        "if(!log_suffix_ok){::controlled_runtime_error(artifact_error);"
        "controller_exit_code=2;controller_exit_requested=true;break;}";
    check(suffix_failure_close != std::string::npos &&
              suffix_failures[0] == suffix_statement_end + 1 &&
              cpp_compact_tokens(
                  tokens,
                  suffix_failures[0],
                  suffix_failure_close + 1) == exact_suffix_failure,
          "failed Task-7 suffix construction exits before write/render");

    const std::vector<CppCallRecord> write_calls = cpp_calls_named(
        tokens,
        suffix_failure_close + 1,
        update_close,
        "write");
    check(write_calls.size() == 1 &&
              write_calls[0].key == ".write" &&
              depth[write_calls[0].name] == update_depth &&
              write_calls[0].name >= 2 &&
              tokens[write_calls[0].name - 2].text ==
                  "deterministic_log",
          "one top-level deterministic_log.write follows the checked builder");
    std::size_t write_statement = 0;
    std::size_t write_statement_end = 0;
    const std::string exact_write =
        "constboollog_ok=deterministic_log.write(log_row,artifact_error,"
        "static_cast<int>(sizeof(artifact_error)));";
    check(cpp_exact_statement_containing(
              tokens,
              depth,
              write_calls[0].name,
              update_open + 1,
              update_close,
              exact_write,
              write_statement,
              write_statement_end),
          "deterministic write consumes exactly the constructed row/error buffer and binds its bool result");
    check(write_statement == suffix_failure_close + 1,
          "deterministic write is the direct checked-suffix fallthrough");

    const std::vector<std::string> write_failure_prefix = {
        "if", "(", "!", "log_ok", ")", "{",
    };
    const std::vector<std::size_t> write_failures =
        cpp_find_token_sequence_at_depth(
            tokens,
            depth,
            write_statement_end + 1,
            update_close,
            write_failure_prefix,
            update_depth);
    check(write_failures.size() == 1,
          "deterministic write result has one direct top-level failure block");
    const std::size_t write_failure_close = cpp_matching_token(
        tokens,
        write_failures[0] + write_failure_prefix.size() - 1,
        "{",
        "}");
    const std::string exact_write_failure =
        "if(!log_ok){::controlled_runtime_error(artifact_error);"
        "controller_exit_code=2;controller_exit_requested=true;break;}";
    check(write_failure_close != std::string::npos &&
              cpp_compact_tokens(
                  tokens, write_failures[0], write_failure_close + 1) ==
                  exact_write_failure,
          "failed deterministic write performs controlled fatal exit before scene scheduling/render");
    check(write_failures[0] == write_statement_end + 1,
          "the deterministic write has no unchecked or alternate fallthrough statement");

    const std::vector<CppCallRecord> candidate_audit_calls =
        cpp_calls_named(
            tokens,
            write_failure_close + 1,
            update_close,
            "write_requested");
    check(candidate_audit_calls.size() == 1 &&
              candidate_audit_calls[0].key == ".write_requested" &&
              depth[candidate_audit_calls[0].name] == update_depth &&
              candidate_audit_calls[0].name >= 2 &&
              tokens[candidate_audit_calls[0].name - 2].text ==
                  "candidate_audit_log",
          "one top-level behavior-inert candidate audit follows the normal log write");
    std::size_t candidate_audit_statement = 0;
    std::size_t candidate_audit_statement_end = 0;
    const std::string exact_candidate_audit =
        "constboolcandidate_audit_ok="
        "candidate_audit_log.write_requested("
        "db,frame_runtime.accepted_state,"
        "frame_runtime.accepted_diagnostic,frame_runtime.publication,"
        "frame_status,scene_ids[active_scene_index],"
        "candidate_audit_config.route,candidate_audit_config.heading,"
        "artifact_error,static_cast<int>(sizeof(artifact_error)));";
    check(cpp_exact_statement_containing(
              tokens,
              depth,
              candidate_audit_calls[0].name,
              update_open + 1,
              update_close,
              exact_candidate_audit,
              candidate_audit_statement,
              candidate_audit_statement_end),
          "candidate audit receives only immutable database/accepted/publication owners and frame status");
    check(candidate_audit_statement == write_failure_close + 1,
          "candidate audit is the direct checked deterministic-log fallthrough");

    const std::vector<std::string> candidate_audit_failure_prefix = {
        "if", "(", "!", "candidate_audit_ok", ")", "{",
    };
    const std::vector<std::size_t> candidate_audit_failures =
        cpp_find_token_sequence_at_depth(
            tokens,
            depth,
            candidate_audit_statement_end + 1,
            update_close,
            candidate_audit_failure_prefix,
            update_depth);
    check(candidate_audit_failures.size() == 1,
          "candidate audit result has one direct top-level failure block");
    const std::size_t candidate_audit_failure_close = cpp_matching_token(
        tokens,
        candidate_audit_failures[0] +
            candidate_audit_failure_prefix.size() - 1,
        "{",
        "}");
    const std::string exact_candidate_audit_failure =
        "if(!candidate_audit_ok){"
        "::controlled_runtime_error(artifact_error);"
        "controller_exit_code=2;controller_exit_requested=true;break;}";
    check(candidate_audit_failure_close != std::string::npos &&
              cpp_compact_tokens(
                  tokens,
                  candidate_audit_failures[0],
                  candidate_audit_failure_close + 1) ==
                  exact_candidate_audit_failure,
          "failed candidate audit performs one controlled fatal exit before scene scheduling/render");
    check(candidate_audit_failures[0] ==
              candidate_audit_statement_end + 1,
          "candidate audit has no unchecked or alternate fallthrough statement");

    const std::vector<std::string> scene_prefix = {
        "if", "(", "frame_status", "==",
        "G1FrameTransactionAccepted", "&&", "test_config", ".",
        "mode", "==", "G1_TestSceneCycle", "&&", "frame_runtime",
        ".", "accepted_state", ".", "scene_frame", ">=",
        "test_config", ".", "scene_dwell_frames", ")", "{",
    };
    const std::vector<std::size_t> scene_gates =
        cpp_find_token_sequence_at_depth(
            tokens,
            depth,
            candidate_audit_failure_close + 1,
            update_close,
            scene_prefix,
            update_depth);
    check(scene_gates.size() == 1,
          "one direct top-level accepted >= dwell SceneCycle gate follows checked logging");
    const std::size_t scene_close = cpp_matching_token(
        tokens,
        scene_gates[0] + scene_prefix.size() - 1,
        "{",
        "}");
    const std::string exact_scene =
        "if(frame_status==G1FrameTransactionAccepted&&"
        "test_config.mode==G1_TestSceneCycle&&"
        "frame_runtime.accepted_state.scene_frame>="
        "test_config.scene_dwell_frames){"
        "pending_scene_index=(active_scene_index+1)%scene_count;"
        "pending_reset=true;}";
    check(scene_close != std::string::npos &&
              cpp_compact_tokens(
                  tokens, scene_gates[0], scene_close + 1) ==
                  exact_scene,
          "SceneCycle gate directly schedules exactly one pending switch/reset with no runtime-dead wrapper");
    check(scene_gates[0] == candidate_audit_failure_close + 1,
          "SceneCycle scheduling is the direct checked-audit fallthrough");

    const std::vector<CppCallRecord> camera_calls = cpp_calls_named(
        tokens, scene_close + 1, update_close,
        "update_g1_camera_from_accepted");
    check(camera_calls.size() == 1 &&
              camera_calls[0].key ==
                  "::update_g1_camera_from_accepted" &&
              depth[camera_calls[0].name] == update_depth,
          "one direct accepted-camera update follows scene scheduling");
    std::size_t camera_statement = 0;
    std::size_t camera_statement_end = 0;
    const std::string exact_camera =
        "::update_g1_camera_from_accepted(camera,"
        "frame_runtime.accepted_state.camera_azimuth,"
        "frame_runtime.accepted_state.camera_altitude,"
        "frame_runtime.accepted_state.camera_distance,"
        "frame_runtime.accepted_state.ik_global_bone_positions(0));";
    check(cpp_exact_statement_containing(
              tokens,
              depth,
              camera_calls[0].name,
              update_open + 1,
              update_close,
              exact_camera,
              camera_statement,
              camera_statement_end),
          "camera helper receives exactly accepted camera scalars and final-FK root");
    check(camera_statement == scene_close + 1,
          "accepted camera publication is the direct SceneCycle fallthrough");

    const std::vector<CppCallRecord> begin_calls = cpp_calls_named(
        tokens, update_open + 1, update_close, "BeginDrawing");
    const std::vector<CppCallRecord> end_calls = cpp_calls_named(
        tokens, update_open + 1, update_close, "EndDrawing");
    const std::vector<CppCallRecord> skeleton_calls = cpp_calls_named(
        tokens, update_open + 1, update_close, "draw_g1_skeleton");
    check(begin_calls.size() == 1 && end_calls.size() == 1 &&
              skeleton_calls.size() == 1 &&
              begin_calls[0].key == "::BeginDrawing" &&
              end_calls[0].key == "::EndDrawing" &&
              skeleton_calls[0].key == "::draw_g1_skeleton" &&
              depth[begin_calls[0].name] == update_depth &&
              depth[end_calls[0].name] == update_depth &&
              depth[skeleton_calls[0].name] == update_depth &&
              camera_statement_end < begin_calls[0].name &&
              begin_calls[0].name < skeleton_calls[0].name &&
              skeleton_calls[0].name < end_calls[0].name,
          "one top-level render interval contains one direct accepted skeleton draw after logging/camera");
    check(begin_calls[0].name > 0 &&
              camera_statement_end + 1 == begin_calls[0].name - 1 &&
              tokens[begin_calls[0].name - 1].text == "::" &&
              end_calls[0].closing + 2 == update_close &&
              tokens[end_calls[0].closing + 1].text == ";",
          "the sole render interval is the final direct camera fallthrough and update statement");
    std::size_t skeleton_statement = 0;
    std::size_t skeleton_statement_end = 0;
    const std::string exact_skeleton =
        "::draw_g1_skeleton("
        "frame_runtime.accepted_state.ik_global_bone_positions,"
        "frame_runtime.accepted_state.ik_global_bone_rotations,"
        "frame_external.db->bone_parents);";
    check(cpp_exact_statement_containing(
              tokens,
              depth,
              skeleton_calls[0].name,
              update_open + 1,
              update_close,
              exact_skeleton,
              skeleton_statement,
              skeleton_statement_end),
          "skeleton draw consumes exactly accepted final-FK position/rotation and immutable parents");

    check(cpp_identifier_count(
              tokens, 0, tokens.size(), "BeginDrawing") == 1 &&
              cpp_identifier_count(
                  tokens, 0, tokens.size(), "EndDrawing") == 1 &&
              cpp_identifier_count(
                  tokens, 0, tokens.size(), "draw_g1_skeleton") == 1 &&
              cpp_identifier_count(
                  tokens,
                  coordinator_statement,
                  update_close,
                  "frame_runtime") == 17 &&
              cpp_identifier_count(
                  tokens, 0, tokens.size(), "accepted_state") == 10 &&
              cpp_identifier_count(
                  tokens, 0, tokens.size(), "accepted_diagnostic") == 3 &&
              cpp_identifier_count(
                  tokens, 0, tokens.size(), "publication") == 3,
          "no pre-log/alternate render or accepted-owner alias/write remains in active main");
    check(cpp_identifier_count(
              tokens, update_open + 1, update_close,
              "g1_frame_transaction_run") == 1 &&
              cpp_identifier_count(
                  tokens, update_open + 1, update_close,
                  "g1_controller_frame_stage_run") == 1 &&
              cpp_identifier_count(
                  tokens, update_open + 1, update_close,
                  "frame_status") == 4 &&
              cpp_direct_assignment_count(
                  tokens, update_open + 1, update_close,
                  "frame_status") == 1 &&
              cpp_identifier_count(
                  tokens, update_open + 1, coordinator_statement_end + 1,
                  "frame_runtime") == 2 &&
              cpp_identifier_count(
                  tokens, update_open + 1, builder_statement,
                  "accepted_state") == 0 &&
              cpp_identifier_count(
                  tokens, update_open + 1, builder_statement,
                  "accepted_diagnostic") == 0 &&
              cpp_identifier_count(
                  tokens, update_open + 1, builder_statement,
                  "publication") == 0,
          "live update has one coordinator/runner identity and no pre-builder accepted-owner alias");
    check(cpp_member_path_write_count(
              tokens,
              update_open + 1,
              update_close,
              {"frame_runtime", ".", "accepted_state"}) == 0 &&
              cpp_member_path_write_count(
                  tokens,
                  update_open + 1,
                  update_close,
                  {"frame_runtime", ".", "accepted_diagnostic"}) == 0 &&
              cpp_member_path_write_count(
                  tokens,
                  update_open + 1,
                  update_close,
                  {"frame_runtime", ".", "publication"}) == 0,
          "main never mutates accepted state, accepted diagnostics, or publication directly");
    check(!cpp_range_has_indirect_callable_syntax(
              tokens, update_open + 1, update_close),
          "live update contains no parenthesized/function-pointer dispatch");
    const char* forbidden_update_transfers[] = {
        "continue", "goto", "return", "throw", "try", "catch",
        "co_await", "co_yield", "co_return", "for", "while", "do",
    };
    for (const char* transfer : forbidden_update_transfers) {
        check(cpp_identifier_count(
                  tokens, update_open + 1, update_close,
                  transfer) == 0,
              "live update contains no unapproved control transfer");
    }
    const std::vector<std::size_t> breaks = cpp_find_token_sequence(
        tokens,
        update_open + 1,
        update_close,
        {"break", ";"});
    check(breaks.size() == 6 &&
              breaks[0] < consumption_close &&
              breaks[1] > global_failures[0] &&
              breaks[1] < global_failure_close &&
              breaks[2] > row_failures[0] &&
              breaks[2] < row_failure_close &&
              breaks[3] > suffix_failures[0] &&
              breaks[3] < suffix_failure_close &&
              breaks[4] > write_failures[0] &&
              breaks[4] < write_failure_close &&
              breaks[5] > candidate_audit_failures[0] &&
              breaks[5] < candidate_audit_failure_close,
          "only the six authenticated fatal blocks may break the live update");
    const std::vector<CppCallRecord> begin_mode_calls = cpp_calls_named(
        tokens, update_open + 1, update_close, "BeginMode3D");
    const std::vector<CppCallRecord> end_mode_calls = cpp_calls_named(
        tokens, update_open + 1, update_close, "EndMode3D");
    check(begin_mode_calls.size() == 1 && end_mode_calls.size() == 1 &&
              begin_calls[0].name < begin_mode_calls[0].name &&
              begin_mode_calls[0].name < skeleton_calls[0].name &&
              skeleton_calls[0].name < end_mode_calls[0].name &&
              end_mode_calls[0].name < end_calls[0].name,
          "the sole accepted skeleton render is inside one balanced 3D interval");
    for (std::size_t index = update_open + 1;
         index + 1 < update_close;
         ++index) {
        if (tokens[index].kind == CppTokenIdentifier &&
            tokens[index].text.size() > 4 &&
            tokens[index].text.compare(0, 4, "Draw") == 0 &&
            tokens[index + 1].text == "(") {
            check(begin_calls[0].name < index &&
                      index < end_calls[0].name,
                  "every direct Draw call is owned by the one render interval");
        }
    }
    check(cpp_find_token_sequence_at_depth(
              tokens,
              depth,
              0,
              update,
              {"int", "pending_scene_index", "=", "-", "1", ";"},
              1).size() == 1 &&
              cpp_find_token_sequence_at_depth(
                  tokens,
                  depth,
                  0,
                  update,
                  {"bool", "pending_reset", "=", "false", ";"},
                  1).size() == 1 &&
              cpp_direct_assignment_count(
                  tokens, 0, tokens.size(), "pending_reset") == 3 &&
              cpp_direct_assignment_count(
                  tokens, 0, tokens.size(), "pending_scene_index") == 3 &&
              cpp_identifier_count(
                  tokens, 0, tokens.size(), "pending_reset") == 4 &&
              cpp_identifier_count(
                  tokens, 0, tokens.size(), "pending_scene_index") == 4 &&
              cpp_identifier_count(
                  tokens,
                  update_open + 1,
                  update_close,
                  "active_scene_index") == 3 &&
              cpp_direct_assignment_count(
                  tokens,
                  update_open + 1,
                  update_close,
                  "active_scene_index") == 0,
          "pending/reset/active scene owners have only consumption/schedule writes");
}

struct ExactFunctionRange
{
    std::size_t signature = 0;
    std::size_t body_begin = 0;
    std::size_t body_end = 0;
};

static bool cpp_exact_function_range(
    const std::vector<CppToken>& tokens,
    const std::vector<std::string>& signature,
    ExactFunctionRange& output)
{
    const std::vector<std::size_t> matches = cpp_find_token_sequence(
        tokens, 0, tokens.size(), signature);
    if (matches.size() != 1) return false;
    const std::size_t body = matches[0] + signature.size();
    if (body >= tokens.size() || tokens[body].directive ||
        tokens[body].text != "{") {
        return false;
    }
    const std::size_t close = cpp_matching_token(
        tokens, body, "{", "}");
    if (close == std::string::npos) return false;
    output.signature = matches[0];
    output.body_begin = body;
    output.body_end = close;
    return true;
}

static std::size_t cpp_member_assignment_count(
    const std::vector<CppToken>& tokens,
    std::size_t begin,
    std::size_t end,
    const std::string& object,
    const std::string& field)
{
    std::size_t count = 0;
    const std::vector<std::size_t> members = cpp_find_token_sequence(
        tokens, begin, end, {object, ".", field});
    for (const std::size_t member : members) {
        std::size_t cursor = member + 3;
        if (cursor < end && cursor < tokens.size() &&
            tokens[cursor].text == "[") {
            const std::size_t close = cpp_matching_token(
                tokens, cursor, "[", "]");
            if (close == std::string::npos || close >= end) continue;
            cursor = close + 1;
        }
        if (cursor < end && cursor < tokens.size() &&
            tokens[cursor].text == "=") {
            ++count;
        }
    }
    return count;
}

static std::size_t cpp_member_mutation_count(
    const std::vector<CppToken>& tokens,
    std::size_t begin,
    std::size_t end,
    const std::string& object,
    const std::string& field)
{
    std::size_t count = 0;
    const std::vector<std::size_t> members = cpp_find_token_sequence(
        tokens, begin, end, {object, ".", field});
    const std::set<std::string> mutations = {
        "=", "+=", "-=", "*=", "/=", "%=", "++", "--",
        "<<=", ">>=",
    };
    for (const std::size_t member : members) {
        if (member > begin &&
            (tokens[member - 1].text == "++" ||
             tokens[member - 1].text == "--")) {
            ++count;
            continue;
        }
        std::size_t cursor = member + 3;
        if (cursor < end && cursor < tokens.size() &&
            tokens[cursor].text == "[") {
            const std::size_t close = cpp_matching_token(
                tokens, cursor, "[", "]");
            if (close == std::string::npos || close >= end) continue;
            cursor = close + 1;
        }
        if (cursor < end && cursor < tokens.size() &&
            mutations.count(tokens[cursor].text) != 0U) {
            ++count;
        } else if (cursor + 1 < end &&
                   (tokens[cursor].text == "&" ||
                    tokens[cursor].text == "|" ||
                    tokens[cursor].text == "^") &&
                   tokens[cursor + 1].text == "=") {
            ++count;
        }
    }
    return count;
}

static void check_structural_main_helper_contracts(
    const std::vector<CppToken>& tokens,
    const NoMainSourceView& no_main_view)
{
    const std::vector<std::string> candidate_parser_signature = {
        "static", "bool", "g1_candidate_audit_config_parse", "(",
        "G1CandidateAuditConfig", "&", "output", ",",
        "const", "char", "*", "path", ",",
        "g1_test_mode", "mode", ",", "int", "frame_limit", ",",
        "const", "char", "*", "mode_name", ",",
        "const", "char", "*", "route", ",",
        "const", "char", "*", "heading", ",",
        "char", "*", "error", ",", "int", "error_capacity", ")",
    };
    ExactFunctionRange candidate_parser;
    check(cpp_exact_function_range(
              tokens, candidate_parser_signature, candidate_parser) &&
              position_is_inside_no_main_guard(
                  no_main_view,
                  tokens[candidate_parser.signature].begin) &&
              cpp_identifier_count(
                  tokens,
                  0,
                  tokens.size(),
                  "g1_candidate_audit_config_parse") == 2,
          "candidate audit parser has one exact main-only bounded-mode/const-text signature");
    const std::vector<std::string> candidate_writer_signature = {
        "bool", "write_requested", "(",
        "const", "database", "&", "db", ",",
        "const", "g1_controller_state", "&", "accepted_state", ",",
        "const", "G1FrameAcceptedDiagnostic", "&",
        "accepted_diagnostic", ",",
        "const", "G1FramePublication", "&", "publication", ",",
        "G1FrameTransactionStatus", "frame_status", ",",
        "const", "char", "*", "scene_id", ",",
        "const", "char", "*", "route", ",",
        "const", "char", "*", "heading", ",",
        "char", "*", "error", ",", "int", "error_capacity", ")",
    };
    ExactFunctionRange candidate_writer;
    const std::vector<std::size_t> candidate_log_structs =
        cpp_find_token_sequence(
            tokens,
            0,
            tokens.size(),
            {"struct", "G1CandidateAuditLog", "{"});
    const std::size_t candidate_log_close =
        candidate_log_structs.size() == 1
        ? cpp_matching_token(
              tokens, candidate_log_structs[0] + 2, "{", "}")
        : std::string::npos;
    check(cpp_exact_function_range(
              tokens, candidate_writer_signature, candidate_writer) &&
              candidate_log_structs.size() == 1 &&
              candidate_log_close != std::string::npos &&
              candidate_writer.signature > candidate_log_structs[0] + 2 &&
              candidate_writer.body_end < candidate_log_close &&
              position_is_inside_no_main_guard(
                  no_main_view,
                  tokens[candidate_writer.signature].begin) &&
              cpp_identifier_count(
                  tokens,
                  0,
                  tokens.size(),
                  "write_requested") == 2,
          "candidate audit writer is inside one exact no-base owner struct with one const definition and one production call");
    std::string mutation_probe_error;
    const std::vector<CppToken> mutation_probe = tokenize_cpp_source(
        "{log_row.frame=value;log_row.frame+=1;"
        "camera.position=value;++camera.position;}",
        mutation_probe_error);
    check(mutation_probe_error.empty() &&
              cpp_member_assignment_count(
                  mutation_probe,
                  0,
                  mutation_probe.size(),
                  "log_row",
                  "frame") == 1 &&
              cpp_member_mutation_count(
                  mutation_probe,
                  0,
                  mutation_probe.size(),
                  "log_row",
                  "frame") == 2 &&
              cpp_member_assignment_count(
                  mutation_probe,
                  0,
                  mutation_probe.size(),
                  "camera",
                  "position") == 1 &&
              cpp_member_mutation_count(
                  mutation_probe,
                  0,
                  mutation_probe.size(),
                  "camera",
                  "position") == 2,
          "helper ownership audit distinguishes exact assignment from compound/prefix mutation");
    const std::vector<std::string> builder_signature = {
        "static", "bool", "g1_build_accepted_log_row", "(",
        "motion_match_log_row", "&", "log_row", ",",
        "const", "g1_controller_state", "&", "accepted_state", ",",
        "const", "G1FrameAcceptedDiagnostic", "&",
        "accepted_diagnostic", ",",
        "const", "G1FramePublication", "&", "publication", ",",
        "const", "G1AcceptedLogContext", "&", "log_context", ",",
        "char", "*", "error", ",", "int", "error_capacity", ")",
    };
    ExactFunctionRange builder;
    check(cpp_exact_function_range(
              tokens, builder_signature, builder) &&
              position_is_inside_no_main_guard(
                  no_main_view, tokens[builder.signature].begin),
          "accepted log builder has one exact const-owner/main-only signature");
    const char* log_fields[] = {
        "frame", "fixed_dt", "scene_id", "mode", "route",
        "query_bits_hex", "query_database_frame", "query_range",
        "selected_database_frame", "database_frame", "range",
        "source_range", "searched", "transitioned", "incumbent_cost",
        "selected_cost", "selected_terrain_error",
        "effective_terrain_weight", "terrain", "terrain_points",
        "raw_selected", "inertialized", "rendered",
        "hips_inertial_offset_y", "runtime_root_surface_height",
        "runtime_left_toe_surface_height",
        "runtime_right_toe_surface_height", "adjustment_xz",
        "adjustment_y", "clamp_xz", "clamp_y", "matching_enabled",
        "adjustment_enabled", "clamping_enabled",
        "support_retargeting_enabled", "ik_enabled", "source_name",
        "source_terrain", "source_index", "continuation_cost",
        "source_root_height", "source_left_toe_height",
        "source_right_toe_height", "runtime_support_root_height",
        "runtime_support_left_toe_height",
        "runtime_support_right_toe_height", "support_root_delta",
        "support_left_toe_delta", "support_right_toe_delta",
        "support_height", "support_velocity", "support_source",
        "airborne_frames", "left_contact", "right_contact",
        "support_retargeted_hips_y", "ik_adjusted_hips_y",
        "simulation_x", "simulation_z", "walkability_class",
        "blocked", "blocked_reason", "blocked_distance",
        "blocked_point_x", "blocked_point_z", "commanded_speed",
        "applied_speed", "route_waypoint", "route_complete",
        "route_target_height", "scene_generation", "scene_frame",
        "scene_reset_count", "scene_switch_failed",
        "motion_pack_load_count", "model_load_count",
        "model_unload_count", "live_model_count",
    };
    for (const char* field : log_fields) {
        check(cpp_member_assignment_count(
                  tokens,
                  builder.body_begin + 1,
                  builder.body_end,
                  "log_row",
                  field) == 1 &&
                  cpp_member_mutation_count(
                      tokens,
                      builder.body_begin + 1,
                      builder.body_end,
                      "log_row",
                      field) == 1,
              "accepted log builder materializes every row field exactly once");
    }
    const std::string builder_body = cpp_compact_tokens(
        tokens, builder.body_begin + 1, builder.body_end);
    const char* exact_owner_materializations[] = {
        "log_row.frame=publication.presentation_frame;",
        "log_row.query_database_frame=effective_query_database_frame;",
        "log_row.query_range=canonical_rejection_baseline?query_range:accepted_diagnostic.query_range;",
        "log_row.selected_database_frame=effective_selected_database_frame;",
        "log_row.database_frame=accepted_state.frame_index;",
        "log_row.searched=accepted_state.searched;",
        "log_row.transitioned=accepted_state.transitioned;",
        "log_row.incumbent_cost=accepted_state.incumbent_cost;",
        "log_row.selected_cost=accepted_state.selected_cost;",
        "log_row.selected_terrain_error=accepted_state.selected_terrain_error;",
        "log_row.effective_terrain_weight=accepted_diagnostic.effective_terrain_weight;",
        "log_row.raw_selected=accepted_diagnostic.raw_selected;",
        "log_row.inertialized=accepted_diagnostic.inertialized;",
        "log_row.rendered=accepted_diagnostic.rendered;",
        "log_row.matching_enabled=accepted_diagnostic.matching_enabled;",
        "log_row.adjustment_enabled=accepted_diagnostic.adjustment_enabled;",
        "log_row.clamping_enabled=accepted_diagnostic.clamping_enabled;",
        "log_row.ik_enabled=log_context.ik_enabled;",
        "log_row.scene_frame=accepted_diagnostic.scene_frame;",
    };
    for (const char* materialization : exact_owner_materializations) {
        check(source_occurrence_count(
                  builder_body, materialization) == 1,
              "accepted log builder binds core row evidence to exact accepted owners");
    }
    const char* exact_provenance_bindings[] = {
        "constboolcanonical_rejection_baseline=!accepted_diagnostic.ready;",
        "constinteffective_query_database_frame=canonical_rejection_baseline?accepted_state.frame_index:accepted_diagnostic.query_database_frame;",
        "constinteffective_selected_database_frame=canonical_rejection_baseline?accepted_state.frame_index:accepted_diagnostic.selected_database_frame;",
        "if(canonical_rejection_baseline&&!publication.rejection.rejected)",
    };
    for (const char* binding : exact_provenance_bindings) {
        check(source_occurrence_count(builder_body, binding) == 1,
              "accepted log builder authenticates its canonical rejection baseline and effective provenance once");
    }
    check(cpp_find_token_sequence(
              tokens,
              builder.body_begin + 1,
              builder.body_end,
              {"if", "(", "false", ")"}).empty() &&
              cpp_find_token_sequence(
                  tokens,
                  builder.body_begin + 1,
                  builder.body_end,
                  {"while", "(", "false", ")"}).empty(),
          "accepted log materialization contains no literal runtime-dead evidence block");
    check(builder_body.size() >= std::string("returntrue;").size() &&
              builder_body.compare(
                  builder_body.size() - std::string("returntrue;").size(),
                  std::string("returntrue;").size(),
                  "returntrue;") == 0 &&
              cpp_find_token_sequence(
                  tokens,
                  builder.body_begin + 1,
                  builder.body_end,
                  {"return", "true", ";"}).size() == 1 &&
              cpp_find_token_sequence(
                  tokens,
                  builder.body_begin + 1,
                  builder.body_end,
                  {"return", "false", ";"}).empty(),
          "accepted log builder has one final live success and no literal false return");
    const std::vector<CppCallRecord> builder_scene_errors =
        cpp_calls_named(
            tokens,
            builder.body_begin + 1,
            builder.body_end,
            "scene_error");
    const std::vector<CppCallRecord> builder_support_names =
        cpp_calls_named(
            tokens,
            builder.body_begin + 1,
            builder.body_end,
            "support_source_name");
    const std::vector<CppCallRecord> builder_walkability_names =
        cpp_calls_named(
            tokens,
            builder.body_begin + 1,
            builder.body_end,
            "walkability_reason_name");
    const auto calls_are_global = [](const std::vector<CppCallRecord>& calls) {
        for (const CppCallRecord& call : calls) {
            if (call.key.empty() || call.key.substr(0, 2) != "::") {
                return false;
            }
        }
        return true;
    };
    check(builder_scene_errors.size() == 4 &&
              builder_support_names.size() == 1 &&
              builder_walkability_names.size() == 1 &&
              calls_are_global(builder_scene_errors) &&
              calls_are_global(builder_support_names) &&
              calls_are_global(builder_walkability_names),
          "accepted log builder has four controlled provenance failures and only authoritative global name serializers");
    const std::set<std::string> allowed_builder_calls = {
        "scene_error", "support_source_name", "walkability_reason_name",
    };
    for (std::size_t index = builder.body_begin + 1;
         index + 1 < builder.body_end;
         ++index) {
        if (tokens[index].kind == CppTokenIdentifier &&
            tokens[index + 1].text == "(" &&
            tokens[index].text != "for" &&
            tokens[index].text != "if" &&
            tokens[index].text != "sizeof" &&
            tokens[index].text != "static_cast" &&
            allowed_builder_calls.count(tokens[index].text) == 0U) {
            check(false,
                  "accepted log builder performs only authenticated provenance/name calls");
        }
    }
    std::set<std::string> builder_locals;
    const std::set<std::string> declaration_types = {
        "auto", "bool", "double", "float", "int", "long",
        "short", "size_t", "unsigned",
    };
    for (std::size_t index = builder.body_begin + 1;
         index + 1 < builder.body_end;
         ++index) {
        if (declaration_types.count(tokens[index].text) != 0U &&
            tokens[index + 1].kind == CppTokenIdentifier) {
            builder_locals.insert(tokens[index + 1].text);
        }
    }
    const std::set<std::string> builder_roots = {
        "accepted_diagnostic", "accepted_state", "error",
        "error_capacity", "false", "for", "if", "int", "log_context",
        "log_row", "nullptr", "publication", "return", "sizeof", "static_cast",
        "true", "void",
    };
    for (std::size_t index = builder.body_begin + 1;
         index < builder.body_end;
         ++index) {
        if (tokens[index].kind != CppTokenIdentifier) continue;
        const bool member = index > builder.body_begin + 1 &&
            (tokens[index - 1].text == "." ||
             tokens[index - 1].text == "->" ||
             tokens[index - 1].text == "::");
        if (member || tokens[index].text == "const" ||
            builder_roots.count(tokens[index].text) != 0U ||
            builder_locals.count(tokens[index].text) != 0U ||
            declaration_types.count(tokens[index].text) != 0U) {
            continue;
        }
        check(false,
              "accepted log builder reads only parameters and body-local indices");
    }

    const std::vector<std::string> camera_signature = {
        "static", "void", "update_g1_camera_from_accepted", "(",
        "Camera3D", "&", "camera", ",",
        "const", "float", "azimuth", ",",
        "const", "float", "altitude", ",",
        "const", "float", "distance", ",",
        "const", "vec3", "&", "target", ")",
    };
    ExactFunctionRange camera;
    check(cpp_exact_function_range(tokens, camera_signature, camera) &&
              position_is_inside_no_main_guard(
                  no_main_view, tokens[camera.signature].begin),
          "accepted camera helper has exact const scalar/root inputs in main-only code");
    check(cpp_member_assignment_count(
              tokens,
              camera.body_begin + 1,
              camera.body_end,
              "camera",
              "target") == 1 &&
              cpp_member_assignment_count(
                  tokens,
                  camera.body_begin + 1,
                  camera.body_end,
                  "camera",
                  "position") == 1 &&
              cpp_member_mutation_count(
                  tokens,
                  camera.body_begin + 1,
                  camera.body_end,
                  "camera",
                  "target") == 1 &&
              cpp_member_mutation_count(
                  tokens,
                  camera.body_begin + 1,
                  camera.body_end,
                  "camera",
                  "position") == 1,
          "accepted camera helper is the single direct target/position writer");
    const std::string camera_body = cpp_compact_tokens(
        tokens, camera.body_begin + 1, camera.body_end);
    check(source_occurrence_count(
              camera_body,
              "camera.target=::to_Vector3(focus);") == 1 &&
              source_occurrence_count(
                  camera_body,
                  "camera.position=::to_Vector3(focus+offset);") == 1 &&
              camera_body.find("constvec3focus=target") !=
                  std::string::npos &&
              camera_body.find("constvec3offset=") !=
                  std::string::npos &&
              camera_body.find("distance") != std::string::npos &&
              camera_body.find("::sinf(azimuth)") !=
                  std::string::npos &&
              camera_body.find("::cosf(azimuth)") !=
                  std::string::npos &&
              camera_body.find("::sinf(altitude)") !=
                  std::string::npos &&
              camera_body.find("::cosf(altitude)") !=
                  std::string::npos &&
              cpp_identifier_count(
                  tokens,
                  camera.body_begin + 1,
                  camera.body_end,
                  "if") == 0 &&
              cpp_identifier_count(
                  tokens,
                  camera.body_begin + 1,
                  camera.body_end,
                  "for") == 0 &&
              cpp_identifier_count(
                  tokens,
                  camera.body_begin + 1,
                  camera.body_end,
                  "while") == 0,
          "camera outputs are direct dataflow from all accepted inputs without dead/alternate paths");
    const std::set<std::string> camera_calls_allowed = {
        "::cosf", "::sinf", "::to_Vector3", "vec3",
    };
    for (std::size_t index = camera.body_begin + 1;
         index + 1 < camera.body_end;
         ++index) {
        if (tokens[index].kind == CppTokenIdentifier &&
            tokens[index + 1].text == "(") {
            check(camera_calls_allowed.count(
                      cpp_call_key(tokens, index)) == 1U,
                  "camera helper has no helper indirection or untrusted call");
        }
    }
    const std::set<std::string> camera_root_names = {
        "altitude", "azimuth", "camera", "const", "cosf", "distance",
        "float", "focus", "offset", "sinf", "target", "to_Vector3",
        "vec3",
    };
    for (std::size_t index = camera.body_begin + 1;
         index < camera.body_end;
         ++index) {
        check(!tokens[index].directive,
              "camera helper has no preprocessor branch");
        if (tokens[index].kind != CppTokenIdentifier) continue;
        const bool member = index > camera.body_begin + 1 &&
            (tokens[index - 1].text == "." ||
             tokens[index - 1].text == "->" ||
             tokens[index - 1].text == "::");
        if (!member) {
            check(camera_root_names.count(tokens[index].text) != 0U,
                  "camera helper reads only its parameters and two local vectors");
        }
    }
    check(cpp_identifier_count(
              tokens, camera.body_begin + 1, camera.body_end,
              "return") == 0 &&
              cpp_member_assignment_count(
                  tokens, 0, tokens.size(), "camera", "target") == 1 &&
              cpp_member_assignment_count(
                  tokens, 0, tokens.size(), "camera", "position") == 1 &&
              cpp_member_mutation_count(
                  tokens, 0, tokens.size(), "camera", "target") == 1 &&
              cpp_member_mutation_count(
                  tokens, 0, tokens.size(), "camera", "position") == 1,
          "camera target/position have one controller-wide owner and no early return");

    const std::vector<std::string> skeleton_signature = {
        "static", "void", "draw_g1_skeleton", "(",
        "const", "array1d", "<", "vec3", ">", "&", "positions", ",",
        "const", "array1d", "<", "quat", ">", "&", "rotations", ",",
        "const", "array1d", "<", "int", ">", "&", "parents", ")",
    };
    ExactFunctionRange skeleton;
    check(cpp_exact_function_range(
              tokens, skeleton_signature, skeleton) &&
              position_is_inside_no_main_guard(
                  no_main_view, tokens[skeleton.signature].begin),
          "skeleton helper has exact immutable pose/parent inputs in main-only code");
    const std::vector<CppCallRecord> sphere_calls = cpp_calls_named(
        tokens, skeleton.body_begin + 1, skeleton.body_end,
        "DrawSphereWires");
    const std::vector<CppCallRecord> cylinder_calls = cpp_calls_named(
        tokens, skeleton.body_begin + 1, skeleton.body_end,
        "DrawCylinderEx");
    const std::vector<CppCallRecord> line_calls = cpp_calls_named(
        tokens, skeleton.body_begin + 1, skeleton.body_end,
        "DrawLine3D");
    check(sphere_calls.size() == 1 && cylinder_calls.size() == 1 &&
              line_calls.size() == 1 &&
              sphere_calls[0].key == "::DrawSphereWires" &&
              cylinder_calls[0].key == "::DrawCylinderEx" &&
              line_calls[0].key == "::DrawLine3D",
          "skeleton helper directly performs one joint/bone/orientation draw family");
    const std::string sphere_call = cpp_compact_tokens(
        tokens, sphere_calls[0].name, sphere_calls[0].closing + 1);
    const std::string cylinder_call = cpp_compact_tokens(
        tokens, cylinder_calls[0].name, cylinder_calls[0].closing + 1);
    const std::string line_call = cpp_compact_tokens(
        tokens, line_calls[0].name, line_calls[0].closing + 1);
    check(sphere_call.rfind(
              "DrawSphereWires(::to_Vector3(positions(bone)),", 0) == 0 &&
              cylinder_call.rfind(
                  "DrawCylinderEx(::to_Vector3(positions(parents(bone))),"
                  "::to_Vector3(positions(bone)),",
                  0) == 0 &&
              line_call.find(
                  "::to_Vector3(positions(bone))") !=
                  std::string::npos &&
              line_call.find(
                  "::quat_mul_vec3(rotations(bone)") !=
                  std::string::npos,
          "skeleton draw calls use their exact immutable position/rotation arguments");
    const std::set<std::string> skeleton_calls_allowed = {
        "::DrawCylinderEx", "::DrawLine3D", "::DrawSphereWires",
        "::quat_mul_vec3", "::to_Vector3", "parents", "positions",
        "rotations", "vec3",
    };
    for (std::size_t index = skeleton.body_begin + 1;
         index + 1 < skeleton.body_end;
         ++index) {
        if (tokens[index].kind == CppTokenIdentifier &&
            tokens[index + 1].text == "(" &&
            tokens[index].text != "for" && tokens[index].text != "if") {
            check(skeleton_calls_allowed.count(
                      cpp_call_key(tokens, index)) == 1U,
                  "skeleton helper has no pose-draw helper indirection");
        }
    }
    const std::vector<int> skeleton_depth = cpp_brace_depth_before(tokens);
    const std::vector<std::size_t> skeleton_loops =
        cpp_find_token_sequence_at_depth(
            tokens,
            skeleton_depth,
            skeleton.body_begin + 1,
            skeleton.body_end,
            {"for", "("},
            skeleton_depth[skeleton.body_begin] + 1);
    check(skeleton_loops.size() == 1,
          "skeleton helper has one direct parent/bone loop");
    const std::size_t loop_condition_close = cpp_matching_token(
        tokens, skeleton_loops[0] + 1, "(", ")");
    check(loop_condition_close != std::string::npos &&
              loop_condition_close + 1 < skeleton.body_end &&
              tokens[loop_condition_close + 1].text == "{" &&
              cpp_identifier_count(
                  tokens,
                  skeleton_loops[0],
                  loop_condition_close,
                  "bone") >= 3,
          "skeleton loop directly owns one bone cursor and balanced body");
    const std::size_t skeleton_loop_close = cpp_matching_token(
        tokens, loop_condition_close + 1, "{", "}");
    check(skeleton_loop_close != std::string::npos &&
              sphere_calls[0].name > loop_condition_close &&
              sphere_calls[0].name < skeleton_loop_close &&
              cylinder_calls[0].name > loop_condition_close &&
              cylinder_calls[0].name < skeleton_loop_close &&
              line_calls[0].name > loop_condition_close &&
              line_calls[0].name < skeleton_loop_close,
          "all three direct skeleton draw families are inside the sole bone loop");
    const char* skeleton_transfers[] = {
        "break", "continue", "goto", "return", "throw",
    };
    for (const char* transfer : skeleton_transfers) {
        check(cpp_identifier_count(
                  tokens,
                  skeleton.body_begin + 1,
                  skeleton.body_end,
                  transfer) == 0,
              "skeleton helper has no early/unsupported transfer");
    }
    const std::set<std::string> skeleton_authenticated_allowed = {
        "DrawCylinderEx", "DrawLine3D", "DrawSphereWires",
        "parents", "positions", "quat_mul_vec3", "rotations", "vec3",
    };
    for (const std::string& name : production_authenticated_source_names()) {
        if (skeleton_authenticated_allowed.count(name) != 0U) continue;
        check(cpp_identifier_count(
                  tokens,
                  skeleton.body_begin + 1,
                  skeleton.body_end,
                  name) == 0,
              "skeleton helper has no authenticated global/controller owner read");
    }
    check(cpp_calls_named(
              tokens, 0, tokens.size(), "DrawSphereWires").size() == 1 &&
              cpp_calls_named(
                  tokens, 0, tokens.size(), "DrawCylinderEx").size() == 1 &&
              cpp_calls_named(
                  tokens, 0, tokens.size(), "DrawLine3D").size() == 1,
          "skeleton helper exclusively owns its three controller-wide draw APIs");
    check(cpp_find_token_sequence(
              tokens,
              skeleton.body_begin + 1,
              skeleton.body_end,
              {"if", "(", "false", ")"}).empty(),
          "skeleton evidence cannot be nested in a literal dead branch");

    const std::vector<std::string> reset_helper_prefix = {
        "static", "bool", "g1_apply_pending_scene_reset", "(",
        "const", "int", "requested_scene_index", ",",
        "int", "&", "active_scene_index", ",",
        "G1FrameRuntime", "&", "frame_runtime", ",",
    };
    const std::vector<std::size_t> reset_helper_matches =
        cpp_find_token_sequence(
            tokens, 0, tokens.size(), reset_helper_prefix);
    check(reset_helper_matches.size() == 1,
          "pending reset helper has one exact ownership-signature prefix");
    const std::size_t reset_context_type =
        reset_helper_matches[0] + reset_helper_prefix.size();
    const std::vector<std::string> reset_helper_suffix = {
        "&", "scene_reset_context", ",", "char", "*", "error", ",",
        "int", "error_capacity", ")",
    };
    check(reset_context_type < tokens.size() &&
              tokens[reset_context_type].kind == CppTokenIdentifier &&
              cpp_token_sequence_at(
                  tokens, reset_context_type + 1, reset_helper_suffix),
          "pending reset helper receives one named mutable context and exact error sink");
    const std::size_t reset_body_begin =
        reset_context_type + 1 + reset_helper_suffix.size();
    check(reset_body_begin < tokens.size() &&
              tokens[reset_body_begin].text == "{",
          "pending reset helper signature is followed directly by its body");
    const std::size_t reset_body_end = cpp_matching_token(
        tokens, reset_body_begin, "{", "}");
    check(reset_body_end != std::string::npos &&
              position_is_inside_no_main_guard(
                  no_main_view,
                  tokens[reset_helper_matches[0]].begin),
          "pending reset helper is one balanced main-only definition");
    const std::vector<CppCallRecord> reset_current_calls = cpp_calls_named(
        tokens,
        reset_body_begin + 1,
        reset_body_end,
        "scene_reset_current");
    const std::vector<CppCallRecord> switch_calls = cpp_calls_named(
        tokens,
        reset_body_begin + 1,
        reset_body_end,
        "scene_switch_transaction");
    check(reset_current_calls.size() == 1 &&
              switch_calls.size() == 1 &&
              reset_current_calls[0].key == "::scene_reset_current" &&
              switch_calls[0].key == "::scene_switch_transaction",
          "pending reset helper has one direct current-reset and one direct scene-switch operation");
    const std::string reset_helper_body = cpp_compact_tokens(
        tokens, reset_body_begin + 1, reset_body_end);
    check(reset_helper_body.rfind(
              "if(requested_scene_index==active_scene_index){"
              "return::scene_reset_current(",
              0) == 0 &&
              source_occurrence_count(
                  reset_helper_body,
                  "return::scene_reset_current(") == 1 &&
              source_occurrence_count(
                  reset_helper_body,
                  "return::scene_switch_transaction(") == 1 &&
              cpp_find_token_sequence(
                  tokens,
                  reset_body_begin + 1,
                  reset_body_end,
                  {"return", "false", ";"}).empty() &&
              cpp_find_token_sequence(
                  tokens,
                  reset_body_begin + 1,
                  reset_body_end,
                  {"return", "true", ";"}).empty(),
          "pending reset helper directly returns exactly one selected atomic operation");
    for (std::size_t index = reset_body_begin + 1;
         index + 1 < reset_body_end;
         ++index) {
        if (tokens[index].kind != CppTokenIdentifier ||
            tokens[index + 1].text != "(" ||
            tokens[index].text == "if") {
            continue;
        }
        const std::string key = cpp_call_key(tokens, index);
        check(key == "::scene_reset_current" ||
                  key == "::scene_switch_transaction",
              "pending reset helper invokes no wrapper or alternate operation");
    }
    check(cpp_direct_assignment_count(
              tokens,
              reset_body_begin + 1,
              reset_body_end,
              "requested_scene_index") == 0 &&
              cpp_direct_assignment_count(
                  tokens,
                  reset_body_begin + 1,
                  reset_body_end,
                  "active_scene_index") == 0 &&
              cpp_direct_assignment_count(
                  tokens,
                  reset_body_begin + 1,
                  reset_body_end,
                  "frame_runtime") == 0 &&
              cpp_direct_assignment_count(
                  tokens,
                  reset_body_begin + 1,
                  reset_body_end,
                  "scene_reset_context") == 0 &&
              cpp_identifier_count(
                  tokens,
                  reset_current_calls[0].opening + 1,
                  reset_current_calls[0].closing,
                  "frame_runtime") >= 1 &&
              cpp_identifier_count(
                  tokens,
                  switch_calls[0].opening + 1,
                  switch_calls[0].closing,
                  "frame_runtime") >= 1 &&
              cpp_identifier_count(
                  tokens,
                  switch_calls[0].opening + 1,
                  switch_calls[0].closing,
                  "requested_scene_index") >= 1 &&
              cpp_identifier_count(
                  tokens,
                  switch_calls[0].opening + 1,
                  switch_calls[0].closing,
                  "active_scene_index") >= 1,
          "pending reset helper forwards exact owners without direct repair writes");
}

static bool audit_parser_common_body(
    const std::vector<CppToken>& tokens,
    const ExactFunctionRange& parser,
    std::string& error)
{
    if (parser.body_begin >= parser.body_end ||
        parser.body_end >= tokens.size()) {
        error = "parser body range is invalid";
        return false;
    }
    if (cpp_range_has_indirect_callable_syntax(
            tokens, parser.body_begin + 1, parser.body_end)) {
        error = "parser contains indirect callable syntax";
        return false;
    }
    const std::set<std::string> forbidden = {
        "asm", "break", "catch", "continue", "co_await",
        "co_return", "co_yield", "do", "for", "goto", "longjmp",
        "setjmp", "switch", "throw", "try", "while",
    };
    for (std::size_t index = parser.body_begin + 1;
         index < parser.body_end;
         ++index) {
        if (tokens[index].directive) {
            error = "parser contains a preprocessor branch";
            return false;
        }
        if (forbidden.count(tokens[index].text) != 0U) {
            error = "parser contains unsupported control transfer";
            return false;
        }
        if (tokens[index].text == "[") {
            const std::size_t close = cpp_matching_token(
                tokens, index, "[", "]");
            if (close != std::string::npos &&
                close + 1 < parser.body_end &&
                (tokens[close + 1].text == "{" ||
                 tokens[close + 1].text == "(")) {
                error = "parser contains a local callable";
                return false;
            }
        }
    }
    return true;
}

static bool cpp_call_is_unconditional_statement(
    const std::vector<CppToken>& tokens,
    const std::vector<int>& depth,
    const CppCallRecord& call,
    std::size_t scope_begin,
    std::size_t scope_end,
    int required_depth)
{
    if (call.name >= depth.size() || depth[call.name] != required_depth ||
        call.closing + 1 >= scope_end ||
        tokens[call.closing + 1].text != ";") {
        return false;
    }
    std::size_t expected_begin = call.name;
    if (call.key == "::snprintf") {
        if (call.name == 0 || tokens[call.name - 1].text != "::") {
            return false;
        }
        expected_begin = call.name - 1;
    } else if (call.key == "std::snprintf") {
        if (call.name < 2 || tokens[call.name - 1].text != "::" ||
            tokens[call.name - 2].text != "std") {
            return false;
        }
        expected_begin = call.name - 2;
    } else {
        return false;
    }
    std::size_t statement_begin = 0;
    std::size_t statement_end = 0;
    return cpp_statement_range(
               tokens,
               depth,
               call.name,
               scope_begin,
               scope_end,
               statement_begin,
               statement_end) &&
           statement_begin == expected_begin &&
           statement_end == call.closing + 1;
}

static bool audit_float_parser_contract(
    const std::vector<CppToken>& tokens,
    const ExactFunctionRange& parser,
    bool search_time,
    std::string& error)
{
    if (!audit_parser_common_body(tokens, parser, error)) return false;
    const std::vector<int> depth = cpp_brace_depth_before(tokens);
    const int body_depth = depth[parser.body_begin] + 1;
    const std::vector<std::string> null_prefix = {
        "if", "(", "text", "==", "NULL", ")", "{",
    };
    const std::vector<std::size_t> null_blocks =
        cpp_find_token_sequence_at_depth(
            tokens,
            depth,
            parser.body_begin + 1,
            parser.body_end,
            null_prefix,
            body_depth);
    if (null_blocks.size() != 1 ||
        null_blocks[0] != parser.body_begin + 1) {
        error = "float parser null branch is not the exact first statement";
        return false;
    }
    const std::size_t null_close = cpp_matching_token(
        tokens,
        null_blocks[0] + null_prefix.size() - 1,
        "{",
        "}");
    const std::string expected_null = search_time
        ? "if(text==NULL){output=0.10f;returntrue;}"
        : "if(text==NULL){returntrue;}";
    if (null_close == std::string::npos ||
        cpp_compact_tokens(
            tokens, null_blocks[0], null_close + 1) != expected_null) {
        error = "float parser null/default behavior is not exact";
        return false;
    }
    const std::vector<std::string> parse_prefix = {
        "errno", "=", "0", ";",
        "char", "*", "end", "=", "NULL", ";",
        "const", "float", "parsed", "=", "::", "strtof", "(",
        "text", ",", "&", "end", ")", ";",
    };
    if (!cpp_token_sequence_at(tokens, null_close + 1, parse_prefix)) {
        error = "float parser does not directly reset errno and bind strtof";
        return false;
    }
    const std::size_t invalid_begin = null_close + 1 + parse_prefix.size();
    const std::vector<std::string> invalid_prefix_std = {
        "if", "(", "end", "==", "text", "||", "*", "end", "!=",
        "'\\0'", "||", "errno", "==", "ERANGE", "||", "!", "std",
        "::", "isfinite", "(", "parsed", ")", "||", "parsed",
        search_time ? "<" : "<=", "0.0f", "||", "parsed", ">",
        "10.0f", ")", "{",
    };
    const std::vector<std::string> invalid_prefix_global = {
        "if", "(", "end", "==", "text", "||", "*", "end", "!=",
        "'\\0'", "||", "errno", "==", "ERANGE", "||", "!", "::",
        "isfinite", "(", "parsed", ")", "||", "parsed",
        search_time ? "<" : "<=", "0.0f", "||", "parsed", ">",
        "10.0f", ")", "{",
    };
    const bool std_finite = cpp_token_sequence_at(
        tokens, invalid_begin, invalid_prefix_std);
    const bool global_finite = cpp_token_sequence_at(
        tokens, invalid_begin, invalid_prefix_global);
    if (std_finite == global_finite) {
        error = "float parser validation predicate is not exact";
        return false;
    }
    const std::size_t invalid_open = invalid_begin +
        (std_finite
             ? invalid_prefix_std.size()
             : invalid_prefix_global.size()) - 1;
    const std::size_t invalid_close = cpp_matching_token(
        tokens, invalid_open, "{", "}");
    if (invalid_close == std::string::npos ||
        invalid_close + 8 != parser.body_end ||
        !cpp_token_sequence_at(
            tokens,
            invalid_close + 1,
            {"output", "=", "parsed", ";", "return", "true", ";"})) {
        error = "float parser success tail is not one final output commit";
        return false;
    }
    if (cpp_compact_tokens(
            tokens, invalid_open + 1, invalid_close).size() <
            std::string("returnfalse;").size() ||
        cpp_compact_tokens(
            tokens,
            invalid_close - 3,
            invalid_close) != "returnfalse;") {
        error = "float parser invalid branch does not end in failure";
        return false;
    }
    const std::vector<std::size_t> invalid_returns =
        cpp_find_token_sequence(
            tokens,
            invalid_open + 1,
            invalid_close,
            {"return", "false", ";"});
    std::size_t invalid_return_begin = 0;
    std::size_t invalid_return_end = 0;
    if (invalid_returns.size() != 1 ||
        depth[invalid_returns[0]] != depth[invalid_open] + 1 ||
        !cpp_exact_statement_containing(
            tokens,
            depth,
            invalid_returns[0],
            invalid_open + 1,
            invalid_close,
            "returnfalse;",
            invalid_return_begin,
            invalid_return_end)) {
        error = "float parser failure return is not unconditional";
        return false;
    }
    const std::size_t expected_output_assignments = search_time ? 2U : 1U;
    if (cpp_direct_assignment_count(
            tokens, parser.body_begin + 1, parser.body_end, "output") !=
            expected_output_assignments ||
        cpp_find_token_sequence(
            tokens,
            parser.body_begin + 1,
            parser.body_end,
            {"return", "true", ";"}).size() != 2 ||
        cpp_find_token_sequence(
            tokens,
            parser.body_begin + 1,
            parser.body_end,
            {"return", "false", ";"}).size() != 1 ||
        cpp_identifier_count(
            tokens, parser.body_begin + 1, parser.body_end, "errno") != 2 ||
        cpp_identifier_count(
            tokens, parser.body_begin + 1, parser.body_end, "end") != 4 ||
        cpp_identifier_count(
            tokens, parser.body_begin + 1, parser.body_end, "parsed") != 5) {
        error = "float parser has alternate assignments, returns, or parse dataflow";
        return false;
    }
    const std::vector<CppCallRecord> strtof_calls = cpp_calls_named(
        tokens, parser.body_begin + 1, parser.body_end, "strtof");
    const std::vector<CppCallRecord> finite_calls = cpp_calls_named(
        tokens, parser.body_begin + 1, parser.body_end, "isfinite");
    const std::vector<CppCallRecord> error_writes = cpp_calls_named(
        tokens, parser.body_begin + 1, parser.body_end, "snprintf");
    if (strtof_calls.size() != 1 ||
        strtof_calls[0].key != "::strtof" ||
        finite_calls.size() != 1 ||
        (finite_calls[0].key != "std::isfinite" &&
         finite_calls[0].key != "::isfinite") ||
        error_writes.size() != 1 ||
        (error_writes[0].key != "::snprintf" &&
         error_writes[0].key != "std::snprintf") ||
        error_writes[0].name <= invalid_open ||
        error_writes[0].closing >= invalid_close ||
        !cpp_call_is_unconditional_statement(
            tokens,
            depth,
            error_writes[0],
            invalid_open + 1,
            invalid_close,
            depth[invalid_open] + 1) ||
        cpp_identifier_count(
            tokens,
            error_writes[0].opening + 1,
            error_writes[0].closing,
            "error") != 1 ||
        cpp_identifier_count(
            tokens,
            error_writes[0].opening + 1,
            error_writes[0].closing,
            "error_capacity") != 1) {
        error = "float parser conversion/finite/error calls are not direct";
        return false;
    }
    for (std::size_t index = parser.body_begin + 1;
         index + 1 < parser.body_end;
         ++index) {
        if (tokens[index].kind != CppTokenIdentifier ||
            tokens[index + 1].text != "(" ||
            tokens[index].text == "if") {
            continue;
        }
        const std::string key = cpp_call_key(tokens, index);
        if (key != "::strtof" && key != "std::isfinite" &&
            key != "::isfinite" && key != "::snprintf" &&
            key != "std::snprintf") {
            error = "float parser calls an unauthenticated helper";
            return false;
        }
    }
    return true;
}

static bool audit_binary_parser_contract(
    const std::vector<CppToken>& tokens,
    const ExactFunctionRange& parser,
    std::string& error)
{
    if (!audit_parser_common_body(tokens, parser, error)) return false;
    const std::vector<int> depth = cpp_brace_depth_before(tokens);
    const std::string body = cpp_compact_tokens(
        tokens, parser.body_begin + 1, parser.body_end);
    const std::string null_block =
        "if(text==NULL){output=false;returntrue;}";
    const std::string zero_block =
        "if(::strcmp(text,\"0\")==0){output=false;returntrue;}";
    const std::string one_block =
        "if(::strcmp(text,\"1\")==0){output=true;returntrue;}";
    if (body.rfind(null_block + zero_block + one_block, 0) != 0 ||
        body.size() < std::string("returnfalse;").size() ||
        body.compare(
            body.size() - std::string("returnfalse;").size(),
            std::string("returnfalse;").size(),
            "returnfalse;") != 0) {
        error = "binary parser null/0/1/failure flow is not exact";
        return false;
    }
    if (cpp_direct_assignment_count(
            tokens, parser.body_begin + 1, parser.body_end, "output") != 3 ||
        cpp_find_token_sequence(
            tokens,
            parser.body_begin + 1,
            parser.body_end,
            {"return", "true", ";"}).size() != 3 ||
        cpp_find_token_sequence(
            tokens,
            parser.body_begin + 1,
            parser.body_end,
            {"return", "false", ";"}).size() != 1) {
        error = "binary parser has alternate output assignments or returns";
        return false;
    }
    const std::vector<CppCallRecord> comparisons = cpp_calls_named(
        tokens, parser.body_begin + 1, parser.body_end, "strcmp");
    const std::vector<CppCallRecord> error_writes = cpp_calls_named(
        tokens, parser.body_begin + 1, parser.body_end, "snprintf");
    const std::vector<std::size_t> failure_returns =
        cpp_find_token_sequence(
            tokens,
            parser.body_begin + 1,
            parser.body_end,
            {"return", "false", ";"});
    std::size_t failure_return_begin = 0;
    std::size_t failure_return_end = 0;
    if (comparisons.size() != 2 ||
        comparisons[0].key != "::strcmp" ||
        comparisons[1].key != "::strcmp" ||
        cpp_compact_tokens(
            tokens,
            comparisons[0].name,
            comparisons[0].closing + 1) != "strcmp(text,\"0\")" ||
        cpp_compact_tokens(
            tokens,
            comparisons[1].name,
            comparisons[1].closing + 1) != "strcmp(text,\"1\")" ||
        error_writes.size() != 1 ||
        (error_writes[0].key != "::snprintf" &&
         error_writes[0].key != "std::snprintf") ||
        !cpp_call_is_unconditional_statement(
            tokens,
            depth,
            error_writes[0],
            parser.body_begin + 1,
            parser.body_end,
            depth[parser.body_begin] + 1) ||
        failure_returns.size() != 1 ||
        depth[failure_returns[0]] != depth[parser.body_begin] + 1 ||
        !cpp_exact_statement_containing(
            tokens,
            depth,
            failure_returns[0],
            parser.body_begin + 1,
            parser.body_end,
            "returnfalse;",
            failure_return_begin,
            failure_return_end) ||
        cpp_identifier_count(
            tokens,
            error_writes[0].opening + 1,
            error_writes[0].closing,
            "error") != 1 ||
        cpp_identifier_count(
            tokens,
            error_writes[0].opening + 1,
            error_writes[0].closing,
            "error_capacity") != 1) {
        error = "binary parser comparisons/error write are not exact direct checks";
        return false;
    }
    for (std::size_t index = parser.body_begin + 1;
         index + 1 < parser.body_end;
         ++index) {
        if (tokens[index].kind != CppTokenIdentifier ||
            tokens[index + 1].text != "(" ||
            tokens[index].text == "if") {
            continue;
        }
        const std::string key = cpp_call_key(tokens, index);
        if (key != "::strcmp" && key != "::snprintf" &&
            key != "std::snprintf") {
            error = "binary parser calls an unauthenticated helper";
            return false;
        }
    }
    return true;
}

static void check_structural_startup_and_input_contract(
    const std::vector<CppToken>& controller_tokens,
    const NoMainSourceView& no_main_view,
    const std::string& main_body_raw)
{
    const std::vector<std::vector<std::string>> parser_signatures = {
        {"static", "bool", "g1_parse_search_time", "(",
         "float", "&", "output", ",", "const", "char", "*", "text",
         ",", "char", "*", "error", ",", "int", "error_capacity", ")"},
        {"static", "bool", "g1_parse_ik_enabled", "(",
         "bool", "&", "output", ",", "const", "char", "*", "text",
         ",", "char", "*", "error", ",", "int", "error_capacity", ")"},
        {"static", "bool", "g1_parse_halflife", "(",
         "float", "&", "output", ",", "const", "char", "*", "text",
         ",", "char", "*", "error", ",", "int", "error_capacity", ")"},
        {"static", "bool", "g1_parse_strafe_enabled", "(",
         "bool", "&", "output", ",", "const", "char", "*", "text",
         ",", "char", "*", "error", ",", "int", "error_capacity", ")"},
    };
    std::vector<ExactFunctionRange> parsers(parser_signatures.size());
    for (std::size_t index = 0; index < parser_signatures.size(); ++index) {
        check(cpp_exact_function_range(
                  controller_tokens,
                  parser_signatures[index],
                  parsers[index]) &&
                  position_is_inside_no_main_guard(
                      no_main_view,
                      controller_tokens[parsers[index].signature].begin),
              "each startup parser has one exact main-only output/const-text/error signature");
    }
    const auto float_parser_probe = [](
        const std::string& body, bool search_time) {
        std::string probe_error;
        const std::vector<CppToken> probe_tokens =
            tokenize_cpp_source(body, probe_error);
        if (!probe_error.empty() || probe_tokens.size() < 2 ||
            probe_tokens.front().text != "{" ||
            probe_tokens.back().text != "}") {
            return false;
        }
        ExactFunctionRange probe;
        probe.body_begin = 0;
        probe.body_end = probe_tokens.size() - 1;
        return audit_float_parser_contract(
            probe_tokens, probe, search_time, probe_error);
    };
    const std::string valid_search_parser =
        "{if(text==NULL){output=0.10f;return true;}"
        "errno=0;char* end=NULL;"
        "const float parsed=::strtof(text,&end);"
        "if(end==text||*end!='\\0'||errno==ERANGE||"
        "!std::isfinite(parsed)||parsed<0.0f||parsed>10.0f){"
        "::snprintf(error,error_capacity,\"invalid\");return false;}"
        "output=parsed;return true;}";
    const std::string valid_halflife_parser =
        "{if(text==NULL){return true;}errno=0;char* end=NULL;"
        "const float parsed=::strtof(text,&end);"
        "if(end==text||*end!='\\0'||errno==ERANGE||"
        "!::isfinite(parsed)||parsed<=0.0f||parsed>10.0f){"
        "::snprintf(error,error_capacity,\"invalid\");return false;}"
        "output=parsed;return true;}";
    check(float_parser_probe(valid_search_parser, true) &&
              float_parser_probe(valid_halflife_parser, false),
          "float parser audit accepts only the exact full-string bounded contracts");
    std::string missing_erange = valid_search_parser;
    const std::size_t erange_begin = missing_erange.find(
        "||errno==ERANGE");
    check(erange_begin != std::string::npos,
          "float parser ERANGE mutation probe is well formed");
    missing_erange.erase(
        erange_begin, std::string("||errno==ERANGE").size());
    std::string early_output = valid_search_parser;
    const std::size_t errno_begin = early_output.find("errno=0;");
    check(errno_begin != std::string::npos,
          "float parser premature-commit mutation probe is well formed");
    early_output.insert(errno_begin, "output=1.0f;");
    std::string wrapped_conversion = valid_search_parser;
    const std::size_t conversion_begin = wrapped_conversion.find(
        "::strtof(text,&end)");
    check(conversion_begin != std::string::npos,
          "float parser wrapper mutation probe is well formed");
    wrapped_conversion.replace(
        conversion_begin,
        std::string("::strtof(text,&end)").size(),
        "parse_float(text,&end)");
    std::string dead_error_write = valid_search_parser;
    const std::string live_error_write =
        "::snprintf(error,error_capacity,\"invalid\");";
    const std::size_t error_write_begin = dead_error_write.find(
        live_error_write);
    check(error_write_begin != std::string::npos,
          "float parser dead-error mutation probe is well formed");
    dead_error_write.replace(
        error_write_begin,
        live_error_write.size(),
        "if(false){" + live_error_write + "}");
    check(!float_parser_probe(missing_erange, true) &&
              !float_parser_probe(early_output, true) &&
              !float_parser_probe(wrapped_conversion, true) &&
              !float_parser_probe(dead_error_write, true),
          "float parser audit rejects missing ERANGE, early commit, wrapped conversion, and dead error writes");

    const auto binary_parser_probe = [](
        const std::string& body) {
        std::string probe_error;
        const std::vector<CppToken> probe_tokens =
            tokenize_cpp_source(body, probe_error);
        if (!probe_error.empty() || probe_tokens.size() < 2 ||
            probe_tokens.front().text != "{" ||
            probe_tokens.back().text != "}") {
            return false;
        }
        ExactFunctionRange probe;
        probe.body_begin = 0;
        probe.body_end = probe_tokens.size() - 1;
        return audit_binary_parser_contract(
            probe_tokens, probe, probe_error);
    };
    const std::string valid_binary_parser =
        "{if(text==NULL){output=false;return true;}"
        "if(::strcmp(text,\"0\")==0){output=false;return true;}"
        "if(::strcmp(text,\"1\")==0){output=true;return true;}"
        "::snprintf(error,error_capacity,\"invalid\");return false;}";
    std::string permissive_binary_parser = valid_binary_parser;
    const std::size_t binary_failure = permissive_binary_parser.find(
        "::snprintf");
    check(binary_failure != std::string::npos,
          "binary parser mutation probe is well formed");
    permissive_binary_parser.insert(
        binary_failure,
        "if(::strcmp(text,\"2\")==0){output=true;return true;}");
    std::string dead_binary_error = valid_binary_parser;
    const std::size_t binary_error_begin = dead_binary_error.find(
        live_error_write);
    check(binary_error_begin != std::string::npos,
          "binary parser dead-error mutation probe is well formed");
    dead_binary_error.replace(
        binary_error_begin,
        live_error_write.size(),
        "if(false){" + live_error_write + "}");
    check(binary_parser_probe(valid_binary_parser) &&
              !binary_parser_probe(permissive_binary_parser) &&
              !binary_parser_probe(dead_binary_error),
          "binary parser audit accepts exact null/0/1 and rejects permissive/dead-error variants");

    std::string parser_contract_error;
    check(audit_float_parser_contract(
              controller_tokens,
              parsers[0],
              true,
              parser_contract_error),
          parser_contract_error.empty()
              ? "search-time parser has exact null/full-string/finite/[0,10] semantics"
              : parser_contract_error.c_str());
    parser_contract_error.clear();
    check(audit_binary_parser_contract(
              controller_tokens,
              parsers[1],
              parser_contract_error),
          parser_contract_error.empty()
              ? "IK parser has exact null/0/1 semantics"
              : parser_contract_error.c_str());
    parser_contract_error.clear();
    check(audit_float_parser_contract(
              controller_tokens,
              parsers[2],
              false,
              parser_contract_error),
          parser_contract_error.empty()
              ? "halflife parser has exact null/full-string/finite/positive bounded semantics"
              : parser_contract_error.c_str());
    parser_contract_error.clear();
    check(audit_binary_parser_contract(
              controller_tokens,
              parsers[3],
              parser_contract_error),
          parser_contract_error.empty()
              ? "strafe parser has exact null/0/1 semantics"
              : parser_contract_error.c_str());
    const std::vector<CppCallRecord> search_strtof = cpp_calls_named(
        controller_tokens,
        parsers[0].body_begin + 1,
        parsers[0].body_end,
        "strtof");
    const std::vector<CppCallRecord> halflife_strtof = cpp_calls_named(
        controller_tokens,
        parsers[2].body_begin + 1,
        parsers[2].body_end,
        "strtof");
    check(search_strtof.size() == 1 && halflife_strtof.size() == 1 &&
              search_strtof[0].key == "::strtof" &&
              halflife_strtof[0].key == "::strtof" &&
              cpp_compact_tokens(
                  controller_tokens,
                  search_strtof[0].name,
                  search_strtof[0].closing + 1) ==
                  "strtof(text,&end)" &&
              cpp_compact_tokens(
                  controller_tokens,
                  halflife_strtof[0].name,
                  halflife_strtof[0].closing + 1) ==
                  "strtof(text,&end)",
          "float parsers call only exact global-qualified strtof(text,&end)");
    for (const std::size_t parser_index : {std::size_t(1), std::size_t(3)}) {
        const std::vector<CppCallRecord> comparisons = cpp_calls_named(
            controller_tokens,
            parsers[parser_index].body_begin + 1,
            parsers[parser_index].body_end,
            "strcmp");
        check(comparisons.size() == 2 &&
                  comparisons[0].key == "::strcmp" &&
                  comparisons[1].key == "::strcmp" &&
                  cpp_compact_tokens(
                      controller_tokens,
                      comparisons[0].name,
                      comparisons[0].closing + 1) ==
                      "strcmp(text,\"0\")" &&
                  cpp_compact_tokens(
                      controller_tokens,
                      comparisons[1].name,
                      comparisons[1].closing + 1) ==
                      "strcmp(text,\"1\")",
              "binary parsers use exact global-qualified strcmp for only 0/1");
    }

    std::string error;
    const std::vector<CppToken> main_tokens =
        tokenize_cpp_source(main_body_raw, error);
    check(error.empty() && !main_tokens.empty(),
          error.empty() ? "main startup tokenizes" : error.c_str());
    const std::vector<int> depth = cpp_brace_depth_before(main_tokens);
    const std::vector<std::size_t> candidate_log_owners =
        cpp_find_token_sequence_at_depth(
            main_tokens,
            depth,
            0,
            main_tokens.size(),
            {"G1CandidateAuditLog", "candidate_audit_log", ";"},
            1);
    check(candidate_log_owners.size() == 1 &&
              cpp_identifier_count(
                  main_tokens,
                  0,
                  main_tokens.size(),
                  "candidate_audit_log") == 4,
          "main has one exact candidate-log owner consumed only by close, open, and the authenticated write call");
    const std::vector<CppCallRecord> candidate_parser_calls =
        cpp_calls_named(
            main_tokens,
            0,
            main_tokens.size(),
            "g1_candidate_audit_config_parse");
    check(candidate_parser_calls.size() == 1 &&
              candidate_parser_calls[0].key ==
                  "::g1_candidate_audit_config_parse" &&
              cpp_compact_tokens(
                  main_tokens,
                  candidate_parser_calls[0].name,
                  candidate_parser_calls[0].closing + 1) ==
                  "g1_candidate_audit_config_parse("
                  "candidate_audit_config,candidate_audit_environment,"
                  "test_config.mode,test_config.frame_limit,"
                  "test_config.mode_name.c_str(),"
                  "test_config.route.c_str(),"
                  "test_config.test_heading.empty()?nullptr:"
                  "test_config.test_heading.c_str(),artifact_error,"
                  "static_cast<int>(sizeof(artifact_error)))",
          "main binds the sole candidate parser to the exact environment and immutable bounded-mode arguments");
    const std::vector<std::string> startup_begin_pattern = {
        "float", "parsed_initial_search_time", "=", "0.10f", ";",
    };
    const std::vector<std::size_t> startup_begins =
        cpp_find_token_sequence_at_depth(
            main_tokens,
            depth,
            0,
            main_tokens.size(),
            startup_begin_pattern,
            1);
    check(startup_begins.size() == 1,
          "startup parsing begins once at main top level");
    const std::vector<std::string> process_config_pattern = {
        "const", "G1ProcessConfig", "process_config", "{",
        "parsed_initial_search_time", ",", "parsed_ik_enabled", ",",
        "parsed_inertialize_blending_halflife", ",",
        "parsed_simulation_rotation_halflife", ",",
        "parsed_desired_strafe", "}", ";",
    };
    const std::vector<std::size_t> process_configs =
        cpp_find_token_sequence_at_depth(
            main_tokens,
            depth,
            startup_begins[0],
            main_tokens.size(),
            process_config_pattern,
            1);
    check(process_configs.size() == 1,
          "five checked process controls and one inert audit path freeze before process_config");
    const std::size_t startup_end =
        process_configs[0] + process_config_pattern.size();
    const std::string exact_startup =
        "floatparsed_initial_search_time=0.10f;"
        "constboolinitial_search_time_ok=::g1_parse_search_time("
        "parsed_initial_search_time,::getenv(\"MM_SEARCHT\"),startup_error,"
        "static_cast<int>(sizeof(startup_error)));"
        "if(!initial_search_time_ok){::controlled_runtime_error(startup_error);return1;}"
        "boolparsed_ik_enabled=false;"
        "constboolik_enabled_ok=::g1_parse_ik_enabled("
        "parsed_ik_enabled,::getenv(\"MM_IK\"),startup_error,"
        "static_cast<int>(sizeof(startup_error)));"
        "if(!ik_enabled_ok){::controlled_runtime_error(startup_error);return1;}"
        "floatparsed_inertialize_blending_halflife=0.10f;"
        "constboolinertialize_halflife_ok=::g1_parse_halflife("
        "parsed_inertialize_blending_halflife,::getenv(\"MM_HALFLIFE\"),"
        "startup_error,static_cast<int>(sizeof(startup_error)));"
        "if(!inertialize_halflife_ok){::controlled_runtime_error(startup_error);return1;}"
        "floatparsed_simulation_rotation_halflife=0.27f;"
        "constboolsimulation_halflife_ok=::g1_parse_halflife("
        "parsed_simulation_rotation_halflife,::getenv(\"MM_SIMROT_HL\"),"
        "startup_error,static_cast<int>(sizeof(startup_error)));"
        "if(!simulation_halflife_ok){::controlled_runtime_error(startup_error);return1;}"
        "boolparsed_desired_strafe=false;"
        "constbooldesired_strafe_ok=::g1_parse_strafe_enabled("
        "parsed_desired_strafe,::getenv(\"MM_STRAFE\"),startup_error,"
        "static_cast<int>(sizeof(startup_error)));"
        "if(!desired_strafe_ok){::controlled_runtime_error(startup_error);return1;}"
        "constchar*constcandidate_audit_environment="
        "::getenv(\"MM_CANDIDATE_AUDIT\");"
        "constG1ProcessConfigprocess_config{parsed_initial_search_time,"
        "parsed_ik_enabled,parsed_inertialize_blending_halflife,"
        "parsed_simulation_rotation_halflife,parsed_desired_strafe};";
    check(cpp_compact_tokens(
              main_tokens, startup_begins[0], startup_end) ==
              exact_startup,
          "each direct getenv is nested in one checked parser result before immutable process_config construction");

    const char* environment_names[] = {
        "\"MM_SEARCHT\"", "\"MM_IK\"", "\"MM_HALFLIFE\"",
        "\"MM_SIMROT_HL\"", "\"MM_STRAFE\"",
        "\"MM_CANDIDATE_AUDIT\"",
    };
    const std::vector<CppCallRecord> main_getenv = cpp_calls_named(
        main_tokens, 0, main_tokens.size(), "getenv");
    const std::vector<CppCallRecord> all_getenv = cpp_calls_named(
        controller_tokens, 0, controller_tokens.size(), "getenv");
    check(main_getenv.size() == 6 && all_getenv.size() == 6 &&
              cpp_identifier_count(
                  controller_tokens, 0, controller_tokens.size(),
                  "getenv") == 6,
          "controller has exactly six direct process-environment reads");
    for (std::size_t index = 0; index < main_getenv.size(); ++index) {
        check(main_getenv[index].key == "::getenv" &&
                  cpp_compact_tokens(
                      main_tokens,
                      main_getenv[index].opening + 1,
                      main_getenv[index].closing) ==
                      environment_names[index] &&
                  main_getenv[index].name >= startup_begins[0] &&
                  main_getenv[index].closing < startup_end,
              "each environment key is one exact global-qualified startup call in declared order");
    }
    const char* forbidden_environment_apis[] = {
        "__environ", "environ", "putenv", "secure_getenv", "setenv",
        "unsetenv",
    };
    for (const char* api : forbidden_environment_apis) {
        check(cpp_identifier_count(
                  controller_tokens, 0, controller_tokens.size(), api) == 0,
              "controller has no alternate process-environment read/write path");
    }
    const std::vector<std::string> update_condition = {
        "while", "(", "!", "::", "WindowShouldClose", "(", ")",
        "&&", "!", "controller_exit_requested", ")",
    };
    const std::vector<std::size_t> updates =
        cpp_find_token_sequence_at_depth(
            main_tokens,
            depth,
            0,
            main_tokens.size(),
            update_condition,
            1);
    check(updates.size() == 1 && startup_end < updates[0],
          "all process environment reads and freezing precede the update loop");
    const std::size_t update_open =
        updates[0] + update_condition.size();
    check(update_open < main_tokens.size() &&
              main_tokens[update_open].text == "{",
          "startup audit finds the exact ordinary update body");
    const std::size_t update_close = cpp_matching_token(
        main_tokens, update_open, "{", "}");
    check(update_close != std::string::npos,
          "startup audit finds a balanced ordinary update body");
    const int update_depth = depth[update_open] + 1;
    const std::vector<CppCallRecord> resets = cpp_calls_named(
        main_tokens, 0, updates[0], "g1_frame_runtime_reset");
    const std::vector<CppCallRecord> init_windows = cpp_calls_named(
        main_tokens, 0, updates[0], "InitWindow");
    check(resets.size() == 1 && init_windows.size() == 1 &&
              startup_end < resets[0].name &&
              startup_end < init_windows[0].name,
          "immutable process_config is established before reset and Raylib startup");

    const std::vector<std::string> lowerings[] = {
        {"initial_search_time", "=", "process_config", ".", "initial_search_time", ";"},
        {"ik_enabled", "=", "process_config", ".", "ik_enabled", ";"},
        {"tuning", ".", "initial_search_time", "=", "process_config", ".", "initial_search_time", ";"},
        {"tuning", ".", "ik_enabled", "=", "process_config", ".", "ik_enabled", ";"},
        {"tuning", ".", "inertialize_blending_halflife", "=", "process_config", ".", "inertialize_blending_halflife", ";"},
        {"tuning", ".", "simulation_rotation_halflife", "=", "process_config", ".", "simulation_rotation_halflife", ";"},
        {"input", ".", "desired_strafe", "=", "process_config", ".", "desired_strafe", ";"},
    };
    const char* exact_live_lowerings[] = {
        "frame_external.tuning.initial_search_time=process_config.initial_search_time;",
        "frame_external.tuning.ik_enabled=process_config.ik_enabled;",
        "frame_external.tuning.initial_search_time=process_config.initial_search_time;",
        "frame_external.tuning.ik_enabled=process_config.ik_enabled;",
        "frame_external.tuning.inertialize_blending_halflife=process_config.inertialize_blending_halflife;",
        "frame_external.tuning.simulation_rotation_halflife=process_config.simulation_rotation_halflife;",
        "frame_external.input.desired_strafe=process_config.desired_strafe;",
    };
    const auto unconditional_lowering = [&main_tokens, &depth](
        std::size_t match,
        std::size_t scope_begin,
        std::size_t scope_end,
        int required_depth,
        const char* exact_compact) {
        std::size_t statement_begin = 0;
        std::size_t statement_end = 0;
        if (match >= depth.size() || depth[match] != required_depth ||
            !cpp_statement_range(
                main_tokens,
                depth,
                match,
                scope_begin,
                scope_end,
                statement_begin,
                statement_end)) {
            return false;
        }
        const std::string statement = cpp_compact_tokens(
            main_tokens, statement_begin, statement_end + 1);
        if (statement.find("if(") != std::string::npos ||
            statement.find("for(") != std::string::npos ||
            statement.find("while(") != std::string::npos ||
            statement.find("switch(") != std::string::npos ||
            statement.find("&&") != std::string::npos ||
            statement.find("||") != std::string::npos ||
            statement.find('?') != std::string::npos) {
            return false;
        }
        return exact_compact == NULL || statement == exact_compact;
    };
    for (std::size_t lowering_index = 0;
         lowering_index < sizeof(lowerings) / sizeof(lowerings[0]);
         ++lowering_index) {
        const std::vector<std::size_t> matches = cpp_find_token_sequence(
            main_tokens,
            startup_end,
            main_tokens.size(),
            lowerings[lowering_index]);
        if (lowering_index < 2) {
            check(matches.size() == 2 &&
                      matches[0] < updates[0] &&
                      unconditional_lowering(
                          matches[0],
                          startup_end,
                          updates[0],
                          1,
                          NULL) &&
                      matches[1] > update_open &&
                      matches[1] < update_close &&
                      unconditional_lowering(
                          matches[1],
                          update_open + 1,
                          update_close,
                          update_depth,
                          exact_live_lowerings[lowering_index]),
                  "search/IK process controls lower once to reset and once to the live frame");
        } else {
            check(matches.size() == 1 &&
                      matches[0] > update_open &&
                      matches[0] < update_close &&
                      unconditional_lowering(
                          matches[0],
                          update_open + 1,
                          update_close,
                          update_depth,
                          exact_live_lowerings[lowering_index]),
                  "each remaining process control lowers once at live update depth");
        }
    }
    const char* parsed_values[] = {
        "parsed_initial_search_time", "parsed_ik_enabled",
        "parsed_inertialize_blending_halflife",
        "parsed_simulation_rotation_halflife", "parsed_desired_strafe",
    };
    for (const char* parsed : parsed_values) {
        check(cpp_identifier_count(
                  main_tokens, 0, main_tokens.size(), parsed) == 3 &&
                  cpp_direct_assignment_count(
                      main_tokens, 0, main_tokens.size(), parsed) == 1,
              "mutable parser output appears only in initialization, parser output, and immutable aggregate construction");
    }
    check(cpp_identifier_count(
              main_tokens, 0, main_tokens.size(), "process_config") == 9 &&
              cpp_direct_assignment_count(
                  main_tokens, 0, main_tokens.size(), "process_config") == 0,
          "process_config cannot be reassigned or shadowed before any lowering");

    const std::vector<std::string> gamepad_signature = {
        "static", "vec3", "gamepad_get_stick", "(",
        "const", "int", "stick", ",", "const", "float", "deadzone",
        "=", "0.2f", ")",
    };
    ExactFunctionRange gamepad;
    check(cpp_exact_function_range(
              controller_tokens, gamepad_signature, gamepad) &&
              position_is_inside_no_main_guard(
                  no_main_view, controller_tokens[gamepad.signature].begin),
          "gamepad sampler has one exact internal main-only signature");
    const std::vector<int> controller_depth =
        cpp_brace_depth_before(controller_tokens);
    const std::vector<CppCallRecord> get_time_calls = cpp_calls_named(
        controller_tokens,
        gamepad.body_begin + 1,
        gamepad.body_end,
        "GetTime");
    check(get_time_calls.size() == 1 &&
              get_time_calls[0].key == "::GetTime" &&
              cpp_compact_tokens(
                  controller_tokens,
                  get_time_calls[0].name,
                  get_time_calls[0].closing + 1) ==
                  "GetTime()" &&
              cpp_identifier_count(
                  controller_tokens,
                  gamepad.body_begin + 1,
                  gamepad.body_end,
                  "NOT_MM_AUTODRIVE") == 0,
          "autodrive uses exact global GetTime(), never a prefixed decoy");
    bool exact_autodrive_directive = false;
    std::size_t autodrive_open = std::string::npos;
    const std::vector<std::string> autodrive_directive = {
        "#", "if", "defined", "(", "MM_AUTODRIVE", ")",
    };
    for (std::size_t index = gamepad.body_begin + 1;
         index + 6 < gamepad.body_end;
         ++index) {
        bool matches = controller_tokens[index].directive;
        for (std::size_t offset = 0;
             matches && offset < autodrive_directive.size();
             ++offset) {
            matches = index + offset < controller_tokens.size() &&
                controller_tokens[index + offset].directive &&
                controller_tokens[index + offset].line ==
                    controller_tokens[index].line &&
                controller_tokens[index + offset].text ==
                    autodrive_directive[offset];
        }
        if (matches) {
            exact_autodrive_directive = true;
            autodrive_open = index;
        }
    }
    check(exact_autodrive_directive,
          "GetTime autodrive path is guarded by exact MM_AUTODRIVE tokens");
    std::size_t autodrive_else = std::string::npos;
    std::size_t autodrive_close = std::string::npos;
    int directive_depth = 0;
    for (std::size_t index = autodrive_open;
         index + 1 < gamepad.body_end;
         ++index) {
        if (!controller_tokens[index].directive ||
            controller_tokens[index].text != "#" ||
            !controller_tokens[index + 1].directive ||
            controller_tokens[index + 1].line !=
                controller_tokens[index].line) {
            continue;
        }
        const std::string& directive =
            controller_tokens[index + 1].text;
        if (directive == "if" || directive == "ifdef" ||
            directive == "ifndef") {
            ++directive_depth;
        } else if (directive == "else" && directive_depth == 1) {
            autodrive_else = index;
        } else if (directive == "endif") {
            --directive_depth;
            if (directive_depth == 0) {
                autodrive_close = index;
                break;
            }
        }
    }
    check(autodrive_else != std::string::npos &&
              autodrive_close != std::string::npos &&
              autodrive_open < get_time_calls[0].name &&
              get_time_calls[0].name < autodrive_else &&
              autodrive_else < autodrive_close &&
              cpp_identifier_count(
                  controller_tokens,
                  gamepad.body_begin + 1,
                  gamepad.body_end,
                  "GetTime") == 1 &&
              controller_depth[get_time_calls[0].name] ==
                  controller_depth[gamepad.body_begin] + 1 &&
              !cpp_range_has_indirect_callable_syntax(
                  controller_tokens,
                  gamepad.body_begin + 1,
                  gamepad.body_end),
          "GetTime is called once only in the exact autodrive branch with no callable wrapper");
    std::size_t get_time_statement_begin = 0;
    std::size_t get_time_statement_end = 0;
    check(cpp_statement_range(
              controller_tokens,
              controller_depth,
              get_time_calls[0].name,
              gamepad.body_begin + 1,
              gamepad.body_end,
              get_time_statement_begin,
              get_time_statement_end),
          "GetTime belongs to one ordinary autodrive statement");
    const std::string get_time_statement = cpp_compact_tokens(
        controller_tokens,
        get_time_statement_begin,
        get_time_statement_end + 1);
    check(get_time_statement.find("if(") == std::string::npos &&
              get_time_statement.find("for(") == std::string::npos &&
              get_time_statement.find("while(") == std::string::npos &&
              get_time_statement.find("&&") == std::string::npos &&
              get_time_statement.find("||") == std::string::npos &&
              get_time_statement.find('?') == std::string::npos,
          "GetTime autodrive statement is unconditional inside its exact compile-time branch");

    const std::vector<CppCallRecord> gamepad_calls = cpp_calls_named(
        main_tokens, update_open + 1, update_close, "gamepad_get_stick");
    check(gamepad_calls.size() == 2 &&
              gamepad_calls[0].key == "::gamepad_get_stick" &&
              gamepad_calls[1].key == "::gamepad_get_stick" &&
              depth[gamepad_calls[0].name] == update_depth &&
              depth[gamepad_calls[1].name] == update_depth,
          "left/right gamepad samples are two direct top-level global calls");
    std::size_t left_begin = 0;
    std::size_t left_end = 0;
    std::size_t right_begin = 0;
    std::size_t right_end = 0;
    check(cpp_exact_statement_containing(
              main_tokens,
              depth,
              gamepad_calls[0].name,
              update_open + 1,
              update_close,
              "constvec3gamepadstick_left=::gamepad_get_stick(GAMEPAD_STICK_LEFT);",
              left_begin,
              left_end) &&
              cpp_exact_statement_containing(
                  main_tokens,
                  depth,
                  gamepad_calls[1].name,
                  update_open + 1,
                  update_close,
                  "constvec3gamepadstick_right=::gamepad_get_stick(GAMEPAD_STICK_RIGHT);",
                  right_begin,
                  right_end),
          "left/right samples bind exact stick arguments to immutable per-frame values");
    const std::vector<std::size_t> move_lowering = cpp_find_token_sequence(
        main_tokens,
        update_open + 1,
        update_close,
        {"frame_external", ".", "input", ".", "move_stick", "=",
         "gamepadstick_left", ";"});
    const std::vector<std::size_t> look_lowering = cpp_find_token_sequence(
        main_tokens,
        update_open + 1,
        update_close,
        {"frame_external", ".", "input", ".", "look_stick", "=",
         "gamepadstick_right", ";"});
    const std::vector<CppCallRecord> coordinator = cpp_calls_named(
        main_tokens,
        update_open + 1,
        update_close,
        "g1_frame_transaction_run");
    std::size_t move_statement_begin = 0;
    std::size_t move_statement_end = 0;
    std::size_t look_statement_begin = 0;
    std::size_t look_statement_end = 0;
    check(move_lowering.size() == 1 && look_lowering.size() == 1 &&
              coordinator.size() == 1 &&
              cpp_exact_statement_containing(
                  main_tokens,
                  depth,
                  move_lowering[0],
                  update_open + 1,
                  update_close,
                  "frame_external.input.move_stick=gamepadstick_left;",
                  move_statement_begin,
                  move_statement_end) &&
              cpp_exact_statement_containing(
                  main_tokens,
                  depth,
                  look_lowering[0],
                  update_open + 1,
                  update_close,
                  "frame_external.input.look_stick=gamepadstick_right;",
                  look_statement_begin,
                  look_statement_end) &&
              depth[move_lowering[0]] == update_depth &&
              depth[look_lowering[0]] == update_depth &&
              left_end < move_lowering[0] &&
              right_end < look_lowering[0] &&
              move_lowering[0] < coordinator[0].name &&
              look_lowering[0] < coordinator[0].name,
          "both exact samples lower once into frame input before the coordinator");
    for (std::size_t lowering_index = 0;
         lowering_index < sizeof(lowerings) / sizeof(lowerings[0]);
         ++lowering_index) {
        const std::vector<std::size_t> matches = cpp_find_token_sequence(
            main_tokens,
            update_open + 1,
            update_close,
            lowerings[lowering_index]);
        check(matches.size() == 1 &&
                  matches[0] < coordinator[0].name,
              "every live immutable process control is lowered before the sole coordinator");
    }
}

static void check_structural_production_runner_contract(
    const std::vector<CppToken>& tokens,
    const NoMainSourceView& no_main_view)
{
    const std::vector<std::string> signature = {
        "G1FrameStageOutcome", "g1_controller_frame_stage_run", "(",
        "G1FrameTransactionStage", "stage", ",",
        "g1_controller_state", "&", "working_state", ",",
        "G1FrameTransactionScratch", "&", "scratch", ",",
        "const", "G1FrameExternalInputs", "&", "external", ",",
        "char", "*", "error", ",", "int", "error_capacity", ")",
    };
    ExactFunctionRange runner;
    check(cpp_exact_function_range(tokens, signature, runner) &&
              !position_is_inside_no_main_guard(
                  no_main_view, tokens[runner.signature].begin),
          "the unique exact production runner remains linked outside no-main");
    for (std::size_t index = runner.body_begin + 1;
         index < runner.body_end;
         ++index) {
        check(!tokens[index].directive,
              "the production runner contains no preprocessor branch");
    }

    const char* forbidden_identifiers[] = {
        "accepted_state", "accepted_diagnostic", "publication",
        "G1FramePublication", "G1FrameRuntime", "frame_runtime",
        "g1_ik_frame_evaluate", "const_cast", "reinterpret_cast",
        "dynamic_cast", "getenv", "GetGamepadAxisMovement",
        "GetGamepadButtonPressed", "IsGamepadAvailable", "IsKeyDown",
        "GetTime", "Camera3D", "Model", "BeginDrawing", "EndDrawing",
        "BeginMode3D", "EndMode3D", "InitWindow", "WindowShouldClose",
        "SetConfigFlags", "SetTargetFPS", "LoadModel", "UnloadModel",
        "deterministic_log", "controlled_runtime_error", "g_frame",
        "g_log",
    };
    for (const char* identifier : forbidden_identifiers) {
        check(cpp_identifier_count(
                  tokens,
                  runner.body_begin + 1,
                  runner.body_end,
                  identifier) == 0,
              "the production runner has no accepted/publication/input/UI/log/cleanup authority");
    }

    const std::vector<int> depth = cpp_brace_depth_before(tokens);
    const std::vector<std::size_t> switches =
        cpp_find_token_sequence_at_depth(
            tokens,
            depth,
            runner.body_begin + 1,
            runner.body_end,
            {"switch", "(", "stage", ")", "{"},
            depth[runner.body_begin] + 1);
    check(switches.size() == 1,
          "the runner has one direct switch over its stage parameter");
    const std::size_t switch_open = switches[0] + 4;
    const std::size_t switch_close = cpp_matching_token(
        tokens, switch_open, "{", "}");
    check(switch_close != std::string::npos &&
              switch_close < runner.body_end,
          "the direct production stage switch is balanced inside the runner");
    const int case_depth = depth[switch_open] + 1;
    const char* stage_names[] = {
        "G1FrameStageInputRouteCommand",
        "G1FrameStageMatcherSearch",
        "G1FrameStageCandidateApply",
        "G1FrameStageInertialization",
        "G1FrameStageSimulationUpdate",
        "G1FrameStageSupportObservation",
        "G1FrameStageSupportRetarget",
        "G1FrameStageContactUpdate",
        "G1FrameStageFootprintObservation",
        "G1FrameStageRawBegin",
        "G1FrameStageRawFirstFoot",
        "G1FrameStageRawSecondFoot",
        "G1FrameStageRawFinalFk",
        "G1FrameStageRawPoseCertificate",
        "G1FrameStageIkBegin",
        "G1FrameStageIkFirstFoot",
        "G1FrameStageIkSecondFoot",
        "G1FrameStageIkFinalFk",
        "G1FrameStageIkPoseCertificate",
        "G1FrameStageAcceptedFinalize",
    };
    std::vector<std::size_t> cases;
    for (const char* stage_name : stage_names) {
        const std::vector<std::size_t> matches =
            cpp_find_token_sequence_at_depth(
                tokens,
                depth,
                switch_open + 1,
                switch_close,
                {"case", stage_name, ":"},
                case_depth);
        check(matches.size() == 1 &&
                  (cases.empty() || cases.back() < matches[0]),
              "each authenticated production stage has one ordered direct case");
        cases.push_back(matches[0]);
    }
    check(cpp_find_token_sequence_at_depth(
              tokens,
              depth,
              switch_open + 1,
              switch_close,
              {"default", ":"},
              case_depth).size() == 1,
          "the production stage switch has one direct unknown-stage default");

    const auto check_case_call = [&tokens, &cases, switch_close](
        std::size_t case_index,
        const char* name,
        std::size_t expected_count,
        bool global_call = true) {
        const std::size_t end = case_index + 1 < cases.size()
            ? cases[case_index + 1]
            : switch_close;
        const std::vector<CppCallRecord> calls = cpp_calls_named(
            tokens, cases[case_index] + 3, end, name);
        const std::string expected_key = global_call
            ? "::" + std::string(name)
            : std::string(name);
        bool exact_owner = calls.size() == expected_count;
        for (const CppCallRecord& call : calls) {
            exact_owner = exact_owner && call.key == expected_key;
        }
        check(exact_owner,
              "each stage invokes only its exact direct global production operation");
    };
    check_case_call(0, "deterministic_route_command", 1);
    check_case_call(1, "database_search", 1);
    check_case_call(2, "g1_frame_candidate_record_is_valid", 1);
    check_case_call(3, "inertialize_pose_update", 1);
    check_case_call(4, "simulation_positions_update", 1);
    check_case_call(4, "simulation_rotations_update", 1);
    check_case_call(5, "g1_ik_checked_forward_kinematics", 1);
    check_case_call(5, "support_observation_build", 1);
    check_case_call(6, "support_pose_apply", 1);
    check_case_call(7, "g1_ik_checked_forward_kinematics", 1);
    check_case_call(7, "contact_update", 1);
    check_case_call(8, "g1_foot_contact_schedule_build", 1);
    check_case_call(8, "g1_footprint_observe_v2", 1);
    check_case_call(8, "g1_ik_frame_begin", 0);
    check_case_call(9, "g1_runner_certificate_begin", 1, false);
    check_case_call(10, "g1_runner_certificate_foot", 1, false);
    check_case_call(11, "g1_runner_certificate_foot", 1, false);
    check_case_call(12, "g1_runner_certificate_finish", 1, false);
    check_case_call(13, "g1_runner_pose_certificate", 1, false);
    check_case_call(14, "g1_runner_certificate_begin", 1, false);
    check_case_call(15, "g1_runner_certificate_foot", 1, false);
    check_case_call(16, "g1_runner_certificate_foot", 1, false);
    check_case_call(17, "g1_runner_certificate_finish", 1, false);
    check_case_call(18, "g1_runner_pose_certificate", 1, false);
    check_case_call(19, "motion_match_pose_snapshot", 1, false);
    check(cpp_calls_named(
              tokens,
              runner.body_begin + 1,
              runner.body_end,
              "contact_update").size() == 1 &&
              cpp_calls_named(
                  tokens,
                  runner.body_begin + 1,
                  runner.body_end,
                  "inertialize_pose_update").size() == 1 &&
              cpp_calls_named(
                  tokens,
                  runner.body_begin + 1,
                  runner.body_end,
                  "simulation_positions_update").size() == 1 &&
              cpp_calls_named(
                  tokens,
                  runner.body_begin + 1,
                  runner.body_end,
                  "simulation_rotations_update").size() == 1,
          "runner owns one total legacy contact/inertialization/simulation operation each");

    const char* consumed_member_paths[] = {
        "external.tuning.mode",
        "external.tuning.initial_search_time",
        "external.tuning.inertialize_blending_halflife",
        "external.tuning.simulation_rotation_halflife",
        "external.tuning.contact_unlock_radius",
        "external.tuning.contact_foot_height",
        "external.tuning.contact_blending_halflife",
        "external.tuning.effective_terrain_weight",
        "external.input.move_stick",
        "external.input.look_stick",
        "external.input.gait_target",
        "external.input.camera_zoom_axis",
        "external.input.scripted_azimuth_delta",
        "external.input.desired_strafe",
        "external.heading_override",
    };
    for (const char* compact_path : consumed_member_paths) {
        std::string path(compact_path);
        std::vector<std::string> expected;
        std::size_t begin = 0;
        while (begin < path.size()) {
            const std::size_t dot = path.find('.', begin);
            expected.push_back(path.substr(begin, dot - begin));
            if (dot == std::string::npos) break;
            expected.push_back(".");
            begin = dot + 1;
        }
        check(cpp_find_token_sequence(
                  tokens,
                  runner.body_begin + 1,
                  runner.body_end,
                  expected).size() >= 1,
              "the runner directly consumes every immutable typed input/tuning owner");
    }
    const char* modes[] = {
        "G1_TestLive", "G1_TestSequential", "G1_TestFlat",
        "G1_TestTerrain", "G1_TestRoute", "G1_TestSceneCycle",
    };
    for (const char* mode : modes) {
        check(cpp_identifier_count(
                  tokens,
                  runner.body_begin + 1,
                  runner.body_end,
                  mode) >= 1,
              "the production runner explicitly authenticates every controller mode");
    }
    check(cpp_find_token_sequence(
              tokens,
              cases[19] + 3,
              switch_close,
              {"scratch", ".", "accepted_diagnostic_candidate"}).size() >= 1,
          "pose certification builds only the scratch accepted-diagnostic candidate");
}

static bool cpp_case_has_unconditional_terminal(
    const std::vector<CppToken>& tokens,
    const std::vector<int>& depth,
    std::size_t case_begin,
    std::size_t case_end,
    int case_depth,
    const char* status)
{
    const std::vector<std::size_t> returns = cpp_find_token_sequence(
        tokens,
        case_begin + 3,
        case_end,
        {"return", status, ";"});
    if (returns.size() != 1) return false;
    const std::string& preceding = tokens[returns[0] - 1].text;
    if (preceding != ";" && preceding != "{" && preceding != "}" &&
        preceding != ":") {
        return false;
    }
    std::size_t first = case_begin + 3;
    while (first < case_end && tokens[first].directive) ++first;
    std::size_t tail_limit = case_end;
    if (first < case_end && tokens[first].text == "{") {
        const std::size_t close = cpp_matching_token(
            tokens, first, "{", "}");
        if (close == std::string::npos || close >= case_end ||
            depth[returns[0]] != case_depth + 1) {
            return false;
        }
        std::size_t after_close = close + 1;
        while (after_close < case_end &&
               tokens[after_close].directive) {
            ++after_close;
        }
        if (after_close != case_end) return false;
        tail_limit = close;
    } else if (depth[returns[0]] != case_depth) {
        return false;
    }
    for (std::size_t index = returns[0] + 3;
         index < tail_limit;
         ++index) {
        if (!tokens[index].directive) return false;
    }
    return true;
}

static void check_coordinator_header_has_generic_runner_identity()
{
    std::string terminal_probe_error;
    const std::vector<CppToken> terminal_probe = tokenize_cpp_source(
        "{case G1FrameStageFiniteReject:if(false){"
        "return G1FrameTransactionFiniteRejected;}"
        "case G1FrameStageGlobalError:{"
        "return G1FrameTransactionGlobalError;}default:break;}",
        terminal_probe_error);
    const std::vector<int> terminal_probe_depth =
        cpp_brace_depth_before(terminal_probe);
    const std::vector<std::size_t> probe_finite = cpp_find_token_sequence(
        terminal_probe,
        0,
        terminal_probe.size(),
        {"case", "G1FrameStageFiniteReject", ":"});
    const std::vector<std::size_t> probe_global = cpp_find_token_sequence(
        terminal_probe,
        0,
        terminal_probe.size(),
        {"case", "G1FrameStageGlobalError", ":"});
    const std::vector<std::size_t> probe_default = cpp_find_token_sequence(
        terminal_probe,
        0,
        terminal_probe.size(),
        {"default", ":"});
    check(terminal_probe_error.empty() &&
              probe_finite.size() == 1 && probe_global.size() == 1 &&
              probe_default.size() == 1 &&
              !cpp_case_has_unconditional_terminal(
                  terminal_probe,
                  terminal_probe_depth,
                  probe_finite[0],
                  probe_global[0],
                  1,
                  "G1FrameTransactionFiniteRejected") &&
              cpp_case_has_unconditional_terminal(
                  terminal_probe,
                  terminal_probe_depth,
                  probe_global[0],
                  probe_default[0],
                  1,
                  "G1FrameTransactionGlobalError"),
          "coordinator terminal audit rejects dead conditional returns and accepts exact case terminals");
    const std::string source = read_source_file("g1_frame_transaction.h");
    check(source.find("%:") == std::string::npos &&
              source.find("\\\n") == std::string::npos &&
              source.find("\\\r\n") == std::string::npos,
          "the transaction header has no directive or line-splice spoofing");
    std::string error;
    const std::vector<CppToken> tokens = tokenize_cpp_source(source, error);
    check(error.empty(),
          error.empty() ? "transaction header tokenizes" : error.c_str());
    check(cpp_identifier_count(
              tokens,
              0,
              tokens.size(),
              "g1_controller_frame_stage_run") == 0,
          "the generic coordinator header never names the production runner");
    const std::set<std::string> audited_names = {
        "g1_controller_frame_stage_run", "g1_frame_transaction_run",
        "g1_frame_run_one_stage", "run_stage", "outcome", "hook",
    };
    check(audit_controller_macros(tokens, audited_names, error),
          error.empty()
              ? "coordinator runner identities cannot be macro-spoofed"
              : error.c_str());

    std::size_t definition_name = std::string::npos;
    std::size_t parameters_begin = std::string::npos;
    std::size_t parameters_end = std::string::npos;
    std::size_t body_begin = std::string::npos;
    std::size_t body_end = std::string::npos;
    for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
        if (tokens[index].directive ||
            tokens[index].text != "g1_frame_transaction_run" ||
            tokens[index + 1].text != "(") {
            continue;
        }
        const std::size_t close = cpp_matching_token(
            tokens, index + 1, "(", ")");
        if (close == std::string::npos) continue;
        std::size_t next = close + 1;
        while (next < tokens.size() && tokens[next].directive) ++next;
        if (next < tokens.size() && tokens[next].text == "{") {
            check(definition_name == std::string::npos,
                  "the coordinator header has one definition");
            definition_name = index;
            parameters_begin = index + 1;
            parameters_end = close;
            body_begin = next;
            body_end = cpp_matching_token(tokens, next, "{", "}");
        }
    }
    check(definition_name != std::string::npos &&
              body_end != std::string::npos,
          "the coordinator header exposes one tokenized inline definition");
    const std::vector<std::size_t> runner_types = cpp_find_token_sequence(
        tokens,
        parameters_begin + 1,
        parameters_end,
        {"G1FrameStageRunner", "run_stage"});
    check(runner_types.size() == 1,
          "the coordinator receives one generic typed stage runner");

    std::size_t helper_name = std::string::npos;
    std::size_t helper_parameters_begin = std::string::npos;
    std::size_t helper_parameters_end = std::string::npos;
    std::size_t helper_body_begin = std::string::npos;
    std::size_t helper_body_end = std::string::npos;
    for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
        if (tokens[index].directive ||
            tokens[index].text != "g1_frame_run_one_stage" ||
            tokens[index + 1].text != "(") {
            continue;
        }
        const std::size_t close = cpp_matching_token(
            tokens, index + 1, "(", ")");
        if (close == std::string::npos) continue;
        std::size_t next = close + 1;
        while (next < tokens.size() && tokens[next].directive) ++next;
        if (next < tokens.size() && tokens[next].text == "{") {
            check(helper_name == std::string::npos,
                  "the coordinator header has one stage-run helper definition");
            helper_name = index;
            helper_parameters_begin = index + 1;
            helper_parameters_end = close;
            helper_body_begin = next;
            helper_body_end = cpp_matching_token(tokens, next, "{", "}");
        }
    }
    check(helper_name != std::string::npos &&
              helper_body_end != std::string::npos,
          "the generic stage-run helper has one tokenized inline definition");
    check(cpp_find_token_sequence(
              tokens,
              helper_parameters_begin + 1,
              helper_parameters_end,
              {"G1FrameStageRunner", "run_stage"}).size() == 1,
          "the stage-run helper receives the generic typed runner directly");
    const std::vector<CppCallRecord> helper_runner_calls = cpp_calls_named(
        tokens, helper_body_begin + 1, helper_body_end, "run_stage");
    check(helper_runner_calls.size() == 1 &&
              helper_runner_calls[0].key == "run_stage" &&
              cpp_compact_tokens(
                  tokens,
                  helper_runner_calls[0].name,
                  helper_runner_calls[0].closing + 1) ==
                  "run_stage(stage,state,scratch,external,error,error_capacity)",
          "the stage-run helper invokes its generic runner once with exact direct stage owners");
    const std::vector<int> depth = cpp_brace_depth_before(tokens);
    std::size_t runner_statement_begin = 0;
    std::size_t runner_statement_end = 0;
    check(cpp_exact_statement_containing(
              tokens,
              depth,
              helper_runner_calls[0].name,
              helper_body_begin + 1,
              helper_body_end,
              "constG1FrameStageOutcomeoutcome="
              "run_stage(stage,state,scratch,external,error,error_capacity);",
              runner_statement_begin,
              runner_statement_end),
          "the helper's direct runner outcome is bound once to an immutable exact stage result");
    const std::vector<std::size_t> outcome_guards =
        cpp_find_token_sequence(
            tokens,
            runner_statement_end + 1,
            helper_body_end,
            {"if", "(", "outcome", "!=",
             "G1FrameStageContinue", ")", "{"});
    check(outcome_guards.size() == 1,
          "the stage-run helper has one exact non-Continue outcome guard");
    const std::size_t outcome_guard_open = outcome_guards[0] + 6;
    const std::size_t outcome_guard_close = cpp_matching_token(
        tokens, outcome_guard_open, "{", "}");
    check(outcome_guard_close != std::string::npos &&
              cpp_compact_tokens(
                  tokens,
                  outcome_guards[0],
                  outcome_guard_close + 1) ==
                  "if(outcome!=G1FrameStageContinue){"
                  "returnoutcome==G1FrameStageFiniteReject||"
                  "outcome==G1FrameStageGlobalError?outcome:"
                  "G1FrameStageGlobalError;}",
          "only authentic finite/global outcomes propagate; unknown runner outcomes become global errors");

    const std::vector<CppCallRecord> hook_calls = cpp_calls_named(
        tokens, helper_body_begin + 1, helper_body_end, "hook");
    check(hook_calls.size() == 1 &&
              hook_calls[0].key == "->hook" &&
              outcome_guard_close < hook_calls[0].name,
          "test hook is invoked once only after a real Continue outcome");
    for (std::size_t index = helper_body_begin + 1;
         index + 1 < helper_body_end;
         ++index) {
        if (tokens[index].kind != CppTokenIdentifier ||
            tokens[index + 1].text != "(") {
            continue;
        }
        const std::string key = cpp_call_key(tokens, index);
        if ((!key.empty() && key[0] == '.') ||
            key.rfind("->", 0) == 0) {
            check(index == hook_calls[0].name && key == "->hook",
                  "coordinator has no member/alternate callable dispatch");
        }
    }
    check(cpp_find_token_sequence(
              tokens,
              helper_body_begin + 1,
              helper_body_end,
              {"run_stage", "==", "nullptr"}).size() == 1 &&
              cpp_find_token_sequence(
                  tokens,
                  helper_body_begin + 1,
                  helper_body_end,
                  {"run_stage", "!=", "nullptr"}).empty() &&
              cpp_direct_assignment_count(
                  tokens,
                  helper_body_begin + 1,
                  helper_body_end,
                  "run_stage") == 0,
          "generic runner identity is used only for one null preflight and one exact call");

    const std::vector<CppCallRecord> coordinator_runner_calls =
        cpp_calls_named(tokens, body_begin + 1, body_end, "run_stage");
    const std::vector<CppCallRecord> coordinator_stage_helpers =
        cpp_calls_named(
            tokens,
            body_begin + 1,
            body_end,
            "g1_frame_run_one_stage");
    const std::vector<CppCallRecord> coordinator_candidate_helpers =
        cpp_calls_named(
            tokens,
            body_begin + 1,
            body_end,
            "g1_frame_candidate_evaluate");
    bool exact_propagation =
        coordinator_stage_helpers.size() == 2 &&
        coordinator_candidate_helpers.size() == 1;
    for (const CppCallRecord& call : coordinator_stage_helpers) {
        exact_propagation =
            exact_propagation &&
            cpp_identifier_count(
                tokens, call.name, call.closing + 1, "run_stage") == 1;
    }
    for (const CppCallRecord& call : coordinator_candidate_helpers) {
        exact_propagation =
            exact_propagation &&
            cpp_identifier_count(
                tokens, call.name, call.closing + 1, "run_stage") == 1;
    }
    check(coordinator_runner_calls.empty() && exact_propagation &&
              cpp_identifier_count(
                  tokens,
                  body_begin + 1,
                  body_end,
                  "run_stage") == 4 &&
              cpp_find_token_sequence(
                  tokens,
                  body_begin + 1,
                  body_end,
                  {"run_stage", "==", "nullptr"}).size() == 1 &&
              cpp_direct_assignment_count(
                  tokens,
                  body_begin + 1,
                  body_end,
                  "run_stage") == 0,
          "the coordinator only null-checks and forwards its immutable runner to exact typed helpers");
}

static void test_controller_source_and_no_main_contract()
{
    const std::string lexical_probe =
        "kept_code(); // accepted_state { fake }\n"
        "const char* text = \"publication getenv { }\"; "
        "/* working_state } */ kept_tail(); "
        "auto raw = R\"tag(accepted_state publication)tag\"; kept_raw();";
    const std::string masked_probe =
        mask_source_noncode(lexical_probe, true);
    check(masked_probe.find("kept_code") != std::string::npos &&
              masked_probe.find("kept_tail") != std::string::npos &&
              masked_probe.find("kept_raw") != std::string::npos &&
              masked_probe.find("accepted_state") == std::string::npos &&
              masked_probe.find("publication") == std::string::npos &&
              masked_probe.find("getenv") == std::string::npos &&
              masked_probe.find("working_state") == std::string::npos,
          "lexical guard preserves executable code while masking comments and literals");
    const std::string active_call_probe =
        "notstrcmp(text, \"0\"); "
        "const char* decoy = \"strcmp(text, \\\"0\\\")\"; "
        "strcmp(text, \"0\");";
    std::string token_probe_error;
    const std::vector<CppToken> active_call_tokens =
        tokenize_cpp_source(active_call_probe, token_probe_error);
    const std::vector<std::size_t> exact_probe_calls =
        cpp_find_token_sequence(
            active_call_tokens,
            0,
            active_call_tokens.size(),
            {"strcmp", "(", "text", ",", "\"0\"", ")"});
    check(exact_active_call_count(
              mask_source_noncode(active_call_probe, true),
              mask_source_noncode(active_call_probe, false),
              "strcmp(",
              "strcmp(text,\"0\")") == 1 &&
              token_probe_error.empty() &&
              exact_probe_calls.size() == 1 &&
              cpp_compact_tokens(
                  active_call_tokens,
                  exact_probe_calls[0],
                  exact_probe_calls[0] + 6) ==
                  "strcmp(text,\"0\")",
          "active-call authentication rejects identifier-prefix and string-literal decoys");
    std::string structural_error;
    const std::set<std::string> probe_trusted = {"::trusted_leaf"};
    check(analyze_no_main_root_closure(
              "static int helper(){return ::trusted_leaf();}"
              "int runner(){return helper();}",
              "runner",
              probe_trusted,
              structural_error),
          "structural closure accepts one reachable static helper and one trusted leaf");
    structural_error.clear();
    check(analyze_no_main_root_closure(
              "int runner(const slice1d<int> values){"
              "array1d<int> local(1);local(0)=values(0);"
              "return local(0);}",
              "runner",
              probe_trusted,
              structural_error),
          "structural closure accepts checked const-slice and local-array indexing owners");
    structural_error.clear();
    check(!analyze_no_main_root_closure(
              "int runner(){return rogue(0);}",
              "runner",
              probe_trusted,
              structural_error) &&
              !analyze_no_main_root_closure(
                  "int runner(Callable callable){return callable(0);}",
                  "runner",
                  probe_trusted,
                  structural_error),
          "structural closure rejects undeclared calls and non-array callable objects");
    structural_error.clear();
    check(!analyze_no_main_root_closure(
              "static int orphan(){return 1;}int runner(){return 0;}",
              "runner",
              probe_trusted,
              structural_error),
          "structural closure rejects an orphan static helper");
    structural_error.clear();
    check(!analyze_no_main_root_closure(
              "int mutable_state;int runner(){return mutable_state;}",
              "runner",
              probe_trusted,
              structural_error),
          "structural closure rejects mutable file-scope data");
    structural_error.clear();
    check(!analyze_no_main_root_closure(
              "namespace hidden{int mutable_state;}"
              "int runner(){return 0;}",
              "runner",
              probe_trusted,
              structural_error),
          "structural closure cannot hide mutable roots in a namespace");
    structural_error.clear();
    check(!analyze_no_main_root_closure(
              "int runner(){auto dead=[](){return ::trusted_leaf();};"
              "return 0;}",
              "runner",
              probe_trusted,
              structural_error),
          "structural closure rejects a runtime-dead lambda evidence root");
    structural_error.clear();
    const bool member_shadow_rejected =
        !analyze_no_main_root_closure(
            "int runner(){return object.trusted_leaf();}",
            "runner",
            probe_trusted,
            structural_error);
    structural_error.clear();
    const bool namespace_shadow_rejected =
        !analyze_no_main_root_closure(
            "int runner(){return decoy::trusted_leaf();}",
            "runner",
            probe_trusted,
            structural_error);
    check(member_shadow_rejected && namespace_shadow_rejected,
          "structural closure rejects member and namespace shadow calls");
    structural_error.clear();
    check(!analyze_no_main_root_closure(
              "[[maybe_unused]] static int orphan(){return 0;}"
              "int runner(){return 0;}",
              "runner",
              probe_trusted,
              structural_error),
          "structural closure rejects unused-attribute orphan escapes");
    structural_error.clear();
    check(!analyze_no_main_root_closure(
              "int runner(){return (::trusted_leaf)();}",
              "runner",
              probe_trusted,
              structural_error) &&
              !analyze_no_main_root_closure(
                  "int runner(){int(*hook)()=nullptr;return (*hook)();}",
                  "runner",
                  probe_trusted,
                  structural_error) &&
              !analyze_no_main_root_closure(
                  "using Hook=int(*)();int runner(){return 0;}",
                  "runner",
                  probe_trusted,
                  structural_error) &&
              !analyze_no_main_root_closure(
                  "template<class T>static int evil(T&){return 0;}"
                  "int runner(){int value=0;return ::evil<int>(value);}",
                  "runner",
                  probe_trusted,
                  structural_error) &&
              !analyze_no_main_root_closure(
                  "int runner(){return ::evil<std::pair<int,long>>(working_state);}",
                  "runner",
                  probe_trusted,
                  structural_error) &&
              !analyze_no_main_root_closure(
                  "int runner(){return callable.operator()();}",
                  "runner",
                  probe_trusted,
                  structural_error),
          "structural closure rejects parenthesized, pointer, simple/nested template, operator, and callable-root dispatch");
    structural_error.clear();
    check(!analyze_no_main_root_closure(
              "extern int mutable_state;int runner(){return mutable_state;}",
              "runner",
              probe_trusted,
              structural_error) &&
              !analyze_no_main_root_closure(
                  "int runner(){goto done;done:return 0;}",
                  "runner",
                  probe_trusted,
                  structural_error) &&
              !analyze_no_main_root_closure(
                  "int runner()<%return 0;%>",
                  "runner",
                  probe_trusted,
                  structural_error),
          "structural closure rejects extern data, goto, and brace digraphs");
    check(exact_no_main_excluding_opening(
              "#if!defined(G1_CONTROLLER_NO_MAIN)") &&
              exact_no_main_excluding_opening(
                  "#ifndef G1_CONTROLLER_NO_MAIN") &&
              !exact_no_main_excluding_opening(
                  "#if!definedG1_CONTROLLER_NO_MAIN") &&
              !exact_no_main_excluding_opening(
                  "#if!defined(G1_CONTROLLER_NO_MAIN)||defined(DECOY)") &&
              !exact_no_main_excluding_opening(
                  "#ifdefG1_CONTROLLER_NO_MAIN"),
          "only an exact one-branch no-main exclusion qualifies");
    NoMainSourceView invalid_view;
    structural_error.clear();
    check(!build_no_main_source_view(
              "%:if !defined(G1_CONTROLLER_NO_MAIN)\n"
              "int main(){}\n%:endif\n",
              invalid_view,
              structural_error) &&
              !build_no_main_source_view(
                  "#ifndef G1_CONTROLLER_NO_MAIN\n"
                  "int main(){}\n#else\nint decoy;\n#endif\n",
                  invalid_view,
                  structural_error),
          "no-main parser rejects directive digraphs and compiled else branches");
    const std::string source = read_source_file("controller.cpp");
    check(source.find("%:") == std::string::npos &&
              source.find("<%") == std::string::npos &&
              source.find("%>") == std::string::npos &&
              source.find("\\\n") == std::string::npos &&
              source.find("\\\r\n") == std::string::npos,
          "audited controller source has no digraph or translation-phase line splice");
    const NoMainSourceView no_main_view =
        no_main_source_view(source);
    std::string controller_token_error;
    const std::vector<CppToken> controller_tokens =
        tokenize_cpp_source(source, controller_token_error);
    check(controller_token_error.empty(),
          controller_token_error.empty()
              ? "controller token stream is valid"
              : controller_token_error.c_str());
    const std::set<std::string> audited_macro_names =
        production_authenticated_source_names();
    std::string macro_probe_error;
    std::vector<CppToken> macro_probe = tokenize_cpp_source(
        "#define contact_update evil\n", macro_probe_error);
    check(!audit_controller_macros(
              macro_probe, audited_macro_names, macro_probe_error),
          "macro audit rejects an authenticated call-name definition");
    macro_probe_error.clear();
    macro_probe = tokenize_cpp_source(
        "#define write_requested(...) close()\n",
        macro_probe_error);
    check(!audit_controller_macros(
              macro_probe, audited_macro_names, macro_probe_error),
          "macro audit rejects candidate-writer function-like redirection");
    macro_probe_error.clear();
    macro_probe = tokenize_cpp_source(
        "#define candidate_audit_log decoy_owner\n",
        macro_probe_error);
    check(!audit_controller_macros(
              macro_probe, audited_macro_names, macro_probe_error),
          "macro audit rejects candidate-audit owner redirection");
    macro_probe_error.clear();
    macro_probe = tokenize_cpp_source(
        "#define candidate_audit_environment nullptr\n",
        macro_probe_error);
    check(!audit_controller_macros(
              macro_probe, audited_macro_names, macro_probe_error),
          "macro audit rejects candidate-audit environment redirection");
    macro_probe_error.clear();
    macro_probe = tokenize_cpp_source(
        "#define const\n",
        macro_probe_error);
    check(!audit_controller_macros(
              macro_probe, audited_macro_names, macro_probe_error),
          "macro audit rejects removal of candidate-writer const ownership");
    macro_probe_error.clear();
    macro_probe = tokenize_cpp_source(
        "#define G1FramePublication MutablePublication\n",
        macro_probe_error);
    check(!audit_controller_macros(
              macro_probe, audited_macro_names, macro_probe_error),
          "macro audit rejects candidate-writer accepted input type redirection");
    macro_probe_error.clear();
    macro_probe = tokenize_cpp_source(
        "#define G1FrameTransactionAccepted decoy\n",
        macro_probe_error);
    check(!audit_controller_macros(
              macro_probe, audited_macro_names, macro_probe_error),
          "macro audit rejects an authenticated status-name definition");
    macro_probe_error.clear();
    macro_probe = tokenize_cpp_source(
        "#undef frame_status\n", macro_probe_error);
    check(!audit_controller_macros(
              macro_probe, audited_macro_names, macro_probe_error),
          "macro audit rejects an authenticated binding-name undefinition");
    macro_probe_error.clear();
    macro_probe = tokenize_cpp_source(
        "#define move_stick look_stick\n", macro_probe_error);
    check(!audit_controller_macros(
              macro_probe, audited_macro_names, macro_probe_error),
          "macro audit rejects an authenticated input-member redirection");
    macro_probe_error.clear();
    macro_probe = tokenize_cpp_source(
        "#undef ERANGE\n", macro_probe_error);
    check(!audit_controller_macros(
              macro_probe, audited_macro_names, macro_probe_error),
          "macro audit rejects an authenticated parser-token redirection");
    macro_probe_error.clear();
    macro_probe = tokenize_cpp_source(
        "#define UNRELATED_PROBE 1\n", macro_probe_error);
    check(macro_probe_error.empty() &&
              audit_controller_macros(
                  macro_probe, audited_macro_names, macro_probe_error),
          "macro audit accepts an unrelated definition");
    check(audit_controller_macros(
              controller_tokens,
              audited_macro_names,
              controller_token_error),
          controller_token_error.empty()
              ? "audited controller identifiers are not macro-spoofed"
              : controller_token_error.c_str());
    std::string header_probe_error;
    check(audit_project_header_roots(
              "valid_probe.h",
              "extern int trusted_function();static const int readonly=1;",
              audited_macro_names,
              header_probe_error),
          header_probe_error.empty()
              ? "header authority audit accepts function declarations and read-only roots"
              : header_probe_error.c_str());
    header_probe_error.clear();
    check(!audit_project_header_roots(
              "extern_probe.h",
              "extern int shared_state;",
              audited_macro_names,
              header_probe_error),
          "header authority audit rejects mutable extern roots");
    header_probe_error.clear();
    check(!audit_project_header_roots(
              "pointer_probe.h",
              "using Hook=int(*)(int);",
              audited_macro_names,
              header_probe_error),
          "header authority audit rejects function-pointer roots");
    header_probe_error.clear();
    check(!audit_project_header_roots(
              "macro_probe.h",
              "#define contact_update decoy\n",
              audited_macro_names,
              header_probe_error),
          "header authority audit rejects authenticated header macros");
    header_probe_error.clear();
    check(!audit_project_header_roots(
              "namespace_probe.h",
              "namespace hidden{inline int mutable_state=0;}",
              audited_macro_names,
              header_probe_error) &&
              !audit_project_header_roots(
                  "class_probe.h",
                  "struct Hidden{inline static int mutable_state=0;};",
                  audited_macro_names,
                  header_probe_error) &&
              !audit_project_header_roots(
                  "direct_init_probe.h",
                  "namespace hidden{inline Rogue state(0);}",
                  audited_macro_names,
                  header_probe_error),
          "header authority audit recursively rejects namespace, direct-init, and inline-static mutable roots");
    header_probe_error.clear();
    check(!audit_project_header_roots(
              "digraph_probe.h",
              "%:define contact_update decoy\n",
              audited_macro_names,
              header_probe_error) &&
              !audit_project_header_roots(
                  "splice_probe.h",
                  "#define con\\\ntact_update decoy\n",
                  audited_macro_names,
                  header_probe_error),
          "header authority audit rejects directive digraph and line-spliced macro names");
    header_probe_error.clear();
    check(!audit_project_header_roots(
              "environment_probe.h",
              "inline int read_environment(){return getenv(\"X\")!=0;}",
              audited_macro_names,
              header_probe_error),
          "header authority audit rejects environment authority in owned function bodies");
    controller_token_error.clear();
    check(audit_direct_project_headers(
              controller_tokens,
              audited_macro_names,
              controller_token_error),
          controller_token_error.empty()
              ? "direct project headers have immutable authenticated roots"
              : controller_token_error.c_str());
    check_structural_main_helper_contracts(
        controller_tokens, no_main_view);
    structural_error.clear();
    std::string authenticated_closure_source;
    check(build_authenticated_production_no_main_closure_source(
              no_main_view,
              authenticated_closure_source,
              structural_error),
          structural_error.empty()
              ? "legacy query builder has one authenticated out-of-line boundary"
              : structural_error.c_str());
    structural_error.clear();
    check(analyze_no_main_root_closure(
              authenticated_closure_source,
              "g1_controller_frame_stage_run",
              production_trusted_no_main_calls(),
              structural_error),
          structural_error.empty()
              ? "no-main controller is one closed runner call graph"
              : structural_error.c_str());
    check_structural_production_runner_contract(
        controller_tokens, no_main_view);
    check_coordinator_header_has_generic_runner_identity();

    ExactFunctionRange main_function;
    const bool empty_main = cpp_exact_function_range(
        controller_tokens,
        {"int", "main", "(", ")"},
        main_function);
    ExactFunctionRange void_main_function;
    const bool void_main = cpp_exact_function_range(
        controller_tokens,
        {"int", "main", "(", "void", ")"},
        void_main_function);
    ExactFunctionRange argv_main_function;
    const bool argv_main = cpp_exact_function_range(
        controller_tokens,
        {"int", "main", "(", "int", "argc", ",",
         "char", "*", "*", "argv", ")"},
        argv_main_function);
    check(static_cast<int>(empty_main) + static_cast<int>(void_main) +
              static_cast<int>(argv_main) == 1,
          "the controller has one exact ordinary main definition");
    if (void_main) main_function = void_main_function;
    if (argv_main) main_function = argv_main_function;
    check(position_is_inside_no_main_guard(
              no_main_view,
              controller_tokens[main_function.signature].begin),
          "the sole ordinary main definition is inside the exact no-main exclusion");
    const std::string main_body_raw = source.substr(
        controller_tokens[main_function.body_begin].begin,
        controller_tokens[main_function.body_end].end -
            controller_tokens[main_function.body_begin].begin);
    std::string main_ownership_error;
    const std::vector<CppToken> valid_runtime_probe = tokenize_cpp_source(
        "{G1FrameRuntime frame_runtime;"
        "::g1_frame_runtime_reset(frame_runtime);"
        "if(frame_runtime.accepted_state.scene_frame>0){}}",
        main_ownership_error);
    check(main_ownership_error.empty() &&
              audit_whole_main_runtime_ownership(
                  valid_runtime_probe, main_ownership_error),
          "whole-main runtime audit accepts only direct declared/reset/SceneCycle ownership");
    main_ownership_error.clear();
    const std::vector<CppToken> wrong_audit_owner_probe =
        tokenize_cpp_source(
            "{G1FrameRuntime frame_runtime;"
            "::g1_frame_runtime_reset(frame_runtime);"
            "decoy.write_requested(frame_runtime);"
            "if(frame_runtime.accepted_state.scene_frame>0){}}",
            main_ownership_error);
    check(main_ownership_error.empty() &&
              !audit_whole_main_runtime_ownership(
                  wrong_audit_owner_probe, main_ownership_error),
          "whole-main runtime audit rejects candidate calls through a decoy receiver");
    main_ownership_error.clear();
    const std::vector<CppToken> runtime_alias_probe = tokenize_cpp_source(
        "{G1FrameRuntime frame_runtime;auto* alias=&frame_runtime;"
        "::g1_frame_runtime_reset(frame_runtime);"
        "if(frame_runtime.accepted_state.scene_frame>0){}"
        "mutate_runtime(*alias);}",
        main_ownership_error);
    check(main_ownership_error.empty() &&
              !audit_whole_main_runtime_ownership(
                  runtime_alias_probe, main_ownership_error),
          "whole-main runtime audit rejects pre-loop pointer/helper aliases");
    main_ownership_error.clear();
    const std::vector<CppToken> main_ownership_tokens =
        tokenize_cpp_source(main_body_raw, main_ownership_error);
    check(main_ownership_error.empty() &&
              audit_whole_main_runtime_ownership(
                  main_ownership_tokens, main_ownership_error),
          main_ownership_error.empty()
              ? "all whole-main runtime occurrences have exact approved ownership"
              : main_ownership_error.c_str());
    check_structural_main_update_contract(main_body_raw);
    check_structural_startup_and_input_contract(
        controller_tokens, no_main_view, main_body_raw);
}

static void test_real_runner_checkpoint_atomicity()
{
    for (int stage = 0; stage < G1FrameStageCount; ++stage) {
        fixture finite;
        configure_production_mode(finite, G1_TestTerrain);
        const ProductionEvidence before =
            production_evidence(finite.runtime);
        const G1FramePublication publication_before =
            finite.runtime.publication;
        const ConstArtifactEvidence finite_artifacts =
            const_artifact_evidence(finite.external);
        G1FrameTransactionTestSeam seam;
        seam.hook = production_hook;
        seam.control.injected_stage =
            static_cast<G1FrameTransactionStage>(stage);
        seam.control.injected_outcome = G1FrameInjectFiniteReject;
        char error[1024] = {};
        reset_production_trace(
            finite.external,
            finite.runtime.accepted_state.camera_azimuth);
        check(g1_frame_transaction_run(
                  finite.runtime,
                  g1_controller_frame_stage_run,
                  finite.external,
                  &seam,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionFiniteRejected,
              error);
        check_production_trace_through(
            stage,
            "finite seam runs only after the real stage and stops all later stages");
        check(production_trace.immutable_context_exact &&
                  production_trace.requested_intent_ready,
              "finite real checkpoint retains ready exact input intent");
        check(state_logical_digest(finite.runtime.accepted_state) ==
                  before.accepted &&
                  same_storage_identities(
                      state_storage_identities(
                          finite.runtime.accepted_state),
                      before.accepted_storage) &&
                  diagnostic_logical_digest(
                      finite.runtime.accepted_diagnostic) ==
                      before.diagnostic,
              "finite real checkpoint preserves every accepted value and owner");
        G1FramePublication expected = publication_before;
        expected.requested_intent = production_trace.requested_intent;
        expected.rejection = production_finite_rejection();
        expected.ik_safe_stop_latched = true;
        expected.presentation_frame =
            finite.external.input.presentation_frame;
        check(publication_logical_digest(finite.runtime.publication) ==
                  publication_logical_digest(expected),
              "finite real checkpoint changes exactly the publication whitelist");
        check(same_storage_identities(
                  state_storage_identities(finite.runtime.working_state),
                  before.working_storage),
              "finite checkpoint preserves every working storage identity");
        check_const_artifacts(
            finite.external,
            finite_artifacts,
            "finite checkpoint leaves every const production artifact exact");

        fixture global;
        configure_production_mode(global, G1_TestTerrain);
        const ProductionEvidence global_before =
            production_evidence(global.runtime);
        const ConstArtifactEvidence global_artifacts =
            const_artifact_evidence(global.external);
        seam.control.injected_outcome = G1FrameInjectGlobalError;
        reset_production_trace(
            global.external,
            global.runtime.accepted_state.camera_azimuth);
        check(g1_frame_transaction_run(
                  global.runtime,
                  g1_controller_frame_stage_run,
                  global.external,
                  &seam,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionGlobalError,
              "global seam returns a controlled transaction error");
        check_production_trace_through(
            stage,
            "global seam runs only after the real stage and stops later stages");
        check_production_global_preservation(
            global.runtime,
            global_before,
            "global real checkpoint preserves state, identities, publication, and diagnostic");
        check_const_artifacts(
            global.external,
            global_artifacts,
            "global checkpoint leaves every const production artifact exact");
    }
}

static void test_each_other_mode_has_real_atomic_checkpoint()
{
    struct ModeCheckpoint
    {
        g1_test_mode mode;
        G1FrameTransactionStage stage;
    };
    const ModeCheckpoint cases[] = {
        {G1_TestLive, G1FrameStageInputRouteCommand},
        {G1_TestSequential, G1FrameStageMatcherSearch},
        {G1_TestFlat, G1FrameStageSupportRetarget},
        {G1_TestRoute, G1FrameStageFootprintObservation},
        {G1_TestSceneCycle, G1FrameStageAcceptedFinalize},
    };
    for (const ModeCheckpoint& test : cases) {
        fixture finite;
        configure_production_mode(finite, test.mode);
        const ProductionEvidence before = production_evidence(finite.runtime);
        const StateStorageIdentities working_before =
            state_storage_identities(finite.runtime.working_state);
        const G1FramePublication publication_before =
            finite.runtime.publication;
        const ConstArtifactEvidence artifacts_before =
            const_artifact_evidence(finite.external);
        G1FrameTransactionTestSeam seam;
        seam.hook = production_hook;
        seam.control.injected_stage = test.stage;
        seam.control.injected_outcome = G1FrameInjectFiniteReject;
        char error[1024] = {};
        reset_production_trace(
            finite.external, finite.runtime.accepted_state.camera_azimuth);
        check(g1_frame_transaction_run(
                  finite.runtime,
                  g1_controller_frame_stage_run,
                  finite.external,
                  &seam,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionFiniteRejected,
              error);
        check_production_trace_through(
            static_cast<int>(test.stage),
            "mode-specific finite checkpoint executes only its real prefix");
        check(state_logical_digest(finite.runtime.accepted_state) ==
                  before.accepted &&
                  diagnostic_logical_digest(
                      finite.runtime.accepted_diagnostic) ==
                      before.diagnostic &&
                  same_storage_identities(
                      state_storage_identities(
                          finite.runtime.working_state),
                      working_before),
              "mode-specific finite checkpoint preserves accepted values and all storage");
        G1FramePublication expected = publication_before;
        expected.requested_intent = production_trace.requested_intent;
        expected.rejection = production_finite_rejection();
        expected.ik_safe_stop_latched = true;
        expected.presentation_frame =
            finite.external.input.presentation_frame;
        check(publication_logical_digest(finite.runtime.publication) ==
                  publication_logical_digest(expected),
              "mode-specific finite checkpoint keeps the exact whitelist");
        check_const_artifacts(
            finite.external,
            artifacts_before,
            "mode-specific finite checkpoint preserves immutable artifacts");

        fixture global;
        configure_production_mode(global, test.mode);
        const ProductionEvidence global_before =
            production_evidence(global.runtime);
        const ConstArtifactEvidence global_artifacts =
            const_artifact_evidence(global.external);
        seam.control.injected_outcome = G1FrameInjectGlobalError;
        reset_production_trace(
            global.external, global.runtime.accepted_state.camera_azimuth);
        check(g1_frame_transaction_run(
                  global.runtime,
                  g1_controller_frame_stage_run,
                  global.external,
                  &seam,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionGlobalError,
              "mode-specific global checkpoint returns controlled failure");
        check_production_trace_through(
            static_cast<int>(test.stage),
            "mode-specific global checkpoint executes only its real prefix");
        check_production_global_preservation(
            global.runtime,
            global_before,
            "mode-specific global checkpoint preserves all published owners");
        check_const_artifacts(
            global.external,
            global_artifacts,
            "mode-specific global checkpoint preserves immutable artifacts");
    }
}

static void test_real_input_checkpoint_publication()
{
    fixture value;
    configure_production_mode(value, G1_TestLive);
    const ProductionEvidence before = production_evidence(value.runtime);
    G1FrameTransactionTestSeam seam;
    seam.hook = production_hook;
    seam.control.injected_stage = G1FrameStageInputRouteCommand;
    seam.control.injected_outcome = G1FrameInjectFiniteReject;
    char error[1024] = {};
    reset_production_trace(
        value.external, value.runtime.accepted_state.camera_azimuth);
    check(g1_frame_transaction_run(
              value.runtime,
              g1_controller_frame_stage_run,
              value.external,
              &seam,
              error,
              static_cast<int>(sizeof(error))) ==
              G1FrameTransactionFiniteRejected,
          error);
    check(production_trace.requested_intent_ready &&
              same_intent_bits(
                  value.runtime.publication.requested_intent,
                  production_trace.requested_intent) &&
              state_logical_digest(value.runtime.accepted_state) ==
                  before.accepted &&
              diagnostic_logical_digest(
                  value.runtime.accepted_diagnostic) == before.diagnostic,
          "input checkpoint publishes exact ready intent without accepted mutation");
}

static void test_production_contact_tuning_preflight()
{
    for (int invalid = 0; invalid < 6; ++invalid) {
        fixture value;
        configure_production_mode(value, G1_TestLive);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        switch (invalid) {
        case 0: value.external.tuning.contact_unlock_radius = -0.01f; break;
        case 1: value.external.tuning.contact_unlock_radius = nan; break;
        case 2: value.external.tuning.contact_foot_height = -0.01f; break;
        case 3: value.external.tuning.contact_foot_height = nan; break;
        case 4:
            value.external.tuning.contact_blending_halflife = 0.0f;
            break;
        default:
            value.external.tuning.contact_blending_halflife = nan;
            break;
        }
        value.runtime.working_state.camera_azimuth += 0.125f;
        value.runtime.working_state.selected_cost += 0.25f;
        check(g1_controller_state_is_valid(value.runtime.working_state) &&
                  state_logical_digest(value.runtime.working_state) !=
                      state_logical_digest(value.runtime.accepted_state),
              "preflight destination is valid but logically distinct");
        const ProductionEvidence before = production_evidence(value.runtime);
        const ConstArtifactEvidence artifacts_before =
            const_artifact_evidence(value.external);
        G1FrameTransactionTestSeam seam;
        seam.hook = production_hook;
        char error[1024] = {};
        reset_production_trace(
            value.external, value.runtime.accepted_state.camera_azimuth);
        check(g1_frame_transaction_run(
                  value.runtime,
                  g1_controller_frame_stage_run,
                  value.external,
                  &seam,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionGlobalError,
              "invalid contact tuning is rejected by coordinator preflight");
        check(production_trace.total == 0,
              "contact tuning preflight rejects before the real runner or hook");
        check_production_preflight_preservation(
            value.runtime,
            before,
            "contact tuning preflight preserves accepted and working runtime exactly");
        check_const_artifacts(
            value.external,
            artifacts_before,
            "contact tuning preflight preserves every const artifact exactly");
    }
}

static void test_real_route_latch_consumption_and_resume()
{
    fixture value;
    configure_production_mode(value, G1_TestRoute);
    value.runtime.accepted_state.simulation_velocity =
        vec3(0.50f, -0.25f, -0.375f);
    value.runtime.accepted_state.simulation_acceleration =
        vec3(-0.125f, 0.1875f, 0.25f);
    check(g1_controller_state_is_valid(value.runtime.accepted_state),
          "route latch fixture remains a valid accepted state");
    const int route_index = value.runtime.accepted_state.route_index;
    const int route_waypoint = value.runtime.accepted_state.route_waypoint;
    const int route_frames = value.runtime.accepted_state.route_frames;
    const uint32_t velocity_y = terrain_float_bits(
        value.runtime.accepted_state.simulation_velocity.y);
    const uint32_t acceleration_y = terrain_float_bits(
        value.runtime.accepted_state.simulation_acceleration.y);
    const quat heading = value.runtime.accepted_state.desired_rotation;

    G1FrameTransactionTestSeam seam;
    seam.hook = production_hook;
    seam.control.injected_stage = G1FrameStageMatcherSearch;
    seam.control.injected_outcome = G1FrameInjectFiniteReject;
    char error[1024] = {};
    reset_production_trace(
        value.external, value.runtime.accepted_state.camera_azimuth);
    check(g1_frame_transaction_run(
              value.runtime,
              g1_controller_frame_stage_run,
              value.external,
              &seam,
              error,
              static_cast<int>(sizeof(error))) ==
              G1FrameTransactionFiniteRejected &&
              value.runtime.publication.ik_safe_stop_latched,
          error);

    value.external.input.presentation_frame = 74;
    seam.control.injected_outcome = G1FrameInjectGlobalError;
    const ProductionEvidence before_global =
        production_evidence(value.runtime);
    reset_production_trace(
        value.external, value.runtime.accepted_state.camera_azimuth);
    check(g1_frame_transaction_run(
              value.runtime,
              g1_controller_frame_stage_run,
              value.external,
              &seam,
              error,
              static_cast<int>(sizeof(error))) ==
              G1FrameTransactionGlobalError &&
              production_trace.prior_latch_seen,
          "a global retry receives but does not consume the finite latch");
    check_production_global_preservation(
        value.runtime,
        before_global,
        "global retry preserves the incoming latch and every accepted owner");

    value.external.input.presentation_frame = 75;
    seam.control.injected_stage = G1FrameStageCount;
    seam.control.injected_outcome = G1FrameInjectContinue;
    const ConstArtifactEvidence latch_artifacts =
        const_artifact_evidence(value.external);
    reset_production_trace(
        value.external, value.runtime.accepted_state.camera_azimuth);
    check(g1_frame_transaction_run(
              value.runtime,
              g1_controller_frame_stage_run,
              value.external,
              &seam,
              error,
              static_cast<int>(sizeof(error))) ==
              G1FrameTransactionAccepted,
          error);
    check_production_trace_through(
        G1FrameStageAcceptedFinalize,
        "latch consumption completes all twelve real stages");
    check_complete_success_publication(
        value.runtime,
        value.external,
        "latch consumption publishes canonical rejection payload, exact intent/frame, and the complete accepted diagnostic");
    check_complete_accepted_final_fk(
        value,
        "latch consumption publishes the completed independently checked final FK");
    check_const_artifacts(
        value.external,
        latch_artifacts,
        "accepted latch consumption preserves every immutable production artifact");
    check(production_trace.prior_latch_seen &&
              production_trace.latch_handoff.cancel_planar_inertia &&
              production_trace.latch_handoff.force_search &&
              terrain_float_bits(
                  production_trace.latch_handoff.applied_velocity.x) == 0U &&
              terrain_float_bits(
                  production_trace.latch_handoff.applied_velocity.z) == 0U &&
              terrain_float_bits(
                  production_trace.latch_handoff.applied_velocity.y) ==
                  terrain_float_bits(
                      production_trace.requested_intent
                          .requested_velocity.y),
          "the real input stage receives the exact planar-only safe-stop handoff");
    check(!value.runtime.publication.ik_safe_stop_latched &&
              value.runtime.accepted_state.route_index == route_index &&
              value.runtime.accepted_state.route_waypoint == route_waypoint &&
              value.runtime.accepted_state.route_frames == route_frames &&
              terrain_float_bits(
                  value.runtime.accepted_state.simulation_velocity.x) == 0U &&
              terrain_float_bits(
                  value.runtime.accepted_state.simulation_velocity.z) == 0U &&
              terrain_float_bits(
                  value.runtime.accepted_state.simulation_acceleration.x) ==
                  0U &&
              terrain_float_bits(
                  value.runtime.accepted_state.simulation_acceleration.z) ==
                  0U &&
              terrain_float_bits(
                  value.runtime.accepted_state.simulation_velocity.y) ==
                  velocity_y &&
              terrain_float_bits(
                  value.runtime.accepted_state.simulation_acceleration.y) ==
                  acceleration_y &&
              same_quat_bits(
                  value.runtime.accepted_state.desired_rotation, heading) &&
              same_quat_bits(
                  value.runtime.publication.requested_intent.desired_heading,
                  heading),
          "accepted retry consumes one latch, freezes the complete route cursor, "
          "zeros only planar inertia, and preserves vertical motion and heading");

    value.external.input.presentation_frame = 76;
    const ConstArtifactEvidence resume_artifacts =
        const_artifact_evidence(value.external);
    reset_production_trace(
        value.external, value.runtime.accepted_state.camera_azimuth);
    check(g1_frame_transaction_run(
              value.runtime,
              g1_controller_frame_stage_run,
              value.external,
              &seam,
              error,
              static_cast<int>(sizeof(error))) ==
              G1FrameTransactionAccepted &&
              !production_trace.prior_latch_seen &&
              value.runtime.accepted_state.route_frames > route_frames,
          "the next real frame resumes the same route after one frozen retry");
    check_production_trace_through(
        G1FrameStageAcceptedFinalize,
        "post-latch route resume completes all twelve real stages");
    check_complete_success_publication(
        value.runtime,
        value.external,
        "post-latch route resume publishes a complete canonical success payload");
    check_complete_accepted_final_fk(
        value,
        "post-latch route resume preserves exact final-FK ownership");
    check_const_artifacts(
        value.external,
        resume_artifacts,
        "post-latch route resume preserves every immutable production artifact");
}

static void configure_down_step_fixture(fixture& value)
{
    for (int frame = 0; frame < value.db.nframes(); ++frame) {
        value.db.contact_states(frame, 0) = false;
        value.db.contact_states(frame, 1) = true;
    }
    for (int current = 0; current <= 6; ++current) {
        value.db.contact_states(current, 0) = false;
        value.db.contact_states(current + 9, 0) = false;
        value.db.contact_states(current + 17, 0) = true;
        value.db.contact_states(current + 25, 0) = true;
    }
    for (int z = 0; z < value.scene.terrain.nz; ++z) {
        for (int x = 0; x < value.scene.terrain.nx; ++x) {
            const float world_x = value.scene.terrain.origin_x +
                value.scene.terrain.cell_size * static_cast<float>(x);
            value.scene.terrain.heights(
                z * value.scene.terrain.nx + x) =
                world_x >= 2.25f ? -0.32f : 0.0f;
        }
    }
    value.scene.metadata.routes.clear();
    value.scene.metadata.routes.push_back(
        production_route("down-step-route", 4.0f, 2.0f));
    value.external.tuning.mode = G1_TestRoute;
    value.external.tuning.frame_limit = 96;
    value.external.tuning.ik_enabled = true;
    value.external.tuning.initial_search_time = 0.375f;
    value.external.tuning.dt = 1.0f / 25.0f;
    value.external.tuning.trajectory_sample_time = 1.0f / 3.0f;
    value.external.tuning.route_speed = 0.50f;
    value.external.input.scripted_azimuth_delta = 0.0f;
    value.external.heading_override.active = false;
    G1FrameResetConfig config;
    config.route_mode = true;
    config.route_id = value.scene.metadata.routes[0].id.c_str();
    config.ik_enabled = true;
    config.dt = value.external.tuning.dt;
    config.trajectory_sample_time =
        value.external.tuning.trajectory_sample_time;
    config.initial_search_time = value.external.tuning.initial_search_time;
    char error[1024] = {};
    check(g1_frame_runtime_reset(
              value.runtime,
              value.db,
              value.support,
              value.scene,
              config,
              error,
              static_cast<int>(sizeof(error))),
          error);
    value.external.db = &value.db;
    value.external.support = &value.support;
    value.external.scene = &value.scene;
    value.external.route = &value.scene.metadata.routes[0];
}

static bool production_ik_forgery_applied = false;
static bool production_ik_forgery_structurally_valid = false;

static G1FrameStageOutcome forge_production_ik_rejection_normal(
    G1FrameTransactionStage stage,
    g1_controller_state& working_state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity)
{
    const G1FrameStageOutcome outcome = g1_controller_frame_stage_run(
        stage,
        working_state,
        scratch,
        external,
        error,
        error_capacity);
    if (outcome != G1FrameStageFiniteReject ||
        !scratch.rejection.attempted_ik_available) {
        return outcome;
    }
    for (int foot = 0; foot < 2; ++foot) {
        G1FootTarget& target =
            scratch.rejection.ik_frame.feet[foot].target;
        if (!g1_foot_target_is_valid(target)) continue;
        uint32_t forged_bits =
            terrain_float_bits(target.desired_sole_normal.y) ^ 1U;
        std::memcpy(
            &target.desired_sole_normal.y,
            &forged_bits,
            sizeof(forged_bits));
        production_ik_forgery_applied = true;
        production_ik_forgery_structurally_valid =
            g1_frame_rejection_is_valid(scratch.rejection);
        break;
    }
    return outcome;
}

static_assert(std::is_same<
                  decltype(&forge_production_ik_rejection_normal),
                  G1FrameStageRunner>::value,
              "the rejection forgery proxy has the exact runner type");

static void test_coordinator_rejects_structurally_valid_ik_forgery()
{
    fixture value;
    for (int frame = 0; frame < value.db.nframes(); ++frame) {
        value.db.contact_states(frame, 0) = false;
        value.db.contact_states(frame, 1) = false;
    }
    reset_ik_enabled_fixture(value, G1_TestSequential);
    force_swing_history_below_terrain(
        value.runtime.accepted_state, 0);
    value.external.input.presentation_frame = 201;
    const ProductionEvidence before = production_evidence(value.runtime);
    const ConstArtifactEvidence artifacts_before =
        const_artifact_evidence(value.external);
    production_ik_forgery_applied = false;
    production_ik_forgery_structurally_valid = false;
    char error[1024] = {};
    error[0] = '\0';
    check(g1_frame_transaction_run(
              value.runtime,
              forge_production_ik_rejection_normal,
              value.external,
              nullptr,
              error,
              static_cast<int>(sizeof(error))) ==
              G1FrameTransactionGlobalError &&
              production_ik_forgery_applied &&
              production_ik_forgery_structurally_valid,
          "the coordinator rejects a structurally valid one-bit IK rejection forgery");
    check_production_global_preservation(
        value.runtime,
        before,
        "the rejected IK forgery preserves accepted and published owners");
    check_const_artifacts(
        value.external,
        artifacts_before,
        "the rejected IK forgery preserves immutable production artifacts");
}

static void test_synthetic_non_descending_step_safe_stops_atomically()
{
    fixture value;
    configure_production_mode(value, G1_TestRoute);
    configure_down_step_fixture(value);

    bool pose_rows_repeat = true;
    for (int frame = 1; frame < value.db.nframes(); ++frame) {
        for (int bone = 0; bone < G1_BoneCount; ++bone) {
            pose_rows_repeat = pose_rows_repeat &&
                same_vec3_bits(
                    value.db.bone_positions(frame, bone),
                    value.db.bone_positions(0, bone)) &&
                same_quat_bits(
                    value.db.bone_rotations(frame, bone),
                    value.db.bone_rotations(0, bone));
        }
    }
    check(pose_rows_repeat,
          "the adversarial step fixture intentionally repeats one pose row");

    G1SurfaceSample upper_plateau;
    G1SurfaceSample lower_plateau;
    check(g1_surface_query_v2(
              upper_plateau, value.scene.terrain, 1.75f, 2.0f) ==
              G1SurfaceQueryValid &&
              g1_surface_query_v2(
                  lower_plateau, value.scene.terrain, 2.50f, 2.0f) ==
                  G1SurfaceQueryValid &&
              same_float_bits(upper_plateau.height, 0.0f) &&
              same_float_bits(lower_plateau.height, -0.32f),
          "the adversarial fixture retains its exact 0.32 m terrain step");

    int accepted_frames = 0;
    int finite_rejections = 0;
    int target_unreachable_rejections = 0;
    int landing_patch_rejections = 0;
    bool source_preserving_target_authenticated = false;
    bool impossible_recorded_lock_authenticated = false;
    char error[1024] = {};
    expect_down_step_schedule = true;
    for (int frame = 0; frame < 6; ++frame) {
        const int current = value.runtime.accepted_state.frame_index;
        check(current >= 0 && current <= 6 &&
                  !value.db.contact_states(current, 0) &&
                  !value.db.contact_states(current + 9, 0) &&
                  value.db.contact_states(current + 17, 0) &&
                  value.db.contact_states(current + 25, 0),
              "each adversarial attempt retains the exact contact horizon");

        const uint64_t accepted_before =
            state_logical_digest(value.runtime.accepted_state);
        const StateStorageIdentities accepted_storage_before =
            state_storage_identities(value.runtime.accepted_state);
        const uint64_t accepted_diagnostic_before =
            diagnostic_logical_digest(value.runtime.accepted_diagnostic);
        const int database_frame_before =
            value.runtime.accepted_state.frame_index;
        const int scene_frame_before =
            value.runtime.accepted_state.scene_frame;
        const int route_frames_before =
            value.runtime.accepted_state.route_frames;
        value.external.input.presentation_frame = 200 + frame;
        const ConstArtifactEvidence artifacts_before =
            const_artifact_evidence(value.external);
        G1FrameTransactionTestSeam seam;
        seam.hook = production_hook;
        reset_production_trace(
            value.external, value.runtime.accepted_state.camera_azimuth);
        error[0] = '\0';
        const G1FrameTransactionStatus status = g1_frame_transaction_run(
            value.runtime,
            g1_controller_frame_stage_run,
            value.external,
            &seam,
            error,
            static_cast<int>(sizeof(error)));
        check(status != G1FrameTransactionGlobalError, error);
        check_const_artifacts(
            value.external,
            artifacts_before,
            "each adversarial outcome preserves database, terrain, route, and external input");

        if (status == G1FrameTransactionAccepted) {
            ++accepted_frames;
            check(false,
                  "the impossible repeated-pose fixture never publishes accepted progress");
            continue;
        }

        ++finite_rejections;
        const G1FrameRejectionDiagnostic& rejection =
            value.runtime.publication.rejection;
        check(status == G1FrameTransactionFiniteRejected &&
                  value.runtime.publication.ik_safe_stop_latched &&
                  g1_frame_publication_is_valid(
                      value.runtime.publication) &&
                  g1_frame_rejection_is_valid(rejection) &&
                  rejection.rejected &&
                  state_logical_digest(value.runtime.accepted_state) ==
                      accepted_before &&
                  same_storage_identities(
                      state_storage_identities(
                          value.runtime.accepted_state),
                      accepted_storage_before) &&
                  diagnostic_logical_digest(
                      value.runtime.accepted_diagnostic) ==
                      accepted_diagnostic_before &&
                  value.runtime.accepted_state.frame_index ==
                      database_frame_before &&
                  value.runtime.accepted_state.scene_frame ==
                      scene_frame_before &&
                  value.runtime.accepted_state.route_frames ==
                      route_frames_before,
              "each impossible repeated-pose attempt publishes one finite stop without committing any accepted owner");

        if (rejection.stop_reason == G1IkStopTargetUnreachable) {
            ++target_unreachable_rejections;
            check(rejection.stage == G1FrameRejectIkCandidate &&
                      rejection.attempted_footprint_available &&
                      rejection.attempted_ik_available &&
                      g1_root_reach_plan_is_valid(
                          rejection.ik_frame.root_reach) &&
                      rejection.ik_frame.root_reach.active &&
                      !rejection.ik_frame.root_reach
                           .common_interval_found &&
                      !rejection.ik_frame.root_reach.applied &&
                      terrain_float_bits(
                          rejection.ik_frame.root_reach
                              .root_y_delta_m) == 0U,
                  "the impossible established lock rejects at the IK candidate stage");

            const G1FootFrameResult& left = rejection.ik_frame.feet[0];
            const G1FootFrameResult& right = rejection.ik_frame.feet[1];
            const G1FootprintFootObservation& left_lookahead =
                rejection.attempted_footprint.feet[0];
            const G1LegConfig left_config = g1_left_leg_config();
            const G1LegConfig right_config = g1_right_leg_config();
            vec3 current_left_sole;
            vec3 current_left_normal;
            vec3 current_right_sole;
            G1SurfaceSample current_left_surface;
            G1SurfaceSample future_left_surface;
            G1SurfaceTarget expected_current_left_target;
            char surface_error[256] = {};
            check(g1_ik_checked_physical_sole_centroid(
                      current_left_sole,
                      value.runtime.working_state
                          .global_bone_positions(left_config.contact),
                      value.runtime.working_state
                          .global_bone_rotations(left_config.contact),
                      left_config) &&
                      ik_checked_quat_rotate(
                          current_left_normal,
                          value.runtime.working_state
                              .global_bone_rotations(left_config.contact),
                          left_config.sole_normal_local) &&
                      g1_ik_checked_physical_sole_centroid(
                          current_right_sole,
                          value.runtime.working_state
                              .global_bone_positions(right_config.contact),
                          value.runtime.working_state
                              .global_bone_rotations(right_config.contact),
                          right_config) &&
                      g1_surface_query_v2(
                          current_left_surface,
                          value.scene.terrain,
                          current_left_sole.x,
                          current_left_sole.z) ==
                          G1SurfaceQueryValid &&
                      g1_surface_query_v2(
                          future_left_surface,
                          value.scene.terrain,
                          left_lookahead
                              .predicted_landing_sole_center.x,
                          left_lookahead
                              .predicted_landing_sole_center.z) ==
                          G1SurfaceQueryValid &&
                      g1_surface_target_sample(
                          expected_current_left_target,
                          value.scene.terrain,
                          current_left_sole.x,
                          current_left_sole.z,
                          left_config.planted_clearance_m,
                          surface_error,
                          static_cast<int>(sizeof(surface_error))),
                  surface_error[0] == '\0'
                      ? "the rejected repeated pose has independently queried physical sole and terrain endpoints"
                      : surface_error);
            const bool future_xz_is_distinct =
                !same_float_bits(
                    left_lookahead.predicted_landing_sole_center.x,
                    current_left_sole.x) ||
                !same_float_bits(
                    left_lookahead.predicted_landing_sole_center.z,
                    current_left_sole.z);
            check(frame == 0 &&
                      !left.recorded_contact &&
                      left_lookahead.landing_expected &&
                      left_lookahead.landing_patch_ready &&
                      same_vec3_bits(
                          left.target.sole_center,
                          current_left_sole) &&
                      same_vec3_bits(
                          left.target.desired_sole_normal,
                          current_left_normal) &&
                      same_vec3_bits(
                          left.target.surface.point,
                          expected_current_left_target.point) &&
                      same_vec3_bits(
                          left.target.surface.normal,
                          expected_current_left_target.normal) &&
                      same_vec3_bits(
                          expected_current_left_target.normal,
                          current_left_surface.normal) &&
                      same_float_bits(
                          left_lookahead
                              .predicted_landing_surface.height,
                          future_left_surface.height) &&
                      same_vec3_bits(
                          left_lookahead
                              .predicted_landing_surface.normal,
                          future_left_surface.normal) &&
                      future_xz_is_distinct &&
                      left_lookahead.predicted_landing_surface.height -
                          left.target.surface.point.y < -0.15f,
                  "independent current and future terrain queries prove the complete live target surface is source-preserving while distinct lower terrain remains validation-only lookahead");
            check(right.recorded_contact &&
                      right.position.applied &&
                      !right.position.reachable &&
                      right.position.safe_stop_requested &&
                      std::fabs(
                          static_cast<double>(
                              right.target.sole_center.y) -
                          static_cast<double>(
                              current_right_sole.y)) > 0.40,
                  "the repeated pose cannot satisfy its established recorded-contact lock");
            source_preserving_target_authenticated = true;
            impossible_recorded_lock_authenticated = true;
        } else if (rejection.stop_reason ==
                       G1IkStopLandingPatchUnavailable) {
            ++landing_patch_rejections;
            check(frame >= 1 && frame <= 5 &&
                      rejection.stage == G1FrameRejectLandingPatch,
                  "each of the five post-stop retries rejects at the unavailable landing-patch boundary");
        } else {
            check(false,
                  "the adversarial fixture returned an unexpected finite stop");
        }
    }
    expect_down_step_schedule = false;

    check(accepted_frames == 0 &&
              finite_rejections == 6 &&
              target_unreachable_rejections == 1 &&
              landing_patch_rejections == 5 &&
              source_preserving_target_authenticated &&
              impossible_recorded_lock_authenticated &&
              value.runtime.accepted_state.frame_index == 0 &&
              value.runtime.accepted_state.scene_frame == 0 &&
              value.runtime.accepted_state.route_frames == 0,
          "the non-descending six-frame fixture supplies pure safe-stop evidence: one impossible-lock rejection then five unavailable-patch retries, all atomic at zero cursors");
}

static void test_quantized_planner_terminal_rejects_atomically()
{
    fixture direct;
    fixture coordinator;
    configure_quantized_all_contact_fixture(direct, false);
    configure_quantized_all_contact_fixture(coordinator, false);
    const ProductionEvidence direct_before =
        production_evidence(direct.runtime);
    const ConstArtifactEvidence direct_artifacts_before =
        const_artifact_evidence(direct.external);
    G1FrameTransactionScratch scratch;
    G1FrameTransactionStage terminal = G1FrameStageCount;
    char error[1024] = {};
    check(run_real_stage_prefix(
              direct,
              scratch,
              G1FrameStageIkSecondFoot,
              terminal,
              error,
              static_cast<int>(sizeof(error))) ==
                  G1FrameStageFiniteReject &&
              terminal == G1FrameStageIkSecondFoot,
          "quantized planner terminal finite-rejects at the real second-foot stage");
    const G1IkFrameTransaction& transaction =
        scratch.ik_certificate.ik_transaction;
    const G1IkFrameResult& attempted =
        transaction.candidate_result;
    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    check(transaction.initialized &&
              transaction.next_foot == 2U &&
              attempted.safe_stop_requested &&
              attempted.stop_reason == G1IkStopTargetUnreachable &&
              attempted.root_reach.active &&
              !attempted.root_reach.common_interval_found &&
              !attempted.root_reach.applied &&
              terrain_float_bits(
                  attempted.root_reach.root_y_delta_m) == 0U &&
              g1_frame_successful_foot_is_valid(
                  attempted.feet[0], configs[0]) &&
              g1_frame_successful_foot_is_valid(
                  attempted.feet[1], configs[1]) &&
              !attempted.feet[0].position.safe_stop_requested &&
              !attempted.feet[0].orientation.safe_stop_requested &&
              !attempted.feet[1].position.safe_stop_requested &&
              !attempted.feet[1].orientation.safe_stop_requested &&
              scratch.rejection.rejected &&
              scratch.rejection.stage == G1FrameRejectIkCandidate &&
              scratch.rejection.attempted_ik_available &&
              g1_frame_ik_result_equal(
                  scratch.rejection.ik_frame, attempted),
          "quantized planner terminal retains two successful feet and one frame-level stop");
    check_real_rejection_snapshot_table(
        transaction, G1IkRejectionAfterFoot1);
    check_real_coordinator_ik_rejection(
        coordinator,
        direct,
        direct_before,
        direct_artifacts_before,
        G1FrameStageIkSecondFoot,
        scratch.rejection);
}

static void test_pose_certificate_discards_working_plan_and_prior_owners()
{
    fixture value;
    configure_quantized_all_contact_fixture(value, true);
    char error[1024] = {};
    const G1RootReachPlan prior_plan =
        value.runtime.accepted_state.ik_frame.root_reach;
    check(g1_root_reach_plan_is_valid(prior_plan) &&
              !prior_plan.active,
          "pose-certificate rollback starts from a canonical prior accepted plan");
    const ProductionEvidence before =
        production_evidence(value.runtime);
    const ConstArtifactEvidence artifacts_before =
        const_artifact_evidence(value.external);
    G1FrameTransactionTestSeam seam;
    seam.hook = production_hook;
    seam.control.injected_stage = G1FrameStageIkPoseCertificate;
    seam.control.injected_outcome = G1FrameInjectGlobalError;
    error[0] = '\0';
    reset_production_trace(
        value.external,
        value.runtime.accepted_state.camera_azimuth);
    const G1FrameTransactionStatus status =
        g1_frame_transaction_run(
            value.runtime,
            g1_controller_frame_stage_run,
            value.external,
            &seam,
            error,
            static_cast<int>(sizeof(error)));
    check(status == G1FrameTransactionGlobalError,
          "the pose-certificate checkpoint returns the injected global rollback");
    check_production_trace_through(
        G1FrameStageIkPoseCertificate,
        "the pose-certificate rollback executes the complete real attempted frame");
    const G1RootReachPlan& attempted_plan =
        production_trace.final_candidate_result.root_reach;
    const G1RootReachPlan& retained_plan =
        value.runtime.accepted_state.ik_frame.root_reach;
    check(production_trace.final_fk_completed &&
              !production_trace.accepted_diagnostic_ready &&
              g1_root_reach_plan_is_valid(attempted_plan) &&
              attempted_plan.active &&
              attempted_plan.common_interval_found &&
              !attempted_plan.applied &&
              terrain_float_bits(
                  attempted_plan.root_y_delta_m) == 0U &&
              terrain_float_bits(
                  production_trace.final_candidate_result.feet[0]
                      .position.contact_residual_m) ==
                  UINT32_C(0x33820000) &&
              terrain_float_bits(
                  production_trace.final_candidate_result.feet[1]
                      .position.contact_residual_m) ==
                  UINT32_C(0x33820000) &&
              g1_frame_successful_foot_is_valid(
                  production_trace.final_candidate_result.feet[0],
                  g1_left_leg_config()) &&
              g1_frame_successful_foot_is_valid(
                  production_trace.final_candidate_result.feet[1],
                  g1_right_leg_config()) &&
              g1_root_reach_plan_is_valid(retained_plan) &&
              retained_plan.active == prior_plan.active &&
              retained_plan.common_interval_found ==
                  prior_plan.common_interval_found &&
              retained_plan.applied == prior_plan.applied &&
              terrain_float_bits(retained_plan.root_y_delta_m) ==
                  terrain_float_bits(prior_plan.root_y_delta_m),
          "the recorded-contact working plan reaches pose certification but cannot replace the prior accepted plan");
    check_production_global_preservation(
        value.runtime,
        before,
        "pose-certificate rollback preserves prior support, locks/history, command, route, travel, heading, publication, and diagnostic owners");
    check_const_artifacts(
        value.external,
        artifacts_before,
        "pose-certificate rollback preserves all immutable production artifacts");
}

static constexpr int RealIncumbentSelectedFrame = 32;
static constexpr int RealIncumbentExecutedFrame = 33;
static constexpr int RealCandidateASelectedFrame = 64;
static constexpr int RealCandidateAExecutedFrame = 65;
static constexpr int RealCandidateBSelectedFrame = 96;
static constexpr int RealCandidateBExecutedFrame = 97;
static constexpr int RealCandidateCSelectedFrame = 128;
static constexpr int RealCandidateCExecutedFrame = 129;

static bool real_candidate_record_bits_equal(
    const G1CandidateRecord& first,
    const G1CandidateRecord& second)
{
    return first.kind == second.kind &&
           first.selected_frame == second.selected_frame &&
           first.executed_frame == second.executed_frame &&
           first.source_range == second.source_range &&
           terrain_float_bits(first.selected_cost) ==
               terrain_float_bits(second.selected_cost) &&
           first.recovery_rank == second.recovery_rank &&
           first.transitioned == second.transitioned;
}

static void configure_real_candidate_fixture(
    fixture& value,
    bool hostile_left_history,
    bool hostile_right_history,
    int incumbent_frame = RealIncumbentSelectedFrame,
    bool legacy_selects_incumbent = false)
{
    make_database(value.db, 160);
    value.support.values.resize(value.db.nframes(), 3);
    value.support.values.set(-1.0f);
    value.db.contact_states.set(true);
    value.scene.metadata.spawn_yaw = 0.0f;
    value.db.features_offset(22) = 1.0f;
    value.db.features_offset(24) = 1.0f;
    value.db.features_offset(26) = 1.0f;
    for (int frame = 0; frame < value.db.nframes(); ++frame) {
        value.db.features(frame, 15) = 4.0f;
    }
    value.db.features(RealIncumbentSelectedFrame, 15) =
        legacy_selects_incumbent ? 0.0f : 2.0f;
    value.db.features(159, 15) = 2.0f;
    value.db.features(RealCandidateASelectedFrame, 15) =
        legacy_selects_incumbent ? 4.0f : 0.0f;
    value.db.features(RealCandidateBSelectedFrame, 15) =
        legacy_selects_incumbent ? 4.0f : 0.5f;
    value.db.features(RealCandidateCSelectedFrame, 15) =
        legacy_selects_incumbent ? 4.0f : 0.75f;

    value.db.contact_states(RealCandidateAExecutedFrame, 0) = false;
    value.db.contact_states(RealCandidateAExecutedFrame, 1) = true;
    value.db.contact_states(RealCandidateBSelectedFrame, 0) = false;
    value.db.contact_states(RealCandidateBSelectedFrame, 1) = true;
    value.db.contact_states(RealCandidateBExecutedFrame, 0) = true;
    value.db.contact_states(RealCandidateBExecutedFrame, 1) = false;
    value.db.contact_states(RealCandidateCExecutedFrame, 0) = false;
    value.db.contact_states(RealCandidateCExecutedFrame, 1) = true;
    if (legacy_selects_incumbent) {
        value.db.contact_states(RealIncumbentExecutedFrame, 0) = false;
        value.db.contact_states(RealIncumbentExecutedFrame, 1) = true;
    }
    database_build_bounds(value.db);
    align_begin_fixture_source_support(value);
    shift_aligned_support_to_first_common_word(value);

    G1FrameResetConfig config;
    config.initial_search_time = 0.375f;
    config.ik_enabled = false;
    config.dt = 1.0f / 25.0f;
    config.trajectory_sample_time = 1.0f / 3.0f;
    char error[1024] = {};
    check(g1_frame_runtime_reset(
              value.runtime,
              value.db,
              value.support,
              value.scene,
              config,
              error,
              static_cast<int>(sizeof(error))),
          error);
    value.external = G1FrameExternalInputs{};
    value.external.db = &value.db;
    value.external.support = &value.support;
    value.external.scene = &value.scene;
    value.external.input.move_stick = vec3();
    value.external.input.look_stick = vec3();
    value.external.input.presentation_frame = 83;
    value.external.tuning.mode = G1_TestFlat;
    value.external.tuning.frame_limit = value.db.nframes();
    value.external.tuning.initial_search_time = config.initial_search_time;
    value.external.tuning.dt = config.dt;
    value.external.tuning.trajectory_sample_time =
        config.trajectory_sample_time;
    value.external.tuning.effective_terrain_weight = 0.0f;
    value.external.tuning.ik_enabled = false;

    g1_controller_state& accepted = value.runtime.accepted_state;
    accepted.frame_index = incumbent_frame;
    accepted.search_timer = 0.0f;
    accepted.force_search_timer = accepted.search_time;
    if (hostile_left_history) {
        force_swing_history_below_terrain(accepted, 0);
    } else {
        for (int probe = 0; probe < 4; ++probe) {
            accepted.ik.feet[0]
                .swing.previous_sphere_centers[probe].y += 0.05f;
        }
    }
    if (hostile_right_history) {
        force_swing_history_below_terrain(accepted, 1);
    } else {
        for (int probe = 0; probe < 4; ++probe) {
            accepted.ik.feet[1]
                .swing.previous_sphere_centers[probe].y += 0.05f;
        }
    }
    check(g1_controller_state_is_valid(accepted),
          "the real bounded-candidate source state is valid");
}

static G1FrameInjectedOutcome real_candidate_c_guard(
    G1FrameTransactionStage,
    const g1_controller_state&,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs&,
    const G1FrameTransactionTestControl&,
    char*,
    int)
{
    return scratch.active_candidate.selected_frame ==
               RealCandidateCSelectedFrame
        ? G1FrameInjectGlobalError
        : G1FrameInjectContinue;
}

static G1RecoveryProviderStatus real_provider_global_error(
    G1RecoveryCandidateSet&,
    const G1RecoveryRequest&,
    char*,
    int)
{
    return G1RecoveryProviderGlobalError;
}

static G1FrameTransactionStatus run_real_candidate_fixture(
    fixture& value,
    G1CandidateCertificationTrace& certification,
    G1RecoveryProvider provider,
    G1FrameTransactionTestHook hook,
    char* error,
    int error_capacity)
{
    G1FrameTransactionTestSeam seam;
    seam.hook = hook;
    seam.certification_trace = &certification;
    const G1FrameCoordinator coordinator =
        static_cast<G1FrameCoordinator>(&::g1_frame_transaction_run);
    return coordinator(
        value.runtime,
        g1_controller_frame_stage_run,
        provider,
        value.external,
        &seam,
        error,
        error_capacity);
}

static G1FrameTransactionStatus run_real_candidate_fixture(
    fixture& value,
    G1CandidateCertificationTrace& certification,
    char* error,
    int error_capacity)
{
    return run_real_candidate_fixture(
        value,
        certification,
        ::g1_recovery_candidates_build,
        real_candidate_c_guard,
        error,
        error_capacity);
}

static bool real_candidate_common_diagnostic_equal(
    const G1FrameAcceptedDiagnostic& first,
    const G1FrameAcceptedDiagnostic& second)
{
    if (first.ready != second.ready ||
        first.presentation_frame != second.presentation_frame ||
        first.scene_frame != second.scene_frame ||
        first.query_database_frame != second.query_database_frame ||
        first.query_range != second.query_range ||
        first.selected_database_frame !=
            second.selected_database_frame ||
        !g1_frame_pose_diagnostic_equal(
            first.raw_selected, second.raw_selected) ||
        !g1_frame_pose_diagnostic_equal(
            first.inertialized, second.inertialized) ||
        !g1_frame_pose_diagnostic_equal(
            first.support_retargeted,
            second.support_retargeted) ||
        first.matching_enabled != second.matching_enabled ||
        first.adjustment_enabled != second.adjustment_enabled ||
        first.clamping_enabled != second.clamping_enabled ||
        !g1_frame_float_bits_equal(
            first.effective_terrain_weight,
            second.effective_terrain_weight)) {
        return false;
    }
    for (int feature = 0; feature < 31; ++feature) {
        if (!g1_frame_float_bits_equal(
                first.query[feature], second.query[feature])) {
            return false;
        }
    }
    for (int sample = 0; sample < 4; ++sample) {
        if (!g1_frame_float_bits_equal(
                first.terrain_query.values[sample],
                second.terrain_query.values[sample]) ||
            !g1_frame_vec3_bits_equal(
                first.terrain_query.points[sample],
                second.terrain_query.points[sample])) {
            return false;
        }
    }
    return true;
}

static bool real_candidate_visible_projection_matches(
    const g1_controller_state& visible,
    const g1_controller_state& branch)
{
    return g1_frame_array_values_equal(
               visible.ik_bone_positions,
               branch.ik_bone_positions) &&
           g1_frame_array_values_equal(
               visible.ik_bone_rotations,
               branch.ik_bone_rotations) &&
           g1_frame_array_values_equal(
               visible.ik_global_bone_positions,
               branch.ik_global_bone_positions) &&
           g1_frame_array_values_equal(
               visible.ik_global_bone_rotations,
               branch.ik_global_bone_rotations) &&
           g1_frame_ik_result_equal(
               visible.ik_frame,
               branch.ik_frame) &&
           g1_controller_state_pose_clearance_equal(
               visible.ik_clearance,
               branch.ik_clearance);
}

static void check_real_abc_trace(
    const G1CandidateCertificationTrace& trace,
    const char* message)
{
    check(trace.attempt_count == 2U &&
              trace.legacy_traversals == 1U &&
              trace.recovery_provider_calls == 1U &&
              trace.common_evaluations == 2U &&
              trace.raw_evaluations == 2U &&
              trace.ik_evaluations == 2U &&
              trace.attempts[0].candidate.kind == G1CandidateLegacy &&
              trace.attempts[0].candidate.selected_frame ==
                  RealCandidateASelectedFrame &&
              trace.attempts[0].candidate.executed_frame ==
                  RealCandidateAExecutedFrame &&
              trace.attempts[0].candidate.source_range == 0 &&
              trace.attempts[0].score_owner == G1CandidateScoreLegacy &&
              trace.attempts[0].common ==
                  G1CandidateDispositionAccepted &&
              trace.attempts[0].raw ==
                  G1CandidateDispositionAccepted &&
              trace.attempts[0].ik ==
                  G1CandidateDispositionFiniteRejected &&
              trace.attempts[0].rejection_stage ==
                  G1FrameRejectIkCandidate &&
              trace.attempts[0].stop_reason ==
                  G1IkStopNoSwingCandidate &&
              trace.attempts[1].candidate.kind ==
                  G1CandidateRecoveryTransition &&
              trace.attempts[1].candidate.selected_frame ==
                  RealCandidateBSelectedFrame &&
              trace.attempts[1].candidate.executed_frame ==
                  RealCandidateBExecutedFrame &&
              trace.attempts[1].candidate.source_range == 0 &&
              trace.attempts[1].candidate.recovery_rank == 0U &&
              trace.attempts[1].score_owner ==
                  G1CandidateScoreStrictRecovery &&
              trace.attempts[1].common ==
                  G1CandidateDispositionAccepted &&
              trace.attempts[1].raw ==
                  G1CandidateDispositionAccepted &&
              trace.attempts[1].ik ==
                  G1CandidateDispositionAccepted,
          message);
}

static void run_real_abc_mode(
    fixture& value,
    G1CandidateCertificationTrace& certification,
    bool ik_enabled)
{
    configure_real_candidate_fixture(value, true, false);
    value.external.tuning.ik_enabled = ik_enabled;
    char error[1024] = {};
    check(run_real_candidate_fixture(
              value,
              certification,
              error,
              static_cast<int>(sizeof(error))) ==
              G1FrameTransactionAccepted,
          error[0] != '\0'
              ? error
              : "the real A/B recovery fixture accepts its first dual "
                "certificate candidate");
    check_real_abc_trace(
        certification,
        "real A/B certification has exact order, ownership, and branch outcomes");
}

static void test_real_abc_selects_b_in_both_modes()
{
    fixture raw;
    fixture ik;
    G1CandidateCertificationTrace raw_trace;
    G1CandidateCertificationTrace ik_trace;
    run_real_abc_mode(raw, raw_trace, false);
    run_real_abc_mode(ik, ik_trace, true);
    const g1_controller_state& raw_state = raw.runtime.accepted_state;
    const g1_controller_state& ik_state = ik.runtime.accepted_state;
    check(raw_state.frame_index == RealCandidateBExecutedFrame &&
              ik_state.frame_index == RealCandidateBExecutedFrame &&
              raw.runtime.accepted_diagnostic.selected_database_frame ==
                  RealCandidateBSelectedFrame &&
              ik.runtime.accepted_diagnostic.selected_database_frame ==
                  RealCandidateBSelectedFrame &&
              raw.runtime.accepted_diagnostic.query_database_frame ==
                  RealIncumbentSelectedFrame &&
              raw.runtime.accepted_diagnostic.query_range == 0 &&
              real_candidate_record_bits_equal(
                  raw_trace.attempts[1].candidate,
                  ik_trace.attempts[1].candidate) &&
              g1_frame_float_bits_equal(
                  raw_state.selected_cost,
                  raw_trace.attempts[1].candidate.selected_cost) &&
              g1_frame_float_bits_equal(
                  ik_state.selected_cost,
                  ik_trace.attempts[1].candidate.selected_cost) &&
              raw_state.selected_cost < raw_state.incumbent_cost &&
              ik_state.selected_cost < ik_state.incumbent_cost &&
              g1_frame_controller_states_equal(
                  raw.runtime.candidates.common_state,
                  ik.runtime.candidates.common_state) &&
              g1_frame_controller_states_equal(
                  raw.runtime.candidates.raw_state,
                  ik.runtime.candidates.raw_state) &&
              g1_frame_controller_states_equal(
                  raw.runtime.candidates.ik_state,
                  ik.runtime.candidates.ik_state) &&
              g1_frame_ik_state_equal(raw_state.ik, ik_state.ik) &&
              real_candidate_common_diagnostic_equal(
                  raw.runtime.accepted_diagnostic,
                  ik.runtime.accepted_diagnostic) &&
              raw_state.scene_frame == 1 && ik_state.scene_frame == 1 &&
              raw_state.route_frames == 0 && ik_state.route_frames == 0,
          "real B selection is mode-independent across query, cost, provenance, common, hidden, and lifecycle owners");
    check(real_candidate_visible_projection_matches(
              raw_state, raw.runtime.candidates.raw_state) &&
              g1_frame_ik_result_is_canonical(raw_state.ik_frame) &&
              raw_state.ik_candidate_clearance_status == G1ClearanceOk &&
              real_candidate_visible_projection_matches(
                  ik_state, ik.runtime.candidates.ik_state) &&
              ik_state.ik_frame.applied &&
              !ik_state.ik_frame.safe_stop_requested &&
              ik_state.ik_candidate_clearance_status == G1ClearanceOk,
          "raw publishes the canonical off branch while IK publishes the successful on branch");
}

static void test_real_a_raw_passes_and_ik_reports_no_swing_candidate()
{
    fixture value;
    G1CandidateCertificationTrace trace;
    run_real_abc_mode(value, trace, false);
    check(trace.attempts[0].raw == G1CandidateDispositionAccepted &&
              trace.attempts[0].ik ==
                  G1CandidateDispositionFiniteRejected &&
              trace.attempts[0].rejection_stage ==
                  G1FrameRejectIkCandidate &&
              trace.attempts[0].stop_reason ==
                  G1IkStopNoSwingCandidate,
          "real A passes raw certification and finite-rejects only at the unchanged IK swing boundary");
}

static void test_real_b_contact_release_is_not_an_eligibility_gate()
{
    fixture value;
    G1CandidateCertificationTrace trace;
    run_real_abc_mode(value, trace, true);
    check(value.db.contact_states(RealCandidateBSelectedFrame, 1) &&
              !value.db.contact_states(RealCandidateBExecutedFrame, 1) &&
              trace.attempts[1].candidate.selected_frame ==
                  RealCandidateBSelectedFrame &&
              trace.attempts[1].common ==
                  G1CandidateDispositionAccepted &&
              trace.attempts[1].raw ==
                  G1CandidateDispositionAccepted &&
              trace.attempts[1].ik ==
                  G1CandidateDispositionAccepted,
          "the strict B record executes and accepts across a real immediate right-contact release");
}

static void test_real_c_never_executes_after_b_accepts()
{
    fixture value;
    G1CandidateCertificationTrace trace;
    run_real_abc_mode(value, trace, false);
    check(trace.attempt_count == 2U &&
              trace.recovery_set.count == 3U &&
              trace.recovery_set.records[1].selected_frame ==
                  RealCandidateCSelectedFrame,
          "C is materialized by the strict provider but its fail-on-entry guard never executes after B accepts");
}

static void run_real_incumbent_fallback(
    fixture& value,
    G1CandidateCertificationTrace& trace,
    int incumbent_frame)
{
    configure_real_candidate_fixture(
        value, true, true, incumbent_frame, false);
    char error[1024] = {};
    check(run_real_candidate_fixture(
              value,
              trace,
              ::g1_recovery_candidates_build,
              nullptr,
              error,
              static_cast<int>(sizeof(error))) ==
              G1FrameTransactionAccepted,
          error[0] != '\0'
              ? error
              : "the real incumbent accepts after all three ranked "
                "recovery candidates finite-reject");
}

static void test_real_incumbent_runs_last_with_clamped_plus_one()
{
    fixture value;
    G1CandidateCertificationTrace trace;
    run_real_incumbent_fallback(
        value, trace, RealIncumbentSelectedFrame);
    check(trace.attempt_count == 4U &&
              trace.attempts[0].candidate.selected_frame ==
                  RealCandidateASelectedFrame &&
              trace.attempts[1].candidate.selected_frame ==
                  RealCandidateBSelectedFrame &&
              trace.attempts[2].candidate.selected_frame ==
                  RealCandidateCSelectedFrame &&
              trace.attempts[3].candidate.kind == G1CandidateIncumbent &&
              trace.attempts[3].candidate.selected_frame ==
                  RealIncumbentSelectedFrame &&
              trace.attempts[3].candidate.executed_frame ==
                  RealIncumbentExecutedFrame &&
              trace.attempts[3].score_owner ==
                  G1CandidateScoreIncumbent &&
              trace.attempts[3].common ==
                  G1CandidateDispositionAccepted &&
              trace.attempts[3].raw ==
                  G1CandidateDispositionAccepted &&
              trace.attempts[3].ik ==
                  G1CandidateDispositionAccepted &&
              value.runtime.accepted_state.frame_index ==
                  RealIncumbentExecutedFrame &&
              !value.runtime.accepted_state.transitioned &&
              value.runtime.accepted_diagnostic.selected_database_frame ==
                  RealIncumbentSelectedFrame,
          "A, B, and C finite-reject in rank order before one incumbent executes ordinary clamped plus one");
}

static void test_real_legacy_incumbent_is_not_retried()
{
    fixture value;
    configure_real_candidate_fixture(
        value, true, false, RealIncumbentSelectedFrame, true);
    G1CandidateCertificationTrace trace;
    char error[1024] = {};
    check(run_real_candidate_fixture(
              value,
              trace,
              ::g1_recovery_candidates_build,
              nullptr,
              error,
              static_cast<int>(sizeof(error))) ==
              G1FrameTransactionFiniteRejected,
          error);
    check(trace.attempt_count == 1U &&
              trace.legacy_traversals == 1U &&
              trace.recovery_provider_calls == 1U &&
              trace.attempts[0].candidate.kind == G1CandidateLegacy &&
              trace.attempts[0].candidate.selected_frame ==
                  RealIncumbentSelectedFrame &&
              trace.recovery_set.count == 0U,
          "a scheduled legacy record at the incumbent is certified once and never appended for retry");
}

static void test_real_provider_global_error_rolls_back_every_owner()
{
    fixture value;
    configure_real_candidate_fixture(value, true, false);
    const ProductionEvidence before = production_evidence(value.runtime);
    const ConstArtifactEvidence artifacts_before =
        const_artifact_evidence(value.external);
    G1CandidateCertificationTrace trace;
    char error[1024] = {};
    check(run_real_candidate_fixture(
              value,
              trace,
              real_provider_global_error,
              nullptr,
              error,
              static_cast<int>(sizeof(error))) ==
              G1FrameTransactionGlobalError,
          "a strict-provider global error aborts the real transaction");
    check(trace.attempt_count == 1U &&
              trace.recovery_provider_calls == 1U &&
              trace.attempts[0].stop_reason ==
                  G1IkStopNoSwingCandidate,
          "provider failure follows one authentic A rejection");
    check_production_global_preservation(
        value.runtime,
        before,
        "provider global error preserves accepted, publication, diagnostic, and storage owners");
    check_const_artifacts(
        value.external,
        artifacts_before,
        "provider global error preserves every immutable owner");
}

static void test_legacy_slot_zero_acceptance_has_exact_pre_feature_public_owners()
{
    for (int mode = 0; mode < 2; ++mode) {
        fixture value;
        configure_real_candidate_fixture(value, false, false);
        value.external.tuning.ik_enabled = mode != 0;
        G1CandidateCertificationTrace trace;
        char error[1024] = {};
        check(run_real_candidate_fixture(
                  value,
                  trace,
                  ::g1_recovery_candidates_build,
                  nullptr,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionAccepted,
              error);
        bool positive_zero_query = true;
        for (int feature = 0; feature < 31; ++feature) {
            const float normalized =
                (value.runtime.accepted_diagnostic.query[feature] -
                 value.db.features_offset(feature)) /
                value.db.features_scale(feature);
            positive_zero_query = positive_zero_query &&
                terrain_float_bits(normalized) == 0U;
        }
        check(positive_zero_query &&
                  trace.attempt_count == 1U &&
                  trace.legacy_traversals == 1U &&
                  trace.recovery_provider_calls == 0U &&
                  trace.attempts[0].candidate.kind ==
                      G1CandidateLegacy &&
                  trace.attempts[0].candidate.selected_frame ==
                      RealCandidateASelectedFrame &&
                  trace.attempts[0].candidate.executed_frame ==
                      RealCandidateAExecutedFrame &&
                  trace.attempts[0].candidate.source_range == 0 &&
                  terrain_float_bits(
                      trace.attempts[0].candidate.selected_cost) ==
                      terrain_float_bits(1.0f) &&
                  value.runtime.accepted_state.frame_index ==
                      RealCandidateAExecutedFrame &&
                  value.runtime.accepted_state.transitioned &&
                  terrain_float_bits(
                      value.runtime.accepted_state.incumbent_cost) ==
                      terrain_float_bits(4.0f) &&
                  terrain_float_bits(
                      value.runtime.accepted_state.selected_cost) ==
                      terrain_float_bits(1.0f) &&
                  terrain_float_bits(
                      value.runtime.accepted_state
                          .selected_terrain_error) == 0U &&
                  value.runtime.accepted_diagnostic
                          .query_database_frame ==
                      RealIncumbentSelectedFrame &&
                  value.runtime.accepted_diagnostic.query_range == 0 &&
                  value.runtime.accepted_diagnostic
                          .selected_database_frame ==
                      RealCandidateASelectedFrame,
              "legacy slot zero retains every exact pre-feature query, frame, transition, cost, terrain, and provenance owner");
    }
}

static void test_legacy_slot_zero_acceptance_never_materializes_recovery()
{
    fixture value;
    configure_real_candidate_fixture(value, false, false);
    G1CandidateCertificationTrace trace;
    char error[1024] = {};
    check(run_real_candidate_fixture(
              value,
              trace,
              ::g1_recovery_candidates_build,
              nullptr,
              error,
              static_cast<int>(sizeof(error))) ==
              G1FrameTransactionAccepted,
          error);
    check(trace.attempt_count == 1U &&
              trace.legacy_traversals == 1U &&
              trace.recovery_provider_calls == 0U &&
              !trace.recovery_request_available &&
              trace.recovery_set.count == 0U,
          "accepted slot zero has counters (1,0), one attempt, and no recovery materialization");
}

static void test_matching_disabled_and_unscheduled_frames_attempt_only_incumbent()
{
    for (int variant = 0; variant < 2; ++variant) {
        fixture value;
        configure_real_candidate_fixture(value, false, false);
        if (variant == 0) {
            value.external.tuning.mode = G1_TestSequential;
            value.external.tuning.frame_limit = 1;
        } else {
            value.runtime.accepted_state.search_timer =
                value.runtime.accepted_state.search_time;
        }
        G1CandidateCertificationTrace trace;
        char error[1024] = {};
        check(run_real_candidate_fixture(
                  value,
                  trace,
                  ::g1_recovery_candidates_build,
                  nullptr,
                  error,
                  static_cast<int>(sizeof(error))) ==
                  G1FrameTransactionAccepted,
              error);
        check(trace.attempt_count == 1U &&
                  trace.legacy_traversals == 0U &&
                  trace.recovery_provider_calls == 0U &&
                  trace.attempts[0].candidate.kind ==
                      G1CandidateIncumbent &&
                  trace.attempts[0].score_owner ==
                      G1CandidateScoreIncumbent,
              "matching-disabled and unscheduled frames each attempt only the incumbent");
    }
}

static void test_end_of_animation_public_sentinels_do_not_leak_private_score()
{
    fixture value;
    G1CandidateCertificationTrace trace;
    run_real_incumbent_fallback(value, trace, 159);
    check(trace.attempt_count == 4U &&
              trace.attempts[3].candidate.kind == G1CandidateIncumbent &&
              trace.attempts[3].candidate.selected_frame == 159 &&
              trace.attempts[3].candidate.executed_frame == 159 &&
              terrain_float_bits(
                  trace.attempts[3].candidate.selected_cost) ==
                  terrain_float_bits(FLT_MAX) &&
              terrain_float_bits(
                  value.runtime.accepted_state.incumbent_cost) ==
                  terrain_float_bits(FLT_MAX) &&
              terrain_float_bits(
                  value.runtime.accepted_state.selected_cost) ==
                  terrain_float_bits(FLT_MAX),
          "end-of-animation incumbent preserves exact public FLT_MAX words instead of its private strict score");
}

static ExactFunctionRange real_controller_runner_range(
    const std::vector<CppToken>& tokens)
{
    const std::vector<std::string> signature = {
        "G1FrameStageOutcome", "g1_controller_frame_stage_run", "(",
        "G1FrameTransactionStage", "stage", ",",
        "g1_controller_state", "&", "working_state", ",",
        "G1FrameTransactionScratch", "&", "scratch", ",",
        "const", "G1FrameExternalInputs", "&", "external", ",",
        "char", "*", "error", ",", "int", "error_capacity", ")",
    };
    ExactFunctionRange runner;
    check(cpp_exact_function_range(tokens, signature, runner),
          "the exact production stage runner is present");
    return runner;
}

static void test_stage_runner_trusted_call_closure_is_exact()
{
    const std::string source = read_source_file("controller.cpp");
    const NoMainSourceView no_main_view = no_main_source_view(source);
    std::string error;
    std::string closure_source;
    check(build_authenticated_production_no_main_closure_source(
              no_main_view,
              closure_source,
              error),
          error.empty()
              ? "the legacy query builder has one exact authenticated out-of-line boundary"
              : error.c_str());
    error.clear();
    check(analyze_no_main_root_closure(
              closure_source,
              "g1_controller_frame_stage_run",
              production_trusted_no_main_calls(),
              error),
          error.empty()
              ? "the expanded production runner has one exact trusted no-main closure"
              : error.c_str());
}

static void test_controller_contains_one_production_database_search_call()
{
    std::string error;
    const std::vector<CppToken> tokens = tokenize_cpp_source(
        read_source_file("controller.cpp"), error);
    check(error.empty(), error.empty() ? "controller tokenizes" : error.c_str());
    const ExactFunctionRange runner = real_controller_runner_range(tokens);
    const std::vector<CppCallRecord> all_searches = cpp_calls_named(
        tokens, 0, tokens.size(), "database_search");
    const std::vector<CppCallRecord> runner_searches = cpp_calls_named(
        tokens, runner.body_begin + 1, runner.body_end, "database_search");
    const std::vector<CppCallRecord> runner_providers = cpp_calls_named(
        tokens,
        runner.body_begin + 1,
        runner.body_end,
        "g1_recovery_candidates_build");
    const std::vector<std::size_t> matcher_cases = cpp_find_token_sequence(
        tokens,
        runner.body_begin + 1,
        runner.body_end,
        {"case", "G1FrameStageMatcherSearch", ":"});
    const std::vector<std::size_t> apply_cases = cpp_find_token_sequence(
        tokens,
        runner.body_begin + 1,
        runner.body_end,
        {"case", "G1FrameStageCandidateApply", ":"});
    check(all_searches.size() == 1U &&
              runner_searches.size() == 1U &&
              runner_searches[0].key == "::database_search" &&
              matcher_cases.size() == 1U &&
              apply_cases.size() == 1U &&
              runner_searches[0].name > matcher_cases[0] &&
              runner_searches[0].name < apply_cases[0] &&
              runner_providers.empty(),
          "controller production code has one database_search, only MatcherSearch owns it, and the runner never calls recovery");
}

static void test_controller_has_no_candidate_contact_rank_gate()
{
    std::string error;
    const std::vector<CppToken> tokens = tokenize_cpp_source(
        read_source_file("controller.cpp"), error);
    check(error.empty(), error.empty() ? "controller tokenizes" : error.c_str());
    const ExactFunctionRange runner = real_controller_runner_range(tokens);
    const std::vector<std::size_t> apply_cases = cpp_find_token_sequence(
        tokens,
        runner.body_begin + 1,
        runner.body_end,
        {"case", "G1FrameStageCandidateApply", ":"});
    const std::vector<std::size_t> inertial_cases = cpp_find_token_sequence(
        tokens,
        runner.body_begin + 1,
        runner.body_end,
        {"case", "G1FrameStageInertialization", ":"});
    check(apply_cases.size() == 1U && inertial_cases.size() == 1U,
          "candidate-apply case has an exact bounded range");
    const std::vector<std::size_t> active_outside =
        cpp_find_token_sequence(
            tokens,
            inertial_cases[0],
            runner.body_end,
            {"scratch", ".", "active_candidate"});
    check(active_outside.empty(),
          "no contact, footprint, certificate, or finalize stage can rank-gate an active candidate");
}

static void test_controller_has_no_persistent_dual_certificate_owner()
{
    std::string error;
    const std::vector<CppToken> tokens = tokenize_cpp_source(
        read_source_file("controller.cpp"), error);
    check(error.empty(), error.empty() ? "controller tokenizes" : error.c_str());
    check(cpp_find_token_sequence(
                  tokens,
                  0,
                  tokens.size(),
                  {"state", ".", "raw_certificate"}).empty() &&
              cpp_find_token_sequence(
                  tokens,
                  0,
                  tokens.size(),
                  {"state", ".", "ik_certificate"}).empty() &&
              cpp_find_token_sequence(
                  tokens,
                  0,
                  tokens.size(),
                  {"working_state", ".", "raw_certificate"}).empty() &&
              cpp_find_token_sequence(
                  tokens,
                  0,
                  tokens.size(),
                  {"working_state", ".", "ik_certificate"}).empty() &&
              cpp_find_token_sequence(
                  tokens,
                  0,
                  tokens.size(),
                  {"G1FrameBranchCertificateScratch",
                   "raw_certificate"}).empty() &&
              cpp_find_token_sequence(
                  tokens,
                  0,
                  tokens.size(),
                  {"G1FrameBranchCertificateScratch",
                   "ik_certificate"}).empty(),
          "dual certificate ownership exists only in transaction scratch, never persistent controller state");
}

static void test_production_no_seam_links_with_strict_recovery_provider()
{
    std::string error;
    const std::vector<CppToken> tokens = tokenize_cpp_source(
        read_source_file("controller.cpp"), error);
    check(error.empty(), error.empty() ? "controller tokenizes" : error.c_str());
    const std::vector<CppCallRecord> calls = cpp_calls_named(
        tokens, 0, tokens.size(), "g1_frame_transaction_run");
    check(calls.size() == 1U &&
              calls[0].key == "::g1_frame_transaction_run" &&
              cpp_find_token_sequence(
                  tokens,
                  calls[0].name,
                  calls[0].closing + 1,
                  {"::", "g1_controller_frame_stage_run", ",",
                   "::", "g1_recovery_candidates_build", ",",
                   "frame_external"}).size() == 1U,
          "the sole production coordinator call binds the real runner directly to the strict provider");
}

int main()
{
    test_production_root_reach_digest_ownership();
    test_fixed_nonassociative_root_reach_fk_ownership();
    test_real_runner_covers_all_six_modes_and_lowerings();
    test_scene_cycle_reaches_exact_dwell_boundary();
    test_search_inertialization_and_simulation_tuning_oracles();
    test_contact_tuning_fields_drive_real_history();
    test_real_runner_checkpoint_atomicity();
    test_each_other_mode_has_real_atomic_checkpoint();
    test_real_input_checkpoint_publication();
    test_production_contact_tuning_preflight();
    test_real_route_latch_consumption_and_resume();
    test_coordinator_rejects_structurally_valid_ik_forgery();
    test_synthetic_non_descending_step_safe_stops_atomically();
    test_quantized_planner_terminal_rejects_atomically();
    test_pose_certificate_discards_working_plan_and_prior_owners();
    test_real_abc_selects_b_in_both_modes();
    test_real_a_raw_passes_and_ik_reports_no_swing_candidate();
    test_real_b_contact_release_is_not_an_eligibility_gate();
    test_real_c_never_executes_after_b_accepts();
    test_real_incumbent_runs_last_with_clamped_plus_one();
    test_real_legacy_incumbent_is_not_retried();
    test_real_provider_global_error_rolls_back_every_owner();
    test_legacy_slot_zero_acceptance_has_exact_pre_feature_public_owners();
    test_legacy_slot_zero_acceptance_never_materializes_recovery();
    test_matching_disabled_and_unscheduled_frames_attempt_only_incumbent();
    test_end_of_animation_public_sentinels_do_not_leak_private_score();
    test_stage_runner_trusted_call_closure_is_exact();
    test_controller_contains_one_production_database_search_call();
    test_controller_has_no_candidate_contact_rank_gate();
    test_controller_has_no_persistent_dual_certificate_owner();
    test_production_no_seam_links_with_strict_recovery_provider();
    test_genuine_foot0_and_foot1_safe_stops();
    test_genuine_begin_time_safe_stop();
    test_real_pose_certificate_classification();
    test_controller_source_and_no_main_contract();
    return 0;
}
