#include "sonic/cpp/g1_runtime.h"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "G1 runtime test failed: %s\n", message);
        std::exit(1);
    }
}

static bool same_float_bits(float first, float second)
{
    return terrain_float_bits(first) == terrain_float_bits(second);
}

static bool same_double_bits(double first, double second)
{
    std::uint64_t first_bits = 0;
    std::uint64_t second_bits = 0;
    std::memcpy(&first_bits, &first, sizeof(first));
    std::memcpy(&second_bits, &second, sizeof(second));
    return first_bits == second_bits;
}

static bool same_vec3_bits(vec3 first, vec3 second)
{
    return same_float_bits(first.x, second.x) &&
           same_float_bits(first.y, second.y) &&
           same_float_bits(first.z, second.z);
}

static bool same_quat_bits(quat first, quat second)
{
    return same_float_bits(first.w, second.w) &&
           same_float_bits(first.x, second.x) &&
           same_float_bits(first.y, second.y) &&
           same_float_bits(first.z, second.z);
}

static void test_g1_rate_contract_is_exact_60_hz()
{
    const g1_runtime_config config;
    check(same_float_bits(config.dt, 1.0f / 60.0f),
          "G1 runtime default is exact binary32 60 Hz");
    check(g1_dt_is_exact_60_hz(1.0f / 60.0f),
          "exact binary32 60 Hz G1 dt is accepted");
    check(!g1_dt_is_exact_60_hz(1.0f / 25.0f),
          "stale exact binary32 25 Hz G1 dt is rejected");
    check(g1_manifest_rate_compatible(60.0f) &&
              !g1_manifest_rate_compatible(25.0f),
          "only the exact 60 Hz G1 manifest rate is compatible");
}

static void make_database(database& db)
{
    static const int parents[G1_BoneCount] = {
        -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
    };
    const int frames = 48;
    db.bone_positions.resize(frames, G1_BoneCount);
    db.bone_velocities.resize(frames, G1_BoneCount);
    db.bone_rotations.resize(frames, G1_BoneCount);
    db.bone_angular_velocities.resize(frames, G1_BoneCount);
    db.bone_parents.resize(G1_BoneCount);
    db.contact_states.resize(frames, 2);
    db.range_starts.resize(1);
    db.range_stops.resize(1);
    db.range_starts(0) = 0;
    db.range_stops(0) = frames;
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        db.bone_parents(bone) = parents[bone];
    }
    for (int frame = 0; frame < frames; ++frame) {
        for (int bone = 0; bone < G1_BoneCount; ++bone) {
            db.bone_positions(frame, bone) = bone == G1_Simulation
                ? vec3(0.0f, 0.0f, 0.02f * static_cast<float>(frame))
                : vec3(0.0f, bone == G1_Hips ? 1.0f : 0.01f, 0.0f);
            db.bone_velocities(frame, bone) = vec3();
            db.bone_rotations(frame, bone) = quat();
            db.bone_angular_velocities(frame, bone) = vec3();
        }
    }
    db.contact_states.zero();

    db.features.resize(frames, 31);
    db.features.zero();
    db.features_offset.resize(31);
    db.features_offset.zero();
    db.features_scale.resize(31);
    db.features_scale.set(1.0f);
    db.terrain_features.resize(frames, 4);
    db.terrain_features.zero();
    database_build_bounds(db);
}

static scene_pack make_scene()
{
    scene_pack scene;
    scene.metadata.id = "runtime-fixture";
    scene.metadata.spawn_position = vec3();
    scene.metadata.spawn_yaw = 0.0f;
    scene.metadata.playable_bounds = {-4.0f, -4.0f, 4.0f, 4.0f};
    scene.metadata.lookahead_bounds = scene.metadata.playable_bounds;
    scene.terrain.version = 2;
    scene.terrain.nx = 65;
    scene.terrain.nz = 65;
    scene.terrain.origin_x = -8.0f;
    scene.terrain.origin_z = -8.0f;
    scene.terrain.cell_size = 0.25f;
    scene.terrain.exterior_height = 0.0f;
    scene.terrain.heights.resize(
        scene.terrain.nx * scene.terrain.nz);
    scene.terrain.heights.set(2.0f);
    scene.walkability.nx = scene.terrain.nx;
    scene.walkability.nz = scene.terrain.nz;
    scene.walkability.cells.resize(
        scene.walkability.nx * scene.walkability.nz);
    scene.walkability.cells.set(1);
    return scene;
}

static void make_support(terrain_support_set& support, int frames)
{
    support.values.resize(frames, 3);
    support.values.zero();
}

static g1_runtime_step_request make_direct_request(bool matching_enabled)
{
    g1_runtime_step_request request;
    request.mode = G1RuntimeDirect;
    request.requested_velocity_holden = vec3();
    request.desired_heading_holden = quat();
    request.matching_enabled = matching_enabled;
    return request;
}

static void reset_runtime_state(
    g1_controller_state& state,
    database& db,
    const terrain_support_set& support,
    const scene_pack& scene)
{
    char error[512] = {};
    check(g1_controller_state_reset(
              state,
              db,
              support,
              scene,
              error,
              static_cast<int>(sizeof(error))),
          error);
}

static g1_runtime_frame_feasibility make_feasibility_view(
    const array1d<unsigned char>& raw_safe,
    const array1d<unsigned char>& search_safe,
    int count)
{
    g1_runtime_frame_feasibility feasibility;
    feasibility.raw_safe = raw_safe.data;
    feasibility.search_safe = search_safe.data;
    feasibility.count = count;
    return feasibility;
}

struct scripted_joint_preview
{
    int reject_selected = -1;
    bool reject_all = false;
    int fatal_selected = -1;
    bool malformed_rejection = false;
    int selected[64] = {};
    int emitted[64] = {};
    int count = 0;
    int accepted_selected = -1;
    quat accepted_rotations[G1_BoneCount] = {};
    vec3 accepted_angular_velocities[G1_BoneCount] = {};
};

static g1_runtime_joint_preview_verdict evaluate_scripted_joint_preview(
    void* raw,
    int selected_database_frame,
    int emitted_database_frame,
    slice1d<quat> local_rotations,
    slice1d<vec3> local_angular_velocities,
    int& rejected_joint_index,
    double& rejected_joint_position,
    char* error,
    int capacity)
{
    scripted_joint_preview& script =
        *static_cast<scripted_joint_preview*>(raw);
    check(script.count < 64, "preview script call capacity is sufficient");
    script.selected[script.count] = selected_database_frame;
    script.emitted[script.count] = emitted_database_frame;
    ++script.count;
    if (selected_database_frame == script.fatal_selected) {
        std::snprintf(
            error,
            static_cast<std::size_t>(capacity),
            "scripted joint preview fatal");
        return G1RuntimeJointPreviewFatal;
    }
    if (script.reject_all ||
        selected_database_frame == script.reject_selected) {
        rejected_joint_index = script.malformed_rejection ? -1 : 5;
        rejected_joint_position = script.malformed_rejection
            ? std::numeric_limits<double>::quiet_NaN()
            : -0.30;
        return G1RuntimeJointPreviewRejectLimit;
    }
    check(local_rotations.size == G1_BoneCount,
          "preview rotation shape is fixed");
    check(local_angular_velocities.size == G1_BoneCount,
          "preview angular velocity shape is fixed");
    script.accepted_selected = selected_database_frame;
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        script.accepted_rotations[bone] = local_rotations(bone);
        script.accepted_angular_velocities[bone] =
            local_angular_velocities(bone);
    }
    return G1RuntimeJointPreviewAccept;
}

static g1_runtime_joint_preview_validator make_scripted_preview_validator(
    scripted_joint_preview& script)
{
    g1_runtime_joint_preview_validator validator;
    validator.context = &script;
    validator.evaluate = evaluate_scripted_joint_preview;
    return validator;
}

static std::uint64_t hash_bytes(
    std::uint64_t hash, const void* memory, std::size_t size)
{
    const unsigned char* bytes =
        static_cast<const unsigned char*>(memory);
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= static_cast<std::uint64_t>(bytes[index]);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

template<typename T>
static std::uint64_t hash_array(
    std::uint64_t hash, const array1d<T>& values)
{
    hash = hash_bytes(hash, &values.size, sizeof(values.size));
    if (values.size > 0) {
        hash = hash_bytes(
            hash,
            values.data,
            static_cast<std::size_t>(values.size) * sizeof(T));
    }
    return hash;
}

static std::uint64_t state_array_hash(const g1_controller_state& state)
{
    std::uint64_t hash = UINT64_C(1469598103934665603);
#define HASH_ARRAY(name) hash = hash_array(hash, state.name)
    HASH_ARRAY(lmm_features);
    HASH_ARRAY(lmm_latent);
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
#undef HASH_ARRAY
    return hash;
}

struct state_guard
{
    unsigned char object[sizeof(g1_controller_state)];
    std::uint64_t arrays = 0;
};

static state_guard guard_state(const g1_controller_state& state)
{
    state_guard guard;
    std::memcpy(guard.object, &state, sizeof(state));
    guard.arrays = state_array_hash(state);
    return guard;
}

static bool state_is_unchanged(
    const g1_controller_state& state, const state_guard& guard)
{
    return std::memcmp(guard.object, &state, sizeof(state)) == 0 &&
           guard.arrays == state_array_hash(state);
}

static g1_runtime_step_result sentinel_result()
{
    g1_runtime_step_result result;
    result.query_database_frame = 91;
    result.selected_database_frame = 92;
    result.query_range = 93;
    for (int index = 0; index < 31; ++index) {
        result.query[index] = 100.0f + static_cast<float>(index);
    }
    for (int index = 0; index < 4; ++index) {
        result.terrain.values[index] = 200.0f + static_cast<float>(index);
        result.terrain.points[index] = vec3(
            300.0f + static_cast<float>(index),
            400.0f + static_cast<float>(index),
            500.0f + static_cast<float>(index));
    }
    result.traversal.blocked = true;
    result.traversal.walkability_class = 2;
    result.traversal.reason = walkability_blocked_cell;
    result.traversal.distance = 600.0f;
    result.traversal.commanded_speed = 601.0f;
    result.traversal.applied_speed = 602.0f;
    result.traversal.point = vec3(603.0f, 604.0f, 605.0f);
    result.raw_selected.hips_y = 700.0f;
    result.inertialized.hips_y = 701.0f;
    result.projected.hips_y = 702.0f;
    result.candidate_preview.candidate_preview_count = 81;
    result.candidate_preview.candidate_limit_rejection_count = 82;
    result.candidate_preview.first_rejected_database_frame = 83;
    result.candidate_preview.first_rejected_joint_index = 84;
    result.candidate_preview.first_rejected_joint_position = 85.0;
    return result;
}

static void check_failed_step_is_transactional(
    g1_controller_state& state,
    database& db,
    const terrain_support_set& support,
    const scene_pack& scene,
    const g1_runtime_step_request& request,
    const g1_runtime_config& config,
    const char* expected_error)
{
    g1_runtime_step_result result = sentinel_result();
    unsigned char result_before[sizeof(result)];
    std::memcpy(result_before, &result, sizeof(result));
    const state_guard state_before = guard_state(state);
    char error[512] = {};
    check(!g1_runtime_step(
              result,
              state,
              db,
              support,
              scene,
              request,
              config,
              error,
              static_cast<int>(sizeof(error))),
          "invalid runtime request is rejected");
    check(std::string(error).find(expected_error) != std::string::npos,
          "runtime failure publishes the expected checked diagnostic");
    check(state_is_unchanged(state, state_before),
          "failed runtime step preserves all state bytes and array values");
    check(std::memcmp(result_before, &result, sizeof(result)) == 0,
          "failed runtime step preserves the complete result object");
}

static void check_failed_feasible_step_is_transactional(
    g1_controller_state& state,
    database& db,
    const terrain_support_set& support,
    const scene_pack& scene,
    const g1_runtime_frame_feasibility& feasibility,
    const g1_runtime_step_request& request,
    const g1_runtime_config& config,
    const char* expected_error,
    bool exact_error)
{
    g1_runtime_step_result result = sentinel_result();
    unsigned char result_before[sizeof(result)];
    std::memcpy(result_before, &result, sizeof(result));
    const state_guard state_before = guard_state(state);
    char error[512] = {};
    check(!g1_runtime_step(
              result,
              state,
              db,
              support,
              scene,
              feasibility,
              request,
              config,
              error,
              static_cast<int>(sizeof(error))),
          "invalid feasible runtime step is rejected");
    if (exact_error) {
        check(std::string(error) == expected_error,
              "runtime failure publishes the exact checked diagnostic");
    } else {
        check(std::string(error).find(expected_error) != std::string::npos,
              "runtime failure publishes the expected checked diagnostic");
    }
    check(state_is_unchanged(state, state_before),
          "failed feasible runtime step preserves state transactionally");
    check(std::memcmp(result_before, &result, sizeof(result)) == 0,
          "failed feasible runtime step preserves the complete result");
}

static void check_failed_preview_step_is_transactional(
    g1_controller_state& state,
    database& db,
    const terrain_support_set& support,
    const scene_pack& scene,
    const g1_runtime_frame_feasibility& feasibility,
    const g1_runtime_step_request& request,
    const g1_runtime_config& config,
    scripted_joint_preview& script,
    const char* expected_error,
    bool exact_error)
{
    g1_runtime_step_result result = sentinel_result();
    unsigned char result_before[sizeof(result)];
    std::memcpy(result_before, &result, sizeof(result));
    const state_guard state_before = guard_state(state);
    const g1_runtime_joint_preview_validator validator =
        make_scripted_preview_validator(script);
    char error[512] = {};
    check(!g1_runtime_step(
              result,
              state,
              db,
              support,
              scene,
              feasibility,
              request,
              config,
              validator,
              error,
              static_cast<int>(sizeof(error))),
          "invalid preview runtime step is rejected");
    if (exact_error) {
        check(std::string(error) == expected_error,
              "preview failure publishes the exact diagnostic");
    } else {
        check(std::string(error).find(expected_error) != std::string::npos,
              "preview failure publishes the checked diagnostic");
    }
    check(state_is_unchanged(state, state_before),
          "failed preview step preserves state transactionally");
    check(std::memcmp(result_before, &result, sizeof(result)) == 0,
          "failed preview step preserves the complete result");
}

static void capture_legacy_query(
    float (&query)[31],
    database& db,
    const terrain_support_set& support,
    const scene_pack& scene)
{
    g1_controller_state state;
    reset_runtime_state(state, db, support, scene);
    const g1_runtime_step_request request = make_direct_request(false);
    const g1_runtime_config config;
    g1_runtime_step_result result;
    char error[512] = {};
    check(g1_runtime_step(
              result,
              state,
              db,
              support,
              scene,
              request,
              config,
              error,
              static_cast<int>(sizeof(error))),
          error);
    for (int dimension = 0; dimension < 31; ++dimension) {
        query[dimension] = result.query[dimension];
    }
}

static void make_masked_candidate_cost_order(
    database& db,
    const terrain_support_set& support,
    const scene_pack& scene)
{
    float query[31] = {};
    capture_legacy_query(query, db, support, scene);
    for (int frame = 0; frame < db.nframes(); ++frame) {
        for (int dimension = 0; dimension < 31; ++dimension) {
            db.features(frame, dimension) = query[dimension] + 50.0f;
        }
    }
    for (int dimension = 0; dimension < 31; ++dimension) {
        db.features(0, dimension) = 0.0f;
        db.features(20, dimension) = query[dimension];
        db.features(21, dimension) = query[dimension];
    }
    db.features(21, 0) += 1.0f;
    database_build_bounds(db);
}

static void make_forced_neighbor_cost_order(
    database& db,
    const terrain_support_set& support,
    const scene_pack& scene)
{
    float query[31] = {};
    capture_legacy_query(query, db, support, scene);
    for (int frame = 0; frame < db.nframes(); ++frame) {
        for (int dimension = 0; dimension < 31; ++dimension) {
            db.features(frame, dimension) = query[dimension] + 50.0f;
        }
    }
    for (int dimension = 0; dimension < 31; ++dimension) {
        db.features(2, dimension) = query[dimension];
        db.features(20, dimension) = query[dimension];
    }
    database_build_bounds(db);
}

static void test_direct_runtime_boundary_and_advance()
{
    database db;
    make_database(db);
    scene_pack scene = make_scene();
    terrain_support_set support;
    make_support(support, db.nframes());
    g1_controller_state state;
    char error[512] = {};
    check(g1_controller_state_reset(
              state,
              db,
              support,
              scene,
              error,
              static_cast<int>(sizeof(error))),
          error);

    g1_runtime_step_request request;
    request.mode = G1RuntimeDirect;
    request.requested_velocity_holden = vec3(0.125f, 0.0f, 0.25f);
    request.desired_heading_holden = quat();
    request.matching_enabled = false;
    g1_runtime_config config;

    g1_runtime_step_request invalid_heading = request;
    invalid_heading.desired_heading_holden = quat(2.0f, 0.0f, 0.0f, 0.0f);
    check_failed_step_is_transactional(
        state, db, support, scene, invalid_heading, config, "unit heading");

    g1_runtime_config invalid_config = config;
    invalid_config.dt = 0.0f;
    check_failed_step_is_transactional(
        state, db, support, scene, request, invalid_config, "dt");

    const int query_frame = state.frame_index;
    g1_runtime_step_result result;
    check(g1_runtime_step(
              result,
              state,
              db,
              support,
              scene,
              request,
              config,
              error,
              static_cast<int>(sizeof(error))),
          error);

    check(result.query_database_frame == query_frame &&
              result.selected_database_frame == query_frame &&
              result.query_range == 0,
          "result records the pre-advance query and selected frame boundary");
    check(state.frame_index == query_frame + 1,
          "one runtime call advances the ordinary database pose once");
    check(motion_match_query_is_finite_31d(
              slice1d<float>(31, result.query)),
          "runtime publishes exactly 31 finite query values");
    check(motion_match_query_is_finite_31d(
              slice1d<float>(31, result.query_normalized)),
          "runtime publishes exactly 31 finite normalized query values");
    for (int dimension = 0; dimension < 31; ++dimension) {
        const float independently_normalized = normalize_query_feature(
            result.query[dimension],
            db.features_offset(dimension),
            db.features_scale(dimension));
        check(std::fabs(
                  result.query_normalized[dimension] -
                  independently_normalized) <= 1e-6f,
              "runtime normalized query has 1e-6 parity with raw query");
    }
    check(same_vec3_bits(
              state.command.intent.requested_velocity,
              request.requested_velocity_holden) &&
              same_quat_bits(
                  state.command.intent.desired_heading,
                  request.desired_heading_holden),
          "direct command snapshot retains the latched raw intent");
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        check(same_vec3_bits(
                  state.command.predicted_desired_velocities[index],
                  request.requested_velocity_holden),
              "direct command latches one velocity across all four horizons");
        check(same_quat_bits(
                  state.command.predicted_desired_headings[index],
                  request.desired_heading_holden),
              "direct command latches one heading across all four horizons");
    }
    check(same_float_bits(
              state.global_bone_positions(G1_Hips).y,
              result.projected.hips_y),
          "result observes the final support-retargeted global pelvis");
    check(!same_float_bits(
              result.inertialized.hips_y,
              result.projected.hips_y),
          "fixture distinguishes inertialized from support-retargeted pelvis");
    check(g1_pose_diagnostic_is_finite(result.raw_selected) &&
              g1_pose_diagnostic_is_finite(result.inertialized) &&
              g1_pose_diagnostic_is_finite(result.projected),
              "all runtime pose boundaries are finite");
}

static void test_feasible_runtime_progression_and_masked_search()
{
    database db;
    make_database(db);
    scene_pack scene = make_scene();
    terrain_support_set support;
    make_support(support, db.nframes());
    array1d<unsigned char> raw_safe(db.nframes());
    array1d<unsigned char> search_safe(db.nframes());
    raw_safe.set(1);
    search_safe.set(1);
    const g1_runtime_frame_feasibility all_safe = make_feasibility_view(
        raw_safe, search_safe, db.nframes());
    const g1_runtime_step_request request = make_direct_request(true);
    const g1_runtime_config config;
    char error[512] = {};

    g1_controller_state state;
    reset_runtime_state(state, db, support, scene);
    state.search_timer = 100.0f;
    g1_runtime_step_result result;
    check(g1_runtime_step(
              result,
              state,
              db,
              support,
              scene,
              all_safe,
              request,
              config,
              error,
              static_cast<int>(sizeof(error))),
          error);
    check(!state.searched && result.selected_database_frame == 0,
          "a raw-safe successor keeps ordinary continuation");
    check(state.frame_index == 1,
          "safe ordinary continuation emits the exact successor");

    raw_safe.set(1);
    search_safe.zero();
    raw_safe(1) = 0;
    search_safe(20) = 1;
    const g1_runtime_frame_feasibility guarded = make_feasibility_view(
        raw_safe, search_safe, db.nframes());
    const g1_runtime_step_request matching_disabled_request =
        make_direct_request(false);
    reset_runtime_state(state, db, support, scene);
    state.search_timer = 100.0f;
    std::memset(error, 0, sizeof(error));
    check(g1_runtime_step(
              result,
              state,
              db,
              support,
              scene,
              guarded,
              matching_disabled_request,
              config,
              error,
              static_cast<int>(sizeof(error))),
          error);
    check(state.searched && state.transitioned,
          "an unsafe ordinary successor forces the existing search path");
    check(result.selected_database_frame == 20 && state.frame_index == 21,
          "forced search emits the certified target successor without skips");

    make_masked_candidate_cost_order(db, support, scene);
    raw_safe.set(1);
    search_safe.zero();
    raw_safe(1) = 0;
    search_safe(21) = 1;
    const g1_runtime_frame_feasibility next_best = make_feasibility_view(
        raw_safe, search_safe, db.nframes());
    reset_runtime_state(state, db, support, scene);
    state.search_timer = 100.0f;
    std::memset(error, 0, sizeof(error));
    check(g1_runtime_step(
              result,
              state,
              db,
              support,
              scene,
              next_best,
              request,
              config,
              error,
              static_cast<int>(sizeof(error))),
          error);
    check(result.selected_database_frame == 21 && state.frame_index == 22,
          "runtime skips the cheaper search-unsafe candidate");

    make_forced_neighbor_cost_order(db, support, scene);
    raw_safe.set(1);
    search_safe.zero();
    raw_safe(1) = 0;
    search_safe(2) = 1;
    search_safe(20) = 1;
    float forced_query_values[31] = {};
    capture_legacy_query(forced_query_values, db, support, scene);
    array1d<float> forced_query(31);
    for (int dimension = 0; dimension < 31; ++dimension) {
        forced_query(dimension) = forced_query_values[dimension];
    }
    int direct_index = -1;
    float direct_cost = FLT_MAX;
    database_search(
        direct_index,
        direct_cost,
        db,
        forced_query,
        0.0f,
        20,
        20,
        search_safe.data,
        search_safe.size);
    check(direct_index == 2,
          "fixture makes the nearby frame the uncentered search winner");
    const g1_runtime_frame_feasibility preserve_neighborhood =
        make_feasibility_view(raw_safe, search_safe, db.nframes());
    reset_runtime_state(state, db, support, scene);
    state.search_timer = 100.0f;
    std::memset(error, 0, sizeof(error));
    check(g1_runtime_step(
              result,
              state,
              db,
              support,
              scene,
              preserve_neighborhood,
              request,
              config,
              error,
              static_cast<int>(sizeof(error))),
          error);
    check(result.selected_database_frame == 20 && state.frame_index == 21,
          "forced masked search preserves the current-frame neighborhood");
}

static void test_joint_preview_selection_and_live_parity()
{
    database db;
    make_database(db);
    scene_pack scene = make_scene();
    terrain_support_set support;
    make_support(support, db.nframes());
    array1d<unsigned char> raw_safe(db.nframes());
    array1d<unsigned char> search_safe(db.nframes());
    raw_safe.set(1);
    search_safe.set(1);
    const g1_runtime_frame_feasibility feasibility = make_feasibility_view(
        raw_safe, search_safe, db.nframes());
    const g1_runtime_config config;
    char error[512] = {};

    g1_controller_state state;
    reset_runtime_state(state, db, support, scene);
    for (int bone = 2; bone < G1_BoneCount; ++bone) {
        const float scale = static_cast<float>(bone + 1);
        state.bone_offset_rotations(bone) = quat_from_angle_axis(
            0.001f * scale,
            vec3(1.0f, 0.0f, 0.0f));
        state.bone_offset_angular_velocities(bone) =
            vec3(0.0003f * scale, -0.0002f * scale, 0.0001f * scale);
        db.bone_rotations(20, bone) = quat_from_angle_axis(
            0.002f * scale,
            vec3(0.0f, 0.0f, 1.0f));
        db.bone_rotations(21, bone) = quat_from_angle_axis(
            -0.0015f * scale,
            vec3(0.0f, 1.0f, 0.0f));
        db.bone_angular_velocities(20, bone) =
            vec3(-0.003f * scale, 0.002f * scale, 0.001f * scale);
        db.bone_angular_velocities(21, bone) =
            vec3(0.0025f * scale, -0.001f * scale, 0.0015f * scale);
    }
    state.search_timer = 100.0f;
    scripted_joint_preview reject_ordinary;
    reject_ordinary.reject_selected = 0;
    const g1_runtime_joint_preview_validator reject_validator =
        make_scripted_preview_validator(reject_ordinary);
    g1_runtime_step_result result;
    check(g1_runtime_step(
              result,
              state,
              db,
              support,
              scene,
              feasibility,
              make_direct_request(false),
              config,
              reject_validator,
              error,
              static_cast<int>(sizeof(error))),
          error);
    check(state.searched && state.transitioned,
          "a rejected ordinary preview forces search when matching is disabled");
    check(result.selected_database_frame == 20 && state.frame_index == 21,
          "the next cost-ordered accepted preview becomes live");
    check(reject_ordinary.count == 2,
          "ordinary and accepted transition candidates are each previewed once");
    check(reject_ordinary.selected[0] == 0 &&
              reject_ordinary.emitted[0] == 1 &&
              reject_ordinary.selected[1] == 20 &&
              reject_ordinary.emitted[1] == 21,
          "preview receives selected and range-clamped emitted frames");
    check(result.candidate_preview.candidate_preview_count == 2 &&
              result.candidate_preview.candidate_limit_rejection_count == 1,
          "preview audit counts exact projection attempts and rejections");
    check(result.candidate_preview.first_rejected_database_frame == 0 &&
              result.candidate_preview.first_rejected_joint_index == 5 &&
              same_double_bits(
                  result.candidate_preview.first_rejected_joint_position,
                  -0.30),
          "preview audit preserves the first structured rejection");
    check(reject_ordinary.accepted_selected == 20,
          "accepted preview cache identifies the live transition");
    for (int bone = 2; bone < G1_BoneCount; ++bone) {
        check(same_quat_bits(
                  reject_ordinary.accepted_rotations[bone],
                  state.bone_rotations(bone)),
              "accepted preview rotation matches the live source joint bits");
        check(same_vec3_bits(
                  reject_ordinary.accepted_angular_velocities[bone],
                  state.bone_angular_velocities(bone)),
              "accepted preview angular velocity matches live source joint bits");
    }

    reset_runtime_state(state, db, support, scene);
    state.search_timer = 0.0f;
    scripted_joint_preview cached_ordinary;
    const g1_runtime_joint_preview_validator cached_validator =
        make_scripted_preview_validator(cached_ordinary);
    std::memset(error, 0, sizeof(error));
    check(g1_runtime_step(
              result,
              state,
              db,
              support,
              scene,
              feasibility,
              make_direct_request(true),
              config,
              cached_validator,
              error,
              static_cast<int>(sizeof(error))),
          error);
    check(state.searched && !state.transitioned,
          "a scheduled search retains its valid ordinary incumbent");
    check(result.selected_database_frame == 0 && state.frame_index == 1,
          "the cached ordinary preview advances exactly once");
    check(cached_ordinary.count == 1 &&
              result.candidate_preview.candidate_preview_count == 1 &&
              result.candidate_preview.candidate_limit_rejection_count == 0,
          "the prevalidated incumbent is not projected twice");
    check(result.candidate_preview.first_rejected_database_frame == -1 &&
              result.candidate_preview.first_rejected_joint_index == -1 &&
              same_double_bits(
                  result.candidate_preview.first_rejected_joint_position,
                  0.0),
          "zero rejection audit uses exact sentinels");
}

static void test_joint_preview_failures_are_transactional()
{
    database db;
    make_database(db);
    scene_pack scene = make_scene();
    terrain_support_set support;
    make_support(support, db.nframes());
    array1d<unsigned char> raw_safe(db.nframes());
    array1d<unsigned char> search_safe(db.nframes());
    raw_safe.set(1);
    search_safe.set(1);
    g1_runtime_frame_feasibility feasibility = make_feasibility_view(
        raw_safe, search_safe, db.nframes());
    const g1_runtime_step_request request = make_direct_request(false);
    const g1_runtime_config config;
    g1_controller_state state;

    reset_runtime_state(state, db, support, scene);
    scripted_joint_preview fatal;
    fatal.fatal_selected = 0;
    check_failed_preview_step_is_transactional(
        state,
        db,
        support,
        scene,
        feasibility,
        request,
        config,
        fatal,
        "scripted joint preview fatal",
        true);

    reset_runtime_state(state, db, support, scene);
    scripted_joint_preview malformed;
    malformed.reject_selected = 0;
    malformed.malformed_rejection = true;
    check_failed_preview_step_is_transactional(
        state,
        db,
        support,
        scene,
        feasibility,
        request,
        config,
        malformed,
        "preview rejection diagnostic is invalid",
        false);

    reset_runtime_state(state, db, support, scene);
    scripted_joint_preview all_rejected;
    all_rejected.reject_all = true;
    check_failed_preview_step_is_transactional(
        state,
        db,
        support,
        scene,
        feasibility,
        request,
        config,
        all_rejected,
        "no inertialized-joint-safe database candidate",
        true);
    check(all_rejected.count > 1,
          "dynamic exhaustion previews ordinary and search candidates");

    raw_safe.set(1);
    raw_safe(1) = 0;
    search_safe.zero();
    feasibility = make_feasibility_view(
        raw_safe, search_safe, db.nframes());
    reset_runtime_state(state, db, support, scene);
    scripted_joint_preview no_raw_candidate;
    check_failed_preview_step_is_transactional(
        state,
        db,
        support,
        scene,
        feasibility,
        request,
        config,
        no_raw_candidate,
        "no joint-limit-safe database candidate",
        true);
    check(no_raw_candidate.count == 0,
          "raw-safe exhaustion performs no dynamic projection");
}

static void test_feasible_runtime_failures_are_transactional()
{
    database db;
    make_database(db);
    scene_pack scene = make_scene();
    terrain_support_set support;
    make_support(support, db.nframes());
    array1d<unsigned char> raw_safe(db.nframes());
    array1d<unsigned char> search_safe(db.nframes());
    raw_safe.set(1);
    search_safe.zero();
    raw_safe(1) = 0;
    const g1_runtime_step_request request = make_direct_request(true);
    const g1_runtime_config config;
    g1_controller_state state;

    reset_runtime_state(state, db, support, scene);
    state.search_timer = 100.0f;
    const g1_runtime_frame_feasibility no_candidate = make_feasibility_view(
        raw_safe, search_safe, db.nframes());
    check_failed_feasible_step_is_transactional(
        state,
        db,
        support,
        scene,
        no_candidate,
        request,
        config,
        "no joint-limit-safe database candidate",
        true);

    raw_safe.set(1);
    search_safe.set(1);
    reset_runtime_state(state, db, support, scene);
    const g1_runtime_frame_feasibility wrong_count = make_feasibility_view(
        raw_safe, search_safe, db.nframes() - 1);
    check_failed_feasible_step_is_transactional(
        state,
        db,
        support,
        scene,
        wrong_count,
        request,
        config,
        "feasibility shape",
        false);

    reset_runtime_state(state, db, support, scene);
    g1_runtime_frame_feasibility missing_raw = make_feasibility_view(
        raw_safe, search_safe, db.nframes());
    missing_raw.raw_safe = nullptr;
    check_failed_feasible_step_is_transactional(
        state,
        db,
        support,
        scene,
        missing_raw,
        request,
        config,
        "feasibility shape",
        false);
}

static bool nearly(float value, float expected)
{
    return std::fabs(value - expected) <= 1.0e-4f;
}

static quat yaw_quat(float radians)
{
    return quat_from_angle_axis(radians, vec3(0.0f, 1.0f, 0.0f));
}

static float yaw_degrees(quat q)
{
    return 2.0f * std::atan2(q.y, q.w) * 180.0f / PIf;
}

// The turn profile caps the desired heading presented to prediction and
// application at 120 deg/s. At the 1/60 s step this is exactly 2 degrees;
// at the 1/3 s trajectory sample it is exactly 40 degrees per sample.
static void test_turn_profile_caps_current_and_future_heading()
{
    database db;
    make_database(db);
    scene_pack scene = make_scene();
    terrain_support_set support;
    make_support(support, db.nframes());
    g1_runtime_config config;
    char error[512] = {};

    g1_controller_state state;
    check(g1_controller_state_reset(
              state, db, support, scene, error,
              static_cast<int>(sizeof(error))),
          error);
    state.movement_model_profile = G1MovementHoldenTurnV1;
    state.desired_rotation = yaw_quat(0.0f);
    g1_runtime_step_request request;
    request.mode = G1RuntimeDirect;
    request.requested_velocity_holden = vec3();
    request.desired_heading_holden = yaw_quat(PIf);
    request.matching_enabled = false;
    g1_runtime_step_result result;
    check(g1_runtime_step(
              result, state, db, support, scene, request, config, error,
              static_cast<int>(sizeof(error))),
          error);
    check(nearly(yaw_degrees(state.desired_rotation), 2.0f),
          "current desired heading is capped to 2 degrees");
    check(same_quat_bits(
              state.command.intent.desired_heading, state.desired_rotation),
          "command evidence uses capped current heading");
    check(nearly(yaw_degrees(state.command.predicted_desired_headings[0]),
                 42.0f),
          "first future heading advances another 40 degrees");
    check(nearly(yaw_degrees(state.command.predicted_desired_headings[3]),
                 162.0f),
          "future rollout advances one shared capped path");
}

static void test_turn_profile_reversal_takes_at_least_1p5_seconds()
{
    database db;
    make_database(db);
    scene_pack scene = make_scene();
    terrain_support_set support;
    make_support(support, db.nframes());
    g1_runtime_config config;
    char error[512] = {};

    g1_controller_state state;
    check(g1_controller_state_reset(
              state, db, support, scene, error,
              static_cast<int>(sizeof(error))),
          error);
    state.movement_model_profile = G1MovementHoldenTurnV1;
    state.desired_rotation = yaw_quat(0.0f);
    g1_runtime_step_request request;
    request.mode = G1RuntimeDirect;
    request.requested_velocity_holden = vec3();
    request.desired_heading_holden = yaw_quat(PIf);
    request.matching_enabled = false;

    int steps = 0;
    float previous = 0.0f;
    while (yaw_degrees(state.desired_rotation) < 179.999f && steps < 120) {
        g1_runtime_step_result result;
        check(g1_runtime_step(
                  result, state, db, support, scene, request, config, error,
                  static_cast<int>(sizeof(error))),
              error);
        const float current = yaw_degrees(state.desired_rotation);
        check(current - previous <= 2.0001f,
              "each committed heading delta is at most the 2-degree cap");
        check(current + 1.0e-4f >= previous,
              "committed heading advances monotonically toward the target");
        previous = current;
        ++steps;
    }
    check(steps >= 90, "a 180-degree reversal needs at least 90 60-Hz steps");
}

static void test_raw_and_holden_v1_headings_are_exact()
{
    database db;
    make_database(db);
    scene_pack scene = make_scene();
    terrain_support_set support;
    make_support(support, db.nframes());
    g1_runtime_config config;
    char error[512] = {};

    const quat requested = yaw_quat(PIf);
    const g1_movement_model_profile profiles[] = {
        G1MovementRaw, G1MovementHoldenV1
    };
    for (g1_movement_model_profile profile : profiles) {
        g1_controller_state state;
        check(g1_controller_state_reset(
                  state, db, support, scene, error,
                  static_cast<int>(sizeof(error))),
              error);
        state.movement_model_profile = profile;
        state.desired_rotation = yaw_quat(0.0f);
        g1_runtime_step_request request;
        request.mode = G1RuntimeDirect;
        request.requested_velocity_holden = vec3();
        request.desired_heading_holden = requested;
        request.matching_enabled = false;
        g1_runtime_step_result result;
        check(g1_runtime_step(
                  result, state, db, support, scene, request, config, error,
                  static_cast<int>(sizeof(error))),
              error);
        check(same_quat_bits(state.desired_rotation, requested),
              "raw/holden-v1 commit the exact requested heading");
        check(same_quat_bits(
                  state.command.intent.desired_heading, requested),
              "raw/holden-v1 command evidence is the exact requested heading");
        for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
            check(same_quat_bits(
                      state.command.predicted_desired_headings[index],
                      requested),
                  "raw/holden-v1 predicted headings equal the request exactly");
        }
    }
}

static void test_turn_profile_prediction_failure_is_transactional()
{
    database db;
    make_database(db);
    scene_pack scene = make_scene();
    terrain_support_set support;
    make_support(support, db.nframes());
    g1_runtime_config config;
    char error[512] = {};

    g1_controller_state state;
    check(g1_controller_state_reset(
              state, db, support, scene, error,
              static_cast<int>(sizeof(error))),
          error);
    state.movement_model_profile = G1MovementHoldenTurnV1;
    state.desired_rotation = yaw_quat(0.3f);
    // A non-finite trajectory sample time makes the heading prediction fail;
    // the committed desired rotation must be preserved bit-for-bit.
    g1_runtime_config invalid_config = config;
    invalid_config.trajectory_sample_time =
        std::numeric_limits<float>::quiet_NaN();
    g1_runtime_step_request request;
    request.mode = G1RuntimeDirect;
    request.requested_velocity_holden = vec3();
    request.desired_heading_holden = yaw_quat(PIf);
    request.matching_enabled = false;
    const quat rotation_before = state.desired_rotation;
    g1_runtime_step_result result;
    check(!g1_runtime_step(
              result, state, db, support, scene, request, invalid_config,
              error, static_cast<int>(sizeof(error))),
          "non-finite prediction sample time fails the turn-profile step");
    check(same_quat_bits(state.desired_rotation, rotation_before),
          "failed turn-profile prediction preserves committed heading");
}

static void test_holden_movement_model_shapes_current_and_future()
{
    database db;
    make_database(db);
    scene_pack scene = make_scene();
    terrain_support_set support;
    make_support(support, db.nframes());
    g1_runtime_config config;
    char error[512] = {};

    // Accelerating from rest: the first applied velocity respects the
    // 1.5 m/s^2 acceleration bound (0.025 m/s per 1/60 s step) and the
    // predicted samples ramp further toward the limited target.
    g1_controller_state accel_state;
    check(g1_controller_state_reset(
              accel_state, db, support, scene, error,
              static_cast<int>(sizeof(error))),
          error);
    accel_state.movement_model_profile = G1MovementHoldenV1;
    g1_runtime_step_request accel_request;
    accel_request.mode = G1RuntimeDirect;
    accel_request.requested_velocity_holden = vec3(0.9f, 0.0f, 0.0f);
    accel_request.desired_heading_holden = quat();
    accel_request.matching_enabled = false;
    g1_runtime_step_result accel_result;
    check(g1_runtime_step(
              accel_result, accel_state, db, support, scene,
              accel_request, config, error,
              static_cast<int>(sizeof(error))),
          error);
    check(nearly(accel_state.command.applied_velocity.x, 0.025f),
          "holden-v1 shapes the current applied velocity by acceleration");
    check(nearly(accel_state.movement_velocity.x, 0.025f),
          "holden-v1 advances persistent movement velocity once per step");
    check(accel_state.trajectory_desired_velocities(0).x > 0.025f &&
              accel_state.trajectory_desired_velocities(0).x <= 0.560001f,
          "holden-v1 predicts a forward-ramping future sample from a copy");

    // Reversal: request the opposite direction from an established
    // positive velocity; the first shaped applied value must remain
    // positive and brake by at most the 1/30 m/s deceleration step.
    g1_controller_state brake_state;
    check(g1_controller_state_reset(
              brake_state, db, support, scene, error,
              static_cast<int>(sizeof(error))),
          error);
    brake_state.movement_model_profile = G1MovementHoldenV1;
    brake_state.movement_velocity = vec3(0.9f, 0.0f, 0.0f);
    g1_runtime_step_request brake_request = accel_request;
    brake_request.requested_velocity_holden = vec3(-0.9f, 0.0f, 0.0f);
    g1_runtime_step_result brake_result;
    check(g1_runtime_step(
              brake_result, brake_state, db, support, scene,
              brake_request, config, error,
              static_cast<int>(sizeof(error))),
          error);
    check(brake_state.command.applied_velocity.x > 0.0f &&
              brake_state.command.applied_velocity.x >=
                  0.9f - (2.0f / 60.0f + 1.0e-6f),
          "holden-v1 brakes an abrupt reversal through a bounded decel step");
}

static void test_holden_prediction_failure_is_transactional()
{
    database db;
    make_database(db);
    scene_pack scene = make_scene();
    terrain_support_set support;
    make_support(support, db.nframes());
    g1_runtime_config config;
    char error[512] = {};

    g1_controller_state state;
    check(g1_controller_state_reset(
              state, db, support, scene, error,
              static_cast<int>(sizeof(error))),
          error);
    state.movement_model_profile = G1MovementHoldenV1;
    state.movement_velocity = vec3(0.3f, 0.0f, -0.2f);
    // A non-finite trajectory sample time makes the movement-model
    // prediction fail; the active state's profile and movement velocity
    // must be preserved bit-for-bit.
    g1_runtime_config invalid_config = config;
    invalid_config.trajectory_sample_time =
        std::numeric_limits<float>::quiet_NaN();
    g1_runtime_step_request request;
    request.mode = G1RuntimeDirect;
    request.requested_velocity_holden = vec3(0.5f, 0.0f, 0.0f);
    request.desired_heading_holden = quat();
    request.matching_enabled = false;
    const g1_movement_model_profile profile_before =
        state.movement_model_profile;
    const vec3 velocity_before = state.movement_velocity;
    g1_runtime_step_result result;
    check(!g1_runtime_step(
              result, state, db, support, scene, request, invalid_config,
              error, static_cast<int>(sizeof(error))),
          "non-finite prediction sample time fails the holden-v1 step");
    check(state.movement_model_profile == profile_before &&
              same_vec3_bits(state.movement_velocity, velocity_before),
          "failed holden-v1 prediction preserves active movement state");
}

static void test_holden_blocked_clears_planar_movement_velocity()
{
    database db;
    make_database(db);
    scene_pack scene = make_scene();
    // Block every cell so the traversability limiter forces a full stop.
    scene.walkability.cells.set(0);
    terrain_support_set support;
    make_support(support, db.nframes());
    g1_runtime_config config;
    char error[512] = {};

    g1_controller_state state;
    check(g1_controller_state_reset(
              state, db, support, scene, error,
              static_cast<int>(sizeof(error))),
          error);
    state.movement_model_profile = G1MovementHoldenV1;
    state.movement_velocity = vec3(0.6f, 0.0f, 0.4f);
    g1_runtime_step_request request;
    request.mode = G1RuntimeDirect;
    request.requested_velocity_holden = vec3(0.9f, 0.0f, 0.0f);
    request.desired_heading_holden = quat();
    request.matching_enabled = false;
    g1_runtime_step_result result;
    check(g1_runtime_step(
              result, state, db, support, scene, request, config, error,
              static_cast<int>(sizeof(error))),
          error);
    check(result.traversal.blocked,
          "fixture blocks the commanded traversal");
    check(state.movement_velocity.x == 0.0f &&
              state.movement_velocity.z == 0.0f,
          "a blocked holden-v1 step clears planar movement velocity");
}

static void test_holden_partial_block_preserves_limited_motion()
{
    database db;
    make_database(db);
    scene_pack scene = make_scene();
    scene.metadata.spawn_position = vec3(0.3f, 0.0f, 0.2f);
    scene.metadata.playable_bounds = {0.0f, 0.0f, 1.0f, 0.4f};
    scene.metadata.lookahead_bounds = scene.metadata.playable_bounds;
    scene.terrain.nx = 11;
    scene.terrain.nz = 5;
    scene.terrain.origin_x = 0.0f;
    scene.terrain.origin_z = 0.0f;
    scene.terrain.cell_size = 0.10f;
    scene.terrain.heights.resize(55);
    scene.terrain.heights.zero();
    scene.walkability.nx = 11;
    scene.walkability.nz = 5;
    scene.walkability.cells.resize(55);
    scene.walkability.cells.set(1);
    for (int z = 0; z < scene.walkability.nz; ++z) {
        scene.walkability.cells(z * scene.walkability.nx + 7) = 0;
    }
    scene.walkability.cells(2 * scene.walkability.nx + 4) = 2;
    terrain_support_set support;
    make_support(support, db.nframes());
    g1_runtime_config config;
    char error[512] = {};

    g1_controller_state state;
    check(g1_controller_state_reset(
              state, db, support, scene, error,
              static_cast<int>(sizeof(error))),
          error);
    state.movement_model_profile = G1MovementHoldenV1;
    state.movement_velocity = vec3(0.2f, 0.0f, 0.0f);
    g1_runtime_step_request request;
    request.mode = G1RuntimeDirect;
    request.requested_velocity_holden = vec3(0.5f, 0.0f, 0.0f);
    request.desired_heading_holden = quat();
    request.matching_enabled = false;
    g1_runtime_step_result result;
    check(g1_runtime_step(
              result, state, db, support, scene, request, config, error,
              static_cast<int>(sizeof(error))),
          error);
    check(result.traversal.blocked &&
              result.traversal.applied_speed > 1.0e-4f,
          "fixture finds a block while retaining positive limited speed");
    check(state.movement_velocity.x > 0.0f &&
              state.command.applied_velocity.x > 0.0f,
          "partial lookahead blocking preserves shaped limited motion");
    check(state.trajectory_desired_velocities(0).x > 0.0f,
          "partial lookahead blocking predicts toward the limited target");
}

static void test_raw_profile_preserves_generated_arrays()
{
    database db;
    make_database(db);
    scene_pack scene = make_scene();
    terrain_support_set support;
    make_support(support, db.nframes());
    g1_runtime_config config;
    char error[512] = {};

    g1_runtime_step_request request;
    request.mode = G1RuntimeDirect;
    request.requested_velocity_holden = vec3(0.35f, 0.0f, -0.2f);
    request.desired_heading_holden = quat();
    request.matching_enabled = false;

    // The default raw profile must reproduce the exact command and
    // trajectory arrays regardless of any stale movement-model velocity.
    g1_controller_state baseline;
    check(g1_controller_state_reset(
              baseline, db, support, scene, error,
              static_cast<int>(sizeof(error))),
          error);
    g1_runtime_step_result baseline_result;
    check(g1_runtime_step(
              baseline_result, baseline, db, support, scene, request, config,
              error, static_cast<int>(sizeof(error))),
          error);

    g1_controller_state with_stale;
    check(g1_controller_state_reset(
              with_stale, db, support, scene, error,
              static_cast<int>(sizeof(error))),
          error);
    with_stale.movement_model_profile = G1MovementRaw;
    with_stale.movement_velocity = vec3(5.0f, 0.0f, -7.0f);
    g1_runtime_step_result stale_result;
    check(g1_runtime_step(
              stale_result, with_stale, db, support, scene, request, config,
              error, static_cast<int>(sizeof(error))),
          error);

    check(same_vec3_bits(baseline.command.applied_velocity,
                         with_stale.command.applied_velocity),
          "raw applied velocity ignores movement-model state");
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        check(same_vec3_bits(
                  baseline.trajectory_desired_velocities(index),
                  with_stale.trajectory_desired_velocities(index)),
              "raw trajectory desired velocities are byte-identical");
        check(same_vec3_bits(
                  baseline.command.predicted_desired_velocities[index],
                  with_stale.command.predicted_desired_velocities[index]),
              "raw predicted desired velocities are byte-identical");
    }
}

static void test_unsafe_flat_bundle_rejected(const char* root)
{
    char error[512] = {};
    motion_pack_manifest manifest;
    check(!motion_manifest_load_and_verify(
              manifest, root, error, static_cast<int>(sizeof(error))),
          "noncanonical v1/v2 flat bundle is rejected before controller state");
}

static void test_flat_bundle_adapter(const char* root)
{
    char error[512] = {};
    motion_pack_manifest manifest;
    check(motion_manifest_load_and_verify(
              manifest, root, error, static_cast<int>(sizeof(error))),
          error);
    check(manifest.flat_lmm_bundle &&
              g1_manifest_rate_compatible(manifest.output_fps) &&
              manifest.database_frames == 256 &&
              manifest.sources.size() == 1 &&
              manifest.sources[0].range_start == 0 &&
              manifest.sources[0].range_stop == 256 &&
              manifest.sources[0].name ==
                  "LocomotionFlat01_000-walk-only-7659-8171-120hz" &&
              manifest.sources[0].terrain_id == "flat",
          "flat adapter preserves the canonical v3 walk-only contract");
    std::string database_path;
    std::string feature_path;
    check(scene_join(
              database_path, root, manifest.database.path,
              error, static_cast<int>(sizeof(error))) &&
              scene_join(
                  feature_path, root, manifest.matching_features.path,
                  error, static_cast<int>(sizeof(error))),
          error);
    database db;
    database_load(db, database_path.c_str());
    check(database_load_matching_features_checked(
              db, feature_path.c_str(), error,
              static_cast<int>(sizeof(error))),
          error);
    db.terrain_features.resize(db.nframes(), 4);
    db.terrain_features.zero();
    check(motion_manifest_validate_database(
              manifest, db, error, static_cast<int>(sizeof(error))),
          error);
    scene_catalog catalog;
    scene_pack scene;
    check(flat_scene_catalog_build(
              catalog, manifest, error, static_cast<int>(sizeof(error))) &&
              flat_scene_pack_build(
                  scene, manifest, error, static_cast<int>(sizeof(error))),
          error);
    check(catalog.ids.size() == 1 &&
              terrain_heightfield_is_queryable(scene.terrain) &&
              heightfield_sample_v2(scene.terrain, 0.0f, 0.0f) == 0.0f,
          "flat adapter creates only the authenticated zero-height context");

    terrain_support_set support;
    support.values.resize(db.nframes(), 3);
    support.values.zero();
    g1_controller_state state;
    check(g1_controller_state_reset(
              state, db, support, scene, error,
              static_cast<int>(sizeof(error))),
          error);
    g1_runtime_step_request request = make_direct_request(true);
    request.matching_enabled = false;
    request.requested_velocity_holden = vec3(0.0f, 0.0f, 0.4f);
    const g1_runtime_config config;
    float maximum_joint_step = 0.0f;
    int maximum_joint_frame = -1;
    int maximum_joint_bone = -1;
    check(db.nranges() == 1 && db.range_starts(0) == 0 &&
              db.range_stops(0) == 256 && state.frame_index == 0,
          "ordinary source replay starts at the sole safe v3 range");
    const int safe_advance_count = db.range_stops(0) - db.range_starts(0) - 1;
    check(safe_advance_count == 255,
          "ordinary source replay has exactly 255 in-range advances");
    for (int frame = 0; frame < safe_advance_count; ++frame) {
        array1d<quat> previous = state.bone_rotations;
        const int previous_database_frame = state.frame_index;
        g1_runtime_step_result result;
        check(g1_runtime_step(
                  result, state, db, support, scene, request, config,
                  error, static_cast<int>(sizeof(error))),
              error);
        check(result.query_database_frame == previous_database_frame &&
                  result.selected_database_frame == previous_database_frame &&
                  state.frame_index == previous_database_frame + 1 &&
                  !state.searched && !state.transitioned,
              "ordinary source replay advances once without search or a seam");
        check(g1_pose_diagnostic_is_finite(result.projected),
              "finite in-range flat replay keeps a finite pose");
        for (int bone = 0; bone < G1_BoneCount; ++bone) {
            const float joint_step = quat_angle_between(
                previous(bone), state.bone_rotations(bone));
            if (joint_step > maximum_joint_step) {
                maximum_joint_step = joint_step;
                maximum_joint_frame = frame;
                maximum_joint_bone = bone;
            }
        }
        for (int dimension = 0; dimension < 31; ++dimension) {
            const float independently_normalized = normalize_query_feature(
                result.query[dimension],
                db.features_offset(dimension),
                db.features_scale(dimension));
            check(std::fabs(
                      independently_normalized -
                      result.query_normalized[dimension]) <= 1e-6f,
                  "finite in-range flat replay preserves feature parity");
        }
        ++state.scene_frame;
    }
    check(state.frame_index == db.range_stops(0) - 1,
          "ordinary source replay stops at the last safe row without looping");
    if (maximum_joint_step > 0.25f) {
        std::fprintf(
            stderr,
            "flat replay max joint step=%.9g rad frame=%d bone=%d\n",
            maximum_joint_step, maximum_joint_frame, maximum_joint_bone);
    }
    check(maximum_joint_step <= 0.25f,
          "finite in-range ordinary replay joint step stays within 0.25 rad");

    namespace fs = std::filesystem;
    const fs::path tampered = "/tmp/test_g1_runtime_size_tamper";
    fs::remove_all(tampered);
    fs::copy(root, tampered, fs::copy_options::recursive);
    const fs::path tampered_manifest = tampered / "manifest.json";
    std::ifstream manifest_input(tampered_manifest, std::ios::binary);
    check(manifest_input.good(), "open flat manifest for size-only tamper");
    std::string manifest_text{
        std::istreambuf_iterator<char>(manifest_input),
        std::istreambuf_iterator<char>()};
    const std::uintmax_t database_size = fs::file_size(tampered / "database.bin");
    const std::string size_receipt =
        "\"size_bytes\": " + std::to_string(database_size);
    const size_t size_position = manifest_text.find(size_receipt);
    check(size_position != std::string::npos,
          "locate database size receipt for size-only tamper");
    manifest_text.replace(
        size_position,
        size_receipt.size(),
        "\"size_bytes\": " + std::to_string(database_size + 1));
    std::ofstream manifest_output(
        tampered_manifest, std::ios::binary | std::ios::trunc);
    check(manifest_output.good(), "open flat manifest size-only tamper output");
    manifest_output.write(
        manifest_text.data(), static_cast<std::streamsize>(manifest_text.size()));
    check(manifest_output.good(), "write flat manifest size-only tamper");
    manifest_output.close();

    motion_pack_manifest rejected = manifest;
    error[0] = '\0';
    check(!motion_manifest_load_and_verify(
              rejected, tampered.c_str(), error,
              static_cast<int>(sizeof(error))),
          "flat manifest size-only tamper is rejected");
    check(std::strstr(error, "size") != nullptr,
          "flat manifest size-only tamper reports the size mismatch");
    fs::remove_all(tampered);

    const fs::path noncanonical = "/tmp/test_g1_runtime_manifest_tamper";
    fs::remove_all(noncanonical);
    fs::copy(root, noncanonical, fs::copy_options::recursive);
    const fs::path noncanonical_manifest = noncanonical / "manifest.json";
    std::ifstream noncanonical_input(noncanonical_manifest, std::ios::binary);
    check(noncanonical_input.good(), "open flat manifest for canonical tamper");
    std::string noncanonical_text{
        std::istreambuf_iterator<char>(noncanonical_input),
        std::istreambuf_iterator<char>()};
    const size_t opening_brace = noncanonical_text.find('{');
    check(opening_brace != std::string::npos,
          "locate flat manifest opening brace for canonical tamper");
    noncanonical_text.insert(opening_brace + 1, " ");
    std::ofstream noncanonical_output(
        noncanonical_manifest, std::ios::binary | std::ios::trunc);
    check(noncanonical_output.good(), "open canonical tamper output");
    noncanonical_output.write(
        noncanonical_text.data(),
        static_cast<std::streamsize>(noncanonical_text.size()));
    check(noncanonical_output.good(), "write canonical manifest tamper");
    noncanonical_output.close();

    rejected = manifest;
    error[0] = '\0';
    check(!motion_manifest_load_and_verify(
              rejected, noncanonical.c_str(), error,
              static_cast<int>(sizeof(error))),
          "semantically identical noncanonical v3 manifest is rejected");
    check(std::strstr(error, "canonical") != nullptr,
          "noncanonical v3 manifest reports the identity mismatch");
    fs::remove_all(noncanonical);
}

int main(int argc, char** argv)
{
    test_g1_rate_contract_is_exact_60_hz();
    test_direct_runtime_boundary_and_advance();
    test_holden_movement_model_shapes_current_and_future();
    test_holden_prediction_failure_is_transactional();
    test_holden_blocked_clears_planar_movement_velocity();
    test_holden_partial_block_preserves_limited_motion();
    test_turn_profile_caps_current_and_future_heading();
    test_turn_profile_reversal_takes_at_least_1p5_seconds();
    test_raw_and_holden_v1_headings_are_exact();
    test_turn_profile_prediction_failure_is_transactional();
    test_raw_profile_preserves_generated_arrays();
    test_feasible_runtime_progression_and_masked_search();
    test_feasible_runtime_failures_are_transactional();
    test_joint_preview_selection_and_live_parity();
    test_joint_preview_failures_are_transactional();
    if (argc == 3 && std::strcmp(argv[1], "--flat-bundle") == 0) {
        test_flat_bundle_adapter(argv[2]);
    } else if (argc == 3 &&
               std::strcmp(argv[1], "--unsafe-flat-bundle") == 0) {
        test_unsafe_flat_bundle_rejected(argv[2]);
    } else {
        check(argc == 1,
              "usage: test_g1_runtime [--flat-bundle DIRECTORY | "
              "--unsafe-flat-bundle DIRECTORY]");
    }
    return 0;
}
