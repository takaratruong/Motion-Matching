#include "sonic/cpp/g1_runtime.h"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    // 1.5 m/s^2 acceleration bound (0.06 m/s per 0.04 s step) and the
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
    check(nearly(accel_state.command.applied_velocity.x, 0.06f),
          "holden-v1 shapes the current applied velocity by acceleration");
    check(nearly(accel_state.movement_velocity.x, 0.06f),
          "holden-v1 advances persistent movement velocity once per step");
    check(accel_state.trajectory_desired_velocities(0).x > 0.06f &&
              accel_state.trajectory_desired_velocities(0).x <= 0.560001f,
          "holden-v1 predicts a forward-ramping future sample from a copy");

    // Reversal: request the opposite direction from an established
    // positive velocity; the first shaped applied value must remain
    // positive and brake by at most the 0.08 m/s deceleration step.
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
              brake_state.command.applied_velocity.x >= 0.9f - 0.080001f,
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

int main()
{
    test_direct_runtime_boundary_and_advance();
    test_holden_movement_model_shapes_current_and_future();
    test_holden_prediction_failure_is_transactional();
    test_holden_blocked_clears_planar_movement_velocity();
    test_holden_partial_block_preserves_limited_motion();
    test_raw_profile_preserves_generated_arrays();
    test_feasible_runtime_progression_and_masked_search();
    test_feasible_runtime_failures_are_transactional();
    test_joint_preview_selection_and_live_parity();
    test_joint_preview_failures_are_transactional();
    return 0;
}
