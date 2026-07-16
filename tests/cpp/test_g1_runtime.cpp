#include "sonic/cpp/g1_runtime.h"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    const int frames = 4;
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

int main()
{
    test_direct_runtime_boundary_and_advance();
    return 0;
}
