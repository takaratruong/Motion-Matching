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
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = hash_bytes(hash, &state, sizeof(state));
    g1_controller_state_memory_range ranges[64] = {};
    int count = 0;
    check(g1_controller_state_storage_ranges(
              state, ranges, count, 64) && count == 49,
          "state digest covers every dynamic owner");
    for (int index = 0; index < count; ++index) {
        hash = hash_bytes(hash, ranges[index].data, ranges[index].bytes);
    }
    return hash;
}

struct state_storage_snapshot
{
    std::array<const void*, 49> data = {};
    std::array<std::size_t, 49> bytes = {};
};

static state_storage_snapshot state_storage(
    const g1_controller_state& state)
{
    g1_controller_state_memory_range ranges[64] = {};
    int count = 0;
    check(g1_controller_state_storage_ranges(
              state, ranges, count, 64) && count == 49,
          "state storage snapshot covers every dynamic owner");
    state_storage_snapshot output;
    for (int index = 0; index < count; ++index) {
        output.data[static_cast<std::size_t>(index)] = ranges[index].data;
        output.bytes[static_cast<std::size_t>(index)] = ranges[index].bytes;
    }
    return output;
}

static bool same_storage(
    const state_storage_snapshot& first,
    const state_storage_snapshot& second)
{
    return first.data == second.data && first.bytes == second.bytes;
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
    uint64_t snapshot_accepted_digest = 0;
    uint64_t snapshot_working_digest = 0;
    uint64_t snapshot_publication_digest = 0;
    uint64_t snapshot_accepted_diagnostic_digest = 0;
    state_storage_snapshot snapshot_accepted_storage;
    state_storage_snapshot snapshot_working_storage;

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
        runtime.accepted_state.scene_frame = 17;
        runtime.accepted_state.route_frames = 5;
        runtime.working_state.scene_frame = 23;
        runtime.working_state.route_frames = 7;
        runtime.publication.ik_safe_stop_latched = true;
        runtime.publication.presentation_frame = 71;
        runtime.accepted_diagnostic.ready = true;
        runtime.accepted_diagnostic.presentation_frame = 53;
        runtime.accepted_diagnostic.scene_frame = 17;
        models.live = 1;
        capture_active();
    }

    void capture_active()
    {
        snapshot_index = active_index;
        snapshot_scene_id = active_scene.metadata.id;
        snapshot_mesh_path = active_scene.mesh_path;
        snapshot_heights = active_scene.terrain.heights.data;
        snapshot_model_id = active_model.id;
        snapshot_accepted_digest = state_digest(runtime.accepted_state);
        snapshot_working_digest = state_digest(runtime.working_state);
        snapshot_publication_digest = publication_digest(runtime.publication);
        snapshot_accepted_diagnostic_digest =
            accepted_diagnostic_digest(runtime.accepted_diagnostic);
        snapshot_accepted_storage = state_storage(runtime.accepted_state);
        snapshot_working_storage = state_storage(runtime.working_state);
    }

    bool active_matches_snapshot() const
    {
        return active_index == snapshot_index &&
               active_scene.metadata.id == snapshot_scene_id &&
               active_scene.mesh_path == snapshot_mesh_path &&
               active_scene.terrain.heights.data == snapshot_heights &&
               active_model.id == snapshot_model_id &&
               state_digest(runtime.accepted_state) ==
                   snapshot_accepted_digest &&
               state_digest(runtime.working_state) ==
                   snapshot_working_digest &&
               publication_digest(runtime.publication) ==
                   snapshot_publication_digest &&
               accepted_diagnostic_digest(runtime.accepted_diagnostic) ==
                   snapshot_accepted_diagnostic_digest &&
               same_storage(
                   state_storage(runtime.accepted_state),
                   snapshot_accepted_storage) &&
               same_storage(
                   state_storage(runtime.working_state),
                   snapshot_working_storage);
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
                check(active_index == target &&
                          active_scene.metadata.id ==
                              catalog.ids[static_cast<size_t>(target)] &&
                          runtime.accepted_state.simulation_position.x ==
                              2.0f &&
                          runtime.working_state.simulation_position.x ==
                              2.0f &&
                          active_model.id == models.last_loaded_id,
                      "scene, state pair, model, and index commit before old unload");
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
    g1_controller_state_memory_range accepted_ranges[64] = {};
    g1_controller_state_memory_range working_ranges[64] = {};
    int accepted_count = 0;
    int working_count = 0;
    check(g1_frame_controller_states_equal(
              fixture.runtime.accepted_state,
              fixture.runtime.working_state) &&
              g1_frame_state_pair_storage_is_exact(
                  fixture.runtime.accepted_state,
                  fixture.runtime.working_state,
                  accepted_ranges,
                  accepted_count,
                  working_ranges,
                  working_count,
                  64) &&
              accepted_count == 49 && working_count == 49,
          "successful reset/switch installs equal, completely disjoint states");
    check(fixture.runtime.accepted_state.route_index == 0 &&
              fixture.runtime.accepted_state.route_waypoint == 1 &&
              fixture.runtime.accepted_state.route_frames == 0 &&
              fixture.runtime.working_state.route_index == 0 &&
              fixture.runtime.working_state.route_waypoint == 1 &&
              fixture.runtime.working_state.route_frames == 0,
          "successful reset/switch installs identical route cursors");
    check(g1_frame_publication_is_valid(fixture.runtime.publication) &&
              !fixture.runtime.publication.rejection.rejected &&
              !fixture.runtime.publication.ik_safe_stop_latched &&
              fixture.runtime.publication.presentation_frame == 0 &&
              g1_frame_accepted_diagnostic_is_valid(
                  fixture.runtime.accepted_diagnostic) &&
              !fixture.runtime.accepted_diagnostic.ready,
          "successful reset/switch clears rejection, latch, and diagnostics");
    check_reset_state_member(
        fixture.runtime.accepted_state,
        fixture.db,
        fixture.active_scene,
        fixture.config,
        "accepted reset state passes every independent certification gate");
    check_reset_state_member(
        fixture.runtime.working_state,
        fixture.db,
        fixture.active_scene,
        fixture.config,
        "working reset state passes every independent certification gate");
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

static void test_success_commits_atomically_and_unloads_old_model_once()
{
    transaction_fixture fixture;
    char error[512] = {};
    fixture.capture_active();
    const int old_model_id = fixture.active_model.id;
    fixture.verify_precommit = true;
    fixture.verify_commit_before_old_unload = true;

    check(fixture.switch_to(1, error, static_cast<int>(sizeof(error))), error);
    check(fixture.active_index == 1 &&
              fixture.active_scene.metadata.id == "two" &&
              fixture.active_scene.mesh_path == "two.obj" &&
              fixture.runtime.accepted_state.scene_frame == 0 &&
              fixture.runtime.working_state.scene_frame == 0 &&
              fixture.runtime.accepted_state.simulation_position.x == 2.0f &&
              fixture.runtime.working_state.simulation_position.x == 2.0f &&
              fixture.active_model.id == fixture.models.last_loaded_id,
          "successful transaction commits all candidate objects");
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

    fixture.runtime.accepted_state.scene_frame = 33;
    fixture.runtime.accepted_state.route_frames = 21;
    fixture.runtime.working_state.scene_frame = 44;
    fixture.runtime.working_state.route_frames = 22;
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
    check(g1_frame_runtime_reset(
              candidate,
              fixture.db,
              fixture.support,
              fixture.active_scene,
              fixture.config,
              error,
              static_cast<int>(sizeof(error))),
          error);
    candidate.accepted_state.footprint_status = G1FootprintInvalidInput;
    check(!scene_frame_runtime_reset_candidate_is_valid(
              candidate, fixture.config),
          "poisoned accepted candidate gate is rejected");
    check(fixture.active_matches_snapshot(),
          "accepted-candidate gate cannot publish into the live unit");

    check(g1_frame_runtime_reset(
              candidate,
              fixture.db,
              fixture.support,
              fixture.active_scene,
              fixture.config,
              error,
              static_cast<int>(sizeof(error))),
          error);
    candidate.working_state.ik_candidate_clearance_status =
        G1ClearanceInvalidInput;
    check(!scene_frame_runtime_reset_candidate_is_valid(
              candidate, fixture.config),
          "poisoned working candidate gate is rejected");
    check(fixture.active_matches_snapshot(),
          "working-candidate gate cannot publish into the live unit");

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

int main()
{
    test_scene_load_failure_and_invalid_target_skip_model_allocation();
    test_reset_failure_preserves_active_objects_and_skips_model_load();
    test_model_failures_cleanup_only_allocated_candidate();
    test_success_commits_atomically_and_unloads_old_model_once();
    test_reset_changes_only_state_and_is_transactional();
    test_repeated_switches_leave_exactly_one_live_model();
    test_each_candidate_member_gate_is_independent_and_nonpublishing();
    return 0;
}
