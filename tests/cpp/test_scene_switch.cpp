#include "scene_switch.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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
    db.bone_positions.resize(1, G1_BoneCount);
    db.bone_velocities.resize(1, G1_BoneCount);
    db.bone_rotations.resize(1, G1_BoneCount);
    db.bone_angular_velocities.resize(1, G1_BoneCount);
    db.contact_states.resize(1, 2);
    db.range_starts.resize(1);
    db.range_stops.resize(1);
    db.bone_positions.set(vec3());
    db.bone_velocities.set(vec3());
    db.bone_rotations.set(quat());
    db.bone_angular_velocities.set(vec3());
    db.contact_states.zero();
    db.range_starts(0) = 0;
    db.range_stops(0) = 1;
}

static scene_pack fixture_scene(const char* id, float x)
{
    scene_pack scene;
    scene.metadata.id = id;
    scene.metadata.spawn_position = vec3(x, 0.0f, 0.0f);
    scene.metadata.spawn_yaw = 0.0f;
    scene.metadata.playable_bounds = {x - 1.0f, -1.0f, x + 1.0f, 1.0f};
    scene.terrain.version = 2;
    scene.terrain.nx = 2;
    scene.terrain.nz = 2;
    scene.terrain.origin_x = x - 1.0f;
    scene.terrain.origin_z = -1.0f;
    scene.terrain.cell_size = 2.0f;
    scene.terrain.exterior_height = 0.0f;
    scene.terrain.heights.resize(4);
    scene.terrain.heights.zero();
    scene.walkability.nx = 2;
    scene.walkability.nz = 2;
    scene.walkability.cells.resize(4);
    scene.walkability.cells.set(1);
    scene.mesh_path = std::string(id) + ".obj";
    return scene;
}

struct transaction_fixture
{
    database db;
    terrain_support_set support;
    scene_catalog catalog;
    motion_pack_manifest manifest;
    scene_pack active_scene = fixture_scene("one", 0.0f);
    g1_controller_state state;
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
    int snapshot_scene_frame = 0;
    float snapshot_simulation_x = 0.0f;
    vec3* snapshot_bone_positions = NULL;

    transaction_fixture()
    {
        fixture_db(db);
        support.values.resize(1, 3);
        support.values.zero();
        catalog.ids = {"one", "two"};
        char error[512] = {};
        check(g1_controller_state_reset(
                  state,
                  db,
                  support,
                  active_scene,
                  error,
                  static_cast<int>(sizeof(error))),
              error);
        state.scene_frame = 17;
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
        snapshot_scene_frame = state.scene_frame;
        snapshot_simulation_x = state.simulation_position.x;
        snapshot_bone_positions = state.bone_positions.data;
    }

    bool active_matches_snapshot() const
    {
        return active_index == snapshot_index &&
               active_scene.metadata.id == snapshot_scene_id &&
               active_scene.mesh_path == snapshot_mesh_path &&
               active_scene.terrain.heights.data == snapshot_heights &&
               active_model.id == snapshot_model_id &&
               state.scene_frame == snapshot_scene_frame &&
               state.simulation_position.x == snapshot_simulation_x &&
               state.bone_positions.data == snapshot_bone_positions;
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
                index == 0 ? 0.0f : 2.0f);
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
                          state.simulation_position.x ==
                              (target == 0 ? 0.0f : 2.0f) &&
                          active_model.id == models.last_loaded_id,
                      "scene, state, model, and index commit before old unload");
            }
            ++models.unloads;
            --models.live;
            models.last_unloaded_id = model.id;
            models.unloaded_ids.push_back(model.id);
            model.id = 0;
        };
        return scene_switch_transaction(
            active_scene,
            state,
            active_model,
            active_index,
            target,
            db,
            support,
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
    check(std::string(error).find("spawn is invalid") != std::string::npos,
          "candidate reset failure reason is preserved");
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
    check(std::string(error).find("not ready") != std::string::npos,
          "allocated model failure reason is preserved");
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
              fixture.state.scene_frame == 0 &&
              fixture.state.simulation_position.x == 2.0f &&
              fixture.active_model.id == fixture.models.last_loaded_id,
          "successful transaction commits all candidate objects");
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

    fixture.state.scene_frame = 91;
    fixture.state.simulation_position.x = 99.0f;
    check(scene_reset_current(
              fixture.state,
              fixture.db,
              fixture.support,
              fixture.active_scene,
              error,
              static_cast<int>(sizeof(error))),
          error);
    check(fixture.state.scene_frame == 0 &&
              fixture.state.simulation_position.x == 0.0f,
          "reset commits fresh state for the current scene");
    check(fixture.scene_loads == scene_loads &&
              fixture.models.loads == model_loads &&
              fixture.models.unloads == model_unloads &&
              fixture.active_model.id == model_id &&
              fixture.active_scene.metadata.id == scene_id &&
              fixture.active_scene.mesh_path == mesh_path &&
              fixture.active_scene.terrain.heights.data == heights,
          "reset does not reload or replace scene or model");

    fixture.state.scene_frame = 33;
    fixture.state.simulation_position.x = 44.0f;
    fixture.capture_active();
    terrain_support_set bad_support;
    bad_support.values = fixture.support.values;
    bad_support.values(0, 0) = std::numeric_limits<float>::quiet_NaN();
    error[0] = '\0';
    check(!scene_reset_current(
              fixture.state,
              fixture.db,
              bad_support,
              fixture.active_scene,
              error,
              static_cast<int>(sizeof(error))),
          "failed current-scene reset is reported");
    check(std::string(error).find("non-finite initial support") !=
              std::string::npos,
          "failed current-scene reset reason is preserved");
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

int main()
{
    test_scene_load_failure_and_invalid_target_skip_model_allocation();
    test_reset_failure_preserves_active_objects_and_skips_model_load();
    test_model_failures_cleanup_only_allocated_candidate();
    test_success_commits_atomically_and_unloads_old_model_once();
    test_reset_changes_only_state_and_is_transactional();
    test_repeated_switches_leave_exactly_one_live_model();
    return 0;
}
