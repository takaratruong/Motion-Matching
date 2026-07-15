#include "g1_controller_state.h"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

static void check(bool value, const char* message)
{
    if (!value) { std::fprintf(stderr, "controller reset test failed: %s\n", message); std::exit(1); }
}

static bool identifier_character(const char value)
{
    return (value >= 'a' && value <= 'z') ||
           (value >= 'A' && value <= 'Z') ||
           (value >= '0' && value <= '9') || value == '_';
}

static bool source_has_call(
    const std::string& source, const char* function_name)
{
    const std::string name(function_name);
    std::size_t position = 0;
    while ((position = source.find(name, position)) != std::string::npos) {
        const bool left_boundary =
            position == 0 || !identifier_character(source[position - 1]);
        std::size_t after = position + name.size();
        const bool right_boundary =
            after == source.size() || !identifier_character(source[after]);
        while (after < source.size() &&
               (source[after] == ' ' || source[after] == '\t' ||
                source[after] == '\r' || source[after] == '\n')) {
            ++after;
        }
        if (left_boundary && right_boundary && after < source.size() &&
            source[after] == '(') {
            return true;
        }
        position += name.size();
    }
    return false;
}

static std::string read_source(const char* path, const char* description)
{
    std::ifstream input(path, std::ios::binary);
    check(input.good(), description);
    const std::string source(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    check(!input.bad(), description);
    return source;
}

static std::string source_call_text(
    const std::string& source,
    const char* function_name,
    const std::size_t start,
    const char* description)
{
    const std::size_t name = source.find(function_name, start);
    check(name != std::string::npos, description);
    const std::size_t open = source.find('(', name);
    check(open != std::string::npos, description);
    int depth = 0;
    for (std::size_t i = open; i < source.size(); ++i) {
        if (source[i] == '(') {
            ++depth;
        } else if (source[i] == ')' && --depth == 0) {
            return source.substr(name, i - name + 1);
        }
    }
    check(false, description);
    return std::string();
}

static int source_call_argument_count(const std::string& call)
{
    const std::size_t open = call.find('(');
    check(open != std::string::npos, "source call has an argument list");
    int depth = 0;
    int arguments = 1;
    for (std::size_t i = open + 1; i < call.size(); ++i) {
        if (call[i] == '(') {
            ++depth;
        } else if (call[i] == ')') {
            if (depth == 0) return arguments;
            --depth;
        } else if (call[i] == ',' && depth == 0) {
            ++arguments;
        }
    }
    check(false, "source call argument list closes");
    return 0;
}

static void check_source_uses_checked_v2_queries(
    const std::string& source,
    const bool requires_centerline)
{
    const char* legacy_calls[] = {
        "heightfield_sample",
        "heightfield_sample_v1_legacy",
        "heightfield_sample_versioned",
        "terrain_centerline_snapshot_compute",
        "terrain_centerline_query"
    };
    for (const char* legacy : legacy_calls) {
        check(!source_has_call(source, legacy),
              "active controller source contains a legacy terrain query");
    }
    check(source_has_call(source, "heightfield_sample_v2"),
          "active controller source uses checked v2 height queries");
    if (requires_centerline) {
        check(source_has_call(source, "terrain_centerline_snapshot_compute_v2"),
              "controller source uses the v2 centerline snapshot");
    }
}

static std::string read_controller_source()
{
    const char* override_path = std::getenv("G1_CONTROLLER_SOURCE");
    const char* path = override_path != NULL ? override_path : "controller.cpp";
    return read_source(path, "controller source opens and reads");
}

static std::size_t count_occurrences(
    const std::string& source, const char* needle)
{
    std::size_t count = 0;
    std::size_t position = 0;
    while ((position = source.find(needle, position)) != std::string::npos) {
        ++count;
        position += std::char_traits<char>::length(needle);
    }
    return count;
}

static void test_active_scene_sources_use_checked_v2_queries()
{
    const std::string controller_source = read_controller_source();
    check_source_uses_checked_v2_queries(controller_source, true);

    const char* override_path = std::getenv("G1_STATE_SOURCE");
    const char* path =
        override_path != NULL ? override_path : "g1_controller_state.h";
    const std::string state_source =
        read_source(path, "controller state source opens and reads");
    check_source_uses_checked_v2_queries(state_source, false);
}

static void test_controller_wires_idle_match_transition_cost()
{
    const std::string source = read_controller_source();
    const std::size_t prior = source.find(
        "const int prior_index = state.frame_index;");
    check(prior != std::string::npos,
          "ordinary matcher captures the incumbent frame");
    const std::size_t policy = source.find(
        "const float transition_cost =", prior);
    check(policy != std::string::npos,
          "ordinary matcher computes a transition cost");
    const std::string policy_call = source_call_text(
        source,
        "g1_idle_match_transition_cost",
        policy,
        "ordinary matcher calls the idle transition-cost policy");
    check(source_call_argument_count(policy_call) == 2 &&
              policy_call.find("traversal.commanded_speed") !=
                  std::string::npos &&
              policy_call.find(
                  "walkability_xz_length(state.simulation_velocity)") !=
                  std::string::npos,
          "idle policy consumes raw command and planar simulation speeds");

    const std::size_t search = source.find("database_search(", policy);
    check(search != std::string::npos && policy < search,
          "idle transition cost is computed immediately before search");
    const std::string search_call = source_call_text(
        source,
        "database_search",
        policy,
        "ordinary database search follows idle policy");
    check(source_call_argument_count(search_call) == 5 &&
              search_call.find("transition_cost") != std::string::npos,
          "idle transition cost is the fifth database_search argument");
}

static void test_controller_validates_ik_geometry_before_window()
{
    const std::string source = read_controller_source();
    check(source.find("#include \"g1_ik.h\"") != std::string::npos,
          "controller includes the explicit G1 IK contract");
    const std::size_t skeleton = source.find("if (!g1_skeleton_validate(");
    const std::size_t ik = source.find(
        "if (!g1_leg_configs_validate(", skeleton);
    const std::size_t features = source.find(
        "database_build_matching_features(", skeleton);
    const std::size_t window = source.find("InitWindow(", skeleton);
    check(skeleton != std::string::npos && ik != std::string::npos &&
              features != std::string::npos && window != std::string::npos &&
              skeleton < ik && ik < features && features < window,
          "IK geometry validation follows skeleton validation and precedes "
          "features and Raylib startup");
    const std::string call = source_call_text(
        source,
        "g1_leg_configs_validate",
        skeleton,
        "controller calls the explicit G1 IK validator");
    check(source_call_argument_count(call) == 3 &&
              call.find("db") != std::string::npos &&
              call.find("artifact_error") != std::string::npos,
          "startup validator consumes the loaded database and error buffer");
}

static void test_controller_publishes_independent_travel_and_heading()
{
    const std::string source = read_controller_source();
    const std::size_t heading_parse = source.find(
        "g1_test_heading_override_parse(");
    const std::size_t heading_mode_guard = source.find(
        "if (test_heading.active", heading_parse);
    const std::size_t heading_guard_mode = source.find(
        "test_config.mode != G1_TestRoute", heading_mode_guard);
    const std::size_t window = source.find("InitWindow(");
    check(heading_parse != std::string::npos &&
              heading_mode_guard != std::string::npos &&
              heading_guard_mode != std::string::npos &&
              window != std::string::npos &&
              heading_parse < heading_mode_guard &&
              heading_mode_guard < heading_guard_mode &&
              heading_guard_mode < window,
          "heading override parses and rejects live use before Raylib startup");

    const std::size_t command = source.find(
        "const vec3 commanded_velocity = desired_velocity_curr;");
    const std::size_t heading = source.find(
        "quat desired_rotation_curr = desired_rotation_update(", command);
    const std::size_t override_guard = source.find(
        "if (test_heading.active)", heading);
    const std::size_t override_assign = source.find(
        "desired_rotation_curr = test_heading.heading;", override_guard);
    const std::size_t traversal = source.find(
        "desired_velocity_curr = traversability_limit_command(",
        override_assign);
    check(command != std::string::npos && heading != std::string::npos &&
              override_guard != std::string::npos &&
              override_assign != std::string::npos &&
              traversal != std::string::npos &&
              command < heading && heading < override_guard &&
              override_guard < override_assign && override_assign < traversal,
          "heading selection and override precede terrain velocity limiting");
    const std::string heading_call = source_call_text(
        source,
        "desired_rotation_update",
        heading,
        "controller has the current heading-selection call");
    check(heading_call.find("commanded_velocity") != std::string::npos &&
              heading_call.find("desired_velocity_curr") == std::string::npos,
          "heading selection consumes requested travel, not limited travel");
    const std::string traversal_call = source_call_text(
        source,
        "traversability_limit_command",
        traversal,
        "controller has the terrain velocity limiter call");
    check(traversal_call.find("rotation") == std::string::npos &&
              traversal_call.find("heading") == std::string::npos,
          "terrain limiter receives no heading reference");

    const std::size_t route_prediction = source.find(
        "deterministic_route_predict_commands(", traversal);
    const std::size_t heading_trajectory_override = source.find(
        "state.trajectory_desired_rotations.set(test_heading.heading);",
        route_prediction);
    const std::size_t synthetic_prediction = source.find(
        "trajectory_desired_velocities_predict(", route_prediction);
    const std::size_t position_prediction = source.find(
        "trajectory_positions_predict(", synthetic_prediction);
    const std::size_t snapshot = source.find(
        "g1_command_snapshot_build(", position_prediction);
    const std::size_t query = source.find(
        "// Make query vector for search.", snapshot);
    check(route_prediction != std::string::npos &&
              heading_trajectory_override != std::string::npos &&
              synthetic_prediction != std::string::npos &&
              position_prediction != std::string::npos &&
              snapshot != std::string::npos && query != std::string::npos &&
              route_prediction < heading_trajectory_override &&
              heading_trajectory_override < synthetic_prediction &&
              synthetic_prediction < position_prediction &&
              position_prediction < snapshot && snapshot < query,
          "route/heading prediction publishes before the immutable snapshot");
    const std::string route_call = source_call_text(
        source,
        "deterministic_route_predict_commands",
        route_prediction,
        "controller has deterministic route prediction call");
    check(route_call.find("desired_velocity_curr") != std::string::npos &&
              route_call.find("state.traversal_speed_scale") !=
                  std::string::npos &&
              route_call.find("gamepad") == std::string::npos,
          "route prediction consumes applied travel and no synthetic gamepad");
    const std::string snapshot_call = source_call_text(
        source,
        "g1_command_snapshot_build",
        snapshot,
        "controller builds complete command snapshot");
    check(source_call_argument_count(snapshot_call) == 9 &&
              snapshot_call.find("state.command") != std::string::npos &&
              snapshot_call.find("command_intent") != std::string::npos &&
              snapshot_call.find("desired_velocity_curr") !=
                  std::string::npos &&
              snapshot_call.find("state.trajectory_desired_velocities") !=
                  std::string::npos &&
              snapshot_call.find("state.trajectory_desired_rotations") !=
                  std::string::npos,
          "controller publishes all command snapshot owners transactionally");
    const std::string frame_path = source.substr(command, query - command);
    check(frame_path.find(
              "command_intent.requested_velocity = commanded_velocity;") !=
              std::string::npos &&
              frame_path.find(
                  "command_intent.desired_heading = desired_rotation_curr;") !=
                  std::string::npos,
          "immutable intent records requested travel and independent heading");
}

static void test_failed_model_load_reaches_counted_shared_cleanup()
{
    const std::string source = read_controller_source();
    const std::size_t load =
        source.find("Model terrain_model = LoadModel");
    const std::size_t camera = source.find("// Camera", load);
    check(load != std::string::npos && camera != std::string::npos,
          "terrain model startup block is present");
    const std::string startup = source.substr(load, camera - load);
    const std::size_t allocation =
        startup.find("const bool terrain_model_allocated");
    const std::size_t readiness = startup.find("IsModelReady(terrain_model)");
    const std::size_t allocation_guard =
        startup.find("if (terrain_model_allocated)");
    const std::size_t load_count = startup.find("++model_load_count;");
    check(allocation != std::string::npos &&
              readiness != std::string::npos && allocation < readiness,
          "model allocation is recorded before readiness validation");
    check(allocation_guard != std::string::npos &&
              load_count != std::string::npos &&
              allocation_guard < load_count && load_count < readiness,
          "allocated startup model is counted before readiness validation");
    check(startup.find("UnloadModel(terrain_model)") == std::string::npos &&
              startup.find("CloseWindow()") == std::string::npos &&
              startup.find("return 2;") == std::string::npos &&
              startup.find("controller_exit_code = 2;") !=
                  std::string::npos &&
              startup.find("controller_exit_requested = true;") !=
                  std::string::npos,
          "failed model startup requests the shared cleanup path");

    const std::size_t log_close = source.find("deterministic_log.close(");
    const std::size_t unload = source.find(
        "model_unloader(terrain_model);", log_close);
    const std::size_t close = source.find("CloseWindow();", unload);
    const std::size_t cleanup = source.find("cleanup_report_write(", close);
    check(log_close != std::string::npos && unload != std::string::npos &&
              close != std::string::npos && cleanup != std::string::npos &&
              log_close < unload && unload < close && close < cleanup,
          "failed startup reaches counted unload, window close, and report");
}

static void test_controller_marks_no_route_cursor_inactive_after_resets()
{
    const std::string source = read_controller_source();
    const std::size_t helper = source.find("auto configure_route_cursor");
    const std::size_t camera = source.find("// Camera", helper);
    check(helper != std::string::npos && camera != std::string::npos,
          "controller defines route cursor configuration before runtime");
    const std::string configuration = source.substr(helper, camera - helper);
    check(configuration.find(
              "current.route_index = configured_route_index;") !=
              std::string::npos &&
              configuration.find(
                  "current.route_waypoint = configured_route_index >= 0 ? "
                  "1 : 0;") !=
                  std::string::npos &&
              configuration.find("current.route_frames = 0;") !=
                  std::string::npos,
          "inactive modes expose waypoint zero while route mode starts at one");
    check(
        count_occurrences(source, "configure_route_cursor(state);") == 3,
        "startup, current-scene reset, and successful switch configure the "
        "route cursor");
}

static bool same_vec3(const vec3& first, const vec3& second)
{
    return first.x == second.x && first.y == second.y && first.z == second.z;
}

static bool same_quat(const quat& first, const quat& second)
{
    return first.w == second.w && first.x == second.x &&
           first.y == second.y && first.z == second.z;
}

static bool same_float_bits(const float first, const float second)
{
    return terrain_float_bits(first) == terrain_float_bits(second);
}

static bool same_vec3_bits(const vec3& first, const vec3& second)
{
    return same_float_bits(first.x, second.x) &&
           same_float_bits(first.y, second.y) &&
           same_float_bits(first.z, second.z);
}

static bool same_quat_bits(const quat& first, const quat& second)
{
    return same_float_bits(first.w, second.w) &&
           same_float_bits(first.x, second.x) &&
           same_float_bits(first.y, second.y) &&
           same_float_bits(first.z, second.z);
}

static bool same_command_snapshot_bits(
    const G1CommandSnapshot& first,
    const G1CommandSnapshot& second)
{
    if (!same_vec3_bits(first.intent.requested_velocity,
                        second.intent.requested_velocity) ||
        !same_quat_bits(first.intent.desired_heading,
                        second.intent.desired_heading) ||
        !same_vec3_bits(first.applied_velocity, second.applied_velocity)) {
        return false;
    }
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        if (!same_vec3_bits(first.predicted_desired_velocities[index],
                            second.predicted_desired_velocities[index]) ||
            !same_vec3_bits(first.predicted_root_positions[index],
                            second.predicted_root_positions[index]) ||
            !same_quat_bits(first.predicted_root_rotations[index],
                            second.predicted_root_rotations[index]) ||
            !same_quat_bits(first.predicted_desired_headings[index],
                            second.predicted_desired_headings[index])) {
            return false;
        }
    }
    return true;
}

static void check_idle_match_transition_cost(
    const float command_speed,
    const float planar_simulation_speed,
    const float expected,
    const char* message)
{
    check(same_float_bits(
              g1_idle_match_transition_cost(
                  command_speed, planar_simulation_speed),
              expected),
          message);
}

static void test_idle_match_transition_cost_policy()
{
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float command_above = std::nextafter(1.0e-4f, infinity);
    const float simulation_above = std::nextafter(0.05f, infinity);

    check_idle_match_transition_cost(
        0.0f, 0.0f, 1.0f, "zero speeds use exact idle transition cost");
    check_idle_match_transition_cost(
        1.0e-4f, 0.05f, 1.0f,
        "inclusive idle boundaries use exact transition cost");
    check_idle_match_transition_cost(
        command_above, 0.0f, 0.0f,
        "command just above idle boundary preserves active matching");
    check_idle_match_transition_cost(
        0.0f, simulation_above, 0.0f,
        "simulation just above settled boundary preserves active matching");
    check_idle_match_transition_cost(
        0.25f, 0.0f, 0.0f,
        "active command with stopped simulation has no transition cost");
    check_idle_match_transition_cost(
        0.0f, 0.25f, 0.0f,
        "idle command with moving simulation has no transition cost");

    const float invalid[] = {-1.0f, nan, infinity};
    for (const float value : invalid) {
        check_idle_match_transition_cost(
            value, 0.0f, 0.0f,
            "invalid command speed has no transition cost");
        check_idle_match_transition_cost(
            0.0f, value, 0.0f,
            "invalid simulation speed has no transition cost");
    }
}

static void test_scene_first_frame_seeds_desired_trajectory()
{
    const std::string controller_source = read_controller_source();
    check(source_has_call(
              controller_source,
              "g1_controller_state_seed_first_frame_desired_velocity"),
          "controller delegates first-frame seeding to resettable state");
    check(controller_source.find("rendered_frames == 0") ==
              std::string::npos &&
              controller_source.find("rendered_frames==0") ==
              std::string::npos,
          "lifetime frame count does not control scene initialization");

    g1_controller_state state;
    state.trajectory_desired_velocities.resize(4);
    state.trajectory_desired_velocities.set(vec3(9.0f, 8.0f, 7.0f));
    state.desired_velocity = vec3(1.0f, 2.0f, 3.0f);
    state.scene_frame = 0;

    g1_controller_state_seed_first_frame_desired_velocity(state);
    for (int i = 0; i < state.trajectory_desired_velocities.size; ++i) {
        check(same_vec3(
                  state.trajectory_desired_velocities(i),
                  state.desired_velocity),
              "first scene frame seeds the desired trajectory");
    }

    state.scene_frame = 1;
    state.desired_velocity = vec3(4.0f, 5.0f, 6.0f);
    g1_controller_state_seed_first_frame_desired_velocity(state);
    check(same_vec3(
              state.trajectory_desired_velocities(0),
              vec3(1.0f, 2.0f, 3.0f)),
          "later scene frames preserve the desired trajectory history");

    state.scene_frame = 0;
    g1_controller_state_seed_first_frame_desired_velocity(state);
    for (int i = 0; i < state.trajectory_desired_velocities.size; ++i) {
        check(same_vec3(
                  state.trajectory_desired_velocities(i),
                  vec3(4.0f, 5.0f, 6.0f)),
              "a subsequent scene reset seeds its own first frame");
    }
}

static void make_database(database& db)
{
    db.bone_positions.resize(2, G1_BoneCount);
    db.bone_velocities.resize(2, G1_BoneCount);
    db.bone_rotations.resize(2, G1_BoneCount);
    db.bone_angular_velocities.resize(2, G1_BoneCount);
    db.contact_states.resize(2, 2);
    db.range_starts.resize(1); db.range_stops.resize(1);
    db.range_starts(0)=0; db.range_stops(0)=2;
    for (int frame = 0; frame < 2; ++frame) {
        for (int bone = 0; bone < G1_BoneCount; ++bone) {
            db.bone_positions(frame, bone) = vec3(
                0.25f + static_cast<float>(frame),
                0.50f + 0.01f * static_cast<float>(bone),
                -0.75f - 0.02f * static_cast<float>(bone));
            db.bone_velocities(frame, bone) = vec3(
                -0.10f - static_cast<float>(frame),
                0.20f + 0.01f * static_cast<float>(bone),
                0.30f);
            db.bone_rotations(frame, bone) = quat_from_angle_axis(
                0.10f + 0.01f * static_cast<float>(bone + frame),
                vec3(0.0f, 1.0f, 0.0f));
            db.bone_angular_velocities(frame, bone) = vec3(
                0.40f,
                -0.20f - 0.01f * static_cast<float>(bone),
                0.05f + static_cast<float>(frame));
        }
    }
    db.contact_states.zero();
    db.contact_states(0, 0) = true;
    db.contact_states(1, 1) = true;
}

static scene_pack make_scene()
{
    scene_pack scene;
    scene.metadata.id="fixture";
    scene.metadata.spawn_position=vec3(10.5f,7.75f,20.25f);
    scene.metadata.spawn_yaw=0.65f;
    scene.metadata.playable_bounds={10.0f,20.0f,12.0f,22.0f};
    scene.terrain.version=2; scene.terrain.nx=2; scene.terrain.nz=2;
    scene.terrain.origin_x=10.0f; scene.terrain.origin_z=20.0f;
    scene.terrain.cell_size=2.0f; scene.terrain.exterior_height=-17.0f;
    scene.terrain.heights.resize(4);
    scene.terrain.heights(0)=1.0f;
    scene.terrain.heights(1)=5.0f;
    scene.terrain.heights(2)=9.0f;
    scene.terrain.heights(3)=21.0f;
    scene.walkability.nx=2; scene.walkability.nz=2;
    scene.walkability.cells.resize(4); scene.walkability.cells.set(1);
    return scene;
}

static void make_support(terrain_support_set& support)
{
    support.values.resize(2, 3);
    support.values(0, 0) = 1.25f;
    support.values(0, 1) = -2.0f;
    support.values(0, 2) = 6.0f;
    support.values(1, 0) = -4.0f;
    support.values(1, 1) = 8.0f;
    support.values(1, 2) = 12.0f;
}

static void poison_array(array1d<vec3>& values)
{
    values.resize(3);
    values.set(vec3(91.0f, 92.0f, 93.0f));
}

static void poison_array(array1d<quat>& values)
{
    values.resize(3);
    values.set(quat(94.0f, 95.0f, 96.0f, 97.0f));
}

static void poison_array(array1d<bool>& values)
{
    values.resize(3);
    values.set(true);
}

static void poison_array(array1d<int>& values)
{
    values.resize(3);
    values.set(98);
}

static void poison_state(g1_controller_state& state)
{
    state.frame_index = 999;
    state.scene_frame = 998;
    state.search_time = 9.0f;
    state.search_timer = -9.0f;
    state.force_search_timer = -8.0f;

    poison_array(state.curr_bone_positions);
    poison_array(state.curr_bone_velocities);
    poison_array(state.trns_bone_positions);
    poison_array(state.trns_bone_velocities);
    poison_array(state.curr_bone_rotations);
    poison_array(state.trns_bone_rotations);
    poison_array(state.curr_bone_angular_velocities);
    poison_array(state.trns_bone_angular_velocities);
    poison_array(state.curr_bone_contacts);
    poison_array(state.trns_bone_contacts);
    poison_array(state.bone_positions);
    poison_array(state.bone_velocities);
    poison_array(state.bone_angular_velocities);
    poison_array(state.bone_rotations);
    poison_array(state.bone_offset_positions);
    poison_array(state.bone_offset_velocities);
    poison_array(state.bone_offset_angular_velocities);
    poison_array(state.bone_offset_rotations);
    poison_array(state.adjusted_bone_positions);
    poison_array(state.adjusted_bone_rotations);
    poison_array(state.global_bone_positions);
    poison_array(state.global_bone_velocities);
    poison_array(state.global_bone_rotations);
    poison_array(state.global_bone_angular_velocities);
    poison_array(state.global_bone_computed);

    state.transition_src_position = vec3(31.0f, 32.0f, 33.0f);
    state.transition_dst_position = vec3(34.0f, 35.0f, 36.0f);
    state.transition_src_rotation = quat(37.0f, 38.0f, 39.0f, 40.0f);
    state.transition_dst_rotation = quat(41.0f, 42.0f, 43.0f, 44.0f);

    state.desired_velocity = vec3(1.0f, 2.0f, 3.0f);
    state.desired_velocity_change_curr = vec3(4.0f, 5.0f, 6.0f);
    state.desired_velocity_change_prev = vec3(7.0f, 8.0f, 9.0f);
    state.desired_rotation = quat(10.0f, 11.0f, 12.0f, 13.0f);
    state.desired_rotation_change_curr = vec3(14.0f, 15.0f, 16.0f);
    state.desired_rotation_change_prev = vec3(17.0f, 18.0f, 19.0f);
    state.desired_gait = 20.0f;
    state.desired_gait_velocity = 21.0f;
    state.simulation_position = vec3(22.0f, 23.0f, 24.0f);
    state.simulation_velocity = vec3(25.0f, 26.0f, 27.0f);
    state.simulation_acceleration = vec3(28.0f, 29.0f, 30.0f);
    state.simulation_rotation = quat(45.0f, 46.0f, 47.0f, 48.0f);
    state.simulation_angular_velocity = vec3(49.0f, 50.0f, 51.0f);
    poison_array(state.trajectory_desired_velocities);
    poison_array(state.trajectory_positions);
    poison_array(state.trajectory_velocities);
    poison_array(state.trajectory_accelerations);
    poison_array(state.trajectory_angular_velocities);
    poison_array(state.trajectory_desired_rotations);
    poison_array(state.trajectory_rotations);

    state.command.intent.requested_velocity = vec3(101.0f, 102.0f, 103.0f);
    state.command.intent.desired_heading =
        quat(104.0f, 105.0f, 106.0f, 107.0f);
    state.command.applied_velocity = vec3(108.0f, 109.0f, 110.0f);
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        const float offset = static_cast<float>(index);
        state.command.predicted_desired_velocities[index] =
            vec3(111.0f + offset, 112.0f + offset, 113.0f + offset);
        state.command.predicted_root_positions[index] =
            vec3(114.0f + offset, 115.0f + offset, 116.0f + offset);
        state.command.predicted_root_rotations[index] =
            quat(117.0f + offset, 118.0f + offset,
                 119.0f + offset, 120.0f + offset);
        state.command.predicted_desired_headings[index] =
            quat(121.0f + offset, 122.0f + offset,
                 123.0f + offset, 124.0f + offset);
    }

    poison_array(state.contact_bones);
    poison_array(state.contact_states);
    poison_array(state.contact_locks);
    poison_array(state.contact_positions);
    poison_array(state.contact_velocities);
    poison_array(state.contact_points);
    poison_array(state.contact_targets);
    poison_array(state.contact_offset_positions);
    poison_array(state.contact_offset_velocities);

    state.support.height = 52.0f;
    state.support.velocity = 53.0f;
    state.support.nominal_height = 54.0f;
    state.support.nominal_velocity = 55.0f;
    state.support.offset_height = 56.0f;
    state.support.offset_velocity = 57.0f;
    state.support.airborne_frames = 58;
    state.support.source = support_right;
    state.support.initialized = false;
    for (int i = 0; i < 3; ++i) {
        state.support_observation_now.source_height[i] = 59.0f + i;
        state.support_observation_now.runtime_height[i] = 62.0f + i;
        state.support_observation_now.delta[i] = 65.0f + i;
    }
    state.support_observation_now.contact[0] = true;
    state.support_observation_now.contact[1] = true;
    state.traversal_speed_scale = 0.0f;
    state.traversal_speed_scale_velocity = 68.0f;
    state.blocked = true;
    state.walkability_class = 2;
    state.blocked_distance = -1.0f;
    state.blocked_point = vec3(69.0f, 70.0f, 71.0f);

    state.route_index = 72;
    state.route_waypoint = 73;
    state.route_frames = 74;
    state.camera_azimuth = 75.0f;
    state.camera_altitude = 76.0f;
    state.camera_distance = 77.0f;

    state.searched = true;
    state.transitioned = true;
    state.incumbent_cost = 78.0f;
    state.selected_cost = 79.0f;
    state.selected_terrain_error = 80.0f;
    state.adjustment_xz = 81.0f;
    state.adjustment_y = 82.0f;
    state.clamp_xz = 83.0f;
    state.clamp_y = 84.0f;
}

static void check_vec_array(
    const array1d<vec3>& values,
    const int expected_size,
    const vec3& expected,
    const char* message)
{
    check(values.size == expected_size, message);
    for (int i = 0; i < values.size; ++i) {
        check(same_vec3(values(i), expected), message);
    }
}

static void check_quat_array(
    const array1d<quat>& values,
    const int expected_size,
    const quat& expected,
    const char* message)
{
    check(values.size == expected_size, message);
    for (int i = 0; i < values.size; ++i) {
        check(same_quat(values(i), expected), message);
    }
}

static void check_false_array(
    const array1d<bool>& values,
    const int expected_size,
    const char* message)
{
    check(values.size == expected_size, message);
    for (int i = 0; i < values.size; ++i) {
        check(!values(i), message);
    }
}

static void test_reset_clears_every_dynamic_subsystem()
{
    database db; make_database(db);
    terrain_support_set support; make_support(support);
    scene_pack scene=make_scene();
    g1_controller_state state;
    poison_state(state);

    const vec3 spawn = scene.metadata.spawn_position;
    const quat spawn_rotation = quat_from_angle_axis(
        scene.metadata.spawn_yaw, vec3(0.0f, 1.0f, 0.0f));
    const float runtime_height = heightfield_sample_v2(
        scene.terrain, spawn.x, spawn.z);
    const float bounds_midpoint_height = heightfield_sample_v2(
        scene.terrain,
        0.5f * (scene.metadata.playable_bounds.min_x +
                scene.metadata.playable_bounds.max_x),
        0.5f * (scene.metadata.playable_bounds.min_z +
                scene.metadata.playable_bounds.max_z));
    const float legacy_height = heightfield_sample(
        scene.terrain, spawn.x, spawn.z);
    const float bounds_midpoint_x =
        0.5f * (scene.metadata.playable_bounds.min_x +
                scene.metadata.playable_bounds.max_x);
    const float bounds_midpoint_z =
        0.5f * (scene.metadata.playable_bounds.min_z +
                scene.metadata.playable_bounds.max_z);
    check(runtime_height == 4.0f, "fixture has the expected v2 surface sample");
    check(spawn.x != bounds_midpoint_x && spawn.z != bounds_midpoint_z,
          "fixture metadata XZ differs from both bounds midpoint coordinates");
    check(runtime_height != legacy_height,
          "fixture distinguishes the v2 triangular sampler from legacy");
    check(runtime_height != bounds_midpoint_height,
          "fixture distinguishes metadata spawn from bounds midpoint");
    check(runtime_height != spawn.y,
          "fixture distinguishes terrain sample from metadata spawn Y");
    check(runtime_height != support.values(0, 0),
          "fixture distinguishes runtime terrain from source support");
    check(scene.metadata.spawn_yaw != 0.0f &&
              !same_quat(spawn_rotation,
                         db.bone_rotations(0, G1_Simulation)),
          "fixture metadata yaw differs from defaults and database root yaw");
    const float expected_support = runtime_height - support.values(0, 0);

    char error[512]={};
    check(g1_controller_state_reset(state,db,support,scene,error,sizeof(error)),error);

    check(state.frame_index == db.range_starts(0) && state.scene_frame == 0,
          "frame cursors reset to the first database range");
    check(state.search_time == 0.10f &&
              state.search_timer == state.search_time &&
              state.force_search_timer == state.search_time,
          "search timers reset");

    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        check(same_vec3(state.curr_bone_positions(bone),
                        db.bone_positions(0, bone)),
              "current positions map the first database frame");
        check(same_vec3(state.curr_bone_velocities(bone),
                        db.bone_velocities(0, bone)),
              "current velocities map the first database frame");
        check(same_quat(state.curr_bone_rotations(bone),
                        db.bone_rotations(0, bone)),
              "current rotations map the first database frame");
        check(same_vec3(state.curr_bone_angular_velocities(bone),
                        db.bone_angular_velocities(0, bone)),
              "current angular velocities map the first database frame");
        check(same_vec3(state.trns_bone_positions(bone),
                        state.curr_bone_positions(bone)) &&
                  same_vec3(state.trns_bone_velocities(bone),
                            state.curr_bone_velocities(bone)) &&
                  same_quat(state.trns_bone_rotations(bone),
                            state.curr_bone_rotations(bone)) &&
                  same_vec3(state.trns_bone_angular_velocities(bone),
                            state.curr_bone_angular_velocities(bone)),
              "transition pose starts from the selected pose");
    }
    check(state.curr_bone_contacts.size == 2 &&
              state.trns_bone_contacts.size == 2 &&
              state.curr_bone_contacts(0) &&
              !state.curr_bone_contacts(1) &&
              state.trns_bone_contacts(0) &&
              !state.trns_bone_contacts(1),
          "recorded contacts map the first database frame");
    check(same_vec3(state.transition_src_position,
                    db.bone_positions(0, G1_Simulation)) &&
              same_quat(state.transition_src_rotation,
                        db.bone_rotations(0, G1_Simulation)),
          "transition source retains exact database root provenance");

    check(same_float_bits(state.transition_dst_position.x, spawn.x) &&
              same_float_bits(state.transition_dst_position.z, spawn.z) &&
              state.transition_dst_position.y == 0.0f &&
              same_quat(state.transition_dst_rotation, spawn_rotation),
          "transition destination uses exact metadata XZ and yaw");
    check(same_float_bits(state.simulation_position.x, spawn.x) &&
              same_float_bits(state.simulation_position.z, spawn.z) &&
              state.simulation_position.y == 0.0f &&
              same_quat(state.simulation_rotation, spawn_rotation),
          "simulation uses exact metadata XZ/yaw and planar Y");
    check(same_vec3(state.bone_positions(G1_Simulation),
                    vec3(spawn.x, 0.0f, spawn.z)) &&
              same_quat(state.bone_rotations(G1_Simulation), spawn_rotation) &&
              same_vec3(state.bone_velocities(G1_Simulation), vec3()) &&
              same_vec3(state.bone_angular_velocities(G1_Simulation), vec3()),
          "inertial pose root is support-local at metadata spawn");
    for (int bone = 1; bone < G1_BoneCount; ++bone) {
        check(same_vec3(state.bone_positions(bone),
                        db.bone_positions(0, bone)) &&
                  same_vec3(state.bone_velocities(bone),
                            db.bone_velocities(0, bone)) &&
                  same_quat(state.bone_rotations(bone),
                            db.bone_rotations(0, bone)) &&
                  same_vec3(state.bone_angular_velocities(bone),
                            db.bone_angular_velocities(0, bone)),
              "non-root inertial pose maps the first database frame");
    }
    check_vec_array(state.bone_offset_positions, G1_BoneCount, vec3(),
                    "position offsets reset");
    check_vec_array(state.bone_offset_velocities, G1_BoneCount, vec3(),
                    "velocity offsets reset");
    check_vec_array(state.bone_offset_angular_velocities, G1_BoneCount, vec3(),
                    "angular velocity offsets reset");
    check_quat_array(state.bone_offset_rotations, G1_BoneCount, quat(),
                     "rotation offsets reset");

    check(same_vec3(state.desired_velocity, vec3()) &&
              same_vec3(state.desired_velocity_change_curr, vec3()) &&
              same_vec3(state.desired_velocity_change_prev, vec3()) &&
              same_quat(state.desired_rotation, spawn_rotation) &&
              same_vec3(state.desired_rotation_change_curr, vec3()) &&
              same_vec3(state.desired_rotation_change_prev, vec3()) &&
              state.desired_gait == 0.0f &&
              state.desired_gait_velocity == 0.0f,
          "input intent resets around metadata yaw");
    check(same_vec3(state.simulation_velocity, vec3()) &&
              same_vec3(state.simulation_acceleration, vec3()) &&
              same_vec3(state.simulation_angular_velocity, vec3()),
          "simulation derivatives reset");
    check_vec_array(state.trajectory_desired_velocities, 4, vec3(),
                    "trajectory desired velocities reset");
    check_vec_array(state.trajectory_positions, 4,
                    vec3(spawn.x, 0.0f, spawn.z),
                    "trajectory positions reset to exact metadata XZ");
    check_vec_array(state.trajectory_velocities, 4, vec3(),
                    "trajectory velocities reset");
    check_vec_array(state.trajectory_accelerations, 4, vec3(),
                    "trajectory accelerations reset");
    check_vec_array(state.trajectory_angular_velocities, 4, vec3(),
                    "trajectory angular velocities reset");
    check_quat_array(state.trajectory_desired_rotations, 4, spawn_rotation,
                     "trajectory desired rotations reset to metadata yaw");
    check_quat_array(state.trajectory_rotations, 4, spawn_rotation,
                     "trajectory rotations reset to metadata yaw");

    check(g1_command_snapshot_is_valid(state.command),
          "reset publishes a valid immutable command snapshot");
    check(same_vec3_bits(state.command.intent.requested_velocity, vec3()) &&
              same_quat_bits(
                  state.command.intent.desired_heading, spawn_rotation) &&
              same_vec3_bits(state.command.applied_velocity, vec3()),
          "reset command owns zero requested/applied travel and spawn heading");
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        check(same_vec3_bits(
                  state.command.predicted_desired_velocities[index], vec3()) &&
                  same_vec3_bits(
                      state.command.predicted_root_positions[index],
                      vec3(spawn.x, 0.0f, spawn.z)) &&
                  same_quat_bits(
                      state.command.predicted_root_rotations[index],
                      spawn_rotation) &&
                  same_quat_bits(
                      state.command.predicted_desired_headings[index],
                      spawn_rotation),
              "reset command publishes four spawn trajectory copies");
    }

    check(state.contact_bones.size == 2 &&
              state.contact_bones(0) == G1_LeftToe &&
              state.contact_bones(1) == G1_RightToe,
          "contact bone mapping resets");
    check_false_array(state.contact_states, 2, "contact states reset");
    check_false_array(state.contact_locks, 2, "contact locks reset");
    check_vec_array(state.contact_positions, 2, vec3(),
                    "contact positions reset");
    check_vec_array(state.contact_velocities, 2, vec3(),
                    "contact velocities reset");
    check_vec_array(state.contact_points, 2, vec3(),
                    "dormant contact points reset");
    check_vec_array(state.contact_targets, 2, vec3(),
                    "dormant contact targets reset");
    check_vec_array(state.contact_offset_positions, 2, vec3(),
                    "dormant contact position offsets reset");
    check_vec_array(state.contact_offset_velocities, 2, vec3(),
                    "dormant contact velocity offsets reset");

    check(same_float_bits(state.support.height, expected_support) &&
              state.support.velocity == 0.0f &&
              same_float_bits(state.support.nominal_height,
                              expected_support) &&
              state.support.nominal_velocity == 0.0f &&
              state.support.offset_height == 0.0f &&
              state.support.offset_velocity == 0.0f &&
              state.support.airborne_frames == 0 &&
              state.support.source == support_root &&
              state.support.initialized,
          "support resets from v2 spawn sample minus source row");
    for (int i = 0; i < 3; ++i) {
        check(state.support_observation_now.source_height[i] == 0.0f &&
                  state.support_observation_now.runtime_height[i] == 0.0f &&
                  state.support_observation_now.delta[i] == 0.0f,
              "support observation buffers reset");
    }
    check(!state.support_observation_now.contact[0] &&
              !state.support_observation_now.contact[1],
          "support observation contacts reset");
    check(state.traversal_speed_scale == 1.0f &&
              state.traversal_speed_scale_velocity == 0.0f &&
              !state.blocked && state.walkability_class == 1 &&
              state.blocked_distance == FLT_MAX &&
              same_vec3(state.blocked_point, vec3()),
          "traversal state resets");

    check(state.route_index == 0 && state.route_waypoint == 1 &&
              state.route_frames == 0,
          "route cursor resets");
    check(same_float_bits(state.camera_azimuth,
                          scene.metadata.spawn_yaw) &&
              state.camera_altitude == 0.4f &&
              state.camera_distance == 4.0f,
          "camera resets to exact metadata yaw and default orbit");
    check(!state.searched && !state.transitioned &&
              state.incumbent_cost == 0.0f &&
              state.selected_cost == 0.0f &&
              state.selected_terrain_error == 0.0f &&
              state.adjustment_xz == 0.0f &&
              state.adjustment_y == 0.0f &&
              state.clamp_xz == 0.0f && state.clamp_y == 0.0f,
          "per-scene diagnostics reset");

    check(state.adjusted_bone_positions.size == G1_BoneCount &&
              same_float_bits(
                  state.adjusted_bone_positions(G1_Simulation).x,
                  spawn.x) &&
              same_float_bits(
                  state.adjusted_bone_positions(G1_Simulation).y,
                  expected_support) &&
              same_float_bits(
                  state.adjusted_bone_positions(G1_Simulation).z,
                  spawn.z),
          "support applies only to adjusted Simulation Y");
    for (int bone = 1; bone < G1_BoneCount; ++bone) {
        check(same_vec3(state.adjusted_bone_positions(bone),
                        state.bone_positions(bone)),
              "adjusted non-root positions retain inertial pose");
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        check(same_quat(state.adjusted_bone_rotations(bone),
                        state.bone_rotations(bone)),
              "adjusted rotations retain inertial pose");
    }
    check_vec_array(state.global_bone_positions, G1_BoneCount, vec3(),
                    "global FK positions reset");
    check_vec_array(state.global_bone_velocities, G1_BoneCount, vec3(),
                    "global FK velocities reset");
    check_quat_array(state.global_bone_rotations, G1_BoneCount, quat(),
                     "global FK rotations reset");
    check_vec_array(state.global_bone_angular_velocities,
                    G1_BoneCount, vec3(),
                    "global FK angular velocities reset");
    check_false_array(state.global_bone_computed, G1_BoneCount,
                      "global FK computed flags reset");
}

static void test_failed_reset_preserves_prior_state()
{
    database db; make_database(db);
    terrain_support_set support; make_support(support);
    scene_pack valid=make_scene();
    g1_controller_state active;
    char error[512]={};
    check(g1_controller_state_reset(active,db,support,valid,error,sizeof(error)),error);
    active.scene_frame = 17;
    active.simulation_position = vec3(101.0f, 102.0f, 103.0f);
    active.support.height = 104.0f;
    active.bone_positions(G1_Hips) = vec3(105.0f, 106.0f, 107.0f);
    active.command.intent.requested_velocity =
        vec3(108.0f, 109.0f, 110.0f);
    const int prior_scene_frame = active.scene_frame;
    const vec3 prior_simulation = active.simulation_position;
    const float prior_support=active.support.height;
    const int prior_bone_count=active.bone_positions.size;
    vec3* const prior_bone_data=active.bone_positions.data;
    const vec3 prior_bone=active.bone_positions(G1_Hips);
    const G1CommandSnapshot prior_command = active.command;

    support.values(0, 0) = std::numeric_limits<float>::quiet_NaN();
    check(!g1_controller_state_reset(
              active,db,support,valid,error,sizeof(error)),
          "non-finite initial support rejected after candidate allocation");
    check(std::string(error).find("non-finite initial support") !=
              std::string::npos,
          "late reset failure reaches the initial-support validation");
    check(active.scene_frame == prior_scene_frame &&
              same_vec3(active.simulation_position, prior_simulation) &&
              active.support.height == prior_support,
          "active scalar state preserved after late reset failure");
    check(active.bone_positions.size==prior_bone_count&&
          active.bone_positions.data==prior_bone_data,
          "active owning array pointer and size preserved");
    check(same_vec3(active.bone_positions(G1_Hips), prior_bone),
          "active array content preserved after late reset failure");
    check(same_command_snapshot_bits(active.command, prior_command),
          "active command snapshot preserved after late reset failure");
}

static G1CommandSnapshot command_swap_fixture(float base)
{
    G1CommandSnapshot value;
    value.intent.requested_velocity = vec3(base, base + 1.0f, base + 2.0f);
    value.intent.desired_heading = quat(base + 3.0f, base + 4.0f,
                                        base + 5.0f, base + 6.0f);
    value.applied_velocity = vec3(base + 7.0f, base + 8.0f, base + 9.0f);
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        const float offset = 10.0f * static_cast<float>(index);
        value.predicted_desired_velocities[index] =
            vec3(base + 10.0f + offset,
                 base + 11.0f + offset,
                 base + 12.0f + offset);
        value.predicted_root_positions[index] =
            vec3(base + 13.0f + offset,
                 base + 14.0f + offset,
                 base + 15.0f + offset);
        value.predicted_root_rotations[index] =
            quat(base + 16.0f + offset,
                 base + 17.0f + offset,
                 base + 18.0f + offset,
                 base + 19.0f + offset);
        value.predicted_desired_headings[index] =
            quat(base + 20.0f + offset,
                 base + 21.0f + offset,
                 base + 22.0f + offset,
                 base + 23.0f + offset);
    }
    return value;
}

static void test_swap_owns_complete_command_snapshot()
{
    g1_controller_state first;
    g1_controller_state second;
    first.command = command_swap_fixture(100.0f);
    second.command = command_swap_fixture(500.0f);
    const G1CommandSnapshot first_before = first.command;
    const G1CommandSnapshot second_before = second.command;
    g1_controller_state_swap(first, second);
    check(same_command_snapshot_bits(first.command, second_before) &&
              same_command_snapshot_bits(second.command, first_before),
          "state swap exchanges every command snapshot member");
}

int main()
{
    test_active_scene_sources_use_checked_v2_queries();
    test_controller_wires_idle_match_transition_cost();
    test_controller_validates_ik_geometry_before_window();
    test_controller_publishes_independent_travel_and_heading();
    test_failed_model_load_reaches_counted_shared_cleanup();
    test_controller_marks_no_route_cursor_inactive_after_resets();
    test_idle_match_transition_cost_policy();
    test_scene_first_frame_seeds_desired_trajectory();
    test_reset_clears_every_dynamic_subsystem();
    test_failed_reset_preserves_prior_state();
    test_swap_owns_complete_command_snapshot();
    return 0;
}
