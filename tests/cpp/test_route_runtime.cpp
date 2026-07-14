#include "route_runtime.h"
#include "g1_runtime_diagnostics.h"
#include "motion_match_log.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "route runtime test failed: %s\n", message);
        std::exit(1);
    }
}

static uint32_t bits(float value)
{
    uint32_t out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

static std::string read_source(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    check(file.good(), "open source contract input");
    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());
}

static void check_snapshot_publication_contract(const char* path)
{
    const std::string source = read_source(path);
    const size_t build = source.find("if (!g1_runtime_diagnostics_build(");
    const size_t serialize = source.find(
        "if (!motion_match_query_bits_hex(", build);
    const size_t suffix = source.find("log_row.source_name =", serialize);
    const size_t write = source.find(
        "if (!deterministic_log.write(", suffix);
    const size_t publish = source.find(
        "runtime_snapshot = snapshot_candidate;", build);
    const size_t ready = source.find(
        "runtime_snapshot_ready = true;", publish);
    const size_t clear = source.find(
        "scene_switch_failed = false;", ready);
    check(build != std::string::npos && serialize != std::string::npos &&
              suffix != std::string::npos && write != std::string::npos &&
              publish != std::string::npos && ready != std::string::npos &&
              clear != std::string::npos,
          "snapshot publication source markers exist");
    check(build < serialize && serialize < suffix && suffix < write &&
              write < publish && publish < ready && ready < clear,
          "snapshot publishes only after successful serialization and write");
    const std::string suffix_population = source.substr(suffix, write - suffix);
    check(suffix_population.find("runtime_snapshot.") == std::string::npos,
          "runtime suffix population does not read persistent snapshot");
    check(suffix_population.find("snapshot_candidate.") != std::string::npos,
          "runtime suffix population reads local snapshot candidate");
}

static void check_route_sample_cursor_contract(const char* path)
{
    const std::string source = read_source(path);
    check(source.find(
              "for (int step = 0; step <= steps; ++step)") ==
              std::string::npos,
          "route target sampling does not increment INT_MAX int cursor");
    const size_t loop = source.find("for (int64_t sample_index = 0;");
    const size_t cast = source.find(
        "const int step = static_cast<int>(sample_index);", loop);
    check(loop != std::string::npos && cast != std::string::npos && loop < cast,
          "route target sampling uses a safe 64-bit cursor");
}

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--controller") == 0) {
        check_snapshot_publication_contract(argv[2]);
        return 0;
    }
    if (argc == 3 && std::strcmp(argv[1], "--route-header") == 0) {
        check_route_sample_cursor_contract(argv[2]);
        return 0;
    }
    check(argc == 1,
          "usage: test_route_runtime [--controller path|--route-header path]");

    scene_route route;
    route.id = "corner";
    route.expected_outcome = "traverse";
    route.walkability_class = 1;
    route.landing_hold_seconds = 2;
    route.waypoints_xz = {{0, 0}, {1, 0}, {1, 1}, {2, 1}};

    deterministic_route_sample a, b;
    char error[256] = {};
    check(deterministic_route_command(
              a, route, 0, 0.04f, 0.50f, error, sizeof(error)),
          error);
    check(a.waypoint == 1 && bits(a.command.x) == bits(0.50f) &&
              a.command.z == 0 && !a.complete,
          "first segment");
    check(deterministic_route_command(
              a, route, 49, 0.04f, 0.50f, error, sizeof(error)),
          error);
    check(a.waypoint == 1 && a.command.x == 0.50f,
          "last frame first segment");
    check(deterministic_route_command(
              a, route, 50, 0.04f, 0.50f, error, sizeof(error)),
          error);
    check(a.waypoint == 2 && a.command.x == 0 && a.command.z == 0.50f,
          "second segment");
    check(deterministic_route_command(
              a, route, 100, 0.04f, 0.50f, error, sizeof(error)),
          error);
    check(!a.complete && a.waypoint == 2 && a.command.x == 0 &&
              a.command.z == 0,
          "landing hold begins");
    check(deterministic_route_command(
              a, route, 149, 0.04f, 0.50f, error, sizeof(error)),
          error);
    check(!a.complete && a.waypoint == 2 && a.command.x == 0 &&
              a.command.z == 0,
          "landing hold lasts two seconds");
    check(deterministic_route_command(
              a, route, 150, 0.04f, 0.50f, error, sizeof(error)),
          error);
    check(!a.complete && a.waypoint == 3 && a.command.x == 0.50f &&
              a.command.z == 0,
          "motion resumes after hold");
    check(deterministic_route_command(
              a, route, 200, 0.04f, 0.50f, error, sizeof(error)),
          error);
    check(a.complete && a.command.x == 0 && a.command.z == 0,
          "route complete");
    check(deterministic_route_motion_frames(route) == 200,
          "motion count includes hold");
    for (int frame = 0; frame < 225; ++frame) {
        check(deterministic_route_command(
                  a, route, frame, 0.04f, 0.50f, error, sizeof(error)),
              error);
        check(deterministic_route_command(
                  b, route, frame, 0.04f, 0.50f, error, sizeof(error)),
              error);
        check(bits(a.command.x) == bits(b.command.x) &&
                  bits(a.command.z) == bits(b.command.z) &&
                  a.waypoint == b.waypoint && a.complete == b.complete,
              "repeatable route input");
    }
    float rounded_sample = 0;
    check(scene_binary32_lerp(
              rounded_sample,
              -4.821664810180664f,
              0.22549442946910858f,
              13,
              23),
          "awkward route interpolation");
    check(bits(rounded_sample) == UINT32_C(0xbffc05aa),
          "route mul then add rounds separately without FMA");
    route.waypoints_xz[1] = route.waypoints_xz[0];
    const deterministic_route_sample preserved = a;
    check(!deterministic_route_command(
              a, route, 0, 0.04f, 0.50f, error, sizeof(error)),
          "zero segment rejected");
    check(bits(a.command.x) == bits(preserved.command.x) &&
              bits(a.command.z) == bits(preserved.command.z) &&
              a.waypoint == preserved.waypoint &&
              a.complete == preserved.complete,
          "route failure preserves output");

    route.waypoints_xz = {{0, 0}, {1, 0}, {1, 1}, {2, 1}};
    check(!deterministic_route_command(
              a, route, -1, 0.04f, 0.50f, error, sizeof(error)),
          "negative frame rejected");
    check(!deterministic_route_command(
              a, route, 0, 0.0f, 0.50f, error, sizeof(error)),
          "zero dt rejected");
    check(!deterministic_route_command(
              a, route, 0, 0.04f, INFINITY, error, sizeof(error)),
          "infinite speed rejected");
    route.landing_hold_seconds = 0x1p31f;
    check(deterministic_route_motion_frames(route, 1.0f, 1.0f) == -1,
          "hold frame int overflow rejected");
    route.landing_hold_seconds = 0.0f;
    route.waypoints_xz = {
        {0.0f, 0.0f}, {20000000.0f, 0.0f},
        {40000000.0f, 0.0f}, {60000000.0f, 0.0f}};
    check(deterministic_route_motion_frames(route) == -1,
          "total frame int overflow rejected");
    check(!deterministic_route_command(
              a, route, 0, 0.04f, 0.50f, error, sizeof(error)),
          "overflowing route schedule rejected");

    heightfield terrain;
    terrain.version = 2;
    terrain.nx = 3;
    terrain.nz = 2;
    terrain.origin_x = 0.0f;
    terrain.origin_z = 0.0f;
    terrain.cell_size = 1.0f;
    terrain.exterior_height = 0.0f;
    terrain.heights.resize(6);
    terrain.heights(0) = 0.0f;
    terrain.heights(1) = 1.0f;
    terrain.heights(2) = 1.0f;
    terrain.heights(3) = 0.0f;
    terrain.heights(4) = 1.0f;
    terrain.heights(5) = 1.0f;
    scene_route height_route;
    height_route.id = "height";
    height_route.waypoints_xz = {{0.0f, 0.0f}, {2.0f, 0.0f}};
    check(bits(deterministic_route_target_height(height_route, terrain)) ==
              bits(1.0f),
          "route target chooses persistent elevated height");
    height_route.waypoints_xz.clear();
    check(!terrain_float_is_finite(
              deterministic_route_target_height(height_route, terrain)),
          "invalid route target input rejected");

    motion_pack_manifest manifest;
    manifest.sources.push_back(
        motion_source_record{"source", "flat", 10, 20});
    g1_controller_state state;
    state.frame_index = 12;
    state.scene_frame = 7;
    state.incumbent_cost = 1.25f;
    state.support_observation_now.source_height[0] = 0.1f;
    state.support_observation_now.source_height[1] = 0.2f;
    state.support_observation_now.source_height[2] = 0.3f;
    state.support_observation_now.runtime_height[0] = 0.4f;
    state.support_observation_now.runtime_height[1] = 0.5f;
    state.support_observation_now.runtime_height[2] = 0.6f;
    state.support_observation_now.delta[0] = 0.3f;
    state.support_observation_now.delta[1] = 0.3f;
    state.support_observation_now.delta[2] = 0.3f;
    state.support_observation_now.contact[0] = true;
    state.support_observation_now.contact[1] = false;
    state.support.height = 0.3f;
    state.support.velocity = 0.01f;
    state.support.nominal_height = 0.3f;
    state.support.nominal_velocity = 0.01f;
    state.support.offset_height = 0.0f;
    state.support.offset_velocity = 0.0f;
    state.support.airborne_frames = 0;
    state.support.source = support_left;
    state.support.initialized = true;
    state.global_bone_positions.resize(G1_BoneCount);
    state.global_bone_positions.set(vec3());
    state.global_bone_positions(G1_Hips).y = 0.9f;
    state.simulation_position = vec3(1.0f, 0.0f, 2.0f);
    state.walkability_class = 1;
    traversability_diagnostics traversal;
    traversal.walkability_class = 1;
    traversal.reason = walkability_clear;
    traversal.distance = FLT_MAX;
    traversal.commanded_speed = 0.50f;
    traversal.applied_speed = 0.40f;
    traversal.point = vec3(3.0f, 0.0f, 4.0f);
    deterministic_route_sample diagnostic_route;
    diagnostic_route.command = vec3(0.50f, 0.0f, 0.0f);
    diagnostic_route.waypoint = 2;
    g1_runtime_diagnostic_snapshot snapshot;
    check(g1_runtime_diagnostics_build(
              snapshot, manifest, state, traversal, diagnostic_route, 1.0f,
              0, 1, false, 1, 1, 0, error, sizeof(error)),
          error);
    check(std::strcmp(snapshot.source_name, "source") == 0 &&
              std::strcmp(snapshot.source_terrain, "flat") == 0 &&
              snapshot.source_index == 0 &&
              bits(snapshot.continuation_cost) == bits(1.25f) &&
              bits(snapshot.source_left_toe_height) == bits(0.2f) &&
              bits(snapshot.runtime_support_right_toe_height) == bits(0.6f) &&
              bits(snapshot.support_root_delta) == bits(0.3f) &&
              std::strcmp(snapshot.support_source, "left") == 0 &&
              snapshot.left_contact && !snapshot.right_contact &&
              bits(snapshot.support_retargeted_hips_y) == bits(0.9f) &&
              bits(snapshot.ik_adjusted_hips_y) == bits(0.9f) &&
              snapshot.route_waypoint == 2 && !snapshot.route_complete &&
              snapshot.scene_frame == 7 && snapshot.live_model_count == 1,
          "diagnostic snapshot maps one shared runtime state");

    const g1_runtime_diagnostic_snapshot preserved_snapshot = snapshot;
    state.support.nominal_height = INFINITY;
    check(!g1_runtime_diagnostics_build(
              snapshot, manifest, state, traversal, diagnostic_route, 1.0f,
              0, 1, false, 1, 1, 0, error, sizeof(error)),
          "non-finite support state rejected");
    check(snapshot.source_index == preserved_snapshot.source_index &&
              bits(snapshot.support_height) ==
                  bits(preserved_snapshot.support_height),
          "diagnostic failure preserves output");
    state.support.nominal_height = 0.3f;
    state.walkability_class = 3;
    check(!g1_runtime_diagnostics_build(
              snapshot, manifest, state, traversal, diagnostic_route, 1.0f,
              0, 1, false, 1, 1, 0, error, sizeof(error)),
          "invalid current walkability class rejected");
    state.walkability_class = 1;
    check(!g1_runtime_diagnostics_build(
              snapshot, manifest, state, traversal, diagnostic_route, 1.0f,
              0, 1, false, 1, 2, 0, error, sizeof(error)),
          "more than one live model rejected");

    const char* log_path = "/tmp/test_route_runtime_log.csv";
    (void)std::remove(log_path);
    motion_match_log log;
    check(log.open(log_path, error, sizeof(error)), error);
    motion_match_log_row row;
    row.source_name = "s";
    row.source_terrain = "t";
    row.source_index = 3;
    row.continuation_cost = 1.0f;
    row.source_root_height = 2.0f;
    row.source_left_toe_height = 3.0f;
    row.source_right_toe_height = 4.0f;
    row.runtime_support_root_height = 5.0f;
    row.runtime_support_left_toe_height = 6.0f;
    row.runtime_support_right_toe_height = 7.0f;
    row.support_root_delta = 8.0f;
    row.support_left_toe_delta = 9.0f;
    row.support_right_toe_delta = 10.0f;
    row.support_height = 11.0f;
    row.support_velocity = 12.0f;
    row.support_source = "both";
    row.airborne_frames = 13;
    row.left_contact = true;
    row.right_contact = false;
    row.support_retargeted_hips_y = 14.0f;
    row.ik_adjusted_hips_y = 15.0f;
    row.simulation_x = 16.0f;
    row.simulation_z = 17.0f;
    row.walkability_class = 2;
    row.blocked = true;
    row.blocked_reason = "blocked-cell";
    row.blocked_distance = 18.0f;
    row.blocked_point_x = 19.0f;
    row.blocked_point_z = 20.0f;
    row.commanded_speed = 0.5f;
    row.applied_speed = 0.25f;
    row.route_waypoint = 4;
    row.route_complete = true;
    row.route_target_height = 21.0f;
    row.scene_generation = 5;
    row.scene_frame = 6;
    row.scene_reset_count = 7;
    row.scene_switch_failed = true;
    row.motion_pack_load_count = 1;
    row.model_load_count = 8;
    row.model_unload_count = 7;
    row.live_model_count = 1;
    check(log.write(row, error, sizeof(error)), error);
    check(log.close(error, sizeof(error)), error);
    FILE* log_file = std::fopen(log_path, "rb");
    check(log_file != NULL, "open exact runtime log");
    char header[8192] = {};
    char data[8192] = {};
    check(std::fgets(header, sizeof(header), log_file) != NULL,
          "read runtime header");
    check(std::fgets(data, sizeof(data), log_file) != NULL,
          "read runtime row");
    check(std::fclose(log_file) == 0, "close exact runtime log");
    const char* expected_header_suffix =
        ",source_name,source_terrain,source_index,continuation_cost,"
        "source_root_height,source_left_toe_height,source_right_toe_height,"
        "runtime_support_root_height,runtime_support_left_toe_height,"
        "runtime_support_right_toe_height,support_root_delta,"
        "support_left_toe_delta,support_right_toe_delta,support_height,"
        "support_velocity,support_source,airborne_frames,left_contact,"
        "right_contact,support_retargeted_hips_y,ik_adjusted_hips_y,"
        "simulation_x,simulation_z,walkability_class,blocked,blocked_reason,"
        "blocked_distance,blocked_point_x,blocked_point_z,commanded_speed,"
        "applied_speed,route_waypoint,route_complete,route_target_height,"
        "scene_generation,scene_frame,scene_reset_count,scene_switch_failed,"
        "motion_pack_load_count,model_load_count,model_unload_count,"
        "live_model_count\n";
    check(std::strstr(header, expected_header_suffix) != NULL,
          "exact append-only runtime header suffix");
    check(std::strstr(
              data,
              ",s,t,3,1,2,3,4,5,6,7,8,9,10,11,12,both,13,1,0,14,15,"
              "16,17,2,1,blocked-cell,18,19,20,0.5,0.25,4,1,21,5,6,7,1,"
              "1,8,7,1\n") != NULL,
          "runtime row suffix matches header order");
    check(std::remove(log_path) == 0, "remove exact runtime log");
    return 0;
}
