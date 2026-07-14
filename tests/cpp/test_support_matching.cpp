#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#include "database.h"
#include "g1_skeleton.h"
#include "support_runtime.h"

#include <cfloat>
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
        std::fprintf(stderr, "support matching test failed: %s\n", message);
        std::exit(1);
    }
}

static uint32_t bits(float value)
{
    uint32_t output = 0;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

static size_t find_required(
    const std::string& source,
    const char* needle,
    const size_t start,
    const char* message)
{
    const size_t position = source.find(needle, start);
    check(position != std::string::npos, message);
    return position;
}

static size_t count_occurrences(
    const std::string& source, const char* needle)
{
    size_t count = 0;
    size_t position = 0;
    while ((position = source.find(needle, position)) != std::string::npos) {
        ++count;
        position += std::strlen(needle);
    }
    return count;
}

static std::string call_text(
    const std::string& source, const char* function, size_t start)
{
    const size_t name = find_required(
        source, function, start, "matcher call exists");
    const size_t open = find_required(
        source, "(", name + std::strlen(function), "matcher call opens");
    int depth = 0;
    for (size_t i = open; i < source.size(); ++i) {
        if (source[i] == '(') {
            ++depth;
        } else if (source[i] == ')' && --depth == 0) {
            return source.substr(name, i - name + 1);
        }
    }
    check(false, "matcher call closes");
    return std::string();
}

static void check_matcher_call_is_support_free(
    const std::string& call, const char* message)
{
    check(call.find("state.support") == std::string::npos &&
          call.find("support_rows") == std::string::npos &&
          call.find("terrain_support") == std::string::npos &&
          call.find("terrain_support.bin") == std::string::npos,
          message);
}

static void test_pure_matching_invariance()
{
    database db;
    db.bone_positions.resize(3, G1_BoneCount);
    db.features.resize(3, 31);
    db.features_offset.resize(31);
    db.features_scale.resize(31);
    db.range_starts.resize(1);
    db.range_stops.resize(1);
    db.terrain_features.resize(3, 4);
    db.features.zero();
    db.features_offset.zero();
    db.features_scale.set(1);
    db.features(0, 0) = 2;
    db.features(1, 0) = 0;
    db.features(2, 0) = 3;
    db.range_starts(0) = 0;
    db.range_stops(0) = 3;
    database_build_bounds(db);

    array1d<float> query(31);
    query.zero();
    int before = 0;
    float before_cost = FLT_MAX;
    database_search(before, before_cost, db, query, 0, 0, 1);

    array1d<vec3> pose(G1_BoneCount);
    array1d<vec3> retargeted(G1_BoneCount);
    pose.set(vec3());
    pose(G1_Hips) = vec3(0, 0.8f, 0);
    support_pose_apply(retargeted, pose, 0.36f);

    int after = 0;
    float after_cost = FLT_MAX;
    database_search(after, after_cost, db, query, 0, 0, 1);
    check(before == 1 && after == before, "selected frame invariant");
    check(bits(before_cost) == bits(after_cost),
          "selected cost bit invariant");
    check(retargeted(G1_Simulation).y == 0.36f &&
          bits(retargeted(G1_Hips).y) == bits(pose(G1_Hips).y),
          "only support transform changed");

    const vec3 character(1.0f, -0.0f, 2.0f);
    const vec3 simulation(4.0f, 19.0f, -3.0f);
    const vec3 velocity(0.4f, 11.0f, -0.2f);
    const vec3 adjusted = horizontal_adjust_character_position(
        character, simulation, 0.1f, 0.04f);
    const vec3 velocity_adjusted =
        horizontal_adjust_character_position_by_velocity(
            character, velocity, simulation, 0.5f, 0.1f, 0.04f);
    const vec3 clamped = horizontal_clamp_character_position(
        character, simulation, 0.15f);
    check(bits(adjusted.y) == bits(character.y) &&
          bits(velocity_adjusted.y) == bits(character.y) &&
          bits(clamped.y) == bits(character.y),
          "horizontal helpers preserve exact source Y bits");
    check(adjusted.y - character.y == 0.0f &&
          velocity_adjusted.y - character.y == 0.0f &&
          clamped.y - character.y == 0.0f,
          "horizontal helper Y displacements are zero");
}

static void test_controller_source_contract(const char* path)
{
    std::ifstream input(path);
    check(input.good(), "controller source opens");
    const std::string source(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());

    const size_t update_begin = find_required(
        source, "auto update_func = [&]()", 0, "fixed update exists");
    const size_t update_end = find_required(
        source, "#if defined(PLATFORM_WEB)", update_begin,
        "fixed update closes");
    const std::string update = source.substr(
        update_begin, update_end - update_begin);

    check(count_occurrences(update, "state.transitioned = false;") == 1 &&
          count_occurrences(update, "state.transitioned = true;") == 1,
          "transition flag has one reset and one searched-frame assignment");
    const size_t transitioned_reset = find_required(
        update, "state.transitioned = false;", 0,
        "transition reset exists");
    const size_t input_work = find_required(
        update, "gamepad_get_stick", 0, "input work exists");
    const size_t query_build = find_required(
        update, "array1d<float> query", 0, "query construction exists");
    const size_t search = find_required(
        update, "database_search(", query_build, "database search exists");
    const size_t prior = find_required(
        update, "const int prior_index = state.frame_index;", query_build,
        "searched source frame is captured");
    const size_t searched_change = find_required(
        update, "best_index != prior_index", search,
        "transition compares the searched source frame");
    const size_t transition_true = find_required(
        update, "state.transitioned = true;", searched_change,
        "real searched change sets transition");
    const size_t sequential_advance = find_required(
        update, "database_trajectory_index_clamp(\n                db, state.frame_index, 1)",
        transition_true, "sequential source advance exists");
    const size_t inertialization = find_required(
        update, "inertialize_pose_update(", sequential_advance,
        "pose inertialization exists");
    const size_t simulation_xz = find_required(
        update, "simulation_positions_update(", inertialization,
        "simulation XZ integration exists");
    const size_t preliminary_fk = find_required(
        update, "forward_kinematics_full(", simulation_xz,
        "preliminary support-local FK exists");
    const size_t observation = find_required(
        update, "support_observation_build_walkable(", preliminary_fk,
        "walkability-filtered support observation exists");
    const size_t support_update = find_required(
        update, "support_frame_update(", observation,
        "support update exists");
    const size_t adjustment = find_required(
        update, "horizontal_adjust_character_position", support_update,
        "horizontal adjustment follows support update");
    const size_t clamp = find_required(
        update, "horizontal_clamp_character_position", adjustment,
        "horizontal clamp follows adjustment");
    const size_t support_apply = find_required(
        update, "support_pose_apply(", clamp,
        "world support follows horizontal root work");
    const size_t final_fk = find_required(
        update, "forward_kinematics_full(", support_apply,
        "final FK follows world support");

    check(transitioned_reset < input_work && input_work < query_build &&
          query_build < prior && prior < search && search < searched_change &&
          searched_change < transition_true &&
          transition_true < sequential_advance &&
          sequential_advance < inertialization &&
          inertialization < simulation_xz && simulation_xz < preliminary_fk &&
          preliminary_fk < observation && observation < support_update &&
          support_update < adjustment && adjustment < clamp &&
          clamp < support_apply && support_apply < final_fk,
          "fixed update stage order preserves matching and rebases support");
    check(count_occurrences(
              update, "support_observation_build_walkable(") == 1 &&
          count_occurrences(update, "support_frame_update(") == 1 &&
          count_occurrences(update, "support_pose_apply(") == 1 &&
          count_occurrences(update, "forward_kinematics_full(") == 2,
          "support is observed, updated, and applied exactly once");
    const std::string observation_call = call_text(
        update, "support_observation_build_walkable", preliminary_fk);
    check(observation_call.find("support_rows") != std::string::npos &&
          observation_call.find("state.frame_index") != std::string::npos &&
          observation_call.find("active_scene.terrain") !=
              std::string::npos &&
          observation_call.find("active_scene.walkability") !=
              std::string::npos &&
          observation_call.find(
              "state.global_bone_positions(G1_Simulation)") !=
              std::string::npos &&
          observation_call.find(
              "state.global_bone_positions(G1_LeftToe)") !=
              std::string::npos &&
          observation_call.find(
              "state.global_bone_positions(G1_RightToe)") !=
              std::string::npos &&
          observation_call.find("state.curr_bone_contacts(0)") !=
              std::string::npos &&
          observation_call.find("state.curr_bone_contacts(1)") !=
              std::string::npos,
          "support observes the current G1 source frame and recorded contacts");
    const std::string update_call = call_text(
        update, "support_frame_update", observation);
    check(update_call.find("state.support_observation_now") !=
              std::string::npos &&
          update_call.find("state.transitioned") != std::string::npos &&
          update_call.find("dt") != std::string::npos,
          "support inertializer rebases from the same-frame transition flag");

    const std::string query_to_search = update.substr(
        query_build, search - query_build);
    check(query_to_search.find("state.support") == std::string::npos &&
          query_to_search.find("support_rows") == std::string::npos &&
          query_to_search.find("terrain_support") == std::string::npos,
          "query construction has no support coupling");
    check(query_to_search.find("active_scene.terrain") != std::string::npos &&
          query_to_search.find("runtime_terrain") == std::string::npos,
          "query samples only active-scene G1HF/v2");
    check_matcher_call_is_support_free(
        call_text(update, "database_search", query_build),
        "database search has no support coupling");
    const char* isolated_apis[] = {
        "database_frame_cost",
        "database_raw_terrain_error",
        "database_build_bounds",
    };
    for (const char* api : isolated_apis) {
        size_t call_start = 0;
        while ((call_start = source.find(api, call_start)) !=
               std::string::npos) {
            check_matcher_call_is_support_free(
                call_text(source, api, call_start),
                "matching cost/bounds API has no support coupling");
            ++call_start;
        }
    }
    const size_t startup_builder = find_required(
        source, "database_build_matching_features(", 0,
        "startup feature builder exists");
    const size_t ui_builder = find_required(
        source, "database_build_matching_features(", startup_builder + 1,
        "UI feature builder exists");
    check(source.find("database_build_matching_features(", ui_builder + 1) ==
              std::string::npos,
          "only startup and UI feature builders exist");
    const std::string startup_builder_call = call_text(
        source, "database_build_matching_features", startup_builder);
    const std::string ui_builder_call = call_text(
        source, "database_build_matching_features", ui_builder);
    check_matcher_call_is_support_free(
        startup_builder_call, "startup feature builder has no support coupling");
    check_matcher_call_is_support_free(
        ui_builder_call, "UI feature builder has no support coupling");
    check(startup_builder_call.find("effective_terrain_weight") !=
              std::string::npos &&
          ui_builder_call.find("requested_terrain_weight") !=
              std::string::npos,
          "feature builders consume only published or requested UI weight");
    const size_t rebuild_validation = find_required(
        source, "g1_matching_features_validate(", ui_builder,
        "UI rebuild validates 31D features");
    const size_t effective_publish = find_required(
        source, "effective_terrain_weight = requested_terrain_weight;",
        rebuild_validation, "validated UI weight is published");
    check(ui_builder < rebuild_validation &&
          rebuild_validation < effective_publish,
          "effective terrain weight publishes only after validation");

    check(source.find("vec3 adjust_character_position(") ==
              std::string::npos &&
          source.find("vec3 adjust_character_position_by_velocity(") ==
              std::string::npos &&
          source.find("vec3 clamp_character_position(") ==
              std::string::npos,
          "dead 3D root-position alternatives are absent");
    check(update.find(
              "state.adjusted_bone_positions = state.bone_positions;") ==
              std::string::npos,
          "support output is not overwritten before final FK");
    check(source.find("static constexpr bool ik_enabled = false;") !=
              std::string::npos &&
          source.find("static constexpr bool lmm_enabled = false;") !=
              std::string::npos &&
          source.find("if constexpr (ik_enabled)") != std::string::npos,
          "IK and LMM stay compile-time disabled");
    check(source.find("log_row.support_retargeting_enabled = true;") !=
              std::string::npos &&
          source.find("world-Y support enabled; IK disabled") !=
              std::string::npos,
          "support-enabled log and UI claims are honest");
    check(source.find("requested_terrain_weight") != std::string::npos &&
          source.find("effective_terrain_weight") != std::string::npos &&
          source.find("applied_feature_weight_terrain") == std::string::npos &&
          source.find("feature_weight_terrain") == std::string::npos,
          "terrain weight has separate requested and effective state");
    check(source.find(
              "log_row.effective_terrain_weight =\n"
              "                effective_terrain_weight;") !=
              std::string::npos &&
          source.find("requested %.3f (unapplied %.3f)") !=
              std::string::npos &&
          source.find("&requested_terrain_weight") != std::string::npos,
          "logs use effective weight and UI exposes unapplied requests");
    check(source.find("runtime_terrain") == std::string::npos,
          "legacy runtime terrain alias is absent");
}

int main(int argc, char** argv)
{
    test_pure_matching_invariance();
    if (argc == 3 && std::strcmp(argv[1], "--controller") == 0) {
        test_controller_source_contract(argv[2]);
    } else {
        check(argc == 1, "usage: test_support_matching [--controller path]");
    }
    return 0;
}
