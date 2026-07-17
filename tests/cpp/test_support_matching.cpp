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

static std::string compact_source(const std::string& source)
{
    std::string output;
    output.reserve(source.size());
    for (const char value : source) {
        if (value != ' ' && value != '\t' &&
            value != '\r' && value != '\n') {
            output.push_back(value);
        }
    }
    return output;
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

    check(source.find("static constexpr bool ik_enabled = false;") ==
              std::string::npos &&
          source.find("if constexpr (ik_enabled)") == std::string::npos,
          "Task6 runtime IK must replace the compile-time-disabled IK path");
    check(source.find("auto update_func = [&]()") == std::string::npos,
          "Task6 uses a named noncapturing frame runner, not the old update lambda");
    check(source.find("#include \"g1_controller_frame_runtime.h\"") !=
              std::string::npos,
          "controller includes the shared typed production-runner declaration");

    const size_t runner = find_required(
        source, "G1FrameStageOutcome g1_controller_frame_stage_run(", 0,
        "Task6 named production stage runner exists");
    const size_t signature_end = find_required(
        source, ")", runner, "production runner signature closes");
    const std::string signature = compact_source(source.substr(
        runner, signature_end - runner + 1));
    check(signature.find("G1FrameTransactionStagestage") !=
              std::string::npos &&
          signature.find("g1_controller_state&working_state") !=
              std::string::npos &&
          signature.find("G1FrameTransactionScratch&scratch") !=
              std::string::npos &&
          signature.find("constG1FrameExternalInputs&external") !=
              std::string::npos &&
          signature.find("char*error") != std::string::npos &&
          signature.find("interror_capacity") != std::string::npos,
          "production runner has the exact typed noncapturing authority boundary");

    const size_t body_open = find_required(
        source, "{", signature_end, "production runner body opens");
    int brace_depth = 0;
    size_t body_close = std::string::npos;
    for (size_t cursor = body_open; cursor < source.size(); ++cursor) {
        if (source[cursor] == '{') {
            ++brace_depth;
        } else if (source[cursor] == '}' && --brace_depth == 0) {
            body_close = cursor;
            break;
        }
    }
    check(body_close != std::string::npos,
          "production runner body closes");
    const std::string body = compact_source(source.substr(
        body_open, body_close - body_open + 1));

    const size_t diagnostic_candidate = body.find(
        "scratch.accepted_diagnostic_candidate=diagnostic;");
    const size_t diagnostic_ready = body.find(
        "scratch.accepted_diagnostic_ready=true;", diagnostic_candidate);
    check(diagnostic_candidate != std::string::npos &&
              diagnostic_ready != std::string::npos &&
              diagnostic_candidate < diagnostic_ready &&
              count_occurrences(
                  body, "scratch.accepted_diagnostic_candidate=diagnostic;") ==
                  1 &&
              count_occurrences(
                  body, "scratch.accepted_diagnostic_ready=true;") == 1 &&
              count_occurrences(body, "accepted_diagnostic") == 2,
          "runner publishes exactly one diagnostic candidate through scratch");

    const char* forbidden_runner_authority[] = {
        "accepted_state", "publication",
        "frame_runtime", "getenv", "gamepad_get_stick",
        "GetGamepadAxisMovement", "deterministic_log", "Camera3D",
        "controlled_runtime_error", "BeginDrawing",
        "draw_g1_skeleton", "g1_ik_frame_evaluate",
    };
    for (const char* forbidden : forbidden_runner_authority) {
        check(body.find(forbidden) == std::string::npos,
              "runner has no accepted/publication/input/UI/log/cleanup authority");
    }
    check(body.find("g1_controller_state&state=working_state;") !=
              std::string::npos,
          "runner may name only its working-state parameter as local state");
    check(body.find("external.input") != std::string::npos &&
          body.find("external.tuning") != std::string::npos,
          "runner consumes immutable typed input and tuning snapshots");

    static const char* stages[] = {
        "caseG1FrameStageInputRouteCommand:",
        "caseG1FrameStageMatcherSearch:",
        "caseG1FrameStageCandidateApply:",
        "caseG1FrameStageInertialization:",
        "caseG1FrameStageSimulationUpdate:",
        "caseG1FrameStageSupportObservation:",
        "caseG1FrameStageSupportRetarget:",
        "caseG1FrameStageContactUpdate:",
        "caseG1FrameStageFootprintObservation:",
        "caseG1FrameStageRawBegin:",
        "caseG1FrameStageRawFirstFoot:",
        "caseG1FrameStageRawSecondFoot:",
        "caseG1FrameStageRawFinalFk:",
        "caseG1FrameStageRawPoseCertificate:",
        "caseG1FrameStageIkBegin:",
        "caseG1FrameStageIkFirstFoot:",
        "caseG1FrameStageIkSecondFoot:",
        "caseG1FrameStageIkFinalFk:",
        "caseG1FrameStageIkPoseCertificate:",
        "caseG1FrameStageAcceptedFinalize:",
    };
    size_t stage_positions[20] = {};
    size_t stage_start = 0;
    for (int index = 0; index < 20; ++index) {
        stage_positions[index] = find_required(
            body, stages[index], stage_start,
            "runner has every authenticated stage exactly once and in order");
        check(count_occurrences(body, stages[index]) == 1,
              "runner has one case for each authenticated stage");
        stage_start = stage_positions[index] + 1;
    }

    const std::string raw_support = body.substr(
        stage_positions[5], stage_positions[6] - stage_positions[5]);
    const size_t raw_fk = find_required(
        raw_support, "g1_ik_checked_forward_kinematics(", 0,
        "support observation begins with checked raw-pose FK");
    const size_t observation = find_required(
        raw_support, "support_observation_build(", raw_fk,
        "support observation consumes checked raw-pose FK");
    check(raw_fk < observation &&
          count_occurrences(raw_support,
              "g1_ik_checked_forward_kinematics(") == 1 &&
          count_occurrences(raw_support, "support_observation_build(") == 1,
          "raw FK and support observation each execute exactly once");

    const std::string retarget = body.substr(
        stage_positions[6], stage_positions[7] - stage_positions[6]);
    check(count_occurrences(retarget, "support_pose_apply(") == 1,
          "support-retarget stage applies the support pose exactly once");

    const std::string contact = body.substr(
        stage_positions[7], stage_positions[8] - stage_positions[7]);
    const size_t retargeted_fk = find_required(
        contact, "g1_ik_checked_forward_kinematics(", 0,
        "contact stage begins with checked support-retargeted FK");
    const size_t contact_history = find_required(
        contact, "contact_update(", retargeted_fk,
        "contact history advances from checked support-retargeted toe positions");
    check(retargeted_fk < contact_history &&
          count_occurrences(contact, "contact_update(") == 1 &&
          count_occurrences(body, "contact_update(") == 1,
          "one contact-history update follows the checked retargeted FK");
    check(contact.find("external.tuning.contact_unlock_radius") !=
              std::string::npos &&
          contact.find("external.tuning.contact_foot_height") !=
              std::string::npos &&
          contact.find("external.tuning.contact_blending_halflife") !=
              std::string::npos,
          "contact history consumes immutable typed contact tuning");

    const size_t ik_parser = find_required(
        source, "g1_parse_ik_enabled(", 0,
        "startup has a checked exact MM_IK parser");
    const size_t ik_environment = find_required(
        source, "getenv(\"MM_IK\")", ik_parser,
        "MM_IK is sampled once at startup");
    const size_t raylib_start = find_required(
        source, "InitWindow(", ik_environment,
        "Raylib starts after immutable startup parsing");
    check(ik_environment < raylib_start &&
          count_occurrences(source, "getenv(\"MM_IK\")") == 1 &&
          body.find("MM_IK") == std::string::npos,
          "MM_IK is startup-only and never reread by the runner");
    check(compact_source(source).find(
              "frame_external.tuning.ik_enabled=process_config.ik_enabled;") !=
              std::string::npos,
          "immutable startup IK is lowered into the typed frame context");
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
