#include "g1_controller_state.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>

static_assert(!std::is_copy_constructible<g1_controller_state>::value,
              "controller state cannot bypass checked copy by value");
static_assert(!std::is_copy_assignable<g1_controller_state>::value,
              "controller state cannot bypass checked copy assignment");
static_assert(!std::is_move_constructible<g1_controller_state>::value,
              "controller state ownership moves only through named swap");
static_assert(!std::is_move_assignable<g1_controller_state>::value,
              "controller state ownership moves only through named swap");

static uint64_t state_storage_identity_digest(
    const g1_controller_state& state);
static uint64_t state_logical_digest(const g1_controller_state& state);

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

static std::string source_braced_statement_text(
    const std::string& source,
    const char* marker,
    const std::size_t start,
    const char* description)
{
    const std::size_t statement = source.find(marker, start);
    check(statement != std::string::npos, description);
    const std::size_t open = source.find('{', statement);
    check(open != std::string::npos, description);
    int depth = 0;
    for (std::size_t i = open; i < source.size(); ++i) {
        if (source[i] == '{') {
            ++depth;
        } else if (source[i] == '}' && --depth == 0) {
            return source.substr(statement, i - statement + 1);
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
        "scratch.transition_cost = ::g1_idle_match_transition_cost(",
        prior);
    check(policy != std::string::npos,
          "ordinary matcher stores transition cost in transaction scratch");
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
    check(search != std::string::npos && policy < search &&
              count_occurrences(source, "::database_search(") == 1,
          "idle transition cost is computed immediately before search");
    const std::string search_call = source_call_text(
        source,
        "database_search",
        policy,
        "ordinary database search follows idle policy");
    check(source_call_argument_count(search_call) == 5 &&
              search_call.find("scratch.transition_cost") !=
                  std::string::npos,
          "scratch transition cost is the fifth database_search argument");
    const std::size_t recovery_cost = source.find(
        "scratch.recovery_request.transition_cost =\n"
        "                scratch.transition_cost;",
        search);
    check(recovery_cost != std::string::npos && search < recovery_cost,
          "ordinary search freezes the same scratch transition-cost word into the recovery request");
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
        "test_heading.active", heading_parse);
    const std::size_t heading_guard_mode = source.find(
        "test_config.mode != G1_TestRoute", heading_mode_guard);
    const std::size_t window = source.find("InitWindow(", heading_parse);
    check(heading_parse != std::string::npos &&
              heading_mode_guard != std::string::npos &&
              heading_guard_mode != std::string::npos &&
              window != std::string::npos &&
              heading_parse < heading_mode_guard &&
              heading_mode_guard < heading_guard_mode &&
              heading_guard_mode < window,
          "heading override parses and rejects live use before Raylib startup");
    const std::string heading_parse_call = source_call_text(
        source,
        "g1_test_heading_override_parse",
        heading_parse,
        "controller has the startup heading-override parser call");
    check(source_call_argument_count(heading_parse_call) == 4 &&
              heading_parse_call.find("test_heading") != std::string::npos &&
              heading_parse_call.find("heading_text") != std::string::npos &&
              heading_parse_call.find("artifact_error") != std::string::npos,
          "startup heading parser owns the typed override and controlled error");

    const std::size_t runner = source.find(
        "G1FrameStageOutcome g1_controller_frame_stage_run(");
    const std::size_t working_alias = source.find(
        "g1_controller_state& state = working_state;", runner);
    const std::size_t input_stage = source.find(
        "case G1FrameStageInputRouteCommand:", working_alias);
    const std::size_t matcher_stage = source.find(
        "case G1FrameStageMatcherSearch:", input_stage);
    const std::size_t inertialization_stage = source.find(
        "case G1FrameStageInertialization:", matcher_stage);
    check(runner != std::string::npos &&
              working_alias != std::string::npos &&
              input_stage != std::string::npos &&
              matcher_stage != std::string::npos &&
              inertialization_stage != std::string::npos &&
              runner < working_alias && working_alias < input_stage &&
              input_stage < matcher_stage &&
              matcher_stage < inertialization_stage,
          "transactional runner has ordered input, matcher, and inertialization stages");

    const std::string input_path = source.substr(
        input_stage, matcher_stage - input_stage);
    const std::size_t command = input_path.find(
        "scratch.commanded_velocity =");
    const std::size_t heading = input_path.find(
        "quat heading = g1_runner_desired_heading(", command);
    const std::size_t override_guard = input_path.find(
        "if (external.heading_override.active)", heading);
    const std::size_t override_assign = input_path.find(
        "heading = external.heading_override.heading;", override_guard);
    const std::size_t route_heading_guard = input_path.find(
        "if (external.route != nullptr &&", override_assign);
    const std::size_t override_protection = input_path.find(
        "!external.heading_override.active", route_heading_guard);
    const std::size_t intent_velocity = input_path.find(
        "scratch.requested_intent.requested_velocity =", override_protection);
    const std::size_t intent_heading = input_path.find(
        "scratch.requested_intent.desired_heading =", intent_velocity);
    const std::size_t safe_stop = input_path.find(
        "g1_ik_safe_stop_handoff(", intent_heading);
    const std::size_t traversal = input_path.find(
        "scratch.traversal_input =", safe_stop);
    const std::size_t applied_velocity = input_path.find(
        "state.desired_velocity =", traversal);
    const std::size_t applied_heading = input_path.find(
        "state.desired_rotation = heading;", applied_velocity);
    const std::size_t prediction = input_path.find(
        "g1_runner_build_prediction(", applied_heading);
    const std::size_t intent_ready = input_path.find(
        "scratch.requested_intent_ready = true;", prediction);
    check(command != std::string::npos && heading != std::string::npos &&
              override_guard != std::string::npos &&
              override_assign != std::string::npos &&
              route_heading_guard != std::string::npos &&
              override_protection != std::string::npos &&
              intent_velocity != std::string::npos &&
              intent_heading != std::string::npos &&
              safe_stop != std::string::npos &&
              traversal != std::string::npos &&
              applied_velocity != std::string::npos &&
              applied_heading != std::string::npos &&
              prediction != std::string::npos &&
              intent_ready != std::string::npos &&
              command < heading && heading < override_guard &&
              override_guard < override_assign &&
              override_assign < route_heading_guard &&
              route_heading_guard < override_protection &&
              override_protection < intent_velocity &&
              intent_velocity < intent_heading && intent_heading < safe_stop &&
              safe_stop < traversal && traversal < applied_velocity &&
              applied_velocity < applied_heading &&
              applied_heading < prediction && prediction < intent_ready,
          "input stage captures independent intent before the velocity handoff and snapshot");
    const std::string heading_call = source_call_text(
        input_path,
        "g1_runner_desired_heading",
        heading,
        "input stage has the current heading-selection call");
    check(source_call_argument_count(heading_call) == 6 &&
              heading_call.find("scratch.commanded_velocity") !=
                  std::string::npos &&
              heading_call.find("scratch.traversal_input") ==
                  std::string::npos &&
              heading_call.find("state.desired_velocity") ==
                  std::string::npos,
          "heading selection consumes requested travel, not limited travel");
    const std::string safe_stop_call = source_call_text(
        input_path,
        "g1_ik_safe_stop_handoff",
        safe_stop,
        "input stage has the checked safe-stop velocity handoff");
    check(source_call_argument_count(safe_stop_call) == 5 &&
              safe_stop_call.find("scratch.commanded_velocity") !=
                  std::string::npos &&
              safe_stop_call.find("heading") == std::string::npos &&
              safe_stop_call.find("rotation") == std::string::npos,
          "velocity handoff receives no heading owner");
    const std::string prediction_call = source_call_text(
        input_path,
        "g1_runner_build_prediction",
        prediction,
        "input stage builds one immutable command snapshot");
    check(source_call_argument_count(prediction_call) == 4 &&
              prediction_call.find("scratch.requested_intent") !=
                  std::string::npos &&
              prediction_call.find("state.desired_velocity") !=
                  std::string::npos &&
              prediction_call.find("external") != std::string::npos,
          "snapshot builder receives independent requested intent and applied velocity");

    const std::string matcher_path = source.substr(
        matcher_stage, inertialization_stage - matcher_stage);
    check(matcher_path.find("g1_runner_build_query(") != std::string::npos &&
              matcher_path.find("scratch.terrain_query") !=
                  std::string::npos &&
              matcher_path.find("state.desired_rotation =") ==
                  std::string::npos &&
              matcher_path.find("requested_intent.desired_heading =") ==
                  std::string::npos &&
              matcher_path.find("heading =") == std::string::npos,
          "terrain query and matching cannot rotate the independent heading owner");

    const std::size_t prediction_helper = source.find(
        "static void g1_runner_build_prediction(");
    const std::size_t query_helper = source.find(
        "static void g1_runner_build_query(", prediction_helper);
    check(prediction_helper != std::string::npos &&
              query_helper != std::string::npos &&
              prediction_helper < query_helper,
          "command snapshot helper precedes the query helper");
    const std::string prediction_path = source.substr(
        prediction_helper, query_helper - prediction_helper);
    check(prediction_path.find(
              "state.command.predicted_desired_velocities[sample]") !=
                  std::string::npos &&
              prediction_path.find(
                  "state.command.predicted_root_positions[sample]") !=
                  std::string::npos &&
              prediction_path.find(
                  "state.command.predicted_root_rotations[sample]") !=
                  std::string::npos &&
              prediction_path.find(
                  "state.command.predicted_desired_headings[sample]") !=
                  std::string::npos &&
              count_occurrences(
                  prediction_path, "state.command.intent = intent;") == 1 &&
              count_occurrences(
                  prediction_path,
                  "state.command.applied_velocity = applied_velocity;") == 1,
          "snapshot helper fills every immutable command owner exactly once");
    check(input_path.find("accepted_state") == std::string::npos &&
              input_path.find("runtime.publication") == std::string::npos,
          "input stage can mutate only the transaction working state and scratch");

    const std::size_t transaction = source.find(
        "g1_frame_transaction_run(");
    const std::string transaction_call = source_call_text(
        source,
        "g1_frame_transaction_run",
        transaction,
        "controller invokes the frame transaction coordinator");
    const std::size_t runtime_argument = transaction_call.find(
        "frame_runtime");
    const std::size_t runner_argument = transaction_call.find(
        "::g1_controller_frame_stage_run", runtime_argument);
    const std::size_t provider_argument = transaction_call.find(
        "::g1_recovery_candidates_build", runner_argument);
    const std::size_t external_argument = transaction_call.find(
        "frame_external", provider_argument);
    const std::size_t seam_guard = transaction_call.find(
        "#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)",
        external_argument);
    const std::size_t seam_argument = transaction_call.find(
        "test_seam_pointer", seam_guard);
    const std::size_t seam_end = transaction_call.find(
        "#endif", seam_argument);
    const std::size_t error_argument = transaction_call.find(
        "artifact_error", seam_end);
    const std::size_t capacity_argument = transaction_call.find(
        "static_cast<int>(sizeof(artifact_error))", error_argument);
    check(source_call_argument_count(transaction_call) == 7 &&
              runtime_argument < runner_argument &&
              runner_argument < provider_argument &&
              provider_argument < external_argument &&
              external_argument < seam_guard &&
              seam_guard < seam_argument && seam_argument < seam_end &&
              seam_end < error_argument &&
              error_argument < capacity_argument &&
              count_occurrences(
                  transaction_call, "test_seam_pointer") == 1 &&
              source.find("frame_runtime.accepted_state.command =") ==
                  std::string::npos,
          "controller publishes command state only through the exact provider-expanded transaction coordinator");

    const char* transaction_override =
        std::getenv("G1_FRAME_TRANSACTION_SOURCE");
    const char* transaction_path = transaction_override != NULL
        ? transaction_override
        : "g1_frame_transaction.h";
    const std::string transaction_source = read_source(
        transaction_path, "frame transaction source opens and reads");
    const std::size_t candidate_validator = transaction_source.find(
        "static inline bool g1_frame_success_candidates_are_valid(");
    const std::size_t coordinator = transaction_source.find(
        "static inline G1FrameTransactionStatus g1_frame_transaction_run(",
        candidate_validator);
    check(candidate_validator != std::string::npos &&
              coordinator != std::string::npos &&
              candidate_validator < coordinator,
          "transaction validates success candidates before coordinating publication");
    const std::string candidate_path = transaction_source.substr(
        candidate_validator, coordinator - candidate_validator);
    check(candidate_path.find(
              "!g1_controller_state_is_valid(working_state)") !=
                  std::string::npos &&
              candidate_path.find("!scratch.requested_intent_ready") !=
                  std::string::npos &&
              candidate_path.find(
                  "!g1_frame_intent_is_valid(scratch.requested_intent)") !=
                  std::string::npos &&
              candidate_path.find(
                  "publication_candidate.requested_intent =") !=
                  std::string::npos &&
              candidate_path.find("scratch.requested_intent;") !=
                  std::string::npos,
          "transaction validates the complete working snapshot and ready intent");
    const std::size_t coordinator_end = transaction_source.find(
        "static inline bool g1_frame_runtime_reset(", coordinator);
    check(coordinator_end != std::string::npos,
          "transaction coordinator has a bounded source region");
    const std::string coordinator_path = transaction_source.substr(
        coordinator, coordinator_end - coordinator);
    const std::size_t success_validation = coordinator_path.find(
        "g1_frame_success_candidates_are_valid(");
    const std::size_t accepted_swap = coordinator_path.find(
        "g1_controller_state_swap(", success_validation);
    const std::size_t diagnostic_publication = coordinator_path.find(
        "runtime.accepted_diagnostic =\n"
        "                accepted_diagnostic_candidate;",
        accepted_swap);
    const std::size_t intent_publication = coordinator_path.find(
        "runtime.publication = publication_candidate;",
        diagnostic_publication);
    check(success_validation != std::string::npos &&
              accepted_swap != std::string::npos &&
              diagnostic_publication != std::string::npos &&
              intent_publication != std::string::npos &&
              success_validation < accepted_swap &&
              accepted_swap < diagnostic_publication &&
              diagnostic_publication < intent_publication,
          "validated command snapshot publishes with one accepted-state swap");
    const std::string swap_call = source_call_text(
        coordinator_path,
        "g1_controller_state_swap",
        accepted_swap,
        "transaction coordinator swaps the accepted and working states");
    check(source_call_argument_count(swap_call) == 2 &&
              swap_call.find("runtime.accepted_state") !=
                  std::string::npos &&
              swap_call.find("runtime.working_state") !=
                  std::string::npos,
          "transaction publishes the immutable command snapshot atomically");
}

static void test_failed_model_load_reaches_counted_shared_cleanup()
{
    const std::string source = read_controller_source();
    const std::size_t loader = source.find("struct G1ModelLoader");
    const std::size_t unloader = source.find(
        "struct G1ModelUnloader", loader);
    const std::size_t reset_context = source.find(
        "struct G1SceneResetContext", unloader);
    check(loader != std::string::npos && unloader != std::string::npos &&
              reset_context != std::string::npos &&
              loader < unloader && unloader < reset_context,
          "controller defines typed counted model ownership helpers");
    const std::string loader_path = source.substr(loader, unloader - loader);
    const std::size_t allocation = loader_path.find(
        "const bool allocated =");
    const std::size_t load_count = loader_path.find(
        "if (allocated) ++*load_count;", allocation);
    const std::size_t readiness = loader_path.find(
        "const bool ready =", load_count);
    const std::size_t result = loader_path.find(
        "return scene_model_load_result{allocated, ready};", readiness);
    check(allocation != std::string::npos &&
              load_count != std::string::npos &&
              readiness != std::string::npos &&
              result != std::string::npos &&
              allocation < load_count && load_count < readiness &&
              readiness < result,
          "typed loader counts every allocation before reporting readiness");
    const std::string unloader_path = source.substr(
        unloader, reset_context - unloader);
    const std::size_t unload_guard = unloader_path.find(
        "if (allocated)");
    const std::size_t unload_call = unloader_path.find(
        "UnloadModel(model);", unload_guard);
    const std::size_t unload_count = unloader_path.find(
        "++*unload_count;", unload_call);
    const std::size_t clear_model = unloader_path.find(
        "model = Model{};", unload_count);
    check(unload_guard != std::string::npos &&
              unload_call != std::string::npos &&
              unload_count != std::string::npos &&
              clear_model != std::string::npos &&
              unload_guard < unload_call && unload_call < unload_count &&
              unload_count < clear_model,
          "typed unloader releases every partial allocation exactly once");

    const std::size_t window = source.find("SetTargetFPS(25);");
    const std::size_t counter = source.find(
        "int model_load_count = 0;", window);
    const std::size_t terrain_model = source.find(
        "Model terrain_model = {};", counter);
    const std::size_t deterministic_log = source.find(
        "motion_match_log deterministic_log;", terrain_model);
    const std::size_t cleanup_lambda = source.find(
        "auto normal_cleanup =", deterministic_log);
    const std::size_t initial_load = source.find(
        "const scene_model_load_result initial_model =", cleanup_lambda);
    const std::size_t model_failure = source.find(
        "if (!initial_model.ready || !initial_model.allocated)", initial_load);
    check(window != std::string::npos && counter != std::string::npos &&
              terrain_model != std::string::npos &&
              deterministic_log != std::string::npos &&
              cleanup_lambda != std::string::npos &&
              initial_load != std::string::npos &&
              model_failure != std::string::npos &&
              window < counter && counter < terrain_model &&
              terrain_model < deterministic_log &&
              deterministic_log < cleanup_lambda &&
              cleanup_lambda < initial_load && initial_load < model_failure,
          "shared cleanup exists before the initial model readiness branch");

    const std::string post_window = source.substr(window);
    check(count_occurrences(post_window, "::CloseWindow();") == 1 &&
              count_occurrences(
                  post_window, "model_unloader(terrain_model);") == 1 &&
              post_window.find("UnloadModel(terrain_model)") ==
                  std::string::npos,
          "post-window lifetime has one counted model unload and one window close");
    const std::size_t cleanup_end = source.find("    };", cleanup_lambda);
    check(cleanup_end != std::string::npos && cleanup_end < initial_load,
          "shared cleanup lambda has a bounded body before model loading");
    const std::string cleanup_path = source.substr(
        cleanup_lambda, cleanup_end - cleanup_lambda);
    const std::size_t cleanup_guard = cleanup_path.find(
        "if (cleanup_complete) return;");
    const std::size_t log_close = cleanup_path.find(
        "deterministic_log.close(", cleanup_guard);
    const std::size_t unload = cleanup_path.find(
        "model_unloader(terrain_model);", log_close);
    const std::size_t close = cleanup_path.find(
        "::CloseWindow();", unload);
    const std::size_t cleanup = cleanup_path.find(
        "cleanup_report_write(", close);
    check(cleanup_guard != std::string::npos &&
              log_close != std::string::npos && unload != std::string::npos &&
              close != std::string::npos && cleanup != std::string::npos &&
              cleanup_guard < log_close && log_close < unload &&
              unload < close && close < cleanup,
          "shared cleanup closes log, unloads model, closes window, and reports once");
    for (const char* field : {
             "cleanup.exit_code = controller_exit_code;",
             "cleanup.motion_pack_load_count = motion_pack_load_count;",
             "cleanup.model_load_count = model_load_count;",
             "cleanup.model_unload_count = model_unload_count;",
             "cleanup.log_closed = log_closed;",
             "cleanup.window_closed = window_closed;"}) {
        check(cleanup_path.find(field, close) != std::string::npos,
              "cleanup report consumes final typed ownership state");
    }

    const std::size_t log_open = source.find(
        "if (!deterministic_log.open(", model_failure);
    check(log_open != std::string::npos,
          "deterministic log opens after a ready owned model");
    const std::string model_failure_path = source.substr(
        model_failure, log_open - model_failure);
    check(model_failure_path.find("controller_exit_code = 2;") !=
                  std::string::npos &&
              model_failure_path.find("normal_cleanup();") !=
                  std::string::npos &&
              model_failure_path.find("return controller_exit_code;") !=
                  std::string::npos &&
              model_failure_path.find("UnloadModel(") == std::string::npos &&
              model_failure_path.find("CloseWindow(") == std::string::npos,
          "allocated-but-not-ready startup routes through shared counted cleanup");
    const std::size_t log_failure_end = source.find(
        "std::vector<const char*> source_names;", log_open);
    check(log_failure_end != std::string::npos,
          "log startup failure has a bounded path");
    const std::string log_failure_path = source.substr(
        log_open, log_failure_end - log_open);
    check(log_failure_path.find("controller_exit_code = 2;") !=
                  std::string::npos &&
              log_failure_path.find("normal_cleanup();") !=
                  std::string::npos &&
              log_failure_path.find("return controller_exit_code;") !=
                  std::string::npos &&
              log_failure_path.find("G1ModelUnloader cleanup_model") ==
                  std::string::npos &&
              log_failure_path.find("CloseWindow(") == std::string::npos,
          "unopened or failed log also routes through shared cleanup");

    const std::size_t normal_cleanup = source.rfind("normal_cleanup();");
    const std::size_t final_return = source.find(
        "return controller_exit_code;", normal_cleanup);
    check(normal_cleanup != std::string::npos &&
              final_return != std::string::npos &&
              normal_cleanup < final_return &&
              count_occurrences(post_window, "normal_cleanup();") == 3,
          "model failure, log failure, and normal exit share one cleanup operation");
}

static void test_live_loop_failure_exit_codes_are_dataflow_complete()
{
    const std::string source = read_controller_source();
    const std::size_t loop_begin = source.find(
        "while (!::WindowShouldClose() && !controller_exit_requested)");
    const std::size_t cleanup = source.find(
        "normal_cleanup();", loop_begin);
    check(loop_begin != std::string::npos &&
              cleanup != std::string::npos &&
              loop_begin < cleanup,
          "controller live loop is bounded by shared cleanup");
    const std::string live_loop = source.substr(
        loop_begin, cleanup - loop_begin);

    const char* failure_markers[] = {
        "if (!scene_reset_ok)",
        "if (transaction_failed)",
        "if (!log_row_ok)",
        "if (!log_suffix_ok)",
        "if (!log_ok)",
        "if (!candidate_audit_ok)"
    };
    for (const char* marker : failure_markers) {
        const std::string branch = source_braced_statement_text(
            live_loop, marker, 0,
            "live-loop failure branch owns a braced statement");
        const std::size_t diagnostic = branch.find(
            "::controlled_runtime_error(");
        const std::size_t exit_code = branch.find(
            "controller_exit_code = 2;");
        const std::size_t request = branch.find(
            "controller_exit_requested = true;");
        const std::size_t stop = branch.find("break;");
        check(diagnostic != std::string::npos &&
                  exit_code != std::string::npos &&
                  request != std::string::npos &&
                  stop != std::string::npos &&
                  diagnostic < exit_code &&
                  exit_code < request && request < stop &&
                  count_occurrences(
                      branch, "controller_exit_code = 2;") == 1,
              "each live-loop failure publishes code two before exit request");
    }
    check(count_occurrences(
              live_loop, "controller_exit_code = 2;") == 6,
          "exactly six live-loop failure paths own code two");

    const std::string frame_limit = source_braced_statement_text(
        live_loop,
        "if (test_config.mode != G1_TestLive &&",
        0,
        "normal frame-limit exit owns a braced statement");
    check(frame_limit.find("controller_exit_requested = true;") !=
                  std::string::npos &&
              frame_limit.find("controller_exit_code") ==
                  std::string::npos &&
              live_loop.find("!::WindowShouldClose()") !=
                  std::string::npos,
          "frame-limit and window-close exits preserve the initial zero code");
}

static void test_controller_marks_no_route_cursor_inactive_after_resets()
{
    const std::string source = read_controller_source();
    const std::string transaction_source = read_source(
        "g1_frame_transaction.h",
        "controller reset test opens the typed frame transaction source");

    const std::size_t config = source.find(
        "G1FrameResetConfig reset_config_builder;");
    const std::size_t initial_reset = source.find(
        "if (!::g1_frame_runtime_reset(", config);
    const std::size_t window = source.find("::SetConfigFlags(", initial_reset);
    check(config != std::string::npos &&
              initial_reset != std::string::npos &&
              window != std::string::npos &&
              config < initial_reset && initial_reset < window,
          "controller builds typed reset configuration before startup reset and window ownership");
    const std::string startup_path = source.substr(config, window - config);
    const std::size_t route_mode = startup_path.find(
        "reset_config_builder.route_mode = test_config.mode == G1_TestRoute;");
    const std::size_t route_id = startup_path.find(
        "reset_config_builder.route_id = reset_config_builder.route_mode",
        route_mode);
    const std::size_t immutable_config = startup_path.find(
        "const G1FrameResetConfig frame_reset_config = reset_config_builder;",
        route_id);
    check(route_mode != std::string::npos && route_id != std::string::npos &&
              immutable_config != std::string::npos &&
              route_mode < route_id && route_id < immutable_config,
          "controller lowers route mode and ID into one immutable typed reset configuration");
    const std::string startup_reset_call = source_call_text(
        source,
        "g1_frame_runtime_reset",
        initial_reset,
        "controller has the typed startup frame reset call");
    check(source_call_argument_count(startup_reset_call) == 7 &&
              startup_reset_call.find("frame_runtime") != std::string::npos &&
              startup_reset_call.find("frame_reset_config") !=
                  std::string::npos,
          "startup reset consumes the immutable typed route configuration");

    const std::size_t reset_helper = source.find(
        "static bool g1_apply_pending_scene_reset(");
    const std::size_t main = source.find("int main(", reset_helper);
    check(reset_helper != std::string::npos && main != std::string::npos,
          "controller defines the typed scene reset coordinator");
    const std::string reset_path = source.substr(
        reset_helper, main - reset_helper);
    const std::size_t current_reset = reset_path.find(
        "scene_reset_current(");
    const std::size_t scene_switch = reset_path.find(
        "scene_switch_transaction(", current_reset);
    check(current_reset != std::string::npos &&
              scene_switch != std::string::npos &&
              current_reset < scene_switch &&
              count_occurrences(
                  reset_path, "*scene_reset_context.reset_config") == 2,
          "current-scene reset and scene switch consume the same typed cursor owner");
    const std::size_t context_config = source.find(
        "scene_reset_context.reset_config = &frame_reset_config;", window);
    check(context_config != std::string::npos,
          "runtime scene transactions retain the immutable reset configuration");

    const std::size_t typed_reset = transaction_source.find(
        "static inline bool g1_frame_runtime_reset(");
    check(typed_reset != std::string::npos,
          "typed frame transaction owns runtime reset");
    const std::string typed_path = transaction_source.substr(typed_reset);
    const std::size_t resolved_index = typed_path.find(
        "int route_index = -1;");
    static const char five_state_owners[] =
        "g1_controller_state* candidate_states[5] = {\n"
        "        &candidate.accepted_state,\n"
        "        &candidate.working_state,\n"
        "        &candidate.candidates.common_state,\n"
        "        &candidate.candidates.raw_state,\n"
        "        &candidate.candidates.ik_state,\n"
        "    };";
    const std::size_t state_owners = typed_path.find(
        five_state_owners, resolved_index);
    const std::size_t route_branch = typed_path.find(
        "if (config.route_mode)", state_owners);
    const std::size_t inactive_branch = typed_path.find(
        "} else {", route_branch);
    check(resolved_index != std::string::npos &&
              state_owners != std::string::npos &&
              route_branch != std::string::npos &&
              inactive_branch != std::string::npos &&
              resolved_index < state_owners &&
              state_owners < route_branch && route_branch < inactive_branch,
          "typed reset resolves route mode and binds all five state owners before its explicit inactive branch");
    const std::string routed_path = typed_path.substr(
        route_branch, inactive_branch - route_branch);
    check(routed_path.find(
              "for (int state_index = 0; state_index < 5; "
              "++state_index) {") != std::string::npos,
          "typed route reset iterates all five independent state owners");
    for (const char* assignment : {
             "candidate_states[state_index]->route_index = route_index;",
             "candidate_states[state_index]->route_waypoint = 1;",
             "candidate_states[state_index]->route_frames = 0;"}) {
        check(routed_path.find(assignment) != std::string::npos,
              "typed route reset publishes the resolved cursor to all five states");
    }
    const std::size_t inactive_end = typed_path.find(
        "    }", inactive_branch + 1U);
    check(inactive_end != std::string::npos,
          "typed inactive cursor branch has a bounded body");
    const std::string inactive_path = typed_path.substr(
        inactive_branch, inactive_end - inactive_branch);
    check(inactive_path.find(
              "for (int state_index = 0; state_index < 5; "
              "++state_index) {") != std::string::npos,
          "typed inactive reset iterates all five independent state owners");
    for (const char* assignment : {
             "candidate_states[state_index]->route_index = -1;",
             "candidate_states[state_index]->route_waypoint = 0;",
             "candidate_states[state_index]->route_frames = 0;"}) {
        check(inactive_path.find(assignment) != std::string::npos,
              "typed non-route reset publishes the canonical inactive cursor to all five states");
    }
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
    const std::size_t input_stage = controller_source.find(
        "case G1FrameStageInputRouteCommand:");
    const std::size_t matcher_stage = controller_source.find(
        "case G1FrameStageMatcherSearch:", input_stage);
    check(input_stage != std::string::npos &&
              matcher_stage != std::string::npos &&
              input_stage < matcher_stage,
          "controller defines the bounded transactional input stage");
    const std::string input_path = controller_source.substr(
        input_stage, matcher_stage - input_stage);
    const std::size_t scene_gate = input_path.find(
        "if (state.scene_frame == 0)");
    const std::size_t sample_loop = input_path.find(
        "sample < G1CommandTrajectorySampleCount", scene_gate);
    const std::size_t seed = input_path.find(
        "state.trajectory_desired_velocities(sample) =", sample_loop);
    const std::size_t desired_velocity = input_path.find(
        "state.desired_velocity;", seed);
    check(scene_gate != std::string::npos &&
              sample_loop != std::string::npos &&
              seed != std::string::npos &&
              desired_velocity != std::string::npos &&
              scene_gate < sample_loop && sample_loop < seed &&
              seed < desired_velocity,
          "transactional input seeds every desired trajectory sample from the first scene frame");
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

static void make_database(database& db, int frames = 2)
{
    static const int parents[G1_BoneCount] = {
        -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
    };
    db.bone_positions.resize(frames, G1_BoneCount);
    db.bone_velocities.resize(frames, G1_BoneCount);
    db.bone_rotations.resize(frames, G1_BoneCount);
    db.bone_angular_velocities.resize(frames, G1_BoneCount);
    db.contact_states.resize(frames, 2);
    db.bone_parents.resize(G1_BoneCount);
    db.range_starts.resize(1); db.range_stops.resize(1);
    db.range_starts(0)=0; db.range_stops(0)=frames;
    for (int frame = 0; frame < frames; ++frame) {
        for (int bone = 0; bone < G1_BoneCount; ++bone) {
            db.bone_positions(frame, bone) = vec3();
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
        db.bone_positions(frame, G1_Simulation) =
            vec3(0.25f + static_cast<float>(frame), 0.50f, -0.75f);
        db.bone_positions(frame, G1_Hips) = vec3(0.0f, 0.60f, 0.0f);
        db.bone_positions(frame, G1_LeftHipYaw) = vec3(-0.10f, 0.0f, 0.0f);
        db.bone_positions(frame, G1_RightHipYaw) = vec3(0.10f, 0.0f, 0.0f);
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
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        db.bone_parents(bone) = parents[bone];
    }
    db.contact_states.zero();
    db.contact_states(0, 0) = true;
    db.contact_states(frames - 1, 1) = true;
}

static scene_pack make_scene()
{
    scene_pack scene;
    scene.metadata.id="fixture";
    scene.metadata.spawn_position=vec3(10.65625f,7.75f,20.390625f);
    scene.metadata.spawn_yaw=0.65f;
    scene.metadata.playable_bounds={10.0f,20.0f,12.0f,22.0f};
    scene.terrain.version=2; scene.terrain.nx=33; scene.terrain.nz=33;
    scene.terrain.origin_x=10.0f; scene.terrain.origin_z=20.0f;
    scene.terrain.cell_size=0.0625f; scene.terrain.exterior_height=-17.0f;
    scene.terrain.heights.resize(33 * 33);
    scene.terrain.heights.set(4.0f);
    scene.walkability.nx=33; scene.walkability.nz=33;
    scene.walkability.cells.resize(33 * 33); scene.walkability.cells.set(1);
    return scene;
}

static void make_support(terrain_support_set& support)
{
    support.values.resize(2, 3);
    support.values(0, 0) = -2.0f;
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
    poison_array(state.ik_bone_positions);
    poison_array(state.ik_bone_rotations);
    poison_array(state.ik_global_bone_positions);
    poison_array(state.ik_global_bone_rotations);
    poison_array(state.ik_candidate_bone_positions);
    poison_array(state.ik_candidate_bone_rotations);
    poison_array(state.ik_candidate_global_bone_positions);
    poison_array(state.ik_candidate_global_bone_rotations);
    state.footprint_status = G1FootprintOutsideDomain;
    state.footprint = G1FootprintObservation{};
    state.footprint.blocked = true;
    state.footprint.work.sweeps = 777U;
    state.ik = G1IkState{};
    state.ik.feet[0].swing.previous_sphere_centers[0] =
        vec3(85.0f, 86.0f, 87.0f);
    state.ik_frame = G1IkFrameResult{};
    state.ik_frame.applied = true;
    state.ik_frame.safe_stop_requested = true;
    state.ik_frame.stop_reason = G1IkStopNoSwingCandidate;
    state.ik_frame.root_reach.active = true;
    state.ik_frame.root_reach.common_interval_found = true;
    state.ik_frame.root_reach.applied = true;
    state.ik_frame.root_reach.root_y_delta_m = 0.03125f;
    state.ik_frame.feet[0].target.desired_sole_normal =
        vec3(82.0f, 83.0f, 84.0f);
    state.ik_clearance = G1PoseClearance{};
    state.ik_clearance.minimum.lower_bound_m = 88.0;
    state.ik_candidate_clearance = G1PoseClearance{};
    state.ik_candidate_clearance.minimum.lower_bound_m = 89.0;
    state.ik_candidate_clearance_status = G1ClearanceInvalidField;
    state.ik_candidate_rejected = true;

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
    const float bounds_midpoint_x =
        0.5f * (scene.metadata.playable_bounds.min_x +
                scene.metadata.playable_bounds.max_x);
    const float bounds_midpoint_z =
        0.5f * (scene.metadata.playable_bounds.min_z +
                scene.metadata.playable_bounds.max_z);
    check(runtime_height == 4.0f, "fixture has the expected v2 surface sample");
    check(spawn.x != bounds_midpoint_x && spawn.z != bounds_midpoint_z,
          "fixture metadata XZ differs from both bounds midpoint coordinates");
    heightfield sampler_fixture;
    sampler_fixture.version = 2;
    sampler_fixture.nx = 2;
    sampler_fixture.nz = 2;
    sampler_fixture.origin_x = 10.0f;
    sampler_fixture.origin_z = 20.0f;
    sampler_fixture.cell_size = 2.0f;
    sampler_fixture.exterior_height = -17.0f;
    sampler_fixture.heights.resize(4);
    sampler_fixture.heights(0) = 1.0f;
    sampler_fixture.heights(1) = 5.0f;
    sampler_fixture.heights(2) = 9.0f;
    sampler_fixture.heights(3) = 21.0f;
    check(heightfield_sample_v2(
              sampler_fixture, 10.5f, 20.25f) !=
              heightfield_sample(
                  sampler_fixture, 10.5f, 20.25f),
          "independent fixture distinguishes v2 triangular from legacy");
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
    array1d<vec3> expected_globals(G1_BoneCount);
    array1d<quat> expected_global_rotations(G1_BoneCount);
    check(g1_ik_checked_forward_kinematics(
              expected_globals,
              expected_global_rotations,
              state.adjusted_bone_positions,
              state.adjusted_bone_rotations,
              db.bone_parents,
              error,
              static_cast<int>(sizeof(error))),
          error);
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        check(same_vec3(
                  state.global_bone_positions(bone),
                  expected_globals(bone)) &&
              same_quat(
                  state.global_bone_rotations(bone),
                  expected_global_rotations(bone)) &&
              same_vec3(
                  state.ik_bone_positions(bone),
                  state.adjusted_bone_positions(bone)) &&
              same_quat(
                  state.ik_bone_rotations(bone),
                  state.adjusted_bone_rotations(bone)) &&
              same_vec3(
                  state.ik_global_bone_positions(bone),
                  expected_globals(bone)) &&
              same_quat(
                  state.ik_global_bone_rotations(bone),
                  expected_global_rotations(bone)) &&
              same_vec3(
                  state.ik_candidate_bone_positions(bone),
                  state.adjusted_bone_positions(bone)) &&
              same_quat(
                  state.ik_candidate_bone_rotations(bone),
                  state.adjusted_bone_rotations(bone)) &&
              same_vec3(
                  state.ik_candidate_global_bone_positions(bone),
                  expected_globals(bone)) &&
              same_quat(
                  state.ik_candidate_global_bone_rotations(bone),
                  expected_global_rotations(bone)),
              "reset owns checked support/IK local and global poses");
    }
    check_vec_array(state.global_bone_velocities, G1_BoneCount, vec3(),
                    "global FK velocities reset");
    check_vec_array(state.global_bone_angular_velocities,
                    G1_BoneCount, vec3(),
                    "global FK angular velocities reset");
    check_false_array(state.global_bone_computed, G1_BoneCount,
                      "global FK computed flags reset");
    check(state.footprint_status == G1FootprintOk &&
              !state.footprint.blocked && state.ik.initialized &&
              !state.ik_frame.applied &&
              !state.ik_frame.safe_stop_requested &&
              state.ik_frame.stop_reason == G1IkStopNone &&
              g1_root_reach_plan_is_valid(
                  state.ik_frame.root_reach) &&
              !state.ik_frame.root_reach.active &&
              state.ik_candidate_clearance_status == G1ClearanceOk &&
              !state.ik_candidate_rejected &&
              state.ik_clearance.minimum.lower_bound_m >= -0.01 &&
              state.ik_candidate_clearance.minimum.lower_bound_m >= -0.01 &&
              g1_controller_state_is_valid(state),
          "reset independently certifies footprint, IK, clearance, and state");
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
    const uint64_t prior_logical_digest = state_logical_digest(active);
    const uint64_t prior_storage_digest =
        state_storage_identity_digest(active);

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
    check(state_logical_digest(active) == prior_logical_digest &&
              state_storage_identity_digest(active) == prior_storage_digest,
          "late reset failure preserves every logical owner and identity");
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

static void test_task6_state_ownership_contract_compiles()
{
    g1_controller_state state;
    state.footprint_status = G1FootprintInvalidInput;
    state.footprint = G1FootprintObservation{};
    state.ik = G1IkState{};
    state.ik_frame = G1IkFrameResult{};
    state.ik_bone_positions.resize(G1_BoneCount);
    state.ik_bone_rotations.resize(G1_BoneCount);
    state.ik_global_bone_positions.resize(G1_BoneCount);
    state.ik_global_bone_rotations.resize(G1_BoneCount);
    state.ik_candidate_bone_positions.resize(G1_BoneCount);
    state.ik_candidate_bone_rotations.resize(G1_BoneCount);
    state.ik_candidate_global_bone_positions.resize(G1_BoneCount);
    state.ik_candidate_global_bone_rotations.resize(G1_BoneCount);
    state.ik_clearance = G1PoseClearance{};
    state.ik_candidate_clearance = G1PoseClearance{};
    state.ik_candidate_clearance_status = G1ClearanceInvalidInput;
    state.ik_candidate_rejected = false;

    g1_controller_state copy;
    char error[64] = {};
    check(!g1_controller_state_copy(
              copy, state, error, static_cast<int>(sizeof(error))),
          "shape-incomplete state copy is rejected transactionally");
}

static uint64_t state_storage_identity_digest(
    const g1_controller_state& state)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    const auto add = [&hash](const void* data, int size) {
        const uintptr_t pointer = reinterpret_cast<uintptr_t>(data);
        hash ^= static_cast<uint64_t>(pointer);
        hash *= UINT64_C(1099511628211);
        hash ^= static_cast<uint64_t>(size);
        hash *= UINT64_C(1099511628211);
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
    ADD_ARRAY(adjusted_bone_rotations);
    ADD_ARRAY(global_bone_positions);
    ADD_ARRAY(global_bone_velocities);
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
    return hash;
}

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

static void logical_hash_command(
    uint64_t& hash, const G1CommandSnapshot& value)
{
    logical_hash_value(hash, value.intent.requested_velocity);
    logical_hash_value(hash, value.intent.desired_heading);
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
                hash,
                value.feet[foot].swing.previous_sphere_centers[probe]);
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

static void poison_array_against(
    array1d<vec3>& output,
    const array1d<vec3>& input,
    int salt)
{
    for (int index = 0; index < output.size; ++index) {
        const float offset = static_cast<float>(salt + index + 1);
        output(index) = input(index) + vec3(offset, offset + 1.0f, offset + 2.0f);
    }
}

static void poison_array_against(
    array1d<quat>& output,
    const array1d<quat>& input,
    int)
{
    for (int index = 0; index < output.size; ++index) {
        output(index) = quat(
            -input(index).w,
            -input(index).x,
            -input(index).y,
            -input(index).z);
    }
}

static void poison_array_against(
    array1d<bool>& output,
    const array1d<bool>& input,
    int)
{
    for (int index = 0; index < output.size; ++index) {
        output(index) = !input(index);
    }
}

static void poison_array_against(
    array1d<int>& output,
    const array1d<int>& input,
    int salt)
{
    for (int index = 0; index < output.size; ++index) {
        output(index) = input(index) + 1000 + salt + index;
    }
}

static void poison_all_state_values_against(
    g1_controller_state& output,
    const g1_controller_state& input)
{
    int salt = 1;
#define POISON_ARRAY(name) \
    poison_array_against(output.name, input.name, salt++)
    POISON_ARRAY(curr_bone_positions);
    POISON_ARRAY(curr_bone_velocities);
    POISON_ARRAY(trns_bone_positions);
    POISON_ARRAY(trns_bone_velocities);
    POISON_ARRAY(curr_bone_rotations);
    POISON_ARRAY(trns_bone_rotations);
    POISON_ARRAY(curr_bone_angular_velocities);
    POISON_ARRAY(trns_bone_angular_velocities);
    POISON_ARRAY(curr_bone_contacts);
    POISON_ARRAY(trns_bone_contacts);
    POISON_ARRAY(bone_positions);
    POISON_ARRAY(bone_velocities);
    POISON_ARRAY(bone_angular_velocities);
    POISON_ARRAY(bone_rotations);
    POISON_ARRAY(bone_offset_positions);
    POISON_ARRAY(bone_offset_velocities);
    POISON_ARRAY(bone_offset_angular_velocities);
    POISON_ARRAY(bone_offset_rotations);
    POISON_ARRAY(adjusted_bone_positions);
    POISON_ARRAY(global_bone_positions);
    POISON_ARRAY(global_bone_velocities);
    POISON_ARRAY(adjusted_bone_rotations);
    POISON_ARRAY(global_bone_rotations);
    POISON_ARRAY(global_bone_angular_velocities);
    POISON_ARRAY(global_bone_computed);
    POISON_ARRAY(ik_bone_positions);
    POISON_ARRAY(ik_bone_rotations);
    POISON_ARRAY(ik_global_bone_positions);
    POISON_ARRAY(ik_global_bone_rotations);
    POISON_ARRAY(ik_candidate_bone_positions);
    POISON_ARRAY(ik_candidate_bone_rotations);
    POISON_ARRAY(ik_candidate_global_bone_positions);
    POISON_ARRAY(ik_candidate_global_bone_rotations);
    POISON_ARRAY(trajectory_desired_velocities);
    POISON_ARRAY(trajectory_positions);
    POISON_ARRAY(trajectory_velocities);
    POISON_ARRAY(trajectory_accelerations);
    POISON_ARRAY(trajectory_angular_velocities);
    POISON_ARRAY(trajectory_desired_rotations);
    POISON_ARRAY(trajectory_rotations);
    POISON_ARRAY(contact_bones);
    POISON_ARRAY(contact_states);
    POISON_ARRAY(contact_locks);
    POISON_ARRAY(contact_positions);
    POISON_ARRAY(contact_velocities);
    POISON_ARRAY(contact_points);
    POISON_ARRAY(contact_targets);
    POISON_ARRAY(contact_offset_positions);
    POISON_ARRAY(contact_offset_velocities);
#undef POISON_ARRAY

    output.frame_index = -101;
    output.scene_frame = -102;
    output.search_time = -103.0f;
    output.search_timer = -104.0f;
    output.force_search_timer = -105.0f;
    output.transition_src_position = vec3(106.0f, 107.0f, 108.0f);
    output.transition_dst_position = vec3(109.0f, 110.0f, 111.0f);
    output.transition_src_rotation = quat(112.0f, 113.0f, 114.0f, 115.0f);
    output.transition_dst_rotation = quat(116.0f, 117.0f, 118.0f, 119.0f);
    output.desired_velocity = vec3(120.0f, 121.0f, 122.0f);
    output.desired_velocity_change_curr = vec3(123.0f, 124.0f, 125.0f);
    output.desired_velocity_change_prev = vec3(126.0f, 127.0f, 128.0f);
    output.desired_rotation = quat(129.0f, 130.0f, 131.0f, 132.0f);
    output.desired_rotation_change_curr = vec3(133.0f, 134.0f, 135.0f);
    output.desired_rotation_change_prev = vec3(136.0f, 137.0f, 138.0f);
    output.desired_gait = 139.0f;
    output.desired_gait_velocity = 140.0f;
    output.simulation_position = vec3(141.0f, 142.0f, 143.0f);
    output.simulation_velocity = vec3(144.0f, 145.0f, 146.0f);
    output.simulation_acceleration = vec3(147.0f, 148.0f, 149.0f);
    output.simulation_rotation = quat(150.0f, 151.0f, 152.0f, 153.0f);
    output.simulation_angular_velocity = vec3(154.0f, 155.0f, 156.0f);
    output.command = G1CommandSnapshot{};
    output.command.intent.requested_velocity = vec3(157.0f, 158.0f, 159.0f);
    output.command.intent.desired_heading = quat(160.0f, 161.0f, 162.0f, 163.0f);
    output.command.applied_velocity = vec3(164.0f, 165.0f, 166.0f);
    for (int sample = 0; sample < G1CommandTrajectorySampleCount; ++sample) {
        const float base = 170.0f + static_cast<float>(sample) * 16.0f;
        output.command.predicted_desired_velocities[sample] =
            vec3(base, base + 1.0f, base + 2.0f);
        output.command.predicted_root_positions[sample] =
            vec3(base + 3.0f, base + 4.0f, base + 5.0f);
        output.command.predicted_root_rotations[sample] =
            quat(base + 6.0f, base + 7.0f, base + 8.0f, base + 9.0f);
        output.command.predicted_desired_headings[sample] =
            quat(base + 10.0f, base + 11.0f, base + 12.0f, base + 13.0f);
    }
    output.footprint_status = G1FootprintInvalidField;
    output.footprint = G1FootprintObservation{};
    output.footprint.blocked = true;
    output.ik = G1IkState{};
    output.ik_frame = G1IkFrameResult{};
    output.ik_frame.applied = true;
    output.ik_frame.root_reach.active = true;
    output.ik_frame.root_reach.common_interval_found = true;
    output.ik_frame.root_reach.applied = true;
    output.ik_frame.root_reach.root_y_delta_m = 0.03125f;
    output.ik_frame.feet[0].target.desired_sole_normal =
        vec3(222.0f, 223.0f, 224.0f);
    output.ik_clearance = G1PoseClearance{};
    output.ik_candidate_clearance = G1PoseClearance{};
    output.ik_candidate_clearance.minimum.lower_bound_m = 1.0;
    output.ik_candidate_clearance_status = G1ClearanceInvalidField;
    output.ik_candidate_rejected = true;
    output.support = support_frame_state{};
    output.support.height = 201.0f;
    output.support_observation_now = support_observation{};
    output.support_observation_now.delta[0] = 202.0f;
    output.traversal_speed_scale = 203.0f;
    output.traversal_speed_scale_velocity = 204.0f;
    output.blocked = !input.blocked;
    output.walkability_class = input.walkability_class + 10;
    output.blocked_distance = 205.0f;
    output.blocked_point = vec3(206.0f, 207.0f, 208.0f);
    output.route_index = -209;
    output.route_waypoint = -210;
    output.route_frames = -211;
    output.camera_azimuth = 212.0f;
    output.camera_altitude = 213.0f;
    output.camera_distance = 214.0f;
    output.searched = !input.searched;
    output.transitioned = !input.transitioned;
    output.incumbent_cost = 215.0f;
    output.selected_cost = 216.0f;
    output.selected_terrain_error = 217.0f;
    output.adjustment_xz = 218.0f;
    output.adjustment_y = 219.0f;
    output.clamp_xz = 220.0f;
    output.clamp_y = 221.0f;
}

static void test_swap_owns_complete_state_and_storage()
{
    database db;
    make_database(db);
    terrain_support_set support;
    make_support(support);
    scene_pack scene = make_scene();
    g1_controller_state first;
    g1_controller_state second;
    char error[512] = {};
    check(g1_controller_state_reset(
              first, db, support, scene,
              error, static_cast<int>(sizeof(error))) &&
              g1_controller_state_reset(
                  second, db, support, scene,
                  error, static_cast<int>(sizeof(error))),
          error);
    poison_all_state_values_against(second, first);
    const uint64_t first_values = state_logical_digest(first);
    const uint64_t second_values = state_logical_digest(second);
    const uint64_t first_storage = state_storage_identity_digest(first);
    const uint64_t second_storage = state_storage_identity_digest(second);
    check(first_values != second_values && first_storage != second_storage,
          "swap fixtures have distinct recursive values and storage");
    g1_controller_state_swap(first, second);
    check(state_logical_digest(first) == second_values &&
              state_logical_digest(second) == first_values &&
              state_storage_identity_digest(first) == second_storage &&
              state_storage_identity_digest(second) == first_storage,
          "manual swap exchanges every logical owner and all buffer identities");
}

template<class T>
static bool same_array_bytes(
    const array1d<T>& first, const array1d<T>& second)
{
    return first.size == second.size && first.data != second.data &&
           std::memcmp(
               first.data,
               second.data,
               static_cast<size_t>(first.size) * sizeof(T)) == 0;
}

static bool copied_array_payloads_match(
    const g1_controller_state& first,
    const g1_controller_state& second)
{
#define SAME_ARRAY(name) \
    if (!same_array_bytes(first.name, second.name)) return false
    SAME_ARRAY(curr_bone_positions);
    SAME_ARRAY(curr_bone_velocities);
    SAME_ARRAY(trns_bone_positions);
    SAME_ARRAY(trns_bone_velocities);
    SAME_ARRAY(curr_bone_rotations);
    SAME_ARRAY(trns_bone_rotations);
    SAME_ARRAY(curr_bone_angular_velocities);
    SAME_ARRAY(trns_bone_angular_velocities);
    SAME_ARRAY(curr_bone_contacts);
    SAME_ARRAY(trns_bone_contacts);
    SAME_ARRAY(bone_positions);
    SAME_ARRAY(bone_velocities);
    SAME_ARRAY(bone_angular_velocities);
    SAME_ARRAY(bone_rotations);
    SAME_ARRAY(bone_offset_positions);
    SAME_ARRAY(bone_offset_velocities);
    SAME_ARRAY(bone_offset_angular_velocities);
    SAME_ARRAY(bone_offset_rotations);
    SAME_ARRAY(adjusted_bone_positions);
    SAME_ARRAY(adjusted_bone_rotations);
    SAME_ARRAY(global_bone_positions);
    SAME_ARRAY(global_bone_velocities);
    SAME_ARRAY(global_bone_rotations);
    SAME_ARRAY(global_bone_angular_velocities);
    SAME_ARRAY(global_bone_computed);
    SAME_ARRAY(trajectory_desired_velocities);
    SAME_ARRAY(trajectory_positions);
    SAME_ARRAY(trajectory_velocities);
    SAME_ARRAY(trajectory_accelerations);
    SAME_ARRAY(trajectory_angular_velocities);
    SAME_ARRAY(trajectory_desired_rotations);
    SAME_ARRAY(trajectory_rotations);
    SAME_ARRAY(contact_bones);
    SAME_ARRAY(contact_states);
    SAME_ARRAY(contact_locks);
    SAME_ARRAY(contact_positions);
    SAME_ARRAY(contact_velocities);
    SAME_ARRAY(contact_points);
    SAME_ARRAY(contact_targets);
    SAME_ARRAY(contact_offset_positions);
    SAME_ARRAY(contact_offset_velocities);
    SAME_ARRAY(ik_bone_positions);
    SAME_ARRAY(ik_bone_rotations);
    SAME_ARRAY(ik_global_bone_positions);
    SAME_ARRAY(ik_global_bone_rotations);
    SAME_ARRAY(ik_candidate_bone_positions);
    SAME_ARRAY(ik_candidate_bone_rotations);
    SAME_ARRAY(ik_candidate_global_bone_positions);
    SAME_ARRAY(ik_candidate_global_bone_rotations);
#undef SAME_ARRAY
    return true;
}

static void test_checked_copy_overwrites_dirty_values_without_reallocating()
{
    database db;
    make_database(db);
    terrain_support_set support;
    make_support(support);
    scene_pack scene = make_scene();
    g1_controller_state source;
    g1_controller_state destination;
    char error[512] = {};
    check(g1_controller_state_reset(
              source, db, support, scene,
              error, static_cast<int>(sizeof(error))) &&
          g1_controller_state_reset(
              destination, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          error);

    source.scene_frame = 17;
    source.route_frames = 19;
    source.camera_azimuth += 0.125f;
    source.contact_points(0).x += 0.25f;
    source.support_observation_now.contact[0] =
        source.curr_bone_contacts(0);
    source.support_observation_now.contact[1] =
        source.curr_bone_contacts(1);

    check(g1_controller_state_is_valid(source),
          "distinct checked-copy source remains semantically valid");
    const uint64_t source_digest = state_logical_digest(source);
    poison_all_state_values_against(destination, source);
    check(state_logical_digest(destination) != source_digest,
          "every destination logical owner is distinct before copy");
    const uint64_t identities_before =
        state_storage_identity_digest(destination);

    check(g1_controller_state_copy(
              destination, source,
              error, static_cast<int>(sizeof(error))),
          error);
    check(state_storage_identity_digest(destination) == identities_before,
          "checked copy retains every destination buffer identity");
    check(copied_array_payloads_match(destination, source),
          "checked copy copies every owning array byte into disjoint storage");
    check(state_logical_digest(destination) == source_digest,
          "checked copy overwrites every recursive logical state owner");
    check(destination.frame_index == source.frame_index &&
              destination.scene_frame == source.scene_frame &&
              destination.route_frames == source.route_frames &&
              same_float_bits(
                  destination.camera_azimuth, source.camera_azimuth) &&
              same_command_snapshot_bits(
                  destination.command, source.command) &&
              destination.ik_candidate_clearance_status ==
                  source.ik_candidate_clearance_status &&
              destination.ik_candidate_rejected ==
                  source.ik_candidate_rejected,
          "checked copy overwrites dirty scalar and value semantics exactly");

    const uint64_t destination_before_failure =
        state_logical_digest(destination);
    const uint64_t source_before_failure = state_logical_digest(source);
    vec3* const destination_buffer = destination.bone_positions.data;
    destination.bone_positions.data = source.bone_positions.data;
    check(!g1_controller_state_copy(
              destination, source,
              error, static_cast<int>(sizeof(error))),
          "source/destination alias is rejected before the first write");
    destination.bone_positions.data = destination_buffer;
    check(state_logical_digest(destination) == destination_before_failure &&
              state_logical_digest(source) == source_before_failure,
          "alias rejection performs no partial scalar or array write");

    source.search_time = std::numeric_limits<float>::quiet_NaN();
    const uint64_t invalid_source_digest = state_logical_digest(source);
    check(!g1_controller_state_copy(
              destination, source,
              error, static_cast<int>(sizeof(error))),
          "invalid source semantics are rejected before overwrite");
    check(state_logical_digest(destination) == destination_before_failure &&
              state_logical_digest(source) == invalid_source_digest,
          "invalid source rejection preserves destination bytes");
}

static void test_state_semantic_and_diagnostic_alias_preflight()
{
    database db;
    make_database(db);
    terrain_support_set support;
    make_support(support);
    scene_pack scene = make_scene();
    g1_controller_state source;
    g1_controller_state destination;
    char error[512] = {};
    check(g1_controller_state_reset(
              source, db, support, scene,
              error, static_cast<int>(sizeof(error))) &&
          g1_controller_state_reset(
              destination, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          error);

    const float original_probe = source.footprint.feet[0]
        .probes[0].current_sphere_center.x;
    source.footprint.feet[0].probes[0].current_sphere_center.x += 0.01f;
    check(!g1_controller_state_is_valid(source),
          "nested footprint geometry participates in source semantics");
    source.footprint.feet[0].probes[0].current_sphere_center.x =
        original_probe;

    int landing_foot = -1;
    int nonlanding_foot = -1;
    for (int foot = 0; foot < 2; ++foot) {
        if (source.footprint.feet[foot].landing_expected) {
            landing_foot = foot;
        } else {
            nonlanding_foot = foot;
        }
    }
    check(landing_foot >= 0 && nonlanding_foot >= 0,
          "fixture contains landing and canonical nonlanding observations");
    G1FootprintObservation footprint_before = source.footprint;
    source.footprint.feet[landing_foot].probes[0]
        .selected_landing_surface.height =
            std::numeric_limits<float>::quiet_NaN();
    check(!g1_controller_state_is_valid(source),
          "accepted landing selected-surface NaN is rejected");
    source.footprint = footprint_before;

    source.footprint.feet[landing_foot].probes[0]
        .predicted_surface_status[0] = G1SurfaceQueryOutside;
    check(!g1_controller_state_is_valid(source),
          "accepted footprint requires every predicted surface to be valid");
    source.footprint = footprint_before;

    source.footprint.feet[landing_foot].probes[0]
        .selected_landing_surface.height += 0.01f;
    check(!g1_controller_state_is_valid(source),
          "selected landing surface matches the selected predicted sample");
    source.footprint = footprint_before;

    source.footprint.feet[landing_foot].corridor_maximum_height += 0.01f;
    check(!g1_controller_state_is_valid(source),
          "accepted footprint aggregate corridor extrema are coherent");
    source.footprint = footprint_before;

    source.footprint.feet[nonlanding_foot]
        .predicted_landing_sole_center.x = 0.25f;
    check(!g1_controller_state_is_valid(source),
          "unavailable landing diagnostics remain canonical");
    source.footprint = footprint_before;

    source.ik_frame.feet[0].target.sole_center.x =
        std::numeric_limits<float>::quiet_NaN();
    check(!g1_controller_state_is_valid(source),
          "nested accepted IK-frame fields participate in source semantics");
    source.ik_frame = G1IkFrameResult{};

    source.ik_global_bone_positions(G1_Hips).x += 0.01f;
    check(!g1_controller_state_is_valid(source),
          "accepted IK global positions are bound to checked local-pose FK");
    source.ik_global_bone_positions =
        destination.ik_global_bone_positions;

    source.ik_candidate_bone_rotations(G1_LeftKnee) =
        quat_from_angle_axis(0.01f, vec3(1.0f, 0.0f, 0.0f));
    check(!g1_controller_state_is_valid(source),
          "accepted and candidate IK poses are exact coherent owners");
    source.ik_candidate_bone_rotations =
        destination.ik_candidate_bone_rotations;

    source.ik_clearance.minimum.witness.segment_parameter = -0.25;
    check(!g1_controller_state_is_valid(source),
          "clearance witness domains participate in source semantics");
    source.ik_clearance = destination.ik_clearance;

    source.ik_clearance.minimum.witness.body_x =
        std::numeric_limits<double>::quiet_NaN();
    check(!g1_controller_state_is_valid(source),
          "NaN clearance witnesses are rejected under every caller mode");
    source.ik_clearance = destination.ik_clearance;

    source.ik_clearance.minimum.witness.terrain_weight_0 =
        std::numeric_limits<double>::infinity();
    check(!g1_controller_state_is_valid(source),
          "infinite clearance weights are rejected under every caller mode");
    source.ik_clearance = destination.ik_clearance;

    source.ik_candidate_clearance.minimum.work.point_queries =
        g1_pose_clearance_budget().maximum_point_queries + 1U;
    check(!g1_controller_state_is_valid(source),
          "clearance work budgets participate in source semantics");
    source.ik_candidate_clearance = destination.ik_candidate_clearance;

    source.trajectory_positions(0).x += 0.01f;
    check(!g1_controller_state_is_valid(source),
          "legacy trajectory positions match the command snapshot");
    source.trajectory_positions = destination.trajectory_positions;

    source.desired_velocity.x += 0.01f;
    check(!g1_controller_state_is_valid(source),
          "accepted desired velocity matches command applied velocity");
    source.desired_velocity = destination.desired_velocity;

    source.desired_rotation =
        quat_from_angle_axis(0.01f, vec3(0.0f, 1.0f, 0.0f));
    check(!g1_controller_state_is_valid(source),
          "accepted desired rotation matches command desired heading");
    source.desired_rotation = destination.desired_rotation;

    source.adjustment_y =
        std::nextafter(0.0f, std::numeric_limits<float>::infinity());
    check(!g1_controller_state_is_valid(source),
          "accepted vertical adjustment diagnostic is exact positive zero");
    source.adjustment_y = 0.0f;
    source.clamp_y =
        std::nextafter(0.0f, std::numeric_limits<float>::infinity());
    check(!g1_controller_state_is_valid(source),
          "accepted vertical clamp diagnostic is exact positive zero");
    source.clamp_y = 0.0f;

    const int destination_scene_frame = destination.scene_frame;
    source.search_time = std::numeric_limits<float>::quiet_NaN();
    check(!g1_controller_state_copy(
              destination,
              source,
              reinterpret_cast<char*>(&destination.scene_frame),
              static_cast<int>(sizeof(destination.scene_frame))),
          "diagnostic alias into destination object is rejected silently");
    check(destination.scene_frame == destination_scene_frame,
          "destination object diagnostic alias cannot mutate state");

    const float destination_bone_x = destination.bone_positions(0).x;
    check(!g1_controller_state_copy(
              destination,
              source,
              reinterpret_cast<char*>(&destination.bone_positions(0).x),
              static_cast<int>(sizeof(float))),
          "diagnostic alias into destination array is rejected silently");
    check(same_float_bits(
              destination.bone_positions(0).x,
              destination_bone_x),
          "destination array diagnostic alias cannot mutate state");

    const float source_bone_x = source.bone_positions(0).x;
    check(!g1_controller_state_copy(
              destination,
              source,
              reinterpret_cast<char*>(&source.bone_positions(0).x),
              static_cast<int>(sizeof(float))),
          "diagnostic alias into invalid source array is rejected silently");
    check(same_float_bits(source.bone_positions(0).x, source_bone_x),
          "source array diagnostic alias cannot mutate state");

    source.search_time = 0.10f;
    vec3* const destination_positions = destination.bone_positions.data;
    destination.bone_positions.data =
        reinterpret_cast<vec3*>(&destination);
    check(!g1_controller_state_storage_is_valid(destination),
          "an owned array may not overlap its controller-state object");
    destination.bone_positions.data = destination_positions;

    bool* const destination_contacts = destination.contact_states.data;
    const int source_frame_before = source.frame_index;
    const int destination_frame_before = destination.frame_index;
    const uint64_t source_logical_before = state_logical_digest(source);
    const uint64_t destination_logical_before =
        state_logical_digest(destination);
    const uint64_t source_identity_before =
        state_storage_identity_digest(source);
    const uint64_t destination_identity_before =
        state_storage_identity_digest(destination);
    destination.contact_states.data = reinterpret_cast<bool*>(&source);
    check(!g1_controller_state_copy(
              destination,
              source,
              error,
              static_cast<int>(sizeof(error))),
          "destination array may not overlap the source state object");
    destination.contact_states.data = destination_contacts;
    check(source.frame_index == source_frame_before &&
              destination.frame_index == destination_frame_before &&
              state_logical_digest(source) == source_logical_before &&
              state_logical_digest(destination) ==
                  destination_logical_before &&
              state_storage_identity_digest(source) ==
                  source_identity_before &&
              state_storage_identity_digest(destination) ==
                  destination_identity_before,
          "cross-object destination alias rejects before any state write");

    bool* const source_computed = source.global_bone_computed.data;
    source.global_bone_computed.data =
        reinterpret_cast<bool*>(&destination);
    check(!g1_controller_state_copy(
              destination,
              source,
              error,
              static_cast<int>(sizeof(error))),
          "source array may not overlap the destination state object");
    source.global_bone_computed.data = source_computed;
    check(source.frame_index == source_frame_before &&
              destination.frame_index == destination_frame_before &&
              state_logical_digest(source) == source_logical_before &&
              state_logical_digest(destination) ==
                  destination_logical_before &&
              state_storage_identity_digest(source) ==
                  source_identity_before &&
              state_storage_identity_digest(destination) ==
                  destination_identity_before,
          "cross-object source alias rejects before any state write");
}

static void test_bit_safe_nested_double_validation()
{
    G1SwingCandidateDiagnostic selected;
    selected.candidate_index = 0U;
    selected.lift_bits = G1SwingLiftCandidateBits[0];
    selected.materialized_command_y_bits = 0U;
    selected.clearance_status = G1ClearanceOk;
    selected.controller_constraints_passed = true;
    selected.clearance_certified = true;
    selected.lower_margin_m = 0.0;
    selected.witness_upper_margin_m = 0.0;
    check(g1_controller_state_swing_candidate_is_selected(selected, 0U),
          "finite selected-swing diagnostic fixture is valid");
    selected.witness_upper_margin_m =
        std::numeric_limits<double>::infinity();
    check(!g1_controller_state_swing_candidate_is_selected(selected, 0U),
          "infinite selected-swing witness is rejected under fast callers");
    selected.witness_upper_margin_m =
        selected.lower_margin_m +
        2.0 * G1ClearanceMaximumCertificateWidthM;
    check(!g1_controller_state_swing_candidate_is_selected(selected, 0U),
          "selected-swing certificate width retains the strict kernel bound");

    G1SwingClearanceValidation defensive;
    check(g1_controller_state_defensive_swing_is_valid(defensive, true),
          "finite defensive-swing diagnostic fixture is valid");
    defensive.witness_upper_m =
        std::numeric_limits<double>::infinity();
    check(!g1_controller_state_defensive_swing_is_valid(defensive, true),
          "infinite defensive-swing witness is rejected under fast callers");
    defensive.lower_margin_m = 0.0;
    defensive.witness_upper_m =
        2.0 * G1ClearanceMaximumCertificateWidthM;
    defensive.sweep_evaluated = true;
    check(!g1_controller_state_defensive_swing_is_valid(defensive, false),
          "defensive-swing certificate width retains the strict kernel bound");
}

static void reset_source_hash_bytes(
    uint64_t& hash, const void* data, std::size_t bytes)
{
    const unsigned char* const values =
        static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index < bytes; ++index) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }
}

template<class T>
static void reset_source_hash_array(
    uint64_t& hash, const array1d<T>& values)
{
    logical_hash_word(
        hash,
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(values.data)));
    logical_hash_value(hash, values.size);
    reset_source_hash_bytes(
        hash,
        values.data,
        static_cast<std::size_t>(values.size) * sizeof(T));
}

template<class T>
static void reset_source_hash_array(
    uint64_t& hash, const array2d<T>& values)
{
    logical_hash_word(
        hash,
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(values.data)));
    logical_hash_value(hash, values.rows);
    logical_hash_value(hash, values.cols);
    reset_source_hash_bytes(
        hash,
        values.data,
        static_cast<std::size_t>(values.rows) *
            static_cast<std::size_t>(values.cols) * sizeof(T));
}

static void reset_source_hash_string(
    uint64_t& hash, const std::string& value)
{
    logical_hash_word(
        hash,
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(value.data())));
    logical_hash_word(hash, static_cast<uint64_t>(value.size()));
    logical_hash_word(hash, static_cast<uint64_t>(value.capacity()));
    reset_source_hash_bytes(
        hash, value.c_str(), value.size() + 1U);
}

static void reset_source_hash_strings(
    uint64_t& hash, const std::vector<std::string>& values)
{
    logical_hash_word(
        hash,
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(values.data())));
    logical_hash_word(hash, static_cast<uint64_t>(values.size()));
    logical_hash_word(hash, static_cast<uint64_t>(values.capacity()));
    for (const std::string& value : values) {
        reset_source_hash_string(hash, value);
    }
}

static uint64_t reset_sources_digest(
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    reset_source_hash_array(hash, db.bone_positions);
    reset_source_hash_array(hash, db.bone_velocities);
    reset_source_hash_array(hash, db.bone_rotations);
    reset_source_hash_array(hash, db.bone_angular_velocities);
    reset_source_hash_array(hash, db.contact_states);
    reset_source_hash_array(hash, db.bone_parents);
    reset_source_hash_array(hash, db.range_starts);
    reset_source_hash_array(hash, db.range_stops);
    reset_source_hash_array(hash, db.features);
    reset_source_hash_array(hash, db.features_offset);
    reset_source_hash_array(hash, db.features_scale);
    reset_source_hash_array(hash, db.terrain_features);
    reset_source_hash_array(hash, db.bound_sm_min);
    reset_source_hash_array(hash, db.bound_sm_max);
    reset_source_hash_array(hash, db.bound_lr_min);
    reset_source_hash_array(hash, db.bound_lr_max);
    reset_source_hash_array(hash, support.values);
    logical_hash_value(hash, scene.terrain.nx);
    logical_hash_value(hash, scene.terrain.nz);
    logical_hash_value(hash, scene.terrain.origin_x);
    logical_hash_value(hash, scene.terrain.origin_z);
    logical_hash_value(hash, scene.terrain.cell_size);
    logical_hash_value(hash, scene.terrain.exterior_height);
    logical_hash_value(hash, scene.terrain.version);
    reset_source_hash_array(hash, scene.terrain.heights);
    logical_hash_value(hash, scene.walkability.nx);
    logical_hash_value(hash, scene.walkability.nz);
    reset_source_hash_array(hash, scene.walkability.cells);
    reset_source_hash_string(hash, scene.metadata.id);
    reset_source_hash_string(hash, scene.metadata.label);
    reset_source_hash_string(hash, scene.metadata.provenance_kind);
    reset_source_hash_strings(
        hash, scene.metadata.provenance_source_ids);
    reset_source_hash_string(hash, scene.metadata.coordinate_signature);
    reset_source_hash_string(hash, scene.metadata.surface_signature);
    reset_source_hash_string(hash, scene.scene_path);
    reset_source_hash_string(hash, scene.terrain_path);
    reset_source_hash_string(hash, scene.mesh_path);
    reset_source_hash_string(hash, scene.walkability_path);
    logical_hash_value(hash, scene.metadata.spawn_position);
    logical_hash_value(hash, scene.metadata.spawn_yaw);
    logical_hash_value(hash, scene.metadata.playable_bounds.min_x);
    logical_hash_value(hash, scene.metadata.playable_bounds.min_z);
    logical_hash_value(hash, scene.metadata.playable_bounds.max_x);
    logical_hash_value(hash, scene.metadata.playable_bounds.max_z);
    return hash;
}

static void test_reset_preflights_output_ownership()
{
    database db;
    make_database(db);
    db.features.resize(2, 3);
    db.features.set(42.0f);
    terrain_support_set support;
    make_support(support);
    scene_pack scene = make_scene();
    scene.metadata.id.assign(128U, 'i');
    scene.scene_path.assign(128U, 'p');
    scene.metadata.provenance_source_ids.reserve(8U);
    scene.metadata.provenance_source_ids.push_back(
        std::string(96U, 'q'));
    g1_controller_state active;
    char error[512] = {};
    check(g1_controller_state_reset(
              active, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          error);
    const uint64_t logical_before = state_logical_digest(active);
    const uint64_t identities_before =
        state_storage_identity_digest(active);
    const uint64_t sources_before =
        reset_sources_digest(db, support, scene);
    float* const features_before = db.features.data;
    const float feature_value_before = db.features(0, 0);
    const char* const scene_path_before = scene.scene_path.data();
    const std::string scene_path_value_before = scene.scene_path;
    const std::string* const provenance_before =
        scene.metadata.provenance_source_ids.data();
    const std::string provenance_value_before =
        scene.metadata.provenance_source_ids[0];
    vec3* const saved_positions = active.bone_positions.data;
    const int saved_position_count = active.bone_positions.size;

    active.bone_positions.data = db.bone_positions.data;
    active.bone_positions.size = G1_BoneCount;
    check(!g1_controller_state_reset(
              active, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          "reset rejects an output owner borrowing database backing");
    active.bone_positions.data = saved_positions;
    active.bone_positions.size = saved_position_count;
    check(state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "database-backing alias rejection preserves output and sources");

    active.bone_positions.data =
        reinterpret_cast<vec3*>(db.features.data);
    active.bone_positions.size = 1;
    check(!g1_controller_state_reset(
              active, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          "reset protects unused matcher database backing from ownership");
    active.bone_positions.data = saved_positions;
    active.bone_positions.size = saved_position_count;
    check(db.features.data == features_before &&
              same_float_bits(db.features(0, 0), feature_value_before) &&
              state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before,
          "unused database-owner alias rejection is transactional");

    active.bone_positions.data = reinterpret_cast<vec3*>(
        &scene.scene_path[0]);
    active.bone_positions.size = 1;
    check(!g1_controller_state_reset(
              active, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          "reset protects long scene string backing from ownership");
    active.bone_positions.data = saved_positions;
    active.bone_positions.size = saved_position_count;
    check(scene.scene_path.data() == scene_path_before &&
              scene.scene_path == scene_path_value_before &&
              state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before,
          "scene-string alias rejection is transactional");

    active.bone_positions.data = reinterpret_cast<vec3*>(
        scene.metadata.provenance_source_ids.data());
    active.bone_positions.size = 1;
    check(!g1_controller_state_reset(
              active, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          "reset protects nested scene vector backing from ownership");
    active.bone_positions.data = saved_positions;
    active.bone_positions.size = saved_position_count;
    check(scene.metadata.provenance_source_ids.data() ==
                  provenance_before &&
              scene.metadata.provenance_source_ids[0] ==
                  provenance_value_before &&
              state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before,
          "nested scene-vector alias rejection is transactional");

    active.bone_positions.data = reinterpret_cast<vec3*>(&db);
    active.bone_positions.size = 1;
    check(!g1_controller_state_reset(
              active, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          "reset rejects an output owner borrowing a source object");
    active.bone_positions.data = saved_positions;
    active.bone_positions.size = saved_position_count;
    check(state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "source-object alias rejection preserves output and sources");

    active.bone_positions.data = active.bone_velocities.data;
    active.bone_positions.size = active.bone_velocities.size;
    check(!g1_controller_state_reset(
              active, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          "reset rejects overlapping output-owner ranges");
    active.bone_positions.data = saved_positions;
    active.bone_positions.size = saved_position_count;
    check(state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "peer-owner alias rejection preserves output and sources");

    active.bone_positions.data = reinterpret_cast<vec3*>(&active);
    active.bone_positions.size = 1;
    check(!g1_controller_state_reset(
              active, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          "reset rejects output backing inside the state object");
    active.bone_positions.data = saved_positions;
    active.bone_positions.size = saved_position_count;
    check(state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "state-object alias rejection preserves output and sources");

    active.bone_positions.size = 0;
    check(!g1_controller_state_reset(
              active, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          "reset rejects a zero-sized owner with non-null backing");
    active.bone_positions.size = saved_position_count;
    active.bone_positions.data = NULL;
    check(!g1_controller_state_reset(
              active, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          "reset rejects a nonempty owner with null backing");
    active.bone_positions.data = saved_positions;
    check(state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "malformed owner-pair rejection preserves output and sources");
}

static void test_reset_preflights_all_source_shapes_and_diagnostics()
{
    for (int malformed = 0; malformed < 6; ++malformed) {
        database db;
        make_database(db);
        terrain_support_set support;
        make_support(support);
        scene_pack scene = make_scene();
        g1_controller_state active;
        char error[512] = {};
        check(g1_controller_state_reset(
                  active, db, support, scene,
                  error, static_cast<int>(sizeof(error))),
              error);
        active.scene_frame = 23;
        const int scene_frame_before = active.scene_frame;
        const uint64_t identities_before =
            state_storage_identity_digest(active);
        const uint64_t logical_before = state_logical_digest(active);
        if (malformed == 0) {
            db.bone_velocities.resize(1, G1_BoneCount);
        } else if (malformed == 1) {
            db.range_stops.resize(0);
        } else if (malformed == 2) {
            db.range_starts(0) = -1;
        } else if (malformed == 3) {
            db.range_stops(0) = db.nframes() + 1;
        } else if (malformed == 4) {
            db.contact_states.resize(db.nframes(), 1);
        } else {
            support.values.resize(db.nframes() - 1, 3);
        }
        check(!g1_controller_state_reset(
                  active, db, support, scene,
                  error, static_cast<int>(sizeof(error))),
              "malformed reset peer/range shape is rejected before slicing");
        check(active.scene_frame == scene_frame_before &&
                  state_storage_identity_digest(active) == identities_before &&
                  state_logical_digest(active) == logical_before,
                  "malformed reset input preserves live state and identities");
    }

    for (int malformed_partition = 0;
         malformed_partition < 5;
         ++malformed_partition) {
        database db;
        make_database(db, 4);
        terrain_support_set support;
        support.values.resize(4, 3);
        support.values.set(-2.0f);
        scene_pack scene = make_scene();
        g1_controller_state active;
        char error[512] = {};
        check(g1_controller_state_reset(
                  active, db, support, scene,
                  error, static_cast<int>(sizeof(error))),
              error);
        const uint64_t identities_before =
            state_storage_identity_digest(active);
        const uint64_t logical_before = state_logical_digest(active);
        if (malformed_partition == 0) {
            db.range_starts(0) = 1;
        } else if (malformed_partition == 1) {
            db.range_stops(0) = 3;
        } else {
            db.range_starts.resize(2);
            db.range_stops.resize(2);
            db.range_starts(0) = 0;
            db.range_stops(0) = malformed_partition == 2 ? 1 : 3;
            db.range_starts(1) = malformed_partition == 2 ? 2 : 2;
            db.range_stops(1) = 4;
            if (malformed_partition == 4) {
                db.range_starts(1) = 1;
            }
        }
        check(!g1_controller_state_reset(
                  active, db, support, scene,
                  error, static_cast<int>(sizeof(error))),
              "reset requires a sorted contiguous full frame partition");
        check(state_storage_identity_digest(active) == identities_before &&
                  state_logical_digest(active) == logical_before,
              "partition rejection preserves every live owner and identity");
    }

    database db;
    make_database(db);
    db.features.resize(2, 3);
    db.features.set(43.0f);
    terrain_support_set support;
    make_support(support);
    scene_pack scene = make_scene();
    scene.metadata.id.assign(128U, 'j');
    scene.scene_path.assign(128U, 'r');
    scene.metadata.provenance_source_ids.reserve(8U);
    scene.metadata.provenance_source_ids.push_back(
        std::string(96U, 's'));
    g1_controller_state active;
    char error[512] = {};
    check(g1_controller_state_reset(
              active, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          error);
    active.scene_frame = 29;
    support.values(0, 0) =
        std::numeric_limits<float>::quiet_NaN();
    const int scene_frame_before = active.scene_frame;
    const uint64_t identities_before =
        state_storage_identity_digest(active);
    const uint64_t logical_before = state_logical_digest(active);
    const uint64_t sources_before =
        reset_sources_digest(db, support, scene);
    check(!g1_controller_state_reset(
              active, db, support, scene,
              reinterpret_cast<char*>(&active.scene_frame),
              static_cast<int>(sizeof(active.scene_frame))),
          "reset diagnostic may not alias the live state object");
    check(active.scene_frame == scene_frame_before &&
              state_storage_identity_digest(active) == identities_before &&
              state_logical_digest(active) == logical_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "object-alias diagnostic rejection preserves the full live unit");
    const float bone_x_before = active.bone_positions(0).x;
    check(!g1_controller_state_reset(
              active, db, support, scene,
              reinterpret_cast<char*>(&active.bone_positions(0).x),
              static_cast<int>(sizeof(float))),
          "reset diagnostic may not alias any live owning array");
    check(same_float_bits(active.bone_positions(0).x, bone_x_before) &&
              state_storage_identity_digest(active) == identities_before &&
              state_logical_digest(active) == logical_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "array-alias diagnostic rejection preserves the full live unit");

    const float support_before = support.values(0, 0);
    check(!g1_controller_state_reset(
              active, db, support, scene,
              reinterpret_cast<char*>(&support.values(0, 0)),
              static_cast<int>(sizeof(float))),
          "reset diagnostic may not alias support backing");
    check(same_float_bits(support.values(0, 0), support_before) &&
              state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "support backing remains byte-identical after alias rejection");
    check(!g1_controller_state_reset(
              active, db, support, scene,
              reinterpret_cast<char*>(db.features.data), 1),
          "reset diagnostic may not alias unused database backing");
    check(state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "unused database diagnostic alias preserves all owners");
    check(!g1_controller_state_reset(
              active, db, support, scene,
              &scene.scene_path[0], 1),
          "reset diagnostic may not alias long scene string backing");
    check(state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "scene-string diagnostic alias preserves all owners");
    check(!g1_controller_state_reset(
              active, db, support, scene,
              reinterpret_cast<char*>(
                  scene.metadata.provenance_source_ids.data()),
              1),
          "reset diagnostic may not alias nested scene vector backing");
    check(state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "scene-vector diagnostic alias preserves all owners");
    const int range_before = db.range_starts(0);
    check(!g1_controller_state_reset(
              active, db, support, scene,
              reinterpret_cast<char*>(&db.range_starts(0)),
              static_cast<int>(sizeof(int))),
          "reset diagnostic may not alias database range backing");
    check(db.range_starts(0) == range_before &&
              state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "database range backing remains unchanged");
    const float terrain_before = scene.terrain.heights(0);
    check(!g1_controller_state_reset(
              active, db, support, scene,
              reinterpret_cast<char*>(&scene.terrain.heights(0)),
              static_cast<int>(sizeof(float))),
          "reset diagnostic may not alias terrain backing");
    check(same_float_bits(scene.terrain.heights(0), terrain_before) &&
              state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "terrain backing remains unchanged");
    const uint8_t walkability_before = scene.walkability.cells(0);
    check(!g1_controller_state_reset(
              active, db, support, scene,
              reinterpret_cast<char*>(&scene.walkability.cells(0)),
              static_cast<int>(sizeof(uint8_t))),
          "reset diagnostic may not alias walkability backing");
    check(scene.walkability.cells(0) == walkability_before &&
              state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "walkability backing remains unchanged");
    const char id_before = scene.metadata.id[0];
    check(!g1_controller_state_reset(
              active, db, support, scene,
              &scene.metadata.id[0], 1),
          "reset diagnostic may not alias scene identifier backing");
    check(scene.metadata.id[0] == id_before &&
              state_logical_digest(active) == logical_before &&
              state_storage_identity_digest(active) == identities_before &&
              reset_sources_digest(db, support, scene) == sources_before,
          "scene identifier backing remains unchanged");
}

static void test_footprint_validation_uses_support_retargeted_pose()
{
    database db;
    make_database(db);
    terrain_support_set support;
    make_support(support);
    scene_pack scene = make_scene();
    g1_controller_state state;
    char error[512] = {};
    check(g1_controller_state_reset(
              state, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          error);
    check(g1_controller_state_accepted_footprint_is_valid(state),
          "reset footprint is authenticated against support-retargeted FK");
    state.ik_global_bone_positions(G1_LeftAnkle).y += 0.01f;
    check(g1_controller_state_accepted_footprint_is_valid(state),
          "footprint validation is independent of final IK-owned globals");
    check(!g1_controller_state_is_valid(state),
          "independent final-pose validation still rejects incoherent IK FK");
}

static void test_ik_off_accepts_coherent_unready_landing_patch()
{
    database db;
    make_database(db);
    terrain_support_set support;
    make_support(support);
    scene_pack scene = make_scene();
    scene.walkability.cells.set(2);
    g1_controller_state state;
    char error[512] = {};
    check(g1_controller_state_reset(
              state, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          error);
    int landing_foot = -1;
    for (int foot = 0; foot < 2; ++foot) {
        if (state.footprint.feet[foot].landing_expected) {
            landing_foot = foot;
        }
    }
    check(landing_foot >= 0,
          "IK-off fixture has one observed future landing");
    check(!state.ik_frame.applied &&
              state.footprint_status == G1FootprintOk &&
              state.footprint.feet[landing_foot]
                      .predicted_landing_walkability_class == 2 &&
              !state.footprint.feet[landing_foot].landing_patch_ready &&
              g1_controller_state_accepted_footprint_is_valid(state) &&
              g1_controller_state_is_valid(state),
          "production observation admits a coherent class-2 unready landing "
          "with IK disabled");
}

static void test_pose_clearance_acceptance_threshold_boundaries()
{
    database db;
    make_database(db);
    terrain_support_set support;
    make_support(support);
    scene_pack scene = make_scene();
    g1_controller_state state;
    char error[512] = {};
    check(g1_controller_state_reset(
              state, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          error);
    const G1PoseClearance valid_clearance = state.ik_clearance;
    G1PoseClearance clearance = state.ik_clearance;
    clearance.left.toe.lower_bound_m = -0.005;
    clearance.left.foot.lower_bound_m = -0.005;
    clearance.right.toe.lower_bound_m = -0.005;
    clearance.right.foot.lower_bound_m = -0.005;
    clearance.minimum.lower_bound_m = -0.01;
    check(g1_controller_state_pose_clearance_meets_thresholds(clearance),
          "pose acceptance includes each exact lower-bound threshold");
    clearance.left.toe.lower_bound_m =
        std::nextafter(-0.005, -std::numeric_limits<double>::infinity());
    check(!g1_controller_state_pose_clearance_meets_thresholds(clearance),
          "next binary64 value below planted toe threshold is rejected");
    clearance = state.ik_clearance;
    clearance.minimum.lower_bound_m =
        std::nextafter(-0.01, -std::numeric_limits<double>::infinity());
    check(!g1_controller_state_pose_clearance_meets_thresholds(clearance),
          "next binary64 value below overall threshold is rejected");

    G1ClearanceResult first = state.ik_clearance.left.knee;
    G1ClearanceResult second = first;
    second.witness.primitive_index = first.witness.primitive_index + 1U;
    G1ClearanceResult summary = first;
    summary.work = G1ClearanceWork{};
    check(g1_controller_state_checked_add_work(summary.work, first.work) &&
              g1_controller_state_checked_add_work(
                  summary.work, second.work),
          "tie-break fixture work sum is representable");
    const G1ClearanceResult* components[] = {&first, &second};
    check(g1_controller_state_clearance_summary_is_valid(
              summary, components, 2),
          "least-key tied clearance witness is canonical");
    summary.witness = second.witness;
    check(!g1_controller_state_clearance_summary_is_valid(
              summary, components, 2),
          "larger-key tied clearance witness is rejected");

    G1PoseClearance forged = state.ik_clearance;
    check(!g1_controller_state_clearance_witness_equal(
              forged.left.thigh.witness,
              forged.left.minimum.witness),
          "primitive-role mutation fixture uses a non-minimum leaf");
    forged.left.thigh.witness.primitive_index = 9U;
    check(!g1_controller_state_pose_clearance_is_coherent(forged),
          "a globally valid but wrong primitive role is rejected");
    state.ik_clearance.left.thigh.witness.primitive_index = 9U;
    state.ik_candidate_clearance.left.thigh.witness.primitive_index = 9U;
    check(!g1_controller_state_is_valid(state),
          "accepted and candidate clearances authenticate every non-min leaf");

    forged = state.ik_candidate_clearance;
    forged.left.thigh.witness.primitive_index = 8U;
    forged.hips.witness.candidate_kind = 2U;
    check(!g1_controller_state_pose_clearance_is_valid(forged),
          "point leaves require the point-special candidate key");
    forged = state.ik_candidate_clearance;
    forged.left.thigh.witness.primitive_index = 8U;
    forged.hips.witness.patch_index = 1U;
    check(!g1_controller_state_pose_clearance_is_valid(forged),
          "point leaves require the canonical zero patch index");
    forged = state.ik_candidate_clearance;
    forged.left.thigh.witness.primitive_index = 8U;
    forged.left.foot.witness.patch_index = G1ClearancePatchesPerPair;
    check(!g1_controller_state_pose_clearance_is_valid(forged),
          "sphere leaves reject a forged patch outside the fixed patch set");
    forged = state.ik_candidate_clearance;
    forged.left.thigh.witness.primitive_index = 8U;
    forged.left.foot.witness.candidate_kind = 3U;
    check(!g1_controller_state_pose_clearance_is_valid(forged),
          "sphere leaves reject the point-special candidate kind");
    forged = valid_clearance;
    forged.left.thigh.witness.candidate_kind = 0U;
    forged.left.thigh.witness.candidate_subindex = 1U;
    check(!g1_controller_state_clearance_result_is_valid(
              forged.left.thigh),
          "kind-0 witnesses require the canonical zero subindex");
    forged.left.thigh.witness.candidate_kind = 1U;
    forged.left.thigh.witness.candidate_subindex = 9U;
    check(!g1_controller_state_clearance_result_is_valid(
              forged.left.thigh),
          "kind-1 witnesses reject the exact subindex ceiling");

    forged = valid_clearance;
    forged.left.thigh.witness.candidate_kind = 2U;
    forged.left.thigh.witness.candidate_subindex = 35U;
    check(g1_controller_state_clearance_result_is_valid(
              forged.left.thigh),
          "kind-2 immutable creation ordinals may exceed pair candidates");
    check(g1_controller_state_pose_clearance_is_coherent(forged),
          "a non-minimum kind-2 ordinal above 31 remains coherent");
    state.ik_clearance = forged;
    state.ik_candidate_clearance = forged;
    check(g1_controller_state_is_valid(state),
          "accepted state preserves a legitimate high kind-2 ordinal");
    forged.left.thigh.witness.candidate_subindex =
        g1_pose_clearance_budget().maximum_face_patches +
        g1_pose_clearance_budget().maximum_subdivision_nodes;
    check(!g1_controller_state_clearance_result_is_valid(
              forged.left.thigh),
          "kind-2 immutable creation ordinal rejects the exact ceiling");
}

static void authenticate_test_ik_pose(
    g1_controller_state& state,
    const database& db,
    const scene_pack& scene,
    char* error,
    int error_capacity)
{
    check(g1_ik_checked_forward_kinematics(
              state.ik_global_bone_positions,
              state.ik_global_bone_rotations,
              state.ik_bone_positions,
              state.ik_bone_rotations,
              db.bone_parents,
              error,
              error_capacity),
          error);
    state.ik_candidate_bone_positions = state.ik_bone_positions;
    state.ik_candidate_bone_rotations = state.ik_bone_rotations;
    state.ik_candidate_global_bone_positions =
        state.ik_global_bone_positions;
    state.ik_candidate_global_bone_rotations =
        state.ik_global_bone_rotations;
    check(g1_measure_pose_clearance(
              state.ik_clearance,
              g1_pose_clearance_budget(),
              scene.terrain,
              state.ik_global_bone_positions,
              state.ik_global_bone_rotations,
              error,
              error_capacity) == G1ClearanceOk,
          error);
    state.ik_candidate_clearance = state.ik_clearance;
    state.ik_candidate_clearance_status = G1ClearanceOk;
    state.ik_candidate_rejected = false;
}

static void test_applied_ik_result_coherence()
{
    G1FootTarget signed_zero_inactive = {};
    signed_zero_inactive.surface.normal.x = -0.0f;
    check(!g1_foot_target_is_valid(signed_zero_inactive),
          "inactive target rejects negative-zero terrain-normal bytes");

    database db;
    make_database(db);
    db.contact_states.set(true);
    terrain_support_set support;
    make_support(support);
    array1d<vec3> source_global_positions(G1_BoneCount);
    array1d<quat> source_global_rotations(G1_BoneCount);
    vec3 source_left_sole;
    vec3 source_right_sole;
    float source_relative_sole_height = 0.0f;
    float source_surface_height = 0.0f;
    char error[512] = {};
    check(g1_ik_checked_forward_kinematics(
              source_global_positions,
              source_global_rotations,
              db.bone_positions(0),
              db.bone_rotations(0),
              db.bone_parents,
              error,
              static_cast<int>(sizeof(error))) &&
              g1_ik_checked_physical_sole_centroid(
                  source_left_sole,
                  source_global_positions(G1_LeftToe),
                  source_global_rotations(G1_LeftToe),
                  g1_left_leg_config()) &&
              g1_ik_checked_physical_sole_centroid(
                  source_right_sole,
                  source_global_positions(G1_RightToe),
                  source_global_rotations(G1_RightToe),
                  g1_right_leg_config()) &&
              same_float_bits(
                  source_left_sole.y,
                  source_right_sole.y) &&
              terrain_f32_sub(
                  source_relative_sole_height,
                  source_left_sole.y,
                  db.bone_positions(0, G1_Simulation).y) &&
              terrain_f32_sub(
                  source_surface_height,
                  source_relative_sole_height,
                  g1_left_leg_config().planted_clearance_m),
          error[0] == '\0'
              ? "all-contact fixture has symmetric physical source soles"
              : error);
    support.values(0, 0) = source_surface_height;
    scene_pack scene = make_scene();
    g1_controller_state state;
    check(g1_controller_state_reset(
              state, db, support, scene,
              error, static_cast<int>(sizeof(error))),
          error);
    check(g1_ik_frame_evaluate(
              state.ik_bone_positions,
              state.ik_bone_rotations,
              state.ik,
              state.adjusted_bone_positions,
              state.adjusted_bone_rotations,
              db.bone_parents,
              state.curr_bone_contacts,
              scene.terrain,
              state.footprint,
              true,
              1.0f / 25.0f,
              state.ik_frame,
              error,
              static_cast<int>(sizeof(error))),
          error);
    check(state.ik_frame.applied &&
              !state.ik_frame.safe_stop_requested,
          "real all-contact Task-5 frame produces an applied result");
    authenticate_test_ik_pose(
        state, db, scene,
        error, static_cast<int>(sizeof(error)));
    check(g1_controller_state_is_valid(state),
          "real applied Task-5 result passes complete accepted-state gates");

    const G1IkFrameResult valid_result = state.ik_frame;
    check(g1_root_reach_plan_is_valid(valid_result.root_reach) &&
              valid_result.root_reach.active ==
                  (state.curr_bone_contacts(0) ||
                   state.curr_bone_contacts(1)) &&
              (!valid_result.root_reach.active ||
               valid_result.root_reach.common_interval_found),
          "accepted all-contact IK owns a valid common root-reach plan");
    const uint64_t valid_plan_digest = state_logical_digest(state);
    const auto reject_plan_mutation = [
        &state, &valid_result, valid_plan_digest](
        const G1IkFrameResult& forged,
        const char* message) {
        state.ik_frame = forged;
        check(state_logical_digest(state) != valid_plan_digest &&
                  !g1_controller_state_is_valid(state),
              message);
        state.ik_frame = valid_result;
    };
    {
        G1IkFrameResult forged = valid_result;
        forged.root_reach.active = !forged.root_reach.active;
        reject_plan_mutation(
            forged,
            "accepted-state digest and validator own root-plan active");
    }
    {
        G1IkFrameResult forged = valid_result;
        forged.root_reach.common_interval_found =
            !forged.root_reach.common_interval_found;
        reject_plan_mutation(
            forged,
            "accepted-state digest and validator own root-plan common interval");
    }
    {
        G1IkFrameResult forged = valid_result;
        forged.root_reach.applied = !forged.root_reach.applied;
        reject_plan_mutation(
            forged,
            "accepted-state digest and validator own root-plan application");
    }
    {
        G1IkFrameResult forged = valid_result;
        if (forged.root_reach.applied) {
            forged.root_reach.root_y_delta_m = std::nextafter(
                forged.root_reach.root_y_delta_m,
                forged.root_reach.root_y_delta_m < 0.0f
                    ? -std::numeric_limits<float>::infinity()
                    : std::numeric_limits<float>::infinity());
        } else {
            forged.root_reach.root_y_delta_m = 0.03125f;
        }
        reject_plan_mutation(
            forged,
            "accepted-state digest and validator own exact root-plan delta bits");
    }
    {
        const float original =
            state.adjusted_bone_positions(G1_Simulation).y;
        state.adjusted_bone_positions(G1_Simulation).y =
            std::nextafter(
                original,
                std::numeric_limits<float>::infinity());
        check(state_logical_digest(state) != valid_plan_digest &&
                  !g1_controller_state_is_valid(state),
              "accepted-state validator owns adjusted local Simulation Y as the root-plan baseline");
        state.adjusted_bone_positions(G1_Simulation).y = original;
        check(g1_controller_state_is_valid(state),
              "restoring adjusted local Simulation Y restores the accepted pose certificate");
    }
    {
        const float original =
            state.ik_bone_positions(G1_Simulation).y;
        state.ik_bone_positions(G1_Simulation).y =
            std::nextafter(
                original,
                std::numeric_limits<float>::infinity());
        check(state_logical_digest(state) != valid_plan_digest &&
                  !g1_controller_state_is_valid(state),
              "accepted-state validator owns accepted IK local Simulation Y as the strict plan result");
        state.ik_bone_positions(G1_Simulation).y = original;
        check(g1_controller_state_is_valid(state),
              "restoring accepted IK local Simulation Y restores the accepted pose certificate");
    }
    {
        const float original =
            state.global_bone_positions(G1_Hips).y;
        state.global_bone_positions(G1_Hips).y =
            std::nextafter(
                original,
                std::numeric_limits<float>::infinity());
        check(state_logical_digest(state) != valid_plan_digest &&
                  !g1_controller_state_is_valid(state),
              "accepted-state validator owns support-retargeted global Hips Y");
        state.global_bone_positions(G1_Hips).y = original;
        check(g1_controller_state_is_valid(state),
              "restoring support-retargeted global Hips Y restores the accepted pose certificate");
    }
    {
        const float original =
            state.ik_global_bone_positions(G1_Hips).y;
        state.ik_global_bone_positions(G1_Hips).y =
            std::nextafter(
                original,
                std::numeric_limits<float>::infinity());
        check(state_logical_digest(state) != valid_plan_digest &&
                  !g1_controller_state_is_valid(state),
              "accepted-state validator owns accepted IK global Hips Y");
        state.ik_global_bone_positions(G1_Hips).y = original;
        check(g1_controller_state_is_valid(state),
              "restoring accepted IK global Hips Y restores the accepted pose certificate");
    }
    check(g1_controller_state_vec3_bits_equal(
              state.ik_frame.feet[0].target.surface.point,
              state.ik.feet[0].lock.lock_point) &&
              g1_controller_state_vec3_bits_equal(
                  state.ik_frame.feet[0]
                      .target.desired_sole_normal,
                  state.ik_frame.feet[0].target.surface.normal),
          "contact target owns exact lock position and terrain normal");
    const uint64_t accepted_contact_digest =
        state_logical_digest(state);
    state.ik_frame.feet[0].target.desired_sole_normal.y =
        std::nextafter(1.0f, 0.0f);
    check(g1_foot_target_is_valid(
              state.ik_frame.feet[0].target),
          "one-bit contact desired-normal forgery remains structurally valid");
    const uint64_t forged_contact_digest =
        state_logical_digest(state);
    check(forged_contact_digest != accepted_contact_digest,
          "controller logical digest owns the desired sole normal");
    const uint64_t forged_contact_storage =
        state_storage_identity_digest(state);
    check(!g1_controller_state_is_valid(state) &&
              state_logical_digest(state) == forged_contact_digest &&
              state_storage_identity_digest(state) ==
                  forged_contact_storage,
          "contact desired normal authenticates to terrain without state mutation");
    state.ik_frame = valid_result;
    const vec3 accepted_baseline_normal =
        state.ik.feet[0].baseline_sole_normal;
    state.ik.feet[0].baseline_sole_normal.y =
        std::nextafter(
            accepted_baseline_normal.y,
            0.0f);
    check(g1_ik_runtime_state_is_valid(state.ik),
          "one-bit IK baseline-normal forgery remains structurally valid");
    const uint64_t forged_baseline_digest =
        state_logical_digest(state);
    const uint64_t forged_baseline_storage =
        state_storage_identity_digest(state);
    check(!g1_controller_state_is_valid(state) &&
              state_logical_digest(state) == forged_baseline_digest &&
              state_storage_identity_digest(state) ==
                  forged_baseline_storage,
          "controller authenticates immutable IK baseline normal without mutation");
    state.ik.feet[0].baseline_sole_normal =
        accepted_baseline_normal;
    const G1FootLockState accepted_contact_lock =
        state.ik.feet[0].lock;
    state.ik.feet[0].lock.output_position.x =
        std::nextafter(
            accepted_contact_lock.lock_point.x,
            std::numeric_limits<float>::infinity());
    state.ik_frame.feet[0].target.sole_center =
        state.ik.feet[0].lock.output_position;
    check(g1_ik_runtime_state_is_valid(state.ik) &&
              g1_foot_target_is_valid(
                  state.ik_frame.feet[0].target) &&
              !g1_controller_state_vec3_bits_equal(
                  state.ik_frame.feet[0].target.sole_center,
                  state.ik.feet[0].lock.lock_point) &&
              !g1_controller_state_is_valid(state),
          "accepted recorded contact rejects the stale spring output in favor of the exact lock");
    state.ik.feet[0].lock = accepted_contact_lock;
    state.ik_frame = valid_result;
    state.ik_frame.feet[0].recorded_contact = false;
    check(!g1_controller_state_is_valid(state),
          "applied IK contact roles match accepted contact bits");
    state.ik_frame = valid_result;
    state.ik_frame.feet[0].position.iterations = 5;
    check(!g1_controller_state_is_valid(state),
          "applied IK iteration count retains the Task-5 bound");
    state.ik_frame = valid_result;
    state.ik_frame.feet[0].defensive_swing.lower_margin_m = 0.001;
    check(!g1_controller_state_is_valid(state),
          "contact-foot defensive sweep diagnostics remain canonical");
    state.ik_frame = valid_result;
    state.ik_frame.feet[0]
        .swing_selection.selected.lower_margin_m = -0.0;
    check(!g1_controller_state_is_valid(state),
          "contact-foot selected diagnostics require positive-zero margins");
    state.ik_frame = valid_result;
    state.ik_frame.feet[0].orientation.requested_correction_radians =
        std::nextafter(
            state.ik_frame.feet[0].orientation.correction_radians,
            std::numeric_limits<float>::infinity());
    check(!g1_controller_state_is_valid(state),
          "unlimited orientation requested/actual corrections agree exactly");
    state.ik_frame = valid_result;
    const vec3 accepted_previous_input =
        state.ik.feet[0].lock.previous_input;
    const vec3 accepted_contact_fk = g1_ik_vec3_canonicalize(
        state.global_bone_positions(g1_left_leg_config().contact));
    state.ik.feet[0].lock.previous_input.x = std::nextafter(
        accepted_previous_input.x,
        std::numeric_limits<float>::infinity());
    check(!g1_controller_state_is_valid(state),
          "applied lock input is bound to support-retargeted physical sole FK");
    state.ik.feet[0].lock.previous_input = accepted_previous_input;
    if (g1_controller_state_vec3_bits_equal(
            accepted_previous_input, accepted_contact_fk)) {
        std::fprintf(
            stderr,
            "sole-centroid auth RED evidence: accepted lock input is Toe "
            "FK (%.9f,%.9f,%.9f)\n",
            accepted_contact_fk.x,
            accepted_contact_fk.y,
            accepted_contact_fk.z);
    }
    state.ik.feet[0].lock.previous_input = accepted_contact_fk;
    check(!g1_controller_state_is_valid(state),
          "applied endpoint authentication rejects Toe in favor of sole FK");
    state.ik.feet[0].lock.previous_input = accepted_previous_input;

    g1_controller_state unready;
    database landing_db;
    make_database(landing_db);
    check(g1_controller_state_reset(
              unready, landing_db, support, scene,
              error, static_cast<int>(sizeof(error))),
          error);
    int landing_foot = -1;
    for (int foot = 0; foot < 2; ++foot) {
        if (unready.footprint.feet[foot].landing_expected) {
            landing_foot = foot;
        }
    }
    check(landing_foot >= 0,
          "enabled-context fixture contains a future landing");
    unready.footprint.feet[landing_foot]
        .predicted_landing_walkability_class = 2;
    unready.footprint.feet[landing_foot].landing_patch_ready = false;
    check(!g1_controller_state_ik_frame_is_valid(
              valid_result,
              state.curr_bone_contacts,
              unready.footprint),
          "applied IK context rejects every expected unready landing");

    database swing_db;
    make_database(swing_db);
    swing_db.bone_rotations.set(quat());
    terrain_support_set swing_support;
    make_support(swing_support);
    swing_support.values(0, 0) = source_surface_height;
    swing_support.values(0, 1) = source_surface_height;
    swing_support.values(0, 2) = source_surface_height;
    g1_controller_state swing;
    check(g1_controller_state_reset(
              swing, swing_db, swing_support, scene,
              error, static_cast<int>(sizeof(error))),
          error);
    for (int probe = 0; probe < 4; ++probe) {
        swing.ik.feet[1].swing
            .previous_sphere_centers[probe].y += 0.20f;
    }
    check(g1_ik_frame_evaluate(
              swing.ik_bone_positions,
              swing.ik_bone_rotations,
              swing.ik,
              swing.adjusted_bone_positions,
              swing.adjusted_bone_rotations,
              swing_db.bone_parents,
              swing.curr_bone_contacts,
              scene.terrain,
              swing.footprint,
              true,
              1.0f / 25.0f,
              swing.ik_frame,
              error,
              static_cast<int>(sizeof(error))),
          error);
    check(swing.ik_frame.applied &&
              swing.curr_bone_contacts(0) &&
              !swing.curr_bone_contacts(1),
          "real mixed-contact Task-5 frame produces one selected swing");
    authenticate_test_ik_pose(
        swing, swing_db, scene,
        error, static_cast<int>(sizeof(error)));
    check(g1_controller_state_is_valid(swing),
          "real mixed-contact Task-5 result passes complete state gates");
    check(g1_root_reach_plan_is_valid(
              swing.ik_frame.root_reach) &&
              swing.ik_frame.root_reach.active &&
              swing.ik_frame.root_reach.common_interval_found,
          "accepted mixed-contact IK retains its common root-reach plan");

    const G1FootTarget accepted_target = swing.ik_frame.feet[1].target;
    vec3 accepted_swing_normal;
    check(ik_checked_quat_rotate(
              accepted_swing_normal,
              swing.global_bone_rotations(
                  g1_right_leg_config().contact),
              g1_right_leg_config().sole_normal_local) &&
              g1_controller_state_vec3_bits_equal(
                  accepted_target.sole_center,
                  swing.ik.feet[1].lock.output_position) &&
              terrain_float_bits(accepted_target.surface.point.x) ==
                  terrain_float_bits(
                      swing.ik.feet[1].lock.previous_input.x) &&
              terrain_float_bits(accepted_target.surface.point.z) ==
                  terrain_float_bits(
                      swing.ik.feet[1].lock.previous_input.z) &&
              g1_controller_state_vec3_bits_equal(
                  accepted_target.desired_sole_normal,
                  accepted_swing_normal) &&
              swing.footprint.feet[1].landing_expected &&
              swing.footprint.feet[1].landing_patch_ready,
          "swing target owns current sole provenance while landing remains lookahead");
    swing.ik_frame.feet[1].target.desired_sole_normal.y =
        std::nextafter(
            accepted_target.desired_sole_normal.y,
            0.0f);
    check(g1_foot_target_is_valid(
              swing.ik_frame.feet[1].target),
          "one-bit swing desired-normal forgery remains structurally valid");
    const uint64_t forged_swing_digest =
        state_logical_digest(swing);
    const uint64_t forged_swing_storage =
        state_storage_identity_digest(swing);
    check(!g1_controller_state_is_valid(swing) &&
              state_logical_digest(swing) == forged_swing_digest &&
              state_storage_identity_digest(swing) ==
                  forged_swing_storage,
          "swing desired normal authenticates to pre-IK sole without state mutation");
    swing.ik_frame.feet[1].target = accepted_target;

    swing.ik_frame.feet[1].target.locked = true;
    swing.ik_frame.feet[1].target.position_active = true;
    check(g1_foot_target_is_valid(swing.ik_frame.feet[1].target) &&
              !g1_controller_state_is_valid(swing),
          "applied target transition flags match the accepted lock state");
    swing.ik_frame.feet[1].target = accepted_target;

    const G1FootLockState accepted_lock = swing.ik.feet[1].lock;
    swing.ik.feet[1].lock.contact = true;
    swing.ik.feet[1].lock.locked = true;
    swing.ik.feet[1].lock.position_active = true;
    swing.ik_frame.feet[1].target.locked = true;
    swing.ik_frame.feet[1].target.position_active = true;
    check(g1_foot_lock_state_is_valid(swing.ik.feet[1].lock) &&
              g1_foot_target_is_valid(swing.ik_frame.feet[1].target) &&
              !g1_controller_state_is_valid(swing),
          "applied lock contact remains exact to the accepted contact bit");
    swing.ik.feet[1].lock = accepted_lock;
    swing.ik_frame.feet[1].target = accepted_target;

    swing.ik_frame.feet[1].target.surface.point.x = std::nextafter(
        accepted_target.surface.point.x,
        std::numeric_limits<float>::infinity());
    check(g1_foot_target_is_valid(swing.ik_frame.feet[1].target) &&
              !g1_controller_state_is_valid(swing),
          "noncontact terrain query X/Z is bound to the current sole");
    swing.ik_frame.feet[1].target = accepted_target;

    uint32_t& materialized_y_bits = swing.ik_frame.feet[1]
        .swing_selection.selected.materialized_command_y_bits;
    const uint32_t accepted_materialized_y_bits = materialized_y_bits;
    float accepted_materialized_y = 0.0f;
    std::memcpy(
        &accepted_materialized_y,
        &accepted_materialized_y_bits,
        sizeof(accepted_materialized_y));
    const float forged_materialized_y = std::nextafter(
        accepted_materialized_y,
        std::numeric_limits<float>::infinity());
    std::memcpy(
        &materialized_y_bits,
        &forged_materialized_y,
        sizeof(materialized_y_bits));
    check(!g1_controller_state_is_valid(swing),
          "selected lift materialization is bit-exact to its base target");
    materialized_y_bits = accepted_materialized_y_bits;

    const vec3 history_center =
        swing.ik.feet[1].swing.previous_sphere_centers[0];
    swing.ik.feet[1].swing.previous_sphere_centers[0].x =
        std::nextafter(
            history_center.x,
            std::numeric_limits<float>::infinity());
    check(!g1_controller_state_is_valid(swing),
          "committed swing history is bound to exact final FK centers");
    swing.ik.feet[1].swing.previous_sphere_centers[0] = history_center;

    uint32_t& endpoint_bits = swing.ik_frame.feet[1]
        .swing_selection.selected.actual_sphere_center_bits[0][0];
    const uint32_t accepted_endpoint_bits = endpoint_bits;
    float accepted_endpoint = 0.0f;
    std::memcpy(
        &accepted_endpoint,
        &accepted_endpoint_bits,
        sizeof(accepted_endpoint));
    const float forged_endpoint = std::nextafter(
        accepted_endpoint,
        std::numeric_limits<float>::infinity());
    std::memcpy(&endpoint_bits, &forged_endpoint, sizeof(endpoint_bits));
    check(!g1_controller_state_is_valid(swing),
          "selected swing diagnostics bind exact final FK endpoint bits");
    endpoint_bits = accepted_endpoint_bits;

    const double selected_upper = swing.ik_frame.feet[1]
        .swing_selection.selected.witness_upper_margin_m;
    swing.ik_frame.feet[1]
        .swing_selection.selected.witness_upper_margin_m =
        swing.ik_frame.feet[1]
                .swing_selection.selected.lower_margin_m +
            2.0 * G1ClearanceMaximumCertificateWidthM;
    check(!g1_controller_state_is_valid(swing),
          "stored selected-swing diagnostics retain certificate width");
    swing.ik_frame.feet[1]
        .swing_selection.selected.witness_upper_margin_m = selected_upper;

    const double defensive_upper =
        swing.ik_frame.feet[1].defensive_swing.witness_upper_m;
    swing.ik_frame.feet[1].defensive_swing.witness_upper_m =
        swing.ik_frame.feet[1].defensive_swing.lower_margin_m +
        2.0 * G1ClearanceMaximumCertificateWidthM;
    check(!g1_controller_state_is_valid(swing),
          "stored defensive-swing diagnostics retain certificate width");
    swing.ik_frame.feet[1].defensive_swing.witness_upper_m =
        defensive_upper;
}

int main(int argc, char** argv)
{
    if (argc == 2 &&
        std::strcmp(argv[1], "--sole-auth-contract") == 0) {
        test_applied_ik_result_coherence();
        return 0;
    }
    test_active_scene_sources_use_checked_v2_queries();
    test_controller_wires_idle_match_transition_cost();
    test_controller_validates_ik_geometry_before_window();
    test_controller_publishes_independent_travel_and_heading();
    test_failed_model_load_reaches_counted_shared_cleanup();
    test_live_loop_failure_exit_codes_are_dataflow_complete();
    test_controller_marks_no_route_cursor_inactive_after_resets();
    test_idle_match_transition_cost_policy();
    test_scene_first_frame_seeds_desired_trajectory();
    test_reset_clears_every_dynamic_subsystem();
    test_failed_reset_preserves_prior_state();
    test_swap_owns_complete_command_snapshot();
    test_swap_owns_complete_state_and_storage();
    test_task6_state_ownership_contract_compiles();
    test_checked_copy_overwrites_dirty_values_without_reallocating();
    test_state_semantic_and_diagnostic_alias_preflight();
    test_bit_safe_nested_double_validation();
    test_reset_preflights_output_ownership();
    test_reset_preflights_all_source_shapes_and_diagnostics();
    test_footprint_validation_uses_support_retargeted_pose();
    test_ik_off_accepts_coherent_unready_landing_patch();
    test_pose_clearance_acceptance_threshold_boundaries();
    test_applied_ik_result_coherence();
    return 0;
}
