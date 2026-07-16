#define main g1_candidate_audit_embedded_controller_main
#include "controller.cpp"
#undef main

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

#include <unistd.h>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(
            stderr,
            "G1 candidate audit controller test failed: %s\n",
            message);
        std::exit(1);
    }
}

static std::string fixture_path(const char* suffix)
{
    char path[256] = {};
    const int written = std::snprintf(
        path,
        sizeof(path),
        "/tmp/test_g1_candidate_audit_controller_%ld_%s",
        static_cast<long>(getpid()),
        suffix);
    check(written > 0 &&
              static_cast<std::size_t>(written) < sizeof(path),
          "fixture path fits");
    return std::string(path);
}

static std::string read_text(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    check(stream.good(), "open audit fixture output");
    return std::string(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

static void write_text(const std::string& path, const std::string& text)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    check(stream.good(), "open fixture output for writing");
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    stream.close();
    check(stream.good(), "write and close fixture output");
}

static void write_manifest_fixture(const std::string& path)
{
    std::string text =
        "{\"schema\":\"g1-terrain-artifacts/v2\","
        "\"feature_dimensions\":31,\"database_frames\":140,"
        "\"total_clips\":2,\"sources\":[";
    for (int range = 0; range < 2; ++range) {
        if (range != 0) text += ',';
        const int start = range * 70;
        text +=
            "{\"name\":\"source-" + std::to_string(range) +
            "\",\"terrain_id\":\"fixture\",\"source_fps\":25.0,"
            "\"source_frames\":70,\"output_frames\":70,"
            "\"range_start\":" + std::to_string(start) +
            ",\"range_stop\":" + std::to_string(start + 70) +
            ",\"source_frame_map\":[";
        for (int frame = 0; frame < 70; ++frame) {
            if (frame != 0) text += ',';
            text += std::to_string(frame);
        }
        text += "]}";
    }
    text += "]}\n";
    write_text(path, text);
}

static bool run_analyzer(
    const std::string& manifest,
    const std::string& audit,
    const std::string& summary)
{
    const std::string command =
        "python3 resources/audit_g1_directional_candidates.py"
        " --manifest " + manifest +
        " --audit " + audit +
        " --output " + summary +
        " >/dev/null 2>&1";
    return std::system(command.c_str()) == 0;
}

static std::size_t count_occurrences(
    const std::string& text,
    const std::string& needle)
{
    std::size_t count = 0;
    std::size_t cursor = 0;
    while ((cursor = text.find(needle, cursor)) != std::string::npos) {
        ++count;
        cursor += needle.size();
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

static void test_configuration_is_optional_absolute_and_bounded()
{
    char error[256] = "unchanged";
    G1CandidateAuditConfig config;
    check(g1_candidate_audit_config_parse(
              config,
              nullptr,
              G1_TestLive,
              0,
              nullptr,
              nullptr,
              nullptr,
              error,
              sizeof(error)),
          "missing environment variable disables the audit");
    check(!config.enabled && config.path == nullptr &&
              config.route == nullptr &&
              config.heading == nullptr &&
              std::strcmp(error, "unchanged") == 0,
          "disabled audit has one inert representation");

    check(g1_candidate_audit_config_parse(
              config,
              "/tmp/audit.jsonl",
              G1_TestRoute,
              800,
              "route",
              "curb-forward",
              "positive-x",
              error,
              sizeof(error)),
          "absolute bounded directional route enables the audit");
    check(config.enabled &&
              std::strcmp(config.path, "/tmp/audit.jsonl") == 0 &&
              std::strcmp(config.route, "curb-forward") == 0 &&
              std::strcmp(config.heading, "positive-x") == 0,
          "enabled audit retains immutable configuration pointers");

    const G1CandidateAuditConfig before = config;
    check(!g1_candidate_audit_config_parse(
              config,
              "relative.jsonl",
              G1_TestRoute,
              800,
              "route",
              "curb-forward",
              "positive-x",
              error,
              sizeof(error)) &&
              config.enabled == before.enabled &&
              config.path == before.path &&
              config.route == before.route &&
              config.heading == before.heading,
          "relative audit path is rejected transactionally");
    check(!g1_candidate_audit_config_parse(
              config,
              "/tmp/audit.jsonl",
              G1_TestLive,
              800,
              "live",
              "manual",
              "positive-x",
              error,
              sizeof(error)),
          "live candidate audit is rejected");
    check(!g1_candidate_audit_config_parse(
              config,
              "/tmp/audit.jsonl",
              G1_TestRoute,
              0,
              "route",
              "curb-forward",
              "positive-x",
              error,
              sizeof(error)),
          "unbounded candidate audit is rejected");
    check(g1_candidate_audit_config_parse(
              config,
              "/tmp/audit.jsonl",
              G1_TestSequential,
              100,
              "sequential",
              "curb-forward",
              nullptr,
              error,
              sizeof(error)),
          "every bounded deterministic mode permits candidate auditing");
    check(std::strcmp(config.route, "curb-forward") == 0 &&
              std::strcmp(config.heading, "relative") == 0,
          "non-route audit preserves a real route and labels its dynamic heading relative");
    const g1_test_mode remaining_bounded_modes[] = {
        G1_TestFlat,
        G1_TestTerrain,
    };
    const char* remaining_bounded_names[] = {
        "flat",
        "terrain",
    };
    for (std::size_t index = 0;
         index < sizeof(remaining_bounded_modes) /
             sizeof(remaining_bounded_modes[0]);
         ++index) {
        check(g1_candidate_audit_config_parse(
                  config,
                  "/tmp/audit.jsonl",
                  remaining_bounded_modes[index],
                  100,
                  remaining_bounded_names[index],
                  "curb-forward",
                  nullptr,
                  error,
                  sizeof(error)) &&
                  std::strcmp(config.route, "curb-forward") == 0 &&
                  std::strcmp(config.heading, "relative") == 0,
              "flat and terrain bounded modes preserve their real route and relative heading");
    }
    check(g1_candidate_audit_config_parse(
              config,
              "/tmp/audit.jsonl",
              G1_TestSceneCycle,
              350,
              "scene-cycle",
              "ignored-nonroute-option",
              nullptr,
              error,
              sizeof(error)),
          "bounded scene-cycle candidate auditing is permitted");
    check(std::strcmp(config.route, "scene-cycle") == 0 &&
              std::strcmp(config.heading, "relative") == 0,
          "scene-cycle uses its bounded mode name and relative heading instead of an unused route option");
    check(g1_candidate_audit_config_parse(
              config,
              "/tmp/audit.jsonl",
              G1_TestRoute,
              100,
              "route",
              "curb-forward",
              nullptr,
              error,
              sizeof(error)),
          "a bounded route without a fixed heading uses the relative cell");
    check(std::strcmp(config.route, "curb-forward") == 0 &&
              std::strcmp(config.heading, "relative") == 0,
          "route audit preserves its real route for relative heading motion");
    check(!g1_candidate_audit_config_parse(
              config,
              "/tmp/audit.jsonl",
              G1_TestRoute,
              100,
              "route",
              "curb-forward",
              "Positive-X",
              error,
              sizeof(error)),
          "candidate audit rejects a noncanonical heading spelling");
}

struct AuditFixture
{
    database db;
    g1_controller_state accepted_state;
    G1FrameAcceptedDiagnostic diagnostic;
    G1FramePublication publication;

    AuditFixture()
    {
        constexpr int frames = 140;
        db.features.resize(frames, G1CandidateAuditFeatureCount);
        db.features.set(0.0f);
        db.features_offset.resize(G1CandidateAuditFeatureCount);
        db.features_offset.set(2.0f);
        db.features_scale.resize(G1CandidateAuditFeatureCount);
        db.features_scale.set(2.0f);
        db.range_starts.resize(2);
        db.range_stops.resize(2);
        db.range_starts(0) = 0;
        db.range_stops(0) = 70;
        db.range_starts(1) = 70;
        db.range_stops(1) = frames;
        for (int frame = 0; frame < frames; ++frame) {
            db.features(frame, 0) =
                static_cast<float>((frame % 13) + 1) * 0.125f;
        }
        diagnostic.ready = true;
        diagnostic.query_database_frame = 30;
        diagnostic.query_range = 0;
        diagnostic.selected_database_frame = 5;
        for (int dimension = 0;
             dimension < G1CandidateAuditFeatureCount;
             ++dimension) {
            diagnostic.query[dimension] = 2.0f;
        }
        accepted_state.frame_index = 6;
        publication.presentation_frame = 1;
    }
};

static uint64_t feature_checksum(const database& db)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(
        db.features.data);
    const std::size_t size =
        static_cast<std::size_t>(db.features.rows) *
        static_cast<std::size_t>(db.features.cols) * sizeof(float);
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void test_sampling_serialization_and_input_inertness()
{
    const std::string path = fixture_path("records.jsonl");
    (void)unlink(path.c_str());
    char error[256] = {};
    G1CandidateAuditConfig config;
    check(g1_candidate_audit_config_parse(
              config,
              path.c_str(),
              G1_TestRoute,
              100,
              "route",
              "route\\cell",
              "positive-x",
              error,
              sizeof(error)),
          error);
    G1CandidateAuditLog log;
    check(log.open(config, error, sizeof(error)), error);

    AuditFixture fixture;
    const int accepted_frame_before = fixture.accepted_state.frame_index;
    float query_before[G1CandidateAuditFeatureCount] = {};
    std::memcpy(
        query_before,
        fixture.diagnostic.query,
        sizeof(query_before));
    const uint64_t features_before = feature_checksum(fixture.db);

    fixture.publication.presentation_frame = 24;
    check(log.write_requested(
              fixture.db,
              fixture.accepted_state,
              fixture.diagnostic,
              fixture.publication,
              G1FrameTransactionAccepted,
              "fixture\"scene",
              config.route,
              config.heading,
              error,
              sizeof(error)),
          error);
    fixture.publication.presentation_frame = 25;
    check(log.write_requested(
              fixture.db,
              fixture.accepted_state,
              fixture.diagnostic,
              fixture.publication,
              G1FrameTransactionAccepted,
              "fixture\"scene",
              config.route,
              config.heading,
              error,
              sizeof(error)),
          error);
    fixture.publication.presentation_frame = 26;
    fixture.publication.rejection.rejected = true;
    check(log.write_requested(
              fixture.db,
              fixture.accepted_state,
              fixture.diagnostic,
              fixture.publication,
              G1FrameTransactionFiniteRejected,
              "fixture\"scene",
              config.route,
              config.heading,
              error,
              sizeof(error)),
          error);
    fixture.publication.presentation_frame = 27;
    fixture.diagnostic.ready = false;
    fixture.diagnostic.query_database_frame = -1;
    check(log.write_requested(
              fixture.db,
              fixture.accepted_state,
              fixture.diagnostic,
              fixture.publication,
              G1FrameTransactionFiniteRejected,
              "fixture\"scene",
              config.route,
              config.heading,
              error,
              sizeof(error)),
          "the first finite rejection is auditable before any accepted diagnostic");
    check(log.close(error, sizeof(error)), error);

    const std::string output = read_text(path);
    check(count_occurrences(output, "\n") == 3,
          "only each 25th request and each finite rejection are audited");
    check(output.find(
              "{\"schema\":\"g1-directional-candidate-audit/v1\","
              "\"requested_frame\":25,\"query_bits_hex\":\"") == 0,
          "audit begins with the canonical schema/key order");
    check(output.find("\"scene_id\":\"fixture\\\"scene\"") !=
              std::string::npos &&
              output.find("\"route\":\"route\\\\cell\"") !=
                  std::string::npos,
          "JSON text fields are escaped canonically");
    check(output.find(
              "\"heading\":\"positive-x\",\"accepted_frame\":6,") !=
              std::string::npos,
          "audit records the fixed heading and accepted database frame");
    check(output.find("\"requested_frame\":26") != std::string::npos,
          "finite rejection is audited outside the 25-frame cadence");
    check(output.find(
              "\"requested_frame\":27,\"query_bits_hex\":\"3f600000") !=
              std::string::npos,
          "startup rejection uses the immutable accepted database row as its baseline query");
    check(count_occurrences(output, "\"query_bits_hex\":\"") == 3 &&
              count_occurrences(output, std::string(31 * 8, '0')) == 2,
          "accepted diagnostics write all normalized query words in lowercase hex");
    check(count_occurrences(output, "\"top_count\":16") == 3 &&
              count_occurrences(output, "\"cost_bits\":\"") == 48,
          "each record writes exactly the canonical top-16 entries");
    const std::string exact_accepted_counts =
        "\"eligible_count\":61,\"within_best_plus_0_25\":19,"
        "\"within_best_plus_1\":38,\"within_best_plus_4\":61,";
    const std::string exact_accepted_top =
        "\"top\":["
        "{\"frame\":0,\"range\":0,\"cost_bits\":\"3c800000\"},"
        "{\"frame\":78,\"range\":1,\"cost_bits\":\"3c800000\"},"
        "{\"frame\":91,\"range\":1,\"cost_bits\":\"3c800000\"},"
        "{\"frame\":104,\"range\":1,\"cost_bits\":\"3c800000\"},"
        "{\"frame\":117,\"range\":1,\"cost_bits\":\"3c800000\"},"
        "{\"frame\":1,\"range\":0,\"cost_bits\":\"3d800000\"},"
        "{\"frame\":79,\"range\":1,\"cost_bits\":\"3d800000\"},"
        "{\"frame\":92,\"range\":1,\"cost_bits\":\"3d800000\"},"
        "{\"frame\":105,\"range\":1,\"cost_bits\":\"3d800000\"},"
        "{\"frame\":118,\"range\":1,\"cost_bits\":\"3d800000\"},"
        "{\"frame\":2,\"range\":0,\"cost_bits\":\"3e100000\"},"
        "{\"frame\":80,\"range\":1,\"cost_bits\":\"3e100000\"},"
        "{\"frame\":93,\"range\":1,\"cost_bits\":\"3e100000\"},"
        "{\"frame\":106,\"range\":1,\"cost_bits\":\"3e100000\"},"
        "{\"frame\":119,\"range\":1,\"cost_bits\":\"3e100000\"},"
        "{\"frame\":3,\"range\":0,\"cost_bits\":\"3e800000\"}]";
    check(count_occurrences(output, exact_accepted_counts) == 2 &&
              count_occurrences(output, exact_accepted_top) == 2,
          "known scalar fixture locks every accepted count/range/cost word against plausible serializer mutations");
    check(fixture.accepted_state.frame_index == accepted_frame_before &&
              std::memcmp(
                  query_before,
                  fixture.diagnostic.query,
                  sizeof(query_before)) == 0 &&
              feature_checksum(fixture.db) == features_before,
          "candidate auditing leaves accepted state/query/database immutable");

    const std::string manifest = fixture_path("manifest.json");
    const std::string summary = fixture_path("summary.json");
    const std::string mutated = fixture_path("mutated.jsonl");
    write_manifest_fixture(manifest);
    (void)unlink(summary.c_str());
    check(run_analyzer(manifest, path, summary),
          "the real analyzer parses controller JSONL and publishes a summary");
    const std::string summary_text = read_text(summary);
    check(summary_text.find(
              "\"record_count\":3,\"schema\":"
              "\"g1-directional-candidate-summary/v1\"") !=
              std::string::npos,
          "analyzer summary accounts for every serialized controller record");

    std::string mutation = output;
    std::size_t mutation_position = mutation.find("\"range\":1");
    check(mutation_position != std::string::npos,
          "fixture contains a top candidate from its second source range");
    mutation[mutation_position + 8] = '0';
    write_text(mutated, mutation);
    check(!run_analyzer(manifest, mutated, summary),
          "analyzer rejects a serialized range/source mutation");

    mutation = output;
    mutation_position = mutation.find("\"cost_bits\":\"");
    check(mutation_position != std::string::npos,
          "fixture contains serialized cost words");
    mutation.replace(mutation_position + 13, 8, "7f800000");
    write_text(mutated, mutation);
    check(!run_analyzer(manifest, mutated, summary),
          "analyzer rejects a serialized nonfinite cost mutation");

    mutation = output;
    mutation_position = mutation.find("\"eligible_count\":");
    check(mutation_position != std::string::npos,
          "fixture contains serialized candidate counts");
    const std::size_t count_begin = mutation_position + 17;
    const std::size_t count_end = mutation.find(',', count_begin);
    check(count_end != std::string::npos,
          "serialized eligible count has a canonical delimiter");
    mutation.replace(count_begin, count_end - count_begin, "15");
    write_text(mutated, mutation);
    check(!run_analyzer(manifest, mutated, summary),
          "analyzer rejects a serialized candidate-count mutation");

    check(unlink(mutated.c_str()) == 0, "remove mutated audit output");
    check(unlink(summary.c_str()) == 0, "remove candidate summary output");
    check(unlink(manifest.c_str()) == 0, "remove candidate manifest fixture");
    check(unlink(path.c_str()) == 0, "remove candidate audit output");
}

static void test_first_requested_frame_rejection_uses_accepted_baseline()
{
    const std::string path = fixture_path("first-rejection.jsonl");
    const std::string manifest = fixture_path("first-rejection-manifest.json");
    const std::string summary = fixture_path("first-rejection-summary.json");
    (void)unlink(path.c_str());
    (void)unlink(summary.c_str());
    char error[256] = {};
    G1CandidateAuditConfig config;
    check(g1_candidate_audit_config_parse(
              config,
              path.c_str(),
              G1_TestRoute,
              100,
              "route",
              "curb-forward",
              "forward",
              error,
              sizeof(error)),
          error);
    G1CandidateAuditLog log;
    check(log.open(config, error, sizeof(error)), error);
    AuditFixture fixture;
    fixture.diagnostic = G1FrameAcceptedDiagnostic{};
    fixture.publication.presentation_frame = 0;
    fixture.publication.rejection.rejected = true;
    const uint64_t features_before = feature_checksum(fixture.db);
    check(log.write_requested(
              fixture.db,
              fixture.accepted_state,
              fixture.diagnostic,
              fixture.publication,
              G1FrameTransactionFiniteRejected,
              "first-rejection-scene",
              config.route,
              config.heading,
              error,
              sizeof(error)),
          "requested frame zero is auditable before the first accepted diagnostic");
    check(log.close(error, sizeof(error)), error);
    const std::string output = read_text(path);
    check(count_occurrences(output, "\n") == 1 &&
              output.find(
                  "\"requested_frame\":0,\"query_bits_hex\":\"3f600000") !=
                  std::string::npos &&
              output.find("\"accepted_frame\":6") != std::string::npos,
          "first rejection records the immutable accepted frame/feature baseline");
    check(!fixture.diagnostic.ready &&
              fixture.diagnostic.query_database_frame == -1 &&
              feature_checksum(fixture.db) == features_before,
          "first-rejection auditing does not synthesize accepted diagnostics or mutate the database");
    write_manifest_fixture(manifest);
    check(run_analyzer(manifest, path, summary),
          "the real analyzer accepts the first-frame rejection record");
    check(unlink(summary.c_str()) == 0,
          "remove first-rejection candidate summary output");
    check(unlink(manifest.c_str()) == 0,
          "remove first-rejection candidate manifest fixture");
    check(unlink(path.c_str()) == 0,
          "remove first-rejection candidate audit output");
}

static void test_open_write_and_close_errors_are_controlled()
{
    char error[256] = {};
    G1CandidateAuditConfig config;
    const std::string missing = fixture_path("missing/out.jsonl");
    check(g1_candidate_audit_config_parse(
              config,
              missing.c_str(),
              G1_TestRoute,
              100,
              "route",
              "curb-forward",
              "forward",
              error,
              sizeof(error)),
          error);
    G1CandidateAuditLog open_failure;
    check(!open_failure.open(config, error, sizeof(error)) &&
              open_failure.stream == nullptr,
          "audit open failure is controlled and leaves no owner");

    check(g1_candidate_audit_config_parse(
              config,
              "/dev/full",
              G1_TestRoute,
              100,
              "route",
              "curb-forward",
              "forward",
              error,
              sizeof(error)),
          error);
    G1CandidateAuditLog write_failure;
    check(write_failure.open(config, error, sizeof(error)), error);
    AuditFixture fixture;
    fixture.publication.presentation_frame = 0;
    check(!write_failure.write_requested(
              fixture.db,
              fixture.accepted_state,
              fixture.diagnostic,
              fixture.publication,
              G1FrameTransactionAccepted,
              "scene",
              config.route,
              config.heading,
              error,
              sizeof(error)),
          "audit write/flush failure is controlled");
    (void)write_failure.close(error, sizeof(error));

    const std::string close_path = fixture_path("close.jsonl");
    (void)unlink(close_path.c_str());
    check(g1_candidate_audit_config_parse(
              config,
              close_path.c_str(),
              G1_TestRoute,
              100,
              "route",
              "curb-forward",
              "forward",
              error,
              sizeof(error)),
          error);
    G1CandidateAuditLog close_failure;
    check(close_failure.open(config, error, sizeof(error)), error);
    const int descriptor = fileno(close_failure.stream);
    check(descriptor >= 0 && ::close(descriptor) == 0,
          "invalidate fixture descriptor before close");
    check(!close_failure.close(error, sizeof(error)) &&
              close_failure.stream == nullptr,
          "audit close failure is controlled and releases its owner");
    check(unlink(close_path.c_str()) == 0,
          "remove close-failure fixture output");
}

static void test_non_route_relative_record_is_analyzer_compatible()
{
    const std::string path = fixture_path("relative.jsonl");
    const std::string manifest = fixture_path("relative-manifest.json");
    const std::string summary = fixture_path("relative-summary.json");
    (void)unlink(path.c_str());
    (void)unlink(summary.c_str());
    char error[256] = {};
    G1CandidateAuditConfig config;
    check(g1_candidate_audit_config_parse(
              config,
              path.c_str(),
              G1_TestSceneCycle,
              350,
              "scene-cycle",
              "",
              nullptr,
              error,
              sizeof(error)),
          error);
    G1CandidateAuditLog log;
    check(log.open(config, error, sizeof(error)), error);
    AuditFixture fixture;
    fixture.publication.presentation_frame = 0;
    check(log.write_requested(
              fixture.db,
              fixture.accepted_state,
              fixture.diagnostic,
              fixture.publication,
              G1FrameTransactionAccepted,
              "scene-cycle-fixture",
              config.route,
              config.heading,
              error,
              sizeof(error)),
          error);
    check(log.close(error, sizeof(error)), error);
    const std::string output = read_text(path);
    check(output.find(
              "\"route\":\"scene-cycle\",\"heading\":\"relative\"") !=
              std::string::npos,
          "route-less bounded mode serializes its exact relative cell");
    write_manifest_fixture(manifest);
    check(run_analyzer(manifest, path, summary),
          "analyzer accepts the controller's explicit non-route relative cell");
    check(read_text(summary).find(
              "\"heading\":\"relative\",\"route\":\"scene-cycle\"") !=
              std::string::npos,
          "summary preserves the explicit relative heading and mode route label");
    check(unlink(summary.c_str()) == 0,
          "remove relative candidate summary output");
    check(unlink(manifest.c_str()) == 0,
          "remove relative candidate manifest fixture");
    check(unlink(path.c_str()) == 0,
          "remove relative candidate audit output");
}

static void test_production_main_owns_audit_after_publication_and_log()
{
    const std::string source = read_text("controller.cpp");
    const std::string compact = compact_source(source);
    check(count_occurrences(
              compact,
              "getenv(\"MM_CANDIDATE_AUDIT\")") == 1,
          "production parses MM_CANDIDATE_AUDIT exactly once");
    const std::string exact_parser_call =
        "g1_candidate_audit_config_parse(candidate_audit_config,"
        "candidate_audit_environment,test_config.mode,"
        "test_config.frame_limit,test_config.mode_name.c_str(),"
        "test_config.route.c_str(),test_config.test_heading.empty()?"
        "nullptr:test_config.test_heading.c_str(),artifact_error,"
        "static_cast<int>(sizeof(artifact_error)))";
    check(count_occurrences(compact, exact_parser_call) == 1,
          "production binds the one environment read to the exact immutable bounded-mode parser inputs");
    const std::string exact_const_writer_signature =
        "boolwrite_requested(constdatabase&db,"
        "constg1_controller_state&accepted_state,"
        "constG1FrameAcceptedDiagnostic&accepted_diagnostic,"
        "constG1FramePublication&publication,"
        "G1FrameTransactionStatusframe_status,constchar*scene_id,"
        "constchar*route,constchar*heading,char*error,"
        "interror_capacity)";
    check(count_occurrences(compact, exact_const_writer_signature) == 1,
          "candidate writer authenticates const database, accepted-state, diagnostic, and publication inputs");
    const std::size_t parse_arguments = compact.find(
        "g1_parse_arguments(parsed_test_config,");
    const std::size_t audit_parse = compact.find(
        "g1_candidate_audit_config_parse(candidate_audit_config,");
    const std::size_t audit_owner = compact.find(
        "G1CandidateAuditLogcandidate_audit_log;");
    const std::size_t cleanup = compact.find("autonormal_cleanup=[&]()");
    const std::size_t deterministic_close = compact.find(
        "deterministic_log.close(", cleanup);
    const std::size_t audit_close = compact.find(
        "candidate_audit_log.close(", deterministic_close);
    const std::size_t unload = compact.find(
        "model_unloader(terrain_model);", audit_close);
    const std::size_t deterministic_open = compact.find(
        "deterministic_log.open(");
    const std::size_t audit_open = compact.find(
        "candidate_audit_log.open(", deterministic_open);
    check(parse_arguments != std::string::npos &&
              audit_parse != std::string::npos &&
              audit_owner != std::string::npos &&
              cleanup != std::string::npos &&
              deterministic_close != std::string::npos &&
              audit_close != std::string::npos &&
              unload != std::string::npos &&
              deterministic_open != std::string::npos &&
              audit_open != std::string::npos &&
              parse_arguments < audit_parse && audit_parse < audit_owner &&
              audit_owner < cleanup &&
              deterministic_close < audit_close && audit_close < unload &&
              deterministic_open < audit_open,
          "audit configuration, owner, open, and close follow deterministic lifetime order");

    const std::size_t coordinator = compact.find(
        "g1_frame_transaction_run(frame_runtime,");
    const std::size_t log_materialize = compact.find(
        "g1_build_task7_log_suffix(", coordinator);
    const std::size_t log_write = compact.find(
        "deterministic_log.write(", log_materialize);
    const std::size_t audit_write = compact.find(
        "candidate_audit_log.write_requested(", log_write);
    const std::size_t camera = compact.find(
        "update_g1_camera_from_accepted(", audit_write);
    check(coordinator != std::string::npos &&
              log_materialize != std::string::npos &&
              log_write != std::string::npos &&
              audit_write != std::string::npos &&
              camera != std::string::npos &&
              coordinator < log_materialize &&
              log_materialize < log_write && log_write < audit_write &&
              audit_write < camera,
          "audit executes only after acceptance and normal log materialization/write");
    const std::string audit_call = compact.substr(
        audit_write, camera - audit_write);
    check(audit_call.find("frame_runtime.accepted_state") !=
              std::string::npos &&
              audit_call.find("frame_runtime.accepted_diagnostic") !=
                  std::string::npos &&
              audit_call.find("frame_runtime.publication") !=
                  std::string::npos &&
              audit_call.find("frame_runtime.working_state") ==
                  std::string::npos &&
              audit_call.find("frame_status") != std::string::npos,
          "production audit consumes only immutable accepted owners and frame status");
}

int main()
{
    test_configuration_is_optional_absolute_and_bounded();
    test_sampling_serialization_and_input_inertness();
    test_first_requested_frame_rejection_uses_accepted_baseline();
    test_open_write_and_close_errors_are_controlled();
    test_non_route_relative_record_is_analyzer_compatible();
    test_production_main_owns_audit_after_publication_and_log();
    return 0;
}
