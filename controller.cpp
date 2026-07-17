#include "g1_controller_frame_runtime.h"
#include "g1_candidate_certification_trace.h"

#if !defined(G1_CONTROLLER_NO_MAIN)
#include "g1_candidate_audit.h"

#if defined(G1_FRAME_TRANSACTION_BENCHMARK)
#include <chrono>
#endif
#include <cerrno>
#include <cstdio>
#include <cstring>

#if defined(G1_FRAME_TRANSACTION_BENCHMARK)
struct G1TransactionTimings
{
    std::FILE* stream = nullptr;
    uint32_t next_presentation_frame = 0U;
    const char* failure = nullptr;

    bool open(
        const char* path,
        char* error,
        int error_capacity)
    {
        if (path == nullptr || path[0] == '\0') return true;
        const auto fail = [error, error_capacity](const char* message) {
            if (error != nullptr && error_capacity > 0) {
                std::snprintf(
                    error,
                    static_cast<std::size_t>(error_capacity),
                    "%s",
                    message);
            }
            return false;
        };
        if (stream != nullptr || failure != nullptr) {
            return fail("transaction timing file is already open");
        }
        if (path[0] != '/') {
            return fail("MM_TRANSACTION_TIMINGS must be an absolute path");
        }
        for (const unsigned char* cursor =
                 reinterpret_cast<const unsigned char*>(path);
             *cursor != 0U;
             ++cursor) {
            if (*cursor == '\r' || *cursor == '\n') {
                return fail(
                    "MM_TRANSACTION_TIMINGS path contains a line break");
            }
        }
        std::FILE* const candidate = std::fopen(path, "wb");
        if (candidate == nullptr) {
            return fail("transaction timing file could not be opened");
        }
        static const char header[] =
            "presentation_frame\tduration_ns\n";
        if (std::fwrite(
                header, 1U, sizeof(header) - 1U, candidate) !=
                sizeof(header) - 1U ||
            std::fflush(candidate) != 0) {
            std::fclose(candidate);
            return fail("transaction timing header write failed");
        }
        stream = candidate;
        next_presentation_frame = 0U;
        failure = nullptr;
        return true;
    }

    template<class Rep, class Period>
    void append(std::chrono::duration<Rep, Period> duration)
    {
        if (stream == nullptr || failure != nullptr) return;
        const long long duration_ns = static_cast<long long>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                duration).count());
        char row[96] = {};
        const int length = std::snprintf(
            row,
            sizeof(row),
            "%u\t%lld\n",
            static_cast<unsigned>(next_presentation_frame),
            duration_ns);
        if (length < 0 ||
            static_cast<std::size_t>(length) >= sizeof(row)) {
            failure = "transaction timing row formatting failed";
            return;
        }
        if (std::fwrite(
                row,
                1U,
                static_cast<std::size_t>(length),
                stream) != static_cast<std::size_t>(length) ||
            std::fflush(stream) != 0) {
            failure = "transaction timing row write failed";
            return;
        }
        ++next_presentation_frame;
    }

    bool healthy(char* error, int error_capacity) const
    {
        if (failure == nullptr) return true;
        if (error != nullptr && error_capacity > 0) {
            std::snprintf(
                error,
                static_cast<std::size_t>(error_capacity),
                "%s",
                failure);
        }
        return false;
    }

    bool close(char* error, int error_capacity)
    {
        if (stream != nullptr) {
            if (std::fclose(stream) != 0 && failure == nullptr) {
                failure = "transaction timing close failed";
            }
            stream = nullptr;
        }
        return healthy(error, error_capacity);
    }
};
#endif


struct G1CandidateAuditConfig
{
    bool enabled = false;
    const char* path = nullptr;
    const char* route = nullptr;
    const char* heading = nullptr;
};

static bool g1_candidate_audit_error(
    char* error,
    int error_capacity,
    const char* message)
{
    if (error != nullptr && error_capacity > 0) {
        ::snprintf(error, static_cast<std::size_t>(error_capacity), "%s",
                   message != nullptr ? message :
                       "candidate audit failure");
    }
    return false;
}

static bool g1_candidate_audit_heading_is_canonical(const char* heading)
{
    if (heading == nullptr) return false;
    return ::strcmp(heading, "forward") == 0 ||
           ::strcmp(heading, "backward") == 0 ||
           ::strcmp(heading, "positive-x") == 0 ||
           ::strcmp(heading, "negative-x") == 0 ||
           ::strcmp(heading, "diagonal-positive-x") == 0 ||
           ::strcmp(heading, "diagonal-negative-x") == 0;
}

static bool g1_candidate_audit_heading_is_supported(const char* heading)
{
    return g1_candidate_audit_heading_is_canonical(heading) ||
           (heading != nullptr && ::strcmp(heading, "relative") == 0);
}

static bool g1_candidate_audit_text_is_safe(const char* value)
{
    if (value == nullptr || value[0] == '\0') return false;
    for (const unsigned char* cursor =
             reinterpret_cast<const unsigned char*>(value);
         *cursor != 0;
         ++cursor) {
        if (*cursor < 0x20U || *cursor > 0x7eU) return false;
    }
    return true;
}

static bool g1_candidate_audit_path_is_absolute_safe(const char* path)
{
    if (path == nullptr || path[0] != '/') return false;
    for (const unsigned char* cursor =
             reinterpret_cast<const unsigned char*>(path);
         *cursor != 0;
         ++cursor) {
        if (*cursor == '\r' || *cursor == '\n') return false;
    }
    return true;
}

[[maybe_unused]] static bool g1_candidate_audit_config_parse(
    G1CandidateAuditConfig& output,
    const char* path,
    g1_test_mode mode,
    int frame_limit,
    const char* mode_name,
    const char* route,
    const char* heading,
    char* error,
    int error_capacity)
{
    if (path == nullptr) {
        output = G1CandidateAuditConfig{};
        return true;
    }
    if (!g1_candidate_audit_path_is_absolute_safe(path)) {
        return g1_candidate_audit_error(
            error,
            error_capacity,
            "MM_CANDIDATE_AUDIT must be a safe absolute path");
    }
    if (mode == G1_TestLive || frame_limit <= 0 ||
        mode < G1_TestSequential || mode > G1_TestSceneCycle) {
        return g1_candidate_audit_error(
            error,
            error_capacity,
            "MM_CANDIDATE_AUDIT requires a bounded deterministic test");
    }
    if (!g1_candidate_audit_text_is_safe(mode_name) ||
        (route != nullptr && route[0] != '\0' &&
         !g1_candidate_audit_text_is_safe(route)) ||
        (heading != nullptr &&
         !g1_candidate_audit_heading_is_canonical(heading))) {
        return g1_candidate_audit_error(
            error,
            error_capacity,
            "MM_CANDIDATE_AUDIT metadata is not an exact safe test cell");
    }

    G1CandidateAuditConfig candidate;
    candidate.enabled = true;
    candidate.path = path;
    candidate.route = mode != G1_TestSceneCycle &&
            route != nullptr && route[0] != '\0'
        ? route
        : mode_name;
    candidate.heading = heading != nullptr ? heading : "relative";
    output = candidate;
    return true;
}

struct G1CandidateAuditLog
{
    FILE* stream = nullptr;
    bool enabled = false;

    static bool put_character(FILE* file, int value)
    {
        return file != nullptr && ::fputc(value, file) != EOF;
    }

    static bool put_text(FILE* file, const char* value)
    {
        return file != nullptr && value != nullptr &&
               ::fputs(value, file) >= 0;
    }

    static bool put_json_text(FILE* file, const char* value)
    {
        if (!g1_candidate_audit_text_is_safe(value) ||
            !put_character(file, '"')) {
            return false;
        }
        const unsigned char* cursor =
            reinterpret_cast<const unsigned char*>(value);
        for (; *cursor != 0; ++cursor) {
            if (*cursor == '"') {
                if (!put_text(file, "\\\"")) return false;
            } else if (*cursor == '\\') {
                if (!put_text(file, "\\\\")) return false;
            } else if (*cursor < 0x20U) {
                if (::fprintf(file, "\\u%04x", unsigned(*cursor)) < 0) {
                    return false;
                }
            } else if (!put_character(file, *cursor)) {
                return false;
            }
        }
        return put_character(file, '"');
    }

    bool open(
        const G1CandidateAuditConfig& config,
        char* error,
        int error_capacity)
    {
        if (stream != nullptr || enabled) {
            return g1_candidate_audit_error(
                error,
                error_capacity,
                "candidate audit log already owns a stream");
        }
        if (!config.enabled) return true;
        if (!g1_candidate_audit_path_is_absolute_safe(config.path) ||
            !g1_candidate_audit_text_is_safe(config.route) ||
            !g1_candidate_audit_heading_is_supported(config.heading)) {
            return g1_candidate_audit_error(
                error,
                error_capacity,
                "candidate audit configuration changed before open");
        }
        errno = 0;
        FILE* const candidate = ::fopen(config.path, "wb");
        if (candidate == nullptr) {
            return g1_candidate_audit_error(
                error,
                error_capacity,
                "could not open MM_CANDIDATE_AUDIT output");
        }
        stream = candidate;
        enabled = true;
        return true;
    }

    bool write_requested(
        const database& db,
        const g1_controller_state& accepted_state,
        const G1FrameAcceptedDiagnostic& accepted_diagnostic,
        const G1FramePublication& publication,
        G1FrameTransactionStatus frame_status,
        const char* scene_id,
        const char* route,
        const char* heading,
        char* error,
        int error_capacity)
    {
        if (!enabled) return true;
        if (stream == nullptr) {
            return g1_candidate_audit_error(
                error,
                error_capacity,
                "candidate audit stream owner is missing");
        }
        const bool finite_rejection =
            frame_status == G1FrameTransactionFiniteRejected;
        if ((frame_status != G1FrameTransactionAccepted &&
             !finite_rejection) ||
            publication.rejection.rejected != finite_rejection) {
            return g1_candidate_audit_error(
                error,
                error_capacity,
                "candidate audit received inconsistent frame status");
        }
        if (publication.presentation_frame < 0) {
            return g1_candidate_audit_error(
                error,
                error_capacity,
                "candidate audit requested frame is invalid");
        }
        if (publication.presentation_frame % 25 != 0 &&
            !finite_rejection) {
            return true;
        }
        if (accepted_state.frame_index < 0 ||
            accepted_state.frame_index >= db.features.rows ||
            (!accepted_diagnostic.ready && !finite_rejection) ||
            (accepted_diagnostic.ready &&
             (accepted_diagnostic.query_database_frame < 0 ||
              accepted_diagnostic.query_database_frame >=
                  db.features.rows)) ||
            !g1_candidate_audit_text_is_safe(scene_id) ||
            !g1_candidate_audit_text_is_safe(route) ||
            !g1_candidate_audit_heading_is_supported(heading)) {
            return g1_candidate_audit_error(
                error,
                error_capacity,
                "candidate audit received invalid immutable diagnostics");
        }

        float normalized_query[G1CandidateAuditFeatureCount] = {};
        if (db.features_offset.size != G1CandidateAuditFeatureCount ||
            db.features_scale.size != G1CandidateAuditFeatureCount ||
            db.features.cols != G1CandidateAuditFeatureCount ||
            db.features.data == nullptr ||
            db.features_offset.data == nullptr ||
            db.features_scale.data == nullptr) {
            return g1_candidate_audit_error(
                error,
                error_capacity,
                "candidate audit database normalization is invalid");
        }
        for (int dimension = 0;
             dimension < G1CandidateAuditFeatureCount;
             ++dimension) {
            // Before the first accepted transaction there is no accepted
            // diagnostic query. A finite rejection still needs a canonical
            // immutable record, so use the accepted database row as the
            // baseline query without reading the rejected working state.
            normalized_query[dimension] = accepted_diagnostic.ready
                ? ::normalize_query_feature(
                      accepted_diagnostic.query[dimension],
                      db.features_offset(dimension),
                      db.features_scale(dimension))
                : db.features(accepted_state.frame_index, dimension);
        }
        const int incumbent_index = accepted_diagnostic.ready
            ? accepted_diagnostic.query_database_frame
            : accepted_state.frame_index;

        G1CandidateAudit audit;
        if (!::g1_candidate_audit_build(
                audit,
                slice1d<float>(
                    G1CandidateAuditFeatureCount,
                    normalized_query),
                slice2d<float>(
                    db.features.rows,
                    db.features.cols,
                    db.features.data),
                slice1d<int>(
                    db.range_starts.size,
                    db.range_starts.data),
                slice1d<int>(
                    db.range_stops.size,
                    db.range_stops.data),
                incumbent_index) ||
            audit.top_count != G1CandidateAuditTopCapacity) {
            return g1_candidate_audit_error(
                error,
                error_capacity,
                "candidate audit could not build a canonical top-16");
        }

        bool ok = put_text(
            stream,
            "{\"schema\":\"g1-directional-candidate-audit/v1\","
            "\"requested_frame\":") &&
            ::fprintf(stream, "%d", publication.presentation_frame) >= 0 &&
            put_text(stream, ",\"query_bits_hex\":\"");
        for (int dimension = 0;
             ok && dimension < G1CandidateAuditFeatureCount;
             ++dimension) {
            ok = ::fprintf(
                     stream,
                     "%08x",
                     unsigned(g1_candidate_audit_float_bits(
                         normalized_query[dimension]))) >= 0;
        }
        ok = ok && put_text(stream, "\",\"scene_id\":") &&
             put_json_text(stream, scene_id) &&
             put_text(stream, ",\"route\":") &&
             put_json_text(stream, route) &&
             put_text(stream, ",\"heading\":") &&
             put_json_text(stream, heading) &&
             put_text(stream, ",\"accepted_frame\":") &&
             ::fprintf(stream, "%d", accepted_state.frame_index) >= 0 &&
             put_text(stream, ",\"eligible_count\":") &&
             ::fprintf(stream, "%u", unsigned(audit.eligible_count)) >= 0 &&
             put_text(stream, ",\"within_best_plus_0_25\":") &&
             ::fprintf(
                 stream,
                 "%u",
                 unsigned(audit.within_best_plus_0_25)) >= 0 &&
             put_text(stream, ",\"within_best_plus_1\":") &&
             ::fprintf(
                 stream,
                 "%u",
                 unsigned(audit.within_best_plus_1)) >= 0 &&
             put_text(stream, ",\"within_best_plus_4\":") &&
             ::fprintf(
                 stream,
                 "%u",
                 unsigned(audit.within_best_plus_4)) >= 0 &&
             put_text(stream, ",\"top_count\":16,\"top\":[");
        for (uint32_t index = 0; ok && index < audit.top_count; ++index) {
            const G1CandidateAuditEntry& entry = audit.top[index];
            ok = (index == 0U || put_character(stream, ',')) &&
                 put_text(stream, "{\"frame\":") &&
                 ::fprintf(stream, "%d", entry.frame) >= 0 &&
                 put_text(stream, ",\"range\":") &&
                 ::fprintf(stream, "%d", entry.range) >= 0 &&
                 put_text(stream, ",\"cost_bits\":\"") &&
                 ::fprintf(stream, "%08x", unsigned(entry.cost_bits)) >= 0 &&
                 put_text(stream, "\"}");
        }
        ok = ok && put_text(stream, "]}\n") && ::fflush(stream) == 0;
        if (!ok) {
            return g1_candidate_audit_error(
                error,
                error_capacity,
                "could not write MM_CANDIDATE_AUDIT output");
        }
        return true;
    }

    bool close(char* error, int error_capacity)
    {
        if (!enabled && stream == nullptr) return true;
        FILE* const owned = stream;
        stream = nullptr;
        enabled = false;
        if (owned == nullptr || ::fclose(owned) != 0) {
            return g1_candidate_audit_error(
                error,
                error_capacity,
                "could not close MM_CANDIDATE_AUDIT output");
        }
        return true;
    }
};

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wenum-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#if defined(PLATFORM_WEB)
#include <emscripten/emscripten.h>
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include "raylib.h"
#include "raymath.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "common.h"
#include "vec.h"
#include "quat.h"
#include "spring.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "array.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "character.h"
#include "scene_runtime.h"
#include "route_runtime.h"
#include "support_runtime.h"
#include "g1_controller_state.h"
#include "g1_ik.h"
#include "g1_runtime_diagnostics.h"
#include "scene_switch.h"
#include "motion_match_log.h"
#include "cleanup_runtime.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "nnet.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "lmm.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <functional>


struct G1ProcessConfig
{
    float initial_search_time;
    bool ik_enabled;
    float inertialize_blending_halflife;
    float simulation_rotation_halflife;
    bool desired_strafe;
};

struct G1ArgumentConfig
{
    std::string terrain_directory = "./resources/g1_terrain";
    float terrain_weight = 0.0f;
    g1_test_mode mode = G1_TestLive;
    std::string mode_name = "live";
    std::string route = "manual";
    int scene_dwell_frames = 25;
    int frame_limit = 0;
    std::string test_heading;
    std::string terrain_scene;
    std::string log_path;
    std::string cleanup_log_path;
    int discrete_mode = 0;
    int discrete_snap_frames = 12;
};

struct G1AcceptedLogContext
{
    float fixed_dt = 1.0f / 25.0f;
    const char* mode = "live";
    const char* route = "manual";
    const char* empty_query_bits = "";
    bool ik_enabled = false;
    const database* db = nullptr;
    const motion_source_record* sources = nullptr;
    const char* const* source_names = nullptr;
    const char* const* source_terrains = nullptr;
    int source_count = 0;
    const char* const* scene_ids = nullptr;
    const int* active_scene_index = nullptr;
    float route_target_height = 0.0f;
    const int* scene_generation = nullptr;
    const int* scene_reset_count = nullptr;
    const bool* scene_switch_failed = nullptr;
    const int* motion_pack_load_count = nullptr;
    const int* model_load_count = nullptr;
    const int* model_unload_count = nullptr;
};

static bool g1_parse_argument_float(
    float& output,
    const char* text,
    float minimum,
    float maximum,
    const char* label,
    char* error,
    int error_capacity)
{
    errno = 0;
    char* end = nullptr;
    const float parsed = ::strtof(text, &end);
    if (text == nullptr || text[0] == '\0' || end == text ||
        *end != '\0' || errno == ERANGE || !std::isfinite(parsed) ||
        parsed < minimum || parsed > maximum) {
        return ::scene_error(
            error,
            error_capacity,
            "%s must be finite in [%.9g,%.9g]",
            label,
            minimum,
            maximum);
    }
    output = parsed;
    return true;
}

static bool g1_parse_argument_int(
    int& output,
    const char* text,
    int minimum,
    int maximum,
    const char* label,
    char* error,
    int error_capacity)
{
    errno = 0;
    char* end = nullptr;
    const long parsed = ::strtol(text, &end, 10);
    if (text == nullptr || text[0] == '\0' || end == text ||
        *end != '\0' || errno == ERANGE || parsed < minimum ||
        parsed > maximum) {
        return ::scene_error(
            error,
            error_capacity,
            "%s must be an integer in [%d,%d]",
            label,
            minimum,
            maximum);
    }
    output = static_cast<int>(parsed);
    return true;
}

static bool g1_parse_arguments(
    G1ArgumentConfig& output,
    int argument_count,
    char** arguments,
    char* error,
    int error_capacity)
{
    bool terrain_weight_set = false;
    for (int index = 1; index < argument_count; ++index) {
        const char* option = arguments[index];
        if (index + 1 >= argument_count) {
            return ::scene_error(
                error,
                error_capacity,
                "controller option '%s' requires a value",
                option);
        }
        const char* value = arguments[++index];
        if (::strcmp(option, "--terrain-dir") == 0) {
            output.terrain_directory = value;
        } else if (::strcmp(option, "--terrain-weight") == 0) {
            if (!::g1_parse_argument_float(
                    output.terrain_weight,
                    value,
                    0.0f,
                    10.0f,
                    option,
                    error,
                    error_capacity)) {
                return false;
            }
            terrain_weight_set = true;
        } else if (::strcmp(option, "--test-mode") == 0) {
            if (::strcmp(value, "live") == 0) {
                output.mode = G1_TestLive;
                output.mode_name = value;
            } else if (::strcmp(value, "sequential") == 0) {
                output.mode = G1_TestSequential;
                output.mode_name = value;
                output.route = "curb-forward";
            } else if (::strcmp(value, "flat") == 0) {
                output.mode = G1_TestFlat;
                output.mode_name = value;
                output.route = "curb-forward";
            } else if (::strcmp(value, "terrain") == 0) {
                output.mode = G1_TestTerrain;
                output.mode_name = value;
                output.route = "curb-forward";
            } else if (::strcmp(value, "route") == 0) {
                output.mode = G1_TestRoute;
                output.mode_name = value;
            } else if (::strcmp(value, "scene-cycle") == 0) {
                output.mode = G1_TestSceneCycle;
                output.mode_name = value;
                output.route.clear();
            } else {
                return ::scene_error(
                    error,
                    error_capacity,
                    "unknown --test-mode '%s'",
                    value);
            }
        } else if (::strcmp(option, "--test-route") == 0) {
            output.route = value;
        } else if (::strcmp(option, "--scene-dwell-frames") == 0) {
            if (!::g1_parse_argument_int(
                    output.scene_dwell_frames,
                    value,
                    1,
                    10000,
                    option,
                    error,
                    error_capacity)) {
                return false;
            }
        } else if (::strcmp(option, "--test-frames") == 0) {
            if (!::g1_parse_argument_int(
                    output.frame_limit,
                    value,
                    1,
                    INT_MAX,
                    option,
                    error,
                    error_capacity)) {
                return false;
            }
        } else if (::strcmp(option, "--test-heading") == 0) {
            output.test_heading = value;
        } else if (::strcmp(option, "--terrain-scene") == 0) {
            output.terrain_scene = value;
        } else if (::strcmp(option, "--log") == 0) {
            output.log_path = value;
        } else if (::strcmp(option, "--cleanup-log") == 0) {
            output.cleanup_log_path = value;
        } else if (::strcmp(option, "--mode") == 0) {
            if (!::g1_parse_argument_int(
                    output.discrete_mode,
                    value,
                    0,
                    3,
                    option,
                    error,
                    error_capacity)) {
                return false;
            }
        } else if (::strcmp(option, "--snapn") == 0) {
            if (!::g1_parse_argument_int(
                    output.discrete_snap_frames,
                    value,
                    1,
                    10000,
                    option,
                    error,
                    error_capacity)) {
                return false;
            }
        } else {
            return ::scene_error(
                error,
                error_capacity,
                "unknown controller option '%s'",
                option);
        }
    }
    if (output.terrain_directory.empty()) {
        return ::scene_error(
            error, error_capacity, "--terrain-dir may not be empty");
    }
    if (output.mode == G1_TestRoute && output.route.empty()) {
        return ::scene_error(
            error,
            error_capacity,
            "--test-route is required for route mode");
    }
    if (output.mode != G1_TestLive && output.frame_limit <= 0) {
        return ::scene_error(
            error,
            error_capacity,
            "--test-frames is required for deterministic modes");
    }
    if (output.mode == G1_TestSceneCycle &&
        output.frame_limit < 14 * output.scene_dwell_frames) {
        return ::scene_error(
            error,
            error_capacity,
            "scene-cycle needs at least 14*dwell frames");
    }
    if (output.mode == G1_TestFlat ||
        output.mode == G1_TestSequential) {
        output.terrain_weight = 0.0f;
    } else if (output.mode == G1_TestTerrain && !terrain_weight_set) {
        output.terrain_weight = 4.0f;
    }
    return true;
}

static void controlled_runtime_error(const char* message)
{
    ::fprintf(
        stderr,
        "G1 controlled runtime error: %s\n",
        message != nullptr ? message : "unknown runtime failure");
}

static inline Vector3 to_Vector3(vec3 value)
{
    return Vector3{value.x, value.y, value.z};
}

static bool g1_parse_search_time(
    float& output,
    const char* text,
    char* error,
    int error_capacity)
{
    if (text == NULL) { output = 0.10f; return true; }
    errno = 0;
    char* end = NULL;
    const float parsed = ::strtof(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE ||
        !std::isfinite(parsed) || parsed < 0.0f || parsed > 10.0f) {
        ::snprintf(error, error_capacity, "MM_SEARCHT must be finite in [0,10]");
        return false;
    }
    output = parsed;
    return true;
}

static bool g1_parse_ik_enabled(
    bool& output,
    const char* text,
    char* error,
    int error_capacity)
{
    if (text == NULL) { output = false; return true; }
    if (::strcmp(text, "0") == 0) { output = false; return true; }
    if (::strcmp(text, "1") == 0) { output = true; return true; }
    ::snprintf(error, error_capacity, "MM_IK must be 0 or 1");
    return false;
}

static bool g1_parse_halflife(
    float& output,
    const char* text,
    char* error,
    int error_capacity)
{
    if (text == NULL) { return true; }
    errno = 0;
    char* end = NULL;
    const float parsed = ::strtof(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE ||
        !std::isfinite(parsed) || parsed <= 0.0f || parsed > 10.0f) {
        ::snprintf(error, error_capacity, "halflife must be finite in (0,10]");
        return false;
    }
    output = parsed;
    return true;
}

static bool g1_parse_strafe_enabled(
    bool& output,
    const char* text,
    char* error,
    int error_capacity)
{
    if (text == NULL) { output = false; return true; }
    if (::strcmp(text, "0") == 0) { output = false; return true; }
    if (::strcmp(text, "1") == 0) { output = true; return true; }
    ::snprintf(error, error_capacity, "MM_STRAFE must be 0 or 1");
    return false;
}

enum
{
    GAMEPAD_PLAYER = 0,
};

enum
{
    GAMEPAD_STICK_LEFT,
    GAMEPAD_STICK_RIGHT,
};

static vec3 gamepad_get_stick(
    const int stick,
    const float deadzone = 0.2f)
{
#if defined(MM_AUTODRIVE)
    const float time = static_cast<float>(::GetTime());
    if (stick == GAMEPAD_STICK_LEFT) {
        return vec3(
            0.7f * ::sinf(0.4f * time),
            0.0f,
            -0.7f * ::cosf(0.4f * time));
    }
    return vec3();
#else
    float x = ::GetGamepadAxisMovement(
        GAMEPAD_PLAYER,
        stick == GAMEPAD_STICK_LEFT
            ? GAMEPAD_AXIS_LEFT_X
            : GAMEPAD_AXIS_RIGHT_X);
    float z = ::GetGamepadAxisMovement(
        GAMEPAD_PLAYER,
        stick == GAMEPAD_STICK_LEFT
            ? GAMEPAD_AXIS_LEFT_Y
            : GAMEPAD_AXIS_RIGHT_Y);
    if (stick == GAMEPAD_STICK_LEFT) {
        if (::IsKeyDown(KEY_A)) x -= 1.0f;
        if (::IsKeyDown(KEY_D)) x += 1.0f;
        if (::IsKeyDown(KEY_W)) z -= 1.0f;
        if (::IsKeyDown(KEY_S)) z += 1.0f;
    } else {
        if (::IsKeyDown(KEY_LEFT)) x -= 1.0f;
        if (::IsKeyDown(KEY_RIGHT)) x += 1.0f;
        if (::IsKeyDown(KEY_UP)) z -= 1.0f;
        if (::IsKeyDown(KEY_DOWN)) z += 1.0f;
    }
    const float magnitude = ::sqrtf(x * x + z * z);
    if (magnitude <= deadzone) return vec3();
    const float remapped = (magnitude - deadzone) / (1.0f - deadzone);
    const float scale = remapped * remapped / magnitude;
    return vec3(x * scale, 0.0f, z * scale);
#endif
}

static void update_g1_camera_from_accepted(
    Camera3D& camera,
    const float azimuth,
    const float altitude,
    const float distance,
    const vec3& target)
{
    const vec3 focus = target;
    const vec3 offset = distance * vec3(
        ::cosf(altitude) * ::sinf(azimuth),
        ::sinf(altitude),
        ::cosf(altitude) * ::cosf(azimuth));
    camera.target = ::to_Vector3(focus);
    camera.position = ::to_Vector3(focus + offset);
}

static void draw_g1_skeleton(
    const array1d<vec3>& positions,
    const array1d<quat>& rotations,
    const array1d<int>& parents)
{
    for (int bone = 1; bone < positions.size; ++bone) {
        ::DrawSphereWires(
            ::to_Vector3(positions(bone)), 0.035f, 4, 8, SKYBLUE);
        ::DrawCylinderEx(
            ::to_Vector3(positions(parents(bone))),
            ::to_Vector3(positions(bone)),
            0.022f, 0.016f, 6, LIGHTGRAY);
        ::DrawLine3D(
            ::to_Vector3(positions(bone)),
            ::to_Vector3(
                positions(bone) + 0.10f * ::quat_mul_vec3(
                    rotations(bone), vec3(0.0f, 0.0f, 1.0f))),
            RED);
    }
}

static const char* g1_log_footprint_status_name(G1FootprintStatus status)
{
    switch (status) {
    case G1FootprintOk: return "ok";
    case G1FootprintOutsideDomain: return "outside-domain";
    case G1FootprintBudgetExceeded: return "budget-exceeded";
    case G1FootprintInvalidInput: return "invalid-input";
    case G1FootprintInvalidField: return "invalid-field";
    case G1FootprintArithmeticFailure: return "arithmetic-failure";
    default: return "unknown";
    }
}

static const char* g1_log_surface_status_name(G1SurfaceQueryStatus status)
{
    switch (status) {
    case G1SurfaceQueryValid: return "valid";
    case G1SurfaceQueryOutside: return "outside";
    case G1SurfaceQueryInvalid: return "invalid";
    default: return "unknown";
    }
}

static void g1_log_quat_bits_hex(char output[4 * 8 + 1], quat value)
{
    const float components[4] = {value.w, value.x, value.y, value.z};
    for (int component = 0; component < 4; ++component) {
        ::snprintf(
            output + component * 8,
            9,
            "%08x",
            (unsigned)terrain_float_bits(components[component]));
    }
    output[4 * 8] = '\0';
}

static void g1_log_predicted_heading_bits_hex(
    char output[4 * 4 * 8 + 1],
    const quat values[G1CommandTrajectorySampleCount])
{
    for (int sample = 0; sample < G1CommandTrajectorySampleCount; ++sample) {
        g1_log_quat_bits_hex(output + sample * 4 * 8, values[sample]);
    }
    output[4 * 4 * 8] = '\0';
}

static void g1_log_sphere_bits_hex(
    char output[12 * 8 + 1],
    const uint32_t values[4][3])
{
    for (int probe = 0; probe < 4; ++probe) {
        for (int axis = 0; axis < 3; ++axis) {
            const int word = probe * 3 + axis;
            ::snprintf(
                output + word * 8,
                9,
                "%08x",
                (unsigned)values[probe][axis]);
        }
    }
    output[12 * 8] = '\0';
}

static double g1_log_heading_error_degrees(quat desired, quat actual)
{
    quat normalized_desired;
    quat normalized_actual;
    vec3 desired_forward;
    vec3 actual_forward;
    if (!ik_checked_quat_normalize(normalized_desired, desired) ||
        !ik_checked_quat_normalize(normalized_actual, actual) ||
        !ik_checked_quat_rotate(
            desired_forward,
            normalized_desired,
            vec3(0.0f, 0.0f, 1.0f)) ||
        !ik_checked_quat_rotate(
            actual_forward,
            normalized_actual,
            vec3(0.0f, 0.0f, 1.0f))) {
        return std::numeric_limits<double>::infinity();
    }
    const double desired_yaw = ::atan2(
        static_cast<double>(desired_forward.x),
        static_cast<double>(desired_forward.z));
    const double actual_yaw = ::atan2(
        static_cast<double>(actual_forward.x),
        static_cast<double>(actual_forward.z));
    double error = ::fabs(desired_yaw - actual_yaw);
    const double pi = 3.14159265358979323846264338327950288;
    if (error > pi) error = 2.0 * pi - error;
    return error * (180.0 / pi);
}

static void g1_log_clearance_work(
    motion_match_clearance_work_diagnostic& output,
    const G1ClearanceWork& value)
{
    output.point_queries = value.point_queries;
    output.cells_visited = value.cells_visited;
    output.primitive_triangle_pairs = value.primitive_triangle_pairs;
    output.face_patches = value.face_patches;
    output.candidate_tests = value.candidate_tests;
    output.subdivision_nodes = value.subdivision_nodes;
}

static void g1_log_hash_word(uint64_t& hash, uint64_t value)
{
    for (int byte = 0; byte < 8; ++byte) {
        hash ^= (value >> (byte * 8)) & UINT64_C(0xff);
        hash *= UINT64_C(1099511628211);
    }
}

static void g1_log_hash_value(uint64_t& hash, bool value)
{
    g1_log_hash_word(hash, value ? 1U : 0U);
}

static void g1_log_hash_value(uint64_t& hash, int value)
{
    g1_log_hash_word(hash, static_cast<uint32_t>(value));
}

static void g1_log_hash_value(uint64_t& hash, uint32_t value)
{
    g1_log_hash_word(hash, value);
}

static void g1_log_hash_value(uint64_t& hash, float value)
{
    g1_log_hash_word(hash, terrain_float_bits(value));
}

static void g1_log_hash_value(uint64_t& hash, double value)
{
    uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    g1_log_hash_word(hash, bits);
}

static void g1_log_hash_value(uint64_t& hash, vec3 value)
{
    g1_log_hash_value(hash, value.x);
    g1_log_hash_value(hash, value.y);
    g1_log_hash_value(hash, value.z);
}

static void g1_log_hash_value(uint64_t& hash, quat value)
{
    g1_log_hash_value(hash, value.w);
    g1_log_hash_value(hash, value.x);
    g1_log_hash_value(hash, value.y);
    g1_log_hash_value(hash, value.z);
}

template<class T>
static void g1_log_hash_array(
    uint64_t& hash, const array1d<T>& values)
{
    g1_log_hash_value(hash, values.size);
    for (int index = 0; index < values.size; ++index) {
        g1_log_hash_value(hash, values(index));
    }
}

static void g1_log_hash_surface(
    uint64_t& hash, const G1SurfaceSample& value)
{
    g1_log_hash_value(hash, value.height);
    g1_log_hash_value(hash, value.normal);
}

static void g1_log_hash_clearance_work(
    uint64_t& hash, const G1ClearanceWork& value)
{
    g1_log_hash_value(hash, value.point_queries);
    g1_log_hash_value(hash, value.cells_visited);
    g1_log_hash_value(hash, value.primitive_triangle_pairs);
    g1_log_hash_value(hash, value.face_patches);
    g1_log_hash_value(hash, value.candidate_tests);
    g1_log_hash_value(hash, value.subdivision_nodes);
}

static void g1_log_hash_clearance_result(
    uint64_t& hash, const G1ClearanceResult& value)
{
    g1_log_hash_value(hash, value.lower_bound_m);
    g1_log_hash_value(hash, value.witness_upper_m);
    g1_log_hash_value(hash, value.witness.body_x);
    g1_log_hash_value(hash, value.witness.body_y);
    g1_log_hash_value(hash, value.witness.body_z);
    g1_log_hash_value(hash, value.witness.surface_x);
    g1_log_hash_value(hash, value.witness.surface_y);
    g1_log_hash_value(hash, value.witness.surface_z);
    g1_log_hash_value(hash, value.witness.segment_parameter);
    g1_log_hash_value(hash, value.witness.terrain_weight_0);
    g1_log_hash_value(hash, value.witness.terrain_weight_1);
    g1_log_hash_value(hash, value.witness.terrain_weight_2);
    g1_log_hash_value(hash, value.witness.primitive_index);
    g1_log_hash_value(hash, value.witness.cell_x);
    g1_log_hash_value(hash, value.witness.cell_z);
    g1_log_hash_value(hash, value.witness.terrain_triangle_index);
    g1_log_hash_value(hash, value.witness.patch_index);
    g1_log_hash_value(hash, value.witness.candidate_kind);
    g1_log_hash_value(hash, value.witness.candidate_subindex);
    g1_log_hash_clearance_work(hash, value.work);
}

static void g1_log_hash_leg_clearance(
    uint64_t& hash, const G1LegClearance& value)
{
    g1_log_hash_clearance_result(hash, value.knee);
    g1_log_hash_clearance_result(hash, value.ankle);
    g1_log_hash_clearance_result(hash, value.toe);
    g1_log_hash_clearance_result(hash, value.foot);
    g1_log_hash_clearance_result(hash, value.thigh);
    g1_log_hash_clearance_result(hash, value.shin);
    g1_log_hash_clearance_result(hash, value.minimum);
}

static void g1_log_hash_pose_clearance(
    uint64_t& hash, const G1PoseClearance& value)
{
    g1_log_hash_clearance_result(hash, value.hips);
    g1_log_hash_leg_clearance(hash, value.left);
    g1_log_hash_leg_clearance(hash, value.right);
    g1_log_hash_clearance_result(hash, value.minimum);
}

static void g1_log_hash_command(
    uint64_t& hash, const G1CommandSnapshot& value)
{
    g1_log_hash_value(hash, value.intent.requested_velocity);
    g1_log_hash_value(hash, value.intent.desired_heading);
    g1_log_hash_value(hash, value.applied_velocity);
    for (int sample = 0;
         sample < G1CommandTrajectorySampleCount;
         ++sample) {
        g1_log_hash_value(
            hash, value.predicted_desired_velocities[sample]);
        g1_log_hash_value(hash, value.predicted_root_positions[sample]);
        g1_log_hash_value(hash, value.predicted_root_rotations[sample]);
        g1_log_hash_value(hash, value.predicted_desired_headings[sample]);
    }
}

static void g1_log_hash_support(
    uint64_t& hash, const support_frame_state& value)
{
    g1_log_hash_value(hash, value.height);
    g1_log_hash_value(hash, value.velocity);
    g1_log_hash_value(hash, value.nominal_height);
    g1_log_hash_value(hash, value.nominal_velocity);
    g1_log_hash_value(hash, value.offset_height);
    g1_log_hash_value(hash, value.offset_velocity);
    g1_log_hash_value(hash, value.airborne_frames);
    g1_log_hash_value(hash, static_cast<int>(value.source));
    g1_log_hash_value(hash, value.initialized);
}

static void g1_log_hash_support_observation(
    uint64_t& hash, const support_observation& value)
{
    for (int index = 0; index < 3; ++index) {
        g1_log_hash_value(hash, value.source_height[index]);
        g1_log_hash_value(hash, value.runtime_height[index]);
        g1_log_hash_value(hash, value.delta[index]);
    }
    g1_log_hash_value(hash, value.contact[0]);
    g1_log_hash_value(hash, value.contact[1]);
}

static void g1_log_hash_footprint(
    uint64_t& hash, const G1FootprintObservation& value)
{
    g1_log_hash_surface(hash, value.root_surface);
    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        const G1FootprintFootObservation& foot = value.feet[foot_index];
        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            const G1FootprintProbe& probe = foot.probes[probe_index];
            g1_log_hash_value(hash, probe.current_sphere_center);
            g1_log_hash_value(hash, probe.current_sole_point);
            g1_log_hash_surface(hash, probe.current_surface);
            for (int sample = 0;
                 sample < G1CommandTrajectorySampleCount;
                 ++sample) {
                g1_log_hash_value(
                    hash, probe.predicted_sphere_centers[sample]);
                g1_log_hash_value(
                    hash, probe.predicted_sole_points[sample]);
                g1_log_hash_value(
                    hash,
                    static_cast<int>(
                        probe.predicted_surface_status[sample]));
                g1_log_hash_surface(
                    hash, probe.predicted_surfaces[sample]);
            }
            g1_log_hash_surface(hash, probe.selected_landing_surface);
            g1_log_hash_value(hash, probe.corridor_minimum_height);
            g1_log_hash_value(hash, probe.corridor_maximum_height);
            g1_log_hash_value(hash, probe.encountered_walkability_class);
        }
        g1_log_hash_value(hash, foot.current_contact);
        g1_log_hash_value(hash, foot.landing_expected);
        g1_log_hash_value(hash, foot.landing_patch_ready);
        g1_log_hash_value(hash, foot.landing_sample);
        g1_log_hash_value(hash, foot.predicted_landing_sole_center);
        g1_log_hash_value(
            hash,
            static_cast<int>(foot.predicted_landing_surface_status));
        g1_log_hash_surface(hash, foot.predicted_landing_surface);
        g1_log_hash_value(
            hash, foot.predicted_landing_walkability_class);
        g1_log_hash_value(hash, foot.landing_patch_maximum_residual_m);
        g1_log_hash_value(hash, foot.corridor_minimum_height);
        g1_log_hash_value(hash, foot.corridor_maximum_height);
        g1_log_hash_value(hash, foot.maximum_root_split_m);
        g1_log_hash_value(hash, foot.encountered_walkability_class);
        g1_log_hash_value(hash, foot.multilevel);
    }
    g1_log_hash_value(hash, value.blocked);
    g1_log_hash_value(hash, static_cast<int>(value.blocked_reason));
    g1_log_hash_value(hash, value.work.sweeps);
    g1_log_hash_value(hash, value.work.surface_queries);
    g1_log_hash_value(hash, value.work.node_visits);
}

static void g1_log_hash_lock(
    uint64_t& hash, const G1FootLockState& value)
{
    g1_log_hash_value(hash, value.initialized);
    g1_log_hash_value(hash, value.contact);
    g1_log_hash_value(hash, value.locked);
    g1_log_hash_value(hash, value.position_active);
    g1_log_hash_value(hash, value.releasing);
    g1_log_hash_value(hash, value.release_frames);
    g1_log_hash_value(hash, value.previous_input);
    g1_log_hash_value(hash, value.lock_point);
    g1_log_hash_value(hash, value.output_position);
    g1_log_hash_value(hash, value.output_velocity);
    g1_log_hash_value(hash, value.offset_position);
    g1_log_hash_value(hash, value.offset_velocity);
}

static void g1_log_hash_ik_state(uint64_t& hash, const G1IkState& value)
{
    g1_log_hash_value(hash, value.initialized);
    for (int foot = 0; foot < 2; ++foot) {
        g1_log_hash_lock(hash, value.feet[foot].lock);
        g1_log_hash_value(hash, value.feet[foot].swing.initialized);
        for (int probe = 0; probe < 4; ++probe) {
            g1_log_hash_value(
                hash, value.feet[foot].swing.previous_sphere_centers[probe]);
        }
        g1_log_hash_value(hash, value.feet[foot].baseline_sole_normal);
    }
}

static void g1_log_hash_target(
    uint64_t& hash, const G1FootTarget& value)
{
    g1_log_hash_value(hash, value.locked);
    g1_log_hash_value(hash, value.position_active);
    g1_log_hash_value(hash, value.releasing);
    g1_log_hash_value(hash, value.drift_limit_exceeded);
    g1_log_hash_value(hash, value.surface.point);
    g1_log_hash_value(hash, value.surface.normal);
    g1_log_hash_value(hash, value.desired_sole_normal);
    g1_log_hash_value(hash, value.sole_center);
    g1_log_hash_value(hash, value.horizontal_drift_m);
}

static void g1_log_hash_swing_candidate(
    uint64_t& hash, const G1SwingCandidateDiagnostic& value)
{
    g1_log_hash_value(hash, value.candidate_index);
    g1_log_hash_value(hash, value.lift_bits);
    g1_log_hash_value(hash, value.materialized_command_y_bits);
    for (int probe = 0; probe < 4; ++probe) {
        for (int axis = 0; axis < 3; ++axis) {
            g1_log_hash_value(
                hash, value.actual_sphere_center_bits[probe][axis]);
        }
    }
    g1_log_hash_value(hash, static_cast<int>(value.clearance_status));
    g1_log_hash_value(hash, value.controller_constraints_passed);
    g1_log_hash_value(hash, value.clearance_certified);
    g1_log_hash_value(hash, value.lower_margin_m);
    g1_log_hash_value(hash, value.witness_upper_margin_m);
    g1_log_hash_clearance_work(hash, value.clearance_work);
}

static void g1_log_hash_frame_result(
    uint64_t& hash, const G1IkFrameResult& value)
{
    g1_log_hash_value(hash, value.applied);
    g1_log_hash_value(hash, value.safe_stop_requested);
    g1_log_hash_value(hash, static_cast<int>(value.stop_reason));
    g1_log_hash_value(hash, value.max_correction_radians);
    g1_log_hash_value(hash, value.root_reach.active);
    g1_log_hash_value(hash, value.root_reach.common_interval_found);
    g1_log_hash_value(hash, value.root_reach.applied);
    g1_log_hash_value(hash, value.root_reach.root_y_delta_m);
    for (int foot = 0; foot < 2; ++foot) {
        const G1FootFrameResult& result = value.feet[foot];
        g1_log_hash_value(hash, result.recorded_contact);
        g1_log_hash_target(hash, result.target);
        g1_log_hash_value(
            hash, result.swing_selection.candidates_evaluated);
        g1_log_hash_value(hash, result.swing_selection.selected_index);
        g1_log_hash_swing_candidate(
            hash, result.swing_selection.selected);
        g1_log_hash_clearance_work(
            hash, result.swing_selection.total_clearance_work);
        g1_log_hash_value(hash, result.defensive_swing.lower_margin_m);
        g1_log_hash_value(hash, result.defensive_swing.witness_upper_m);
        g1_log_hash_value(hash, result.defensive_swing.sweep_evaluated);
        g1_log_hash_clearance_work(hash, result.defensive_swing.work);
        const G1LegSolveResult& position = result.position;
        g1_log_hash_value(hash, position.applied);
        g1_log_hash_value(hash, position.reachable);
        g1_log_hash_value(hash, position.correction_limited);
        g1_log_hash_value(hash, position.safe_stop_requested);
        g1_log_hash_value(hash, position.iterations);
        g1_log_hash_value(
            hash, static_cast<int>(position.iteration_provenance));
        g1_log_hash_value(hash, position.requested_ankle_target);
        g1_log_hash_value(hash, position.clamped_ankle_target);
        g1_log_hash_value(hash, position.hinge_axis_world);
        g1_log_hash_value(hash, position.bend_direction);
        g1_log_hash_value(hash, position.bend_used_current_projection);
        g1_log_hash_value(hash, position.bend_used_hinge_fallback);
        g1_log_hash_value(hash, position.bend_used_safe_perpendicular);
        g1_log_hash_value(hash, position.bend_sign_flipped);
        g1_log_hash_value(hash, position.raw_distance_m);
        g1_log_hash_value(hash, position.clamped_distance_m);
        g1_log_hash_value(hash, position.max_correction_radians);
        g1_log_hash_value(hash, position.contact_residual_m);
        const G1FootOrientationResult& orientation = result.orientation;
        g1_log_hash_value(hash, orientation.applied);
        g1_log_hash_value(hash, orientation.correction_limited);
        g1_log_hash_value(hash, orientation.safe_stop_requested);
        g1_log_hash_value(hash, orientation.target_global_rotation);
        g1_log_hash_value(
            hash, orientation.requested_correction_radians);
        g1_log_hash_value(hash, orientation.correction_radians);
    }
}

static uint64_t g1_log_accepted_state_digest(
    const g1_controller_state& state)
{
    uint64_t hash = UINT64_C(1469598103934665603);
#define G1_LOG_HASH_VALUE(name) g1_log_hash_value(hash, state.name)
#define G1_LOG_HASH_ARRAY(name) g1_log_hash_array(hash, state.name)
    G1_LOG_HASH_VALUE(frame_index);
    G1_LOG_HASH_VALUE(scene_frame);
    G1_LOG_HASH_VALUE(search_time);
    G1_LOG_HASH_VALUE(search_timer);
    G1_LOG_HASH_VALUE(force_search_timer);
    G1_LOG_HASH_ARRAY(curr_bone_positions);
    G1_LOG_HASH_ARRAY(curr_bone_velocities);
    G1_LOG_HASH_ARRAY(trns_bone_positions);
    G1_LOG_HASH_ARRAY(trns_bone_velocities);
    G1_LOG_HASH_ARRAY(curr_bone_rotations);
    G1_LOG_HASH_ARRAY(trns_bone_rotations);
    G1_LOG_HASH_ARRAY(curr_bone_angular_velocities);
    G1_LOG_HASH_ARRAY(trns_bone_angular_velocities);
    G1_LOG_HASH_ARRAY(curr_bone_contacts);
    G1_LOG_HASH_ARRAY(trns_bone_contacts);
    G1_LOG_HASH_ARRAY(bone_positions);
    G1_LOG_HASH_ARRAY(bone_velocities);
    G1_LOG_HASH_ARRAY(bone_angular_velocities);
    G1_LOG_HASH_ARRAY(bone_rotations);
    G1_LOG_HASH_ARRAY(bone_offset_positions);
    G1_LOG_HASH_ARRAY(bone_offset_velocities);
    G1_LOG_HASH_ARRAY(bone_offset_angular_velocities);
    G1_LOG_HASH_ARRAY(bone_offset_rotations);
    G1_LOG_HASH_ARRAY(adjusted_bone_positions);
    G1_LOG_HASH_ARRAY(global_bone_positions);
    G1_LOG_HASH_ARRAY(global_bone_velocities);
    G1_LOG_HASH_ARRAY(adjusted_bone_rotations);
    G1_LOG_HASH_ARRAY(global_bone_rotations);
    G1_LOG_HASH_ARRAY(global_bone_angular_velocities);
    G1_LOG_HASH_ARRAY(global_bone_computed);
    G1_LOG_HASH_ARRAY(ik_bone_positions);
    G1_LOG_HASH_ARRAY(ik_bone_rotations);
    G1_LOG_HASH_ARRAY(ik_global_bone_positions);
    G1_LOG_HASH_ARRAY(ik_global_bone_rotations);
    G1_LOG_HASH_ARRAY(ik_candidate_bone_positions);
    G1_LOG_HASH_ARRAY(ik_candidate_bone_rotations);
    G1_LOG_HASH_ARRAY(ik_candidate_global_bone_positions);
    G1_LOG_HASH_ARRAY(ik_candidate_global_bone_rotations);
    G1_LOG_HASH_ARRAY(trajectory_desired_velocities);
    G1_LOG_HASH_ARRAY(trajectory_positions);
    G1_LOG_HASH_ARRAY(trajectory_velocities);
    G1_LOG_HASH_ARRAY(trajectory_accelerations);
    G1_LOG_HASH_ARRAY(trajectory_angular_velocities);
    G1_LOG_HASH_ARRAY(trajectory_desired_rotations);
    G1_LOG_HASH_ARRAY(trajectory_rotations);
    G1_LOG_HASH_ARRAY(contact_bones);
    G1_LOG_HASH_ARRAY(contact_states);
    G1_LOG_HASH_ARRAY(contact_locks);
    G1_LOG_HASH_ARRAY(contact_positions);
    G1_LOG_HASH_ARRAY(contact_velocities);
    G1_LOG_HASH_ARRAY(contact_points);
    G1_LOG_HASH_ARRAY(contact_targets);
    G1_LOG_HASH_ARRAY(contact_offset_positions);
    G1_LOG_HASH_ARRAY(contact_offset_velocities);
    G1_LOG_HASH_VALUE(transition_src_position);
    G1_LOG_HASH_VALUE(transition_dst_position);
    G1_LOG_HASH_VALUE(transition_src_rotation);
    G1_LOG_HASH_VALUE(transition_dst_rotation);
    G1_LOG_HASH_VALUE(desired_velocity);
    G1_LOG_HASH_VALUE(desired_velocity_change_curr);
    G1_LOG_HASH_VALUE(desired_velocity_change_prev);
    G1_LOG_HASH_VALUE(desired_rotation);
    G1_LOG_HASH_VALUE(desired_rotation_change_curr);
    G1_LOG_HASH_VALUE(desired_rotation_change_prev);
    G1_LOG_HASH_VALUE(desired_gait);
    G1_LOG_HASH_VALUE(desired_gait_velocity);
    G1_LOG_HASH_VALUE(simulation_position);
    G1_LOG_HASH_VALUE(simulation_velocity);
    G1_LOG_HASH_VALUE(simulation_acceleration);
    G1_LOG_HASH_VALUE(simulation_rotation);
    G1_LOG_HASH_VALUE(simulation_angular_velocity);
    g1_log_hash_command(hash, state.command);
    G1_LOG_HASH_VALUE(footprint_status);
    g1_log_hash_footprint(hash, state.footprint);
    g1_log_hash_ik_state(hash, state.ik);
    g1_log_hash_frame_result(hash, state.ik_frame);
    g1_log_hash_pose_clearance(hash, state.ik_clearance);
    g1_log_hash_pose_clearance(hash, state.ik_candidate_clearance);
    G1_LOG_HASH_VALUE(ik_candidate_clearance_status);
    G1_LOG_HASH_VALUE(ik_candidate_rejected);
    g1_log_hash_support(hash, state.support);
    g1_log_hash_support_observation(
        hash, state.support_observation_now);
    G1_LOG_HASH_VALUE(traversal_speed_scale);
    G1_LOG_HASH_VALUE(traversal_speed_scale_velocity);
    G1_LOG_HASH_VALUE(blocked);
    G1_LOG_HASH_VALUE(walkability_class);
    G1_LOG_HASH_VALUE(blocked_distance);
    G1_LOG_HASH_VALUE(blocked_point);
    G1_LOG_HASH_VALUE(route_index);
    G1_LOG_HASH_VALUE(route_waypoint);
    G1_LOG_HASH_VALUE(route_frames);
    G1_LOG_HASH_VALUE(camera_azimuth);
    G1_LOG_HASH_VALUE(camera_altitude);
    G1_LOG_HASH_VALUE(camera_distance);
    G1_LOG_HASH_VALUE(searched);
    G1_LOG_HASH_VALUE(transitioned);
    G1_LOG_HASH_VALUE(incumbent_cost);
    G1_LOG_HASH_VALUE(selected_cost);
    G1_LOG_HASH_VALUE(selected_terrain_error);
    G1_LOG_HASH_VALUE(adjustment_xz);
    G1_LOG_HASH_VALUE(adjustment_y);
    G1_LOG_HASH_VALUE(clamp_xz);
    G1_LOG_HASH_VALUE(clamp_y);
#undef G1_LOG_HASH_ARRAY
#undef G1_LOG_HASH_VALUE
    return hash;
}

static void g1_log_landing(
    motion_match_landing_diagnostic& output,
    const G1FootprintFootObservation& foot)
{
    output.expected = foot.landing_expected;
    output.patch_ready = foot.landing_patch_ready;
    output.sample = foot.landing_sample;
    output.surface_status =
        g1_log_surface_status_name(foot.predicted_landing_surface_status);
    output.walkability_class = foot.predicted_landing_walkability_class;
    output.center = foot.predicted_landing_sole_center;
    output.height = foot.predicted_landing_surface.height;
    output.normal = foot.predicted_landing_surface.normal;
    output.patch_maximum_residual =
        foot.landing_patch_maximum_residual_m;
}

static void g1_log_swing(
    motion_match_swing_diagnostic& output,
    const G1SwingSelectionDiagnostic& selection)
{
    output.candidates_evaluated = selection.candidates_evaluated;
    output.selected_index = selection.selected_index;
    output.selected_lift_bits = selection.selected.lift_bits;
    output.materialized_command_y_bits =
        selection.selected.materialized_command_y_bits;
    g1_log_sphere_bits_hex(
        output.actual_sphere_center_bits_hex,
        selection.selected.actual_sphere_center_bits);
    output.selected_clearance_status =
        g1_clearance_status_name(selection.selected.clearance_status);
    output.selected_controller_constraints_passed =
        selection.selected.controller_constraints_passed;
    output.selected_clearance_certified =
        selection.selected.clearance_certified;
    output.lower_margin = selection.selected.lower_margin_m;
    output.witness_upper_margin =
        selection.selected.witness_upper_margin_m;
    g1_log_clearance_work(
        output.selected_work, selection.selected.clearance_work);
    g1_log_clearance_work(
        output.total_work, selection.total_clearance_work);
}

static void g1_log_canonical_disabled_ik(
    motion_match_ik_diagnostic& output)
{
    output = motion_match_ik_diagnostic{};
}

static bool g1_log_accepted_ik(
    motion_match_ik_diagnostic& output,
    const g1_controller_state& state,
    char* error,
    int error_capacity)
{
    g1_log_canonical_disabled_ik(output);
    output.applied = state.ik_frame.applied;
    output.safe_stop_requested = state.ik_frame.safe_stop_requested;
    output.stop_reason = g1_ik_stop_reason_name(state.ik_frame.stop_reason);
    output.max_correction = state.ik_frame.max_correction_radians;
    output.actual_simulation_speed = ::sqrtf(
        state.simulation_velocity.x * state.simulation_velocity.x +
        state.simulation_velocity.z * state.simulation_velocity.z);
    output.candidate_rejected = state.ik_candidate_rejected;
    output.candidate_clearance_status =
        g1_clearance_status_name(state.ik_candidate_clearance_status);
    if (state.ik_candidate_clearance_status == G1ClearanceOk) {
        output.candidate_toe_clearance[0] =
            state.ik_candidate_clearance.left.toe.lower_bound_m;
        output.candidate_foot_clearance[0] =
            state.ik_candidate_clearance.left.foot.lower_bound_m;
        output.candidate_toe_clearance[1] =
            state.ik_candidate_clearance.right.toe.lower_bound_m;
        output.candidate_foot_clearance[1] =
            state.ik_candidate_clearance.right.foot.lower_bound_m;
        output.candidate_minimum_clearance =
            state.ik_candidate_clearance.minimum.lower_bound_m;
    }

    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    const G1LegClearance* clearance[2] = {
        &state.ik_clearance.left, &state.ik_clearance.right
    };
    for (int foot = 0; foot < 2; ++foot) {
        const G1FootFrameResult& result = state.ik_frame.feet[foot];
        const G1FootLockState& lock = state.ik.feet[foot].lock;
        output.recorded_contact[foot] = result.recorded_contact;
        output.locked[foot] = result.target.locked;
        output.observed_lock_drift[foot] =
            result.target.horizontal_drift_m;
        output.reachable[foot] = result.position.reachable;
        output.contact_residual[foot] = result.position.applied
            ? result.position.contact_residual_m : 0.0f;
        output.target_height[foot] = result.target.surface.point.y;
        output.target_normal[foot] = result.target.desired_sole_normal;
        g1_log_swing(output.swing[foot], result.swing_selection);

        vec3 sole_center;
        if (!g1_ik_checked_physical_sole_centroid(
                sole_center,
                state.ik_global_bone_positions(configs[foot].contact),
                state.ik_global_bone_rotations(configs[foot].contact),
                configs[foot])) {
            if (error != nullptr && error_capacity > 0) {
                ::snprintf(
                    error, static_cast<std::size_t>(error_capacity),
                    "G1 log could not materialize accepted sole center");
            }
            return false;
        }
        if (lock.locked) {
            const float dx = sole_center.x - lock.lock_point.x;
            const float dz = sole_center.z - lock.lock_point.z;
            output.lock_drift[foot] = ::sqrtf(dx * dx + dz * dz);
        }
        const vec3 sole_normal = quat_mul_vec3(
            state.ik_global_bone_rotations(configs[foot].contact),
            configs[foot].sole_normal_local);
        output.sole_normal_alignment[foot] = dot(
            sole_normal, result.target.desired_sole_normal);

        output.knee_clearance[foot] = clearance[foot]->knee.lower_bound_m;
        output.ankle_clearance[foot] = clearance[foot]->ankle.lower_bound_m;
        output.toe_clearance[foot] = clearance[foot]->toe.lower_bound_m;
        output.foot_clearance[foot] = clearance[foot]->foot.lower_bound_m;
        output.shin_clearance[foot] = clearance[foot]->shin.lower_bound_m;
        output.thigh_clearance[foot] = clearance[foot]->thigh.lower_bound_m;
    }
    output.hips_clearance = state.ik_clearance.hips.lower_bound_m;
    output.minimum_clearance = state.ik_clearance.minimum.lower_bound_m;
    return true;
}

static void g1_log_rejected_foot(
    motion_match_rejected_foot_diagnostic& output,
    const G1FrameRejectionDiagnostic& rejection,
    int foot)
{
    if (rejection.attempted_footprint_available) {
        g1_log_landing(
            output.landing,
            rejection.attempted_footprint.feet[foot]);
    }
    if (rejection.attempted_ik_available) {
        const G1FootFrameResult& result = rejection.ik_frame.feet[foot];
        output.target = result.target.sole_center;
        output.target_normal = result.target.desired_sole_normal;
        output.reachable = result.position.reachable;
        output.correction_limited =
            result.position.correction_limited ||
            result.orientation.correction_limited;
        output.selected_clearance_status = g1_clearance_status_name(
            result.swing_selection.selected.clearance_status);
        output.selected_lower_margin =
            result.swing_selection.selected.lower_margin_m;
        output.selected_witness_upper =
            result.swing_selection.selected.witness_upper_margin_m;
    }
}

static void g1_log_directional(
    motion_match_directional_diagnostic& output,
    const g1_controller_state& state,
    const G1FramePublication& publication)
{
    output = motion_match_directional_diagnostic{};
    output.requested_velocity =
        publication.requested_intent.requested_velocity;
    output.applied_velocity = state.command.applied_velocity;
    g1_log_quat_bits_hex(
        output.desired_heading_bits_hex,
        publication.requested_intent.desired_heading);
    g1_log_predicted_heading_bits_hex(
        output.predicted_heading_bits_hex,
        state.command.predicted_desired_headings);
    output.simulation_heading_error_deg = g1_log_heading_error_degrees(
        publication.requested_intent.desired_heading,
        state.simulation_rotation);
    output.rendered_heading_error_deg = g1_log_heading_error_degrees(
        publication.requested_intent.desired_heading,
        state.ik_global_bone_rotations(0));

    output.footprint_status =
        g1_log_footprint_status_name(state.footprint_status);
    output.footprint_blocked = state.footprint.blocked;
    output.footprint_blocked_reason =
        ::walkability_reason_name(state.footprint.blocked_reason);
    output.footprint_root_height = state.footprint.root_surface.height;
    for (int foot = 0; foot < 2; ++foot) {
        const G1FootprintFootObservation& observed =
            state.footprint.feet[foot];
        output.footprint_min_height[foot] =
            observed.corridor_minimum_height;
        output.footprint_max_height[foot] =
            observed.corridor_maximum_height;
        output.maximum_root_split[foot] = observed.maximum_root_split_m;
        output.footprint_multilevel[foot] = observed.multilevel;
        g1_log_landing(output.landing[foot], observed);
    }
    output.footprint_sweeps = state.footprint.work.sweeps;
    output.footprint_surface_queries =
        state.footprint.work.surface_queries;
    output.footprint_node_visits = state.footprint.work.node_visits;

    const G1FrameRejectionDiagnostic& rejection = publication.rejection;
    output.frame_rejected = rejection.rejected;
    output.frame_rejection_stage =
        g1_frame_rejection_stage_name(rejection.stage);
    output.ik_safe_stop_latched = publication.ik_safe_stop_latched;
    output.rejected_attempted_footprint_available =
        rejection.attempted_footprint_available;
    output.rejected_attempted_ik_available =
        rejection.attempted_ik_available;
    output.rejected_stop_reason =
        g1_ik_stop_reason_name(rejection.stop_reason);
    output.rejected_attempted_pose_available =
        rejection.attempted_pose_available;
    output.rejected_pose_status =
        g1_clearance_status_name(rejection.pose_status);
    if (rejection.attempted_pose_available) {
        output.rejected_pose_minimum_clearance =
            rejection.pose_clearance.minimum.lower_bound_m;
    }
    g1_log_rejected_foot(output.rejected[0], rejection, 0);
    g1_log_rejected_foot(output.rejected[1], rejection, 1);
    ::snprintf(
        output.accepted_state_digest_hex,
        sizeof(output.accepted_state_digest_hex),
        "%016llx",
        static_cast<unsigned long long>(
            g1_log_accepted_state_digest(state)));
}

static bool g1_build_accepted_log_row(
    motion_match_log_row& log_row,
    const g1_controller_state& accepted_state,
    const G1FrameAcceptedDiagnostic& accepted_diagnostic,
    const G1FramePublication& publication,
    const G1AcceptedLogContext& log_context,
    char* error,
    int error_capacity)
{
    const bool canonical_rejection_baseline = !accepted_diagnostic.ready;
    if (canonical_rejection_baseline &&
        !publication.rejection.rejected) {
        return ::scene_error(
            error,
            error_capacity,
            "G1 accepted log diagnostic is unready without a rejection");
    }
    const int effective_query_database_frame =
        canonical_rejection_baseline
            ? accepted_state.frame_index
            : accepted_diagnostic.query_database_frame;
    const int effective_selected_database_frame =
        canonical_rejection_baseline
            ? accepted_state.frame_index
            : accepted_diagnostic.selected_database_frame;
    int range = 0;
    int query_range = 0;
    int source_range = 0;
    int source_index = 0;
    int database_frame_matches = 0;
    int query_frame_matches = 0;
    int selected_frame_matches = 0;
    if (log_context.source_count <= 0 ||
        log_context.sources == nullptr ||
        log_context.source_names == nullptr ||
        log_context.source_terrains == nullptr) {
        return ::scene_error(
            error,
            error_capacity,
            "G1 accepted log source metadata is unavailable");
    }
    for (int index = 0; index < log_context.source_count; ++index) {
        if (accepted_state.frame_index >=
                log_context.sources[index].range_start &&
            accepted_state.frame_index <
                log_context.sources[index].range_stop) {
            range = index;
            source_index = index;
            ++database_frame_matches;
        }
        if (effective_query_database_frame >=
                log_context.sources[index].range_start &&
            effective_query_database_frame <
                log_context.sources[index].range_stop) {
            query_range = index;
            ++query_frame_matches;
        }
        if (effective_selected_database_frame >=
                log_context.sources[index].range_start &&
            effective_selected_database_frame <
                log_context.sources[index].range_stop) {
            source_range = index;
            ++selected_frame_matches;
        }
    }
    if (database_frame_matches != 1 || query_frame_matches != 1 ||
        selected_frame_matches != 1) {
        return ::scene_error(
            error,
            error_capacity,
            "G1 accepted log source provenance is not unique "
            "(database=%d query=%d selected=%d)",
            database_frame_matches,
            query_frame_matches,
            selected_frame_matches);
    }
    if ((!canonical_rejection_baseline &&
         accepted_diagnostic.query_range != query_range) ||
        range != source_range) {
        return ::scene_error(
            error,
            error_capacity,
            "G1 accepted log source provenance is inconsistent");
    }
    log_row.frame = publication.presentation_frame;
    log_row.fixed_dt = log_context.fixed_dt;
    log_row.scene_id =
        log_context.scene_ids[*log_context.active_scene_index];
    log_row.mode = log_context.mode;
    log_row.route = log_context.route;
    log_row.query_bits_hex = log_context.empty_query_bits;
    log_row.query_database_frame = effective_query_database_frame;
    log_row.query_range = canonical_rejection_baseline
        ? query_range
        : accepted_diagnostic.query_range;
    log_row.selected_database_frame =
        effective_selected_database_frame;
    log_row.database_frame = accepted_state.frame_index;
    log_row.range = range;
    log_row.source_range = source_range;
    log_row.searched = accepted_state.searched;
    log_row.transitioned = accepted_state.transitioned;
    log_row.incumbent_cost = accepted_state.incumbent_cost;
    log_row.selected_cost = accepted_state.selected_cost;
    log_row.selected_terrain_error = accepted_state.selected_terrain_error;
    log_row.effective_terrain_weight =
        accepted_diagnostic.effective_terrain_weight;
    for (int index = 0; index < 4; ++index) {
        log_row.terrain[index] =
            accepted_diagnostic.terrain_query.values[index];
        log_row.terrain_points[index] =
            accepted_diagnostic.terrain_query.points[index];
    }
    log_row.raw_selected = accepted_diagnostic.raw_selected;
    log_row.inertialized = accepted_diagnostic.inertialized;
    log_row.rendered = accepted_diagnostic.rendered;
    log_row.hips_inertial_offset_y =
        accepted_diagnostic.inertialized.hips_y -
        accepted_diagnostic.raw_selected.hips_y;
    log_row.runtime_root_surface_height =
        accepted_state.support_observation_now.runtime_height[0];
    log_row.runtime_left_toe_surface_height =
        accepted_state.support_observation_now.runtime_height[1];
    log_row.runtime_right_toe_surface_height =
        accepted_state.support_observation_now.runtime_height[2];
    log_row.adjustment_xz = accepted_state.adjustment_xz;
    log_row.adjustment_y = accepted_state.adjustment_y;
    log_row.clamp_xz = accepted_state.clamp_xz;
    log_row.clamp_y = accepted_state.clamp_y;
    log_row.matching_enabled = accepted_diagnostic.matching_enabled;
    log_row.adjustment_enabled = accepted_diagnostic.adjustment_enabled;
    log_row.clamping_enabled = accepted_diagnostic.clamping_enabled;
    log_row.support_retargeting_enabled = true;
    log_row.ik_enabled = log_context.ik_enabled;
    log_row.source_name = log_context.source_names[source_index];
    log_row.source_terrain = log_context.source_terrains[source_index];
    log_row.source_index = source_index;
    log_row.continuation_cost = accepted_state.incumbent_cost;
    log_row.source_root_height =
        accepted_state.support_observation_now.source_height[0];
    log_row.source_left_toe_height =
        accepted_state.support_observation_now.source_height[1];
    log_row.source_right_toe_height =
        accepted_state.support_observation_now.source_height[2];
    log_row.runtime_support_root_height =
        accepted_state.support_observation_now.runtime_height[0];
    log_row.runtime_support_left_toe_height =
        accepted_state.support_observation_now.runtime_height[1];
    log_row.runtime_support_right_toe_height =
        accepted_state.support_observation_now.runtime_height[2];
    log_row.support_root_delta =
        accepted_state.support_observation_now.delta[0];
    log_row.support_left_toe_delta =
        accepted_state.support_observation_now.delta[1];
    log_row.support_right_toe_delta =
        accepted_state.support_observation_now.delta[2];
    log_row.support_height = accepted_state.support.height;
    log_row.support_velocity = accepted_state.support.velocity;
    log_row.support_source =
        ::support_source_name(accepted_state.support.source);
    log_row.airborne_frames = accepted_state.support.airborne_frames;
    log_row.left_contact =
        accepted_state.support_observation_now.contact[0];
    log_row.right_contact =
        accepted_state.support_observation_now.contact[1];
    log_row.support_retargeted_hips_y =
        accepted_diagnostic.support_retargeted.hips_y;
    log_row.ik_adjusted_hips_y = accepted_diagnostic.rendered.hips_y;
    log_row.simulation_x = accepted_state.simulation_position.x;
    log_row.simulation_z = accepted_state.simulation_position.z;
    log_row.walkability_class = accepted_state.walkability_class;
    log_row.blocked = accepted_diagnostic.traversal.blocked;
    log_row.blocked_reason =
        ::walkability_reason_name(accepted_diagnostic.traversal.reason);
    log_row.blocked_distance = accepted_diagnostic.traversal.distance;
    log_row.blocked_point_x = accepted_diagnostic.traversal.point.x;
    log_row.blocked_point_z = accepted_diagnostic.traversal.point.z;
    log_row.commanded_speed = accepted_diagnostic.traversal.commanded_speed;
    log_row.applied_speed = accepted_diagnostic.traversal.applied_speed;
    log_row.route_waypoint = accepted_diagnostic.route.waypoint;
    log_row.route_complete = accepted_diagnostic.route.complete;
    log_row.route_target_height = log_context.route_target_height;
    log_row.scene_generation = *log_context.scene_generation;
    log_row.scene_frame = accepted_diagnostic.scene_frame;
    log_row.scene_reset_count = *log_context.scene_reset_count;
    log_row.scene_switch_failed = *log_context.scene_switch_failed;
    log_row.motion_pack_load_count = *log_context.motion_pack_load_count;
    log_row.model_load_count = *log_context.model_load_count;
    log_row.model_unload_count = *log_context.model_unload_count;
    log_row.live_model_count =
        *log_context.model_load_count - *log_context.model_unload_count;
    return true;
}

static bool g1_build_task7_log_suffix(
    motion_match_log_row& log_row,
    const g1_controller_state& accepted_state,
    const G1FrameAcceptedDiagnostic& accepted_diagnostic,
    const G1FramePublication& publication,
    bool ik_enabled,
    char* error,
    int error_capacity)
{
    for (int dimension = 0; dimension < 31; ++dimension) {
        ::snprintf(
            log_row.query_bits_storage + dimension * 8,
            9,
            "%08x",
            (unsigned)terrain_float_bits(
                accepted_diagnostic.query[dimension]));
    }
    log_row.query_bits_storage[31 * 8] = '\0';
    log_row.query_bits_hex = log_row.query_bits_storage;
    if (ik_enabled) {
        if (!g1_log_accepted_ik(
                log_row.ik,
                accepted_state,
                error,
                error_capacity)) {
            return false;
        }
    } else {
        g1_log_canonical_disabled_ik(log_row.ik);
    }
    g1_log_directional(
        log_row.directional,
        accepted_state,
        publication);
    return true;
}

struct G1SceneLoader
{
    const char* terrain_directory = nullptr;
    const motion_pack_manifest* manifest = nullptr;
    const scene_catalog* catalog = nullptr;

    bool operator()(
        scene_pack& output,
        int index,
        char* error,
        int error_capacity) const
    {
        return ::scene_pack_load(
            output,
            terrain_directory,
            *manifest,
            *catalog,
            index,
            error,
            error_capacity);
    }
};

struct G1ModelLoader
{
    int* load_count = nullptr;
    int* scene_generation = nullptr;
    int* scene_reset_count = nullptr;

    scene_model_load_result operator()(
        Model& output,
        const char* path,
        char* error,
        int error_capacity) const
    {
        output = ::LoadModel(path);
        const bool allocated = output.meshes != nullptr ||
            output.materials != nullptr ||
            output.meshMaterial != nullptr || output.bones != nullptr ||
            output.bindPose != nullptr;
        if (allocated) ++*load_count;
        const bool ready = ::IsModelReady(output) && output.meshCount > 0;
        if (ready && allocated && scene_generation != nullptr &&
            scene_reset_count != nullptr) {
            ++*scene_generation;
            ++*scene_reset_count;
        }
        if (!ready) {
            ::scene_error(
                error,
                error_capacity,
                "%s: Raylib model is not ready",
                path);
        }
        return scene_model_load_result{allocated, ready};
    }
};

struct G1ModelUnloader
{
    int* unload_count = nullptr;

    void operator()(Model& model) const
    {
        const bool allocated = model.meshes != nullptr ||
            model.materials != nullptr || model.meshMaterial != nullptr ||
            model.bones != nullptr || model.bindPose != nullptr;
        if (allocated) {
            ::UnloadModel(model);
            ++*unload_count;
        }
        model = Model{};
    }
};

struct G1SceneResetContext
{
    scene_pack* active_scene = nullptr;
    Model* terrain_model = nullptr;
    const database* db = nullptr;
    const terrain_support_set* support = nullptr;
    const G1FrameResetConfig* reset_config = nullptr;
    G1SceneLoader scene_loader;
    G1ModelLoader model_loader;
    G1ModelUnloader model_unloader;
};

static bool g1_apply_pending_scene_reset(
    const int requested_scene_index,
    int& active_scene_index,
    G1FrameRuntime& frame_runtime,
    G1SceneResetContext& scene_reset_context,
    char* error,
    int error_capacity)
{
    if (requested_scene_index == active_scene_index) {
        return ::scene_reset_current(
            frame_runtime,
            *scene_reset_context.db,
            *scene_reset_context.support,
            *scene_reset_context.active_scene,
            *scene_reset_context.reset_config,
            error,
            error_capacity);
    }
    return ::scene_switch_transaction(
        *scene_reset_context.active_scene,
        frame_runtime,
        *scene_reset_context.terrain_model,
        active_scene_index,
        requested_scene_index,
        *scene_reset_context.db,
        *scene_reset_context.support,
        *scene_reset_context.reset_config,
        scene_reset_context.scene_loader,
        scene_reset_context.model_loader,
        scene_reset_context.model_unloader,
        error,
        error_capacity);
}

int main(int argc, char** argv)
{
    char startup_error[512] = {};
    float parsed_initial_search_time = 0.10f;
    const bool initial_search_time_ok = ::g1_parse_search_time(
        parsed_initial_search_time, ::getenv("MM_SEARCHT"), startup_error,
        static_cast<int>(sizeof(startup_error)));
    if (!initial_search_time_ok) { ::controlled_runtime_error(startup_error); return 1; }
    bool parsed_ik_enabled = false;
    const bool ik_enabled_ok = ::g1_parse_ik_enabled(
        parsed_ik_enabled, ::getenv("MM_IK"), startup_error,
        static_cast<int>(sizeof(startup_error)));
    if (!ik_enabled_ok) { ::controlled_runtime_error(startup_error); return 1; }
    float parsed_inertialize_blending_halflife = 0.10f;
    const bool inertialize_halflife_ok = ::g1_parse_halflife(
        parsed_inertialize_blending_halflife, ::getenv("MM_HALFLIFE"),
        startup_error, static_cast<int>(sizeof(startup_error)));
    if (!inertialize_halflife_ok) { ::controlled_runtime_error(startup_error); return 1; }
    float parsed_simulation_rotation_halflife = 0.27f;
    const bool simulation_halflife_ok = ::g1_parse_halflife(
        parsed_simulation_rotation_halflife, ::getenv("MM_SIMROT_HL"),
        startup_error, static_cast<int>(sizeof(startup_error)));
    if (!simulation_halflife_ok) { ::controlled_runtime_error(startup_error); return 1; }
    bool parsed_desired_strafe = false;
    const bool desired_strafe_ok = ::g1_parse_strafe_enabled(
        parsed_desired_strafe, ::getenv("MM_STRAFE"), startup_error,
        static_cast<int>(sizeof(startup_error)));
    if (!desired_strafe_ok) { ::controlled_runtime_error(startup_error); return 1; }
    const char* const candidate_audit_environment =
        ::getenv("MM_CANDIDATE_AUDIT");
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM) && defined(G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM)
    const char* const candidate_trace_environment =
        ::getenv("MM_CANDIDATE_TRACE");
#endif
#if defined(G1_FRAME_TRANSACTION_BENCHMARK)
    const char* const transaction_timings_environment =
        ::getenv("MM_TRANSACTION_TIMINGS");
#endif
    const G1ProcessConfig process_config{
        parsed_initial_search_time, parsed_ik_enabled,
        parsed_inertialize_blending_halflife,
        parsed_simulation_rotation_halflife, parsed_desired_strafe};

    char artifact_error[512] = {};
    G1ArgumentConfig parsed_test_config;
    if (!::g1_parse_arguments(
            parsed_test_config,
            argc,
            argv,
            artifact_error,
            static_cast<int>(sizeof(artifact_error)))) {
        ::controlled_runtime_error(artifact_error);
        return 2;
    }
    const G1ArgumentConfig test_config = parsed_test_config;
    G1CandidateAuditConfig candidate_audit_config;
    if (!::g1_candidate_audit_config_parse(
            candidate_audit_config,
            candidate_audit_environment,
            test_config.mode,
            test_config.frame_limit,
            test_config.mode_name.c_str(),
            test_config.route.c_str(),
            test_config.test_heading.empty()
                ? nullptr
                : test_config.test_heading.c_str(),
            artifact_error,
            static_cast<int>(sizeof(artifact_error)))) {
        ::controlled_runtime_error(artifact_error);
        return 2;
    }
    const char* terrain_directory = test_config.terrain_directory.c_str();

    motion_pack_manifest motion_manifest;
    if (!::motion_manifest_load_and_verify(
            motion_manifest,
            terrain_directory,
            artifact_error,
            static_cast<int>(sizeof(artifact_error)))) {
        ::controlled_runtime_error(artifact_error);
        return 2;
    }

    std::string database_path;
    std::string feature_path;
    std::string support_path;
    if (!::scene_join(
            database_path,
            terrain_directory,
            motion_manifest.database.path,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))) ||
        !::scene_join(
            feature_path,
            terrain_directory,
            motion_manifest.terrain_features.path,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))) ||
        !::scene_join(
            support_path,
            terrain_directory,
            motion_manifest.terrain_support.path,
            artifact_error,
            static_cast<int>(sizeof(artifact_error)))) {
        ::controlled_runtime_error(artifact_error);
        return 2;
    }

    terrain_feature_set terrain_rows;
    if (!::terrain_features_load(
            terrain_rows,
            feature_path.c_str(),
            artifact_error,
            static_cast<int>(sizeof(artifact_error)))) {
        ::controlled_runtime_error(artifact_error);
        return 2;
    }
    database db;
    ::database_load(db, database_path.c_str());
    if (db.nframes() <= 0 || db.nbones() != G1_BoneCount ||
        terrain_rows.values.rows != db.nframes() ||
        terrain_rows.values.cols != 4) {
        ::controlled_runtime_error(
            artifact_error[0] != '\0'
                ? artifact_error
                : "database/terrain feature shape mismatch");
        return 2;
    }
    db.terrain_features = terrain_rows.values;
    if (!g1_skeleton_validate(
            db,
            artifact_error,
            static_cast<int>(sizeof(artifact_error)))) {
        ::controlled_runtime_error(artifact_error);
        return 2;
    }
    if (!g1_leg_configs_validate(
            db,
            artifact_error,
            static_cast<int>(sizeof(artifact_error)))) {
        ::controlled_runtime_error(artifact_error);
        return 2;
    }
    ::database_build_matching_features(
        db,
        0.75f,
        1.0f,
        1.0f,
        1.0f,
        1.5f,
        G1_LeftAnkle,
        G1_RightAnkle,
        G1_Hips,
        test_config.terrain_weight);
    if (!::motion_manifest_validate_database(
            motion_manifest,
            db,
            artifact_error,
            static_cast<int>(sizeof(artifact_error)))) {
        ::controlled_runtime_error(artifact_error);
        return 2;
    }

    scene_catalog catalog;
    if (!::scene_catalog_load(
            catalog,
            terrain_directory,
            motion_manifest,
            artifact_error,
            static_cast<int>(sizeof(artifact_error)))) {
        ::controlled_runtime_error(artifact_error);
        return 2;
    }
    const char* requested_scene = test_config.terrain_scene.empty()
        ? catalog.default_scene_id.c_str()
        : test_config.terrain_scene.c_str();
    int active_scene_index = ::scene_catalog_find(catalog, requested_scene);
    if (active_scene_index < 0) {
        ::scene_error(
            artifact_error,
            static_cast<int>(sizeof(artifact_error)),
            "unknown terrain scene '%s'",
            requested_scene);
        ::controlled_runtime_error(artifact_error);
        return 2;
    }
    scene_pack active_scene;
    if (!::scene_pack_load(
            active_scene,
            terrain_directory,
            motion_manifest,
            catalog,
            active_scene_index,
            artifact_error,
            static_cast<int>(sizeof(artifact_error)))) {
        ::controlled_runtime_error(artifact_error);
        return 2;
    }

    terrain_support_set support_rows;
    if (!::terrain_support_load(
            support_rows,
            support_path.c_str(),
            db.nframes(),
            artifact_error,
            static_cast<int>(sizeof(artifact_error)))) {
        ::controlled_runtime_error(artifact_error);
        return 2;
    }

    G1TestHeadingOverride test_heading;
    const char* heading_text = test_config.test_heading.empty()
        ? nullptr
        : test_config.test_heading.c_str();
    if (!::g1_test_heading_override_parse(
            test_heading,
            heading_text,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))) ||
        (test_heading.active && test_config.mode != G1_TestRoute)) {
        ::controlled_runtime_error(
            artifact_error[0] != '\0'
                ? artifact_error
                : "--test-heading is valid only for route mode");
        return 2;
    }
    const scene_route* configured_route =
        test_config.mode == G1_TestRoute
            ? ::scene_route_find(active_scene.metadata, test_config.route.c_str())
            : nullptr;
    if (test_config.mode == G1_TestRoute && configured_route == nullptr) {
        ::scene_error(
            artifact_error,
            static_cast<int>(sizeof(artifact_error)),
            "scene '%s' has no route '%s'",
            active_scene.metadata.id.c_str(),
            test_config.route.c_str());
        ::controlled_runtime_error(artifact_error);
        return 2;
    }
    const float route_target_height = configured_route != nullptr
        ? ::deterministic_route_target_height(
              *configured_route, active_scene.terrain)
        : 0.0f;

    G1FrameResetConfig reset_config_builder;
    reset_config_builder.route_mode = test_config.mode == G1_TestRoute;
    reset_config_builder.route_id = reset_config_builder.route_mode
        ? test_config.route.c_str()
        : nullptr;
    reset_config_builder.ik_enabled = process_config.ik_enabled;
    reset_config_builder.dt = 1.0f / 25.0f;
    reset_config_builder.trajectory_sample_time = 1.0f / 3.0f;
    reset_config_builder.initial_search_time =
        process_config.initial_search_time;
    const G1FrameResetConfig frame_reset_config = reset_config_builder;

    G1FrameRuntime frame_runtime;
    if (!::g1_frame_runtime_reset(
            frame_runtime,
            db,
            support_rows,
            active_scene,
            frame_reset_config,
            artifact_error,
            static_cast<int>(sizeof(artifact_error)))) {
        ::controlled_runtime_error(artifact_error);
        return 2;
    }

    ::SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
    ::InitWindow(
        1280,
        720,
        "G1 terrain motion matching - transactional IK");
    if (!::IsWindowReady()) {
        ::controlled_runtime_error("G1 terrain visualizer could not open a window");
        return 2;
    }
    ::SetTargetFPS(25);

    int model_load_count = 0;
    int model_unload_count = 0;
    const int motion_pack_load_count = 1;
    G1ModelLoader initial_model_loader{&model_load_count};
    G1ModelUnloader model_unloader{&model_unload_count};
    Model terrain_model = {};
    motion_match_log deterministic_log;
    G1CandidateAuditLog candidate_audit_log;
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM) && defined(G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM)
    G1CandidateTraceFile candidate_trace;
    G1CandidateCertificationTrace candidate_certification_trace;
#endif
#if defined(G1_FRAME_TRANSACTION_BENCHMARK)
    G1TransactionTimings transaction_timings;
#endif
    const char* deterministic_log_path = test_config.log_path.empty()
        ? nullptr
        : test_config.log_path.c_str();
    const char* cleanup_path = test_config.cleanup_log_path.empty()
        ? nullptr
        : test_config.cleanup_log_path.c_str();
    bool window_open = true;
    bool cleanup_complete = false;
    int controller_exit_code = 0;
    auto normal_cleanup = [&]()
    {
        if (cleanup_complete) return;
        cleanup_complete = true;
        const bool log_evidence_ok = deterministic_log.close(
            artifact_error,
            static_cast<int>(sizeof(artifact_error)));
        const bool log_closed = true;
        if (!log_evidence_ok) {
            ::controlled_runtime_error(artifact_error);
            controller_exit_code = 2;
        }
        const bool candidate_audit_evidence_ok =
            candidate_audit_log.close(
                artifact_error,
                static_cast<int>(sizeof(artifact_error)));
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM) && defined(G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM)
        ::g1_candidate_trace_close(candidate_trace);
#endif
#if defined(G1_FRAME_TRANSACTION_BENCHMARK)
        const bool transaction_timings_evidence_ok =
            transaction_timings.close(
                artifact_error,
                static_cast<int>(sizeof(artifact_error)));
#endif
        if (!candidate_audit_evidence_ok
#if defined(G1_FRAME_TRANSACTION_BENCHMARK)
            || !transaction_timings_evidence_ok
#endif
            ) {
            ::controlled_runtime_error(artifact_error);
            controller_exit_code = 2;
        }
        model_unloader(terrain_model);
        if (window_open) {
            ::CloseWindow();
            window_open = false;
        }
        const bool window_closed = !window_open;
        cleanup_report cleanup;
        cleanup.exit_code = controller_exit_code;
        cleanup.motion_pack_load_count = motion_pack_load_count;
        cleanup.model_load_count = model_load_count;
        cleanup.model_unload_count = model_unload_count;
        cleanup.log_closed = log_closed;
        cleanup.window_closed = window_closed;
        if (!::cleanup_report_write(
                cleanup_path,
                cleanup,
                artifact_error,
                static_cast<int>(sizeof(artifact_error)))) {
            ::controlled_runtime_error(artifact_error);
            controller_exit_code = 2;
        }
    };
    const scene_model_load_result initial_model = initial_model_loader(
        terrain_model,
        active_scene.mesh_path.c_str(),
        artifact_error,
        static_cast<int>(sizeof(artifact_error)));
    if (!initial_model.ready || !initial_model.allocated) {
        ::controlled_runtime_error(artifact_error);
        controller_exit_code = 2;
        normal_cleanup();
        return controller_exit_code;
    }

    if (!deterministic_log.open(
            deterministic_log_path,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))) ||
        !candidate_audit_log.open(
            candidate_audit_config,
            artifact_error,
            static_cast<int>(sizeof(artifact_error)))
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM) && defined(G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM)
        || (candidate_trace_environment != nullptr &&
            candidate_trace_environment[0] != '\0' &&
            !::g1_candidate_trace_open(
                candidate_trace,
                candidate_trace_environment,
                artifact_error,
                static_cast<int>(sizeof(artifact_error))))
#endif
#if defined(G1_FRAME_TRANSACTION_BENCHMARK)
        || !transaction_timings.open(
            transaction_timings_environment,
            artifact_error,
            static_cast<int>(sizeof(artifact_error)))
#endif
        ) {
        ::controlled_runtime_error(artifact_error);
        controller_exit_code = 2;
        normal_cleanup();
        return controller_exit_code;
    }

    std::vector<const char*> source_names;
    std::vector<const char*> source_terrains;
    source_names.reserve(motion_manifest.sources.size());
    source_terrains.reserve(motion_manifest.sources.size());
    for (const motion_source_record& source : motion_manifest.sources) {
        source_names.push_back(source.name.c_str());
        source_terrains.push_back(source.terrain_id.c_str());
    }
    std::vector<const char*> scene_ids;
    scene_ids.reserve(catalog.ids.size());
    for (const std::string& scene_id : catalog.ids) {
        scene_ids.push_back(scene_id.c_str());
    }

    int scene_generation = 0;
    int scene_reset_count = 1;
    bool scene_switch_failed = false;
    const G1AcceptedLogContext log_context{
        1.0f / 25.0f,
        test_config.mode_name.c_str(),
        test_config.route.c_str(),
        "",
        process_config.ik_enabled,
        &db,
        motion_manifest.sources.data(),
        source_names.data(),
        source_terrains.data(),
        static_cast<int>(motion_manifest.sources.size()),
        scene_ids.data(),
        &active_scene_index,
        route_target_height,
        &scene_generation,
        &scene_reset_count,
        &scene_switch_failed,
        &motion_pack_load_count,
        &model_load_count,
        &model_unload_count};

    G1SceneResetContext scene_reset_context;
    scene_reset_context.active_scene = &active_scene;
    scene_reset_context.terrain_model = &terrain_model;
    scene_reset_context.db = &db;
    scene_reset_context.support = &support_rows;
    scene_reset_context.reset_config = &frame_reset_config;
    scene_reset_context.scene_loader = G1SceneLoader{
        terrain_directory, &motion_manifest, &catalog};
    scene_reset_context.model_loader = G1ModelLoader{
        &model_load_count, &scene_generation, &scene_reset_count};
    scene_reset_context.model_unloader = model_unloader;

    G1FrameExternalInputs frame_external;
    frame_external.db = &db;
    frame_external.support = &support_rows;
    frame_external.scene = &active_scene;
    frame_external.route = configured_route;
    frame_external.heading_override = test_heading;
    frame_external.tuning.mode = test_config.mode;
    frame_external.tuning.frame_limit = test_config.frame_limit;
    frame_external.tuning.scene_dwell_frames =
        test_config.scene_dwell_frames;
    frame_external.tuning.dt = 1.0f / 25.0f;
    frame_external.tuning.trajectory_sample_time = 1.0f / 3.0f;
    frame_external.tuning.effective_terrain_weight =
        test_config.terrain_weight;

    Camera3D camera{};
    camera.up = Vector3{0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    const int scene_count = static_cast<int>(catalog.ids.size());
    int pending_scene_index = -1;
    bool pending_reset = false;
    bool controller_exit_requested = false;
    int presentation_frame = 0;
    motion_match_log_row log_row;

    while (!::WindowShouldClose() && !controller_exit_requested) {
        if (pending_reset) {
            const int requested_scene_index = pending_scene_index;
            const bool scene_reset_ok = ::g1_apply_pending_scene_reset(
                requested_scene_index, active_scene_index, frame_runtime,
                scene_reset_context,
                artifact_error, static_cast<int>(sizeof(artifact_error)));
            if (!scene_reset_ok) {
                ::controlled_runtime_error(artifact_error);
                controller_exit_code = 2;
                controller_exit_requested = true;
                break;
            }
            pending_reset = false;
            pending_scene_index = -1;
        }
        frame_external.scene = &active_scene;
        frame_external.route = test_config.mode == G1_TestRoute
            ? ::scene_route_find(
                  active_scene.metadata, test_config.route.c_str())
            : nullptr;
        const vec3 gamepadstick_left = ::gamepad_get_stick(GAMEPAD_STICK_LEFT);
        const vec3 gamepadstick_right = ::gamepad_get_stick(GAMEPAD_STICK_RIGHT);
        frame_external.input.move_stick = gamepadstick_left;
        frame_external.input.look_stick = gamepadstick_right;
        frame_external.input.gait_target =
            ::IsKeyDown(KEY_LEFT_SHIFT) ? 1.0f : 0.0f;
        frame_external.input.camera_zoom_axis = ::GetMouseWheelMove();
        float scripted_azimuth_delta = 0.0f;
#if defined(MM_DISCRETE)
        if (test_config.mode == G1_TestLive) {
            if (test_config.discrete_mode == 0) {
                if (presentation_frame == 120 ||
                    presentation_frame == 240) {
                    scripted_azimuth_delta = 0.5f * PIf;
                } else if (presentation_frame == 360) {
                    scripted_azimuth_delta = -0.5f * PIf;
                }
            } else if (test_config.discrete_mode == 1 &&
                       presentation_frame >= 60 &&
                       presentation_frame %
                               test_config.discrete_snap_frames == 0) {
                scripted_azimuth_delta =
                    (presentation_frame /
                             test_config.discrete_snap_frames %
                         2
                        ? -1.0f
                        : 1.0f) *
                    0.5f * PIf;
            } else if (test_config.discrete_mode == 2 &&
                       presentation_frame >= 60) {
                scripted_azimuth_delta = 2.0f / 25.0f;
            } else if (test_config.discrete_mode == 3 &&
                       presentation_frame >= 60 &&
                       presentation_frame %
                               test_config.discrete_snap_frames == 0) {
                scripted_azimuth_delta = PIf;
            }
        }
#endif
        frame_external.input.scripted_azimuth_delta =
            scripted_azimuth_delta;
        frame_external.input.presentation_frame = presentation_frame;
        ++presentation_frame;
        if (test_config.mode != G1_TestLive &&
            presentation_frame >= test_config.frame_limit) {
            controller_exit_requested = true;
        }
        frame_external.tuning.initial_search_time =
            process_config.initial_search_time;
        frame_external.tuning.ik_enabled = process_config.ik_enabled;
        frame_external.tuning.inertialize_blending_halflife =
            process_config.inertialize_blending_halflife;
        frame_external.tuning.simulation_rotation_halflife =
            process_config.simulation_rotation_halflife;
        frame_external.input.desired_strafe = process_config.desired_strafe;
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
        const G1FrameTransactionTestSeam* test_seam_pointer = nullptr;
#if defined(G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM)
        G1FrameTransactionTestSeam test_seam;
        if (candidate_trace.stream != nullptr) {
            test_seam.certification_trace = &candidate_certification_trace;
            test_seam_pointer = &test_seam;
        }
#endif
#endif
#if defined(G1_FRAME_TRANSACTION_BENCHMARK)
        const auto transaction_begin = std::chrono::steady_clock::now();
#endif
        const G1FrameTransactionStatus frame_status =
            ::g1_frame_transaction_run(frame_runtime,
            ::g1_controller_frame_stage_run,
            ::g1_recovery_candidates_build,
            frame_external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            test_seam_pointer,
#endif
            artifact_error,
            static_cast<int>(sizeof(artifact_error)));
#if defined(G1_FRAME_TRANSACTION_BENCHMARK)
        const auto transaction_end = std::chrono::steady_clock::now();
        transaction_timings.append(transaction_end - transaction_begin);
#endif
        bool transaction_failed =
            frame_status == G1FrameTransactionGlobalError;
#if defined(G1_FRAME_TRANSACTION_BENCHMARK)
        if (!transaction_timings.healthy(
                artifact_error,
                static_cast<int>(sizeof(artifact_error)))) {
            transaction_failed = true;
        }
#endif
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM) && defined(G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM)
        if (!transaction_failed && candidate_trace.stream != nullptr &&
            !::g1_candidate_trace_append_after_transaction(
                candidate_trace,
                static_cast<uint32_t>(
                    frame_external.input.presentation_frame),
                candidate_certification_trace,
                artifact_error,
                static_cast<int>(sizeof(artifact_error)))) {
            transaction_failed = true;
        }
#endif
        if (transaction_failed) {
            ::controlled_runtime_error(artifact_error);
            controller_exit_code = 2;
            controller_exit_requested = true;
            break;
        }
        const bool log_row_ok = ::g1_build_accepted_log_row(
            log_row, frame_runtime.accepted_state,
            frame_runtime.accepted_diagnostic, frame_runtime.publication,
            log_context, artifact_error,
            static_cast<int>(sizeof(artifact_error)));
        if (!log_row_ok) {
            ::controlled_runtime_error(artifact_error);
            controller_exit_code = 2;
            controller_exit_requested = true;
            break;
        }
        const bool log_suffix_ok = ::g1_build_task7_log_suffix(
            log_row, frame_runtime.accepted_state,
            frame_runtime.accepted_diagnostic, frame_runtime.publication,
            log_context.ik_enabled,
            artifact_error, static_cast<int>(sizeof(artifact_error)));
        if (!log_suffix_ok) {
            ::controlled_runtime_error(artifact_error);
            controller_exit_code = 2;
            controller_exit_requested = true;
            break;
        }
        const bool log_ok = deterministic_log.write(
            log_row, artifact_error,
            static_cast<int>(sizeof(artifact_error)));
        if (!log_ok) {
            ::controlled_runtime_error(artifact_error);
            controller_exit_code = 2;
            controller_exit_requested = true;
            break;
        }
        const bool candidate_audit_ok =
            candidate_audit_log.write_requested(
                db,
                frame_runtime.accepted_state,
                frame_runtime.accepted_diagnostic,
                frame_runtime.publication,
                frame_status,
                scene_ids[active_scene_index],
                candidate_audit_config.route,
                candidate_audit_config.heading,
                artifact_error,
                static_cast<int>(sizeof(artifact_error)));
        if (!candidate_audit_ok) {
            ::controlled_runtime_error(artifact_error);
            controller_exit_code = 2;
            controller_exit_requested = true;
            break;
        }
        if (frame_status == G1FrameTransactionAccepted &&
            test_config.mode == G1_TestSceneCycle &&
            frame_runtime.accepted_state.scene_frame >=
                test_config.scene_dwell_frames) {
            pending_scene_index = (active_scene_index + 1) % scene_count;
            pending_reset = true;
        }
        ::update_g1_camera_from_accepted(
            camera, frame_runtime.accepted_state.camera_azimuth,
            frame_runtime.accepted_state.camera_altitude,
            frame_runtime.accepted_state.camera_distance,
            frame_runtime.accepted_state.ik_global_bone_positions(0));
        ::BeginDrawing();
        ::ClearBackground(Color{18, 20, 25, 255});
        ::BeginMode3D(camera);
        ::DrawModel(terrain_model, Vector3{0.0f, 0.0f, 0.0f}, 1.0f, WHITE);
        ::draw_g1_skeleton(
            frame_runtime.accepted_state.ik_global_bone_positions,
            frame_runtime.accepted_state.ik_global_bone_rotations,
            frame_external.db->bone_parents);
        ::EndMode3D();
        ::DrawText("Transactional terrain IK", 20, 20, 20, RAYWHITE);
        ::EndDrawing();
    }

    normal_cleanup();
    return controller_exit_code;
}

#endif

static G1FrameStageOutcome g1_runner_fail(
    char* error,
    int error_capacity,
    const char* message)
{
    if (error != nullptr && error_capacity > 0) {
        ::snprintf(
            error,
            static_cast<size_t>(error_capacity),
            "%s",
            message);
    }
    return G1FrameStageGlobalError;
}

static float g1_runner_canonical_float(float value)
{
    return value == 0.0f ? 0.0f : value;
}

static vec3 g1_runner_canonical_vec3(vec3 value)
{
    return vec3(
        g1_runner_canonical_float(value.x),
        g1_runner_canonical_float(value.y),
        g1_runner_canonical_float(value.z));
}

static float g1_runner_fast_negexp(float value)
{
    return 1.0f /
        (1.0f + value + 0.48f * value * value +
         0.235f * value * value * value);
}

static float g1_runner_sqrt(float value)
{
    if (value <= 0.0f) return 0.0f;
    float estimate = value >= 1.0f ? value : 1.0f;
    for (int iteration = 0; iteration < 12; ++iteration) {
        estimate = 0.5f * (estimate + value / estimate);
    }
    float scaled = estimate;
    float unit = FLT_EPSILON;
    while (scaled >= 2.0f) {
        scaled *= 0.5f;
        unit *= 2.0f;
    }
    while (scaled < 1.0f) {
        scaled *= 2.0f;
        unit *= 0.5f;
    }
    float best = estimate;
    double best_error = static_cast<double>(best) * best - value;
    if (best_error < 0.0) best_error = -best_error;
    for (int step = -2; step <= 2; ++step) {
        const float candidate = estimate +
            static_cast<float>(step) * unit;
        double candidate_error =
            static_cast<double>(candidate) * candidate - value;
        if (candidate_error < 0.0) {
            candidate_error = -candidate_error;
        }
        if (candidate_error < best_error) {
            best = candidate;
            best_error = candidate_error;
        }
    }
    return best;
}

static float g1_runner_acos(float value)
{
    const float clamped = ::clampf(value, -1.0f, 1.0f);
    if (clamped <= -1.0f) return PIf;
    if (clamped >= 1.0f) return 0.0f;
    const float half_angle_form = 2.0f * ::atan2f(
        g1_runner_sqrt(1.0f - clamped),
        g1_runner_sqrt(1.0f + clamped));
    const float product_form = ::atan2f(
        g1_runner_sqrt(
            (1.0f - clamped) * (1.0f + clamped)),
        clamped);
    return 0.5f * (half_angle_form + product_form);
}

static quat g1_runner_quat_normalize(quat value)
{
    const float magnitude = g1_runner_sqrt(
        value.w * value.w + value.x * value.x +
        value.y * value.y + value.z * value.z);
    const float divisor = magnitude + 1.0e-8f;
    return quat(
        value.w / divisor,
        value.x / divisor,
        value.y / divisor,
        value.z / divisor);
}

static vec3 g1_runner_quat_to_scaled_axis(quat value)
{
    value = ::quat_abs(value);
    const vec3 vector = vec3(value.x, value.y, value.z);
    const float magnitude = ::length(vector);
    if (magnitude < 1.0e-8f) return 2.0f * vector;
    const float halfangle = g1_runner_acos(value.w);
    return 2.0f * (halfangle * (vector / magnitude));
}

static quat g1_runner_quat_from_scaled_axis(vec3 value)
{
    const vec3 half = value / 2.0f;
    const float halfangle = ::length(half);
    if (halfangle < 1.0e-8f) {
        return g1_runner_quat_normalize(
            quat(1.0f, half.x, half.y, half.z));
    }
    const quat trigonometry = ::quat_from_angle_axis(
        2.0f * halfangle, vec3(1.0f, 0.0f, 0.0f));
    const float scale = trigonometry.x / halfangle;
    return quat(
        trigonometry.w,
        scale * half.x,
        scale * half.y,
        scale * half.z);
}

static void g1_runner_decay_vec3(
    vec3& value,
    vec3& velocity,
    float halflife,
    float dt)
{
    const float damping = (4.0f * LN2f) / (halflife + 1.0e-5f);
    const float y = damping / 2.0f;
    const vec3 j1 = velocity + value * y;
    const float eydt = g1_runner_fast_negexp(y * dt);
    value = eydt * (value + j1 * dt);
    velocity = eydt * (velocity - j1 * y * dt);
}

static void g1_runner_decay_quat(
    quat& value,
    vec3& velocity,
    float halflife,
    float dt)
{
    const float damping = (4.0f * LN2f) / (halflife + 1.0e-5f);
    const float y = damping / 2.0f;
    const vec3 j0 = g1_runner_quat_to_scaled_axis(::quat_abs(value));
    const vec3 j1 = velocity + j0 * y;
    const float eydt = g1_runner_fast_negexp(y * dt);
    value = g1_runner_quat_from_scaled_axis(
        eydt * (j0 + j1 * dt));
    velocity = eydt * (velocity - j1 * y * dt);
}

static void g1_runner_spring_scalar(
    float& value,
    float& velocity,
    float goal,
    float halflife,
    float dt)
{
    const float damping = (4.0f * LN2f) / (halflife + 1.0e-5f);
    const float y = damping / 2.0f;
    const float j0 = value - goal;
    const float j1 = velocity + j0 * y;
    const float eydt = g1_runner_fast_negexp(y * dt);
    value = eydt * (j0 + j1 * dt) + goal;
    velocity = eydt * (velocity - j1 * y * dt);
}

static void g1_runner_spring_quat(
    quat& value,
    vec3& velocity,
    quat goal,
    float halflife,
    float dt)
{
    const float damping = (4.0f * LN2f) / (halflife + 1.0e-5f);
    const float y = damping / 2.0f;
    const vec3 j0 = g1_runner_quat_to_scaled_axis(
        ::quat_abs(::quat_mul(value, ::quat_inv(goal))));
    const vec3 j1 = velocity + j0 * y;
    const float eydt = g1_runner_fast_negexp(y * dt);
    value = ::quat_mul(
        g1_runner_quat_from_scaled_axis(
            eydt * (j0 + j1 * dt)),
        goal);
    velocity = eydt * (velocity - j1 * y * dt);
}

static void g1_runner_inertialize_vec3(
    vec3& output,
    vec3& output_velocity,
    vec3& offset,
    vec3& offset_velocity,
    vec3 input,
    vec3 input_velocity,
    float halflife,
    float dt)
{
    g1_runner_decay_vec3(offset, offset_velocity, halflife, dt);
    output = input + offset;
    output_velocity = input_velocity + offset_velocity;
}

static void g1_runner_inertialize_quat(
    quat& output,
    vec3& output_velocity,
    quat& offset,
    vec3& offset_velocity,
    quat input,
    vec3 input_velocity,
    float halflife,
    float dt)
{
    g1_runner_decay_quat(offset, offset_velocity, halflife, dt);
    output = ::quat_mul(offset, input);
    output_velocity = offset_velocity +
        ::quat_mul_vec3(offset, input_velocity);
}

static void g1_runner_transition_vec3(
    vec3& offset,
    vec3& offset_velocity,
    vec3 source,
    vec3 source_velocity,
    vec3 destination,
    vec3 destination_velocity)
{
    offset = source + offset - destination;
    offset_velocity = source_velocity + offset_velocity -
        destination_velocity;
}

static void g1_runner_transition_quat(
    quat& offset,
    vec3& offset_velocity,
    quat source,
    vec3 source_velocity,
    quat destination,
    vec3 destination_velocity)
{
    offset = ::quat_abs(::quat_mul(
        ::quat_mul(offset, source),
        ::quat_inv(destination)));
    offset_velocity = offset_velocity + source_velocity -
        destination_velocity;
}

static void g1_runner_pose_transition(
    g1_controller_state& state)
{
    state.transition_dst_position = state.bone_positions(0);
    state.transition_dst_rotation = state.bone_rotations(0);
    state.transition_src_position = state.trns_bone_positions(0);
    state.transition_src_rotation = state.trns_bone_rotations(0);
    const vec3 destination_velocity = ::quat_mul_vec3(
        state.transition_dst_rotation,
        ::quat_mul_vec3(
            ::quat_inv(state.transition_src_rotation),
            state.trns_bone_velocities(0)));
    const vec3 destination_angular_velocity = ::quat_mul_vec3(
        state.transition_dst_rotation,
        ::quat_mul_vec3(
            ::quat_inv(state.transition_src_rotation),
            state.trns_bone_angular_velocities(0)));
    g1_runner_transition_vec3(
        state.bone_offset_positions(0),
        state.bone_offset_velocities(0),
        state.bone_positions(0),
        state.bone_velocities(0),
        state.bone_positions(0),
        destination_velocity);
    g1_runner_transition_quat(
        state.bone_offset_rotations(0),
        state.bone_offset_angular_velocities(0),
        state.bone_rotations(0),
        state.bone_angular_velocities(0),
        state.bone_rotations(0),
        destination_angular_velocity);
    for (int bone = 1; bone < G1_BoneCount; ++bone) {
        g1_runner_transition_vec3(
            state.bone_offset_positions(bone),
            state.bone_offset_velocities(bone),
            state.curr_bone_positions(bone),
            state.curr_bone_velocities(bone),
            state.trns_bone_positions(bone),
            state.trns_bone_velocities(bone));
        g1_runner_transition_quat(
            state.bone_offset_rotations(bone),
            state.bone_offset_angular_velocities(bone),
            state.curr_bone_rotations(bone),
            state.curr_bone_angular_velocities(bone),
            state.trns_bone_rotations(bone),
            state.trns_bone_angular_velocities(bone));
    }
}

static void g1_runner_root_adjust(
    g1_controller_state& state,
    vec3 input_position,
    quat input_rotation)
{
    vec3& offset = state.bone_offset_positions(0);
    vec3& position = state.bone_positions(0);
    quat& rotation = state.bone_rotations(0);
    const vec3 position_difference = input_position - position;
    position = position_difference + position;
    state.transition_dst_position = position_difference +
        state.transition_dst_position;
    state.transition_src_position = state.transition_src_position +
        ::quat_mul_vec3(
            state.transition_src_rotation,
            ::quat_mul_vec3(
                ::quat_inv(state.transition_dst_rotation),
                position - offset - state.transition_dst_position));
    state.transition_dst_position = position;
    offset = vec3();
    const quat rotation_difference = g1_runner_quat_normalize(
        ::quat_mul_inv(input_rotation, rotation));
    rotation = ::quat_mul(rotation_difference, rotation);
    state.transition_dst_rotation = ::quat_mul(
        rotation_difference,
        state.transition_dst_rotation);
}

static vec3 g1_runner_desired_velocity(
    vec3 move_stick,
    float camera_azimuth,
    quat simulation_rotation,
    float forward_speed,
    float side_speed,
    float back_speed)
{
    const vec3 world_stick = ::quat_mul_vec3(
        ::quat_from_angle_axis(
            camera_azimuth, vec3(0.0f, 1.0f, 0.0f)),
        move_stick);
    const vec3 local_stick = ::quat_mul_vec3(
        ::quat_inv(simulation_rotation), world_stick);
    const vec3 scaled =
        (local_stick.z > 0.0f
             ? vec3(side_speed, 0.0f, forward_speed)
             : vec3(side_speed, 0.0f, back_speed)) *
        local_stick;
    return g1_runner_canonical_vec3(
        ::quat_mul_vec3(simulation_rotation, scaled));
}

static quat g1_runner_desired_heading(
    quat previous,
    vec3 move_stick,
    vec3 look_stick,
    float camera_azimuth,
    bool strafe,
    vec3 desired_velocity)
{
    const quat camera_rotation = ::quat_from_angle_axis(
        camera_azimuth, vec3(0.0f, 1.0f, 0.0f));
    if (strafe) {
        vec3 direction = ::quat_mul_vec3(
            camera_rotation, vec3(0.0f, 0.0f, -1.0f));
        if (::length(look_stick) > 0.01f) {
            direction = ::quat_mul_vec3(
                camera_rotation, ::normalize(look_stick));
        }
        return ::quat_from_angle_axis(
            ::atan2f(direction.x, direction.z),
            vec3(0.0f, 1.0f, 0.0f));
    }
    if (::length(move_stick) > 0.01f &&
        ::length(desired_velocity) > 0.01f) {
        const vec3 direction = ::normalize(desired_velocity);
        return ::quat_from_angle_axis(
            ::atan2f(direction.x, direction.z),
            vec3(0.0f, 1.0f, 0.0f));
    }
    return previous;
}

static int g1_runner_active_range(const database& db, int frame)
{
    for (int range = 0; range < db.range_starts.size; ++range) {
        if (frame >= db.range_starts(range) &&
            frame < db.range_stops(range)) {
            return range;
        }
    }
    return -1;
}

static int g1_runner_trajectory_clamp(
    const database& db,
    int frame,
    int offset)
{
    const int range = g1_runner_active_range(db, frame);
    if (range < 0) return frame;
    const int start = db.range_starts(range);
    const int stop = db.range_stops(range);
    return std::max(start, std::min(stop - 1, frame + offset));
}

static float g1_runner_database_cost(
    const database& db,
    int frame,
    const float query[31])
{
    float cost = 0.0f;
    for (int dimension = 0; dimension < 31; ++dimension) {
        const float normalized =
            (query[dimension] - db.features_offset(dimension)) /
            db.features_scale(dimension);
        const float difference =
            normalized - db.features(frame, dimension);
        cost += difference * difference;
    }
    return cost;
}

static float g1_runner_terrain_error(
    const database& db,
    int frame,
    const float query[31])
{
    float error = 0.0f;
    for (int feature = 0; feature < 4; ++feature) {
        const float difference = query[27 + feature] -
            db.terrain_features(frame, feature);
        error += difference * difference;
    }
    return error;
}

static motion_match_pose_diagnostic motion_match_pose_snapshot(
    const slice1d<vec3> global_positions,
    const heightfield& terrain)
{
    motion_match_pose_diagnostic output;
    output.hips_y = global_positions(G1_Hips).y;
    output.hips_clearance = output.hips_y - ::heightfield_sample_v2(
        terrain,
        global_positions(G1_Hips).x,
        global_positions(G1_Hips).z);
    output.left_toe_clearance = global_positions(G1_LeftToe).y -
        ::heightfield_sample_v2(
            terrain,
            global_positions(G1_LeftToe).x,
            global_positions(G1_LeftToe).z);
    output.right_toe_clearance = global_positions(G1_RightToe).y -
        ::heightfield_sample_v2(
            terrain,
            global_positions(G1_RightToe).x,
            global_positions(G1_RightToe).z);
    output.minimum_clearance = output.hips_clearance;
    const int probes[6] = {
        G1_LeftKnee, G1_RightKnee, G1_LeftAnkle,
        G1_RightAnkle, G1_LeftToe, G1_RightToe
    };
    for (int probe = 0; probe < 6; ++probe) {
        const vec3 point = global_positions(probes[probe]);
        const float clearance = point.y - ::heightfield_sample_v2(
            terrain, point.x, point.z);
        output.minimum_clearance = ::fminf(
            output.minimum_clearance, clearance);
    }
    return output;
}

static void g1_runner_predict_position(
    vec3& position,
    vec3& velocity,
    vec3& acceleration,
    vec3 desired_velocity,
    float halflife,
    float dt)
{
    const float damping = (4.0f * LN2f) / (halflife + 1.0e-5f);
    const float y = damping / 2.0f;
    const vec3 j0 = velocity - desired_velocity;
    const vec3 j1 = acceleration + j0 * y;
    const float eydt = g1_runner_fast_negexp(y * dt);
    const vec3 previous = position;
    position = eydt * (((-j1) / (y * y)) +
        ((-j0 - j1 * dt) / y)) +
        j1 / (y * y) + j0 / y +
        desired_velocity * dt + previous;
    velocity = eydt * (j0 + j1 * dt) + desired_velocity;
    acceleration = eydt * (acceleration - j1 * y * dt);
}

static void g1_runner_build_prediction(
    g1_controller_state& state,
    G1CommandIntent intent,
    vec3 applied_velocity,
    const G1FrameExternalInputs& external)
{
    vec3 predicted_position = state.simulation_position;
    vec3 predicted_velocity = state.simulation_velocity;
    vec3 predicted_acceleration = state.simulation_acceleration;
    quat predicted_rotation = state.simulation_rotation;
    vec3 predicted_angular_velocity =
        state.simulation_angular_velocity;
    for (int sample = 0;
         sample < G1CommandTrajectorySampleCount;
         ++sample) {
        if (sample > 0) {
            g1_runner_predict_position(
                predicted_position,
                predicted_velocity,
                predicted_acceleration,
                applied_velocity * external.tuning.future_speed_scale,
                external.tuning.simulation_velocity_halflife,
                external.tuning.trajectory_sample_time);
            g1_runner_spring_quat(
                predicted_rotation,
                predicted_angular_velocity,
                intent.desired_heading,
                external.tuning.simulation_rotation_halflife,
                external.tuning.trajectory_sample_time);
        }
        state.trajectory_desired_velocities(sample) = applied_velocity;
        state.trajectory_positions(sample) = predicted_position;
        state.trajectory_velocities(sample) = predicted_velocity;
        state.trajectory_accelerations(sample) = predicted_acceleration;
        state.trajectory_rotations(sample) = predicted_rotation;
        state.trajectory_angular_velocities(sample) =
            predicted_angular_velocity;
        state.trajectory_desired_rotations(sample) =
            intent.desired_heading;
        state.command.predicted_desired_velocities[sample] =
            state.trajectory_desired_velocities(sample);
        state.command.predicted_root_positions[sample] =
            state.trajectory_positions(sample);
        state.command.predicted_root_rotations[sample] =
            state.trajectory_rotations(sample);
        state.command.predicted_desired_headings[sample] =
            state.trajectory_desired_rotations(sample);
    }
    state.command.intent = intent;
    state.command.applied_velocity = applied_velocity;
}

[[gnu::noinline]] static void g1_runner_build_query(
    float query[31],
    terrain_centerline_snapshot& terrain_query,
    const g1_controller_state& state,
    const G1FrameExternalInputs& external)
{
    int offset = 0;
    const slice1d<float> features =
        external.db->features(state.frame_index);
    for (int dimension = 0; dimension < 15; ++dimension) {
        query[offset] = features(offset) *
            external.db->features_scale(offset) +
            external.db->features_offset(offset);
        ++offset;
    }
    for (int sample = 1; sample < 4; ++sample) {
        const vec3 local = ::quat_mul_vec3(
            ::quat_inv(state.bone_rotations(0)),
            state.trajectory_positions(sample) -
                state.bone_positions(0));
        query[offset++] = local.x;
        query[offset++] = local.z;
    }
    for (int sample = 1; sample < 4; ++sample) {
        const vec3 direction = ::quat_mul_vec3(
            ::quat_inv(state.bone_rotations(0)),
            ::quat_mul_vec3(
                state.trajectory_rotations(sample),
                vec3(0.0f, 0.0f, 1.0f)));
        query[offset++] = direction.x;
        query[offset++] = direction.z;
    }
    ::terrain_centerline_snapshot_compute_v2(
        terrain_query,
        external.scene->terrain,
        state.bone_positions(0),
        state.trajectory_positions,
        state.trajectory_rotations);
    for (int feature = 0; feature < 4; ++feature) {
        query[offset++] = terrain_query.values[feature];
    }
}

static bool g1_runner_support_update(
    support_frame_state& state,
    const support_observation& observation,
    bool source_frame_changed,
    float dt)
{
    support_frame_state next = state;
    if (!next.initialized) {
        next = {};
        next.height = observation.delta[0];
        next.nominal_height = observation.delta[0];
        next.initialized = true;
    }
    float target = next.nominal_height;
    support_source source = support_held;
    if (observation.contact[0] && observation.contact[1]) {
        next.airborne_frames = 0;
        target = 0.5f *
            (observation.delta[1] + observation.delta[2]);
        source = support_both;
    } else if (observation.contact[0]) {
        next.airborne_frames = 0;
        target = observation.delta[1];
        source = support_left;
    } else if (observation.contact[1]) {
        next.airborne_frames = 0;
        target = observation.delta[2];
        source = support_right;
    } else {
        ++next.airborne_frames;
        if (next.airborne_frames <= 2) {
            target = next.nominal_height;
            source = support_held;
        } else {
            target = observation.delta[0];
            source = support_airborne_root;
        }
    }
    const bool source_changed = source_frame_changed ||
        source != next.source;
    const float target_delta = target - next.nominal_height;
    const bool discontinuity = source_changed ||
        ::fabsf(target_delta) > 0.02f;
    const float target_velocity =
        source == support_held || discontinuity
            ? 0.0f
            : target_delta / dt;
    if (discontinuity) {
        next.offset_height = next.height - target;
        next.offset_velocity = next.velocity - target_velocity;
        next.nominal_height = target;
        next.nominal_velocity = target_velocity;
    } else {
        next.nominal_height = target;
        next.nominal_velocity = target_velocity;
    }
    next.source = source;
    if (source != support_held) {
        const float damping = (4.0f * LN2f) / (0.10f + 1.0e-5f);
        const float y = damping / 2.0f;
        const float j1 = next.offset_velocity +
            next.offset_height * y;
        const float eydt = g1_runner_fast_negexp(y * dt);
        next.offset_height = eydt *
            (next.offset_height + j1 * dt);
        next.offset_velocity = eydt *
            (next.offset_velocity - j1 * y * dt);
    }
    next.height = next.nominal_height + next.offset_height;
    next.velocity = next.nominal_velocity + next.offset_velocity;
    state = next;
    return ::isfinite(state.height) && ::isfinite(state.velocity);
}

static void inertialize_pose_update(
    g1_controller_state& state,
    const database& db,
    float halflife,
    float dt)
{
    const vec3 world_position = ::quat_mul_vec3(
        state.transition_dst_rotation,
        ::quat_mul_vec3(
            ::quat_inv(state.transition_src_rotation),
            db.bone_positions(state.frame_index, 0) -
                state.transition_src_position)) +
        state.transition_dst_position;
    const vec3 world_velocity = ::quat_mul_vec3(
        state.transition_dst_rotation,
        ::quat_mul_vec3(
            ::quat_inv(state.transition_src_rotation),
            db.bone_velocities(state.frame_index, 0)));
    const quat world_rotation = g1_runner_quat_normalize(
        ::quat_mul(
            state.transition_dst_rotation,
            ::quat_mul(
                ::quat_inv(state.transition_src_rotation),
                db.bone_rotations(state.frame_index, 0))));
    const vec3 world_angular_velocity = ::quat_mul_vec3(
        state.transition_dst_rotation,
        ::quat_mul_vec3(
            ::quat_inv(state.transition_src_rotation),
            db.bone_angular_velocities(state.frame_index, 0)));
    g1_runner_inertialize_vec3(
        state.bone_positions(0),
        state.bone_velocities(0),
        state.bone_offset_positions(0),
        state.bone_offset_velocities(0),
        world_position,
        world_velocity,
        halflife,
        dt);
    g1_runner_inertialize_quat(
        state.bone_rotations(0),
        state.bone_angular_velocities(0),
        state.bone_offset_rotations(0),
        state.bone_offset_angular_velocities(0),
        world_rotation,
        world_angular_velocity,
        halflife,
        dt);
    for (int bone = 1; bone < G1_BoneCount; ++bone) {
        g1_runner_inertialize_vec3(
            state.bone_positions(bone),
            state.bone_velocities(bone),
            state.bone_offset_positions(bone),
            state.bone_offset_velocities(bone),
            db.bone_positions(state.frame_index, bone),
            db.bone_velocities(state.frame_index, bone),
            halflife,
            dt);
        g1_runner_inertialize_quat(
            state.bone_rotations(bone),
            state.bone_angular_velocities(bone),
            state.bone_offset_rotations(bone),
            state.bone_offset_angular_velocities(bone),
            db.bone_rotations(state.frame_index, bone),
            db.bone_angular_velocities(state.frame_index, bone),
            halflife,
            dt);
    }
}

static void simulation_positions_update(
    g1_controller_state& state,
    float halflife,
    float dt,
    bool safe_stop_retry)
{
    if (safe_stop_retry) return;
    g1_runner_predict_position(
        state.simulation_position,
        state.simulation_velocity,
        state.simulation_acceleration,
        state.desired_velocity,
        halflife,
        dt);
}

static void simulation_rotations_update(
    g1_controller_state& state,
    float halflife,
    float dt)
{
    g1_runner_spring_quat(
        state.simulation_rotation,
        state.simulation_angular_velocity,
        state.desired_rotation,
        halflife,
        dt);
}

static void g1_runner_adjust_pose(
    g1_controller_state& state,
    const G1FrameTuning& tuning)
{
    state.adjustment_xz = 0.0f;
    state.adjustment_y = 0.0f;
    state.clamp_xz = 0.0f;
    state.clamp_y = 0.0f;
    if (tuning.synchronization_enabled) return;
    vec3 root_position = state.bone_positions(0);
    quat root_rotation = state.bone_rotations(0);
    if (tuning.adjustment_enabled) {
        const float factor = 1.0f - g1_runner_fast_negexp(
            (LN2f * tuning.dt) /
            (tuning.adjustment_position_halflife + 1.0e-5f));
        vec3 delta = vec3(
            state.simulation_position.x - root_position.x,
            0.0f,
            state.simulation_position.z - root_position.z);
        delta = delta * factor;
        if (tuning.adjustment_by_velocity_enabled) {
            const float maximum =
                tuning.adjustment_position_max_ratio *
                ::length(vec3(
                    state.bone_velocities(0).x,
                    0.0f,
                    state.bone_velocities(0).z)) * tuning.dt;
            const float magnitude = ::length(delta);
            if (magnitude > maximum && magnitude > 1.0e-8f) {
                delta = delta * (maximum / magnitude);
            }
        }
        root_position = root_position + delta;
        state.adjustment_xz = ::length(delta);
        const quat difference = ::quat_abs(g1_runner_quat_normalize(
            ::quat_mul_inv(
                state.simulation_rotation, root_rotation)));
        vec3 rotation_delta = g1_runner_quat_to_scaled_axis(difference);
        const float rotation_factor = 1.0f -
            g1_runner_fast_negexp(
                (LN2f * tuning.dt) /
                (tuning.adjustment_rotation_halflife + 1.0e-5f));
        rotation_delta = rotation_delta * rotation_factor;
        if (tuning.adjustment_by_velocity_enabled) {
            const float maximum =
                tuning.adjustment_rotation_max_ratio *
                ::length(state.bone_angular_velocities(0)) * tuning.dt;
            const float magnitude = ::length(rotation_delta);
            if (magnitude > maximum && magnitude > 1.0e-8f) {
                rotation_delta = rotation_delta * (maximum / magnitude);
            }
        }
        root_rotation = ::quat_mul(
            g1_runner_quat_from_scaled_axis(rotation_delta),
            root_rotation);
    }
    if (tuning.clamping_enabled) {
        vec3 delta = vec3(
            root_position.x - state.simulation_position.x,
            0.0f,
            root_position.z - state.simulation_position.z);
        const float magnitude = ::length(delta);
        if (magnitude > tuning.clamping_max_distance &&
            magnitude > 1.0e-8f) {
            const vec3 clamped = delta *
                (tuning.clamping_max_distance / magnitude);
            const vec3 before = root_position;
            root_position = vec3(
                state.simulation_position.x + clamped.x,
                root_position.y,
                state.simulation_position.z + clamped.z);
            state.clamp_xz = ::length(root_position - before);
        }
        vec3 difference = g1_runner_quat_to_scaled_axis(
            ::quat_abs(::quat_mul_inv(
                root_rotation, state.simulation_rotation)));
        const float angle = ::length(difference);
        if (angle > tuning.clamping_max_angle && angle > 1.0e-8f) {
            difference = difference *
                (tuning.clamping_max_angle / angle);
            root_rotation = ::quat_mul(
                g1_runner_quat_from_scaled_axis(difference),
                state.simulation_rotation);
        }
    }
    g1_runner_root_adjust(state, root_position, root_rotation);
}

static void g1_runner_contact_advance(
    bool& contact_state,
    bool& contact_lock,
    vec3& contact_position,
    vec3& contact_velocity,
    vec3& contact_point,
    vec3& contact_target,
    vec3& contact_offset_position,
    vec3& contact_offset_velocity,
    vec3 input_position,
    bool input_state,
    float unlock_radius,
    float foot_height,
    float halflife,
    float dt)
{
    const vec3 input_velocity =
        (input_position - contact_target) / (dt + 1.0e-8f);
    contact_target = input_position;
    g1_runner_inertialize_vec3(
        contact_position,
        contact_velocity,
        contact_offset_position,
        contact_offset_velocity,
        contact_lock ? contact_point : input_position,
        contact_lock ? vec3() : input_velocity,
        halflife,
        dt);
    const bool unlock = contact_lock &&
        ::length(contact_point - input_position) > unlock_radius;
    if (!contact_state && input_state) {
        contact_lock = true;
        contact_point = contact_position;
        contact_point.y = foot_height;
        g1_runner_transition_vec3(
            contact_offset_position,
            contact_offset_velocity,
            input_position,
            input_velocity,
            contact_point,
            vec3());
    } else if ((contact_lock && contact_state && !input_state) ||
               unlock) {
        contact_lock = false;
        g1_runner_transition_vec3(
            contact_offset_position,
            contact_offset_velocity,
            contact_point,
            vec3(),
            input_position,
            input_velocity);
    }
    contact_state = input_state;
}

static void contact_update(
    g1_controller_state& state,
    float unlock_radius,
    float foot_height,
    float blending_halflife,
    float dt)
{
    for (int foot = 0; foot < 2; ++foot) {
        const int bone = state.contact_bones(foot);
        g1_runner_contact_advance(
            state.contact_states(foot),
            state.contact_locks(foot),
            state.contact_positions(foot),
            state.contact_velocities(foot),
            state.contact_points(foot),
            state.contact_targets(foot),
            state.contact_offset_positions(foot),
            state.contact_offset_velocities(foot),
            state.global_bone_positions(bone),
            state.curr_bone_contacts(foot),
            unlock_radius,
            foot_height,
            blending_halflife,
            dt);
    }
}

static void g1_runner_footprint_rejection(
    G1FrameTransactionScratch& scratch,
    G1FootprintStatus status,
    G1IkStopReason reason,
    bool observation_available)
{
    G1FrameRejectionDiagnostic rejection;
    rejection.rejected = true;
    rejection.stage = G1FrameRejectFootprint;
    rejection.stop_reason = reason;
    rejection.footprint_status = status;
    rejection.attempted_footprint_available = observation_available;
    if (observation_available) {
        rejection.attempted_footprint = scratch.footprint;
    }
    scratch.rejection = rejection;
}

static void g1_runner_ik_rejection(
    G1FrameTransactionScratch& scratch,
    G1FrameRejectionStage stage,
    const G1IkFrameResult& attempted)
{
    G1FrameRejectionDiagnostic rejection;
    rejection.rejected = true;
    rejection.stage = stage;
    rejection.stop_reason = attempted.stop_reason;
    rejection.attempted_footprint_available = true;
    rejection.footprint_status = G1FootprintOk;
    rejection.attempted_footprint = scratch.footprint;
    rejection.attempted_ik_available = true;
    rejection.ik_frame = attempted;
    scratch.rejection = rejection;
}

static bool g1_runner_copy_final_pose(
    g1_controller_state& state,
    const database& db,
    char* error,
    int error_capacity)
{
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        state.ik_bone_positions(bone) =
            state.ik_candidate_bone_positions(bone);
        state.ik_bone_rotations(bone) =
            state.ik_candidate_bone_rotations(bone);
    }
    if (!::g1_ik_checked_forward_kinematics(
            state.ik_candidate_global_bone_positions,
            state.ik_candidate_global_bone_rotations,
            state.ik_candidate_bone_positions,
            state.ik_candidate_bone_rotations,
            db.bone_parents,
            error,
            error_capacity)) {
        return false;
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        state.ik_global_bone_positions(bone) =
            state.ik_candidate_global_bone_positions(bone);
        state.ik_global_bone_rotations(bone) =
            state.ik_candidate_global_bone_rotations(bone);
    }
    return true;
}

static G1FrameStageOutcome g1_runner_certificate_begin(
    g1_controller_state& state,
    G1FrameBranchCertificateScratch& branch,
    G1FrameCertificateBranch branch_kind,
    bool enabled,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity)
{
    if ((branch_kind != G1FrameCertificateRaw &&
         branch_kind != G1FrameCertificateIk) ||
        enabled != (branch_kind == G1FrameCertificateIk)) {
        return g1_runner_fail(
            error,
            error_capacity,
            "G1 certificate begin received an invalid branch identity");
    }
    scratch.rejection_branch = branch_kind;
    if (!::g1_ik_frame_begin(
            branch.ik_transaction,
            state.ik_candidate_bone_positions,
            state.ik_candidate_bone_rotations,
            state.ik,
            state.adjusted_bone_positions,
            state.adjusted_bone_rotations,
            external.db->bone_parents,
            state.curr_bone_contacts,
            external.scene->terrain,
            scratch.footprint,
            enabled,
            external.tuning.dt,
            error,
            error_capacity)) {
        return G1FrameStageGlobalError;
    }
    if (!branch.ik_transaction.candidate_result
             .safe_stop_requested) {
        return G1FrameStageContinue;
    }
    G1IkFrameResult attempted;
    if (!::g1_ik_frame_rejection_snapshot(
            attempted,
            branch.ik_transaction,
            G1IkRejectionAfterBegin,
            error,
            error_capacity)) {
        return G1FrameStageGlobalError;
    }
    g1_runner_ik_rejection(
        scratch, G1FrameRejectLandingPatch, attempted);
    return G1FrameStageFiniteReject;
}

static G1FrameStageOutcome g1_runner_certificate_foot(
    g1_controller_state& state,
    G1FrameBranchCertificateScratch& branch,
    G1FrameCertificateBranch branch_kind,
    int foot,
    bool enabled,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity)
{
    if ((branch_kind != G1FrameCertificateRaw &&
         branch_kind != G1FrameCertificateIk) ||
        enabled != (branch_kind == G1FrameCertificateIk) ||
        (foot != 0 && foot != 1)) {
        return g1_runner_fail(
            error,
            error_capacity,
            "G1 certificate foot received an invalid branch identity");
    }
    scratch.rejection_branch = branch_kind;
    if (!::g1_ik_frame_stage_foot(
            branch.ik_transaction,
            state.ik_candidate_bone_positions,
            state.ik_candidate_bone_rotations,
            foot,
            external.db->bone_parents,
            state.curr_bone_contacts,
            external.scene->terrain,
            scratch.footprint,
            enabled,
            external.tuning.dt,
            error,
            error_capacity)) {
        return G1FrameStageGlobalError;
    }
    if (!branch.ik_transaction.candidate_result
             .safe_stop_requested) {
        return G1FrameStageContinue;
    }
    G1IkFrameResult attempted;
    const G1IkRejectionCheckpoint checkpoint = foot == 0
        ? G1IkRejectionAfterFoot0
        : G1IkRejectionAfterFoot1;
    if (!::g1_ik_frame_rejection_snapshot(
            attempted,
            branch.ik_transaction,
            checkpoint,
            error,
            error_capacity)) {
        return G1FrameStageGlobalError;
    }
    g1_runner_ik_rejection(
        scratch, G1FrameRejectIkCandidate, attempted);
    return G1FrameStageFiniteReject;
}

static G1FrameStageOutcome g1_runner_certificate_finish(
    g1_controller_state& state,
    G1FrameBranchCertificateScratch& branch,
    bool enabled,
    G1FrameTransactionScratch&,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity)
{
    if (!::g1_ik_frame_finish(
            state.ik,
            state.ik_frame,
            branch.ik_transaction,
            state.ik_candidate_bone_positions,
            state.ik_candidate_bone_rotations,
            external.db->bone_parents,
            external.scene->terrain,
            external.tuning.dt,
            error,
            error_capacity)) {
        return G1FrameStageGlobalError;
    }
    if (state.ik_frame.safe_stop_requested ||
        enabled != state.ik_frame.applied) {
        return g1_runner_fail(
            error,
            error_capacity,
            "G1 IK finish produced an invalid terminal branch result");
    }
    if (!g1_runner_copy_final_pose(
            state,
            *external.db,
            error,
            error_capacity)) {
        return G1FrameStageGlobalError;
    }
    branch.ik_transaction.candidate_state = state.ik;
    branch.ik_transaction.candidate_result = state.ik_frame;
    return G1FrameStageContinue;
}

static G1FrameStageOutcome g1_runner_pose_certificate(
    g1_controller_state& state,
    G1FrameBranchCertificateScratch& branch,
    G1FrameCertificateBranch branch_kind,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity)
{
    if (branch_kind != G1FrameCertificateRaw &&
        branch_kind != G1FrameCertificateIk) {
        return g1_runner_fail(
            error,
            error_capacity,
            "G1 pose certificate received an invalid branch identity");
    }
    scratch.rejection_branch = branch_kind;
    const G1ClearanceBudget limits = ::g1_pose_clearance_budget();
    branch.pose_status = ::g1_measure_pose_clearance(
        branch.pose_clearance,
        limits,
        external.scene->terrain,
        state.ik_global_bone_positions,
        state.ik_global_bone_rotations,
        error,
        error_capacity);
    const bool threshold_failure =
        branch.pose_status == G1ClearanceOk &&
        (branch.pose_clearance.left.toe.lower_bound_m < -0.005 ||
         branch.pose_clearance.left.foot.lower_bound_m < -0.005 ||
         branch.pose_clearance.right.toe.lower_bound_m < -0.005 ||
         branch.pose_clearance.right.foot.lower_bound_m < -0.005 ||
         branch.pose_clearance.minimum.lower_bound_m < -0.01);
    if (branch.pose_status == G1ClearanceOutsideDomain ||
        branch.pose_status == G1ClearanceBudgetExceeded ||
        branch.pose_status == G1ClearanceUncertified ||
        threshold_failure) {
        G1FrameRejectionDiagnostic rejection;
        rejection.rejected = true;
        rejection.stage = G1FrameRejectPoseCertificate;
        rejection.stop_reason = G1IkStopPoseClearanceRejected;
        rejection.attempted_footprint_available = true;
        rejection.footprint_status = G1FootprintOk;
        rejection.attempted_footprint = scratch.footprint;
        rejection.pose_status = branch.pose_status;
        rejection.attempted_pose_available = threshold_failure;
        if (threshold_failure) {
            rejection.pose_clearance = branch.pose_clearance;
        }
        scratch.rejection = rejection;
        return G1FrameStageFiniteReject;
    }
    if (branch.pose_status != G1ClearanceOk) {
        return G1FrameStageGlobalError;
    }
    state.footprint_status = G1FootprintOk;
    state.footprint = scratch.footprint;
    state.ik_clearance = branch.pose_clearance;
    state.ik_candidate_clearance = branch.pose_clearance;
    state.ik_candidate_clearance_status = G1ClearanceOk;
    state.ik_candidate_rejected = false;
    return G1FrameStageContinue;
}

G1FrameStageOutcome g1_controller_frame_stage_run(
    G1FrameTransactionStage stage,
    g1_controller_state& working_state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity)
{
    g1_controller_state& state = working_state;
    const bool known_mode =
        external.tuning.mode == G1_TestLive ||
        external.tuning.mode == G1_TestSequential ||
        external.tuning.mode == G1_TestFlat ||
        external.tuning.mode == G1_TestTerrain ||
        external.tuning.mode == G1_TestRoute ||
        external.tuning.mode == G1_TestSceneCycle;
    if (!known_mode ||
        ::terrain_float_bits(state.search_time) !=
            ::terrain_float_bits(
                external.tuning.initial_search_time)) {
        return g1_runner_fail(
            error, error_capacity,
            "G1 frame runner received inconsistent immutable tuning");
    }

    switch (stage) {
    case G1FrameStageInputRouteCommand: {
        state.transitioned = false;
        state.adjustment_xz = 0.0f;
        state.adjustment_y = 0.0f;
        state.clamp_xz = 0.0f;
        state.clamp_y = 0.0f;
        state.camera_azimuth +=
            external.input.scripted_azimuth_delta;
        if (external.tuning.mode == G1_TestLive) {
            g1_runner_spring_scalar(
                state.desired_gait,
                state.desired_gait_velocity,
                external.input.gait_target,
                0.10f,
                external.tuning.dt);
        } else {
            state.desired_gait = 0.0f;
            state.desired_gait_velocity = 0.0f;
        }
        const float forward_speed =
            external.tuning.simulation_run_forward_speed *
                (1.0f - state.desired_gait) +
            external.tuning.simulation_walk_forward_speed *
                state.desired_gait;
        const float side_speed =
            external.tuning.simulation_run_side_speed *
                (1.0f - state.desired_gait) +
            external.tuning.simulation_walk_side_speed *
                state.desired_gait;
        const float back_speed =
            external.tuning.simulation_run_back_speed *
                (1.0f - state.desired_gait) +
            external.tuning.simulation_walk_back_speed *
                state.desired_gait;
        vec3 command = g1_runner_desired_velocity(
            external.input.move_stick,
            state.camera_azimuth,
            state.simulation_rotation,
            forward_speed,
            side_speed,
            back_speed);
        scratch.route_sample = {};
        scratch.route_sample.waypoint = state.route_waypoint;
        if (external.route != nullptr) {
            if (!::deterministic_route_command(
                    scratch.route_sample,
                    *external.route,
                    state.route_frames,
                    external.tuning.dt,
                    external.tuning.route_speed,
                    error,
                    error_capacity)) {
                return G1FrameStageGlobalError;
            }
            command = scratch.route_sample.command;
            if (!scratch.prior_safe_stop_latched) {
                state.route_waypoint = scratch.route_sample.waypoint;
            }
        }
        scratch.commanded_velocity =
            g1_runner_canonical_vec3(command);
        quat heading = g1_runner_desired_heading(
            state.desired_rotation,
            external.input.move_stick,
            external.input.look_stick,
            state.camera_azimuth,
            external.input.desired_strafe,
            scratch.commanded_velocity);
        if (external.heading_override.active) {
            heading = external.heading_override.heading;
        }
        if (external.route != nullptr &&
            ::length(scratch.commanded_velocity) > 0.01f &&
            !external.heading_override.active) {
            heading = ::quat_from_angle_axis(
                ::atan2f(
                    scratch.commanded_velocity.x,
                    scratch.commanded_velocity.z),
                vec3(0.0f, 1.0f, 0.0f));
        }
        if (scratch.prior_safe_stop_latched) {
            heading = state.desired_rotation;
        }
        scratch.requested_intent.requested_velocity =
            scratch.commanded_velocity;
        scratch.requested_intent.desired_heading = heading;
        if (!::g1_ik_safe_stop_handoff(
                scratch.safe_stop_handoff,
                scratch.prior_safe_stop_latched,
                scratch.commanded_velocity,
                error,
                error_capacity)) {
            return G1FrameStageGlobalError;
        }
        if (scratch.safe_stop_handoff.cancel_planar_inertia) {
            state.simulation_velocity.x = 0.0f;
            state.simulation_velocity.z = 0.0f;
            state.simulation_acceleration.x = 0.0f;
            state.simulation_acceleration.z = 0.0f;
        }
        scratch.traversal_input =
            scratch.safe_stop_handoff.applied_velocity;
        scratch.force_search =
            scratch.safe_stop_handoff.force_search;
        scratch.traversal = {};
        scratch.traversal.commanded_speed = ::length(vec3(
            scratch.traversal_input.x,
            0.0f,
            scratch.traversal_input.z));
        scratch.traversal.applied_speed =
            scratch.traversal.commanded_speed;
        scratch.traversal.distance = FLT_MAX;
        state.blocked = false;
        state.walkability_class = 1;
        state.blocked_distance = FLT_MAX;
        state.blocked_point = vec3();
        state.desired_velocity_change_prev =
            state.desired_velocity_change_curr;
        state.desired_velocity_change_curr =
            (scratch.traversal_input - state.desired_velocity) /
            external.tuning.dt;
        state.desired_velocity =
            g1_runner_canonical_vec3(scratch.traversal_input);
        if (state.scene_frame == 0) {
            for (int sample = 0;
                 sample < G1CommandTrajectorySampleCount;
                 ++sample) {
                state.trajectory_desired_velocities(sample) =
                    state.desired_velocity;
            }
        }
        state.desired_rotation_change_prev =
            state.desired_rotation_change_curr;
        state.desired_rotation_change_curr =
            g1_runner_quat_to_scaled_axis(::quat_abs(::quat_mul_inv(
                heading, state.desired_rotation))) /
            external.tuning.dt;
        state.desired_rotation = heading;
        if (state.force_search_timer <= 0.0f &&
            ((::length(state.desired_velocity_change_prev) >=
                  external.tuning.desired_velocity_change_threshold &&
              ::length(state.desired_velocity_change_curr) <
                  external.tuning.desired_velocity_change_threshold) ||
             (::length(state.desired_rotation_change_prev) >=
                  external.tuning.desired_rotation_change_threshold &&
              ::length(state.desired_rotation_change_curr) <
                  external.tuning.desired_rotation_change_threshold))) {
            scratch.force_search = true;
            state.force_search_timer = state.search_time;
        } else if (state.force_search_timer > 0.0f) {
            float timer = 0.0f;
            if (!::terrain_f32_sub(
                    timer,
                    state.force_search_timer,
                    external.tuning.dt)) {
                return g1_runner_fail(
                    error, error_capacity,
                    "force-search timer subtraction failed");
            }
            state.force_search_timer = timer;
        }
        g1_runner_build_prediction(
            state,
            scratch.requested_intent,
            state.desired_velocity,
            external);
        state.camera_distance = ::clampf(
            state.camera_distance + 10.0f * external.tuning.dt *
                external.input.camera_zoom_axis,
            0.1f,
            100.0f);
        scratch.requested_intent_ready = true;
        return G1FrameStageContinue;
    }
    case G1FrameStageMatcherSearch: {
        g1_runner_build_query(
            scratch.query,
            scratch.terrain_query,
            state,
            external);
        scratch.query_database_frame = state.frame_index;
        scratch.query_range = g1_runner_active_range(
            *external.db, state.frame_index);
        const int next = g1_runner_trajectory_clamp(
            *external.db, state.frame_index, 1);
        const bool end_of_animation = next == state.frame_index;
        const bool matching_enabled =
            external.tuning.mode != G1_TestSequential;
        state.searched = matching_enabled &&
            (scratch.force_search || state.search_timer <= 0.0f ||
             end_of_animation);
        state.incumbent_cost = end_of_animation
            ? FLT_MAX
            : g1_runner_database_cost(
                *external.db, state.frame_index, scratch.query);
        state.selected_cost = state.incumbent_cost;
        state.selected_terrain_error = g1_runner_terrain_error(
            *external.db, state.frame_index, scratch.query);
        int selected = state.frame_index;
        const int prior_index = state.frame_index;
        const traversability_diagnostics& traversal =
            scratch.traversal;
        scratch.transition_cost = ::g1_idle_match_transition_cost(
            traversal.commanded_speed,
            ::walkability_xz_length(state.simulation_velocity));
        if (state.searched) {
            float best_cost = FLT_MAX;
            const slice1d<float> query(31, scratch.query);
            ::database_search(
                selected,
                best_cost,
                *external.db,
                query,
                scratch.transition_cost);
            if (selected >= 0 && selected != prior_index) {
                state.selected_cost = best_cost;
                state.selected_terrain_error =
                    g1_runner_terrain_error(
                        *external.db, selected, scratch.query);
            }
            state.search_timer = state.search_time;
            state.force_search_timer = state.search_time;
        }

        const int slot_selected = selected >= 0
            ? selected
            : prior_index;
        G1CandidateRecord slot_zero;
        slot_zero.kind = state.searched
            ? G1CandidateLegacy
            : G1CandidateIncumbent;
        slot_zero.selected_frame = slot_selected;
        slot_zero.executed_frame = g1_runner_trajectory_clamp(
            *external.db, slot_selected, 1);
        slot_zero.source_range = g1_runner_active_range(
            *external.db, slot_selected);
        slot_zero.selected_cost = state.searched
            ? state.selected_cost
            : state.incumbent_cost;
        slot_zero.recovery_rank = UINT32_MAX;
        slot_zero.transitioned = state.searched &&
            slot_selected != prior_index;
        scratch.slot_zero_record = slot_zero;
        scratch.matching_scheduled = state.searched;
        scratch.legacy_search_performed = state.searched;
        scratch.recovery_request = G1RecoveryRequest{};
        scratch.recovery_request_ready = matching_enabled;
        if (matching_enabled) {
            scratch.recovery_request.db = external.db;
            for (uint32_t feature = 0U;
                 feature < G1RecoveryFeatureCount;
                 ++feature) {
                scratch.recovery_request.raw_query[feature] =
                    scratch.query[feature];
            }
            scratch.recovery_request.incumbent_frame = prior_index;
            scratch.recovery_request.legacy_selected_frame =
                slot_selected;
            scratch.recovery_request.transition_cost =
                scratch.transition_cost;
            scratch.recovery_request.public_incumbent_cost =
                state.incumbent_cost;
            scratch.recovery_request.ignore_range_end = 20;
            scratch.recovery_request.ignore_surrounding = 20;
        }
        return G1FrameStageContinue;
    }
    case G1FrameStageCandidateApply: {
        const G1CandidateRecord& candidate =
            scratch.active_candidate;
        const bool matching_enabled =
            external.tuning.mode != G1_TestSequential;
        const bool recovery_context_valid =
            matching_enabled &&
            scratch.recovery_request_ready &&
            scratch.matching_scheduled ==
                scratch.legacy_search_performed;
        const int source_range = g1_runner_active_range(
            *external.db, candidate.selected_frame);
        const int executed_frame = g1_runner_trajectory_clamp(
            *external.db, candidate.selected_frame, 1);
        bool owner_valid = false;
        if (candidate.kind == G1CandidateLegacy) {
            owner_valid = scratch.matching_scheduled &&
                scratch.legacy_search_performed &&
                scratch.recovery_request_ready &&
                candidate.selected_frame ==
                    scratch.slot_zero_record.selected_frame &&
                candidate.executed_frame ==
                    scratch.slot_zero_record.executed_frame &&
                candidate.source_range ==
                    scratch.slot_zero_record.source_range &&
                ::terrain_float_bits(candidate.selected_cost) ==
                    ::terrain_float_bits(
                        scratch.slot_zero_record.selected_cost) &&
                candidate.recovery_rank == UINT32_MAX &&
                candidate.transitioned ==
                    scratch.slot_zero_record.transitioned;
        } else if (candidate.kind ==
                   G1CandidateRecoveryTransition) {
            owner_valid = recovery_context_valid &&
                candidate.recovery_rank <
                    G1RecoveryTransitionCapacity &&
                candidate.transitioned &&
                candidate.selected_frame !=
                    scratch.recovery_request.incumbent_frame &&
                candidate.selected_frame !=
                    scratch.recovery_request.legacy_selected_frame &&
                (::terrain_float_bits(
                     scratch.recovery_request.public_incumbent_cost) ==
                     ::terrain_float_bits(FLT_MAX) ||
                 candidate.selected_cost <
                     scratch.recovery_request.public_incumbent_cost);
        } else if (candidate.kind == G1CandidateIncumbent) {
            const int incumbent_frame = recovery_context_valid
                ? scratch.recovery_request.incumbent_frame
                : scratch.slot_zero_record.selected_frame;
            const float incumbent_cost = recovery_context_valid
                ? scratch.recovery_request.public_incumbent_cost
                : scratch.slot_zero_record.selected_cost;
            owner_valid = candidate.selected_frame ==
                    incumbent_frame &&
                ::terrain_float_bits(candidate.selected_cost) ==
                    ::terrain_float_bits(incumbent_cost) &&
                candidate.recovery_rank == UINT32_MAX &&
                !candidate.transitioned;
        }
        if (!owner_valid || source_range < 0 ||
            candidate.source_range != source_range ||
            candidate.executed_frame != executed_frame ||
            !::g1_frame_candidate_record_is_valid(
                candidate, *external.db)) {
            return g1_runner_fail(
                error,
                error_capacity,
                "G1 candidate application received invalid provenance");
        }

        if (candidate.kind == G1CandidateRecoveryTransition) {
            state.selected_cost = candidate.selected_cost;
            state.selected_terrain_error = g1_runner_terrain_error(
                *external.db,
                candidate.selected_frame,
                scratch.query);
        } else if (candidate.kind == G1CandidateIncumbent) {
            state.selected_cost = state.incumbent_cost;
            state.selected_terrain_error = g1_runner_terrain_error(
                *external.db,
                candidate.selected_frame,
                scratch.query);
        }
        if (candidate.transitioned) {
            for (int bone = 0; bone < G1_BoneCount; ++bone) {
                state.trns_bone_positions(bone) =
                    external.db->bone_positions(
                        candidate.selected_frame, bone);
                state.trns_bone_velocities(bone) =
                    external.db->bone_velocities(
                        candidate.selected_frame, bone);
                state.trns_bone_rotations(bone) =
                    external.db->bone_rotations(
                        candidate.selected_frame, bone);
                state.trns_bone_angular_velocities(bone) =
                    external.db->bone_angular_velocities(
                        candidate.selected_frame, bone);
            }
            g1_runner_pose_transition(state);
            state.frame_index = candidate.selected_frame;
            state.transitioned = true;
        } else {
            state.transitioned = false;
        }
        scratch.selected_database_frame = candidate.selected_frame;
        float timer = 0.0f;
        if (!::terrain_f32_sub(
                timer,
                state.search_timer,
                external.tuning.dt)) {
            return g1_runner_fail(
                error, error_capacity,
                "search timer subtraction failed");
        }
        state.search_timer = timer;
        if (state.searched) {
            if (!::terrain_f32_sub(
                    timer,
                    state.force_search_timer,
                    external.tuning.dt)) {
                return g1_runner_fail(
                    error, error_capacity,
                    "force-search timer subtraction failed");
            }
            state.force_search_timer = timer;
        }
        state.frame_index = candidate.executed_frame;
        for (int bone = 0; bone < G1_BoneCount; ++bone) {
            state.curr_bone_positions(bone) =
                external.db->bone_positions(state.frame_index, bone);
            state.curr_bone_velocities(bone) =
                external.db->bone_velocities(state.frame_index, bone);
            state.curr_bone_rotations(bone) =
                external.db->bone_rotations(state.frame_index, bone);
            state.curr_bone_angular_velocities(bone) =
                external.db->bone_angular_velocities(
                    state.frame_index, bone);
        }
        for (int foot = 0; foot < 2; ++foot) {
            state.curr_bone_contacts(foot) =
                external.db->contact_states(state.frame_index, foot);
        }
        return G1FrameStageContinue;
    }
    case G1FrameStageInertialization: {
        ::inertialize_pose_update(
            state,
            *external.db,
            external.tuning.inertialize_blending_halflife,
            external.tuning.dt);
        return G1FrameStageContinue;
    }
    case G1FrameStageSimulationUpdate: {
        ::simulation_positions_update(
            state,
            external.tuning.simulation_velocity_halflife,
            external.tuning.dt,
            scratch.prior_safe_stop_latched);
        ::simulation_rotations_update(
            state,
            external.tuning.simulation_rotation_halflife,
            external.tuning.dt);
        return G1FrameStageContinue;
    }
    case G1FrameStageSupportObservation: {
        if (!::g1_ik_checked_forward_kinematics(
                state.global_bone_positions,
                state.global_bone_rotations,
                state.bone_positions,
                state.bone_rotations,
                external.db->bone_parents,
                error,
                error_capacity)) {
            return G1FrameStageGlobalError;
        }
        if (!::support_observation_build(
                state.support_observation_now,
                *external.support,
                state.frame_index,
                external.scene->terrain,
                state.global_bone_positions(G1_Simulation),
                state.global_bone_positions(G1_LeftToe),
                state.global_bone_positions(G1_RightToe),
                state.curr_bone_contacts(0),
                state.curr_bone_contacts(1),
                error,
                error_capacity)) {
            return G1FrameStageGlobalError;
        }
        if (!g1_runner_support_update(
                state.support,
                state.support_observation_now,
                state.transitioned,
                external.tuning.dt)) {
            return g1_runner_fail(
                error, error_capacity,
                "support update produced a non-finite state");
        }
        scratch.raw_selected_diagnostic = motion_match_pose_snapshot(
            state.global_bone_positions, external.scene->terrain);
        scratch.inertialized_diagnostic =
            scratch.raw_selected_diagnostic;
        return G1FrameStageContinue;
    }
    case G1FrameStageSupportRetarget: {
        g1_runner_adjust_pose(state, external.tuning);
        for (int bone = 0; bone < G1_BoneCount; ++bone) {
            state.adjusted_bone_rotations(bone) =
                state.bone_rotations(bone);
        }
        ::support_pose_apply(
            state.adjusted_bone_positions,
            state.bone_positions,
            state.support.height);
        return G1FrameStageContinue;
    }
    case G1FrameStageContactUpdate: {
        if (!::g1_ik_checked_forward_kinematics(
                state.global_bone_positions,
                state.global_bone_rotations,
                state.adjusted_bone_positions,
                state.adjusted_bone_rotations,
                external.db->bone_parents,
                error,
                error_capacity)) {
            return G1FrameStageGlobalError;
        }
        ::contact_update(
            state,
            external.tuning.contact_unlock_radius,
            external.tuning.contact_foot_height,
            external.tuning.contact_blending_halflife,
            external.tuning.dt);
        for (int bone = 0; bone < G1_BoneCount; ++bone) {
            state.ik_bone_positions(bone) =
                state.adjusted_bone_positions(bone);
            state.ik_bone_rotations(bone) =
                state.adjusted_bone_rotations(bone);
            state.ik_global_bone_positions(bone) =
                state.global_bone_positions(bone);
            state.ik_global_bone_rotations(bone) =
                state.global_bone_rotations(bone);
            state.ik_candidate_bone_positions(bone) =
                state.adjusted_bone_positions(bone);
            state.ik_candidate_bone_rotations(bone) =
                state.adjusted_bone_rotations(bone);
            state.ik_candidate_global_bone_positions(bone) =
                state.global_bone_positions(bone);
            state.ik_candidate_global_bone_rotations(bone) =
                state.global_bone_rotations(bone);
        }
        state.ik_frame = G1IkFrameResult{};
        scratch.support_retargeted_diagnostic =
            motion_match_pose_snapshot(
                state.global_bone_positions,
                external.scene->terrain);
        return G1FrameStageContinue;
    }
    case G1FrameStageFootprintObservation: {
        if (!::g1_foot_contact_schedule_build(
                scratch.contact_schedule,
                external.db->contact_states,
                external.db->range_starts,
                external.db->range_stops,
                state.frame_index,
                state.curr_bone_contacts(0),
                state.curr_bone_contacts(1),
                external.tuning.dt,
                external.tuning.trajectory_sample_time,
                error,
                error_capacity)) {
            return G1FrameStageGlobalError;
        }
        const G1FootprintBudget limits = ::g1_footprint_budget();
        scratch.footprint_status = ::g1_footprint_observe_v2(
            scratch.footprint,
            limits,
            external.scene->terrain,
            external.scene->walkability,
            state.command,
            scratch.contact_schedule,
            state.global_bone_positions,
            state.global_bone_rotations,
            error,
            error_capacity);
        if (scratch.footprint_status == G1FootprintOutsideDomain) {
            g1_runner_footprint_rejection(
                scratch,
                scratch.footprint_status,
                G1IkStopFootprintOutsideDomain,
                false);
            return G1FrameStageFiniteReject;
        }
        if (scratch.footprint_status == G1FootprintBudgetExceeded) {
            g1_runner_footprint_rejection(
                scratch,
                scratch.footprint_status,
                G1IkStopFootprintBudgetExceeded,
                false);
            return G1FrameStageFiniteReject;
        }
        if (scratch.footprint_status != G1FootprintOk) {
            return G1FrameStageGlobalError;
        }
        if (scratch.footprint.blocked) {
            g1_runner_footprint_rejection(
                scratch,
                G1FootprintOk,
                G1IkStopFootprintBlocked,
                true);
            return G1FrameStageFiniteReject;
        }
        state.footprint_status = G1FootprintOk;
        state.footprint = scratch.footprint;
        return G1FrameStageContinue;
    }
    case G1FrameStageRawBegin:
        return g1_runner_certificate_begin(
            state,
            scratch.raw_certificate,
            G1FrameCertificateRaw,
            false,
            scratch,
            external,
            error,
            error_capacity);
    case G1FrameStageRawFirstFoot:
        return g1_runner_certificate_foot(
            state,
            scratch.raw_certificate,
            G1FrameCertificateRaw,
            0,
            false,
            scratch,
            external,
            error,
            error_capacity);
    case G1FrameStageRawSecondFoot:
        return g1_runner_certificate_foot(
            state,
            scratch.raw_certificate,
            G1FrameCertificateRaw,
            1,
            false,
            scratch,
            external,
            error,
            error_capacity);
    case G1FrameStageRawFinalFk:
        return g1_runner_certificate_finish(
            state,
            scratch.raw_certificate,
            false,
            scratch,
            external,
            error,
            error_capacity);
    case G1FrameStageRawPoseCertificate:
        return g1_runner_pose_certificate(
            state,
            scratch.raw_certificate,
            G1FrameCertificateRaw,
            scratch,
            external,
            error,
            error_capacity);
    case G1FrameStageIkBegin:
        return g1_runner_certificate_begin(
            state,
            scratch.ik_certificate,
            G1FrameCertificateIk,
            true,
            scratch,
            external,
            error,
            error_capacity);
    case G1FrameStageIkFirstFoot:
        return g1_runner_certificate_foot(
            state,
            scratch.ik_certificate,
            G1FrameCertificateIk,
            0,
            true,
            scratch,
            external,
            error,
            error_capacity);
    case G1FrameStageIkSecondFoot:
        return g1_runner_certificate_foot(
            state,
            scratch.ik_certificate,
            G1FrameCertificateIk,
            1,
            true,
            scratch,
            external,
            error,
            error_capacity);
    case G1FrameStageIkFinalFk:
        return g1_runner_certificate_finish(
            state,
            scratch.ik_certificate,
            true,
            scratch,
            external,
            error,
            error_capacity);
    case G1FrameStageIkPoseCertificate:
        return g1_runner_pose_certificate(
            state,
            scratch.ik_certificate,
            G1FrameCertificateIk,
            scratch,
            external,
            error,
            error_capacity);
    case G1FrameStageAcceptedFinalize: {
        const G1FrameBranchCertificateScratch& visible_certificate =
            external.tuning.ik_enabled
                ? scratch.ik_certificate
                : scratch.raw_certificate;
        state.footprint_status = G1FootprintOk;
        state.footprint = scratch.footprint;
        state.ik_clearance = visible_certificate.pose_clearance;
        state.ik_candidate_clearance =
            visible_certificate.pose_clearance;
        state.ik_candidate_clearance_status = G1ClearanceOk;
        state.ik_candidate_rejected = false;
        scratch.rendered_diagnostic = motion_match_pose_snapshot(
            state.ik_global_bone_positions,
            external.scene->terrain);
        G1FrameAcceptedDiagnostic diagnostic;
        diagnostic.ready = true;
        diagnostic.presentation_frame =
            external.input.presentation_frame;
        diagnostic.scene_frame = state.scene_frame;
        diagnostic.route = scratch.route_sample;
        diagnostic.traversal = scratch.traversal;
        for (int dimension = 0; dimension < 31; ++dimension) {
            diagnostic.query[dimension] = scratch.query[dimension];
        }
        diagnostic.query_database_frame =
            scratch.query_database_frame;
        diagnostic.query_range = scratch.query_range;
        diagnostic.selected_database_frame =
            scratch.selected_database_frame;
        diagnostic.terrain_query = scratch.terrain_query;
        diagnostic.raw_selected =
            scratch.raw_selected_diagnostic;
        diagnostic.inertialized =
            scratch.inertialized_diagnostic;
        diagnostic.support_retargeted =
            scratch.support_retargeted_diagnostic;
        diagnostic.rendered = scratch.rendered_diagnostic;
        diagnostic.matching_enabled =
            external.tuning.mode != G1_TestSequential;
        diagnostic.adjustment_enabled =
            external.tuning.adjustment_enabled;
        diagnostic.clamping_enabled =
            external.tuning.clamping_enabled;
        diagnostic.ik_enabled = external.tuning.ik_enabled;
        diagnostic.effective_terrain_weight =
            external.tuning.effective_terrain_weight;
        if (state.scene_frame == INT_MAX) {
            return g1_runner_fail(
                error, error_capacity,
                "scene frame counter overflow");
        }
        ++state.scene_frame;
        if (external.tuning.mode == G1_TestRoute &&
            !scratch.prior_safe_stop_latched) {
            if (state.route_frames == INT_MAX) {
                return g1_runner_fail(
                    error, error_capacity,
                    "route frame counter overflow");
            }
            ++state.route_frames;
        }
        scratch.accepted_diagnostic_candidate = diagnostic;
        scratch.accepted_diagnostic_ready = true;
        return G1FrameStageContinue;
    }
    default:
        return G1FrameStageGlobalError;
    }
}
