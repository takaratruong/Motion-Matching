#include "json_runtime.h"
#include "scene_runtime.h"
#include "sha256.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

static const char* const json_fixture_path = "/tmp/test_scene_json.json";

static const char* expected_scene_ids[] = {
    "grail-curb-default", "grail-curb-low", "grail-curb-medium",
    "grail-curb-high", "stairs-shallow", "stairs-standard",
    "stairs-unseen-variable", "ramp-05-up-down", "ramp-10-up-down",
    "ramp-15-stress", "cross-slope-05", "cross-slope-10",
    "mixed-multilevel", "blocked-course",
};
static const int expected_route_counts[] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,2,
};
static const char* expected_route_ids[][2] = {
    {"curb-forward",NULL},{"curb-forward",NULL},
    {"curb-forward",NULL},{"curb-forward",NULL},
    {"ascent-landing-descent",NULL},
    {"ascent-landing-descent",NULL},
    {"ascent-landing-descent",NULL},
    {"up-landing-down",NULL},{"up-landing-down",NULL},
    {"up-landing-down",NULL},
    {"forward-cross-slope",NULL},{"forward-cross-slope",NULL},
    {"full-course",NULL},{"wall-safe-stop","ramp-safe-stop"},
};
static const char* expected_route_outcomes[][2] = {
    {"traverse-or-safe-stop",NULL},{"traverse",NULL},
    {"traverse-or-safe-stop",NULL},{"traverse-or-safe-stop",NULL},
    {"traverse",NULL},{"traverse",NULL},{"traverse",NULL},
    {"traverse",NULL},{"traverse",NULL},
    {"traverse-or-safe-stop",NULL},
    {"traverse",NULL},{"traverse",NULL},{"traverse",NULL},
    {"safe-stop","safe-stop"},
};
static const int expected_route_classes[][2] = {
    {2,-1},{1,-1},{2,-1},{2,-1},{1,-1},{1,-1},{1,-1},
    {1,-1},{1,-1},{2,-1},{1,-1},{1,-1},{1,-1},{0,0},
};

static const char* const expected_surface_semantics_json = R"json({"barycentric_tolerance":1e-10,"bbox_tolerance_m":1e-12,"cell_size_m":0.02,"coordinate_signature":"holden-y-up-right-handed-forward-plus-z","degenerate_projected_triangle_policy":"ignore","exterior_height_m":0.0,"heightfield_cell_domain":"positive-normal-binary32","heightfield_denormal_policy":"reject-nonzero-binary32-subnormals","heightfield_diagonal":"min-x-min-z_to_max-x-max-z","heightfield_diagonal_tie_policy":"tx-greater-or-equal-tz-uses-p00-p10-p11","heightfield_domain_policy":"inclusive-authoritative-node-rectangle","heightfield_evaluation_precision":"binary64-from-binary32-samples-and-promoted-node-weights","heightfield_exterior_normal":[0.0,1.0,0.0],"heightfield_grid_line_policy":"positive-index-cell-except-maximum-edge","heightfield_interpolation":"fixed-diagonal-triangles","heightfield_normal_evaluation":"selected-triangle-binary64-gradient-scale-safe-unit-normalization","heightfield_obj_coordinate_quantization":"binary32-round-of-promoted-origin-plus-index-times-cell","heightfield_obj_face_order":"p00-p11-p10_then_p00-p01-p11","heightfield_obj_float_format":".9g-final-newline","heightfield_obj_vertex_order":"z-major-x-minor","heightfield_raster_bounds_policy":"float32-minimum-rounded-down-and-maximum-ceil-covered","heightfield_runtime_height_output":"finite-binary64-interpolation-rounded-to-binary32","heightfield_runtime_node_distinguishability_policy":"normal-or-positive-zero-strictly-increasing-proven-by-endpoints-near-zero-candidates-max-binary32-spacing-and-aligned-equality","heightfield_runtime_node_domain":"normal-or-zero-binary32","heightfield_runtime_normal_output":"unit-normal-components-rounded-to-binary32","heightfield_runtime_output_ftz_policy":"binary32-subnormals-and-signed-zero-canonicalized-to-positive-zero","heightfield_runtime_parity_domain":"normal-or-zero-binary32-coordinates","heightfield_runtime_query_domain":"normal-or-zero-binary32-coordinates","heightfield_runtime_query_encoding":"normal-or-zero-binary32-canonicalized-positive-and-promoted-to-binary64","heightfield_scalar_domain":"normal-or-zero-binary32","heightfield_scalar_encoding":"ieee754-binary32-little-endian","heightfield_schema":"G1HF/v2","heightfield_source_node_encoding":"binary32-header-values-promoted-to-binary64-arithmetic","heightfield_version":2,"heightfield_zero_encoding":"canonical-positive-zero","overlap_height_policy":"maximum-y","polygon_triangulation":"fan-from-first-index","projected_area_epsilon_m2":1e-12,"projected_area_measure":"absolute-two-times-area","projected_boundary_policy":"closed","schema":"g1-terrain-surface/v1","source_query":"vertical-triangle-top","triangle_winding_policy":"orientation-independent"})json";

static const char* const expected_surface_signature =
    "f151c2b1c7f0498880f76c37f48a47c46c48bcf58c1285863fabc9a09fd7993a";

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "scene runtime test failed: %s\n", message);
        std::exit(1);
    }
}

static void write_text(const char* path, const std::string& text)
{
    FILE* file = std::fopen(path, "wb");
    check(file != NULL, "open fixture");
    check(std::fwrite(text.data(), 1, text.size(), file) == text.size(),
          "write fixture");
    check(std::fclose(file) == 0, "close fixture");
}

static bool load_text(
    json_value& out, const std::string& text, char error[512]);

static void write_bytes(
    const std::string& path, const std::vector<unsigned char>& bytes)
{
    FILE* file = std::fopen(path.c_str(), "wb");
    check(file != NULL, "open binary fixture");
    check(bytes.empty() ||
          std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size(),
          "write binary fixture");
    check(std::fclose(file) == 0, "close binary fixture");
}

static std::string file_sha(const std::string& path)
{
    char error[512] = {};
    std::string digest;
    check(sha256_file_hex(digest, path.c_str(), error, sizeof(error)), error);
    return digest;
}

static json_value parse_json_text(const std::string& text)
{
    char error[512] = {};
    json_value value;
    check(load_text(value, text, error), error);
    return value;
}

static std::string dump_json_string(const std::string& value)
{
    std::string out = "\"";
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < value.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(static_cast<char>(c));
        } else if (c == '\b') out += "\\b";
        else if (c == '\f') out += "\\f";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20u) {
            out += "\\u00";
            out.push_back(hex[c >> 4]);
            out.push_back(hex[c & 15u]);
        } else out.push_back(static_cast<char>(c));
    }
    out.push_back('"');
    return out;
}

static std::string dump_json(const json_value& value)
{
    if (value.kind == json_null) return "null";
    if (value.kind == json_boolean)
        return value.boolean_value ? "true" : "false";
    if (value.kind == json_number) {
        std::ostringstream out;
        out << std::setprecision(std::numeric_limits<double>::max_digits10)
            << value.number_value;
        return out.str();
    }
    if (value.kind == json_string)
        return dump_json_string(value.string_value);
    std::string out = value.kind == json_array ? "[" : "{";
    if (value.kind == json_array) {
        for (size_t i = 0; i < value.array_value.size(); ++i) {
            if (i != 0) out.push_back(',');
            out += dump_json(value.array_value[i]);
        }
        out.push_back(']');
    } else {
        for (size_t i = 0; i < value.object_value.size(); ++i) {
            if (i != 0) out.push_back(',');
            out += dump_json_string(value.object_value[i].first);
            out.push_back(':');
            out += dump_json(value.object_value[i].second);
        }
        out.push_back('}');
    }
    return out;
}

static bool artifact_equal(
    const artifact_reference& first, const artifact_reference& second)
{
    return first.path == second.path && first.schema == second.schema &&
           first.sha256 == second.sha256 && first.version == second.version &&
           first.dimensions == second.dimensions &&
           first.columns == second.columns;
}

static bool manifest_equal(
    const motion_pack_manifest& first, const motion_pack_manifest& second)
{
    if (first.output_fps != second.output_fps ||
        first.feature_dimensions != second.feature_dimensions ||
        first.terrain_dimensions != second.terrain_dimensions ||
        first.support_dimensions != second.support_dimensions ||
        first.total_clips != second.total_clips ||
        first.grail_clips != second.grail_clips ||
        first.skipped_clips != second.skipped_clips ||
        first.database_frames != second.database_frames ||
        first.diagnostic_mode != second.diagnostic_mode ||
        first.surface.signature != second.surface.signature ||
        first.surface.coordinate_signature !=
            second.surface.coordinate_signature ||
        first.surface.heightfield_interpolation !=
            second.surface.heightfield_interpolation ||
        first.surface.heightfield_diagonal !=
            second.surface.heightfield_diagonal ||
        first.surface.cell_size != second.surface.cell_size ||
        first.surface.exterior_height != second.surface.exterior_height ||
        first.sources.size() != second.sources.size() ||
        !artifact_equal(first.database, second.database) ||
        !artifact_equal(first.terrain_features, second.terrain_features) ||
        !artifact_equal(first.terrain_support, second.terrain_support) ||
        !artifact_equal(first.scene_index, second.scene_index) ||
        !artifact_equal(first.validation_file, second.validation_file))
        return false;
    for (size_t i = 0; i < first.sources.size(); ++i)
        if (first.sources[i].name != second.sources[i].name ||
            first.sources[i].terrain_id != second.sources[i].terrain_id ||
            first.sources[i].range_start != second.sources[i].range_start ||
            first.sources[i].range_stop != second.sources[i].range_stop)
            return false;
    return true;
}

static bool bounds2_equal(const bounds2& first, const bounds2& second)
{
    return first.min_x == second.min_x && first.min_z == second.min_z &&
           first.max_x == second.max_x && first.max_z == second.max_z;
}

static bool point3_equal(const point3d& first, const point3d& second)
{
    return first.x == second.x && first.y == second.y && first.z == second.z;
}

static bool metadata_equal(
    const scene_metadata& first, const scene_metadata& second)
{
    if (first.id != second.id || first.label != second.label ||
        first.provenance_kind != second.provenance_kind ||
        first.provenance_source_ids != second.provenance_source_ids ||
        first.coordinate_signature != second.coordinate_signature ||
        first.surface_signature != second.surface_signature ||
        !artifact_equal(first.heightfield, second.heightfield) ||
        !artifact_equal(first.mesh, second.mesh) ||
        !artifact_equal(first.walkability, second.walkability) ||
        first.heightfield_nx != second.heightfield_nx ||
        first.heightfield_nz != second.heightfield_nz ||
        first.walkability_nx != second.walkability_nx ||
        first.walkability_nz != second.walkability_nz ||
        first.heightfield_origin_x != second.heightfield_origin_x ||
        first.heightfield_origin_z != second.heightfield_origin_z ||
        first.heightfield_cell_size != second.heightfield_cell_size ||
        first.heightfield_exterior_height !=
            second.heightfield_exterior_height ||
        !point3_equal(first.mesh_bounds.minimum,
                      second.mesh_bounds.minimum) ||
        !point3_equal(first.mesh_bounds.maximum,
                      second.mesh_bounds.maximum) ||
        !point3_equal(first.heightfield_bounds.minimum,
                      second.heightfield_bounds.minimum) ||
        !point3_equal(first.heightfield_bounds.maximum,
                      second.heightfield_bounds.maximum) ||
        !bounds2_equal(first.playable_bounds, second.playable_bounds) ||
        !bounds2_equal(first.lookahead_bounds, second.lookahead_bounds) ||
        first.spawn_position.x != second.spawn_position.x ||
        first.spawn_position.y != second.spawn_position.y ||
        first.spawn_position.z != second.spawn_position.z ||
        first.spawn_yaw != second.spawn_yaw ||
        first.certified_regions.size() != second.certified_regions.size() ||
        first.stress_regions.size() != second.stress_regions.size() ||
        first.blocked_regions.size() != second.blocked_regions.size() ||
        first.routes.size() != second.routes.size()) return false;
    const std::vector<scene_region>* first_regions[] = {
        &first.certified_regions, &first.stress_regions, &first.blocked_regions};
    const std::vector<scene_region>* second_regions[] = {
        &second.certified_regions, &second.stress_regions, &second.blocked_regions};
    for (int group = 0; group < 3; ++group)
        for (size_t i = 0; i < first_regions[group]->size(); ++i)
            if ((*first_regions[group])[i].id !=
                    (*second_regions[group])[i].id ||
                !bounds2_equal((*first_regions[group])[i].bounds,
                               (*second_regions[group])[i].bounds))
                return false;
    for (size_t i = 0; i < first.routes.size(); ++i)
        if (first.routes[i].id != second.routes[i].id ||
            first.routes[i].expected_outcome !=
                second.routes[i].expected_outcome ||
            first.routes[i].walkability_class !=
                second.routes[i].walkability_class ||
            first.routes[i].landing_hold_seconds !=
                second.routes[i].landing_hold_seconds ||
            first.routes[i].waypoints_xz != second.routes[i].waypoints_xz)
            return false;
    return true;
}

static bool pack_equal(const scene_pack& first, const scene_pack& second)
{
    if (!metadata_equal(first.metadata, second.metadata) ||
        first.scene_path != second.scene_path ||
        first.terrain_path != second.terrain_path ||
        first.mesh_path != second.mesh_path ||
        first.walkability_path != second.walkability_path ||
        first.terrain.version != second.terrain.version ||
        first.terrain.nx != second.terrain.nx ||
        first.terrain.nz != second.terrain.nz ||
        first.terrain.origin_x != second.terrain.origin_x ||
        first.terrain.origin_z != second.terrain.origin_z ||
        first.terrain.cell_size != second.terrain.cell_size ||
        first.terrain.exterior_height != second.terrain.exterior_height ||
        first.terrain.heights.size != second.terrain.heights.size ||
        first.walkability.nx != second.walkability.nx ||
        first.walkability.nz != second.walkability.nz ||
        first.walkability.cells.size != second.walkability.cells.size)
        return false;
    for (int i = 0; i < first.terrain.heights.size; ++i)
        if (first.terrain.heights(i) != second.terrain.heights(i)) return false;
    for (int i = 0; i < first.walkability.cells.size; ++i)
        if (first.walkability.cells(i) != second.walkability.cells(i))
            return false;
    return true;
}

static bool catalog_equal(
    const scene_catalog& first, const scene_catalog& second)
{
    if (first.default_scene_id != second.default_scene_id ||
        first.coordinate_signature != second.coordinate_signature ||
        first.surface_signature != second.surface_signature ||
        first.ids != second.ids || first.scenes.size() != second.scenes.size())
        return false;
    for (size_t i = 0; i < first.scenes.size(); ++i)
        if (first.scenes[i].id != second.scenes[i].id ||
            first.scenes[i].path != second.scenes[i].path ||
            first.scenes[i].sha256 != second.scenes[i].sha256)
            return false;
    return true;
}

static bool load_text(
    json_value& out, const std::string& text, char error[512])
{
    write_text(json_fixture_path, text);
    error[0] = '\0';
    return json_document_load(out, json_fixture_path, error, 512);
}

static json_value sentinel_value()
{
    json_value value;
    value.kind = json_string;
    value.boolean_value = true;
    value.number_value = 42.5;
    value.string_value = "sentinel";
    value.array_value.push_back(json_value());
    value.object_value.push_back(
        std::make_pair(std::string("preserved"), json_value()));
    value.source_offset = 123;
    return value;
}

static void check_sentinel(const json_value& value, const char* message)
{
    check(value.kind == json_string, message);
    check(value.boolean_value, message);
    check(value.number_value == 42.5, message);
    check(value.string_value == "sentinel", message);
    check(value.array_value.size() == 1, message);
    check(value.object_value.size() == 1, message);
    check(value.object_value[0].first == "preserved", message);
    check(value.source_offset == 123, message);
}

static void expect_json_failure(const std::string& text, const char* message)
{
    char error[512] = {};
    json_value value = sentinel_value();
    check(!load_text(value, text, error), message);
    check(std::strstr(error, json_fixture_path) != NULL,
          "JSON failure contains fixture path");
    check_sentinel(value, "JSON failure is transactional");
}

static void test_json_value_kinds_members_and_offsets()
{
    char error[512] = {};
    json_value value;
    const std::string text =
        " \n{\"none\":null,\"yes\":true,\"no\":false,"
        "\"number\":-1.25e2,\"string\":\"value\","
        "\"array\":[0],\"object\":{}}";
    check(load_text(value, text, error), error);
    check(value.kind == json_object, "root object kind");
    check(value.source_offset == text.find('{'), "root source offset");

    const json_value* none = json_member(value, "none");
    const json_value* yes = json_member(value, "yes");
    const json_value* no = json_member(value, "no");
    const json_value* number = json_member(value, "number");
    const json_value* string = json_member(value, "string");
    const json_value* array = json_member(value, "array");
    const json_value* object = json_member(value, "object");
    check(none != NULL && none->kind == json_null, "null kind");
    check(yes != NULL && yes->kind == json_boolean && yes->boolean_value,
          "true boolean kind");
    check(no != NULL && no->kind == json_boolean && !no->boolean_value,
          "false boolean kind");
    check(number != NULL && number->kind == json_number &&
          number->number_value == -125.0, "number kind and decode");
    check(string != NULL && string->kind == json_string &&
          string->string_value == "value", "string kind and decode");
    check(array != NULL && array->kind == json_array &&
          array->array_value.size() == 1, "array kind");
    check(object != NULL && object->kind == json_object &&
          object->object_value.empty(), "object kind");

    check(none->source_offset == text.find("null"), "null source offset");
    check(array->source_offset == text.find('['), "array source offset");
    check(array->array_value[0].source_offset == text.find("[0") + 1,
          "array member source offset");
    check(json_member(value, "missing") == NULL, "missing member lookup");
    check(json_member(value, NULL) == NULL, "null member key");
    check(json_member(*string, "anything") == NULL,
          "member lookup requires object");
}

static void test_json_full_document_and_unique_keys()
{
    const std::string failures[] = {
        "",
        " \t\r\n",
        "[] trailing",
        "[1,]",
        "{\"x\":1,}",
        "{\"x\" 1}",
        "{\"x\":1",
        "{\"x\":tru}",
        "{\"x\":1,\"x\":2}",
        "{\"x\":1,\"\\u0078\":2}",
    };
    for (size_t i = 0; i < sizeof(failures) / sizeof(failures[0]); ++i)
        expect_json_failure(failures[i], "strict full-document rejection");
}

static void test_json_number_grammar_and_finite_range()
{
    char error[512] = {};
    json_value value;
    check(load_text(value,
        "[0,-0,17,-1.25,6.022e23,1E-3]", error), error);
    check(value.kind == json_array && value.array_value.size() == 6,
          "valid JSON number forms");
    for (size_t i = 0; i < value.array_value.size(); ++i)
        check(value.array_value[i].kind == json_number,
              "valid number has number kind");
    check(value.array_value[3].number_value == -1.25,
          "fraction number decode");
    check(value.array_value[4].number_value > 6.0e23,
          "exponent number decode");

    const std::string failures[] = {
        "{\"x\":01}",
        "{\"x\":-01}",
        "{\"x\":+1}",
        "{\"x\":.1}",
        "{\"x\":1.}",
        "{\"x\":1e}",
        "{\"x\":1e+}",
        "{\"x\":-}",
        "{\"x\":NaN}",
        "{\"x\":Infinity}",
        "{\"x\":1e999}",
        "{\"x\":1e-9999}",
    };
    for (size_t i = 0; i < sizeof(failures) / sizeof(failures[0]); ++i)
        expect_json_failure(failures[i], "invalid or non-finite number rejection");
}

static std::string json_string_containing(const std::string& bytes)
{
    return std::string("{\"x\":\"") + bytes + "\"}";
}

static void test_json_utf8_escapes_and_surrogates()
{
    char error[512] = {};
    json_value value;
    const std::string valid =
        "{\"escapes\":\"\\\"\\\\\\/\\b\\f\\n\\r\\t\","
        "\"bmp\":\"\\u2603\",\"pair\":\"\\uD83D\\uDE00\","
        "\"raw\":\"\xe2\x98\x83\",\"nul\":\"\\u0000\"}";
    check(load_text(value, valid, error), error);
    check(json_member(value, "escapes")->string_value ==
          std::string("\"\\/\b\f\n\r\t"), "simple escape decode");
    check(json_member(value, "bmp")->string_value == "\xe2\x98\x83",
          "BMP unicode escape decode");
    check(json_member(value, "pair")->string_value == "\xf0\x9f\x98\x80",
          "surrogate pair decode");
    check(json_member(value, "raw")->string_value == "\xe2\x98\x83",
          "valid raw UTF-8 preservation");
    check(json_member(value, "nul")->string_value.size() == 1 &&
          json_member(value, "nul")->string_value[0] == '\0',
          "escaped null decode");

    std::vector<std::string> failures;
    failures.push_back("{\"x\":\"\\q\"}");
    failures.push_back("{\"x\":\"\\u12G4\"}");
    failures.push_back("{\"x\":\"\\u123\"}");
    failures.push_back("{\"x\":\"\\uD800\"}");
    failures.push_back("{\"x\":\"\\uD800\\u0041\"}");
    failures.push_back("{\"x\":\"\\uDC00\"}");
    failures.push_back("{\"x\":\"unterminated}");
    failures.push_back(json_string_containing(std::string(1, '\0')));
    failures.push_back(json_string_containing(std::string(1, '\x01')));
    failures.push_back(json_string_containing(std::string(1, '\x80')));
    failures.push_back(json_string_containing(std::string("\xc0\xaf", 2)));
    failures.push_back(json_string_containing(std::string("\xe0\x80\xaf", 3)));
    failures.push_back(json_string_containing(std::string("\xe2\x28\xa1", 3)));
    failures.push_back(json_string_containing(std::string("\xed\xa0\x80", 3)));
    failures.push_back(json_string_containing(std::string("\xf4\x90\x80\x80", 4)));
    failures.push_back(std::string("{\"x\":\"") + std::string(1, '\xe2'));
    for (size_t i = 0; i < failures.size(); ++i)
        expect_json_failure(failures[i], "invalid string encoding rejection");
}

static std::string nested_array_document(const int depth)
{
    std::string text(static_cast<size_t>(depth), '[');
    text += "null";
    text.append(static_cast<size_t>(depth), ']');
    return text;
}

static void test_json_depth_limit_is_exactly_64()
{
    char error[512] = {};
    json_value value;
    check(load_text(value, nested_array_document(64), error), error);
    check(value.kind == json_array, "depth 64 accepted");
    expect_json_failure(nested_array_document(65), "depth 65 rejected");
}

static void test_json_size_limit_is_exactly_16_mib()
{
    static const size_t maximum = 16u * 1024u * 1024u;
    char error[512] = {};
    json_value value;
    std::string exact = "null";
    exact.append(maximum - exact.size(), ' ');
    check(exact.size() == maximum, "exact JSON ceiling fixture size");
    check(load_text(value, exact, error), error);
    check(value.kind == json_null, "exact 16 MiB JSON accepted");

    exact.push_back(' ');
    json_value prior = sentinel_value();
    check(!load_text(prior, exact, error), "JSON above 16 MiB rejected");
    check(std::strstr(error, json_fixture_path) != NULL,
          "oversize JSON path diagnostic");
    check(std::strstr(error, "exceeds 16 MiB") != NULL,
          "oversize JSON limit diagnostic");
    check_sentinel(prior, "oversize JSON failure is transactional");
}

static void test_json_file_errors_are_transactional()
{
    const char* const missing = "/tmp/test_scene_json_missing.json";
    std::remove(missing);
    char error[512] = {};
    json_value value = sentinel_value();
    check(!json_document_load(value, missing, error, 512),
          "missing JSON file rejected");
    check(std::strstr(error, missing) != NULL, "missing JSON path diagnostic");
    check_sentinel(value, "missing JSON failure is transactional");

    check(!json_document_load(value, NULL, error, 512),
          "null JSON path rejected");
    check(std::strstr(error, "<null>") != NULL, "null JSON path diagnostic");
    check_sentinel(value, "null path JSON failure is transactional");
    check(!json_document_load(value, "", NULL, 0),
          "empty JSON path rejected without error buffer");
    check_sentinel(value, "empty path JSON failure is transactional");
}

static void check_sha_vector(
    const char* path, const std::string& contents, const char* expected)
{
    write_text(path, contents);
    char error[512] = {};
    std::string digest = "sentinel";
    check(sha256_file_hex(digest, path, error, 512), error);
    check(digest == expected, "SHA-256 known vector");
    check(digest.size() == 64, "SHA-256 digest length");
    for (size_t i = 0; i < digest.size(); ++i)
        check((digest[i] >= '0' && digest[i] <= '9') ||
              (digest[i] >= 'a' && digest[i] <= 'f'),
              "SHA-256 lowercase hexadecimal output");
}

static void test_sha256_known_vectors_and_file_errors()
{
    check_sha_vector("/tmp/test_sha_empty", "",
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855");
    check_sha_vector("/tmp/test_sha_abc", "abc",
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad");
    check_sha_vector("/tmp/test_sha_multiblock",
        "abcdbcdecdefdefgefghfghighijhijk"
        "ijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039"
        "a33ce45964ff2167f6ecedd419db06c1");
    check_sha_vector("/tmp/test_sha_stream", std::string(1000000, 'a'),
        "cdc76e5c9914fb9281a1c7e284d73e67"
        "f1809a48a497200e046d39ccc7112cd0");

    const char* const missing = "/tmp/test_sha_missing_task2";
    std::remove(missing);
    char error[512] = {};
    std::string digest = "sentinel";
    check(!sha256_file_hex(digest, missing, error, 512),
          "missing SHA file rejected");
    check(std::strstr(error, missing) != NULL, "missing SHA path diagnostic");
    check(digest == "sentinel", "missing SHA failure is transactional");

    check(!sha256_file_hex(digest, NULL, error, 512),
          "null SHA path rejected");
    check(std::strstr(error, "<null>") != NULL, "null SHA path diagnostic");
    check(digest == "sentinel", "null SHA failure is transactional");
    check(!sha256_file_hex(digest, "", NULL, 0),
          "empty SHA path rejected without error buffer");
    check(digest == "sentinel", "empty SHA failure is transactional");
}

static void test_scene_numeric_precision_helpers()
{
    char error[512] = {};
    json_value promoted_max, mesh_max, encoded_coordinate, unrounded_coordinate;
    promoted_max.kind = mesh_max.kind = encoded_coordinate.kind =
        unrounded_coordinate.kind = json_number;
    promoted_max.number_value = -0.9800000004470348;
    mesh_max.number_value = -0.9800000190734863;
    encoded_coordinate.number_value =
        static_cast<double>(static_cast<float>(0.02));
    unrounded_coordinate.number_value = 0.02;
    double heightfield_x = 0.0, mesh_x = 0.0;
    float coordinate = 0.0f;
    check(scene_number_double(heightfield_x, promoted_max,
          "heightfield maximum", error, sizeof(error)), error);
    check(scene_number_double(mesh_x, mesh_max,
          "mesh maximum", error, sizeof(error)), error);
    check(heightfield_x != mesh_x &&
          static_cast<float>(heightfield_x) == static_cast<float>(mesh_x),
          "binary64 bounds preserve promoted-vs-OBJ maximum");
    check(scene_number_binary32(coordinate, encoded_coordinate,
          "published coordinate", error, sizeof(error)), error);
    check(!scene_number_binary32(coordinate, unrounded_coordinate,
          "published coordinate", error, sizeof(error)),
          "unrounded coordinate is not binary32-authoritative");
    check(scene_inside(bounds2{-1.0f, -2.0f, 1.0f, 2.0f},
                       -1.0f, 2.0f),
          "bounds containment includes published edges");
    check(!scene_inside(bounds2{-1.0f, -2.0f, 1.0f, 2.0f},
                        std::numeric_limits<float>::infinity(), 0.0f),
          "bounds containment rejects non-finite values");
    bounds3d authoritative;
    authoritative.minimum = point3d{-1.0, 0.0, -1.0};
    authoritative.maximum =
        point3d{-0.9800000004470348, 0.0, -0.9800000004470348};
    const float beyond = std::nextafter(
        static_cast<float>(authoritative.maximum.x),
        std::numeric_limits<float>::infinity());
    check(!scene_bounds2_inside_binary64(
            bounds2{-1.0f, -1.0f, beyond, beyond}, authoritative),
          "binary32 containment does not narrow binary64 bounds");
    json_value invalid_array = parse_json_text("[1.0,0.02]");
    float array_output[2] = {7.0f, 8.0f};
    check(!scene_binary32_array(array_output, 2, invalid_array,
          "published coordinates", error, sizeof(error)),
          "array rejects a later non-authoritative coordinate");
    check(array_output[0] == 7.0f && array_output[1] == 8.0f,
          "failed binary32 array parse is transactional");

    int sample_count = 123456789;
    check(!scene_route_sample_count(
              sample_count, std::make_pair(0.0f, 0.0f),
              std::make_pair(1.0f, 0.0f), 0x1p-31f),
          "route sample count rejects a rounded value above INT_MAX");
    check(sample_count == 123456789,
          "route sample count overflow rejection preserves prior output");
}

struct manifest_fixture_hashes
{
    std::string database, features, support, index, validation;
};

static std::string validation_fixture_json()
{
    std::ostringstream out;
    out << "{\"schema\":\"g1-terrain-validation/v1\",";
    const char* names[] = {
        "duration_error_s", "fk_max_error_m",
        "quaternion_norm_max_error"};
    for (int field = 0; field < 3; ++field) {
        if (field != 0) out << ',';
        out << '\"' << names[field] << "\":[";
        for (int i = 0; i < 1770; ++i) {
            if (i != 0) out << ',';
            out << "0.0";
        }
        out << ']';
    }
    out << '}';
    return out.str();
}

static std::string manifest_fixture_json(
    const std::string& semantics,
    const std::string& validation,
    const manifest_fixture_hashes& hashes)
{
    static const char* names[G1_BoneCount] = {
        "Simulation", "Hips", "LeftHipPitch", "LeftHipRoll",
        "LeftHipYaw", "LeftKnee", "LeftAnkle", "LeftToe",
        "RightHipPitch", "RightHipRoll", "RightHipYaw", "RightKnee",
        "RightAnkle", "RightToe", "Spine", "Spine1", "Spine2",
        "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw",
        "LeftElbow", "LeftWristRoll", "LeftWristPitch", "LeftWrist",
        "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw",
        "RightElbow", "RightWristRoll", "RightWristPitch", "RightWrist"};
    static const int parents[G1_BoneCount] = {
        -1,0,1,2,3,4,5,6,1,8,9,10,11,12,1,14,
        15,16,17,18,19,20,21,22,16,24,25,26,27,28,29};
    std::ostringstream out;
    out << "{\"schema\":\"g1-terrain-artifacts/v2\",";
    out << "\"output_fps\":25.0,\"feature_dimensions\":31,";
    out << "\"terrain_dimensions\":4,\"support_dimensions\":3,";
    out << "\"terrain_feature_distances_m\":[0.25,0.5,0.75,1.0],";
    out << "\"total_clips\":1770,\"grail_clips\":1769,";
    out << "\"skipped_clips\":0,\"database_frames\":1770,";
    out << "\"diagnostic_mode\":false,\"sources\":[";
    for (int i = 0; i < 1770; ++i) {
        if (i != 0) out << ',';
        out << "{\"name\":\"source-" << i
            << "\",\"output_frames\":1,\"range_start\":" << i
            << ",\"range_stop\":" << i + 1
            << ",\"source_fps\":25.0,\"source_frame_map\":[0],"
               "\"source_frames\":1,\"terrain_id\":\"flat\"}";
    }
    out << "],\"skeleton\":{\"names\":[";
    for (int i = 0; i < G1_BoneCount; ++i) {
        if (i != 0) out << ',';
        out << dump_json_string(names[i]);
    }
    out << "],\"parents\":[";
    for (int i = 0; i < G1_BoneCount; ++i) {
        if (i != 0) out << ',';
        out << parents[i];
    }
    out << "],\"signature\":\"" << G1_SkeletonSignature << "\"},";
    out << "\"contact\":{\"speed_threshold\":0.15,";
    out << "\"height_threshold\":0.06,\"median_filter_frames\":3},";
    out << "\"surface\":{\"semantics\":" << semantics
        << ",\"signature\":\"" << expected_surface_signature << "\"},";
    out << "\"database\":{\"path\":\"database.bin\",";
    out << "\"schema\":\"holden-database/v1\",\"sha256\":\""
        << hashes.database << "\"},";
    out << "\"sidecars\":{\"terrain_features\":{";
    out << "\"path\":\"terrain_features.bin\",\"schema\":\"G1TF/v1\",";
    out << "\"version\":1,\"dimensions\":4,\"sha256\":\""
        << hashes.features << "\"},\"terrain_support\":{";
    out << "\"path\":\"terrain_support.bin\",\"schema\":\"G1SP/v1\",";
    out << "\"version\":1,\"dimensions\":3,\"columns\":[";
    out << "\"source_root_height_m\",\"source_left_toe_height_m\",";
    out << "\"source_right_toe_height_m\"],\"sha256\":\""
        << hashes.support << "\"}},";
    out << "\"scene_index\":{\"path\":\"scenes/index.json\",";
    out << "\"schema\":\"g1-terrain-scene-index/v1\",\"sha256\":\""
        << hashes.index << "\"},";
    out << "\"validation_file\":{\"path\":\"validation.json\",";
    out << "\"schema\":\"g1-terrain-validation/v1\",\"sha256\":\""
        << hashes.validation << "\"},";
    out << "\"validation\":" << validation << '}';
    return out.str();
}

static motion_pack_manifest manifest_sentinel()
{
    motion_pack_manifest manifest;
    manifest.output_fps = 91.0f;
    manifest.feature_dimensions = 92;
    manifest.terrain_dimensions = 93;
    manifest.support_dimensions = 94;
    manifest.total_clips = 95;
    manifest.grail_clips = 96;
    manifest.skipped_clips = 97;
    manifest.database_frames = 98;
    manifest.diagnostic_mode = true;
    manifest.sources.push_back(motion_source_record{"sentinel", "terrain", 4, 9});
    manifest.surface.signature = "sentinel-signature";
    manifest.surface.coordinate_signature = "sentinel-coordinate";
    manifest.surface.heightfield_interpolation = "sentinel-interpolation";
    manifest.surface.heightfield_diagonal = "sentinel-diagonal";
    manifest.surface.cell_size = 3.0f;
    manifest.surface.exterior_height = 4.0f;
    manifest.database.path = "sentinel-db";
    manifest.database.schema = "sentinel-schema";
    manifest.database.sha256 = "sentinel-sha";
    manifest.database.version = 12;
    manifest.database.dimensions = 13;
    manifest.database.columns.push_back("sentinel-column");
    manifest.terrain_features = manifest.database;
    manifest.terrain_support = manifest.database;
    manifest.scene_index = manifest.database;
    manifest.validation_file = manifest.database;
    return manifest;
}

static void test_manifest_surface_contract_is_exact_and_transactional()
{
    namespace fs = std::filesystem;
    const fs::path root = "/tmp/test_scene_manifest";
    fs::remove_all(root);
    fs::create_directories(root / "scenes");
    write_text((root / "database.bin").c_str(), "database");
    write_text((root / "terrain_features.bin").c_str(), "features");
    write_text((root / "terrain_support.bin").c_str(), "support");
    write_text((root / "scenes/index.json").c_str(), "{}");
    const std::string validation = validation_fixture_json();
    write_text((root / "validation.json").c_str(), validation);
    manifest_fixture_hashes hashes;
    hashes.database = file_sha(root / "database.bin");
    hashes.features = file_sha(root / "terrain_features.bin");
    hashes.support = file_sha(root / "terrain_support.bin");
    hashes.index = file_sha(root / "scenes/index.json");
    hashes.validation = file_sha(root / "validation.json");

    json_value semantics = parse_json_text(expected_surface_semantics_json);
    check(semantics.kind == json_object &&
          semantics.object_value.size() == 43,
          "surface fixture contains all 43 committed keys");
    write_text((root / "manifest.json").c_str(),
        manifest_fixture_json(dump_json(semantics), validation, hashes));
    char error[1024] = {};
    motion_pack_manifest loaded = manifest_sentinel();
    check(motion_manifest_load_and_verify(
        loaded, root.c_str(), error, sizeof(error)), error);
    check(loaded.surface.signature == expected_surface_signature &&
          loaded.sources.size() == 1770,
          "exact 43-key manifest surface accepted");

    const motion_pack_manifest sentinel = manifest_sentinel();
    const auto expect_rejection = [&](const json_value& candidate,
                                      const char* label) {
        write_text((root / "manifest.json").c_str(),
            manifest_fixture_json(dump_json(candidate), validation, hashes));
        motion_pack_manifest output = sentinel;
        error[0] = '\0';
        check(!motion_manifest_load_and_verify(
            output, root.c_str(), error, sizeof(error)), label);
        check(manifest_equal(output, sentinel),
              "manifest surface failure is transactional");
    };

    for (size_t i = 0; i < semantics.object_value.size(); ++i) {
        json_value missing = semantics;
        missing.object_value.erase(missing.object_value.begin() +
                                   static_cast<std::ptrdiff_t>(i));
        expect_rejection(missing, "missing surface semantic rejected");

        json_value changed = semantics;
        json_value& value = changed.object_value[i].second;
        if (value.kind == json_string) value.string_value += "-changed";
        else if (value.kind == json_number) value.number_value += 1.0;
        else if (value.kind == json_array)
            value.array_value[0].number_value += 1.0;
        expect_rejection(changed, "changed surface semantic rejected");

        json_value wrong_type = semantics;
        wrong_type.object_value[i].second = json_value();
        wrong_type.object_value[i].second.kind = json_boolean;
        wrong_type.object_value[i].second.boolean_value = true;
        expect_rejection(wrong_type, "surface semantic type change rejected");
    }
    json_value extra = semantics;
    json_value extra_value;
    extra_value.kind = json_string;
    extra_value.string_value = "forbidden";
    extra.object_value.push_back(std::make_pair("extra", extra_value));
    expect_rejection(extra, "extra surface semantic rejected");

    json_value former_subset = semantics;
    former_subset.object_value.resize(9);
    expect_rejection(former_subset, "former nine-key surface rejected");

    write_text((root / "manifest.json").c_str(),
        manifest_fixture_json(dump_json(semantics), validation, hashes));
    motion_pack_manifest valid;
    check(motion_manifest_load_and_verify(
        valid, root.c_str(), error, sizeof(error)), error);
    check(motion_source_for_frame(valid, 0) == 0 &&
          motion_source_for_frame(valid, 1769) == 1769 &&
          motion_source_for_frame(valid, 1770) == -1,
          "fixture source ownership boundaries");

    write_text((root / "validation.json").c_str(), "{");
    motion_pack_manifest tamper_output = sentinel;
    error[0] = '\0';
    check(!motion_manifest_load_and_verify(
        tamper_output, root.c_str(), error, sizeof(error)),
        "changed validation bytes rejected");
    check(std::strstr(error, "SHA-256") != NULL &&
          std::strstr(error, "JSON") == NULL,
          "validation SHA check precedes validation JSON parse");
    check(manifest_equal(tamper_output, sentinel),
          "validation byte tamper preserves prior manifest");
    write_text((root / "validation.json").c_str(), validation);

    std::string extreme_ranges = manifest_fixture_json(
        dump_json(semantics), validation, hashes);
    const std::string ordinary_range =
        "\"range_start\":1,\"range_stop\":2";
    const size_t range_position = extreme_ranges.find(ordinary_range);
    check(range_position != std::string::npos,
          "locate source range for overflow fixture");
    extreme_ranges.replace(range_position, ordinary_range.size(),
        "\"range_start\":1,\"range_stop\":-2147483648");
    write_text((root / "manifest.json").c_str(), extreme_ranges);
    motion_pack_manifest range_output = sentinel;
    check(!motion_manifest_load_and_verify(
        range_output, root.c_str(), error, sizeof(error)),
        "extreme source ranges rejected without overflow");
    check(manifest_equal(range_output, sentinel),
          "extreme range failure preserves prior manifest");
}

static std::string fixture_descriptor_digest(const int index)
{
    static const char hex[] = "0123456789abcdef";
    std::string digest(64, '0');
    digest[62] = hex[(index >> 4) & 15];
    digest[63] = hex[index & 15];
    return digest;
}

static std::string catalog_fixture_json(const std::string& mutation)
{
    std::ostringstream out;
    out << "{\"schema\":\"g1-terrain-scene-index/v1\",";
    out << "\"default_scene_id\":\"grail-curb-default\",";
    out << "\"scene_ids\":[";
    for (int i = 0; i < 14; ++i) {
        if (i != 0) out << ',';
        out << dump_json_string(expected_scene_ids[i]);
    }
    out << "],\"coordinate_signature\":";
    out << "\"holden-y-up-right-handed-forward-plus-z\",";
    out << "\"surface_signature\":\"" << expected_surface_signature << '\"';
    if (mutation != "missing-scenes") {
        out << ",\"scenes\":[";
        for (int i = 0; i < 14; ++i) {
            if (i != 0) out << ',';
            int descriptor = i;
            if (mutation == "reordered") {
                if (i == 0) descriptor = 1;
                else if (i == 1) descriptor = 0;
            }
            const std::string id = expected_scene_ids[descriptor];
            std::string path = std::string("scenes/") + id + "/scene.json";
            std::string digest = fixture_descriptor_digest(descriptor);
            if (mutation == "duplicate-path" && i == 1)
                path = std::string("scenes/") + expected_scene_ids[0] +
                    "/scene.json";
            if (mutation == "unsafe-path" && i == 0)
                path = "scenes/../fixture/scene.json";
            if (mutation == "uppercase-digest" && i == 0)
                digest = std::string(63, '0') + "A";
            if (mutation == "short-digest" && i == 0)
                digest = "abc";
            if (mutation == "duplicate-digest" && i == 1)
                digest = fixture_descriptor_digest(0);
            out << "{\"id\":" << dump_json_string(id)
                << ",\"path\":" << dump_json_string(path)
                << ",\"sha256\":" << dump_json_string(digest) << '}';
        }
        out << ']';
    }
    if (mutation == "extra-key") out << ",\"extra\":true";
    out << '}';
    return out.str();
}

static motion_pack_manifest catalog_manifest(const std::string& digest)
{
    motion_pack_manifest manifest;
    manifest.surface.signature = expected_surface_signature;
    manifest.surface.coordinate_signature =
        "holden-y-up-right-handed-forward-plus-z";
    manifest.scene_index.path = "scenes/index.json";
    manifest.scene_index.schema = "g1-terrain-scene-index/v1";
    manifest.scene_index.sha256 = digest;
    return manifest;
}

static scene_catalog catalog_sentinel()
{
    scene_catalog catalog;
    catalog.default_scene_id = "sentinel-default";
    catalog.coordinate_signature = "sentinel-coordinate";
    catalog.surface_signature = "sentinel-surface";
    catalog.ids.push_back("sentinel-id");
    catalog.scenes.push_back(scene_descriptor{
        "sentinel-descriptor", "sentinel/path", "sentinel-sha"});
    return catalog;
}

static void test_catalog_contract_paths_and_tamper_order()
{
    namespace fs = std::filesystem;
    const fs::path root = "/tmp/test_scene_catalog";
    const fs::path index = root / "scenes/index.json";
    fs::remove_all(root);
    fs::create_directories(index.parent_path());
    write_text(index.c_str(), catalog_fixture_json("valid"));
    motion_pack_manifest manifest = catalog_manifest(file_sha(index.string()));
    char error[1024] = {};
    scene_catalog loaded = catalog_sentinel();
    check(scene_catalog_load(
        loaded, root.c_str(), manifest, error, sizeof(error)), error);
    check(loaded.ids.size() == 14 && loaded.scenes.size() == 14 &&
          loaded.ids.front() == "grail-curb-default",
          "exact ordered catalog accepted");
    check(scene_catalog_find(loaded, "blocked-course") == 13 &&
          scene_catalog_find(loaded, "../escape") == -1 &&
          scene_catalog_find(loaded, NULL) == -1,
          "catalog lookup accepts exact IDs only");

    const char* mutations[] = {
        "missing-scenes", "extra-key", "reordered", "duplicate-path",
        "unsafe-path", "uppercase-digest", "short-digest",
        "duplicate-digest"};
    const scene_catalog sentinel = catalog_sentinel();
    for (size_t i = 0; i < sizeof(mutations) / sizeof(mutations[0]); ++i) {
        write_text(index.c_str(), catalog_fixture_json(mutations[i]));
        manifest = catalog_manifest(file_sha(index.string()));
        scene_catalog output = sentinel;
        error[0] = '\0';
        check(!scene_catalog_load(
            output, root.c_str(), manifest, error, sizeof(error)),
            "malformed catalog rejected");
        check(catalog_equal(output, sentinel),
              "catalog rejection preserves complete prior value");
    }

    write_text(index.c_str(), catalog_fixture_json("valid"));
    manifest = catalog_manifest(file_sha(index.string()));
    scene_catalog prior = catalog_sentinel();
    check(scene_catalog_load(
        prior, root.c_str(), manifest, error, sizeof(error)), error);
    write_text(index.c_str(), "{");
    scene_catalog output = prior;
    error[0] = '\0';
    check(!scene_catalog_load(
        output, root.c_str(), manifest, error, sizeof(error)),
        "changed index bytes rejected");
    check(std::strstr(error, "SHA-256") != NULL &&
          std::strstr(error, "JSON") == NULL,
          "index SHA check precedes index JSON parse");
    check(catalog_equal(output, prior),
          "index byte tamper preserves prior catalog");
}

static void append_u32(std::vector<unsigned char>& out, const uint32_t value)
{
    out.push_back(static_cast<unsigned char>(value));
    out.push_back(static_cast<unsigned char>(value >> 8));
    out.push_back(static_cast<unsigned char>(value >> 16));
    out.push_back(static_cast<unsigned char>(value >> 24));
}

static void append_float(std::vector<unsigned char>& out, const float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32(out, bits);
}

static std::vector<unsigned char> fixture_heightfield_bytes()
{
    std::vector<unsigned char> out;
    out.insert(out.end(), {'G','1','H','F'});
    append_u32(out, 2);
    append_u32(out, 2);
    append_u32(out, 2);
    append_float(out, -1.0f);
    append_float(out, -1.0f);
    append_float(out, 0.02f);
    append_float(out, 0.0f);
    for (int i = 0; i < 4; ++i) append_float(out, 0.0f);
    return out;
}

static std::vector<unsigned char> fixture_walkability_bytes(
    const bool blocked)
{
    std::vector<unsigned char> out;
    out.insert(out.end(), {'G','1','W','M'});
    append_u32(out, 1);
    append_u32(out, 2);
    append_u32(out, 2);
    if (blocked) out.insert(out.end(), {1, 1, 0, 0});
    else out.insert(out.end(), {2, 2, 2, 2});
    return out;
}

static std::vector<unsigned char> fixture_walkability_uniform_bytes(
    const unsigned char classification)
{
    std::vector<unsigned char> out;
    out.insert(out.end(), {'G','1','W','M'});
    append_u32(out, 1);
    append_u32(out, 2);
    append_u32(out, 2);
    out.insert(out.end(), 4, classification);
    return out;
}

struct scene_asset_hashes
{
    std::string terrain, mesh, walkability;
};

static int fixture_scene_index(const std::string& id)
{
    for (int i = 0; i < 14; ++i)
        if (id == expected_scene_ids[i]) return i;
    return -1;
}

static std::string fixture_route_json(
    const std::string& id, const char* outcome, const int classification)
{
    std::ostringstream out;
    out << "{\"id\":" << dump_json_string(id)
        << ",\"waypoints_xz\":[[-1.0,-1.0],"
        << "[-1.0,-0.9800000190734863]],"
        << "\"expected_outcome\":" << dump_json_string(outcome)
        << ",\"walkability_class\":" << classification
        << ",\"landing_hold_seconds\":0.0}";
    return out.str();
}

static std::string fixture_routes_json(
    const std::string& id, const std::string& mutation)
{
    const int scene = fixture_scene_index(id);
    check(scene >= 0, "fixture route scene is in the locked catalog");
    std::vector<std::string> routes;
    for (int route = 0; route < expected_route_counts[scene]; ++route) {
        std::string route_id = expected_route_ids[scene][route];
        const char* outcome = expected_route_outcomes[scene][route];
        int classification = expected_route_classes[scene][route];
        if (mutation == "rename" && route == 0) route_id = "renamed-route";
        if (mutation == "pair") {
            if (classification == 1) {
                outcome = "traverse-or-safe-stop";
                classification = 2;
            } else {
                outcome = "traverse";
                classification = 1;
            }
        }
        routes.push_back(fixture_route_json(
            route_id, outcome, classification));
    }
    const std::string extra =
        "{\"id\":\"extra-route\",\"waypoints_xz\":"
        "[[-1.0,-1.0],[-1.0,-0.9800000190734863]],"
        "\"expected_outcome\":\"traverse-or-safe-stop\","
        "\"walkability_class\":2,\"landing_hold_seconds\":0.0}";
    if (mutation == "extra") routes.push_back(extra);
    std::string out = "[";
    for (size_t i = 0; i < routes.size(); ++i) {
        const size_t index = mutation == "reverse" ? routes.size() - 1 - i : i;
        if (i != 0) out.push_back(',');
        out += routes[index];
    }
    out.push_back(']');
    return out;
}

static std::string fixture_scene_json(
    const std::string& id,
    const std::string& route_mutation,
    const scene_asset_hashes& hashes)
{
    const bool blocked = id == "blocked-course";
    std::ostringstream out;
    out << "{\"schema\":\"g1-terrain-scene/v1\",\"id\":"
        << dump_json_string(id) << ",\"label\":\"Fixture Scene\",";
    out << "\"provenance\":{\"kind\":\"procedural\",";
    out << "\"source_ids\":[],\"parameters\":{}},";
    out << "\"coordinate_signature\":";
    out << "\"holden-y-up-right-handed-forward-plus-z\",";
    out << "\"surface_signature\":\"" << expected_surface_signature << "\",";
    out << "\"terrain_feature_distances_m\":[0.25,0.5,0.75,1.0],";
    out << "\"heightfield\":{\"path\":\"terrain.bin\",";
    out << "\"schema\":\"G1HF/v2\",\"version\":2,\"nx\":2,\"nz\":2,";
    out << "\"origin_x\":-1.0,\"origin_z\":-1.0,";
    out << "\"cell_size_m\":0.019999999552965164,";
    out << "\"exterior_height_m\":0.0,";
    out << "\"interpolation\":\"fixed-diagonal-triangles\",";
    out << "\"diagonal\":\"min-x-min-z_to_max-x-max-z\",";
    out << "\"sha256\":\"" << hashes.terrain << "\"},";
    out << "\"mesh\":{\"path\":\"terrain.obj\",";
    out << "\"schema\":\"obj/v1\",\"sha256\":\""
        << hashes.mesh << "\"},";
    out << "\"walkability\":{\"path\":\"walkability.bin\",";
    out << "\"schema\":\"G1WM/v1\",\"version\":1,\"nx\":2,\"nz\":2,";
    out << "\"classes\":{\"blocked\":0,\"certified\":1,\"stress\":2},";
    out << "\"sha256\":\"" << hashes.walkability << "\"},";
    out << "\"bounds\":{\"mesh_min_xyz\":[-1.0,0.0,-1.0],";
    out << "\"mesh_max_xyz\":[-0.9800000190734863,0.0,-0.9800000190734863],";
    out << "\"heightfield_min_xyz\":[-1.0,0.0,-1.0],";
    out << "\"heightfield_max_xyz\":[-0.9800000004470348,0.0,-0.9800000004470348],";
    out << "\"playable_min_xz\":[-1.0,-1.0],";
    out << "\"playable_max_xz\":[-0.9800000190734863,-0.9800000190734863],";
    out << "\"lookahead_min_xz\":[-1.0,-1.0],";
    out << "\"lookahead_max_xz\":[-0.9800000190734863,-0.9800000190734863]},";
    out << "\"spawn\":{\"position\":[-1.0,0.0,-1.0],";
    out << "\"yaw_radians\":0.0},";
    if (blocked) {
        out << "\"regions\":{\"certified\":[{\"id\":\"approach\",";
        out << "\"bounds_xz\":[-1.0,-0.9800000190734863,-1.0,-0.9800000190734863]}],";
        out << "\"stress\":[],\"blocked\":[{\"id\":\"wall\",";
        out << "\"bounds_xz\":[-1.0,-0.9800000190734863,-1.0,-0.9800000190734863]}]},";
    } else {
        out << "\"regions\":{\"certified\":[],\"stress\":[{";
        out << "\"id\":\"course\",\"bounds_xz\":";
        out << "[-1.0,-0.9800000190734863,-1.0,-0.9800000190734863]}],";
        out << "\"blocked\":[]},";
    }
    out << "\"routes\":" << fixture_routes_json(id, route_mutation) << '}';
    return out.str();
}

static scene_asset_hashes prepare_scene_assets(
    const std::filesystem::path& root,
    const std::string& id,
    const std::vector<unsigned char>& walkability)
{
    namespace fs = std::filesystem;
    const fs::path directory = root / "scenes" / id;
    fs::create_directories(directory);
    write_bytes((directory / "terrain.bin").string(),
                fixture_heightfield_bytes());
    write_text((directory / "terrain.obj").c_str(), "fixture mesh\n");
    write_bytes((directory / "walkability.bin").string(),
                walkability);
    scene_asset_hashes hashes;
    hashes.terrain = file_sha((directory / "terrain.bin").string());
    hashes.mesh = file_sha((directory / "terrain.obj").string());
    hashes.walkability = file_sha((directory / "walkability.bin").string());
    return hashes;
}

static scene_asset_hashes prepare_scene_assets(
    const std::filesystem::path& root,
    const std::string& id,
    const bool blocked)
{
    return prepare_scene_assets(
        root, id, fixture_walkability_bytes(blocked));
}

static scene_catalog one_scene_catalog(
    const std::string& id, const std::string& digest)
{
    scene_catalog catalog;
    catalog.default_scene_id = id;
    catalog.coordinate_signature =
        "holden-y-up-right-handed-forward-plus-z";
    catalog.surface_signature = expected_surface_signature;
    catalog.ids.push_back(id);
    catalog.scenes.push_back(scene_descriptor{
        id, std::string("scenes/") + id + "/scene.json", digest});
    return catalog;
}

static motion_pack_manifest scene_fixture_manifest()
{
    motion_pack_manifest manifest;
    manifest.surface.signature = expected_surface_signature;
    manifest.surface.coordinate_signature =
        "holden-y-up-right-handed-forward-plus-z";
    manifest.surface.heightfield_interpolation = "fixed-diagonal-triangles";
    manifest.surface.heightfield_diagonal =
        "min-x-min-z_to_max-x-max-z";
    manifest.surface.cell_size = 0.02f;
    manifest.surface.exterior_height = 0.0f;
    return manifest;
}

static scene_pack pack_sentinel()
{
    scene_pack pack;
    pack.metadata.id = "sentinel-id";
    pack.metadata.label = "sentinel-label";
    pack.metadata.provenance_kind = "sentinel-kind";
    pack.metadata.provenance_source_ids.push_back("sentinel-source");
    pack.metadata.coordinate_signature = "sentinel-coordinate";
    pack.metadata.surface_signature = "sentinel-surface";
    pack.metadata.heightfield.path = "sentinel-heightfield";
    pack.metadata.heightfield.schema = "sentinel-heightfield-schema";
    pack.metadata.heightfield.sha256 = "sentinel-heightfield-sha";
    pack.metadata.heightfield.version = 7;
    pack.metadata.heightfield.dimensions = 8;
    pack.metadata.heightfield.columns.push_back("sentinel-column");
    pack.metadata.mesh = pack.metadata.heightfield;
    pack.metadata.walkability = pack.metadata.heightfield;
    pack.metadata.heightfield_nx = 9;
    pack.metadata.heightfield_nz = 10;
    pack.metadata.walkability_nx = 11;
    pack.metadata.walkability_nz = 12;
    pack.metadata.heightfield_origin_x = 13.0f;
    pack.metadata.heightfield_origin_z = 14.0f;
    pack.metadata.heightfield_cell_size = 15.0f;
    pack.metadata.heightfield_exterior_height = 16.0f;
    pack.metadata.mesh_bounds.minimum = point3d{1.0, 2.0, 3.0};
    pack.metadata.mesh_bounds.maximum = point3d{4.0, 5.0, 6.0};
    pack.metadata.heightfield_bounds.minimum = point3d{7.0, 8.0, 9.0};
    pack.metadata.heightfield_bounds.maximum = point3d{10.0, 11.0, 12.0};
    pack.metadata.playable_bounds = bounds2{1.0f, 2.0f, 3.0f, 4.0f};
    pack.metadata.lookahead_bounds = bounds2{5.0f, 6.0f, 7.0f, 8.0f};
    pack.metadata.spawn_position = vec3(17.0f, 18.0f, 19.0f);
    pack.metadata.spawn_yaw = 20.0f;
    pack.metadata.certified_regions.push_back(
        scene_region{"sentinel-certified", bounds2{1,2,3,4}});
    pack.metadata.stress_regions.push_back(
        scene_region{"sentinel-stress", bounds2{5,6,7,8}});
    pack.metadata.blocked_regions.push_back(
        scene_region{"sentinel-blocked", bounds2{9,10,11,12}});
    scene_route route;
    route.id = "sentinel-route";
    route.expected_outcome = "sentinel-outcome";
    route.walkability_class = 21;
    route.landing_hold_seconds = 22.0f;
    route.waypoints_xz.push_back(std::make_pair(23.0f, 24.0f));
    pack.metadata.routes.push_back(route);
    pack.terrain.version = 25;
    pack.terrain.nx = 26;
    pack.terrain.nz = 27;
    pack.terrain.origin_x = 28.0f;
    pack.terrain.origin_z = 29.0f;
    pack.terrain.cell_size = 30.0f;
    pack.terrain.exterior_height = 31.0f;
    pack.terrain.heights.resize(2);
    pack.terrain.heights(0) = 32.0f;
    pack.terrain.heights(1) = 33.0f;
    pack.walkability.nx = 34;
    pack.walkability.nz = 35;
    pack.walkability.cells.resize(2);
    pack.walkability.cells(0) = 36;
    pack.walkability.cells(1) = 37;
    pack.scene_path = "sentinel-scene-path";
    pack.terrain_path = "sentinel-terrain-path";
    pack.mesh_path = "sentinel-mesh-path";
    pack.walkability_path = "sentinel-walkability-path";
    return pack;
}

static void test_scene_candidate_hostile_dimensions_are_transactional()
{
    const int hostile_dimensions[][2] = {
        {1, 2},
        {std::numeric_limits<int>::max(),
         std::numeric_limits<int>::max()},
    };
    for (size_t i = 0;
         i < sizeof(hostile_dimensions) / sizeof(hostile_dimensions[0]); ++i) {
        scene_pack candidate = pack_sentinel();
        const int nx = hostile_dimensions[i][0];
        const int nz = hostile_dimensions[i][1];
        candidate.metadata.heightfield_nx = nx;
        candidate.metadata.heightfield_nz = nz;
        candidate.metadata.walkability_nx = nx;
        candidate.metadata.walkability_nz = nz;
        candidate.metadata.heightfield_origin_x = candidate.terrain.origin_x;
        candidate.metadata.heightfield_origin_z = candidate.terrain.origin_z;
        candidate.metadata.heightfield_cell_size = candidate.terrain.cell_size;
        candidate.metadata.heightfield_exterior_height =
            candidate.terrain.exterior_height;
        candidate.terrain.version = 2;
        candidate.terrain.nx = nx;
        candidate.terrain.nz = nz;
        candidate.walkability.nx = nx;
        candidate.walkability.nz = nz;
        const scene_pack prior = candidate;
        char error[512] = {};
        check(!scene_candidate_validate(candidate, error, sizeof(error)),
              "hostile scene grid dimensions are rejected");
        check(std::strstr(error, "dimension") != NULL,
              "hostile scene grid dimensions have a stable diagnostic");
        check(pack_equal(candidate, prior),
              "hostile dimension rejection preserves the complete candidate");
    }
}

static void test_scene_descriptor_hash_precedes_json_parse()
{
    namespace fs = std::filesystem;
    const fs::path root = "/tmp/test_scene_descriptor_order";
    const fs::path scene = root / "scenes/fixture/scene.json";
    fs::remove_all(root);
    fs::create_directories(scene.parent_path());
    write_text(scene.c_str(), "{ invalid JSON");
    scene_catalog catalog;
    catalog.ids.push_back("fixture");
    catalog.scenes.push_back(scene_descriptor{
        "fixture", "scenes/fixture/scene.json", std::string(64, '0')});
    motion_pack_manifest manifest = scene_fixture_manifest();
    scene_pack active = pack_sentinel();
    const scene_pack prior = active;
    char error[1024] = {};
    check(!scene_pack_load(active, root.c_str(), manifest, catalog, 0,
          error, sizeof(error)), "wrong scene descriptor digest rejected");
    check(std::strstr(error, "SHA-256") != NULL &&
          std::strstr(error, "JSON") == NULL,
          "scene descriptor SHA check precedes JSON parse");
    check(pack_equal(active, prior),
          "scene descriptor SHA failure preserves every active field");

    catalog.scenes[0].sha256 = file_sha(scene.string());
    error[0] = '\0';
    check(!scene_pack_load(active, root.c_str(), manifest, catalog, 0,
          error, sizeof(error)), "verified invalid scene JSON rejected");
    check(std::strstr(error, "JSON") != NULL,
          "verified scene bytes reach JSON parser");
    check(pack_equal(active, prior),
          "scene JSON failure preserves every active field");
}

static void expect_pack_failure_preserves(
    scene_pack& active,
    const scene_pack& prior,
    const std::filesystem::path& root,
    const motion_pack_manifest& manifest,
    const scene_catalog& catalog,
    const char* message,
    const char* diagnostic)
{
    char error[1024] = {};
    check(!scene_pack_load(active, root.c_str(), manifest, catalog, 0,
          error, sizeof(error)), message);
    check(diagnostic == NULL || std::strstr(error, diagnostic) != NULL,
          "scene failure diagnostic");
    check(pack_equal(active, prior),
          "scene load failure preserves complete active pack");
}

static void test_scene_route_semantic_matrix_is_exact_and_transactional()
{
    namespace fs = std::filesystem;
    const fs::path root = "/tmp/test_scene_route_semantics";
    fs::remove_all(root);
    const motion_pack_manifest manifest = scene_fixture_manifest();
    for (int scene = 0; scene < 14; ++scene) {
        const std::string id = expected_scene_ids[scene];
        const unsigned char mutated_class =
            expected_route_classes[scene][0] == 1 ? 2 : 1;
        const scene_asset_hashes hashes = prepare_scene_assets(
            root, id, fixture_walkability_uniform_bytes(mutated_class));
        const fs::path descriptor = root / "scenes" / id / "scene.json";
        write_text(descriptor.c_str(), fixture_scene_json(id, "pair", hashes));
        const scene_catalog catalog =
            one_scene_catalog(id, file_sha(descriptor.string()));
        scene_pack active = pack_sentinel();
        const scene_pack prior = active;
        char error[1024] = {};
        check(!scene_pack_load(active, root.c_str(), manifest, catalog, 0,
              error, sizeof(error)),
              "scene-specific route outcome/class mutation is rejected");
        check(std::strstr(error, "outcome/class") != NULL,
              "scene-specific route tuple diagnostic");
        check(pack_equal(active, prior),
              "route tuple rejection preserves the complete active pack");
    }
}

static void test_scene_pack_tamper_chain_routes_and_transactionality()
{
    namespace fs = std::filesystem;
    const fs::path root = "/tmp/test_scene_pack";
    fs::remove_all(root);
    const motion_pack_manifest manifest = scene_fixture_manifest();
    const std::string grail = "grail-curb-default";
    const fs::path grail_dir = root / "scenes" / grail;
    const fs::path grail_scene = grail_dir / "scene.json";
    const scene_asset_hashes grail_hashes =
        prepare_scene_assets(root, grail, false);
    const std::string valid_scene =
        fixture_scene_json(grail, "valid", grail_hashes);
    write_text(grail_scene.c_str(), valid_scene);
    scene_catalog catalog = one_scene_catalog(grail, file_sha(grail_scene.string()));
    char error[1024] = {};
    scene_pack active = pack_sentinel();
    check(scene_pack_load(active, root.c_str(), manifest, catalog, 0,
          error, sizeof(error)), error);
    check(active.metadata.id == grail && active.terrain.version == 2 &&
          active.walkability.nx == active.terrain.nx &&
          active.walkability.nz == active.terrain.nz,
          "valid fixture scene pack loads");
    check(scene_route_find(active.metadata, "curb-forward") != NULL &&
          scene_route_find(active.metadata, "missing") == NULL &&
          scene_route_find(active.metadata, NULL) == NULL,
          "route lookup is exact and null-safe");
    const scene_pack grail_prior = active;

    const std::string syntactically_valid_semantic_change =
        fixture_scene_json(grail, "rename", grail_hashes);
    write_text(grail_scene.c_str(), syntactically_valid_semantic_change);
    error[0] = '\0';
    check(!scene_pack_load(active, root.c_str(), manifest, catalog, 0,
          error, sizeof(error)),
          "stale descriptor digest rejects valid changed scene JSON");
    check(std::strstr(error, "SHA-256") != NULL &&
          std::strstr(error, "route") == NULL,
          "scene SHA diagnostic precedes semantic parsing");
    check(pack_equal(active, grail_prior),
          "stale descriptor digest preserves complete active pack");
    const scene_catalog authenticated_semantic_change = one_scene_catalog(
        grail, file_sha(grail_scene.string()));
    error[0] = '\0';
    check(!scene_pack_load(active, root.c_str(), manifest,
          authenticated_semantic_change, 0, error, sizeof(error)),
          "authenticated semantic scene mutation is rejected");
    check(std::strstr(error, "route") != NULL &&
          std::strstr(error, "SHA-256") == NULL,
          "updated descriptor digest exposes semantic diagnostic");
    check(pack_equal(active, grail_prior),
          "authenticated semantic failure preserves complete active pack");
    write_text(grail_scene.c_str(), valid_scene);

    const fs::path artifact_paths[] = {
        grail_dir / "terrain.bin", grail_dir / "terrain.obj",
        grail_dir / "walkability.bin"};
    for (int artifact = 0; artifact < 3; ++artifact) {
        write_text(artifact_paths[artifact].c_str(), "changed");
        expect_pack_failure_preserves(
            active, grail_prior, root, manifest, catalog,
            "scene-owned binary digest rejected", "SHA-256");
        if (artifact == 0)
            write_bytes(artifact_paths[artifact].string(),
                        fixture_heightfield_bytes());
        else if (artifact == 1)
            write_text(artifact_paths[artifact].c_str(), "fixture mesh\n");
        else
            write_bytes(artifact_paths[artifact].string(),
                        fixture_walkability_bytes(false));
    }

    const char* grail_mutations[] = {"rename", "extra"};
    for (size_t i = 0;
         i < sizeof(grail_mutations) / sizeof(grail_mutations[0]); ++i) {
        write_text(grail_scene.c_str(),
            fixture_scene_json(grail, grail_mutations[i], grail_hashes));
        scene_catalog changed_catalog =
            one_scene_catalog(grail, file_sha(grail_scene.string()));
        expect_pack_failure_preserves(
            active, grail_prior, root, manifest, changed_catalog,
            "wrong route identity/count rejected", "route");
    }
    write_text(grail_scene.c_str(), valid_scene);

    scene_catalog bad_index = catalog;
    error[0] = '\0';
    check(!scene_pack_load(active, root.c_str(), manifest, bad_index, -1,
          error, sizeof(error)), "negative scene index rejected directly");
    check(pack_equal(active, grail_prior),
          "bad scene index preserves active pack");

    const std::string blocked = "blocked-course";
    const fs::path blocked_scene = root / "scenes" / blocked / "scene.json";
    const scene_asset_hashes blocked_hashes =
        prepare_scene_assets(root, blocked, true);
    write_text(blocked_scene.c_str(),
        fixture_scene_json(blocked, "valid", blocked_hashes));
    scene_catalog blocked_catalog =
        one_scene_catalog(blocked, file_sha(blocked_scene.string()));
    check(scene_pack_load(active, root.c_str(), manifest, blocked_catalog, 0,
          error, sizeof(error)), error);
    check(active.metadata.routes.size() == 2 &&
          active.metadata.routes[0].id == "wall-safe-stop" &&
          active.metadata.routes[1].id == "ramp-safe-stop",
          "ordered blocked fixture loads");
    const scene_pack blocked_prior = active;
    write_text(blocked_scene.c_str(),
        fixture_scene_json(blocked, "reverse", blocked_hashes));
    blocked_catalog = one_scene_catalog(
        blocked, file_sha(blocked_scene.string()));
    expect_pack_failure_preserves(
        active, blocked_prior, root, manifest, blocked_catalog,
        "reversed blocked route pair rejected", "route");

    std::string bad_digest(64, '0');
    write_text("/tmp/test_scene_hash", "changed");
    check(!scene_verify_sha("/tmp/test_scene_hash", bad_digest,
          error, sizeof(error)), "hash mismatch rejected");
    check(std::strstr(error, "/tmp/test_scene_hash") != NULL &&
          std::strstr(error, "SHA-256") != NULL, "hash diagnostic");
}

static void test_motion_database_contract()
{
    static const int parents[G1_BoneCount] = {
        -1,0,1,2,3,4,5,6,1,8,9,10,11,12,1,14,
        15,16,17,18,19,20,21,22,16,24,25,26,27,28,29};
    motion_pack_manifest manifest;
    manifest.database_frames = 2;
    manifest.sources.push_back(motion_source_record{"a", "flat", 0, 1});
    manifest.sources.push_back(motion_source_record{"b", "flat", 1, 2});
    database db;
    db.bone_positions.resize(2, G1_BoneCount);
    db.bone_parents.resize(G1_BoneCount);
    db.features.resize(2, 31);
    db.terrain_features.resize(2, 4);
    db.range_starts.resize(2);
    db.range_stops.resize(2);
    for (int i = 0; i < G1_BoneCount; ++i) db.bone_parents(i) = parents[i];
    db.range_starts(0) = 0;
    db.range_stops(0) = 1;
    db.range_starts(1) = 1;
    db.range_stops(1) = 2;
    char error[512] = {};
    check(motion_manifest_validate_database(
        manifest, db, error, sizeof(error)), error);
    db.terrain_features.cols = 3;
    check(!motion_manifest_validate_database(
        manifest, db, error, sizeof(error)),
        "terrain feature width mismatch rejected");
}

static void test_published_scene_contract(const char* root)
{
    char error[1024] = {};
    motion_pack_manifest manifest;
    check(motion_manifest_load_and_verify(
        manifest, root, error, sizeof(error)), error);
    check(manifest.output_fps == 25.0f &&
          manifest.feature_dimensions == 31 &&
          manifest.terrain_dimensions == 4 &&
          manifest.support_dimensions == 3, "motion dimensions");
    check(manifest.total_clips == 1770 && manifest.grail_clips == 1769 &&
          manifest.skipped_clips == 0 && !manifest.diagnostic_mode,
          "complete motion pack");
    check(manifest.surface.signature ==
          "f151c2b1c7f0498880f76c37f48a47c46c48bcf58c1285863fabc9a09fd7993a",
          "complete 43-key surface signature");
    check(manifest.sources.size() == 1770 &&
          manifest.sources.front().range_start == 0 &&
          manifest.sources.back().range_stop == manifest.database_frames,
          "source coverage");

    scene_catalog catalog;
    check(scene_catalog_load(
        catalog, root, manifest, error, sizeof(error)), error);
    check(catalog.default_scene_id == "grail-curb-default" &&
          catalog.ids.size() == 14 && catalog.scenes.size() == 14,
          "catalog size/default");
    for (int i = 0; i < 14; ++i) {
        check(catalog.ids[static_cast<size_t>(i)] == expected_scene_ids[i],
              "catalog order");
        const scene_descriptor& descriptor =
            catalog.scenes[static_cast<size_t>(i)];
        check(descriptor.id == expected_scene_ids[i] &&
              descriptor.path == std::string("scenes/") +
                  expected_scene_ids[i] + "/scene.json" &&
              scene_sha_is_valid(descriptor.sha256),
              "catalog descriptor order/path/hash");
    }

    scene_pack active;
    for (int i = 0; i < 14; ++i) {
        check(scene_pack_load(active, root, manifest, catalog, i,
              error, sizeof(error)), error);
        check(active.metadata.id == expected_scene_ids[i] &&
              active.terrain.version == 2 &&
              active.walkability.nx == active.terrain.nx &&
              active.walkability.nz == active.terrain.nz,
              "loaded scene identity/grid");
        check(active.metadata.routes.size() ==
              static_cast<size_t>(expected_route_counts[i]),
              "scene route count");
        for (int route = 0; route < expected_route_counts[i]; ++route) {
            const scene_route& loaded_route =
                active.metadata.routes[static_cast<size_t>(route)];
            check(loaded_route.id == expected_route_ids[i][route],
                  "scene route ID/order");
            check(loaded_route.expected_outcome ==
                      expected_route_outcomes[i][route] &&
                  loaded_route.walkability_class ==
                      expected_route_classes[i][route],
                  "scene route outcome/class tuple");
        }
    }

    const std::string prior_id = active.metadata.id;
    const float prior_origin = active.terrain.origin_x;
    check(!scene_pack_load(active, root, manifest, catalog, -1,
          error, sizeof(error)), "bad scene index rejected");
    check(active.metadata.id == prior_id &&
          active.terrain.origin_x == prior_origin,
          "failed load preserves active pack");

    std::string bad_digest(64, '0');
    write_text("/tmp/test_scene_hash", "changed");
    check(!scene_verify_sha("/tmp/test_scene_hash", bad_digest,
          error, sizeof(error)), "hash mismatch rejected");
    check(std::strstr(error, "/tmp/test_scene_hash") != NULL &&
          std::strstr(error, "SHA-256") != NULL, "hash diagnostic");
    check(scene_catalog_find(catalog, "../escape") == -1,
          "path-like ID is not selected");
    check(motion_source_for_frame(manifest, 0) == 0 &&
          motion_source_for_frame(manifest, manifest.database_frames - 1) ==
              static_cast<int>(manifest.sources.size()) - 1 &&
          motion_source_for_frame(manifest, manifest.database_frames) == -1,
          "source lookup boundaries");
}

int main(int argc, char** argv)
{
    check(argc == 1 || (argc == 3 &&
          std::strcmp(argv[1], "--real") == 0),
          "usage: test_scene_runtime [--real ROOT]");
    test_json_value_kinds_members_and_offsets();
    test_json_full_document_and_unique_keys();
    test_json_number_grammar_and_finite_range();
    test_json_utf8_escapes_and_surrogates();
    test_json_depth_limit_is_exactly_64();
    test_json_size_limit_is_exactly_16_mib();
    test_json_file_errors_are_transactional();
    test_sha256_known_vectors_and_file_errors();
    test_scene_numeric_precision_helpers();
    test_manifest_surface_contract_is_exact_and_transactional();
    test_catalog_contract_paths_and_tamper_order();
    test_scene_candidate_hostile_dimensions_are_transactional();
    test_scene_descriptor_hash_precedes_json_parse();
    test_scene_route_semantic_matrix_is_exact_and_transactional();
    test_scene_pack_tamper_chain_routes_and_transactionality();
    test_motion_database_contract();
    if (argc == 3) test_published_scene_contract(argv[2]);
    return 0;
}
