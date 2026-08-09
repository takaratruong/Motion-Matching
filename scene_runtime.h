#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "g1_skeleton.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#include "json_runtime.h"
#include "sha256.h"
#include "terrain_runtime.h"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

struct bounds2 { float min_x=0,min_z=0,max_x=0,max_z=0; };
struct point3d { double x=0,y=0,z=0; };
struct bounds3d { point3d minimum,maximum; };
struct motion_source_record {
    std::string name,terrain_id;
    int range_start=0,range_stop=0;
};
struct artifact_reference {
    std::string path,schema,sha256;
    int version=0,dimensions=0;
    std::vector<std::string> columns;
};
struct surface_contract {
    std::string signature,coordinate_signature;
    std::string heightfield_interpolation,heightfield_diagonal;
    float cell_size=0,exterior_height=0;
};
struct motion_pack_manifest {
    float output_fps=0;
    int feature_dimensions=0,terrain_dimensions=0,support_dimensions=0;
    int total_clips=0,grail_clips=0,skipped_clips=0,database_frames=0;
    bool diagnostic_mode=false;
    bool flat_lmm_bundle=false;
    float continuity_maximum_local_step=0.0f;
    std::vector<motion_source_record> sources;
    std::vector<float> feature_offset,feature_scale;
    surface_contract surface;
    artifact_reference database,terrain_features,terrain_support;
    artifact_reference matching_features;
    artifact_reference scene_index,validation_file;
};
struct scene_region { std::string id; bounds2 bounds; };
struct scene_descriptor {
    std::string id,path,sha256;
};
struct scene_route {
    std::string id,expected_outcome;
    int walkability_class=0;
    float landing_hold_seconds=0;
    std::vector<std::pair<float,float> > waypoints_xz;
};
struct scene_metadata {
    std::string id,label,provenance_kind;
    std::vector<std::string> provenance_source_ids;
    std::string coordinate_signature,surface_signature;
    artifact_reference heightfield,mesh,walkability;
    int heightfield_nx=0,heightfield_nz=0,walkability_nx=0,walkability_nz=0;
    float heightfield_origin_x=0,heightfield_origin_z=0;
    float heightfield_cell_size=0,heightfield_exterior_height=0;
    bounds3d mesh_bounds,heightfield_bounds;
    bounds2 playable_bounds,lookahead_bounds;
    vec3 spawn_position; float spawn_yaw=0;
    std::vector<scene_region> certified_regions,stress_regions,blocked_regions;
    std::vector<scene_route> routes;
};
struct scene_catalog {
    std::string default_scene_id,coordinate_signature,surface_signature;
    std::vector<std::string> ids;
    std::vector<scene_descriptor> scenes;
};
struct scene_pack {
    scene_metadata metadata; heightfield terrain; walkability_grid walkability;
    std::string scene_path,terrain_path,mesh_path,walkability_path;
};

static const char* const G1_RuntimeSceneIds[14] = {
    "grail-curb-default","grail-curb-low","grail-curb-medium",
    "grail-curb-high","stairs-shallow","stairs-standard",
    "stairs-unseen-variable","ramp-05-up-down","ramp-10-up-down",
    "ramp-15-stress","cross-slope-05","cross-slope-10",
    "mixed-multilevel","blocked-course"
};

static const char G1_RuntimeCoordinateSignature[] =
    "holden-y-up-right-handed-forward-plus-z";
static const char G1_RuntimeSurfaceSignature[] =
    "f151c2b1c7f0498880f76c37f48a47c46c48bcf58c1285863fabc9a09fd7993a";
static const char G1_RuntimeSurfaceSemanticsJson[] =
    R"json({"barycentric_tolerance":1e-10,"bbox_tolerance_m":1e-12,"cell_size_m":0.02,"coordinate_signature":"holden-y-up-right-handed-forward-plus-z","degenerate_projected_triangle_policy":"ignore","exterior_height_m":0.0,"heightfield_cell_domain":"positive-normal-binary32","heightfield_denormal_policy":"reject-nonzero-binary32-subnormals","heightfield_diagonal":"min-x-min-z_to_max-x-max-z","heightfield_diagonal_tie_policy":"tx-greater-or-equal-tz-uses-p00-p10-p11","heightfield_domain_policy":"inclusive-authoritative-node-rectangle","heightfield_evaluation_precision":"binary64-from-binary32-samples-and-promoted-node-weights","heightfield_exterior_normal":[0.0,1.0,0.0],"heightfield_grid_line_policy":"positive-index-cell-except-maximum-edge","heightfield_interpolation":"fixed-diagonal-triangles","heightfield_normal_evaluation":"selected-triangle-binary64-gradient-scale-safe-unit-normalization","heightfield_obj_coordinate_quantization":"binary32-round-of-promoted-origin-plus-index-times-cell","heightfield_obj_face_order":"p00-p11-p10_then_p00-p01-p11","heightfield_obj_float_format":".9g-final-newline","heightfield_obj_vertex_order":"z-major-x-minor","heightfield_raster_bounds_policy":"float32-minimum-rounded-down-and-maximum-ceil-covered","heightfield_runtime_height_output":"finite-binary64-interpolation-rounded-to-binary32","heightfield_runtime_node_distinguishability_policy":"normal-or-positive-zero-strictly-increasing-proven-by-endpoints-near-zero-candidates-max-binary32-spacing-and-aligned-equality","heightfield_runtime_node_domain":"normal-or-zero-binary32","heightfield_runtime_normal_output":"unit-normal-components-rounded-to-binary32","heightfield_runtime_output_ftz_policy":"binary32-subnormals-and-signed-zero-canonicalized-to-positive-zero","heightfield_runtime_parity_domain":"normal-or-zero-binary32-coordinates","heightfield_runtime_query_domain":"normal-or-zero-binary32-coordinates","heightfield_runtime_query_encoding":"normal-or-zero-binary32-canonicalized-positive-and-promoted-to-binary64","heightfield_scalar_domain":"normal-or-zero-binary32","heightfield_scalar_encoding":"ieee754-binary32-little-endian","heightfield_schema":"G1HF/v2","heightfield_source_node_encoding":"binary32-header-values-promoted-to-binary64-arithmetic","heightfield_version":2,"heightfield_zero_encoding":"canonical-positive-zero","overlap_height_policy":"maximum-y","polygon_triangulation":"fan-from-first-index","projected_area_epsilon_m2":1e-12,"projected_area_measure":"absolute-two-times-area","projected_boundary_policy":"closed","schema":"g1-terrain-surface/v1","source_query":"vertical-triangle-top","triangle_winding_policy":"orientation-independent"})json";

static inline bool scene_error(
    char* out, const int capacity, const char* format, ...)
{
    if (out != NULL && capacity > 0) {
        va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(out, static_cast<size_t>(capacity), format, arguments);
        va_end(arguments);
    }
    return false;
}

static inline bool scene_required(
    const json_value*& out, const json_value& object, const char* key,
    const char* label, char* error, const int capacity)
{
    out = json_member(object, key);
    if (out == NULL)
        return scene_error(error, capacity, "%s is missing required key '%s'",
                           label, key != NULL ? key : "<null>");
    return true;
}

static inline bool scene_exact_keys(
    const json_value& object,
    const std::initializer_list<const char*>& expected,
    const char* label, char* error, const int capacity)
{
    if (object.kind != json_object)
        return scene_error(error, capacity, "%s must be an object", label);
    std::vector<std::string> actual;
    std::vector<std::string> wanted;
    actual.reserve(object.object_value.size());
    wanted.reserve(expected.size());
    for (size_t i = 0; i < object.object_value.size(); ++i)
        actual.push_back(object.object_value[i].first);
    for (std::initializer_list<const char*>::const_iterator it=expected.begin();
         it != expected.end(); ++it) wanted.push_back(*it);
    std::sort(actual.begin(), actual.end());
    std::sort(wanted.begin(), wanted.end());
    if (actual != wanted)
        return scene_error(error, capacity, "%s has an incorrect key set", label);
    return true;
}

static inline bool json_equal(const json_value& first, const json_value& second)
{
    if (first.kind != second.kind) return false;
    if (first.kind == json_null) return true;
    if (first.kind == json_boolean)
        return first.boolean_value == second.boolean_value;
    if (first.kind == json_number) {
        uint64_t first_bits = 0, second_bits = 0;
        std::memcpy(&first_bits, &first.number_value, sizeof(first_bits));
        std::memcpy(&second_bits, &second.number_value, sizeof(second_bits));
        return first_bits == second_bits;
    }
    if (first.kind == json_string)
        return first.string_value == second.string_value;
    if (first.kind == json_array) {
        if (first.array_value.size() != second.array_value.size()) return false;
        for (size_t i = 0; i < first.array_value.size(); ++i)
            if (!json_equal(first.array_value[i], second.array_value[i]))
                return false;
        return true;
    }
    if (first.object_value.size() != second.object_value.size()) return false;
    for (size_t i = 0; i < first.object_value.size(); ++i) {
        const json_value* value =
            json_member(second, first.object_value[i].first.c_str());
        if (value == NULL || !json_equal(first.object_value[i].second, *value))
            return false;
    }
    return true;
}

static inline bool scene_number_double(
    double& out,const json_value& value,const char* label,char* error,int capacity)
{
    if(value.kind!=json_number||!terrain_double_is_finite(value.number_value))
        return scene_error(error,capacity,
            "%s must be a finite binary64 number",label);
    out=value.number_value;
    return true;
}

static inline bool scene_number_float(
    float& out, const json_value& value, const char* label,
    char* error, const int capacity)
{
    double parsed = 0.0;
    if (!scene_number_double(parsed, value, label, error, capacity)) return false;
    const float encoded = static_cast<float>(parsed);
    if (!terrain_float_is_finite(encoded))
        return scene_error(error, capacity,
            "%s must remain finite as binary32", label);
    out = encoded == 0.0f ? 0.0f : encoded;
    return true;
}

static inline bool scene_number_binary32(
    float& out,const json_value& value,const char* label,char* error,int capacity)
{
    double parsed=0;
    if(!scene_number_double(parsed,value,label,error,capacity))return false;
    const float encoded=static_cast<float>(parsed);
    uint32_t bits=0;
    std::memcpy(&bits,&encoded,sizeof(bits));
    if(!terrain_float_is_finite(encoded)||
       ((bits&0x7f800000u)==0u&&(bits&0x007fffffu)!=0u)||
       static_cast<double>(encoded)!=parsed)
        return scene_error(error,capacity,
            "%s must be an exact normal-or-zero binary32 value",label);
    out=encoded==0.0f?0.0f:encoded;
    return true;
}

static inline bool scene_number_int(
    int& out, const json_value& value, const char* label,
    char* error, const int capacity)
{
    double parsed = 0.0;
    if (!scene_number_double(parsed, value, label, error, capacity) ||
        parsed < static_cast<double>(INT_MIN) ||
        parsed > static_cast<double>(INT_MAX) ||
        std::floor(parsed) != parsed)
        return scene_error(error, capacity, "%s must be an integral int", label);
    out = static_cast<int>(parsed);
    return true;
}

static inline bool scene_double_array(
    double* out, const int count, const json_value& value, const char* label,
    char* error, const int capacity)
{
    if (out == NULL || count < 0 || value.kind != json_array ||
        value.array_value.size() != static_cast<size_t>(count))
        return scene_error(error, capacity,
            "%s must be an array of length %d", label, count);
    std::vector<double> candidate(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
        if (!scene_number_double(candidate[static_cast<size_t>(i)],
                value.array_value[static_cast<size_t>(i)],
                label, error, capacity)) return false;
    for (int i = 0; i < count; ++i)
        out[i] = candidate[static_cast<size_t>(i)];
    return true;
}

static inline bool scene_float_array(
    float* out, const int count, const json_value& value, const char* label,
    char* error, const int capacity)
{
    if (out == NULL || count < 0 || value.kind != json_array ||
        value.array_value.size() != static_cast<size_t>(count))
        return scene_error(error, capacity,
            "%s must be an array of length %d", label, count);
    std::vector<float> candidate(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
        if (!scene_number_float(candidate[static_cast<size_t>(i)],
                value.array_value[static_cast<size_t>(i)],
                label, error, capacity)) return false;
    for (int i = 0; i < count; ++i)
        out[i] = candidate[static_cast<size_t>(i)];
    return true;
}

static inline bool scene_binary32_array(
    float* out, const int count, const json_value& value, const char* label,
    char* error, const int capacity)
{
    if (out == NULL || count < 0 || value.kind != json_array ||
        value.array_value.size() != static_cast<size_t>(count))
        return scene_error(error, capacity,
            "%s must be an array of length %d", label, count);
    std::vector<float> candidate(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
        if (!scene_number_binary32(
                candidate[static_cast<size_t>(i)],
                value.array_value[static_cast<size_t>(i)],
                label, error, capacity)) return false;
    for (int i = 0; i < count; ++i)
        out[i] = candidate[static_cast<size_t>(i)];
    return true;
}

static inline bool scene_string(
    std::string& out, const json_value& value, const char* label,
    char* error, const int capacity)
{
    if (value.kind != json_string)
        return scene_error(error, capacity, "%s must be a string", label);
    out = value.string_value;
    return true;
}

static inline bool scene_boolean(
    bool& out, const json_value& value, const char* label,
    char* error, const int capacity)
{
    if (value.kind != json_boolean)
        return scene_error(error, capacity, "%s must be a boolean", label);
    out = value.boolean_value;
    return true;
}

static inline bool scene_sha_is_valid(const std::string& digest)
{
    if (digest.size() != 64) return false;
    for (size_t i = 0; i < digest.size(); ++i)
        if (!((digest[i] >= '0' && digest[i] <= '9') ||
              (digest[i] >= 'a' && digest[i] <= 'f'))) return false;
    return true;
}

static inline bool scene_verify_sha(
    const std::string& path, const std::string& expected,
    char* error, const int capacity)
{
    if (!scene_sha_is_valid(expected))
        return scene_error(error, capacity,
            "%s: invalid expected SHA-256 digest", path.c_str());
    std::string observed;
    if (!sha256_file_hex(observed, path.c_str(), error, capacity)) return false;
    if (observed != expected)
        return scene_error(error, capacity,
            "%s: SHA-256 mismatch (expected %s, got %s)",
            path.c_str(), expected.c_str(), observed.c_str());
    return true;
}

static inline bool scene_verify_sha(
    const char* path, const std::string& expected,
    char* error, const int capacity)
{
    if (path == NULL)
        return scene_error(error, capacity, "<null>: invalid SHA-256 path");
    return scene_verify_sha(std::string(path), expected, error, capacity);
}

static inline bool scene_id_is_safe(const std::string& id)
{if(id.empty()||id.size()>64)return false;for(char c:id)if(!((c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='-'))return false;return true;}

static inline bool scene_relative_is_safe(const std::string& path)
{if(path.empty()||path.size()>4095||path[0]=='/'||path.find('\\')!=std::string::npos)return false;size_t begin=0;while(begin<=path.size()){const size_t end=path.find('/',begin);const std::string part=path.substr(begin,end==std::string::npos?std::string::npos:end-begin);if(part.empty()||part=="."||part=="..")return false;if(end==std::string::npos)break;begin=end+1;}return true;}

static inline bool scene_join(std::string& out,const char* root,const std::string& relative,char* error,int capacity)
{if(root==NULL||root[0]=='\0'||!scene_relative_is_safe(relative))return scene_error(error,capacity,"unsafe artifact path '%s'",relative.c_str());const std::string joined=std::string(root)+"/"+relative;if(joined.size()>4095)return scene_error(error,capacity,"artifact path exceeds 4095 bytes: %s",relative.c_str());out=joined;return true;}

static inline bool scene_inside(const bounds2& bounds,float x,float z)
{return terrain_float_is_finite(x)&&terrain_float_is_finite(z)&&x>=bounds.min_x&&x<=bounds.max_x&&z>=bounds.min_z&&z<=bounds.max_z;}

static inline bool scene_binary32_lerp(float& out,float start,float stop,int step,int steps)
{if(step<0||steps<1||step>steps)return false;if(step==0){out=start;return true;}if(step==steps){out=stop;return true;}return terrain_f32_lerp(out,start,stop,step,steps);}

static inline bool scene_route_segment(float& dx,float& dz,float& length,const std::pair<float,float>& start,const std::pair<float,float>& stop)
{float dx2=0,dz2=0,sum=0;return terrain_f32_sub(dx,stop.first,start.first)&&terrain_f32_sub(dz,stop.second,start.second)&&terrain_f32_mul(dx2,dx,dx)&&terrain_f32_mul(dz2,dz,dz)&&terrain_f32_add(sum,dx2,dz2)&&terrain_f32_sqrt(length,sum)&&length>0.0f;}

static inline bool scene_route_sample_count(int& count,const std::pair<float,float>& start,const std::pair<float,float>& stop,float maximum_step)
{float dx=0,dz=0,length=0,ratio=0;if(!terrain_float_is_positive_normal(maximum_step)||!scene_route_segment(dx,dz,length,start,stop)||!terrain_f32_div(ratio,length,maximum_step))return false;const float rounded=ceilf(ratio);if(!terrain_float_is_finite(rounded)||static_cast<double>(rounded)>static_cast<double>(INT_MAX))return false;const int candidate=rounded<1.0f?1:static_cast<int>(rounded);count=candidate;return true;}

static inline bool scene_json_load_verified(
    json_value& out, const char* path, const std::string& expected,
    char* error, const int capacity)
{
    if (path == NULL || path[0] == '\0')
        return json_runtime_error(error, capacity,
            path != NULL ? path : "<null>", 0, "invalid path");
    if (!scene_sha_is_valid(expected))
        return scene_error(error, capacity,
            "%s: invalid expected SHA-256 digest", path);
    FILE* file = std::fopen(path, "rb");
    if (file == NULL)
        return json_runtime_error(error, capacity, path, 0, "cannot open");
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        return json_runtime_error(error, capacity, path, 0, "cannot size");
    }
    const long end = std::ftell(file);
    static const size_t maximum = 16u * 1024u * 1024u;
    if (end < 0 || static_cast<unsigned long>(end) > maximum ||
        std::fseek(file, 0, SEEK_SET) != 0) {
        std::fclose(file);
        return json_runtime_error(error, capacity, path, 0,
            end < 0 ? "cannot size" : "document exceeds 16 MiB");
    }
    std::string text(static_cast<size_t>(end), '\0');
    bool failed = !text.empty() &&
        std::fread(&text[0], 1, text.size(), file) != text.size();
    if (std::fclose(file) != 0) failed = true;
    if (failed)
        return json_runtime_error(error, capacity, path, 0, "cannot read");
    sha256_state state;
    sha256_update(state,
        reinterpret_cast<const uint8_t*>(text.data()), text.size());
    const std::string observed = sha256_finish(state);
    if (observed != expected)
        return scene_error(error, capacity,
            "%s: SHA-256 mismatch (expected %s, got %s)",
            path, expected.c_str(), observed.c_str());
    json_value loaded;
    json_parser parser(path, text);
    if (!parser.value(loaded))
        return json_runtime_error(error, capacity, path,
            parser.cursor, parser.reason.c_str());
    parser.whitespace();
    if (parser.cursor != text.size())
        return json_runtime_error(error, capacity, path,
            parser.cursor, "trailing data");
    out = std::move(loaded);
    return true;
}

static inline bool scene_member_string(
    std::string& out, const json_value& object, const char* key,
    const char* label, char* error, const int capacity)
{
    const json_value* value = NULL;
    return scene_required(value, object, key, label, error, capacity) &&
           scene_string(out, *value, key, error, capacity);
}

static inline bool scene_member_int(
    int& out, const json_value& object, const char* key,
    const char* label, char* error, const int capacity)
{
    const json_value* value = NULL;
    return scene_required(value, object, key, label, error, capacity) &&
           scene_number_int(out, *value, key, error, capacity);
}

static inline bool scene_member_float(
    float& out, const json_value& object, const char* key,
    const char* label, char* error, const int capacity)
{
    const json_value* value = NULL;
    return scene_required(value, object, key, label, error, capacity) &&
           scene_number_float(out, *value, key, error, capacity);
}

static inline bool scene_surface_parse(
    surface_contract& out, const json_value& value,
    char* error, const int capacity)
{
    if (!scene_exact_keys(value, {"semantics","signature"},
                          "surface", error, capacity)) return false;
    const json_value* semantics = NULL;
    const json_value* signature = NULL;
    if (!scene_required(semantics, value, "semantics", "surface",
                        error, capacity) ||
        !scene_required(signature, value, "signature", "surface",
                        error, capacity)) return false;
    if (semantics->kind != json_object ||
        semantics->object_value.size() != 43)
        return scene_error(error, capacity,
            "surface.semantics must contain exactly 43 keys");
    const std::string canonical(G1_RuntimeSurfaceSemanticsJson);
    json_value expected;
    json_parser parser("<surface-contract>", canonical);
    if (!parser.value(expected))
        return scene_error(error, capacity,
            "internal surface semantics contract is invalid");
    parser.whitespace();
    if (parser.cursor != canonical.size() || !json_equal(*semantics, expected))
        return scene_error(error, capacity,
            "surface.semantics does not match the exact 43-key contract");
    std::string declared;
    if (!scene_string(declared, *signature, "surface.signature",
                      error, capacity)) return false;
    sha256_state state;
    sha256_update(state,
        reinterpret_cast<const uint8_t*>(canonical.data()), canonical.size());
    const std::string computed = sha256_finish(state);
    if (computed != G1_RuntimeSurfaceSignature || declared != computed)
        return scene_error(error, capacity,
            "surface.signature does not match canonical surface SHA-256");
    const json_value* coordinate = json_member(*semantics,
                                                "coordinate_signature");
    const json_value* interpolation = json_member(*semantics,
                                                   "heightfield_interpolation");
    const json_value* diagonal = json_member(*semantics,
                                              "heightfield_diagonal");
    const json_value* cell = json_member(*semantics, "cell_size_m");
    const json_value* exterior = json_member(*semantics, "exterior_height_m");
    surface_contract candidate;
    candidate.signature = declared;
    if (coordinate == NULL || interpolation == NULL || diagonal == NULL ||
        cell == NULL || exterior == NULL ||
        !scene_string(candidate.coordinate_signature, *coordinate,
                      "surface coordinate signature", error, capacity) ||
        !scene_string(candidate.heightfield_interpolation, *interpolation,
                      "heightfield interpolation", error, capacity) ||
        !scene_string(candidate.heightfield_diagonal, *diagonal,
                      "heightfield diagonal", error, capacity) ||
        !scene_number_float(candidate.cell_size, *cell,
                            "surface cell size", error, capacity) ||
        !scene_number_float(candidate.exterior_height, *exterior,
                            "surface exterior height", error, capacity))
        return false;
    out = candidate;
    return true;
}

static inline bool scene_artifact_reference_parse(
    artifact_reference& out, const json_value& value, const char* label,
    const char* expected_path, const char* expected_schema,
    const int expected_version, const int expected_dimensions,
    const std::vector<std::string>& expected_columns,
    char* error, const int capacity)
{
    if (expected_version == 0) {
        if (!scene_exact_keys(value, {"path","schema","sha256"},
                              label, error, capacity)) return false;
    } else if (expected_columns.empty()) {
        if (!scene_exact_keys(value,
                {"path","schema","version","dimensions","sha256"},
                label, error, capacity)) return false;
    } else if (!scene_exact_keys(value,
            {"path","schema","version","dimensions","columns","sha256"},
            label, error, capacity)) return false;
    artifact_reference candidate;
    if (!scene_member_string(candidate.path, value, "path", label,
                             error, capacity) ||
        !scene_member_string(candidate.schema, value, "schema", label,
                             error, capacity) ||
        !scene_member_string(candidate.sha256, value, "sha256", label,
                             error, capacity)) return false;
    if (candidate.path != expected_path || !scene_relative_is_safe(candidate.path))
        return scene_error(error, capacity,
            "%s path must be exactly '%s'", label, expected_path);
    if (candidate.schema != expected_schema)
        return scene_error(error, capacity,
            "%s schema must be exactly '%s'", label, expected_schema);
    if (!scene_sha_is_valid(candidate.sha256))
        return scene_error(error, capacity,
            "%s sha256 must be lowercase 64-hex", label);
    if (expected_version != 0) {
        if (!scene_member_int(candidate.version, value, "version", label,
                              error, capacity) ||
            !scene_member_int(candidate.dimensions, value, "dimensions", label,
                              error, capacity)) return false;
        if (candidate.version != expected_version ||
            candidate.dimensions != expected_dimensions)
            return scene_error(error, capacity,
                "%s version/dimensions do not match the published contract",
                label);
    }
    if (!expected_columns.empty()) {
        const json_value* columns = NULL;
        if (!scene_required(columns, value, "columns", label,
                            error, capacity) || columns->kind != json_array ||
            columns->array_value.size() != expected_columns.size())
            return scene_error(error, capacity,
                "%s columns do not match the published contract", label);
        for (size_t i = 0; i < expected_columns.size(); ++i) {
            std::string column;
            if (!scene_string(column, columns->array_value[i],
                              "artifact column", error, capacity) ||
                column != expected_columns[i])
                return scene_error(error, capacity,
                    "%s columns do not match the published contract", label);
            candidate.columns.push_back(column);
        }
    }
    out = candidate;
    return true;
}

static inline bool scene_validation_validate(
    const json_value& value, const int count,
    char* error, const int capacity)
{
    if (!scene_exact_keys(value,
            {"schema","duration_error_s","fk_max_error_m",
             "quaternion_norm_max_error"},
            "validation", error, capacity)) return false;
    std::string schema;
    if (!scene_member_string(schema, value, "schema", "validation",
                             error, capacity) ||
        schema != "g1-terrain-validation/v1")
        return scene_error(error, capacity,
            "validation schema must be g1-terrain-validation/v1");
    const char* arrays[] = {
        "duration_error_s", "fk_max_error_m", "quaternion_norm_max_error"};
    for (int field = 0; field < 3; ++field) {
        const json_value* array = json_member(value, arrays[field]);
        if (array == NULL || array->kind != json_array ||
            array->array_value.size() != static_cast<size_t>(count))
            return scene_error(error, capacity,
                "validation.%s must contain %d entries", arrays[field], count);
        for (size_t i = 0; i < array->array_value.size(); ++i) {
            double number = 0.0;
            if (!scene_number_double(number, array->array_value[i],
                                     arrays[field], error, capacity) ||
                number < 0.0)
                return scene_error(error, capacity,
                    "validation.%s values must be finite and nonnegative",
                    arrays[field]);
        }
    }
    return true;
}

static inline bool flat_artifact_reference_parse(
    artifact_reference& output,
    const json_value& artifacts,
    const char* name,
    char* error,
    const int capacity)
{
    const json_value* descriptor = json_member(artifacts, name);
    int size_bytes = 0;
    if (descriptor == NULL || !scene_exact_keys(
            *descriptor, {"path","size_bytes","sha256"}, name,
            error, capacity) ||
        !scene_member_string(output.path, *descriptor, "path", name,
                             error, capacity) ||
        output.path != name ||
        !scene_member_string(output.sha256, *descriptor, "sha256", name,
                             error, capacity) ||
        !scene_member_int(size_bytes, *descriptor, "size_bytes", name,
                          error, capacity) || size_bytes <= 0)
    {
        return scene_error(
            error, capacity, "flat artifact '%s' is invalid", name);
    }
    return true;
}

static inline bool flat_motion_manifest_parse_and_verify(
    motion_pack_manifest& out,
    const json_value& document,
    const char* root,
    char* error,
    const int capacity)
{
    if (!scene_exact_keys(document,
            {"schema","output_fps","trajectory_horizons",
             "feature_dimensions","feature_names","feature_weights",
             "feature_offset","feature_scale","feature_signature",
             "terrain_features","database_frames","total_clips",
             "source_count","dimensions","skeleton","range_count","ranges",
             "continuity","time_filters","contact",
             "sources","validation","status","artifacts"},
            "flat motion manifest", error, capacity))
        return false;

    motion_pack_manifest candidate;
    candidate.flat_lmm_bundle = true;
    std::string schema, status;
    float fps = 0.0f;
    if (!scene_member_string(schema, document, "schema", "flat manifest",
                             error, capacity) ||
        schema != "g1-lmm-flat-data/v2" ||
        !scene_member_string(status, document, "status", "flat manifest",
                             error, capacity) || status != "accepted" ||
        !scene_member_float(fps, document, "output_fps", "flat manifest",
                            error, capacity) ||
        !g1_manifest_rate_compatible(fps))
        return scene_error(
            error, capacity,
            "flat bundle schema/status/rate is incompatible");
    candidate.output_fps = fps;

    const json_value* horizons = json_member(document, "trajectory_horizons");
    if (horizons == NULL || horizons->kind != json_array ||
        horizons->array_value.size() != 3)
        return scene_error(error, capacity,
                           "flat trajectory_horizons must be [20,40,60]");
    const int expected_horizons[3] = {20, 40, 60};
    for (int index = 0; index < 3; ++index) {
        int value = 0;
        if (!scene_number_int(
                value, horizons->array_value[static_cast<size_t>(index)],
                "flat horizon", error, capacity) ||
            value != expected_horizons[index])
            return scene_error(error, capacity,
                               "flat trajectory_horizons must be [20,40,60]");
    }
    if (!scene_member_int(
            candidate.feature_dimensions, document, "feature_dimensions",
            "flat manifest", error, capacity) ||
        candidate.feature_dimensions != 31 ||
        !scene_member_int(candidate.database_frames, document,
                          "database_frames", "flat manifest", error,
                          capacity) || candidate.database_frames < 61 ||
        !scene_member_int(candidate.total_clips, document, "total_clips",
                          "flat manifest", error, capacity) ||
        candidate.total_clips != 1)
        return scene_error(error, capacity,
                           "flat bundle dimensions or clip count changed");
    candidate.terrain_dimensions = 4;
    candidate.support_dimensions = 3;
    const json_value* dimensions = json_member(document, "dimensions");
    int bones = 0, features = 0, contacts = 0;
    if (dimensions == NULL || !scene_exact_keys(
            *dimensions, {"bones","features","contacts"},
            "flat dimensions", error, capacity) ||
        !scene_member_int(bones, *dimensions, "bones", "flat dimensions",
                          error, capacity) || bones != 31 ||
        !scene_member_int(features, *dimensions, "features",
                          "flat dimensions", error, capacity) ||
        features != 31 ||
        !scene_member_int(contacts, *dimensions, "contacts",
                          "flat dimensions", error, capacity) || contacts != 2)
        return scene_error(error, capacity,
                           "flat dimensions must be 31/31/2");

    const json_value* feature_names = json_member(document, "feature_names");
    const json_value* feature_weights = json_member(document, "feature_weights");
    const json_value* feature_offset = json_member(document, "feature_offset");
    const json_value* feature_scale = json_member(document, "feature_scale");
    std::string feature_signature;
    if (feature_names == NULL || feature_names->kind != json_array ||
        feature_names->array_value.size() != 31 ||
        feature_weights == NULL || feature_weights->kind != json_array ||
        feature_weights->array_value.size() != 6 ||
        feature_offset == NULL || feature_offset->kind != json_array ||
        feature_offset->array_value.size() != 31 ||
        feature_scale == NULL || feature_scale->kind != json_array ||
        feature_scale->array_value.size() != 31 ||
        !scene_member_string(feature_signature, document, "feature_signature",
                             "flat manifest", error, capacity) ||
        !scene_sha_is_valid(feature_signature))
        return scene_error(error, capacity,
                           "flat feature receipt dimensions changed");
    static const char* const expected_feature_names[31] = {
        "left_foot_position_x","left_foot_position_y","left_foot_position_z",
        "right_foot_position_x","right_foot_position_y","right_foot_position_z",
        "left_foot_velocity_x","left_foot_velocity_y","left_foot_velocity_z",
        "right_foot_velocity_x","right_foot_velocity_y","right_foot_velocity_z",
        "hip_velocity_x","hip_velocity_y","hip_velocity_z",
        "root_position_20_x","root_position_20_z",
        "root_position_40_x","root_position_40_z",
        "root_position_60_x","root_position_60_z",
        "root_facing_20_x","root_facing_20_z",
        "root_facing_40_x","root_facing_40_z",
        "root_facing_60_x","root_facing_60_z",
        "terrain_height_025","terrain_height_050",
        "terrain_height_075","terrain_height_100"};
    for (int index = 0; index < 31; ++index) {
        std::string name;
        if (!scene_string(
                name,
                feature_names->array_value[static_cast<size_t>(index)],
                "flat feature name", error, capacity) ||
            name != expected_feature_names[index])
            return scene_error(error, capacity,
                               "flat feature name %d changed", index);
    }
    const float expected_weights[6] = {
        0.75f, 1.0f, 1.0f, 1.0f, 1.5f, 1.0f};
    for (int index = 0; index < 6; ++index) {
        float weight = 0.0f;
        if (!scene_number_float(
                weight,
                feature_weights->array_value[static_cast<size_t>(index)],
                "flat feature weight", error, capacity) ||
            feature_float_bits(weight) !=
                feature_float_bits(expected_weights[index]))
            return scene_error(error, capacity,
                               "flat feature weight %d changed", index);
    }
    candidate.feature_offset.reserve(31);
    candidate.feature_scale.reserve(31);
    for (int index = 0; index < 31; ++index) {
        float offset = 0.0f, scale = 0.0f;
        if (!scene_number_float(
                offset,
                feature_offset->array_value[static_cast<size_t>(index)],
                "flat feature offset", error, capacity) ||
            !scene_number_float(
                scale,
                feature_scale->array_value[static_cast<size_t>(index)],
                "flat feature scale", error, capacity) ||
            !feature_float_is_finite(offset) ||
            !feature_float_is_positive_finite(scale))
            return scene_error(error, capacity,
                               "flat feature normalization is invalid");
        candidate.feature_offset.push_back(offset);
        candidate.feature_scale.push_back(scale);
    }
    const json_value* filters = json_member(document, "time_filters");
    int root_position_frames = 0, root_position_order = 0;
    int root_direction_frames = 0, root_direction_order = 0;
    int contact_median_frames = 0, forward_path_rows = 0;
    if (filters == NULL || !scene_exact_keys(
            *filters,
            {"root_position_frames","root_position_order",
             "root_direction_frames","root_direction_order",
             "contact_median_frames","forward_terrain_path_rows"},
            "flat time filters", error, capacity) ||
        !scene_member_int(root_position_frames, *filters,
                          "root_position_frames", "flat time filters",
                          error, capacity) || root_position_frames != 31 ||
        !scene_member_int(root_position_order, *filters,
                          "root_position_order", "flat time filters",
                          error, capacity) || root_position_order != 3 ||
        !scene_member_int(root_direction_frames, *filters,
                          "root_direction_frames", "flat time filters",
                          error, capacity) || root_direction_frames != 61 ||
        !scene_member_int(root_direction_order, *filters,
                          "root_direction_order", "flat time filters",
                          error, capacity) || root_direction_order != 3 ||
        !scene_member_int(contact_median_frames, *filters,
                          "contact_median_frames", "flat time filters",
                          error, capacity) || contact_median_frames != 7 ||
        !scene_member_int(forward_path_rows, *filters,
                          "forward_terrain_path_rows", "flat time filters",
                          error, capacity) || forward_path_rows != 121)
        return scene_error(error, capacity,
                           "flat time-filter receipt changed");
    const json_value* flat_contact = json_member(document, "contact");
    float contact_speed = 0.0f, contact_height = 0.0f;
    int contact_filter = 0;
    if (flat_contact == NULL || !scene_exact_keys(
            *flat_contact,
            {"speed_threshold","height_threshold","median_filter_frames"},
            "flat contact", error, capacity) ||
        !scene_member_float(contact_speed, *flat_contact, "speed_threshold",
                            "flat contact", error, capacity) ||
        contact_speed != 0.15f ||
        !scene_member_float(contact_height, *flat_contact, "height_threshold",
                            "flat contact", error, capacity) ||
        contact_height != 0.06f ||
        !scene_member_int(contact_filter, *flat_contact,
                          "median_filter_frames", "flat contact",
                          error, capacity) || contact_filter != 7)
        return scene_error(error, capacity,
                           "flat contact receipt changed");

    const json_value* validation = json_member(document, "validation");
    float fk_error = 0.0f;
    float duration_error = 0.0f;
    float quaternion_error = 0.0f;
    if (validation == NULL || !scene_exact_keys(
            *validation,
            {"fk_max_error_m","duration_error_s",
             "quaternion_norm_max_error"},
            "flat validation", error, capacity) ||
        !scene_member_float(fk_error, *validation, "fk_max_error_m",
                            "flat validation", error, capacity) ||
        !scene_member_float(duration_error, *validation, "duration_error_s",
                            "flat validation", error, capacity) ||
        !scene_member_float(quaternion_error, *validation,
                            "quaternion_norm_max_error", "flat validation",
                            error, capacity) ||
        fk_error < 0.0f || fk_error > 1.0e-5f ||
        duration_error < 0.0f || duration_error > 1.0f / 60.0f ||
        quaternion_error < 0.0f || quaternion_error > 1.0e-4f)
        return scene_error(error, capacity,
                           "flat validation receipt is incompatible");

    const json_value* terrain = json_member(document, "terrain_features");
    std::string semantics;
    double flat_value = 1.0;
    const json_value* indices = terrain == NULL
        ? NULL : json_member(*terrain, "indices");
    if (terrain == NULL || !scene_exact_keys(
            *terrain, {"indices","semantics","value_m"},
            "flat terrain_features", error, capacity) ||
        indices == NULL || indices->kind != json_array ||
        indices->array_value.size() != 4 ||
        !scene_member_string(semantics, *terrain, "semantics",
                             "flat terrain_features", error, capacity) ||
        semantics != "authenticated-flat-root-relative-height-deltas")
        return scene_error(error, capacity,
                           "flat bundle lacks the authenticated zero terrain claim");
    const json_value* flat_value_json = json_member(*terrain, "value_m");
    if (flat_value_json == NULL || !scene_number_double(
            flat_value, *flat_value_json, "flat terrain value", error,
            capacity) || flat_value != 0.0)
        return scene_error(error, capacity,
                           "flat bundle terrain claim must be exactly zero");
    for (int index = 0; index < 4; ++index) {
        int feature_index = -1;
        if (!scene_number_int(
                feature_index,
                indices->array_value[static_cast<size_t>(index)],
                "flat terrain index", error, capacity) ||
            feature_index != 27 + index)
            return scene_error(error, capacity,
                               "flat terrain feature indices changed");
    }

    const json_value* ranges = json_member(document, "ranges");
    const json_value* sources = json_member(document, "sources");
    int range_count = 0;
    int source_count = 0;
    if (ranges == NULL || ranges->kind != json_array ||
        ranges->array_value.empty() ||
        !scene_member_int(range_count, document, "range_count",
                          "flat manifest", error, capacity) ||
        range_count != static_cast<int>(ranges->array_value.size()) ||
        range_count != 13 ||
        !scene_member_int(source_count, document, "source_count",
                          "flat manifest", error, capacity) ||
        source_count != 1 ||
        sources == NULL || sources->kind != json_array ||
        sources->array_value.size() != static_cast<size_t>(source_count))
        return scene_error(error, capacity,
                           "flat bundle range/source receipts are incomplete");
    int expected_start = 0;
    std::vector<int> range_source_first;
    std::vector<int> range_source_last;
    range_source_first.reserve(ranges->array_value.size());
    range_source_last.reserve(ranges->array_value.size());
    sha256_state range_digest_state;
    for (size_t index = 0; index < ranges->array_value.size(); ++index) {
        const json_value& range = ranges->array_value[index];
        motion_source_record source;
        std::string motion_class, terrain_class;
        int source_first_frame = -1, source_last_frame = -1;
        if (!scene_exact_keys(range,
                {"start","stop","source","source_first_frame",
                 "source_last_frame","motion_class","terrain_class"},
                "flat range", error, capacity) ||
            !scene_member_int(source.range_start, range, "start", "flat range",
                              error, capacity) ||
            source.range_start != expected_start ||
            !scene_member_int(source.range_stop, range, "stop", "flat range",
                              error, capacity) ||
            source.range_stop - source.range_start < 61 ||
            !scene_member_int(source_first_frame, range, "source_first_frame",
                              "flat range", error, capacity) ||
            !scene_member_int(source_last_frame, range, "source_last_frame",
                              "flat range", error, capacity) ||
            source_first_frame < 0 || source_last_frame < source_first_frame ||
            !scene_member_string(motion_class, range, "motion_class",
                                 "flat range", error, capacity) ||
            motion_class != "flat-walk" ||
            !scene_member_string(terrain_class, range, "terrain_class",
                                 "flat range", error, capacity) ||
            terrain_class != "flat" ||
            !scene_member_string(source.name, range, "source", "flat range",
                                 error, capacity) || source.name.empty())
            return scene_error(error, capacity,
                               "flat range is not a contiguous flat walk");
        source.terrain_id = "flat";
        expected_start = source.range_stop;
        candidate.sources.push_back(source);
        range_source_first.push_back(source_first_frame);
        range_source_last.push_back(source_last_frame);
        const int digest_values[4] = {
            source.range_start, source.range_stop,
            source_first_frame, source_last_frame};
        unsigned char digest_bytes[16] = {};
        for (int value_index = 0; value_index < 4; ++value_index) {
            const uint32_t value = static_cast<uint32_t>(
                digest_values[value_index]);
            digest_bytes[value_index * 4 + 0] =
                static_cast<unsigned char>(value);
            digest_bytes[value_index * 4 + 1] =
                static_cast<unsigned char>(value >> 8);
            digest_bytes[value_index * 4 + 2] =
                static_cast<unsigned char>(value >> 16);
            digest_bytes[value_index * 4 + 3] =
                static_cast<unsigned char>(value >> 24);
        }
        sha256_update(range_digest_state, digest_bytes, sizeof(digest_bytes));
    }
    if (expected_start != candidate.database_frames)
        return scene_error(error, capacity,
                           "flat ranges do not cover the database exactly");
    const std::string computed_range_digest =
        sha256_finish(range_digest_state);
    const json_value* continuity = json_member(document, "continuity");
    std::string continuity_schema, range_digest, source_map_digest;
    float continuity_threshold = 0.0f;
    float maximum_native_step = 0.0f;
    float maximum_local_step = 0.0f;
    int minimum_range_frames = 0;
    int source_rejected_edges = 0;
    int local_rejected_edges = 0;
    int union_rejected_edges = 0;
    int dropped_fragment_count = 0;
    int dropped_frame_count = 0;
    int published_range_count = 0;
    int published_frame_count = 0;
    if (continuity == NULL || !scene_exact_keys(
            *continuity,
            {"schema","threshold_rad_per_frame","minimum_range_frames",
             "source_native_rejected_edge_count",
             "database_local_rejected_edge_count",
             "union_rejected_edge_count","dropped_fragment_count",
             "dropped_frame_count","published_range_count",
             "published_frame_count","maximum_admitted_native_step_rad",
             "maximum_admitted_local_rotation_step_rad",
             "range_digest_sha256","source_map_digest_sha256"},
            "flat continuity", error, capacity) ||
        !scene_member_string(continuity_schema, *continuity, "schema",
                             "flat continuity", error, capacity) ||
        continuity_schema != "g1-lmm-continuity/v1" ||
        !scene_member_float(continuity_threshold, *continuity,
                            "threshold_rad_per_frame", "flat continuity",
                            error, capacity) || continuity_threshold != 0.25f ||
        !scene_member_int(minimum_range_frames, *continuity,
                          "minimum_range_frames", "flat continuity",
                          error, capacity) || minimum_range_frames != 61 ||
        !scene_member_int(source_rejected_edges, *continuity,
                          "source_native_rejected_edge_count",
                          "flat continuity", error, capacity) ||
        source_rejected_edges != 31 ||
        !scene_member_int(local_rejected_edges, *continuity,
                          "database_local_rejected_edge_count",
                          "flat continuity", error, capacity) ||
        local_rejected_edges != 32 ||
        !scene_member_int(union_rejected_edges, *continuity,
                          "union_rejected_edge_count", "flat continuity",
                          error, capacity) || union_rejected_edges != 32 ||
        !scene_member_int(dropped_fragment_count, *continuity,
                          "dropped_fragment_count", "flat continuity",
                          error, capacity) || dropped_fragment_count != 20 ||
        !scene_member_int(dropped_frame_count, *continuity,
                          "dropped_frame_count", "flat continuity",
                          error, capacity) || dropped_frame_count != 233 ||
        !scene_member_int(published_range_count, *continuity,
                          "published_range_count", "flat continuity",
                          error, capacity) || published_range_count != 13 ||
        !scene_member_int(published_frame_count, *continuity,
                          "published_frame_count", "flat continuity",
                          error, capacity) ||
        published_frame_count != candidate.database_frames ||
        published_frame_count != 3853 ||
        !scene_member_float(maximum_native_step, *continuity,
                            "maximum_admitted_native_step_rad",
                            "flat continuity", error, capacity) ||
        !scene_member_float(maximum_local_step, *continuity,
                            "maximum_admitted_local_rotation_step_rad",
                            "flat continuity", error, capacity) ||
        maximum_native_step < 0.0f || maximum_native_step > 0.25f ||
        maximum_local_step < 0.0f || maximum_local_step > 0.25f ||
        !scene_member_string(range_digest, *continuity,
                             "range_digest_sha256", "flat continuity",
                             error, capacity) ||
        range_digest != computed_range_digest ||
        !scene_member_string(source_map_digest, *continuity,
                             "source_map_digest_sha256", "flat continuity",
                             error, capacity))
        return scene_error(error, capacity,
                           "flat continuity receipt is incompatible");
    candidate.continuity_maximum_local_step = maximum_local_step;
    const json_value& source_receipt = sources->array_value[0];
    std::string source_name, source_terrain;
    std::string source_path, source_sha, receipt_path, receipt_sha;
    std::string receipt_schema, receipt_status;
    int source_frames = 0, output_frames = 0;
    float source_fps = 0.0f;
    const json_value* left_indices = json_member(
        source_receipt, "left_source_index");
    const json_value* right_indices = json_member(
        source_receipt, "right_source_index");
    const json_value* source_alpha = json_member(
        source_receipt, "source_alpha");
    if (!scene_exact_keys(source_receipt,
            {"name","terrain_id","path","sha256","receipt_path",
             "receipt_sha256","receipt_schema","receipt_status",
             "source_fps","source_frames","output_frames",
             "left_source_index","right_source_index","source_alpha"},
            "flat source", error, capacity) ||
        !scene_member_string(source_terrain, source_receipt, "terrain_id",
                             "flat source", error, capacity) ||
        source_terrain != "flat" ||
        !scene_member_string(source_name, source_receipt, "name",
                             "flat source", error, capacity) ||
        source_name.empty() ||
        !scene_member_string(source_path, source_receipt, "path",
                             "flat source", error, capacity) ||
        source_path.empty() ||
        !scene_member_string(source_sha, source_receipt, "sha256",
                             "flat source", error, capacity) ||
        !scene_sha_is_valid(source_sha) ||
        !scene_member_string(receipt_path, source_receipt, "receipt_path",
                             "flat source", error, capacity) ||
        receipt_path.empty() ||
        !scene_member_string(receipt_sha, source_receipt, "receipt_sha256",
                             "flat source", error, capacity) ||
        !scene_sha_is_valid(receipt_sha) ||
        !scene_member_string(receipt_schema, source_receipt, "receipt_schema",
                             "flat source", error, capacity) ||
        receipt_schema != "native-g1-pfnn-sample-retarget/v1" ||
        !scene_member_string(receipt_status, source_receipt, "receipt_status",
                             "flat source", error, capacity) ||
        receipt_status != "accepted" ||
        !scene_member_float(source_fps, source_receipt, "source_fps",
                            "flat source", error, capacity) ||
        source_fps != 120.0f ||
        !scene_member_int(source_frames, source_receipt, "source_frames",
                          "flat source", error, capacity) ||
        !scene_member_int(output_frames, source_receipt, "output_frames",
                          "flat source", error, capacity) ||
        output_frames != candidate.database_frames || source_frames <= 0 ||
        left_indices == NULL || right_indices == NULL ||
        source_alpha == NULL || left_indices->kind != json_array ||
        right_indices->kind != json_array || source_alpha->kind != json_array ||
        left_indices->array_value.size() !=
            static_cast<size_t>(candidate.database_frames) ||
        right_indices->array_value.size() != left_indices->array_value.size() ||
        source_alpha->array_value.size() != left_indices->array_value.size())
        return scene_error(error, capacity,
                           "flat source reconstruction map is incompatible");
    for (size_t index = 0; index < candidate.sources.size(); ++index) {
        if (candidate.sources[index].name != source_name)
            return scene_error(error, capacity,
                               "flat range source identity changed");
    }
    size_t active_range = 0;
    int previous_source_index = -1;
    std::vector<int> source_map_left;
    std::vector<int> source_map_right;
    std::vector<float> source_map_alpha;
    source_map_left.reserve(static_cast<size_t>(candidate.database_frames));
    source_map_right.reserve(static_cast<size_t>(candidate.database_frames));
    source_map_alpha.reserve(static_cast<size_t>(candidate.database_frames));
    for (int frame = 0; frame < candidate.database_frames; ++frame) {
        while (active_range + 1 < candidate.sources.size() &&
               frame >= candidate.sources[active_range].range_stop)
            ++active_range;
        int left = -1, right = -1;
        float alpha = -1.0f;
        if (!scene_number_int(
                left, left_indices->array_value[static_cast<size_t>(frame)],
                "flat left source index", error, capacity) ||
            !scene_number_int(
                right, right_indices->array_value[static_cast<size_t>(frame)],
                "flat right source index", error, capacity) ||
            !scene_number_float(
                alpha, source_alpha->array_value[static_cast<size_t>(frame)],
                "flat source alpha", error, capacity) ||
            left < 0 || left >= source_frames || right != left ||
            feature_float_bits(alpha) != 0 ||
            (frame > candidate.sources[active_range].range_start &&
             left != previous_source_index + 2) ||
            (frame == candidate.sources[active_range].range_start &&
             (previous_source_index >= left ||
              left != range_source_first[active_range])) ||
            (frame == candidate.sources[active_range].range_stop - 1 &&
             left != range_source_last[active_range]))
            return scene_error(error, capacity,
                               "flat source map changes inside a retained range");
        source_map_left.push_back(left);
        source_map_right.push_back(right);
        source_map_alpha.push_back(alpha);
        previous_source_index = left;
    }
    sha256_state source_map_digest_state;
    for (int pass = 0; pass < 3; ++pass) {
        for (int frame = 0; frame < candidate.database_frames; ++frame) {
            const uint32_t value = pass == 0
                ? static_cast<uint32_t>(source_map_left[
                    static_cast<size_t>(frame)])
                : pass == 1
                    ? static_cast<uint32_t>(source_map_right[
                        static_cast<size_t>(frame)])
                    : feature_float_bits(source_map_alpha[
                        static_cast<size_t>(frame)]);
            const unsigned char bytes[4] = {
                static_cast<unsigned char>(value),
                static_cast<unsigned char>(value >> 8),
                static_cast<unsigned char>(value >> 16),
                static_cast<unsigned char>(value >> 24),
            };
            sha256_update(source_map_digest_state, bytes, sizeof(bytes));
        }
    }
    if (sha256_finish(source_map_digest_state) != source_map_digest)
        return scene_error(error, capacity,
                           "flat source map digest differs");

    const json_value* skeleton = json_member(document, "skeleton");
    std::string skeleton_signature, basis;
    const json_value* names = skeleton == NULL ? NULL : json_member(*skeleton, "names");
    const json_value* parents = skeleton == NULL ? NULL : json_member(*skeleton, "parents");
    if (skeleton == NULL || !scene_exact_keys(
            *skeleton, {"names","parents","basis","signature"},
            "flat skeleton", error, capacity) || names == NULL ||
        parents == NULL || names->kind != json_array ||
        parents->kind != json_array ||
        names->array_value.size() != G1_BoneCount ||
        parents->array_value.size() != G1_BoneCount ||
        !scene_member_string(basis, *skeleton, "basis", "flat skeleton",
                             error, capacity) ||
        basis != "holden-y-up-right-handed-forward-plus-z" ||
        !scene_member_string(skeleton_signature, *skeleton, "signature",
                             "flat skeleton", error, capacity) ||
        skeleton_signature != G1_SkeletonSignature)
        return scene_error(error, capacity,
                           "flat skeleton receipt is incompatible with G1");
    static const char* const expected_flat_bone_names[G1_BoneCount] = {
        "Simulation","Hips","LeftHipPitch","LeftHipRoll","LeftHipYaw",
        "LeftKnee","LeftAnkle","LeftToe","RightHipPitch","RightHipRoll",
        "RightHipYaw","RightKnee","RightAnkle","RightToe","Spine",
        "Spine1","Spine2","LeftShoulderPitch","LeftShoulderRoll",
        "LeftShoulderYaw","LeftElbow","LeftWristRoll","LeftWristPitch",
        "LeftWrist","RightShoulderPitch","RightShoulderRoll",
        "RightShoulderYaw","RightElbow","RightWristRoll","RightWristPitch",
        "RightWrist"};
    static const int expected_flat_parents[G1_BoneCount] = {
        -1,0,1,2,3,4,5,6,1,8,9,10,11,12,1,14,
        15,16,17,18,19,20,21,22,16,24,25,26,27,28,29};
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        std::string name;
        int parent = -2;
        if (!scene_string(
                name, names->array_value[static_cast<size_t>(bone)],
                "flat bone name", error, capacity) ||
            !scene_number_int(
                parent, parents->array_value[static_cast<size_t>(bone)],
                "flat bone parent", error, capacity) ||
            name != expected_flat_bone_names[bone] ||
            parent != expected_flat_parents[bone])
            return scene_error(error, capacity,
                               "flat skeleton bone %d changed", bone);
    }

    const json_value* artifacts = json_member(document, "artifacts");
    if (artifacts == NULL || !scene_exact_keys(
            *artifacts, {"database.bin","features.bin"}, "flat artifacts",
            error, capacity) ||
        !flat_artifact_reference_parse(
            candidate.database, *artifacts, "database.bin", error,
            capacity) ||
        !flat_artifact_reference_parse(
            candidate.matching_features, *artifacts, "features.bin", error,
            capacity))
        return false;
    std::string database_path, features_path;
    if (!scene_join(database_path, root, candidate.database.path,
                    error, capacity) ||
        !scene_join(features_path, root, candidate.matching_features.path,
                    error, capacity) ||
        !scene_verify_sha(database_path, candidate.database.sha256,
                          error, capacity) ||
        !scene_verify_sha(features_path, candidate.matching_features.sha256,
                          error, capacity))
        return false;
    out = std::move(candidate);
    return true;
}

static inline bool motion_manifest_load_and_verify(
    motion_pack_manifest& out, const char* root,
    char* error, const int capacity)
{
    std::string manifest_path;
    if (!scene_join(manifest_path, root, "manifest.json", error, capacity))
        return false;
    json_value document;
    if (!json_document_load(document, manifest_path.c_str(), error, capacity))
        return false;
    std::string detected_schema;
    if (!scene_member_string(
            detected_schema, document, "schema", "motion manifest",
            error, capacity))
        return false;
    if (detected_schema == "g1-lmm-flat-data/v2") {
        return flat_motion_manifest_parse_and_verify(
            out, document, root, error, capacity);
    }
    if (!scene_exact_keys(document,
            {"schema","output_fps","feature_dimensions","terrain_dimensions",
             "support_dimensions","terrain_feature_distances_m","total_clips",
             "grail_clips","skipped_clips","database_frames",
             "diagnostic_mode","sources","skeleton","contact","surface",
             "database","sidecars","scene_index","validation_file",
             "validation"},
            "motion manifest", error, capacity)) return false;

    motion_pack_manifest candidate;
    std::string schema;
    if (!scene_member_string(schema, document, "schema", "motion manifest",
                             error, capacity) ||
        schema != "g1-terrain-artifacts/v2")
        return scene_error(error, capacity,
            "motion manifest schema must be g1-terrain-artifacts/v2");
    const json_value* value = NULL;
    double fps = 0.0;
    if (!scene_required(value, document, "output_fps", "motion manifest",
                        error, capacity) ||
        !scene_number_double(fps, *value, "output_fps", error, capacity) ||
        fps != 60.0 ||
        !g1_manifest_rate_compatible(static_cast<float>(fps)))
        return scene_error(error, capacity, "output_fps must be exactly 60");
    candidate.output_fps = 60.0f;
    if (!scene_member_int(candidate.feature_dimensions, document,
                          "feature_dimensions", "motion manifest",
                          error, capacity) ||
        !scene_member_int(candidate.terrain_dimensions, document,
                          "terrain_dimensions", "motion manifest",
                          error, capacity) ||
        !scene_member_int(candidate.support_dimensions, document,
                          "support_dimensions", "motion manifest",
                          error, capacity) ||
        candidate.feature_dimensions != 31 ||
        candidate.terrain_dimensions != 4 ||
        candidate.support_dimensions != 3)
        return scene_error(error, capacity,
            "motion feature/terrain/support dimensions must be 31/4/3");
    const json_value* distances = json_member(document,
                                               "terrain_feature_distances_m");
    static const double expected_distances[4] = {0.25,0.5,0.75,1.0};
    if (distances == NULL || distances->kind != json_array ||
        distances->array_value.size() != 4)
        return scene_error(error, capacity,
            "terrain_feature_distances_m must contain four values");
    for (int i = 0; i < 4; ++i) {
        double distance = 0.0;
        if (!scene_number_double(distance,
                distances->array_value[static_cast<size_t>(i)],
                "terrain feature distance", error, capacity) ||
            distance != expected_distances[i])
            return scene_error(error, capacity,
                "terrain_feature_distances_m changed");
    }
    if (!scene_member_int(candidate.total_clips, document, "total_clips",
                          "motion manifest", error, capacity) ||
        !scene_member_int(candidate.grail_clips, document, "grail_clips",
                          "motion manifest", error, capacity) ||
        !scene_member_int(candidate.skipped_clips, document, "skipped_clips",
                          "motion manifest", error, capacity) ||
        !scene_member_int(candidate.database_frames, document,
                          "database_frames", "motion manifest",
                          error, capacity) ||
        candidate.total_clips != 1770 || candidate.grail_clips != 1769 ||
        candidate.skipped_clips != 0 || candidate.database_frames <= 0)
        return scene_error(error, capacity,
            "motion manifest is not the complete 1770-clip pack");
    value = json_member(document, "diagnostic_mode");
    if (value == NULL || !scene_boolean(candidate.diagnostic_mode, *value,
            "diagnostic_mode", error, capacity) || candidate.diagnostic_mode)
        return scene_error(error, capacity,
            "diagnostic_mode must be exactly false");

    const json_value* skeleton = json_member(document, "skeleton");
    if (skeleton == NULL || !scene_exact_keys(*skeleton,
            {"names","parents","signature"}, "skeleton", error, capacity))
        return false;
    static const char* const bone_names[G1_BoneCount] = {
        "Simulation","Hips","LeftHipPitch","LeftHipRoll","LeftHipYaw",
        "LeftKnee","LeftAnkle","LeftToe","RightHipPitch","RightHipRoll",
        "RightHipYaw","RightKnee","RightAnkle","RightToe","Spine",
        "Spine1","Spine2","LeftShoulderPitch","LeftShoulderRoll",
        "LeftShoulderYaw","LeftElbow","LeftWristRoll","LeftWristPitch",
        "LeftWrist","RightShoulderPitch","RightShoulderRoll",
        "RightShoulderYaw","RightElbow","RightWristRoll","RightWristPitch",
        "RightWrist"};
    static const int bone_parents[G1_BoneCount] = {
        -1,0,1,2,3,4,5,6,1,8,9,10,11,12,1,14,
        15,16,17,18,19,20,21,22,16,24,25,26,27,28,29};
    const json_value* names = json_member(*skeleton, "names");
    const json_value* parents = json_member(*skeleton, "parents");
    std::string skeleton_signature;
    if (names == NULL || parents == NULL || names->kind != json_array ||
        parents->kind != json_array ||
        names->array_value.size() != G1_BoneCount ||
        parents->array_value.size() != G1_BoneCount ||
        !scene_member_string(skeleton_signature, *skeleton, "signature",
                             "skeleton", error, capacity) ||
        skeleton_signature != G1_SkeletonSignature)
        return scene_error(error, capacity,
            "skeleton shape/signature does not match G1");
    for (int i = 0; i < G1_BoneCount; ++i) {
        std::string name;
        int parent = 0;
        if (!scene_string(name, names->array_value[static_cast<size_t>(i)],
                          "skeleton name", error, capacity) ||
            !scene_number_int(parent,
                parents->array_value[static_cast<size_t>(i)],
                "skeleton parent", error, capacity) ||
            name != bone_names[i] || parent != bone_parents[i])
            return scene_error(error, capacity,
                "skeleton bone %d does not match G1", i);
    }

    const json_value* contact = json_member(document, "contact");
    if (contact == NULL || !scene_exact_keys(*contact,
            {"speed_threshold","height_threshold","median_filter_frames"},
            "contact", error, capacity)) return false;
    float speed = 0.0f, height = 0.0f;
    int median = 0;
    if (!scene_member_float(speed, *contact, "speed_threshold", "contact",
                            error, capacity) ||
        !scene_member_float(height, *contact, "height_threshold", "contact",
                            error, capacity) ||
        !scene_member_int(median, *contact, "median_filter_frames", "contact",
                          error, capacity) ||
        speed < 0.0f || height < 0.0f || median <= 0 || (median & 1) == 0)
        return scene_error(error, capacity,
            "contact thresholds/filter are invalid");

    const json_value* surface = json_member(document, "surface");
    if (surface == NULL || !scene_surface_parse(candidate.surface, *surface,
                                                error, capacity)) return false;

    const json_value* sources = json_member(document, "sources");
    if (sources == NULL || sources->kind != json_array ||
        sources->array_value.size() !=
            static_cast<size_t>(candidate.total_clips))
        return scene_error(error, capacity,
            "sources must contain exactly 1770 records");
    int expected_start = 0;
    std::vector<std::string> source_names;
    source_names.reserve(sources->array_value.size());
    candidate.sources.reserve(sources->array_value.size());
    for (size_t i = 0; i < sources->array_value.size(); ++i) {
        const json_value& source = sources->array_value[i];
        if (!scene_exact_keys(source,
                {"name","output_frames","range_start","range_stop",
                 "source_fps","source_frame_map","source_frames","terrain_id"},
                "source", error, capacity)) return false;
        motion_source_record record;
        int output_frames = 0, source_frames = 0;
        float source_fps = 0.0f;
        if (!scene_member_string(record.name, source, "name", "source",
                                 error, capacity) || record.name.empty() ||
            !scene_member_string(record.terrain_id, source, "terrain_id",
                                 "source", error, capacity) ||
            record.terrain_id.empty() ||
            !scene_member_int(output_frames, source, "output_frames", "source",
                              error, capacity) ||
            !scene_member_int(record.range_start, source, "range_start", "source",
                              error, capacity) ||
            !scene_member_int(record.range_stop, source, "range_stop", "source",
                              error, capacity) ||
            !scene_member_int(source_frames, source, "source_frames", "source",
                              error, capacity) ||
            !scene_member_float(source_fps, source, "source_fps", "source",
                                error, capacity) ||
            output_frames <= 0 || source_frames <= 0 || source_fps <= 0.0f ||
            record.range_start != expected_start ||
            static_cast<int64_t>(record.range_stop) -
                static_cast<int64_t>(record.range_start) !=
                    static_cast<int64_t>(output_frames))
            return scene_error(error, capacity,
                "source %zu has invalid frame ownership", i);
        for (size_t prior = 0; prior < source_names.size(); ++prior)
            if (source_names[prior] == record.name)
                return scene_error(error, capacity,
                    "source name '%s' is duplicated", record.name.c_str());
        source_names.push_back(record.name);
        const json_value* frame_map = json_member(source, "source_frame_map");
        if (frame_map == NULL || frame_map->kind != json_array ||
            frame_map->array_value.size() != static_cast<size_t>(output_frames))
            return scene_error(error, capacity,
                "source %zu frame map length does not equal output_frames", i);
        for (size_t frame = 0; frame < frame_map->array_value.size(); ++frame) {
            int mapped = 0;
            if (!scene_number_int(mapped, frame_map->array_value[frame],
                                  "source frame map", error, capacity) ||
                mapped < 0 || mapped >= source_frames)
                return scene_error(error, capacity,
                    "source %zu frame map is out of range", i);
        }
        expected_start = record.range_stop;
        candidate.sources.push_back(record);
    }
    if (expected_start != candidate.database_frames)
        return scene_error(error, capacity,
            "source ranges stop at %d instead of database frame %d",
            expected_start, candidate.database_frames);

    const json_value* database = json_member(document, "database");
    const json_value* sidecars = json_member(document, "sidecars");
    const json_value* index = json_member(document, "scene_index");
    const json_value* validation_ref = json_member(document, "validation_file");
    const std::vector<std::string> no_columns;
    const std::vector<std::string> support_columns = {
        "source_root_height_m", "source_left_toe_height_m",
        "source_right_toe_height_m"};
    if (database == NULL || sidecars == NULL || index == NULL ||
        validation_ref == NULL ||
        !scene_artifact_reference_parse(candidate.database, *database,
            "database", "database.bin", "holden-database/v1", 0, 0,
            no_columns, error, capacity) ||
        !scene_exact_keys(*sidecars,
            {"terrain_features","terrain_support"}, "sidecars",
            error, capacity)) return false;
    const json_value* features = json_member(*sidecars, "terrain_features");
    const json_value* support_ref = json_member(*sidecars, "terrain_support");
    if (features == NULL || support_ref == NULL ||
        !scene_artifact_reference_parse(candidate.terrain_features, *features,
            "terrain_features", "terrain_features.bin", "G1TF/v1", 1, 4,
            no_columns, error, capacity) ||
        !scene_artifact_reference_parse(candidate.terrain_support, *support_ref,
            "terrain_support", "terrain_support.bin", "G1SP/v1", 1, 3,
            support_columns, error, capacity) ||
        !scene_artifact_reference_parse(candidate.scene_index, *index,
            "scene_index", "scenes/index.json",
            "g1-terrain-scene-index/v1", 0, 0, no_columns,
            error, capacity) ||
        !scene_artifact_reference_parse(candidate.validation_file,
            *validation_ref, "validation_file", "validation.json",
            "g1-terrain-validation/v1", 0, 0, no_columns,
            error, capacity)) return false;

    std::string database_path, features_path, support_path, index_path;
    std::string validation_path;
    if (!scene_join(database_path, root, candidate.database.path,
                    error, capacity) ||
        !scene_join(features_path, root, candidate.terrain_features.path,
                    error, capacity) ||
        !scene_join(support_path, root, candidate.terrain_support.path,
                    error, capacity) ||
        !scene_join(index_path, root, candidate.scene_index.path,
                    error, capacity) ||
        !scene_join(validation_path, root, candidate.validation_file.path,
                    error, capacity) ||
        !scene_verify_sha(database_path, candidate.database.sha256,
                          error, capacity) ||
        !scene_verify_sha(features_path, candidate.terrain_features.sha256,
                          error, capacity) ||
        !scene_verify_sha(support_path, candidate.terrain_support.sha256,
                          error, capacity) ||
        !scene_verify_sha(index_path, candidate.scene_index.sha256,
                          error, capacity)) return false;

    const json_value* embedded_validation = json_member(document, "validation");
    json_value file_validation;
    if (embedded_validation == NULL)
        return scene_error(error, capacity,
            "motion manifest is missing embedded validation");
    if (!scene_json_load_verified(file_validation, validation_path.c_str(),
            candidate.validation_file.sha256, error, capacity)) return false;
    if (!scene_validation_validate(*embedded_validation, candidate.total_clips,
                                   error, capacity) ||
        !scene_validation_validate(file_validation, candidate.total_clips,
                                   error, capacity)) return false;
    if (!json_equal(*embedded_validation, file_validation))
        return scene_error(error, capacity,
            "embedded validation does not exactly equal validation.json");
    out = std::move(candidate);
    return true;
}

static inline int motion_source_for_frame(
    const motion_pack_manifest& manifest,const int frame)
{int low=0,high=static_cast<int>(manifest.sources.size());while(low<high){const int middle=low+(high-low)/2;const motion_source_record& source=manifest.sources[static_cast<size_t>(middle)];if(frame<source.range_start)high=middle;else if(frame>=source.range_stop)low=middle+1;else return middle;}return -1;}

static inline bool motion_manifest_validate_database(
    const motion_pack_manifest& manifest, const database& db,
    char* error, const int capacity)
{
    if (db.nframes() != manifest.database_frames)
        return scene_error(error, capacity,
            "database frame count %d does not match manifest %d",
            db.nframes(), manifest.database_frames);
    if (db.nbones() != G1_BoneCount ||
        !g1_skeleton_validate(db, error, capacity)) return false;
    if (db.features.rows != db.nframes() || db.features.cols != 31)
        return scene_error(error, capacity,
            "database matching features must be shaped frames x 31");
    if (manifest.flat_lmm_bundle) {
        float observed_maximum = 0.0f;
        if (!database_rotation_continuity_validate(
                db, 0.25f, error, capacity, &observed_maximum) ||
            fabsf(observed_maximum -
                  manifest.continuity_maximum_local_step) > 1.0e-6f)
            return scene_error(error, capacity,
                "flat local continuity maximum differs: %.9g vs %.9g",
                observed_maximum,
                manifest.continuity_maximum_local_step);
        if (manifest.feature_offset.size() != 31 ||
            manifest.feature_scale.size() != 31 ||
            db.features_offset.size != 31 || db.features_scale.size != 31)
            return scene_error(error, capacity,
                "flat feature normalization receipt is incomplete");
        for (int dimension = 0; dimension < 31; ++dimension) {
            if (feature_float_bits(db.features_offset(dimension)) !=
                    feature_float_bits(manifest.feature_offset[
                        static_cast<size_t>(dimension)]) ||
                feature_float_bits(db.features_scale(dimension)) !=
                    feature_float_bits(manifest.feature_scale[
                        static_cast<size_t>(dimension)]))
                return scene_error(error, capacity,
                    "flat feature normalization differs at dimension %d",
                    dimension);
        }
        for (int dimension = 27; dimension < 31; ++dimension) {
            if (feature_float_bits(db.features_offset(dimension)) != 0 ||
                feature_float_bits(db.features_scale(dimension)) !=
                    feature_float_bits(FLT_MAX))
                return scene_error(error, capacity,
                    "flat terrain normalization changed at dimension %d",
                    dimension);
            for (int frame = 0; frame < db.nframes(); ++frame) {
                if (feature_float_bits(db.features(frame, dimension)) != 0)
                    return scene_error(error, capacity,
                        "flat normalized terrain is nonzero at frame %d",
                        frame);
            }
        }
    }
    if (db.terrain_features.rows != db.nframes() ||
        db.terrain_features.cols != 4)
        return scene_error(error, capacity,
            "database terrain features must be shaped frames x 4");
    if (db.range_starts.size != static_cast<int>(manifest.sources.size()) ||
        db.range_stops.size != static_cast<int>(manifest.sources.size()) ||
        db.range_starts.data == NULL || db.range_stops.data == NULL)
        return scene_error(error, capacity,
            "database range arrays do not match manifest sources");
    for (size_t i = 0; i < manifest.sources.size(); ++i)
        if (db.range_starts(static_cast<int>(i)) !=
                manifest.sources[i].range_start ||
            db.range_stops(static_cast<int>(i)) !=
                manifest.sources[i].range_stop)
            return scene_error(error, capacity,
                "database range %zu does not match its manifest source", i);
    return true;
}

static inline bool scene_catalog_load(
    scene_catalog& out, const char* root,
    const motion_pack_manifest& manifest,
    char* error, const int capacity)
{
    if (manifest.scene_index.path != "scenes/index.json" ||
        manifest.scene_index.schema != "g1-terrain-scene-index/v1" ||
        !scene_sha_is_valid(manifest.scene_index.sha256))
        return scene_error(error, capacity,
            "manifest scene_index descriptor is not the published contract");
    std::string path;
    if (!scene_join(path, root, manifest.scene_index.path, error, capacity))
        return false;
    json_value document;
    if (!scene_json_load_verified(document, path.c_str(),
            manifest.scene_index.sha256, error, capacity)) return false;
    if (!scene_exact_keys(document,
            {"schema","default_scene_id","scene_ids","coordinate_signature",
             "surface_signature","scenes"},
            "scene index", error, capacity)) return false;
    scene_catalog candidate;
    std::string schema;
    if (!scene_member_string(schema, document, "schema", "scene index",
                             error, capacity) ||
        schema != "g1-terrain-scene-index/v1" ||
        !scene_member_string(candidate.default_scene_id, document,
            "default_scene_id", "scene index", error, capacity) ||
        candidate.default_scene_id != "grail-curb-default" ||
        !scene_member_string(candidate.coordinate_signature, document,
            "coordinate_signature", "scene index", error, capacity) ||
        candidate.coordinate_signature != G1_RuntimeCoordinateSignature ||
        candidate.coordinate_signature !=
            manifest.surface.coordinate_signature ||
        !scene_member_string(candidate.surface_signature, document,
            "surface_signature", "scene index", error, capacity) ||
        candidate.surface_signature != G1_RuntimeSurfaceSignature ||
        candidate.surface_signature != manifest.surface.signature)
        return scene_error(error, capacity,
            "scene index schema/default/signatures changed");
    const json_value* ids = json_member(document, "scene_ids");
    const json_value* scenes = json_member(document, "scenes");
    if (ids == NULL || scenes == NULL || ids->kind != json_array ||
        scenes->kind != json_array || ids->array_value.size() != 14 ||
        scenes->array_value.size() != 14)
        return scene_error(error, capacity,
            "scene index must contain exactly 14 ordered IDs/descriptors");
    for (int i = 0; i < 14; ++i) {
        std::string id;
        if (!scene_string(id, ids->array_value[static_cast<size_t>(i)],
                          "scene ID", error, capacity) ||
            id != G1_RuntimeSceneIds[i] || !scene_id_is_safe(id))
            return scene_error(error, capacity,
                "scene_ids position %d changed", i);
        candidate.ids.push_back(id);
        const json_value& descriptor_value =
            scenes->array_value[static_cast<size_t>(i)];
        if (!scene_exact_keys(descriptor_value, {"id","path","sha256"},
                              "scene descriptor", error, capacity)) return false;
        scene_descriptor descriptor;
        if (!scene_member_string(descriptor.id, descriptor_value, "id",
                                 "scene descriptor", error, capacity) ||
            !scene_member_string(descriptor.path, descriptor_value, "path",
                                 "scene descriptor", error, capacity) ||
            !scene_member_string(descriptor.sha256, descriptor_value, "sha256",
                                 "scene descriptor", error, capacity))
            return false;
        const std::string expected_path =
            std::string("scenes/") + id + "/scene.json";
        if (descriptor.id != id || descriptor.path != expected_path ||
            !scene_relative_is_safe(descriptor.path) ||
            !scene_sha_is_valid(descriptor.sha256))
            return scene_error(error, capacity,
                "scene descriptor %d ID/path/hash changed", i);
        for (size_t prior = 0; prior < candidate.scenes.size(); ++prior)
            if (candidate.scenes[prior].id == descriptor.id ||
                candidate.scenes[prior].path == descriptor.path ||
                candidate.scenes[prior].sha256 == descriptor.sha256)
                return scene_error(error, capacity,
                    "scene descriptor %d duplicates an ID/path/hash", i);
        candidate.scenes.push_back(descriptor);
    }
    out = std::move(candidate);
    return true;
}

static inline int scene_catalog_find(
    const scene_catalog& catalog,const char* id)
{if(id==NULL)return -1;for(size_t i=0;i<catalog.ids.size();++i)if(catalog.ids[i]==id)return static_cast<int>(i);return -1;}

static inline const scene_route* scene_route_find(
    const scene_metadata& scene,const char* id)
{if(id==NULL)return NULL;for(size_t i=0;i<scene.routes.size();++i)if(scene.routes[i].id==id)return &scene.routes[i];return NULL;}

static inline bool scene_bounds2_valid(const bounds2& bounds)
{
    return terrain_float_is_finite(bounds.min_x) &&
           terrain_float_is_finite(bounds.min_z) &&
           terrain_float_is_finite(bounds.max_x) &&
           terrain_float_is_finite(bounds.max_z) &&
           bounds.min_x < bounds.max_x && bounds.min_z < bounds.max_z;
}

static inline bool scene_bounds3_valid(const bounds3d& bounds)
{
    return terrain_double_is_finite(bounds.minimum.x) &&
           terrain_double_is_finite(bounds.minimum.y) &&
           terrain_double_is_finite(bounds.minimum.z) &&
           terrain_double_is_finite(bounds.maximum.x) &&
           terrain_double_is_finite(bounds.maximum.y) &&
           terrain_double_is_finite(bounds.maximum.z) &&
           bounds.minimum.x < bounds.maximum.x &&
           bounds.minimum.z < bounds.maximum.z &&
           bounds.minimum.y <= bounds.maximum.y;
}

static inline bool scene_bounds_inside(
    const bounds2& inner, const bounds2& outer)
{
    return inner.min_x >= outer.min_x && inner.max_x <= outer.max_x &&
           inner.min_z >= outer.min_z && inner.max_z <= outer.max_z;
}

static inline bool scene_bounds2_inside_binary64(
    const bounds2& inner, const bounds3d& outer)
{
    return terrain_float_is_finite(inner.min_x) &&
           terrain_float_is_finite(inner.min_z) &&
           terrain_float_is_finite(inner.max_x) &&
           terrain_float_is_finite(inner.max_z) &&
           static_cast<double>(inner.min_x) >= outer.minimum.x &&
           static_cast<double>(inner.max_x) <= outer.maximum.x &&
           static_cast<double>(inner.min_z) >= outer.minimum.z &&
           static_cast<double>(inner.max_z) <= outer.maximum.z;
}

static inline bool scene_parse_bounds2_pair(
    bounds2& out, const json_value& minimum, const json_value& maximum,
    const char* label, char* error, const int capacity)
{
    float low[2] = {}, high[2] = {};
    if (!scene_binary32_array(low, 2, minimum, label, error, capacity) ||
        !scene_binary32_array(high, 2, maximum, label, error, capacity))
        return false;
    bounds2 candidate;
    candidate.min_x = low[0];
    candidate.min_z = low[1];
    candidate.max_x = high[0];
    candidate.max_z = high[1];
    if (!scene_bounds2_valid(candidate))
        return scene_error(error, capacity,
            "%s must have strict finite X/Z extent", label);
    out = candidate;
    return true;
}

static inline bool scene_parse_bounds3(
    bounds3d& out, const json_value& minimum, const json_value& maximum,
    const char* label, char* error, const int capacity)
{
    double low[3] = {}, high[3] = {};
    if (!scene_double_array(low, 3, minimum, label, error, capacity) ||
        !scene_double_array(high, 3, maximum, label, error, capacity))
        return false;
    bounds3d candidate;
    candidate.minimum = point3d{low[0],low[1],low[2]};
    candidate.maximum = point3d{high[0],high[1],high[2]};
    if (!scene_bounds3_valid(candidate))
        return scene_error(error, capacity,
            "%s must have strict finite X/Z and ordered Y extent", label);
    out = candidate;
    return true;
}

static inline bool scene_region_parse(
    scene_region& out, const json_value& value,
    const bounds2& lookahead, const std::vector<std::string>& prior_ids,
    char* error, const int capacity)
{
    if (!scene_exact_keys(value, {"id","bounds_xz"},
                          "scene region", error, capacity)) return false;
    scene_region candidate;
    if (!scene_member_string(candidate.id, value, "id", "scene region",
                             error, capacity) || candidate.id.empty())
        return scene_error(error, capacity,
            "scene region ID must be nonempty");
    for (size_t i = 0; i < prior_ids.size(); ++i)
        if (prior_ids[i] == candidate.id)
            return scene_error(error, capacity,
                "scene region ID '%s' is duplicated", candidate.id.c_str());
    const json_value* encoded = json_member(value, "bounds_xz");
    float bounds[4] = {};
    if (encoded == NULL ||
        !scene_binary32_array(bounds, 4, *encoded, "region bounds",
                              error, capacity)) return false;
    candidate.bounds.min_x = bounds[0];
    candidate.bounds.max_x = bounds[1];
    candidate.bounds.min_z = bounds[2];
    candidate.bounds.max_z = bounds[3];
    if (!scene_bounds2_valid(candidate.bounds) ||
        !scene_bounds_inside(candidate.bounds, lookahead))
        return scene_error(error, capacity,
            "scene region '%s' leaves lookahead bounds",
            candidate.id.c_str());
    out = candidate;
    return true;
}

static inline bool scene_route_parse(
    scene_route& out, const json_value& value, const bounds2& lookahead,
    const vec3& spawn, const std::vector<std::string>& prior_ids,
    char* error, const int capacity)
{
    if (!scene_exact_keys(value,
            {"id","waypoints_xz","expected_outcome","walkability_class",
             "landing_hold_seconds"},
            "scene route", error, capacity)) return false;
    scene_route candidate;
    if (!scene_member_string(candidate.id, value, "id", "scene route",
                             error, capacity) || candidate.id.empty())
        return scene_error(error, capacity, "scene route ID must be nonempty");
    for (size_t i = 0; i < prior_ids.size(); ++i)
        if (prior_ids[i] == candidate.id)
            return scene_error(error, capacity,
                "scene route ID '%s' is duplicated", candidate.id.c_str());
    if (!scene_member_string(candidate.expected_outcome, value,
            "expected_outcome", "scene route", error, capacity) ||
        !scene_member_int(candidate.walkability_class, value,
            "walkability_class", "scene route", error, capacity)) return false;
    const bool mapped =
        (candidate.expected_outcome == "traverse" &&
         candidate.walkability_class == 1) ||
        (candidate.expected_outcome == "safe-stop" &&
         candidate.walkability_class == 0) ||
        (candidate.expected_outcome == "traverse-or-safe-stop" &&
         candidate.walkability_class == 2);
    if (!mapped)
        return scene_error(error, capacity,
            "scene route outcome/class mapping is invalid");
    const json_value* hold = json_member(value, "landing_hold_seconds");
    if (hold == NULL || !scene_number_binary32(candidate.landing_hold_seconds,
            *hold, "landing_hold_seconds", error, capacity) ||
        candidate.landing_hold_seconds < 0.0f)
        return scene_error(error, capacity,
            "scene route landing hold must be nonnegative binary32");
    const json_value* waypoints = json_member(value, "waypoints_xz");
    if (waypoints == NULL || waypoints->kind != json_array ||
        waypoints->array_value.size() < 2)
        return scene_error(error, capacity,
            "scene route requires at least two waypoints");
    for (size_t i = 0; i < waypoints->array_value.size(); ++i) {
        float point[2] = {};
        if (!scene_binary32_array(point, 2, waypoints->array_value[i],
                                  "route waypoint", error, capacity) ||
            !scene_inside(lookahead, point[0], point[1]))
            return scene_error(error, capacity,
                "scene route waypoint %zu leaves lookahead bounds", i);
        candidate.waypoints_xz.push_back(std::make_pair(point[0], point[1]));
    }
    if (candidate.waypoints_xz.front().first != spawn.x ||
        candidate.waypoints_xz.front().second != spawn.z)
        return scene_error(error, capacity,
            "scene route must start at the published spawn");
    if (candidate.landing_hold_seconds > 0.0f &&
        candidate.waypoints_xz.size() < 4)
        return scene_error(error, capacity,
            "positive landing hold requires landing index 2 and a later exit");
    out = candidate;
    return true;
}

static inline bool scene_metadata_parse(
    scene_metadata& out, const json_value& document, const char* expected_id,
    const motion_pack_manifest& manifest, const char* path,
    char* error, const int capacity)
{
    const char* shown = path != NULL ? path : "<scene>";
    if (expected_id == NULL || !scene_id_is_safe(expected_id))
        return scene_error(error, capacity,
            "%s: expected scene ID is unsafe", shown);
    if (!scene_exact_keys(document,
            {"schema","id","label","provenance","coordinate_signature",
             "surface_signature","terrain_feature_distances_m","heightfield",
             "mesh","walkability","bounds","spawn","regions","routes"},
            "scene", error, capacity)) return false;
    scene_metadata candidate;
    std::string schema;
    if (!scene_member_string(schema, document, "schema", "scene",
                             error, capacity) ||
        schema != "g1-terrain-scene/v1" ||
        !scene_member_string(candidate.id, document, "id", "scene",
                             error, capacity) ||
        candidate.id != expected_id || !scene_id_is_safe(candidate.id) ||
        !scene_member_string(candidate.label, document, "label", "scene",
                             error, capacity) || candidate.label.empty())
        return scene_error(error, capacity,
            "%s: scene schema/ID/label changed", shown);
    int scene_position = -1;
    for (int i = 0; i < 14; ++i)
        if (candidate.id == G1_RuntimeSceneIds[i]) scene_position = i;
    if (scene_position < 0)
        return scene_error(error, capacity,
            "%s: scene ID is not in the locked 14-scene catalog", shown);
    if (!scene_member_string(candidate.coordinate_signature, document,
            "coordinate_signature", "scene", error, capacity) ||
        candidate.coordinate_signature != G1_RuntimeCoordinateSignature ||
        candidate.coordinate_signature != manifest.surface.coordinate_signature ||
        !scene_member_string(candidate.surface_signature, document,
            "surface_signature", "scene", error, capacity) ||
        candidate.surface_signature != G1_RuntimeSurfaceSignature ||
        candidate.surface_signature != manifest.surface.signature)
        return scene_error(error, capacity,
            "%s: scene coordinate/surface signature changed", shown);
    const json_value* distances = json_member(document,
                                               "terrain_feature_distances_m");
    static const double expected_distances[4] = {0.25,0.5,0.75,1.0};
    if (distances == NULL || distances->kind != json_array ||
        distances->array_value.size() != 4)
        return scene_error(error, capacity,
            "%s: terrain feature distances changed", shown);
    for (int i = 0; i < 4; ++i) {
        double distance = 0.0;
        if (!scene_number_double(distance,
                distances->array_value[static_cast<size_t>(i)],
                "terrain feature distance", error, capacity) ||
            distance != expected_distances[i])
            return scene_error(error, capacity,
                "%s: terrain feature distances changed", shown);
    }

    const json_value* provenance = json_member(document, "provenance");
    if (provenance == NULL || !scene_exact_keys(*provenance,
            {"kind","source_ids","parameters"}, "provenance",
            error, capacity) ||
        !scene_member_string(candidate.provenance_kind, *provenance,
            "kind", "provenance", error, capacity) ||
        (candidate.provenance_kind != "grail" &&
         candidate.provenance_kind != "procedural"))
        return scene_error(error, capacity,
            "%s: scene provenance kind/shape is invalid", shown);
    const json_value* source_ids = json_member(*provenance, "source_ids");
    const json_value* parameters = json_member(*provenance, "parameters");
    if (source_ids == NULL || source_ids->kind != json_array ||
        parameters == NULL || parameters->kind != json_object)
        return scene_error(error, capacity,
            "%s: scene provenance source_ids/parameters are invalid", shown);
    for (size_t i = 0; i < source_ids->array_value.size(); ++i) {
        std::string source;
        if (!scene_string(source, source_ids->array_value[i],
                          "provenance source ID", error, capacity) ||
            source.empty())
            return scene_error(error, capacity,
                "%s: provenance source IDs must be nonempty strings", shown);
        candidate.provenance_source_ids.push_back(source);
    }

    const json_value* heightfield_value = json_member(document, "heightfield");
    if (heightfield_value == NULL || !scene_exact_keys(*heightfield_value,
            {"path","schema","version","nx","nz","origin_x","origin_z",
             "cell_size_m","exterior_height_m","interpolation","diagonal",
             "sha256"},
            "heightfield", error, capacity)) return false;
    if (!scene_member_string(candidate.heightfield.path, *heightfield_value,
            "path", "heightfield", error, capacity) ||
        candidate.heightfield.path != "terrain.bin" ||
        !scene_relative_is_safe(candidate.heightfield.path) ||
        !scene_member_string(candidate.heightfield.schema, *heightfield_value,
            "schema", "heightfield", error, capacity) ||
        candidate.heightfield.schema != "G1HF/v2" ||
        !scene_member_string(candidate.heightfield.sha256, *heightfield_value,
            "sha256", "heightfield", error, capacity) ||
        !scene_sha_is_valid(candidate.heightfield.sha256) ||
        !scene_member_int(candidate.heightfield.version, *heightfield_value,
            "version", "heightfield", error, capacity) ||
        candidate.heightfield.version != 2 ||
        !scene_member_int(candidate.heightfield_nx, *heightfield_value,
            "nx", "heightfield", error, capacity) ||
        !scene_member_int(candidate.heightfield_nz, *heightfield_value,
            "nz", "heightfield", error, capacity) ||
        candidate.heightfield_nx < 2 || candidate.heightfield_nz < 2)
        return scene_error(error, capacity,
            "%s: heightfield descriptor changed", shown);
    const json_value* origin_x = json_member(*heightfield_value, "origin_x");
    const json_value* origin_z = json_member(*heightfield_value, "origin_z");
    const json_value* cell_size = json_member(*heightfield_value, "cell_size_m");
    const json_value* exterior = json_member(*heightfield_value,
                                             "exterior_height_m");
    std::string interpolation, diagonal;
    if (origin_x == NULL || origin_z == NULL || cell_size == NULL ||
        exterior == NULL ||
        !scene_number_binary32(candidate.heightfield_origin_x, *origin_x,
            "heightfield origin_x", error, capacity) ||
        !scene_number_binary32(candidate.heightfield_origin_z, *origin_z,
            "heightfield origin_z", error, capacity) ||
        !scene_number_binary32(candidate.heightfield_cell_size, *cell_size,
            "heightfield cell_size_m", error, capacity) ||
        !terrain_float_is_positive_normal(candidate.heightfield_cell_size) ||
        !scene_number_binary32(candidate.heightfield_exterior_height, *exterior,
            "heightfield exterior_height_m", error, capacity) ||
        candidate.heightfield_cell_size != manifest.surface.cell_size ||
        candidate.heightfield_exterior_height !=
            manifest.surface.exterior_height ||
        !scene_member_string(interpolation, *heightfield_value,
            "interpolation", "heightfield", error, capacity) ||
        interpolation != manifest.surface.heightfield_interpolation ||
        interpolation != "fixed-diagonal-triangles" ||
        !scene_member_string(diagonal, *heightfield_value,
            "diagonal", "heightfield", error, capacity) ||
        diagonal != manifest.surface.heightfield_diagonal ||
        diagonal != "min-x-min-z_to_max-x-max-z")
        return scene_error(error, capacity,
            "%s: heightfield scalar/surface contract changed", shown);

    const json_value* mesh_value = json_member(document, "mesh");
    if (mesh_value == NULL || !scene_exact_keys(*mesh_value,
            {"path","schema","sha256"}, "mesh", error, capacity) ||
        !scene_member_string(candidate.mesh.path, *mesh_value, "path", "mesh",
                             error, capacity) ||
        candidate.mesh.path != "terrain.obj" ||
        !scene_relative_is_safe(candidate.mesh.path) ||
        !scene_member_string(candidate.mesh.schema, *mesh_value,
            "schema", "mesh", error, capacity) ||
        candidate.mesh.schema != "obj/v1" ||
        !scene_member_string(candidate.mesh.sha256, *mesh_value,
            "sha256", "mesh", error, capacity) ||
        !scene_sha_is_valid(candidate.mesh.sha256))
        return scene_error(error, capacity,
            "%s: mesh descriptor changed", shown);

    const json_value* walkability_value = json_member(document, "walkability");
    if (walkability_value == NULL || !scene_exact_keys(*walkability_value,
            {"path","schema","version","nx","nz","classes","sha256"},
            "walkability", error, capacity) ||
        !scene_member_string(candidate.walkability.path, *walkability_value,
            "path", "walkability", error, capacity) ||
        candidate.walkability.path != "walkability.bin" ||
        !scene_relative_is_safe(candidate.walkability.path) ||
        !scene_member_string(candidate.walkability.schema, *walkability_value,
            "schema", "walkability", error, capacity) ||
        candidate.walkability.schema != "G1WM/v1" ||
        !scene_member_string(candidate.walkability.sha256, *walkability_value,
            "sha256", "walkability", error, capacity) ||
        !scene_sha_is_valid(candidate.walkability.sha256) ||
        !scene_member_int(candidate.walkability.version, *walkability_value,
            "version", "walkability", error, capacity) ||
        candidate.walkability.version != 1 ||
        !scene_member_int(candidate.walkability_nx, *walkability_value,
            "nx", "walkability", error, capacity) ||
        !scene_member_int(candidate.walkability_nz, *walkability_value,
            "nz", "walkability", error, capacity) ||
        candidate.walkability_nx < 2 || candidate.walkability_nz < 2)
        return scene_error(error, capacity,
            "%s: walkability descriptor changed", shown);
    const json_value* classes = json_member(*walkability_value, "classes");
    int blocked_class = -1, certified_class = -1, stress_class = -1;
    if (classes == NULL || !scene_exact_keys(*classes,
            {"blocked","certified","stress"}, "walkability classes",
            error, capacity) ||
        !scene_member_int(blocked_class, *classes, "blocked",
            "walkability classes", error, capacity) || blocked_class != 0 ||
        !scene_member_int(certified_class, *classes, "certified",
            "walkability classes", error, capacity) || certified_class != 1 ||
        !scene_member_int(stress_class, *classes, "stress",
            "walkability classes", error, capacity) || stress_class != 2)
        return scene_error(error, capacity,
            "%s: walkability class mapping changed", shown);

    const json_value* bounds = json_member(document, "bounds");
    if (bounds == NULL || !scene_exact_keys(*bounds,
            {"mesh_min_xyz","mesh_max_xyz","heightfield_min_xyz",
             "heightfield_max_xyz","playable_min_xz","playable_max_xz",
             "lookahead_min_xz","lookahead_max_xz"},
            "bounds", error, capacity)) return false;
    const json_value* mesh_min = json_member(*bounds, "mesh_min_xyz");
    const json_value* mesh_max = json_member(*bounds, "mesh_max_xyz");
    const json_value* hf_min = json_member(*bounds, "heightfield_min_xyz");
    const json_value* hf_max = json_member(*bounds, "heightfield_max_xyz");
    const json_value* playable_min = json_member(*bounds, "playable_min_xz");
    const json_value* playable_max = json_member(*bounds, "playable_max_xz");
    const json_value* lookahead_min = json_member(*bounds, "lookahead_min_xz");
    const json_value* lookahead_max = json_member(*bounds, "lookahead_max_xz");
    if (mesh_min == NULL || mesh_max == NULL || hf_min == NULL ||
        hf_max == NULL || playable_min == NULL || playable_max == NULL ||
        lookahead_min == NULL || lookahead_max == NULL ||
        !scene_parse_bounds3(candidate.mesh_bounds, *mesh_min, *mesh_max,
                             "mesh bounds", error, capacity) ||
        !scene_parse_bounds3(candidate.heightfield_bounds, *hf_min, *hf_max,
                             "heightfield bounds", error, capacity) ||
        !scene_parse_bounds2_pair(candidate.playable_bounds,
            *playable_min, *playable_max, "playable bounds", error, capacity) ||
        !scene_parse_bounds2_pair(candidate.lookahead_bounds,
            *lookahead_min, *lookahead_max, "lookahead bounds", error, capacity))
        return false;
    if (!scene_bounds2_inside_binary64(candidate.lookahead_bounds,
                                       candidate.heightfield_bounds) ||
        !scene_bounds2_inside_binary64(candidate.playable_bounds,
                                       candidate.heightfield_bounds) ||
        !scene_bounds_inside(candidate.playable_bounds,
                             candidate.lookahead_bounds))
        return scene_error(error, capacity,
            "%s: playable/lookahead bounds leave the heightfield", shown);

    const json_value* spawn = json_member(document, "spawn");
    if (spawn == NULL || !scene_exact_keys(*spawn,
            {"position","yaw_radians"}, "spawn", error, capacity)) return false;
    const json_value* position = json_member(*spawn, "position");
    const json_value* yaw = json_member(*spawn, "yaw_radians");
    float spawn_values[3] = {};
    if (position == NULL || yaw == NULL ||
        !scene_binary32_array(spawn_values, 3, *position,
                              "spawn position", error, capacity) ||
        !scene_number_binary32(candidate.spawn_yaw, *yaw,
                               "spawn yaw", error, capacity)) return false;
    candidate.spawn_position = vec3(
        spawn_values[0], spawn_values[1], spawn_values[2]);
    if (!scene_inside(candidate.playable_bounds,
                      candidate.spawn_position.x, candidate.spawn_position.z))
        return scene_error(error, capacity,
            "%s: spawn leaves playable bounds", shown);

    const json_value* regions = json_member(document, "regions");
    if (regions == NULL || !scene_exact_keys(*regions,
            {"certified","stress","blocked"}, "regions",
            error, capacity)) return false;
    const char* region_names[] = {"certified","stress","blocked"};
    std::vector<scene_region>* region_outputs[] = {
        &candidate.certified_regions, &candidate.stress_regions,
        &candidate.blocked_regions};
    std::vector<std::string> region_ids;
    for (int group = 0; group < 3; ++group) {
        const json_value* entries = json_member(*regions, region_names[group]);
        if (entries == NULL || entries->kind != json_array)
            return scene_error(error, capacity,
                "%s: region group %s must be an array", shown,
                region_names[group]);
        for (size_t i = 0; i < entries->array_value.size(); ++i) {
            scene_region region;
            if (!scene_region_parse(region, entries->array_value[i],
                    candidate.lookahead_bounds, region_ids,
                    error, capacity)) return false;
            region_ids.push_back(region.id);
            region_outputs[group]->push_back(region);
        }
    }

    const json_value* routes = json_member(document, "routes");
    if (routes == NULL || routes->kind != json_array ||
        routes->array_value.empty())
        return scene_error(error, capacity,
            "%s: scene must have at least one route", shown);
    std::vector<std::string> route_ids;
    for (size_t i = 0; i < routes->array_value.size(); ++i) {
        scene_route route;
        if (!scene_route_parse(route, routes->array_value[i],
                candidate.lookahead_bounds, candidate.spawn_position,
                route_ids, error, capacity)) return false;
        route_ids.push_back(route.id);
        candidate.routes.push_back(route);
    }
    static const int route_counts[14] = {
        1,1,1,1,3,2,1,1,1,1,1,1,2,2};
    static const uint32_t flat_positive_z_waypoint_bits[][2] = {
        {UINT32_C(0x00000000), UINT32_C(0x00000000)},
        {UINT32_C(0x00000000), UINT32_C(0x3f800000)},
    };
    static const uint32_t flat_positive_x_waypoint_bits[][2] = {
        {UINT32_C(0x00000000), UINT32_C(0x00000000)},
        {UINT32_C(0x3f800000), UINT32_C(0x00000000)},
    };
    static const uint32_t landing_side_exit_stress_waypoint_bits[][2] = {
        {UINT32_C(0x00000000), UINT32_C(0x00000000)},
        {UINT32_C(0x00000000), UINT32_C(0x3fe00000)},
        {UINT32_C(0x00000000), UINT32_C(0x407d70a4)},
        {UINT32_C(0x00000000), UINT32_C(0x40b0f5c3)},
    };
    static const uint32_t tangent_level_boundary_waypoint_bits[][2] = {
        {UINT32_C(0x00000000), UINT32_C(0x00000000)},
        {UINT32_C(0x3f1eb852), UINT32_C(0x40000000)},
        {UINT32_C(0x3f1eb852), UINT32_C(0x40c00000)},
    };
    struct route_contract {
        const char* id;
        const char* outcome;
        int classification;
        const uint32_t (*waypoint_bit_contract)[2];
        size_t waypoint_count;
        uint32_t landing_hold_bits;
    };
    static const route_contract expected_routes[14][3] = {
        {{"curb-forward","traverse-or-safe-stop",2,NULL,0,0}},
        {{"curb-forward","traverse",1,NULL,0,0}},
        {{"curb-forward","traverse-or-safe-stop",2,NULL,0,0}},
        {{"curb-forward","traverse-or-safe-stop",2,NULL,0,0}},
        {
            {"ascent-landing-descent","traverse",1,NULL,0,0},
            {"flat-positive-z","traverse",1,
             flat_positive_z_waypoint_bits,2,UINT32_C(0x00000000)},
            {"flat-positive-x","traverse",1,
             flat_positive_x_waypoint_bits,2,UINT32_C(0x00000000)},
        },
        {
            {"ascent-landing-descent","traverse",1,NULL,0,0},
            {"landing-side-exit-stress","traverse",1,
             landing_side_exit_stress_waypoint_bits,4,
             UINT32_C(0x00000000)},
        },
        {{"ascent-landing-descent","traverse",1,NULL,0,0}},
        {{"up-landing-down","traverse",1,NULL,0,0}},
        {{"up-landing-down","traverse",1,NULL,0,0}},
        {{"up-landing-down","traverse-or-safe-stop",2,NULL,0,0}},
        {{"forward-cross-slope","traverse",1,NULL,0,0}},
        {{"forward-cross-slope","traverse",1,NULL,0,0}},
        {
            {"full-course","traverse",1,NULL,0,0},
            {"tangent-level-boundary","traverse",1,
             tangent_level_boundary_waypoint_bits,3,
             UINT32_C(0x00000000)},
        },
        {
            {"wall-safe-stop","safe-stop",0,NULL,0,0},
            {"ramp-safe-stop","safe-stop",0,NULL,0,0},
        },
    };
    if (candidate.routes.size() !=
        static_cast<size_t>(route_counts[scene_position]))
        return scene_error(error, capacity,
            "%s: scene route count changed", shown);
    for (int i = 0; i < route_counts[scene_position]; ++i) {
        const scene_route& route = candidate.routes[static_cast<size_t>(i)];
        const route_contract& expected = expected_routes[scene_position][i];
        if (route.id != expected.id)
            return scene_error(error, capacity,
                "%s: scene route ID/order changed", shown);
        if (route.expected_outcome != expected.outcome ||
            route.walkability_class != expected.classification)
            return scene_error(error, capacity,
                "%s: scene route outcome/class changed", shown);
        if (expected.waypoint_bit_contract != NULL) {
            if (route.waypoints_xz.size() != expected.waypoint_count)
                return scene_error(error, capacity,
                    "%s: scene route waypoint count changed", shown);
            const json_value* encoded_waypoints = json_member(
                routes->array_value[static_cast<size_t>(i)],
                "waypoints_xz");
            const json_value* encoded_hold = json_member(
                routes->array_value[static_cast<size_t>(i)],
                "landing_hold_seconds");
            if (encoded_waypoints == NULL ||
                encoded_waypoints->kind != json_array ||
                encoded_waypoints->array_value.size() !=
                    expected.waypoint_count ||
                encoded_hold == NULL || encoded_hold->kind != json_number)
                return scene_error(error, capacity,
                    "%s: scene route bit encoding changed", shown);
            for (size_t waypoint = 0;
                 waypoint < expected.waypoint_count; ++waypoint) {
                const json_value& encoded_point =
                    encoded_waypoints->array_value[waypoint];
                if (encoded_point.kind != json_array ||
                    encoded_point.array_value.size() != 2)
                    return scene_error(error, capacity,
                        "%s: scene route waypoint encoding changed", shown);
                const float parsed[2] = {
                    route.waypoints_xz[waypoint].first,
                    route.waypoints_xz[waypoint].second,
                };
                for (int axis = 0; axis < 2; ++axis) {
                    const json_value& encoded_component =
                        encoded_point.array_value[static_cast<size_t>(axis)];
                    if (encoded_component.kind != json_number ||
                        terrain_float_bits(parsed[axis]) !=
                            expected.waypoint_bit_contract[waypoint][axis] ||
                        terrain_float_bits(static_cast<float>(
                            encoded_component.number_value)) !=
                            expected.waypoint_bit_contract[waypoint][axis])
                        return scene_error(error, capacity,
                            "%s: scene route waypoint bits changed", shown);
                }
            }
            if (terrain_float_bits(route.landing_hold_seconds) !=
                    expected.landing_hold_bits ||
                terrain_float_bits(static_cast<float>(
                    encoded_hold->number_value)) !=
                    expected.landing_hold_bits)
                return scene_error(error, capacity,
                    "%s: scene route landing hold bits changed", shown);
        }
    }
    out = std::move(candidate);
    return true;
}

#if defined(__GNUC__) || defined(__clang__)
#define SCENE_RUNTIME_NOINLINE __attribute__((noinline))
#else
#define SCENE_RUNTIME_NOINLINE
#endif

static inline SCENE_RUNTIME_NOINLINE double scene_axis_max_binary64(
    const float origin, const int count, const float cell_size)
{
    const volatile double product =
        static_cast<double>(count - 1) * static_cast<double>(cell_size);
    return static_cast<double>(origin) + product;
}

static inline bool scene_candidate_validate(
    const scene_pack& candidate, char* error, const int capacity)
{
    const scene_metadata& metadata = candidate.metadata;
    const heightfield& field = candidate.terrain;
    const walkability_grid& grid = candidate.walkability;
    size_t field_count = 0;
    size_t grid_count = 0;
    if (field.nx < 2 || field.nz < 2 || grid.nx < 2 || grid.nz < 2 ||
        !terrain_size_multiply(static_cast<size_t>(field.nx),
                               static_cast<size_t>(field.nz), field_count) ||
        field_count > static_cast<size_t>(INT_MAX) ||
        !terrain_size_multiply(static_cast<size_t>(grid.nx),
                               static_cast<size_t>(grid.nz), grid_count) ||
        grid_count > static_cast<size_t>(INT_MAX))
        return scene_error(error, capacity,
            "scene grid dimensions are invalid or exceed runtime capacity");
    if (field.version != 2 || field.nx != metadata.heightfield_nx ||
        field.nz != metadata.heightfield_nz ||
        field.origin_x != metadata.heightfield_origin_x ||
        field.origin_z != metadata.heightfield_origin_z ||
        field.cell_size != metadata.heightfield_cell_size ||
        field.exterior_height != metadata.heightfield_exterior_height)
        return scene_error(error, capacity,
            "%s: G1HF header does not equal scene metadata",
            candidate.terrain_path.c_str());
    if (grid.nx != metadata.walkability_nx ||
        grid.nz != metadata.walkability_nz ||
        grid.nx != field.nx || grid.nz != field.nz)
        return scene_error(error, capacity,
            "%s: G1WM grid does not equal scene metadata/G1HF",
            candidate.walkability_path.c_str());
    if (field.heights.size != static_cast<int>(field_count) ||
        field.heights.data == NULL ||
        grid.cells.size != static_cast<int>(grid_count) ||
        grid.cells.data == NULL)
        return scene_error(error, capacity,
            "scene binary grids are incomplete");
    double minimum_y = static_cast<double>(field.heights(0));
    double maximum_y = minimum_y;
    for (int i = 1; i < field.heights.size; ++i) {
        const double height = static_cast<double>(field.heights(i));
        if (height < minimum_y) minimum_y = height;
        if (height > maximum_y) maximum_y = height;
    }
    const double hf_min_x = static_cast<double>(field.origin_x);
    const double hf_min_z = static_cast<double>(field.origin_z);
    const double hf_max_x = scene_axis_max_binary64(
        field.origin_x, field.nx, field.cell_size);
    const double hf_max_z = scene_axis_max_binary64(
        field.origin_z, field.nz, field.cell_size);
    if (metadata.heightfield_bounds.minimum.x != hf_min_x ||
        metadata.heightfield_bounds.minimum.y != minimum_y ||
        metadata.heightfield_bounds.minimum.z != hf_min_z ||
        metadata.heightfield_bounds.maximum.x != hf_max_x ||
        metadata.heightfield_bounds.maximum.y != maximum_y ||
        metadata.heightfield_bounds.maximum.z != hf_max_z)
        return scene_error(error, capacity,
            "scene heightfield binary64 bounds do not equal decoded G1HF");
    const double mesh_min_x = static_cast<double>(field.origin_x);
    const double mesh_min_z = static_cast<double>(field.origin_z);
    const double mesh_max_x = static_cast<double>(static_cast<float>(hf_max_x));
    const double mesh_max_z = static_cast<double>(static_cast<float>(hf_max_z));
    if (metadata.mesh_bounds.minimum.x != mesh_min_x ||
        metadata.mesh_bounds.minimum.y != minimum_y ||
        metadata.mesh_bounds.minimum.z != mesh_min_z ||
        metadata.mesh_bounds.maximum.x != mesh_max_x ||
        metadata.mesh_bounds.maximum.y != maximum_y ||
        metadata.mesh_bounds.maximum.z != mesh_max_z)
        return scene_error(error, capacity,
            "scene mesh binary32-rounded bounds do not equal decoded G1HF");

    float maximum_step = 0.0f;
    if (!terrain_f32_div(maximum_step, field.cell_size, 2.0f) ||
        !terrain_float_is_positive_normal(maximum_step))
        return scene_error(error, capacity,
            "scene route sampling step is not representable");
    for (size_t route_index = 0;
         route_index < metadata.routes.size(); ++route_index) {
        const scene_route& route = metadata.routes[route_index];
        bool first_sample = true;
        bool saw_blocked = false;
        int transitions = 0;
        int prior_class = -1;
        for (size_t segment = 1; segment < route.waypoints_xz.size(); ++segment) {
            const std::pair<float,float>& start =
                route.waypoints_xz[segment - 1];
            const std::pair<float,float>& stop = route.waypoints_xz[segment];
            int samples = 0;
            if (!scene_route_sample_count(samples, start, stop, maximum_step))
                return scene_error(error, capacity,
                    "scene route '%s' has an invalid/zero-length segment",
                    route.id.c_str());
            for (int step = 0; step <= samples; ++step) {
                if (segment > 1 && step == 0) continue;
                float x = 0.0f, z = 0.0f;
                if (!scene_binary32_lerp(x, start.first, stop.first,
                                         step, samples) ||
                    !scene_binary32_lerp(z, start.second, stop.second,
                                         step, samples))
                    return scene_error(error, capacity,
                        "scene route '%s' sample is not representable",
                        route.id.c_str());
                int ix = 0, iz = 0;
                if (!walkability_nearest_axis(
                        ix, x, field.origin_x, field.cell_size, field.nx) ||
                    !walkability_nearest_axis(
                        iz, z, field.origin_z, field.cell_size, field.nz))
                    return scene_error(error, capacity,
                        "scene route '%s' leaves the G1HF/G1WM grid",
                        route.id.c_str());
                const int classification = walkability_class_at(grid, field, x, z);
                if (route.expected_outcome == "traverse" &&
                    classification != 1)
                    return scene_error(error, capacity,
                        "traverse route '%s' leaves class 1", route.id.c_str());
                if (route.expected_outcome == "traverse-or-safe-stop" &&
                    classification != 2)
                    return scene_error(error, capacity,
                        "stress route '%s' leaves class 2", route.id.c_str());
                if (route.expected_outcome == "safe-stop") {
                    if (first_sample && classification != 1)
                        return scene_error(error, capacity,
                            "safe-stop route '%s' must begin in class 1",
                            route.id.c_str());
                    if (classification == 0) saw_blocked = true;
                    else if (classification == 1 && saw_blocked)
                        return scene_error(error, capacity,
                            "safe-stop route '%s' re-enters class 1",
                            route.id.c_str());
                    else if (classification != 1)
                        return scene_error(error, capacity,
                            "safe-stop route '%s' uses an invalid class",
                            route.id.c_str());
                    if (prior_class == 1 && classification == 0) ++transitions;
                }
                prior_class = classification;
                first_sample = false;
            }
        }
        if (route.expected_outcome == "safe-stop" &&
            (!saw_blocked || transitions != 1))
            return scene_error(error, capacity,
                "safe-stop route '%s' must cross exactly once to class 0",
                route.id.c_str());
    }
    return true;
}

static inline void scene_pack_swap(scene_pack& first, scene_pack& second)
{
    std::swap(first.metadata, second.metadata);
    std::swap(first.terrain.version, second.terrain.version);
    std::swap(first.terrain.nx, second.terrain.nx);
    std::swap(first.terrain.nz, second.terrain.nz);
    std::swap(first.terrain.origin_x, second.terrain.origin_x);
    std::swap(first.terrain.origin_z, second.terrain.origin_z);
    std::swap(first.terrain.cell_size, second.terrain.cell_size);
    std::swap(first.terrain.exterior_height, second.terrain.exterior_height);
    std::swap(first.terrain.heights.size, second.terrain.heights.size);
    std::swap(first.terrain.heights.data, second.terrain.heights.data);
    std::swap(first.walkability.nx, second.walkability.nx);
    std::swap(first.walkability.nz, second.walkability.nz);
    std::swap(first.walkability.cells.size, second.walkability.cells.size);
    std::swap(first.walkability.cells.data, second.walkability.cells.data);
    std::swap(first.scene_path, second.scene_path);
    std::swap(first.terrain_path, second.terrain_path);
    std::swap(first.mesh_path, second.mesh_path);
    std::swap(first.walkability_path, second.walkability_path);
}

static inline bool flat_scene_catalog_build(
    scene_catalog& out,
    const motion_pack_manifest& manifest,
    char* error,
    const int capacity)
{
    if (!manifest.flat_lmm_bundle ||
        !g1_manifest_rate_compatible(manifest.output_fps) ||
        manifest.sources.size() != 13)
        return scene_error(error, capacity,
                           "flat scene catalog requires a validated flat bundle");
    for (size_t index = 0; index < manifest.sources.size(); ++index) {
        if (manifest.sources[index].terrain_id != "flat")
            return scene_error(error, capacity,
                               "flat scene catalog range is not flat");
    }
    scene_catalog candidate;
    candidate.default_scene_id = "g1-lmm-flat";
    candidate.coordinate_signature =
        "holden-y-up-right-handed-forward-plus-z";
    candidate.surface_signature = "authenticated-flat-zero/v1";
    candidate.ids.push_back(candidate.default_scene_id);
    scene_descriptor descriptor;
    descriptor.id = candidate.default_scene_id;
    candidate.scenes.push_back(descriptor);
    out = std::move(candidate);
    return true;
}

static inline bool flat_scene_pack_build(
    scene_pack& out,
    const motion_pack_manifest& manifest,
    char* error,
    const int capacity)
{
    if (!manifest.flat_lmm_bundle || manifest.sources.size() != 13)
        return scene_error(error, capacity,
                           "flat scene requires a validated flat bundle");
    for (size_t index = 0; index < manifest.sources.size(); ++index) {
        if (manifest.sources[index].terrain_id != "flat")
            return scene_error(error, capacity,
                               "flat scene range is not flat");
    }
    scene_pack candidate;
    candidate.metadata.id = "g1-lmm-flat";
    candidate.metadata.label = "G1 LMM authenticated flat";
    candidate.metadata.coordinate_signature =
        "holden-y-up-right-handed-forward-plus-z";
    candidate.metadata.surface_signature = "authenticated-flat-zero/v1";
    candidate.metadata.playable_bounds = {-32.0f, -32.0f, 32.0f, 32.0f};
    candidate.metadata.lookahead_bounds =
        candidate.metadata.playable_bounds;
    candidate.metadata.spawn_position = vec3();
    candidate.metadata.spawn_yaw = 0.0f;
    candidate.terrain.version = 2;
    candidate.terrain.nx = 257;
    candidate.terrain.nz = 257;
    candidate.terrain.origin_x = -32.0f;
    candidate.terrain.origin_z = -32.0f;
    candidate.terrain.cell_size = 0.25f;
    candidate.terrain.exterior_height = 0.0f;
    candidate.terrain.heights.resize(
        candidate.terrain.nx * candidate.terrain.nz);
    candidate.terrain.heights.zero();
    candidate.walkability.nx = candidate.terrain.nx;
    candidate.walkability.nz = candidate.terrain.nz;
    candidate.walkability.cells.resize(
        candidate.walkability.nx * candidate.walkability.nz);
    candidate.walkability.cells.set(1);
    if (!terrain_heightfield_is_queryable(candidate.terrain) ||
        !walkability_grid_matches_heightfield(
            candidate.walkability, candidate.terrain))
        return scene_error(error, capacity,
                           "flat scene construction failed its query contract");
    scene_pack_swap(out, candidate);
    return true;
}

static inline bool scene_pack_load(
    scene_pack& out,const char* root,const motion_pack_manifest& manifest,
    const scene_catalog& catalog,int index,char* error,int capacity)
{
    if(index<0||index>=static_cast<int>(catalog.ids.size())||
       catalog.scenes.size()!=catalog.ids.size())
        return scene_error(error,capacity,
            "scene index %d is out of range",index);
    const scene_descriptor& descriptor=
        catalog.scenes[static_cast<size_t>(index)];
    const std::string& id=catalog.ids[static_cast<size_t>(index)];
    scene_pack candidate;
    if(descriptor.id!=id||
       !scene_join(candidate.scene_path,root,descriptor.path,error,capacity))
        return false;
    json_value document;
    if(!scene_json_load_verified(document,candidate.scene_path.c_str(),
            descriptor.sha256,error,capacity)||
       !scene_metadata_parse(candidate.metadata,document,descriptor.id.c_str(),
            manifest,candidate.scene_path.c_str(),error,capacity))return false;
    const std::string prefix="scenes/"+descriptor.id+"/";
    if(!scene_join(candidate.terrain_path,root,
            prefix+candidate.metadata.heightfield.path,error,capacity)||
       !scene_join(candidate.mesh_path,root,
            prefix+candidate.metadata.mesh.path,error,capacity)||
       !scene_join(candidate.walkability_path,root,
            prefix+candidate.metadata.walkability.path,error,capacity)||
       !scene_verify_sha(candidate.terrain_path,
            candidate.metadata.heightfield.sha256,error,capacity)||
       !scene_verify_sha(candidate.mesh_path,
            candidate.metadata.mesh.sha256,error,capacity)||
       !scene_verify_sha(candidate.walkability_path,
            candidate.metadata.walkability.sha256,error,capacity)||
       !heightfield_load(candidate.terrain,candidate.terrain_path.c_str(),
            error,capacity))return false;
    if(candidate.terrain.version!=2)
        return scene_error(error,capacity,
            "%s: published scene '%s' requires G1HF version 2, got %u",
            candidate.terrain_path.c_str(),id.c_str(),
            static_cast<unsigned>(candidate.terrain.version));
    if(!walkability_load(candidate.walkability,
            candidate.walkability_path.c_str(),candidate.terrain,
            error,capacity)||
       !scene_candidate_validate(candidate,error,capacity))return false;
    scene_pack_swap(out,candidate);
    return true;
}

#undef SCENE_RUNTIME_NOINLINE
