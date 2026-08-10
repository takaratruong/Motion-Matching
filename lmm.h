#pragma once

#include "common.h"
#include "vec.h"
#include "quat.h"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "array.h"
#include "nnet.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include "scene_runtime.h"

#include <cfloat>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static const int G1_LMM_FeatureCount = 31;
static const int G1_LMM_LatentCount = 32;
static const int G1_LMM_BoneCount = 31;
static const int G1_LMM_ContactCount = 2;
static constexpr const char* G1_LMMAcceptedModelScope =
    "single-clip-overfit-canary";

static inline bool g1_lmm_dimensions_valid(
    const int features,
    const int latent,
    const int bones,
    const int contacts)
{
    return features == G1_LMM_FeatureCount &&
           latent == G1_LMM_LatentCount &&
           bones == G1_LMM_BoneCount &&
           contacts == G1_LMM_ContactCount;
}

static inline float projector_cost_normalized(
    const slice1d<float> query_normalized,
    const slice1d<float> projected_normalized)
{
    if (query_normalized.size != G1_LMM_FeatureCount ||
        projected_normalized.size != G1_LMM_FeatureCount ||
        query_normalized.data == NULL || projected_normalized.data == NULL)
    {
        return FLT_MAX;
    }
    float squared = 0.0f;
    for (int index = 0; index < G1_LMM_FeatureCount; ++index)
    {
        if (!terrain_float_is_finite(query_normalized(index)) ||
            !terrain_float_is_finite(projected_normalized(index)))
        {
            return FLT_MAX;
        }
        squared += squaref(
            query_normalized(index) - projected_normalized(index));
    }
    return terrain_float_is_finite(squared) ? std::sqrt(squared) : FLT_MAX;
}

struct g1_lmm_model_bundle
{
    nnet decompressor;
    nnet stepper;
    nnet projector;
    nnet_evaluation decompressor_evaluation;
    nnet_evaluation stepper_evaluation;
    nnet_evaluation projector_evaluation;
    array2d<float> latent;
    std::string data_manifest_sha256;
    std::string model_scope;
    bool authenticated = false;
    int evaluation_allocation_count = 0;
};

template<typename T>
static inline void g1_lmm_swap_array1d(array1d<T>& first, array1d<T>& second)
{
    std::swap(first.size, second.size);
    std::swap(first.data, second.data);
}

template<typename T>
static inline void g1_lmm_swap_array2d(array2d<T>& first, array2d<T>& second)
{
    std::swap(first.rows, second.rows);
    std::swap(first.cols, second.cols);
    std::swap(first.data, second.data);
}

static inline void g1_lmm_swap_nnet(nnet& first, nnet& second)
{
    g1_lmm_swap_array1d(first.input_mean, second.input_mean);
    g1_lmm_swap_array1d(first.input_std, second.input_std);
    g1_lmm_swap_array1d(first.output_mean, second.output_mean);
    g1_lmm_swap_array1d(first.output_std, second.output_std);
    first.weights.swap(second.weights);
    first.biases.swap(second.biases);
}

static inline void g1_lmm_model_bundle_swap(
    g1_lmm_model_bundle& first,
    g1_lmm_model_bundle& second)
{
    g1_lmm_swap_nnet(first.decompressor, second.decompressor);
    g1_lmm_swap_nnet(first.stepper, second.stepper);
    g1_lmm_swap_nnet(first.projector, second.projector);
    first.decompressor_evaluation.layers.swap(
        second.decompressor_evaluation.layers);
    first.stepper_evaluation.layers.swap(second.stepper_evaluation.layers);
    first.projector_evaluation.layers.swap(
        second.projector_evaluation.layers);
    g1_lmm_swap_array2d(first.latent, second.latent);
    first.data_manifest_sha256.swap(second.data_manifest_sha256);
    first.model_scope.swap(second.model_scope);
    std::swap(first.authenticated, second.authenticated);
    std::swap(
        first.evaluation_allocation_count,
        second.evaluation_allocation_count);
}

struct g1_lmm_binary_reader
{
    const unsigned char* data = NULL;
    size_t size = 0;
    size_t cursor = 0;
};

static inline bool g1_lmm_read_u32(
    g1_lmm_binary_reader& reader,
    uint32_t& out)
{
    if (reader.cursor > reader.size || reader.size - reader.cursor < 4)
        return false;
    const unsigned char* bytes = reader.data + reader.cursor;
    out = static_cast<uint32_t>(bytes[0]) |
          (static_cast<uint32_t>(bytes[1]) << 8) |
          (static_cast<uint32_t>(bytes[2]) << 16) |
          (static_cast<uint32_t>(bytes[3]) << 24);
    reader.cursor += 4;
    return true;
}

static inline bool g1_lmm_read_float(
    g1_lmm_binary_reader& reader,
    float& out)
{
    uint32_t bits = 0;
    if (!g1_lmm_read_u32(reader, bits)) return false;
    std::memcpy(&out, &bits, sizeof(out));
    return terrain_float_is_finite(out);
}

static inline bool g1_lmm_read_file(
    std::vector<unsigned char>& out,
    const std::string& path,
    const int expected_size,
    char* error,
    const int capacity)
{
    if (expected_size <= 0 || expected_size > 64 * 1024 * 1024)
        return scene_error(
            error, capacity, "%s: runtime artifact size is invalid",
            path.c_str());
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == NULL)
        return scene_error(error, capacity, "%s: cannot open", path.c_str());
    std::vector<unsigned char> candidate(
        static_cast<size_t>(expected_size));
    const bool failed = std::fread(
        candidate.data(), 1, candidate.size(), file) != candidate.size();
    const int trailing = failed ? EOF : std::fgetc(file);
    const bool close_failed = std::fclose(file) != 0;
    if (failed || trailing != EOF || close_failed)
        return scene_error(error, capacity, "%s: cannot read exactly", path.c_str());
    out.swap(candidate);
    return true;
}

static inline bool g1_lmm_read_vector(
    array1d<float>& out,
    g1_lmm_binary_reader& reader,
    const int expected,
    const bool positive,
    const char* label,
    char* error,
    const int capacity)
{
    uint32_t count = 0;
    if (!g1_lmm_read_u32(reader, count) ||
        count != static_cast<uint32_t>(expected))
        return scene_error(
            error, capacity, "%s length must be %d", label, expected);
    array1d<float> candidate(expected);
    for (int index = 0; index < expected; ++index)
    {
        if (!g1_lmm_read_float(reader, candidate(index)) ||
            (positive && candidate(index) <= 0.0f))
            return scene_error(
                error, capacity, "%s contains an invalid value", label);
    }
    g1_lmm_swap_array1d(out, candidate);
    return true;
}

static inline bool g1_lmm_network_load_checked(
    nnet& out,
    const std::string& path,
    const int expected_size,
    const int input_count,
    const int output_count,
    const int (*layers)[2],
    const int layer_count,
    char* error,
    const int capacity)
{
    std::vector<unsigned char> bytes;
    if (!g1_lmm_read_file(
            bytes, path, expected_size, error, capacity)) return false;
    g1_lmm_binary_reader reader;
    reader.data = bytes.data();
    reader.size = bytes.size();
    nnet candidate;
    if (!g1_lmm_read_vector(
            candidate.input_mean, reader, input_count, false,
            "network input_mean", error, capacity) ||
        !g1_lmm_read_vector(
            candidate.input_std, reader, input_count, true,
            "network input_std", error, capacity) ||
        !g1_lmm_read_vector(
            candidate.output_mean, reader, output_count, false,
            "network output_mean", error, capacity) ||
        !g1_lmm_read_vector(
            candidate.output_std, reader, output_count, true,
            "network output_std", error, capacity))
        return false;

    uint32_t encoded_layers = 0;
    if (!g1_lmm_read_u32(reader, encoded_layers) ||
        encoded_layers != static_cast<uint32_t>(layer_count))
        return scene_error(
            error, capacity, "%s: network layer count mismatch", path.c_str());
    candidate.weights.resize(static_cast<size_t>(layer_count));
    candidate.biases.resize(static_cast<size_t>(layer_count));
    for (int layer = 0; layer < layer_count; ++layer)
    {
        uint32_t rows = 0;
        uint32_t cols = 0;
        if (!g1_lmm_read_u32(reader, rows) ||
            !g1_lmm_read_u32(reader, cols) ||
            rows != static_cast<uint32_t>(layers[layer][0]) ||
            cols != static_cast<uint32_t>(layers[layer][1]))
            return scene_error(
                error, capacity, "%s: network layer %d shape mismatch",
                path.c_str(), layer);
        candidate.weights[static_cast<size_t>(layer)].resize(
            static_cast<int>(rows), static_cast<int>(cols));
        for (uint64_t index = 0;
             index < static_cast<uint64_t>(rows) * cols;
             ++index)
        {
            float value = 0.0f;
            if (!g1_lmm_read_float(reader, value))
                return scene_error(
                    error, capacity, "%s: network weight is invalid",
                    path.c_str());
            candidate.weights[static_cast<size_t>(layer)].data[index] = value;
        }
        uint32_t biases = 0;
        if (!g1_lmm_read_u32(reader, biases) || biases != cols)
            return scene_error(
                error, capacity, "%s: network bias shape mismatch",
                path.c_str());
        candidate.biases[static_cast<size_t>(layer)].resize(
            static_cast<int>(biases));
        for (uint32_t index = 0; index < biases; ++index)
        {
            if (!g1_lmm_read_float(
                    reader,
                    candidate.biases[static_cast<size_t>(layer)](
                        static_cast<int>(index))))
                return scene_error(
                    error, capacity, "%s: network bias is invalid",
                    path.c_str());
        }
    }
    if (reader.cursor != reader.size)
        return scene_error(
            error, capacity, "%s: network has trailing data", path.c_str());
    g1_lmm_swap_nnet(out, candidate);
    return true;
}

static inline bool g1_lmm_latent_load_checked(
    array2d<float>& out,
    const std::string& path,
    const int expected_size,
    const int expected_rows,
    char* error,
    const int capacity)
{
    std::vector<unsigned char> bytes;
    if (!g1_lmm_read_file(
            bytes, path, expected_size, error, capacity)) return false;
    g1_lmm_binary_reader reader;
    reader.data = bytes.data();
    reader.size = bytes.size();
    uint32_t rows = 0;
    uint32_t cols = 0;
    if (!g1_lmm_read_u32(reader, rows) || !g1_lmm_read_u32(reader, cols) ||
        rows != static_cast<uint32_t>(expected_rows) ||
        cols != static_cast<uint32_t>(G1_LMM_LatentCount))
        return scene_error(
            error, capacity, "%s: latent shape mismatch", path.c_str());
    array2d<float> candidate(static_cast<int>(rows), static_cast<int>(cols));
    for (uint64_t index = 0;
         index < static_cast<uint64_t>(rows) * cols;
         ++index)
    {
        if (!g1_lmm_read_float(reader, candidate.data[index]))
            return scene_error(
                error, capacity, "%s: latent value is invalid", path.c_str());
    }
    if (reader.cursor != reader.size)
        return scene_error(
            error, capacity, "%s: latent has trailing data", path.c_str());
    g1_lmm_swap_array2d(out, candidate);
    return true;
}

static inline bool g1_lmm_model_load_and_verify(
    g1_lmm_model_bundle& out,
    const char* model_root,
    const char* data_root,
    char* error,
    const int capacity)
{
    motion_pack_manifest data_manifest;
    if (!motion_manifest_load_and_verify(
            data_manifest, data_root, error, capacity))
        return false;
    if (!data_manifest.flat_lmm_bundle ||
        data_manifest.database_frames <= 0 ||
        data_manifest.database.path != "database.bin" ||
        data_manifest.matching_features.path != "features.bin" ||
        !scene_sha_is_valid(data_manifest.database.sha256) ||
        !scene_sha_is_valid(data_manifest.matching_features.sha256))
        return scene_error(
            error, capacity, "LMM requires an authenticated flat data bundle");

    std::string data_manifest_path;
    std::string data_database_path;
    std::string data_features_path;
    if (!scene_join(
            data_manifest_path, data_root, "manifest.json", error, capacity) ||
        !scene_join(
            data_database_path, data_root, data_manifest.database.path,
            error, capacity) ||
        !scene_join(
            data_features_path, data_root,
            data_manifest.matching_features.path, error, capacity) ||
        !scene_verify_size(
            data_database_path, data_manifest.database.size_bytes,
            error, capacity) ||
        !scene_verify_size(
            data_features_path, data_manifest.matching_features.size_bytes,
            error, capacity) ||
        !scene_verify_sha(
            data_database_path, data_manifest.database.sha256,
            error, capacity) ||
        !scene_verify_sha(
            data_features_path, data_manifest.matching_features.sha256,
            error, capacity))
        return false;

    std::string observed_data_manifest_sha;
    if (!sha256_file_hex(
            observed_data_manifest_sha,
            data_manifest_path.c_str(), error, capacity)) return false;

    std::string model_manifest_path;
    if (!scene_join(
            model_manifest_path, model_root, "manifest.json", error, capacity))
        return false;
    std::string observed_model_manifest_sha;
    if (!sha256_file_hex(
            observed_model_manifest_sha,
            model_manifest_path.c_str(), error, capacity)) return false;
    json_value document;
    if (!scene_json_load_verified(
            document, model_manifest_path.c_str(),
            observed_model_manifest_sha, error, capacity))
        return false;
    if (json_member(document, "model_scope") == NULL)
        return scene_error(
            error, capacity, "G1 LMM model scope is required");
    if (!scene_exact_keys(
            document,
            {"artifacts","data_artifacts","data_manifest_schema",
             "data_manifest_sha256","dimensions","model_scope",
             "output_fps","schema","status"},
            "G1 LMM model manifest", error, capacity))
        return false;

    std::string schema;
    std::string status;
    std::string data_schema;
    std::string data_sha;
    std::string model_scope;
    float output_fps = 0.0f;
    if (!scene_member_string(
            schema, document, "schema", "G1 LMM model manifest",
            error, capacity) || schema != "g1-lmm-model/v1" ||
        !scene_member_string(
            status, document, "status", "G1 LMM model manifest",
            error, capacity) || status != "accepted" ||
        !scene_member_string(
            data_schema, document, "data_manifest_schema",
            "G1 LMM model manifest", error, capacity) ||
        data_schema != G1_LMMFlatDataSchema ||
        !scene_member_string(
            data_sha, document, "data_manifest_sha256",
            "G1 LMM model manifest", error, capacity) ||
        !scene_sha_is_valid(data_sha) ||
        data_sha != observed_data_manifest_sha ||
        !scene_member_float(
            output_fps, document, "output_fps", "G1 LMM model manifest",
            error, capacity) || !g1_manifest_rate_compatible(output_fps))
        return scene_error(
            error, capacity,
            "G1 LMM model schema/status/rate/data binding is incompatible");
    if (!scene_member_string(
            model_scope, document, "model_scope", "G1 LMM model manifest",
            error, capacity))
        return false;
    if (model_scope != G1_LMMAcceptedModelScope)
        return scene_error(
            error, capacity, "G1 LMM model scope is incompatible");

    const json_value* dimensions = json_member(document, "dimensions");
    int features = 0;
    int latent = 0;
    int bones = 0;
    int contacts = 0;
    if (dimensions == NULL ||
        !scene_exact_keys(
            *dimensions, {"bones","contacts","features","latent"},
            "G1 LMM dimensions", error, capacity) ||
        !scene_member_int(
            features, *dimensions, "features", "G1 LMM dimensions",
            error, capacity) ||
        !scene_member_int(
            latent, *dimensions, "latent", "G1 LMM dimensions",
            error, capacity) ||
        !scene_member_int(
            bones, *dimensions, "bones", "G1 LMM dimensions",
            error, capacity) ||
        !scene_member_int(
            contacts, *dimensions, "contacts", "G1 LMM dimensions",
            error, capacity) ||
        !g1_lmm_dimensions_valid(features, latent, bones, contacts))
        return scene_error(
            error, capacity, "G1 LMM dimensions are incompatible");

    const json_value* data_artifacts = json_member(document, "data_artifacts");
    std::string database_sha;
    std::string features_sha;
    if (data_artifacts == NULL ||
        !scene_exact_keys(
            *data_artifacts, {"database.bin","features.bin"},
            "G1 LMM data artifacts", error, capacity) ||
        !scene_member_string(
            database_sha, *data_artifacts, "database.bin",
            "G1 LMM data artifacts", error, capacity) ||
        !scene_member_string(
            features_sha, *data_artifacts, "features.bin",
            "G1 LMM data artifacts", error, capacity) ||
        database_sha != data_manifest.database.sha256 ||
        features_sha != data_manifest.matching_features.sha256)
        return scene_error(
            error, capacity, "G1 LMM data artifact binding mismatch");

    const json_value* artifacts = json_member(document, "artifacts");
    const char* names[] = {
        "latent.bin", "decompressor.bin", "stepper.bin", "projector.bin",
        "training.json", "evaluation.json"};
    if (artifacts == NULL ||
        !scene_exact_keys(
            *artifacts,
            {"latent.bin","decompressor.bin","stepper.bin","projector.bin",
             "training.json","evaluation.json"},
            "G1 LMM artifacts", error, capacity))
        return false;
    artifact_reference descriptors[6];
    std::string paths[6];
    for (int index = 0; index < 6; ++index)
    {
        if (!flat_artifact_reference_parse(
                descriptors[index], *artifacts, names[index],
                error, capacity) ||
            !scene_join(
                paths[index], model_root, descriptors[index].path,
                error, capacity) ||
            !scene_verify_size(
                paths[index], descriptors[index].size_bytes,
                error, capacity) ||
            !scene_verify_sha(
                paths[index], descriptors[index].sha256,
                error, capacity))
            return false;
    }

    static const int decompressor_layers[][2] = {
        {63, 512}, {512, 458}};
    static const int stepper_layers[][2] = {
        {63, 512}, {512, 512}, {512, 63}};
    static const int projector_layers[][2] = {
        {31, 512}, {512, 512}, {512, 512}, {512, 512}, {512, 63}};
    g1_lmm_model_bundle candidate;
    if (!g1_lmm_latent_load_checked(
            candidate.latent, paths[0], descriptors[0].size_bytes,
            data_manifest.database_frames, error, capacity) ||
        !g1_lmm_network_load_checked(
            candidate.decompressor, paths[1], descriptors[1].size_bytes,
            63, 458, decompressor_layers, 2, error, capacity) ||
        !g1_lmm_network_load_checked(
            candidate.stepper, paths[2], descriptors[2].size_bytes,
            63, 63, stepper_layers, 3, error, capacity) ||
        !g1_lmm_network_load_checked(
            candidate.projector, paths[3], descriptors[3].size_bytes,
            31, 63, projector_layers, 5, error, capacity))
        return false;

    candidate.decompressor_evaluation.resize(candidate.decompressor);
    candidate.stepper_evaluation.resize(candidate.stepper);
    candidate.projector_evaluation.resize(candidate.projector);
    candidate.evaluation_allocation_count = 3;
    candidate.data_manifest_sha256 = data_sha;
    candidate.model_scope = model_scope;
    candidate.authenticated = true;
    g1_lmm_model_bundle_swap(out, candidate);
    return true;
}

// This function uses the decompressor network
// to generate the pose of the character. It 
// requires as input the feature values and latent 
// values as well as a current root position and 
// rotation.
void decompressor_evaluate(
    slice1d<vec3> bone_positions,
    slice1d<vec3> bone_velocities,
    slice1d<quat> bone_rotations,
    slice1d<vec3> bone_angular_velocities,
    slice1d<bool> bone_contacts,
    nnet_evaluation& evaluation,
    const slice1d<float> features,
    const slice1d<float> latent,
    const vec3 root_position,
    const quat root_rotation,
    const nnet& nn,
    const float dt = 1.0f / 60.0f)
{
    slice1d<float> input_layer = evaluation.layers.front();
    slice1d<float> output_layer = evaluation.layers.back();
  
    // First copy feature values and latent variables to 
    // the input layer of the network
  
    for (int i = 0; i < features.size; i++)
    {
        input_layer(i) = features(i);
    }
    
    for (int i = 0; i < latent.size; i++)
    {
        input_layer(features.size + i) = latent(i);
    }
    
    // Evaluate network
    nnet_evaluate(evaluation, nn);
    
    // Extract bone positions
    int offset = 0;
    for (int i = 0; i < bone_positions.size - 1; i++)
    {
        bone_positions(i + 1) = vec3(
            output_layer(offset+i*3+0),
            output_layer(offset+i*3+1),
            output_layer(offset+i*3+2));
    }
    offset += (bone_positions.size - 1) * 3;
    
    // Extract bone rotations, convert from 2-axis representation
    for (int i = 0; i < bone_rotations.size - 1; i++)
    {   
        bone_rotations(i + 1) = quat_from_xform_xy(
            vec3(output_layer(offset+i*6+0),
                 output_layer(offset+i*6+2),
                 output_layer(offset+i*6+4)),
            vec3(output_layer(offset+i*6+1),
                 output_layer(offset+i*6+3),
                 output_layer(offset+i*6+5)));
    }
    offset += (bone_rotations.size - 1) * 6;
    
    // Extract bone velocities
    for (int i = 0; i < bone_velocities.size - 1; i++)
    {
        bone_velocities(i + 1) = vec3(
            output_layer(offset+i*3+0),
            output_layer(offset+i*3+1),
            output_layer(offset+i*3+2));
    }
    offset += (bone_velocities.size - 1) * 3;
    
    // Extract bone angular velocities
    for (int i = 0; i < bone_angular_velocities.size - 1; i++)
    {
        bone_angular_velocities(i + 1) = vec3(
            output_layer(offset+i*3+0),
            output_layer(offset+i*3+1),
            output_layer(offset+i*3+2));
    }
    offset += (bone_angular_velocities.size - 1) * 3;
    
    // Extract root velocities and put in world space
    
    vec3 root_velocity = quat_mul_vec3(root_rotation, vec3(
        output_layer(offset+0),
        output_layer(offset+1),
        output_layer(offset+2)));
        
    vec3 root_angular_velocity = quat_mul_vec3(root_rotation, vec3(
        output_layer(offset+3),
        output_layer(offset+4),
        output_layer(offset+5)));
    
    offset += 6;

    // Find new root position/rotation/velocities etc.
    
    bone_positions(0) = dt * root_velocity + root_position;
    bone_rotations(0) = quat_mul(quat_from_scaled_angle_axis(root_angular_velocity * dt), root_rotation);
    bone_velocities(0) = root_velocity;
    bone_angular_velocities(0) = root_angular_velocity;    
    
    // Extract bone contacts
    if (bone_contacts.data != nullptr)
    {
        bone_contacts(0) = output_layer(offset+0) > 0.5f;
        bone_contacts(1) = output_layer(offset+1) > 0.5f;
    }

    offset += 2;
    
    // Check we got everything!
    assert(offset == nn.output_mean.size);
}

// This function updates the feature and latent values
// using the stepper network and a given dt.
void stepper_evaluate(
    slice1d<float> features,
    slice1d<float> latent,
    nnet_evaluation& evaluation,
    const nnet& nn,
    const float dt = 1.0f / 60.0f)
{
    slice1d<float> input_layer = evaluation.layers.front();
    slice1d<float> output_layer = evaluation.layers.back();
  
    // Copy features and latents to input
  
    for (int i = 0; i < features.size; i++)
    {
        input_layer(i) = features(i);
    }
    
    for (int i = 0; i < latent.size; i++)
    {
        input_layer(features.size + i) = latent(i);
    }
    
    // Evaluate network
    
    nnet_evaluate(evaluation, nn);
    
    // Update features and latents using result
    
    for (int i = 0; i < features.size; i++)
    {
        features(i) += dt * output_layer(i);
    }
    
    for (int i = 0; i < latent.size; i++)
    {
        latent(i) += dt * output_layer(features.size + i);
    }
}

// Projector input and output features are both normalized by the Task 1
// feature offset/scale contract before training. Keeping this entry point
// normalized prevents a raw-metre versus normalized-unit distance regression.
static inline bool projector_evaluate_normalized(
    bool& transition,
    float& best_cost,
    slice1d<float> projected_features,
    slice1d<float> projected_latent,
    nnet_evaluation& evaluation,
    const slice1d<float> query_normalized,
    const slice1d<float> current_features,
    const slice1d<float> current_latent,
    const nnet& nn,
    char* error,
    const int capacity,
    const float transition_cost = 0.0f)
{
    if (query_normalized.size != G1_LMM_FeatureCount ||
        projected_features.size != G1_LMM_FeatureCount ||
        current_features.size != G1_LMM_FeatureCount ||
        projected_latent.size != G1_LMM_LatentCount ||
        current_latent.size != G1_LMM_LatentCount ||
        evaluation.layers.empty() ||
        evaluation.layers.front().size != G1_LMM_FeatureCount ||
        evaluation.layers.back().size !=
            G1_LMM_FeatureCount + G1_LMM_LatentCount ||
        !terrain_float_is_finite(transition_cost) || transition_cost < 0.0f)
        return scene_error(
            error, capacity, "normalized projector shapes are invalid");

    nnet_evaluation candidate_evaluation;
    candidate_evaluation.resize(nn);
    slice1d<float> input_layer = candidate_evaluation.layers.front();
    for (int index = 0; index < G1_LMM_FeatureCount; ++index)
    {
        if (!terrain_float_is_finite(query_normalized(index)) ||
            !terrain_float_is_finite(current_features(index)))
            return scene_error(
                error, capacity, "normalized projector input is non-finite");
        input_layer(index) = query_normalized(index);
    }
    for (int index = 0; index < G1_LMM_LatentCount; ++index)
        if (!terrain_float_is_finite(current_latent(index)))
            return scene_error(
                error, capacity, "current LMM latent is non-finite");

    nnet_evaluate(candidate_evaluation, nn);
    const slice1d<float> output_layer = candidate_evaluation.layers.back();
    array1d<float> candidate_features(G1_LMM_FeatureCount);
    array1d<float> candidate_latent(G1_LMM_LatentCount);
    for (int index = 0; index < G1_LMM_FeatureCount; ++index)
    {
        if (!terrain_float_is_finite(output_layer(index)))
            return scene_error(
                error, capacity, "normalized projector output is non-finite");
        candidate_features(index) = output_layer(index);
    }
    for (int index = 0; index < G1_LMM_LatentCount; ++index)
    {
        const float value = output_layer(G1_LMM_FeatureCount + index);
        if (!terrain_float_is_finite(value))
            return scene_error(
                error, capacity, "projected LMM latent is non-finite");
        candidate_latent(index) = value;
    }

    float candidate_cost = projector_cost_normalized(
        query_normalized, candidate_features);
    if (!terrain_float_is_finite(candidate_cost) || candidate_cost == FLT_MAX)
        return scene_error(
            error, capacity, "normalized projector cost is invalid");

    float transition_squared = 0.0f;
    for (int index = 0; index < G1_LMM_FeatureCount; ++index)
        transition_squared += squaref(
            current_features(index) - candidate_features(index));
    if (!terrain_float_is_finite(transition_squared))
        return scene_error(
            error, capacity, "normalized projector transition is invalid");

    const bool candidate_transition =
        transition_squared > squaref(transition_cost);
    if (candidate_transition)
    {
        candidate_cost += transition_cost;
    }
    else
    {
        for (int index = 0; index < G1_LMM_FeatureCount; ++index)
            candidate_features(index) = current_features(index);
        for (int index = 0; index < G1_LMM_LatentCount; ++index)
            candidate_latent(index) = current_latent(index);
        candidate_cost = projector_cost_normalized(
            query_normalized, current_features);
    }
    if (!terrain_float_is_finite(candidate_cost) || candidate_cost == FLT_MAX)
        return scene_error(
            error, capacity, "normalized projector cost is invalid");

    evaluation.layers.swap(candidate_evaluation.layers);
    for (int index = 0; index < G1_LMM_FeatureCount; ++index)
        projected_features(index) = candidate_features(index);
    for (int index = 0; index < G1_LMM_LatentCount; ++index)
        projected_latent(index) = candidate_latent(index);
    transition = candidate_transition;
    best_cost = candidate_cost;
    return true;
}

// This function projects a set of feature values onto
// the nearest in the trained database, also outputting the 
// associated latent values. It also produces the matching 
// cost using the distance of the projection, and detects 
// transitions for a given transition cost by measuring the 
// distance between the projected result and the current
// feature values
void projector_evaluate(
    bool& transition,
    float& best_cost,
    slice1d<float> proj_features,
    slice1d<float> proj_latent,
    nnet_evaluation& evaluation,
    const slice1d<float> query,
    const slice1d<float> features_offset,
    const slice1d<float> features_scale,
    const slice1d<float> curr_features,
    const nnet& nn,
    const float transition_cost = 0.0f)
{
    array1d<float> query_normalized(query.size);
    array1d<float> current_latent(proj_latent.size);
    current_latent.zero();
    for (int index = 0; index < query.size; ++index)
        query_normalized(index) =
            (query(index) - features_offset(index)) / features_scale(index);
    char ignored[1] = {};
    if (!projector_evaluate_normalized(
            transition,
            best_cost,
            proj_features,
            proj_latent,
            evaluation,
            query_normalized,
            curr_features,
            current_latent,
            nn,
            ignored,
            static_cast<int>(sizeof(ignored)),
            transition_cost))
    {
        transition = false;
        best_cost = FLT_MAX;
    }
}
