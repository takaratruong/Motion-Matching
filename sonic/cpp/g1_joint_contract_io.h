#pragma once

#include "json_runtime.h"
#include "sonic/cpp/g1_joint_projection.h"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

struct sonic_joint_contract_metadata
{
    std::string source_mjcf_sha256;
    std::string target_order_source_sha256;
    std::vector<std::string> source_joint_names;
    std::vector<std::string> target_joint_names;
};

static inline bool sonic_joint_contract_exact_keys(
    const json_value& value,
    std::initializer_list<const char*> expected,
    char* error,
    int capacity)
{
    if (value.kind != json_object ||
        value.object_value.size() != expected.size()) {
        return sonic_projection_error(
            error, capacity, "joint contract object keys are invalid");
    }
    for (const char* key : expected) {
        if (json_member(value, key) == nullptr) {
            return sonic_projection_error(
                error, capacity, "joint contract object key is missing");
        }
    }
    return true;
}

static inline bool sonic_joint_contract_string(
    std::string& output,
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    const json_value* value = json_member(object, key);
    if (value == nullptr || value->kind != json_string ||
        value->string_value.empty()) {
        return sonic_projection_error(
            error, capacity, "joint contract string is invalid");
    }
    output = value->string_value;
    return true;
}

static inline bool sonic_joint_contract_integer(
    int& output,
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    const json_value* value = json_member(object, key);
    if (value == nullptr || value->kind != json_number ||
        value->number_value <
            static_cast<double>(std::numeric_limits<int>::min()) ||
        value->number_value >
            static_cast<double>(std::numeric_limits<int>::max()) ||
        std::floor(value->number_value) != value->number_value) {
        return sonic_projection_error(
            error, capacity, "joint contract integer is invalid");
    }
    output = static_cast<int>(value->number_value);
    return true;
}

static inline bool sonic_joint_contract_float_value(
    float& output,
    const json_value& value,
    char* error,
    int capacity)
{
    if (value.kind != json_number || !std::isfinite(value.number_value) ||
        value.number_value <
            -static_cast<double>(std::numeric_limits<float>::max()) ||
        value.number_value >
            static_cast<double>(std::numeric_limits<float>::max())) {
        return sonic_projection_error(
            error, capacity, "joint contract float is invalid");
    }
    output = static_cast<float>(value.number_value);
    if (!std::isfinite(output)) {
        return sonic_projection_error(
            error, capacity, "joint contract float is invalid");
    }
    return true;
}

static inline bool sonic_joint_contract_float_member(
    float& output,
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    const json_value* value = json_member(object, key);
    return value != nullptr && sonic_joint_contract_float_value(
        output, *value, error, capacity);
}

static inline bool sonic_joint_contract_double_member(
    double& output,
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    const json_value* value = json_member(object, key);
    if (value == nullptr || value->kind != json_number ||
        !std::isfinite(value->number_value)) {
        return sonic_projection_error(
            error, capacity, "joint contract double is invalid");
    }
    output = value->number_value;
    return true;
}

static inline bool sonic_joint_contract_vec3_member(
    vec3& output,
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    const json_value* value = json_member(object, key);
    if (value == nullptr || value->kind != json_array ||
        value->array_value.size() != 3) {
        return sonic_projection_error(
            error, capacity, "joint contract vec3 is invalid");
    }
    return sonic_joint_contract_float_value(
               output.x, value->array_value[0], error, capacity) &&
           sonic_joint_contract_float_value(
               output.y, value->array_value[1], error, capacity) &&
           sonic_joint_contract_float_value(
               output.z, value->array_value[2], error, capacity);
}

static inline bool sonic_joint_contract_quat_member(
    quat& output,
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    const json_value* value = json_member(object, key);
    if (value == nullptr || value->kind != json_array ||
        value->array_value.size() != 4) {
        return sonic_projection_error(
            error, capacity, "joint contract quaternion is invalid");
    }
    return sonic_joint_contract_float_value(
               output.w, value->array_value[0], error, capacity) &&
           sonic_joint_contract_float_value(
               output.x, value->array_value[1], error, capacity) &&
           sonic_joint_contract_float_value(
               output.y, value->array_value[2], error, capacity) &&
           sonic_joint_contract_float_value(
               output.z, value->array_value[3], error, capacity);
}

static inline int sonic_joint_contract_bone_index(const std::string& name)
{
    static const char* const names[G1_BoneCount] = {
        "Simulation", "Hips", "LeftHipPitch", "LeftHipRoll",
        "LeftHipYaw", "LeftKnee", "LeftAnkle", "LeftToe",
        "RightHipPitch", "RightHipRoll", "RightHipYaw", "RightKnee",
        "RightAnkle", "RightToe", "Spine", "Spine1", "Spine2",
        "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw",
        "LeftElbow", "LeftWristRoll", "LeftWristPitch", "LeftWrist",
        "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw",
        "RightElbow", "RightWristRoll", "RightWristPitch", "RightWrist",
    };
    for (int index = 0; index < G1_BoneCount; ++index) {
        if (name == names[index]) return index;
    }
    return -1;
}

static inline bool sonic_joint_contract_hash_is_valid(
    const json_value* value)
{
    if (value == nullptr || value->kind != json_string ||
        value->string_value.size() != 64) {
        return false;
    }
    for (char character : value->string_value) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

static inline bool sonic_joint_contract_load(
    sonic_joint_contract_entry (&contract)[SonicG1JointCount],
    sonic_joint_contract_metadata* metadata,
    const char* path,
    char* error,
    int capacity)
{
    json_value document;
    if (!json_document_load(document, path, error, capacity) ||
        !sonic_joint_contract_exact_keys(
            document,
            {"rows", "source_mjcf_sha256", "target_order_source_sha256"},
            error,
            capacity)) {
        return false;
    }
    const json_value* source_hash = json_member(document, "source_mjcf_sha256");
    const json_value* target_hash =
        json_member(document, "target_order_source_sha256");
    const json_value* rows = json_member(document, "rows");
    if (!sonic_joint_contract_hash_is_valid(source_hash) ||
        !sonic_joint_contract_hash_is_valid(target_hash) || rows == nullptr ||
        rows->kind != json_array ||
        rows->array_value.size() != SonicG1JointCount) {
        return sonic_projection_error(
            error, capacity, "joint contract identity or row count is invalid");
    }

    sonic_joint_contract_entry candidate[SonicG1JointCount];
    sonic_joint_contract_metadata candidate_metadata;
    candidate_metadata.source_mjcf_sha256 = source_hash->string_value;
    candidate_metadata.target_order_source_sha256 = target_hash->string_value;
    candidate_metadata.source_joint_names.resize(SonicG1JointCount);
    candidate_metadata.target_joint_names.resize(SonicG1JointCount);
    for (int row = 0; row < SonicG1JointCount; ++row) {
        const json_value& encoded =
            rows->array_value[static_cast<std::size_t>(row)];
        if (!sonic_joint_contract_exact_keys(
                encoded,
                {"axis_holden", "lower", "qpos_address", "sign",
                 "source_bone", "source_index", "source_joint",
                 "source_parent", "static_local_holden_wxyz",
                 "target_index", "target_name", "upper", "zero_offset"},
                error,
                capacity)) {
            return false;
        }
        sonic_joint_contract_entry& entry = candidate[row];
        std::string source_bone;
        std::string source_parent;
        int qpos_address = -1;
        if (!sonic_joint_contract_integer(
                entry.source_index,
                encoded,
                "source_index",
                error,
                capacity) ||
            !sonic_joint_contract_integer(
                entry.target_index,
                encoded,
                "target_index",
                error,
                capacity) ||
            !sonic_joint_contract_integer(
                qpos_address,
                encoded,
                "qpos_address",
                error,
                capacity) ||
            qpos_address < 0 ||
            !sonic_joint_contract_string(
                source_bone, encoded, "source_bone", error, capacity) ||
            !sonic_joint_contract_string(
                source_parent, encoded, "source_parent", error, capacity) ||
            !sonic_joint_contract_string(
                entry.source_joint,
                encoded,
                "source_joint",
                error,
                capacity) ||
            !sonic_joint_contract_string(
                entry.target_joint,
                encoded,
                "target_name",
                error,
                capacity) ||
            !sonic_joint_contract_vec3_member(
                entry.axis_holden,
                encoded,
                "axis_holden",
                error,
                capacity) ||
            !sonic_joint_contract_quat_member(
                entry.static_local_holden,
                encoded,
                "static_local_holden_wxyz",
                error,
                capacity) ||
            !sonic_joint_contract_float_member(
                entry.sign, encoded, "sign", error, capacity) ||
            !sonic_joint_contract_float_member(
                entry.zero_offset,
                encoded,
                "zero_offset",
                error,
                capacity) ||
            !sonic_joint_contract_double_member(
                entry.lower, encoded, "lower", error, capacity) ||
            !sonic_joint_contract_double_member(
                entry.upper, encoded, "upper", error, capacity)) {
            return false;
        }
        entry.source_bone = sonic_joint_contract_bone_index(source_bone);
        entry.source_parent = sonic_joint_contract_bone_index(source_parent);
        if (entry.source_bone < 0 || entry.source_parent < 0 ||
            entry.source_index < 0 || entry.source_index >= SonicG1JointCount ||
            entry.target_index < 0 || entry.target_index >= SonicG1JointCount ||
            !candidate_metadata.source_joint_names[
                 static_cast<std::size_t>(entry.source_index)].empty() ||
            !candidate_metadata.target_joint_names[
                 static_cast<std::size_t>(entry.target_index)].empty()) {
            return sonic_projection_error(
                error,
                capacity,
                "joint contract row %d identity is invalid",
                row);
        }
        candidate_metadata.source_joint_names[
            static_cast<std::size_t>(entry.source_index)] = entry.source_joint;
        candidate_metadata.target_joint_names[
            static_cast<std::size_t>(entry.target_index)] = entry.target_joint;
    }
    if (!sonic_projection_contract_valid(candidate, error, capacity)) {
        return false;
    }
    for (int row = 0; row < SonicG1JointCount; ++row) {
        contract[row] = candidate[row];
    }
    if (metadata != nullptr) {
        *metadata = candidate_metadata;
    }
    return true;
}
