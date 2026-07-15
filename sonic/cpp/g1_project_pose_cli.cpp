#include "json_runtime.h"
#include "sonic/cpp/g1_joint_projection.h"

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static bool cli_error(char* error, int capacity, const char* message)
{
    if (error != NULL && capacity > 0) {
        std::snprintf(
            error, static_cast<std::size_t>(capacity), "%s", message);
    }
    return false;
}

static bool cli_exact_keys(
    const json_value& value,
    const std::initializer_list<const char*> expected,
    char* error,
    int capacity)
{
    if (value.kind != json_object ||
        value.object_value.size() != expected.size()) {
        return cli_error(error, capacity, "contract object keys are invalid");
    }
    for (const char* key : expected) {
        if (json_member(value, key) == NULL) {
            return cli_error(
                error, capacity, "contract object key is missing");
        }
    }
    return true;
}

static bool cli_string(
    std::string& output,
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    const json_value* value = json_member(object, key);
    if (value == NULL || value->kind != json_string ||
        value->string_value.empty()) {
        return cli_error(error, capacity, "contract string is invalid");
    }
    output = value->string_value;
    return true;
}

static bool cli_integer(
    int& output,
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    const json_value* value = json_member(object, key);
    if (value == NULL || value->kind != json_number ||
        value->number_value < static_cast<double>(std::numeric_limits<int>::min()) ||
        value->number_value > static_cast<double>(std::numeric_limits<int>::max()) ||
        std::floor(value->number_value) != value->number_value) {
        return cli_error(error, capacity, "contract integer is invalid");
    }
    output = static_cast<int>(value->number_value);
    return true;
}

static bool cli_float_value(
    float& output,
    const json_value& value,
    char* error,
    int capacity)
{
    if (value.kind != json_number || !std::isfinite(value.number_value) ||
        value.number_value < -static_cast<double>(std::numeric_limits<float>::max()) ||
        value.number_value > static_cast<double>(std::numeric_limits<float>::max())) {
        return cli_error(error, capacity, "contract float is invalid");
    }
    output = static_cast<float>(value.number_value);
    if (!std::isfinite(output)) {
        return cli_error(error, capacity, "contract float is invalid");
    }
    return true;
}

static bool cli_float_member(
    float& output,
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    const json_value* value = json_member(object, key);
    return value != NULL && cli_float_value(output, *value, error, capacity);
}

static bool cli_vec3_member(
    vec3& output,
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    const json_value* value = json_member(object, key);
    if (value == NULL || value->kind != json_array ||
        value->array_value.size() != 3) {
        return cli_error(error, capacity, "contract vec3 is invalid");
    }
    return cli_float_value(output.x, value->array_value[0], error, capacity) &&
           cli_float_value(output.y, value->array_value[1], error, capacity) &&
           cli_float_value(output.z, value->array_value[2], error, capacity);
}

static bool cli_quat_member(
    quat& output,
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    const json_value* value = json_member(object, key);
    if (value == NULL || value->kind != json_array ||
        value->array_value.size() != 4) {
        return cli_error(error, capacity, "contract quaternion is invalid");
    }
    return cli_float_value(output.w, value->array_value[0], error, capacity) &&
           cli_float_value(output.x, value->array_value[1], error, capacity) &&
           cli_float_value(output.y, value->array_value[2], error, capacity) &&
           cli_float_value(output.z, value->array_value[3], error, capacity);
}

static int cli_bone_index(const std::string& name)
{
    static const char* const names[G1_BoneCount] = {
        "Simulation", "Hips", "LeftHipPitch", "LeftHipRoll",
        "LeftHipYaw", "LeftKnee", "LeftAnkle", "LeftToe",
        "RightHipPitch", "RightHipRoll", "RightHipYaw", "RightKnee",
        "RightAnkle", "RightToe", "Spine", "Spine1", "Spine2",
        "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw",
        "LeftElbow", "LeftWristRoll", "LeftWristPitch", "LeftWrist",
        "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw",
        "RightElbow", "RightWristRoll", "RightWristPitch", "RightWrist"
    };
    for (int index = 0; index < G1_BoneCount; ++index) {
        if (name == names[index]) return index;
    }
    return -1;
}

static bool cli_hash_valid(const json_value* value)
{
    if (value == NULL || value->kind != json_string ||
        value->string_value.size() != 64) {
        return false;
    }
    for (const char character : value->string_value) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

static bool cli_load_contract(
    sonic_joint_contract_entry (&contract)[SonicG1JointCount],
    const char* path,
    char* error,
    int capacity)
{
    json_value document;
    if (!json_document_load(document, path, error, capacity)) return false;
    if (!cli_exact_keys(
            document,
            {"rows", "source_mjcf_sha256", "target_order_source_sha256"},
            error,
            capacity) ||
        !cli_hash_valid(json_member(document, "source_mjcf_sha256")) ||
        !cli_hash_valid(json_member(document, "target_order_source_sha256"))) {
        return cli_error(error, capacity, "contract identity is invalid");
    }
    const json_value* rows = json_member(document, "rows");
    if (rows == NULL || rows->kind != json_array ||
        rows->array_value.size() != SonicG1JointCount) {
        return cli_error(error, capacity, "contract must contain 29 rows");
    }
    for (int row = 0; row < SonicG1JointCount; ++row) {
        const json_value& source =
            rows->array_value[static_cast<std::size_t>(row)];
        if (!cli_exact_keys(
                source,
                {"axis_holden", "lower", "qpos_address", "sign",
                 "source_bone", "source_index", "source_joint",
                 "source_parent", "static_local_holden_wxyz",
                 "target_index", "target_name", "upper", "zero_offset"},
                error,
                capacity)) {
            return false;
        }
        sonic_joint_contract_entry& entry = contract[row];
        std::string source_bone;
        std::string source_parent;
        int qpos_address = -1;
        if (!cli_integer(
                entry.source_index, source, "source_index", error, capacity) ||
            !cli_integer(
                entry.target_index, source, "target_index", error, capacity) ||
            !cli_integer(
                qpos_address, source, "qpos_address", error, capacity) ||
            qpos_address < 0 ||
            !cli_string(
                source_bone, source, "source_bone", error, capacity) ||
            !cli_string(
                source_parent, source, "source_parent", error, capacity) ||
            !cli_string(
                entry.source_joint, source, "source_joint", error, capacity) ||
            !cli_string(
                entry.target_joint, source, "target_name", error, capacity) ||
            !cli_vec3_member(
                entry.axis_holden, source, "axis_holden", error, capacity) ||
            !cli_quat_member(
                entry.static_local_holden,
                source,
                "static_local_holden_wxyz",
                error,
                capacity) ||
            !cli_float_member(
                entry.sign, source, "sign", error, capacity) ||
            !cli_float_member(
                entry.zero_offset, source, "zero_offset", error, capacity) ||
            !cli_float_member(
                entry.lower, source, "lower", error, capacity) ||
            !cli_float_member(
                entry.upper, source, "upper", error, capacity)) {
            return false;
        }
        entry.source_bone = cli_bone_index(source_bone);
        entry.source_parent = cli_bone_index(source_parent);
        if (entry.source_bone < 0 || entry.source_parent < 0) {
            return cli_error(error, capacity, "contract bone name is invalid");
        }
    }
    return sonic_projection_contract_valid(contract, error, capacity);
}

static bool cli_read_float(float& output)
{
    double value = 0.0;
    if (!(std::cin >> value) || !std::isfinite(value) ||
        value < -static_cast<double>(std::numeric_limits<float>::max()) ||
        value > static_cast<double>(std::numeric_limits<float>::max())) {
        return false;
    }
    output = static_cast<float>(value);
    return std::isfinite(output);
}

static bool cli_read_pose(
    quat (&local_rotations)[G1_BoneCount],
    vec3 (&local_angular_velocities)[G1_BoneCount],
    vec3 (&global_positions)[G1_BoneCount],
    quat (&global_rotations)[G1_BoneCount])
{
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (!cli_read_float(local_rotations[bone].w) ||
            !cli_read_float(local_rotations[bone].x) ||
            !cli_read_float(local_rotations[bone].y) ||
            !cli_read_float(local_rotations[bone].z)) {
            return false;
        }
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (!cli_read_float(local_angular_velocities[bone].x) ||
            !cli_read_float(local_angular_velocities[bone].y) ||
            !cli_read_float(local_angular_velocities[bone].z)) {
            return false;
        }
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (!cli_read_float(global_positions[bone].x) ||
            !cli_read_float(global_positions[bone].y) ||
            !cli_read_float(global_positions[bone].z)) {
            return false;
        }
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (!cli_read_float(global_rotations[bone].w) ||
            !cli_read_float(global_rotations[bone].x) ||
            !cli_read_float(global_rotations[bone].y) ||
            !cli_read_float(global_rotations[bone].z)) {
            return false;
        }
    }
    return true;
}

static void cli_emit_pose(const sonic_projected_pose& pose)
{
    for (int index = 0; index < SonicG1JointCount; ++index)
        std::cout << ' ' << pose.source_joint_position[index];
    for (int index = 0; index < SonicG1JointCount; ++index)
        std::cout << ' ' << pose.source_joint_velocity[index];
    for (int index = 0; index < SonicG1JointCount; ++index)
        std::cout << ' ' << pose.off_axis_residual[index];
    std::cout << ' ' << pose.physical_pelvis_position_holden.x
              << ' ' << pose.physical_pelvis_position_holden.y
              << ' ' << pose.physical_pelvis_position_holden.z
              << ' ' << pose.physical_pelvis_orientation_holden.w
              << ' ' << pose.physical_pelvis_orientation_holden.x
              << ' ' << pose.physical_pelvis_orientation_holden.y
              << ' ' << pose.physical_pelvis_orientation_holden.z;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: g1_project_pose_cli CONTRACT.json\n");
        return 2;
    }
    char error[512] = {};
    sonic_joint_contract_entry contract[SonicG1JointCount];
    if (!cli_load_contract(contract, argv[1], error, sizeof(error))) {
        std::fprintf(stderr, "%s\n", error);
        return 2;
    }
    int pose_count = 0;
    if (!(std::cin >> pose_count) || pose_count <= 0 || pose_count > 100000) {
        std::fprintf(stderr, "input pose count is invalid\n");
        return 2;
    }
    std::vector<sonic_projected_pose> outputs(
        static_cast<std::size_t>(pose_count));
    for (int pose_index = 0; pose_index < pose_count; ++pose_index) {
        quat local_rotations[G1_BoneCount];
        vec3 local_angular_velocities[G1_BoneCount];
        vec3 global_positions[G1_BoneCount];
        quat global_rotations[G1_BoneCount];
        if (!cli_read_pose(
                local_rotations,
                local_angular_velocities,
                global_positions,
                global_rotations)) {
            std::fprintf(stderr, "pose %d input is truncated or invalid\n", pose_index);
            return 2;
        }
        if (!sonic_project_pose(
                outputs[static_cast<std::size_t>(pose_index)],
                contract,
                slice1d<quat>(G1_BoneCount, local_rotations),
                slice1d<vec3>(G1_BoneCount, local_angular_velocities),
                slice1d<vec3>(G1_BoneCount, global_positions),
                slice1d<quat>(G1_BoneCount, global_rotations),
                error,
                sizeof(error))) {
            std::fprintf(stderr, "pose %d: %s\n", pose_index, error);
            return 1;
        }
    }
    std::cin >> std::ws;
    if (!std::cin.eof()) {
        std::fprintf(stderr, "input contains trailing data\n");
        return 2;
    }

    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
              << pose_count;
    for (const sonic_projected_pose& pose : outputs) cli_emit_pose(pose);
    std::cout << '\n';
    return std::cout ? 0 : 2;
}
