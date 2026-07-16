#include "sonic/cpp/g1_joint_contract_io.h"

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

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
    if (!sonic_joint_contract_load(
            contract, nullptr, argv[1], error, sizeof(error))) {
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
