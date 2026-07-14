#pragma once
#include <array>
#include <cstdint>
#include <string_view>

namespace g1_skeleton {
enum Bone : int32_t {
    Simulation, Hips,
    LeftHipPitch, LeftHipRoll, LeftHipYaw, LeftKnee, LeftAnkle, LeftToe,
    RightHipPitch, RightHipRoll, RightHipYaw, RightKnee, RightAnkle, RightToe,
    Spine, Spine1, Spine2,
    LeftShoulderPitch, LeftShoulderRoll, LeftShoulderYaw, LeftElbow,
    LeftWristRoll, LeftWristPitch, LeftWrist,
    RightShoulderPitch, RightShoulderRoll, RightShoulderYaw, RightElbow,
    RightWristRoll, RightWristPitch, RightWrist,
    BoneCount
};

inline constexpr std::array<std::string_view, BoneCount> kBoneNames = {
    "Simulation", "Hips",
    "LeftHipPitch", "LeftHipRoll", "LeftHipYaw", "LeftKnee", "LeftAnkle", "LeftToe",
    "RightHipPitch", "RightHipRoll", "RightHipYaw", "RightKnee", "RightAnkle", "RightToe",
    "Spine", "Spine1", "Spine2",
    "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbow",
    "LeftWristRoll", "LeftWristPitch", "LeftWrist",
    "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbow",
    "RightWristRoll", "RightWristPitch", "RightWrist"
};

inline constexpr std::array<int32_t, BoneCount> kParents = {
    -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
};

inline constexpr std::string_view kSkeletonSignature =
    "6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7";
static_assert(kBoneNames.size() == BoneCount);
static_assert(kParents.size() == BoneCount);
}
