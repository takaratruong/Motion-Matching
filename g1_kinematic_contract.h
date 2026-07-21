#pragma once

#include "vec.h"

#include <cstdint>
#include <cstring>

// Sole production owner of the fixed G1 skeleton identifiers, signature,
// named-leg layout/geometry, and runtime sample-period predicate.  This
// contract is intentionally independent of database, skeleton, and IK
// implementation headers so strict certified translation units can use it.
enum G1Bone
{
    G1_Simulation = 0,
    G1_Hips = 1,
    G1_LeftHipPitch = 2,
    G1_LeftHipRoll = 3,
    G1_LeftHipYaw = 4,
    G1_LeftKnee = 5,
    G1_LeftAnkle = 6,
    G1_LeftToe = 7,
    G1_RightHipPitch = 8,
    G1_RightHipRoll = 9,
    G1_RightHipYaw = 10,
    G1_RightKnee = 11,
    G1_RightAnkle = 12,
    G1_RightToe = 13,
    G1_Spine = 14,
    G1_Spine1 = 15,
    G1_Spine2 = 16,
    G1_LeftShoulderPitch = 17,
    G1_LeftShoulderRoll = 18,
    G1_LeftShoulderYaw = 19,
    G1_LeftElbow = 20,
    G1_LeftWristRoll = 21,
    G1_LeftWristPitch = 22,
    G1_LeftWrist = 23,
    G1_RightShoulderPitch = 24,
    G1_RightShoulderRoll = 25,
    G1_RightShoulderYaw = 26,
    G1_RightElbow = 27,
    G1_RightWristRoll = 28,
    G1_RightWristPitch = 29,
    G1_RightWrist = 30,
    G1_BoneCount = 31
};

inline constexpr char G1_SkeletonSignature[] =
    "6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7";

struct G1LegConfig
{
    const char* name;
    int hip;
    int knee;
    int ankle;
    int contact;
    vec3 knee_hinge_axis_local;
    vec3 foot_forward_local;
    vec3 sole_normal_local;
    vec3 foot_sphere_centers_local[4];
    vec3 sole_points_local[4];
    float foot_sphere_radius_m;
    vec3 thigh_start_local;
    vec3 thigh_end_local;
    float thigh_radius_m;
    vec3 shin_start_local;
    vec3 shin_end_local;
    float shin_radius_m;
    float reach_buffer_m;
    float planted_clearance_m;
    float swing_clearance_m;
    float max_swing_lift_m;
    float max_correction_radians;
};

static inline G1LegConfig g1_leg_config(
    const char* name,
    int hip,
    int knee,
    int ankle,
    int contact)
{
    G1LegConfig config = {};
    config.name = name;
    config.hip = hip;
    config.knee = knee;
    config.ankle = ankle;
    config.contact = contact;
    config.knee_hinge_axis_local = vec3(0.0f, 0.0f, -1.0f);
    config.foot_forward_local = vec3(1.0f, 0.0f, 0.0f);
    config.sole_normal_local = vec3(0.0f, 1.0f, 0.0f);
    config.foot_sphere_radius_m = 0.02f;
    config.foot_sphere_centers_local[0] =
        vec3(-0.05f, -0.03f, -0.025f);
    config.foot_sphere_centers_local[1] =
        vec3(-0.05f, -0.03f, +0.025f);
    config.foot_sphere_centers_local[2] =
        vec3(+0.12f, -0.03f, -0.030f);
    config.foot_sphere_centers_local[3] =
        vec3(+0.12f, -0.03f, +0.030f);
    for (int index = 0; index < 4; ++index) {
        config.sole_points_local[index] =
            config.foot_sphere_centers_local[index] -
            config.sole_normal_local * config.foot_sphere_radius_m;
    }
    config.thigh_start_local = vec3(0.0f, -0.02f, 0.0f);
    config.thigh_end_local = vec3(-0.078f, -0.17f, 0.0f);
    config.thigh_radius_m = 0.05f;
    config.shin_start_local = vec3(0.0f, -0.05f, 0.0f);
    config.shin_end_local = vec3(0.0f, -0.28f, 0.0f);
    config.shin_radius_m = 0.04f;
    config.reach_buffer_m = 0.015f;
    config.planted_clearance_m = 0.005f;
    config.swing_clearance_m = 0.015f;
    config.max_swing_lift_m = 0.08f;
    config.max_correction_radians = 0.35f;
    return config;
}

static inline G1LegConfig g1_left_leg_config()
{
    return g1_leg_config(
        "left", G1_LeftHipYaw, G1_LeftKnee,
        G1_LeftAnkle, G1_LeftToe);
}

static inline G1LegConfig g1_right_leg_config()
{
    return g1_leg_config(
        "right", G1_RightHipYaw, G1_RightKnee,
        G1_RightAnkle, G1_RightToe);
}

static inline bool g1_dt_is_exact_25_hz(float dt)
{
    static_assert(sizeof(float) == sizeof(uint32_t),
                  "G1 timing contract requires binary32 storage");
    uint32_t bits = 0;
    std::memcpy(&bits, &dt, sizeof(bits));
    // Exact binary32 encoding of 1.0f / 25.0f (0.04f).
    return bits == UINT32_C(0x3d23d70a);
}
