#pragma once

#include "interaction_features.h"

#include <array>
#include <cstdint>

namespace interaction {

enum class Phase : uint8_t {
    Approach = 0,
    Reach = 1,
    Contact = 2,
    Lift = 3,
    Hold = 4,
};

enum class Reason : uint8_t {
    None,
    PackUnavailable,
    TargetUnavailable,
    TargetChanged,
    OutOfRange,
    NoCandidate,
    PoorMatch,
    BlockedPath,
    CorrectionLimit,
    Cancelled,
    ContactPosition,
    ContactOrientation,
    JointLimit,
    LostContact,
    ClipEnded,
    Reset,
};

struct MatchConfig {
    float maximum_approach_m = 1.00F;
    float maximum_root_correction_m = 0.25F;
    float maximum_yaw_correction_radians = 0.436332313F;
    float maximum_hand_correction_m = 0.12F;
    float maximum_hand_orientation_radians = 0.436332313F;
    std::array<float, 5> group_weights{1, 1, 2, 2, 1};
    float maximum_cost = 9.0F;
};

struct MatchCandidate {
    int32_t clip = -1;
    int32_t entry_frame = -1;
    int32_t contact_frame = -1;
    int32_t lift_frame = -1;
    int32_t hold_frame = -1;
    Transform scene_from_source{};
    vec3 entry_root_offset{};
    float entry_yaw_offset = 0.0F;
    float total_cost = 0.0F;
    std::array<float, 5> group_costs{};
};

struct MatchInput {
    const Database* database = nullptr;
    const Features* features = nullptr;
    NormalizedQuery query{};
    LocomotionSnapshot locomotion{};
    InteractionTarget target{};
    GraspAffordance affordance{};
    PickRequest request{};
};

struct MatchResult {
    bool accepted = false;
    MatchCandidate candidate{};
    Reason reason = Reason::None;
};

MatchResult select_whole_clip(
    const MatchInput& input,
    const MatchConfig& config);

}  // namespace interaction
