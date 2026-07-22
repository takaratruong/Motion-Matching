#pragma once

#include "interaction_hand_trajectories.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace interaction {

enum class ReuseAuditStatus : uint8_t {
    Complete = 0U,
    Incomplete = 1U,
};

struct ReuseAuditConfig {
    float maximum_contact_correction_m = 0.45F;
    size_t worker_count = 4U;
    uint32_t deadline_milliseconds = 30000U;
    size_t display_limit = 12U;
};

struct ReuseAuditCounts {
    size_t total = 0U;
    size_t processed = 0U;
    size_t contact_accepted = 0U;
    size_t fully_shaped = 0U;
    size_t object_rejected = 0U;
    size_t environment_rejected = 0U;
    size_t reusable = 0U;
};

struct ReuseAuditMotion {
    HandTrajectory source;
    ShapedHandTrajectory shaped;
};

struct ReuseAuditResult {
    ReuseAuditStatus status = ReuseAuditStatus::Incomplete;
    ReuseAuditCounts counts{};
    uint64_t elapsed_milliseconds = 0U;
    std::vector<ReuseAuditMotion> displayed;
};

ReuseAuditResult audit_reusable_hand_trajectories(
    const Database& database,
    const HandTrajectoryQuery& query,
    const OrientedBox& object,
    const EnvironmentGeometry& environment,
    const TrajectoryCollisionConfig& collision,
    const ReuseAuditConfig& config = ReuseAuditConfig{});

}  // namespace interaction
