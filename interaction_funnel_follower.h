#pragma once

// Pure 25 Hz learned-funnel follower.
//
// The follower consumes one certified proposal's execution-order samples and
// publishes each sample exactly once on consecutive 25 Hz ticks. It is a pure
// state machine: it reads the tracked planar pose and produces a target to
// publish, but never writes the simulation root, displayed root, joints, the
// object registry, or attachment state. Callers apply the published target.
//
// The follower cancels (terminally) on a missed tick or when tracking error
// exceeds 0.18 m of translation or 25 degrees of yaw.

#include "interaction_funnel_artifact.h"

#include <array>
#include <cstdint>

namespace interaction {

enum class FunnelFollowerState {
    Following,
    Completed,
    Cancelled,
};

enum class FunnelCancelReason {
    None,
    MissedTick,
    TranslationError,
    YawError,
};

// Tracking thresholds above which the follower cancels.
constexpr float kFunnelMaxTrackingTranslation = 0.18F;        // metres
constexpr float kFunnelMaxTrackingYawRadians = 0.43633231F;   // 25 degrees

// One 25 Hz observation: the monotonically increasing tick index and the
// current tracked planar pose (a FunnelSample carries planar x, y and a unit
// yaw vector).
struct FunnelFollowerInput {
    uint64_t tick_index = 0U;
    FunnelSample tracked_pose{};
};

// Result of a single tick.
struct FunnelFollowerOutput {
    bool published = false;        // true iff sample is a fresh target to apply
    FunnelSample sample{};         // the published execution-order target
    FunnelFollowerState state = FunnelFollowerState::Following;
};

// Immutable diagnostics snapshot.
struct FunnelFollowerDiagnostics {
    FunnelFollowerState state = FunnelFollowerState::Following;
    FunnelCancelReason cancel_reason = FunnelCancelReason::None;
    int published_count = 0;
    uint64_t proposal_seed = 0U;
};

class InteractionFunnelFollower {
public:
    // Construct a follower bound to one proposal's immutable identity (seed and
    // execution-order samples) and the first tick index it expects.
    InteractionFunnelFollower(
        uint64_t proposal_seed,
        const std::array<FunnelSample, kFunnelSampleCount>& samples,
        uint64_t first_tick_index);

    // Advance one 25 Hz tick. Publishes the next sample, or cancels.
    FunnelFollowerOutput tick(const FunnelFollowerInput& input);

    uint64_t proposal_seed() const { return proposal_seed_; }
    const FunnelFollowerDiagnostics& diagnostics() const { return diagnostics_; }

private:
    void cancel(FunnelCancelReason reason);

    const uint64_t proposal_seed_;
    const std::array<FunnelSample, kFunnelSampleCount> samples_;
    uint64_t expected_tick_index_;
    bool started_ = false;
    int next_sample_ = 0;
    FunnelFollowerDiagnostics diagnostics_{};
};

}  // namespace interaction
