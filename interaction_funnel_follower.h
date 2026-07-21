#pragma once

// Pure 25 Hz learned-funnel follower.
//
// The follower consumes one certified proposal's execution-order samples as a
// geometric route. On consecutive 25 Hz ticks it advances monotonic spatial
// progress and publishes a point ahead of the tracked root. It never writes
// simulation state; callers apply the published target through locomotion.
//
// The follower cancels on missed ticks, timeout, or when the tracked root is
// more than 0.18 m or 25 degrees from the remaining route.

#include "interaction_funnel_timing.h"

#include <array>
#include <cstdint>
#include <vector>

namespace interaction {

enum class FunnelFollowerState {
    Following,
    Completed,
    Cancelled,
};

enum class FunnelCancelReason {
    None,
    ExplicitCancel,
    MissedTick,
    TranslationError,
    YawError,
    Timeout,
};

// Tracking thresholds above which the follower cancels.
constexpr float kFunnelMaxTrackingTranslation = 0.18F;        // metres
constexpr float kFunnelMaxTrackingYawRadians = 0.43633231F;   // 25 degrees
constexpr float kFunnelLookaheadDistance = 0.12F;
constexpr float kFunnelTerminalPositionTolerance = 0.04F;
constexpr float kFunnelTerminalYawTolerance = 0.34906585F;   // 20 degrees
constexpr uint32_t kFunnelRequiredTerminalTicks = 3U;
constexpr uint32_t kFunnelMaximumFollowTicks = 250U;

// One 25 Hz observation: the monotonically increasing tick index and the
// current tracked object-local planar pose (x, z, sin(yaw), cos(yaw)).
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
    int progress_index = 0;
    int lookahead_index = 0;
    uint32_t terminal_settle_ticks = 0U;
    uint64_t proposal_seed = 0U;
};

using FunnelRoute = std::vector<FunnelSample>;

class InteractionFunnelFollower {
public:
    // Construct a follower bound to one proposal's immutable identity (seed and
    // execution-order samples) and the first tick index it expects.
    InteractionFunnelFollower(
        uint64_t proposal_seed,
        const std::array<FunnelSample, kFunnelSampleCount>& samples,
        uint64_t first_tick_index);
    InteractionFunnelFollower(
        uint64_t proposal_seed,
        FunnelRoute route,
        uint64_t first_tick_index,
        uint32_t required_terminal_ticks = kFunnelRequiredTerminalTicks);

    // Advance one 25 Hz tick. Publishes the next sample, or cancels.
    FunnelFollowerOutput tick(const FunnelFollowerInput& input);

    // Terminally cancel an armed follower without publishing another sample.
    void cancel(FunnelCancelReason reason);

    uint64_t proposal_seed() const { return proposal_seed_; }
    const FunnelFollowerDiagnostics& diagnostics() const { return diagnostics_; }

private:
    const uint64_t proposal_seed_;
    const FunnelRoute targets_;
    uint64_t expected_tick_index_;
    const uint32_t required_terminal_ticks_;
    uint32_t follow_ticks_ = 0U;
    int progress_index_ = 0;
    FunnelFollowerDiagnostics diagnostics_{};
};

}  // namespace interaction
