#include "interaction_funnel_follower.h"

#include <cmath>

namespace interaction {

namespace {

float translation_error(const FunnelSample& tracked, const FunnelSample& target) {
    float dx = tracked.x - target.x;
    float dz = tracked.z - target.z;
    return std::sqrt(dx * dx + dz * dz);
}

float yaw_error(const FunnelSample& tracked, const FunnelSample& target) {
    float cross = tracked.yaw_cos * target.yaw_sin - tracked.yaw_sin * target.yaw_cos;
    float dot = tracked.yaw_cos * target.yaw_cos + tracked.yaw_sin * target.yaw_sin;
    return std::abs(std::atan2(cross, dot));
}

}  // namespace

InteractionFunnelFollower::InteractionFunnelFollower(
    uint64_t proposal_seed,
    const std::array<FunnelSample, kFunnelSampleCount>& samples,
    uint64_t first_tick_index)
    : proposal_seed_(proposal_seed),
      samples_(samples),
      expected_tick_index_(first_tick_index) {
    diagnostics_.proposal_seed = proposal_seed;
    diagnostics_.state = FunnelFollowerState::Following;
}

void InteractionFunnelFollower::cancel(FunnelCancelReason reason) {
    diagnostics_.state = FunnelFollowerState::Cancelled;
    diagnostics_.cancel_reason = reason;
}

FunnelFollowerOutput InteractionFunnelFollower::tick(const FunnelFollowerInput& input) {
    // A follower that is no longer following is terminal: publish nothing.
    if (diagnostics_.state != FunnelFollowerState::Following) {
        return {false, FunnelSample{}, diagnostics_.state};
    }

    // Each tick must land on the next consecutive 25 Hz index. Any gap or
    // repeat is a missed tick and cancels the follower before publishing.
    if (input.tick_index != expected_tick_index_) {
        cancel(FunnelCancelReason::MissedTick);
        return {false, FunnelSample{}, diagnostics_.state};
    }

    // Before publishing anything past the first sample, verify the robot has
    // tracked the previously published target closely enough.
    if (started_) {
        const FunnelSample& last_target = samples_[static_cast<size_t>(next_sample_ - 1)];
        if (translation_error(input.tracked_pose, last_target) >
            kFunnelMaxTrackingTranslation) {
            cancel(FunnelCancelReason::TranslationError);
            return {false, FunnelSample{}, diagnostics_.state};
        }
        if (yaw_error(input.tracked_pose, last_target) > kFunnelMaxTrackingYawRadians) {
            cancel(FunnelCancelReason::YawError);
            return {false, FunnelSample{}, diagnostics_.state};
        }
    }

    FunnelSample published = samples_[static_cast<size_t>(next_sample_)];
    ++next_sample_;
    ++expected_tick_index_;
    started_ = true;
    ++diagnostics_.published_count;
    if (next_sample_ >= kFunnelSampleCount) {
        diagnostics_.state = FunnelFollowerState::Completed;
    }
    return {true, published, diagnostics_.state};
}

}  // namespace interaction
