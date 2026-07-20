#include "interaction_funnel_follower.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace {

using interaction::FunnelFollowerOutput;
using interaction::FunnelFollowerState;
using interaction::FunnelCancelReason;
using interaction::FunnelSample;
using interaction::InteractionFunnelFollower;
using interaction::kFunnelSampleCount;

// A smooth reference proposal: advances 0.04 m along x each sample with a
// constant unit yaw vector (cos=1, sin=0).
std::array<FunnelSample, kFunnelSampleCount> smooth_samples() {
    std::array<FunnelSample, kFunnelSampleCount> samples{};
    for (int s = 0; s < kFunnelSampleCount; ++s) {
        samples[static_cast<size_t>(s)].x = 0.04F * static_cast<float>(s);
        samples[static_cast<size_t>(s)].z = 0.0F;
        samples[static_cast<size_t>(s)].yaw_sin = 0.0F;
        samples[static_cast<size_t>(s)].yaw_cos = 1.0F;
    }
    return samples;
}

FunnelSample tracked_at(const FunnelSample& target) {
    return target;  // perfect tracking of a previously published target
}

void test_publishes_every_sample_once_on_consecutive_ticks() {
    auto samples = smooth_samples();
    InteractionFunnelFollower follower(42U, samples, 100U);
    assert(follower.proposal_seed() == 42U);

    // First tick publishes sample 0; robot has not executed anything yet.
    FunnelFollowerOutput first = follower.tick({100U, samples[0]});
    assert(first.published);
    assert(first.sample.x == samples[0].x);
    assert(first.state == FunnelFollowerState::Following);

    for (int s = 1; s < kFunnelSampleCount; ++s) {
        // The tracked pose reflects the previously published target.
        FunnelSample tracked = tracked_at(samples[static_cast<size_t>(s - 1)]);
        FunnelFollowerOutput out =
            follower.tick({100U + static_cast<uint64_t>(s), tracked});
        assert(out.published);
        assert(out.sample.x == samples[static_cast<size_t>(s)].x);
    }
    assert(follower.diagnostics().published_count == kFunnelSampleCount);
    assert(follower.diagnostics().state == FunnelFollowerState::Completed);
    assert(follower.diagnostics().cancel_reason == FunnelCancelReason::None);
}

void test_completed_follower_publishes_nothing_further() {
    auto samples = smooth_samples();
    InteractionFunnelFollower follower(1U, samples, 0U);
    for (int s = 0; s < kFunnelSampleCount; ++s) {
        FunnelSample tracked =
            (s == 0) ? samples[0] : tracked_at(samples[static_cast<size_t>(s - 1)]);
        follower.tick({static_cast<uint64_t>(s), tracked});
    }
    assert(follower.diagnostics().state == FunnelFollowerState::Completed);
    FunnelFollowerOutput extra =
        follower.tick({static_cast<uint64_t>(kFunnelSampleCount), samples[15]});
    assert(!extra.published);
    assert(extra.state == FunnelFollowerState::Completed);
    assert(follower.diagnostics().published_count == kFunnelSampleCount);
}

void test_missed_tick_cancels_before_publishing() {
    auto samples = smooth_samples();
    InteractionFunnelFollower follower(7U, samples, 10U);
    follower.tick({10U, samples[0]});           // publishes sample 0
    // Skip tick 11; deliver tick 12 -> missed tick.
    FunnelFollowerOutput out = follower.tick({12U, samples[0]});
    assert(!out.published);
    assert(out.state == FunnelFollowerState::Cancelled);
    assert(follower.diagnostics().cancel_reason == FunnelCancelReason::MissedTick);
    assert(follower.diagnostics().published_count == 1);
}

void test_repeated_tick_index_cancels_as_missed() {
    auto samples = smooth_samples();
    InteractionFunnelFollower follower(7U, samples, 0U);
    follower.tick({0U, samples[0]});
    FunnelFollowerOutput out = follower.tick({0U, samples[0]});  // not consecutive
    assert(!out.published);
    assert(follower.diagnostics().cancel_reason == FunnelCancelReason::MissedTick);
}

void test_translation_tracking_error_cancels() {
    auto samples = smooth_samples();
    InteractionFunnelFollower follower(3U, samples, 0U);
    follower.tick({0U, samples[0]});  // publishes sample 0
    // At tick 1 the robot should be near sample 0; place it 0.2 m away.
    FunnelSample drifted = samples[0];
    drifted.z = 0.2F;  // exceeds 0.18 m
    FunnelFollowerOutput out = follower.tick({1U, drifted});
    assert(!out.published);
    assert(out.state == FunnelFollowerState::Cancelled);
    assert(follower.diagnostics().cancel_reason == FunnelCancelReason::TranslationError);
    assert(follower.diagnostics().published_count == 1);
}

void test_translation_within_threshold_keeps_following() {
    auto samples = smooth_samples();
    InteractionFunnelFollower follower(3U, samples, 0U);
    follower.tick({0U, samples[0]});
    FunnelSample drifted = samples[0];
    drifted.z = 0.17F;  // within 0.18 m
    FunnelFollowerOutput out = follower.tick({1U, drifted});
    assert(out.published);
    assert(out.state == FunnelFollowerState::Following);
}

void test_yaw_tracking_error_cancels() {
    auto samples = smooth_samples();
    InteractionFunnelFollower follower(9U, samples, 0U);
    follower.tick({0U, samples[0]});
    // 30 degrees of yaw error exceeds the 25 degree limit.
    FunnelSample rotated = samples[0];
    rotated.yaw_cos = std::cos(30.0F * 3.14159265F / 180.0F);
    rotated.yaw_sin = std::sin(30.0F * 3.14159265F / 180.0F);
    FunnelFollowerOutput out = follower.tick({1U, rotated});
    assert(!out.published);
    assert(follower.diagnostics().cancel_reason == FunnelCancelReason::YawError);
}

void test_cancel_is_terminal() {
    auto samples = smooth_samples();
    InteractionFunnelFollower follower(9U, samples, 0U);
    follower.tick({0U, samples[0]});
    follower.tick({5U, samples[0]});  // missed tick -> cancelled
    assert(follower.diagnostics().state == FunnelFollowerState::Cancelled);
    // Even a well-formed subsequent tick stays cancelled and publishes nothing.
    FunnelFollowerOutput out = follower.tick({6U, samples[0]});
    assert(!out.published);
    assert(out.state == FunnelFollowerState::Cancelled);
    assert(follower.diagnostics().cancel_reason == FunnelCancelReason::MissedTick);
}

}  // namespace

int main() {
    test_publishes_every_sample_once_on_consecutive_ticks();
    test_completed_follower_publishes_nothing_further();
    test_missed_tick_cancels_before_publishing();
    test_repeated_tick_index_cancels_as_missed();
    test_translation_tracking_error_cancels();
    test_translation_within_threshold_keeps_following();
    test_yaw_tracking_error_cancels();
    test_cancel_is_terminal();
    return 0;
}
