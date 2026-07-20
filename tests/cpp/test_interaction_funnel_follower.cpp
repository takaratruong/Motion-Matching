#include "interaction_funnel_follower.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace {

using interaction::FunnelCancelReason;
using interaction::FunnelFollowerOutput;
using interaction::FunnelFollowerState;
using interaction::FunnelSample;
using interaction::InteractionFunnelFollower;
using interaction::kFunnelLookaheadDistance;
using interaction::kFunnelMaximumFollowTicks;
using interaction::kFunnelRequiredTerminalTicks;
using interaction::kFunnelSampleCount;

std::array<FunnelSample, kFunnelSampleCount> smooth_samples() {
    std::array<FunnelSample, kFunnelSampleCount> samples{};
    for (int index = 0; index < kFunnelSampleCount; ++index) {
        samples[static_cast<size_t>(index)] = {
            0.04F * static_cast<float>(index), 0.0F, 0.0F, 1.0F};
    }
    return samples;
}

FunnelSample sample_at(float x, float z = 0.0F, float yaw = 0.0F) {
    return {x, z, std::sin(yaw), std::cos(yaw)};
}

void test_lagging_character_keeps_a_forward_lookahead() {
    InteractionFunnelFollower follower(42U, smooth_samples(), 100U);
    const FunnelFollowerOutput first = follower.tick({100U, sample_at(0.0F)});
    const FunnelFollowerOutput lagged = follower.tick({101U, sample_at(0.01F)});

    assert(first.published);
    assert(lagged.published);
    assert(first.sample.x >= kFunnelLookaheadDistance - 0.01F);
    assert(lagged.sample.x >= first.sample.x);
    assert(follower.diagnostics().progress_index >= 0);
    assert(follower.diagnostics().lookahead_index >=
           follower.diagnostics().progress_index);
}

void test_tracked_pose_ahead_skips_passed_points() {
    InteractionFunnelFollower follower(7U, smooth_samples(), 10U);
    const FunnelFollowerOutput output =
        follower.tick({10U, sample_at(0.30F)});
    assert(output.published);
    assert(output.sample.x >= 0.40F);
    assert(follower.diagnostics().progress_index > 30);
}

void test_lateral_error_never_commands_an_earlier_target() {
    InteractionFunnelFollower follower(3U, smooth_samples(), 0U);
    const auto ahead = follower.tick({0U, sample_at(0.28F)});
    const int progress = follower.diagnostics().progress_index;
    const auto lateral = follower.tick({1U, sample_at(0.27F, 0.10F)});
    assert(ahead.published && lateral.published);
    assert(follower.diagnostics().progress_index >= progress);
    assert(lateral.sample.x >= ahead.sample.x);
}

void test_completion_requires_three_terminal_observations() {
    InteractionFunnelFollower follower(9U, smooth_samples(), 20U);
    const FunnelSample terminal = smooth_samples().back();
    for (uint32_t settled = 0U;
         settled + 1U < kFunnelRequiredTerminalTicks;
         ++settled) {
        const auto output = follower.tick({20U + settled, terminal});
        assert(output.published);
        assert(output.state == FunnelFollowerState::Following);
        assert(follower.diagnostics().terminal_settle_ticks == settled + 1U);
    }
    const auto completed = follower.tick({
        20U + kFunnelRequiredTerminalTicks - 1U, terminal});
    assert(!completed.published);
    assert(completed.state == FunnelFollowerState::Completed);
    assert(follower.diagnostics().state == FunnelFollowerState::Completed);
}

void test_leaving_terminal_tolerance_resets_settle() {
    InteractionFunnelFollower follower(9U, smooth_samples(), 0U);
    follower.tick({0U, smooth_samples().back()});
    assert(follower.diagnostics().terminal_settle_ticks == 1U);
    const auto output = follower.tick({1U, sample_at(0.50F)});
    assert(output.published);
    assert(follower.diagnostics().terminal_settle_ticks == 0U);
}

void test_missed_or_repeated_tick_cancels() {
    InteractionFunnelFollower skipped(1U, smooth_samples(), 5U);
    skipped.tick({5U, sample_at(0.0F)});
    assert(!skipped.tick({7U, sample_at(0.0F)}).published);
    assert(skipped.diagnostics().cancel_reason == FunnelCancelReason::MissedTick);

    InteractionFunnelFollower repeated(1U, smooth_samples(), 5U);
    repeated.tick({5U, sample_at(0.0F)});
    assert(!repeated.tick({5U, sample_at(0.0F)}).published);
    assert(repeated.diagnostics().cancel_reason == FunnelCancelReason::MissedTick);
}

void test_cross_track_error_cancels_without_publication() {
    InteractionFunnelFollower follower(2U, smooth_samples(), 0U);
    const auto output = follower.tick({0U, sample_at(0.1F, 0.19F)});
    assert(!output.published);
    assert(output.state == FunnelFollowerState::Cancelled);
    assert(follower.diagnostics().cancel_reason ==
           FunnelCancelReason::TranslationError);
}

void test_route_yaw_error_cancels_without_publication() {
    InteractionFunnelFollower follower(2U, smooth_samples(), 0U);
    const float thirty_degrees = 30.0F * 3.14159265F / 180.0F;
    const auto output = follower.tick({0U, sample_at(0.1F, 0.0F, thirty_degrees)});
    assert(!output.published);
    assert(output.state == FunnelFollowerState::Cancelled);
    assert(follower.diagnostics().cancel_reason == FunnelCancelReason::YawError);
}

void test_follow_timeout_cancels() {
    InteractionFunnelFollower follower(2U, smooth_samples(), 0U);
    for (uint32_t tick = 0U; tick < kFunnelMaximumFollowTicks; ++tick) {
        const auto output = follower.tick({tick, sample_at(0.0F)});
        assert(output.published);
    }
    const auto timed_out = follower.tick({
        static_cast<uint64_t>(kFunnelMaximumFollowTicks), sample_at(0.0F)});
    assert(!timed_out.published);
    assert(timed_out.state == FunnelFollowerState::Cancelled);
    assert(follower.diagnostics().cancel_reason == FunnelCancelReason::Timeout);
}

void test_explicit_cancel_is_terminal() {
    InteractionFunnelFollower follower(9U, smooth_samples(), 0U);
    follower.cancel(FunnelCancelReason::ExplicitCancel);
    const auto output = follower.tick({0U, sample_at(0.0F)});
    assert(!output.published);
    assert(output.state == FunnelFollowerState::Cancelled);
    assert(follower.diagnostics().cancel_reason ==
           FunnelCancelReason::ExplicitCancel);
}

}  // namespace

int main() {
    test_lagging_character_keeps_a_forward_lookahead();
    test_tracked_pose_ahead_skips_passed_points();
    test_lateral_error_never_commands_an_earlier_target();
    test_completion_requires_three_terminal_observations();
    test_leaving_terminal_tolerance_resets_settle();
    test_missed_or_repeated_tick_cancels();
    test_cross_track_error_cancels_without_publication();
    test_route_yaw_error_cancels_without_publication();
    test_follow_timeout_cancels();
    test_explicit_cancel_is_terminal();
    return 0;
}
