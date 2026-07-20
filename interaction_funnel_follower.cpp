#include "interaction_funnel_follower.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace interaction {

namespace {

float translation_error(const FunnelSample& tracked, const FunnelSample& target) {
    float dx = tracked.x - target.x;
    float dz = tracked.z - target.z;
    return std::sqrt(dx * dx + dz * dz);
}

FunnelSample interpolate(
    const FunnelSample& left,
    const FunnelSample& right,
    float alpha) {
    const double left_yaw = std::atan2(left.yaw_sin, left.yaw_cos);
    const double right_yaw = std::atan2(right.yaw_sin, right.yaw_cos);
    const double delta = std::atan2(
        std::sin(right_yaw - left_yaw),
        std::cos(right_yaw - left_yaw));
    const double yaw = left_yaw + static_cast<double>(alpha) * delta;
    return {
        left.x + alpha * (right.x - left.x),
        left.z + alpha * (right.z - left.z),
        static_cast<float>(std::sin(yaw)),
        static_cast<float>(std::cos(yaw)),
    };
}

struct RouteProjection {
    int segment = 0;
    float alpha = 0.0F;
    float distance = std::numeric_limits<float>::infinity();
    FunnelSample sample{};
};

RouteProjection project_to_remaining_route(
    const FunnelExecutionTargets& targets,
    int progress_index,
    const FunnelSample& tracked) {
    RouteProjection best{};
    const int terminal = kFunnelExecutionTickCount - 1;
    if (progress_index >= terminal) {
        best.segment = terminal;
        best.alpha = 0.0F;
        best.distance = translation_error(tracked, targets.back());
        best.sample = targets.back();
        return best;
    }
    for (int segment = std::max(0, progress_index);
         segment < terminal;
         ++segment) {
        const FunnelSample& left = targets[static_cast<size_t>(segment)];
        const FunnelSample& right = targets[static_cast<size_t>(segment + 1)];
        const double dx = static_cast<double>(right.x) - left.x;
        const double dz = static_cast<double>(right.z) - left.z;
        const double length_squared = dx * dx + dz * dz;
        const double from_left_x = static_cast<double>(tracked.x) - left.x;
        const double from_left_z = static_cast<double>(tracked.z) - left.z;
        const float alpha = length_squared <= 1.0e-12
            ? 0.0F
            : static_cast<float>(std::clamp(
                  (from_left_x * dx + from_left_z * dz) / length_squared,
                  0.0,
                  1.0));
        const FunnelSample projected = interpolate(left, right, alpha);
        const float distance = translation_error(tracked, projected);
        if (distance < best.distance) {
            best = {segment, alpha, distance, projected};
        }
    }
    return best;
}

FunnelSample lookahead_target(
    const FunnelExecutionTargets& targets,
    const RouteProjection& projection,
    int& lookahead_index) {
    const int terminal = kFunnelExecutionTickCount - 1;
    if (projection.segment >= terminal) {
        lookahead_index = terminal;
        return targets.back();
    }
    FunnelSample cursor = projection.sample;
    float remaining = kFunnelLookaheadDistance;
    for (int segment = projection.segment; segment < terminal; ++segment) {
        const FunnelSample& endpoint =
            targets[static_cast<size_t>(segment + 1)];
        const float segment_length = translation_error(cursor, endpoint);
        if (segment_length >= remaining && segment_length > 1.0e-8F) {
            lookahead_index = segment + 1;
            return interpolate(cursor, endpoint, remaining / segment_length);
        }
        remaining -= segment_length;
        cursor = endpoint;
    }
    lookahead_index = terminal;
    return targets.back();
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
      targets_(expand_funnel_execution(samples)),
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

    if (follow_ticks_ >= kFunnelMaximumFollowTicks) {
        cancel(FunnelCancelReason::Timeout);
        return {false, FunnelSample{}, diagnostics_.state};
    }
    ++follow_ticks_;

    const RouteProjection projection = project_to_remaining_route(
        targets_, progress_index_, input.tracked_pose);
    if (projection.distance > kFunnelMaxTrackingTranslation) {
        cancel(FunnelCancelReason::TranslationError);
        return {false, FunnelSample{}, diagnostics_.state};
    }
    if (yaw_error(input.tracked_pose, projection.sample) >
        kFunnelMaxTrackingYawRadians) {
        cancel(FunnelCancelReason::YawError);
        return {false, FunnelSample{}, diagnostics_.state};
    }

    const int projected_index = std::min(
        projection.segment + (projection.alpha >= 0.5F ? 1 : 0),
        kFunnelExecutionTickCount - 1);
    progress_index_ = std::max(progress_index_, projected_index);
    diagnostics_.progress_index = progress_index_;

    const FunnelSample& terminal = targets_.back();
    const bool terminal_position =
        translation_error(input.tracked_pose, terminal) <=
        kFunnelTerminalPositionTolerance;
    const bool terminal_yaw =
        yaw_error(input.tracked_pose, terminal) <=
        kFunnelTerminalYawTolerance;
    diagnostics_.terminal_settle_ticks =
        terminal_position && terminal_yaw
        ? diagnostics_.terminal_settle_ticks + 1U
        : 0U;
    ++expected_tick_index_;
    if (diagnostics_.terminal_settle_ticks >=
        kFunnelRequiredTerminalTicks) {
        diagnostics_.state = FunnelFollowerState::Completed;
        diagnostics_.progress_index = kFunnelExecutionTickCount - 1;
        diagnostics_.lookahead_index = kFunnelExecutionTickCount - 1;
        return {false, FunnelSample{}, diagnostics_.state};
    }

    int lookahead_index = progress_index_;
    FunnelSample published = lookahead_target(
        targets_, projection, lookahead_index);
    diagnostics_.lookahead_index = std::max(
        diagnostics_.lookahead_index, lookahead_index);
    ++diagnostics_.published_count;
    return {true, published, diagnostics_.state};
}

}  // namespace interaction
