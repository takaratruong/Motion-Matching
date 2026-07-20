#include "interaction_funnel_timing.h"

#include <cmath>
#include <cstddef>

namespace interaction {

FunnelExecutionTargets expand_funnel_execution(
    const std::array<FunnelSample, kFunnelKnotCount>& knots) {
    FunnelExecutionTargets targets{};
    int interval = 0;
    for (int tick = 0; tick < kFunnelExecutionTickCount; ++tick) {
        while (interval + 1 < kFunnelKnotCount &&
               tick > kFunnelKnotTickOffsets[static_cast<size_t>(interval + 1)]) {
            ++interval;
        }

        if (tick == kFunnelKnotTickOffsets[static_cast<size_t>(interval)]) {
            targets[static_cast<size_t>(tick)] =
                knots[static_cast<size_t>(interval)];
            continue;
        }
        if (interval + 1 < kFunnelKnotCount &&
            tick == kFunnelKnotTickOffsets[static_cast<size_t>(interval + 1)]) {
            targets[static_cast<size_t>(tick)] =
                knots[static_cast<size_t>(interval + 1)];
            ++interval;
            continue;
        }

        const int left_tick =
            kFunnelKnotTickOffsets[static_cast<size_t>(interval)];
        const int right_tick =
            kFunnelKnotTickOffsets[static_cast<size_t>(interval + 1)];
        const double alpha = static_cast<double>(tick - left_tick) /
            static_cast<double>(right_tick - left_tick);
        const FunnelSample& left = knots[static_cast<size_t>(interval)];
        const FunnelSample& right = knots[static_cast<size_t>(interval + 1)];

        const double left_yaw = std::atan2(
            static_cast<double>(left.yaw_sin),
            static_cast<double>(left.yaw_cos));
        const double right_yaw = std::atan2(
            static_cast<double>(right.yaw_sin),
            static_cast<double>(right.yaw_cos));
        const double delta = std::atan2(
            std::sin(right_yaw - left_yaw),
            std::cos(right_yaw - left_yaw));
        const double yaw = left_yaw + alpha * delta;
        const double left_x = static_cast<double>(left.x);
        const double left_z = static_cast<double>(left.z);
        const double right_x = static_cast<double>(right.x);
        const double right_z = static_cast<double>(right.z);

        targets[static_cast<size_t>(tick)] = {
            static_cast<float>(left_x + alpha * (right_x - left_x)),
            static_cast<float>(left_z + alpha * (right_z - left_z)),
            static_cast<float>(std::sin(yaw)),
            static_cast<float>(std::cos(yaw)),
        };
    }
    return targets;
}

}  // namespace interaction
