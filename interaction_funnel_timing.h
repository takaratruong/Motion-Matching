#pragma once

#include "interaction_funnel_artifact.h"

#include <array>

namespace interaction {

constexpr int kFunnelKnotCount = kFunnelSampleCount;
constexpr int kFunnelExecutionTickCount = 75;
constexpr std::array<int, kFunnelKnotCount> kFunnelKnotTickOffsets{
    0, 5, 10, 15, 20, 25, 30, 35,
    39, 44, 49, 54, 59, 64, 69, 74,
};

using FunnelExecutionTargets =
    std::array<FunnelSample, kFunnelExecutionTickCount>;

FunnelExecutionTargets expand_funnel_execution(
    const std::array<FunnelSample, kFunnelKnotCount>& knots);

}  // namespace interaction
