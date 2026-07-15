#pragma once

#include <array>
#include <cstdint>

namespace locomotion_timing {

inline constexpr int kRateHz = 25;
inline constexpr float kStepSeconds = 1.0F / 25.0F;

inline constexpr std::array<int32_t, 3> kTrajectoryFrameOffsets = {
    8, 17, 25};
inline constexpr std::array<float, 3> kTrajectorySampleTimesSeconds = {
    static_cast<float>(kTrajectoryFrameOffsets[0]) /
        static_cast<float>(kRateHz),
    static_cast<float>(kTrajectoryFrameOffsets[1]) /
        static_cast<float>(kRateHz),
    static_cast<float>(kTrajectoryFrameOffsets[2]) /
        static_cast<float>(kRateHz)};
inline constexpr std::array<float, 3> kTrajectoryStepSeconds = {
    static_cast<float>(kTrajectoryFrameOffsets[0]) /
        static_cast<float>(kRateHz),
    static_cast<float>(
        kTrajectoryFrameOffsets[1] - kTrajectoryFrameOffsets[0]) /
        static_cast<float>(kRateHz),
    static_cast<float>(
        kTrajectoryFrameOffsets[2] - kTrajectoryFrameOffsets[1]) /
        static_cast<float>(kRateHz)};

constexpr uint32_t ticks_for_milliseconds(uint32_t milliseconds) {
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(milliseconds) *
             static_cast<uint64_t>(kRateHz) +
         999U) /
        1000U);
}

}  // namespace locomotion_timing
