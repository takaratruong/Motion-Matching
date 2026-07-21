#pragma once

#include "interaction_pose.h"

#include <array>
#include <cstddef>
#include <filesystem>

namespace interaction::offline_overlap {

inline constexpr size_t kFrameCount = 80U;
inline constexpr size_t kFrameRateHz = 25U;
inline constexpr size_t kContactCount = 2U;

struct Clip {
    std::array<Pose, kFrameCount> frames{};

    static Clip load(const std::filesystem::path& path);
};

class Player {
public:
    explicit Player(Clip clip, bool loop);

    const Pose& pose() const;
    size_t frame_index() const;
    bool finished() const;
    void advance_25hz();
    void restart();

private:
    Clip clip_;
    size_t frame_index_ = 0U;
    bool loop_ = false;
    bool finished_ = false;
};

}  // namespace interaction::offline_overlap
