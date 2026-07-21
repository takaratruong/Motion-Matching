#pragma once

#include "interaction_pose.h"

#include <array>
#include <cstddef>
#include <filesystem>

namespace interaction::offline_overlap {

inline constexpr size_t kFrameCount = 80U;
inline constexpr size_t kFrameRateHz = 25U;
inline constexpr size_t kContactCount = 2U;
inline constexpr size_t kBridgeSettleFrame = 49U;
inline constexpr size_t kPickupContactFrame = 75U;

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
    void restart_aligned_to(const Pose& target_pose);
    void restart_bridged_to(
        const Pose& live_pose,
        const Transform& destination_root);
    void restart_bridged_to_frame(
        const Pose& live_pose,
        const Transform& destination_frame);
    Pose aligned_pose() const;

private:
    Clip clip_;
    size_t frame_index_ = 0U;
    bool loop_ = false;
    bool finished_ = false;
    Transform alignment_{};
    bool bridge_enabled_ = false;
    vec3 endpoint_translation_{};
    quat endpoint_rotation_{};
};

class AttachedObjectFollower {
public:
    void attach(
        const Transform& hand_world,
        const Transform& object_world);
    void reset();
    bool attached() const;
    Transform follow(const Transform& hand_world) const;

private:
    bool attached_ = false;
    Transform object_in_hand_{};
};

}  // namespace interaction::offline_overlap
