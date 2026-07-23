#pragma once

#include "episode_reach_planner.h"
#include "g1_flat_motion_matcher.h"
#include "interaction_attachment.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace episode {

enum class EpisodeState : uint8_t {
    FreeLocomotion,
    Searching,
    Approach,
    Bridge,
    Reach,
    CarryBlend,
    Carry,
    Failed,
};

struct EpisodeConfig {
    float entry_position_m = 0.08F;
    float entry_yaw_radians = 0.174532925F;
    float entry_speed_mps = 0.08F;
    int stable_entry_ticks = 3;
    float approach_timeout_seconds = 12.0F;
    float bridge_seconds = 0.24F;
    float carry_blend_seconds = 0.36F;
};

struct FrozenAttempt {
    ObjectSnapshot object{};
    GraspCandidate grasp{};
    ReachPlan plan{};
    uint64_t request_id = 0U;
};

struct EpisodeInput {
    float dt = 0.0F;
    LocomotionCommand command{};
    uint64_t current_object_generation = 0U;
    bool cancel_pressed = false;
    bool reset_pressed = false;
};

struct EpisodeOutput {
    interaction::Pose pose{};
    interaction::Transform object_world{};
    EpisodeState state = EpisodeState::FreeLocomotion;
    interaction::Hand selected_hand = interaction::Hand::Right;
    bool attached = false;
    bool owns_pose = true;
    std::string failure;
};

class InteractionEpisode {
public:
    InteractionEpisode(
        std::filesystem::path walking_database,
        std::filesystem::path carry_left_database,
        std::filesystem::path carry_right_database,
        EpisodeConfig config = EpisodeConfig{});

    bool commit(FrozenAttempt attempt);
    const EpisodeOutput& update(const EpisodeInput& input);
    void reset();

    EpisodeState state() const;
    const std::optional<FrozenAttempt>& attempt() const;
    const EpisodeOutput& output() const;

private:
    void update_approach(const EpisodeInput& input);
    void update_bridge(float dt);
    void update_reach(float dt);
    void update_carry(const EpisodeInput& input);
    void publish(interaction::Pose pose);
    void fail(std::string reason);
    void cancel_before_contact();
    bool attach_at_contact();
    interaction::ContactMeasurement hand_measurement(
        const interaction::Pose& pose,
        bool contact_event) const;

    std::filesystem::path walking_database_;
    std::filesystem::path carry_left_database_;
    std::filesystem::path carry_right_database_;
    EpisodeConfig config_{};
    FlatMotionMatcher matcher_;
    EpisodeState state_ = EpisodeState::FreeLocomotion;
    std::optional<FrozenAttempt> attempt_;
    EpisodeOutput output_{};

    interaction::TargetRegistry registry_;
    std::optional<interaction::AttachmentController> attachment_;
    interaction::TargetHandle target_handle_{};

    interaction::Pose bridge_start_{};
    interaction::Pose contact_pose_{};
    float state_seconds_ = 0.0F;
    float frame_accumulator_ = 0.0F;
    size_t reach_frame_ = 0U;
    int stable_entry_ticks_ = 0;
};

}  // namespace episode
