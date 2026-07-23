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
    PlaceApproach,
    PlaceBridge,
    PlaceReach,
    PlaceReturn,
    Failed,
};

struct EpisodeConfig {
    float entry_position_m = 0.08F;
    float entry_yaw_radians = 0.174532925F;
    float place_entry_position_m = 0.15F;
    float place_entry_yaw_radians = 1.047197551F;
    float waypoint_reached_m = 0.10F;
    int stable_entry_ticks = 3;
    float approach_timeout_seconds = 24.0F;
    float bridge_seconds = 0.24F;
    float place_bridge_seconds = 0.50F;
    float carry_blend_seconds = 0.36F;
    interaction::AttachmentConfig attachment = [] {
        interaction::AttachmentConfig config{};
        config.require_lift_for_hold = false;
        return config;
    }();
};

struct FrozenAttempt {
    ObjectSnapshot object{};
    GraspCandidate grasp{};
    ReachPlan plan{};
    uint64_t request_id = 0U;
};

struct FrozenPlaceAttempt {
    ObjectSnapshot destination{};
    GraspCandidate grasp{};
    ReachPlan plan{};
    interaction::PlacedSupportContext support{};
    uint64_t request_id = 0U;
};

struct EpisodeInput {
    float dt = 0.0F;
    LocomotionCommand command{};
    uint64_t current_object_generation = 0U;
    bool cancel_pressed = false;
    bool reset_pressed = false;
    uint64_t current_destination_generation = 0U;
};

struct EpisodeOutput {
    interaction::Pose pose{};
    interaction::Transform object_world{};
    EpisodeState state = EpisodeState::FreeLocomotion;
    interaction::Hand selected_hand = interaction::Hand::Right;
    FlatSkeletonWorldPose flat_locomotion{};
    float approach_distance_m = 0.0F;
    float approach_yaw_error_radians = 0.0F;
    float locomotion_speed_mps = 0.0F;
    float bridge_alpha = 0.0F;
    bool attached = false;
    bool place_ready = false;
    bool placed = false;
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
    bool commit_place(FrozenPlaceAttempt attempt);
    const EpisodeOutput& update(const EpisodeInput& input);
    void reset();

    EpisodeState state() const;
    const std::optional<FrozenAttempt>& attempt() const;
    const std::optional<FrozenPlaceAttempt>& place_attempt() const;
    const EpisodeOutput& output() const;

private:
    void update_approach(const EpisodeInput& input);
    void update_bridge(float dt);
    void update_reach(float dt);
    void update_carry(const EpisodeInput& input);
    void update_place_approach(const EpisodeInput& input);
    void update_place_bridge(float dt);
    void update_place_reach(float dt);
    void update_place_return(float dt);
    void update_attached_object(float dt);
    void publish(interaction::Pose pose);
    void fail(std::string reason);
    void cancel_before_contact();
    void cancel_place_before_release();
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
    std::optional<FrozenPlaceAttempt> place_attempt_;
    EpisodeOutput output_{};

    interaction::TargetRegistry registry_;
    std::optional<interaction::AttachmentController> attachment_;
    interaction::TargetHandle target_handle_{};

    interaction::Pose bridge_start_{};
    FlatSkeletonWorldPose bridge_flat_start_{};
    interaction::Pose contact_pose_{};
    float state_seconds_ = 0.0F;
    float frame_accumulator_ = 0.0F;
    size_t reach_frame_ = 0U;
    size_t approach_waypoint_ = 0U;
    int stable_entry_ticks_ = 0;
};

}  // namespace episode
