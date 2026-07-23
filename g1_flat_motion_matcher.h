#pragma once

#include "interaction_pose.h"

#include <filesystem>
#include <memory>

struct database;

namespace episode {

struct LocomotionCommand {
    vec3 desired_velocity_world{};
    quat desired_heading_world{};
};

class FlatMotionMatcher {
public:
    explicit FlatMotionMatcher(const std::filesystem::path& database_path);
    ~FlatMotionMatcher();
    void update(const LocomotionCommand& command, float dt);
    void switch_database(const std::filesystem::path& database_path);
    const interaction::LocomotionSnapshot& snapshot() const;
    float planar_speed() const;

private:
    void prepare(const database& source);
    void tick(const LocomotionCommand& command);
    interaction::Pose mapped_pose(int frame) const;

    std::unique_ptr<database> database_;
    int source_frame_ = 0;
    float accumulator_seconds_ = 0.0F;
    float search_seconds_ = 0.0F;
    interaction::Transform world_from_source_{};
    interaction::LocomotionSnapshot snapshot_{};
    interaction::Pose transition_source_{};
    float transition_seconds_ = 0.20F;
};

}  // namespace episode
