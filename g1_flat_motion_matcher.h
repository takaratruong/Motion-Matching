#pragma once

#include "interaction_pose.h"

#include <filesystem>
#include <array>
#include <memory>

struct database;

namespace episode {

struct LocomotionCommand {
    vec3 desired_velocity_world{};
    quat desired_heading_world{};
};

inline constexpr size_t kFlatSkeletonBoneCount = 23U;
inline constexpr std::array<int32_t, kFlatSkeletonBoneCount>
    kFlatSkeletonParents = {
        -1, 0, 1, 2, 3, 4, 1, 6, 7, 8, 1, 10,
        11, 12, 13, 12, 15, 16, 17, 12, 19, 20, 21};

struct FlatSkeletonWorldPose {
    std::array<vec3, kFlatSkeletonBoneCount> positions{};
    std::array<quat, kFlatSkeletonBoneCount> rotations{};
    bool valid = false;
};

class FlatMotionMatcher {
public:
    explicit FlatMotionMatcher(
        const std::filesystem::path& database_path,
        const std::filesystem::path& g1_reference_database = {});
    ~FlatMotionMatcher();
    void update(const LocomotionCommand& command, float dt);
    void switch_database(const std::filesystem::path& database_path);
    void switch_database(
        const std::filesystem::path& database_path,
        const interaction::Pose& live_pose);
    void reset_database(const std::filesystem::path& database_path);
    const interaction::LocomotionSnapshot& snapshot() const;
    const FlatSkeletonWorldPose& flat_skeleton() const;
    bool uses_native_g1() const;
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
    interaction::Pose geometry_reference_{};
    bool has_geometry_reference_ = false;
    bool uses_flat_controller_skeleton_ = false;
    mutable FlatSkeletonWorldPose flat_skeleton_{};
    FlatSkeletonWorldPose flat_transition_source_{};
    float transition_seconds_ = 0.20F;
};

}  // namespace episode
