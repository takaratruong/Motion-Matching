#pragma once

#include "interaction_ik.h"

#include <cstdint>
#include <vector>

namespace interaction {

struct CarryConfig {
    int32_t minimum_hold_frames = 25;
    float maximum_grasp_drift_m = 0.02F;
    float maximum_grasp_drift_radians = 0.174532925F;
    float minimum_root_displacement_m = 0.30F;
    float minimum_average_speed_mps = 0.20F;
    float search_interval_seconds = 0.10F;
    float spine_weight = 0.25F;
    float inactive_arm_weight = 0.0F;
};

struct CarryRange {
    int32_t clip = -1;
    int32_t start_frame = -1;
    int32_t stop_frame = -1;
    Hand hand = Hand::Right;
};

struct CarryRanges {
    std::vector<CarryRange> recorded;
    std::vector<int32_t> rejected;
};

CarryRanges classify_carry_ranges(
    const Database& database,
    const CarryConfig& config = CarryConfig{});

class CarryController {
public:
    CarryController(
        const Database& database,
        const Features& features,
        CarryRanges ranges,
        CarryConfig config = CarryConfig{},
        IKConfig ik_config = IKConfig{});
    void start(
        const Pose& final_hold_pose,
        Hand hand,
        const GraspAffordance& affordance,
        Transform object_world);
    Pose update(const LocomotionSnapshot& locomotion, float dt);
    bool recorded() const;
    bool inactive_arm_tracks_locomotion() const;
    Transform object_world() const;

private:
    const Database* database_ = nullptr;
    const Features* features_ = nullptr;
    CarryRanges ranges_{};
    CarryConfig config_{};
    IKConfig ik_config_{};
    Pose final_hold_pose_{};
    GraspAffordance affordance_{};
    Transform object_world_{};
    Hand hand_ = Hand::Right;
    bool recorded_ = false;
    Pose last_safe_pose_{};
    float search_seconds_ = 0.0F;
    bool started_ = false;
    bool search_pending_ = true;
    Transform object_in_root_{};
    int32_t recorded_range_index_ = -1;
    double source_frame_exact_ = 0.0;
    double search_seconds_exact_ = 0.0;
    Pose transition_source_pose_{};
    float transition_progress_ = 0.0F;
    int64_t recorded_selection_epoch_ = 0;
    int64_t published_seam_key_ = -2;
    int64_t transition_seam_key_ = -2;
    bool transition_active_ = false;
};

}  // namespace interaction
