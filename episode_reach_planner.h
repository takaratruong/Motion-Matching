#pragma once

#include "episode_grasp_provider.h"
#include "reach_search.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace episode {

inline constexpr float kEntryPathCornerMarginM = 0.18F;

struct ReachPlanCost {
    float entry_distance_m = 0.0F;
    float entry_facing_error_radians = 0.0F;
    float grasp_error = 0.0F;
    float directness_cost = 0.0F;
    size_t clip = 0U;
    uint8_t yaw_index = 0U;
};

struct ReachPlan {
    GraspCandidate grasp{};
    reach::Evaluation reach{};
    reach::Hand hand = reach::Hand::Left;
    interaction::Transform entry_root_world{};
    std::vector<vec3> entry_waypoints_world;
    ReachPlanCost cost{};
    std::vector<interaction::Pose> return_poses;
};

bool reach_plan_cost_less(
    const ReachPlanCost& left,
    const ReachPlanCost& right);

bool entry_segment_clear(
    vec3 start_world,
    vec3 stop_world,
    const interaction::EnvironmentGeometry& environment,
    float expansion_m);

std::optional<std::vector<vec3>> find_entry_path(
    vec3 start_world,
    vec3 certified_entry_world,
    const interaction::EnvironmentGeometry& environment,
    float expansion_m);

std::optional<size_t> find_support_index(
    const interaction::EnvironmentGeometry& environment,
    const interaction::Transform& marker,
    vec3 object_dimensions);

bool reach_hand_allowed(
    reach::Hand candidate,
    std::optional<reach::Hand> required_hand);

std::optional<ReachPlan> choose_reach_plan(
    const reach::Pack& pack,
    const reach::SearchResult& compact,
    const reach::ExhaustiveQuery& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const reach::SearchConfig& search_config,
    const interaction::Pose& live_pose,
    GraspCandidate grasp = GraspCandidate{},
    std::optional<reach::Hand> required_hand = std::nullopt);

}  // namespace episode
