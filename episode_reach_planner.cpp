#include "episode_reach_planner.h"

#include "g1_skeleton.h"
#include "reach_placement.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace episode {
namespace {

float yaw(quat rotation) {
    const vec3 facing = quat_mul_vec3(
        rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(facing.x, facing.z);
}

float angle_delta(float left, float right) {
    return std::abs(std::atan2(
        std::sin(left - right), std::cos(left - right)));
}

interaction::Transform entry_root(
    const reach::Pack& pack,
    const reach::Candidate& candidate,
    vec3 target_position) {
    const int32_t frame =
        pack.database.range_starts.at(candidate.clip);
    const interaction::Pose pose = reach::place_pose(
        pack,
        candidate.clip,
        candidate.yaw_index,
        target_position,
        frame);
    const size_t root =
        static_cast<size_t>(g1_skeleton::Simulation);
    return {pose.positions[root], pose.rotations[root]};
}

bool segment_intersects_expanded_box(
    vec3 start,
    vec3 stop,
    const interaction::OrientedBox& box,
    float expansion) {
    const quat inverse_rotation = quat_inv(box.world.rotation);
    const vec3 local_start = quat_mul_vec3(
        inverse_rotation, start - box.world.position);
    const vec3 local_stop = quat_mul_vec3(
        inverse_rotation, stop - box.world.position);
    const vec3 delta = local_stop - local_start;
    const float half_x = 0.5F * box.dimensions.x + expansion;
    const float half_z = 0.5F * box.dimensions.z + expansion;
    float minimum = 0.0F;
    float maximum = 1.0F;
    const auto clip_axis = [&](float origin, float direction, float half) {
        if (std::abs(direction) <= 1.0e-8F) {
            return origin >= -half && origin <= half;
        }
        float first = (-half - origin) / direction;
        float second = (half - origin) / direction;
        if (first > second) std::swap(first, second);
        minimum = std::max(minimum, first);
        maximum = std::min(maximum, second);
        return minimum <= maximum;
    };
    return clip_axis(local_start.x, delta.x, half_x) &&
           clip_axis(local_start.z, delta.z, half_z);
}

}  // namespace

bool reach_plan_cost_less(
    const ReachPlanCost& left,
    const ReachPlanCost& right) {
    return std::tie(
               left.entry_distance_m,
               left.entry_facing_error_radians,
               left.grasp_error,
               left.directness_cost,
               left.clip,
               left.yaw_index) <
           std::tie(
               right.entry_distance_m,
               right.entry_facing_error_radians,
               right.grasp_error,
               right.directness_cost,
               right.clip,
               right.yaw_index);
}

bool entry_segment_clear(
    vec3 start_world,
    vec3 stop_world,
    const interaction::EnvironmentGeometry& environment,
    float expansion_m) {
    if (!std::isfinite(expansion_m) || expansion_m < 0.0F) {
        throw std::invalid_argument("invalid entry segment expansion");
    }
    return std::none_of(
        environment.boxes.begin(),
        environment.boxes.end(),
        [&](const interaction::OrientedBox& box) {
            return segment_intersects_expanded_box(
                start_world, stop_world, box, expansion_m);
        });
}

std::optional<ReachPlan> choose_reach_plan(
    const reach::Pack& pack,
    const reach::SearchResult& compact,
    const reach::ExhaustiveQuery& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const reach::SearchConfig& search_config,
    const interaction::Pose& live_pose,
    GraspCandidate grasp) {
    if (!compact.complete) return std::nullopt;
    const size_t root =
        static_cast<size_t>(g1_skeleton::Simulation);
    const vec3 live_root = live_pose.positions[root];
    const float live_yaw = yaw(live_pose.rotations[root]);
    size_t best_index = std::numeric_limits<size_t>::max();
    ReachPlanCost best_cost{};
    for (size_t evaluation_index : compact.accepted) {
        const reach::Evaluation& evaluation =
            compact.evaluations.at(evaluation_index).evaluation;
        const interaction::Transform entry = entry_root(
            pack, evaluation.candidate, query.target.position);
        if (!entry_segment_clear(
                live_root, entry.position, environment, 0.28F)) {
            continue;
        }
        const vec3 delta = entry.position - live_root;
        const ReachPlanCost cost{
            std::sqrt(delta.x * delta.x + delta.z * delta.z),
            angle_delta(yaw(entry.rotation), live_yaw),
            evaluation.position_error_m +
                0.10F * evaluation.orientation_error_radians +
                0.05F * evaluation.approach_error_radians,
            evaluation.directness_cost,
            evaluation.candidate.clip,
            evaluation.candidate.yaw_index,
        };
        if (best_index == std::numeric_limits<size_t>::max() ||
            reach_plan_cost_less(cost, best_cost)) {
            best_index = evaluation_index;
            best_cost = cost;
        }
    }
    if (best_index == std::numeric_limits<size_t>::max()) {
        return std::nullopt;
    }
    reach::Evaluation full = reach::regenerate(
        pack,
        compact.evaluations.at(best_index),
        query,
        object,
        environment,
        search_config);
    if (full.rejection != reach::Rejection::None || full.poses.empty()) {
        return std::nullopt;
    }
    const reach::Hand hand = static_cast<reach::Hand>(
        pack.database.active_hands.at(full.candidate.clip));
    if (grasp.grasp_id == 0U) {
        grasp.hand_world = query.target;
        grasp.approach_world = query.approach_world;
        grasp.grasp_id = 1U;
    }
    return ReachPlan{
        grasp,
        std::move(full),
        hand,
        entry_root(pack, compact.evaluations.at(best_index).evaluation.candidate,
                   query.target.position),
        best_cost,
    };
}

}  // namespace episode
