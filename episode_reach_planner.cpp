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

std::optional<std::vector<vec3>> find_entry_path(
    vec3 start_world,
    vec3 certified_entry_world,
    const interaction::EnvironmentGeometry& environment,
    float expansion_m) {
    if (!std::isfinite(expansion_m) || expansion_m < 0.0F) {
        throw std::invalid_argument("invalid entry path expansion");
    }
    constexpr float kMaximumCertifiedEndpointOverlap = 0.30F;
    std::vector<vec3> nodes = {start_world, certified_entry_world};
    nodes.reserve(2U + 8U * environment.boxes.size());
    for (const interaction::OrientedBox& box : environment.boxes) {
        const float half_x =
            0.5F * box.dimensions.x + expansion_m +
            kEntryPathCornerMarginM;
        const float half_z =
            0.5F * box.dimensions.z + expansion_m +
            kEntryPathCornerMarginM;
        for (const float x : {-half_x, half_x}) {
            for (const float z : {-half_z, half_z}) {
                const vec3 world = box.world.position +
                    quat_mul_vec3(
                        box.world.rotation, vec3(x, 0.0F, z));
                nodes.push_back(
                    vec3(world.x, start_world.y, world.z));
            }
        }
        for (const vec3 local : {
                 vec3(-half_x, 0.0F, 0.0F),
                 vec3(half_x, 0.0F, 0.0F),
                 vec3(0.0F, 0.0F, -half_z),
                 vec3(0.0F, 0.0F, half_z)}) {
            const vec3 world = box.world.position +
                quat_mul_vec3(box.world.rotation, local);
            nodes.push_back(vec3(world.x, start_world.y, world.z));
        }
    }
    const auto point_clear = [&](vec3 point) {
        return std::none_of(
            environment.boxes.begin(),
            environment.boxes.end(),
            [&](const interaction::OrientedBox& box) {
                return segment_intersects_expanded_box(
                    point, point, box, expansion_m);
            });
    };
    const auto edge_clear = [&](size_t from, size_t to) {
        const vec3 original_start = nodes[from];
        const vec3 original_stop = nodes[to];
        const vec3 delta(
            original_stop.x - original_start.x,
            0.0F,
            original_stop.z - original_start.z);
        const float distance = length(delta);
        if (distance <= 1.0e-5F) return true;
        const float step = std::min(1.0F, 0.02F / distance);
        float first = 0.0F;
        float last = 1.0F;
        if (from == 0U) {
            while (first < last &&
                   !point_clear(lerp(
                       original_start, original_stop, first))) {
                first += step;
            }
            if (first * distance >
                kMaximumCertifiedEndpointOverlap) {
                return false;
            }
        }
        if (to == 1U) {
            while (last > first &&
                   !point_clear(lerp(
                       original_start, original_stop, last))) {
                last -= step;
            }
            if ((1.0F - last) * distance >
                kMaximumCertifiedEndpointOverlap) {
                return false;
            }
        }
        if (first >= last) return false;
        const vec3 start = lerp(
            original_start, original_stop, first);
        const vec3 stop = lerp(
            original_start, original_stop, last);
        return entry_segment_clear(
            start, stop, environment, expansion_m);
    };

    const size_t count = nodes.size();
    std::vector<float> distances(
        count, std::numeric_limits<float>::infinity());
    std::vector<size_t> previous(count, count);
    std::vector<uint8_t> visited(count, 0U);
    distances[0] = 0.0F;
    for (size_t iteration = 0U; iteration < count; ++iteration) {
        size_t current = count;
        for (size_t node = 0U; node < count; ++node) {
            if (visited[node] == 0U &&
                (current == count ||
                 distances[node] < distances[current])) {
                current = node;
            }
        }
        if (current == count ||
            !std::isfinite(distances[current])) break;
        if (current == 1U) break;
        visited[current] = 1U;
        for (size_t next = 1U; next < count; ++next) {
            if (next == current || visited[next] != 0U ||
                !edge_clear(current, next)) {
                continue;
            }
            const vec3 delta = nodes[next] - nodes[current];
            const float edge =
                std::hypot(delta.x, delta.z);
            const float candidate = distances[current] + edge;
            if (candidate < distances[next]) {
                distances[next] = candidate;
                previous[next] = current;
            }
        }
    }
    if (!std::isfinite(distances[1])) return std::nullopt;
    std::vector<vec3> reversed;
    for (size_t node = 1U; node != 0U; node = previous[node]) {
        if (node >= count || previous[node] >= count) {
            return std::nullopt;
        }
        reversed.push_back(nodes[node]);
    }
    return std::vector<vec3>(reversed.rbegin(), reversed.rend());
}

std::optional<size_t> find_support_index(
    const interaction::EnvironmentGeometry& environment,
    const interaction::Transform& marker,
    vec3 object_dimensions) {
    std::optional<size_t> best;
    float best_cost = std::numeric_limits<float>::infinity();
    const vec3 object_bottom = marker.position +
        quat_mul_vec3(
            marker.rotation,
            vec3(0.0F, -0.5F * object_dimensions.y, 0.0F));
    for (size_t index = 0U; index < environment.boxes.size(); ++index) {
        const interaction::OrientedBox& box = environment.boxes[index];
        const vec3 support_normal = quat_mul_vec3(
            box.world.rotation, vec3(0.0F, 1.0F, 0.0F));
        if (box.dimensions.y >
                0.5F * std::min(box.dimensions.x, box.dimensions.z) ||
            dot(support_normal, vec3(0.0F, 1.0F, 0.0F)) < 0.965925826F) {
            continue;
        }
        const vec3 local = quat_inv_mul_vec3(
            box.world.rotation, marker.position - box.world.position);
        const float outside_x = std::max(
            0.0F, std::abs(local.x) - 0.5F * box.dimensions.x);
        const float outside_z = std::max(
            0.0F, std::abs(local.z) - 0.5F * box.dimensions.z);
        const vec3 support_top = box.world.position +
            quat_mul_vec3(
                box.world.rotation,
                vec3(0.0F, 0.5F * box.dimensions.y, 0.0F));
        const float vertical_gap =
            std::abs(object_bottom.y - support_top.y);
        if (outside_x > 0.5F * object_dimensions.x ||
            outside_z > 0.5F * object_dimensions.z ||
            vertical_gap > 0.08F) {
            continue;
        }
        const float cost =
            2.0F * std::hypot(outside_x, outside_z) + vertical_gap;
        if (cost < best_cost) {
            best = index;
            best_cost = cost;
        }
    }
    return best;
}

bool reach_hand_allowed(
    reach::Hand candidate,
    std::optional<reach::Hand> required_hand) {
    return !required_hand.has_value() || candidate == *required_hand;
}

std::optional<ReachPlan> choose_reach_plan(
    const reach::Pack& pack,
    const reach::SearchResult& compact,
    const reach::ExhaustiveQuery& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const reach::SearchConfig& search_config,
    const interaction::Pose& live_pose,
    GraspCandidate grasp,
    std::optional<reach::Hand> required_hand) {
    if (!compact.complete) return std::nullopt;
    const size_t root =
        static_cast<size_t>(g1_skeleton::Simulation);
    const vec3 live_root = live_pose.positions[root];
    const float live_yaw = yaw(live_pose.rotations[root]);
    size_t best_index = std::numeric_limits<size_t>::max();
    ReachPlanCost best_cost{};
    std::vector<vec3> best_path;
    for (size_t evaluation_index : compact.accepted) {
        const reach::Evaluation& evaluation =
            compact.evaluations.at(evaluation_index).evaluation;
        const reach::Hand candidate_hand = static_cast<reach::Hand>(
            pack.database.active_hands.at(
                evaluation.candidate.clip));
        if (!reach_hand_allowed(candidate_hand, required_hand)) {
            continue;
        }
        const interaction::Transform entry = entry_root(
            pack, evaluation.candidate, query.target.position);
        const std::optional<std::vector<vec3>> path =
            find_entry_path(
                live_root, entry.position, environment, 0.28F);
        if (!path.has_value()) continue;
        float path_distance = 0.0F;
        vec3 previous_point = live_root;
        for (const vec3 waypoint : *path) {
            const vec3 segment = waypoint - previous_point;
            path_distance += std::hypot(segment.x, segment.z);
            previous_point = waypoint;
        }
        const ReachPlanCost cost{
            path_distance,
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
            best_path = *path;
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
        std::move(best_path),
        best_cost,
    };
}

}  // namespace episode
