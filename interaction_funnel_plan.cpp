#include "interaction_funnel_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace interaction {
namespace {

constexpr std::array<float, 16> kCandidateOffsetRadians{
    0.0F,
    22.5F * PIf / 180.0F, -22.5F * PIf / 180.0F,
    45.0F * PIf / 180.0F, -45.0F * PIf / 180.0F,
    67.5F * PIf / 180.0F, -67.5F * PIf / 180.0F,
    90.0F * PIf / 180.0F, -90.0F * PIf / 180.0F,
    112.5F * PIf / 180.0F, -112.5F * PIf / 180.0F,
    135.0F * PIf / 180.0F, -135.0F * PIf / 180.0F,
    157.5F * PIf / 180.0F, -157.5F * PIf / 180.0F,
    PIf,
};
constexpr float kCornerClearanceEpsilon = 0.002F;

float planar_distance(Transform left, Transform right) {
    const float dx = right.position.x - left.position.x;
    const float dz = right.position.z - left.position.z;
    return std::sqrt(dx * dx + dz * dz);
}

bool finite_transform(const Transform& value) {
    const float components[]{
        value.position.x, value.position.y, value.position.z,
        value.rotation.w, value.rotation.x,
        value.rotation.y, value.rotation.z,
    };
    for (float component : components) {
        if (!std::isfinite(component)) return false;
    }
    return true;
}

bool edge_is_clear(
    Transform start,
    Transform end,
    const InteractionTarget& target,
    const std::vector<PickNavigationObstacle>& obstacles,
    const PickSlotConfig& config) {
    PickSlotConfig edge_config = config;
    edge_config.maximum_direct_travel_m = std::max(
        config.maximum_direct_travel_m,
        planar_distance(start, end) + config.travel_tolerance_m);
    return revalidate_frozen_pick_slot(
               start, end, target, obstacles, edge_config) ==
        PickSlotReason::None;
}

std::vector<Transform> subdivide_path(
    const std::vector<Transform>& path,
    float maximum_segment_m) {
    if (path.empty() || !(maximum_segment_m > 0.0F)) return {};
    std::vector<Transform> result{path.front()};
    for (size_t index = 1U; index < path.size(); ++index) {
        const Transform start = path[index - 1U];
        const Transform end = path[index];
        const float distance = planar_distance(start, end);
        const size_t segment_count = std::max<size_t>(
            1U,
            static_cast<size_t>(std::ceil(distance / maximum_segment_m)));
        for (size_t segment = 1U; segment <= segment_count; ++segment) {
            const float alpha = static_cast<float>(segment) /
                static_cast<float>(segment_count);
            Transform sample = start;
            sample.position = start.position +
                alpha * (end.position - start.position);
            sample.rotation = end.rotation;
            result.push_back(sample);
        }
    }
    return result;
}

std::vector<Transform> visibility_nodes(
    Transform root,
    Transform entry,
    const std::vector<PickNavigationObstacle>& obstacles,
    const PickSlotConfig& config) {
    std::vector<Transform> nodes{root, entry};
    const float envelope = config.obstacle_root_radius_m +
        config.obstacle_safety_margin_m + kCornerClearanceEpsilon;
    for (const PickNavigationObstacle& obstacle : obstacles) {
        const float half_x = 0.5F * obstacle.size_world.x + envelope;
        const float half_z = 0.5F * obstacle.size_world.z + envelope;
        const float signs[][2]{{-1.0F, -1.0F}, {-1.0F, 1.0F},
                               {1.0F, -1.0F}, {1.0F, 1.0F}};
        for (const auto& sign : signs) {
            Transform corner = root;
            corner.position.x = obstacle.center_world.x + sign[0] * half_x;
            corner.position.z = obstacle.center_world.z + sign[1] * half_z;
            nodes.push_back(corner);
        }
    }
    return nodes;
}

std::vector<Transform> shortest_visibility_path(
    Transform root,
    Transform entry,
    const InteractionTarget& target,
    const std::vector<PickNavigationObstacle>& obstacles,
    const PickSlotConfig& config) {
    if (edge_is_clear(root, entry, target, obstacles, config)) {
        return {root, entry};
    }
    std::vector<Transform> nodes = visibility_nodes(
        root, entry, obstacles, config);
    const size_t count = nodes.size();
    std::vector<double> distance(
        count, std::numeric_limits<double>::infinity());
    std::vector<size_t> parent(count, count);
    std::vector<bool> visited(count, false);
    distance[0] = 0.0;
    for (size_t iteration = 0U; iteration < count; ++iteration) {
        size_t current = count;
        for (size_t node = 0U; node < count; ++node) {
            if (visited[node]) continue;
            if (current == count || distance[node] < distance[current] ||
                (distance[node] == distance[current] && node < current)) {
                current = node;
            }
        }
        if (current == count || !std::isfinite(distance[current])) break;
        if (current == 1U) break;
        visited[current] = true;
        for (size_t next = 0U; next < count; ++next) {
            if (next == current || visited[next] ||
                !edge_is_clear(
                    nodes[current], nodes[next], target, obstacles, config)) {
                continue;
            }
            const double candidate = distance[current] +
                planar_distance(nodes[current], nodes[next]);
            if (candidate < distance[next] ||
                (candidate == distance[next] && current < parent[next])) {
                distance[next] = candidate;
                parent[next] = current;
            }
        }
    }
    if (!std::isfinite(distance[1])) return {};
    std::vector<Transform> reversed;
    size_t node = 1U;
    while (true) {
        reversed.push_back(nodes[node]);
        if (node == 0U) break;
        node = parent[node];
        if (node == count) return {};
    }
    std::reverse(reversed.begin(), reversed.end());
    for (size_t index = 0U; index + 1U < reversed.size(); ++index) {
        const vec3 delta = reversed[index + 1U].position -
            reversed[index].position;
        reversed[index].rotation = quat_from_angle_axis(
            std::atan2(delta.x, delta.z), vec3(0.0F, 1.0F, 0.0F));
    }
    reversed.back().rotation = entry.rotation;
    return subdivide_path(reversed, config.maximum_direct_travel_m);
}

Transform entry_candidate(
    Transform root,
    const InteractionTarget& target,
    float radius,
    float offset) {
    const float dx = root.position.x - target.object_world.position.x;
    const float dz = root.position.z - target.object_world.position.z;
    const float length = std::sqrt(dx * dx + dz * dz);
    const float ux = dx / length;
    const float uz = dz / length;
    const float sine = std::sin(offset);
    const float cosine = std::cos(offset);
    const float direction_x = cosine * ux + sine * uz;
    const float direction_z = -sine * ux + cosine * uz;
    const float x = target.object_world.position.x + radius * direction_x;
    const float z = target.object_world.position.z + radius * direction_z;
    return {
        vec3(x, root.position.y, z),
        quat_from_angle_axis(
            std::atan2(
                target.object_world.position.x - x,
                target.object_world.position.z - z),
            vec3(0.0F, 1.0F, 0.0F)),
    };
}

}  // namespace

FunnelApproachPlan plan_funnel_approach(
    Transform frozen_root,
    const InteractionTarget& target,
    const std::vector<PickNavigationObstacle>& obstacles,
    const FunnelCaptureConfig& capture) {
    FunnelApproachPlan best{};
    if (!finite_transform(frozen_root) || !finite_transform(target.object_world) ||
        !(capture.minimum_object_radius_m > 0.0F) ||
        capture.maximum_object_radius_m < capture.minimum_object_radius_m) {
        return best;
    }
    const float dx = frozen_root.position.x - target.object_world.position.x;
    const float dz = frozen_root.position.z - target.object_world.position.z;
    const float root_radius = std::sqrt(dx * dx + dz * dz);
    if (!(root_radius > 1.0e-6F) ||
        root_radius > capture.maximum_activation_object_radius_m) {
        return best;
    }
    const float radius = std::clamp(
        root_radius,
        capture.minimum_object_radius_m,
        capture.maximum_object_radius_m);
    float best_length = std::numeric_limits<float>::infinity();
    for (float offset : kCandidateOffsetRadians) {
        const Transform entry = entry_candidate(
            frozen_root, target, radius, offset);
        std::vector<Transform> path = shortest_visibility_path(
            frozen_root, entry, target, obstacles, capture.route);
        if (path.empty()) continue;
        float length = 0.0F;
        for (size_t index = 1U; index < path.size(); ++index) {
            length += planar_distance(path[index - 1U], path[index]);
        }
        if (!best.feasible || length < best_length) {
            best.feasible = true;
            best.entry_world = entry;
            best.waypoints_world = std::move(path);
            best_length = length;
        }
    }
    return best;
}

}  // namespace interaction
