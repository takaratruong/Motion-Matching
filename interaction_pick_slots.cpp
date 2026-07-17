#include "interaction_pick_slots.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>

namespace interaction {

namespace detail {

struct PickRouteEvaluation {
    PickSlotReason reason = PickSlotReason::InvalidGeometry;
    int32_t obstacle_index = -1;
};

PickRouteEvaluation evaluate_pick_route(
    const Transform& route_start_world,
    const Transform& clearance_endpoint_world,
    const InteractionTarget& target,
    const std::vector<PickNavigationObstacle>& obstacles,
    const PickSlotConfig& config);

}  // namespace detail

namespace {

constexpr float kNearZeroRouteMetres = 1.0e-5F;
constexpr float kTableSlabEpsilon = 1.0e-7F;
constexpr double kQuantizationScale = 1000.0;
constexpr double kUint64ExclusiveUpperBound =
    18446744073709551616.0;
constexpr size_t kMaximumSegmentAabbBreakpoints = 8U;

bool finite_float(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value), "float must be binary32");
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

bool finite_vec3(vec3 value) {
    return finite_float(value.x) &&
        finite_float(value.y) &&
        finite_float(value.z);
}

bool finite_quat(quat value) {
    return finite_float(value.w) &&
        finite_float(value.x) &&
        finite_float(value.y) &&
        finite_float(value.z);
}

bool finite_transform(const Transform& value) {
    return finite_vec3(value.position) && finite_quat(value.rotation);
}

bool nonnegative_finite_vec3(vec3 value) {
    return finite_vec3(value) &&
        value.x >= 0.0F &&
        value.y >= 0.0F &&
        value.z >= 0.0F;
}

bool positive_finite_vec3(vec3 value) {
    return finite_vec3(value) &&
        value.x > 0.0F &&
        value.y > 0.0F &&
        value.z > 0.0F;
}

bool half_up_key(float value, uint64_t& key) {
    if (!finite_float(value) || value < 0.0F) return false;

    const double shifted =
        static_cast<double>(value) * kQuantizationScale + 0.5;
    if (!(shifted < kUint64ExclusiveUpperBound)) return false;

    key = static_cast<uint64_t>(std::floor(shifted));
    return true;
}

bool valid_config(const PickSlotConfig& config) {
    const float values[] = {
        config.maximum_direct_travel_m,
        config.travel_tolerance_m,
        config.table_root_expansion_m,
        config.obstacle_root_radius_m,
        config.obstacle_safety_margin_m,
    };
    uint64_t ignored_key = 0U;
    for (float value : values) {
        if (!finite_float(value) ||
            !(value > 0.0F) ||
            !half_up_key(value, ignored_key)) {
            return false;
        }
    }

    const float maximum_route =
        config.maximum_direct_travel_m + config.travel_tolerance_m;
    const float obstacle_envelope =
        config.obstacle_root_radius_m + config.obstacle_safety_margin_m;
    return half_up_key(maximum_route, ignored_key) &&
        half_up_key(obstacle_envelope, ignored_key);
}

bool planar_forward_and_yaw(
    quat rotation,
    vec3& forward,
    float& yaw) {
    if (!finite_quat(rotation)) return false;

    const vec3 projected = quat_mul_vec3(
        rotation, vec3(0.0F, 0.0F, 1.0F));
    if (!finite_vec3(projected)) return false;

    const double planar_length_squared =
        static_cast<double>(projected.x) * projected.x +
        static_cast<double>(projected.z) * projected.z;
    if (!(planar_length_squared > 0.0)) return false;

    const double inverse_length =
        1.0 / std::sqrt(planar_length_squared);
    forward = vec3(
        static_cast<float>(projected.x * inverse_length),
        0.0F,
        static_cast<float>(projected.z * inverse_length));
    yaw = std::atan2(forward.x, forward.z);
    return finite_vec3(forward) && finite_float(yaw);
}

bool planar_distance(vec3 left, vec3 right, float& distance) {
    if (!finite_vec3(left) || !finite_vec3(right)) return false;

    const double delta_x =
        static_cast<double>(right.x) - static_cast<double>(left.x);
    const double delta_z =
        static_cast<double>(right.z) - static_cast<double>(left.z);
    const double distance_double = std::sqrt(
        delta_x * delta_x + delta_z * delta_z);
    if (distance_double >
        static_cast<double>(std::numeric_limits<float>::max())) {
        return false;
    }

    distance = static_cast<float>(distance_double);
    return finite_float(distance);
}

bool world_bounds_center(
    const InteractionTarget& target,
    vec3& center_world) {
    if (!finite_transform(target.object_world) ||
        !finite_vec3(target.object_bounds.center_object) ||
        !nonnegative_finite_vec3(target.object_bounds.half_extents_object)) {
        return false;
    }

    const vec3 rotated_center = quat_mul_vec3(
        target.object_world.rotation,
        target.object_bounds.center_object);
    if (!finite_vec3(rotated_center)) return false;

    center_world = vec3(
        target.object_world.position.x + rotated_center.x,
        target.object_world.position.y + rotated_center.y,
        target.object_world.position.z + rotated_center.z);
    return finite_vec3(center_world);
}

bool map_and_measure_slot(
    Transform live_root,
    const InteractionTarget& target,
    const GraspInteractionSlot& slot,
    MappedPickSlot& candidate) {
    candidate = MappedPickSlot{};
    candidate.id = slot.id;
    candidate.obstacle_index = -1;

    if (slot.id == 0U ||
        !finite_transform(live_root) ||
        !finite_transform(target.object_world) ||
        !finite_float(slot.root_x_object_m) ||
        !finite_float(slot.root_z_object_m) ||
        !finite_float(slot.root_yaw_object_radians)) {
        return false;
    }

    vec3 object_forward{};
    float object_yaw = 0.0F;
    if (!planar_forward_and_yaw(
            target.object_world.rotation,
            object_forward,
            object_yaw)) {
        return false;
    }
    const vec3 object_right(
        object_forward.z, 0.0F, -object_forward.x);

    const vec3 mapped_position(
        target.object_world.position.x +
            slot.root_x_object_m * object_right.x +
            slot.root_z_object_m * object_forward.x,
        live_root.position.y,
        target.object_world.position.z +
            slot.root_x_object_m * object_right.z +
            slot.root_z_object_m * object_forward.z);
    if (!finite_vec3(mapped_position)) return false;

    const float unwrapped_yaw =
        object_yaw + slot.root_yaw_object_radians;
    if (!finite_float(unwrapped_yaw)) return false;
    const float mapped_yaw = std::atan2(
        std::sin(unwrapped_yaw), std::cos(unwrapped_yaw));
    if (!finite_float(mapped_yaw)) return false;

    candidate.root_world = Transform{
        mapped_position,
        quat_from_angle_axis(
            mapped_yaw, vec3(0.0F, 1.0F, 0.0F)),
    };
    if (!finite_transform(candidate.root_world) ||
        !planar_distance(
            live_root.position,
            mapped_position,
            candidate.route_length_m)) {
        return false;
    }

    vec3 live_forward{};
    float live_yaw = 0.0F;
    if (!planar_forward_and_yaw(
            live_root.rotation, live_forward, live_yaw)) {
        return false;
    }

    float desired_yaw = mapped_yaw;
    if (candidate.route_length_m > kNearZeroRouteMetres) {
        desired_yaw = std::atan2(
            mapped_position.x - live_root.position.x,
            mapped_position.z - live_root.position.z);
    }
    if (!finite_float(desired_yaw)) return false;

    const float heading_delta = desired_yaw - live_yaw;
    if (!finite_float(heading_delta)) return false;
    candidate.heading_change_radians = std::fabs(std::atan2(
        std::sin(heading_delta), std::cos(heading_delta)));
    if (!finite_float(candidate.heading_change_radians)) return false;

    vec3 bounds_center{};
    if (!world_bounds_center(target, bounds_center) ||
        !planar_distance(
            mapped_position,
            target.object_world.position,
            candidate.object_origin_distance_m) ||
        !planar_distance(
            mapped_position,
            bounds_center,
            candidate.object_bounds_center_distance_m) ||
        !half_up_key(
            candidate.route_length_m,
            candidate.route_millimetres) ||
        !half_up_key(
            candidate.heading_change_radians,
            candidate.heading_milliradians)) {
        return false;
    }

    candidate.reason = PickSlotReason::None;
    return true;
}

bool within_travel_envelope(
    float route_length_m,
    const PickSlotConfig& config) {
    const float maximum_route =
        config.maximum_direct_travel_m + config.travel_tolerance_m;
    return finite_float(maximum_route) && route_length_m <= maximum_route;
}

bool better_candidate(
    const MappedPickSlot& left,
    const MappedPickSlot& right) {
    return std::tuple<uint64_t, uint64_t, uint32_t>(
               left.route_millimetres,
               left.heading_milliradians,
               left.id) <
        std::tuple<uint64_t, uint64_t, uint32_t>(
               right.route_millimetres,
               right.heading_milliradians,
               right.id);
}

PickSlotReason no_winner_reason(
    const std::vector<MappedPickSlot>& candidates) {
    bool saw_invalid = false;
    bool saw_blocked = false;
    for (const MappedPickSlot& candidate : candidates) {
        saw_invalid = saw_invalid ||
            candidate.reason == PickSlotReason::InvalidGeometry;
        saw_blocked = saw_blocked ||
            candidate.reason == PickSlotReason::TableBlocked ||
            candidate.reason == PickSlotReason::ObstacleBlocked;
    }
    if (saw_invalid) return PickSlotReason::InvalidGeometry;
    if (saw_blocked) return PickSlotReason::AllSlotsBlocked;
    return PickSlotReason::OutsideTravelEnvelope;
}

bool valid_obstacle(const PickNavigationObstacle& obstacle) {
    return finite_vec3(obstacle.center_world) &&
        positive_finite_vec3(obstacle.size_world);
}

bool closed_segment_intersects_slab(
    double start,
    double end,
    double half_extent,
    double& entry_parameter,
    double& exit_parameter) {
    const double direction = end - start;
    if (std::fabs(direction) <=
        static_cast<double>(kTableSlabEpsilon)) {
        if ((start < -half_extent && end < -half_extent) ||
            (start > half_extent && end > half_extent)) {
            return false;
        }
        if (start >= -half_extent && start <= half_extent &&
            end >= -half_extent && end <= half_extent) {
            return true;
        }
    }

    double first = (-half_extent - start) / direction;
    double second = (half_extent - start) / direction;
    if (first > second) std::swap(first, second);
    entry_parameter = std::max(entry_parameter, first);
    exit_parameter = std::min(exit_parameter, second);
    return entry_parameter <= exit_parameter;
}

bool route_intersects_expanded_table(
    vec3 route_start,
    vec3 route_end,
    const InteractionTarget& target,
    const PickSlotConfig& config,
    bool& intersects) {
    intersects = false;
    if (!finite_vec3(route_start) ||
        !finite_vec3(route_end) ||
        !finite_transform(target.table_world) ||
        !positive_finite_vec3(target.table_size)) {
        return false;
    }

    vec3 table_forward{};
    float table_yaw = 0.0F;
    if (!planar_forward_and_yaw(
            target.table_world.rotation,
            table_forward,
            table_yaw)) {
        return false;
    }
    const vec3 table_right(
        table_forward.z, 0.0F, -table_forward.x);

    const float expanded_half_x =
        0.5F * target.table_size.x + config.table_root_expansion_m;
    const float expanded_half_z =
        0.5F * target.table_size.z + config.table_root_expansion_m;
    if (!finite_float(expanded_half_x) ||
        !finite_float(expanded_half_z) ||
        !(expanded_half_x > 0.0F) ||
        !(expanded_half_z > 0.0F)) {
        return false;
    }

    const double start_delta_x =
        static_cast<double>(route_start.x) -
        static_cast<double>(target.table_world.position.x);
    const double start_delta_z =
        static_cast<double>(route_start.z) -
        static_cast<double>(target.table_world.position.z);
    const double end_delta_x =
        static_cast<double>(route_end.x) -
        static_cast<double>(target.table_world.position.x);
    const double end_delta_z =
        static_cast<double>(route_end.z) -
        static_cast<double>(target.table_world.position.z);

    const double start_local_x =
        start_delta_x * static_cast<double>(table_right.x) +
        start_delta_z * static_cast<double>(table_right.z);
    const double start_local_z =
        start_delta_x * static_cast<double>(table_forward.x) +
        start_delta_z * static_cast<double>(table_forward.z);
    const double end_local_x =
        end_delta_x * static_cast<double>(table_right.x) +
        end_delta_z * static_cast<double>(table_right.z);
    const double end_local_z =
        end_delta_x * static_cast<double>(table_forward.x) +
        end_delta_z * static_cast<double>(table_forward.z);

    double entry_parameter = 0.0;
    double exit_parameter = 1.0;
    if (!closed_segment_intersects_slab(
            start_local_x,
            end_local_x,
            static_cast<double>(expanded_half_x),
            entry_parameter,
            exit_parameter)) {
        return true;
    }
    if (!closed_segment_intersects_slab(
            start_local_z,
            end_local_z,
            static_cast<double>(expanded_half_z),
            entry_parameter,
            exit_parameter)) {
        return true;
    }

    intersects = entry_parameter <= 1.0 && exit_parameter >= 0.0;
    return true;
}

struct WorldAabb {
    std::array<double, 3U> minimum{};
    std::array<double, 3U> maximum{};
};

WorldAabb world_aabb(const PickNavigationObstacle& obstacle) {
    const std::array<double, 3U> center{
        static_cast<double>(obstacle.center_world.x),
        static_cast<double>(obstacle.center_world.y),
        static_cast<double>(obstacle.center_world.z),
    };
    const std::array<double, 3U> half_extent{
        0.5 * static_cast<double>(obstacle.size_world.x),
        0.5 * static_cast<double>(obstacle.size_world.y),
        0.5 * static_cast<double>(obstacle.size_world.z),
    };

    WorldAabb bounds{};
    for (size_t axis = 0U; axis < center.size(); ++axis) {
        bounds.minimum[axis] = center[axis] - half_extent[axis];
        bounds.maximum[axis] = center[axis] + half_extent[axis];
    }
    return bounds;
}

double squared_distance_at_parameter(
    const std::array<double, 3U>& start,
    const std::array<double, 3U>& direction,
    const WorldAabb& bounds,
    double parameter) {
    double squared_distance = 0.0;
    for (size_t axis = 0U; axis < start.size(); ++axis) {
        const double coordinate =
            start[axis] + parameter * direction[axis];
        double gap = 0.0;
        if (coordinate < bounds.minimum[axis]) {
            gap = bounds.minimum[axis] - coordinate;
        } else if (coordinate > bounds.maximum[axis]) {
            gap = coordinate - bounds.maximum[axis];
        }
        squared_distance += gap * gap;
    }
    return squared_distance;
}

double closed_segment_aabb_squared_distance(
    vec3 route_start,
    vec3 route_end,
    const PickNavigationObstacle& obstacle) {
    const std::array<double, 3U> start{
        static_cast<double>(route_start.x),
        static_cast<double>(route_start.y),
        static_cast<double>(route_start.z),
    };
    const std::array<double, 3U> end{
        static_cast<double>(route_end.x),
        static_cast<double>(route_end.y),
        static_cast<double>(route_end.z),
    };
    std::array<double, 3U> direction{};
    for (size_t axis = 0U; axis < start.size(); ++axis) {
        direction[axis] = end[axis] - start[axis];
    }
    const WorldAabb bounds = world_aabb(obstacle);

    std::array<double, kMaximumSegmentAabbBreakpoints> breakpoints{};
    size_t breakpoint_count = 0U;
    breakpoints[breakpoint_count++] = 0.0;
    breakpoints[breakpoint_count++] = 1.0;
    for (size_t axis = 0U; axis < start.size(); ++axis) {
        if (direction[axis] == 0.0) continue;
        const double planes[] = {
            bounds.minimum[axis],
            bounds.maximum[axis],
        };
        for (double plane : planes) {
            const double parameter =
                (plane - start[axis]) / direction[axis];
            if (parameter > 0.0 && parameter < 1.0) {
                breakpoints[breakpoint_count++] = parameter;
            }
        }
    }

    std::sort(
        breakpoints.begin(),
        breakpoints.begin() +
            static_cast<std::ptrdiff_t>(breakpoint_count));
    size_t unique_count = 0U;
    for (size_t index = 0U; index < breakpoint_count; ++index) {
        if (unique_count == 0U ||
            breakpoints[index] != breakpoints[unique_count - 1U]) {
            breakpoints[unique_count++] = breakpoints[index];
        }
    }

    double minimum_squared_distance =
        std::numeric_limits<double>::max();
    for (size_t interval = 0U; interval + 1U < unique_count; ++interval) {
        const double lower = breakpoints[interval];
        const double upper = breakpoints[interval + 1U];
        const double midpoint = lower + 0.5 * (upper - lower);

        double quadratic_coefficient = 0.0;
        double linear_coefficient = 0.0;
        for (size_t axis = 0U; axis < start.size(); ++axis) {
            const double midpoint_coordinate =
                start[axis] + midpoint * direction[axis];
            double offset = 0.0;
            double slope = 0.0;
            if (midpoint_coordinate < bounds.minimum[axis]) {
                offset = bounds.minimum[axis] - start[axis];
                slope = -direction[axis];
            } else if (midpoint_coordinate > bounds.maximum[axis]) {
                offset = start[axis] - bounds.maximum[axis];
                slope = direction[axis];
            }
            quadratic_coefficient += slope * slope;
            linear_coefficient += 2.0 * offset * slope;
        }

        minimum_squared_distance = std::min(
            minimum_squared_distance,
            squared_distance_at_parameter(
                start, direction, bounds, lower));
        minimum_squared_distance = std::min(
            minimum_squared_distance,
            squared_distance_at_parameter(
                start, direction, bounds, upper));
        if (quadratic_coefficient > 0.0) {
            double stationary_parameter =
                -linear_coefficient /
                (2.0 * quadratic_coefficient);
            stationary_parameter = std::max(
                lower, std::min(upper, stationary_parameter));
            minimum_squared_distance = std::min(
                minimum_squared_distance,
                squared_distance_at_parameter(
                    start,
                    direction,
                    bounds,
                    stationary_parameter));
        }
    }
    return minimum_squared_distance;
}

}  // namespace

detail::PickRouteEvaluation detail::evaluate_pick_route(
    const Transform& route_start_world,
    const Transform& clearance_endpoint_world,
    const InteractionTarget& target,
    const std::vector<PickNavigationObstacle>& obstacles,
    const PickSlotConfig& config) {
    PickRouteEvaluation evaluation{};
    if (!finite_transform(route_start_world) ||
        !finite_transform(clearance_endpoint_world)) {
        return evaluation;
    }

    for (size_t index = 0U; index < obstacles.size(); ++index) {
        if (!valid_obstacle(obstacles[index])) {
            evaluation.obstacle_index = static_cast<int32_t>(index);
            return evaluation;
        }
    }

    bool table_intersects = false;
    if (!route_intersects_expanded_table(
            route_start_world.position,
            clearance_endpoint_world.position,
            target,
            config,
            table_intersects)) {
        return evaluation;
    }
    if (table_intersects) {
        evaluation.reason = PickSlotReason::TableBlocked;
        return evaluation;
    }

    const float obstacle_envelope =
        config.obstacle_root_radius_m +
        config.obstacle_safety_margin_m;
    if (!finite_float(obstacle_envelope) ||
        !(obstacle_envelope > 0.0F)) {
        return evaluation;
    }
    const double squared_obstacle_envelope =
        static_cast<double>(obstacle_envelope) *
        static_cast<double>(obstacle_envelope);
    for (size_t index = 0U; index < obstacles.size(); ++index) {
        const double squared_distance =
            closed_segment_aabb_squared_distance(
                route_start_world.position,
                clearance_endpoint_world.position,
                obstacles[index]);
        if (!(squared_distance > squared_obstacle_envelope)) {
            evaluation.reason = PickSlotReason::ObstacleBlocked;
            evaluation.obstacle_index = static_cast<int32_t>(index);
            return evaluation;
        }
    }

    evaluation.reason = PickSlotReason::None;
    return evaluation;
}

PickSlotSelection select_pick_slot(
    Transform live_root,
    const InteractionTarget& target,
    const GraspAffordance& affordance,
    const std::vector<PickNavigationObstacle>& obstacles,
    const PickSlotConfig& config) {
    PickSlotSelection selection{};
    if (!valid_config(config)) {
        selection.reason = PickSlotReason::InvalidGeometry;
        return selection;
    }
    if (affordance.interaction_slots.empty()) return selection;

    selection.ordered.reserve(affordance.interaction_slots.size());
    for (const GraspInteractionSlot& slot : affordance.interaction_slots) {
        MappedPickSlot candidate{};
        if (map_and_measure_slot(live_root, target, slot, candidate)) {
            if (!within_travel_envelope(candidate.route_length_m, config)) {
                candidate.reason = PickSlotReason::OutsideTravelEnvelope;
            } else {
                const detail::PickRouteEvaluation evaluation =
                    detail::evaluate_pick_route(
                        live_root,
                        candidate.root_world,
                        target,
                        obstacles,
                        config);
                candidate.reason = evaluation.reason;
                candidate.obstacle_index = evaluation.obstacle_index;
            }
        }
        selection.ordered.push_back(candidate);
    }

    for (size_t index = 0U; index < selection.ordered.size(); ++index) {
        if (selection.ordered[index].reason != PickSlotReason::None) continue;
        if (!selection.selected_index.has_value() ||
            better_candidate(
                selection.ordered[index],
                selection.ordered[*selection.selected_index])) {
            selection.selected_index = index;
        }
    }

    if (selection.selected_index.has_value()) {
        selection.reason = PickSlotReason::None;
    } else {
        selection.reason = no_winner_reason(selection.ordered);
    }
    return selection;
}

PickSlotReason revalidate_frozen_pick_slot(
    Transform live_root,
    Transform frozen_root,
    const InteractionTarget& target,
    const std::vector<PickNavigationObstacle>& obstacles,
    const PickSlotConfig& config) {
    if (!valid_config(config) ||
        !finite_transform(live_root) ||
        !finite_transform(frozen_root)) {
        return PickSlotReason::InvalidGeometry;
    }

    float remaining_route_length_m = 0.0F;
    if (!planar_distance(
            live_root.position,
            frozen_root.position,
            remaining_route_length_m)) {
        return PickSlotReason::InvalidGeometry;
    }
    uint64_t remaining_route_millimetres = 0U;
    if (!half_up_key(
            remaining_route_length_m,
            remaining_route_millimetres)) {
        return PickSlotReason::InvalidGeometry;
    }
    if (!within_travel_envelope(remaining_route_length_m, config)) {
        return PickSlotReason::OutsideTravelEnvelope;
    }

    Transform clearance_endpoint = frozen_root;
    clearance_endpoint.position.y = live_root.position.y;
    return detail::evaluate_pick_route(
               live_root,
               clearance_endpoint,
               target,
               obstacles,
               config)
        .reason;
}

}  // namespace interaction
