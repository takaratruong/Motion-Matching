#include "interaction_hand_trajectories.h"

#include "interaction_pose.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace interaction {
namespace {

constexpr uint8_t kReachPhase = 1U;
constexpr uint8_t kContactPhase = 2U;
constexpr uint8_t kLiftPhase = 3U;
constexpr float kQuaternionEpsilon = 1.0e-6F;

bool finite(float value) {
    return std::isfinite(value);
}

bool finite(vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(quat value) {
    return finite(value.w) && finite(value.x) &&
           finite(value.y) && finite(value.z);
}

bool positive(vec3 value) {
    return finite(value) && value.x > 0.0F && value.y > 0.0F && value.z > 0.0F;
}

bool valid_rotation(quat value) {
    if (!finite(value)) return false;
    const float squared = value.w * value.w + value.x * value.x +
        value.y * value.y + value.z * value.z;
    return squared > kQuaternionEpsilon;
}

bool valid_box(const OrientedBox& box) {
    return finite(box.world.position) && valid_rotation(box.world.rotation) &&
           positive(box.dimensions);
}

vec3 point_in_box(const OrientedBox& box, vec3 point) {
    return compose(
        inverse(Transform{box.world.position, quat_normalize(box.world.rotation)}),
        Transform{point, quat()}).position;
}

bool sphere_intersects_box(
    vec3 center,
    float radius,
    const OrientedBox& box) {
    const vec3 local = point_in_box(box, center);
    const vec3 half = box.dimensions * 0.5F;
    const vec3 closest = clamp(local, -half, half);
    const vec3 offset = local - closest;
    return dot(offset, offset) <= radius * radius;
}

bool update_slab(
    float start,
    float delta,
    float half_extent,
    float& minimum_time,
    float& maximum_time) {
    constexpr float epsilon = 1.0e-8F;
    if (std::abs(delta) <= epsilon) {
        return start >= -half_extent && start <= half_extent;
    }
    float first = (-half_extent - start) / delta;
    float second = (half_extent - start) / delta;
    if (first > second) std::swap(first, second);
    minimum_time = std::max(minimum_time, first);
    maximum_time = std::min(maximum_time, second);
    return minimum_time <= maximum_time;
}

bool capsule_intersects_box(
    vec3 start,
    vec3 stop,
    float radius,
    const OrientedBox& box) {
    const vec3 local_start = point_in_box(box, start);
    const vec3 local_stop = point_in_box(box, stop);
    const vec3 delta = local_stop - local_start;
    const vec3 expanded = box.dimensions * 0.5F + radius;
    float minimum_time = 0.0F;
    float maximum_time = 1.0F;
    return update_slab(
               local_start.x, delta.x, expanded.x,
               minimum_time, maximum_time) &&
           update_slab(
               local_start.y, delta.y, expanded.y,
               minimum_time, maximum_time) &&
           update_slab(
               local_start.z, delta.z, expanded.z,
               minimum_time, maximum_time);
}

bool arm_intersects_box(
    vec3 wrist,
    vec3 elbow,
    const OrientedBox& box,
    const TrajectoryCollisionConfig& config) {
    return sphere_intersects_box(wrist, config.wrist_radius_m, box) ||
           capsule_intersects_box(
               elbow, wrist, config.forearm_radius_m, box);
}

vec3 read_vec3(const std::vector<float>& values, size_t index) {
    const size_t offset = 3U * index;
    return vec3(values.at(offset), values.at(offset + 1U), values.at(offset + 2U));
}

quat read_quat(const std::vector<float>& values, size_t index) {
    const size_t offset = 4U * index;
    return quat(
        values.at(offset), values.at(offset + 1U),
        values.at(offset + 2U), values.at(offset + 3U));
}

float quaternion_angle(quat left, quat right) {
    left = quat_normalize(left);
    right = quat_normalize(right);
    const float absolute_dot = std::min(1.0F, std::abs(
        left.w * right.w + left.x * right.x +
        left.y * right.y + left.z * right.z));
    return 2.0F * std::acos(absolute_dot);
}

float yaw_radians(quat rotation) {
    const vec3 facing = quat_mul_vec3(
        quat_normalize(rotation), vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(facing.x, facing.z);
}

float shortest_angle(float angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

Transform upright_scene_alignment(
    const Transform& source_object,
    const Transform& target_object) {
    const float yaw = shortest_angle(
        yaw_radians(target_object.rotation) -
        yaw_radians(source_object.rotation));
    const quat rotation = quat_from_angle_axis(
        yaw, vec3(0.0F, 1.0F, 0.0F));
    const vec3 rotated_source = quat_mul_vec3(
        rotation, source_object.position);
    return {
        vec3(
            target_object.position.x - rotated_source.x,
            0.0F,
            target_object.position.z - rotated_source.z),
        rotation,
    };
}

float log_dimension_error(vec3 source, vec3 target) {
    return std::max({
        std::abs(std::log(source.x / target.x)),
        std::abs(std::log(source.y / target.y)),
        std::abs(std::log(source.z / target.z)),
    });
}

struct ClipPhases {
    int32_t reach = -1;
    int32_t contact = -1;
    int32_t lift = -1;
};

ClipPhases clip_phases(const Database& database, size_t clip) {
    ClipPhases result{};
    const int32_t start = database.range_starts.at(clip);
    const int32_t stop = database.range_stops.at(clip);
    if (start < 0 || stop <= start || static_cast<uint32_t>(stop) > database.frame_count) {
        return result;
    }
    for (int32_t frame = start; frame < stop; ++frame) {
        const uint8_t phase = database.phases.at(static_cast<size_t>(frame));
        if (phase == kReachPhase && result.reach < 0) result.reach = frame;
        if (phase == kContactPhase && result.contact < 0) result.contact = frame;
        if (phase == kLiftPhase && result.lift < 0) result.lift = frame;
    }
    if (!(result.reach >= start && result.reach < result.contact &&
          result.contact < result.lift && result.lift < stop)) {
        return ClipPhases{};
    }
    return result;
}

Transform object_transform(const Database& database, int32_t frame) {
    return {
        read_vec3(database.object_positions, static_cast<size_t>(frame)),
        quat_normalize(read_quat(
            database.object_rotations, static_cast<size_t>(frame))),
    };
}

size_t wrist_index(Hand hand) {
    return hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : static_cast<size_t>(g1_skeleton::RightWrist);
}

size_t elbow_index(Hand hand) {
    return hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftElbow)
        : static_cast<size_t>(g1_skeleton::RightElbow);
}

Transform hand_transform(const WorldPose& world, Hand hand) {
    const size_t wrist = wrist_index(hand);
    return {world.positions[wrist], world.rotations[wrist]};
}

void validate_query_and_config(
    const HandTrajectoryQuery& query,
    const HandTrajectoryConfig& config) {
    if (!finite(query.object_world.position) ||
        !valid_rotation(query.object_world.rotation) ||
        !positive(query.object_dimensions) ||
        !finite(query.grasp_world_position) ||
        (query.grasp_world_rotation.has_value() &&
         !valid_rotation(*query.grasp_world_rotation)) ||
        !finite(config.maximum_grasp_position_error_m) ||
        config.maximum_grasp_position_error_m < 0.0F ||
        !finite(config.maximum_grasp_orientation_error_radians) ||
        config.maximum_grasp_orientation_error_radians < 0.0F ||
        !finite(config.maximum_log_dimension_error) ||
        config.maximum_log_dimension_error < 0.0F ||
        config.maximum_compatible_clips == 0U ||
        config.maximum_compatible_clips > 4096U) {
        throw std::invalid_argument("invalid hand trajectory query or config");
    }
}

void validate_trajectory(const HandTrajectory& trajectory) {
    if (trajectory.hands_in_source_object.empty() ||
        trajectory.hands_in_source_object.size() !=
            trajectory.elbows_in_source_object.size() ||
        trajectory.contact_point >= trajectory.hands_in_source_object.size() ||
        !finite(trajectory.source_object.position) ||
        !valid_rotation(trajectory.source_object.rotation)) {
        throw std::invalid_argument("invalid hand trajectory");
    }
}

}  // namespace

std::vector<HandTrajectory> select_hand_trajectories(
    const Database& database,
    const HandTrajectoryQuery& query,
    const HandTrajectoryConfig& config) {
    validate_query_and_config(query, config);
    const Transform target_object{
        query.object_world.position,
        quat_normalize(query.object_world.rotation)};

    std::vector<HandTrajectory> selected;
    for (size_t clip = 0; clip < database.clip_count; ++clip) {
        if (database.active_hands.at(clip) != static_cast<uint8_t>(query.hand)) {
            continue;
        }
        const ClipPhases phases = clip_phases(database, clip);
        if (phases.reach < 0) continue;
        const vec3 source_dimensions = read_vec3(database.object_dimensions, clip);
        if (!positive(source_dimensions)) {
            continue;
        }
        const float dimension_error = log_dimension_error(
            source_dimensions, query.object_dimensions);
        if (dimension_error > config.maximum_log_dimension_error) {
            continue;
        }
        HandTrajectory trajectory{};
        trajectory.clip = static_cast<int32_t>(clip);
        trajectory.reach_frame = phases.reach;
        trajectory.contact_frame = phases.contact;
        trajectory.lift_frame = phases.lift;
        trajectory.contact_point = static_cast<size_t>(phases.contact - phases.reach);
        trajectory.source_object = object_transform(
            database, phases.contact - 1);
        const Transform source_from_world = inverse(trajectory.source_object);
        for (int32_t frame = phases.reach; frame <= phases.lift; ++frame) {
            const WorldPose world = world_pose(pose_at_frame(database, frame));
            trajectory.hands_in_source_object.push_back(compose(
                source_from_world,
                hand_transform(world, query.hand)));
            trajectory.elbows_in_source_object.push_back(compose(
                source_from_world,
                Transform{world.positions[elbow_index(query.hand)], quat()}
            ).position);
        }
        const Transform alignment = upright_scene_alignment(
            trajectory.source_object, target_object);
        const Transform source_contact = compose(
            trajectory.source_object,
            trajectory.hands_in_source_object[trajectory.contact_point]);
        const Transform mapped_contact = compose(alignment, source_contact);
        const float position_error = length(
            mapped_contact.position - query.grasp_world_position);
        const float orientation_error = query.grasp_world_rotation.has_value()
            ? quaternion_angle(
                mapped_contact.rotation, *query.grasp_world_rotation)
            : 0.0F;
        if (position_error > config.maximum_grasp_position_error_m ||
            orientation_error >
                config.maximum_grasp_orientation_error_radians) {
            continue;
        }
        const float position_normalizer = std::max(
            config.maximum_grasp_position_error_m, 1.0e-6F);
        const float orientation_normalizer = std::max(
            config.maximum_grasp_orientation_error_radians, 1.0e-6F);
        const float dimension_normalizer = std::max(
            config.maximum_log_dimension_error, 1.0e-6F);
        trajectory.cost =
            position_error / position_normalizer +
            orientation_error / orientation_normalizer +
            dimension_error / dimension_normalizer;
        selected.push_back(std::move(trajectory));
        if (selected.size() > config.maximum_compatible_clips) {
            throw std::length_error("compatible hand trajectory limit exceeded");
        }
    }
    std::sort(selected.begin(), selected.end(),
        [](const HandTrajectory& left, const HandTrajectory& right) {
            return std::tie(left.cost, left.clip) <
                   std::tie(right.cost, right.clip);
        });
    return selected;
}

Transform hand_trajectory_scene_alignment(
    const HandTrajectory& trajectory,
    const HandTrajectoryQuery& query) {
    validate_query_and_config(query, HandTrajectoryConfig{});
    validate_trajectory(trajectory);
    const Transform target_object{
        query.object_world.position,
        quat_normalize(query.object_world.rotation)};
    return upright_scene_alignment(trajectory.source_object, target_object);
}

MappedHandTrajectory map_hand_trajectory(
    const HandTrajectory& trajectory,
    const HandTrajectoryQuery& query) {
    const Transform mapping = hand_trajectory_scene_alignment(trajectory, query);
    MappedHandTrajectory mapped{};
    mapped.hands.reserve(trajectory.hands_in_source_object.size());
    mapped.elbows.reserve(trajectory.elbows_in_source_object.size());
    for (const Transform& hand : trajectory.hands_in_source_object) {
        mapped.hands.push_back(compose(
            mapping, compose(trajectory.source_object, hand)));
    }
    for (const vec3 elbow : trajectory.elbows_in_source_object) {
        mapped.elbows.push_back(compose(
            mapping,
            compose(
                trajectory.source_object,
                Transform{elbow, quat()})).position);
    }
    return mapped;
}

ShelfGeometry make_recorded_table_geometry(
    const Transform& table_world,
    vec3 table_dimensions,
    float leg_thickness_m) {
    const OrientedBox tabletop{
        {table_world.position, quat_normalize(table_world.rotation)},
        table_dimensions};
    const float underside_height =
        table_world.position.y - 0.5F * table_dimensions.y;
    if (!valid_box(tabletop) ||
        !finite(leg_thickness_m) || leg_thickness_m <= 0.0F ||
        table_dimensions.x <= 2.0F * leg_thickness_m ||
        table_dimensions.z <= 2.0F * leg_thickness_m ||
        !finite(underside_height) || underside_height <= 0.0F) {
        throw std::invalid_argument("invalid recorded table geometry");
    }

    ShelfGeometry geometry{};
    geometry.boxes[0] = tabletop;
    const float x_offset = 0.5F * table_dimensions.x - leg_thickness_m;
    const float z_offset = 0.5F * table_dimensions.z - leg_thickness_m;
    const float local_y =
        -0.5F * table_dimensions.y - 0.5F * underside_height;
    const std::array<vec3, 4> leg_positions{{
        vec3(-x_offset, local_y, -z_offset),
        vec3(x_offset, local_y, -z_offset),
        vec3(-x_offset, local_y, z_offset),
        vec3(x_offset, local_y, z_offset),
    }};
    for (size_t leg = 0U; leg < leg_positions.size(); ++leg) {
        geometry.boxes[leg + 1U] = {
            compose(tabletop.world, Transform{leg_positions[leg], quat()}),
            vec3(leg_thickness_m, underside_height, leg_thickness_m),
        };
    }
    return geometry;
}

TrajectoryFeasibility evaluate_trajectory_feasibility(
    const MappedHandTrajectory& trajectory,
    size_t contact_point,
    const OrientedBox& object,
    const ShelfGeometry& shelf,
    const TrajectoryCollisionConfig& config) {
    if (trajectory.hands.empty() ||
        trajectory.hands.size() != trajectory.elbows.size() ||
        contact_point >= trajectory.hands.size() ||
        !valid_box(object) ||
        !finite(config.wrist_radius_m) || config.wrist_radius_m < 0.0F ||
        !finite(config.forearm_radius_m) || config.forearm_radius_m < 0.0F) {
        throw std::invalid_argument("invalid trajectory collision query");
    }
    for (const OrientedBox& box : shelf.boxes) {
        if (!valid_box(box)) {
            throw std::invalid_argument("invalid shelf geometry");
        }
    }
    for (size_t sample = 0U; sample < contact_point; ++sample) {
        if (arm_intersects_box(
                trajectory.hands[sample].position,
                trajectory.elbows[sample], object, config)) {
            return {TrajectoryFeasibilityReason::ObjectCollision, sample};
        }
    }
    for (size_t sample = 0U; sample < trajectory.hands.size(); ++sample) {
        for (const OrientedBox& box : shelf.boxes) {
            if (arm_intersects_box(
                    trajectory.hands[sample].position,
                    trajectory.elbows[sample], box, config)) {
                return {TrajectoryFeasibilityReason::ShelfCollision, sample};
            }
        }
    }
    return {};
}

}  // namespace interaction
