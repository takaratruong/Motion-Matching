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

}  // namespace

std::vector<HandTrajectory> select_hand_trajectories(
    const Database& database,
    const HandTrajectoryQuery& query,
    const HandTrajectoryConfig& config) {
    validate_query_and_config(query, config);
    const Transform target_object{
        query.object_world.position,
        quat_normalize(query.object_world.rotation)};
    const vec3 target_grasp_object = compose(
        inverse(target_object),
        Transform{query.grasp_world_position, quat()}).position;
    const std::optional<quat> target_grasp_rotation_object =
        query.grasp_world_rotation.has_value()
        ? std::optional<quat>(quat_normalize(quat_mul(
            quat_inv(target_object.rotation),
            *query.grasp_world_rotation)))
        : std::nullopt;

    std::vector<HandTrajectory> selected;
    for (size_t clip = 0; clip < database.clip_count; ++clip) {
        if (database.active_hands.at(clip) != static_cast<uint8_t>(query.hand)) {
            continue;
        }
        const ClipPhases phases = clip_phases(database, clip);
        if (phases.reach < 0) continue;
        const vec3 source_grasp = read_vec3(database.grasp_positions_object, clip);
        const quat source_grasp_rotation = read_quat(
            database.grasp_rotations_object, clip);
        const vec3 source_dimensions = read_vec3(database.object_dimensions, clip);
        if (!finite(source_grasp) || !valid_rotation(source_grasp_rotation) ||
            !positive(source_dimensions)) {
            continue;
        }
        const float position_error = length(source_grasp - target_grasp_object);
        const float dimension_error = log_dimension_error(
            source_dimensions, query.object_dimensions);
        const float orientation_error = target_grasp_rotation_object.has_value()
            ? quaternion_angle(source_grasp_rotation, *target_grasp_rotation_object)
            : 0.0F;
        if (position_error > config.maximum_grasp_position_error_m ||
            dimension_error > config.maximum_log_dimension_error ||
            orientation_error > config.maximum_grasp_orientation_error_radians) {
            continue;
        }
        HandTrajectory trajectory{};
        trajectory.clip = static_cast<int32_t>(clip);
        trajectory.reach_frame = phases.reach;
        trajectory.contact_frame = phases.contact;
        trajectory.lift_frame = phases.lift;
        trajectory.contact_point = static_cast<size_t>(phases.contact - phases.reach);
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

MappedHandTrajectory map_hand_trajectory(
    const HandTrajectory& trajectory,
    const HandTrajectoryQuery& query) {
    validate_query_and_config(query, HandTrajectoryConfig{});
    if (trajectory.hands_in_source_object.empty() ||
        trajectory.hands_in_source_object.size() !=
            trajectory.elbows_in_source_object.size() ||
        trajectory.contact_point >= trajectory.hands_in_source_object.size()) {
        throw std::invalid_argument("invalid hand trajectory");
    }
    const Transform target_object{
        query.object_world.position,
        quat_normalize(query.object_world.rotation)};
    MappedHandTrajectory mapped{};
    mapped.hands.reserve(trajectory.hands_in_source_object.size());
    mapped.elbows.reserve(trajectory.elbows_in_source_object.size());
    for (const Transform& hand : trajectory.hands_in_source_object) {
        mapped.hands.push_back(compose(target_object, hand));
    }
    for (const vec3 elbow : trajectory.elbows_in_source_object) {
        mapped.elbows.push_back(compose(
            target_object, Transform{elbow, quat()}).position);
    }
    const Transform mapped_contact = mapped.hands[trajectory.contact_point];
    if (query.grasp_world_rotation.has_value()) {
        const Transform requested{
            query.grasp_world_position,
            quat_normalize(*query.grasp_world_rotation)};
        const Transform residual = compose(requested, inverse(mapped_contact));
        for (Transform& hand : mapped.hands) hand = compose(residual, hand);
        for (vec3& elbow : mapped.elbows) {
            elbow = compose(residual, Transform{elbow, quat()}).position;
        }
    } else {
        const vec3 translation =
            query.grasp_world_position - mapped_contact.position;
        for (Transform& hand : mapped.hands) {
            hand.position = hand.position + translation;
        }
        for (vec3& elbow : mapped.elbows) elbow = elbow + translation;
    }
    return mapped;
}

}  // namespace interaction
