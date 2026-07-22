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
constexpr float kSupportMetadataTolerance = 1.0e-4F;

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

bool torso_bone(size_t bone) {
    return bone == static_cast<size_t>(g1_skeleton::Hips) ||
        bone == static_cast<size_t>(g1_skeleton::Spine) ||
        bone == static_cast<size_t>(g1_skeleton::Spine1) ||
        bone == static_cast<size_t>(g1_skeleton::Spine2);
}

bool wrist_bone(size_t bone) {
    return bone == static_cast<size_t>(g1_skeleton::LeftWristRoll) ||
        bone == static_cast<size_t>(g1_skeleton::LeftWristPitch) ||
        bone == static_cast<size_t>(g1_skeleton::LeftWrist) ||
        bone == static_cast<size_t>(g1_skeleton::RightWristRoll) ||
        bone == static_cast<size_t>(g1_skeleton::RightWristPitch) ||
        bone == static_cast<size_t>(g1_skeleton::RightWrist);
}

bool active_grasp_chain_bone(size_t bone, Hand hand) {
    if (hand == Hand::Left) {
        return bone == static_cast<size_t>(g1_skeleton::LeftWristRoll) ||
            bone == static_cast<size_t>(g1_skeleton::LeftWristPitch) ||
            bone == static_cast<size_t>(g1_skeleton::LeftWrist);
    }
    return bone == static_cast<size_t>(g1_skeleton::RightWristRoll) ||
        bone == static_cast<size_t>(g1_skeleton::RightWristPitch) ||
        bone == static_cast<size_t>(g1_skeleton::RightWrist);
}

float joint_radius(
    size_t bone,
    const TrajectoryCollisionConfig& config) {
    if (torso_bone(bone)) return config.torso_radius_m;
    if (wrist_bone(bone)) return config.wrist_radius_m;
    return config.joint_radius_m;
}

float segment_radius(
    size_t bone,
    const TrajectoryCollisionConfig& config) {
    if (torso_bone(bone)) return config.torso_radius_m;
    if (wrist_bone(bone)) return config.forearm_radius_m;
    return config.limb_radius_m;
}

bool skeleton_intersects_box(
    const WorldPose& world,
    const OrientedBox& box,
    Hand hand,
    bool exempt_active_grasp_chain,
    const TrajectoryCollisionConfig& config) {
    for (size_t bone = 1U; bone < g1_skeleton::BoneCount; ++bone) {
        if (exempt_active_grasp_chain &&
            active_grasp_chain_bone(bone, hand)) {
            continue;
        }
        if (sphere_intersects_box(
                world.positions[bone], joint_radius(bone, config), box)) {
            return true;
        }
        const int32_t parent = g1_skeleton::kParents[bone];
        if (parent >= 0 && capsule_intersects_box(
                world.positions[static_cast<size_t>(parent)],
                world.positions[bone], segment_radius(bone, config), box)) {
            return true;
        }
    }
    return false;
}

bool valid_collision_config(const TrajectoryCollisionConfig& config) {
    return finite(config.wrist_radius_m) && config.wrist_radius_m >= 0.0F &&
        finite(config.forearm_radius_m) && config.forearm_radius_m >= 0.0F &&
        finite(config.joint_radius_m) && config.joint_radius_m >= 0.0F &&
        finite(config.limb_radius_m) && config.limb_radius_m >= 0.0F &&
        finite(config.torso_radius_m) && config.torso_radius_m >= 0.0F;
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

bool near_support_metadata(float left, float right) {
    return std::abs(left - right) <= kSupportMetadataTolerance;
}

bool near_support_metadata(vec3 left, vec3 right) {
    return near_support_metadata(left.x, right.x) &&
        near_support_metadata(left.y, right.y) &&
        near_support_metadata(left.z, right.z);
}

bool near_support_metadata(quat left, quat right) {
    return near_support_metadata(left.w, right.w) &&
        near_support_metadata(left.x, right.x) &&
        near_support_metadata(left.y, right.y) &&
        near_support_metadata(left.z, right.z);
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

Transform upright_grasp_alignment(
    const Transform& source_contact,
    const HandTrajectoryQuery& query) {
    const float yaw = shortest_angle(
        yaw_radians(query.grasp_world_rotation) -
        yaw_radians(source_contact.rotation));
    const quat rotation = quat_from_angle_axis(
        yaw, vec3(0.0F, 1.0F, 0.0F));
    const vec3 rotated_source = quat_mul_vec3(
        rotation, source_contact.position);
    return {
        vec3(
            query.grasp_world_position.x - rotated_source.x,
            0.0F,
            query.grasp_world_position.z - rotated_source.z),
        rotation,
    };
}

struct ClipPhases {
    int32_t reach = -1;
    int32_t contact = -1;
    int32_t first_lift = -1;
    int32_t last_lift = -1;
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
        if (phase == kLiftPhase) {
            if (result.first_lift < 0) {
                result.first_lift = frame;
                result.last_lift = frame;
            } else if (frame == result.last_lift + 1) {
                result.last_lift = frame;
            }
        } else if (result.first_lift >= 0) {
            break;
        }
    }
    if (!(result.reach >= start && result.reach < result.contact &&
          result.contact < result.first_lift &&
          result.first_lift <= result.last_lift &&
          result.last_lift < stop)) {
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
    if (!finite(query.grasp_world_position) ||
        !valid_rotation(query.grasp_world_rotation) ||
        !finite(config.maximum_grasp_position_error_m) ||
        config.maximum_grasp_position_error_m < 0.0F ||
        !finite(config.maximum_grasp_orientation_error_radians) ||
        config.maximum_grasp_orientation_error_radians < 0.0F ||
        config.maximum_compatible_clips == 0U ||
        config.maximum_compatible_clips > 4096U) {
        throw std::invalid_argument("invalid hand trajectory query or config");
    }
}

void validate_trajectory(const HandTrajectory& trajectory) {
    if (trajectory.hands_in_source_object.empty() ||
        trajectory.hands_in_source_object.size() !=
            trajectory.elbows_in_source_object.size() ||
        trajectory.reach_point >= trajectory.hands_in_source_object.size() ||
        trajectory.contact_point >= trajectory.hands_in_source_object.size() ||
        trajectory.reach_point >= trajectory.contact_point ||
        !finite(trajectory.source_object.position) ||
        !valid_rotation(trajectory.source_object.rotation)) {
        throw std::invalid_argument("invalid hand trajectory");
    }
}

}  // namespace

SupportKind support_kind(const Database& database, size_t clip) {
    const vec3 position = read_vec3(database.table_positions, clip);
    const quat rotation = read_quat(database.table_rotations, clip);
    const vec3 size = read_vec3(database.table_sizes, clip);
    const bool virtual_floor =
        near_support_metadata(position, vec3(0.0F, -0.02F, 0.0F)) &&
        near_support_metadata(size, vec3(20.0F, 0.04F, 20.0F)) &&
        near_support_metadata(rotation, quat()) &&
        near_support_metadata(position.y + 0.5F * size.y, 0.0F);
    return virtual_floor ? SupportKind::Ground : SupportKind::Table;
}

std::vector<HandTrajectory> select_hand_trajectories(
    const Database& database,
    const HandTrajectoryQuery& query,
    const HandTrajectoryConfig& config) {
    validate_query_and_config(query, config);
    std::vector<HandTrajectory> selected;
    for (size_t clip = 0; clip < database.clip_count; ++clip) {
        if (database.active_hands.at(clip) != static_cast<uint8_t>(query.hand)) {
            continue;
        }
        const ClipPhases phases = clip_phases(database, clip);
        if (phases.reach < 0) continue;
        const Transform source_object = object_transform(
            database, phases.contact - 1);
        const WorldPose contact_world = world_pose(
            pose_at_frame(database, phases.contact));
        const Transform source_contact = hand_transform(contact_world, query.hand);
        const Transform alignment = upright_grasp_alignment(
            source_contact, query);
        const Transform mapped_contact = compose(alignment, source_contact);
        const float position_error = length(
            mapped_contact.position - query.grasp_world_position);
        const float orientation_error = query.constrain_grasp_orientation
            ? quaternion_angle(
                mapped_contact.rotation, query.grasp_world_rotation)
            : 0.0F;
        if (position_error > config.maximum_grasp_position_error_m ||
            orientation_error >
                config.maximum_grasp_orientation_error_radians) {
            continue;
        }
        HandTrajectory trajectory{};
        trajectory.clip = static_cast<int32_t>(clip);
        trajectory.start_frame = database.range_starts.at(clip);
        trajectory.reach_frame = phases.reach;
        trajectory.contact_frame = phases.contact;
        trajectory.lift_frame = phases.last_lift;
        trajectory.reach_point = static_cast<size_t>(
            phases.reach - trajectory.start_frame);
        trajectory.contact_point = static_cast<size_t>(
            phases.contact - trajectory.start_frame);
        trajectory.support = support_kind(database, clip);
        trajectory.source_object = source_object;
        const Transform source_from_world = inverse(trajectory.source_object);
        for (int32_t frame = trajectory.start_frame;
             frame <= phases.last_lift;
             ++frame) {
            const WorldPose world = world_pose(pose_at_frame(database, frame));
            trajectory.hands_in_source_object.push_back(compose(
                source_from_world,
                hand_transform(world, query.hand)));
            trajectory.elbows_in_source_object.push_back(compose(
                source_from_world,
                Transform{world.positions[elbow_index(query.hand)], quat()}
            ).position);
        }
        const float position_normalizer = std::max(
            config.maximum_grasp_position_error_m, 1.0e-6F);
        const float orientation_normalizer = std::max(
            config.maximum_grasp_orientation_error_radians, 1.0e-6F);
        trajectory.cost =
            position_error / position_normalizer +
            orientation_error / orientation_normalizer;
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
    const Transform source_contact = compose(
        trajectory.source_object,
        trajectory.hands_in_source_object[trajectory.contact_point]);
    return upright_grasp_alignment(source_contact, query);
}

ShapedHandTrajectory shape_hand_trajectory(
    const Database& database,
    const HandTrajectory& trajectory,
    const HandTrajectoryQuery& query,
    const IKConfig& config) {
    validate_trajectory(trajectory);
    const Transform alignment = hand_trajectory_scene_alignment(
        trajectory, query);
    const size_t sample_count = trajectory.hands_in_source_object.size();
    if (trajectory.start_frame < 0 ||
        trajectory.reach_frame < trajectory.start_frame ||
        trajectory.contact_frame <= trajectory.reach_frame ||
        trajectory.lift_frame < trajectory.contact_frame ||
        trajectory.reach_point != static_cast<size_t>(
            trajectory.reach_frame - trajectory.start_frame) ||
        trajectory.contact_point != static_cast<size_t>(
            trajectory.contact_frame - trajectory.start_frame) ||
        static_cast<size_t>(trajectory.lift_frame - trajectory.start_frame + 1) !=
            sample_count) {
        throw std::invalid_argument("trajectory frame range does not match samples");
    }

    std::vector<Pose> aligned;
    aligned.reserve(sample_count);
    std::vector<Transform> base_hands;
    base_hands.reserve(sample_count);
    for (size_t sample = 0U; sample < sample_count; ++sample) {
        Pose pose = pose_at_frame(
            database,
            trajectory.start_frame + static_cast<int32_t>(sample));
        const size_t root = static_cast<size_t>(g1_skeleton::Simulation);
        const Transform mapped_root = compose(
            alignment, {pose.positions[root], pose.rotations[root]});
        pose.positions[root] = mapped_root.position;
        pose.rotations[root] = mapped_root.rotation;
        const WorldPose world = world_pose(pose);
        base_hands.push_back(hand_transform(world, query.hand));
        aligned.push_back(std::move(pose));
    }

    const Transform& base_contact = base_hands[trajectory.contact_point];
    const vec3 contact_translation =
        query.grasp_world_position - base_contact.position;
    const quat contact_rotation = query.constrain_grasp_orientation
        ? quat_normalize(quat_mul(
              query.grasp_world_rotation,
              quat_inv(base_contact.rotation)))
        : quat();

    ShapedHandTrajectory shaped{};
    shaped.poses.reserve(sample_count);
    shaped.path.hands.reserve(sample_count);
    shaped.path.elbows.reserve(sample_count);
    for (size_t sample = 0U; sample < sample_count; ++sample) {
        const float linear_weight = sample <= trajectory.reach_point
            ? 0.0F
            : std::min(
                  1.0F,
                  static_cast<float>(sample - trajectory.reach_point) /
                      static_cast<float>(
                          trajectory.contact_point - trajectory.reach_point));
        const float weight =
            linear_weight * linear_weight * (3.0F - 2.0F * linear_weight);
        Transform target = base_hands[sample];
        target.position = target.position + weight * contact_translation;
        if (query.constrain_grasp_orientation) {
            const quat corrected = quat_normalize(quat_mul(
                contact_rotation, base_hands[sample].rotation));
            target.rotation = quat_nlerp_shortest(
                base_hands[sample].rotation, corrected, weight);
        }

        Pose pose = std::move(aligned[sample]);
        IKConfig solve_config = config;
        if (!query.constrain_grasp_orientation) {
            constexpr float pi = 3.141592654F;
            solve_config.maximum_request_orientation_radians = pi;
            solve_config.accepted_orientation_radians = pi;
            solve_config.orientation_scale_m_per_radian = 0.0F;
        }
        IKResult result{};
        if (weight > 0.0F) {
            result = solve_hand_ik(pose, query.hand, target, solve_config);
        }
        const WorldPose world = world_pose(pose);
        shaped.path.hands.push_back(hand_transform(world, query.hand));
        shaped.path.elbows.push_back(
            world.positions[elbow_index(query.hand)]);
        shaped.poses.push_back(std::move(pose));
        if (sample == trajectory.contact_point) {
            shaped.contact_accepted = result.accepted;
            shaped.reason = result.reason;
        }
    }
    return shaped;
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
        !valid_box(object) || !valid_collision_config(config)) {
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

TrajectoryFeasibility evaluate_shaped_trajectory_feasibility(
    const ShapedHandTrajectory& trajectory,
    size_t contact_point,
    Hand hand,
    const OrientedBox& object,
    const ShelfGeometry& shelf,
    const TrajectoryCollisionConfig& config) {
    if (trajectory.poses.empty() ||
        trajectory.path.hands.size() != trajectory.poses.size() ||
        trajectory.path.elbows.size() != trajectory.poses.size() ||
        contact_point >= trajectory.poses.size() ||
        !valid_box(object) || !valid_collision_config(config)) {
        throw std::invalid_argument("invalid shaped trajectory collision query");
    }
    for (const OrientedBox& box : shelf.boxes) {
        if (!valid_box(box)) {
            throw std::invalid_argument("invalid shelf geometry");
        }
    }
    for (size_t sample = 0U; sample < trajectory.poses.size(); ++sample) {
        const WorldPose world = world_pose(trajectory.poses[sample]);
        const bool after_contact = sample >= contact_point;
        if (skeleton_intersects_box(
                world, object, hand, after_contact, config)) {
            return {TrajectoryFeasibilityReason::ObjectCollision, sample};
        }
        for (const OrientedBox& box : shelf.boxes) {
            if (skeleton_intersects_box(
                    world, box, hand, false, config)) {
                return {TrajectoryFeasibilityReason::ShelfCollision, sample};
            }
        }
    }
    return {};
}

}  // namespace interaction
