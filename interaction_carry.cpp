#include "interaction_carry.h"

#include "g1_arm_joint_metadata.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace interaction {
namespace {

constexpr size_t kVectorComponents = 3U;
constexpr size_t kQuaternionComponents = 4U;
constexpr size_t kHandCount = 2U;
constexpr float kCarrySeamSeconds = 0.50F;
constexpr int32_t kCarrySeamAttempts = 8;
constexpr int64_t kLayeredSeamKey = -1;
constexpr int64_t kUnsetSeamKey = -2;
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

bool valid_rotation(quat value) {
    if (!finite(value)) return false;
    const float norm = quat_length(value);
    return norm > 1.0e-6F && std::abs(norm - 1.0F) <= 1.0e-3F;
}

quat normalized(quat value) {
    return value / quat_length(value);
}

bool within_inclusive(float value, float maximum) {
    const float representational_slack =
        8.0F * std::numeric_limits<float>::epsilon() * maximum;
    return value <= maximum + representational_slack;
}

size_t checked_product(size_t left, size_t right) {
    if (right != 0U &&
        left > std::numeric_limits<size_t>::max() / right) {
        throw std::invalid_argument("interaction carry database overflow");
    }
    return left * right;
}

void require_size(
    size_t actual,
    size_t expected,
    const char* label) {
    if (actual != expected) {
        throw std::invalid_argument(
            std::string("interaction carry invalid ") + label + " size");
    }
}

void require_finite_values(
    const std::vector<float>& values,
    const char* label) {
    if (!std::all_of(
            values.begin(), values.end(),
            [](float value) { return finite(value); })) {
        throw std::invalid_argument(
            std::string("interaction carry non-finite ") + label);
    }
}

void require_valid_rotations(
    const std::vector<float>& values,
    const char* label) {
    for (size_t offset = 0U; offset < values.size(); offset += 4U) {
        const quat value(
            values[offset], values[offset + 1U],
            values[offset + 2U], values[offset + 3U]);
        if (!valid_rotation(value)) {
            throw std::invalid_argument(
                std::string("interaction carry invalid ") + label);
        }
    }
}

bool valid_config(const CarryConfig& config) {
    return config.minimum_hold_frames >= 2 &&
           finite(config.maximum_grasp_drift_m) &&
           config.maximum_grasp_drift_m >= 0.0F &&
           finite(config.maximum_grasp_drift_radians) &&
           config.maximum_grasp_drift_radians >= 0.0F &&
           finite(config.minimum_root_displacement_m) &&
           config.minimum_root_displacement_m >= 0.0F &&
           finite(config.minimum_average_speed_mps) &&
           config.minimum_average_speed_mps >= 0.0F &&
           finite(config.search_interval_seconds) &&
           config.search_interval_seconds > 0.0F &&
           finite(config.spine_weight) &&
           config.spine_weight >= 0.0F && config.spine_weight <= 1.0F &&
           finite(config.inactive_arm_weight) &&
           config.inactive_arm_weight >= 0.0F &&
           config.inactive_arm_weight <= 1.0F;
}

bool valid_ik_config(const IKConfig& config) {
    return finite(config.maximum_request_position_m) &&
           config.maximum_request_position_m >= 0.0F &&
           finite(config.maximum_request_orientation_radians) &&
           config.maximum_request_orientation_radians >= 0.0F &&
           finite(config.accepted_position_m) &&
           config.accepted_position_m >= 0.0F &&
           finite(config.accepted_orientation_radians) &&
           config.accepted_orientation_radians >= 0.0F &&
           finite(config.damping) && config.damping > 0.0F &&
           finite(config.finite_difference_radians) &&
           config.finite_difference_radians > 0.0F &&
           finite(config.orientation_scale_m_per_radian) &&
           config.orientation_scale_m_per_radian > 0.0F &&
           finite(config.maximum_step_radians) &&
           config.maximum_step_radians > 0.0F &&
           config.maximum_iterations >= 0;
}

void validate_database(const Database& database) {
    if (database.version != 1U ||
        database.endian_marker != 0x01020304U ||
        database.frame_count == 0U || database.clip_count == 0U ||
        database.frame_count >
            static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        database.bone_count != g1_skeleton::BoneCount ||
        database.hand_dof_count != 14U ||
        database.fps_numerator == 0U || database.fps_denominator == 0U) {
        throw std::invalid_argument("interaction carry invalid database header");
    }

    const size_t frames = database.frame_count;
    const size_t bones = g1_skeleton::BoneCount;
    const size_t clips = database.clip_count;
    const size_t frame_bones = checked_product(frames, bones);
    require_size(database.parents.size(), bones, "parents");
    require_size(database.range_starts.size(), clips, "range_starts");
    require_size(database.range_stops.size(), clips, "range_stops");
    require_size(database.active_hands.size(), clips, "active_hands");
    require_size(database.positions.size(),
                 checked_product(frame_bones, kVectorComponents),
                 "positions");
    require_size(database.velocities.size(),
                 checked_product(frame_bones, kVectorComponents),
                 "velocities");
    require_size(database.rotations.size(),
                 checked_product(frame_bones, kQuaternionComponents),
                 "rotations");
    require_size(database.angular_velocities.size(),
                 checked_product(frame_bones, kVectorComponents),
                 "angular_velocities");
    require_size(database.foot_contacts.size(),
                 checked_product(frames, kHandCount),
                 "foot_contacts");
    require_size(database.hand_contacts.size(),
                 checked_product(frames, kHandCount),
                 "hand_contacts");
    require_size(database.hand_dof.size(),
                 checked_product(frames, 14U), "hand_dof");
    require_size(database.hand_dof_velocities.size(),
                 checked_product(frames, 14U), "hand_dof_velocities");
    require_size(database.phases.size(), frames, "phases");
    require_size(database.object_positions.size(),
                 checked_product(frames, kVectorComponents),
                 "object_positions");
    require_size(database.object_rotations.size(),
                 checked_product(frames, kQuaternionComponents),
                 "object_rotations");
    require_size(database.grasp_positions_object.size(),
                 checked_product(clips, kVectorComponents),
                 "grasp_positions_object");
    require_size(database.grasp_rotations_object.size(),
                 checked_product(clips, kQuaternionComponents),
                 "grasp_rotations_object");

    if (!std::equal(
            database.parents.begin(),
            database.parents.end(),
            g1_skeleton::kParents.begin())) {
        throw std::invalid_argument(
            "interaction carry database parents do not match G1");
    }

    int32_t previous_stop = 0;
    for (size_t clip = 0; clip < clips; ++clip) {
        const int32_t start = database.range_starts[clip];
        const int32_t stop = database.range_stops[clip];
        if (start != previous_stop || stop <= start ||
            stop > static_cast<int32_t>(database.frame_count) ||
            database.active_hands[clip] > 1U) {
            throw std::invalid_argument(
                "interaction carry invalid clip range");
        }
        previous_stop = stop;
    }
    if (previous_stop != static_cast<int32_t>(database.frame_count)) {
        throw std::invalid_argument(
            "interaction carry clip ranges do not cover database");
    }
    if (!std::all_of(
            database.phases.begin(), database.phases.end(),
            [](uint8_t value) {
                return value <= static_cast<uint8_t>(Phase::Hold);
            }) ||
        !std::all_of(
            database.hand_contacts.begin(), database.hand_contacts.end(),
            [](uint8_t value) { return value <= 1U; }) ||
        !std::all_of(
            database.foot_contacts.begin(), database.foot_contacts.end(),
            [](uint8_t value) { return value <= 1U; })) {
        throw std::invalid_argument(
            "interaction carry invalid phase or contact value");
    }

    require_finite_values(database.positions, "positions");
    require_finite_values(database.velocities, "velocities");
    require_finite_values(database.rotations, "rotations");
    require_finite_values(database.angular_velocities, "angular_velocities");
    require_finite_values(database.hand_dof, "hand_dof");
    require_finite_values(
        database.hand_dof_velocities, "hand_dof_velocities");
    require_finite_values(database.object_positions, "object_positions");
    require_finite_values(database.object_rotations, "object_rotations");
    require_finite_values(
        database.grasp_positions_object, "grasp_positions_object");
    require_finite_values(
        database.grasp_rotations_object, "grasp_rotations_object");
    require_valid_rotations(database.rotations, "pose rotation");
    require_valid_rotations(database.object_rotations, "object rotation");
    require_valid_rotations(
        database.grasp_rotations_object, "canonical grasp rotation");
}

void validate_features(
    const Database& database,
    const Features& features) {
    if (features.version != 1U ||
        features.endian_marker != 0x01020304U ||
        features.frame_count != database.frame_count ||
        features.feature_count != kFeatureDimension ||
        features.dimension != kFeatureDimension ||
        features.group_count != kFeatureGroupStarts.size() ||
        features.group_starts.size() != kFeatureGroupStarts.size() ||
        features.group_stops.size() != kFeatureGroupStops.size() ||
        features.offsets.size() != kFeatureDimension ||
        features.scales.size() != kFeatureDimension ||
        features.values.size() != checked_product(
            static_cast<size_t>(database.frame_count),
            kFeatureDimension)) {
        throw std::invalid_argument("interaction carry invalid features");
    }
    for (size_t group = 0; group < kFeatureGroupStarts.size(); ++group) {
        if (features.group_starts[group] != kFeatureGroupStarts[group] ||
            features.group_stops[group] != kFeatureGroupStops[group]) {
            throw std::invalid_argument(
                "interaction carry invalid feature groups");
        }
    }
    require_finite_values(features.offsets, "feature offsets");
    require_finite_values(features.scales, "feature scales");
    require_finite_values(features.values, "feature values");
    if (!std::all_of(
            features.scales.begin(), features.scales.end(),
            [](float value) { return value > 0.0F; })) {
        throw std::invalid_argument(
            "interaction carry feature scales must be positive");
    }
}

bool valid_hand(Hand hand) {
    return static_cast<uint8_t>(hand) <=
        static_cast<uint8_t>(Hand::Right);
}

bool certified_range(
    const Database& database,
    const CarryRange& range,
    const CarryConfig& config);

void validate_ranges(
    const Database& database,
    CarryRanges& ranges,
    const CarryConfig& config) {
    std::sort(
        ranges.recorded.begin(), ranges.recorded.end(),
        [](const CarryRange& left, const CarryRange& right) {
            if (left.clip != right.clip) return left.clip < right.clip;
            if (left.start_frame != right.start_frame) {
                return left.start_frame < right.start_frame;
            }
            return left.stop_frame < right.stop_frame;
        });
    std::sort(ranges.rejected.begin(), ranges.rejected.end());

    std::vector<bool> rejected(database.clip_count, false);
    for (int32_t clip : ranges.rejected) {
        if (clip < 0 || static_cast<uint32_t>(clip) >= database.clip_count ||
            rejected[static_cast<size_t>(clip)]) {
            throw std::invalid_argument(
                "interaction carry invalid rejected clip");
        }
        rejected[static_cast<size_t>(clip)] = true;
    }

    int32_t previous_clip = -1;
    int32_t previous_stop = -1;
    for (const CarryRange& range : ranges.recorded) {
        if (range.clip < 0 ||
            static_cast<uint32_t>(range.clip) >= database.clip_count ||
            !valid_hand(range.hand)) {
            throw std::invalid_argument(
                "interaction carry invalid recorded range");
        }
        const size_t clip = static_cast<size_t>(range.clip);
        const Hand active = database.active_hands[clip] == 0U
            ? Hand::Left
            : Hand::Right;
        if (range.hand != active ||
            range.start_frame < database.range_starts[clip] ||
            range.stop_frame > database.range_stops[clip] ||
            range.stop_frame <= range.start_frame ||
            (range.clip == previous_clip &&
             range.start_frame < previous_stop) ||
            rejected[clip]) {
            throw std::invalid_argument(
                "interaction carry recorded range is inconsistent");
        }
        const size_t hand_offset = range.hand == Hand::Left ? 0U : 1U;
        for (int32_t frame = range.start_frame;
             frame < range.stop_frame;
             ++frame) {
            if (database.phases[static_cast<size_t>(frame)] !=
                    static_cast<uint8_t>(Phase::Hold) ||
                database.hand_contacts[
                    static_cast<size_t>(frame) * 2U + hand_offset] != 1U) {
                throw std::invalid_argument(
                    "interaction carry range lacks continuous hold contact");
            }
        }
        if (!certified_range(database, range, config)) {
            throw std::invalid_argument(
                "interaction carry range is not certified");
        }
        previous_clip = range.clip;
        previous_stop = range.stop_frame;
    }
}

bool valid_transform(const Transform& value) {
    return finite(value.position) && valid_rotation(value.rotation);
}

bool valid_pose(const Pose& pose) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        if (!finite(pose.positions[bone]) ||
            !finite(pose.velocities[bone]) ||
            !valid_rotation(pose.rotations[bone]) ||
            !finite(pose.angular_velocities[bone])) {
            return false;
        }
    }
    for (float value : pose.hand_dof) {
        if (!finite(value)) return false;
    }
    for (float value : pose.hand_dof_velocities) {
        if (!finite(value)) return false;
    }
    return std::all_of(
        pose.foot_contacts.begin(), pose.foot_contacts.end(),
        [](uint8_t value) { return value <= 1U; });
}

bool valid_locomotion(const LocomotionSnapshot& locomotion) {
    if (!valid_pose(locomotion.pose)) return false;
    for (size_t sample = 0;
         sample < locomotion.future_root_positions.size();
         ++sample) {
        if (!finite(locomotion.future_root_positions[sample]) ||
            !valid_rotation(locomotion.future_root_rotations[sample])) {
            return false;
        }
    }
    return true;
}

vec3 read_vec3(const std::vector<float>& values, size_t index) {
    const size_t offset = index * kVectorComponents;
    return vec3(values[offset], values[offset + 1U], values[offset + 2U]);
}

quat read_quat(const std::vector<float>& values, size_t index) {
    const size_t offset = index * kQuaternionComponents;
    return normalized(quat(
        values[offset], values[offset + 1U],
        values[offset + 2U], values[offset + 3U]));
}

Transform object_transform(const Database& database, int32_t frame) {
    return {
        read_vec3(database.object_positions, static_cast<size_t>(frame)),
        read_quat(database.object_rotations, static_cast<size_t>(frame)),
    };
}

Transform hand_transform(
    const Database& database,
    int32_t frame,
    Hand hand) {
    const WorldPose world = world_pose(pose_at_frame(database, frame));
    const size_t bone = hand == Hand::Left ? kLeftHandBone : kRightHandBone;
    return {world.positions[bone], normalized(world.rotations[bone])};
}

Transform hand_transform(const Pose& pose, Hand hand) {
    const WorldPose world = world_pose(pose);
    const size_t bone = hand == Hand::Left ? kLeftHandBone : kRightHandBone;
    return {world.positions[bone], normalized(world.rotations[bone])};
}

Transform root_transform(const Pose& pose) {
    return {
        pose.positions[g1_skeleton::Simulation],
        normalized(pose.rotations[g1_skeleton::Simulation]),
    };
}

const std::array<HingeJoint, 7>& arm(Hand hand) {
    return hand == Hand::Left ? kLeftArm : kRightArm;
}

float yaw_radians(quat rotation) {
    const vec3 facing = quat_mul_vec3(
        rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(facing.x, facing.z);
}

float shortest_angle(float value) {
    return std::atan2(std::sin(value), std::cos(value));
}

Transform planar_alignment(
    Transform source_root,
    Transform live_root) {
    const float yaw = shortest_angle(
        yaw_radians(live_root.rotation) -
        yaw_radians(source_root.rotation));
    const quat rotation = quat_from_angle_axis(
        yaw, vec3(0.0F, 1.0F, 0.0F));
    const vec3 rotated_source = quat_mul_vec3(
        rotation, source_root.position);
    return {
        vec3(
            live_root.position.x - rotated_source.x,
            0.0F,
            live_root.position.z - rotated_source.z),
        rotation,
    };
}

Transform canonical_grasp(const Database& database, int32_t clip) {
    return {
        read_vec3(
            database.grasp_positions_object,
            static_cast<size_t>(clip)),
        read_quat(
            database.grasp_rotations_object,
            static_cast<size_t>(clip)),
    };
}

float rotation_distance(quat left, quat right);

bool compatible_range(
    const Database& database,
    const CarryRange& range,
    Hand hand,
    const GraspAffordance& affordance,
    const CarryConfig& config) {
    if (range.hand != hand || affordance.hand != hand) return false;
    const Transform canonical = canonical_grasp(database, range.clip);
    return within_inclusive(
               length(
                   canonical.position -
                   affordance.hand_in_object.position),
               config.maximum_grasp_drift_m) &&
           within_inclusive(
               rotation_distance(
                   canonical.rotation,
                   affordance.hand_in_object.rotation),
               config.maximum_grasp_drift_radians);
}

NormalizedQuery recorded_query(
    const Features& features,
    const LocomotionSnapshot& locomotion,
    Hand hand,
    Transform object_world,
    const GraspAffordance& affordance) {
    QueryInput input{};
    input.locomotion = locomotion;
    input.grasp_world = compose(
        object_world, affordance.hand_in_object);
    input.table_world = {vec3(), quat()};
    input.table_size = vec3(1.0F, 1.0F, 1.0F);
    input.approach_direction_object = vec3(0.0F, 0.0F, -1.0F);
    input.object_dimensions = vec3(1.0F, 1.0F, 1.0F);
    input.hand = hand;
    return normalize_query(build_raw_query(input), features);
}

double pose_trajectory_cost(
    const Features& features,
    const NormalizedQuery& query,
    int32_t frame) {
    double total = 0.0;
    for (size_t group = 0; group < 2U; ++group) {
        const size_t start = kFeatureGroupStarts[group];
        const size_t stop = kFeatureGroupStops[group];
        double squared = 0.0;
        for (size_t dimension = start; dimension < stop; ++dimension) {
            const size_t index =
                static_cast<size_t>(frame) * kFeatureDimension + dimension;
            const double difference =
                static_cast<double>(features.values[index]) -
                static_cast<double>(query[dimension]);
            squared += difference * difference;
        }
        total += squared / static_cast<double>(stop - start);
    }
    return 0.5 * total;
}

bool equal_cost(double left, double right) {
    const double scale = std::max({1.0, std::abs(left), std::abs(right)});
    return std::abs(left - right) <=
        8.0 * std::numeric_limits<double>::epsilon() * scale;
}

Pose sample_recorded_pose(
    const Database& database,
    const CarryRange& range,
    double source_frame) {
    const int32_t left = static_cast<int32_t>(std::floor(source_frame));
    const int32_t right = std::min(left + 1, range.stop_frame - 1);
    const float alpha = static_cast<float>(
        source_frame - static_cast<double>(left));
    return interpolate_pose(
        pose_at_frame(database, left),
        pose_at_frame(database, right),
        alpha);
}

Transform sample_recorded_object(
    const Database& database,
    const CarryRange& range,
    double source_frame) {
    const int32_t left = static_cast<int32_t>(std::floor(source_frame));
    const int32_t right = std::min(left + 1, range.stop_frame - 1);
    const float alpha = static_cast<float>(
        source_frame - static_cast<double>(left));
    const Transform first = object_transform(database, left);
    const Transform second = object_transform(database, right);
    return {
        lerp(first.position, second.position, alpha),
        quat_nlerp_shortest(first.rotation, second.rotation, alpha),
    };
}

Transform hand_in_object(
    const Database& database,
    int32_t frame,
    Hand hand) {
    return compose(
        inverse(object_transform(database, frame)),
        hand_transform(database, frame, hand));
}

float rotation_distance(quat left, quat right) {
    const float cosine = std::abs(quat_dot(
        normalized(left), normalized(right)));
    return 2.0F * std::acos(std::clamp(cosine, 0.0F, 1.0F));
}

bool stable_from(
    const Transform& anchor,
    const Transform& current,
    const CarryConfig& config) {
    return within_inclusive(
               length(current.position - anchor.position),
               config.maximum_grasp_drift_m) &&
           within_inclusive(
               rotation_distance(current.rotation, anchor.rotation),
               config.maximum_grasp_drift_radians);
}

bool continuous_object(
    const Transform& desired,
    const Transform& candidate,
    const CarryConfig& carry_config,
    const IKConfig& ik_config) {
    const float maximum_position = std::min(
        carry_config.maximum_grasp_drift_m,
        ik_config.maximum_request_position_m);
    const float maximum_orientation = std::min(
        carry_config.maximum_grasp_drift_radians,
        ik_config.maximum_request_orientation_radians);
    return within_inclusive(
               length(candidate.position - desired.position),
               maximum_position) &&
           within_inclusive(
               rotation_distance(
                   candidate.rotation, desired.rotation),
               maximum_orientation);
}

Pose remap_safe_pose_to_live_root(
    const Pose& safe,
    const Pose& locomotion) {
    Pose remapped = safe;
    constexpr size_t kRoot = g1_skeleton::Simulation;
    remapped.positions[kRoot] = locomotion.positions[kRoot];
    remapped.velocities[kRoot] = locomotion.velocities[kRoot];
    remapped.rotations[kRoot] = locomotion.rotations[kRoot];
    remapped.angular_velocities[kRoot] =
        locomotion.angular_velocities[kRoot];
    return remapped;
}

Pose blend_carry_transition(
    const Pose& safe,
    const Pose& target,
    const Pose& active_arm_source,
    Hand active_hand,
    bool preserve_active_arm,
    float alpha) {
    Pose blended = safe;
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        if (bone == static_cast<size_t>(g1_skeleton::Simulation)) {
            blended.positions[bone] = target.positions[bone];
            blended.velocities[bone] = target.velocities[bone];
            blended.rotations[bone] = target.rotations[bone];
            blended.angular_velocities[bone] =
                target.angular_velocities[bone];
            continue;
        }
        blended.positions[bone] =
            (1.0F - alpha) * safe.positions[bone] +
            alpha * target.positions[bone];
        blended.velocities[bone] =
            (1.0F - alpha) * safe.velocities[bone] +
            alpha * target.velocities[bone];
        blended.rotations[bone] = quat_nlerp_shortest(
            safe.rotations[bone], target.rotations[bone], alpha);
        blended.angular_velocities[bone] =
            (1.0F - alpha) * safe.angular_velocities[bone] +
            alpha * target.angular_velocities[bone];
    }
    for (size_t dof = 0U; dof < blended.hand_dof.size(); ++dof) {
        blended.hand_dof[dof] =
            (1.0F - alpha) * safe.hand_dof[dof] +
            alpha * target.hand_dof[dof];
        blended.hand_dof_velocities[dof] =
            (1.0F - alpha) * safe.hand_dof_velocities[dof] +
            alpha * target.hand_dof_velocities[dof];
    }
    if (preserve_active_arm) {
        for (const HingeJoint& joint : arm(active_hand)) {
            const size_t bone = static_cast<size_t>(joint.bone);
            blended.rotations[bone] = active_arm_source.rotations[bone];
        }
    }
    blended.foot_contacts = target.foot_contacts;
    return blended;
}

struct CarryCandidate {
    Pose pose{};
    Transform object{};
    bool recorded = false;
    bool directly_solved = false;
    int64_t seam_key = kLayeredSeamKey;
};

class CarryUpdateTransaction {
public:
    CarryUpdateTransaction(
        Transform& object_world,
        Transform& object_in_root,
        bool& recorded,
        float& search_seconds,
        bool& search_pending,
        int32_t& recorded_range_index,
        double& source_frame_exact,
        double& search_seconds_exact,
        Pose& transition_source_pose,
        float& transition_progress,
        int64_t& recorded_selection_epoch,
        int64_t& published_seam_key,
        int64_t& transition_seam_key,
        bool& transition_active) noexcept
        : object_world_(object_world),
          object_in_root_(object_in_root),
          recorded_(recorded),
          search_seconds_(search_seconds),
          search_pending_(search_pending),
          recorded_range_index_(recorded_range_index),
          source_frame_exact_(source_frame_exact),
          search_seconds_exact_(search_seconds_exact),
          transition_source_pose_(transition_source_pose),
          transition_progress_(transition_progress),
          recorded_selection_epoch_(recorded_selection_epoch),
          published_seam_key_(published_seam_key),
          transition_seam_key_(transition_seam_key),
          transition_active_(transition_active),
          previous_object_world_(object_world),
          previous_object_in_root_(object_in_root),
          previous_recorded_(recorded),
          previous_search_seconds_(search_seconds),
          previous_search_pending_(search_pending),
          previous_recorded_range_index_(recorded_range_index),
          previous_source_frame_exact_(source_frame_exact),
          previous_search_seconds_exact_(search_seconds_exact),
          previous_transition_source_pose_(transition_source_pose),
          previous_transition_progress_(transition_progress),
          previous_recorded_selection_epoch_(recorded_selection_epoch),
          previous_published_seam_key_(published_seam_key),
          previous_transition_seam_key_(transition_seam_key),
          previous_transition_active_(transition_active) {}

    ~CarryUpdateTransaction() noexcept {
        if (committed_) return;
        object_world_ = previous_object_world_;
        object_in_root_ = previous_object_in_root_;
        recorded_ = previous_recorded_;
        search_seconds_ = previous_search_seconds_;
        search_pending_ = previous_search_pending_;
        recorded_range_index_ = previous_recorded_range_index_;
        source_frame_exact_ = previous_source_frame_exact_;
        search_seconds_exact_ = previous_search_seconds_exact_;
        transition_source_pose_ = previous_transition_source_pose_;
        transition_progress_ = previous_transition_progress_;
        recorded_selection_epoch_ = previous_recorded_selection_epoch_;
        published_seam_key_ = previous_published_seam_key_;
        transition_seam_key_ = previous_transition_seam_key_;
        transition_active_ = previous_transition_active_;
    }

    CarryUpdateTransaction(const CarryUpdateTransaction&) = delete;
    CarryUpdateTransaction& operator=(const CarryUpdateTransaction&) = delete;

    void commit() noexcept {
        committed_ = true;
    }

private:
    Transform& object_world_;
    Transform& object_in_root_;
    bool& recorded_;
    float& search_seconds_;
    bool& search_pending_;
    int32_t& recorded_range_index_;
    double& source_frame_exact_;
    double& search_seconds_exact_;
    Pose& transition_source_pose_;
    float& transition_progress_;
    int64_t& recorded_selection_epoch_;
    int64_t& published_seam_key_;
    int64_t& transition_seam_key_;
    bool& transition_active_;
    Transform previous_object_world_{};
    Transform previous_object_in_root_{};
    bool previous_recorded_ = false;
    float previous_search_seconds_ = 0.0F;
    bool previous_search_pending_ = false;
    int32_t previous_recorded_range_index_ = -1;
    double previous_source_frame_exact_ = 0.0;
    double previous_search_seconds_exact_ = 0.0;
    Pose previous_transition_source_pose_{};
    float previous_transition_progress_ = 0.0F;
    int64_t previous_recorded_selection_epoch_ = 0;
    int64_t previous_published_seam_key_ = kUnsetSeamKey;
    int64_t previous_transition_seam_key_ = kUnsetSeamKey;
    bool previous_transition_active_ = false;
    bool committed_ = false;
};

vec3 root_position(const Database& database, int32_t frame) {
    const size_t index = static_cast<size_t>(frame) *
        g1_skeleton::BoneCount + g1_skeleton::Simulation;
    return read_vec3(database.positions, index);
}

float planar_distance(vec3 left, vec3 right) {
    return std::hypot(left.x - right.x, left.z - right.z);
}

bool qualifies(
    const Database& database,
    int32_t start,
    int32_t stop,
    const CarryConfig& config) {
    const int32_t count = stop - start;
    if (count < config.minimum_hold_frames) return false;
    const vec3 first = root_position(database, start);
    const vec3 last = root_position(database, stop - 1);
    const float endpoint = planar_distance(first, last);
    if (!finite(endpoint)) {
        throw std::invalid_argument(
            "interaction carry root displacement overflow");
    }
    if (!within_inclusive(
            config.minimum_root_displacement_m, endpoint)) {
        return false;
    }

    float path = 0.0F;
    vec3 previous = first;
    for (int32_t frame = start + 1; frame < stop; ++frame) {
        const vec3 current = root_position(database, frame);
        path += planar_distance(previous, current);
        if (!finite(path)) {
            throw std::invalid_argument(
                "interaction carry root path overflow");
        }
        previous = current;
    }
    const float database_fps =
        static_cast<float>(database.fps_numerator) /
        static_cast<float>(database.fps_denominator);
    const float duration = static_cast<float>(count - 1) / database_fps;
    const float average_speed = path / duration;
    if (!finite(database_fps) || database_fps <= 0.0F ||
        !finite(duration) || duration <= 0.0F ||
        !finite(average_speed)) {
        throw std::invalid_argument(
            "interaction carry invalid database timing");
    }
    return within_inclusive(
        config.minimum_average_speed_mps, average_speed);
}

bool certified_range(
    const Database& database,
    const CarryRange& range,
    const CarryConfig& config) {
    if (!qualifies(
            database,
            range.start_frame,
            range.stop_frame,
            config)) {
        return false;
    }
    const Transform anchor = hand_in_object(
        database, range.start_frame, range.hand);
    for (int32_t frame = range.start_frame + 1;
         frame < range.stop_frame;
         ++frame) {
        if (!stable_from(
                anchor,
                hand_in_object(database, frame, range.hand),
                config)) {
            return false;
        }
    }
    return true;
}

void consider_run(
    CarryRanges& result,
    const Database& database,
    int32_t clip,
    int32_t start,
    int32_t stop,
    Hand hand,
    const CarryConfig& config,
    bool& accepted_clip) {
    if (start >= 0 && qualifies(database, start, stop, config)) {
        result.recorded.push_back({clip, start, stop, hand});
        accepted_clip = true;
    }
}

}  // namespace

CarryRanges classify_carry_ranges(
    const Database& database,
    const CarryConfig& config) {
    if (!valid_config(config)) {
        throw std::invalid_argument("interaction carry invalid config");
    }
    validate_database(database);

    CarryRanges result{};
    for (uint32_t clip_index = 0U;
         clip_index < database.clip_count;
         ++clip_index) {
        const int32_t clip = static_cast<int32_t>(clip_index);
        const int32_t start = database.range_starts[clip_index];
        const int32_t stop = database.range_stops[clip_index];
        const Hand hand = database.active_hands[clip_index] == 0U
            ? Hand::Left
            : Hand::Right;
        const size_t contact_offset = hand == Hand::Left ? 0U : 1U;
        bool accepted_clip = false;
        int32_t run_start = -1;
        Transform run_grasp{};
        for (int32_t frame = start; frame < stop; ++frame) {
            const bool holding = database.phases[static_cast<size_t>(frame)] ==
                static_cast<uint8_t>(Phase::Hold);
            const bool contact = database.hand_contacts[
                static_cast<size_t>(frame) * 2U + contact_offset] == 1U;
            if (!holding || !contact) {
                consider_run(
                    result, database, clip, run_start, frame,
                    hand, config, accepted_clip);
                run_start = -1;
                continue;
            }

            const Transform current = hand_in_object(database, frame, hand);
            if (run_start < 0) {
                run_start = frame;
                run_grasp = current;
            } else if (!stable_from(run_grasp, current, config)) {
                consider_run(
                    result, database, clip, run_start, frame,
                    hand, config, accepted_clip);
                run_start = frame;
                run_grasp = current;
            }
        }
        consider_run(
            result, database, clip, run_start, stop,
            hand, config, accepted_clip);
        if (!accepted_clip) result.rejected.push_back(clip);
    }
    return result;
}

CarryController::CarryController(
    const Database& database,
    const Features& features,
    CarryRanges ranges,
    CarryConfig config,
    IKConfig ik_config)
    : database_(&database),
      features_(&features),
      ranges_(std::move(ranges)),
      config_(config),
      ik_config_(ik_config) {
    if (!valid_config(config_) || !valid_ik_config(ik_config_)) {
        throw std::invalid_argument(
            "interaction carry invalid controller config");
    }
    validate_database(*database_);
    if (database_->fps_numerator != 25U ||
        database_->fps_denominator != 1U) {
        throw std::invalid_argument(
            "interaction carry controller requires database fps 25/1");
    }
    validate_features(*database_, *features_);
    validate_ranges(*database_, ranges_, config_);
}

void CarryController::start(
    const Pose& final_hold_pose,
    Hand hand,
    const GraspAffordance& affordance,
    Transform object_world) {
    if (!valid_hand(hand) || affordance.hand != hand ||
        !valid_transform(affordance.hand_in_object) ||
        !finite(affordance.approach_direction_object) ||
        !finite(affordance.clearance_radius) ||
        affordance.clearance_radius < 0.0F ||
        !valid_transform(object_world) ||
        !valid_pose(final_hold_pose)) {
        throw std::invalid_argument("interaction carry invalid start input");
    }
    const Transform object_in_root = compose(
        inverse(root_transform(final_hold_pose)), object_world);
    if (!valid_transform(object_in_root)) {
        throw std::invalid_argument(
            "interaction carry invalid root-relative object anchor");
    }
    final_hold_pose_ = final_hold_pose;
    hand_ = hand;
    affordance_ = affordance;
    object_world_ = object_world;
    object_in_root_ = object_in_root;
    recorded_ = false;
    last_safe_pose_ = final_hold_pose;
    search_seconds_ = 0.0F;
    started_ = true;
    search_pending_ = true;
    recorded_range_index_ = -1;
    source_frame_exact_ = 0.0;
    search_seconds_exact_ = 0.0;
    transition_source_pose_ = final_hold_pose;
    transition_progress_ = 0.0F;
    recorded_selection_epoch_ = 0;
    published_seam_key_ = kUnsetSeamKey;
    transition_seam_key_ = kUnsetSeamKey;
    transition_active_ = false;
}

Pose CarryController::update(
    const LocomotionSnapshot& locomotion,
    float dt) {
    if (!started_) {
        throw std::logic_error("interaction carry has not started");
    }
    if (!finite(dt) || dt < 0.0F || !valid_locomotion(locomotion)) {
        throw std::invalid_argument("interaction carry invalid update input");
    }

    CarryUpdateTransaction transaction(
        object_world_,
        object_in_root_,
        recorded_,
        search_seconds_,
        search_pending_,
        recorded_range_index_,
        source_frame_exact_,
        search_seconds_exact_,
        transition_source_pose_,
        transition_progress_,
        recorded_selection_epoch_,
        published_seam_key_,
        transition_seam_key_,
        transition_active_);
    const Transform live_root = root_transform(locomotion.pose);
    const Transform desired_object = compose(live_root, object_in_root_);
    if (!valid_transform(desired_object)) {
        throw std::invalid_argument(
            "interaction carry invalid moving object anchor");
    }
    const double interval =
        static_cast<double>(config_.search_interval_seconds);
    const double accumulated =
        search_seconds_exact_ + static_cast<double>(dt);
    if (!std::isfinite(accumulated)) {
        throw std::invalid_argument("interaction carry search time overflow");
    }
    const bool interval_due = accumulated >= interval;
    search_seconds_exact_ = interval_due
        ? std::fmod(accumulated, interval)
        : accumulated;
    search_seconds_ = static_cast<float>(search_seconds_exact_);

    if (search_pending_ || interval_due) {
        search_pending_ = false;
        const NormalizedQuery query = recorded_query(
            *features_, locomotion, hand_, desired_object, affordance_);
        bool terminal_current = false;
        if (recorded_range_index_ >= 0 &&
            static_cast<size_t>(recorded_range_index_) <
                ranges_.recorded.size()) {
            const CarryRange& current = ranges_.recorded[
                static_cast<size_t>(recorded_range_index_)];
            terminal_current = source_frame_exact_ >=
                static_cast<double>(current.stop_frame - 1);
        }
        for (size_t dimension = kFeatureGroupStarts[0];
             dimension < kFeatureGroupStops[1];
             ++dimension) {
            if (!finite(query[dimension])) {
                throw std::invalid_argument(
                    "interaction carry invalid pose/trajectory query");
            }
        }
        int32_t best_range = -1;
        int32_t best_frame = -1;
        double best_cost = std::numeric_limits<double>::infinity();
        for (size_t range_index = 0U;
             range_index < ranges_.recorded.size();
             ++range_index) {
            const CarryRange& range = ranges_.recorded[range_index];
            if (!compatible_range(
                    *database_, range, hand_, affordance_, config_)) {
                continue;
            }
            for (int32_t frame = range.start_frame;
                 frame < range.stop_frame;
                 ++frame) {
                const double cost = pose_trajectory_cost(
                    *features_, query, frame);
                if (std::isfinite(cost) && cost < best_cost) {
                    best_cost = cost;
                    best_range = static_cast<int32_t>(range_index);
                    best_frame = frame;
                }
            }
        }

        bool continue_current = false;
        if (!terminal_current && best_range >= 0 &&
            recorded_range_index_ >= 0 &&
            static_cast<size_t>(recorded_range_index_) <
                ranges_.recorded.size()) {
            const CarryRange& current = ranges_.recorded[
                static_cast<size_t>(recorded_range_index_)];
            if (compatible_range(
                    *database_, current, hand_, affordance_, config_)) {
                const int32_t frame = std::clamp(
                    static_cast<int32_t>(std::floor(source_frame_exact_)),
                    current.start_frame,
                    current.stop_frame - 1);
                const double cost = pose_trajectory_cost(
                    *features_, query, frame);
                continue_current =
                    std::isfinite(cost) && equal_cost(cost, best_cost);
            }
        }
        if (!continue_current) {
            recorded_range_index_ = best_range;
            if (best_frame >= 0) {
                if (recorded_selection_epoch_ ==
                    std::numeric_limits<int64_t>::max()) {
                    throw std::invalid_argument(
                        "interaction carry recorded selection overflow");
                }
                ++recorded_selection_epoch_;
                const CarryRange& selected = ranges_.recorded[
                    static_cast<size_t>(best_range)];
                source_frame_exact_ = terminal_current
                    ? static_cast<double>(selected.start_frame)
                    : static_cast<double>(best_frame);
            } else {
                source_frame_exact_ = 0.0;
            }
        }
    }

    CarryCandidate candidate{};
    bool candidate_ready = false;
    if (recorded_range_index_ >= 0) {
        const CarryRange& range = ranges_.recorded.at(
            static_cast<size_t>(recorded_range_index_));
        const double last = static_cast<double>(range.stop_frame - 1);
        const double advance = static_cast<double>(dt) * 25.0;
        const double remaining = last - source_frame_exact_;
        source_frame_exact_ = advance >= remaining
            ? last
            : source_frame_exact_ + advance;

        Pose recorded_pose = sample_recorded_pose(
            *database_, range, source_frame_exact_);
        const Transform source_root = root_transform(recorded_pose);
        const Transform alignment = planar_alignment(
            source_root, root_transform(locomotion.pose));
        recorded_pose.positions[g1_skeleton::Simulation] =
            locomotion.pose.positions[g1_skeleton::Simulation];
        recorded_pose.rotations[g1_skeleton::Simulation] =
            locomotion.pose.rotations[g1_skeleton::Simulation];
        recorded_pose.velocities[g1_skeleton::Simulation] =
            locomotion.pose.velocities[g1_skeleton::Simulation];
        recorded_pose.angular_velocities[g1_skeleton::Simulation] =
            locomotion.pose.angular_velocities[g1_skeleton::Simulation];

        const Transform mapped_object = compose(
            alignment,
            sample_recorded_object(
                *database_, range, source_frame_exact_));
        if (!valid_transform(mapped_object)) {
            throw std::invalid_argument(
                "interaction carry invalid recorded object mapping");
        }
        if (continuous_object(
                desired_object,
                mapped_object,
                config_,
                ik_config_)) {
            Pose solved = recorded_pose;
            const IKResult ik = solve_hand_ik(
                solved,
                hand_,
                compose(mapped_object, affordance_.hand_in_object),
                ik_config_);
            if (ik.accepted) {
                const Transform published = compose(
                    hand_transform(solved, hand_),
                    inverse(affordance_.hand_in_object));
                if (!valid_transform(published)) {
                    throw std::invalid_argument(
                        "interaction carry invalid recorded solved grasp");
                }
                if (continuous_object(
                        desired_object,
                        published,
                        config_,
                        ik_config_)) {
                    if (recorded_selection_epoch_ <= 0) {
                        throw std::logic_error(
                            "interaction carry recorded selection lacks epoch");
                    }
                    candidate.pose = solved;
                    candidate.object = published;
                    candidate.recorded = true;
                    candidate.directly_solved = true;
                    candidate.seam_key = recorded_selection_epoch_;
                    candidate_ready = true;
                }
            }
        }
    }

    if (!candidate_ready) {
        Pose output = locomotion.pose;
        for (const HingeJoint& joint : arm(hand_)) {
            const size_t bone = static_cast<size_t>(joint.bone);
            output.rotations[bone] = final_hold_pose_.rotations[bone];
        }
        constexpr std::array<int32_t, 3> kSpine = {
            g1_skeleton::Spine,
            g1_skeleton::Spine1,
            g1_skeleton::Spine2,
        };
        for (int32_t bone : kSpine) {
            output.rotations[static_cast<size_t>(bone)] =
                quat_nlerp_shortest(
                    locomotion.pose.rotations[static_cast<size_t>(bone)],
                    final_hold_pose_.rotations[static_cast<size_t>(bone)],
                    config_.spine_weight);
        }
        const Hand inactive = hand_ == Hand::Left
            ? Hand::Right
            : Hand::Left;
        for (const HingeJoint& joint : arm(inactive)) {
            const size_t bone = static_cast<size_t>(joint.bone);
            output.rotations[bone] = quat_nlerp_shortest(
                locomotion.pose.rotations[bone],
                final_hold_pose_.rotations[bone],
                config_.inactive_arm_weight);
        }

        const auto accept_layered = [&](Pose requested) {
            Pose solved = requested;
            const IKResult ik = solve_hand_ik(
                solved,
                hand_,
                compose(desired_object, affordance_.hand_in_object),
                ik_config_);
            if (!ik.accepted) return false;
            const Transform published = compose(
                hand_transform(solved, hand_),
                inverse(affordance_.hand_in_object));
            if (!valid_transform(published)) {
                throw std::invalid_argument(
                    "interaction carry invalid layered solved grasp");
            }
            if (!continuous_object(
                    desired_object, published, config_, ik_config_)) {
                return false;
            }
            candidate.pose = solved;
            candidate.object = published;
            candidate.recorded = false;
            candidate.directly_solved = true;
            candidate.seam_key = kLayeredSeamKey;
            return true;
        };

        candidate_ready = accept_layered(output);
        if (!candidate_ready) {
            Pose seeded = output;
            for (const HingeJoint& joint : arm(hand_)) {
                const size_t bone = static_cast<size_t>(joint.bone);
                seeded.rotations[bone] = last_safe_pose_.rotations[bone];
            }
            candidate_ready = accept_layered(seeded);
        }
        if (!candidate_ready) {
            candidate.pose = output;
            candidate.object = desired_object;
            candidate.recorded = false;
            candidate.directly_solved = false;
            candidate.seam_key = kLayeredSeamKey;
            candidate_ready = true;
        }
    }

    if (!candidate_ready || !valid_transform(candidate.object)) {
        throw std::logic_error("interaction carry candidate was not produced");
    }

    const bool changed_key = transition_active_
        ? candidate.seam_key != transition_seam_key_
        : candidate.seam_key != published_seam_key_;
    if (changed_key ||
        (!transition_active_ && !candidate.directly_solved)) {
        transition_source_pose_ = last_safe_pose_;
        transition_progress_ = 0.0F;
        transition_seam_key_ = candidate.seam_key;
        transition_active_ = true;
    }

    if (!transition_active_) {
        const Transform published_in_root = compose(
            inverse(live_root), candidate.object);
        if (!valid_transform(published_in_root)) {
            throw std::invalid_argument(
                "interaction carry invalid direct solved grasp");
        }
        object_world_ = candidate.object;
        object_in_root_ = published_in_root;
        recorded_ = candidate.recorded;
        last_safe_pose_ = candidate.pose;
        transaction.commit();
        return candidate.pose;
    }

    const bool completed_before_update = transition_progress_ >= 1.0F;
    const float requested_progress = std::min(
        1.0F,
        transition_progress_ + dt / kCarrySeamSeconds);
    IKConfig transition_ik = ik_config_;
    transition_ik.accepted_position_m = std::min(
        transition_ik.accepted_position_m,
        config_.maximum_grasp_drift_m);
    transition_ik.accepted_orientation_radians = std::min(
        transition_ik.accepted_orientation_radians,
        config_.maximum_grasp_drift_radians);

    float trial_progress = requested_progress;
    for (int32_t attempt = 0; attempt < kCarrySeamAttempts; ++attempt) {
        for (int preserve_index = 0; preserve_index < 2; ++preserve_index) {
            const bool preserve_active_arm = preserve_index != 0;
            Pose transition = blend_carry_transition(
                transition_source_pose_,
                candidate.pose,
                last_safe_pose_,
                hand_,
                preserve_active_arm,
                trial_progress);
            const IKResult transition_result = solve_hand_ik(
                transition,
                hand_,
                compose(candidate.object, affordance_.hand_in_object),
                transition_ik);
            if (!transition_result.accepted) continue;
            const Transform transition_object = compose(
                hand_transform(transition, hand_),
                inverse(affordance_.hand_in_object));
            if (!valid_transform(transition_object) ||
                !continuous_object(
                    desired_object,
                    transition_object,
                    config_,
                    transition_ik)) {
                continue;
            }
            const Transform transition_object_in_root = compose(
                inverse(live_root), transition_object);
            if (!valid_transform(transition_object_in_root)) {
                throw std::invalid_argument(
                    "interaction carry invalid transition solved grasp");
            }
            transition_progress_ = trial_progress;
            object_world_ = transition_object;
            object_in_root_ = transition_object_in_root;
            recorded_ = candidate.recorded;
            last_safe_pose_ = transition;
            if (completed_before_update &&
                trial_progress >= 1.0F &&
                candidate.directly_solved &&
                !preserve_active_arm) {
                transition_active_ = false;
                published_seam_key_ = candidate.seam_key;
            }
            transaction.commit();
            return transition;
        }
        if (!(trial_progress > transition_progress_)) break;
        trial_progress = transition_progress_ +
            0.5F * (trial_progress - transition_progress_);
    }

    const Pose safe = remap_safe_pose_to_live_root(
        last_safe_pose_, locomotion.pose);
    const Transform safe_object = compose(
        hand_transform(safe, hand_),
        inverse(affordance_.hand_in_object));
    if (!valid_transform(safe_object) ||
        !continuous_object(
            desired_object, safe_object, config_, transition_ik)) {
        throw std::invalid_argument(
            "interaction carry invalid transition safe grasp");
    }
    const Transform safe_object_in_root = compose(
        inverse(live_root), safe_object);
    if (!valid_transform(safe_object_in_root)) {
        throw std::invalid_argument(
            "interaction carry invalid transition safe anchor");
    }
    object_world_ = safe_object;
    object_in_root_ = safe_object_in_root;
    recorded_ = candidate.recorded;
    last_safe_pose_ = safe;
    transaction.commit();
    return safe;
}

bool CarryController::recorded() const {
    return recorded_;
}

Transform CarryController::object_world() const {
    return object_world_;
}

}  // namespace interaction
