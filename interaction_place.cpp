#include "interaction_place.h"
#include "interaction_place_collision.h"
#include "interaction_rotation_gate.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace interaction {
namespace {

constexpr float kCanonicalFps = 25.0F;
constexpr float kMinimumPlaybackSpeed = 0.85F;
constexpr float kMaximumPlaybackSpeed = 1.15F;
constexpr float kMaximumEntryRootM = 0.25F;
constexpr float kMaximumEntryYawRadians = 0.436332313F;
constexpr float kMaximumRequestPositionM = 0.12F;
constexpr float kMaximumRequestOrientationRadians = 0.436332313F;
constexpr float kRecordedPositionToleranceM = 0.02F;
constexpr float kRecordedOrientationToleranceRadians = 0.174532925F;
constexpr float kBoundsToleranceM = 0.001F;
constexpr float kUnitTolerance = 0.001F;
constexpr float kMinimumNorm = 1.0e-12F;
constexpr size_t kStableHoldSamples = 5U;
constexpr double kSourceFrameSnapTolerance = 1.0e-6;
constexpr size_t kRootBone = static_cast<size_t>(g1_skeleton::Simulation);
constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

bool finite(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

bool finite(vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(quat value) {
    return finite(value.w) && finite(value.x) && finite(value.y) &&
           finite(value.z);
}

bool valid_rotation(quat value) {
    if (!finite(value)) return false;
    const double squared_norm =
        static_cast<double>(value.w) * value.w +
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z;
    if (!(squared_norm > static_cast<double>(kMinimumNorm) * kMinimumNorm)) {
        return false;
    }
    return std::abs(std::sqrt(squared_norm) - 1.0) <= kUnitTolerance;
}

bool valid_transform(const Transform& value) {
    return finite(value.position) && valid_rotation(value.rotation);
}

bool positive(vec3 value) {
    return finite(value) && value.x > 0.0F && value.y > 0.0F &&
           value.z > 0.0F;
}

bool valid_bounds(ObjectLocalBounds value) {
    return finite(value.center_object) && positive(value.half_extents_object);
}

double distance(vec3 left, vec3 right) {
    const double x = static_cast<double>(left.x) - right.x;
    const double y = static_cast<double>(left.y) - right.y;
    const double z = static_cast<double>(left.z) - right.z;
    return std::sqrt(x * x + y * y + z * z);
}

quat normalized(quat value) {
    const double norm = std::sqrt(
        static_cast<double>(value.w) * value.w +
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z);
    if (!(norm > kMinimumNorm) ||
        !rotation_gate::finite_bits(norm)) {
        return quat();
    }
    const float inverse_norm = static_cast<float>(1.0 / norm);
    return value * inverse_norm;
}

double rotation_error(quat left, quat right) {
    const rotation_gate::Measure measured = rotation_gate::measure(
        left, right);
    return measured.valid
        ? measured.radians
        : std::numeric_limits<double>::max();
}

float yaw_radians(quat rotation) {
    const vec3 forward = quat_mul_vec3(
        normalized(rotation), vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(forward.x, forward.z);
}

float shortest_angle(float value) {
    return std::atan2(std::sin(value), std::cos(value));
}

float planar_distance(vec3 left, vec3 right) {
    return std::hypot(left.x - right.x, left.z - right.z);
}

Transform pose_root(const Pose& pose) {
    const WorldPose world = world_pose(pose);
    return {world.positions[kRootBone], world.rotations[kRootBone]};
}

size_t hand_bone(Hand hand) {
    return hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : static_cast<size_t>(g1_skeleton::RightWrist);
}

struct RotationTransform {
    Transform value{};
    rotation_gate::Rotation rotation{};
};

RotationTransform with_rotation_evidence(Transform value) {
    return {value, rotation_gate::from_quat(value.rotation)};
}

Transform pose_hand(const Pose& pose, Hand hand) {
    const WorldPose world = world_pose(pose);
    const size_t bone = hand_bone(hand);
    return {world.positions[bone], normalized(world.rotations[bone])};
}

rotation_gate::Rotation pose_hand_rotation_evidence(
    const Pose& pose,
    Hand hand) {
    return world_rotation_evidence(pose, hand_bone(hand));
}

RotationTransform pose_hand_for_gate(const Pose& pose, Hand hand) {
    return {
        pose_hand(pose, hand),
        pose_hand_rotation_evidence(pose, hand),
    };
}

RotationTransform mapped_pose_hand_for_gate(
    const Pose& source_pose,
    Transform scene,
    Hand hand) {
    return {
        compose(scene, pose_hand(source_pose, hand)),
        rotation_gate::multiply(
            rotation_gate::from_quat(scene.rotation),
            pose_hand_rotation_evidence(source_pose, hand)),
    };
}

Pose rigidly_mapped_pose(Pose pose, Transform mapping) {
    const Transform mapped_root = compose(
        mapping,
        Transform{pose.positions[kRootBone], pose.rotations[kRootBone]});
    pose.positions[kRootBone] = mapped_root.position;
    pose.rotations[kRootBone] = mapped_root.rotation;
    pose.velocities[kRootBone] = quat_mul_vec3(
        mapping.rotation, pose.velocities[kRootBone]);
    pose.angular_velocities[kRootBone] = quat_mul_vec3(
        mapping.rotation, pose.angular_velocities[kRootBone]);
    return pose;
}

Transform planar_alignment(Transform source, Transform target) {
    const float yaw = shortest_angle(
        yaw_radians(target.rotation) - yaw_radians(source.rotation));
    const quat rotation = quat_from_angle_axis(
        yaw, vec3(0.0F, 1.0F, 0.0F));
    const vec3 rotated_source = quat_mul_vec3(rotation, source.position);
    return {
        vec3(
            target.position.x - rotated_source.x,
            0.0F,
            target.position.z - rotated_source.z),
        rotation,
    };
}

bool pose_finite(const Pose& pose) {
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
    return pose.foot_contacts[0] <= 1U && pose.foot_contacts[1] <= 1U;
}

bool same_float(float left, float right) {
    if (left == 0.0F && right == 0.0F) return true;
    uint32_t left_bits = 0U;
    uint32_t right_bits = 0U;
    std::memcpy(&left_bits, &left, sizeof(left_bits));
    std::memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

bool same_vec(vec3 left, vec3 right) {
    return same_float(left.x, right.x) && same_float(left.y, right.y) &&
           same_float(left.z, right.z);
}

std::array<float, 4> canonical_rotation_components(quat value) {
    std::array<float, 4> components = {
        value.w, value.x, value.y, value.z};
    bool negate = false;
    for (float component : components) {
        if (component == 0.0F) continue;
        negate = component < 0.0F;
        break;
    }
    if (negate) {
        for (float& component : components) component = -component;
    }

    const double squared_norm =
        static_cast<double>(components[0]) * components[0] +
        static_cast<double>(components[1]) * components[1] +
        static_cast<double>(components[2]) * components[2] +
        static_cast<double>(components[3]) * components[3];
    const double inverse_norm = 1.0 / std::sqrt(squared_norm);
    for (float& component : components) {
        component = static_cast<float>(
            static_cast<double>(component) * inverse_norm);
        if (component == 0.0F) component = 0.0F;
    }
    return components;
}

bool same_rotation(quat left, quat right) {
    if (!valid_rotation(left) || !valid_rotation(right)) return false;
    const std::array<float, 4> canonical_left =
        canonical_rotation_components(left);
    const std::array<float, 4> canonical_right =
        canonical_rotation_components(right);
    for (size_t component = 0; component < canonical_left.size();
         ++component) {
        if (!same_float(
                canonical_left[component], canonical_right[component])) {
            return false;
        }
    }
    return true;
}

bool same_transform(Transform left, Transform right) {
    return same_vec(left.position, right.position) &&
           same_rotation(left.rotation, right.rotation);
}

bool same_place_affordance(
    const PlaceAffordance& left,
    const PlaceAffordance& right) {
    return left.id == right.id &&
           same_transform(left.object_in_surface, right.object_in_surface) &&
           same_vec(left.support_point_object, right.support_point_object) &&
           same_vec(
               left.approach_direction_surface,
               right.approach_direction_surface) &&
           same_float(left.clearance_radius, right.clearance_radius);
}

const PlaceAffordance* find_affordance(
    const PlacementSurface& surface,
    uint32_t id) {
    for (const PlaceAffordance& affordance : surface.affordances) {
        if (affordance.id == id) return &affordance;
    }
    return nullptr;
}

class CanonicalHash {
public:
    void tag(uint32_t value) {
        byte(0xd3U);
        u32(value);
    }

    void boolean(bool value) {
        byte(value ? 1U : 0U);
    }

    void u8(uint8_t value) {
        byte(value);
    }

    void u32(uint32_t value) {
        for (size_t shift = 0; shift < 32U; shift += 8U) {
            byte(static_cast<uint8_t>((value >> shift) & 0xffU));
        }
    }

    void i32(int32_t value) {
        uint32_t bits = 0U;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        u32(bits);
    }

    void u64(uint64_t value) {
        for (size_t shift = 0; shift < 64U; shift += 8U) {
            byte(static_cast<uint8_t>((value >> shift) & 0xffU));
        }
    }

    void text(std::string_view value) {
        u64(static_cast<uint64_t>(value.size()));
        for (char character : value) {
            byte(static_cast<uint8_t>(character));
        }
    }

    void scalar(float value) {
        uint32_t bits = 0U;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        if ((bits & 0x7fffffffU) == 0U) bits = 0U;
        u32(bits);
    }

    void vector(vec3 value) {
        scalar(value.x);
        scalar(value.y);
        scalar(value.z);
    }

    void rotation(quat value) {
        if (!valid_rotation(value)) {
            byte(0U);
            scalar(value.w);
            scalar(value.x);
            scalar(value.y);
            scalar(value.z);
            return;
        }
        byte(1U);
        for (float component : canonical_rotation_components(value)) {
            scalar(component);
        }
    }

    void transform(const Transform& value) {
        vector(value.position);
        rotation(value.rotation);
    }

    uint64_t finish() const {
        return state_ == 0U ? 0x9e3779b97f4a7c15ULL : state_;
    }

private:
    void byte(uint8_t value) {
        state_ ^= value;
        state_ *= kFnvPrime;
    }

    uint64_t state_ = kFnvOffset;
};

void hash_pose(CanonicalHash& hash, const Pose& pose) {
    hash.tag(0x5001U);
    for (vec3 value : pose.positions) hash.vector(value);
    for (vec3 value : pose.velocities) hash.vector(value);
    for (quat value : pose.rotations) hash.rotation(value);
    for (vec3 value : pose.angular_velocities) hash.vector(value);
    for (float value : pose.hand_dof) hash.scalar(value);
    for (float value : pose.hand_dof_velocities) hash.scalar(value);
    for (uint8_t value : pose.foot_contacts) hash.u8(value);
}

void hash_bounds(CanonicalHash& hash, ObjectLocalBounds value) {
    hash.tag(0x5002U);
    hash.vector(value.center_object);
    hash.vector(value.half_extents_object);
}

void hash_grasp(CanonicalHash& hash, const GraspAffordance& value) {
    hash.tag(0x5003U);
    hash.u32(value.id);
    hash.u8(static_cast<uint8_t>(value.hand));
    hash.transform(value.hand_in_object);
    hash.vector(value.approach_direction_object);
    hash.scalar(value.clearance_radius);
    hash.u64(static_cast<uint64_t>(value.interaction_slots.size()));
    for (const GraspInteractionSlot& slot : value.interaction_slots) {
        hash.u32(slot.id);
        hash.scalar(slot.root_x_object_m);
        hash.scalar(slot.root_z_object_m);
        hash.scalar(slot.root_yaw_object_radians);
    }
}

void hash_place_affordance(CanonicalHash& hash, const PlaceAffordance& value) {
    hash.tag(0x5004U);
    hash.u32(value.id);
    hash.transform(value.object_in_surface);
    hash.vector(value.support_point_object);
    hash.vector(value.approach_direction_surface);
    hash.scalar(value.clearance_radius);
}

void hash_surface(CanonicalHash& hash, const PlacementSurface& value) {
    hash.tag(0x5005U);
    hash.u64(value.handle.id);
    hash.u32(value.handle.generation);
    hash.transform(value.surface_world);
    hash.transform(value.support_volume_world);
    hash.vector(value.support_volume_size);
    hash.scalar(value.half_extent_x_m);
    hash.scalar(value.half_extent_z_m);
    hash.scalar(value.overhead_clearance_m);
    hash.u64(static_cast<uint64_t>(value.affordances.size()));
    for (const PlaceAffordance& affordance : value.affordances) {
        hash_place_affordance(hash, affordance);
    }
}

void hash_timing(CanonicalHash& hash, const PlaceTimingConfig& value) {
    hash.tag(0x5006U);
    hash.scalar(value.canonical_fps);
    hash.scalar(value.playback_speed);
    hash.scalar(value.entry_blend_seconds);
    hash.scalar(value.reversed_commit_seconds);
    hash.scalar(value.maximum_alignment_seconds);
}

void hash_match(CanonicalHash& hash, const PlaceMatchConfig& value) {
    hash.tag(0x5007U);
    hash.scalar(value.maximum_entry_root_error_m);
    hash.scalar(value.maximum_entry_yaw_error_radians);
}

void hash_ik(CanonicalHash& hash, const IKConfig& value) {
    hash.tag(0x5008U);
    hash.scalar(value.maximum_request_position_m);
    hash.scalar(value.maximum_request_orientation_radians);
    hash.scalar(value.accepted_position_m);
    hash.scalar(value.accepted_orientation_radians);
    hash.scalar(value.damping);
    hash.scalar(value.finite_difference_radians);
    hash.scalar(value.orientation_scale_m_per_radian);
    hash.scalar(value.maximum_step_radians);
    hash.i32(value.maximum_iterations);
}

void hash_pickup_candidate(CanonicalHash& hash, const MatchCandidate& value) {
    hash.tag(0x5009U);
    hash.i32(value.clip);
    hash.i32(value.entry_frame);
    hash.i32(value.contact_frame);
    hash.i32(value.lift_frame);
    hash.i32(value.hold_frame);
    hash.transform(value.scene_from_source);
    hash.vector(value.entry_root_offset);
    hash.scalar(value.entry_yaw_offset);
    hash.scalar(value.total_cost);
    for (float cost : value.group_costs) hash.scalar(cost);
}

void hash_recorded_clip(CanonicalHash& hash, const RecordedPlaceClip& value) {
    hash.tag(0x5010U);
    hash.u64(value.id);
    hash.u64(value.object_profile_id);
    hash.u32(value.fps_numerator);
    hash.u32(value.fps_denominator);
    hash.u64(static_cast<uint64_t>(value.poses.size()));
    for (const Pose& pose : value.poses) hash_pose(hash, pose);
    hash.u64(static_cast<uint64_t>(value.object_poses.size()));
    for (const Transform& object : value.object_poses) hash.transform(object);
    hash.u64(static_cast<uint64_t>(value.active_hand_contacts.size()));
    for (uint8_t contact : value.active_hand_contacts) hash.u8(contact);
    hash.i32(value.entry_frame);
    hash.i32(value.commit_frame);
    hash.i32(value.release_frame);
    hash.i32(value.retract_stop_frame);
    hash.u8(static_cast<uint8_t>(value.hand));
    hash.transform(value.hand_in_object);
    hash_bounds(hash, value.object_bounds);
    hash_surface(hash, value.source_surface);
    hash.u32(value.source_affordance_id);
}

void hash_precomputed_reversed_clip(
    CanonicalHash& hash,
    const PrecomputedReversedPickupClip& value) {
    hash.tag(0x5013U);
    hash.u64(value.id);
    hash.u64(value.object_profile_id);
    hash.text(value.sequence_id);
    hash.i32(value.clip);
    hash.i32(value.entry_frame);
    hash.i32(value.contact_frame);
    hash.i32(value.lift_frame);
    hash.i32(value.hold_frame);
    hash.i32(value.reverse_start_frame);
    hash.u8(static_cast<uint8_t>(value.hand));
    hash.transform(value.source_hand_in_object);
    hash_bounds(hash, value.source_object_bounds);
    hash_surface(hash, value.source_surface);
    hash.u32(value.source_affordance_id);
    hash.scalar(value.source_support_height_m);
    hash.scalar(value.source_grasp_height_above_support_m);
}

void hash_float_span(
    CanonicalHash& hash,
    const std::vector<float>& values,
    size_t begin,
    size_t count) {
    hash.u64(static_cast<uint64_t>(begin));
    hash.u64(static_cast<uint64_t>(count));
    if (begin > values.size() || count > values.size() - begin) {
        hash.boolean(false);
        return;
    }
    hash.boolean(true);
    for (size_t index = begin; index < begin + count; ++index) {
        hash.scalar(values[index]);
    }
}

void hash_byte_span(
    CanonicalHash& hash,
    const std::vector<uint8_t>& values,
    size_t begin,
    size_t count) {
    hash.u64(static_cast<uint64_t>(begin));
    hash.u64(static_cast<uint64_t>(count));
    if (begin > values.size() || count > values.size() - begin) {
        hash.boolean(false);
        return;
    }
    hash.boolean(true);
    for (size_t index = begin; index < begin + count; ++index) {
        hash.u8(values[index]);
    }
}

void hash_quaternion_span(
    CanonicalHash& hash,
    const std::vector<float>& values,
    size_t begin_quaternion,
    size_t quaternion_count) {
    hash.u64(static_cast<uint64_t>(begin_quaternion));
    hash.u64(static_cast<uint64_t>(quaternion_count));
    if (begin_quaternion > std::numeric_limits<size_t>::max() / 4U ||
        quaternion_count > std::numeric_limits<size_t>::max() / 4U) {
        hash.boolean(false);
        return;
    }
    const size_t begin = begin_quaternion * 4U;
    const size_t count = quaternion_count * 4U;
    if (begin > values.size() || count > values.size() - begin) {
        hash.boolean(false);
        return;
    }
    hash.boolean(true);
    for (size_t index = begin; index < begin + count; index += 4U) {
        hash.rotation(quat(
            values[index],
            values[index + 1U],
            values[index + 2U],
            values[index + 3U]));
    }
}

void hash_int_span(
    CanonicalHash& hash,
    const std::vector<int32_t>& values,
    size_t begin,
    size_t count) {
    hash.u64(static_cast<uint64_t>(begin));
    hash.u64(static_cast<uint64_t>(count));
    if (begin > values.size() || count > values.size() - begin) {
        hash.boolean(false);
        return;
    }
    hash.boolean(true);
    for (size_t index = begin; index < begin + count; ++index) {
        hash.i32(values[index]);
    }
}

void hash_pickup_prefix(
    CanonicalHash& hash,
    const Database& database,
    const MatchCandidate& candidate,
    int32_t prefix_stop) {
    hash.tag(0x5011U);
    hash.u32(database.version);
    hash.u32(database.endian_marker);
    hash.u32(database.fps_numerator);
    hash.u32(database.fps_denominator);
    hash.u32(database.bone_count);
    hash.u32(database.hand_dof_count);
    hash.i32(candidate.clip);

    if (candidate.clip < 0 ||
        static_cast<size_t>(candidate.clip) >= database.range_starts.size() ||
        static_cast<size_t>(candidate.clip) >= database.range_stops.size()) {
        hash.boolean(false);
        return;
    }
    hash.boolean(true);
    const size_t clip = static_cast<size_t>(candidate.clip);
    const int32_t range_start = database.range_starts[clip];
    const int32_t range_stop = database.range_stops[clip];
    if (clip < database.active_hands.size()) {
        hash.boolean(true);
        hash.u8(database.active_hands[clip]);
    } else {
        hash.boolean(false);
    }

    hash.u64(static_cast<uint64_t>(database.parents.size()));
    for (int32_t parent : database.parents) hash.i32(parent);

    const size_t clip3 = clip * 3U;
    hash_float_span(hash, database.table_positions, clip3, 3U);
    hash_quaternion_span(hash, database.table_rotations, clip, 1U);
    hash_float_span(hash, database.table_sizes, clip3, 3U);
    hash_float_span(hash, database.object_dimensions, clip3, 3U);
    hash_float_span(hash, database.grasp_positions_object, clip3, 3U);
    hash_quaternion_span(hash, database.grasp_rotations_object, clip, 1U);
    hash_float_span(
        hash, database.approach_directions_object, clip3, 3U);

    const int32_t begin_frame = std::max(range_start, candidate.entry_frame);
    const int32_t end_frame = std::min(prefix_stop, range_stop - 1);
    if (begin_frame < 0 || end_frame < begin_frame) {
        hash.boolean(false);
        return;
    }
    hash.boolean(true);
    hash.i32(begin_frame);
    hash.i32(end_frame);
    const size_t begin = static_cast<size_t>(begin_frame);
    const size_t count = static_cast<size_t>(end_frame - begin_frame + 1);
    const size_t bones = g1_skeleton::BoneCount;
    hash_float_span(hash, database.positions, begin * bones * 3U,
                    count * bones * 3U);
    hash_float_span(hash, database.velocities, begin * bones * 3U,
                    count * bones * 3U);
    hash_quaternion_span(
        hash, database.rotations, begin * bones, count * bones);
    hash_float_span(hash, database.angular_velocities, begin * bones * 3U,
                    count * bones * 3U);
    hash_byte_span(hash, database.foot_contacts, begin * 2U, count * 2U);
    hash_byte_span(hash, database.hand_contacts, begin * 2U, count * 2U);
    hash_float_span(hash, database.hand_dof, begin * 14U, count * 14U);
    hash_float_span(
        hash, database.hand_dof_velocities, begin * 14U, count * 14U);
    hash_byte_span(hash, database.phases, begin, count);
    hash_float_span(hash, database.time_to_contact, begin, count);
    hash_float_span(hash, database.object_positions, begin * 3U, count * 3U);
    hash_quaternion_span(hash, database.object_rotations, begin, count);
    hash_float_span(hash, database.object_velocities, begin * 3U, count * 3U);
    hash_float_span(
        hash, database.object_angular_velocities, begin * 3U, count * 3U);
    hash_int_span(hash, database.source_frames, begin, count);
}

void hash_candidate_without_selection(
    CanonicalHash& hash,
    const PlaceCandidate& value) {
    hash.tag(0x5012U);
    hash.u8(static_cast<uint8_t>(value.mode));
    hash.u64(value.source_id);
    hash.scalar(value.source_support_height_m);
    hash.scalar(value.requested_support_height_m);
    hash.scalar(value.target_support_height_m);
    hash.scalar(value.requested_vertical_correction_m);
    hash_timing(hash, value.timing);
    hash_match(hash, value.match);
    hash_ik(hash, value.ik);
    hash.i32(value.clip);
    hash.i32(value.entry_frame);
    hash.i32(value.commit_frame);
    hash.i32(value.release_frame);
    hash.i32(value.stop_frame);
    hash.i32(value.direction);
    hash.transform(value.scene_from_source);
    hash.transform(value.staging_root_world);
    hash.vector(value.entry_root_offset);
    hash.scalar(value.entry_yaw_offset);
    hash.scalar(value.total_cost);
}

uint64_t ik_fingerprint(const IKConfig& config) {
    CanonicalHash hash;
    hash.tag(0x494b4346U);
    hash_ik(hash, config);
    return hash.finish();
}

int32_t default_pickup_prefix_stop(const PlaceMatchInput& input) {
    if (input.pickup_database == nullptr) return -1;
    const Database& database = *input.pickup_database;
    const int32_t clip = input.pickup_candidate.clip;
    if (clip < 0 || static_cast<size_t>(clip) >= database.range_stops.size()) {
        return -1;
    }
    const int64_t desired =
        static_cast<int64_t>(input.pickup_candidate.hold_frame) +
        static_cast<int64_t>(kStableHoldSamples - 1U);
    const int64_t last = static_cast<int64_t>(database.range_stops[clip]) - 1;
    return static_cast<int32_t>(std::clamp<int64_t>(desired, -1, last));
}

uint64_t selection_fingerprint(
    const PlaceMatchInput& input,
    const PlaceCandidate& selected,
    int32_t pickup_prefix_stop) {
    CanonicalHash hash;
    hash.tag(0x504c4143U);
    hash.u64(input.held_target.id);
    hash.u32(input.held_target.generation);
    hash_pickup_candidate(hash, input.pickup_candidate);
    hash_pose(hash, input.current_pose);
    hash.transform(input.current_object_world);
    hash.u64(input.held_object_profile_id);
    hash_bounds(hash, input.held_object_bounds);
    hash_grasp(hash, input.held_affordance);
    hash_surface(hash, input.surface);
    hash_place_affordance(hash, input.place_affordance);
    hash.vector(input.object_dimensions);
    hash_timing(hash, input.timing);
    hash_match(hash, input.match);
    hash_ik(hash, input.ik);
    if (input.pickup_database == nullptr) {
        hash.boolean(false);
    } else {
        hash.boolean(true);
        hash_pickup_prefix(
            hash,
            *input.pickup_database,
            input.pickup_candidate,
            pickup_prefix_stop);
    }
    if (input.library == nullptr) {
        hash.boolean(false);
    } else {
        hash.boolean(true);
        hash.u64(static_cast<uint64_t>(input.library->recorded.size()));
        for (const RecordedPlaceClip& row : input.library->recorded) {
            hash_recorded_clip(hash, row);
        }
        hash.u64(static_cast<uint64_t>(
            input.library->precomputed_reversed.size()));
        for (const PrecomputedReversedPickupClip& row :
             input.library->precomputed_reversed) {
            hash_precomputed_reversed_clip(hash, row);
            if (input.pickup_database == nullptr) {
                hash.boolean(false);
            } else {
                hash.boolean(true);
                MatchCandidate source{};
                source.clip = row.clip;
                source.entry_frame = row.entry_frame;
                source.contact_frame = row.contact_frame;
                source.lift_frame = row.lift_frame;
                source.hold_frame = row.hold_frame;
                hash_pickup_prefix(
                    hash,
                    *input.pickup_database,
                    source,
                    row.reverse_start_frame);
            }
        }
    }
    hash_candidate_without_selection(hash, selected);
    return hash.finish();
}

uint64_t reverse_source_id(
    const PlaceMatchInput& input,
    int32_t prefix_stop) {
    CanonicalHash hash;
    hash.tag(0x52565253U);
    hash_pickup_candidate(hash, input.pickup_candidate);
    hash_pickup_prefix(
        hash, *input.pickup_database, input.pickup_candidate, prefix_stop);
    return hash.finish();
}

std::optional<float> database_source_support_height(
    const Database& database,
    int32_t clip) {
    if (clip < 0) return std::nullopt;
    const size_t index = static_cast<size_t>(clip);
    if (index > std::numeric_limits<size_t>::max() / 4U) {
        return std::nullopt;
    }
    if (index * 3U > database.table_positions.size() ||
        3U > database.table_positions.size() - index * 3U ||
        index * 4U > database.table_rotations.size() ||
        4U > database.table_rotations.size() - index * 4U ||
        index * 3U > database.table_sizes.size() ||
        3U > database.table_sizes.size() - index * 3U) {
        return std::nullopt;
    }
    const size_t position = index * 3U;
    const size_t rotation = index * 4U;
    const Transform table{
        vec3(
            database.table_positions[position],
            database.table_positions[position + 1U],
            database.table_positions[position + 2U]),
        quat(
            database.table_rotations[rotation],
            database.table_rotations[rotation + 1U],
            database.table_rotations[rotation + 2U],
            database.table_rotations[rotation + 3U]),
    };
    const vec3 size(
        database.table_sizes[position],
        database.table_sizes[position + 1U],
        database.table_sizes[position + 2U]);
    if (!valid_transform(table) || !positive(size)) return std::nullopt;
    const Transform support = compose(
        table,
        Transform{vec3(0.0F, 0.5F * size.y, 0.0F), quat()});
    if (!valid_transform(support)) return std::nullopt;
    return support.position.y;
}

bool valid_timing(const PlaceTimingConfig& timing) {
    return finite(timing.canonical_fps) &&
           finite(timing.playback_speed) &&
           finite(timing.entry_blend_seconds) &&
           finite(timing.reversed_commit_seconds) &&
           finite(timing.maximum_alignment_seconds) &&
           timing.canonical_fps == kCanonicalFps &&
           timing.playback_speed >= kMinimumPlaybackSpeed &&
           timing.playback_speed <= kMaximumPlaybackSpeed &&
           timing.entry_blend_seconds > 0.0F &&
           timing.reversed_commit_seconds > 0.0F &&
           timing.maximum_alignment_seconds > 0.0F &&
           timing.maximum_alignment_seconds <= 1.0F &&
           timing.maximum_alignment_seconds >= timing.entry_blend_seconds &&
           timing.maximum_alignment_seconds >= timing.reversed_commit_seconds;
}

bool valid_match(const PlaceMatchConfig& match) {
    return finite(match.maximum_entry_root_error_m) &&
           finite(match.maximum_entry_yaw_error_radians) &&
           match.maximum_entry_root_error_m > 0.0F &&
           match.maximum_entry_yaw_error_radians > 0.0F &&
           match.maximum_entry_root_error_m <= kMaximumEntryRootM &&
           match.maximum_entry_yaw_error_radians <=
               kMaximumEntryYawRadians;
}

bool valid_place_ik(const IKConfig& config) {
    return finite(config.maximum_request_position_m) &&
           finite(config.maximum_request_orientation_radians) &&
           finite(config.accepted_position_m) &&
           finite(config.accepted_orientation_radians) &&
           finite(config.damping) &&
           finite(config.finite_difference_radians) &&
           finite(config.orientation_scale_m_per_radian) &&
           finite(config.maximum_step_radians) &&
           config.maximum_request_position_m >= 0.0F &&
           config.maximum_request_orientation_radians >= 0.0F &&
           config.accepted_position_m >= 0.0F &&
           config.accepted_orientation_radians >= 0.0F &&
           config.damping > 0.0F &&
           config.finite_difference_radians > 0.0F &&
           config.orientation_scale_m_per_radian > 0.0F &&
           config.maximum_step_radians > 0.0F &&
           config.maximum_iterations >= 0 &&
           config.maximum_request_position_m <= kMaximumRequestPositionM &&
           config.maximum_request_orientation_radians <=
               kMaximumRequestOrientationRadians;
}

struct ValidatedInput {
    const PlaceAffordance* requested_affordance = nullptr;
    Transform goal_object{};
    rotation_gate::Rotation goal_object_rotation_evidence{};
};

std::optional<Reason> validate_input(
    const PlaceMatchInput& input,
    ValidatedInput& validated) {
    if (input.pickup_database == nullptr || input.library == nullptr) {
        return Reason::PackUnavailable;
    }
    if (!valid_timing(input.timing) || !valid_match(input.match) ||
        !valid_place_ik(input.ik)) {
        return Reason::CorrectionLimit;
    }
    if (input.held_target.id == 0U || input.held_target.generation == 0U ||
        input.held_object_profile_id == 0U ||
        !valid_bounds(input.held_object_bounds) ||
        !positive(input.object_dimensions) ||
        input.held_affordance.id == 0U ||
        (input.held_affordance.hand != Hand::Left &&
         input.held_affordance.hand != Hand::Right) ||
        !valid_transform(input.held_affordance.hand_in_object) ||
        !finite(input.held_affordance.approach_direction_object) ||
        !finite(input.held_affordance.clearance_radius) ||
        input.held_affordance.clearance_radius < 0.0F ||
        !pose_finite(input.current_pose) ||
        !valid_transform(input.current_object_world)) {
        return Reason::TargetUnavailable;
    }

    const float approach_length = length(
        input.held_affordance.approach_direction_object);
    if (!finite(approach_length) ||
        std::abs(approach_length - 1.0F) > kUnitTolerance) {
        return Reason::TargetUnavailable;
    }

    validated.requested_affordance = find_affordance(
        input.surface, input.place_affordance.id);
    if (validated.requested_affordance == nullptr ||
        !same_place_affordance(
            *validated.requested_affordance, input.place_affordance)) {
        return Reason::SurfaceUnavailable;
    }
    try {
        const PlacementFit fit = evaluate_placement_fit(
            input.surface,
            *validated.requested_affordance,
            input.held_object_bounds);
        if (!fit.accepted) return fit.reason;
        validated.goal_object = placement_goal_world(
            input.surface,
            validated.requested_affordance->object_in_surface);
        validated.goal_object_rotation_evidence = rotation_gate::multiply(
            rotation_gate::from_quat(
                input.surface.surface_world.rotation),
            rotation_gate::from_quat(
                validated.requested_affordance
                    ->object_in_surface.rotation));
    } catch (const std::exception&) {
        return Reason::SurfaceUnavailable;
    }
    return std::nullopt;
}

bool within_transform_error(
    const RotationTransform& actual,
    const RotationTransform& expected,
    float maximum_position,
    float maximum_orientation) {
    return distance(
               actual.value.position,
               expected.value.position) <= maximum_position &&
           rotation_gate::within(
               actual.rotation,
               expected.rotation,
               maximum_orientation);
}

bool within_transform_error(
    Transform actual,
    Transform expected,
    float maximum_position,
    float maximum_orientation) {
    return within_transform_error(
        with_rotation_evidence(actual),
        with_rotation_evidence(expected),
        maximum_position,
        maximum_orientation);
}

RotationTransform goal_hand_for_gate(
    const PlaceMatchInput& input,
    const ValidatedInput& validated) {
    return {
        compose(
            validated.goal_object,
            input.held_affordance.hand_in_object),
        rotation_gate::multiply(
            validated.goal_object_rotation_evidence,
            rotation_gate::from_quat(
                input.held_affordance.hand_in_object.rotation)),
    };
}

struct SelectionFailures {
    bool blocked_path = false;
    bool placement_out_of_bounds = false;
    bool correction_limit = false;
    bool joint_limit = false;

    void remember(Reason reason) {
        if (reason == Reason::BlockedPath) {
            blocked_path = true;
        } else if (reason == Reason::PlacementOutOfBounds) {
            placement_out_of_bounds = true;
        } else if (reason == Reason::JointLimit) {
            joint_limit = true;
        } else if (reason == Reason::CorrectionLimit) {
            correction_limit = true;
        }
    }

    Reason strongest() const {
        if (joint_limit) return Reason::JointLimit;
        if (correction_limit) return Reason::CorrectionLimit;
        if (placement_out_of_bounds) return Reason::PlacementOutOfBounds;
        if (blocked_path) return Reason::BlockedPath;
        return Reason::NoCandidate;
    }
};

bool populate_candidate_heights(
    PlaceCandidate& candidate,
    const PlaceMatchInput& input,
    float source_support_height_m,
    SelectionFailures& failures) {
    const float requested_support_height_m =
        input.surface.surface_world.position.y;
    if (!finite(source_support_height_m) ||
        !finite(requested_support_height_m)) {
        failures.remember(Reason::CorrectionLimit);
        return false;
    }
    candidate.source_support_height_m = source_support_height_m;
    candidate.requested_support_height_m = requested_support_height_m;
    candidate.target_support_height_m = requested_support_height_m;
    candidate.requested_vertical_correction_m =
        candidate.target_support_height_m -
        candidate.source_support_height_m;
    if (!finite(candidate.requested_vertical_correction_m) ||
        std::abs(candidate.requested_vertical_correction_m) >
            kMaximumRequestPositionM) {
        failures.remember(Reason::CorrectionLimit);
        return false;
    }
    return true;
}

Reason mapped_clearance_failure(
    const std::vector<Transform>& object_samples,
    const PlaceMatchInput& input,
    Transform release_object) {
    if (object_samples.size() < 2U) {
        return Reason::BlockedPath;
    }
    try {
        const PlacementFit fit = evaluate_actual_placement_fit(
            input.surface,
            input.place_affordance,
            release_object,
            input.held_object_bounds);
        if (!fit.accepted) return fit.reason;
    } catch (const std::exception&) {
        return Reason::PlacementOutOfBounds;
    }
    const size_t release = object_samples.size() - 1U;
    for (size_t sample = 0; sample < release; ++sample) {
        if (place_collision::object_intersects_support(
                object_samples[sample],
                input.held_object_bounds,
                input.surface)) {
            return Reason::BlockedPath;
        }
    }
    for (size_t sample = 1U; sample < object_samples.size(); ++sample) {
        if (sample != release &&
            place_collision::swept_interval_intersects_support(
                object_samples[sample - 1U],
                object_samples[sample],
                input.held_object_bounds,
                input.surface)) {
            return Reason::BlockedPath;
        }
    }
    return Reason::None;
}

std::optional<Transform> solve_release_object(
    Pose source_pose,
    Transform scene,
    Hand hand,
    const RotationTransform& goal_hand,
    const PlaceMatchInput& input,
    SelectionFailures& failures) {
    Pose mapped_pose = rigidly_mapped_pose(source_pose, scene);
    const RotationTransform mapped_hand = mapped_pose_hand_for_gate(
        source_pose, scene, hand);
    if (!within_transform_error(
            mapped_hand,
            goal_hand,
            input.ik.maximum_request_position_m,
            input.ik.maximum_request_orientation_radians)) {
        failures.remember(Reason::CorrectionLimit);
        return std::nullopt;
    }
    const IKResult result = solve_hand_ik_with_rotation_evidence(
        mapped_pose,
        hand,
        goal_hand.value,
        goal_hand.rotation,
        rotation_gate::multiply(
            rotation_gate::from_quat(scene.rotation),
            rotation_gate::from_quat(
                source_pose.rotations[kRootBone])),
        input.ik);
    if (!result.accepted) {
        failures.remember(
            result.reason == Reason::JointLimit
                ? Reason::JointLimit
                : Reason::CorrectionLimit);
        return std::nullopt;
    }
    const Transform solved = compose(
        pose_hand(mapped_pose, hand),
        inverse(input.held_affordance.hand_in_object));
    if (!valid_transform(solved)) {
        failures.remember(Reason::CorrectionLimit);
        return std::nullopt;
    }
    return solved;
}

bool current_attachment_within_request(
    const PlaceMatchInput& input,
    Transform staging_root) {
    const Transform current_root = pose_root(input.current_pose);
    const Transform hypothetical = compose(staging_root, inverse(current_root));
    if (!valid_transform(hypothetical)) return false;
    // A common rigid staging transform cancels exactly in object-local space.
    // Compare there so an inclusive request boundary is not changed by two
    // avoidable world-space compose round trips.
    const Transform current_hand = pose_hand(
        input.current_pose, input.held_affordance.hand);
    const quat object_rotation = normalized(
        input.current_object_world.rotation);
    const RotationTransform actual_hand_in_object{
        Transform{
            quat_mul_vec3(
                quat_inv(object_rotation),
                current_hand.position - input.current_object_world.position),
            quat(),
        },
        rotation_gate::multiply(
            rotation_gate::inverse(rotation_gate::from_quat(
                input.current_object_world.rotation)),
            pose_hand_rotation_evidence(
                input.current_pose, input.held_affordance.hand)),
    };
    return within_transform_error(
        actual_hand_in_object,
        with_rotation_evidence(
            input.held_affordance.hand_in_object),
        input.ik.maximum_request_position_m,
        input.ik.maximum_request_orientation_radians);
}

PlaceResult reject(Reason reason) {
    return {false, PlaceCandidate{}, reason};
}

struct RecordedRow {
    size_t library_index = 0U;
    const RecordedPlaceClip* clip = nullptr;
    const PlaceAffordance* source_affordance = nullptr;
};

std::optional<int32_t> observable_event_ticks(
    int64_t source_samples,
    float playback_speed) {
    if (source_samples <= 0 ||
        !rotation_gate::finite_bits(playback_speed) ||
        !(playback_speed > 0.0F)) {
        return std::nullopt;
    }
    const double tick_value = std::ceil(
        static_cast<double>(source_samples) /
        static_cast<double>(playback_speed));
    if (!rotation_gate::finite_bits(tick_value) || tick_value < 1.0 ||
        tick_value > std::numeric_limits<int32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<int32_t>(tick_value);
}

bool observable_commit_timing_valid(
    int32_t source_samples,
    const PlaceTimingConfig& timing) {
    const std::optional<int32_t> ticks = observable_event_ticks(
        source_samples, timing.playback_speed);
    if (!ticks.has_value()) return false;
    const float elapsed_seconds = static_cast<float>(
        static_cast<double>(*ticks) /
        static_cast<double>(kCanonicalFps));
    return elapsed_seconds >= timing.entry_blend_seconds &&
           elapsed_seconds <= timing.maximum_alignment_seconds;
}

bool recorded_timing_valid(
    const RecordedPlaceClip& clip,
    const PlaceTimingConfig& timing) {
    const int32_t source_samples = clip.commit_frame - clip.entry_frame;
    return observable_commit_timing_valid(source_samples, timing);
}

std::optional<RecordedRow> validate_recorded_row(
    const RecordedPlaceClip& clip,
    size_t library_index,
    const PlaceTimingConfig& timing,
    SelectionFailures& failures) {
    if (clip.id == 0U || clip.object_profile_id == 0U ||
        clip.fps_numerator != 25U || clip.fps_denominator != 1U ||
        clip.poses.empty() ||
        clip.poses.size() != clip.object_poses.size() ||
        clip.poses.size() != clip.active_hand_contacts.size() ||
        clip.entry_frame < 0 ||
        !(clip.entry_frame < clip.commit_frame &&
          clip.commit_frame < clip.release_frame &&
          clip.release_frame < clip.retract_stop_frame) ||
        static_cast<size_t>(clip.retract_stop_frame) >= clip.poses.size() ||
        (clip.hand != Hand::Left && clip.hand != Hand::Right) ||
        !valid_transform(clip.hand_in_object) ||
        !valid_bounds(clip.object_bounds) ||
        clip.source_affordance_id == 0U ||
        !recorded_timing_valid(clip, timing)) {
        return std::nullopt;
    }

    for (size_t frame = 0; frame < clip.poses.size(); ++frame) {
        if (!pose_finite(clip.poses[frame]) ||
            !valid_transform(clip.object_poses[frame]) ||
            clip.active_hand_contacts[frame] > 1U) {
            return std::nullopt;
        }
    }
    for (int32_t frame = clip.entry_frame;
         frame <= clip.release_frame;
         ++frame) {
        if (clip.active_hand_contacts[static_cast<size_t>(frame)] != 1U) {
            return std::nullopt;
        }
        const RotationTransform actual = pose_hand_for_gate(
            clip.poses[static_cast<size_t>(frame)], clip.hand);
        const RotationTransform expected{
            compose(
                clip.object_poses[static_cast<size_t>(frame)],
                clip.hand_in_object),
            rotation_gate::multiply(
                rotation_gate::from_quat(
                    clip.object_poses[static_cast<size_t>(frame)].rotation),
                rotation_gate::from_quat(
                    clip.hand_in_object.rotation)),
        };
        if (!within_transform_error(
                actual,
                expected,
                kRecordedPositionToleranceM,
                kRecordedOrientationToleranceRadians)) {
            return std::nullopt;
        }
    }
    for (size_t frame = static_cast<size_t>(clip.release_frame + 1);
         frame < clip.active_hand_contacts.size();
         ++frame) {
        if (clip.active_hand_contacts[frame] != 0U) {
            return std::nullopt;
        }
    }

    const PlaceAffordance* source_affordance = find_affordance(
        clip.source_surface, clip.source_affordance_id);
    if (source_affordance == nullptr) return std::nullopt;
    try {
        const PlacementFit fit = evaluate_actual_placement_fit(
            clip.source_surface,
            *source_affordance,
            clip.object_poses[static_cast<size_t>(clip.release_frame)],
            clip.object_bounds);
        if (!fit.accepted) {
            failures.remember(fit.reason);
            return std::nullopt;
        }
    } catch (const std::exception&) {
        return std::nullopt;
    }
    return RecordedRow{library_index, &clip, source_affordance};
}

bool bounds_compatible(
    ObjectLocalBounds demonstrated,
    ObjectLocalBounds held) {
    return std::abs(demonstrated.center_object.x - held.center_object.x) <=
               kBoundsToleranceM &&
           std::abs(demonstrated.center_object.y - held.center_object.y) <=
               kBoundsToleranceM &&
           std::abs(demonstrated.center_object.z - held.center_object.z) <=
               kBoundsToleranceM &&
           std::abs(
               demonstrated.half_extents_object.x -
               held.half_extents_object.x) <= kBoundsToleranceM &&
           std::abs(
               demonstrated.half_extents_object.y -
               held.half_extents_object.y) <= kBoundsToleranceM &&
           std::abs(
               demonstrated.half_extents_object.z -
               held.half_extents_object.z) <= kBoundsToleranceM;
}

float bounds_cost(
    ObjectLocalBounds demonstrated,
    ObjectLocalBounds held) {
    return
        std::abs(demonstrated.center_object.x - held.center_object.x) +
        std::abs(demonstrated.center_object.y - held.center_object.y) +
        std::abs(demonstrated.center_object.z - held.center_object.z) +
        std::abs(
            demonstrated.half_extents_object.x -
            held.half_extents_object.x) +
        std::abs(
            demonstrated.half_extents_object.y -
            held.half_extents_object.y) +
        std::abs(
            demonstrated.half_extents_object.z -
            held.half_extents_object.z);
}

double squared_distance(vec3 left, vec3 right) {
    const double x = static_cast<double>(left.x) - right.x;
    const double y = static_cast<double>(left.y) - right.y;
    const double z = static_cast<double>(left.z) - right.z;
    return x * x + y * y + z * z;
}

std::optional<float> aligned_entry_continuity_cost(
    const PlaceMatchInput& input,
    const RecordedPlaceClip& clip,
    Transform scene,
    Transform staging_root) {
    const Transform current_root = pose_root(input.current_pose);
    const Transform current_to_staging = compose(
        staging_root, inverse(current_root));
    if (!valid_transform(current_to_staging)) return std::nullopt;

    const Pose aligned_current = rigidly_mapped_pose(
        input.current_pose, current_to_staging);
    const Pose mapped_entry = rigidly_mapped_pose(
        clip.poses[static_cast<size_t>(clip.entry_frame)], scene);
    const WorldPose current_world = world_pose(aligned_current);
    const WorldPose entry_world = world_pose(mapped_entry);
    const std::array<size_t, 6> continuity_bones = {
        static_cast<size_t>(g1_skeleton::LeftToe),
        static_cast<size_t>(g1_skeleton::RightToe),
        static_cast<size_t>(g1_skeleton::Hips),
        static_cast<size_t>(g1_skeleton::Spine2),
        hand_bone(clip.hand),
        kRootBone,
    };
    double cost = 0.0;
    for (size_t bone : continuity_bones) {
        cost += squared_distance(
            current_world.positions[bone], entry_world.positions[bone]);
        cost += squared_distance(
            current_world.velocities[bone], entry_world.velocities[bone]);
        const double orientation = rotation_error(
            current_world.rotations[bone], entry_world.rotations[bone]);
        cost += orientation * orientation;
        cost += squared_distance(
            current_world.angular_velocities[bone],
            entry_world.angular_velocities[bone]);
    }
    for (size_t dof = 0; dof < aligned_current.hand_dof.size(); ++dof) {
        const double position = static_cast<double>(
            aligned_current.hand_dof[dof]) - mapped_entry.hand_dof[dof];
        const double velocity = static_cast<double>(
            aligned_current.hand_dof_velocities[dof]) -
            mapped_entry.hand_dof_velocities[dof];
        cost += position * position + velocity * velocity;
    }
    for (size_t foot = 0; foot < aligned_current.foot_contacts.size(); ++foot) {
        if (aligned_current.foot_contacts[foot] !=
            mapped_entry.foot_contacts[foot]) {
            cost += 1.0;
        }
    }

    const Transform aligned_object = compose(
        current_to_staging, input.current_object_world);
    const Transform mapped_entry_object = compose(
        scene,
        clip.object_poses[static_cast<size_t>(clip.entry_frame)]);
    cost += squared_distance(
        aligned_object.position, mapped_entry_object.position);
    const double object_orientation = rotation_error(
        aligned_object.rotation, mapped_entry_object.rotation);
    cost += object_orientation * object_orientation;
    if (!rotation_gate::finite_bits(cost) ||
        cost > static_cast<double>(std::numeric_limits<float>::max())) {
        return std::nullopt;
    }
    return static_cast<float>(cost);
}

std::optional<PlaceCandidate> recorded_candidate(
    const PlaceMatchInput& input,
    const ValidatedInput& validated,
    const RecordedRow& row,
    SelectionFailures& failures) {
    const RecordedPlaceClip& clip = *row.clip;
    if (clip.object_profile_id != input.held_object_profile_id ||
        clip.hand != input.held_affordance.hand ||
        !bounds_compatible(clip.object_bounds, input.held_object_bounds) ||
        !within_transform_error(
            clip.hand_in_object,
            input.held_affordance.hand_in_object,
            kRecordedPositionToleranceM,
            kRecordedOrientationToleranceRadians)) {
        return std::nullopt;
    }
    if (!within_transform_error(
            clip.hand_in_object,
            input.held_affordance.hand_in_object,
            input.ik.maximum_request_position_m,
            input.ik.maximum_request_orientation_radians)) {
        failures.remember(Reason::CorrectionLimit);
        return std::nullopt;
    }

    const Transform source_release_object =
        clip.object_poses[static_cast<size_t>(clip.release_frame)];
    const Transform scene = planar_alignment(
        source_release_object, validated.goal_object);
    const RotationTransform mapped_release_hand = mapped_pose_hand_for_gate(
        clip.poses[static_cast<size_t>(clip.release_frame)],
        scene,
        clip.hand);
    const RotationTransform goal_hand = goal_hand_for_gate(input, validated);
    if (!within_transform_error(
            mapped_release_hand,
            goal_hand,
            input.ik.maximum_request_position_m,
            input.ik.maximum_request_orientation_radians)) {
        failures.remember(Reason::CorrectionLimit);
        return std::nullopt;
    }
    const std::optional<Transform> solved_release_object =
        solve_release_object(
            clip.poses[static_cast<size_t>(clip.release_frame)],
            scene,
            clip.hand,
            goal_hand,
            input,
            failures);
    if (!solved_release_object.has_value()) return std::nullopt;

    const Transform staging_root = compose(
        scene,
        pose_root(clip.poses[static_cast<size_t>(clip.entry_frame)]));
    if (!current_attachment_within_request(input, staging_root)) {
        failures.remember(Reason::CorrectionLimit);
        return std::nullopt;
    }
    const Transform current_to_staging = compose(
        staging_root, inverse(pose_root(input.current_pose)));
    std::vector<Transform> object_samples;
    object_samples.reserve(static_cast<size_t>(
        clip.release_frame - clip.entry_frame + 2));
    object_samples.push_back(compose(
        current_to_staging, input.current_object_world));
    for (int32_t frame = clip.entry_frame;
         frame < clip.release_frame;
         ++frame) {
        object_samples.push_back(compose(
            scene, clip.object_poses[static_cast<size_t>(frame)]));
    }
    object_samples.push_back(*solved_release_object);
    const Reason clearance_failure = mapped_clearance_failure(
        object_samples, input, *solved_release_object);
    if (clearance_failure != Reason::None) {
        failures.remember(clearance_failure);
        return std::nullopt;
    }
    const std::optional<float> continuity_cost =
        aligned_entry_continuity_cost(
            input, clip, scene, staging_root);
    if (!continuity_cost.has_value()) return std::nullopt;
    const Transform current_root = pose_root(input.current_pose);
    const vec3 root_offset(
        current_root.position.x - staging_root.position.x,
        0.0F,
        current_root.position.z - staging_root.position.z);
    const float yaw_offset = shortest_angle(
        yaw_radians(current_root.rotation) -
        yaw_radians(staging_root.rotation));

    PlaceCandidate candidate{};
    candidate.mode = PlaceMotionMode::RecordedPlace;
    candidate.source_id = clip.id;
    candidate.timing = input.timing;
    candidate.match = input.match;
    candidate.ik = input.ik;
    candidate.clip = static_cast<int32_t>(row.library_index);
    candidate.entry_frame = clip.entry_frame;
    candidate.commit_frame = clip.commit_frame;
    candidate.release_frame = clip.release_frame;
    candidate.stop_frame = clip.retract_stop_frame;
    candidate.direction = 1;
    candidate.scene_from_source = scene;
    candidate.staging_root_world = staging_root;
    candidate.entry_root_offset = root_offset;
    candidate.entry_yaw_offset = yaw_offset;
    if (!populate_candidate_heights(
            candidate,
            input,
            clip.source_surface.surface_world.position.y,
            failures)) {
        return std::nullopt;
    }
    const Transform held_grasp = input.held_affordance.hand_in_object;
    candidate.total_cost = static_cast<float>(
        distance(clip.hand_in_object.position, held_grasp.position) +
        rotation_error(clip.hand_in_object.rotation, held_grasp.rotation)) +
        bounds_cost(clip.object_bounds, input.held_object_bounds) +
        *continuity_cost +
        static_cast<float>(
            distance(
                mapped_release_hand.value.position,
                goal_hand.value.position) +
            rotation_error(
                mapped_release_hand.value.rotation,
                goal_hand.value.rotation));
    if (!finite(candidate.total_cost)) return std::nullopt;
    return candidate;
}

bool better_recorded(
    const PlaceCandidate& candidate,
    const PlaceCandidate& best) {
    return std::tie(
               candidate.total_cost,
               candidate.source_id,
               candidate.entry_frame,
               candidate.commit_frame,
               candidate.release_frame,
               candidate.stop_frame) <
           std::tie(
               best.total_cost,
               best.source_id,
               best.entry_frame,
               best.commit_frame,
               best.release_frame,
               best.stop_frame);
}

std::optional<PlaceCandidate> select_recorded_tier(
    const PlaceMatchInput& input,
    const ValidatedInput& validated,
    SelectionFailures& failures) {
    std::vector<RecordedRow> structurally_valid;
    structurally_valid.reserve(input.library->recorded.size());
    for (size_t index = 0; index < input.library->recorded.size(); ++index) {
        const std::optional<RecordedRow> row = validate_recorded_row(
            input.library->recorded[index], index, input.timing, failures);
        if (row.has_value()) structurally_valid.push_back(*row);
    }

    std::map<uint64_t, size_t> id_counts;
    for (const RecordedRow& row : structurally_valid) {
        ++id_counts[row.clip->id];
    }

    std::optional<PlaceCandidate> best;
    for (const RecordedRow& row : structurally_valid) {
        if (id_counts[row.clip->id] != 1U) continue;
        const std::optional<PlaceCandidate> candidate = recorded_candidate(
            input, validated, row, failures);
        if (!candidate.has_value()) continue;
        if (!best.has_value() || better_recorded(*candidate, *best)) {
            best = *candidate;
        }
    }
    return best;
}

bool span_available(size_t size, size_t begin, size_t count) {
    return begin <= size && count <= size - begin;
}

bool database_clip_metadata_valid(
    const PlaceMatchInput& input,
    int32_t clip_value,
    Hand hand,
    int32_t& range_start,
    int32_t& range_stop) {
    const Database& database = *input.pickup_database;
    if (database.fps_numerator != 25U || database.fps_denominator != 1U ||
        database.bone_count != g1_skeleton::BoneCount ||
        database.hand_dof_count != 14U ||
        clip_value < 0 ||
        static_cast<uint32_t>(clip_value) >= database.clip_count ||
        static_cast<size_t>(clip_value) >= database.range_starts.size() ||
        static_cast<size_t>(clip_value) >= database.range_stops.size() ||
        static_cast<size_t>(clip_value) >= database.active_hands.size() ||
        database.parents.size() != g1_skeleton::BoneCount ||
        !std::equal(
            database.parents.begin(),
            database.parents.end(),
            g1_skeleton::kParents.begin())) {
        return false;
    }
    const size_t clip = static_cast<size_t>(clip_value);
    range_start = database.range_starts[clip];
    range_stop = database.range_stops[clip];
    return range_start >= 0 && range_stop > range_start &&
           static_cast<uint32_t>(range_stop) <= database.frame_count &&
           database.active_hands[clip] == static_cast<uint8_t>(hand);
}

bool selected_database_metadata_valid(
    const PlaceMatchInput& input,
    int32_t& range_start,
    int32_t& range_stop) {
    const MatchCandidate& candidate = input.pickup_candidate;
    if (!valid_transform(candidate.scene_from_source) ||
        !finite(candidate.entry_root_offset) ||
        !finite(candidate.entry_yaw_offset) ||
        !finite(candidate.total_cost)) {
        return false;
    }
    for (float cost : candidate.group_costs) {
        if (!finite(cost)) return false;
    }
    return database_clip_metadata_valid(
        input,
        candidate.clip,
        input.held_affordance.hand,
        range_start,
        range_stop);
}

bool database_frame_storage_valid(
    const Database& database,
    int32_t frame) {
    if (frame < 0 || static_cast<uint32_t>(frame) >= database.frame_count) {
        return false;
    }
    const size_t index = static_cast<size_t>(frame);
    const size_t bones = g1_skeleton::BoneCount;
    return span_available(database.positions.size(), index * bones * 3U,
                          bones * 3U) &&
           span_available(database.velocities.size(), index * bones * 3U,
                          bones * 3U) &&
           span_available(database.rotations.size(), index * bones * 4U,
                          bones * 4U) &&
           span_available(
               database.angular_velocities.size(), index * bones * 3U,
               bones * 3U) &&
           span_available(database.foot_contacts.size(), index * 2U, 2U) &&
           span_available(database.hand_contacts.size(), index * 2U, 2U) &&
           span_available(database.hand_dof.size(), index * 14U, 14U) &&
           span_available(
               database.hand_dof_velocities.size(), index * 14U, 14U) &&
           span_available(database.phases.size(), index, 1U) &&
           span_available(database.time_to_contact.size(), index, 1U) &&
           span_available(database.object_positions.size(), index * 3U, 3U) &&
           span_available(database.object_rotations.size(), index * 4U, 4U) &&
           span_available(database.object_velocities.size(), index * 3U, 3U) &&
           span_available(
               database.object_angular_velocities.size(), index * 3U, 3U) &&
           span_available(database.source_frames.size(), index, 1U);
}

Transform database_object(const Database& database, int32_t frame) {
    const size_t index = static_cast<size_t>(frame);
    const size_t position = index * 3U;
    const size_t rotation = index * 4U;
    return {
        vec3(
            database.object_positions[position],
            database.object_positions[position + 1U],
            database.object_positions[position + 2U]),
        quat(
            database.object_rotations[rotation],
            database.object_rotations[rotation + 1U],
            database.object_rotations[rotation + 2U],
            database.object_rotations[rotation + 3U]),
    };
}

bool database_frame_valid(const Database& database, int32_t frame) {
    if (!database_frame_storage_valid(database, frame)) return false;
    const size_t index = static_cast<size_t>(frame);
    try {
        if (!pose_finite(pose_at_frame(database, frame)) ||
            !valid_transform(database_object(database, frame)) ||
            database.hand_contacts[index * 2U] > 1U ||
            database.hand_contacts[index * 2U + 1U] > 1U ||
            database.phases[index] > static_cast<uint8_t>(Phase::Hold) ||
            !finite(database.time_to_contact[index]) ||
            database.source_frames[index] < 0) {
            return false;
        }
        for (size_t component = 0; component < 3U; ++component) {
            if (!finite(database.object_velocities[index * 3U + component]) ||
                !finite(
                    database.object_angular_velocities[
                        index * 3U + component])) {
                return false;
            }
        }
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

RotationTransform database_hand_in_object(
    const Database& database,
    int32_t frame,
    Hand hand) {
    const Transform object = database_object(database, frame);
    Transform normalized_object = object;
    normalized_object.rotation = normalized(object.rotation);
    const Pose pose = pose_at_frame(database, frame);
    return {
        compose(inverse(normalized_object), pose_hand(pose, hand)),
        rotation_gate::multiply(
            rotation_gate::inverse(
                rotation_gate::from_quat(object.rotation)),
            pose_hand_rotation_evidence(pose, hand)),
    };
}

bool stable_hold_window(
    const Database& database,
    int32_t window_start,
    Hand hand) {
    const size_t active = static_cast<size_t>(hand);
    std::array<RotationTransform, kStableHoldSamples> relative{};
    for (size_t sample = 0; sample < kStableHoldSamples; ++sample) {
        const int32_t frame = window_start + static_cast<int32_t>(sample);
        if (!database_frame_valid(database, frame) ||
            database.phases[static_cast<size_t>(frame)] !=
                static_cast<uint8_t>(Phase::Hold) ||
            database.hand_contacts[static_cast<size_t>(frame) * 2U + active] !=
                1U) {
            return false;
        }
        relative[sample] = database_hand_in_object(database, frame, hand);
    }
    for (size_t left = 0; left < relative.size(); ++left) {
        for (size_t right = left + 1U; right < relative.size(); ++right) {
            if (!within_transform_error(
                    relative[left],
                    relative[right],
                    kRecordedPositionToleranceM,
                    kRecordedOrientationToleranceRadians)) {
                return false;
            }
        }
    }
    return true;
}

std::optional<int32_t> earliest_stable_reverse_start(
    const PlaceMatchInput& input,
    int32_t range_stop) {
    const int32_t first = input.pickup_candidate.hold_frame;
    const int64_t last_start =
        static_cast<int64_t>(range_stop) -
        static_cast<int64_t>(kStableHoldSamples);
    if (first < 0 || static_cast<int64_t>(first) > last_start) {
        return std::nullopt;
    }
    for (int32_t window_start = first;
         static_cast<int64_t>(window_start) <= last_start;
         ++window_start) {
        if (stable_hold_window(
                *input.pickup_database,
                window_start,
                input.held_affordance.hand)) {
            return window_start + static_cast<int32_t>(kStableHoldSamples - 1U);
        }
    }
    return std::nullopt;
}

struct ReverseSource {
    PlaceMotionMode mode = PlaceMotionMode::ReversedPickup;
    uint64_t source_id = 0U;
    int32_t clip = -1;
    int32_t entry_frame = -1;
    int32_t contact_frame = -1;
    int32_t lift_frame = -1;
    int32_t hold_frame = -1;
    int32_t reverse_start_frame = -1;
};

bool reverse_prefix_valid(
    const Database& database,
    int32_t range_start,
    int32_t range_stop,
    const ReverseSource& source,
    Hand hand) {
    if (!(range_start <= source.entry_frame &&
          source.entry_frame < source.contact_frame &&
          source.contact_frame < source.lift_frame &&
          source.lift_frame < source.hold_frame &&
          source.hold_frame <= source.reverse_start_frame &&
          source.reverse_start_frame < range_stop) ||
        !database_frame_storage_valid(database, source.contact_frame) ||
        !database_frame_storage_valid(database, source.lift_frame) ||
        !database_frame_storage_valid(database, source.hold_frame) ||
        database.phases[static_cast<size_t>(source.contact_frame)] !=
            static_cast<uint8_t>(Phase::Contact) ||
        database.phases[static_cast<size_t>(source.lift_frame)] !=
            static_cast<uint8_t>(Phase::Lift) ||
        database.phases[static_cast<size_t>(source.hold_frame)] !=
            static_cast<uint8_t>(Phase::Hold)) {
        return false;
    }
    const size_t active = static_cast<size_t>(hand);
    uint8_t previous_precontact_phase = static_cast<uint8_t>(Phase::Approach);
    for (int32_t frame = source.entry_frame;
         frame <= source.reverse_start_frame;
         ++frame) {
        if (!database_frame_valid(database, frame)) return false;
        const uint8_t phase = database.phases[static_cast<size_t>(frame)];
        if (frame < source.contact_frame) {
            if (phase > static_cast<uint8_t>(Phase::Reach) ||
                (frame > source.entry_frame &&
                 phase < previous_precontact_phase)) {
                return false;
            }
            previous_precontact_phase = phase;
        } else if (frame < source.lift_frame) {
            if (phase != static_cast<uint8_t>(Phase::Contact)) return false;
        } else if (frame < source.hold_frame) {
            if (phase != static_cast<uint8_t>(Phase::Lift)) return false;
        } else if (phase != static_cast<uint8_t>(Phase::Hold)) {
            return false;
        }
        if (frame >= source.contact_frame &&
            database.hand_contacts[static_cast<size_t>(frame) * 2U + active] !=
                1U) {
            return false;
        }
    }
    return true;
}

std::optional<PlaceCandidate> reverse_candidate(
    const PlaceMatchInput& input,
    const ValidatedInput& validated,
    const ReverseSource& source,
    SelectionFailures& failures) {
    const Transform source_contact_hand = pose_hand(
        pose_at_frame(*input.pickup_database, source.contact_frame),
        input.held_affordance.hand);
    const RotationTransform goal_hand = goal_hand_for_gate(input, validated);
    const Transform scene = planar_alignment(
        source_contact_hand, goal_hand.value);
    const RotationTransform mapped_contact_hand = mapped_pose_hand_for_gate(
        pose_at_frame(*input.pickup_database, source.contact_frame),
        scene,
        input.held_affordance.hand);
    if (!within_transform_error(
            mapped_contact_hand,
            goal_hand,
            input.ik.maximum_request_position_m,
            input.ik.maximum_request_orientation_radians)) {
        failures.remember(Reason::CorrectionLimit);
        return std::nullopt;
    }
    const std::optional<Transform> solved_release_object =
        solve_release_object(
            pose_at_frame(
                *input.pickup_database, source.contact_frame),
            scene,
            input.held_affordance.hand,
            goal_hand,
            input,
            failures);
    if (!solved_release_object.has_value()) return std::nullopt;

    const Transform staging_root = compose(
        scene,
        pose_root(pose_at_frame(
            *input.pickup_database, source.reverse_start_frame)));
    if (!current_attachment_within_request(input, staging_root)) {
        failures.remember(Reason::CorrectionLimit);
        return std::nullopt;
    }
    const Transform current_to_staging = compose(
        staging_root, inverse(pose_root(input.current_pose)));
    std::vector<Transform> object_samples;
    object_samples.reserve(static_cast<size_t>(
        source.reverse_start_frame - source.contact_frame + 2));
    object_samples.push_back(compose(
        current_to_staging, input.current_object_world));
    for (int32_t frame = source.reverse_start_frame;
         frame > source.contact_frame;
         --frame) {
        const Transform mapped_hand = compose(
            scene,
            pose_hand(
                pose_at_frame(*input.pickup_database, frame),
                input.held_affordance.hand));
        object_samples.push_back(compose(
            mapped_hand,
            inverse(input.held_affordance.hand_in_object)));
    }
    object_samples.push_back(*solved_release_object);
    const Reason clearance_failure = mapped_clearance_failure(
        object_samples, input, *solved_release_object);
    if (clearance_failure != Reason::None) {
        failures.remember(clearance_failure);
        return std::nullopt;
    }

    const double offset_value =
        static_cast<double>(input.timing.reversed_commit_seconds) *
        kCanonicalFps * input.timing.playback_speed;
    if (!rotation_gate::finite_bits(offset_value) ||
        offset_value > std::numeric_limits<int32_t>::max()) {
        return std::nullopt;
    }
    const int32_t offset = static_cast<int32_t>(std::floor(offset_value));
    if (offset <= 0) return std::nullopt;
    const int32_t commit = source.reverse_start_frame - offset;
    if (!(source.contact_frame < commit &&
          commit < source.reverse_start_frame)) {
        return std::nullopt;
    }
    if (!observable_commit_timing_valid(offset, input.timing)) {
        return std::nullopt;
    }

    const Transform current_root = pose_root(input.current_pose);
    PlaceCandidate candidate{};
    candidate.mode = source.mode;
    candidate.source_id = source.source_id;
    candidate.timing = input.timing;
    candidate.match = input.match;
    candidate.ik = input.ik;
    candidate.clip = source.clip;
    candidate.entry_frame = source.reverse_start_frame;
    candidate.commit_frame = commit;
    candidate.release_frame = source.contact_frame;
    candidate.stop_frame = source.entry_frame;
    candidate.direction = -1;
    candidate.scene_from_source = scene;
    candidate.staging_root_world = staging_root;
    candidate.entry_root_offset = vec3(
        current_root.position.x - staging_root.position.x,
        0.0F,
        current_root.position.z - staging_root.position.z);
    candidate.entry_yaw_offset = shortest_angle(
        yaw_radians(current_root.rotation) -
        yaw_radians(staging_root.rotation));
    candidate.total_cost = static_cast<float>(
        distance(
            mapped_contact_hand.value.position,
            goal_hand.value.position) +
        rotation_error(
            mapped_contact_hand.value.rotation,
            goal_hand.value.rotation));
    if (!finite(candidate.total_cost) || candidate.source_id == 0U) {
        return std::nullopt;
    }
    return candidate;
}

std::optional<Transform> database_authored_grasp(
    const Database& database,
    int32_t clip) {
    if (clip < 0) return std::nullopt;
    const size_t index = static_cast<size_t>(clip);
    if (index > std::numeric_limits<size_t>::max() / 4U ||
        !span_available(
            database.grasp_positions_object.size(), index * 3U, 3U) ||
        !span_available(
            database.grasp_rotations_object.size(), index * 4U, 4U)) {
        return std::nullopt;
    }
    const size_t position = index * 3U;
    const size_t rotation = index * 4U;
    Transform grasp{
        vec3(
            database.grasp_positions_object[position],
            database.grasp_positions_object[position + 1U],
            database.grasp_positions_object[position + 2U]),
        quat(
            database.grasp_rotations_object[rotation],
            database.grasp_rotations_object[rotation + 1U],
            database.grasp_rotations_object[rotation + 2U],
            database.grasp_rotations_object[rotation + 3U]),
    };
    if (!valid_transform(grasp)) return std::nullopt;
    return grasp;
}

std::optional<ObjectLocalBounds> database_authored_bounds(
    const Database& database,
    int32_t clip) {
    if (clip < 0) return std::nullopt;
    const size_t index = static_cast<size_t>(clip);
    if (index > std::numeric_limits<size_t>::max() / 3U ||
        !span_available(database.object_dimensions.size(), index * 3U, 3U)) {
        return std::nullopt;
    }
    const size_t dimension = index * 3U;
    const vec3 dimensions(
        database.object_dimensions[dimension],
        database.object_dimensions[dimension + 1U],
        database.object_dimensions[dimension + 2U]);
    if (!positive(dimensions)) return std::nullopt;
    return ObjectLocalBounds{vec3(), 0.5F * dimensions};
}

struct PrecomputedRow {
    const PrecomputedReversedPickupClip* clip = nullptr;
    ReverseSource source{};
};

std::optional<PrecomputedRow> validate_precomputed_row(
    const PlaceMatchInput& input,
    const PrecomputedReversedPickupClip& clip,
    SelectionFailures& failures) {
    if (clip.id == 0U || clip.object_profile_id == 0U ||
        clip.sequence_id.empty() ||
        (clip.hand != Hand::Left && clip.hand != Hand::Right) ||
        clip.hand != input.held_affordance.hand ||
        clip.object_profile_id != input.held_object_profile_id ||
        !valid_transform(clip.source_hand_in_object) ||
        !valid_bounds(clip.source_object_bounds) ||
        clip.source_affordance_id == 0U ||
        !finite(clip.source_support_height_m) ||
        !finite(clip.source_grasp_height_above_support_m)) {
        return std::nullopt;
    }
    int32_t range_start = -1;
    int32_t range_stop = -1;
    if (!database_clip_metadata_valid(
            input, clip.clip, clip.hand, range_start, range_stop)) {
        return std::nullopt;
    }
    const ReverseSource source{
        PlaceMotionMode::PrecomputedReversedPickup,
        clip.id,
        clip.clip,
        clip.entry_frame,
        clip.contact_frame,
        clip.lift_frame,
        clip.hold_frame,
        clip.reverse_start_frame,
    };
    const int64_t stable_start =
        static_cast<int64_t>(clip.reverse_start_frame) -
        static_cast<int64_t>(kStableHoldSamples - 1U);
    if (!reverse_prefix_valid(
            *input.pickup_database,
            range_start,
            range_stop,
            source,
            clip.hand) ||
        stable_start < clip.hold_frame || stable_start < 0 ||
        !stable_hold_window(
            *input.pickup_database,
            static_cast<int32_t>(stable_start),
            clip.hand)) {
        return std::nullopt;
    }

    const std::optional<Transform> database_grasp = database_authored_grasp(
        *input.pickup_database, clip.clip);
    const std::optional<ObjectLocalBounds> database_bounds =
        database_authored_bounds(*input.pickup_database, clip.clip);
    const std::optional<float> database_support =
        database_source_support_height(*input.pickup_database, clip.clip);
    if (!database_grasp.has_value() || !database_bounds.has_value() ||
        !database_support.has_value() ||
        !within_transform_error(
            clip.source_hand_in_object,
            *database_grasp,
            kRecordedPositionToleranceM,
            kRecordedOrientationToleranceRadians) ||
        !within_transform_error(
            clip.source_hand_in_object,
            input.held_affordance.hand_in_object,
            kRecordedPositionToleranceM,
            kRecordedOrientationToleranceRadians) ||
        !bounds_compatible(clip.source_object_bounds, *database_bounds) ||
        !bounds_compatible(
            clip.source_object_bounds, input.held_object_bounds) ||
        std::abs(*database_support - clip.source_support_height_m) >
            kUnitTolerance ||
        std::abs(
            clip.source_surface.surface_world.position.y -
            clip.source_support_height_m) > kUnitTolerance) {
        return std::nullopt;
    }

    const Pose contact_pose = pose_at_frame(
        *input.pickup_database, clip.contact_frame);
    const Transform contact_hand = pose_hand(contact_pose, clip.hand);
    const Transform contact_object = database_object(
        *input.pickup_database, clip.contact_frame);
    const float grasp_height =
        contact_hand.position.y - clip.source_support_height_m;
    if (!finite(grasp_height) ||
        std::abs(
            grasp_height - clip.source_grasp_height_above_support_m) >
            kRecordedPositionToleranceM ||
        !within_transform_error(
            contact_hand,
            compose(contact_object, clip.source_hand_in_object),
            kRecordedPositionToleranceM,
            kRecordedOrientationToleranceRadians)) {
        return std::nullopt;
    }

    const PlaceAffordance* source_affordance = find_affordance(
        clip.source_surface, clip.source_affordance_id);
    if (source_affordance == nullptr) return std::nullopt;
    try {
        const PlacementFit fit = evaluate_actual_placement_fit(
            clip.source_surface,
            *source_affordance,
            contact_object,
            clip.source_object_bounds);
        if (!fit.accepted) {
            failures.remember(fit.reason);
            return std::nullopt;
        }
    } catch (const std::exception&) {
        return std::nullopt;
    }
    return PrecomputedRow{&clip, source};
}

bool better_reverse(
    const PlaceCandidate& candidate,
    const PlaceCandidate& best) {
    return std::tie(
               candidate.total_cost,
               candidate.source_id,
               candidate.clip,
               candidate.entry_frame,
               candidate.commit_frame,
               candidate.release_frame,
               candidate.stop_frame) <
           std::tie(
               best.total_cost,
               best.source_id,
               best.clip,
               best.entry_frame,
               best.commit_frame,
               best.release_frame,
               best.stop_frame);
}

std::optional<PlaceCandidate> select_precomputed_reverse_tier(
    const PlaceMatchInput& input,
    const ValidatedInput& validated,
    SelectionFailures& failures) {
    std::vector<PrecomputedRow> structurally_valid;
    structurally_valid.reserve(input.library->precomputed_reversed.size());
    for (const PrecomputedReversedPickupClip& clip :
         input.library->precomputed_reversed) {
        const std::optional<PrecomputedRow> row = validate_precomputed_row(
            input, clip, failures);
        if (row.has_value()) structurally_valid.push_back(*row);
    }
    std::map<uint64_t, size_t> id_counts;
    for (const PrecomputedRow& row : structurally_valid) {
        ++id_counts[row.clip->id];
    }

    std::optional<PlaceCandidate> best;
    for (const PrecomputedRow& row : structurally_valid) {
        if (id_counts[row.clip->id] != 1U) continue;
        std::optional<PlaceCandidate> candidate = reverse_candidate(
            input, validated, row.source, failures);
        if (!candidate.has_value() ||
            !populate_candidate_heights(
                *candidate,
                input,
                row.clip->source_support_height_m,
                failures)) {
            continue;
        }
        if (!best.has_value() || better_reverse(*candidate, *best)) {
            best = *candidate;
        }
    }
    return best;
}

std::optional<PlaceCandidate> select_reverse_tier(
    const PlaceMatchInput& input,
    const ValidatedInput& validated,
    int32_t& certified_prefix_stop,
    SelectionFailures& failures) {
    int32_t range_start = -1;
    int32_t range_stop = -1;
    if (!selected_database_metadata_valid(input, range_start, range_stop)) {
        return std::nullopt;
    }
    const MatchCandidate& pickup = input.pickup_candidate;
    const std::optional<int32_t> reverse_start =
        earliest_stable_reverse_start(input, range_stop);
    if (!reverse_start.has_value()) return std::nullopt;
    ReverseSource source{};
    source.mode = PlaceMotionMode::ReversedPickup;
    source.source_id = reverse_source_id(input, *reverse_start);
    source.clip = pickup.clip;
    source.entry_frame = pickup.entry_frame;
    source.contact_frame = pickup.contact_frame;
    source.lift_frame = pickup.lift_frame;
    source.hold_frame = pickup.hold_frame;
    source.reverse_start_frame = *reverse_start;
    if (!reverse_prefix_valid(
            *input.pickup_database,
            range_start,
            range_stop,
            source,
            input.held_affordance.hand)) {
        return std::nullopt;
    }
    certified_prefix_stop = *reverse_start;
    std::optional<PlaceCandidate> candidate = reverse_candidate(
        input, validated, source, failures);
    const std::optional<float> source_support =
        database_source_support_height(*input.pickup_database, pickup.clip);
    if (!candidate.has_value() || !source_support.has_value() ||
        !populate_candidate_heights(
            *candidate, input, *source_support, failures)) {
        return std::nullopt;
    }
    return candidate;
}

bool same_timing(
    const PlaceTimingConfig& left,
    const PlaceTimingConfig& right) {
    return same_float(left.canonical_fps, right.canonical_fps) &&
           same_float(left.playback_speed, right.playback_speed) &&
           same_float(left.entry_blend_seconds, right.entry_blend_seconds) &&
           same_float(
               left.reversed_commit_seconds,
               right.reversed_commit_seconds) &&
           same_float(
               left.maximum_alignment_seconds,
               right.maximum_alignment_seconds);
}

bool same_match(
    const PlaceMatchConfig& left,
    const PlaceMatchConfig& right) {
    return same_float(
               left.maximum_entry_root_error_m,
               right.maximum_entry_root_error_m) &&
           same_float(
               left.maximum_entry_yaw_error_radians,
               right.maximum_entry_yaw_error_radians);
}

bool same_ik(const IKConfig& left, const IKConfig& right) {
    return same_float(
               left.maximum_request_position_m,
               right.maximum_request_position_m) &&
           same_float(
               left.maximum_request_orientation_radians,
               right.maximum_request_orientation_radians) &&
           same_float(left.accepted_position_m, right.accepted_position_m) &&
           same_float(
               left.accepted_orientation_radians,
               right.accepted_orientation_radians) &&
           same_float(left.damping, right.damping) &&
           same_float(
               left.finite_difference_radians,
               right.finite_difference_radians) &&
           same_float(
               left.orientation_scale_m_per_radian,
               right.orientation_scale_m_per_radian) &&
           same_float(
               left.maximum_step_radians,
               right.maximum_step_radians) &&
           left.maximum_iterations == right.maximum_iterations;
}

bool same_candidate(
    const PlaceCandidate& left,
    const PlaceCandidate& right) {
    return left.mode == right.mode &&
           left.source_id == right.source_id &&
           left.selection_id == right.selection_id &&
           same_float(
               left.source_support_height_m,
               right.source_support_height_m) &&
           same_float(
               left.requested_support_height_m,
               right.requested_support_height_m) &&
           same_float(
               left.target_support_height_m,
               right.target_support_height_m) &&
           same_float(
               left.requested_vertical_correction_m,
               right.requested_vertical_correction_m) &&
           same_timing(left.timing, right.timing) &&
           same_match(left.match, right.match) &&
           same_ik(left.ik, right.ik) &&
           left.clip == right.clip &&
           left.entry_frame == right.entry_frame &&
           left.commit_frame == right.commit_frame &&
           left.release_frame == right.release_frame &&
           left.stop_frame == right.stop_frame &&
           left.direction == right.direction &&
           same_transform(left.scene_from_source, right.scene_from_source) &&
           same_transform(left.staging_root_world, right.staging_root_world) &&
           same_vec(left.entry_root_offset, right.entry_root_offset) &&
           same_float(left.entry_yaw_offset, right.entry_yaw_offset) &&
           same_float(left.total_cost, right.total_cost);
}

Pose interpolate_source_pose(
    const PlaceCandidate& candidate,
    const PlaceMatchInput& input,
    double source_frame) {
    const int32_t left = static_cast<int32_t>(std::floor(source_frame));
    const int32_t maximum = candidate.direction > 0
        ? candidate.stop_frame
        : candidate.entry_frame;
    const int32_t right = std::min(left + 1, maximum);
    const float alpha = static_cast<float>(source_frame - left);
    if (candidate.mode == PlaceMotionMode::RecordedPlace) {
        const RecordedPlaceClip& clip =
            input.library->recorded.at(static_cast<size_t>(candidate.clip));
        if (alpha == 0.0F) {
            return clip.poses.at(static_cast<size_t>(left));
        }
        return interpolate_pose(
            clip.poses.at(static_cast<size_t>(left)),
            clip.poses.at(static_cast<size_t>(right)),
            alpha);
    }
    const bool database_backed =
        candidate.mode == PlaceMotionMode::ReversedPickup ||
        candidate.mode == PlaceMotionMode::PrecomputedReversedPickup;
    if (!database_backed) {
        throw std::invalid_argument(
            "place candidate has no motion source");
    }
    if (alpha == 0.0F) {
        return pose_at_frame(*input.pickup_database, left);
    }
    return interpolate_pose(
        pose_at_frame(*input.pickup_database, left),
        pose_at_frame(*input.pickup_database, right),
        alpha);
}

Transform interpolate_transform(
    Transform left,
    Transform right,
    float alpha) {
    return {
        lerp(left.position, right.position, alpha),
        quat_nlerp_shortest(left.rotation, right.rotation, alpha),
    };
}

Transform recorded_source_object(
    const PlaceCandidate& candidate,
    const PlaceMatchInput& input,
    double source_frame) {
    const int32_t left = static_cast<int32_t>(std::floor(source_frame));
    const int32_t right = std::min(left + 1, candidate.stop_frame);
    const float alpha = static_cast<float>(source_frame - left);
    const RecordedPlaceClip& clip =
        input.library->recorded.at(static_cast<size_t>(candidate.clip));
    if (alpha == 0.0F) {
        return compose(
            candidate.scene_from_source,
            clip.object_poses.at(static_cast<size_t>(left)));
    }
    return compose(
        candidate.scene_from_source,
        interpolate_transform(
            clip.object_poses.at(static_cast<size_t>(left)),
            clip.object_poses.at(static_cast<size_t>(right)),
            alpha));
}

Pose mapped_source_pose(
    const PlaceCandidate& candidate,
    const PlaceMatchInput& input,
    double source_frame) {
    Pose pose = interpolate_source_pose(candidate, input, source_frame);
    const Transform mapped_root = compose(
        candidate.scene_from_source,
        Transform{pose.positions[kRootBone], pose.rotations[kRootBone]});
    pose.positions[kRootBone] = mapped_root.position;
    pose.rotations[kRootBone] = mapped_root.rotation;

    const float derivative_scale =
        static_cast<float>(candidate.direction) *
        candidate.timing.playback_speed;
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        if (bone == kRootBone) {
            pose.velocities[bone] = derivative_scale * quat_mul_vec3(
                candidate.scene_from_source.rotation,
                pose.velocities[bone]);
            pose.angular_velocities[bone] = derivative_scale * quat_mul_vec3(
                candidate.scene_from_source.rotation,
                pose.angular_velocities[bone]);
        } else {
            pose.velocities[bone] = derivative_scale * pose.velocities[bone];
            pose.angular_velocities[bone] =
                derivative_scale * pose.angular_velocities[bone];
        }
    }
    for (float& velocity : pose.hand_dof_velocities) {
        velocity *= derivative_scale;
    }
    return pose;
}

int32_t discrete_source_frame(
    double source_frame,
    int32_t direction,
    std::optional<int32_t> hidden_event_frame = std::nullopt) {
    const double rounded = std::round(source_frame);
    if (std::abs(source_frame - rounded) <= kSourceFrameSnapTolerance) {
        const int32_t rounded_frame = static_cast<int32_t>(rounded);
        if (hidden_event_frame.has_value() &&
            rounded_frame == *hidden_event_frame &&
            source_frame != rounded) {
            return rounded_frame - direction;
        }
        return rounded_frame;
    }
    return static_cast<int32_t>(std::floor(source_frame));
}

std::optional<int32_t> hidden_pending_event_frame(
    const PlaceCandidate& candidate,
    bool committed,
    bool release_latched,
    bool release_acknowledged,
    bool finished) {
    if (finished || release_latched) return std::nullopt;
    if (!committed) return candidate.commit_frame;
    return release_acknowledged
        ? std::optional<int32_t>(candidate.stop_frame)
        : std::optional<int32_t>(candidate.release_frame);
}

PlaceResult accept_with_fingerprint(
    const PlaceMatchInput& input,
    PlaceCandidate candidate,
    int32_t pickup_prefix_stop) {
    candidate.selection_id = selection_fingerprint(
        input, candidate, pickup_prefix_stop);
    if (candidate.selection_id == 0U) return reject(Reason::NoCandidate);
    return {true, candidate, Reason::None};
}

PlaceResult accept_with_fingerprint(
    const PlaceMatchInput& input,
    PlaceCandidate candidate) {
    return accept_with_fingerprint(
        input, candidate, default_pickup_prefix_stop(input));
}

}  // namespace

PlaceResult select_place_motion(const PlaceMatchInput& input) {
    ValidatedInput validated{};
    if (const std::optional<Reason> invalid =
            validate_input(input, validated)) {
        return reject(*invalid);
    }

    SelectionFailures failures{};

    if (auto recorded = select_recorded_tier(
            input, validated, failures)) {
        return accept_with_fingerprint(input, *recorded);
    }
    if (auto precomputed = select_precomputed_reverse_tier(
            input, validated, failures)) {
        return accept_with_fingerprint(input, *precomputed);
    }
    int32_t certified_prefix_stop = -1;
    if (auto immediate = select_reverse_tier(
            input,
            validated,
            certified_prefix_stop,
            failures)) {
        return accept_with_fingerprint(
            input, *immediate, certified_prefix_stop);
    }
    return reject(failures.strongest());
}

PlaceStagingPreview preview_place_motion(const PlaceMatchInput& input) {
    const PlaceResult result = select_place_motion(input);
    PlaceStagingPreview preview{};
    preview.accepted = result.accepted;
    preview.reason = result.reason;
    if (!result.accepted) return preview;
    preview.candidate = result.candidate;
    preview.ik = input.ik;
    preview.ik_config_fingerprint = ik_fingerprint(input.ik);
    preview.staging_root_world = result.candidate.staging_root_world;
    const Transform current_root = pose_root(input.current_pose);
    preview.root_error_m = planar_distance(
        current_root.position, result.candidate.staging_root_world.position);
    preview.yaw_error_radians = std::abs(shortest_angle(
        yaw_radians(current_root.rotation) -
        yaw_radians(result.candidate.staging_root_world.rotation)));
    preview.ready =
        preview.root_error_m <= input.match.maximum_entry_root_error_m &&
        preview.yaw_error_radians <=
            input.match.maximum_entry_yaw_error_radians;
    return preview;
}

void PlacePlayer::start(
    const PlaceCandidate& candidate,
    const PlaceMatchInput& input) {
    const PlaceResult selected = select_place_motion(input);
    if (!selected.accepted || !same_candidate(candidate, selected.candidate) ||
        !same_timing(candidate.timing, input.timing) ||
        !same_match(candidate.match, input.match) ||
        !same_ik(candidate.ik, input.ik)) {
        throw std::invalid_argument(
            "place player candidate does not match frozen selection");
    }

    start_validated(candidate, input);
}

void PlacePlayer::start_validated(
    const PlaceCandidate& candidate,
    const PlaceMatchInput& input) {
    const auto event_distance = [](int32_t start, int32_t stop) {
        const int64_t difference =
            static_cast<int64_t>(stop) - start;
        return difference < 0 ? -difference : difference;
    };
    const std::optional<int32_t> commit_ticks = observable_event_ticks(
        event_distance(candidate.entry_frame, candidate.commit_frame),
        candidate.timing.playback_speed);
    const std::optional<int32_t> release_ticks = observable_event_ticks(
        event_distance(candidate.commit_frame, candidate.release_frame),
        candidate.timing.playback_speed);
    const std::optional<int32_t> stop_ticks = observable_event_ticks(
        event_distance(candidate.release_frame, candidate.stop_frame),
        candidate.timing.playback_speed);
    if (!commit_ticks.has_value() || !release_ticks.has_value() ||
        !stop_ticks.has_value()) {
        throw std::invalid_argument(
            "place player candidate event timing is invalid");
    }

    candidate_ = candidate;
    input_ = input;
    source_frame_ = candidate.entry_frame;
    segment_tick_ = 0;
    commit_tick_count_ = *commit_ticks;
    release_tick_count_ = *release_ticks;
    stop_tick_count_ = *stop_ticks;
    started_ = true;
    committed_ = false;
    release_latched_ = false;
    release_acknowledged_ = false;
    finished_ = false;
}

void PlacePlayer::advance(float dt) {
    if (!started_) throw std::logic_error("place player is not started");
    if (!finite(dt) || dt != 0.04F) {
        throw std::invalid_argument("place player requires one 0.04 second tick");
    }
    if (finished_ || release_latched_) return;

    ++segment_tick_;
    const int32_t segment_start = !committed_
        ? candidate_.entry_frame
        : release_acknowledged_
            ? candidate_.release_frame
            : candidate_.commit_frame;
    const int32_t event_frame = !committed_
        ? candidate_.commit_frame
        : release_acknowledged_
            ? candidate_.stop_frame
            : candidate_.release_frame;
    const int32_t event_tick_count = !committed_
        ? commit_tick_count_
        : release_acknowledged_
            ? stop_tick_count_
            : release_tick_count_;

    if (segment_tick_ >= event_tick_count) {
        source_frame_ = event_frame;
        segment_tick_ = 0;
        if (!committed_) {
            committed_ = true;
        } else if (!release_acknowledged_) {
            release_latched_ = true;
        } else {
            finished_ = true;
        }
        return;
    }

    double proposed = static_cast<double>(segment_start) +
        static_cast<double>(candidate_.direction) *
        static_cast<double>(segment_tick_) *
        static_cast<double>(candidate_.timing.playback_speed);
    const double nearest_source_frame = std::round(proposed);
    if (std::abs(proposed - nearest_source_frame) <=
            kSourceFrameSnapTolerance &&
        nearest_source_frame != static_cast<double>(event_frame)) {
        proposed = nearest_source_frame;
    }

    source_frame_ = proposed;
}

PlaceSample PlacePlayer::sample() const {
    if (!started_) throw std::logic_error("place player is not started");
    PlaceSample result{};
    result.pose = mapped_source_pose(candidate_, input_, source_frame_);
    result.source_frame = discrete_source_frame(
        source_frame_,
        candidate_.direction,
        hidden_pending_event_frame(
            candidate_,
            committed_,
            release_latched_,
            release_acknowledged_,
            finished_));
    result.phase = phase();
    result.committed = committed_;
    if (candidate_.mode == PlaceMotionMode::RecordedPlace) {
        result.source_object = recorded_source_object(
            candidate_, input_, source_frame_);
    } else {
        result.source_object = compose(
            pose_hand(result.pose, input_.held_affordance.hand),
            inverse(input_.held_affordance.hand_in_object));
    }
    return result;
}

rotation_gate::Rotation PlacePlayer::mapped_root_rotation_evidence() const {
    if (!started_) throw std::logic_error("place player is not started");
    const Pose source = interpolate_source_pose(
        candidate_, input_, source_frame_);
    return rotation_gate::multiply(
        rotation_gate::from_quat(
            candidate_.scene_from_source.rotation),
        rotation_gate::from_quat(source.rotations[kRootBone]));
}

int32_t PlacePlayer::source_frame() const {
    return started_
        ? discrete_source_frame(
            source_frame_,
            candidate_.direction,
            hidden_pending_event_frame(
                candidate_,
                committed_,
                release_latched_,
                release_acknowledged_,
                finished_))
        : -1;
}

double PlacePlayer::source_frame_exact() const {
    return started_ ? source_frame_ : -1.0;
}

PlacePhase PlacePlayer::phase() const {
    if (!started_) throw std::logic_error("place player is not started");
    if (finished_) return PlacePhase::Finished;
    if (release_latched_) return PlacePhase::Release;
    if (release_acknowledged_) return PlacePhase::Retract;
    if (committed_) return PlacePhase::Lower;
    return PlacePhase::Align;
}

bool PlacePlayer::committed() const {
    return started_ && committed_;
}

bool PlacePlayer::release_due() const {
    return started_ && release_latched_;
}

void PlacePlayer::acknowledge_release() {
    if (!started_) throw std::logic_error("place player is not started");
    if (!release_latched_ || release_acknowledged_) {
        throw std::logic_error("place release is not pending");
    }
    release_latched_ = false;
    release_acknowledged_ = true;
    segment_tick_ = 0;
}

bool PlacePlayer::finished() const {
    return !started_ || finished_;
}

}  // namespace interaction
