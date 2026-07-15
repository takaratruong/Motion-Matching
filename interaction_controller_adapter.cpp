#include "interaction_controller_adapter.h"
#include "spring.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace interaction {
namespace {

constexpr int kControllerRate = locomotion_timing::kRateHz;
constexpr int kInteractionRuntimeRate = locomotion_timing::kRateHz;
static_assert(kControllerRate == kInteractionRuntimeRate);
constexpr float kOwnershipBlendSeconds = 0.25F;
constexpr float kInactiveArmBlendSeconds = 0.50F;
constexpr float kInactiveArmTranslationStepLimitM = 0.20F;
constexpr float kInactiveArmRotationStepLimitRadians = 1.047197551F;
constexpr int kInactiveArmBackoffIterations = 16;
constexpr size_t kLayeredCarryLowerBodyBoneCount = 10U;
constexpr float kLayeredCarryLowerBodyHalfLifeSeconds = 0.10F;
constexpr float kLayeredCarryLowerBodyNominalSeconds = 0.50F;
constexpr float kLayeredCarryTerminalPositionM = 0.001F;
constexpr float kLayeredCarryTerminalVelocityMps = 0.01F;
constexpr float kLayeredCarryTerminalRotationRadians =
    0.5F * 3.14159265358979323846F / 180.0F;
constexpr float kLayeredCarryTerminalAngularVelocityRadiansPerSecond = 0.05F;
constexpr float kLayeredCarryTerminalStepTranslationM = 0.05F;
constexpr float kLayeredCarryTerminalStepRotationRadians =
    15.0F * 3.14159265358979323846F / 180.0F;
constexpr int kLayeredCarryLowerBodyBackoffIterations = 16;
constexpr float kUnitRotationTolerance = 1.0e-3F;

constexpr std::array<int32_t, g1_skeleton::BoneCount> kG1ToFlatBone = {
    0, 1, 2, -1, -1, 3, 4, 5, 6, -1, -1, 7, 8, 9, 10, 11,
    12, 15, -1, 16, 17, -1, -1, 18, 19, -1, 20, 21, -1, -1, 22};
constexpr std::array<int32_t, kFlatControllerBoneCount> kFlatToG1Bone = {
    0, 1, 2, 5, 6, 7, 8, 11, 12, 13, 14, 15,
    16, -1, -1, 17, 19, 20, 23, 24, 26, 27, 30};

struct FlatWorldPose {
    std::array<vec3, kFlatControllerBoneCount> positions{};
    std::array<vec3, kFlatControllerBoneCount> velocities{};
    std::array<quat, kFlatControllerBoneCount> rotations{};
    std::array<vec3, kFlatControllerBoneCount> angular_velocities{};
};

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

void validate_rotation(quat rotation, const std::string& label) {
    if (!finite(rotation)) {
        throw FormatError(label + " rotation is non-finite");
    }
    const float magnitude = quat_length(rotation);
    if (!finite(magnitude) ||
        std::fabs(magnitude - 1.0F) > kUnitRotationTolerance) {
        throw FormatError(label + " rotation is not unit length");
    }
}

void validate_flat_pose(const FlatControllerPose& pose) {
    for (size_t bone = 0; bone < kFlatControllerBoneCount; ++bone) {
        const std::string label =
            "flat controller bone " + std::to_string(bone);
        if (!finite(pose.positions[bone]) ||
            !finite(pose.velocities[bone]) ||
            !finite(pose.angular_velocities[bone])) {
            throw FormatError(label + " has non-finite vector data");
        }
        validate_rotation(pose.rotations[bone], label);
    }
    for (uint8_t contact : pose.foot_contacts) {
        if (contact > 1U) {
            throw FormatError("flat controller foot contact is not boolean");
        }
    }
}

void validate_interaction_pose(const Pose& pose) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        const std::string label =
            "G1 interaction bone " + std::to_string(bone);
        if (!finite(pose.positions[bone]) ||
            !finite(pose.velocities[bone]) ||
            !finite(pose.angular_velocities[bone])) {
            throw FormatError(label + " has non-finite vector data");
        }
        validate_rotation(pose.rotations[bone], label);
    }
    for (float value : pose.hand_dof) {
        if (!finite(value)) {
            throw FormatError("G1 interaction hand DOF is non-finite");
        }
    }
    for (float value : pose.hand_dof_velocities) {
        if (!finite(value)) {
            throw FormatError("G1 interaction hand DOF velocity is non-finite");
        }
    }
    for (uint8_t contact : pose.foot_contacts) {
        if (contact > 1U) {
            throw FormatError("G1 interaction foot contact is not boolean");
        }
    }
}

bool raw_channels_equal(vec3 left, vec3 right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool raw_channels_equal(quat left, quat right) {
    return left.w == right.w && left.x == right.x && left.y == right.y &&
           left.z == right.z;
}

bool raw_pose_channels_equal(const Pose& left, const Pose& right) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        if (!raw_channels_equal(left.positions[bone], right.positions[bone]) ||
            !raw_channels_equal(left.velocities[bone], right.velocities[bone]) ||
            !raw_channels_equal(left.rotations[bone], right.rotations[bone]) ||
            !raw_channels_equal(
                left.angular_velocities[bone],
                right.angular_velocities[bone])) {
            return false;
        }
    }
    for (size_t joint = 0; joint < left.hand_dof.size(); ++joint) {
        if (left.hand_dof[joint] != right.hand_dof[joint] ||
            left.hand_dof_velocities[joint] !=
                right.hand_dof_velocities[joint]) {
            return false;
        }
    }
    return left.foot_contacts == right.foot_contacts;
}

bool valid_constraint_hand(Hand hand) {
    return hand == Hand::Left || hand == Hand::Right;
}

bool valid_constraint_transform(Transform transform) {
    if (!finite(transform.position) || !finite(transform.rotation)) {
        return false;
    }
    const float magnitude = quat_length(transform.rotation);
    return finite(magnitude) &&
        std::fabs(magnitude - 1.0F) <= kUnitRotationTolerance;
}

bool constraint_identity_matches(
    const ControllerInteractionHandConstraint& constraint,
    const RuntimeDiagnostics& diagnostics) {
    return constraint.target.id != 0U &&
        constraint.target.generation != 0U &&
        constraint.affordance_id != 0U &&
        valid_constraint_hand(constraint.hand) &&
        valid_constraint_transform(constraint.grasp_world) &&
        constraint.target == diagnostics.target &&
        constraint.affordance_id == diagnostics.affordance_id &&
        constraint.hand == diagnostics.hand;
}

bool constraint_identity_matches(
    const ControllerInteractionHandConstraint& left,
    const ControllerInteractionHandConstraint& right) {
    return left.target == right.target &&
        left.affordance_id == right.affordance_id &&
        left.hand == right.hand;
}

quat normalized_rotation(quat rotation) {
    const float magnitude = quat_length(rotation);
    return rotation / magnitude;
}

FlatWorldPose flat_world_pose(const FlatControllerPose& pose) {
    FlatWorldPose world;
    for (size_t bone = 0; bone < kFlatControllerBoneCount; ++bone) {
        const int32_t parent = kFlatControllerParents[bone];
        if (parent < 0) {
            world.positions[bone] = pose.positions[bone];
            world.velocities[bone] = pose.velocities[bone];
            world.rotations[bone] = normalized_rotation(pose.rotations[bone]);
            world.angular_velocities[bone] = pose.angular_velocities[bone];
            continue;
        }

        const size_t parent_bone = static_cast<size_t>(parent);
        const vec3 offset = quat_mul_vec3(
            world.rotations[parent_bone], pose.positions[bone]);
        world.positions[bone] = world.positions[parent_bone] + offset;
        world.rotations[bone] = normalized_rotation(quat_mul(
            world.rotations[parent_bone], pose.rotations[bone]));
        world.velocities[bone] =
            world.velocities[parent_bone] +
            cross(world.angular_velocities[parent_bone], offset) +
            quat_mul_vec3(
                world.rotations[parent_bone], pose.velocities[bone]);
        world.angular_velocities[bone] =
            world.angular_velocities[parent_bone] +
            quat_mul_vec3(
                world.rotations[parent_bone],
                pose.angular_velocities[bone]);
    }
    return world;
}

void copy_flat_bone_channels(
    FlatControllerPose& destination,
    const FlatControllerPose& source,
    size_t bone) {
    destination.positions[bone] = source.positions[bone];
    destination.velocities[bone] = source.velocities[bone];
    destination.rotations[bone] = source.rotations[bone];
    destination.angular_velocities[bone] = source.angular_velocities[bone];
}

void blend_flat_arm_channels(
    FlatControllerPose& destination,
    const FlatControllerPose& source,
    const FlatControllerPose& target,
    size_t arm_begin,
    float alpha) {
    for (size_t bone = arm_begin; bone < arm_begin + 4U; ++bone) {
        if (alpha <= 0.0F) {
            copy_flat_bone_channels(destination, source, bone);
        } else if (alpha >= 1.0F) {
            copy_flat_bone_channels(destination, target, bone);
        } else {
            destination.positions[bone] = lerp(
                source.positions[bone], target.positions[bone], alpha);
            destination.velocities[bone] = lerp(
                source.velocities[bone], target.velocities[bone], alpha);
            destination.rotations[bone] = quat_nlerp_shortest(
                source.rotations[bone], target.rotations[bone], alpha);
            destination.angular_velocities[bone] = lerp(
                source.angular_velocities[bone],
                target.angular_velocities[bone],
                alpha);
        }
    }
}

bool inactive_arm_step_within_limits(
    const FlatControllerPose& previous,
    const FlatControllerPose& candidate,
    size_t arm_begin) {
    const FlatWorldPose previous_world = flat_world_pose(previous);
    const FlatWorldPose candidate_world = flat_world_pose(candidate);
    for (size_t bone = arm_begin; bone < arm_begin + 4U; ++bone) {
        if (length(
                candidate_world.positions[bone] -
                previous_world.positions[bone]) >
                kInactiveArmTranslationStepLimitM ||
            quat_angle_between(
                previous_world.rotations[bone],
                candidate_world.rotations[bone]) >
                kInactiveArmRotationStepLimitRadians) {
            return false;
        }
    }
    return true;
}

bool flat_pose_step_within_limits(
    const FlatControllerPose& previous,
    const FlatControllerPose& candidate,
    float translation_limit_m,
    float rotation_limit_radians) {
    const FlatWorldPose previous_world = flat_world_pose(previous);
    const FlatWorldPose candidate_world = flat_world_pose(candidate);
    for (size_t bone = 0; bone < kFlatControllerBoneCount; ++bone) {
        if (length(
                candidate_world.positions[bone] -
                previous_world.positions[bone]) >
                translation_limit_m ||
            quat_angle_between(
                previous_world.rotations[bone],
                candidate_world.rotations[bone]) >
                rotation_limit_radians) {
            return false;
        }
    }
    return true;
}

void begin_lower_body_inertialization(
    FlatControllerPose& offsets,
    const FlatControllerPose& source,
    const FlatControllerPose& target) {
    offsets = {};
    for (size_t bone = 0;
         bone < kLayeredCarryLowerBodyBoneCount;
         ++bone) {
        inertialize_transition(
            offsets.positions[bone],
            offsets.velocities[bone],
            source.positions[bone],
            source.velocities[bone],
            target.positions[bone],
            target.velocities[bone]);
        inertialize_transition(
            offsets.rotations[bone],
            offsets.angular_velocities[bone],
            source.rotations[bone],
            source.angular_velocities[bone],
            target.rotations[bone],
            target.angular_velocities[bone]);
    }
}

void copy_lower_body_channels(
    FlatControllerPose& destination,
    const FlatControllerPose& source) {
    for (size_t bone = 0;
         bone < kLayeredCarryLowerBodyBoneCount;
         ++bone) {
        copy_flat_bone_channels(destination, source, bone);
    }
}

bool lower_body_step_within_limits(
    const FlatControllerPose& previous,
    const FlatControllerPose& candidate,
    float translation_limit_m = kInactiveArmTranslationStepLimitM,
    float rotation_limit_radians =
        kInactiveArmRotationStepLimitRadians) {
    FlatControllerPose baseline = candidate;
    copy_lower_body_channels(baseline, previous);
    return flat_pose_step_within_limits(
        baseline,
        candidate,
        translation_limit_m,
        rotation_limit_radians);
}

bool lower_body_offsets_are_terminal(
    const FlatControllerPose& offsets) {
    for (size_t bone = 0;
         bone < kLayeredCarryLowerBodyBoneCount;
         ++bone) {
        if (length(offsets.positions[bone]) >
                kLayeredCarryTerminalPositionM ||
            length(offsets.velocities[bone]) >
                kLayeredCarryTerminalVelocityMps ||
            quat_angle_between(offsets.rotations[bone], quat()) >
                kLayeredCarryTerminalRotationRadians ||
            length(offsets.angular_velocities[bone]) >
                kLayeredCarryTerminalAngularVelocityRadiansPerSecond) {
            return false;
        }
    }
    return true;
}

void update_lower_body_inertialization(
    FlatControllerPose& output,
    FlatControllerPose& offsets,
    const FlatControllerPose& target,
    float dt) {
    for (size_t bone = 0;
         bone < kLayeredCarryLowerBodyBoneCount;
         ++bone) {
        inertialize_update(
            output.positions[bone],
            output.velocities[bone],
            offsets.positions[bone],
            offsets.velocities[bone],
            target.positions[bone],
            target.velocities[bone],
            kLayeredCarryLowerBodyHalfLifeSeconds,
            dt);
        inertialize_update(
            output.rotations[bone],
            output.angular_velocities[bone],
            offsets.rotations[bone],
            offsets.angular_velocities[bone],
            target.rotations[bone],
            target.angular_velocities[bone],
            kLayeredCarryLowerBodyHalfLifeSeconds,
            dt);
        output.rotations[bone] = normalized_rotation(
            output.rotations[bone]);
    }
}

void solve_local_channel(
    vec3& local_position,
    vec3& local_velocity,
    quat& local_rotation,
    vec3& local_angular_velocity,
    vec3 desired_position,
    vec3 desired_velocity,
    quat desired_rotation,
    vec3 desired_angular_velocity,
    vec3 parent_position,
    vec3 parent_velocity,
    quat parent_rotation,
    vec3 parent_angular_velocity) {
    const vec3 world_offset = desired_position - parent_position;
    local_position = quat_inv_mul_vec3(parent_rotation, world_offset);
    local_rotation = normalized_rotation(
        quat_inv_mul(parent_rotation, desired_rotation));
    local_velocity = quat_inv_mul_vec3(
        parent_rotation,
        desired_velocity - parent_velocity -
            cross(parent_angular_velocity, world_offset));
    local_angular_velocity = quat_inv_mul_vec3(
        parent_rotation,
        desired_angular_velocity - parent_angular_velocity);
}

void update_g1_world_bone(
    WorldPose& world,
    const Pose& pose,
    size_t bone) {
    const int32_t parent = g1_skeleton::kParents[bone];
    if (parent < 0) {
        world.positions[bone] = pose.positions[bone];
        world.velocities[bone] = pose.velocities[bone];
        world.rotations[bone] = normalized_rotation(pose.rotations[bone]);
        world.angular_velocities[bone] = pose.angular_velocities[bone];
        return;
    }
    const size_t parent_bone = static_cast<size_t>(parent);
    const vec3 offset = quat_mul_vec3(
        world.rotations[parent_bone], pose.positions[bone]);
    world.positions[bone] = world.positions[parent_bone] + offset;
    world.rotations[bone] = normalized_rotation(quat_mul(
        world.rotations[parent_bone], pose.rotations[bone]));
    world.velocities[bone] =
        world.velocities[parent_bone] +
        cross(world.angular_velocities[parent_bone], offset) +
        quat_mul_vec3(
            world.rotations[parent_bone], pose.velocities[bone]);
    world.angular_velocities[bone] =
        world.angular_velocities[parent_bone] +
        quat_mul_vec3(
            world.rotations[parent_bone],
            pose.angular_velocities[bone]);
}

void update_flat_world_bone(
    FlatWorldPose& world,
    const FlatControllerPose& pose,
    size_t bone) {
    const int32_t parent = kFlatControllerParents[bone];
    if (parent < 0) {
        world.positions[bone] = pose.positions[bone];
        world.velocities[bone] = pose.velocities[bone];
        world.rotations[bone] = normalized_rotation(pose.rotations[bone]);
        world.angular_velocities[bone] = pose.angular_velocities[bone];
        return;
    }
    const size_t parent_bone = static_cast<size_t>(parent);
    const vec3 offset = quat_mul_vec3(
        world.rotations[parent_bone], pose.positions[bone]);
    world.positions[bone] = world.positions[parent_bone] + offset;
    world.rotations[bone] = normalized_rotation(quat_mul(
        world.rotations[parent_bone], pose.rotations[bone]));
    world.velocities[bone] =
        world.velocities[parent_bone] +
        cross(world.angular_velocities[parent_bone], offset) +
        quat_mul_vec3(
            world.rotations[parent_bone], pose.velocities[bone]);
    world.angular_velocities[bone] =
        world.angular_velocities[parent_bone] +
        quat_mul_vec3(
            world.rotations[parent_bone],
            pose.angular_velocities[bone]);
}

void preserve_flat_arm_world_channels(
    FlatControllerPose& destination,
    const FlatControllerPose& previous,
    size_t arm_begin) {
    const FlatWorldPose previous_world = flat_world_pose(previous);
    FlatWorldPose destination_world = flat_world_pose(destination);
    for (size_t bone = arm_begin; bone < arm_begin + 4U; ++bone) {
        const int32_t parent = kFlatControllerParents[bone];
        if (parent < 0) {
            throw FormatError("inactive arm bone has no parent");
        }
        const size_t parent_bone = static_cast<size_t>(parent);
        solve_local_channel(
            destination.positions[bone],
            destination.velocities[bone],
            destination.rotations[bone],
            destination.angular_velocities[bone],
            previous_world.positions[bone],
            previous_world.velocities[bone],
            previous_world.rotations[bone],
            previous_world.angular_velocities[bone],
            destination_world.positions[parent_bone],
            destination_world.velocities[parent_bone],
            destination_world.rotations[parent_bone],
            destination_world.angular_velocities[parent_bone]);
        update_flat_world_bone(destination_world, destination, bone);
    }
}

bool apply_bounded_flat_arm_target(
    FlatControllerPose& output,
    const FlatControllerPose& desired,
    const FlatControllerPose& previous,
    size_t arm_begin) {
    if (inactive_arm_step_within_limits(previous, desired, arm_begin)) {
        output = desired;
        return true;
    }

    FlatControllerPose accepted = output;
    preserve_flat_arm_world_channels(accepted, previous, arm_begin);
    if (!inactive_arm_step_within_limits(previous, accepted, arm_begin)) {
        throw FormatError(
            "inactive arm world-preserving fallback exceeded step limits");
    }
    const FlatControllerPose baseline = accepted;
    float lower = 0.0F;
    float upper = 1.0F;
    for (int iteration = 0;
         iteration < kInactiveArmBackoffIterations;
         ++iteration) {
        const float trial_alpha = 0.5F * (lower + upper);
        FlatControllerPose trial = output;
        blend_flat_arm_channels(
            trial, baseline, desired, arm_begin, trial_alpha);
        if (inactive_arm_step_within_limits(previous, trial, arm_begin)) {
            lower = trial_alpha;
            accepted = trial;
        } else {
            upper = trial_alpha;
        }
    }
    output = accepted;
    return false;
}

Transform frame_transform(
    const std::vector<float>& positions,
    const std::vector<float>& rotations,
    size_t index) {
    return {
        vec3(
            positions.at(index * 3U),
            positions.at(index * 3U + 1U),
            positions.at(index * 3U + 2U)),
        quat(
            rotations.at(index * 4U),
            rotations.at(index * 4U + 1U),
            rotations.at(index * 4U + 2U),
            rotations.at(index * 4U + 3U))};
}

vec3 clip_vector(const std::vector<float>& values, size_t clip) {
    return vec3(
        values.at(clip * 3U),
        values.at(clip * 3U + 1U),
        values.at(clip * 3U + 2U));
}

Transform clip_transform(
    const std::vector<float>& positions,
    const std::vector<float>& rotations,
    size_t clip) {
    return frame_transform(positions, rotations, clip);
}

FlatControllerPose blend_flat_pose(
    const FlatControllerPose& source,
    const FlatControllerPose& target,
    float alpha) {
    FlatControllerPose blended;
    for (size_t bone = 0; bone < blended.positions.size(); ++bone) {
        blended.positions[bone] =
            lerp(source.positions[bone], target.positions[bone], alpha);
        blended.velocities[bone] =
            lerp(source.velocities[bone], target.velocities[bone], alpha);
        blended.rotations[bone] = quat_nlerp_shortest(
            source.rotations[bone], target.rotations[bone], alpha);
        blended.angular_velocities[bone] = lerp(
            source.angular_velocities[bone],
            target.angular_velocities[bone],
            alpha);
    }
    blended.foot_contacts =
        alpha < 0.5F ? source.foot_contacts : target.foot_contacts;
    return blended;
}

}  // namespace

Pose expand_flat_controller_pose(
    const FlatControllerPose& flat_pose,
    const Pose& interaction_reference) {
    validate_flat_pose(flat_pose);
    validate_interaction_pose(interaction_reference);

    const FlatWorldPose flat_world = flat_world_pose(flat_pose);
    Pose expanded = interaction_reference;
    WorldPose expanded_world{};
    for (size_t g1_bone = 0; g1_bone < g1_skeleton::BoneCount; ++g1_bone) {
        const int32_t flat_bone_value = kG1ToFlatBone[g1_bone];
        if (flat_bone_value >= 0) {
            const size_t flat_bone =
                static_cast<size_t>(flat_bone_value);
            const int32_t parent = g1_skeleton::kParents[g1_bone];
            if (parent < 0) {
                expanded.positions[g1_bone] = flat_world.positions[flat_bone];
                expanded.velocities[g1_bone] =
                    flat_world.velocities[flat_bone];
                expanded.rotations[g1_bone] = normalized_rotation(
                    flat_world.rotations[flat_bone]);
                expanded.angular_velocities[g1_bone] =
                    flat_world.angular_velocities[flat_bone];
            } else {
                const size_t parent_bone = static_cast<size_t>(parent);
                solve_local_channel(
                    expanded.positions[g1_bone],
                    expanded.velocities[g1_bone],
                    expanded.rotations[g1_bone],
                    expanded.angular_velocities[g1_bone],
                    flat_world.positions[flat_bone],
                    flat_world.velocities[flat_bone],
                    flat_world.rotations[flat_bone],
                    flat_world.angular_velocities[flat_bone],
                    expanded_world.positions[parent_bone],
                    expanded_world.velocities[parent_bone],
                    expanded_world.rotations[parent_bone],
                    expanded_world.angular_velocities[parent_bone]);
            }
        }
        update_g1_world_bone(expanded_world, expanded, g1_bone);
    }
    expanded.foot_contacts = flat_pose.foot_contacts;
    validate_interaction_pose(expanded);
    return expanded;
}

FlatControllerPose collapse_interaction_pose(
    const Pose& interaction_pose,
    const Pose& interaction_reference,
    const FlatControllerPose& flat_reference) {
    validate_interaction_pose(interaction_pose);
    validate_interaction_pose(interaction_reference);
    validate_flat_pose(flat_reference);

    if (raw_pose_channels_equal(interaction_pose, interaction_reference)) {
        return flat_reference;
    }

    const WorldPose source_world = world_pose(interaction_pose);
    const WorldPose source_reference_world = world_pose(interaction_reference);
    const FlatWorldPose flat_reference_world =
        flat_world_pose(flat_reference);
    FlatControllerPose retargeted = flat_reference;
    FlatWorldPose retargeted_world{};

    for (size_t flat_bone = 0; flat_bone < kFlatControllerBoneCount;
         ++flat_bone) {
        const int32_t source_value = kFlatToG1Bone[flat_bone];
        if (source_value >= 0) {
            const size_t source_bone = static_cast<size_t>(source_value);
            const quat world_delta = normalized_rotation(quat_mul(
                source_world.rotations[source_bone],
                quat_inv(source_reference_world.rotations[source_bone])));
            const quat desired_world_rotation = normalized_rotation(quat_mul(
                world_delta,
                flat_reference_world.rotations[flat_bone]));
            const vec3 desired_world_angular_velocity =
                flat_reference_world.angular_velocities[flat_bone] +
                source_world.angular_velocities[source_bone] -
                source_reference_world.angular_velocities[source_bone];

            const int32_t parent = kFlatControllerParents[flat_bone];
            if (parent < 0) {
                retargeted.positions[flat_bone] =
                    flat_reference.positions[flat_bone] +
                    source_world.positions[source_bone] -
                    source_reference_world.positions[source_bone];
                retargeted.velocities[flat_bone] =
                    flat_reference.velocities[flat_bone] +
                    source_world.velocities[source_bone] -
                    source_reference_world.velocities[source_bone];
                retargeted.rotations[flat_bone] = desired_world_rotation;
                retargeted.angular_velocities[flat_bone] =
                    desired_world_angular_velocity;
            } else {
                const size_t parent_bone = static_cast<size_t>(parent);
                retargeted.positions[flat_bone] =
                    flat_reference.positions[flat_bone];
                retargeted.velocities[flat_bone] =
                    flat_reference.velocities[flat_bone];
                retargeted.rotations[flat_bone] = normalized_rotation(
                    quat_inv_mul(
                        retargeted_world.rotations[parent_bone],
                        desired_world_rotation));
                retargeted.angular_velocities[flat_bone] =
                    quat_inv_mul_vec3(
                        retargeted_world.rotations[parent_bone],
                        desired_world_angular_velocity -
                            retargeted_world
                                .angular_velocities[parent_bone]);
            }
        }
        update_flat_world_bone(retargeted_world, retargeted, flat_bone);
    }
    retargeted.foot_contacts = interaction_pose.foot_contacts;
    validate_flat_pose(retargeted);
    return retargeted;
}

const RuntimeOutput& ControllerInteractionScheduler::tick(
    ControllerInteractionEdges edges,
    const LocomotionProvider& locomotion_provider,
    const PickRequestResolver& request_resolver,
    const RuntimeUpdate& runtime_update) {
    updated_last_tick_ = false;
    pending_interact_ = pending_interact_ || edges.interact_pressed;
    pending_cancel_ = pending_cancel_ || edges.cancel_pressed;
    pending_reset_ = pending_reset_ || edges.reset_pressed;

    phase_ += kInteractionRuntimeRate;
    if (phase_ < kControllerRate) {
        return cached_output_;
    }
    phase_ -= kControllerRate;

    RuntimeInput input;
    input.dt = kInteractionRuntimeStepSeconds;
    input.locomotion = locomotion_provider();
    input.interact_pressed = pending_interact_;
    input.cancel_pressed = pending_cancel_;
    input.reset_pressed = pending_reset_;
    if (pending_interact_) {
        input.pick_request = request_resolver(input.locomotion);
    }

    RuntimeOutput next_output = runtime_update(input);
    cached_output_ = std::move(next_output);
    updated_last_tick_ = true;
    pending_interact_ = false;
    pending_cancel_ = false;
    pending_reset_ = false;
    return cached_output_;
}

int ControllerInteractionScheduler::phase() const {
    return phase_;
}

bool ControllerInteractionScheduler::updated_last_tick() const {
    return updated_last_tick_;
}

const RuntimeOutput& ControllerInteractionScheduler::cached_output() const {
    return cached_output_;
}

ControllerInteractionFrameState ControllerInteractionFrameHandoff::apply(
    const FlatControllerPose& locomotion_pose,
    const RuntimeOutput& runtime_output,
    float dt,
    std::optional<ControllerInteractionHandConstraint> hand_constraint) {
    const bool layered_carry =
        runtime_output.owns_pose &&
        runtime_output.diagnostics.state == RuntimeState::Carry &&
        !runtime_output.diagnostics.recorded_carry;
    const bool validate_lower_handoff_final_composite =
        runtime_owned_last_update_ && layered_carry &&
        (!layered_carry_last_update_ ||
         lower_body_inertialization_active_);
    std::optional<ControllerInteractionFrameHandoff> handoff_before;
    if (validate_lower_handoff_final_composite) {
        handoff_before.emplace(*this);
    }
    ControllerInteractionFrameState state;
    state.runtime_owns_pose = runtime_output.owns_pose;

    if (runtime_output.owns_pose) {
        if (!runtime_owned_last_update_) {
            ownership_interaction_reference_ = runtime_output.pose;
            ownership_flat_reference_ =
                release_active_ ? last_rendered_pose_ : locomotion_pose;
            release_active_ = false;
            runtime_owned_last_update_ = true;
            state.pose = ownership_flat_reference_;
            layered_carry_last_update_ = layered_carry;
            lower_body_inertialization_active_ = false;
            lower_body_inertialization_seconds_ = 0.0F;
            lower_body_inertial_offsets_ = {};
            target_rig_arm_ik_.reset();
            ownership_hand_constraint_.reset();
            if (hand_constraint.has_value() &&
                constraint_identity_matches(
                    *hand_constraint, runtime_output.diagnostics)) {
                ownership_hand_constraint_ = hand_constraint;
                target_rig_arm_ik_.begin_epoch(
                    ownership_interaction_reference_,
                    ownership_flat_reference_,
                    hand_constraint->hand);
            }
        } else {
            FlatControllerPose target = collapse_interaction_pose(
                runtime_output.pose,
                ownership_interaction_reference_,
                ownership_flat_reference_);
            for (size_t bone = 0; bone < target.rotations.size(); ++bone) {
                if (quat_dot(
                        last_rendered_pose_.rotations[bone],
                        target.rotations[bone]) < 0.0F) {
                    target.rotations[bone] = -target.rotations[bone];
                }
            }

            if (layered_carry) {
                state.pose = locomotion_pose;
                for (size_t bone = 10U;
                     bone < kFlatControllerBoneCount;
                     ++bone) {
                    state.pose.positions[bone] = target.positions[bone];
                    state.pose.velocities[bone] = target.velocities[bone];
                    state.pose.rotations[bone] = target.rotations[bone];
                    state.pose.angular_velocities[bone] =
                        target.angular_velocities[bone];
                }
                state.pose.foot_contacts = locomotion_pose.foot_contacts;

                if (!layered_carry_last_update_) {
                    begin_lower_body_inertialization(
                        lower_body_inertial_offsets_,
                        last_rendered_pose_,
                        locomotion_pose);
                    copy_lower_body_channels(
                        state.pose, last_rendered_pose_);
                    if (!lower_body_step_within_limits(
                            last_rendered_pose_, state.pose)) {
                        throw FormatError(
                            "layered Carry lower-body seam exceeded step limits");
                    }
                    lower_body_inertialization_active_ = true;
                    lower_body_inertialization_seconds_ = 0.0F;
                } else if (lower_body_inertialization_active_) {
                    const float step_seconds = std::max(dt, 0.0F);
                    const bool nominal_time_elapsed =
                        lower_body_inertialization_seconds_ + step_seconds >=
                        kLayeredCarryLowerBodyNominalSeconds;
                    const bool terminal_offsets =
                        lower_body_offsets_are_terminal(
                            lower_body_inertial_offsets_);
                    const bool terminal_step = lower_body_step_within_limits(
                        last_rendered_pose_,
                        state.pose,
                        kLayeredCarryTerminalStepTranslationM,
                        kLayeredCarryTerminalStepRotationRadians);
                    if (nominal_time_elapsed && terminal_offsets &&
                        terminal_step) {
                        lower_body_inertialization_active_ = false;
                        lower_body_inertialization_seconds_ = 0.0F;
                        lower_body_inertial_offsets_ = {};
                    } else if (nominal_time_elapsed && !terminal_step) {
                        begin_lower_body_inertialization(
                            lower_body_inertial_offsets_,
                            last_rendered_pose_,
                            locomotion_pose);
                        copy_lower_body_channels(
                            state.pose, last_rendered_pose_);
                        lower_body_inertialization_seconds_ = 0.0F;
                    } else {
                        FlatControllerPose trial = state.pose;
                        FlatControllerPose trial_offsets =
                            lower_body_inertial_offsets_;
                        update_lower_body_inertialization(
                            trial,
                            trial_offsets,
                            locomotion_pose,
                            step_seconds);
                        if (lower_body_step_within_limits(
                                last_rendered_pose_, trial)) {
                            state.pose = trial;
                            lower_body_inertial_offsets_ = trial_offsets;
                            lower_body_inertialization_seconds_ = std::min(
                                kLayeredCarryLowerBodyNominalSeconds,
                                lower_body_inertialization_seconds_ +
                                    step_seconds);
                        } else {
                            FlatControllerPose backoff_source_offsets =
                                lower_body_inertial_offsets_;
                            FlatControllerPose zero_decay = state.pose;
                            update_lower_body_inertialization(
                                zero_decay,
                                backoff_source_offsets,
                                locomotion_pose,
                                0.0F);
                            float elapsed_before_backoff =
                                lower_body_inertialization_seconds_;
                            FlatControllerPose accepted = zero_decay;
                            if (!lower_body_step_within_limits(
                                    last_rendered_pose_, zero_decay)) {
                                begin_lower_body_inertialization(
                                    backoff_source_offsets,
                                    last_rendered_pose_,
                                    locomotion_pose);
                                accepted = state.pose;
                                copy_lower_body_channels(
                                    accepted, last_rendered_pose_);
                                elapsed_before_backoff = 0.0F;
                            }
                            if (!lower_body_step_within_limits(
                                    last_rendered_pose_, accepted)) {
                                throw FormatError(
                                    "layered Carry lower-body rebase exceeded step limits");
                            }

                            FlatControllerPose accepted_offsets =
                                backoff_source_offsets;
                            float accepted_seconds = 0.0F;
                            float lower = 0.0F;
                            float upper = step_seconds;
                            for (int iteration = 0;
                                 iteration <
                                     kLayeredCarryLowerBodyBackoffIterations;
                                 ++iteration) {
                                const float trial_seconds =
                                    0.5F * (lower + upper);
                                FlatControllerPose backoff_trial = state.pose;
                                FlatControllerPose backoff_trial_offsets =
                                    backoff_source_offsets;
                                update_lower_body_inertialization(
                                    backoff_trial,
                                    backoff_trial_offsets,
                                    locomotion_pose,
                                    trial_seconds);
                                if (lower_body_step_within_limits(
                                        last_rendered_pose_,
                                        backoff_trial)) {
                                    lower = trial_seconds;
                                    accepted = backoff_trial;
                                    accepted_offsets =
                                        backoff_trial_offsets;
                                    accepted_seconds = trial_seconds;
                                } else {
                                    upper = trial_seconds;
                                }
                            }
                            state.pose = accepted;
                            lower_body_inertial_offsets_ = accepted_offsets;
                            lower_body_inertialization_seconds_ = std::min(
                                kLayeredCarryLowerBodyNominalSeconds,
                                elapsed_before_backoff + accepted_seconds);
                        }
                    }
                }
            } else {
                state.pose = target;
                lower_body_inertialization_active_ = false;
                lower_body_inertialization_seconds_ = 0.0F;
                lower_body_inertial_offsets_ = {};
            }

            const bool valid_active_hand =
                runtime_output.diagnostics.hand == Hand::Left ||
                runtime_output.diagnostics.hand == Hand::Right;
            const bool locomotion_target =
                layered_carry && valid_active_hand &&
                runtime_output.diagnostics.inactive_arm_targets_locomotion;
            if (locomotion_target) {
                inactive_arm_return_hand_.reset();
                if (!inactive_arm_locomotion_hand_.has_value() ||
                    *inactive_arm_locomotion_hand_ !=
                        runtime_output.diagnostics.hand) {
                    inactive_arm_locomotion_hand_ =
                        runtime_output.diagnostics.hand;
                    inactive_arm_blend_seconds_ = 0.0F;
                    inactive_arm_blend_source_ = last_rendered_pose_;
                }
                const float inactive_arm_alpha = std::clamp(
                    inactive_arm_blend_seconds_ / kInactiveArmBlendSeconds,
                    0.0F,
                    1.0F);
                const size_t inactive_arm_begin =
                    runtime_output.diagnostics.hand == Hand::Left
                    ? 19U
                    : 15U;
                FlatControllerPose desired = state.pose;
                blend_flat_arm_channels(
                    desired,
                    inactive_arm_blend_source_,
                    locomotion_pose,
                    inactive_arm_begin,
                    inactive_arm_alpha);
                (void)apply_bounded_flat_arm_target(
                    state.pose,
                    desired,
                    last_rendered_pose_,
                    inactive_arm_begin);
                inactive_arm_blend_seconds_ = std::min(
                    kInactiveArmBlendSeconds,
                    inactive_arm_blend_seconds_ + std::max(dt, 0.0F));
            } else {
                if (inactive_arm_locomotion_hand_.has_value()) {
                    inactive_arm_return_hand_ =
                        inactive_arm_locomotion_hand_;
                    inactive_arm_locomotion_hand_.reset();
                    inactive_arm_blend_seconds_ = 0.0F;
                    inactive_arm_blend_source_ = last_rendered_pose_;
                }
                if (inactive_arm_return_hand_.has_value()) {
                    if (!valid_active_hand ||
                        *inactive_arm_return_hand_ !=
                            runtime_output.diagnostics.hand) {
                        throw FormatError(
                            "inactive arm return changed active hand");
                    }
                    const float inactive_arm_alpha = std::clamp(
                        inactive_arm_blend_seconds_ /
                            kInactiveArmBlendSeconds,
                        0.0F,
                        1.0F);
                    const size_t inactive_arm_begin =
                        runtime_output.diagnostics.hand == Hand::Left
                        ? 19U
                        : 15U;
                    FlatControllerPose desired = state.pose;
                    blend_flat_arm_channels(
                        desired,
                        inactive_arm_blend_source_,
                        state.pose,
                        inactive_arm_begin,
                        inactive_arm_alpha);
                    const bool full_target_applied =
                        apply_bounded_flat_arm_target(
                            state.pose,
                            desired,
                            last_rendered_pose_,
                            inactive_arm_begin);
                    inactive_arm_blend_seconds_ = std::min(
                        kInactiveArmBlendSeconds,
                        inactive_arm_blend_seconds_ + std::max(dt, 0.0F));
                    if (inactive_arm_alpha >= 1.0F &&
                        full_target_applied) {
                        inactive_arm_return_hand_.reset();
                        inactive_arm_blend_seconds_ = 0.0F;
                    }
                } else {
                    inactive_arm_blend_seconds_ = 0.0F;
                }
            }
        }

        if (target_rig_arm_ik_.active()) {
            state.hand_constraint_calibration_rotation =
                target_rig_arm_ik_.calibration_rotation();
        }
        if (hand_constraint.has_value() &&
            ownership_hand_constraint_.has_value() &&
            target_rig_arm_ik_.active() &&
            constraint_identity_matches(
                *hand_constraint, runtime_output.diagnostics) &&
            constraint_identity_matches(
                *hand_constraint, *ownership_hand_constraint_)) {
            state.hand_constraint_validated = true;
            state.hand_constraint_result = target_rig_arm_ik_.solve(
                state.pose,
                hand_constraint->grasp_world,
                runtime_output.diagnostics.hand_constraint_weight,
                dt);
        }

        if (validate_lower_handoff_final_composite &&
            !flat_pose_step_within_limits(
                handoff_before->last_rendered_pose_,
                state.pose,
                kInactiveArmTranslationStepLimitM,
                kInactiveArmRotationStepLimitRadians)) {
            *this = *handoff_before;
            throw FormatError(
                "layered Carry final composite exceeded step limits");
        }

        state.overrides_locomotion_pose = true;
        state.synchronize_simulation_root = !layered_carry;
        state.simulation_root_position = state.pose.positions[0];
        state.simulation_root_rotation = state.pose.rotations[0];
        last_rendered_pose_ = state.pose;
        layered_carry_last_update_ = layered_carry;
        return state;
    }

    if (runtime_owned_last_update_) {
        runtime_owned_last_update_ = false;
        layered_carry_last_update_ = false;
        lower_body_inertialization_active_ = false;
        lower_body_inertialization_seconds_ = 0.0F;
        lower_body_inertial_offsets_ = {};
        inactive_arm_locomotion_hand_.reset();
        inactive_arm_return_hand_.reset();
        inactive_arm_blend_seconds_ = 0.0F;
        release_active_ = true;
        blend_source_ = last_rendered_pose_;
        blend_seconds_ = 0.0F;
    }

    if (release_active_) {
        const float alpha = std::clamp(
            blend_seconds_ / kOwnershipBlendSeconds, 0.0F, 1.0F);
        if (alpha >= 1.0F) {
            state.pose = locomotion_pose;
            state.overrides_locomotion_pose = false;
            release_active_ = false;
        } else {
            state.pose = alpha <= 0.0F
                ? blend_source_
                : blend_flat_pose(blend_source_, locomotion_pose, alpha);
            state.overrides_locomotion_pose = true;
            last_rendered_pose_ = state.pose;
            blend_seconds_ += std::max(dt, 0.0F);
        }
        return state;
    }

    state.pose = locomotion_pose;
    return state;
}

void ControllerInteractionFrameHandoff::reset() {
    runtime_owned_last_update_ = false;
    release_active_ = false;
    blend_seconds_ = 0.0F;
    blend_source_ = {};
    ownership_interaction_reference_ = {};
    ownership_flat_reference_ = {};
    last_rendered_pose_ = {};
    target_rig_arm_ik_.reset();
    ownership_hand_constraint_.reset();
    inactive_arm_locomotion_hand_.reset();
    inactive_arm_return_hand_.reset();
    inactive_arm_blend_seconds_ = 0.0F;
    inactive_arm_blend_source_ = {};
    layered_carry_last_update_ = false;
    lower_body_inertialization_active_ = false;
    lower_body_inertialization_seconds_ = 0.0F;
    lower_body_inertial_offsets_ = {};
}

void ControllerInteractionSceneHandoff::reset_authority() {
    has_runtime_pose_ = false;
    runtime_target_ = {};
    current_runtime_object_world_ = {};
}

ControllerInteractionSceneState ControllerInteractionSceneHandoff::apply(
    const InteractionTarget* registry_target,
    const RuntimeOutput& runtime_output,
    const Transform& authored_fallback,
    float alpha,
    bool runtime_sample_updated) {
    if (!finite(alpha) || alpha < 0.0F || alpha >= 1.0F) {
        throw FormatError("scene interpolation alpha must be finite in [0, 1)");
    }

    if (registry_target == nullptr) {
        reset_authority();
        return {authored_fallback, false};
    }

    if (registry_target->state == ObjectState::Free ||
        registry_target->state == ObjectState::Targeted) {
        reset_authority();
        return {registry_target->object_world, false};
    }

    if (runtime_output.diagnostics.target != registry_target->handle) {
        reset_authority();
        return {registry_target->object_world, false};
    }

    if (!has_runtime_pose_ || runtime_target_ != registry_target->handle) {
        has_runtime_pose_ = true;
        runtime_target_ = registry_target->handle;
        current_runtime_object_world_ = runtime_output.object_world;
        return {current_runtime_object_world_, true};
    }

    if (runtime_sample_updated) {
        current_runtime_object_world_ = runtime_output.object_world;
    }
    return {current_runtime_object_world_, true};
}

const char* controller_carry_mode_label(const RuntimeOutput& output) {
    if (output.diagnostics.state != RuntimeState::Carry) {
        return "none";
    }
    return output.diagnostics.recorded_carry ? "recorded" : "layered";
}

InteractionTarget make_controller_demo_target(const Database& database) {
    if (database.clip_count == 0U || database.range_starts.empty() ||
        database.range_stops.empty()) {
        throw FormatError("interaction database has no clip 0");
    }

    const int32_t start = database.range_starts.at(0);
    const int32_t stop = database.range_stops.at(0);
    int32_t contact = -1;
    for (int32_t frame = start; frame < stop; ++frame) {
        if (database.phases.at(static_cast<size_t>(frame)) ==
            static_cast<uint8_t>(Phase::Contact)) {
            contact = frame;
            break;
        }
    }
    if (contact <= start) {
        throw FormatError("clip 0 has no pre-contact object sample");
    }

    const Transform source_table = clip_transform(
        database.table_positions, database.table_rotations, 0U);
    const Transform source_object = frame_transform(
        database.object_positions,
        database.object_rotations,
        static_cast<size_t>(contact - 1));
    const Transform object_in_table = compose(inverse(source_table), source_object);

    InteractionTarget target;
    target.handle = {1U, 1U};
    target.table_world = source_table;
    target.table_world.position.x = 0.0F;
    target.table_world.position.z = 3.0F;
    target.object_world = compose(target.table_world, object_in_table);
    target.table_size = clip_vector(database.table_sizes, 0U);
    target.object_dimensions = clip_vector(database.object_dimensions, 0U);
    target.state = ObjectState::Free;

    GraspAffordance affordance;
    affordance.id = 1U;
    affordance.hand = static_cast<Hand>(database.active_hands.at(0));
    affordance.hand_in_object = clip_transform(
        database.grasp_positions_object,
        database.grasp_rotations_object,
        0U);
    affordance.approach_direction_object =
        clip_vector(database.approach_directions_object, 0U);
    target.affordances.push_back(affordance);
    return target;
}

void validate_controller_interaction_pack(
    const Database& database,
    const Features& features) {
    if (database.frame_count != features.frame_count) {
        throw FormatError(
            "interaction database/features frame count mismatch");
    }
}

}  // namespace interaction
