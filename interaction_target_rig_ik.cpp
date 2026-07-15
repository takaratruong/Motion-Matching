#include "interaction_target_rig_ik.h"

#include "interaction_controller_adapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace interaction {
namespace {

constexpr float kPi = 3.14159265358979323846F;

struct Chain {
    size_t clavicle;
    size_t upper_arm;
    size_t forearm;
    size_t hand;
    size_t source_hand;
};

struct FlatWorldPose {
    std::array<vec3, kFlatControllerBoneCount> positions{};
    std::array<quat, kFlatControllerBoneCount> rotations{};
};

Chain chain_for(Hand hand) {
    if (hand == Hand::Left) {
        return {15U, 16U, 17U, 18U,
                static_cast<size_t>(g1_skeleton::LeftWrist)};
    }
    return {19U, 20U, 21U, 22U,
            static_cast<size_t>(g1_skeleton::RightWrist)};
}

bool finite(float value) {
    return std::isfinite(value);
}

bool finite(vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(quat value) {
    return finite(value.w) && finite(value.x) && finite(value.y) &&
           finite(value.z);
}

bool finite(Transform value) {
    return finite(value.position) && finite(value.rotation);
}

bool valid_rotation(quat value) {
    const float magnitude = quat_length(value);
    return finite(value) && finite(magnitude) && magnitude > 1.0e-8F;
}

bool valid_hand(Hand hand) {
    return hand == Hand::Left || hand == Hand::Right;
}

bool valid_flat_pose(const FlatControllerPose& pose) {
    for (size_t bone = 0; bone < kFlatControllerBoneCount; ++bone) {
        if (!finite(pose.positions[bone]) ||
            !finite(pose.velocities[bone]) ||
            !valid_rotation(pose.rotations[bone]) ||
            !finite(pose.angular_velocities[bone])) {
            return false;
        }
    }
    return pose.foot_contacts[0] <= 1U && pose.foot_contacts[1] <= 1U;
}

quat normalized(quat value) {
    const float magnitude = quat_length(value);
    if (!(magnitude > 1.0e-8F) || !finite(magnitude)) return quat();
    return value / magnitude;
}

vec3 safe_direction(vec3 value, vec3 fallback, float epsilon) {
    float magnitude = length(value);
    if (magnitude > epsilon && finite(magnitude)) return value / magnitude;
    magnitude = length(fallback);
    if (magnitude > epsilon && finite(magnitude)) return fallback / magnitude;
    return vec3(1.0F, 0.0F, 0.0F);
}

quat same_hemisphere(quat reference, quat value) {
    value = normalized(value);
    return quat_dot(reference, value) < 0.0F ? -value : value;
}

vec3 perpendicular(
    vec3 value,
    vec3 direction,
    float epsilon) {
    const vec3 projected = value - direction * dot(value, direction);
    const float magnitude = length(projected);
    if (!(magnitude > epsilon) || !finite(magnitude)) return vec3();
    return projected / magnitude;
}

vec3 deterministic_perpendicular(
    vec3 direction,
    quat spine_rotation,
    float epsilon) {
    constexpr std::array<vec3, 4> axes = {
        vec3(0.0F, 1.0F, 0.0F),
        vec3(0.0F, 0.0F, 1.0F),
        vec3(1.0F, 0.0F, 0.0F),
        vec3(0.0F, -1.0F, 0.0F)};
    for (vec3 axis : axes) {
        const vec3 candidate = perpendicular(
            quat_mul_vec3(spine_rotation, axis), direction, epsilon);
        if (length(candidate) > epsilon) return candidate;
    }
    return vec3(0.0F, 1.0F, 0.0F);
}

quat rotation_between(
    vec3 from,
    vec3 to,
    vec3 fallback_axis,
    float epsilon) {
    from = safe_direction(from, vec3(1.0F, 0.0F, 0.0F), epsilon);
    to = safe_direction(to, from, epsilon);
    const float cosine = std::clamp(dot(from, to), -1.0F, 1.0F);
    if (cosine > 1.0F - epsilon) return quat();
    if (cosine < -1.0F + epsilon) {
        vec3 axis = perpendicular(fallback_axis, from, epsilon);
        if (length(axis) <= epsilon) {
            axis = perpendicular(vec3(0.0F, 1.0F, 0.0F), from, epsilon);
        }
        if (length(axis) <= epsilon) {
            axis = perpendicular(vec3(0.0F, 0.0F, 1.0F), from, epsilon);
        }
        return normalized(quat_from_angle_axis(kPi, axis));
    }
    const vec3 axis = cross(from, to);
    return normalized(quat(
        1.0F + cosine, axis.x, axis.y, axis.z));
}

FlatWorldPose flat_world(const FlatControllerPose& pose) {
    FlatWorldPose world{};
    for (size_t bone = 0; bone < kFlatControllerBoneCount; ++bone) {
        const int32_t parent = kFlatControllerParents[bone];
        if (parent < 0) {
            world.positions[bone] = pose.positions[bone];
            world.rotations[bone] = normalized(pose.rotations[bone]);
            continue;
        }
        const size_t parent_bone = static_cast<size_t>(parent);
        world.positions[bone] = world.positions[parent_bone] +
            quat_mul_vec3(
                world.rotations[parent_bone], pose.positions[bone]);
        world.rotations[bone] = normalized(quat_mul(
            world.rotations[parent_bone], pose.rotations[bone]));
    }
    return world;
}

quat local_rotation(quat parent_world, quat desired_world, quat reference) {
    return same_hemisphere(
        reference,
        quat_inv_mul(parent_world, desired_world));
}

float orientation_error(quat left, quat right) {
    const float value = std::clamp(
        std::fabs(quat_dot(normalized(left), normalized(right))),
        0.0F,
        1.0F);
    return 2.0F * std::acos(value);
}

void update_angular_velocity(
    FlatControllerPose& pose,
    const FlatControllerPose& before,
    size_t bone,
    float dt) {
    pose.angular_velocities[bone] = quat_differentiate_angular_velocity(
        same_hemisphere(before.rotations[bone], pose.rotations[bone]),
        before.rotations[bone],
        dt);
}

}  // namespace

TargetRigArmIK::TargetRigArmIK(TargetRigArmIKConfig config)
    : config_(config) {
    if (!finite(config_.reach_epsilon_m) ||
        !finite(config_.pole_epsilon_m) ||
        !(config_.reach_epsilon_m > 0.0F) ||
        !(config_.pole_epsilon_m > 0.0F)) {
        throw std::invalid_argument("target-rig IK config is invalid");
    }
}

void TargetRigArmIK::begin_epoch(
    const Pose& interaction_reference,
    const FlatControllerPose& flat_reference,
    Hand hand) {
    reset();
    if (!valid_hand(hand) || !valid_flat_pose(flat_reference)) return;
    const Chain chain = chain_for(hand);
    const WorldPose source_world = world_pose(interaction_reference);
    const FlatWorldPose target_world = flat_world(flat_reference);
    if (!valid_rotation(source_world.rotations[chain.source_hand]) ||
        !valid_rotation(target_world.rotations[chain.hand])) {
        return;
    }
    calibration_rotation_ = normalized(quat_inv_mul(
        source_world.rotations[chain.source_hand],
        target_world.rotations[chain.hand]));
    hand_ = hand;
    active_ = true;
    has_previous_pole_ = false;

    const vec3 arm_direction = safe_direction(
        target_world.positions[chain.hand] -
            target_world.positions[chain.upper_arm],
        vec3(hand == Hand::Left ? -1.0F : 1.0F, 0.0F, 0.0F),
        config_.pole_epsilon_m);
    const vec3 previous_pole_world = perpendicular(
        target_world.positions[chain.forearm] -
            target_world.positions[chain.upper_arm],
        arm_direction,
        config_.pole_epsilon_m);
    previous_pole_spine_ = quat_inv_mul_vec3(
        target_world.rotations[12U], previous_pole_world);
    has_previous_pole_ =
        length(previous_pole_spine_) > config_.pole_epsilon_m;
}

TargetRigArmIKResult TargetRigArmIK::solve(
    FlatControllerPose& pose,
    Transform grasp_world,
    float weight,
    float dt) {
    TargetRigArmIKResult result{};
    if (!active_ || !(weight > 0.0F)) return result;
    if (!valid_flat_pose(pose) ||
        !finite(grasp_world) || !finite(weight) || !finite(dt) ||
        weight > 1.0F || !(dt > 0.0F) ||
        quat_length(grasp_world.rotation) <= 1.0e-8F) {
        return result;
    }

    const Chain chain = chain_for(hand_);
    const FlatControllerPose before = pose;
    FlatWorldPose world = flat_world(pose);
    const vec3 previous_pole_world = has_previous_pole_
        ? quat_mul_vec3(world.rotations[12U], previous_pole_spine_)
        : vec3();
    const vec3 base_hand_position = world.positions[chain.hand];
    const quat base_hand_rotation = world.rotations[chain.hand];
    const vec3 requested_target = lerp(
        base_hand_position, grasp_world.position, weight);
    const quat calibrated_grasp_rotation = normalized(quat_mul(
        normalized(grasp_world.rotation), calibration_rotation_));
    const quat requested_rotation = quat_nlerp_shortest(
        base_hand_rotation, calibrated_grasp_rotation, weight);

    const float clavicle_length = length(pose.positions[chain.upper_arm]);
    const float upper_length = length(pose.positions[chain.forearm]);
    const float forearm_length = length(pose.positions[chain.hand]);
    if (!(clavicle_length > config_.reach_epsilon_m) ||
        !(upper_length > config_.reach_epsilon_m) ||
        !(forearm_length > config_.reach_epsilon_m)) {
        return result;
    }

    const float arm_outer_physical = upper_length + forearm_length;
    const float arm_inner_physical =
        std::fabs(upper_length - forearm_length);
    const float arm_outer = std::max(
        config_.reach_epsilon_m,
        arm_outer_physical - config_.reach_epsilon_m);
    const float arm_inner = std::min(
        arm_outer,
        arm_inner_physical + config_.reach_epsilon_m);
    const float full_outer_physical =
        clavicle_length + arm_outer_physical;
    const float full_inner_physical = std::max({
        clavicle_length - upper_length - forearm_length,
        upper_length - clavicle_length - forearm_length,
        forearm_length - clavicle_length - upper_length,
        0.0F});
    const float full_outer = std::max(
        config_.reach_epsilon_m,
        clavicle_length + arm_outer);
    const float full_inner = std::max(
        config_.reach_epsilon_m,
        full_inner_physical + config_.reach_epsilon_m);

    const vec3 shoulder = world.positions[chain.clavicle];
    const vec3 current_hand_from_shoulder =
        world.positions[chain.hand] - shoulder;
    const float requested_distance = length(requested_target - shoulder);
    vec3 full_direction = safe_direction(
        requested_target - shoulder,
        current_hand_from_shoulder,
        config_.pole_epsilon_m);
    float solved_distance = requested_distance;
    result.reachable =
        requested_distance <=
            full_outer_physical + config_.reach_epsilon_m &&
        requested_distance >=
            full_inner_physical - config_.reach_epsilon_m;
    if (requested_distance > full_outer) solved_distance = full_outer;
    if (requested_distance < full_inner) solved_distance = full_inner;
    const vec3 solved_target = shoulder + full_direction * solved_distance;
    result.reach_shortfall_m = std::fabs(requested_distance - solved_distance);

    vec3 upper_root = world.positions[chain.upper_arm];
    float arm_distance = length(solved_target - upper_root);
    if (arm_distance > arm_outer + config_.reach_epsilon_m ||
        arm_distance < arm_inner - config_.reach_epsilon_m) {
        const float desired_arm_distance =
            arm_distance > arm_outer ? arm_outer : arm_inner;
        const float shoulder_target_distance = length(solved_target - shoulder);
        const vec3 shoulder_target_direction = safe_direction(
            solved_target - shoulder,
            world.positions[chain.hand] - shoulder,
            config_.pole_epsilon_m);
        const float denominator = std::max(
            2.0F * shoulder_target_distance,
            config_.reach_epsilon_m);
        const float along = std::clamp(
            (clavicle_length * clavicle_length -
             desired_arm_distance * desired_arm_distance +
             shoulder_target_distance * shoulder_target_distance) /
                denominator,
            -clavicle_length,
            clavicle_length);
        const float radial_length = std::sqrt(std::max(
            0.0F,
            clavicle_length * clavicle_length - along * along));
        vec3 radial = perpendicular(
            upper_root - shoulder,
            shoulder_target_direction,
            config_.pole_epsilon_m);
        if (length(radial) <= config_.pole_epsilon_m &&
            has_previous_pole_) {
            radial = perpendicular(
                previous_pole_world, shoulder_target_direction,
                config_.pole_epsilon_m);
        }
        if (length(radial) <= config_.pole_epsilon_m) {
            radial = deterministic_perpendicular(
                shoulder_target_direction,
                world.rotations[12U],
                config_.pole_epsilon_m);
        }
        const vec3 desired_upper_root =
            shoulder + shoulder_target_direction * along +
            radial * radial_length;
        const int32_t clavicle_parent_value =
            kFlatControllerParents[chain.clavicle];
        const size_t clavicle_parent =
            static_cast<size_t>(clavicle_parent_value);
        const quat clavicle_delta = rotation_between(
            upper_root - shoulder,
            desired_upper_root - shoulder,
            radial,
            config_.pole_epsilon_m);
        const quat desired_clavicle_world = normalized(quat_mul(
            clavicle_delta, world.rotations[chain.clavicle]));
        pose.rotations[chain.clavicle] = local_rotation(
            world.rotations[clavicle_parent],
            desired_clavicle_world,
            before.rotations[chain.clavicle]);
        world = flat_world(pose);
        upper_root = world.positions[chain.upper_arm];
        arm_distance = length(solved_target - upper_root);
        result.used_clavicle = orientation_error(
            pose.rotations[chain.clavicle],
            before.rotations[chain.clavicle]) > 1.0e-6F;
    }

    const vec3 arm_direction = safe_direction(
        solved_target - upper_root,
        world.positions[chain.hand] - upper_root,
        config_.pole_epsilon_m);
    const float clamped_arm_distance = std::clamp(
        arm_distance, arm_inner, arm_outer);
    const vec3 arm_target =
        upper_root + arm_direction * clamped_arm_distance;
    const float elbow_along = std::clamp(
        (upper_length * upper_length +
         clamped_arm_distance * clamped_arm_distance -
         forearm_length * forearm_length) /
            std::max(
                2.0F * clamped_arm_distance,
                config_.reach_epsilon_m),
        -upper_length,
        upper_length);
    const float elbow_radial = std::sqrt(std::max(
        0.0F,
        upper_length * upper_length - elbow_along * elbow_along));
    vec3 pole = perpendicular(
        world.positions[chain.forearm] - upper_root,
        arm_direction,
        config_.pole_epsilon_m);
    if (length(pole) <= config_.pole_epsilon_m && has_previous_pole_) {
        pole = perpendicular(
            previous_pole_world, arm_direction, config_.pole_epsilon_m);
    }
    if (length(pole) <= config_.pole_epsilon_m) {
        pole = deterministic_perpendicular(
            arm_direction, world.rotations[12U], config_.pole_epsilon_m);
    }
    if (has_previous_pole_ && dot(pole, previous_pole_world) < 0.0F) {
        pole = -pole;
    }
    const vec3 desired_elbow =
        upper_root + arm_direction * elbow_along + pole * elbow_radial;

    const quat upper_delta = rotation_between(
        world.positions[chain.forearm] - upper_root,
        desired_elbow - upper_root,
        pole,
        config_.pole_epsilon_m);
    const quat desired_upper_world = normalized(quat_mul(
        upper_delta, world.rotations[chain.upper_arm]));
    pose.rotations[chain.upper_arm] = local_rotation(
        world.rotations[chain.clavicle],
        desired_upper_world,
        before.rotations[chain.upper_arm]);
    world = flat_world(pose);

    const vec3 elbow = world.positions[chain.forearm];
    const quat forearm_delta = rotation_between(
        world.positions[chain.hand] - elbow,
        arm_target - elbow,
        pole,
        config_.pole_epsilon_m);
    const quat desired_forearm_world = normalized(quat_mul(
        forearm_delta, world.rotations[chain.forearm]));
    pose.rotations[chain.forearm] = local_rotation(
        world.rotations[chain.upper_arm],
        desired_forearm_world,
        before.rotations[chain.forearm]);
    world = flat_world(pose);

    pose.rotations[chain.hand] = local_rotation(
        world.rotations[chain.forearm],
        requested_rotation,
        before.rotations[chain.hand]);

    if (result.used_clavicle) {
        update_angular_velocity(pose, before, chain.clavicle, dt);
    }
    for (size_t bone : {chain.upper_arm, chain.forearm, chain.hand}) {
        update_angular_velocity(pose, before, bone, dt);
    }
    world = flat_world(pose);
    const vec3 solved_pole_world = perpendicular(
        world.positions[chain.forearm] - world.positions[chain.upper_arm],
        safe_direction(
            world.positions[chain.hand] - world.positions[chain.upper_arm],
            arm_direction,
            config_.pole_epsilon_m),
        config_.pole_epsilon_m);
    previous_pole_spine_ = quat_inv_mul_vec3(
        world.rotations[12U], solved_pole_world);
    has_previous_pole_ =
        length(previous_pole_spine_) > config_.pole_epsilon_m;

    result.applied = true;
    result.position_error_m = length(
        world.positions[chain.hand] - requested_target);
    result.orientation_error_radians = orientation_error(
        world.rotations[chain.hand], requested_rotation);
    if (result.position_error_m > 2.0F * config_.reach_epsilon_m) {
        result.reachable = false;
        result.reach_shortfall_m = std::max(
            result.reach_shortfall_m, result.position_error_m);
    }
    return result;
}

void TargetRigArmIK::reset() {
    active_ = false;
    hand_ = Hand::Right;
    calibration_rotation_ = quat();
    has_previous_pole_ = false;
    previous_pole_spine_ = vec3();
}

bool TargetRigArmIK::active() const {
    return active_;
}

quat TargetRigArmIK::calibration_rotation() const {
    return calibration_rotation_;
}

}  // namespace interaction
