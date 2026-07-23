#include "interaction_posture_ik.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace interaction {
namespace {

constexpr size_t kJointCount = kUpperBodyJointCount;
constexpr size_t kTaskResidualStart = 0U;
constexpr size_t kOrientationResidualStart = 3U;
constexpr size_t kElbowResidualStart = 6U;
constexpr size_t kSourceResidualStart = 9U;
constexpr size_t kTemporalResidualStart = 19U;
constexpr size_t kResidualDimension = 29U;
constexpr float kPi = 3.14159265358979323846F;
constexpr float kTwoPi = 2.0F * kPi;
constexpr float kLimitTolerance = 1.0e-5F;

using Residual = std::array<double, kResidualDimension>;
using Jacobian =
    std::array<std::array<double, kJointCount>, kResidualDimension>;
using NormalMatrix =
    std::array<std::array<double, kJointCount>, kJointCount>;
using NormalVector = std::array<double, kJointCount>;

struct Evaluation {
    Pose pose{};
    Residual residual{};
    float position_error_m = 0.0F;
    float orientation_error_radians = 0.0F;
    float elbow_pole_error_radians = 0.0F;
    bool elbow_pole_degenerate = false;
    double objective = 0.0;
};

const HingeJoint& metadata(Hand hand, size_t joint) {
    if (joint < kWaist.size()) return kWaist[joint];
    const size_t arm_joint = joint - kWaist.size();
    return hand == Hand::Left
        ? kLeftArm[arm_joint]
        : kRightArm[arm_joint];
}

bool finite(float value) {
    return std::isfinite(value);
}

bool finite(double value) {
    return std::isfinite(value);
}

bool finite(vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(quat value) {
    return finite(value.w) && finite(value.x) &&
           finite(value.y) && finite(value.z);
}

quat normalize_exact(quat value) {
    const float norm = quat_length(value);
    if (!(norm > 1.0e-12F) || !finite(norm)) return quat();
    return value / norm;
}

vec3 shortest_log(quat value) {
    value = quat_abs(normalize_exact(value));
    const vec3 vector(value.x, value.y, value.z);
    const float vector_length = length(vector);
    if (vector_length < 1.0e-8F) return 2.0F * vector;
    const float angle = 2.0F * std::atan2(vector_length, value.w);
    return (angle / vector_length) * vector;
}

vec3 orientation_delta(quat target, quat current) {
    return shortest_log(quat_mul(
        normalize_exact(target),
        quat_inv(normalize_exact(current))));
}

float wrap_angle(float value) {
    if (!finite(value)) return value;
    value = std::fmod(value + kPi, kTwoPi);
    if (value < 0.0F) value += kTwoPi;
    return value - kPi;
}

float decompose_angle(const HingeJoint& joint, quat local_rotation) {
    const quat delta = quat_abs(normalize_exact(quat_inv_mul(
        normalize_exact(joint.rest_rotation),
        normalize_exact(local_rotation))));
    return dot(shortest_log(delta), joint.axis);
}

quat joint_rotation(const HingeJoint& joint, float angle) {
    return normalize_exact(quat_mul(
        joint.rest_rotation,
        quat_from_angle_axis(angle, joint.axis)));
}

Transform hand_world(const Pose& pose, Hand hand) {
    const WorldPose world = world_pose(pose);
    const size_t wrist = hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : static_cast<size_t>(g1_skeleton::RightWrist);
    return {
        world.positions[wrist],
        normalize_exact(world.rotations[wrist]),
    };
}

quat shortest_arc(vec3 from, vec3 to) {
    const float from_length = length(from);
    const float to_length = length(to);
    if (!(from_length > 1.0e-8F) || !(to_length > 1.0e-8F)) {
        return quat();
    }
    from = from / from_length;
    to = to / to_length;
    const float cosine = std::clamp(dot(from, to), -1.0F, 1.0F);
    if (cosine > 1.0F - 1.0e-7F) return quat();
    if (cosine < -1.0F + 1.0e-7F) {
        vec3 basis = std::abs(from.x) < std::abs(from.y)
            ? vec3(1.0F, 0.0F, 0.0F)
            : vec3(0.0F, 1.0F, 0.0F);
        if (std::abs(dot(from, basis)) > 0.9F) {
            basis = vec3(0.0F, 0.0F, 1.0F);
        }
        const vec3 axis = normalize(cross(from, basis));
        return quat_from_angle_axis(kPi, axis);
    }
    return normalize_exact(quat(
        1.0F + cosine,
        cross(from, to).x,
        cross(from, to).y,
        cross(from, to).z));
}

vec3 transported_direction(
    const ElbowPole& reference,
    const ElbowPole& current) {
    return normalize(quat_mul_vec3(
        shortest_arc(reference.axis, current.axis),
        reference.direction));
}

bool finite_pose_kinematics(const Pose& pose) {
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        if (!finite(pose.positions[bone]) ||
            !finite(pose.rotations[bone]) ||
            !(quat_length(pose.rotations[bone]) > 1.0e-12F)) {
            return false;
        }
    }
    return true;
}

bool finite_transform(const Transform& value) {
    return finite(value.position) && finite(value.rotation) &&
           quat_length(value.rotation) > 1.0e-12F;
}

bool valid_config(const PostureIKConfig& config) {
    if (!finite(config.accepted_position_m) ||
        !finite(config.accepted_orientation_radians) ||
        !finite(config.initial_damping) ||
        !finite(config.minimum_damping) ||
        !finite(config.maximum_damping) ||
        !finite(config.finite_difference_radians) ||
        !finite(config.orientation_scale_m_per_radian) ||
        !finite(config.elbow_pole_scale_m) ||
        !finite(config.maximum_step_radians) ||
        config.accepted_position_m < 0.0F ||
        config.accepted_orientation_radians < 0.0F ||
        config.initial_damping <= 0.0F ||
        config.minimum_damping <= 0.0F ||
        config.maximum_damping < config.minimum_damping ||
        config.finite_difference_radians <= 0.0F ||
        config.orientation_scale_m_per_radian < 0.0F ||
        config.elbow_pole_scale_m < 0.0F ||
        config.maximum_step_radians <= 0.0F ||
        config.maximum_iterations < 0) {
        return false;
    }
    for (size_t joint = 0U; joint < kJointCount; ++joint) {
        if (!finite(config.source_scale_m_per_radian[joint]) ||
            !finite(config.temporal_scale_m_per_radian[joint]) ||
            config.source_scale_m_per_radian[joint] < 0.0F ||
            config.temporal_scale_m_per_radian[joint] < 0.0F) {
            return false;
        }
    }
    return true;
}

bool bounded(Hand hand, const UpperBodyAngles& angles) {
    for (size_t joint = 0U; joint < kJointCount; ++joint) {
        const HingeJoint& item = metadata(hand, joint);
        if (!finite(angles[joint]) || angles[joint] < item.lower ||
            angles[joint] > item.upper) {
            return false;
        }
    }
    return true;
}

bool task_accepted(
    const Evaluation& evaluation,
    const PostureIKConfig& config) {
    return evaluation.position_error_m <= config.accepted_position_m &&
           evaluation.orientation_error_radians <=
               config.accepted_orientation_radians;
}

Evaluation evaluate(
    Hand hand,
    const Pose& source_pose,
    const UpperBodyAngles& angles,
    const UpperBodyAngles& source_angles,
    const UpperBodyAngles& temporal_seed,
    const ElbowPole& source_pole,
    Transform target,
    const PostureIKConfig& config,
    bool preserve_source_pose = false) {
    Evaluation evaluation{};
    evaluation.pose = source_pose;
    if (!preserve_source_pose) {
        apply_upper_body(evaluation.pose, hand, angles);
    }
    const Transform achieved_hand = hand_world(evaluation.pose, hand);
    const vec3 position = target.position - achieved_hand.position;
    const vec3 orientation = orientation_delta(
        target.rotation, achieved_hand.rotation);
    evaluation.position_error_m = length(position);
    evaluation.orientation_error_radians = length(orientation);
    evaluation.residual[kTaskResidualStart + 0U] = position.x;
    evaluation.residual[kTaskResidualStart + 1U] = position.y;
    evaluation.residual[kTaskResidualStart + 2U] = position.z;
    evaluation.residual[kOrientationResidualStart + 0U] =
        config.orientation_scale_m_per_radian * orientation.x;
    evaluation.residual[kOrientationResidualStart + 1U] =
        config.orientation_scale_m_per_radian * orientation.y;
    evaluation.residual[kOrientationResidualStart + 2U] =
        config.orientation_scale_m_per_radian * orientation.z;

    const ElbowPole current_pole =
        hand_elbow_pole(evaluation.pose, hand);
    evaluation.elbow_pole_degenerate =
        !source_pole.valid || !current_pole.valid;
    if (!evaluation.elbow_pole_degenerate) {
        const vec3 target_pole = transported_direction(
            source_pole, current_pole);
        const vec3 pole_residual = target_pole - current_pole.direction;
        evaluation.residual[kElbowResidualStart + 0U] =
            config.elbow_pole_scale_m * pole_residual.x;
        evaluation.residual[kElbowResidualStart + 1U] =
            config.elbow_pole_scale_m * pole_residual.y;
        evaluation.residual[kElbowResidualStart + 2U] =
            config.elbow_pole_scale_m * pole_residual.z;
        evaluation.elbow_pole_error_radians =
            transported_elbow_pole_error(source_pole, current_pole);
    }
    for (size_t joint = 0U; joint < kJointCount; ++joint) {
        evaluation.residual[kSourceResidualStart + joint] =
            config.source_scale_m_per_radian[joint] *
            wrap_angle(source_angles[joint] - angles[joint]);
        evaluation.residual[kTemporalResidualStart + joint] =
            config.temporal_scale_m_per_radian[joint] *
            wrap_angle(temporal_seed[joint] - angles[joint]);
    }
    for (const double value : evaluation.residual) {
        evaluation.objective += value * value;
    }
    return evaluation;
}

Jacobian numerical_jacobian(
    Hand hand,
    const Pose& source_pose,
    const UpperBodyAngles& angles,
    const UpperBodyAngles& source_angles,
    const UpperBodyAngles& temporal_seed,
    const ElbowPole& source_pole,
    Transform target,
    const PostureIKConfig& config,
    const Residual& current) {
    Jacobian jacobian{};
    for (size_t joint = 0U; joint < kJointCount; ++joint) {
        const HingeJoint& item = metadata(hand, joint);
        float step = config.finite_difference_radians;
        if (angles[joint] + step > item.upper) {
            step = -config.finite_difference_radians;
        }
        UpperBodyAngles perturbed = angles;
        perturbed[joint] = std::clamp(
            perturbed[joint] + step, item.lower, item.upper);
        const float actual_step = perturbed[joint] - angles[joint];
        if (std::abs(actual_step) < 1.0e-12F) continue;
        const Evaluation value = evaluate(
            hand, source_pose, perturbed, source_angles, temporal_seed,
            source_pole, target, config);
        for (size_t row = 0U; row < kResidualDimension; ++row) {
            jacobian[row][joint] =
                (value.residual[row] - current[row]) / actual_step;
        }
    }
    return jacobian;
}

bool solve_normal(
    NormalMatrix matrix,
    NormalVector right,
    NormalVector& solution) {
    for (size_t column = 0U; column < kJointCount; ++column) {
        size_t pivot = column;
        for (size_t row = column + 1U; row < kJointCount; ++row) {
            if (std::abs(matrix[row][column]) >
                std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(matrix[pivot][column]) < 1.0e-14) return false;
        if (pivot != column) {
            std::swap(matrix[pivot], matrix[column]);
            std::swap(right[pivot], right[column]);
        }
        const double diagonal = matrix[column][column];
        for (size_t entry = column; entry < kJointCount; ++entry) {
            matrix[column][entry] /= diagonal;
        }
        right[column] /= diagonal;
        for (size_t row = 0U; row < kJointCount; ++row) {
            if (row == column) continue;
            const double factor = matrix[row][column];
            if (factor == 0.0) continue;
            for (size_t entry = column; entry < kJointCount; ++entry) {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            right[row] -= factor * right[column];
        }
    }
    solution = right;
    return true;
}

bool damped_step(
    const Jacobian& jacobian,
    const Residual& residual,
    float damping,
    NormalVector& step) {
    NormalMatrix normal{};
    NormalVector right{};
    for (size_t row = 0U; row < kJointCount; ++row) {
        for (size_t column = 0U; column < kJointCount; ++column) {
            for (size_t residual_row = 0U;
                 residual_row < kResidualDimension;
                 ++residual_row) {
                normal[row][column] +=
                    jacobian[residual_row][row] *
                    jacobian[residual_row][column];
            }
        }
        normal[row][row] += static_cast<double>(damping) * damping;
        for (size_t residual_row = 0U;
             residual_row < kResidualDimension;
             ++residual_row) {
            right[row] -=
                jacobian[residual_row][row] * residual[residual_row];
        }
    }
    return solve_normal(normal, right, step);
}

PostureIKResult make_result(
    Hand hand,
    const Evaluation& evaluation,
    const UpperBodyAngles& angles,
    bool hit_joint_limit,
    int32_t iterations,
    const PostureIKConfig& config) {
    PostureIKResult result{};
    result.accepted =
        task_accepted(evaluation, config) && bounded(hand, angles);
    result.reason = result.accepted
        ? Reason::None
        : (hit_joint_limit ? Reason::JointLimit : Reason::CorrectionLimit);
    result.joint_limit_saturated = hit_joint_limit;
    result.elbow_pole_degenerate = evaluation.elbow_pole_degenerate;
    result.position_error_m = evaluation.position_error_m;
    result.orientation_error_radians = evaluation.orientation_error_radians;
    result.elbow_pole_error_radians =
        evaluation.elbow_pole_error_radians;
    result.objective = evaluation.objective;
    result.iterations = iterations;
    result.joint_angles = angles;
    for (size_t joint = 0U; joint < kJointCount; ++joint) {
        const HingeJoint& item = metadata(hand, joint);
        result.at_limit[joint] =
            std::abs(angles[joint] - item.lower) <= kLimitTolerance ||
            std::abs(angles[joint] - item.upper) <= kLimitTolerance;
    }
    return result;
}

PostureIKResult invalid_result(
    Pose& pose,
    const Pose& source_pose,
    const UpperBodyAngles& source_angles) {
    pose = source_pose;
    PostureIKResult result{};
    result.reason = Reason::CorrectionLimit;
    result.position_error_m = std::numeric_limits<float>::max();
    result.orientation_error_radians = std::numeric_limits<float>::max();
    result.elbow_pole_error_radians = std::numeric_limits<float>::max();
    result.objective = std::numeric_limits<double>::max();
    result.joint_angles = source_angles;
    return result;
}

}  // namespace

UpperBodyAngles decompose_upper_body(const Pose& pose, Hand hand) {
    UpperBodyAngles angles{};
    for (size_t joint = 0U; joint < kJointCount; ++joint) {
        const HingeJoint& item = metadata(hand, joint);
        angles[joint] = decompose_angle(
            item, pose.rotations[static_cast<size_t>(item.bone)]);
    }
    return angles;
}

void apply_upper_body(
    Pose& pose,
    Hand hand,
    const UpperBodyAngles& angles) {
    for (size_t joint = 0U; joint < kJointCount; ++joint) {
        const HingeJoint& item = metadata(hand, joint);
        pose.rotations[static_cast<size_t>(item.bone)] =
            joint_rotation(item, angles[joint]);
    }
}

ElbowPole hand_elbow_pole(const Pose& pose, Hand hand) {
    const WorldPose world = world_pose(pose);
    const size_t shoulder_bone = hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftShoulderPitch)
        : static_cast<size_t>(g1_skeleton::RightShoulderPitch);
    const size_t elbow_bone = hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftElbow)
        : static_cast<size_t>(g1_skeleton::RightElbow);
    const size_t wrist_bone = hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : static_cast<size_t>(g1_skeleton::RightWrist);
    const vec3 shoulder = world.positions[shoulder_bone];
    const vec3 elbow = world.positions[elbow_bone];
    const vec3 wrist = world.positions[wrist_bone];
    const vec3 axis_vector = wrist - shoulder;
    const float axis_length = length(axis_vector);
    if (!(axis_length > 1.0e-6F) || !finite(axis_length)) return {};
    const vec3 axis = axis_vector / axis_length;
    const vec3 elbow_vector = elbow - shoulder;
    const vec3 perpendicular =
        elbow_vector - dot(elbow_vector, axis) * axis;
    const float pole_length = length(perpendicular);
    if (!(pole_length > 1.0e-5F) || !finite(pole_length)) return {};
    return {true, shoulder, axis, perpendicular / pole_length};
}

float transported_elbow_pole_error(
    const ElbowPole& reference,
    const ElbowPole& current) {
    if (!reference.valid || !current.valid) return 0.0F;
    const vec3 transported = transported_direction(reference, current);
    return std::acos(std::clamp(
        dot(transported, current.direction), -1.0F, 1.0F));
}

PostureIKResult solve_hand_posture_ik(
    Pose& pose,
    Hand hand,
    Transform target_hand_world,
    const Pose& source_pose,
    const UpperBodyAngles& temporal_seed,
    const PostureIKConfig& config) {
    const UpperBodyAngles source_angles =
        decompose_upper_body(source_pose, hand);
    if (!finite_pose_kinematics(source_pose) ||
        !finite_transform(target_hand_world) || !valid_config(config)) {
        return invalid_result(pose, source_pose, source_angles);
    }
    for (const float value : temporal_seed) {
        if (!finite(value)) {
            return invalid_result(pose, source_pose, source_angles);
        }
    }

    const ElbowPole source_pole = hand_elbow_pole(source_pose, hand);
    bool seed_is_source = true;
    for (size_t joint = 0U; joint < kJointCount; ++joint) {
        seed_is_source = seed_is_source &&
            std::abs(wrap_angle(
                temporal_seed[joint] - source_angles[joint])) <= 1.0e-7F;
    }
    const Evaluation exact_source = evaluate(
        hand, source_pose, source_angles, source_angles, temporal_seed,
        source_pole, target_hand_world, config, true);
    if (seed_is_source && bounded(hand, source_angles) &&
        task_accepted(exact_source, config)) {
        pose = source_pose;
        return make_result(
            hand, exact_source, source_angles, false, 0, config);
    }

    bool hit_joint_limit = false;
    UpperBodyAngles angles = temporal_seed;
    for (size_t joint = 0U; joint < kJointCount; ++joint) {
        const HingeJoint& item = metadata(hand, joint);
        const float value = std::clamp(
            angles[joint], item.lower, item.upper);
        hit_joint_limit = hit_joint_limit || value != angles[joint];
        angles[joint] = value;
    }

    Evaluation working = evaluate(
        hand, source_pose, angles, source_angles, temporal_seed, source_pole,
        target_hand_world, config);
    Evaluation best = working;
    UpperBodyAngles best_angles = angles;
    if (bounded(hand, source_angles) && finite(exact_source.objective) &&
        exact_source.objective <= best.objective) {
        best = exact_source;
        best_angles = source_angles;
    }
    float damping = std::clamp(
        config.initial_damping,
        config.minimum_damping,
        config.maximum_damping);
    int32_t iterations = 0;

    for (; iterations < config.maximum_iterations; ++iterations) {
        const Jacobian jacobian = numerical_jacobian(
            hand, source_pose, angles, source_angles, temporal_seed, source_pole,
            target_hand_world, config, working.residual);
        NormalVector raw_step{};
        if (!damped_step(
                jacobian, working.residual, damping, raw_step)) {
            damping = std::min(
                config.maximum_damping, damping * 4.0F);
            if (task_accepted(working, config)) break;
            continue;
        }

        UpperBodyAngles trial_angles = angles;
        for (size_t joint = 0U; joint < kJointCount; ++joint) {
            const HingeJoint& item = metadata(hand, joint);
            const float step = std::clamp(
                static_cast<float>(raw_step[joint]),
                -config.maximum_step_radians,
                config.maximum_step_radians);
            const float proposed = angles[joint] + step;
            trial_angles[joint] = std::clamp(
                proposed, item.lower, item.upper);
            hit_joint_limit =
                hit_joint_limit || trial_angles[joint] != proposed;
        }
        const Evaluation trial = evaluate(
            hand, source_pose, trial_angles, source_angles, temporal_seed,
            source_pole, target_hand_world, config);
        if (finite(trial.objective) &&
            trial.objective + 1.0e-15 < working.objective) {
            angles = trial_angles;
            working = trial;
            damping = std::max(
                config.minimum_damping, damping * 0.5F);
            if (working.objective < best.objective) {
                best = working;
                best_angles = angles;
            }
        } else {
            damping = std::min(
                config.maximum_damping, damping * 4.0F);
            if (task_accepted(working, config)) {
                ++iterations;
                break;
            }
        }
    }

    pose = best.pose;
    return make_result(
        hand, best, best_angles, hit_joint_limit, iterations, config);
}

LeftUpperBodyAngles decompose_left_upper_body(const Pose& pose) {
    return decompose_upper_body(pose, Hand::Left);
}

void apply_left_upper_body(
    Pose& pose,
    const LeftUpperBodyAngles& angles) {
    apply_upper_body(pose, Hand::Left, angles);
}

ElbowPole left_elbow_pole(const Pose& pose) {
    return hand_elbow_pole(pose, Hand::Left);
}

PostureIKResult solve_left_hand_posture_ik(
    Pose& pose,
    Transform target_hand_world,
    const Pose& source_pose,
    const LeftUpperBodyAngles& temporal_seed,
    const PostureIKConfig& config) {
    return solve_hand_posture_ik(
        pose,
        Hand::Left,
        target_hand_world,
        source_pose,
        temporal_seed,
        config);
}

}  // namespace interaction
