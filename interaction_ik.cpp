#include "interaction_ik.h"

#include "g1_arm_joint_metadata.h"
#include "interaction_rotation_gate.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace interaction {
namespace {

constexpr size_t kJointCount = 7U;
constexpr size_t kResidualDimension = 6U;
constexpr size_t kRootBone = static_cast<size_t>(g1_skeleton::Simulation);

using ArmMetadata = std::array<HingeJoint, kJointCount>;
using JointAngles = std::array<float, kJointCount>;
using Residual = std::array<double, kResidualDimension>;
using Jacobian =
    std::array<std::array<double, kJointCount>, kResidualDimension>;
using Matrix6 =
    std::array<std::array<double, kResidualDimension>, kResidualDimension>;

struct Evaluation {
    vec3 position_residual{};
    vec3 orientation_residual{};
    rotation_gate::Measure orientation_measure{};
    float position_error = 0.0F;
    float orientation_error = 0.0F;
    double score = 0.0;
};

const ArmMetadata& metadata_for(Hand hand) {
    return hand == Hand::Left ? kLeftArm : kRightArm;
}

quat normalize_exact(quat value) {
    const float norm = quat_length(value);
    if (!(norm > 1.0e-12F) || !rotation_gate::finite_bits(norm)) {
        return quat();
    }
    return value / norm;
}

vec3 shortest_log(quat value) {
    value = quat_abs(normalize_exact(value));
    const vec3 vector(value.x, value.y, value.z);
    const float vector_length = length(vector);
    if (vector_length < 1.0e-8F) {
        return 2.0F * vector;
    }
    const float angle = 2.0F * std::atan2(vector_length, value.w);
    return (angle / vector_length) * vector;
}

vec3 orientation_delta(quat target, quat current) {
    return shortest_log(quat_mul(
        normalize_exact(target), quat_inv(normalize_exact(current))));
}

Transform hand_world(const Pose& pose, const ArmMetadata& metadata) {
    const WorldPose world = world_pose(pose);
    const size_t bone = static_cast<size_t>(metadata.back().bone);
    return {world.positions[bone], normalize_exact(world.rotations[bone])};
}

rotation_gate::Rotation hand_rotation_evidence(
    const Pose& pose,
    const ArmMetadata& metadata,
    const rotation_gate::Rotation& root_rotation_evidence) {
    return world_rotation_evidence(
        pose,
        static_cast<size_t>(metadata.back().bone),
        root_rotation_evidence);
}

float decompose_angle(const HingeJoint& joint, quat local_rotation) {
    const quat delta = quat_abs(normalize_exact(quat_inv_mul(
        normalize_exact(joint.rest_rotation),
        normalize_exact(local_rotation))));
    return dot(shortest_log(delta), joint.axis);
}

JointAngles decompose_angles(
    const Pose& pose,
    const ArmMetadata& metadata) {
    JointAngles angles{};
    for (size_t joint = 0; joint < metadata.size(); ++joint) {
        angles[joint] = decompose_angle(
            metadata[joint],
            pose.rotations[static_cast<size_t>(metadata[joint].bone)]);
    }
    return angles;
}

void apply_angles(
    Pose& pose,
    const ArmMetadata& metadata,
    const JointAngles& angles) {
    for (size_t joint = 0; joint < metadata.size(); ++joint) {
        pose.rotations[static_cast<size_t>(metadata[joint].bone)] =
            normalize_exact(quat_mul(
                metadata[joint].rest_rotation,
                quat_from_angle_axis(angles[joint], metadata[joint].axis)));
    }
}

Evaluation evaluate(
    const Pose& pose,
    const ArmMetadata& metadata,
    Transform target,
    const rotation_gate::Rotation& target_rotation_evidence,
    const rotation_gate::Rotation& root_rotation_evidence,
    float orientation_scale) {
    const Transform current = hand_world(pose, metadata);
    Evaluation evaluation{};
    evaluation.position_residual = target.position - current.position;
    evaluation.orientation_residual = orientation_delta(
        target.rotation, current.rotation);
    evaluation.orientation_measure = rotation_gate::measure(
        target_rotation_evidence,
        hand_rotation_evidence(
            pose, metadata, root_rotation_evidence));
    evaluation.position_error = length(evaluation.position_residual);
    evaluation.orientation_error = rotation_gate::radians(
        evaluation.orientation_measure);
    const double px = evaluation.position_residual.x;
    const double py = evaluation.position_residual.y;
    const double pz = evaluation.position_residual.z;
    const double ox = orientation_scale * evaluation.orientation_residual.x;
    const double oy = orientation_scale * evaluation.orientation_residual.y;
    const double oz = orientation_scale * evaluation.orientation_residual.z;
    evaluation.score =
        px * px + py * py + pz * pz + ox * ox + oy * oy + oz * oz;
    return evaluation;
}

Residual residual(const Evaluation& evaluation, float orientation_scale) {
    return {
        evaluation.position_residual.x,
        evaluation.position_residual.y,
        evaluation.position_residual.z,
        orientation_scale * evaluation.orientation_residual.x,
        orientation_scale * evaluation.orientation_residual.y,
        orientation_scale * evaluation.orientation_residual.z,
    };
}

bool finite_transform(const Transform& value) {
    return rotation_gate::finite_bits(value.position.x) &&
           rotation_gate::finite_bits(value.position.y) &&
           rotation_gate::finite_bits(value.position.z) &&
           rotation_gate::finite_bits(value.rotation) &&
           quat_length(value.rotation) > 1.0e-12F;
}

bool valid_config(const IKConfig& config) {
    return rotation_gate::finite_bits(config.maximum_request_position_m) &&
           rotation_gate::finite_bits(
               config.maximum_request_orientation_radians) &&
           rotation_gate::finite_bits(config.accepted_position_m) &&
           rotation_gate::finite_bits(
               config.accepted_orientation_radians) &&
           rotation_gate::finite_bits(config.damping) &&
           rotation_gate::finite_bits(config.finite_difference_radians) &&
           rotation_gate::finite_bits(
               config.orientation_scale_m_per_radian) &&
           rotation_gate::finite_bits(config.maximum_step_radians) &&
           config.maximum_request_position_m >= 0.0F &&
           config.maximum_request_orientation_radians >= 0.0F &&
           config.accepted_position_m >= 0.0F &&
           config.accepted_orientation_radians >= 0.0F &&
           config.damping > 0.0F &&
           config.finite_difference_radians > 0.0F &&
           config.orientation_scale_m_per_radian > 0.0F &&
           config.maximum_step_radians > 0.0F &&
           config.maximum_iterations >= 0;
}

bool accepted(const Evaluation& evaluation, const IKConfig& config) {
    return evaluation.position_error <= config.accepted_position_m &&
           rotation_gate::within(
               evaluation.orientation_measure,
               config.accepted_orientation_radians);
}

Jacobian numerical_jacobian(
    const Pose& pose,
    const ArmMetadata& metadata,
    const JointAngles& angles,
    Transform current_hand,
    float finite_difference,
    float orientation_scale) {
    Jacobian jacobian{};
    for (size_t joint = 0; joint < metadata.size(); ++joint) {
        float step = finite_difference;
        if (angles[joint] + step > metadata[joint].upper) {
            step = -finite_difference;
        }
        JointAngles perturbed_angles = angles;
        perturbed_angles[joint] += step;
        Pose perturbed_pose = pose;
        apply_angles(perturbed_pose, metadata, perturbed_angles);
        const Transform perturbed_hand = hand_world(perturbed_pose, metadata);
        const vec3 position_column =
            (perturbed_hand.position - current_hand.position) / step;
        const vec3 orientation_column =
            orientation_delta(
                perturbed_hand.rotation, current_hand.rotation) / step;
        jacobian[0][joint] = position_column.x;
        jacobian[1][joint] = position_column.y;
        jacobian[2][joint] = position_column.z;
        jacobian[3][joint] =
            orientation_scale * orientation_column.x;
        jacobian[4][joint] =
            orientation_scale * orientation_column.y;
        jacobian[5][joint] =
            orientation_scale * orientation_column.z;
    }
    return jacobian;
}

bool solve_linear(Matrix6 matrix, Residual right, Residual& solution) {
    for (size_t column = 0; column < kResidualDimension; ++column) {
        size_t pivot = column;
        for (size_t row = column + 1U; row < kResidualDimension; ++row) {
            if (std::abs(matrix[row][column]) >
                std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(matrix[pivot][column]) < 1.0e-12) {
            return false;
        }
        if (pivot != column) {
            std::swap(matrix[pivot], matrix[column]);
            std::swap(right[pivot], right[column]);
        }
        const double diagonal = matrix[column][column];
        for (size_t entry = column; entry < kResidualDimension; ++entry) {
            matrix[column][entry] /= diagonal;
        }
        right[column] /= diagonal;
        for (size_t row = 0; row < kResidualDimension; ++row) {
            if (row == column) {
                continue;
            }
            const double factor = matrix[row][column];
            if (factor == 0.0) {
                continue;
            }
            for (size_t entry = column; entry < kResidualDimension; ++entry) {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            right[row] -= factor * right[column];
        }
    }
    solution = right;
    return true;
}

JointAngles damped_step(
    const Jacobian& jacobian,
    const Residual& target_residual,
    float damping) {
    Matrix6 normal{};
    for (size_t row = 0; row < kResidualDimension; ++row) {
        for (size_t column = 0; column < kResidualDimension; ++column) {
            for (size_t joint = 0; joint < kJointCount; ++joint) {
                normal[row][column] +=
                    jacobian[row][joint] * jacobian[column][joint];
            }
        }
        normal[row][row] += static_cast<double>(damping) * damping;
    }
    Residual intermediate{};
    JointAngles step{};
    if (!solve_linear(normal, target_residual, intermediate)) {
        return step;
    }
    for (size_t joint = 0; joint < kJointCount; ++joint) {
        double value = 0.0;
        for (size_t row = 0; row < kResidualDimension; ++row) {
            value += jacobian[row][joint] * intermediate[row];
        }
        step[joint] = static_cast<float>(value);
    }
    return step;
}

IKResult make_result(
    bool is_accepted,
    Reason reason,
    const Evaluation& evaluation,
    const JointAngles& angles,
    float orientation_boundary,
    bool classify_orientation) {
    IKResult result{};
    result.accepted = is_accepted;
    result.reason = reason;
    result.position_error_m = evaluation.position_error;
    result.orientation_error_radians = classify_orientation
        ? rotation_gate::classified_radians(
              evaluation.orientation_measure, orientation_boundary)
        : evaluation.orientation_error;
    result.joint_angles = angles;
    return result;
}

}  // namespace

IKResult solve_hand_ik(
    Pose& pose,
    Hand hand,
    Transform target_hand_world,
    const IKConfig& config) {
    return solve_hand_ik_with_rotation_evidence(
        pose,
        hand,
        target_hand_world,
        rotation_gate::from_quat(target_hand_world.rotation),
        config);
}

IKResult solve_hand_ik_with_rotation_evidence(
    Pose& pose,
    Hand hand,
    Transform target_hand_world,
    const rotation_gate::Rotation& target_rotation_evidence,
    const IKConfig& config) {
    return solve_hand_ik_with_rotation_evidence(
        pose,
        hand,
        target_hand_world,
        target_rotation_evidence,
        rotation_gate::from_quat(pose.rotations[kRootBone]),
        config);
}

IKResult solve_hand_ik_with_rotation_evidence(
    Pose& pose,
    Hand hand,
    Transform target_hand_world,
    const rotation_gate::Rotation& target_rotation_evidence,
    const rotation_gate::Rotation& root_rotation_evidence,
    const IKConfig& config) {
    const ArmMetadata& metadata = metadata_for(hand);
    const JointAngles requested_angles = decompose_angles(pose, metadata);
    const rotation_gate::Measure target_validity = rotation_gate::measure(
        target_rotation_evidence, rotation_gate::Rotation{});
    if (!finite_transform(target_hand_world) || !target_validity.valid ||
        !valid_config(config)) {
        const Evaluation invalid = evaluate(
            pose,
            metadata,
            target_hand_world,
            target_rotation_evidence,
            root_rotation_evidence,
            config.orientation_scale_m_per_radian);
        return make_result(
            false,
            Reason::CorrectionLimit,
            invalid,
            requested_angles,
            0.0F,
            false);
    }
    const Evaluation requested = evaluate(
        pose,
        metadata,
        target_hand_world,
        target_rotation_evidence,
        root_rotation_evidence,
        config.orientation_scale_m_per_radian);
    if (!rotation_gate::finite_bits(requested.position_error) ||
        !requested.orientation_measure.valid ||
        requested.position_error > config.maximum_request_position_m ||
        !rotation_gate::within(
            requested.orientation_measure,
            config.maximum_request_orientation_radians)) {
        return make_result(
            false,
            Reason::CorrectionLimit,
            requested,
            requested_angles,
            config.maximum_request_orientation_radians,
            true);
    }
    bool hit_joint_limit = false;
    bool requested_angles_bounded = true;
    JointAngles angles = requested_angles;
    for (size_t joint = 0; joint < metadata.size(); ++joint) {
        const float bounded = std::clamp(
            angles[joint], metadata[joint].lower, metadata[joint].upper);
        if (bounded != angles[joint]) {
            requested_angles_bounded = false;
            hit_joint_limit = true;
            angles[joint] = bounded;
        }
    }
    if (requested_angles_bounded && accepted(requested, config)) {
        return make_result(
            true,
            Reason::None,
            requested,
            requested_angles,
            config.accepted_orientation_radians,
            true);
    }

    Pose working_pose = pose;
    apply_angles(working_pose, metadata, angles);
    Evaluation working = evaluate(
        working_pose,
        metadata,
        target_hand_world,
        target_rotation_evidence,
        root_rotation_evidence,
        config.orientation_scale_m_per_radian);
    Pose best_pose = working_pose;
    JointAngles best_angles = angles;
    Evaluation best = working;

    for (int32_t iteration = 0;
         iteration < config.maximum_iterations && !accepted(best, config);
         ++iteration) {
        const Transform current_hand = hand_world(working_pose, metadata);
        const Jacobian jacobian = numerical_jacobian(
            working_pose, metadata, angles, current_hand,
            config.finite_difference_radians,
            config.orientation_scale_m_per_radian);
        JointAngles step = damped_step(
            jacobian,
            residual(working, config.orientation_scale_m_per_radian),
            config.damping);
        JointAngles trial_angles = angles;
        for (size_t joint = 0; joint < metadata.size(); ++joint) {
            step[joint] = std::clamp(
                step[joint],
                -config.maximum_step_radians,
                config.maximum_step_radians);
            const float proposed = angles[joint] + step[joint];
            trial_angles[joint] = std::clamp(
                proposed, metadata[joint].lower, metadata[joint].upper);
            if (trial_angles[joint] != proposed) {
                hit_joint_limit = true;
            }
        }
        Pose trial_pose = working_pose;
        apply_angles(trial_pose, metadata, trial_angles);
        const Evaluation trial = evaluate(
            trial_pose,
            metadata,
            target_hand_world,
            target_rotation_evidence,
            root_rotation_evidence,
            config.orientation_scale_m_per_radian);
        if (rotation_gate::finite_bits(trial.score) &&
            trial.score < best.score) {
            best = trial;
            best_angles = trial_angles;
            best_pose = trial_pose;
        }
        working = trial;
        working_pose = trial_pose;
        angles = trial_angles;
    }

    pose = best_pose;
    const bool is_accepted = accepted(best, config);
    return make_result(
        is_accepted,
        is_accepted
            ? Reason::None
            : (hit_joint_limit ? Reason::JointLimit
                               : Reason::CorrectionLimit),
        best,
        best_angles,
        config.accepted_orientation_radians,
        true);
}

}  // namespace interaction
