#pragma once

#include <cstdlib>

#include "array.h"
#include "g1_kinematic_contract.h"
#include "quat.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <string>

static constexpr int SonicG1JointCount = 29;
static constexpr float SonicG1ProjectionMaximumResidual = 0.001f;

enum sonic_joint_projection_failure {
    SonicJointProjectionValid = 0,
    SonicJointProjectionShape,
    SonicJointProjectionContract,
    SonicJointProjectionInput,
    SonicJointProjectionSingular,
    SonicJointProjectionResidual,
    SonicJointProjectionLimit,
    SonicJointProjectionVelocity,
};

struct sonic_joint_projection_diagnostic
{
    sonic_joint_projection_failure failure = SonicJointProjectionValid;
    int row = -1;
    float position = 0.0f;
    float lower = 0.0f;
    float upper = 0.0f;
};

struct sonic_joint_contract_entry
{
    int source_index = -1;
    int source_bone = -1;
    int source_parent = -1;
    int target_index = -1;
    vec3 axis_holden;
    quat static_local_holden;
    float sign = 0.0f;
    float zero_offset = 0.0f;
    float lower = 0.0f;
    float upper = 0.0f;
    std::string source_joint;
    std::string target_joint;
};

struct sonic_projected_pose
{
    float source_joint_position[SonicG1JointCount] = {};
    float source_joint_velocity[SonicG1JointCount] = {};
    float off_axis_residual[SonicG1JointCount] = {};
    vec3 physical_pelvis_position_holden;
    quat physical_pelvis_orientation_holden;
};

static inline bool sonic_projection_error(
    char* output, int capacity, const char* format, ...)
{
    if (output != NULL && capacity > 0) {
        va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(
            output, static_cast<std::size_t>(capacity), format, arguments);
        va_end(arguments);
    }
    return false;
}

static inline bool sonic_projection_diagnostic_error(
    sonic_joint_projection_diagnostic& diagnostic,
    sonic_joint_projection_failure failure,
    int row,
    float position,
    float lower,
    float upper,
    char* output,
    int capacity,
    const char* format,
    ...)
{
    sonic_joint_projection_diagnostic candidate;
    candidate.failure = failure;
    candidate.row = row;
    candidate.position = position;
    candidate.lower = lower;
    candidate.upper = upper;
    if (output != NULL && capacity > 0) {
        va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(
            output, static_cast<std::size_t>(capacity), format, arguments);
        va_end(arguments);
    }
    diagnostic = candidate;
    return false;
}

static inline bool sonic_projection_vec3_finite(const vec3 value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

static inline bool sonic_projection_quat_finite(const quat value)
{
    return std::isfinite(value.w) && std::isfinite(value.x) &&
           std::isfinite(value.y) && std::isfinite(value.z);
}

static inline float sonic_projection_quat_norm_squared(const quat value)
{
    return value.w * value.w + value.x * value.x +
           value.y * value.y + value.z * value.z;
}

static inline bool sonic_projection_quat_is_unit(const quat value)
{
    return sonic_projection_quat_finite(value) &&
           std::fabs(sonic_projection_quat_norm_squared(value) - 1.0f) <=
               2.0e-4f;
}

static inline quat sonic_projection_quat_normalized(const quat value)
{
    const float inverse_length =
        1.0f / std::sqrt(sonic_projection_quat_norm_squared(value));
    return value * inverse_length;
}

static inline quat sonic_projection_quat_canonical(const quat value)
{
    const quat normalized = sonic_projection_quat_normalized(value);
    if (normalized.w < 0.0f ||
        (normalized.w == 0.0f && normalized.x < 0.0f) ||
        (normalized.w == 0.0f && normalized.x == 0.0f &&
         normalized.y < 0.0f) ||
        (normalized.w == 0.0f && normalized.x == 0.0f &&
         normalized.y == 0.0f && normalized.z < 0.0f)) {
        return -normalized;
    }
    return normalized;
}

static inline float sonic_projection_rotation_angle(const quat value)
{
    const quat normalized = sonic_projection_quat_normalized(value);
    const float vector_length = std::sqrt(
        normalized.x * normalized.x +
        normalized.y * normalized.y +
        normalized.z * normalized.z);
    return 2.0f * std::atan2(vector_length, std::fabs(normalized.w));
}

static inline bool sonic_projection_contract_valid(
    const sonic_joint_contract_entry (&contract)[SonicG1JointCount],
    char* error,
    int error_capacity)
{
    static const int expected_parents[G1_BoneCount] = {
        -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
    };
    bool source_seen[SonicG1JointCount] = {};
    bool target_seen[SonicG1JointCount] = {};
    for (int row = 0; row < SonicG1JointCount; ++row) {
        const sonic_joint_contract_entry& entry = contract[row];
        if (entry.source_index != row || entry.source_index < 0 ||
            entry.source_index >= SonicG1JointCount ||
            source_seen[entry.source_index]) {
            return sonic_projection_error(
                error, error_capacity,
                "joint contract source index is invalid at row %d", row);
        }
        source_seen[entry.source_index] = true;
        if (entry.target_index < 0 ||
            entry.target_index >= SonicG1JointCount ||
            target_seen[entry.target_index]) {
            return sonic_projection_error(
                error, error_capacity,
                "joint contract target index is invalid at row %d", row);
        }
        target_seen[entry.target_index] = true;
        if (entry.source_bone != row + 2 ||
            entry.source_bone < 0 || entry.source_bone >= G1_BoneCount ||
            entry.source_parent != expected_parents[entry.source_bone]) {
            return sonic_projection_error(
                error, error_capacity,
                "joint contract source bone hierarchy is invalid at row %d",
                row);
        }
        if (entry.source_joint.empty() || entry.target_joint.empty() ||
            entry.source_joint != entry.target_joint) {
            return sonic_projection_error(
                error, error_capacity,
                "joint contract name mapping is invalid at row %d", row);
        }
        if (!sonic_projection_vec3_finite(entry.axis_holden) ||
            !sonic_projection_quat_is_unit(entry.static_local_holden) ||
            !std::isfinite(entry.sign) ||
            !std::isfinite(entry.zero_offset) ||
            !std::isfinite(entry.lower) || !std::isfinite(entry.upper)) {
            return sonic_projection_error(
                error, error_capacity,
                "joint contract contains non-finite data at row %d", row);
        }
        const float axis_length_squared = dot(
            entry.axis_holden, entry.axis_holden);
        if (std::fabs(axis_length_squared - 1.0f) > 2.0e-4f ||
            (entry.sign != -1.0f && entry.sign != 1.0f) ||
            entry.lower > entry.zero_offset ||
            entry.zero_offset > entry.upper ||
            entry.lower >= entry.upper) {
            return sonic_projection_error(
                error, error_capacity,
                "joint contract scalar range is invalid at row %d", row);
        }
    }
    return true;
}

static inline bool sonic_project_joint_state(
    float (&positions)[SonicG1JointCount],
    float (&velocities)[SonicG1JointCount],
    float (&residuals)[SonicG1JointCount],
    sonic_joint_projection_diagnostic& diagnostic,
    const sonic_joint_contract_entry (&contract)[SonicG1JointCount],
    slice1d<quat> local_rotations,
    slice1d<vec3> local_angular_velocities,
    char* error,
    int error_capacity)
{
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (local_rotations.size != G1_BoneCount ||
        local_angular_velocities.size != G1_BoneCount ||
        local_rotations.data == NULL ||
        local_angular_velocities.data == NULL) {
        return sonic_projection_diagnostic_error(
            diagnostic,
            SonicJointProjectionShape,
            -1,
            0.0f,
            0.0f,
            0.0f,
            error,
            error_capacity,
            "projection input shape must contain exactly %d G1 bones",
            G1_BoneCount);
    }
    if (!sonic_projection_contract_valid(contract, error, error_capacity)) {
        sonic_joint_projection_diagnostic candidate;
        candidate.failure = SonicJointProjectionContract;
        diagnostic = candidate;
        return false;
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        const int row = bone >= 2 ? bone - 2 : -1;
        if (!sonic_projection_quat_is_unit(local_rotations(bone))) {
            return sonic_projection_diagnostic_error(
                diagnostic,
                SonicJointProjectionInput,
                row,
                0.0f,
                0.0f,
                0.0f,
                error,
                error_capacity,
                "projection input contains non-finite or invalid data at bone %d",
                bone);
        }
        if (!sonic_projection_vec3_finite(local_angular_velocities(bone))) {
            return sonic_projection_diagnostic_error(
                diagnostic,
                SonicJointProjectionVelocity,
                row,
                0.0f,
                0.0f,
                0.0f,
                error,
                error_capacity,
                "projection input contains non-finite or invalid data at bone %d",
                bone);
        }
    }

    float candidate_positions[SonicG1JointCount] = {};
    float candidate_velocities[SonicG1JointCount] = {};
    float candidate_residuals[SonicG1JointCount] = {};
    for (int row = 0; row < SonicG1JointCount; ++row) {
        const sonic_joint_contract_entry& entry = contract[row];
        const quat static_local =
            sonic_projection_quat_canonical(entry.static_local_holden);
        const quat local =
            sonic_projection_quat_canonical(local_rotations(entry.source_bone));
        const quat delta = sonic_projection_quat_canonical(
            quat_mul(quat_inv(static_local), local));
        const float projected_vector =
            delta.x * entry.axis_holden.x +
            delta.y * entry.axis_holden.y +
            delta.z * entry.axis_holden.z;
        const quat twist_unnormalized(
            delta.w,
            entry.axis_holden.x * projected_vector,
            entry.axis_holden.y * projected_vector,
            entry.axis_holden.z * projected_vector);
        const float twist_norm_squared =
            sonic_projection_quat_norm_squared(twist_unnormalized);
        if (!std::isfinite(twist_norm_squared) ||
            twist_norm_squared < 1.0e-12f) {
            return sonic_projection_diagnostic_error(
                diagnostic,
                SonicJointProjectionSingular,
                row,
                0.0f,
                0.0f,
                0.0f,
                error,
                error_capacity,
                "joint %s has a singular signed twist",
                entry.source_joint.c_str());
        }
        const quat twist =
            sonic_projection_quat_canonical(twist_unnormalized);
        const float signed_twist = 2.0f * std::atan2(
            twist.x * entry.axis_holden.x +
                twist.y * entry.axis_holden.y +
                twist.z * entry.axis_holden.z,
            twist.w);
        const quat reconstructed = sonic_projection_quat_canonical(
            quat_mul(static_local, twist));
        const quat off_axis_difference = quat_mul(quat_inv(twist), delta);
        const float off_axis_residual =
            sonic_projection_rotation_angle(off_axis_difference);
        if (!std::isfinite(off_axis_residual) ||
            off_axis_residual > SonicG1ProjectionMaximumResidual) {
            return sonic_projection_diagnostic_error(
                diagnostic,
                SonicJointProjectionResidual,
                row,
                0.0f,
                0.0f,
                0.0f,
                error,
                error_capacity,
                "joint %s off-axis residual %.9g exceeds %.9g rad",
                entry.source_joint.c_str(),
                static_cast<double>(off_axis_residual),
                static_cast<double>(SonicG1ProjectionMaximumResidual));
        }
        const quat reconstruction_difference =
            quat_mul(quat_inv(reconstructed), local);
        const float reconstruction_error =
            sonic_projection_rotation_angle(reconstruction_difference);
        if (!std::isfinite(reconstruction_error) ||
            reconstruction_error > SonicG1ProjectionMaximumResidual) {
            return sonic_projection_diagnostic_error(
                diagnostic,
                SonicJointProjectionResidual,
                row,
                0.0f,
                0.0f,
                0.0f,
                error,
                error_capacity,
                "joint %s reconstructed local rotation error %.9g exceeds "
                "%.9g rad",
                entry.source_joint.c_str(),
                static_cast<double>(reconstruction_error),
                static_cast<double>(SonicG1ProjectionMaximumResidual));
        }

        const float position =
            entry.sign * signed_twist + entry.zero_offset;
        if (!std::isfinite(position) || position < entry.lower ||
            position > entry.upper) {
            return sonic_projection_diagnostic_error(
                diagnostic,
                SonicJointProjectionLimit,
                row,
                position,
                entry.lower,
                entry.upper,
                error,
                error_capacity,
                "joint %s position %.9g is outside range [%.9g, %.9g]",
                entry.source_joint.c_str(),
                static_cast<double>(position),
                static_cast<double>(entry.lower),
                static_cast<double>(entry.upper));
        }
        const vec3 axis_in_parent =
            quat_mul_vec3(static_local, entry.axis_holden);
        const float velocity = entry.sign * dot(
            local_angular_velocities(entry.source_bone), axis_in_parent);
        if (!std::isfinite(velocity)) {
            return sonic_projection_diagnostic_error(
                diagnostic,
                SonicJointProjectionVelocity,
                row,
                0.0f,
                0.0f,
                0.0f,
                error,
                error_capacity,
                "joint %s velocity is non-finite",
                entry.source_joint.c_str());
        }
        candidate_positions[entry.source_index] = position;
        candidate_velocities[entry.source_index] = velocity;
        candidate_residuals[entry.source_index] = off_axis_residual;
    }

    for (int row = 0; row < SonicG1JointCount; ++row) {
        positions[row] = candidate_positions[row];
        velocities[row] = candidate_velocities[row];
        residuals[row] = candidate_residuals[row];
    }
    diagnostic = sonic_joint_projection_diagnostic();
    return true;
}

static inline bool sonic_project_pose(
    sonic_projected_pose& out,
    const sonic_joint_contract_entry (&contract)[SonicG1JointCount],
    slice1d<quat> local_rotations,
    slice1d<vec3> local_angular_velocities,
    slice1d<vec3> global_positions,
    slice1d<quat> global_rotations,
    char* error,
    int error_capacity)
{
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (local_rotations.size != G1_BoneCount ||
        local_angular_velocities.size != G1_BoneCount ||
        global_positions.size != G1_BoneCount ||
        global_rotations.size != G1_BoneCount ||
        local_rotations.data == NULL ||
        local_angular_velocities.data == NULL ||
        global_positions.data == NULL || global_rotations.data == NULL) {
        return sonic_projection_error(
            error, error_capacity,
            "projection input shape must contain exactly %d G1 bones",
            G1_BoneCount);
    }
    if (!sonic_projection_contract_valid(contract, error, error_capacity)) {
        return false;
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (!sonic_projection_quat_is_unit(local_rotations(bone)) ||
            !sonic_projection_vec3_finite(local_angular_velocities(bone)) ||
            !sonic_projection_vec3_finite(global_positions(bone)) ||
            !sonic_projection_quat_is_unit(global_rotations(bone))) {
            return sonic_projection_error(
                error, error_capacity,
                "projection input contains non-finite or invalid data at bone %d",
                bone);
        }
    }

    float positions[SonicG1JointCount];
    float velocities[SonicG1JointCount];
    float residuals[SonicG1JointCount];
    sonic_joint_projection_diagnostic diagnostic;
    if (!sonic_project_joint_state(
            positions,
            velocities,
            residuals,
            diagnostic,
            contract,
            local_rotations,
            local_angular_velocities,
            error,
            error_capacity)) {
        return false;
    }

    sonic_projected_pose candidate;
    for (int row = 0; row < SonicG1JointCount; ++row) {
        candidate.source_joint_position[row] = positions[row];
        candidate.source_joint_velocity[row] = velocities[row];
        candidate.off_axis_residual[row] = residuals[row];
    }
    candidate.physical_pelvis_position_holden = global_positions(G1_Hips);
    candidate.physical_pelvis_orientation_holden =
        sonic_projection_quat_canonical(global_rotations(G1_Hips));

    out = candidate;
    return true;
}
