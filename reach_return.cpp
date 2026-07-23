#include "reach_return.h"

#include "g1_arm_joint_metadata.h"
#include "interaction_posture_ik.h"
#include "reach_placement.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace reach {
namespace {

bool finite(float value) {
    return std::isfinite(value);
}

bool finite(vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(quat value) {
    return finite(value.w) && finite(value.x) &&
        finite(value.y) && finite(value.z) &&
        quat_length(value) > 1.0e-8F;
}

bool positive(vec3 value) {
    return finite(value) &&
        value.x > 0.0F && value.y > 0.0F && value.z > 0.0F;
}

bool finite_pose(const interaction::Pose& pose) {
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        if (!finite(pose.positions[bone]) ||
            !finite(pose.velocities[bone]) ||
            !finite(pose.rotations[bone]) ||
            !finite(pose.angular_velocities[bone])) {
            return false;
        }
    }
    for (const float value : pose.hand_dof) {
        if (!finite(value)) return false;
    }
    for (const float value : pose.hand_dof_velocities) {
        if (!finite(value)) return false;
    }
    return true;
}

float smoothstep(float value) {
    const float x = std::clamp(value, 0.0F, 1.0F);
    return x * x * (3.0F - 2.0F * x);
}

float wrap_angle(float value) {
    constexpr float kPi = 3.14159265358979323846F;
    value = std::fmod(value + kPi, 2.0F * kPi);
    if (value < 0.0F) value += 2.0F * kPi;
    return value - kPi;
}

float rotation_error(quat left, quat right) {
    const quat delta = quat_abs(quat_mul(
        quat_normalize(left), quat_inv(quat_normalize(right))));
    return length(quat_to_scaled_angle_axis(delta));
}

std::array<vec3, 3U> box_axes(quat rotation) {
    rotation = quat_normalize(rotation);
    return {
        quat_mul_vec3(rotation, vec3(1.0F, 0.0F, 0.0F)),
        quat_mul_vec3(rotation, vec3(0.0F, 1.0F, 0.0F)),
        quat_mul_vec3(rotation, vec3(0.0F, 0.0F, 1.0F)),
    };
}

std::array<float, 3U> half_extents(vec3 dimensions) {
    return {0.5F * dimensions.x, 0.5F * dimensions.y, 0.5F * dimensions.z};
}

bool oriented_boxes_overlap(
    const interaction::OrientedBox& left,
    const interaction::OrientedBox& right) {
    constexpr float kOverlapToleranceM = 1.0e-6F;
    const std::array<vec3, 3U> left_axes =
        box_axes(left.world.rotation);
    const std::array<vec3, 3U> right_axes =
        box_axes(right.world.rotation);
    const std::array<float, 3U> left_half =
        half_extents(left.dimensions);
    const std::array<float, 3U> right_half =
        half_extents(right.dimensions);
    float rotation[3U][3U]{};
    float absolute_rotation[3U][3U]{};
    for (size_t left_axis = 0U; left_axis < 3U; ++left_axis) {
        for (size_t right_axis = 0U; right_axis < 3U; ++right_axis) {
            rotation[left_axis][right_axis] = dot(
                left_axes[left_axis], right_axes[right_axis]);
            absolute_rotation[left_axis][right_axis] =
                std::abs(rotation[left_axis][right_axis]);
        }
    }
    const vec3 world_translation =
        right.world.position - left.world.position;
    const std::array<float, 3U> translation{
        dot(world_translation, left_axes[0]),
        dot(world_translation, left_axes[1]),
        dot(world_translation, left_axes[2]),
    };

    for (size_t axis = 0U; axis < 3U; ++axis) {
        const float right_radius =
            right_half[0] * absolute_rotation[axis][0] +
            right_half[1] * absolute_rotation[axis][1] +
            right_half[2] * absolute_rotation[axis][2];
        if (std::abs(translation[axis]) >=
            left_half[axis] + right_radius - kOverlapToleranceM) {
            return false;
        }
    }
    for (size_t axis = 0U; axis < 3U; ++axis) {
        const float left_radius =
            left_half[0] * absolute_rotation[0][axis] +
            left_half[1] * absolute_rotation[1][axis] +
            left_half[2] * absolute_rotation[2][axis];
        const float projected_translation = std::abs(
            translation[0] * rotation[0][axis] +
            translation[1] * rotation[1][axis] +
            translation[2] * rotation[2][axis]);
        if (projected_translation >=
            left_radius + right_half[axis] - kOverlapToleranceM) {
            return false;
        }
    }
    for (size_t left_axis = 0U; left_axis < 3U; ++left_axis) {
        const size_t left_next = (left_axis + 1U) % 3U;
        const size_t left_last = (left_axis + 2U) % 3U;
        for (size_t right_axis = 0U;
             right_axis < 3U;
             ++right_axis) {
            constexpr float kDegenerateAxisSquared = 1.0e-8F;
            const float axis_squared = 1.0F -
                rotation[left_axis][right_axis] *
                    rotation[left_axis][right_axis];
            if (axis_squared <= kDegenerateAxisSquared) {
                continue;
            }
            const size_t right_next = (right_axis + 1U) % 3U;
            const size_t right_last = (right_axis + 2U) % 3U;
            const float left_radius =
                left_half[left_next] *
                    absolute_rotation[left_last][right_axis] +
                left_half[left_last] *
                    absolute_rotation[left_next][right_axis];
            const float right_radius =
                right_half[right_next] *
                    absolute_rotation[left_axis][right_last] +
                right_half[right_last] *
                    absolute_rotation[left_axis][right_next];
            const float projected_translation = std::abs(
                translation[left_last] *
                    rotation[left_next][right_axis] -
                translation[left_next] *
                    rotation[left_last][right_axis]);
            if (projected_translation >=
                left_radius + right_radius -
                    kOverlapToleranceM * std::sqrt(axis_squared)) {
                return false;
            }
        }
    }
    return true;
}

ReturnRejection frame_collision(
    const interaction::Pose& pose,
    Hand hand,
    interaction::Transform hand_in_object,
    vec3 object_dimensions,
    const interaction::EnvironmentGeometry& environment,
    const interaction::TrajectoryCollisionConfig& config) {
    const interaction::WorldPose world = interaction::world_pose(pose);
    const size_t wrist = detail::wrist_bone(hand);
    const interaction::Transform solved_hand{
        world.positions[wrist], world.rotations[wrist]};
    const interaction::OrientedBox object{
        interaction::compose(
            solved_hand, interaction::inverse(hand_in_object)),
        object_dimensions,
    };
    interaction::ShapedHandTrajectory single{};
    single.poses.push_back(pose);
    single.path.hands.push_back(solved_hand);
    single.path.elbows.push_back(
        world.positions[detail::elbow_bone(hand)]);
    single.contact_accepted = true;
    const interaction::TrajectoryFeasibility feasibility =
        interaction::evaluate_shaped_trajectory_feasibility(
            single,
            0U,
            detail::interaction_hand(hand),
            object,
            environment,
            config);
    if (feasibility.reason ==
        interaction::TrajectoryFeasibilityReason::ObjectCollision) {
        return ReturnRejection::ObjectCollision;
    }
    if (feasibility.reason ==
        interaction::TrajectoryFeasibilityReason::EnvironmentCollision) {
        return ReturnRejection::EnvironmentCollision;
    }
    for (const interaction::OrientedBox& box : environment.boxes) {
        if (oriented_boxes_overlap(object, box)) {
            return ReturnRejection::EnvironmentCollision;
        }
    }
    return ReturnRejection::None;
}

}  // namespace

ShapedReturn shape_recorded_return(
    const Pack& pack,
    const Candidate& candidate,
    const Query& query,
    const interaction::Pose& solved_contact,
    interaction::Transform hand_in_object,
    vec3 object_dimensions,
    const interaction::EnvironmentGeometry& environment,
    const SearchConfig& config) {
    ShapedReturn shaped{};
    if (candidate.clip >= pack.database.clip_count ||
        candidate.yaw_index >= kYawPlacementCount ||
        pack.database.active_hands.at(candidate.clip) !=
            static_cast<uint8_t>(query.hand) ||
        !finite_pose(solved_contact) ||
        !finite(hand_in_object.position) ||
        !finite(hand_in_object.rotation) ||
        !positive(object_dimensions)) {
        shaped.rejection = ReturnRejection::InvalidSolver;
        return shaped;
    }
    const int32_t contact =
        clip_contact_frame(pack.database, candidate.clip);
    const int32_t start =
        clip_return_start(pack.database, candidate.clip);
    const int32_t stop =
        clip_return_stop(pack.database, candidate.clip);
    if (contact < pack.database.range_starts.at(candidate.clip) ||
        start != contact + 1 ||
        start >= stop ||
        static_cast<uint32_t>(stop) > pack.database.frame_count) {
        shaped.rejection = ReturnRejection::InvalidSolver;
        return shaped;
    }
    const size_t return_count =
        static_cast<size_t>(stop - start);
    shaped.poses.reserve(return_count + 1U);
    shaped.poses.push_back(solved_contact);

    ReturnRejection collision = frame_collision(
        solved_contact,
        query.hand,
        hand_in_object,
        object_dimensions,
        environment,
        config.collision);
    if (collision != ReturnRejection::None) {
        shaped.rejection = collision;
        return shaped;
    }

    const interaction::Pose aligned_source_contact = place_pose(
        pack,
        candidate.clip,
        candidate.yaw_index,
        query.target.position,
        contact);
    const interaction::Transform solved_contact_hand =
        detail::hand_transform(solved_contact, query.hand);
    const interaction::Transform aligned_source_contact_hand =
        detail::hand_transform(aligned_source_contact, query.hand);
    const interaction::Transform delta = interaction::compose(
        solved_contact_hand,
        interaction::inverse(aligned_source_contact_hand));
    interaction::UpperBodyAngles previous_source =
        interaction::decompose_upper_body(
            aligned_source_contact,
            detail::interaction_hand(query.hand));
    interaction::UpperBodyAngles previous_solution =
        interaction::decompose_upper_body(
            solved_contact,
            detail::interaction_hand(query.hand));
    interaction::PostureIKConfig posture_config{};
    posture_config.accepted_position_m =
        config.coverage.accepted_position_m;
    posture_config.accepted_orientation_radians =
        config.coverage.accepted_orientation_radians;
    constexpr float kEndpointPositionToleranceM = 1.0e-6F;
    constexpr float kEndpointOrientationToleranceRadians = 1.0e-6F;

    for (size_t sample = 1U; sample <= return_count; ++sample) {
        interaction::Pose pose = place_pose(
            pack,
            candidate.clip,
            candidate.yaw_index,
            query.target.position,
            contact + static_cast<int32_t>(sample));
        const interaction::Pose source_pose = pose;
        const interaction::Transform aligned_source_hand =
            detail::hand_transform(source_pose, query.hand);
        const float weight = 1.0F - smoothstep(
            static_cast<float>(sample) /
            static_cast<float>(return_count));
        const interaction::Transform weighted_delta{
            weight * delta.position,
            quat_nlerp_shortest(
                quat(), delta.rotation, weight),
        };
        const interaction::Transform target = interaction::compose(
            weighted_delta, aligned_source_hand);
        const interaction::UpperBodyAngles source_angles =
            interaction::decompose_upper_body(
                source_pose,
                detail::interaction_hand(query.hand));
        interaction::UpperBodyAngles temporal_seed = source_angles;
        const bool final_sample = sample == return_count;
        for (size_t joint = 0U;
             joint < interaction::kUpperBodyJointCount;
             ++joint) {
            temporal_seed[joint] += wrap_angle(
                previous_solution[joint] -
                previous_source[joint]);
        }
        interaction::PostureIKConfig sample_config = posture_config;
        if (final_sample) {
            sample_config.accepted_position_m =
                kEndpointPositionToleranceM;
            sample_config.accepted_orientation_radians =
                kEndpointOrientationToleranceRadians;
        }
        const interaction::PostureIKResult ik =
            interaction::solve_hand_posture_ik_task_priority(
                pose,
                detail::interaction_hand(query.hand),
                target,
                source_pose,
                temporal_seed,
                sample_config);
        if (!ik.accepted || !finite_pose(pose) ||
            !finite(ik.position_error_m) ||
            !finite(ik.orientation_error_radians) ||
            !finite(static_cast<float>(ik.objective))) {
            shaped.rejection = ReturnRejection::InvalidSolver;
            shaped.rejected_sample = sample;
            return shaped;
        }
        if (final_sample) {
            const interaction::Transform achieved =
                detail::hand_transform(pose, query.hand);
            if (length(achieved.position - aligned_source_hand.position) >
                    kEndpointPositionToleranceM ||
                rotation_error(
                    achieved.rotation,
                    aligned_source_hand.rotation) >
                    kEndpointOrientationToleranceRadians) {
                shaped.rejection = ReturnRejection::InvalidSolver;
                shaped.rejected_sample = sample;
                return shaped;
            }
        }
        previous_source = source_angles;
        previous_solution = ik.joint_angles;
        shaped.poses.push_back(std::move(pose));
        collision = frame_collision(
            shaped.poses.back(),
            query.hand,
            hand_in_object,
            object_dimensions,
            environment,
            config.collision);
        if (collision != ReturnRejection::None) {
            shaped.rejection = collision;
            shaped.rejected_sample = sample;
            return shaped;
        }
    }
    return shaped;
}

}  // namespace reach
