#include "reach_coverage.h"

#include "g1_arm_joint_metadata.h"
#include "g1_skeleton.h"
#include "interaction_posture_ik.h"
#include "reach_placement.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace reach {
namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kPostureCorrectionSeconds = 0.6F;
constexpr size_t kLockedApproachIntervals = 5U;

bool finite(float value) {
    return std::isfinite(value);
}

bool finite(vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(quat value) {
    return finite(value.w) && finite(value.x) && finite(value.y) &&
           finite(value.z) && quat_length(value) > 1.0e-8F;
}

bool valid_query(const Query& query) {
    return finite(query.target.position) && finite(query.target.rotation) &&
           finite(query.approach_world) &&
           length(query.approach_world) > 1.0e-8F;
}

bool valid_config(const CoverageConfig& config) {
    return finite(config.maximum_request_position_m) &&
           finite(config.accepted_position_m) &&
           finite(config.accepted_approach_radians) &&
           finite(config.accepted_orientation_radians) &&
           config.maximum_request_position_m >= 0.0F &&
           config.accepted_position_m >= 0.0F &&
           config.accepted_approach_radians >= 0.0F &&
           config.accepted_approach_radians <= kPi &&
           config.accepted_orientation_radians >= 0.0F &&
           config.accepted_orientation_radians <= kPi;
}

interaction::Hand interaction_hand(Hand hand) {
    return hand == Hand::Left
        ? interaction::Hand::Left
        : interaction::Hand::Right;
}

const std::array<interaction::HingeJoint, 7U>& arm_metadata(Hand hand) {
    return hand == Hand::Left
        ? interaction::kLeftArm
        : interaction::kRightArm;
}

const interaction::HingeJoint& upper_body_metadata(
    Hand hand,
    size_t joint) {
    if (joint < interaction::kWaist.size()) {
        return interaction::kWaist[joint];
    }
    return arm_metadata(hand)[joint - interaction::kWaist.size()];
}

size_t wrist_bone(Hand hand) {
    return hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : static_cast<size_t>(g1_skeleton::RightWrist);
}

size_t elbow_bone(Hand hand) {
    return hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftElbow)
        : static_cast<size_t>(g1_skeleton::RightElbow);
}

float direction_angle(vec3 left, vec3 right) {
    const float left_length = length(left);
    const float right_length = length(right);
    if (!(left_length > 1.0e-8F) || !(right_length > 1.0e-8F) ||
        !finite(left) || !finite(right)) {
        return std::numeric_limits<float>::infinity();
    }
    return std::acos(std::clamp(
        dot(left / left_length, right / right_length), -1.0F, 1.0F));
}

float rotation_angle(quat target, quat achieved) {
    if (!finite(target) || !finite(achieved)) {
        return std::numeric_limits<float>::infinity();
    }
    const quat delta = quat_abs(quat_mul(
        quat_normalize(target), quat_inv(quat_normalize(achieved))));
    return length(quat_to_scaled_angle_axis(delta));
}

float wrap_angle(float value) {
    value = std::fmod(value + kPi, 2.0F * kPi);
    if (value < 0.0F) value += 2.0F * kPi;
    return value - kPi;
}

float smoothstep(float value) {
    const float x = std::clamp(value, 0.0F, 1.0F);
    return x * x * (3.0F - 2.0F * x);
}

quat direction_alignment(vec3 source, vec3 target) {
    source = normalize(source);
    target = normalize(target);
    const float cosine = std::clamp(dot(source, target), -1.0F, 1.0F);
    if (cosine > 1.0F - 1.0e-6F) {
        return quat();
    }
    if (cosine < -1.0F + 1.0e-6F) {
        const vec3 basis = std::abs(source.x) < 0.8F
            ? vec3(1, 0, 0)
            : vec3(0, 1, 0);
        return quat_from_angle_axis(kPi, normalize(cross(source, basis)));
    }
    return quat_between(source, target);
}

interaction::Transform hand_transform(
    const interaction::Pose& pose,
    Hand hand) {
    const interaction::WorldPose world = interaction::world_pose(pose);
    const size_t wrist = wrist_bone(hand);
    return {world.positions[wrist], world.rotations[wrist]};
}

bool finite_pose(const interaction::Pose& pose) {
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        if (!finite(pose.positions[bone]) || !finite(pose.rotations[bone])) {
            return false;
        }
    }
    return true;
}

vec3 measured_approach_displacement(
    const std::vector<interaction::Pose>& poses,
    Hand hand) {
    const size_t final_sample = poses.size() - 1U;
    const interaction::Transform final_hand = hand_transform(
        poses[final_sample], hand);
    const size_t preferred_sample = final_sample > 5U
        ? final_sample - 5U
        : 0U;
    vec3 displacement = final_hand.position - hand_transform(
        poses[preferred_sample], hand).position;
    if (length(displacement) >= 0.01F) {
        return displacement;
    }
    const size_t earliest_sample = final_sample > 25U
        ? final_sample - 25U
        : 0U;
    for (size_t sample = preferred_sample;
         sample > earliest_sample;
         --sample) {
        const vec3 candidate = final_hand.position - hand_transform(
            poses[sample - 1U], hand).position;
        if (length(candidate) >= 0.01F) {
            return candidate;
        }
    }
    return displacement;
}

void assign_final_errors(
    Evaluation& evaluation,
    const Query& query) {
    if (evaluation.poses.empty()) {
        evaluation.rejection = Rejection::InvalidSolver;
        return;
    }
    const size_t final_sample = evaluation.poses.size() - 1U;
    const interaction::Transform final_hand = hand_transform(
        evaluation.poses[final_sample], query.hand);
    evaluation.position_error_m = length(
        final_hand.position - query.target.position);
    evaluation.approach_error_radians = direction_angle(
        measured_approach_displacement(evaluation.poses, query.hand),
        query.approach_world);
    evaluation.orientation_error_radians = rotation_angle(
        query.target.rotation, final_hand.rotation);
}

}  // namespace

std::vector<Candidate> enumerate_candidates(const Pack& pack) {
    std::vector<Candidate> candidates;
    candidates.reserve(
        static_cast<size_t>(pack.database.clip_count) * kYawPlacementCount);
    for (size_t clip = 0U; clip < pack.database.clip_count; ++clip) {
        for (uint8_t yaw = 0U; yaw < kYawPlacementCount; ++yaw) {
            candidates.push_back({
                clip,
                yaw,
                placement_yaw(yaw),
                0.0F,
                0.0F,
            });
        }
    }
    return candidates;
}

std::vector<Candidate> select_candidates(
    const Pack& pack,
    const Query& query,
    const CoverageConfig& config) {
    std::vector<Candidate> candidates;
    if (!valid_query(query) || !valid_config(config) ||
        pack.database.clip_count == 0U) {
        return candidates;
    }
    const vec3 requested_approach = normalize(query.approach_world);
    candidates.reserve(std::min(
        static_cast<size_t>(pack.database.clip_count) * kYawPlacementCount,
        config.maximum_candidates));
    for (Candidate candidate : enumerate_candidates(pack)) {
        if (pack.database.active_hands.at(candidate.clip) !=
            static_cast<uint8_t>(query.hand)) {
            continue;
        }
        const float approach_error = direction_angle(
            place_direction(
                approach_direction(pack.database, candidate.clip),
                candidate.yaw_index),
            requested_approach);
        if (!finite(approach_error)) {
            continue;
        }
        candidate.source_approach_error_radians = approach_error;
        candidate.cost = approach_error;
        candidates.push_back(candidate);
    }
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [](const Candidate& left, const Candidate& right) {
            if (left.cost != right.cost) return left.cost < right.cost;
            if (left.source_approach_error_radians !=
                right.source_approach_error_radians) {
                return left.source_approach_error_radians <
                       right.source_approach_error_radians;
            }
            if (left.clip != right.clip) return left.clip < right.clip;
            return left.yaw_index < right.yaw_index;
        });
    if (candidates.size() > config.maximum_candidates) {
        candidates.resize(config.maximum_candidates);
    }
    return candidates;
}

Evaluation shape_candidate(
    const Pack& pack,
    const Candidate& candidate,
    const Query& query,
    const CoverageConfig& config) {
    Evaluation evaluation{};
    evaluation.candidate = candidate;
    if (!valid_query(query) || !valid_config(config) ||
        candidate.clip >= pack.database.clip_count ||
        candidate.yaw_index >= kYawPlacementCount ||
        pack.database.active_hands.at(candidate.clip) !=
            static_cast<uint8_t>(query.hand)) {
        evaluation.rejection = Rejection::InvalidSolver;
        return evaluation;
    }
    const int32_t start = pack.database.range_starts.at(candidate.clip);
    const int32_t stop = pack.database.range_stops.at(candidate.clip);
    if (start < 0 || stop <= start ||
        static_cast<uint32_t>(stop) > pack.database.frame_count) {
        evaluation.rejection = Rejection::InvalidSolver;
        return evaluation;
    }
    const size_t frame_count = static_cast<size_t>(stop - start);
    evaluation.poses.reserve(frame_count);
    const vec3 source_approach = place_direction(
        approach_direction(pack.database, candidate.clip),
        candidate.yaw_index);
    const quat approach_alignment = direction_alignment(
        source_approach, query.approach_world);
    const bool warps_approach =
        direction_angle(source_approach, query.approach_world) > 1.0e-6F;
    const interaction::Pose placed_final_pose = place_pose(
        pack,
        candidate.clip,
        candidate.yaw_index,
        query.target.position,
        stop - 1);
    const interaction::Transform placed_endpoint = hand_transform(
        placed_final_pose, query.hand);
    const vec3 contact_offset =
        query.target.position - placed_endpoint.position;
    const bool warps_translation = length(contact_offset) > 1.0e-6F;
    const quat correction = quat_mul(
        query.target.rotation, quat_inv(placed_endpoint.rotation));
    const bool warps_rotation =
        rotation_angle(correction, quat()) > 1.0e-6F;
    const bool requires_correction =
        warps_translation || warps_approach || warps_rotation;
    if (pack.database.fps_denominator == 0U) {
        evaluation.rejection = Rejection::InvalidSolver;
        return evaluation;
    }
    const float fps =
        static_cast<float>(pack.database.fps_numerator) /
        static_cast<float>(pack.database.fps_denominator);
    if (!finite(fps) || !(fps > 0.0F)) {
        evaluation.rejection = Rejection::InvalidSolver;
        return evaluation;
    }
    const size_t correction_intervals = static_cast<size_t>(
        std::ceil(kPostureCorrectionSeconds * fps));
    const size_t available_intervals = frame_count - 1U;
    if (requires_correction &&
        available_intervals < correction_intervals) {
        evaluation.rejection = Rejection::PositionError;
        return evaluation;
    }
    const size_t correction_start = available_intervals >
            correction_intervals
        ? available_intervals - correction_intervals
        : 0U;
    const size_t aligned_sample = available_intervals >
            kLockedApproachIntervals
        ? available_intervals - kLockedApproachIntervals
        : available_intervals;
    if (requires_correction && aligned_sample <= correction_start) {
        evaluation.rejection = Rejection::PositionError;
        return evaluation;
    }
    interaction::PostureIKConfig posture_config{};
    posture_config.accepted_position_m = config.accepted_position_m;
    posture_config.accepted_orientation_radians =
        config.accepted_orientation_radians;
    interaction::UpperBodyAngles previous_source{};
    interaction::UpperBodyAngles previous_solution{};
    bool have_previous = false;
    constexpr size_t upper_body_joint_count =
        interaction::kUpperBodyJointCount;
    for (size_t sample = 0U; sample < frame_count; ++sample) {
        interaction::Pose pose = place_pose(
            pack,
            candidate.clip,
            candidate.yaw_index,
            query.target.position,
            start + static_cast<int32_t>(sample));
        const interaction::Pose placed_pose = pose;
        const interaction::Transform placed_hand = hand_transform(
            pose, query.hand);
        const float correction_u = sample <= correction_start
            ? 0.0F
            : (sample >= aligned_sample
                ? 1.0F
                : static_cast<float>(sample - correction_start) /
                      static_cast<float>(
                          aligned_sample - correction_start));
        const float correction_weight = smoothstep(correction_u);
        const vec3 translated_position =
            placed_hand.position + correction_weight * contact_offset;
        const vec3 relative =
            translated_position - query.target.position;
        const vec3 rotated = quat_mul_vec3(approach_alignment, relative);
        const interaction::Transform desired{
            translated_position +
                correction_weight * (rotated - relative),
            quat_mul(
                quat_nlerp_shortest(
                    quat(), correction, correction_weight),
                placed_hand.rotation),
        };
        const interaction::Hand hand = interaction_hand(query.hand);
        const interaction::UpperBodyAngles source_angles =
            interaction::decompose_upper_body(placed_pose, hand);
        interaction::UpperBodyAngles temporal_seed = source_angles;
        if (have_previous) {
            for (size_t joint = 0U;
                 joint < upper_body_joint_count;
                 ++joint) {
                temporal_seed[joint] += wrap_angle(
                    previous_solution[joint] -
                    previous_source[joint]);
            }
        }
        interaction::PostureIKResult ik{};
        ik.joint_angles = source_angles;
        if (requires_correction && correction_weight > 0.0F) {
            ik = interaction::solve_hand_posture_ik_task_priority(
                pose,
                hand,
                desired,
                placed_pose,
                temporal_seed,
                posture_config);
            evaluation.joint_limit_saturated =
                evaluation.joint_limit_saturated ||
                ik.joint_limit_saturated;
        }
        previous_source = source_angles;
        previous_solution = ik.joint_angles;
        have_previous = true;
        if (!finite_pose(pose) || !finite(ik.position_error_m) ||
            !finite(ik.orientation_error_radians) ||
            !finite(static_cast<float>(ik.objective))) {
            evaluation.rejection = Rejection::InvalidSolver;
            evaluation.poses.clear();
            return evaluation;
        }
        for (size_t joint_index = 0U;
             joint_index < upper_body_joint_count;
             ++joint_index) {
            const interaction::HingeJoint& joint =
                upper_body_metadata(query.hand, joint_index);
            const size_t bone = static_cast<size_t>(joint.bone);
            const float angle = rotation_angle(
                placed_pose.rotations[bone], pose.rotations[bone]);
            evaluation.active_arm_deformation += angle * angle;
        }
        evaluation.poses.push_back(std::move(pose));
    }
    evaluation.active_arm_deformation /= static_cast<float>(
        frame_count * upper_body_joint_count);
    assign_final_errors(evaluation, query);
    if (!finite(evaluation.position_error_m) ||
        !finite(evaluation.approach_error_radians) ||
        !finite(evaluation.orientation_error_radians) ||
        !finite(evaluation.active_arm_deformation)) {
        evaluation.rejection = Rejection::InvalidSolver;
    } else if (evaluation.position_error_m > config.accepted_position_m) {
        evaluation.rejection = Rejection::PositionError;
    } else if (evaluation.approach_error_radians >
               config.accepted_approach_radians) {
        evaluation.rejection = Rejection::ApproachAxisError;
    } else if (evaluation.orientation_error_radians >
               config.accepted_orientation_radians) {
        evaluation.rejection = Rejection::FullOrientationError;
    }
    return evaluation;
}

Evaluation evaluate_candidate(
    const Pack& pack,
    const Candidate& candidate,
    const Query& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const CoverageConfig& config,
    const interaction::TrajectoryCollisionConfig& collision_config) {
    Evaluation evaluation = shape_candidate(pack, candidate, query, config);
    if (evaluation.poses.empty()) {
        return evaluation;
    }
    interaction::ShapedHandTrajectory shaped{};
    shaped.poses = evaluation.poses;
    shaped.contact_accepted = true;
    shaped.path.hands.reserve(shaped.poses.size());
    shaped.path.elbows.reserve(shaped.poses.size());
    for (const interaction::Pose& pose : shaped.poses) {
        const interaction::WorldPose world = interaction::world_pose(pose);
        const size_t wrist = wrist_bone(query.hand);
        shaped.path.hands.push_back({
            world.positions[wrist], world.rotations[wrist]});
        shaped.path.elbows.push_back(world.positions[elbow_bone(query.hand)]);
    }
    const interaction::TrajectoryFeasibility feasibility =
        interaction::evaluate_shaped_trajectory_feasibility(
            shaped,
            shaped.poses.size() - 1U,
            interaction_hand(query.hand),
            object,
            environment,
            collision_config);
    evaluation.collision_sample = feasibility.sample;
    evaluation.object_collision_observed =
        feasibility.object_collision_observed;
    evaluation.environment_collision_observed =
        feasibility.environment_collision_observed;
    if (feasibility.reason ==
        interaction::TrajectoryFeasibilityReason::ObjectCollision) {
        if (evaluation.rejection == Rejection::None) {
            evaluation.rejection = Rejection::ObjectCollision;
        }
    } else if (feasibility.reason ==
               interaction::TrajectoryFeasibilityReason::EnvironmentCollision) {
        if (evaluation.rejection == Rejection::None) {
            evaluation.rejection = Rejection::EnvironmentCollision;
        }
    }
    return evaluation;
}

Diagnostics summarize_evaluations(
    const Pack& pack,
    const std::vector<Evaluation>& evaluations) {
    Diagnostics diagnostics{};
    for (const Evaluation& evaluation : evaluations) {
        if (evaluation.candidate.clip >= pack.database.clip_count) {
            throw std::out_of_range("reach diagnostic clip outside database");
        }
        ++diagnostics.evaluations;
        ++diagnostics.rejection_counts.at(
            static_cast<size_t>(evaluation.rejection));
        if (evaluation.rejection == Rejection::None) {
            ++diagnostics.accepted;
        }
        if (evaluation.joint_limit_saturated) {
            ++diagnostics.joint_limit_saturated;
        }
        ++diagnostics.hand_evaluations.at(
            pack.database.active_hands.at(evaluation.candidate.clip));
        ++diagnostics.augmentation_evaluations.at(
            pack.database.augmentations.at(evaluation.candidate.clip));
    }
    return diagnostics;
}

const char* rejection_name(Rejection rejection) {
    switch (rejection) {
        case Rejection::None: return "accepted";
        case Rejection::OutsideEnvelope: return "outside-envelope";
        case Rejection::InvalidSolver: return "invalid-solver";
        case Rejection::PositionError: return "position-error";
        case Rejection::ApproachAxisError: return "approach-axis-error";
        case Rejection::FullOrientationError: return "full-orientation-error";
        case Rejection::ObjectCollision: return "object-collision";
        case Rejection::EnvironmentCollision: return "environment-collision";
    }
    return "unknown";
}

}  // namespace reach
