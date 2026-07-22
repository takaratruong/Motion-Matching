#include "reach_coverage.h"

#include "g1_skeleton.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace reach {
namespace {

constexpr float kPi = 3.14159265358979323846F;

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

void assign_final_errors(
    Evaluation& evaluation,
    const Query& query) {
    if (evaluation.poses.empty()) {
        evaluation.rejection = Rejection::InvalidSolver;
        return;
    }
    const size_t final_sample = evaluation.poses.size() - 1U;
    const size_t approach_sample = final_sample > 5U
        ? final_sample - 5U
        : 0U;
    const interaction::Transform final_hand = hand_transform(
        evaluation.poses[final_sample], query.hand);
    const interaction::Transform earlier_hand = hand_transform(
        evaluation.poses[approach_sample], query.hand);
    evaluation.position_error_m = length(
        final_hand.position - query.target.position);
    evaluation.approach_error_radians = direction_angle(
        final_hand.position - earlier_hand.position,
        query.approach_world);
    evaluation.orientation_error_radians = rotation_angle(
        query.target.rotation, final_hand.rotation);
}

}  // namespace

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
        static_cast<size_t>(pack.database.clip_count),
        config.maximum_candidates));
    for (size_t clip = 0U; clip < pack.database.clip_count; ++clip) {
        if (pack.database.active_hands.at(clip) !=
            static_cast<uint8_t>(query.hand)) {
            continue;
        }
        const interaction::Transform endpoint = endpoint_transform(
            pack.database, clip);
        const float position_error = length(
            endpoint.position - query.target.position);
        if (!finite(position_error) ||
            position_error > config.maximum_request_position_m) {
            continue;
        }
        const float approach_error = direction_angle(
            approach_direction(pack.database, clip), requested_approach);
        if (!finite(approach_error)) {
            continue;
        }
        candidates.push_back({
            clip,
            position_error,
            approach_error,
            position_error + 0.05F * approach_error,
        });
    }
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [](const Candidate& left, const Candidate& right) {
            if (left.cost != right.cost) return left.cost < right.cost;
            if (left.source_position_error_m != right.source_position_error_m) {
                return left.source_position_error_m <
                       right.source_position_error_m;
            }
            if (left.source_approach_error_radians !=
                right.source_approach_error_radians) {
                return left.source_approach_error_radians <
                       right.source_approach_error_radians;
            }
            return left.clip < right.clip;
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
        pack.database.active_hands.at(candidate.clip) !=
            static_cast<uint8_t>(query.hand)) {
        evaluation.rejection = Rejection::InvalidSolver;
        return evaluation;
    }
    const interaction::Transform source_endpoint = endpoint_transform(
        pack.database, candidate.clip);
    if (length(query.target.position - source_endpoint.position) >
        config.maximum_request_position_m) {
        evaluation.rejection = Rejection::OutsideEnvelope;
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
    const vec3 endpoint_offset =
        query.target.position - source_endpoint.position;
    const quat correction = quat_mul(
        query.target.rotation, quat_inv(source_endpoint.rotation));
    const interaction::IKConfig ik_config{
        0.45F,
        kPi,
        config.accepted_position_m,
        config.accepted_orientation_radians,
        0.05F,
        0.001F,
        0.10F,
        0.10F,
        20,
    };
    for (size_t sample = 0U; sample < frame_count; ++sample) {
        interaction::Pose pose = pose_at_frame(
            pack.database, start + static_cast<int32_t>(sample));
        const interaction::Transform source_hand = hand_transform(
            pose, query.hand);
        const float u = frame_count == 1U
            ? 1.0F
            : static_cast<float>(sample) /
                  static_cast<float>(frame_count - 1U);
        const float alpha = u * u * (3.0F - 2.0F * u);
        const interaction::Transform desired{
            source_hand.position + alpha * endpoint_offset,
            quat_mul(
                quat_nlerp_shortest(quat(), correction, alpha),
                source_hand.rotation),
        };
        const interaction::IKResult ik = interaction::solve_hand_ik(
            pose, interaction_hand(query.hand), desired, ik_config);
        evaluation.joint_limit_saturated =
            evaluation.joint_limit_saturated ||
            ik.reason == interaction::Reason::JointLimit;
        if (!finite_pose(pose) || !finite(ik.position_error_m) ||
            !finite(ik.orientation_error_radians)) {
            evaluation.rejection = Rejection::InvalidSolver;
            evaluation.poses.clear();
            return evaluation;
        }
        evaluation.poses.push_back(std::move(pose));
    }
    assign_final_errors(evaluation, query);
    if (!finite(evaluation.position_error_m) ||
        !finite(evaluation.approach_error_radians) ||
        !finite(evaluation.orientation_error_radians)) {
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
    if (evaluation.rejection != Rejection::None) {
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
    if (feasibility.reason ==
        interaction::TrajectoryFeasibilityReason::ObjectCollision) {
        evaluation.rejection = Rejection::ObjectCollision;
    } else if (feasibility.reason ==
               interaction::TrajectoryFeasibilityReason::EnvironmentCollision) {
        evaluation.rejection = Rejection::EnvironmentCollision;
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
