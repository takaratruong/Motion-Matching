#include "interaction_matcher.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace interaction {
namespace {

constexpr float kRootClearanceRadius = 0.25F;
constexpr float kRootClearanceProxyTolerance = 0.01F;
constexpr float kSlabEpsilon = 1.0e-7F;

struct ClipFrames {
    int32_t clip = -1;
    int32_t start = -1;
    int32_t stop = -1;
    int32_t reach = -1;
    int32_t contact = -1;
    int32_t lift = -1;
    int32_t hold = -1;
    Hand hand = Hand::Right;
};

struct FailureSet {
    bool out_of_range = false;
    bool correction_limit = false;
    bool blocked_path = false;
    bool poor_match = false;
};

enum class CandidateStatus {
    Accepted,
    NoCandidate,
    OutOfRange,
    CorrectionLimit,
    BlockedPath,
    PoorMatch,
};

struct CandidateEvaluation {
    CandidateStatus status = CandidateStatus::OutOfRange;
    bool path_feasible = false;
    bool total_cost_available = false;
    MatchCandidate candidate{};
};

MatchResult reject(Reason reason) {
    MatchResult result{};
    result.reason = reason;
    return result;
}

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

bool positive_dimensions(vec3 value) {
    return finite(value) && value.x > 0.0F &&
           value.y > 0.0F && value.z > 0.0F;
}

bool same_affordance(
    const GraspAffordance& left,
    const GraspAffordance& right) {
    return same_authored_grasp_affordance(left, right);
}

const GraspAffordance* exact_target_affordance(
    const matcher_detail::PickEvaluationInput& input) {
    for (const GraspAffordance& affordance : input.target.affordances) {
        if (affordance.id == input.affordance.id) return &affordance;
    }
    return nullptr;
}

std::optional<Reason> validate_pick_context(
    const matcher_detail::PickEvaluationInput& input) {
    if (input.database == nullptr || input.features == nullptr) {
        return Reason::PackUnavailable;
    }
    if (input.target.handle.id == 0U ||
        input.target.handle.generation == 0U) {
        return Reason::TargetUnavailable;
    }
    const GraspAffordance* authored = exact_target_affordance(input);
    if (authored == nullptr || !same_affordance(*authored, input.affordance)) {
        return Reason::TargetUnavailable;
    }
    if (input.target.state == ObjectState::Attached ||
        input.target.state == ObjectState::Held) {
        return Reason::TargetUnavailable;
    }
    if (!finite(input.target.object_world.position) ||
        !finite(input.target.object_world.rotation) ||
        !positive_dimensions(input.target.object_dimensions) ||
        !finite(input.target.table_world.position) ||
        !finite(input.target.table_world.rotation) ||
        !positive_dimensions(input.target.table_size) ||
        !finite(input.affordance.hand_in_object.position) ||
        !finite(input.affordance.hand_in_object.rotation) ||
        !finite(input.affordance.clearance_radius) ||
        input.affordance.clearance_radius < 0.0F) {
        return Reason::TargetUnavailable;
    }
    return std::nullopt;
}

matcher_detail::PickEvaluationInput pick_evaluation_input(
    const MatchInput& input) {
    return {
        input.database,
        input.features,
        input.query,
        input.locomotion,
        input.target,
        input.affordance,
    };
}

std::optional<Reason> validate_request(const MatchInput& input) {
    if (input.request.target.id == 0U ||
        input.request.target.generation == 0U ||
        input.target.handle.id == 0U ||
        input.target.handle.generation == 0U ||
        input.request.request_id == 0U) {
        return Reason::TargetUnavailable;
    }
    if (input.request.target != input.target.handle) {
        return Reason::TargetChanged;
    }
    if (input.request.affordance_id != input.affordance.id) {
        return Reason::TargetUnavailable;
    }
    const matcher_detail::PickEvaluationInput evaluation_input =
        pick_evaluation_input(input);
    const GraspAffordance* authored = exact_target_affordance(
        evaluation_input);
    if (authored == nullptr || !same_affordance(*authored, input.affordance)) {
        return Reason::TargetUnavailable;
    }
    if (input.target.state == ObjectState::Attached ||
        input.target.state == ObjectState::Held) {
        return Reason::TargetUnavailable;
    }
    if (input.target.state == ObjectState::Targeted &&
        input.target.owner_request != input.request.request_id) {
        return Reason::TargetChanged;
    }
    return validate_pick_context(evaluation_input);
}

vec3 read_vec3(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 3U;
    return vec3(
        values.at(offset), values.at(offset + 1U), values.at(offset + 2U));
}

quat read_quat(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 4U;
    return quat(
        values.at(offset),
        values.at(offset + 1U),
        values.at(offset + 2U),
        values.at(offset + 3U));
}

float yaw_radians(quat rotation) {
    const vec3 facing = quat_mul_vec3(rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(facing.x, facing.z);
}

float shortest_angle(float angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

Transform source_object_transform(
    const Database& database,
    int32_t frame) {
    return {
        read_vec3(database.object_positions, static_cast<size_t>(frame)),
        read_quat(database.object_rotations, static_cast<size_t>(frame)),
    };
}

Transform scene_alignment(
    const Transform& source_object,
    const Transform& target_object) {
    const float yaw = shortest_angle(
        yaw_radians(target_object.rotation) -
        yaw_radians(source_object.rotation));
    const quat rotation = quat_from_angle_axis(
        yaw, vec3(0.0F, 1.0F, 0.0F));
    const vec3 rotated_source =
        quat_mul_vec3(rotation, source_object.position);
    return {
        vec3(
            target_object.position.x - rotated_source.x,
            0.0F,
            target_object.position.z - rotated_source.z),
        rotation,
    };
}

Transform root_transform(const WorldPose& pose) {
    return {
        pose.positions[g1_skeleton::Simulation],
        pose.rotations[g1_skeleton::Simulation],
    };
}

Transform hand_transform(const WorldPose& pose, Hand hand) {
    const size_t bone = hand == Hand::Left ? kLeftHandBone : kRightHandBone;
    return {pose.positions[bone], pose.rotations[bone]};
}

float planar_distance(vec3 left, vec3 right) {
    return std::hypot(left.x - right.x, left.z - right.z);
}

vec3 box_local_point(Transform box, vec3 point) {
    return quat_mul_vec3(
        quat_inv(box.rotation), point - box.position);
}

bool segment_intersects_expanded_box(
    vec3 start_world,
    vec3 stop_world,
    Transform box,
    vec3 box_size,
    float expansion) {
    const vec3 start = box_local_point(box, start_world);
    const vec3 stop = box_local_point(box, stop_world);
    const vec3 delta = stop - start;
    const vec3 half = 0.5F * box_size + expansion;
    const std::array<float, 3> starts = {start.x, start.y, start.z};
    const std::array<float, 3> deltas = {delta.x, delta.y, delta.z};
    const std::array<float, 3> halves = {half.x, half.y, half.z};
    float minimum = 0.0F;
    float maximum = 1.0F;
    for (size_t axis = 0; axis < starts.size(); ++axis) {
        if (std::abs(deltas[axis]) <= kSlabEpsilon) {
            if (starts[axis] < -halves[axis] ||
                starts[axis] > halves[axis]) {
                return false;
            }
            continue;
        }
        float first = (-halves[axis] - starts[axis]) / deltas[axis];
        float second = (halves[axis] - starts[axis]) / deltas[axis];
        if (first > second) std::swap(first, second);
        minimum = std::max(minimum, first);
        maximum = std::min(maximum, second);
        if (minimum > maximum) return false;
    }
    return true;
}

bool root_intersects_table(
    vec3 root_world,
    Transform table,
    vec3 table_size) {
    const float yaw = yaw_radians(table.rotation);
    const quat inverse_yaw = quat_from_angle_axis(
        -yaw, vec3(0.0F, 1.0F, 0.0F));
    const vec3 local = quat_mul_vec3(
        inverse_yaw, root_world - table.position);
    const float half_x = 0.5F * table_size.x +
        kRootClearanceRadius - kRootClearanceProxyTolerance;
    const float half_z = 0.5F * table_size.z +
        kRootClearanceRadius - kRootClearanceProxyTolerance;
    return std::abs(local.x) <= half_x && std::abs(local.z) <= half_z;
}

float entry_correction_weight(
    int32_t entry_frame,
    int32_t contact_frame,
    int32_t frame) {
    if (frame <= entry_frame) return 1.0F;
    if (frame >= contact_frame) return 0.0F;
    const float alpha = static_cast<float>(frame - entry_frame) /
        static_cast<float>(contact_frame - entry_frame);
    const float smoothstep = alpha * alpha * (3.0F - 2.0F * alpha);
    return 1.0F - smoothstep;
}

vec3 corrected_root_position(
    vec3 mapped_root,
    vec3 entry_root_offset,
    float weight) {
    return mapped_root + weight * entry_root_offset;
}

vec3 corrected_hand_position(
    vec3 mapped_hand,
    vec3 mapped_root,
    vec3 entry_root_offset,
    float entry_yaw_offset,
    float weight) {
    const quat yaw_correction = quat_from_angle_axis(
        weight * entry_yaw_offset, vec3(0.0F, 1.0F, 0.0F));
    return corrected_root_position(mapped_root, entry_root_offset, weight) +
        quat_mul_vec3(yaw_correction, mapped_hand - mapped_root);
}

bool path_is_clear(
    const matcher_detail::PickEvaluationInput& input,
    const ClipFrames& frames,
    const Transform& scene_from_source,
    int32_t entry_frame,
    vec3 entry_root_offset,
    float entry_yaw_offset) {
    const Database& database = *input.database;
    for (int32_t frame = entry_frame; frame < frames.stop; ++frame) {
        const Transform mapped_root = compose(
            scene_from_source,
            root_transform(world_pose(pose_at_frame(database, frame))));
        const float weight = entry_correction_weight(
            entry_frame, frames.contact, frame);
        const vec3 root = corrected_root_position(
            mapped_root.position, entry_root_offset, weight);
        if (root_intersects_table(
                root,
                input.target.table_world,
                input.target.table_size)) {
            return false;
        }
    }

    const WorldPose entry_pose = world_pose(
        pose_at_frame(database, entry_frame));
    const Transform entry_mapped_root = compose(
        scene_from_source, root_transform(entry_pose));
    const Transform entry_mapped_hand = compose(
        scene_from_source,
        hand_transform(entry_pose, frames.hand));
    vec3 previous = corrected_hand_position(
        entry_mapped_hand.position,
        entry_mapped_root.position,
        entry_root_offset,
        entry_yaw_offset,
        1.0F);
    for (int32_t frame = entry_frame + 1;
         frame < frames.stop;
         ++frame) {
        const WorldPose source_pose = world_pose(
            pose_at_frame(database, frame));
        const Transform mapped_root = compose(
            scene_from_source, root_transform(source_pose));
        const Transform mapped_hand = compose(
            scene_from_source,
            hand_transform(source_pose, frames.hand));
        const float weight = entry_correction_weight(
            entry_frame, frames.contact, frame);
        const vec3 current = corrected_hand_position(
            mapped_hand.position,
            mapped_root.position,
            entry_root_offset,
            entry_yaw_offset,
            weight);
        if (segment_intersects_expanded_box(
                previous,
                current,
                input.target.table_world,
                input.target.table_size,
                input.affordance.clearance_radius)) {
            return false;
        }
        if (frame < frames.contact && segment_intersects_expanded_box(
                previous,
                current,
                input.target.object_world,
                input.target.object_dimensions,
                input.affordance.clearance_radius)) {
            return false;
        }
        previous = current;
    }
    return true;
}

std::optional<ClipFrames> inspect_clip(
    const Database& database,
    int32_t clip,
    Hand requested_hand,
    FailureSet& failures) {
    try {
        const uint8_t hand_value = database.active_hands.at(
            static_cast<size_t>(clip));
        if (hand_value > 1U) {
            failures.out_of_range = true;
            return std::nullopt;
        }
        const Hand hand = hand_value == 0U ? Hand::Left : Hand::Right;
        if (hand != requested_hand) return std::nullopt;

        ClipFrames frames{};
        frames.clip = clip;
        frames.start = database.range_starts.at(static_cast<size_t>(clip));
        frames.stop = database.range_stops.at(static_cast<size_t>(clip));
        frames.hand = hand;
        if (frames.start < 0 || frames.stop <= frames.start ||
            frames.stop > static_cast<int32_t>(database.frame_count)) {
            failures.out_of_range = true;
            return std::nullopt;
        }

        uint8_t previous_phase = 0U;
        bool first = true;
        for (int32_t frame = frames.start; frame < frames.stop; ++frame) {
            const uint8_t phase = database.phases.at(
                static_cast<size_t>(frame));
            if (phase > static_cast<uint8_t>(Phase::Hold) ||
                (!first && phase < previous_phase)) {
                failures.out_of_range = true;
                return std::nullopt;
            }
            first = false;
            previous_phase = phase;
            if (phase == static_cast<uint8_t>(Phase::Reach) &&
                frames.reach < 0) {
                frames.reach = frame;
            } else if (phase == static_cast<uint8_t>(Phase::Contact) &&
                       frames.contact < 0) {
                frames.contact = frame;
            } else if (phase == static_cast<uint8_t>(Phase::Lift) &&
                       frames.lift < 0) {
                frames.lift = frame;
            } else if (phase == static_cast<uint8_t>(Phase::Hold) &&
                       frames.hold < 0) {
                frames.hold = frame;
            }
        }
        if (!(frames.start <= frames.reach &&
              frames.reach < frames.contact &&
              frames.contact < frames.lift &&
              frames.lift < frames.hold &&
              frames.hold < frames.stop)) {
            failures.out_of_range = true;
            return std::nullopt;
        }
        return frames;
    } catch (const std::out_of_range&) {
        failures.out_of_range = true;
        return std::nullopt;
    }
}

std::optional<std::array<float, 5>> group_costs(
    const matcher_detail::PickEvaluationInput& input,
    int32_t entry_frame) {
    const Features& features = *input.features;
    if (features.dimension != kFeatureDimension ||
        features.feature_count != kFeatureDimension ||
        features.group_count != 5U ||
        features.frame_count <= static_cast<uint32_t>(entry_frame) ||
        features.group_starts.size() != 5U ||
        features.group_stops.size() != 5U) {
        return std::nullopt;
    }
    std::array<float, 5> costs{};
    for (size_t group = 0; group < costs.size(); ++group) {
        const size_t start = features.group_starts.at(group);
        const size_t stop = features.group_stops.at(group);
        if (stop <= start || stop > kFeatureDimension) return std::nullopt;
        float squared_sum = 0.0F;
        for (size_t dimension = start; dimension < stop; ++dimension) {
            const size_t index =
                static_cast<size_t>(entry_frame) * features.dimension +
                dimension;
            const float difference =
                features.values.at(index) - input.query.at(dimension);
            squared_sum += difference * difference;
        }
        costs[group] = squared_sum / static_cast<float>(stop - start);
    }
    return costs;
}

std::optional<float> total_cost(
    const std::array<float, 5>& costs,
    const MatchConfig& config) {
    float weighted = 0.0F;
    float weight_sum = 0.0F;
    for (size_t group = 0; group < costs.size(); ++group) {
        const float weight = config.group_weights[group];
        if (!finite(weight) || weight < 0.0F) return std::nullopt;
        weighted += weight * costs[group];
        weight_sum += weight;
    }
    if (!(weight_sum > 0.0F) || !finite(weighted)) return std::nullopt;
    return weighted / weight_sum;
}

CandidateEvaluation evaluate_candidate(
    const matcher_detail::PickEvaluationInput& input,
    const MatchConfig& config,
    const ClipFrames& frames,
    int32_t entry_frame,
    const Transform& current_root,
    const matcher_detail::CandidateFeasibility& candidate_feasibility) {
    CandidateEvaluation evaluation{};
    try {
        const Database& database = *input.database;
        const Transform source_object = source_object_transform(
            database, frames.contact - 1);
        const Transform scene_from_source = scene_alignment(
            source_object, input.target.object_world);
        const WorldPose entry_pose = world_pose(
            pose_at_frame(database, entry_frame));
        const Transform mapped_root = compose(
            scene_from_source, root_transform(entry_pose));
        const vec3 root_offset(
            current_root.position.x - mapped_root.position.x,
            0.0F,
            current_root.position.z - mapped_root.position.z);
        const float yaw_offset = shortest_angle(
            yaw_radians(current_root.rotation) -
            yaw_radians(mapped_root.rotation));
        if (std::hypot(root_offset.x, root_offset.z) >
                config.maximum_root_correction_m ||
            std::abs(yaw_offset) >
                config.maximum_yaw_correction_radians) {
            evaluation.status = CandidateStatus::CorrectionLimit;
            return evaluation;
        }

        const Transform mapped_hand = compose(
            scene_from_source,
            hand_transform(
                world_pose(pose_at_frame(database, frames.contact)),
                frames.hand));
        const Transform target_hand = compose(
            input.target.object_world,
            input.affordance.hand_in_object);
        if (length(mapped_hand.position - target_hand.position) >
                config.maximum_hand_correction_m ||
            quat_angle_between(mapped_hand.rotation, target_hand.rotation) >
                config.maximum_hand_orientation_radians) {
            evaluation.status = CandidateStatus::CorrectionLimit;
            return evaluation;
        }

        if (!path_is_clear(
                input,
                frames,
                scene_from_source,
                entry_frame,
                root_offset,
                yaw_offset)) {
            evaluation.status = CandidateStatus::BlockedPath;
            return evaluation;
        }

        evaluation.candidate.clip = frames.clip;
        evaluation.candidate.entry_frame = entry_frame;
        evaluation.candidate.contact_frame = frames.contact;
        evaluation.candidate.lift_frame = frames.lift;
        evaluation.candidate.hold_frame = frames.hold;
        evaluation.candidate.scene_from_source = scene_from_source;
        evaluation.candidate.entry_root_offset = root_offset;
        evaluation.candidate.entry_yaw_offset = yaw_offset;
    } catch (const std::out_of_range&) {
        evaluation.status = CandidateStatus::OutOfRange;
        return evaluation;
    }

    if (candidate_feasibility) {
        const Reason reason = candidate_feasibility(evaluation.candidate);
        switch (reason) {
        case Reason::None:
            break;
        case Reason::NoCandidate:
            evaluation.status = CandidateStatus::NoCandidate;
            return evaluation;
        case Reason::OutOfRange:
            evaluation.status = CandidateStatus::OutOfRange;
            return evaluation;
        case Reason::CorrectionLimit:
            evaluation.status = CandidateStatus::CorrectionLimit;
            return evaluation;
        case Reason::BlockedPath:
            evaluation.status = CandidateStatus::BlockedPath;
            return evaluation;
        default:
            throw std::invalid_argument(
                "interaction candidate feasibility returned a non-hard reason");
        }
    }

    evaluation.path_feasible = true;

    try {
        const std::optional<std::array<float, 5>> costs =
            group_costs(input, entry_frame);
        if (!costs.has_value()) {
            evaluation.status = CandidateStatus::OutOfRange;
            return evaluation;
        }
        evaluation.candidate.group_costs = *costs;
        const std::optional<float> cost = total_cost(*costs, config);
        if (cost.has_value()) {
            evaluation.total_cost_available = true;
            evaluation.candidate.total_cost = *cost;
        }
        if (!cost.has_value() || !finite(config.maximum_cost) ||
            *cost > config.maximum_cost) {
            evaluation.status = CandidateStatus::PoorMatch;
            return evaluation;
        }

        evaluation.status = CandidateStatus::Accepted;
        return evaluation;
    } catch (const std::out_of_range&) {
        evaluation.status = CandidateStatus::OutOfRange;
        return evaluation;
    }
}

void remember_failure(FailureSet& failures, CandidateStatus status) {
    switch (status) {
    case CandidateStatus::Accepted:
        return;
    case CandidateStatus::NoCandidate:
        return;
    case CandidateStatus::OutOfRange:
        failures.out_of_range = true;
        return;
    case CandidateStatus::CorrectionLimit:
        failures.correction_limit = true;
        return;
    case CandidateStatus::BlockedPath:
        failures.blocked_path = true;
        return;
    case CandidateStatus::PoorMatch:
        failures.poor_match = true;
        return;
    }
}

bool better_candidate(
    const MatchCandidate& candidate,
    const MatchCandidate& best) {
    if (candidate.total_cost != best.total_cost) {
        return candidate.total_cost < best.total_cost;
    }
    if (candidate.clip != best.clip) return candidate.clip < best.clip;
    return candidate.entry_frame < best.entry_frame;
}

void consider(
    const CandidateEvaluation& evaluation,
    std::optional<MatchCandidate>& best,
    FailureSet& failures,
    matcher_detail::PickEvaluation& result) {
    if (evaluation.path_feasible && !result.path_feasible) {
        result.path_feasible = true;
        result.feasible_entry_frame = evaluation.candidate.entry_frame;
        result.contact_frame = evaluation.candidate.contact_frame;
    }
    if (evaluation.total_cost_available &&
        (!result.total_cost_available ||
         evaluation.candidate.total_cost < result.total_cost)) {
        result.total_cost_available = true;
        result.total_cost = evaluation.candidate.total_cost;
    }
    if (evaluation.status != CandidateStatus::Accepted) {
        remember_failure(failures, evaluation.status);
        return;
    }
    if (!best.has_value() || better_candidate(evaluation.candidate, *best)) {
        best = evaluation.candidate;
    }
}

Reason aggregate_reason(const FailureSet& failures) {
    if (failures.out_of_range) return Reason::OutOfRange;
    if (failures.correction_limit) return Reason::CorrectionLimit;
    if (failures.blocked_path) return Reason::BlockedPath;
    if (failures.poor_match) return Reason::PoorMatch;
    return Reason::NoCandidate;
}

Reason aggregate_hard_reason(const FailureSet& failures) {
    if (failures.out_of_range) return Reason::OutOfRange;
    if (failures.correction_limit) return Reason::CorrectionLimit;
    if (failures.blocked_path) return Reason::BlockedPath;
    return Reason::NoCandidate;
}

void finish_evaluation(
    matcher_detail::PickEvaluation& result,
    const std::optional<MatchCandidate>& best,
    const FailureSet& failures) {
    result.path_reason = result.path_feasible
        ? Reason::None
        : aggregate_hard_reason(failures);
    if (best.has_value()) {
        result.match_ready = true;
        result.match_reason = Reason::None;
        result.selection = {true, *best, Reason::None};
        result.total_cost_available = true;
        result.total_cost = best->total_cost;
        return;
    }
    result.match_ready = false;
    result.selection = reject(aggregate_reason(failures));
    result.match_reason =
        !failures.out_of_range && failures.poor_match
            ? Reason::PoorMatch
            : result.selection.reason;
}

}  // namespace

matcher_detail::PickEvaluation matcher_detail::evaluate_pick_entries(
    const PickEvaluationInput& input,
    const MatchConfig& config,
    const CandidateFeasibility& candidate_feasibility) {
    PickEvaluation result{};
    if (const std::optional<Reason> invalid = validate_pick_context(input)) {
        result.path_reason = *invalid;
        result.match_reason = *invalid;
        result.selection = reject(*invalid);
        return result;
    }

    const WorldPose current_pose = world_pose(input.locomotion.pose);
    const Transform current_root = root_transform(current_pose);
    if (!finite(current_root.position) || !finite(current_root.rotation) ||
        !finite(config.maximum_approach_m) ||
        planar_distance(
            current_root.position,
            input.target.object_world.position) > config.maximum_approach_m) {
        result.path_reason = Reason::OutOfRange;
        result.match_reason = Reason::OutOfRange;
        result.selection = reject(Reason::OutOfRange);
        return result;
    }

    const Database& database = *input.database;
    FailureSet failures{};
    std::vector<ClipFrames> clips;
    for (uint32_t clip = 0; clip < database.clip_count; ++clip) {
        const std::optional<ClipFrames> frames = inspect_clip(
            database,
            static_cast<int32_t>(clip),
            input.affordance.hand,
            failures);
        if (frames.has_value()) clips.push_back(*frames);
    }

    std::optional<MatchCandidate> best;
    for (const ClipFrames& frames : clips) {
        consider(
            evaluate_candidate(
                input,
                config,
                frames,
                frames.reach,
                current_root,
                candidate_feasibility),
            best,
            failures,
            result);
    }
    if (best.has_value()) {
        finish_evaluation(result, best, failures);
        return result;
    }

    for (const ClipFrames& frames : clips) {
        for (int32_t frame = frames.reach - 1;
             frame >= frames.start;
             --frame) {
            if (database.phases.at(static_cast<size_t>(frame)) !=
                static_cast<uint8_t>(Phase::Approach)) {
                continue;
            }
            consider(
                evaluate_candidate(
                    input,
                    config,
                    frames,
                    frame,
                    current_root,
                    candidate_feasibility),
                best,
                failures,
                result);
        }
    }
    finish_evaluation(result, best, failures);
    return result;
}

MatchResult select_whole_clip(
    const MatchInput& input,
    const MatchConfig& config) {
    if (input.database == nullptr || input.features == nullptr) {
        return reject(Reason::PackUnavailable);
    }
    if (const std::optional<Reason> invalid = validate_request(input)) {
        return reject(*invalid);
    }
    return matcher_detail::evaluate_pick_entries(
        pick_evaluation_input(input), config).selection;
}

}  // namespace interaction
