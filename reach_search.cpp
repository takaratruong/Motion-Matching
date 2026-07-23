#include "reach_search.h"

#include "g1_skeleton.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace reach {
namespace {

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

void validate(
    const ExhaustiveQuery& query,
    const SearchConfig& config) {
    if (!finite(query.target.position) || !finite(query.target.rotation) ||
        !finite(query.approach_world) ||
        length(query.approach_world) <= 1.0e-8F) {
        throw std::invalid_argument("invalid exhaustive reach query");
    }
    if (config.worker_count == 0U || config.worker_count > 64U ||
        config.deadline < std::chrono::steady_clock::duration::zero()) {
        throw std::invalid_argument("invalid exhaustive reach search config");
    }
}

Hand candidate_hand(const Pack& pack, const Candidate& candidate) {
    return static_cast<Hand>(
        pack.database.active_hands.at(candidate.clip));
}

size_t wrist_bone(Hand hand) {
    return hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : static_cast<size_t>(g1_skeleton::RightWrist);
}

std::vector<vec3> extract_active_wrist_path(
    const std::vector<interaction::Pose>& poses,
    Hand hand) {
    std::vector<vec3> path;
    path.reserve(poses.size());
    const size_t wrist = wrist_bone(hand);
    for (const interaction::Pose& pose : poses) {
        path.push_back(interaction::world_pose(pose).positions[wrist]);
    }
    return path;
}

bool same_metric(float left, float right) {
    return finite(left) && finite(right) && std::abs(left - right) <= 1.0e-5F;
}

}  // namespace

SearchResult search_all(
    const Pack& pack,
    const ExhaustiveQuery& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const SearchConfig& config) {
    validate(query, config);
    const std::vector<Candidate> candidates = enumerate_candidates(pack);
    const auto started = std::chrono::steady_clock::now();
    const auto expires = started + config.deadline;
    std::vector<std::optional<CompactEvaluation>> slots(candidates.size());
    std::atomic<size_t> next{0U};
    std::atomic<size_t> processed{0U};

    const auto worker = [&] {
        while (std::chrono::steady_clock::now() < expires) {
            const size_t index = next.fetch_add(1U);
            if (index >= candidates.size()) return;
            const Candidate candidate = candidates[index];
            Query hand_query{};
            hand_query.hand = candidate_hand(pack, candidate);
            hand_query.target = query.target;
            hand_query.approach_world = query.approach_world;
            Evaluation evaluation = evaluate_candidate(
                pack,
                candidate,
                hand_query,
                object,
                environment,
                config.coverage,
                config.collision);
            CompactEvaluation compact{};
            compact.hand_path = extract_active_wrist_path(
                evaluation.poses, hand_query.hand);
            compact.evaluation = std::move(evaluation);
            std::vector<interaction::Pose>().swap(compact.evaluation.poses);
            slots[index] = std::move(compact);
            processed.fetch_add(1U);
        }
    };

    std::vector<std::thread> workers;
    const size_t worker_count = std::min(
        config.worker_count, std::max<size_t>(1U, candidates.size()));
    workers.reserve(worker_count);
    for (size_t worker_index = 0U;
         worker_index < worker_count;
         ++worker_index) {
        workers.emplace_back(worker);
    }
    for (std::thread& thread : workers) thread.join();

    SearchResult result{};
    result.total = candidates.size();
    result.processed = processed.load();
    result.elapsed = std::chrono::steady_clock::now() - started;
    result.evaluations.reserve(result.processed);
    for (std::optional<CompactEvaluation>& slot : slots) {
        if (slot.has_value()) {
            result.evaluations.push_back(std::move(*slot));
        }
    }
    result.complete = result.processed == result.total &&
                      result.evaluations.size() == result.total;
    if (!result.complete) return result;

    for (size_t index = 0U; index < result.evaluations.size(); ++index) {
        if (result.evaluations[index].evaluation.rejection == Rejection::None) {
            result.accepted.push_back(index);
        }
    }
    std::stable_sort(
        result.accepted.begin(), result.accepted.end(),
        [&](size_t left_index, size_t right_index) {
            const Evaluation& left =
                result.evaluations[left_index].evaluation;
            const Evaluation& right =
                result.evaluations[right_index].evaluation;
            return std::tie(
                       left.active_arm_deformation,
                       left.orientation_error_radians,
                       left.approach_error_radians,
                       left.candidate.clip,
                       left.candidate.yaw_index) <
                   std::tie(
                       right.active_arm_deformation,
                       right.orientation_error_radians,
                       right.approach_error_radians,
                       right.candidate.clip,
                       right.candidate.yaw_index);
        });
    return result;
}

Evaluation regenerate(
    const Pack& pack,
    const CompactEvaluation& compact,
    const ExhaustiveQuery& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const SearchConfig& config) {
    validate(query, config);
    Query hand_query{};
    hand_query.hand = candidate_hand(pack, compact.evaluation.candidate);
    hand_query.target = query.target;
    hand_query.approach_world = query.approach_world;
    Evaluation evaluation = evaluate_candidate(
        pack,
        compact.evaluation.candidate,
        hand_query,
        object,
        environment,
        config.coverage,
        config.collision);
    if (evaluation.rejection != compact.evaluation.rejection ||
        evaluation.object_collision_observed !=
            compact.evaluation.object_collision_observed ||
        evaluation.environment_collision_observed !=
            compact.evaluation.environment_collision_observed ||
        !same_metric(
            evaluation.position_error_m,
            compact.evaluation.position_error_m) ||
        !same_metric(
            evaluation.approach_error_radians,
            compact.evaluation.approach_error_radians) ||
        !same_metric(
            evaluation.orientation_error_radians,
            compact.evaluation.orientation_error_radians) ||
        !same_metric(
            evaluation.active_arm_deformation,
            compact.evaluation.active_arm_deformation)) {
        throw std::runtime_error(
            "regenerated reach evaluation disagrees with compact search");
    }
    return evaluation;
}

}  // namespace reach
