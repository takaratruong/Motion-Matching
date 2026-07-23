#pragma once

#include "reach_coverage.h"

#include <chrono>
#include <cstddef>
#include <vector>

namespace reach {

struct ExhaustiveQuery {
    interaction::Transform target{};
    vec3 approach_world{1.0F, 0.0F, 0.0F};
};

struct SearchConfig {
    size_t worker_count = 4U;
    std::chrono::steady_clock::duration deadline = std::chrono::seconds(30);
    CoverageConfig coverage{};
    interaction::TrajectoryCollisionConfig collision{};
};

struct CompactEvaluation {
    Evaluation evaluation{};
    std::vector<vec3> hand_path;
};

struct SearchResult {
    bool complete = false;
    size_t total = 0U;
    size_t processed = 0U;
    std::chrono::steady_clock::duration elapsed{};
    std::vector<CompactEvaluation> evaluations;
    std::vector<size_t> accepted;
};

namespace detail {

bool complete_within_deadline(
    size_t processed,
    size_t total,
    size_t retained,
    std::chrono::steady_clock::duration elapsed,
    std::chrono::steady_clock::duration deadline);

bool accepted_quality_less(
    const Evaluation& left,
    const Evaluation& right);

}  // namespace detail

SearchResult search_all(
    const Pack& pack,
    const ExhaustiveQuery& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const SearchConfig& config = SearchConfig{});

Evaluation regenerate(
    const Pack& pack,
    const CompactEvaluation& compact,
    const ExhaustiveQuery& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const SearchConfig& config = SearchConfig{});

}  // namespace reach
