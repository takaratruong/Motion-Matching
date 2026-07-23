#pragma once

#include "interaction_hand_trajectories.h"
#include "reach_motion.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace reach {

struct Query {
    Hand hand = Hand::Left;
    interaction::Transform target{};
    vec3 approach_world{1.0F, 0.0F, 0.0F};
};

struct CoverageConfig {
    float maximum_request_position_m = 0.45F;
    float accepted_position_m = 0.001F;
    float accepted_approach_radians = 0.261799388F;
    float accepted_orientation_radians = 1.047197551F;
    size_t maximum_candidates = 8192U;
};

struct Candidate {
    size_t clip = 0U;
    uint8_t yaw_index = 0U;
    float yaw_radians = 0.0F;
    float source_approach_error_radians = 0.0F;
    float cost = 0.0F;
};

enum class Rejection : uint8_t {
    None = 0U,
    OutsideEnvelope = 1U,
    InvalidSolver = 2U,
    PositionError = 3U,
    ApproachAxisError = 4U,
    FullOrientationError = 5U,
    ObjectCollision = 6U,
    EnvironmentCollision = 7U,
};

inline constexpr size_t kRejectionCount = 8U;

struct WristPathQuality {
    float backtrack_ratio = 0.0F;
    float excess_path_ratio = 0.0F;
    float directness_cost = 0.0F;
};

struct Evaluation {
    Candidate candidate{};
    std::vector<interaction::Pose> poses;
    Rejection rejection = Rejection::None;
    bool joint_limit_saturated = false;
    bool object_collision_observed = false;
    bool environment_collision_observed = false;
    float position_error_m = 0.0F;
    float approach_error_radians = 0.0F;
    float orientation_error_radians = 0.0F;
    float active_arm_deformation = 0.0F;
    float backtrack_ratio = 0.0F;
    float excess_path_ratio = 0.0F;
    float directness_cost = 0.0F;
    size_t collision_sample = 0U;
};

struct Diagnostics {
    size_t evaluations = 0U;
    size_t accepted = 0U;
    size_t joint_limit_saturated = 0U;
    std::array<size_t, kRejectionCount> rejection_counts{};
    std::array<size_t, 2U> hand_evaluations{};
    std::array<size_t, 2U> augmentation_evaluations{};
};

std::vector<Candidate> enumerate_candidates(const Pack& pack);

WristPathQuality measure_wrist_path_quality(
    const std::vector<vec3>& path);

std::vector<Candidate> select_candidates(
    const Pack& pack,
    const Query& query,
    const CoverageConfig& config = CoverageConfig{});

Evaluation shape_candidate(
    const Pack& pack,
    const Candidate& candidate,
    const Query& query,
    const CoverageConfig& config = CoverageConfig{});

Evaluation evaluate_candidate(
    const Pack& pack,
    const Candidate& candidate,
    const Query& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const CoverageConfig& config = CoverageConfig{},
    const interaction::TrajectoryCollisionConfig& collision_config =
        interaction::TrajectoryCollisionConfig{});

Diagnostics summarize_evaluations(
    const Pack& pack,
    const std::vector<Evaluation>& evaluations);

const char* rejection_name(Rejection rejection);

}  // namespace reach
