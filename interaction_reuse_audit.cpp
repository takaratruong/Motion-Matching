#include "interaction_reuse_audit.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace interaction {
namespace {

constexpr uint8_t kReachPhase = 1U;
constexpr uint8_t kContactPhase = 2U;
constexpr uint8_t kLiftPhase = 3U;

bool finite(float value) {
    return std::isfinite(value);
}

bool finite(vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool valid_rotation(quat value) {
    if (!finite(value.w) || !finite(value.x) ||
        !finite(value.y) || !finite(value.z)) {
        return false;
    }
    const float squared = value.w * value.w + value.x * value.x +
        value.y * value.y + value.z * value.z;
    return squared > 1.0e-6F;
}

bool positive(vec3 value) {
    return finite(value) && value.x > 0.0F &&
        value.y > 0.0F && value.z > 0.0F;
}

bool valid_box(const OrientedBox& box) {
    return finite(box.world.position) &&
        valid_rotation(box.world.rotation) && positive(box.dimensions);
}

bool valid_collision_config(const TrajectoryCollisionConfig& config) {
    return finite(config.wrist_radius_m) && config.wrist_radius_m >= 0.0F &&
        finite(config.forearm_radius_m) && config.forearm_radius_m >= 0.0F &&
        finite(config.joint_radius_m) && config.joint_radius_m >= 0.0F &&
        finite(config.limb_radius_m) && config.limb_radius_m >= 0.0F &&
        finite(config.torso_radius_m) && config.torso_radius_m >= 0.0F;
}

bool complete_clip(const Database& database, size_t clip) {
    const int32_t start = database.range_starts.at(clip);
    const int32_t stop = database.range_stops.at(clip);
    if (start < 0 || stop <= start ||
        static_cast<uint32_t>(stop) > database.frame_count) {
        return false;
    }
    int32_t reach = -1;
    int32_t contact = -1;
    int32_t first_lift = -1;
    int32_t last_lift = -1;
    for (int32_t frame = start; frame < stop; ++frame) {
        const uint8_t phase = database.phases.at(static_cast<size_t>(frame));
        if (phase == kReachPhase && reach < 0) reach = frame;
        if (phase == kContactPhase && contact < 0) contact = frame;
        if (phase == kLiftPhase) {
            if (first_lift < 0) {
                first_lift = frame;
                last_lift = frame;
            } else if (frame == last_lift + 1) {
                last_lift = frame;
            }
        } else if (first_lift >= 0) {
            break;
        }
    }
    return reach >= start && reach < contact && contact < first_lift &&
        first_lift <= last_lift && last_lift < stop;
}

size_t audit_population(const Database& database, Hand hand) {
    size_t total = 0U;
    for (size_t clip = 0U; clip < database.clip_count; ++clip) {
        if (database.active_hands.at(clip) == static_cast<uint8_t>(hand) &&
            complete_clip(database, clip)) {
            ++total;
        }
    }
    return total;
}

void validate_audit_input(
    const HandTrajectoryQuery& query,
    const OrientedBox& object,
    const EnvironmentGeometry& environment,
    const TrajectoryCollisionConfig& collision,
    const ReuseAuditConfig& config) {
    if (!finite(config.maximum_contact_correction_m) ||
        config.maximum_contact_correction_m < 0.0F ||
        config.worker_count == 0U || config.worker_count > 4U ||
        config.display_limit == 0U ||
        (query.orientation_mode != GraspOrientationMode::ApproachAxis &&
         query.orientation_mode != GraspOrientationMode::PositionOnly) ||
        !valid_box(object) || !valid_collision_config(collision)) {
        throw std::invalid_argument("invalid motion reuse audit input");
    }
    for (const OrientedBox& box : environment.boxes) {
        if (!valid_box(box)) {
            throw std::invalid_argument("invalid audit environment geometry");
        }
    }
}

enum class OutcomeKind : uint8_t {
    Unprocessed = 0U,
    ContactRejected = 1U,
    ObjectRejected = 2U,
    EnvironmentRejected = 3U,
    Reusable = 4U,
};

struct Outcome {
    OutcomeKind kind = OutcomeKind::Unprocessed;
    std::unique_ptr<ReuseAuditMotion> motion;
};

using Clock = std::chrono::steady_clock;

uint64_t elapsed_milliseconds(Clock::time_point start) {
    return static_cast<uint64_t>(std::chrono::duration_cast<
        std::chrono::milliseconds>(Clock::now() - start).count());
}

}  // namespace

ReuseAuditResult audit_reusable_hand_trajectories(
    const Database& database,
    const HandTrajectoryQuery& query,
    const OrientedBox& object,
    const EnvironmentGeometry& environment,
    const TrajectoryCollisionConfig& collision,
    const ReuseAuditConfig& config) {
    validate_audit_input(query, object, environment, collision, config);
    const Clock::time_point start = Clock::now();
    const Clock::time_point deadline = start +
        std::chrono::milliseconds(config.deadline_milliseconds);

    ReuseAuditResult result{};
    result.counts.total = audit_population(database, query.hand);
    HandTrajectoryConfig selection{};
    selection.maximum_grasp_position_error_m =
        config.maximum_contact_correction_m;
    selection.maximum_grasp_orientation_error_radians = 3.141592654F;
    selection.maximum_compatible_clips = 4096U;
    std::vector<HandTrajectory> candidates = select_hand_trajectories(
        database, query, selection);
    if (candidates.size() > result.counts.total) {
        throw std::logic_error("audit selection exceeds complete population");
    }
    const size_t outside_envelope =
        result.counts.total - candidates.size();

    std::vector<Outcome> outcomes(candidates.size());
    std::atomic<size_t> next{0U};
    std::atomic<bool> stop{false};
    std::exception_ptr worker_error;
    std::mutex error_mutex;
    IKConfig ik{};
    ik.maximum_request_position_m = config.maximum_contact_correction_m;

    const auto work = [&]() {
        try {
            while (!stop.load(std::memory_order_relaxed)) {
                if (Clock::now() >= deadline) return;
                const size_t index = next.fetch_add(
                    1U, std::memory_order_relaxed);
                if (index >= candidates.size()) return;
                const HandTrajectory& candidate = candidates[index];
                const ShapedHandContact contact =
                    shape_hand_trajectory_contact(
                        database, candidate, query, ik);
                if (!contact.accepted) {
                    outcomes[index].kind = OutcomeKind::ContactRejected;
                    continue;
                }
                ShapedHandTrajectory shaped = shape_hand_trajectory(
                    database, candidate, query, ik);
                if (!shaped.contact_accepted ||
                    shaped.reason != contact.reason) {
                    throw std::logic_error(
                        "Contact-only and full audit shaping diverged");
                }
                const TrajectoryFeasibility feasibility =
                    evaluate_shaped_trajectory_feasibility(
                        shaped,
                        candidate.contact_point,
                        query.hand,
                        object,
                        environment,
                        collision);
                if (feasibility.reason ==
                    TrajectoryFeasibilityReason::ObjectCollision) {
                    outcomes[index].kind = OutcomeKind::ObjectRejected;
                    continue;
                }
                if (feasibility.reason ==
                    TrajectoryFeasibilityReason::EnvironmentCollision) {
                    outcomes[index].kind = OutcomeKind::EnvironmentRejected;
                    continue;
                }
                std::vector<Pose>{}.swap(shaped.poses);
                outcomes[index].kind = OutcomeKind::Reusable;
                outcomes[index].motion = std::make_unique<ReuseAuditMotion>(
                    ReuseAuditMotion{candidate, std::move(shaped)});
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!worker_error) worker_error = std::current_exception();
            }
            stop.store(true, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(config.worker_count);
    for (size_t worker = 0U; worker < config.worker_count; ++worker) {
        workers.emplace_back(work);
    }
    for (std::thread& worker : workers) worker.join();
    if (worker_error) std::rethrow_exception(worker_error);

    result.counts.processed = outside_envelope;
    for (const Outcome& outcome : outcomes) {
        if (outcome.kind == OutcomeKind::Unprocessed) continue;
        ++result.counts.processed;
        if (outcome.kind == OutcomeKind::ContactRejected) continue;
        ++result.counts.contact_accepted;
        ++result.counts.fully_shaped;
        if (outcome.kind == OutcomeKind::ObjectRejected) {
            ++result.counts.object_rejected;
        } else if (outcome.kind == OutcomeKind::EnvironmentRejected) {
            ++result.counts.environment_rejected;
        } else if (outcome.kind == OutcomeKind::Reusable) {
            ++result.counts.reusable;
        }
    }
    result.elapsed_milliseconds = elapsed_milliseconds(start);
    if (result.counts.processed != result.counts.total) {
        result.status = ReuseAuditStatus::Incomplete;
        return result;
    }

    result.status = ReuseAuditStatus::Complete;
    for (Outcome& outcome : outcomes) {
        if (outcome.motion) {
            result.displayed.push_back(std::move(*outcome.motion));
        }
    }
    std::stable_sort(
        result.displayed.begin(),
        result.displayed.end(),
        [](const ReuseAuditMotion& left, const ReuseAuditMotion& right) {
            return std::tie(
                       left.shaped.achieved_orientation_error_radians,
                       left.source.cost,
                       left.source.clip) <
                std::tie(
                       right.shaped.achieved_orientation_error_radians,
                       right.source.cost,
                       right.source.clip);
        });
    if (result.displayed.size() > config.display_limit) {
        result.displayed.resize(config.display_limit);
    }
    return result;
}

}  // namespace interaction
