#include "interaction_learned_pickup_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>

namespace interaction {
namespace {

float yaw_of(quat rotation) {
    const vec3 forward = quat_mul_vec3(
        rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(forward.x, forward.z);
}

float planar_distance(vec3 left, vec3 right) {
    const double x = static_cast<double>(right.x) - left.x;
    const double z = static_cast<double>(right.z) - left.z;
    return static_cast<float>(std::sqrt(x * x + z * z));
}

float yaw_error(quat left, quat right) {
    const float delta = yaw_of(right) - yaw_of(left);
    return std::abs(std::atan2(std::sin(delta), std::cos(delta)));
}

bool same_obstacles(
    const std::vector<PickNavigationObstacle>& left,
    const std::vector<PickNavigationObstacle>& right) {
    if (left.size() != right.size()) return false;
    for (size_t index = 0U; index < left.size(); ++index) {
        const float a[]{
            left[index].center_world.x, left[index].center_world.y,
            left[index].center_world.z, left[index].size_world.x,
            left[index].size_world.y, left[index].size_world.z,
        };
        const float b[]{
            right[index].center_world.x, right[index].center_world.y,
            right[index].center_world.z, right[index].size_world.x,
            right[index].size_world.y, right[index].size_world.z,
        };
        if (std::memcmp(a, b, sizeof(a)) != 0) return false;
    }
    return true;
}

bool same_root(PickEntryRoot left, PickEntryRoot right) {
    const float a[]{left.world_x, left.world_z, left.world_yaw_radians};
    const float b[]{right.world_x, right.world_z, right.world_yaw_radians};
    return std::memcmp(a, b, sizeof(a)) == 0;
}

Transform relative_to_world(
    Transform entry,
    const FunnelSample& relative) {
    const double entry_yaw = yaw_of(entry.rotation);
    const double sine = std::sin(entry_yaw);
    const double cosine = std::cos(entry_yaw);
    const double relative_yaw = std::atan2(
        static_cast<double>(relative.yaw_sin),
        static_cast<double>(relative.yaw_cos));
    return {
        vec3(
            static_cast<float>(entry.position.x +
                cosine * relative.x + sine * relative.z),
            entry.position.y,
            static_cast<float>(entry.position.z -
                sine * relative.x + cosine * relative.z)),
        quat_from_angle_axis(
            static_cast<float>(entry_yaw + relative_yaw),
            vec3(0.0F, 1.0F, 0.0F)),
    };
}

FunnelSample world_to_relative(Transform entry, Transform world) {
    const double entry_yaw = yaw_of(entry.rotation);
    const double world_yaw = yaw_of(world.rotation);
    const double sine = std::sin(entry_yaw);
    const double cosine = std::cos(entry_yaw);
    const double dx = static_cast<double>(world.position.x) - entry.position.x;
    const double dz = static_cast<double>(world.position.z) - entry.position.z;
    const double relative_yaw = world_yaw - entry_yaw;
    return {
        static_cast<float>(cosine * dx - sine * dz),
        static_cast<float>(sine * dx + cosine * dz),
        static_cast<float>(std::sin(relative_yaw)),
        static_cast<float>(std::cos(relative_yaw)),
    };
}

FunnelExecutionTargets world_targets(
    Transform entry,
    const FunnelProposal& proposal) {
    const FunnelExecutionTargets relative =
        expand_funnel_execution(proposal.samples);
    FunnelExecutionTargets world{};
    for (size_t index = 0U; index < world.size(); ++index) {
        const Transform target = relative_to_world(entry, relative[index]);
        world[index] = {
            target.position.x,
            target.position.z,
            std::sin(yaw_of(target.rotation)),
            std::cos(yaw_of(target.rotation)),
        };
    }
    return world;
}

Transform target_transform(float y, const FunnelSample& sample) {
    return {
        vec3(sample.x, y, sample.z),
        quat_from_angle_axis(
            std::atan2(sample.yaw_sin, sample.yaw_cos),
            vec3(0.0F, 1.0F, 0.0F)),
    };
}

PickEntryRoot entry_root(const FunnelSample& sample) {
    return {
        sample.x,
        sample.z,
        std::atan2(sample.yaw_sin, sample.yaw_cos),
    };
}

const GraspAffordance* find_affordance(
    const InteractionTarget& target,
    uint32_t affordance_id) {
    for (const GraspAffordance& affordance : target.affordances) {
        if (affordance.id == affordance_id) return &affordance;
    }
    return nullptr;
}

}  // namespace

LearnedSmartPickupBackend::LearnedSmartPickupBackend(
    InteractionFunnelArtifact artifact)
    : artifact_(std::move(artifact)) {
    int accepted = 0;
    for (const FunnelProposal& proposal : artifact_->proposals()) {
        if (proposal.accepted) ++accepted;
    }
    learned_diagnostics_.accepted_proposal_count = accepted;
}

LearnedSmartPickupBackend::LearnedSmartPickupBackend(
    FunnelProposalProvider& provider,
    LearnedPickupConfig config)
    : provider_(&provider), config_(std::move(config)) {}

bool LearnedSmartPickupBackend::begin(
    const PickAssistStart& start,
    const InteractionTarget* post_step_target) {
    if (provider_ == nullptr) return authored_.begin(start, post_step_target);
    if (active() || post_step_target == nullptr ||
        !same_interaction_target_snapshot(
            start.target_snapshot, *post_step_target) ||
        find_affordance(start.target_snapshot, start.affordance_id) == nullptr) {
        fail(LearnedPickupFailureReason::TargetChanged);
        return false;
    }
    start_ = start;
    if (start_.live_obstacles.empty()) start_.live_obstacles = start_.obstacles;
    capture_ = select_funnel_capture(
        start.root_world, start.target_snapshot,
        start_.live_obstacles, config_.capture);
    diagnostics_ = {};
    diagnostics_.target = start.target_snapshot.handle;
    diagnostics_.affordance_id = start.affordance_id;
    learned_diagnostics_ = {};
    if (capture_.reason != PickSlotReason::None) {
        fail(LearnedPickupFailureReason::NoSafeCapture);
        return false;
    }
    learned_diagnostics_.state = LearnedPickupState::CoarseCapture;
    diagnostics_.state = PickAssistState::SlotApproach;
    return true;
}

void LearnedSmartPickupBackend::cancel() {
    if (provider_ == nullptr) {
        authored_.cancel();
        if (follower_.has_value()) {
            follower_->cancel(FunnelCancelReason::ExplicitCancel);
            learned_diagnostics_.armed = false;
            learned_diagnostics_.armed_seed = 0U;
            refresh_follower_diagnostics();
        }
        return;
    }
    if (follower_.has_value()) {
        follower_->cancel(FunnelCancelReason::ExplicitCancel);
        refresh_follower_diagnostics();
    }
    fail(LearnedPickupFailureReason::Cancelled);
}

PickAssistOutput LearnedSmartPickupBackend::observe(
    const PickAssistObservation& observation) {
    if (provider_ == nullptr) return authored_.observe(observation);
    return observe_learned(observation);
}

PickAssistOutput LearnedSmartPickupBackend::observe_learned(
    const PickAssistObservation& observation) {
    if (!active()) return {};
    if (!authority_matches(observation)) {
        return fail(observation.runtime_state == RuntimeState::Locomotion
            ? LearnedPickupFailureReason::TargetChanged
            : LearnedPickupFailureReason::RuntimeChanged);
    }

    const float capture_error = planar_distance(
        observation.displayed_root.position,
        capture_.target_world.position);
    const float capture_yaw_error = yaw_error(
        observation.displayed_root.rotation,
        capture_.target_world.rotation);
    const bool capture_settled =
        capture_error <= config_.capture_position_tolerance_m &&
        capture_yaw_error <= config_.capture_yaw_tolerance_radians &&
        observation.displayed_planar_speed_mps <=
            config_.capture_speed_tolerance_mps;

    switch (learned_diagnostics_.state) {
    case LearnedPickupState::CoarseCapture:
    case LearnedPickupState::CaptureSettling: {
        if (revalidate_frozen_pick_slot(
                observation.displayed_root,
                capture_.target_world,
                start_.target_snapshot,
                observation.live_obstacles,
                config_.capture.route) != PickSlotReason::None) {
            return fail(LearnedPickupFailureReason::RouteBlocked);
        }
        if (!capture_settled) {
            learned_diagnostics_.state = LearnedPickupState::CoarseCapture;
            learned_diagnostics_.settled_capture_ticks = 0U;
            diagnostics_.state = PickAssistState::SlotApproach;
            PickAssistOutput output{};
            output.override_steering = true;
            output.left_stick = arrival_navigation_stick(
                capture_.target_world.position,
                observation.displayed_root.position,
                observation.camera_azimuth,
                config_.arrival);
            output.right_stick = arrival_facing_stick(
                quat_mul_vec3(
                    capture_.target_world.rotation,
                    vec3(0.0F, 0.0F, 1.0F)),
                observation.camera_azimuth);
            return output;
        }
        learned_diagnostics_.state = LearnedPickupState::CaptureSettling;
        diagnostics_.state = PickAssistState::Settling;
        ++learned_diagnostics_.settled_capture_ticks;
        if (learned_diagnostics_.settled_capture_ticks <
            config_.required_capture_settle_ticks) {
            return braking_output();
        }
        const GraspAffordance* affordance = find_affordance(
            start_.target_snapshot, start_.affordance_id);
        const auto condition = affordance == nullptr
            ? std::nullopt
            : build_funnel_condition(
                  observation.displayed_root,
                  observation.simulation_velocity,
                  start_.target_snapshot,
                  *affordance);
        if (!condition.has_value()) {
            return fail(LearnedPickupFailureReason::InvalidCondition);
        }
        frozen_entry_world_ = observation.displayed_root;
        proposal_request_ = {};
        proposal_request_.request_id = observation.controller_tick ==
                std::numeric_limits<uint64_t>::max()
            ? 1U
            : observation.controller_tick + 1U;
        proposal_request_.batch_seed = config_.batch_seed;
        proposal_request_.checkpoint_sha256 = config_.checkpoint_sha256;
        proposal_request_.condition = *condition;
        learned_diagnostics_.frozen_condition = *condition;
        if (!provider_->begin(proposal_request_)) {
            return fail(LearnedPickupFailureReason::ProposalLaunchFailed);
        }
        learned_diagnostics_.state = LearnedPickupState::ProposalPending;
        diagnostics_.state = PickAssistState::SlotSelectionPreview;
        return braking_output();
    }
    case LearnedPickupState::ProposalPending: {
        ++learned_diagnostics_.proposal_pending_ticks;
        if (learned_diagnostics_.proposal_pending_ticks >
            config_.maximum_proposal_pending_ticks) {
            return fail(LearnedPickupFailureReason::ProposalTimeout);
        }
        FunnelProposalPoll poll = provider_->poll();
        if (poll.state == FunnelProposalPollState::Pending) {
            return braking_output();
        }
        if (poll.state != FunnelProposalPollState::Ready ||
            !poll.artifact.has_value()) {
            return fail(LearnedPickupFailureReason::ProposalWorkerFailed);
        }
        if (poll.artifact->request_id() != proposal_request_.request_id ||
            poll.artifact->batch_seed() != proposal_request_.batch_seed ||
            poll.artifact->checkpoint_sha256() !=
                proposal_request_.checkpoint_sha256 ||
            std::memcmp(
                poll.artifact->condition().data(),
                proposal_request_.condition.data(),
                sizeof(float) * kFunnelConditionDim) != 0) {
            return fail(LearnedPickupFailureReason::ProposalIdentityMismatch);
        }
        artifact_ = std::move(*poll.artifact);
        selection_requests_.clear();
        selection_proposal_indices_.clear();
        int accepted = 0;
        for (size_t proposal_index = 0U;
             proposal_index < artifact_->proposals().size();
             ++proposal_index) {
            const FunnelProposal& proposal =
                artifact_->proposals()[proposal_index];
            if (!proposal.accepted) continue;
            ++accepted;
            const FunnelExecutionTargets targets =
                world_targets(frozen_entry_world_, proposal);
            bool route_valid = true;
            Transform previous = frozen_entry_world_;
            for (const FunnelSample& target : targets) {
                const Transform next = target_transform(
                    frozen_entry_world_.position.y, target);
                if (revalidate_frozen_pick_slot(
                        previous, next, start_.target_snapshot,
                        observation.live_obstacles,
                        config_.capture.route) != PickSlotReason::None) {
                    route_valid = false;
                    break;
                }
                previous = next;
            }
            if (!route_valid) continue;
            selection_proposal_indices_.push_back(proposal_index);
            selection_requests_.push_back({
                static_cast<uint32_t>(proposal_index + 1U),
                entry_root(targets.back()),
            });
        }
        learned_diagnostics_.accepted_proposal_count = accepted;
        if (selection_requests_.empty()) {
            return fail(LearnedPickupFailureReason::NoAcceptedProposal);
        }
        learned_diagnostics_.state = LearnedPickupState::SelectionPreview;
        PickAssistOutput output = braking_output();
        output.preview_requests = selection_requests_;
        return output;
    }
    case LearnedPickupState::SelectionPreview: {
        if (observation.preview_results.size() != selection_requests_.size() ||
            observation.preview_snapshot_fingerprint !=
                observation.snapshot_fingerprint) {
            return fail(LearnedPickupFailureReason::SelectionPreviewRejected);
        }
        std::optional<size_t> winner{};
        std::tuple<uint64_t, uint64_t, double, float, size_t> winner_key{};
        for (size_t index = 0U; index < selection_requests_.size(); ++index) {
            const PickAssistPreviewResult& result =
                observation.preview_results[index];
            if (result.request.slot_id != selection_requests_[index].slot_id ||
                !same_root(result.request.root, selection_requests_[index].root) ||
                !result.preview.has_value()) {
                continue;
            }
            const PickEntryPreview& preview = *result.preview;
            if (!preview.path_feasible || !preview.match_ready ||
                !same_root(preview.prospective_root, result.request.root) ||
                !std::isfinite(preview.total_cost)) {
                continue;
            }
            const size_t proposal_index = selection_proposal_indices_[index];
            const FunnelExecutionTargets targets = world_targets(
                frozen_entry_world_, artifact_->proposals()[proposal_index]);
            double route = 0.0;
            for (size_t tick = 1U; tick < targets.size(); ++tick) {
                route += planar_distance(
                    vec3(targets[tick - 1U].x, 0.0F, targets[tick - 1U].z),
                    vec3(targets[tick].x, 0.0F, targets[tick].z));
            }
            const uint64_t route_mm = static_cast<uint64_t>(
                std::floor(route * 1000.0 + 0.5));
            const double heading_delta =
                std::atan2(targets.back().yaw_sin, targets.back().yaw_cos) -
                yaw_of(frozen_entry_world_.rotation);
            const uint64_t heading_mrad = static_cast<uint64_t>(std::floor(
                std::abs(std::atan2(
                    std::sin(heading_delta), std::cos(heading_delta))) *
                    1000.0 + 0.5));
            double minimum_clearance = std::numeric_limits<double>::infinity();
            for (const FunnelSample& target : targets) {
                for (const PickNavigationObstacle& obstacle :
                     observation.live_obstacles) {
                    const double dx = static_cast<double>(target.x) -
                        obstacle.center_world.x;
                    const double dz = static_cast<double>(target.z) -
                        obstacle.center_world.z;
                    minimum_clearance = std::min(
                        minimum_clearance, std::sqrt(dx * dx + dz * dz));
                }
            }
            const auto key = std::make_tuple(
                route_mm, heading_mrad, -minimum_clearance,
                preview.total_cost, proposal_index);
            if (!winner.has_value() || key < winner_key) {
                winner = index;
                winner_key = key;
            }
        }
        if (!winner.has_value()) {
            return fail(LearnedPickupFailureReason::SelectionPreviewRejected);
        }
        const size_t proposal_index = selection_proposal_indices_[*winner];
        const FunnelProposal& proposal = artifact_->proposals()[proposal_index];
        selected_world_targets_ = world_targets(frozen_entry_world_, proposal);
        final_root_ = entry_root(selected_world_targets_.back());
        follower_.emplace(
            proposal.seed, proposal.samples, observation.controller_tick + 1U);
        learned_diagnostics_.armed = true;
        learned_diagnostics_.armed_seed = proposal.seed;
        learned_diagnostics_.selected_proposal_index =
            static_cast<int>(proposal_index);
        learned_diagnostics_.state = LearnedPickupState::FunnelFollow;
        refresh_follower_diagnostics();
        return braking_output();
    }
    case LearnedPickupState::FunnelFollow: {
        const int next = follower_->diagnostics().published_count;
        Transform previous = observation.displayed_root;
        for (int tick = next; tick < kFunnelExecutionTickCount; ++tick) {
            const Transform target = target_transform(
                frozen_entry_world_.position.y,
                selected_world_targets_[static_cast<size_t>(tick)]);
            if (revalidate_frozen_pick_slot(
                    previous, target, start_.target_snapshot,
                    observation.live_obstacles,
                    config_.capture.route) != PickSlotReason::None) {
                return fail(LearnedPickupFailureReason::RouteBlocked);
            }
            previous = target;
        }
        const FunnelFollowerOutput followed = follower_->tick({
            observation.controller_tick,
            world_to_relative(
                frozen_entry_world_, observation.displayed_root),
        });
        refresh_follower_diagnostics();
        if (followed.state == FunnelFollowerState::Cancelled) {
            return fail(LearnedPickupFailureReason::FollowerFailed);
        }
        if (followed.state == FunnelFollowerState::Completed) {
            learned_diagnostics_.state = LearnedPickupState::FinalPreview;
            diagnostics_.state = PickAssistState::FinalPreview;
            PickAssistOutput output = braking_output();
            output.preview_requests.push_back({
                static_cast<uint32_t>(
                    learned_diagnostics_.selected_proposal_index + 1),
                final_root_,
            });
            return output;
        }
        if (!followed.published) return braking_output();
        const Transform target = relative_to_world(
            frozen_entry_world_, followed.sample);
        PickAssistOutput output{};
        output.override_steering = true;
        output.left_stick = arrival_navigation_stick(
            target.position,
            observation.displayed_root.position,
            observation.camera_azimuth,
            config_.arrival);
        output.right_stick = arrival_facing_stick(
            quat_mul_vec3(target.rotation, vec3(0.0F, 0.0F, 1.0F)),
            observation.camera_azimuth);
        return output;
    }
    case LearnedPickupState::FinalPreview: {
        if (observation.preview_results.size() != 1U ||
            observation.preview_snapshot_fingerprint !=
                observation.snapshot_fingerprint) {
            return fail(LearnedPickupFailureReason::FinalPreviewRejected);
        }
        const PickAssistPreviewResult& result =
            observation.preview_results.front();
        if (!same_root(result.request.root, final_root_) ||
            !result.preview.has_value() ||
            !result.preview->path_feasible ||
            !result.preview->match_ready ||
            !same_root(result.preview->prospective_root, final_root_)) {
            return fail(LearnedPickupFailureReason::FinalPreviewRejected);
        }
        learned_diagnostics_.state = LearnedPickupState::ReadyToSubmit;
        diagnostics_.state = PickAssistState::ReadyToSubmit;
        PickAssistOutput output = braking_output();
        output.submit_interact = true;
        return output;
    }
    case LearnedPickupState::ReadyToSubmit:
        return braking_output();
    case LearnedPickupState::Idle:
    case LearnedPickupState::Submitted:
    case LearnedPickupState::Failed:
        return {};
    }
    return {};
}

std::optional<PickRequest> LearnedSmartPickupBackend::take_submission(
    uint64_t request_id) {
    if (provider_ == nullptr) return authored_.take_submission(request_id);
    if (learned_diagnostics_.state != LearnedPickupState::ReadyToSubmit ||
        request_id == 0U) {
        return std::nullopt;
    }
    learned_diagnostics_.state = LearnedPickupState::Submitted;
    diagnostics_.state = PickAssistState::Submitted;
    learned_diagnostics_.armed = false;
    return PickRequest{
        start_.target_snapshot.handle,
        start_.affordance_id,
        request_id,
    };
}

bool LearnedSmartPickupBackend::active() const {
    if (provider_ == nullptr) return authored_.active();
    return learned_diagnostics_.state != LearnedPickupState::Idle &&
        learned_diagnostics_.state != LearnedPickupState::Submitted &&
        learned_diagnostics_.state != LearnedPickupState::Failed;
}

bool LearnedSmartPickupBackend::owns_manual_interact() const {
    return provider_ == nullptr
        ? authored_.owns_manual_interact()
        : active();
}

const PickAssistDiagnostics& LearnedSmartPickupBackend::diagnostics() const {
    return provider_ == nullptr ? authored_.diagnostics() : diagnostics_;
}

bool LearnedSmartPickupBackend::arm(uint64_t first_tick_index) {
    if (!artifact_.has_value()) return false;
    for (const FunnelProposal& proposal : artifact_->proposals()) {
        if (!proposal.accepted) continue;
        follower_.emplace(proposal.seed, proposal.samples, first_tick_index);
        learned_diagnostics_.armed = true;
        learned_diagnostics_.armed_seed = proposal.seed;
        refresh_follower_diagnostics();
        return true;
    }
    return false;
}

FunnelFollowerOutput LearnedSmartPickupBackend::follow(
    const FunnelFollowerInput& input) {
    if (!follower_.has_value()) {
        return {false, FunnelSample{}, FunnelFollowerState::Following};
    }
    FunnelFollowerOutput output = follower_->tick(input);
    refresh_follower_diagnostics();
    return output;
}

void LearnedSmartPickupBackend::refresh_follower_diagnostics() {
    if (!follower_.has_value()) return;
    const FunnelFollowerDiagnostics& follower = follower_->diagnostics();
    learned_diagnostics_.follower_state = follower.state;
    learned_diagnostics_.follower_cancel_reason = follower.cancel_reason;
    learned_diagnostics_.follower_published_count = follower.published_count;
}

PickAssistOutput LearnedSmartPickupBackend::fail(
    LearnedPickupFailureReason reason) {
    if (provider_ != nullptr &&
        learned_diagnostics_.state == LearnedPickupState::ProposalPending) {
        provider_->cancel();
    }
    learned_diagnostics_.state = LearnedPickupState::Failed;
    learned_diagnostics_.failure_reason = reason;
    learned_diagnostics_.armed = false;
    diagnostics_.state = PickAssistState::Failed;
    diagnostics_.reason = reason == LearnedPickupFailureReason::Cancelled
        ? PickAssistReason::Cancelled
        : PickAssistReason::InvalidGeometry;
    return {};
}

PickAssistOutput LearnedSmartPickupBackend::braking_output() const {
    PickAssistOutput output{};
    output.override_steering = true;
    output.stationary_constraint = true;
    return output;
}

bool LearnedSmartPickupBackend::authority_matches(
    const PickAssistObservation& observation) const {
    return observation.runtime_state == RuntimeState::Locomotion &&
        observation.target != nullptr &&
        same_interaction_target_snapshot(
            start_.target_snapshot, *observation.target) &&
        same_obstacles(start_.live_obstacles, observation.live_obstacles);
}

}  // namespace interaction
