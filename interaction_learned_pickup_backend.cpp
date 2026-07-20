#include "interaction_learned_pickup_backend.h"

#include <utility>

namespace interaction {

LearnedSmartPickupBackend::LearnedSmartPickupBackend(InteractionFunnelArtifact artifact)
    : artifact_(std::move(artifact)) {
    int accepted = 0;
    for (const FunnelProposal& proposal : artifact_.proposals()) {
        if (proposal.accepted) ++accepted;
    }
    learned_diagnostics_.accepted_proposal_count = accepted;
}

bool LearnedSmartPickupBackend::begin(
    const PickAssistStart& start,
    const InteractionTarget* post_step_target) {
    return authored_.begin(start, post_step_target);
}

void LearnedSmartPickupBackend::cancel() {
    authored_.cancel();
}

PickAssistOutput LearnedSmartPickupBackend::observe(
    const PickAssistObservation& observation) {
    return authored_.observe(observation);
}

std::optional<PickRequest> LearnedSmartPickupBackend::take_submission(
    uint64_t request_id) {
    return authored_.take_submission(request_id);
}

bool LearnedSmartPickupBackend::active() const {
    return authored_.active();
}

bool LearnedSmartPickupBackend::owns_manual_interact() const {
    return authored_.owns_manual_interact();
}

const PickAssistDiagnostics& LearnedSmartPickupBackend::diagnostics() const {
    return authored_.diagnostics();
}

bool LearnedSmartPickupBackend::arm(uint64_t first_tick_index) {
    for (const FunnelProposal& proposal : artifact_.proposals()) {
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

}  // namespace interaction
