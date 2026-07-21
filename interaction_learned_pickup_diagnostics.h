#pragma once

// Diagnostics for the learned Smart Pickup backend adapter.
//
// These expose the state of the learned-funnel seam that sits alongside the
// preserved authored pick-assist controller. They are observational only: the
// backend never writes the simulation root, displayed root, joints, the object
// registry, or attachment state.

#include "interaction_funnel_follower.h"

#include <array>
#include <cstdint>

namespace interaction {

enum class LearnedPickupState : uint8_t {
    Idle,
    CoarseCapture,
    CaptureSettling,
    ProposalPending,
    SelectionPreview,
    AwaitEntry,
    FunnelFollow,
    FinalPreview,
    ReadyToSubmit,
    Submitted,
    Failed,
};

enum class LearnedPickupFailureReason : uint8_t {
    None,
    Cancelled,
    TargetChanged,
    RuntimeChanged,
    NoSafeCapture,
    InvalidCondition,
    ProposalLaunchFailed,
    ProposalTimeout,
    ProposalWorkerFailed,
    ProposalIdentityMismatch,
    NoAcceptedProposal,
    SelectionPreviewRejected,
    RouteBlocked,
    FollowerFailed,
    FinalPreviewRejected,
};

struct LearnedPickupDiagnostics {
    LearnedPickupState state = LearnedPickupState::Idle;
    LearnedPickupFailureReason failure_reason =
        LearnedPickupFailureReason::None;
    // True once an accepted proposal has been selected to drive the follower.
    bool armed = false;
    // Seed of the accepted proposal currently armed (0 when not armed).
    uint64_t armed_seed = 0U;
    // Number of accepted, certifiable proposals available in the artifact.
    int accepted_proposal_count = 0;
    // Live follower state mirrored from the armed follower.
    FunnelFollowerState follower_state = FunnelFollowerState::Following;
    FunnelCancelReason follower_cancel_reason = FunnelCancelReason::None;
    int follower_published_count = 0;
    int follower_progress_index = 0;
    int follower_lookahead_index = 0;
    uint32_t follower_terminal_settle_ticks = 0U;
    int selected_proposal_index = -1;
    uint32_t proposal_pending_ticks = 0U;
    std::array<float, kFunnelConditionDim> frozen_condition{};
};

}  // namespace interaction
