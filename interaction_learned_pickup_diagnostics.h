#pragma once

// Diagnostics for the learned Smart Pickup backend adapter.
//
// These expose the state of the learned-funnel seam that sits alongside the
// preserved authored pick-assist controller. They are observational only: the
// backend never writes the simulation root, displayed root, joints, the object
// registry, or attachment state.

#include "interaction_funnel_follower.h"

#include <cstdint>

namespace interaction {

struct LearnedPickupDiagnostics {
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
};

}  // namespace interaction
