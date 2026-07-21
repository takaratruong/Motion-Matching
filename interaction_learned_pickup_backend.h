#pragma once

// Learned Smart Pickup backend adapter.
//
// LearnedSmartPickupBackend implements the existing SmartPickupAssistBackend
// interface by delegating every authored method to a preserved
// ControllerPickAssist instance, so authored Smart Pickup behavior and its
// tests are untouched. Alongside that, it owns a strict learned-funnel artifact
// and a pure 25 Hz follower, exposing an arm/follow seam that replaces authored
// entry-slot motion with offline diffusion proposals.
//
// The adapter is observational with respect to world state: it never writes the
// simulation root, displayed root, joints, the object registry, or attachment
// state. Callers apply the follower's published targets.

#include "interaction_funnel_artifact.h"
#include "interaction_funnel_capture.h"
#include "interaction_funnel_follower.h"
#include "interaction_funnel_plan.h"
#include "interaction_funnel_worker.h"
#include "interaction_learned_pickup_diagnostics.h"
#include "interaction_pick_assist.h"
#include "interaction_smart_pickup_controller.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace interaction {

struct LearnedPickupConfig {
    FunnelCaptureConfig capture{};
    ArrivalConfig arrival{};
    uint32_t maximum_proposal_pending_ticks = 250U;
    size_t maximum_preview_proposals = 3U;
    float capture_position_tolerance_m = 0.03F;
    float handoff_position_tolerance_m = 0.12F;
    float capture_yaw_tolerance_radians = 20.0F * PIf / 180.0F;
    float prefetch_minimum_planar_speed_mps = 0.05F;
    float supported_minimum_entry_radius_m = 0.45F;
    float supported_maximum_entry_radius_m = 0.75F;
    float supported_minimum_entry_speed_mps = 0.05F;
    float supported_maximum_entry_speed_mps = 0.30F;
    uint64_t batch_seed = 2026071901U;
    std::array<uint8_t, 32> checkpoint_sha256{};
};

class LearnedSmartPickupBackend final : public SmartPickupAssistBackend {
public:
    explicit LearnedSmartPickupBackend(InteractionFunnelArtifact artifact);
    LearnedSmartPickupBackend(
        FunnelProposalProvider& provider,
        LearnedPickupConfig config);

    // Authored SmartPickupAssistBackend interface, delegated unchanged.
    bool begin(
        const PickAssistStart& start,
        const InteractionTarget* post_step_target) override;
    void cancel() override;
    PickAssistOutput observe(const PickAssistObservation& observation) override;
    std::optional<PickRequest> take_submission(uint64_t request_id) override;
    std::optional<PickEntryPreview> take_certified_preview() override;
    bool active() const override;
    bool owns_manual_interact() const override;
    const PickAssistDiagnostics& diagnostics() const override;
    std::optional<LearnedPickupDebugSnapshot>
    learned_debug_snapshot() const override;

    // Learned-funnel seam. arm() selects the first accepted proposal in the
    // artifact and binds a fresh follower to the given first tick index; it
    // returns false when no accepted proposal exists. follow() advances the
    // armed follower one 25 Hz tick (publishing nothing when not armed).
    bool arm(uint64_t first_tick_index);
    FunnelFollowerOutput follow(const FunnelFollowerInput& input);

    const LearnedPickupDiagnostics& learned_diagnostics() const {
        return learned_diagnostics_;
    }

private:
    void refresh_follower_diagnostics();
    PickAssistOutput observe_learned(const PickAssistObservation& observation);
    PickAssistOutput consume_proposal_poll(
        const PickAssistObservation& observation,
        FunnelProposalPoll poll);
    PickAssistOutput coarse_capture_output(
        const PickAssistObservation& observation) const;
    PickAssistOutput fail(LearnedPickupFailureReason reason);
    PickAssistOutput braking_output() const;
    bool authority_matches(const PickAssistObservation& observation) const;

    std::optional<InteractionFunnelArtifact> artifact_{};
    FunnelProposalProvider* provider_ = nullptr;
    LearnedPickupConfig config_{};
    ControllerPickAssist authored_{};
    PickAssistStart start_{};
    FunnelCaptureSelection capture_{};
    FunnelApproachPlan approach_plan_{};
    Transform frozen_root_world_{};
    Transform frozen_entry_world_{};
    Transform frozen_object_world_{};
    Transform tracked_root_world_{};
    FunnelProposalRequest proposal_request_{};
    std::vector<PickAssistPreviewRequest> selection_requests_{};
    std::vector<size_t> selection_proposal_indices_{};
    FunnelRoute selected_route_object_{};
    std::vector<FunnelSample> selected_world_targets_{};
    PickEntryRoot final_root_{};
    std::optional<PickEntryPreview> selected_preview_{};
    std::optional<InteractionFunnelFollower> follower_{};
    PickAssistDiagnostics diagnostics_{};
    LearnedPickupDiagnostics learned_diagnostics_{};
};

}  // namespace interaction
