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
#include "interaction_funnel_follower.h"
#include "interaction_learned_pickup_diagnostics.h"
#include "interaction_pick_assist.h"
#include "interaction_smart_pickup_controller.h"

#include <cstdint>
#include <optional>

namespace interaction {

class LearnedSmartPickupBackend final : public SmartPickupAssistBackend {
public:
    explicit LearnedSmartPickupBackend(InteractionFunnelArtifact artifact);

    // Authored SmartPickupAssistBackend interface, delegated unchanged.
    bool begin(
        const PickAssistStart& start,
        const InteractionTarget* post_step_target) override;
    void cancel() override;
    PickAssistOutput observe(const PickAssistObservation& observation) override;
    std::optional<PickRequest> take_submission(uint64_t request_id) override;
    bool active() const override;
    bool owns_manual_interact() const override;
    const PickAssistDiagnostics& diagnostics() const override;

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

    InteractionFunnelArtifact artifact_;
    ControllerPickAssist authored_{};
    std::optional<InteractionFunnelFollower> follower_{};
    LearnedPickupDiagnostics learned_diagnostics_{};
};

}  // namespace interaction
