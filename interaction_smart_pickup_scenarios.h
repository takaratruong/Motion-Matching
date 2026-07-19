#pragma once

#include "interaction_pick_slots.h"
#include "interaction_pickup_provenance.h"
#include "interaction_runtime.h"

#include <array>
#include <cstdint>
#include <vector>

namespace interaction {

inline constexpr uint32_t kSmartPickupDemoRepeatCount = 10U;

enum class SmartPickupScenarioClass : uint8_t {
    Clear,
    AlternateSlot,
    AllBlocked,
};

struct SmartPickupDemoScenario {
    const char* id = nullptr;
    Transform initial_root_world{};
    Transform scene_from_authored{};
    std::vector<PickNavigationObstacle> obstacles{};
    SmartPickupScenarioClass expected_class =
        SmartPickupScenarioClass::Clear;
};

enum class SmartPickupTargetCompatibility : uint8_t {
    Compatible,
    MissingAffordance,
    UnsupportedHand,
    NoAuthoredSlot,
};

struct SmartPickupLifecycleWitness {
    uint64_t request_id = 0U;
    TargetHandle target_before{};
    TargetHandle target_after{};
    uint64_t owner_request = 0U;
    RuntimeState terminal_state = RuntimeState::Disabled;
    ObjectState terminal_object_state = ObjectState::Free;
    uint32_t request_count = 0U;
    uint32_t attachment_edges = 0U;
    uint32_t release_edges = 0U;
    bool attached = false;
    bool post_carry_tick_observed = false;
    bool grasp_preserved = false;
    float carry_root_displacement_m = 0.0F;
    float carry_object_displacement_m = 0.0F;
    CertifiedPickupSourceIdentity actual_source{};
};

const std::array<SmartPickupDemoScenario, 7>&
smart_pickup_demo_scenarios();
SmartPickupTargetCompatibility classify_smart_pickup_target(
    const InteractionTarget& target,
    uint32_t affordance_id);
const char* smart_pickup_target_compatibility_name(
    SmartPickupTargetCompatibility value);
SmartPickupLifecycleWitness run_smart_pickup_lifecycle(
    const Database& database,
    const Features& features,
    const LocomotionSnapshot& published_flat_snapshot,
    const SmartPickupDemoScenario& scenario,
    const CertifiedPickupSourceRegistry* pickup_source_registry = nullptr);

}  // namespace interaction
