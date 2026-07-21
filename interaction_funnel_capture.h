#pragma once

#include "interaction_pick_slots.h"

#include <array>
#include <optional>
#include <vector>

namespace interaction {

struct FunnelCaptureConfig {
    float minimum_object_radius_m = 0.45F;
    float maximum_object_radius_m = 1.00F;
    float maximum_activation_object_radius_m = 3.00F;
    PickSlotConfig route{};
};

struct FunnelCaptureSelection {
    Transform target_world{};
    int candidate_index = -1;
    PickSlotReason reason = PickSlotReason::InvalidGeometry;
};

FunnelCaptureSelection select_funnel_capture(
    Transform live_root,
    const InteractionTarget& target,
    const std::vector<PickNavigationObstacle>& obstacles,
    const FunnelCaptureConfig& config = {});

std::optional<std::array<float, 24>> build_funnel_condition(
    Transform entry_root_world,
    vec3 simulation_velocity_world,
    const InteractionTarget& target,
    const GraspAffordance& affordance);

}  // namespace interaction
