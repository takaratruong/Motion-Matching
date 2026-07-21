#pragma once

#include "interaction_funnel_capture.h"

#include <vector>

namespace interaction {

struct FunnelApproachPlan {
    bool feasible = false;
    Transform entry_world{};
    std::vector<Transform> waypoints_world{};
};

FunnelApproachPlan plan_funnel_approach(
    Transform frozen_root,
    const InteractionTarget& target,
    const std::vector<PickNavigationObstacle>& obstacles,
    const FunnelCaptureConfig& capture = {});

}  // namespace interaction
