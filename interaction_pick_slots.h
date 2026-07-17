#pragma once

#include "interaction_target.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace interaction {

enum class PickSlotReason : uint8_t {
    None,
    NoAuthoredSlot,
    InvalidGeometry,
    OutsideTravelEnvelope,
    TableBlocked,
    ObstacleBlocked,
    AllSlotsBlocked,
};

struct PickNavigationObstacle {
    vec3 center_world{};
    vec3 size_world{};
};

struct MappedPickSlot {
    uint32_t id = 0U;
    Transform root_world{};
    float route_length_m = 0.0F;
    float heading_change_radians = 0.0F;
    float object_origin_distance_m = 0.0F;
    float object_bounds_center_distance_m = 0.0F;
    uint64_t route_millimetres = 0U;
    uint64_t heading_milliradians = 0U;
    PickSlotReason reason = PickSlotReason::InvalidGeometry;
    int32_t obstacle_index = -1;
};

struct PickSlotSelection {
    std::vector<MappedPickSlot> ordered{};
    std::optional<size_t> selected_index{};
    PickSlotReason reason = PickSlotReason::NoAuthoredSlot;
};

struct PickSlotConfig {
    float maximum_direct_travel_m = 1.00F;
    float travel_tolerance_m = 2.0e-5F;
    float table_root_expansion_m = 0.24F;
    float obstacle_root_radius_m = 0.60F;
    float obstacle_safety_margin_m = 2.0e-5F;
};

PickSlotSelection select_pick_slot(
    Transform live_root,
    const InteractionTarget& target,
    const GraspAffordance& affordance,
    const std::vector<PickNavigationObstacle>& obstacles,
    const PickSlotConfig& config = {});

PickSlotReason revalidate_frozen_pick_slot(
    Transform live_root,
    Transform frozen_root,
    const InteractionTarget& target,
    const std::vector<PickNavigationObstacle>& obstacles,
    const PickSlotConfig& config = {});

}  // namespace interaction
