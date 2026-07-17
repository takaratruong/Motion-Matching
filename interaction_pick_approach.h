#pragma once

#include "interaction_runtime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace interaction {

enum class PickEntrySlotIdentity : uint8_t { Plus, Minus };

struct PickEntrySlot {
    PickEntrySlotIdentity identity{};
    Transform waypoint{};
    PickEntryRoot prospective_root{};
    float hand_score = 0.0F;
};

struct PickEntrySlots {
    std::array<PickEntrySlot, 2> ordered{};
    float clearance_chord_m = 0.0F;
    float preserved_standoff_m = 0.0F;
};

Transform make_pick_reach_waypoint(
    const Database& database,
    const InteractionTarget& target,
    uint32_t clip_index = 0U);

PickEntrySlots make_pick_entry_slots(
    const Transform& reach_waypoint,
    const InteractionTarget& target);

std::optional<size_t> choose_pick_entry_slot(
    const PickEntrySlots& slots,
    const std::array<PickEntryPreview, 2>& previews,
    const std::array<bool, 2>& permitted,
    Hand active_hand);

const char* pick_entry_slot_name(PickEntrySlotIdentity identity);

}  // namespace interaction
