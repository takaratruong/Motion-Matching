#pragma once

#include "interaction_place_target.h"
#include "interaction_target.h"

#include <array>
#include <cstdint>

namespace interaction {

struct SmartPickupSlotProvenance {
    uint32_t slot_id = 0U;
    const char* sequence_id = nullptr;
    int32_t entry_local_frame = -1;
    int32_t contact_local_frame = -1;
};

InteractionTarget make_smart_pickup_demo_target();
PlacementSurface make_smart_pickup_demo_destination_surface(
    const InteractionTarget& source_target);
const std::array<SmartPickupSlotProvenance, 3>&
smart_pickup_demo_slot_provenance();

}  // namespace interaction
