#pragma once

#include "interaction_target.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace interaction {

struct PickupSourceProvenance {
    int32_t clip_ordinal = -1;
    int32_t range_start = -1;
    int32_t range_stop = -1;
    int32_t entry_global_frame = -1;
    int32_t contact_global_frame = -1;
    int32_t lift_global_frame = -1;
    int32_t hold_global_frame = -1;
    Hand active_hand = Hand::Right;
    uint64_t object_profile_id = 0U;
    ObjectLocalBounds object_bounds{};
    Transform hand_in_object{};
    float source_support_height_m = 0.0F;
};

struct CertifiedPickupSourceIdentity {
    PickupSourceProvenance provenance{};
    std::string sequence_id{};
    std::string object_id{};
    int32_t reverse_start_global_frame = -1;
    std::string join_key_sha256{};
    uint64_t source_id = 0U;
};

struct CertifiedPickupSourceLocalRow {
    int32_t clip_ordinal = -1;
    int32_t range_start = -1;
    int32_t range_stop = -1;
    int32_t entry_local_frame = -1;
    int32_t contact_local_frame = -1;
    int32_t lift_local_frame = -1;
    int32_t hold_local_frame = -1;
    int32_t reverse_start_local_frame = -1;
    Hand active_hand = Hand::Right;
    uint64_t object_profile_id = 0U;
    ObjectLocalBounds object_bounds{};
    Transform hand_in_object{};
    float source_support_height_m = 0.0F;
    std::string sequence_id{};
    std::string object_id{};
    std::string join_key_sha256{};
    uint64_t source_id = 0U;
};

int32_t checked_pickup_global_to_local_frame(
    int32_t global_frame,
    int32_t range_start,
    int32_t range_stop);
int32_t checked_pickup_local_to_global_frame(
    int32_t local_frame,
    int32_t range_start,
    int32_t range_stop);
CertifiedPickupSourceIdentity make_certified_pickup_source_identity(
    const CertifiedPickupSourceLocalRow& row);

class CertifiedPickupSourceRegistry {
public:
    // Rows are already-global identities returned by the factory above.
    explicit CertifiedPickupSourceRegistry(
        std::vector<CertifiedPickupSourceIdentity> rows);
    std::optional<CertifiedPickupSourceIdentity> resolve_exact(
        const PickupSourceProvenance& actual) const;

private:
    std::vector<CertifiedPickupSourceIdentity> rows_{};
};

}  // namespace interaction
