#pragma once

#include "reach_coverage.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace reach {

size_t count_placed_root_azimuth_sectors(
    const Pack& pack,
    const std::vector<Candidate>& candidates,
    vec3 target_position,
    size_t sector_count);

uint64_t accepted_candidate_set_hash(
    const std::vector<Candidate>& candidates);

}  // namespace reach
