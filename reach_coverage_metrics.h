#pragma once

#include "reach_coverage.h"

#include <cstddef>
#include <vector>

namespace reach {

size_t count_placed_root_azimuth_sectors(
    const Pack& pack,
    const std::vector<Candidate>& candidates,
    vec3 target_position,
    size_t sector_count);

}  // namespace reach
