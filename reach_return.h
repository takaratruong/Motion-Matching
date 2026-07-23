#pragma once

#include "reach_search.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace reach {

enum class ReturnRejection : uint8_t {
    None,
    InvalidSolver,
    Seam,
    ObjectCollision,
    EnvironmentCollision,
};

struct ShapedReturn {
    std::vector<interaction::Pose> poses;
    ReturnRejection rejection = ReturnRejection::None;
    size_t rejected_sample = 0U;
};

ShapedReturn shape_recorded_return(
    const Pack& pack,
    const Candidate& candidate,
    const Query& query,
    const interaction::Pose& solved_contact,
    interaction::Transform hand_in_object,
    vec3 object_dimensions,
    const interaction::EnvironmentGeometry& environment,
    const SearchConfig& config);

}  // namespace reach
