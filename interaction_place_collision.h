#pragma once

#include "interaction_place_target.h"

#include <array>

namespace interaction {
namespace place_collision {

bool object_intersects_support(
    Transform object_world,
    ObjectLocalBounds bounds,
    const PlacementSurface& surface);
std::array<vec3, 8> object_corners_in_support(
    Transform object_world,
    ObjectLocalBounds bounds,
    const PlacementSurface& surface);
double maximum_corner_radius(ObjectLocalBounds bounds);
bool swept_interval_intersects_support(
    Transform previous,
    Transform current,
    ObjectLocalBounds bounds,
    const PlacementSurface& surface);

}  // namespace place_collision
}  // namespace interaction
