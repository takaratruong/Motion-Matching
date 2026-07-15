#pragma once

#include "interaction_matcher.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace interaction {

struct SurfaceHandle {
    uint64_t id = 0;
    uint32_t generation = 0;
    friend constexpr bool operator==(
        SurfaceHandle left,
        SurfaceHandle right) {
        return left.id == right.id && left.generation == right.generation;
    }
};

struct PlaceAffordance {
    uint32_t id = 0;
    Transform object_in_surface{};
    vec3 support_point_object{};
    vec3 approach_direction_surface{};
    float clearance_radius = 0.04F;
};

struct PlacementSurface {
    SurfaceHandle handle{};
    Transform surface_world{};
    Transform support_volume_world{};
    vec3 support_volume_size{};
    float half_extent_x_m = 0.0F;
    float half_extent_z_m = 0.0F;
    float overhead_clearance_m = 0.0F;
    std::vector<PlaceAffordance> affordances;
};

struct PlacementFit {
    bool accepted = false;
    Reason reason = Reason::None;
    float support_gap_m = 0.0F;
    float lowest_corner_m = 0.0F;
    float highest_corner_m = 0.0F;
    bool footprint_valid = false;
    bool overhead_valid = false;
};

class PlacementSurfaceRegistry {
public:
    SurfaceHandle upsert(PlacementSurface surface);
    const PlacementSurface* find(SurfaceHandle handle) const;
    const PlacementSurface* find_by_id(uint64_t id) const;
    const PlaceAffordance* find_affordance(
        SurfaceHandle handle,
        uint32_t affordance_id) const;
    std::optional<SurfaceHandle> resolve_single_surface(
        vec3 character_root,
        float maximum_distance_m) const;
private:
    std::vector<PlacementSurface> surfaces_;
};

struct PlaceRequest {
    TargetHandle held_target{};
    SurfaceHandle surface{};
    uint32_t affordance_id = 0;
    uint64_t request_id = 0;
    uint64_t selection_id = 0;
};

Transform placement_goal_world(
    const PlacementSurface& surface,
    const Transform& object_in_surface);
PlacementFit evaluate_placement_fit(
    const PlacementSurface& surface,
    const PlaceAffordance& affordance,
    ObjectLocalBounds object_bounds);
PlacementFit evaluate_actual_placement_fit(
    const PlacementSurface& surface,
    const PlaceAffordance& affordance,
    Transform actual_object_world,
    ObjectLocalBounds object_bounds);

}  // namespace interaction
