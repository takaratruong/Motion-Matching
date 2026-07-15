#include "interaction_place_target.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace interaction {
namespace {

constexpr float kMinimumNorm = 1.0e-6F;
constexpr float kUnitTolerance = 1.0e-3F;
constexpr double kMaximumSurfaceTiltRadians = 0.087266463;
constexpr double kMaximumTopPositionErrorM = 0.001;
constexpr double kMaximumTopOrientationErrorRadians = 0.001745329;
constexpr float kMinimumSupportGapM = -0.005F;
constexpr float kMaximumSupportGapM = 0.020F;
constexpr float kGeometryRoundoffM = 1.0e-7F;
constexpr double kAngleRoundoffRadians = 1.0e-7;

bool finite(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

bool finite(vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(quat value) {
    return finite(value.w) && finite(value.x) && finite(value.y) &&
           finite(value.z);
}

bool valid_rotation(quat value) {
    if (!finite(value)) return false;
    const float norm = quat_length(value);
    return finite(norm) && norm > kMinimumNorm &&
           std::abs(norm - 1.0F) <= kUnitTolerance;
}

void validate_transform(const Transform& transform, const char* label) {
    if (!finite(transform.position) || !valid_rotation(transform.rotation)) {
        throw std::invalid_argument(
            std::string("placement surface invalid ") + label);
    }
}

void validate_positive(vec3 value, const char* label) {
    if (!finite(value) || !(value.x > 0.0F) || !(value.y > 0.0F) ||
        !(value.z > 0.0F)) {
        throw std::invalid_argument(
            std::string("placement surface invalid ") + label);
    }
}

void validate_bounds(ObjectLocalBounds bounds) {
    if (!finite(bounds.center_object)) {
        throw std::invalid_argument(
            "placement object bounds center must be finite");
    }
    validate_positive(
        bounds.half_extents_object,
        "object bounds half extents");
}

double quaternion_angle(quat left, quat right) {
    const double left_norm = std::sqrt(
        static_cast<double>(left.w) * left.w +
        static_cast<double>(left.x) * left.x +
        static_cast<double>(left.y) * left.y +
        static_cast<double>(left.z) * left.z);
    const double right_norm = std::sqrt(
        static_cast<double>(right.w) * right.w +
        static_cast<double>(right.x) * right.x +
        static_cast<double>(right.y) * right.y +
        static_cast<double>(right.z) * right.z);
    const double cosine = std::clamp(
        std::abs(
            (static_cast<double>(left.w) * right.w +
             static_cast<double>(left.x) * right.x +
             static_cast<double>(left.y) * right.y +
             static_cast<double>(left.z) * right.z) /
            (left_norm * right_norm)),
        0.0,
        1.0);
    return 2.0 * std::acos(cosine);
}

double surface_tilt(quat rotation) {
    const double norm = std::sqrt(
        static_cast<double>(rotation.w) * rotation.w +
        static_cast<double>(rotation.x) * rotation.x +
        static_cast<double>(rotation.y) * rotation.y +
        static_cast<double>(rotation.z) * rotation.z);
    const double x = rotation.x / norm;
    const double z = rotation.z / norm;
    const double normal_y = std::clamp(
        1.0 - 2.0 * (x * x + z * z), -1.0, 1.0);
    return std::acos(normal_y);
}

double distance(vec3 left, vec3 right) {
    const double x = static_cast<double>(left.x) - right.x;
    const double y = static_cast<double>(left.y) - right.y;
    const double z = static_cast<double>(left.z) - right.z;
    return std::sqrt(x * x + y * y + z * z);
}

void validate_affordance(const PlaceAffordance& affordance) {
    if (affordance.id == 0U) {
        throw std::invalid_argument(
            "placement affordance ID must be nonzero");
    }
    validate_transform(
        affordance.object_in_surface,
        "affordance object transform");
    if (!finite(affordance.support_point_object)) {
        throw std::invalid_argument(
            "placement support point must be finite");
    }
    const float approach_length = length(
        affordance.approach_direction_surface);
    if (!finite(affordance.approach_direction_surface) ||
        !finite(approach_length) || !(approach_length > kMinimumNorm) ||
        std::abs(approach_length - 1.0F) > kUnitTolerance) {
        throw std::invalid_argument(
            "placement approach direction must be unit length");
    }
    if (!finite(affordance.clearance_radius) ||
        affordance.clearance_radius < 0.0F) {
        throw std::invalid_argument(
            "placement clearance must be finite and nonnegative");
    }
}

void validate_surface(const PlacementSurface& surface) {
    if (surface.handle.id == 0U) {
        throw std::invalid_argument(
            "placement surface ID must be nonzero");
    }
    if (surface.handle.generation == 0U) {
        throw std::invalid_argument(
            "placement surface generation must be nonzero");
    }
    validate_transform(surface.surface_world, "surface transform");
    validate_transform(
        surface.support_volume_world,
        "support volume transform");
    validate_positive(surface.support_volume_size, "support volume size");
    if (!finite(surface.half_extent_x_m) ||
        !(surface.half_extent_x_m > 0.0F) ||
        !finite(surface.half_extent_z_m) ||
        !(surface.half_extent_z_m > 0.0F)) {
        throw std::invalid_argument(
            "placement usable half extents must be finite and positive");
    }
    if (surface.half_extent_x_m > 0.5F * surface.support_volume_size.x ||
        surface.half_extent_z_m > 0.5F * surface.support_volume_size.z) {
        throw std::invalid_argument(
            "placement usable extents exceed physical support");
    }
    if (!finite(surface.overhead_clearance_m) ||
        !(surface.overhead_clearance_m > 0.0F)) {
        throw std::invalid_argument(
            "placement overhead clearance must be finite and positive");
    }
    if (surface_tilt(surface.surface_world.rotation) >
        kMaximumSurfaceTiltRadians + kAngleRoundoffRadians) {
        throw std::invalid_argument(
            "placement surface tilt exceeds five degrees");
    }

    const Transform volume_top = compose(
        surface.support_volume_world,
        Transform{
            vec3(0.0F, 0.5F * surface.support_volume_size.y, 0.0F),
            quat()});
    if (distance(volume_top.position, surface.surface_world.position) >
            kMaximumTopPositionErrorM ||
        quaternion_angle(
            volume_top.rotation,
            surface.surface_world.rotation) >
            kMaximumTopOrientationErrorRadians + kAngleRoundoffRadians) {
        throw std::invalid_argument(
            "placement support volume top does not match support plane");
    }
    if (surface.affordances.empty()) {
        throw std::invalid_argument(
            "placement surface must have an affordance");
    }
    for (size_t left = 0; left < surface.affordances.size(); ++left) {
        validate_affordance(surface.affordances[left]);
        for (size_t right = left + 1U;
             right < surface.affordances.size();
             ++right) {
            if (surface.affordances[left].id ==
                surface.affordances[right].id) {
                throw std::invalid_argument(
                    "placement affordance IDs must be unique");
            }
        }
    }
}

SurfaceHandle incremented(SurfaceHandle handle) {
    if (handle.generation == std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("placement surface generation overflow");
    }
    ++handle.generation;
    return handle;
}

float snap_boundary(float value, float boundary) {
    return std::abs(value - boundary) <= kGeometryRoundoffM
        ? boundary
        : value;
}

PlacementFit evaluate_local_fit(
    const PlacementSurface& surface,
    const PlaceAffordance& affordance,
    const Transform& object_in_surface,
    ObjectLocalBounds bounds) {
    float minimum_x = std::numeric_limits<float>::infinity();
    float maximum_x = -std::numeric_limits<float>::infinity();
    float minimum_z = std::numeric_limits<float>::infinity();
    float maximum_z = -std::numeric_limits<float>::infinity();
    float lowest = std::numeric_limits<float>::infinity();
    float highest = -std::numeric_limits<float>::infinity();
    for (int sign_x : {-1, 1}) {
        for (int sign_y : {-1, 1}) {
            for (int sign_z : {-1, 1}) {
                const vec3 corner_object =
                    bounds.center_object + vec3(
                        static_cast<float>(sign_x) *
                            bounds.half_extents_object.x,
                        static_cast<float>(sign_y) *
                            bounds.half_extents_object.y,
                        static_cast<float>(sign_z) *
                            bounds.half_extents_object.z);
                const vec3 corner_surface =
                    object_in_surface.position + quat_mul_vec3(
                        object_in_surface.rotation,
                        corner_object);
                minimum_x = std::min(minimum_x, corner_surface.x);
                maximum_x = std::max(maximum_x, corner_surface.x);
                minimum_z = std::min(minimum_z, corner_surface.z);
                maximum_z = std::max(maximum_z, corner_surface.z);
                lowest = std::min(lowest, corner_surface.y);
                highest = std::max(highest, corner_surface.y);
            }
        }
    }

    const vec3 support_surface =
        object_in_surface.position + quat_mul_vec3(
            object_in_surface.rotation,
            affordance.support_point_object);
    const float support_gap = snap_boundary(
        snap_boundary(support_surface.y, kMinimumSupportGapM),
        kMaximumSupportGapM);
    lowest = snap_boundary(lowest, kMinimumSupportGapM);
    highest = snap_boundary(highest, surface.overhead_clearance_m);

    PlacementFit fit{};
    fit.support_gap_m = support_gap;
    fit.lowest_corner_m = lowest;
    fit.highest_corner_m = highest;
    fit.footprint_valid =
        minimum_x - affordance.clearance_radius >=
            -surface.half_extent_x_m - kGeometryRoundoffM &&
        maximum_x + affordance.clearance_radius <=
            surface.half_extent_x_m + kGeometryRoundoffM &&
        minimum_z - affordance.clearance_radius >=
            -surface.half_extent_z_m - kGeometryRoundoffM &&
        maximum_z + affordance.clearance_radius <=
            surface.half_extent_z_m + kGeometryRoundoffM;
    const bool lowest_corner_valid = lowest >= kMinimumSupportGapM;
    fit.overhead_valid = highest <= surface.overhead_clearance_m;
    const bool support_valid =
        support_gap >= kMinimumSupportGapM &&
        support_gap <= kMaximumSupportGapM;
    fit.accepted =
        fit.footprint_valid && lowest_corner_valid &&
        fit.overhead_valid && support_valid;
    fit.reason = fit.accepted ? Reason::None : Reason::PlacementOutOfBounds;
    return fit;
}

}  // namespace

SurfaceHandle PlacementSurfaceRegistry::upsert(PlacementSurface surface) {
    validate_surface(surface);
    for (PlacementSurface& stored : surfaces_) {
        if (stored.handle.id != surface.handle.id) continue;
        surface.handle = incremented(stored.handle);
        stored = std::move(surface);
        return stored.handle;
    }
    surfaces_.push_back(std::move(surface));
    return surfaces_.back().handle;
}

const PlacementSurface* PlacementSurfaceRegistry::find(
    SurfaceHandle handle) const {
    for (const PlacementSurface& surface : surfaces_) {
        if (surface.handle == handle) return &surface;
    }
    return nullptr;
}

const PlacementSurface* PlacementSurfaceRegistry::find_by_id(
    uint64_t id) const {
    for (const PlacementSurface& surface : surfaces_) {
        if (surface.handle.id == id) return &surface;
    }
    return nullptr;
}

const PlaceAffordance* PlacementSurfaceRegistry::find_affordance(
    SurfaceHandle handle,
    uint32_t affordance_id) const {
    const PlacementSurface* surface = find(handle);
    if (surface == nullptr) return nullptr;
    for (const PlaceAffordance& affordance : surface->affordances) {
        if (affordance.id == affordance_id) return &affordance;
    }
    return nullptr;
}

std::optional<SurfaceHandle> PlacementSurfaceRegistry::resolve_single_surface(
    vec3 character_root,
    float maximum_distance_m) const {
    if (!finite(character_root) || !finite(maximum_distance_m) ||
        maximum_distance_m < 0.0F) {
        throw std::invalid_argument(
            "placement resolver arguments must be finite and distance nonnegative");
    }

    std::optional<SurfaceHandle> resolved;
    for (const PlacementSurface& surface : surfaces_) {
        bool in_range = false;
        for (const PlaceAffordance& affordance : surface.affordances) {
            const Transform goal = compose(
                surface.surface_world,
                affordance.object_in_surface);
            const float delta_x = goal.position.x - character_root.x;
            const float delta_z = goal.position.z - character_root.z;
            if (std::hypot(delta_x, delta_z) <= maximum_distance_m) {
                in_range = true;
                break;
            }
        }
        if (!in_range) continue;
        if (resolved.has_value()) return std::nullopt;
        resolved = surface.handle;
    }
    return resolved;
}

Transform placement_goal_world(
    const PlacementSurface& surface,
    const Transform& object_in_surface) {
    validate_transform(surface.surface_world, "surface transform");
    validate_transform(object_in_surface, "object-in-surface transform");
    return compose(surface.surface_world, object_in_surface);
}

PlacementFit evaluate_placement_fit(
    const PlacementSurface& surface,
    const PlaceAffordance& affordance,
    ObjectLocalBounds object_bounds) {
    validate_surface(surface);
    validate_affordance(affordance);
    validate_bounds(object_bounds);
    return evaluate_local_fit(
        surface,
        affordance,
        affordance.object_in_surface,
        object_bounds);
}

PlacementFit evaluate_actual_placement_fit(
    const PlacementSurface& surface,
    const PlaceAffordance& affordance,
    Transform actual_object_world,
    ObjectLocalBounds object_bounds) {
    validate_surface(surface);
    validate_affordance(affordance);
    validate_transform(actual_object_world, "actual object transform");
    validate_bounds(object_bounds);
    return evaluate_local_fit(
        surface,
        affordance,
        compose(inverse(surface.surface_world), actual_object_world),
        object_bounds);
}

}  // namespace interaction
