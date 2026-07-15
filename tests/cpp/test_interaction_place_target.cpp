#include "interaction_place_target.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace {

constexpr float kTolerance = 1.0e-5F;
constexpr float kFiveDegrees = 0.087266463F;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

#define TEST_CHECK(condition) \
    require(static_cast<bool>(condition), #condition)

template<class Function>
bool throws_invalid_argument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
    }
    return false;
}

template<class Function>
bool throws_overflow_error(Function&& function) {
    try {
        function();
    } catch (const std::overflow_error&) {
        return true;
    } catch (...) {
    }
    return false;
}

bool near(float left, float right, float tolerance = kTolerance) {
    return std::abs(left - right) <= tolerance;
}

bool near(vec3 left, vec3 right, float tolerance = kTolerance) {
    return near(left.x, right.x, tolerance) &&
           near(left.y, right.y, tolerance) &&
           near(left.z, right.z, tolerance);
}

bool near(quat left, quat right, float tolerance = kTolerance) {
    return quat_angle_between(left, right) <= tolerance;
}

bool near(
    const interaction::Transform& left,
    const interaction::Transform& right,
    float tolerance = kTolerance) {
    return near(left.position, right.position, tolerance) &&
           near(left.rotation, right.rotation, tolerance);
}

void align_support_volume(interaction::PlacementSurface& surface) {
    surface.support_volume_world.rotation = surface.surface_world.rotation;
    surface.support_volume_world.position =
        surface.surface_world.position - quat_mul_vec3(
            surface.surface_world.rotation,
            vec3(0.0F, 0.5F * surface.support_volume_size.y, 0.0F));
}

interaction::ObjectLocalBounds make_bounds() {
    return {
        vec3(0.03F, 0.01F, -0.02F),
        vec3(0.10F, 0.15F, 0.10F),
    };
}

interaction::PlacementSurface make_surface(
    uint64_t id = 41U,
    uint32_t generation = 1U) {
    using namespace interaction;
    PlacementSurface surface{};
    surface.handle = {id, generation};
    surface.surface_world = {vec3(0.0F, 0.66F, 3.0F), quat()};
    surface.support_volume_size = vec3(0.80F, 0.10F, 1.20F);
    surface.half_extent_x_m = 0.40F;
    surface.half_extent_z_m = 0.60F;
    surface.overhead_clearance_m = 1.0F;
    align_support_volume(surface);
    PlaceAffordance affordance{};
    affordance.id = 7U;
    affordance.object_in_surface = {
        vec3(0.25F, 0.16F, 0.50F), quat()};
    affordance.support_point_object = vec3(0.03F, -0.14F, -0.02F);
    affordance.approach_direction_surface = vec3(0.0F, 0.0F, -1.0F);
    affordance.clearance_radius = 0.02F;
    surface.affordances = {affordance};
    return surface;
}

void test_public_contract_and_reason_abi() {
    using namespace interaction;

    static_assert(std::is_same_v<
        decltype(PlaceAffordance{}.object_in_surface), Transform>);
    static_assert(std::is_same_v<
        decltype(PlacementSurface{}.affordances),
        std::vector<PlaceAffordance>>);
    static_assert(std::is_same_v<
        decltype(PlaceRequest{}.held_target), TargetHandle>);
    static_assert(std::is_same_v<
        decltype(PlaceRequest{}.surface), SurfaceHandle>);
    static_assert(std::is_same_v<
        decltype(PlaceRequest{}.affordance_id), uint32_t>);
    static_assert(std::is_same_v<
        decltype(PlaceRequest{}.request_id), uint64_t>);
    static_assert(std::is_same_v<
        decltype(PlaceRequest{}.selection_id), uint64_t>);
    static_assert(
        static_cast<uint8_t>(Reason::SurfaceUnavailable) ==
        static_cast<uint8_t>(Reason::Reset) + 1U);
    static_assert(
        static_cast<uint8_t>(Reason::SurfaceChanged) ==
        static_cast<uint8_t>(Reason::SurfaceUnavailable) + 1U);
    static_assert(
        static_cast<uint8_t>(Reason::PlacementOutOfBounds) ==
        static_cast<uint8_t>(Reason::SurfaceChanged) + 1U);
    static_assert(
        static_cast<uint8_t>(Reason::ReleasePosition) ==
        static_cast<uint8_t>(Reason::PlacementOutOfBounds) + 1U);
    static_assert(
        static_cast<uint8_t>(Reason::ReleaseOrientation) ==
        static_cast<uint8_t>(Reason::ReleasePosition) + 1U);

    const PlaceRequest request{
        TargetHandle{9U, 3U}, SurfaceHandle{41U, 2U}, 7U, 88U, 99U};
    TEST_CHECK(request.held_target == (TargetHandle{9U, 3U}));
    TEST_CHECK(request.surface == (SurfaceHandle{41U, 2U}));
    TEST_CHECK(request.affordance_id == 7U);
    TEST_CHECK(request.request_id == 88U);
    TEST_CHECK(request.selection_id == 99U);
    TEST_CHECK(!((SurfaceHandle{41U, 1U}) == (SurfaceHandle{41U, 2U})));
}

void test_registry_lookup_goal_and_generation() {
    using namespace interaction;

    PlacementSurfaceRegistry registry;
    const SurfaceHandle first = registry.upsert(make_surface());
    TEST_CHECK(first == (SurfaceHandle{41U, 1U}));
    TEST_CHECK(registry.find(first) != nullptr);
    TEST_CHECK(registry.find_by_id(41U) == registry.find(first));
    TEST_CHECK(registry.find_by_id(99U) == nullptr);
    TEST_CHECK(registry.find_affordance(first, 7U) != nullptr);
    TEST_CHECK(registry.find_affordance(first, 8U) == nullptr);

    const Transform goal = placement_goal_world(
        *registry.find(first),
        registry.find_affordance(first, 7U)->object_in_surface);
    TEST_CHECK(near(goal.position, vec3(0.25F, 0.82F, 3.50F)));

    PlacementSurface replacement = make_surface(41U, 999U);
    replacement.surface_world.position.x = 2.0F;
    align_support_volume(replacement);
    const SurfaceHandle second = registry.upsert(replacement);
    TEST_CHECK(second == (SurfaceHandle{41U, 2U}));
    TEST_CHECK(registry.find(first) == nullptr);
    TEST_CHECK(registry.find_affordance(first, 7U) == nullptr);
    TEST_CHECK(registry.find(second) != nullptr);
    TEST_CHECK(registry.find_by_id(41U)->handle == second);
    TEST_CHECK(near(registry.find(second)->surface_world.position.x, 2.0F));

    constexpr uint32_t maximum = std::numeric_limits<uint32_t>::max();
    PlacementSurfaceRegistry overflow;
    const SurfaceHandle maximum_handle =
        overflow.upsert(make_surface(70U, maximum));
    TEST_CHECK(throws_overflow_error([&] {
        (void)overflow.upsert(make_surface(70U, 1U));
    }));
    TEST_CHECK(overflow.find(maximum_handle) != nullptr);
}

void test_surface_validation_is_strict_and_transactional() {
    using namespace interaction;
    struct InvalidSurfaceCase {
        void (*mutate)(PlacementSurface&);
    };
    const InvalidSurfaceCase invalid_cases[] = {
        {+[](PlacementSurface& value) { value.handle.id = 0U; }},
        {+[](PlacementSurface& value) { value.handle.generation = 0U; }},
        {+[](PlacementSurface& value) {
            value.surface_world.position.x =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](PlacementSurface& value) {
            value.surface_world.rotation = quat(0.0F, 0.0F, 0.0F, 0.0F);
        }},
        {+[](PlacementSurface& value) {
            value.surface_world.rotation = quat(2.0F, 0.0F, 0.0F, 0.0F);
        }},
        {+[](PlacementSurface& value) {
            value.support_volume_world.position.z =
                std::numeric_limits<float>::infinity();
        }},
        {+[](PlacementSurface& value) {
            value.support_volume_world.rotation =
                quat(0.5F, 0.0F, 0.0F, 0.0F);
        }},
        {+[](PlacementSurface& value) {
            value.support_volume_size.x = 0.0F;
        }},
        {+[](PlacementSurface& value) {
            value.support_volume_size.y = -0.1F;
        }},
        {+[](PlacementSurface& value) {
            value.support_volume_size.z =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](PlacementSurface& value) { value.half_extent_x_m = 0.0F; }},
        {+[](PlacementSurface& value) { value.half_extent_z_m = -0.1F; }},
        {+[](PlacementSurface& value) {
            value.half_extent_x_m =
                std::numeric_limits<float>::infinity();
        }},
        {+[](PlacementSurface& value) {
            value.overhead_clearance_m = 0.0F;
        }},
        {+[](PlacementSurface& value) {
            value.overhead_clearance_m = -0.1F;
        }},
        {+[](PlacementSurface& value) {
            value.overhead_clearance_m =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](PlacementSurface& value) { value.affordances.clear(); }},
        {+[](PlacementSurface& value) {
            value.affordances.front().id = 0U;
        }},
        {+[](PlacementSurface& value) {
            value.affordances.push_back(value.affordances.front());
        }},
        {+[](PlacementSurface& value) {
            value.affordances.front().object_in_surface.position.y =
                std::numeric_limits<float>::infinity();
        }},
        {+[](PlacementSurface& value) {
            value.affordances.front().object_in_surface.rotation =
                quat(0.0F, 0.0F, 0.0F, 0.0F);
        }},
        {+[](PlacementSurface& value) {
            value.affordances.front().object_in_surface.rotation =
                quat(1.01F, 0.0F, 0.0F, 0.0F);
        }},
        {+[](PlacementSurface& value) {
            value.affordances.front().support_point_object.x =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](PlacementSurface& value) {
            value.affordances.front().approach_direction_surface = vec3();
        }},
        {+[](PlacementSurface& value) {
            value.affordances.front().approach_direction_surface =
                vec3(0.0F, 0.0F, 2.0F);
        }},
        {+[](PlacementSurface& value) {
            value.affordances.front().approach_direction_surface.x =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](PlacementSurface& value) {
            value.affordances.front().clearance_radius = -0.000001F;
        }},
        {+[](PlacementSurface& value) {
            value.affordances.front().clearance_radius =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](PlacementSurface& value) {
            value.affordances.front().clearance_radius =
                std::numeric_limits<float>::infinity();
        }},
    };

    for (const InvalidSurfaceCase& invalid : invalid_cases) {
        PlacementSurfaceRegistry registry;
        const SurfaceHandle original = registry.upsert(make_surface(52U, 8U));
        const PlacementSurface snapshot = *registry.find(original);
        PlacementSurface malformed = make_surface(52U, 900U);
        invalid.mutate(malformed);
        TEST_CHECK(throws_invalid_argument([&] {
            (void)registry.upsert(malformed);
        }));
        const PlacementSurface* preserved = registry.find(original);
        TEST_CHECK(preserved != nullptr);
        TEST_CHECK(preserved->handle == snapshot.handle);
        TEST_CHECK(near(preserved->surface_world, snapshot.surface_world));
        TEST_CHECK(preserved->affordances.size() == snapshot.affordances.size());
    }

    PlacementSurface zero_clearance = make_surface(53U, 1U);
    zero_clearance.affordances.front().clearance_radius = 0.0F;
    PlacementSurfaceRegistry registry;
    TEST_CHECK(registry.upsert(zero_clearance) == (SurfaceHandle{53U, 1U}));
}

void test_surface_numeric_boundaries() {
    using namespace interaction;

    PlacementSurfaceRegistry registry;
    PlacementSurface tilted = make_surface(60U, 1U);
    tilted.surface_world.rotation = quat_from_angle_axis(
        kFiveDegrees, vec3(1.0F, 0.0F, 0.0F));
    align_support_volume(tilted);
    TEST_CHECK(registry.upsert(tilted) == (SurfaceHandle{60U, 1U}));

    PlacementSurface over_tilt = make_surface(61U, 1U);
    over_tilt.surface_world.rotation = quat_from_angle_axis(
        kFiveDegrees + 0.0000174533F, vec3(1.0F, 0.0F, 0.0F));
    align_support_volume(over_tilt);
    TEST_CHECK(throws_invalid_argument([&] {
        (void)registry.upsert(over_tilt);
    }));

    PlacementSurface top_boundary = make_surface(62U, 1U);
    top_boundary.support_volume_world.position.y =
        top_boundary.surface_world.position.y + 0.001F -
        0.5F * top_boundary.support_volume_size.y;
    TEST_CHECK(registry.upsert(top_boundary) == (SurfaceHandle{62U, 1U}));

    PlacementSurface top_outside = make_surface(63U, 1U);
    top_outside.support_volume_world.position.y =
        top_outside.surface_world.position.y + 0.001001F -
        0.5F * top_outside.support_volume_size.y;
    TEST_CHECK(throws_invalid_argument([&] {
        (void)registry.upsert(top_outside);
    }));

    PlacementSurface rotation_boundary = make_surface(64U, 1U);
    rotation_boundary.support_volume_world.rotation = quat_from_angle_axis(
        0.001745329F, vec3(0.0F, 1.0F, 0.0F));
    TEST_CHECK(registry.upsert(rotation_boundary) ==
               (SurfaceHandle{64U, 1U}));

    PlacementSurface rotation_outside = make_surface(65U, 1U);
    rotation_outside.support_volume_world.rotation = quat_from_angle_axis(
        0.00190F, vec3(0.0F, 1.0F, 0.0F));
    TEST_CHECK(throws_invalid_argument([&] {
        (void)registry.upsert(rotation_outside);
    }));

    PlacementSurface usable_boundary = make_surface(66U, 1U);
    usable_boundary.half_extent_x_m =
        0.5F * usable_boundary.support_volume_size.x;
    usable_boundary.half_extent_z_m =
        0.5F * usable_boundary.support_volume_size.z;
    TEST_CHECK(registry.upsert(usable_boundary) ==
               (SurfaceHandle{66U, 1U}));

    PlacementSurface usable_x_outside = make_surface(67U, 1U);
    usable_x_outside.half_extent_x_m = std::nextafter(
        0.5F * usable_x_outside.support_volume_size.x,
        std::numeric_limits<float>::infinity());
    TEST_CHECK(throws_invalid_argument([&] {
        (void)registry.upsert(usable_x_outside);
    }));

    PlacementSurface usable_z_outside = make_surface(68U, 1U);
    usable_z_outside.half_extent_z_m = std::nextafter(
        0.5F * usable_z_outside.support_volume_size.z,
        std::numeric_limits<float>::infinity());
    TEST_CHECK(throws_invalid_argument([&] {
        (void)registry.upsert(usable_z_outside);
    }));
}

void test_bounds_validation_and_requested_fit() {
    using namespace interaction;

    PlacementSurface surface = make_surface();
    const ObjectLocalBounds bounds = make_bounds();
    const PlaceAffordance& affordance = surface.affordances.front();
    const PlacementFit exact_fit =
        evaluate_placement_fit(surface, affordance, bounds);
    TEST_CHECK(exact_fit.accepted);
    TEST_CHECK(exact_fit.reason == Reason::None);
    TEST_CHECK(exact_fit.support_gap_m == 0.020F);
    TEST_CHECK(exact_fit.footprint_valid);
    TEST_CHECK(exact_fit.overhead_valid);

    PlacementSurface outside = surface;
    outside.affordances.front().object_in_surface.position.x += 0.000001F;
    const PlacementFit outside_fit = evaluate_placement_fit(
        outside, outside.affordances.front(), bounds);
    TEST_CHECK(!outside_fit.accepted);
    TEST_CHECK(outside_fit.reason == Reason::PlacementOutOfBounds);
    TEST_CHECK(!outside_fit.footprint_valid);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    const ObjectLocalBounds invalid_bounds[] = {
        {vec3(nan, 0.0F, 0.0F), vec3(1.0F, 1.0F, 1.0F)},
        {vec3(0.0F, infinity, 0.0F), vec3(1.0F, 1.0F, 1.0F)},
        {vec3(), vec3(0.0F, 1.0F, 1.0F)},
        {vec3(), vec3(1.0F, -0.1F, 1.0F)},
        {vec3(), vec3(1.0F, 1.0F, nan)},
        {vec3(), vec3(infinity, 1.0F, 1.0F)},
    };
    for (ObjectLocalBounds invalid : invalid_bounds) {
        TEST_CHECK(throws_invalid_argument([&] {
            (void)evaluate_placement_fit(surface, affordance, invalid);
        }));
        TEST_CHECK(throws_invalid_argument([&] {
            (void)evaluate_actual_placement_fit(
                surface,
                affordance,
                placement_goal_world(surface, affordance.object_in_surface),
                invalid);
        }));
    }
}

void test_support_corner_and_overhead_boundaries() {
    using namespace interaction;
    const ObjectLocalBounds bounds = make_bounds();

    PlacementSurface low_gap = make_surface();
    low_gap.affordances.front().support_point_object.y = -0.165F;
    const PlacementFit low_boundary = evaluate_placement_fit(
        low_gap, low_gap.affordances.front(), bounds);
    TEST_CHECK(low_boundary.accepted);
    TEST_CHECK(low_boundary.support_gap_m == -0.005F);
    low_gap.affordances.front().support_point_object.y -= 0.000001F;
    TEST_CHECK(!evaluate_placement_fit(
        low_gap, low_gap.affordances.front(), bounds).accepted);

    PlacementSurface high_gap = make_surface();
    high_gap.affordances.front().support_point_object.y += 0.000001F;
    TEST_CHECK(!evaluate_placement_fit(
        high_gap, high_gap.affordances.front(), bounds).accepted);

    ObjectLocalBounds floor_boundary = bounds;
    floor_boundary.center_object.y = -0.015F;
    const PlacementSurface floor_surface = make_surface();
    const PlacementFit floor_exact = evaluate_placement_fit(
        floor_surface, floor_surface.affordances.front(), floor_boundary);
    TEST_CHECK(floor_exact.accepted);
    TEST_CHECK(floor_exact.lowest_corner_m == -0.005F);
    floor_boundary.center_object.y -= 0.000001F;
    TEST_CHECK(!evaluate_placement_fit(
        floor_surface, floor_surface.affordances.front(), floor_boundary)
                    .accepted);

    PlacementSurface overhead_boundary = make_surface();
    overhead_boundary.overhead_clearance_m = 0.32F;
    const PlacementFit overhead_exact = evaluate_placement_fit(
        overhead_boundary, overhead_boundary.affordances.front(), bounds);
    TEST_CHECK(overhead_exact.accepted);
    TEST_CHECK(overhead_exact.highest_corner_m == 0.32F);
    overhead_boundary.overhead_clearance_m -= 0.000001F;
    const PlacementFit overhead_outside = evaluate_placement_fit(
        overhead_boundary, overhead_boundary.affordances.front(), bounds);
    TEST_CHECK(!overhead_outside.accepted);
    TEST_CHECK(!overhead_outside.overhead_valid);
}

void test_oriented_offcenter_and_actual_pose_fit() {
    using namespace interaction;

    PlacementSurface oriented = make_surface(80U, 1U);
    oriented.surface_world = {
        vec3(1.0F, 0.70F, -2.0F),
        quat_from_angle_axis(0.4F, vec3(0.0F, 1.0F, 0.0F))};
    oriented.support_volume_size = vec3(0.24F, 0.10F, 0.44F);
    oriented.half_extent_x_m = 0.12F;
    oriented.half_extent_z_m = 0.22F;
    oriented.overhead_clearance_m = 0.20F;
    align_support_volume(oriented);
    PlaceAffordance& affordance = oriented.affordances.front();
    affordance.object_in_surface = {
        vec3(0.0F, 0.05F, 0.0F),
        quat_from_angle_axis(
            1.570796327F, vec3(0.0F, 1.0F, 0.0F))};
    affordance.support_point_object = vec3(0.0F, -0.03F, 0.0F);
    affordance.clearance_radius = 0.02F;
    const ObjectLocalBounds bounds{
        vec3(), vec3(0.20F, 0.05F, 0.10F)};

    const PlacementFit requested =
        evaluate_placement_fit(oriented, affordance, bounds);
    TEST_CHECK(requested.accepted);
    TEST_CHECK(requested.footprint_valid);

    const Transform actual =
        placement_goal_world(oriented, affordance.object_in_surface);
    const PlacementFit actual_fit = evaluate_actual_placement_fit(
        oriented, affordance, actual, bounds);
    TEST_CHECK(actual_fit.accepted);
    TEST_CHECK(near(actual_fit.support_gap_m, requested.support_gap_m));
    TEST_CHECK(near(actual_fit.lowest_corner_m, requested.lowest_corner_m));
    TEST_CHECK(near(actual_fit.highest_corner_m, requested.highest_corner_m));

    Transform outside_local = affordance.object_in_surface;
    outside_local.position.x += 0.000001F;
    const PlacementFit outside = evaluate_actual_placement_fit(
        oriented,
        affordance,
        placement_goal_world(oriented, outside_local),
        bounds);
    TEST_CHECK(!outside.accepted);
    TEST_CHECK(outside.reason == Reason::PlacementOutOfBounds);

    Transform invalid_actual = actual;
    invalid_actual.position.z = std::numeric_limits<float>::infinity();
    TEST_CHECK(throws_invalid_argument([&] {
        (void)evaluate_actual_placement_fit(
            oriented, affordance, invalid_actual, bounds);
    }));
}

void test_surface_resolver_selects_unique_surface_not_affordance() {
    using namespace interaction;

    PlacementSurfaceRegistry registry;
    PlacementSurface first = make_surface(90U, 1U);
    PlaceAffordance second_slot = first.affordances.front();
    second_slot.id = 8U;
    second_slot.object_in_surface.position.z -= 0.05F;
    first.affordances.push_back(second_slot);
    const SurfaceHandle first_handle = registry.upsert(first);

    const vec3 root(0.25F, -100.0F, 3.50F);
    TEST_CHECK(registry.resolve_single_surface(root, 0.0F) == first_handle);
    TEST_CHECK(registry.resolve_single_surface(root, 0.01F) == first_handle);
    TEST_CHECK(throws_invalid_argument([&] {
        (void)registry.resolve_single_surface(root, -0.01F);
    }));
    TEST_CHECK(throws_invalid_argument([&] {
        (void)registry.resolve_single_surface(
            root, std::numeric_limits<float>::quiet_NaN());
    }));
    TEST_CHECK(throws_invalid_argument([&] {
        (void)registry.resolve_single_surface(
            root, std::numeric_limits<float>::infinity());
    }));
    TEST_CHECK(throws_invalid_argument([&] {
        (void)registry.resolve_single_surface(
            vec3(std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F),
            1.0F);
    }));
    TEST_CHECK(throws_invalid_argument([&] {
        (void)registry.resolve_single_surface(
            vec3(0.0F, std::numeric_limits<float>::infinity(), 0.0F),
            1.0F);
    }));
    TEST_CHECK(throws_invalid_argument([&] {
        (void)registry.resolve_single_surface(
            vec3(0.0F, 0.0F, -std::numeric_limits<float>::infinity()),
            1.0F);
    }));
    TEST_CHECK(!registry.resolve_single_surface(
        vec3(10.0F, 0.0F, 10.0F), 0.10F).has_value());

    PlacementSurface second = make_surface(91U, 1U);
    second.surface_world.position.x += 0.25F;
    align_support_volume(second);
    (void)registry.upsert(second);
    TEST_CHECK(!registry.resolve_single_surface(root, 0.30F).has_value());
    TEST_CHECK(registry.resolve_single_surface(root, 0.10F) == first_handle);
}

}  // namespace

int main() {
    test_public_contract_and_reason_abi();
    test_registry_lookup_goal_and_generation();
    test_surface_validation_is_strict_and_transactional();
    test_surface_numeric_boundaries();
    test_bounds_validation_and_requested_fit();
    test_support_corner_and_overhead_boundaries();
    test_oriented_offcenter_and_actual_pose_fit();
    test_surface_resolver_selects_unique_surface_not_affordance();
}
