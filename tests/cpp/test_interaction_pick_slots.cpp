#include "interaction_pick_slots.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace {

constexpr float kTolerance = 1.0e-5F;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_near(
    float actual,
    float expected,
    const char* message) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > kTolerance) {
        throw std::runtime_error(message);
    }
}

void require_near(
    vec3 actual,
    vec3 expected,
    const char* message) {
    require_near(actual.x, expected.x, message);
    require_near(actual.y, expected.y, message);
    require_near(actual.z, expected.z, message);
}

float planar_yaw(quat rotation) {
    const vec3 forward = quat_mul_vec3(
        rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(forward.x, forward.z);
}

interaction::InteractionTarget make_target(
    interaction::Transform object_world) {
    using namespace interaction;

    InteractionTarget target{};
    target.handle = {1U, 1U};
    target.object_world = object_world;
    target.object_profile_id = 1U;
    target.object_bounds = {
        vec3(), vec3(0.05F, 0.10F, 0.05F)};
    target.object_dimensions = vec3(0.10F, 0.20F, 0.10F);
    target.table_world = {vec3(100.0F, 0.0F, 100.0F), quat()};
    target.table_size = vec3(1.0F, 0.10F, 1.0F);

    GraspAffordance affordance{};
    affordance.id = 3U;
    affordance.hand_in_object = {vec3(), quat()};
    affordance.approach_direction_object = vec3(0.0F, 0.0F, 1.0F);
    affordance.interaction_slots = {{7U, 0.20F, -0.40F, 0.25F}};
    target.affordances.push_back(affordance);
    return target;
}

interaction::PickSlotSelection map_only_slot(
    const interaction::InteractionTarget& target,
    float live_root_y) {
    using namespace interaction;

    const Transform live_root{
        vec3(
            target.object_world.position.x,
            live_root_y,
            target.object_world.position.z),
        quat()};
    return select_pick_slot(
        live_root, target, target.affordances.front(), {});
}

interaction::MappedPickSlot only_mapped_slot(
    const interaction::PickSlotSelection& selection) {
    require(selection.ordered.size() == 1U, "expected exactly one mapped slot");
    return selection.ordered.front();
}

void test_identity_and_translated_rotated_object_planar_mapping() {
    using namespace interaction;

    InteractionTarget target = make_target(
        Transform{vec3(1.0F, 0.8F, 3.0F), quat()});
    PickSlotSelection selection = map_only_slot(target, 0.0F);
    const MappedPickSlot mapped = only_mapped_slot(selection);
    require_near(
        mapped.root_world.position,
        vec3(1.20F, 0.0F, 2.60F),
        "identity object mapped the slot to the wrong position");
    require_near(
        planar_yaw(mapped.root_world.rotation),
        0.25F,
        "identity object mapped the slot to the wrong yaw");

    target.object_world = {
        vec3(4.0F, 0.8F, -2.0F),
        quat_from_angle_axis(0.5F * PIf, vec3(0.0F, 1.0F, 0.0F))};
    selection = map_only_slot(target, 0.0F);
    const MappedPickSlot translated_rotated = only_mapped_slot(selection);
    require_near(
        translated_rotated.root_world.position,
        vec3(3.60F, 0.0F, -2.20F),
        "translated rotated object mapped the slot to the wrong position");
    require_near(
        planar_yaw(translated_rotated.root_world.rotation),
        0.25F + 0.5F * PIf,
        "translated rotated object mapped the slot to the wrong yaw");
}

void test_object_pitch_and_roll_do_not_tilt_the_root() {
    using namespace interaction;

    const quat object_rotations[] = {
        quat_from_angle_axis(0.35F, vec3(1.0F, 0.0F, 0.0F)),
        quat_from_angle_axis(-0.45F, vec3(0.0F, 0.0F, 1.0F)),
    };
    for (quat object_rotation : object_rotations) {
        const InteractionTarget target = make_target(
            Transform{vec3(1.0F, 0.8F, 3.0F), object_rotation});
        const MappedPickSlot mapped = only_mapped_slot(
            map_only_slot(target, 0.0F));
        require_near(
            mapped.root_world.position,
            vec3(1.20F, 0.0F, 2.60F),
            "object pitch or roll changed planar slot mapping");
        require_near(
            quat_mul_vec3(
                mapped.root_world.rotation,
                vec3(0.0F, 1.0F, 0.0F)),
            vec3(0.0F, 1.0F, 0.0F),
            "object pitch or roll tilted the mapped root");
        require_near(
            planar_yaw(mapped.root_world.rotation),
            0.25F,
            "object pitch or roll changed mapped root yaw");
    }
}

void test_live_root_y_is_preserved_exactly() {
    using namespace interaction;

    const InteractionTarget target = make_target(
        Transform{vec3(1.0F, 0.8F, 3.0F), quat()});
    const MappedPickSlot mapped = only_mapped_slot(
        map_only_slot(target, 1.25F));
    require(
        mapped.root_world.position.y == 1.25F,
        "mapped root did not preserve the live root Y exactly");
}

void test_zero_planar_object_forward_is_invalid_geometry() {
    using namespace interaction;

    const quat vertical_forward(0.5F, 0.5F, 0.5F, -0.5F);
    const InteractionTarget target = make_target(
        Transform{vec3(1.0F, 0.8F, 3.0F), vertical_forward});
    const MappedPickSlot mapped = only_mapped_slot(
        map_only_slot(target, 0.0F));
    require(
        mapped.reason == PickSlotReason::InvalidGeometry,
        "zero planar object-forward projection was not invalid geometry");
}

interaction::InteractionTarget make_route_target(
    vec3 mapped_root_position) {
    using namespace interaction;

    InteractionTarget target = make_target(Transform{
        vec3(mapped_root_position.x, 0.8F, mapped_root_position.z),
        quat()});
    target.affordances.front().interaction_slots = {{
        7U,
        0.0F,
        0.0F,
        0.0F,
    }};
    return target;
}

void set_only_slot_root(
    interaction::InteractionTarget& target,
    vec3 mapped_root_position) {
    target.affordances.front().interaction_slots = {{
        7U,
        mapped_root_position.x - target.object_world.position.x,
        mapped_root_position.z - target.object_world.position.z,
        0.0F,
    }};
}

interaction::PickSlotSelection select_route(
    vec3 live_root_position,
    const interaction::InteractionTarget& target,
    const std::vector<interaction::PickNavigationObstacle>& obstacles = {},
    const interaction::PickSlotConfig& config = {}) {
    return interaction::select_pick_slot(
        interaction::Transform{live_root_position, quat()},
        target,
        target.affordances.front(),
        obstacles,
        config);
}

interaction::PickNavigationObstacle obstacle_above_route(
    float surface_distance) {
    return {
        vec3(0.50F, surface_distance + 0.25F, 0.0F),
        vec3(0.20F, 0.50F, 0.20F),
    };
}

void test_rotated_table_blocks_only_the_middle_of_the_segment() {
    using namespace interaction;

    InteractionTarget target = make_route_target(vec3(0.75F, 0.0F, 0.0F));
    target.table_world = {
        vec3(0.0F, 0.0F, 0.0F),
        quat_from_angle_axis(
            0.25F * PIf, vec3(0.0F, 1.0F, 0.0F))};
    target.table_size = vec3(0.20F, 0.10F, 0.20F);

    PickSlotConfig config{};
    config.maximum_direct_travel_m = 2.0F;
    const float expanded_half_extent =
        0.5F * target.table_size.x + config.table_root_expansion_m;
    const float endpoint_local_component = 0.75F * std::sqrt(0.5F);
    require(
        endpoint_local_component > expanded_half_extent,
        "rotated-table fixture did not leave both route endpoints outside");

    const MappedPickSlot mapped = only_mapped_slot(select_route(
        vec3(-0.75F, 0.0F, 0.0F), target, {}, config));
    require(
        mapped.reason == PickSlotReason::TableBlocked,
        "rotated table intersecting only the route middle was not blocked");
}

void test_table_expansion_boundary_is_closed_and_next_float_is_clear() {
    using namespace interaction;

    PickSlotConfig config{};
    InteractionTarget blocked_target = make_route_target(vec3(0.50F, 0.0F, 0.0F));
    blocked_target.table_world = {vec3(0.0F, 0.0F, 0.0F), quat()};
    blocked_target.table_size = vec3(0.20F, 0.10F, 0.20F);
    const float boundary =
        0.5F * blocked_target.table_size.z + config.table_root_expansion_m;
    blocked_target.object_world.position.z = boundary;
    set_only_slot_root(blocked_target, vec3(0.50F, 0.0F, boundary));

    const MappedPickSlot touching = only_mapped_slot(select_route(
        vec3(-0.50F, 0.0F, boundary), blocked_target, {}, config));
    require(
        touching.reason == PickSlotReason::TableBlocked,
        "route touching the expanded table boundary was not blocked");

    const float just_outside = std::nextafter(
        boundary, std::numeric_limits<float>::infinity());
    InteractionTarget clear_target = blocked_target;
    clear_target.object_world.position.z = just_outside;
    set_only_slot_root(clear_target, vec3(0.50F, 0.0F, just_outside));
    const MappedPickSlot clear = only_mapped_slot(select_route(
        vec3(-0.50F, 0.0F, just_outside), clear_target, {}, config));
    require(
        clear.reason == PickSlotReason::None,
        "route one representable float beyond the table boundary was blocked");
}

void test_tiny_table_boundary_straddle_is_blocked() {
    using namespace interaction;

    PickSlotConfig config{};
    InteractionTarget target = make_route_target(vec3(0.0F, 0.0F, 0.0F));
    target.table_world = {vec3(0.0F, 0.0F, 0.0F), quat()};
    target.table_size = vec3(0.20F, 0.10F, 0.20F);
    const float boundary =
        0.5F * target.table_size.z + config.table_root_expansion_m;
    const float just_outside = std::nextafter(
        boundary, std::numeric_limits<float>::infinity());
    const float just_inside = std::nextafter(
        boundary, -std::numeric_limits<float>::infinity());
    require(
        just_outside > boundary && just_inside < boundary &&
            just_outside - just_inside <= 1.0e-7F,
        "tiny table-boundary straddle fixture did not span the boundary");

    target.object_world.position.z = just_inside;
    set_only_slot_root(target, vec3(0.0F, 0.0F, just_inside));
    const MappedPickSlot straddling = only_mapped_slot(select_route(
        vec3(0.0F, 0.0F, just_outside), target, {}, config));
    require(
        straddling.reason == PickSlotReason::TableBlocked,
        "tiny segment crossing the closed table boundary was not blocked");
}

void test_obstacle_safety_envelope_uses_strict_exact_distance() {
    using namespace interaction;

    const InteractionTarget target = make_route_target(vec3(1.0F, 0.0F, 0.0F));
    const PickSlotConfig config{};
    const float safety_envelope =
        config.obstacle_root_radius_m + config.obstacle_safety_margin_m;
    const float just_below = std::nextafter(safety_envelope, 0.0F);
    const float just_above = std::nextafter(
        safety_envelope, std::numeric_limits<float>::infinity());

    struct ClearanceCase {
        float surface_distance;
        PickSlotReason expected_reason;
        const char* message;
    };
    const ClearanceCase cases[] = {
        {0.60000F, PickSlotReason::ObstacleBlocked,
         "obstacle at distance 0.60000 was not blocked"},
        {0.60002F, PickSlotReason::ObstacleBlocked,
         "obstacle at distance 0.60002 was not blocked"},
        {0.60003F, PickSlotReason::None,
         "obstacle at distance 0.60003 was not clear"},
        {just_below, PickSlotReason::ObstacleBlocked,
         "next float below the obstacle safety envelope was not blocked"},
        {just_above, PickSlotReason::None,
         "next float above the obstacle safety envelope was not clear"},
    };

    for (const ClearanceCase& clearance_case : cases) {
        const MappedPickSlot mapped = only_mapped_slot(select_route(
            vec3(0.0F, 0.0F, 0.0F),
            target,
            {obstacle_above_route(clearance_case.surface_distance)},
            config));
        require(mapped.reason == clearance_case.expected_reason,
                clearance_case.message);
        require(
            mapped.obstacle_index ==
                (clearance_case.expected_reason == PickSlotReason::ObstacleBlocked
                     ? 0
                     : -1),
            "obstacle safety-envelope result reported the wrong obstacle index");
    }
}

void test_exact_segment_aabb_corner_distance_beats_expanded_box_shortcut() {
    using namespace interaction;

    const InteractionTarget target = make_route_target(vec3(1.0F, 0.0F, 0.0F));
    const PickSlotConfig config{};
    constexpr float kCornerDistance = 0.60003F;
    const float x_gap = 3.0F * kCornerDistance / 5.0F;
    const float z_gap = 4.0F * kCornerDistance / 5.0F;
    const float safety_envelope =
        config.obstacle_root_radius_m + config.obstacle_safety_margin_m;
    require_near(
        std::sqrt(x_gap * x_gap + z_gap * z_gap),
        kCornerDistance,
        "corner-distance fixture was not exactly 0.60003");
    require(
        x_gap < safety_envelope && z_gap < safety_envelope,
        "corner fixture would not overlap an independently expanded AABB");

    const PickNavigationObstacle corner_obstacle{
        vec3(1.0F + x_gap + 0.25F, 0.0F, z_gap + 0.25F),
        vec3(0.50F, 0.20F, 0.50F),
    };
    const MappedPickSlot mapped = only_mapped_slot(select_route(
        vec3(0.0F, 0.0F, 0.0F), target, {corner_obstacle}, config));
    require(
        mapped.reason == PickSlotReason::None,
        "exact clear corner distance was rejected like an expanded-box overlap");
    require(
        mapped.obstacle_index == -1,
        "clear corner-distance route reported a blocking obstacle");
}

void test_obstacle_between_clear_endpoints_blocks_the_segment() {
    using namespace interaction;

    const InteractionTarget target = make_route_target(vec3(0.75F, 0.0F, 0.0F));
    const PickNavigationObstacle obstacle{
        vec3(0.0F, 0.0F, 0.0F),
        vec3(0.10F, 0.10F, 0.10F),
    };
    PickSlotConfig config{};
    config.maximum_direct_travel_m = 2.0F;
    const float safety_envelope =
        config.obstacle_root_radius_m + config.obstacle_safety_margin_m;
    require(
        0.75F - 0.05F > safety_envelope,
        "between-endpoints fixture did not leave both endpoints clear");

    const MappedPickSlot mapped = only_mapped_slot(select_route(
        vec3(-0.75F, 0.0F, 0.0F), target, {obstacle}, config));
    require(
        mapped.reason == PickSlotReason::ObstacleBlocked,
        "obstacle between two clear endpoints did not block the segment");
    require(
        mapped.obstacle_index == 0,
        "between-endpoints blocker reported the wrong obstacle index");
}

void require_malformed_obstacle_rejected(
    const std::vector<interaction::PickNavigationObstacle>& obstacles,
    int32_t expected_index,
    const char* message) {
    using namespace interaction;

    const InteractionTarget target = make_route_target(vec3(1.0F, 0.0F, 0.0F));
    const PickSlotSelection selection = select_route(
        vec3(0.0F, 0.0F, 0.0F), target, obstacles);
    const MappedPickSlot mapped = only_mapped_slot(selection);
    require(mapped.reason == PickSlotReason::InvalidGeometry, message);
    require(
        mapped.obstacle_index == expected_index,
        "malformed obstacle reported the wrong first malformed index");
    require(
        selection.reason == PickSlotReason::InvalidGeometry,
        "malformed obstacle did not invalidate the overall selection");
}

void test_malformed_obstacle_data_fails_closed() {
    using namespace interaction;

    constexpr float kFar = 50.0F;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    require_malformed_obstacle_rejected(
        {{{kFar, kFar, kFar}, {0.0F, 0.20F, 0.20F}}},
        0,
        "zero obstacle size did not fail closed");
    require_malformed_obstacle_rejected(
        {{{kFar, kFar, kFar}, {0.20F, -0.20F, 0.20F}}},
        0,
        "negative obstacle size did not fail closed");
    require_malformed_obstacle_rejected(
        {{{nan, kFar, kFar}, {0.20F, 0.20F, 0.20F}}},
        0,
        "NaN obstacle center did not fail closed");

    const PickNavigationObstacle valid_far{
        vec3(kFar, kFar, kFar), vec3(0.20F, 0.20F, 0.20F)};
    const PickNavigationObstacle mismatched_malformed{
        vec3(kFar, nan, kFar), vec3(0.20F, 0.20F, -0.20F)};
    require_malformed_obstacle_rejected(
        {valid_far, mismatched_malformed},
        1,
        "cross-axis malformed obstacle values did not fail closed");
}

void test_target_is_exempt_but_target_distance_diagnostics_move() {
    using namespace interaction;

    const vec3 live_root_position(0.0F, 0.0F, 0.0F);
    const vec3 mapped_root_position(1.0F, 0.0F, 0.0F);
    InteractionTarget first_target = make_target(Transform{
        vec3(0.25F, 0.8F, -0.25F), quat()});
    first_target.object_bounds = {
        vec3(0.10F, 0.05F, -0.05F),
        vec3(0.05F, 0.10F, 0.05F)};
    set_only_slot_root(first_target, mapped_root_position);

    InteractionTarget moved_target = first_target;
    moved_target.object_world.position = vec3(0.75F, 0.8F, 0.50F);
    set_only_slot_root(moved_target, mapped_root_position);

    const MappedPickSlot first = only_mapped_slot(select_route(
        live_root_position, first_target));
    const MappedPickSlot moved = only_mapped_slot(select_route(
        live_root_position, moved_target));
    require(
        first.reason == PickSlotReason::None &&
            moved.reason == PickSlotReason::None,
        "moving the tabletop target across the route changed eligibility");
    require_near(
        first.root_world.position,
        moved.root_world.position,
        "target-exemption fixture did not preserve the mapped route");
    require(
        std::abs(
            first.object_origin_distance_m -
            moved.object_origin_distance_m) > kTolerance,
        "moving the target did not update object-origin distance");
    require(
        std::abs(
            first.object_bounds_center_distance_m -
            moved.object_bounds_center_distance_m) > kTolerance,
        "moving the target did not update object-bounds-center distance");
}

void test_direct_travel_tolerance_boundary_is_inclusive() {
    using namespace interaction;

    const MappedPickSlot at_boundary = only_mapped_slot(select_route(
        vec3(0.0F, 0.0F, 0.0F),
        make_route_target(vec3(1.00002F, 0.0F, 0.0F))));
    require(
        at_boundary.reason == PickSlotReason::None,
        "direct route of length 1.00002 was not eligible");
    require_near(
        at_boundary.route_length_m,
        1.00002F,
        "eligible travel-boundary route reported the wrong length");

    const MappedPickSlot beyond_boundary = only_mapped_slot(select_route(
        vec3(0.0F, 0.0F, 0.0F),
        make_route_target(vec3(1.00003F, 0.0F, 0.0F))));
    require(
        beyond_boundary.reason == PickSlotReason::OutsideTravelEnvelope,
        "direct route of length 1.00003 was not rejected");
    require_near(
        beyond_boundary.route_length_m,
        1.00003F,
        "ineligible travel-boundary route reported the wrong length");

    const InteractionTarget admitted_target =
        make_route_target(vec3(1.00003F, 0.0F, 0.0F));
    const PickSlotReason admitted_revalidation = revalidate_frozen_pick_slot(
        Transform{vec3(0.0F, 0.0F, 0.0F), quat()},
        Transform{vec3(1.00003F, 0.0F, 0.0F), quat()},
        admitted_target,
        {});
    require(
        admitted_revalidation == PickSlotReason::None,
        "post-admission frozen route reapplied the activation radius");
}

const interaction::MappedPickSlot& selected_mapped_slot(
    const interaction::PickSlotSelection& selection,
    const char* message) {
    require(selection.selected_index.has_value(), message);
    require(
        *selection.selected_index < selection.ordered.size(),
        "selected pick-slot index was outside the diagnostic vector");
    return selection.ordered[*selection.selected_index];
}

interaction::InteractionTarget make_multi_slot_target(
    std::vector<interaction::GraspInteractionSlot> slots) {
    interaction::InteractionTarget target = make_target(
        interaction::Transform{vec3(0.0F, 0.8F, 0.0F), quat()});
    target.affordances.front().interaction_slots = std::move(slots);
    return target;
}

using PickSlotRankingKey = std::tuple<uint64_t, uint64_t, uint32_t>;

PickSlotRankingKey ranking_key(
    const interaction::MappedPickSlot& candidate) {
    return {
        candidate.route_millimetres,
        candidate.heading_milliradians,
        candidate.id,
    };
}

interaction::InteractionTarget make_tuple_ranking_target(
    bool reverse_exact_tie) {
    using interaction::GraspInteractionSlot;

    std::vector<GraspInteractionSlot> slots{
        {40U, 0.5000F, 0.0F, 0.0F},
        {5U, 0.0F, 0.5010F, 0.0F},
    };
    if (reverse_exact_tie) {
        slots.push_back({3U, 0.0F, 0.5002F, 0.0F});
        slots.push_back({8U, 0.0F, 0.5002F, 0.0F});
    } else {
        slots.push_back({8U, 0.0F, 0.5002F, 0.0F});
        slots.push_back({3U, 0.0F, 0.5002F, 0.0F});
    }
    return make_multi_slot_target(std::move(slots));
}

void test_ranking_uses_quantized_route_heading_and_id_tuple() {
    using namespace interaction;

    const PickSlotSelection selection = select_route(
        vec3(0.0F, 0.0F, 0.0F),
        make_tuple_ranking_target(false));
    require(
        selection.ordered.size() == 4U,
        "ranking fixture did not evaluate all four authored slots");
    for (const MappedPickSlot& candidate : selection.ordered) {
        require(
            candidate.reason == PickSlotReason::None,
            "ranking fixture did not keep every authored slot clear");
    }

    const MappedPickSlot& shorter_worse_heading = selection.ordered[0];
    const MappedPickSlot& longer_better_heading = selection.ordered[1];
    const MappedPickSlot& same_bucket_better_heading = selection.ordered[2];
    const MappedPickSlot& lower_id_exact_tie = selection.ordered[3];
    require(
        shorter_worse_heading.route_millimetres == 500U &&
            longer_better_heading.route_millimetres == 501U,
        "one-millimetre ranking fixture produced the wrong route keys");
    require(
        shorter_worse_heading.heading_milliradians >
            longer_better_heading.heading_milliradians,
        "one-millimetre fixture did not give the longer route less heading");
    require(
        ranking_key(shorter_worse_heading) <
            ranking_key(longer_better_heading),
        "one-millimetre shorter route did not outrank smaller heading");

    require(
        same_bucket_better_heading.route_millimetres ==
            shorter_worse_heading.route_millimetres,
        "same-bucket fixture produced different route keys");
    require(
        ranking_key(same_bucket_better_heading) <
            ranking_key(shorter_worse_heading),
        "routes in one millimetre bucket did not use heading next");
    require(
        lower_id_exact_tie.route_millimetres ==
                same_bucket_better_heading.route_millimetres &&
            lower_id_exact_tie.heading_milliradians ==
                same_bucket_better_heading.heading_milliradians,
        "exact-tie fixture did not produce equal route and heading keys");
    require(
        ranking_key(lower_id_exact_tie) <
            ranking_key(same_bucket_better_heading),
        "exact route/heading tie did not use the lower nonzero slot ID");
    require(
        ranking_key(lower_id_exact_tie) <
            ranking_key(same_bucket_better_heading) &&
            ranking_key(same_bucket_better_heading) <
                ranking_key(shorter_worse_heading) &&
            ranking_key(shorter_worse_heading) <
                ranking_key(longer_better_heading),
        "four-slot ordering did not match the exact ranking tuple");
    require(
        selected_mapped_slot(selection, "ranking fixture selected no slot").id ==
            3U,
        "four-slot tuple ranking selected the wrong slot");

    const PickSlotSelection reversed = select_route(
        vec3(0.0F, 0.0F, 0.0F),
        make_tuple_ranking_target(true));
    require(
        selected_mapped_slot(
            reversed,
            "reversed exact-tie fixture selected no slot").id == 3U,
        "lower-ID exact tie depended on authored vector order");
}

void test_near_zero_route_uses_mapped_yaw_for_heading() {
    using namespace interaction;

    const InteractionTarget target = make_multi_slot_target({
        {40U, 1.0e-5F, 0.0F, 0.0F},
        {3U, 0.0F, 1.0e-5F, 0.50F},
    });
    const PickSlotSelection selection = select_route(
        vec3(0.0F, 0.0F, 0.0F), target);
    require(
        selection.ordered.size() == 2U,
        "near-zero heading fixture did not evaluate both slots");
    require(
        selection.ordered[0].route_millimetres == 0U &&
            selection.ordered[1].route_millimetres == 0U,
        "near-zero routes did not quantize into the zero-millimetre bucket");
    require(
        selection.ordered[0].heading_milliradians == 0U &&
            selection.ordered[1].heading_milliradians == 500U,
        "near-zero route did not derive heading from mapped slot yaw");
    require(
        selected_mapped_slot(
            selection,
            "near-zero heading fixture selected no slot").id == 40U,
        "near-zero route used route direction instead of mapped yaw");
}

interaction::PickSlotSelection select_half_millimetre_fixture() {
    return select_route(
        vec3(0.0F, 0.0F, 0.0F),
        make_multi_slot_target({
            {10U, 0.0F, 0.0005F, 0.0F},
            {11U, 0.0F, 0.0015F, 0.0F},
        }));
}

interaction::PickSlotSelection select_half_milliradian_fixture() {
    constexpr float kRadius = 0.50F;
    constexpr float kFirstHeading = 0.0005F;
    constexpr float kSecondHeading = 0.0015F;
    return select_route(
        vec3(0.0F, 0.0F, 0.0F),
        make_multi_slot_target({
            {
                20U,
                kRadius * std::sin(kFirstHeading),
                kRadius * std::cos(kFirstHeading),
                0.0F,
            },
            {
                21U,
                kRadius * std::sin(kSecondHeading),
                kRadius * std::cos(kSecondHeading),
                0.0F,
            },
        }));
}

void test_quantization_rounds_half_millimetres_and_milliradians_up() {
    using namespace interaction;

    const PickSlotSelection route_selection =
        select_half_millimetre_fixture();
    require(
        route_selection.ordered.size() == 2U,
        "half-millimetre fixture did not evaluate both slots");
    require(
        route_selection.ordered[0].reason == PickSlotReason::None &&
            route_selection.ordered[1].reason == PickSlotReason::None,
        "half-millimetre fixture was not clear");
    require(
        route_selection.ordered[0].route_millimetres == 1U &&
            route_selection.ordered[1].route_millimetres == 2U,
        "half-millimetre values did not round half-up");

    const PickSlotSelection heading_selection =
        select_half_milliradian_fixture();
    require(
        heading_selection.ordered.size() == 2U,
        "half-milliradian fixture did not evaluate both slots");
    require(
        heading_selection.ordered[0].reason == PickSlotReason::None &&
            heading_selection.ordered[1].reason == PickSlotReason::None,
        "half-milliradian fixture was not clear");
    require(
        heading_selection.ordered[0].heading_milliradians == 1U &&
            heading_selection.ordered[1].heading_milliradians == 2U,
        "half-milliradian values did not round half-up");
}

interaction::PickSlotSelection select_invalid_quantization_fixture() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    constexpr float kFiniteScaledOverflow = 2.0e16F;
    require(
        static_cast<double>(kFiniteScaledOverflow) * 1000.0 >
            static_cast<double>(std::numeric_limits<uint64_t>::max()),
        "finite quantization-overflow fixture did not exceed uint64_t");
    return select_route(
        vec3(0.0F, 0.0F, 0.0F),
        make_multi_slot_target({
            {30U, nan, 0.0F, 0.0F},
            {31U, infinity, 0.0F, 0.0F},
            {32U, kFiniteScaledOverflow, 0.0F, 0.0F},
            {33U, 0.0F, 0.50F, nan},
            {34U, 0.0F, 0.50F, infinity},
        }));
}

void test_nonfinite_and_scaled_overflow_values_reject_without_wrapping() {
    using namespace interaction;

    const PickSlotSelection selection =
        select_invalid_quantization_fixture();
    require(
        selection.ordered.size() == 5U,
        "invalid-quantization fixture did not retain every diagnostic");
    for (const MappedPickSlot& candidate : selection.ordered) {
        require(
            candidate.reason == PickSlotReason::InvalidGeometry,
            "nonfinite or scaled-overflow slot did not reject as invalid");
    }
    require(
        !selection.selected_index.has_value(),
        "invalid quantization value wrapped into a selectable key");
    require(
        selection.reason == PickSlotReason::InvalidGeometry,
        "all-invalid quantization fixture reported the wrong result");
}

interaction::PickSlotSelection select_blocked_nearest_fixture() {
    using namespace interaction;

    const InteractionTarget target = make_multi_slot_target({
        {41U, 0.80F, 0.0F, 0.0F},
        {42U, 0.0F, 0.90F, 0.0F},
    });
    const PickNavigationObstacle blocks_only_nearest{
        vec3(0.70F, 0.0F, 0.0F),
        vec3(0.10F, 0.10F, 0.10F),
    };
    return select_route(
        vec3(0.0F, 0.0F, 0.0F),
        target,
        {blocks_only_nearest});
}

void test_blocked_nearest_slot_falls_back_to_next_clear_slot() {
    using namespace interaction;

    const PickSlotSelection selection = select_blocked_nearest_fixture();
    require(
        selection.ordered.size() == 2U,
        "blocked-nearest fixture did not evaluate both authored slots");
    require(
        selection.ordered[0].reason == PickSlotReason::ObstacleBlocked &&
            selection.ordered[0].obstacle_index == 0,
        "nearest slot was not rejected by its blocker");
    require(
        selection.ordered[1].reason == PickSlotReason::None,
        "next slot was not clear after the nearest slot was blocked");
    require(
        selection.reason == PickSlotReason::None &&
            selected_mapped_slot(
                selection,
                "blocked-nearest fixture selected no fallback").id == 42U,
        "blocked nearest slot did not permit the next clear slot");
}

interaction::PickSlotSelection select_no_authored_fixture() {
    interaction::InteractionTarget target = make_multi_slot_target({});
    return select_route(vec3(0.0F, 0.0F, 0.0F), target);
}

interaction::PickSlotSelection select_all_outside_fixture() {
    return select_route(
        vec3(0.0F, 0.0F, 0.0F),
        make_multi_slot_target({
            {50U, 0.0F, 1.10F, 0.0F},
            {51U, 0.0F, 1.20F, 0.0F},
        }));
}

interaction::PickSlotSelection select_all_blocked_fixture() {
    using namespace interaction;

    const PickNavigationObstacle blocks_current_root{
        vec3(0.0F, 0.0F, 0.0F),
        vec3(0.10F, 0.10F, 0.10F),
    };
    return select_route(
        vec3(0.0F, 0.0F, 0.0F),
        make_multi_slot_target({
            {60U, 0.75F, 0.0F, 0.0F},
            {61U, 0.0F, 0.80F, 0.0F},
        }),
        {blocks_current_root});
}

void test_no_winner_reasons_distinguish_empty_outside_and_blocked() {
    using namespace interaction;

    const PickSlotSelection no_authored = select_no_authored_fixture();
    require(
        no_authored.ordered.empty() &&
            !no_authored.selected_index.has_value() &&
            no_authored.reason == PickSlotReason::NoAuthoredSlot,
        "empty authored-slot set did not report NoAuthoredSlot");

    const PickSlotSelection all_outside = select_all_outside_fixture();
    require(
        all_outside.ordered.size() == 2U &&
            !all_outside.selected_index.has_value() &&
            all_outside.reason == PickSlotReason::OutsideTravelEnvelope,
        "all-over-distance set did not report OutsideTravelEnvelope");
    for (const MappedPickSlot& candidate : all_outside.ordered) {
        require(
            candidate.reason == PickSlotReason::OutsideTravelEnvelope,
            "all-over-distance diagnostic reported the wrong reason");
    }

    const PickSlotSelection all_blocked = select_all_blocked_fixture();
    require(
        all_blocked.ordered.size() == 2U &&
            !all_blocked.selected_index.has_value() &&
            all_blocked.reason == PickSlotReason::AllSlotsBlocked,
        "nonempty all-blocked set did not report AllSlotsBlocked");
    for (const MappedPickSlot& candidate : all_blocked.ordered) {
        require(
            candidate.reason == PickSlotReason::ObstacleBlocked,
            "all-blocked diagnostic did not retain its concrete blocker");
    }
}

void require_malformed_pick_slot_config_rejected(
    const interaction::PickSlotConfig& config,
    const char* message) {
    using namespace interaction;

    const InteractionTarget target = make_multi_slot_target({
        {70U, 0.0F, 0.50F, 0.0F},
    });
    const PickSlotSelection selection = select_route(
        vec3(0.0F, 0.0F, 0.0F), target, {}, config);
    require(selection.reason == PickSlotReason::InvalidGeometry, message);
    require(
        !selection.selected_index.has_value(),
        "malformed pick-slot config selected a slot");

    const PickSlotReason frozen_reason = revalidate_frozen_pick_slot(
        Transform{vec3(0.0F, 0.0F, 0.0F), quat()},
        Transform{vec3(0.0F, 0.0F, 0.50F), quat()},
        target,
        {},
        config);
    require(
        frozen_reason == PickSlotReason::InvalidGeometry,
        "frozen-route call accepted a malformed pick-slot config");
}

void test_every_malformed_pick_slot_config_fails_closed() {
    using namespace interaction;

    using ConfigMember = float PickSlotConfig::*;
    struct MalformedConfigCase {
        ConfigMember member;
        float value;
        const char* message;
    };

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    const float overflow_prone = std::numeric_limits<float>::max();
    const MalformedConfigCase cases[] = {
        {&PickSlotConfig::maximum_direct_travel_m, 0.0F,
         "zero maximum travel did not fail closed"},
        {&PickSlotConfig::maximum_direct_travel_m, -1.0F,
         "negative maximum travel did not fail closed"},
        {&PickSlotConfig::maximum_direct_travel_m, nan,
         "NaN maximum travel did not fail closed"},
        {&PickSlotConfig::maximum_direct_travel_m, infinity,
         "infinite maximum travel did not fail closed"},
        {&PickSlotConfig::maximum_direct_travel_m, overflow_prone,
         "overflow-prone maximum travel did not fail closed"},
        {&PickSlotConfig::travel_tolerance_m, 0.0F,
         "zero travel tolerance did not fail closed"},
        {&PickSlotConfig::travel_tolerance_m, -1.0F,
         "negative travel tolerance did not fail closed"},
        {&PickSlotConfig::travel_tolerance_m, nan,
         "NaN travel tolerance did not fail closed"},
        {&PickSlotConfig::travel_tolerance_m, infinity,
         "infinite travel tolerance did not fail closed"},
        {&PickSlotConfig::travel_tolerance_m, overflow_prone,
         "overflow-prone travel tolerance did not fail closed"},
        {&PickSlotConfig::table_root_expansion_m, 0.0F,
         "zero table expansion did not fail closed"},
        {&PickSlotConfig::table_root_expansion_m, -1.0F,
         "negative table expansion did not fail closed"},
        {&PickSlotConfig::table_root_expansion_m, nan,
         "NaN table expansion did not fail closed"},
        {&PickSlotConfig::table_root_expansion_m, infinity,
         "infinite table expansion did not fail closed"},
        {&PickSlotConfig::table_root_expansion_m, overflow_prone,
         "overflow-prone table expansion did not fail closed"},
        {&PickSlotConfig::obstacle_root_radius_m, 0.0F,
         "zero obstacle radius did not fail closed"},
        {&PickSlotConfig::obstacle_root_radius_m, -1.0F,
         "negative obstacle radius did not fail closed"},
        {&PickSlotConfig::obstacle_root_radius_m, nan,
         "NaN obstacle radius did not fail closed"},
        {&PickSlotConfig::obstacle_root_radius_m, infinity,
         "infinite obstacle radius did not fail closed"},
        {&PickSlotConfig::obstacle_root_radius_m, overflow_prone,
         "overflow-prone obstacle radius did not fail closed"},
        {&PickSlotConfig::obstacle_safety_margin_m, 0.0F,
         "zero obstacle safety margin did not fail closed"},
        {&PickSlotConfig::obstacle_safety_margin_m, -1.0F,
         "negative obstacle safety margin did not fail closed"},
        {&PickSlotConfig::obstacle_safety_margin_m, nan,
         "NaN obstacle safety margin did not fail closed"},
        {&PickSlotConfig::obstacle_safety_margin_m, infinity,
         "infinite obstacle safety margin did not fail closed"},
        {&PickSlotConfig::obstacle_safety_margin_m, overflow_prone,
         "overflow-prone obstacle safety margin did not fail closed"},
    };

    for (const MalformedConfigCase& malformed : cases) {
        PickSlotConfig config{};
        config.*(malformed.member) = malformed.value;
        require_malformed_pick_slot_config_rejected(
            config, malformed.message);
    }
}

bool exactly_equal(
    const interaction::Transform& left,
    const interaction::Transform& right) {
    return left.position.x == right.position.x &&
        left.position.y == right.position.y &&
        left.position.z == right.position.z &&
        left.rotation.w == right.rotation.w &&
        left.rotation.x == right.rotation.x &&
        left.rotation.y == right.rotation.y &&
        left.rotation.z == right.rotation.z;
}

void test_frozen_revalidation_uses_exact_transform_without_reselection() {
    using namespace interaction;

    const Transform live_root{vec3(0.0F, 1.25F, 0.0F), quat()};
    const Transform frozen_root{
        vec3(0.80F, 1.25F, 0.0F),
        quat_from_angle_axis(0.37F, vec3(0.0F, 1.0F, 0.0F)),
    };
    const Transform frozen_before = frozen_root;
    const InteractionTarget target = make_multi_slot_target({
        {80U, 0.0F, 0.80F, 0.0F},
        {81U, 0.0F, 0.90F, 0.0F},
    });
    const PickNavigationObstacle blocks_only_frozen_route{
        vec3(0.70F, 1.25F, 0.0F),
        vec3(0.10F, 0.10F, 0.10F),
    };

    const PickSlotSelection authored_selection = select_route(
        live_root.position, target, {blocks_only_frozen_route});
    require(
        authored_selection.reason == PickSlotReason::None,
        "frozen-route fixture did not leave an authored alternative clear");
    const PickSlotReason reason = revalidate_frozen_pick_slot(
        live_root,
        frozen_root,
        target,
        {blocks_only_frozen_route});
    require(
        reason == PickSlotReason::ObstacleBlocked,
        "frozen-route revalidation remapped or selected another slot");
    require(
        exactly_equal(frozen_root, frozen_before),
        "frozen-route revalidation changed the frozen transform");
}

interaction::PickSlotReason fixed_current_y_revalidation_reason() {
    using namespace interaction;

    const Transform live_root{vec3(0.0F, 2.0F, 0.0F), quat()};
    const Transform frozen_root{vec3(1.0F, 0.0F, 0.0F), quat()};
    const InteractionTarget target = make_multi_slot_target({
        {90U, 1.0F, 0.0F, 0.0F},
    });
    const PickNavigationObstacle frozen_xz_at_current_y{
        vec3(1.0F, 2.0F, 0.0F),
        vec3(0.10F, 0.10F, 0.10F),
    };
    return revalidate_frozen_pick_slot(
        live_root,
        frozen_root,
        target,
        {frozen_xz_at_current_y});
}

void test_frozen_obstacle_route_uses_current_y_at_both_endpoints() {
    using namespace interaction;

    const PickSlotConfig config{};
    constexpr float kSlopedMinimumXGap = 0.74F;
    constexpr float kSlopedMinimumYGap = 0.37F;
    const float sloped_minimum_distance = std::sqrt(
        kSlopedMinimumXGap * kSlopedMinimumXGap +
        kSlopedMinimumYGap * kSlopedMinimumYGap);
    require(
        sloped_minimum_distance >
            config.obstacle_root_radius_m +
                config.obstacle_safety_margin_m,
        "different-Y fixture would also block the sloped 3D segment");
    require(
        fixed_current_y_revalidation_reason() ==
            PickSlotReason::ObstacleBlocked,
        "frozen obstacle route used stored frozen Y instead of current root Y");
}

void run_step_3_and_step_7_fast_math_canary() {
    using namespace interaction;

    test_ranking_uses_quantized_route_heading_and_id_tuple();
    test_near_zero_route_uses_mapped_yaw_for_heading();
    test_quantization_rounds_half_millimetres_and_milliradians_up();
    test_nonfinite_and_scaled_overflow_values_reject_without_wrapping();
    test_blocked_nearest_slot_falls_back_to_next_clear_slot();
    test_no_winner_reasons_distinguish_empty_outside_and_blocked();
    test_every_malformed_pick_slot_config_fails_closed();
    test_frozen_revalidation_uses_exact_transform_without_reselection();
    test_frozen_obstacle_route_uses_current_y_at_both_endpoints();

    const PickSlotSelection ranked = select_route(
        vec3(0.0F, 0.0F, 0.0F),
        make_tuple_ranking_target(false));
    const PickSlotSelection half_mm = select_half_millimetre_fixture();
    const PickSlotSelection half_mrad = select_half_milliradian_fixture();
    const PickSlotSelection no_authored = select_no_authored_fixture();
    const PickSlotSelection all_outside = select_all_outside_fixture();
    const PickSlotSelection all_blocked = select_all_blocked_fixture();
    const PickSlotSelection invalid = select_invalid_quantization_fixture();
    std::cout
        << "pick-slot-canary selected="
        << selected_mapped_slot(ranked, "canary ranking selected no slot").id
        << " half-mm=" << half_mm.ordered[0].route_millimetres
        << ',' << half_mm.ordered[1].route_millimetres
        << " half-mrad=" << half_mrad.ordered[0].heading_milliradians
        << ',' << half_mrad.ordered[1].heading_milliradians
        << " no-authored=" << static_cast<unsigned int>(no_authored.reason)
        << " outside=" << static_cast<unsigned int>(all_outside.reason)
        << " blocked=" << static_cast<unsigned int>(all_blocked.reason)
        << " invalid=" << static_cast<unsigned int>(invalid.reason)
        << " frozen-y=" << static_cast<unsigned int>(
               fixed_current_y_revalidation_reason())
        << '\n';
}

bool requested_fast_math_canary_only(int argc, char** argv) {
    if (argc == 1) return false;
    require(
        argc == 2 && std::string(argv[1]) == "--fast-math-canary",
        "expected no argument or --fast-math-canary");
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const bool canary_only = requested_fast_math_canary_only(argc, argv);
    if (!canary_only) {
        test_identity_and_translated_rotated_object_planar_mapping();
        test_object_pitch_and_roll_do_not_tilt_the_root();
        test_live_root_y_is_preserved_exactly();
        test_zero_planar_object_forward_is_invalid_geometry();
        test_rotated_table_blocks_only_the_middle_of_the_segment();
        test_table_expansion_boundary_is_closed_and_next_float_is_clear();
        test_tiny_table_boundary_straddle_is_blocked();
        test_obstacle_safety_envelope_uses_strict_exact_distance();
        test_exact_segment_aabb_corner_distance_beats_expanded_box_shortcut();
        test_obstacle_between_clear_endpoints_blocks_the_segment();
        test_malformed_obstacle_data_fails_closed();
        test_target_is_exempt_but_target_distance_diagnostics_move();
        test_direct_travel_tolerance_boundary_is_inclusive();
    }
    run_step_3_and_step_7_fast_math_canary();
}
