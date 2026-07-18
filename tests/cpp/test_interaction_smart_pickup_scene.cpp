#include "interaction_place_target.h"
#include "interaction_smart_pickup_scene.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

constexpr float kDerivedTolerance = 1.0e-6F;
constexpr float kDestinationTranslationZ = 1.20F;

uint32_t float_bits(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool exact(float left, float right) {
    return float_bits(left) == float_bits(right);
}

bool exact(vec3 left, vec3 right) {
    return exact(left.x, right.x) &&
           exact(left.y, right.y) &&
           exact(left.z, right.z);
}

bool exact(quat left, quat right) {
    return exact(left.w, right.w) &&
           exact(left.x, right.x) &&
           exact(left.y, right.y) &&
           exact(left.z, right.z);
}

bool exact(
    const interaction::Transform& left,
    const interaction::Transform& right) {
    return exact(left.position, right.position) &&
           exact(left.rotation, right.rotation);
}

bool near(float left, float right) {
    return std::abs(left - right) <= kDerivedTolerance;
}

bool near(vec3 left, vec3 right) {
    return near(left.x, right.x) &&
           near(left.y, right.y) &&
           near(left.z, right.z);
}

bool near_direct(quat left, quat right) {
    return near(left.w, right.w) &&
           near(left.x, right.x) &&
           near(left.y, right.y) &&
           near(left.z, right.z);
}

bool near(quat left, quat right) {
    return near_direct(left, right) || near_direct(left, -right);
}

bool near(
    const interaction::Transform& left,
    const interaction::Transform& right) {
    return near(left.position, right.position) &&
           near(left.rotation, right.rotation);
}

interaction::Transform support_plane_for(
    const interaction::InteractionTarget& target) {
    return interaction::compose(
        target.table_world,
        interaction::Transform{
            vec3(0.0F, 0.5F * target.table_size.y, 0.0F), quat()});
}

void test_frozen_beer_target_and_provenance() {
    using namespace interaction;

    using ProvenanceRows = std::array<SmartPickupSlotProvenance, 3>;
    static_assert(std::is_same_v<
        decltype(make_smart_pickup_demo_target()), InteractionTarget>);
    static_assert(std::is_same_v<
        decltype(make_smart_pickup_demo_destination_surface(
            std::declval<const InteractionTarget&>())),
        PlacementSurface>);
    static_assert(std::is_same_v<
        decltype(smart_pickup_demo_slot_provenance()),
        const ProvenanceRows&>);

    const InteractionTarget target = make_smart_pickup_demo_target();
    assert(target.handle == (TargetHandle{1U, 1U}));
    assert(target.object_profile_id == 1U);
    assert(target.state == ObjectState::Free);
    assert(target.owner_request == 0U);

    assert(exact(
        target.table_world.position,
        vec3(0.0F, 0.360757500F, 3.0F)));
    assert(exact(target.table_world.rotation, quat(1.0F, 0.0F, 0.0F, 0.0F)));
    assert(exact(
        target.table_size,
        vec3(2.0F, 0.0399999991F, 0.600000024F)));
    assert(exact(
        target.object_world.position,
        vec3(0.00394439697F, 0.503655553F, 2.77000808716F)));
    assert(exact(
        target.object_world.rotation,
        quat(
            -0.0669774629F,
            0.670088462F,
            0.0799996098F,
            -0.734911910F)));
    assert(exact(
        target.object_dimensions,
        vec3(0.0645366386F, 0.0645366609F, 0.240097240F)));
    assert(exact(target.object_bounds.center_object, vec3()));
    assert(exact(
        target.object_bounds.half_extents_object,
        0.5F * target.object_dimensions));

    assert(target.affordances.size() == 1U);
    const GraspAffordance& affordance = target.affordances.front();
    assert(affordance.id == 1U);
    assert(affordance.hand == Hand::Right);
    assert(exact(
        affordance.hand_in_object.position,
        vec3(0.0930671170F, -0.119263843F, 0.0375832170F)));
    assert(exact(
        affordance.hand_in_object.rotation,
        quat(
            0.308746904F,
            0.0609171167F,
            -0.155041456F,
            0.936443567F)));
    assert(exact(
        affordance.approach_direction_object,
        vec3(-0.997760296F, 0.0F, 0.0668911785F)));
    assert(exact(affordance.clearance_radius, 0.04F));

    const auto& slots = affordance.interaction_slots;
    assert(slots.size() == 3U);
    constexpr std::array<uint32_t, 3> expected_slot_ids{
        1U, 2U, 3U};
    constexpr std::array<uint32_t, 3> expected_root_x_bits{
        0xbecb255aU, 0xbece326fU, 0xbeef8c76U};
    constexpr std::array<uint32_t, 3> expected_root_z_bits{
        0xbd29bb33U, 0xbe7ab911U, 0xbe47a9beU};
    constexpr std::array<uint32_t, 3> expected_root_yaw_bits{
        0x3fb1e198U, 0x3fdb9d3aU, 0x3fa5633fU};
    for (size_t index = 0; index < slots.size(); ++index) {
        assert(slots[index].id == expected_slot_ids[index]);
        assert(float_bits(slots[index].root_x_object_m) ==
               expected_root_x_bits[index]);
        assert(float_bits(slots[index].root_z_object_m) ==
               expected_root_z_bits[index]);
        assert(float_bits(slots[index].root_yaw_object_radians) ==
               expected_root_yaw_bits[index]);
        assert(std::isfinite(slots[index].root_x_object_m));
        assert(std::isfinite(slots[index].root_z_object_m));
        assert(std::isfinite(slots[index].root_yaw_object_radians));
        if (index != 0U) {
            assert(slots[index - 1U].id < slots[index].id);
        }
    }

    TargetRegistry registry;
    const TargetHandle validated = registry.upsert(target);
    assert(validated == target.handle);
    assert(registry.find(validated) != nullptr);

    const ProvenanceRows& provenance =
        smart_pickup_demo_slot_provenance();
    assert(provenance.size() == slots.size());
    constexpr std::array<std::string_view, 3> expected_sequence_ids{
        "pickup_table__alcohol_10__005",
        "pickup_table__alcohol_13__005",
        "pickup_table__apple_1__000"};
    constexpr std::array<int, 3> expected_entry_local_frames{
        125, 118, 90};
    constexpr std::array<int, 3> expected_contact_local_frames{
        150, 143, 115};
    std::array<std::string_view, 3> sequence_ids{};
    for (size_t index = 0; index < provenance.size(); ++index) {
        const SmartPickupSlotProvenance& row = provenance[index];
        assert(row.slot_id == expected_slot_ids[index]);
        assert(row.slot_id == slots[index].id);
        assert(row.sequence_id != nullptr);
        sequence_ids[index] = row.sequence_id;
        assert(sequence_ids[index] == expected_sequence_ids[index]);
        assert(row.entry_local_frame == expected_entry_local_frames[index]);
        assert(row.contact_local_frame ==
               expected_contact_local_frames[index]);
        for (size_t earlier = 0; earlier < index; ++earlier) {
            assert(sequence_ids[earlier] != sequence_ids[index]);
        }
    }
}

void test_destination_is_derived_without_mutating_source() {
    using namespace interaction;

    InteractionTarget source = make_smart_pickup_demo_target();
    const vec3 input_translation(0.37F, 0.0F, -0.25F);
    source.table_world.position =
        source.table_world.position + input_translation;
    source.object_world.position =
        source.object_world.position + input_translation;
    source.table_size.x = 2.20F;
    source.table_size.z = 0.80F;
    source.object_world.position.x += 0.05F;
    source.object_world.position.z += 0.10F;
    Transform varied_object_in_surface = compose(
        inverse(support_plane_for(source)), source.object_world);
    varied_object_in_surface.rotation = quat_normalize(quat_mul(
        quat_from_angle_axis(0.17F, vec3(0.0F, 1.0F, 0.0F)),
        varied_object_in_surface.rotation));
    source.table_world.rotation = quat_from_angle_axis(
        0.31F, vec3(0.0F, 1.0F, 0.0F));
    source.object_world = compose(
        support_plane_for(source), varied_object_in_surface);
    source.affordances.front().clearance_radius = 0.03F;
    const InteractionTarget snapshot = source;

    const Transform source_surface = support_plane_for(source);
    const Transform expected_object_in_surface = compose(
        inverse(source_surface), source.object_world);
    Transform expected_support_volume = source.table_world;
    expected_support_volume.position.z =
        source.table_world.position.z + kDestinationTranslationZ;
    Transform expected_surface = source_surface;
    expected_surface.position.z =
        source_surface.position.z + kDestinationTranslationZ;
    Transform expected_object_world = source.object_world;
    expected_object_world.position.z =
        source.object_world.position.z + kDestinationTranslationZ;

    const PlacementSurface destination =
        make_smart_pickup_demo_destination_surface(source);
    assert(same_interaction_target_snapshot(source, snapshot));
    assert(exact(
        destination.support_volume_world, expected_support_volume));
    assert(exact(destination.support_volume_size, source.table_size));
    assert(exact(destination.surface_world, expected_surface));
    assert(exact(
        destination.half_extent_x_m, 0.5F * source.table_size.x));
    assert(exact(
        destination.half_extent_z_m, 0.5F * source.table_size.z));

    assert(destination.affordances.size() == 1U);
    const PlaceAffordance& affordance = destination.affordances.front();
    assert(near(
        affordance.object_in_surface, expected_object_in_surface));
    assert(exact(
        affordance.clearance_radius,
        source.affordances.front().clearance_radius));
    assert(near(
        placement_goal_world(destination, affordance.object_in_surface),
        expected_object_world));

    const PlacementFit fit = evaluate_placement_fit(
        destination, affordance, source.object_bounds);
    assert(fit.accepted);
    assert(fit.reason == Reason::None);
    assert(fit.footprint_valid);
    assert(fit.overhead_valid);
    assert(same_interaction_target_snapshot(source, snapshot));
}

}  // namespace

int main() {
    test_frozen_beer_target_and_provenance();
    test_destination_is_derived_without_mutating_source();
}
