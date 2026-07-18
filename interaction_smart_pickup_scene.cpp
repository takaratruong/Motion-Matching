#include "interaction_smart_pickup_scene.h"

#include <array>
#include <stdexcept>

namespace interaction {
namespace {

constexpr float kDestinationTranslationZ = 1.20F;

constexpr std::array<SmartPickupSlotProvenance, 3> kSlotProvenance{{
    {1U, "pickup_table__alcohol_10__005", 125, 150},
    {2U, "pickup_table__alcohol_13__005", 118, 143},
    {3U, "pickup_table__apple_1__000", 90, 115},
}};

Transform support_plane_for(const InteractionTarget& target) {
    return compose(
        target.table_world,
        Transform{
            vec3(0.0F, 0.5F * target.table_size.y, 0.0F), quat()});
}

vec3 support_point_in_object(
    const InteractionTarget& target,
    const Transform& surface_world) {
    const vec3 surface_normal = quat_mul_vec3(
        surface_world.rotation, vec3(0.0F, 1.0F, 0.0F));
    const float projection_distance = dot(
        surface_world.position - target.object_world.position,
        surface_normal);
    const vec3 projected_support_world =
        target.object_world.position +
        projection_distance * surface_normal;
    return compose(
        inverse(target.object_world),
        Transform{projected_support_world, quat()}).position;
}

}  // namespace

InteractionTarget make_smart_pickup_demo_target() {
    InteractionTarget target;
    target.handle = {1U, 1U};
    target.object_profile_id = 1U;
    target.state = ObjectState::Free;
    target.table_world = Transform{
        vec3(0.0F, 0.360757500F, 3.0F),
        quat(1.0F, 0.0F, 0.0F, 0.0F)};
    target.table_size = vec3(
        2.0F, 0.0399999991F, 0.600000024F);
    target.object_world = Transform{
        vec3(0.00394439697F, 0.503655553F, 2.77000808716F),
        quat(
            -0.0669774629F,
            0.670088462F,
            0.0799996098F,
            -0.734911910F)};
    target.object_dimensions = vec3(
        0.0645366386F, 0.0645366609F, 0.240097240F);
    target.object_bounds = {
        vec3(), 0.5F * target.object_dimensions};

    GraspAffordance affordance;
    affordance.id = 1U;
    affordance.hand = Hand::Right;
    affordance.hand_in_object = Transform{
        vec3(0.0930671170F, -0.119263843F, 0.0375832170F),
        quat(
            0.308746904F,
            0.0609171167F,
            -0.155041456F,
            0.936443567F)};
    affordance.approach_direction_object =
        vec3(-0.997760296F, 0.0F, 0.0668911785F);
    affordance.clearance_radius = 0.04F;
    affordance.interaction_slots = {
        {1U, -0.396769345F, -0.0414382927F, 1.38969707F},
        {2U, -0.402728528F, -0.244846597F, 1.71573567F},
        {3U, -0.467868507F, -0.194983453F, 1.29209125F},
    };
    target.affordances.push_back(affordance);
    return target;
}

PlacementSurface make_smart_pickup_demo_destination_surface(
    const InteractionTarget& source_target) {
    const Transform source_surface = support_plane_for(source_target);

    PlacementSurface destination;
    destination.handle = {2U, 1U};
    destination.support_volume_world = source_target.table_world;
    destination.support_volume_world.position.z +=
        kDestinationTranslationZ;
    destination.support_volume_size = source_target.table_size;
    destination.surface_world = source_surface;
    destination.surface_world.position.z += kDestinationTranslationZ;
    destination.half_extent_x_m = 0.5F * source_target.table_size.x;
    destination.half_extent_z_m = 0.5F * source_target.table_size.z;
    destination.overhead_clearance_m = 2.00F;

    PlaceAffordance affordance;
    affordance.id = 1U;
    affordance.object_in_surface = compose(
        inverse(source_surface), source_target.object_world);
    affordance.support_point_object = support_point_in_object(
        source_target, source_surface);
    affordance.approach_direction_surface = vec3(0.0F, 1.0F, 0.0F);
    affordance.clearance_radius =
        source_target.affordances.front().clearance_radius;
    destination.affordances.push_back(affordance);

    const PlacementFit fit = evaluate_placement_fit(
        destination,
        destination.affordances.front(),
        source_target.object_bounds);
    if (!fit.accepted) {
        throw std::runtime_error(
            "smart pickup destination is not placement-fit");
    }
    return destination;
}

const std::array<SmartPickupSlotProvenance, 3>&
smart_pickup_demo_slot_provenance() {
    return kSlotProvenance;
}

}  // namespace interaction
