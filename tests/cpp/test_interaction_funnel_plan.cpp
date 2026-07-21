#include "interaction_funnel_plan.h"

#include <cassert>
#include <cmath>

namespace {

interaction::InteractionTarget target_at_origin() {
    interaction::InteractionTarget target{};
    target.object_world = {
        vec3(0.0F, 0.8F, 0.0F),
        quat(1.0F, 0.0F, 0.0F, 0.0F),
    };
    target.object_dimensions = vec3(0.1F, 0.2F, 0.1F);
    target.object_bounds.half_extents_object = vec3(0.05F, 0.1F, 0.05F);
    target.table_world = {
        vec3(100.0F, 0.0F, 100.0F),
        quat(1.0F, 0.0F, 0.0F, 0.0F),
    };
    target.table_size = vec3(1.0F, 0.1F, 1.0F);
    return target;
}

interaction::Transform root_at(float x, float z) {
    return {
        vec3(x, 0.0F, z),
        quat_from_angle_axis(PIf, vec3(0.0F, 1.0F, 0.0F)),
    };
}

void test_direct_plan_uses_data_supported_entry_radius() {
    interaction::FunnelCaptureConfig config{};
    config.maximum_object_radius_m = 0.75F;
    const interaction::FunnelApproachPlan plan =
        interaction::plan_funnel_approach(
            root_at(0.0F, 3.0F), target_at_origin(), {}, config);
    assert(plan.feasible);
    assert(plan.waypoints_world.size() == 2U);
    const float radius = length(vec2(
        plan.entry_world.position.x,
        plan.entry_world.position.z));
    assert(std::abs(radius - 0.75F) < 1.0e-5F);
}

void test_blocked_direct_segment_uses_collision_free_detour() {
    interaction::FunnelCaptureConfig config{};
    config.maximum_object_radius_m = 0.75F;
    config.route.obstacle_root_radius_m = 0.20F;
    const interaction::PickNavigationObstacle obstacle{
        vec3(0.0F, 0.0F, 1.75F),
        vec3(0.30F, 1.0F, 0.30F),
    };
    const auto target = target_at_origin();
    const interaction::FunnelApproachPlan plan =
        interaction::plan_funnel_approach(
            root_at(0.0F, 3.0F), target, {obstacle}, config);
    assert(plan.feasible);
    assert(plan.waypoints_world.size() > 2U);
    for (size_t index = 1U; index < plan.waypoints_world.size(); ++index) {
        assert(interaction::revalidate_frozen_pick_slot(
                   plan.waypoints_world[index - 1U],
                   plan.waypoints_world[index],
                   target,
                   {obstacle},
                   config.route) == interaction::PickSlotReason::None);
    }
}

}  // namespace

int main() {
    test_direct_plan_uses_data_supported_entry_radius();
    test_blocked_direct_segment_uses_collision_free_detour();
    return 0;
}
