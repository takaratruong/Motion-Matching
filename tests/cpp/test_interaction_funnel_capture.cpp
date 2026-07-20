#include "interaction_funnel_capture.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846F;

interaction::Transform root_at(float x, float z) {
    return {
        vec3(x, 0.0F, z),
        quat(1.0F, 0.0F, 0.0F, 0.0F),
    };
}

interaction::InteractionTarget target_at_origin() {
    interaction::InteractionTarget target{};
    target.object_world = {
        vec3(0.0F, 0.8F, 0.0F),
        quat(1.0F, 0.0F, 0.0F, 0.0F),
    };
    target.object_bounds.half_extents_object = vec3(0.05F, 0.1F, 0.05F);
    target.table_world = {
        vec3(100.0F, 0.0F, 100.0F),
        quat(1.0F, 0.0F, 0.0F, 0.0F),
    };
    target.table_size = vec3(1.0F, 0.1F, 1.0F);
    return target;
}

float yaw_of(quat rotation) {
    const vec3 forward = quat_mul_vec3(rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(forward.x, forward.z);
}

void test_direct_bearing_is_clamped_to_annulus_and_faces_object() {
    const auto result = interaction::select_funnel_capture(
        root_at(0.0F, 2.0F), target_at_origin(), {});
    assert(result.reason == interaction::PickSlotReason::None);
    assert(result.candidate_index == 0);
    assert(std::abs(result.target_world.position.x) < 1.0e-6F);
    assert(std::abs(result.target_world.position.z - 1.0F) < 1.0e-6F);
    assert(std::abs(std::abs(yaw_of(result.target_world.rotation)) - kPi) < 1.0e-6F);

    const auto near = interaction::select_funnel_capture(
        root_at(0.0F, 0.2F), target_at_origin(), {});
    assert(near.reason == interaction::PickSlotReason::None);
    assert(std::abs(near.target_world.position.z - 0.45F) < 1.0e-6F);
}

void test_blocked_direct_connector_uses_positive_angular_offset_first() {
    interaction::FunnelCaptureConfig config{};
    config.route.obstacle_root_radius_m = 0.05F;
    config.route.maximum_direct_travel_m = 2.0F;
    const std::vector<interaction::PickNavigationObstacle> obstacles{
        {vec3(0.0F, 0.0F, 1.5F), vec3(0.05F, 0.05F, 0.05F)},
    };
    const auto result = interaction::select_funnel_capture(
        root_at(0.0F, 2.0F), target_at_origin(), obstacles, config);
    assert(result.reason == interaction::PickSlotReason::None);
    assert(result.candidate_index == 1);
    assert(result.target_world.position.x > 0.0F);
}

void test_selection_does_not_read_authored_interaction_slots() {
    interaction::InteractionTarget first = target_at_origin();
    interaction::GraspAffordance affordance{};
    affordance.interaction_slots.push_back({91U, 100.0F, -50.0F, 2.0F});
    first.affordances.push_back(affordance);
    interaction::InteractionTarget second = first;
    second.affordances.front().interaction_slots.clear();

    const auto left = interaction::select_funnel_capture(
        root_at(0.0F, 2.0F), first, {});
    const auto right = interaction::select_funnel_capture(
        root_at(0.0F, 2.0F), second, {});
    assert(left.reason == interaction::PickSlotReason::None);
    assert(right.reason == interaction::PickSlotReason::None);
    assert(left.target_world.position.x == right.target_world.position.x);
    assert(left.target_world.position.z == right.target_world.position.z);
}

void test_rejects_when_every_connector_is_unsafe() {
    interaction::FunnelCaptureConfig config{};
    config.route.obstacle_root_radius_m = 0.05F;
    const std::vector<interaction::PickNavigationObstacle> obstacles{
        {vec3(0.0F, 0.0F, 0.0F), vec3(10.0F, 1.0F, 10.0F)},
    };
    const auto result = interaction::select_funnel_capture(
        root_at(0.0F, 2.0F), target_at_origin(), obstacles, config);
    assert(result.reason != interaction::PickSlotReason::None);
}

void test_builds_exact_24_value_object_local_condition() {
    interaction::InteractionTarget target = target_at_origin();
    target.object_dimensions = vec3(0.1F, 0.2F, 0.3F);
    target.table_world.position.y = 0.25F;
    interaction::GraspAffordance affordance{};
    affordance.hand = interaction::Hand::Right;
    affordance.hand_in_object = {
        vec3(1.0F, 2.0F, 3.0F),
        quat(1.0F, 0.0F, 0.0F, 0.0F),
    };
    affordance.approach_direction_object = vec3(3.0F, 9.0F, 4.0F);
    const auto condition = interaction::build_funnel_condition(
        root_at(0.0F, 1.0F), vec3(0.25F, 7.0F, -0.5F), target, affordance);
    assert(condition.has_value());
    const std::array<float, 24> expected{
        0.0F, 1.0F,
        1.0F, 2.0F, 3.0F,
        1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F,
        0.6F, 0.8F,
        0.1F, 0.2F, 0.3F,
        0.55F, 2.0F,
        0.0F, 1.0F, 0.0F, 1.0F, 0.25F, -0.5F,
    };
    for (size_t index = 0U; index < expected.size(); ++index) {
        assert(std::abs((*condition)[index] - expected[index]) < 1.0e-6F);
    }
}

}  // namespace

int main() {
    test_direct_bearing_is_clamped_to_annulus_and_faces_object();
    test_blocked_direct_connector_uses_positive_angular_offset_first();
    test_selection_does_not_read_authored_interaction_slots();
    test_rejects_when_every_connector_is_unsafe();
    test_builds_exact_24_value_object_local_condition();
    return 0;
}
