#include "episode_grasp_provider.h"
#include "episode_reach_planner.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool near(vec3 left, vec3 right, float tolerance = 1.0e-5F) {
    return length(left - right) <= tolerance;
}

void test_known_grasp_is_object_relative_and_hand_agnostic() {
    const interaction::Transform local_grasp{
        vec3(0.08F, 0.02F, -0.01F),
        quat_from_angle_axis(0.4F, vec3(0.0F, 1.0F, 0.0F))};
    const episode::KnownGraspProvider provider(
        local_grasp, vec3(-1.0F, 0.0F, 0.0F));
    const episode::ObjectSnapshot object{
        7U,
        {
            vec3(2.0F, 0.7F, -1.0F),
            quat_from_angle_axis(0.5F, vec3(0.0F, 1.0F, 0.0F)),
        },
        vec3(0.10F, 0.12F, 0.08F),
    };

    const std::vector<episode::GraspCandidate> grasps =
        provider.query(object);

    require(grasps.size() == 1U, "known provider changed grasp count");
    const interaction::Transform expected =
        interaction::compose(object.world, local_grasp);
    require(
        near(grasps.front().hand_world.position, expected.position),
        "known grasp position was not object-relative");
    require(
        !grasps.front().preferred_hand.has_value(),
        "known grasp unexpectedly excluded one hand");
    require(
        near(
            grasps.front().approach_world,
            quat_mul_vec3(
                object.world.rotation, vec3(-1.0F, 0.0F, 0.0F))),
        "known grasp approach was not object-relative");
}

void test_entry_compatibility_precedes_directness() {
    const episode::ReachPlanCost near_hooked{
        0.30F, 0.05F, 0.01F, 0.20F, 4U, 0U};
    const episode::ReachPlanCost far_direct{
        2.00F, 0.00F, 0.00F, 0.00F, 1U, 0U};

    require(
        episode::reach_plan_cost_less(near_hooked, far_direct),
        "directness incorrectly outranked playable entry distance");
    require(
        !episode::reach_plan_cost_less(far_direct, near_hooked),
        "entry ranking was not antisymmetric");
}

void test_ties_are_deterministic_by_clip_then_yaw() {
    episode::ReachPlanCost clip_first{
        1.0F, 0.2F, 0.1F, 0.05F, 3U, 9U};
    episode::ReachPlanCost yaw_second = clip_first;
    yaw_second.clip = 4U;
    yaw_second.yaw_index = 0U;
    require(
        episode::reach_plan_cost_less(clip_first, yaw_second),
        "clip tie break was not deterministic");

    episode::ReachPlanCost yaw_first = clip_first;
    yaw_first.yaw_index = 2U;
    require(
        episode::reach_plan_cost_less(yaw_first, clip_first),
        "yaw tie break was not deterministic");
}

void test_entry_segment_rejects_expanded_obstacle() {
    interaction::EnvironmentGeometry environment{};
    environment.boxes.push_back({
        {vec3(0.0F, 0.5F, 0.0F), quat()},
        vec3(0.40F, 1.0F, 0.40F),
    });
    require(
        !episode::entry_segment_clear(
            vec3(-1.0F, 0.0F, 0.0F),
            vec3(1.0F, 0.0F, 0.0F),
            environment,
            0.28F),
        "entry segment passed through expanded obstacle");
    require(
        episode::entry_segment_clear(
            vec3(-1.0F, 0.0F, 1.0F),
            vec3(1.0F, 0.0F, 1.0F),
            environment,
            0.28F),
        "clear entry segment was rejected");
}

}  // namespace

int main() {
    try {
        test_known_grasp_is_object_relative_and_hand_agnostic();
        test_entry_compatibility_precedes_directness();
        test_ties_are_deterministic_by_clip_then_yaw();
        test_entry_segment_rejects_expanded_obstacle();
        std::cout << "episode reach planner PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "episode reach planner FAILED: " << error.what() << '\n';
        return 1;
    }
}
