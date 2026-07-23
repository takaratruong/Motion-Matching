#include "episode_grasp_provider.h"
#include "episode_reach_planner.h"
#include "interaction_episode.h"

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

void test_required_hand_filter_is_contextual() {
    require(
        episode::reach_hand_allowed(
            reach::Hand::Left, std::nullopt) &&
            episode::reach_hand_allowed(
                reach::Hand::Right, std::nullopt),
        "pickup hand-agnostic mode rejected a hand");
    require(
        episode::reach_hand_allowed(
            reach::Hand::Left, reach::Hand::Left) &&
            !episode::reach_hand_allowed(
                reach::Hand::Right, reach::Hand::Left),
        "placement did not retain the currently attached hand");
}

void test_entry_path_routes_around_blocking_geometry() {
    interaction::EnvironmentGeometry environment{};
    environment.boxes.push_back({
        {vec3(0.0F, 0.5F, 0.0F), quat()},
        vec3(0.40F, 1.0F, 0.40F),
    });
    const vec3 start(-1.0F, 0.0F, 0.0F);
    const vec3 entry(1.0F, 0.0F, 0.0F);
    require(
        !episode::entry_segment_clear(
            start, entry, environment, 0.28F),
        "path fixture was not blocked");
    const auto path = episode::find_entry_path(
        start, entry, environment, 0.28F);
    require(path.has_value(), "visibility graph did not find a detour");
    require(
        path->size() >= 2U && path->back().x == entry.x &&
            path->back().z == entry.z,
        "entry path did not retain detour and certified endpoint");
    const auto certified_start_path = episode::find_entry_path(
        vec3(-0.30F, 0.0F, 0.0F),
        entry,
        environment,
        0.28F);
    require(
        certified_start_path.has_value(),
        "path could not exit a certified start neighborhood");
}

void test_entry_path_rejects_an_unbounded_endpoint_exemption() {
    interaction::EnvironmentGeometry environment{};
    environment.boxes.push_back({
        {vec3(0.0F, 0.5F, 0.0F), quat()},
        vec3(2.0F, 1.0F, 2.0F),
    });
    const auto path = episode::find_entry_path(
        vec3(0.0F, 0.0F, 0.0F),
        vec3(2.0F, 0.0F, 0.0F),
        environment,
        0.28F);
    require(
        !path.has_value(),
        "path accepted an arbitrarily long colliding endpoint prefix");
}

void test_waypoint_arrival_stays_inside_corner_clearance_margin() {
    const episode::EpisodeConfig config{};
    require(
        config.waypoint_reached_m >= 0.08F,
        "waypoint tolerance is below locomotion convergence resolution");
    require(
        config.waypoint_reached_m + 0.05F <=
            episode::kEntryPathCornerMarginM,
        "waypoint follower can turn outside the planner corner margin");
    require(
        config.approach_timeout_seconds >= 20.0F,
        "routed approach timeout cannot cover a furniture detour");
    require(
        !config.attachment.require_lift_for_hold,
        "episode defaults still prevent downward high-shelf pickup");
}

void test_nearest_support_tracks_a_moved_destination() {
    interaction::EnvironmentGeometry environment{};
    environment.boxes.push_back({
        {vec3(0.0F, 0.47F, 0.0F), quat()},
        vec3(1.0F, 0.06F, 1.0F),
    });
    environment.boxes.push_back({
        {vec3(2.0F, 0.98F, 0.0F), quat()},
        vec3(0.8F, 0.04F, 0.8F),
    });
    const vec3 object_size(0.10F, 0.10F, 0.10F);
    const std::optional<size_t> lower =
        episode::find_support_index(
            environment,
            {vec3(0.0F, 0.55F, 0.0F), quat()},
            object_size);
    require(
        lower.has_value() && *lower == 0U,
        "lower destination did not retain its support");
    const std::optional<size_t> upper =
        episode::find_support_index(
            environment,
            {vec3(2.0F, 1.05F, 0.0F), quat()},
            object_size);
    require(
        upper.has_value() && *upper == 1U,
        "moved destination retained stale support metadata");
    require(
        !episode::find_support_index(
            environment,
            {vec3(2.0F, 1.40F, 0.0F), quat()},
            object_size).has_value(),
        "floating destination inherited an unrelated support");

    interaction::EnvironmentGeometry leg_only{};
    leg_only.boxes.push_back({
        {vec3(0.0F, 0.50F, 0.0F), quat()},
        vec3(0.05F, 1.00F, 0.05F),
    });
    require(
        !episode::find_support_index(
            leg_only,
            {vec3(0.0F, 1.05F, 0.0F), quat()},
            object_size).has_value(),
        "vertical furniture member was treated as a support surface");

    interaction::EnvironmentGeometry rotated_slab{};
    rotated_slab.boxes.push_back({
        {
            vec3(0.0F, 0.50F, 0.0F),
            quat_from_angle_axis(
                1.570796327F, vec3(0.0F, 0.0F, 1.0F)),
        },
        vec3(1.0F, 0.04F, 1.0F),
    });
    require(
        !episode::find_support_index(
            rotated_slab,
            {vec3(0.0F, 0.55F, 0.0F), quat()},
            object_size).has_value(),
        "vertically rotated slab was treated as horizontal support");
}

}  // namespace

int main() {
    try {
        test_known_grasp_is_object_relative_and_hand_agnostic();
        test_entry_compatibility_precedes_directness();
        test_ties_are_deterministic_by_clip_then_yaw();
        test_entry_segment_rejects_expanded_obstacle();
        test_required_hand_filter_is_contextual();
        test_entry_path_routes_around_blocking_geometry();
        test_entry_path_rejects_an_unbounded_endpoint_exemption();
        test_waypoint_arrival_stays_inside_corner_clearance_margin();
        test_nearest_support_tracks_a_moved_destination();
        std::cout << "episode reach planner PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "episode reach planner FAILED: " << error.what() << '\n';
        return 1;
    }
}
