#include "episode_grasp_provider.h"
#include "episode_reach_planner.h"
#include "g1_posture_ik_fixture.h"
#include "interaction_episode.h"
#include "reach_placement.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool near(vec3 left, vec3 right, float tolerance = 1.0e-5F) {
    return length(left - right) <= tolerance;
}

void append_pose(
    reach::Database& database,
    const interaction::Pose& pose) {
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        database.positions.insert(database.positions.end(), {
            pose.positions[bone].x,
            pose.positions[bone].y,
            pose.positions[bone].z,
        });
        database.velocities.insert(database.velocities.end(), {
            pose.velocities[bone].x,
            pose.velocities[bone].y,
            pose.velocities[bone].z,
        });
        database.rotations.insert(database.rotations.end(), {
            pose.rotations[bone].w,
            pose.rotations[bone].x,
            pose.rotations[bone].y,
            pose.rotations[bone].z,
        });
        database.angular_velocities.insert(
            database.angular_velocities.end(), {
                pose.angular_velocities[bone].x,
                pose.angular_velocities[bone].y,
                pose.angular_velocities[bone].z,
            });
    }
    database.foot_contacts.insert(
        database.foot_contacts.end(),
        pose.foot_contacts.begin(),
        pose.foot_contacts.end());
}

interaction::Pose pose_with_shoulder_pitch(float delta) {
    interaction::Pose pose = g1_posture_fixture::make_pose();
    interaction::UpperBodyAngles angles =
        interaction::decompose_upper_body(
            pose, interaction::Hand::Left);
    angles[interaction::kWaist.size()] += delta;
    interaction::apply_upper_body(
        pose, interaction::Hand::Left, angles);
    return pose;
}

reach::Pack fallback_pack() {
    constexpr size_t kOutboundFrames = 20U;
    constexpr size_t kReturnFrames = 3U;
    reach::Pack pack{};
    reach::Database& database = pack.database;
    database.version = 2U;
    database.endian_marker = 0x01020304U;
    database.fps_numerator = 25U;
    database.fps_denominator = 1U;
    database.bone_count = g1_skeleton::BoneCount;
    database.clip_count = 3U;
    database.source_count = 1U;
    database.parents.assign(
        g1_skeleton::kParents.begin(),
        g1_skeleton::kParents.end());
    database.range_starts = {
        0,
        static_cast<int32_t>(kOutboundFrames),
        static_cast<int32_t>(
            2U * kOutboundFrames + kReturnFrames),
    };
    database.range_stops = {
        static_cast<int32_t>(kOutboundFrames),
        static_cast<int32_t>(
            2U * kOutboundFrames + kReturnFrames),
        static_cast<int32_t>(
            3U * kOutboundFrames + 2U * kReturnFrames),
    };
    database.contact_frames = {
        static_cast<int32_t>(kOutboundFrames - 1U),
        static_cast<int32_t>(2U * kOutboundFrames - 1U),
        static_cast<int32_t>(
            3U * kOutboundFrames + kReturnFrames - 1U),
    };
    for (size_t clip = 0U; clip < 3U; ++clip) {
        for (size_t frame = 0U; frame < kOutboundFrames; ++frame) {
            const float u = static_cast<float>(frame) /
                static_cast<float>(kOutboundFrames - 1U);
            append_pose(
                database,
                pose_with_shoulder_pitch(-0.25F + 0.50F * u));
            database.source_frames.push_back(
                static_cast<int32_t>(
                    1000U * clip + frame));
        }
        if (clip == 0U) continue;
        for (size_t frame = 0U; frame < kReturnFrames; ++frame) {
            interaction::Pose pose = pose_with_shoulder_pitch(
                0.10F - 0.20F * static_cast<float>(frame));
            if (clip == 1U && frame == 0U) {
                pose.positions[g1_skeleton::RightWrist].x =
                    std::numeric_limits<float>::quiet_NaN();
            }
            append_pose(database, pose);
            database.source_frames.push_back(
                static_cast<int32_t>(
                    1000U * clip + kOutboundFrames + frame));
        }
    }
    database.frame_count = static_cast<uint32_t>(
        3U * kOutboundFrames + 2U * kReturnFrames);
    database.active_hands.assign(
        3U, static_cast<uint8_t>(reach::Hand::Left));
    database.augmentations.assign(
        3U, static_cast<uint8_t>(reach::Augmentation::Captured));
    database.source_indices.assign(3U, 0U);
    database.original_indices.assign(3U, -1);
    database.source_names = {"fallback_fixture"};
    for (size_t clip = 0U; clip < 3U; ++clip) {
        const interaction::WorldPose contact =
            interaction::world_pose(reach::pose_at_frame(
                database, database.contact_frames[clip]));
        const interaction::WorldPose before =
            interaction::world_pose(reach::pose_at_frame(
                database, database.contact_frames[clip] - 5));
        const vec3 approach = normalize(
            contact.positions[g1_skeleton::LeftWrist] -
            before.positions[g1_skeleton::LeftWrist]);
        const interaction::Transform endpoint{
            contact.positions[g1_skeleton::LeftWrist],
            contact.rotations[g1_skeleton::LeftWrist],
        };
        database.endpoint_positions.insert(
            database.endpoint_positions.end(), {
                endpoint.position.x,
                endpoint.position.y,
                endpoint.position.z,
            });
        database.endpoint_rotations.insert(
            database.endpoint_rotations.end(), {
                endpoint.rotation.w,
                endpoint.rotation.x,
                endpoint.rotation.y,
                endpoint.rotation.z,
            });
        database.approach_directions.insert(
            database.approach_directions.end(), {
                approach.x,
                approach.y,
                approach.z,
            });
    }
    pack.features.version = 2U;
    pack.features.endian_marker = 0x01020304U;
    pack.features.clip_count = 3U;
    pack.features.dimension = 10U;
    for (size_t clip = 0U; clip < 3U; ++clip) {
        const size_t position = 3U * clip;
        const size_t rotation = 4U * clip;
        pack.features.values.insert(pack.features.values.end(), {
            database.endpoint_positions[position],
            database.endpoint_positions[position + 1U],
            database.endpoint_positions[position + 2U],
            database.approach_directions[position],
            database.approach_directions[position + 1U],
            database.approach_directions[position + 2U],
            database.endpoint_rotations[rotation],
            database.endpoint_rotations[rotation + 1U],
            database.endpoint_rotations[rotation + 2U],
            database.endpoint_rotations[rotation + 3U],
        });
    }
    return pack;
}

reach::SearchResult compact_outbound_candidates(
    const reach::Pack& pack,
    const reach::ExhaustiveQuery& query,
    const interaction::OrientedBox& object,
    const reach::SearchConfig& config) {
    reach::SearchResult result{};
    result.complete = true;
    result.total = 3U;
    result.processed = 3U;
    for (size_t clip = 0U; clip < 3U; ++clip) {
        const reach::Candidate candidate{
            clip, 0U, reach::placement_yaw(0U), 0.0F, 0.0F};
        reach::Query hand_query{};
        hand_query.hand = reach::Hand::Left;
        hand_query.target = query.target;
        hand_query.approach_world = query.approach_world;
        reach::Evaluation evaluation = reach::evaluate_candidate(
            pack,
            candidate,
            hand_query,
            object,
            interaction::EnvironmentGeometry{},
            config.coverage,
            config.collision);
        require(
            evaluation.rejection == reach::Rejection::None,
            "fallback fixture outbound was not accepted");
        reach::CompactEvaluation compact{};
        compact.evaluation = std::move(evaluation);
        compact.evaluation.poses.clear();
        result.evaluations.push_back(std::move(compact));
        result.accepted.push_back(clip);
    }
    return result;
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

void test_plan_skips_empty_return_and_falls_back_after_failed_preflight() {
    const reach::Pack pack = fallback_pack();
    const interaction::Transform target =
        reach::endpoint_transform(pack.database, 0U);
    const reach::ExhaustiveQuery query{
        target,
        reach::approach_direction(pack.database, 0U),
    };
    const interaction::OrientedBox object{
        {vec3(10.0F, 10.0F, 10.0F), quat()},
        vec3(0.01F, 0.01F, 0.01F),
    };
    reach::SearchConfig config{};
    config.coverage.accepted_approach_radians = 3.141592654F;
    config.coverage.accepted_orientation_radians = 3.141592654F;
    const reach::SearchResult compact =
        compact_outbound_candidates(pack, query, object, config);
    require(
        compact.accepted.size() == 3U &&
        compact.evaluations[0].evaluation.candidate.clip == 0U,
        "empty-return identity did not remain in outbound coverage");
    episode::GraspCandidate grasp{};
    grasp.hand_world = target;
    grasp.approach_world = query.approach_world;
    grasp.grasp_id = 7U;

    const std::optional<episode::ReachPlan> plan =
        episode::choose_reach_plan(
            pack,
            compact,
            query,
            object,
            interaction::EnvironmentGeometry{},
            config,
            reach::pose_at_frame(pack.database, 0),
            grasp);

    require(plan.has_value(), "planner did not reach valid fallback return");
    require(
        plan->reach.candidate.clip == 2U,
        "planner selected empty or failed return instead of fallback");
    require(
        plan->return_poses.size() == 4U,
        "planner did not retain contact plus recorded return poses");
    require(
        g1_posture_fixture::same_pose(
            plan->return_poses.front(), plan->reach.poses.back()),
        "stored return did not begin at selected solved contact");
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
        test_plan_skips_empty_return_and_falls_back_after_failed_preflight();
        std::cout << "episode reach planner PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "episode reach planner FAILED: " << error.what() << '\n';
        return 1;
    }
}
