#include "g1_arm_joint_metadata.h"
#include "reach_placement.h"
#include "reach_search.h"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace {

quat joint_rotation(const interaction::HingeJoint& joint, float angle) {
    return quat_normalize(quat_mul(
        joint.rest_rotation, quat_from_angle_axis(angle, joint.axis)));
}

interaction::Pose pose_at_step(float delta) {
    interaction::Pose pose{};
    for (quat& rotation : pose.rotations) rotation = quat();
    pose.positions[g1_skeleton::Hips] = vec3(0.0F, 0.82F, 0.0F);
    pose.positions[g1_skeleton::Spine] = vec3(0.0F, 0.10F, 0.0F);
    pose.positions[g1_skeleton::Spine1] = vec3(0.0F, 0.10F, 0.0F);
    pose.positions[g1_skeleton::Spine2] = vec3(0.0F, 0.10F, 0.0F);
    pose.positions[g1_skeleton::LeftShoulderPitch] =
        vec3(0.0039563F, 0.23778F, -0.10022F);
    pose.positions[g1_skeleton::LeftShoulderRoll] =
        vec3(0.0F, -0.013831F, -0.038F);
    pose.positions[g1_skeleton::LeftShoulderYaw] =
        vec3(0.0F, -0.1032F, -0.00624F);
    pose.positions[g1_skeleton::LeftElbow] =
        vec3(0.015783F, -0.080518F, 0.0F);
    pose.positions[g1_skeleton::LeftWristRoll] =
        vec3(0.10F, -0.01F, -0.00188791F);
    pose.positions[g1_skeleton::LeftWristPitch] = vec3(0.038F, 0.0F, 0.0F);
    pose.positions[g1_skeleton::LeftWrist] = vec3(0.046F, 0.0F, 0.0F);
    pose.positions[g1_skeleton::RightShoulderPitch] =
        vec3(0.0039563F, 0.23778F, 0.10021F);
    pose.positions[g1_skeleton::RightShoulderRoll] =
        vec3(0.0F, -0.013831F, 0.038F);
    pose.positions[g1_skeleton::RightShoulderYaw] =
        vec3(0.0F, -0.1032F, 0.00624F);
    pose.positions[g1_skeleton::RightElbow] =
        vec3(0.015783F, -0.080518F, 0.0F);
    pose.positions[g1_skeleton::RightWristRoll] =
        vec3(0.10F, -0.01F, 0.00188791F);
    pose.positions[g1_skeleton::RightWristPitch] = vec3(0.038F, 0.0F, 0.0F);
    pose.positions[g1_skeleton::RightWrist] = vec3(0.046F, 0.0F, 0.0F);
    const float left[7] = {
        0.10F + delta, 0.35F, -0.20F, 0.80F, 0.10F, -0.15F, 0.05F};
    const float right[7] = {
        0.10F + delta, -0.35F, 0.20F, 0.80F, -0.10F, -0.15F, -0.05F};
    for (size_t joint = 0U; joint < 7U; ++joint) {
        pose.rotations[static_cast<size_t>(interaction::kLeftArm[joint].bone)] =
            joint_rotation(interaction::kLeftArm[joint], left[joint]);
        pose.rotations[static_cast<size_t>(interaction::kRightArm[joint].bone)] =
            joint_rotation(interaction::kRightArm[joint], right[joint]);
    }
    return pose;
}

void append_pose(reach::Database& database, const interaction::Pose& pose) {
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        database.positions.insert(database.positions.end(), {
            pose.positions[bone].x, pose.positions[bone].y, pose.positions[bone].z});
        database.velocities.insert(database.velocities.end(), {0, 0, 0});
        database.rotations.insert(database.rotations.end(), {
            pose.rotations[bone].w, pose.rotations[bone].x,
            pose.rotations[bone].y, pose.rotations[bone].z});
        database.angular_velocities.insert(
            database.angular_velocities.end(), {0, 0, 0});
    }
    database.foot_contacts.insert(database.foot_contacts.end(), {1, 1});
}

reach::Pack bilateral_fixture() {
    constexpr size_t frames_per_clip = 20U;
    reach::Pack pack{};
    reach::Database& database = pack.database;
    database.version = 1U;
    database.endian_marker = 0x01020304U;
    database.fps_numerator = 25U;
    database.fps_denominator = 1U;
    database.frame_count = 2U * frames_per_clip;
    database.bone_count = g1_skeleton::BoneCount;
    database.clip_count = 2U;
    database.source_count = 1U;
    database.parents.assign(
        g1_skeleton::kParents.begin(), g1_skeleton::kParents.end());
    database.range_starts = {0, static_cast<int32_t>(frames_per_clip)};
    database.range_stops = {
        static_cast<int32_t>(frames_per_clip),
        static_cast<int32_t>(2U * frames_per_clip)};
    for (size_t clip = 0U; clip < 2U; ++clip) {
        for (size_t frame = 0U; frame < frames_per_clip; ++frame) {
            append_pose(database, pose_at_step(0.035F * static_cast<float>(frame)));
            database.source_frames.push_back(
                static_cast<int32_t>(clip * 100U + frame));
        }
    }
    database.active_hands = {0U, 1U};
    database.augmentations = {0U, 1U};
    database.source_indices = {0U, 0U};
    database.original_indices = {-1, 0};
    database.source_names = {"fixture"};
    for (size_t clip = 0U; clip < 2U; ++clip) {
        const int32_t start = database.range_starts[clip];
        const int32_t stop = database.range_stops[clip];
        const size_t wrist = clip == 0U
            ? static_cast<size_t>(g1_skeleton::LeftWrist)
            : static_cast<size_t>(g1_skeleton::RightWrist);
        const interaction::WorldPose final = interaction::world_pose(
            reach::pose_at_frame(database, stop - 1));
        const interaction::WorldPose before = interaction::world_pose(
            reach::pose_at_frame(database, start + 2));
        const vec3 approach = normalize(
            final.positions[wrist] - before.positions[wrist]);
        database.endpoint_positions.insert(database.endpoint_positions.end(), {
            final.positions[wrist].x,
            final.positions[wrist].y,
            final.positions[wrist].z,
        });
        database.endpoint_rotations.insert(database.endpoint_rotations.end(), {
            final.rotations[wrist].w,
            final.rotations[wrist].x,
            final.rotations[wrist].y,
            final.rotations[wrist].z,
        });
        database.approach_directions.insert(database.approach_directions.end(), {
            approach.x, approach.y, approach.z,
        });
    }
    return pack;
}

reach::SearchResult run_fixture_search(size_t workers) {
    reach::SearchConfig config{};
    config.worker_count = workers;
    config.deadline = std::chrono::seconds(30);
    return reach::search_all(
        bilateral_fixture(),
        {{{1.0F, 0.85F, -0.25F}, quat()}, normalize(vec3(-1, 0, 0))},
        {{vec3(10, 10, 10), quat()}, vec3(0.01F, 0.01F, 0.01F)},
        interaction::EnvironmentGeometry{},
        config);
}

void test_search_processes_every_bilateral_yaw_instance() {
    const reach::SearchResult result = run_fixture_search(2U);
    assert(result.complete);
    assert(result.total == 2U * reach::kYawPlacementCount);
    assert(result.processed == result.total);
    assert(result.evaluations.size() == result.total);
    for (const reach::CompactEvaluation& value : result.evaluations) {
        assert(value.evaluation.poses.empty());
        assert(!value.hand_path.empty());
    }
}

void test_search_order_is_independent_of_worker_count() {
    const reach::SearchResult serial = run_fixture_search(1U);
    const reach::SearchResult parallel = run_fixture_search(4U);
    assert(serial.evaluations.size() == parallel.evaluations.size());
    for (size_t index = 0U; index < serial.evaluations.size(); ++index) {
        assert(serial.evaluations[index].evaluation.candidate.clip ==
               parallel.evaluations[index].evaluation.candidate.clip);
        assert(serial.evaluations[index].evaluation.candidate.yaw_index ==
               parallel.evaluations[index].evaluation.candidate.yaw_index);
        assert(serial.evaluations[index].evaluation.rejection ==
               parallel.evaluations[index].evaluation.rejection);
    }
}

void test_zero_deadline_never_publishes_partial_results_as_complete() {
    reach::SearchConfig config{};
    config.worker_count = 2U;
    config.deadline = std::chrono::steady_clock::duration::zero();
    const reach::SearchResult result = reach::search_all(
        bilateral_fixture(),
        {{{1.0F, 0.85F, -0.25F}, quat()}, normalize(vec3(-1, 0, 0))},
        {{vec3(10, 10, 10), quat()}, vec3(0.01F, 0.01F, 0.01F)},
        interaction::EnvironmentGeometry{},
        config);
    assert(!result.complete);
    assert(result.processed < result.total);
    assert(result.accepted.empty());
}

void test_complete_search_never_exceeds_its_deadline() {
    const auto deadline = std::chrono::milliseconds(20);
    assert(reach::detail::complete_within_deadline(
        24U, 24U, 24U, deadline, deadline));
    assert(!reach::detail::complete_within_deadline(
        24U, 24U, 24U, deadline + std::chrono::nanoseconds(1), deadline));
    assert(!reach::detail::complete_within_deadline(
        23U, 24U, 23U, deadline, deadline));
}

void test_selected_regeneration_matches_compact_metrics() {
    const reach::Pack pack = bilateral_fixture();
    const reach::ExhaustiveQuery query{
        {{1.0F, 0.85F, -0.25F}, quat()}, normalize(vec3(-1, 0, 0))};
    const interaction::OrientedBox object{
        {vec3(10, 10, 10), quat()}, vec3(0.01F, 0.01F, 0.01F)};
    reach::SearchConfig config{};
    const reach::SearchResult result = reach::search_all(
        pack, query, object, interaction::EnvironmentGeometry{}, config);
    assert(result.complete && !result.evaluations.empty());
    const reach::Evaluation regenerated = reach::regenerate(
        pack, result.evaluations.front(), query, object,
        interaction::EnvironmentGeometry{}, config);
    assert(regenerated.rejection ==
           result.evaluations.front().evaluation.rejection);
    assert(!regenerated.poses.empty());
}

}  // namespace

int main() {
    test_search_processes_every_bilateral_yaw_instance();
    test_search_order_is_independent_of_worker_count();
    test_zero_deadline_never_publishes_partial_results_as_complete();
    test_complete_search_never_exceeds_its_deadline();
    test_selected_regeneration_matches_compact_metrics();
}
