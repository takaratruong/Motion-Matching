#include "g1_arm_joint_metadata.h"
#include "reach_placement.h"
#include "reach_search.h"
#include "reach_straight_approach.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

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
    database.version = 2U;
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
    database.contact_frames = {
        static_cast<int32_t>(frames_per_clip - 1U),
        static_cast<int32_t>(2U * frames_per_clip - 1U)};
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
    config.coverage.accepted_position_m = 10.0F;
    config.coverage.accepted_approach_radians = 3.141592654F;
    config.coverage.accepted_orientation_radians = 3.141592654F;
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
    assert(serial.accepted.size() == parallel.accepted.size());
    assert(serial.accepted.size() > 1U);
    for (size_t index = 0U; index < serial.accepted.size(); ++index) {
        const reach::Evaluation& serial_evaluation =
            serial.evaluations[serial.accepted[index]].evaluation;
        const reach::Evaluation& parallel_evaluation =
            parallel.evaluations[parallel.accepted[index]].evaluation;
        assert(serial_evaluation.candidate.clip ==
               parallel_evaluation.candidate.clip);
        assert(serial_evaluation.candidate.yaw_index ==
               parallel_evaluation.candidate.yaw_index);
        if (index > 0U) {
            const reach::Evaluation& previous =
                serial.evaluations[serial.accepted[index - 1U]].evaluation;
            assert(!reach::detail::accepted_quality_less(
                serial_evaluation, previous));
        }
    }
}

void test_cancelled_search_returns_without_processing_the_pack() {
    const reach::Pack pack = bilateral_fixture();
    reach::SearchConfig config{};
    config.cancellation = std::make_shared<std::atomic_bool>(true);
    const reach::SearchResult result = reach::search_all(
        pack,
        {{{1.0F, 0.85F, -0.25F}, quat()}, normalize(vec3(-1, 0, 0))},
        {{vec3(10, 10, 10), quat()}, vec3(0.01F, 0.01F, 0.01F)},
        interaction::EnvironmentGeometry{},
        config);
    assert(!result.complete);
    assert(result.processed == 0U);
    assert(result.evaluations.empty());
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

void test_search_rejects_nonunit_approach() {
    bool threw = false;
    try {
        reach::SearchConfig config{};
        (void)reach::search_all(
            bilateral_fixture(),
            {{{1.0F, 0.85F, -0.25F}, quat()}, vec3(-2.0F, 0.0F, 0.0F)},
            {{vec3(10, 10, 10), quat()}, vec3(0.01F, 0.01F, 0.01F)},
            interaction::EnvironmentGeometry{},
            config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
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
    const reach::Evaluation& compact =
        result.evaluations.front().evaluation;
    assert(std::abs(
        regenerated.backtrack_ratio - compact.backtrack_ratio) <= 1.0e-5F);
    assert(std::abs(
        regenerated.excess_path_ratio - compact.excess_path_ratio) <= 1.0e-5F);
    assert(std::abs(
        regenerated.directness_cost - compact.directness_cost) <= 1.0e-5F);
}

void test_direct_reach_ranks_before_hooked_fallback() {
    reach::Evaluation direct{};
    direct.rejection = reach::Rejection::None;
    direct.directness_cost = 0.05F;
    direct.active_arm_deformation = 1.0F;
    direct.candidate.clip = 1U;

    reach::Evaluation hooked{};
    hooked.rejection = reach::Rejection::None;
    hooked.directness_cost = 0.50F;
    hooked.active_arm_deformation = 0.0F;
    hooked.candidate.clip = 0U;

    assert(reach::detail::accepted_quality_less(direct, hooked));
    assert(!reach::detail::accepted_quality_less(hooked, direct));
    assert(direct.rejection == reach::Rejection::None);
    assert(hooked.rejection == reach::Rejection::None);
}

reach::CompactEvaluation accepted_with_path(
    size_t clip, const std::vector<vec3>& path) {
    reach::CompactEvaluation compact{};
    compact.evaluation.rejection = reach::Rejection::None;
    compact.evaluation.candidate.clip = clip;
    compact.hand_path = path;
    compact.straight_approach = reach::measure_straight_approach(
        path, vec3(0.0F, 0.0F, 0.0F), vec3(1.0F, 0.0F, 0.0F));
    return compact;
}

void test_preferred_preserves_identity_and_ranks_straight_first() {
    // Three accepted candidates: hooked, sliding, straight (raw order).
    const std::vector<vec3> hooked = {
        {-0.20F, 0.0F, 0.0F}, {-0.05F, 0.10F, 0.0F}, {0.0F, 0.0F, 0.0F}};
    const std::vector<vec3> slide = {
        {-0.20F, 0.06F, 0.0F}, {-0.10F, 0.04F, 0.0F}, {0.0F, 0.0F, 0.0F}};
    const std::vector<vec3> straight = {
        {-0.20F, 0.0F, 0.0F}, {-0.10F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}};

    std::vector<reach::CompactEvaluation> evaluations;
    evaluations.push_back(accepted_with_path(0U, hooked));
    evaluations.push_back(accepted_with_path(1U, slide));
    const size_t straight_index = evaluations.size();
    evaluations.push_back(accepted_with_path(2U, straight));

    const std::vector<size_t> accepted = {0U, 1U, 2U};
    const std::vector<size_t> preferred =
        reach::detail::build_preferred(evaluations, accepted);

    assert(accepted.size() == preferred.size());
    std::vector<size_t> raw_identities = accepted;
    std::vector<size_t> preferred_identities = preferred;
    std::sort(raw_identities.begin(), raw_identities.end());
    std::sort(preferred_identities.begin(), preferred_identities.end());
    assert(raw_identities == preferred_identities);
    assert(preferred.front() == straight_index);
}

void test_search_publishes_preferred_with_same_identities_as_accepted() {
    const reach::SearchResult result = run_fixture_search(2U);
    assert(result.complete);
    assert(result.accepted.size() == result.preferred.size());
    std::vector<size_t> raw = result.accepted;
    std::vector<size_t> pref = result.preferred;
    std::sort(raw.begin(), raw.end());
    std::sort(pref.begin(), pref.end());
    assert(raw == pref);
    // Raw acceptance order remains controlled by accepted_quality_less.
    for (size_t index = 1U; index < result.accepted.size(); ++index) {
        const reach::Evaluation& previous =
            result.evaluations[result.accepted[index - 1U]].evaluation;
        const reach::Evaluation& current =
            result.evaluations[result.accepted[index]].evaluation;
        assert(!reach::detail::accepted_quality_less(current, previous));
    }
    // Every accepted evaluation carries a finite straight-approach measurement.
    for (size_t index : result.accepted) {
        assert(result.evaluations[index].straight_approach.finite);
    }
}

}  // namespace

int main() {
    test_search_processes_every_bilateral_yaw_instance();
    test_search_order_is_independent_of_worker_count();
    test_cancelled_search_returns_without_processing_the_pack();
    test_zero_deadline_never_publishes_partial_results_as_complete();
    test_complete_search_never_exceeds_its_deadline();
    test_search_rejects_nonunit_approach();
    test_selected_regeneration_matches_compact_metrics();
    test_direct_reach_ranks_before_hooked_fallback();
    test_preferred_preserves_identity_and_ranks_straight_first();
    test_search_publishes_preferred_with_same_identities_as_accepted();
}
