#include "g1_arm_joint_metadata.h"
#include "reach_coverage.h"
#include "reach_placement.h"

#include <array>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

quat joint_rotation(const interaction::HingeJoint& joint, float angle) {
    return quat_normalize(quat_mul(
        joint.rest_rotation, quat_from_angle_axis(angle, joint.axis)));
}

interaction::Pose base_pose(float shoulder_delta) {
    interaction::Pose pose{};
    for (quat& rotation : pose.rotations) {
        rotation = quat(1, 0, 0, 0);
    }
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
    const std::array<float, 7> left = {
        0.10F + shoulder_delta, 0.35F, -0.20F, 0.80F,
        0.10F, -0.15F, 0.05F};
    const std::array<float, 7> right = {
        0.10F, -0.35F, 0.20F, 0.80F, -0.10F, -0.15F, -0.05F};
    for (size_t joint = 0; joint < 7U; ++joint) {
        pose.rotations[static_cast<size_t>(interaction::kLeftArm[joint].bone)] =
            joint_rotation(interaction::kLeftArm[joint], left[joint]);
        pose.rotations[static_cast<size_t>(interaction::kRightArm[joint].bone)] =
            joint_rotation(interaction::kRightArm[joint], right[joint]);
    }
    return pose;
}

void append_pose(reach::Database& database, const interaction::Pose& pose) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
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

reach::Pack fixture(
    size_t frame_count = 6U,
    size_t final_moving_frame = std::numeric_limits<size_t>::max()) {
    reach::Pack pack{};
    reach::Database& database = pack.database;
    database.version = 1U;
    database.endian_marker = 0x01020304U;
    database.fps_numerator = 25U;
    database.fps_denominator = 1U;
    database.frame_count = static_cast<uint32_t>(frame_count);
    database.bone_count = 31U;
    database.clip_count = 1U;
    database.source_count = 1U;
    database.parents.assign(
        g1_skeleton::kParents.begin(), g1_skeleton::kParents.end());
    database.range_starts = {0};
    database.range_stops = {static_cast<int32_t>(frame_count)};
    for (size_t frame = 0U; frame < frame_count; ++frame) {
        append_pose(database, base_pose(
            0.02F * static_cast<float>(std::min(frame, final_moving_frame))));
        database.source_frames.push_back(static_cast<int32_t>(100 + frame));
    }
    database.active_hands = {0U};
    database.augmentations = {0U};
    database.source_indices = {0U};
    database.original_indices = {-1};
    database.source_names = {"pickup_north_0"};
    const interaction::WorldPose last = interaction::world_pose(
        reach::pose_at_frame(database, static_cast<int32_t>(frame_count - 1U)));
    const size_t wrist = g1_skeleton::LeftWrist;
    const size_t final_sample = frame_count - 1U;
    size_t approach_sample = final_sample > 5U ? final_sample - 5U : 0U;
    vec3 delta = last.positions[wrist] - interaction::world_pose(
        reach::pose_at_frame(database, static_cast<int32_t>(approach_sample)))
                                           .positions[wrist];
    const size_t earliest_sample = final_sample > 25U
        ? final_sample - 25U
        : 0U;
    while (length(delta) < 0.01F && approach_sample > earliest_sample) {
        --approach_sample;
        delta = last.positions[wrist] - interaction::world_pose(
            reach::pose_at_frame(
                database, static_cast<int32_t>(approach_sample)))
                                                .positions[wrist];
    }
    const vec3 approach = normalize(delta);
    const interaction::Transform endpoint{
        last.positions[wrist], last.rotations[wrist]};
    database.endpoint_positions = {
        endpoint.position.x, endpoint.position.y, endpoint.position.z};
    database.endpoint_rotations = {
        endpoint.rotation.w, endpoint.rotation.x,
        endpoint.rotation.y, endpoint.rotation.z};
    database.approach_directions = {approach.x, approach.y, approach.z};
    pack.features.version = 1U;
    pack.features.endian_marker = 0x01020304U;
    pack.features.clip_count = 1U;
    pack.features.dimension = 10U;
    pack.features.values = {
        endpoint.position.x, endpoint.position.y, endpoint.position.z,
        approach.x, approach.y, approach.z,
        endpoint.rotation.w, endpoint.rotation.x,
        endpoint.rotation.y, endpoint.rotation.z};
    return pack;
}

reach::Query zero_query(const reach::Pack& pack) {
    reach::Query query{};
    query.hand = reach::Hand::Left;
    query.target = reach::endpoint_transform(pack.database, 0U);
    query.approach_world = reach::approach_direction(pack.database, 0U);
    return query;
}

std::vector<size_t> clip_set(const std::vector<reach::Candidate>& candidates) {
    std::vector<size_t> clips;
    for (const reach::Candidate& candidate : candidates) {
        clips.push_back(candidate.clip);
    }
    std::sort(clips.begin(), clips.end());
    return clips;
}

void test_retrieval_is_same_hand_spatial_and_wrist_orientation_independent() {
    const reach::Pack pack = fixture();
    reach::Query query = zero_query(pack);
    auto candidates = reach::select_candidates(pack, query);
    assert(candidates.size() == 1U && candidates[0].clip == 0U);

    query.target.rotation = quat_mul(
        quat_from_angle_axis(3.14159265F, vec3(0, 1, 0)),
        query.target.rotation);
    assert(reach::select_candidates(pack, query).size() == 1U);

    query = zero_query(pack);
    const std::vector<size_t> baseline = clip_set(
        reach::select_candidates(pack, query));
    query.approach_world = normalize(quat_mul_vec3(
        quat_from_angle_axis(1.570796327F, vec3(0, 1, 0)),
        query.approach_world));
    assert(clip_set(reach::select_candidates(pack, query)) == baseline);

    query = zero_query(pack);
    query.hand = reach::Hand::Right;
    assert(reach::select_candidates(pack, query).empty());
    query = zero_query(pack);
    query.target.position.x += 0.451F;
    assert(reach::select_candidates(pack, query).empty());
}

void test_one_reach_warps_to_a_different_approach_direction() {
    const reach::Pack pack = fixture(20U);
    reach::Query query = zero_query(pack);
    query.approach_world = normalize(quat_mul_vec3(
        quat_from_angle_axis(0.523598776F, vec3(0, 1, 0)),
        query.approach_world));
    const reach::Candidate candidate = reach::select_candidates(
        pack, query)[0];

    const reach::Evaluation result = reach::shape_candidate(
        pack, candidate, query);

    assert(result.rejection == reach::Rejection::None);
    assert(result.approach_error_radians <= 0.261799388F);
    for (size_t frame = 0U; frame < result.poses.size(); ++frame) {
        const interaction::Pose source = reach::pose_at_frame(
            pack.database, static_cast<int32_t>(frame));
        const size_t root = static_cast<size_t>(g1_skeleton::Simulation);
        assert(result.poses[frame].positions[root].x ==
               source.positions[root].x);
        assert(result.poses[frame].positions[root].y ==
               source.positions[root].y);
        assert(result.poses[frame].positions[root].z ==
               source.positions[root].z);
        assert(result.poses[frame].rotations[root].w ==
               source.rotations[root].w);
        assert(result.poses[frame].rotations[root].x ==
               source.rotations[root].x);
        assert(result.poses[frame].rotations[root].y ==
               source.rotations[root].y);
        assert(result.poses[frame].rotations[root].z ==
               source.rotations[root].z);
        for (const interaction::HingeJoint& joint : interaction::kRightArm) {
            const size_t bone = static_cast<size_t>(joint.bone);
            assert(result.poses[frame].positions[bone].x ==
                   source.positions[bone].x);
            assert(result.poses[frame].positions[bone].y ==
                   source.positions[bone].y);
            assert(result.poses[frame].positions[bone].z ==
                   source.positions[bone].z);
            assert(result.poses[frame].rotations[bone].w ==
                   source.rotations[bone].w);
            assert(result.poses[frame].rotations[bone].x ==
                   source.rotations[bone].x);
            assert(result.poses[frame].rotations[bone].y ==
                   source.rotations[bone].y);
            assert(result.poses[frame].rotations[bone].z ==
                   source.rotations[bone].z);
        }
    }
}

void test_short_reach_keeps_its_initial_pose_unwarped() {
    const reach::Pack pack = fixture();
    reach::Query query = zero_query(pack);
    query.approach_world = normalize(quat_mul_vec3(
        quat_from_angle_axis(0.523598776F, vec3(0, 1, 0)),
        query.approach_world));

    const reach::Evaluation result = reach::shape_candidate(
        pack, reach::select_candidates(pack, query)[0], query);
    const interaction::Pose source = reach::pose_at_frame(pack.database, 0);

    for (const interaction::HingeJoint& joint : interaction::kLeftArm) {
        const size_t bone = static_cast<size_t>(joint.bone);
        assert(result.poses[0].rotations[bone].w == source.rotations[bone].w);
        assert(result.poses[0].rotations[bone].x == source.rotations[bone].x);
        assert(result.poses[0].rotations[bone].y == source.rotations[bone].y);
        assert(result.poses[0].rotations[bone].z == source.rotations[bone].z);
    }
}

void test_small_translation_without_approach_warp_uses_public_tolerance() {
    const reach::Pack pack = fixture();
    reach::Query query = zero_query(pack);
    query.target.position.y += 0.01F;

    const reach::Evaluation result = reach::shape_candidate(
        pack, reach::select_candidates(pack, query)[0], query);
    const interaction::Pose source = reach::pose_at_frame(pack.database, 5);

    assert(result.rejection == reach::Rejection::None);
    for (const interaction::HingeJoint& joint : interaction::kLeftArm) {
        const size_t bone = static_cast<size_t>(joint.bone);
        assert(result.poses[5].rotations[bone].w == source.rotations[bone].w);
        assert(result.poses[5].rotations[bone].x == source.rotations[bone].x);
        assert(result.poses[5].rotations[bone].y == source.rotations[bone].y);
        assert(result.poses[5].rotations[bone].z == source.rotations[bone].z);
    }
}

void test_retrieval_ranks_position_then_approach_then_clip() {
    reach::Pack pack = fixture();
    const interaction::Transform endpoint =
        reach::endpoint_transform(pack.database, 0U);
    const vec3 approach = reach::approach_direction(pack.database, 0U);
    pack.database.clip_count = 3U;
    pack.database.active_hands = {0U, 0U, 0U};
    pack.database.endpoint_positions = {
        endpoint.position.x, endpoint.position.y, endpoint.position.z,
        endpoint.position.x + 0.10F, endpoint.position.y, endpoint.position.z,
        endpoint.position.x + 0.10F, endpoint.position.y, endpoint.position.z,
    };
    pack.database.endpoint_rotations.resize(12U, 0.0F);
    for (size_t clip = 0U; clip < 3U; ++clip) {
        pack.database.endpoint_rotations[clip * 4U] = 1.0F;
    }
    pack.database.approach_directions = {
        approach.x, approach.y, approach.z,
        approach.x, approach.y, approach.z,
        -approach.z, approach.y, approach.x,
    };
    reach::Query query = zero_query(pack);
    query.target.position.x += 0.05F;
    const auto candidates = reach::select_candidates(pack, query);
    assert(candidates.size() == 3U);
    assert(candidates[0].clip == 0U);
    assert(candidates[1].clip == 1U);
    assert(candidates[2].clip == 2U);
}

void test_zero_retarget_reproduces_endpoint_and_keeps_root_fixed() {
    const reach::Pack pack = fixture();
    const reach::Query query = zero_query(pack);
    const reach::Candidate candidate = reach::select_candidates(pack, query)[0];

    const reach::Evaluation result = reach::shape_candidate(
        pack, candidate, query);

    assert(result.rejection == reach::Rejection::None);
    assert(result.position_error_m <= 0.001F);
    assert(result.approach_error_radians <= 0.008726646F);
    assert(result.poses.size() == 6U);
    for (size_t frame = 0U; frame < result.poses.size(); ++frame) {
        const interaction::Pose source = reach::pose_at_frame(
            pack.database, static_cast<int32_t>(frame));
        assert(result.poses[frame].positions[g1_skeleton::Simulation].x ==
               source.positions[g1_skeleton::Simulation].x);
        assert(result.poses[frame].rotations[g1_skeleton::Simulation].w ==
               source.rotations[g1_skeleton::Simulation].w);
    }
}

void test_zero_retarget_uses_pre_pause_approach_evidence() {
    const reach::Pack pack = fixture(20U, 10U);
    const reach::Query query = zero_query(pack);

    const reach::Evaluation result = reach::shape_candidate(
        pack, reach::Candidate{0U}, query);

    assert(result.rejection == reach::Rejection::None);
    assert(result.approach_error_radians <= 0.008726646F);
}

void test_unreachable_target_reports_a_specific_final_gate() {
    const reach::Pack pack = fixture();
    reach::Query query = zero_query(pack);
    query.target.position.x += 0.40F;
    const auto candidates = reach::select_candidates(pack, query);
    assert(candidates.size() == 1U);
    const reach::Evaluation result = reach::shape_candidate(
        pack, candidates[0], query);
    assert(result.rejection == reach::Rejection::PositionError ||
           result.rejection == reach::Rejection::ApproachAxisError ||
           result.rejection == reach::Rejection::FullOrientationError);
}

void test_position_and_rotation_perturbations_have_exclusive_outcomes() {
    const reach::Pack pack = fixture();
    const std::array<float, 5> offsets = {0.05F, 0.10F, 0.20F, 0.30F, 0.45F};
    for (const float offset : offsets) {
        reach::Query query = zero_query(pack);
        query.target.position.z += offset;
        const auto candidates = reach::select_candidates(pack, query);
        assert(candidates.size() == 1U);
        const reach::Evaluation result = reach::shape_candidate(
            pack, candidates[0], query);
        assert(result.rejection != reach::Rejection::OutsideEnvelope);
        assert(result.rejection != reach::Rejection::InvalidSolver);
    }
    const std::array<float, 5> angles = {
        0.261799388F, 0.523598776F, 1.047197551F,
        1.570796327F, 3.141592654F};
    const std::array<vec3, 3> axes = {
        vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1)};
    for (const vec3 axis : axes) {
        for (const float angle : angles) {
            for (const float sign : {-1.0F, 1.0F}) {
                reach::Query query = zero_query(pack);
                query.target.rotation = quat_mul(
                    query.target.rotation,
                    quat_from_angle_axis(sign * angle, axis));
                const reach::Candidate candidate =
                    reach::select_candidates(pack, query)[0];
                const reach::Evaluation result = reach::shape_candidate(
                    pack, candidate, query);
                assert(result.rejection != reach::Rejection::OutsideEnvelope);
                assert(result.rejection != reach::Rejection::InvalidSolver);
            }
        }
    }
}

void test_collision_stages_and_active_contact_exemption() {
    const reach::Pack pack = fixture();
    const reach::Query query = zero_query(pack);
    const reach::Candidate candidate = reach::select_candidates(pack, query)[0];
    const interaction::OrientedBox far_object{
        {vec3(10, 10, 10), quat()}, vec3(0.01F, 0.01F, 0.01F)};
    const interaction::EnvironmentGeometry open{};
    assert(reach::evaluate_candidate(
        pack, candidate, query, far_object, open).rejection ==
        reach::Rejection::None);

    const interaction::OrientedBox body_object{
        {vec3(0.0F, 0.82F, 0.0F), quat()}, vec3(0.05F, 0.05F, 0.05F)};
    assert(reach::evaluate_candidate(
        pack, candidate, query, body_object, open).rejection ==
        reach::Rejection::ObjectCollision);

    interaction::EnvironmentGeometry blocked{};
    blocked.boxes.push_back(body_object);
    assert(reach::evaluate_candidate(
        pack, candidate, query, far_object, blocked).rejection ==
        reach::Rejection::EnvironmentCollision);

    interaction::TrajectoryCollisionConfig points_only{};
    points_only.wrist_radius_m = 0.0F;
    points_only.forearm_radius_m = 0.0F;
    points_only.joint_radius_m = 0.0F;
    points_only.limb_radius_m = 0.0F;
    points_only.torso_radius_m = 0.0F;
    const interaction::OrientedBox grasp_object{
        {query.target.position, quat()}, vec3(1.0e-5F, 1.0e-5F, 1.0e-5F)};
    assert(reach::evaluate_candidate(
        pack, candidate, query, grasp_object, open,
        reach::CoverageConfig{}, points_only).rejection ==
        reach::Rejection::None);
}

void test_collision_observations_survive_an_earlier_kinematic_rejection() {
    const reach::Pack pack = fixture();
    reach::Query query = zero_query(pack);
    query.approach_world = normalize(quat_mul_vec3(
        quat_from_angle_axis(0.523598776F, vec3(0, 1, 0)),
        query.approach_world));
    const reach::Candidate candidate = reach::select_candidates(
        pack, query)[0];
    reach::CoverageConfig strict{};
    strict.accepted_approach_radians = 0.01F;
    const interaction::OrientedBox far_object{
        {vec3(10, 10, 10), quat()}, vec3(0.01F, 0.01F, 0.01F)};
    interaction::EnvironmentGeometry blocked{};
    blocked.boxes.push_back({
        {vec3(0.0F, 0.82F, 0.0F), quat()},
        vec3(0.05F, 0.05F, 0.05F)});

    const reach::Evaluation collision = reach::evaluate_candidate(
        pack, candidate, query, far_object, blocked, strict);

    assert(collision.rejection == reach::Rejection::ApproachAxisError);
    assert(!collision.object_collision_observed);
    assert(collision.environment_collision_observed);

    const interaction::OrientedBox body_object{
        {vec3(0.0F, 0.82F, 0.0F), quat()},
        vec3(0.05F, 0.05F, 0.05F)};
    const reach::Evaluation both = reach::evaluate_candidate(
        pack, candidate, query, body_object, blocked, strict);
    assert(both.rejection == reach::Rejection::ApproachAxisError);
    assert(both.object_collision_observed);
    assert(both.environment_collision_observed);

    const reach::Evaluation open = reach::evaluate_candidate(
        pack, candidate, query, far_object,
        interaction::EnvironmentGeometry{}, strict);
    assert(open.rejection == reach::Rejection::ApproachAxisError);
    assert(!open.environment_collision_observed);
}

void test_diagnostics_separate_hand_and_augmentation_counts() {
    reach::Pack pack = fixture();
    pack.database.clip_count = 2U;
    pack.database.active_hands = {0U, 1U};
    pack.database.augmentations = {0U, 1U};
    reach::Evaluation captured{};
    captured.candidate.clip = 0U;
    reach::Evaluation mirrored{};
    mirrored.candidate.clip = 1U;
    mirrored.rejection = reach::Rejection::PositionError;
    mirrored.joint_limit_saturated = true;
    const reach::Diagnostics diagnostics = reach::summarize_evaluations(
        pack, std::vector<reach::Evaluation>{captured, mirrored});
    assert(diagnostics.evaluations == 2U);
    assert(diagnostics.accepted == 1U);
    assert(diagnostics.hand_evaluations[0] == 1U);
    assert(diagnostics.hand_evaluations[1] == 1U);
    assert(diagnostics.augmentation_evaluations[0] == 1U);
    assert(diagnostics.augmentation_evaluations[1] == 1U);
    assert(diagnostics.rejection_counts[0] == 1U);
    assert(diagnostics.rejection_counts[3] == 1U);
    assert(diagnostics.joint_limit_saturated == 1U);
}

void test_contact_placement_is_exact_and_upright() {
    const reach::Pack pack = fixture(20U);
    const vec3 target(1.25F, 0.83F, -0.70F);
    for (uint8_t yaw = 0U; yaw < reach::kYawPlacementCount; ++yaw) {
        const interaction::Pose placed = reach::place_pose(
            pack, 0U, yaw, target, 19);
        const interaction::WorldPose world = interaction::world_pose(placed);
        const size_t wrist = static_cast<size_t>(g1_skeleton::LeftWrist);
        assert(length(world.positions[wrist] - target) <= 1.0e-5F);
        const vec3 up = quat_mul_vec3(
            world.rotations[g1_skeleton::Simulation], vec3(0, 1, 0));
        assert(length(up - vec3(0, 1, 0)) <= 1.0e-5F);
    }
}

void test_contact_placement_translates_every_sample_rigidly() {
    const reach::Pack pack = fixture(20U);
    const vec3 first_target(0.4F, 0.9F, -0.2F);
    const vec3 moved_target = first_target + vec3(1.0F, -0.3F, 0.5F);
    for (int32_t frame = 0; frame < 20; ++frame) {
        const interaction::WorldPose first = interaction::world_pose(
            reach::place_pose(pack, 0U, 3U, first_target, frame));
        const interaction::WorldPose moved = interaction::world_pose(
            reach::place_pose(pack, 0U, 3U, moved_target, frame));
        for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
            assert(length(
                (moved.positions[bone] - first.positions[bone]) -
                (moved_target - first_target)) <= 1.0e-5F);
        }
    }
}

}  // namespace

int main() {
    test_retrieval_is_same_hand_spatial_and_wrist_orientation_independent();
    test_one_reach_warps_to_a_different_approach_direction();
    test_short_reach_keeps_its_initial_pose_unwarped();
    test_small_translation_without_approach_warp_uses_public_tolerance();
    test_retrieval_ranks_position_then_approach_then_clip();
    test_zero_retarget_reproduces_endpoint_and_keeps_root_fixed();
    test_zero_retarget_uses_pre_pause_approach_evidence();
    test_unreachable_target_reports_a_specific_final_gate();
    test_position_and_rotation_perturbations_have_exclusive_outcomes();
    test_collision_stages_and_active_contact_exemption();
    test_collision_observations_survive_an_earlier_kinematic_rejection();
    test_diagnostics_separate_hand_and_augmentation_counts();
    test_contact_placement_is_exact_and_upright();
    test_contact_placement_translates_every_sample_rigidly();
}
