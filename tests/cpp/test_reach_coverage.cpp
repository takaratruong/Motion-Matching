#include "g1_arm_joint_metadata.h"
#include "reach_coverage.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

reach::Pack fixture() {
    reach::Pack pack{};
    reach::Database& database = pack.database;
    database.version = 1U;
    database.endian_marker = 0x01020304U;
    database.fps_numerator = 25U;
    database.fps_denominator = 1U;
    database.frame_count = 6U;
    database.bone_count = 31U;
    database.clip_count = 1U;
    database.source_count = 1U;
    database.parents.assign(
        g1_skeleton::kParents.begin(), g1_skeleton::kParents.end());
    database.range_starts = {0};
    database.range_stops = {6};
    for (size_t frame = 0U; frame < 6U; ++frame) {
        append_pose(database, base_pose(0.02F * static_cast<float>(frame)));
        database.source_frames.push_back(static_cast<int32_t>(100 + frame));
    }
    database.active_hands = {0U};
    database.augmentations = {0U};
    database.source_indices = {0U};
    database.original_indices = {-1};
    database.source_names = {"pickup_north_0"};
    const interaction::WorldPose first = interaction::world_pose(
        reach::pose_at_frame(database, 0));
    const interaction::WorldPose last = interaction::world_pose(
        reach::pose_at_frame(database, 5));
    const size_t wrist = g1_skeleton::LeftWrist;
    const vec3 delta = last.positions[wrist] - first.positions[wrist];
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
    query.hand = reach::Hand::Right;
    assert(reach::select_candidates(pack, query).empty());
    query = zero_query(pack);
    query.target.position.x += 0.451F;
    assert(reach::select_candidates(pack, query).empty());
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

}  // namespace

int main() {
    test_retrieval_is_same_hand_spatial_and_wrist_orientation_independent();
    test_retrieval_ranks_position_then_approach_then_clip();
    test_zero_retarget_reproduces_endpoint_and_keeps_root_fixed();
    test_unreachable_target_reports_a_specific_final_gate();
    test_position_and_rotation_perturbations_have_exclusive_outcomes();
    test_collision_stages_and_active_contact_exemption();
    test_diagnostics_separate_hand_and_augmentation_counts();
}
