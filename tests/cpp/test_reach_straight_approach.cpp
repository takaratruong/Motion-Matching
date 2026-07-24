#include "reach_straight_approach.h"

#include "g1_arm_joint_metadata.h"
#include "g1_skeleton.h"
#include "interaction_pose.h"
#include "interaction_posture_ik.h"
#include "quat.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

void test_straight_path_outranks_slide_and_hook() {
    const vec3 contact(0.0F, 0.0F, 0.0F);
    const vec3 approach(1.0F, 0.0F, 0.0F);
    const std::vector<vec3> straight = {
        {-0.20F, 0.0F, 0.0F},
        {-0.10F, 0.0F, 0.0F},
        { 0.00F, 0.0F, 0.0F},
    };
    const std::vector<vec3> slide = {
        {-0.20F, 0.06F, 0.0F},
        {-0.10F, 0.04F, 0.0F},
        { 0.00F, 0.00F, 0.0F},
    };
    const auto direct = reach::measure_straight_approach(
        straight, contact, approach);
    const auto lateral = reach::measure_straight_approach(
        slide, contact, approach);
    require(direct.reaches_pregrasp_plane, "straight path missed pre-grasp");
    require(direct.maximum_lateral_m < 1.0e-6F,
        "straight path gained lateral drift");
    require(reach::straight_approach_quality_less(direct, lateral),
        "lateral slide outranked straight approach");
}

void test_backward_motion_is_measured() {
    const vec3 contact(0.0F, 0.0F, 0.0F);
    const vec3 approach(1.0F, 0.0F, 0.0F);
    // Advances, retreats, then advances to contact along the approach axis.
    const std::vector<vec3> retreat = {
        {-0.20F, 0.0F, 0.0F},
        {-0.05F, 0.0F, 0.0F},
        {-0.12F, 0.0F, 0.0F},
        { 0.00F, 0.0F, 0.0F},
    };
    const auto quality = reach::measure_straight_approach(
        retreat, contact, approach);
    require(quality.finite, "retreat path not finite");
    require(quality.backward_ratio > 0.0F,
        "backward motion was not measured");
}

void test_short_path_reports_no_pregrasp_coverage() {
    const vec3 contact(0.0F, 0.0F, 0.0F);
    const vec3 approach(1.0F, 0.0F, 0.0F);
    // Whole path stays within 10 cm of contact, never reaching the pre-grasp.
    const std::vector<vec3> shortp = {
        {-0.08F, 0.0F, 0.0F},
        {-0.04F, 0.0F, 0.0F},
        { 0.00F, 0.0F, 0.0F},
    };
    const auto quality = reach::measure_straight_approach(
        shortp, contact, approach);
    require(quality.finite, "short path not finite");
    require(!quality.reaches_pregrasp_plane,
        "short path incorrectly reached pre-grasp plane");
}

void test_rigid_transform_leaves_metrics_unchanged() {
    const vec3 contact(0.0F, 0.0F, 0.0F);
    const vec3 approach(1.0F, 0.0F, 0.0F);
    const std::vector<vec3> path = {
        {-0.20F, 0.02F, 0.0F},
        {-0.10F, 0.01F, 0.0F},
        { 0.00F, 0.00F, 0.0F},
    };
    const auto base = reach::measure_straight_approach(path, contact, approach);

    // Apply an arbitrary rigid transform (rotation + translation) to all inputs.
    const quat rotation = quat_normalize(
        quat_from_angle_axis(0.7F, normalize(vec3(0.3F, 1.0F, -0.5F))));
    const vec3 translation(1.5F, -2.0F, 3.25F);
    std::vector<vec3> moved(path.size());
    for (size_t i = 0U; i < path.size(); ++i) {
        moved[i] = quat_mul_vec3(rotation, path[i]) + translation;
    }
    const vec3 moved_contact = quat_mul_vec3(rotation, contact) + translation;
    const vec3 moved_approach = quat_mul_vec3(rotation, approach);
    const auto moved_quality = reach::measure_straight_approach(
        moved, moved_contact, moved_approach);

    require(moved_quality.reaches_pregrasp_plane == base.reaches_pregrasp_plane,
        "rigid transform changed pre-grasp coverage");
    require(std::fabs(moved_quality.maximum_lateral_m - base.maximum_lateral_m)
        < 1.0e-5F, "rigid transform changed max lateral");
    require(std::fabs(moved_quality.rms_lateral_m - base.rms_lateral_m)
        < 1.0e-5F, "rigid transform changed rms lateral");
    require(std::fabs(moved_quality.backward_ratio - base.backward_ratio)
        < 1.0e-5F, "rigid transform changed backward ratio");
    require(std::fabs(
        moved_quality.maximum_angle_radians - base.maximum_angle_radians)
        < 1.0e-5F, "rigid transform changed max angle");
    require(std::fabs(
        moved_quality.rms_angle_radians - base.rms_angle_radians)
        < 1.0e-5F, "rigid transform changed rms angle");
}

// ---- Task 3: lazy corridor retarget ---------------------------------------

quat joint_rotation(const interaction::HingeJoint& joint, float angle) {
    return quat_normalize(quat_mul(
        joint.rest_rotation, quat_from_angle_axis(angle, joint.axis)));
}

// A synthetic G1 pose with a swinging left arm; shoulder_delta drives the
// active wrist along an arc toward the grasp point.
interaction::Pose synthetic_pose(float shoulder_delta) {
    interaction::Pose pose{};
    for (quat& rotation : pose.rotations) rotation = quat(1, 0, 0, 0);
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
    pose.positions[g1_skeleton::LeftElbow] = vec3(0.015783F, -0.080518F, 0.0F);
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
    pose.positions[g1_skeleton::RightElbow] = vec3(0.015783F, -0.080518F, 0.0F);
    pose.positions[g1_skeleton::RightWristRoll] =
        vec3(0.10F, -0.01F, 0.00188791F);
    pose.positions[g1_skeleton::RightWristPitch] = vec3(0.038F, 0.0F, 0.0F);
    pose.positions[g1_skeleton::RightWrist] = vec3(0.046F, 0.0F, 0.0F);
    const std::array<float, 7> left = {
        0.10F + shoulder_delta, 0.35F, -0.20F, 0.80F, 0.10F, -0.15F, 0.05F};
    const std::array<float, 7> right = {
        0.10F, -0.35F, 0.20F, 0.80F, -0.10F, -0.15F, -0.05F};
    for (size_t joint = 0U; joint < 7U; ++joint) {
        pose.rotations[static_cast<size_t>(interaction::kLeftArm[joint].bone)] =
            joint_rotation(interaction::kLeftArm[joint], left[joint]);
        pose.rotations[static_cast<size_t>(interaction::kRightArm[joint].bone)] =
            joint_rotation(interaction::kRightArm[joint], right[joint]);
    }
    // A non-default root transform so root preservation is a real assertion.
    pose.positions[g1_skeleton::Simulation] = vec3(0.05F, 0.0F, -0.10F);
    pose.rotations[g1_skeleton::Simulation] =
        quat_from_angle_axis(0.15F, vec3(0.0F, 1.0F, 0.0F));
    return pose;
}

std::vector<interaction::Pose> synthetic_sequence(size_t count) {
    std::vector<interaction::Pose> source;
    source.reserve(count);
    for (size_t i = 0U; i < count; ++i) {
        const float delta =
            -1.2F + 1.2F * static_cast<float>(i) /
                static_cast<float>(count - 1U);
        source.push_back(synthetic_pose(delta));
    }
    return source;
}

vec3 active_wrist(const interaction::Pose& pose) {
    return interaction::world_pose(pose)
        .positions[g1_skeleton::LeftWrist];
}

quat active_wrist_rotation(const interaction::Pose& pose) {
    return interaction::world_pose(pose)
        .rotations[g1_skeleton::LeftWrist];
}

bool same_vec3_bits(vec3 left, vec3 right) {
    return std::memcmp(&left, &right, sizeof(vec3)) == 0;
}

bool same_quat_bits(quat left, quat right) {
    return std::memcmp(&left, &right, sizeof(quat)) == 0;
}

interaction::OrientedBox far_box() {
    return {{vec3(10.0F, 10.0F, 10.0F), quat()},
            vec3(0.01F, 0.01F, 0.01F)};
}

reach::CorridorRetargetResult run_retarget(
    const std::vector<interaction::Pose>& source,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment) {
    // Grasp exactly at the recorded final wrist pose, approaching along the
    // recorded net travel direction.
    const vec3 contact = active_wrist(source.back());
    const vec3 start = active_wrist(source.front());
    const vec3 approach = normalize(contact - start);
    const interaction::Transform hand_world{
        contact, active_wrist_rotation(source.back())};
    return reach::retarget_straight_approach(
        source, reach::Hand::Left, hand_world, approach,
        object, environment, 30.0F);
}

void test_retarget_preserves_root_and_holds_corridor() {
    const std::vector<interaction::Pose> source = synthetic_sequence(16U);
    const reach::CorridorRetargetResult result =
        run_retarget(source, far_box(), interaction::EnvironmentGeometry{});
    if (!result.accepted) {
        std::fprintf(
            stderr, "retarget diagnostic: failure=%u sample=%zu blend=%zu corridor=%zu\n",
            static_cast<unsigned>(result.failure),
            result.failure_sample,
            result.blend_start,
            result.corridor_start);
    }
    require(result.accepted, "corridor retarget failed on reachable path");
    require(result.poses.size() == source.size(),
        "corridor retarget changed sample count");

    // Root position and rotation are bit-identical at every sample.
    for (size_t i = 0U; i < source.size(); ++i) {
        require(
            same_vec3_bits(
                result.poses[i].positions[g1_skeleton::Simulation],
                source[i].positions[g1_skeleton::Simulation]) &&
            same_quat_bits(
                result.poses[i].rotations[g1_skeleton::Simulation],
                source[i].rotations[g1_skeleton::Simulation]),
            "corridor retarget moved the root");
    }

    // Samples before the blend are bit-identical everywhere.
    for (size_t i = 0U; i < result.blend_start; ++i) {
        for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
            require(
                same_vec3_bits(
                    result.poses[i].positions[bone],
                    source[i].positions[bone]) &&
                same_quat_bits(
                    result.poses[i].rotations[bone],
                    source[i].rotations[bone]),
                "corridor retarget disturbed a pre-blend sample");
        }
    }

    // The wrist stays within the corridor and never steps backward past 2 mm.
    const vec3 contact = active_wrist(source.back());
    const vec3 approach = normalize(contact - active_wrist(source.front()));
    float previous_remaining = 1.0e9F;
    for (size_t i = result.corridor_start; i < result.poses.size(); ++i) {
        const vec3 wrist = active_wrist(result.poses[i]);
        const vec3 from_contact = contact - wrist;
        const float remaining = dot(from_contact, approach);
        const vec3 lateral = from_contact - remaining * approach;
        require(length(lateral) <= 0.03F + 1.0e-4F,
            "retargeted wrist left the corridor");
        if (i > result.corridor_start) {
            const float step = previous_remaining - remaining;
            require(step >= -0.002F - 1.0e-5F,
                "retargeted wrist stepped backward past 2 mm");
        }
        previous_remaining = remaining;
    }

    // Final wrist position lands exactly on the requested grasp.
    const vec3 final_wrist = active_wrist(result.poses.back());
    require(length(final_wrist - contact) <= 1.0e-3F,
        "retargeted final wrist missed the grasp point");
}

void test_retarget_holds_gripper_open_until_close_distance() {
    const std::vector<interaction::Pose> source = synthetic_sequence(16U);
    reach::StraightApproachConfig config{};
    config.open_active_hand_dof = {0.9F, 0.9F, 0.9F, 0.9F, 0.9F, 0.9F, 0.9F};
    // Give every source frame a different hand pose. The closure target must
    // remain the single recorded contact pose, not chase each source frame.
    std::vector<interaction::Pose> seeded = source;
    for (size_t sample = 0U; sample < seeded.size(); ++sample) {
        for (size_t dof = 0U; dof < 7U; ++dof) {
            seeded[sample].hand_dof[dof] =
                0.4F + 0.01F * static_cast<float>(sample);
        }
    }
    for (size_t dof = 0U; dof < 7U; ++dof) {
        seeded.back().hand_dof[dof] = 0.1F;
    }
    const vec3 contact = active_wrist(seeded.back());
    const vec3 approach = normalize(contact - active_wrist(seeded.front()));
    const interaction::Transform hand_world{
        contact, active_wrist_rotation(seeded.back())};
    const reach::CorridorRetargetResult result =
        reach::retarget_straight_approach(
            seeded, reach::Hand::Left, hand_world, approach,
            far_box(), interaction::EnvironmentGeometry{}, 30.0F, config);
    require(result.accepted, "seeded corridor retarget failed");

    for (size_t i = result.corridor_start; i < result.poses.size(); ++i) {
        const vec3 wrist = active_wrist(result.poses[i]);
        const float remaining = dot(contact - wrist, approach);
        if (remaining > config.close_distance_m + 1.0e-4F) {
            for (size_t dof = 0U; dof < 7U; ++dof) {
                require(std::fabs(
                    result.poses[i].hand_dof[dof] -
                    config.open_active_hand_dof[dof]) < 1.0e-4F,
                    "gripper closed before the final close distance");
            }
        }
    }
    // At contact the active hand reaches the recorded closed DOFs.
    for (size_t dof = 0U; dof < 7U; ++dof) {
        require(std::fabs(result.poses.back().hand_dof[dof] - 0.1F) < 1.0e-3F,
            "gripper did not close to the recorded contact DOFs");
    }
    const size_t closing_sample = result.poses.size() - 2U;
    for (size_t dof = 0U; dof < 7U; ++dof) {
        require(result.poses[closing_sample].hand_dof[dof] < 0.9F &&
                result.poses[closing_sample].hand_dof[dof] >= 0.1F,
            "gripper closure did not converge on the fixed contact target");
    }
}

void test_explicit_closed_hand_target_overrides_recorded_contact() {
    std::vector<interaction::Pose> source = synthetic_sequence(16U);
    for (interaction::Pose& pose : source) {
        for (size_t dof = 0U; dof < 7U; ++dof) {
            pose.hand_dof[dof] = 0.1F;
        }
    }
    const vec3 contact = active_wrist(source.back());
    const vec3 approach = normalize(contact - active_wrist(source.front()));
    const interaction::Transform hand_world{
        contact, active_wrist_rotation(source.back())};
    reach::StraightApproachConfig config{};
    config.use_explicit_closed_hand_dof = true;
    config.closed_active_hand_dof =
        {0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F, 0.8F};
    const reach::CorridorRetargetResult result =
        reach::retarget_straight_approach(
            source, reach::Hand::Left, hand_world, approach,
            far_box(), interaction::EnvironmentGeometry{}, 30.0F, config);
    require(result.accepted, "explicit closed-hand retarget failed");
    for (size_t dof = 0U; dof < 7U; ++dof) {
        require(std::fabs(
            result.poses.back().hand_dof[dof] -
            config.closed_active_hand_dof[dof]) < 1.0e-3F,
            "explicit closed-hand target was not reached");
    }
}

void test_nonfinite_hand_schedule_fails_closed() {
    const std::vector<interaction::Pose> source = synthetic_sequence(16U);
    const vec3 contact = active_wrist(source.back());
    const vec3 approach = normalize(contact - active_wrist(source.front()));
    const interaction::Transform hand_world{
        contact, active_wrist_rotation(source.back())};
    reach::StraightApproachConfig config{};
    config.open_active_hand_dof[3U] =
        std::numeric_limits<float>::quiet_NaN();
    const reach::CorridorRetargetResult result =
        reach::retarget_straight_approach(
            source, reach::Hand::Left, hand_world, approach,
            far_box(), interaction::EnvironmentGeometry{}, 30.0F, config);
    require(!result.accepted, "non-finite hand schedule was accepted");
    require(result.failure == reach::CorridorRetargetFailure::InvalidInput,
        "non-finite hand schedule did not fail as invalid input");
}

void test_open_gripper_proxy_rejects_early_terminal_sweep_contact() {
    const std::vector<interaction::Pose> source = synthetic_sequence(16U);
    const reach::CorridorRetargetResult clear =
        run_retarget(source, far_box(), interaction::EnvironmentGeometry{});
    require(clear.accepted, "clear proxy fixture did not retarget");
    const vec3 previous = active_wrist(clear.poses[clear.poses.size() - 2U]);
    const vec3 current = active_wrist(clear.poses.back());
    const vec3 approach = normalize(current - active_wrist(source.front()));
    vec3 finger_axis = quat_mul_vec3(
        active_wrist_rotation(clear.poses.back()), vec3(0, 0, 1));
    finger_axis = finger_axis - dot(finger_axis, approach) * approach;
    finger_axis = normalize(finger_axis);
    const interaction::OrientedBox near_open_fingers{
        {
            lerp(previous, current, 0.5F) +
                0.03F * approach + 0.065F * finger_axis,
            quat()},
        vec3(0.005F, 0.005F, 0.005F)};
    const reach::CorridorRetargetResult blocked =
        run_retarget(
            source, near_open_fingers, interaction::EnvironmentGeometry{});
    require(!blocked.accepted,
        "early terminal sweep contact escaped the open-gripper proxy");
    require(blocked.failure == reach::CorridorRetargetFailure::ObjectCollision,
        "terminal sweep contact did not report object collision");
}

void test_open_fingers_can_surround_the_intended_grasp_object() {
    const std::vector<interaction::Pose> source = synthetic_sequence(16U);
    const vec3 contact = active_wrist(source.back());
    const vec3 approach = normalize(contact - active_wrist(source.front()));
    const interaction::OrientedBox grasped_object{
        {contact + 0.06F * approach, quat()},
        vec3(0.04F, 0.002F, 0.002F)};
    const reach::CorridorRetargetResult result =
        run_retarget(
            source, grasped_object, interaction::EnvironmentGeometry{});
    if (!result.accepted) {
        std::fprintf(
            stderr,
            "grasp-object diagnostic: failure=%u sample=%zu\n",
            static_cast<unsigned>(result.failure),
            result.failure_sample);
    }
    require(result.accepted,
        "open finger spacing was treated as penetration of the grasp object");
}

void test_short_path_reports_no_pregrasp_coverage_on_retarget() {
    // Only the last few samples of the swing: never reaches 20 cm out.
    std::vector<interaction::Pose> full = synthetic_sequence(16U);
    std::vector<interaction::Pose> shortp(full.end() - 4, full.end());
    const reach::CorridorRetargetResult result =
        run_retarget(shortp, far_box(), interaction::EnvironmentGeometry{});
    require(!result.accepted, "short path was accepted");
    require(result.failure == reach::CorridorRetargetFailure::NoPregraspCoverage,
        "short path did not report NoPregraspCoverage");
}

void test_environment_collision_fails_closed() {
    const std::vector<interaction::Pose> source = synthetic_sequence(16U);
    // A large box straddling the corridor midpoint forces a body collision.
    const vec3 contact = active_wrist(source.back());
    const vec3 start = active_wrist(source.front());
    const vec3 mid = lerp(start, contact, 0.5F);
    interaction::EnvironmentGeometry environment{};
    environment.boxes.push_back({{mid, quat()}, vec3(0.4F, 0.4F, 0.4F)});
    const reach::CorridorRetargetResult result =
        run_retarget(source, far_box(), environment);
    require(!result.accepted, "colliding environment was accepted");
    require(
        result.failure == reach::CorridorRetargetFailure::EnvironmentCollision ||
        result.failure == reach::CorridorRetargetFailure::ObjectCollision,
        "collision did not fail closed");
}

void test_unaccepted_ik_fails_closed() {
    const std::vector<interaction::Pose> source = synthetic_sequence(16U);
    const vec3 contact = active_wrist(source.back());
    const vec3 approach = normalize(contact - active_wrist(source.front()));
    const interaction::Transform hand_world{
        contact, active_wrist_rotation(source.back())};
    interaction::PostureIKConfig impossible{};
    impossible.accepted_position_m = 0.0F;
    impossible.accepted_orientation_radians = 0.0F;
    const reach::CorridorRetargetResult result =
        reach::retarget_straight_approach(
            source, reach::Hand::Left, hand_world, approach,
            far_box(), interaction::EnvironmentGeometry{}, 30.0F,
            reach::StraightApproachConfig{}, impossible);
    require(!result.accepted, "unaccepted IK solve was accepted");
    require(result.failure == reach::CorridorRetargetFailure::InvalidSolver,
        "unaccepted IK solve did not fail as InvalidSolver");
}

void test_final_frame_targets_requested_contact() {
    const std::vector<interaction::Pose> source = synthetic_sequence(16U);
    const vec3 recorded_contact = active_wrist(source.back());
    const vec3 approach =
        normalize(recorded_contact - active_wrist(source.front()));
    const vec3 requested_contact = recorded_contact + 0.03F * approach;
    const interaction::Transform hand_world{
        requested_contact, active_wrist_rotation(source.back())};
    const reach::CorridorRetargetResult result =
        reach::retarget_straight_approach(
            source, reach::Hand::Left, hand_world, approach,
            far_box(), interaction::EnvironmentGeometry{}, 30.0F);
    require(result.accepted, "offset contact retarget failed");
    require(
        length(active_wrist(result.poses.back()) - requested_contact) <= 1.0e-3F,
        "final frame did not target the requested contact");
}

}  // namespace

int main() {
    test_straight_path_outranks_slide_and_hook();
    test_backward_motion_is_measured();
    test_short_path_reports_no_pregrasp_coverage();
    test_rigid_transform_leaves_metrics_unchanged();
    test_retarget_preserves_root_and_holds_corridor();
    test_retarget_holds_gripper_open_until_close_distance();
    test_explicit_closed_hand_target_overrides_recorded_contact();
    test_nonfinite_hand_schedule_fails_closed();
    test_open_gripper_proxy_rejects_early_terminal_sweep_contact();
    test_open_fingers_can_surround_the_intended_grasp_object();
    test_short_path_reports_no_pregrasp_coverage_on_retarget();
    test_environment_collision_fails_closed();
    test_unaccepted_ik_fails_closed();
    test_final_frame_targets_requested_contact();
    std::printf("straight approach metrics PASS\n");
    return 0;
}
