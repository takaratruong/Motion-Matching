#include "g1_posture_ik_fixture.h"
#include "reach_placement.h"
#include "reach_return.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr size_t kContactFrame = 1U;
constexpr size_t kReturnStop = 5U;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool near(vec3 left, vec3 right, float tolerance = 2.0e-3F) {
    return length(left - right) <= tolerance;
}

bool near(quat left, quat right, float tolerance = 2.0e-3F) {
    const quat delta = quat_abs(quat_mul(
        quat_normalize(left), quat_inv(quat_normalize(right))));
    return length(quat_to_scaled_angle_axis(delta)) <= tolerance;
}

float rotation_error(quat left, quat right) {
    const quat delta = quat_abs(quat_mul(
        quat_normalize(left), quat_inv(quat_normalize(right))));
    return length(quat_to_scaled_angle_axis(delta));
}

float wrap_angle(float value) {
    constexpr float kPi = 3.14159265358979323846F;
    value = std::fmod(value + kPi, 2.0F * kPi);
    if (value < 0.0F) value += 2.0F * kPi;
    return value - kPi;
}

bool same_pose(
    const interaction::Pose& left,
    const interaction::Pose& right) {
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        if (!g1_posture_fixture::exact(
                left.positions[bone], right.positions[bone]) ||
            !g1_posture_fixture::exact(
                left.velocities[bone], right.velocities[bone]) ||
            !g1_posture_fixture::exact(
                left.rotations[bone], right.rotations[bone]) ||
            !g1_posture_fixture::exact(
                left.angular_velocities[bone],
                right.angular_velocities[bone])) {
            return false;
        }
    }
    return left.hand_dof == right.hand_dof &&
        left.hand_dof_velocities == right.hand_dof_velocities &&
        left.foot_contacts == right.foot_contacts;
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

reach::Pack return_fixture() {
    reach::Pack pack{};
    reach::Database& database = pack.database;
    database.version = 2U;
    database.endian_marker = 0x01020304U;
    database.fps_numerator = 25U;
    database.fps_denominator = 1U;
    database.frame_count = kReturnStop;
    database.bone_count = g1_skeleton::BoneCount;
    database.clip_count = 1U;
    database.source_count = 1U;
    database.parents.assign(
        g1_skeleton::kParents.begin(),
        g1_skeleton::kParents.end());
    database.range_starts = {0};
    database.range_stops = {
        static_cast<int32_t>(kReturnStop)};
    database.contact_frames = {
        static_cast<int32_t>(kContactFrame)};
    const float shoulder_deltas[kReturnStop] = {
        0.20F, 0.40F, 0.25F, 0.05F, -0.20F};
    for (size_t frame = 0U; frame < kReturnStop; ++frame) {
        append_pose(
            database,
            pose_with_shoulder_pitch(shoulder_deltas[frame]));
        database.source_frames.push_back(
            static_cast<int32_t>(100U + frame));
    }
    database.active_hands = {
        static_cast<uint8_t>(reach::Hand::Left)};
    database.augmentations = {
        static_cast<uint8_t>(reach::Augmentation::Captured)};
    database.source_indices = {0U};
    database.original_indices = {-1};
    database.source_names = {"return_fixture"};
    const interaction::Transform endpoint =
        g1_posture_fixture::left_hand_transform(
            reach::pose_at_frame(
                database,
                static_cast<int32_t>(kContactFrame)));
    database.endpoint_positions = {
        endpoint.position.x,
        endpoint.position.y,
        endpoint.position.z,
    };
    database.endpoint_rotations = {
        endpoint.rotation.w,
        endpoint.rotation.x,
        endpoint.rotation.y,
        endpoint.rotation.z,
    };
    database.approach_directions = {1.0F, 0.0F, 0.0F};
    pack.features.version = 2U;
    pack.features.endian_marker = 0x01020304U;
    pack.features.clip_count = 1U;
    pack.features.dimension = 10U;
    pack.features.values = {
        endpoint.position.x,
        endpoint.position.y,
        endpoint.position.z,
        1.0F,
        0.0F,
        0.0F,
        endpoint.rotation.w,
        endpoint.rotation.x,
        endpoint.rotation.y,
        endpoint.rotation.z,
    };
    return pack;
}

struct ReturnInput {
    reach::Pack pack = return_fixture();
    reach::Candidate candidate{
        0U,
        2U,
        reach::placement_yaw(2U),
        0.0F,
        0.0F,
    };
    reach::Query query{};
    interaction::Pose aligned_contact{};
    interaction::Pose solved_contact{};
    interaction::Transform hand_in_object{
        vec3(0.03F, -2.5F, 0.02F),
        quat_from_angle_axis(
            0.20F, vec3(0.0F, 1.0F, 0.0F)),
    };

    ReturnInput() {
        query.hand = reach::Hand::Left;
        query.target.position = vec3(1.1F, 1.0F, -0.6F);
        aligned_contact = reach::place_pose(
            pack,
            candidate.clip,
            candidate.yaw_index,
            query.target.position,
            static_cast<int32_t>(kContactFrame));
        const interaction::Transform aligned_hand =
            g1_posture_fixture::left_hand_transform(aligned_contact);
        query.target = {
            aligned_hand.position + vec3(0.0F, 0.012F, 0.0F),
            quat_mul(
                quat_from_angle_axis(
                    0.08F, vec3(0.0F, 1.0F, 0.0F)),
                aligned_hand.rotation),
        };
        query.approach_world = reach::place_direction(
            vec3(1.0F, 0.0F, 0.0F), candidate.yaw_index);
        aligned_contact = reach::place_pose(
            pack,
            candidate.clip,
            candidate.yaw_index,
            query.target.position,
            static_cast<int32_t>(kContactFrame));
        solved_contact = aligned_contact;
        interaction::PostureIKConfig config{};
        const interaction::PostureIKResult result =
            interaction::solve_hand_posture_ik_task_priority(
                solved_contact,
                interaction::Hand::Left,
                query.target,
                aligned_contact,
                interaction::decompose_upper_body(
                    aligned_contact, interaction::Hand::Left),
                config);
        require(result.accepted, "fixture contact IK was not accepted");
    }
};

void test_inverse_time_warp_is_continuous_and_object_rigid() {
    ReturnInput input{};
    input.hand_in_object = interaction::Transform{vec3(), quat()};
    reach::SearchConfig config{};
    config.coverage.accepted_orientation_radians = 0.02F;
    const reach::ShapedReturn shaped = reach::shape_recorded_return(
        input.pack,
        input.candidate,
        input.query,
        input.solved_contact,
        input.hand_in_object,
        vec3(0.006F, 0.006F, 0.006F),
        interaction::EnvironmentGeometry{},
        config);

    require(
        shaped.rejection == reach::ReturnRejection::None,
        "reachable return was rejected");
    require(
        shaped.poses.size() ==
            kReturnStop - kContactFrame,
        "return did not publish contact plus every recorded frame");
    require(
        same_pose(shaped.poses.front(), input.solved_contact),
        "return sample zero was not the exact solved contact");

    const interaction::Pose aligned_final = reach::place_pose(
        input.pack,
        input.candidate.clip,
        input.candidate.yaw_index,
        input.query.target.position,
        static_cast<int32_t>(kReturnStop - 1U));
    const interaction::Transform expected_final =
        g1_posture_fixture::left_hand_transform(aligned_final);
    const interaction::Transform actual_final =
        g1_posture_fixture::left_hand_transform(shaped.poses.back());
    require(
        near(actual_final.position, expected_final.position),
        "zero-weight return endpoint changed recorded wrist position");
    require(
        near(actual_final.rotation, expected_final.rotation),
        "zero-weight return endpoint changed recorded wrist rotation");

    const interaction::Transform aligned_contact_hand =
        g1_posture_fixture::left_hand_transform(input.aligned_contact);
    const interaction::Transform solved_contact_hand =
        g1_posture_fixture::left_hand_transform(input.solved_contact);
    const interaction::Transform contact_delta = interaction::compose(
        solved_contact_hand,
        interaction::inverse(aligned_contact_hand));
    const size_t return_count = shaped.poses.size() - 1U;
    for (size_t sample = 0U; sample < shaped.poses.size(); ++sample) {
        interaction::Transform expected_hand = solved_contact_hand;
        if (sample > 0U) {
            const interaction::Pose aligned_source = reach::place_pose(
                input.pack,
                input.candidate.clip,
                input.candidate.yaw_index,
                input.query.target.position,
                static_cast<int32_t>(kContactFrame + sample));
            const interaction::Transform aligned_source_hand =
                g1_posture_fixture::left_hand_transform(aligned_source);
            const float u = static_cast<float>(sample) /
                static_cast<float>(return_count);
            const float smooth = u * u * (3.0F - 2.0F * u);
            const float weight = 1.0F - smooth;
            expected_hand = interaction::compose(
                {
                    weight * contact_delta.position,
                    quat_nlerp_shortest(
                        quat(), contact_delta.rotation, weight),
                },
                aligned_source_hand);
        }
        const interaction::Transform actual_hand =
            g1_posture_fixture::left_hand_transform(shaped.poses[sample]);
        const interaction::Transform expected_object =
            interaction::compose(
                expected_hand,
                interaction::inverse(input.hand_in_object));
        const interaction::Transform actual_object =
            interaction::compose(
                actual_hand,
                interaction::inverse(input.hand_in_object));
        require(
            near(
                actual_object.position,
                expected_object.position,
                config.coverage.accepted_position_m + 1.0e-5F),
            "attached object left expected frozen-hand trajectory at sample " +
                std::to_string(sample));
        require(
            rotation_error(
                actual_object.rotation,
                expected_object.rotation) <=
                config.coverage.accepted_orientation_radians + 1.0e-5F,
            "attached object left expected frozen-hand orientation at sample " +
                std::to_string(sample));
    }
}

void test_unreachable_return_reports_invalid_solver() {
    ReturnInput input{};
    input.solved_contact.positions[g1_skeleton::Simulation] =
        input.solved_contact.positions[g1_skeleton::Simulation] +
        vec3(20.0F, 0.0F, 0.0F);

    const reach::ShapedReturn shaped = reach::shape_recorded_return(
        input.pack,
        input.candidate,
        input.query,
        input.solved_contact,
        input.hand_in_object,
        vec3(0.01F, 0.01F, 0.01F),
        interaction::EnvironmentGeometry{},
        reach::SearchConfig{});

    require(
        shaped.rejection == reach::ReturnRejection::InvalidSolver,
        "unreachable return did not report invalid solver");
    require(
        shaped.rejected_sample == 1U,
        "unreachable return did not identify its first solved frame");
}

void test_short_return_transports_correction_and_reaches_exact_endpoint() {
    ReturnInput input{};
    input.pack.database.range_stops.at(0) = 3;
    input.aligned_contact = reach::place_pose(
        input.pack,
        input.candidate.clip,
        input.candidate.yaw_index,
        input.query.target.position,
        static_cast<int32_t>(kContactFrame));
    const interaction::Transform aligned_contact_hand =
        g1_posture_fixture::left_hand_transform(input.aligned_contact);
    input.query.target = {
        aligned_contact_hand.position + vec3(0.0F, 0.025F, 0.0F),
        quat_mul(
            quat_from_angle_axis(
                0.18F, vec3(0.0F, 1.0F, 0.0F)),
            aligned_contact_hand.rotation),
    };
    input.solved_contact = input.aligned_contact;
    interaction::PostureIKConfig contact_config{};
    contact_config.maximum_iterations = 60;
    const interaction::PostureIKResult contact =
        interaction::solve_hand_posture_ik_task_priority(
            input.solved_contact,
            interaction::Hand::Left,
            input.query.target,
            input.aligned_contact,
            interaction::decompose_upper_body(
                input.aligned_contact, interaction::Hand::Left),
            contact_config);
    require(
        contact.accepted,
        "short-return fixture contact correction was not valid");

    const reach::ShapedReturn shaped = reach::shape_recorded_return(
        input.pack,
        input.candidate,
        input.query,
        input.solved_contact,
        input.hand_in_object,
        vec3(0.01F, 0.01F, 0.01F),
        interaction::EnvironmentGeometry{},
        reach::SearchConfig{});

    require(
        shaped.rejection == reach::ReturnRejection::None,
        "short reachable return was rejected");
    require(
        shaped.poses.size() == 2U,
        "one-frame return did not publish contact and endpoint");
    const interaction::Pose aligned_endpoint = reach::place_pose(
        input.pack,
        input.candidate.clip,
        input.candidate.yaw_index,
        input.query.target.position,
        2);
    const interaction::Transform expected =
        g1_posture_fixture::left_hand_transform(aligned_endpoint);
    const interaction::Transform actual =
        g1_posture_fixture::left_hand_transform(shaped.poses.back());
    const float position_error = length(actual.position - expected.position);
    const float orientation_error =
        rotation_error(actual.rotation, expected.rotation);
    require(
        position_error <= 1.0e-6F,
        "final wrist retained transported position correction: " +
            std::to_string(position_error));
    require(
        orientation_error <= 1.0e-6F,
        "final wrist retained transported orientation correction: " +
            std::to_string(orientation_error));

    const interaction::UpperBodyAngles contact_source_angles =
        interaction::decompose_upper_body(
            input.aligned_contact, interaction::Hand::Left);
    const interaction::UpperBodyAngles contact_solution_angles =
        interaction::decompose_upper_body(
            input.solved_contact, interaction::Hand::Left);
    const interaction::UpperBodyAngles endpoint_source_angles =
        interaction::decompose_upper_body(
            aligned_endpoint, interaction::Hand::Left);
    interaction::UpperBodyAngles transported_seed =
        endpoint_source_angles;
    float transported_seed_difference = 0.0F;
    for (size_t joint = 0U; joint < transported_seed.size(); ++joint) {
        transported_seed[joint] += wrap_angle(
            contact_solution_angles[joint] -
            contact_source_angles[joint]);
        transported_seed_difference = std::max(
            transported_seed_difference,
            std::abs(wrap_angle(
                transported_seed[joint] -
                endpoint_source_angles[joint])));
    }
    require(
        transported_seed_difference > 1.0e-3F,
        "large contact correction did not produce a distinct transported "
        "final seed");
    interaction::PostureIKConfig endpoint_config{};
    endpoint_config.accepted_position_m = 1.0e-6F;
    endpoint_config.accepted_orientation_radians = 1.0e-6F;
    interaction::Pose transported_endpoint = aligned_endpoint;
    const interaction::PostureIKResult transported =
        interaction::solve_hand_posture_ik_task_priority(
            transported_endpoint,
            interaction::Hand::Left,
            expected,
            aligned_endpoint,
            transported_seed,
            endpoint_config);
    require(
        transported.accepted,
        "transported final-frame reference solve was not accepted");
    const interaction::UpperBodyAngles actual_angles =
        interaction::decompose_upper_body(
            shaped.poses.back(), interaction::Hand::Left);
    float transported_match_error = 0.0F;
    for (size_t joint = 0U; joint < actual_angles.size(); ++joint) {
        transported_match_error = std::max(
            transported_match_error,
            std::abs(wrap_angle(
                actual_angles[joint] -
                transported.joint_angles[joint])));
    }
    require(
        transported_match_error <= 1.0e-5F,
        "final-frame posture did not preserve transported correction: " +
            std::to_string(transported_match_error));
}

void test_held_object_crossing_rotated_box_reports_environment_collision() {
    ReturnInput input{};
    input.solved_contact = input.aligned_contact;
    const interaction::Pose aligned_final = reach::place_pose(
        input.pack,
        input.candidate.clip,
        input.candidate.yaw_index,
        input.query.target.position,
        static_cast<int32_t>(kReturnStop - 1U));
    const interaction::Transform final_hand =
        g1_posture_fixture::left_hand_transform(aligned_final);
    const interaction::Transform final_object =
        interaction::compose(
            final_hand,
            interaction::inverse(input.hand_in_object));
    interaction::EnvironmentGeometry environment{};
    environment.boxes.push_back({
        {
            final_object.position,
            quat_mul(
                quat_from_angle_axis(
                    0.63F, normalize(vec3(1.0F, 1.0F, 0.4F))),
                final_object.rotation),
        },
        vec3(0.008F, 0.008F, 0.008F),
    });

    const reach::ShapedReturn shaped = reach::shape_recorded_return(
        input.pack,
        input.candidate,
        input.query,
        input.solved_contact,
        input.hand_in_object,
        vec3(0.008F, 0.008F, 0.008F),
        environment,
        reach::SearchConfig{});

    require(
        shaped.rejection == reach::ReturnRejection::EnvironmentCollision,
        "held object crossing rotated box was not rejected");
    require(
        shaped.rejected_sample ==
            kReturnStop - kContactFrame - 1U,
        "held-object collision was not reported at the first overlap");
}

void test_body_inside_held_object_reports_object_collision() {
    ReturnInput input{};
    input.solved_contact = input.aligned_contact;
    const interaction::WorldPose world =
        interaction::world_pose(input.solved_contact);
    const interaction::Transform hand{
        world.positions[g1_skeleton::LeftWrist],
        world.rotations[g1_skeleton::LeftWrist],
    };
    const interaction::Transform body_object{
        world.positions[g1_skeleton::Spine1],
        quat(),
    };
    input.hand_in_object = interaction::compose(
        interaction::inverse(body_object), hand);

    const reach::ShapedReturn shaped = reach::shape_recorded_return(
        input.pack,
        input.candidate,
        input.query,
        input.solved_contact,
        input.hand_in_object,
        vec3(0.04F, 0.04F, 0.04F),
        interaction::EnvironmentGeometry{},
        reach::SearchConfig{});

    require(
        shaped.rejection == reach::ReturnRejection::ObjectCollision,
        "body inside held object did not report object collision");
    require(
        shaped.rejected_sample == 0U,
        "body/object collision did not report contact sample");
}

void test_body_inside_environment_reports_environment_collision() {
    ReturnInput input{};
    input.solved_contact = input.aligned_contact;
    const interaction::WorldPose world =
        interaction::world_pose(input.solved_contact);
    interaction::EnvironmentGeometry environment{};
    environment.boxes.push_back({
        {
            world.positions[g1_skeleton::Hips],
            quat_from_angle_axis(
                0.31F, normalize(vec3(1.0F, 0.4F, 0.2F))),
        },
        vec3(0.02F, 0.02F, 0.02F),
    });

    const reach::ShapedReturn shaped = reach::shape_recorded_return(
        input.pack,
        input.candidate,
        input.query,
        input.solved_contact,
        input.hand_in_object,
        vec3(0.008F, 0.008F, 0.008F),
        environment,
        reach::SearchConfig{});

    require(
        shaped.rejection == reach::ReturnRejection::EnvironmentCollision,
        "body inside environment did not report environment collision");
    require(
        shaped.rejected_sample == 0U,
        "body/environment collision did not report contact sample");
}

void test_active_grasp_wrist_inside_held_object_is_exempt() {
    ReturnInput input{};
    input.solved_contact = input.aligned_contact;
    input.hand_in_object = interaction::Transform{
        vec3(), quat()};

    const reach::ShapedReturn shaped = reach::shape_recorded_return(
        input.pack,
        input.candidate,
        input.query,
        input.solved_contact,
        input.hand_in_object,
        vec3(0.006F, 0.006F, 0.006F),
        interaction::EnvironmentGeometry{},
        reach::SearchConfig{});

    require(
        shaped.rejection == reach::ReturnRejection::None,
        "active grasp wrist inside held object was not exempt");
    require(
        shaped.poses.size() == kReturnStop - kContactFrame,
        "active-wrist exemption did not preserve the full return");
}

void test_sat_tolerance_ignores_submicrometre_contact_but_rejects_overlap() {
    ReturnInput input{};
    input.solved_contact = input.aligned_contact;
    constexpr vec3 object_dimensions(0.008F, 0.008F, 0.008F);
    float minimum_bottom = std::numeric_limits<float>::infinity();
    size_t minimum_sample = 0U;
    for (int32_t frame = static_cast<int32_t>(kContactFrame);
         frame < static_cast<int32_t>(kReturnStop);
         ++frame) {
        const interaction::Pose pose = reach::place_pose(
            input.pack,
            input.candidate.clip,
            input.candidate.yaw_index,
            input.query.target.position,
            frame);
        const interaction::Transform hand =
            g1_posture_fixture::left_hand_transform(pose);
        const interaction::Transform object =
            interaction::compose(
                hand,
                interaction::inverse(input.hand_in_object));
        const vec3 axis_x = quat_mul_vec3(
            object.rotation, vec3(1.0F, 0.0F, 0.0F));
        const vec3 axis_y = quat_mul_vec3(
            object.rotation, vec3(0.0F, 1.0F, 0.0F));
        const vec3 axis_z = quat_mul_vec3(
            object.rotation, vec3(0.0F, 0.0F, 1.0F));
        const float vertical_radius = 0.5F * (
            object_dimensions.x * std::abs(axis_x.y) +
            object_dimensions.y * std::abs(axis_y.y) +
            object_dimensions.z * std::abs(axis_z.y));
        const float bottom = object.position.y - vertical_radius;
        if (bottom < minimum_bottom) {
            minimum_bottom = bottom;
            minimum_sample = static_cast<size_t>(
                frame - static_cast<int32_t>(kContactFrame));
        }
    }
    const interaction::WorldPose contact_world =
        interaction::world_pose(input.solved_contact);
    const interaction::Transform contact_hand{
        contact_world.positions[g1_skeleton::LeftWrist],
        contact_world.rotations[g1_skeleton::LeftWrist],
    };
    const interaction::OrientedBox contact_object{
        interaction::compose(
            contact_hand,
            interaction::inverse(input.hand_in_object)),
        object_dimensions,
    };
    const auto environment_with_penetration =
        [&](float penetration_m) {
        interaction::EnvironmentGeometry environment{};
        environment.boxes.push_back({
            {
                vec3(
                    0.0F,
                    minimum_bottom - 0.05F + penetration_m,
                    0.0F),
                quat(),
            },
            vec3(20.0F, 0.10F, 20.0F),
        });
        return environment;
    };
    const interaction::EnvironmentGeometry submicrometre =
        environment_with_penetration(0.5e-6F);
    interaction::ShapedHandTrajectory single{};
    single.poses.push_back(input.solved_contact);
    single.path.hands.push_back(contact_hand);
    single.path.elbows.push_back(
        contact_world.positions[g1_skeleton::LeftElbow]);
    single.contact_accepted = true;
    const interaction::TrajectoryFeasibility body_feasibility =
        interaction::evaluate_shaped_trajectory_feasibility(
            single,
            0U,
            interaction::Hand::Left,
            contact_object,
            submicrometre);
    require(
        body_feasibility.reason ==
            interaction::TrajectoryFeasibilityReason::None,
        "support-contact fixture intersects the body");

    const reach::ShapedReturn tolerated = reach::shape_recorded_return(
        input.pack,
        input.candidate,
        input.query,
        input.solved_contact,
        input.hand_in_object,
        object_dimensions,
        submicrometre,
        reach::SearchConfig{});
    require(
        tolerated.rejection == reach::ReturnRejection::None,
        "0.5 micrometre SAT penetration exceeded nominal 1 micrometre "
        "overlap tolerance");

    const reach::ShapedReturn overlapping = reach::shape_recorded_return(
        input.pack,
        input.candidate,
        input.query,
        input.solved_contact,
        input.hand_in_object,
        object_dimensions,
        environment_with_penetration(2.0e-6F),
        reach::SearchConfig{});
    require(
        overlapping.rejection ==
            reach::ReturnRejection::EnvironmentCollision,
        "2 micrometre SAT penetration was not rejected");
    require(
        overlapping.rejected_sample == minimum_sample,
        "SAT boundary collision reported the wrong explicit sample");
}

}  // namespace

int main() {
    try {
        test_inverse_time_warp_is_continuous_and_object_rigid();
        test_unreachable_return_reports_invalid_solver();
        test_short_return_transports_correction_and_reaches_exact_endpoint();
        test_held_object_crossing_rotated_box_reports_environment_collision();
        test_body_inside_held_object_reports_object_collision();
        test_body_inside_environment_reports_environment_collision();
        test_active_grasp_wrist_inside_held_object_is_exempt();
        test_sat_tolerance_ignores_submicrometre_contact_but_rejects_overlap();
        std::cout << "reach return PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "reach return FAILED: " << error.what() << '\n';
        return 1;
    }
}
