#include "g1_posture_ik_fixture.h"
#include "reach_placement.h"
#include "reach_return.h"

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
    const ReturnInput input{};
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

    for (const interaction::Pose& pose : shaped.poses) {
        const interaction::Transform solved_hand =
            g1_posture_fixture::left_hand_transform(pose);
        const interaction::Transform object_world =
            interaction::compose(
                solved_hand,
                interaction::inverse(input.hand_in_object));
        const interaction::Transform reconstructed_hand =
            interaction::compose(
                object_world, input.hand_in_object);
        require(
            near(reconstructed_hand.position, solved_hand.position, 1.0e-5F),
            "frozen hand-in-object did not reconstruct wrist position");
        require(
            near(reconstructed_hand.rotation, solved_hand.rotation, 1.0e-5F),
            "frozen hand-in-object did not reconstruct wrist rotation");
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
        shaped.rejected_sample == shaped.poses.size() - 1U,
        "held-object collision was not reported at the first overlap");
}

void test_zero_depth_support_contact_is_not_object_overlap() {
    ReturnInput input{};
    input.solved_contact = input.aligned_contact;
    constexpr vec3 object_dimensions(0.008F, 0.008F, 0.008F);
    float minimum_bottom = std::numeric_limits<float>::infinity();
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
        minimum_bottom = std::min(
            minimum_bottom,
            object.position.y - vertical_radius);
    }
    interaction::EnvironmentGeometry environment{};
    environment.boxes.push_back({
        {
            vec3(0.0F, minimum_bottom - 0.05F, 0.0F),
            quat(),
        },
        vec3(20.0F, 0.10F, 20.0F),
    });
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
    const vec3 contact_axis_x = quat_mul_vec3(
        contact_object.world.rotation, vec3(1.0F, 0.0F, 0.0F));
    const vec3 contact_axis_y = quat_mul_vec3(
        contact_object.world.rotation, vec3(0.0F, 1.0F, 0.0F));
    const vec3 contact_axis_z = quat_mul_vec3(
        contact_object.world.rotation, vec3(0.0F, 0.0F, 1.0F));
    const float contact_vertical_radius = 0.5F * (
        object_dimensions.x * std::abs(contact_axis_x.y) +
        object_dimensions.y * std::abs(contact_axis_y.y) +
        object_dimensions.z * std::abs(contact_axis_z.y));
    const float contact_bottom =
        contact_object.world.position.y - contact_vertical_radius;
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
            environment);
    require(
        body_feasibility.reason ==
            interaction::TrajectoryFeasibilityReason::None,
        "support-contact fixture intersects the body");

    const reach::ShapedReturn shaped = reach::shape_recorded_return(
        input.pack,
        input.candidate,
        input.query,
        input.solved_contact,
        input.hand_in_object,
        object_dimensions,
        environment,
        reach::SearchConfig{});

    require(
        shaped.rejection == reach::ReturnRejection::None,
        "zero-depth support contact was treated as object overlap: rejection " +
            std::to_string(static_cast<int>(shaped.rejection)) +
            " at sample " + std::to_string(shaped.rejected_sample) +
            ", contact gap " +
            std::to_string(contact_bottom - minimum_bottom));
}

}  // namespace

int main() {
    try {
        test_inverse_time_warp_is_continuous_and_object_rigid();
        test_unreachable_return_reports_invalid_solver();
        test_held_object_crossing_rotated_box_reports_environment_collision();
        test_zero_depth_support_contact_is_not_object_overlap();
        std::cout << "reach return PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "reach return FAILED: " << error.what() << '\n';
        return 1;
    }
}
