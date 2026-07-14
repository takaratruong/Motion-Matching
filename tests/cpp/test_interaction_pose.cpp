#include "interaction_pose.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

constexpr float kTolerance = 1.0e-5F;

bool near(float left, float right, float tolerance = kTolerance) {
    return std::abs(left - right) <= tolerance;
}

bool near(vec3 left, vec3 right, float tolerance = kTolerance) {
    return near(left.x, right.x, tolerance) &&
           near(left.y, right.y, tolerance) &&
           near(left.z, right.z, tolerance);
}

bool near(quat left, quat right, float tolerance = kTolerance) {
    return near(left.w, right.w, tolerance) &&
           near(left.x, right.x, tolerance) &&
           near(left.y, right.y, tolerance) &&
           near(left.z, right.z, tolerance);
}

template<class Function>
bool throws_range_error(Function&& function) {
    try {
        function();
    } catch (const std::out_of_range&) {
        return true;
    } catch (...) {
    }
    return false;
}

size_t vector_offset(uint32_t frame, uint32_t bone) {
    return (static_cast<size_t>(frame) * g1_skeleton::BoneCount + bone) * 3U;
}

size_t quaternion_offset(uint32_t frame, uint32_t bone) {
    return (static_cast<size_t>(frame) * g1_skeleton::BoneCount + bone) * 4U;
}

void store_vec3(
    std::vector<float>& values,
    uint32_t frame,
    uint32_t bone,
    vec3 value) {
    const size_t offset = vector_offset(frame, bone);
    values[offset] = value.x;
    values[offset + 1U] = value.y;
    values[offset + 2U] = value.z;
}

void store_quat(
    std::vector<float>& values,
    uint32_t frame,
    uint32_t bone,
    quat value) {
    const size_t offset = quaternion_offset(frame, bone);
    values[offset] = value.w;
    values[offset + 1U] = value.x;
    values[offset + 2U] = value.y;
    values[offset + 3U] = value.z;
}

interaction::Database make_database() {
    using namespace interaction;

    Database database{};
    database.fps_numerator = 25;
    database.fps_denominator = 1;
    database.frame_count = 2;
    database.bone_count = g1_skeleton::BoneCount;
    database.clip_count = 1;
    database.hand_dof_count = 14;
    database.range_starts = {0};
    database.range_stops = {2};

    const size_t vector_value_count =
        static_cast<size_t>(database.frame_count) * database.bone_count * 3U;
    const size_t quaternion_value_count =
        static_cast<size_t>(database.frame_count) * database.bone_count * 4U;
    database.positions.resize(vector_value_count);
    database.velocities.resize(vector_value_count);
    database.rotations.resize(quaternion_value_count);
    database.angular_velocities.resize(vector_value_count);
    database.foot_contacts = {0U, 1U, 1U, 0U};
    database.hand_dof.resize(
        static_cast<size_t>(database.frame_count) * database.hand_dof_count);
    database.hand_dof_velocities.resize(database.hand_dof.size());

    for (uint32_t frame = 0; frame < database.frame_count; ++frame) {
        for (uint32_t bone = 0; bone < database.bone_count; ++bone) {
            const float base = static_cast<float>(frame * 100U + bone * 3U);
            store_vec3(
                database.positions, frame, bone,
                vec3(base, base + 1.0F, base + 2.0F));
            store_vec3(
                database.velocities, frame, bone,
                vec3(base + 300.0F, base + 301.0F, base + 302.0F));
            store_quat(database.rotations, frame, bone, quat());
            store_vec3(
                database.angular_velocities, frame, bone,
                vec3(base + 600.0F, base + 601.0F, base + 602.0F));
        }
        for (uint32_t dof = 0; dof < database.hand_dof_count; ++dof) {
            const size_t offset =
                static_cast<size_t>(frame) * database.hand_dof_count + dof;
            database.hand_dof[offset] = static_cast<float>(frame * 20U + dof);
            database.hand_dof_velocities[offset] =
                static_cast<float>(frame * 20U + dof) + 40.0F;
        }
    }

    return database;
}

void test_transform_primitives() {
    using namespace interaction;

    const Transform a{vec3(1.0F, 0.0F, 2.0F), quat()};
    const Transform b{vec3(3.0F, 4.0F, 5.0F), quat()};
    assert(near(compose(a, b).position, vec3(4.0F, 4.0F, 7.0F)));
    assert(near(compose(a, inverse(a)).position, vec3()));

    const Transform turned{
        vec3(2.0F, 1.0F, -3.0F),
        quat_from_angle_axis(PIf / 2.0F, vec3(0.0F, 1.0F, 0.0F))};
    const Transform local{vec3(0.0F, 0.0F, 2.0F), quat()};
    assert(near(compose(turned, local).position, vec3(4.0F, 1.0F, -3.0F)));
    const Transform identity = compose(turned, inverse(turned));
    assert(near(identity.position, vec3()));
    assert(near(quat_abs(identity.rotation), quat()));
}

void test_pose_at_frame() {
    using namespace interaction;

    const Database database = make_database();
    const Pose pose = pose_at_frame(database, 1);
    assert(near(pose.positions[g1_skeleton::RightWrist],
                vec3(190.0F, 191.0F, 192.0F)));
    assert(near(pose.velocities[g1_skeleton::LeftToe],
                vec3(421.0F, 422.0F, 423.0F)));
    assert(near(pose.rotations[g1_skeleton::Spine], quat()));
    assert(near(pose.angular_velocities[g1_skeleton::RightWrist],
                vec3(790.0F, 791.0F, 792.0F)));
    assert(near(pose.hand_dof[13], 33.0F));
    assert(near(pose.hand_dof_velocities[13], 73.0F));
    assert(pose.foot_contacts[0] == 1U);
    assert(pose.foot_contacts[1] == 0U);
    assert(throws_range_error([&] { pose_at_frame(database, -1); }));
    assert(throws_range_error([&] { pose_at_frame(database, 2); }));
}

void test_interpolate_pose() {
    using namespace interaction;

    Pose left{};
    Pose right{};
    left.positions[0] = vec3(0.0F, 2.0F, 4.0F);
    right.positions[0] = vec3(8.0F, 6.0F, 4.0F);
    left.velocities[0] = vec3(1.0F, 3.0F, 5.0F);
    right.velocities[0] = vec3(5.0F, 7.0F, 9.0F);
    left.rotations[0] = quat();
    right.rotations[0] = -quat();
    left.angular_velocities[0] = vec3(2.0F, 4.0F, 6.0F);
    right.angular_velocities[0] = vec3(6.0F, 8.0F, 10.0F);
    left.hand_dof[0] = 2.0F;
    right.hand_dof[0] = 10.0F;
    left.hand_dof_velocities[0] = 3.0F;
    right.hand_dof_velocities[0] = 7.0F;
    left.foot_contacts = {0U, 1U};
    right.foot_contacts = {1U, 0U};

    const Pose quarter = interpolate_pose(left, right, 0.25F);
    assert(near(quarter.positions[0], vec3(2.0F, 3.0F, 4.0F)));
    assert(near(quarter.velocities[0], vec3(2.0F, 4.0F, 6.0F)));
    assert(near(quarter.rotations[0], quat()));
    assert(near(quarter.angular_velocities[0], vec3(3.0F, 5.0F, 7.0F)));
    assert(near(quarter.hand_dof[0], 4.0F));
    assert(near(quarter.hand_dof_velocities[0], 4.0F));
    assert(quarter.foot_contacts == left.foot_contacts);
    assert(interpolate_pose(left, right, 0.5F).foot_contacts ==
           right.foot_contacts);
}

void test_world_pose_hierarchy() {
    using namespace interaction;

    Pose pose{};
    pose.positions[g1_skeleton::Simulation] = vec3(2.0F, 0.0F, 3.0F);
    pose.positions[g1_skeleton::Hips] = vec3(0.0F, 1.0F, 0.0F);
    const WorldPose world = world_pose(pose);
    assert(near(world.positions[g1_skeleton::Hips],
                vec3(2.0F, 1.0F, 3.0F)));

    Pose moving{};
    moving.positions[g1_skeleton::Simulation] = vec3(1.0F, 2.0F, 3.0F);
    moving.velocities[g1_skeleton::Simulation] = vec3(4.0F, 5.0F, 6.0F);
    moving.rotations[g1_skeleton::Simulation] =
        quat_from_angle_axis(PIf / 2.0F, vec3(0.0F, 1.0F, 0.0F));
    moving.angular_velocities[g1_skeleton::Simulation] =
        vec3(0.0F, 2.0F, 0.0F);
    moving.positions[g1_skeleton::Hips] = vec3(0.0F, 0.0F, 2.0F);
    moving.velocities[g1_skeleton::Hips] = vec3(1.0F, 0.0F, 0.0F);
    moving.rotations[g1_skeleton::Hips] =
        quat_from_angle_axis(PIf / 3.0F, vec3(1.0F, 0.0F, 0.0F));
    moving.angular_velocities[g1_skeleton::Hips] =
        vec3(0.0F, 0.0F, 3.0F);

    const WorldPose moving_world = world_pose(moving);
    const vec3 offset = quat_mul_vec3(
        moving.rotations[g1_skeleton::Simulation],
        moving.positions[g1_skeleton::Hips]);
    assert(near(moving_world.positions[g1_skeleton::Hips],
                moving.positions[g1_skeleton::Simulation] + offset));
    assert(near(
        moving_world.velocities[g1_skeleton::Hips],
        moving.velocities[g1_skeleton::Simulation] +
            cross(moving.angular_velocities[g1_skeleton::Simulation], offset) +
            quat_mul_vec3(
                moving.rotations[g1_skeleton::Simulation],
                moving.velocities[g1_skeleton::Hips])));
    assert(near(
        moving_world.rotations[g1_skeleton::Hips],
        quat_normalize(quat_mul(
            moving.rotations[g1_skeleton::Simulation],
            moving.rotations[g1_skeleton::Hips]))));
    assert(near(
        moving_world.angular_velocities[g1_skeleton::Hips],
        moving.angular_velocities[g1_skeleton::Simulation] +
            quat_mul_vec3(
                moving.rotations[g1_skeleton::Simulation],
                moving.angular_velocities[g1_skeleton::Hips])));
}

void test_sample_pose() {
    using namespace interaction;

    const Database database = make_database();
    const Pose halfway = sample_pose(database, 0, 0.02F);
    assert(near(halfway.positions[0],
                0.5F * (pose_at_frame(database, 0).positions[0] +
                        pose_at_frame(database, 1).positions[0])));
    assert(halfway.foot_contacts == pose_at_frame(database, 1).foot_contacts);
    assert(near(sample_pose(database, 0, 0.04F).positions[0],
                pose_at_frame(database, 1).positions[0]));
    assert(throws_range_error([&] { sample_pose(database, 0, 99.0F); }));
    assert(throws_range_error([&] { sample_pose(database, 0, -0.01F); }));
}

}  // namespace

int main() {
    test_transform_primitives();
    test_pose_at_frame();
    test_interpolate_pose();
    test_world_pose_hierarchy();
    test_sample_pose();
}
