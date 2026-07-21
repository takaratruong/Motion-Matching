#include "interaction_native_g1_bridge.h"

#include "array.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool bits_equal(float left, float right) {
    return std::memcmp(&left, &right, sizeof(float)) == 0;
}

bool bits_equal(vec3 left, vec3 right) {
    return bits_equal(left.x, right.x) &&
           bits_equal(left.y, right.y) &&
           bits_equal(left.z, right.z);
}

bool bits_equal(quat left, quat right) {
    return bits_equal(left.w, right.w) &&
           bits_equal(left.x, right.x) &&
           bits_equal(left.y, right.y) &&
           bits_equal(left.z, right.z);
}

bool pose_bits_equal(
    const interaction::Pose& left,
    const interaction::Pose& right) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        if (!bits_equal(left.positions[bone], right.positions[bone]) ||
            !bits_equal(left.velocities[bone], right.velocities[bone]) ||
            !bits_equal(left.rotations[bone], right.rotations[bone]) ||
            !bits_equal(
                left.angular_velocities[bone],
                right.angular_velocities[bone])) {
            return false;
        }
    }
    return left.hand_dof == right.hand_dof &&
           left.hand_dof_velocities == right.hand_dof_velocities &&
           left.foot_contacts == right.foot_contacts;
}

interaction::Pose make_pose(float base) {
    interaction::Pose pose{};
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        const float value = base + static_cast<float>(bone);
        pose.positions[bone] = vec3(value, value + 0.25F, value + 0.50F);
        pose.velocities[bone] = vec3(value + 1.0F, value + 1.25F, value + 1.5F);
        pose.rotations[bone] = quat_from_angle_axis(
            0.002F * value,
            normalize(vec3(1.0F, 2.0F, 3.0F)));
        pose.angular_velocities[bone] =
            vec3(value + 2.0F, value + 2.25F, value + 2.5F);
    }
    for (size_t dof = 0; dof < pose.hand_dof.size(); ++dof) {
        pose.hand_dof[dof] = base + static_cast<float>(dof) * 0.1F;
        pose.hand_dof_velocities[dof] =
            base + 5.0F + static_cast<float>(dof) * 0.1F;
    }
    pose.foot_contacts = {1U, 0U};
    return pose;
}

void test_capture_and_write_are_exact() {
    constexpr int bone_count = static_cast<int>(g1_skeleton::BoneCount);
    std::array<vec3, g1_skeleton::BoneCount> positions{};
    std::array<vec3, g1_skeleton::BoneCount> velocities{};
    std::array<quat, g1_skeleton::BoneCount> rotations{};
    std::array<vec3, g1_skeleton::BoneCount> angular_velocities{};
    std::array<bool, 2> contacts{true, false};
    std::array<vec3, 3> trajectory_positions{};
    std::array<quat, 3> trajectory_rotations{};

    for (int bone = 0; bone < bone_count; ++bone) {
        const float value = 10.0F + static_cast<float>(bone);
        positions[bone] = vec3(value, value + 0.1F, value + 0.2F);
        velocities[bone] = vec3(value + 1.0F, value + 1.1F, value + 1.2F);
        rotations[bone] = quat_from_angle_axis(
            0.003F * value,
            normalize(vec3(3.0F, 1.0F, 2.0F)));
        angular_velocities[bone] =
            vec3(value + 2.0F, value + 2.1F, value + 2.2F);
    }
    for (size_t sample = 0; sample < trajectory_positions.size(); ++sample) {
        const float value = 100.0F + static_cast<float>(sample);
        trajectory_positions[sample] = vec3(value, value + 1.0F, value + 2.0F);
        trajectory_rotations[sample] = quat_from_angle_axis(
            0.1F * static_cast<float>(sample + 1U),
            vec3(0.0F, 1.0F, 0.0F));
    }

    const interaction::LocomotionSnapshot snapshot =
        interaction::capture_native_g1_snapshot(
            slice1d<vec3>(bone_count, positions.data()),
            slice1d<vec3>(bone_count, velocities.data()),
            slice1d<quat>(bone_count, rotations.data()),
            slice1d<vec3>(bone_count, angular_velocities.data()),
            slice1d<bool>(2, contacts.data()),
            slice1d<vec3>(3, trajectory_positions.data()),
            slice1d<quat>(3, trajectory_rotations.data()));

    for (int bone = 0; bone < bone_count; ++bone) {
        require(bits_equal(snapshot.pose.positions[bone], positions[bone]),
                "captured position changed bits");
        require(bits_equal(snapshot.pose.velocities[bone], velocities[bone]),
                "captured velocity changed bits");
        require(bits_equal(snapshot.pose.rotations[bone], rotations[bone]),
                "captured rotation changed bits");
        require(bits_equal(
                    snapshot.pose.angular_velocities[bone],
                    angular_velocities[bone]),
                "captured angular velocity changed bits");
    }
    require(snapshot.pose.foot_contacts[0] == 1U,
            "left foot contact was not captured");
    require(snapshot.pose.foot_contacts[1] == 0U,
            "right foot contact was not captured");
    for (size_t sample = 0; sample < trajectory_positions.size(); ++sample) {
        require(bits_equal(
                    snapshot.future_root_positions[sample],
                    trajectory_positions[sample]),
                "future root position changed bits");
        require(bits_equal(
                    snapshot.future_root_rotations[sample],
                    trajectory_rotations[sample]),
                "future root rotation changed bits");
    }

    std::array<vec3, g1_skeleton::BoneCount> written_positions{};
    std::array<vec3, g1_skeleton::BoneCount> written_velocities{};
    std::array<quat, g1_skeleton::BoneCount> written_rotations{};
    std::array<vec3, g1_skeleton::BoneCount> written_angular_velocities{};
    std::array<bool, 2> written_contacts{false, false};
    interaction::write_native_g1_pose(
        snapshot.pose,
        slice1d<vec3>(bone_count, written_positions.data()),
        slice1d<vec3>(bone_count, written_velocities.data()),
        slice1d<quat>(bone_count, written_rotations.data()),
        slice1d<vec3>(bone_count, written_angular_velocities.data()),
        slice1d<bool>(2, written_contacts.data()));

    for (int bone = 0; bone < bone_count; ++bone) {
        require(bits_equal(written_positions[bone], positions[bone]),
                "written position changed bits");
        require(bits_equal(written_velocities[bone], velocities[bone]),
                "written velocity changed bits");
        require(bits_equal(written_rotations[bone], rotations[bone]),
                "written rotation changed bits");
        require(bits_equal(
                    written_angular_velocities[bone],
                    angular_velocities[bone]),
                "written angular velocity changed bits");
    }
    require(written_contacts == contacts, "written contacts changed");
}

void test_capture_rejects_wrong_shapes() {
    std::array<vec3, g1_skeleton::BoneCount> vectors{};
    std::array<quat, g1_skeleton::BoneCount> rotations{};
    std::array<bool, 2> contacts{};
    std::array<vec3, 3> trajectory_positions{};
    std::array<quat, 3> trajectory_rotations{};
    bool threw = false;
    try {
        (void)interaction::capture_native_g1_snapshot(
            slice1d<vec3>(30, vectors.data()),
            slice1d<vec3>(31, vectors.data()),
            slice1d<quat>(31, rotations.data()),
            slice1d<vec3>(31, vectors.data()),
            slice1d<bool>(2, contacts.data()),
            slice1d<vec3>(3, trajectory_positions.data()),
            slice1d<quat>(3, trajectory_rotations.data()));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "capture accepted a 30-bone position slice");
}

void test_native_handoff_owns_then_releases_without_a_snap() {
    interaction::NativeG1PoseHandoff handoff;
    const interaction::Pose locomotion_pose = make_pose(50.0F);
    const interaction::Pose interaction_pose = make_pose(-20.0F);
    interaction::RuntimeOutput runtime{};
    runtime.owns_pose = true;
    runtime.pose = interaction_pose;

    const interaction::NativeG1FrameState owned =
        handoff.apply(locomotion_pose, runtime, 0.04F);
    require(owned.runtime_owns_pose, "runtime ownership was lost");
    require(owned.synchronize_simulation_root,
            "owned root was not synchronized");
    require(pose_bits_equal(owned.pose, interaction_pose),
            "owned pose was modified");

    runtime.owns_pose = false;
    const interaction::NativeG1FrameState first_release =
        handoff.apply(locomotion_pose, runtime, 0.04F);
    require(!first_release.runtime_owns_pose,
            "release retained runtime ownership");
    require(pose_bits_equal(first_release.pose, interaction_pose),
            "first release frame snapped away from displayed pose");

    float previous_error = std::abs(
        first_release.pose.positions[0].x -
        locomotion_pose.positions[0].x);
    interaction::NativeG1FrameState released = first_release;
    for (int frame = 0; frame < 7; ++frame) {
        released = handoff.apply(locomotion_pose, runtime, 0.04F);
        const float error = std::abs(
            released.pose.positions[0].x -
            locomotion_pose.positions[0].x);
        require(error <= previous_error + 1.0e-6F,
                "release did not converge monotonically");
        previous_error = error;
    }
    require(pose_bits_equal(released.pose, locomotion_pose),
            "release did not finish at exact locomotion pose");

    interaction::NativeG1PoseHandoff inactive;
    const interaction::NativeG1FrameState passthrough =
        inactive.apply(locomotion_pose, runtime, 0.04F);
    require(pose_bits_equal(passthrough.pose, locomotion_pose),
            "inactive handoff modified locomotion");
}

}  // namespace

int main() {
    test_capture_and_write_are_exact();
    test_capture_rejects_wrong_shapes();
    test_native_handoff_owns_then_releases_without_a_snap();
}
