#include "interaction_native_g1_bridge.h"

#include "common.h"

#include <cstddef>
#include <stdexcept>

namespace interaction {
namespace {

constexpr int kNativeG1BoneCount =
    static_cast<int>(g1_skeleton::BoneCount);
constexpr int kFootContactCount = 2;
constexpr int kFutureRootSampleCount = 3;
constexpr float kReleaseDurationSeconds = 0.25F;

template<typename T>
void require_slice(slice1d<T> values, int expected, const char* name) {
    if (values.size != expected || values.data == nullptr) {
        throw std::invalid_argument(name);
    }
}

void require_pose_slices(
    slice1d<vec3> positions,
    slice1d<vec3> velocities,
    slice1d<quat> rotations,
    slice1d<vec3> angular_velocities,
    slice1d<bool> contacts) {
    require_slice(positions, kNativeG1BoneCount, "native G1 positions");
    require_slice(velocities, kNativeG1BoneCount, "native G1 velocities");
    require_slice(rotations, kNativeG1BoneCount, "native G1 rotations");
    require_slice(
        angular_velocities,
        kNativeG1BoneCount,
        "native G1 angular velocities");
    require_slice(contacts, kFootContactCount, "native G1 foot contacts");
}

}  // namespace

LocomotionSnapshot capture_native_g1_snapshot(
    slice1d<vec3> positions,
    slice1d<vec3> velocities,
    slice1d<quat> rotations,
    slice1d<vec3> angular_velocities,
    slice1d<bool> contacts,
    slice1d<vec3> trajectory_positions,
    slice1d<quat> trajectory_rotations) {
    require_pose_slices(
        positions,
        velocities,
        rotations,
        angular_velocities,
        contacts);
    require_slice(
        trajectory_positions,
        kFutureRootSampleCount,
        "native G1 future root positions");
    require_slice(
        trajectory_rotations,
        kFutureRootSampleCount,
        "native G1 future root rotations");

    LocomotionSnapshot snapshot{};
    for (int bone = 0; bone < kNativeG1BoneCount; ++bone) {
        const size_t index = static_cast<size_t>(bone);
        snapshot.pose.positions[index] = positions(bone);
        snapshot.pose.velocities[index] = velocities(bone);
        snapshot.pose.rotations[index] = rotations(bone);
        snapshot.pose.angular_velocities[index] = angular_velocities(bone);
    }
    snapshot.pose.foot_contacts = {
        static_cast<uint8_t>(contacts(0)),
        static_cast<uint8_t>(contacts(1))};
    for (int sample = 0; sample < kFutureRootSampleCount; ++sample) {
        const size_t index = static_cast<size_t>(sample);
        snapshot.future_root_positions[index] = trajectory_positions(sample);
        snapshot.future_root_rotations[index] = trajectory_rotations(sample);
    }
    return snapshot;
}

void write_native_g1_pose(
    const Pose& pose,
    slice1d<vec3> positions,
    slice1d<vec3> velocities,
    slice1d<quat> rotations,
    slice1d<vec3> angular_velocities,
    slice1d<bool> contacts) {
    require_pose_slices(
        positions,
        velocities,
        rotations,
        angular_velocities,
        contacts);
    for (int bone = 0; bone < kNativeG1BoneCount; ++bone) {
        const size_t index = static_cast<size_t>(bone);
        positions(bone) = pose.positions[index];
        velocities(bone) = pose.velocities[index];
        rotations(bone) = pose.rotations[index];
        angular_velocities(bone) = pose.angular_velocities[index];
    }
    contacts(0) = pose.foot_contacts[0] != 0U;
    contacts(1) = pose.foot_contacts[1] != 0U;
}

NativeG1FrameState NativeG1PoseHandoff::apply(
    const Pose& locomotion_pose,
    const RuntimeOutput& runtime_output,
    float dt) {
    if (runtime_output.owns_pose) {
        displayed_ = runtime_output.pose;
        owned_last_tick_ = true;
        release_active_ = false;
        release_seconds_ = 0.0F;
        return {displayed_, true, true};
    }

    if (owned_last_tick_) {
        release_source_ = displayed_;
        owned_last_tick_ = false;
        release_active_ = true;
        release_seconds_ = 0.0F;
        return {displayed_, false, true};
    }

    if (release_active_) {
        release_seconds_ += dt > 0.0F ? dt : 0.0F;
        const float alpha = clampf(
            release_seconds_ / kReleaseDurationSeconds,
            0.0F,
            1.0F);
        if (alpha >= 1.0F) {
            displayed_ = locomotion_pose;
            release_active_ = false;
            return {displayed_, false, false};
        }
        displayed_ = interpolate_pose(
            release_source_, locomotion_pose, alpha);
        return {displayed_, false, true};
    }

    displayed_ = locomotion_pose;
    return {displayed_, false, false};
}

void NativeG1PoseHandoff::reset() {
    owned_last_tick_ = false;
    release_active_ = false;
    release_seconds_ = 0.0F;
    release_source_ = Pose{};
    displayed_ = Pose{};
}

}  // namespace interaction
