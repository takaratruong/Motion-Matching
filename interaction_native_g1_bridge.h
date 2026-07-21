#pragma once

#include <cstdlib>

#include "array.h"
#include "interaction_runtime.h"

namespace interaction {

struct NativeG1FrameState {
    Pose pose{};
    bool runtime_owns_pose = false;
    bool synchronize_simulation_root = false;
};

LocomotionSnapshot capture_native_g1_snapshot(
    slice1d<vec3> positions,
    slice1d<vec3> velocities,
    slice1d<quat> rotations,
    slice1d<vec3> angular_velocities,
    slice1d<bool> contacts,
    slice1d<vec3> trajectory_positions,
    slice1d<quat> trajectory_rotations);

void write_native_g1_pose(
    const Pose& pose,
    slice1d<vec3> positions,
    slice1d<vec3> velocities,
    slice1d<quat> rotations,
    slice1d<vec3> angular_velocities,
    slice1d<bool> contacts);

class NativeG1PoseHandoff {
public:
    NativeG1FrameState apply(
        const Pose& locomotion_pose,
        const RuntimeOutput& runtime_output,
        float dt);
    void reset();

private:
    bool owned_last_tick_ = false;
    bool release_active_ = false;
    float release_seconds_ = 0.0F;
    Pose release_source_{};
    Pose displayed_{};
};

}  // namespace interaction
