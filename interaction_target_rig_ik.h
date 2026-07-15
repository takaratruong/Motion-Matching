#pragma once

#include "interaction_target.h"

namespace interaction {

struct FlatControllerPose;

struct TargetRigArmIKConfig {
    float reach_epsilon_m = 1.0e-4F;
    float pole_epsilon_m = 1.0e-5F;
};

struct TargetRigArmIKResult {
    bool applied = false;
    bool reachable = false;
    bool used_clavicle = false;
    float position_error_m = 0.0F;
    float orientation_error_radians = 0.0F;
    float reach_shortfall_m = 0.0F;
};

class TargetRigArmIK {
public:
    explicit TargetRigArmIK(TargetRigArmIKConfig config = {});

    void begin_epoch(
        const Pose& interaction_reference,
        const FlatControllerPose& flat_reference,
        Hand hand);
    TargetRigArmIKResult solve(
        FlatControllerPose& pose,
        Transform grasp_world,
        float weight,
        float dt);
    void reset();

    bool active() const;
    quat calibration_rotation() const;

private:
    TargetRigArmIKConfig config_{};
    bool active_ = false;
    Hand hand_ = Hand::Right;
    quat calibration_rotation_{};
    bool has_previous_pole_ = false;
    vec3 previous_pole_spine_{};
};

}  // namespace interaction
