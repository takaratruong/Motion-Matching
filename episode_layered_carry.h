#pragma once

#include "interaction_posture_ik.h"

namespace episode {

class LayeredCarry {
public:
    void start(
        const interaction::Pose& final_hold_pose,
        const interaction::Pose& nominal_carry_pose,
        interaction::Hand hand,
        interaction::Transform hand_in_object,
        interaction::Transform object_world);
    const interaction::Pose& update(
        const interaction::LocomotionSnapshot& locomotion);
    const interaction::Pose& pose() const;
    interaction::Transform object_world() const;
    bool last_solve_accepted() const;
    float last_position_error_m() const;
    float last_orientation_error_radians() const;

private:
    interaction::Pose nominal_carry_pose_{};
    interaction::Pose last_safe_pose_{};
    interaction::UpperBodyAngles temporal_seed_{};
    interaction::Transform hand_in_object_{};
    interaction::Transform object_world_{};
    interaction::Hand hand_ = interaction::Hand::Right;
    bool started_ = false;
    bool last_solve_accepted_ = false;
    float last_position_error_m_ = 0.0F;
    float last_orientation_error_radians_ = 0.0F;
};

}  // namespace episode
