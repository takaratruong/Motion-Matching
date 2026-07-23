#pragma once

#include "interaction_pose.h"
#include "interaction_target.h"

namespace episode {

class LayeredCarry {
public:
    void start(
        const interaction::Pose& nominal_return_pose,
        interaction::Hand hand,
        interaction::Transform hand_in_object);
    const interaction::Pose& update(
        const interaction::LocomotionSnapshot& locomotion);
    const interaction::Pose& pose() const;
    interaction::Transform object_world() const;

private:
    interaction::Pose nominal_return_pose_{};
    interaction::Pose pose_{};
    interaction::Transform hand_in_object_{};
    interaction::Transform object_world_{};
    interaction::Hand hand_ = interaction::Hand::Right;
    bool started_ = false;
};

}  // namespace episode
