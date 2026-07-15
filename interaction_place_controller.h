#pragma once

#include "interaction_place.h"
#include "interaction_rotation_gate.h"

#include <cstdint>

namespace interaction {

struct PlaceControllerConfig {
    PlaceTimingConfig timing{};
    PlaceMatchConfig match{};
    float release_position_m = 0.02F;
    float release_orientation_radians = 0.174532925F;
};

struct PlaceStep {
    Pose pose{};
    Transform object_world{};
    PlacePhase phase = PlacePhase::Align;
    int32_t source_frame = -1;
    double source_frame_exact = -1.0;
    bool committed = false;
    bool release_due = false;
    bool retract_finished = false;
    bool recover_to_carry = false;
    Reason reason = Reason::None;
    float hand_position_error_m = 0.0F;
    float hand_orientation_error_radians = 0.0F;
    float requested_root_correction_m = 0.0F;
    float applied_root_correction_m = 0.0F;
    float requested_yaw_correction_radians = 0.0F;
    float applied_yaw_correction_radians = 0.0F;
    float requested_hand_correction_m = 0.0F;
    float applied_hand_correction_m = 0.0F;
    float requested_hand_orientation_radians = 0.0F;
    float applied_hand_orientation_radians = 0.0F;
    PlacementFit actual_fit{};
    bool support_sweep_clear = false;
};

struct PlaceBeginInput {
    PlaceMatchInput match_input{};
    PlaceCandidate candidate{};
};

struct PlaceBeginResult {
    bool accepted = false;
    Reason reason = Reason::None;
};

class PlaceController {
public:
    PlaceController(PlaceControllerConfig config, IKConfig ik_config);
    PlaceBeginResult begin(const PlaceBeginInput& input);
    PlaceStep update(float dt);
    void acknowledge_release(Transform placed_world);
    PlaceStep cancel();

private:
    PlaceControllerConfig config_{};
    IKConfig ik_config_{};
    PlacePlayer player_{};
    PlaceBeginInput begin_{};
    Transform goal_object_{};
    Transform goal_hand_{};
    rotation_gate::Rotation goal_object_rotation_evidence_{};
    rotation_gate::Rotation goal_hand_rotation_evidence_{};
    vec3 release_hand_translation_{};
    quat release_hand_rotation_{};
    Pose last_safe_pose_{};
    Transform last_safe_object_{};
    Transform previous_object_{};
    Transform frozen_object_{};
    PlaceStep last_step_{};
    int32_t entry_blend_ticks_ = 0;
    int32_t output_ticks_ = 0;
    bool started_ = false;
    bool recovering_ = false;
    bool release_pending_ = false;
    bool release_pulse_emitted_ = false;
    bool release_acknowledged_ = false;
};

}  // namespace interaction
