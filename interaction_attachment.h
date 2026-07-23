#pragma once

#include "interaction_matcher.h"

#include <cstdint>
#include <optional>

namespace interaction {

enum class ResultCode : uint8_t {
    None = 0,
    Accepted = 1,
    Succeeded = 2,
    Rejected = 3,
    Cancelled = 4,
    Failed = 5,
    Reset = 6,
};

struct AttachmentConfig {
    float maximum_position_error_m = 0.04F;
    float maximum_orientation_error_radians = 0.261799388F;
    float required_lift_m = 0.15F;
    float required_hold_seconds = 1.00F;
    bool require_lift_for_hold = true;
};

struct ContactMeasurement {
    TargetHandle target{};
    Hand hand = Hand::Right;
    Transform hand_world{};
    float position_error_m = 0.0F;
    float orientation_error_radians = 0.0F;
    bool stable_contact_event = false;
    bool hand_contact = false;
    bool joints_valid = false;
    bool clearance_valid = false;
};

Reason attachment_contact_geometric_reason(
    const ContactMeasurement& measurement,
    Hand expected_hand,
    const Transform& hand_in_object,
    const AttachmentConfig& config);

class AttachmentController {
public:
    AttachmentController(
        TargetRegistry& registry,
        AttachmentConfig config = AttachmentConfig{});

    bool begin(
        const InteractionTarget& target,
        const PickRequest& request,
        const GraspAffordance& affordance,
        float pre_lift_object_height);
    bool try_contact(const ContactMeasurement& measurement);
    void update(const ContactMeasurement& measurement, float dt);
    std::optional<TargetHandle> commit_place(
        Transform placed_world,
        PlacedSupportContext destination_support);
    TargetHandle reset(Transform restored_object_world);

    Transform object_world() const;
    ObjectState state() const;
    ResultCode result() const;
    Reason reason() const;
    float held_seconds() const;

private:
    TargetRegistry* registry_ = nullptr;
    AttachmentConfig config_{};
    InteractionTarget target_{};
    PickRequest request_{};
    GraspAffordance affordance_{};
    Transform object_world_{};
    ObjectState state_ = ObjectState::Free;
    float pre_lift_object_height_ = 0.0F;
    float held_seconds_ = 0.0F;
    ResultCode result_ = ResultCode::None;
    Reason reason_ = Reason::None;
    bool post_attach_failed_ = false;
};

}  // namespace interaction
