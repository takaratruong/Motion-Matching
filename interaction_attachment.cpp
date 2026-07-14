#include "interaction_attachment.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace interaction {
namespace {

bool finite_nonnegative(float value) {
    return std::isfinite(value) && value >= 0.0F;
}

bool exact(vec3 left, vec3 right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool exact(quat left, quat right) {
    return left.w == right.w && left.x == right.x &&
           left.y == right.y && left.z == right.z;
}

bool exact(const Transform& left, const Transform& right) {
    return exact(left.position, right.position) &&
           exact(left.rotation, right.rotation);
}

bool valid_position(vec3 position) {
    return std::isfinite(position.x) && std::isfinite(position.y) &&
           std::isfinite(position.z);
}

bool valid_rotation(quat rotation) {
    if (!std::isfinite(rotation.w) || !std::isfinite(rotation.x) ||
        !std::isfinite(rotation.y) || !std::isfinite(rotation.z)) {
        return false;
    }
    const double squared_length =
        static_cast<double>(rotation.w) * rotation.w +
        static_cast<double>(rotation.x) * rotation.x +
        static_cast<double>(rotation.y) * rotation.y +
        static_cast<double>(rotation.z) * rotation.z;
    return squared_length > 0.0 && std::isfinite(squared_length);
}

bool valid_transform(const Transform& transform) {
    return valid_position(transform.position) &&
           valid_rotation(transform.rotation);
}

bool exact(const GraspAffordance& left, const GraspAffordance& right) {
    return left.id == right.id && left.hand == right.hand &&
           exact(left.hand_in_object, right.hand_in_object) &&
           exact(
               left.approach_direction_object,
               right.approach_direction_object) &&
           left.clearance_radius == right.clearance_radius;
}

bool exact_target_core(
    const InteractionTarget& left,
    const InteractionTarget& right) {
    return left.handle == right.handle &&
           exact(left.object_world, right.object_world) &&
           exact(left.object_dimensions, right.object_dimensions) &&
           exact(left.table_world, right.table_world) &&
           exact(left.table_size, right.table_size) &&
           left.state == right.state &&
           left.owner_request == right.owner_request;
}

bool exact_affordances(
    const std::vector<GraspAffordance>& left,
    const std::vector<GraspAffordance>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (size_t index = 0; index < left.size(); ++index) {
        if (!exact(left[index], right[index])) {
            return false;
        }
    }
    return true;
}

}  // namespace

AttachmentController::AttachmentController(
    TargetRegistry& registry,
    AttachmentConfig config)
    : registry_(&registry), config_(config) {
    if (!finite_nonnegative(config_.maximum_position_error_m) ||
        !finite_nonnegative(config_.maximum_orientation_error_radians) ||
        !finite_nonnegative(config_.required_lift_m) ||
        !finite_nonnegative(config_.required_hold_seconds)) {
        throw std::invalid_argument(
            "attachment configuration values must be finite and nonnegative");
    }
}

bool AttachmentController::begin(
    const InteractionTarget& target,
    const PickRequest& request,
    const GraspAffordance& affordance,
    float pre_lift_object_height) {
    if (!std::isfinite(pre_lift_object_height)) {
        throw std::invalid_argument(
            "pre-lift object height must be finite");
    }

    const InteractionTarget* stored = registry_->find(request.target);
    if (state_ != ObjectState::Free || target.handle != request.target ||
        stored == nullptr || !registry_->validate(
            request.target, request.request_id) ||
        stored->state != ObjectState::Targeted ||
        stored->owner_request != request.request_id ||
        !exact_target_core(target, *stored)) {
        result_ = ResultCode::Rejected;
        reason_ = Reason::TargetChanged;
        return false;
    }

    const GraspAffordance* authored = registry_->find_affordance(
        request.target, request.affordance_id);
    if (authored == nullptr || affordance.id != request.affordance_id ||
        !exact(affordance, *authored) ||
        !exact_affordances(target.affordances, stored->affordances) ||
        !valid_transform(stored->object_world) ||
        !valid_transform(authored->hand_in_object)) {
        result_ = ResultCode::Rejected;
        reason_ = Reason::TargetUnavailable;
        return false;
    }

    target_ = *stored;
    request_ = request;
    affordance_ = *authored;
    object_world_ = stored->object_world;
    state_ = ObjectState::Targeted;
    pre_lift_object_height_ = pre_lift_object_height;
    held_seconds_ = 0.0F;
    result_ = ResultCode::Accepted;
    reason_ = Reason::None;
    post_attach_failed_ = false;
    return true;
}

bool AttachmentController::try_contact(
    const ContactMeasurement& measurement) {
    if (state_ != ObjectState::Targeted) {
        return false;
    }

    auto fail = [this](Reason reason) {
        result_ = ResultCode::Failed;
        reason_ = reason;
        held_seconds_ = 0.0F;
        return false;
    };

    const InteractionTarget* stored = registry_->find(request_.target);
    if (measurement.target != request_.target || stored == nullptr ||
        !registry_->validate(request_.target, request_.request_id) ||
        stored->state != ObjectState::Targeted ||
        stored->owner_request != request_.request_id) {
        return fail(Reason::TargetChanged);
    }
    if (!measurement.stable_contact_event ||
        measurement.hand != affordance_.hand ||
        !measurement.hand_contact) {
        return fail(Reason::LostContact);
    }
    if (!valid_position(measurement.hand_world.position) ||
        !finite_nonnegative(measurement.position_error_m) ||
        measurement.position_error_m > config_.maximum_position_error_m) {
        return fail(Reason::ContactPosition);
    }
    if (!valid_rotation(measurement.hand_world.rotation) ||
        !finite_nonnegative(measurement.orientation_error_radians) ||
        measurement.orientation_error_radians >
            config_.maximum_orientation_error_radians) {
        return fail(Reason::ContactOrientation);
    }
    if (!measurement.joints_valid) {
        return fail(Reason::JointLimit);
    }
    if (!measurement.clearance_valid) {
        return fail(Reason::BlockedPath);
    }

    const Transform attached_object_world = compose(
        measurement.hand_world, inverse(affordance_.hand_in_object));
    if (!valid_transform(attached_object_world)) {
        return fail(Reason::ContactOrientation);
    }
    if (!registry_->attach(request_.target, request_.request_id)) {
        return fail(Reason::TargetChanged);
    }

    object_world_ = attached_object_world;
    target_.state = ObjectState::Attached;
    state_ = ObjectState::Attached;
    result_ = ResultCode::Accepted;
    reason_ = Reason::None;
    return true;
}

void AttachmentController::update(
    const ContactMeasurement& measurement,
    float dt) {
    if (!finite_nonnegative(dt)) {
        throw std::invalid_argument(
            "attachment update dt must be finite and nonnegative");
    }
    if ((state_ != ObjectState::Attached && state_ != ObjectState::Held) ||
        post_attach_failed_) {
        return;
    }

    auto fail = [this](Reason reason) {
        if (state_ == ObjectState::Attached) {
            held_seconds_ = 0.0F;
        }
        result_ = ResultCode::Failed;
        reason_ = reason;
        post_attach_failed_ = true;
    };

    const InteractionTarget* stored = registry_->find(request_.target);
    if (measurement.target != request_.target || stored == nullptr ||
        !registry_->validate(request_.target, request_.request_id) ||
        stored->state != state_ ||
        stored->owner_request != request_.request_id) {
        fail(Reason::TargetChanged);
        return;
    }
    if (measurement.hand != affordance_.hand ||
        !measurement.hand_contact) {
        fail(Reason::LostContact);
        return;
    }
    if (!valid_position(measurement.hand_world.position)) {
        fail(Reason::ContactPosition);
        return;
    }
    if (!valid_rotation(measurement.hand_world.rotation)) {
        fail(Reason::ContactOrientation);
        return;
    }

    const Transform next_object_world = compose(
        measurement.hand_world, inverse(affordance_.hand_in_object));
    if (!valid_transform(next_object_world)) {
        fail(Reason::ContactOrientation);
        return;
    }

    if (state_ == ObjectState::Held) {
        object_world_ = next_object_world;
        return;
    }

    float next_held_seconds = 0.0F;
    const float lift_threshold =
        pre_lift_object_height_ + config_.required_lift_m;
    if (next_object_world.position.y >= lift_threshold) {
        const double accumulated =
            static_cast<double>(held_seconds_) + static_cast<double>(dt);
        next_held_seconds = static_cast<float>(std::min(
            accumulated,
            static_cast<double>(config_.required_hold_seconds)));
    }

    if (next_held_seconds >= config_.required_hold_seconds &&
        next_object_world.position.y >= lift_threshold) {
        if (!registry_->hold(request_.target, request_.request_id)) {
            fail(Reason::TargetChanged);
            return;
        }
        state_ = ObjectState::Held;
        target_.state = ObjectState::Held;
        result_ = ResultCode::Succeeded;
        reason_ = Reason::None;
    } else {
        result_ = ResultCode::Accepted;
        reason_ = Reason::None;
    }

    object_world_ = next_object_world;
    held_seconds_ = next_held_seconds;
}

TargetHandle AttachmentController::reset(Transform restored_object_world) {
    if (!valid_transform(restored_object_world)) {
        throw std::invalid_argument(
            "restored object transform must be finite and valid");
    }
    if (request_.target.id == 0) {
        throw std::logic_error("attachment controller has no target to reset");
    }

    const TargetHandle reset_handle = registry_->reset(
        request_.target.id, restored_object_world);
    target_ = InteractionTarget{};
    request_ = PickRequest{};
    affordance_ = GraspAffordance{};
    object_world_ = restored_object_world;
    state_ = ObjectState::Free;
    pre_lift_object_height_ = 0.0F;
    held_seconds_ = 0.0F;
    result_ = ResultCode::Reset;
    reason_ = Reason::Reset;
    post_attach_failed_ = false;
    return reset_handle;
}

Transform AttachmentController::object_world() const {
    return object_world_;
}

ObjectState AttachmentController::state() const {
    return state_;
}

ResultCode AttachmentController::result() const {
    return result_;
}

Reason AttachmentController::reason() const {
    return reason_;
}

float AttachmentController::held_seconds() const {
    return held_seconds_;
}

}  // namespace interaction
