#include "interaction_target.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace interaction {
namespace {

constexpr float kMinimumRotationNorm = 1.0e-6F;
constexpr float kRotationNormTolerance = 1.0e-3F;

bool finite(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

bool finite(vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(quat value) {
    return finite(value.w) && finite(value.x) && finite(value.y) &&
           finite(value.z);
}

bool valid_rotation(quat value) {
    if (!finite(value)) return false;
    const float norm = quat_length(value);
    return finite(norm) && norm > kMinimumRotationNorm &&
           std::abs(norm - 1.0F) <= kRotationNormTolerance;
}

void validate_transform(const Transform& transform, const char* label) {
    if (!finite(transform.position) || !valid_rotation(transform.rotation)) {
        throw std::invalid_argument(
            std::string("interaction target invalid ") + label);
    }
}

void validate_positive_size(vec3 value, const char* label) {
    if (!finite(value) || !(value.x > 0.0F) || !(value.y > 0.0F) ||
        !(value.z > 0.0F)) {
        throw std::invalid_argument(
            std::string("interaction target invalid ") + label);
    }
}

bool valid_hand(Hand hand) {
    return static_cast<uint8_t>(hand) <= static_cast<uint8_t>(Hand::Right);
}

void validate_owner(const InteractionTarget& target) {
    switch (target.state) {
        case ObjectState::Free:
            if (target.owner_request != 0U) {
                throw std::invalid_argument(
                    "interaction target free state cannot have an owner");
            }
            return;
        case ObjectState::Targeted:
        case ObjectState::Attached:
        case ObjectState::Held:
            if (target.owner_request == 0U) {
                throw std::invalid_argument(
                    "interaction target owned state requires an owner");
            }
            return;
    }
    throw std::invalid_argument("interaction target state is invalid");
}

void validate_affordance(const GraspAffordance& affordance) {
    if (affordance.id == 0U) {
        throw std::invalid_argument(
            "interaction target affordance ID must be nonzero");
    }
    if (!valid_hand(affordance.hand)) {
        throw std::invalid_argument(
            "interaction target affordance hand is invalid");
    }
    validate_transform(affordance.hand_in_object, "grasp transform");
    const float approach_length = length(
        affordance.approach_direction_object);
    if (!finite(affordance.approach_direction_object) ||
        !finite(approach_length) ||
        !(approach_length > kMinimumRotationNorm)) {
        throw std::invalid_argument(
            "interaction target approach direction must be finite and nonzero");
    }
    if (!finite(affordance.clearance_radius) ||
        affordance.clearance_radius < 0.0F) {
        throw std::invalid_argument(
            "interaction target clearance must be finite and nonnegative");
    }
}

void validate_target(const InteractionTarget& target) {
    if (target.handle.id == 0) {
        throw std::invalid_argument("interaction target ID must be nonzero");
    }
    if (target.handle.generation == 0) {
        throw std::invalid_argument(
            "interaction target generation must be nonzero");
    }
    validate_transform(target.object_world, "object transform");
    validate_positive_size(target.object_dimensions, "object dimensions");
    validate_transform(target.table_world, "table transform");
    validate_positive_size(target.table_size, "table size");
    validate_owner(target);
    if (target.affordances.empty()) {
        throw std::invalid_argument(
            "interaction target must have an affordance");
    }
    for (size_t left = 0; left < target.affordances.size(); ++left) {
        validate_affordance(target.affordances[left]);
        for (size_t right = left + 1U;
             right < target.affordances.size();
             ++right) {
            if (target.affordances[left].id == target.affordances[right].id) {
                throw std::invalid_argument(
                    "interaction target affordance IDs must be unique");
            }
        }
    }
}

TargetHandle incremented(TargetHandle handle) {
    if (handle.generation == std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("interaction target generation overflow");
    }
    ++handle.generation;
    return handle;
}

InteractionTarget& target_with_id(
    std::vector<InteractionTarget>& targets,
    uint64_t id) {
    for (InteractionTarget& target : targets) {
        if (target.handle.id == id) {
            return target;
        }
    }
    throw std::out_of_range("interaction target ID is not registered");
}

}  // namespace

TargetHandle TargetRegistry::upsert(InteractionTarget target) {
    validate_target(target);
    for (InteractionTarget& stored : targets_) {
        if (stored.handle.id != target.handle.id) {
            continue;
        }
        target.handle = incremented(stored.handle);
        target.state = ObjectState::Free;
        target.owner_request = 0;
        stored = std::move(target);
        return stored.handle;
    }

    targets_.push_back(std::move(target));
    return targets_.back().handle;
}

const InteractionTarget* TargetRegistry::find(TargetHandle handle) const {
    for (const InteractionTarget& target : targets_) {
        if (target.handle == handle) {
            return &target;
        }
    }
    return nullptr;
}

const InteractionTarget* TargetRegistry::find_by_id(uint64_t id) const {
    for (const InteractionTarget& target : targets_) {
        if (target.handle.id == id) return &target;
    }
    return nullptr;
}

InteractionTarget* TargetRegistry::find(TargetHandle handle) {
    for (InteractionTarget& target : targets_) {
        if (target.handle == handle) {
            return &target;
        }
    }
    return nullptr;
}

std::optional<TargetHandle> TargetRegistry::resolve_single_target(
    vec3 character_root,
    float maximum_distance) const {
    if (!(maximum_distance >= 0.0F)) {
        return std::nullopt;
    }

    std::optional<TargetHandle> resolved;
    for (const InteractionTarget& target : targets_) {
        if (target.state != ObjectState::Free) {
            continue;
        }
        const float delta_x =
            target.object_world.position.x - character_root.x;
        const float delta_z =
            target.object_world.position.z - character_root.z;
        if (std::hypot(delta_x, delta_z) > maximum_distance) {
            continue;
        }
        if (resolved.has_value()) {
            return std::nullopt;
        }
        resolved = target.handle;
    }
    return resolved;
}

bool TargetRegistry::reserve(TargetHandle handle, uint64_t request_id) {
    InteractionTarget* target = find(handle);
    if (request_id == 0 || target == nullptr ||
        target->state != ObjectState::Free) {
        return false;
    }
    target->state = ObjectState::Targeted;
    target->owner_request = request_id;
    return true;
}

bool TargetRegistry::validate(
    TargetHandle handle,
    uint64_t request_id) const {
    const InteractionTarget* target = find(handle);
    return request_id != 0 && target != nullptr &&
           target->state != ObjectState::Free &&
           target->owner_request == request_id;
}

bool TargetRegistry::attach(TargetHandle handle, uint64_t request_id) {
    InteractionTarget* target = find(handle);
    if (request_id == 0 || target == nullptr ||
        target->state != ObjectState::Targeted ||
        target->owner_request != request_id) {
        return false;
    }
    target->state = ObjectState::Attached;
    return true;
}

bool TargetRegistry::hold(TargetHandle handle, uint64_t request_id) {
    InteractionTarget* target = find(handle);
    if (request_id == 0 || target == nullptr ||
        target->state != ObjectState::Attached ||
        target->owner_request != request_id) {
        return false;
    }
    target->state = ObjectState::Held;
    return true;
}

bool TargetRegistry::release(TargetHandle handle, uint64_t request_id) {
    InteractionTarget* target = find(handle);
    if (request_id == 0 || target == nullptr ||
        target->state == ObjectState::Free ||
        target->owner_request != request_id) {
        return false;
    }
    target->state = ObjectState::Free;
    target->owner_request = 0;
    return true;
}

TargetHandle TargetRegistry::replace_pose(
    uint64_t id,
    Transform object_world) {
    validate_transform(object_world, "replacement object transform");
    InteractionTarget& target = target_with_id(targets_, id);
    const TargetHandle replacement = incremented(target.handle);
    target.handle = replacement;
    target.object_world = object_world;
    target.state = ObjectState::Free;
    target.owner_request = 0;
    return replacement;
}

TargetHandle TargetRegistry::reset(uint64_t id, Transform object_world) {
    return replace_pose(id, object_world);
}

const GraspAffordance* TargetRegistry::find_affordance(
    TargetHandle handle,
    uint32_t affordance_id) const {
    const InteractionTarget* target = find(handle);
    if (target == nullptr) {
        return nullptr;
    }
    for (const GraspAffordance& affordance : target->affordances) {
        if (affordance.id == affordance_id) {
            return &affordance;
        }
    }
    return nullptr;
}

}  // namespace interaction
