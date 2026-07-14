#pragma once

#include "interaction_pose.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace interaction {

enum class Hand : uint8_t { Left = 0, Right = 1 };
enum class ObjectState : uint8_t { Free, Targeted, Attached, Held };

struct TargetHandle {
    uint64_t id = 0;
    uint32_t generation = 0;
    friend constexpr bool operator==(TargetHandle left, TargetHandle right) {
        return left.id == right.id && left.generation == right.generation;
    }
    friend constexpr bool operator!=(TargetHandle left, TargetHandle right) {
        return !(left == right);
    }
};

struct GraspAffordance {
    uint32_t id = 0;
    Hand hand = Hand::Right;
    Transform hand_in_object{};
    vec3 approach_direction_object{};
    float clearance_radius = 0.04F;
};

struct InteractionTarget {
    TargetHandle handle{};
    Transform object_world{};
    vec3 object_dimensions{};
    Transform table_world{};
    vec3 table_size{};
    ObjectState state = ObjectState::Free;
    uint64_t owner_request = 0;
    std::vector<GraspAffordance> affordances;
};

struct PickRequest {
    TargetHandle target{};
    uint32_t affordance_id = 0;
    uint64_t request_id = 0;
};

class TargetRegistry {
public:
    TargetHandle upsert(InteractionTarget target);
    const InteractionTarget* find(TargetHandle handle) const;
    InteractionTarget* find(TargetHandle handle);
    std::optional<TargetHandle> resolve_single_target(
        vec3 character_root, float maximum_distance) const;
    bool reserve(TargetHandle handle, uint64_t request_id);
    bool validate(TargetHandle handle, uint64_t request_id) const;
    bool attach(TargetHandle handle, uint64_t request_id);
    bool hold(TargetHandle handle, uint64_t request_id);
    bool release(TargetHandle handle, uint64_t request_id);
    TargetHandle replace_pose(uint64_t id, Transform object_world);
    TargetHandle reset(uint64_t id, Transform object_world);
    const GraspAffordance* find_affordance(
        TargetHandle handle, uint32_t affordance_id) const;
private:
    std::vector<InteractionTarget> targets_;
};

}  // namespace interaction
