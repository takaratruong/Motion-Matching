#pragma once

#include "interaction_target.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace episode {

struct ObjectSnapshot {
    uint64_t generation = 0U;
    interaction::Transform world{};
    vec3 dimensions{};
};

struct GraspCandidate {
    interaction::Transform hand_world{};
    vec3 approach_world{1.0F, 0.0F, 0.0F};
    std::optional<interaction::Hand> preferred_hand{};
    uint64_t grasp_id = 0U;
};

class GraspProvider {
public:
    virtual ~GraspProvider() = default;
    virtual std::vector<GraspCandidate> query(
        const ObjectSnapshot& object) const = 0;
};

class KnownGraspProvider final : public GraspProvider {
public:
    KnownGraspProvider(
        interaction::Transform hand_in_object,
        vec3 approach_in_object);
    std::vector<GraspCandidate> query(
        const ObjectSnapshot& object) const override;

private:
    interaction::Transform hand_in_object_{};
    vec3 approach_in_object_{};
};

}  // namespace episode
