#include "episode_grasp_provider.h"

#include <cmath>
#include <stdexcept>

namespace episode {
namespace {

bool finite(vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

bool valid(interaction::Transform value) {
    return finite(value.position) &&
           std::isfinite(value.rotation.w) &&
           std::isfinite(value.rotation.x) &&
           std::isfinite(value.rotation.y) &&
           std::isfinite(value.rotation.z) &&
           quat_length(value.rotation) > 1.0e-8F;
}

}  // namespace

KnownGraspProvider::KnownGraspProvider(
    interaction::Transform hand_in_object,
    vec3 approach_in_object)
    : hand_in_object_(hand_in_object),
      approach_in_object_(approach_in_object) {
    if (!valid(hand_in_object_) || !finite(approach_in_object_) ||
        length(approach_in_object_) <= 1.0e-8F) {
        throw std::invalid_argument("invalid known grasp provider");
    }
    hand_in_object_.rotation = quat_normalize(hand_in_object_.rotation);
    approach_in_object_ = normalize(approach_in_object_);
}

std::vector<GraspCandidate> KnownGraspProvider::query(
    const ObjectSnapshot& object) const {
    if (!valid(object.world) || !finite(object.dimensions) ||
        object.dimensions.x <= 0.0F || object.dimensions.y <= 0.0F ||
        object.dimensions.z <= 0.0F) {
        throw std::invalid_argument("invalid known grasp object snapshot");
    }
    return {{
        interaction::compose(object.world, hand_in_object_),
        normalize(quat_mul_vec3(
            object.world.rotation, approach_in_object_)),
        std::nullopt,
        1U,
    }};
}

}  // namespace episode
