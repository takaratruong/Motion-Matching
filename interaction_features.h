#pragma once

#include "interaction_database.h"
#include "interaction_pose.h"
#include "interaction_target.h"

#include <array>
#include <cstddef>

namespace interaction {

inline constexpr size_t kFeatureDimension = 71U;
inline constexpr std::array<size_t, 5> kFeatureGroupStarts = {
    0U, 33U, 45U, 57U, 65U};
inline constexpr std::array<size_t, 5> kFeatureGroupStops = {
    33U, 45U, 57U, 65U, 71U};
inline constexpr size_t kLeftHandBone = 23U;
inline constexpr size_t kRightHandBone = 30U;

struct QueryInput {
    LocomotionSnapshot locomotion;
    Transform grasp_world;
    vec3 grasp_linear_velocity{};
    vec3 grasp_angular_velocity{};
    Transform table_world;
    vec3 table_size{};
    vec3 approach_direction_object{};
    vec3 object_dimensions{};
    Hand hand = Hand::Right;
};

using RawQuery = std::array<float, kFeatureDimension>;
using NormalizedQuery = std::array<float, kFeatureDimension>;

RawQuery build_raw_query(const QueryInput& input);
NormalizedQuery normalize_query(const RawQuery& raw, const Features& features);

}  // namespace interaction
