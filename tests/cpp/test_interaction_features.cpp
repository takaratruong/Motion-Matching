#include "interaction_features.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>

namespace {

bool near(float left, float right, float tolerance = 1.0e-6F) {
    return std::abs(left - right) <= tolerance;
}

void assert_range_near(
    const interaction::RawQuery& actual,
    size_t start,
    const std::initializer_list<float>& expected) {
    size_t dimension = start;
    for (float value : expected) {
        assert(near(actual.at(dimension), value));
        ++dimension;
    }
}

void test_frozen_layout_and_identity_values() {
    using namespace interaction;

    static_assert(kFeatureDimension == 71U);
    static_assert(kFeatureGroupStarts.size() == 5U);
    static_assert(kFeatureGroupStops.size() == 5U);
    static_assert(kFeatureGroupStarts[0] == 0U);
    static_assert(kFeatureGroupStarts[1] == 33U);
    static_assert(kFeatureGroupStarts[2] == 45U);
    static_assert(kFeatureGroupStarts[3] == 57U);
    static_assert(kFeatureGroupStarts[4] == 65U);
    static_assert(kFeatureGroupStops[0] == 33U);
    static_assert(kFeatureGroupStops[1] == 45U);
    static_assert(kFeatureGroupStops[2] == 57U);
    static_assert(kFeatureGroupStops[3] == 65U);
    static_assert(kFeatureGroupStops[4] == 71U);
    static_assert(kLeftHandBone == 23U);
    static_assert(kRightHandBone == 30U);

    QueryInput input{};
    input.locomotion.future_root_positions = {
        vec3(1.0F, 2.0F, 3.0F),
        vec3(-4.0F, 5.0F, -6.0F),
        vec3(7.0F, 8.0F, 9.0F),
    };
    input.grasp_world = {
        vec3(10.0F, 20.0F, 30.0F),
        quat(),
    };
    input.grasp_linear_velocity = vec3(1.0F, 2.0F, 3.0F);
    input.grasp_angular_velocity = vec3(4.0F, 5.0F, 6.0F);
    input.table_world = {
        vec3(0.0F, 15.0F, 0.0F),
        quat(),
    };
    input.table_size = vec3(8.0F, 4.0F, 6.0F);
    input.approach_direction_object = vec3(0.25F, 0.5F, -0.75F);
    input.object_dimensions = vec3(2.0F, 3.0F, 4.0F);

    const RawQuery raw = build_raw_query(input);
    assert(raw.size() == kFeatureDimension);
    for (size_t dimension = 0; dimension < 33U; ++dimension) {
        assert(near(raw[dimension], 0.0F));
    }
    assert_range_near(
        raw, 33U,
        {1.0F, 3.0F, -4.0F, -6.0F, 7.0F, 9.0F,
         0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F});
    assert_range_near(
        raw, 45U,
        {-10.0F, -20.0F, -30.0F,
         0.0F, 0.0F, 0.0F,
         -1.0F, -2.0F, -3.0F,
         -4.0F, -5.0F, -6.0F});
    assert_range_near(
        raw, 57U,
        {-10.0F, -20.0F, -30.0F,
         0.0F, 1.0F,
         -1.0F, -2.0F, -3.0F});
    assert_range_near(
        raw, 65U,
        {3.0F, 0.25F, -0.75F, 2.0F, 3.0F, 4.0F});
}

void test_active_hand_bone_selection() {
    using namespace interaction;

    QueryInput input{};
    input.locomotion.pose.positions[kLeftHandBone] =
        vec3(1.0F, 2.0F, 3.0F);
    input.locomotion.pose.positions[kRightHandBone] =
        vec3(4.0F, 5.0F, 6.0F);

    input.hand = Hand::Left;
    assert_range_near(
        build_raw_query(input), 45U, {1.0F, 2.0F, 3.0F});

    input.hand = Hand::Right;
    assert_range_near(
        build_raw_query(input), 45U, {4.0F, 5.0F, 6.0F});
}

void test_normalization_uses_serialized_offsets_and_scales() {
    using namespace interaction;

    RawQuery raw{};
    Features features{};
    features.offsets.resize(kFeatureDimension);
    features.scales.resize(kFeatureDimension);
    for (size_t dimension = 0; dimension < kFeatureDimension; ++dimension) {
        raw[dimension] = static_cast<float>(dimension) + 3.0F;
        features.offsets[dimension] = static_cast<float>(dimension) + 1.0F;
        features.scales[dimension] = 2.0F;
    }

    const NormalizedQuery normalized = normalize_query(raw, features);
    for (float value : normalized) {
        assert(near(value, 1.0F));
    }
}

}  // namespace

int main() {
    test_frozen_layout_and_identity_values();
    test_active_hand_bone_selection();
    test_normalization_uses_serialized_offsets_and_scales();
    return 0;
}
