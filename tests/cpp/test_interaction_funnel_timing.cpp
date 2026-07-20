#include "interaction_funnel_timing.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>

namespace {

constexpr double kPi = 3.14159265358979323846;

interaction::FunnelSample sample(float x, float z, double yaw_degrees) {
    const double yaw = yaw_degrees * kPi / 180.0;
    return {
        x,
        z,
        static_cast<float>(std::sin(yaw)),
        static_cast<float>(std::cos(yaw)),
    };
}

void test_expands_all_frozen_offsets_exactly() {
    std::array<interaction::FunnelSample, interaction::kFunnelKnotCount> knots{};
    for (int index = 0; index < interaction::kFunnelKnotCount; ++index) {
        knots[static_cast<size_t>(index)] = sample(
            static_cast<float>(index),
            static_cast<float>(-2 * index),
            static_cast<double>(index * 7));
    }

    const interaction::FunnelExecutionTargets targets =
        interaction::expand_funnel_execution(knots);
    static_assert(targets.size() == 75U);
    for (int index = 0; index < interaction::kFunnelKnotCount; ++index) {
        const auto& expected = knots[static_cast<size_t>(index)];
        const auto& actual = targets[static_cast<size_t>(
            interaction::kFunnelKnotTickOffsets[static_cast<size_t>(index)])];
        assert(actual.x == expected.x);
        assert(actual.z == expected.z);
        assert(actual.yaw_sin == expected.yaw_sin);
        assert(actual.yaw_cos == expected.yaw_cos);
    }
    assert(targets.front().x == knots.front().x);
    assert(targets.back().x == knots.back().x);
}

void test_linearly_interpolates_position() {
    std::array<interaction::FunnelSample, interaction::kFunnelKnotCount> knots{};
    for (auto& knot : knots) knot = sample(0.0F, 0.0F, 0.0);
    knots[0] = sample(0.0F, 2.0F, 0.0);
    knots[1] = sample(10.0F, 12.0F, 0.0);

    const auto targets = interaction::expand_funnel_execution(knots);
    assert(targets[2].x == 4.0F);
    assert(targets[2].z == 6.0F);
}

void test_yaw_uses_unit_shortest_arc() {
    std::array<interaction::FunnelSample, interaction::kFunnelKnotCount> knots{};
    for (auto& knot : knots) knot = sample(0.0F, 0.0F, -179.0);
    knots[0] = sample(0.0F, 0.0F, 179.0);
    knots[1] = sample(0.0F, 0.0F, -179.0);

    const auto targets = interaction::expand_funnel_execution(knots);
    for (const auto& target : targets) {
        const float norm = std::sqrt(
            target.yaw_sin * target.yaw_sin +
            target.yaw_cos * target.yaw_cos);
        assert(std::abs(norm - 1.0F) < 2.0e-6F);
    }
    const double middle_yaw = std::atan2(targets[2].yaw_sin, targets[2].yaw_cos);
    assert(std::abs(std::abs(middle_yaw) - kPi) < 0.02);
}

}  // namespace

int main() {
    test_expands_all_frozen_offsets_exactly();
    test_linearly_interpolates_position();
    test_yaw_uses_unit_shortest_arc();
    return 0;
}
