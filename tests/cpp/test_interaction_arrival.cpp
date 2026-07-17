#include "interaction_arrival.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {

using interaction::ArrivalConfig;
using interaction::arrival_facing_stick;
using interaction::arrival_navigation_stick;
using interaction::arrival_ready;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_near(
    float actual,
    float expected,
    float tolerance,
    const char* message) {
    uint32_t actual_bits = 0U;
    uint32_t expected_bits = 0U;
    static_assert(sizeof(actual_bits) == sizeof(actual));
    static_assert(sizeof(expected_bits) == sizeof(expected));
    std::memcpy(&actual_bits, &actual, sizeof(actual));
    std::memcpy(&expected_bits, &expected, sizeof(expected));
    constexpr uint32_t exponent_mask = 0x7f800000U;
    if ((actual_bits & exponent_mask) == exponent_mask ||
        (expected_bits & exponent_mask) == exponent_mask) {
        throw std::runtime_error(message);
    }
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

void require_near(
    vec3 actual,
    vec3 expected,
    float tolerance,
    const char* message) {
    require_near(actual.x, expected.x, tolerance, message);
    require_near(actual.y, expected.y, tolerance, message);
    require_near(actual.z, expected.z, tolerance, message);
}

float planar_length(vec3 value) {
    return static_cast<float>(std::sqrt(
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.z) * value.z));
}

float float_from_bits(uint32_t bits) {
    float value = 0.0F;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool is_finite_bits(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(value));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

float& component(vec3& value, size_t index) {
    if (index == 0U) return value.x;
    if (index == 1U) return value.y;
    return value.z;
}

template<class Function>
void require_invalid_argument(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    } catch (...) {
        throw std::runtime_error(message);
    }
    throw std::runtime_error(message);
}

constexpr std::array<float ArrivalConfig::*, 6> kConfigFields{
    &ArrivalConfig::slow_radius_m,
    &ArrivalConfig::latch_position_error_m,
    &ArrivalConfig::latch_simulation_speed_mps,
    &ArrivalConfig::maximum_yaw_error_radians,
    &ArrivalConfig::minimum_standoff_m,
    &ArrivalConfig::maximum_standoff_m,
};

const std::array<float, 4> kRawNonfinite{
    float_from_bits(0x7f800000U),
    float_from_bits(0xff800000U),
    float_from_bits(0x7fc00001U),
    float_from_bits(0x7f800001U),
};

void test_require_near_rejects_nonfinite_operands() {
    for (float invalid : kRawNonfinite) {
        bool rejected_actual = false;
        try {
            require_near(invalid, 0.0F, 0.0F, "nonfinite actual");
        } catch (const std::runtime_error&) {
            rejected_actual = true;
        }
        require(rejected_actual, "require_near accepted a nonfinite actual");

        bool rejected_expected = false;
        try {
            require_near(0.0F, invalid, 0.0F, "nonfinite expected");
        } catch (const std::runtime_error&) {
            rejected_expected = true;
        }
        require(
            rejected_expected,
            "require_near accepted a nonfinite expected");
    }
}

void test_public_contract_and_defaults() {
    using NavigationSignature = vec3 (*)(
        vec3, vec3, float, const ArrivalConfig&);
    using FacingSignature = vec3 (*)(vec3, float);
    using ReadySignature = bool (*)(
        float, float, float, float, const ArrivalConfig&);
    static_assert(std::is_same_v<
        decltype(static_cast<NavigationSignature>(&arrival_navigation_stick)),
        NavigationSignature>);
    static_assert(std::is_same_v<
        decltype(static_cast<FacingSignature>(&arrival_facing_stick)),
        FacingSignature>);
    static_assert(std::is_same_v<
        decltype(static_cast<ReadySignature>(&arrival_ready)),
        ReadySignature>);

    const ArrivalConfig config{};
    require(config.slow_radius_m == 0.60F, "wrong default slow radius");
    require(
        config.latch_position_error_m == 0.03F,
        "wrong default position latch threshold");
    require(
        config.latch_simulation_speed_mps == 0.05F,
        "wrong default speed latch threshold");
    require(
        config.maximum_yaw_error_radians == 20.0F * PIf / 180.0F,
        "wrong default yaw latch threshold");
    require(
        config.minimum_standoff_m == 0.35F,
        "wrong default minimum standoff");
    require(
        config.maximum_standoff_m == 0.45F,
        "wrong default maximum standoff");
}

void test_navigation_exact_proportional_magnitudes() {
    struct Example {
        float distance;
        float expected_magnitude;
    };
    constexpr std::array<Example, 5> examples{{
        {0.00F, 0.00F},
        {0.03F, 0.05F},
        {0.30F, 0.50F},
        {0.60F, 1.00F},
        {1.20F, 1.00F},
    }};
    for (const Example example : examples) {
        const vec3 command = arrival_navigation_stick(
            vec3(example.distance, 9.0F, 0.0F),
            vec3(0.0F, -4.0F, 0.0F),
            0.0F);
        require_near(
            planar_length(command),
            example.expected_magnitude,
            1.0e-6F,
            "navigation proportional magnitude mismatch");
        require_near(
            command,
            vec3(example.expected_magnitude, 0.0F, 0.0F),
            1.0e-6F,
            "navigation proportional direction mismatch");
    }

    const vec3 small = arrival_navigation_stick(
        vec3(0.03F, 0.0F, 0.0F), vec3(), 0.0F);
    require(
        planar_length(small) < 0.15F,
        "navigation retained the old positive 0.15 minimum");

    ArrivalConfig custom{};
    custom.slow_radius_m = 0.30F;
    require_near(
        arrival_navigation_stick(
            vec3(0.15F, 0.0F, 0.0F), vec3(), 0.0F, custom),
        vec3(0.50F, 0.0F, 0.0F),
        1.0e-6F,
        "navigation ignored nondefault slow radius");
}

void test_navigation_tiny_error_is_not_epsilon_biased() {
    constexpr float tiny = 1.0e-6F;
    const float expected = static_cast<float>(
        static_cast<double>(tiny) / static_cast<double>(0.60F));
    const vec3 command = arrival_navigation_stick(
        vec3(tiny, 0.0F, 0.0F), vec3(), 0.0F);
    require(command.x > 0.0F, "tiny nonzero navigation error returned zero");
    require_near(
        command.x,
        expected,
        1.0e-12F,
        "tiny navigation error was epsilon biased");
    require(
        command.y == 0.0F && command.z == 0.0F,
        "tiny x navigation error changed another axis");
}

void test_navigation_translation_direction_and_vertical_invariance() {
    const vec3 translated = arrival_navigation_stick(
        vec3(5.0F, 100.0F, -7.0F),
        vec3(4.0F, -100.0F, -7.0F),
        0.0F);
    require_near(
        translated,
        vec3(1.0F, 0.0F, 0.0F),
        1.0e-6F,
        "navigation was not translation invariant");

    const vec3 diagonal = arrival_navigation_stick(
        vec3(0.30F, 8.0F, 0.40F), vec3(0.0F, -3.0F, 0.0F), 0.0F);
    require_near(
        diagonal,
        vec3(0.50F, 0.0F, 2.0F / 3.0F),
        1.0e-6F,
        "navigation diagonal direction or magnitude was wrong");

    const vec3 flat = arrival_navigation_stick(
        vec3(0.18F, 0.0F, -0.24F), vec3(), 0.0F);
    const vec3 vertical = arrival_navigation_stick(
        vec3(0.18F, 500.0F, -0.24F),
        vec3(0.0F, -500.0F, 0.0F),
        0.0F);
    require_near(
        vertical,
        flat,
        0.0F,
        "navigation depended on vertical components");

    const vec3 zero = arrival_navigation_stick(
        vec3(-0.0F, 20.0F, 0.0F), vec3(0.0F, -20.0F, -0.0F), 0.0F);
    require(
        zero.x == 0.0F && zero.y == 0.0F && zero.z == 0.0F,
        "exact planar zero did not return the zero stick");
}

void test_navigation_inverse_camera_cardinals_and_widened_distance() {
    require_near(
        arrival_navigation_stick(vec3(1.0F, 0.0F, 0.0F), vec3(), 0.0F),
        vec3(1.0F, 0.0F, 0.0F),
        1.0e-6F,
        "camera zero did not map world +X to stick +X");
    require_near(
        arrival_navigation_stick(
            vec3(0.0F, 0.0F, 1.0F), vec3(), 0.5F * PIf),
        vec3(-1.0F, 0.0F, 0.0F),
        1.0e-6F,
        "camera +pi/2 did not map world +Z to stick -X");

    const float large = std::numeric_limits<float>::max();
    const float inverse_sqrt_two = static_cast<float>(1.0 / std::sqrt(2.0));
    const vec3 finite_max_command = arrival_navigation_stick(
        vec3(large, 0.0F, large),
        vec3(-large, 0.0F, -large),
        0.0F);
    require(
        is_finite_bits(finite_max_command.x) &&
            is_finite_bits(finite_max_command.y) &&
            is_finite_bits(finite_max_command.z),
        "finite-max navigation returned a nonfinite component");
    require_near(
        finite_max_command,
        vec3(inverse_sqrt_two, 0.0F, inverse_sqrt_two),
        2.0e-6F,
        "navigation distance arithmetic overflowed for finite inputs");
}

void test_facing_unit_direction_and_camera_mapping() {
    const vec3 plus_x = arrival_facing_stick(
        vec3(4.0F, 20.0F, 0.0F), 0.0F);
    require(
        is_finite_bits(plus_x.x) && is_finite_bits(plus_x.y) &&
            is_finite_bits(plus_x.z),
        "unit +X facing returned a nonfinite component");
    require_near(
        plus_x,
        vec3(1.0F, 0.0F, 0.0F),
        1.0e-6F,
        "facing camera zero did not map world +X to stick +X");
    require_near(
        planar_length(plus_x),
        1.0F,
        1.0e-6F,
        "facing stick was not unit length");

    const vec3 rotated_plus_z = arrival_facing_stick(
        vec3(0.0F, -9.0F, 5.0F), 0.5F * PIf);
    require(
        is_finite_bits(rotated_plus_z.x) &&
            is_finite_bits(rotated_plus_z.y) &&
            is_finite_bits(rotated_plus_z.z),
        "rotated +Z facing cardinal returned a nonfinite component");
    require_near(
        rotated_plus_z,
        vec3(-1.0F, 0.0F, 0.0F),
        1.0e-6F,
        "facing camera +pi/2 did not map world +Z to stick -X");

    const vec3 diagonal = arrival_facing_stick(
        vec3(3.0F, 0.0F, 4.0F), 0.0F);
    require_near(
        diagonal,
        vec3(0.60F, 0.0F, 0.80F),
        1.0e-6F,
        "facing diagonal was not normalized exactly");
    require_near(
        arrival_facing_stick(vec3(3.0F, 1000.0F, 4.0F), 0.0F),
        diagonal,
        0.0F,
        "facing depended on its vertical component");

    const float large = std::numeric_limits<float>::max();
    const float inverse_sqrt_two = static_cast<float>(1.0 / std::sqrt(2.0));
    require_near(
        arrival_facing_stick(vec3(large, 0.0F, large), 0.0F),
        vec3(inverse_sqrt_two, 0.0F, inverse_sqrt_two),
        2.0e-6F,
        "facing length arithmetic overflowed for finite inputs");
}

void test_config_validation_on_every_config_taking_api() {
    for (float invalid : kRawNonfinite) {
        for (float ArrivalConfig::* field : kConfigFields) {
            ArrivalConfig config{};
            config.*field = invalid;
            require_invalid_argument(
                [&] {
                    (void)arrival_navigation_stick(
                        vec3(), vec3(), 0.0F, config);
                },
                "navigation accepted a raw-bit nonfinite config field");
            require_invalid_argument(
                [&] {
                    (void)arrival_ready(
                        0.0F, 0.0F, 0.0F, 0.40F, config);
                },
                "readiness accepted a raw-bit nonfinite config field");
        }
    }

    const std::array<float, 3> nonpositive{
        0.0F,
        float_from_bits(0x80000000U),
        -1.0F,
    };
    for (float invalid : nonpositive) {
        for (float ArrivalConfig::* field : kConfigFields) {
            ArrivalConfig config{};
            config.*field = invalid;
            require_invalid_argument(
                [&] {
                    (void)arrival_navigation_stick(
                        vec3(0.10F, 0.0F, 0.0F), vec3(), 0.0F, config);
                },
                "navigation accepted a nonpositive config field");
            require_invalid_argument(
                [&] {
                    (void)arrival_ready(
                        0.0F, 0.0F, 0.0F, 0.40F, config);
                },
                "readiness accepted a nonpositive config field");
        }
    }

    for (float maximum : {0.35F, std::nextafter(0.35F, 0.0F)}) {
        ArrivalConfig unordered{};
        unordered.maximum_standoff_m = maximum;
        require_invalid_argument(
            [&] {
                (void)arrival_navigation_stick(
                    vec3(0.10F, 0.0F, 0.0F), vec3(), 0.0F, unordered);
            },
            "navigation accepted unordered standoff limits");
        require_invalid_argument(
            [&] {
                (void)arrival_ready(
                    0.0F, 0.0F, 0.0F, 0.40F, unordered);
            },
            "readiness accepted unordered standoff limits");
    }
}

void test_vector_camera_and_metric_raw_bit_validation() {
    for (float invalid : kRawNonfinite) {
        for (size_t index = 0U; index < 3U; ++index) {
            vec3 target(0.10F, 0.20F, 0.30F);
            component(target, index) = invalid;
            require_invalid_argument(
                [&] {
                    (void)arrival_navigation_stick(target, vec3(), 0.0F);
                },
                "navigation accepted a nonfinite target component");

            vec3 current(0.10F, 0.20F, 0.30F);
            component(current, index) = invalid;
            require_invalid_argument(
                [&] {
                    (void)arrival_navigation_stick(
                        vec3(0.20F, 0.30F, 0.40F), current, 0.0F);
                },
                "navigation accepted a nonfinite current component");

            vec3 facing(1.0F, 2.0F, 3.0F);
            component(facing, index) = invalid;
            require_invalid_argument(
                [&] { (void)arrival_facing_stick(facing, 0.0F); },
                "facing accepted a nonfinite vector component");
        }

        require_invalid_argument(
            [&] {
                (void)arrival_navigation_stick(
                    vec3(1.0F, 0.0F, 0.0F), vec3(), invalid);
            },
            "navigation accepted a nonfinite camera azimuth");
        require_invalid_argument(
            [&] {
                (void)arrival_facing_stick(
                    vec3(1.0F, 0.0F, 0.0F), invalid);
            },
            "facing accepted a nonfinite camera azimuth");

        for (size_t metric = 0U; metric < 4U; ++metric) {
            std::array<float, 4> values{0.0F, 0.0F, 0.0F, 0.40F};
            values[metric] = invalid;
            require_invalid_argument(
                [&] {
                    (void)arrival_ready(
                        values[0], values[1], values[2], values[3]);
                },
                "readiness accepted a nonfinite metric");
        }
    }
}

void test_facing_malformed_planar_vectors_reject() {
    require_invalid_argument(
        [] { (void)arrival_facing_stick(vec3(0.0F, 1.0F, 0.0F), 0.0F); },
        "facing accepted an exact-zero planar direction");
    require_invalid_argument(
        [] {
            (void)arrival_facing_stick(
                vec3(float_from_bits(0x80000000U), -7.0F, 0.0F),
                0.0F);
        },
        "facing accepted a signed-zero planar direction");
}

void test_readiness_adjacent_inclusive_boundaries() {
    const ArrivalConfig config{};
    const float positive_infinity = std::numeric_limits<float>::infinity();
    const float negative_infinity = -positive_infinity;

    const auto ready = [&](float position, float speed, float yaw, float gap) {
        return arrival_ready(position, speed, yaw, gap, config);
    };

    const float position_below = std::nextafter(
        config.latch_position_error_m, negative_infinity);
    const float position_above = std::nextafter(
        config.latch_position_error_m, positive_infinity);
    require(ready(position_below, 0.0F, 0.0F, 0.40F),
            "position just below the boundary did not latch");
    require(ready(config.latch_position_error_m, 0.0F, 0.0F, 0.40F),
            "position at the boundary did not latch");
    require(!ready(position_above, 0.0F, 0.0F, 0.40F),
            "position just above the boundary latched");

    const float speed_below = std::nextafter(
        config.latch_simulation_speed_mps, negative_infinity);
    const float speed_above = std::nextafter(
        config.latch_simulation_speed_mps, positive_infinity);
    require(ready(0.0F, speed_below, 0.0F, 0.40F),
            "speed just below the boundary did not latch");
    require(ready(0.0F, config.latch_simulation_speed_mps, 0.0F, 0.40F),
            "speed at the boundary did not latch");
    require(!ready(0.0F, speed_above, 0.0F, 0.40F),
            "speed just above the boundary latched");

    const float yaw_below = std::nextafter(
        config.maximum_yaw_error_radians, negative_infinity);
    const float yaw_above = std::nextafter(
        config.maximum_yaw_error_radians, positive_infinity);
    require(ready(0.0F, 0.0F, yaw_below, 0.40F),
            "yaw just below the boundary did not latch");
    require(ready(0.0F, 0.0F, config.maximum_yaw_error_radians, 0.40F),
            "yaw at the boundary did not latch");
    require(!ready(0.0F, 0.0F, yaw_above, 0.40F),
            "yaw just above the boundary latched");

    const float minimum_below = std::nextafter(
        config.minimum_standoff_m, negative_infinity);
    const float minimum_above = std::nextafter(
        config.minimum_standoff_m, positive_infinity);
    require(!ready(0.0F, 0.0F, 0.0F, minimum_below),
            "standoff just below the minimum latched");
    require(ready(0.0F, 0.0F, 0.0F, config.minimum_standoff_m),
            "standoff at the minimum did not latch");
    require(ready(0.0F, 0.0F, 0.0F, minimum_above),
            "standoff just above the minimum did not latch");

    const float maximum_below = std::nextafter(
        config.maximum_standoff_m, negative_infinity);
    const float maximum_above = std::nextafter(
        config.maximum_standoff_m, positive_infinity);
    require(ready(0.0F, 0.0F, 0.0F, maximum_below),
            "standoff just below the maximum did not latch");
    require(ready(0.0F, 0.0F, 0.0F, config.maximum_standoff_m),
            "standoff at the maximum did not latch");
    require(!ready(0.0F, 0.0F, 0.0F, maximum_above),
            "standoff just above the maximum latched");
}

void test_readiness_metric_validation_and_config_semantics() {
    for (size_t metric = 0U; metric < 4U; ++metric) {
        std::array<float, 4> values{0.0F, 0.0F, 0.0F, 0.40F};
        values[metric] = -1.0F;
        require_invalid_argument(
            [&] {
                (void)arrival_ready(
                    values[0], values[1], values[2], values[3]);
            },
            "readiness accepted a negative metric");
    }

    const float negative_zero = float_from_bits(0x80000000U);
    require(
        arrival_ready(negative_zero, negative_zero, negative_zero, 0.40F),
        "readiness rejected nonnegative signed-zero errors");

    ArrivalConfig custom{};
    custom.slow_radius_m = 1.20F;
    custom.latch_position_error_m = 0.10F;
    custom.latch_simulation_speed_mps = 0.20F;
    custom.maximum_yaw_error_radians = 0.30F;
    custom.minimum_standoff_m = 0.20F;
    custom.maximum_standoff_m = 0.80F;
    require(
        arrival_ready(0.10F, 0.20F, 0.30F, 0.20F, custom),
        "readiness ignored custom inclusive thresholds");
    require(
        arrival_ready(0.10F, 0.20F, 0.30F, 0.80F, custom),
        "readiness ignored custom maximum standoff");
    require(
        !arrival_ready(
            std::nextafter(0.10F, 1.0F), 0.20F, 0.30F, 0.50F, custom),
        "readiness ignored custom position threshold");
    require(
        !arrival_ready(
            0.10F, std::nextafter(0.20F, 1.0F), 0.30F, 0.50F, custom),
        "readiness ignored custom speed threshold");
    require(
        !arrival_ready(
            0.10F, 0.20F, std::nextafter(0.30F, 1.0F), 0.50F, custom),
        "readiness ignored custom yaw threshold");
}

void test_task11_snapshot_commands_correction_and_cannot_latch() {
    const vec3 command = arrival_navigation_stick(
        vec3(0.262F, 0.0F, 0.0F), vec3(), 0.0F);
    require(
        planar_length(command) > 0.0F,
        "Task 11 position snapshot did not command correction");
    require_near(
        planar_length(command),
        0.262F / 0.60F,
        1.0e-6F,
        "Task 11 position snapshot command was not proportional");
    require(
        !arrival_ready(0.262F, 0.67F, 0.0F, 0.40F),
        "Task 11 moving snapshot incorrectly latched");
}

}  // namespace

int main() {
    try {
        test_require_near_rejects_nonfinite_operands();
        test_public_contract_and_defaults();
        test_navigation_exact_proportional_magnitudes();
        test_navigation_tiny_error_is_not_epsilon_biased();
        test_navigation_translation_direction_and_vertical_invariance();
        test_navigation_inverse_camera_cardinals_and_widened_distance();
        test_facing_unit_direction_and_camera_mapping();
        test_config_validation_on_every_config_taking_api();
        test_vector_camera_and_metric_raw_bit_validation();
        test_facing_malformed_planar_vectors_reject();
        test_readiness_adjacent_inclusive_boundaries();
        test_readiness_metric_validation_and_config_semantics();
        test_task11_snapshot_commands_correction_and_cannot_latch();
    } catch (const std::exception& error) {
        std::cerr << "interaction_arrival FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "interaction_arrival tests passed\n";
    return 0;
}
