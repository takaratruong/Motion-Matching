#include "g1_command_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "G1 command runtime test failed: %s\n", message);
        std::exit(1);
    }
}

static uint32_t bits(float value)
{
    uint32_t output = 0;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

static bool same_vec3_bits(vec3 left, vec3 right)
{
    return bits(left.x) == bits(right.x) &&
           bits(left.y) == bits(right.y) &&
           bits(left.z) == bits(right.z);
}

static bool same_quat_bits(quat left, quat right)
{
    return bits(left.w) == bits(right.w) &&
           bits(left.x) == bits(right.x) &&
           bits(left.y) == bits(right.y) &&
           bits(left.z) == bits(right.z);
}

static void poison_snapshot(G1CommandSnapshot& value)
{
    unsigned char* const bytes =
        reinterpret_cast<unsigned char*>(&value);
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        bytes[index] = static_cast<unsigned char>(0xa5U + index % 17U);
    }
}

static bool same_snapshot_bytes(
    const G1CommandSnapshot& left,
    const G1CommandSnapshot& right)
{
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

struct CommandFixture
{
    G1CommandIntent intent;
    vec3 applied;
    vec3 desired_velocities[G1CommandTrajectorySampleCount];
    vec3 root_positions[G1CommandTrajectorySampleCount];
    quat root_rotations[G1CommandTrajectorySampleCount];
    quat desired_headings[G1CommandTrajectorySampleCount];

    CommandFixture()
    {
        intent.requested_velocity = vec3(0.25f, -0.0f, -0.75f);
        intent.desired_heading =
            quat(0.923879504f, 0.0f, 0.382683426f, 0.0f);
        applied = vec3(0.125f, 0.0f, -0.375f);
        desired_velocities[0] = vec3(-0.0f, 0.0f, 0.25f);
        desired_velocities[1] = vec3(0.10f, -0.0f, 0.20f);
        desired_velocities[2] = vec3(0.20f, 0.0f, 0.30f);
        desired_velocities[3] = vec3(0.30f, 0.0f, 0.40f);
        root_positions[0] = vec3(1.0f, -0.0f, 2.0f);
        root_positions[1] = vec3(1.1f, 0.0f, 2.2f);
        root_positions[2] = vec3(1.2f, 0.0f, 2.4f);
        root_positions[3] = vec3(1.3f, 0.0f, 2.6f);
        root_rotations[0] = quat(1.0f, 0.0f, 0.0f, 0.0f);
        root_rotations[1] = quat(0.0f, 0.0f, 1.0f, 0.0f);
        root_rotations[2] =
            quat(0.707106769f, 0.0f, 0.707106769f, 0.0f);
        root_rotations[3] =
            quat(0.707106769f, 0.0f, -0.707106769f, 0.0f);
        desired_headings[0] = intent.desired_heading;
        desired_headings[1] = intent.desired_heading;
        desired_headings[2] = intent.desired_heading;
        desired_headings[3] = intent.desired_heading;
    }

    slice1d<vec3> velocities(int count = G1CommandTrajectorySampleCount)
    {
        return slice1d<vec3>(count, desired_velocities);
    }

    slice1d<vec3> positions(int count = G1CommandTrajectorySampleCount)
    {
        return slice1d<vec3>(count, root_positions);
    }

    slice1d<quat> rotations(int count = G1CommandTrajectorySampleCount)
    {
        return slice1d<quat>(count, root_rotations);
    }

    slice1d<quat> headings(int count = G1CommandTrajectorySampleCount)
    {
        return slice1d<quat>(count, desired_headings);
    }
};

static bool build(
    G1CommandSnapshot& output,
    CommandFixture& fixture,
    char* error,
    int capacity)
{
    return g1_command_snapshot_build(
        output,
        fixture.intent,
        fixture.applied,
        fixture.velocities(),
        fixture.positions(),
        fixture.rotations(),
        fixture.headings(),
        error,
        capacity);
}

static void require_failed_build_preserves(
    G1CommandSnapshot& output,
    const G1CommandSnapshot& before,
    bool result,
    const char* message)
{
    check(!result, message);
    check(same_snapshot_bytes(output, before), message);
}

static void test_exact_heading_parser()
{
    static const struct
    {
        const char* text;
        quat expected;
    } headings[] = {
        {"forward", quat(1.0f, 0.0f, 0.0f, 0.0f)},
        {"backward", quat(0.0f, 0.0f, 1.0f, 0.0f)},
        {"positive-x", quat(0.707106769f, 0.0f, 0.707106769f, 0.0f)},
        {"negative-x", quat(0.707106769f, 0.0f, -0.707106769f, 0.0f)},
        {"diagonal-positive-x",
         quat(0.923879504f, 0.0f, 0.382683426f, 0.0f)},
        {"diagonal-negative-x",
         quat(0.923879504f, 0.0f, -0.382683426f, 0.0f)},
        {"backward-positive-x",
         quat(0.382683426f, 0.0f, 0.923879504f, 0.0f)},
        {"backward-negative-x",
         quat(0.382683426f, 0.0f, -0.923879504f, 0.0f)},
    };

    char error[256] = {};
    for (const auto& heading : headings) {
        G1TestHeadingOverride output;
        output.active = false;
        output.heading = quat(9.0f, 8.0f, 7.0f, 6.0f);
        check(g1_test_heading_override_parse(
                  output, heading.text, error, sizeof(error)),
              error);
        check(output.active, "exact heading activates override");
        check(same_quat_bits(output.heading, heading.expected),
              "exact heading preserves specified quaternion bits");
    }

    G1TestHeadingOverride inactive;
    inactive.active = true;
    inactive.heading = quat(9.0f, 8.0f, 7.0f, 6.0f);
    check(g1_test_heading_override_parse(
              inactive, NULL, error, sizeof(error)),
          error);
    check(!inactive.active &&
              same_quat_bits(inactive.heading, quat(1.0f, 0.0f, 0.0f, 0.0f)),
          "null heading is a deterministic inactive identity override");

    const char* hostile[] = {
        "", " ", "\t", "forward ", " forward", "forward\n",
        "Forward", "FORWARD", "positive-X", "forward-suffix",
        "backward0", "0", "1", "-1", "+1", "0.0", "-90",
        "+90.0", "1e2", "-1.25e-3", "nan", "NaN", "inf", "+inf"
    };
    for (const char* text : hostile) {
        G1TestHeadingOverride output;
        output.active = true;
        output.heading = quat(9.0f, 8.0f, 7.0f, 6.0f);
        const G1TestHeadingOverride before = output;
        check(!g1_test_heading_override_parse(
                  output, text, error, sizeof(error)),
              "hostile heading spelling is rejected");
        check(output.active == before.active &&
                  same_quat_bits(output.heading, before.heading),
              "heading parse failure is transactional");
    }
}

static void test_snapshot_build_and_heading_independence()
{
    CommandFixture fixture;
    char error[256] = {};
    G1CommandSnapshot first;
    poison_snapshot(first);
    check(build(first, fixture, error, sizeof(error)), error);
    check(g1_command_snapshot_is_valid(first),
          "published command snapshot is valid");
    check(same_vec3_bits(first.intent.requested_velocity,
                         vec3(0.25f, 0.0f, -0.75f)),
          "requested travel is owned and signed zero is canonical");
    check(same_vec3_bits(first.applied_velocity, fixture.applied),
          "applied travel is owned separately");
    check(bits(first.predicted_desired_velocities[0].x) == 0 &&
              bits(first.predicted_root_positions[0].y) == 0,
          "trajectory vector signed zeros are canonical positive zero");

    CommandFixture limited = fixture;
    limited.applied = vec3(0.03125f, 0.0f, -0.09375f);
    G1CommandSnapshot second;
    poison_snapshot(second);
    check(build(second, limited, error, sizeof(error)), error);
    check(!same_vec3_bits(first.applied_velocity, second.applied_velocity),
          "fixture changes only applied travel");
    check(same_vec3_bits(first.intent.requested_velocity,
                         second.intent.requested_velocity) &&
              same_quat_bits(first.intent.desired_heading,
                             second.intent.desired_heading),
          "terrain-limited velocity cannot alter command intent");
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        check(same_quat_bits(first.predicted_root_rotations[index],
                             second.predicted_root_rotations[index]) &&
                  same_quat_bits(first.predicted_desired_headings[index],
                                 second.predicted_desired_headings[index]),
              "applied-velocity changes preserve every heading bit");
    }
}

static void test_snapshot_failure_transactionality()
{
    CommandFixture fixture;
    char error[256] = {};
    G1CommandSnapshot output;
    poison_snapshot(output);
    const G1CommandSnapshot poison = output;

    require_failed_build_preserves(
        output, poison,
        g1_command_snapshot_build(
            output, fixture.intent, fixture.applied,
            fixture.velocities(3), fixture.positions(), fixture.rotations(),
            fixture.headings(), error, sizeof(error)),
        "short desired-velocity shape is rejected transactionally");
    require_failed_build_preserves(
        output, poison,
        g1_command_snapshot_build(
            output, fixture.intent, fixture.applied,
            fixture.velocities(), fixture.positions(5), fixture.rotations(),
            fixture.headings(), error, sizeof(error)),
        "long root-position shape is rejected transactionally");
    require_failed_build_preserves(
        output, poison,
        g1_command_snapshot_build(
            output, fixture.intent, fixture.applied,
            fixture.velocities(), fixture.positions(), fixture.rotations(-1),
            fixture.headings(), error, sizeof(error)),
        "negative rotation shape is rejected transactionally");
    require_failed_build_preserves(
        output, poison,
        g1_command_snapshot_build(
            output, fixture.intent, fixture.applied,
            fixture.velocities(), fixture.positions(), fixture.rotations(),
            slice1d<quat>(G1CommandTrajectorySampleCount, NULL),
            error, sizeof(error)),
        "null heading storage is rejected transactionally");

    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    fixture.intent.requested_velocity.x = infinity;
    require_failed_build_preserves(
        output, poison, build(output, fixture, error, sizeof(error)),
        "nonfinite requested velocity is rejected transactionally");
    fixture = CommandFixture();
    fixture.applied.z = nan;
    require_failed_build_preserves(
        output, poison, build(output, fixture, error, sizeof(error)),
        "nonfinite applied velocity is rejected transactionally");
    fixture = CommandFixture();
    fixture.desired_velocities[2].y = infinity;
    require_failed_build_preserves(
        output, poison, build(output, fixture, error, sizeof(error)),
        "nonfinite predicted velocity is rejected transactionally");
    fixture = CommandFixture();
    fixture.root_positions[1].z = nan;
    require_failed_build_preserves(
        output, poison, build(output, fixture, error, sizeof(error)),
        "nonfinite root position is rejected transactionally");
    fixture = CommandFixture();
    fixture.intent.desired_heading = quat(2.0f, 0.0f, 0.0f, 0.0f);
    require_failed_build_preserves(
        output, poison, build(output, fixture, error, sizeof(error)),
        "non-unit intent heading is rejected transactionally");
    fixture = CommandFixture();
    fixture.root_rotations[3] = quat(infinity, 0.0f, 0.0f, 0.0f);
    require_failed_build_preserves(
        output, poison, build(output, fixture, error, sizeof(error)),
        "nonfinite root rotation is rejected transactionally");
    fixture = CommandFixture();
    fixture.desired_headings[2] = quat(0.5f, 0.0f, 0.0f, 0.0f);
    require_failed_build_preserves(
        output, poison, build(output, fixture, error, sizeof(error)),
        "non-unit predicted heading is rejected transactionally");
}

static void test_snapshot_alias_rejection()
{
    CommandFixture fixture;
    char error[256] = {};
    G1CommandSnapshot output;
    poison_snapshot(output);
    const G1CommandSnapshot before = output;

    require_failed_build_preserves(
        output, before,
        g1_command_snapshot_build(
            output, fixture.intent, fixture.applied,
            slice1d<vec3>(G1CommandTrajectorySampleCount,
                          output.predicted_desired_velocities),
            fixture.positions(), fixture.rotations(), fixture.headings(),
            error, sizeof(error)),
        "desired-velocity/output alias is rejected transactionally");
    require_failed_build_preserves(
        output, before,
        g1_command_snapshot_build(
            output, fixture.intent, fixture.applied, fixture.velocities(),
            slice1d<vec3>(G1CommandTrajectorySampleCount,
                          output.predicted_root_positions),
            fixture.rotations(), fixture.headings(), error, sizeof(error)),
        "root-position/output alias is rejected transactionally");
    require_failed_build_preserves(
        output, before,
        g1_command_snapshot_build(
            output, fixture.intent, fixture.applied, fixture.velocities(),
            fixture.positions(),
            slice1d<quat>(G1CommandTrajectorySampleCount,
                          output.predicted_root_rotations),
            fixture.headings(), error, sizeof(error)),
        "root-rotation/output alias is rejected transactionally");
    require_failed_build_preserves(
        output, before,
        g1_command_snapshot_build(
            output, fixture.intent, fixture.applied, fixture.velocities(),
            fixture.positions(), fixture.rotations(),
            slice1d<quat>(G1CommandTrajectorySampleCount,
                          output.predicted_desired_headings),
            error, sizeof(error)),
        "desired-heading/output alias is rejected transactionally");
    require_failed_build_preserves(
        output, before,
        g1_command_snapshot_build(
            output, fixture.intent, fixture.applied, fixture.velocities(),
            fixture.positions(), fixture.rotations(), fixture.headings(),
            reinterpret_cast<char*>(&output),
            static_cast<int>(sizeof(output))),
        "diagnostic/output alias is rejected without corrupting output");
}

static void test_snapshot_validity_requires_canonical_vectors()
{
    CommandFixture fixture;
    char error[256] = {};
    G1CommandSnapshot value;
    check(build(value, fixture, error, sizeof(error)), error);

    value.applied_velocity.x = -0.0f;
    check(!g1_command_snapshot_is_valid(value),
          "manual negative-zero applied velocity is not publishable");
    check(build(value, fixture, error, sizeof(error)), error);
    value.predicted_root_positions[1].y =
        std::numeric_limits<float>::denorm_min();
    check(!g1_command_snapshot_is_valid(value),
          "manual subnormal root position is not publishable");
    check(build(value, fixture, error, sizeof(error)), error);
    value.predicted_root_rotations[0] = quat(2.0f, 0.0f, 0.0f, 0.0f);
    check(!g1_command_snapshot_is_valid(value),
          "manual non-unit root rotation is not publishable");
}

int main()
{
    static_assert(G1CommandTrajectorySampleCount == 4,
                  "command snapshot shape is exactly four");
    test_exact_heading_parser();
    test_snapshot_build_and_heading_independence();
    test_snapshot_failure_transactionality();
    test_snapshot_alias_rejection();
    test_snapshot_validity_requires_canonical_vectors();
    return 0;
}
