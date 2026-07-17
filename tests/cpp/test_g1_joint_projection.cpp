#include "sonic/cpp/g1_joint_projection.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

static void check(bool condition, const char* expression, int line)
{
    if (!condition) {
        std::fprintf(
            stderr,
            "G1 joint projection check failed at line %d: %s\n",
            line,
            expression);
        std::exit(1);
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

using ProjectSignature = bool (*)(
    sonic_projected_pose&,
    const sonic_joint_contract_entry (&)[SonicG1JointCount],
    slice1d<quat>,
    slice1d<vec3>,
    slice1d<vec3>,
    slice1d<quat>,
    char*,
    int);

using ProjectJointStateSignature = bool (*)(
    float (&)[SonicG1JointCount],
    float (&)[SonicG1JointCount],
    float (&)[SonicG1JointCount],
    sonic_joint_projection_diagnostic&,
    const sonic_joint_contract_entry (&)[SonicG1JointCount],
    slice1d<quat>,
    slice1d<vec3>,
    char*,
    int);

static_assert(SonicG1JointCount == 29, "SONIC G1 joint count");
static_assert(
    std::is_same<decltype(&sonic_project_pose), ProjectSignature>::value,
    "public projection signature");
static_assert(
    std::is_same<
        decltype(&sonic_project_joint_state),
        ProjectJointStateSignature>::value,
    "public joint-state projection signature");
static_assert(SonicJointProjectionValid == 0, "valid diagnostic value");
static_assert(SonicJointProjectionShape == 1, "shape diagnostic value");
static_assert(SonicJointProjectionContract == 2, "contract diagnostic value");
static_assert(SonicJointProjectionInput == 3, "input diagnostic value");
static_assert(SonicJointProjectionSingular == 4, "singular diagnostic value");
static_assert(SonicJointProjectionResidual == 5, "residual diagnostic value");
static_assert(SonicJointProjectionLimit == 6, "limit diagnostic value");
static_assert(SonicJointProjectionVelocity == 7, "velocity diagnostic value");

static bool near(float left, float right, float tolerance = 2.0e-5f)
{
    return std::fabs(left - right) <= tolerance;
}

static bool same_pose_bytes(
    const sonic_projected_pose& left,
    const sonic_projected_pose& right)
{
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

static void poison(sonic_projected_pose& value)
{
    unsigned char* bytes = reinterpret_cast<unsigned char*>(&value);
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        bytes[index] = static_cast<unsigned char>(0xa5U + index % 23U);
    }
}

static void poison_joint_outputs(
    float (&positions)[SonicG1JointCount],
    float (&velocities)[SonicG1JointCount],
    float (&residuals)[SonicG1JointCount])
{
    for (int row = 0; row < SonicG1JointCount; ++row) {
        positions[row] = 101.0f + static_cast<float>(row);
        velocities[row] = -202.0f - static_cast<float>(row);
        residuals[row] = 303.0f + 2.0f * static_cast<float>(row);
    }
}

static bool same_joint_output_bytes(
    const float (&left)[SonicG1JointCount],
    const float (&right)[SonicG1JointCount])
{
    return std::memcmp(left, right, sizeof(left)) == 0;
}

struct ProjectionFixture
{
    static const int BoneCount = 31;

    sonic_joint_contract_entry contract[SonicG1JointCount];
    quat local_rotations[BoneCount];
    vec3 local_angular_velocities[BoneCount];
    vec3 global_positions[BoneCount];
    quat global_rotations[BoneCount];
    char error[256];

    ProjectionFixture()
    {
        static const int parents[BoneCount] = {
            -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
        };
        for (int bone = 0; bone < BoneCount; ++bone) {
            local_rotations[bone] = quat();
            local_angular_velocities[bone] = vec3();
            global_positions[bone] = vec3(
                static_cast<float>(bone),
                static_cast<float>(bone) * -0.25f,
                static_cast<float>(bone) * 0.5f);
            global_rotations[bone] = quat();
        }
        global_positions[1] = vec3(1.25f, 0.875f, -2.5f);
        global_rotations[1] = quat_from_angle_axis(
            0.4f, normalize(vec3(1.0f, 2.0f, -3.0f)));
        for (int index = 0; index < SonicG1JointCount; ++index) {
            const int bone = index + 2;
            contract[index].source_index = index;
            contract[index].source_bone = bone;
            contract[index].source_parent = parents[bone];
            contract[index].target_index = index;
            contract[index].axis_holden = vec3(1.0f, 0.0f, 0.0f);
            contract[index].static_local_holden = quat();
            contract[index].sign = 1.0f;
            contract[index].zero_offset = 0.0f;
            contract[index].lower = -3.13f;
            contract[index].upper = 3.13f;
            contract[index].source_joint =
                std::string("source_joint_") + std::to_string(index);
            contract[index].target_joint = contract[index].source_joint;
        }
        std::memset(error, 0, sizeof(error));
    }

    void pose(int row, float angle)
    {
        const sonic_joint_contract_entry& entry = contract[row];
        local_rotations[entry.source_bone] = quat_mul(
            entry.static_local_holden,
            quat_from_angle_axis(angle, entry.axis_holden));
    }

    void velocity(int row, float speed, vec3 perpendicular = vec3())
    {
        const sonic_joint_contract_entry& entry = contract[row];
        const vec3 axis_parent = quat_mul_vec3(
            entry.static_local_holden, entry.axis_holden);
        local_angular_velocities[entry.source_bone] =
            axis_parent * speed + perpendicular;
    }

    bool project(sonic_projected_pose& output)
    {
        std::memset(error, 0, sizeof(error));
        return sonic_project_pose(
            output,
            contract,
            slice1d<quat>(BoneCount, local_rotations),
            slice1d<vec3>(BoneCount, local_angular_velocities),
            slice1d<vec3>(BoneCount, global_positions),
            slice1d<quat>(BoneCount, global_rotations),
            error,
            static_cast<int>(sizeof(error)));
    }

    bool project_joint_state(
        float (&positions)[SonicG1JointCount],
        float (&velocities)[SonicG1JointCount],
        float (&residuals)[SonicG1JointCount],
        sonic_joint_projection_diagnostic& diagnostic)
    {
        std::memset(error, 0, sizeof(error));
        return sonic_project_joint_state(
            positions,
            velocities,
            residuals,
            diagnostic,
            contract,
            slice1d<quat>(BoneCount, local_rotations),
            slice1d<vec3>(BoneCount, local_angular_velocities),
            error,
            static_cast<int>(sizeof(error)));
    }
};

static void require_structured_failure(
    ProjectionFixture& fixture,
    slice1d<quat> local_rotations,
    slice1d<vec3> local_angular_velocities,
    sonic_joint_projection_failure expected_failure,
    int expected_row = -1)
{
    float positions[SonicG1JointCount];
    float velocities[SonicG1JointCount];
    float residuals[SonicG1JointCount];
    poison_joint_outputs(positions, velocities, residuals);
    float positions_before[SonicG1JointCount];
    float velocities_before[SonicG1JointCount];
    float residuals_before[SonicG1JointCount];
    std::memcpy(positions_before, positions, sizeof(positions));
    std::memcpy(velocities_before, velocities, sizeof(velocities));
    std::memcpy(residuals_before, residuals, sizeof(residuals));
    sonic_joint_projection_diagnostic diagnostic;
    diagnostic.failure = SonicJointProjectionValid;
    diagnostic.row = 91;
    diagnostic.position = 92.0f;
    diagnostic.lower = 93.0f;
    diagnostic.upper = 94.0f;

    std::memset(fixture.error, 0, sizeof(fixture.error));
    CHECK(!sonic_project_joint_state(
        positions,
        velocities,
        residuals,
        diagnostic,
        fixture.contract,
        local_rotations,
        local_angular_velocities,
        fixture.error,
        static_cast<int>(sizeof(fixture.error))));
    CHECK(diagnostic.failure == expected_failure);
    CHECK(diagnostic.row == expected_row);
    CHECK(near(diagnostic.position, 0.0f));
    CHECK(near(diagnostic.lower, 0.0f));
    CHECK(near(diagnostic.upper, 0.0f));
    CHECK(same_joint_output_bytes(positions, positions_before));
    CHECK(same_joint_output_bytes(velocities, velocities_before));
    CHECK(same_joint_output_bytes(residuals, residuals_before));
}

static void test_structured_shape_contract_input_and_singularity_failures()
{
    {
        ProjectionFixture fixture;
        require_structured_failure(
            fixture,
            slice1d<quat>(ProjectionFixture::BoneCount - 1,
                          fixture.local_rotations),
            slice1d<vec3>(ProjectionFixture::BoneCount,
                          fixture.local_angular_velocities),
            SonicJointProjectionShape);
    }
    {
        ProjectionFixture fixture;
        fixture.contract[1].target_index = fixture.contract[0].target_index;
        require_structured_failure(
            fixture,
            slice1d<quat>(ProjectionFixture::BoneCount,
                          fixture.local_rotations),
            slice1d<vec3>(ProjectionFixture::BoneCount,
                          fixture.local_angular_velocities),
            SonicJointProjectionContract);
    }
    {
        ProjectionFixture fixture;
        fixture.local_rotations[fixture.contract[3].source_bone].w =
            std::numeric_limits<float>::quiet_NaN();
        require_structured_failure(
            fixture,
            slice1d<quat>(ProjectionFixture::BoneCount,
                          fixture.local_rotations),
            slice1d<vec3>(ProjectionFixture::BoneCount,
                          fixture.local_angular_velocities),
            SonicJointProjectionInput,
            3);
    }
    {
        ProjectionFixture fixture;
        fixture.local_rotations[fixture.contract[4].source_bone] =
            quat_from_angle_axis(
                std::acos(-1.0f), vec3(0.0f, 1.0f, 0.0f));
        require_structured_failure(
            fixture,
            slice1d<quat>(ProjectionFixture::BoneCount,
                          fixture.local_rotations),
            slice1d<vec3>(ProjectionFixture::BoneCount,
                          fixture.local_angular_velocities),
            SonicJointProjectionSingular,
            4);
    }
}

static void test_structured_residual_and_velocity_failures()
{
    {
        ProjectionFixture fixture;
        fixture.local_rotations[fixture.contract[5].source_bone] =
            quat_from_angle_axis(0.00101f, vec3(0.0f, 1.0f, 0.0f));
        require_structured_failure(
            fixture,
            slice1d<quat>(ProjectionFixture::BoneCount,
                          fixture.local_rotations),
            slice1d<vec3>(ProjectionFixture::BoneCount,
                          fixture.local_angular_velocities),
            SonicJointProjectionResidual,
            5);
    }
    {
        ProjectionFixture fixture;
        fixture.contract[6].axis_holden =
            normalize(vec3(1.0f, 1.0f, 0.0f));
        fixture.local_angular_velocities[fixture.contract[6].source_bone] =
            vec3(
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max(),
                0.0f);
        require_structured_failure(
            fixture,
            slice1d<quat>(ProjectionFixture::BoneCount,
                          fixture.local_rotations),
            slice1d<vec3>(ProjectionFixture::BoneCount,
                          fixture.local_angular_velocities),
            SonicJointProjectionVelocity,
            6);
        CHECK(std::strcmp(
            fixture.error,
            "joint source_joint_6 velocity is non-finite") == 0);
    }
}

static void test_structured_limit_failures_report_both_directions()
{
    const float angles[2] = {-0.5001f, 0.5001f};
    for (int direction = 0; direction < 2; ++direction) {
        ProjectionFixture fixture;
        const int row = 7;
        fixture.contract[row].lower = -0.5f;
        fixture.contract[row].upper = 0.5f;
        fixture.pose(row, angles[direction]);

        float positions[SonicG1JointCount];
        float velocities[SonicG1JointCount];
        float residuals[SonicG1JointCount];
        poison_joint_outputs(positions, velocities, residuals);
        float positions_before[SonicG1JointCount];
        float velocities_before[SonicG1JointCount];
        float residuals_before[SonicG1JointCount];
        std::memcpy(positions_before, positions, sizeof(positions));
        std::memcpy(velocities_before, velocities, sizeof(velocities));
        std::memcpy(residuals_before, residuals, sizeof(residuals));
        sonic_joint_projection_diagnostic diagnostic;

        CHECK(!fixture.project_joint_state(
            positions, velocities, residuals, diagnostic));
        CHECK(diagnostic.failure == SonicJointProjectionLimit);
        CHECK(diagnostic.row == row);
        CHECK(near(diagnostic.position, angles[direction]));
        CHECK(near(diagnostic.lower, -0.5f));
        CHECK(near(diagnostic.upper, 0.5f));
        CHECK(same_joint_output_bytes(positions, positions_before));
        CHECK(same_joint_output_bytes(velocities, velocities_before));
        CHECK(same_joint_output_bytes(residuals, residuals_before));
    }
}

static void test_structured_success_publishes_all_outputs_atomically()
{
    ProjectionFixture fixture;
    fixture.pose(0, 0.25f);
    fixture.velocity(0, -0.75f);
    float positions[SonicG1JointCount];
    float velocities[SonicG1JointCount];
    float residuals[SonicG1JointCount];
    poison_joint_outputs(positions, velocities, residuals);
    sonic_joint_projection_diagnostic diagnostic;
    diagnostic.failure = SonicJointProjectionLimit;
    diagnostic.row = 12;

    CHECK(fixture.project_joint_state(
        positions, velocities, residuals, diagnostic));
    CHECK(diagnostic.failure == SonicJointProjectionValid);
    CHECK(diagnostic.row == -1);
    CHECK(near(positions[0], 0.25f));
    CHECK(near(velocities[0], -0.75f));
    CHECK(near(residuals[0], 0.0f));
}

static void test_signed_angles_on_all_axes_and_pelvis_copy()
{
    ProjectionFixture fixture;
    fixture.contract[0].axis_holden = vec3(1.0f, 0.0f, 0.0f);
    fixture.contract[1].axis_holden = vec3(0.0f, 1.0f, 0.0f);
    fixture.contract[2].axis_holden = vec3(0.0f, 0.0f, 1.0f);
    fixture.pose(0, 0.4f);
    fixture.pose(1, -0.7f);
    fixture.pose(2, 1.2f);

    sonic_projected_pose output;
    CHECK(fixture.project(output));
    CHECK(near(output.source_joint_position[0], 0.4f));
    CHECK(near(output.source_joint_position[1], -0.7f));
    CHECK(near(output.source_joint_position[2], 1.2f));
    CHECK(near(output.physical_pelvis_position_holden.x, 1.25f));
    CHECK(near(output.physical_pelvis_position_holden.y, 0.875f));
    CHECK(near(output.physical_pelvis_position_holden.z, -2.5f));
    CHECK(near(
        quat_angle_between(
            output.physical_pelvis_orientation_holden,
            fixture.global_rotations[1]),
        0.0f));
}

static void test_antipodal_local_quaternions_are_equivalent()
{
    ProjectionFixture fixture;
    fixture.contract[0].axis_holden = vec3(0.0f, 0.0f, 1.0f);
    fixture.contract[0].static_local_holden = quat_from_angle_axis(
        0.8f, normalize(vec3(1.0f, -2.0f, 0.5f)));
    fixture.pose(0, -0.9f);

    sonic_projected_pose positive;
    CHECK(fixture.project(positive));
    for (int bone = 0; bone < ProjectionFixture::BoneCount; ++bone) {
        fixture.local_rotations[bone] = -fixture.local_rotations[bone];
    }
    sonic_projected_pose negative;
    CHECK(fixture.project(negative));
    for (int index = 0; index < SonicG1JointCount; ++index) {
        CHECK(near(
            positive.source_joint_position[index],
            negative.source_joint_position[index]));
        CHECK(near(
            positive.off_axis_residual[index],
            negative.off_axis_residual[index]));
    }
}

static void test_local_angular_velocity_projection_uses_static_frame()
{
    ProjectionFixture fixture;
    fixture.contract[3].axis_holden = vec3(1.0f, 0.0f, 0.0f);
    fixture.contract[3].static_local_holden =
        quat_from_angle_axis(1.1f, vec3(0.0f, 0.0f, 1.0f));
    fixture.contract[3].sign = -1.0f;
    fixture.contract[3].zero_offset = 0.1f;
    fixture.pose(3, 0.2f);
    const vec3 axis_parent = quat_mul_vec3(
        fixture.contract[3].static_local_holden,
        fixture.contract[3].axis_holden);
    const vec3 perpendicular = cross(axis_parent, vec3(0.0f, 0.0f, 1.0f));
    fixture.velocity(3, 1.75f, perpendicular * 4.0f);

    sonic_projected_pose output;
    CHECK(fixture.project(output));
    CHECK(near(output.source_joint_position[3], -0.1f));
    CHECK(near(output.source_joint_velocity[3], -1.75f));
}

static void test_inclusive_range_boundaries_and_transactional_outside_failure()
{
    ProjectionFixture fixture;
    fixture.contract[0].lower = -0.5f;
    fixture.contract[0].upper = 0.5f;

    sonic_projected_pose output;
    fixture.pose(0, -0.5f);
    CHECK(fixture.project(output));
    CHECK(near(output.source_joint_position[0], -0.5f));
    fixture.pose(0, 0.5f);
    CHECK(fixture.project(output));
    CHECK(near(output.source_joint_position[0], 0.5f));

    fixture.pose(0, 0.5001f);
    poison(output);
    const sonic_projected_pose before = output;
    CHECK(!fixture.project(output));
    CHECK(std::strstr(fixture.error, "range") != NULL);
    CHECK(same_pose_bytes(output, before));
}

static void test_legacy_left_ankle_roll_limit_message_is_exact()
{
    ProjectionFixture fixture;
    const int row = 12;
    fixture.contract[row].source_joint = "left_ankle_roll_joint";
    fixture.contract[row].target_joint = "left_ankle_roll_joint";
    fixture.contract[row].lower = -0.261799991f;
    fixture.contract[row].upper = 0.261799991f;
    fixture.pose(row, -0.307408422f);

    sonic_projected_pose output;
    poison(output);
    const sonic_projected_pose before = output;
    CHECK(!fixture.project(output));
    CHECK(std::strcmp(
        fixture.error,
        "joint left_ankle_roll_joint position -0.307408422 is outside "
        "range [-0.261799991, 0.261799991]") == 0);
    CHECK(same_pose_bytes(output, before));
}

static void test_off_axis_threshold_is_strict_and_transactional()
{
    ProjectionFixture fixture;
    fixture.contract[0].axis_holden = vec3(1.0f, 0.0f, 0.0f);
    fixture.local_rotations[fixture.contract[0].source_bone] =
        quat_from_angle_axis(0.00099f, vec3(0.0f, 1.0f, 0.0f));

    sonic_projected_pose output;
    CHECK(fixture.project(output));
    CHECK(near(output.off_axis_residual[0], 0.00099f, 3.0e-6f));

    fixture.local_rotations[fixture.contract[0].source_bone] =
        quat_from_angle_axis(0.00101f, vec3(0.0f, 1.0f, 0.0f));
    poison(output);
    const sonic_projected_pose before = output;
    CHECK(!fixture.project(output));
    CHECK(std::strstr(fixture.error, "off-axis") != NULL);
    CHECK(same_pose_bytes(output, before));
}

static void require_transactional_failure(
    ProjectionFixture& fixture,
    slice1d<quat> local_rotations,
    slice1d<vec3> local_angular_velocities,
    slice1d<vec3> global_positions,
    slice1d<quat> global_rotations,
    const char* expected_error)
{
    sonic_projected_pose output;
    poison(output);
    const sonic_projected_pose before = output;
    std::memset(fixture.error, 0, sizeof(fixture.error));
    CHECK(!sonic_project_pose(
        output,
        fixture.contract,
        local_rotations,
        local_angular_velocities,
        global_positions,
        global_rotations,
        fixture.error,
        static_cast<int>(sizeof(fixture.error))));
    CHECK(std::strstr(fixture.error, expected_error) != NULL);
    CHECK(same_pose_bytes(output, before));
}

static void test_bad_shapes_and_nonfinite_data_are_transactional()
{
    {
        ProjectionFixture fixture;
        require_transactional_failure(
            fixture,
            slice1d<quat>(30, fixture.local_rotations),
            slice1d<vec3>(31, fixture.local_angular_velocities),
            slice1d<vec3>(31, fixture.global_positions),
            slice1d<quat>(31, fixture.global_rotations),
            "shape");
    }
    {
        ProjectionFixture fixture;
        fixture.local_rotations[2].w =
            std::numeric_limits<float>::quiet_NaN();
        require_transactional_failure(
            fixture,
            slice1d<quat>(31, fixture.local_rotations),
            slice1d<vec3>(31, fixture.local_angular_velocities),
            slice1d<vec3>(31, fixture.global_positions),
            slice1d<quat>(31, fixture.global_rotations),
            "finite");
    }
    {
        ProjectionFixture fixture;
        fixture.local_angular_velocities[5].z =
            std::numeric_limits<float>::infinity();
        require_transactional_failure(
            fixture,
            slice1d<quat>(31, fixture.local_rotations),
            slice1d<vec3>(31, fixture.local_angular_velocities),
            slice1d<vec3>(31, fixture.global_positions),
            slice1d<quat>(31, fixture.global_rotations),
            "finite");
    }
    {
        ProjectionFixture fixture;
        fixture.global_positions[1].x =
            std::numeric_limits<float>::quiet_NaN();
        require_transactional_failure(
            fixture,
            slice1d<quat>(31, fixture.local_rotations),
            slice1d<vec3>(31, fixture.local_angular_velocities),
            slice1d<vec3>(31, fixture.global_positions),
            slice1d<quat>(31, fixture.global_rotations),
            "finite");
    }
    {
        ProjectionFixture fixture;
        fixture.global_rotations[1].x =
            std::numeric_limits<float>::infinity();
        require_transactional_failure(
            fixture,
            slice1d<quat>(31, fixture.local_rotations),
            slice1d<vec3>(31, fixture.local_angular_velocities),
            slice1d<vec3>(31, fixture.global_positions),
            slice1d<quat>(31, fixture.global_rotations),
            "finite");
    }
}

int main()
{
    test_structured_shape_contract_input_and_singularity_failures();
    test_structured_residual_and_velocity_failures();
    test_structured_limit_failures_report_both_directions();
    test_structured_success_publishes_all_outputs_atomically();
    test_signed_angles_on_all_axes_and_pelvis_copy();
    test_antipodal_local_quaternions_are_equivalent();
    test_local_angular_velocity_projection_uses_static_frame();
    test_inclusive_range_boundaries_and_transactional_outside_failure();
    test_legacy_left_ankle_roll_limit_message_is_exact();
    test_off_axis_threshold_is_strict_and_transactional();
    test_bad_shapes_and_nonfinite_data_are_transactional();
    std::puts("G1 joint projection tests passed");
    return 0;
}
