#include "g1_sole_diagnostics.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static void check(bool condition, const char* expression, int line)
{
    if (!condition) {
        std::fprintf(stderr, "CHECK failed at line %d: %s\n", line, expression);
        std::exit(1);
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

static void check_near(float actual, float expected, float tolerance, int line)
{
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        std::fprintf(
            stderr,
            "CHECK_NEAR failed at line %d: actual=%.9g expected=%.9g tolerance=%.9g\n",
            line,
            actual,
            expected,
            tolerance);
        std::exit(1);
    }
}

#define CHECK_NEAR(actual, expected, tolerance) \
    check_near((actual), (expected), (tolerance), __LINE__)

struct pose_fixture
{
    array1d<vec3> positions;
    array1d<quat> rotations;

    pose_fixture() : positions(G1_BoneCount), rotations(G1_BoneCount)
    {
        for (int bone = 0; bone < G1_BoneCount; ++bone) {
            positions(bone) = vec3();
            rotations(bone) = quat();
        }
    }
};

static void make_plane_heightfield(
    heightfield& field,
    float base,
    float slope_x,
    float slope_z)
{
    field.version = 2;
    field.nx = 101;
    field.nz = 101;
    field.origin_x = 0.0f;
    field.origin_z = 0.0f;
    field.cell_size = 0.01f;
    field.exterior_height = 0.0f;
    field.heights.resize(field.nx * field.nz);
    for (int z = 0; z < field.nz; ++z) {
        for (int x = 0; x < field.nx; ++x) {
            const float world_x = x * field.cell_size;
            const float world_z = z * field.cell_size;
            field.heights(z * field.nx + x) =
                base + slope_x * world_x + slope_z * world_z;
        }
    }
    CHECK(terrain_heightfield_is_queryable(field));
}

static G1SoleDiagnosticStatus observe(
    G1SoleDiagnosticSnapshot& output,
    G1SoleDiagnosticState& state,
    const heightfield& field,
    const pose_fixture& pose,
    const G1LegConfig (&legs)[2],
    bool left_contact,
    bool right_contact,
    int scene_generation,
    bool explicit_reset,
    char* error,
    int error_capacity)
{
    const bool contacts[2] = {left_contact, right_contact};
    return g1_sole_diagnostics_observe(
        output,
        state,
        field,
        pose.positions,
        pose.rotations,
        legs,
        contacts,
        scene_generation,
        explicit_reset,
        error,
        error_capacity);
}

static void check_exact_authoritative_transforms_and_full_rotation()
{
    heightfield field;
    make_plane_heightfield(field, 0.1f, 0.0f, 0.0f);
    pose_fixture pose;
    const G1LegConfig legs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    pose.positions(G1_LeftAnkle) = vec3(0.40f, 0.80f, 0.40f);
    pose.positions(G1_RightAnkle) = vec3(0.70f, 0.90f, 0.50f);
    pose.rotations(G1_LeftAnkle) =
        quat_from_angle_axis(0.5f * PIf, vec3(0.0f, 0.0f, 1.0f));

    G1SoleDiagnosticState state;
    G1SoleDiagnosticSnapshot output;
    char error[256] = {};
    CHECK(observe(
        output, state, field, pose, legs, true, true, 0, false,
        error, sizeof(error)) == G1SoleDiagnosticOk);

    // A +90 degree rotation about Z maps (x,y,z) to (-y,x,z). This proves
    // the diagnostic applies the full final ankle quaternion, not yaw alone.
    const vec3 expected_left[4] = {
        vec3(0.45f, 0.75f, 0.375f),
        vec3(0.45f, 0.75f, 0.425f),
        vec3(0.45f, 0.92f, 0.370f),
        vec3(0.45f, 0.92f, 0.430f),
    };
    const vec3 expected_right[4] = {
        vec3(0.65f, 0.85f, 0.475f),
        vec3(0.65f, 0.85f, 0.525f),
        vec3(0.82f, 0.85f, 0.470f),
        vec3(0.82f, 0.85f, 0.530f),
    };
    for (int probe = 0; probe < 4; ++probe) {
        CHECK_NEAR(output.world_points[0][probe].x, expected_left[probe].x, 1e-6f);
        CHECK_NEAR(output.world_points[0][probe].y, expected_left[probe].y, 1e-6f);
        CHECK_NEAR(output.world_points[0][probe].z, expected_left[probe].z, 1e-6f);
        CHECK_NEAR(output.world_points[1][probe].x, expected_right[probe].x, 1e-6f);
        CHECK_NEAR(output.world_points[1][probe].y, expected_right[probe].y, 1e-6f);
        CHECK_NEAR(output.world_points[1][probe].z, expected_right[probe].z, 1e-6f);
        CHECK_NEAR(output.surface_heights[0][probe], 0.1f, 1e-6f);
        CHECK_NEAR(output.surface_heights[1][probe], 0.1f, 1e-6f);
    }
    CHECK(output.contact[0] && output.contact[1]);
    CHECK(output.slip_reset[0] && output.slip_reset[1]);
    CHECK(output.stance_slip[0] == 0.0f);
    CHECK(output.stance_slip[1] == 0.0f);
}

static void check_sloped_surface_mixed_corners_and_minima()
{
    heightfield field;
    make_plane_heightfield(field, 0.1f, 0.5f, 0.25f);
    pose_fixture pose;
    const G1LegConfig legs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    pose.positions(G1_LeftAnkle) = vec3(0.30f, 0.50f, 0.30f);
    pose.positions(G1_RightAnkle) = vec3(0.60f, 0.65f, 0.30f);

    G1SoleDiagnosticState state;
    G1SoleDiagnosticSnapshot output;
    char error[256] = {};
    CHECK(observe(
        output, state, field, pose, legs, false, false, 3, false,
        error, sizeof(error)) == G1SoleDiagnosticOk);

    float expected_minima[2] = {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    for (int foot = 0; foot < 2; ++foot) {
        for (int probe = 0; probe < 4; ++probe) {
            const vec3 point = output.world_points[foot][probe];
            const float surface = 0.1f + 0.5f * point.x + 0.25f * point.z;
            const float clearance = point.y - surface;
            CHECK_NEAR(output.surface_heights[foot][probe], surface, 2e-6f);
            CHECK_NEAR(output.clearances[foot][probe], clearance, 2e-6f);
            expected_minima[foot] =
                clearance < expected_minima[foot]
                    ? clearance : expected_minima[foot];
        }
        CHECK_NEAR(output.minimum_clearance[foot], expected_minima[foot], 2e-6f);
    }
    CHECK_NEAR(
        output.global_minimum_clearance,
        expected_minima[0] < expected_minima[1]
            ? expected_minima[0] : expected_minima[1],
        2e-6f);
    CHECK(output.clearances[0][0] != output.clearances[0][3]);
    CHECK(output.clearances[1][0] != output.clearances[1][3]);
}

static void check_stance_slip_accumulation_and_resets()
{
    heightfield field;
    make_plane_heightfield(field, 0.1f, 0.0f, 0.0f);
    pose_fixture pose;
    const G1LegConfig legs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    pose.positions(G1_LeftAnkle) = vec3(0.30f, 0.50f, 0.30f);
    pose.positions(G1_RightAnkle) = vec3(0.60f, 0.50f, 0.30f);
    G1SoleDiagnosticState state;
    G1SoleDiagnosticSnapshot output;
    char error[256] = {};

    CHECK(observe(
        output, state, field, pose, legs, true, false, 7, false,
        error, sizeof(error)) == G1SoleDiagnosticOk);
    CHECK(output.slip_reset[0]);
    CHECK(output.stance_slip[0] == 0.0f);

    pose.positions(G1_LeftAnkle).x += 0.03f;
    pose.positions(G1_LeftAnkle).z += 0.04f;
    CHECK(observe(
        output, state, field, pose, legs, true, false, 7, false,
        error, sizeof(error)) == G1SoleDiagnosticOk);
    CHECK(!output.slip_reset[0]);
    CHECK_NEAR(output.stance_slip[0], 0.05f, 1e-6f);

    pose.positions(G1_LeftAnkle).x += 0.06f;
    pose.positions(G1_LeftAnkle).z += 0.08f;
    CHECK(observe(
        output, state, field, pose, legs, true, false, 7, false,
        error, sizeof(error)) == G1SoleDiagnosticOk);
    CHECK_NEAR(output.stance_slip[0], 0.15f, 1e-6f);

    // Releasing contact resets the accumulator and centroid history.
    CHECK(observe(
        output, state, field, pose, legs, false, false, 7, false,
        error, sizeof(error)) == G1SoleDiagnosticOk);
    CHECK(output.slip_reset[0]);
    CHECK(output.stance_slip[0] == 0.0f);
    pose.positions(G1_LeftAnkle).x += 0.08f;
    CHECK(observe(
        output, state, field, pose, legs, true, false, 7, false,
        error, sizeof(error)) == G1SoleDiagnosticOk);
    CHECK(output.slip_reset[0]);
    CHECK(output.stance_slip[0] == 0.0f);

    // A generation change resets both feet exactly once.
    pose.positions(G1_LeftAnkle).x += 0.02f;
    CHECK(observe(
        output, state, field, pose, legs, true, true, 8, false,
        error, sizeof(error)) == G1SoleDiagnosticOk);
    CHECK(output.slip_reset[0] && output.slip_reset[1]);
    CHECK(output.stance_slip[0] == 0.0f);
    CHECK(output.stance_slip[1] == 0.0f);

    pose.positions(G1_LeftAnkle).x += 0.02f;
    pose.positions(G1_RightAnkle).z += 0.03f;
    CHECK(observe(
        output, state, field, pose, legs, true, true, 8, false,
        error, sizeof(error)) == G1SoleDiagnosticOk);
    CHECK_NEAR(output.stance_slip[0], 0.02f, 1e-6f);
    CHECK_NEAR(output.stance_slip[1], 0.03f, 1e-6f);

    // Explicit state reset also establishes new centroids without movement.
    pose.positions(G1_LeftAnkle).x += 0.03f;
    pose.positions(G1_RightAnkle).z += 0.04f;
    CHECK(observe(
        output, state, field, pose, legs, true, true, 8, true,
        error, sizeof(error)) == G1SoleDiagnosticOk);
    CHECK(output.slip_reset[0] && output.slip_reset[1]);
    CHECK(output.stance_slip[0] == 0.0f);
    CHECK(output.stance_slip[1] == 0.0f);
}

static void check_transactional_failure_cases()
{
    heightfield field;
    make_plane_heightfield(field, 0.1f, 0.0f, 0.0f);
    pose_fixture pose;
    G1LegConfig legs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    pose.positions(G1_LeftAnkle) = vec3(0.30f, 0.50f, 0.30f);
    pose.positions(G1_RightAnkle) = vec3(0.60f, 0.50f, 0.30f);
    G1SoleDiagnosticState state;
    G1SoleDiagnosticSnapshot output;
    char error[256] = {};

    CHECK(observe(
        output, state, field, pose, legs, true, true, 4, false,
        error, sizeof(error)) == G1SoleDiagnosticOk);
    pose.positions(G1_LeftAnkle).x += 0.02f;
    CHECK(observe(
        output, state, field, pose, legs, true, true, 4, false,
        error, sizeof(error)) == G1SoleDiagnosticOk);

    const G1SoleDiagnosticState accepted_state = state;
    const G1SoleDiagnosticSnapshot accepted_output = output;

    auto check_preserved = [&]() {
        CHECK(std::memcmp(&state, &accepted_state, sizeof(state)) == 0);
        CHECK(std::memcmp(&output, &accepted_output, sizeof(output)) == 0);
    };

    pose.positions(G1_LeftAnkle).x =
        std::numeric_limits<float>::quiet_NaN();
    CHECK(observe(
        output, state, field, pose, legs, true, true, 4, false,
        error, sizeof(error)) == G1SoleDiagnosticInvalidInput);
    CHECK(std::strstr(error, "pose") != nullptr);
    check_preserved();
    pose.positions(G1_LeftAnkle).x = 0.32f;

    pose.rotations(G1_RightAnkle).w =
        std::numeric_limits<float>::infinity();
    CHECK(observe(
        output, state, field, pose, legs, true, true, 4, false,
        error, sizeof(error)) == G1SoleDiagnosticInvalidInput);
    check_preserved();
    pose.rotations(G1_RightAnkle) = quat();

    legs[0].sole_points_local[2].z =
        std::numeric_limits<float>::quiet_NaN();
    CHECK(observe(
        output, state, field, pose, legs, true, true, 4, false,
        error, sizeof(error)) == G1SoleDiagnosticInvalidInput);
    CHECK(std::strstr(error, "sole") != nullptr);
    check_preserved();
    legs[0] = g1_left_leg_config();

    // A valid pose outside the authoritative node rectangle is controlled
    // separately from malformed field/input failures.
    pose.positions(G1_RightAnkle).x = 1.20f;
    CHECK(observe(
        output, state, field, pose, legs, true, true, 4, false,
        error, sizeof(error)) == G1SoleDiagnosticOutsideDomain);
    CHECK(std::strstr(error, "outside") != nullptr);
    check_preserved();
    pose.positions(G1_RightAnkle).x = 0.60f;

    const uint32_t accepted_version = field.version;
    field.version = 1;
    CHECK(observe(
        output, state, field, pose, legs, true, true, 4, false,
        error, sizeof(error)) == G1SoleDiagnosticInvalidField);
    check_preserved();
    field.version = accepted_version;

    const int queried_cell = 2 * field.nx;
    const float accepted_height = field.heights(queried_cell);
    field.heights(queried_cell) =
        std::numeric_limits<float>::quiet_NaN();
    pose.positions(G1_LeftAnkle) = vec3(0.05f, 0.5f, 0.05f);
    CHECK(observe(
        output, state, field, pose, legs, true, true, 4, true,
        error, sizeof(error)) == G1SoleDiagnosticInvalidSurface);
    CHECK(std::strstr(error, "surface") != nullptr);
    check_preserved();
    field.heights(queried_cell) = accepted_height;

    CHECK(observe(
        output, state, field, pose, legs, true, true, -1, false,
        error, sizeof(error)) == G1SoleDiagnosticInvalidInput);
    check_preserved();
}

int main()
{
    check_exact_authoritative_transforms_and_full_rotation();
    check_sloped_surface_mixed_corners_and_minima();
    check_stance_slip_accumulation_and_resets();
    check_transactional_failure_cases();
    std::puts("VALID read-only transactional physical sole diagnostics");
    return 0;
}
