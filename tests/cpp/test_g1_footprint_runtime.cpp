#include "../../g1_footprint_runtime.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "G1 footprint runtime test failed: %s\n", message);
        std::exit(1);
    }
}

static uint32_t bits(float value)
{
    uint32_t output = 0;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

static float from_bits(uint32_t value)
{
    float output = 0.0f;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

static bool same_vec3_bits(vec3 left, vec3 right)
{
    return bits(left.x) == bits(right.x) &&
           bits(left.y) == bits(right.y) &&
           bits(left.z) == bits(right.z);
}

static bool same_surface_bits(
    const G1SurfaceSample& left,
    const G1SurfaceSample& right)
{
    return bits(left.height) == bits(right.height) &&
           same_vec3_bits(left.normal, right.normal);
}

template<typename T>
static void poison(T& value)
{
    unsigned char* const bytes = reinterpret_cast<unsigned char*>(&value);
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        bytes[index] = static_cast<unsigned char>(0xa5U + index % 19U);
    }
}

template<typename T>
static bool same_bytes(const T& left, const T& right)
{
    return std::memcmp(&left, &right, sizeof(T)) == 0;
}

struct TerrainFixture
{
    heightfield field;
    walkability_grid grid;

    TerrainFixture(
        int nx = 101,
        int nz = 101,
        float origin_x = 0.0f,
        float origin_z = 0.0f,
        float cell_size = 0.02f)
    {
        field.version = 2;
        field.nx = nx;
        field.nz = nz;
        field.origin_x = origin_x;
        field.origin_z = origin_z;
        field.cell_size = cell_size;
        field.exterior_height = 0.0f;
        field.heights.resize(nx * nz);
        field.heights.set(0.0f);
        grid.nx = nx;
        grid.nz = nz;
        grid.cells.resize(nx * nz);
        grid.cells.set(1);
    }

    void set_x_step(float last_upper_x, float upper_height)
    {
        for (int z = 0; z < field.nz; ++z) {
            for (int x = 0; x < field.nx; ++x) {
                const float node_x = field.origin_x +
                    static_cast<float>(x) * field.cell_size;
                field.heights(z * field.nx + x) =
                    node_x <= last_upper_x ? upper_height : 0.0f;
            }
        }
    }
};

struct PoseFixture
{
    vec3 positions[G1_BoneCount];
    quat rotations[G1_BoneCount];

    PoseFixture()
    {
        for (int bone = 0; bone < G1_BoneCount; ++bone) {
            positions[bone] = vec3(1.0f, 1.0f, 1.0f);
            rotations[bone] = quat(1.0f, 0.0f, 0.0f, 0.0f);
        }
        positions[G1_LeftAnkle] = vec3(0.80f, 0.08f, 0.80f);
        positions[G1_RightAnkle] = vec3(0.80f, 0.08f, 1.10f);
    }

    slice1d<vec3> position_slice()
    {
        return slice1d<vec3>(G1_BoneCount, positions);
    }

    slice1d<quat> rotation_slice()
    {
        return slice1d<quat>(G1_BoneCount, rotations);
    }
};

static G1CommandSnapshot make_command(
    const vec3 roots[G1CommandTrajectorySampleCount],
    const quat rotations[G1CommandTrajectorySampleCount])
{
    G1CommandSnapshot command = {};
    command.intent.requested_velocity = vec3(0.0f, 0.0f, 0.0f);
    command.intent.desired_heading = quat(1.0f, 0.0f, 0.0f, 0.0f);
    command.applied_velocity = vec3(0.0f, 0.0f, 0.0f);
    for (int sample = 0; sample < G1CommandTrajectorySampleCount; ++sample) {
        command.predicted_desired_velocities[sample] =
            vec3(0.0f, 0.0f, 0.0f);
        command.predicted_root_positions[sample] = roots[sample];
        command.predicted_root_rotations[sample] = rotations[sample];
        command.predicted_desired_headings[sample] =
            quat(1.0f, 0.0f, 0.0f, 0.0f);
    }
    return command;
}

static G1CommandSnapshot stationary_command(float x = 0.80f, float z = 0.95f)
{
    vec3 roots[G1CommandTrajectorySampleCount];
    quat rotations[G1CommandTrajectorySampleCount];
    for (int sample = 0; sample < G1CommandTrajectorySampleCount; ++sample) {
        roots[sample] = vec3(x, 0.0f, z);
        rotations[sample] = quat(1.0f, 0.0f, 0.0f, 0.0f);
    }
    return make_command(roots, rotations);
}

static G1FootContactSchedule contact_schedule(
    bool left0,
    bool left1,
    bool left2,
    bool left3,
    bool right0 = false,
    bool right1 = false,
    bool right2 = false,
    bool right3 = false)
{
    G1FootContactSchedule schedule = {};
    const bool left[] = {left0, left1, left2, left3};
    const bool right[] = {right0, right1, right2, right3};
    for (int sample = 0; sample < G1CommandTrajectorySampleCount; ++sample) {
        schedule.contact[0][sample] = left[sample];
        schedule.contact[1][sample] = right[sample];
    }
    return schedule;
}

static G1FootprintStatus observe(
    G1FootprintObservation& output,
    const G1FootprintBudget& budget,
    TerrainFixture& terrain,
    const G1CommandSnapshot& command,
    const G1FootContactSchedule& contacts,
    PoseFixture& pose,
    char* error = NULL,
    int error_capacity = 0)
{
    return g1_footprint_observe_v2(
        output,
        budget,
        terrain.field,
        terrain.grid,
        command,
        contacts,
        pose.position_slice(),
        pose.rotation_slice(),
        error,
        error_capacity);
}

static void require_observation_failure_preserves(
    G1FootprintObservation& output,
    const G1FootprintObservation& before,
    G1FootprintStatus actual,
    G1FootprintStatus expected,
    const char* message)
{
    check(actual == expected, message);
    check(same_bytes(output, before), message);
}

static void test_contact_schedule_exact_horizons_and_transaction()
{
    bool table[60 * 2] = {};
    const int sampled_frames[] = {5, 14, 22, 29};
    const bool expected_left[] = {true, false, true, true};
    const bool expected_right[] = {false, true, false, true};
    for (int index = 0; index < 4; ++index) {
        table[sampled_frames[index] * 2] = expected_left[index];
        table[sampled_frames[index] * 2 + 1] = expected_right[index];
    }
    int starts[] = {0, 30};
    int stops[] = {30, 60};
    G1FootContactSchedule schedule;
    poison(schedule);
    char error[256] = {};
    check(g1_foot_contact_schedule_build(
              schedule,
              slice2d<bool>(60, 2, table),
              slice1d<int>(2, starts),
              slice1d<int>(2, stops),
              5,
              true,
              false,
              0.04f,
              1.0f / 3.0f,
              error,
              static_cast<int>(sizeof(error))),
          error);
    for (int sample = 0; sample < 4; ++sample) {
        check(schedule.contact[0][sample] == expected_left[sample],
              "contact schedule uses exact left 0,9,17,25 offsets");
        check(schedule.contact[1][sample] == expected_right[sample],
              "contact schedule uses exact right 0,9,17,25 offsets");
    }

    table[10 * 2] = true;
    table[10 * 2 + 1] = false;
    table[19 * 2] = false;
    table[19 * 2 + 1] = true;
    table[27 * 2] = true;
    table[27 * 2 + 1] = false;
    table[29 * 2] = false;
    table[29 * 2 + 1] = true;
    check(g1_foot_contact_schedule_build(
              schedule,
              slice2d<bool>(60, 2, table),
              slice1d<int>(2, starts),
              slice1d<int>(2, stops),
              10,
              true,
              false,
              0.04f,
              1.0f / 3.0f,
              error,
              static_cast<int>(sizeof(error))),
          "contact schedule clamps only to current range stop minus one");
    check(!schedule.contact[0][3] && schedule.contact[1][3],
          "contact schedule final horizon is range-clamped frame 29");

    const float infinity = std::numeric_limits<float>::infinity();
    struct InvalidCase
    {
        int rows;
        int cols;
        int range_count;
        int current_frame;
        bool current_left;
        bool current_right;
        float dt;
        float sample_time;
        int second_start;
        const char* message;
    } invalid[] = {
        {60, 2, 2, 10, false, false, 0.04f, 1.0f / 3.0f, 30,
         "current left contact mismatch is transactional"},
        {60, 2, 2, 10, true, true, 0.04f, 1.0f / 3.0f, 30,
         "current right contact mismatch is transactional"},
        {60, 2, 2, 60, false, false, 0.04f, 1.0f / 3.0f, 30,
         "frame outside all ranges is transactional"},
        {60, 2, 2, 10, true, false, 0.04f, 1.0f / 3.0f, 29,
         "overlapping ranges are transactional"},
        {60, 2, 2, 10, true, false, 0.04f, 1.0f / 3.0f, 31,
         "gapped ranges are transactional"},
        {60, 1, 2, 10, true, false, 0.04f, 1.0f / 3.0f, 30,
         "wrong contact column shape is transactional"},
        {60, 2, 1, 10, true, false, 0.04f, 1.0f / 3.0f, 30,
         "wrong range shape is transactional"},
        {60, 2, 2, 10, true, false,
         std::nextafter(0.04f, infinity), 1.0f / 3.0f, 30,
         "non-25-Hz dt is transactional"},
        {60, 2, 2, 10, true, false, 0.04f, 0.0f, 30,
         "zero horizon is transactional"},
        {60, 2, 2, 10, true, false, 0.04f, infinity, 30,
         "nonfinite horizon is transactional"},
        {60, 2, 2, 10, true, false, 0.04f, 1.0e20f, 30,
         "overflowing horizon is transactional"},
    };
    for (const InvalidCase& input : invalid) {
        starts[1] = input.second_start;
        poison(schedule);
        const G1FootContactSchedule before = schedule;
        const bool result = g1_foot_contact_schedule_build(
            schedule,
            slice2d<bool>(input.rows, input.cols, table),
            slice1d<int>(input.range_count, starts),
            slice1d<int>(input.range_count, stops),
            input.current_frame,
            input.current_left,
            input.current_right,
            input.dt,
            input.sample_time,
            error,
            static_cast<int>(sizeof(error)));
        check(!result && same_bytes(schedule, before), input.message);
    }
    starts[1] = 30;

    poison(schedule);
    const G1FootContactSchedule before_alias = schedule;
    check(!g1_foot_contact_schedule_build(
              schedule,
              slice2d<bool>(60, 2, table),
              slice1d<int>(2, starts),
              slice1d<int>(2, stops),
              10,
              true,
              false,
              0.04f,
              1.0f / 3.0f,
              reinterpret_cast<char*>(&schedule) + 1,
              2),
          "contact diagnostic/output alias is rejected");
    check(same_bytes(schedule, before_alias),
          "contact diagnostic/output alias preserves poison");

    poison(schedule);
    const G1FootContactSchedule before_contact_alias = schedule;
    int alias_starts[] = {0};
    int alias_stops[] = {4};
    check(!g1_foot_contact_schedule_build(
              schedule,
              slice2d<bool>(4, 2, reinterpret_cast<bool*>(&schedule)),
              slice1d<int>(1, alias_starts),
              slice1d<int>(1, alias_stops),
              0,
              false,
              false,
              0.04f,
              1.0f / 3.0f,
              error,
              static_cast<int>(sizeof(error))),
          "contact table/output storage alias is rejected before input read");
    check(same_bytes(schedule, before_contact_alias),
          "contact table/output alias preserves poison");

    poison(schedule);
    const G1FootContactSchedule before_range_alias = schedule;
    check(!g1_foot_contact_schedule_build(
              schedule,
              slice2d<bool>(4, 2, table),
              slice1d<int>(1, reinterpret_cast<int*>(&schedule)),
              slice1d<int>(1, alias_stops),
              0,
              false,
              false,
              0.04f,
              1.0f / 3.0f,
              error,
              static_cast<int>(sizeof(error))),
          "range/output storage alias is rejected before input read");
    check(same_bytes(schedule, before_range_alias),
          "range/output alias preserves poison");
}

static void test_six_heading_transport_and_phase_anchors()
{
    const quat headings[] = {
        quat(1.0f, 0.0f, 0.0f, 0.0f),
        quat(0.0f, 0.0f, 1.0f, 0.0f),
        quat(0.707106769f, 0.0f, 0.707106769f, 0.0f),
        quat(0.707106769f, 0.0f, -0.707106769f, 0.0f),
        quat(0.923879504f, 0.0f, 0.382683426f, 0.0f),
        quat(0.923879504f, 0.0f, -0.382683426f, 0.0f),
    };
    TerrainFixture terrain(151, 151, 0.0f, 0.0f, 0.02f);
    PoseFixture pose;
    pose.positions[G1_LeftAnkle] = vec3(1.0f, 0.08f, 1.0f);
    pose.positions[G1_RightAnkle] = vec3(1.0f, 0.08f, 1.30f);
    const G1LegConfig legs[] = {g1_left_leg_config(), g1_right_leg_config()};

    for (const quat heading : headings) {
        vec3 roots[4];
        quat rotations[4];
        for (int sample = 0; sample < 4; ++sample) {
            roots[sample] = vec3(1.0f, 0.0f, 1.15f);
            rotations[sample] = sample == 0
                ? quat(1.0f, 0.0f, 0.0f, 0.0f)
                : heading;
        }
        const G1CommandSnapshot command = make_command(roots, rotations);
        G1FootprintObservation observation = {};
        check(observe(
                  observation,
                  g1_footprint_budget(),
                  terrain,
                  command,
                  contact_schedule(false, false, false, false),
                  pose) == G1FootprintOk,
              "six-heading swing observation succeeds without mode enum");
        for (int foot = 0; foot < 2; ++foot) {
            for (int probe = 0; probe < 4; ++probe) {
                const vec3 expected_current_sphere =
                    pose.positions[legs[foot].ankle] +
                    quat_mul_vec3(
                        pose.rotations[legs[foot].ankle],
                        legs[foot].foot_sphere_centers_local[probe]);
                const vec3 expected_current_sole =
                    pose.positions[legs[foot].ankle] +
                    quat_mul_vec3(
                        pose.rotations[legs[foot].ankle],
                        legs[foot].sole_points_local[probe]);
                check(same_vec3_bits(
                          observation.feet[foot].probes[probe]
                              .predicted_sphere_centers[0],
                          expected_current_sphere),
                      "sample zero sphere preserves exact current bits");
                check(same_vec3_bits(
                          observation.feet[foot].probes[probe]
                              .predicted_sole_points[0],
                          expected_current_sole),
                      "sample zero sole preserves exact current bits");
                const vec3 sphere_local = quat_inv_mul_vec3(
                    rotations[0], expected_current_sphere - roots[0]);
                const vec3 sole_local = quat_inv_mul_vec3(
                    rotations[0], expected_current_sole - roots[0]);
                for (int sample = 1; sample < 4; ++sample) {
                    const vec3 expected_sphere = roots[sample] +
                        quat_mul_vec3(rotations[sample], sphere_local);
                    const vec3 expected_sole = roots[sample] +
                        quat_mul_vec3(rotations[sample], sole_local);
                    check(same_vec3_bits(
                              observation.feet[foot].probes[probe]
                                  .predicted_sphere_centers[sample],
                              expected_sphere),
                          "six-heading sphere transport is bit exact");
                    check(same_vec3_bits(
                              observation.feet[foot].probes[probe]
                                  .predicted_sole_points[sample],
                              expected_sole),
                          "six-heading sole transport is bit exact");
                }
            }
        }

        observation = G1FootprintObservation{};
        check(observe(
                  observation,
                  g1_footprint_budget(),
                  terrain,
                  command,
                  contact_schedule(
                      true, true, false, false,
                      true, true, false, false),
                  pose) == G1FootprintOk,
              "planted/release observation succeeds");
        for (int foot = 0; foot < 2; ++foot) {
            for (int probe = 0; probe < 4; ++probe) {
                const G1FootprintProbe& observed =
                    observation.feet[foot].probes[probe];
                check(same_vec3_bits(
                          observed.predicted_sphere_centers[0],
                          observed.predicted_sphere_centers[1]) &&
                          same_vec3_bits(
                              observed.predicted_sole_points[0],
                              observed.predicted_sole_points[1]),
                      "contact anchor holds exact bits before release");
                const vec3 current_sphere = observed.current_sphere_center;
                const vec3 current_sole = observed.current_sole_point;
                const vec3 sphere_local = quat_inv_mul_vec3(
                    rotations[0], current_sphere - roots[0]);
                const vec3 sole_local = quat_inv_mul_vec3(
                    rotations[0], current_sole - roots[0]);
                for (int sample = 2; sample < 4; ++sample) {
                    check(same_vec3_bits(
                              observed.predicted_sphere_centers[sample],
                              roots[sample] + quat_mul_vec3(
                                  rotations[sample], sphere_local)) &&
                              same_vec3_bits(
                                  observed.predicted_sole_points[sample],
                                  roots[sample] + quat_mul_vec3(
                                      rotations[sample], sole_local)),
                          "true-to-false edge releases into transported geometry");
                }
            }
        }
    }
}

static void test_lateral_split_and_centerline_invariance()
{
    TerrainFixture terrain;
    terrain.set_x_step(0.60f, 0.32f);
    PoseFixture pose;
    pose.positions[G1_LeftAnkle] = vec3(0.63f, 0.40f, 0.80f);
    pose.positions[G1_RightAnkle] = vec3(0.90f, 0.08f, 1.10f);
    G1CommandSnapshot command = stationary_command(0.62f, 0.95f);
    G1FootprintObservation observation = {};
    check(observe(
              observation,
              g1_footprint_budget(),
              terrain,
              command,
              contact_schedule(
                  true, true, true, true,
                  true, true, true, true),
              pose) == G1FootprintOk,
          "lateral root/foot split observation succeeds");
    check(bits(observation.root_surface.height) == bits(0.0f),
          "lateral split root surface remains exactly lower level");
    check(observation.feet[0].maximum_root_split_m >=
              static_cast<double>(0.32f),
          "lateral physical footprint exposes at least 0.32 metre split");
    check(observation.feet[0].multilevel,
          "lateral split crosses exact multilevel threshold");

    terrain_centerline_snapshot centerline = {};
    terrain_centerline_snapshot_compute_v2(
        centerline,
        terrain.field,
        command.predicted_root_positions[0],
        slice1d<vec3>(4, command.predicted_root_positions),
        slice1d<quat>(4, command.predicted_root_rotations));
    for (int sample = 0; sample < 4; ++sample) {
        check(bits(centerline.values[sample]) == bits(0.0f),
              "root-centerline matcher terrain stays exactly zero");
    }
}

static void test_exact_multilevel_threshold()
{
    const float threshold = 0.04f;
    const float heights[] = {
        threshold,
        std::nextafter(threshold, 0.0f)
    };
    for (int index = 0; index < 2; ++index) {
        TerrainFixture terrain;
        terrain.set_x_step(0.60f, heights[index]);
        PoseFixture pose;
        pose.positions[G1_LeftAnkle] = vec3(0.63f, 0.08f, 0.80f);
        pose.positions[G1_RightAnkle] = vec3(0.90f, 0.08f, 1.10f);
        G1FootprintObservation observation = {};
        check(observe(
                  observation,
                  g1_footprint_budget(),
                  terrain,
                  stationary_command(0.62f, 0.95f),
                  contact_schedule(
                      true, true, true, true,
                      true, true, true, true),
                  pose) == G1FootprintOk,
              "multilevel threshold observation succeeds");
        check(observation.feet[0].maximum_root_split_m ==
                  static_cast<double>(heights[index]),
              "multilevel threshold preserves exact binary32 split");
        check(observation.feet[0].multilevel == (index == 0),
              "exact 0.04 split is multilevel and one ULP below is not");
    }
}

static void test_swept_interior_node_and_reversal_invariance()
{
    TerrainFixture terrain(81, 81, 0.0f, 0.0f, 0.025f);
    PoseFixture pose;
    pose.positions[G1_LeftAnkle] = vec3(0.75f, 0.08f, 0.80f);
    pose.positions[G1_RightAnkle] = vec3(0.75f, 0.08f, 1.10f);
    vec3 roots[] = {
        vec3(0.75f, 0.0f, 0.95f),
        vec3(0.90f, 0.0f, 0.95f),
        vec3(1.05f, 0.0f, 0.95f),
        vec3(1.20f, 0.0f, 0.95f),
    };
    quat rotations[] = {
        quat(1.0f, 0.0f, 0.0f, 0.0f),
        quat(1.0f, 0.0f, 0.0f, 0.0f),
        quat(1.0f, 0.0f, 0.0f, 0.0f),
        quat(1.0f, 0.0f, 0.0f, 0.0f),
    };
    const int ridge_x = 31;  // 0.775, strictly between samples 0 and 1.
    const int ridge_z = 31;  // 0.775, under the left rear probe envelope.
    terrain.field.heights(ridge_z * terrain.field.nx + ridge_x) = 0.25f;
    const G1CommandSnapshot command = make_command(roots, rotations);
    G1FootprintObservation forward = {};
    check(observe(
              forward,
              g1_footprint_budget(),
              terrain,
              command,
              contact_schedule(false, false, false, false),
              pose) == G1FootprintOk,
          "interior-node forward sweep succeeds");
    const G1FootprintProbe& probe = forward.feet[0].probes[0];
    for (int sample = 0; sample < 4; ++sample) {
        check(bits(probe.predicted_surfaces[sample].height) == bits(0.0f),
              "interior ridge is absent from all endpoint surface queries");
    }
    check(probe.corridor_maximum_height >= 0.25f,
          "swept envelope detects interior one-cell height change");

    PoseFixture reverse_pose;
    const vec3 displacement = roots[3] - roots[0];
    reverse_pose.positions[G1_LeftAnkle] =
        pose.positions[G1_RightAnkle] + displacement;
    reverse_pose.positions[G1_RightAnkle] =
        pose.positions[G1_LeftAnkle] + displacement;
    vec3 reverse_roots[] = {roots[3], roots[2], roots[1], roots[0]};
    const G1CommandSnapshot reverse_command =
        make_command(reverse_roots, rotations);
    G1FootprintObservation reverse = {};
    check(observe(
              reverse,
              g1_footprint_budget(),
              terrain,
              reverse_command,
              contact_schedule(false, false, false, false),
              reverse_pose) == G1FootprintOk,
          "mirrored reverse sweep succeeds");
    float forward_minimum = std::numeric_limits<float>::max();
    float reverse_minimum = std::numeric_limits<float>::max();
    float forward_maximum = 0.0f;
    float reverse_maximum = 0.0f;
    int forward_class = 1;
    int reverse_class = 1;
    double forward_split = 0.0;
    double reverse_split = 0.0;
    for (int foot = 0; foot < 2; ++foot) {
        forward_minimum = std::min(
            forward_minimum, forward.feet[foot].corridor_minimum_height);
        reverse_minimum = std::min(
            reverse_minimum, reverse.feet[foot].corridor_minimum_height);
        forward_maximum = std::max(
            forward_maximum, forward.feet[foot].corridor_maximum_height);
        reverse_maximum = std::max(
            reverse_maximum, reverse.feet[foot].corridor_maximum_height);
        forward_class = std::max(
            forward_class, forward.feet[foot].encountered_walkability_class);
        reverse_class = std::max(
            reverse_class, reverse.feet[foot].encountered_walkability_class);
        forward_split = std::max(
            forward_split, forward.feet[foot].maximum_root_split_m);
        reverse_split = std::max(
            reverse_split, reverse.feet[foot].maximum_root_split_m);
    }
    check(bits(forward_minimum) == bits(reverse_minimum) &&
              bits(forward_maximum) == bits(reverse_maximum) &&
              forward_class == reverse_class &&
              forward_split == reverse_split &&
              same_bytes(forward.work, reverse.work),
          "mirroring feet and reversing every segment preserves aggregate evidence");
}

static float independent_centroid_component(
    const G1FootprintFootObservation& foot,
    uint32_t sample,
    bool use_x)
{
    volatile double sum = 0.0;
    for (int probe = 0; probe < 4; ++probe) {
        const vec3 point = foot.probes[probe].predicted_sole_points[sample];
        sum = sum + static_cast<double>(use_x ? point.x : point.z);
    }
    const volatile double mean = sum / 4.0;
    return static_cast<float>(mean);
}

static void check_landing_case(bool down_step)
{
    TerrainFixture terrain(151, 151, 0.0f, 0.0f, 0.02f);
    terrain.set_x_step(0.80f, 0.32f);
    PoseFixture pose;
    vec3 roots[4];
    if (down_step) {
        pose.positions[G1_LeftAnkle] = vec3(0.65f, 0.40f, 0.80f);
        roots[0] = vec3(0.65f, 0.32f, 0.95f);
        roots[1] = vec3(0.75f, 0.32f, 0.95f);
        roots[2] = vec3(1.05f, 0.00f, 0.95f);
        roots[3] = vec3(1.20f, 0.00f, 0.95f);
    } else {
        pose.positions[G1_LeftAnkle] = vec3(1.15f, 0.08f, 0.80f);
        roots[0] = vec3(1.15f, 0.00f, 0.95f);
        roots[1] = vec3(1.00f, 0.00f, 0.95f);
        roots[2] = vec3(0.55f, 0.32f, 0.95f);
        roots[3] = vec3(0.40f, 0.32f, 0.95f);
    }
    pose.positions[G1_RightAnkle] = vec3(1.40f, 0.08f, 1.10f);
    quat rotations[4] = {
        quat(1.0f, 0.0f, 0.0f, 0.0f),
        quat(1.0f, 0.0f, 0.0f, 0.0f),
        quat(1.0f, 0.0f, 0.0f, 0.0f),
        quat(1.0f, 0.0f, 0.0f, 0.0f),
    };
    const G1CommandSnapshot command = make_command(roots, rotations);
    G1FootprintObservation observation = {};
    char error[256] = {};
    const G1FootprintStatus observed_status = observe(
        observation,
        g1_footprint_budget(),
        terrain,
        command,
        contact_schedule(false, false, true, true),
        pose,
        error,
        static_cast<int>(sizeof(error)));
    check(observed_status == G1FootprintOk, error[0] != '\0' ? error :
          (down_step ? "down-step landing observation succeeds" :
                       "up-step landing observation succeeds"));
    const G1FootprintFootObservation& foot = observation.feet[0];
    check(foot.landing_expected && foot.landing_sample == 2U,
          "first false-to-true edge selects exact sample two");
    check(foot.landing_patch_ready,
          "coplanar landing patch is ready");
    check(foot.predicted_landing_surface_status == G1SurfaceQueryValid &&
              bits(foot.predicted_landing_surface.height) ==
                  bits(down_step ? 0.0f : 0.32f) &&
              same_vec3_bits(
                  foot.predicted_landing_surface.normal,
                  vec3(0.0f, 1.0f, 0.0f)) &&
              foot.predicted_landing_walkability_class == 1,
          "landing publishes exact authoritative height normal and class");
    check(bits(foot.predicted_landing_sole_center.x) ==
              bits(independent_centroid_component(foot, 2U, true)) &&
              bits(foot.predicted_landing_sole_center.z) ==
              bits(independent_centroid_component(foot, 2U, false)),
          "landing centroid uses independently checked binary64 mean");
    check(foot.landing_patch_maximum_residual_m == 0.0,
          "coplanar landing has exact zero tangent-plane residual");
    for (int probe_index = 0; probe_index < 4; ++probe_index) {
        const G1FootprintProbe& landing_probe = foot.probes[probe_index];
        check(same_surface_bits(
                  landing_probe.selected_landing_surface,
                  landing_probe.predicted_surfaces[2]),
              "landing selects four already-sampled probe surfaces");
        check(same_vec3_bits(
                  landing_probe.predicted_sphere_centers[2],
                  landing_probe.predicted_sphere_centers[3]) &&
                  same_vec3_bits(
                      landing_probe.predicted_sole_points[2],
                      landing_probe.predicted_sole_points[3]),
              "post-edge contact remains locked to landing geometry");
    }
}

static void test_landing_edges_and_discontinuity()
{
    check_landing_case(true);
    check_landing_case(false);

    TerrainFixture terrain(151, 151, 0.0f, 0.0f, 0.02f);
    terrain.set_x_step(0.80f, 0.32f);
    PoseFixture pose;
    pose.positions[G1_LeftAnkle] = vec3(0.40f, 0.40f, 0.80f);
    pose.positions[G1_RightAnkle] = vec3(1.30f, 0.08f, 1.10f);
    vec3 roots[] = {
        vec3(0.40f, 0.32f, 0.95f),
        vec3(0.60f, 0.32f, 0.95f),
        vec3(0.80f, 0.00f, 0.95f),
        vec3(1.00f, 0.00f, 0.95f),
    };
    quat rotations[] = {
        quat(1.0f, 0.0f, 0.0f, 0.0f),
        quat(1.0f, 0.0f, 0.0f, 0.0f),
        quat(1.0f, 0.0f, 0.0f, 0.0f),
        quat(1.0f, 0.0f, 0.0f, 0.0f),
    };
    const G1CommandSnapshot command = make_command(roots, rotations);
    G1FootprintObservation observation = {};
    check(observe(
              observation,
              g1_footprint_budget(),
              terrain,
              command,
              contact_schedule(false, false, true, true),
              pose) == G1FootprintOk,
          "straddled landing observation remains an expected result");
    const G1FootprintFootObservation& landing = observation.feet[0];
    check(landing.landing_expected && landing.landing_sample == 2U &&
              !landing.landing_patch_ready &&
              landing.landing_patch_maximum_residual_m > 0.005 &&
              bits(landing.predicted_landing_surface.height) == bits(0.0f),
          "discontinuous landing is published unready without level averaging");

    observation = G1FootprintObservation{};
    check(observe(
              observation,
              g1_footprint_budget(),
              terrain,
              stationary_command(0.40f, 0.95f),
              contact_schedule(true, true, true, true),
              pose) == G1FootprintOk &&
              !observation.feet[0].landing_expected,
          "current contact with no rising edge suppresses landing selection");
    observation = G1FootprintObservation{};
    check(observe(
              observation,
              g1_footprint_budget(),
              terrain,
              stationary_command(0.40f, 0.95f),
              contact_schedule(false, false, false, false),
              pose) == G1FootprintOk &&
              !observation.feet[0].landing_expected,
          "schedule with no rising edge does not invent a landing");
}

static void test_exact_landing_residual_threshold()
{
    const float threshold = from_bits(UINT32_C(0x3ba3d70a));
    const float heights[] = {
        threshold,
        std::nextafter(threshold, std::numeric_limits<float>::infinity())
    };
    for (int index = 0; index < 2; ++index) {
        TerrainFixture terrain(151, 151, 0.0f, 0.0f, 0.02f);
        terrain.set_x_step(0.80f, heights[index]);
        PoseFixture pose;
        pose.positions[G1_LeftAnkle] = vec3(0.40f, 0.08f, 0.80f);
        pose.positions[G1_RightAnkle] = vec3(1.30f, 0.08f, 1.10f);
        vec3 roots[] = {
            vec3(0.40f, 0.0f, 0.95f),
            vec3(0.60f, 0.0f, 0.95f),
            vec3(0.80f, 0.0f, 0.95f),
            vec3(1.00f, 0.0f, 0.95f),
        };
        quat rotations[] = {
            quat(1.0f, 0.0f, 0.0f, 0.0f),
            quat(1.0f, 0.0f, 0.0f, 0.0f),
            quat(1.0f, 0.0f, 0.0f, 0.0f),
            quat(1.0f, 0.0f, 0.0f, 0.0f),
        };
        G1FootprintObservation observation = {};
        check(observe(
                  observation,
                  g1_footprint_budget(),
                  terrain,
                  make_command(roots, rotations),
                  contact_schedule(false, false, true, true),
                  pose) == G1FootprintOk,
              "residual threshold landing observation succeeds");
        const G1FootprintFootObservation& landing = observation.feet[0];
        check(landing.landing_expected && landing.landing_sample == 2U,
              "residual threshold fixture selects landing edge");
        check(landing.landing_patch_maximum_residual_m ==
                  static_cast<double>(heights[index]),
              "residual threshold fixture preserves exact binary32 height");
        check(landing.landing_patch_ready == (index == 0),
              "exact 0.005 residual is ready and one ULP above is unready");
    }
}

static void test_landing_centroid_walkability_distinguishes_malformed()
{
    TerrainFixture terrain;
    PoseFixture pose;
    const G1CommandSnapshot command = stationary_command();
    const G1FootContactSchedule landing_contacts =
        contact_schedule(false, false, true, true);
    const int centroid_x = 42;
    const int centroid_z = 40;
    const int centroid_index = centroid_z * terrain.grid.nx + centroid_x;

    terrain.grid.cells(centroid_index) = 3;
    G1FootprintObservation malformed_output;
    poison(malformed_output);
    const G1FootprintObservation malformed_before = malformed_output;
    char error[256] = {};
    const G1FootprintStatus malformed_status = observe(
        malformed_output,
        g1_footprint_budget(),
        terrain,
        command,
        landing_contacts,
        pose,
        error,
        static_cast<int>(sizeof(error)));
    check(malformed_status == G1FootprintInvalidField,
          "centroid-only malformed walkability value is a field failure");
    check(same_bytes(malformed_output, malformed_before),
          "centroid-only malformed walkability preserves poisoned output");
    check(std::strstr(error, "centroid") != NULL &&
              std::strstr(error, "walkability") != NULL,
          "centroid-only malformed walkability has a useful diagnostic");

    terrain.grid.cells(centroid_index) = 0;
    G1FootprintObservation blocked = {};
    std::memset(error, 0, sizeof(error));
    check(observe(
              blocked,
              g1_footprint_budget(),
              terrain,
              command,
              landing_contacts,
              pose,
              error,
              static_cast<int>(sizeof(error))) == G1FootprintOk,
          "legitimate class-zero landing centroid remains an Ok observation");
    check(blocked.blocked &&
              blocked.blocked_reason == walkability_blocked_cell &&
              blocked.feet[0].landing_expected &&
              blocked.feet[0].predicted_landing_walkability_class == 0 &&
              !blocked.feet[0].landing_patch_ready &&
              blocked.work.sweeps == 24U &&
              blocked.work.surface_queries == 34U &&
              blocked.work.node_visits == 864U,
          "legitimate blocked centroid remains distinct with fixed work");
}

static void test_failure_budget_alias_and_blocked_matrix()
{
    TerrainFixture terrain;
    PoseFixture pose;
    const G1CommandSnapshot command = stationary_command();
    const G1FootContactSchedule contacts =
        contact_schedule(false, false, false, false);
    G1FootprintObservation baseline = {};
    check(observe(
              baseline,
              g1_footprint_budget(),
              terrain,
              command,
              contacts,
              pose) == G1FootprintOk,
          "baseline work-count observation succeeds");
    check(baseline.work.sweeps == 24U &&
              baseline.work.surface_queries == 33U &&
              baseline.work.node_visits > 0U &&
              baseline.work.node_visits <= 65536U,
          "baseline has exact fixed work bounds");

    G1FootprintObservation maximum_query_work = {};
    check(observe(
              maximum_query_work,
              g1_footprint_budget(),
              terrain,
              command,
              contact_schedule(
                  false, false, true, true,
                  false, false, true, true),
              pose) == G1FootprintOk &&
              maximum_query_work.work.surface_queries == 35U,
          "two selected landing centers reach exact 35-query maximum");

    G1FootprintBudget exact = {};
    exact.maximum_sweeps = baseline.work.sweeps;
    exact.maximum_surface_queries = baseline.work.surface_queries;
    exact.maximum_node_visits = baseline.work.node_visits;
    G1FootprintObservation exact_output = {};
    check(observe(
              exact_output,
              exact,
              terrain,
              command,
              contacts,
              pose) == G1FootprintOk &&
              same_bytes(exact_output.work, baseline.work),
          "exact aggregate budget succeeds");

    G1FootprintBudget budgets[] = {exact, exact, exact};
    --budgets[0].maximum_sweeps;
    --budgets[1].maximum_surface_queries;
    --budgets[2].maximum_node_visits;
    for (const G1FootprintBudget& budget : budgets) {
        G1FootprintObservation output;
        poison(output);
        const G1FootprintObservation before = output;
        require_observation_failure_preserves(
            output,
            before,
            observe(output, budget, terrain, command, contacts, pose),
            G1FootprintBudgetExceeded,
            "one-below aggregate budget is transactional");
    }

    const int blocked_index = 39 * terrain.grid.nx + 38;
    terrain.grid.cells(blocked_index) = 0;
    G1FootprintObservation blocked = {};
    check(observe(
              blocked,
              g1_footprint_budget(),
              terrain,
              command,
              contacts,
              pose) == G1FootprintOk &&
              blocked.blocked &&
              blocked.blocked_reason == walkability_blocked_cell &&
              blocked.work.sweeps == 24U,
          "blocked cells retain first reason while completing checked work");
    terrain.grid.cells(blocked_index) = 1;

    G1FootprintObservation output;
    poison(output);
    G1FootprintObservation before = output;
    const uint32_t saved_version = terrain.field.version;
    terrain.field.version = 1;
    require_observation_failure_preserves(
        output,
        before,
        observe(output, g1_footprint_budget(), terrain, command, contacts, pose),
        G1FootprintInvalidField,
        "non-v2 field is transactional");
    terrain.field.version = saved_version;

    poison(output);
    before = output;
    const int saved_height_size = terrain.field.heights.size;
    terrain.field.heights.size = -1;
    require_observation_failure_preserves(
        output,
        before,
        observe(output, g1_footprint_budget(), terrain, command, contacts, pose),
        G1FootprintInvalidField,
        "malformed height storage shape is transactional field failure");
    terrain.field.heights.size = saved_height_size;

    poison(output);
    before = output;
    const int saved_grid_nx = terrain.grid.nx;
    terrain.grid.nx = saved_grid_nx - 1;
    require_observation_failure_preserves(
        output,
        before,
        observe(output, g1_footprint_budget(), terrain, command, contacts, pose),
        G1FootprintInvalidField,
        "malformed walkability grid is transactional");
    terrain.grid.nx = saved_grid_nx;

    poison(output);
    before = output;
    const uint8_t saved_cell = terrain.grid.cells(blocked_index);
    terrain.grid.cells(blocked_index) = 3;
    require_observation_failure_preserves(
        output,
        before,
        observe(output, g1_footprint_budget(), terrain, command, contacts, pose),
        G1FootprintInvalidField,
        "malformed visited walkability cell is transactional");
    terrain.grid.cells(blocked_index) = saved_cell;

    poison(output);
    before = output;
    const float saved_height = terrain.field.heights(blocked_index);
    terrain.field.heights(blocked_index) = from_bits(UINT32_C(0x7fc00001));
    require_observation_failure_preserves(
        output,
        before,
        observe(output, g1_footprint_budget(), terrain, command, contacts, pose),
        G1FootprintInvalidField,
        "visited nonfinite height is transactional");
    terrain.field.heights(blocked_index) = saved_height;

    G1CommandSnapshot invalid_command = command;
    invalid_command.predicted_root_positions[1].x =
        std::numeric_limits<float>::infinity();
    poison(output);
    before = output;
    require_observation_failure_preserves(
        output,
        before,
        observe(
            output,
            g1_footprint_budget(),
            terrain,
            invalid_command,
            contacts,
            pose),
        G1FootprintInvalidInput,
        "nonfinite command is transactional");

    const vec3 saved_pose = pose.positions[G1_LeftAnkle];
    pose.positions[G1_LeftAnkle].x =
        std::numeric_limits<float>::quiet_NaN();
    poison(output);
    before = output;
    require_observation_failure_preserves(
        output,
        before,
        observe(output, g1_footprint_budget(), terrain, command, contacts, pose),
        G1FootprintInvalidInput,
        "nonfinite pose is transactional");
    pose.positions[G1_LeftAnkle] = saved_pose;

    const quat saved_rotation = pose.rotations[G1_LeftAnkle];
    pose.rotations[G1_LeftAnkle] = quat(2.0f, 0.0f, 0.0f, 0.0f);
    poison(output);
    before = output;
    require_observation_failure_preserves(
        output,
        before,
        observe(output, g1_footprint_budget(), terrain, command, contacts, pose),
        G1FootprintInvalidInput,
        "nonunit pose rotation is transactional");
    pose.rotations[G1_LeftAnkle] = saved_rotation;

    PoseFixture arithmetic_pose;
    arithmetic_pose.positions[G1_LeftAnkle].x =
        std::numeric_limits<float>::max();
    G1CommandSnapshot arithmetic_command = stationary_command();
    for (int sample = 0; sample < 4; ++sample) {
        arithmetic_command.predicted_root_positions[sample].x =
            -std::numeric_limits<float>::max();
    }
    poison(output);
    before = output;
    require_observation_failure_preserves(
        output,
        before,
        observe(
            output,
            g1_footprint_budget(),
            terrain,
            arithmetic_command,
            contacts,
            arithmetic_pose),
        G1FootprintArithmeticFailure,
        "root transport arithmetic overflow is transactional");

    PoseFixture boundary_pose;
    boundary_pose.positions[G1_LeftAnkle] = vec3(0.07f, 0.08f, 0.80f);
    boundary_pose.positions[G1_RightAnkle] = vec3(0.30f, 0.08f, 1.10f);
    vec3 boundary_roots[] = {
        vec3(0.07f, 0.0f, 0.95f),
        vec3(std::nextafter(0.07f, -std::numeric_limits<float>::infinity()),
             0.0f, 0.95f),
        vec3(0.07f, 0.0f, 0.95f),
        vec3(0.07f, 0.0f, 0.95f),
    };
    quat identities[] = {
        quat(1.0f, 0.0f, 0.0f, 0.0f),
        quat(1.0f, 0.0f, 0.0f, 0.0f),
        quat(1.0f, 0.0f, 0.0f, 0.0f),
        quat(1.0f, 0.0f, 0.0f, 0.0f),
    };
    const G1CommandSnapshot boundary_command =
        make_command(boundary_roots, identities);
    poison(output);
    before = output;
    require_observation_failure_preserves(
        output,
        before,
        observe(
            output,
            g1_footprint_budget(),
            terrain,
            boundary_command,
            contacts,
            boundary_pose),
        G1FootprintOutsideDomain,
        "one-ULP footprint excursion outside domain is transactional");

    poison(output);
    before = output;
    check(g1_footprint_observe_v2(
              output,
              g1_footprint_budget(),
              terrain.field,
              terrain.grid,
              command,
              contacts,
              pose.position_slice(),
              pose.rotation_slice(),
              reinterpret_cast<char*>(&output) + 3,
              4) == G1FootprintInvalidInput,
          "observation diagnostic/output alias is rejected");
    check(same_bytes(output, before),
          "observation diagnostic/output alias preserves poison");

    poison(output);
    before = output;
    check(g1_footprint_observe_v2(
              output,
              g1_footprint_budget(),
              terrain.field,
              terrain.grid,
              command,
              contacts,
              slice1d<vec3>(
                  G1_BoneCount,
                  reinterpret_cast<vec3*>(&output)),
              pose.rotation_slice(),
              NULL,
              0) == G1FootprintInvalidInput,
          "observation position/output storage alias is rejected");
    check(same_bytes(output, before),
          "observation position/output alias preserves poison");

    poison(output);
    before = output;
    check(g1_footprint_observe_v2(
              output,
              g1_footprint_budget(),
              terrain.field,
              terrain.grid,
              command,
              contacts,
              pose.position_slice(),
              slice1d<quat>(
                  G1_BoneCount,
                  reinterpret_cast<quat*>(&output)),
              NULL,
              0) == G1FootprintInvalidInput,
          "observation rotation/output storage alias is rejected");
    check(same_bytes(output, before),
          "observation rotation/output alias preserves poison");

    poison(output);
    before = output;
    check(g1_footprint_observe_v2(
              output,
              g1_footprint_budget(),
              terrain.field,
              terrain.grid,
              command,
              contacts,
              slice1d<vec3>(G1_BoneCount - 1, pose.positions),
              pose.rotation_slice(),
              NULL,
              0) == G1FootprintInvalidInput,
          "short pose shape is rejected");
    check(same_bytes(output, before),
          "short pose shape preserves poison");

    poison(output);
    before = output;
    check(g1_footprint_observe_v2(
              output,
              g1_footprint_budget(),
              terrain.field,
              terrain.grid,
              command,
              contacts,
              slice1d<vec3>(G1_BoneCount, NULL),
              pose.rotation_slice(),
              NULL,
              0) == G1FootprintInvalidInput,
          "null pose storage is rejected");
    check(same_bytes(output, before),
          "null pose storage preserves poison");
}

static void test_public_defaults_and_parity(bool parity)
{
    const G1FootprintBudget budget = g1_footprint_budget();
    check(budget.maximum_sweeps == 24U &&
              budget.maximum_surface_queries == 35U &&
              budget.maximum_node_visits == 65536U,
          "public default budgets are exact");

    if (parity) {
        TerrainFixture terrain;
        PoseFixture pose;
        G1FootprintObservation observation = {};
        check(observe(
                  observation,
                  budget,
                  terrain,
                  stationary_command(),
                  contact_schedule(false, false, true, true),
                  pose) == G1FootprintOk,
              "parity observation succeeds");
        std::printf(
            "footprint-v1 sweeps=%u queries=%u nodes=%u root=%08x "
            "landing=%u ready=%d residual=%.17g\n",
            observation.work.sweeps,
            observation.work.surface_queries,
            observation.work.node_visits,
            bits(observation.root_surface.height),
            observation.feet[0].landing_sample,
            observation.feet[0].landing_patch_ready ? 1 : 0,
            observation.feet[0].landing_patch_maximum_residual_m);
    }
}

int main(int argc, char** argv)
{
    const bool parity = argc == 2 && std::strcmp(argv[1], "--parity") == 0;
    check(argc == 1 || parity, "only optional --parity argument is accepted");
    static_assert(G1CommandTrajectorySampleCount == 4,
                  "footprint contract requires four command samples");
    test_contact_schedule_exact_horizons_and_transaction();
    test_six_heading_transport_and_phase_anchors();
    test_lateral_split_and_centerline_invariance();
    test_exact_multilevel_threshold();
    test_swept_interior_node_and_reversal_invariance();
    test_landing_edges_and_discontinuity();
    test_exact_landing_residual_threshold();
    test_landing_centroid_walkability_distinguishes_malformed();
    test_failure_budget_alias_and_blocked_matrix();
    test_public_defaults_and_parity(parity);
    return 0;
}
