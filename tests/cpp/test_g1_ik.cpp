#include "g1_ik_runtime.h"

#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>

static_assert(G1SwingLiftCandidateCount == 41,
              "runtime owns the exact 41-stage swing ladder");
static_assert(G1SwingNoCandidate == UINT32_MAX,
              "runtime owns the no-candidate sentinel");

static constexpr uint32_t g1_test_exact_rational_binary32(
    uint32_t numerator,
    uint32_t denominator)
{
    if (numerator == 0U) return 0U;
    uint64_t normalized_numerator = numerator;
    int exponent = 0;
    while (normalized_numerator < denominator) {
        normalized_numerator <<= 1U;
        --exponent;
    }
    const uint32_t shift = static_cast<uint32_t>(23 - exponent);
    const uint64_t scaled =
        static_cast<uint64_t>(numerator) << shift;
    uint64_t significand = scaled / denominator;
    const uint64_t remainder = scaled % denominator;
    const uint64_t twice_remainder = remainder * 2U;
    if (twice_remainder > denominator ||
        (twice_remainder == denominator &&
         (significand & UINT64_C(1)) != 0U)) {
        ++significand;
    }
    if (significand == (UINT64_C(1) << 24U)) {
        significand >>= 1U;
        ++exponent;
    }
    return (static_cast<uint32_t>(exponent + 127) << 23U) |
           static_cast<uint32_t>(
               significand - (UINT64_C(1) << 23U));
}

static_assert(g1_test_exact_rational_binary32(1U, 500U) ==
                  UINT32_C(0x3b03126f) &&
              g1_test_exact_rational_binary32(40U, 500U) ==
                  UINT32_C(0x3da3d70a),
              "independent rational-to-binary32 oracle is exact at bounds");

static void check(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "G1 IK test failed: %s\n", message);
        std::exit(1);
    }
}

static void set_mapped_leg_offsets(database& db)
{
    for (int frame = 0; frame < db.bone_positions.rows; ++frame) {
        db.bone_positions(frame, G1_LeftKnee) =
            vec3(-0.078273f, -0.17734f, -0.0021489f);
        db.bone_positions(frame, G1_LeftAnkle) =
            vec3(0.0f, -0.30001f, +0.000094445f);
        db.bone_positions(frame, G1_LeftToe) =
            vec3(0.0f, -0.017558f, 0.0f);
        db.bone_positions(frame, G1_RightKnee) =
            vec3(-0.078273f, -0.17734f, +0.0021489f);
        db.bone_positions(frame, G1_RightAnkle) =
            vec3(0.0f, -0.30001f, -0.000094445f);
        db.bone_positions(frame, G1_RightToe) =
            vec3(0.0f, -0.017558f, 0.0f);
    }
}

static void set_legacy_xml_leg_offsets(database& db, int frame)
{
    db.bone_positions(frame, G1_LeftKnee) =
        vec3(-0.078273f, +0.0021489f, -0.17734f);
    db.bone_positions(frame, G1_LeftAnkle) =
        vec3(0.0f, -0.000094445f, -0.30001f);
    db.bone_positions(frame, G1_LeftToe) =
        vec3(0.0f, 0.0f, -0.017558f);
    db.bone_positions(frame, G1_RightKnee) =
        vec3(-0.078273f, -0.0021489f, -0.17734f);
    db.bone_positions(frame, G1_RightAnkle) =
        vec3(0.0f, +0.000094445f, -0.30001f);
    db.bone_positions(frame, G1_RightToe) =
        vec3(0.0f, 0.0f, -0.017558f);
}

static void make_g1_database(database& db, int frames = 1)
{
    static const int parents[G1_BoneCount] = {
        -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
    };
    db.bone_positions.resize(frames, G1_BoneCount);
    db.bone_rotations.resize(frames, G1_BoneCount);
    db.bone_parents.resize(G1_BoneCount);
    db.bone_positions.set(vec3());
    db.bone_rotations.set(quat());
    for (int i = 0; i < G1_BoneCount; ++i) {
        db.bone_parents(i) = parents[i];
    }
    set_mapped_leg_offsets(db);
}

static void check_vec3(vec3 actual, vec3 expected, const char* message)
{
    check(actual.x == expected.x &&
          actual.y == expected.y &&
          actual.z == expected.z,
          message);
}

static void check_error_contains(
    const char* error, const char* expected, const char* message)
{
    check(std::strstr(error, expected) != NULL, message);
}

static void check_config_rejected(
    const database& db,
    const G1LegConfig& config,
    const char* expected_error,
    const char* message)
{
    char error[256] = {};
    check(!g1_leg_config_validate(
              db, config, error, static_cast<int>(sizeof(error))),
          message);
    check_error_contains(error, expected_error, message);
}

static void test_explicit_leg_geometry()
{
    const G1LegConfig left = g1_left_leg_config();
    const G1LegConfig right = g1_right_leg_config();
    check(std::strcmp(left.name, "left") == 0 &&
          std::strcmp(right.name, "right") == 0,
          "leg names");
    check(left.hip == G1_LeftHipYaw && left.knee == G1_LeftKnee,
          "left hip/knee names");
    check(left.ankle == G1_LeftAnkle && left.contact == G1_LeftToe,
          "left ankle/contact names");
    check(right.hip == G1_RightHipYaw && right.knee == G1_RightKnee,
          "right hip/knee names");
    check(right.ankle == G1_RightAnkle && right.contact == G1_RightToe,
          "right ankle/contact names");

    const G1LegConfig configs[2] = {left, right};
    const vec3 sphere_centers[4] = {
        vec3(-0.05f, -0.03f, -0.025f),
        vec3(-0.05f, -0.03f, +0.025f),
        vec3(+0.12f, -0.03f, -0.030f),
        vec3(+0.12f, -0.03f, +0.030f)
    };
    for (int leg = 0; leg < 2; ++leg) {
        const G1LegConfig& config = configs[leg];
        check_vec3(
            config.knee_hinge_axis_local,
            vec3(0.0f, 0.0f, -1.0f),
            "MuJoCo +Y knee hinge maps to Holden -Z");
        check_vec3(
            config.foot_forward_local,
            vec3(1.0f, 0.0f, 0.0f),
            "MuJoCo +X foot forward maps to Holden +X");
        check_vec3(
            config.sole_normal_local,
            vec3(0.0f, 1.0f, 0.0f),
            "MuJoCo +Z sole normal maps to Holden +Y");
        for (int probe = 0; probe < 4; ++probe) {
            check_vec3(
                config.foot_sphere_centers_local[probe],
                sphere_centers[probe],
                "mapped XML foot-sphere center");
            check_vec3(
                config.sole_points_local[probe],
                sphere_centers[probe] -
                    vec3(0.0f, 1.0f, 0.0f) * 0.02f,
                "sphere-bottom sole probe");
        }
        check(config.foot_sphere_radius_m == 0.02f,
              "XML foot-sphere radius");
        check_vec3(
            config.thigh_start_local,
            vec3(0.0f, -0.02f, 0.0f),
            "mapped XML thigh start");
        check_vec3(
            config.thigh_end_local,
            vec3(-0.078f, -0.17f, 0.0f),
            "mapped XML thigh end");
        check(config.thigh_radius_m == 0.05f, "XML thigh radius");
        check_vec3(
            config.shin_start_local,
            vec3(0.0f, -0.05f, 0.0f),
            "mapped XML shin start");
        check_vec3(
            config.shin_end_local,
            vec3(0.0f, -0.28f, 0.0f),
            "mapped XML shin end");
        check(config.shin_radius_m == 0.04f, "XML shin radius");
        check(config.reach_buffer_m == 0.015f &&
              config.planted_clearance_m == 0.005f &&
              config.swing_clearance_m == 0.015f &&
              config.max_swing_lift_m == 0.08f &&
              config.max_correction_radians == 0.35f,
              "IK bounds");
    }

    database db;
    make_g1_database(db);
    char error[256] = {};
    check(g1_leg_configs_validate(
              db, error, static_cast<int>(sizeof(error))),
          error);
}

static void test_database_shape_and_chain_validation()
{
    char error[256] = {};

    database left_hip;
    make_g1_database(left_hip);
    left_hip.bone_parents(G1_LeftHipYaw) = G1_Hips;
    check(!g1_leg_configs_validate(
              left_hip, error, static_cast<int>(sizeof(error))),
          "wrong left named hip chain rejected");
    check_error_contains(error, "LeftHipYaw", "left hip diagnostic");

    database right_hip;
    make_g1_database(right_hip);
    right_hip.bone_parents(G1_RightHipYaw) = G1_Hips;
    check(!g1_leg_configs_validate(
              right_hip, error, static_cast<int>(sizeof(error))),
          "wrong right named hip chain rejected");
    check_error_contains(error, "RightHipYaw", "right hip diagnostic");

    database left_knee;
    make_g1_database(left_knee);
    left_knee.bone_parents(G1_LeftKnee) = G1_LeftHipRoll;
    check(!g1_leg_configs_validate(
              left_knee, error, static_cast<int>(sizeof(error))),
          "wrong left named chain rejected");
    check_error_contains(error, "LeftKnee", "left knee diagnostic");

    database right_knee;
    make_g1_database(right_knee);
    right_knee.bone_parents(G1_RightKnee) = G1_RightHipRoll;
    check(!g1_leg_configs_validate(
              right_knee, error, static_cast<int>(sizeof(error))),
          "wrong right named chain rejected");
    check_error_contains(error, "RightKnee", "right knee diagnostic");

    database ankle;
    make_g1_database(ankle);
    ankle.bone_parents(G1_LeftAnkle) = G1_LeftHipYaw;
    check(!g1_leg_configs_validate(
              ankle, error, static_cast<int>(sizeof(error))),
          "wrong ankle chain rejected");
    check_error_contains(error, "ankle", "ankle diagnostic");

    database contact;
    make_g1_database(contact);
    contact.bone_parents(G1_RightToe) = G1_RightKnee;
    check(!g1_leg_configs_validate(
              contact, error, static_cast<int>(sizeof(error))),
          "wrong contact chain rejected");
    check_error_contains(error, "contact", "contact diagnostic");

    database unrelated_parent;
    make_g1_database(unrelated_parent);
    unrelated_parent.bone_parents(G1_Spine1) = G1_Hips;
    check(!g1_leg_configs_validate(
              unrelated_parent, error, static_cast<int>(sizeof(error))),
          "non-leg skeleton mismatch rejected");
    check_error_contains(error, "parent", "full parent contract diagnostic");

    database position_columns;
    make_g1_database(position_columns);
    position_columns.bone_positions.resize(1, G1_BoneCount - 1);
    check(!g1_leg_configs_validate(
              position_columns, error, static_cast<int>(sizeof(error))),
          "position column mismatch rejected");
    check_error_contains(error, "position", "position shape diagnostic");

    database rotation_columns;
    make_g1_database(rotation_columns);
    rotation_columns.bone_rotations.resize(1, G1_BoneCount - 1);
    check(!g1_leg_configs_validate(
              rotation_columns, error, static_cast<int>(sizeof(error))),
          "rotation column mismatch rejected");
    check_error_contains(error, "rotation", "rotation shape diagnostic");

    database rotation_rows;
    make_g1_database(rotation_rows);
    rotation_rows.bone_rotations.resize(2, G1_BoneCount);
    check(!g1_leg_configs_validate(
              rotation_rows, error, static_cast<int>(sizeof(error))),
          "rotation row mismatch rejected");
    check_error_contains(error, "rotation", "rotation row diagnostic");

    database no_frames;
    make_g1_database(no_frames);
    no_frames.bone_positions.resize(0, 0);
    check(!g1_leg_configs_validate(
              no_frames, error, static_cast<int>(sizeof(error))),
          "empty position data rejected");
    check_error_contains(error, "position", "empty position diagnostic");

    database parent_shape;
    make_g1_database(parent_shape);
    parent_shape.bone_parents.resize(G1_BoneCount - 1);
    check(!g1_leg_configs_validate(
              parent_shape, error, static_cast<int>(sizeof(error))),
          "parent shape rejected");
    check_error_contains(error, "parent", "parent shape diagnostic");

    database no_rotations;
    make_g1_database(no_rotations);
    no_rotations.bone_rotations.resize(0, 0);
    check(!g1_leg_configs_validate(
              no_rotations, error, static_cast<int>(sizeof(error))),
          "empty rotation data rejected");
    check_error_contains(error, "rotation", "empty rotation diagnostic");

    database null_positions;
    make_g1_database(null_positions);
    vec3* const saved_positions = null_positions.bone_positions.data;
    null_positions.bone_positions.data = NULL;
    const bool positions_rejected = !g1_leg_configs_validate(
        null_positions, error, static_cast<int>(sizeof(error)));
    null_positions.bone_positions.data = saved_positions;
    check(positions_rejected, "null position pointer rejected");
    check_error_contains(error, "position", "null position diagnostic");

    database null_rotations;
    make_g1_database(null_rotations);
    quat* const saved_rotations = null_rotations.bone_rotations.data;
    null_rotations.bone_rotations.data = NULL;
    const bool rotations_rejected = !g1_leg_configs_validate(
        null_rotations, error, static_cast<int>(sizeof(error)));
    null_rotations.bone_rotations.data = saved_rotations;
    check(rotations_rejected, "null rotation pointer rejected");
    check_error_contains(error, "rotation", "null rotation diagnostic");

    database null_parents;
    make_g1_database(null_parents);
    int* const saved_parents = null_parents.bone_parents.data;
    null_parents.bone_parents.data = NULL;
    const bool parents_rejected = !g1_leg_configs_validate(
        null_parents, error, static_cast<int>(sizeof(error)));
    null_parents.bone_parents.data = saved_parents;
    check(parents_rejected, "null parent pointer rejected");
    check_error_contains(error, "parent", "null parent diagnostic");
}

static void test_database_local_basis_validation()
{
    char error[256] = {};

    database mapped;
    make_g1_database(mapped, 2);
    check(g1_leg_configs_validate(
              mapped, error, static_cast<int>(sizeof(error))),
          "mapped v2 local basis accepted");

    database legacy;
    make_g1_database(legacy);
    set_legacy_xml_leg_offsets(legacy, 0);
    check(!g1_leg_configs_validate(
              legacy, error, static_cast<int>(sizeof(error))),
          "legacy XML-local basis rejected");
    check_error_contains(error, "local basis", "legacy basis diagnostic");
    check_error_contains(error, "frame 0", "legacy frame diagnostic");
    check_error_contains(error, "LeftKnee", "legacy bone diagnostic");

    database later_frame;
    make_g1_database(later_frame, 2);
    later_frame.bone_positions(1, G1_RightToe) =
        vec3(0.0f, 0.0f, -0.017558f);
    check(!g1_leg_configs_validate(
              later_frame, error, static_cast<int>(sizeof(error))),
          "legacy basis in a later frame rejected");
    check_error_contains(error, "local basis", "later basis diagnostic");
    check_error_contains(error, "frame 1", "later frame diagnostic");
    check_error_contains(error, "RightToe", "later bone diagnostic");

    database nonfinite;
    make_g1_database(nonfinite);
    nonfinite.bone_positions(0, G1_LeftAnkle).x =
        std::numeric_limits<float>::quiet_NaN();
    check(!g1_leg_configs_validate(
              nonfinite, error, static_cast<int>(sizeof(error))),
          "non-finite local basis rejected");
    check_error_contains(error, "local basis", "non-finite basis diagnostic");
    check_error_contains(error, "LeftAnkle", "non-finite bone diagnostic");
}

static void test_hostile_config_validation()
{
    database db;
    make_g1_database(db);
    const G1LegConfig valid = g1_left_leg_config();
    char error[256] = {};
    check(g1_leg_config_validate(
              db, valid, error, static_cast<int>(sizeof(error))),
          "valid standalone config");
    const G1LegConfig valid_right = g1_right_leg_config();
    check(g1_leg_config_validate(
              db, valid_right, error, static_cast<int>(sizeof(error))),
          "valid right standalone config");

    G1LegConfig bad_name = valid;
    bad_name.name = NULL;
    check_config_rejected(db, bad_name, "name", "null name rejected");
    bad_name.name = "";
    check_config_rejected(db, bad_name, "name", "empty name rejected");
    bad_name.name = "LEFT";
    check_config_rejected(db, bad_name, "name", "malformed name rejected");
    bad_name.name = "right";
    check_config_rejected(
        db, bad_name, "bone contract", "name/bone side mismatch rejected");

    int G1LegConfig::* const index_members[] = {
        &G1LegConfig::hip,
        &G1LegConfig::knee,
        &G1LegConfig::ankle,
        &G1LegConfig::contact
    };
    for (size_t i = 0;
         i < sizeof(index_members) / sizeof(index_members[0]);
         ++i) {
        G1LegConfig negative = valid;
        negative.*index_members[i] = -1;
        check_config_rejected(
            db, negative, "bone contract", "negative bone index rejected");
        G1LegConfig too_large = valid;
        too_large.*index_members[i] = G1_BoneCount;
        check_config_rejected(
            db, too_large, "bone contract", "large bone index rejected");
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();

    G1LegConfig nonfinite_axis = valid;
    nonfinite_axis.knee_hinge_axis_local.x = nan;
    check_config_rejected(
        db, nonfinite_axis, "non-finite", "non-finite hinge axis rejected");

    G1LegConfig nonfinite_center = valid;
    nonfinite_center.foot_sphere_centers_local[2].z = infinity;
    check_config_rejected(
        db, nonfinite_center, "non-finite", "non-finite sphere rejected");

    G1LegConfig nonfinite_sole = valid;
    nonfinite_sole.sole_points_local[3].y = nan;
    check_config_rejected(
        db, nonfinite_sole, "non-finite", "non-finite sole rejected");

    G1LegConfig nonfinite_capsule = valid;
    nonfinite_capsule.shin_end_local.x = infinity;
    check_config_rejected(
        db, nonfinite_capsule, "non-finite", "non-finite capsule rejected");

    G1LegConfig nonfinite_bound = valid;
    nonfinite_bound.max_correction_radians = nan;
    check_config_rejected(
        db, nonfinite_bound, "non-finite", "non-finite bound rejected");

    G1LegConfig nonunit_axis = valid;
    nonunit_axis.knee_hinge_axis_local = vec3(0.0f, 0.0f, -0.5f);
    check_config_rejected(
        db, nonunit_axis, "unit", "non-unit hinge axis rejected");

    G1LegConfig nonorthogonal_axes = valid;
    nonorthogonal_axes.foot_forward_local =
        nonorthogonal_axes.sole_normal_local;
    check_config_rejected(
        db, nonorthogonal_axes, "orthogonal", "non-orthogonal axes rejected");

    G1LegConfig wrong_mapped_axis = valid;
    wrong_mapped_axis.knee_hinge_axis_local = vec3(0.0f, 0.0f, 1.0f);
    check_config_rejected(
        db, wrong_mapped_axis, "basis", "wrong mapped hinge axis rejected");

    float G1LegConfig::* const positive_members[] = {
        &G1LegConfig::foot_sphere_radius_m,
        &G1LegConfig::thigh_radius_m,
        &G1LegConfig::shin_radius_m,
        &G1LegConfig::reach_buffer_m,
        &G1LegConfig::planted_clearance_m,
        &G1LegConfig::swing_clearance_m,
        &G1LegConfig::max_swing_lift_m,
        &G1LegConfig::max_correction_radians
    };
    for (size_t i = 0;
         i < sizeof(positive_members) / sizeof(positive_members[0]);
         ++i) {
        G1LegConfig zero = valid;
        zero.*positive_members[i] = 0.0f;
        check_config_rejected(
            db, zero, "positive", "zero geometry scalar rejected");
        G1LegConfig negative = valid;
        negative.*positive_members[i] = -0.01f;
        check_config_rejected(
            db, negative, "positive", "negative geometry scalar rejected");
    }

    G1LegConfig degenerate_thigh = valid;
    degenerate_thigh.thigh_end_local = degenerate_thigh.thigh_start_local;
    check_config_rejected(
        db, degenerate_thigh, "capsule", "degenerate thigh rejected");

    G1LegConfig degenerate_shin = valid;
    degenerate_shin.shin_end_local = degenerate_shin.shin_start_local;
    check_config_rejected(
        db, degenerate_shin, "capsule", "degenerate shin rejected");

    G1LegConfig invalid_probe = valid;
    invalid_probe.sole_points_local[0].y += 0.001f;
    check_config_rejected(
        db, invalid_probe, "sole", "inconsistent sole probe rejected");

    G1LegConfig duplicate_probe = valid;
    duplicate_probe.foot_sphere_centers_local[1] =
        duplicate_probe.foot_sphere_centers_local[0];
    duplicate_probe.sole_points_local[1] =
        duplicate_probe.sole_points_local[0];
    check_config_rejected(
        db, duplicate_probe, "sole", "duplicate sole probe rejected");

    G1LegConfig nonplanar_probe = valid;
    nonplanar_probe.foot_sphere_centers_local[1].y += 0.001f;
    nonplanar_probe.sole_points_local[1].y += 0.001f;
    check_config_rejected(
        db, nonplanar_probe, "sole", "non-planar sole probe rejected");

    G1LegConfig hostile_right = valid_right;
    hostile_right.knee_hinge_axis_local = vec3();
    check_config_rejected(
        db, hostile_right, "unit", "right hostile axis rejected");
}

static void test_null_and_short_error_buffers()
{
    database db;
    make_g1_database(db);
    db.bone_parents(G1_LeftKnee) = G1_LeftHipRoll;

    check(!g1_leg_configs_validate(db, NULL, 0),
          "null error buffer remains safe");

    char one_byte[1] = {'x'};
    check(!g1_leg_configs_validate(db, one_byte, 1),
          "one-byte error buffer rejected input");
    check(one_byte[0] == '\0', "one-byte error buffer terminated");

    char short_error[8];
    std::memset(short_error, 'x', sizeof(short_error));
    check(!g1_leg_configs_validate(
              db, short_error, static_cast<int>(sizeof(short_error))),
          "short error buffer rejected input");
    check(short_error[sizeof(short_error) - 1] == '\0',
          "short error buffer terminated");
    check(std::strncmp(short_error, "left", 4) == 0,
          "short error buffer preserves diagnostic prefix");

    char untouched = 'q';
    check(!g1_leg_configs_validate(db, &untouched, 0),
          "zero-capacity error buffer rejected input");
    check(untouched == 'q', "zero-capacity error buffer untouched");
}

static float g1_test_float_from_bits(uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool g1_test_float_same(float left, float right)
{
    return terrain_float_bits(left) == terrain_float_bits(right);
}

static bool g1_test_vec3_same(vec3 left, vec3 right)
{
    return g1_test_float_same(left.x, right.x) &&
           g1_test_float_same(left.y, right.y) &&
           g1_test_float_same(left.z, right.z);
}

static bool g1_test_surface_same(
    const G1SurfaceTarget& left, const G1SurfaceTarget& right)
{
    return g1_test_vec3_same(left.point, right.point) &&
           g1_test_vec3_same(left.normal, right.normal);
}

static bool g1_test_lock_state_same(
    const G1FootLockState& left, const G1FootLockState& right)
{
    return left.initialized == right.initialized &&
           left.contact == right.contact &&
           left.locked == right.locked &&
           left.position_active == right.position_active &&
           left.releasing == right.releasing &&
           left.release_frames == right.release_frames &&
           g1_test_vec3_same(left.previous_input, right.previous_input) &&
           g1_test_vec3_same(left.lock_point, right.lock_point) &&
           g1_test_vec3_same(left.output_position, right.output_position) &&
           g1_test_vec3_same(left.output_velocity, right.output_velocity) &&
           g1_test_vec3_same(left.offset_position, right.offset_position) &&
           g1_test_vec3_same(left.offset_velocity, right.offset_velocity);
}

static bool g1_test_foot_target_same(
    const G1FootTarget& left, const G1FootTarget& right)
{
    return left.locked == right.locked &&
           left.position_active == right.position_active &&
           left.releasing == right.releasing &&
           left.drift_limit_exceeded == right.drift_limit_exceeded &&
           g1_test_surface_same(left.surface, right.surface) &&
           g1_test_vec3_same(left.sole_center, right.sole_center) &&
           g1_test_float_same(
               left.horizontal_drift_m, right.horizontal_drift_m);
}

static void g1_test_make_surface(
    heightfield& field,
    int nx,
    int nz,
    float origin_x = 0.0f,
    float origin_z = 0.0f,
    float cell_size = 1.0f)
{
    field.version = 2;
    field.nx = nx;
    field.nz = nz;
    field.origin_x = origin_x;
    field.origin_z = origin_z;
    field.cell_size = cell_size;
    field.exterior_height = -10.0f;
    field.heights.resize(nx * nz);
    field.heights.set(0.0f);
}

static void test_checked_surface_query()
{
    heightfield flat_surface;
    g1_test_make_surface(flat_surface, 2, 2);
    G1SurfaceSample flat_sample = {};
    check(g1_surface_query_v2(
              flat_sample, flat_surface, 0.25f, 0.75f) ==
              G1SurfaceQueryValid &&
          flat_sample.height == 0.0f &&
          g1_test_vec3_same(
              flat_sample.normal, vec3(0.0f, 1.0f, 0.0f)),
          "flat surface height and normal");

    heightfield ramp_surface;
    g1_test_make_surface(ramp_surface, 2, 2);
    ramp_surface.heights(0) = 0.0f;
    ramp_surface.heights(1) = 0.1f;
    ramp_surface.heights(2) = 0.0f;
    ramp_surface.heights(3) = 0.1f;
    G1SurfaceSample ramp_sample = {};
    check(g1_surface_query_v2(
              ramp_sample, ramp_surface, 0.25f, 0.50f) ==
              G1SurfaceQueryValid &&
          std::fabs(ramp_sample.height - 0.025f) < 1.0e-7f &&
          ramp_sample.normal.x < 0.0f && ramp_sample.normal.y > 0.99f,
          "ramp height and upward normal");

    heightfield field;
    g1_test_make_surface(field, 2, 2);
    field.heights(0) = 0.0f;
    field.heights(1) = 1.0f;
    field.heights(2) = 2.0f;
    field.heights(3) = 4.0f;

    const vec3 points[] = {
        vec3(0.75f, 0.0f, 0.25f),
        vec3(0.25f, 0.0f, 0.75f),
        vec3(0.50f, 0.0f, 0.50f)
    };
    for (int i = 0; i < 3; ++i) {
        G1SurfaceSample sample = {};
        check(g1_surface_query_v2(
                  sample, field, points[i].x, points[i].z) ==
                  G1SurfaceQueryValid,
              "checked non-planar query accepted");
        check(g1_test_float_same(
                  sample.height,
                  heightfield_sample_v2(field, points[i].x, points[i].z)),
              "checked height preserves fixed-diagonal bits");
        check(g1_test_vec3_same(
                  sample.normal,
                  heightfield_normal(field, points[i].x, points[i].z)),
              "checked normal preserves fixed-diagonal bits");
    }
    G1SurfaceSample first_triangle = {};
    G1SurfaceSample second_triangle = {};
    G1SurfaceSample diagonal_tie = {};
    check(g1_surface_query_v2(first_triangle, field, 0.75f, 0.25f) ==
              G1SurfaceQueryValid &&
          g1_surface_query_v2(second_triangle, field, 0.25f, 0.75f) ==
              G1SurfaceQueryValid &&
          g1_surface_query_v2(diagonal_tie, field, 0.5f, 0.5f) ==
              G1SurfaceQueryValid,
          "both fixed-diagonal triangles and tie accepted");
    check(first_triangle.height == 1.5f,
          "first fixed-diagonal triangle height");
    check(second_triangle.height == 2.0f,
          "second fixed-diagonal triangle height");
    check(diagonal_tie.height == 2.0f &&
          g1_test_vec3_same(diagonal_tie.normal, first_triangle.normal),
          "diagonal tie belongs to first triangle");

    const float corners[4][2] = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}
    };
    for (int i = 0; i < 4; ++i) {
        G1SurfaceSample corner = {};
        check(g1_surface_query_v2(
                  corner, field, corners[i][0], corners[i][1]) ==
                  G1SurfaceQueryValid,
              "inclusive terrain corner accepted");
    }

    heightfield shifted;
    g1_test_make_surface(shifted, 2, 2, 0.5f, 0.5f);
    const float negative_outward = std::nextafter(
        0.5f, -std::numeric_limits<float>::infinity());
    const float positive_outward = std::nextafter(
        1.0f, std::numeric_limits<float>::infinity());
    G1SurfaceSample sentinel = {7.0f, vec3(3.0f, 4.0f, 5.0f)};
    const G1SurfaceSample sentinel_before = sentinel;
    check(g1_surface_query_v2(sentinel, shifted, negative_outward, 1.0f) ==
              G1SurfaceQueryOutside,
          "one-ulp lower exterior query rejected");
    check(g1_test_float_same(sentinel.height, sentinel_before.height) &&
          g1_test_vec3_same(sentinel.normal, sentinel_before.normal),
          "outside checked query is transactional");
    check(g1_surface_query_v2(sentinel, field, positive_outward, 0.5f) ==
              G1SurfaceQueryOutside,
          "one-ulp upper exterior query rejected");

    const float nan = g1_test_float_from_bits(UINT32_C(0x7fc00001));
    const float subnormal = g1_test_float_from_bits(UINT32_C(0x00000001));
    check(g1_surface_query_v2(sentinel, field, nan, 0.5f) ==
              G1SurfaceQueryInvalid,
          "NaN checked coordinate is invalid");
    check(g1_surface_query_v2(sentinel, field, subnormal, 0.5f) ==
              G1SurfaceQueryInvalid,
          "subnormal checked coordinate is invalid");

    field.version = 1;
    check(g1_surface_query_v2(sentinel, field, 0.5f, 0.5f) ==
              G1SurfaceQueryInvalid,
          "non-v2 checked field is invalid");
    field.version = 2;
    float* const saved_data = field.heights.data;
    field.heights.data = NULL;
    check(g1_surface_query_v2(sentinel, field, 0.5f, 0.5f) ==
              G1SurfaceQueryInvalid,
          "missing checked height storage is invalid");
    field.heights.data = saved_data;
    const int saved_size = field.heights.size;
    field.heights.size = saved_size - 1;
    check(g1_surface_query_v2(sentinel, field, 0.5f, 0.5f) ==
              G1SurfaceQueryInvalid,
          "malformed checked height storage size is invalid");
    field.heights.size = saved_size;
    field.heights(3) = nan;
    check(g1_surface_query_v2(sentinel, field, 0.75f, 0.25f) ==
              G1SurfaceQueryInvalid,
          "selected NaN height is invalid");
    field.heights(3) = subnormal;
    check(g1_surface_query_v2(sentinel, field, 0.75f, 0.25f) ==
              G1SurfaceQueryInvalid,
          "selected subnormal height is invalid");

    heightfield flat;
    g1_test_make_surface(flat, 2, 2);
    G1SurfaceTarget target = {
        vec3(8.0f, 9.0f, 10.0f), vec3(0.0f, 1.0f, 0.0f)};
    const G1SurfaceTarget target_before = target;
    char error[256] = {};
    check(!g1_surface_target_sample(
              target, flat, positive_outward, 0.5f, 0.005f,
              error, static_cast<int>(sizeof(error))),
          "public exterior target rejected instead of using fallback");
    check_error_contains(error, "outside", "public exterior diagnostic");
    check(g1_test_surface_same(target, target_before),
          "failed public target is transactional");
    flat.heights.set(std::numeric_limits<float>::max());
    check(!g1_surface_target_sample(
              target, flat, 0.5f, 0.5f,
              std::numeric_limits<float>::max(),
              error, static_cast<int>(sizeof(error))),
          "overflowing height plus clearance rejected");
    check(g1_test_surface_same(target, target_before),
          "overflowing public target is transactional");
}

static void test_planted_lock_lifecycle()
{
    heightfield field;
    g1_test_make_surface(field, 3, 3);
    for (int z = 0; z < 3; ++z) {
        for (int x = 0; x < 3; ++x) {
            field.heights(z * 3 + x) = static_cast<float>(x) * 0.1f;
        }
    }
    const G1LegConfig leg = g1_left_leg_config();
    const float dt = 1.0f / 25.0f;
    char error[256] = {};

    G1FootLockState state = {};
    const vec3 initial(0.25f, 0.40f, 0.50f);
    check(g1_foot_lock_reset(
              state, initial, error, static_cast<int>(sizeof(error))),
          error);
    G1FootTarget target = {};
    check(g1_foot_lock_update(
              state, target, field, leg, initial, true, dt,
              error, static_cast<int>(sizeof(error))),
          error);
    check(state.locked && state.position_active && !state.releasing &&
          target.locked && target.position_active && !target.releasing,
          "contact rising edge activates planted lock");
    check(g1_test_vec3_same(target.sole_center, initial),
          "rising-edge frame owns exact pre-transition output");
    check(state.lock_point.x == initial.x &&
          state.lock_point.z == initial.z,
          "rising edge freezes support XZ");

    for (int frame = 0; frame < 100; ++frame) {
        const vec3 drifting(0.30f, 0.42f, 0.50f);
        check(g1_foot_lock_update(
                  state, target, field, leg, drifting, true, dt,
                  error, static_cast<int>(sizeof(error))),
              error);
    }
    check(std::fabs(target.sole_center.x - state.lock_point.x) < 1.0e-4f &&
          std::fabs(target.sole_center.y - state.lock_point.y) < 1.0e-4f &&
          std::fabs(target.sole_center.z - state.lock_point.z) < 1.0e-4f,
          "planted spring converges to locked sole target");

    const vec3 before_release = target.sole_center;
    const vec3 released_input(0.35f, 0.55f, 0.50f);
    check(g1_foot_lock_update(
              state, target, field, leg, released_input, false, dt,
              error, static_cast<int>(sizeof(error))),
          error);
    check(!state.locked && state.position_active && state.releasing &&
          !target.locked && target.position_active && target.releasing,
          "falling edge starts explicit release phase");
    check(g1_test_vec3_same(target.sole_center, before_release),
          "falling-edge frame is position-continuous");

    check(g1_foot_lock_update(
              state, target, field, leg, released_input, false, dt,
              error, static_cast<int>(sizeof(error))),
          error);
    check(state.releasing && state.release_frames > 0,
          "release remains active across multiple frames");
    const vec3 before_recontact = target.sole_center;
    check(g1_foot_lock_update(
              state, target, field, leg, released_input, true, dt,
              error, static_cast<int>(sizeof(error))),
          error);
    check(state.locked && state.position_active && !state.releasing,
          "recontact during release restores lock");
    check(g1_test_vec3_same(target.sole_center, before_recontact),
          "recontact transition is position-continuous");

    check(g1_foot_lock_update(
              state, target, field, leg, released_input, false, dt,
              error, static_cast<int>(sizeof(error))),
          error);
    int release_updates = 0;
    while (state.position_active &&
           release_updates <= G1_FootReleaseMaximumFrames) {
        check(g1_foot_lock_update(
                  state, target, field, leg, released_input, false, dt,
                  error, static_cast<int>(sizeof(error))),
              error);
        ++release_updates;
    }
    check(!state.position_active && !state.releasing && !target.position_active,
          "release eventually deactivates");
    check(release_updates <= G1_FootReleaseMaximumFrames,
          "release deactivation has a deterministic bounded cap");
    check(g1_test_vec3_same(target.sole_center, released_input),
          "settled release snaps exactly to animation target");

    heightfield flat;
    g1_test_make_surface(flat, 2, 2);
    G1FootLockState zero_offset_state = {};
    const vec3 flat_target(0.5f, leg.planted_clearance_m, 0.5f);
    check(g1_foot_lock_reset(
              zero_offset_state, flat_target,
              error, static_cast<int>(sizeof(error))),
          error);
    G1FootTarget zero_offset_target = {};
    check(g1_foot_lock_update(
              zero_offset_state, zero_offset_target, flat, leg,
              flat_target, true, dt,
              error, static_cast<int>(sizeof(error))) &&
          g1_foot_lock_update(
              zero_offset_state, zero_offset_target, flat, leg,
              flat_target, true, dt,
              error, static_cast<int>(sizeof(error))) &&
          g1_foot_lock_update(
              zero_offset_state, zero_offset_target, flat, leg,
              flat_target, false, dt,
              error, static_cast<int>(sizeof(error))),
          error);
    check(zero_offset_state.releasing &&
          zero_offset_state.release_frames == 0,
          "zero-offset falling edge starts release");
    check(g1_foot_lock_update(
              zero_offset_state, zero_offset_target, flat, leg,
              flat_target, false, dt,
              error, static_cast<int>(sizeof(error))),
          error);
    check(zero_offset_state.position_active &&
          zero_offset_state.releasing &&
          zero_offset_state.release_frames == 1,
          "zero-offset release cannot settle on first stable update");
    check(g1_foot_lock_update(
              zero_offset_state, zero_offset_target, flat, leg,
              flat_target, false, dt,
              error, static_cast<int>(sizeof(error))),
          error);
    check(!zero_offset_state.position_active &&
          !zero_offset_state.releasing,
          "zero-offset release settles after two stable updates");
}

static void test_planted_lock_drift_dt_and_rollback()
{
    heightfield field;
    g1_test_make_surface(field, 3, 3);
    const G1LegConfig valid_leg = g1_left_leg_config();
    const float dt = 1.0f / 25.0f;
    char error[256] = {};

    G1FootLockState state = {};
    check(g1_foot_lock_reset(
              state, vec3(0.0f, 0.2f, 0.5f), error,
              static_cast<int>(sizeof(error))),
          error);
    G1FootTarget target = {};
    check(g1_foot_lock_update(
              state, target, field, valid_leg,
              vec3(0.0f, 0.2f, 0.5f), true, dt,
              error, static_cast<int>(sizeof(error))),
          error);
    check(g1_foot_lock_update(
              state, target, field, valid_leg,
              vec3(0.20f, 0.2f, 0.5f), true, dt,
              error, static_cast<int>(sizeof(error))),
          error);
    check(state.locked && !target.drift_limit_exceeded &&
          g1_test_float_same(target.horizontal_drift_m, 0.20f),
          "exact 0.20 m drift remains allowed and locked");
    const float above_limit = std::nextafter(
        0.20f, std::numeric_limits<float>::infinity());
    check(g1_foot_lock_update(
              state, target, field, valid_leg,
              vec3(above_limit, 0.2f, 0.5f), true, dt,
              error, static_cast<int>(sizeof(error))),
          error);
    check(state.locked && target.locked && target.drift_limit_exceeded,
          "one-ulp-above drift flags without auto-unlock");

    const float invalid_dts[] = {
        std::nextafter(dt, 0.0f),
        std::nextafter(dt, std::numeric_limits<float>::infinity()),
        std::numeric_limits<float>::infinity(),
        g1_test_float_from_bits(UINT32_C(0x7fc00001))
    };
    for (size_t i = 0;
         i < sizeof(invalid_dts) / sizeof(invalid_dts[0]);
         ++i) {
        const G1FootLockState state_before = state;
        const G1FootTarget target_before = target;
        check(!g1_foot_lock_update(
                  state, target, field, valid_leg,
                  vec3(0.1f, 0.2f, 0.5f), true, invalid_dts[i],
                  error, static_cast<int>(sizeof(error))),
              "non-exact 25 Hz timestep rejected");
        check_error_contains(error, "25 Hz", "exact timestep diagnostic");
        check(g1_test_lock_state_same(state, state_before) &&
              g1_test_foot_target_same(target, target_before),
              "bad timestep rolls back state and output");
    }

    G1LegConfig poisoned_config = valid_leg;
    poisoned_config.planted_clearance_m =
        g1_test_float_from_bits(UINT32_C(0x7fc00001));
    G1FootLockState state_before = state;
    G1FootTarget target_before = target;
    check(!g1_foot_lock_update(
              state, target, field, poisoned_config,
              vec3(0.1f, 0.2f, 0.5f), true, dt,
              error, static_cast<int>(sizeof(error))),
          "poisoned complete config rejected");
    check(g1_test_lock_state_same(state, state_before) &&
          g1_test_foot_target_same(target, target_before),
          "poisoned config rolls back state and output");

    state_before = state;
    target_before = target;
    check(!g1_foot_lock_update(
              state, target, field, valid_leg,
              vec3(g1_test_float_from_bits(UINT32_C(0x7fc00001)),
                   0.2f, 0.5f),
              true, dt, error, static_cast<int>(sizeof(error))),
          "non-finite input center rejected");
    check(g1_test_lock_state_same(state, state_before) &&
          g1_test_foot_target_same(target, target_before),
          "non-finite input rolls back state and output");

    G1FootLockState poisoned_state = state;
    poisoned_state.offset_velocity.z =
        g1_test_float_from_bits(UINT32_C(0x7fc00001));
    const G1FootLockState poisoned_before = poisoned_state;
    target_before = target;
    check(!g1_foot_lock_update(
              poisoned_state, target, field, valid_leg,
              vec3(0.1f, 0.2f, 0.5f), true, dt,
              error, static_cast<int>(sizeof(error))),
          "poisoned complete lock state rejected");
    check(g1_test_lock_state_same(poisoned_state, poisoned_before) &&
          g1_test_foot_target_same(target, target_before),
          "poisoned state rolls back state and output");

    G1FootLockState finite_corrupt_state = state;
    finite_corrupt_state.lock_point.y = 0.10f;
    const G1FootLockState finite_corrupt_before = finite_corrupt_state;
    target_before = target;
    check(!g1_foot_lock_update(
              finite_corrupt_state, target, field, valid_leg,
              vec3(above_limit, 0.2f, 0.5f), true, dt,
              error, static_cast<int>(sizeof(error))),
          "finite-corrupted lock point rejected");
    check_error_contains(
        error, "lock point", "finite-corrupted lock point diagnostic");
    check(g1_test_lock_state_same(
              finite_corrupt_state, finite_corrupt_before) &&
          g1_test_foot_target_same(target, target_before),
          "finite-corrupted lock point rolls back state and output");

    G1FootTarget poisoned_target = target;
    poisoned_target.sole_center.x =
        g1_test_float_from_bits(UINT32_C(0x7fc00001));
    const G1FootTarget poisoned_target_before = poisoned_target;
    state_before = state;
    check(!g1_foot_lock_update(
              state, poisoned_target, field, valid_leg,
              vec3(0.1f, 0.2f, 0.5f), true, dt,
              error, static_cast<int>(sizeof(error))),
          "poisoned complete output rejected");
    check(g1_test_lock_state_same(state, state_before) &&
          g1_test_foot_target_same(poisoned_target, poisoned_target_before),
          "poisoned output rolls back state and output");

    G1FootLockState overflow_state = {};
    check(g1_foot_lock_reset(
              overflow_state,
              vec3(0.5f, -std::numeric_limits<float>::max(), 0.5f),
              error, static_cast<int>(sizeof(error))),
          error);
    G1FootTarget overflow_target = {};
    const G1FootLockState overflow_before = overflow_state;
    const G1FootTarget overflow_target_before = overflow_target;
    check(!g1_foot_lock_update(
              overflow_state, overflow_target, field, valid_leg,
              vec3(0.5f, std::numeric_limits<float>::max(), 0.5f),
              false, dt, error, static_cast<int>(sizeof(error))),
          "finite extreme input velocity overflow rejected");
    check(g1_test_lock_state_same(overflow_state, overflow_before) &&
          g1_test_foot_target_same(
              overflow_target, overflow_target_before),
          "extreme arithmetic failure rolls back state and output");

    G1FootLockState reset_state = state;
    const G1FootLockState reset_before = reset_state;
    check(!g1_foot_lock_reset(
              reset_state,
              vec3(g1_test_float_from_bits(UINT32_C(0x7fc00001)), 0.0f, 0.0f),
              error, static_cast<int>(sizeof(error))),
          "non-finite reset center rejected");
    check(g1_test_lock_state_same(reset_state, reset_before),
          "failed reset is transactional");

    state_before = state;
    target_before = target;
    check(!g1_foot_lock_update(
              state, target, field, valid_leg,
              vec3(0.1f, 0.2f, 0.5f), true,
              std::nextafter(dt, 0.0f), NULL, 0),
          "null diagnostic buffer remains safe");
    char one_byte[1] = {'x'};
    check(!g1_foot_lock_update(
              state, target, field, valid_leg,
              vec3(0.1f, 0.2f, 0.5f), true,
              std::nextafter(dt, 0.0f), one_byte, 1),
          "one-byte diagnostic buffer remains safe");
    check(one_byte[0] == '\0', "one-byte lock diagnostic terminated");
    check(g1_test_lock_state_same(state, state_before) &&
          g1_test_foot_target_same(target, target_before),
          "hostile diagnostic buffers preserve rollback");
}

static bool g1_test_quat_bits_same(quat left, quat right)
{
    return g1_test_float_same(left.w, right.w) &&
           g1_test_float_same(left.x, right.x) &&
           g1_test_float_same(left.y, right.y) &&
           g1_test_float_same(left.z, right.z);
}

static bool g1_test_target_projection_same(
    const IKTargetProjection& left, const IKTargetProjection& right)
{
    return left.reachable == right.reachable &&
           g1_test_vec3_same(
               left.clamped_target, right.clamped_target) &&
           g1_test_float_same(
               left.raw_distance_m, right.raw_distance_m) &&
           g1_test_float_same(
               left.clamped_distance_m, right.clamped_distance_m) &&
           g1_test_float_same(
               left.minimum_distance_m, right.minimum_distance_m) &&
           g1_test_float_same(
               left.maximum_distance_m, right.maximum_distance_m);
}

static bool g1_test_clamp_result_same(
    const IKClampResult& left, const IKClampResult& right)
{
    return g1_test_quat_bits_same(left.value, right.value) &&
           g1_test_float_same(
               left.requested_radians, right.requested_radians) &&
           g1_test_float_same(
               left.actual_radians, right.actual_radians) &&
           left.limited == right.limited;
}

static bool g1_test_bend_selection_same(
    const IKBendSelection& left, const IKBendSelection& right)
{
    return g1_test_vec3_same(left.direction, right.direction) &&
           g1_test_float_same(
               left.current_projection_length,
               right.current_projection_length) &&
           left.used_current_projection ==
               right.used_current_projection &&
           left.used_hinge_fallback == right.used_hinge_fallback &&
           left.used_safe_perpendicular ==
               right.used_safe_perpendicular &&
           left.sign_flipped == right.sign_flipped;
}

static bool g1_test_two_bone_result_same(
    const IKTwoBoneResult& left, const IKTwoBoneResult& right)
{
    return left.applied == right.applied &&
           left.reachable == right.reachable &&
           left.correction_limited == right.correction_limited &&
           g1_test_quat_bits_same(left.root_local, right.root_local) &&
           g1_test_quat_bits_same(
               left.middle_local, right.middle_local) &&
           g1_test_float_same(
               left.root_correction_radians,
               right.root_correction_radians) &&
           g1_test_float_same(
               left.middle_correction_radians,
               right.middle_correction_radians) &&
           g1_test_target_projection_same(left.target, right.target) &&
           g1_test_bend_selection_same(left.bend, right.bend);
}

static bool g1_test_leg_solve_result_same(
    const G1LegSolveResult& left, const G1LegSolveResult& right)
{
    return left.applied == right.applied &&
           left.reachable == right.reachable &&
           left.correction_limited == right.correction_limited &&
           left.safe_stop_requested == right.safe_stop_requested &&
           left.iterations == right.iterations &&
           g1_test_vec3_same(
               left.requested_ankle_target,
               right.requested_ankle_target) &&
           g1_test_vec3_same(
               left.clamped_ankle_target,
               right.clamped_ankle_target) &&
           g1_test_vec3_same(
               left.hinge_axis_world, right.hinge_axis_world) &&
           g1_test_vec3_same(
               left.bend_direction, right.bend_direction) &&
           left.bend_used_current_projection ==
               right.bend_used_current_projection &&
           left.bend_used_hinge_fallback ==
               right.bend_used_hinge_fallback &&
           left.bend_used_safe_perpendicular ==
               right.bend_used_safe_perpendicular &&
           left.bend_sign_flipped == right.bend_sign_flipped &&
           g1_test_float_same(
               left.raw_distance_m, right.raw_distance_m) &&
           g1_test_float_same(
               left.clamped_distance_m, right.clamped_distance_m) &&
           g1_test_float_same(
               left.max_correction_radians,
               right.max_correction_radians) &&
           g1_test_float_same(
               left.contact_residual_m,
               right.contact_residual_m);
}

static void g1_test_copy_g1_pose(
    array1d<quat>& output, const slice1d<quat> input)
{
    check(input.size == G1_BoneCount && input.data != NULL,
          "test fixture provides a complete G1 pose");
    output.resize(G1_BoneCount);
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        output(bone) = input.data[bone];
    }
}

static bool g1_test_pose_bytes_same(
    const array1d<quat>& left, const array1d<quat>& right)
{
    return left.size == right.size &&
           std::memcmp(
               left.data, right.data,
               static_cast<size_t>(left.size) * sizeof(quat)) == 0;
}

static void g1_test_global_pose(
    array1d<vec3>& positions,
    array1d<quat>& rotations,
    const database& db,
    const array1d<quat>& local_rotations)
{
    positions.resize(G1_BoneCount);
    rotations.resize(G1_BoneCount);
    forward_kinematics_full(
        positions, rotations,
        db.bone_positions(0), local_rotations, db.bone_parents);
}

static void test_generic_checked_ik_math()
{
    const float maximum = 0.35f;
    const quat identity;

    vec3 subnormal_commit_sentinel(7.0f, 8.0f, 9.0f);
    const vec3 subnormal_commit_before = subnormal_commit_sentinel;
    const float minimum_normal = std::numeric_limits<float>::min();
    const float next_minimum_normal = std::nextafter(
        minimum_normal, std::numeric_limits<float>::infinity());
    check(!ik_checked_vec3_subtract(
              subnormal_commit_sentinel,
              vec3(next_minimum_normal, 0.0f, 0.0f),
              vec3(minimum_normal, 0.0f, 0.0f)),
          "derived nonzero binary32 subnormal is rejected at commit");
    check(g1_test_vec3_same(
              subnormal_commit_sentinel, subnormal_commit_before),
          "derived-subnormal rejection leaves vector output unchanged");

    IKClampResult clamp = {};
    float boundary_source_angle = maximum;
    for (int step = 0; step < 64; ++step) {
        const quat candidate = quat_from_angle_axis(
            boundary_source_angle, vec3(1.0f, 0.0f, 0.0f));
        check(ik_clamp_local_delta(
                  clamp, identity, candidate, maximum),
              "clamp accepts representable boundary search candidate");
        if (!clamp.limited) break;
        boundary_source_angle = std::nextafter(
            boundary_source_angle, 0.0f);
    }
    check(!clamp.limited && clamp.requested_radians <= maximum &&
          clamp.actual_radians <= maximum,
          "largest discovered non-over-bound correction is not limited");
    check(ik_quat_is_unit(clamp.value),
          "boundary clamp produces unit quaternion");

    float over_source_angle = std::nextafter(
        boundary_source_angle, std::numeric_limits<float>::infinity());
    for (int step = 0; step < 64; ++step) {
        const quat candidate = quat_from_angle_axis(
            over_source_angle, vec3(1.0f, 0.0f, 0.0f));
        check(ik_clamp_local_delta(
                  clamp, identity, candidate, maximum),
              "clamp accepts precise over-bound search candidate");
        if (clamp.limited) break;
        over_source_angle = std::nextafter(
            over_source_angle, std::numeric_limits<float>::infinity());
    }
    check(clamp.limited && clamp.actual_radians <= maximum,
          "precise over-bound correction is classified and clamped");

    const vec3 clamp_axes[3] = {
        vec3(1.0f, 0.0f, 0.0f),
        vec3(0.0f, 1.0f, 0.0f),
        vec3(0.0f, 0.0f, 1.0f)
    };
    for (int baseline_axis = 0; baseline_axis < 3; ++baseline_axis) {
        for (int desired_axis = 0; desired_axis < 3; ++desired_axis) {
            for (int baseline_step = 0; baseline_step < 16;
                 ++baseline_step) {
                for (int desired_step = 0; desired_step < 32;
                     ++desired_step) {
                    const quat baseline = quat_from_angle_axis(
                        0.11f * static_cast<float>(baseline_step),
                        clamp_axes[baseline_axis]);
                    const quat desired = quat_from_angle_axis(
                        0.09f * static_cast<float>(desired_step),
                        clamp_axes[desired_axis]);
                    IKClampResult swept = {};
                    const bool swept_ok = ik_clamp_local_delta(
                        swept, baseline, desired, maximum);
                    check(swept_ok,
                          "bounded clamp sweep remains solvable");
                    check(swept.actual_radians <= maximum,
                          "every swept clamp reports an in-bound actual delta");
                }
            }
        }
    }

    const quat precise_cap_baseline(
        g1_test_float_from_bits(UINT32_C(0x3f7f9d9f)),
        g1_test_float_from_bits(UINT32_C(0x3d605936)),
        0.0f, 0.0f);
    const quat precise_cap_desired(
        g1_test_float_from_bits(UINT32_C(0x3f78de34)),
        g1_test_float_from_bits(UINT32_C(0x3e700638)),
        0.0f, 0.0f);
    const float precise_cap_maximum =
        g1_test_float_from_bits(UINT32_C(0x3eb33333));
    IKClampResult precise_cap_result = {};
    check(ik_clamp_local_delta(
              precise_cap_result, precise_cap_baseline,
              precise_cap_desired, precise_cap_maximum),
          "fixed-bit precise-cap correction remains solvable");
    double independently_measured_precise_cap = 0.0;
    float independently_measured_rounded_cap = 0.0f;
    check(ik_checked_quat_angle(
              independently_measured_precise_cap,
              independently_measured_rounded_cap,
              precise_cap_baseline, precise_cap_result.value),
          "fixed-bit clamp result is independently measurable");
    check(independently_measured_precise_cap <=
              static_cast<double>(precise_cap_maximum) &&
          independently_measured_rounded_cap <= precise_cap_maximum &&
          precise_cap_result.actual_radians ==
              independently_measured_rounded_cap,
          "precise and reported applied corrections both obey the cap");

    const quat tolerated_scaled_identity(0.99999f, 0.0f, 0.0f, 0.0f);
    check(ik_quat_is_unit(tolerated_scaled_identity),
          "scaled-identity fixture is inside the admitted unit tolerance");
    check(ik_clamp_local_delta(
              clamp, tolerated_scaled_identity,
              tolerated_scaled_identity, 0.005f),
          "admitted scaled equivalent quaternions are normalized safely");
    check(!clamp.limited && clamp.requested_radians == 0.0f &&
          clamp.actual_radians == 0.0f &&
          g1_test_quat_bits_same(clamp.value, identity),
          "normalization prevents a false correction on equal rotations");

    const quat antipodal(-1.0f, 0.0f, 0.0f, 0.0f);
    check(ik_clamp_local_delta(
              clamp, identity, antipodal, maximum),
          "antipodal equivalent quaternion accepted");
    check(!clamp.limited && clamp.requested_radians == 0.0f &&
          clamp.actual_radians == 0.0f && ik_quat_is_unit(clamp.value),
          "antipodal equivalence has zero correction");

    quat hostile[4] = {
        quat(0.0f, 0.0f, 0.0f, 0.0f),
        quat(2.0f, 0.0f, 0.0f, 0.0f),
        quat(g1_test_float_from_bits(UINT32_C(0x7fc00001)),
             0.0f, 0.0f, 0.0f),
        quat(std::numeric_limits<float>::infinity(),
             0.0f, 0.0f, 0.0f)
    };
    for (int i = 0; i < 4; ++i) {
        IKClampResult sentinel = {};
        sentinel.value = quat(0.5f, 0.5f, 0.5f, 0.5f);
        sentinel.requested_radians = 7.0f;
        sentinel.actual_radians = 8.0f;
        sentinel.limited = true;
        const IKClampResult before = sentinel;
        check(!ik_clamp_local_delta(
                  sentinel, identity, hostile[i], maximum),
              "invalid desired quaternion rejected transactionally");
        check(g1_test_clamp_result_same(sentinel, before),
              "failed clamp leaves complete output unchanged");
    }

    IKTargetProjection projection = {};
    check(ik_project_target(
              projection,
              vec3(), vec3(0.0f, -0.4f, 0.0f),
              vec3(0.0f, -0.8f, 0.0f),
              vec3(0.2f, -0.7f, 0.0f), 0.015f),
          "checked target projection accepts ordinary chain");
    check(projection.reachable &&
          projection.minimum_distance_m == 0.015f &&
          projection.maximum_distance_m == 0.8f,
          "checked projection expands the outer shell to the current end");

    const vec3 precise_outer_request(
        g1_test_float_from_bits(UINT32_C(0x40000000)),
        g1_test_float_from_bits(UINT32_C(0x3752d427)),
        0.0f);
    check(ik_project_target(
              projection,
              vec3(), vec3(0.0f, -0.4f, 0.0f),
              vec3(0.0f, -0.8f, 0.0f),
              precise_outer_request, 0.015f),
          "fixed-bit outer-shell target remains projectable");
    double materialized_outer_radius = 0.0;
    float materialized_outer_report = 0.0f;
    check(ik_checked_distance_precise(
              materialized_outer_radius, materialized_outer_report,
              projection.clamped_target, vec3()),
          "materialized outer-shell target is independently measurable");
    check(!projection.reachable &&
          materialized_outer_radius <= 0.8 &&
          materialized_outer_radius <=
              static_cast<double>(projection.maximum_distance_m) &&
          projection.clamped_distance_m == materialized_outer_report,
          "materialized projected target stays inside its precise shell");

    for (int step = 0; step < 128; ++step) {
        const vec3 current_end(
            0.071f + 0.0013f * static_cast<float>(step),
            -0.63f + 0.0007f * static_cast<float>(step),
            0.019f - 0.0002f * static_cast<float>(step));
        const vec3 current_middle = current_end * 0.5f;
        check(ik_project_target(
                  projection, vec3(), current_middle, current_end,
                  current_end, 0.015f),
              "current-end shell sweep remains checked");
        check(projection.reachable &&
              projection.raw_distance_m ==
                  projection.clamped_distance_m,
              "effective projection shell always contains its current endpoint");
    }

    check(ik_project_target(
              projection,
              vec3(), vec3(0.0f, -0.4f, 0.0f), vec3(),
              vec3(), 0.015f),
          "folded zero-direction target uses deterministic fallback");
    check(projection.reachable &&
          projection.minimum_distance_m == 0.0f &&
          projection.clamped_distance_m == 0.0f &&
          g1_test_vec3_same(
              projection.clamped_target, vec3()),
          "folded projection expands the inner shell to the current end");

    const float subnormal =
        g1_test_float_from_bits(UINT32_C(0x00000001));
    const vec3 bad_link_cases[][3] = {
        {vec3(), vec3(), vec3(0.0f, -0.4f, 0.0f)},
        {vec3(), vec3(subnormal, 0.0f, 0.0f),
         vec3(0.0f, -0.4f, 0.0f)},
        {vec3(-std::numeric_limits<float>::max(), 0.0f, 0.0f),
         vec3(std::numeric_limits<float>::max(), 0.0f, 0.0f),
         vec3()},
        {vec3(), vec3(1.0e20f, 0.0f, 0.0f),
         vec3(2.0e20f, 0.0f, 0.0f)}
    };
    for (size_t i = 0;
         i < sizeof(bad_link_cases) / sizeof(bad_link_cases[0]);
         ++i) {
        IKTargetProjection sentinel = {};
        sentinel.reachable = true;
        sentinel.raw_distance_m = 9.0f;
        sentinel.clamped_target = vec3(7.0f, 8.0f, 9.0f);
        const IKTargetProjection before = sentinel;
        check(!ik_project_target(
                  sentinel,
                  bad_link_cases[i][0], bad_link_cases[i][1],
                  bad_link_cases[i][2], vec3(0.0f, -0.3f, 0.0f),
                  0.015f),
              "zero/subnormal/extreme link rejected");
        check(g1_test_target_projection_same(sentinel, before),
              "failed projection leaves complete output unchanged");
    }

    const vec3 target_direction(0.0f, -1.0f, 0.0f);
    const vec3 hinge(0.0f, 0.0f, -1.0f);
    IKBendSelection bend = {};

    const vec3 precise_primary_direction(
        g1_test_float_from_bits(UINT32_C(0xbec8b8e9)),
        g1_test_float_from_bits(UINT32_C(0xbf6b1efb)),
        g1_test_float_from_bits(UINT32_C(0xbd57b38c)));
    const vec3 precise_primary_upper(
        g1_test_float_from_bits(UINT32_C(0xbb156e4c)),
        g1_test_float_from_bits(UINT32_C(0xbbaf0a98)),
        g1_test_float_from_bits(UINT32_C(0xb9a00f0e)));
    double precise_primary_along = 0.0;
    check(ik_checked_dot(
              precise_primary_along,
              precise_primary_upper, precise_primary_direction),
          "fixed-bit primary projection dot remains checked");
    const double precise_primary_components[3] = {
        static_cast<double>(precise_primary_upper.x) -
            static_cast<double>(precise_primary_direction.x) *
                precise_primary_along,
        static_cast<double>(precise_primary_upper.y) -
            static_cast<double>(precise_primary_direction.y) *
                precise_primary_along,
        static_cast<double>(precise_primary_upper.z) -
            static_cast<double>(precise_primary_direction.z) *
                precise_primary_along
    };
    double precise_primary_length = 0.0;
    float precise_primary_report = 0.0f;
    check(ik_checked_norm_components(
              precise_primary_length, precise_primary_report,
              precise_primary_components, 3, true) &&
          precise_primary_length >= static_cast<double>(1.0e-6f),
          "pre-materialization primary projection reaches 1e-6 threshold");
    check(ik_select_bend_direction(
              bend, precise_primary_upper,
              precise_primary_direction, hinge),
          "fixed-bit primary bend remains selectable");
    check(bend.used_current_projection &&
          !bend.used_hinge_fallback,
          "primary bend predicate retains promoted projection length");

    const vec3 precise_cross_direction(
        g1_test_float_from_bits(UINT32_C(0x3eb5f44d)),
        g1_test_float_from_bits(UINT32_C(0xbb7fd6e6)),
        g1_test_float_from_bits(UINT32_C(0x3f6f496a)));
    const vec3 precise_cross_hinge(
        g1_test_float_from_bits(UINT32_C(0x3eb5f44d)),
        g1_test_float_from_bits(UINT32_C(0xbb7fc61f)),
        g1_test_float_from_bits(UINT32_C(0x3f6f496a)));
    const double precise_cross_components[3] = {
        static_cast<double>(precise_cross_hinge.y) *
                static_cast<double>(precise_cross_direction.z) -
            static_cast<double>(precise_cross_hinge.z) *
                static_cast<double>(precise_cross_direction.y),
        static_cast<double>(precise_cross_hinge.z) *
                static_cast<double>(precise_cross_direction.x) -
            static_cast<double>(precise_cross_hinge.x) *
                static_cast<double>(precise_cross_direction.z),
        static_cast<double>(precise_cross_hinge.x) *
                static_cast<double>(precise_cross_direction.y) -
            static_cast<double>(precise_cross_hinge.y) *
                static_cast<double>(precise_cross_direction.x)
    };
    double precise_cross_length = 0.0;
    float precise_cross_report = 0.0f;
    check(ik_checked_norm_components(
              precise_cross_length, precise_cross_report,
              precise_cross_components, 3, true) &&
          precise_cross_length >= static_cast<double>(1.0e-6f),
          "pre-materialization hinge cross reaches 1e-6 threshold");
    check(ik_select_bend_direction(
              bend, vec3(),
              precise_cross_direction, precise_cross_hinge),
          "fixed-bit hinge-cross bend remains selectable");
    check(bend.used_hinge_fallback &&
          !bend.used_safe_perpendicular,
          "hinge fallback predicate retains promoted cross length");

    const vec3 precise_sign_direction(
        g1_test_float_from_bits(UINT32_C(0xbf332833)),
        g1_test_float_from_bits(UINT32_C(0x3a6c18c8)),
        g1_test_float_from_bits(UINT32_C(0x3f36dccf)));
    const vec3 precise_sign_upper(
        g1_test_float_from_bits(UINT32_C(0xba9a15ea)),
        g1_test_float_from_bits(UINT32_C(0x35cc663b)),
        g1_test_float_from_bits(UINT32_C(0x3a9d45c6)));
    double precise_sign_along = 0.0;
    check(ik_checked_dot(
              precise_sign_along,
              precise_sign_upper, precise_sign_direction),
          "fixed-bit sign projection dot remains checked");
    const double precise_sign_components[3] = {
        static_cast<double>(precise_sign_upper.x) -
            static_cast<double>(precise_sign_direction.x) *
                precise_sign_along,
        static_cast<double>(precise_sign_upper.y) -
            static_cast<double>(precise_sign_direction.y) *
                precise_sign_along,
        static_cast<double>(precise_sign_upper.z) -
            static_cast<double>(precise_sign_direction.z) *
                precise_sign_along
    };
    double precise_sign_length = 0.0;
    float precise_sign_report = 0.0f;
    check(ik_checked_norm_components(
              precise_sign_length, precise_sign_report,
              precise_sign_components, 3, true) &&
          precise_sign_length > static_cast<double>(1.0e-8f),
          "pre-materialization sign projection exceeds 1e-8 threshold");
    check(ik_select_bend_direction(
              bend, precise_sign_upper,
              precise_sign_direction, vec3(1.0f, 0.0f, 0.0f)),
          "fixed-bit sign bend remains selectable");
    check(bend.used_hinge_fallback && bend.sign_flipped,
          "fallback sign predicate retains promoted projection length");

    check(ik_select_bend_direction(
              bend, vec3(1.0e-6f, -0.4f, 0.0f),
              target_direction, hinge),
          "primary bend threshold accepted");
    check(bend.used_current_projection &&
          !bend.used_hinge_fallback && bend.direction.x > 0.99f,
          "projection length equal to 1e-6 uses primary");
    check(ik_select_bend_direction(
              bend,
              vec3(std::nextafter(1.0e-6f, 0.0f), -0.4f, 0.0f),
              target_direction, hinge),
          "below-primary bend threshold accepted");
    check(!bend.used_current_projection && bend.used_hinge_fallback,
          "one-ulp below 1e-6 uses hinge fallback");
    check(ik_select_bend_direction(
              bend, vec3(1.0e-8f, -0.4f, 0.0f),
              target_direction, hinge),
          "sign threshold equality accepted");
    check(bend.direction.x < -0.99f && !bend.sign_flipped,
          "projection length equal to 1e-8 does not flip fallback");
    check(ik_select_bend_direction(
              bend,
              vec3(std::nextafter(
                       1.0e-8f,
                       std::numeric_limits<float>::infinity()),
                   -0.4f, 0.0f),
              target_direction, hinge),
          "above-sign threshold accepted");
    check(bend.direction.x > 0.99f && bend.sign_flipped,
          "one-ulp above 1e-8 flips disagreeing fallback");
    check(ik_select_bend_direction(
              bend, vec3(0.0f, -0.4f, 0.0f),
              target_direction, hinge),
          "deterministic hinge fallback accepted");
    check(g1_test_vec3_same(
              bend.direction, vec3(-1.0f, 0.0f, 0.0f)),
          "cross(-Z,-Y) gives deterministic -X bend");
    check(ik_select_bend_direction(
              bend, vec3(0.0f, -0.4f, 0.0f),
              target_direction, target_direction),
          "parallel hinge uses checked perpendicular");
    check(bend.used_safe_perpendicular && ik_vec3_is_unit(bend.direction),
          "safe perpendicular is reserved for degenerate hinge cross");

    IKTwoBoneResult solve = {};
    check(ik_two_bone_bounded(
              solve,
              identity, identity,
              vec3(), vec3(0.0f, -0.4f, 0.0f),
              vec3(0.0f, -0.8f, 0.0f),
              vec3(0.2f, -0.7f, 0.0f),
              hinge, identity, identity, identity,
              0.015f, maximum),
          "checked two-bone solve accepts reachable target");
    check(solve.applied && solve.reachable &&
          ik_quat_is_unit(solve.root_local) &&
          ik_quat_is_unit(solve.middle_local) &&
          solve.root_correction_radians <= maximum + 2.0e-6f &&
          solve.middle_correction_radians <= maximum + 2.0e-6f,
          "checked two-bone result is unit and correction-bounded");

    const vec3 folded_just_over_noop(
        g1_test_float_from_bits(UINT32_C(0x353dcf3b)),
        g1_test_float_from_bits(UINT32_C(0x353dd0ca)),
        0.0f);
    double folded_motion_precise = 0.0;
    float folded_motion_report = 0.0f;
    check(ik_checked_distance_precise(
              folded_motion_precise, folded_motion_report,
              folded_just_over_noop, vec3()) &&
          folded_motion_precise > static_cast<double>(1.0e-6f),
          "fixed-bit folded target is precisely above no-op tolerance");
    IKTwoBoneResult folded_motion = {};
    check(ik_two_bone_bounded(
              folded_motion,
              identity, identity,
              vec3(), vec3(0.0f, -0.4f, 0.0f), vec3(),
              folded_just_over_noop, hinge,
              identity, identity, identity,
              0.015f, maximum),
          "precisely nonzero folded motion remains solvable");
    check(folded_motion.root_correction_radians > 0.0f ||
          folded_motion.middle_correction_radians > 0.0f,
          "promoted motion above tolerance is not discarded as a no-op");

    for (int step = 0; step < 32; ++step) {
        const float angle =
            (2.0f * PIf * static_cast<float>(step)) / 32.0f;
        const vec3 boundary_target(
            0.8f * std::sin(angle),
            -0.8f * std::cos(angle),
            0.0f);
        IKTwoBoneResult boundary_solve = {};
        const bool boundary_ok = ik_two_bone_bounded(
            boundary_solve,
            identity, identity,
            vec3(), vec3(0.0f, -0.4f, 0.0f),
            vec3(0.0f, -0.8f, 0.0f),
            boundary_target, hinge,
            identity, identity, identity,
            0.015f, maximum);
        check(boundary_ok,
              "dynamically expanded outer-boundary solve stays checked");
        check(boundary_solve.applied &&
              boundary_solve.root_correction_radians <= maximum &&
              boundary_solve.middle_correction_radians <= maximum,
              "expanded-boundary sweep remains correction-bounded");
    }

    const vec3 invalid_hinges[] = {
        vec3(),
        vec3(0.0f, 0.0f, -2.0f),
        vec3(subnormal, 0.0f, 0.0f),
        vec3(g1_test_float_from_bits(UINT32_C(0x7fc00001)), 0.0f, 0.0f)
    };
    for (size_t i = 0;
         i < sizeof(invalid_hinges) / sizeof(invalid_hinges[0]);
         ++i) {
        IKTwoBoneResult sentinel = {};
        sentinel.applied = true;
        sentinel.root_correction_radians = 7.0f;
        const IKTwoBoneResult before = sentinel;
        check(!ik_two_bone_bounded(
                  sentinel,
                  identity, identity,
                  vec3(), vec3(0.0f, -0.4f, 0.0f),
                  vec3(0.0f, -0.8f, 0.0f),
                  vec3(0.2f, -0.7f, 0.0f),
                  invalid_hinges[i], identity, identity, identity,
                  0.015f, maximum),
              "invalid hinge rejected transactionally");
        check(g1_test_two_bone_result_same(sentinel, before),
              "failed two-bone solve leaves complete output unchanged");
    }

    IKTwoBoneResult extreme = {};
    extreme.applied = true;
    extreme.root_correction_radians = 11.0f;
    const IKTwoBoneResult extreme_before = extreme;
    check(!ik_two_bone_bounded(
              extreme,
              identity, identity,
              vec3(), vec3(1.0e20f, 0.0f, 0.0f),
              vec3(2.0e20f, 0.0f, 0.0f),
              vec3(1.0e20f, 1.0f, 0.0f),
              hinge, identity, identity, identity,
              0.015f, maximum),
          "finite 1e20 two-bone geometry fails cleanly");
    check(g1_test_two_bone_result_same(extreme, extreme_before),
          "extreme generic failure is transactional");
}

static G1LegSolveResult g1_test_leg_result_sentinel()
{
    G1LegSolveResult result = {};
    result.applied = true;
    result.reachable = true;
    result.correction_limited = true;
    result.safe_stop_requested = true;
    result.iterations = 7;
    result.requested_ankle_target = vec3(1.0f, 2.0f, 3.0f);
    result.clamped_ankle_target = vec3(4.0f, 5.0f, 6.0f);
    result.hinge_axis_world = vec3(0.0f, 0.0f, -1.0f);
    result.bend_direction = vec3(-1.0f, 0.0f, 0.0f);
    result.raw_distance_m = 7.0f;
    result.clamped_distance_m = 8.0f;
    result.max_correction_radians = 9.0f;
    result.contact_residual_m = 10.0f;
    return result;
}

static void g1_test_make_straight_solver_database(database& db)
{
    make_g1_database(db);
    db.bone_positions(0, G1_LeftKnee) = vec3(0.0f, -0.4f, 0.0f);
    db.bone_positions(0, G1_LeftAnkle) = vec3(0.0f, -0.4f, 0.0f);
    db.bone_positions(0, G1_LeftToe) = vec3(0.0f, -0.02f, 0.0f);
    db.bone_positions(0, G1_RightKnee) = vec3(0.0f, -0.4f, 0.0f);
    db.bone_positions(0, G1_RightAnkle) = vec3(0.0f, -0.4f, 0.0f);
    db.bone_positions(0, G1_RightToe) = vec3(0.0f, -0.02f, 0.0f);
}

static void test_named_solver_success_and_bend_mapping()
{
    database db;
    g1_test_make_straight_solver_database(db);
    const G1LegConfig left = g1_left_leg_config();
    const G1LegConfig right = g1_right_leg_config();
    char error[256] = {};

    array1d<quat> working = db.bone_rotations(0);
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        working(bone) = quat_from_angle_axis(
            0.001f * static_cast<float>(bone + 1),
            vec3(1.0f, 0.0f, 0.0f));
    }
    const array1d<quat> before_left = working;
    G1LegSolveResult left_result = {};
    check(g1_apply_named_position_ik(
              working, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, left, vec3(0.0f, -0.7f, 0.0f),
              left_result, error, static_cast<int>(sizeof(error))),
          error);
    check(left_result.applied && left_result.reachable &&
          left_result.max_correction_radians <=
              left.max_correction_radians + 2.0e-6f,
          "named left solve applies within correction bound");
    check(g1_test_vec3_same(
              left_result.hinge_axis_world,
              vec3(0.0f, 0.0f, -1.0f)) &&
          g1_test_vec3_same(
              left_result.bend_direction,
              vec3(-1.0f, 0.0f, 0.0f)),
          "identity baseline maps local -Z hinge and deterministic -X bend");
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (bone != left.hip && bone != left.knee) {
            check(g1_test_quat_bits_same(
                      working(bone), before_left(bone)),
                  "left solve preserves every non-target rotation byte");
        }
    }

    const quat solved_left_hip = working(left.hip);
    const quat solved_left_knee = working(left.knee);
    G1LegSolveResult right_result = {};
    check(g1_apply_named_position_ik(
              working, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, right, vec3(0.0f, -0.7f, 0.0f),
              right_result, error, static_cast<int>(sizeof(error))),
          error);
    check(g1_test_quat_bits_same(working(left.hip), solved_left_hip) &&
          g1_test_quat_bits_same(working(left.knee), solved_left_knee),
          "second-leg solve preserves first-leg staged rotations");
    check(g1_test_vec3_same(
              right_result.hinge_axis_world,
              vec3(0.0f, 0.0f, -1.0f)) &&
          g1_test_vec3_same(
              right_result.bend_direction,
              vec3(-1.0f, 0.0f, 0.0f)),
          "mirrored named legs share exact mapped hinge fallback");

    array1d<quat> rotated_baseline = db.bone_rotations(0);
    rotated_baseline(left.knee) = quat_from_angle_axis(
        0.5f * PIf, vec3(0.0f, 1.0f, 0.0f));
    array1d<quat> rotated_output = rotated_baseline;
    array1d<vec3> rotated_global_positions;
    array1d<quat> rotated_global_rotations;
    g1_test_global_pose(
        rotated_global_positions, rotated_global_rotations,
        db, rotated_baseline);
    G1LegSolveResult rotated_result = {};
    check(g1_apply_named_position_ik(
              rotated_output, db.bone_positions(0), rotated_baseline,
              db.bone_parents, left,
              rotated_global_positions(left.ankle),
              rotated_result, error, static_cast<int>(sizeof(error))),
          error);
    array1d<vec3> checked_global_positions(G1_BoneCount);
    array1d<quat> checked_global_rotations(G1_BoneCount);
    check(g1_ik_checked_forward_kinematics(
              checked_global_positions, checked_global_rotations,
              db.bone_positions(0), rotated_baseline, db.bone_parents,
              error, static_cast<int>(sizeof(error))),
          error);
    vec3 expected_hinge;
    check(ik_checked_quat_rotate(
              expected_hinge,
              checked_global_rotations(left.knee),
              left.knee_hinge_axis_local),
          "checked expected hinge mapping");
    check(g1_test_vec3_same(
              rotated_result.hinge_axis_world, expected_hinge),
          "baseline knee global rotation maps configured local hinge");

    array1d<vec3> baseline_global_positions(G1_BoneCount);
    array1d<quat> baseline_global_rotations(G1_BoneCount);
    check(g1_ik_checked_forward_kinematics(
              baseline_global_positions, baseline_global_rotations,
              db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, error, static_cast<int>(sizeof(error))),
          error);
    array1d<quat> fixed_point_pose(G1_BoneCount);
    g1_test_copy_g1_pose(fixed_point_pose, db.bone_rotations(0));
    fixed_point_pose(right.hip) = quat_from_angle_axis(
        0.11f, vec3(1.0f, 0.0f, 0.0f));
    fixed_point_pose(right.knee) = quat_from_angle_axis(
        0.17f, vec3(0.0f, 0.0f, 1.0f));
    const quat staged_right_hip = fixed_point_pose(right.hip);
    const quat staged_right_knee = fixed_point_pose(right.knee);
    G1LegSolveResult fixed_point_result = {};
    check(g1_apply_named_position_ik(
              fixed_point_pose, db.bone_positions(0),
              db.bone_rotations(0), db.bone_parents, left,
              baseline_global_positions(left.ankle),
              fixed_point_result, error,
              static_cast<int>(sizeof(error))),
          error);
    check(g1_test_quat_bits_same(
              fixed_point_pose(left.hip),
              db.bone_rotations(0, left.hip)) &&
          g1_test_quat_bits_same(
              fixed_point_pose(left.knee),
              db.bone_rotations(0, left.knee)) &&
          g1_test_quat_bits_same(
              fixed_point_pose(right.hip), staged_right_hip) &&
          g1_test_quat_bits_same(
              fixed_point_pose(right.knee), staged_right_knee),
          "current ankle target restores target baseline and preserves staged other leg");
    check(fixed_point_result.applied && fixed_point_result.reachable &&
          !fixed_point_result.correction_limited &&
          !fixed_point_result.safe_stop_requested &&
          fixed_point_result.max_correction_radians == 0.0f &&
          g1_test_vec3_same(
              fixed_point_result.requested_ankle_target,
              fixed_point_result.clamped_ankle_target),
          "current ankle target is an explicit zero-correction no-op");

    g1_test_copy_g1_pose(fixed_point_pose, db.bone_rotations(0));
    fixed_point_pose(right.hip) = staged_right_hip;
    fixed_point_pose(right.knee) = staged_right_knee;
    const vec3 inward_target =
        baseline_global_positions(left.ankle) +
        vec3(0.0f, 0.001f, 0.0f);
    check(g1_apply_named_position_ik(
              fixed_point_pose, db.bone_positions(0),
              db.bone_rotations(0), db.bone_parents, left,
              inward_target, fixed_point_result, error,
              static_cast<int>(sizeof(error))),
          error);
    check(fixed_point_result.reachable &&
          !fixed_point_result.safe_stop_requested &&
          fixed_point_result.max_correction_radians < 0.11f &&
          g1_test_quat_bits_same(
              fixed_point_pose(right.hip), staged_right_hip) &&
          g1_test_quat_bits_same(
              fixed_point_pose(right.knee), staged_right_knee),
          "one-millimeter inward update moves away from singularity without snapping");

    g1_test_copy_g1_pose(fixed_point_pose, db.bone_rotations(0));
    fixed_point_pose(right.hip) = staged_right_hip;
    fixed_point_pose(right.knee) = staged_right_knee;
    const vec3 outward_target =
        baseline_global_positions(left.ankle) -
        vec3(0.0f, 0.001f, 0.0f);
    check(g1_apply_named_position_ik(
              fixed_point_pose, db.bone_positions(0),
              db.bone_rotations(0), db.bone_parents, left,
              outward_target, fixed_point_result, error,
              static_cast<int>(sizeof(error))),
          error);
    check(!fixed_point_result.reachable &&
          fixed_point_result.safe_stop_requested &&
          fixed_point_result.max_correction_radians < 0.001f &&
          g1_test_quat_bits_same(
              fixed_point_pose(left.hip),
              db.bone_rotations(0, left.hip)) &&
          g1_test_quat_bits_same(
              fixed_point_pose(left.knee),
              db.bone_rotations(0, left.knee)),
          "one-millimeter farther extension clamps at current pose without snapping");

    g1_test_copy_g1_pose(fixed_point_pose, db.bone_rotations(0));
    fixed_point_pose(right.hip) = staged_right_hip;
    fixed_point_pose(right.knee) = staged_right_knee;
    check(g1_apply_named_contact_position_ik(
              fixed_point_pose, db.bone_positions(0),
              db.bone_rotations(0), db.bone_parents, left,
              baseline_global_positions(left.contact),
              fixed_point_result, error,
              static_cast<int>(sizeof(error))),
          error);
    check(g1_test_quat_bits_same(
              fixed_point_pose(left.hip),
              db.bone_rotations(0, left.hip)) &&
          g1_test_quat_bits_same(
              fixed_point_pose(left.knee),
              db.bone_rotations(0, left.knee)) &&
          g1_test_quat_bits_same(
              fixed_point_pose(right.hip), staged_right_hip) &&
          g1_test_quat_bits_same(
              fixed_point_pose(right.knee), staged_right_knee),
          "current contact target is a target-leg baseline no-op only");
    check(fixed_point_result.iterations == 1 &&
          fixed_point_result.reachable &&
          !fixed_point_result.correction_limited &&
          !fixed_point_result.safe_stop_requested &&
          fixed_point_result.max_correction_radians == 0.0f &&
          fixed_point_result.contact_residual_m == 0.0f,
          "current contact target reports an exact zero residual no-op");

    database folded_db;
    g1_test_make_straight_solver_database(folded_db);
    folded_db.bone_positions(0, left.knee) =
        vec3(0.0f, -0.4f, 0.0f);
    folded_db.bone_positions(0, left.ankle) =
        vec3(0.0f, +0.4f, 0.0f);
    folded_db.bone_positions(0, left.contact) = vec3();
    array1d<vec3> folded_global_positions(G1_BoneCount);
    array1d<quat> folded_global_rotations(G1_BoneCount);
    check(g1_ik_checked_forward_kinematics(
              folded_global_positions, folded_global_rotations,
              folded_db.bone_positions(0),
              folded_db.bone_rotations(0),
              folded_db.bone_parents, error,
              static_cast<int>(sizeof(error))),
          error);
    check(g1_test_vec3_same(
              folded_global_positions(left.hip),
              folded_global_positions(left.ankle)) &&
          g1_test_vec3_same(
              folded_global_positions(left.ankle),
              folded_global_positions(left.contact)),
          "nonzero folded links return ankle and contact exactly to hip");

    array1d<quat> folded_pose = folded_db.bone_rotations(0);
    folded_pose(right.hip) = staged_right_hip;
    folded_pose(right.knee) = staged_right_knee;
    const array1d<quat> folded_pose_before = folded_pose;
    G1LegSolveResult folded_result = {};
    check(g1_apply_named_position_ik(
              folded_pose, folded_db.bone_positions(0),
              folded_db.bone_rotations(0), folded_db.bone_parents,
              left, folded_global_positions(left.ankle),
              folded_result, error,
              static_cast<int>(sizeof(error))),
          error);
    check(g1_test_pose_bytes_same(folded_pose, folded_pose_before) &&
          folded_result.applied && folded_result.iterations == 1 &&
          folded_result.reachable &&
          !folded_result.correction_limited &&
          !folded_result.safe_stop_requested &&
          folded_result.clamped_distance_m == 0.0f &&
          folded_result.max_correction_radians == 0.0f,
          "exact folded ankle target is a valid named zero-correction fixed point");

    folded_pose = folded_pose_before;
    folded_result = {};
    check(g1_apply_named_contact_position_ik(
              folded_pose, folded_db.bone_positions(0),
              folded_db.bone_rotations(0), folded_db.bone_parents,
              left, folded_global_positions(left.contact),
              folded_result, error,
              static_cast<int>(sizeof(error))),
          error);
    check(g1_test_pose_bytes_same(folded_pose, folded_pose_before) &&
          folded_result.applied && folded_result.iterations == 1 &&
          folded_result.reachable &&
          !folded_result.correction_limited &&
          !folded_result.safe_stop_requested &&
          folded_result.clamped_distance_m == 0.0f &&
          folded_result.max_correction_radians == 0.0f &&
          folded_result.contact_residual_m == 0.0f,
          "exact folded contact target is a valid named zero-residual fixed point");
}

static void test_named_solver_preflight_and_rollback()
{
    database db;
    g1_test_make_straight_solver_database(db);
    const G1LegConfig leg = g1_left_leg_config();
    const vec3 target(0.0f, -0.7f, 0.0f);
    char error[256] = {};

    array1d<quat> valid_working = db.bone_rotations(0);
    const array1d<quat> valid_before = valid_working;
    G1LegSolveResult result = g1_test_leg_result_sentinel();
    const G1LegSolveResult result_before = result;

    check(!g1_apply_named_position_ik(
              slice1d<quat>(G1_BoneCount, NULL),
              db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, target,
              result, error, static_cast<int>(sizeof(error))),
          "null working-pose pointer rejected before indexing");
    check(g1_test_leg_solve_result_same(result, result_before),
          "null working pointer preserves result sentinel");

    result = result_before;
    check(!g1_apply_named_position_ik(
              valid_working,
              slice1d<vec3>(G1_BoneCount, NULL),
              db.bone_rotations(0), db.bone_parents,
              leg, target, result,
              error, static_cast<int>(sizeof(error))),
          "null local-position pointer rejected before FK");
    check(g1_test_pose_bytes_same(valid_working, valid_before) &&
          g1_test_leg_solve_result_same(result, result_before),
          "null local pointer rolls back pose and result");

    result = result_before;
    check(!g1_apply_named_position_ik(
              valid_working, db.bone_positions(0),
              slice1d<quat>(G1_BoneCount, NULL),
              db.bone_parents, leg, target, result,
              error, static_cast<int>(sizeof(error))),
          "null baseline-rotation pointer rejected before FK");
    check(g1_test_pose_bytes_same(valid_working, valid_before) &&
          g1_test_leg_solve_result_same(result, result_before),
          "null baseline pointer rolls back pose and result");

    result = result_before;
    check(!g1_apply_named_position_ik(
              valid_working, db.bone_positions(0),
              db.bone_rotations(0),
              slice1d<int>(G1_BoneCount, NULL),
              leg, target, result,
              error, static_cast<int>(sizeof(error))),
          "null parent pointer rejected before indexing");
    check(g1_test_pose_bytes_same(valid_working, valid_before) &&
          g1_test_leg_solve_result_same(result, result_before),
          "null parent pointer rolls back pose and result");

    valid_working = valid_before;
    result = result_before;
    check(!g1_apply_named_position_ik(
              slice1d<quat>(G1_BoneCount - 1, valid_working.data),
              db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, target, result,
              error, static_cast<int>(sizeof(error))),
          "short working-pose slice rejected before indexing");
    check(g1_test_pose_bytes_same(valid_working, valid_before) &&
          g1_test_leg_solve_result_same(result, result_before),
          "short working slice rolls back pose and result");

    valid_working = valid_before;
    result = result_before;
    check(!g1_apply_named_position_ik(
              valid_working,
              slice1d<vec3>(
                  G1_BoneCount - 1, db.bone_positions(0).data),
              db.bone_rotations(0), db.bone_parents,
              leg, target, result,
              error, static_cast<int>(sizeof(error))),
          "short local-position slice rejected before FK");
    check(g1_test_pose_bytes_same(valid_working, valid_before) &&
          g1_test_leg_solve_result_same(result, result_before),
          "short local slice rolls back pose and result");

    valid_working = valid_before;
    result = result_before;
    check(!g1_apply_named_position_ik(
              valid_working, db.bone_positions(0),
              slice1d<quat>(
                  G1_BoneCount - 1, db.bone_rotations(0).data),
              db.bone_parents, leg, target, result,
              error, static_cast<int>(sizeof(error))),
          "short baseline-rotation slice rejected before FK");
    check(g1_test_pose_bytes_same(valid_working, valid_before) &&
          g1_test_leg_solve_result_same(result, result_before),
          "short baseline slice rolls back pose and result");

    valid_working = valid_before;
    result = result_before;
    check(!g1_apply_named_position_ik(
              valid_working, db.bone_positions(0),
              db.bone_rotations(0),
              slice1d<int>(G1_BoneCount - 1, db.bone_parents.data),
              leg, target, result,
              error, static_cast<int>(sizeof(error))),
          "short parent slice rejected before indexing");
    check(g1_test_pose_bytes_same(valid_working, valid_before) &&
          g1_test_leg_solve_result_same(result, result_before),
          "short parent slice rolls back pose and result");

    array1d<quat> aliased = db.bone_rotations(0);
    const array1d<quat> aliased_before = aliased;
    result = result_before;
    check(!g1_apply_named_position_ik(
              aliased, db.bone_positions(0), aliased,
              db.bone_parents, leg, target, result,
              error, static_cast<int>(sizeof(error))),
          "working/baseline alias rejected");
    check(g1_test_pose_bytes_same(aliased, aliased_before) &&
          g1_test_leg_solve_result_same(result, result_before),
          "alias failure rolls back pose and result");

    array1d<quat> partial_alias(G1_BoneCount + 1);
    partial_alias.set(quat());
    const array1d<quat> partial_alias_before = partial_alias;
    result = result_before;
    check(!g1_apply_named_position_ik(
              slice1d<quat>(
                  G1_BoneCount, partial_alias.data + 1),
              db.bone_positions(0),
              slice1d<quat>(G1_BoneCount, partial_alias.data),
              db.bone_parents, leg, target, result,
              error, static_cast<int>(sizeof(error))),
          "partially overlapping working/baseline ranges rejected");
    check(g1_test_pose_bytes_same(
              partial_alias, partial_alias_before) &&
          g1_test_leg_solve_result_same(result, result_before),
          "partial alias failure rolls back all storage and result");

    array1d<quat> overlapping_fk_outputs(G1_BoneCount);
    overlapping_fk_outputs.set(quat(0.5f, 0.5f, 0.5f, 0.5f));
    const array1d<quat> overlapping_fk_outputs_before =
        overlapping_fk_outputs;
    check(!g1_ik_checked_forward_kinematics(
              slice1d<vec3>(
                  G1_BoneCount,
                  reinterpret_cast<vec3*>(overlapping_fk_outputs.data)),
              slice1d<quat>(
                  G1_BoneCount, overlapping_fk_outputs.data),
              db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, error,
              static_cast<int>(sizeof(error))),
          "overlapping checked-FK output ranges are rejected");
    check(g1_test_pose_bytes_same(
              overlapping_fk_outputs,
              overlapping_fk_outputs_before),
          "checked-FK output alias failure preserves complete storage");

    array1d<quat> partially_overlapping_fk_outputs(
        G1_BoneCount + 1);
    partially_overlapping_fk_outputs.set(
        quat(0.5f, 0.5f, 0.5f, 0.5f));
    const array1d<quat> partially_overlapping_fk_outputs_before =
        partially_overlapping_fk_outputs;
    check(!g1_ik_checked_forward_kinematics(
              slice1d<vec3>(
                  G1_BoneCount,
                  reinterpret_cast<vec3*>(
                      partially_overlapping_fk_outputs.data)),
              slice1d<quat>(
                  G1_BoneCount,
                  partially_overlapping_fk_outputs.data + 1),
              db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, error,
              static_cast<int>(sizeof(error))),
          "partially overlapping checked-FK outputs are rejected");
    check(g1_test_pose_bytes_same(
              partially_overlapping_fk_outputs,
              partially_overlapping_fk_outputs_before),
          "partial output alias preserves every shared storage bit");

    array1d<vec3> output_position_local_position_alias =
        db.bone_positions(0);
    const array1d<vec3> output_position_local_position_before =
        output_position_local_position_alias;
    array1d<quat> disjoint_fk_rotations(G1_BoneCount);
    check(!g1_ik_checked_forward_kinematics(
              output_position_local_position_alias,
              disjoint_fk_rotations,
              output_position_local_position_alias,
              db.bone_rotations(0), db.bone_parents,
              error, static_cast<int>(sizeof(error))),
          "checked-FK position output/local-position alias is rejected");
    check(std::memcmp(
              output_position_local_position_alias.data,
              output_position_local_position_before.data,
              static_cast<size_t>(G1_BoneCount) * sizeof(vec3)) == 0,
          "position input alias failure preserves every storage bit");

    array1d<quat> output_rotation_local_rotation_alias =
        db.bone_rotations(0);
    const array1d<quat> output_rotation_local_rotation_before =
        output_rotation_local_rotation_alias;
    array1d<vec3> disjoint_fk_positions(G1_BoneCount);
    check(!g1_ik_checked_forward_kinematics(
              disjoint_fk_positions,
              output_rotation_local_rotation_alias,
              db.bone_positions(0),
              output_rotation_local_rotation_alias,
              db.bone_parents, error,
              static_cast<int>(sizeof(error))),
          "checked-FK rotation output/local-rotation alias is rejected");
    check(g1_test_pose_bytes_same(
              output_rotation_local_rotation_alias,
              output_rotation_local_rotation_before),
          "rotation input alias failure preserves every storage bit");

    array1d<quat> position_output_rotation_input_alias =
        db.bone_rotations(0);
    const array1d<quat> position_output_rotation_input_before =
        position_output_rotation_input_alias;
    check(!g1_ik_checked_forward_kinematics(
              slice1d<vec3>(
                  G1_BoneCount,
                  reinterpret_cast<vec3*>(
                      position_output_rotation_input_alias.data)),
              disjoint_fk_rotations,
              db.bone_positions(0),
              position_output_rotation_input_alias,
              db.bone_parents, error,
              static_cast<int>(sizeof(error))),
          "checked-FK position output/rotation input alias is rejected");
    check(g1_test_pose_bytes_same(
              position_output_rotation_input_alias,
              position_output_rotation_input_before),
          "cross-type rotation input alias preserves every storage bit");

    array1d<quat> rotation_output_position_input_alias(G1_BoneCount);
    rotation_output_position_input_alias.set(quat());
    const array1d<quat> rotation_output_position_input_before =
        rotation_output_position_input_alias;
    check(!g1_ik_checked_forward_kinematics(
              disjoint_fk_positions,
              rotation_output_position_input_alias,
              slice1d<vec3>(
                  G1_BoneCount,
                  reinterpret_cast<vec3*>(
                      rotation_output_position_input_alias.data)),
              db.bone_rotations(0), db.bone_parents,
              error, static_cast<int>(sizeof(error))),
          "checked-FK rotation output/position input alias is rejected");
    check(g1_test_pose_bytes_same(
              rotation_output_position_input_alias,
              rotation_output_position_input_before),
          "cross-type position input alias preserves every storage bit");

    array1d<int> position_output_parent_input_alias(
        G1_BoneCount * static_cast<int>(sizeof(quat) / sizeof(int)));
    position_output_parent_input_alias.set(0);
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        position_output_parent_input_alias(bone) =
            db.bone_parents(bone);
    }
    const array1d<int> position_output_parent_input_before =
        position_output_parent_input_alias;
    check(!g1_ik_checked_forward_kinematics(
              slice1d<vec3>(
                  G1_BoneCount,
                  reinterpret_cast<vec3*>(
                      position_output_parent_input_alias.data)),
              disjoint_fk_rotations,
              db.bone_positions(0), db.bone_rotations(0),
              slice1d<int>(
                  G1_BoneCount,
                  position_output_parent_input_alias.data),
              error, static_cast<int>(sizeof(error))),
          "checked-FK position output/parent input alias is rejected");
    check(std::memcmp(
              position_output_parent_input_alias.data,
              position_output_parent_input_before.data,
              static_cast<size_t>(
                  position_output_parent_input_alias.size) *
                  sizeof(int)) == 0,
          "position/parent alias preserves every storage bit");

    array1d<int> rotation_output_parent_input_alias(
        G1_BoneCount * static_cast<int>(sizeof(quat) / sizeof(int)));
    rotation_output_parent_input_alias.set(0);
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        rotation_output_parent_input_alias(bone) =
            db.bone_parents(bone);
    }
    const array1d<int> rotation_output_parent_input_before =
        rotation_output_parent_input_alias;
    check(!g1_ik_checked_forward_kinematics(
              disjoint_fk_positions,
              slice1d<quat>(
                  G1_BoneCount,
                  reinterpret_cast<quat*>(
                      rotation_output_parent_input_alias.data)),
              db.bone_positions(0), db.bone_rotations(0),
              slice1d<int>(
                  G1_BoneCount,
                  rotation_output_parent_input_alias.data),
              error, static_cast<int>(sizeof(error))),
          "checked-FK rotation output/parent input alias is rejected");
    check(std::memcmp(
              rotation_output_parent_input_alias.data,
              rotation_output_parent_input_before.data,
              static_cast<size_t>(
                  rotation_output_parent_input_alias.size) *
                  sizeof(int)) == 0,
          "rotation/parent alias preserves every storage bit");

    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        array1d<int> bad_parents = db.bone_parents;
        bad_parents(bone) = bad_parents(bone) == -1 ? bone : -1;
        valid_working = db.bone_rotations(0);
        const array1d<quat> pose_before = valid_working;
        result = result_before;
        check(!g1_apply_named_position_ik(
                  valid_working, db.bone_positions(0),
                  db.bone_rotations(0), bad_parents,
                  leg, target, result,
                  error, static_cast<int>(sizeof(error))),
              "every complete-parent-topology corruption rejected");
        check(g1_test_pose_bytes_same(valid_working, pose_before) &&
              g1_test_leg_solve_result_same(result, result_before),
              "parent/cycle failure rolls back pose and result");
    }

    G1LegConfig bad_configs[7] = {
        leg, leg, leg, leg, leg, leg, leg
    };
    bad_configs[0].name = NULL;
    bad_configs[1].name = "right";
    bad_configs[2].hip = -1;
    bad_configs[3].knee = G1_BoneCount;
    bad_configs[4].ankle = leg.hip;
    bad_configs[5].contact = leg.knee;
    bad_configs[6].knee_hinge_axis_local = vec3();
    for (size_t i = 0;
         i < sizeof(bad_configs) / sizeof(bad_configs[0]);
         ++i) {
        valid_working = db.bone_rotations(0);
        const array1d<quat> pose_before = valid_working;
        result = result_before;
        check(!g1_apply_named_position_ik(
                  valid_working, db.bone_positions(0),
                  db.bone_rotations(0), db.bone_parents,
                  bad_configs[i], target, result,
                  error, static_cast<int>(sizeof(error))),
              "bad name/index/config/chain rejected before indexing");
        check(g1_test_pose_bytes_same(valid_working, pose_before) &&
              g1_test_leg_solve_result_same(result, result_before),
              "bad config rolls back pose and result");
    }

    const quat invalid_quaternions[] = {
        quat(0.0f, 0.0f, 0.0f, 0.0f),
        quat(2.0f, 0.0f, 0.0f, 0.0f),
        quat(g1_test_float_from_bits(UINT32_C(0x7fc00001)),
             0.0f, 0.0f, 0.0f),
        quat(std::numeric_limits<float>::infinity(),
             0.0f, 0.0f, 0.0f)
    };
    for (size_t i = 0;
         i < sizeof(invalid_quaternions) /
             sizeof(invalid_quaternions[0]);
         ++i) {
        array1d<quat> bad_baseline(G1_BoneCount);
        g1_test_copy_g1_pose(bad_baseline, db.bone_rotations(0));
        bad_baseline(leg.knee) = invalid_quaternions[i];
        g1_test_copy_g1_pose(valid_working, db.bone_rotations(0));
        const array1d<quat> valid_pose_before = valid_working;
        result = result_before;
        check(!g1_apply_named_position_ik(
                  valid_working, db.bone_positions(0), bad_baseline,
                  db.bone_parents, leg, target, result,
                  error, static_cast<int>(sizeof(error))),
              "invalid baseline quaternion rejected");
        check(g1_test_pose_bytes_same(
                  valid_working, valid_pose_before) &&
              g1_test_leg_solve_result_same(result, result_before),
              "invalid baseline quaternion rolls back outputs");

        array1d<quat> bad_working(G1_BoneCount);
        g1_test_copy_g1_pose(bad_working, db.bone_rotations(0));
        bad_working(G1_RightWrist) = invalid_quaternions[i];
        const array1d<quat> bad_pose_before = bad_working;
        result = result_before;
        check(!g1_apply_named_position_ik(
                  bad_working, db.bone_positions(0),
                  db.bone_rotations(0), db.bone_parents,
                  leg, target, result,
                  error, static_cast<int>(sizeof(error))),
              "invalid working quaternion rejected");
        check(g1_test_pose_bytes_same(bad_working, bad_pose_before) &&
              g1_test_leg_solve_result_same(result, result_before),
              "invalid working quaternion remains byte-identical");
    }

    array1d<vec3> bad_positions = db.bone_positions(0);
    bad_positions(leg.knee).x =
        g1_test_float_from_bits(UINT32_C(0x00000001));
    valid_working = db.bone_rotations(0);
    const array1d<quat> subnormal_pose_before = valid_working;
    result = result_before;
    check(!g1_apply_named_position_ik(
              valid_working, bad_positions, db.bone_rotations(0),
              db.bone_parents, leg, target, result,
              error, static_cast<int>(sizeof(error))),
          "subnormal local position rejected before FK");
    check(g1_test_pose_bytes_same(
              valid_working, subnormal_pose_before) &&
          g1_test_leg_solve_result_same(result, result_before),
          "subnormal local position rolls back outputs");

    bad_positions = db.bone_positions(0);
    bad_positions(G1_Simulation) = vec3(
        std::numeric_limits<float>::max(), 0.0f, 0.0f);
    bad_positions(G1_Hips) = vec3(
        std::numeric_limits<float>::max(), 0.0f, 0.0f);
    valid_working = db.bone_rotations(0);
    const array1d<quat> overflow_pose_before = valid_working;
    result = result_before;
    check(!g1_apply_named_position_ik(
              valid_working, bad_positions, db.bone_rotations(0),
              db.bone_parents, leg, target, result,
              error, static_cast<int>(sizeof(error))),
          "finite baseline-FK position overflow rejected");
    check(g1_test_pose_bytes_same(valid_working, overflow_pose_before) &&
          g1_test_leg_solve_result_same(result, result_before),
          "baseline-FK failure rolls back pose and result");

    bad_positions = db.bone_positions(0);
    bad_positions(leg.contact) = vec3(2.9e38f, 2.9e38f, 0.0f);
    array1d<vec3> candidate_overflow_positions(G1_BoneCount);
    array1d<quat> candidate_overflow_rotations(G1_BoneCount);
    check(g1_ik_checked_forward_kinematics(
              candidate_overflow_positions,
              candidate_overflow_rotations,
              bad_positions, db.bone_rotations(0), db.bone_parents,
              error, static_cast<int>(sizeof(error))),
          "candidate-only overflow fixture has valid baseline FK");
    valid_working = db.bone_rotations(0);
    const array1d<quat> candidate_overflow_pose_before = valid_working;
    result = result_before;
    check(!g1_apply_named_position_ik(
              valid_working, bad_positions, db.bone_rotations(0),
              db.bone_parents, leg,
              vec3(0.2f, -0.7f, 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          "candidate-only post-solve FK overflow rejected");
    check(g1_test_pose_bytes_same(
              valid_working, candidate_overflow_pose_before) &&
          g1_test_leg_solve_result_same(result, result_before),
          "candidate post-FK failure rolls back pose and result");

    valid_working = db.bone_rotations(0);
    const array1d<quat> target_pose_before = valid_working;
    result = result_before;
    check(!g1_apply_named_position_ik(
              valid_working, db.bone_positions(0),
              db.bone_rotations(0), db.bone_parents,
              leg,
              vec3(g1_test_float_from_bits(UINT32_C(0x7fc00001)),
                   0.0f, 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          "non-finite named target rejected");
    check(g1_test_pose_bytes_same(valid_working, target_pose_before) &&
          g1_test_leg_solve_result_same(result, result_before),
          "non-finite target rolls back pose and result");
}

static void test_named_contact_residual_contract()
{
    database db;
    make_g1_database(db);
    const G1LegConfig leg = g1_left_leg_config();
    char error[256] = {};

    check(g1_ik_contact_residual_is_converged(0.0f) &&
          g1_ik_contact_residual_is_converged(0.005f) &&
          !g1_ik_contact_residual_is_converged(std::nextafter(
              0.005f, std::numeric_limits<float>::infinity())),
          "contact convergence owns exact 0.005 m threshold");

    double precise_over_limit = 0.0;
    float rounded_at_limit = 0.0f;
    check(ik_checked_distance_precise(
              precise_over_limit, rounded_at_limit,
              vec3(0.005f, 1.0e-6f, 0.0f), vec3()) &&
          rounded_at_limit == 0.005f &&
          precise_over_limit > static_cast<double>(0.005f),
          "fixed residual reports 5 mm while promoted value exceeds it");
    check(!g1_ik_contact_residual_is_converged_precise(
              precise_over_limit),
          "contact convergence classifies the retained promoted residual");

    array1d<quat> reference_pose = db.bone_rotations(0);
    G1LegSolveResult reference_result = {};
    check(g1_apply_named_position_ik(
              reference_pose, db.bone_positions(0),
              db.bone_rotations(0), db.bone_parents,
              leg, vec3(-0.05f, -0.43f, 0.0f),
              reference_result, error, static_cast<int>(sizeof(error))),
          error);
    array1d<vec3> reference_global_positions;
    array1d<quat> reference_global_rotations;
    g1_test_global_pose(
        reference_global_positions, reference_global_rotations,
        db, reference_pose);
    const vec3 desired_contact =
        reference_global_positions(leg.contact);

    array1d<quat> output = db.bone_rotations(0);
    G1LegSolveResult result = {};
    check(g1_apply_named_contact_position_ik(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, desired_contact,
              result, error, static_cast<int>(sizeof(error))),
          error);
    check(result.applied && result.iterations >= 1 &&
          result.iterations <= 4,
          "contact solve reports bounded iteration ownership");
    array1d<vec3> final_global_positions;
    array1d<quat> final_global_rotations;
    final_global_positions.resize(G1_BoneCount);
    final_global_rotations.resize(G1_BoneCount);
    check(g1_ik_checked_forward_kinematics(
              final_global_positions, final_global_rotations,
              db.bone_positions(0), output, db.bone_parents,
              error, static_cast<int>(sizeof(error))),
          error);
    float fresh_residual = 0.0f;
    check(ik_checked_distance(
              fresh_residual,
              final_global_positions(leg.contact), desired_contact),
          "fresh committed contact residual is measurable");
    check(g1_test_float_same(
              result.contact_residual_m, fresh_residual),
          "reported residual equals fresh FK of committed pose");

    output = db.bone_rotations(0);
    check(g1_apply_named_contact_position_ik(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, vec3(3.0f, 0.0f, 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          error);
    check(result.iterations == 4 && result.safe_stop_requested &&
          result.contact_residual_m > 0.005f,
          "unreachable contact commits closest result at four-iteration cap");

    output = db.bone_rotations(0);
    const array1d<quat> late_pose_before = output;
    G1LegSolveResult late_result = g1_test_leg_result_sentinel();
    const G1LegSolveResult late_result_before = late_result;
    check(!g1_apply_named_contact_position_ik(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg,
              vec3(1.0e19f, 0.0f, 0.0f),
              late_result, error, static_cast<int>(sizeof(error))),
          "later residual-iteration overflow fails transactionally");
    check(g1_test_pose_bytes_same(output, late_pose_before) &&
          g1_test_leg_solve_result_same(
              late_result, late_result_before),
          "later residual failure rolls back complete pose and result");
}

static G1FootOrientationResult g1_test_orientation_result_sentinel()
{
    G1FootOrientationResult result = {};
    result.applied = true;
    result.correction_limited = true;
    result.safe_stop_requested = true;
    result.target_global_rotation = quat(0.5f, 0.5f, 0.5f, 0.5f);
    result.requested_correction_radians = 7.0f;
    result.correction_radians = 8.0f;
    return result;
}

static bool g1_test_orientation_result_same(
    const G1FootOrientationResult& left,
    const G1FootOrientationResult& right)
{
    return left.applied == right.applied &&
           left.correction_limited == right.correction_limited &&
           left.safe_stop_requested == right.safe_stop_requested &&
           g1_test_quat_bits_same(
               left.target_global_rotation,
               right.target_global_rotation) &&
           g1_test_float_same(
               left.requested_correction_radians,
               right.requested_correction_radians) &&
           g1_test_float_same(
               left.correction_radians,
               right.correction_radians);
}

static void g1_test_checked_global_pose(
    array1d<vec3>& positions,
    array1d<quat>& rotations,
    const database& db,
    const array1d<quat>& local_rotations,
    char* error,
    int error_capacity)
{
    positions.resize(G1_BoneCount);
    rotations.resize(G1_BoneCount);
    check(g1_ik_checked_forward_kinematics(
              positions, rotations,
              db.bone_positions(0), local_rotations,
              db.bone_parents, error, error_capacity),
          error);
}

static vec3 g1_test_projected_heading(
    quat current_global_rotation,
    const G1LegConfig& config,
    vec3 surface_normal)
{
    vec3 current_heading;
    double heading_dot = 0.0;
    vec3 normal_component;
    vec3 projected;
    vec3 normalized;
    check(ik_checked_quat_rotate(
              current_heading,
              current_global_rotation,
              config.foot_forward_local) &&
          ik_checked_dot(
              heading_dot, current_heading, surface_normal) &&
          ik_checked_vec3_scale(
              normal_component, surface_normal, heading_dot) &&
          ik_checked_vec3_subtract(
              projected, current_heading, normal_component) &&
          ik_checked_normalize(normalized, projected),
          "test heading projection remains checked");
    return normalized;
}

static void test_surface_aligned_named_foot_orientation()
{
    database db;
    make_g1_database(db);
    const G1LegConfig left = g1_left_leg_config();
    const G1LegConfig right = g1_right_leg_config();
    char error[256] = {};

    array1d<quat> output = db.bone_rotations(0);
    const array1d<quat> flat_before = output;
    G1FootOrientationResult result = {};
    check(g1_apply_named_foot_orientation(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, left, vec3(0.0f, 1.0f, 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          error);
    check(result.applied && !result.correction_limited &&
          !result.safe_stop_requested &&
          result.requested_correction_radians == 0.0f &&
          result.correction_radians == 0.0f,
          "flat identity orientation is an exact no-op");
    check(g1_test_pose_bytes_same(output, flat_before),
          "flat identity preserves every pose byte");

    const float angle = 10.0f * PIf / 180.0f;
    const vec3 ramp_normal(
        -std::sin(angle), std::cos(angle), 0.0f);
    output = db.bone_rotations(0);
    check(g1_apply_named_foot_orientation(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, left, ramp_normal,
              result, error, static_cast<int>(sizeof(error))),
          error);
    array1d<vec3> global_positions;
    array1d<quat> global_rotations;
    g1_test_checked_global_pose(
        global_positions, global_rotations, db, output,
        error, static_cast<int>(sizeof(error)));
    vec3 ramp_up;
    vec3 ramp_forward;
    check(ik_checked_quat_rotate(
              ramp_up, global_rotations(left.contact),
              left.sole_normal_local) &&
          ik_checked_quat_rotate(
              ramp_forward, global_rotations(left.contact),
              left.foot_forward_local),
          "ramp foot axes remain checked");
    check(dot(ramp_up, ramp_normal) > 0.9999f,
          "left foot aligns to longitudinal ramp normal");
    check(dot(
              ramp_forward,
              vec3(std::cos(angle), std::sin(angle), 0.0f)) > 0.9999f &&
          std::fabs(dot(ramp_forward, ramp_normal)) < 1.0e-5f,
          "longitudinal ramp preserves tangent heading");

    const vec3 cross_normal(
        0.0f, std::cos(angle), -std::sin(angle));
    output = db.bone_rotations(0);
    check(g1_apply_named_foot_orientation(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, left, cross_normal,
              result, error, static_cast<int>(sizeof(error))),
          error);
    g1_test_checked_global_pose(
        global_positions, global_rotations, db, output,
        error, static_cast<int>(sizeof(error)));
    vec3 cross_up;
    vec3 cross_forward;
    check(ik_checked_quat_rotate(
              cross_up, global_rotations(left.contact),
              left.sole_normal_local) &&
          ik_checked_quat_rotate(
              cross_forward, global_rotations(left.contact),
              left.foot_forward_local),
          "cross-slope foot axes remain checked");
    check(dot(cross_up, cross_normal) > 0.9999f,
          "left foot aligns to cross-slope normal");
    check(cross_forward.x > 0.999f &&
          std::fabs(dot(cross_forward, cross_normal)) < 1.0e-5f,
          "cross-slope preserves tangent heading");

    const vec3 mirrored_cross_normal(
        0.0f, std::cos(angle), std::sin(angle));
    output = db.bone_rotations(0);
    check(g1_apply_named_foot_orientation(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, right, mirrored_cross_normal,
              result, error, static_cast<int>(sizeof(error))),
          error);
    g1_test_checked_global_pose(
        global_positions, global_rotations, db, output,
        error, static_cast<int>(sizeof(error)));
    vec3 right_up;
    check(ik_checked_quat_rotate(
              right_up, global_rotations(right.contact),
              right.sole_normal_local),
          "right-foot normal remains checked");
    check(dot(right_up, mirrored_cross_normal) > 0.9999f &&
          result.applied && !result.safe_stop_requested,
          "mirrored right foot aligns without a limit");
}

static void test_surface_orientation_staged_pose_and_fallback()
{
    database db;
    make_g1_database(db);
    const G1LegConfig left = g1_left_leg_config();
    const G1LegConfig right = g1_right_leg_config();
    char error[256] = {};

    array1d<quat> working = db.bone_rotations(0);
    working(G1_Hips) = quat_from_angle_axis(
        0.08f, vec3(0.0f, 1.0f, 0.0f));
    working(left.hip) = quat_from_angle_axis(
        0.03f, vec3(0.0f, 0.0f, 1.0f));
    working(left.knee) = quat_from_angle_axis(
        -0.02f, vec3(0.0f, 0.0f, 1.0f));
    working(left.contact) = quat_from_angle_axis(
        0.12f, vec3(0.0f, 1.0f, 0.0f));
    working(right.hip) = quat_from_angle_axis(
        0.06f, vec3(1.0f, 0.0f, 0.0f));
    working(right.knee) = quat_from_angle_axis(
        -0.04f, vec3(0.0f, 0.0f, 1.0f));
    const array1d<quat> before = working;

    array1d<vec3> before_global_positions;
    array1d<quat> before_global_rotations;
    g1_test_checked_global_pose(
        before_global_positions, before_global_rotations,
        db, working, error, static_cast<int>(sizeof(error)));
    const float angle = 6.0f * PIf / 180.0f;
    const vec3 surface_normal(
        0.0f, std::cos(angle), -std::sin(angle));
    const vec3 expected_heading = g1_test_projected_heading(
        before_global_rotations(left.contact), left, surface_normal);

    G1FootOrientationResult result = {};
    check(g1_apply_named_foot_orientation(
              working, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, left, surface_normal,
              result, error, static_cast<int>(sizeof(error))),
          error);
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (bone != left.contact) {
            check(g1_test_quat_bits_same(working(bone), before(bone)),
                  "orientation changes only the named contact bone");
        }
    }
    check(g1_test_quat_bits_same(
              working(right.hip), before(right.hip)) &&
          g1_test_quat_bits_same(
              working(right.knee), before(right.knee)),
          "orientation preserves a staged second leg");

    array1d<vec3> after_global_positions;
    array1d<quat> after_global_rotations;
    g1_test_checked_global_pose(
        after_global_positions, after_global_rotations,
        db, working, error, static_cast<int>(sizeof(error)));
    vec3 actual_up;
    vec3 actual_forward;
    check(ik_checked_quat_rotate(
              actual_up, after_global_rotations(left.contact),
              left.sole_normal_local) &&
          ik_checked_quat_rotate(
              actual_forward, after_global_rotations(left.contact),
              left.foot_forward_local),
          "staged-pose output axes remain checked");
    check(dot(actual_up, surface_normal) > 0.9999f &&
          dot(actual_forward, expected_heading) > 0.9999f,
          "arbitrary parent/current heading is aligned and preserved");

    working = db.bone_rotations(0);
    working(left.contact) = quat_from_angle_axis(
        0.5f * PIf - 5.0e-7f, vec3(0.0f, 0.0f, 1.0f));
    check(g1_apply_named_foot_orientation(
              working, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, left, vec3(0.0f, 1.0f, 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          error);
    vec3 fallback_forward;
    vec3 fallback_up;
    check(ik_checked_quat_rotate(
              fallback_forward, result.target_global_rotation,
              left.foot_forward_local) &&
          ik_checked_quat_rotate(
              fallback_up, result.target_global_rotation,
              left.sole_normal_local),
          "fallback target axes remain checked");
    check(fallback_forward.x > 0.9999f &&
          dot(fallback_up, vec3(0.0f, 1.0f, 0.0f)) > 0.9999f,
          "near-degenerate projected heading uses deterministic tangent fallback");

    const quat vertical_heading = quat_from_angle_axis(
        0.5f * PIf, vec3(0.0f, 0.0f, 1.0f));
    vec3 exact_direction;
    vec3 exact_normal;
    check(ik_checked_quat_rotate(
              exact_direction, vertical_heading,
              left.foot_forward_local) &&
          ik_checked_normalize(exact_normal, exact_direction),
          "exact-degenerate fixture remains checked");
    quat exact_target = quat(0.5f, 0.5f, 0.5f, 0.5f);
    check(g1_surface_aligned_foot_rotation(
              exact_target, vertical_heading, left, exact_normal,
              error, static_cast<int>(sizeof(error))),
          error);
    vec3 exact_target_up;
    check(ik_checked_quat_rotate(
              exact_target_up, exact_target,
              left.sole_normal_local),
          "exact-degenerate fallback target remains checked");
    check(dot(exact_target_up, exact_normal) > 0.9999f,
          "exact-degenerate heading also uses a valid deterministic fallback");

    const quat nonidentity = quat_from_angle_axis(
        1.1f, vec3(0.0f, 1.0f, 0.0f));
    const quat scaled_nonidentity = nonidentity * 0.99999f;
    check(ik_quat_is_unit(scaled_nonidentity),
          "scaled non-identity fixture is inside unit tolerance");
    quat normalized_nonidentity;
    check(ik_checked_quat_normalize(
              normalized_nonidentity, scaled_nonidentity),
          "scaled non-identity fixture has a checked normalized equivalent");
    const vec3 comparison_normal(
        0.0f, std::cos(angle), -std::sin(angle));
    quat scaled_target;
    quat normalized_target;
    check(g1_surface_aligned_foot_rotation(
              scaled_target, scaled_nonidentity, left,
              comparison_normal, error,
              static_cast<int>(sizeof(error))) &&
          g1_surface_aligned_foot_rotation(
              normalized_target, normalized_nonidentity, left,
              comparison_normal, error,
              static_cast<int>(sizeof(error))),
          error);
    vec3 scaled_heading;
    vec3 normalized_heading;
    check(ik_checked_quat_rotate(
              scaled_heading, scaled_target,
              left.foot_forward_local) &&
          ik_checked_quat_rotate(
              normalized_heading, normalized_target,
              left.foot_forward_local),
          "scaled and normalized targets expose checked headings");
    check(g1_test_quat_bits_same(scaled_target, normalized_target) &&
          g1_test_vec3_same(scaled_heading, normalized_heading),
          "admitted scaled rotation matches its normalized target and heading");
}

static void test_surface_orientation_limit_semantics()
{
    database db;
    make_g1_database(db);
    const G1LegConfig leg = g1_left_leg_config();
    char error[256] = {};
    array1d<quat> output = db.bone_rotations(0);
    G1FootOrientationResult result = {};

    float boundary_angle = leg.max_correction_radians;
    for (int step = 0; step < 64; ++step) {
        const vec3 normal(
            -std::sin(boundary_angle),
            std::cos(boundary_angle), 0.0f);
        g1_test_copy_g1_pose(output, db.bone_rotations(0));
        check(g1_apply_named_foot_orientation(
                  output, db.bone_positions(0), db.bone_rotations(0),
                  db.bone_parents, leg, normal,
                  result, error, static_cast<int>(sizeof(error))),
              error);
        if (!result.correction_limited) break;
        boundary_angle = std::nextafter(boundary_angle, 0.0f);
    }
    check(!result.correction_limited && !result.safe_stop_requested &&
          result.requested_correction_radians <=
              leg.max_correction_radians &&
          result.correction_radians <= leg.max_correction_radians,
          "largest discovered boundary correction remains unlimited");

    float over_angle = std::nextafter(
        boundary_angle, std::numeric_limits<float>::infinity());
    for (int step = 0; step < 64; ++step) {
        const vec3 normal(
            -std::sin(over_angle), std::cos(over_angle), 0.0f);
        g1_test_copy_g1_pose(output, db.bone_rotations(0));
        check(g1_apply_named_foot_orientation(
                  output, db.bone_positions(0), db.bone_rotations(0),
                  db.bone_parents, leg, normal,
                  result, error, static_cast<int>(sizeof(error))),
              error);
        if (result.correction_limited) break;
        over_angle = std::nextafter(
            over_angle, std::numeric_limits<float>::infinity());
    }
    check(result.correction_limited && result.safe_stop_requested &&
          result.requested_correction_radians >
              leg.max_correction_radians &&
          result.correction_radians <= leg.max_correction_radians,
          "first discovered over-limit correction clamps and safe-stops");
    IKClampResult independent = {};
    check(ik_clamp_local_delta(
              independent,
              db.bone_rotations(0, leg.contact),
              result.target_global_rotation,
              leg.max_correction_radians),
          "orientation result remains independently clampable");
    check(g1_test_quat_bits_same(
              output(leg.contact), independent.value) &&
          g1_test_float_same(
              result.requested_correction_radians,
              independent.requested_radians) &&
          g1_test_float_same(
              result.correction_radians,
              independent.actual_radians) &&
          result.correction_limited == independent.limited,
          "orientation reports exact generic-clamp semantics");

    const float steep = 45.0f * PIf / 180.0f;
    output = db.bone_rotations(0);
    const array1d<quat> steep_before = output;
    check(g1_apply_named_foot_orientation(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg,
              vec3(-std::sin(steep), std::cos(steep), 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          error);
    check(result.correction_limited && result.safe_stop_requested &&
          result.correction_radians <= leg.max_correction_radians,
          "steep correction remains bounded and requests safe stop");
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (bone != leg.contact) {
            check(g1_test_quat_bits_same(
                      output(bone), steep_before(bone)),
                  "limited orientation changes only named contact");
        }
    }
}

static void test_surface_orientation_hostile_rollback()
{
    database db;
    make_g1_database(db);
    const G1LegConfig leg = g1_left_leg_config();
    char error[256] = {};
    const float nan = g1_test_float_from_bits(UINT32_C(0x7fc00001));
    const float subnormal = g1_test_float_from_bits(UINT32_C(0x00000001));

    quat aliased_direct = quat(0.5f, 0.5f, 0.5f, 0.5f);
    const quat aliased_direct_before = aliased_direct;
    char* const direct_error_alias =
        reinterpret_cast<char*>(&aliased_direct) + 3;
    check(!g1_surface_aligned_foot_rotation(
              aliased_direct, quat(), leg, vec3(),
              direct_error_alias, 1),
          "direct diagnostic aliasing quaternion is rejected");
    check(g1_test_quat_bits_same(
              aliased_direct, aliased_direct_before),
          "direct diagnostic alias preserves quaternion bytes");

    const vec3 invalid_normals[] = {
        vec3(),
        vec3(1.0f, 0.0f, 0.0f),
        vec3(0.0f, -1.0f, 0.0f),
        vec3(0.0f, 2.0f, 0.0f),
        vec3(0.0f, 1.001f, 0.0f),
        vec3(subnormal, 1.0f, 0.0f),
        vec3(nan, 1.0f, 0.0f),
        vec3(std::numeric_limits<float>::infinity(), 1.0f, 0.0f)
    };
    for (size_t index = 0;
         index < sizeof(invalid_normals) / sizeof(invalid_normals[0]);
         ++index) {
        quat direct = quat(0.5f, 0.5f, 0.5f, 0.5f);
        const quat direct_before = direct;
        check(!g1_surface_aligned_foot_rotation(
                  direct, quat(), leg, invalid_normals[index],
                  error, static_cast<int>(sizeof(error))),
              "invalid surface normal is rejected");
        check(g1_test_quat_bits_same(direct, direct_before),
              "direct normal failure preserves quaternion output");

        array1d<quat> pose = db.bone_rotations(0);
        const array1d<quat> pose_before = pose;
        G1FootOrientationResult result =
            g1_test_orientation_result_sentinel();
        const G1FootOrientationResult result_before = result;
        check(!g1_apply_named_foot_orientation(
                  pose, db.bone_positions(0), db.bone_rotations(0),
                  db.bone_parents, leg, invalid_normals[index],
                  result, error, static_cast<int>(sizeof(error))),
              "invalid normal fails named orientation");
        check(g1_test_pose_bytes_same(pose, pose_before) &&
              g1_test_orientation_result_same(result, result_before),
              "invalid normal rolls back pose and result");
    }

    const quat invalid_quaternions[] = {
        quat(0.0f, 0.0f, 0.0f, 0.0f),
        quat(2.0f, 0.0f, 0.0f, 0.0f),
        quat(1.0f, subnormal, 0.0f, 0.0f),
        quat(nan, 0.0f, 0.0f, 0.0f),
        quat(std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 0.0f)
    };
    for (size_t index = 0;
         index < sizeof(invalid_quaternions) /
             sizeof(invalid_quaternions[0]);
         ++index) {
        quat direct = quat(0.5f, 0.5f, 0.5f, 0.5f);
        const quat direct_before = direct;
        check(!g1_surface_aligned_foot_rotation(
                  direct, invalid_quaternions[index], leg,
                  vec3(0.0f, 1.0f, 0.0f),
                  error, static_cast<int>(sizeof(error))),
              "invalid current quaternion is rejected");
        check(g1_test_quat_bits_same(direct, direct_before),
              "invalid current quaternion preserves direct output");
    }

    G1LegConfig bad_config = leg;
    bad_config.contact = G1_BoneCount;
    array1d<quat> pose = db.bone_rotations(0);
    const array1d<quat> pose_before = pose;
    G1FootOrientationResult result =
        g1_test_orientation_result_sentinel();
    const G1FootOrientationResult result_before = result;
    check(!g1_apply_named_foot_orientation(
              pose, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, bad_config,
              vec3(0.0f, 1.0f, 0.0f), result,
              error, static_cast<int>(sizeof(error))),
          "out-of-range contact config is rejected before indexing");
    check(g1_test_pose_bytes_same(pose, pose_before) &&
          g1_test_orientation_result_same(result, result_before),
          "invalid config rolls back complete outputs");

    bad_config = leg;
    bad_config.foot_forward_local = vec3(0.0f, 2.0f, 0.0f);
    quat direct = quat(0.5f, 0.5f, 0.5f, 0.5f);
    const quat direct_before = direct;
    check(!g1_surface_aligned_foot_rotation(
              direct, quat(), bad_config,
              vec3(0.0f, 1.0f, 0.0f),
              error, static_cast<int>(sizeof(error))) &&
          g1_test_quat_bits_same(direct, direct_before),
          "malformed configured axes fail direct orientation transactionally");

    bad_config = leg;
    bad_config.name = NULL;
    pose = db.bone_rotations(0);
    result = result_before;
    check(!g1_apply_named_foot_orientation(
              pose, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, bad_config,
              vec3(0.0f, 1.0f, 0.0f), result,
              error, static_cast<int>(sizeof(error))),
          "null config name is rejected before indexing");
    check(g1_test_pose_bytes_same(pose, pose_before) &&
          g1_test_orientation_result_same(result, result_before),
          "null config name preserves outputs");

    const struct {
        int output_size;
        quat* output_data;
        int position_size;
        vec3* position_data;
        int baseline_size;
        quat* baseline_data;
        int parent_size;
        int* parent_data;
    } invalid_slices[] = {
        {G1_BoneCount, NULL, G1_BoneCount, db.bone_positions(0).data,
         G1_BoneCount, db.bone_rotations(0).data,
         G1_BoneCount, db.bone_parents.data},
        {G1_BoneCount - 1, pose.data,
         G1_BoneCount, db.bone_positions(0).data,
         G1_BoneCount, db.bone_rotations(0).data,
         G1_BoneCount, db.bone_parents.data},
        {G1_BoneCount, pose.data, G1_BoneCount, NULL,
         G1_BoneCount, db.bone_rotations(0).data,
         G1_BoneCount, db.bone_parents.data},
        {G1_BoneCount, pose.data,
         G1_BoneCount - 1, db.bone_positions(0).data,
         G1_BoneCount, db.bone_rotations(0).data,
         G1_BoneCount, db.bone_parents.data},
        {G1_BoneCount, pose.data,
         G1_BoneCount, db.bone_positions(0).data,
         G1_BoneCount, NULL,
         G1_BoneCount, db.bone_parents.data},
        {G1_BoneCount, pose.data,
         G1_BoneCount, db.bone_positions(0).data,
         G1_BoneCount - 1, db.bone_rotations(0).data,
         G1_BoneCount, db.bone_parents.data},
        {G1_BoneCount, pose.data,
         G1_BoneCount, db.bone_positions(0).data,
         G1_BoneCount, db.bone_rotations(0).data,
         G1_BoneCount, NULL},
        {G1_BoneCount, pose.data,
         G1_BoneCount, db.bone_positions(0).data,
         G1_BoneCount, db.bone_rotations(0).data,
         G1_BoneCount - 1, db.bone_parents.data}
    };
    for (size_t index = 0;
         index < sizeof(invalid_slices) / sizeof(invalid_slices[0]);
         ++index) {
        pose = db.bone_rotations(0);
        const array1d<quat> before = pose;
        result = result_before;
        check(!g1_apply_named_foot_orientation(
                  slice1d<quat>(
                      invalid_slices[index].output_size,
                      invalid_slices[index].output_data),
                  slice1d<vec3>(
                      invalid_slices[index].position_size,
                      invalid_slices[index].position_data),
                  slice1d<quat>(
                      invalid_slices[index].baseline_size,
                      invalid_slices[index].baseline_data),
                  slice1d<int>(
                      invalid_slices[index].parent_size,
                      invalid_slices[index].parent_data),
                  leg, vec3(0.0f, 1.0f, 0.0f),
                  result, error, static_cast<int>(sizeof(error))),
              "null or short orientation slice is rejected");
        check(g1_test_pose_bytes_same(pose, before) &&
              g1_test_orientation_result_same(result, result_before),
              "slice preflight failure preserves outputs");
    }

    array1d<quat> overlap_storage(G1_BoneCount + 1);
    overlap_storage.set(quat());
    result = result_before;
    check(!g1_apply_named_foot_orientation(
              slice1d<quat>(G1_BoneCount, overlap_storage.data + 1),
              db.bone_positions(0),
              slice1d<quat>(G1_BoneCount, overlap_storage.data),
              db.bone_parents, leg, vec3(0.0f, 1.0f, 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          "partial working/baseline alias is rejected");
    check(g1_test_orientation_result_same(result, result_before),
          "partial alias preserves result output");

    result = result_before;
    check(!g1_apply_named_foot_orientation(
              slice1d<quat>(
                  G1_BoneCount,
                  reinterpret_cast<quat*>(db.bone_positions(0).data)),
              db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, vec3(0.0f, 1.0f, 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          "working pose aliasing local positions is rejected before reads");
    check(g1_test_orientation_result_same(result, result_before),
          "cross-type position alias preserves result");

    result = result_before;
    check(!g1_apply_named_foot_orientation(
              slice1d<quat>(
                  G1_BoneCount,
                  reinterpret_cast<quat*>(db.bone_parents.data)),
              db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, vec3(0.0f, 1.0f, 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          "working pose aliasing parents is rejected before reads");
    check(g1_test_orientation_result_same(result, result_before),
          "cross-type parent alias preserves result");

    G1LegConfig aliased_config = leg;
    unsigned char aliased_config_before[sizeof(G1LegConfig)] = {};
    std::memcpy(
        aliased_config_before, &aliased_config,
        sizeof(aliased_config));
    result = result_before;
    check(!g1_apply_named_foot_orientation(
              slice1d<quat>(
                  G1_BoneCount,
                  reinterpret_cast<quat*>(&aliased_config)),
              db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, aliased_config,
              vec3(0.0f, 1.0f, 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          "working pose aliasing immutable config is rejected before reads");
    check(std::memcmp(
              &aliased_config, aliased_config_before,
              sizeof(aliased_config)) == 0 &&
          g1_test_orientation_result_same(result, result_before),
          "config alias preserves immutable config and result bytes");

    pose = db.bone_rotations(0);
    pose(G1_Spine) = quat(2.0f, 0.0f, 0.0f, 0.0f);
    const array1d<quat> invalid_working_before = pose;
    result = result_before;
    check(!g1_apply_named_foot_orientation(
              pose, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, vec3(0.0f, 1.0f, 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          "invalid staged quaternion is rejected");
    check(g1_test_pose_bytes_same(pose, invalid_working_before) &&
          g1_test_orientation_result_same(result, result_before),
          "invalid staged quaternion rolls back outputs");

    array1d<quat> invalid_baseline = db.bone_rotations(0);
    invalid_baseline(leg.contact) = quat(0.0f, 0.0f, 0.0f, 0.0f);
    pose = db.bone_rotations(0);
    const array1d<quat> invalid_baseline_pose_before = pose;
    result = result_before;
    check(!g1_apply_named_foot_orientation(
              pose, db.bone_positions(0), invalid_baseline,
              db.bone_parents, leg, vec3(0.0f, 1.0f, 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          "invalid immutable baseline quaternion is rejected");
    check(g1_test_pose_bytes_same(pose, invalid_baseline_pose_before) &&
          g1_test_orientation_result_same(result, result_before),
          "invalid baseline quaternion rolls back outputs");

    array1d<quat> tolerated_baseline = db.bone_rotations(0);
    tolerated_baseline(leg.contact) =
        quat(0.99999f, 0.0f, 0.0f, 0.0f);
    pose = tolerated_baseline;
    result = {};
    check(g1_apply_named_foot_orientation(
              pose, db.bone_positions(0), tolerated_baseline,
              db.bone_parents, leg, vec3(0.0f, 1.0f, 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          error);
    check(result.applied &&
          result.requested_correction_radians == 0.0f &&
          result.correction_radians == 0.0f &&
          g1_test_quat_bits_same(pose(leg.contact), quat()),
          "admitted quaternion tolerance is normalized without false correction");

    database bad_parent;
    make_g1_database(bad_parent);
    bad_parent.bone_parents(G1_LeftToe) = G1_LeftKnee;
    pose = db.bone_rotations(0);
    const array1d<quat> bad_parent_before = pose;
    result = result_before;
    check(!g1_apply_named_foot_orientation(
              pose, db.bone_positions(0), db.bone_rotations(0),
              bad_parent.bone_parents, leg,
              vec3(0.0f, 1.0f, 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          "invalid parent topology is rejected");
    check(g1_test_pose_bytes_same(pose, bad_parent_before) &&
          g1_test_orientation_result_same(result, result_before),
          "parent failure rolls back complete outputs");

    database extreme;
    make_g1_database(extreme);
    extreme.bone_positions(0, G1_Hips) =
        vec3(3.0e38f, 3.0e38f, 3.0e38f);
    extreme.bone_positions(0, G1_Spine) =
        vec3(3.0e38f, 3.0e38f, 3.0e38f);
    pose = db.bone_rotations(0);
    const array1d<quat> extreme_before = pose;
    result = result_before;
    check(!g1_apply_named_foot_orientation(
              pose, extreme.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, vec3(0.0f, 1.0f, 0.0f),
              result, error, static_cast<int>(sizeof(error))),
          "unsafe checked-FK intermediate is rejected");
    check(g1_test_pose_bytes_same(pose, extreme_before) &&
          g1_test_orientation_result_same(result, result_before),
          "late checked-FK failure remains transactional");

    char one_byte[1] = {'x'};
    pose = db.bone_rotations(0);
    result = result_before;
    check(!g1_apply_named_foot_orientation(
              pose, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, bad_config,
              vec3(0.0f, 1.0f, 0.0f),
              result, one_byte, 1),
          "short diagnostic buffer is safe");
    check(one_byte[0] == '\0' &&
          g1_test_pose_bytes_same(pose, pose_before) &&
          g1_test_orientation_result_same(result, result_before),
          "short diagnostic preserves transactional outputs");

    pose = db.bone_rotations(0);
    const array1d<quat> aliased_error_before = pose;
    result = result_before;
    char* const aliased_error =
        reinterpret_cast<char*>(pose.data) + 2;
    check(!g1_apply_named_foot_orientation(
              slice1d<quat>(G1_BoneCount - 1, pose.data),
              db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, vec3(0.0f, 1.0f, 0.0f),
              result, aliased_error, 1),
          "shape diagnostic aliasing pose is rejected without writing");
    check(g1_test_pose_bytes_same(pose, aliased_error_before) &&
          g1_test_orientation_result_same(result, result_before),
          "aliased shape diagnostic preserves pose and result bytes");
}

#if defined(G1_IK_ENABLE_TEST_SEAMS)

using G1IkResetSignature = bool (*)(
    G1IkState&, const slice1d<vec3>, const slice1d<quat>,
    const slice1d<int>, char*, int);
using G1IkBeginSignature = bool (*)(
    G1IkFrameTransaction&, array1d<vec3>&, array1d<quat>&,
    const G1IkState&, const slice1d<vec3>, const slice1d<quat>,
    const slice1d<int>, const slice1d<bool>, const heightfield&,
    const G1FootprintObservation&, bool, float, char*, int);
using G1IkStageSignature = bool (*)(
    G1IkFrameTransaction&, array1d<vec3>&, array1d<quat>&, uint32_t,
    const slice1d<int>, const slice1d<bool>, const heightfield&,
    const G1FootprintObservation&, bool, float, char*, int);
using G1IkFinishSignature = bool (*)(
    G1IkState&, G1IkFrameResult&, G1IkFrameTransaction&,
    array1d<vec3>&, array1d<quat>&, const slice1d<int>,
    const heightfield&, float, char*, int);
using G1IkEvaluateSignature = bool (*)(
    array1d<vec3>&, array1d<quat>&, G1IkState&,
    const slice1d<vec3>, const slice1d<quat>, const slice1d<int>,
    const slice1d<bool>, const heightfield&, const G1FootprintObservation&,
    bool, float, G1IkFrameResult&, char*, int);
using G1IkCandidateTestSignature = bool (*)(
    G1SwingCandidateDiagnostic&,
    const slice1d<vec3>,
    const slice1d<quat>,
    const slice1d<int>,
    const G1SwingHistory&,
    const heightfield&,
    const G1LegConfig&,
    const G1FootTarget&,
    uint32_t,
    float,
    char*,
    int);

static_assert(std::is_same<decltype(&g1_ik_state_reset),
                           G1IkResetSignature>::value,
              "runtime state reset signature is fixed");
static_assert(std::is_same<decltype(&g1_ik_frame_begin),
                           G1IkBeginSignature>::value,
              "runtime begin signature is fixed");
static_assert(std::is_same<decltype(&g1_ik_frame_stage_foot),
                           G1IkStageSignature>::value,
              "runtime stage signature is fixed");
static_assert(std::is_same<decltype(&g1_ik_frame_finish),
                           G1IkFinishSignature>::value,
              "runtime finish signature is fixed");
static_assert(std::is_same<decltype(&g1_ik_frame_evaluate),
                           G1IkEvaluateSignature>::value,
              "runtime evaluate signature is fixed");
static_assert(
    std::is_same<decltype(&g1_ik_stage_swing_candidate_for_test),
                 G1IkCandidateTestSignature>::value,
    "runtime test seam is exactly one diagnostic-only stage wrapper");

template<typename T, typename = void>
struct G1RuntimeHasEnabledMember : std::false_type {};
template<typename T>
struct G1RuntimeHasEnabledMember<
    T, std::void_t<decltype(&T::enabled)>> : std::true_type {};

template<typename T, typename = void>
struct G1RuntimeHasTerminalSafeStopMember : std::false_type {};
template<typename T>
struct G1RuntimeHasTerminalSafeStopMember<
    T, std::void_t<decltype(&T::terminal_safe_stop)>> : std::true_type {};

template<typename T, typename = void>
struct G1RuntimeHasRecordedContactsMember : std::false_type {};
template<typename T>
struct G1RuntimeHasRecordedContactsMember<
    T, std::void_t<decltype(&T::recorded_contacts)>> : std::true_type {};

template<typename T, typename = void>
struct G1RuntimeHasBaseTargetsMember : std::false_type {};
template<typename T>
struct G1RuntimeHasBaseTargetsMember<
    T, std::void_t<decltype(&T::base_targets)>> : std::true_type {};

template<typename T, typename = void>
struct G1RuntimeHasAcceptedPositionsMember : std::false_type {};
template<typename T>
struct G1RuntimeHasAcceptedPositionsMember<
    T, std::void_t<decltype(&T::accepted_positions)>> : std::true_type {};

template<typename T, typename = void>
struct G1RuntimeHasAcceptedRotationsMember : std::false_type {};
template<typename T>
struct G1RuntimeHasAcceptedRotationsMember<
    T, std::void_t<decltype(&T::accepted_rotations)>> : std::true_type {};

static_assert(std::is_same<
                  decltype(&G1IkFrameTransaction::initialized),
                  bool G1IkFrameTransaction::*>::value &&
              std::is_same<
                  decltype(&G1IkFrameTransaction::next_foot),
                  uint32_t G1IkFrameTransaction::*>::value &&
              std::is_same<
                  decltype(&G1IkFrameTransaction::candidate_state),
                  G1IkState G1IkFrameTransaction::*>::value &&
              std::is_same<
                  decltype(&G1IkFrameTransaction::candidate_result),
                  G1IkFrameResult G1IkFrameTransaction::*>::value,
              "runtime transaction publishes the four fixed member types");
static_assert(!G1RuntimeHasEnabledMember<G1IkFrameTransaction>::value &&
              !G1RuntimeHasTerminalSafeStopMember<
                  G1IkFrameTransaction>::value &&
              !G1RuntimeHasRecordedContactsMember<
                  G1IkFrameTransaction>::value &&
              !G1RuntimeHasBaseTargetsMember<
                  G1IkFrameTransaction>::value &&
              !G1RuntimeHasAcceptedPositionsMember<
                  G1IkFrameTransaction>::value &&
              !G1RuntimeHasAcceptedRotationsMember<
                  G1IkFrameTransaction>::value,
              "runtime transaction exposes no duplicate public scratch owners");

static inline void g1_runtime_bind_published_transaction_shape(
    G1IkFrameTransaction& transaction)
{
    auto& [initialized, next_foot, candidate_state, candidate_result] =
        transaction;
    (void)initialized;
    (void)next_foot;
    (void)candidate_state;
    (void)candidate_result;
}

template<typename T>
struct G1RuntimeByteSnapshot
{
    unsigned char bytes[sizeof(T)];

    explicit G1RuntimeByteSnapshot(const T& value)
    {
        std::memcpy(bytes, &value, sizeof(value));
    }

    bool same(const T& value) const
    {
        return std::memcmp(bytes, &value, sizeof(value)) == 0;
    }
};

template<typename T>
static void g1_runtime_poison_bytes(T& value, unsigned char byte)
{
    unsigned char* const bytes =
        reinterpret_cast<unsigned char*>(&value);
    for (size_t index = 0; index < sizeof(value); ++index) {
        bytes[index] = byte;
    }
}

static bool g1_runtime_vec3_bits_same(vec3 left, vec3 right)
{
    return terrain_float_bits(left.x) == terrain_float_bits(right.x) &&
           terrain_float_bits(left.y) == terrain_float_bits(right.y) &&
           terrain_float_bits(left.z) == terrain_float_bits(right.z);
}

static uint64_t g1_runtime_double_bits(double value)
{
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static bool g1_runtime_array_vec3_same(
    const array1d<vec3>& left, const array1d<vec3>& right)
{
    return left.size == right.size &&
           std::memcmp(left.data, right.data,
                       static_cast<size_t>(left.size) * sizeof(vec3)) == 0;
}

static bool g1_runtime_array_quat_same(
    const array1d<quat>& left, const array1d<quat>& right)
{
    return left.size == right.size &&
           std::memcmp(left.data, right.data,
                       static_cast<size_t>(left.size) * sizeof(quat)) == 0;
}

struct G1RuntimeFixture
{
    database db;
    heightfield field;
    array1d<vec3> global_positions;
    array1d<quat> global_rotations;
    bool contact_values[2] = {false, false};
    G1FootprintObservation footprint;
    G1IkState state;
};

static vec3 g1_runtime_sphere_center(
    const G1RuntimeFixture& fixture,
    const G1LegConfig& config,
    int probe)
{
    return fixture.global_positions(config.ankle) +
           quat_mul_vec3(
               fixture.global_rotations(config.ankle),
               config.foot_sphere_centers_local[probe]);
}

static vec3 g1_runtime_sole_point(
    const G1RuntimeFixture& fixture,
    const G1LegConfig& config,
    int probe)
{
    return fixture.global_positions(config.ankle) +
           quat_mul_vec3(
               fixture.global_rotations(config.ankle),
               config.sole_points_local[probe]);
}

#if defined(__GNUC__) || defined(__clang__)
#define G1_IK_TEST_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define G1_IK_TEST_NOINLINE __declspec(noinline)
#else
#define G1_IK_TEST_NOINLINE
#endif

static G1_IK_TEST_NOINLINE bool
g1_runtime_independent_foot_centers(
    vec3 output[4],
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    const G1LegConfig& config)
{
    for (int probe = 0; probe < 4; ++probe) {
        const vec3 offset = quat_mul_vec3(
            global_rotations(config.ankle),
            config.foot_sphere_centers_local[probe]);
        const vec3 center =
            global_positions(config.ankle) + offset;
        if (!g1_ik_vec3_is_runtime_value(offset) ||
            !g1_ik_vec3_is_runtime_value(center)) {
            return false;
        }
        output[probe] = center;
    }
    return true;
}

#undef G1_IK_TEST_NOINLINE

static void g1_runtime_refresh_current_probes(G1RuntimeFixture& fixture)
{
    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    fixture.footprint = G1FootprintObservation{};
    G1SurfaceQueryStatus root_status = g1_surface_query_v2(
        fixture.footprint.root_surface,
        fixture.field,
        fixture.global_positions(G1_Simulation).x,
        fixture.global_positions(G1_Simulation).z);
    check(root_status == G1SurfaceQueryValid,
          "runtime fixture root is on the checked field");
    fixture.footprint.blocked = false;
    fixture.footprint.blocked_reason = walkability_clear;
    fixture.footprint.work.sweeps = 24;
    fixture.footprint.work.surface_queries = 33;
    fixture.footprint.work.node_visits = 256;
    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        G1FootprintFootObservation& foot =
            fixture.footprint.feet[foot_index];
        foot.current_contact = fixture.contact_values[foot_index];
        foot.landing_sample = UINT32_MAX;
        foot.encountered_walkability_class = 1;
        foot.predicted_landing_walkability_class = 0;
        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            G1FootprintProbe& probe = foot.probes[probe_index];
            probe.current_sphere_center = g1_runtime_sphere_center(
                fixture, configs[foot_index], probe_index);
            probe.current_sole_point = g1_runtime_sole_point(
                fixture, configs[foot_index], probe_index);
            G1SurfaceQueryStatus status = g1_surface_query_v2(
                probe.current_surface,
                fixture.field,
                probe.current_sole_point.x,
                probe.current_sole_point.z);
            check(status == G1SurfaceQueryValid,
                  "runtime fixture sole probe is on the checked field");
            probe.selected_landing_surface = probe.current_surface;
            probe.corridor_minimum_height = probe.current_surface.height;
            probe.corridor_maximum_height = probe.current_surface.height;
            probe.encountered_walkability_class = 1;
            for (int sample = 0;
                 sample < G1CommandTrajectorySampleCount;
                 ++sample) {
                probe.predicted_sphere_centers[sample] =
                    probe.current_sphere_center;
                probe.predicted_sole_points[sample] =
                    probe.current_sole_point;
                probe.predicted_surface_status[sample] =
                    G1SurfaceQueryValid;
                probe.predicted_surfaces[sample] =
                    probe.current_surface;
            }
        }
    }
}

static void g1_runtime_make_fixture(G1RuntimeFixture& fixture)
{
    make_g1_database(fixture.db);
    fixture.db.bone_positions(0, G1_Simulation) =
        vec3(2.0f, 1.0f, 2.0f);
    fixture.field.version = 2;
    fixture.field.nx = 65;
    fixture.field.nz = 65;
    fixture.field.origin_x = -4.0f;
    fixture.field.origin_z = -4.0f;
    fixture.field.cell_size = 0.25f;
    fixture.field.exterior_height = -10.0f;
    fixture.field.heights.resize(65 * 65);
    fixture.field.heights.set(0.0f);
    fixture.global_positions.resize(G1_BoneCount);
    fixture.global_rotations.resize(G1_BoneCount);
    char error[256] = {};
    check(g1_ik_checked_forward_kinematics(
              fixture.global_positions,
              fixture.global_rotations,
              fixture.db.bone_positions(0),
              fixture.db.bone_rotations(0),
              fixture.db.bone_parents,
              error, static_cast<int>(sizeof(error))),
          error);
    g1_runtime_refresh_current_probes(fixture);
    check(g1_ik_state_reset(
              fixture.state,
              fixture.db.bone_positions(0),
              fixture.db.bone_rotations(0),
              fixture.db.bone_parents,
              error, static_cast<int>(sizeof(error))),
          error);
}

static void g1_runtime_set_landing(
    G1RuntimeFixture& fixture,
    int foot_index,
    bool ready,
    vec3 centroid,
    float height,
    vec3 normal)
{
    G1FootprintFootObservation& foot =
        fixture.footprint.feet[foot_index];
    foot.landing_expected = true;
    foot.landing_patch_ready = ready;
    foot.landing_sample = 2;
    foot.predicted_landing_sole_center = centroid;
    foot.predicted_landing_surface_status = G1SurfaceQueryValid;
    foot.predicted_landing_surface.height = height;
    foot.predicted_landing_surface.normal = normal;
    foot.predicted_landing_walkability_class = 1;
    foot.landing_patch_maximum_residual_m = 0.0;
}

static void g1_runtime_begin(
    G1RuntimeFixture& fixture,
    const G1FootprintObservation& footprint,
    G1IkFrameTransaction& transaction,
    array1d<vec3>& positions,
    array1d<quat>& rotations)
{
    positions.resize(G1_BoneCount);
    rotations.resize(G1_BoneCount);
    positions.set(vec3(7.0f, 8.0f, 9.0f));
    rotations.set(quat(0.5f, 0.5f, 0.5f, 0.5f));
    char error[256] = {};
    check(g1_ik_frame_begin(
              transaction, positions, rotations,
              fixture.state,
              fixture.db.bone_positions(0),
              fixture.db.bone_rotations(0),
              fixture.db.bone_parents,
              slice1d<bool>(2, fixture.contact_values),
              fixture.field, footprint, true, 0.04f,
              error, static_cast<int>(sizeof(error))),
          error);
}

static void test_runtime_lift_ladder_and_observer_crosspath()
{
    for (uint32_t candidate = 0;
         candidate < G1SwingLiftCandidateCount;
         ++candidate) {
        const uint32_t expected = candidate == 0
            ? 0U
            : g1_test_exact_rational_binary32(candidate, 500U);
        check(G1SwingLiftCandidateBits[candidate] == expected,
              "all 41 lift words equal the independent exact-rational oracle");
        if (candidate != 0) {
            check(G1SwingLiftCandidateBits[candidate - 1] < expected,
                  "all adjacent lift words are strictly ordered");
        }
    }

    G1RuntimeFixture fixture;
    g1_runtime_make_fixture(fixture);
    fixture.db.bone_rotations(0, G1_Simulation) =
        quat_from_angle_axis(0.37f, vec3(0.0f, 1.0f, 0.0f));
    char error[256] = {};
    check(g1_ik_checked_forward_kinematics(
              fixture.global_positions,
              fixture.global_rotations,
              fixture.db.bone_positions(0),
              fixture.db.bone_rotations(0),
              fixture.db.bone_parents,
              error, static_cast<int>(sizeof(error))) &&
              g1_ik_state_reset(
                  fixture.state,
                  fixture.db.bone_positions(0),
                  fixture.db.bone_rotations(0),
                  fixture.db.bone_parents,
                  error, static_cast<int>(sizeof(error))),
          error);
    check(!g1_test_quat_bits_same(
              fixture.global_rotations(G1_Simulation),
              quat(1.0f, 0.0f, 0.0f, 0.0f)),
          "Task4/Task5 cross-path pose has nontrivial global rotation");

    walkability_grid grid;
    grid.nx = fixture.field.nx;
    grid.nz = fixture.field.nz;
    grid.cells.resize(grid.nx * grid.nz);
    grid.cells.set(1);
    G1CommandSnapshot command = {};
    command.intent.desired_heading =
        fixture.global_rotations(G1_Simulation);
    for (int sample = 0;
         sample < G1CommandTrajectorySampleCount;
         ++sample) {
        command.predicted_root_positions[sample] =
            fixture.global_positions(G1_Simulation);
        command.predicted_root_rotations[sample] =
            fixture.global_rotations(G1_Simulation);
        command.predicted_desired_headings[sample] =
            fixture.global_rotations(G1_Simulation);
    }
    const G1RuntimeByteSnapshot<G1CommandSnapshot>
        command_before(command);
    G1FootContactSchedule schedule = {};
    G1FootprintObservation observed = {};
    check(g1_footprint_observe_v2(
              observed,
              g1_footprint_budget(),
              fixture.field,
              grid,
              command,
              schedule,
              fixture.global_positions,
              fixture.global_rotations,
              error,
              static_cast<int>(sizeof(error))) == G1FootprintOk,
          error);
    check(command_before.same(command),
          "real Task4 observation preserves the command snapshot");

    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    for (int foot = 0; foot < 2; ++foot) {
        vec3 independently_materialized[4] = {};
        check(g1_runtime_independent_foot_centers(
                  independently_materialized,
                  fixture.global_positions,
                  fixture.global_rotations,
                  configs[foot]),
              "cross-path centers independently materialize");
        for (int probe = 0; probe < 4; ++probe) {
            const vec3 task4 = observed.feet[foot]
                .probes[probe].current_sphere_center;
            check(g1_runtime_vec3_bits_same(
                      task4,
                      independently_materialized[probe]) &&
                  g1_runtime_vec3_bits_same(
                      task4,
                      fixture.state.feet[foot].swing
                          .previous_sphere_centers[probe]),
                  "all eight Task4 current centers cross into Task5 exactly");
        }
    }

    G1IkFrameTransaction transaction = {};
    array1d<vec3> scratch_positions(G1_BoneCount);
    array1d<quat> scratch_rotations(G1_BoneCount);
    check(g1_ik_frame_begin(
              transaction,
              scratch_positions,
              scratch_rotations,
              fixture.state,
              fixture.db.bone_positions(0),
              fixture.db.bone_rotations(0),
              fixture.db.bone_parents,
              slice1d<bool>(2, fixture.contact_values),
              fixture.field,
              observed,
              true,
              0.04f,
              error,
              static_cast<int>(sizeof(error))) &&
              transaction.initialized &&
              command_before.same(command),
          error);
}

static void test_runtime_state_reset_and_footprint_preflight()
{
    G1IkFrameTransaction published_shape = {};
    g1_runtime_bind_published_transaction_shape(published_shape);

    G1RuntimeFixture fixture;
    g1_runtime_make_fixture(fixture);
    check(fixture.state.initialized &&
              fixture.state.feet[0].lock.initialized &&
              fixture.state.feet[1].lock.initialized &&
              fixture.state.feet[0].swing.initialized &&
              fixture.state.feet[1].swing.initialized,
          "runtime reset initializes both lock and swing histories");
    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        for (int probe = 0; probe < 4; ++probe) {
            check(g1_runtime_vec3_bits_same(
                      fixture.state.feet[foot_index]
                          .swing.previous_sphere_centers[probe],
                      fixture.footprint.feet[foot_index]
                          .probes[probe].current_sphere_center),
                  "runtime reset history owns exact FK sphere bits");
        }
    }

    G1IkState poisoned = fixture.state;
    poisoned.feet[1].swing.previous_sphere_centers[3].x =
        g1_test_float_from_bits(UINT32_C(0x7fc00001));
    const G1RuntimeByteSnapshot<G1IkState> poisoned_before(poisoned);
    array1d<quat> invalid_rotations = fixture.db.bone_rotations(0);
    invalid_rotations(G1_RightWrist) = quat(2.0f, 0.0f, 0.0f, 0.0f);
    check(!g1_ik_state_reset(
              poisoned,
              fixture.db.bone_positions(0), invalid_rotations,
              fixture.db.bone_parents, NULL, 0) &&
              poisoned_before.same(poisoned),
          "runtime reset validates the complete pose before assignment");

    G1FootprintObservation mismatch = fixture.footprint;
    mismatch.feet[1].probes[2].current_sphere_center.y =
        std::nextafter(
            mismatch.feet[1].probes[2].current_sphere_center.y,
            std::numeric_limits<float>::infinity());
    G1IkFrameTransaction transaction;
    g1_runtime_poison_bytes(transaction, 0xa5);
    const G1RuntimeByteSnapshot<G1IkFrameTransaction>
        transaction_before(transaction);
    array1d<vec3> scratch_positions(G1_BoneCount);
    array1d<quat> scratch_rotations(G1_BoneCount);
    scratch_positions.set(vec3(3.0f, 4.0f, 5.0f));
    scratch_rotations.set(quat(0.5f, 0.5f, 0.5f, 0.5f));
    const array1d<vec3> positions_before = scratch_positions;
    const array1d<quat> rotations_before = scratch_rotations;
    char error[256] = {};
    check(!g1_ik_frame_begin(
              transaction, scratch_positions, scratch_rotations,
              fixture.state,
              fixture.db.bone_positions(0),
              fixture.db.bone_rotations(0),
              fixture.db.bone_parents,
              slice1d<bool>(2, fixture.contact_values),
              fixture.field, mismatch, true, 0.04f,
              error, static_cast<int>(sizeof(error))),
          "one-ULP current footprint/FK mismatch is rejected");
    check(transaction_before.same(transaction) &&
              g1_runtime_array_vec3_same(
                  scratch_positions, positions_before) &&
              g1_runtime_array_quat_same(
                  scratch_rotations, rotations_before),
          "footprint/FK mismatch rolls back transaction and pose scratch");

}

static vec3 g1_runtime_current_sole_centroid(
    const G1RuntimeFixture& fixture,
    const G1LegConfig& config)
{
    volatile double sum_x = 0.0;
    volatile double sum_y = 0.0;
    volatile double sum_z = 0.0;
    for (int probe = 0; probe < 4; ++probe) {
        const vec3 point =
            g1_runtime_sole_point(fixture, config, probe);
        sum_x = sum_x + static_cast<double>(point.x);
        sum_y = sum_y + static_cast<double>(point.y);
        sum_z = sum_z + static_cast<double>(point.z);
    }
    vec3 output;
    check(terrain_v2_round_output(sum_x / 4.0, output.x) &&
              terrain_v2_round_output(sum_y / 4.0, output.y) &&
              terrain_v2_round_output(sum_z / 4.0, output.z),
          "current sole centroid rounds through the checked v2 path");
    return output;
}

static float g1_runtime_configure_real_step_landing(
    G1RuntimeFixture& fixture,
    int direction)
{
    check(direction == -1 || direction == 1,
          "step landing direction is exactly down or up");
    const G1LegConfig config = g1_left_leg_config();
    const vec3 current_centroid =
        g1_runtime_current_sole_centroid(fixture, config);
    float landing_height = 0.0f;
    const bool height_ok = direction < 0
        ? terrain_f32_sub(
              landing_height, current_centroid.y, 0.32f)
        : terrain_f32_add(
              landing_height, current_centroid.y, 0.32f);
    check(height_ok,
          "real step surface is one checked binary32 0.32 m offset");

    fixture.field.nx = 161;
    fixture.field.nz = 161;
    fixture.field.origin_x = -4.0f;
    fixture.field.origin_z = -4.0f;
    fixture.field.cell_size = 0.05f;
    fixture.field.heights.resize(161 * 161);
    for (int z = 0; z < fixture.field.nz; ++z) {
        for (int x = 0; x < fixture.field.nx; ++x) {
            const float node_x = fixture.field.origin_x +
                static_cast<float>(x) * fixture.field.cell_size;
            fixture.field.heights(z * fixture.field.nx + x) =
                node_x <= 2.10f
                    ? current_centroid.y
                    : landing_height;
        }
    }
    g1_runtime_refresh_current_probes(fixture);

    G1FootprintFootObservation& foot = fixture.footprint.feet[0];
    const uint32_t landing_sample = 2;
    for (int probe = 0; probe < 4; ++probe) {
        G1FootprintProbe& footprint_probe = foot.probes[probe];
        footprint_probe.predicted_sphere_centers[landing_sample].x +=
            0.30f;
        footprint_probe.predicted_sole_points[landing_sample].x +=
            0.30f;
        G1SurfaceSample sample = {};
        const G1SurfaceQueryStatus status = g1_surface_query_v2(
            sample, fixture.field,
            footprint_probe.predicted_sole_points[landing_sample].x,
            footprint_probe.predicted_sole_points[landing_sample].z);
        check(status == G1SurfaceQueryValid &&
                  terrain_float_bits(sample.height) ==
                      terrain_float_bits(landing_height),
              "shifted landing probe samples the real opposite plateau");
        footprint_probe.predicted_surface_status[landing_sample] = status;
        footprint_probe.predicted_surfaces[landing_sample] = sample;
        footprint_probe.selected_landing_surface = sample;
    }
    const vec3 predicted_centroid(
        current_centroid.x + 0.30f,
        current_centroid.y,
        current_centroid.z);
    G1SurfaceSample centroid_surface = {};
    check(g1_surface_query_v2(
              centroid_surface, fixture.field,
              predicted_centroid.x,
              predicted_centroid.z) == G1SurfaceQueryValid &&
              terrain_float_bits(centroid_surface.height) ==
                  terrain_float_bits(landing_height),
          "landing centroid samples the same real plateau");
    g1_runtime_set_landing(
        fixture, 0, true,
        predicted_centroid,
        centroid_surface.height,
        centroid_surface.normal);
    return landing_height;
}

static bool g1_runtime_stage_candidate_direct(
    const G1RuntimeFixture& fixture,
    uint32_t foot_index,
    uint32_t candidate_index,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const G1SwingHistory& history,
    const G1FootTarget& target,
    G1SwingCandidateDiagnostic& diagnostic,
    char* error,
    int error_capacity);

static void test_runtime_blocked_and_landing_contract()
{
    G1RuntimeFixture fixture;
    g1_runtime_make_fixture(fixture);
    array1d<vec3> output_positions = fixture.db.bone_positions(0);
    array1d<quat> output_rotations = fixture.db.bone_rotations(0);
    const array1d<vec3> positions_before = output_positions;
    const array1d<quat> rotations_before = output_rotations;
    const G1RuntimeByteSnapshot<G1IkState> state_before(fixture.state);
    G1IkFrameResult result;
    g1_runtime_poison_bytes(result, 0x6b);

    G1FootprintObservation blocked = fixture.footprint;
    blocked.blocked = true;
    blocked.blocked_reason = walkability_blocked_cell;
    char error[256] = {};
    check(g1_ik_frame_evaluate(
              output_positions, output_rotations, fixture.state,
              fixture.db.bone_positions(0),
              fixture.db.bone_rotations(0),
              fixture.db.bone_parents,
              slice1d<bool>(2, fixture.contact_values),
              fixture.field, blocked, true, 0.04f,
              result, error, static_cast<int>(sizeof(error))),
          error);
    check(result.safe_stop_requested && !result.applied &&
              result.stop_reason == G1IkStopFootprintBlocked &&
              state_before.same(fixture.state) &&
              g1_runtime_array_vec3_same(
                  output_positions, positions_before) &&
              g1_runtime_array_quat_same(
                  output_rotations, rotations_before),
          "blocked footprint safe-stops without pose/state/history publication");

    G1FootprintObservation unready = fixture.footprint;
    const vec3 ordinary =
        fixture.global_positions(g1_left_leg_config().contact);
    g1_runtime_set_landing(
        fixture, 0, false,
        vec3(ordinary.x + 0.10f, ordinary.y, ordinary.z),
        0.0f, vec3(0.0f, 1.0f, 0.0f));
    unready = fixture.footprint;
    result = G1IkFrameResult{};
    check(g1_ik_frame_evaluate(
              output_positions, output_rotations, fixture.state,
              fixture.db.bone_positions(0),
              fixture.db.bone_rotations(0),
              fixture.db.bone_parents,
              slice1d<bool>(2, fixture.contact_values),
              fixture.field, unready, true, 0.04f,
              result, error, static_cast<int>(sizeof(error))),
          error);
    check(result.safe_stop_requested &&
              result.stop_reason == G1IkStopLandingPatchUnavailable &&
              state_before.same(fixture.state) &&
              g1_runtime_array_vec3_same(
                  output_positions, positions_before) &&
              g1_runtime_array_quat_same(
                  output_rotations, rotations_before),
          "unready expected landing stops with all accepted owners unchanged");

    for (int direction = -1; direction <= 1; direction += 2) {
        G1RuntimeFixture staged_landing;
        g1_runtime_make_fixture(staged_landing);
        const vec3 current_centroid = g1_runtime_current_sole_centroid(
            staged_landing, g1_left_leg_config());
        const float landing_height =
            g1_runtime_configure_real_step_landing(
                staged_landing, direction);
        G1FootprintFootObservation& staged_foot =
            staged_landing.footprint.feet[0];
        const float corridor_minimum = direction < 0
            ? landing_height
            : current_centroid.y;
        const float corridor_maximum = direction < 0
            ? current_centroid.y
            : landing_height;
        staged_foot.corridor_minimum_height = corridor_minimum;
        staged_foot.corridor_maximum_height = corridor_maximum;
        staged_foot.maximum_root_split_m =
            std::fabs(
                static_cast<double>(
                    staged_landing.footprint.root_surface.height) -
                static_cast<double>(landing_height));
        staged_foot.multilevel = true;
        for (int probe = 0; probe < 4; ++probe) {
            staged_foot.probes[probe].corridor_minimum_height =
                corridor_minimum;
            staged_foot.probes[probe].corridor_maximum_height =
                corridor_maximum;
            staged_foot.probes[probe].encountered_walkability_class = 1;
        }
        const uint32_t expected_surface_bits = direction < 0
            ? UINT32_C(0x3e1c5048)
            : UINT32_C(0x3f4aeb1c);
        float independently_offset = 0.0f;
        check((direction < 0
                   ? terrain_f32_sub(
                         independently_offset,
                         current_centroid.y, 0.32f)
                   : terrain_f32_add(
                         independently_offset,
                         current_centroid.y, 0.32f)) &&
                  terrain_float_bits(independently_offset) ==
                      terrain_float_bits(landing_height) &&
                  terrain_float_bits(current_centroid.y) ==
                      UINT32_C(0x3ef1ff2e) &&
                  terrain_float_bits(landing_height) ==
                      expected_surface_bits &&
                  staged_foot.encountered_walkability_class == 1 &&
                  staged_foot.predicted_landing_walkability_class == 1 &&
                  staged_foot.multilevel &&
                  staged_foot.maximum_root_split_m ==
                      static_cast<double>(0.32f) &&
                  terrain_float_bits(staged_foot
                          .corridor_minimum_height) ==
                      terrain_float_bits(corridor_minimum) &&
                  terrain_float_bits(staged_foot
                          .corridor_maximum_height) ==
                      terrain_float_bits(corridor_maximum),
              "class-one multilevel surface/corridor is a real checked 0.32 offset");
        G1IkFrameTransaction transaction = {};
        array1d<vec3> scratch_positions;
        array1d<quat> scratch_rotations;
        g1_runtime_begin(
            staged_landing, staged_landing.footprint, transaction,
            scratch_positions, scratch_rotations);
        float expected_y = 91.0f;
        check(g1_apply_swing_lift_y(
                  expected_y, landing_height,
                  g1_left_leg_config().swing_clearance_m,
                  error, static_cast<int>(sizeof(error))) ==
                  G1ClearanceOk,
              error);
        const vec3 predicted = staged_landing.footprint.feet[0]
            .predicted_landing_sole_center;
        const vec3 landing_normal = staged_landing.footprint.feet[0]
            .predicted_landing_surface.normal;
        const G1FootTarget& target =
            transaction.candidate_result.feet[0].target;
        check(terrain_float_bits(target.surface.point.x) ==
                  terrain_float_bits(predicted.x) &&
              terrain_float_bits(target.surface.point.z) ==
                  terrain_float_bits(predicted.z) &&
              terrain_float_bits(target.surface.point.y) ==
                  terrain_float_bits(landing_height) &&
              g1_runtime_vec3_bits_same(
                  target.surface.normal, landing_normal) &&
              terrain_float_bits(target.sole_center.x) ==
                  terrain_float_bits(predicted.x) &&
              terrain_float_bits(target.sole_center.z) ==
                  terrain_float_bits(predicted.z) &&
              terrain_float_bits(target.sole_center.y) ==
                  terrain_float_bits(expected_y),
              "ready up/down landing replaces XZ, height, normal, and base Y");

        for (uint32_t candidate = 0;
             candidate < G1SwingLiftCandidateCount;
             ++candidate) {
            G1SwingCandidateDiagnostic diagnostic = {};
            check(g1_runtime_stage_candidate_direct(
                      staged_landing, 0, candidate,
                      scratch_positions, scratch_rotations,
                      transaction.candidate_state.feet[0].swing,
                      transaction.candidate_result.feet[0].target,
                      diagnostic,
                      error, static_cast<int>(sizeof(error))),
                  error);
            float lift = 0.0f;
            std::memcpy(
                &lift, &G1SwingLiftCandidateBits[candidate],
                sizeof(lift));
            float materialized_y = 0.0f;
            check(g1_apply_swing_lift_y(
                      materialized_y, expected_y, lift,
                      error, static_cast<int>(sizeof(error))) ==
                      G1ClearanceOk &&
                      diagnostic.materialized_command_y_bits ==
                          terrain_float_bits(materialized_y),
                  "every real ladder stage adds once to the landing-aware base");
        }
        check(g1_ik_frame_stage_foot(
                  transaction, scratch_positions, scratch_rotations, 0,
                  staged_landing.db.bone_parents,
                  slice1d<bool>(2, staged_landing.contact_values),
                  staged_landing.field, staged_landing.footprint,
                  true, 0.04f,
                  error, static_cast<int>(sizeof(error))) &&
                  transaction.next_foot == 1 &&
                  transaction.candidate_result.feet[0]
                      .swing_selection.candidates_evaluated > 0,
              "real class-one multilevel up/down plateau reaches foot staging");
    }

    g1_runtime_refresh_current_probes(fixture);
    G1IkFrameTransaction ordinary_transaction = {};
    array1d<vec3> ordinary_positions;
    array1d<quat> ordinary_rotations;
    g1_runtime_begin(
        fixture, fixture.footprint, ordinary_transaction,
        ordinary_positions, ordinary_rotations);
    check(!fixture.footprint.feet[0].landing_expected &&
              g1_runtime_vec3_bits_same(
                  ordinary_transaction.candidate_result.feet[0]
                      .target.sole_center,
                  ordinary),
          "no expected landing retains the checked ordinary swing base");
}

static G1ClearanceWork g1_runtime_work_add(
    G1ClearanceWork left, const G1ClearanceWork& right)
{
    left.point_queries += right.point_queries;
    left.cells_visited += right.cells_visited;
    left.primitive_triangle_pairs += right.primitive_triangle_pairs;
    left.face_patches += right.face_patches;
    left.candidate_tests += right.candidate_tests;
    left.subdivision_nodes += right.subdivision_nodes;
    return left;
}

static bool g1_runtime_work_same(
    const G1ClearanceWork& left,
    const G1ClearanceWork& right)
{
    return left.point_queries == right.point_queries &&
           left.cells_visited == right.cells_visited &&
           left.primitive_triangle_pairs ==
               right.primitive_triangle_pairs &&
           left.face_patches == right.face_patches &&
           left.candidate_tests == right.candidate_tests &&
           left.subdivision_nodes == right.subdivision_nodes;
}

static bool g1_runtime_candidate_same(
    const G1SwingCandidateDiagnostic& left,
    const G1SwingCandidateDiagnostic& right)
{
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

static bool g1_runtime_stage_candidate_direct(
    const G1RuntimeFixture& fixture,
    uint32_t foot_index,
    uint32_t candidate_index,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const G1SwingHistory& history,
    const G1FootTarget& target,
    G1SwingCandidateDiagnostic& diagnostic,
    char* error,
    int error_capacity)
{
    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    const bool ok = g1_ik_stage_swing_candidate_for_test(
        diagnostic,
        local_positions,
        baseline_rotations,
        fixture.db.bone_parents,
        history,
        fixture.field,
        configs[foot_index],
        target,
        candidate_index,
        0.04f,
        error,
        error_capacity);
    return ok;
}

static bool g1_runtime_candidate_passes(
    const G1SwingCandidateDiagnostic& diagnostic)
{
    return diagnostic.clearance_status == G1ClearanceOk &&
           diagnostic.lower_margin_m >= 0.0 &&
           diagnostic.controller_constraints_passed &&
           diagnostic.clearance_certified;
}

static G1FootTarget g1_runtime_current_target(
    const G1RuntimeFixture& fixture,
    int foot_index)
{
    const G1LegConfig config = foot_index == 0
        ? g1_left_leg_config()
        : g1_right_leg_config();
    G1FootTarget target = {};
    target.sole_center =
        fixture.global_positions(config.contact);
    target.surface.point = target.sole_center;
    target.surface.normal = vec3(0.0f, 1.0f, 0.0f);
    return target;
}

static bool g1_runtime_evaluate(
    G1RuntimeFixture& fixture,
    array1d<vec3>& positions,
    array1d<quat>& rotations,
    G1IkFrameResult& result,
    char* error,
    int error_capacity)
{
    return g1_ik_frame_evaluate(
        positions, rotations, fixture.state,
        fixture.db.bone_positions(0),
        fixture.db.bone_rotations(0),
        fixture.db.bone_parents,
        slice1d<bool>(2, fixture.contact_values),
        fixture.field, fixture.footprint,
        true, 0.04f, result, error, error_capacity);
}

static void test_runtime_real_41_stage_selector()
{
    G1RuntimeFixture fixture;
    g1_runtime_make_fixture(fixture);

    G1IkFrameTransaction transaction = {};
    array1d<vec3> scratch_positions;
    array1d<quat> scratch_rotations;
    g1_runtime_begin(
        fixture, fixture.footprint, transaction,
        scratch_positions, scratch_rotations);

    G1SwingCandidateDiagnostic independent[G1SwingLiftCandidateCount] = {};
    uint32_t first_passing = G1SwingNoCandidate;
    for (uint32_t candidate = 0;
         candidate < G1SwingLiftCandidateCount;
         ++candidate) {
        char error[256] = {};
        check(g1_runtime_stage_candidate_direct(
                  fixture, 0, candidate,
                  scratch_positions, scratch_rotations,
                  transaction.candidate_state.feet[0].swing,
                  transaction.candidate_result.feet[0].target,
                  independent[candidate],
                  error, static_cast<int>(sizeof(error))),
              error);
        check(independent[candidate].candidate_index == candidate &&
                  independent[candidate].lift_bits ==
                      G1SwingLiftCandidateBits[candidate],
              "diagnostic seam authenticates index and literal lift bits");
        if (g1_runtime_candidate_passes(independent[candidate]) &&
            first_passing == G1SwingNoCandidate) {
            first_passing = candidate;
        }
    }
    check(first_passing != G1SwingNoCandidate,
          "independent probing discovers at least one real admissible stage");
    G1ClearanceWork expected_selected_work = {};
    for (uint32_t candidate = 0; candidate <= first_passing; ++candidate) {
        if (independent[candidate].clearance_status == G1ClearanceOk) {
            expected_selected_work = g1_runtime_work_add(
                expected_selected_work,
                independent[candidate].clearance_work);
        }
    }

    array1d<vec3> positions = fixture.db.bone_positions(0);
    array1d<quat> rotations = fixture.db.bone_rotations(0);
    G1IkFrameResult result = {};
    char error[256] = {};
    check(g1_runtime_evaluate(
              fixture, positions, rotations, result,
              error, static_cast<int>(sizeof(error))),
          error);
    const G1SwingSelectionDiagnostic& selection =
        result.feet[0].swing_selection;
    check(result.applied && !result.safe_stop_requested &&
              selection.selected_index == first_passing &&
              selection.candidates_evaluated == first_passing + 1 &&
              g1_runtime_candidate_same(
                  selection.selected, independent[first_passing]) &&
              g1_runtime_work_same(
                  selection.total_clearance_work,
                  expected_selected_work),
          "full selector chooses the independently authenticated first stage");

    array1d<vec3> final_globals(G1_BoneCount);
    array1d<quat> final_global_rotations(G1_BoneCount);
    check(g1_ik_checked_forward_kinematics(
              final_globals, final_global_rotations,
              positions, rotations, fixture.db.bone_parents,
              error, static_cast<int>(sizeof(error))),
          error);
    vec3 independently_materialized[4] = {};
    check(g1_runtime_independent_foot_centers(
              independently_materialized,
              final_globals, final_global_rotations,
              g1_left_leg_config()),
          "selected public pose has independently materializable centers");
    for (int probe = 0; probe < 4; ++probe) {
        const uint32_t actual[3] = {
            terrain_float_bits(independently_materialized[probe].x),
            terrain_float_bits(independently_materialized[probe].y),
            terrain_float_bits(independently_materialized[probe].z)
        };
        for (int axis = 0; axis < 3; ++axis) {
            check(actual[axis] == selection.selected
                      .actual_sphere_center_bits[probe][axis],
                  "selected public pose owns all twelve recorded FK words");
        }
    }
}

static bool g1_runtime_work_is_zero(const G1ClearanceWork& work)
{
    return g1_runtime_work_same(work, G1ClearanceWork{});
}

static bool g1_runtime_candidate_is_default(
    const G1SwingCandidateDiagnostic& diagnostic)
{
    if (diagnostic.candidate_index != G1SwingNoCandidate ||
        diagnostic.lift_bits != 0 ||
        diagnostic.materialized_command_y_bits != 0 ||
        diagnostic.clearance_status != G1ClearanceInvalidInput ||
        diagnostic.controller_constraints_passed ||
        diagnostic.clearance_certified ||
        diagnostic.lower_margin_m != 0.0 ||
        diagnostic.witness_upper_margin_m != 0.0 ||
        !g1_runtime_work_is_zero(diagnostic.clearance_work)) {
        return false;
    }
    for (int probe = 0; probe < 4; ++probe) {
        for (int axis = 0; axis < 3; ++axis) {
            if (diagnostic.actual_sphere_center_bits[probe][axis] != 0) {
                return false;
            }
        }
    }
    return true;
}

static void g1_runtime_configure_finite_status(
    G1RuntimeFixture& fixture,
    G1ClearanceStatus expected)
{
    if (expected == G1ClearanceOutsideDomain) {
        fixture.field.nx = 5;
        fixture.field.nz = 5;
        fixture.field.origin_x = 1.86f;
        fixture.field.origin_z = 1.95f;
        fixture.field.cell_size = 0.05f;
        fixture.field.heights.resize(25);
        fixture.field.heights.set(0.0f);
    } else if (expected == G1ClearanceBudgetExceeded) {
        fixture.field.nx = 129;
        fixture.field.nz = 129;
        fixture.field.origin_x = 1.70f;
        fixture.field.origin_z = 1.70f;
        fixture.field.cell_size = 0.005f;
        fixture.field.heights.resize(129 * 129);
        fixture.field.heights.set(0.0f);
    } else {
        check(expected == G1ClearanceUncertified,
              "finite fixture status is one of the three real classes");
        fixture.db.bone_positions(0, G1_Simulation).y += 19.5f;
        char error[256] = {};
        check(g1_ik_checked_forward_kinematics(
                  fixture.global_positions,
                  fixture.global_rotations,
                  fixture.db.bone_positions(0),
                  fixture.db.bone_rotations(0),
                  fixture.db.bone_parents,
                  error, static_cast<int>(sizeof(error))) &&
              g1_ik_state_reset(
                  fixture.state,
                  fixture.db.bone_positions(0),
                  fixture.db.bone_rotations(0),
                  fixture.db.bone_parents,
                  error, static_cast<int>(sizeof(error))),
              error);
        fixture.field.heights.set(16.0f);
    }
    fixture.contact_values[1] = true;
    g1_runtime_refresh_current_probes(fixture);
}

static void g1_runtime_require_unchanged_failure(
    G1RuntimeFixture& fixture,
    const char* message)
{
    array1d<vec3> positions = fixture.db.bone_positions(0);
    array1d<quat> rotations = fixture.db.bone_rotations(0);
    const array1d<vec3> positions_before = positions;
    const array1d<quat> rotations_before = rotations;
    const G1RuntimeByteSnapshot<G1IkState> state_before(fixture.state);
    G1IkFrameResult result;
    g1_runtime_poison_bytes(result, 0x9d);
    const G1RuntimeByteSnapshot<G1IkFrameResult> result_before(result);
    check(!g1_runtime_evaluate(
              fixture, positions, rotations, result, NULL, 0),
          message);
    check(state_before.same(fixture.state) &&
              result_before.same(result) &&
              g1_runtime_array_vec3_same(
                  positions, positions_before) &&
              g1_runtime_array_quat_same(
                  rotations, rotations_before),
          "fatal runtime rejection preserves every accepted owner");
}

static void test_runtime_status_table_and_local_rejections()
{
    const G1ClearanceStatus finite[] = {
        G1ClearanceOutsideDomain,
        G1ClearanceBudgetExceeded,
        G1ClearanceUncertified
    };
    for (const G1ClearanceStatus expected : finite) {
        G1RuntimeFixture fixture;
        g1_runtime_make_fixture(fixture);
        g1_runtime_configure_finite_status(fixture, expected);
        const G1FootTarget target =
            g1_runtime_current_target(fixture, 0);
        for (uint32_t candidate = 0; candidate < 2; ++candidate) {
            G1SwingCandidateDiagnostic diagnostic = {};
            char error[256] = {};
            check(g1_runtime_stage_candidate_direct(
                      fixture, 0, candidate,
                      fixture.db.bone_positions(0),
                      fixture.db.bone_rotations(0),
                      fixture.state.feet[0].swing,
                      target, diagnostic,
                      error, static_cast<int>(sizeof(error))),
                  error);
            check(diagnostic.clearance_status == expected &&
                      diagnostic.lower_margin_m == 0.0 &&
                      diagnostic.witness_upper_margin_m == 0.0 &&
                      g1_runtime_work_is_zero(
                          diagnostic.clearance_work) &&
                      !diagnostic.clearance_certified,
                  "real finite status publishes truthful default work/margins");
        }

        array1d<vec3> positions = fixture.db.bone_positions(0);
        array1d<quat> rotations = fixture.db.bone_rotations(0);
        const array1d<vec3> positions_before = positions;
        const array1d<quat> rotations_before = rotations;
        const G1RuntimeByteSnapshot<G1IkState> state_before(fixture.state);
        G1IkFrameResult result = {};
        char error[256] = {};
        check(g1_runtime_evaluate(
                  fixture, positions, rotations, result,
                  error, static_cast<int>(sizeof(error))),
              error);
        const G1SwingSelectionDiagnostic& selection =
            result.feet[0].swing_selection;
        check(result.safe_stop_requested && !result.applied &&
                  result.stop_reason == G1IkStopNoSwingCandidate &&
                  selection.candidates_evaluated ==
                      G1SwingLiftCandidateCount &&
                  selection.selected_index == G1SwingNoCandidate &&
                  g1_runtime_candidate_is_default(
                      selection.selected) &&
                  g1_runtime_work_is_zero(
                      selection.total_clearance_work) &&
                  state_before.same(fixture.state) &&
                  g1_runtime_array_vec3_same(
                      positions, positions_before) &&
                  g1_runtime_array_quat_same(
                      rotations, rotations_before),
              "each real finite status advances through all 41 stages safely");
    }

    G1RuntimeFixture recovering;
    g1_runtime_make_fixture(recovering);
    recovering.field.heights.set(0.458f);
    for (int probe = 0; probe < 4; ++probe) {
        recovering.state.feet[0].swing
            .previous_sphere_centers[probe].y += 0.20f;
    }
    recovering.contact_values[1] = true;
    g1_runtime_refresh_current_probes(recovering);
    const G1FootTarget recovering_target =
        g1_runtime_current_target(recovering, 0);
    G1SwingCandidateDiagnostic recovery[2] = {};
    for (uint32_t candidate = 0; candidate < 2; ++candidate) {
        char error[256] = {};
        check(g1_runtime_stage_candidate_direct(
                  recovering, 0, candidate,
                  recovering.db.bone_positions(0),
                  recovering.db.bone_rotations(0),
                  recovering.state.feet[0].swing,
                  recovering_target,
                  recovery[candidate],
                  error, static_cast<int>(sizeof(error))),
              error);
    }
    check(recovery[0].clearance_status == G1ClearanceOk &&
              recovery[0].lower_margin_m < 0.0 &&
              !recovery[0].clearance_certified &&
              !g1_runtime_work_is_zero(
                  recovery[0].clearance_work) &&
              g1_runtime_candidate_passes(recovery[1]),
          "real negative Ok certificate rejects locally before later recovery");
    array1d<vec3> recovery_positions =
        recovering.db.bone_positions(0);
    array1d<quat> recovery_rotations =
        recovering.db.bone_rotations(0);
    G1IkFrameResult recovery_result = {};
    char error[256] = {};
    check(g1_runtime_evaluate(
              recovering, recovery_positions, recovery_rotations,
              recovery_result,
              error, static_cast<int>(sizeof(error))),
          error);
    G1ClearanceWork recovery_work = {};
    recovery_work = g1_runtime_work_add(
        recovery_work, recovery[0].clearance_work);
    recovery_work = g1_runtime_work_add(
        recovery_work, recovery[1].clearance_work);
    check(recovery_result.applied &&
              recovery_result.feet[0].swing_selection.selected_index == 1 &&
              g1_runtime_work_same(
                  recovery_result.feet[0].swing_selection
                      .total_clearance_work,
                  recovery_work),
          "negative Ok work remains truthful and later real stage wins");

    G1RuntimeFixture fatal_base;
    g1_runtime_make_fixture(fatal_base);
    vec3 current[4] = {};
    for (int probe = 0; probe < 4; ++probe) {
        current[probe] = fatal_base.state.feet[0].swing
            .previous_sphere_centers[probe];
    }

    G1SwingClearanceValidation strict_output;
    g1_runtime_poison_bytes(strict_output, 0x81);
    const G1RuntimeByteSnapshot<G1SwingClearanceValidation>
        strict_before(strict_output);
    G1SwingHistory invalid_history = fatal_base.state.feet[0].swing;
    invalid_history.initialized = false;
    check(g1_swing_clearance_validate(
              strict_output, g1_swing_foot_clearance_budget(),
              invalid_history, fatal_base.field,
              g1_left_leg_config(), current, false, 0.04f,
              NULL, 0) == G1ClearanceInvalidInput &&
              strict_before.same(strict_output),
          "real strict kernel reports transactional InvalidInput");
    G1RuntimeFixture invalid_runtime;
    g1_runtime_make_fixture(invalid_runtime);
    invalid_runtime.state.feet[0].swing.initialized = false;
    g1_runtime_require_unchanged_failure(
        invalid_runtime,
        "runtime preflight rejects the same invalid history unchanged");

    heightfield invalid_field = fatal_base.field;
    invalid_field.version = 1;
    check(g1_swing_clearance_validate(
              strict_output, g1_swing_foot_clearance_budget(),
              fatal_base.state.feet[0].swing, invalid_field,
              g1_left_leg_config(), current, false, 0.04f,
              NULL, 0) == G1ClearanceInvalidField &&
              strict_before.same(strict_output),
          "real strict kernel reports transactional InvalidField");
    G1RuntimeFixture invalid_field_runtime;
    g1_runtime_make_fixture(invalid_field_runtime);
    invalid_field_runtime.field.version = 1;
    g1_runtime_require_unchanged_failure(
        invalid_field_runtime,
        "runtime preflight rejects the same invalid field unchanged");

    const int saved_rounding = std::fegetround();
    check(saved_rounding != -1 && std::fesetround(FE_DOWNWARD) == 0,
          "runtime test enters the strict hostile arithmetic environment");
    check(g1_swing_clearance_validate(
              strict_output, g1_swing_foot_clearance_budget(),
              fatal_base.state.feet[0].swing, fatal_base.field,
              g1_left_leg_config(), current, false, 0.04f,
              NULL, 0) == G1ClearanceArithmeticFailure &&
              strict_before.same(strict_output),
          "real strict kernel reports transactional ArithmeticFailure");
    g1_runtime_require_unchanged_failure(
        fatal_base,
        "runtime aborts unchanged in the same hostile arithmetic environment");
    check(std::fesetround(saved_rounding) == 0,
          "runtime test restores caller rounding mode");

    G1RuntimeFixture controller;
    g1_runtime_make_fixture(controller);
    const G1LegConfig config = g1_left_leg_config();
    const vec3 current_contact =
        controller.global_positions(config.contact);
    array1d<quat> valid_pose = controller.db.bone_rotations(0);
    G1LegSolveResult valid_position = {};
    check(g1_apply_named_contact_position_ik(
              valid_pose,
              controller.db.bone_positions(0),
              controller.db.bone_rotations(0),
              controller.db.bone_parents,
              config, current_contact, valid_position,
              error, static_cast<int>(sizeof(error))),
          error);
    const array1d<quat> valid_orientation_baseline = valid_pose;
    G1FootOrientationResult valid_orientation = {};
    check(g1_apply_named_foot_orientation(
              valid_pose,
              controller.db.bone_positions(0),
              valid_orientation_baseline,
              controller.db.bone_parents,
              config, vec3(0.0f, 1.0f, 0.0f),
              valid_orientation,
              error, static_cast<int>(sizeof(error))),
          error);
    check(g1_ik_runtime_controller_constraints_pass(
              valid_position, valid_orientation,
              current_contact, current_contact),
          "shared production controller predicate accepts real no-op geometry");

    array1d<quat> far_pose = controller.db.bone_rotations(0);
    G1LegSolveResult far_position = {};
    check(g1_apply_named_contact_position_ik(
              far_pose,
              controller.db.bone_positions(0),
              controller.db.bone_rotations(0),
              controller.db.bone_parents,
              config, vec3(3.0f, 0.0f, 0.0f), far_position,
              error, static_cast<int>(sizeof(error))),
          error);
    // The checked contact solver converges reachable, unlimited results by
    // contract.  Its real residual rejection therefore overlaps reach here;
    // invoke the exact residual predicate explicitly before the shared
    // aggregate predicate exercises the same immutable result.
    check(!far_position.reachable &&
              far_position.contact_residual_m > 0.005f &&
              !g1_ik_contact_residual_is_converged(
                  far_position.contact_residual_m) &&
              !g1_ik_runtime_controller_constraints_pass(
                  far_position, valid_orientation,
                  current_contact, current_contact),
          "real reach/residual result is rejected by the production predicate");

    array1d<vec3> far_globals(G1_BoneCount);
    array1d<quat> far_global_rotations(G1_BoneCount);
    check(g1_ik_checked_forward_kinematics(
              far_globals, far_global_rotations,
              controller.db.bone_positions(0), far_pose,
              controller.db.bone_parents,
              error, static_cast<int>(sizeof(error))),
          error);

    array1d<quat> steep_pose = controller.db.bone_rotations(0);
    G1FootOrientationResult steep_orientation = {};
    const float steep = 45.0f * PIf / 180.0f;
    check(g1_apply_named_foot_orientation(
              steep_pose,
              controller.db.bone_positions(0),
              controller.db.bone_rotations(0),
              controller.db.bone_parents,
              config,
              vec3(-std::sin(steep), std::cos(steep), 0.0f),
              steep_orientation,
              error, static_cast<int>(sizeof(error))),
          error);
    check(steep_orientation.correction_limited &&
              !g1_ik_runtime_controller_constraints_pass(
                  valid_position, steep_orientation,
                  current_contact, current_contact),
          "real correction-limited result is rejected by production predicate");

    check(!g1_ik_runtime_controller_constraints_pass(
              valid_position, valid_orientation,
              current_contact, far_globals(config.contact)),
          "distinct real FK endpoints exercise invariance rejection");

    G1RuntimeFixture controller_ladder;
    g1_runtime_make_fixture(controller_ladder);
    const vec3 old_contact = controller_ladder.global_positions(
        g1_left_leg_config().contact);
    G1FootTarget established_target = {};
    check(g1_foot_lock_update(
              controller_ladder.state.feet[0].lock,
              established_target,
              controller_ladder.field,
              g1_left_leg_config(), old_contact,
              true, 0.04f,
              error, static_cast<int>(sizeof(error))),
          error);
    controller_ladder.db.bone_positions(0, G1_Simulation).y += 0.08f;
    check(g1_ik_checked_forward_kinematics(
              controller_ladder.global_positions,
              controller_ladder.global_rotations,
              controller_ladder.db.bone_positions(0),
              controller_ladder.db.bone_rotations(0),
              controller_ladder.db.bone_parents,
              error, static_cast<int>(sizeof(error))),
          error);
    g1_runtime_refresh_current_probes(controller_ladder);
    G1IkFrameTransaction controller_transaction = {};
    array1d<vec3> controller_positions;
    array1d<quat> controller_rotations;
    g1_runtime_begin(
        controller_ladder,
        controller_ladder.footprint,
        controller_transaction,
        controller_positions,
        controller_rotations);
    G1SwingCandidateDiagnostic controller_first = {};
    G1SwingCandidateDiagnostic controller_last = {};
    check(g1_runtime_stage_candidate_direct(
              controller_ladder, 0, 0,
              controller_positions, controller_rotations,
              controller_transaction.candidate_state.feet[0].swing,
              controller_transaction.candidate_result.feet[0].target,
              controller_first,
              error, static_cast<int>(sizeof(error))) &&
              g1_runtime_stage_candidate_direct(
                  controller_ladder, 0, 40,
                  controller_positions, controller_rotations,
                  controller_transaction.candidate_state.feet[0].swing,
                  controller_transaction.candidate_result.feet[0].target,
                  controller_last,
                  error, static_cast<int>(sizeof(error))),
          error);
    check(!controller_first.controller_constraints_passed &&
              g1_runtime_candidate_passes(controller_last) &&
              g1_ik_frame_stage_foot(
                  controller_transaction,
                  controller_positions,
                  controller_rotations,
                  0,
                  controller_ladder.db.bone_parents,
                  slice1d<bool>(2, controller_ladder.contact_values),
                  controller_ladder.field,
                  controller_ladder.footprint,
                  true, 0.04f,
                  error, static_cast<int>(sizeof(error))) &&
              controller_transaction.candidate_result.feet[0]
                  .swing_selection.selected_index == 40 &&
              controller_transaction.candidate_result.feet[0]
                  .swing_selection.candidates_evaluated == 41,
          "real controller-local rejection advances to the later ladder winner");
}

static void test_runtime_all_41_and_two_foot_composition()
{
    G1RuntimeFixture composed;
    g1_runtime_make_fixture(composed);
    composed.field.heights.set(0.46f);
    const vec3 old_left_contact =
        composed.global_positions(g1_left_leg_config().contact);
    G1FootTarget established = {};
    char error[256] = {};
    check(g1_foot_lock_update(
              composed.state.feet[0].lock,
              established,
              composed.field,
              g1_left_leg_config(),
              old_left_contact,
              true, 0.04f,
              error, static_cast<int>(sizeof(error))),
          error);
    composed.db.bone_positions(0, G1_Simulation).y += 0.01f;
    check(g1_ik_checked_forward_kinematics(
              composed.global_positions,
              composed.global_rotations,
              composed.db.bone_positions(0),
              composed.db.bone_rotations(0),
              composed.db.bone_parents,
              error, static_cast<int>(sizeof(error))),
          error);
    g1_runtime_refresh_current_probes(composed);
    for (int foot = 0; foot < 2; ++foot) {
        for (int probe = 0; probe < 4; ++probe) {
            composed.state.feet[foot].swing
                .previous_sphere_centers[probe].y += 0.20f;
        }
    }

    G1IkFrameTransaction transaction = {};
    array1d<vec3> scratch_positions;
    array1d<quat> scratch_rotations;
    g1_runtime_begin(
        composed, composed.footprint, transaction,
        scratch_positions, scratch_rotations);
    G1SwingCandidateDiagnostic expected[2] = {};
    uint32_t winners[2] = {
        G1SwingNoCandidate, G1SwingNoCandidate
    };
    const G1LegConfig composed_configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    quat selected_rotations[2][4] = {};
    for (uint32_t foot = 0; foot < 2; ++foot) {
        for (uint32_t candidate = 0;
             candidate < G1SwingLiftCandidateCount;
             ++candidate) {
            G1SwingCandidateDiagnostic diagnostic = {};
            check(g1_runtime_stage_candidate_direct(
                      composed, foot, candidate,
                      scratch_positions, scratch_rotations,
                      transaction.candidate_state.feet[foot].swing,
                      transaction.candidate_result.feet[foot].target,
                      diagnostic,
                      error, static_cast<int>(sizeof(error))),
                  error);
            if (g1_runtime_candidate_passes(diagnostic)) {
                winners[foot] = candidate;
                expected[foot] = diagnostic;
                break;
            }
        }
        check(winners[foot] != G1SwingNoCandidate,
              "real lock-release fixture discovers a winner for each foot");
        check(g1_ik_frame_stage_foot(
                  transaction, scratch_positions, scratch_rotations,
                  foot,
                  composed.db.bone_parents,
                  slice1d<bool>(2, composed.contact_values),
                  composed.field, composed.footprint,
                  true, 0.04f,
                  error, static_cast<int>(sizeof(error))),
              error);
        const G1SwingSelectionDiagnostic& selected =
            transaction.candidate_result.feet[foot]
                .swing_selection;
        check(selected.selected_index == winners[foot] &&
                  g1_runtime_candidate_same(
                      selected.selected, expected[foot]),
              "each real per-foot stage publishes its independent winner");
        const int selected_bones[] = {
            composed_configs[foot].hip,
            composed_configs[foot].knee,
            composed_configs[foot].ankle,
            composed_configs[foot].contact
        };
        for (int selected_bone = 0;
             selected_bone < 4;
             ++selected_bone) {
            selected_rotations[foot][selected_bone] =
                scratch_rotations(selected_bones[selected_bone]);
        }
        if (foot == 1) {
            const int left_bones[] = {
                composed_configs[0].hip,
                composed_configs[0].knee,
                composed_configs[0].ankle,
                composed_configs[0].contact
            };
            for (int selected_bone = 0;
                 selected_bone < 4;
                 ++selected_bone) {
                check(g1_test_quat_bits_same(
                          scratch_rotations(
                              left_bones[selected_bone]),
                          selected_rotations[0][selected_bone]),
                      "right staging preserves every named left winner rotation");
            }
        }
    }
    check(winners[0] == 5 && winners[1] == 0,
          "real released-lock geometry produces different fixed winners");

    G1IkState composed_state = composed.state;
    G1IkFrameResult composed_result = {};
    check(g1_ik_frame_finish(
              composed_state, composed_result,
              transaction, scratch_positions, scratch_rotations,
              composed.db.bone_parents,
              composed.field, 0.04f,
              error, static_cast<int>(sizeof(error))),
          error);
    check(composed_result.applied &&
              !composed_result.safe_stop_requested,
          "right-foot staging composes on the accepted left winner");

    for (int foot = 0; foot < 2; ++foot) {
        const int selected_bones[] = {
            composed_configs[foot].hip,
            composed_configs[foot].knee,
            composed_configs[foot].ankle,
            composed_configs[foot].contact
        };
        for (int selected_bone = 0;
             selected_bone < 4;
             ++selected_bone) {
            check(g1_test_quat_bits_same(
                      scratch_rotations(
                          selected_bones[selected_bone]),
                      selected_rotations[foot][selected_bone]),
                  "both selected foot rotation records survive finish exactly");
        }
    }

    array1d<vec3> final_globals(G1_BoneCount);
    array1d<quat> final_global_rotations(G1_BoneCount);
    check(g1_ik_checked_forward_kinematics(
              final_globals, final_global_rotations,
              scratch_positions, scratch_rotations,
              composed.db.bone_parents,
              error, static_cast<int>(sizeof(error))),
          error);
    for (int foot = 0; foot < 2; ++foot) {
        vec3 centers[4] = {};
        check(g1_runtime_independent_foot_centers(
                  centers, final_globals, final_global_rotations,
                  composed_configs[foot]),
              "composed public pose has independently checked endpoints");
        for (int probe = 0; probe < 4; ++probe) {
            const uint32_t words[3] = {
                terrain_float_bits(centers[probe].x),
                terrain_float_bits(centers[probe].y),
                terrain_float_bits(centers[probe].z)
            };
            for (int axis = 0; axis < 3; ++axis) {
                check(words[axis] == composed_result.feet[foot]
                          .swing_selection.selected
                          .actual_sphere_center_bits[probe][axis],
                      "both selected endpoint records survive final FK");
            }
        }
        G1SwingClearanceValidation independently_defensive = {};
        check(g1_swing_clearance_validate(
                  independently_defensive,
                  g1_swing_foot_clearance_budget(),
                  composed.state.feet[foot].swing,
                  composed.field,
                  composed_configs[foot],
                  centers,
                  false,
                  0.04f,
                  error,
                  static_cast<int>(sizeof(error))) == G1ClearanceOk,
              error);
        const G1SwingClearanceValidation& published_defensive =
            composed_result.feet[foot].defensive_swing;
        check(g1_runtime_double_bits(
                  published_defensive.lower_margin_m) ==
                  g1_runtime_double_bits(
                      independently_defensive.lower_margin_m) &&
              g1_runtime_double_bits(
                  published_defensive.witness_upper_m) ==
                  g1_runtime_double_bits(
                      independently_defensive.witness_upper_m) &&
              published_defensive.sweep_evaluated ==
                  independently_defensive.sweep_evaluated &&
              g1_runtime_work_same(
                  published_defensive.work,
                  independently_defensive.work),
              "finish publishes each independent defensive certificate exactly");
        for (int probe = 0; probe < 4; ++probe) {
            check(g1_runtime_vec3_bits_same(
                      composed_state.feet[foot].swing
                          .previous_sphere_centers[probe],
                      centers[probe]),
                  "finish commits each exact final center into swing history");
        }
    }
}

static void test_runtime_landing_ladder_and_late_rollback()
{
    G1RuntimeFixture late;
    g1_runtime_make_fixture(late);
    late.field.heights.set(0.458f);
    for (int foot = 0; foot < 2; ++foot) {
        for (int probe = 0; probe < 4; ++probe) {
            late.state.feet[foot].swing
                .previous_sphere_centers[probe].y += 0.20f;
        }
    }
    g1_runtime_refresh_current_probes(late);
    const G1RuntimeByteSnapshot<G1IkState> late_state_before(late.state);
    G1IkFrameTransaction transaction = {};
    array1d<vec3> scratch_positions;
    array1d<quat> scratch_rotations;
    g1_runtime_begin(
        late, late.footprint, transaction,
        scratch_positions, scratch_rotations);
    char error[256] = {};
    check(g1_ik_frame_stage_foot(
              transaction, scratch_positions, scratch_rotations, 0,
              late.db.bone_parents,
              slice1d<bool>(2, late.contact_values),
              late.field, late.footprint,
              true, 0.04f,
              error, static_cast<int>(sizeof(error))) &&
              transaction.candidate_result.feet[0]
                  .swing_selection.selected_index == 1,
          "real first-foot winner exists before a late split-call failure");
    const array1d<vec3> staged_positions = scratch_positions;
    const array1d<quat> staged_rotations = scratch_rotations;
    transaction.candidate_state.feet[1].swing.initialized = false;
    const G1RuntimeByteSnapshot<G1IkFrameTransaction>
        transaction_before(transaction);
    check(!g1_ik_frame_stage_foot(
              transaction, scratch_positions, scratch_rotations, 1,
              late.db.bone_parents,
              slice1d<bool>(2, late.contact_values),
              late.field, late.footprint,
              true, 0.04f, NULL, 0) &&
              transaction_before.same(transaction) &&
              g1_runtime_array_vec3_same(
                  scratch_positions, staged_positions) &&
              g1_runtime_array_quat_same(
                  scratch_rotations, staged_rotations) &&
              late_state_before.same(late.state),
          "late split failure leaves disposable work dirty but accepted state untouched");
}

static void test_runtime_poison_and_alias_rollback()
{
    G1RuntimeFixture fixture;
    g1_runtime_make_fixture(fixture);
    fixture.state.initialized = false;
    array1d<vec3> positions = fixture.db.bone_positions(0);
    array1d<quat> rotations = fixture.db.bone_rotations(0);
    const array1d<vec3> positions_before = positions;
    const array1d<quat> rotations_before = rotations;
    const G1RuntimeByteSnapshot<G1IkState> state_before(fixture.state);
    G1IkFrameResult result;
    g1_runtime_poison_bytes(result, 0xd9);
    const G1RuntimeByteSnapshot<G1IkFrameResult> result_before(result);
    check(!g1_runtime_evaluate(
              fixture, positions, rotations, result, NULL, 0),
          "poisoned uninitialized state is rejected");
    check(state_before.same(fixture.state) &&
              result_before.same(result) &&
              g1_runtime_array_vec3_same(positions, positions_before) &&
              g1_runtime_array_quat_same(rotations, rotations_before),
          "poisoned state rejection preserves every caller owner");

    G1RuntimeFixture alias;
    g1_runtime_make_fixture(alias);
    array1d<vec3> aliased_positions = alias.db.bone_positions(0);
    array1d<quat> aliased_rotations = alias.db.bone_rotations(0);
    const array1d<vec3> aliased_positions_before = aliased_positions;
    const array1d<quat> aliased_rotations_before = aliased_rotations;
    const G1RuntimeByteSnapshot<G1IkState> alias_state_before(alias.state);
    G1IkFrameResult alias_result;
    g1_runtime_poison_bytes(alias_result, 0xeb);
    const G1RuntimeByteSnapshot<G1IkFrameResult>
        alias_result_before(alias_result);
    char error[256] = {};
    check(!g1_ik_frame_evaluate(
              aliased_positions, aliased_rotations, alias.state,
              aliased_positions, alias.db.bone_rotations(0),
              alias.db.bone_parents,
              slice1d<bool>(2, alias.contact_values),
              alias.field, alias.footprint, true, 0.04f,
              alias_result,
              error, static_cast<int>(sizeof(error))),
          "mutable pose output aliasing immutable baseline is rejected");
    check(alias_state_before.same(alias.state) &&
              alias_result_before.same(alias_result) &&
              g1_runtime_array_vec3_same(
                  aliased_positions, aliased_positions_before) &&
              g1_runtime_array_quat_same(
                  aliased_rotations, aliased_rotations_before),
          "baseline/output alias rejection preserves all caller bytes");
}

static void test_runtime_lock_stage_order_and_command_invariance()
{
    G1RuntimeFixture fixture;
    g1_runtime_make_fixture(fixture);
    fixture.contact_values[0] = true;
    g1_runtime_refresh_current_probes(fixture);
    array1d<vec3> positions = fixture.db.bone_positions(0);
    array1d<quat> rotations = fixture.db.bone_rotations(0);
    G1IkFrameResult result = {};
    char error[256] = {};
    check(g1_runtime_evaluate(
              fixture, positions, rotations, result,
              error, static_cast<int>(sizeof(error))),
          error);
    check(result.applied && result.feet[0].recorded_contact &&
              result.feet[0].swing_selection.candidates_evaluated == 0,
          "recorded contact bypasses the 41-stage ladder");
    const vec3 locked_world = fixture.state.feet[0].lock.lock_point;

    fixture.db.bone_positions(0, G1_Simulation).y += 0.32f;
    check(g1_ik_checked_forward_kinematics(
              fixture.global_positions, fixture.global_rotations,
              fixture.db.bone_positions(0),
              fixture.db.bone_rotations(0),
              fixture.db.bone_parents,
              error, static_cast<int>(sizeof(error))),
          error);
    g1_runtime_refresh_current_probes(fixture);
    G1IkFrameTransaction level_transaction = {};
    array1d<vec3> level_positions;
    array1d<quat> level_rotations;
    g1_runtime_begin(
        fixture, fixture.footprint, level_transaction,
        level_positions, level_rotations);
    check(level_transaction.candidate_result.feet[0].recorded_contact &&
              g1_runtime_vec3_bits_same(
                  level_transaction.candidate_result.feet[0]
                      .target.sole_center,
                  locked_world),
          "planted lock retains its world-space target across root level change");

    fixture.contact_values[0] = false;
    g1_runtime_refresh_current_probes(fixture);
    G1IkFrameTransaction release_transaction = {};
    array1d<vec3> release_positions;
    array1d<quat> release_rotations;
    g1_runtime_begin(
        fixture, fixture.footprint, release_transaction,
        release_positions, release_rotations);
    check(release_transaction.candidate_state.feet[0].lock.releasing &&
              !release_transaction.candidate_result.feet[0]
                   .recorded_contact,
          "checked falling edge enters bounded release before swing staging");

    G1RuntimeFixture ordered;
    g1_runtime_make_fixture(ordered);
    G1CommandSnapshot command;
    g1_runtime_poison_bytes(command, 0x73);
    const G1RuntimeByteSnapshot<G1CommandSnapshot> command_before(command);
    G1IkFrameTransaction transaction = {};
    array1d<vec3> scratch_positions;
    array1d<quat> scratch_rotations;
    g1_runtime_begin(
        ordered, ordered.footprint, transaction,
        scratch_positions, scratch_rotations);
    check(command_before.same(command),
          "begin cannot mutate a sentinel command snapshot");
    const G1RuntimeByteSnapshot<G1IkFrameTransaction>
        before_wrong_order(transaction);
    const array1d<vec3> before_wrong_positions = scratch_positions;
    const array1d<quat> before_wrong_rotations = scratch_rotations;
    check(!g1_ik_frame_stage_foot(
              transaction, scratch_positions, scratch_rotations, 1,
              ordered.db.bone_parents,
              slice1d<bool>(2, ordered.contact_values),
              ordered.field, ordered.footprint,
              true, 0.04f, NULL, 0) &&
              before_wrong_order.same(transaction) &&
              g1_runtime_array_vec3_same(
                  scratch_positions, before_wrong_positions) &&
              g1_runtime_array_quat_same(
                  scratch_rotations, before_wrong_rotations),
          "stage rejects a foot that is not exactly next");
    check(command_before.same(command),
          "failed stage cannot mutate command bytes");
    check(g1_ik_frame_stage_foot(
              transaction, scratch_positions, scratch_rotations, 0,
              ordered.db.bone_parents,
              slice1d<bool>(2, ordered.contact_values),
              ordered.field, ordered.footprint,
              true, 0.04f,
              error, static_cast<int>(sizeof(error))),
          error);
    G1IkState output_state = ordered.state;
    G1IkFrameResult output_result = {};
    const G1RuntimeByteSnapshot<G1IkState>
        output_state_before(output_state);
    const G1RuntimeByteSnapshot<G1IkFrameResult>
        output_result_before(output_result);
    check(!g1_ik_frame_finish(
              output_state, output_result, transaction,
              scratch_positions, scratch_rotations,
              ordered.db.bone_parents, ordered.field, 0.04f,
              NULL, 0) &&
              output_state_before.same(output_state) &&
              output_result_before.same(output_result),
          "finish requires two fully staged feet transactionally");
    check(g1_ik_frame_stage_foot(
              transaction, scratch_positions, scratch_rotations, 1,
              ordered.db.bone_parents,
              slice1d<bool>(2, ordered.contact_values),
              ordered.field, ordered.footprint,
              true, 0.04f,
              error, static_cast<int>(sizeof(error))),
          error);
    check(g1_ik_frame_finish(
              output_state, output_result, transaction,
              scratch_positions, scratch_rotations,
              ordered.db.bone_parents, ordered.field, 0.04f,
              error, static_cast<int>(sizeof(error))),
          error);
    check(command_before.same(command),
          "finish cannot mutate requested travel or heading bytes");

    G1RuntimeFixture disabled;
    g1_runtime_make_fixture(disabled);
    const G1IkFrameResult canonical_disabled_result = {};
    const G1RuntimeByteSnapshot<G1IkFrameResult>
        canonical_disabled_bytes(canonical_disabled_result);
    G1IkFrameTransaction disabled_transaction = {};
    array1d<vec3> disabled_positions(G1_BoneCount);
    array1d<quat> disabled_rotations(G1_BoneCount);
    check(g1_ik_frame_begin(
              disabled_transaction,
              disabled_positions,
              disabled_rotations,
              disabled.state,
              disabled.db.bone_positions(0),
              disabled.db.bone_rotations(0),
              disabled.db.bone_parents,
              slice1d<bool>(2, disabled.contact_values),
              disabled.field,
              disabled.footprint,
              false,
              0.04f,
              error,
              static_cast<int>(sizeof(error))) &&
              canonical_disabled_bytes.same(
                  disabled_transaction.candidate_result) &&
              disabled_transaction.next_foot == 0 &&
              g1_runtime_array_vec3_same(
                  disabled_positions,
                  disabled.db.bone_positions(0)) &&
              g1_runtime_array_quat_same(
                  disabled_rotations,
                  disabled.db.bone_rotations(0)),
          error);
    const G1RuntimeByteSnapshot<G1IkFrameTransaction>
        disabled_before_active_stage(disabled_transaction);
    const array1d<vec3> disabled_positions_before_active_stage =
        disabled_positions;
    const array1d<quat> disabled_rotations_before_active_stage =
        disabled_rotations;
    check(!g1_ik_frame_stage_foot(
              disabled_transaction,
              disabled_positions,
              disabled_rotations,
              0,
              disabled.db.bone_parents,
              slice1d<bool>(2, disabled.contact_values),
              disabled.field,
              disabled.footprint,
              true,
              0.04f,
              NULL,
              0) &&
              disabled_before_active_stage.same(
                  disabled_transaction) &&
              g1_runtime_array_vec3_same(
                  disabled_positions,
                  disabled_positions_before_active_stage) &&
              g1_runtime_array_quat_same(
                  disabled_rotations,
                  disabled_rotations_before_active_stage),
          "disabled begin rejects an active split stage transactionally");

    G1RuntimeFixture active_mode;
    g1_runtime_make_fixture(active_mode);
    G1IkFrameTransaction active_mode_transaction = {};
    array1d<vec3> active_mode_positions(G1_BoneCount);
    array1d<quat> active_mode_rotations(G1_BoneCount);
    check(g1_ik_frame_begin(
              active_mode_transaction,
              active_mode_positions,
              active_mode_rotations,
              active_mode.state,
              active_mode.db.bone_positions(0),
              active_mode.db.bone_rotations(0),
              active_mode.db.bone_parents,
              slice1d<bool>(2, active_mode.contact_values),
              active_mode.field,
              active_mode.footprint,
              true,
              0.04f,
              error,
              static_cast<int>(sizeof(error))) &&
              !canonical_disabled_bytes.same(
                  active_mode_transaction.candidate_result),
          error);
    const G1RuntimeByteSnapshot<G1IkFrameTransaction>
        active_before_disabled_stage(active_mode_transaction);
    const array1d<vec3> active_positions_before_disabled_stage =
        active_mode_positions;
    const array1d<quat> active_rotations_before_disabled_stage =
        active_mode_rotations;
    check(!g1_ik_frame_stage_foot(
              active_mode_transaction,
              active_mode_positions,
              active_mode_rotations,
              0,
              active_mode.db.bone_parents,
              slice1d<bool>(2, active_mode.contact_values),
              active_mode.field,
              active_mode.footprint,
              false,
              0.04f,
              NULL,
              0) &&
              active_before_disabled_stage.same(
                  active_mode_transaction) &&
              g1_runtime_array_vec3_same(
                  active_mode_positions,
                  active_positions_before_disabled_stage) &&
              g1_runtime_array_quat_same(
                  active_mode_rotations,
                  active_rotations_before_disabled_stage),
          "active begin rejects a disabled split stage transactionally");

    check(g1_ik_frame_stage_foot(
              disabled_transaction,
              disabled_positions,
              disabled_rotations,
              0,
              disabled.db.bone_parents,
              slice1d<bool>(2, disabled.contact_values),
              disabled.field,
              disabled.footprint,
              false,
              0.04f,
              error,
              static_cast<int>(sizeof(error))) &&
              g1_ik_frame_stage_foot(
                  disabled_transaction,
                  disabled_positions,
                  disabled_rotations,
                  1,
                  disabled.db.bone_parents,
                  slice1d<bool>(2, disabled.contact_values),
                  disabled.field,
                  disabled.footprint,
                  false,
                  0.04f,
                  error,
                  static_cast<int>(sizeof(error))),
          error);
    G1IkState disabled_output_state = disabled.state;
    const G1RuntimeByteSnapshot<G1IkState>
        disabled_state_before(disabled_output_state);
    G1IkFrameResult disabled_output_result;
    g1_runtime_poison_bytes(disabled_output_result, 0x4d);
    check(g1_ik_frame_finish(
              disabled_output_state,
              disabled_output_result,
              disabled_transaction,
              disabled_positions,
              disabled_rotations,
              disabled.db.bone_parents,
              disabled.field,
              0.04f,
              error,
              static_cast<int>(sizeof(error))) &&
              canonical_disabled_bytes.same(
                  disabled_output_result) &&
              canonical_disabled_bytes.same(
                  disabled_transaction.candidate_result) &&
              disabled_transaction.next_foot == 2 &&
              disabled_transaction.candidate_result.feet[0]
                      .swing_selection.candidates_evaluated == 0 &&
              disabled_transaction.candidate_result.feet[1]
                      .swing_selection.candidates_evaluated == 0 &&
              disabled_state_before.same(disabled_output_state) &&
              g1_runtime_array_vec3_same(
                  disabled_positions,
                  disabled.db.bone_positions(0)) &&
              g1_runtime_array_quat_same(
                  disabled_rotations,
                  disabled.db.bone_rotations(0)),
          "disabled split publishes the exact default result without state or pose");

    G1IkSafeStopHandoff handoff = {};
    check(g1_ik_safe_stop_handoff(
              handoff, true, vec3(1.25f, 0.75f, -0.50f),
              error, static_cast<int>(sizeof(error))) &&
              terrain_float_bits(handoff.applied_velocity.x) == 0 &&
              terrain_float_bits(handoff.applied_velocity.z) == 0 &&
              terrain_float_bits(handoff.applied_velocity.y) ==
                  terrain_float_bits(0.75f) &&
              handoff.cancel_planar_inertia && handoff.force_search,
          "latched safe stop zeroes only planar travel and forces one search");
    check(g1_ik_safe_stop_handoff(
              handoff, false, vec3(1.25f, 0.75f, -0.50f),
              error, static_cast<int>(sizeof(error)) ) &&
              g1_runtime_vec3_bits_same(
                  handoff.applied_velocity,
                  vec3(1.25f, 0.75f, -0.50f)) &&
              !handoff.cancel_planar_inertia && !handoff.force_search,
          "unlatched handoff preserves the complete requested velocity");

    const float negative_zero = -0.0f;
    check(g1_ik_safe_stop_handoff(
              handoff, false,
              vec3(negative_zero, negative_zero, 0.0f),
              error, static_cast<int>(sizeof(error))) &&
              terrain_float_bits(handoff.applied_velocity.x) ==
                  UINT32_C(0x80000000) &&
              terrain_float_bits(handoff.applied_velocity.y) ==
                  UINT32_C(0x80000000) &&
              terrain_float_bits(handoff.applied_velocity.z) ==
                  UINT32_C(0x00000000),
          "unlatched handoff copies positive/negative signed-zero words exactly");
    check(g1_ik_safe_stop_handoff(
              handoff, true,
              vec3(negative_zero, negative_zero, negative_zero),
              error, static_cast<int>(sizeof(error))) &&
              terrain_float_bits(handoff.applied_velocity.x) ==
                  UINT32_C(0x00000000) &&
              terrain_float_bits(handoff.applied_velocity.y) ==
                  UINT32_C(0x80000000) &&
              terrain_float_bits(handoff.applied_velocity.z) ==
                  UINT32_C(0x00000000) &&
              handoff.cancel_planar_inertia && handoff.force_search,
          "latched handoff canonicalizes only planar zeros and retains signed Y");

    const char* const names[] = {
        "none", "footprint-blocked", "footprint-outside-domain",
        "footprint-budget-exceeded", "landing-patch-unavailable",
        "target-unreachable", "no-swing-candidate",
        "pose-clearance-rejected"
    };
    for (int reason = G1IkStopNone;
         reason <= G1IkStopPoseClearanceRejected;
         ++reason) {
        check(std::strcmp(
                  g1_ik_stop_reason_name(
                      static_cast<G1IkStopReason>(reason)),
                  names[reason]) == 0,
              "stop reason diagnostic order is exact");
    }
    check(std::strcmp(
              g1_ik_stop_reason_name(
                  static_cast<G1IkStopReason>(999)),
              "invalid") == 0,
          "unknown stop reason is diagnostic-only invalid");
}

static int run_runtime_parity_mode()
{
    G1RuntimeFixture fixture;
    g1_runtime_make_fixture(fixture);
    G1SwingHistory history = fixture.state.feet[0].swing;
    vec3 centers[4] = {};
    for (int probe = 0; probe < 4; ++probe) {
        centers[probe] = fixture.footprint.feet[0]
            .probes[probe].current_sphere_center;
    }
    G1SwingClearanceValidation validation = {};
    const G1ClearanceStatus status = g1_swing_clearance_validate(
        validation, g1_swing_foot_clearance_budget(),
        history, fixture.field, g1_left_leg_config(),
        centers, false, 0.04f, NULL, 0);
    std::printf(
        "runtime-parity status=%u ladder=%08x,%08x,%08x "
        "margins=%016llx,%016llx work=%u,%u,%u,%u,%u,%u\n",
        static_cast<unsigned int>(status),
        G1SwingLiftCandidateBits[0],
        G1SwingLiftCandidateBits[20],
        G1SwingLiftCandidateBits[40],
        static_cast<unsigned long long>(
            g1_runtime_double_bits(validation.lower_margin_m)),
        static_cast<unsigned long long>(
            g1_runtime_double_bits(validation.witness_upper_m)),
        validation.work.point_queries,
        validation.work.cells_visited,
        validation.work.primitive_triangle_pairs,
        validation.work.face_patches,
        validation.work.candidate_tests,
        validation.work.subdivision_nodes);
    return status == G1ClearanceOk ? 0 : 1;
}

#endif

int main(int argc, char** argv)
{
#if defined(G1_IK_ENABLE_TEST_SEAMS)
    if (argc == 2 && std::strcmp(argv[1], "--parity") == 0) {
        return run_runtime_parity_mode();
    }
#else
    (void)argc;
    (void)argv;
#endif
    test_explicit_leg_geometry();
    test_database_shape_and_chain_validation();
    test_database_local_basis_validation();
    test_hostile_config_validation();
    test_null_and_short_error_buffers();
    test_checked_surface_query();
    test_planted_lock_lifecycle();
    test_planted_lock_drift_dt_and_rollback();
    test_generic_checked_ik_math();
    test_named_solver_success_and_bend_mapping();
    test_named_solver_preflight_and_rollback();
    test_named_contact_residual_contract();
    test_surface_aligned_named_foot_orientation();
    test_surface_orientation_staged_pose_and_fallback();
    test_surface_orientation_limit_semantics();
    test_surface_orientation_hostile_rollback();
#if defined(G1_IK_ENABLE_TEST_SEAMS)
    test_runtime_lift_ladder_and_observer_crosspath();
    test_runtime_state_reset_and_footprint_preflight();
    test_runtime_blocked_and_landing_contract();
    test_runtime_real_41_stage_selector();
    test_runtime_status_table_and_local_rejections();
    test_runtime_all_41_and_two_foot_composition();
    test_runtime_landing_ladder_and_late_rollback();
    test_runtime_poison_and_alias_rollback();
    test_runtime_lock_stage_order_and_command_invariance();
#endif
    return 0;
}
