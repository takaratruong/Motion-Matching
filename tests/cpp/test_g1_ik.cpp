#include "g1_ik.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static void check(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "G1 IK test failed: %s\n", message);
        std::exit(1);
    }
}

static void make_g1_database(database& db)
{
    static const int parents[G1_BoneCount] = {
        -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
    };
    db.bone_positions.resize(1, G1_BoneCount);
    db.bone_rotations.resize(1, G1_BoneCount);
    db.bone_parents.resize(G1_BoneCount);
    db.bone_positions.set(vec3());
    db.bone_rotations.set(quat());
    for (int i = 0; i < G1_BoneCount; ++i) {
        db.bone_parents(i) = parents[i];
    }
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

int main()
{
    test_explicit_leg_geometry();
    test_database_shape_and_chain_validation();
    test_hostile_config_validation();
    test_null_and_short_error_buffers();
    return 0;
}
