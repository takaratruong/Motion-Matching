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

int main()
{
    test_explicit_leg_geometry();
    test_database_shape_and_chain_validation();
    test_database_local_basis_validation();
    test_hostile_config_validation();
    test_null_and_short_error_buffers();
    test_checked_surface_query();
    test_planted_lock_lifecycle();
    test_planted_lock_drift_dt_and_rollback();
    return 0;
}
