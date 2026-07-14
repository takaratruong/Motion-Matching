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
    test_generic_checked_ik_math();
    test_named_solver_success_and_bend_mapping();
    test_named_solver_preflight_and_rollback();
    test_named_contact_residual_contract();
    return 0;
}
