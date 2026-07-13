#include "database.h"
#include "g1_skeleton.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(bool condition, const char* expression, int line)
{
    if (!condition) {
        std::fprintf(stderr, "CHECK failed at line %d: %s\n", line, expression);
        std::exit(1);
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

static uint32_t float_bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float float_from_bits(uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool finite_bits(float value)
{
    return (float_bits(value) & UINT32_C(0x7f800000)) !=
           UINT32_C(0x7f800000);
}

static bool close_enough(float actual, float expected, float tolerance = 2e-4f)
{
    return std::fabs(actual - expected) <= tolerance;
}

static void fill_source_database(database& db, int frames)
{
    db.bone_positions.resize(frames, G1_BoneCount);
    db.bone_velocities.resize(frames, G1_BoneCount);
    db.bone_rotations.resize(frames, G1_BoneCount);
    db.bone_angular_velocities.resize(frames, G1_BoneCount);
    db.bone_parents.resize(G1_BoneCount);
    db.range_starts.resize(1);
    db.range_stops.resize(1);
    db.terrain_features.resize(frames, 4);

    db.bone_parents.set(-1);
    db.bone_parents(G1_Hips) = G1_Simulation;
    db.bone_parents(G1_LeftAnkle) = G1_Simulation;
    db.bone_parents(G1_RightAnkle) = G1_Simulation;
    db.range_starts(0) = 0;
    db.range_stops(0) = frames;

    for (int frame = 0; frame < frames; ++frame) {
        const float f = static_cast<float>(frame);
        for (int bone = 0; bone < G1_BoneCount; ++bone) {
            db.bone_positions(frame, bone) = vec3();
            db.bone_velocities(frame, bone) = vec3();
            db.bone_rotations(frame, bone) = quat();
            db.bone_angular_velocities(frame, bone) = vec3();
        }
        db.bone_positions(frame, G1_Simulation) =
            vec3(0.11f * f, 0.0f, 0.013f * f * f + 0.07f * f);
        db.bone_rotations(frame, G1_Simulation) =
            quat_from_angle_axis(0.025f * f, vec3(0.0f, 1.0f, 0.0f));

        db.bone_positions(frame, G1_LeftAnkle) =
            vec3(-0.2f + 0.003f * f, -0.8f + 0.002f * f, 0.1f + 0.004f * f);
        db.bone_positions(frame, G1_RightAnkle) =
            vec3(0.2f - 0.002f * f, -0.79f + 0.001f * f, -0.1f + 0.005f * f);

        db.bone_velocities(frame, G1_LeftAnkle) =
            vec3(0.01f * f, 0.02f + 0.003f * f, -0.04f + 0.002f * f);
        db.bone_velocities(frame, G1_RightAnkle) =
            vec3(-0.008f * f, 0.03f + 0.002f * f, 0.01f + 0.004f * f);
        db.bone_velocities(frame, G1_Hips) =
            vec3(0.02f + 0.003f * f, -0.01f + 0.001f * f, 0.04f + 0.005f * f);

        for (int dimension = 0; dimension < 4; ++dimension) {
            db.terrain_features(frame, dimension) =
                0.1f * static_cast<float>(dimension + 1) +
                0.006f * static_cast<float>((dimension + 1) * frame);
        }
    }
}

static void fill_expected_raw_features(
    const database& db, int frame, float expected[31])
{
    int offset = 0;
    const int position_bones[2] = {G1_LeftAnkle, G1_RightAnkle};
    for (int i = 0; i < 2; ++i) {
        const vec3 value = db.bone_positions(frame, position_bones[i]);
        expected[offset++] = value.x;
        expected[offset++] = value.y;
        expected[offset++] = value.z;
    }

    const int velocity_bones[3] = {
        G1_LeftAnkle, G1_RightAnkle, G1_Hips
    };
    for (int i = 0; i < 3; ++i) {
        const vec3 value = db.bone_velocities(frame, velocity_bones[i]);
        expected[offset++] = value.x;
        expected[offset++] = value.y;
        expected[offset++] = value.z;
    }

    int horizons[3] = {};
    database_trajectory_horizons(horizons);
    for (int i = 0; i < 3; ++i) {
        const int future =
            frame + horizons[i] < db.range_stops(0)
                ? frame + horizons[i]
                : db.range_stops(0) - 1;
        const vec3 value = quat_inv_mul_vec3(
            db.bone_rotations(frame, G1_Simulation),
            db.bone_positions(future, G1_Simulation) -
                db.bone_positions(frame, G1_Simulation));
        expected[offset++] = value.x;
        expected[offset++] = value.z;
    }

    for (int i = 0; i < 3; ++i) {
        const int future =
            frame + horizons[i] < db.range_stops(0)
                ? frame + horizons[i]
                : db.range_stops(0) - 1;
        const vec3 value = quat_inv_mul_vec3(
            db.bone_rotations(frame, G1_Simulation),
            quat_mul_vec3(
                db.bone_rotations(future, G1_Simulation),
                vec3(0.0f, 0.0f, 1.0f)));
        expected[offset++] = value.x;
        expected[offset++] = value.z;
    }

    for (int dimension = 0; dimension < 4; ++dimension) {
        expected[offset++] = db.terrain_features(frame, dimension);
    }
    CHECK(offset == 31);
}

static void build_features(database& db, float terrain_weight = 1.0f)
{
    database_build_matching_features(
        db,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        G1_LeftAnkle,
        G1_RightAnkle,
        G1_Hips,
        terrain_weight);
}

static void test_zero_weight_is_exact_and_safe()
{
    array2d<float> features(3, 4);
    array1d<float> offsets(4);
    array1d<float> scales(4);
    features.set(7.25f);

    normalize_feature(features, offsets, scales, 0, 4, 0.0f);
    for (int row = 0; row < features.rows; ++row) {
        for (int column = 0; column < features.cols; ++column) {
            CHECK(float_bits(features(row, column)) == UINT32_C(0));
        }
    }
    for (int column = 0; column < 4; ++column) {
        CHECK(float_bits(scales(column)) == float_bits(FLT_MAX));
        CHECK(float_bits(offsets(column)) == float_bits(7.25f));
    }

    array1d<float> normalized(6);
    array1d<float> denormalized_offsets(6);
    array1d<float> disabled_scales(6);
    normalized(0) = FLT_MAX;
    normalized(1) = -FLT_MAX;
    normalized(2) = float_from_bits(UINT32_C(0x7f800000));
    normalized(3) = float_from_bits(UINT32_C(0xff800000));
    normalized(4) = float_from_bits(UINT32_C(0x7fc12345));
    normalized(5) = float_from_bits(UINT32_C(0xffc54321));
    for (int i = 0; i < normalized.size; ++i) {
        denormalized_offsets(i) =
            float_from_bits(UINT32_C(0x3f000000) + static_cast<uint32_t>(i));
        disabled_scales(i) = FLT_MAX;
    }
    denormalize_features(normalized, denormalized_offsets, disabled_scales);
    for (int i = 0; i < normalized.size; ++i) {
        CHECK(float_bits(normalized(i)) == float_bits(denormalized_offsets(i)));
    }
}

static void test_builder_layout_and_real_horizons()
{
    database db;
    fill_source_database(db, 30);
    build_features(db);

    CHECK(db.features.rows == 30);
    CHECK(db.features.cols == 31);
    CHECK(db.features_offset.size == 31);
    CHECK(db.features_scale.size == 31);

    const int frames_to_check[2] = {0, 28};
    for (int frame_index = 0; frame_index < 2; ++frame_index) {
        const int frame = frames_to_check[frame_index];
        float expected[31] = {};
        fill_expected_raw_features(db, frame, expected);
        for (int dimension = 0; dimension < 31; ++dimension) {
            const float actual =
                db.features(frame, dimension) * db.features_scale(dimension) +
                db.features_offset(dimension);
            CHECK(close_enough(actual, expected[dimension]));
        }
    }

    int horizons[3] = {};
    database_trajectory_horizons(horizons);
    CHECK(horizons[0] == 8);
    CHECK(horizons[1] == 17);
    CHECK(horizons[2] == 25);
}

static void seed_matching_outputs(database& db)
{
    db.features.resize(2, 3);
    db.features_offset.resize(3);
    db.features_scale.resize(3);
    db.bound_sm_min.resize(1, 3);
    db.bound_sm_max.resize(1, 3);
    db.bound_lr_min.resize(1, 3);
    db.bound_lr_max.resize(1, 3);
    for (int i = 0; i < 6; ++i) db.features.data[i] = 10.0f + i;
    for (int i = 0; i < 3; ++i) {
        db.features_offset(i) = 20.0f + i;
        db.features_scale(i) = 30.0f + i;
        db.bound_sm_min(0, i) = 40.0f + i;
        db.bound_sm_max(0, i) = 50.0f + i;
        db.bound_lr_min(0, i) = 60.0f + i;
        db.bound_lr_max(0, i) = 70.0f + i;
    }
}

static void check_matching_outputs_unchanged(const database& db)
{
    CHECK(db.features.rows == 2 && db.features.cols == 3);
    for (int i = 0; i < 6; ++i) CHECK(db.features.data[i] == 10.0f + i);
    for (int i = 0; i < 3; ++i) {
        CHECK(db.features_offset(i) == 20.0f + i);
        CHECK(db.features_scale(i) == 30.0f + i);
        CHECK(db.bound_sm_min(0, i) == 40.0f + i);
        CHECK(db.bound_sm_max(0, i) == 50.0f + i);
        CHECK(db.bound_lr_min(0, i) == 60.0f + i);
        CHECK(db.bound_lr_max(0, i) == 70.0f + i);
    }
}

static void test_builder_rejects_bad_terrain_before_mutation()
{
    {
        database db;
        fill_source_database(db, 30);
        db.terrain_features.resize(0, 0);
        seed_matching_outputs(db);
        build_features(db);
        check_matching_outputs_unchanged(db);
    }
    {
        database db;
        fill_source_database(db, 30);
        db.terrain_features.resize(0, 0);
        db.terrain_features.resize(29, 4);
        seed_matching_outputs(db);
        build_features(db);
        check_matching_outputs_unchanged(db);
    }
    {
        database db;
        fill_source_database(db, 30);
        db.terrain_features.resize(0, 0);
        db.terrain_features.resize(30, 3);
        seed_matching_outputs(db);
        build_features(db);
        check_matching_outputs_unchanged(db);
    }
    {
        database db;
        fill_source_database(db, 30);
        seed_matching_outputs(db);
        database_build_matching_features(
            db, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            G1_LeftAnkle, G1_RightAnkle, G1_Hips, -1.0f);
        check_matching_outputs_unchanged(db);
    }
}

static void test_builder_rejects_every_nonfinite_weight_before_mutation()
{
    const float invalid_weights[3] = {
        float_from_bits(UINT32_C(0x7fc12345)),
        float_from_bits(UINT32_C(0x7f800000)),
        float_from_bits(UINT32_C(0xff800000))
    };

    for (int weight_index = 0; weight_index < 6; ++weight_index) {
        for (int value_index = 0; value_index < 3; ++value_index) {
            database db;
            fill_source_database(db, 30);
            seed_matching_outputs(db);
            float weights[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            weights[weight_index] = invalid_weights[value_index];
            database_build_matching_features(
                db,
                weights[0],
                weights[1],
                weights[2],
                weights[3],
                weights[4],
                G1_LeftAnkle,
                G1_RightAnkle,
                G1_Hips,
                weights[5]);
            check_matching_outputs_unchanged(db);
        }
    }
}

static void test_positive_weight_flat_terrain_is_release_safe()
{
#ifdef NDEBUG
    database db;
    fill_source_database(db, 30);
    db.terrain_features.set(2.5f);
    build_features(db, 1.0f);

    CHECK(db.features.rows == db.nframes());
    CHECK(db.features.cols == 31);
    for (int dimension = 27; dimension < 31; ++dimension) {
        CHECK(finite_bits(db.features_offset(dimension)));
        CHECK(close_enough(db.features_offset(dimension), 2.5f, 1e-5f));
        CHECK(float_bits(db.features_scale(dimension)) == float_bits(FLT_MAX));
        for (int frame = 0; frame < db.nframes(); ++frame) {
            CHECK(float_bits(db.features(frame, dimension)) == UINT32_C(0));
        }
    }

    array1d<float> query(31);
    for (int dimension = 0; dimension < 31; ++dimension) {
        query(dimension) = db.features_offset(dimension);
    }
    query(27) = float_from_bits(UINT32_C(0x7fc12345));
    query(28) = float_from_bits(UINT32_C(0x7f800000));
    query(29) = float_from_bits(UINT32_C(0xff800000));
    query(30) = FLT_MAX;

    const float frame_cost = database_frame_cost(db, 0, query);
    CHECK(finite_bits(frame_cost));
    int best_index = 0;
    float best_cost = FLT_MAX;
    database_search(best_index, best_cost, db, query, 0.0f, 0, 1);
    CHECK(best_index >= 0 && best_index < db.nframes());
    CHECK(finite_bits(best_cost));
#endif
}

static void test_all_disabled_constant_builder()
{
    database db;
    fill_source_database(db, 3);
    for (int frame = 0; frame < db.nframes(); ++frame) {
        for (int bone = 0; bone < db.nbones(); ++bone) {
            db.bone_positions(frame, bone) = vec3();
            db.bone_velocities(frame, bone) = vec3();
            db.bone_rotations(frame, bone) = quat();
            db.bone_angular_velocities(frame, bone) = vec3();
        }
    }
    db.terrain_features.set(2.5f);

    database_build_matching_features(
        db, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        G1_LeftAnkle, G1_RightAnkle, G1_Hips, 0.0f);
    CHECK(db.features.rows == db.nframes());
    CHECK(db.features.cols == 31);
    for (int dimension = 0; dimension < 31; ++dimension) {
        CHECK(float_bits(db.features_scale(dimension)) == float_bits(FLT_MAX));
        for (int frame = 0; frame < db.nframes(); ++frame) {
            CHECK(float_bits(db.features(frame, dimension)) == UINT32_C(0));
        }
    }
}

static void fill_search_database(database& db)
{
    db.bone_positions.resize(4, 1);
    db.features.resize(4, 31);
    db.features_offset.resize(31);
    db.features_scale.resize(31);
    db.range_starts.resize(1);
    db.range_stops.resize(1);
    db.terrain_features.resize(4, 4);
    db.features.zero();
    db.features_offset.zero();
    db.features_scale.set(FLT_MAX);
    db.features_scale(0) = 1.0f;
    db.features(0, 0) = 3.0f;
    db.features(1, 0) = 2.0f;
    db.features(2, 0) = 0.0f;
    db.features(3, 0) = 4.0f;
    db.range_starts(0) = 0;
    db.range_stops(0) = 4;
    for (int frame = 0; frame < 4; ++frame) {
        for (int dimension = 0; dimension < 4; ++dimension) {
            db.terrain_features(frame, dimension) =
                static_cast<float>(10 * frame + dimension);
        }
    }
    database_build_bounds(db);
}

static void fill_adversarial_disabled_query(array1d<float>& query)
{
    query.resize(31);
    query.zero();
    const float values[6] = {
        FLT_MAX,
        -FLT_MAX,
        float_from_bits(UINT32_C(0x7f800000)),
        float_from_bits(UINT32_C(0xff800000)),
        float_from_bits(UINT32_C(0x7fc12345)),
        float_from_bits(UINT32_C(0xffc54321))
    };
    for (int dimension = 1; dimension < query.size; ++dimension) {
        query(dimension) = values[(dimension - 1) % 6];
    }
}

static void test_cost_and_incumbent_search_semantics()
{
    database db;
    fill_search_database(db);
    array1d<float> query;
    fill_adversarial_disabled_query(query);

    const float current_cost = database_frame_cost(db, 0, query);
    CHECK(float_bits(current_cost) == float_bits(9.0f));
    CHECK(finite_bits(current_cost));

    db.features_scale(30) = 1.0f;
    query(30) = 2.0f;
    const float active_terrain_cost = database_frame_cost(db, 0, query);
    CHECK(float_bits(active_terrain_cost) == float_bits(13.0f));
    CHECK(finite_bits(active_terrain_cost));
    db.features_scale(30) = FLT_MAX;
    fill_adversarial_disabled_query(query);

    int best_index = 2;
    float best_cost = FLT_MAX;
    database_search(best_index, best_cost, db, query, 0.0f, 0, 1);
    CHECK(best_index == 2);
    CHECK(float_bits(best_cost) == UINT32_C(0));

    best_index = 0;
    best_cost = FLT_MAX;
    database_search(best_index, best_cost, db, query, 0.0f, 0, 2);
    CHECK(best_index == 2);
    CHECK(float_bits(best_cost) == UINT32_C(0));
    CHECK(finite_bits(best_cost));

    query(27) = 21.0f;
    query(28) = 23.0f;
    query(29) = 25.0f;
    query(30) = 27.0f;
    const float raw_error = database_raw_terrain_error(db, 2, query);
    CHECK(float_bits(raw_error) == float_bits(30.0f));
    CHECK(finite_bits(raw_error));
}

int main()
{
    test_zero_weight_is_exact_and_safe();
    test_builder_layout_and_real_horizons();
    test_builder_rejects_bad_terrain_before_mutation();
    test_builder_rejects_every_nonfinite_weight_before_mutation();
    test_positive_weight_flat_terrain_is_release_safe();
    test_all_disabled_constant_builder();
    test_cost_and_incumbent_search_semantics();
    return 0;
}
