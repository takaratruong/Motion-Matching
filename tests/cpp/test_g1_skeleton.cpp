#include "database.h"
#include "g1_skeleton.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <unistd.h>

static void check(bool condition, const char* expression, int line)
{
    if (!condition) {
        std::fprintf(stderr, "CHECK failed at line %d: %s\n", line, expression);
        std::exit(1);
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

static const int expected_parents[G1_BoneCount] = {
    -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
};

static void make_valid_database(database& db)
{
    db.bone_positions.resize(1, G1_BoneCount);
    db.bone_parents.resize(G1_BoneCount);
    for (int i = 0; i < G1_BoneCount; ++i) {
        db.bone_parents(i) = expected_parents[i];
    }
}

static void test_named_indices_and_signature()
{
    const int expected_indices[G1_BoneCount] = {
        G1_Simulation,
        G1_Hips,
        G1_LeftHipPitch,
        G1_LeftHipRoll,
        G1_LeftHipYaw,
        G1_LeftKnee,
        G1_LeftAnkle,
        G1_LeftToe,
        G1_RightHipPitch,
        G1_RightHipRoll,
        G1_RightHipYaw,
        G1_RightKnee,
        G1_RightAnkle,
        G1_RightToe,
        G1_Spine,
        G1_Spine1,
        G1_Spine2,
        G1_LeftShoulderPitch,
        G1_LeftShoulderRoll,
        G1_LeftShoulderYaw,
        G1_LeftElbow,
        G1_LeftWristRoll,
        G1_LeftWristPitch,
        G1_LeftWrist,
        G1_RightShoulderPitch,
        G1_RightShoulderRoll,
        G1_RightShoulderYaw,
        G1_RightElbow,
        G1_RightWristRoll,
        G1_RightWristPitch,
        G1_RightWrist
    };
    for (int i = 0; i < G1_BoneCount; ++i) {
        CHECK(expected_indices[i] == i);
    }
    CHECK(std::strcmp(
        G1_SkeletonSignature,
        "6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7") == 0);
}

static void test_skeleton_contract()
{
    char error[256] = {};
    database db;
    make_valid_database(db);
    CHECK(g1_skeleton_validate(db, error, sizeof(error)));

    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        database mutated;
        make_valid_database(mutated);
        mutated.bone_parents(bone) = expected_parents[bone] == -1 ? 0 : -1;
        std::memset(error, 0, sizeof(error));
        CHECK(!g1_skeleton_validate(mutated, error, sizeof(error)));
        CHECK(std::strstr(error, "parent") != NULL);
    }

    database wrong_columns;
    make_valid_database(wrong_columns);
    wrong_columns.bone_positions.resize(1, G1_BoneCount - 1);
    CHECK(!g1_skeleton_validate(wrong_columns, error, sizeof(error)));
    CHECK(std::strstr(error, "bone") != NULL);

    database missing_parents;
    missing_parents.bone_positions.resize(1, G1_BoneCount);
    CHECK(!g1_skeleton_validate(missing_parents, error, sizeof(error)));
    CHECK(std::strstr(error, "parent") != NULL);

    database short_parents;
    make_valid_database(short_parents);
    short_parents.bone_parents.resize(G1_BoneCount - 1);
    CHECK(!g1_skeleton_validate(short_parents, error, sizeof(error)));
    CHECK(std::strstr(error, "parent") != NULL);

    database long_parents;
    make_valid_database(long_parents);
    long_parents.bone_parents.resize(G1_BoneCount + 1);
    CHECK(!g1_skeleton_validate(long_parents, error, sizeof(error)));
    CHECK(std::strstr(error, "parent") != NULL);

    CHECK(!g1_skeleton_validate(wrong_columns, NULL, 0));
}

static void test_matching_feature_bone_indices_are_checked()
{
    database db;
    db.bone_positions.resize(1, G1_BoneCount);
    const int invalid_indices[][3] = {
        {-1, G1_RightAnkle, G1_Hips},
        {G1_BoneCount, G1_RightAnkle, G1_Hips},
        {G1_LeftAnkle, -1, G1_Hips},
        {G1_LeftAnkle, G1_BoneCount, G1_Hips},
        {G1_LeftAnkle, G1_RightAnkle, -1},
        {G1_LeftAnkle, G1_RightAnkle, G1_BoneCount}
    };
    for (size_t i = 0;
         i < sizeof(invalid_indices) / sizeof(invalid_indices[0]);
         ++i) {
        database_build_matching_features(
            db,
            1.0f,
            1.0f,
            1.0f,
            1.0f,
            1.0f,
            invalid_indices[i][0],
            invalid_indices[i][1],
            invalid_indices[i][2]);
        CHECK(db.features.rows == 0);
        CHECK(db.features.cols == 0);
    }
}

static std::string make_temp_path()
{
    char path[] = "/tmp/test_g1_manifest_XXXXXX";
    const int descriptor = mkstemp(path);
    CHECK(descriptor >= 0);
    CHECK(close(descriptor) == 0);
    return std::string(path);
}

static void write_text(const std::string& path, const std::string& text)
{
    FILE* file = std::fopen(path.c_str(), "wb");
    CHECK(file != NULL);
    CHECK(std::fwrite(text.data(), 1, text.size(), file) == text.size());
    CHECK(std::fclose(file) == 0);
}

static bool validate_text(const std::string& text, char* error, int capacity)
{
    const std::string path = make_temp_path();
    write_text(path, text);
    const bool valid = g1_manifest_validate(path.c_str(), error, capacity);
    CHECK(unlink(path.c_str()) == 0);
    return valid;
}

static std::string quoted_signature()
{
    return std::string("\"") + G1_SkeletonSignature + "\"";
}

static void test_manifest_contract()
{
    char error[256] = {};
    const std::string signature = quoted_signature();

    CHECK(validate_text(
        std::string("{\"schema\":\"v1\",\"skeleton\":{\"signature\":") +
            signature + "},\"sources\":[1,true,null,{\"x\":-2.5e+3}]}",
        error,
        sizeof(error)));

    CHECK(validate_text(
        std::string("{\"skeleton\":{\"signat\\u0075re\":") +
            signature + "}}",
        error,
        sizeof(error)));

    CHECK(!validate_text("{\"skeleton\":{}}", error, sizeof(error)));
    CHECK(std::strstr(error, "signature") != NULL);

    CHECK(!validate_text(
        "{\"signature\":\"6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7\"}",
        error,
        sizeof(error)));
    CHECK(std::strstr(error, "skeleton") != NULL);

    CHECK(!validate_text(
        std::string("{\"note\":") + signature +
            ",\"skeleton\":{\"signature\":\"wrong\"}}",
        error,
        sizeof(error)));
    CHECK(std::strstr(error, "mismatch") != NULL);

    CHECK(!validate_text(
        std::string("{\"skeleton\":{\"signature\":") + signature +
            ",\"signature\":" + signature + "}}",
        error,
        sizeof(error)));
    CHECK(std::strstr(error, "ambiguous") != NULL);

    CHECK(!validate_text(
        std::string("{\"skeleton\":{\"signature\":") + signature +
            "},\"skeleton\":{\"signature\":" + signature + "}}",
        error,
        sizeof(error)));
    CHECK(std::strstr(error, "ambiguous") != NULL);

    CHECK(!validate_text(
        std::string("{\"skeleton\":{\"signature\":7},\"note\":") +
            signature + "}",
        error,
        sizeof(error)));
    CHECK(std::strstr(error, "mismatch") != NULL);

    const char* malformed[] = {
        "",
        "{",
        "[]",
        "{\"skeleton\":{\"signature\":\"unterminated}}",
        "{\"skeleton\":{\"signature\":\"bad\\q\"}}",
        "{\"skeleton\":{\"signature\":01}}",
        "{\"skeleton\":{\"signature\":true,}}",
        "{\"skeleton\":{\"signature\":true}} trailing"
    };
    for (size_t i = 0; i < sizeof(malformed) / sizeof(malformed[0]); ++i) {
        CHECK(!validate_text(malformed[i], error, sizeof(error)));
        CHECK(std::strstr(error, "JSON") != NULL);
    }

    CHECK(!g1_manifest_validate(
        "/tmp/a_g1_manifest_that_does_not_exist", error, sizeof(error)));
    CHECK(std::strstr(error, "open") != NULL);
    CHECK(!g1_manifest_validate(NULL, error, sizeof(error)));
    CHECK(std::strstr(error, "path") != NULL);
}

static void test_manifest_size_preflight()
{
    if (sizeof(off_t) < 8) {
        return;
    }

    char path[] = "/tmp/test_g1_manifest_large_XXXXXX";
    const int descriptor = mkstemp(path);
    CHECK(descriptor >= 0);
    CHECK(ftruncate(descriptor, static_cast<off_t>(INT_MAX)) == 0);
    CHECK(close(descriptor) == 0);

    char error[256] = {};
    CHECK(!g1_manifest_validate(path, error, sizeof(error)));
    CHECK(std::strstr(error, "large") != NULL);
    CHECK(unlink(path) == 0);
}

static bool read_u32_le(FILE* file, uint32_t& value)
{
    unsigned char bytes[4];
    if (std::fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
        return false;
    }
    value = static_cast<uint32_t>(bytes[0]) |
            (static_cast<uint32_t>(bytes[1]) << 8) |
            (static_cast<uint32_t>(bytes[2]) << 16) |
            (static_cast<uint32_t>(bytes[3]) << 24);
    return true;
}

static bool skip_real_array2d(
    FILE* file,
    uint32_t element_size,
    uint32_t& expected_rows,
    uint32_t& expected_columns)
{
    uint32_t rows = 0;
    uint32_t columns = 0;
    if (!read_u32_le(file, rows) || !read_u32_le(file, columns)) {
        return false;
    }
    if (expected_rows == 0) {
        expected_rows = rows;
        expected_columns = columns;
    } else if (rows != expected_rows || columns != expected_columns) {
        return false;
    }
    const uint64_t bytes =
        static_cast<uint64_t>(rows) * columns * element_size;
    return bytes <= static_cast<uint64_t>(LONG_MAX) &&
           std::fseek(file, static_cast<long>(bytes), SEEK_CUR) == 0;
}

static bool load_real_database_skeleton(const char* path, database& db)
{
    FILE* file = std::fopen(path, "rb");
    if (file == NULL) {
        return false;
    }

    uint32_t rows = 0;
    uint32_t columns = 0;
    const bool arrays_ok =
        skip_real_array2d(file, 12, rows, columns) &&
        skip_real_array2d(file, 12, rows, columns) &&
        skip_real_array2d(file, 16, rows, columns) &&
        skip_real_array2d(file, 12, rows, columns);
    uint32_t parent_count = 0;
    if (!arrays_ok || !read_u32_le(file, parent_count) ||
        columns > static_cast<uint32_t>(INT_MAX) ||
        parent_count > static_cast<uint32_t>(INT_MAX)) {
        std::fclose(file);
        return false;
    }

    db.bone_positions.resize(1, static_cast<int>(columns));
    db.bone_parents.resize(static_cast<int>(parent_count));
    for (uint32_t i = 0; i < parent_count; ++i) {
        uint32_t encoded = 0;
        if (!read_u32_le(file, encoded)) {
            std::fclose(file);
            return false;
        }
        db.bone_parents(static_cast<int>(i)) = static_cast<int32_t>(encoded);
    }
    return std::fclose(file) == 0;
}

static void test_real_artifacts(const char* manifest_path, const char* database_path)
{
    char error[256] = {};
    CHECK(g1_manifest_validate(manifest_path, error, sizeof(error)));
    database db;
    CHECK(load_real_database_skeleton(database_path, db));
    CHECK(g1_skeleton_validate(db, error, sizeof(error)));
}

int main(int argc, char** argv)
{
    test_named_indices_and_signature();
    test_skeleton_contract();
    test_matching_feature_bone_indices_are_checked();
    test_manifest_contract();
    test_manifest_size_preflight();

    if (argc == 4 && std::strcmp(argv[1], "--real") == 0) {
        test_real_artifacts(argv[2], argv[3]);
    } else {
        CHECK(argc == 1);
    }
    return 0;
}
