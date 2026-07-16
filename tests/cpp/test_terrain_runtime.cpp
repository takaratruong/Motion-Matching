#include <stddef.h>
#include <stdlib.h>

static bool test_terrain_payload_allocation_failure_armed = false;

static void* test_terrain_payload_allocate(size_t bytes)
{
    if (test_terrain_payload_allocation_failure_armed) {
        test_terrain_payload_allocation_failure_armed = false;
        return NULL;
    }
    return malloc(bytes);
}

#define TERRAIN_RUNTIME_PAYLOAD_ALLOCATE(bytes) \
    test_terrain_payload_allocate(bytes)
#include "terrain_runtime.h"
#undef TERRAIN_RUNTIME_PAYLOAD_ALLOCATE
#include "quat.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <limits>
#include <random>
#include <string>
#include <vector>

typedef std::vector<unsigned char> byte_buffer;

static void check(bool condition, const char* message)
{
    if (!condition) {
        fprintf(stderr, "terrain runtime test failed: %s\n", message);
        abort();
    }
}

static void check_close(float actual, float expected, const char* message)
{
    check(terrain_float_is_finite(actual), message);
    check(fabsf(actual - expected) < 1e-5f, message);
}

static uint32_t float_bits(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float float_from_bits(uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void check_float_bits(
    float actual, uint32_t expected, const char* message)
{
    check(float_bits(actual) == expected, message);
}

static void check_vec3_bits(
    vec3 actual,
    uint32_t expected_x,
    uint32_t expected_y,
    uint32_t expected_z,
    const char* message)
{
    check(float_bits(actual.x) == expected_x, message);
    check(float_bits(actual.y) == expected_y, message);
    check(float_bits(actual.z) == expected_z, message);
}

static void check_centerline_snapshot_bits_equal(
    const terrain_centerline_snapshot& actual,
    const terrain_centerline_snapshot& expected,
    const char* message)
{
    for (int i = 0; i < 4; ++i) {
        check(float_bits(actual.values[i]) == float_bits(expected.values[i]),
              message);
        check(float_bits(actual.points[i].x) ==
                  float_bits(expected.points[i].x),
              message);
        check(float_bits(actual.points[i].y) ==
                  float_bits(expected.points[i].y),
              message);
        check(float_bits(actual.points[i].z) ==
                  float_bits(expected.points[i].z),
              message);
    }
}

static float expected_centerline_relative_value(
    float sample_height, float base_height)
{
    const double difference =
        static_cast<double>(sample_height) -
        static_cast<double>(base_height);
    return static_cast<float>(difference);
}

static void check_centerline_latch_matches_sweep(
    const terrain_centerline_snapshot& snapshot,
    const heightfield& field,
    vec3 query_root,
    int first_latched,
    const walkability_sweep_result& expected_sweep,
    const char* message)
{
    const float safe_height = heightfield_sample_v2(
        field, expected_sweep.point.x, expected_sweep.point.z);
    const float base_height = heightfield_sample_v2(
        field, query_root.x, query_root.z);
    const float expected_value = expected_centerline_relative_value(
        safe_height, base_height);
    const vec3 expected_point(
        expected_sweep.point.x, safe_height, expected_sweep.point.z);
    for (int i = first_latched; i < 4; ++i) {
        check_float_bits(
            snapshot.values[i], float_bits(expected_value), message);
        check_vec3_bits(
            snapshot.points[i],
            float_bits(expected_point.x),
            float_bits(expected_point.y),
            float_bits(expected_point.z),
            message);
    }
}

static quat heading_positive_x()
{
    return quat_from_angle_axis(0.5f * PIf, vec3(0.0f, 1.0f, 0.0f));
}

static quat heading_vertical()
{
    return quat_from_angle_axis(0.5f * PIf, vec3(1.0f, 0.0f, 0.0f));
}

static void initialize_heightfield(
    heightfield& field,
    int nx,
    int nz,
    float origin_x,
    float origin_z,
    float cell_size,
    float exterior_height,
    uint32_t version = 1)
{
    field.version = version;
    field.nx = nx;
    field.nz = nz;
    field.origin_x = origin_x;
    field.origin_z = origin_z;
    field.cell_size = cell_size;
    field.exterior_height = exterior_height;
    field.heights.resize(nx * nz);
}

static void append_bytes(byte_buffer& out, const void* data, size_t size)
{
    const unsigned char* first = static_cast<const unsigned char*>(data);
    out.insert(out.end(), first, first + size);
}

static void append_u32_le(byte_buffer& out, uint32_t value)
{
    unsigned char encoded[4] = {
        static_cast<unsigned char>(value),
        static_cast<unsigned char>(value >> 8),
        static_cast<unsigned char>(value >> 16),
        static_cast<unsigned char>(value >> 24),
    };
    append_bytes(out, encoded, sizeof(encoded));
}

static void append_float_le(byte_buffer& out, float value)
{
    static_assert(sizeof(float) == sizeof(uint32_t),
                  "terrain artifacts require 32-bit floats");
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    append_u32_le(out, bits);
}

static byte_buffer make_sidecar(
    uint32_t version,
    uint32_t frames,
    uint32_t dimensions,
    const std::vector<float>& values)
{
    byte_buffer out;
    append_bytes(out, "G1TF", 4);
    append_u32_le(out, version);
    append_u32_le(out, frames);
    append_u32_le(out, dimensions);
    for (size_t i = 0; i < values.size(); ++i) {
        append_float_le(out, values[i]);
    }
    return out;
}

static byte_buffer make_heightfield(
    uint32_t version,
    uint32_t nx,
    uint32_t nz,
    float origin_x,
    float origin_z,
    float cell_size,
    float exterior_height,
    const std::vector<float>& heights)
{
    byte_buffer out;
    append_bytes(out, "G1HF", 4);
    append_u32_le(out, version);
    append_u32_le(out, nx);
    append_u32_le(out, nz);
    append_float_le(out, origin_x);
    append_float_le(out, origin_z);
    append_float_le(out, cell_size);
    append_float_le(out, exterior_height);
    for (size_t i = 0; i < heights.size(); ++i) {
        append_float_le(out, heights[i]);
    }
    return out;
}

static byte_buffer make_support(
    uint32_t version,
    uint32_t frames,
    uint32_t dimensions,
    const std::vector<float>& values)
{
    byte_buffer out;
    append_bytes(out, "G1SP", 4);
    append_u32_le(out, version);
    append_u32_le(out, frames);
    append_u32_le(out, dimensions);
    for (size_t i = 0; i < values.size(); ++i) append_float_le(out, values[i]);
    return out;
}

static byte_buffer make_walkability(
    uint32_t version,
    uint32_t nx,
    uint32_t nz,
    const std::vector<uint8_t>& cells)
{
    byte_buffer out;
    append_bytes(out, "G1WM", 4);
    append_u32_le(out, version);
    append_u32_le(out, nx);
    append_u32_le(out, nz);
    if (!cells.empty()) append_bytes(out, cells.data(), cells.size());
    return out;
}

static void write_prefix(
    const char* path, const byte_buffer& payload, size_t length)
{
    check(length <= payload.size(), "fixture prefix length");
    FILE* file = fopen(path, "wb");
    check(file != NULL, "fixture file open");
    if (length != 0) {
        check(fwrite(payload.data(), 1, length, file) == length,
              "fixture file write");
    }
    check(fclose(file) == 0, "fixture file close");
}

static void write_payload(const char* path, const byte_buffer& payload)
{
    write_prefix(path, payload, payload.size());
}

static void assert_error(
    const char* error, const char* path, const char* expected_reason)
{
    check(error[0] != '\0', "rejection error is non-empty");
    check(strstr(error, path) != NULL, "rejection error includes path");
    check(strstr(error, expected_reason) != NULL,
          "rejection error includes reason");
}

static void expect_sidecar_rejected(
    const char* path, const char* expected_reason)
{
    terrain_feature_set destination;
    destination.values.resize(2, 4);
    for (int i = 0; i < destination.values.rows; ++i) {
        for (int j = 0; j < destination.values.cols; ++j) {
            destination.values(i, j) = 100.0f + i * 10.0f + j;
        }
    }

    char error[256] = {};
    check(!terrain_features_load(
              destination, path, error, static_cast<int>(sizeof(error))),
          "sidecar fixture rejected");
    assert_error(error, path, expected_reason);
    check(destination.values.rows == 2, "sidecar rejection preserves rows");
    check(destination.values.cols == 4, "sidecar rejection preserves columns");
    for (int i = 0; i < destination.values.rows; ++i) {
        for (int j = 0; j < destination.values.cols; ++j) {
            check(destination.values(i, j) == 100.0f + i * 10.0f + j,
                  "sidecar rejection preserves values");
        }
    }
}

static void expect_heightfield_rejected(
    const char* path, const char* expected_reason)
{
    heightfield destination;
    destination.version = 77;
    destination.nx = 3;
    destination.nz = 2;
    destination.origin_x = 10.0f;
    destination.origin_z = 20.0f;
    destination.cell_size = 0.25f;
    destination.exterior_height = -5.0f;
    destination.heights.resize(6);
    for (int i = 0; i < destination.heights.size; ++i) {
        destination.heights(i) = 200.0f + i;
    }

    char error[256] = {};
    check(!heightfield_load(
              destination, path, error, static_cast<int>(sizeof(error))),
          "heightfield fixture rejected");
    assert_error(error, path, expected_reason);
    check(destination.version == 77,
          "heightfield rejection preserves version");
    check(destination.nx == 3, "heightfield rejection preserves nx");
    check(destination.nz == 2, "heightfield rejection preserves nz");
    check(destination.origin_x == 10.0f,
          "heightfield rejection preserves origin x");
    check(destination.origin_z == 20.0f,
          "heightfield rejection preserves origin z");
    check(destination.cell_size == 0.25f,
          "heightfield rejection preserves cell size");
    check(destination.exterior_height == -5.0f,
          "heightfield rejection preserves exterior height");
    check(destination.heights.size == 6,
          "heightfield rejection preserves storage size");
    for (int i = 0; i < destination.heights.size; ++i) {
        check(destination.heights(i) == 200.0f + i,
              "heightfield rejection preserves values");
    }
}

static void test_payload_allocation_failures_are_actionable_and_transactional()
{
    {
        const char* path = "/tmp/test_g1tf_allocation.bin";
        write_payload(path, make_sidecar(
            1, 2, 4,
            {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}));
        terrain_feature_set destination;
        destination.values.resize(2, 4);
        for (int i = 0; i < 8; ++i)
            destination.values.data[i] = 101.0f + static_cast<float>(i);
        const int rows = destination.values.rows;
        const int cols = destination.values.cols;
        float* const data = destination.values.data;
        const std::vector<float> values(data, data + rows * cols);
        char error[256] = {};
        test_terrain_payload_allocation_failure_armed = true;
        check(!terrain_features_load(
                  destination, path, error, static_cast<int>(sizeof(error))),
              "G1TF allocation failure rejection");
        check(!test_terrain_payload_allocation_failure_armed,
              "G1TF allocation hook consumed");
        assert_error(error, path, "allocate");
        check(destination.values.rows == rows &&
                  destination.values.cols == cols &&
                  destination.values.data == data,
              "G1TF allocation failure preserves destination fields");
        for (int i = 0; i < rows * cols; ++i)
            check(destination.values.data[i] == values[static_cast<size_t>(i)],
                  "G1TF allocation failure preserves destination payload");
    }

    {
        const char* path = "/tmp/test_g1hf_allocation.bin";
        write_payload(path, make_heightfield(
            2, 2, 2, 0.0f, 0.0f, 1.0f, -1.0f,
            {0.0f, 1.0f, 2.0f, 3.0f}));
        heightfield destination;
        destination.version = 77;
        destination.nx = 3;
        destination.nz = 2;
        destination.origin_x = 10.0f;
        destination.origin_z = 20.0f;
        destination.cell_size = 0.25f;
        destination.exterior_height = -5.0f;
        destination.heights.resize(6);
        for (int i = 0; i < destination.heights.size; ++i)
            destination.heights(i) = 201.0f + static_cast<float>(i);
        const uint32_t version = destination.version;
        const int nx = destination.nx;
        const int nz = destination.nz;
        const float origin_x = destination.origin_x;
        const float origin_z = destination.origin_z;
        const float cell_size = destination.cell_size;
        const float exterior_height = destination.exterior_height;
        const int count = destination.heights.size;
        float* const data = destination.heights.data;
        const std::vector<float> values(data, data + count);
        char error[256] = {};
        test_terrain_payload_allocation_failure_armed = true;
        check(!heightfield_load(
                  destination, path, error, static_cast<int>(sizeof(error))),
              "G1HF allocation failure rejection");
        check(!test_terrain_payload_allocation_failure_armed,
              "G1HF allocation hook consumed");
        assert_error(error, path, "allocate");
        check(destination.version == version &&
                  destination.nx == nx && destination.nz == nz &&
                  destination.origin_x == origin_x &&
                  destination.origin_z == origin_z &&
                  destination.cell_size == cell_size &&
                  destination.exterior_height == exterior_height &&
                  destination.heights.size == count &&
                  destination.heights.data == data,
              "G1HF allocation failure preserves destination fields");
        for (int i = 0; i < count; ++i)
            check(destination.heights(i) == values[static_cast<size_t>(i)],
                  "G1HF allocation failure preserves destination payload");
    }

    {
        const char* path = "/tmp/test_g1sp_allocation.bin";
        write_payload(path, make_support(
            1, 2, 3, {0.0f, 0.1f, 0.2f, 1.0f, 1.1f, 1.2f}));
        terrain_support_set destination;
        destination.values.resize(2, 3);
        for (int i = 0; i < 6; ++i)
            destination.values.data[i] = 301.0f + static_cast<float>(i);
        const int rows = destination.values.rows;
        const int cols = destination.values.cols;
        float* const data = destination.values.data;
        const std::vector<float> values(data, data + rows * cols);
        char error[256] = {};
        test_terrain_payload_allocation_failure_armed = true;
        check(!terrain_support_load(
                  destination, path, 2, error,
                  static_cast<int>(sizeof(error))),
              "G1SP allocation failure rejection");
        check(!test_terrain_payload_allocation_failure_armed,
              "G1SP allocation hook consumed");
        assert_error(error, path, "allocate");
        check(destination.values.rows == rows &&
                  destination.values.cols == cols &&
                  destination.values.data == data,
              "G1SP allocation failure preserves destination fields");
        for (int i = 0; i < rows * cols; ++i)
            check(destination.values.data[i] == values[static_cast<size_t>(i)],
                  "G1SP allocation failure preserves destination payload");
    }

    {
        const char* path = "/tmp/test_g1wm_allocation.bin";
        write_payload(path, make_walkability(1, 3, 2, {0, 1, 2, 2, 1, 0}));
        heightfield field;
        initialize_heightfield(
            field, 3, 2, -1.0f, 2.0f, 0.02f, -3.0f, 2);
        field.heights.zero();
        walkability_grid destination;
        destination.nx = 3;
        destination.nz = 2;
        destination.cells.resize(6);
        for (int i = 0; i < destination.cells.size; ++i)
            destination.cells(i) = static_cast<uint8_t>(2 - i % 3);
        const int nx = destination.nx;
        const int nz = destination.nz;
        const int count = destination.cells.size;
        uint8_t* const data = destination.cells.data;
        const std::vector<uint8_t> values(data, data + count);
        char error[256] = {};
        test_terrain_payload_allocation_failure_armed = true;
        check(!walkability_load(
                  destination, path, field, error,
                  static_cast<int>(sizeof(error))),
              "G1WM allocation failure rejection");
        check(!test_terrain_payload_allocation_failure_armed,
              "G1WM allocation hook consumed");
        assert_error(error, path, "allocate");
        check(destination.nx == nx && destination.nz == nz &&
                  destination.cells.size == count &&
                  destination.cells.data == data,
              "G1WM allocation failure preserves destination fields");
        for (int i = 0; i < count; ++i)
            check(destination.cells(i) == values[static_cast<size_t>(i)],
                  "G1WM allocation failure preserves destination payload");
    }
}

static void test_support_loader_is_strict_transactional_and_frame_exact()
{
    const char* path = "/tmp/test_g1sp.bin";
    const std::vector<float> values = {
        0.0f, 0.1f, 0.2f,
        1.0f, 1.1f, 1.2f,
    };
    write_payload(path, make_support(1, 2, 3, values));
    terrain_support_set support;
    char error[256] = {};
    check(terrain_support_load(support, path, 2, error, sizeof(error)), error);
    check(support.values.rows == 2 && support.values.cols == 3,
          "G1SP shape");
    check_close(support.values(1, 2), 1.2f, "G1SP payload");

    support.values.set(9.0f);
    write_payload(path, make_support(2, 2, 3, values));
    check(!terrain_support_load(support, path, 2, error, sizeof(error)),
          "G1SP version rejection");
    check(strstr(error, path) && strstr(error, "version"),
          "G1SP version diagnostic");
    check(support.values(0, 0) == 9.0f, "G1SP transaction");

    write_payload(path, make_support(1, 2, 4, values));
    check(!terrain_support_load(support, path, 2, error, sizeof(error)),
          "G1SP dimensions rejection");
    write_payload(path, make_support(1, 2, 3, values));
    check(!terrain_support_load(support, path, 3, error, sizeof(error)),
          "G1SP frame parity rejection");

    std::vector<float> nonfinite = values;
    nonfinite[4] = std::numeric_limits<float>::quiet_NaN();
    write_payload(path, make_support(1, 2, 3, nonfinite));
    check(!terrain_support_load(support, path, 2, error, sizeof(error)),
          "G1SP finite rejection");

    const byte_buffer valid = make_support(1, 2, 3, values);
    for (size_t size = 0; size < valid.size(); ++size) {
        write_prefix(path, valid, size);
        check(!terrain_support_load(support, path, 2, error, sizeof(error)),
              "G1SP truncation rejection");
    }
    byte_buffer trailing = valid;
    trailing.push_back(0x7f);
    write_payload(path, trailing);
    check(!terrain_support_load(support, path, 2, error, sizeof(error)),
          "G1SP trailing rejection");
}

static void test_walkability_loader_is_strict_transactional_and_grid_exact()
{
    const char* path = "/tmp/test_g1wm.bin";
    heightfield field;
    field.version = 2;
    initialize_heightfield(field, 3, 2, -1.0f, 2.0f, 0.02f, -3.0f);
    field.heights.zero();
    const std::vector<uint8_t> cells = {0, 1, 2, 2, 1, 0};
    write_payload(path, make_walkability(1, 3, 2, cells));

    walkability_grid grid;
    char error[256] = {};
    check(walkability_load(grid, path, field, error, sizeof(error)), error);
    check(grid.nx == 3 && grid.nz == 2 && grid.cells.size == 6,
          "G1WM shape");
    for (int i = 0; i < grid.cells.size; ++i)
        check(grid.cells(i) == cells[static_cast<size_t>(i)], "G1WM payload");

    grid.cells.set(2);
    write_payload(path, make_walkability(2, 3, 2, cells));
    check(!walkability_load(grid, path, field, error, sizeof(error)),
          "G1WM version rejection");
    check(grid.nx == 3 && grid.nz == 2 && grid.cells(0) == 2,
          "G1WM transaction");

    write_payload(path, make_walkability(1, 2, 2, {0, 1, 2, 0}));
    check(!walkability_load(grid, path, field, error, sizeof(error)),
          "G1WM grid mismatch rejection");
    write_payload(path, make_walkability(1, 3, 2, {0, 1, 3, 2, 1, 0}));
    check(!walkability_load(grid, path, field, error, sizeof(error)),
          "G1WM class rejection");

    const byte_buffer valid = make_walkability(1, 3, 2, cells);
    for (size_t size = 0; size < valid.size(); ++size) {
        write_prefix(path, valid, size);
        check(!walkability_load(grid, path, field, error, sizeof(error)),
              "G1WM truncation rejection");
    }
    byte_buffer trailing = valid;
    trailing.push_back(1);
    write_payload(path, trailing);
    check(!walkability_load(grid, path, field, error, sizeof(error)),
          "G1WM trailing rejection");
}

static void test_walkability_binary32_half_cell_parity()
{
    heightfield field;
    field.version = 2;
    initialize_heightfield(field, 3, 2, -0.02f, 0.0f, 0.02f, 0.0f);
    field.heights.zero();
    walkability_grid grid;
    grid.nx = 3; grid.nz = 2; grid.cells.resize(6);
    const uint8_t values[6] = {1, 2, 1, 1, 2, 1};
    for (int i = 0; i < 6; ++i) grid.cells(i) = values[i];
    const float tie = -0.01f;
    check(walkability_class_at(
          grid, field, nextafterf(tie, -INFINITY), 0.0f) == 1,
          "binary32 predecessor stays below half-cell");
    check(walkability_class_at(grid, field, tie, 0.0f) == 2,
          "binary32 half-cell tie chooses positive index");
    check(walkability_class_at(
          grid, field, nextafterf(tie, INFINITY), 0.0f) == 2,
          "binary32 successor stays above half-cell");

    heightfield rounded;
    rounded.version = 2;
    initialize_heightfield(rounded, 3, 2, -1.0f, 0.0f, 0.02f, 0.0f);
    rounded.heights.zero();
    check(walkability_class_at(
          grid, rounded, -0.9900000095367432f, 0.0f) == 1,
          "rounded mathematical midpoint follows one-round producer ops");
}

static void test_walkability_nearest_axis_preserves_large_endpoint()
{
    const int count = 8388610;
    const float origin = 0.0f;
    const float cell_size = 1.0f;
    const float last_node = 8388609.0f;
    check(terrain_v2_runtime_axis_is_valid(
              origin, static_cast<uint32_t>(count), cell_size),
          "large endpoint probe uses a valid v2 axis");

    int index = -17;
    check(walkability_nearest_axis(
              index, last_node, origin, cell_size, count) &&
              index == count - 1,
          "large valid endpoint clamps shifted rounding to last index");

    index = -19;
    check(walkability_nearest_axis(
              index, nextafterf(last_node, -INFINITY),
              origin, cell_size, count) &&
              index == count - 2,
          "large endpoint predecessor retains its adjacent index");

    index = 23;
    check(!walkability_nearest_axis(
              index, nextafterf(last_node, INFINITY),
              origin, cell_size, count) &&
              index == 23,
          "large endpoint successor is exterior and leaves output unchanged");

    index = 29;
    check(!walkability_nearest_axis(
              index, float_from_bits(UINT32_C(0x4f000000)),
              0.0f, 1.0f, INT_MAX) &&
              index == 29,
          "hostile count rejects float-cast overflow transactionally");
}

static void initialize_walkability_guard_fixture(
    heightfield& field, walkability_grid& grid)
{
    initialize_heightfield(
        field, 11, 5, 0.0f, 0.0f, 0.10f, 0.0f, 2);
    field.heights.zero();
    grid.nx = 11;
    grid.nz = 5;
    grid.cells.resize(55);
    grid.cells.set(1);
    for (int z = 0; z < grid.nz; ++z) grid.cells(z * grid.nx + 7) = 0;
    grid.cells(2 * grid.nx + 4) = 2;
}

static void test_walkability_reason_names_are_stable()
{
    check(strcmp(walkability_reason_name(walkability_clear), "clear") == 0,
          "clear walkability reason name");
    check(strcmp(walkability_reason_name(walkability_blocked_cell),
                 "blocked-cell") == 0,
          "blocked-cell walkability reason name");
    check(strcmp(walkability_reason_name(walkability_out_of_bounds),
                 "out-of-bounds") == 0,
          "out-of-bounds walkability reason name");
    check(strcmp(walkability_reason_name(walkability_nonfinite),
                 "nonfinite") == 0,
          "nonfinite walkability reason name");
    check(strcmp(walkability_reason_name(
                     static_cast<walkability_reason>(INT_MAX)),
                 "nonfinite") == 0,
          "unknown walkability reason maps to nonfinite");
}

static void test_walkability_structural_gate_fails_closed()
{
    heightfield field;
    walkability_grid grid;
    initialize_walkability_guard_fixture(field, grid);

    heightfield undersized_heights(field);
    undersized_heights.heights.resize(1);
    walkability_reason reason = walkability_clear;
    check(!walkability_grid_matches_heightfield(grid, undersized_heights) &&
              walkability_class_at(
                  grid, undersized_heights, 0.4f, 0.2f) == 0 &&
              walkability_footprint_class(
                  grid, undersized_heights, 0.4f, 0.2f, 0.0f,
                  reason) == 0 && reason == walkability_nonfinite,
          "undersized height storage fails closed");

    heightfield null_heights;
    null_heights.nx = field.nx;
    null_heights.nz = field.nz;
    null_heights.origin_x = field.origin_x;
    null_heights.origin_z = field.origin_z;
    null_heights.cell_size = field.cell_size;
    null_heights.exterior_height = field.exterior_height;
    null_heights.version = field.version;
    null_heights.heights.size = field.heights.size;
    null_heights.heights.data = NULL;
    reason = walkability_clear;
    const walkability_sweep_result null_height_sweep = walkability_sweep(
        grid, null_heights, vec3(0.4f, 0.0f, 0.2f),
        vec3(0.5f, 0.0f, 0.2f), 0.0f);
    check(!walkability_grid_matches_heightfield(grid, null_heights) &&
              walkability_footprint_class(
                  grid, null_heights, 0.4f, 0.2f, 0.0f,
                  reason) == 0 && reason == walkability_nonfinite &&
              null_height_sweep.blocked &&
              null_height_sweep.encountered_class == 0 &&
              null_height_sweep.reason == walkability_nonfinite,
          "null height storage fails closed");

    heightfield invalid_axis(field);
    invalid_axis.origin_x = 1.0e20f;
    reason = walkability_clear;
    check(!walkability_grid_matches_heightfield(grid, invalid_axis) &&
              walkability_footprint_class(
                  grid, invalid_axis, invalid_axis.origin_x, 0.2f,
                  0.0f, reason) == 0 &&
              reason == walkability_nonfinite,
          "invalid v2 runtime axis fails closed");

    walkability_grid null_cells;
    null_cells.nx = grid.nx;
    null_cells.nz = grid.nz;
    null_cells.cells.size = grid.cells.size;
    null_cells.cells.data = NULL;
    reason = walkability_clear;
    check(!walkability_grid_matches_heightfield(null_cells, field) &&
              walkability_class_at(
                  null_cells, field, 0.4f, 0.2f) == 0 &&
              walkability_footprint_class(
                  null_cells, field, 0.4f, 0.2f, 0.0f,
                  reason) == 0 && reason == walkability_nonfinite,
          "null walkability storage fails closed");

    heightfield hostile_axis;
    hostile_axis.version = 2;
    hostile_axis.nx = 1073741823;
    hostile_axis.nz = 2;
    hostile_axis.origin_x = float_from_bits(UINT32_C(0x71800001));
    hostile_axis.origin_z = 0.0f;
    hostile_axis.cell_size = float_from_bits(UINT32_C(0x56800000));
    hostile_axis.exterior_height = 0.0f;
    walkability_grid hostile_grid;
    hostile_grid.nx = hostile_axis.nx;
    hostile_grid.nz = hostile_axis.nz;
    hostile_grid.cells.size = 2147483646;
    hostile_grid.cells.data =
        static_cast<uint8_t*>(malloc(sizeof(uint8_t)));
    check(hostile_grid.cells.data != NULL,
          "hostile walkability probe allocation");
    hostile_grid.cells.data[0] = 1;
    float hostile_x = 0.0f;
    check(walkability_node_coordinate(
              hostile_x, hostile_axis.origin_x, hostile_axis.cell_size,
              hostile_axis.nx - 1) &&
              float_bits(hostile_x) == UINT32_C(0x71800002),
          "hostile walkability endpoint reproduces rounded coordinate");
    reason = walkability_clear;
    check(walkability_footprint_class(
              hostile_grid, hostile_axis, hostile_x, 0.0f, 0.0f,
              reason) == 0 && reason == walkability_nonfinite,
          "hostile walkability dimension fails before integer conversion");
}

static void test_walkability_footprint_is_conservative_and_release_safe()
{
    heightfield field;
    walkability_grid grid;
    initialize_walkability_guard_fixture(field, grid);

    check(walkability_class_at(grid, field, 0.4f, 0.2f) == 2,
          "stress lookup remains diagnostic");
    check(walkability_class_at(grid, field, -0.01f, 0.2f) == 0,
          "exterior lookup remains blocked");

    walkability_reason reason = walkability_nonfinite;
    check(walkability_footprint_class(
              grid, field, 0.4f, 0.2f, 0.0f, reason) == 2 &&
              reason == walkability_clear,
          "stress footprint remains traversable");
    check(walkability_footprint_class(
              grid, field, 0.3f, 0.2f, 0.1f, reason) == 2 &&
              reason == walkability_clear,
          "circle contact records stress node");
    check(walkability_footprint_class(
              grid, field, 0.5f, 0.2f, 0.2f, reason) == 0 &&
              reason == walkability_blocked_cell,
          "circle boundary contact blocks at node");

    heightfield rounded_contact_field;
    initialize_heightfield(
        rounded_contact_field, 11, 11, 0.0f, 0.0f, 0.10f, 0.0f, 2);
    rounded_contact_field.heights.zero();
    walkability_grid rounded_contact_grid;
    rounded_contact_grid.nx = 11;
    rounded_contact_grid.nz = 11;
    rounded_contact_grid.cells.resize(121);
    rounded_contact_grid.cells.set(1);
    rounded_contact_grid.cells(5 * 11 + 5) = 0;
    check(walkability_footprint_class(
              rounded_contact_grid,
              rounded_contact_field,
              float_from_bits(UINT32_C(0x3f332c1a)),
              float_from_bits(UINT32_C(0x3efca135)),
              0.20f,
              reason) == 0 && reason == walkability_blocked_cell,
          "binary32 rounded circle boundary remains conservative");
    check(walkability_footprint_class(
              grid, field, 0.2f, 0.2f, 0.2f, reason) == 2 &&
              reason == walkability_clear,
          "footprint may touch the grid boundary");
    check(walkability_footprint_class(
              grid, field, nextafterf(0.2f, -INFINITY), 0.2f, 0.2f,
              reason) == 0 && reason == walkability_out_of_bounds,
          "footprint crossing the grid boundary blocks");

    walkability_grid malformed;
    malformed.nx = grid.nx;
    malformed.nz = grid.nz;
    malformed.cells.resize(1);
    malformed.cells(0) = 1;
    check(walkability_footprint_class(
              malformed, field, 0.2f, 0.2f, 0.2f, reason) == 0 &&
              reason == walkability_nonfinite,
          "malformed walkability shape stops safely");

    walkability_grid invalid_cell(grid);
    invalid_cell.cells(2 * invalid_cell.nx + 4) = 3;
    check(walkability_footprint_class(
              invalid_cell, field, 0.4f, 0.2f, 0.0f, reason) == 0 &&
              reason == walkability_nonfinite,
          "malformed walkability cell stops safely");

    heightfield wrong_version(field);
    wrong_version.version = 1;
    check(walkability_footprint_class(
              grid, wrong_version, 0.2f, 0.2f, 0.0f, reason) == 0 &&
              reason == walkability_nonfinite,
          "walkability guard accepts only v2 terrain");
    check(walkability_footprint_class(
              grid, field, 0.2f, 0.2f, -0.1f, reason) == 0 &&
              reason == walkability_nonfinite,
          "negative footprint radius stops safely");
    check(walkability_footprint_class(
              grid, field, 0.2f, 0.2f,
              std::numeric_limits<float>::quiet_NaN(), reason) == 0 &&
              reason == walkability_nonfinite,
          "nonfinite footprint radius stops safely");
}

static void test_walkability_footprint_caps_conservative_window()
{
    heightfield field;
    initialize_heightfield(
        field, 1025, 1025, 0.0f, 0.0f, 0.001f, 0.0f, 2);
    field.heights.zero();
    walkability_grid grid;
    grid.nx = field.nx;
    grid.nz = field.nz;
    grid.cells.resize(field.nx * field.nz);
    grid.cells.set(1);

    walkability_reason reason = walkability_clear;
    check(walkability_footprint_class(
              grid, field, 0.512f, 0.512f, 0.512f, reason) == 0 &&
              reason == walkability_nonfinite,
          "oversized conservative footprint window fails closed");
}

static void test_walkability_sweep_handles_clear_blocked_and_hostile_steps()
{
    heightfield field;
    walkability_grid grid;
    initialize_walkability_guard_fixture(field, grid);

    const vec3 start(0.2f, 8.0f, 0.2f);
    const vec3 stop(0.9f, -8.0f, 0.2f);
    const walkability_sweep_result blocked =
        walkability_sweep(grid, field, start, stop, 0.20f);
    check(blocked.blocked &&
              blocked.reason == walkability_blocked_cell,
          "blocked sweep reports blocked cell");
    check(blocked.safe_fraction >= 0.0f && blocked.safe_fraction < 1.0f &&
              blocked.distance >= 0.0f && blocked.point.y == 0.0f,
          "blocked sweep reports planar last-safe boundary");
    check(blocked.point.x >= start.x && blocked.point.x < 0.5f,
          "blocked sweep stops before circle-node contact");
    check(blocked.encountered_class == 2,
          "blocked sweep records traversed stress class");

    const walkability_sweep_result clear = walkability_sweep(
        grid, field, vec3(0.2f, 3.0f, 0.1f),
        vec3(0.3f, -4.0f, 0.1f), 0.0f);
    check(!clear.blocked && clear.reason == walkability_clear &&
              clear.safe_fraction == 1.0f && clear.distance == FLT_MAX,
          "clear sweep retains clear defaults");
    check_vec3_bits(clear.point, float_bits(0.3f), UINT32_C(0),
                    float_bits(0.1f), "clear sweep returns planar stop");

    const walkability_sweep_result zero = walkability_sweep(
        grid, field, vec3(0.3f, 2.0f, 0.1f),
        vec3(0.3f, -2.0f, 0.1f), 0.0f);
    check(!zero.blocked && zero.safe_fraction == 1.0f &&
              zero.distance == FLT_MAX,
          "zero-length clear sweep succeeds");
    const walkability_sweep_result zero_stress = walkability_sweep(
        grid, field, vec3(0.4f, 2.0f, 0.2f),
        vec3(0.4f, -2.0f, 0.2f), 0.0f);
    check(!zero_stress.blocked && zero_stress.encountered_class == 2,
          "zero-length stress sweep remains traversable");
    const walkability_sweep_result zero_blocked = walkability_sweep(
        grid, field, vec3(0.7f, 2.0f, 0.2f),
        vec3(0.7f, -2.0f, 0.2f), 0.0f);
    check(zero_blocked.blocked &&
              zero_blocked.reason == walkability_blocked_cell &&
              zero_blocked.safe_fraction == 0.0f &&
              zero_blocked.distance == 0.0f,
          "zero-length blocked sweep stops at start");

    const walkability_sweep_result outside = walkability_sweep(
        grid, field, vec3(-0.01f, 0.0f, 0.2f),
        vec3(0.3f, 0.0f, 0.2f), 0.0f);
    check(outside.blocked &&
              outside.reason == walkability_out_of_bounds &&
              outside.safe_fraction == 0.0f && outside.distance == 0.0f,
          "outside sweep start stops safely");
    const walkability_sweep_result nonfinite_start = walkability_sweep(
        grid, field,
        vec3(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.2f),
        vec3(0.3f, 0.0f, 0.2f), 0.0f);
    check(nonfinite_start.blocked &&
              nonfinite_start.reason == walkability_nonfinite &&
              nonfinite_start.safe_fraction == 0.0f &&
              nonfinite_start.distance == 0.0f,
          "nonfinite sweep start stops safely");
    const walkability_sweep_result nonfinite_stop = walkability_sweep(
        grid, field, vec3(0.2f, 0.0f, 0.2f),
        vec3(std::numeric_limits<float>::infinity(), 0.0f, 0.2f),
        0.0f);
    check(nonfinite_stop.blocked &&
              nonfinite_stop.reason == walkability_nonfinite &&
              nonfinite_stop.safe_fraction == 0.0f,
          "nonfinite sweep stop stops safely");
    const walkability_sweep_result hostile_count = walkability_sweep(
        grid, field, vec3(0.2f, 0.0f, 0.2f),
        vec3(1.0e30f, 0.0f, 0.2f), 0.0f);
    check(hostile_count.blocked &&
              hostile_count.reason == walkability_nonfinite &&
              hostile_count.safe_fraction == 0.0f &&
              hostile_count.distance == 0.0f,
          "unrepresentable sweep sample count stops without conversion UB");
    const walkability_sweep_result inexact_float_count = walkability_sweep(
        grid, field, vec3(0.2f, 0.0f, 0.2f),
        vec3(1.0e6f, 0.0f, 0.2f), 0.0f);
    check(inexact_float_count.blocked &&
              inexact_float_count.reason == walkability_nonfinite &&
              inexact_float_count.safe_fraction == 0.0f &&
              inexact_float_count.distance == 0.0f,
          "inexact binary32 sweep step counter stops safely");
}

static double walkability_test_maximum_sample_gap(
    vec3 start, vec3 stop, int steps)
{
    double maximum = 0.0;
    float previous_x = start.x;
    float previous_z = start.z;
    for (int step = 1; step <= steps; ++step) {
        float x = 0.0f;
        float z = 0.0f;
        check(terrain_f32_lerp(x, start.x, stop.x, step, steps) &&
                  terrain_f32_lerp(z, start.z, stop.z, step, steps),
              "exact spacing probe interpolation");
        const double dx =
            static_cast<double>(x) - static_cast<double>(previous_x);
        const double dz =
            static_cast<double>(z) - static_cast<double>(previous_z);
        const double gap = sqrt(dx * dx + dz * dz);
        if (gap > maximum) maximum = gap;
        previous_x = x;
        previous_z = z;
    }
    return maximum;
}

static void test_walkability_checked_conversion_and_exact_sample_spacing()
{
    int converted = 123;
    check(!walkability_checked_floor_to_int(
              converted, 2147483648.0, 0, INT_MAX),
          "out-of-range floor conversion fails without undefined behavior");

    heightfield field;
    initialize_heightfield(
        field, 210, 210, -0.25f, -0.25f,
        float_from_bits(UINT32_C(0x3b1efa48)), 0.0f, 2);
    field.heights.zero();
    walkability_grid grid;
    grid.nx = field.nx;
    grid.nz = field.nz;
    grid.cells.resize(field.nx * field.nz);
    grid.cells.set(1);
    const vec3 start(0.0f, 0.0f, 0.0f);
    const vec3 stop(
        float_from_bits(UINT32_C(0x3b6e7765)), 0.0f,
        float_from_bits(UINT32_C(0x36723088)));
    const double half_cell = 0.5 * static_cast<double>(field.cell_size);
    const double old_maximum_gap =
        walkability_test_maximum_sample_gap(start, stop, 3);
    check(old_maximum_gap > half_cell,
          "exact spacing vector reproduces former three-step gap");

    int steps = 0;
    check(walkability_sweep_step_count(
              steps, start, stop, field.cell_size) && steps >= 4,
          "exact spacing vector derives a conservative step count");
    check(walkability_test_maximum_sample_gap(start, stop, steps) <=
              half_cell,
          "actual rounded sweep samples stay within half a cell");
    const walkability_sweep_result sweep =
        walkability_sweep(grid, field, start, stop, 0.20f);
    check(!sweep.blocked && sweep.reason == walkability_clear,
          "exact spacing vector remains traversable");
}

static void check_invalid_traversability_command(
    const walkability_grid& grid,
    const heightfield& field,
    float scale,
    float scale_velocity,
    vec3 position,
    vec3 command,
    float dt,
    const char* message)
{
    traversability_diagnostics diagnostics = {};
    const vec3 applied = traversability_limit_command(
        scale, scale_velocity, diagnostics, grid, field,
        position, command, dt);
    check(diagnostics.blocked &&
              diagnostics.walkability_class == 0 &&
              diagnostics.reason == walkability_nonfinite &&
              diagnostics.distance == 0.0f &&
              applied.x == 0.0f && applied.y == 0.0f &&
              applied.z == 0.0f && scale == 0.0f &&
              scale_velocity == 0.0f,
          message);
}

static void test_traversability_command_limits_safely_and_recovers()
{
    heightfield field;
    walkability_grid grid;
    initialize_walkability_guard_fixture(field, grid);

    float horizon_scale = 1.0f;
    float horizon_scale_velocity = 0.0f;
    traversability_diagnostics horizon_diagnostics = {};
    traversability_limit_command(
        horizon_scale, horizon_scale_velocity, horizon_diagnostics,
        grid, field, vec3(0.249f, 0.0f, 0.2f),
        vec3(0.5f, 0.0f, 0.0f), 1.0f / 25.0f);
    check(!horizon_diagnostics.blocked,
          "half-second lookahead stays clear just before contact");
    horizon_scale = 1.0f;
    horizon_scale_velocity = 0.0f;
    traversability_limit_command(
        horizon_scale, horizon_scale_velocity, horizon_diagnostics,
        grid, field, vec3(0.251f, 0.0f, 0.2f),
        vec3(0.5f, 0.0f, 0.0f), 1.0f / 25.0f);
    check(horizon_diagnostics.blocked &&
              horizon_diagnostics.reason == walkability_blocked_cell,
          "half-second lookahead reaches contact just across horizon");

    float scale = 1.0f;
    float scale_velocity = 0.0f;
    traversability_diagnostics diagnostics = {};
    const vec3 first = traversability_limit_command(
        scale, scale_velocity, diagnostics, grid, field,
        vec3(0.3f, 0.0f, 0.2f), vec3(0.5f, 0.0f, 0.0f),
        1.0f / 25.0f);
    check_float_bits(
        traversability_blocked_reserve,
        float_bits(0.04f),
        "blocked command reserve is exactly four centimeters");
    const float expected_first_scale = clampf(
        (diagnostics.distance - traversability_blocked_reserve) /
            (diagnostics.commanded_speed * 0.50f),
        0.0f, 1.0f);
    check(diagnostics.blocked &&
              diagnostics.reason == walkability_blocked_cell &&
              diagnostics.walkability_class == 2 &&
              diagnostics.commanded_speed == 0.5f &&
              diagnostics.applied_speed > 0.0f &&
              diagnostics.applied_speed < diagnostics.commanded_speed &&
              scale > 0.0f && scale < 1.0f &&
              float_bits(scale) == float_bits(expected_first_scale) &&
              first.x == diagnostics.applied_speed && first.y == 0.0f &&
              first.z == 0.0f,
          "lookahead smoothly limits command before a block");
    const float first_scale = scale;
    traversability_limit_command(
        scale, scale_velocity, diagnostics, grid, field,
        vec3(0.3f, 0.0f, 0.2f), vec3(0.5f, 0.0f, 0.0f),
        1.0f / 25.0f);
    check(scale <= first_scale,
          "repeated blocked lookahead does not increase speed scale");

    walkability_grid clear_grid(grid);
    clear_grid.cells.set(1);
    float reference_scale = 0.5f;
    float reference_scale_velocity = 0.0f;
    traversability_diagnostics reference_diagnostics = {};
    const vec3 reference_applied = traversability_limit_command(
        reference_scale, reference_scale_velocity, reference_diagnostics,
        clear_grid, field, vec3(0.5f, 0.0f, 0.2f),
        vec3(0.5f, 0.0f, 0.0f), 1.0f / 25.0f);
    check(!reference_diagnostics.blocked &&
              float_bits(reference_scale) == UINT32_C(0x3f13be8f) &&
              float_bits(reference_scale_velocity) == UINT32_C(0x403ff497) &&
              float_bits(reference_applied.x) == UINT32_C(0x3e93be8f),
          "0.08-second half-life reference step is exact");
    const float reduced_scale = scale;
    for (int frame = 0; frame < 100; ++frame) {
        traversability_limit_command(
            scale, scale_velocity, diagnostics, clear_grid, field,
            vec3(0.5f, 0.0f, 0.2f), vec3(0.5f, 0.0f, 0.0f),
            1.0f / 25.0f);
        check(!diagnostics.blocked &&
                  diagnostics.reason == walkability_clear &&
                  diagnostics.walkability_class == 1,
              "clear command reports clear current path");
    }
    check(scale > reduced_scale && scale > 0.99f,
          "speed scale recovers on clear terrain");

    scale = 0.25f;
    scale_velocity = 0.0f;
    const vec3 stopped = traversability_limit_command(
        scale, scale_velocity, diagnostics, grid, field,
        vec3(0.7f, 0.0f, 0.2f), vec3(), 1.0f / 25.0f);
    check(diagnostics.blocked &&
              diagnostics.reason == walkability_blocked_cell &&
              diagnostics.commanded_speed == 0.0f &&
              diagnostics.applied_speed == 0.0f && scale == 0.0f &&
              stopped.x == 0.0f && stopped.y == 0.0f &&
              stopped.z == 0.0f,
          "zero command on a blocked footprint stays stopped");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    check_invalid_traversability_command(
        grid, field, nan, 0.0f, vec3(0.3f, 0.0f, 0.2f),
        vec3(0.5f, 0.0f, 0.0f), 1.0f / 25.0f,
        "nonfinite scale stops safely");
    check_invalid_traversability_command(
        grid, field, 1.0f, inf, vec3(0.3f, 0.0f, 0.2f),
        vec3(0.5f, 0.0f, 0.0f), 1.0f / 25.0f,
        "nonfinite scale velocity stops safely");
    check_invalid_traversability_command(
        grid, field, 1.0f, 0.0f, vec3(nan, 0.0f, 0.2f),
        vec3(0.5f, 0.0f, 0.0f), 1.0f / 25.0f,
        "nonfinite position x stops safely");
    check_invalid_traversability_command(
        grid, field, 1.0f, 0.0f, vec3(0.3f, inf, 0.2f),
        vec3(0.5f, 0.0f, 0.0f), 1.0f / 25.0f,
        "nonfinite position y stops safely");
    check_invalid_traversability_command(
        grid, field, 1.0f, 0.0f, vec3(0.3f, 0.0f, 0.2f),
        vec3(nan, 0.0f, 0.0f), 1.0f / 25.0f,
        "nonfinite command x stops safely");
    check_invalid_traversability_command(
        grid, field, 1.0f, 0.0f, vec3(0.3f, 0.0f, 0.2f),
        vec3(0.5f, nan, 0.0f), 1.0f / 25.0f,
        "nonfinite command y stops safely");
    check_invalid_traversability_command(
        grid, field, 1.0f, 0.0f, vec3(0.3f, 0.0f, 0.2f),
        vec3(0.5f, 0.0f, 0.0f), 0.0f,
        "zero command timestep stops safely");
    check_invalid_traversability_command(
        grid, field, 1.0f, 0.0f, vec3(0.3f, 0.0f, 0.2f),
        vec3(0.5f, 0.0f, 0.0f), nan,
        "nonfinite command timestep stops safely");

    walkability_grid malformed;
    malformed.nx = grid.nx;
    malformed.nz = grid.nz;
    malformed.cells.resize(1);
    malformed.cells(0) = 1;
    check_invalid_traversability_command(
        malformed, field, 1.0f, 0.0f, vec3(0.3f, 0.0f, 0.2f),
        vec3(0.5f, 0.0f, 0.0f), 1.0f / 25.0f,
        "malformed command terrain stops safely");
}

static void test_blocked_planar_stop_preserves_vertical_bits()
{
    traversability_diagnostics stopped = {};
    stopped.blocked = true;
    stopped.applied_speed = 1.0e-4f;
    vec3 velocity(
        float_from_bits(UINT32_C(0x3f800001)),
        float_from_bits(UINT32_C(0x7fc23456)),
        float_from_bits(UINT32_C(0xbf000001)));
    vec3 acceleration(
        float_from_bits(UINT32_C(0x40000001)),
        float_from_bits(UINT32_C(0x80000000)),
        float_from_bits(UINT32_C(0xc0400001)));
    const uint32_t velocity_y = float_bits(velocity.y);
    const uint32_t acceleration_y = float_bits(acceleration.y);

    traversability_stop_blocked_planar_dynamics(
        stopped, velocity, acceleration);
    check_vec3_bits(
        velocity, UINT32_C(0x00000000), velocity_y, UINT32_C(0x00000000),
        "blocked stopped velocity writes positive-zero XZ only");
    check_vec3_bits(
        acceleration, UINT32_C(0x00000000), acceleration_y,
        UINT32_C(0x00000000),
        "blocked stopped acceleration writes positive-zero XZ only");

    const vec3 original_velocity(
        float_from_bits(UINT32_C(0x3f123456)),
        float_from_bits(UINT32_C(0xffc34567)),
        float_from_bits(UINT32_C(0xbf234567)));
    const vec3 original_acceleration(
        float_from_bits(UINT32_C(0x40123456)),
        float_from_bits(UINT32_C(0x00000001)),
        float_from_bits(UINT32_C(0xc0123456)));

    traversability_diagnostics clear = {};
    clear.blocked = false;
    clear.applied_speed = 0.0f;
    velocity = original_velocity;
    acceleration = original_acceleration;
    traversability_stop_blocked_planar_dynamics(
        clear, velocity, acceleration);
    check_vec3_bits(
        velocity, float_bits(original_velocity.x),
        float_bits(original_velocity.y), float_bits(original_velocity.z),
        "clear diagnostics preserve every velocity bit");
    check_vec3_bits(
        acceleration, float_bits(original_acceleration.x),
        float_bits(original_acceleration.y),
        float_bits(original_acceleration.z),
        "clear diagnostics preserve every acceleration bit");

    traversability_diagnostics moving = {};
    moving.blocked = true;
    moving.applied_speed = nextafterf(1.0e-4f, INFINITY);
    velocity = original_velocity;
    acceleration = original_acceleration;
    traversability_stop_blocked_planar_dynamics(
        moving, velocity, acceleration);
    check_vec3_bits(
        velocity, float_bits(original_velocity.x),
        float_bits(original_velocity.y), float_bits(original_velocity.z),
        "above-threshold blocked diagnostics preserve velocity bits");
    check_vec3_bits(
        acceleration, float_bits(original_acceleration.x),
        float_bits(original_acceleration.y),
        float_bits(original_acceleration.z),
        "above-threshold blocked diagnostics preserve acceleration bits");

    traversability_diagnostics invalid = {};
    invalid.blocked = true;
    invalid.applied_speed = float_from_bits(UINT32_C(0x7fc23456));
    velocity = original_velocity;
    acceleration = original_acceleration;
    traversability_stop_blocked_planar_dynamics(
        invalid, velocity, acceleration);
    check_vec3_bits(
        velocity, float_bits(original_velocity.x),
        float_bits(original_velocity.y), float_bits(original_velocity.z),
        "nonfinite blocked diagnostics preserve velocity bits");
    check_vec3_bits(
        acceleration, float_bits(original_acceleration.x),
        float_bits(original_acceleration.y),
        float_bits(original_acceleration.z),
        "nonfinite blocked diagnostics preserve acceleration bits");
}

static void test_traversability_clip_is_planar_and_bit_preserving()
{
    heightfield field;
    walkability_grid grid;
    initialize_walkability_guard_fixture(field, grid);

    const vec3 start(0.3f, float_from_bits(UINT32_C(0x3f000001)), 0.2f);
    vec3 candidate(0.9f, float_from_bits(UINT32_C(0x80000000)), 0.2f);
    vec3 velocity(0.5f, float_from_bits(UINT32_C(0x3f800001)), -0.25f);
    vec3 acceleration(0.75f, float_from_bits(UINT32_C(0x80000000)), 0.5f);
    float support = float_from_bits(UINT32_C(0x7fc54321));
    const uint32_t candidate_y = float_bits(candidate.y);
    const uint32_t velocity_y = float_bits(velocity.y);
    const uint32_t acceleration_y = float_bits(acceleration.y);
    const uint32_t support_bits = float_bits(support);
    const walkability_sweep_result expected = traversability_preflight_step(
        start, candidate, velocity, acceleration,
        grid, field, 0.20f);
    vec3 applied_candidate = candidate;
    vec3 applied_velocity = velocity;
    vec3 applied_acceleration = acceleration;
    traversability_diagnostics applied_diagnostics = {};
    applied_diagnostics.walkability_class = 1;
    check(!traversability_apply_sweep_result(
              start, applied_candidate, applied_velocity,
              applied_acceleration, applied_diagnostics, expected) &&
              applied_candidate.x == expected.point.x &&
              applied_candidate.z == expected.point.z &&
              applied_velocity.x == 0.0f &&
              applied_velocity.z == 0.0f &&
              applied_acceleration.x == 0.0f &&
              applied_acceleration.z == 0.0f &&
              applied_diagnostics.walkability_class ==
                  expected.encountered_class,
          "preflight result applies without a second sweep");
    traversability_diagnostics diagnostics = {};
    diagnostics.walkability_class = 1;
    check(!traversability_clip_step(
              start, candidate, velocity, acceleration, diagnostics,
              grid, field, 0.20f),
          "blocked integration step hard clips");
    check_float_bits(candidate.y, candidate_y,
                     "hard clip preserves candidate y bits");
    check_float_bits(velocity.y, velocity_y,
                     "hard clip preserves velocity y bits");
    check_float_bits(acceleration.y, acceleration_y,
                     "hard clip preserves acceleration y bits");
    check_float_bits(support, support_bits,
                     "hard clip leaves support bits untouched");
    check(candidate.x == expected.point.x &&
              candidate.z == expected.point.z &&
              velocity.x == 0.0f && velocity.z == 0.0f &&
              acceleration.x == 0.0f && acceleration.z == 0.0f,
          "hard clip applies the last safe XZ fraction only");
    check(diagnostics.blocked &&
              diagnostics.reason == walkability_blocked_cell &&
              diagnostics.walkability_class == expected.encountered_class &&
              diagnostics.distance == expected.distance &&
              diagnostics.point.x == expected.point.x &&
              diagnostics.point.z == expected.point.z &&
              diagnostics.applied_speed == 0.0f,
          "hard clip updates prospective diagnostics");
    walkability_reason current_reason = walkability_nonfinite;
    check(walkability_footprint_class(
              grid, field, candidate.x, candidate.z, 0.20f,
              current_reason) != 0 && current_reason == walkability_clear,
          "hard-clipped footprint is accepted");

    vec3 clear_candidate(0.35f, float_from_bits(UINT32_C(0x80000000)),
                         0.2f);
    vec3 clear_velocity(0.25f, 7.0f, -0.5f);
    vec3 clear_acceleration(-0.5f, -8.0f, 0.25f);
    const vec3 clear_candidate_before = clear_candidate;
    const vec3 clear_velocity_before = clear_velocity;
    const vec3 clear_acceleration_before = clear_acceleration;
    diagnostics = traversability_diagnostics();
    check(traversability_clip_step(
              start, clear_candidate, clear_velocity, clear_acceleration,
              diagnostics, grid, field, 0.20f),
          "clear integration step is accepted");
    check_vec3_bits(clear_candidate,
                    float_bits(clear_candidate_before.x),
                    float_bits(clear_candidate_before.y),
                    float_bits(clear_candidate_before.z),
                    "clear clip preserves candidate");
    check_vec3_bits(clear_velocity,
                    float_bits(clear_velocity_before.x),
                    float_bits(clear_velocity_before.y),
                    float_bits(clear_velocity_before.z),
                    "clear clip preserves velocity");
    check_vec3_bits(clear_acceleration,
                    float_bits(clear_acceleration_before.x),
                    float_bits(clear_acceleration_before.y),
                    float_bits(clear_acceleration_before.z),
                    "clear clip preserves acceleration");

    vec3 invalid_candidate(
        std::numeric_limits<float>::quiet_NaN(),
        float_from_bits(UINT32_C(0x7fc23456)), 0.2f);
    velocity = vec3(0.25f, 4.0f, 0.5f);
    acceleration = vec3(0.5f, 5.0f, 0.75f);
    const uint32_t invalid_y = float_bits(invalid_candidate.y);
    diagnostics = traversability_diagnostics();
    diagnostics.walkability_class = 2;
    check(!traversability_clip_step(
              start, invalid_candidate, velocity, acceleration,
              diagnostics, grid, field, 0.20f) &&
              diagnostics.reason == walkability_nonfinite &&
              diagnostics.walkability_class == 0,
          "nonfinite candidate hard stops");
    check(invalid_candidate.x == start.x &&
              invalid_candidate.z == start.z &&
              velocity.x == 0.0f && velocity.z == 0.0f &&
              acceleration.x == 0.0f && acceleration.z == 0.0f,
          "nonfinite candidate cannot leak through zero fraction");
    check_float_bits(invalid_candidate.y, invalid_y,
                     "nonfinite hard stop preserves candidate y bits");

    vec3 finite_candidate(0.35f, -3.0f, 0.2f);
    velocity = vec3(std::numeric_limits<float>::infinity(), 4.0f, 0.5f);
    acceleration = vec3(0.5f, 5.0f, 0.75f);
    diagnostics = traversability_diagnostics();
    check(!traversability_clip_step(
              start, finite_candidate, velocity, acceleration,
              diagnostics, grid, field, 0.20f) &&
              diagnostics.reason == walkability_nonfinite &&
              finite_candidate.x == start.x &&
              finite_candidate.z == start.z,
          "nonfinite planar simulation state hard stops");

    vec3 vertical_candidate(
        0.35f, float_from_bits(UINT32_C(0x7fc34567)), 0.2f);
    velocity = vec3(0.25f, 4.0f, 0.5f);
    acceleration = vec3(0.5f, 5.0f, 0.75f);
    const uint32_t vertical_candidate_y = float_bits(vertical_candidate.y);
    diagnostics = traversability_diagnostics();
    check(!traversability_clip_step(
              start, vertical_candidate, velocity, acceleration,
              diagnostics, grid, field, 0.20f) &&
              diagnostics.reason == walkability_nonfinite &&
              vertical_candidate.x == start.x &&
              vertical_candidate.z == start.z,
          "nonfinite candidate y hard stops planar motion");
    check_float_bits(vertical_candidate.y, vertical_candidate_y,
                     "nonfinite hard stop preserves candidate y payload");

    vertical_candidate = vec3(0.35f, -3.0f, 0.2f);
    velocity = vec3(
        0.25f, std::numeric_limits<float>::infinity(), 0.5f);
    acceleration = vec3(0.5f, 5.0f, 0.75f);
    diagnostics = traversability_diagnostics();
    const uint32_t vertical_velocity_y = float_bits(velocity.y);
    check(!traversability_clip_step(
              start, vertical_candidate, velocity, acceleration,
              diagnostics, grid, field, 0.20f) &&
              diagnostics.reason == walkability_nonfinite &&
              float_bits(velocity.y) == vertical_velocity_y,
          "nonfinite velocity y hard stops without rewriting y");

    vertical_candidate = vec3(0.35f, -3.0f, 0.2f);
    velocity = vec3(0.25f, 4.0f, 0.5f);
    acceleration = vec3(
        0.5f, float_from_bits(UINT32_C(0x7fc45678)), 0.75f);
    const uint32_t vertical_acceleration_y = float_bits(acceleration.y);
    diagnostics = traversability_diagnostics();
    check(!traversability_clip_step(
              start, vertical_candidate, velocity, acceleration,
              diagnostics, grid, field, 0.20f) &&
              diagnostics.reason == walkability_nonfinite,
          "nonfinite acceleration y hard stops planar motion");
    check_float_bits(acceleration.y, vertical_acceleration_y,
                     "nonfinite hard stop preserves acceleration y payload");
}

static void test_walkability_guard_reaches_safe_stop()
{
    heightfield field;
    walkability_grid grid;
    initialize_walkability_guard_fixture(field, grid);
    float scale = 1.0f;
    float scale_velocity = 0.0f;
    vec3 position(0.2f, 0.0f, 0.2f);
    vec3 velocity(0.5f, 11.0f, 0.0f);
    vec3 acceleration(0.0f, -12.0f, 0.0f);
    float previous_speed = 0.5f;
    bool saw_block = false;
    for (int frame = 0; frame < 100; ++frame) {
        traversability_diagnostics diagnostics = {};
        const vec3 applied = traversability_limit_command(
            scale, scale_velocity, diagnostics, grid, field, position,
            vec3(0.5f, 0.0f, 0.0f), 1.0f / 25.0f);
        check(walkability_xz_length(applied) <= previous_speed + 1e-6f ||
                  !diagnostics.blocked,
              "speed reduces near block");
        previous_speed = walkability_xz_length(applied);
        vec3 candidate = position + applied * (1.0f / 25.0f);
        if (!traversability_clip_step(
                position, candidate, velocity, acceleration,
                diagnostics, grid, field, 0.20f)) {
            saw_block = true;
        }
        position = candidate;
        check(walkability_class_at(grid, field, position.x, position.z) != 0,
              "guard center never enters blocked node");
    }
    check(saw_block || previous_speed < 0.01f,
          "guard reaches a safe stop");
    check(position.x < 0.50f,
          "guard stops footprint before blocked x=0.7 node");
}

static std::vector<char> read_controller_source()
{
    std::string adjacent_path = __FILE__;
    const std::string test_suffix =
        "tests/cpp/test_terrain_runtime.cpp";
    const size_t suffix_position = adjacent_path.rfind(test_suffix);
    if (suffix_position != std::string::npos) {
        adjacent_path.erase(suffix_position);
        adjacent_path += "controller.cpp";
    }
    const char* paths[] = {
        "controller.cpp",
        "../../controller.cpp",
        adjacent_path.c_str()
    };
    FILE* file = NULL;
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); ++i) {
        file = fopen(paths[i], "rb");
        if (file != NULL) break;
    }
    if (file == NULL) {
        check(false, "controller source is available to data-flow regression");
        return std::vector<char>(1, '\0');
    }
    check(fseek(file, 0, SEEK_END) == 0,
          "controller source size seek");
    const long size = ftell(file);
    check(size >= 0, "controller source size");
    check(fseek(file, 0, SEEK_SET) == 0,
          "controller source rewind");
    std::vector<char> source(static_cast<size_t>(size) + 1u, '\0');
    check(size == 0 ||
              fread(source.data(), 1, static_cast<size_t>(size), file) ==
                  static_cast<size_t>(size),
          "controller source read");
    check(fclose(file) == 0, "controller source close");
    return source;
}

static const char* require_source_token(
    const char* begin, const char* token, const char* message)
{
    const char* found = strstr(begin, token);
    check(found != NULL, message);
    return found;
}

static size_t source_token_count(const char* begin, const char* token)
{
    size_t count = 0;
    const size_t length = strlen(token);
    while ((begin = strstr(begin, token)) != NULL) {
        ++count;
        begin += length;
    }
    return count;
}

static std::string compact_source_segment(const std::string& input)
{
    std::string output;
    output.reserve(input.size());
    for (const char value : input) {
        if (value != ' ' && value != '\t' &&
            value != '\r' && value != '\n') {
            output.push_back(value);
        }
    }
    return output;
}

static void test_controller_task6_terrain_ik_pipeline()
{
    const std::vector<char> source = read_controller_source();
    const char* const source_begin = source.data();
    const char* runner = require_source_token(
        source_begin,
        "G1FrameStageOutcome g1_controller_frame_stage_run(",
        "Task6 named stage runner must own the terrain and IK pipeline");
    const char* signature_close = require_source_token(
        runner, ")", "Task6 runner signature closes");
    const char* body_open = require_source_token(
        signature_close, "{", "Task6 runner body opens");
    int depth = 0;
    const char* body_close = NULL;
    for (const char* cursor = body_open; *cursor != '\0'; ++cursor) {
        if (*cursor == '{') {
            ++depth;
        } else if (*cursor == '}' && --depth == 0) {
            body_close = cursor;
            break;
        }
    }
    check(body_close != NULL, "Task6 runner body closes");
    const std::string body_raw(
        body_open, static_cast<size_t>(body_close - body_open + 1));
    const std::string body = compact_source_segment(body_raw);

    const size_t footprint_stage = body.find(
        "caseG1FrameStageFootprintObservation:");
    const size_t foot0_stage = body.find("caseG1FrameStageFirstFootIk:");
    const size_t foot1_stage = body.find("caseG1FrameStageSecondFootIk:");
    const size_t final_fk_stage = body.find("caseG1FrameStageFinalFk:");
    const size_t pose_stage = body.find("caseG1FrameStagePoseCertificate:");
    check(footprint_stage != std::string::npos &&
              foot0_stage != std::string::npos &&
              foot1_stage != std::string::npos &&
              final_fk_stage != std::string::npos &&
              pose_stage != std::string::npos &&
              footprint_stage < foot0_stage && foot0_stage < foot1_stage &&
              foot1_stage < final_fk_stage && final_fk_stage < pose_stage,
          "footprint, split-foot IK, final FK, and pose certification stages are ordered");

    const std::string footprint = body.substr(
        footprint_stage, foot0_stage - footprint_stage);
    const size_t schedule = footprint.find(
        "g1_foot_contact_schedule_build(");
    const size_t observe = footprint.find("g1_footprint_observe_v2(");
    const size_t begin = footprint.find("g1_ik_frame_begin(");
    const size_t begin_stop = footprint.find(
        "candidate_result.safe_stop_requested", begin);
    const size_t begin_snapshot = footprint.find(
        "g1_ik_frame_rejection_snapshot(", begin_stop);
    check(schedule != std::string::npos && observe != std::string::npos &&
              begin != std::string::npos && begin_stop != std::string::npos &&
              begin_snapshot != std::string::npos &&
              schedule < observe && observe < begin &&
              begin < begin_stop && begin_stop < begin_snapshot,
          "contact horizons and footprint precede checked begin-time IK safe-stop inspection");
    check(source_token_count(
              footprint.c_str(), "g1_foot_contact_schedule_build(") == 1 &&
          source_token_count(
              footprint.c_str(), "g1_footprint_observe_v2(") == 1 &&
          source_token_count(footprint.c_str(), "g1_ik_frame_begin(") == 1,
          "footprint stage owns one schedule, observation, and IK begin call");
    check(footprint.find("state.command") != std::string::npos &&
          footprint.find("external.scene->terrain") != std::string::npos &&
          footprint.find("external.scene->walkability") !=
              std::string::npos &&
          footprint.find("desired_rotation =") == std::string::npos,
          "physical footprint consumes immutable command/scene data without rewriting heading");

    const std::string foot0 = body.substr(
        foot0_stage, foot1_stage - foot0_stage);
    const std::string foot1 = body.substr(
        foot1_stage, final_fk_stage - foot1_stage);
    check(source_token_count(foot0.c_str(), "g1_ik_frame_stage_foot(") == 1 &&
          foot0.find(",0,") != std::string::npos &&
          foot0.find("G1IkRejectionAfterFoot0") != std::string::npos,
          "first-foot stage runs only foot zero and snapshots its real rejection checkpoint");
    check(source_token_count(foot1.c_str(), "g1_ik_frame_stage_foot(") == 1 &&
          foot1.find(",1,") != std::string::npos &&
          foot1.find("G1IkRejectionAfterFoot1") != std::string::npos,
          "second-foot stage runs only foot one and snapshots its real rejection checkpoint");

    const std::string final_fk = body.substr(
        final_fk_stage, pose_stage - final_fk_stage);
    check(source_token_count(final_fk.c_str(), "g1_ik_frame_finish(") == 1 &&
          final_fk.find("state.ik_candidate_bone_positions") !=
              std::string::npos &&
          final_fk.find("state.ik_candidate_bone_rotations") !=
              std::string::npos,
          "final-FK stage finishes the split transaction from candidate pose owners");

    const std::string pose = body.substr(pose_stage);
    const size_t clearance_budget = pose.find("g1_pose_clearance_budget(");
    const size_t clearance = pose.find("g1_measure_pose_clearance(");
    check(clearance_budget != std::string::npos &&
              clearance != std::string::npos &&
              clearance_budget < clearance &&
              source_token_count(pose.c_str(),
                  "g1_measure_pose_clearance(") == 1,
          "pose stage certifies the final split-IK pose exactly once");
    check(pose.find("state.footprint=scratch.footprint;") !=
              std::string::npos &&
          pose.find("state.footprint_status=G1FootprintOk;") !=
              std::string::npos &&
          pose.find("scratch.accepted_diagnostic_candidate") !=
              std::string::npos,
          "only the fully certified working frame stages accepted footprint and diagnostics");

    check(body.find("g1_ik_frame_evaluate(") == std::string::npos &&
          body.find("ik_two_bone(") == std::string::npos &&
          body.find("ik_look_at(") == std::string::npos &&
          body.find("if constexpr (ik_enabled)") == std::string::npos,
          "controller runner contains no one-shot or legacy pose-IK authority");
}

static void test_f32_helpers_match_one_round_producer_operations()
{
    float value = -7.0f;
    check(terrain_f32_add(value, 0.1f, 0.2f), "binary32 add accepted");
    check_float_bits(value, UINT32_C(0x3e99999a), "binary32 add result");
    check(terrain_f32_sub(value, 1.0f, 0.1f), "binary32 subtract accepted");
    check_float_bits(value, UINT32_C(0x3f666666), "binary32 subtract result");
    check(terrain_f32_div(value, 1.0f, 3.0f), "binary32 divide accepted");
    check_float_bits(value, UINT32_C(0x3eaaaaab), "binary32 divide result");
    check(terrain_f32_sqrt(value, 2.0f), "binary32 square root accepted");
    check_float_bits(value, UINT32_C(0x3fb504f3),
                     "binary32 square root result");

    float product = -1.0f;
    check(terrain_f32_mul(
              product,
              float_from_bits(UINT32_C(0x4f800001)),
              float_from_bits(UINT32_C(0x4f7ffffe))),
          "binary32 multiply accepted");
    check_float_bits(product, UINT32_C(0x5f800000),
                     "binary32 multiply materializes before add");
    float sum = -1.0f;
    check(terrain_f32_add(
              sum, product, float_from_bits(UINT32_C(0xdf800000))),
          "binary32 post-multiply add accepted");
    check_float_bits(sum, UINT32_C(0x00000000),
                     "binary32 multiply and add do not contract");

    value = 17.0f;
    check(!terrain_f32_div(value, 1.0f, 0.0f),
          "binary32 divide rejects infinity");
    check_float_bits(value, UINT32_C(0x41880000),
                     "binary32 rejection preserves destination");
}

static void test_sidecar_loads_valid_file()
{
    const char* path = "/tmp/test_g1tf_valid.bin";
    const byte_buffer payload = make_sidecar(
        1, 2, 4, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    write_payload(path, payload);

    terrain_feature_set features;
    features.values.resize(1, 8);
    features.values.set(-99.0f);
    char error[256] = {};
    check(terrain_features_load(
              features, path, error, static_cast<int>(sizeof(error))),
          "valid sidecar loads");
    check(features.values.rows == 2, "valid sidecar rows");
    check(features.values.cols == 4, "valid sidecar columns");
    for (int i = 0; i < 8; ++i) {
        check(features.values.data[i] == static_cast<float>(i),
              "valid sidecar values");
    }
}

static void test_sidecar_rejects_every_truncation()
{
    const char* path = "/tmp/test_g1tf_truncated.bin";
    const byte_buffer payload = make_sidecar(
        1, 2, 4, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    for (size_t length = 0; length < payload.size(); ++length) {
        write_prefix(path, payload, length);
        expect_sidecar_rejected(path, "truncated");
    }
}

static void test_sidecar_rejects_invalid_schema_sizes_and_values()
{
    const char* path = "/tmp/test_g1tf_corrupt.bin";
    const std::vector<float> valid_values = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

    byte_buffer payload = make_sidecar(1, 2, 4, valid_values);
    payload[0] = 'B';
    write_payload(path, payload);
    expect_sidecar_rejected(path, "magic");

    write_payload(path, make_sidecar(2, 2, 4, valid_values));
    expect_sidecar_rejected(path, "version");

    write_payload(path, make_sidecar(1, 2, 3, valid_values));
    expect_sidecar_rejected(path, "dimension");

    write_payload(path, make_sidecar(1, 0, 4, {}));
    expect_sidecar_rejected(path, "frame count");

    const uint32_t first_unsafe_frame_count =
        static_cast<uint32_t>(INT_MAX / 4) + 1u;
    write_payload(path, make_sidecar(1, first_unsafe_frame_count, 4, {}));
    expect_sidecar_rejected(path, "frame count");

    const uint32_t largest_safe_frame_count =
        static_cast<uint32_t>(INT_MAX / 4);
    write_payload(path, make_sidecar(1, largest_safe_frame_count, 4, {}));
    expect_sidecar_rejected(path, "truncated");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    write_payload(path, make_sidecar(1, 1, 4, {0.0f, nan, 2.0f, 3.0f}));
    expect_sidecar_rejected(path, "finite");

    const float infinity = std::numeric_limits<float>::infinity();
    write_payload(
        path, make_sidecar(1, 1, 4, {0.0f, 1.0f, -infinity, 3.0f}));
    expect_sidecar_rejected(path, "finite");

    payload = make_sidecar(1, 2, 4, valid_values);
    payload.push_back(0x7f);
    write_payload(path, payload);
    expect_sidecar_rejected(path, "trailing");
}

static void test_sidecar_open_failure_is_actionable_and_transactional()
{
    const char* path = "/tmp/test_g1tf_does_not_exist.bin";
    remove(path);
    expect_sidecar_rejected(path, "cannot open");
}

static byte_buffer valid_heightfield_payload(uint32_t version = 1)
{
    return make_heightfield(
        version, 2, 2, 0.0f, 0.0f, 1.0f, -1.0f,
        {0.0f, 1.0f, 2.0f, 3.0f});
}

static void test_heightfield_loads_and_samples_valid_file()
{
    const char* path = "/tmp/test_g1hf_valid.bin";
    write_payload(path, valid_heightfield_payload());

    heightfield field;
    field.version = 88;
    field.nx = 4;
    field.nz = 1;
    field.origin_x = 99.0f;
    field.origin_z = 98.0f;
    field.cell_size = 97.0f;
    field.exterior_height = 96.0f;
    field.heights.resize(4);
    field.heights.set(-99.0f);
    char error[256] = {};
    check(heightfield_load(
              field, path, error, static_cast<int>(sizeof(error))),
          "valid v1 heightfield loads");
    check(field.version == 1, "valid v1 heightfield version");
    check(field.nx == 2, "valid heightfield nx");
    check(field.nz == 2, "valid heightfield nz");
    check(field.origin_x == 0.0f, "valid heightfield origin x");
    check(field.origin_z == 0.0f, "valid heightfield origin z");
    check(field.cell_size == 1.0f, "valid heightfield cell size");
    check(field.exterior_height == -1.0f,
          "valid heightfield exterior height");
    check(field.heights.size == 4, "valid heightfield storage size");

    check(heightfield_sample(field, 0.0f, 0.0f) == 0.0f,
          "v1 sample lower-left node");
    check(heightfield_sample(field, 1.0f, 0.0f) == 1.0f,
          "v1 sample lower-right node");
    check(heightfield_sample(field, 0.0f, 1.0f) == 2.0f,
          "v1 sample upper-left node");
    check(heightfield_sample(field, 1.0f, 1.0f) == 3.0f,
          "v1 sample upper-right node");
    check(fabsf(heightfield_sample(field, 0.5f, 0.5f) - 1.5f) < 1e-6f,
          "v1 sample cell center");
    check(fabsf(heightfield_sample(field, 1.0f, 0.5f) - 2.0f) < 1e-6f,
          "v1 sample maximum x edge");

    check(heightfield_sample(field, -0.001f, 0.0f) == -1.0f,
          "v1 sample negative x exterior");
    check(heightfield_sample(field, 0.0f, -0.001f) == -1.0f,
          "v1 sample negative z exterior");
    check(heightfield_sample(field, 1.001f, 0.0f) == -1.0f,
          "v1 sample positive x exterior");
    check(heightfield_sample(field, 0.0f, 1.001f) == -1.0f,
          "v1 sample positive z exterior");
    check(heightfield_sample(field, FLT_MAX, 0.0f) == -1.0f,
          "v1 sample extreme exterior");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    check(heightfield_sample(field, nan, 0.0f) == -1.0f,
          "v1 sample NaN x exterior");
    check(heightfield_sample(field, 0.0f, nan) == -1.0f,
          "v1 sample NaN z exterior");
    check(heightfield_sample(field, infinity, 0.0f) == -1.0f,
          "v1 sample infinite x exterior");
    check(heightfield_sample(field, 0.0f, -infinity) == -1.0f,
          "v1 sample infinite z exterior");
}

static void test_heightfield_rejects_every_truncation()
{
    const char* path = "/tmp/test_g1hf_truncated.bin";
    for (uint32_t version = 1; version <= 2; ++version) {
        const byte_buffer payload = valid_heightfield_payload(version);
        for (size_t length = 0; length < payload.size(); ++length) {
            write_prefix(path, payload, length);
            expect_heightfield_rejected(path, "truncated");
        }
    }
}

static void test_heightfield_rejects_invalid_schema_and_sizes()
{
    const char* path = "/tmp/test_g1hf_schema.bin";
    const std::vector<float> heights = {0.0f, 1.0f, 2.0f, 3.0f};

    byte_buffer payload = valid_heightfield_payload();
    payload[0] = 'B';
    write_payload(path, payload);
    expect_heightfield_rejected(path, "magic");

    write_payload(
        path, make_heightfield(3, 2, 2, 0, 0, 1, -1, heights));
    expect_heightfield_rejected(path, "version");

    for (uint32_t version = 1; version <= 2; ++version) {
        write_payload(
            path, make_heightfield(version, 1, 2, 0, 0, 1, -1, {}));
        expect_heightfield_rejected(path, "grid dimensions");

        write_payload(
            path, make_heightfield(version, 2, 1, 0, 0, 1, -1, {}));
        expect_heightfield_rejected(path, "grid dimensions");

        write_payload(
            path, make_heightfield(version, 0, 2, 0, 0, 1, -1, {}));
        expect_heightfield_rejected(path, "grid dimensions");

        write_payload(
            path, make_heightfield(version, 2, 0, 0, 0, 1, -1, {}));
        expect_heightfield_rejected(path, "grid dimensions");

        write_payload(path, make_heightfield(
            version, 65536, 32768, 0, 0, 1, -1, {}));
        expect_heightfield_rejected(path, "grid dimensions");

        write_payload(path, make_heightfield(
            version, UINT32_MAX, 2, 0, 0, 1, -1, {}));
        expect_heightfield_rejected(path, "grid dimensions");

        write_payload(path, make_heightfield(
            version, 65535, 32768, 0, 0, 1, -1, {}));
        expect_heightfield_rejected(path, "truncated");

        payload = valid_heightfield_payload(version);
        payload.push_back(0x7f);
        write_payload(path, payload);
        expect_heightfield_rejected(path, "trailing");
    }
}

static void test_heightfield_rejects_nonfinite_metadata_and_heights()
{
    const char* path = "/tmp/test_g1hf_numeric.bin";
    const std::vector<float> heights = {0.0f, 1.0f, 2.0f, 3.0f};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();

    for (uint32_t version = 1; version <= 2; ++version) {
        write_payload(path, make_heightfield(
            version, 2, 2, nan, 0, 1, -1, heights));
        expect_heightfield_rejected(path, "origin");

        write_payload(path, make_heightfield(
            version, 2, 2, 0, infinity, 1, -1, heights));
        expect_heightfield_rejected(path, "origin");

        write_payload(path, make_heightfield(
            version, 2, 2, 0, 0, 0, -1, heights));
        expect_heightfield_rejected(path, "cell size");

        write_payload(path, make_heightfield(
            version, 2, 2, 0, 0, -1, -1, heights));
        expect_heightfield_rejected(path, "cell size");

        write_payload(path, make_heightfield(
            version, 2, 2, 0, 0, nan, -1, heights));
        expect_heightfield_rejected(path, "cell size");

        write_payload(path, make_heightfield(
            version, 2, 2, 0, 0, infinity, -1, heights));
        expect_heightfield_rejected(path, "cell size");

        write_payload(path, make_heightfield(
            version, 2, 2, 0, 0, 1, nan, heights));
        expect_heightfield_rejected(path, "exterior height");

        write_payload(path, make_heightfield(
            version, 2, 2, 0, 0, 1, infinity, heights));
        expect_heightfield_rejected(path, "exterior height");

        write_payload(path, make_heightfield(
            version, 2, 2, 0, 0, 1, -1,
            {0.0f, nan, 2.0f, 3.0f}));
        expect_heightfield_rejected(path, "finite");

        write_payload(path, make_heightfield(
            version, 2, 2, 0, 0, 1, -1,
            {0.0f, 1.0f, -infinity, 3.0f}));
        expect_heightfield_rejected(path, "finite");
    }
}

static void test_heightfield_open_failure_is_actionable_and_transactional()
{
    const char* path = "/tmp/test_g1hf_does_not_exist.bin";
    remove(path);
    expect_heightfield_rejected(path, "cannot open");
}

static heightfield load_heightfield_fixture(
    const char* path, const byte_buffer& payload)
{
    write_payload(path, payload);
    heightfield field;
    char error[256] = {};
    const bool loaded = heightfield_load(
        field, path, error, static_cast<int>(sizeof(error)));
    check(loaded, error[0] != '\0' ? error : "heightfield fixture load");
    return field;
}

static void test_heightfield_versions_preserve_v1_and_use_v2_triangles()
{
    const std::vector<float> heights = {0.0f, 0.0f, 0.0f, 1.0f};
    const heightfield v1 = load_heightfield_fixture(
        "/tmp/test_g1hf_v1_asymmetric.bin",
        make_heightfield(1, 2, 2, 0.0f, 0.0f, 1.0f, -9.0f, heights));
    const heightfield v2 = load_heightfield_fixture(
        "/tmp/test_g1hf_v2_asymmetric.bin",
        make_heightfield(2, 2, 2, 0.0f, 0.0f, 1.0f, -9.0f, heights));

    check(v1.version == 1, "v1 fixture reports version one");
    check(v2.version == 2, "v2 fixture reports version two");
    check_float_bits(
        heightfield_sample(v1, 0.75f, 0.25f), UINT32_C(0x3e400000),
        "v1 fixture retains bilinear interpolation");
    check_float_bits(
        heightfield_sample_versioned(v2, 0.75f, 0.25f),
        UINT32_C(0x3e800000),
        "v2 fixture uses fixed first triangle");
    check_vec3_bits(
        heightfield_normal(v1, 0.75f, 0.25f),
        UINT32_C(0x00000000), UINT32_C(0x3f800000),
        UINT32_C(0x00000000),
        "v1 normal is the exact migration fallback");
}

static void test_v1_coordinate_arithmetic_remains_literal()
{
    const float origin = -36257.83203125f;
    const float cell = 0.04736527055501938f;
    const float query_x = -32593.607421875f;
    const float legacy_grid_x = (query_x - origin) / cell;
    const int legacy_x0 = static_cast<int>(floorf(legacy_grid_x));
    check_float_bits(
        legacy_grid_x, UINT32_C(0x47971880),
        "v1 arithmetic regression grid coordinate");
    check(legacy_x0 == 77361, "v1 arithmetic regression cell");

    heightfield field;
    initialize_heightfield(
        field, legacy_x0 + 2, 2, origin, 0.0f, cell, -9.0f, 1);
    field.heights.zero();
    field.heights(legacy_x0) = 0.0f;
    field.heights(legacy_x0 + 1) = 1.0f;
    field.heights(field.nx + legacy_x0) = 0.0f;
    field.heights(field.nx + legacy_x0 + 1) = 1.0f;
    check_float_bits(
        heightfield_sample(field, query_x, 0.0f), UINT32_C(0x00000000),
        "v1 arithmetic regression output bits");

    const double double_grid_x =
        (static_cast<double>(query_x) - static_cast<double>(origin)) /
        static_cast<double>(cell);
    check(double_grid_x < 77361.0 && double_grid_x > 77360.0,
          "v1 regression distinguishes the v2 double locator");
}

static byte_buffer awkward_python_oracle_bytes()
{
    static const unsigned char bytes[] = {
        0x47, 0x31, 0x48, 0x46, 0x02, 0x00, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0xcd, 0xcc, 0xcc, 0x3d, 0x9a, 0x99, 0x99, 0xbe,
        0x29, 0x5c, 0x8f, 0x3d, 0x9a, 0x99, 0xd9, 0xbf,
        0x00, 0x00, 0x00, 0x3e, 0x00, 0x00, 0xa0, 0x3f,
        0x00, 0x00, 0x20, 0x40, 0x00, 0x00, 0x70, 0x40,
        0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc8, 0x40,
    };
    return byte_buffer(bytes, bytes + sizeof(bytes));
}

static void test_v2_awkward_python_byte_and_query_oracle()
{
    const byte_buffer oracle = awkward_python_oracle_bytes();
    const byte_buffer independently_constructed = make_heightfield(
        2, 3, 2,
        float_from_bits(UINT32_C(0x3dcccccd)),
        float_from_bits(UINT32_C(0xbe99999a)),
        float_from_bits(UINT32_C(0x3d8f5c29)),
        float_from_bits(UINT32_C(0xbfd9999a)),
        {
            float_from_bits(UINT32_C(0x3e000000)),
            float_from_bits(UINT32_C(0x3fa00000)),
            float_from_bits(UINT32_C(0x40200000)),
            float_from_bits(UINT32_C(0x40700000)),
            float_from_bits(UINT32_C(0x40a00000)),
            float_from_bits(UINT32_C(0x40c80000)),
        });
    check(independently_constructed == oracle,
          "C++ serialization matches complete Python G1HF bytes");

    const heightfield field = load_heightfield_fixture(
        "/tmp/test_g1hf_python_oracle.bin", oracle);
    check(field.version == 2, "Python oracle version");
    check(field.nx == 3 && field.nz == 2, "Python oracle dimensions");
    check_float_bits(field.origin_x, UINT32_C(0x3dcccccd),
                     "Python oracle origin x bits");
    check_float_bits(field.origin_z, UINT32_C(0xbe99999a),
                     "Python oracle origin z bits");
    check_float_bits(field.cell_size, UINT32_C(0x3d8f5c29),
                     "Python oracle cell bits");
    check_float_bits(field.exterior_height, UINT32_C(0xbfd9999a),
                     "Python oracle exterior bits");
    const uint32_t height_bits[] = {
        UINT32_C(0x3e000000), UINT32_C(0x3fa00000),
        UINT32_C(0x40200000), UINT32_C(0x40700000),
        UINT32_C(0x40a00000), UINT32_C(0x40c80000),
    };
    for (int i = 0; i < field.heights.size; ++i) {
        check_float_bits(field.heights(i), height_bits[i],
                         "Python oracle height bits");
    }

    float runtime_node = -1.0f;
    check(terrain_v2_runtime_node_from_double(
              static_cast<double>(field.origin_x), runtime_node),
          "Python oracle first x node classification");
    check_float_bits(runtime_node, UINT32_C(0x3dcccccd),
                     "Python oracle first x node bits");
    const volatile double middle_x_product =
        1.0 * static_cast<double>(field.cell_size);
    const volatile double middle_x_source =
        static_cast<double>(field.origin_x) + middle_x_product;
    check(terrain_v2_runtime_node_from_double(middle_x_source, runtime_node),
          "Python oracle middle x node classification");
    check_float_bits(runtime_node, UINT32_C(0x3e2e147b),
                     "Python oracle middle x node bits");
    const volatile double final_x_product =
        2.0 * static_cast<double>(field.cell_size);
    const volatile double final_x_source =
        static_cast<double>(field.origin_x) + final_x_product;
    check(terrain_v2_runtime_node_from_double(final_x_source, runtime_node),
          "Python oracle final x node classification");
    check_float_bits(runtime_node, UINT32_C(0x3e75c290),
                     "Python oracle final x node bits");

    const float first_x = float_from_bits(UINT32_C(0x3e1c28f6));
    const float first_z = float_from_bits(UINT32_C(0xbe90a3d7));
    check_float_bits(
        heightfield_sample_versioned(field, first_x, first_z),
        UINT32_C(0x3ff40006),
        "Python oracle first-triangle height");
    check_vec3_bits(
        heightfield_normal(field, first_x, first_z),
        UINT32_C(0xbe93193e), UINT32_C(0x3c9271e1),
        UINT32_C(0xbf752a13),
        "Python oracle first-triangle normal");

    const float second_x = float_from_bits(UINT32_C(0x3df0a3d7));
    const float second_z = float_from_bits(UINT32_C(0xbe7d70a5));
    check_float_bits(
        heightfield_sample_versioned(field, second_x, second_z),
        UINT32_C(0x4049ffff),
        "Python oracle second-triangle height");
    check_vec3_bits(
        heightfield_normal(field, second_x, second_z),
        UINT32_C(0xbea6e122), UINT32_C(0x3c958623),
        UINT32_C(0xbf71f9a4),
        "Python oracle second-triangle normal");

    const float diagonal_x = float_from_bits(UINT32_C(0x3e0a3d71));
    const float diagonal_z = float_from_bits(UINT32_C(0xbe87ae15));
    check_float_bits(
        heightfield_sample_versioned(field, diagonal_x, diagonal_z),
        UINT32_C(0x4023ffff), "Python oracle diagonal height");
    check_vec3_bits(
        heightfield_normal(field, diagonal_x, diagonal_z),
        UINT32_C(0xbe93193e), UINT32_C(0x3c9271e1),
        UINT32_C(0xbf752a13),
        "Python oracle diagonal uses first triangle");

    const float x_line = float_from_bits(UINT32_C(0x3e2e147b));
    check_float_bits(
        heightfield_sample_versioned(field, x_line, first_z),
        UINT32_C(0x400c0003),
        "Python oracle interior x grid-line height");
    check_vec3_bits(
        heightfield_normal(field, x_line, first_z),
        UINT32_C(0xbea1e21c), UINT32_C(0x3c910c21),
        UINT32_C(0xbf72d32a),
        "Python oracle interior x grid-line owns positive-index cell");

    const float inward_x = float_from_bits(UINT32_C(0x3e75c28f));
    const float inward_z = float_from_bits(UINT32_C(0xbe6b8521));
    check_float_bits(
        heightfield_sample_versioned(field, inward_x, inward_z),
        UINT32_C(0x40c7fffd),
        "Python oracle inward maximum sample");
    check_vec3_bits(
        heightfield_normal(field, inward_x, inward_z),
        UINT32_C(0xbea1e21c), UINT32_C(0x3c910c21),
        UINT32_C(0xbf72d32a),
        "Python oracle inward maximum normal");

    const float rounded_outward_x = float_from_bits(UINT32_C(0x3e75c290));
    const float rounded_z = float_from_bits(UINT32_C(0xbe6b8520));
    check_float_bits(
        heightfield_sample_versioned(field, rounded_outward_x, rounded_z),
        UINT32_C(0xbfd9999a),
        "Python oracle outward-rounded maximum is exterior");
    check_vec3_bits(
        heightfield_normal(field, rounded_outward_x, rounded_z),
        UINT32_C(0x00000000), UINT32_C(0x3f800000),
        UINT32_C(0x00000000),
        "Python oracle outward-rounded maximum normal is up");

    check_float_bits(
        heightfield_sample_versioned(
            field, float_from_bits(UINT32_C(0x3dcccccc)), field.origin_z),
        UINT32_C(0xbfd9999a), "Python oracle lower outward edge");
    check_float_bits(
        heightfield_sample_versioned(
            field, float_from_bits(UINT32_C(0x3dccccce)), field.origin_z),
        UINT32_C(0x3e000008), "Python oracle lower inward edge");
}

static void test_v2_diagonal_decision_stays_binary64()
{
    heightfield field;
    initialize_heightfield(
        field, 2, 2,
        7.857595920562744f, -0.7662742137908936f,
        21.012887954711914f, -9.0f, 2);
    field.heights(0) = 0.0f;
    field.heights(1) = 2.0f;
    field.heights(2) = 4.0f;
    field.heights(3) = 10.0f;
    const float query_x = 22.33635711669922f;
    const float query_z = 13.71248722076416f;
    const double tx =
        (static_cast<double>(query_x) - static_cast<double>(field.origin_x)) /
        static_cast<double>(field.cell_size);
    const double tz =
        (static_cast<double>(query_z) - static_cast<double>(field.origin_z)) /
        static_cast<double>(field.cell_size);
    check(tx < tz, "diagonal regression double fractions choose second");
    check(static_cast<float>(tx) == static_cast<float>(tz),
          "diagonal regression float fractions tie");
    check_float_bits(
        heightfield_sample_versioned(field, query_x, query_z),
        UINT32_C(0x40dc7e51),
        "diagonal regression sample uses second triangle");
    check_vec3_bits(
        heightfield_normal(field, query_x, query_z),
        UINT32_C(0xbe8a47ae), UINT32_C(0x3f722376),
        UINT32_C(0xbe385f93),
        "diagonal regression normal uses second triangle");
}

static void test_v2_edges_grid_lines_and_diagonal_tie()
{
    heightfield field;
    initialize_heightfield(field, 3, 3, -1.0f, 2.0f, 1.0f, -99.0f, 2);
    const float values[] = {
        0.0f, 1.0f, 4.0f,
        2.0f, 8.0f, 16.0f,
        3.0f, 12.0f, 25.0f,
    };
    for (int i = 0; i < field.heights.size; ++i) {
        field.heights(i) = values[i];
    }

    for (int iz = 0; iz < field.nz; ++iz) {
        for (int ix = 0; ix < field.nx; ++ix) {
            const float x = field.origin_x + ix * field.cell_size;
            const float z = field.origin_z + iz * field.cell_size;
            check_float_bits(
                heightfield_sample_versioned(field, x, z),
                float_bits(values[iz * 3 + ix]),
                "v2 exact node sample");
        }
    }
    check_float_bits(
        heightfield_sample_versioned(field, 0.0f, 3.0f), float_bits(8.0f),
        "v2 interior grid intersection owns positive-index cell");
    check_float_bits(
        heightfield_sample_versioned(field, 1.0f, 4.0f), float_bits(25.0f),
        "v2 exact maximum owns final cell");

    const float interior_x = nextafterf(-1.0f, INFINITY);
    const float exterior_x = nextafterf(-1.0f, -INFINITY);
    check(heightfield_sample_versioned(field, interior_x, 3.0f) != -99.0f,
          "v2 lower-edge inward nextafter is interior");
    check(heightfield_sample_versioned(field, exterior_x, 3.0f) == -99.0f,
          "v2 lower-edge outward nextafter is exterior");
    const float maximum_inward_x = nextafterf(1.0f, -INFINITY);
    const float maximum_outward_x = nextafterf(1.0f, INFINITY);
    check(heightfield_sample_versioned(field, maximum_inward_x, 3.0f) != -99.0f,
          "v2 upper-edge inward nextafter is interior");
    check(heightfield_sample_versioned(field, maximum_outward_x, 3.0f) == -99.0f,
          "v2 upper-edge outward nextafter is exterior");
    const float lower_inward_z = nextafterf(2.0f, INFINITY);
    const float lower_outward_z = nextafterf(2.0f, -INFINITY);
    check(heightfield_sample_versioned(field, 0.0f, lower_inward_z) != -99.0f,
          "v2 lower z inward nextafter is interior");
    check(heightfield_sample_versioned(field, 0.0f, lower_outward_z) == -99.0f,
          "v2 lower z outward nextafter is exterior");
    const float upper_inward_z = nextafterf(4.0f, -INFINITY);
    const float upper_outward_z = nextafterf(4.0f, INFINITY);
    check(heightfield_sample_versioned(field, 0.0f, upper_inward_z) != -99.0f,
          "v2 upper z inward nextafter is interior");
    check(heightfield_sample_versioned(field, 0.0f, upper_outward_z) == -99.0f,
          "v2 upper z outward nextafter is exterior");

    heightfield diagonal;
    initialize_heightfield(diagonal, 2, 2, 0.0f, 0.0f, 1.0f, -9.0f, 2);
    diagonal.heights(0) = 0.0f;
    diagonal.heights(1) = 2.0f;
    diagonal.heights(2) = 4.0f;
    diagonal.heights(3) = 10.0f;
    check_vec3_bits(
        heightfield_normal(diagonal, 0.5f, 0.5f),
        UINT32_C(0xbe768cdc), UINT32_C(0x3df68cdc),
        UINT32_C(0xbf768cdc),
        "v2 exact diagonal tie uses first triangle");
}

static void check_sample_and_normal_finite(
    const heightfield& field, float x, float z, const char* message)
{
    const float sample = heightfield_sample_versioned(field, x, z);
    const vec3 normal = heightfield_normal(field, x, z);
    check(terrain_float_is_finite(sample), message);
    check(terrain_float_is_finite(normal.x), message);
    check(terrain_float_is_finite(normal.y), message);
    check(terrain_float_is_finite(normal.z), message);
}

static void test_v2_normals_extremes_and_ftz_outputs()
{
    heightfield flat;
    initialize_heightfield(flat, 2, 2, 0.0f, 0.0f, 0.02f, -9.0f, 2);
    flat.heights.set(4.0f);
    check_vec3_bits(
        heightfield_normal(flat, 0.01f, 0.01f),
        UINT32_C(0x00000000), UINT32_C(0x3f800000),
        UINT32_C(0x00000000),
        "flat v2 normal is bit-exact up");

    heightfield minimum_cell;
    initialize_heightfield(
        minimum_cell, 2, 2, 0.0f, 0.0f, FLT_MIN, -7.0f, 2);
    minimum_cell.heights(0) = -FLT_MAX;
    minimum_cell.heights(1) = FLT_MAX;
    minimum_cell.heights(2) = FLT_MAX;
    minimum_cell.heights(3) = -FLT_MAX;
    check_sample_and_normal_finite(
        minimum_cell, 0.0f, 0.0f,
        "minimum-normal cell extreme height query stays finite");
    check_sample_and_normal_finite(
        minimum_cell, FLT_MIN, FLT_MIN,
        "minimum-normal cell maximum node stays finite");

    const float huge_cell = float_from_bits(UINT32_C(0x7f7fffff));
    heightfield huge;
    initialize_heightfield(
        huge, 2, 2, 0.0f, 0.0f, huge_cell, -5.0f, 2);
    huge.heights(0) = 0.0f;
    huge.heights(1) = FLT_MIN;
    huge.heights(2) = 0.0f;
    huge.heights(3) = FLT_MIN;
    const float midpoint = huge_cell * 0.5f;
    check_float_bits(
        heightfield_sample_versioned(huge, midpoint, midpoint),
        UINT32_C(0x00000000),
        "v2 subnormal interpolation result is flushed to positive zero");
    check_vec3_bits(
        heightfield_normal(huge, midpoint, midpoint),
        UINT32_C(0x00000000), UINT32_C(0x3f800000),
        UINT32_C(0x00000000),
        "v2 subnormal normal components are flushed to positive zero");

    heightfield large_cell_extremes;
    initialize_heightfield(
        large_cell_extremes, 2, 2, 0.0f, 0.0f,
        float_from_bits(UINT32_C(0x71800000)), -3.0f, 2);
    large_cell_extremes.heights(0) = -FLT_MAX;
    large_cell_extremes.heights(1) = FLT_MAX;
    large_cell_extremes.heights(2) = -FLT_MAX;
    large_cell_extremes.heights(3) = FLT_MAX;
    check_sample_and_normal_finite(
        large_cell_extremes,
        large_cell_extremes.cell_size * 0.5f,
        large_cell_extremes.cell_size * 0.5f,
        "large-cell extreme interpolation and normal stay finite");
}

static void test_v2_runtime_node_precise_rounding_thresholds()
{
    const double lower_tie = 0x1p-150;
    const double upper_tie = 0x1p-126 - 0x1p-150;
    float rounded = -1.0f;

    check(terrain_v2_runtime_node_from_double(lower_tie, rounded),
          "positive lower tie is accepted");
    check_float_bits(rounded, UINT32_C(0x00000000),
                     "positive lower tie rounds to positive zero");
    check(terrain_v2_runtime_node_from_double(-lower_tie, rounded),
          "negative lower tie is accepted");
    check_float_bits(rounded, UINT32_C(0x00000000),
                     "negative lower tie canonicalizes positive zero");
    check(terrain_v2_runtime_node_from_double(
              nextafter(lower_tie, 0.0), rounded),
          "value below lower tie is accepted");
    check_float_bits(rounded, UINT32_C(0x00000000),
                     "value below lower tie rounds to positive zero");
    check(!terrain_v2_runtime_node_from_double(
              nextafter(lower_tie, INFINITY), rounded),
          "value above lower tie is rejected as subnormal");

    check(terrain_v2_runtime_node_from_double(upper_tie, rounded),
          "positive upper tie is accepted");
    check_float_bits(rounded, UINT32_C(0x00800000),
                     "positive upper tie rounds to minimum normal");
    check(terrain_v2_runtime_node_from_double(-upper_tie, rounded),
          "negative upper tie is accepted");
    check_float_bits(rounded, UINT32_C(0x80800000),
                     "negative upper tie rounds to minimum normal");
    check(!terrain_v2_runtime_node_from_double(
              nextafter(upper_tie, 0.0), rounded),
          "value below upper tie is rejected as subnormal");
    check(terrain_v2_runtime_node_from_double(
              nextafter(upper_tie, INFINITY), rounded),
          "value above upper tie is accepted as normal");
    check_float_bits(rounded, UINT32_C(0x00800000),
                     "value above upper tie remains minimum normal");
}

static void test_v2_rejects_encoded_domain_and_axis_failures()
{
    const char* path = "/tmp/test_g1hf_v2_domain.bin";
    const float minimum_subnormal = float_from_bits(UINT32_C(0x00000001));
    const float negative_subnormal = float_from_bits(UINT32_C(0x80000001));
    const float negative_zero = float_from_bits(UINT32_C(0x80000000));
    const float next_minimum_normal =
        float_from_bits(UINT32_C(0x00800001));
    const std::vector<float> zeros4(4, 0.0f);

    write_payload(path, make_heightfield(
        2, 2, 2, minimum_subnormal, 0.0f, 1.0f, 0.0f, zeros4));
    expect_heightfield_rejected(path, "normal-or-positive-zero");
    write_payload(path, make_heightfield(
        2, 2, 2, 0.0f, negative_subnormal, 1.0f, 0.0f, zeros4));
    expect_heightfield_rejected(path, "normal-or-positive-zero");
    write_payload(path, make_heightfield(
        2, 2, 2, 0.0f, 0.0f, minimum_subnormal, 0.0f, zeros4));
    expect_heightfield_rejected(path, "cell size");
    write_payload(path, make_heightfield(
        2, 2, 2, 0.0f, 0.0f, 1.0f, minimum_subnormal, zeros4));
    expect_heightfield_rejected(path, "normal-or-positive-zero");
    write_payload(path, make_heightfield(
        2, 2, 2, negative_zero, 0.0f, 1.0f, 0.0f, zeros4));
    expect_heightfield_rejected(path, "normal-or-positive-zero");
    write_payload(path, make_heightfield(
        2, 2, 2, 0.0f, negative_zero, 1.0f, 0.0f, zeros4));
    expect_heightfield_rejected(path, "normal-or-positive-zero");
    write_payload(path, make_heightfield(
        2, 2, 2, 0.0f, 0.0f, 1.0f, negative_zero, zeros4));
    expect_heightfield_rejected(path, "normal-or-positive-zero");
    write_payload(path, make_heightfield(
        2, 2, 2, 0.0f, 0.0f, 1.0f, 0.0f,
        {0.0f, minimum_subnormal, 0.0f, 0.0f}));
    expect_heightfield_rejected(path, "normal-or-positive-zero");
    write_payload(path, make_heightfield(
        2, 2, 2, 0.0f, 0.0f, 1.0f, 0.0f,
        {0.0f, negative_zero, 0.0f, 0.0f}));
    expect_heightfield_rejected(path, "normal-or-positive-zero");

    write_payload(path, make_heightfield(
        2, 2, 2, -FLT_MIN, 0.0f, next_minimum_normal, 0.0f, zeros4));
    expect_heightfield_rejected(path, "normal-or-zero runtime nodes");
    write_payload(path, make_heightfield(
        2, 2, 2, 1.0e8f, 0.0f, 0.02f, 0.0f, zeros4));
    expect_heightfield_rejected(path, "runtime-distinguishable");
    write_payload(path, make_heightfield(
        2, 2, 2, FLT_MAX, 0.0f, FLT_MAX, 0.0f, zeros4));
    expect_heightfield_rejected(path, "runtime-distinguishable");
    write_payload(path, make_heightfield(
        2, 4, 2, 1.0e8f, 0.0f, 5.0f, 0.0f,
        std::vector<float>(8, 0.0f)));
    expect_heightfield_rejected(path, "runtime-distinguishable");

    const heightfield v1_subnormal = load_heightfield_fixture(
        "/tmp/test_g1hf_v1_subnormal_domain.bin",
        make_heightfield(
            1, 2, 2, minimum_subnormal, negative_subnormal,
            1.0f, negative_zero,
            {0.0f, minimum_subnormal, negative_zero, 0.0f}));
    check(v1_subnormal.version == 1,
          "v1 retains historical subnormal and signed-zero acceptance");
    check_float_bits(v1_subnormal.origin_x, UINT32_C(0x00000001),
                     "v1 preserves subnormal origin bits");
    check_float_bits(v1_subnormal.exterior_height, UINT32_C(0x80000000),
                     "v1 preserves negative-zero exterior bits");

    const heightfield v1_collapsed = load_heightfield_fixture(
        "/tmp/test_g1hf_v1_collapsed_axis.bin",
        make_heightfield(
            1, 4, 2, 1.0e8f, 0.0f, 5.0f, 0.0f,
            std::vector<float>(8, 0.0f)));
    check(v1_collapsed.version == 1,
          "v1 retains historical collapsed-axis acceptance");
}

static void test_v2_query_domain_and_invalid_structures_are_safe()
{
    heightfield valid;
    initialize_heightfield(valid, 2, 2, 0.0f, 0.0f, 1.0f, -17.0f, 2);
    valid.heights(0) = 5.0f;
    valid.heights(1) = 6.0f;
    valid.heights(2) = 7.0f;
    valid.heights(3) = 8.0f;
    const float minimum_subnormal = float_from_bits(UINT32_C(0x00000001));
    const float negative_subnormal = float_from_bits(UINT32_C(0x80000001));
    const float negative_zero = float_from_bits(UINT32_C(0x80000000));

    check_float_bits(
        heightfield_sample_versioned(valid, minimum_subnormal, 0.0f),
        float_bits(valid.exterior_height),
        "positive subnormal v2 query is exterior");
    check_float_bits(
        heightfield_sample_versioned(valid, negative_subnormal, 0.0f),
        float_bits(valid.exterior_height),
        "negative subnormal v2 query is exterior");
    check_vec3_bits(
        heightfield_normal(valid, minimum_subnormal, 0.0f),
        UINT32_C(0x00000000), UINT32_C(0x3f800000),
        UINT32_C(0x00000000),
        "subnormal v2 query normal is up");
    check_float_bits(
        heightfield_sample_versioned(valid, negative_zero, negative_zero),
        float_bits(5.0f), "query signed zero canonicalizes positive");

    heightfield invalid_version;
    initialize_heightfield(
        invalid_version, 2, 2, 0.0f, 0.0f, 1.0f, -17.0f, 3);
    invalid_version.heights.set(1.0f);
    check(heightfield_sample_v2(invalid_version, 0.0f, 0.0f) == -17.0f,
          "direct v2 entry rejects a wrong in-memory version");
    check(heightfield_sample_versioned(invalid_version, 0.0f, 0.0f) == -17.0f,
          "unknown in-memory version returns exterior");
    check_vec3_bits(
        heightfield_normal(invalid_version, 0.0f, 0.0f),
        UINT32_C(0x00000000), UINT32_C(0x3f800000),
        UINT32_C(0x00000000),
        "unknown in-memory version normal is up");

    heightfield bad_storage;
    initialize_heightfield(
        bad_storage, 2, 2, 0.0f, 0.0f, 1.0f, -17.0f, 2);
    bad_storage.heights.resize(3);
    check(heightfield_sample_v2(bad_storage, 0.0f, 0.0f) == -17.0f,
          "direct v2 entry rejects mismatched in-memory storage");
    check(heightfield_sample_versioned(bad_storage, 0.0f, 0.0f) == -17.0f,
          "mismatched in-memory storage returns exterior");
    check_vec3_bits(
        heightfield_normal(bad_storage, 0.0f, 0.0f),
        UINT32_C(0x00000000), UINT32_C(0x3f800000),
        UINT32_C(0x00000000),
        "mismatched in-memory storage normal is up");

    heightfield null_storage;
    null_storage.version = 2;
    null_storage.nx = 2;
    null_storage.nz = 2;
    null_storage.origin_x = 0.0f;
    null_storage.origin_z = 0.0f;
    null_storage.cell_size = 1.0f;
    null_storage.exterior_height = -17.0f;
    null_storage.heights.size = 4;
    check(heightfield_sample_v2(null_storage, 0.0f, 0.0f) == -17.0f,
          "direct v2 entry rejects null in-memory storage");

    heightfield bad_cell;
    initialize_heightfield(
        bad_cell, 2, 2, 0.0f, 0.0f, 0.0f, -17.0f, 2);
    bad_cell.heights.set(1.0f);
    check(heightfield_sample_v2(bad_cell, 0.0f, 0.0f) == -17.0f,
          "direct v2 entry rejects invalid in-memory metadata");

    heightfield bad_v1_storage;
    initialize_heightfield(
        bad_v1_storage, 2, 2, 0.0f, 0.0f, 1.0f, -17.0f, 1);
    bad_v1_storage.heights.resize(0);
    check(heightfield_sample_versioned(
              bad_v1_storage, 0.0f, 0.0f) == -17.0f,
          "versioned entry rejects mismatched in-memory v1 storage");

    heightfield collapsed;
    initialize_heightfield(
        collapsed, 4, 2, 1.0e8f, 0.0f, 5.0f, -17.0f, 2);
    collapsed.heights.zero();
    check(heightfield_sample_versioned(collapsed, 1.0e8f, 0.0f) == -17.0f,
          "collapsed in-memory v2 axis returns exterior");
    check_vec3_bits(
        heightfield_normal(collapsed, 1.0e8f, 0.0f),
        UINT32_C(0x00000000), UINT32_C(0x3f800000),
        UINT32_C(0x00000000),
        "collapsed in-memory v2 axis normal is up");

    valid.heights(0) = minimum_subnormal;
    check(heightfield_sample_versioned(valid, 0.0f, 0.0f) == -17.0f,
          "invalid selected in-memory v2 height returns exterior");
    check_vec3_bits(
        heightfield_normal(valid, 0.0f, 0.0f),
        UINT32_C(0x00000000), UINT32_C(0x3f800000),
        UINT32_C(0x00000000),
        "invalid selected in-memory v2 height normal is up");
}

static bool test_float_is_normal_or_positive_zero(float value)
{
    const uint32_t bits = float_bits(value);
    const uint32_t magnitude = bits & UINT32_C(0x7fffffff);
    const uint32_t exponent = magnitude & UINT32_C(0x7f800000);
    return bits == 0 ||
           (exponent != 0 && exponent != UINT32_C(0x7f800000));
}

static void test_v2_axis_validator_vectors_and_seeded_property()
{
    const float aligned_cell = float_from_bits(UINT32_C(0x34800000));
    check(terrain_v2_runtime_axis_is_valid(0.0f, 2, 1.0f),
          "basic v2 axis is valid");
    check(terrain_v2_runtime_axis_is_valid(2.0f, 5, aligned_cell),
          "aligned equal-spacing axis is valid");
    check(!terrain_v2_runtime_axis_is_valid(
              float_from_bits(UINT32_C(0x3fffffff)), 5, aligned_cell),
          "unaligned equal-spacing axis is rejected");
    check(!terrain_v2_runtime_axis_is_valid(1.0e20f, 2, 1.0f),
          "binary64-collapsed axis is rejected");
    check(!terrain_v2_runtime_axis_is_valid(1.0e8f, 2, 0.02f),
          "binary32-collapsed axis is rejected");
    check(!terrain_v2_runtime_axis_is_valid(1.0e8f, 4, 5.0f),
          "interior-collapsed axis is rejected");
    check(!terrain_v2_runtime_axis_is_valid(
              -FLT_MIN, 2, float_from_bits(UINT32_C(0x00800001))),
          "derived-subnormal axis is rejected");
    check(!terrain_v2_runtime_axis_is_valid(
              float_from_bits(UINT32_C(0x81400001)), 7,
              float_from_bits(UINT32_C(0x00800001))),
          "seven-node interior-derived-subnormal axis is rejected");
    check(!terrain_v2_runtime_axis_is_valid(0.0f, UINT32_MAX, 1.0f),
          "huge header axis is validated without dimension-dependent work");

    std::mt19937 generator(UINT32_C(20260713));
    int accepted = 0;
    for (int trial = 0; trial < 4000; ++trial) {
        const uint32_t origin_bits = generator();
        const uint32_t cell_bits =
            (generator() & UINT32_C(0x7fffffff));
        const float origin = float_from_bits(origin_bits);
        const float cell = float_from_bits(cell_bits);
        const uint32_t count = 2u + generator() % 31u;
        if (!test_float_is_normal_or_positive_zero(origin) ||
            !test_float_is_normal_or_positive_zero(cell) || cell <= 0.0f ||
            !terrain_v2_runtime_axis_is_valid(origin, count, cell)) {
            continue;
        }
        ++accepted;
        double previous_source = 0.0;
        float previous_runtime = 0.0f;
        for (uint32_t index = 0; index < count; ++index) {
            const volatile double product =
                static_cast<double>(index) * static_cast<double>(cell);
            const volatile double source =
                static_cast<double>(origin) + product;
            float runtime = 0.0f;
            check(terrain_v2_runtime_node_from_double(source, runtime),
                  "accepted axis node domain");
            if (index != 0) {
                check(source > previous_source,
                      "accepted axis source nodes strictly increase");
                check(runtime > previous_runtime,
                      "accepted axis runtime nodes strictly increase");
            }
            previous_source = source;
            previous_runtime = runtime;
        }
    }
    check(accepted > 100, "axis property has enough accepted fixtures");
}

static void test_heightfield_rejects_float_rounded_upper_grid_index()
{
    heightfield field;
    field.version = 1;
    field.nx = 16777220;
    field.nz = 2;
    field.origin_x = 0.0f;
    field.origin_z = 0.0f;
    field.cell_size = 1.0f;
    field.exterior_height = -99.0f;
    field.heights.resize(field.nx * field.nz);
    field.heights.zero();

    const float rounded_upper_x = static_cast<float>(field.nx - 1);
    check(static_cast<double>(rounded_upper_x) >
              static_cast<double>(field.nx - 1),
          "v1 rounded upper coordinate is outward");
    const volatile float outward_sample =
        heightfield_sample(field, rounded_upper_x, 1.0f);
    check(outward_sample == field.exterior_height,
          "v1 rounded upper coordinate returns exterior");

    const float inward_x = nextafterf(rounded_upper_x, -INFINITY);
    const int inward_x0 = static_cast<int>(floorf(inward_x));
    check(inward_x0 >= 0 && inward_x0 + 1 < field.nx,
          "v1 inward coordinate has valid cell");
    const volatile float inward_sample =
        heightfield_sample(field, inward_x, 1.0f);
    check(inward_sample == 0.0f,
          "v1 inward coordinate samples storage");
}

static void test_stationary_centerline_extends_current_heading()
{
    heightfield field;
    initialize_heightfield(field, 2, 5, 0.0f, 0.0f, 0.25f, -9.0f);
    for (int z = 0; z < field.nz; ++z) {
        for (int x = 0; x < field.nx; ++x) {
            field.heights(z * field.nx + x) = 0.25f * z;
        }
    }

    array1d<vec3> positions(4);
    array1d<quat> rotations(4);
    for (int i = 0; i < positions.size; ++i) {
        positions(i) = vec3(0.0f, 10.0f * i, 0.0f);
        rotations(i) = quat();
    }

    const vec3 endpoint = terrain_centerline_point_at_arc(
        vec3(0.0f, 3.0f, 0.0f), positions, rotations, 1.0f);
    check_close(endpoint.x, 0.0f, "stationary endpoint x");
    check_close(endpoint.y, 0.0f, "stationary endpoint ignores y");
    check_close(endpoint.z, 1.0f, "stationary endpoint heading extension");

    float query[4] = {};
    terrain_centerline_query(
        query, field, vec3(0.0f, 3.0f, 0.0f), positions, rotations);
    const float expected[4] = {0.25f, 0.50f, 0.75f, 1.00f};
    for (int i = 0; i < 4; ++i) {
        check_close(query[i], expected[i], "stationary terrain query");
    }
}

static void test_straight_step_query_is_ground_relative_on_elevated_base()
{
    heightfield field;
    initialize_heightfield(field, 5, 2, 0.0f, 0.0f, 0.25f, 1.0f);
    for (int z = 0; z < field.nz; ++z) {
        for (int x = 0; x < field.nx; ++x) {
            field.heights(z * field.nx + x) =
                1.0f + (x >= 2 ? 0.29f : 0.0f);
        }
    }

    array1d<vec3> positions(2);
    positions(0) = vec3(0.0f, 0.0f, 0.0f);
    positions(1) = vec3(1.0f, 100.0f, 0.0f);
    array1d<quat> rotations(2);
    rotations(0) = heading_positive_x();
    rotations(1) = heading_positive_x();

    float query[4] = {};
    terrain_centerline_query(
        query, field, vec3(0.0f, 1.0f, 0.0f), positions, rotations);
    const float expected[4] = {0.0f, 0.29f, 0.29f, 0.29f};
    for (int i = 0; i < 4; ++i) {
        check_close(query[i], expected[i], "elevated step terrain query");
    }
}

static void test_curved_query_matches_python_geometric_arc_fixture()
{
    heightfield field;
    initialize_heightfield(field, 5, 5, 0.0f, 0.0f, 0.25f, -99.0f);
    for (int z = 0; z < field.nz; ++z) {
        for (int x = 0; x < field.nx; ++x) {
            const float world_x = 0.25f * x;
            const float world_z = 0.25f * z;
            field.heights(z * field.nx + x) = world_x + 10.0f * world_z;
        }
    }

    array1d<vec3> positions(3);
    positions(0) = vec3(0.0f, 0.0f, 0.0f);
    positions(1) = vec3(0.5f, 20.0f, 0.0f);
    positions(2) = vec3(0.5f, -20.0f, 0.5f);
    array1d<quat> rotations(3);
    rotations(0) = heading_positive_x();
    rotations(1) = heading_positive_x();
    rotations(2) = quat();

    float query[4] = {};
    terrain_centerline_query(
        query, field, vec3(0.0f, 0.0f, 0.0f), positions, rotations);
    const float expected[4] = {0.25f, 0.50f, 3.0f, 5.5f};
    for (int i = 0; i < 4; ++i) {
        check_close(query[i], expected[i], "curved Python parity query");
    }
}

static void test_centerline_snapshot_keeps_query_and_markers_in_one_state()
{
    heightfield field;
    initialize_heightfield(field, 9, 9, 0.0f, 0.0f, 0.25f, -99.0f);
    for (int z = 0; z < field.nz; ++z) {
        for (int x = 0; x < field.nx; ++x) {
            field.heights(z * field.nx + x) =
                0.25f * static_cast<float>(x) +
                2.0f * 0.25f * static_cast<float>(z);
        }
    }

    vec3 root(0.25f, 3.0f, 0.25f);
    array1d<vec3> positions(3);
    positions(0) = root;
    positions(1) = vec3(0.75f, 20.0f, 0.25f);
    positions(2) = vec3(1.25f, -20.0f, 0.25f);
    array1d<quat> rotations(3);
    rotations(0) = heading_positive_x();
    rotations(1) = heading_positive_x();
    rotations(2) = heading_positive_x();

    terrain_centerline_snapshot snapshot = {};
    terrain_centerline_snapshot_compute(
        snapshot, field, root, positions, rotations);
    const terrain_centerline_snapshot original = snapshot;
    const float original_base = heightfield_sample(field, root.x, root.z);
    for (int i = 0; i < 4; ++i) {
        const float marker_height = heightfield_sample(
            field, snapshot.points[i].x, snapshot.points[i].z);
        check_close(
            snapshot.points[i].y,
            marker_height,
            "snapshot marker surface height");
        check_close(
            snapshot.values[i],
            marker_height - original_base,
            "snapshot query uses marker geometry");
    }

    // Simulate the controller advancing to a different pose/frame after the
    // search query was built. The prior diagnostic snapshot must stay intact.
    root = vec3(1.0f, 7.0f, 1.0f);
    positions(0) = root;
    positions(1) = vec3(1.0f, 0.0f, 1.5f);
    positions(2) = vec3(1.0f, 0.0f, 2.0f);
    rotations(0) = quat();
    rotations(1) = quat();
    rotations(2) = quat();
    terrain_centerline_snapshot advanced = {};
    terrain_centerline_snapshot_compute(
        advanced, field, root, positions, rotations);

    check(
        advanced.points[0].x != snapshot.points[0].x ||
        advanced.points[0].z != snapshot.points[0].z,
        "advanced state must differ from query snapshot");
    for (int i = 0; i < 4; ++i) {
        check_close(
            snapshot.values[i],
            original.values[i],
            "query snapshot value changed after live advance");
        check_close(
            snapshot.points[i].x,
            original.points[i].x,
            "query snapshot marker x changed after live advance");
        check_close(
            snapshot.points[i].y,
            original.points[i].y,
            "query snapshot marker y changed after live advance");
        check_close(
            snapshot.points[i].z,
            original.points[i].z,
            "query snapshot marker z changed after live advance");
    }
}

static void initialize_centerline_walkability_fixture(
    heightfield& field,
    walkability_grid& grid,
    bool rising_ramp)
{
    initialize_heightfield(
        field, 136, 31, -1.20f, 0.0f, 0.02f, 0.0f, 2);
    grid.nx = field.nx;
    grid.nz = field.nz;
    grid.cells.resize(field.nx * field.nz);
    grid.cells.set(1);
    for (int z = 0; z < field.nz; ++z) {
        for (int x = 0; x < field.nx; ++x) {
            const float world_x =
                field.origin_x + field.cell_size * static_cast<float>(x);
            float height = 0.0f;
            if (rising_ramp) {
                height = world_x > 0.50f ?
                    minf(0.35f, (world_x - 0.50f) * 0.70f) : 0.0f;
            } else if (world_x >= 0.60f && world_x <= 1.00f) {
                height = 0.45f;
            }
            field.heights(z * field.nx + x) = height;
            if (world_x >= 0.70f &&
                (rising_ramp || world_x <= 1.00f)) {
                grid.cells(z * grid.nx + x) = 0;
            }
        }
    }
}

static void compute_stationary_centerline_snapshot(
    terrain_centerline_snapshot& snapshot,
    const heightfield& field,
    vec3 root,
    quat heading)
{
    array1d<vec3> positions(2);
    array1d<quat> rotations(2);
    positions(0) = root;
    positions(1) = root;
    rotations(0) = heading;
    rotations(1) = heading;
    terrain_centerline_snapshot_compute_v2(
        snapshot, field, root, positions, rotations);
}

static void check_centerline_walkability_latch(bool rising_ramp)
{
    heightfield field;
    walkability_grid grid;
    initialize_centerline_walkability_fixture(field, grid, rising_ramp);
    const vec3 query_root(0.20f, 3.0f, 0.30f);
    const vec3 footprint_origin(0.20f, -4.0f, 0.30f);
    terrain_centerline_snapshot raw = {};
    compute_stationary_centerline_snapshot(
        raw, field, query_root, heading_positive_x());
    check(raw.points[1].y > raw.points[0].y,
          "blocked terrain fixture rises at the second sample");
    if (!rising_ramp) {
        check(raw.points[3].y == 0.0f,
              "wall fixture exposes flat raw terrain beyond the barrier");
    }

    terrain_centerline_snapshot filtered = raw;
    check(terrain_centerline_snapshot_apply_walkability_v2(
              filtered, field, grid, query_root, footprint_origin, 0.20f),
          "walkability centerline filtering succeeds");
    check(float_bits(filtered.values[0]) == float_bits(raw.values[0]),
          "clear centerline value remains bit-identical");
    check_vec3_bits(
        filtered.points[0],
        float_bits(raw.points[0].x),
        float_bits(raw.points[0].y),
        float_bits(raw.points[0].z),
        "clear centerline point remains bit-identical");
    const walkability_sweep_result expected_sweep = walkability_sweep(
        grid, field, raw.points[0], raw.points[1], 0.20f);
    check(expected_sweep.blocked &&
              expected_sweep.reason == walkability_blocked_cell,
          "centerline wall/ramp sweep reaches a blocked cell");
    check(float_bits(expected_sweep.point.x) !=
                  float_bits(raw.points[0].x) ||
              float_bits(expected_sweep.point.z) !=
                  float_bits(raw.points[0].z),
          "wall/ramp last-safe point advances beyond the earlier sample");
    check_centerline_latch_matches_sweep(
        filtered, field, query_root, 1, expected_sweep,
        "wall/ramp latch equals the exact direct sweep result");
}

static void test_centerline_walkability_latches_wall_and_ramp()
{
    check_centerline_walkability_latch(false);
    check_centerline_walkability_latch(true);
}

static void test_centerline_walkability_preserves_clear_classes_and_recovers()
{
    heightfield field;
    walkability_grid grid;
    initialize_centerline_walkability_fixture(field, grid, false);
    const vec3 root(0.20f, 3.0f, 0.30f);
    terrain_centerline_snapshot raw = {};
    compute_stationary_centerline_snapshot(
        raw, field, root, heading_positive_x());

    grid.cells.set(1);
    terrain_centerline_snapshot class_one = raw;
    check(terrain_centerline_snapshot_apply_walkability_v2(
              class_one, field, grid, root, root, 0.20f),
          "class-one centerline filtering succeeds");
    check_centerline_snapshot_bits_equal(
        class_one, raw, "class-one centerline remains bit-identical");

    grid.cells.set(2);
    terrain_centerline_snapshot class_two = raw;
    check(terrain_centerline_snapshot_apply_walkability_v2(
              class_two, field, grid, root, root, 0.20f),
          "class-two centerline filtering succeeds");
    check_centerline_snapshot_bits_equal(
        class_two, raw, "class-two centerline remains bit-identical");

    initialize_centerline_walkability_fixture(field, grid, false);
    terrain_centerline_snapshot forward = raw;
    check(terrain_centerline_snapshot_apply_walkability_v2(
              forward, field, grid, root, root, 0.20f),
          "stationary forward centerline filtering succeeds");
    check(float_bits(forward.values[1]) != float_bits(raw.values[1]) ||
              float_bits(forward.points[1].x) != float_bits(raw.points[1].x),
          "stationary forward centerline remains filtered");

    terrain_centerline_snapshot away_raw = {};
    const quat away_heading = quat_from_angle_axis(
        -0.5f * PIf, vec3(0.0f, 1.0f, 0.0f));
    compute_stationary_centerline_snapshot(
        away_raw, field, root, away_heading);
    terrain_centerline_snapshot away_filtered = away_raw;
    check(terrain_centerline_snapshot_apply_walkability_v2(
              away_filtered, field, grid, root, root, 0.20f),
          "fresh away-facing centerline filtering succeeds");
    check_centerline_snapshot_bits_equal(
        away_filtered, away_raw,
        "fresh away-facing centerline recovers without a persistent latch");
}

static void test_centerline_walkability_invalid_inputs_are_transactional()
{
    heightfield field;
    walkability_grid grid;
    initialize_centerline_walkability_fixture(field, grid, false);
    const vec3 root(0.20f, 3.0f, 0.30f);
    terrain_centerline_snapshot raw = {};
    compute_stationary_centerline_snapshot(
        raw, field, root, heading_positive_x());

    const auto expect_failure = [&field](
        const terrain_centerline_snapshot& seed,
        const walkability_grid& candidate_grid,
        vec3 query_root,
        vec3 footprint_origin,
        float radius,
        const char* message) {
        terrain_centerline_snapshot actual = seed;
        const terrain_centerline_snapshot expected = actual;
        check(!terrain_centerline_snapshot_apply_walkability_v2(
                  actual, field, candidate_grid, query_root,
                  footprint_origin, radius),
              message);
        check_centerline_snapshot_bits_equal(actual, expected, message);
    };

    walkability_grid mismatch(grid);
    mismatch.nx -= 1;
    expect_failure(raw, mismatch, root, root, 0.20f,
                   "mismatched centerline grid is transactional");
    expect_failure(
        raw, grid,
        vec3(float_from_bits(UINT32_C(0x7fc23456)), root.y, root.z),
        root, 0.20f,
        "nonfinite animation query root is transactional");
    expect_failure(
        raw, grid, root,
        vec3(root.x, std::numeric_limits<float>::infinity(), root.z),
        0.20f,
        "nonfinite footprint origin is transactional");
    expect_failure(raw, grid, root, root, -0.20f,
                   "negative centerline radius is transactional");
    expect_failure(
        raw, grid, root, root,
        float_from_bits(UINT32_C(0x7fc34567)),
        "nonfinite centerline radius is transactional");

    terrain_centerline_snapshot malformed = raw;
    malformed.values[2] = float_from_bits(UINT32_C(0x7fc45678));
    malformed.points[3].z = std::numeric_limits<float>::infinity();
    expect_failure(malformed, grid, root, root, 0.20f,
                   "malformed centerline snapshot is transactional");

    terrain_centerline_snapshot blocked_start = raw;
    check(terrain_centerline_snapshot_apply_walkability_v2(
              blocked_start, field, grid, root,
              vec3(0.80f, -7.0f, 0.30f), 0.20f),
          "blocked starting footprint produces a safe root hold");
    const float root_height = heightfield_sample_v2(field, root.x, root.z);
    for (int i = 0; i < 4; ++i) {
        check_float_bits(
            blocked_start.values[i], UINT32_C(0x00000000),
            "blocked start repeats zero root-relative terrain");
        check_vec3_bits(
            blocked_start.points[i], float_bits(root.x),
            float_bits(root_height), float_bits(root.z),
            "blocked start repeats the query-root surface point");
    }
}

static void initialize_centerline_planar_contract_fixture(
    heightfield& field,
    walkability_grid& grid,
    int nx)
{
    initialize_heightfield(
        field, nx, 101, -1.0f, -1.0f, 0.02f, -3.0f, 2);
    grid.nx = field.nx;
    grid.nz = field.nz;
    grid.cells.resize(field.nx * field.nz);
    grid.cells.set(1);
    for (int z = 0; z < field.nz; ++z) {
        for (int x = 0; x < field.nx; ++x) {
            const float world_x =
                field.origin_x + field.cell_size * static_cast<float>(x);
            const float world_z =
                field.origin_z + field.cell_size * static_cast<float>(z);
            field.heights(z * field.nx + x) =
                0.60f + 0.40f * world_x + 0.15f * world_z;
        }
    }
}

static void test_centerline_walkability_separates_query_and_footprint_roots()
{
    heightfield field;
    walkability_grid grid;
    initialize_centerline_planar_contract_fixture(field, grid, 151);
    for (int z = 0; z < field.nz; ++z) {
        for (int x = 0; x < field.nx; ++x) {
            const float world_x =
                field.origin_x + field.cell_size * static_cast<float>(x);
            const float world_z =
                field.origin_z + field.cell_size * static_cast<float>(z);
            if (world_x >= -0.22f && world_x <= 0.22f &&
                world_z >= -0.02f && world_z <= 0.02f) {
                grid.cells(z * grid.nx + x) = 0;
            }
        }
    }

    const vec3 query_root(0.0f, 3.0f, 0.35f);
    const vec3 footprint_origin(-0.35f, -4.0f, -0.35f);
    check(float_bits(query_root.x) != float_bits(footprint_origin.x) &&
              float_bits(query_root.z) != float_bits(footprint_origin.z),
          "query and footprint roots differ in both planar coordinates");
    const float query_base = heightfield_sample_v2(
        field, query_root.x, query_root.z);
    const float footprint_surface = heightfield_sample_v2(
        field, footprint_origin.x, footprint_origin.z);
    check(float_bits(query_base) != float_bits(footprint_surface),
          "animation-root base is distinguishable from footprint terrain");

    terrain_centerline_snapshot raw = {};
    compute_stationary_centerline_snapshot(
        raw, field, query_root, heading_positive_x());
    const walkability_sweep_result swapped_origin_sweep = walkability_sweep(
        grid, field, query_root, raw.points[0], 0.20f);
    check(!swapped_origin_sweep.blocked,
          "animation-root-to-sample sweep remains clear");
    const walkability_sweep_result expected_sweep = walkability_sweep(
        grid, field, footprint_origin, raw.points[0], 0.20f);
    check(expected_sweep.blocked &&
              expected_sweep.reason == walkability_blocked_cell,
          "footprint-origin-to-sample sweep reaches the blocker");
    const float safe_height = heightfield_sample_v2(
        field, expected_sweep.point.x, expected_sweep.point.z);
    const float query_relative = expected_centerline_relative_value(
        safe_height, query_base);
    const float footprint_relative = expected_centerline_relative_value(
        safe_height, footprint_surface);
    check(float_bits(query_relative) != float_bits(footprint_relative),
          "latched value distinguishes animation and footprint bases");
    check(float_bits(expected_sweep.point.x) !=
                  float_bits(footprint_origin.x) ||
              float_bits(expected_sweep.point.z) !=
                  float_bits(footprint_origin.z),
          "distinct-root sweep advances before reaching the blocker");

    terrain_centerline_snapshot filtered = raw;
    check(terrain_centerline_snapshot_apply_walkability_v2(
              filtered, field, grid, query_root, footprint_origin, 0.20f),
          "distinct-root centerline filtering succeeds");
    check_centerline_latch_matches_sweep(
        filtered, field, query_root, 0, expected_sweep,
        "distinct-root latch uses footprint sweep and animation base");
}

static void test_centerline_walkability_latches_out_of_bounds_segment()
{
    heightfield field;
    walkability_grid grid;
    initialize_centerline_planar_contract_fixture(field, grid, 93);
    const vec3 query_root(0.0f, 2.0f, 0.0f);
    const vec3 footprint_origin(-0.10f, -2.0f, 0.05f);
    terrain_centerline_snapshot raw = {};
    compute_stationary_centerline_snapshot(
        raw, field, query_root, heading_positive_x());

    const walkability_sweep_result first = walkability_sweep(
        grid, field, footprint_origin, raw.points[0], 0.20f);
    const walkability_sweep_result second = walkability_sweep(
        grid, field, raw.points[0], raw.points[1], 0.20f);
    const walkability_sweep_result expected_sweep = walkability_sweep(
        grid, field, raw.points[1], raw.points[2], 0.20f);
    check(!first.blocked && !second.blocked,
          "out-of-bounds fixture preserves its first two segments");
    check(expected_sweep.blocked &&
              expected_sweep.reason == walkability_out_of_bounds,
          "third centerline segment exits the grid footprint bounds");
    check(float_bits(expected_sweep.point.x) !=
                  float_bits(raw.points[1].x) ||
              float_bits(expected_sweep.point.z) !=
                  float_bits(raw.points[1].z),
          "out-of-bounds last-safe point advances beyond the prior sample");

    terrain_centerline_snapshot filtered = raw;
    check(terrain_centerline_snapshot_apply_walkability_v2(
              filtered, field, grid, query_root, footprint_origin, 0.20f),
          "out-of-bounds centerline filtering succeeds");
    for (int i = 0; i < 2; ++i) {
        check_float_bits(
            filtered.values[i], float_bits(raw.values[i]),
            "clear pre-boundary value remains bit-identical");
        check_vec3_bits(
            filtered.points[i],
            float_bits(raw.points[i].x),
            float_bits(raw.points[i].y),
            float_bits(raw.points[i].z),
            "clear pre-boundary point remains bit-identical");
    }
    check_centerline_latch_matches_sweep(
        filtered, field, query_root, 2, expected_sweep,
        "out-of-bounds latch equals the exact direct sweep result");
}

static void test_v2_centerline_uses_checked_triangular_height_samples()
{
    heightfield legacy;
    initialize_heightfield(
        legacy, 2, 2, 0.0f, 0.0f, 1.0f, -9.0f, 1);
    legacy.heights(0) = 0.0f;
    legacy.heights(1) = 0.0f;
    legacy.heights(2) = 0.0f;
    legacy.heights(3) = 1.0f;

    heightfield field;
    initialize_heightfield(
        field, 2, 2, 0.0f, 0.0f, 1.0f, -9.0f, 2);
    field.heights(0) = 0.0f;
    field.heights(1) = 0.0f;
    field.heights(2) = 0.0f;
    field.heights(3) = 1.0f;

    const vec3 root(0.0f, 4.0f, 0.25f);
    array1d<vec3> positions(2);
    positions(0) = root;
    positions(1) = vec3(1.0f, -8.0f, 0.25f);
    array1d<quat> rotations(2);
    rotations(0) = heading_positive_x();
    rotations(1) = heading_positive_x();

    terrain_centerline_snapshot legacy_snapshot = {};
    terrain_centerline_snapshot_compute(
        legacy_snapshot, legacy, root, positions, rotations);
    static const uint32_t point_x_bits[4] = {
        UINT32_C(0x3e800000), UINT32_C(0x3f000000),
        UINT32_C(0x3f400000), UINT32_C(0x3f800000),
    };
    static const uint32_t legacy_height_bits[4] = {
        UINT32_C(0x3d800000), UINT32_C(0x3e000000),
        UINT32_C(0x3e400000), UINT32_C(0x3e800000),
    };
    for (int i = 0; i < 4; ++i) {
        check_float_bits(
            legacy_snapshot.values[i], legacy_height_bits[i],
            "legacy v1 centerline query stays bit-identical");
        check_vec3_bits(
            legacy_snapshot.points[i], point_x_bits[i],
            legacy_height_bits[i], UINT32_C(0x3e800000),
            "legacy v1 centerline marker stays bit-identical");
    }

    terrain_centerline_snapshot snapshot = {};
    terrain_centerline_snapshot_compute_v2(
        snapshot, field, root, positions, rotations);
    float query[4] = {};
    terrain_centerline_query_v2(
        query, field, root, positions, rotations);
    const float base = heightfield_sample_v2(field, root.x, root.z);
    check_float_bits(base, UINT32_C(0x00000000),
                     "v2 centerline direct base sample");
    for (int i = 0; i < 4; ++i) {
        check_float_bits(snapshot.points[i].x, point_x_bits[i],
                         "v2 centerline marker x");
        check_float_bits(snapshot.points[i].z, UINT32_C(0x3e800000),
                         "v2 centerline marker z");
        const float direct = heightfield_sample_v2(
            field, snapshot.points[i].x, snapshot.points[i].z);
        const float direct_difference = static_cast<float>(
            static_cast<double>(direct) - static_cast<double>(base));
        check_float_bits(snapshot.points[i].y, float_bits(direct),
                         "v2 centerline marker matches direct sample");
        check_float_bits(snapshot.values[i], float_bits(direct_difference),
                         "v2 centerline value matches direct sample");
        check_float_bits(query[i], float_bits(direct_difference),
                         "v2 centerline query matches snapshot");
        check_float_bits(snapshot.values[i], UINT32_C(0x3e800000),
                         "v2 centerline uses triangular interpolation");
    }
    check(float_bits(snapshot.values[0]) !=
              float_bits(legacy_snapshot.values[0]),
          "v2 centerline fixture differs from bilinear v1");

    terrain_centerline_snapshot rejected = {};
    for (int i = 0; i < 4; ++i) rejected.values[i] = 9.0f;
    terrain_centerline_snapshot_compute_v2(
        rejected, legacy, root, positions, rotations);
    for (int i = 0; i < 4; ++i) {
        check_float_bits(rejected.values[i], UINT32_C(0x00000000),
                         "v2 centerline rejects legacy field version");
        check_vec3_bits(
            rejected.points[i], UINT32_C(0x00000000),
            UINT32_C(0x00000000), UINT32_C(0x3e800000),
            "v2 centerline rejection uses safe root marker");
    }
}

static void test_centerline_uses_root_skips_flat_repeats_and_latest_heading()
{
    const vec3 root(1.0f, 7.0f, 1.0f);
    array1d<vec3> positions(5);
    positions(0) = vec3(99.0f, 99.0f, 99.0f);
    positions(1) = vec3(1.0f, 70.0f, 1.0f);
    positions(2) = vec3(1.0f, -70.0f, 1.0f);
    positions(3) = vec3(1.5f, 500.0f, 1.0f);
    positions(4) = vec3(1.5f, -500.0f, 1.0f);
    array1d<quat> rotations(5);
    rotations(0) = heading_positive_x();
    rotations(1) = heading_positive_x();
    rotations(2) = heading_vertical();
    rotations(3) = heading_vertical();
    rotations(4) = heading_vertical();

    vec3 point = terrain_centerline_point_at_arc(
        root, positions, rotations, 0.25f);
    check_close(point.x, 1.25f, "root-authoritative centerline x");
    check_close(point.z, 1.0f, "root-authoritative centerline z");

    point = terrain_centerline_point_at_arc(root, positions, rotations, 0.75f);
    check_close(point.x, 1.75f, "degenerate heading inherits prior x");
    check_close(point.z, 1.0f, "degenerate heading inherits prior z");

    rotations(4) = quat(0.0f, 0.0f, 0.0f, 0.0f);
    point = terrain_centerline_point_at_arc(root, positions, rotations, 0.75f);
    check_close(point.x, 1.75f, "zero quaternion inherits prior x");
    check_close(point.z, 1.0f, "zero quaternion inherits prior z");

    rotations(4) = quat();
    point = terrain_centerline_point_at_arc(root, positions, rotations, 0.75f);
    check_close(point.x, 1.5f, "latest heading extension x");
    check_close(point.z, 1.25f, "latest heading extension z");

    positions(1) = root;
    positions(2) = root;
    positions(3) = root;
    positions(4) = root;
    for (int i = 0; i < rotations.size; ++i) {
        rotations(i) = quat(0.0f, 0.0f, 0.0f, 0.0f);
    }
    point = terrain_centerline_point_at_arc(root, positions, rotations, 1.0f);
    check_close(point.x, 1.0f, "all-degenerate heading fallback x");
    check_close(point.z, 2.0f, "all-degenerate heading fallback +Z");
}

static void test_centerline_query_uses_heightfield_exterior_at_boundary()
{
    heightfield field;
    initialize_heightfield(field, 5, 2, 0.0f, 0.0f, 0.25f, -3.0f);
    field.heights.set(2.0f);

    array1d<vec3> positions(2);
    positions(0) = vec3(0.75f, 0.0f, 0.0f);
    positions(1) = vec3(0.75f, 0.0f, 0.0f);
    array1d<quat> rotations(2);
    rotations(0) = heading_positive_x();
    rotations(1) = heading_positive_x();

    float query[4] = {};
    terrain_centerline_query(
        query, field, vec3(0.75f, 0.0f, 0.0f), positions, rotations);
    const float expected[4] = {0.0f, -5.0f, -5.0f, -5.0f};
    for (int i = 0; i < 4; ++i) {
        check_close(query[i], expected[i], "boundary exterior terrain query");
    }
}

static void check_zero_query(const float query[4], const char* message)
{
    for (int i = 0; i < 4; ++i) {
        check(terrain_float_is_finite(query[i]), message);
        check(query[i] == 0.0f, message);
    }
}

static void test_centerline_invalid_shapes_are_release_safe()
{
    heightfield field;
    initialize_heightfield(field, 2, 2, 0.0f, 0.0f, 1.0f, -1.0f);
    field.heights.zero();
    array1d<vec3> positions(2);
    array1d<quat> rotations(2);
    positions(0) = vec3();
    positions(1) = vec3();
    rotations(0) = quat();
    rotations(1) = quat();

    float query[4] = {9.0f, 9.0f, 9.0f, 9.0f};
    terrain_centerline_query(
        query, field, vec3(), slice1d<vec3>(0, NULL), rotations);
    check_zero_query(query, "empty positions query");

    for (int i = 0; i < 4; ++i) query[i] = 9.0f;
    terrain_centerline_query(
        query, field, vec3(), positions, slice1d<quat>(0, NULL));
    check_zero_query(query, "empty rotations query");

    for (int i = 0; i < 4; ++i) query[i] = 9.0f;
    terrain_centerline_query(
        query, field, vec3(), positions, slice1d<quat>(1, rotations.data));
    check_zero_query(query, "mismatched trajectory query");

    for (int i = 0; i < 4; ++i) query[i] = 9.0f;
    terrain_centerline_query(
        query, field, vec3(), slice1d<vec3>(1, NULL),
        slice1d<quat>(1, NULL));
    check_zero_query(query, "null trajectory storage query");

    terrain_centerline_query(
        NULL, field, vec3(), positions, rotations);
}

static void test_centerline_nonfinite_inputs_are_release_safe_under_fast_math()
{
    heightfield field;
    initialize_heightfield(field, 2, 2, 0.0f, 0.0f, 1.0f, -1.0f);
    field.heights(0) = 0.0f;
    field.heights(1) = 0.0f;
    field.heights(2) = 1.0f;
    field.heights(3) = 1.0f;
    array1d<vec3> positions(2);
    array1d<quat> rotations(2);
    positions(0) = vec3();
    positions(1) = vec3(0.0f, 0.0f, 1.0f);
    rotations(0) = quat();
    rotations(1) = quat();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();

    float query[4] = {9.0f, 9.0f, 9.0f, 9.0f};
    terrain_centerline_query(
        query, field, vec3(nan, 0.0f, 0.0f), positions, rotations);
    check_zero_query(query, "nonfinite root query");

    for (int i = 0; i < 4; ++i) query[i] = 9.0f;
    terrain_centerline_query(
        query, field, vec3(0.0f, nan, 0.0f), positions, rotations);
    check_zero_query(query, "nonfinite root vertical component query");

    positions(1).z = infinity;
    for (int i = 0; i < 4; ++i) query[i] = 9.0f;
    terrain_centerline_query(query, field, vec3(), positions, rotations);
    check_zero_query(query, "nonfinite position query");
    positions(1).z = 1.0f;

    positions(1).y = -infinity;
    for (int i = 0; i < 4; ++i) query[i] = 9.0f;
    terrain_centerline_query(query, field, vec3(), positions, rotations);
    check_zero_query(query, "nonfinite position vertical component query");
    positions(1).y = 0.0f;

    rotations(1).w = nan;
    for (int i = 0; i < 4; ++i) query[i] = 9.0f;
    terrain_centerline_query(query, field, vec3(), positions, rotations);
    check_zero_query(query, "nonfinite rotation query");
    rotations(1) = quat();

    const vec3 nan_point = terrain_centerline_point_at_arc(
        vec3(2.0f, 8.0f, 3.0f), positions, rotations, nan);
    check_close(nan_point.x, 2.0f, "nonfinite distance fallback x");
    check_close(nan_point.y, 0.0f, "nonfinite distance fallback y");
    check_close(nan_point.z, 3.0f, "nonfinite distance fallback z");
    const vec3 negative_point = terrain_centerline_point_at_arc(
        vec3(2.0f, 8.0f, 3.0f), positions, rotations, -1.0f);
    check_close(negative_point.x, 2.0f, "negative distance fallback x");
    check_close(negative_point.z, 3.0f, "negative distance fallback z");
    const vec3 positive_infinite_point = terrain_centerline_point_at_arc(
        vec3(2.0f, 8.0f, 3.0f), positions, rotations, infinity);
    check_close(
        positive_infinite_point.x, 2.0f,
        "positive infinite distance fallback x");
    check_close(
        positive_infinite_point.z, 3.0f,
        "positive infinite distance fallback z");
    const vec3 negative_infinite_point = terrain_centerline_point_at_arc(
        vec3(2.0f, 8.0f, 3.0f), positions, rotations, -infinity);
    check_close(
        negative_infinite_point.x, 2.0f,
        "negative infinite distance fallback x");
    check_close(
        negative_infinite_point.z, 3.0f,
        "negative infinite distance fallback z");
}

static void test_centerline_query_does_not_mutate_inputs()
{
    heightfield field;
    initialize_heightfield(field, 3, 3, -1.0f, -1.0f, 1.0f, -5.0f);
    for (int i = 0; i < field.heights.size; ++i) {
        field.heights(i) = static_cast<float>(i);
    }
    array1d<vec3> positions(3);
    positions(0) = vec3(10.0f, 11.0f, 12.0f);
    positions(1) = vec3(0.25f, 13.0f, 0.0f);
    positions(2) = vec3(0.50f, 14.0f, 0.0f);
    array1d<quat> rotations(3);
    rotations(0) = heading_positive_x();
    rotations(1) = heading_positive_x();
    rotations(2) = quat();
    const array1d<float> original_heights(field.heights);
    const array1d<vec3> original_positions(positions);
    const array1d<quat> original_rotations(rotations);
    const int original_nx = field.nx;
    const int original_nz = field.nz;
    const float original_origin_x = field.origin_x;
    const float original_origin_z = field.origin_z;
    const float original_cell_size = field.cell_size;
    const float original_exterior = field.exterior_height;

    float query[4] = {};
    terrain_centerline_query(
        query, field, vec3(0.0f, 0.0f, 0.0f), positions, rotations);
    (void)terrain_centerline_point_at_arc(
        vec3(0.0f, 0.0f, 0.0f), positions, rotations, 0.75f);

    check(field.nx == original_nx && field.nz == original_nz,
          "heightfield dimensions mutated");
    check(field.origin_x == original_origin_x &&
          field.origin_z == original_origin_z &&
          field.cell_size == original_cell_size &&
          field.exterior_height == original_exterior,
          "heightfield metadata mutated");
    for (int i = 0; i < field.heights.size; ++i) {
        check(field.heights(i) == original_heights(i),
              "heightfield values mutated");
    }
    for (int i = 0; i < positions.size; ++i) {
        check(positions(i).x == original_positions(i).x &&
              positions(i).y == original_positions(i).y &&
              positions(i).z == original_positions(i).z,
              "trajectory positions mutated");
        check(rotations(i).w == original_rotations(i).w &&
              rotations(i).x == original_rotations(i).x &&
              rotations(i).y == original_rotations(i).y &&
              rotations(i).z == original_rotations(i).z,
              "trajectory rotations mutated");
    }
}

static int probe_generated_artifacts(
    const char* sidecar_path,
    const char* heightfield_path,
    uint32_t expected_heightfield_version)
{
    char error[512] = {};
    terrain_feature_set features;
    const bool features_loaded = terrain_features_load(
        features, sidecar_path, error, static_cast<int>(sizeof(error)));
    check(features_loaded,
          error[0] != '\0' ? error : "generated terrain sidecar load");
    check(features.values.rows > 0,
          "generated terrain sidecar has rows");
    check(features.values.cols == 4,
          "generated terrain sidecar dimension count");

    heightfield field;
    error[0] = '\0';
    const bool heightfield_loaded = heightfield_load(
        field, heightfield_path, error, static_cast<int>(sizeof(error)));
    check(heightfield_loaded,
          error[0] != '\0' ? error : "generated terrain heightfield load");
    check(field.version == expected_heightfield_version,
          "generated terrain heightfield expected version");
    check(field.nx >= 2, "generated terrain heightfield nx");
    check(field.nz >= 2, "generated terrain heightfield nz");
    check(static_cast<int64_t>(field.heights.size) ==
              static_cast<int64_t>(field.nx) * field.nz,
          "generated terrain heightfield value count");

    const float center_x = field.origin_x +
        field.cell_size * static_cast<float>(field.nx - 1) * 0.5f;
    const float center_z = field.origin_z +
        field.cell_size * static_cast<float>(field.nz - 1) * 0.5f;
    check(terrain_float_is_finite(
              heightfield_sample_versioned(field, center_x, center_z)),
          "generated terrain center sample finite");
    check(terrain_float_is_finite(heightfield_sample_versioned(
              field, field.origin_x, field.origin_z)),
          "generated terrain origin sample finite");
    check(heightfield_sample_versioned(
              field, field.origin_x - field.cell_size, center_z) ==
              field.exterior_height,
          "generated terrain negative-x exterior sample");
    check(heightfield_sample_versioned(
              field, center_x,
              field.origin_z +
                  field.cell_size * static_cast<float>(field.nz)) ==
              field.exterior_height,
          "generated terrain positive-z exterior sample");
    return features.values.rows;
}

static void probe_support_and_scene(
    const char* support_path,
    const char* terrain_path,
    const char* walkability_path,
    int expected_frames)
{
    char error[512] = {};
    terrain_support_set support;
    check(terrain_support_load(support, support_path, expected_frames,
          error, sizeof(error)), error);
    check(support.values.rows == expected_frames && support.values.cols == 3,
          "published G1SP dimensions");
    heightfield field;
    check(heightfield_load(field, terrain_path, error, sizeof(error)), error);
    check(field.version == 2, "published scenes require G1HF/v2");
    walkability_grid grid;
    check(walkability_load(grid, walkability_path, field,
          error, sizeof(error)), error);
}

int main(int argc, char** argv)
{
    check(argc == 1 || argc == 6,
          "expected zero or five artifact arguments");
    test_controller_task6_terrain_ik_pipeline();
    test_sidecar_loads_valid_file();
    test_sidecar_rejects_every_truncation();
    test_sidecar_rejects_invalid_schema_sizes_and_values();
    test_sidecar_open_failure_is_actionable_and_transactional();

    test_heightfield_loads_and_samples_valid_file();
    test_heightfield_rejects_every_truncation();
    test_heightfield_rejects_invalid_schema_and_sizes();
    test_heightfield_rejects_nonfinite_metadata_and_heights();
    test_heightfield_open_failure_is_actionable_and_transactional();
    test_support_loader_is_strict_transactional_and_frame_exact();
    test_walkability_loader_is_strict_transactional_and_grid_exact();
    test_walkability_binary32_half_cell_parity();
    test_walkability_nearest_axis_preserves_large_endpoint();
    test_walkability_reason_names_are_stable();
    test_walkability_structural_gate_fails_closed();
    test_walkability_footprint_is_conservative_and_release_safe();
    test_walkability_footprint_caps_conservative_window();
    test_walkability_sweep_handles_clear_blocked_and_hostile_steps();
    test_walkability_checked_conversion_and_exact_sample_spacing();
    test_traversability_command_limits_safely_and_recovers();
    test_blocked_planar_stop_preserves_vertical_bits();
    test_traversability_clip_is_planar_and_bit_preserving();
    test_walkability_guard_reaches_safe_stop();
    test_payload_allocation_failures_are_actionable_and_transactional();
    test_f32_helpers_match_one_round_producer_operations();
    test_heightfield_versions_preserve_v1_and_use_v2_triangles();
    test_v1_coordinate_arithmetic_remains_literal();
    test_v2_awkward_python_byte_and_query_oracle();
    test_v2_diagonal_decision_stays_binary64();
    test_v2_edges_grid_lines_and_diagonal_tie();
    test_v2_normals_extremes_and_ftz_outputs();
    test_v2_runtime_node_precise_rounding_thresholds();
    test_v2_rejects_encoded_domain_and_axis_failures();
    test_v2_query_domain_and_invalid_structures_are_safe();
    test_v2_axis_validator_vectors_and_seeded_property();
    test_heightfield_rejects_float_rounded_upper_grid_index();
    test_stationary_centerline_extends_current_heading();
    test_straight_step_query_is_ground_relative_on_elevated_base();
    test_curved_query_matches_python_geometric_arc_fixture();
    test_centerline_snapshot_keeps_query_and_markers_in_one_state();
    test_centerline_walkability_latches_wall_and_ramp();
    test_centerline_walkability_preserves_clear_classes_and_recovers();
    test_centerline_walkability_invalid_inputs_are_transactional();
    test_centerline_walkability_separates_query_and_footprint_roots();
    test_centerline_walkability_latches_out_of_bounds_segment();
    test_v2_centerline_uses_checked_triangular_height_samples();
    test_centerline_uses_root_skips_flat_repeats_and_latest_heading();
    test_centerline_query_uses_heightfield_exterior_at_boundary();
    test_centerline_invalid_shapes_are_release_safe();
    test_centerline_nonfinite_inputs_are_release_safe_under_fast_math();
    test_centerline_query_does_not_mutate_inputs();
    if (argc == 6) {
        uint32_t expected_heightfield_version = 0;
        if (strcmp(argv[3], "1") == 0) {
            expected_heightfield_version = 1;
        } else if (strcmp(argv[3], "2") == 0) {
            expected_heightfield_version = 2;
        } else {
            check(false, "expected G1HF version text exactly 1 or 2");
        }
        const int expected_frames = probe_generated_artifacts(
            argv[1], argv[2], expected_heightfield_version);
        probe_support_and_scene(
            argv[4], argv[2], argv[5], expected_frames);
    }
    return 0;
}
