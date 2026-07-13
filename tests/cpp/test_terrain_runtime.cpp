#include "terrain_runtime.h"
#include "quat.h"

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <limits>
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
    float exterior_height)
{
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

static void write_prefix(
    const char* path, const byte_buffer& payload, size_t length)
{
    assert(length <= payload.size());
    FILE* file = fopen(path, "wb");
    assert(file != NULL);
    if (length != 0) {
        assert(fwrite(payload.data(), 1, length, file) == length);
    }
    assert(fclose(file) == 0);
}

static void write_payload(const char* path, const byte_buffer& payload)
{
    write_prefix(path, payload, payload.size());
}

static void assert_error(
    const char* error, const char* path, const char* expected_reason)
{
    assert(error[0] != '\0');
    assert(strstr(error, path) != NULL);
    assert(strstr(error, expected_reason) != NULL);
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
    assert(!terrain_features_load(
        destination, path, error, static_cast<int>(sizeof(error))));
    assert_error(error, path, expected_reason);
    assert(destination.values.rows == 2);
    assert(destination.values.cols == 4);
    for (int i = 0; i < destination.values.rows; ++i) {
        for (int j = 0; j < destination.values.cols; ++j) {
            assert(destination.values(i, j) == 100.0f + i * 10.0f + j);
        }
    }
}

static void expect_heightfield_rejected(
    const char* path, const char* expected_reason)
{
    heightfield destination;
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
    assert(!heightfield_load(
        destination, path, error, static_cast<int>(sizeof(error))));
    assert_error(error, path, expected_reason);
    assert(destination.nx == 3);
    assert(destination.nz == 2);
    assert(destination.origin_x == 10.0f);
    assert(destination.origin_z == 20.0f);
    assert(destination.cell_size == 0.25f);
    assert(destination.exterior_height == -5.0f);
    assert(destination.heights.size == 6);
    for (int i = 0; i < destination.heights.size; ++i) {
        assert(destination.heights(i) == 200.0f + i);
    }
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
    assert(terrain_features_load(
        features, path, error, static_cast<int>(sizeof(error))));
    assert(features.values.rows == 2);
    assert(features.values.cols == 4);
    for (int i = 0; i < 8; ++i) {
        assert(features.values.data[i] == static_cast<float>(i));
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

static byte_buffer valid_heightfield_payload()
{
    return make_heightfield(
        1, 2, 2, 0.0f, 0.0f, 1.0f, -1.0f,
        {0.0f, 1.0f, 2.0f, 3.0f});
}

static void test_heightfield_loads_and_samples_valid_file()
{
    const char* path = "/tmp/test_g1hf_valid.bin";
    write_payload(path, valid_heightfield_payload());

    heightfield field;
    field.nx = 4;
    field.nz = 1;
    field.origin_x = 99.0f;
    field.origin_z = 98.0f;
    field.cell_size = 97.0f;
    field.exterior_height = 96.0f;
    field.heights.resize(4);
    field.heights.set(-99.0f);
    char error[256] = {};
    assert(heightfield_load(
        field, path, error, static_cast<int>(sizeof(error))));
    assert(field.nx == 2);
    assert(field.nz == 2);
    assert(field.origin_x == 0.0f);
    assert(field.origin_z == 0.0f);
    assert(field.cell_size == 1.0f);
    assert(field.exterior_height == -1.0f);
    assert(field.heights.size == 4);

    assert(heightfield_sample(field, 0.0f, 0.0f) == 0.0f);
    assert(heightfield_sample(field, 1.0f, 0.0f) == 1.0f);
    assert(heightfield_sample(field, 0.0f, 1.0f) == 2.0f);
    assert(heightfield_sample(field, 1.0f, 1.0f) == 3.0f);
    assert(fabsf(heightfield_sample(field, 0.5f, 0.5f) - 1.5f) < 1e-6f);
    assert(fabsf(heightfield_sample(field, 1.0f, 0.5f) - 2.0f) < 1e-6f);

    assert(heightfield_sample(field, -0.001f, 0.0f) == -1.0f);
    assert(heightfield_sample(field, 0.0f, -0.001f) == -1.0f);
    assert(heightfield_sample(field, 1.001f, 0.0f) == -1.0f);
    assert(heightfield_sample(field, 0.0f, 1.001f) == -1.0f);
    assert(heightfield_sample(field, FLT_MAX, 0.0f) == -1.0f);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    assert(heightfield_sample(field, nan, 0.0f) == -1.0f);
    assert(heightfield_sample(field, 0.0f, nan) == -1.0f);
    assert(heightfield_sample(field, infinity, 0.0f) == -1.0f);
    assert(heightfield_sample(field, 0.0f, -infinity) == -1.0f);
}

static void test_heightfield_rejects_every_truncation()
{
    const char* path = "/tmp/test_g1hf_truncated.bin";
    const byte_buffer payload = valid_heightfield_payload();
    for (size_t length = 0; length < payload.size(); ++length) {
        write_prefix(path, payload, length);
        expect_heightfield_rejected(path, "truncated");
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
        path, make_heightfield(2, 2, 2, 0, 0, 1, -1, heights));
    expect_heightfield_rejected(path, "version");

    write_payload(path, make_heightfield(1, 1, 2, 0, 0, 1, -1, {}));
    expect_heightfield_rejected(path, "grid dimensions");

    write_payload(path, make_heightfield(1, 2, 1, 0, 0, 1, -1, {}));
    expect_heightfield_rejected(path, "grid dimensions");

    write_payload(path, make_heightfield(1, 0, 2, 0, 0, 1, -1, {}));
    expect_heightfield_rejected(path, "grid dimensions");

    write_payload(path, make_heightfield(1, 2, 0, 0, 0, 1, -1, {}));
    expect_heightfield_rejected(path, "grid dimensions");

    write_payload(path, make_heightfield(1, 65536, 32768, 0, 0, 1, -1, {}));
    expect_heightfield_rejected(path, "grid dimensions");

    write_payload(path, make_heightfield(1, UINT32_MAX, 2, 0, 0, 1, -1, {}));
    expect_heightfield_rejected(path, "grid dimensions");

    write_payload(path, make_heightfield(1, 65535, 32768, 0, 0, 1, -1, {}));
    expect_heightfield_rejected(path, "truncated");

    payload = valid_heightfield_payload();
    payload.push_back(0x7f);
    write_payload(path, payload);
    expect_heightfield_rejected(path, "trailing");
}

static void test_heightfield_rejects_nonfinite_metadata_and_heights()
{
    const char* path = "/tmp/test_g1hf_numeric.bin";
    const std::vector<float> heights = {0.0f, 1.0f, 2.0f, 3.0f};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();

    write_payload(
        path, make_heightfield(1, 2, 2, nan, 0, 1, -1, heights));
    expect_heightfield_rejected(path, "origin");

    write_payload(
        path, make_heightfield(1, 2, 2, 0, infinity, 1, -1, heights));
    expect_heightfield_rejected(path, "origin");

    write_payload(
        path, make_heightfield(1, 2, 2, 0, 0, 0, -1, heights));
    expect_heightfield_rejected(path, "cell size");

    write_payload(
        path, make_heightfield(1, 2, 2, 0, 0, -1, -1, heights));
    expect_heightfield_rejected(path, "cell size");

    write_payload(
        path, make_heightfield(1, 2, 2, 0, 0, nan, -1, heights));
    expect_heightfield_rejected(path, "cell size");

    write_payload(
        path, make_heightfield(1, 2, 2, 0, 0, infinity, -1, heights));
    expect_heightfield_rejected(path, "cell size");

    write_payload(
        path, make_heightfield(1, 2, 2, 0, 0, 1, nan, heights));
    expect_heightfield_rejected(path, "exterior height");

    write_payload(
        path, make_heightfield(1, 2, 2, 0, 0, 1, infinity, heights));
    expect_heightfield_rejected(path, "exterior height");

    write_payload(
        path, make_heightfield(1, 2, 2, 0, 0, 1, -1,
                               {0.0f, nan, 2.0f, 3.0f}));
    expect_heightfield_rejected(path, "finite");

    write_payload(
        path, make_heightfield(1, 2, 2, 0, 0, 1, -1,
                               {0.0f, 1.0f, -infinity, 3.0f}));
    expect_heightfield_rejected(path, "finite");
}

static void test_heightfield_open_failure_is_actionable_and_transactional()
{
    const char* path = "/tmp/test_g1hf_does_not_exist.bin";
    remove(path);
    expect_heightfield_rejected(path, "cannot open");
}

static void test_heightfield_rejects_float_rounded_upper_grid_index()
{
    heightfield field;
    field.nx = 16777220;
    field.nz = 2;
    field.origin_x = 0.0f;
    field.origin_z = 0.0f;
    field.cell_size = 1.0f;
    field.exterior_height = -99.0f;
    field.heights.resize(field.nx * field.nz);
    field.heights.zero();

    const float rounded_upper_x = static_cast<float>(field.nx - 1);
    assert(static_cast<double>(rounded_upper_x) >
           static_cast<double>(field.nx - 1));
    const volatile float outward_sample =
        heightfield_sample(field, rounded_upper_x, 1.0f);
    assert(outward_sample == field.exterior_height);

    const float inward_x = nextafterf(rounded_upper_x, -INFINITY);
    const int inward_x0 = static_cast<int>(floorf(inward_x));
    assert(inward_x0 >= 0 && inward_x0 + 1 < field.nx);
    const volatile float inward_sample =
        heightfield_sample(field, inward_x, 1.0f);
    assert(inward_sample == 0.0f);
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

static void probe_generated_artifacts(
    const char* sidecar_path, const char* heightfield_path)
{
    char error[512] = {};
    terrain_feature_set features;
    const bool features_loaded = terrain_features_load(
        features, sidecar_path, error, static_cast<int>(sizeof(error)));
    check(features_loaded,
          error[0] != '\0' ? error : "generated terrain sidecar load");
    check(features.values.rows == 459682,
          "generated terrain sidecar frame count");
    check(features.values.cols == 4,
          "generated terrain sidecar dimension count");

    heightfield field;
    error[0] = '\0';
    const bool heightfield_loaded = heightfield_load(
        field, heightfield_path, error, static_cast<int>(sizeof(error)));
    check(heightfield_loaded,
          error[0] != '\0' ? error : "generated terrain heightfield load");
    check(field.nx == 265, "generated terrain heightfield nx");
    check(field.nz == 399, "generated terrain heightfield nz");
    check(field.heights.size == field.nx * field.nz,
          "generated terrain heightfield value count");

    const float center_x = field.origin_x +
        field.cell_size * static_cast<float>(field.nx - 1) * 0.5f;
    const float center_z = field.origin_z +
        field.cell_size * static_cast<float>(field.nz - 1) * 0.5f;
    check(terrain_float_is_finite(
              heightfield_sample(field, center_x, center_z)),
          "generated terrain center sample finite");
    check(terrain_float_is_finite(heightfield_sample(
              field, field.origin_x, field.origin_z)),
          "generated terrain origin sample finite");
    check(heightfield_sample(
              field, field.origin_x - field.cell_size, center_z) ==
              field.exterior_height,
          "generated terrain negative-x exterior sample");
    check(heightfield_sample(
              field, center_x,
              field.origin_z +
                  field.cell_size * static_cast<float>(field.nz)) ==
              field.exterior_height,
          "generated terrain positive-z exterior sample");
}

int main(int argc, char** argv)
{
    check(argc == 1 || argc == 3,
          "expected zero or two artifact path arguments");
    test_sidecar_loads_valid_file();
    test_sidecar_rejects_every_truncation();
    test_sidecar_rejects_invalid_schema_sizes_and_values();
    test_sidecar_open_failure_is_actionable_and_transactional();

    test_heightfield_loads_and_samples_valid_file();
    test_heightfield_rejects_every_truncation();
    test_heightfield_rejects_invalid_schema_and_sizes();
    test_heightfield_rejects_nonfinite_metadata_and_heights();
    test_heightfield_open_failure_is_actionable_and_transactional();
    test_heightfield_rejects_float_rounded_upper_grid_index();
    test_stationary_centerline_extends_current_heading();
    test_straight_step_query_is_ground_relative_on_elevated_base();
    test_curved_query_matches_python_geometric_arc_fixture();
    test_centerline_uses_root_skips_flat_repeats_and_latest_heading();
    test_centerline_query_uses_heightfield_exterior_at_boundary();
    test_centerline_invalid_shapes_are_release_safe();
    test_centerline_nonfinite_inputs_are_release_safe_under_fast_math();
    test_centerline_query_does_not_mutate_inputs();
    if (argc == 3) {
        probe_generated_artifacts(argv[1], argv[2]);
    }
    return 0;
}
