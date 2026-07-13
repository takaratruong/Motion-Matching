#include "terrain_runtime.h"

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

static void probe_generated_artifacts(
    const char* sidecar_path, const char* heightfield_path)
{
    char error[512] = {};
    terrain_feature_set features;
    assert(terrain_features_load(
        features, sidecar_path, error, static_cast<int>(sizeof(error))));
    assert(features.values.rows == 459682);
    assert(features.values.cols == 4);

    heightfield field;
    assert(heightfield_load(
        field, heightfield_path, error, static_cast<int>(sizeof(error))));
    assert(field.nx == 261);
    assert(field.nz == 228);
    assert(field.heights.size == field.nx * field.nz);

    const float center_x = field.origin_x +
        field.cell_size * static_cast<float>(field.nx - 1) * 0.5f;
    const float center_z = field.origin_z +
        field.cell_size * static_cast<float>(field.nz - 1) * 0.5f;
    assert(isfinite(heightfield_sample(field, center_x, center_z)));
    assert(isfinite(heightfield_sample(
        field, field.origin_x, field.origin_z)));
    assert(heightfield_sample(
        field, field.origin_x - field.cell_size, center_z) ==
        field.exterior_height);
    assert(heightfield_sample(
        field, center_x,
        field.origin_z + field.cell_size * static_cast<float>(field.nz)) ==
        field.exterior_height);
}

int main(int argc, char** argv)
{
    assert(argc == 1 || argc == 3);
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
    if (argc == 3) {
        probe_generated_artifacts(argv[1], argv[2]);
    }
    return 0;
}
