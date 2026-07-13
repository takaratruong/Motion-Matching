#pragma once

#include <stdlib.h>

#include "array.h"
#include "vec.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <utility>

struct terrain_feature_set
{
    array2d<float> values;
};

struct heightfield
{
    int nx = 0;
    int nz = 0;
    float origin_x = 0.0f;
    float origin_z = 0.0f;
    float cell_size = 0.0f;
    float exterior_height = 0.0f;
    array1d<float> heights;
};

static inline bool terrain_error(
    char* out, int cap, const char* format, ...)
{
    if (out != NULL && cap > 0) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(out, static_cast<size_t>(cap), format, arguments);
        va_end(arguments);
    }
    return false;
}

static inline bool terrain_read_exact(FILE* file, void* destination, size_t size)
{
    return size == 0 || fread(destination, 1, size, file) == size;
}

static inline uint32_t terrain_decode_u32_le(const unsigned char* bytes)
{
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) |
           (static_cast<uint32_t>(bytes[3]) << 24);
}

static inline float terrain_decode_float_le(const unsigned char* bytes)
{
    const uint32_t bits = terrain_decode_u32_le(bytes);
    float value = 0.0f;
    static_assert(sizeof(value) == sizeof(bits),
                  "terrain artifacts require 32-bit floats");
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline bool terrain_float_is_finite(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static inline bool terrain_host_is_little_endian()
{
    const uint16_t value = 1;
    return *reinterpret_cast<const unsigned char*>(&value) == 1;
}

static inline void terrain_decode_float_array_le(float* values, size_t count)
{
    if (terrain_host_is_little_endian()) {
        return;
    }

    unsigned char* bytes = reinterpret_cast<unsigned char*>(values);
    for (size_t i = 0; i < count; ++i) {
        unsigned char* value = bytes + i * sizeof(float);
        const unsigned char byte0 = value[0];
        const unsigned char byte1 = value[1];
        value[0] = value[3];
        value[1] = value[2];
        value[2] = byte1;
        value[3] = byte0;
    }
}

static inline bool terrain_size_multiply(
    size_t left, size_t right, size_t& result)
{
    if (right != 0 && left > SIZE_MAX / right) {
        return false;
    }
    result = left * right;
    return true;
}

static inline bool terrain_size_add(size_t left, size_t right, size_t& result)
{
    if (left > SIZE_MAX - right) {
        return false;
    }
    result = left + right;
    return true;
}

static inline bool terrain_file_size(FILE* file, size_t& size)
{
    if (fseek(file, 0, SEEK_END) != 0) {
        return false;
    }
    const long end = ftell(file);
    if (end < 0 || static_cast<uintmax_t>(end) >
                       static_cast<uintmax_t>(SIZE_MAX)) {
        return false;
    }
    size = static_cast<size_t>(end);
    return fseek(file, 0, SEEK_SET) == 0;
}

static inline bool terrain_finish_read(
    FILE* file, const char* path, char* error, int error_capacity)
{
    const int extra = fgetc(file);
    if (extra != EOF) {
        fclose(file);
        return terrain_error(
            error, error_capacity, "%s: trailing bytes", path);
    }
    if (ferror(file)) {
        fclose(file);
        return terrain_error(
            error, error_capacity, "%s: failed while reading", path);
    }
    fclose(file);
    return true;
}

static inline bool terrain_features_load(
    terrain_feature_set& out,
    const char* path,
    char* error,
    int error_capacity)
{
    const char* display_path = path != NULL ? path : "<null>";
    if (path == NULL || path[0] == '\0') {
        return terrain_error(
            error, error_capacity, "%s: invalid path", display_path);
    }

    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return terrain_error(
            error, error_capacity, "%s: cannot open (%s)",
            path, strerror(errno));
    }

    size_t file_length = 0;
    if (!terrain_file_size(file, file_length)) {
        fclose(file);
        return terrain_error(
            error, error_capacity, "%s: cannot determine file length", path);
    }

    static const size_t header_size = 16;
    if (file_length < header_size) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: truncated G1TF header (expected %zu bytes, got %zu)",
            path, header_size, file_length);
    }

    unsigned char header[header_size];
    if (!terrain_read_exact(file, header, sizeof(header))) {
        fclose(file);
        return terrain_error(
            error, error_capacity, "%s: truncated G1TF header", path);
    }

    if (memcmp(header, "G1TF", 4) != 0) {
        fclose(file);
        return terrain_error(
            error, error_capacity, "%s: invalid G1TF magic", path);
    }

    const uint32_t version = terrain_decode_u32_le(header + 4);
    const uint32_t frames = terrain_decode_u32_le(header + 8);
    const uint32_t dimensions = terrain_decode_u32_le(header + 12);
    if (version != 1) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: unsupported G1TF version %u (expected 1)",
            path, static_cast<unsigned>(version));
    }
    if (dimensions != 4) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: invalid G1TF dimension %u (expected 4)",
            path, static_cast<unsigned>(dimensions));
    }
    if (frames == 0 || frames > static_cast<uint32_t>(INT_MAX / 4)) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: invalid G1TF frame count %u (must be 1..%d)",
            path, static_cast<unsigned>(frames), INT_MAX / 4);
    }

    size_t value_count = 0;
    size_t payload_size = 0;
    size_t expected_length = 0;
    if (!terrain_size_multiply(
            static_cast<size_t>(frames),
            static_cast<size_t>(dimensions), value_count) ||
        !terrain_size_multiply(value_count, sizeof(float), payload_size) ||
        !terrain_size_add(header_size, payload_size, expected_length)) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: G1TF byte length overflows this platform", path);
    }
    if (file_length < expected_length) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: truncated G1TF values (expected %zu bytes, got %zu)",
            path, expected_length, file_length);
    }
    if (file_length > expected_length) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: trailing bytes after G1TF values (expected %zu bytes, got %zu)",
            path, expected_length, file_length);
    }

    terrain_feature_set loaded;
    loaded.values.resize(
        static_cast<int>(frames), static_cast<int>(dimensions));
    if (!terrain_read_exact(file, loaded.values.data, payload_size)) {
        fclose(file);
        return terrain_error(
            error, error_capacity, "%s: truncated G1TF values", path);
    }
    terrain_decode_float_array_le(loaded.values.data, value_count);
    for (size_t i = 0; i < value_count; ++i) {
        if (!terrain_float_is_finite(loaded.values.data[i])) {
            fclose(file);
            return terrain_error(
                error, error_capacity,
                "%s: G1TF values must all be finite (index %zu)", path, i);
        }
    }

    if (!terrain_finish_read(file, path, error, error_capacity)) {
        return false;
    }

    std::swap(out.values.rows, loaded.values.rows);
    std::swap(out.values.cols, loaded.values.cols);
    std::swap(out.values.data, loaded.values.data);
    return true;
}

static inline bool heightfield_load(
    heightfield& out,
    const char* path,
    char* error,
    int error_capacity)
{
    const char* display_path = path != NULL ? path : "<null>";
    if (path == NULL || path[0] == '\0') {
        return terrain_error(
            error, error_capacity, "%s: invalid path", display_path);
    }

    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return terrain_error(
            error, error_capacity, "%s: cannot open (%s)",
            path, strerror(errno));
    }

    size_t file_length = 0;
    if (!terrain_file_size(file, file_length)) {
        fclose(file);
        return terrain_error(
            error, error_capacity, "%s: cannot determine file length", path);
    }

    static const size_t header_size = 32;
    if (file_length < header_size) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: truncated G1HF header (expected %zu bytes, got %zu)",
            path, header_size, file_length);
    }

    unsigned char header[header_size];
    if (!terrain_read_exact(file, header, sizeof(header))) {
        fclose(file);
        return terrain_error(
            error, error_capacity, "%s: truncated G1HF header", path);
    }

    if (memcmp(header, "G1HF", 4) != 0) {
        fclose(file);
        return terrain_error(
            error, error_capacity, "%s: invalid G1HF magic", path);
    }

    const uint32_t version = terrain_decode_u32_le(header + 4);
    const uint32_t nx = terrain_decode_u32_le(header + 8);
    const uint32_t nz = terrain_decode_u32_le(header + 12);
    const float origin_x = terrain_decode_float_le(header + 16);
    const float origin_z = terrain_decode_float_le(header + 20);
    const float cell_size = terrain_decode_float_le(header + 24);
    const float exterior_height = terrain_decode_float_le(header + 28);

    if (version != 1) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: unsupported G1HF version %u (expected 1)",
            path, static_cast<unsigned>(version));
    }
    if (nx < 2 || nz < 2 ||
        nx > static_cast<uint32_t>(INT_MAX) / nz) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: invalid G1HF grid dimensions %u x %u",
            path, static_cast<unsigned>(nx), static_cast<unsigned>(nz));
    }
    if (!terrain_float_is_finite(origin_x) ||
        !terrain_float_is_finite(origin_z)) {
        fclose(file);
        return terrain_error(
            error, error_capacity, "%s: G1HF origin must be finite", path);
    }
    if (!terrain_float_is_finite(cell_size) || cell_size <= 0.0f) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: G1HF cell size must be finite and positive", path);
    }
    if (!terrain_float_is_finite(exterior_height)) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: G1HF exterior height must be finite", path);
    }

    size_t height_count = 0;
    size_t payload_size = 0;
    size_t expected_length = 0;
    if (!terrain_size_multiply(
            static_cast<size_t>(nx), static_cast<size_t>(nz), height_count) ||
        height_count > static_cast<size_t>(INT_MAX) ||
        !terrain_size_multiply(height_count, sizeof(float), payload_size) ||
        !terrain_size_add(header_size, payload_size, expected_length)) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: G1HF byte length overflows this platform", path);
    }
    if (file_length < expected_length) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: truncated G1HF heights (expected %zu bytes, got %zu)",
            path, expected_length, file_length);
    }
    if (file_length > expected_length) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: trailing bytes after G1HF heights (expected %zu bytes, got %zu)",
            path, expected_length, file_length);
    }

    heightfield loaded;
    loaded.nx = static_cast<int>(nx);
    loaded.nz = static_cast<int>(nz);
    loaded.origin_x = origin_x;
    loaded.origin_z = origin_z;
    loaded.cell_size = cell_size;
    loaded.exterior_height = exterior_height;
    loaded.heights.resize(static_cast<int>(height_count));
    if (!terrain_read_exact(file, loaded.heights.data, payload_size)) {
        fclose(file);
        return terrain_error(
            error, error_capacity, "%s: truncated G1HF heights", path);
    }
    terrain_decode_float_array_le(loaded.heights.data, height_count);
    for (size_t i = 0; i < height_count; ++i) {
        if (!terrain_float_is_finite(loaded.heights.data[i])) {
            fclose(file);
            return terrain_error(
                error, error_capacity,
                "%s: G1HF heights must all be finite (index %zu)", path, i);
        }
    }

    if (!terrain_finish_read(file, path, error, error_capacity)) {
        return false;
    }

    std::swap(out.nx, loaded.nx);
    std::swap(out.nz, loaded.nz);
    std::swap(out.origin_x, loaded.origin_x);
    std::swap(out.origin_z, loaded.origin_z);
    std::swap(out.cell_size, loaded.cell_size);
    std::swap(out.exterior_height, loaded.exterior_height);
    std::swap(out.heights.size, loaded.heights.size);
    std::swap(out.heights.data, loaded.heights.data);
    return true;
}

static inline float heightfield_sample(
    const heightfield& field, float x, float z)
{
    if (!terrain_float_is_finite(x) || !terrain_float_is_finite(z)) {
        return field.exterior_height;
    }

    const float grid_x = (x - field.origin_x) / field.cell_size;
    const float grid_z = (z - field.origin_z) / field.cell_size;
    if (!terrain_float_is_finite(grid_x) ||
        !terrain_float_is_finite(grid_z) ||
        grid_x < 0.0f || grid_z < 0.0f ||
        grid_x > static_cast<float>(field.nx - 1) ||
        grid_z > static_cast<float>(field.nz - 1)) {
        return field.exterior_height;
    }

    const int x0 = static_cast<int>(floorf(grid_x));
    const int z0 = static_cast<int>(floorf(grid_z));
    if (x0 < 0 || x0 >= field.nx || z0 < 0 || z0 >= field.nz) {
        return field.exterior_height;
    }
    const int x1 = x0 < field.nx - 1 ? x0 + 1 : x0;
    const int z1 = z0 < field.nz - 1 ? z0 + 1 : z0;
    const float tx = grid_x - static_cast<float>(x0);
    const float tz = grid_z - static_cast<float>(z0);
    const float row0 = lerpf(
        field.heights(z0 * field.nx + x0),
        field.heights(z0 * field.nx + x1), tx);
    const float row1 = lerpf(
        field.heights(z1 * field.nx + x0),
        field.heights(z1 * field.nx + x1), tx);
    return lerpf(row0, row1, tz);
}
