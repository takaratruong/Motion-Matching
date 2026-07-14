#pragma once

#include <stdlib.h>

#include "array.h"
#include "quat.h"

#include <errno.h>
#include <float.h>
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
    // Appended so every historical v1 member keeps its original offset.
    uint32_t version = 0;
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

static inline bool terrain_double_is_finite(double value)
{
    uint64_t bits = 0;
    static_assert(sizeof(value) == sizeof(bits),
                  "terrain runtime requires 64-bit doubles");
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) !=
           UINT64_C(0x7ff0000000000000);
}

static inline uint32_t terrain_float_bits(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static inline bool terrain_float_is_normal_or_positive_zero(float value)
{
    const uint32_t bits = terrain_float_bits(value);
    const uint32_t exponent = bits & UINT32_C(0x7f800000);
    return bits == 0 ||
           (exponent != 0 && exponent != UINT32_C(0x7f800000));
}

static inline bool terrain_float_is_positive_normal(float value)
{
    const uint32_t bits = terrain_float_bits(value);
    const uint32_t exponent = bits & UINT32_C(0x7f800000);
    return (bits & UINT32_C(0x80000000)) == 0 &&
           exponent != 0 && exponent != UINT32_C(0x7f800000);
}

static inline bool terrain_float_is_normal_or_zero_query(float value)
{
    const uint32_t magnitude =
        terrain_float_bits(value) & UINT32_C(0x7fffffff);
    const uint32_t exponent = magnitude & UINT32_C(0x7f800000);
    return magnitude == 0 ||
           (exponent != 0 && exponent != UINT32_C(0x7f800000));
}

static inline float terrain_runtime_canonicalize_output(float value)
{
    const uint32_t bits = terrain_float_bits(value);
    if ((bits & UINT32_C(0x7f800000)) == 0) {
        return 0.0f;
    }
    return value;
}

static inline bool terrain_v2_runtime_node_from_double(
    double source, float& runtime)
{
    if (!terrain_double_is_finite(source)) {
        return false;
    }

    uint64_t source_bits = 0;
    memcpy(&source_bits, &source, sizeof(source_bits));
    const bool negative =
        (source_bits & UINT64_C(0x8000000000000000)) != 0;
    const uint64_t magnitude_bits =
        source_bits & UINT64_C(0x7fffffffffffffff);
    double magnitude = 0.0;
    memcpy(&magnitude, &magnitude_bits, sizeof(magnitude));

    // These are the exact round-to-nearest-even ties around the binary32
    // subnormal interval.  Classify in binary64 before a production
    // -ffast-math conversion can flush a derived subnormal.
    const double zero_tie = 0x1p-150;
    const double minimum_normal_tie = 0x1p-126 - 0x1p-150;
    if (magnitude <= zero_tie) {
        runtime = 0.0f;
        return true;
    }
    if (magnitude < minimum_normal_tie) {
        return false;
    }
    if (magnitude < 0x1p-126) {
        uint32_t rounded_bits = UINT32_C(0x00800000);
        // Avoid all binary64-to-binary32 arithmetic in this interval: the
        // exact IEEE result is minimum normal, while FTZ hardware conversion
        // may first produce and then flush a subnormal.
        if (negative) {
            rounded_bits |= UINT32_C(0x80000000);
        }
        memcpy(&runtime, &rounded_bits, sizeof(runtime));
        return true;
    }

    const volatile double materialized_source = source;
    const volatile float rounded =
        static_cast<float>(materialized_source);
    runtime = rounded;
    return terrain_float_is_normal_or_positive_zero(runtime);
}

static inline bool terrain_v2_axis_node(
    float origin,
    uint32_t index,
    float cell_size,
    double& source,
    float& runtime)
{
    const volatile double product =
        static_cast<double>(index) * static_cast<double>(cell_size);
    const volatile double coordinate =
        static_cast<double>(origin) + product;
    source = coordinate;
    return terrain_v2_runtime_node_from_double(source, runtime);
}

static inline double terrain_v2_source_maximum_inward_spacing(
    double first, double last)
{
    if (first >= 0.0) {
        const volatile double previous = nextafter(last, -INFINITY);
        const volatile double spacing = last - previous;
        return spacing;
    }
    if (last <= 0.0) {
        const volatile double next = nextafter(first, INFINITY);
        const volatile double spacing = next - first;
        return spacing;
    }
    const volatile double next = nextafter(first, INFINITY);
    const volatile double previous = nextafter(last, -INFINITY);
    const volatile double first_spacing = next - first;
    const volatile double last_spacing = last - previous;
    return first_spacing > last_spacing ? first_spacing : last_spacing;
}

static inline double terrain_v2_float_inward_spacing(float endpoint)
{
    const uint32_t magnitude =
        terrain_float_bits(endpoint) & UINT32_C(0x7fffffff);
    const int encoded_exponent =
        static_cast<int>((magnitude >> 23) & UINT32_C(0xff));
    const uint32_t significand = magnitude & UINT32_C(0x007fffff);
    if (significand == 0) {
        if (encoded_exponent == 1) {
            return ldexp(1.0, -149);
        }
        return ldexp(1.0, encoded_exponent - 127 - 24);
    }
    return ldexp(1.0, encoded_exponent - 127 - 23);
}

static inline double terrain_v2_runtime_maximum_inward_spacing(
    float first, float last)
{
    if (first >= 0.0f) {
        return terrain_v2_float_inward_spacing(last);
    }
    if (last <= 0.0f) {
        return terrain_v2_float_inward_spacing(first);
    }
    const double first_spacing = terrain_v2_float_inward_spacing(first);
    const double last_spacing = terrain_v2_float_inward_spacing(last);
    return first_spacing > last_spacing ? first_spacing : last_spacing;
}

static inline bool terrain_v2_runtime_axis_is_valid(
    float origin, uint32_t count, float cell_size)
{
    if (count < 2 ||
        !terrain_float_is_normal_or_positive_zero(origin) ||
        !terrain_float_is_positive_normal(cell_size)) {
        return false;
    }

    double first_source = 0.0;
    double second_source = 0.0;
    double penultimate_source = 0.0;
    double last_source = 0.0;
    float first_runtime = 0.0f;
    float second_runtime = 0.0f;
    float penultimate_runtime = 0.0f;
    float last_runtime = 0.0f;
    if (!terrain_v2_axis_node(
            origin, 0, cell_size, first_source, first_runtime) ||
        !terrain_v2_axis_node(
            origin, 1, cell_size, second_source, second_runtime) ||
        !terrain_v2_axis_node(
            origin, count - 2, cell_size,
            penultimate_source, penultimate_runtime) ||
        !terrain_v2_axis_node(
            origin, count - 1, cell_size, last_source, last_runtime) ||
        second_source <= first_source ||
        second_runtime <= first_runtime ||
        last_source <= penultimate_source ||
        last_runtime <= penultimate_runtime) {
        return false;
    }

    if (first_source < 0.0 && last_source > 0.0) {
        const volatile double negated_origin =
            -static_cast<double>(origin);
        const volatile double zero_index =
            negated_origin / static_cast<double>(cell_size);
        if (!terrain_double_is_finite(zero_index)) {
            return false;
        }
        const double base_value = floor(zero_index);
        if (base_value < 0.0 ||
            base_value > static_cast<double>(count - 1)) {
            return false;
        }
        const int64_t base = static_cast<int64_t>(base_value);
        for (int offset = -1; offset <= 2; ++offset) {
            const int64_t candidate = base + offset;
            if (candidate < 0 ||
                candidate >= static_cast<int64_t>(count)) {
                continue;
            }
            double source = 0.0;
            float runtime = 0.0f;
            if (!terrain_v2_axis_node(
                    origin, static_cast<uint32_t>(candidate), cell_size,
                    source, runtime)) {
                return false;
            }
        }
    }

    if (count <= 3) {
        return true;
    }

    const double source_spacing =
        terrain_v2_source_maximum_inward_spacing(
            first_source, last_source);
    const double runtime_spacing =
        terrain_v2_runtime_maximum_inward_spacing(
            first_runtime, last_runtime);
    const double maximum_spacing =
        source_spacing > runtime_spacing ? source_spacing : runtime_spacing;
    const double promoted_cell_size = static_cast<double>(cell_size);
    if (promoted_cell_size > maximum_spacing) {
        return true;
    }
    if (promoted_cell_size < maximum_spacing) {
        return false;
    }

    const volatile double quotient =
        static_cast<double>(origin) / promoted_cell_size;
    return terrain_double_is_finite(quotient) &&
           quotient == floor(quotient);
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

#ifndef TERRAIN_RUNTIME_PAYLOAD_ALLOCATE
#define TERRAIN_RUNTIME_PAYLOAD_ALLOCATE(bytes) malloc(bytes)
#define TERRAIN_RUNTIME_DEFAULT_PAYLOAD_ALLOCATE
#endif

template<typename T>
static inline bool terrain_payload_allocate_empty(
    array1d<T>& out, int count, const char* path, const char* label,
    char* error, int capacity)
{
    size_t bytes = 0;
    if (count <= 0 || out.size != 0 || out.data != NULL ||
        !terrain_size_multiply(
            static_cast<size_t>(count), sizeof(T), bytes)) {
        return terrain_error(error, capacity,
            "%s: invalid %s allocation size", path, label);
    }
    T* data = static_cast<T*>(TERRAIN_RUNTIME_PAYLOAD_ALLOCATE(bytes));
    if (data == NULL) {
        return terrain_error(error, capacity,
            "%s: cannot allocate %s payload", path, label);
    }
    out.size = count;
    out.data = data;
    return true;
}

template<typename T>
static inline bool terrain_payload_allocate_empty(
    array2d<T>& out, int rows, int cols, const char* path, const char* label,
    char* error, int capacity)
{
    if (rows <= 0 || cols <= 0 || rows > INT_MAX / cols ||
        out.rows != 0 || out.cols != 0 || out.data != NULL) {
        return terrain_error(error, capacity,
            "%s: invalid %s allocation shape", path, label);
    }
    size_t bytes = 0;
    if (!terrain_size_multiply(
            static_cast<size_t>(rows) * static_cast<size_t>(cols),
            sizeof(T), bytes)) {
        return terrain_error(error, capacity,
            "%s: invalid %s allocation size", path, label);
    }
    T* data = static_cast<T*>(TERRAIN_RUNTIME_PAYLOAD_ALLOCATE(bytes));
    if (data == NULL) {
        return terrain_error(error, capacity,
            "%s: cannot allocate %s payload", path, label);
    }
    out.rows = rows;
    out.cols = cols;
    out.data = data;
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
    if (!terrain_payload_allocate_empty(
            loaded.values,
            static_cast<int>(frames),
            static_cast<int>(dimensions),
            path,
            "G1TF",
            error,
            error_capacity)) {
        fclose(file);
        return false;
    }
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

    if (version != 1 && version != 2) {
        fclose(file);
        return terrain_error(
            error, error_capacity,
            "%s: unsupported G1HF version %u (expected 1 or 2)",
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
    if (version == 2) {
        if (!terrain_float_is_normal_or_positive_zero(origin_x) ||
            !terrain_float_is_normal_or_positive_zero(origin_z) ||
            !terrain_float_is_normal_or_positive_zero(exterior_height)) {
            fclose(file);
            return terrain_error(
                error, error_capacity,
                "%s: G1HF v2 origin and exterior height must be "
                "normal-or-positive-zero binary32", path);
        }
        if (!terrain_float_is_positive_normal(cell_size)) {
            fclose(file);
            return terrain_error(
                error, error_capacity,
                "%s: G1HF v2 cell size must encode as positive normal "
                "binary32", path);
        }
        if (!terrain_v2_runtime_axis_is_valid(origin_x, nx, cell_size)) {
            fclose(file);
            return terrain_error(
                error, error_capacity,
                "%s: G1HF v2 X nodes must be normal-or-zero runtime nodes "
                "and runtime-distinguishable", path);
        }
        if (!terrain_v2_runtime_axis_is_valid(origin_z, nz, cell_size)) {
            fclose(file);
            return terrain_error(
                error, error_capacity,
                "%s: G1HF v2 Z nodes must be normal-or-zero runtime nodes "
                "and runtime-distinguishable", path);
        }
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
    loaded.version = version;
    loaded.nx = static_cast<int>(nx);
    loaded.nz = static_cast<int>(nz);
    loaded.origin_x = origin_x;
    loaded.origin_z = origin_z;
    loaded.cell_size = cell_size;
    loaded.exterior_height = exterior_height;
    if (!terrain_payload_allocate_empty(
            loaded.heights,
            static_cast<int>(height_count),
            path,
            "G1HF",
            error,
            error_capacity)) {
        fclose(file);
        return false;
    }
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
        if (version == 2 &&
            !terrain_float_is_normal_or_positive_zero(
                loaded.heights.data[i])) {
            fclose(file);
            return terrain_error(
                error, error_capacity,
                "%s: G1HF v2 heights must all be "
                "normal-or-positive-zero binary32 (index %zu)", path, i);
        }
    }

    if (!terrain_finish_read(file, path, error, error_capacity)) {
        return false;
    }

    std::swap(out.version, loaded.version);
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

struct terrain_support_set
{
    array2d<float> values;
};

struct walkability_grid
{
    int nx = 0;
    int nz = 0;
    array1d<uint8_t> cells;
};

static inline bool terrain_support_load(
    terrain_support_set& out,
    const char* path,
    const int expected_frames,
    char* error,
    const int error_capacity)
{
    const char* shown = path != NULL ? path : "<null>";
    if (path == NULL || path[0] == '\0' || expected_frames <= 0) {
        return terrain_error(error, error_capacity,
            "%s: invalid G1SP path or expected frame count %d",
            shown, expected_frames);
    }
    FILE* file = fopen(path, "rb");
    if (file == NULL)
        return terrain_error(error, error_capacity,
            "%s: cannot open (%s)", path, strerror(errno));

    size_t actual_size = 0;
    unsigned char header[16] = {};
    if (!terrain_file_size(file, actual_size) || actual_size < sizeof(header) ||
        !terrain_read_exact(file, header, sizeof(header))) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: truncated G1SP header", path);
    }
    if (memcmp(header, "G1SP", 4) != 0) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: invalid G1SP magic", path);
    }
    const uint32_t version = terrain_decode_u32_le(header + 4);
    const uint32_t frames = terrain_decode_u32_le(header + 8);
    const uint32_t dimensions = terrain_decode_u32_le(header + 12);
    if (version != 1) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: unsupported G1SP version %u (expected 1)", path,
            static_cast<unsigned>(version));
    }
    if (dimensions != 3) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: invalid G1SP dimension %u (expected 3)", path,
            static_cast<unsigned>(dimensions));
    }
    if (frames != static_cast<uint32_t>(expected_frames)) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: G1SP frame mismatch: database=%d support=%u", path,
            expected_frames, static_cast<unsigned>(frames));
    }

    size_t count = 0;
    size_t bytes = 0;
    size_t expected_size = 0;
    if (!terrain_size_multiply(static_cast<size_t>(frames), 3u, count) ||
        count > static_cast<size_t>(INT_MAX) ||
        !terrain_size_multiply(count, sizeof(float), bytes) ||
        !terrain_size_add(sizeof(header), bytes, expected_size)) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: G1SP size overflow", path);
    }
    if (actual_size != expected_size) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: %s G1SP payload (expected %zu bytes, got %zu)", path,
            actual_size < expected_size ? "truncated" : "trailing",
            expected_size, actual_size);
    }

    terrain_support_set loaded;
    if (!terrain_payload_allocate_empty(
            loaded.values, expected_frames, 3, path, "G1SP",
            error, error_capacity)) {
        fclose(file);
        return false;
    }
    if (!terrain_read_exact(file, loaded.values.data, bytes)) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: truncated G1SP values", path);
    }
    terrain_decode_float_array_le(loaded.values.data, count);
    for (size_t i = 0; i < count; ++i) {
        if (!terrain_float_is_finite(loaded.values.data[i])) {
            fclose(file);
            return terrain_error(error, error_capacity,
                "%s: G1SP values must be finite (index %zu)", path, i);
        }
    }
    if (!terrain_finish_read(file, path, error, error_capacity)) return false;
    std::swap(out.values.rows, loaded.values.rows);
    std::swap(out.values.cols, loaded.values.cols);
    std::swap(out.values.data, loaded.values.data);
    return true;
}

static inline bool walkability_load(
    walkability_grid& out,
    const char* path,
    const heightfield& field,
    char* error,
    const int error_capacity)
{
    const char* shown = path != NULL ? path : "<null>";
    if (path == NULL || path[0] == '\0' || field.nx < 2 || field.nz < 2) {
        return terrain_error(error, error_capacity,
            "%s: invalid G1WM path or reference heightfield", shown);
    }
    FILE* file = fopen(path, "rb");
    if (file == NULL)
        return terrain_error(error, error_capacity,
            "%s: cannot open (%s)", path, strerror(errno));

    size_t actual_size = 0;
    unsigned char header[16] = {};
    if (!terrain_file_size(file, actual_size) || actual_size < sizeof(header) ||
        !terrain_read_exact(file, header, sizeof(header))) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: truncated G1WM header", path);
    }
    if (memcmp(header, "G1WM", 4) != 0) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: invalid G1WM magic", path);
    }
    const uint32_t version = terrain_decode_u32_le(header + 4);
    const uint32_t nx = terrain_decode_u32_le(header + 8);
    const uint32_t nz = terrain_decode_u32_le(header + 12);
    if (version != 1) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: unsupported G1WM version %u (expected 1)", path,
            static_cast<unsigned>(version));
    }
    if (nx != static_cast<uint32_t>(field.nx) ||
        nz != static_cast<uint32_t>(field.nz)) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: G1WM grid %ux%u does not match G1HF %dx%d", path,
            static_cast<unsigned>(nx), static_cast<unsigned>(nz),
            field.nx, field.nz);
    }
    size_t count = 0;
    size_t expected_size = 0;
    if (!terrain_size_multiply(static_cast<size_t>(nx),
                               static_cast<size_t>(nz), count) ||
        count > static_cast<size_t>(INT_MAX) ||
        !terrain_size_add(sizeof(header), count, expected_size)) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: G1WM size overflow", path);
    }
    if (actual_size != expected_size) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: %s G1WM payload (expected %zu bytes, got %zu)", path,
            actual_size < expected_size ? "truncated" : "trailing",
            expected_size, actual_size);
    }

    walkability_grid loaded;
    loaded.nx = static_cast<int>(nx);
    loaded.nz = static_cast<int>(nz);
    if (!terrain_payload_allocate_empty(
            loaded.cells, static_cast<int>(count), path, "G1WM",
            error, error_capacity)) {
        fclose(file);
        return false;
    }
    if (!terrain_read_exact(file, loaded.cells.data, count)) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: truncated G1WM cells", path);
    }
    for (size_t i = 0; i < count; ++i) {
        if (loaded.cells.data[i] > 2) {
            fclose(file);
            return terrain_error(error, error_capacity,
                "%s: invalid G1WM class %u at cell %zu", path,
                static_cast<unsigned>(loaded.cells.data[i]), i);
        }
    }
    if (!terrain_finish_read(file, path, error, error_capacity)) return false;
    std::swap(out.nx, loaded.nx);
    std::swap(out.nz, loaded.nz);
    std::swap(out.cells.size, loaded.cells.size);
    std::swap(out.cells.data, loaded.cells.data);
    return true;
}

#if defined(__GNUC__) || defined(__clang__)
#define TERRAIN_F32_NOINLINE __attribute__((noinline))
#else
#define TERRAIN_F32_NOINLINE
#endif

static inline bool terrain_f32_accept(float& out, float value)
{
    if (!terrain_float_is_normal_or_zero_query(value)) return false;
    out = value == 0.0f ? 0.0f : value;
    return true;
}

static inline TERRAIN_F32_NOINLINE bool terrain_f32_add(
    float& out, float left, float right)
{
    const volatile float a = left, b = right;
    const volatile float rounded = a + b;
    return terrain_f32_accept(out, rounded);
}

static inline TERRAIN_F32_NOINLINE bool terrain_f32_sub(
    float& out, float left, float right)
{
    const volatile float a = left, b = right;
    const volatile float rounded = a - b;
    return terrain_f32_accept(out, rounded);
}

static inline TERRAIN_F32_NOINLINE bool terrain_f32_mul(
    float& out, float left, float right)
{
    const volatile float a = left, b = right;
    const volatile float rounded = a * b;
    return terrain_f32_accept(out, rounded);
}

static inline TERRAIN_F32_NOINLINE bool terrain_f32_div(
    float& out, float left, float right)
{
    const volatile float a = left, b = right;
    const volatile float rounded = a / b;
    return terrain_f32_accept(out, rounded);
}

static inline TERRAIN_F32_NOINLINE bool terrain_f32_sqrt(
    float& out, float value)
{
    const volatile float input = value;
    const volatile float rounded = sqrtf(input);
    return terrain_f32_accept(out, rounded);
}

#undef TERRAIN_F32_NOINLINE

static inline bool terrain_f32_lerp(
    float& out, float start, float stop, int step, int steps)
{
    if (step < 0 || steps < 1 || step > steps) return false;
    float delta = 0.0f;
    float alpha = 0.0f;
    float scaled = 0.0f;
    const volatile float numerator = static_cast<float>(step);
    const volatile float denominator = static_cast<float>(steps);
    return terrain_f32_sub(delta, stop, start) &&
           terrain_f32_div(alpha, numerator, denominator) &&
           terrain_f32_mul(scaled, alpha, delta) &&
           terrain_f32_add(out, start, scaled);
}

static inline bool walkability_grid_matches_heightfield(
    const walkability_grid& grid, const heightfield& field)
{
    size_t count = 0;
    return grid.nx == field.nx && grid.nz == field.nz &&
           grid.nx >= 2 && grid.nz >= 2 &&
           terrain_float_is_normal_or_positive_zero(field.origin_x) &&
           terrain_float_is_normal_or_positive_zero(field.origin_z) &&
           terrain_float_is_positive_normal(field.cell_size) &&
           terrain_size_multiply(
               static_cast<size_t>(grid.nx),
               static_cast<size_t>(grid.nz), count) &&
           count <= static_cast<size_t>(INT_MAX) &&
           grid.cells.size == static_cast<int>(count);
}

static inline bool terrain_v2_query_coordinate(
    float input, float& canonical);

static inline bool walkability_nearest_axis(
    int& index, float input, float origin, float cell_size, int count)
{
    float canonical = 0.0f;
    float numerator = 0.0f;
    float normalized = 0.0f;
    float shifted = 0.0f;
    if (count < 2 ||
        !terrain_v2_query_coordinate(input, canonical) ||
        !terrain_f32_sub(numerator, canonical, origin) ||
        !terrain_f32_div(normalized, numerator, cell_size) ||
        static_cast<double>(normalized) < 0.0 ||
        static_cast<double>(normalized) > static_cast<double>(count - 1) ||
        !terrain_f32_add(shifted, normalized, 0.5f)) {
        return false;
    }
    const int rounded = static_cast<int>(floorf(shifted));
    index = rounded < count ? rounded : count - 1;
    return index >= 0 && index < count;
}

static inline int walkability_class_at(
    const walkability_grid& grid,
    const heightfield& field,
    float x,
    float z)
{
    if (!walkability_grid_matches_heightfield(grid, field)) return 0;
    int ix = 0;
    int iz = 0;
    if (!walkability_nearest_axis(
            ix, x, field.origin_x, field.cell_size, grid.nx) ||
        !walkability_nearest_axis(
            iz, z, field.origin_z, field.cell_size, grid.nz)) {
        return 0;
    }
    const int value = grid.cells(iz * grid.nx + ix);
    return value <= 2 ? value : 0;
}

static inline bool terrain_heightfield_is_queryable(const heightfield& field);

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

static inline float heightfield_sample_v1_legacy(
    const heightfield& field, float x, float z)
{
    return heightfield_sample(field, x, z);
}

struct heightfield_cell
{
    int x0;
    int z0;
    double tx;
    double tz;
};

static inline bool terrain_v2_query_coordinate(
    float input, float& canonical)
{
    if (!terrain_float_is_normal_or_zero_query(input)) {
        return false;
    }
    canonical = (terrain_float_bits(input) & UINT32_C(0x7fffffff)) == 0
        ? 0.0f
        : input;
    return true;
}

static inline bool terrain_v2_axis_cell(
    double value,
    float origin,
    int count,
    float cell_size,
    int& index,
    double& fraction)
{
    const volatile double difference =
        value - static_cast<double>(origin);
    const volatile double coordinate =
        difference / static_cast<double>(cell_size);
    if (!terrain_double_is_finite(coordinate)) {
        return false;
    }

    const double floored = floor(coordinate);
    if (floored <= 0.0) {
        index = 0;
    } else if (floored >= static_cast<double>(count - 2)) {
        index = count - 2;
    } else {
        index = static_cast<int>(floored);
    }

    while (index > 0) {
        const volatile double product =
            static_cast<double>(index) * static_cast<double>(cell_size);
        const volatile double node =
            static_cast<double>(origin) + product;
        if (!(value < node)) {
            break;
        }
        --index;
    }
    while (index < count - 2) {
        const volatile double product =
            static_cast<double>(index + 1) *
            static_cast<double>(cell_size);
        const volatile double next_node =
            static_cast<double>(origin) + product;
        if (!(value >= next_node)) {
            break;
        }
        ++index;
    }

    const volatile double product =
        static_cast<double>(index) * static_cast<double>(cell_size);
    const volatile double node =
        static_cast<double>(origin) + product;
    const volatile double local_difference = value - node;
    const volatile double local_fraction =
        local_difference / static_cast<double>(cell_size);
    if (!terrain_double_is_finite(local_fraction)) {
        return false;
    }
    if (local_fraction <= 0.0) {
        fraction = 0.0;
    } else if (local_fraction >= 1.0) {
        fraction = 1.0;
    } else {
        fraction = local_fraction;
    }
    return true;
}

static inline bool terrain_v2_locate_cell(
    const heightfield& field,
    float input_x,
    float input_z,
    heightfield_cell& cell)
{
    float canonical_x = 0.0f;
    float canonical_z = 0.0f;
    if (!terrain_v2_query_coordinate(input_x, canonical_x) ||
        !terrain_v2_query_coordinate(input_z, canonical_z)) {
        return false;
    }

    const double x = static_cast<double>(canonical_x);
    const double z = static_cast<double>(canonical_z);
    const volatile double maximum_x_product =
        static_cast<double>(field.nx - 1) *
        static_cast<double>(field.cell_size);
    const volatile double maximum_z_product =
        static_cast<double>(field.nz - 1) *
        static_cast<double>(field.cell_size);
    const volatile double maximum_x =
        static_cast<double>(field.origin_x) + maximum_x_product;
    const volatile double maximum_z =
        static_cast<double>(field.origin_z) + maximum_z_product;
    if (!terrain_double_is_finite(maximum_x) ||
        !terrain_double_is_finite(maximum_z) ||
        x < static_cast<double>(field.origin_x) || x > maximum_x ||
        z < static_cast<double>(field.origin_z) || z > maximum_z) {
        return false;
    }

    return terrain_v2_axis_cell(
               x, field.origin_x, field.nx, field.cell_size,
               cell.x0, cell.tx) &&
           terrain_v2_axis_cell(
               z, field.origin_z, field.nz, field.cell_size,
               cell.z0, cell.tz);
}

static inline bool terrain_v2_cell_heights(
    const heightfield& field,
    const heightfield_cell& cell,
    double& h00,
    double& h10,
    double& h01,
    double& h11)
{
    const int offset = cell.z0 * field.nx + cell.x0;
    const float value00 = field.heights(offset);
    const float value10 = field.heights(offset + 1);
    const float value01 = field.heights(offset + field.nx);
    const float value11 = field.heights(offset + field.nx + 1);
    if (!terrain_float_is_normal_or_positive_zero(value00) ||
        !terrain_float_is_normal_or_positive_zero(value10) ||
        !terrain_float_is_normal_or_positive_zero(value01) ||
        !terrain_float_is_normal_or_positive_zero(value11)) {
        return false;
    }
    h00 = static_cast<double>(value00);
    h10 = static_cast<double>(value10);
    h01 = static_cast<double>(value01);
    h11 = static_cast<double>(value11);
    return true;
}

static inline bool terrain_v2_round_output(double value, float& output)
{
    if (!terrain_double_is_finite(value)) {
        return false;
    }
    const volatile double materialized = value;
    const volatile float rounded = static_cast<float>(materialized);
    if (!terrain_float_is_finite(rounded)) {
        return false;
    }
    output = terrain_runtime_canonicalize_output(rounded);
    return true;
}

// Internal hot path. The caller must already have established that `field`
// is a structurally valid G1HF/v2 heightfield.
static inline float terrain_heightfield_sample_v2_prevalidated(
    const heightfield& field, float x, float z)
{
    heightfield_cell cell = {};
    if (!terrain_v2_locate_cell(field, x, z, cell)) {
        return field.exterior_height;
    }

    double h00 = 0.0;
    double h10 = 0.0;
    double h01 = 0.0;
    double h11 = 0.0;
    if (!terrain_v2_cell_heights(
            field, cell, h00, h10, h01, h11)) {
        return field.exterior_height;
    }

    double value = 0.0;
    if (cell.tx >= cell.tz) {
        const volatile double difference_x = h10 - h00;
        const volatile double x_term = cell.tx * difference_x;
        const volatile double first_sum = h00 + x_term;
        const volatile double difference_z = h11 - h10;
        const volatile double z_term = cell.tz * difference_z;
        const volatile double final_sum = first_sum + z_term;
        value = final_sum;
    } else {
        const volatile double difference_x = h11 - h01;
        const volatile double x_term = cell.tx * difference_x;
        const volatile double first_sum = h00 + x_term;
        const volatile double difference_z = h01 - h00;
        const volatile double z_term = cell.tz * difference_z;
        const volatile double final_sum = first_sum + z_term;
        value = final_sum;
    }

    float output = field.exterior_height;
    return terrain_v2_round_output(value, output)
        ? output
        : field.exterior_height;
}

static inline float heightfield_sample_v2(
    const heightfield& field, float x, float z)
{
    if (field.version != 2 || !terrain_heightfield_is_queryable(field)) {
        return field.exterior_height;
    }
    return terrain_heightfield_sample_v2_prevalidated(field, x, z);
}

static inline float heightfield_sample_versioned(
    const heightfield& field, float x, float z)
{
    if (!terrain_heightfield_is_queryable(field)) {
        return field.exterior_height;
    }
    if (field.version == 1) {
        return heightfield_sample_v1_legacy(field, x, z);
    }
    if (field.version == 2) {
        return terrain_heightfield_sample_v2_prevalidated(field, x, z);
    }
    return field.exterior_height;
}

static inline vec3 heightfield_normal(
    const heightfield& field, float x, float z)
{
    const vec3 up(0.0f, 1.0f, 0.0f);
    if (!terrain_heightfield_is_queryable(field) || field.version != 2) {
        return up;
    }

    heightfield_cell cell = {};
    if (!terrain_v2_locate_cell(field, x, z, cell)) {
        return up;
    }
    double h00 = 0.0;
    double h10 = 0.0;
    double h01 = 0.0;
    double h11 = 0.0;
    if (!terrain_v2_cell_heights(
            field, cell, h00, h10, h01, h11)) {
        return up;
    }

    double slope_x = 0.0;
    double slope_z = 0.0;
    if (cell.tx >= cell.tz) {
        const volatile double difference_x = h10 - h00;
        const volatile double difference_z = h11 - h10;
        const volatile double divided_x =
            difference_x / static_cast<double>(field.cell_size);
        const volatile double divided_z =
            difference_z / static_cast<double>(field.cell_size);
        slope_x = divided_x;
        slope_z = divided_z;
    } else {
        const volatile double difference_x = h11 - h01;
        const volatile double difference_z = h01 - h00;
        const volatile double divided_x =
            difference_x / static_cast<double>(field.cell_size);
        const volatile double divided_z =
            difference_z / static_cast<double>(field.cell_size);
        slope_x = divided_x;
        slope_z = divided_z;
    }
    if (!terrain_double_is_finite(slope_x) ||
        !terrain_double_is_finite(slope_z)) {
        return up;
    }

    const volatile double normal_x = -slope_x;
    const volatile double normal_y = 1.0;
    const volatile double normal_z = -slope_z;
    const double absolute_x = fabs(normal_x);
    const double absolute_y = fabs(normal_y);
    const double absolute_z = fabs(normal_z);
    const double maximum_xy =
        absolute_x > absolute_y ? absolute_x : absolute_y;
    const double maximum =
        maximum_xy > absolute_z ? maximum_xy : absolute_z;
    if (!terrain_double_is_finite(maximum) || maximum <= 0.0) {
        return up;
    }

    const volatile double scaled_x = normal_x / maximum;
    const volatile double scaled_y = normal_y / maximum;
    const volatile double scaled_z = normal_z / maximum;
    const volatile double square_x = scaled_x * scaled_x;
    const volatile double square_y = scaled_y * scaled_y;
    const volatile double square_z = scaled_z * scaled_z;
    const volatile double first_sum = square_x + square_y;
    const volatile double square_sum = first_sum + square_z;
    if (!terrain_double_is_finite(square_sum) || square_sum <= 0.0) {
        return up;
    }
    const volatile double length = sqrt(square_sum);
    if (!terrain_double_is_finite(length) || length <= 0.0) {
        return up;
    }
    const volatile double unit_x = scaled_x / length;
    const volatile double unit_y = scaled_y / length;
    const volatile double unit_z = scaled_z / length;
    float output_x = 0.0f;
    float output_y = 0.0f;
    float output_z = 0.0f;
    if (!terrain_v2_round_output(unit_x, output_x) ||
        !terrain_v2_round_output(unit_y, output_y) ||
        !terrain_v2_round_output(unit_z, output_z)) {
        return up;
    }
    return vec3(output_x, output_y, output_z);
}

static inline bool terrain_centerline_inputs_are_valid(
    vec3 root,
    const slice1d<vec3> trajectory_positions,
    const slice1d<quat> trajectory_rotations)
{
    if (!terrain_float_is_finite(root.x) ||
        !terrain_float_is_finite(root.y) ||
        !terrain_float_is_finite(root.z) ||
        trajectory_positions.size <= 0 ||
        trajectory_positions.size != trajectory_rotations.size ||
        trajectory_positions.data == NULL ||
        trajectory_rotations.data == NULL) {
        return false;
    }

    for (int i = 0; i < trajectory_positions.size; ++i) {
        const vec3 position = trajectory_positions.data[i];
        const quat rotation = trajectory_rotations.data[i];
        if (!terrain_float_is_finite(position.x) ||
            !terrain_float_is_finite(position.y) ||
            !terrain_float_is_finite(position.z) ||
            !terrain_float_is_finite(rotation.w) ||
            !terrain_float_is_finite(rotation.x) ||
            !terrain_float_is_finite(rotation.y) ||
            !terrain_float_is_finite(rotation.z)) {
            return false;
        }
    }
    return true;
}

static inline vec3 terrain_centerline_flattened_heading(
    quat rotation, vec3 fallback)
{
    if (!terrain_float_is_finite(rotation.w) ||
        !terrain_float_is_finite(rotation.x) ||
        !terrain_float_is_finite(rotation.y) ||
        !terrain_float_is_finite(rotation.z)) {
        return fallback;
    }
    const float rotation_maximum = maxf(
        maxf(fabsf(rotation.w), fabsf(rotation.x)),
        maxf(fabsf(rotation.y), fabsf(rotation.z)));
    if (rotation_maximum < 1e-8f) {
        return fallback;
    }
    double w = static_cast<double>(rotation.w) / rotation_maximum;
    double x = static_cast<double>(rotation.x) / rotation_maximum;
    double y = static_cast<double>(rotation.y) / rotation_maximum;
    double z = static_cast<double>(rotation.z) / rotation_maximum;
    const double rotation_length = sqrt(w * w + x * x + y * y + z * z);
    if (rotation_length < 1e-8) {
        return fallback;
    }
    w /= rotation_length;
    x /= rotation_length;
    y /= rotation_length;
    z /= rotation_length;

    const double forward_x = 2.0 * (x * z + w * y);
    const double forward_z = 1.0 - 2.0 * (x * x + y * y);
    const double maximum = fmax(fabs(forward_x), fabs(forward_z));
    if (maximum < 1e-8) {
        return fallback;
    }
    const double scaled_x = forward_x / maximum;
    const double scaled_z = forward_z / maximum;
    const double scaled_length = sqrt(
        scaled_x * scaled_x + scaled_z * scaled_z);
    if (scaled_length < 1e-8) {
        return fallback;
    }
    return vec3(
        static_cast<float>(scaled_x / scaled_length),
        0.0f,
        static_cast<float>(scaled_z / scaled_length));
}

static inline vec3 terrain_centerline_safe_point(
    double x, double z, vec3 fallback)
{
    if (x < -static_cast<double>(FLT_MAX) ||
        x > static_cast<double>(FLT_MAX) ||
        z < -static_cast<double>(FLT_MAX) ||
        z > static_cast<double>(FLT_MAX)) {
        return fallback;
    }
    const vec3 point(static_cast<float>(x), 0.0f, static_cast<float>(z));
    if (!terrain_float_is_finite(point.x) ||
        !terrain_float_is_finite(point.z)) {
        return fallback;
    }
    return point;
}

static inline vec3 terrain_centerline_point_at_arc(
    vec3 root,
    const slice1d<vec3> trajectory_positions,
    const slice1d<quat> trajectory_rotations,
    float distance)
{
    const vec3 safe_root(
        terrain_float_is_finite(root.x) ? root.x : 0.0f,
        0.0f,
        terrain_float_is_finite(root.z) ? root.z : 0.0f);
    if (!terrain_float_is_finite(distance) || distance <= 0.0f ||
        !terrain_centerline_inputs_are_valid(
            root, trajectory_positions, trajectory_rotations)) {
        return safe_root;
    }

    vec3 latest_heading = terrain_centerline_flattened_heading(
        trajectory_rotations.data[0], vec3(0.0f, 0.0f, 1.0f));
    double previous_x = static_cast<double>(root.x);
    double previous_z = static_cast<double>(root.z);
    double remaining = static_cast<double>(distance);

    for (int i = 1; i < trajectory_positions.size; ++i) {
        const double next_x =
            static_cast<double>(trajectory_positions.data[i].x);
        const double next_z =
            static_cast<double>(trajectory_positions.data[i].z);
        const double delta_x = next_x - previous_x;
        const double delta_z = next_z - previous_z;
        const double segment_length = sqrt(
            delta_x * delta_x + delta_z * delta_z);

        if (segment_length > 1e-6) {
            if (remaining <= segment_length) {
                const double alpha = remaining / segment_length;
                return terrain_centerline_safe_point(
                    previous_x + delta_x * alpha,
                    previous_z + delta_z * alpha,
                    safe_root);
            }
            remaining -= segment_length;
            previous_x = next_x;
            previous_z = next_z;
        }

        latest_heading = terrain_centerline_flattened_heading(
            trajectory_rotations.data[i], latest_heading);
    }

    const vec3 last_point = terrain_centerline_safe_point(
        previous_x, previous_z, safe_root);
    return terrain_centerline_safe_point(
        previous_x + static_cast<double>(latest_heading.x) * remaining,
        previous_z + static_cast<double>(latest_heading.z) * remaining,
        last_point);
}

static inline bool terrain_heightfield_is_queryable(const heightfield& field)
{
    if ((field.version != 1 && field.version != 2) ||
        field.nx < 2 || field.nz < 2 ||
        field.heights.data == NULL ||
        !terrain_float_is_finite(field.origin_x) ||
        !terrain_float_is_finite(field.origin_z) ||
        !terrain_float_is_finite(field.cell_size) ||
        field.cell_size <= 0.0f ||
        !terrain_float_is_finite(field.exterior_height)) {
        return false;
    }
    const int64_t expected_size =
        static_cast<int64_t>(field.nx) * static_cast<int64_t>(field.nz);
    if (expected_size != static_cast<int64_t>(field.heights.size)) {
        return false;
    }
    if (field.version == 2 &&
        (!terrain_float_is_normal_or_positive_zero(field.origin_x) ||
         !terrain_float_is_normal_or_positive_zero(field.origin_z) ||
         !terrain_float_is_positive_normal(field.cell_size) ||
         !terrain_float_is_normal_or_positive_zero(field.exterior_height) ||
         !terrain_v2_runtime_axis_is_valid(
             field.origin_x, static_cast<uint32_t>(field.nx),
             field.cell_size) ||
         !terrain_v2_runtime_axis_is_valid(
             field.origin_z, static_cast<uint32_t>(field.nz),
             field.cell_size))) {
        return false;
    }
    return true;
}

struct terrain_centerline_snapshot
{
    float values[4];
    vec3 points[4];
};

static inline void terrain_centerline_snapshot_compute(
    terrain_centerline_snapshot& out,
    const heightfield& field,
    vec3 root,
    const slice1d<vec3> trajectory_positions,
    const slice1d<quat> trajectory_rotations)
{
    const vec3 safe_root(
        terrain_float_is_finite(root.x) ? root.x : 0.0f,
        0.0f,
        terrain_float_is_finite(root.z) ? root.z : 0.0f);
    for (int i = 0; i < 4; ++i) {
        out.values[i] = 0.0f;
        out.points[i] = safe_root;
    }
    if (!terrain_heightfield_is_queryable(field) ||
        !terrain_centerline_inputs_are_valid(
            root, trajectory_positions, trajectory_rotations)) {
        return;
    }

    const float base_height = heightfield_sample(field, root.x, root.z);
    if (!terrain_float_is_finite(base_height)) {
        return;
    }

    static const float distances[4] = {
        0.25f, 0.50f, 0.75f, 1.00f};
    for (int i = 0; i < 4; ++i) {
        const vec3 point = terrain_centerline_point_at_arc(
            root, trajectory_positions, trajectory_rotations, distances[i]);
        const float sample_height =
            heightfield_sample(field, point.x, point.z);
        if (!terrain_float_is_finite(sample_height)) {
            continue;
        }
        out.points[i] = vec3(point.x, sample_height, point.z);
        const double difference =
            static_cast<double>(sample_height) -
            static_cast<double>(base_height);
        if (difference >= -static_cast<double>(FLT_MAX) &&
            difference <= static_cast<double>(FLT_MAX)) {
            out.values[i] = static_cast<float>(difference);
        }
    }
}

static inline void terrain_centerline_query(
    float out[4],
    const heightfield& field,
    vec3 root,
    const slice1d<vec3> trajectory_positions,
    const slice1d<quat> trajectory_rotations)
{
    if (out == NULL) {
        return;
    }
    terrain_centerline_snapshot snapshot = {};
    terrain_centerline_snapshot_compute(
        snapshot,
        field,
        root,
        trajectory_positions,
        trajectory_rotations);
    for (int i = 0; i < 4; ++i) {
        out[i] = snapshot.values[i];
    }
}

static inline void terrain_centerline_snapshot_compute_v2(
    terrain_centerline_snapshot& out,
    const heightfield& field,
    vec3 root,
    const slice1d<vec3> trajectory_positions,
    const slice1d<quat> trajectory_rotations)
{
    const vec3 safe_root(
        terrain_float_is_finite(root.x) ? root.x : 0.0f,
        0.0f,
        terrain_float_is_finite(root.z) ? root.z : 0.0f);
    for (int i = 0; i < 4; ++i) {
        out.values[i] = 0.0f;
        out.points[i] = safe_root;
    }
    if (field.version != 2 ||
        !terrain_heightfield_is_queryable(field) ||
        !terrain_centerline_inputs_are_valid(
            root, trajectory_positions, trajectory_rotations)) {
        return;
    }

    const float base_height = heightfield_sample_v2(field, root.x, root.z);
    if (!terrain_float_is_finite(base_height)) {
        return;
    }

    static const float distances[4] = {
        0.25f, 0.50f, 0.75f, 1.00f};
    for (int i = 0; i < 4; ++i) {
        const vec3 point = terrain_centerline_point_at_arc(
            root, trajectory_positions, trajectory_rotations, distances[i]);
        const float sample_height =
            heightfield_sample_v2(field, point.x, point.z);
        if (!terrain_float_is_finite(sample_height)) {
            continue;
        }
        out.points[i] = vec3(point.x, sample_height, point.z);
        const double difference =
            static_cast<double>(sample_height) -
            static_cast<double>(base_height);
        if (difference >= -static_cast<double>(FLT_MAX) &&
            difference <= static_cast<double>(FLT_MAX)) {
            out.values[i] = static_cast<float>(difference);
        }
    }
}

static inline void terrain_centerline_query_v2(
    float out[4],
    const heightfield& field,
    vec3 root,
    const slice1d<vec3> trajectory_positions,
    const slice1d<quat> trajectory_rotations)
{
    if (out == NULL) {
        return;
    }
    terrain_centerline_snapshot snapshot = {};
    terrain_centerline_snapshot_compute_v2(
        snapshot,
        field,
        root,
        trajectory_positions,
        trajectory_rotations);
    for (int i = 0; i < 4; ++i) {
        out[i] = snapshot.values[i];
    }
}

#ifdef TERRAIN_RUNTIME_DEFAULT_PAYLOAD_ALLOCATE
#undef TERRAIN_RUNTIME_PAYLOAD_ALLOCATE
#undef TERRAIN_RUNTIME_DEFAULT_PAYLOAD_ALLOCATE
#endif
