#pragma once

#include <cerrno>
#include <climits>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

enum : uint16_t
{
    MOTION_DIRECTION_IDLE = 0x001,
    MOTION_DIRECTION_FORWARD = 0x002,
    MOTION_DIRECTION_FORWARD_RIGHT = 0x004,
    MOTION_DIRECTION_RIGHT = 0x008,
    MOTION_DIRECTION_BACK_RIGHT = 0x010,
    MOTION_DIRECTION_BACKWARD = 0x020,
    MOTION_DIRECTION_BACK_LEFT = 0x040,
    MOTION_DIRECTION_LEFT = 0x080,
    MOTION_DIRECTION_FORWARD_LEFT = 0x100,
};

enum : uint8_t
{
    MOTION_SPEED_LOW = 0x01,
    MOTION_SPEED_MOVING = 0x02,
};

static const int MOTION_INDEX_SMALL_BOUND_SIZE = 16;
static const int MOTION_INDEX_LARGE_BOUND_SIZE = 64;

struct motion_index_runtime
{
    std::vector<uint16_t> direction_masks;
    std::vector<uint8_t> speed_masks;
    std::vector<int8_t> elevation_modes;

    std::vector<uint16_t> range_direction_masks;
    std::vector<uint8_t> range_speed_masks;
    std::vector<uint16_t> small_bound_direction_masks;
    std::vector<uint8_t> small_bound_speed_masks;
    std::vector<uint16_t> large_bound_direction_masks;
    std::vector<uint8_t> large_bound_speed_masks;
    int small_bound_size = 0;
    int large_bound_size = 0;

    size_t frame_count() const { return direction_masks.size(); }
};

static inline bool motion_index_error(
    char* error, int capacity, const char* format, ...)
{
    if (error != NULL && capacity > 0) {
        va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(error, static_cast<size_t>(capacity), format, arguments);
        va_end(arguments);
    }
    return false;
}

static inline uint16_t motion_index_u16_le(const unsigned char* bytes)
{
    return static_cast<uint16_t>(
        static_cast<uint16_t>(bytes[0]) |
        static_cast<uint16_t>(static_cast<uint16_t>(bytes[1]) << 8));
}

static inline uint32_t motion_index_u32_le(const unsigned char* bytes)
{
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) |
           (static_cast<uint32_t>(bytes[3]) << 24);
}

static inline bool motion_index_direction_is_valid(uint16_t mask)
{
    if (mask == MOTION_DIRECTION_IDLE) return true;
    if ((mask & MOTION_DIRECTION_IDLE) != 0 ||
        (mask & static_cast<uint16_t>(~UINT16_C(0x1ff))) != 0)
        return false;
    static const uint16_t moving[] = {
        MOTION_DIRECTION_FORWARD,
        MOTION_DIRECTION_FORWARD_RIGHT,
        MOTION_DIRECTION_RIGHT,
        MOTION_DIRECTION_BACK_RIGHT,
        MOTION_DIRECTION_BACKWARD,
        MOTION_DIRECTION_BACK_LEFT,
        MOTION_DIRECTION_LEFT,
        MOTION_DIRECTION_FORWARD_LEFT,
    };
    int first = -1;
    int second = -1;
    for (int i = 0; i < 8; ++i) {
        if ((mask & moving[i]) == 0) continue;
        if (first < 0) first = i;
        else if (second < 0) second = i;
        else return false;
    }
    if (first < 0) return false;
    if (second < 0) return true;
    return second == first + 1 || (first == 0 && second == 7);
}

static inline bool motion_index_speed_is_valid(uint8_t mask)
{
    return mask == MOTION_SPEED_LOW || mask == MOTION_SPEED_MOVING ||
           mask == static_cast<uint8_t>(
               MOTION_SPEED_LOW | MOTION_SPEED_MOVING);
}

static inline bool motion_index_file_size(FILE* file, size_t& size)
{
    if (std::fseek(file, 0, SEEK_END) != 0) return false;
    const long end = std::ftell(file);
    if (end < 0 || static_cast<uintmax_t>(end) >
                       static_cast<uintmax_t>(SIZE_MAX)) return false;
    size = static_cast<size_t>(end);
    return std::fseek(file, 0, SEEK_SET) == 0;
}

static inline bool motion_index_load(
    motion_index_runtime& out, const char* path,
    char* error, int capacity)
{
    const char* shown = path != NULL ? path : "<null>";
    if (path == NULL || path[0] == '\0')
        return motion_index_error(error, capacity, "%s: invalid path", shown);
    FILE* file = std::fopen(path, "rb");
    if (file == NULL)
        return motion_index_error(error, capacity, "%s: cannot open (%s)",
                                  path, std::strerror(errno));
    size_t file_size = 0;
    if (!motion_index_file_size(file, file_size)) {
        std::fclose(file);
        return motion_index_error(error, capacity,
                                  "%s: cannot determine file size", path);
    }
    static const size_t header_size = 16;
    if (file_size < header_size) {
        std::fclose(file);
        return motion_index_error(error, capacity,
                                  "%s: truncated G1MI header", path);
    }
    unsigned char header[header_size];
    if (std::fread(header, 1, sizeof(header), file) != sizeof(header)) {
        std::fclose(file);
        return motion_index_error(error, capacity,
                                  "%s: truncated G1MI header", path);
    }
    if (std::memcmp(header, "G1MI", 4) != 0) {
        std::fclose(file);
        return motion_index_error(error, capacity,
                                  "%s: invalid G1MI magic", path);
    }
    const uint32_t version = motion_index_u32_le(header + 4);
    const uint32_t frames = motion_index_u32_le(header + 8);
    const uint32_t row_width = motion_index_u32_le(header + 12);
    if (version != 1) {
        std::fclose(file);
        return motion_index_error(error, capacity,
                                  "%s: unsupported G1MI version %u", path,
                                  static_cast<unsigned>(version));
    }
    if (row_width != 4) {
        std::fclose(file);
        return motion_index_error(error, capacity,
                                  "%s: invalid G1MI row width %u", path,
                                  static_cast<unsigned>(row_width));
    }
    if (frames == 0 || frames > static_cast<uint32_t>(INT_MAX)) {
        std::fclose(file);
        return motion_index_error(error, capacity,
                                  "%s: invalid G1MI frame count %u", path,
                                  static_cast<unsigned>(frames));
    }
    if (static_cast<size_t>(frames) > (SIZE_MAX - header_size) / 4u) {
        std::fclose(file);
        return motion_index_error(error, capacity,
                                  "%s: G1MI byte length overflows", path);
    }
    const size_t expected_size =
        header_size + static_cast<size_t>(frames) * 4u;
    if (file_size < expected_size) {
        std::fclose(file);
        return motion_index_error(error, capacity,
            "%s: truncated G1MI rows (expected %zu bytes, got %zu)",
            path, expected_size, file_size);
    }
    if (file_size > expected_size) {
        std::fclose(file);
        return motion_index_error(error, capacity,
            "%s: trailing G1MI bytes (expected %zu bytes, got %zu)",
            path, expected_size, file_size);
    }

    motion_index_runtime candidate;
    try {
        candidate.direction_masks.resize(frames);
        candidate.speed_masks.resize(frames);
        candidate.elevation_modes.resize(frames);
    } catch (const std::bad_alloc&) {
        std::fclose(file);
        return motion_index_error(error, capacity,
                                  "%s: cannot allocate G1MI rows", path);
    } catch (const std::length_error&) {
        std::fclose(file);
        return motion_index_error(error, capacity,
                                  "%s: G1MI row allocation overflows", path);
    }
    unsigned char row[4];
    for (uint32_t frame = 0; frame < frames; ++frame) {
        if (std::fread(row, 1, sizeof(row), file) != sizeof(row)) {
            std::fclose(file);
            return motion_index_error(error, capacity,
                "%s: truncated G1MI row %u", path,
                static_cast<unsigned>(frame));
        }
        const uint16_t direction = motion_index_u16_le(row);
        const uint8_t speed = row[2];
        const int8_t elevation = static_cast<int8_t>(row[3]);
        if (!motion_index_direction_is_valid(direction)) {
            std::fclose(file);
            return motion_index_error(error, capacity,
                "%s: invalid G1MI direction mask at row %u", path,
                static_cast<unsigned>(frame));
        }
        if (!motion_index_speed_is_valid(speed)) {
            std::fclose(file);
            return motion_index_error(error, capacity,
                "%s: invalid G1MI speed mask at row %u", path,
                static_cast<unsigned>(frame));
        }
        if (elevation < -1 || elevation > 1) {
            std::fclose(file);
            return motion_index_error(error, capacity,
                "%s: invalid G1MI elevation at row %u", path,
                static_cast<unsigned>(frame));
        }
        candidate.direction_masks[frame] = direction;
        candidate.speed_masks[frame] = speed;
        candidate.elevation_modes[frame] = elevation;
    }
    if (std::fclose(file) != 0)
        return motion_index_error(error, capacity,
                                  "%s: failed while closing G1MI", path);
    out = std::move(candidate);
    return true;
}

static inline bool motion_index_build_aggregates(
    motion_index_runtime& index,
    const int* range_starts, const int* range_stops, size_t range_count,
    int small_bound_size, int large_bound_size,
    char* error, int capacity)
{
    const size_t frames = index.direction_masks.size();
    if (frames == 0 || index.speed_masks.size() != frames ||
        index.elevation_modes.size() != frames)
        return motion_index_error(error, capacity,
                                  "invalid motion index row vectors");
    if (range_starts == NULL || range_stops == NULL || range_count == 0 ||
        range_count > static_cast<size_t>(INT_MAX))
        return motion_index_error(error, capacity,
                                  "invalid motion index range arrays");
    if (small_bound_size <= 0 || large_bound_size <= 0)
        return motion_index_error(error, capacity,
                                  "invalid motion index bound size");
    size_t cursor = 0;
    for (size_t range = 0; range < range_count; ++range) {
        if (range_starts[range] < 0 || range_stops[range] <= range_starts[range] ||
            static_cast<size_t>(range_starts[range]) != cursor ||
            static_cast<size_t>(range_stops[range]) > frames)
            return motion_index_error(error, capacity,
                "motion index ranges must be nonempty, gap-free, and bounded");
        cursor = static_cast<size_t>(range_stops[range]);
    }
    if (cursor != frames)
        return motion_index_error(error, capacity,
                                  "motion index ranges do not cover frames");

    std::vector<uint16_t> range_directions;
    std::vector<uint8_t> range_speeds;
    std::vector<uint16_t> small_directions;
    std::vector<uint8_t> small_speeds;
    std::vector<uint16_t> large_directions;
    std::vector<uint8_t> large_speeds;
    try {
        range_directions.assign(range_count, 0);
        range_speeds.assign(range_count, 0);
        const size_t small_count =
            (frames - 1) / static_cast<size_t>(small_bound_size) + 1;
        const size_t large_count =
            (frames - 1) / static_cast<size_t>(large_bound_size) + 1;
        small_directions.assign(small_count, 0);
        small_speeds.assign(small_count, 0);
        large_directions.assign(large_count, 0);
        large_speeds.assign(large_count, 0);
    } catch (const std::bad_alloc&) {
        return motion_index_error(error, capacity,
                                  "cannot allocate motion index aggregates");
    } catch (const std::length_error&) {
        return motion_index_error(error, capacity,
                                  "motion index aggregate size overflows");
    }
    for (size_t range = 0; range < range_count; ++range) {
        for (int frame = range_starts[range]; frame < range_stops[range]; ++frame) {
            range_directions[range] |=
                index.direction_masks[static_cast<size_t>(frame)];
            range_speeds[range] |=
                index.speed_masks[static_cast<size_t>(frame)];
        }
    }
    for (size_t frame = 0; frame < frames; ++frame) {
        const size_t small = frame / static_cast<size_t>(small_bound_size);
        const size_t large = frame / static_cast<size_t>(large_bound_size);
        small_directions[small] |= index.direction_masks[frame];
        small_speeds[small] |= index.speed_masks[frame];
        large_directions[large] |= index.direction_masks[frame];
        large_speeds[large] |= index.speed_masks[frame];
    }

    index.range_direction_masks.swap(range_directions);
    index.range_speed_masks.swap(range_speeds);
    index.small_bound_direction_masks.swap(small_directions);
    index.small_bound_speed_masks.swap(small_speeds);
    index.large_bound_direction_masks.swap(large_directions);
    index.large_bound_speed_masks.swap(large_speeds);
    index.small_bound_size = small_bound_size;
    index.large_bound_size = large_bound_size;
    return true;
}

static inline bool motion_index_load(
    motion_index_runtime& out, const char* path,
    const int* range_starts, const int* range_stops, size_t range_count,
    int small_bound_size, int large_bound_size,
    char* error, int capacity)
{
    motion_index_runtime candidate;
    if (!motion_index_load(candidate, path, error, capacity) ||
        !motion_index_build_aggregates(
            candidate, range_starts, range_stops, range_count,
            small_bound_size, large_bound_size, error, capacity))
        return false;
    out = std::move(candidate);
    return true;
}
