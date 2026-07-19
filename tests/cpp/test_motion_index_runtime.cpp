#include "motion_index_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using bytes = std::vector<unsigned char>;

static void check(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "motion index runtime test failed: %s\n", message);
        std::abort();
    }
}

static void append_u16(bytes& out, uint16_t value)
{
    out.push_back(static_cast<unsigned char>(value));
    out.push_back(static_cast<unsigned char>(value >> 8));
}

static void append_u32(bytes& out, uint32_t value)
{
    for (int byte = 0; byte < 4; ++byte)
        out.push_back(static_cast<unsigned char>(value >> (8 * byte)));
}

static bytes index_bytes(
    const std::vector<uint16_t>& directions,
    const std::vector<uint8_t>& speeds,
    const std::vector<int8_t>& elevations,
    uint32_t version = 1,
    uint32_t row_width = 4)
{
    check(directions.size() == speeds.size() &&
          directions.size() == elevations.size(), "fixture row shapes");
    bytes out{'G', '1', 'M', 'I'};
    append_u32(out, version);
    append_u32(out, static_cast<uint32_t>(directions.size()));
    append_u32(out, row_width);
    for (size_t i = 0; i < directions.size(); ++i) {
        append_u16(out, directions[i]);
        out.push_back(speeds[i]);
        out.push_back(static_cast<unsigned char>(elevations[i]));
    }
    return out;
}

static void write_bytes(const char* path, const bytes& payload)
{
    FILE* file = std::fopen(path, "wb");
    check(file != NULL, "open fixture");
    check(payload.empty() ||
          std::fwrite(payload.data(), 1, payload.size(), file) == payload.size(),
          "write fixture");
    check(std::fclose(file) == 0, "close fixture");
}

static motion_index_runtime sentinel()
{
    motion_index_runtime value;
    value.direction_masks = {0x100};
    value.speed_masks = {3};
    value.elevation_modes = {-1};
    value.range_direction_masks = {0x080};
    value.range_speed_masks = {2};
    value.small_bound_direction_masks = {0x040};
    value.small_bound_speed_masks = {1};
    value.large_bound_direction_masks = {0x020};
    value.large_bound_speed_masks = {3};
    value.small_bound_size = 7;
    value.large_bound_size = 11;
    return value;
}

static bool same(
    const motion_index_runtime& first, const motion_index_runtime& second)
{
    return first.direction_masks == second.direction_masks &&
           first.speed_masks == second.speed_masks &&
           first.elevation_modes == second.elevation_modes &&
           first.range_direction_masks == second.range_direction_masks &&
           first.range_speed_masks == second.range_speed_masks &&
           first.small_bound_direction_masks ==
               second.small_bound_direction_masks &&
           first.small_bound_speed_masks == second.small_bound_speed_masks &&
           first.large_bound_direction_masks ==
               second.large_bound_direction_masks &&
           first.large_bound_speed_masks == second.large_bound_speed_masks &&
           first.small_bound_size == second.small_bound_size &&
           first.large_bound_size == second.large_bound_size;
}

static void expect_rejected(const char* path, const char* reason)
{
    motion_index_runtime output = sentinel();
    const motion_index_runtime prior = output;
    char error[512] = {};
    check(!motion_index_load(output, path, error, sizeof(error)),
          "invalid index rejected");
    check(std::strstr(error, path) != NULL, "failure names path");
    check(std::strstr(error, reason) != NULL, "failure names reason");
    check(same(output, prior), "failed load preserves prior index");
}

static void test_exact_rows_are_owned()
{
    const char* path = "/tmp/test_g1mi_valid.bin";
    const bytes payload = index_bytes(
        {MOTION_DIRECTION_IDLE, MOTION_DIRECTION_FORWARD,
         static_cast<uint16_t>(MOTION_DIRECTION_RIGHT |
                               MOTION_DIRECTION_BACK_RIGHT),
         static_cast<uint16_t>(MOTION_DIRECTION_FORWARD_LEFT |
                               MOTION_DIRECTION_FORWARD)},
        {MOTION_SPEED_LOW, MOTION_SPEED_MOVING,
         static_cast<uint8_t>(MOTION_SPEED_LOW | MOTION_SPEED_MOVING),
         MOTION_SPEED_MOVING},
        {-1, 0, 1, 0});
    write_bytes(path, payload);

    motion_index_runtime index = sentinel();
    char error[512] = {};
    check(motion_index_load(index, path, error, sizeof(error)), error);
    check(index.direction_masks.size() == 4 &&
          index.speed_masks.size() == 4 &&
          index.elevation_modes.size() == 4, "valid owned row vectors");
    check(index.direction_masks[2] ==
              (MOTION_DIRECTION_RIGHT | MOTION_DIRECTION_BACK_RIGHT) &&
          index.speed_masks[2] == 3 && index.elevation_modes[2] == 1,
          "exact little-endian row values");
    check(index.range_direction_masks.empty() &&
          index.small_bound_direction_masks.empty(),
          "raw load clears stale derived aggregates");

    write_bytes(path, index_bytes(
        {MOTION_DIRECTION_IDLE}, {MOTION_SPEED_LOW}, {0}));
    check(index.direction_masks.size() == 4 &&
          index.direction_masks[1] == MOTION_DIRECTION_FORWARD,
          "loaded vectors do not point into file storage");
}

static void test_headers_sizes_and_rows_are_strict()
{
    const char* path = "/tmp/test_g1mi_invalid.bin";
    const bytes valid = index_bytes(
        {MOTION_DIRECTION_IDLE, MOTION_DIRECTION_FORWARD},
        {MOTION_SPEED_LOW, MOTION_SPEED_MOVING}, {0, 1});

    for (size_t size = 0; size < valid.size(); ++size) {
        write_bytes(path, bytes(valid.begin(), valid.begin() + size));
        expect_rejected(path, "truncated");
    }

    bytes changed = valid;
    changed[0] = 'X';
    write_bytes(path, changed);
    expect_rejected(path, "magic");

    write_bytes(path, index_bytes(
        {MOTION_DIRECTION_IDLE}, {MOTION_SPEED_LOW}, {0}, 2, 4));
    expect_rejected(path, "version");
    write_bytes(path, index_bytes(
        {MOTION_DIRECTION_IDLE}, {MOTION_SPEED_LOW}, {0}, 1, 5));
    expect_rejected(path, "row width");

    changed.assign({'G', '1', 'M', 'I'});
    append_u32(changed, 1);
    append_u32(changed, 0);
    append_u32(changed, 4);
    write_bytes(path, changed);
    expect_rejected(path, "frame count");

    changed.assign({'G', '1', 'M', 'I'});
    append_u32(changed, 1);
    append_u32(changed, UINT32_MAX);
    append_u32(changed, 4);
    write_bytes(path, changed);
    expect_rejected(path, "frame count");

    changed = valid;
    changed.push_back(0xff);
    write_bytes(path, changed);
    expect_rejected(path, "trailing");

    const uint16_t bad_directions[] = {
        0, 0x200,
        static_cast<uint16_t>(MOTION_DIRECTION_IDLE | MOTION_DIRECTION_FORWARD),
        static_cast<uint16_t>(MOTION_DIRECTION_FORWARD | MOTION_DIRECTION_RIGHT),
        static_cast<uint16_t>(MOTION_DIRECTION_FORWARD | MOTION_DIRECTION_BACKWARD),
    };
    for (uint16_t direction : bad_directions) {
        write_bytes(path, index_bytes({direction}, {MOTION_SPEED_LOW}, {0}));
        expect_rejected(path, "direction");
    }

    const uint16_t sectors[] = {
        MOTION_DIRECTION_FORWARD, MOTION_DIRECTION_FORWARD_RIGHT,
        MOTION_DIRECTION_RIGHT, MOTION_DIRECTION_BACK_RIGHT,
        MOTION_DIRECTION_BACKWARD, MOTION_DIRECTION_BACK_LEFT,
        MOTION_DIRECTION_LEFT, MOTION_DIRECTION_FORWARD_LEFT,
    };
    for (int i = 0; i < 8; ++i) {
        write_bytes(path, index_bytes(
            {static_cast<uint16_t>(sectors[i] | sectors[(i + 1) % 8])},
            {MOTION_SPEED_MOVING}, {0}));
        motion_index_runtime output;
        char error[512] = {};
        check(motion_index_load(output, path, error, sizeof(error)),
              "every adjacent overlap accepted including wrap");
    }

    for (uint8_t speed : {uint8_t(0), uint8_t(4), uint8_t(0xff)}) {
        write_bytes(path, index_bytes(
            {MOTION_DIRECTION_IDLE}, {speed}, {0}));
        expect_rejected(path, "speed");
    }
    for (int8_t elevation : {int8_t(-2), int8_t(2), int8_t(127)}) {
        write_bytes(path, index_bytes(
            {MOTION_DIRECTION_IDLE}, {MOTION_SPEED_LOW}, {elevation}));
        expect_rejected(path, "elevation");
    }
}

static void test_range_and_bound_aggregates_are_validated()
{
    const char* path = "/tmp/test_g1mi_aggregates.bin";
    write_bytes(path, index_bytes(
        {MOTION_DIRECTION_IDLE, MOTION_DIRECTION_FORWARD,
         MOTION_DIRECTION_RIGHT, MOTION_DIRECTION_BACKWARD,
         MOTION_DIRECTION_LEFT},
        {MOTION_SPEED_LOW, MOTION_SPEED_MOVING, MOTION_SPEED_MOVING,
         MOTION_SPEED_LOW, 3},
        {0, 0, 1, -1, 0}));
    const int starts[] = {0, 2};
    const int stops[] = {2, 5};
    motion_index_runtime index;
    char error[512] = {};
    check(motion_index_load(index, path, starts, stops, 2, 2, 4,
                            error, sizeof(error)), error);
    check(index.range_direction_masks == std::vector<uint16_t>({
              static_cast<uint16_t>(MOTION_DIRECTION_IDLE |
                                    MOTION_DIRECTION_FORWARD),
              static_cast<uint16_t>(MOTION_DIRECTION_RIGHT |
                                    MOTION_DIRECTION_BACKWARD |
                                    MOTION_DIRECTION_LEFT)}) &&
          index.range_speed_masks == std::vector<uint8_t>({3, 3}),
          "per-range aggregate masks");
    check(index.small_bound_direction_masks == std::vector<uint16_t>({
              static_cast<uint16_t>(MOTION_DIRECTION_IDLE |
                                    MOTION_DIRECTION_FORWARD),
              static_cast<uint16_t>(MOTION_DIRECTION_RIGHT |
                                    MOTION_DIRECTION_BACKWARD),
              MOTION_DIRECTION_LEFT}) &&
          index.small_bound_speed_masks == std::vector<uint8_t>({3, 3, 3}),
          "per-small-bound aggregate masks");
    check(index.large_bound_direction_masks == std::vector<uint16_t>({
              static_cast<uint16_t>(MOTION_DIRECTION_IDLE |
                                    MOTION_DIRECTION_FORWARD |
                                    MOTION_DIRECTION_RIGHT |
                                    MOTION_DIRECTION_BACKWARD),
              MOTION_DIRECTION_LEFT}) &&
          index.large_bound_speed_masks == std::vector<uint8_t>({3, 3}) &&
          index.small_bound_size == 2 && index.large_bound_size == 4,
          "per-large-bound aggregate masks");

    const motion_index_runtime prior = index;
    const struct {
        int starts[2];
        int stops[2];
        size_t count;
        int small;
        int large;
    } invalid[] = {
        {{1, 2}, {2, 5}, 2, 2, 4},
        {{0, 3}, {2, 5}, 2, 2, 4},
        {{0, 1}, {3, 5}, 2, 2, 4},
        {{0, 2}, {2, 6}, 2, 2, 4},
        {{0, 2}, {2, 5}, 0, 2, 4},
        {{0, 2}, {2, 5}, 2, 0, 4},
        {{0, 2}, {2, 5}, 2, 2, 0},
    };
    for (const auto& candidate : invalid) {
        error[0] = '\0';
        check(!motion_index_build_aggregates(
                  index, candidate.starts, candidate.stops, candidate.count,
                  candidate.small, candidate.large, error, sizeof(error)),
              "invalid range/bound coverage rejected");
        check(same(index, prior), "aggregate failure preserves prior index");
    }
}

static void test_invalid_paths_preserve_output()
{
    motion_index_runtime output = sentinel();
    const motion_index_runtime prior = output;
    char error[256] = {};
    check(!motion_index_load(output, NULL, error, sizeof(error)),
          "null path rejected");
    check(same(output, prior), "null path transactional");
    check(!motion_index_load(output, "/tmp/g1mi-does-not-exist", error,
                             sizeof(error)), "missing path rejected");
    check(same(output, prior), "missing path transactional");
}

int main()
{
    test_exact_rows_are_owned();
    test_headers_sizes_and_rows_are_strict();
    test_range_and_bound_aggregates_are_validated();
    test_invalid_paths_preserve_output();
    return 0;
}
