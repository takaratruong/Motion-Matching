#include "database.h"
#include "g1_controller_state.h"
#include "motion_index_runtime.h"

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static void check(bool condition, const char* expression, int line)
{
    if (!condition) {
        std::fprintf(stderr, "indexed database search CHECK failed at line %d: %s\n",
                     line, expression);
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

static void make_database(
    database& db, int frames, int range_count, int terrain_dimensions = 12)
{
    CHECK(frames > 0 && range_count > 0 && frames % range_count == 0);
    const int feature_dimensions = 27 + terrain_dimensions;
    db.bone_positions.resize(frames, 1);
    db.features.resize(frames, feature_dimensions);
    db.features_offset.resize(feature_dimensions);
    db.features_scale.resize(feature_dimensions);
    db.terrain_features.resize(frames, terrain_dimensions);
    db.range_starts.resize(range_count);
    db.range_stops.resize(range_count);
    db.features.zero();
    db.features_offset.zero();
    db.features_scale.set(FLT_MAX);
    db.features_scale(0) = 1.0f;
    db.terrain_features.zero();
    const int range_size = frames / range_count;
    for (int range = 0; range < range_count; ++range) {
        db.range_starts(range) = range * range_size;
        db.range_stops(range) = (range + 1) * range_size;
    }
}

static void make_index(motion_index_runtime& index, int frames)
{
    index.direction_masks.assign(static_cast<size_t>(frames),
                                 MOTION_DIRECTION_FORWARD);
    index.speed_masks.assign(static_cast<size_t>(frames), MOTION_SPEED_MOVING);
    index.elevation_modes.assign(static_cast<size_t>(frames), 0);
}

static void finish_fixture(database& db, motion_index_runtime& index)
{
    database_build_bounds(db);
    char error[256] = {};
    CHECK(motion_index_build_aggregates(
        index, db.range_starts.data, db.range_stops.data,
        static_cast<size_t>(db.nranges()), BOUND_SM_SIZE, BOUND_LR_SIZE,
        error, sizeof(error)));
}

static bool row_compatible(
    const motion_index_runtime& index, int frame,
    uint16_t direction, uint8_t speed, int elevation)
{
    return (index.direction_masks[static_cast<size_t>(frame)] & direction) != 0 &&
           (index.speed_masks[static_cast<size_t>(frame)] & speed) != 0 &&
           index.elevation_modes[static_cast<size_t>(frame)] == elevation;
}

static bool row_publishable(
    const database& db,
    const motion_index_runtime& index,
    int frame,
    uint16_t direction,
    uint8_t speed,
    int elevation)
{
    const int published = database_trajectory_index_clamp(db, frame, 1);
    return published != frame &&
           row_compatible(index, frame, direction, speed, elevation) &&
           row_compatible(index, published, direction, speed, elevation);
}

static int source_range_for_frame(const database& db, int frame)
{
    for (int range = 0; range < db.nranges(); ++range) {
        if (frame >= db.range_starts(range) && frame < db.range_stops(range))
            return range;
    }
    return -1;
}

static bool selected_range(const std::vector<int>& ranges, int range)
{
    for (int candidate : ranges) if (candidate == range) return true;
    return false;
}

static database_indexed_search_result brute_force(
    const database& db,
    const motion_index_runtime& index,
    const std::vector<int>& ranges,
    uint16_t direction,
    uint8_t speed,
    int elevation,
    const slice1d<float> query,
    int incumbent,
    float transition_cost,
    int ignore_range_end,
    int ignore_surrounding,
    int minimum_future_published_frames = 1)
{
    database_indexed_search_result result = {};
    result.status = DATABASE_INDEXED_SEARCH_EMPTY;
    result.index = -1;
    result.cost = FLT_MAX;

    const int incumbent_range = source_range_for_frame(db, incumbent);
    const bool incumbent_compatible =
        incumbent >= 0 && selected_range(ranges, incumbent_range) &&
        row_publishable(
            db, index, incumbent, direction, speed, elevation);
    if (incumbent_compatible) {
        result.status = DATABASE_INDEXED_SEARCH_FOUND;
        result.index = incumbent;
        result.cost = database_frame_cost(db, incumbent, query);
        ++result.evaluated_frame_count;
    }

    for (int range : ranges) {
        const int stop = db.range_stops(range) - ignore_range_end;
        for (int frame = db.range_starts(range); frame < stop; ++frame) {
            if (range == incumbent_range && incumbent >= 0 &&
                std::abs(frame - incumbent) < ignore_surrounding)
                continue;
            if (!database_indexed_frame_has_compatible_published_horizon(
                    db, index, frame, direction, speed, elevation,
                    minimum_future_published_frames))
                continue;
            ++result.eligible_frame_count;
            ++result.evaluated_frame_count;
            float cost = transition_cost;
            for (int dimension = 0; dimension < db.nfeatures(); ++dimension) {
                const float normalized = normalize_query_feature(
                    query(dimension), db.features_offset(dimension),
                    db.features_scale(dimension));
                cost += squaref(normalized - db.features(frame, dimension));
                if (cost >= result.cost) break;
            }
            if (cost < result.cost) {
                result.status = DATABASE_INDEXED_SEARCH_FOUND;
                result.index = frame;
                result.cost = cost;
            }
        }
    }
    return result;
}

static database_indexed_search_status search(
    database_indexed_search_result& result,
    const database& db,
    const motion_index_runtime& index,
    const std::vector<int>& ranges,
    uint16_t direction,
    uint8_t speed,
    int elevation,
    const slice1d<float> query,
    int incumbent = -1,
    float transition_cost = 0.0f,
    int ignore_range_end = 0,
    int ignore_surrounding = 0,
    int minimum_future_published_frames = 1)
{
    return database_search_indexed(
        result, db, index,
        ranges.empty() ? NULL : ranges.data(),
        static_cast<int>(ranges.size()), direction, speed, elevation,
        query, incumbent, transition_cost,
        ignore_range_end, ignore_surrounding,
        minimum_future_published_frames);
}

static void test_candidate_requires_configured_compatible_horizon()
{
    database db;
    motion_index_runtime index;
    make_database(db, 64, 1);
    make_index(index, 64);
    for (int frame = 0; frame < 64; ++frame) {
        db.features(frame, 0) = 100.0f;
        index.direction_masks[static_cast<size_t>(frame)] =
            MOTION_DIRECTION_FORWARD;
    }
    for (int frame = 0; frame <= 5; ++frame)
        index.direction_masks[static_cast<size_t>(frame)] =
            MOTION_DIRECTION_RIGHT;
    for (int frame = 16; frame <= 40; ++frame)
        index.direction_masks[static_cast<size_t>(frame)] =
            MOTION_DIRECTION_RIGHT;
    db.features(0, 0) = 0.0f;
    db.features(16, 0) = 1.0f;
    finish_fixture(db, index);

    array1d<float> query(39);
    query.zero();
    database_indexed_search_result result = {};
    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_RIGHT,
                 MOTION_SPEED_MOVING, 0, query, -1, 0.0f, 0, 0, 1) ==
          DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 0);
    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_RIGHT,
                 MOTION_SPEED_MOVING, 0, query, -1, 0.0f, 0, 0, 13) ==
          DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 16);

    for (int frame = 16; frame <= 40; ++frame)
        index.direction_masks[static_cast<size_t>(frame)] =
            MOTION_DIRECTION_FORWARD;
    finish_fixture(db, index);
    result.index = 123;
    result.cost = 456.0f;
    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_RIGHT,
                 MOTION_SPEED_MOVING, 0, query, -1, 0.0f, 0, 0, 13) ==
          DATABASE_INDEXED_SEARCH_EMPTY);
    CHECK(result.status == DATABASE_INDEXED_SEARCH_EMPTY &&
          result.index == -1 && result.cost == FLT_MAX &&
          result.eligible_frame_count == 0);
}

static void test_family_ties_direction_speed_and_elevation()
{
    database db;
    motion_index_runtime index;
    make_database(db, 128, 4);
    make_index(index, 128);

    for (int frame = 32; frame < 64; ++frame)
        index.direction_masks[static_cast<size_t>(frame)] = MOTION_DIRECTION_RIGHT;
    for (int frame = 64; frame < 96; ++frame) {
        index.direction_masks[static_cast<size_t>(frame)] = MOTION_DIRECTION_LEFT;
        index.elevation_modes[static_cast<size_t>(frame)] = -1;
    }
    for (int frame = 96; frame < 128; ++frame)
        index.elevation_modes[static_cast<size_t>(frame)] = 1;
    index.speed_masks[33] =
        static_cast<uint8_t>(MOTION_SPEED_LOW | MOTION_SPEED_MOVING);
    index.speed_masks[34] = MOTION_SPEED_LOW;
    index.speed_masks[35] = MOTION_SPEED_LOW;
    finish_fixture(db, index);

    array1d<float> query(39);
    query.zero();
    database_indexed_search_result result = {};

    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_FORWARD,
                 MOTION_SPEED_MOVING, 0, query) == DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 0);
    CHECK(search(result, db, index, {1}, MOTION_DIRECTION_RIGHT,
                 MOTION_SPEED_MOVING, 0, query) == DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 32);
    CHECK(search(result, db, index, {2}, MOTION_DIRECTION_LEFT,
                 MOTION_SPEED_MOVING, -1, query) == DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 64);
    CHECK(search(result, db, index, {3}, MOTION_DIRECTION_FORWARD,
                 MOTION_SPEED_MOVING, 1, query) == DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 96);

    query(0) = 33.0f;
    db.features(33, 0) = 33.0f;
    db.features(34, 0) = 34.0f;
    database_build_bounds(db);
    CHECK(search(result, db, index, {1}, MOTION_DIRECTION_RIGHT,
                 MOTION_SPEED_LOW, 0, query) == DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 33);
    query(0) = 34.0f;
    CHECK(search(result, db, index, {1}, MOTION_DIRECTION_RIGHT,
                 MOTION_SPEED_LOW, 0, query) == DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 34);

    const uint16_t sectors[] = {
        MOTION_DIRECTION_FORWARD, MOTION_DIRECTION_FORWARD_RIGHT,
        MOTION_DIRECTION_RIGHT, MOTION_DIRECTION_BACK_RIGHT,
        MOTION_DIRECTION_BACKWARD, MOTION_DIRECTION_BACK_LEFT,
        MOTION_DIRECTION_LEFT, MOTION_DIRECTION_FORWARD_LEFT,
    };
    database sectors_db;
    motion_index_runtime sectors_index;
    make_database(sectors_db, 32, 1);
    make_index(sectors_index, 32);
    for (int sector = 0; sector < 8; ++sector) {
        const int first = sector * 2;
        sectors_index.direction_masks[static_cast<size_t>(first)] =
            sectors[sector];
        sectors_index.direction_masks[static_cast<size_t>(first + 1)] =
            sectors[sector];
        sectors_db.features(first, 0) = static_cast<float>(sector);
        sectors_db.features(first + 1, 0) = static_cast<float>(sector);
    }
    sectors_index.direction_masks[16] = static_cast<uint16_t>(
        MOTION_DIRECTION_FORWARD | MOTION_DIRECTION_FORWARD_RIGHT);
    sectors_index.direction_masks[17] = static_cast<uint16_t>(
        MOTION_DIRECTION_FORWARD | MOTION_DIRECTION_FORWARD_RIGHT);
    sectors_db.features(16, 0) = 8.0f;
    sectors_db.features(17, 0) = 8.0f;
    finish_fixture(sectors_db, sectors_index);
    for (int sector = 0; sector < 8; ++sector) {
        query(0) = static_cast<float>(sector);
        CHECK(search(result, sectors_db, sectors_index, {0}, sectors[sector],
                     MOTION_SPEED_MOVING, 0, query) ==
              DATABASE_INDEXED_SEARCH_FOUND);
        CHECK(result.index == sector * 2);
    }
    query(0) = 8.0f;
    CHECK(search(result, sectors_db, sectors_index, {0},
                 static_cast<uint16_t>(MOTION_DIRECTION_FORWARD |
                                       MOTION_DIRECTION_FORWARD_RIGHT),
                 MOTION_SPEED_MOVING, 0, query) ==
          DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 16);
}

static void test_incumbent_transition_clip_and_source_surrounding()
{
    database db;
    motion_index_runtime index;
    make_database(db, 64, 2);
    make_index(index, 64);
    for (int frame = 0; frame < 64; ++frame) db.features(frame, 0) = 10.0f;
    db.features(4, 0) = 0.4f;
    db.features(10, 0) = 0.0f;
    db.features(32, 0) = 0.1f;
    index.direction_masks[4] = MOTION_DIRECTION_LEFT;
    finish_fixture(db, index);
    array1d<float> query(39);
    query.zero();
    database_indexed_search_result result = {};

    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_FORWARD,
                 MOTION_SPEED_MOVING, 0, query, 4, 1.0f) ==
          DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 10);

    index.direction_masks[4] = MOTION_DIRECTION_FORWARD;
    char error[128] = {};
    CHECK(motion_index_build_aggregates(
        index, db.range_starts.data, db.range_stops.data, 2,
        BOUND_SM_SIZE, BOUND_LR_SIZE, error, sizeof(error)));
    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_FORWARD,
                 MOTION_SPEED_MOVING, 0, query, 4, 0.20f) ==
          DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 4);
    CHECK(float_bits(result.cost) == float_bits(squaref(0.4f)));
    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_FORWARD,
                 MOTION_SPEED_MOVING, 0, query, 4, 0.10f) ==
          DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 10);
    CHECK(float_bits(result.cost) == float_bits(0.10f));

    db.features(31, 0) = -1.0f;
    db.features(32, 0) = 0.0f;
    database_build_bounds(db);
    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_FORWARD,
                 MOTION_SPEED_MOVING, 0, query, -1, 0.0f, 1, 0) ==
          DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index != 31);
    CHECK(search(result, db, index, {1}, MOTION_DIRECTION_FORWARD,
                 MOTION_SPEED_MOVING, 0, query, 31, 0.0f, 0, 4) ==
          DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 32);
}

static void test_selection_requires_compatible_published_successor()
{
    database db;
    motion_index_runtime index;
    make_database(db, 64, 1);
    make_index(index, 64);
    for (int frame = 0; frame < 64; ++frame)
        db.features(frame, 0) = 10.0f;
    db.features(10, 0) = 0.0f;
    db.features(20, 0) = 1.0f;
    index.direction_masks[10] = MOTION_DIRECTION_RIGHT;
    index.direction_masks[20] = MOTION_DIRECTION_RIGHT;
    index.direction_masks[21] = MOTION_DIRECTION_RIGHT;
    finish_fixture(db, index);

    array1d<float> query(39);
    query.zero();
    database_indexed_search_result result = {};
    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_RIGHT,
                 MOTION_SPEED_MOVING, 0, query) ==
          DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 20);
    CHECK(result.eligible_frame_count == 1);

    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_RIGHT,
                 MOTION_SPEED_MOVING, 0, query, 10) ==
          DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(result.index == 20);
    CHECK(result.evaluated_frame_count == 1);
}

static bool same_result(
    const database_indexed_search_result& first,
    const database_indexed_search_result& second)
{
    return first.status == second.status && first.index == second.index &&
           float_bits(first.cost) == float_bits(second.cost) &&
           first.eligible_frame_count == second.eligible_frame_count &&
           first.evaluated_frame_count == second.evaluated_frame_count &&
           first.considered_bound_count == second.considered_bound_count &&
           first.skipped_bound_count == second.skipped_bound_count;
}

static void test_empty_and_invalid_are_transactional()
{
    database db;
    motion_index_runtime index;
    make_database(db, 64, 2);
    make_index(index, 64);
    finish_fixture(db, index);
    array1d<float> query(39);
    query.zero();

    database_indexed_search_result result = {};
    CHECK(search(result, db, index, {}, MOTION_DIRECTION_FORWARD,
                 MOTION_SPEED_MOVING, 0, query) == DATABASE_INDEXED_SEARCH_EMPTY);
    CHECK(result.status == DATABASE_INDEXED_SEARCH_EMPTY && result.index == -1 &&
          result.eligible_frame_count == 0 && result.evaluated_frame_count == 0);
    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_LEFT,
                 MOTION_SPEED_MOVING, 0, query) == DATABASE_INDEXED_SEARCH_EMPTY);

    database_indexed_search_result sentinel = {};
    sentinel.status = DATABASE_INDEXED_SEARCH_FOUND;
    sentinel.index = 47;
    sentinel.cost = 12.5f;
    sentinel.eligible_frame_count = 101;
    sentinel.evaluated_frame_count = 99;
    sentinel.considered_bound_count = 77;
    sentinel.skipped_bound_count = 55;

    const std::vector<std::vector<int> > bad_ranges = {{-1}, {2}, {1, 0}, {0, 0}};
    for (const std::vector<int>& ranges : bad_ranges) {
        result = sentinel;
        CHECK(search(result, db, index, ranges, MOTION_DIRECTION_FORWARD,
                     MOTION_SPEED_MOVING, 0, query) == DATABASE_INDEXED_SEARCH_INVALID);
        CHECK(same_result(result, sentinel));
    }
    struct invalid_masks { uint16_t direction; uint8_t speed; int elevation; };
    const invalid_masks masks[] = {
        {0, MOTION_SPEED_MOVING, 0},
        {MOTION_DIRECTION_FORWARD, 0, 0},
        {MOTION_DIRECTION_FORWARD, MOTION_SPEED_MOVING, 2},
    };
    for (const invalid_masks& value : masks) {
        result = sentinel;
        CHECK(search(result, db, index, {0}, value.direction, value.speed,
                     value.elevation, query) == DATABASE_INDEXED_SEARCH_INVALID);
        CHECK(same_result(result, sentinel));
    }
    array1d<float> short_query(38);
    short_query.zero();
    result = sentinel;
    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_FORWARD,
                 MOTION_SPEED_MOVING, 0, short_query) ==
          DATABASE_INDEXED_SEARCH_INVALID);
    CHECK(same_result(result, sentinel));
    query(0) = std::numeric_limits<float>::quiet_NaN();
    result = sentinel;
    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_FORWARD,
                 MOTION_SPEED_MOVING, 0, query) == DATABASE_INDEXED_SEARCH_INVALID);
    CHECK(same_result(result, sentinel));
    query.zero();
    result = sentinel;
    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_FORWARD,
                 MOTION_SPEED_MOVING, 0, query, 64) == DATABASE_INDEXED_SEARCH_INVALID);
    CHECK(same_result(result, sentinel));
    result = sentinel;
    CHECK(search(result, db, index, {0}, MOTION_DIRECTION_FORWARD,
                 MOTION_SPEED_MOVING, 0, query, -1, -0.1f) ==
          DATABASE_INDEXED_SEARCH_INVALID);
    CHECK(same_result(result, sentinel));
}

static void test_aggregate_skips_and_randomized_brute_force_parity()
{
    database db;
    motion_index_runtime index;
    make_database(db, 256, 8);
    make_index(index, 256);
    std::mt19937 generator(0x5eed1234u);
    const uint16_t sectors[] = {
        MOTION_DIRECTION_FORWARD, MOTION_DIRECTION_FORWARD_RIGHT,
        MOTION_DIRECTION_RIGHT, MOTION_DIRECTION_BACK_RIGHT,
        MOTION_DIRECTION_BACKWARD, MOTION_DIRECTION_BACK_LEFT,
        MOTION_DIRECTION_LEFT, MOTION_DIRECTION_FORWARD_LEFT,
    };
    for (int frame = 0; frame < 256; ++frame) {
        const int sector = static_cast<int>(generator() % 8u);
        uint16_t direction = sectors[sector];
        if ((generator() & 3u) == 0)
            direction = static_cast<uint16_t>(direction | sectors[(sector + 1) % 8]);
        index.direction_masks[static_cast<size_t>(frame)] = direction;
        index.speed_masks[static_cast<size_t>(frame)] =
            static_cast<uint8_t>(1u + generator() % 3u);
        index.elevation_modes[static_cast<size_t>(frame)] =
            static_cast<int8_t>(static_cast<int>(generator() % 3u) - 1);
        for (int dimension = 0; dimension < 6; ++dimension) {
            db.features(frame, dimension) =
                static_cast<float>(static_cast<int>(generator() % 2001u) - 1000) /
                211.0f;
            db.features_scale(dimension) = 1.0f;
        }
    }
    finish_fixture(db, index);

    array1d<float> query(39);
    query.zero();
    for (int iteration = 0; iteration < 160; ++iteration) {
        for (int dimension = 0; dimension < 6; ++dimension)
            query(dimension) =
                static_cast<float>(static_cast<int>(generator() % 2001u) - 1000) /
                197.0f;
        std::vector<int> ranges;
        for (int range = 0; range < 8; ++range)
            if ((generator() & 1u) != 0) ranges.push_back(range);
        const uint16_t direction = sectors[generator() % 8u];
        const uint8_t speed = static_cast<uint8_t>(1u << (generator() & 1u));
        const int elevation = static_cast<int>(generator() % 3u) - 1;
        const int incumbent = (generator() & 1u) != 0
            ? static_cast<int>(generator() % 256u) : -1;
        const float transition = static_cast<float>(generator() % 9u) * 0.125f;
        const int ignore_end = static_cast<int>(generator() % 5u);
        const int ignore_surrounding = static_cast<int>(generator() % 7u);

        database_indexed_search_result actual = {};
        const database_indexed_search_status status = search(
            actual, db, index, ranges, direction, speed, elevation, query,
            incumbent, transition, ignore_end, ignore_surrounding);
        const database_indexed_search_result expected = brute_force(
            db, index, ranges, direction, speed, elevation, query,
            incumbent, transition, ignore_end, ignore_surrounding);
        CHECK(status == expected.status);
        CHECK(actual.status == expected.status);
        CHECK(actual.index == expected.index);
        CHECK(float_bits(actual.cost) == float_bits(expected.cost));
        CHECK(actual.eligible_frame_count == expected.eligible_frame_count);
        CHECK(actual.evaluated_frame_count <= expected.evaluated_frame_count);
    }

    database skip_db;
    motion_index_runtime skip_index;
    make_database(skip_db, 256, 4);
    make_index(skip_index, 256);
    for (int frame = 0; frame < 192; ++frame)
        skip_index.direction_masks[static_cast<size_t>(frame)] = MOTION_DIRECTION_FORWARD;
    for (int frame = 192; frame < 256; ++frame)
        skip_index.direction_masks[static_cast<size_t>(frame)] = MOTION_DIRECTION_LEFT;
    finish_fixture(skip_db, skip_index);
    query.zero();
    database_indexed_search_result skipped = {};
    CHECK(search(skipped, skip_db, skip_index, {0, 1, 2, 3},
                 MOTION_DIRECTION_LEFT, MOTION_SPEED_MOVING, 0, query) ==
          DATABASE_INDEXED_SEARCH_FOUND);
    CHECK(skipped.index == 192 && skipped.skipped_bound_count > 0 &&
          skipped.considered_bound_count > skipped.skipped_bound_count &&
          skipped.evaluated_frame_count < 256);
}

static void test_four_million_row_eligibility_gate()
{
    const int frames = 4000000;
    database db;
    motion_index_runtime index;
    make_database(db, frames, 4, 4);
    make_index(index, frames);
    for (int frame = 1000000; frame < 2000000; ++frame)
        index.direction_masks[static_cast<size_t>(frame)] = MOTION_DIRECTION_RIGHT;
    for (int frame = 2000000; frame < 3000000; ++frame)
        index.direction_masks[static_cast<size_t>(frame)] = MOTION_DIRECTION_LEFT;
    for (int frame = 3000000; frame < 4000000; ++frame)
        index.elevation_modes[static_cast<size_t>(frame)] = 1;

    const int small_bounds = (frames + BOUND_SM_SIZE - 1) / BOUND_SM_SIZE;
    const int large_bounds = (frames + BOUND_LR_SIZE - 1) / BOUND_LR_SIZE;
    db.bound_sm_min.resize(small_bounds, db.nfeatures());
    db.bound_sm_max.resize(small_bounds, db.nfeatures());
    db.bound_lr_min.resize(large_bounds, db.nfeatures());
    db.bound_lr_max.resize(large_bounds, db.nfeatures());
    db.bound_sm_min.zero();
    db.bound_sm_max.zero();
    db.bound_lr_min.zero();
    db.bound_lr_max.zero();
    char error[256] = {};
    CHECK(motion_index_build_aggregates(
        index, db.range_starts.data, db.range_stops.data, 4,
        BOUND_SM_SIZE, BOUND_LR_SIZE, error, sizeof(error)));

    array1d<float> query(31);
    query.zero();
    const struct probe {
        int range;
        uint16_t direction;
        int elevation;
    } probes[] = {
        {0, MOTION_DIRECTION_FORWARD, 0},
        {1, MOTION_DIRECTION_RIGHT, 0},
        {3, MOTION_DIRECTION_FORWARD, 1},
    };
    for (const probe& value : probes) {
        database_indexed_search_result result = {};
        const std::vector<int> ranges(1, value.range);
        const std::chrono::steady_clock::time_point begin =
            std::chrono::steady_clock::now();
        CHECK(search(result, db, index, ranges, value.direction,
                     MOTION_SPEED_MOVING, value.elevation, query,
                     -1, 0.0f, 0, 0,
                     G1_MOTION_MATCH_MINIMUM_FUTURE_PUBLISHED_FRAMES) ==
              DATABASE_INDEXED_SEARCH_FOUND);
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        CHECK(milliseconds < 40.0);
        CHECK(result.eligible_frame_count < static_cast<uint64_t>(frames));
        CHECK(result.evaluated_frame_count < result.eligible_frame_count);
        std::printf(
            "4M eligibility gate range=%d elapsed_ms=%.3f eligible=%llu evaluated=%llu\n",
            value.range, milliseconds,
            static_cast<unsigned long long>(result.eligible_frame_count),
            static_cast<unsigned long long>(result.evaluated_frame_count));
    }
}

static void test_runtime_candidate_requires_fifteen_percent_improvement()
{
    CHECK(!g1_motion_match_candidate_beats_continuation(
        true, 85.0f, 100.0f));
    CHECK(g1_motion_match_candidate_beats_continuation(
        true, std::nextafter(85.0f, 0.0f), 100.0f));
    CHECK(!g1_motion_match_candidate_beats_continuation(
        true, 90.0f, 100.0f));
    CHECK(g1_motion_match_candidate_beats_continuation(
        true, 84.0f, 100.0f));

    CHECK(g1_motion_match_candidate_beats_continuation(
        false, 1000.0f, 1.0f));
    CHECK(!g1_motion_match_candidate_beats_continuation(
        true, 0.0f, 0.0f));
    CHECK(g1_motion_match_candidate_beats_continuation(
        true, FLT_MAX * 0.84f, FLT_MAX));
    CHECK(!g1_motion_match_candidate_beats_continuation(
        true, FLT_MAX, FLT_MAX));

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    CHECK(!g1_motion_match_candidate_beats_continuation(
        false, nan, 100.0f));
    CHECK(!g1_motion_match_candidate_beats_continuation(
        false, infinity, 100.0f));
    CHECK(!g1_motion_match_candidate_beats_continuation(
        false, -1.0f, 100.0f));
    CHECK(!g1_motion_match_candidate_beats_continuation(
        true, 1.0f, nan));
    CHECK(!g1_motion_match_candidate_beats_continuation(
        true, 1.0f, infinity));
    CHECK(!g1_motion_match_candidate_beats_continuation(
        true, 1.0f, -1.0f));
}

int main()
{
    test_family_ties_direction_speed_and_elevation();
    test_incumbent_transition_clip_and_source_surrounding();
    test_selection_requires_compatible_published_successor();
    test_candidate_requires_configured_compatible_horizon();
    test_empty_and_invalid_are_transactional();
    test_aggregate_skips_and_randomized_brute_force_parity();
    test_four_million_row_eligibility_gate();
    test_runtime_candidate_requires_fifteen_percent_improvement();
    return 0;
}
