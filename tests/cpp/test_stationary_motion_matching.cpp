#include "stationary_motion_matching.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <float.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

bool near(float left, float right, float tolerance = 1.0e-6F) {
    return std::fabs(left - right) <=
        tolerance * std::max(1.0F, std::fabs(right));
}

template <typename Function>
void require_invalid_argument(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    } catch (...) {
        fail(message + " (wrong exception type)");
    }
    fail(message + " (no exception)");
}

float float_from_bits(uint32_t bits) {
    float value = 0.0F;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void set_component(vec3& value, int component, float replacement) {
    if (component == 0) {
        value.x = replacement;
    } else if (component == 1) {
        value.y = replacement;
    } else {
        value.z = replacement;
    }
}

void set_ranges(
    database& db,
    const std::vector<int>& starts,
    const std::vector<int>& stops) {
    db.range_starts.resize(static_cast<int>(starts.size()));
    db.range_stops.resize(static_cast<int>(stops.size()));
    for (size_t index = 0; index < starts.size(); ++index) {
        db.range_starts(static_cast<int>(index)) = starts[index];
    }
    for (size_t index = 0; index < stops.size(); ++index) {
        db.range_stops(static_cast<int>(index)) = stops[index];
    }
}

void initialize_candidate_database(database& db, int frames) {
    db.bone_positions.resize(frames, 1);
    db.bone_positions.set(vec3());
    db.bone_velocities.resize(frames, 1);
    db.bone_velocities.set(vec3());
    db.contact_states.resize(frames, 2);
    db.contact_states.set(true);
    set_ranges(db, {0}, {frames});
}

void initialize_search_database(database& db) {
    initialize_candidate_database(db, 3);
    db.features.resize(3, 2);
    db.features.zero();
    db.features_offset.resize(2);
    db.features_offset.zero();
    db.features_scale.resize(2);
    db.features_scale.set(1.0F);
}

void test_inclusive_window_clip_and_threshold_contract() {
    using namespace stationary_motion_matching;

    database db;
    initialize_candidate_database(db, 27);
    set_ranges(db, {0}, {26});

    require(
        is_candidate(db, 0),
        "frame+horizon == stop-1 must be eligible");
    require(
        !is_candidate(db, 1),
        "frame+horizon == stop must cross the clip boundary");

    db.bone_velocities(0, 0).x = 0.10F;
    db.bone_velocities(25, 0).z = 0.10F;
    require(is_candidate(db, 0), "0.10 m/s speed boundary was exclusive");
    db.bone_velocities(0, 0) = vec3();
    db.bone_velocities(25, 0) = vec3();

    db.bone_velocities(0, 0).x = 0.100001F;
    require(!is_candidate(db, 0), "window start speed was not checked");
    db.bone_velocities(0, 0) = vec3();
    db.bone_velocities(25, 0).z = 0.100001F;
    require(!is_candidate(db, 0), "window end speed was not checked");
    db.bone_velocities(25, 0) = vec3();

    db.bone_positions(25, 0).x = 0.05F;
    require(
        is_candidate(db, 0),
        "0.05 m displacement boundary was exclusive");
    db.bone_positions(25, 0).x = 0.050001F;
    require(
        !is_candidate(db, 0),
        "displacement above 0.05 m was accepted");
    db.bone_positions(25, 0) = vec3();

    for (int channel = 0; channel < 2; ++channel) {
        db.contact_states(0, channel) = false;
        require(
            !is_candidate(db, 0),
            "contact-clear window start was not checked");
        db.contact_states(0, channel) = true;
        db.contact_states(25, channel) = false;
        require(
            !is_candidate(db, 0),
            "contact-clear window end was not checked");
        db.contact_states(25, channel) = true;
    }
}

void test_positive_nondefault_horizon_uses_configured_window() {
    using namespace stationary_motion_matching;

    database db;
    initialize_candidate_database(db, 6);
    CandidateConfig config;
    config.horizon_frames = 2;

    require(
        derive_candidates(db, config) == std::vector<int>({0, 1, 2, 3}),
        "positive nondefault horizon did not produce four eligible windows");

    db.bone_velocities(2, 0).z = 0.100001F;
    require(
        !is_candidate(db, 0, config),
        "configured horizon endpoint speed was not checked");
}

void test_interior_window_samples_are_checked() {
    using namespace stationary_motion_matching;

    constexpr int interior_sample = 13;

    {
        database db;
        initialize_candidate_database(db, 26);
        db.bone_velocities(interior_sample, 0).x = 0.100001F;
        require(
            !is_candidate(db, 0),
            "isolated interior speed violation was not checked");
    }

    for (int channel = 0; channel < 2; ++channel) {
        database db;
        initialize_candidate_database(db, 26);
        db.contact_states(interior_sample, channel) = false;
        require(
            !is_candidate(db, 0),
            "isolated interior contact channel " +
                std::to_string(channel) + " violation was not checked");
    }

    {
        database db;
        initialize_candidate_database(db, 26);
        db.bone_positions(interior_sample, 0).z =
            float_from_bits(0x7fc00001U);
        require_invalid_argument(
            [&] { static_cast<void>(is_candidate(db, 0)); },
            "interior raw-bit nonfinite root position was accepted");
    }

    {
        database db;
        initialize_candidate_database(db, 26);
        db.bone_velocities(interior_sample, 0).x =
            float_from_bits(0x7f800000U);
        require_invalid_argument(
            [&] { static_cast<void>(is_candidate(db, 0)); },
            "interior raw-bit nonfinite root velocity was accepted");
    }
}

void test_z_axis_displacement_threshold_contract() {
    using namespace stationary_motion_matching;

    database db;
    initialize_candidate_database(db, 26);
    db.bone_positions(25, 0).z = 0.05F;
    require(
        is_candidate(db, 0),
        "finite z displacement at exactly 0.05 m was rejected");

    db.bone_positions(25, 0).z = 0.050001F;
    require(
        !is_candidate(db, 0),
        "finite z displacement above 0.05 m was accepted");
}

void test_derivation_handles_ranges_oversized_horizon_and_no_idle() {
    using namespace stationary_motion_matching;

    database clips;
    initialize_candidate_database(clips, 52);
    set_ranges(clips, {0, 26}, {26, 52});
    const std::vector<int> candidates = derive_candidates(clips);
    require(
        candidates == std::vector<int>({0, 26}),
        "candidate derivation did not honor both data ranges");

    CandidateConfig oversized;
    oversized.horizon_frames = INT_MAX;
    require(
        derive_candidates(clips, oversized).empty(),
        "oversized horizon must return no candidates");
    require(
        !is_candidate(clips, 0, oversized),
        "oversized horizon overflowed candidate window");

    database moving;
    initialize_candidate_database(moving, 26);
    moving.bone_velocities.set(vec3(0.100001F, 0.0F, 0.0F));
    require(
        derive_candidates(moving).empty(),
        "valid database with no idle window must return empty");
}

void test_config_and_classifier_shape_validation() {
    using namespace stationary_motion_matching;

    const float nan = float_from_bits(0x7fc00001U);
    const float infinity = float_from_bits(0x7f800000U);

    database valid;
    initialize_candidate_database(valid, 26);

    for (int horizon : {0, -1, INT_MIN}) {
        CandidateConfig config;
        config.horizon_frames = horizon;
        require_invalid_argument(
            [&] { static_cast<void>(derive_candidates(valid, config)); },
            "nonpositive horizon was accepted");
    }

    for (float invalid : {-0.001F, nan, infinity, -infinity}) {
        CandidateConfig speed;
        speed.maximum_planar_speed_mps = invalid;
        require_invalid_argument(
            [&] { static_cast<void>(derive_candidates(valid, speed)); },
            "invalid maximum speed was accepted");

        CandidateConfig displacement;
        displacement.maximum_planar_displacement_m = invalid;
        require_invalid_argument(
            [&] {
                static_cast<void>(derive_candidates(valid, displacement));
            },
            "invalid maximum displacement was accepted");
    }

    require_invalid_argument(
        [&] { static_cast<void>(is_candidate(valid, -1)); },
        "negative is_candidate frame was accepted");
    require_invalid_argument(
        [&] { static_cast<void>(is_candidate(valid, valid.nframes())); },
        "past-end is_candidate frame was accepted");

    {
        database db;
        require_invalid_argument(
            [&] { static_cast<void>(derive_candidates(db)); },
            "empty database shape was accepted");
    }
    {
        database db;
        initialize_candidate_database(db, 26);
        db.bone_velocities.resize(25, 1);
        require_invalid_argument(
            [&] { static_cast<void>(derive_candidates(db)); },
            "mismatched velocity rows were accepted");
    }
    {
        database db;
        initialize_candidate_database(db, 26);
        db.bone_velocities.resize(26, 2);
        require_invalid_argument(
            [&] { static_cast<void>(derive_candidates(db)); },
            "mismatched velocity columns were accepted");
    }
    {
        database db;
        initialize_candidate_database(db, 26);
        db.contact_states.resize(25, 2);
        require_invalid_argument(
            [&] { static_cast<void>(derive_candidates(db)); },
            "mismatched contact rows were accepted");
    }
    {
        database db;
        initialize_candidate_database(db, 26);
        db.contact_states.resize(26, 1);
        require_invalid_argument(
            [&] { static_cast<void>(derive_candidates(db)); },
            "database with fewer than two contacts was accepted");
    }
    {
        database db;
        initialize_candidate_database(db, 26);
        db.range_starts.resize(0);
        db.range_stops.resize(0);
        require_invalid_argument(
            [&] { static_cast<void>(derive_candidates(db)); },
            "empty clip ranges were accepted");
    }
    {
        database db;
        initialize_candidate_database(db, 26);
        db.range_stops.resize(2);
        require_invalid_argument(
            [&] { static_cast<void>(derive_candidates(db)); },
            "mismatched clip range arrays were accepted");
    }

    const std::vector<std::pair<std::vector<int>, std::vector<int>>>
        invalid_ranges{
            {{-1}, {26}},
            {{0}, {27}},
            {{5}, {5}},
            {{10, 0}, {20, 10}},
            {{0, 10}, {20, 26}},
        };
    for (const auto& ranges : invalid_ranges) {
        database db;
        initialize_candidate_database(db, 26);
        set_ranges(db, ranges.first, ranges.second);
        require_invalid_argument(
            [&] { static_cast<void>(derive_candidates(db)); },
            "invalid clip range topology was accepted");
    }

    database gapped;
    initialize_candidate_database(gapped, 26);
    set_ranges(gapped, {0, 15}, {10, 26});
    require(
        derive_candidates(gapped).empty(),
        "valid nonoverlapping gapped ranges were rejected");
}

void test_classifier_rejects_nonfinite_root_data_under_fast_math() {
    using namespace stationary_motion_matching;

    const std::vector<float> invalid_values{
        float_from_bits(0x7fc00001U),
        float_from_bits(0x7f800000U),
    };
    for (float invalid : invalid_values) {
        for (int sample : {0, 25}) {
            for (int component = 0; component < 3; ++component) {
                database positions;
                initialize_candidate_database(positions, 26);
                set_component(
                    positions.bone_positions(sample, 0), component, invalid);
                require_invalid_argument(
                    [&] {
                        static_cast<void>(derive_candidates(positions));
                    },
                    "nonfinite root position was accepted");

                database velocities;
                initialize_candidate_database(velocities, 26);
                set_component(
                    velocities.bone_velocities(sample, 0),
                    component,
                    invalid);
                require_invalid_argument(
                    [&] {
                        static_cast<void>(derive_candidates(velocities));
                    },
                    "nonfinite root velocity was accepted");
            }
        }
    }

    database mixed_invalid;
    initialize_candidate_database(mixed_invalid, 26);
    mixed_invalid.bone_velocities(0, 0).x = 0.100001F;
    mixed_invalid.bone_positions(25, 0).z = invalid_values.front();
    require_invalid_argument(
        [&] { static_cast<void>(derive_candidates(mixed_invalid)); },
        "early ineligibility skipped later nonfinite root data");
}

void test_search_validation_and_lower_frame_tie() {
    using namespace stationary_motion_matching;

    database db;
    initialize_search_database(db);
    db.features(0, 0) = 1.0F;
    db.features(1, 0) = -1.0F;
    array1d<float> query(2);
    query.zero();

    const SearchResult tie = search(db, query, {0, 1});
    require(tie.frame == 0, "equal search cost did not choose lower frame");
    require(tie.cost == 1.0F, "equal-cost search returned wrong cost");

    require_invalid_argument(
        [&] { static_cast<void>(search(db, query, {})); },
        "empty candidate list was accepted");
    require_invalid_argument(
        [&] { static_cast<void>(search(db, query, {1, 0})); },
        "descending candidate list was accepted");
    require_invalid_argument(
        [&] { static_cast<void>(search(db, query, {0, 0})); },
        "duplicate candidate was accepted");
    for (int invalid : {-1, db.nframes(), INT_MIN, INT_MAX}) {
        require_invalid_argument(
            [&] { static_cast<void>(search(db, query, {invalid})); },
            "invalid candidate index was accepted");
    }

    {
        database malformed;
        initialize_search_database(malformed);
        malformed.features.resize(2, 2);
        require_invalid_argument(
            [&] { static_cast<void>(search(malformed, query, {0})); },
            "feature/frame row mismatch was accepted");
    }
    {
        database malformed;
        initialize_search_database(malformed);
        malformed.features.resize(0, 0);
        malformed.features.rows = malformed.nframes();
        require_invalid_argument(
            [&] { static_cast<void>(search(malformed, query, {0})); },
            "zero feature width was accepted");
    }
    {
        array1d<float> short_query(1);
        short_query.zero();
        require_invalid_argument(
            [&] { static_cast<void>(search(db, short_query, {0})); },
            "wrong query dimension was accepted");
    }
    {
        database malformed;
        initialize_search_database(malformed);
        malformed.features_offset.resize(1);
        require_invalid_argument(
            [&] { static_cast<void>(search(malformed, query, {0})); },
            "wrong offset dimension was accepted");
    }
    {
        database malformed;
        initialize_search_database(malformed);
        malformed.features_scale.resize(1);
        require_invalid_argument(
            [&] { static_cast<void>(search(malformed, query, {0})); },
            "wrong scale dimension was accepted");
    }

    const std::vector<float> nonfinite_values{
        float_from_bits(0x7fc00001U),
        float_from_bits(0x7f800000U),
    };
    for (float invalid : nonfinite_values) {
        {
            database malformed;
            initialize_search_database(malformed);
            malformed.features_offset(0) = invalid;
            require_invalid_argument(
                [&] { static_cast<void>(search(malformed, query, {0})); },
                "nonfinite feature offset was accepted");
        }
        {
            database malformed;
            initialize_search_database(malformed);
            malformed.features_scale(0) = invalid;
            require_invalid_argument(
                [&] { static_cast<void>(search(malformed, query, {0})); },
                "nonfinite feature scale was accepted");
        }
        {
            array1d<float> invalid_query(query);
            invalid_query(0) = invalid;
            require_invalid_argument(
                [&] { static_cast<void>(search(db, invalid_query, {0})); },
                "nonfinite query was accepted");
        }
        {
            database malformed;
            initialize_search_database(malformed);
            malformed.features(0, 0) = invalid;
            require_invalid_argument(
                [&] { static_cast<void>(search(malformed, query, {0})); },
                "nonfinite candidate feature was accepted");
        }
    }

    for (float invalid_scale : {0.0F, -1.0F}) {
        database malformed;
        initialize_search_database(malformed);
        malformed.features_scale(0) = invalid_scale;
        require_invalid_argument(
            [&] { static_cast<void>(search(malformed, query, {0})); },
            "nonpositive feature scale was accepted");
    }
}

array1d<float> stationary_query_for_row(const database& db, int row) {
    array1d<float> query(db.nfeatures());
    for (int dimension = 0; dimension < 15; ++dimension) {
        query(dimension) =
            db.features(row, dimension) * db.features_scale(dimension) +
            db.features_offset(dimension);
    }
    for (int dimension = 15; dimension < 21; ++dimension) {
        query(dimension) = 0.0F;
    }
    const float straight_directions[6] = {
        0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F};
    for (int dimension = 21; dimension < 27; ++dimension) {
        query(dimension) = straight_directions[dimension - 21];
    }
    return query;
}

std::pair<int, float> minimum_candidate_cost(
    const database& db,
    const slice1d<float> query,
    const std::vector<int>& candidates) {
    array1d<float> normalized(db.nfeatures());
    for (int dimension = 0; dimension < db.nfeatures(); ++dimension) {
        normalized(dimension) =
            (query(dimension) - db.features_offset(dimension)) /
            db.features_scale(dimension);
    }

    int best_frame = -1;
    float best_cost = FLT_MAX;
    for (int frame : candidates) {
        float cost = 0.0F;
        for (int dimension = 0; dimension < db.nfeatures(); ++dimension) {
            const float difference =
                normalized(dimension) - db.features(frame, dimension);
            cost += difference * difference;
        }
        if (cost < best_cost) {
            best_frame = frame;
            best_cost = cost;
        }
    }
    return {best_frame, best_cost};
}

void test_authoritative_candidates_and_constrained_search() {
    using namespace stationary_motion_matching;

    database db;
    database_load(db, "resources/database.bin");
    database_build_matching_features(
        db,
        0.75F,
        1.0F,
        1.0F,
        1.0F,
        1.5F);
    require(db.nfeatures() == 27, "ordinary matching feature width changed");

    const std::vector<int> candidates = derive_candidates(db);
    require(candidates.size() == 186U, "authoritative stationary count changed");
    require(candidates.front() == 0, "first stationary candidate changed");
    require(candidates[92] == 92, "first stationary span changed");
    require(candidates[93] == 118, "second stationary span changed");
    require(candidates.back() == 210, "last stationary candidate changed");

    std::vector<std::pair<int, int>> spans;
    int span_start = candidates.front();
    int previous = candidates.front();
    for (size_t index = 1; index < candidates.size(); ++index) {
        if (candidates[index] != previous + 1) {
            spans.emplace_back(span_start, previous);
            span_start = candidates[index];
        }
        previous = candidates[index];
    }
    spans.emplace_back(span_start, previous);
    require(
        spans == std::vector<std::pair<int, int>>({{0, 92}, {118, 210}}),
        "authoritative stationary spans changed");

    int representative_rows = 0;
    for (int row = 93; row < 118 && representative_rows < 3; ++row) {
        const array1d<float> query = stationary_query_for_row(db, row);
        int ordinary_frame = -1;
        float ordinary_cost = FLT_MAX;
        database_search(ordinary_frame, ordinary_cost, db, query);
        if (ordinary_frame < 0 ||
            std::binary_search(
                candidates.begin(), candidates.end(), ordinary_frame)) {
            continue;
        }

        const SearchResult constrained = search(db, query, candidates);
        require(
            std::binary_search(
                candidates.begin(), candidates.end(), constrained.frame),
            "constrained search escaped stationary candidate set");
        const std::pair<int, float> expected =
            minimum_candidate_cost(db, query, candidates);
        require(
            constrained.frame == expected.first,
            "constrained search did not return minimum eligible frame");
        require(
            near(constrained.cost, expected.second),
            "constrained search did not return minimum eligible cost: actual=" +
                std::to_string(constrained.cost) +
                " expected=" + std::to_string(expected.second));
        ++representative_rows;
    }
    require(
        representative_rows >= 2,
        "ordinary search did not return moving matches for representative rows");
}

}  // namespace

int main() {
    try {
        test_inclusive_window_clip_and_threshold_contract();
        test_positive_nondefault_horizon_uses_configured_window();
        test_interior_window_samples_are_checked();
        test_z_axis_displacement_threshold_contract();
        test_derivation_handles_ranges_oversized_horizon_and_no_idle();
        test_config_and_classifier_shape_validation();
        test_classifier_rejects_nonfinite_root_data_under_fast_math();
        test_search_validation_and_lower_frame_tie();
        test_authoritative_candidates_and_constrained_search();
        std::cout
            << "stationary_motion_matching candidates=186 spans=0-92,118-210\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "stationary_motion_matching FAILED: " << error.what()
                  << '\n';
        return 1;
    }
}
