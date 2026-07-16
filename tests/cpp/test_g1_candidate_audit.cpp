#include "g1_candidate_audit.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "G1 candidate audit test failed: %s\n", message);
        std::exit(1);
    }
}

static uint32_t bits(float value)
{
    uint32_t output = 0;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

static float from_bits(uint32_t value)
{
    float output = 0.0f;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

static float rounded_subtract(float left, float right)
{
    volatile float result = left - right;
    return result;
}

static float rounded_multiply(float left, float right)
{
    volatile float result = left * right;
    return result;
}

static float rounded_add(float left, float right)
{
    volatile float result = left + right;
    return result;
}

static float brute_cost(
    const std::vector<float>& query,
    const std::vector<float>& features,
    int frame)
{
    float cost = 0.0f;
    for (int feature = 0; feature < G1CandidateAuditFeatureCount; ++feature) {
        const float difference = rounded_subtract(
            query[static_cast<std::size_t>(feature)],
            features[static_cast<std::size_t>(frame) *
                         G1CandidateAuditFeatureCount +
                     static_cast<std::size_t>(feature)]);
        const float square = rounded_multiply(difference, difference);
        cost = rounded_add(cost, square);
    }
    return cost;
}

struct BruteEntry
{
    int frame;
    int range;
    float cost;
};

static std::vector<BruteEntry> brute_entries(
    const std::vector<float>& query,
    const std::vector<float>& features,
    const std::vector<int>& range_starts,
    const std::vector<int>& range_stops,
    int incumbent)
{
    std::vector<BruteEntry> output;
    for (std::size_t range = 0; range < range_starts.size(); ++range) {
        const int search_stop = range_stops[range] -
            G1CandidateAuditIgnoreRangeEnd;
        for (int frame = range_starts[range]; frame < search_stop; ++frame) {
            const int64_t separation = incumbent < 0
                ? std::numeric_limits<int64_t>::max()
                : std::llabs(
                      static_cast<int64_t>(frame) -
                      static_cast<int64_t>(incumbent));
            if (incumbent >= 0 &&
                separation < G1CandidateAuditIgnoreSurrounding) {
                continue;
            }
            output.push_back(BruteEntry{
                frame,
                static_cast<int>(range),
                brute_cost(query, features, frame),
            });
        }
    }
    std::stable_sort(
        output.begin(), output.end(),
        [](const BruteEntry& left, const BruteEntry& right) {
            if (left.cost < right.cost) return true;
            if (right.cost < left.cost) return false;
            return left.frame < right.frame;
        });
    return output;
}

static slice1d<float> float_slice(std::vector<float>& values)
{
    return slice1d<float>(
        static_cast<int>(values.size()), values.empty() ? NULL : values.data());
}

static slice1d<int> int_slice(std::vector<int>& values)
{
    return slice1d<int>(
        static_cast<int>(values.size()), values.empty() ? NULL : values.data());
}

static slice2d<float> feature_slice(
    std::vector<float>& values,
    int rows,
    int columns = G1CandidateAuditFeatureCount)
{
    return slice2d<float>(
        rows, columns, values.empty() ? NULL : values.data());
}

static bool same_audit_bytes(
    const G1CandidateAudit& left,
    const G1CandidateAudit& right)
{
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

static void poison(G1CandidateAudit& output, unsigned char seed = 0xa5U)
{
    unsigned char* const bytes_out =
        reinterpret_cast<unsigned char*>(&output);
    for (std::size_t index = 0; index < sizeof(output); ++index) {
        bytes_out[index] = static_cast<unsigned char>(seed + index % 19U);
    }
}

struct Fixture
{
    static constexpr int Frames = 140;

    std::vector<float> query;
    std::vector<float> features;
    std::vector<int> starts;
    std::vector<int> stops;

    Fixture()
        : query(G1CandidateAuditFeatureCount, 0.0f),
          features(
              static_cast<std::size_t>(Frames) *
                  G1CandidateAuditFeatureCount,
              0.0f),
          starts{0, 70},
          stops{70, 140}
    {
        for (int frame = 0; frame < Frames; ++frame) {
            at(frame, 0) = 3.0f;
        }

        at(0, 0) = 0.0f;
        at(1, 0) = 0.5f;
        at(2, 0) = 1.0f;
        at(3, 0) = 2.0f;
        at(4, 0) = 0.5f;
        at(5, 0) = 0.0f;
        for (int feature = 0;
             feature < G1CandidateAuditFeatureCount;
             ++feature) {
            at(6, feature) = 0.1f;
        }
        at(70, 0) = 0.75f;

        // These would win if either exclusion boundary were implemented
        // incorrectly. Frame 11 is inside the incumbent neighborhood; frames
        // 69 and 120 are in the ignored range tails.
        at(11, 0) = 0.0f;
        at(69, 0) = 0.0f;
        at(120, 0) = 0.0f;
    }

    float& at(int frame, int feature)
    {
        return features[static_cast<std::size_t>(frame) *
                            G1CandidateAuditFeatureCount +
                        static_cast<std::size_t>(feature)];
    }

    bool run(G1CandidateAudit& output, int incumbent = 30)
    {
        return g1_candidate_audit_build(
            output,
            float_slice(query),
            feature_slice(features, Frames),
            int_slice(starts),
            int_slice(stops),
            incumbent);
    }
};

static void test_brute_force_costs_order_counts_and_ranges()
{
    check(G1CandidateAuditDelta0_25Bits == UINT32_C(0x3e800000),
          "0.25 threshold word is exact");
    check(G1CandidateAuditDelta1Bits == UINT32_C(0x3f800000),
          "1.0 threshold word is exact");
    check(G1CandidateAuditDelta4Bits == UINT32_C(0x40800000),
          "4.0 threshold word is exact");

    Fixture fixture;
    const std::vector<BruteEntry> expected = brute_entries(
        fixture.query,
        fixture.features,
        fixture.starts,
        fixture.stops,
        30);
    check(expected.size() == 61,
          "independent fixture has the intended eligible population");

    G1CandidateAudit output;
    poison(output);
    check(fixture.run(output), "valid candidate audit succeeds");
    check(g1_candidate_audit_is_valid(output),
          "published candidate audit validates");
    check(output.eligible_count == expected.size(),
          "audit visits every eligible frame exactly once");
    check(output.top_count == G1CandidateAuditTopCapacity,
          "audit publishes the fixed top-16 capacity");

    uint32_t expected_0_25 = 0;
    uint32_t expected_1 = 0;
    uint32_t expected_4 = 0;
    const float threshold_0_25 = rounded_add(
        expected[0].cost, from_bits(G1CandidateAuditDelta0_25Bits));
    const float threshold_1 = rounded_add(
        expected[0].cost, from_bits(G1CandidateAuditDelta1Bits));
    const float threshold_4 = rounded_add(
        expected[0].cost, from_bits(G1CandidateAuditDelta4Bits));
    for (const BruteEntry& entry : expected) {
        expected_0_25 += entry.cost <= threshold_0_25 ? 1U : 0U;
        expected_1 += entry.cost <= threshold_1 ? 1U : 0U;
        expected_4 += entry.cost <= threshold_4 ? 1U : 0U;
    }
    check(expected_0_25 == 4 && expected_1 == 7 && expected_4 == 8,
          "fixture locks exact best-plus threshold inclusivity");
    check(output.within_best_plus_0_25 == expected_0_25 &&
              output.within_best_plus_1 == expected_1 &&
              output.within_best_plus_4 == expected_4,
          "audit counts exact binary32 best-plus deltas");

    for (uint32_t index = 0; index < output.top_count; ++index) {
        const G1CandidateAuditEntry& actual = output.top[index];
        const BruteEntry& wanted = expected[index];
        check(actual.frame == wanted.frame &&
                  actual.range == wanted.range &&
                  actual.cost_bits == bits(wanted.cost),
              "top entries match independent all-frame scalar costs");
    }
    check(output.top[0].frame == 0 && output.top[1].frame == 5 &&
              output.top[2].frame == 1 && output.top[3].frame == 4,
          "equal costs are stable by ascending frame index");
    check(output.top[4].frame == 6 &&
              output.top[4].cost_bits == bits(brute_cost(
                  fixture.query, fixture.features, 6)) &&
              output.top[4].cost_bits == UINT32_C(0x3e9eb852),
          "31 binary32 squares accumulate to the locked feature-order word");
    for (int index = static_cast<int>(output.top_count);
         index < G1CandidateAuditTopCapacity;
         ++index) {
        check(output.top[index].frame == -1 &&
                  output.top[index].range == -1 &&
                  output.top[index].cost_bits == 0,
              "unused top slots retain canonical defaults");
    }

    G1CandidateAudit repeated;
    poison(repeated, 0x3cU);
    check(fixture.run(repeated), "repeated candidate audit succeeds");
    check(same_audit_bytes(output, repeated),
          "candidate audit overwrites padding and is byte deterministic");
}

static void require_failure_preserves(
    bool result,
    const G1CandidateAudit& output,
    const G1CandidateAudit& before,
    const char* message)
{
    check(!result, message);
    check(same_audit_bytes(output, before), message);
}

static void test_validation_and_transactionality()
{
    Fixture fixture;
    G1CandidateAudit output;
    poison(output);
    const G1CandidateAudit before = output;

    std::vector<float> short_query(
        G1CandidateAuditFeatureCount - 1, 0.0f);
    require_failure_preserves(
        g1_candidate_audit_build(
            output, float_slice(short_query),
            feature_slice(fixture.features, Fixture::Frames),
            int_slice(fixture.starts), int_slice(fixture.stops), 30),
        output, before, "query must contain exactly 31 values");

    slice1d<float> null_query(G1CandidateAuditFeatureCount, NULL);
    require_failure_preserves(
        g1_candidate_audit_build(
            output, null_query,
            feature_slice(fixture.features, Fixture::Frames),
            int_slice(fixture.starts), int_slice(fixture.stops), 30),
        output, before, "null query storage is rejected");

    fixture.query[7] = std::numeric_limits<float>::infinity();
    require_failure_preserves(
        fixture.run(output), output, before,
        "nonfinite normalized query is rejected transactionally");
    fixture.query[7] = 0.0f;

    require_failure_preserves(
        g1_candidate_audit_build(
            output, float_slice(fixture.query),
            feature_slice(
                fixture.features, Fixture::Frames,
                G1CandidateAuditFeatureCount - 1),
            int_slice(fixture.starts), int_slice(fixture.stops), 30),
        output, before, "feature matrix must have exactly 31 columns");

    slice2d<float> null_features(
        Fixture::Frames, G1CandidateAuditFeatureCount, NULL);
    require_failure_preserves(
        g1_candidate_audit_build(
            output, float_slice(fixture.query), null_features,
            int_slice(fixture.starts), int_slice(fixture.stops), 30),
        output, before, "null feature storage is rejected");

    // A poisoned value in an ignored tail is still malformed immutable input.
    fixture.at(139, 30) = std::numeric_limits<float>::quiet_NaN();
    require_failure_preserves(
        fixture.run(output), output, before,
        "all feature values are preflighted, including ignored tails");
    fixture.at(139, 30) = 0.0f;

    std::vector<int> short_stops{70};
    require_failure_preserves(
        g1_candidate_audit_build(
            output, float_slice(fixture.query),
            feature_slice(fixture.features, Fixture::Frames),
            int_slice(fixture.starts), int_slice(short_stops), 30),
        output, before, "range arrays must have matching shapes");

    std::vector<int> bad_starts = fixture.starts;
    bad_starts[1] = 71;
    require_failure_preserves(
        g1_candidate_audit_build(
            output, float_slice(fixture.query),
            feature_slice(fixture.features, Fixture::Frames),
            int_slice(bad_starts), int_slice(fixture.stops), 30),
        output, before, "ranges must form one ascending contiguous partition");

    std::vector<int> bad_stops = fixture.stops;
    bad_stops[1] = Fixture::Frames + 1;
    require_failure_preserves(
        g1_candidate_audit_build(
            output, float_slice(fixture.query),
            feature_slice(fixture.features, Fixture::Frames),
            int_slice(fixture.starts), int_slice(bad_stops), 30),
        output, before, "range stops may not exceed feature rows");

    require_failure_preserves(
        g1_candidate_audit_build(
            output, float_slice(fixture.query),
            feature_slice(fixture.features, Fixture::Frames),
            int_slice(fixture.starts), int_slice(fixture.stops),
            Fixture::Frames),
        output, before, "incumbent index must be -1 or an in-range frame");

    require_failure_preserves(
        g1_candidate_audit_build(
            output, float_slice(fixture.query),
            feature_slice(fixture.features, Fixture::Frames),
            int_slice(fixture.starts), int_slice(fixture.stops), 30,
            G1CandidateAuditIgnoreRangeEnd - 1,
            G1CandidateAuditIgnoreSurrounding),
        output, before, "audit rejects a noncanonical ignored range tail");
    require_failure_preserves(
        g1_candidate_audit_build(
            output, float_slice(fixture.query),
            feature_slice(fixture.features, Fixture::Frames),
            int_slice(fixture.starts), int_slice(fixture.stops), 30,
            G1CandidateAuditIgnoreRangeEnd,
            G1CandidateAuditIgnoreSurrounding + 1),
        output, before, "audit rejects a noncanonical incumbent exclusion");

    fixture.query[0] = -std::numeric_limits<float>::max();
    fixture.at(0, 0) = std::numeric_limits<float>::max();
    require_failure_preserves(
        fixture.run(output), output, before,
        "finite binary32 subtraction overflow is checked");
}

static void test_empty_result_and_structural_validation()
{
    std::vector<float> query(G1CandidateAuditFeatureCount, 0.0f);
    std::vector<float> features(
        10 * G1CandidateAuditFeatureCount, 0.0f);
    std::vector<int> starts{0};
    std::vector<int> stops{10};
    G1CandidateAudit empty;
    poison(empty);
    check(g1_candidate_audit_build(
              empty, float_slice(query), feature_slice(features, 10),
              int_slice(starts), int_slice(stops), -1),
          "a valid database with no eligible frames audits successfully");
    check(empty.eligible_count == 0 && empty.top_count == 0 &&
              empty.within_best_plus_0_25 == 0 &&
              empty.within_best_plus_1 == 0 &&
              empty.within_best_plus_4 == 0 &&
              g1_candidate_audit_is_valid(empty),
          "empty audit has one canonical valid representation");

    Fixture fixture;
    G1CandidateAudit valid;
    check(fixture.run(valid), "structural mutation fixture builds");

    G1CandidateAudit mutated = valid;
    mutated.within_best_plus_1 = mutated.within_best_plus_0_25 - 1U;
    check(!g1_candidate_audit_is_valid(mutated),
          "threshold counts must be monotonic");

    mutated = valid;
    mutated.top[1].frame = mutated.top[0].frame - 1;
    mutated.top[1].cost_bits = mutated.top[0].cost_bits;
    check(!g1_candidate_audit_is_valid(mutated),
          "equal-cost entries must be frame-stable");

    mutated = valid;
    mutated.top[0].cost_bits = UINT32_C(0x7f800000);
    check(!g1_candidate_audit_is_valid(mutated),
          "nonfinite cost words are structurally invalid");

    G1CandidateAudit mutated_empty = empty;
    mutated_empty.top[0].frame = 12;
    check(!g1_candidate_audit_is_valid(mutated_empty),
          "unused top slots must retain exact defaults");
}

int main()
{
    test_brute_force_costs_order_counts_and_ranges();
    test_validation_and_transactionality();
    test_empty_result_and_structural_validation();
    return 0;
}
