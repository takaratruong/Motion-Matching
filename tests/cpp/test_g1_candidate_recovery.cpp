#include "g1_candidate_recovery.h"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

static void check(bool value, const char* expression, int line)
{
    if (!value) {
        std::fprintf(
            stderr,
            "G1 recovery provider test failed at line %d: %s\n",
            line,
            expression);
        std::exit(1);
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

static_assert(G1RecoveryFeatureCount == 31U, "recovery feature count changed");
static_assert(
    G1CandidateAttemptCapacity == 8U,
    "attempt storage capacity changed");
static_assert(
    G1RecoveryTransitionCapacity == 6U,
    "transition capacity changed");
static_assert(G1RecoveryTailCapacity == 7U, "tail capacity changed");
static_assert(
    sizeof(((G1RecoveryCandidateSet*)nullptr)->records) /
            sizeof(G1CandidateRecord) ==
        G1CandidateAttemptCapacity,
    "candidate set must retain eight physical records");

static uint32_t float_bits(float value)
{
    uint32_t output = 0U;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

static float float_from_bits(uint32_t value)
{
    float output = 0.0f;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

static float strict_add(float left, float right)
{
    volatile float output = left + right;
    return output;
}

static float strict_multiply(float left, float right)
{
    volatile float output = left * right;
    return output;
}

static float strict_score_words(
    const uint32_t (&difference_bits)[G1RecoveryFeatureCount],
    float seed)
{
    float score = seed;
    for (uint32_t feature = 0U;
         feature < G1RecoveryFeatureCount;
         ++feature) {
        const float difference = float_from_bits(difference_bits[feature]);
        const float square = strict_multiply(difference, difference);
        score = strict_add(score, square);
    }
    return score;
}

static void poison_bytes(void* object, std::size_t size, uint8_t seed)
{
    uint8_t* const bytes = static_cast<uint8_t*>(object);
    for (std::size_t index = 0U; index < size; ++index) {
        bytes[index] = static_cast<uint8_t>(seed + index % 29U);
    }
}

template<typename T>
static bool object_bytes_equal(const T& left, const T& right)
{
    return std::memcmp(&left, &right, sizeof(T)) == 0;
}

struct Fixture
{
    database db;

    explicit Fixture(int frames)
    {
        initialize(frames, std::vector<std::pair<int, int> >{
                               std::make_pair(0, frames)});
    }

    Fixture(
        int frames,
        const std::vector<std::pair<int, int> >& ranges)
    {
        initialize(frames, ranges);
    }

    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;

    void initialize(
        int frames,
        const std::vector<std::pair<int, int> >& ranges)
    {
        CHECK(frames > 0);
        CHECK(!ranges.empty());

        db.bone_positions.resize(frames, 1);
        db.bone_velocities.resize(frames, 1);
        db.bone_rotations.resize(frames, 1);
        db.bone_angular_velocities.resize(frames, 1);
        db.bone_parents.resize(1);
        db.range_starts.resize(static_cast<int>(ranges.size()));
        db.range_stops.resize(static_cast<int>(ranges.size()));
        db.features.resize(frames, static_cast<int>(G1RecoveryFeatureCount));
        db.features_offset.resize(static_cast<int>(G1RecoveryFeatureCount));
        db.features_scale.resize(static_cast<int>(G1RecoveryFeatureCount));
        db.terrain_features.resize(frames, 4);
        db.contact_states.resize(frames, 2);

        db.bone_positions.set(vec3());
        db.bone_velocities.set(vec3());
        db.bone_rotations.set(quat());
        db.bone_angular_velocities.set(vec3());
        db.bone_parents(0) = -1;
        db.features.set(20.0f);
        db.features_offset.set(0.0f);
        db.features_scale.set(1.0f);
        db.terrain_features.set(0.0f);
        db.contact_states.set(false);
        for (std::size_t range = 0U; range < ranges.size(); ++range) {
            db.range_starts(static_cast<int>(range)) = ranges[range].first;
            db.range_stops(static_cast<int>(range)) = ranges[range].second;
        }
        rebuild_bounds();
    }

    void rebuild_bounds()
    {
        database_build_bounds(db);
    }

    void zero_query_model()
    {
        db.features_offset.set(0.0f);
        db.features_scale.set(1.0f);
    }

    void set_row(float value, int frame)
    {
        for (uint32_t feature = 0U;
             feature < G1RecoveryFeatureCount;
             ++feature) {
            db.features(frame, static_cast<int>(feature)) = value;
        }
    }

    void set_row_zero(int frame)
    {
        set_row(0.0f, frame);
    }

    G1RecoveryRequest request(
        int incumbent,
        int legacy,
        float transition_cost = 0.0f,
        float public_incumbent_cost = FLT_MAX) const
    {
        G1RecoveryRequest request_value;
        request_value.db = &db;
        request_value.incumbent_frame = incumbent;
        request_value.legacy_selected_frame = legacy;
        request_value.transition_cost = transition_cost;
        request_value.public_incumbent_cost = public_incumbent_cost;
        request_value.ignore_range_end = 0;
        request_value.ignore_surrounding = 0;
        for (uint32_t feature = 0U;
             feature < G1RecoveryFeatureCount;
             ++feature) {
            request_value.raw_query[feature] = 0.0f;
        }
        return request_value;
    }
};

static bool records_equal(
    const G1CandidateRecord& left,
    const G1CandidateRecord& right)
{
    return left.kind == right.kind &&
           left.selected_frame == right.selected_frame &&
           left.executed_frame == right.executed_frame &&
           left.source_range == right.source_range &&
           float_bits(left.selected_cost) == float_bits(right.selected_cost) &&
           left.recovery_rank == right.recovery_rank &&
           left.transitioned == right.transitioned;
}

static void check_sets_equal(
    const G1RecoveryCandidateSet& accelerated,
    const G1RecoveryCandidateSet& exhaustive)
{
    CHECK(accelerated.count == exhaustive.count);
    for (uint32_t index = 0U; index < accelerated.count; ++index) {
        CHECK(records_equal(
            accelerated.records[index], exhaustive.records[index]));
    }
}

static void check_set_contract(const G1RecoveryCandidateSet& output)
{
    CHECK(output.count <= G1RecoveryTailCapacity);
    CHECK(output.count <= G1CandidateAttemptCapacity);
    uint32_t transitions = 0U;
    bool saw_incumbent = false;
    for (uint32_t index = 0U; index < output.count; ++index) {
        const G1CandidateRecord& record = output.records[index];
        if (record.kind == G1CandidateRecoveryTransition) {
            CHECK(!saw_incumbent);
            CHECK(record.transitioned);
            CHECK(record.recovery_rank == transitions);
            ++transitions;
        } else {
            CHECK(record.kind == G1CandidateIncumbent);
            CHECK(index + 1U == output.count);
            CHECK(!record.transitioned);
            CHECK(record.recovery_rank == UINT32_MAX);
            saw_incumbent = true;
        }
    }
    CHECK(transitions <= G1RecoveryTransitionCapacity);
}

static void run_accelerated_and_exhaustive(
    Fixture& fixture,
    const G1RecoveryRequest& request,
    G1RecoveryCandidateSet& accelerated,
    G1RecoveryProviderAudit& accelerated_audit,
    G1RecoveryCandidateSet& exhaustive,
    G1RecoveryProviderAudit& exhaustive_audit)
{
    fixture.rebuild_bounds();
    char accelerated_error[256] = {};
    char exhaustive_error[256] = {};
    CHECK(g1_recovery_candidates_build_audited(
              accelerated,
              request,
              accelerated_audit,
              accelerated_error,
              static_cast<int>(sizeof(accelerated_error))) ==
          G1RecoveryProviderOk);
    CHECK(g1_recovery_candidates_exhaustive_for_test(
              exhaustive,
              request,
              exhaustive_audit,
              exhaustive_error,
              static_cast<int>(sizeof(exhaustive_error))) ==
          G1RecoveryProviderOk);
    CHECK(accelerated.work.accelerated_traversals == 1U);
    CHECK(exhaustive.work.accelerated_traversals == 0U);
    CHECK(accelerated.work.full_scores_materialized >= accelerated.count -
              (accelerated.count > 0U &&
                       accelerated.records[accelerated.count - 1U].kind ==
                           G1CandidateIncumbent
                   ? 1U
                   : 0U));
    CHECK(accelerated_audit.recovery_incumbent_score_bits ==
          exhaustive_audit.recovery_incumbent_score_bits);
    for (uint32_t feature = 0U;
         feature < G1RecoveryFeatureCount;
         ++feature) {
        CHECK(accelerated_audit.normalized_query_bits[feature] ==
              exhaustive_audit.normalized_query_bits[feature]);
    }
    CHECK(exhaustive_audit.exhaustive_rows_tested ==
          exhaustive.work.rows_tested);
    check_sets_equal(accelerated, exhaustive);
    check_set_contract(accelerated);
}

static void test_strict_normalization_owns_all_31_words();
static void test_seeded_transition_and_zero_seeded_incumbent_are_distinct();
static void test_both_incumbent_admission_bounds_are_strict();
static void test_top_six_ties_eviction_dedup_and_incumbent_order();
static void test_exclusions_sequential_idle_and_end_of_animation();
static void test_accelerated_provider_matches_exhaustive_oracle();
static void test_malformed_alias_overflow_and_capacity_fail_closed();
static void print_provider_parity_transcript();

static void expect_global_error_transactional(
    const G1RecoveryRequest& request)
{
    G1RecoveryCandidateSet output;
    G1RecoveryProviderAudit audit;
    poison_bytes(&output, sizeof(output), UINT8_C(0x51));
    poison_bytes(&audit, sizeof(audit), UINT8_C(0xa3));
    const G1RecoveryCandidateSet output_before = output;
    const G1RecoveryProviderAudit audit_before = audit;
    char error[192] = {};
    CHECK(g1_recovery_candidates_build_audited(
              output,
              request,
              audit,
              error,
              static_cast<int>(sizeof(error))) ==
          G1RecoveryProviderGlobalError);
    CHECK(object_bytes_equal(output, output_before));
    CHECK(object_bytes_equal(audit, audit_before));
}

static void test_strict_normalization_owns_all_31_words()
{
    static const uint32_t query_bits[G1RecoveryFeatureCount] = {
        UINT32_C(0x3f800001), UINT32_C(0x3f812346),
        UINT32_C(0x3f82468b), UINT32_C(0x3f8369d0),
        UINT32_C(0x3f848d15), UINT32_C(0x3f85b05a),
        UINT32_C(0x3f86d39f), UINT32_C(0x3f87f6e4),
        UINT32_C(0x3f891a29), UINT32_C(0x3f8a3d6e),
        UINT32_C(0x3f8b60b3), UINT32_C(0x3f8c83f8),
        UINT32_C(0x3f8da73d), UINT32_C(0x3f8eca82),
        UINT32_C(0x3f8fedc7), UINT32_C(0x3f91110c),
        UINT32_C(0x3f923451), UINT32_C(0x3f935796),
        UINT32_C(0x3f947adb), UINT32_C(0x3f959e20),
        UINT32_C(0x3f96c165), UINT32_C(0x3f97e4aa),
        UINT32_C(0x3f9907ef), UINT32_C(0x3f9a2b34),
        UINT32_C(0x3f9b4e79), UINT32_C(0x3f9c71be),
        UINT32_C(0x3f9d9503), UINT32_C(0x3f9eb848),
        UINT32_C(0x3f9fdb8d), UINT32_C(0x3fa0fed2),
        UINT32_C(0x3fa22217),
    };
    static const uint32_t offset_bits[G1RecoveryFeatureCount] = {
        UINT32_C(0x3e000101), UINT32_C(0x3e010202),
        UINT32_C(0x3e020303), UINT32_C(0x3e030404),
        UINT32_C(0x3e040505), UINT32_C(0x3e050606),
        UINT32_C(0x3e060707), UINT32_C(0x3e070808),
        UINT32_C(0x3e080909), UINT32_C(0x3e090a0a),
        UINT32_C(0x3e0a0b0b), UINT32_C(0x3e0b0c0c),
        UINT32_C(0x3e0c0d0d), UINT32_C(0x3e0d0e0e),
        UINT32_C(0x3e0e0f0f), UINT32_C(0x3e0f1010),
        UINT32_C(0x3e101111), UINT32_C(0x3e111212),
        UINT32_C(0x3e121313), UINT32_C(0x3e131414),
        UINT32_C(0x3e141515), UINT32_C(0x3e151616),
        UINT32_C(0x3e161717), UINT32_C(0x3e171818),
        UINT32_C(0x3e181919), UINT32_C(0x3e191a1a),
        UINT32_C(0x3e1a1b1b), UINT32_C(0x3e1b1c1c),
        UINT32_C(0x3e1c1d1d), UINT32_C(0x3e1d1e1e),
        UINT32_C(0x3e1e1f1f),
    };
    static const uint32_t scale_bits[G1RecoveryFeatureCount] = {
        UINT32_C(0x3f200011), UINT32_C(0x3f208092),
        UINT32_C(0x3f210113), UINT32_C(0x3f218194),
        UINT32_C(0x3f220215), UINT32_C(0x3f228296),
        UINT32_C(0x3f230317), UINT32_C(0x3f238398),
        UINT32_C(0x3f240419), UINT32_C(0x3f24849a),
        UINT32_C(0x3f25051b), UINT32_C(0x3f25859c),
        UINT32_C(0x3f26061d), UINT32_C(0x3f26869e),
        UINT32_C(0x3f27071f), UINT32_C(0x3f2787a0),
        UINT32_C(0x3f280821), UINT32_C(0x7f7fffff),
        UINT32_C(0x3f290923), UINT32_C(0x3f2989a4),
        UINT32_C(0x3f2a0a25), UINT32_C(0x3f2a8aa6),
        UINT32_C(0x3f2b0b27), UINT32_C(0x3f2b8ba8),
        UINT32_C(0x3f2c0c29), UINT32_C(0x3f2c8caa),
        UINT32_C(0x3f2d0d2b), UINT32_C(0x3f2d8dac),
        UINT32_C(0x3f2e0e2d), UINT32_C(0x3f2e8eae),
        UINT32_C(0x3f2f0f2f),
    };
    static const uint32_t expected_bits[G1RecoveryFeatureCount] = {
        UINT32_C(0x3fb332ef), UINT32_C(0x3fb440cb),
        UINT32_C(0x3fb54cf8), UINT32_C(0x3fb6577a),
        UINT32_C(0x3fb76056), UINT32_C(0x3fb8678f),
        UINT32_C(0x3fb96d29), UINT32_C(0x3fba7128),
        UINT32_C(0x3fbb738f), UINT32_C(0x3fbc7463),
        UINT32_C(0x3fbd73a6), UINT32_C(0x3fbe715e),
        UINT32_C(0x3fbf6d8d), UINT32_C(0x3fc06836),
        UINT32_C(0x3fc1615d), UINT32_C(0x3fc25907),
        UINT32_C(0x3fc34f36), UINT32_C(0x00000000),
        UINT32_C(0x3fc53731), UINT32_C(0x3fc62903),
        UINT32_C(0x3fc71967), UINT32_C(0x3fc80862),
        UINT32_C(0x3fc8f5f5), UINT32_C(0x3fc9e225),
        UINT32_C(0x3fcaccf4), UINT32_C(0x3fcbb665),
        UINT32_C(0x3fcc9e7c), UINT32_C(0x3fcd8539),
        UINT32_C(0x3fce6aa3), UINT32_C(0x3fcf4ebb),
        UINT32_C(0x3fd03185),
    };

    Fixture fixture(3);
    G1RecoveryRequest request = fixture.request(1, 2);
    for (uint32_t feature = 0U;
         feature < G1RecoveryFeatureCount;
         ++feature) {
        request.raw_query[feature] = float_from_bits(query_bits[feature]);
        fixture.db.features_offset(static_cast<int>(feature)) =
            float_from_bits(offset_bits[feature]);
        fixture.db.features_scale(static_cast<int>(feature)) =
            float_from_bits(scale_bits[feature]);
    }
    fixture.rebuild_bounds();

    G1RecoveryCandidateSet output;
    G1RecoveryProviderAudit audit;
    char error[192] = {};
    CHECK(g1_recovery_candidates_build_audited(
              output,
              request,
              audit,
              error,
              static_cast<int>(sizeof(error))) ==
          G1RecoveryProviderOk);
    for (uint32_t feature = 0U;
         feature < G1RecoveryFeatureCount;
         ++feature) {
        CHECK(audit.normalized_query_bits[feature] ==
              expected_bits[feature]);
    }
    CHECK(audit.normalized_query_bits[17] == UINT32_C(0x00000000));

    const uint32_t nonfinite_words[] = {
        UINT32_C(0x7f800000),
        UINT32_C(0xff800000),
        UINT32_C(0x7fc12345),
    };
    for (uint32_t word : nonfinite_words) {
        G1RecoveryRequest bad = request;
        bad.raw_query[0] = float_from_bits(word);
        expect_global_error_transactional(bad);

        const float saved = fixture.db.features_offset(0);
        fixture.db.features_offset(0) = float_from_bits(word);
        expect_global_error_transactional(request);
        fixture.db.features_offset(0) = saved;
    }

    const uint32_t bad_scale_words[] = {
        UINT32_C(0x7f800000),
        UINT32_C(0xff800000),
        UINT32_C(0x7fc12345),
        UINT32_C(0x00000000),
        UINT32_C(0x80000000),
        UINT32_C(0xbf800000),
    };
    for (uint32_t word : bad_scale_words) {
        const float saved = fixture.db.features_scale(0);
        fixture.db.features_scale(0) = float_from_bits(word);
        expect_global_error_transactional(request);
        fixture.db.features_scale(0) = saved;
    }

    G1RecoveryRequest subtraction_overflow = request;
    subtraction_overflow.raw_query[0] = FLT_MAX;
    const float saved_offset = fixture.db.features_offset(0);
    fixture.db.features_offset(0) = -FLT_MAX;
    expect_global_error_transactional(subtraction_overflow);
    fixture.db.features_offset(0) = saved_offset;

    G1RecoveryRequest division_overflow = request;
    division_overflow.raw_query[0] = FLT_MAX;
    const float saved_scale = fixture.db.features_scale(0);
    fixture.db.features_scale(0) = float_from_bits(UINT32_C(0x00000001));
    expect_global_error_transactional(division_overflow);
    fixture.db.features_scale(0) = saved_scale;
}

static void test_seeded_transition_and_zero_seeded_incumbent_are_distinct()
{
    Fixture fixture(3);
    fixture.set_row_zero(0);
    fixture.db.features(0, 0) = 1.0f;
    fixture.set_row_zero(1);
    fixture.db.features(1, 0) = 2.0f;
    const G1RecoveryRequest request = fixture.request(1, 2, 1.0f, 5.0f);

    G1RecoveryCandidateSet accelerated;
    G1RecoveryProviderAudit accelerated_audit;
    G1RecoveryCandidateSet exhaustive;
    G1RecoveryProviderAudit exhaustive_audit;
    run_accelerated_and_exhaustive(
        fixture,
        request,
        accelerated,
        accelerated_audit,
        exhaustive,
        exhaustive_audit);

    CHECK(accelerated.count == 2U);
    CHECK(accelerated.records[0].kind == G1CandidateRecoveryTransition);
    CHECK(accelerated.records[0].selected_frame == 0);
    CHECK(float_bits(accelerated.records[0].selected_cost) ==
          UINT32_C(0x40000000));
    CHECK(accelerated.records[1].kind == G1CandidateIncumbent);
    CHECK(float_bits(accelerated.records[1].selected_cost) ==
          UINT32_C(0x40a00000));
    CHECK(accelerated_audit.recovery_incumbent_score_bits ==
          UINT32_C(0x40800000));
    CHECK(float_bits(accelerated.records[0].selected_cost) !=
          UINT32_C(0x3f800000));
}

static void check_single_bound_case(
    float candidate_score,
    float public_bound,
    float incumbent_feature,
    bool expect_candidate)
{
    Fixture fixture(3);
    fixture.set_row_zero(0);
    fixture.set_row_zero(1);
    fixture.db.features(1, 0) = incumbent_feature;
    const G1RecoveryRequest request =
        fixture.request(1, 2, candidate_score, public_bound);

    G1RecoveryCandidateSet accelerated;
    G1RecoveryProviderAudit accelerated_audit;
    G1RecoveryCandidateSet exhaustive;
    G1RecoveryProviderAudit exhaustive_audit;
    run_accelerated_and_exhaustive(
        fixture,
        request,
        accelerated,
        accelerated_audit,
        exhaustive,
        exhaustive_audit);
    CHECK(accelerated.count == (expect_candidate ? 2U : 1U));
    if (expect_candidate) {
        CHECK(accelerated.records[0].kind ==
              G1CandidateRecoveryTransition);
        CHECK(accelerated.records[0].selected_frame == 0);
        CHECK(float_bits(accelerated.records[0].selected_cost) ==
              float_bits(candidate_score));
    }
    CHECK(accelerated.records[accelerated.count - 1U].kind ==
          G1CandidateIncumbent);
    CHECK(float_bits(
              accelerated.records[accelerated.count - 1U].selected_cost) ==
          float_bits(public_bound));
}

static void test_both_incumbent_admission_bounds_are_strict()
{
    const float four = float_from_bits(UINT32_C(0x40800000));
    const float below_four = float_from_bits(UINT32_C(0x407fffff));
    const float above_four = float_from_bits(UINT32_C(0x40800001));

    check_single_bound_case(below_four, FLT_MAX, 2.0f, true);
    check_single_bound_case(four, FLT_MAX, 2.0f, false);
    check_single_bound_case(above_four, FLT_MAX, 2.0f, false);

    check_single_bound_case(below_four, four, 4.0f, true);
    check_single_bound_case(four, four, 4.0f, false);
    check_single_bound_case(above_four, four, 4.0f, false);

    const float private_score =
        float_from_bits(UINT32_C(0x4040097d));
    const float public_score =
        float_from_bits(UINT32_C(0x404005f1));
    const float gap_candidate =
        float_from_bits(UINT32_C(0x40400700));
    CHECK(gap_candidate < private_score);
    CHECK(public_score < gap_candidate);

    Fixture fixture(3);
    fixture.set_row_zero(0);
    fixture.set_row_zero(1);
    fixture.db.features(1, 0) =
        float_from_bits(UINT32_C(0x3e800000));
    fixture.db.features(1, 1) =
        float_from_bits(UINT32_C(0x3fdb670f));

    G1RecoveryRequest request =
        fixture.request(1, 2, gap_candidate, public_score);
    G1RecoveryCandidateSet accelerated;
    G1RecoveryProviderAudit accelerated_audit;
    G1RecoveryCandidateSet exhaustive;
    G1RecoveryProviderAudit exhaustive_audit;
    run_accelerated_and_exhaustive(
        fixture,
        request,
        accelerated,
        accelerated_audit,
        exhaustive,
        exhaustive_audit);
    CHECK(accelerated_audit.recovery_incumbent_score_bits ==
          float_bits(private_score));
    CHECK(accelerated.count == 1U);
    CHECK(accelerated.records[0].kind == G1CandidateIncumbent);
    CHECK(float_bits(accelerated.records[0].selected_cost) ==
          float_bits(public_score));

    request.public_incumbent_cost = FLT_MAX;
    run_accelerated_and_exhaustive(
        fixture,
        request,
        accelerated,
        accelerated_audit,
        exhaustive,
        exhaustive_audit);
    CHECK(accelerated_audit.recovery_incumbent_score_bits ==
          float_bits(private_score));
    CHECK(accelerated.count == 2U);
    CHECK(accelerated.records[0].kind == G1CandidateRecoveryTransition);
    CHECK(float_bits(accelerated.records[0].selected_cost) ==
          float_bits(gap_candidate));
    CHECK(accelerated.records[1].kind == G1CandidateIncumbent);
    CHECK(float_bits(accelerated.records[1].selected_cost) ==
          UINT32_C(0x7f7fffff));
}

static void set_integer_square_cost(Fixture& fixture, int frame, int cost)
{
    fixture.set_row_zero(frame);
    for (int feature = 0; feature < cost; ++feature) {
        fixture.db.features(frame, feature) = 1.0f;
    }
}

static void test_top_six_ties_eviction_dedup_and_incumbent_order()
{
    for (int admitted = 0; admitted <= 8; ++admitted) {
        Fixture fixture(12);
        fixture.set_row_zero(10);
        fixture.db.features(10, 0) = 10.0f;
        for (int frame = 0; frame < admitted; ++frame) {
            fixture.set_row_zero(frame);
            fixture.db.features(frame, 0) = static_cast<float>(frame + 1);
        }
        const G1RecoveryRequest request =
            fixture.request(10, 11, 0.0f, 100.0f);
        G1RecoveryCandidateSet accelerated;
        G1RecoveryProviderAudit accelerated_audit;
        G1RecoveryCandidateSet exhaustive;
        G1RecoveryProviderAudit exhaustive_audit;
        run_accelerated_and_exhaustive(
            fixture,
            request,
            accelerated,
            accelerated_audit,
            exhaustive,
            exhaustive_audit);

        const uint32_t retained = static_cast<uint32_t>(
            std::min(admitted, static_cast<int>(G1RecoveryTransitionCapacity)));
        CHECK(accelerated.count == retained + 1U);
        for (uint32_t rank = 0U; rank < retained; ++rank) {
            CHECK(accelerated.records[rank].selected_frame ==
                  static_cast<int>(rank));
            CHECK(accelerated.records[rank].recovery_rank == rank);
        }
        const G1CandidateRecord& incumbent =
            accelerated.records[accelerated.count - 1U];
        CHECK(incumbent.kind == G1CandidateIncumbent);
        CHECK(incumbent.selected_frame == 10);
        CHECK(incumbent.executed_frame == 11);
        CHECK(float_bits(incumbent.selected_cost) ==
              UINT32_C(0x42c80000));
    }

    {
        Fixture fixture(11);
        fixture.set_row_zero(9);
        fixture.db.features(9, 0) = 10.0f;
        for (int frame = 0; frame <= 5; ++frame) {
            set_integer_square_cost(fixture, frame, frame + 1);
        }
        set_integer_square_cost(fixture, 6, 6);
        fixture.set_row_zero(7);
        fixture.db.features(7, 0) = 0.5f;
        const G1RecoveryRequest request =
            fixture.request(9, 10, 0.0f, 100.0f);
        G1RecoveryCandidateSet accelerated;
        G1RecoveryProviderAudit accelerated_audit;
        G1RecoveryCandidateSet exhaustive;
        G1RecoveryProviderAudit exhaustive_audit;
        run_accelerated_and_exhaustive(
            fixture,
            request,
            accelerated,
            accelerated_audit,
            exhaustive,
            exhaustive_audit);
        const int expected_frames[] = {7, 0, 1, 2, 3, 4};
        CHECK(accelerated.count == 7U);
        for (uint32_t rank = 0U;
             rank < G1RecoveryTransitionCapacity;
             ++rank) {
            CHECK(accelerated.records[rank].selected_frame ==
                  expected_frames[rank]);
        }
    }

    {
        Fixture fixture(19);
        for (int frame = 0; frame <= 5; ++frame) {
            set_integer_square_cost(fixture, frame, frame + 1);
        }
        set_integer_square_cost(fixture, 16, 6);
        fixture.set_row_zero(17);
        fixture.db.features(17, 0) = 10.0f;
        fixture.rebuild_bounds();
        for (uint32_t feature = 0U;
             feature < G1RecoveryFeatureCount;
             ++feature) {
            const float endpoint = feature < 6U ? 1.0f : 0.0f;
            fixture.db.bound_sm_min(1, static_cast<int>(feature)) = endpoint;
            fixture.db.bound_sm_max(1, static_cast<int>(feature)) = endpoint;
        }
        const G1RecoveryRequest request =
            fixture.request(17, 18, 0.0f, 100.0f);
        G1RecoveryCandidateSet accelerated;
        G1RecoveryProviderAudit accelerated_audit;
        G1RecoveryCandidateSet exhaustive;
        G1RecoveryProviderAudit exhaustive_audit;
        char accelerated_error[192] = {};
        char exhaustive_error[192] = {};
        CHECK(g1_recovery_candidates_build_audited(
                  accelerated,
                  request,
                  accelerated_audit,
                  accelerated_error,
                  static_cast<int>(sizeof(accelerated_error))) ==
              G1RecoveryProviderOk);
        CHECK(g1_recovery_candidates_exhaustive_for_test(
                  exhaustive,
                  request,
                  exhaustive_audit,
                  exhaustive_error,
                  static_cast<int>(sizeof(exhaustive_error))) ==
              G1RecoveryProviderOk);
        check_sets_equal(accelerated, exhaustive);
        CHECK(accelerated.count == 7U);
        CHECK(accelerated.records[5].selected_frame == 5);
        CHECK(accelerated.records[5].source_range == 0);
        CHECK(float_bits(accelerated.records[5].selected_cost) ==
              UINT32_C(0x40c00000));
        CHECK(accelerated.work.rows_tested == 17U);
        CHECK(accelerated.work.full_scores_materialized == 17U);
    }

    static const uint32_t inversion_a[G1RecoveryFeatureCount] = {
        UINT32_C(0x3e76b806), UINT32_C(0x3ea08e4a),
        UINT32_C(0xbed358ea), UINT32_C(0xbef09450),
        UINT32_C(0xbf6f7616), UINT32_C(0xbffbad3a),
        UINT32_C(0x3ca7c2d5), UINT32_C(0xbe90bb1d),
        UINT32_C(0xbf462b72), UINT32_C(0xbebc2ed2),
        UINT32_C(0x3eb6d0d9), UINT32_C(0xbf52f8b9),
        UINT32_C(0xbebcdda3), UINT32_C(0x3e44dcbc),
        UINT32_C(0x3f489f8e), UINT32_C(0x3f3e1335),
        UINT32_C(0xbed05f70), UINT32_C(0x3d8da082),
        UINT32_C(0xbe870002), UINT32_C(0x3e12312d),
        UINT32_C(0xbf26d190), UINT32_C(0xbf309386),
        UINT32_C(0x3c7106e2), UINT32_C(0xbcb0dfde),
        UINT32_C(0x3e95b2a6), UINT32_C(0xbf8594b6),
        UINT32_C(0x3f1a3b4e), UINT32_C(0xbe757c29),
        UINT32_C(0x3e9389b0), UINT32_C(0x3f54c596),
        UINT32_C(0x3eb57e61),
    };
    static const uint32_t inversion_b[G1RecoveryFeatureCount] = {
        UINT32_C(0x3f21629e), UINT32_C(0x3f585617),
        UINT32_C(0x3faea056), UINT32_C(0xbf1fba50),
        UINT32_C(0x3f038e09), UINT32_C(0x3d84b2a9),
        UINT32_C(0xbcb7213c), UINT32_C(0xbf1c6c11),
        UINT32_C(0x3fb3b6b2), UINT32_C(0x3da1eb7d),
        UINT32_C(0xbf0437fd), UINT32_C(0x3cdbcd5c),
        UINT32_C(0xbe41121a), UINT32_C(0x3e581e78),
        UINT32_C(0x3edcba12), UINT32_C(0x3f656afc),
        UINT32_C(0xbdb8ec54), UINT32_C(0x3ec1cd11),
        UINT32_C(0xbf68ed43), UINT32_C(0x3e4af82f),
        UINT32_C(0xbdf8d97a), UINT32_C(0xbf777d29),
        UINT32_C(0x3ed3b084), UINT32_C(0x3f8d4823),
        UINT32_C(0xbe2999bd), UINT32_C(0xbf05d2f2),
        UINT32_C(0x3e338467), UINT32_C(0x3eb032f5),
        UINT32_C(0x3f4fedb4), UINT32_C(0x3d78fac2),
        UINT32_C(0xbeae08de),
    };
    CHECK(float_bits(strict_score_words(inversion_a, 0.0f)) ==
          UINT32_C(0x413e77bb));
    CHECK(float_bits(strict_score_words(inversion_b, 0.0f)) ==
          UINT32_C(0x413e77bc));
    CHECK(float_bits(strict_score_words(inversion_a, 1.0f)) ==
          UINT32_C(0x414e77bc));
    CHECK(float_bits(strict_score_words(inversion_b, 1.0f)) ==
          UINT32_C(0x414e77bb));

    {
        Fixture fixture(9);
        for (int frame = 0; frame < 5; ++frame) {
            set_integer_square_cost(fixture, frame, frame);
        }
        for (uint32_t feature = 0U;
             feature < G1RecoveryFeatureCount;
             ++feature) {
            fixture.db.features(5, static_cast<int>(feature)) =
                float_from_bits(inversion_a[feature] ^
                                UINT32_C(0x80000000));
            fixture.db.features(6, static_cast<int>(feature)) =
                float_from_bits(inversion_b[feature] ^
                                UINT32_C(0x80000000));
        }
        fixture.set_row_zero(8);
        fixture.db.features(8, 0) = 10.0f;
        const G1RecoveryRequest request =
            fixture.request(8, 7, 1.0f, 100.0f);
        G1RecoveryCandidateSet accelerated;
        G1RecoveryProviderAudit accelerated_audit;
        G1RecoveryCandidateSet exhaustive;
        G1RecoveryProviderAudit exhaustive_audit;
        run_accelerated_and_exhaustive(
            fixture,
            request,
            accelerated,
            accelerated_audit,
            exhaustive,
            exhaustive_audit);
        CHECK(accelerated.count == 7U);
        for (uint32_t rank = 0U; rank < 5U; ++rank) {
            CHECK(accelerated.records[rank].selected_frame ==
                  static_cast<int>(rank));
        }
        CHECK(accelerated.records[5].selected_frame == 6);
        CHECK(float_bits(accelerated.records[5].selected_cost) ==
              UINT32_C(0x414e77bb));
        for (uint32_t index = 0U; index < accelerated.count; ++index) {
            CHECK(float_bits(accelerated.records[index].selected_cost) !=
                  UINT32_C(0x413e77bb));
            CHECK(float_bits(accelerated.records[index].selected_cost) !=
                  UINT32_C(0x413e77bc));
        }
        CHECK(accelerated_audit.recovery_incumbent_score_bits !=
              UINT32_C(0x413e77bb));
        CHECK(accelerated_audit.recovery_incumbent_score_bits !=
              UINT32_C(0x413e77bc));
    }

    {
        Fixture fixture(3);
        fixture.set_row_zero(0);
        fixture.db.features(0, 0) = 1.0f;
        fixture.set_row_zero(2);
        fixture.db.features(2, 0) = 2.0f;
        const G1RecoveryRequest request =
            fixture.request(2, 2, 0.0f, 4.0f);
        G1RecoveryCandidateSet accelerated;
        G1RecoveryProviderAudit accelerated_audit;
        G1RecoveryCandidateSet exhaustive;
        G1RecoveryProviderAudit exhaustive_audit;
        run_accelerated_and_exhaustive(
            fixture,
            request,
            accelerated,
            accelerated_audit,
            exhaustive,
            exhaustive_audit);
        CHECK(accelerated.count == 1U);
        CHECK(accelerated.records[0].kind ==
              G1CandidateRecoveryTransition);
        CHECK(accelerated.records[0].selected_frame == 0);
    }
}

static void test_exclusions_sequential_idle_and_end_of_animation()
{
    {
        Fixture fixture(
            145,
            std::vector<std::pair<int, int> >{
                std::make_pair(0, 5),
                std::make_pair(5, 75),
                std::make_pair(75, 145)});
        fixture.set_row_zero(35);
        fixture.db.features(35, 0) = 10.0f;
        const int admitted_frames[] = {5, 15, 75, 124};
        for (int index = 0; index < 4; ++index) {
            fixture.set_row_zero(admitted_frames[index]);
            fixture.db.features(admitted_frames[index], 0) =
                static_cast<float>(index + 1);
        }
        const int excluded_frames[] = {16, 54, 55, 80, 125, 144};
        for (int frame : excluded_frames) {
            fixture.set_row_zero(frame);
        }
        G1RecoveryRequest request =
            fixture.request(35, 80, 0.0f, 100.0f);
        request.ignore_range_end = 20;
        request.ignore_surrounding = 20;
        G1RecoveryCandidateSet accelerated;
        G1RecoveryProviderAudit accelerated_audit;
        G1RecoveryCandidateSet exhaustive;
        G1RecoveryProviderAudit exhaustive_audit;
        run_accelerated_and_exhaustive(
            fixture,
            request,
            accelerated,
            accelerated_audit,
            exhaustive,
            exhaustive_audit);
        CHECK(accelerated.count == 5U);
        for (uint32_t rank = 0U; rank < 4U; ++rank) {
            CHECK(accelerated.records[rank].selected_frame ==
                  admitted_frames[rank]);
            CHECK(accelerated.records[rank].source_range ==
                  (rank < 2U ? 1 : 2));
        }
        CHECK(accelerated.records[3].executed_frame == 125);
        CHECK(accelerated.records[4].kind == G1CandidateIncumbent);
        CHECK(accelerated.records[4].selected_frame == 35);
        CHECK(accelerated.records[4].executed_frame == 36);
    }

    {
        Fixture fixture(3);
        fixture.set_row_zero(0);
        fixture.db.features(0, 0) = 2.0f;
        fixture.set_row_zero(2);
        fixture.db.features(2, 0) = 1.0f;
        const G1RecoveryRequest request =
            fixture.request(0, 1, 0.0f, 4.0f);
        G1RecoveryCandidateSet accelerated;
        G1RecoveryProviderAudit accelerated_audit;
        G1RecoveryCandidateSet exhaustive;
        G1RecoveryProviderAudit exhaustive_audit;
        run_accelerated_and_exhaustive(
            fixture,
            request,
            accelerated,
            accelerated_audit,
            exhaustive,
            exhaustive_audit);
        CHECK(accelerated.count == 2U);
        CHECK(accelerated.records[0].selected_frame == 2);
        CHECK(accelerated.records[0].executed_frame == 2);
    }

    {
        Fixture fixture(4);
        fixture.set_row_zero(2);
        fixture.db.features(2, 0) = 2.0f;
        const G1RecoveryRequest request =
            fixture.request(2, 2, 0.0f, 4.0f);
        G1RecoveryCandidateSet accelerated;
        G1RecoveryProviderAudit accelerated_audit;
        G1RecoveryCandidateSet exhaustive;
        G1RecoveryProviderAudit exhaustive_audit;
        run_accelerated_and_exhaustive(
            fixture,
            request,
            accelerated,
            accelerated_audit,
            exhaustive,
            exhaustive_audit);
        CHECK(accelerated.count == 0U);
    }

    {
        Fixture fixture(3);
        fixture.set_row_zero(1);
        fixture.db.features(1, 0) = 2.0f;
        const G1RecoveryRequest request =
            fixture.request(1, 2, 0.0f, 4.0f);
        G1RecoveryCandidateSet accelerated;
        G1RecoveryProviderAudit accelerated_audit;
        G1RecoveryCandidateSet exhaustive;
        G1RecoveryProviderAudit exhaustive_audit;
        run_accelerated_and_exhaustive(
            fixture,
            request,
            accelerated,
            accelerated_audit,
            exhaustive,
            exhaustive_audit);
        CHECK(accelerated.count == 1U);
        CHECK(accelerated.records[0].kind == G1CandidateIncumbent);
    }

    {
        Fixture fixture(4);
        fixture.set_row_zero(0);
        fixture.set_row_zero(1);
        fixture.db.features(1, 0) = 0.5f;
        fixture.set_row_zero(2);
        fixture.db.features(2, 0) = 2.0f;
        const G1RecoveryRequest request =
            fixture.request(2, 3, 1.0f, 4.0f);
        G1RecoveryCandidateSet accelerated;
        G1RecoveryProviderAudit accelerated_audit;
        G1RecoveryCandidateSet exhaustive;
        G1RecoveryProviderAudit exhaustive_audit;
        run_accelerated_and_exhaustive(
            fixture,
            request,
            accelerated,
            accelerated_audit,
            exhaustive,
            exhaustive_audit);
        CHECK(accelerated.count == 3U);
        CHECK(accelerated.records[0].selected_frame == 0);
        CHECK(float_bits(accelerated.records[0].selected_cost) ==
              UINT32_C(0x3f800000));
        CHECK(accelerated.records[1].selected_frame == 1);
        CHECK(float_bits(accelerated.records[1].selected_cost) ==
              UINT32_C(0x3fa00000));
    }

    {
        Fixture fixture(3);
        fixture.set_row_zero(1);
        fixture.db.features(1, 0) = 1.0f;
        const G1RecoveryRequest request =
            fixture.request(1, 2, 0.0f, FLT_MAX);
        G1RecoveryCandidateSet accelerated;
        G1RecoveryProviderAudit accelerated_audit;
        G1RecoveryCandidateSet exhaustive;
        G1RecoveryProviderAudit exhaustive_audit;
        run_accelerated_and_exhaustive(
            fixture,
            request,
            accelerated,
            accelerated_audit,
            exhaustive,
            exhaustive_audit);
        CHECK(accelerated_audit.recovery_incumbent_score_bits ==
              UINT32_C(0x3f800000));
        CHECK(accelerated.count == 1U);
        CHECK(accelerated.records[0].kind == G1CandidateIncumbent);
        CHECK(float_bits(accelerated.records[0].selected_cost) ==
              UINT32_C(0x7f7fffff));
    }

    {
        Fixture fixture(3);
        fixture.set_row_zero(0);
        fixture.db.features(0, 0) = 1.0f;
        fixture.set_row_zero(1);
        fixture.db.features(1, 0) = 2.0f;
        const G1RecoveryRequest request =
            fixture.request(1, 2, 0.0f, FLT_MAX);
        G1RecoveryCandidateSet accelerated;
        G1RecoveryProviderAudit accelerated_audit;
        G1RecoveryCandidateSet exhaustive;
        G1RecoveryProviderAudit exhaustive_audit;
        run_accelerated_and_exhaustive(
            fixture,
            request,
            accelerated,
            accelerated_audit,
            exhaustive,
            exhaustive_audit);
        CHECK(accelerated_audit.recovery_incumbent_score_bits ==
              UINT32_C(0x40800000));
        CHECK(accelerated.count == 2U);
        CHECK(accelerated.records[0].kind ==
              G1CandidateRecoveryTransition);
        CHECK(float_bits(accelerated.records[0].selected_cost) ==
              UINT32_C(0x3f800000));
        CHECK(accelerated.records[1].kind == G1CandidateIncumbent);
        CHECK(float_bits(accelerated.records[1].selected_cost) ==
              UINT32_C(0x7f7fffff));
    }
}

static void check_work_equal(
    const G1RecoveryWork& left,
    const G1RecoveryWork& right)
{
    CHECK(left.accelerated_traversals == right.accelerated_traversals);
    CHECK(left.large_bounds_tested == right.large_bounds_tested);
    CHECK(left.small_bounds_tested == right.small_bounds_tested);
    CHECK(left.rows_tested == right.rows_tested);
    CHECK(left.full_scores_materialized == right.full_scores_materialized);
}

static void populate_oracle_fixture(Fixture& fixture)
{
    for (int frame = 0; frame < fixture.db.nframes(); ++frame) {
        fixture.set_row_zero(frame);
        fixture.db.features(frame, 0) =
            static_cast<float>(frame % 17) * 0.125f;
        fixture.db.features(frame, 1) =
            static_cast<float>((frame / 3) % 9) * 0.25f;
        fixture.db.features(frame, 2) =
            static_cast<float>((frame / 11) % 5) * 0.5f;
    }
}

static void test_accelerated_provider_matches_exhaustive_oracle()
{
    for (int fixture_index = 0; fixture_index < 3; ++fixture_index) {
        const int frames = 193 + fixture_index * 16;
        Fixture fixture(
            frames,
            std::vector<std::pair<int, int> >{
                std::make_pair(0, 73),
                std::make_pair(73, 139),
                std::make_pair(139, frames)});
        populate_oracle_fixture(fixture);
        const int incumbent = 90 + fixture_index;
        const int legacy = 150 + fixture_index;
        G1RecoveryRequest request = fixture.request(
            incumbent,
            legacy,
            fixture_index == 1 ? 1.0f : 0.0f,
            FLT_MAX);
        request.ignore_range_end = 3 + fixture_index;
        request.ignore_surrounding = 5 + fixture_index;

        G1RecoveryCandidateSet accelerated;
        G1RecoveryProviderAudit accelerated_audit;
        G1RecoveryCandidateSet exhaustive;
        G1RecoveryProviderAudit exhaustive_audit;
        run_accelerated_and_exhaustive(
            fixture,
            request,
            accelerated,
            accelerated_audit,
            exhaustive,
            exhaustive_audit);
        CHECK(accelerated.work.large_bounds_tested > 0U);
        CHECK(accelerated.work.small_bounds_tested > 0U);
        CHECK(accelerated.work.full_scores_materialized <=
              accelerated.work.rows_tested);
        CHECK(accelerated.work.rows_tested <=
              exhaustive.work.rows_tested);
        CHECK(accelerated_audit.exhaustive_rows_tested == 0U);
        CHECK(exhaustive_audit.exhaustive_rows_tested > 0U);

        G1RecoveryCandidateSet ordinary;
        char error[192] = {};
        CHECK(g1_recovery_candidates_build(
                  ordinary,
                  request,
                  error,
                  static_cast<int>(sizeof(error))) ==
              G1RecoveryProviderOk);
        check_sets_equal(ordinary, accelerated);
        check_work_equal(ordinary.work, accelerated.work);
    }
}

enum AliasPair
{
    AliasRequestOutput,
    AliasRequestAudit,
    AliasOutputAudit,
};

static void expect_pair_alias_rejected(
    Fixture& fixture,
    AliasPair pair,
    std::size_t offset)
{
    alignas(std::max_align_t) uint8_t storage[1024];
    poison_bytes(storage, sizeof(storage), UINT8_C(0x37));
    G1RecoveryRequest separate_request = fixture.request(1, 2);
    G1RecoveryCandidateSet separate_output;
    G1RecoveryProviderAudit separate_audit;
    poison_bytes(
        &separate_output, sizeof(separate_output), UINT8_C(0x72));
    poison_bytes(&separate_audit, sizeof(separate_audit), UINT8_C(0x9b));

    G1RecoveryRequest* request = &separate_request;
    G1RecoveryCandidateSet* output = &separate_output;
    G1RecoveryProviderAudit* audit = &separate_audit;
    if (pair == AliasRequestOutput) {
        request = new (storage) G1RecoveryRequest(fixture.request(1, 2));
        output = reinterpret_cast<G1RecoveryCandidateSet*>(storage + offset);
    } else if (pair == AliasRequestAudit) {
        request = new (storage) G1RecoveryRequest(fixture.request(1, 2));
        audit = reinterpret_cast<G1RecoveryProviderAudit*>(storage + offset);
    } else {
        output = new (storage) G1RecoveryCandidateSet();
        poison_bytes(output, sizeof(*output), UINT8_C(0x44));
        audit = reinterpret_cast<G1RecoveryProviderAudit*>(storage + offset);
    }
    uint8_t before[sizeof(storage)];
    std::memcpy(before, storage, sizeof(storage));
    const G1RecoveryCandidateSet separate_output_before = separate_output;
    const G1RecoveryProviderAudit separate_audit_before = separate_audit;
    const G1RecoveryRequest separate_request_before = separate_request;
    char error[192] = {};
    CHECK(g1_recovery_candidates_build_audited(
              *output,
              *request,
              *audit,
              error,
              static_cast<int>(sizeof(error))) ==
          G1RecoveryProviderGlobalError);
    CHECK(std::memcmp(storage, before, sizeof(storage)) == 0);
    CHECK(object_bytes_equal(separate_output, separate_output_before));
    CHECK(object_bytes_equal(separate_audit, separate_audit_before));
    CHECK(object_bytes_equal(separate_request, separate_request_before));
}

enum ErrorAliasOwner
{
    ErrorAliasesRequest,
    ErrorAliasesOutput,
    ErrorAliasesAudit,
};

static void expect_error_alias_rejected(
    Fixture& fixture,
    ErrorAliasOwner owner,
    std::size_t offset)
{
    G1RecoveryRequest request = fixture.request(1, 2);
    G1RecoveryCandidateSet output;
    G1RecoveryProviderAudit audit;
    poison_bytes(&output, sizeof(output), UINT8_C(0x29));
    poison_bytes(&audit, sizeof(audit), UINT8_C(0xc1));
    const G1RecoveryRequest request_before = request;
    const G1RecoveryCandidateSet output_before = output;
    const G1RecoveryProviderAudit audit_before = audit;
    char* error = nullptr;
    if (owner == ErrorAliasesRequest) {
        error = reinterpret_cast<char*>(&request) + offset;
    } else if (owner == ErrorAliasesOutput) {
        error = reinterpret_cast<char*>(&output) + offset;
    } else {
        error = reinterpret_cast<char*>(&audit) + offset;
    }
    CHECK(g1_recovery_candidates_build_audited(
              output, request, audit, error, 16) ==
          G1RecoveryProviderGlobalError);
    CHECK(object_bytes_equal(request, request_before));
    CHECK(object_bytes_equal(output, output_before));
    CHECK(object_bytes_equal(audit, audit_before));
}

static void test_malformed_alias_overflow_and_capacity_fail_closed()
{
    {
        Fixture fixture(3);
        G1RecoveryRequest request = fixture.request(1, 2);
        request.db = nullptr;
        expect_global_error_transactional(request);
    }
    {
        Fixture fixture(3);
        const int invalid_frames[] = {-1, 3, INT_MAX};
        for (int frame : invalid_frames) {
            G1RecoveryRequest request = fixture.request(1, 2);
            request.incumbent_frame = frame;
            expect_global_error_transactional(request);
            request = fixture.request(1, 2);
            request.legacy_selected_frame = frame;
            expect_global_error_transactional(request);
        }
    }
    {
        Fixture fixture(3);
        const uint32_t invalid_cost_words[] = {
            UINT32_C(0xbf800000),
            UINT32_C(0x7f800000),
            UINT32_C(0xff800000),
            UINT32_C(0x7fc00001),
        };
        for (uint32_t word : invalid_cost_words) {
            G1RecoveryRequest request = fixture.request(1, 2);
            request.transition_cost = float_from_bits(word);
            expect_global_error_transactional(request);
            request = fixture.request(1, 2);
            request.public_incumbent_cost = float_from_bits(word);
            expect_global_error_transactional(request);
        }
        G1RecoveryRequest request = fixture.request(1, 2);
        request.ignore_range_end = -1;
        expect_global_error_transactional(request);
        request = fixture.request(1, 2);
        request.ignore_surrounding = -1;
        expect_global_error_transactional(request);
    }

    {
        Fixture fixture(3);
        fixture.db.features.cols = 30;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.features.rows = 2;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.features_offset.size = 30;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.features_scale.size = 32;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.bone_velocities.rows = 2;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.bone_rotations.cols = 2;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.bone_angular_velocities.rows = 4;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.bone_parents.size = 2;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.terrain_features.cols = 3;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.contact_states.rows = 2;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.range_stops.size = 2;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.range_starts(0) = -1;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.range_starts(0) = 2;
        fixture.db.range_stops(0) = 1;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.range_stops(0) = 4;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.bound_sm_min.rows = 2;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.bound_sm_max.cols = 30;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.bound_lr_min.rows = 2;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.bound_lr_max.cols = 32;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.bound_sm_min(0, 0) =
            float_from_bits(UINT32_C(0x7fc00001));
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.bound_lr_max(0, 0) =
            float_from_bits(UINT32_C(0x7f800000));
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.bound_sm_min(0, 0) = 2.0f;
        fixture.db.bound_sm_max(0, 0) = 1.0f;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.db.bound_lr_min(0, 0) = 2.0f;
        fixture.db.bound_lr_max(0, 0) = 1.0f;
        expect_global_error_transactional(fixture.request(1, 2));
    }
    {
        Fixture fixture(3);
        fixture.set_row_zero(0);
        fixture.db.features(0, 0) =
            float_from_bits(UINT32_C(0x7fc00001));
        fixture.rebuild_bounds();
        fixture.db.bound_sm_min(0, 0) = 0.0f;
        fixture.db.bound_sm_max(0, 0) = 0.0f;
        fixture.db.bound_lr_min(0, 0) = 0.0f;
        fixture.db.bound_lr_max(0, 0) = 0.0f;
        expect_global_error_transactional(fixture.request(1, 2));
    }

    {
        Fixture fixture(3);
        G1RecoveryRequest request = fixture.request(1, 2);
        request.ignore_range_end = INT_MAX;
        request.ignore_surrounding = INT_MAX;
        G1RecoveryCandidateSet output;
        G1RecoveryProviderAudit audit;
        char error[64] = {};
        CHECK(g1_recovery_candidates_build_audited(
                  output,
                  request,
                  audit,
                  error,
                  static_cast<int>(sizeof(error))) ==
              G1RecoveryProviderOk);
        CHECK(output.count == 1U);
        CHECK(output.records[0].kind == G1CandidateIncumbent);
    }
    {
        Fixture fixture(3);
        G1RecoveryRequest request = fixture.request(1, 2);
        const int oversized_frames =
            INT_MAX / static_cast<int>(G1RecoveryFeatureCount) + 1;
        fixture.db.bone_positions.rows = oversized_frames;
        fixture.db.bone_velocities.rows = oversized_frames;
        fixture.db.bone_rotations.rows = oversized_frames;
        fixture.db.bone_angular_velocities.rows = oversized_frames;
        fixture.db.features.rows = oversized_frames;
        fixture.db.terrain_features.rows = oversized_frames;
        fixture.db.contact_states.rows = oversized_frames;
        fixture.db.range_stops(0) = oversized_frames;
        const int oversized_small =
            (oversized_frames - 1) / BOUND_SM_SIZE + 1;
        const int oversized_large =
            (oversized_frames - 1) / BOUND_LR_SIZE + 1;
        fixture.db.bound_sm_min.rows = oversized_small;
        fixture.db.bound_sm_max.rows = oversized_small;
        fixture.db.bound_lr_min.rows = oversized_large;
        fixture.db.bound_lr_max.rows = oversized_large;
        expect_global_error_transactional(request);
        fixture.db.bone_positions.rows = 3;
        fixture.db.bone_velocities.rows = 3;
        fixture.db.bone_rotations.rows = 3;
        fixture.db.bone_angular_velocities.rows = 3;
        fixture.db.features.rows = 3;
        fixture.db.terrain_features.rows = 3;
        fixture.db.contact_states.rows = 3;
        fixture.db.range_stops(0) = 3;
        fixture.db.bound_sm_min.rows = 1;
        fixture.db.bound_sm_max.rows = 1;
        fixture.db.bound_lr_min.rows = 1;
        fixture.db.bound_lr_max.rows = 1;
    }
    {
        Fixture fixture(3);
        fixture.db.range_starts.size = INT_MAX;
        fixture.db.range_stops.size = INT_MAX;
        expect_global_error_transactional(fixture.request(1, 2));
        fixture.db.range_starts.size = 1;
        fixture.db.range_stops.size = 1;
    }

    Fixture alias_fixture(3);
    for (AliasPair pair : {AliasRequestOutput,
                           AliasRequestAudit,
                           AliasOutputAudit}) {
        expect_pair_alias_rejected(alias_fixture, pair, 0U);
        expect_pair_alias_rejected(alias_fixture, pair, 16U);
    }
    for (ErrorAliasOwner owner : {ErrorAliasesRequest,
                                  ErrorAliasesOutput,
                                  ErrorAliasesAudit}) {
        expect_error_alias_rejected(alias_fixture, owner, 0U);
        expect_error_alias_rejected(alias_fixture, owner, 4U);
    }

    {
        G1RecoveryRequest request = alias_fixture.request(1, 2);
        G1RecoveryCandidateSet output;
        G1RecoveryProviderAudit audit;
        poison_bytes(&output, sizeof(output), UINT8_C(0x83));
        poison_bytes(&audit, sizeof(audit), UINT8_C(0x14));
        const G1RecoveryCandidateSet output_before = output;
        const G1RecoveryProviderAudit audit_before = audit;
        char error[16] = {};
        CHECK(g1_recovery_candidates_build_audited(
                  output, request, audit, error, -1) ==
              G1RecoveryProviderGlobalError);
        CHECK(object_bytes_equal(output, output_before));
        CHECK(object_bytes_equal(audit, audit_before));
        CHECK(g1_recovery_candidates_build_audited(
                  output, request, audit, nullptr, 1) ==
              G1RecoveryProviderGlobalError);
        CHECK(object_bytes_equal(output, output_before));
        CHECK(object_bytes_equal(audit, audit_before));
    }
}

static void print_provider_parity_transcript()
{
    Fixture fixture(
        67,
        std::vector<std::pair<int, int> >{std::make_pair(3, 67)});
    populate_oracle_fixture(fixture);
    G1RecoveryRequest request = fixture.request(30, 66, 1.0f, FLT_MAX);
    request.ignore_range_end = 2;
    request.ignore_surrounding = 4;
    fixture.rebuild_bounds();

    G1RecoveryCandidateSet output;
    G1RecoveryProviderAudit audit;
    char error[128] = {};
    CHECK(g1_recovery_candidates_build_audited(
              output,
              request,
              audit,
              error,
              static_cast<int>(sizeof(error))) ==
          G1RecoveryProviderOk);
    std::printf(
        "provider count=%u traversal=%u lr=%u sm=%u rows=%u full=%u "
        "incumbent=%08x exhaustive=%u\n",
        output.count,
        output.work.accelerated_traversals,
        output.work.large_bounds_tested,
        output.work.small_bounds_tested,
        output.work.rows_tested,
        output.work.full_scores_materialized,
        audit.recovery_incumbent_score_bits,
        audit.exhaustive_rows_tested);
    std::printf("normalized");
    for (uint32_t feature = 0U;
         feature < G1RecoveryFeatureCount;
         ++feature) {
        std::printf(" %08x", audit.normalized_query_bits[feature]);
    }
    std::printf("\n");
    for (uint32_t index = 0U; index < output.count; ++index) {
        const G1CandidateRecord& record = output.records[index];
        std::printf(
            "record %u kind=%u selected=%d executed=%d range=%d "
            "cost=%08x rank=%u transitioned=%u\n",
            index,
            static_cast<unsigned>(record.kind),
            record.selected_frame,
            record.executed_frame,
            record.source_range,
            float_bits(record.selected_cost),
            record.recovery_rank,
            record.transitioned ? 1U : 0U);
    }
}

int main(int argc, char** argv)
{
    test_strict_normalization_owns_all_31_words();
    test_seeded_transition_and_zero_seeded_incumbent_are_distinct();
    test_both_incumbent_admission_bounds_are_strict();
    test_top_six_ties_eviction_dedup_and_incumbent_order();
    test_exclusions_sequential_idle_and_end_of_animation();
    test_accelerated_provider_matches_exhaustive_oracle();
    test_malformed_alias_overflow_and_capacity_fail_closed();
    if (argc == 2 && std::strcmp(argv[1], "--parity") == 0) {
        print_provider_parity_transcript();
    }
    return 0;
}
