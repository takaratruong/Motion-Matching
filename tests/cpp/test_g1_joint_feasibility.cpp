#include "sonic/cpp/g1_joint_feasibility.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

static void check(bool condition, const char* expression, int line)
{
    if (!condition) {
        std::fprintf(
            stderr,
            "G1 joint feasibility check failed at line %d: %s\n",
            line,
            expression);
        std::exit(1);
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

using BuildCertificateSignature = bool (*)(
    sonic_joint_feasibility_certificate&,
    const database&,
    const sonic_joint_contract_entry (&)[SonicG1JointCount],
    char*,
    int);

static_assert(
    std::is_same<
        decltype(&sonic_build_joint_feasibility_certificate),
        BuildCertificateSignature>::value,
    "public feasibility-certificate signature");

struct FeasibilityFixture
{
    static constexpr int FrameCount = 4;

    database db;
    sonic_joint_contract_entry contract[SonicG1JointCount];
    char error[512];

    FeasibilityFixture()
    {
        static const int parents[G1_BoneCount] = {
            -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
        };

        db.bone_positions.resize(FrameCount, G1_BoneCount);
        db.bone_velocities.resize(FrameCount, G1_BoneCount);
        db.bone_rotations.resize(FrameCount, G1_BoneCount);
        db.bone_angular_velocities.resize(FrameCount, G1_BoneCount);
        db.bone_parents.resize(G1_BoneCount);
        db.range_starts.resize(2);
        db.range_stops.resize(2);
        db.range_starts(0) = 0;
        db.range_stops(0) = 2;
        db.range_starts(1) = 2;
        db.range_stops(1) = FrameCount;

        for (int bone = 0; bone < G1_BoneCount; ++bone) {
            db.bone_parents(bone) = parents[bone];
        }
        for (int frame = 0; frame < FrameCount; ++frame) {
            for (int bone = 0; bone < G1_BoneCount; ++bone) {
                db.bone_positions(frame, bone) = vec3();
                db.bone_velocities(frame, bone) = vec3();
                db.bone_rotations(frame, bone) = quat();
                db.bone_angular_velocities(frame, bone) = vec3();
            }
        }

        for (int row = 0; row < SonicG1JointCount; ++row) {
            const int bone = row + 2;
            contract[row].source_index = row;
            contract[row].source_bone = bone;
            contract[row].source_parent = parents[bone];
            contract[row].target_index = row;
            contract[row].axis_holden = vec3(1.0f, 0.0f, 0.0f);
            contract[row].static_local_holden = quat();
            contract[row].sign = 1.0f;
            contract[row].zero_offset = 0.0f;
            contract[row].lower = -0.5f;
            contract[row].upper = 0.5f;
            contract[row].source_joint =
                std::string("source_joint_") + std::to_string(row);
            contract[row].target_joint = contract[row].source_joint;
        }
        std::memset(error, 0, sizeof(error));
    }

    void pose(int frame, int row, float angle)
    {
        const sonic_joint_contract_entry& entry = contract[row];
        db.bone_rotations(frame, entry.source_bone) = quat_mul(
            entry.static_local_holden,
            quat_from_angle_axis(angle, entry.axis_holden));
    }

    void off_axis_pose(int frame, int row, float angle)
    {
        db.bone_rotations(frame, contract[row].source_bone) =
            quat_from_angle_axis(angle, vec3(0.0f, 1.0f, 0.0f));
    }

    bool build(sonic_joint_feasibility_certificate& output)
    {
        std::memset(error, 0, sizeof(error));
        return sonic_build_joint_feasibility_certificate(
            output,
            db,
            contract,
            error,
            static_cast<int>(sizeof(error)));
    }
};

static sonic_joint_feasibility_certificate sentinel_certificate()
{
    sonic_joint_feasibility_certificate certificate;
    certificate.raw_safe.resize(3);
    certificate.search_safe.resize(2);
    certificate.raw_safe.set(0xa5U);
    certificate.search_safe.set(0x5aU);
    certificate.frame_count = 71;
    certificate.raw_safe_count = 72;
    certificate.raw_unsafe_count = 73;
    certificate.search_safe_count = 74;
    for (int row = 0; row < SonicG1JointCount; ++row) {
        certificate.joint_limit_violation_count[row] = 100 + row;
    }
    std::memset(certificate.mask_sha256, 'f', 64);
    certificate.mask_sha256[64] = '\0';
    return certificate;
}

static bool same_certificate(
    const sonic_joint_feasibility_certificate& left,
    const sonic_joint_feasibility_certificate& right)
{
    if (left.raw_safe.size != right.raw_safe.size ||
        left.search_safe.size != right.search_safe.size ||
        left.frame_count != right.frame_count ||
        left.raw_safe_count != right.raw_safe_count ||
        left.raw_unsafe_count != right.raw_unsafe_count ||
        left.search_safe_count != right.search_safe_count ||
        std::memcmp(left.mask_sha256, right.mask_sha256, 65) != 0) {
        return false;
    }
    for (int frame = 0; frame < left.raw_safe.size; ++frame) {
        if (left.raw_safe(frame) != right.raw_safe(frame)) return false;
    }
    for (int frame = 0; frame < left.search_safe.size; ++frame) {
        if (left.search_safe(frame) != right.search_safe(frame)) return false;
    }
    for (int row = 0; row < SonicG1JointCount; ++row) {
        if (left.joint_limit_violation_count[row] !=
            right.joint_limit_violation_count[row]) {
            return false;
        }
    }
    return true;
}

static bool lowercase_digest(const char* digest)
{
    if (digest[64] != '\0') return false;
    for (int index = 0; index < 64; ++index) {
        const bool decimal = digest[index] >= '0' && digest[index] <= '9';
        const bool lowercase = digest[index] >= 'a' && digest[index] <= 'f';
        if (!decimal && !lowercase) return false;
    }
    return true;
}

static void test_all_safe_counts_and_digest_are_deterministic()
{
    FeasibilityFixture fixture;
    sonic_joint_feasibility_certificate first;
    sonic_joint_feasibility_certificate second;

    CHECK(fixture.build(first));
    CHECK(fixture.build(second));
    CHECK(first.frame_count == FeasibilityFixture::FrameCount);
    CHECK(first.raw_safe.size == first.frame_count);
    CHECK(first.search_safe.size == first.frame_count);
    CHECK(first.raw_safe_count == first.frame_count);
    CHECK(first.raw_unsafe_count == 0);
    CHECK(first.search_safe_count == first.frame_count);
    CHECK(first.raw_safe_count + first.raw_unsafe_count == first.frame_count);
    for (int frame = 0; frame < first.frame_count; ++frame) {
        CHECK(first.raw_safe(frame) == 1U);
        CHECK(first.search_safe(frame) == 1U);
    }
    for (int row = 0; row < SonicG1JointCount; ++row) {
        CHECK(first.joint_limit_violation_count[row] == 0);
    }
    CHECK(lowercase_digest(first.mask_sha256));
    CHECK(std::strcmp(first.mask_sha256, second.mask_sha256) == 0);
    CHECK(std::strcmp(
        first.mask_sha256,
        "40e7c00389a4782e1b533704aa47e6d3627fc7dc10a4ad8597a8cfa73a6250a2")
        == 0);
}

static void test_limit_failure_marks_one_frame_and_predecessor()
{
    FeasibilityFixture fixture;
    const int row = 7;
    fixture.pose(1, row, 0.5001f);
    sonic_joint_feasibility_certificate certificate;

    CHECK(fixture.build(certificate));
    const unsigned char expected_raw[4] = {1U, 0U, 1U, 1U};
    const unsigned char expected_search[4] = {0U, 0U, 1U, 1U};
    for (int frame = 0; frame < certificate.frame_count; ++frame) {
        CHECK(certificate.raw_safe(frame) == expected_raw[frame]);
        CHECK(certificate.search_safe(frame) == expected_search[frame]);
    }
    CHECK(certificate.raw_safe_count == 3);
    CHECK(certificate.raw_unsafe_count == 1);
    CHECK(certificate.search_safe_count == 2);
    for (int index = 0; index < SonicG1JointCount; ++index) {
        CHECK(certificate.joint_limit_violation_count[index] ==
              (index == row ? 1 : 0));
    }
}

static void test_range_end_successor_clamps_without_crossing()
{
    FeasibilityFixture fixture;
    fixture.pose(2, 8, -0.5001f);
    sonic_joint_feasibility_certificate certificate;

    CHECK(fixture.build(certificate));
    CHECK(certificate.raw_safe(1) == 1U);
    CHECK(certificate.raw_safe(2) == 0U);
    CHECK(certificate.search_safe(1) == 1U);
    CHECK(certificate.search_safe(2) == 0U);
    CHECK(certificate.search_safe(3) == 1U);
}

static void require_transactional_failure(
    FeasibilityFixture& fixture,
    const char* expected_error)
{
    sonic_joint_feasibility_certificate output = sentinel_certificate();
    const sonic_joint_feasibility_certificate before = output;
    CHECK(!fixture.build(output));
    CHECK(std::strstr(fixture.error, expected_error) != NULL);
    CHECK(same_certificate(output, before));
}

static void test_residual_and_malformed_inputs_abort_transactionally()
{
    {
        FeasibilityFixture fixture;
        fixture.off_axis_pose(2, 5, 0.00101f);
        require_transactional_failure(fixture, "frame 2 projection");
        CHECK(std::strstr(fixture.error, "residual") != NULL);
    }
    {
        FeasibilityFixture fixture;
        fixture.db.bone_rotations.resize(
            FeasibilityFixture::FrameCount, G1_BoneCount - 1);
        require_transactional_failure(fixture, "frame 0 projection");
        CHECK(std::strstr(fixture.error, "shape") != NULL);
    }
}

static void test_unsafe_frame_zero_and_zero_search_mask_abort()
{
    {
        FeasibilityFixture fixture;
        fixture.pose(0, 3, 0.5001f);
        require_transactional_failure(fixture, "frame 0 must be raw-safe");
    }
    {
        FeasibilityFixture fixture;
        fixture.pose(1, 1, 0.5001f);
        fixture.pose(2, 2, 0.5001f);
        fixture.pose(3, 3, 0.5001f);
        require_transactional_failure(fixture, "no search-safe frames");
    }
}

static void test_negative_frame_count_aborts_before_encoding()
{
    FeasibilityFixture fixture;
    fixture.db.bone_positions.rows = -1;
    require_transactional_failure(fixture, "negative frame count");
}

static void test_one_mask_byte_changes_the_digest()
{
    FeasibilityFixture first_fixture;
    first_fixture.pose(3, 9, 0.5001f);
    sonic_joint_feasibility_certificate first;
    CHECK(first_fixture.build(first));

    FeasibilityFixture second_fixture;
    second_fixture.pose(2, 10, -0.5001f);
    second_fixture.pose(3, 9, 0.5001f);
    sonic_joint_feasibility_certificate second;
    CHECK(second_fixture.build(second));

    int changed_mask_bytes = 0;
    for (int frame = 0; frame < first.frame_count; ++frame) {
        changed_mask_bytes += first.raw_safe(frame) != second.raw_safe(frame);
        changed_mask_bytes +=
            first.search_safe(frame) != second.search_safe(frame);
    }
    CHECK(changed_mask_bytes == 1);
    CHECK(std::strcmp(first.mask_sha256, second.mask_sha256) != 0);
}

int main()
{
    test_all_safe_counts_and_digest_are_deterministic();
    test_limit_failure_marks_one_frame_and_predecessor();
    test_range_end_successor_clamps_without_crossing();
    test_residual_and_malformed_inputs_abort_transactionally();
    test_unsafe_frame_zero_and_zero_search_mask_abort();
    test_negative_frame_count_aborts_before_encoding();
    test_one_mask_byte_changes_the_digest();
    std::puts("G1 joint feasibility certificate tests passed");
    return 0;
}
