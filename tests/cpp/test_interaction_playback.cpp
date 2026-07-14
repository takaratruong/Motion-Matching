#include "interaction_playback.h"
#include "tests/cpp/interaction_runtime_fixture.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {

constexpr float kTolerance = 1.0e-5F;

bool near(float left, float right, float tolerance = kTolerance) {
    return std::abs(left - right) <= tolerance;
}

bool near(vec3 left, vec3 right, float tolerance = kTolerance) {
    return near(left.x, right.x, tolerance) &&
           near(left.y, right.y, tolerance) &&
           near(left.z, right.z, tolerance);
}

bool near(quat left, quat right, float tolerance = kTolerance) {
    return quat_angle_between(left, right) <= tolerance;
}

template<class Exception, class Function>
bool throws_as(Function&& function) {
    try {
        function();
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

struct PlaybackCase {
    interaction::RuntimeFixture fixture;
    interaction::MatchCandidate candidate;
    interaction::Pose current;
};

PlaybackCase make_playback_case() {
    using namespace interaction;
    PlaybackCase value{};
    value.fixture = make_runtime_fixture();
    const MatchResult match = select_whole_clip(
        match_input_for(value.fixture), MatchConfig{});
    assert(match.accepted);
    value.candidate = match.candidate;
    value.current = value.fixture.locomotion.pose;
    return value;
}

interaction::Pose expected_source_pose(
    const interaction::Database& database,
    float source_frame) {
    using namespace interaction;
    const int32_t left = static_cast<int32_t>(std::floor(source_frame));
    const int32_t right = std::min(
        left + 1,
        database.range_stops.at(1) - 1);
    return interpolate_pose(
        pose_at_frame(database, left),
        pose_at_frame(database, right),
        source_frame - static_cast<float>(left));
}

float correction_weight(
    const interaction::MatchCandidate& candidate,
    float source_frame) {
    if (source_frame <= static_cast<float>(candidate.entry_frame)) {
        return 1.0F;
    }
    if (source_frame >= static_cast<float>(candidate.contact_frame)) {
        return 0.0F;
    }
    const float alpha =
        (source_frame - static_cast<float>(candidate.entry_frame)) /
        static_cast<float>(
            candidate.contact_frame - candidate.entry_frame);
    const float smoothstep = alpha * alpha * (3.0F - 2.0F * alpha);
    return 1.0F - smoothstep;
}

interaction::Pose current_for_candidate(
    const interaction::Database& database,
    const interaction::MatchCandidate& candidate) {
    using namespace interaction;
    Pose current = pose_at_frame(database, candidate.entry_frame);
    const size_t root = g1_skeleton::Simulation;
    const Transform mapped = compose(
        candidate.scene_from_source,
        Transform{current.positions[root], current.rotations[root]});
    current.positions[root] = mapped.position + candidate.entry_root_offset;
    const quat yaw = quat_from_angle_axis(
        candidate.entry_yaw_offset, vec3(0.0F, 1.0F, 0.0F));
    current.rotations[root] = quat_normalize(
        quat_mul(yaw, mapped.rotation));
    return current;
}

void assert_corrected_sample(
    const interaction::SequentialPlayer& player,
    const interaction::Database& database,
    const interaction::MatchCandidate& candidate,
    float source_frame) {
    using namespace interaction;
    const Pose source = expected_source_pose(database, source_frame);
    const Pose sampled = player.sample();
    const WorldPose source_world = world_pose(source);
    const WorldPose sampled_world = world_pose(sampled);
    const size_t root = g1_skeleton::Simulation;
    const float weight = correction_weight(candidate, source_frame);
    const Transform mapped_root = compose(
        candidate.scene_from_source,
        Transform{
            source_world.positions[root],
            source_world.rotations[root],
        });
    const Transform mapped_hand = compose(
        candidate.scene_from_source,
        Transform{
            source_world.positions[kRightHandBone],
            source_world.rotations[kRightHandBone],
        });
    const quat yaw = quat_from_angle_axis(
        weight * candidate.entry_yaw_offset,
        vec3(0.0F, 1.0F, 0.0F));
    const vec3 corrected_root =
        mapped_root.position + weight * candidate.entry_root_offset;
    const vec3 corrected_hand = corrected_root + quat_mul_vec3(
        yaw, mapped_hand.position - mapped_root.position);
    const quat corrected_root_rotation = quat_normalize(
        quat_mul(yaw, mapped_root.rotation));

    assert(near(sampled.positions[root], corrected_root));
    assert(near(sampled.rotations[root], corrected_root_rotation));
    assert(near(sampled_world.positions[kRightHandBone], corrected_hand));
}

void test_frozen_public_interface_and_unstarted_state() {
    using namespace interaction;
    static_assert(std::is_constructible_v<SequentialPlayer, const Database&>);
    static_assert(!std::is_convertible_v<const Database&, SequentialPlayer>);
    static_assert(std::is_same_v<
        decltype(&SequentialPlayer::start),
        void (SequentialPlayer::*)(const MatchCandidate&, const Pose&, float)>);
    static_assert(std::is_same_v<
        decltype(&SequentialPlayer::advance),
        void (SequentialPlayer::*)(float)>);
    static_assert(std::is_same_v<
        decltype(&SequentialPlayer::sample),
        Pose (SequentialPlayer::*)() const>);
    static_assert(std::is_same_v<
        decltype(&SequentialPlayer::phase),
        Phase (SequentialPlayer::*)() const>);
    static_assert(std::is_same_v<
        decltype(&SequentialPlayer::frame),
        int32_t (SequentialPlayer::*)() const>);
    static_assert(std::is_same_v<
        decltype(&SequentialPlayer::at_contact),
        bool (SequentialPlayer::*)() const>);
    static_assert(std::is_same_v<
        decltype(&SequentialPlayer::at_hold),
        bool (SequentialPlayer::*)() const>);
    static_assert(std::is_same_v<
        decltype(&SequentialPlayer::finished),
        bool (SequentialPlayer::*)() const>);
    static_assert(std::is_same_v<
        decltype(&SequentialPlayer::elapsed_seconds),
        float (SequentialPlayer::*)() const>);
    static_assert(std::is_same_v<
        decltype(&SequentialPlayer::entry_root_correction),
        vec3 (SequentialPlayer::*)() const>);

    const RuntimeFixture fixture = make_runtime_fixture();
    const SequentialPlayer player(fixture.database);
    assert(player.frame() == -1);
    assert(!player.at_contact());
    assert(!player.at_hold());
    assert(player.finished());
    assert(near(player.elapsed_seconds(), 0.0F));
    assert(near(player.entry_root_correction(), vec3()));
    assert(throws_as<std::logic_error>([&] { (void)player.sample(); }));
    assert(throws_as<std::logic_error>([&] { (void)player.phase(); }));
}

void test_canonical_clock_and_fractional_interpolation() {
    using namespace interaction;
    PlaybackCase value = make_playback_case();

    SequentialPlayer fractional(value.fixture.database);
    fractional.start(value.candidate, value.current);
    fractional.advance(0.02F);
    assert(fractional.frame() == value.candidate.entry_frame);
    assert(fractional.phase() == Phase::Reach);
    const Pose expected = interpolate_pose(
        pose_at_frame(value.fixture.database, value.candidate.entry_frame),
        pose_at_frame(
            value.fixture.database, value.candidate.entry_frame + 1),
        0.5F);
    const Pose actual = fractional.sample();
    assert(near(
        actual.positions[kRightHandBone],
        expected.positions[kRightHandBone]));
    assert(near(
        actual.velocities[kRightHandBone],
        expected.velocities[kRightHandBone]));

    SequentialPlayer sixty_hz(value.fixture.database);
    sixty_hz.start(value.candidate, value.current);
    for (int update = 0; update < 60; ++update) {
        sixty_hz.advance(1.0F / 60.0F);
    }
    assert(near(sixty_hz.elapsed_seconds(), 1.0F));
    assert(sixty_hz.frame() == value.candidate.entry_frame + 25);

    SequentialPlayer one_step(value.fixture.database);
    one_step.start(value.candidate, value.current);
    one_step.advance(1.0F);
    assert(one_step.frame() == sixty_hz.frame());
    const Pose sixty_hz_pose = sixty_hz.sample();
    const Pose one_step_pose = one_step.sample();
    assert(near(
        sixty_hz_pose.positions[kRightHandBone],
        one_step_pose.positions[kRightHandBone]));
    assert(near(
        sixty_hz_pose.positions[g1_skeleton::Simulation],
        one_step_pose.positions[g1_skeleton::Simulation]));
}

void test_positive_tiny_updates_accumulate_without_quantization() {
    using namespace interaction;
    PlaybackCase value = make_playback_case();
    SequentialPlayer partitioned(value.fixture.database);
    SequentialPlayer summed(value.fixture.database);
    partitioned.start(value.candidate, value.current);
    summed.start(value.candidate, value.current);

    constexpr int kUpdates = 1000;
    constexpr float kTinyDt = 1.0e-7F;
    for (int update = 0; update < kUpdates; ++update) {
        partitioned.advance(kTinyDt);
    }
    summed.advance(kTinyDt * static_cast<float>(kUpdates));

    assert(near(
        partitioned.elapsed_seconds(),
        summed.elapsed_seconds(),
        1.0e-6F));
    assert(near(
        partitioned.sample().positions[kRightHandBone],
        summed.sample().positions[kRightHandBone],
        1.0e-6F));
}

void test_clock_is_equivalent_across_update_partitions_and_speeds() {
    using namespace interaction;
    PlaybackCase value = make_playback_case();
    constexpr float kDuration = 0.37F;
    constexpr std::array<int, 3> kUpdateCounts = {30, 60, 120};
    constexpr std::array<float, 3> kSpeeds = {0.85F, 1.0F, 1.15F};

    for (float speed : kSpeeds) {
        SequentialPlayer reference(value.fixture.database);
        reference.start(value.candidate, value.current, speed);
        reference.advance(kDuration);
        const Pose expected = reference.sample();

        for (int update_count : kUpdateCounts) {
            SequentialPlayer partitioned(value.fixture.database);
            partitioned.start(value.candidate, value.current, speed);
            const float dt =
                kDuration / static_cast<float>(update_count);
            for (int update = 0; update < update_count; ++update) {
                partitioned.advance(dt);
            }
            assert(partitioned.frame() == reference.frame());
            assert(near(
                partitioned.elapsed_seconds(),
                reference.elapsed_seconds(),
                1.0e-5F));
            const Pose actual = partitioned.sample();
            assert(near(
                actual.positions[g1_skeleton::Simulation],
                expected.positions[g1_skeleton::Simulation],
                1.0e-5F));
            assert(near(
                actual.positions[kRightHandBone],
                expected.positions[kRightHandBone],
                1.0e-5F));
        }
    }
}

void test_speed_endpoints_and_rejection() {
    using namespace interaction;
    PlaybackCase value = make_playback_case();

    SequentialPlayer slow(value.fixture.database);
    slow.start(value.candidate, value.current, 0.85F);
    slow.advance(0.40F);
    assert(slow.frame() == value.candidate.entry_frame + 8);
    const Pose slow_expected = expected_source_pose(
        value.fixture.database,
        static_cast<float>(value.candidate.entry_frame) + 8.5F);
    assert(near(
        slow.sample().positions[kRightHandBone],
        slow_expected.positions[kRightHandBone]));

    SequentialPlayer fast(value.fixture.database);
    fast.start(value.candidate, value.current, 1.15F);
    fast.advance(0.40F);
    assert(fast.frame() == value.candidate.entry_frame + 11);
    const Pose fast_expected = expected_source_pose(
        value.fixture.database,
        static_cast<float>(value.candidate.entry_frame) + 11.5F);
    assert(near(
        fast.sample().positions[kRightHandBone],
        fast_expected.positions[kRightHandBone]));

    const std::array<float, 4> invalid = {
        0.849F,
        1.151F,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    for (float speed : invalid) {
        SequentialPlayer player(value.fixture.database);
        assert(throws_as<std::invalid_argument>([&] {
            player.start(value.candidate, value.current, speed);
        }));
        assert(player.finished());
        assert(player.frame() == -1);
    }
}

void test_scene_mapping_and_decaying_entry_correction() {
    using namespace interaction;
    PlaybackCase value = make_playback_case();
    value.candidate.scene_from_source = {
        vec3(1.25F, 0.20F, -0.50F),
        quat_from_angle_axis(0.30F, vec3(0.0F, 1.0F, 0.0F)),
    };
    value.candidate.entry_root_offset = vec3(0.12F, 0.0F, -0.08F);
    value.candidate.entry_yaw_offset = 0.20F;
    value.current = current_for_candidate(
        value.fixture.database, value.candidate);

    SequentialPlayer player(value.fixture.database);
    player.start(value.candidate, value.current);
    assert(near(
        player.entry_root_correction(),
        value.candidate.entry_root_offset));
    assert_corrected_sample(
        player,
        value.fixture.database,
        value.candidate,
        static_cast<float>(value.candidate.entry_frame));
    const Pose entry_sample = player.sample();
    assert(near(
        entry_sample.positions[g1_skeleton::Simulation].x,
        value.current.positions[g1_skeleton::Simulation].x));
    assert(near(
        entry_sample.positions[g1_skeleton::Simulation].z,
        value.current.positions[g1_skeleton::Simulation].z));
    assert(near(
        entry_sample.rotations[g1_skeleton::Simulation],
        value.current.rotations[g1_skeleton::Simulation]));

    const float half_frame = 0.5F * static_cast<float>(
        value.candidate.entry_frame + value.candidate.contact_frame);
    player.advance(
        (half_frame - static_cast<float>(value.candidate.entry_frame)) /
        25.0F);
    assert(near(
        player.entry_root_correction(),
        0.5F * value.candidate.entry_root_offset));
    assert_corrected_sample(
        player, value.fixture.database, value.candidate, half_frame);

    player.advance(
        (static_cast<float>(value.candidate.contact_frame) - half_frame) /
        25.0F);
    assert(player.frame() == value.candidate.contact_frame);
    assert(near(player.entry_root_correction(), vec3()));
    assert_corrected_sample(
        player,
        value.fixture.database,
        value.candidate,
        static_cast<float>(value.candidate.contact_frame));

    player.advance(0.20F);
    assert(near(player.entry_root_correction(), vec3()));
}

void test_phase_and_event_predicates_preserve_order() {
    using namespace interaction;
    PlaybackCase value = make_playback_case();
    SequentialPlayer player(value.fixture.database);
    player.start(value.candidate, value.current);

    assert(player.phase() == Phase::Reach);
    assert(!player.at_contact());
    assert(!player.at_hold());

    player.advance(
        (static_cast<float>(value.candidate.contact_frame) - 0.25F -
         static_cast<float>(value.candidate.entry_frame)) /
        25.0F);
    assert(player.frame() == value.candidate.contact_frame - 1);
    assert(player.phase() == Phase::Reach);
    assert(!player.at_contact());
    assert(!player.at_hold());

    player.advance(0.01F);
    assert(player.frame() == value.candidate.contact_frame);
    assert(player.phase() == Phase::Contact);
    assert(player.at_contact());
    assert(!player.at_hold());

    player.advance(static_cast<float>(
        value.candidate.lift_frame - value.candidate.contact_frame) / 25.0F);
    assert(player.phase() == Phase::Lift);
    assert(player.at_contact());
    assert(!player.at_hold());

    player.advance(static_cast<float>(
        value.candidate.hold_frame - value.candidate.lift_frame) / 25.0F);
    assert(player.phase() == Phase::Hold);
    assert(player.at_contact());
    assert(player.at_hold());
}

void test_final_frame_is_clamped_and_repeatable() {
    using namespace interaction;
    PlaybackCase value = make_playback_case();
    SequentialPlayer player(value.fixture.database);
    player.start(value.candidate, value.current);
    const int32_t final_frame = value.fixture.database.range_stops.at(
        static_cast<size_t>(value.candidate.clip)) - 1;

    player.advance(static_cast<float>(
        final_frame - value.candidate.entry_frame) / 25.0F);
    assert(player.finished());
    assert(player.frame() == final_frame);
    assert(player.phase() == Phase::Hold);
    assert(player.at_contact());
    assert(player.at_hold());
    const Pose final_sample = player.sample();
    const Pose expected = pose_at_frame(value.fixture.database, final_frame);
    assert(near(
        final_sample.positions[g1_skeleton::Simulation],
        expected.positions[g1_skeleton::Simulation]));
    assert(near(
        final_sample.positions[kRightHandBone],
        expected.positions[kRightHandBone]));

    player.advance(1000000.0F);
    assert(player.finished());
    assert(player.frame() == final_frame);
    const Pose repeated = player.sample();
    assert(near(
        repeated.positions[g1_skeleton::Simulation],
        final_sample.positions[g1_skeleton::Simulation]));
    assert(near(
        repeated.positions[kRightHandBone],
        final_sample.positions[kRightHandBone]));
}

void test_rejects_invalid_candidates_ranges_current_and_dt() {
    using namespace interaction;
    PlaybackCase value = make_playback_case();
    const MatchCandidate valid = value.candidate;

    std::array<MatchCandidate, 9> invalid_candidates{};
    invalid_candidates.fill(valid);
    invalid_candidates[0].clip = -1;
    invalid_candidates[1].clip = 2;
    invalid_candidates[2].entry_frame = 74;
    invalid_candidates[3].contact_frame = valid.entry_frame;
    invalid_candidates[4].lift_frame = valid.contact_frame;
    invalid_candidates[5].hold_frame = valid.lift_frame;
    invalid_candidates[6].hold_frame = 150;
    invalid_candidates[7].contact_frame = valid.contact_frame - 1;
    invalid_candidates[8].entry_yaw_offset =
        std::numeric_limits<float>::quiet_NaN();
    for (const MatchCandidate& candidate : invalid_candidates) {
        SequentialPlayer player(value.fixture.database);
        assert(throws_as<std::invalid_argument>([&] {
            player.start(candidate, value.current);
        }));
        assert(player.finished());
    }

    MatchCandidate nonfinite = valid;
    nonfinite.scene_from_source.position.x =
        std::numeric_limits<float>::infinity();
    SequentialPlayer nonfinite_player(value.fixture.database);
    assert(throws_as<std::invalid_argument>([&] {
        nonfinite_player.start(nonfinite, value.current);
    }));

    Pose invalid_current = value.current;
    invalid_current.positions[g1_skeleton::Simulation].x =
        std::numeric_limits<float>::quiet_NaN();
    SequentialPlayer current_player(value.fixture.database);
    assert(throws_as<std::invalid_argument>([&] {
        current_player.start(valid, invalid_current);
    }));

    Pose discontinuous = value.current;
    discontinuous.positions[g1_skeleton::Simulation].x += 0.01F;
    SequentialPlayer discontinuous_player(value.fixture.database);
    assert(throws_as<std::invalid_argument>([&] {
        discontinuous_player.start(valid, discontinuous);
    }));

    Database invalid_range = value.fixture.database;
    invalid_range.range_stops.at(1) =
        static_cast<int32_t>(invalid_range.frame_count) + 1;
    SequentialPlayer range_player(invalid_range);
    assert(throws_as<std::invalid_argument>([&] {
        range_player.start(valid, value.current);
    }));

    Database missing_range = value.fixture.database;
    missing_range.range_stops.clear();
    SequentialPlayer missing_range_player(missing_range);
    assert(throws_as<std::invalid_argument>([&] {
        missing_range_player.start(valid, value.current);
    }));

    SequentialPlayer player(value.fixture.database);
    player.start(valid, value.current);
    const Pose before = player.sample();
    const std::array<float, 3> invalid_dt = {
        -0.001F,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    for (float dt : invalid_dt) {
        assert(throws_as<std::invalid_argument>([&] { player.advance(dt); }));
        assert(player.frame() == valid.entry_frame);
        assert(near(player.elapsed_seconds(), 0.0F));
        assert(near(
            player.sample().positions[kRightHandBone],
            before.positions[kRightHandBone]));
    }
    player.advance(0.0F);
    assert(player.frame() == valid.entry_frame);
}

}  // namespace

int main() {
    test_frozen_public_interface_and_unstarted_state();
    test_canonical_clock_and_fractional_interpolation();
    test_positive_tiny_updates_accumulate_without_quantization();
    test_clock_is_equivalent_across_update_partitions_and_speeds();
    test_speed_endpoints_and_rejection();
    test_scene_mapping_and_decaying_entry_correction();
    test_phase_and_event_predicates_preserve_order();
    test_final_frame_is_clamped_and_repeatable();
    test_rejects_invalid_candidates_ranges_current_and_dt();
    return 0;
}
