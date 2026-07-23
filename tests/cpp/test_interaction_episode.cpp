#include "interaction_episode.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kTick = 1.0F / 25.0F;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

episode::FrozenAttempt make_attempt(
    const interaction::Pose& pose,
    reach::Hand hand,
    uint64_t generation,
    uint64_t request_id,
    size_t clip) {
    const size_t wrist = hand == reach::Hand::Left
        ? g1_skeleton::LeftWrist
        : g1_skeleton::RightWrist;
    const interaction::WorldPose world = interaction::world_pose(pose);

    episode::FrozenAttempt attempt{};
    attempt.object.generation = generation;
    attempt.object.world = {
        world.positions[wrist] + vec3(0.0F, -0.05F, 0.0F),
        world.rotations[wrist],
    };
    attempt.object.dimensions = vec3(0.08F, 0.10F, 0.08F);
    attempt.grasp.hand_world = {
        world.positions[wrist],
        world.rotations[wrist],
    };
    attempt.grasp.approach_world = vec3(1.0F, 0.0F, 0.0F);
    attempt.grasp.grasp_id = 11U;
    attempt.plan.grasp = attempt.grasp;
    attempt.plan.hand = hand;
    attempt.plan.entry_root_world = {
        pose.positions[g1_skeleton::Simulation],
        pose.rotations[g1_skeleton::Simulation],
    };
    attempt.plan.reach.candidate.clip = clip;
    attempt.plan.reach.rejection = reach::Rejection::None;
    attempt.plan.reach.poses.assign(4U, pose);
    attempt.request_id = request_id;
    return attempt;
}

episode::EpisodeConfig fast_config() {
    episode::EpisodeConfig config{};
    config.entry_position_m = 0.20F;
    config.entry_yaw_radians = 3.14159265F;
    config.entry_speed_mps = 10.0F;
    config.stable_entry_ticks = 1;
    config.bridge_seconds = kTick;
    config.carry_blend_seconds = kTick;
    return config;
}

void advance_until(
    episode::InteractionEpisode& runtime,
    episode::EpisodeState desired,
    uint64_t generation,
    int maximum_ticks = 80) {
    for (int tick = 0; tick < maximum_ticks; ++tick) {
        runtime.update({
            kTick,
            episode::LocomotionCommand{},
            generation,
            false,
            false,
        });
        if (runtime.state() == desired) return;
    }
    throw std::runtime_error("episode did not reach requested state");
}

void test_freeze_attach_and_selected_carry_hand(reach::Hand hand) {
    const std::filesystem::path pack("build/g1-episode");
    episode::InteractionEpisode runtime(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        fast_config());
    const auto original = make_attempt(
        runtime.output().pose, hand, 7U, 100U, 12U);
    auto replacement = original;
    replacement.plan.reach.candidate.clip = 99U;

    require(runtime.commit(original), "F did not commit a valid plan");
    require(
        runtime.state() == episode::EpisodeState::Approach,
        "committed plan did not enter approach");
    require(
        !runtime.commit(replacement),
        "repeated F replaced an active attempt");
    require(
        runtime.attempt()->plan.reach.candidate.clip == 12U,
        "active plan was not immutable");

    advance_until(runtime, episode::EpisodeState::Reach, 7U);
    advance_until(runtime, episode::EpisodeState::Carry, 7U);
    require(runtime.output().attached, "object did not attach at contact");
    const interaction::Hand expected = hand == reach::Hand::Left
        ? interaction::Hand::Left
        : interaction::Hand::Right;
    require(
        runtime.output().selected_hand == expected,
        "selected reach hand was not propagated into carry");
}

void test_generation_change_and_cancel_fail_before_contact() {
    const std::filesystem::path pack("build/g1-episode");
    episode::InteractionEpisode changed(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        fast_config());
    require(
        changed.commit(make_attempt(
            changed.output().pose, reach::Hand::Left, 4U, 10U, 1U)),
        "generation fixture did not commit");
    changed.update({
        kTick, episode::LocomotionCommand{}, 5U, false, false});
    require(
        changed.state() == episode::EpisodeState::Failed,
        "object generation change did not fail the attempt");

    episode::InteractionEpisode cancelled(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        fast_config());
    require(
        cancelled.commit(make_attempt(
            cancelled.output().pose, reach::Hand::Right, 9U, 11U, 2U)),
        "cancel fixture did not commit");
    cancelled.update({
        kTick, episode::LocomotionCommand{}, 9U, true, false});
    require(
        cancelled.state() == episode::EpisodeState::FreeLocomotion,
        "cancel before contact did not return locomotion control");
}

void test_contact_rejection_timeout_and_reset() {
    const std::filesystem::path pack("build/g1-episode");
    episode::InteractionEpisode rejected(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        fast_config());
    auto bad_contact = make_attempt(
        rejected.output().pose, reach::Hand::Left, 12U, 20U, 3U);
    bad_contact.grasp.hand_world.position.x += 0.20F;
    bad_contact.plan.grasp = bad_contact.grasp;
    require(rejected.commit(bad_contact), "bad-contact fixture did not commit");
    advance_until(rejected, episode::EpisodeState::Failed, 12U);
    require(
        !rejected.output().attached,
        "geometrically invalid contact attached the object");
    rejected.update({
        kTick, episode::LocomotionCommand{}, 12U, false, true});
    require(
        rejected.state() == episode::EpisodeState::FreeLocomotion &&
            !rejected.attempt().has_value(),
        "reset did not clear failed attempt");

    episode::EpisodeConfig timeout_config = fast_config();
    timeout_config.approach_timeout_seconds = kTick;
    episode::InteractionEpisode timed_out(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        timeout_config);
    auto unreachable = make_attempt(
        timed_out.output().pose, reach::Hand::Right, 13U, 21U, 4U);
    unreachable.plan.entry_root_world.position.x += 20.0F;
    require(timed_out.commit(unreachable), "timeout fixture did not commit");
    timed_out.update({
        kTick, episode::LocomotionCommand{}, 13U, false, false});
    require(
        timed_out.state() == episode::EpisodeState::Failed,
        "approach timeout did not fail");
}

}  // namespace

int main() {
    try {
        test_freeze_attach_and_selected_carry_hand(reach::Hand::Left);
        test_freeze_attach_and_selected_carry_hand(reach::Hand::Right);
        test_generation_change_and_cancel_fail_before_contact();
        test_contact_rejection_timeout_and_reset();
        std::cout << "interaction episode PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "interaction episode FAILED: " << error.what() << '\n';
        return 1;
    }
}
